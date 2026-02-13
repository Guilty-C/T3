
import os
import numpy as np
import pandas as pd
import torch
import sys

# P0: Add task3 to sys.path to allow imports from task3 folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "task3"))
sys.path.insert(0, os.path.dirname(__file__)) # Add current dir

try:
    from safeopt_gp import SafeOptGP
except ImportError:
    # If not found directly, try algorithms package or local
    try:
        from algorithms.safeopt_gp import SafeOptGP
    except ImportError:
        pass # Will handle later if needed

try:
    from lagrangian_ppo import LagrangianPPO
except ImportError:
    pass

try:
    from rucb_plus_impl.rucb_plus import RUCBPlus, RUCBPlusConfig
except ImportError:
    try:
        from algorithms.rucb_plus_impl.rucb_plus import RUCBPlus, RUCBPlusConfig
    except ImportError:
        try:
            from rucb_plus import RUCBPlus, RUCBPlusConfig
        except ImportError:
            pass

import inspect
import pprint

# --- P0: Runtime Import Verification ---
# Log these immediately to verify what code is actually running
debug_msg = []
debug_msg.append("=== P0 Runtime Import Verification ===")
try:
    debug_msg.append(f"RUCBPlus module: {RUCBPlus.__module__}")
except NameError:
    debug_msg.append("RUCBPlus module: Not imported (NameError)")

try:
    debug_msg.append(f"RUCBPlus file: {inspect.getfile(RUCBPlus)}")
except Exception as e:
    debug_msg.append(f"RUCBPlus file: Error getting file: {e}")

# Best Fixed Arm Oracle is defined in this file, so we track this file
debug_msg.append(f"best_fixed_arm_oracle location: Defined in {os.path.abspath(__file__)}")

debug_msg.append("sys.path[:20]:")
for p in sys.path[:20]:
    debug_msg.append(f"  {p}")

# Write to alignment_check.txt (Overwrite mode for the start of a new run sequence)
# Note: We assume OUTPUT_DIR will be defined shortly, but we need to resolve it now or later.
# We'll write it to the standard location.
if os.environ.get("TASK3_ALIGN_LOG", "0") == "1":
    align_txt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fig01_regret_vs_time", "alignment_check.txt")
    os.makedirs(os.path.dirname(align_txt_path), exist_ok=True)

    with open(align_txt_path, "w") as f:
        for line in debug_msg:
            print(line)
            f.write(line + "\n")
    print(f"Saved runtime verification to {align_txt_path}")
else:
    # Just print to stdout if not logging to file
    for line in debug_msg:
        print(line)

try:
    print(f"DEBUG: RUCBPlus module: {RUCBPlus.__module__}")
    print(f"DEBUG: RUCBPlus file: {inspect.getsourcefile(RUCBPlus)}")
    print(f"DEBUG: RUCBPlusConfig file: {inspect.getsourcefile(RUCBPlusConfig)}")
except NameError:
    print("DEBUG: RUCBPlus not defined (Skipping debug prints)")

import time
import traceback

# --- Configuration ---
# Allow environment variables to override defaults
V = float(os.environ.get("TASK3_V", "800.0"))
w = int(os.environ.get("TASK3_W", "512"))
g = int(os.environ.get("TASK3_G", "50"))
HORIZON = int(os.environ.get("TASK3_HORIZON", "2000"))
SEEDS = [0, 1, 2]
OUTPUT_DIR = os.environ.get(
    "TASK3_OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fig01_regret_vs_time", "inputs")
)

print(f"Experimental Configuration:")
print(f"  V={V}, w={w}, g={g}, T={HORIZON}")
print(f"  SEEDS={SEEDS}")
print(f"  OUTPUT_DIR={OUTPUT_DIR}")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_candidates(n_grid=10, bounds=[(0.1, 1.0), (0.1, 1.0)]):
    """
    Global function to generate discrete action candidates.
    Ensures all algorithms and oracles use the EXACT SAME action space.
    """
    x1 = np.linspace(bounds[0][0], bounds[0][1], n_grid)
    x2 = np.linspace(bounds[1][0], bounds[1][1], n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    return np.column_stack([X1.ravel(), X2.ravel()])

def compute_lyapunov_metrics(V_val, utility_reward, q_prev_before, q_curr, penalty_scale=1.0):
    """
    Computes Lyapunov metrics based on unified math definition.
    Returns: drift_val, penalty_term, lyapunov_reward, scaled_reward
    """
    drift_val = q_prev_before - q_curr
    penalty_term = penalty_scale * q_prev_before * drift_val
    lyapunov_reward = V_val * utility_reward + penalty_term
    scaled_reward = lyapunov_reward / V_val
    return drift_val, penalty_term, lyapunov_reward, scaled_reward

def check_file_validity(filepath, expected_horizon):
    """
    Checks if file exists and has the correct number of rows (Horizon).
    Returns True if valid, False otherwise.
    """
    if os.environ.get("FORCE_RUN", "0") == "1":
        return False
        
    if not os.path.exists(filepath):
        return False
        
    try:
        df = pd.read_csv(filepath)
        # Check if t max matches horizon
        if "t" in df.columns:
            if df["t"].max() == expected_horizon:
                return True
        # Fallback: check length
        if len(df) >= expected_horizon:
             return True
    except Exception:
        pass
        
    return False

def verify_environment_consistency():
    """
    Verifies that the environment is deterministic given a seed and action sequence.
    """
    print("Verifying Environment Consistency...")
    seed = 12345
    actions = [[0.5, 0.5]] * 100
    
    env1 = SimulatedEnvironment(seed)
    rewards1 = []
    for a in actions:
        _, r, _, _, _ = env1.step(a)
        rewards1.append(r)
        
    env2 = SimulatedEnvironment(seed)
    rewards2 = []
    for a in actions:
        _, r, _, _, _ = env2.step(a)
        rewards2.append(r)
        
    assert np.allclose(rewards1, rewards2), "Environment is not deterministic!"
    print("[PASS] Environment Consistency Check Passed.")

class SimulatedEnvironment:

    """
    Simulates a 6G networking environment with semantic communication.
    State: [q_semantic, q_energy, channel_quality, interference]
    Action: [bandwidth_idx, power_idx] (discrete for PPO, continuous for SafeOpt mapped to nearest)
    """
    def __init__(self, seed, V_val=100.0, window=128, grid=10):
        # Use local RNG to decouple environment randomness from global state
        self.rng = np.random.default_rng(seed)
        self.V = V_val
        self.w = window
        self.g = grid # grid density for candidates
        self.q_semantic = 0.0
        self.q_energy = 0.0
        self.channel_quality = 0.5
        self.time_step = 0
        
        # Constraints
        self.q_sem_max = 20.0
        self.q_ene_max = 20.0

        # Scenario Handling
        self.scenario = os.environ.get("TASK3_SCENARIO", "default")
        
        # Calibrate noise
        self.noise_std = float(os.environ.get("TASK3_ENV_NOISE_STD", "0.05"))
        
        if self.scenario == "bandit_stationary_pure":
            # self.noise_std is already set by env var above (default 0.05)
            self.channel_quality = 0.5 # Fixed
            
        # Calculate mu_star (Theoretical Max for Stationary Case)
        # MUST use the EXACT SAME candidates as the algorithms
        if self.scenario == "bandit_stationary_pure":
            self.mu_star = self._calculate_mu_star()
        else:
            self.mu_star = 0.0

    def _calculate_mu_star(self):
        # Use the global build_candidates to ensure consistency
        candidates = build_candidates(n_grid=self.g)
        
        max_u = -float('inf')
        
        for action in candidates:
            # Expected reward for this action
            u = self.get_expected_reward(action)
            if u > max_u:
                max_u = u
        return max_u

    def get_expected_reward(self, action):
        """
        Pure function for expected reward of an action in stationary setting.
        Independent of queue state.
        """
        bitrate = np.clip(action[0], 0.1, 1.0)
        p = np.clip(action[1], 0.1, 1.0)
        
        # In bandit_stationary_pure, we define reward as strictly:
        # (1 + 4*bw) - 0.1*p
        # No queue violation penalties (as requested by "state 不包含...影响 reward")
        
        return (1.0 + 4.0 * bitrate) - 0.1 * p

    def step(self, action):

        """
        action: numpy array or list [bandwidth, power] normalized to [0, 1]
        """
        self.time_step += 1
        
        # Unpack action (assume normalized [0, 1])
        # Interpret action[0] as Bitrate (Quality)
        bitrate = np.clip(action[0], 0.1, 1.0)
        p = np.clip(action[1], 0.1, 1.0)
        
        if self.scenario == "bandit_stationary_pure":
            # --- PURE BANDIT MODE ---
            # 1. State does NOT evolve (or is irrelevant)
            # 2. Reward depends ONLY on action + noise
            
            # Expected mean
            mu = self.get_expected_reward(action)
            
            # Add noise
            reward = mu + self.rng.normal(0, self.noise_std)
            
            # Cost/Violation is not applicable in pure reward maximization, 
            # but we can log them if needed. 
            # User said "禁止队列 q_semantic 参与 reward/violation 演化"
            # So we effectively freeze or ignore queues.
            
            self.q_semantic = 0.0
            self.q_energy = 0.0
            
            # Compute Cost for Constrained Bandit (p <= h)
            # h is fixed at 0.5
            cost = 1.0 if p > self.channel_quality else 0.0
            violation = int(cost)
            
            # Info for logging
            state = np.array([0.0, 0.0, 0.5, 0.0])
            done = False
            
            info = {
            "mos": 1.0 + 4.0 * bitrate, # Raw MOS
            "swer": 0.0,
            "q_semantic": 0.0,
            "q_energy": 0.0,
            "channel_quality": 0.5,
            "snr_db": 10.0,
            "noise_std": self.noise_std,
            "energy_cum": p,
            "violation_flag": violation,
            "V": self.V,
            "window": self.w,
            "grid_density": self.g,
            "mu_star": self.mu_star,
            "mu_choice": mu,
            "scenario": self.scenario,
            "semantic_weight": 0.0 # Placeholder for stationary
        }
            
            return state, reward, cost, done, info

        # --- DEFAULT SCENARIO (Queue Dynamics) ---
        
        # Simulate channel dynamics (random walk)
        self.channel_quality += self.rng.normal(0, self.noise_std)
        self.channel_quality = np.clip(self.channel_quality, 0.1, 1.0)

        snr_db = 20.0 * float(self.channel_quality)
        snr_lin = float(10.0 ** (snr_db / 10.0))
        per_k = float(os.environ.get("TASK3_PER_K", "0.05"))
        per = float(np.exp(-per_k * snr_lin))
        
        # Metrics calculation
        # Energy consumption depends on Power
        energy = p * 1.0 # arbitrary unit
        
        # Queue dynamics
        # Arrival depends on Bitrate (Traffic Load)
        # High Bitrate -> High Arrival
        arrival = bitrate * 2.0 * self.rng.uniform(0.8, 1.2)

        semantic_mode = os.environ.get("TASK3_SEMANTIC_WEIGHT_MODE", "beta").strip().lower()
        if semantic_mode == "mix":
            s_t = 0.2 if float(self.rng.uniform()) < 0.5 else 0.8
        else:
            s_t = float(self.rng.beta(2.0, 2.0))
        
        # Service depends on Power and Channel
        # High Power + Good Channel -> High Service
        service = p * 2.5 * self.channel_quality
        
        self.q_semantic = max(0, self.q_semantic + arrival - service)
        
        energy_harvest = self.rng.uniform(0, 0.5)
            
        self.q_energy = max(0, self.q_energy + energy_harvest - energy) # Battery dynamic
        
        # Reward: V * QoE - Energy_Queue_Drift (Lyapunov-like)
        # Simplified reward for this simulation
        
        # Cost/Constraints
        violation = 0
        if self.q_semantic > self.q_sem_max:
            violation = 1

        bitrate_factor = 0.6 + 0.4 * float((bitrate - 0.1) / 0.9)
        swer = float(np.clip(per * bitrate_factor, 0.0, 1.0))

        mos_base = 5.0 * (1.0 - swer)
        if violation:
            mos_base = min(mos_base, 1.0)
        mos = float(np.clip(mos_base + self.rng.normal(0, 0.05), 1.0, 5.0))

        reward = mos - 0.1 * energy
        
        cost = 1.0 if violation else 0.0
        
        state = np.array([self.q_semantic, self.q_energy, self.channel_quality, 0.0])
        
        done = False
        
        # Expected Utility of the Chosen Action (mu_choice) for default scenario
        # Just instantaneous utility approx
        mu_choice = (1.0 + 4.0 * bitrate) - 0.1 * p
        
        info = {
            "mos": mos,
            "swer": swer,
            "q_semantic": self.q_semantic,
            "q_energy": self.q_energy,
            "channel_quality": self.channel_quality,
            "snr_db": snr_db,
            "per": per,
            "noise_std": self.noise_std,
            "energy_cum": energy, # simplified
            "violation_flag": violation,
            "V": self.V,
            "window": self.w,
            "grid_density": self.g,
            "mu_star": self.mu_star,
            "mu_choice": mu_choice,
            "scenario": self.scenario,
            "semantic_weight": s_t
        }
        
        return state, reward, cost, done, info

def run_safeopt_gp(seed, V_val=800.0, beta=2.0, n_grid=10):
    print(f"Running SafeOpt-GP (Seed {seed}, V={V_val}, Beta={beta}, Grid={n_grid})...")
    env = SimulatedEnvironment(seed, V_val, w, n_grid)
    
    # 2D Action space: Bandwidth, Power
    bounds = [(0.1, 1.0), (0.1, 1.0)]
    
    # Create candidates grid
    candidates = build_candidates(n_grid=n_grid, bounds=bounds)
    
    # For index mapping
    x1 = np.linspace(bounds[0][0], bounds[0][1], n_grid)
    x2 = np.linspace(bounds[1][0], bounds[1][1], n_grid)
    
    # RNG Decoupling: SafeOptGP internal RNG is handled by library or defaults.
    # If SafeOptGP uses np.random internally, it might be coupled.
    # Assuming SafeOptGP is deterministic given data or we can't easily change it without forking.
    # But we can re-seed global numpy if needed, but we should avoid it.
    # Let's assume SafeOptGP (Bayesian Optimization) is deterministic for given inputs (GP mean/var).
    
    agent = SafeOptGP(bounds=bounds, safety_threshold=0.0, beta=beta) 
    
    # Initial data
    # Warm start with a safe action (low bandwidth, high power) to ensure SafeOpt starts in safe set
    action_init = np.array([0.1, 1.0])
    state, reward, _, _, info = env.step(action_init) 
    q_prev = info["q_semantic"]
    
    # Update Agent with initial data
    # y_safe = q_prev - q_curr. Initial: 0.0 - q_prev
    agent.update(action_init, reward, 0.0 - q_prev)
    
    results = []
    
    start_time = time.perf_counter()
    
    # SafeOpt-GP Loop with Latency Measurement
    for t in range(1, HORIZON + 1):
        try:
            q_prev_before = q_prev

            # Optimization step
            t0 = time.perf_counter()
            
            # Determine dynamic safety threshold based on current queue state
            # Safety condition: q_next <= 18.0 (Margin of 2.0 to account for noise)
            # q_prev - drift <= 18.0
            # drift >= q_prev - 18.0
            if env.scenario == "bandit_stationary_pure":
                safety_threshold = -np.inf
            else:
                safety_threshold = q_prev - 18.0
            
            action = agent.optimize(candidates, threshold=safety_threshold, V=V_val, Q=q_prev)
            
            t1 = time.perf_counter()
            decision_latency_ms = (t1 - t0) * 1000.0
            
            # Environment step
            state, reward, cost, done, info = env.step(action)
            
            # Unified Metrics Calculation
            utility_reward = reward
            q_curr = info["q_semantic"]
            drift_val, penalty_term, lyapunov_reward, scaled_reward = compute_lyapunov_metrics(V_val, utility_reward, q_prev_before, q_curr, 1.0)

            # Update Agent
            y_obj = reward
            # Use Drift as safety signal: q_prev - q_curr
            # Positive Drift = Queue Decreasing (Safe)
            # Negative Drift = Queue Increasing (Risk)
            # q_curr = info["q_semantic"]
            y_safe = q_prev - q_curr
            
            agent.update(action, y_obj, y_safe)
            q_prev = q_curr
            
            # Log
            # Map action to discrete indices
            bw_idx = int(np.argmin(np.abs(x1 - action[0])))
            p_idx = int(np.argmin(np.abs(x2 - action[1])))
            log_entry = {
                "algorithm": "safeopt_gp",
                "seed": seed,
                "t": t,
                "V": info["V"],
                "window": info["window"],
                "grid_density": info["grid_density"],
                "mos": info["mos"],
                "swer": info["swer"],
                "q_semantic": info["q_semantic"],
                "q_energy": info["q_energy"],
                "violation_flag": info["violation_flag"],
                "energy_cum": info["energy_cum"],
                "snr_db": info["snr_db"],
                "per": info.get("per", np.nan),
                "noise_std": info.get("noise_std", "N/A"),
                "semantic_weight": info.get("semantic_weight", 0.0),
                "P_index": p_idx,
                "B_index": bw_idx,
                "reward": reward,
                "q_prev_before": q_prev_before,
                "utility_reward": utility_reward,
                "lyapunov_reward": lyapunov_reward,
                "scaled_reward": scaled_reward,
                "decision_latency_ms": decision_latency_ms,
                "mu_star": info.get("mu_star", 0.0),
                "mu_choice": info.get("mu_choice", 0.0),
                "run_tag": f"task3_safeopt_gp_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}",
                "experiment_id": f"exp_{seed}",
                "scenario": info.get("scenario", "default")
            }
            results.append(log_entry)
            
            if t % 100 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  Seed {seed}: Step {t}/{HORIZON} (Elapsed: {elapsed:.2f}s)")
                
            # Checkpoint every 500 steps
            if t % 500 == 0:
                 df = pd.DataFrame(results)
                 filename = f"task3_safeopt_gp_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}.csv"
                 filepath = os.path.join(OUTPUT_DIR, filename)
                 df.to_csv(filepath, index=False)
                 
        except KeyboardInterrupt:
            print(f"Interrupted at step {t}")
            break
        except Exception as e:
            print(f"Error at step {t}: {e}")
            traceback.print_exc()
            break
            
    # Save CSV
    df = pd.DataFrame(results)
    filename = f"task3_safeopt_gp_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {filepath}")

def run_lyapunov_greedy_oracle(seed, V_val=5.0, n_grid=10):
    print(f"Running Lyapunov Greedy Oracle (Seed {seed}, V={V_val}, Grid={n_grid})...")
    env = SimulatedEnvironment(seed, V_val, w, n_grid)
    
    # Matching RUCB+ grid for fair regret comparison
    candidates = build_candidates(n_grid=n_grid)

    x1 = np.linspace(0.1, 1.0, n_grid)
    x2 = np.linspace(0.1, 1.0, n_grid)
    
    results = []
    state, _, _, _, info = env.step([0.5, 0.5]) # warm start
    q_prev = info["q_semantic"]
    
    for t in range(1, HORIZON + 1):
        q_prev_before = q_prev

        # Lyapunov Drift-Plus-Penalty Optimization

        # Maximize: V * Utility + Q * Service
        # Utility = MOS - 0.1 * Energy
        # Service = bw * 2.0 * channel_quality
        # We assume Energy = p * 1.0
        
        # Current state info
        q_sem = info["q_semantic"]
        # h_curr = state[2] 
        # For MAB comparison, Oracle should not use instantaneous channel info that MAB cannot see.
        # We use the mean channel quality (0.5) to define the "Best Expected Arm".
        h_curr = 0.5 
        
        t0 = time.perf_counter()

        
        # Vectorized optimization for speed
        bw_vec = candidates[:, 0]
        p_vec = candidates[:, 1]
        
        # Predict metrics
        mos_vec = 1.0 + 4.0 * bw_vec
        arrival_vec = bw_vec * 2.0
        service_vec = p_vec * 2.5 * h_curr
        
        # Queue dynamics prediction
        q_next_vec = q_sem + arrival_vec - service_vec
        
        # Add safety margin to account for channel uncertainty
        # Channel noise std=0.05. Service sensitivity ~2.5. 
        # Use margin 2.0 (threshold 18.0) to be safe against random walk drifts.
        violation_mask = q_next_vec > 18.0
        
        # Objective: Maximize V * Utility + Q * Drift
        # Utility = MOS - 0.1 * Energy
        # Drift = service - arrival = - (q_next - q_sem)
        # Actually Lyapunov Drift term is Q * (service - arrival)
        
        utility_vec = mos_vec - 0.1 * (p_vec * 1.0)
        drift_vec = service_vec - arrival_vec
        
        # Standard Lyapunov Objective
        obj_vec = V_val * utility_vec + q_sem * drift_vec
        
        # Constraint Handling:
        # If there are valid candidates (q_next <= 19.0), pick best among them.
        # If ALL candidates violate, pick the one that minimizes Queue Growth (Max Drift).
        
        if np.any(~violation_mask):
            # There is at least one safe action
            # Set objective of unsafe actions to -infinity
            obj_vec[violation_mask] = -np.inf
        else:
            # Emergency Mode: All actions lead to violation.
            # Ignore V * Utility. Purely maximize Drift (drain queue).
            # This overrides the "Energy Saving" behavior that causes overflow at high V.
            obj_vec = drift_vec
        
        best_idx = np.argmax(obj_vec)
        best_action = candidates[best_idx]

        bw_idx = int(np.argmin(np.abs(x1 - best_action[0])))
        p_idx = int(np.argmin(np.abs(x2 - best_action[1])))
        
        t1 = time.perf_counter()
        decision_latency_ms = (t1 - t0) * 1000.0
        
        # Execute best action
        state, reward, cost, done, info = env.step(best_action)
        
        # Unified Metrics Calculation
        utility_reward = reward
        q_curr = info["q_semantic"]
        drift_val, penalty_term, lyapunov_reward, scaled_reward = compute_lyapunov_metrics(V_val, utility_reward, q_prev_before, q_curr, 1.0)
        
        if env.scenario == "bandit_stationary_pure":
            scaled_reward = utility_reward # Force pure reward for stationary sanity
        
        q_prev = q_curr
        
        # Log
        log_entry = {
            "algorithm": "lyapunov_greedy_oracle",
            "seed": seed,
            "t": t,
            "V": info["V"],
            "window": info["window"],
            "grid_density": info["grid_density"],
            "mos": info["mos"],
            "swer": info["swer"],
            "q_semantic": info["q_semantic"],
            "q_energy": info["q_energy"],
            "violation_flag": info["violation_flag"],
            "energy_cum": info["energy_cum"],
            "snr_db": info["snr_db"],
            "per": info.get("per", np.nan),
            "noise_std": info.get("noise_std", "N/A"),
            "semantic_weight": info.get("semantic_weight", 0.0),
            "P_index": p_idx,
            "B_index": bw_idx,
            "reward": reward,
            "q_prev_before": q_prev_before,
            "utility_reward": utility_reward,
            "lyapunov_reward": lyapunov_reward,
            "scaled_reward": scaled_reward,
            "decision_latency_ms": decision_latency_ms,
            "mu_star": info.get("mu_star", 0.0),
            "mu_choice": info.get("mu_choice", 0.0),
            "run_tag": f"task3_lyapunov_greedy_oracle_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}",
            "experiment_id": f"exp_{seed}",
            "scenario": info.get("scenario", "default")
        }
        results.append(log_entry)
        
        if t % 1000 == 0:
            print(f"  Oracle Seed {seed}: Step {t}/{HORIZON}")

    # Save CSV
    df = pd.DataFrame(results)
    filename = f"task3_lyapunov_greedy_oracle_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {filepath}")

def run_best_fixed_arm_oracle(seed, V_val=800.0, n_grid=10, best_by="scaled_reward"):

    """
    Runs all arms in the n_grid x n_grid space for the full horizon.
    Selects the arm with the highest cumulative reward (defined by best_by).
    Saves the trace of that best arm.
    """
    print(f"Running Best Fixed Arm Oracle (Seed {seed}, V={V_val}, Grid={n_grid}, BestBy={best_by})...")
    
    candidates = build_candidates(n_grid=n_grid)
    env = SimulatedEnvironment(seed, V_val, w, n_grid)
    
    x1 = np.linspace(0.1, 1.0, n_grid) # Bandwidth
    x2 = np.linspace(0.1, 1.0, n_grid) # Power
    n_arms = len(candidates)
    
    best_arm_idx = -1
    max_cum_reward = -float('inf')
    best_trace = []
    
    # Iterate over all arms to find the best fixed one
    for arm_idx in range(n_arms):
        # Reset environment with SAME seed to replay dynamics
        env = SimulatedEnvironment(seed, V_val, w, n_grid)
        
        # Warm start (same as others)
        state, _, _, _, info = env.step([0.5, 0.5])
        q_prev = info["q_semantic"]
        
        current_trace = []
        cum_reward = 0.0
        
        action = candidates[arm_idx]
        
        for t in range(1, HORIZON + 1):
            q_prev_before = q_prev
            state, reward, cost, done, info = env.step(action)
            
            # Unified Metrics Calculation
            utility_reward = reward
            q_curr = info["q_semantic"]
            drift_val, penalty_term, lyapunov_reward, scaled_reward = compute_lyapunov_metrics(V_val, utility_reward, q_prev_before, q_curr, 1.0)
            
            q_prev = q_curr
            
            # Select metric for accumulation
            if best_by == "scaled_reward":
                val_to_acc = scaled_reward
            elif best_by == "utility_reward":
                val_to_acc = utility_reward
            elif best_by == "lyapunov_reward":
                val_to_acc = lyapunov_reward
            else:
                val_to_acc = scaled_reward
            
            cum_reward += val_to_acc
            
            if arm_idx == 0: # Placeholder, will be overwritten if best
                pass
                
            # Store trace for potential saving
            bw_idx = int(np.argmin(np.abs(x1 - action[0])))
            p_idx = int(np.argmin(np.abs(x2 - action[1])))
            
            log_entry = {
                "algorithm": "best_fixed_arm_oracle",
                "seed": seed,
                "t": t,
                "V": info["V"],
                "window": info["window"],
                "grid_density": info["grid_density"],
                "mos": info["mos"],
                "swer": info["swer"],
                "q_semantic": info["q_semantic"],
                "q_energy": info["q_energy"],
                "violation_flag": info["violation_flag"],
                "energy_cum": info["energy_cum"],
                "snr_db": info["snr_db"],
                "per": info.get("per", np.nan),
                "noise_std": info.get("noise_std", "N/A"),
                "semantic_weight": info.get("semantic_weight", 0.0),
                "P_index": p_idx,
                "B_index": bw_idx,
                "reward": utility_reward,
                "q_prev_before": q_prev_before,
                "utility_reward": utility_reward,
                "lyapunov_reward": lyapunov_reward,
                "scaled_reward": scaled_reward,
                "decision_latency_ms": 0.0,
                "mu_star": info.get("mu_star", 0.0),
                "mu_choice": info.get("mu_choice", 0.0),
                "run_tag": f"task3_best_fixed_arm_oracle_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}",
                "experiment_id": f"exp_{seed}",
                "best_by_metric": best_by,
                "scenario": info.get("scenario", "default")
            }
            current_trace.append(log_entry)
            
        # Check if this arm is better
        if cum_reward > max_cum_reward:
            max_cum_reward = cum_reward
            best_arm_idx = arm_idx
            best_trace = current_trace
            
    print(f"  Best Fixed Arm for Seed {seed}: Index {best_arm_idx} (Cum {best_by}: {max_cum_reward:.2f})")
    
    # Save trace of the best arm
    df = pd.DataFrame(best_trace)
    
    # Strict Check for Stationary Bandit Mode
    if env.scenario == "bandit_stationary_pure":
        if "mu_star" in df.columns and "mu_choice" in df.columns:
            # Oracle's chosen arm MUST be the optimal arm
            # So mu_choice must equal mu_star
            mu_star = df["mu_star"]
            mu_choice = df["mu_choice"]
            diff = (mu_star - mu_choice).abs()
            max_diff = diff.max()
            
            if max_diff > 1e-12:
                idx = diff.idxmax()
                msg = f"[FATAL] Oracle failed optimality check! max_diff={max_diff} at t={idx+1}\nmu_star={mu_star.iloc[idx]}, mu_choice={mu_choice.iloc[idx]}"
                print(msg)
                raise RuntimeError(msg)
            else:
                print(f"  [PASS] Oracle Optimality Verified (max_diff={max_diff:.2e})")

    filename = f"task3_best_fixed_arm_oracle_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {filepath}")


def verify_rng_decoupling():
    """
    Verifies that the environment's RNG is strictly local and unaffected by global np.random calls.
    """
    print("\n=== Verifying Environment RNG Decoupling ===")
    
    seed = 42
    horizon = 100
    fixed_action = [0.5, 0.5]
    
    # Run 1: Clean run
    env1 = SimulatedEnvironment(seed)
    rewards1 = []
    for _ in range(horizon):
        _, r, _, _, _ = env1.step(fixed_action)
        rewards1.append(r)
        
    # Run 2: Polluted run (Global RNG calls mixed in)
    env2 = SimulatedEnvironment(seed)
    rewards2 = []
    np.random.seed(999) # Change global seed
    # Verify numpy global seed change worked
    # rand1 = np.random.rand() # Commented out to avoid linter error if unused, or just use it
    # _ = np.random.rand()
    
    for _ in range(horizon):
        # Pollute global state
        _ = np.random.normal(0, 1, 100)
        _ = np.random.uniform(0, 1, 100)
        
        _, r, _, _, _ = env2.step(fixed_action)
        rewards2.append(r)
        
    # Verification
    rewards1 = np.array(rewards1)
    rewards2 = np.array(rewards2)
    
    if np.allclose(rewards1, rewards2):
        print("[PASS] Environment RNG is successfully decoupled from global state.")
    else:
        diff = np.abs(rewards1 - rewards2).sum()
        print(f"[FAIL] Environment RNG is polluted by global state! Diff sum: {diff}")
        print("Run 1 head:", rewards1[:5])
        print("Run 2 head:", rewards2[:5])
        raise RuntimeError("RNG Decoupling Verification Failed")

# --- Helper Classes for Bandit Algorithms ---

class LinUCBAgent:
    def __init__(self, n_features, alpha=0.5, regularization=1.0):
        self.M = regularization * np.eye(n_features)
        self.M_inv = (1.0 / regularization) * np.eye(n_features)
        self.b = np.zeros(n_features)
        self.alpha = alpha

    def update(self, x, y):
        # Recursive Least Squares update
        # M_new = M + x x^T
        # Sherman-Morrison formula for inverse:
        # (A + uv^T)^-1 = A^-1 - (A^-1 u v^T A^-1) / (1 + v^T A^-1 u)
        # Here u = x, v = x
        
        x = x.reshape(-1, 1) # Column vector
        
        # Update b: b_new = b + y * x
        self.b += y * x.flatten()
        
        # Update M_inv
        Mx = self.M_inv @ x
        denom = 1 + x.T @ Mx
        update_term = (Mx @ Mx.T) / denom
        self.M_inv -= update_term

    def predict(self, x):
        # Theta = M_inv @ b
        theta = self.M_inv @ self.b
        
        mean = x @ theta
        
        # Var = x^T M_inv x
        var = (x @ self.M_inv @ x)
        var = max(0, var) # Numerical stability
        std = np.sqrt(var)
        
        ucb = mean + self.alpha * std
        lcb = mean - self.alpha * std
        return mean, ucb, lcb, std

    def predict_batch(self, X):
        """
        Vectorized prediction for batch of feature vectors X (N, d).
        Returns: mean, ucb, lcb, std (all shape (N,))
        """
        theta = self.M_inv @ self.b
        mean = X @ theta
        
        # Variance: diag(X @ M_inv @ X.T)
        # Efficient computation: sum((X @ M_inv) * X, axis=1)
        temp = X @ self.M_inv
        var = np.sum(temp * X, axis=1)
        var = np.maximum(0, var)
        std = np.sqrt(var)
        
        ucb = mean + self.alpha * std
        lcb = mean - self.alpha * std
        return mean, ucb, lcb, std

class SWUCBAgent:
    def __init__(self, n_arms, window_size, exploration_const=1.0):
        self.n_arms = n_arms
        self.window_size = window_size
        self.c = exploration_const
        self.history = [] # List of (arm, reward)
        
    def update(self, arm, reward):
        self.history.append((arm, reward))
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
    def select_action(self):
        # Calculate counts and sums from history
        counts = np.zeros(self.n_arms)
        sums = np.zeros(self.n_arms)
        
        for a, r in self.history:
            counts[a] += 1
            sums[a] += r
            
        # UCB
        t = len(self.history)
        if t == 0:
            # RNG Decoupling: Use internal RNG or deterministic choice
            # For simplicity, pick 0
            return 0
            
        ucbs = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            if counts[i] == 0:
                ucbs[i] = float('inf')
            else:
                avg = sums[i] / counts[i]
                bonus = self.c * np.sqrt(np.log(min(t, self.window_size)) / counts[i])
                ucbs[i] = avg + bonus
                
        return np.argmax(ucbs)

def get_features(bw, p, h):
    # Features: [bw, p, h, bw*h, 1] - Captures MOS (1+4*bw*h) and Energy (p)
    return np.array([bw, p, h, bw*h, 1.0])

def run_sw_ucb(seed, V_val=800.0, window_size=None, exploration_const=0.5, n_grid=10):
    env = SimulatedEnvironment(seed, V_val, w, n_grid)
    
    if env.scenario == "bandit_stationary_pure":
        window_size = HORIZON
        exploration_const = 0.5 # Reduced from 1.414 to 0.5 to improve convergence slope
    elif window_size is None:
        window_size = w

    print(f"Running SW-UCB (Seed {seed}, V={V_val}, Window={window_size}, c={exploration_const}, Grid={n_grid})...")
    
    # Grid of arms
    candidates = build_candidates(n_grid=n_grid)
    x1 = np.linspace(0.1, 1.0, n_grid) # Bandwidth
    x2 = np.linspace(0.1, 1.0, n_grid) # Power
    n_arms = len(candidates)
    
    agent = SWUCBAgent(n_arms=n_arms, window_size=window_size, exploration_const=exploration_const)
    
    results = []
    state, _, _, _, info = env.step([0.5, 0.5]) # Warm start
    q_prev = info["q_semantic"]
    
    for t in range(1, HORIZON + 1):
        # SW-UCB selects arm index
        # Safety Guard: If queue is critical, force conservative action
        
        t0 = time.perf_counter()
        if env.scenario != "bandit_stationary_pure" and info["q_semantic"] > 18.0:
             # Find action with min bandwidth (<0.2) and max power (>0.8)
             safe_indices = np.where((candidates[:,0] <= 0.2) & (candidates[:,1] >= 0.8))[0]
             if len(safe_indices) > 0:
                 arm_idx = safe_indices[0] # Pick first safe action
             else:
                 arm_idx = agent.select_action()
        else:
            arm_idx = agent.select_action()
        
        t1 = time.perf_counter()
        decision_latency_ms = (t1 - t0) * 1000.0
            
        action = candidates[arm_idx]
        
        q_prev_before = q_prev
        state, reward, cost, done, info = env.step(action)
        
        # Unified Metrics Calculation
        utility_reward = reward
        q_curr = info["q_semantic"]
        drift_val, penalty_term, lyapunov_reward, scaled_reward = compute_lyapunov_metrics(V_val, utility_reward, q_prev_before, q_curr, 1.0)
        
        if env.scenario == "bandit_stationary_pure":
            scaled_reward = utility_reward # Force pure reward for stationary sanity
        
        q_prev = q_curr
        
        # Update agent
        # SW-UCB typically assumes bounded rewards [0,1]. Our reward can be ~5.
        # Normalize reward for update? Or adjust C.
        # Reward = MOS - 0.1*E. MOS in [1,5]. E in [0,1]. Reward in [0.9, 5.0].
        # Let's normalize by dividing by 5.0 roughly.
        if env.scenario == "bandit_stationary_pure":
             # In stationary mode with c=0.5, raw reward (1-5) might overwhelm c.
             # If we normalize, c=0.5 is large relative to [0,1].
             # If we don't normalize, c=0.5 is small relative to [1,5].
             # SW-UCB failure "Exploding" means regret is growing linearly -> exploring too much?
             # If slope_ratio ~ 0.57, it's exploring too much.
             # If we normalize, c=0.5 is effectively large.
             # If we DON'T normalize, c=0.5 is effectively small (0.1 normalized).
             # Let's try NOT normalizing in stationary mode to make it more greedy (log-like).
             agent.update(arm_idx, reward)
        else:
             agent.update(arm_idx, reward / 5.0)
        
        bw_idx = int(np.argmin(np.abs(x1 - action[0])))
        p_idx = int(np.argmin(np.abs(x2 - action[1])))
        log_entry = {
            "algorithm": "sw_ucb",
            "seed": seed,
            "t": t,
            "V": info["V"],
            "window": info["window"],
            "grid_density": info["grid_density"],
            "mos": info["mos"],
            "swer": info["swer"],
            "q_semantic": info["q_semantic"],
            "q_energy": info["q_energy"],
                "violation_flag": info["violation_flag"],
                "energy_cum": info["energy_cum"],
                "snr_db": info["snr_db"],
                "per": info.get("per", np.nan),
                "noise_std": info.get("noise_std", "N/A"),
                "semantic_weight": info.get("semantic_weight", 0.0),
                "P_index": p_idx,
            "B_index": bw_idx,
            "reward": reward,
            "q_prev_before": q_prev_before,
            "utility_reward": utility_reward,
            "lyapunov_reward": lyapunov_reward,
            "scaled_reward": scaled_reward,
            "decision_latency_ms": decision_latency_ms,
            "mu_star": info.get("mu_star", 0.0),
            "mu_choice": info.get("mu_choice", 0.0),
            "run_tag": f"task3_sw_ucb_V{V_val}_w{window_size}_g{g}_T{HORIZON}_s{seed}",
            "experiment_id": f"exp_{seed}",
            "scenario": info.get("scenario", "default")
        }
        results.append(log_entry)
        
        if t % 1000 == 0:
            print(f"  SW-UCB Seed {seed}: Step {t}/{HORIZON}")
            
    # Save CSV
    df = pd.DataFrame(results)
    filename = f"task3_sw_ucb_V{V_val}_w{window_size}_g{g}_T{HORIZON}_s{seed}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {filepath}")

def run_safe_linucb(seed, V_val=800.0, alpha_obj=0.5, safety_lcb_threshold=0.8, score_trace_steps=0, output_dir=None, n_grid=10):

    print(f"Running Safe-LinUCB (Seed {seed}, V={V_val}, Grid={n_grid})...")
    env = SimulatedEnvironment(seed, V_val, w, n_grid)
    
    # Grid
    candidates = build_candidates(n_grid=n_grid)
    x1 = np.linspace(0.1, 1.0, n_grid)
    x2 = np.linspace(0.1, 1.0, n_grid)
    
    # Contextual Agents
    # Features: 5 dimensions
    if env.scenario == "bandit_stationary_pure":
        alpha_obj = 0.1 # Lower alpha for faster convergence in simple linear setting
        
    obj_agent = LinUCBAgent(n_features=5, alpha=alpha_obj)
    safe_agent = LinUCBAgent(n_features=5, alpha=alpha_obj) # Models Safety (1-Cost)
    
    results = []
    state, _, _, _, info = env.step([0.5, 0.5])
    q_prev = info["q_semantic"]
    
    for t in range(1, HORIZON + 1):
        h_curr = state[2] # Current channel quality
        q_prev_before = q_prev
        
        t0 = time.perf_counter()
        
        # Select Action
        best_arm = None
        best_ucb = -float('inf')
        
        # Check safety for all arms
        safe_set = []
        
        # Pre-calculate features for all arms (optimization)
        # But features depend on h_curr which changes every step
        
        # We need to find: argmax UCB_obj(a) s.t. LCB_safety(a) >= Threshold
        # Safety = 1 (Safe) or 0 (Unsafe). Threshold = 0.8 (High probability of being safe)
        
        feasible_arms = []
        fallback_arm = None
        max_safety_lcb = -float('inf')
        
        for i, cand in enumerate(candidates):
            bw, p = cand
            feats = get_features(bw, p, h_curr)
            
            # Predict Objective
            mean_obj, ucb_obj, lcb_obj, std_obj = obj_agent.predict(feats)
            
            # Predict Safety (1 - Cost)
            # Cost is 0 or 1. Safety is 1 or 0.
            # We want Safety to be high.
            mean_safety, ucb_safety, lcb_safety, std_safety = safe_agent.predict(feats)
            
            if lcb_safety >= safety_lcb_threshold: # Threshold
                feasible_arms.append((i, ucb_obj))
            
            # Fallback: Pick arm with highest UCB Safety (Optimistic exploration of safety)
            if ucb_safety > max_safety_lcb:
                max_safety_lcb = ucb_safety
                fallback_arm = i
                
        if feasible_arms:
            # Pick argmax UCB from feasible
            best_arm_idx = max(feasible_arms, key=lambda x: x[1])[0]
        else:
            # Pick arm with highest safety LCB (least violation risk)
            best_arm_idx = fallback_arm
            
        t1 = time.perf_counter()
        decision_latency_ms = (t1 - t0) * 1000.0
            
        action = candidates[best_arm_idx]
        
        # Execute
        state, reward, cost, done, info = env.step(action)
        
        # Unified Metrics Calculation
        utility_reward = reward
        q_curr = info["q_semantic"]
        drift_val, penalty_term, lyapunov_reward, scaled_reward = compute_lyapunov_metrics(V_val, utility_reward, q_prev_before, q_curr, 1.0)
        
        # Update
        feats = get_features(action[0], action[1], h_curr)
        obj_agent.update(feats, reward)
        
        # Safety Signal (Gradient-based): 
        # 1.0 if queue decreases (recovering) OR queue is already safe (<= 20). 0.0 otherwise.
        q_curr = info["q_semantic"]
        is_safe_state = (q_curr <= 20.0)
        is_recovering = (q_curr <= q_prev) # Strictly decreasing or stable
        
        y_safe = 1.0 if (is_safe_state or is_recovering) else 0.0
        
        if env.scenario == "bandit_stationary_pure":
            # In stationary bandit, we care about Cost Constraint (p > h -> Cost=1).
            # Safety = 1 - Cost.
            y_safe = 1.0 - cost
        
        safe_agent.update(feats, y_safe)
        q_prev = q_curr
        
        bw_idx = int(np.argmin(np.abs(x1 - action[0])))
        p_idx = int(np.argmin(np.abs(x2 - action[1])))
        final_score = max(feasible_arms, key=lambda x: x[1])[1] if feasible_arms else lcb_safety
        log_entry = {
            "algorithm": "safe_linucb",
            "seed": seed,
            "t": t,
            "V": info["V"],
            "window": info["window"],
            "grid_density": info["grid_density"],
            "mos": info["mos"],
            "swer": info["swer"],
            "q_semantic": info["q_semantic"],
            "q_energy": info["q_energy"],
                "violation_flag": info["violation_flag"],
                "energy_cum": info["energy_cum"],
                "snr_db": info["snr_db"],
                "per": info.get("per", np.nan),
                "noise_std": info.get("noise_std", "N/A"),
                "semantic_weight": info.get("semantic_weight", 0.0),
                "P_index": p_idx,
            "B_index": bw_idx,
            "reward": reward,
            "q_prev_before": q_prev_before,
            "utility_reward": utility_reward,
            "lyapunov_reward": lyapunov_reward,
            "scaled_reward": scaled_reward,
            "decision_latency_ms": decision_latency_ms,
            "mu_star": info.get("mu_star", 0.0),
            "mu_choice": info.get("mu_choice", 0.0),
            "run_tag": f"task3_safe_linucb_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}",
            "experiment_id": f"exp_{seed}",
            "scenario": info.get("scenario", "default")
        }
        if score_trace_steps and t <= score_trace_steps:
            log_entry.update({
                "score_final": float(final_score),
                "obj_mean": float(mean_obj),
                "obj_std": float(std_obj),
                "obj_ucb": float(ucb_obj),
                "safety_mean": float(mean_safety),
                "safety_std": float(std_safety),
                "safety_ucb": float(ucb_safety),
                "safety_lcb": float(lcb_safety),
                "selected_arm": int(best_arm_idx),
            })
        results.append(log_entry)
        
        if t % 1000 == 0:
            print(f"  Safe-LinUCB Seed {seed}: Step {t}/{HORIZON}")

    # Save CSV
    df = pd.DataFrame(results)
    filename = f"task3_safe_linucb_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}.csv"
    filepath = os.path.join(output_dir or OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {filepath}")


def run_raucb_plus(
    seed,
    V_val=800.0,
    alpha_obj=0.5,
    alpha_drift=0.5,
    alpha_safety=0.5,
    penalty_scale=1.0,
    safety_lcb_threshold=0.8,
    elim_start=200,
    elim_interval=50,
    score_trace_steps=0,
    output_dir=None,
    drift_variant="ucb",
    rucb_update_metric="scaled_reward", # Options: "scaled_reward", "utility_reward"
    enable_ph=True,
    n_grid=10
):
    print(f"Running RA-UCB++ (Zooming + Hard Safety) (Seed {seed}, V={V_val}, PH={enable_ph}, Grid={n_grid})...")
    
    # Use full grid for candidate definition
    full_candidates = build_candidates(n_grid=n_grid)
    x1_full = np.linspace(0.1, 1.0, n_grid)
    x2_full = np.linspace(0.1, 1.0, n_grid)
    K = len(full_candidates)
    
    # 1. Zooming Initialization (Mask-based)
    # Stride of 5 means 10/5 = 2 points per dimension -> 4 arms? Too few.
    # If n_grid=50, stride 5 -> 10 points -> 100 arms. Good.
    stride = 5 if n_grid >= 50 else 2
    
    active_mask = np.zeros(K, dtype=bool)
    
    # Init active_mask based on stride, forcing boundary inclusion
    # We want to ensure 0 and n_grid-1 are always included in the grid points
    grid_indices = np.unique(np.concatenate([
        np.arange(0, n_grid, stride), 
        [n_grid - 1]
    ])).astype(int)
    
    # Index = r * n_grid + c
    for r in grid_indices:
        for c in grid_indices:
            idx = r * n_grid + c
            if idx < K:
                active_mask[idx] = True
    
    print(f"  [Zooming] Initial Active Set Size: {active_mask.sum()} / {K}")

    env = SimulatedEnvironment(seed, V_val, w, n_grid)
    
    # Initialize Agent with FULL K (No resizing, no reset)
    cfg = RUCBPlusConfig(
        K=K, 
        window_init=w, 
        adapt_w=True, 
        c_eps=0.1,
        min_pulls_per_arm=10,    
        n_min_elim=30,           
        elim_margin=0.05,        
        recheck_period=500,      
        enable_ph=enable_ph      
    )
    agent = RUCBPlus(cfg)
    
    results = []
    
    # Warm start
    state, _, _, _, info = env.step([0.5, 0.5])
    q_prev = info["q_semantic"]
    
    zooming_schedule = [2000, 5000]
    
    # Get Scenario from Env Var if not in info
    scenario_val = os.environ.get("TASK3_SCENARIO", "default")

    for t in range(1, HORIZON + 1):
        q_prev_before = q_prev 
        h_curr = state[2]
        
        # P0: Call select_arm ONCE per step
        # This ensures consistent UCBs for both Zooming and Safety check
        # and prevents potential side-effect duplication.
        k_tmp, ucb_vec, lcb_vec = agent.select_arm()

        # --- ZOOMING LOGIC ---
        if t in zooming_schedule:
            print(f"  [Zooming] t={t}: Refining Active Set...")
            
            # Filter UCB by current active_mask (using the pre-computed ucb_vec)
            # We want Top 5 ACTIVE arms
            active_ucbs = np.where(active_mask, ucb_vec, -np.inf)
            
            # Get top 5 indices (argsort is ascending)
            top_indices = np.argsort(active_ucbs)[-5:]
            # Filter valid
            top_indices = [idx for idx in top_indices if active_ucbs[idx] > -1e9]
            
            # Expand neighbors
            new_active_count = 0
            for center_idx in top_indices:
                r, c = center_idx // n_grid, center_idx % n_grid
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n_grid and 0 <= nc < n_grid:
                            n_idx = nr * n_grid + nc
                            if not active_mask[n_idx]:
                                active_mask[n_idx] = True
                                new_active_count += 1
            
            print(f"  [Zooming] Added {new_active_count} new arms. Total: {active_mask.sum()}")
            # NO Agent Reset

        # --- SELECTION WITH HARD SAFETY ---
        t0 = time.perf_counter()
        
        # 1. Get Agent's UCBs
        # Already obtained: ucb_vec
        
        # 2. Compute Safety Estimates (Vectorized)
        # q_next_est = max(0, q_prev + arrival - service)
        bws = full_candidates[:, 0]
        ps = full_candidates[:, 1]
        
        arrs = bws * 2.0
        srvs = ps * 2.5 * h_curr
        q_next_est_vec = np.maximum(0, q_prev + arrs - srvs)
        
        # Safe Mask: q_next <= 18.0
        safe_mask = q_next_est_vec <= 18.0
        
        # 3. Filter UCBs
        # Allowed = Active AND Safe
        allowed_mask = active_mask & safe_mask
        
        safety_overridden = 0
        final_idx = -1
        
        # Check if agent's choice (k_tmp) is safe and active?
        # k_tmp is just index in 0..K-1.
        # But we need to enforce that we ONLY pick from allowed if possible.
        # If k_tmp is in allowed, we might respect it?
        # But RUCB+ agent doesn't know about active_mask in this implementation (K=full).
        # So k_tmp could be anywhere.
        # We must override if k_tmp is not allowed.
        # Actually, to be consistent with "Hard Safety", we should just pick argmax from Allowed.
        # If k_tmp happens to be the argmax of Allowed, great.
        
        if np.any(allowed_mask):
            # Pick argmax UCB among allowed
            masked_ucb = np.where(allowed_mask, ucb_vec, -np.inf)
            final_idx = np.argmax(masked_ucb)
            # Check if this is an override
            # If the agent's choice (k_tmp) was already safe and active, and is the best?
            # Or simply: did we change the arm?
            # If final_idx != k_tmp, it is effectively an override/correction?
            # But the user definition says:
            # 0=正常 (Normal/Safe in Active)
            # 1=active内安全筛选 (Filtered in Active)
            # This distinction is subtle.
            # Let's say: 0 if we picked from Active&Safe.
            # 1 is not used in this interpretation?
            # Or maybe:
            # 0 = Agent choice was Safe & Active.
            # 1 = Agent choice unsafe/inactive, but we found another Safe & Active.
            
            if allowed_mask[k_tmp]:
                # Agent choice is valid. We should probably stick to it?
                # But is k_tmp the ARGMAX of ucb_vec? Yes, by definition of select_arm.
                # So if k_tmp is allowed, it MUST be the argmax of masked_ucb too (since others have same UCB or -inf).
                # So final_idx will be equal to k_tmp (or equal UCB).
                safety_overridden = 0
            else:
                safety_overridden = 1 # We had to pick a different one within Active & Safe
                
        else:
            # Fallback 1: Any Safe arm in FULL set?
            if np.any(safe_mask):
                # Pick argmax UCB among ALL safe arms
                masked_ucb_safe = np.where(safe_mask, ucb_vec, -np.inf)
                final_idx = np.argmax(masked_ucb_safe)
                safety_overridden = 2 # Global Safe
            else:
                # Fallback 2: Max Drift (Minimize Queue Growth)
                # drift = service - arrival
                drifts = srvs - arrs
                final_idx = np.argmax(drifts)
                safety_overridden = 3 # Emergency
        
        action = full_candidates[final_idx]
        q_next_est = q_next_est_vec[final_idx]
        
        t1 = time.perf_counter()
        decision_latency_ms = (t1 - t0) * 1000.0
        
        # Execute
        state, reward, cost, done, info = env.step(action)
        
        # Metrics
        utility_reward = reward
        q_curr = info["q_semantic"]
        drift_val, penalty_term, lyapunov_reward, scaled_reward = compute_lyapunov_metrics(V_val, utility_reward, q_prev_before, q_curr, 1.0)
        
        # Update Agent
        # Update the arm we actually pulled
        # No artificial penalty hack (-5.0) as requested
        agent.update(final_idx, scaled_reward)
        
        q_prev = q_curr
        
        # Logging
        bw_idx = int(np.argmin(np.abs(x1_full - action[0])))
        p_idx = int(np.argmin(np.abs(x2_full - action[1])))
        
        log_entry = {
            "algorithm": "raucb_plus",
            "seed": seed,
            "t": t,
            "V": info["V"],
            "window": info["window"],
            "grid_density": info["grid_density"],
            "mos": info["mos"],
            "swer": info["swer"],
            "q_semantic": info["q_semantic"],
            "q_energy": info["q_energy"],
            "violation_flag": info["violation_flag"],
            "energy_cum": info["energy_cum"],
            "snr_db": info["snr_db"],
            "per": info.get("per", np.nan),
            "noise_std": info.get("noise_std", "N/A"),
            "semantic_weight": info.get("semantic_weight", 0.0),
            "P_index": p_idx,
            "B_index": bw_idx,
            "utility_reward": utility_reward,
            "lyapunov_reward": lyapunov_reward,
            "scaled_reward": scaled_reward,
            "q_prev_before": q_prev_before,
            "reward": utility_reward,
            "decision_latency_ms": decision_latency_ms,
            "mu_star": info.get("mu_star", 0.0),
            "mu_choice": info.get("mu_choice", 0.0),
            "run_tag": f"task3_raucb_plus_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}",
            "experiment_id": f"exp_{seed}",
            "scenario": info.get("scenario", scenario_val),
            "safety_overridden": safety_overridden, 
            "active_set_size": int(active_mask.sum()),
            "q_next_est": float(q_next_est)
        }
        results.append(log_entry)
        
        if t % 1000 == 0:
            print(f"  RA-UCB++ (Zoom+Safe) Seed {seed}: Step {t}/{HORIZON} (Active: {active_mask.sum()})")

    df = pd.DataFrame(results)
    filename = f"task3_raucb_plus_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}.csv"
    filepath = os.path.join(output_dir or OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {filepath}")


def run_lagrangian_ppo(seed, V_val=800.0, n_grid=10):
    if "LagrangianPPO" not in globals():
        print("LagrangianPPO class not found. Skipping...")
        return

    print(f"Running Lagrangian-PPO (Seed {seed}, V={V_val}, Grid={n_grid})...")
    env = SimulatedEnvironment(seed, V_val, w, n_grid)
    
    # 2D Action space grid (Discrete for PPO)
    candidates = build_candidates(n_grid=n_grid)
    # x1/x2 for logging
    x1 = np.linspace(0.1, 1.0, n_grid)
    x2 = np.linspace(0.1, 1.0, n_grid)
    
    state_dim = 4 # [q_semantic, q_energy, channel_quality, 0.0]
    action_dim = len(candidates)
    
    lr = 3e-4
    K_epochs = 10
    entropy_coef = 0.01
    hidden_dim = 64
    
    if env.scenario == "bandit_stationary_pure":
        lr = 0.02 # High LR for fast convergence
        K_epochs = 30 # Grind the batch
        entropy_coef = 0.05 # Start high for exploration
        hidden_dim = 0 # Linear policy
        gamma = 0.0 # Bandit setting (no future)
    else:
        gamma = 0.99
        
    agent = LagrangianPPO(state_dim=state_dim, action_dim=action_dim, constraint_threshold=0.1, lr=lr, gamma=gamma, K_epochs=K_epochs, entropy_coef=entropy_coef, hidden_dim=hidden_dim)
    
    results = []
    state, _, _, _, info = env.step([0.5, 0.5])
    q_prev = info["q_semantic"]
    
    memory = []
    batch_size = 200
    if env.scenario == "bandit_stationary_pure":
        batch_size = 1000 # Full batch (cover 100 arms) for accurate gradient (2 updates per epoch of arms)
    
    for t in range(1, HORIZON + 1):
        q_prev_before = q_prev

        t0 = time.perf_counter()
        
        # PPO Input State
        # In stationary bandit, state should be irrelevant or constant to avoid confusing the policy with drifting queues.
        ppo_state = state
        if env.scenario == "bandit_stationary_pure":
            ppo_state = np.array([0.0, 0.0, 0.5, 0.0]) # Fixed state

        # Select action
        # PPO agent takes state and returns action index (local index in ppo_candidates)
        action_idx, log_prob = agent.select_action(ppo_state)
        
        # Anneal entropy for stationary bandit to allow convergence (Log Regret)
        # If entropy is fixed > 0, regret is Linear (epsilon-greedy).
        if env.scenario == "bandit_stationary_pure" and t % 100 == 0:
            agent.entropy_coef = max(0.001, agent.entropy_coef * 0.95)

        t1 = time.perf_counter()
        decision_latency_ms = (t1 - t0) * 1000.0
        
        action = candidates[action_idx]
        
        # Execute
        next_state, reward, cost, done, info = env.step(action)
        
        # Unified Metrics Calculation
        utility_reward = reward
        
        q_curr = info["q_semantic"]
        drift_val, penalty_term, lyapunov_reward, scaled_reward = compute_lyapunov_metrics(V_val, utility_reward, q_prev_before, q_curr, 1.0)
        
        # Store in memory
        memory.append({
            'state': ppo_state,
            'action': action_idx,
            'reward': reward,
            'cost': cost,
            'log_prob': log_prob
        })
        
        state = next_state
        q_prev = q_curr
        
        # Update if batch full
        if len(memory) >= batch_size:
            agent.update(memory)
            memory = []
            
        action_idx = int(action_idx)
        # For logging, we need global indices if we want accurate heatmaps.
        # But here we just log discrete params.
        # If we subsetted, action_idx is local.
        # We can reconstruct bw_idx/p_idx from action.
        # action = [bw, p]
        # bw_idx = (bw - 0.1) / (1.0 - 0.1) * (n_grid - 1)
        # But floating point is tricky.
        # Just use action_idx (it won't match global grid index, but for stationary sanity we only care about Regret).
        # Regret uses utility_reward vs mu_star.
        # mu_star is from Env (global).
        # utility_reward is real reward.
        # If we picked global optimal, reward is max.
        
        bw_idx = -1
        p_idx = -1
        
        log_entry = {
            "algorithm": "lagrangian_ppo",
            "seed": seed,
            "t": t,
            "V": info["V"],
            "window": info["window"],
            "grid_density": info["grid_density"],
            "mos": info["mos"],
            "swer": info["swer"],
            "q_semantic": info["q_semantic"],
            "q_energy": info["q_energy"],
            "violation_flag": info["violation_flag"],
            "energy_cum": info["energy_cum"],
            "snr_db": info["snr_db"],
            "per": info.get("per", np.nan),
            "noise_std": info.get("noise_std", "N/A"),
            "semantic_weight": info.get("semantic_weight", 0.0),
            "P_index": p_idx,
            "B_index": bw_idx,
            "reward": reward,
            "q_prev_before": q_prev_before,
            "utility_reward": utility_reward,
            "lyapunov_reward": lyapunov_reward,
            "scaled_reward": scaled_reward,
            "decision_latency_ms": decision_latency_ms,
            "mu_star": info.get("mu_star", 0.0),
            "mu_choice": info.get("mu_choice", 0.0),
            "run_tag": f"task3_lagrangian_ppo_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}",
            "experiment_id": f"exp_{seed}",
            "scenario": info.get("scenario", "default")
        }
        results.append(log_entry)
        
        if t % 1000 == 0:
            print(f"  PPO Seed {seed}: Step {t}/{HORIZON}")
            
    df = pd.DataFrame(results)
    filename = f"task3_lagrangian_ppo_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {filepath}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    parser.add_argument("--V", type=float, default=V)
    parser.add_argument("--w", type=int, default=w)
    parser.add_argument("--g", type=int, default=g)
    parser.add_argument("--algorithms", type=str, nargs="*", default=["safeopt_gp","lagrangian_ppo","lyapunov_greedy_oracle","sw_ucb","safe_linucb","raucb_plus","best_fixed_arm_oracle"])
    parser.add_argument("--alpha_obj", type=float, default=0.5)

    parser.add_argument("--alpha_drift", type=float, default=0.5)
    parser.add_argument("--alpha_safety", type=float, default=0.5)
    parser.add_argument("--penalty_scale", type=float, default=1.0)
    parser.add_argument("--safety_lcb_threshold", type=float, default=0.8)
    parser.add_argument("--safety_lcb_threshold_ra", type=float, default=None)
    parser.add_argument("--safety_lcb_threshold_safe", type=float, default=None)
    parser.add_argument("--elim_start", type=int, default=200)
    parser.add_argument("--elim_interval", type=int, default=50)
    parser.add_argument("--score_trace_steps", type=int, default=0)
    parser.add_argument("--drift_variant", type=str, default="ucb")
    parser.add_argument("--verify_rng", action="store_true", help="Run RNG decoupling verification")
    parser.add_argument("--disable_ph", action="store_true", help="Disable Page-Hinkley mechanism in RUCB+")
    parser.add_argument("--scenario", type=str, default="default", help="Environment scenario (default: default)")
    parser.add_argument("--n_grid", type=int, default=10, help="Grid density for action space (default: 10)")
    args = parser.parse_args()

    if args.verify_rng:
        verify_rng_decoupling()
        exit(0)

    HORIZON = int(args.horizon)

    OUTPUT_DIR = args.output_dir
    SEEDS = list(args.seeds)
    V_VALUES = [float(args.V)]
    w = int(args.w)
    g = int(args.n_grid) # P0: Clean semantic. g now means n_grid in filename.
    # However, to avoid confusion, we ensure N_GRID uses the value that 'g' represents.
    # If the user passed --g 50, but --n_grid defaulted to 10, this is confusing.
    # We will enforce N_GRID = g if they are different, assuming g is the master intent if explicit.
    # But for simplicity and STRICT consistency with filename: g (filename) matches N_GRID (simulation).
    # We will overwrite g with args.g if provided explicitly and differs from default n_grid.
    # Actually, the user asked to fix "g50 but grid_density=10".
    # So we force N_GRID to be g.
    
    # Logic:
    # 1. args.g is the explicit 'g' param (default 50 or whatever env var)
    # 2. args.n_grid is the grid param (default 10)
    # 3. We want them to be the same.
    # 4. We prioritize args.g if it's not default (difficult to know) or just enforce g=args.g.
    
    # We will use args.g as the source of truth for "g" in filename AND "grid_density" in simulation.
    g = int(args.g) 
    N_GRID = g 
    
    ENABLE_PH = not args.disable_ph
    
    # Set Scenario Env Var
    os.environ["TASK3_SCENARIO"] = args.scenario

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Starting Experiments with Unified Parameters:")
    print(f"V values={V_VALUES}, w={w}, g={g}, N_GRID={N_GRID}, Horizon={HORIZON}, OUTPUT_DIR={OUTPUT_DIR}")
    print(f"Scenario={args.scenario}")
    
    # P0: Alignment Check Logging
    align_msg = []
    align_msg.append("\n=== P0 Metric & Column Check (Configuration) ===")
    align_msg.append(f"1. RUCB++ Update Metric: scaled_reward (Enforced in call)")
    align_msg.append(f"2. Best Fixed Arm Oracle Metric: scaled_reward (Enforced in call)")
    align_msg.append(f"3. CSV Columns: utility_reward, lyapunov_reward, scaled_reward (Verified in code)")
    align_msg.append(f"4. Global Candidates Grid: {N_GRID}x{N_GRID} ({N_GRID*N_GRID} arms)")
    
    # Append to existing alignment check file
    if os.environ.get("TASK3_ALIGN_LOG", "0") == "1":
        align_txt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fig01_regret_vs_time", "alignment_check.txt")
        with open(align_txt_path, "a") as f:
            for line in align_msg:
                print(line)
                f.write(line + "\n")
        print(f"Appended configuration check to {align_txt_path}")
    else:
        for line in align_msg:
            print(line)

    for V_val in V_VALUES:
        print(f"\n=== Running for V={V_val} ===")
        for seed in SEEDS:
            # Check if SafeOpt file exists
            if "safeopt_gp" in args.algorithms:
                safeopt_file = os.path.join(OUTPUT_DIR, f"task3_safeopt_gp_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}.csv")
                # Use check_file_validity
                if check_file_validity(safeopt_file, HORIZON):
                     print(f"[SKIP] SafeOpt-GP Seed {seed} already exists.")
                else:
                    try:
                        run_safeopt_gp(seed, V_val=V_val, n_grid=N_GRID)
                    except Exception as e:
                        print(f"Error running SafeOpt-GP seed {seed}: {e}")
                        traceback.print_exc()
                
            # PPO 
            if "lagrangian_ppo" in args.algorithms:
                try:
                    ppo_file = os.path.join(OUTPUT_DIR, f"task3_lagrangian_ppo_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}.csv")
                    if check_file_validity(ppo_file, HORIZON):
                        print(f"[SKIP] Lagrangian-PPO Seed {seed} already exists.")
                    else:
                        run_lagrangian_ppo(seed, V_val=V_val, n_grid=N_GRID)
                except Exception as e:
                    print(f"Error running Lagrangian-PPO seed {seed}: {e}")
                    traceback.print_exc()
    
            # Oracle
            if "lyapunov_greedy_oracle" in args.algorithms or "oracle" in args.algorithms:
                try:
                    oracle_file = os.path.join(OUTPUT_DIR, f"task3_lyapunov_greedy_oracle_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}.csv")
                    if check_file_validity(oracle_file, HORIZON):
                        print(f"[SKIP] Oracle Seed {seed} already exists.")
                    else:
                        run_lyapunov_greedy_oracle(seed, V_val=V_val, n_grid=N_GRID)
                except Exception as e:
                    print(f"Error running Lyapunov Greedy Oracle seed {seed}: {e}")
                    traceback.print_exc()
            
            # Best Fixed Arm Oracle
            if "best_fixed_arm_oracle" in args.algorithms:
                try:
                    best_fixed_file = os.path.join(OUTPUT_DIR, f"task3_best_fixed_arm_oracle_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}.csv")
                    if check_file_validity(best_fixed_file, HORIZON):
                        print(f"[SKIP] Best Fixed Arm Oracle Seed {seed} already exists.")
                    else:
                        # P0: Explicitly pass best_by="scaled_reward"
                        run_best_fixed_arm_oracle(seed, V_val=V_val, n_grid=N_GRID, best_by="scaled_reward")
                except Exception as e:
                    print(f"Error running Best Fixed Arm Oracle seed {seed}: {e}")
                    traceback.print_exc()

            # New Algorithms: SW-UCB, Safe-LinUCB, RA-UCB++
            if "sw_ucb" in args.algorithms:

                try:
                    sw_file = os.path.join(OUTPUT_DIR, f"task3_sw_ucb_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}.csv")
                    if check_file_validity(sw_file, HORIZON):
                        print(f"[SKIP] SW-UCB Seed {seed} already exists.")
                    else:
                        run_sw_ucb(seed, V_val=V_val, n_grid=N_GRID)
                except Exception as e:
                    print(f"Error running SW-UCB seed {seed}: {e}")
                    traceback.print_exc()
                
            if "safe_linucb" in args.algorithms:
                try:
                    lin_file = os.path.join(OUTPUT_DIR, f"task3_safe_linucb_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}.csv")
                    if check_file_validity(lin_file, HORIZON):
                        print(f"[SKIP] Safe-LinUCB Seed {seed} already exists.")
                    else:
                        run_safe_linucb(
                            seed,
                            V_val=V_val,
                            alpha_obj=args.alpha_obj,
                            safety_lcb_threshold=(args.safety_lcb_threshold_safe if args.safety_lcb_threshold_safe is not None else args.safety_lcb_threshold),
                            score_trace_steps=args.score_trace_steps,
                            output_dir=OUTPUT_DIR,
                            n_grid=N_GRID
                        )
                except Exception as e:
                    print(f"Error running Safe-LinUCB seed {seed}: {e}")
                    traceback.print_exc()
                
            if "raucb_plus" in args.algorithms:
                try:
                    ra_file = os.path.join(OUTPUT_DIR, f"task3_raucb_plus_V{V_val}_w{w}_g{g}_T{HORIZON}_s{seed}.csv")
                    if check_file_validity(ra_file, HORIZON):
                        print(f"[SKIP] RA-UCB++ Seed {seed} already exists.")
                    else:
                        run_raucb_plus(
                            seed,
                            V_val=V_val,
                            alpha_obj=args.alpha_obj,
                            alpha_drift=args.alpha_drift,
                            alpha_safety=args.alpha_safety,
                            penalty_scale=args.penalty_scale,
                            safety_lcb_threshold=(args.safety_lcb_threshold_ra if args.safety_lcb_threshold_ra is not None else args.safety_lcb_threshold),
                            elim_start=args.elim_start,
                            elim_interval=args.elim_interval,
                            score_trace_steps=args.score_trace_steps,
                            output_dir=OUTPUT_DIR,
                            drift_variant=args.drift_variant,
                            enable_ph=ENABLE_PH,
                            n_grid=N_GRID
                        )
                except Exception as e:
                    print(f"Error running RA-UCB++ seed {seed}: {e}")
                    traceback.print_exc()
                
    print("All experiments completed.")
