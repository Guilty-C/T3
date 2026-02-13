import argparse
import os
import glob
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import hashlib

def get_file_hash(filepath):
    """Returns MD5 hash of the file."""
    if not os.path.exists(filepath):
        return "N/A (File not found)"
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def parse_args():
    parser = argparse.ArgumentParser(description="Fig01 Plotting Script")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory for CSVs")
    parser.add_argument("--T", type=int, required=True, help="Time horizon")
    parser.add_argument("--algorithms", nargs="+", required=True, help="List of algorithms to plot")
    parser.add_argument("--regret_mode", type=str, default="trajectory_gap", choices=["trajectory_gap", "pseudo_regret"], help="Regret mode")
    parser.add_argument("--baseline", type=str, default="best_fixed_arm_oracle", help="Baseline algorithm name")
    parser.add_argument("--metric", type=str, default="scaled_reward", help="Metric column name")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for traces/plots")
    return parser.parse_args()

def parse_filename_params(filename, T_target, algos=None, baseline="best_fixed_arm_oracle"):
    basename = os.path.basename(filename)
    if f"T{T_target}" not in basename: return None
    match = re.search(r"_s(\d+)\.csv", basename)
    if not match: return None
    seed = int(match.group(1))
    algo_name = None
    if baseline in basename:
         algo_name = baseline
    elif algos:
        for algo in algos:
            if f"task3_{algo}_" in basename:
                algo_name = algo
                break
    else:
        for algo in ["raucb_plus", "safeopt_gp", "safe_linucb", "sw_ucb", "lagrangian_ppo"]:
            if f"task3_{algo}_" in basename:
                algo_name = algo
                break
    if not algo_name: return None
    return {"algo": algo_name, "seed": seed, "T": T_target}

def validate_csv_params(df, filename, audit_lines, strict_mode=False, out_dir=".", fatal_fn=None):
    if fatal_fn is None:
        def fatal_fn(msg, code=2, dump=True):
            audit_lines.append(f"[FATAL] {msg}\n")
            raise RuntimeError(msg)
    cols = df.columns
    if "V" not in cols or "window" not in cols or "grid_density" not in cols:
        msg = f"{filename}: Missing V/window/grid_density columns."
        fatal_fn(msg, code=2)
    noise_val = "N/A"
    if "noise_std" in cols:
        noise_val = df["noise_std"].iloc[0]
    else:
        msg = f"{filename}: Missing 'noise_std' column."
        if strict_mode:
            msg += " Audit required."
            fatal_fn(msg, code=2)
        else:
            audit_lines.append(f"- [WARN] {msg} (Allowed in trajectory_gap)\n")
    v_val = float(df["V"].iloc[0])
    w_val = int(df["window"].iloc[0])
    g_val = int(df["grid_density"].iloc[0])
    return v_val, w_val, g_val, noise_val

def get_slope(series, start, end):
    if start < 1: start = 1
    if end > len(series): end = len(series)
    y_start = series.iloc[start-1]
    y_end = series.iloc[end-1]
    return (y_end - y_start) / (end - start)

def main():
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    audit_lines = []
    
    # --- P0: Code Version Hash ---
    audit_lines.append("# Inputs Audit Report\n")
    audit_lines.append("## Code Version Hash\n")
    this_script = os.path.abspath(__file__)
    hash_this = get_file_hash(this_script)
    audit_lines.append(f"- {os.path.basename(this_script)}: {hash_this}\n")
    run_exp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(this_script))), "run_experiments.STRICT2.py")
    hash_run = get_file_hash(run_exp_path)
    audit_lines.append(f"- run_experiments.STRICT2.py: {hash_run}\n")
    audit_lines.append("\n")
    audit_lines.append(f"Date: {pd.Timestamp.now()}\n")
    audit_lines.append(f"Context: input_dir={args.input_dir}, out_dir={args.out_dir}, regret_mode={args.regret_mode}, T={args.T}\n")
    
    def fatal_exit(msg, code=2, dump=True):
        print(f"[FATAL] {msg}")
        audit_lines.append(f"[FATAL] {msg}\n")
        with open(os.path.join(args.out_dir, "inputs_audit.md"), "w", encoding="utf-8") as f:
            f.writelines(audit_lines)
        sys.exit(code)
    
    required_algos = {"raucb_plus", "safeopt_gp", "safe_linucb", "sw_ucb", "lagrangian_ppo"}
    if set(args.algorithms) != required_algos:
         msg = f"Algorithm mismatch! Expected: {sorted(required_algos)}, Got: {sorted(args.algorithms)}"
         fatal_exit(msg)
    
    audit_lines.append("## File Filtering\n")
    param_registry = {"V": set(), "window": set(), "grid_density": set(), "scenario": set(), "seeds": set(), "baseline_seeds": set()}
    scenario_missing_count = 0
    
    baseline_rewards = {}
    baseline_dfs = {}

    if args.regret_mode != "pseudo_regret":
        baseline_files = glob.glob(os.path.join(args.input_dir, f"task3_{args.baseline}_*.csv"))
        for f in baseline_files:
            params = parse_filename_params(f, args.T, args.algorithms, args.baseline)
            if not params or params["algo"] != args.baseline: continue
            if params["seed"] in baseline_rewards: fatal_exit(f"Duplicate baseline file for seed {params['seed']}: {os.path.basename(f)}")
            try:
                df = pd.read_csv(f)
                if len(df) != args.T:
                     audit_lines.append(f"- [SKIP] {os.path.basename(f)}: Length {len(df)} != {args.T}\n")
                     continue
                csv_params = validate_csv_params(df, os.path.basename(f), audit_lines, strict_mode=False, out_dir=args.out_dir, fatal_fn=fatal_exit)
                if csv_params:
                    v_val, w_val, g_val, noise_val = csv_params
                    param_registry["V"].add(v_val)
                    param_registry["window"].add(w_val)
                    param_registry["grid_density"].add(g_val)
                if "scenario" in df.columns: param_registry["scenario"].add(df["scenario"].iloc[0])
                else: scenario_missing_count += 1
                baseline_rewards[params["seed"]] = df[args.metric]
                baseline_dfs[params["seed"]] = df
                param_registry["baseline_seeds"].add(params["seed"])
                param_registry["seeds"].add(params["seed"])
                audit_lines.append(f"- [LOAD] {os.path.basename(f)} (Baseline Seed {params['seed']})\n")
            except Exception as e:
                audit_lines.append(f"- [ERROR] {os.path.basename(f)}: {e}\n")
        if args.regret_mode == "trajectory_gap" and len(baseline_dfs) == 0:
            fatal_exit(f"No baseline CSV loaded for baseline={args.baseline}, T={args.T} in trajectory_gap mode.", code=2)
    else:
        audit_lines.append("Pseudo-Regret mode: Baseline files not required (using mu_star - mu_choice).\n")

    required_seeds = [0, 1, 2]
    if args.regret_mode == "trajectory_gap":
        loaded_baseline_seeds = sorted(list(param_registry["baseline_seeds"]))
        if loaded_baseline_seeds != required_seeds:
             audit_lines.append(f"Baseline seeds mismatch: required={required_seeds}, got={loaded_baseline_seeds}\n")
             fatal_exit(f"Baseline seeds incomplete/mismatch. Required: {required_seeds}, Got: {loaded_baseline_seeds}")
    else:
        param_registry["baseline_seeds"] = set(required_seeds)

    algo_data = {algo: {} for algo in args.algorithms}
    all_files = glob.glob(os.path.join(args.input_dir, "*.csv"))
    for f in all_files:
        params = parse_filename_params(f, args.T, args.algorithms, args.baseline)
        if not params or params["algo"] not in args.algorithms: continue
        try:
            df = pd.read_csv(f)
            if len(df) != args.T:
                 audit_lines.append(f"- [SKIP] {os.path.basename(f)}: Length {len(df)} != {args.T}\n")
                 continue
            if args.metric not in df.columns: fatal_exit(f"{os.path.basename(f)}: Metric column '{args.metric}' missing.")
            if "violation_flag" not in df.columns: fatal_exit(f"{os.path.basename(f)}: 'violation_flag' column missing.")
            if "snr_db" not in df.columns: fatal_exit(f"{os.path.basename(f)}: 'snr_db' column missing.")
            csv_params = validate_csv_params(df, os.path.basename(f), audit_lines, strict_mode=True, out_dir=args.out_dir, fatal_fn=fatal_exit)
            if csv_params:
                v_val, w_val, g_val, noise_val = csv_params
                param_registry["V"].add(v_val)
                param_registry["window"].add(w_val)
                param_registry["grid_density"].add(g_val)
            if "scenario" in df.columns: param_registry["scenario"].add(df["scenario"].iloc[0])
            else: scenario_missing_count += 1
            algo_data[params["algo"]][params["seed"]] = df
            param_registry["seeds"].add(params["seed"])
            audit_lines.append(f"- [LOAD] {os.path.basename(f)} (Algo {params['algo']} Seed {params['seed']})\n")
        except Exception as e:
            if args.regret_mode == "pseudo_regret": fatal_exit(f"Algorithm CSV read failed: {os.path.basename(f)} err={e}", code=2)
            else: audit_lines.append(f"- [ERROR] {os.path.basename(f)}: {e}\n")

    audit_lines.append("\n## Parameter Consistency Audit\n")
    conflicts = []
    for p in ["V", "window", "grid_density"]:
        if len(param_registry[p]) > 1: conflicts.append(f"{p} has multiple values: {param_registry[p]}")
    if len(param_registry["scenario"]) > 1: conflicts.append(f"scenario has multiple values: {param_registry['scenario']}")
    if conflicts: fatal_exit(f"Parameter conflicts detected: {'; '.join(conflicts)}")

    audit_lines.append("\n## Data Coverage Audit\n")
    loaded_algos = {a for a in required_algos if len(algo_data.get(a, {})) > 0}
    missing_data_algos = required_algos - loaded_algos
    for algo in sorted(loaded_algos):
        seeds = sorted(list(algo_data[algo].keys()))
        audit_lines.append(f"- {algo}: seeds={seeds} (n={len(seeds)})\n")
    if missing_data_algos: fatal_exit(f"Missing CSV data for algorithms: {sorted(missing_data_algos)}")
    
    audit_lines.append("\n## Seed Coverage Audit\n")
    expected_seeds = sorted(list(param_registry["baseline_seeds"]))
    for algo in sorted(loaded_algos):
        algo_seeds = sorted(list(algo_data[algo].keys()))
        missing = set(expected_seeds) - set(algo_seeds)
        extra = set(algo_seeds) - set(expected_seeds)
        audit_lines.append(f"- {algo}: expected={expected_seeds}, got={algo_seeds}, missing={sorted(list(missing))}, extra={sorted(list(extra))}\n")
        if missing: fatal_exit(f"{algo}: missing seeds {sorted(list(missing))} (seed coverage must be complete)")

    final_scenario = list(param_registry["scenario"])[0] if param_registry["scenario"] else "<unknown>"
    if final_scenario == "<unknown>": audit_lines.append(f"- [WARN] Scenario is unknown (missing column in CSVs).\n")
    if scenario_missing_count > 0: audit_lines.append(f"- [WARN] Scenario missing in {scenario_missing_count} files.\n")
    
    if args.regret_mode == "pseudo_regret":
         s_line = f"Summary: This figure reports cumulative Pseudo-Regret (mu_star - mu_choice) under scenario={final_scenario}.\n"
         n_line = "Note: This metric strictly measures theoretical regret. Should be non-negative by definition.\n\n"
    else:
         s_line = f"Summary: This figure reports cumulative Gap (baseline - algo) vs {args.baseline} under scenario={final_scenario}.\n"
         n_line = "Note: Gap can be negative (Algo > Baseline). This is not a bug.\n\n"
    audit_lines.insert(3, n_line)
    audit_lines.insert(3, s_line)
    audit_lines.append(f"- V: {list(param_registry['V'])[0] if param_registry['V'] else 'N/A'}\n")
    audit_lines.append(f"- window: {list(param_registry['window'])[0] if param_registry['window'] else 'N/A'}\n")
    audit_lines.append(f"- grid_density: {list(param_registry['grid_density'])[0] if param_registry['grid_density'] else 'N/A'}\n")
    audit_lines.append(f"- scenario: {final_scenario}\n")
    audit_lines.append(f"- T: {args.T}\n")
    audit_lines.append(f"- seeds: {sorted(list(param_registry['seeds']))}\n")

    audit_lines.append("\n## Trajectory Consistency\n")
    if args.regret_mode == "trajectory_gap":
        exogenous_cols = ["snr_db"]
        info_cols = ["q_semantic", "q_energy", "P_index", "B_index"]
        comparable_pairs_count = 0
        for algo in args.algorithms:
            if algo not in algo_data: continue
            for seed, algo_df in algo_data[algo].items():
                if seed not in baseline_dfs: fatal_exit(f"Trajectory Consistency Check Failed: {algo} seed={seed} missing baseline data.")
                base_df = baseline_dfs[seed]
                comparable_pairs_count += 1
                if "snr_db" not in base_df.columns or "snr_db" not in algo_df.columns:
                     audit_lines.append(f"- [WARN] snr_db missing for seed={seed}, algo={algo}. Skipping exogenous check.\n")
                     comparable_pairs_count -= 1
                     continue
                for col in exogenous_cols:
                    if col in base_df.columns and col in algo_df.columns:
                        max_abs_diff = (base_df[col] - algo_df[col]).abs().max()
                        if max_abs_diff > 1e-6: fatal_exit(f"Trajectory mismatch (Exogenous): seed={seed}, algo={algo}, col={col}, max_abs_diff={max_abs_diff}")
                for col in info_cols:
                    if col in base_df.columns and col in algo_df.columns:
                        diff = (base_df[col] - algo_df[col]).abs()
                        mean_diff = diff.mean()
                        max_diff = diff.max()
                        if max_diff > 1e-6: audit_lines.append(f"- [INFO] {algo} seed={seed} col={col} diff: mean={mean_diff:.4f}, max={max_diff:.4f}\n")
        if comparable_pairs_count > 0: audit_lines.append("Pass: All seeds have consistent exogenous trajectories (snr_db).\n")
        else: audit_lines.append("- [WARN] No comparable seed pairs for exogenous check.\n")
    else:
        audit_lines.append(f"Mode is {args.regret_mode}. Skipping Baseline-Algo trajectory consistency check.\n")
        audit_lines.append("Checking mu_star stability for Pseudo-Regret...\n")
        mu_star_stable = True
        for algo in args.algorithms:
            if algo not in algo_data: continue
            for seed, df in algo_data[algo].items():
                if "mu_star" in df.columns:
                    std_val = df["mu_star"].std()
                    if std_val > 1e-6:
                        audit_lines.append(f"- [WARN] {algo} seed={seed}: mu_star is NOT constant (std={std_val:.6f}). Non-stationary?\n")
                        mu_star_stable = False
                else:
                    fatal_exit(f"{algo} seed={seed}: Missing 'mu_star' column for pseudo_regret.")
        if mu_star_stable: audit_lines.append("Pass: mu_star is constant across time (Stationary).\n")
        
        audit_lines.append("\n## Environment Parameters Audit\n")
        env_params = []
        for algo in args.algorithms:
            if (algo not in algo_data) or (len(algo_data[algo]) == 0): fatal_exit(f"{algo}: No CSV loaded (cannot audit env params).", code=2)
            first_seed = list(algo_data[algo].keys())[0]
            df = algo_data[algo][first_seed]
            mu_val = df["mu_star"].iloc[0] if "mu_star" in df.columns else "N/A"
            grid_val = df["grid_density"].iloc[0] if "grid_density" in df.columns else "N/A"
            noise_val = df["noise_std"].iloc[0] if "noise_std" in df.columns else "N/A"
            if noise_val == "N/A": fatal_exit(f"{algo}: noise_std is N/A in CSV!")
            try: noise_str = f"{float(noise_val):.2f}"
            except: noise_str = str(noise_val)
            env_params.append(noise_str)
            audit_lines.append(f"- {algo}: mu_star={mu_val}, grid_density={grid_val}, noise_std={noise_val}\n")
        if len(set(env_params)) > 1: fatal_exit(f"Inconsistent noise_std across algorithms: {set(env_params)}")
        else: audit_lines.append(f"Pass: All algorithms used consistent noise_std={env_params[0] if env_params else 'None'}.\n")

    audit_lines.append("\n## Regret Definition Note\n")
    if args.regret_mode == "pseudo_regret":
        audit_lines.append("Current regret is 'Pseudo-Regret' = mu_star - mu_choice.\n")
        audit_lines.append("Strictly non-negative by definition (checked with tolerance 1e-9).\n")
    else:
        audit_lines.append("Current regret is calculated as 'Gap' (Baseline - Algo).\n")
        audit_lines.append("inst_regret = baseline_rewards[seed] - algo_data[algo][seed][args.metric]\n")

    audit_lines.append("\n## Violation Summary\n")
    if args.regret_mode == "trajectory_gap":
        audit_lines.append("| Algo | Final Gap | Viol Rate | Min Inst Gap | Neg Frac |\n")
    else:
        audit_lines.append("| Algo | Final Pseudo-Regret | Viol Rate | Min Inst Regret | Neg Frac |\n")
    audit_lines.append("|---|---:|---:|---:|---:|\n")
    
    base_viol_str = "N/A"
    if args.regret_mode == "pseudo_regret":
        for seed in baseline_dfs:
            df = baseline_dfs[seed]
            if "mu_star" not in df.columns or "mu_choice" not in df.columns: fatal_exit(f"Baseline (seed={seed}) missing mu_star/mu_choice in pseudo_regret mode.")
    base_viol_rates = []
    for seed, df in baseline_dfs.items():
         if "violation_flag" in df.columns: base_viol_rates.append(df["violation_flag"].mean())
    if base_viol_rates: base_viol_str = f"{np.mean(base_viol_rates):.4f}"
    audit_lines.append(f"| {args.baseline} | 0.00 | {base_viol_str} | 0.0000 | 0.00% |\n")

    if not any(algo_data.get(algo) for algo in args.algorithms): fatal_exit(f"No valid CSV matched the criteria (T={args.T}). Aborting.", code=1)
    
    plt.figure(figsize=(10, 6))
    colors = {"raucb_plus": "red", "safeopt_gp": "green", "safe_linucb": "blue", "sw_ucb": "orange", "lagrangian_ppo": "purple", "lyapunov_greedy_oracle": "gray"}
    agg_data = {}
    plotted_algos = set()
    warnings_buffer = []
    summary_stats = []

    for algo in args.algorithms:
        seeds = algo_data[algo].keys()
        if not seeds: continue
        regret_runs = []
        valid_seeds = []
        violation_rates = []
        min_inst_regret_all = []
        neg_frac_all = []

        for seed in seeds:
            if args.regret_mode == "trajectory_gap":
                if seed not in baseline_rewards: fatal_exit(f"Missing baseline for {algo} seed {seed} in trajectory_gap mode.")
                inst_regret = baseline_rewards[seed] - algo_data[algo][seed][args.metric]
                min_inst = inst_regret.min()
                neg_count = (inst_regret < 0).sum()
                neg_frac = neg_count / len(inst_regret)
                min_inst_regret_all.append(min_inst)
                neg_frac_all.append(neg_frac)
                cum_regret = inst_regret.cumsum()
                regret_runs.append(cum_regret)
                valid_seeds.append(seed)
            elif args.regret_mode == "pseudo_regret":
                df = algo_data[algo][seed]
                if "mu_star" not in df.columns or "mu_choice" not in df.columns: 
                    fatal_exit(f"{algo} seed {seed}: Missing mu_star/mu_choice.")
                
                # Check for NaNs
                if df["mu_star"].isnull().any() or df["mu_choice"].isnull().any():
                     fatal_exit(f"{algo} seed {seed}: Contains NaN values in mu_star or mu_choice.")

                mu_std = df["mu_star"].std()
                if mu_std > 1e-6: fatal_exit(f"{algo} seed {seed}: mu_star is NOT constant (std={mu_std:.6f}).")
                
                inst = df["mu_star"] - df["mu_choice"]
                
                # Check for NaNs in inst
                if inst.isnull().any():
                     fatal_exit(f"{algo} seed {seed}: Calculated pseudo-regret contains NaNs.")

                min_val = inst.min()
                if min_val < -1e-9:
                     viol_idx = inst[inst < -1e-9].index[0]
                     fatal_exit(f"{algo} seed {seed}: Negative pseudo-regret detected (min={min_val:.12f}). First violation at t={viol_idx+1}: {inst[viol_idx]:.12f}")
                cum_regret = inst.cumsum()
                
                if algo == "safe_linucb":
                     print(f"[DEBUG] safe_linucb seed {seed}: Loaded {len(df)} rows.")
                     print(f"[DEBUG] safe_linucb seed {seed}: Available columns: {list(df.columns)}")
                     print(f"[DEBUG] safe_linucb seed {seed}: Final Pseudo-Regret: {cum_regret.iloc[-1]}")

                regret_runs.append(cum_regret)
                valid_seeds.append(seed)
                min_inst_regret_all.append(min_val)
                neg_frac_all.append(0.0)
            
            if "violation_flag" in algo_data[algo][seed].columns:
                violation_rates.append(algo_data[algo][seed]["violation_flag"].mean())
        
        if not regret_runs: continue
        regret_matrix = pd.concat(regret_runs, axis=1)
        mean_regret = regret_matrix.mean(axis=1)
        std_regret = regret_matrix.std(axis=1)
        agg_data[algo] = mean_regret
        
        if violation_rates:
            avg_v = np.mean(violation_rates)
            final_regret = mean_regret.iloc[-1]
            avg_min_inst = np.mean(min_inst_regret_all)
            avg_neg_frac = np.mean(neg_frac_all)
            v_str = f"{avg_v:.4f}"
            if args.regret_mode == "pseudo_regret": v_str = "N/A"
            audit_lines.append(f"| {algo} | {final_regret:.2f} | {v_str} | {avg_min_inst:.4f} | {avg_neg_frac:.2%} |\n")
            summary_stats.append({"algo": algo, "final": float(final_regret), "viol": float(avg_v)})
            if final_regret < -100 and avg_v > 0.01: warnings_buffer.append(f"WARNING: {algo} has negative regret but high violation rate ({avg_v:.2%}).")
        
        t_axis = np.arange(1, args.T + 1)
        label = algo.replace("task3_", "").replace("_", "-").upper()
        if "RAUCB" in label: label = "RA-UCB++"
        if "PPO" in label: label = "Lagrangian-PPO"
        if "LYAPUNOV-GREEDY-ORACLE" in label: label = "Lyapunov-Oracle"
        
        # Scheme 2: Legend Annotation for safe_linucb
        if algo == "safe_linucb":
            label = f"{label} (final={mean_regret.iloc[-1]:.2f})"
            # Add marker at the end
            plt.plot(t_axis[-1], mean_regret.iloc[-1], marker='o', color=colors.get(algo, "black"), markersize=5)
            # Add text annotation
            # plt.annotate(f"{mean_regret.iloc[-1]:.2f}", (t_axis[-1], mean_regret.iloc[-1]), xytext=(5, 5), textcoords='offset points')

        color = colors.get(algo, "black")
        plt.plot(t_axis, mean_regret, label=label, color=color, linewidth=2)
        plt.fill_between(t_axis, mean_regret.to_numpy(dtype=float) - std_regret.to_numpy(dtype=float), mean_regret.to_numpy(dtype=float) + std_regret.to_numpy(dtype=float), color=color, alpha=0.15)
        plotted_algos.add(algo)

    if warnings_buffer:
        audit_lines.append("\n")
        for w in warnings_buffer: audit_lines.append(f"- {w}\n")

    plt.xlabel("Time Step (t)")
    if args.regret_mode == "pseudo_regret":
        plt.ylabel("Cumulative Pseudo-Regret")
        plt.title(f"Cumulative Pseudo-Regret (Stationary) (T={args.T})")
    else:
        plt.ylabel("Cumulative Trajectory Gap (baseline - algo)")
        plt.title(f"Cumulative Trajectory Gap (baseline - algo) vs Time (T={args.T})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)


    out_path = os.path.join(args.out_dir, "fig01_regret_vs_time.png")
    
    # Audit: Capture actual plot Y-limits
    ax = plt.gca()
    y0, y1 = ax.get_ylim()
    plotted_ymax = y1

    # Check for Unsafe Gain Likely
    if args.regret_mode == "trajectory_gap":
        safe_stats = [s for s in summary_stats if s['viol'] <= 0.01]
        if safe_stats:
            safe_leader = min(safe_stats, key=lambda x: x['final'])
            unsafe_stats = [s for s in summary_stats if s['viol'] > 0.01]
            for u in unsafe_stats:
                diff = safe_leader['final'] - u['final']
                if diff > 1e-9:
                    audit_lines.append(f"\nUnsafe Gain Likely: {u['algo']} (viol={u['viol']:.4f}) beats safe leader {safe_leader['algo']} by {diff:.2f} (gap decrease)\n")

    if args.regret_mode == "trajectory_gap":
        audit_lines.append("\n## Safe-steps-only Gap Summary\n")
        audit_lines.append("| Rank | Algo | Safe-steps-only Final Gap | Safe Step Rate |\n")
        audit_lines.append("|---|---|---|---|\n")
        safe_gaps = []
        for algo in args.algorithms:
            if algo not in algo_data: continue
            total_safe_gap = 0
            total_safe_rate = 0
            count = 0
            for seed, df in algo_data[algo].items():
                if seed not in baseline_rewards or "violation_flag" not in df.columns: continue
                v_flag = df["violation_flag"]
                metric = df[args.metric]
                base_r = baseline_rewards[seed]
                gap = base_r - metric
                safe_mask = (v_flag == 0)
                if safe_mask.any():
                    safe_gap = gap[safe_mask]
                    total_safe_gap += safe_gap.sum()
                total_safe_rate += safe_mask.mean()
                count += 1
            if count > 0:
                avg_safe_gap = total_safe_gap / count
                avg_safe_rate = total_safe_rate / count
                safe_gaps.append((algo, avg_safe_gap, avg_safe_rate))
        safe_gaps.sort(key=lambda x: x[1])
        for i, (algo, val, rate) in enumerate(safe_gaps):
            audit_lines.append(f"| {i+1} | {algo} | {val:.2f} | {rate:.2%} |\n")

        audit_lines.append("\n## Strict-Safe Leaderboard\n")
        audit_lines.append("This answers: best safe algorithm under violation_rate<=1%.\n")
        audit_lines.append("| Rank | Algo | Final Gap |\n")
        audit_lines.append("|---|---|---|\n")
        strict_safe_stats = [s for s in summary_stats if s['viol'] <= 0.01]
        strict_safe_stats.sort(key=lambda x: x['final'])
        for i, s in enumerate(strict_safe_stats):
            audit_lines.append(f"| {i+1} | {s['algo']} | {s['final']:.2f} |\n")

    if args.regret_mode == "pseudo_regret":
        audit_lines.append("\n## Pseudo-Regret Sanity (Stationary)\n")
        audit_lines.append("| Algo | Final | Final/log(T) | Final/log2(T) | Slope(Last 20%) | Check |\n")
        audit_lines.append("|---|---|---|---|---|---|\n")
        logT = np.log(args.T)
        log2T = np.log2(args.T)
        t_20pct = int(0.8 * args.T)
        for algo in args.algorithms:
            if algo not in agg_data: continue
            series = agg_data[algo]
            final_val = series.iloc[-1]
            ratio = final_val / logT
            ratio2 = final_val / log2T
            slope = get_slope(series, t_20pct, args.T)
            check = "OK"
            if slope > 0.5: check = "High Slope?"
            if ratio > 1000: check = "Exploding?"
            audit_lines.append(f"| {algo} | {final_val:.2f} | {ratio:.2f} | {ratio2:.2f} | {slope:.4f} | {check} |\n")

    print("\n" + "="*30)
    print("PHASE 6 FINAL REGRET RANKING")
    print("="*30)
    audit_lines.append("\n## Final Regret Ranking\n")
    if args.regret_mode == "trajectory_gap":
        audit_lines.append("| Rank | Algo | Final Mean Gap |\n")
    else:
        audit_lines.append("| Rank | Algo | Final Mean Regret |\n")
        # Add special line for safe_linucb visualization check
        if "safe_linucb" in agg_data:
             safe_final = agg_data["safe_linucb"].iloc[-1]
             # If we made inset
             inset_str = "N/A"
             if "inset_ymax" in locals():
                 inset_str = f"{inset_ymax:.2f}"
             
             # Main plot ylim is roughly max of all algos * margin
             # We can't easily get current ylim from plt without drawing, but we can estimate
             # max_all = max([s["final"] for s in summary_stats])
             # main_ymax = max_all * 1.05 # Default matplotlib margin is usually small
             
             # Use captured plotted_ymax if available
             main_ymax_str = f"{plotted_ymax:.2f}" if "plotted_ymax" in locals() else "N/A"

             audit_lines.append(f"safe_linucb_final={safe_final:.2f}, plotted_ymin=0, plotted_ymax={main_ymax_str}, inset_ymax={inset_str}\n\n")

    audit_lines.append("|---|---|---|\n")
    final_regrets = []
    for algo, series in agg_data.items():
        final_val = series.iloc[-1]
        final_regrets.append((algo, final_val))
    final_regrets.sort(key=lambda x: x[1])
    for i, (algo, val) in enumerate(final_regrets):
        rank = i + 1
        audit_lines.append(f"| {rank} | {algo} | {val:.2f} |\n")
        print(f"{rank}. {algo}: {val:.2f}")

    with open(os.path.join(args.out_dir, "inputs_audit.md"), "w", encoding="utf-8") as f:
        f.writelines(audit_lines)
    print(f"Generated audit: {os.path.join(args.out_dir, 'inputs_audit.md')}")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Generated plot: {out_path}")

if __name__ == "__main__":
    main()
