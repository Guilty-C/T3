# Inputs Audit Report
## Code Version Hash
- fig01_regret_vs_time.py: cd7048894a35f8c84cc4f238de5b6c41
Summary: This figure reports cumulative Pseudo-Regret (mu_star - mu_choice) under scenario=bandit_stationary_pure.
Note: This metric strictly measures theoretical regret. Should be non-negative by definition.

- run_experiments.STRICT2.py: 0141cc0b9133cc7ead2b19df2cdb9e91

Date: 2026-01-27 22:55:19.372470
Context: input_dir=fig01_regret_vs_time/inputs_stationary_sanity_B, out_dir=fig01_regret_vs_time/trace_stationary_sanity_B, regret_mode=pseudo_regret, T=10000
## File Filtering
Pseudo-Regret mode: Baseline files not required (using mu_star - mu_choice).
- [LOAD] task3_lagrangian_ppo_V800.0_w10000_g10_T10000_s0.csv (Algo lagrangian_ppo Seed 0)
- [LOAD] task3_lagrangian_ppo_V800.0_w10000_g10_T10000_s1.csv (Algo lagrangian_ppo Seed 1)
- [LOAD] task3_lagrangian_ppo_V800.0_w10000_g10_T10000_s2.csv (Algo lagrangian_ppo Seed 2)
- [LOAD] task3_raucb_plus_V800.0_w10000_g10_T10000_s0.csv (Algo raucb_plus Seed 0)
- [LOAD] task3_raucb_plus_V800.0_w10000_g10_T10000_s1.csv (Algo raucb_plus Seed 1)
- [LOAD] task3_raucb_plus_V800.0_w10000_g10_T10000_s2.csv (Algo raucb_plus Seed 2)
- [LOAD] task3_safeopt_gp_V800.0_w10000_g10_T10000_s0.csv (Algo safeopt_gp Seed 0)
- [LOAD] task3_safeopt_gp_V800.0_w10000_g10_T10000_s1.csv (Algo safeopt_gp Seed 1)
- [LOAD] task3_safeopt_gp_V800.0_w10000_g10_T10000_s2.csv (Algo safeopt_gp Seed 2)
- [LOAD] task3_safe_linucb_V800.0_w10000_g10_T10000_s0.csv (Algo safe_linucb Seed 0)
- [LOAD] task3_safe_linucb_V800.0_w10000_g10_T10000_s1.csv (Algo safe_linucb Seed 1)
- [LOAD] task3_safe_linucb_V800.0_w10000_g10_T10000_s2.csv (Algo safe_linucb Seed 2)
- [LOAD] task3_sw_ucb_V800.0_w10000_g10_T10000_s0.csv (Algo sw_ucb Seed 0)
- [LOAD] task3_sw_ucb_V800.0_w10000_g10_T10000_s1.csv (Algo sw_ucb Seed 1)
- [LOAD] task3_sw_ucb_V800.0_w10000_g10_T10000_s2.csv (Algo sw_ucb Seed 2)

## Parameter Consistency Audit

## Data Coverage Audit
- lagrangian_ppo: seeds=[0, 1, 2] (n=3)
- raucb_plus: seeds=[0, 1, 2] (n=3)
- safe_linucb: seeds=[0, 1, 2] (n=3)
- safeopt_gp: seeds=[0, 1, 2] (n=3)
- sw_ucb: seeds=[0, 1, 2] (n=3)

## Seed Coverage Audit
- lagrangian_ppo: expected=[0, 1, 2], got=[0, 1, 2], missing=[], extra=[]
- raucb_plus: expected=[0, 1, 2], got=[0, 1, 2], missing=[], extra=[]
- safe_linucb: expected=[0, 1, 2], got=[0, 1, 2], missing=[], extra=[]
- safeopt_gp: expected=[0, 1, 2], got=[0, 1, 2], missing=[], extra=[]
- sw_ucb: expected=[0, 1, 2], got=[0, 1, 2], missing=[], extra=[]
- V: 800.0
- window: 10000
- grid_density: 10
- scenario: bandit_stationary_pure
- T: 10000
- seeds: [0, 1, 2]

## Trajectory Consistency
Mode is pseudo_regret. Skipping Baseline-Algo trajectory consistency check.
Checking mu_star stability for Pseudo-Regret...
Pass: mu_star is constant across time (Stationary).

## Environment Parameters Audit
- raucb_plus: mu_star=4.99, grid_density=10, noise_std=0.05
- safeopt_gp: mu_star=4.99, grid_density=10, noise_std=0.05
- safe_linucb: mu_star=4.99, grid_density=10, noise_std=0.05
- sw_ucb: mu_star=4.99, grid_density=10, noise_std=0.05
- lagrangian_ppo: mu_star=4.99, grid_density=10, noise_std=0.05
Pass: All algorithms used consistent noise_std=0.05.

## Regret Definition Note
Current regret is 'Pseudo-Regret' = mu_star - mu_choice.
Strictly non-negative by definition (checked with tolerance 1e-9).

## Violation Summary
| Algo | Final Pseudo-Regret | Viol Rate | Min Inst Regret | Neg Frac |
|---|---:|---:|---:|---:|
| best_fixed_arm_oracle | 0.00 | N/A | 0.0000 | 0.00% |
| raucb_plus | 455.31 | N/A | 0.0000 | 0.00% |
| safeopt_gp | 8859.69 | N/A | 0.0000 | 0.00% |
| safe_linucb | 900.00 | N/A | 0.0900 | 0.00% |
| sw_ucb | 2810.95 | N/A | 0.0000 | 0.00% |
| lagrangian_ppo | 18549.02 | N/A | 0.0000 | 0.00% |

## Reward-Violation Link
Skipped in Pseudo-Regret mode.

## Pseudo-Regret Sanity (Stationary)
| Algo | Final/log(T) | Slope(Last 20%) | Check |
|---|---|---|---|
| raucb_plus | 49.44 | 0.0026 | OK |
| safeopt_gp | 961.93 | 0.9225 | High Slope? |
| safe_linucb | 97.72 | 0.0900 | OK |
| sw_ucb | 305.20 | 0.1238 | OK |
| lagrangian_ppo | 2013.93 | 1.8628 | Exploding? |

## Baseline Semantics Note
Current regret is 'Pseudo-Regret' (Expected Utility Gap).
Baseline is 'mu_star' (theoretical optimal arm mean).
> CONCLUSION: This is a theoretical sanity-check metric; under standard finite-armed i.i.d. sub-Gaussian noise assumptions, many UCB-type methods achieve O(log T) expected regret.

## Slope Evidence

[5.1] RA-UCB++ Slope Analysis:
  Interval [1, 2000]: Slope = 0.1624
  Interval [2001, 6000]: Slope = 0.0285
  Interval [6001, 10000]: Slope = 0.0030

[5.2] RA-UCB++ Early Jump Analysis (t < 2000):
  Max Slope Window (width 200): [1, 201]
  Max Slope Value: 1.2700

[Comparison] Final Stage Slope (6001-10000):
  raucb_plus: 0.0030
  safeopt_gp: 0.9218
  safe_linucb: 0.0900
  sw_ucb: 0.1296
  lagrangian_ppo: 1.8512

## Final Regret Ranking
| Rank | Algo | Final Mean Regret |
|---|---|---|
| 1 | raucb_plus | 455.31 |
| 2 | safe_linucb | 900.00 |
| 3 | sw_ucb | 2810.95 |
| 4 | safeopt_gp | 8859.69 |
| 5 | lagrangian_ppo | 18549.02 |
