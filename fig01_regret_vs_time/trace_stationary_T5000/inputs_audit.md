# Inputs Audit Report
## Code Version Hash
- fig01_regret_vs_time.py: 33efb475767cd9d533aa52f190013af0
Summary: This figure reports cumulative Pseudo-Regret (mu_star - mu_choice) under scenario=bandit_stationary_pure.
Note: This metric strictly measures theoretical regret. Should be non-negative by definition.

- run_experiments.STRICT2.py: 0141cc0b9133cc7ead2b19df2cdb9e91

Date: 2026-01-28 08:30:31.081482
Context: input_dir=fig01_regret_vs_time/inputs_stationary_T5000, out_dir=fig01_regret_vs_time/trace_stationary_T5000, regret_mode=pseudo_regret, T=5000
## File Filtering
Pseudo-Regret mode: Baseline files not required (using mu_star - mu_choice).
- [LOAD] task3_lagrangian_ppo_V800.0_w5000_g50_T5000_s0.csv (Algo lagrangian_ppo Seed 0)
- [LOAD] task3_lagrangian_ppo_V800.0_w5000_g50_T5000_s1.csv (Algo lagrangian_ppo Seed 1)
- [LOAD] task3_lagrangian_ppo_V800.0_w5000_g50_T5000_s2.csv (Algo lagrangian_ppo Seed 2)
- [LOAD] task3_raucb_plus_V800.0_w5000_g50_T5000_s0.csv (Algo raucb_plus Seed 0)
- [LOAD] task3_raucb_plus_V800.0_w5000_g50_T5000_s1.csv (Algo raucb_plus Seed 1)
- [LOAD] task3_raucb_plus_V800.0_w5000_g50_T5000_s2.csv (Algo raucb_plus Seed 2)
- [LOAD] task3_safeopt_gp_V800.0_w5000_g50_T5000_s0.csv (Algo safeopt_gp Seed 0)
- [LOAD] task3_safeopt_gp_V800.0_w5000_g50_T5000_s1.csv (Algo safeopt_gp Seed 1)
- [LOAD] task3_safeopt_gp_V800.0_w5000_g50_T5000_s2.csv (Algo safeopt_gp Seed 2)
- [LOAD] task3_safe_linucb_V800.0_w5000_g50_T5000_s0.csv (Algo safe_linucb Seed 0)
- [LOAD] task3_safe_linucb_V800.0_w5000_g50_T5000_s1.csv (Algo safe_linucb Seed 1)
- [LOAD] task3_safe_linucb_V800.0_w5000_g50_T5000_s2.csv (Algo safe_linucb Seed 2)
- [LOAD] task3_sw_ucb_V800.0_w5000_g50_T5000_s0.csv (Algo sw_ucb Seed 0)
- [LOAD] task3_sw_ucb_V800.0_w5000_g50_T5000_s1.csv (Algo sw_ucb Seed 1)
- [LOAD] task3_sw_ucb_V800.0_w5000_g50_T5000_s2.csv (Algo sw_ucb Seed 2)

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
- window: 5000
- grid_density: 50
- scenario: bandit_stationary_pure
- T: 5000
- seeds: [0, 1, 2]

## Trajectory Consistency
Mode is pseudo_regret. Skipping Baseline-Algo trajectory consistency check.
Checking mu_star stability for Pseudo-Regret...
Pass: mu_star is constant across time (Stationary).

## Environment Parameters Audit
- raucb_plus: mu_star=4.99, grid_density=50, noise_std=0.05
- safeopt_gp: mu_star=4.99, grid_density=50, noise_std=0.05
- safe_linucb: mu_star=4.99, grid_density=50, noise_std=0.05
- sw_ucb: mu_star=4.99, grid_density=50, noise_std=0.05
- lagrangian_ppo: mu_star=4.99, grid_density=50, noise_std=0.05
Pass: All algorithms used consistent noise_std=0.05.

## Regret Definition Note
Current regret is 'Pseudo-Regret' = mu_star - mu_choice.
Strictly non-negative by definition (checked with tolerance 1e-9).

## Violation Summary
| Algo | Final Pseudo-Regret | Viol Rate | Min Inst Regret | Neg Frac |
|---|---:|---:|---:|---:|
| best_fixed_arm_oracle | 0.00 | N/A | 0.0000 | 0.00% |
| raucb_plus | 1532.37 | N/A | 0.0000 | 0.00% |
| safeopt_gp | 4428.00 | N/A | 0.0000 | 0.00% |
| safe_linucb | 450.00 | N/A | 0.0900 | 0.00% |
| sw_ucb | 7709.56 | N/A | 0.0000 | 0.00% |
| lagrangian_ppo | 9232.17 | N/A | 0.0000 | 0.00% |

## Pseudo-Regret Sanity (Stationary)
| Algo | Final | Final/log(T) | Final/log2(T) | Slope(Last 20%) | Check |
|---|---|---|---|---|---|
| raucb_plus | 1532.37 | 179.92 | 124.71 | 0.0375 | OK |
| safeopt_gp | 4428.00 | 519.89 | 360.36 | 1.1033 | High Slope? |
| safe_linucb | 450.00 | 52.83 | 36.62 | 0.0900 | OK |
| sw_ucb | 7709.56 | 905.18 | 627.42 | 1.4964 | High Slope? |
| lagrangian_ppo | 9232.17 | 1083.94 | 751.33 | 1.8439 | Exploding? |

## Final Regret Ranking
| Rank | Algo | Final Mean Regret |
|---|---|---|
| 1 | safe_linucb | 450.00 |
| 2 | raucb_plus | 1532.37 |
| 3 | safeopt_gp | 4428.00 |
| 4 | sw_ucb | 7709.56 |
| 5 | lagrangian_ppo | 9232.17 |
