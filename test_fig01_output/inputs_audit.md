# Inputs Audit Report
## Code Version Hash
- fig01_regret_vs_time.py: 33efb475767cd9d533aa52f190013af0
Summary: This figure reports cumulative Pseudo-Regret (mu_star - mu_choice) under scenario=bandit_stationary_pure.
Note: This metric strictly measures theoretical regret. Should be non-negative by definition.

- run_experiments.STRICT2.py: 13e86211772f7e069c0a53cf347f7c6e

Date: 2026-02-08 12:49:37.762652
Context: input_dir=fig01_regret_vs_time/inputs_stationary_T10000, out_dir=test_fig01_output, regret_mode=pseudo_regret, T=10000
## File Filtering
Pseudo-Regret mode: Baseline files not required (using mu_star - mu_choice).
- [LOAD] task3_lagrangian_ppo_V800.0_w10000_g50_T10000_s0.csv (Algo lagrangian_ppo Seed 0)
- [LOAD] task3_lagrangian_ppo_V800.0_w10000_g50_T10000_s1.csv (Algo lagrangian_ppo Seed 1)
- [LOAD] task3_lagrangian_ppo_V800.0_w10000_g50_T10000_s2.csv (Algo lagrangian_ppo Seed 2)
- [LOAD] task3_raucb_plus_V800.0_w10000_g50_T10000_s0.csv (Algo raucb_plus Seed 0)
- [LOAD] task3_raucb_plus_V800.0_w10000_g50_T10000_s1.csv (Algo raucb_plus Seed 1)
- [LOAD] task3_raucb_plus_V800.0_w10000_g50_T10000_s2.csv (Algo raucb_plus Seed 2)
- [LOAD] task3_safeopt_gp_V800.0_w10000_g50_T10000_s0.csv (Algo safeopt_gp Seed 0)
- [LOAD] task3_safeopt_gp_V800.0_w10000_g50_T10000_s1.csv (Algo safeopt_gp Seed 1)
- [LOAD] task3_safeopt_gp_V800.0_w10000_g50_T10000_s2.csv (Algo safeopt_gp Seed 2)
- [LOAD] task3_safe_linucb_V800.0_w10000_g50_T10000_s0.csv (Algo safe_linucb Seed 0)
- [LOAD] task3_safe_linucb_V800.0_w10000_g50_T10000_s1.csv (Algo safe_linucb Seed 1)
- [LOAD] task3_safe_linucb_V800.0_w10000_g50_T10000_s2.csv (Algo safe_linucb Seed 2)
- [LOAD] task3_sw_ucb_V800.0_w10000_g50_T10000_s0.csv (Algo sw_ucb Seed 0)
- [LOAD] task3_sw_ucb_V800.0_w10000_g50_T10000_s1.csv (Algo sw_ucb Seed 1)
- [LOAD] task3_sw_ucb_V800.0_w10000_g50_T10000_s2.csv (Algo sw_ucb Seed 2)

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
- grid_density: 50
- scenario: bandit_stationary_pure
- T: 10000
- seeds: [0, 1, 2]

## Trajectory Consistency
Mode is pseudo_regret. Skipping Baseline-Algo trajectory consistency check.
Checking mu_star stability for Pseudo-Regret...
Pass: mu_star is constant across time (Stationary).

## Environment Parameters Audit
- safe_linucb: mu_star=4.99, grid_density=50, noise_std=0.05
- raucb_plus: mu_star=4.99, grid_density=50, noise_std=0.05
- sw_ucb: mu_star=4.99, grid_density=50, noise_std=0.05
- safeopt_gp: mu_star=4.99, grid_density=50, noise_std=0.05
- lagrangian_ppo: mu_star=4.99, grid_density=50, noise_std=0.05
Pass: All algorithms used consistent noise_std=0.05.

## Regret Definition Note
Current regret is 'Pseudo-Regret' = mu_star - mu_choice.
Strictly non-negative by definition (checked with tolerance 1e-9).

## Violation Summary
| Algo | Final Pseudo-Regret | Viol Rate | Min Inst Regret | Neg Frac |
|---|---:|---:|---:|---:|
| best_fixed_arm_oracle | 0.00 | N/A | 0.0000 | 0.00% |
| safe_linucb | 3.78 | N/A | 0.0000 | 0.00% |
| raucb_plus | 1715.05 | N/A | 0.0000 | 0.00% |
| sw_ucb | 6403.72 | N/A | 0.0000 | 0.00% |
| safeopt_gp | 5292.70 | N/A | 0.0631 | 0.00% |
| lagrangian_ppo | 10786.19 | N/A | 0.0031 | 0.00% |

## Pseudo-Regret Sanity (Stationary)
| Algo | Final | Final/log(T) | Final/log2(T) | Slope(Last 20%) | Check |
|---|---|---|---|---|---|
| safe_linucb | 3.78 | 0.41 | 0.28 | 0.0000 | OK |
| raucb_plus | 1715.05 | 186.21 | 129.07 | 0.0140 | OK |
| sw_ucb | 6403.72 | 695.28 | 481.93 | 0.1794 | OK |
| safeopt_gp | 5292.70 | 574.65 | 398.32 | 0.1752 | OK |
| lagrangian_ppo | 10786.19 | 1171.10 | 811.74 | 0.4705 | Exploding? |

## Final Regret Ranking
| Rank | Algo | Final Mean Regret |
|---|---|---|
| 1 | safe_linucb | 3.78 |
| 2 | raucb_plus | 1715.05 |
| 3 | safeopt_gp | 5292.70 |
| 4 | sw_ucb | 6403.72 |
| 5 | lagrangian_ppo | 10786.19 |
