# Inputs Audit Report
Date: 2026-01-24 19:36:59.825797
Context: input_dir=fig01_regret_vs_time/inputs_stationary_T10000_noise005_clean, out_dir=fig01_regret_vs_time/trace_stationary_T10000_noise005_clean, regret_mode=pseudo_regret, T=10000
Summary: This figure reports cumulative Pseudo-Regret (mu_star - mu_choice) under Stationary scenario.
Note: This metric strictly measures theoretical regret. Should be non-negative by definition. Under standard finite-armed bandit assumptions (i.i.d. sub-Gaussian noise), many UCB-type methods achieve O(log T) expected regret.

## File Filtering
- [LOAD] task3_best_fixed_arm_oracle_V800.0_w300_g10_T10000_s0.csv (Baseline Seed 0)
- [LOAD] task3_best_fixed_arm_oracle_V800.0_w300_g10_T10000_s1.csv (Baseline Seed 1)
- [LOAD] task3_best_fixed_arm_oracle_V800.0_w300_g10_T10000_s2.csv (Baseline Seed 2)
- [LOAD] task3_lagrangian_ppo_V800.0_w300_g10_T10000_s0.csv (Algo lagrangian_ppo Seed 0)
- [LOAD] task3_lagrangian_ppo_V800.0_w300_g10_T10000_s1.csv (Algo lagrangian_ppo Seed 1)
- [LOAD] task3_lagrangian_ppo_V800.0_w300_g10_T10000_s2.csv (Algo lagrangian_ppo Seed 2)
- [LOAD] task3_raucb_plus_V800.0_w300_g10_T10000_s0.csv (Algo raucb_plus Seed 0)
- [LOAD] task3_raucb_plus_V800.0_w300_g10_T10000_s1.csv (Algo raucb_plus Seed 1)
- [LOAD] task3_raucb_plus_V800.0_w300_g10_T10000_s2.csv (Algo raucb_plus Seed 2)
- [LOAD] task3_safeopt_gp_V800.0_w300_g10_T10000_s0.csv (Algo safeopt_gp Seed 0)
- [LOAD] task3_safeopt_gp_V800.0_w300_g10_T10000_s1.csv (Algo safeopt_gp Seed 1)
- [LOAD] task3_safeopt_gp_V800.0_w300_g10_T10000_s2.csv (Algo safeopt_gp Seed 2)
- [LOAD] task3_safe_linucb_V800.0_w300_g10_T10000_s0.csv (Algo safe_linucb Seed 0)
- [LOAD] task3_safe_linucb_V800.0_w300_g10_T10000_s1.csv (Algo safe_linucb Seed 1)
- [LOAD] task3_safe_linucb_V800.0_w300_g10_T10000_s2.csv (Algo safe_linucb Seed 2)
- [LOAD] task3_sw_ucb_V800.0_w300_g10_T10000_s0.csv (Algo sw_ucb Seed 0)
- [LOAD] task3_sw_ucb_V800.0_w300_g10_T10000_s1.csv (Algo sw_ucb Seed 1)
- [LOAD] task3_sw_ucb_V800.0_w300_g10_T10000_s2.csv (Algo sw_ucb Seed 2)

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
| Algo | Final Regret | Avg Violation Rate | Negative Regret? |
|---|---|---|---|
| best_fixed_arm_oracle | 0.00 | N/A | NO |
| raucb_plus | 1120.38 | N/A | NO |
| safeopt_gp | 8859.69 | N/A | NO |
| safe_linucb | 900.00 | N/A | NO |
| sw_ucb | 13134.44 | N/A | NO |
| lagrangian_ppo | 18549.02 | N/A | NO |

## Reward-Violation Link
Skipped in Pseudo-Regret mode.

## Baseline Semantics Note
Current regret is 'Pseudo-Regret' (Expected Utility Gap).
Baseline is 'mu_star' (theoretical optimal arm mean).
> CONCLUSION: This is a theoretical sanity-check metric; under standard finite-armed i.i.d. sub-Gaussian noise assumptions, many UCB-type methods achieve O(log T) expected regret.

## Slope Evidence

[5.1] RA-UCB++ Slope Analysis:
  Interval [1, 2000]: Slope = 0.4990
  Interval [2001, 6000]: Slope = 0.0230
  Interval [6001, 10000]: Slope = 0.0068

[5.2] RA-UCB++ Early Jump Analysis (t < 2000):
  Max Slope Window (width 200): [1, 201]
  Max Slope Value: 1.8271

[Comparison] Final Stage Slope (6001-10000):
  raucb_plus: 0.0068
  safeopt_gp: 0.9218
  safe_linucb: 0.0900
  sw_ucb: 1.3153
  lagrangian_ppo: 1.8512
