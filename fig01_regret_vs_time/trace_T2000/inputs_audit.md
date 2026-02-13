# Inputs Audit Report
Date: 2026-01-24 00:02:31.831986
Summary: This figure reports cumulative trajectory gap vs best_fixed_arm_oracle under default scenario.
Note: logT behavior pertains to stationary pseudo-regret; negative values are allowed here.

## File Filtering
- [LOAD] task3_best_fixed_arm_oracle_V800.0_w300_g50_T2000_s0.csv (Baseline Seed 0)
- [LOAD] task3_best_fixed_arm_oracle_V800.0_w300_g50_T2000_s1.csv (Baseline Seed 1)
- [LOAD] task3_best_fixed_arm_oracle_V800.0_w300_g50_T2000_s2.csv (Baseline Seed 2)
- [LOAD] task3_lagrangian_ppo_V800.0_w300_g50_T2000_s0.csv (Algo lagrangian_ppo Seed 0)
- [LOAD] task3_lagrangian_ppo_V800.0_w300_g50_T2000_s1.csv (Algo lagrangian_ppo Seed 1)
- [LOAD] task3_lagrangian_ppo_V800.0_w300_g50_T2000_s2.csv (Algo lagrangian_ppo Seed 2)
- [LOAD] task3_raucb_plus_V800.0_w300_g50_T2000_s0.csv (Algo raucb_plus Seed 0)
- [LOAD] task3_raucb_plus_V800.0_w300_g50_T2000_s1.csv (Algo raucb_plus Seed 1)
- [LOAD] task3_raucb_plus_V800.0_w300_g50_T2000_s2.csv (Algo raucb_plus Seed 2)
- [LOAD] task3_safeopt_gp_V800.0_w300_g50_T2000_s0.csv (Algo safeopt_gp Seed 0)
- [LOAD] task3_safeopt_gp_V800.0_w300_g50_T2000_s1.csv (Algo safeopt_gp Seed 1)
- [LOAD] task3_safeopt_gp_V800.0_w300_g50_T2000_s2.csv (Algo safeopt_gp Seed 2)
- [LOAD] task3_safe_linucb_V800.0_w300_g50_T2000_s0.csv (Algo safe_linucb Seed 0)
- [LOAD] task3_safe_linucb_V800.0_w300_g50_T2000_s1.csv (Algo safe_linucb Seed 1)
- [LOAD] task3_safe_linucb_V800.0_w300_g50_T2000_s2.csv (Algo safe_linucb Seed 2)
- [LOAD] task3_sw_ucb_V800.0_w300_g50_T2000_s0.csv (Algo sw_ucb Seed 0)
- [LOAD] task3_sw_ucb_V800.0_w300_g50_T2000_s1.csv (Algo sw_ucb Seed 1)
- [LOAD] task3_sw_ucb_V800.0_w300_g50_T2000_s2.csv (Algo sw_ucb Seed 2)

## Trajectory Consistency
- [INFO] raucb_plus seed=0 col=q_semantic diff: mean=8.6539, max=45.6260
- [INFO] raucb_plus seed=0 col=q_energy diff: mean=0.0327, max=1.7109
- [INFO] raucb_plus seed=0 col=P_index diff: mean=2.8255, max=5.0000
- [INFO] raucb_plus seed=0 col=B_index diff: mean=2.7370, max=3.0000
- [INFO] raucb_plus seed=1 col=q_semantic diff: mean=98.8937, max=154.5814
- [INFO] raucb_plus seed=1 col=q_energy diff: mean=76.0494, max=213.2961
- [INFO] raucb_plus seed=1 col=P_index diff: mean=3.9760, max=5.0000
- [INFO] raucb_plus seed=1 col=B_index diff: mean=1.9085, max=3.0000
- [INFO] raucb_plus seed=2 col=q_semantic diff: mean=13.1045, max=123.6550
- [INFO] raucb_plus seed=2 col=q_energy diff: mean=0.0129, max=1.0707
- [INFO] raucb_plus seed=2 col=P_index diff: mean=1.2345, max=5.0000
- [INFO] raucb_plus seed=2 col=B_index diff: mean=2.7305, max=3.0000
- [INFO] safeopt_gp seed=0 col=q_semantic diff: mean=757.3203, max=1518.6369
- [INFO] safeopt_gp seed=0 col=q_energy diff: mean=18.7268, max=47.1423
- [INFO] safeopt_gp seed=0 col=P_index diff: mean=12.2500, max=34.0000
- [INFO] safeopt_gp seed=0 col=B_index diff: mean=22.1835, max=36.0000
- [INFO] safeopt_gp seed=1 col=q_semantic diff: mean=1125.2937, max=1977.0030
- [INFO] safeopt_gp seed=1 col=q_energy diff: mean=19.4227, max=53.1200
- [INFO] safeopt_gp seed=1 col=P_index diff: mean=12.2500, max=34.0000
- [INFO] safeopt_gp seed=1 col=B_index diff: mean=25.9500, max=37.0000
- [INFO] safeopt_gp seed=2 col=q_semantic diff: mean=910.3720, max=1690.6301
- [INFO] safeopt_gp seed=2 col=q_energy diff: mean=20.4833, max=51.9919
- [INFO] safeopt_gp seed=2 col=P_index diff: mean=12.2500, max=34.0000
- [INFO] safeopt_gp seed=2 col=B_index diff: mean=24.9000, max=36.0000
- [INFO] safe_linucb seed=0 col=q_semantic diff: mean=46.8705, max=228.1881
- [INFO] safe_linucb seed=0 col=q_energy diff: mean=0.0001, max=0.2146
- [INFO] safe_linucb seed=0 col=P_index diff: mean=13.9955, max=14.0000
- [INFO] safe_linucb seed=0 col=B_index diff: mean=11.8770, max=16.0000
- [INFO] safe_linucb seed=1 col=q_semantic diff: mean=23.9276, max=117.8450
- [INFO] safe_linucb seed=1 col=q_energy diff: mean=0.0001, max=0.2843
- [INFO] safe_linucb seed=1 col=P_index diff: mean=13.9955, max=14.0000
- [INFO] safe_linucb seed=1 col=B_index diff: mean=9.3475, max=17.0000
- [INFO] safe_linucb seed=2 col=q_semantic diff: mean=10.2046, max=55.8811
- [INFO] safe_linucb seed=2 col=q_energy diff: mean=0.0001, max=0.2073
- [INFO] safe_linucb seed=2 col=P_index diff: mean=13.9955, max=14.0000
- [INFO] safe_linucb seed=2 col=B_index diff: mean=13.6015, max=16.0000
- [INFO] sw_ucb seed=0 col=q_semantic diff: mean=47.2424, max=230.9374
- [INFO] sw_ucb seed=0 col=q_energy diff: mean=0.0857, max=3.5055
- [INFO] sw_ucb seed=0 col=P_index diff: mean=5.5830, max=10.0000
- [INFO] sw_ucb seed=0 col=B_index diff: mean=5.5040, max=16.0000
- [INFO] sw_ucb seed=1 col=q_semantic diff: mean=25.6673, max=118.8983
- [INFO] sw_ucb seed=1 col=q_energy diff: mean=0.0751, max=3.0927
- [INFO] sw_ucb seed=1 col=P_index diff: mean=6.4600, max=10.0000
- [INFO] sw_ucb seed=1 col=B_index diff: mean=4.7335, max=17.0000
- [INFO] sw_ucb seed=2 col=q_semantic diff: mean=17.7463, max=57.2516
- [INFO] sw_ucb seed=2 col=q_energy diff: mean=0.1078, max=3.4672
- [INFO] sw_ucb seed=2 col=P_index diff: mean=4.8850, max=10.0000
- [INFO] sw_ucb seed=2 col=B_index diff: mean=5.8790, max=16.0000
- [INFO] lagrangian_ppo seed=0 col=q_semantic diff: mean=230.8427, max=415.0893
- [INFO] lagrangian_ppo seed=0 col=q_energy diff: mean=0.0408, max=0.9600
- [INFO] lagrangian_ppo seed=0 col=P_index diff: mean=6.1305, max=14.0000
- [INFO] lagrangian_ppo seed=0 col=B_index diff: mean=7.1055, max=16.0000
- [INFO] lagrangian_ppo seed=1 col=q_semantic diff: mean=493.7033, max=799.0243
- [INFO] lagrangian_ppo seed=1 col=q_energy diff: mean=0.0450, max=0.8709
- [INFO] lagrangian_ppo seed=1 col=P_index diff: mean=6.1305, max=14.0000
- [INFO] lagrangian_ppo seed=1 col=B_index diff: mean=7.7895, max=17.0000
- [INFO] lagrangian_ppo seed=2 col=q_semantic diff: mean=175.1408, max=236.8129
- [INFO] lagrangian_ppo seed=2 col=q_energy diff: mean=0.0445, max=0.9287
- [INFO] lagrangian_ppo seed=2 col=P_index diff: mean=6.1305, max=14.0000
- [INFO] lagrangian_ppo seed=2 col=B_index diff: mean=7.1055, max=16.0000
Pass: All seeds have consistent exogenous trajectories (snr_db).

## Regret Definition Note
Current regret is calculated as 'Trajectory Gap' (Baseline - Algo).
Since the baseline (oracle) is fixed, an algorithm can occasionally outperform it
due to stochasticity, baseline semantics, and policy adaptability; whether violations help is evidenced in ‘Reward–Violation Link’.
See code around line 171: inst_regret = baseline_rewards[seed] - algo_data[algo][seed][args.metric]

## Violation Summary
| Algo | Final Regret | Avg Violation Rate | Negative Regret? |
|---|---|---|---|
| raucb_plus | 2674.44 | 0.4912 | NO |
| safeopt_gp | 4910.97 | 0.9950 | NO |
| safe_linucb | -1684.90 | 0.2297 | YES |
  WARNING: safe_linucb has negative regret but high violation rate (22.97%). Gains may come from unsafe actions.
| sw_ucb | 202.32 | 0.0013 | NO |
| lagrangian_ppo | 3217.12 | 0.9822 | NO |

## Reward-Violation Link
| Algo | Avg Reward (Safe) | Avg Reward (Viol) | Viol Rate | Corr(Regret, Viol) | Note |
|---|---|---|---|---|---|
| raucb_plus | 1.9489 | 0.9862 | 0.4912 | -0.4332 | Neg Corr: Violations reduce Regret. |
| safeopt_gp | 4.9451 | 0.0521 | 0.9950 | 0.1660 |  |
| safe_linucb | 4.0793 | 0.9399 | 0.2297 | 0.7593 |  |
| sw_ucb | 2.4326 | 0.9565 | 0.0013 | 0.0225 |  |
| lagrangian_ppo | 3.0875 | 0.8847 | 0.9822 | 0.1855 |  |

## Baseline Semantics Note
Current baseline is 'best_fixed_arm_oracle' trajectory. It is not strictly hindsight fixed arm optimal
(especially when system state is affected by actions).
> CONCLUSION: trajectory_gap can be negative and does not guarantee RA-UCB++ is minimal.
> To align with logT theory, use stationary + pseudo_regret.

## Slope Evidence

[5.1] RA-UCB++ Slope Analysis:
  Interval [1, 400]: Slope = 0.1097
  Interval [401, 1200]: Slope = 1.5766
  Interval [1201, 2000]: Slope = 1.7113

[5.2] RA-UCB++ Early Jump Analysis (t < 400):
  Max Slope Window (width 200): [1, 201]
  Max Slope Value: 0.2384

[Comparison] Final Stage Slope (1201-2000):
  raucb_plus: 1.7113
  safeopt_gp: 3.2810
  safe_linucb: -0.9837
  sw_ucb: 0.3768
  lagrangian_ppo: 2.0551
