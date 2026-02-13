# Inputs Audit Report
## Code Version Hash
- fig01_regret_vs_time.py: 82875d95af5e93600abdf25496d71669
Summary: This figure reports cumulative Gap (baseline - algo) vs best_fixed_arm_oracle under scenario=bandit_dynamic_10000.
Note: Gap can be negative (Algo > Baseline). This is not a bug.

- run_experiments.STRICT2.py: 13e86211772f7e069c0a53cf347f7c6e

Date: 2026-01-28 22:38:55.219548
Context: input_dir=fig01_regret_vs_time/inputs, out_dir=fig01_regret_vs_time/trace, regret_mode=trajectory_gap, T=10000
## File Filtering
- [LOAD] task3_best_fixed_arm_oracle_V800.0_w512_g50_T10000_s0.csv (Baseline Seed 0)
- [LOAD] task3_best_fixed_arm_oracle_V800.0_w512_g50_T10000_s1.csv (Baseline Seed 1)
- [LOAD] task3_best_fixed_arm_oracle_V800.0_w512_g50_T10000_s2.csv (Baseline Seed 2)
- [LOAD] task3_lagrangian_ppo_V800.0_w512_g50_T10000_s0.csv (Algo lagrangian_ppo Seed 0)
- [LOAD] task3_lagrangian_ppo_V800.0_w512_g50_T10000_s1.csv (Algo lagrangian_ppo Seed 1)
- [LOAD] task3_lagrangian_ppo_V800.0_w512_g50_T10000_s2.csv (Algo lagrangian_ppo Seed 2)
- [LOAD] task3_raucb_plus_V800.0_w512_g50_T10000_s0.csv (Algo raucb_plus Seed 0)
- [LOAD] task3_raucb_plus_V800.0_w512_g50_T10000_s1.csv (Algo raucb_plus Seed 1)
- [LOAD] task3_raucb_plus_V800.0_w512_g50_T10000_s2.csv (Algo raucb_plus Seed 2)
- [LOAD] task3_safeopt_gp_V800.0_w512_g50_T10000_s0.csv (Algo safeopt_gp Seed 0)
- [LOAD] task3_safeopt_gp_V800.0_w512_g50_T10000_s1.csv (Algo safeopt_gp Seed 1)
- [LOAD] task3_safeopt_gp_V800.0_w512_g50_T10000_s2.csv (Algo safeopt_gp Seed 2)
- [LOAD] task3_safe_linucb_V800.0_w512_g50_T10000_s0.csv (Algo safe_linucb Seed 0)
- [LOAD] task3_safe_linucb_V800.0_w512_g50_T10000_s1.csv (Algo safe_linucb Seed 1)
- [LOAD] task3_safe_linucb_V800.0_w512_g50_T10000_s2.csv (Algo safe_linucb Seed 2)
- [LOAD] task3_sw_ucb_V800.0_w512_g50_T10000_s0.csv (Algo sw_ucb Seed 0)
- [LOAD] task3_sw_ucb_V800.0_w512_g50_T10000_s1.csv (Algo sw_ucb Seed 1)
- [LOAD] task3_sw_ucb_V800.0_w512_g50_T10000_s2.csv (Algo sw_ucb Seed 2)

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
- window: 512
- grid_density: 50
- scenario: bandit_dynamic_10000
- T: 10000
- seeds: [0, 1, 2]

## Trajectory Consistency
- [INFO] raucb_plus seed=0 col=q_semantic diff: mean=21.1166, max=130.3156
- [INFO] raucb_plus seed=0 col=q_energy diff: mean=0.0109, max=3.2137
- [INFO] raucb_plus seed=0 col=P_index diff: mean=4.6539, max=49.0000
- [INFO] raucb_plus seed=0 col=B_index diff: mean=17.2322, max=28.0000
- [INFO] raucb_plus seed=1 col=q_semantic diff: mean=15.4397, max=59.5812
- [INFO] raucb_plus seed=1 col=q_energy diff: mean=0.0066, max=2.4959
- [INFO] raucb_plus seed=1 col=P_index diff: mean=4.5984, max=49.0000
- [INFO] raucb_plus seed=1 col=B_index diff: mean=17.5901, max=35.0000
- [INFO] raucb_plus seed=2 col=q_semantic diff: mean=25.5363, max=121.3016
- [INFO] raucb_plus seed=2 col=q_energy diff: mean=0.0079, max=2.6407
- [INFO] raucb_plus seed=2 col=P_index diff: mean=4.8836, max=49.0000
- [INFO] raucb_plus seed=2 col=B_index diff: mean=17.5476, max=29.0000
- [INFO] safeopt_gp seed=0 col=q_semantic diff: mean=1218.8615, max=2925.2379
- [INFO] safeopt_gp seed=0 col=q_energy diff: mean=1.3385, max=16.1540
- [INFO] safeopt_gp seed=0 col=P_index diff: mean=24.0212, max=47.0000
- [INFO] safeopt_gp seed=0 col=B_index diff: mean=10.0982, max=28.0000
- [INFO] safeopt_gp seed=1 col=q_semantic diff: mean=3068.3827, max=7279.9360
- [INFO] safeopt_gp seed=1 col=q_energy diff: mean=7.3479, max=34.4693
- [INFO] safeopt_gp seed=1 col=P_index diff: mean=35.0509, max=49.0000
- [INFO] safeopt_gp seed=1 col=B_index diff: mean=13.9918, max=35.0000
- [INFO] safeopt_gp seed=2 col=q_semantic diff: mean=1852.0036, max=4309.5992
- [INFO] safeopt_gp seed=2 col=q_energy diff: mean=2.6279, max=22.2520
- [INFO] safeopt_gp seed=2 col=P_index diff: mean=26.7437, max=49.0000
- [INFO] safeopt_gp seed=2 col=B_index diff: mean=11.7891, max=29.0000
- [INFO] safe_linucb seed=0 col=q_semantic diff: mean=15.8918, max=131.2009
- [INFO] safe_linucb seed=0 col=q_energy diff: mean=0.0000, max=0.2146
- [INFO] safe_linucb seed=0 col=P_index diff: mean=0.0049, max=49.0000
- [INFO] safe_linucb seed=0 col=B_index diff: mean=18.6660, max=28.0000
- [INFO] safe_linucb seed=1 col=q_semantic diff: mean=12.1697, max=58.0810
- [INFO] safe_linucb seed=1 col=q_energy diff: mean=0.0000, max=0.2843
- [INFO] safe_linucb seed=1 col=P_index diff: mean=0.0049, max=49.0000
- [INFO] safe_linucb seed=1 col=B_index diff: mean=18.6420, max=35.0000
- [INFO] safe_linucb seed=2 col=q_semantic diff: mean=21.3990, max=120.9715
- [INFO] safe_linucb seed=2 col=q_energy diff: mean=0.0000, max=0.2073
- [INFO] safe_linucb seed=2 col=P_index diff: mean=0.0049, max=49.0000
- [INFO] safe_linucb seed=2 col=B_index diff: mean=18.8439, max=29.0000
- [INFO] sw_ucb seed=0 col=q_semantic diff: mean=21.0832, max=130.7063
- [INFO] sw_ucb seed=0 col=q_energy diff: mean=0.1783, max=6.2649
- [INFO] sw_ucb seed=0 col=P_index diff: mean=28.7417, max=49.0000
- [INFO] sw_ucb seed=0 col=B_index diff: mean=16.6768, max=28.0000
- [INFO] sw_ucb seed=1 col=q_semantic diff: mean=15.4872, max=59.3357
- [INFO] sw_ucb seed=1 col=q_energy diff: mean=0.1311, max=5.0722
- [INFO] sw_ucb seed=1 col=P_index diff: mean=26.1148, max=49.0000
- [INFO] sw_ucb seed=1 col=B_index diff: mean=14.0216, max=35.0000
- [INFO] sw_ucb seed=2 col=q_semantic diff: mean=25.4378, max=121.5452
- [INFO] sw_ucb seed=2 col=q_energy diff: mean=0.1691, max=4.2563
- [INFO] sw_ucb seed=2 col=P_index diff: mean=27.8438, max=49.0000
- [INFO] sw_ucb seed=2 col=B_index diff: mean=16.4256, max=29.0000
- [INFO] lagrangian_ppo seed=0 col=q_semantic diff: mean=1370.7690, max=2899.1173
- [INFO] lagrangian_ppo seed=0 col=q_energy diff: mean=0.0390, max=1.0889
- [INFO] lagrangian_ppo seed=0 col=P_index diff: mean=50.0000, max=50.0000
- [INFO] lagrangian_ppo seed=0 col=B_index diff: mean=22.0000, max=22.0000
- [INFO] lagrangian_ppo seed=1 col=q_semantic diff: mean=1929.6289, max=3682.5465
- [INFO] lagrangian_ppo seed=1 col=q_energy diff: mean=0.0348, max=0.9926
- [INFO] lagrangian_ppo seed=1 col=P_index diff: mean=50.0000, max=50.0000
- [INFO] lagrangian_ppo seed=1 col=B_index diff: mean=15.0000, max=15.0000
- [INFO] lagrangian_ppo seed=2 col=q_semantic diff: mean=1205.7788, max=2762.9440
- [INFO] lagrangian_ppo seed=2 col=q_energy diff: mean=0.0386, max=0.9871
- [INFO] lagrangian_ppo seed=2 col=P_index diff: mean=50.0000, max=50.0000
- [INFO] lagrangian_ppo seed=2 col=B_index diff: mean=21.0000, max=21.0000
Pass: All seeds have consistent exogenous trajectories (snr_db).

## Metric Definition Note
Current metric is 'Gap' (Baseline - Algo).
inst_gap = baseline_rewards[seed] - algo_data[algo][seed]['scaled_reward']

## Violation Summary
| Algo | Final Gap | Viol Rate | Min Inst Gap | Neg Frac |
|---|---:|---:|---:|---:|
| best_fixed_arm_oracle | 0.00 | 0.2295 | 0.0000 | 0.00% |
| raucb_plus | -12232.02 | 0.0000 | -3.9695 | 84.79% |
| safeopt_gp | 26040.41 | 0.8487 | -2.7800 | 15.72% |
| safe_linucb | -7787.21 | 0.2813 | -3.9585 | 69.86% |
| sw_ucb | 698.91 | 0.0013 | -4.0339 | 48.13% |
| lagrangian_ppo | 18713.45 | 0.9963 | -6.3361 | 11.66% |

- WARNING: safe_linucb has negative gap but high violation rate (28.13%).

## Safe-steps-only Gap Summary
| Rank | Algo | Safe-steps-only Final Gap | Safe Step Rate |
|---|---|---|---|
| 1 | raucb_plus | -12232.02 | 100.00% |
| 2 | safe_linucb | -10811.23 | 71.87% |
| 3 | safeopt_gp | -694.45 | 15.13% |
| 4 | lagrangian_ppo | -15.03 | 0.37% |
| 5 | sw_ucb | 690.23 | 99.87% |

## Strict-Safe Leaderboard
This answers: best safe algorithm under violation_rate<=1%.
| Rank | Algo | Final Gap |
|---|---|---|
| 1 | raucb_plus | -12232.02 |
| 2 | sw_ucb | 698.91 |

## Final Gap Ranking
| Rank | Algo | Final Mean Gap | Status |
|---|---|---|---|
| 1 | raucb_plus | -12232.02 | OK |
| 2 | safe_linucb | -7787.21 | Disqualified (Unsafe) |
| 3 | sw_ucb | 698.91 | OK |
| 4 | lagrangian_ppo | 18713.45 | Disqualified (Unsafe) |
| 5 | safeopt_gp | 26040.41 | Disqualified (Unsafe) |
