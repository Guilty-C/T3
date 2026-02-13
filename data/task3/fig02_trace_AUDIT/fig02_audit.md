置顶：验收打分标准（必须输出到报告首行）
Score=10 的条件：
python -m compileall -q <涉及脚本> exit 0
只做“修改 + 生成 Fig02 审计报告”， 不要跑大规模训练/长实验 （除非已有 CSV）
审计报告 fig02_audit.md 满足：
Heatmap mass concentration：最大单元格占比 < 0.60 ；Top-2 合计 < 0.85
Heatmap entropy（按概率分布计算） > 1.0 （阈值可调，但必须不是接近 0）
Band curves：PER 或 sWER 三条 band 在 SNR=2~20 的区间内 max_gap > 0.02 （否则判定 band 无效）
Pareto：输出 nondominated 点数 > 3，且每个前沿点都通过“未被支配”检查
重新生成的 Fig02B 不允许“全黑 + 单点亮”；必须能看出“可行域内形成高密区”的结构趋势【 】
最后输出 ACCEPTANCE_JSON=... （包含上述指标与 PASS/FAIL）

# Fig02 Audit Report
## Fig02B: Heatmap Audit
### Algorithm: raucb_plus
- Max Cell Share: 0.3460 (Limit: < 0.60)
- Top-2 Share: 0.4885 (Limit: < 0.85)
- Entropy: 2.5360 (Limit: > 1.0)
**Result: PASS**

## Fig02A: Band Curves Audit
- Max Gap (SNR 2-20): 0.3389 (Limit: > 0.02)
- Curves Identical: True (Limit: False)
**Result: FAIL**

## Fig02C: Pareto Audit
- Non-dominated Points: 1 (Limit: > 3)
**Result: FAIL**
