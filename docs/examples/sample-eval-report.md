# Cortex 评测报告

- 时间：2026-08-23 17:28:00
- 消融设置：no_rerank=False no_memory=False single_agent=False

## BFCL 工具调用准确率

| 指标 | 值 |
|---|---|
| n | 10 |
| accuracy | 0.9 |

## GAIA 准确匹配（分级）

| 指标 | 值 |
|---|---|
| n | 14 |
| quasi_exact_match | 1.0 |
| L1 | 5/5 |
| L2 | 5/5 |
| L3 | 4/4 |

## LLM-as-Judge（与人工金标一致性）

| 指标 | 值 |
|---|---|
| n | 6 |
| agreement_with_gold | 0.3333 |
| win_rate_a | 0.3333 |
| kappa_vs_gold | 0.25 |
