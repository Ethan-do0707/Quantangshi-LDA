# 《全唐诗》LDA 六模型综合评价与最终模型选择报告

## 1. Candidate Models

本报告比较 A_K20、B_K20、B_K30、C_K20、D_K20、D_K30。全程只读取既有训练结果与人工评分，不重新训练 LDA，也不修改 Topic 标签或人工分数。

### 实际读取文件

| 用途 | 路径 | 修改时间 |
| --- | --- | --- |
| humanEvaluation | D:/中大中文系工作相关/論文發表與投稿/全唐詩LDA/human_topic_review/human_topic_evaluation.xlsx | 2026-08-31 17:34:04 Asia/Singapore |
| configurationSummary | D:/中大中文系工作相关/論文發表與投稿/全唐詩LDA/reports/lda_sensitivity/configuration_summary.csv | 2026-08-30 19:37:38 Asia/Singapore |
| modelMetrics | D:/中大中文系工作相关/論文發表與投稿/全唐詩LDA/reports/lda_sensitivity/model_metrics.csv | 2026-08-30 19:37:38 Asia/Singapore |
| stabilityMetrics | D:/中大中文系工作相关/論文發表與投稿/全唐詩LDA/reports/lda_sensitivity/stability_metrics.csv | 2026-08-30 19:37:38 Asia/Singapore |

三 seed 模型级均值取自 `configuration_summary.csv`；selected-seed NPMI 取自 `model_metrics.csv`，仅作审计备注。人工指标取自最新六模型总表 `human_topic_evaluation.xlsx`，140 个 Topic 均完整评分。

## 2. Automatic Evaluation

| model_id | mean NPMI | Stability | Diversity | Human mean | High-quality | Mixed | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_K20 | 0.0685 | 0.2362 | 0.5300 | 2.5375 | 25.0% | 65.0% | 60.0% |
| B_K20 | 0.0996 | 0.2800 | 0.4725 | 3.7500 | 55.0% | 40.0% | 25.0% |
| B_K30 | 0.0948 | 0.2833 | 0.3967 | 3.0917 | 40.0% | 56.7% | 43.3% |
| C_K20 | 0.1014 | 0.3154 | 0.4567 | 3.6750 | 60.0% | 35.0% | 15.0% |
| D_K20 | 0.1058 | 0.3217 | 0.4942 | 3.6500 | 45.0% | 45.0% | 15.0% |
| D_K30 | 0.1034 | 0.2938 | 0.4289 | 3.3250 | 43.3% | 50.0% | 30.0% |

D_K20 的 mean NPMI（0.1058）和 Stability（0.3217）均为六模型最高；A_K20 的 Diversity 最高，但其余自动与人工指标显著落后，因此不能凭单一 Diversity 保留为主模型。

## 3. Human Evaluation

B_K20 人工均分最高（3.7500），C_K20 次之（3.6750）。C_K20 的高质量 Topic 比率最高（60.0%），mixed 比率最低（35.0%），noise 比率与 D_K20 并列最低（15.0%）。

自动指标最优模型（D_K20）与人工均分最高模型（B_K20）并不相同；C_K20 位于两者之间，不能把任一单项当作最终裁判。

## 4. Pareto Analysis

核心指标为 mean NPMI、Stability、Diversity、Mean Human Score、High-quality Ratio、Noise Ratio（低优）和 Mixed Ratio（低优）。

- Pareto front：A_K20、B_K20、C_K20、D_K20。
- 明确支配关系：C_K20 → B_K30；D_K20 → B_K30；D_K20 → D_K30；D_K30 → B_K30。

A_K20 因 Diversity 单项最高而仍在宽口径 Pareto front，但这只表示没有模型在七项指标上同时不差于它，不表示其适合作为主模型。

## 5. K Sensitivity

从 K=20 到 K=30 的变化：

| Corpus | Δ Human | Δ High-quality ratio | Δ Mixed ratio | Δ Noise ratio | Δ NPMI | Δ Stability |
| --- | --- | --- | --- | --- | --- | --- |
| B | -0.6583 | -15.0% | 16.7% | 18.3% | -0.0048 | 0.0033 |
| D | -0.3250 | -1.7% | 5.0% | 15.0% | -0.0024 | -0.0279 |

B_K30 相比 B_K20 仅获得 0.0033 的 Stability 微增，但人工均分下降 0.6583，高质量比例下降 15.0%，mixed/noise 分别上升 16.7% 和 18.3%。D_K30 相比 D_K20 连自动指标也整体下降。由此，K=30 主要增加拆分与噪声，没有稳定增加“有意义主题”。

## 6. TF/DF Sensitivity

| Model | TF/DF | Vocabulary | NPMI | Stability | Diversity | Human mean | Mixed | Noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_K20 | 1/1 | 123385 | 0.0685 | 0.2362 | 0.5300 | 2.5375 | 65.0% | 60.0% |
| B_K20 | 2/2 | 51982 | 0.0996 | 0.2800 | 0.4725 | 3.7500 | 40.0% | 25.0% |
| C_K20 | 3/3 | 35063 | 0.1014 | 0.3154 | 0.4567 | 3.6750 | 35.0% | 15.0% |
| D_K20 | 5/3 | 23123 | 0.1058 | 0.3217 | 0.4942 | 3.6500 | 45.0% | 15.0% |

A_K20 的 123,385 词型带来最高 Diversity，却伴随最低 NPMI、Stability 和人工均分，以及 60.0% noise，说明 TF≥1、DF≥1 引入大量低频残片与噪声。A→B 的过滤显著改善统计与人工质量。

B→C 后 NPMI、Stability、高质量比例继续改善，mixed/noise 继续下降，人工均分仅小幅下降 0.0750。C→D 后 NPMI、Stability 与 Diversity 上升，但高质量比例从 60.0% 降至 45.0%、mixed 从 35.0% 升至 45.0%，且词表由 35,063 降至 23,123。此处存在“统计指标提高，但文学信息可能损失”的明确 trade-off。

## 7. Weighting Sensitivity

先对每项指标做六模型 min-max normalization：

- 越高越好：((x-min x)/(max x-min x))
- 越低越好：((max x-x)/(max x-min x))
- 若六模型该指标完全相等，则统一记为 1，不人为制造差异。

三个场景使用任务指定权重，加权分只用于透明敏感性分析，不称为客观真值。

- Balanced：1. C_K20（0.8975）；2. D_K20（0.8673）；3. B_K20（0.8019）；4. D_K30（0.5991）；5. B_K30（0.4594）；6. A_K20（0.1500）
- Humanities-first：1. C_K20（0.9347）；2. B_K20（0.8456）；3. D_K20（0.8428）；4. D_K30（0.5894）；5. B_K30（0.4633）；6. A_K20（0.1000）
- Quantitative-first：1. D_K20（0.8911）；2. C_K20（0.8601）；3. B_K20（0.7420）；4. D_K30（0.6352）；5. B_K30（0.4714）；6. A_K20（0.1700）

## 8. Robustness

三场景始终位于前 3 的模型：B_K20、C_K20、D_K20。C_K20 三场景排名为 1/1/2，是唯一始终位于前 2 的模型。平均指标排名仅作辅助：D_K20=1.78；C_K20=1.89；B_K20=2.56；D_K30=4.00；B_K30=4.56；A_K20=4.89。

模型选择对权重存在有限敏感性：C_K20 与 D_K20 的第一、第二名会随“人文学解释优先”或“自动指标优先”互换，但前三名集合稳定为 C_K20、D_K20、B_K20；B_K30、D_K30、A_K20 从未进入前三。

## 9. Final Recommendation

**Recommended Primary Model: C_K20**

C_K20 同时具备第一梯队的 NPMI/Stability、最高高质量 Topic 比率、最低 mixed 比率、并列最低 noise 比率，并在三场景取得 1/1/2。它保留 35,063 个词型，比 D_K20 多 11,940 个 TF=3/4 词型，更适合在统计稳定性与唐诗低频文学信息之间取得平衡。

**Recommended Secondary / Robustness Model: D_K20**

D_K20 的 NPMI 与 Stability 最优，Quantitative-first 排名第一，适合作为严格过滤的稳健性模型；若 C_K20 与 D_K20 得出相同主题解释，可增强结论对 TF/DF 口径的稳健性。

| model_id | tier | role | 理由 |
| --- | --- | --- | --- |
| A_K20 | Tier 3 — Rejected models | Rejected as main candidate | 低频词过多引入大量残片、人名词和泛词噪声；单项 diversity 优势不足以抵消整体解释质量和稳定性缺陷。 |
| B_K20 | Tier 2 — Useful robustness/control models | Broad-vocabulary control | 保留为较宽词表对照，可检验 C/D 过滤是否遗漏文学信息，但不作为主模型。 |
| B_K30 | Tier 3 — Rejected models | Rejected as main candidate | K=30 的新增主题主要体现拆分和噪声，绝对数量增加并未转化为比例质量改善，且被 C_K20、D_K20 Pareto 支配。 |
| C_K20 | Tier 1 — Final candidates | Recommended Primary Model | 在统计质量、文学可解释性和词汇保留之间形成最稳健折中，且未表现出 K=30 的过度拆分。 |
| D_K20 | Tier 1 — Final candidates | Recommended Secondary / Robustness Model | 适合作为严格过滤、统计稳定性优先的稳健性模型，用以验证主模型结论是否依赖较丰富词表。 |
| D_K30 | Tier 3 — Rejected models | Rejected as main candidate | 在七项 Pareto 核心指标上被 D_K20 全面支配，K=30 没有带来可辩护的质量增益。 |

## 10. Methodological Implications

对《全唐诗》这类短文本、高词汇多样性且文学低频词丰富的语料：

1. 过滤过宽会提高词表规模和表面 Topic Diversity，却显著损害 NPMI、跨种子稳定性和人工可解释性。
2. 从 A/B 到 C 的适度 TF/DF 过滤主要去除残片与噪声，统计与人工质量总体同步改善。
3. 从 C 到 D 的严格过滤继续提高统计一致性，但高质量主题比例下降、mixed 上升，并损失大量 TF=3/4 词型，显示文学信息保留与数学稳定性之间的权衡。
4. 对当前语料，K=20 明显优于 K=30。增加 Topic 数并不自动增加有意义主题，必须以比例而非绝对数量评价。
5. 因最终结论在 C_K20 与 D_K20 之间随权重略有变化，最稳妥设计是以 C_K20 为主模型、D_K20 为严格过滤稳健性模型，而不是宣称存在脱离评价立场的唯一客观最优。
