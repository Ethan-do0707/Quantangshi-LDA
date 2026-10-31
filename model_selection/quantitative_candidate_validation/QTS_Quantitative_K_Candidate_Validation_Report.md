# QTS Quantitative K Candidate Validation Report

## 1. Objective

K=2–30 穷举扫描定位了 K12 全局 NPMI 峰和 K11–15 top-1% 带，但三个随机初始化不足以解释 0.0001–0.001 量级差异。本阶段只对 K11–15 与冻结 K20 reference 做十 seed、两套 prior 的量化复核；不命名主题，不改写 C_K20 taxonomy、主模型或文学史结果。

## 2. Candidate Set

主候选正好为 K11、K12、K13、K14、K15、K20。可选的 K21 未加入主排名，也没有重跑 K2–10、K16–19 或 K22–30。所有结果继续使用冻结 C corpus（TF>=3、DF>=3；43,015 首非空诗；35,063 词）与同一 34,412/8,603 split。

## 3. Why More Seeds?

每个 K×prior 使用同一组十个固定 seed：42、2026、20260830、7、17、101、31415、271828、20260907、12345。seed 不按结果删减。SD、符号一致性与跨 seed Topic matching 描述随机初始化变异，不是对历史诗歌总体的统计推断；本报告未做大规模 p-value 搜索，也未使用 bootstrap CI。

## 4. Original Prior Results

`ORIGINAL_PRIOR` 保持 sklearn 的 `doc_topic_prior=None`、`topic_word_prior=None`，有效 alpha=eta=1/K。十 seed mean NPMI 最高为 K12：0.0989849527 ± 0.0022424664。

| K | NPMI mean ± SD | Test PPL | Diversity | Cosine stability | Jaccard stability |
|---|---|---|---|---|---|
| K11 | 0.0980469325 ± 0.0022656981 | 10641.198 | 0.5355 | 0.7710 | 0.3484 |
| K12 | 0.0989849527 ± 0.0022424664 | 11128.378 | 0.5321 | 0.7596 | 0.3384 |
| K13 | 0.0971656845 ± 0.0036796786 | 11638.722 | 0.5100 | 0.7416 | 0.3263 |
| K14 | 0.0987508174 ± 0.0024775763 | 12173.465 | 0.4982 | 0.7373 | 0.3147 |
| K15 | 0.0986191615 ± 0.0020303656 | 12719.152 | 0.4857 | 0.7260 | 0.3147 |
| K20 | 0.0972602715 ± 0.0036027398 | 15421.878 | 0.4353 | 0.6849 | 0.2939 |

## 5. Fixed Prior Sensitivity

`FIXED_PRIOR_K20` 对所有 K 固定 alpha=eta=0.05。十 seed mean NPMI 最高为 K15：0.0984186340 ± 0.0024978917。这是一项 prior sensitivity，不是新正式模型，不能据此覆盖现有 K20。

| K | NPMI mean ± SD | Test PPL | Diversity | Cosine stability | Jaccard stability |
|---|---|---|---|---|---|
| K11 | 0.0972792139 ± 0.0019520646 | 11194.380 | 0.5173 | 0.7925 | 0.3608 |
| K12 | 0.0975979176 ± 0.0017827850 | 11659.852 | 0.5179 | 0.7791 | 0.3374 |
| K13 | 0.0976078959 ± 0.0030431203 | 12148.972 | 0.4962 | 0.7638 | 0.3372 |
| K14 | 0.0976872481 ± 0.0027211205 | 12616.959 | 0.4814 | 0.7543 | 0.3251 |
| K15 | 0.0984186340 ± 0.0024978917 | 13108.043 | 0.4757 | 0.7420 | 0.3176 |
| K20 | 0.0972602715 ± 0.0036027398 | 15421.878 | 0.4353 | 0.6849 | 0.2939 |

## 6. NPMI Candidate Comparison

K11–15 平台总判断为 `PRIOR_DEPENDENT_OR_PARTIAL_PLATEAU`。ORIGINAL_PRIOR 的 1% 带成员是 K11、K12、K14、K15，状态 `DISTINCT_PEAK`；FIXED_PRIOR_K20 的成员是 K12、K13、K14、K15，状态 `PARTIAL_PLATEAU`。绝对差 0.0005、0.0010 和相对全局峰 1% 三个尺度始终并列报告，不选其中一个冒充唯一显著性阈值。

K14 在 ORIGINAL_PRIOR 的十 seed 均值为 0.0987508174，相邻 K13/K15 分别为 0.0971656845/0.0986191615；它是否维持局部优势据这些固定均值如实判断，不外推为文学解释。

## 7. Held-out Perplexity

Perplexity 仅作为未见测试诗的 predictive-fit 维度。此前全扫描已显示该指标系统偏好低分辨率 K；本轮不以 perplexity 单独淘汰任何候选。train/test score 与 perplexity 均由同一 train-only 模型计算，测试集未参与拟合。

## 8. Cross-seed Topic Stability

每个 K×prior 的十 seed 形成 45 个 seed pair。对完整 Topic×Word 分布的 K×K cosine matrix 做 Hungarian 最大总相似度匹配；Top20 Jaccard 在同一配对上另算，二者不加权、不合成。`candidate_cross_seed_stability.csv` 同时保存均值、SD、最小值与全部 540 个真实 pair。

## 9. Topic Redundancy

模型内冗余分别由完整 Topic-word cosine 和 Top20 Jaccard 描述，并统计 cosine≥0.85/0.90 与 Jaccard≥0.50 的 pair 数。相对候选异常规则标出的 `REDUNDANCY_RISK` 为：无。该标签只表示定量复核优先级，不能直接称为 over-splitting。

## 10. Topic Prevalence

每个 train document-topic matrix 均计算各 Topic 的 mean probability、dominant-topic document count 与 dominant share；mean probability<0.01 只标为 `LOW_PREVALENCE_TOPIC`。逐 Topic 明细见 `topic_prevalence_diagnostics.csv`，模型级 min/max 与小 Topic 数见主表和汇总表；该阈值不自动判定模型失败。

## 11. K12 vs K11/13/14/15/20

| Prior | Comparison | Mean NPMI diff | K12 higher seeds | Practical equivalence |
|---|---|---|---|---|
| ORIGINAL_PRIOR | K12−K11 | +0.0009380202 | 6/10 | ABS_GT_0.0005|ABS_LE_0.0010|BOTH_WITHIN_1PCT_GLOBAL_MAX |
| ORIGINAL_PRIOR | K12−K13 | +0.0018192682 | 6/10 | ABS_GT_0.0005|ABS_GT_0.0010|NOT_BOTH_WITHIN_1PCT_GLOBAL_MAX |
| ORIGINAL_PRIOR | K12−K14 | +0.0002341354 | 5/10 | ABS_LE_0.0005|ABS_LE_0.0010|BOTH_WITHIN_1PCT_GLOBAL_MAX |
| ORIGINAL_PRIOR | K12−K15 | +0.0003657912 | 5/10 | ABS_LE_0.0005|ABS_LE_0.0010|BOTH_WITHIN_1PCT_GLOBAL_MAX |
| ORIGINAL_PRIOR | K12−K20 | +0.0017246812 | 7/10 | ABS_GT_0.0005|ABS_GT_0.0010|NOT_BOTH_WITHIN_1PCT_GLOBAL_MAX |
| FIXED_PRIOR_K20 | K12−K11 | +0.0003187037 | 6/10 | ABS_LE_0.0005|ABS_LE_0.0010|NOT_BOTH_WITHIN_1PCT_GLOBAL_MAX |
| FIXED_PRIOR_K20 | K12−K13 | -0.0000099782 | 6/10 | ABS_LE_0.0005|ABS_LE_0.0010|BOTH_WITHIN_1PCT_GLOBAL_MAX |
| FIXED_PRIOR_K20 | K12−K14 | -0.0000893305 | 5/10 | ABS_LE_0.0005|ABS_LE_0.0010|BOTH_WITHIN_1PCT_GLOBAL_MAX |
| FIXED_PRIOR_K20 | K12−K15 | -0.0008207164 | 3/10 | ABS_GT_0.0005|ABS_LE_0.0010|BOTH_WITHIN_1PCT_GLOBAL_MAX |
| FIXED_PRIOR_K20 | K12−K20 | +0.0003376461 | 7/10 | ABS_LE_0.0005|ABS_LE_0.0010|NOT_BOTH_WITHIN_1PCT_GLOBAL_MAX |

K12 在 ORIGINAL_PRIOR 是否仍为十 seed mean NPMI 最高：是；在 FIXED_PRIOR_K20 是否领先：否。其结构稳定性与冗余风险见上表和状态 `QUANT_COMPETITIVE|PRIOR_SENSITIVE`。

## 12. K15 vs K20

| Prior | Metric | K15 | K20 | Single-metric leader |
|---|---|---|---|---|
| ORIGINAL_PRIOR | NPMI ↑ | 0.0986191615 | 0.0972602715 | K15 |
| ORIGINAL_PRIOR | Perplexity ↓ | 12719.1520088074 | 15421.8781478156 | K15 |
| ORIGINAL_PRIOR | Diversity ↑ | 0.4856666667 | 0.4352500000 | K15 |
| ORIGINAL_PRIOR | Cosine stability ↑ | 0.7260138300 | 0.6849498424 | K15 |
| ORIGINAL_PRIOR | Jaccard stability ↑ | 0.3146935299 | 0.2939139282 | K15 |
| ORIGINAL_PRIOR | Redundancy cosine ↓ | 0.4471676936 | 0.4172198782 | K20 |
| FIXED_PRIOR_K20 | NPMI ↑ | 0.0984186340 | 0.0972602715 | K15 |
| FIXED_PRIOR_K20 | Perplexity ↓ | 13108.0434147445 | 15421.8781478156 | K15 |
| FIXED_PRIOR_K20 | Diversity ↑ | 0.4756666667 | 0.4352500000 | K15 |
| FIXED_PRIOR_K20 | Cosine stability ↑ | 0.7420215542 | 0.6849498424 | K15 |
| FIXED_PRIOR_K20 | Jaccard stability ↑ | 0.3176199035 | 0.2939139282 | K15 |
| FIXED_PRIOR_K20 | Redundancy cosine ↓ | 0.4624985575 | 0.4172198782 | K20 |

每一行只回答一个指标；没有把这些指标加权成总分。K20 的量化代价如实呈现，但其 reference 身份和冻结状态不受改变。

## 13. Prior Sensitivity

`PRIOR_SENSITIVE` 候选为：K11、K12、K13、K15。预注册规则是 NPMI rank 绝对变化至少 2，或是否进入 1% 高值带发生变化；pairwise order reversals 另存表，不因 fixed prior 结果重写原模型。

| K | Original rank | Fixed rank | Rank change | Order reversals | Status |
|---|---|---|---|---|---|
| K11 | 4 | 5 | 1 | 1 | PRIOR_SENSITIVE |
| K12 | 1 | 4 | 3 | 3 | PRIOR_SENSITIVE |
| K13 | 6 | 3 | -3 | 3 | PRIOR_SENSITIVE |
| K14 | 2 | 2 | 0 | 2 | PRIOR_ROBUST |
| K15 | 3 | 1 | -2 | 2 | PRIOR_SENSITIVE |
| K20 | 5 | 6 | 1 | 1 | PRIOR_ROBUST |

## 14. Seed Sensitivity

`SEED_SENSITIVE` 候选为：K20。规则使用每套 prior 内六候选的 Tukey 1.5×IQR 相对异常：NPMI SD 异常高，或 mean matched cosine/Jaccard 异常低。没有根据观察结果删除 seed 或事后移动阈值。

## 15. Pareto Comparison

ORIGINAL_PRIOR 非支配候选：K11、K12、K13、K14、K15、K20；FIXED_PRIOR_K20 非支配候选：K11、K12、K13、K14、K15、K20。维度为 NPMI↑、perplexity↓、diversity↑、两类 stability↑、redundancy↓、low-prevalence count↓；不做标准化总分。Pareto 非支配只表示在本组量化维度未被另一候选全面压倒，不等于最佳文学模型。

## 16. Quantitative Candidate Tier

| K | Status | Both-prior platform | Pareto any |
|---|---|---|---|
| K11 | QUANT_COMPETITIVE|PRIOR_SENSITIVE | False | True |
| K12 | QUANT_COMPETITIVE|PRIOR_SENSITIVE | True | True |
| K13 | QUANT_COMPETITIVE|PRIOR_SENSITIVE | False | True |
| K14 | QUANT_COMPETITIVE | True | True |
| K15 | QUANT_COMPETITIVE|PRIOR_SENSITIVE | True | True |
| K20 | QUANT_COMPETITIVE|SEED_SENSITIVE | False | True |

最终量化 shortlist 为 K12、K15、K20。角色槽位依次保留原先验 NPMI 领跑者、固定先验 NPMI 领跑者、双先验平台内无 seed/redundancy 警告的跨 seed stability 领跑者，并强制保留 K20 reference；重复角色合并，因此不会超过四个。

## 17. What Quantitative Metrics Cannot Decide

数值最高不等于文学语义最清楚。NPMI 只衡量 top-20 词的训练诗文档共现；perplexity 是 sklearn 变分近似下的 predictive fit；Topic Diversity、cosine/Jaccard、prevalence 阈值均不能判断某一语义拆分是否具有人文学意义。冻结词表来自全语料预处理，held-out 不是 document-completion perplexity。所有标签都必须接受代表诗、主题边界与语义可解释性的人工复核。

## 18. Recommendation for Semantic Review

下一阶段只建议对 K12、K15、K20 开展 Humanities Semantic Review，并与现有 K15/K20 审核材料并表。K20 必须继续作为 frozen reference。本任务到此停止，不自动运行语义审核。
