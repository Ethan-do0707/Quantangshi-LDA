# QTS B–C–D Finding-Level Robustness Report

## 1. Objective

本阶段只验证既有 C_K20 finding 对词汇过滤设置的稳健性，不重新选择模型、不重训 LDA、不改写已有 C effect validation 或 LOAO 状态。分析单位是语义场与具体时期 contrast 的组合，不是整个 Topic。
输入 `finding_level_bd_eligible=True` 实际得到 9 项；预期清单差异：{'expected_but_not_eligible': [], 'eligible_beyond_expected': []}。另有 4 项仅保留在排除状态附表，没有进入主验证。

## 2. Model Roles

B_K20：broader-vocabulary control（TF≥2、DF≥2）；C_K20：primary model（TF≥3、DF≥3）；D_K20：stricter-filter robustness control（TF≥5、DF≥3）。三模型保持独立，不平均 probability 或 effect。

## 3. Common Document Universe

| model | total_documents | common_documents | lost_documents | lost_poem_ids |
|---|---|---|---|---|
| B | 43016 | 43014 | 2 | 42083;42239 |
| C | 43015 | 43014 | 1 | 42083 |
| D | 43014 | 43014 | 0 |  |

共同集合严格按 poem_id 取 B∩C∩D；matrix_row 仅用于连接后的矩阵取值。逐时期/STRICT/CORE/EXPANDED 损失明细见 `BCD_common_universe_qc.csv`。
三套时期样本的总损失为 0。B 的一个原生文档没有冻结 C-based 历史 master/sample 行，因此明确标为 metadata unavailable，不重推时期或样本资格；它不进入任何时期样本。
C_ORIGINAL→C_COMMON 最大 |Δ变化|=4.72e-16，DOCUMENT_INTERSECTION_SENSITIVITY=0 行。原 C 区间使用旧 seed，COMMON 区间使用本阶段 seed，因此仅有 CI 差异不自动归因于交集。原 C 结果未覆盖。
B_NATIVE/D_NATIVE 仅作为辅助敏感性，最大 |Δ相对COMMON变化|=0。

## 4. Semantic Mapping Method

映射只读取既有 alignment。DIRECT 使用一个来源目标；BUNDLE 对来源明确 split 的全部目标概率直接求和，不使用相似度权重。single-best 取来源 candidate cosine 最高者，并与 bundle 分开保存。MERGED_PROXY 无法隔离 focal 语义场，结论最多为部分稳健。WEAK_PROXY 仅探索；unmatched 不强制替代。
来源 CSV 未包含 human_review_status 或 NEEDS_HUMAN_REVIEW 字段，因此输出明确保留 NOT_RECORDED_IN_SOURCE，不能把缺失当作人工批准。若将来来源出现复核标记或 LOW confidence，自动标记 ALIGNMENT_REVIEW_REQUIRED 并限制结论等级。MEDIUM bundle 亦不升级为高置信稳健。
同一 target 可能承载多个 C 语义场；这种重用在 overlapping_other_c_topics 中保留，既不强制一对一，也不把各 finding 相加。
统计：5,000 次时期分层作者 cluster bootstrap，seed=20260902；同一时期与样本使用相同作者抽签同时计算 B/C/D 与 POEM/AUTHOR。POEM 为每诗等权；AUTHOR 为先作者内均值、再时期内作者等权。只计算入选 contrasts，不建立新的 p-value family。
标准化效应沿用前阶段作者级 Hedges g 与 Cliff delta；raw Δ ratio 仅诊断，|C Δ|<0.005 时不计算。四时期 Pearson/Spearman 只作描述，不报告 p-value。
分类规则在配置中冻结：ROBUST 要求 B/D 的 CORE 两聚合同 C 同号、各目标至少一种 CORE 聚合的 CI 支持、三模型每个至少5/6规格维持 C 核心方向，且没有语义代理/待复核/明显bundle依赖/目标作者反转限制。稳定反向要求至少4/6数值反向且至少2/6反向CI支持。强DIRECT下两目标均稳定反向、无样本/聚合/CLR解释候选时才允许 NOT_REPLICATED。
CLR 仅在目标与 C raw 方向不同、bundle/single方向不同或原 C compositional warning 时触发。bundle 采用 log(概率和)−mean(log(完整20维概率))，不把多个 CLR 坐标相加；无真零时不加 pseudocount。

## 5. Overall Finding-level Robustness

ROBUST_ACROSS_BCD=4; PARTIALLY_ROBUST=5; MODEL_SENSITIVE=0; SEMANTICALLY_UNRESOLVED=0; NOT_REPLICATED=0

| finding_id | B_mapping_type | D_mapping_type | semantic_mapping_assessment | historical_effect_assessment | finding_level_robustness |
|---|---|---|---|---|---|
| T03_EH | BUNDLE | BUNDLE | ACCEPTABLE_HIGH_CONFIDENCE_BUNDLE | PARTIAL_OR_UNCERTAIN | PARTIALLY_ROBUST |
| T03_EL | BUNDLE | BUNDLE | ACCEPTABLE_HIGH_CONFIDENCE_BUNDLE | DIRECTIONALLY_SUPPORTED | ROBUST_ACROSS_BCD |
| T03_ML | BUNDLE | BUNDLE | ACCEPTABLE_HIGH_CONFIDENCE_BUNDLE | DIRECTIONALLY_SUPPORTED | ROBUST_ACROSS_BCD |
| T06_EL | BUNDLE | BUNDLE | ACCEPTABLE_HIGH_CONFIDENCE_BUNDLE | DIRECTIONALLY_SUPPORTED | ROBUST_ACROSS_BCD |
| T07_EL | BUNDLE | BUNDLE | LIMITED_PROXY_OR_MEDIUM_BUNDLE | PARTIAL_OR_UNCERTAIN | PARTIALLY_ROBUST |
| T08_EL | DIRECT | DIRECT | STRONG_OR_MODERATE_DIRECT | DIRECTIONALLY_SUPPORTED | ROBUST_ACROSS_BCD |
| T11_ML | BUNDLE | BUNDLE | ACCEPTABLE_HIGH_CONFIDENCE_BUNDLE | PARTIAL_OR_UNCERTAIN | PARTIALLY_ROBUST |
| T16_EL | DIRECT | DIRECT | STRONG_OR_MODERATE_DIRECT | PARTIAL_OR_UNCERTAIN | PARTIALLY_ROBUST |
| T19_EH | DIRECT | MERGED_PROXY | LIMITED_PROXY_OR_MEDIUM_BUNDLE | DIRECTIONALLY_SUPPORTED | PARTIALLY_ROBUST |

## 6. T03 E–H

语义场：朝廷礼乐、应制与颂圣。原 C=SUPPORTED；LOAO=CONCENTRATED_BUT_ROBUST（保持原状）。

| target_model | target_topic_ids | alignment_relation_type | alignment_confidence | mapping_testability |
|---|---|---|---|---|
| B | topic_08;topic_19 | split | HIGH | BUNDLE |
| D | topic_17;topic_08 | split | HIGH | BUNDLE |

| model | aggregation | raw_delta | CI95 | bootstrap_supported_direction | hedges_g_if_applicable | hedges_magnitude |
|---|---|---|---|---|---|---|
| B | POEM | -0.028135 | [-0.070661, 0.018794] | UNCERTAIN | NA |  |
| B | AUTHOR | -0.037109 | [-0.112437, 0.040826] | UNCERTAIN | -0.206177 | SMALL |
| C | POEM | -0.074561 | [-0.124936, -0.031400] | SUPPORTED_NEGATIVE | NA |  |
| C | AUTHOR | -0.065791 | [-0.139412, 0.006197] | UNCERTAIN | -0.383385 | SMALL |
| D | POEM | -0.082771 | [-0.127661, -0.030368] | SUPPORTED_NEGATIVE | NA |  |
| D | AUTHOR | -0.078153 | [-0.145045, -0.012533] | SUPPORTED_NEGATIVE | -0.494032 | SMALL |

样本方向：B:6/6;C:6/6;D:6/6；方向模式：MOSTLY_AGREEMENT。最终：**PARTIALLY_ROBUST**。CORE符号一致=True；B/D至少一种聚合的CI支持=False；样本符号=B:6/6;C:6/6;D:6/6；语义=ACCEPTABLE_HIGH_CONFIDENCE_BUNDLE。

## 7. T03 M–L

语义场：朝廷礼乐、应制与颂圣。原 C=STRONG；LOAO=CONCENTRATED_BUT_ROBUST（保持原状）。

| target_model | target_topic_ids | alignment_relation_type | alignment_confidence | mapping_testability |
|---|---|---|---|---|
| B | topic_08;topic_19 | split | HIGH | BUNDLE |
| D | topic_17;topic_08 | split | HIGH | BUNDLE |

| model | aggregation | raw_delta | CI95 | bootstrap_supported_direction | hedges_g_if_applicable | hedges_magnitude |
|---|---|---|---|---|---|---|
| B | POEM | -0.043747 | [-0.076540, -0.013725] | SUPPORTED_NEGATIVE | NA |  |
| B | AUTHOR | -0.062394 | [-0.119014, -0.005488] | SUPPORTED_NEGATIVE | -0.361429 | SMALL |
| C | POEM | -0.023422 | [-0.038643, -0.012284] | SUPPORTED_NEGATIVE | NA |  |
| C | AUTHOR | -0.065189 | [-0.105741, -0.026449] | SUPPORTED_NEGATIVE | -0.489785 | SMALL |
| D | POEM | -0.020732 | [-0.039673, -0.007563] | SUPPORTED_NEGATIVE | NA |  |
| D | AUTHOR | -0.064816 | [-0.119834, -0.005950] | SUPPORTED_NEGATIVE | -0.384777 | SMALL |

样本方向：B:6/6;C:6/6;D:6/6；方向模式：FULL_AGREEMENT。最终：**ROBUST_ACROSS_BCD**。CORE符号一致=True；B/D至少一种聚合的CI支持=True；样本符号=B:6/6;C:6/6;D:6/6；语义=ACCEPTABLE_HIGH_CONFIDENCE_BUNDLE。

## 8. T03 E–L

语义场：朝廷礼乐、应制与颂圣。原 C=STRONG；LOAO=CONCENTRATED_BUT_ROBUST（保持原状）。

| target_model | target_topic_ids | alignment_relation_type | alignment_confidence | mapping_testability |
|---|---|---|---|---|
| B | topic_08;topic_19 | split | HIGH | BUNDLE |
| D | topic_17;topic_08 | split | HIGH | BUNDLE |

| model | aggregation | raw_delta | CI95 | bootstrap_supported_direction | hedges_g_if_applicable | hedges_magnitude |
|---|---|---|---|---|---|---|
| B | POEM | -0.073799 | [-0.114083, -0.042579] | SUPPORTED_NEGATIVE | NA |  |
| B | AUTHOR | -0.069091 | [-0.141118, -0.001849] | SUPPORTED_NEGATIVE | -0.428275 | SMALL |
| C | POEM | -0.094469 | [-0.145956, -0.067823] | SUPPORTED_NEGATIVE | NA |  |
| C | AUTHOR | -0.125133 | [-0.192683, -0.063492] | SUPPORTED_NEGATIVE | -0.823045 | LARGE |
| D | POEM | -0.101930 | [-0.147279, -0.073270] | SUPPORTED_NEGATIVE | NA |  |
| D | AUTHOR | -0.103000 | [-0.172449, -0.031110] | SUPPORTED_NEGATIVE | -0.630690 | MODERATE |

样本方向：B:6/6;C:6/6;D:6/6；方向模式：FULL_AGREEMENT。最终：**ROBUST_ACROSS_BCD**。CORE符号一致=True；B/D至少一种聚合的CI支持=True；样本符号=B:6/6;C:6/6;D:6/6；语义=ACCEPTABLE_HIGH_CONFIDENCE_BUNDLE。

## 9. T06 E–L

语义场：老病身世、仕宦与人生感怀。原 C=STRONG；LOAO=CONCENTRATED_BUT_ROBUST（保持原状）。

| target_model | target_topic_ids | alignment_relation_type | alignment_confidence | mapping_testability |
|---|---|---|---|---|
| B | topic_14;topic_13 | split | HIGH | BUNDLE |
| D | topic_05;topic_11 | split | HIGH | BUNDLE |

| model | aggregation | raw_delta | CI95 | bootstrap_supported_direction | hedges_g_if_applicable | hedges_magnitude |
|---|---|---|---|---|---|---|
| B | POEM | 0.164425 | [0.102522, 0.233424] | SUPPORTED_POSITIVE | NA |  |
| B | AUTHOR | 0.087150 | [0.016786, 0.159670] | SUPPORTED_POSITIVE | 0.508467 | MODERATE |
| C | POEM | 0.093632 | [0.055772, 0.140732] | SUPPORTED_POSITIVE | NA |  |
| C | AUTHOR | 0.070234 | [0.006583, 0.130166] | SUPPORTED_POSITIVE | 0.480172 | SMALL |
| D | POEM | 0.151029 | [0.099200, 0.204850] | SUPPORTED_POSITIVE | NA |  |
| D | AUTHOR | 0.141073 | [0.084009, 0.200359] | SUPPORTED_POSITIVE | 1.017926 | LARGE |

样本方向：B:6/6;C:6/6;D:6/6；方向模式：FULL_AGREEMENT。最终：**ROBUST_ACROSS_BCD**。CORE符号一致=True；B/D至少一种聚合的CI支持=True；样本符号=B:6/6;C:6/6;D:6/6；语义=ACCEPTABLE_HIGH_CONFIDENCE_BUNDLE。

## 10. T07 E–L

语义场：相思离愁、夜月与琴声。原 C=SUPPORTED；LOAO=CONCENTRATED_BUT_ROBUST（保持原状）。

| target_model | target_topic_ids | alignment_relation_type | alignment_confidence | mapping_testability |
|---|---|---|---|---|
| B | topic_18;topic_04;topic_07 | split | MEDIUM | BUNDLE |
| D | topic_16;topic_13;topic_04 | split | MEDIUM | BUNDLE |

| model | aggregation | raw_delta | CI95 | bootstrap_supported_direction | hedges_g_if_applicable | hedges_magnitude |
|---|---|---|---|---|---|---|
| B | POEM | -0.006565 | [-0.077417, 0.064810] | UNCERTAIN | NA |  |
| B | AUTHOR | 0.010497 | [-0.072410, 0.094718] | UNCERTAIN | 0.054434 | VERY_SMALL |
| C | POEM | -0.054517 | [-0.086341, -0.028098] | SUPPORTED_NEGATIVE | NA |  |
| C | AUTHOR | -0.023288 | [-0.066629, 0.021173] | UNCERTAIN | -0.227063 | SMALL |
| D | POEM | -0.018233 | [-0.056035, 0.023663] | UNCERTAIN | NA |  |
| D | AUTHOR | -0.017647 | [-0.103237, 0.065403] | UNCERTAIN | -0.093126 | VERY_SMALL |

样本方向：B:2/6;C:6/6;D:4/6；方向模式：PARTIAL_AGREEMENT。最终：**PARTIALLY_ROBUST**。CORE符号一致=False；B/D至少一种聚合的CI支持=False；样本符号=B:2/6;C:6/6;D:4/6；语义=LIMITED_PROXY_OR_MEDIUM_BUNDLE；bundle与single-best存在方向依赖。

## 11. T08 E–L

语义场：山居寺院、僧侣与幽居。原 C=STRONG；LOAO=CONCENTRATED_BUT_ROBUST（保持原状）。

| target_model | target_topic_ids | alignment_relation_type | alignment_confidence | mapping_testability |
|---|---|---|---|---|
| B | topic_17 | strong_1to1 | HIGH | DIRECT |
| D | topic_12 | strong_1to1 | HIGH | DIRECT |

| model | aggregation | raw_delta | CI95 | bootstrap_supported_direction | hedges_g_if_applicable | hedges_magnitude |
|---|---|---|---|---|---|---|
| B | POEM | 0.035440 | [0.003594, 0.068879] | SUPPORTED_POSITIVE | NA |  |
| B | AUTHOR | 0.049952 | [0.010529, 0.093641] | SUPPORTED_POSITIVE | 0.512861 | MODERATE |
| C | POEM | 0.068992 | [0.042733, 0.095842] | SUPPORTED_POSITIVE | NA |  |
| C | AUTHOR | 0.041259 | [0.013838, 0.072469] | SUPPORTED_POSITIVE | 0.596758 | MODERATE |
| D | POEM | 0.075264 | [0.047216, 0.104456] | SUPPORTED_POSITIVE | NA |  |
| D | AUTHOR | 0.034524 | [0.001731, 0.071101] | SUPPORTED_POSITIVE | 0.438314 | SMALL |

样本方向：B:6/6;C:6/6;D:6/6；方向模式：FULL_AGREEMENT。最终：**ROBUST_ACROSS_BCD**。CORE符号一致=True；B/D至少一种聚合的CI支持=True；样本符号=B:6/6;C:6/6;D:6/6；语义=STRONG_OR_MODERATE_DIRECT。

## 12. T11 M–L

语义场：花香春情、女性与闺情。原 C=SUPPORTED；LOAO=CONCENTRATED_BUT_ROBUST（保持原状）。

| target_model | target_topic_ids | alignment_relation_type | alignment_confidence | mapping_testability |
|---|---|---|---|---|
| B | topic_04;topic_10 | split | HIGH | BUNDLE |
| D | topic_13;topic_06 | split | HIGH | BUNDLE |

| model | aggregation | raw_delta | CI95 | bootstrap_supported_direction | hedges_g_if_applicable | hedges_magnitude |
|---|---|---|---|---|---|---|
| B | POEM | 0.033238 | [-0.019077, 0.094459] | UNCERTAIN | NA |  |
| B | AUTHOR | 0.036981 | [-0.026789, 0.101773] | UNCERTAIN | 0.211496 | SMALL |
| C | POEM | 0.045053 | [0.005593, 0.092475] | SUPPORTED_POSITIVE | NA |  |
| C | AUTHOR | 0.034922 | [0.007336, 0.064434] | SUPPORTED_POSITIVE | 0.516946 | MODERATE |
| D | POEM | 0.032067 | [-0.009261, 0.083005] | UNCERTAIN | NA |  |
| D | AUTHOR | 0.019157 | [-0.025728, 0.064218] | UNCERTAIN | 0.145931 | VERY_SMALL |

样本方向：B:6/6;C:6/6;D:6/6；方向模式：MOSTLY_AGREEMENT。最终：**PARTIALLY_ROBUST**。CORE符号一致=True；B/D至少一种聚合的CI支持=False；样本符号=B:6/6;C:6/6;D:6/6；语义=ACCEPTABLE_HIGH_CONFIDENCE_BUNDLE。

## 13. T16 E–L

语义场：边塞征战与军旅。原 C=SUPPORTED；LOAO=AUTHOR_ROBUST（保持原状）。

| target_model | target_topic_ids | alignment_relation_type | alignment_confidence | mapping_testability |
|---|---|---|---|---|
| B | topic_06 | strong_1to1 | HIGH | DIRECT |
| D | topic_19 | strong_1to1 | HIGH | DIRECT |

| model | aggregation | raw_delta | CI95 | bootstrap_supported_direction | hedges_g_if_applicable | hedges_magnitude |
|---|---|---|---|---|---|---|
| B | POEM | -0.014926 | [-0.042031, 0.001730] | UNCERTAIN | NA |  |
| B | AUTHOR | -0.009024 | [-0.043138, 0.027383] | UNCERTAIN | -0.108820 | VERY_SMALL |
| C | POEM | -0.018122 | [-0.044250, -0.003434] | SUPPORTED_NEGATIVE | NA |  |
| C | AUTHOR | -0.037910 | [-0.076021, -0.003180] | SUPPORTED_NEGATIVE | -0.436072 | SMALL |
| D | POEM | -0.032160 | [-0.061750, -0.014132] | SUPPORTED_NEGATIVE | NA |  |
| D | AUTHOR | -0.054063 | [-0.086423, -0.021428] | SUPPORTED_NEGATIVE | -0.690197 | MODERATE |

样本方向：B:6/6;C:6/6;D:6/6；方向模式：MOSTLY_AGREEMENT。最终：**PARTIALLY_ROBUST**。CORE符号一致=True；B/D至少一种聚合的CI支持=False；样本符号=B:6/6;C:6/6;D:6/6；语义=STRONG_OR_MODERATE_DIRECT。

## 14. T19 E–H

语义场：闲居山林与隐逸生活。原 C=SUPPORTED；LOAO=CONCENTRATED_BUT_ROBUST（保持原状）。

| target_model | target_topic_ids | alignment_relation_type | alignment_confidence | mapping_testability |
|---|---|---|---|---|
| B | topic_13 | moderate_1to1 | MEDIUM | DIRECT |
| D | topic_12 | merge | MEDIUM | MERGED_PROXY |

| model | aggregation | raw_delta | CI95 | bootstrap_supported_direction | hedges_g_if_applicable | hedges_magnitude |
|---|---|---|---|---|---|---|
| B | POEM | 0.051616 | [0.022500, 0.074323] | SUPPORTED_POSITIVE | NA |  |
| B | AUTHOR | 0.044465 | [0.006571, 0.087073] | SUPPORTED_POSITIVE | 0.457828 | SMALL |
| C | POEM | 0.021961 | [0.002094, 0.053213] | SUPPORTED_POSITIVE | NA |  |
| C | AUTHOR | 0.036075 | [0.004755, 0.070367] | SUPPORTED_POSITIVE | 0.438679 | SMALL |
| D | POEM | 0.014102 | [0.003053, 0.038949] | SUPPORTED_POSITIVE | NA |  |
| D | AUTHOR | 0.046165 | [0.006398, 0.095445] | SUPPORTED_POSITIVE | 0.430509 | SMALL |

样本方向：B:6/6;C:6/6;D:5/6；方向模式：FULL_AGREEMENT。最终：**PARTIALLY_ROBUST**。CORE符号一致=True；B/D至少一种聚合的CI支持=True；样本符号=B:6/6;C:6/6;D:5/6；语义=LIMITED_PROXY_OR_MEDIUM_BUNDLE。

## 15. Priority Four

| finding_id | B_poem_delta | C_poem_delta | D_poem_delta | B_author_delta | C_author_delta | D_author_delta | finding_level_robustness |
|---|---|---|---|---|---|---|---|
| T03_EL | -0.073799 | -0.094469 | -0.101930 | -0.069091 | -0.125133 | -0.103000 | ROBUST_ACROSS_BCD |
| T06_EL | 0.164425 | 0.093632 | 0.151029 | 0.087150 | 0.070234 | 0.141073 | ROBUST_ACROSS_BCD |
| T08_EL | 0.035440 | 0.068992 | 0.075264 | 0.049952 | 0.041259 | 0.034524 | ROBUST_ACROSS_BCD |
| T16_EL | -0.014926 | -0.018122 | -0.032160 | -0.009024 | -0.037910 | -0.054063 | PARTIALLY_ROBUST |

优先级只影响展示与额外 Top1/Top3 检查，不改变统计阈值。四期 mean profile 同时展示 POEM 与 AUTHOR；中间路径或峰值不同不自动否定所选 contrast。作者检查按 contrast 涉及时期内的语义质量总和排名，删除作者整簇。

实际运行 32 行 B/D×Top1/Top3×POEM/AUTHOR 检查，符号反转 0 行，不可估计 0 行。仅检查方向，不把这一步称为完整目标模型LOAO。

## 16. Model-sensitive Findings

当前未发现达到预设系统反向阈值的 MODEL_SENSITIVE / NOT_REPLICATED finding。

B supports / D does not：无；D supports / B does not：T03_EH, T16_EL。supports 要求 CORE 两聚合同 C 同号且至少一个区间支持，不以单纯符号或语义相似替代。
D 支持而 B 不支持，至多与宽词表下语义质量更异质、严格过滤下信号更集中这两种可能性 consistent with；当前不能区分它们，也不能排除区间精度或语义范围差异。B 支持而 D 不支持的情况本批次没有观察到，不补造反例。这些只是方法解释候选，不构成过滤机制的因果证据。
条件CLR共 4 行，涉及 T07_EL；raw/CLR符号不同 1 行。详见 `clr_sensitivity.csv`，不以CLR替换主概率效应。

## 17. Semantically Unresolved Findings

没有 finding 被最终分类为 SEMANTICALLY_UNRESOLVED。仍须区分高置信 bundle、MEDIUM bundle 与 MERGED_PROXY 的解释限制。
来源 split/merge 与 single-best 依赖的逐项证据见 `split_single_best_sensitivity.csv` 和 `model_sensitivity_diagnostics.csv`；无法可靠同义比较时不称为未复现。

## 18. Robustness Ladder

| robustness_rank | finding_id | finding_level_robustness | eligible_for_next_stage | next_stage_note |
|---|---|---|---|---|
| 1 | T08_EL | ROBUST_ACROSS_BCD | True | 可进入人工解释与代表诗细读候选；须先确认来源mapping，仍不是历史结论 |
| 2 | T03_EL | ROBUST_ACROSS_BCD | True | 可进入人工解释与代表诗细读候选；须先确认来源mapping，仍不是历史结论 |
| 3 | T03_ML | ROBUST_ACROSS_BCD | True | 可进入人工解释与代表诗细读候选；须先确认来源mapping，仍不是历史结论 |
| 4 | T06_EL | ROBUST_ACROSS_BCD | True | 可进入人工解释与代表诗细读候选；须先确认来源mapping，仍不是历史结论 |
| 5 | T16_EL | PARTIALLY_ROBUST | False | 先处理映射或方法敏感性，不列入跨模型稳健主叙述 |
| 6 | T03_EH | PARTIALLY_ROBUST | False | 先处理映射或方法敏感性，不列入跨模型稳健主叙述 |
| 7 | T11_ML | PARTIALLY_ROBUST | False | 先处理映射或方法敏感性，不列入跨模型稳健主叙述 |
| 8 | T07_EL | PARTIALLY_ROBUST | False | 先处理映射或方法敏感性，不列入跨模型稳健主叙述 |
| 9 | T19_EH | PARTIALLY_ROBUST | False | 先处理映射或方法敏感性，不列入跨模型稳健主叙述 |

## 19. Limitations

- 跨模型 Topic 不是完全相同的潜变量；split/merge 降低精确可比性。
- raw probability magnitude 不在严格相同的语义尺度上；不按Δ相等要求复现。
- 作者时期不等于诗歌创作年代，冻结 metadata 的代表性与缺失偏差仍存在。
- B/D robustness 不是历史因果证据，词汇过滤本身改变语义分辨率。
- 四时期形状相关只有四个点，不作显著性推断；峰值不同不自动推翻具体 contrast。
- human_review_status 在来源缺失，不能声称本阶段获得新的人工语义确认；进入下一阶段前须核对映射与代表诗。
- C 既有统计/LOAO状态保持不变；本阶段只是独立跨模型证据层，止于方法结果，不开始文学史阐释。
