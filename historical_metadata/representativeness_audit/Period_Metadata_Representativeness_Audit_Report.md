# Period Metadata Representativeness Audit Report

# 1. Objective

本报告只检验 author-period metadata 子样本相对于43,015首完整 C_K20 语料的代表性与可观测缺失关联；不比较初唐、盛唐、中唐、晚唐，不推断单篇作品年代，也不修改任何原数据。

权威输入：`D:/中大中文系工作相关/論文發表與投稿/全唐詩LDA/historical_metadata/master/C_K20_literary_history_master_with_author_history.csv`；SHA-256：`93fa45a5d848d0708fd1f4fb44289c26d8e98b7eec5b56bc440dc0de95de801f`。

# 2. Coverage

- KNOWN：479 / 2,528 位作者（18.95%），23,609 / 43,015 首诗（54.89%）。
- UNKNOWN：2,049 位作者（81.05%），19,406 首诗（45.11%）。

# 3. Author Productivity Bias

KNOWN 作者诗作数中位数为 3.0，UNKNOWN 为 1.0；均值分别为 49.288 与 9.471。Mann–Whitney U=640623.000，p=8.02e-30；Cliff's delta=0.3054（SMALL，正值表示 KNOWN 更高）。时期信息可用性明显偏向高产作者时，应将其视为选择偏差诊断，而不是文学史差异。

# 4. Topic Probability Bias

20 Topic 的最大 abs(SMD)=0.1517，平均 abs(SMD)=0.0412；abs(SMD)≥0.10 / 0.20 / 0.50 的 Topic 数分别为 3 / 0 / 0。Bootstrap 使用固定 seed 20260901、1,000次 multinomial-count resampling；效应量优先于大样本 p 值。

| topic_id | topic_label | topic_type | mean_known | mean_unknown | smd | bootstrap_ci_low | bootstrap_ci_high |
|---|---|---|---|---|---|---|---|
| topic_03 | 朝廷礼乐、应制与颂圣 | CLEAR | 0.037096 | 0.055903 | -0.151666 | -0.021098 | -0.016452 |
| topic_06 | 老病身世、仕宦与人生感怀 | CLEAR | 0.086589 | 0.068702 | 0.110437 | 0.014887 | 0.020637 |
| topic_05 | 君子伦理、仁义与政治教化 | BROAD | 0.055681 | 0.042616 | 0.107777 | 0.010639 | 0.015334 |
| topic_11 | 花香春情、女性与闺情 | CLEAR | 0.067617 | 0.078749 | -0.066749 | -0.014302 | -0.007734 |
| topic_20 | 饮酒酬赠、宴集与送别 | CLEAR | 0.062251 | 0.054381 | 0.060080 | 0.005269 | 0.010382 |
| topic_08 | 山居寺院、僧侣与幽居 | CLEAR | 0.077498 | 0.086274 | -0.051873 | -0.011880 | -0.005692 |
| topic_02 | 人生出处、忠义与悲慨（混合） | MIXED | 0.051392 | 0.045274 | 0.050102 | 0.003885 | 0.008443 |
| topic_10 | 江湖羁旅、送别与思归 | CLEAR | 0.098895 | 0.106078 | -0.038012 | -0.010821 | -0.003730 |
| topic_12 | 春水花鸟与自然风物 | BROAD | 0.040045 | 0.043417 | -0.030657 | -0.005421 | -0.001325 |
| topic_13 | 清景物候、咏物与杂咏（混合） | MIXED | 0.036967 | 0.039700 | -0.026507 | -0.004694 | -0.000690 |

# 5. Topic Type Bias

MIXED mass 的 KNOWN/UNKNOWN 均值为 0.133059/0.128669，NOISE 为 0.079926/0.079745。UNKNOWN 相对 KNOWN 的 MIXED+NOISE mass 差为 -0.004570；这里只说明缺失与 topic structure 的关联，不作文学意义判断。

| topic_type | mean_known | mean_unknown | smd | bootstrap_ci_low | bootstrap_ci_high | bias_level |
|---|---|---|---|---|---|---|
| CLEAR | 0.598514 | 0.616289 | -0.064416 | -0.023336 | -0.012359 | NEGLIGIBLE |
| BROAD | 0.188501 | 0.175297 | 0.061674 | 0.009029 | 0.017610 | NEGLIGIBLE |
| MIXED | 0.133059 | 0.128669 | 0.023870 | 0.000812 | 0.007885 | NEGLIGIBLE |
| NOISE | 0.079926 | 0.079745 | 0.001261 | -0.002428 | 0.002846 | NEGLIGIBLE |

# 6. Dominant Topic Bias

KNOWN × dominant topic 的 Chi-square=470.378，p=9.02e-88，Cramér's V=0.1046。Dominant Topic type 的 Cramér's V=0.0372。统计显著性不等同实质差异。

# 7. Topic Mixture / Entropy

Topic entropy 的 KNOWN/UNKNOWN 均值为 1.289659/1.249324，Cliff's delta=0.0657（NEGLIGIBLE）。Normalized entropy 的对应效应量为 0.0657。正值表示 KNOWN 更高；若为负值，只能说明 UNKNOWN 更混合的可观测关联。

# 8. CBDB / Identity Resolution Bias

| cbdb_match_status | author_count | period_known_author_ratio | poem_count | period_known_poem_ratio |
|---|---|---|---|---|
| AMBIGUOUS | 1076 | 0.000000 | 12350 | 0.000000 |
| CBDB_NOT_FOUND | 245 | 0.000000 | 592 | 0.000000 |
| DYNASTY_CONFLICT | 63 | 0.000000 | 411 | 0.000000 |
| MATCHED | 1114 | 0.429982 | 28381 | 0.831859 |
| NOT_ELIGIBLE | 30 | 0.000000 | 1281 | 0.000000 |

| cbdb_match_confidence | author_count | period_known_author_ratio | poem_count | period_known_poem_ratio |
|---|---|---|---|---|
| HIGH | 654 | 0.449541 | 17870 | 0.873531 |
| LOW | 38 | 0.289474 | 245 | 0.273469 |
| MEDIUM | 422 | 0.412322 | 10266 | 0.772648 |
| UNRESOLVED | 1414 | 0.000000 | 14634 | 0.000000 |

Period availability 与 CBDB identity resolution 的可观测状态明显相关时，可表述为 “Observed missingness is inconsistent with a simple random-missing pattern”，但本审计不声称证明 MCAR、MAR 或 MNAR。

# 9. STRICT / CORE / EXPANDED Samples

STRICT=KNOWN+HIGH 且排除 cross-period/boundary；CORE=KNOWN+(HIGH/MEDIUM) 且排除 cross-period/boundary；EXPANDED=全部 KNOWN。三者只建立标签，不删除或修改原数据。

| sample_name | poem_count | author_count | max_abs_topic_smd | mean_abs_topic_smd | entropy_effect_size | dominant_topic_cramers_v | overall_bias_level |
|---|---|---|---|---|---|---|---|
| STRICT | 8722 | 180 | 0.153340 | 0.034737 | 0.064226 | 0.068749 | LOW |
| CORE | 10841 | 207 | 0.113898 | 0.037108 | 0.056498 | 0.070630 | LOW |
| EXPANDED | 23609 | 479 | 0.072041 | 0.018736 | 0.029661 | 0.045979 | LOW |

按预先登记的描述性 bias score，EXPANDED 在三者中最接近 full corpus；这不表示其没有缺失偏差。

# 10. High-risk Topics

| topic_id | topic_label | topic_type | mean_known | mean_unknown | smd | risk_level | risk_reason |
|---|---|---|---|---|---|---|---|
| topic_10 | 江湖羁旅、送别与思归 | CLEAR | 0.098895 | 0.106078 | -0.038012 | LOW | UNKNOWN_OVERREPRESENTED;ABS_SMD=0.038;DOMINANT_RATIO_DIFF=-0.009;BOOTSTRAP_CI_EXCLUDES_ZERO |
| topic_12 | 春水花鸟与自然风物 | BROAD | 0.040045 | 0.043417 | -0.030657 | LOW | UNKNOWN_OVERREPRESENTED;ABS_SMD=0.031;DOMINANT_RATIO_DIFF=-0.003;BOOTSTRAP_CI_EXCLUDES_ZERO |
| topic_13 | 清景物候、咏物与杂咏（混合） | MIXED | 0.036967 | 0.039700 | -0.026507 | LOW | UNKNOWN_OVERREPRESENTED;ABS_SMD=0.027;DOMINANT_RATIO_DIFF=-0.003;BOOTSTRAP_CI_EXCLUDES_ZERO |
| topic_04 | 春景花木与岁时感怀 | BROAD | 0.056584 | 0.053538 | 0.022822 | LOW | KNOWN_OVERREPRESENTED;ABS_SMD=0.023;DOMINANT_RATIO_DIFF=0.004;BOOTSTRAP_CI_EXCLUDES_ZERO |
| topic_19 | 闲居山林与隐逸生活 | CLEAR | 0.064069 | 0.061021 | 0.021391 | LOW | KNOWN_OVERREPRESENTED;ABS_SMD=0.021;DOMINANT_RATIO_DIFF=0.006;BOOTSTRAP_CI_EXCLUDES_ZERO |
| topic_18 | 飞鸟夜景、山水与行旅（混合） | MIXED | 0.023650 | 0.022030 | 0.020843 | LOW | KNOWN_OVERREPRESENTED;ABS_SMD=0.021;DOMINANT_RATIO_DIFF=0.001;BOOTSTRAP_CI_EXCLUDES_ZERO |
| topic_09 | 泛化词主导杂咏（噪声） | NOISE | 0.023291 | 0.024846 | -0.018897 | LOW | UNKNOWN_OVERREPRESENTED;ABS_SMD=0.019;DOMINANT_RATIO_DIFF=0.000;BOOTSTRAP_CI_EXCLUDES_ZERO |
| topic_01 | 宫廷宴乐与杂咏（噪声） | NOISE | 0.026498 | 0.025135 | 0.016406 | LOW | KNOWN_OVERREPRESENTED;ABS_SMD=0.016;DOMINANT_RATIO_DIFF=0.001 |
| topic_14 | 寒景云物与杂咏（混合） | MIXED | 0.021050 | 0.021665 | -0.008130 | LOW | UNKNOWN_OVERREPRESENTED;ABS_SMD=0.008;DOMINANT_RATIO_DIFF=-0.001 |
| topic_07 | 相思离愁、夜月与琴声 | CLEAR | 0.067035 | 0.068247 | -0.008028 | LOW | UNKNOWN_OVERREPRESENTED;ABS_SMD=0.008;DOMINANT_RATIO_DIFF=0.001 |
| topic_16 | 边塞征战与军旅 | CLEAR | 0.037465 | 0.036933 | 0.005025 | LOW | KNOWN_OVERREPRESENTED;ABS_SMD=0.005;DOMINANT_RATIO_DIFF=-0.000 |
| topic_17 | 仙山道观、登临与神仙想象 | BROAD | 0.036192 | 0.035725 | 0.004555 | LOW | KNOWN_OVERREPRESENTED;ABS_SMD=0.005;DOMINANT_RATIO_DIFF=0.002 |
| topic_15 | 功名文士与颂赠杂糅（噪声） | NOISE | 0.030137 | 0.029764 | 0.004128 | LOW | KNOWN_OVERREPRESENTED;ABS_SMD=0.004;DOMINANT_RATIO_DIFF=0.001 |
| topic_03 | 朝廷礼乐、应制与颂圣 | CLEAR | 0.037096 | 0.055903 | -0.151666 | MODERATE | UNKNOWN_OVERREPRESENTED;ABS_SMD=0.152;DOMINANT_RATIO_DIFF=-0.029;BOOTSTRAP_CI_EXCLUDES_ZERO |
| topic_06 | 老病身世、仕宦与人生感怀 | CLEAR | 0.086589 | 0.068702 | 0.110437 | MODERATE | KNOWN_OVERREPRESENTED;ABS_SMD=0.110;DOMINANT_RATIO_DIFF=0.025;BOOTSTRAP_CI_EXCLUDES_ZERO |
| topic_05 | 君子伦理、仁义与政治教化 | BROAD | 0.055681 | 0.042616 | 0.107777 | MODERATE | KNOWN_OVERREPRESENTED;ABS_SMD=0.108;DOMINANT_RATIO_DIFF=0.021;BOOTSTRAP_CI_EXCLUDES_ZERO |
| topic_11 | 花香春情、女性与闺情 | CLEAR | 0.067617 | 0.078749 | -0.066749 | MODERATE | UNKNOWN_OVERREPRESENTED;ABS_SMD=0.067;DOMINANT_RATIO_DIFF=-0.018;BOOTSTRAP_CI_EXCLUDES_ZERO |
| topic_20 | 饮酒酬赠、宴集与送别 | CLEAR | 0.062251 | 0.054381 | 0.060080 | MODERATE | KNOWN_OVERREPRESENTED;ABS_SMD=0.060;DOMINANT_RATIO_DIFF=0.007;BOOTSTRAP_CI_EXCLUDES_ZERO |
| topic_08 | 山居寺院、僧侣与幽居 | CLEAR | 0.077498 | 0.086274 | -0.051873 | MODERATE | UNKNOWN_OVERREPRESENTED;ABS_SMD=0.052;DOMINANT_RATIO_DIFF=-0.016;BOOTSTRAP_CI_EXCLUDES_ZERO |
| topic_02 | 人生出处、忠义与悲慨（混合） | MIXED | 0.051392 | 0.045274 | 0.050102 | MODERATE | KNOWN_OVERREPRESENTED;ABS_SMD=0.050;DOMINANT_RATIO_DIFF=0.008;BOOTSTRAP_CI_EXCLUDES_ZERO |

# 11. CLEAR Topic Assessment

- LOW：topic_07, topic_10, topic_16, topic_19
- MODERATE：topic_03, topic_06, topic_08, topic_11, topic_20
- HIGH：无

| topic_id | topic_label | mean_known | mean_unknown | smd | bootstrap_ci_low | bootstrap_ci_high | dominant_ratio_difference | risk_level |
|---|---|---|---|---|---|---|---|---|
| topic_03 | 朝廷礼乐、应制与颂圣 | 0.037096 | 0.055903 | -0.151666 | -0.021098 | -0.016452 | -0.028676 | MODERATE |
| topic_06 | 老病身世、仕宦与人生感怀 | 0.086589 | 0.068702 | 0.110437 | 0.014887 | 0.020637 | 0.025237 | MODERATE |
| topic_07 | 相思离愁、夜月与琴声 | 0.067035 | 0.068247 | -0.008028 | -0.004063 | 0.001796 | 0.000651 | LOW |
| topic_08 | 山居寺院、僧侣与幽居 | 0.077498 | 0.086274 | -0.051873 | -0.011880 | -0.005692 | -0.015992 | MODERATE |
| topic_10 | 江湖羁旅、送别与思归 | 0.098895 | 0.106078 | -0.038012 | -0.010821 | -0.003730 | -0.008630 | LOW |
| topic_11 | 花香春情、女性与闺情 | 0.067617 | 0.078749 | -0.066749 | -0.014302 | -0.007734 | -0.018143 | MODERATE |
| topic_16 | 边塞征战与军旅 | 0.037465 | 0.036933 | 0.005025 | -0.001559 | 0.002532 | -0.000108 | LOW |
| topic_19 | 闲居山林与隐逸生活 | 0.064069 | 0.061021 | 0.021391 | 0.000541 | 0.005617 | 0.006311 | LOW |
| topic_20 | 饮酒酬赠、宴集与送别 | 0.062251 | 0.054381 | 0.060080 | 0.005269 | 0.010382 | 0.007236 | MODERATE |

# 12. Overall Assessment

A. Known-period poems 与 full corpus 存在可检测差异；是否具实质性主要由上述效应量而非 p 值判断。

B. 20 Topic 的效应量计数为 3/0/0（阈值0.10/0.20/0.50），不把大样本统计显著自动解释为严重偏差。

C. 作者产量 Cliff's delta=0.3054，显示 UNKNOWN 是否偏向低产作者应按其方向与效应等级判断。

D. UNKNOWN 的 MIXED+NOISE mass 相对差为 -0.004570。

E. CLEAR LOW-risk：topic_07, topic_10, topic_16, topic_19。

F. CLEAR MODERATE/HIGH-risk：topic_03, topic_06, topic_08, topic_11, topic_20。

G. 三个未来样本中，EXPANDED 的预登记综合偏差分数最低。

H. 建议代码 2：可以进入 author-period analysis，但必须同步报告 STRICT/CORE/EXPANDED 敏感性结果。

Logistic missingness diagnostic 的 apparent AUC=0.9490，按作者分组的 cross-validated AUC=0.9217。该模型仅诊断 period availability 可预测性，禁止用于时期填补。
