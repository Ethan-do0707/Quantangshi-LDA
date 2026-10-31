# QTS Author-Period Topic Trend Structure Report

# 1. Objective

本分析只描述 C_K20 的20维 Topic probability 如何随 `author_period_primary`（作者所属时期）分布；它不是单篇作品创作年代分析，也不重训 LDA、不修改 taxonomy、Topic probability、作者时期或 CBDB 匹配。

# 2. Analytical Samples

- STRICT：8,722 首、180 位作者。
- CORE：10,841 首、207 位作者。
- EXPANDED：23,609 首、479 位作者。

STRICT 直接使用审计的 `eligible_period_strict`，CORE 使用 `eligible_period_core`，EXPANDED 使用 `eligible_period_expanded`。CORE 是 primary descriptive specification；STRICT 是 metadata-quality sensitivity；EXPANDED 是 coverage sensitivity。三者都不是“真实答案”。

# 3. Aggregation Estimands

- POEM：每首入选诗权重相同，回答现存诗歌文本总体中的平均 Topic probability。
- AUTHOR：先计算每位作者在样本内的 Topic profile，再在同一时期等权平均，回答典型作者的平均 Topic profile。

两者是不同 estimand；差异被视为趋势结构诊断，不视为错误。

# 4. Sample Sizes

| sample | period | poem_count | author_count |
|---|---|---|---|
| STRICT | 初唐 | 432 | 37 |
| STRICT | 盛唐 | 1804 | 35 |
| STRICT | 中唐 | 4128 | 74 |
| STRICT | 晚唐 | 2358 | 34 |
| CORE | 初唐 | 623 | 41 |
| CORE | 盛唐 | 3191 | 43 |
| CORE | 中唐 | 4153 | 83 |
| CORE | 晚唐 | 2874 | 40 |
| EXPANDED | 初唐 | 1155 | 102 |
| EXPANDED | 盛唐 | 4762 | 124 |
| EXPANDED | 中唐 | 10617 | 162 |
| EXPANDED | 晚唐 | 7075 | 91 |

# 5. Overall Topic Structure

CORE_POEM peak 分布：{'初唐': 6, '盛唐': 5, '中唐': 3, '晚唐': 6, 'TIE': 0}。CORE_AUTHOR peak 分布：{'初唐': 4, '盛唐': 4, '中唐': 5, '晚唐': 7, 'TIE': 0}。六规格稳健性为 HIGH/MODERATE/LOW=0/16/4。这些结果只概括统计结构。

# 6. CORE Poem-weighted Trends

| topic_id | topic_label | early_mean | high_mean | middle_mean | late_mean | peak_period | period_rank_order | trend_shape | normalized_range |
|---|---|---|---|---|---|---|---|---|---|
| topic_01 | 宫廷宴乐与杂咏（噪声） | 0.032803 | 0.026544 | 0.025084 | 0.026371 | 初唐 | E>H>L>M | FALL_THEN_RISE | 0.278643 |
| topic_02 | 人生出处、忠义与悲慨（混合） | 0.046010 | 0.089669 | 0.048081 | 0.046557 | 盛唐 | H>M>L>E | RISE_THEN_FALL | 0.758249 |
| topic_03 | 朝廷礼乐、应制与颂圣 | 0.113395 | 0.038834 | 0.042347 | 0.018925 | 初唐 | E>M>H>L | MULTI_TURN | 1.769909 |
| topic_04 | 春景花木与岁时感怀 | 0.039724 | 0.037231 | 0.059025 | 0.054173 | 中唐 | M>L>E>H | MULTI_TURN | 0.458446 |
| topic_05 | 君子伦理、仁义与政治教化 | 0.038307 | 0.077172 | 0.077826 | 0.034707 | 中唐 | M>H>E>L | RISE_THEN_FALL | 0.756433 |
| topic_06 | 老病身世、仕宦与人生感怀 | 0.015770 | 0.048243 | 0.076336 | 0.109402 | 晚唐 | L>M>H>E | MONOTONIC_UP | 1.499608 |
| topic_07 | 相思离愁、夜月与琴声 | 0.104911 | 0.074231 | 0.076401 | 0.050394 | 初唐 | E>M>H>L | MULTI_TURN | 0.712781 |
| topic_08 | 山居寺院、僧侣与幽居 | 0.029127 | 0.033880 | 0.070387 | 0.098120 | 晚唐 | L>M>H>E | MONOTONIC_UP | 1.192020 |
| topic_09 | 泛化词主导杂咏（噪声） | 0.020557 | 0.017993 | 0.022621 | 0.026475 | 晚唐 | L>M>E>H | FALL_THEN_RISE | 0.387128 |
| topic_10 | 江湖羁旅、送别与思归 | 0.090172 | 0.114037 | 0.090663 | 0.117738 | 晚唐 | L>H>M>E | MULTI_TURN | 0.267239 |
| topic_11 | 花香春情、女性与闺情 | 0.060449 | 0.045104 | 0.053299 | 0.098352 | 晚唐 | L>E>M>H | FALL_THEN_RISE | 0.828094 |
| topic_12 | 春水花鸟与自然风物 | 0.043169 | 0.028592 | 0.035507 | 0.050843 | 晚唐 | L>E>M>H | FALL_THEN_RISE | 0.562930 |
| topic_13 | 清景物候、咏物与杂咏（混合） | 0.073737 | 0.042811 | 0.047369 | 0.023277 | 初唐 | E>M>H>L | MULTI_TURN | 1.078247 |
| topic_14 | 寒景云物与杂咏（混合） | 0.045971 | 0.021713 | 0.024175 | 0.015408 | 初唐 | E>M>H>L | MULTI_TURN | 1.139687 |
| topic_15 | 功名文士与颂赠杂糅（噪声） | 0.030443 | 0.046650 | 0.029879 | 0.025521 | 盛唐 | H>E>M>L | RISE_THEN_FALL | 0.637876 |
| topic_16 | 边塞征战与军旅 | 0.046193 | 0.048307 | 0.037297 | 0.028071 | 盛唐 | H>E>M>L | RISE_THEN_FALL | 0.506321 |
| topic_17 | 仙山道观、登临与神仙想象 | 0.043851 | 0.051868 | 0.032036 | 0.027651 | 盛唐 | H>E>M>L | RISE_THEN_FALL | 0.623318 |
| topic_18 | 飞鸟夜景、山水与行旅（混合） | 0.030767 | 0.022100 | 0.024601 | 0.023443 | 初唐 | E>M>L>H | MULTI_TURN | 0.343543 |
| topic_19 | 闲居山林与隐逸生活 | 0.048459 | 0.070420 | 0.062069 | 0.070255 | 盛唐 | H>L>M>E | MULTI_TURN | 0.349687 |
| topic_20 | 饮酒酬赠、宴集与送别 | 0.046186 | 0.064602 | 0.064997 | 0.054316 | 中唐 | M>H>L>E | UNCERTAIN | 0.327000 |

# 7. CORE Author-balanced Trends

| topic_id | topic_label | early_mean | high_mean | middle_mean | late_mean | peak_period | period_rank_order | trend_shape | normalized_range |
|---|---|---|---|---|---|---|---|---|---|
| topic_01 | 宫廷宴乐与杂咏（噪声） | 0.033091 | 0.016417 | 0.029776 | 0.033461 | 晚唐 | L>E>M>H | FALL_THEN_RISE | 0.604689 |
| topic_02 | 人生出处、忠义与悲慨（混合） | 0.026497 | 0.053353 | 0.021452 | 0.041091 | 盛唐 | H>L>E>M | MULTI_TURN | 0.896155 |
| topic_03 | 朝廷礼乐、应制与颂圣 | 0.159758 | 0.093968 | 0.099814 | 0.034625 | 初唐 | E>M>H>L | MULTI_TURN | 1.289481 |
| topic_04 | 春景花木与岁时感怀 | 0.041343 | 0.029233 | 0.046654 | 0.055987 | 晚唐 | L>M>E>H | FALL_THEN_RISE | 0.617812 |
| topic_05 | 君子伦理、仁义与政治教化 | 0.055486 | 0.056112 | 0.076859 | 0.072542 | 中唐 | M>L>H>E | UNCERTAIN | 0.327569 |
| topic_06 | 老病身世、仕宦与人生感怀 | 0.040618 | 0.050171 | 0.068793 | 0.110852 | 晚唐 | L>M>H>E | MONOTONIC_UP | 1.038831 |
| topic_07 | 相思离愁、夜月与琴声 | 0.089499 | 0.055886 | 0.047159 | 0.066211 | 初唐 | E>L>H>M | FALL_THEN_RISE | 0.654527 |
| topic_08 | 山居寺院、僧侣与幽居 | 0.022939 | 0.036226 | 0.049763 | 0.064198 | 晚唐 | L>M>H>E | MONOTONIC_UP | 0.953267 |
| topic_09 | 泛化词主导杂咏（噪声） | 0.035744 | 0.022856 | 0.029321 | 0.037360 | 晚唐 | L>E>M>H | FALL_THEN_RISE | 0.463083 |
| topic_10 | 江湖羁旅、送别与思归 | 0.049461 | 0.081464 | 0.073756 | 0.106807 | 晚唐 | L>H>M>E | MULTI_TURN | 0.736412 |
| topic_11 | 花香春情、女性与闺情 | 0.085162 | 0.058365 | 0.035439 | 0.070361 | 初唐 | E>L>H>M | FALL_THEN_RISE | 0.797707 |
| topic_12 | 春水花鸟与自然风物 | 0.036958 | 0.075100 | 0.064853 | 0.074011 | 盛唐 | H>L>M>E | MULTI_TURN | 0.608036 |
| topic_13 | 清景物候、咏物与杂咏（混合） | 0.080557 | 0.058154 | 0.084815 | 0.028809 | 中唐 | M>E>H>L | MULTI_TURN | 0.887805 |
| topic_14 | 寒景云物与杂咏（混合） | 0.040867 | 0.048242 | 0.030430 | 0.032403 | 盛唐 | H>E>L>M | MULTI_TURN | 0.468920 |
| topic_15 | 功名文士与颂赠杂糅（噪声） | 0.029291 | 0.040547 | 0.044738 | 0.046841 | 晚唐 | L>M>H>E | MONOTONIC_UP | 0.434910 |
| topic_16 | 边塞征战与军旅 | 0.069696 | 0.058046 | 0.024778 | 0.031786 | 初唐 | E>H>L>M | FALL_THEN_RISE | 0.974868 |
| topic_17 | 仙山道观、登临与神仙想象 | 0.028851 | 0.047739 | 0.049383 | 0.014549 | 中唐 | M>H>E>L | RISE_THEN_FALL | 0.991572 |
| topic_18 | 飞鸟夜景、山水与行旅（混合） | 0.025429 | 0.017854 | 0.030448 | 0.016639 | 中唐 | M>E>H>L | MULTI_TURN | 0.611200 |
| topic_19 | 闲居山林与隐逸生活 | 0.026541 | 0.062616 | 0.051852 | 0.029855 | 盛唐 | H>M>L>E | RISE_THEN_FALL | 0.844528 |
| topic_20 | 饮酒酬赠、宴集与送别 | 0.022213 | 0.037652 | 0.039918 | 0.031610 | 中唐 | M>H>L>E | RISE_THEN_FALL | 0.538990 |

# 8. Aggregation Sensitivity

CORE 的20个 Topic 中，poem-weighted 与 author-balanced peak 一致 10 个、不一致 10 个。Pearson/Spearman 只有四个时期，只作描述性诊断，不报告 p 值。

| topic_id | poem_peak_period | author_peak_period | peak_period_agreement | poem_trend_shape | author_trend_shape | pearson_four_period | spearman_four_period | max_abs_aggregation_difference | direction_conflict_count |
|---|---|---|---|---|---|---|---|---|---|
| topic_01 | 初唐 | 晚唐 | False | FALL_THEN_RISE | FALL_THEN_RISE | 0.329570 | 0.000000 | 0.010127 | 0 |
| topic_02 | 盛唐 | 盛唐 | True | RISE_THEN_FALL | MULTI_TURN | 0.805560 | 0.400000 | 0.036316 | 0 |
| topic_03 | 初唐 | 初唐 | True | MULTI_TURN | MULTI_TURN | 0.935167 | 1.000000 | 0.057467 | 0 |
| topic_04 | 中唐 | 晚唐 | False | MULTI_TURN | FALL_THEN_RISE | 0.789895 | 0.800000 | 0.012371 | 0 |
| topic_05 | 中唐 | 中唐 | True | RISE_THEN_FALL | UNCERTAIN | 0.098426 | 0.400000 | 0.037834 | 0 |
| topic_06 | 晚唐 | 晚唐 | True | MONOTONIC_UP | MONOTONIC_UP | 0.954224 | 1.000000 | 0.024848 | 0 |
| topic_07 | 初唐 | 初唐 | True | MULTI_TURN | FALL_THEN_RISE | 0.561210 | 0.200000 | 0.029242 | 0 |
| topic_08 | 晚唐 | 晚唐 | True | MONOTONIC_UP | MONOTONIC_UP | 0.968601 | 1.000000 | 0.033922 | 0 |
| topic_09 | 晚唐 | 晚唐 | True | FALL_THEN_RISE | FALL_THEN_RISE | 0.748682 | 0.800000 | 0.015187 | 0 |
| topic_10 | 晚唐 | 晚唐 | True | MULTI_TURN | MULTI_TURN | 0.841156 | 1.000000 | 0.040711 | 0 |
| topic_11 | 晚唐 | 初唐 | False | FALL_THEN_RISE | FALL_THEN_RISE | 0.375527 | 0.600000 | 0.027990 | 0 |
| topic_12 | 晚唐 | 盛唐 | False | FALL_THEN_RISE | MULTI_TURN | -0.214570 | -0.400000 | 0.046508 | 0 |
| topic_13 | 初唐 | 中唐 | False | MULTI_TURN | MULTI_TURN | 0.818092 | 0.800000 | 0.037446 | 0 |
| topic_14 | 初唐 | 盛唐 | False | MULTI_TURN | MULTI_TURN | 0.264525 | 0.000000 | 0.026528 | 0 |
| topic_15 | 盛唐 | 晚唐 | False | RISE_THEN_FALL | MONOTONIC_UP | -0.143484 | -0.800000 | 0.021320 | 0 |
| topic_16 | 盛唐 | 初唐 | False | RISE_THEN_FALL | FALL_THEN_RISE | 0.801686 | 0.600000 | 0.023503 | 0 |
| topic_17 | 盛唐 | 中唐 | False | RISE_THEN_FALL | RISE_THEN_FALL | 0.484028 | 0.400000 | 0.017347 | 0 |
| topic_18 | 初唐 | 中唐 | False | MULTI_TURN | MULTI_TURN | 0.483278 | 0.600000 | 0.006804 | 0 |
| topic_19 | 盛唐 | 盛唐 | True | MULTI_TURN | RISE_THEN_FALL | 0.521935 | 0.800000 | 0.040400 | 0 |
| topic_20 | 中唐 | 中唐 | True | UNCERTAIN | RISE_THEN_FALL | 0.986173 | 1.000000 | 0.026951 | 0 |

# 9. Sample Sensitivity

`SAMPLE_SENSITIVE` 表示同一聚合口径下 STRICT/CORE/EXPANDED 的 peak 不完全一致；它不自动否定曲线，而是要求后续效应量分析保留样本敏感性。

| topic_id | dominant_peak_period | peak_consistency_ratio | dominant_rank_order | dominant_trend_shape | sample_sensitive | aggregation_sensitive | direction_unstable |
|---|---|---|---|---|---|---|---|
| topic_01 | 初唐 | 0.833333 | UNCERTAIN | FALL_THEN_RISE | True | True | False |
| topic_02 | 盛唐 | 1.000000 | H>L>E>M | MULTI_TURN | False | False | False |
| topic_03 | 初唐 | 1.000000 | E>H>M>L | MONOTONIC_DOWN | False | False | False |
| topic_04 | UNCERTAIN | 0.500000 | L>M>E>H | FALL_THEN_RISE | False | True | False |
| topic_05 | UNCERTAIN | 0.500000 | UNCERTAIN | RISE_THEN_FALL | True | True | False |
| topic_06 | 晚唐 | 0.833333 | L>M>H>E | MONOTONIC_UP | True | True | False |
| topic_07 | 初唐 | 0.833333 | UNCERTAIN | MULTI_TURN | True | True | False |
| topic_08 | 晚唐 | 1.000000 | L>M>H>E | MONOTONIC_UP | False | False | False |
| topic_09 | 晚唐 | 1.000000 | UNCERTAIN | FALL_THEN_RISE | False | False | False |
| topic_10 | 晚唐 | 0.833333 | L>H>M>E | MULTI_TURN | True | True | False |
| topic_11 | 晚唐 | 0.666667 | UNCERTAIN | FALL_THEN_RISE | True | True | False |
| topic_12 | 晚唐 | 0.666667 | L>E>M>H | FALL_THEN_RISE | True | True | False |
| topic_13 | 初唐 | 0.666667 | E>M>H>L | MULTI_TURN | True | True | False |
| topic_14 | 初唐 | 0.666667 | UNCERTAIN | MULTI_TURN | True | True | False |
| topic_15 | 盛唐 | 0.666667 | UNCERTAIN | RISE_THEN_FALL | True | True | False |
| topic_16 | 初唐 | 0.666667 | E>H>L>M | UNCERTAIN | True | True | False |
| topic_17 | 盛唐 | 0.500000 | UNCERTAIN | RISE_THEN_FALL | True | True | False |
| topic_18 | 初唐 | 0.666667 | E>M>H>L | MULTI_TURN | True | True | False |
| topic_19 | 盛唐 | 0.833333 | H>L>M>E | UNCERTAIN | True | True | False |
| topic_20 | 中唐 | 0.833333 | M>H>L>E | RISE_THEN_FALL | True | True | False |

# 10. Robust Trend Structures

HIGH topics：无。稳健性同时考虑 peak、完整 rank order、三个相邻 bootstrap-supported direction 及 poem/author 冲突；不只看 peak。

| topic_id | peak_consistency_count | rank_order_consistency_count | trend_shape_consistency_count | EH_UP_count | EH_DOWN_count | HM_UP_count | HM_DOWN_count | ML_UP_count | ML_DOWN_count | poem_author_direction_conflict_count | trend_robustness |
|---|---|---|---|---|---|---|---|---|---|---|---|
| topic_01 | 5 | 2 | 5 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | MODERATE |
| topic_02 | 6 | 4 | 4 | 3 | 0 | 0 | 6 | 1 | 0 | 0 | MODERATE |
| topic_03 | 6 | 4 | 4 | 0 | 3 | 0 | 1 | 0 | 6 | 0 | MODERATE |
| topic_04 | 3 | 3 | 3 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | LOW |
| topic_05 | 3 | 2 | 5 | 3 | 0 | 0 | 0 | 0 | 2 | 0 | LOW |
| topic_06 | 5 | 4 | 4 | 3 | 0 | 2 | 0 | 2 | 0 | 0 | MODERATE |
| topic_07 | 5 | 2 | 3 | 0 | 2 | 0 | 1 | 0 | 3 | 0 | MODERATE |
| topic_08 | 6 | 5 | 5 | 0 | 0 | 4 | 0 | 1 | 0 | 0 | MODERATE |
| topic_09 | 6 | 2 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | MODERATE |
| topic_10 | 5 | 4 | 4 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | MODERATE |
| topic_11 | 4 | 2 | 6 | 0 | 1 | 0 | 0 | 5 | 0 | 0 | MODERATE |
| topic_12 | 4 | 3 | 4 | 0 | 1 | 0 | 0 | 3 | 0 | 0 | LOW |
| topic_13 | 4 | 3 | 5 | 0 | 2 | 0 | 0 | 0 | 5 | 0 | MODERATE |
| topic_14 | 4 | 2 | 3 | 0 | 3 | 0 | 0 | 0 | 2 | 0 | MODERATE |
| topic_15 | 4 | 2 | 3 | 1 | 0 | 0 | 3 | 0 | 0 | 0 | MODERATE |
| topic_16 | 4 | 3 | 2 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | MODERATE |
| topic_17 | 3 | 2 | 5 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | LOW |
| topic_18 | 4 | 3 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | MODERATE |
| topic_19 | 5 | 3 | 3 | 5 | 0 | 0 | 0 | 0 | 1 | 0 | MODERATE |
| topic_20 | 5 | 4 | 4 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | MODERATE |

# 11. CLEAR Topic Assessment

| Topic | Label | Representativeness Risk | CORE Poem Peak | CORE Author Peak | Peak Consistency | Shape | Trend Robustness | Author Concentration | Interpretation Status |
|---|---|---|---|---|---|---|---|---|---|
| topic_03 | 朝廷礼乐、应制与颂圣 | MODERATE | 初唐 | 初唐 | 1.000000 | MONOTONIC_DOWN | MODERATE | 0.774261 | REVIEW_CANDIDATE |
| topic_06 | 老病身世、仕宦与人生感怀 | MODERATE | 晚唐 | 晚唐 | 0.833333 | MONOTONIC_UP | MODERATE | 0.906327 | REVIEW_CANDIDATE |
| topic_07 | 相思离愁、夜月与琴声 | LOW | 初唐 | 初唐 | 0.833333 | MULTI_TURN | MODERATE | 0.874421 | REVIEW_CANDIDATE |
| topic_08 | 山居寺院、僧侣与幽居 | MODERATE | 晚唐 | 晚唐 | 1.000000 | MONOTONIC_UP | MODERATE | 0.851865 | REVIEW_CANDIDATE |
| topic_10 | 江湖羁旅、送别与思归 | LOW | 晚唐 | 晚唐 | 0.833333 | MULTI_TURN | MODERATE | 0.852861 | REVIEW_CANDIDATE |
| topic_11 | 花香春情、女性与闺情 | MODERATE | 晚唐 | 初唐 | 0.666667 | FALL_THEN_RISE | MODERATE | 0.903339 | REVIEW_CANDIDATE |
| topic_16 | 边塞征战与军旅 | LOW | 盛唐 | 初唐 | 0.666667 | UNCERTAIN | MODERATE | 0.898210 | REVIEW_CANDIDATE |
| topic_19 | 闲居山林与隐逸生活 | LOW | 盛唐 | 盛唐 | 0.833333 | UNCERTAIN | MODERATE | 0.871622 | REVIEW_CANDIDATE |
| topic_20 | 饮酒酬赠、宴集与送别 | MODERATE | 中唐 | 中唐 | 0.833333 | RISE_THEN_FALL | MODERATE | 0.899132 | REVIEW_CANDIDATE |

# 12. Author Concentration

集中度以各时期 poem-weighted Topic mass 的作者份额计算。top1/top5/top10 与 HHI 只说明少数高产作者可能驱动总体质量，不作具体诗人文学解释。

| topic_id | period | author_count | top1_author_share | top5_author_share | top10_author_share | hhi | concentration_level |
|---|---|---|---|---|---|---|---|
| topic_06 | 盛唐 | 43 | 0.566009 | 0.906327 | 0.981074 | 0.361469 | HIGH |
| topic_02 | 盛唐 | 43 | 0.383177 | 0.904260 | 0.982431 | 0.266546 | HIGH |
| topic_11 | 盛唐 | 43 | 0.429760 | 0.903339 | 0.974618 | 0.265704 | HIGH |
| topic_20 | 盛唐 | 43 | 0.335412 | 0.899132 | 0.977121 | 0.231289 | HIGH |
| topic_15 | 盛唐 | 43 | 0.415290 | 0.898961 | 0.964208 | 0.275628 | HIGH |
| topic_16 | 盛唐 | 43 | 0.363428 | 0.898210 | 0.971357 | 0.248118 | HIGH |
| topic_05 | 盛唐 | 43 | 0.471297 | 0.892289 | 0.980639 | 0.288363 | HIGH |
| topic_01 | 盛唐 | 43 | 0.440944 | 0.888264 | 0.974499 | 0.288375 | HIGH |
| topic_17 | 盛唐 | 43 | 0.479063 | 0.884009 | 0.972170 | 0.308170 | HIGH |
| topic_07 | 盛唐 | 43 | 0.473503 | 0.874421 | 0.970710 | 0.275805 | HIGH |
| topic_02 | 初唐 | 41 | 0.466164 | 0.873937 | 0.943783 | 0.307351 | HIGH |
| topic_19 | 初唐 | 41 | 0.440427 | 0.871622 | 0.959120 | 0.266057 | HIGH |
| topic_19 | 盛唐 | 43 | 0.387524 | 0.864457 | 0.968132 | 0.221083 | HIGH |
| topic_14 | 盛唐 | 43 | 0.368349 | 0.857512 | 0.951339 | 0.220943 | HIGH |
| topic_18 | 盛唐 | 43 | 0.327012 | 0.857096 | 0.963008 | 0.208053 | HIGH |
| topic_20 | 初唐 | 41 | 0.317239 | 0.854610 | 0.964863 | 0.198254 | HIGH |
| topic_10 | 初唐 | 41 | 0.417548 | 0.852861 | 0.943531 | 0.244375 | HIGH |
| topic_08 | 盛唐 | 43 | 0.456004 | 0.851865 | 0.978753 | 0.261328 | HIGH |
| topic_10 | 盛唐 | 43 | 0.365887 | 0.846180 | 0.968777 | 0.209145 | HIGH |
| topic_04 | 盛唐 | 43 | 0.428693 | 0.839958 | 0.966735 | 0.248351 | HIGH |

# 13. Candidate Trends for Humanistic Interpretation

以下仅为 candidate robust trends，筛选条件是 CLEAR、representativeness risk 非 HIGH、trend robustness 为 HIGH 或 MODERATE；不构成文学史规律。

| topic_id | topic_label | representativeness_risk | trend_robustness | dominant_peak_period | peak_consistency_ratio | main_direction_pattern | poem_author_agreement | sensitivity_flags |
|---|---|---|---|---|---|---|---|---|
| topic_03 | 朝廷礼乐、应制与颂圣 | MODERATE | MODERATE | 初唐 | 1.000000 | EH:UNCERTAIN|HM:UNCERTAIN|ML:DOWN | HIGH | METADATA_RISK |
| topic_06 | 老病身世、仕宦与人生感怀 | MODERATE | MODERATE | 晚唐 | 0.833333 | EH:UNCERTAIN|HM:UNCERTAIN|ML:UNCERTAIN | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |
| topic_07 | 相思离愁、夜月与琴声 | LOW | MODERATE | 初唐 | 0.833333 | EH:UNCERTAIN|HM:UNCERTAIN|ML:UNCERTAIN | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE |
| topic_08 | 山居寺院、僧侣与幽居 | MODERATE | MODERATE | 晚唐 | 1.000000 | EH:UNCERTAIN|HM:UP|ML:UNCERTAIN | HIGH | METADATA_RISK |
| topic_10 | 江湖羁旅、送别与思归 | LOW | MODERATE | 晚唐 | 0.833333 | EH:UNCERTAIN|HM:UNCERTAIN|ML:UNCERTAIN | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE |
| topic_11 | 花香春情、女性与闺情 | MODERATE | MODERATE | 晚唐 | 0.666667 | EH:UNCERTAIN|HM:UNCERTAIN|ML:UP | LOW | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |
| topic_16 | 边塞征战与军旅 | LOW | MODERATE | 初唐 | 0.666667 | EH:UNCERTAIN|HM:DOWN|ML:UNCERTAIN | LOW | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE |
| topic_19 | 闲居山林与隐逸生活 | LOW | MODERATE | 盛唐 | 0.833333 | EH:UP|HM:UNCERTAIN|ML:UNCERTAIN | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE |
| topic_20 | 饮酒酬赠、宴集与送别 | MODERATE | MODERATE | 中唐 | 0.833333 | EH:UNCERTAIN|HM:UNCERTAIN|ML:UNCERTAIN | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |

# 14. Limitations

- Author period 不等于 poem composition date。
- 已知作者时期只覆盖部分诗歌，且 metadata 明显偏向高产作者。
- 现存《全唐诗》语料本身存在文献保存偏差。
- 作者产量高度不均，poem-weighted 与 author-balanced 回答不同问题。
- Topic probability 是 LDA 混合概率，不等于传统题材的唯一分类。
- Bootstrap CI 与 bootstrap-supported direction 只作 uncertainty 描述，不是正式假设检验；本阶段没有运行 ANOVA、Kruskal–Wallis、pairwise Mann–Whitney 或多重比较校正。

## 下一阶段接口

输出已保留 sample、aggregation、period、poem/author count、均值、bootstrap interval、shape、robustness 与 flags，可进入 Effect Size & Statistical Validation；下一阶段仍须保持三样本、两聚合口径和 representativeness warning，并避免把作者时期解释为作品创作年代。
