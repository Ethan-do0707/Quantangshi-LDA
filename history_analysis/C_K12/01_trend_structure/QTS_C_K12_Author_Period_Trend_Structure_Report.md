# QTS C_K12 Author-Period Trend Structure Report

# 1 Objective

独立重算 C_K12 的作者所属时期主题结构。所有结果描述 author_period_primary 的分布差异，不表示单篇作品创作年代。

# 2 Model and Topic Taxonomy

C_K12 为 FULL_CORPUS_FIT、12维 Topic probability。标签来自主题总表，当前 provenance 为 AI_PROVISIONAL。CLEAR=6，BROAD=4，NOISE=2。

| final_topic_id | final_topic_label | topic_type | interpretation_scope | label_provenance |
| --- | --- | --- | --- | --- |
| T01 | 高山瀑布、仙境寻真与登临清赏 | BROAD | EXPLORATORY_BROAD | AI_PROVISIONAL |
| T02 | 老病退居、诗酒知命与士人自况 | BROAD | EXPLORATORY_BROAD | AI_PROVISIONAL |
| T03 | 江水舟行、秋景与客中怀归 | CLEAR | PRIMARY_INFERENCE | AI_PROVISIONAL |
| T04 | 仕宦行旅、寺观仙踪与咏史杂咏 | NOISE | DESCRIPTIVE_NOISE_ONLY | AI_PROVISIONAL |
| T05 | 皇家礼仪、郊庙宴庆与应制颂德 | CLEAR | PRIMARY_INFERENCE | AI_PROVISIONAL |
| T06 | 宫闺月夜、相思与女性哀怨 | CLEAR | PRIMARY_INFERENCE | AI_PROVISIONAL |
| T07 | 民生疾苦、伦理讽喻与数字杂咏 | NOISE | DESCRIPTIVE_NOISE_ONLY | AI_PROVISIONAL |
| T08 | 王朝功德、郊庙礼乐与忠贤颂咏 | CLEAR | PRIMARY_INFERENCE | AI_PROVISIONAL |
| T09 | 山寺禅居、僧侣往还与清夜幽境 | CLEAR | PRIMARY_INFERENCE | AI_PROVISIONAL |
| T10 | 山水送别、怀友与归隐心绪 | BROAD | EXPLORATORY_BROAD | AI_PROVISIONAL |
| T11 | 边塞征战、征人思归与战争悲悯 | CLEAR | PRIMARY_INFERENCE | AI_PROVISIONAL |
| T12 | 春日花酒、宴游与惜时送别 | BROAD | EXPLORATORY_BROAD | AI_PROVISIONAL |

# 3 Historical Metadata

复用冻结 author_period_primary、author_id_internal、STRICT/CORE/EXPANDED 资格与字段 provenance；没有重建 CBDB、作者年代或时期边界。

# 4 Analytical Samples

CORE 用于主要描述，STRICT 为 metadata-quality sensitivity，EXPANDED 为 coverage sensitivity。

# 5 Aggregation Estimands

POEM 为每首诗等权。AUTHOR 先在作者内取均值，再让每位作者等权。两者回答不同问题，均予保留。

# 6 Sample Sizes

| sample | scope | observed |
| --- | --- | --- |
| STRICT | ALL_POEMS | 8722 |
| STRICT | ALL_AUTHORS | 180 |
| CORE | ALL_POEMS | 10841 |
| CORE | ALL_AUTHORS | 207 |
| EXPANDED | ALL_POEMS | 23609 |
| EXPANDED | ALL_AUTHORS | 479 |

# 7 Overall Topic Structure

十二主题、三样本、两种 aggregation 与四时期均完整计算。BROAD 标为 EXPLORATORY_BROAD，NOISE 标为 DESCRIPTIVE_NOISE_ONLY。

# 8 CORE Poem-weighted Trends

| topic_id | topic_label | topic_type | peak_period | trend_shape | period_rank_order |
| --- | --- | --- | --- | --- | --- |
| topic_01 | 高山瀑布、仙境寻真与登临清赏 | BROAD | 初唐 | MONOTONIC_DOWN | E>H>M>L |
| topic_02 | 老病退居、诗酒知命与士人自况 | BROAD | 晚唐 | MONOTONIC_UP | L>M>H>E |
| topic_03 | 江水舟行、秋景与客中怀归 | CLEAR | 盛唐 | MULTI_TURN | H>L>E>M |
| topic_04 | 仕宦行旅、寺观仙踪与咏史杂咏 | NOISE | 初唐 | MULTI_TURN | E>M>L>H |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | CLEAR | 初唐 | MULTI_TURN | E>M>H>L |
| topic_06 | 宫闺月夜、相思与女性哀怨 | CLEAR | 晚唐 | FALL_THEN_RISE | L>E>M>H |
| topic_07 | 民生疾苦、伦理讽喻与数字杂咏 | NOISE | 盛唐 | RISE_THEN_FALL | H>M>L>E |
| topic_08 | 王朝功德、郊庙礼乐与忠贤颂咏 | CLEAR | 盛唐 | RISE_THEN_FALL | H>E>M>L |
| topic_09 | 山寺禅居、僧侣往还与清夜幽境 | CLEAR | 晚唐 | MONOTONIC_UP | L>M>H>E |
| topic_10 | 山水送别、怀友与归隐心绪 | BROAD | 晚唐 | MONOTONIC_UP | L>M>H>E |
| topic_11 | 边塞征战、征人思归与战争悲悯 | CLEAR | 盛唐 | RISE_THEN_FALL | H>E>M>L |
| topic_12 | 春日花酒、宴游与惜时送别 | BROAD | 晚唐 | FALL_THEN_RISE | L>M>E>H |

# 9 CORE Author-balanced Trends

| topic_id | topic_label | topic_type | peak_period | trend_shape | period_rank_order |
| --- | --- | --- | --- | --- | --- |
| topic_01 | 高山瀑布、仙境寻真与登临清赏 | BROAD | 中唐 | RISE_THEN_FALL | M>H>E>L |
| topic_02 | 老病退居、诗酒知命与士人自况 | BROAD | 晚唐 | FALL_THEN_RISE | L>M>E>H |
| topic_03 | 江水舟行、秋景与客中怀归 | CLEAR | 盛唐 | MULTI_TURN | H>L>M>E |
| topic_04 | 仕宦行旅、寺观仙踪与咏史杂咏 | NOISE | 初唐 | FALL_THEN_RISE | E>L>M>H |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | CLEAR | 初唐 | MULTI_TURN | E>M>H>L |
| topic_06 | 宫闺月夜、相思与女性哀怨 | CLEAR | 初唐 | FALL_THEN_RISE | E>L>H>M |
| topic_07 | 民生疾苦、伦理讽喻与数字杂咏 | NOISE | 晚唐 | FALL_THEN_RISE | L>M>E>H |
| topic_08 | 王朝功德、郊庙礼乐与忠贤颂咏 | CLEAR | 盛唐 | UNCERTAIN | H>M>L>E |
| topic_09 | 山寺禅居、僧侣往还与清夜幽境 | CLEAR | 晚唐 | MONOTONIC_UP | L>M>H>E |
| topic_10 | 山水送别、怀友与归隐心绪 | BROAD | 晚唐 | MULTI_TURN | L>H>M>E |
| topic_11 | 边塞征战、征人思归与战争悲悯 | CLEAR | 初唐 | FALL_THEN_RISE | E>H>L>M |
| topic_12 | 春日花酒、宴游与惜时送别 | BROAD | 晚唐 | FALL_THEN_RISE | L>E>M>H |

# 10 Aggregation Sensitivity

四时期 Pearson/Spearman 只作曲线相似度描述，不报告 p 值。CORE 的 peak 一致与逐相邻方向冲突见 C_K12_aggregation_sensitivity.csv。

# 11 Sample Sensitivity

STRICT/CORE/EXPANDED × POEM/AUTHOR 六规格的 peak、rank、trend_shape 与方向均保存在 C_K12_sample_sensitivity.csv。

# 12 Trend Robustness

| topic_id | topic_label | topic_type | trend_robustness | peak_consistency_count | poem_author_agreement | representativeness_risk | sensitivity_flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| topic_01 | 高山瀑布、仙境寻真与登临清赏 | BROAD | LOW | 4 | LOW | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |
| topic_02 | 老病退居、诗酒知命与士人自况 | BROAD | MODERATE | 5 | MODERATE | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |
| topic_03 | 江水舟行、秋景与客中怀归 | CLEAR | MODERATE | 4 | LOW | LOW | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE |
| topic_04 | 仕宦行旅、寺观仙踪与咏史杂咏 | NOISE | MODERATE | 5 | MODERATE | LOW | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | CLEAR | HIGH | 6 | HIGH | MODERATE | METADATA_RISK |
| topic_06 | 宫闺月夜、相思与女性哀怨 | CLEAR | MODERATE | 5 | MODERATE | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |
| topic_07 | 民生疾苦、伦理讽喻与数字杂咏 | NOISE | LOW | 3 | LOW | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |
| topic_08 | 王朝功德、郊庙礼乐与忠贤颂咏 | CLEAR | MODERATE | 5 | MODERATE | LOW | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE |
| topic_09 | 山寺禅居、僧侣往还与清夜幽境 | CLEAR | MODERATE | 6 | HIGH | MODERATE | METADATA_RISK |
| topic_10 | 山水送别、怀友与归隐心绪 | BROAD | MODERATE | 6 | HIGH | LOW | NONE |
| topic_11 | 边塞征战、征人思归与战争悲悯 | CLEAR | MODERATE | 5 | MODERATE | LOW | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE |
| topic_12 | 春日花酒、宴游与惜时送别 | BROAD | MODERATE | 5 | MODERATE | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |

# 13 CLEAR Topic Assessment

| topic_id | topic_label | topic_type | trend_robustness | peak_consistency_count | poem_author_agreement | representativeness_risk | sensitivity_flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| topic_03 | 江水舟行、秋景与客中怀归 | CLEAR | MODERATE | 4 | LOW | LOW | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | CLEAR | HIGH | 6 | HIGH | MODERATE | METADATA_RISK |
| topic_06 | 宫闺月夜、相思与女性哀怨 | CLEAR | MODERATE | 5 | MODERATE | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |
| topic_08 | 王朝功德、郊庙礼乐与忠贤颂咏 | CLEAR | MODERATE | 5 | MODERATE | LOW | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE |
| topic_09 | 山寺禅居、僧侣往还与清夜幽境 | CLEAR | MODERATE | 6 | HIGH | MODERATE | METADATA_RISK |
| topic_11 | 边塞征战、征人思归与战争悲悯 | CLEAR | MODERATE | 5 | MODERATE | LOW | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE |

# 14 BROAD Topic Descriptive Assessment

| topic_id | topic_label | topic_type | trend_robustness | peak_consistency_count | poem_author_agreement | representativeness_risk | sensitivity_flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| topic_01 | 高山瀑布、仙境寻真与登临清赏 | BROAD | LOW | 4 | LOW | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |
| topic_02 | 老病退居、诗酒知命与士人自况 | BROAD | MODERATE | 5 | MODERATE | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |
| topic_10 | 山水送别、怀友与归隐心绪 | BROAD | MODERATE | 6 | HIGH | LOW | NONE |
| topic_12 | 春日花酒、宴游与惜时送别 | BROAD | MODERATE | 5 | MODERATE | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |

# 15 NOISE Topic Diagnostics

| topic_id | topic_label | topic_type | trend_robustness | peak_consistency_count | poem_author_agreement | representativeness_risk | sensitivity_flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| topic_04 | 仕宦行旅、寺观仙踪与咏史杂咏 | NOISE | MODERATE | 5 | MODERATE | LOW | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE |
| topic_07 | 民生疾苦、伦理讽喻与数字杂咏 | NOISE | LOW | 3 | LOW | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |

NOISE Topic 即使呈现时期差异，也只解释为 residual-topic composition 的变化线索，不进入独立文学史候选。

# 16 Author Concentration

CORE Topic×period 的 Top1/Top5/Top10 probability mass share 与 HHI 已重算。该指标衡量样本贡献集中度，不代表作者文学史重要性。

# 17 Candidate Trends for Statistical Validation

| topic_id | topic_label | representativeness_risk | trend_robustness | dominant_peak_period | peak_consistency_ratio | main_direction_pattern | poem_author_agreement | sensitivity_flags | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| topic_03 | 江水舟行、秋景与客中怀归 | LOW | MODERATE | 盛唐 | 0.666667 | EH:UNCERTAIN/HM:UNCERTAIN/ML:UNCERTAIN | LOW | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE | 统计结构候选：dominant peak=盛唐；shape=MULTI_TURN；peak consistency=0.667；flags=AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE。不构成文学史解释。 |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | MODERATE | HIGH | 初唐 | 1 | EH:DOWN/HM:UNCERTAIN/ML:DOWN | HIGH | METADATA_RISK | 统计结构候选：dominant peak=初唐；shape=MULTI_TURN；peak consistency=1.000；flags=METADATA_RISK。不构成文学史解释。 |
| topic_06 | 宫闺月夜、相思与女性哀怨 | MODERATE | MODERATE | 晚唐 | 0.833333 | EH:UNCERTAIN/HM:UNCERTAIN/ML:UNCERTAIN | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK | 统计结构候选：dominant peak=晚唐；shape=FALL_THEN_RISE；peak consistency=0.833；flags=AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK。不构成文学史解释。 |
| topic_08 | 王朝功德、郊庙礼乐与忠贤颂咏 | LOW | MODERATE | 盛唐 | 0.833333 | EH:UNCERTAIN/HM:DOWN/ML:UNCERTAIN | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE | 统计结构候选：dominant peak=盛唐；shape=RISE_THEN_FALL；peak consistency=0.833；flags=AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE。不构成文学史解释。 |
| topic_09 | 山寺禅居、僧侣往还与清夜幽境 | MODERATE | MODERATE | 晚唐 | 1 | EH:UNCERTAIN/HM:UNCERTAIN/ML:UNCERTAIN | HIGH | METADATA_RISK | 统计结构候选：dominant peak=晚唐；shape=MONOTONIC_UP；peak consistency=1.000；flags=METADATA_RISK。不构成文学史解释。 |
| topic_11 | 边塞征战、征人思归与战争悲悯 | LOW | MODERATE | 盛唐 | 0.833333 | EH:UNCERTAIN/HM:UNCERTAIN/ML:UNCERTAIN | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE | 统计结构候选：dominant peak=盛唐；shape=UNCERTAIN；peak consistency=0.833；flags=AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE。不构成文学史解释。 |

候选只表示适合进入下一阶段统计验证，不等于 LITERARY_HISTORY_CONFIRMED。

# 18 Limitations

作者所属时期不等于诗作年代；taxonomy 仍为 AI_PROVISIONAL；缺失年代作者与已知年代作者的 Topic composition 不完全相同；四时期平均值不能单独支持因果或确定转型叙述。
