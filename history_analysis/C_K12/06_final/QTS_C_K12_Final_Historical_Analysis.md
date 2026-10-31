# QTS C_K12 Final Historical Analysis

# 1 Objective

以 C_K12 FULL_CORPUS_FIT 的 43,015×12 Document×Topic probability matrix，独立重算作者时期主题结构、代表性风险、效应验证及作者影响。所有结果均是统计候选，不等于文学史确认。

# 2 Why the Historical Analysis Was Rerun

主 Topic 模型由原 C_K20 重新评估为 C_K12 候选，因此所有依赖 Topic 定义的历史趋势都必须从 K12 theta 重算。K20 趋势没有映射成 K12 结果；K12 在盲态冻结后才与旧 K20 作事后比较。

# 3 Data and Metadata

theta 来源：`D:\中大中文系工作相关\論文發表與投稿\全唐詩LDA\model_selection\artifact_recovery_K12_K14\full_corpus_blind_models\K12\document_topic_full.npz`，shape=[43015, 12]；finite、非负、行和与 document_index 对齐均通过。历史 metadata、`author_period_primary`、三套 eligibility 与 `author_id_internal` 均复用冻结版本，未重建作者生卒年或时期。

关键解释边界：本分析研究作者所属时期与其现存诗作 Topic probability 的关系。**author period 不等于 poem date**；结果不能解释为单首诗在某年发生变化。

# 4 K12 Topic Taxonomy

标签 provenance 为 `AI_PROVISIONAL`；Topic probability 不因标签措辞改变。CLEAR 6 题进入 primary inference，BROAD 4 题只作探索，NOISE 2 题只作模型诊断。

| Topic | 标签 | 类型 | interpretation_scope |
| --- | --- | --- | --- |
| T01 | 高山瀑布、仙境寻真与登临清赏 | BROAD | EXPLORATORY_BROAD |
| T02 | 老病退居、诗酒知命与士人自况 | BROAD | EXPLORATORY_BROAD |
| T03 | 江水舟行、秋景与客中怀归 | CLEAR | PRIMARY_INFERENCE |
| T04 | 仕宦行旅、寺观仙踪与咏史杂咏 | NOISE | DESCRIPTIVE_NOISE_ONLY |
| T05 | 皇家礼仪、郊庙宴庆与应制颂德 | CLEAR | PRIMARY_INFERENCE |
| T06 | 宫闺月夜、相思与女性哀怨 | CLEAR | PRIMARY_INFERENCE |
| T07 | 民生疾苦、伦理讽喻与数字杂咏 | NOISE | DESCRIPTIVE_NOISE_ONLY |
| T08 | 王朝功德、郊庙礼乐与忠贤颂咏 | CLEAR | PRIMARY_INFERENCE |
| T09 | 山寺禅居、僧侣往还与清夜幽境 | CLEAR | PRIMARY_INFERENCE |
| T10 | 山水送别、怀友与归隐心绪 | BROAD | EXPLORATORY_BROAD |
| T11 | 边塞征战、征人思归与战争悲悯 | CLEAR | PRIMARY_INFERENCE |
| T12 | 春日花酒、宴游与惜时送别 | BROAD | EXPLORATORY_BROAD |

# 5 Sample Coverage

CORE 是主描述规格；STRICT 是 metadata-quality sensitivity；EXPANDED 是 coverage sensitivity。样本核验全部通过。

| 样本 | 时期 | 诗数 | 作者数 | QA |
| --- | --- | --- | --- | --- |
| STRICT | ALL | 8722 | 180 | PASS |
| CORE | ALL | 10841 | 207 | PASS |
| EXPANDED | ALL | 23609 | 479 | PASS |
| CORE | 初唐 | 623 | 41 | PASS |
| CORE | 盛唐 | 3191 | 43 | PASS |
| CORE | 中唐 | 4153 | 83 | PASS |
| CORE | 晚唐 | 2874 | 40 | PASS |

# 6 CORE Poem-weighted Historical Structure

POEM 口径每首诗等权，置信区间按作者 cluster 处理。表中数值为 mean Topic probability。

| Topic | 标签 | 初唐 | 盛唐 | 中唐 | 晚唐 | 峰值 | 形态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T01 | 高山瀑布、仙境寻真与登临清赏 | 0.188761 | 0.142752 | 0.118712 | 0.0517704 | 初唐 | MONOTONIC_DOWN |
| T02 | 老病退居、诗酒知命与士人自况 | 0.0414031 | 0.105662 | 0.123999 | 0.129099 | 晚唐 | MONOTONIC_UP |
| T03 | 江水舟行、秋景与客中怀归 | 0.0946763 | 0.115309 | 0.0782854 | 0.113226 | 盛唐 | MULTI_TURN |
| T04 | 仕宦行旅、寺观仙踪与咏史杂咏 | 0.0358402 | 0.026459 | 0.0338535 | 0.0298672 | 初唐 | MULTI_TURN |
| T05 | 皇家礼仪、郊庙宴庆与应制颂德 | 0.121266 | 0.0325339 | 0.0430476 | 0.023691 | 初唐 | MULTI_TURN |
| T06 | 宫闺月夜、相思与女性哀怨 | 0.0860263 | 0.0546449 | 0.0655967 | 0.102985 | 晚唐 | FALL_THEN_RISE |
| T07 | 民生疾苦、伦理讽喻与数字杂咏 | 0.0310336 | 0.080161 | 0.0741913 | 0.0555159 | 盛唐 | RISE_THEN_FALL |
| T08 | 王朝功德、郊庙礼乐与忠贤颂咏 | 0.0594133 | 0.0687566 | 0.0483994 | 0.0400241 | 盛唐 | RISE_THEN_FALL |
| T09 | 山寺禅居、僧侣往还与清夜幽境 | 0.0462245 | 0.0471638 | 0.0925998 | 0.116059 | 晚唐 | MONOTONIC_UP |
| T10 | 山水送别、怀友与归隐心绪 | 0.066531 | 0.10142 | 0.113232 | 0.126295 | 晚唐 | MONOTONIC_UP |
| T11 | 边塞征战、征人思归与战争悲悯 | 0.119283 | 0.126944 | 0.086724 | 0.0769884 | 盛唐 | RISE_THEN_FALL |
| T12 | 春日花酒、宴游与惜时送别 | 0.109542 | 0.0981937 | 0.12136 | 0.134479 | 晚唐 | FALL_THEN_RISE |

# 7 CORE Author-balanced Historical Structure

AUTHOR 口径先在作者内部平均，再令同一时期作者等权；不按作品数二次加权。

| Topic | 标签 | 初唐 | 盛唐 | 中唐 | 晚唐 | 峰值 | 形态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T01 | 高山瀑布、仙境寻真与登临清赏 | 0.162314 | 0.173581 | 0.182462 | 0.0718819 | 中唐 | RISE_THEN_FALL |
| T02 | 老病退居、诗酒知命与士人自况 | 0.0709584 | 0.0634603 | 0.0969855 | 0.13401 | 晚唐 | FALL_THEN_RISE |
| T03 | 江水舟行、秋景与客中怀归 | 0.0511126 | 0.0988454 | 0.0584294 | 0.0981855 | 盛唐 | MULTI_TURN |
| T04 | 仕宦行旅、寺观仙踪与咏史杂咏 | 0.0390118 | 0.0251573 | 0.0317793 | 0.0334884 | 初唐 | FALL_THEN_RISE |
| T05 | 皇家礼仪、郊庙宴庆与应制颂德 | 0.17706 | 0.0815515 | 0.114712 | 0.0470638 | 初唐 | MULTI_TURN |
| T06 | 宫闺月夜、相思与女性哀怨 | 0.0994187 | 0.0802941 | 0.0755647 | 0.0967279 | 初唐 | FALL_THEN_RISE |
| T07 | 民生疾苦、伦理讽喻与数字杂咏 | 0.0504083 | 0.0345443 | 0.0569627 | 0.06263 | 晚唐 | FALL_THEN_RISE |
| T08 | 王朝功德、郊庙礼乐与忠贤颂咏 | 0.0707872 | 0.0906487 | 0.07204 | 0.0716764 | 盛唐 | UNCERTAIN |
| T09 | 山寺禅居、僧侣往还与清夜幽境 | 0.0312591 | 0.0626579 | 0.0656841 | 0.0734779 | 晚唐 | MONOTONIC_UP |
| T10 | 山水送别、怀友与归隐心绪 | 0.0244303 | 0.0906174 | 0.0807333 | 0.0942716 | 晚唐 | MULTI_TURN |
| T11 | 边塞征战、征人思归与战争悲悯 | 0.11251 | 0.110086 | 0.0596487 | 0.0903796 | 初唐 | FALL_THEN_RISE |
| T12 | 春日花酒、宴游与惜时送别 | 0.110729 | 0.0885562 | 0.104999 | 0.126207 | 晚唐 | FALL_THEN_RISE |

# 8 Aggregation Sensitivity

POEM/AUTHOR 峰值一致 8/12。四时期 Pearson/Spearman 只作描述诊断，不报告显著性 p-value。峰值不同的主题如下；这些差异提示留存作品量与典型作者口径不可混写。

| Topic | POEM峰值 | AUTHOR峰值 | 综合一致度 |
| --- | --- | --- | --- |
| T01 | 初唐 | 中唐 | LOW |
| T06 | 晚唐 | 初唐 | LOW |
| T07 | 盛唐 | 晚唐 | LOW |
| T11 | 盛唐 | 初唐 | LOW |

# 9 Metadata / Sample Sensitivity

六规格 trend robustness 为 HIGH=1、MODERATE=9、LOW=2。STRICT/CORE/EXPANDED planned directions 共发现 13 个符号反转标记，因此不得从三套样本挑选结果。CLEAR representativeness risk：{"topic_03": "LOW", "topic_05": "MODERATE", "topic_06": "MODERATE", "topic_08": "LOW", "topic_09": "MODERATE", "topic_11": "LOW"}。

| Topic | 类型 | 峰值一致率 | 形态一致率 | 稳健度 | 敏感性 |
| --- | --- | --- | --- | --- | --- |
| T01 | BROAD | 0.666667 | 0.5 | LOW | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |
| T02 | BROAD | 0.833333 | 0.5 | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |
| T03 | CLEAR | 0.666667 | 0.833333 | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE |
| T04 | NOISE | 0.833333 | 0.5 | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE |
| T05 | CLEAR | 1 | 0.666667 | HIGH | METADATA_RISK |
| T06 | CLEAR | 0.833333 | 0.833333 | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |
| T07 | NOISE | 0.5 | 0.666667 | LOW | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |
| T08 | CLEAR | 0.833333 | 0.666667 | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE |
| T09 | CLEAR | 1 | 1 | MODERATE | METADATA_RISK |
| T10 | BROAD | 1 | 0.5 | MODERATE | NONE |
| T11 | CLEAR | 0.833333 | 0.333333 | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE |
| T12 | BROAD | 0.833333 | 0.666667 | MODERATE | AGGREGATION_SENSITIVE;SAMPLE_SENSITIVE;METADATA_RISK |

# 10 CLEAR Topic Statistical Validation

CORE_POEM 使用 author-cluster CR1，CORE_AUTHOR 使用 HC3；两套 6-topic omnibus 与两套 24-contrast family 分别作 BH-FDR。CLR 以完整 12 维 composition 计算，只有诊断作用。STRONG：T05 E-H, T05 M-L, T05 E-L, T09 E-L。SUPPORTED：T03 M-L, T06 M-L, T08 H-M, T11 H-M。POEM/AUTHOR 冲突：T06 H-M, T06 E-L, T08 E-L, T11 E-H, T11 M-L。CLR 方向冲突数为 4。

| Topic | 口径 | omnibus q | 方法 |
| --- | --- | --- | --- |
| T03 | AUTHOR | 0.0673629 | HC3 |
| T05 | AUTHOR | 0.000925102 | HC3 |
| T06 | AUTHOR | 0.803705 | HC3 |
| T08 | AUTHOR | 0.822156 | HC3 |
| T09 | AUTHOR | 0.0135453 | HC3 |
| T11 | AUTHOR | 0.0673629 | HC3 |
| T03 | POEM | 0.00971291 | AUTHOR_CLUSTER_CR1 |
| T05 | POEM | 3.53321e-08 | AUTHOR_CLUSTER_CR1 |
| T06 | POEM | 0.0656005 | AUTHOR_CLUSTER_CR1 |
| T08 | POEM | 0.0154426 | AUTHOR_CLUSTER_CR1 |
| T09 | POEM | 8.8912e-07 | AUTHOR_CLUSTER_CR1 |
| T11 | POEM | 0.000476804 | AUTHOR_CLUSTER_CR1 |

| Topic | 比较 | POEM差值 | AUTHOR差值 | Hedges g | 统计状态 |
| --- | --- | --- | --- | --- | --- |
| T03 | M-L | 0.0349405 | 0.0397561 | 0.434648 | SUPPORTED |
| T05 | E-H | -0.0887325 | -0.0955089 | -0.560044 | STRONG |
| T05 | M-L | -0.0193566 | -0.0676481 | -0.522553 | STRONG |
| T05 | E-L | -0.0975755 | -0.129997 | -0.811924 | STRONG |
| T06 | M-L | 0.0373887 | 0.0211631 | 0.19618 | SUPPORTED |
| T08 | H-M | -0.0203572 | -0.0186087 | -0.149496 | SUPPORTED |
| T09 | E-L | 0.0698345 | 0.0422188 | 0.525871 | STRONG |
| T11 | H-M | -0.0402196 | -0.0504371 | -0.387853 | SUPPORTED |

# 11 Author Influence

仅 8 个 STRONG/SUPPORTED candidate 进入 CORE LOAO；删除单位为 `author_id_internal`，并按每个 K12 Topic×Contrast 重新计算 GLOBAL_TOP/PER_PERIOD_TOP 的 Top1/3/5/10。结果为 AUTHOR_ROBUST=0、CONCENTRATED_BUT_ROBUST=7、AUTHOR_SENSITIVE=1。作者贡献集中度表示概率质量集中，不表示文学影响力。

# 12 Robust Candidate Historical Patterns

最终保留 7 条，均为 `KEEP_WITH_CAUTION`；它们只表示在当前 K12、样本及阈值下值得进入研究者阅读队列。

| Topic | 标签 | 比较 | POEM差值 | AUTHOR差值 | 统计状态 | 作者稳健度 | 最终状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T05 | 皇家礼仪、郊庙宴庆与应制颂德 | E-H | -0.0887325 | -0.0955089 | STRONG | CONCENTRATED_BUT_ROBUST | KEEP_WITH_CAUTION |
| T05 | 皇家礼仪、郊庙宴庆与应制颂德 | E-L | -0.0975755 | -0.129997 | STRONG | CONCENTRATED_BUT_ROBUST | KEEP_WITH_CAUTION |
| T05 | 皇家礼仪、郊庙宴庆与应制颂德 | M-L | -0.0193566 | -0.0676481 | STRONG | CONCENTRATED_BUT_ROBUST | KEEP_WITH_CAUTION |
| T06 | 宫闺月夜、相思与女性哀怨 | M-L | 0.0373887 | 0.0211631 | SUPPORTED | CONCENTRATED_BUT_ROBUST | KEEP_WITH_CAUTION |
| T08 | 王朝功德、郊庙礼乐与忠贤颂咏 | H-M | -0.0203572 | -0.0186087 | SUPPORTED | CONCENTRATED_BUT_ROBUST | KEEP_WITH_CAUTION |
| T09 | 山寺禅居、僧侣往还与清夜幽境 | E-L | 0.0698345 | 0.0422188 | STRONG | CONCENTRATED_BUT_ROBUST | KEEP_WITH_CAUTION |
| T11 | 边塞征战、征人思归与战争悲悯 | H-M | -0.0402196 | -0.0504371 | SUPPORTED | CONCENTRATED_BUT_ROBUST | KEEP_WITH_CAUTION |

# 13 BROAD Topic Descriptive Patterns

T01、T02、T10、T12 完成六规格趋势计算，但不进入 primary FDR。即使呈现强趋势，也只能标记为 `RESEARCHER_REVIEW_BROAD`，不能升级为正式 finding。

| Topic | POEM峰值 | AUTHOR峰值 | POEM形态 | AUTHOR形态 | 稳健度 |
| --- | --- | --- | --- | --- | --- |
| T01 | 初唐 | 中唐 | MONOTONIC_DOWN | RISE_THEN_FALL | LOW |
| T02 | 晚唐 | 晚唐 | MONOTONIC_UP | FALL_THEN_RISE | MODERATE |
| T10 | 晚唐 | 晚唐 | MONOTONIC_UP | MULTI_TURN | MODERATE |
| T12 | 晚唐 | 晚唐 | FALL_THEN_RISE | FALL_THEN_RISE | MODERATE |

# 14 NOISE Topic Diagnostics

T04、T07 的四时期均值仍因完整 12 维 composition 而保存；任何时期差异只解释为 residual-topic composition 的变化，不作为独立文学史结论。

| Topic | POEM峰值 | AUTHOR峰值 | 稳健度 |
| --- | --- | --- | --- |
| T04 | 初唐 | 初唐 | MODERATE |
| T07 | 盛唐 | 晚唐 | LOW |

# 15 Comparison with Previous C_K20 Findings

比较仅在 K12 SHA-256 冻结后执行，且基于 semantic family 与 K12→K15→K20 topic-word path，绝不按 Topic ID 直接等同。旧 K20 的 9 条保留 finding 中，5 条在 K12 正式层面复现，1 条只在更宽 BROAD 主题中方向复现，2 条削弱或未复现，1 条因主题重定义不可作单一比较；“合并”与“削弱”可在同一旧 finding 上同时构成解释背景，逐条证据见比较报告。

| K20 finding | K12对应 | 比较结论 |
| --- | --- | --- |
| topic_03 E-H | topic_05;topic_08 | REPRODUCED_IN_REFINED_K12_COMPONENT |
| topic_03 E-L | topic_05;topic_08 | REPRODUCED_IN_REFINED_K12_COMPONENT |
| topic_03 M-L | topic_05;topic_08 | REPRODUCED_IN_REFINED_K12_COMPONENT |
| topic_06 E-L | topic_02 | REPRODUCED_IN_BROADER_K12_EXPLORATORY |
| topic_07 E-L | topic_06 | MERGED_IN_K12_NOT_REPRODUCED |
| topic_08 E-L | topic_09 | REPRODUCED_IN_K12 |
| topic_11 M-L | topic_06 | MERGED_IN_K12_REPRODUCED |
| topic_16 E-L | topic_11 | WEAKENED_IN_K12 |
| topic_19 E-H | topic_01;topic_09 | INCOMPARABLE_TOPIC_REDEFINED |

# 16 Humanistic Interpretation Priorities

优先核读 7 条保留 candidate 的对应时期代表诗、作者归属、同文异署及外部文学史材料。T05 三条应制礼仪比较与 T09 E-L 的统计证据最完整；T06、T08、T11 仍需特别注意样本或作者集中敏感性。研究者解释必须保持“候选”措辞。

# 17 Limitations

Topic 标签仍为 AI_PROVISIONAL；作者时期不是作品创作日期；三套样本受 metadata 覆盖限制；作者集与留存作品不均衡；CLR 与 raw probability 可能给出不同方向；LOAO 只检验作者集中与删除敏感性；跨模型语义对齐尚待研究者正式裁决。因而不得使用“证明”“导致”或“唐诗从此转向”等因果/断代措辞。

# 18 Next Step：B_K12 / C_K12 / D_K12 finding-level robustness

下一阶段应由研究者确认 K12 taxonomy 与本轮候选后，再决定是否执行 B_K12 / C_K12 / D_K12 finding-level robustness。本任务没有训练 B_K12 或 D_K12，没有修改最终 K 选择，也没有改动诗人时期 metadata。
