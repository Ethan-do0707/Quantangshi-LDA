# QTS C_K12 Author Influence and LOAO Report

# 1 Objective

只对第二阶段 STRONG/SUPPORTED 的 Topic×contrast 候选进行作者影响诊断。

# 2 Sample

主分析使用 CORE。删除单位为 author_id_internal，每次删除该作者在样本内的全部诗。

# 3 Estimands

每次排除后分别重算 POEM 与 AUTHOR 两种 estimand，不以其中一套替代另一套。

# 4 Leave-One-Author-Out

完整逐作者结果见 C_K12_LOAO_all_authors.csv；作者排序由 K12 Topic mass 和效应变化重新计算，没有沿用 K20 作者名单。

# 5 Top Contributor Exclusion

GLOBAL_TOP 与 PER_PERIOD_TOP 均执行 Top1、Top3、Top5、Top10；主分类仍采用 GLOBAL_TOP。

# 6 Bootstrap

候选的 Top1/3/5 排除结果使用 2000 次 author-cluster bootstrap，固定 seed=20260902。

# 7 Author Robustness

| topic_id | topic_label | contrast | previous_validation_status | original_poem_delta | original_author_delta | top1_share | top5_share | author_robustness_status | per_period_author_robustness_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| topic_03 | 江水舟行、秋景与客中怀归 | M-L | SUPPORTED | 0.0349405 | 0.0397561 | 0.366487 | 0.776099 | AUTHOR_SENSITIVE | AUTHOR_SENSITIVE |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | E-H | STRONG | -0.0887325 | -0.0955089 | 0.25846 | 0.760326 | CONCENTRATED_BUT_ROBUST | CONCENTRATED_BUT_ROBUST |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | E-L | STRONG | -0.0975755 | -0.129997 | 0.25846 | 0.720266 | CONCENTRATED_BUT_ROBUST | CONCENTRATED_BUT_ROBUST |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | M-L | STRONG | -0.0193566 | -0.0676481 | 0.19632 | 0.720266 | CONCENTRATED_BUT_ROBUST | CONCENTRATED_BUT_ROBUST |
| topic_06 | 宫闺月夜、相思与女性哀怨 | M-L | SUPPORTED | 0.0373887 | 0.0211631 | 0.232463 | 0.790102 | CONCENTRATED_BUT_ROBUST | AUTHOR_SENSITIVE |
| topic_08 | 王朝功德、郊庙礼乐与忠贤颂咏 | H-M | SUPPORTED | -0.0203572 | -0.0186087 | 0.378886 | 0.8297 | CONCENTRATED_BUT_ROBUST | CONCENTRATED_BUT_ROBUST |
| topic_09 | 山寺禅居、僧侣往还与清夜幽境 | E-L | STRONG | 0.0698345 | 0.0422188 | 0.388615 | 0.829516 | CONCENTRATED_BUT_ROBUST | CONCENTRATED_BUT_ROBUST |
| topic_11 | 边塞征战、征人思归与战争悲悯 | H-M | SUPPORTED | -0.0402196 | -0.0504371 | 0.367619 | 0.901688 | CONCENTRATED_BUT_ROBUST | AUTHOR_SENSITIVE |

# 8 Final Candidate Assessment

| topic_id | topic_label | contrast | author_robustness_status | trend_robustness | representativeness_risk | sample_robustness | clr_agreement | final_historical_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| topic_03 | 江水舟行、秋景与客中怀归 | M-L | AUTHOR_SENSITIVE | MODERATE | LOW | MODERATE | True | DOWNGRADE_AUTHOR_SENSITIVE |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | E-H | CONCENTRATED_BUT_ROBUST | HIGH | MODERATE | HIGH | True | KEEP_WITH_CAUTION |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | E-L | CONCENTRATED_BUT_ROBUST | HIGH | MODERATE | HIGH | True | KEEP_WITH_CAUTION |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | M-L | CONCENTRATED_BUT_ROBUST | HIGH | MODERATE | HIGH | True | KEEP_WITH_CAUTION |
| topic_06 | 宫闺月夜、相思与女性哀怨 | M-L | CONCENTRATED_BUT_ROBUST | MODERATE | MODERATE | HIGH | True | KEEP_WITH_CAUTION |
| topic_08 | 王朝功德、郊庙礼乐与忠贤颂咏 | H-M | CONCENTRATED_BUT_ROBUST | MODERATE | LOW | HIGH | True | KEEP_WITH_CAUTION |
| topic_09 | 山寺禅居、僧侣往还与清夜幽境 | E-L | CONCENTRATED_BUT_ROBUST | MODERATE | MODERATE | HIGH | True | KEEP_WITH_CAUTION |
| topic_11 | 边塞征战、征人思归与战争悲悯 | H-M | CONCENTRATED_BUT_ROBUST | MODERATE | LOW | HIGH | True | KEEP_WITH_CAUTION |

# 9 Interpretation

AUTHOR_ROBUST 或 CONCENTRATED_BUT_ROBUST 只表示本样本内作者影响审计允许保留；不等于某位作者造成时期变化，也不代表作者文学史重要性。

# 10 Limitations

作者时期不等于诗作年代。LOAO 是 audit heuristic，方向保持也不保证排除后区间不跨零；taxonomy 仍为 AI_PROVISIONAL。
