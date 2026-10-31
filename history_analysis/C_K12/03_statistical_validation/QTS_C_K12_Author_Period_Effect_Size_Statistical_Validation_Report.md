# QTS C_K12 Author-Period Effect Size and Statistical Validation Report

# 1 Objective

在趋势阶段冻结后，对六个 CLEAR Topics 的四组预设时期对比进行效应量与稳健推断。差值统一为 period B − period A。

# 2 Scope

Primary family 只含 T03、T05、T06、T08、T09、T11。BROAD 仅另存 EXPLORATORY_BROAD 描述，NOISE 不进入正式推断。

# 3 Data and Estimands

CORE_POEM 为每诗等权均值，使用 author-cluster CR1；CORE_AUTHOR 为一作者一行，使用 HC3。POEM uncertainty 未把诗视为独立作者。

# 4 Omnibus Tests

| topic_id | topic_label | aggregation | wald_f | df_num | df_denom | p_value | q_value | fdr_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| topic_03 | 江水舟行、秋景与客中怀归 | AUTHOR | 2.79894 | 3 | 203 | 0.0411557 | 0.0673629 | False |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | AUTHOR | 7.06217 | 3 | 203 | 0.000154184 | 0.000925102 | True |
| topic_06 | 宫闺月夜、相思与女性哀怨 | AUTHOR | 0.51887 | 3 | 203 | 0.669754 | 0.803705 | False |
| topic_08 | 王朝功德、郊庙礼乐与忠贤颂咏 | AUTHOR | 0.304455 | 3 | 203 | 0.822156 | 0.822156 | False |
| topic_09 | 山寺禅居、僧侣往还与清夜幽境 | AUTHOR | 4.48401 | 3 | 203 | 0.00451509 | 0.0135453 | True |
| topic_11 | 边塞征战、征人思归与战争悲悯 | AUTHOR | 2.73181 | 3 | 203 | 0.0449086 | 0.0673629 | False |
| topic_03 | 江水舟行、秋景与客中怀归 | POEM | 4.20826 | 3 | 206 | 0.00647527 | 0.00971291 | True |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | POEM | 15.169 | 3 | 206 | 5.88869e-09 | 3.53321e-08 | True |
| topic_06 | 宫闺月夜、相思与女性哀怨 | POEM | 2.43867 | 3 | 206 | 0.0656005 | 0.0656005 | False |
| topic_08 | 王朝功德、郊庙礼乐与忠贤颂咏 | POEM | 3.68627 | 3 | 206 | 0.0128688 | 0.0154426 | True |
| topic_09 | 山寺禅居、僧侣往还与清夜幽境 | POEM | 11.9661 | 3 | 206 | 2.96373e-07 | 8.8912e-07 | True |
| topic_11 | 边塞征战、征人思归与战争悲悯 | POEM | 6.72287 | 3 | 206 | 0.000238402 | 0.000476804 | True |

# 5 FDR Families

CORE_POEM 六个 omnibus 为一个 BH family，CORE_AUTHOR 六个 omnibus 为另一 family。planned contrasts 同样按 aggregation 分成两个各24项的 family。

# 6 Planned Contrasts

E-H、H-M、M-L、E-L 均在分析前固定，方向为后期减前期。

# 7 Effect Sizes

| topic_id | topic_label | contrast | raw_effect_poem | poem_ci_low | poem_ci_high | raw_effect_author | author_ci_low | author_ci_high | hedges_g | cliff_delta | q_poem | q_author | sample_robustness | clr_agreement | validation_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| topic_03 | 江水舟行、秋景与客中怀归 | E-H | 0.0206325 | -0.00480905 | 0.0672939 | 0.0477327 | 0.00784755 | 0.0901411 | 0.481155 | 0.178673 | 0.303793 | 0.12936 | LOW | True | TENTATIVE |
| topic_03 | 江水舟行、秋景与客中怀归 | H-M | -0.0370234 | -0.0703795 | -0.0149297 | -0.040416 | -0.0805491 | -0.00297157 | -0.445486 | -0.143177 | 0.00330274 | 0.16347 | LOW | True | TENTATIVE |
| topic_03 | 江水舟行、秋景与客中怀归 | M-L | 0.0349405 | -0.0070086 | 0.0826789 | 0.0397561 | 0.00273419 | 0.0843912 | 0.434648 | 0.240964 | 0.283328 | 0.175987 | MODERATE | True | SUPPORTED |
| topic_03 | 江水舟行、秋景与客中怀归 | E-L | 0.0185496 | -0.0267202 | 0.078009 | 0.0470729 | 0.00729846 | 0.0927843 | 0.468093 | 0.3 | 0.541123 | 0.153409 | LOW | True | TENTATIVE |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | E-H | -0.0887325 | -0.13941 | -0.0534136 | -0.0955089 | -0.170138 | -0.0273726 | -0.560044 | -0.340896 | 5.27515e-06 | 0.0987785 | HIGH | True | STRONG |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | H-M | 0.0105137 | -0.0173054 | 0.0273159 | 0.0331604 | -0.0144263 | 0.0777956 | 0.23905 | 0.100028 | 0.303793 | 0.372369 | LOW | True | CONFLICTING |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | M-L | -0.0193566 | -0.034694 | -0.00865578 | -0.0676481 | -0.105912 | -0.0282881 | -0.522553 | -0.235542 | 0.00665982 | 0.0101433 | HIGH | True | STRONG |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | E-L | -0.0975755 | -0.147624 | -0.0719589 | -0.129997 | -0.199363 | -0.0638998 | -0.811924 | -0.492683 | 1.63019e-07 | 0.00718341 | HIGH | True | STRONG |
| topic_06 | 宫闺月夜、相思与女性哀怨 | E-H | -0.0313813 | -0.0792829 | 0.000697602 | -0.0191246 | -0.082782 | 0.0387241 | -0.130937 | -0.068633 | 0.164292 | 0.779059 | LOW | True | UNSUPPORTED |
| topic_06 | 宫闺月夜、相思与女性哀怨 | H-M | 0.0109518 | -0.0162933 | 0.0365113 | -0.00472932 | -0.0503241 | 0.0392719 | -0.0392168 | -0.000840572 | 0.493155 | 0.986187 | LOW | False | CONFLICTING |
| topic_06 | 宫闺月夜、相思与女性哀怨 | M-L | 0.0373887 | 0.00339268 | 0.0788762 | 0.0211631 | -0.0174977 | 0.0595763 | 0.19618 | 0.256024 | 0.095527 | 0.527525 | HIGH | True | SUPPORTED |
| topic_06 | 宫闺月夜、相思与女性哀怨 | E-L | 0.0169591 | -0.0363386 | 0.0619845 | -0.00269079 | -0.0608043 | 0.0482639 | -0.0205674 | 0.15 | 0.505668 | 0.986187 | LOW | False | CONFLICTING |
| topic_08 | 王朝功德、郊庙礼乐与忠贤颂咏 | E-H | 0.00934324 | -0.0102686 | 0.0347147 | 0.0198615 | -0.0227801 | 0.0637405 | 0.188619 | 0.092456 | 0.438509 | 0.662373 | LOW | True | UNSUPPORTED |
| topic_08 | 王朝功德、郊庙礼乐与忠贤颂咏 | H-M | -0.0203572 | -0.0433956 | -0.00331233 | -0.0186087 | -0.0619662 | 0.0256402 | -0.149496 | -0.22051 | 0.045199 | 0.671552 | HIGH | True | SUPPORTED |
| topic_08 | 王朝功德、郊庙礼乐与忠贤颂咏 | M-L | -0.0083753 | -0.0253567 | 0.00817873 | -0.000363664 | -0.0403572 | 0.0413069 | -0.00306679 | 0.13012 | 0.438509 | 0.986187 | LOW | False | CONFLICTING |
| topic_08 | 王朝功德、郊庙礼乐与忠贤颂咏 | E-L | -0.0193893 | -0.0399547 | 0.000761582 | 0.000889153 | -0.0384877 | 0.0423961 | 0.00952817 | 0.00853659 | 0.095527 | 0.986187 | LOW | True | CONFLICTING |
| topic_09 | 山寺禅居、僧侣往还与清夜幽境 | E-H | 0.000939301 | -0.012435 | 0.0273196 | 0.0313988 | 0.000905191 | 0.0659704 | 0.399813 | 0.084515 | 0.906942 | 0.175987 | MODERATE | False | CONFLICTING |
| topic_09 | 山寺禅居、僧侣往还与清夜幽境 | H-M | 0.045436 | 0.0179161 | 0.0660722 | 0.0030262 | -0.0330939 | 0.0356761 | 0.034418 | 0.024937 | 0.000220433 | 0.986187 | MODERATE | True | TENTATIVE |
| topic_09 | 山寺禅居、僧侣往还与清夜幽境 | M-L | 0.0234592 | -0.0140424 | 0.0585033 | 0.00779376 | -0.0264313 | 0.0460106 | 0.087409 | 0.103614 | 0.303793 | 0.908229 | MODERATE | True | UNSUPPORTED |
| topic_09 | 山寺禅居、僧侣往还与清夜幽境 | E-L | 0.0698345 | 0.0359495 | 0.104266 | 0.0422188 | 0.00986446 | 0.0807958 | 0.525871 | 0.230488 | 0.000286508 | 0.122176 | HIGH | True | STRONG |
| topic_11 | 边塞征战、征人思归与战争悲悯 | E-H | 0.00766094 | -0.0396953 | 0.0393588 | -0.00242436 | -0.0676202 | 0.0705443 | -0.0146217 | -0.125355 | 0.692247 | 0.986187 | LOW | True | CONFLICTING |
| topic_11 | 边塞征战、征人思归与战争悲悯 | H-M | -0.0402196 | -0.0653533 | -3.08782e-05 | -0.0504371 | -0.113352 | 0.00220413 | -0.387853 | -0.163911 | 0.00522042 | 0.249439 | HIGH | True | SUPPORTED |
| topic_11 | 边塞征战、征人思归与战争悲悯 | M-L | -0.00973568 | -0.0351724 | 0.0104563 | 0.0307309 | -0.0123477 | 0.0777543 | 0.299 | 0.239759 | 0.452124 | 0.392052 | LOW | True | CONFLICTING |
| topic_11 | 边塞征战、征人思归与战争悲悯 | E-L | -0.0422943 | -0.0799577 | -0.0134733 | -0.0221306 | -0.0790064 | 0.034954 | -0.164432 | -0.109756 | 0.0315675 | 0.692803 | LOW | True | TENTATIVE |

# 8 Bootstrap

所有 raw probability difference 使用 5000 次时期分层 author-cluster bootstrap，seed=20260901。

# 9 Standardized Effects

AUTHOR 口径报告 Hedges' g 与 Cliff's delta；POEM 口径另保留以作者分布尺度计算的 descriptive clustered SMD。

# 10 Sample Sensitivity

六规格的符号反转、幅度敏感性与方向多数均按旧阈值保存，未针对 K12 调整。

# 11 CLR Sensitivity

CLR 在完整12维 composition 上临时计算；raw probability 仍是主结果。真零数量为 0。

# 12 Statistical Status

沿用 STRONG、SUPPORTED、TENTATIVE、UNSUPPORTED、CONFLICTING。状态整合 q、bootstrap、效应量、aggregation、样本、代表性与 CLR，但不是文学史确认。

# 13 Statistical Candidates

| topic_id | topic_label | contrast | direction | raw_effect_poem | raw_effect_author | effect_size_author | hedges_g | cliff_delta | bootstrap_support_poem | bootstrap_support_author | q_poem | q_author | aggregation_agreement | sample_robustness | representativeness_risk | clr_agreement | validation_status | validation_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| topic_03 | 江水舟行、秋景与客中怀归 | M-L | POSITIVE | 0.0349405 | 0.0397561 | 0.434648 | 0.434648 | 0.240964 | UNCERTAIN | SUPPORTED_POSITIVE | 0.283328 | 0.175987 | MODERATE | MODERATE | LOW | True | SUPPORTED | 两种 CORE estimand 同号且至少一项 bootstrap 支持该方向; 作者等权标准化效应达到预设非微小阈值; planned contrast FDR 未通过，结论依赖效应量与区间; 样本定义改变时幅度达到敏感阈值 |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | E-H | NEGATIVE | -0.0887325 | -0.0955089 | -0.560044 | -0.560044 | -0.340896 | SUPPORTED_NEGATIVE | SUPPORTED_NEGATIVE | 5.27515e-06 | 0.0987785 | HIGH | HIGH | MODERATE | True | STRONG | 两种 CORE estimand 同号且 bootstrap 同向支持; 至少一个 planned contrast 通过相应 FDR family; 作者等权标准化效应达到预设非微小阈值; metadata representativeness risk 为 MODERATE |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | M-L | NEGATIVE | -0.0193566 | -0.0676481 | -0.522553 | -0.522553 | -0.235542 | SUPPORTED_NEGATIVE | SUPPORTED_NEGATIVE | 0.00665982 | 0.0101433 | HIGH | HIGH | MODERATE | True | STRONG | 两种 CORE estimand 同号且 bootstrap 同向支持; 至少一个 planned contrast 通过相应 FDR family; 作者等权标准化效应达到预设非微小阈值; metadata representativeness risk 为 MODERATE |
| topic_05 | 皇家礼仪、郊庙宴庆与应制颂德 | E-L | NEGATIVE | -0.0975755 | -0.129997 | -0.811924 | -0.811924 | -0.492683 | SUPPORTED_NEGATIVE | SUPPORTED_NEGATIVE | 1.63019e-07 | 0.00718341 | HIGH | HIGH | MODERATE | True | STRONG | 两种 CORE estimand 同号且 bootstrap 同向支持; 至少一个 planned contrast 通过相应 FDR family; 作者等权标准化效应达到预设非微小阈值; metadata representativeness risk 为 MODERATE |
| topic_06 | 宫闺月夜、相思与女性哀怨 | M-L | POSITIVE | 0.0373887 | 0.0211631 | 0.19618 | 0.19618 | 0.256024 | SUPPORTED_POSITIVE | UNCERTAIN | 0.095527 | 0.527525 | MODERATE | HIGH | MODERATE | True | SUPPORTED | 两种 CORE estimand 同号且至少一项 bootstrap 支持该方向; 作者等权标准化效应达到预设非微小阈值; planned contrast FDR 未通过，结论依赖效应量与区间; metadata representativeness risk 为 MODERATE |
| topic_08 | 王朝功德、郊庙礼乐与忠贤颂咏 | H-M | NEGATIVE | -0.0203572 | -0.0186087 | -0.149496 | -0.149496 | -0.22051 | SUPPORTED_NEGATIVE | UNCERTAIN | 0.045199 | 0.671552 | MODERATE | HIGH | LOW | True | SUPPORTED | 两种 CORE estimand 同号且至少一项 bootstrap 支持该方向; 作者等权标准化效应达到预设非微小阈值 |
| topic_09 | 山寺禅居、僧侣往还与清夜幽境 | E-L | POSITIVE | 0.0698345 | 0.0422188 | 0.525871 | 0.525871 | 0.230488 | SUPPORTED_POSITIVE | SUPPORTED_POSITIVE | 0.000286508 | 0.122176 | HIGH | HIGH | MODERATE | True | STRONG | 两种 CORE estimand 同号且 bootstrap 同向支持; 至少一个 planned contrast 通过相应 FDR family; 作者等权标准化效应达到预设非微小阈值; metadata representativeness risk 为 MODERATE |
| topic_11 | 边塞征战、征人思归与战争悲悯 | H-M | NEGATIVE | -0.0402196 | -0.0504371 | -0.387853 | -0.387853 | -0.163911 | SUPPORTED_NEGATIVE | UNCERTAIN | 0.00522042 | 0.249439 | MODERATE | HIGH | LOW | True | SUPPORTED | 两种 CORE estimand 同号且至少一项 bootstrap 支持该方向; 作者等权标准化效应达到预设非微小阈值 |

# 14 Limitations

作者时期不等于诗作年代；统计关联不构成因果解释；AI_PROVISIONAL taxonomy 与年代缺失选择性仍需研究者复核。
