# QTS Author Influence / Leave-One-Author-Out Report

## 1. Objective
检验既有 C_K20 author-period Topic effects 是否依赖少数高产或高 Topic-contribution 作者。本报告只作 influence / robustness diagnostic。

## 2. Methods
正式分析使用 CORE；以 author_id_internal 为删除单位。分别重算 POEM（剩余诗等权）与 AUTHOR（剩余作者等权）效应，并运行 GLOBAL_TOP、PER_PERIOD_TOP 的 Top1/3/5/10 排除。13组既有统计候选的关键 exclusion 使用 2000 次固定种子作者 cluster bootstrap。

主分类及未加方案前缀的 Top1/3/5 摘要使用 GLOBAL_TOP；PER_PERIOD_TOP 独立报告敏感性状态，不以第二方案中较有利的结果替换主分类。两种方案删去的作者数不同（全局N位与每期N位），不是等删除量对照。所有 delta 均为 period B 减 period A，单位为 Topic probability。

Bootstrap 按时期分层、以作者整簇有放回抽样；基准 seed=20260902，各 Topic/contrast/方案/估计量使用稳定派生子种子。CI 单独报告，不将未跨零的原始 CI 自动继承给删作者后效应；作者稳健性标签不是新的显著性检验。STRICT/EXPANDED 只复核预定优先项与 CORE 敏感/原始反向项，sample_influence_robust 仅指 GLOBAL_TOP Top1 的方向保持。

## 3. Overall Influence Structure
CORE 共 207 位作者、10841 首诗。13组候选状态计数：AUTHOR_ROBUST=1，CONCENTRATED_BUT_ROBUST=8，AUTHOR_SENSITIVE=4，ESTIMAND_DEPENDENT=0，INCONCLUSIVE=0。

## 4. T03
| contrast | original_poem_delta | original_author_delta | poem_loao_max_abs_change | author_loao_max_abs_change | author_robustness_status |
|---|---|---|---|---|---|
| E-H | -0.074561 | -0.065791 | 0.010762 | 0.015195 | CONCENTRATED_BUT_ROBUST |
| E-L | -0.094469 | -0.125133 | 0.010762 | 0.015195 | CONCENTRATED_BUT_ROBUST |
| H-M | 0.003513 | 0.005847 | 0.008546 | 0.010055 | AUTHOR_SENSITIVE |
| M-L | -0.023422 | -0.065189 | 0.003151 | 0.008065 | CONCENTRATED_BUT_ROBUST |

## 5. T06
| contrast | original_poem_delta | original_author_delta | poem_loao_max_abs_change | author_loao_max_abs_change | author_robustness_status |
|---|---|---|---|---|---|
| E-H | 0.032473 | 0.009553 | 0.015380 | 0.021346 | AUTHOR_SENSITIVE |
| E-L | 0.093632 | 0.070234 | 0.017508 | 0.021346 | CONCENTRATED_BUT_ROBUST |
| H-M | 0.028093 | 0.018622 | 0.015380 | 0.012987 | CONCENTRATED_BUT_ROBUST |
| M-L | 0.033066 | 0.042059 | 0.017508 | 0.012821 | CONCENTRATED_BUT_ROBUST |

## 6. T08
| contrast | original_poem_delta | original_author_delta | poem_loao_max_abs_change | author_loao_max_abs_change | author_robustness_status |
|---|---|---|---|---|---|
| E-H | 0.004753 | 0.013286 | 0.004992 | 0.011402 | AUTHOR_SENSITIVE |
| E-L | 0.068992 | 0.041259 | 0.011680 | 0.008156 | CONCENTRATED_BUT_ROBUST |
| H-M | 0.036506 | 0.013537 | 0.007222 | 0.011402 | CONCENTRATED_BUT_ROBUST |
| M-L | 0.027733 | 0.014436 | 0.011680 | 0.008156 | CONCENTRATED_BUT_ROBUST |

## 7. T07
| contrast | original_poem_delta | original_author_delta | conflict_resolution_assessment | max_poem_influence_author |
|---|---|---|---|---|
| H-M | 0.002170 | -0.008727 | RESOLVED_AFTER_TOP5 | 杜甫 |
| M-L | -0.026007 | 0.019053 | PERSISTENT_ESTIMAND_DIFFERENCE | 韦应物 |

## 8. T11
| contrast | original_poem_delta | original_author_delta | conflict_resolution_assessment | max_poem_influence_author |
|---|---|---|---|---|
| E-L | 0.037903 | -0.014800 | PERSISTENT_ESTIMAND_DIFFERENCE | 韩偓 |
| H-M | 0.008195 | -0.022926 | PERSISTENT_ESTIMAND_DIFFERENCE | 李白 |

## 9. T16
| contrast | original_poem_delta | original_author_delta | poem_loao_max_abs_change | author_loao_max_abs_change | author_robustness_status |
|---|---|---|---|---|---|
| E-H | 0.002114 | -0.011650 | 0.008816 | 0.011105 | ESTIMAND_DEPENDENT |
| E-L | -0.018122 | -0.037910 | 0.008816 | 0.011105 | AUTHOR_ROBUST |
| H-M | -0.011010 | -0.033269 | 0.003300 | 0.010912 | AUTHOR_SENSITIVE |
| M-L | -0.009226 | 0.007008 | 0.003300 | 0.006037 | ESTIMAND_DEPENDENT |

## 10. T19
| contrast | original_poem_delta | original_author_delta | conflict_resolution_assessment | max_poem_influence_author |
|---|---|---|---|---|
| M-L | 0.008186 | -0.021997 | PERSISTENT_ESTIMAND_DIFFERENCE | 韦应物 |

## 11. T10 / T20
| topic_id | contrast | original_poem_delta | original_author_delta | author_robustness_status |
|---|---|---|---|---|
| topic_10 | E-H | 0.023865 | 0.032002 | CONCENTRATED_BUT_ROBUST |
| topic_10 | E-L | 0.027566 | 0.057346 | AUTHOR_SENSITIVE |
| topic_10 | H-M | -0.023374 | -0.007707 | AUTHOR_SENSITIVE |
| topic_10 | M-L | 0.027075 | 0.033051 | AUTHOR_SENSITIVE |
| topic_20 | E-H | 0.018416 | 0.015438 | AUTHOR_SENSITIVE |
| topic_20 | E-L | 0.008130 | 0.009397 | AUTHOR_SENSITIVE |
| topic_20 | H-M | 0.000395 | 0.002266 | AUTHOR_SENSITIVE |
| topic_20 | M-L | -0.010681 | -0.008308 | AUTHOR_SENSITIVE |

## 12. Top Contributor Exclusion
TopN contributor 依每个 Topic×contrast 的 Topic mass 排名，不是固定的‘十大诗人’。GLOBAL_TOP 合并两时期排名；PER_PERIOD_TOP 在两侧分别排名。

| topic_id | contrast | top1_exclusion_delta_poem | top3_exclusion_delta_poem | top5_exclusion_delta_poem | top1_exclusion_delta_author | top3_exclusion_delta_author | top5_exclusion_delta_author |
|---|---|---|---|---|---|---|---|
| topic_03 | E-L | -0.101083 | -0.089572 | -0.089888 | -0.126663 | -0.122468 | -0.121522 |
| topic_06 | E-L | 0.076124 | 0.057938 | 0.045004 | 0.066750 | 0.065504 | 0.065840 |
| topic_08 | E-L | 0.067074 | 0.050382 | 0.040149 | 0.040159 | 0.036133 | 0.033759 |

上表为 GLOBAL_TOP。以下为五组重点效应的 PER_PERIOD_TOP 复核（每期各删N位）：

| topic_id | contrast | aggregation | N | new_delta | attenuation_ratio | sign_flip |
|---|---|---|---|---|---|---|
| topic_03 | E-L | AUTHOR | 1 | -0.126593 | -0.011667 | False |
| topic_03 | E-L | AUTHOR | 3 | -0.121833 | 0.026368 | False |
| topic_03 | E-L | AUTHOR | 5 | -0.126491 | -0.010857 | False |
| topic_03 | E-L | POEM | 1 | -0.103616 | -0.096827 | False |
| topic_03 | E-L | POEM | 3 | -0.085430 | 0.095685 | False |
| topic_03 | E-L | POEM | 5 | -0.127087 | -0.345279 | False |
| topic_06 | E-L | AUTHOR | 1 | 0.066232 | 0.056984 | False |
| topic_06 | E-L | AUTHOR | 3 | 0.086811 | -0.236027 | False |
| topic_06 | E-L | AUTHOR | 5 | 0.086809 | -0.235995 | False |
| topic_06 | E-L | POEM | 1 | 0.077955 | 0.167433 | False |
| topic_06 | E-L | POEM | 3 | 0.063045 | 0.326678 | False |
| topic_06 | E-L | POEM | 5 | 0.050668 | 0.458862 | False |
| topic_08 | E-L | AUTHOR | 1 | 0.040569 | 0.016729 | False |
| topic_08 | E-L | AUTHOR | 3 | 0.036818 | 0.107639 | False |
| topic_08 | E-L | AUTHOR | 5 | 0.035192 | 0.147042 | False |
| topic_08 | E-L | POEM | 1 | 0.071610 | -0.037933 | False |
| topic_08 | E-L | POEM | 3 | 0.056787 | 0.176916 | False |
| topic_08 | E-L | POEM | 5 | 0.050965 | 0.261295 | False |
| topic_16 | E-L | AUTHOR | 1 | -0.037833 | 0.002036 | False |
| topic_16 | E-L | AUTHOR | 3 | -0.039439 | -0.040337 | False |
| topic_16 | E-L | AUTHOR | 5 | -0.029844 | 0.212772 | False |
| topic_16 | E-L | POEM | 1 | -0.014303 | 0.210723 | False |
| topic_16 | E-L | POEM | 3 | -0.024707 | -0.363352 | False |
| topic_16 | E-L | POEM | 5 | -0.015632 | 0.137422 | False |
| topic_16 | H-M | AUTHOR | 1 | -0.033927 | -0.019792 | False |
| topic_16 | H-M | AUTHOR | 3 | -0.033953 | -0.020558 | False |
| topic_16 | H-M | AUTHOR | 5 | -0.034954 | -0.050649 | False |
| topic_16 | H-M | POEM | 1 | -0.013833 | -0.256362 | False |
| topic_16 | H-M | POEM | 3 | -0.003369 | 0.693974 | False |
| topic_16 | H-M | POEM | 5 | 0.000961 | 0.912762 | True |

完整两方案 Top1/3/5/10 数值见 top_contributor_exclusion_effects.csv。T16 H-M 在 PER_PERIOD_TOP Top5 的 POEM 效应反向；T10 E-L 与 T20 E-H 在 PER_PERIOD_TOP 方向保持，但不推翻 GLOBAL_TOP 及 LOAO 主诊断。T07 E-L 的 PER_PERIOD_TOP Top5 AUTHOR 衰减约51.44%，因此‘保留’不表示效应幅度不变。

## 13. Author Robustness Classification
AUTHOR_ROBUST、CONCENTRATED_BUT_ROBUST、AUTHOR_SENSITIVE、ESTIMAND_DEPENDENT、INCONCLUSIVE 均由配置化 audit heuristic 生成；复合 influence score 只用于排序。

规则优先级：原始 POEM/AUTHOR 方向不同先列 ESTIMAND_DEPENDENT；必要 LOAO 汇总不可计算则列 INCONCLUSIVE；原始 |delta|≥0.005 时的 LOAO/Top1/3/5 反向，或任一衰减≥70%，列 AUTHOR_SENSITIVE。原始 |delta|<1e-06 时衰减设 NA；所有 raw sign flip 仍完整保留。

其余结果还必须满足两种估计量 Top5 无 raw sign flip 且剩余绝对幅度至少为原值的30%。任一时期 Top1 share≥30% 或 Top5 share≥70% 列 CONCENTRATED_BUT_ROBUST，否则为 AUTHOR_ROBUST。Top10 是额外压力测试，不参与分类。以上为数值审计启发式，并非公认效应量等级。

以下结果的 GLOBAL_TOP 主分类与 PER_PERIOD_TOP 敏感性分类不同；不得把主分类误述为所有排除方式一致：

| topic_id | contrast | author_robustness_status | per_period_author_robustness_status |
|---|---|---|---|
| topic_06 | M-L | CONCENTRATED_BUT_ROBUST | AUTHOR_SENSITIVE |
| topic_08 | M-L | CONCENTRATED_BUT_ROBUST | AUTHOR_SENSITIVE |
| topic_19 | E-L | AUTHOR_SENSITIVE | CONCENTRATED_BUT_ROBUST |
| topic_20 | E-H | AUTHOR_SENSITIVE | CONCENTRATED_BUT_ROBUST |
| topic_20 | E-L | AUTHOR_SENSITIVE | CONCENTRATED_BUT_ROBUST |
| topic_20 | M-L | AUTHOR_SENSITIVE | CONCENTRATED_BUT_ROBUST |

## 14. Candidate Trends After Author Influence Check
| topic_id | contrast | previous_validation_status | author_robustness_status | final_author_influence_assessment | finding_level_bd_eligible |
|---|---|---|---|---|---|
| topic_03 | E-H | SUPPORTED | CONCENTRATED_BUT_ROBUST | KEEP_WITH_CAUTION | True |
| topic_03 | E-L | STRONG | CONCENTRATED_BUT_ROBUST | KEEP_WITH_CAUTION | True |
| topic_03 | M-L | STRONG | CONCENTRATED_BUT_ROBUST | KEEP_WITH_CAUTION | True |
| topic_06 | E-L | STRONG | CONCENTRATED_BUT_ROBUST | KEEP_WITH_CAUTION | True |
| topic_07 | E-L | SUPPORTED | CONCENTRATED_BUT_ROBUST | KEEP_WITH_CAUTION | True |
| topic_08 | E-L | STRONG | CONCENTRATED_BUT_ROBUST | KEEP_WITH_CAUTION | True |
| topic_10 | E-L | SUPPORTED | AUTHOR_SENSITIVE | DOWNGRADE_AUTHOR_SENSITIVE | False |
| topic_11 | M-L | SUPPORTED | CONCENTRATED_BUT_ROBUST | KEEP_WITH_CAUTION | True |
| topic_16 | E-L | SUPPORTED | AUTHOR_ROBUST | UPGRADE_KEEP | True |
| topic_16 | H-M | SUPPORTED | AUTHOR_SENSITIVE | DOWNGRADE_AUTHOR_SENSITIVE | False |
| topic_19 | E-H | SUPPORTED | CONCENTRATED_BUT_ROBUST | KEEP_WITH_CAUTION | True |
| topic_19 | E-L | SUPPORTED | AUTHOR_SENSITIVE | DOWNGRADE_AUTHOR_SENSITIVE | False |
| topic_20 | E-H | SUPPORTED | AUTHOR_SENSITIVE | DOWNGRADE_AUTHOR_SENSITIVE | False |

主分类保留的9组候选在 PER_PERIOD_TOP 复核中也均为 AUTHOR_ROBUST 或 CONCENTRATED_BUT_ROBUST。4组降级仍保留完整估计量与两方案结果，不因敏感性方案较有利而恢复候选资格。

## 15. Limitations
author period 不等于 poem composition date；author influence 不等于文学史重要性；高影响可能来自创作量、现存作品量、《全唐诗》收录量或 Topic probability，不能区分为单一原因。高产不等于因果影响；survival / anthology bias 仍在。即使 UPGRADE_KEEP，也只表示作者影响检查后仍稳健的统计候选趋势。
