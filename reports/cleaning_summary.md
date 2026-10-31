# 《全唐诗》LDA 语料清洗报告

## Run

- 开始时间：2026-08-30T10:21:13+08:00
- 本报告生成时耗时：25.578 秒
- 清洗档位：`default_profile`
- 最小 TF：5
- 最小 DF：3
- 自动最大 DF：None
- 语义映射：仅输出旁表，未改写主语料
- 异常记录：全部保留，只写入人工审核文件

### 实际读取文件

| 角色 | 路径 | SHA-256 |
| --- | --- | --- |
| config | config/cleaning_config.yaml | e50679f9c28e1778247bb610c4afdf80aea2b75265973e78a8b2d260f900b687 |
| corpus | segmentation/data/poems.jsonl | bf8a292bd6f7517c8d965e29aa2cb7c38900abd547b631c8ccb8ef69f97a6a98 |
| tokenized_corpus | segmentation/results/sikufenci_manual_gold.jsonl | 0fa04536fa9a00ec13484f160069d399c9ebfd07ab8400263a08df474a826caf |
| normalization | segmentation/manual_gold/resources/01_variant_normalization.tsv | 6b05bef80d6dcfa492bc6f8274a420a8a462c4a3ae5f268930b8a1e24a9d1393 |
| stopwords_core | segmentation/manual_gold/resources/02_stopwords_core.txt | 6a3479c353bf94dd8fdb137a3ae27f7aefbe451697712b1bcda9d2dfd0a8c6b8 |
| protected_words | segmentation/manual_gold/resources/03_protected_terms.tsv | f3169c6d56d0fe046d0fdc761c11f7e90831fc52a33f9210388b5d150af3cdc6 |
| filtering_rules | segmentation/manual_gold/resources/04_tf_df_rules.json | 9499abd7af4685412fcebb12f668cbbf81557664aa9c86ce57d32c7c24249837 |
| semantic_mapping | segmentation/manual_gold/resources/05_semantic_mapping.tsv | 783d84fae79920b03e3c561a2e561b0e62e310a950a65c31f46abf4981456dd6 |

### 规则资源规模

| 资源 | 记录数 |
| --- | --- |
| normalization | 4 |
| stopwords_core | 105 |
| protected_words | 182 |
| filter_profiles | 2 |
| semantic_mapping | 27 |

## Corpus

- 清洗前诗歌数量：43,030
- 作者总数（非空作者去重）：2,528
- 每首诗平均原始 token 数：57.645782
- 每首诗原始 token 数中位数：42.000000
- 最短诗歌原始 token 数：0
- 最长诗歌原始 token 数：3,867
- 清洗后有效诗歌数量：43,014
- 清洗后零 token 诗歌数量：16
- 空原文数量：11
- 空分词记录数量：11
- 完全重复记录数量（不含首次出现）：2
- 缺失作者数量：0
- 缺失题目数量：0
- 主语料没有自动删除任何诗；零 token 行仍保留并与 `lda_tokens.txt` 对齐。

## Vocabulary

- 清洗前 token 总数（含标点等非汉字 token）：2,480,498
- 清洗后 token 总数：1,485,707
- 清洗前 vocabulary size：123,714
- 清洗后 vocabulary size：23,123
- 清洗前纯汉字 token 总数：2,014,981
- 清洗前纯汉字 vocabulary size：123,490

## Cleaning

- normalization 实际替换次数：0
- STOPWORD 删除词次 / 词型：383,023 / 105
- LOW_TF 删除词次 / 词型：146,003 / 100,235
- LOW_DF 删除词次 / 词型：248 / 27
- HIGH_DF 删除词次 / 词型：0 / 0
- OTHER_RULE 删除词次 / 词型：465,517 / 224
- protected words 保留词次：22,500
- protected words 实际挽救词次 / 词型：42 / 14
- 保护词与停用词冲突：0 个词型

`OTHER_RULE` 在当前规则中表示非纯汉字 token（主要为标点）；没有新增任何其他推断规则。 `token_statistics.csv` 为保证所有删除可追溯，统计规范化视图中的全部 token，因此也保留已标为停用词或非汉字的统计行。

## Retention

- Token Retention Rate：59.895513%
- 汉字 Token Retention Rate：73.733053%
- Vocabulary Retention Rate：18.690690%
- 汉字 Vocabulary Retention Rate：18.724593%
- Poem Retention Rate：99.962817%

## 数据格式与质量问题

| issue_type | 记录数 |
| --- | --- |
| DUPLICATE_RECORD | 2 |
| EMPTY_TEXT | 11 |
| EMPTY_TOKENS | 11 |
| TEXT_TOKEN_MISMATCH | 1 |

详细记录见 `reports/data_quality_issues.csv`；这些记录均未自动删除。

## 规则冲突

保护词与停用词交集共有 0 个词型，均按保护词优先保留。

## 删除数量最多的前 100 个 token

| token | 删除词次 | DF | DF_ratio | 主删除原因 |
| --- | --- | --- | --- | --- |
| ， | 231936 | 42973 | 0.99893071 | OTHER_RULE |
| 。 | 228994 | 43014 | 0.99988377 | OTHER_RULE |
| 不 | 25664 | 17549 | 0.40793603 | STOPWORD |
| 无 | 15082 | 11955 | 0.27790046 | STOPWORD |
| 一 | 14321 | 11412 | 0.26527813 | STOPWORD |
| 有 | 12504 | 9885 | 0.22978219 | STOPWORD |
| 何 | 10203 | 8659 | 0.20128315 | STOPWORD |
| 来 | 9729 | 8515 | 0.19793580 | STOPWORD |
| 上 | 9222 | 8045 | 0.18701039 | STOPWORD |
| 中 | 9007 | 7792 | 0.18112927 | STOPWORD |
| 相 | 8742 | 7219 | 0.16780957 | STOPWORD |
| 自 | 8709 | 7492 | 0.17415561 | STOPWORD |
| 为 | 8605 | 6800 | 0.15806969 | STOPWORD |
| 时 | 8222 | 7327 | 0.17032009 | STOPWORD |
| 见 | 7758 | 6795 | 0.15795346 | STOPWORD |
| 此 | 7449 | 6495 | 0.15097980 | STOPWORD |
| 在 | 7368 | 6588 | 0.15314164 | STOPWORD |
| 得 | 7108 | 6136 | 0.14263465 | STOPWORD |
| 未 | 6855 | 6121 | 0.14228597 | STOPWORD |
| 多 | 6327 | 5712 | 0.13277854 | STOPWORD |
| 下 | 6189 | 5635 | 0.13098863 | STOPWORD |
| 是 | 6164 | 5496 | 0.12775750 | STOPWORD |
| 如 | 6010 | 4865 | 0.11308957 | STOPWORD |
| 我 | 5399 | 3886 | 0.09033218 | STOPWORD |
| 出 | 5317 | 4863 | 0.11304307 | STOPWORD |
| 入 | 5277 | 4861 | 0.11299658 | STOPWORD |
| 与 | 5222 | 4487 | 0.10430275 | STOPWORD |
| 已 | 5025 | 4520 | 0.10506985 | STOPWORD |
| 欲 | 4922 | 4512 | 0.10488389 | STOPWORD |
| 谁 | 4895 | 4466 | 0.10381459 | STOPWORD |
| 还 | 4757 | 4452 | 0.10348916 | STOPWORD |
| 闻 | 4658 | 4274 | 0.09935145 | STOPWORD |
| 可 | 4524 | 3864 | 0.08982078 | STOPWORD |
| 更 | 4429 | 4144 | 0.09632953 | STOPWORD |
| 向 | 4404 | 4131 | 0.09602734 | STOPWORD |
| 应 | 4375 | 4150 | 0.09646900 | STOPWORD |
| 成 | 4290 | 3913 | 0.09095981 | STOPWORD |
| 同 | 4246 | 3927 | 0.09128525 | STOPWORD |
| 从 | 3925 | 3626 | 0.08428834 | STOPWORD |
| 将 | 3736 | 3450 | 0.08019712 | STOPWORD |
| 莫 | 3732 | 3392 | 0.07884888 | STOPWORD |
| 能 | 3672 | 3276 | 0.07615240 | STOPWORD |
| 似 | 3663 | 3319 | 0.07715196 | STOPWORD |
| 复 | 3529 | 3197 | 0.07431600 | STOPWORD |
| 作 | 3463 | 3096 | 0.07196820 | STOPWORD |
| 亦 | 3354 | 2955 | 0.06869058 | STOPWORD |
| 起 | 3058 | 2922 | 0.06792348 | STOPWORD |
| 非 | 3038 | 2687 | 0.06246077 | STOPWORD |
| 犹 | 3001 | 2870 | 0.06671471 | STOPWORD |
| 当 | 2982 | 2730 | 0.06346033 | STOPWORD |
| 之 | 2875 | 1726 | 0.04012181 | STOPWORD |
| 所 | 2660 | 2210 | 0.05137265 | STOPWORD |
| 共 | 2519 | 2422 | 0.05630070 | STOPWORD |
| 岂 | 2486 | 2277 | 0.05293010 | STOPWORD |
| 若 | 2383 | 2098 | 0.04876915 | STOPWORD |
| 因 | 2370 | 2259 | 0.05251168 | STOPWORD |
| 临 | 2360 | 2280 | 0.05299984 | STOPWORD |
| 曾 | 2215 | 2111 | 0.04907134 | STOPWORD |
| 对 | 2205 | 2102 | 0.04886213 | STOPWORD |
| 又 | 2093 | 1943 | 0.04516609 | STOPWORD |
| 只 | 2043 | 1900 | 0.04416653 | STOPWORD |
| 唯 | 2042 | 1955 | 0.04544504 | STOPWORD |
| 且 | 1974 | 1822 | 0.04235338 | STOPWORD |
| 听 | 1819 | 1731 | 0.04023803 | STOPWORD |
| 须 | 1794 | 1662 | 0.03863409 | STOPWORD |
| 其 | 1776 | 1244 | 0.02891746 | STOPWORD |
| 皆 | 1736 | 1601 | 0.03721611 | STOPWORD |
| 兮 | 1718 | 345 | 0.00801971 | STOPWORD |
| 以 | 1681 | 1271 | 0.02954508 | STOPWORD |
| 吾 | 1678 | 1417 | 0.03293893 | STOPWORD |
| 虽 | 1565 | 1451 | 0.03372928 | STOPWORD |
| 于 | 1503 | 1210 | 0.02812711 | STOPWORD |
| 使 | 1489 | 1390 | 0.03231130 | STOPWORD |
| 尚 | 1446 | 1388 | 0.03226481 | STOPWORD |
| __ | 1371 | 157 | 0.00364955 | OTHER_RULE |
| 余 | 1290 | 1202 | 0.02794114 | STOPWORD |
| 便 | 1282 | 1217 | 0.02828983 | STOPWORD |
| 尔 | 1253 | 1014 | 0.02357098 | STOPWORD |
| 及 | 1247 | 1169 | 0.02717404 | STOPWORD |
| 者 | 1188 | 1021 | 0.02373370 | STOPWORD |
| 惟 | 1176 | 1081 | 0.02512843 | STOPWORD |
| 即 | 1161 | 1104 | 0.02566308 | STOPWORD |
| 但 | 1129 | 1074 | 0.02496571 | STOPWORD |
| 仍 | 924 | 905 | 0.02103722 | STOPWORD |
| 每 | 860 | 829 | 0.01927055 | STOPWORD |
| 乃 | 786 | 682 | 0.01585346 | STOPWORD |
| 诸 | 697 | 670 | 0.01557451 | STOPWORD |
| 而 | 657 | 460 | 0.01069295 | STOPWORD |
| 各 | 636 | 608 | 0.01413329 | STOPWORD |
| 也 | 584 | 528 | 0.01227365 | STOPWORD |
| 或 | 573 | 401 | 0.00932146 | STOPWORD |
| 哉 | 541 | 501 | 0.01164602 | STOPWORD |
| 然 | 535 | 511 | 0.01187847 | STOPWORD |
| ） | 522 | 365 | 0.00848462 | OTHER_RULE |
| （ | 520 | 364 | 0.00846138 | OTHER_RULE |
| 斯 | 500 | 457 | 0.01062321 | STOPWORD |
| ？ | 473 | 402 | 0.00934471 | OTHER_RULE |
| 彼 | 466 | 398 | 0.00925173 | STOPWORD |
| 《 | 453 | 294 | 0.00683419 | OTHER_RULE |
| 》 | 450 | 294 | 0.00683419 | OTHER_RULE |

完整表见 `reports/removed_tokens.csv`；独立前 100 表见 `reports/top_100_removed_tokens.csv`。

## DF 最高的前 100 个 token

| token | TF | DF | DF_ratio | protected | stopword | 处理结果 |
| --- | --- | --- | --- | --- | --- | --- |
| 。 | 228994 | 43014 | 0.99988377 | false | false | OTHER_RULE |
| ， | 231936 | 42973 | 0.99893071 | false | false | OTHER_RULE |
| 不 | 25664 | 17549 | 0.40793603 | false | true | STOPWORD |
| 无 | 15082 | 11955 | 0.27790046 | false | true | STOPWORD |
| 一 | 14321 | 11412 | 0.26527813 | false | true | STOPWORD |
| 有 | 12504 | 9885 | 0.22978219 | false | true | STOPWORD |
| 云 | 10581 | 9322 | 0.21669495 | false | false | RETAINED |
| 何 | 10203 | 8659 | 0.20128315 | false | true | STOPWORD |
| 来 | 9729 | 8515 | 0.19793580 | false | true | STOPWORD |
| 人 | 9679 | 8354 | 0.19419326 | false | false | RETAINED |
| 上 | 9222 | 8045 | 0.18701039 | false | true | STOPWORD |
| 中 | 9007 | 7792 | 0.18112927 | false | true | STOPWORD |
| 自 | 8709 | 7492 | 0.17415561 | false | true | STOPWORD |
| 知 | 8369 | 7432 | 0.17276087 | false | false | RETAINED |
| 时 | 8222 | 7327 | 0.17032009 | false | true | STOPWORD |
| 相 | 8742 | 7219 | 0.16780957 | false | true | STOPWORD |
| 为 | 8605 | 6800 | 0.15806969 | false | true | STOPWORD |
| 见 | 7758 | 6795 | 0.15795346 | false | true | STOPWORD |
| 归 | 7180 | 6660 | 0.15481531 | false | false | RETAINED |
| 在 | 7368 | 6588 | 0.15314164 | false | true | STOPWORD |
| 此 | 7449 | 6495 | 0.15097980 | false | true | STOPWORD |
| 得 | 7108 | 6136 | 0.14263465 | false | true | STOPWORD |
| 未 | 6855 | 6121 | 0.14228597 | false | true | STOPWORD |
| 多 | 6327 | 5712 | 0.13277854 | false | true | STOPWORD |
| 日 | 6245 | 5644 | 0.13119784 | false | false | RETAINED |
| 下 | 6189 | 5635 | 0.13098863 | false | true | STOPWORD |
| 去 | 6041 | 5537 | 0.12871057 | false | false | RETAINED |
| 是 | 6164 | 5496 | 0.12775750 | false | true | STOPWORD |
| 心 | 6170 | 5453 | 0.12675794 | false | false | RETAINED |
| 生 | 6113 | 5309 | 0.12341059 | false | false | RETAINED |
| 君 | 6866 | 5303 | 0.12327111 | false | false | RETAINED |
| 行 | 5863 | 5160 | 0.11994700 | false | false | RETAINED |
| 尽 | 5342 | 4926 | 0.11450754 | false | false | RETAINED |
| 如 | 6010 | 4865 | 0.11308957 | false | true | STOPWORD |
| 出 | 5317 | 4863 | 0.11304307 | false | true | STOPWORD |
| 入 | 5277 | 4861 | 0.11299658 | false | true | STOPWORD |
| 千 | 5158 | 4748 | 0.11036984 | false | false | RETAINED |
| 处 | 5060 | 4623 | 0.10746414 | false | false | RETAINED |
| 已 | 5025 | 4520 | 0.10506985 | false | true | STOPWORD |
| 欲 | 4922 | 4512 | 0.10488389 | false | true | STOPWORD |
| 风 | 4809 | 4500 | 0.10460494 | false | false | RETAINED |
| 与 | 5222 | 4487 | 0.10430275 | false | true | STOPWORD |
| 空 | 4793 | 4472 | 0.10395407 | false | false | RETAINED |
| 谁 | 4895 | 4466 | 0.10381459 | false | true | STOPWORD |
| 还 | 4757 | 4452 | 0.10348916 | false | true | STOPWORD |
| 闻 | 4658 | 4274 | 0.09935145 | false | true | STOPWORD |
| 独 | 4477 | 4211 | 0.09788698 | false | false | RETAINED |
| 长 | 4562 | 4179 | 0.09714312 | false | false | RETAINED |
| 应 | 4375 | 4150 | 0.09646900 | false | true | STOPWORD |
| 更 | 4429 | 4144 | 0.09632953 | false | true | STOPWORD |
| 向 | 4404 | 4131 | 0.09602734 | false | true | STOPWORD |
| 山 | 4418 | 4030 | 0.09367954 | false | false | RETAINED |
| 思 | 4353 | 4026 | 0.09358655 | false | false | RETAINED |
| 春 | 4386 | 4002 | 0.09302866 | false | false | RETAINED |
| 满 | 4242 | 3940 | 0.09158744 | false | false | RETAINED |
| 同 | 4246 | 3927 | 0.09128525 | false | true | STOPWORD |
| 成 | 4290 | 3913 | 0.09095981 | false | true | STOPWORD |
| 看 | 4153 | 3908 | 0.09084358 | false | false | RETAINED |
| 天 | 4349 | 3898 | 0.09061113 | false | false | RETAINED |
| 我 | 5399 | 3886 | 0.09033218 | false | true | STOPWORD |
| 可 | 4524 | 3864 | 0.08982078 | false | true | STOPWORD |
| 月 | 4101 | 3820 | 0.08879797 | false | false | RETAINED |
| 水 | 4137 | 3814 | 0.08865850 | false | false | RETAINED |
| 开 | 4006 | 3791 | 0.08812385 | false | false | RETAINED |
| 花 | 4313 | 3754 | 0.08726377 | false | false | RETAINED |
| 三 | 4099 | 3700 | 0.08600851 | false | false | RETAINED |
| 远 | 3899 | 3695 | 0.08589228 | false | false | RETAINED |
| 望 | 3883 | 3659 | 0.08505544 | false | false | RETAINED |
| 年 | 3922 | 3643 | 0.08468351 | false | false | RETAINED |
| 事 | 3974 | 3637 | 0.08454404 | false | false | RETAINED |
| 从 | 3925 | 3626 | 0.08428834 | false | true | STOPWORD |
| 前 | 3853 | 3600 | 0.08368395 | false | false | RETAINED |
| 回 | 3713 | 3504 | 0.08145238 | false | false | RETAINED |
| 身 | 3839 | 3485 | 0.08101072 | false | false | RETAINED |
| 将 | 3736 | 3450 | 0.08019712 | false | true | STOPWORD |
| 莫 | 3732 | 3392 | 0.07884888 | false | true | STOPWORD |
| 深 | 3593 | 3366 | 0.07824450 | false | false | RETAINED |
| 到 | 3539 | 3327 | 0.07733792 | false | false | RETAINED |
| 似 | 3663 | 3319 | 0.07715196 | false | true | STOPWORD |
| 客 | 3479 | 3302 | 0.07675678 | false | false | RETAINED |
| 飞 | 3549 | 3280 | 0.07624538 | false | false | RETAINED |
| 能 | 3672 | 3276 | 0.07615240 | false | true | STOPWORD |
| 夜 | 3407 | 3232 | 0.07512959 | false | false | RETAINED |
| 复 | 3529 | 3197 | 0.07431600 | false | true | STOPWORD |
| 道 | 3527 | 3196 | 0.07429275 | false | false | RETAINED |
| 愁 | 3374 | 3154 | 0.07331644 | false | false | RETAINED |
| 难 | 3425 | 3153 | 0.07329320 | false | false | RETAINED |
| 地 | 3312 | 3123 | 0.07259583 | false | false | RETAINED |
| 作 | 3463 | 3096 | 0.07196820 | false | true | STOPWORD |
| 过 | 3254 | 3092 | 0.07187522 | false | false | RETAINED |
| 落 | 3234 | 3067 | 0.07129408 | false | false | RETAINED |
| 闲 | 3259 | 3061 | 0.07115461 | false | false | RETAINED |
| 别 | 3210 | 3010 | 0.06996908 | false | false | RETAINED |
| 亦 | 3354 | 2955 | 0.06869058 | false | true | STOPWORD |
| 起 | 3058 | 2922 | 0.06792348 | false | true | STOPWORD |
| 酒 | 3193 | 2920 | 0.06787698 | false | false | RETAINED |
| 高 | 3106 | 2917 | 0.06780725 | false | false | RETAINED |
| 重 | 3111 | 2908 | 0.06759804 | false | false | RETAINED |
| 随 | 3027 | 2880 | 0.06694716 | false | false | RETAINED |
| 犹 | 3001 | 2870 | 0.06671471 | false | true | STOPWORD |

独立表见 `reports/top_100_df_tokens.csv`。

## 被保护规则挽救的 token

| token | 词次 | DF | 避免的删除原因 |
| --- | --- | --- | --- |
| 乘槎 | 4 | 4 | LOW_TF |
| 关中 | 4 | 4 | LOW_TF |
| 庞德公 | 4 | 4 | LOW_TF |
| 河东 | 4 | 3 | LOW_TF |
| 烂柯 | 4 | 4 | LOW_TF |
| 颜渊 | 4 | 4 | LOW_TF |
| 马援 | 4 | 4 | LOW_TF |
| 渭滨 | 3 | 3 | LOW_TF |
| 千钟 | 2 | 2 | LOW_TF\|LOW_DF |
| 尧禹 | 2 | 2 | LOW_TF\|LOW_DF |
| 尧舜 | 2 | 2 | LOW_TF\|LOW_DF |
| 投笔 | 2 | 2 | LOW_TF\|LOW_DF |
| 敦煌 | 2 | 2 | LOW_TF\|LOW_DF |
| 函谷关 | 1 | 1 | LOW_TF\|LOW_DF |

完整审计见 `reports/protected_rescues.csv`。

## 疑似保护词错误切分

- 疑似实例数：4,985
- 涉及诗歌数：4,442
- 涉及保护词型数：102
- 程序只报告跨多个既有 token 的连续字面命中，没有自动合并。
- 详细实例见 `reports/protected_word_segmentation_issues.csv`。

## 清洗后 token 过少的诗

- 清洗后 0 token：16
- 非零但达到低 token 审核阈值：212
- retention_ratio 低于审核阈值：13
- 汇总进入人工审核表的诗：228
- 这些审核阈值只用于报告，不会触发删除。
- 详细记录见 `reports/low_retention_poems.csv`。
