# 《全唐诗》LOW_TF 阈值敏感性实验报告

## 1. Experimental Design

本实验由同一份人工金标准覆盖版分词语料出发，只把 `min_tf` 依次设为 1、2、3、5。词形规范、核心停用词、保护词、token 顺序、`min_df=3`、自动最大 DF、非汉字规则和语义映射状态全部固定。原有 `output/` TF5 主结果未被覆盖。

由于固定 `DF≥3`，任何 `TF=1` 或 `TF=2` 的非保护词都必然同时不满足 DF；因此 TF1、TF2、TF3 的最终语料相同。这不是程序错误，而是冻结 DF 阈值后的数学结果。报告分别给出规则主原因统计与相邻语料的实际新增损失。

## 2. Corpus Statistics

| Corpus | Vocabulary | Tokens | Valid poems | Empty | Mean | Median | P95 | P99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TF1 | 35,063 | 1,525,932 | 43015 | 15 | 35.462 | 27.000 | 85.000 | 185.000 |
| TF2 | 35,063 | 1,525,932 | 43015 | 15 | 35.462 | 27.000 | 85.000 | 185.000 |
| TF3 | 35,063 | 1,525,932 | 43015 | 15 | 35.462 | 27.000 | 85.000 | 185.000 |
| TF5 | 23,123 | 1,485,707 | 43014 | 16 | 34.527 | 26.000 | 83.000 | 179.000 |

完整字段见 `reports/tf_sensitivity/tf_threshold_comparison.csv`。

## 3. Vocabulary Loss

| Corpus | Vocab retention | Token retention | Poem retention | LOW_TF attributed types | LOW_TF attributed occurrences |
| --- | --- | --- | --- | --- | --- |
| TF1 | 100.000000% | 100.000000% | 100.000000% | 0 | 0 |
| TF2 | 100.000000% | 100.000000% | 100.000000% | 71,065 | 71,065 |
| TF3 | 100.000000% | 100.000000% | 100.000000% | 88,088 | 105,111 |
| TF5 | 65.947010% | 97.363906% | 99.997675% | 100,235 | 146,003 |

相邻阈值的实际新增损失：

| Transition | Newly lost vocabulary | Newly lost occurrences |
| --- | --- | --- |
| TF1_vs_TF2 | 0 | 0 |
| TF2_vs_TF3 | 0 | 0 |
| TF3_vs_TF5 | 11,940 | 40,225 |

`removed_by_TF*.csv` 是该档位以 LOW_TF 为主删除原因的累计归因表；`TF*_vs_TF*_removed.csv` 才表示相邻主语料之间真正新增消失的词。

## 4. Token Loss

Vocabulary loss 统计消失的不同词型；token occurrence loss 统计这些词在全库中的实际出现次数。两者不能互相替代。低频词可占 vocabulary 的很大比例，但只占 token occurrence 的较小比例。

## 5. Document Length

| Corpus | Min | P10 | P25 | P50 | Mean | P75 | P90 | P95 | P99 | Max | ≤5 | ≤10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TF1 | 0 | 14.000 | 19.000 | 27.000 | 35.462 | 36.000 | 61.000 | 85.000 | 185.000 | 2532 | 208 | 1342 |
| TF2 | 0 | 14.000 | 19.000 | 27.000 | 35.462 | 36.000 | 61.000 | 85.000 | 185.000 | 2532 | 208 | 1342 |
| TF3 | 0 | 14.000 | 19.000 | 27.000 | 35.462 | 36.000 | 61.000 | 85.000 | 185.000 | 2532 | 208 | 1342 |
| TF5 | 0 | 14.000 | 19.000 | 26.000 | 34.527 | 36.000 | 59.000 | 83.000 | 179.000 | 2465 | 228 | 1550 |

百分位数使用 `(n-1)×p` 位置的线性插值；所有空诗和极端长度诗均保留。详见 `extreme_length_poems.csv`。

## 6. Case Study Tokens

| token | TF | DF | protected | TF1 | TF2 | TF3 | TF5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 黄金台 | 2 | 2 | false | REMOVE | REMOVE | REMOVE | REMOVE |
| 玉衡 | 2 | 2 | false | REMOVE | REMOVE | REMOVE | REMOVE |
| 景祚 | 2 | 2 | false | REMOVE | REMOVE | REMOVE | REMOVE |
| 历象 | 2 | 2 | false | REMOVE | REMOVE | REMOVE | REMOVE |
| 桂圃 | 1 | 1 | false | REMOVE | REMOVE | REMOVE | REMOVE |
| 成蹊 | 1 | 1 | false | REMOVE | REMOVE | REMOVE | REMOVE |
| 乾道 | 3 | 3 | false | KEEP | KEEP | KEEP | REMOVE |
| 朝彩 | 3 | 3 | false | KEEP | KEEP | KEEP | REMOVE |
| 金阵 | 3 | 3 | false | KEEP | KEEP | KEEP | REMOVE |
| 商郊 | 2 | 2 | false | REMOVE | REMOVE | REMOVE | REMOVE |
| 田烛 | 1 | 1 | false | REMOVE | REMOVE | REMOVE | REMOVE |
| 抚戎 | 1 | 1 | false | REMOVE | REMOVE | REMOVE | REMOVE |
| 鸡戟 | 1 | 1 | false | REMOVE | REMOVE | REMOVE | REMOVE |
| 相识 | 2 | 2 | false | REMOVE | REMOVE | REMOVE | REMOVE |
| 长安 | 664 | 640 | true | KEEP | KEEP | KEEP | KEEP |
| 蓬莱 | 178 | 165 | true | KEEP | KEEP | KEEP | KEEP |
| 阳关 | 44 | 44 | true | KEEP | KEEP | KEEP | KEEP |
| 来仪 | 6 | 6 | true | KEEP | KEEP | KEEP | KEEP |
| 乘槎 | 4 | 4 | true | KEEP | KEEP | KEEP | KEEP |
| 烂柯 | 4 | 4 | true | KEEP | KEEP | KEEP | KEEP |
| 函谷关 | 1 | 1 | true | KEEP | KEEP | KEEP | KEEP |

`黄金台、玉衡、景祚、历象`是 TF=2、DF=2 的非保护词，因冻结 LOW_DF 而在 TF1 中已不存在；`乾道、朝彩、金阵`为 TF=3、DF=3，到 TF5 才实际消失。

## 7. Protected Words

- 保护词资源词面数：182
- 语料中现成 token 形式出现的保护词型：176
- 保护词出现次数：22,500
- 四组全部正确保留：true
- 未作为完整 token 出现的保护词面：一生、严子陵、折腰、无聊、良可惜、难再逢
- 上述未出现词面不是被 TF 规则删除，而是在冻结的输入分词中没有完整 token；本实验没有重新分词。
- 未重新分词，也未自动合并疑似切分的保护词。

## Figures

- 绘图后端：`deterministic_svg`
- 当前 Python 环境未安装 matplotlib，因此按配置使用确定性 SVG 回退；没有为作图安装或更改项目依赖。
- `reports/tf_sensitivity/figures/vocabulary_size.svg`
- `reports/tf_sensitivity/figures/total_tokens.svg`
- `reports/tf_sensitivity/figures/mean_document_length.svg`
- `reports/tf_sensitivity/figures/vocabulary_retention_rate.svg`

## 8. Recommendation

当前统计只能说明各阈值的词型、词次和文档长度变化，不能仅凭 vocabulary size 宣称某个阈值最佳。最终选择仍须等待下一阶段在完全相同参数和随机种子下比较 Topic Coherence、Topic Diversity、Topic Stability、主题关键词质量、代表诗歌和人工可解释性。

本阶段没有运行 LDA，也没有搜索最佳 K。
