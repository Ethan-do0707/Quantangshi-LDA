# 1 Objective

以匿名证据判断哪一种主题分辨率最适合文学研究，平衡语义一致性、可解释性、欠分与碎片化。本审核为 AI-assisted blind audit，所有判断均为 AI_PROVISIONAL，不能视为 human review completed。

# 2 Blinding Protocol

模型以 MODEL_A–D 匿名，Topic 使用全局 BT 编号并按固定种子打乱；盲态材料不含真实身份、自动量化指标、旧标签或旧审核评分。独立 Topic 审核完成后才按匿名模型汇总。

# 3 Review Rubric

八项评分均为 1–5；一致性、可命名性、文学可解释性与特异性越高越好，overbreadth 与 internal mixedness 越高越差。类型限定为 CLEAR、BROAD、MIXED、NOISE。相似度仅用于候选提示，文本证据负责最终裁决。

# 4 Independent Topic Review

已完成 61/61 个匿名 Topic 的独立审核。每张卡同时检查 Top30 words、Top20 representative poems 与作者平衡样本，且所有标签均为盲态重新生成。

# 5 MODEL_A

CLEAR=6, BROAD=4, MIXED=0, NOISE=2。模型级判断：BALANCED_RESOLUTION；盲审位次 1，信心 MEDIUM。主题规模较紧凑，CLEAR与BROAD合计10/12，平均文学可解释性最高、宽泛度与混杂度最低，且本轮未判定需再拆分；代价是两个明显残余主题仍存在。

# 6 MODEL_B

CLEAR=7, BROAD=6, MIXED=4, NOISE=3。模型级判断：OVER_FRAGMENTED；盲审位次 4，信心 MEDIUM。可识别题域数量最多，但MIXED与NOISE合计7/20，四个主题需再拆分，山寺幽居与林泉归隐又出现语义冗余，新增槽位未稳定转化为可用分辨率。

# 7 MODEL_C

CLEAR=5, BROAD=6, MIXED=3, NOISE=1。模型级判断：HIGH_RESOLUTION_BUT_USEFUL；盲审位次 2，信心 MEDIUM。多数主要题域边界可读，语义特异性最高，且仅一个NOISE；但行路社会悲愤、边塞春行与仙山宫苑三个MIXED提示局部欠拆。

# 8 MODEL_D

CLEAR=7, BROAD=2, MIXED=3, NOISE=2。模型级判断：MIXED_QUALITY；盲审位次 3，信心 MEDIUM。边塞、闺情、江旅、郊庙和宫廷核心主题清晰，CLEAR比例最高；但三个MIXED及两个相近的数字仙道残余主题削弱整体稳定性。

# 9 Under-splitting

共识别 10 个 POTENTIAL_UNDER_SPLIT Topic。模型级比率见下表；每一项均在 CSV 中保留关键词、代表诗与子群文本证据。

| blind_model_id | n_topics | CLEAR_ratio | BROAD_ratio | MIXED_ratio | NOISE_ratio | mean_literary_interpretability | mean_overbreadth | under_split_ratio | fragmentation_topic_ratio | AI_model_level_judgment | AI_blind_rank | AI_rank_confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MODEL_A | 12 | 0.500 | 0.333 | 0.000 | 0.167 | 4.417 | 2.333 | 0.000 | 0.000 | BALANCED_RESOLUTION | 1 | MEDIUM |
| MODEL_C | 15 | 0.333 | 0.400 | 0.200 | 0.067 | 4.333 | 2.467 | 0.200 | 0.000 | HIGH_RESOLUTION_BUT_USEFUL | 2 | MEDIUM |
| MODEL_D | 14 | 0.500 | 0.143 | 0.214 | 0.143 | 3.857 | 2.643 | 0.214 | 0.143 | MIXED_QUALITY | 3 | MEDIUM |
| MODEL_B | 20 | 0.350 | 0.300 | 0.200 | 0.150 | 3.950 | 2.750 | 0.200 | 0.100 | OVER_FRAGMENTED | 4 | MEDIUM |

# 10 Fragmentation

文本复核确认 2 组 POTENTIAL_FRAGMENTATION。候选相似度没有被当作自动裁决。跨模型结构事件如下：

| event_id | source_blind_model_id | target_blind_model_id | relation_type | AI_semantic_judgment |
|---|---|---|---|---|
| SM01 | MODEL_A | MODEL_D | SPLIT | FRAGMENTATION |
| SM02 | MODEL_D | MODEL_C | SPLIT | MEANINGFUL_SPLIT |
| SM03 | MODEL_D | MODEL_C | SPLIT | MEANINGFUL_SPLIT |
| SM04 | MODEL_D | MODEL_C | MERGE | MEANINGFUL_AGGREGATION |
| SM05 | MODEL_C | MODEL_B | SPLIT | MEANINGFUL_SPLIT |
| SM06 | MODEL_C | MODEL_B | SPLIT | FRAGMENTATION |
| SM07 | MODEL_C | MODEL_B | SPLIT | MINOR_REFINEMENT |
| SM08 | MODEL_C | MODEL_B | SPLIT | FRAGMENTATION |
| SM09 | MODEL_C | MODEL_B | MERGE | PROBLEMATIC_MERGE |

# 11 Semantic Family Coverage

由四模型共同结果归纳九个中性 family；覆盖状态不继承任何历史 taxonomy。

| semantic_family_id | MODEL_A | MODEL_B | MODEL_C | MODEL_D |
|---|---|---|---|---|
| Family_01 | CLEARLY_RESOLVED | CLEARLY_RESOLVED | CLEARLY_RESOLVED | CLEARLY_RESOLVED |
| Family_02 | CLEARLY_RESOLVED | CLEARLY_RESOLVED | CLEARLY_RESOLVED | CLEARLY_RESOLVED |
| Family_03 | RESOLVED_BUT_BROAD | CLEARLY_RESOLVED | CLEARLY_RESOLVED | ABSENT |
| Family_04 | CLEARLY_RESOLVED | CLEARLY_RESOLVED | CLEARLY_RESOLVED | CLEARLY_RESOLVED |
| Family_05 | CLEARLY_RESOLVED | FRAGMENTED | CLEARLY_RESOLVED | CLEARLY_RESOLVED |
| Family_06 | RESOLVED_BUT_BROAD | CLEARLY_RESOLVED | RESOLVED_BUT_BROAD | RESOLVED_BUT_BROAD |
| Family_07 | CLEARLY_RESOLVED | CLEARLY_RESOLVED | CLEARLY_RESOLVED | CLEARLY_RESOLVED |
| Family_08 | ABSENT | CLEARLY_RESOLVED | CLEARLY_RESOLVED | WEAKLY_PRESENT |
| Family_09 | WEAKLY_PRESENT | WEAKLY_PRESENT | WEAKLY_PRESENT | FRAGMENTED |

# 12 Blind Model Ranking

盲审排序为 MODEL_A > MODEL_C > MODEL_D > MODEL_B，无并列；信心均为 MEDIUM，因为研究者尚未复核关键边界。排序依据是比例、质量、欠分和碎片化的并列证据，未使用主题绝对数或自动量化指标。

# 13 Blind Recommendation

MODEL_A 呈现最紧凑的质量平衡；MODEL_C 有较强核心但局部欠分；MODEL_D 的清晰核心伴随残余重复；MODEL_B 的额外分辨率主要落在混合与残余槽位。最终决策仍须研究者亲查 MIXED、split、merge 与模型差异最大的 Topic。
