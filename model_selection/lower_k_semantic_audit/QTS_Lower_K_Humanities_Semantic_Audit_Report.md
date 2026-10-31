# QTS Lower-K Humanities Semantic Audit Report

> 状态：`AI_REVIEW` / `PROVISIONAL_REVIEW` / `NEEDS_RESEARCHER_CONFIRMATION`。本报告不是已经完成的人工审核，也不自动决定最终 K。

# 1. Objective

本审核检验 C_K5、C_K10、C_K15 在自动指标具有竞争力时，是否仍保有可供唐诗研究使用的语义分辨率，并重点判断 K15 是否具备挑战冻结主模型 C_K20 的资格。全程读取冻结模型产物，不重新训练、不改写 Topic–word 或 Document–topic 数据。

# 2. Why Automatic Metrics Are Not Enough

Perplexity 衡量留出文本预测，NPMI 衡量局部词共现，stability 衡量不同 seed 的主题相似性，diversity 衡量 Top words 重复程度；这些指标都不能直接说明一个 Topic 是否把边塞、闺情、礼制等文学上可区分的语义场压在一起。K5 的较低 perplexity 与较高 stability 因而不能替代逐诗审读。

# 3. Review Method

每题同时读取带原概率的 Top30 words、Top20 representative poems、至少五作者的 author-balanced 核验样本（数据允许时），并参照 one-to-many K20 alignment。Cosine 只定位潜在关系；最终类型、欠分与合并判断均由词、诗和对齐三类证据共同形成。所有判断均为 AI 暂定值。

# 4. K5 Review

- **T01｜政治伦理、礼制颂德与人生玄思｜MIXED**：高权重词多为泛化的人、心、道、天、名；代表诗横跨政治伦理、礼制颂德、道教神仙与人生议论。 interpretability=2/5，overbreadth=5/5，under-split=TRUE。
- **T02｜花香春情、女性歌舞与闺情｜CLEAR**：花、春、香、开、舞等词与女性歌舞、闺情和惜春诗例互相支持，中心稳定。 interpretability=5/5，overbreadth=1/5，under-split=FALSE。
- **T03｜边塞征战、人生功名与诗酒感怀｜MIXED**：兵戈征战只是其中一支，代表诗同时出现仕途、老病、饮酒、赠答和人生感怀。 interpretability=3/5，overbreadth=5/5，under-split=TRUE。
- **T04｜江湖羁旅、送别与相思｜BROAD**：行、归、路、舟等行旅词与送别、思归和怀人诗例形成连续的离散—归返语义轴。 interpretability=5/5，overbreadth=2/5，under-split=FALSE。
- **T05｜山居寺院、隐逸与闲游｜BROAD**：山、门、寺、林、僧等词与寻幽、游寺、隐居诗例一致，内部各支共享山林幽居空间。 interpretability=4/5，overbreadth=3/5，under-split=FALSE。


K5 只有一个 CLEAR Topic；两个 MIXED 大桶分别压入政治—宗教—人生和军旅—身世—诗酒语义。它是“预测表现不等于文学解释力”的直接实例。

# 5. K10 Review

- **T01｜边塞征战与军旅｜CLEAR**：塞、军、兵、胡、将等词与征战、守边、军功诗例高度一致。 interpretability=5/5，overbreadth=1/5，under-split=FALSE。
- **T02｜仙山道观、山居修隐与寻幽｜BROAD**：仙、山、云、宫、洞等词把道教仙境、山寺与隐逸寻幽组织成共享空间意象。 interpretability=4/5，overbreadth=3/5，under-split=FALSE。
- **T03｜怀人悲音、清景与身世感慨｜MIXED**：夜月琴声与怀人是可见中心，却同时混入清景、仕途、悼亡和药病等诗例。 interpretability=3/5，overbreadth=4/5，under-split=TRUE。
- **T04｜离别哀悼、女性悲情与人生感怀｜MIXED**：悲、泪、别、死等词能概括负面情感，却把闺情、悼亡、行旅与人生议论压在同一情绪桶中。 interpretability=3/5，overbreadth=4/5，under-split=TRUE。
- **T05｜王朝礼制、颂德与政治教化｜BROAD**：朝、君、王、臣、礼等词与应制、颂德、政治劝诫诗例共同构成公共政治表达。 interpretability=4/5，overbreadth=3/5，under-split=FALSE。
- **T06｜江湖羁旅、送别与思归｜CLEAR**：舟、江、路、归、别等词与行旅、送别、归思诗例高度相合。 interpretability=5/5，overbreadth=1/5，under-split=FALSE。
- **T07｜老病身世、仕宦与诗酒自况｜CLEAR**：老、病、身、官、酒等词与自况、退居、诗酒及仕宦感怀诗例形成稳定主体。 interpretability=4/5，overbreadth=2/5，under-split=FALSE。
- **T08｜飞鸟夜声、宫怨与杂咏｜MIXED**：鸟、夜、声等感官词不足以统一宫怨、礼制、季节和道德杂咏。 interpretability=2/5，overbreadth=4/5，under-split=TRUE。
- **T09｜闲居山寺、羁旅思归与诗酒身世｜MIXED**：闲、山、归、酒、诗等词对应隐逸、寺院、行旅、思归和老病自况多个诗群。 interpretability=4/5，overbreadth=5/5，under-split=TRUE。
- **T10｜春花、艳情与自然物候｜BROAD**：春、花、香、风、月等词与惜春、女性情思及花鸟物候诗例相合，但覆盖面较大。 interpretability=4/5，overbreadth=3/5，under-split=TRUE。


K10 相比 K5 增加了边塞、羁旅、老病身世等可辨 Topic，但仍有四个 MIXED Topic；语义分辨率明显改善，尚未稳定消除欠分。

# 6. K15 Review

- **T01｜仙山宫苑、歌舞器物与杂咏｜MIXED**：仙、宫、云等词显示神仙宫苑倾向，但 Top20 同时有歌舞、器物、宫廷与一般杂咏。 interpretability=2/5，overbreadth=4/5，under-split=TRUE。
- **T02｜行路艰难、忠义生死与社会悲慨｜MIXED**：行、难、死、心等词把道路艰难、忠义生死、战争与社会悲慨聚于共同困厄语调。 interpretability=3/5，overbreadth=4/5，under-split=FALSE。
- **T03｜朝廷礼乐、应制与颂圣｜CLEAR**：朝、君、圣、礼、乐等词与郊庙、应制、朝会和颂圣诗例高度一致。 interpretability=5/5，overbreadth=1/5，under-split=FALSE。
- **T04｜边塞军旅、归途与春景杂糅｜MIXED**：军、塞、马等边塞词与归、春、花等行旅春景词并存，代表诗形成两类明显文本。 interpretability=3/5，overbreadth=5/5，under-split=TRUE。
- **T05｜君子伦理、政治劝诫与社会批评｜BROAD**：君、道、义、民、官等词与政治规劝、伦理议论和社会批评诗例构成一致的公共伦理中心。 interpretability=4/5，overbreadth=2/5，under-split=FALSE。
- **T06｜老病身世、诗酒交游与仕宦自况｜BROAD**：老、病、身、酒、诗等词与自况、仕宦、诗酒赠答诗例一致，但人生感怀与宴饮交游仍可区分。 interpretability=4/5，overbreadth=3/5，under-split=TRUE。
- **T07｜相思离愁、夜月与琴声｜CLEAR**：月、夜、愁、思、琴等词与怀人、离愁、夜听和女性情思诗例高度一致。 interpretability=5/5，overbreadth=1/5，under-split=FALSE。
- **T08｜山居寺院、隐逸与寻幽｜BROAD**：山、寺、僧、林、门等词与游寺、山居、隐逸诗例形成紧密相邻的空间和生活方式。 interpretability=5/5，overbreadth=2/5，under-split=FALSE。
- **T09｜诗名仕途、行旅与杂咏噪声｜NOISE**：人、日、知、名等泛化词占据前列，代表诗横跨仕途、行旅、怀人和杂咏，难形成稳定文学中心。 interpretability=1/5，overbreadth=4/5，under-split=FALSE。
- **T10｜江湖羁旅、送别与思归｜CLEAR**：江、舟、路、归、别等词与送别、远行、思归诗例高度相合。 interpretability=5/5，overbreadth=1/5，under-split=FALSE。
- **T11｜春花、艳情与惜春｜BROAD**：花、春、香、红、舞等词与女性情思、歌舞、惜春诗例一致，兼有一般春景。 interpretability=5/5，overbreadth=2/5，under-split=FALSE。
- **T12｜春水花鸟与自然风物｜CLEAR**：春、水、花、鸟、风等词与物候、花鸟、水景诗例高度一致。 interpretability=5/5，overbreadth=1/5，under-split=FALSE。
- **T13｜郊庙礼乐、神祀与祥瑞｜BROAD**：神、天、礼、乐、祀等词与祭祀、祥瑞、宗庙礼仪诗例形成礼制宗教中心。 interpretability=4/5，overbreadth=2/5，under-split=FALSE。
- **T14｜闲居清景、羁旅相思与仕隐杂糅｜MIXED**：闲、山、归、思、官等词与清景闲居、远行思归、相思和仕隐自况多类诗例并存。 interpretability=3/5，overbreadth=5/5，under-split=TRUE。
- **T15｜边塞征战、军功与神异杂咏｜BROAD**：军、将、塞、马等词与征战、军功诗例占主体，但夹有神异、礼仪及一般杂咏。 interpretability=4/5，overbreadth=2/5，under-split=FALSE。


K15 的多数 Topic 已有清楚中心，但 T01、T04、T14 是需要单独报告的严重问题，T09 仍为噪声。总体均值不能遮蔽这些局部合并。

# 7. K-Level Comparison

| K | CLEAR | BROAD | MIXED | NOISE | AI mean interpretability | AI mean overbreadth | automatic merge | semantic under-split |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 2 | 2 | 0 | 3.800 | 3.200 | 80.0% | 40.0% |
| 10 | 3 | 3 | 4 | 0 | 3.800 | 3.000 | 70.0% | 50.0% |
| 15 | 4 | 6 | 4 | 1 | 3.867 | 2.600 | 53.3% | 26.7% |


# 8. K15 vs K20

K15 的 AI mean interpretability 为 3.867，冻结 K20 的既有 human interpretability 为 3.750。两者来源与审核者不同，不能把 0.117 的差值当作显著优势。K15 有三个 `PROBLEMATIC_MERGE`（T01, T04, T14），九个 K20 CLEAR fields 仅三个 `PRESERVED_CLEARLY`，因此结论为 **COMPARABLE**，状态为 **K15_COMPETITIVE_WITH_CAVEATS**。

# 9. Under-splitting Diagnosis

Automatic merge ratio 只统计 one-to-many alignment：K5/K10/K15 分别为 80.0%、70.0%、53.3%。本次语义欠分率分别为 40.0%、50.0%、26.7%。两者不相等，因为山寺—隐逸等 one-to-many 关系可构成合理上位聚合，而一对一的混合 Topic 也未必是相对 K20 的欠分。

# 10. Core Semantic Coverage

九个 C_K20 CLEAR fields 在 K15 均有可检测承载点：3 个清楚保留、4 个保留但更宽、2 个与邻近语义合并。逐项结果见 `data/K15_core_semantic_coverage.csv`；这支持“核心未消失”，但不支持“九项均清楚独立恢复”。

# 11. Does K20 Add Meaningful Resolution?

按每个 K15 Topic 评估，K20 新增分辨率包括：MEANINGFUL_SPLIT=5，MINOR_REFINEMENT=2，FRAGMENTATION=1，UNCERTAIN=7。其中 T04 与 T14 的拆分最有意义；T13 对应的 K20 局部拆分可能把较完整的礼制神祀场碎片化。

# 12. Integrated Judgment

K5=`TOO_COARSE`；K10=`COARSE_BUT_INTERPRETABLE`；K15=`COMPETITIVE_RESOLUTION`。K15 在自动指标上竞争力强，在语义可解释性上与 K20 **COMPARABLE**，但存在三个问题合并和六个未完全独立保留的核心语义场。因此建议 **REVIEW_K15_TO_K20_FINE_GRID**；本轮没有运行 K16–19，等待研究者确认后再决定是否开展细网格搜索。

# 13. Researcher Review Queue

1. **K15 T04｜边塞军旅、归途与春景杂糅**：军旅与春景归途被压入同题，属于高风险不合理合并。
2. **K15 T14｜闲居清景、羁旅相思与仕隐杂糅**：同时承载隐逸、羁旅、相思等多个 K20 语义场，边界最宽。
3. **K15 T01｜仙山宫苑、歌舞器物与杂咏**：神仙山岳与宫苑器物杂咏黏连，需判断是否仅为统计共现。
4. **K15 T06｜老病身世、诗酒交游与仕宦自况**：老病仕宦与饮酒酬赠可能需要拆分，且 Top20 白居易占 65%。
5. **K15 T15｜边塞征战、军功与神异杂咏**：边塞核心明确但夹入神异杂咏，需核定可接受边界。
6. **K15 T13｜郊庙礼乐、神祀与祥瑞**：礼制神祀较完整，却对应两个 K20 主题，需核定 K20 是否碎片化。
7. **K15 T08｜山居寺院、隐逸与寻幽**：山寺与隐逸被合理聚合的判断，需要研究者检验文学层级。
8. **K15 T11｜春花、艳情与惜春**：女性春情与一般春景的合并是否损失关键分辨率，需人工确认。
9. **K15 T02｜行路艰难、忠义生死与社会悲慨**：行路、忠义生死与社会悲慨混合，但与 K20 一对一，需判断问题来源。
10. **K15 T05｜君子伦理、政治劝诫与社会批评**：政治伦理主题较宽，需核定劝诫、社会批评是否可由一标签承载。

研究者应在工作簿空白的 `Researcher_*` 列中记录最终判断；AI 建议列仅作定位和初步比较。
