# 当前状态

- 2026-09-07：已完成独立的 QTS LDA K-value held-out perplexity diagnostic：A/B/C/D 各按 `random_state=20260904` 固定 80/20 拆分，同语料所有 K 共用相同行与顺序；使用 K=20/30/40/50/60、seed=42/2026/20260830、原 Batch LDA 参数重新训练 60 个诊断模型。60项均运行30轮，train/test perplexity 与 score 全部有限，冻结输入及既有模型369个文件哈希未变。
- 四套最低 mean test perplexity 均为 K=20：A/B/C/D 分别为 115,884.403 / 24,770.515 / 15,514.047 / 9,924.195。C 的 K20/30/40/50/60 为 15,514.047±168.002、22,354.411±251.233、31,009.832±319.362、41,046.736±80.534、55,783.629±2,283.522；相邻“改善率”均为负，故不是改善递减型 elbow，而是 K 增加时困惑度恶化。C 的 NPMI、Stability、Perplexity 均偏好 K=20，诊断为 `SUPPORTS_K20`，不触发重新选型。
- 产物位于 `model_selection/perplexity_diagnostic/`：4份split、5份主CSV、5张PNG、中文报告、配置、逐模型检查点及输入/输出/环境审计。评分阶段把纯执行参数 `n_jobs` 临时设为1；与并行评分对照的 perplexity/score逐值相同。训练始终为原 `loky + n_jobs=-1`；线程训练对照存在约0.31%差异，已拒绝混用并保留审计记录。

- 2026-09-04：新增 `generate_pyldavis_qts.py`，读取冻结 C/B/D 模型与各自原始 BoW/词表/文档概率，生成 `outputs/pyldavis/pyLDAvis_{C_K20,B_K20,D_K20}.html`。三套模型/DTM/词表分别为 20×35,063 / 43,015×35,063 / 35,063，20×51,982 / 43,016×51,982 / 51,982，20×23,123 / 43,014×23,123 / 23,123；shape、来源哈希、行列索引与全部60个Topic编号/词项概率/占比核验通过，原始输入未变。
- pyLDAvis 安装包3.4.1（内部版本字符串3.4.0），最终采用官方矩阵 `prepare` 直接复用保存概率，固定 `sort_topics=False`、`start_index=1`、`R=30`；sklearn便捷接口的首次尝试因重复transform多进程开销中止。三份HTML全部重新生成并内嵌JS/CSS；4项异常处理测试、HTML内嵌数据与JavaScript语法检查通过。浏览器工具禁止file://，点击/λ交互尚待本机人工验收；未作自动文学解释。

- 已从 SQL 无损导出 43,030 条诗歌到 `segmentation/data/poems.jsonl`。
- sikufenci、bert-ancient-chinese WordPiece、Jiayan WordNgramTokenizer、jieba 四套全量结果均已完成，各 43,030 条并通过原文重构校验。
- sikufenci 使用百度网盘官方分词权重，已核验 9 类分类头；权重 SHA-256 为 `50479a8de02ae47913c63de4a43d3ff104c57199e44fa70c52ee9d3c69ee975a`。
- 四路比较报告与 CSV/JSON 统计已生成于 `segmentation/comparison/`。
- 全语料相邻汉字边界的四路一致判断率为 42.16%；该指标只表示方法间一致性，不表示准确率。
- 四个基础结果文件各 43,030 行；原比较阶段曾记录原文重构通过。本次复核发现当前 `poems.jsonl` 与 `sikufenci.jsonl` 在诗歌 ID 1 有 1 个字差异（“连薨”/“连甍”），其余 43,029 首一致。
- 已基于 sikufenci 建立《全唐诗》专用停用词：核心表 105 词，扩展表 164 词，另有 105 个明确保护的主题词。
- 候选判断结合总词频、诗歌文档频率、文档熵、跨诗人熵和 20 主题探索性 LDA 主题熵；完整审计表位于 `stopwords/tang_poetry_stopword_audit.csv`。
- 核心表删除 383,191 个纯汉字词元（19.01%），扩展表删除 530,554 个（26.32%）；两者均未新增空文档。源语料本身有 11 条空正文。
- 停用词生成器通过 Python 编译检查和 4 项单元测试，词表通过去重、包含关系和主题保护互斥校验。
- 已新增《全唐诗》LDA 分词规范 v1.0：确定 sikufenci 加确定性人工覆盖的默认口径，并提供 97 条保护词种子、85 条人地名种子和 12 条禁合并规则。
- 参考典故表已完成结构与噪声核查；其词形仅作为候选来源，不作金标准，也不允许整列直接导入词典。
- 已把规范应用到 43,030 首 sikufenci 基础结果并另存为 `segmentation/results/sikufenci_lda.jsonl`；13,290 首词界发生变化，逐诗审计与元数据已生成。
- 新结果含 2,490,931 个 token，较基础结果净增加 9,881 个；新增 14,957 个边界、移除 5,076 个边界，所有行均可还原输入的 sikufenci 字符序列。
- 六个指定表达的所有语料实例均成为完整 token；新增回归测试后，分词测试共 11 项通过。
- 已读取并核验 `tang_poetry 人工金标准.xlsx`：主表 485 行均可唯一对齐到 sikufenci；其中 426 首全文标注、59 首为省略号前缀标注，另有 4 首采用不同字面异文。
- 已生成 `segmentation/results/sikufenci_manual_gold.jsonl`、元数据和 485 行逐条审计；42,545 首未标诗原样保留，485 首中 412 首词界变化。
- 人工覆盖版含 2,480,498 个 token，较基础 sikufenci 减少 552 个；新增 361 个边界、移除 913 个边界。4 个异文只投射词界，所有 43,030 行均保留基础 sikufenci 字符序列。
- 独立复核通过行数、ID、字符重构、未标诗不变、完整/前缀标注落实和输出 SHA-256；分词测试现共 17 项通过。
- 已从全量 `sikufenci_manual_gold.jsonl` 更新 Voyant 输入文本：`segmentation/manual_gold/李白.txt` 含《全唐诗》语料中的 896 首、80,883 个空格分隔词元，`杜甫.txt` 含 1,158 首、101,076 个词元；均为无 BOM UTF-8，以空行分诗且不写入篇名。
- 李白/杜甫文本分别有 11/24 首受到人工金标准覆盖；其中原人工表仅标前缀的 2/7 首，其未标后缀已由全量 sikufenci 补足，因此当前两个 Voyant 文本不再截断诗文。
- 已从 `sikufenci_manual_gold.jsonl` 导出 `segmentation/manual_gold/全唐诗.txt`：按原顺序保留 43,030 首和 2,480,498 个空格分隔词元，含 11 条空正文；文件为无 BOM UTF-8、CRLF 换行、空行分诗，不写入 ID、诗人或篇名，逐诗内容与源 JSONL 完全一致。
- 已基于 `sikufenci_manual_gold.jsonl` 生成五类预处理资源：4 条局部词形/异文记录、105/164 个核心/扩展停用词、182 个保护词面、两档 TF/DF 规则和 27 条可选语义映射。
- 人工叠加版共有 2,014,981 个纯汉字词次、123,490 个词型；默认档 `TF>=5 且 DF>=3` 基线预估保留 23,123 个词型和 73.73% 词次，严格档 `TF>=10 且 DF>=5` 保留 13,510 个词型和 63.35% 词次。
- TF/DF 规则不设自动最大 DF，单字不按长度删除，保护词优先于停用和低频过滤；词形规范与语义映射均不覆盖原词，改变词界/词面后必须重新统计。
- 新增资源构建器和 6 项测试；连同人工覆盖、LDA 规范和停用词既有测试共 23 项通过，输出哈希写入 `segmentation/manual_gold/resources/resource_metadata.json`。
- 已新增 `src/clean_corpus.py` 与集中配置 `config/cleaning_config.yaml`，并按人工金标准覆盖版、核心停用词和默认 `TF>=5、DF>=3` 档完成 43,030 首全量清洗；没有运行 LDA。
- 清洗输入 2,480,498 个 token（其中纯汉字 2,014,981 个、123,490 型），输出保留 1,485,707 个 token、23,123 型；与既有默认档基线预估精确一致。主输出保留全部 43,030 行，其中 43,014 行非空、16 行为零 token。
- 4 条词形/异文记录均明确禁止自动执行，本次实际规范替换为 0；27 条语义映射只生成 `output/semantic_mapping_expanded.csv` 旁表，未写回 LDA 主语料。
- 实际删除 383,023 个停用词、146,003 个低 TF 词、248 个低 DF 词和 465,517 个非汉字 token；规则未启用最大 DF，故高 DF 自动删除为 0。保护规则保留 22,500 个词次，并从低频过滤中挽救 14 型、42 次。
- 数据质量审计共 25 行：11 条空原文、11 条空分词、2 条完全重复记录和 poem_id=1 的 1 条原文/token 重构异文；均未自动删除。另有 5 首非空原诗在清洗后成为零 token。
- 保护词切分扫描报告 4,985 个跨既有 token 的字面实例，涉及 4,442 首和 102 个保护词型；只写入人工审核表，未自动合并。
- 主 CSV、逐行 token 文本和逐诗统计均为 43,030 行且顺序一致；固定种子抽样为 200 首；17 个登记输出哈希复核无误。新增 5 项清洗测试后，清洗、分词资源和停用词相关测试共 32 项通过。
- 已新增 `src/run_tf_sensitivity.py` 与 `config/tf_sensitivity_config.yaml`，只改变 `min_tf=1/2/3/5`，其余规范、核心停用词、保护词、`min_df=3`、最大 DF、非汉字规则、语义映射状态和 token 顺序全部冻结；没有运行 LDA。
- TF1、TF2、TF3 因固定 `DF>=3` 而产生完全相同的最终语料：35,063 词型、1,525,932 词次、43,015 首非空和 15 首空诗。任何 TF=1/2 的非保护词必然也不满足 DF，因此提高到 TF2/TF3 没有新增实际损失。
- TF5 独立对照为 23,123 词型、1,485,707 词次、43,014 首非空和 16 首空诗，与既有 TF5 的 43,030 行内容和 `lda_tokens.txt` 哈希完全一致；既有 TF5 文件哈希在实验前后未变。
- TF3→TF5 实际新增删除 11,940 个词型、40,225 个词次，其中 TF=3 为 7,535 型/22,605 次，TF=4 为 4,405 型/17,620 次。相对 TF1，TF5 vocabulary retention 为 65.947010%，token retention 为 97.363906%。
- 182 个保护词面中有 176 个以完整 token 出现，共 22,500 次，四组全部保留；`一生、严子陵、折腰、无聊、良可惜、难再逢` 在冻结输入中未形成完整 token，本实验未重新分词或合并。
- 案例中 `黄金台、玉衡、景祚、历象` 均为 TF=2/DF=2，因固定 LOW_DF 在四组全部删除；`乾道、朝彩、金阵` 为 TF=3/DF=3，只在 TF5 消失；`长安、蓬莱、阳关、来仪、乘槎、烂柯、函谷关` 四组均受保护。
- TF1/2/3 的平均/中位/P95/P99 文档长度为 35.462/27/85/185，TF5 为 34.527/26/83/179；最长记录由 2,532 降至 2,465 token。≤5 token 诗为 208 对 228，占 0.4834% 对 0.5299%。
- 已生成四组独立语料、相邻差异表、100,249 行纯频率低频候选、案例追踪、1,984 行极端长度审核表和四张 SVG 图；matplotlib 在当前环境不可用，故按配置使用确定性 SVG 回退。37 项相关测试全部通过，27 个登记输出哈希复核无误。
- 已新增 `src/run_tf_df_lda.py` 与 `config/tf_df_lda_config.yaml`，按 A=(TF≥1,DF≥1)、B=(2,2)、C=(3,3)、D=(5,3) 生成四套 TF×DF 对照语料、词典、稀疏 BoW、有效诗索引和空诗清单；保护词继续豁免最低 TF/DF，未重新分词、未应用语义映射。
- A/B/C/D 分别保留 123,385/51,982/35,063/23,123 个词型、1,631,958/1,560,019/1,525,932/1,485,707 个词次和 43,019/43,016/43,015/43,014 首有效诗；四套逐行语料均保留全部 43,030 行，空诗只从 LDA 矩阵排除。
- 已完成 4×5×3=60 个 scikit-learn Batch LDA：K=20/30/40/50/60，随机种子 42/2026/20260830，`max_iter=30`、`n_jobs=-1`。全部模型均运行 30 轮，逐模型主题 top-20、每主题 top-5 代表诗和内容哈希检查点已保存。
- 已生成 60 行逐模型指标、60 行种子对稳定性、20 行 Corpus×K 汇总、48,000 行主题词和 12,000 行代表诗；指标为 document NPMI、Topic Diversity、匈牙利匹配 top-20 Jaccard 稳定性及训练 perplexity，不把 NPMI 称作 C_v。
- 描述性结果中，四套语料的最高平均 NPMI 均出现在 K=20；A/B/C/D 分别为 0.06855/0.09960/0.10136/0.10582。A、C、D 的最高稳定性在 K=20，B 在 K=30；这些单项最大值不作为最终选型。
- 60 个模型校正后的拟合时间合计为 15,351.869 秒。A/K=20 两个受系统休眠污染的墙钟计时已定向重跑，主题指标与原结果完全一致；旧指标保存在 `logs/sleep_affected_checkpoints/` 供审计。
- 独立验收确认 A→B→C→D 的逐诗 token 顺序严格嵌套，四个矩阵形状和 nnz 正确，D token 哈希与既有 TF5 完全一致，215 个登记输出哈希 0 失败；全项目测试 43 项全部通过。
- 已新增 `src/build_human_topic_review.py` 与 `config/human_topic_review_config.yaml`，只读取第一轮产物，为 A-K20、B-K20、B-K30、C-K20、D-K20、D-K30 生成 140 张人工 Topic 审核卡、140 行空白评分表及模型汇总模板；没有重训 LDA、修改语料、自动命名或自动评分。
- 六个主审核 seed 按“最接近同配置三 seed 平均 NPMI”选为 20260830、20260830、2026、2026、42、42；审核卡严格保留第一轮真实 top-20 词、top-5 代表诗和 soft prevalence，并补入既有 cleaned tokens。
- 第一轮未持久化完整模型、完整 topic-word 或 document-topic 矩阵，故 top-30、top-20 代表诗、随机高概率诗、dominant/阈值 prevalence、合规 Topic similarity、跨模型匹配和 K20→K30 分裂映射均不可恢复；审核包以明确缺失状态和表头空表交付，禁止把空表解释为零结果。
- 已精确导出 C 相对 D 独有的 11,940 个 TF=3/4 词型（7,535 个 TF=3、4,405 个 TF=4）；当前不能在缺少跨模型 Topic matching 时判断其语义增益或噪声归属。
- 审核包 66 个输入和 205 个输出哈希全部复核一致；六套卡片数量与 K 相符，现存 top-5 代表诗的 top-1/均值均不低于 0.5。新增 4 项测试后，全项目三个测试目录共 47 项通过。
- 已按第一轮原 BoW、原 scikit-learn 参数和六个 selected seed 精确重建 A-K20、B-K20、B-K30、C-K20、D-K20、D-K30；当前 Python/NumPy/SciPy/scikit-learn、平台和 12 逻辑核环境与第一轮记录逐项一致。
- 六个重建模型的 top-20 主题词 CSV 与第一轮对应文件 SHA-256 全部完全一致，top-5 代表诗 CSV 也全部完全一致；iterations、perplexity、NPMI extrema/mean 与 diversity 的最大绝对差均为 0。六个模型均通过 joblib 和三个 NPZ 的精确回读。
- `output/lda_candidate_rebuild/` 已保存每套 `model.joblib`、原始 Topic×Word components、归一化 Topic×Word、Document×Topic、vocabulary/document 对齐表、top-30、top-20 代表诗及逐模型复现审计；压缩后合计约 121 MiB。
- 已用完整矩阵重新生成 `human_topic_review/`：140 张卡均含 top-30、top-20 代表诗、20 首不重复随机高概率诗、完整 prevalence 和最相似五 Topic；六模型 similarity≥0.50 pair 数依次为 28/34/42/28/26/18。
- C-K20 vs D-K20 已完成 20 对一对一匈牙利匹配，完整分布 cosine 最低/均值/最高为 0.3855/0.6803/0.9299；C-only TF=3/4 词仍为 11,940 个。B、D 的 K20→K30 各生成 60 行 Top-3 映射，按 similarity≥0.40 的双重匹配提示规则各有 19 个 possible split Topic，该提示不等同人工结论。
- 完整代表诗 top-20 中没有 Topic 的 top-1 或均值概率低于 0.5；最低均值为 A-K20 Topic 3 的 0.6227。随机样本均排除 top-20、满足 P≥1/K 且每 Topic 恰有 20 首。
- 重建层 43 输入/79 输出、审核包层 92 输入/216 输出哈希均零失败；新增 7 项重建与完整审核测试后，全项目三个测试目录共 54 项通过。
- 已依 D-K20、D-K30、C-K20 三份既有评分表的同一 1–5 锚点，完成人工核读 A-K20、B-K20、B-K30 共 70 个 Topic；主评分表中 C/D 三组 70 行继续留空。三组平均总分为 2.5375、3.7500、3.091667，高质量 Topic（均分≥4）为 5、11、12，低质量 Topic（均分<2.5）为 12、5、14。
- 已生成三份含逐题标签、四项评分、问题标记、评分依据和公式汇总页的 XLSX，并用 `--summary-only` 更新模型汇总；新增安全写回脚本及 2 项测试。本轮 16 项不依赖 SciPy 的测试通过，完整测试套件因当前可用 Python 环境缺少 SciPy 未能重跑。
- 已以 `human_topic_review/human_topic_evaluation.xlsx`（2026-08-31 17:34:04）作为六模型人工评分的唯一最新来源，并与 `configuration_summary.csv`、`model_metrics.csv`、`stability_metrics.csv` 合并完成六候选综合评价；未重训、未修改 Topic 标签或人工分数。
- 六模型的广义 Pareto 前沿为 A_K20、B_K20、C_K20、D_K20；B_K30 被 C_K20、D_K20、D_K30 支配，D_K30 被 D_K20 支配。A_K20 仅因 Topic Diversity 单项最高留在前沿，不具备主模型资格。
- 三种权重情境的稳定前 3 为 B_K20、C_K20、D_K20；C_K20 的排名为 1/1/2，D_K20 为 2/3/1，B_K20 为 3/2/3。最终建议 C_K20 为主模型、D_K20 为稳健性/严格过滤副模型，B_K20 保留为宽词表对照。
- 已生成综合 CSV、Pareto/排名/情境/K/TF×DF 对照表、两份 XLSX、四张 PNG 和最终模型选择报告，统一位于 `outputs/01a05061-9ee3-7610-a401-abfec81de1b2/model_selection/`；工作簿公式重算、错误扫描和全部工作表逐页渲染复核均通过。
- 已将 B_K20、C_K20、D_K20 复制冻结到 `outputs/01a05061-9ee3-7610-a401-abfec81de1b2/literary_history_dataset/freeze/`；三个模型共 33 项完整性检查全部通过，model.joblib SHA-256 分别为 `d0907b37d5da7cb36e3487851627dbd9f159205050dd09ff164c3e2a8f54463a`、`4b23ee5661afa3127fd52f4938ca166446a9c0903cf8706c0985c964446d6d6f`、`7100c135964812ad224cfffbb20c869094be5961be3236cc315619a4b5ff0e95`。
- C_K20 正式 taxonomy 已冻结为 9 CLEAR、4 BROAD、4 MIXED、3 NOISE；13 个 CLEAR/BROAD Topic 可作 primary/secondary interpretation，4 个 MIXED limited，3 个 NOISE exclude。
- B–C–D 对齐以 C_K20 为锚点，保存 800 行共享词表证据和 20 份审核卡；40 个两侧对齐含 8 strong_1to1、3 moderate_1to1、20 split、3 merge、4 weak、2 unmatched，语义稳定性为 4 ROBUST、11 PARTIALLY_ROBUST、5 PARAMETER_SENSITIVE。
- C_K20 Document×Topic 为 43,015×20，43,015 行均以冻结 document_index 的 poem_id 唯一匹配；0 个 matrix row 未匹配，另有 15 首全语料诗因 C 清洗后为空未进入矩阵并已单列。
- 文学史母数据集已生成 CSV/XLSX，shape 为 43,015×54；保留完整20维概率、Top3、entropy、normalized entropy 和 gap。15 项最终数据测试、33 项冻结检查与主 XLSX 结构/258,090 个公式核验均通过。
- metadata 的 author/title/text 覆盖率为 100%；volume、date/year/period、genre/form/poetic_form、place/region/location 均为 0%，当前只具备作者级描述与文本回查条件，不具备时间、文体或空间分析条件。
- 已完成 QTS 历史元数据第一期核心管线：从 43,015 × 54 母表保守抽取 2,528 个原始作者实体，固定官方 `cbdb_20260829.sqlite3` 快照并保留候选、人物级来源和字段级 provenance；未训练或修改 LDA。
- 2,498 个 NAMED 作者中 CBDB 匹配 1,114 位；HIGH/MEDIUM/LOW/UNRESOLVED 分别为 654/422/38/1,384，匹配作者覆盖 28,381 首（65.98%）。7 个 ANONYMOUS 作者记录覆盖 1,209 首，均未自动匹配。
- 生年、卒年、完整明示 floruit range、只有 index year 的作者分别为 224/318/310/84；明确区分 floruit、dated observed activity 与 index year，未从 index year 伪造生年或作品年代。
- QTS_PERIOD_V1 得到初唐/盛唐/中唐/晚唐作者 102/124/162/91 位，unknown 2,049 位；已知 author-period 覆盖 23,609 首（54.89%），但只表示作者元数据继承，不是 dated poems。cross-period/boundary 作者为 87/74 位。
- 已生成 43,015 × 68 enriched master CSV，全部 17 项 QC 通过，20 维 Topic probability 校验和保持 `29068089d459ff490e14675a8273bd16c82618b26d9f35895577a1a48e14abd9`；加入 6 项大型 XLSX 类型/容差测试后，全项目 69 项测试在含 SciPy 的系统 Python 中通过。
- 作者主表、人工审核、数据字典三份 XLSX 已通过逐页视觉和 ZIP/XML 结构核验。经用户单项授权，大型 43,015 × 68 enriched master XLSX 已用 XlsxWriter 3.2.9、`constant_memory=True` 从权威 CSV 流式生成；`Data!A1:BP43016`、ZIP/XML/relationships、全量 poem_id 顺序以及 103 行全 68 字段比对均 PASS，最大浮点绝对误差 `4.44e-16`。源 CSV/XLSX SHA-256 分别为 `93fa45a5d848d0708fd1f4fb44289c26d8e98b7eec5b56bc440dc0de95de801f` / `bdf8498427465a8a7623c00ff27abbb6bdd328487b4dae2cf22c1820482e3832`。
- 已完成作者时期元数据代表性与缺失偏差审计：KNOWN 为 479/2,528 位作者（18.95%）与 23,609/43,015 首诗（54.89%）；UNKNOWN 为 2,049 位作者与 19,406 首诗。
- 时期信息明显偏向高产作者：KNOWN/UNKNOWN 作者诗作中位数为 3/1、均值为 49.29/9.47，Cliff's delta=0.3054；作者产量分桶覆盖率由单作作者的 12.52% 上升至 200+ 作品作者的 59.26%。
- 20 Topic 的最大/平均绝对 SMD 为 0.1517/0.0412，阈值 ≥0.10/0.20/0.50 的 Topic 数为 3/0/0；dominant topic Cramér's V=0.1046，entropy Cliff's delta=0.0657。CLEAR 风险为 4 LOW、5 MODERATE、0 HIGH。
- 缺失诊断的作者分组 5 折 AUC=0.9217，说明 period availability 与可观测特征明显相关，但不据此声称 MAR/MNAR，也不用于时期填补或重加权。EXPANDED 在预登记偏差分数中最接近 full corpus；建议代码为 2，后续必须同步报告 STRICT/CORE/EXPANDED。
- 审计目录已生成 16 份核心 CSV、43,015 行样本标签、5 张 PNG、Markdown 报告和 14-sheet artifact-tool 工作簿。12 项数据 QC、6 项规则测试、4 项 ZIP/XML/工作簿/CSV 镜像/manifest 独立测试全部通过。
- 已新增 `src/run_author_period_trend_structure.py` 与 `config/trend_structure_config.yaml`，直接复用 Representativeness Audit 冻结的 STRICT/CORE/EXPANDED 标签，对 C_K20 运行 poem-weighted 与 author-balanced 六套作者时期趋势规格；未重训 LDA、未修改20维概率、taxonomy、作者时期或 CBDB 匹配。
- 三套样本分别为 STRICT 8,722首/180作者、CORE 10,841首/207作者、EXPANDED 23,609首/479作者；以作者为 cluster、固定 seed 20260901 完成2,000次 bootstrap，并保存四期均值、差值区间、peak/trough、完整排序、数值 shape、bootstrap-supported direction、poem/author 差异和作者 mass 集中度。
- CORE poem-weighted 的初/盛/中/晚 peak Topic 数为 6/5/3/6，author-balanced 为 4/4/5/7；20 Topic 中 CORE peak 一致10个、不一致10个。按预注册六规格规则为 HIGH 0、MODERATE 16、LOW 4；9个 CLEAR Topic 均为 MODERATE，全部进入后续人工审核候选，但不构成文学史结论。
- 正式产物位于 `outputs/01a05cec-c8db-7761-9760-9d2d42a4853e/author_period_trend_structure/`，含13份数据 CSV、六幅 PNG、14节报告和12-sheet 主工作簿。15项数据不可变性测试、工作簿公式错误扫描与逐页渲染均通过；全仓库 tests/stopwords/segmentation 共94项测试通过。
- 已完成 C_K20 作者时期效应量与统计验证流水线：正式推断仅覆盖9个 CLEAR Topics 的 CORE_POEM（作者 cluster CR1）与 CORE_AUTHOR（一行一作者、HC3），STRICT/EXPANDED 只作六规格效应方向、区间和敏感性；未新增 sensitivity p-value family。
- 以固定 seed 20260901 完成5,000次时期分层作者 cluster bootstrap。CORE_POEM/CORE_AUTHOR omnibus BH-FDR 分别通过5/9与2/9；四类 planned contrast 的 STRONG/SUPPORTED/TENTATIVE/UNSUPPORTED/CONFLICTING 计数依次为 E-H 0/3/4/0/2、H-M 0/1/2/2/4、M-L 1/1/3/0/4、E-L 3/4/1/0/1。
- 共有13个 Topic/contrast 达到 `STATISTICALLY_SUPPORTED_CANDIDATE`；T07、T11、T16、T19 至少一个 primary contrast 存在 CORE_POEM/CORE_AUTHOR 符号冲突。CLR 临时敏感性未发现真零、未使用 pseudocount，共7个 Topic×contrast×aggregation 方向与 raw probability 不同。
- 效应验证产物位于 `outputs/01a05cec-c8db-7761-9760-9d2d42a4853e/author_period_effect_validation/`，含12份数据 CSV、9张 PNG、18节报告和13-sheet artifact-tool 工作簿。15项不可变性检查、13页渲染、公式错误扫描与全仓83项测试均通过；权威 master SHA-256 保持 `93fa45a5d848d0708fd1f4fb44289c26d8e98b7eec5b56bc440dc0de95de801f`。
- C_K20 Author Influence/LOAO 主分析已完成：CORE 10,841首/207作者，9 CLEAR×4 contrasts×2 estimands 共7,452行 LOAO，576行双方案 Top1/3/5/10 exclusion；13组候选的156组关键排除完成2,000次作者 cluster bootstrap（seed=20260902）。另对21组 Topic/contrast 运行 STRICT/EXPANDED 简化复核，无新 p-value 搜索或 B/D 运算。
- 13组候选的 GLOBAL_TOP 主分类为 AUTHOR_ROBUST 1、CONCENTRATED_BUT_ROBUST 8、AUTHOR_SENSITIVE 4；保留9组：T03 E-H/M-L/E-L、T06 E-L、T07 E-L、T08 E-L、T11 M-L、T16 E-L、T19 E-H；降级 T10 E-L、T16 H-M、T19 E-L、T20 E-H。9组保留候选在 PER_PERIOD_TOP 下也通过同一稳健性诊断。
- 两方案分类差异已显式保存：T06 M-L、T08 M-L 在 GLOBAL_TOP 集中但稳健、PER_PERIOD_TOP 敏感；T19 E-L 与 T20 E-H/E-L/M-L 相反。T16 H-M 在 GLOBAL_TOP Top5 的 POEM 衰减74.53%，PER_PERIOD_TOP Top5 更出现反向，不可概括为所有删除方案稳健。
- 产物位于 `outputs/01a05cec-c8db-7761-9760-9d2d42a4853e/author_influence_loao/`。2026-09-03 D: 存取恢复后重跑，15项不可变性检查通过，master SHA-256 不变，bootstrap replicate SHA-256 仍为 `17d5ab794a4455bd384ba28dcbefafb5c34cacd53c2f9a1d23b590ccc96ef547`。
- 最终产物含16份主数据 CSV（含同内容 Top 作者别名表）、7份冲突贡献表、5张 PNG、15节报告与13-sheet 工作簿；13页及两处方案分列区域完成视觉核验，公式错误0，8项输入/34项登记输出哈希通过，最新 `python -m unittest discover -s tests -q` 共122项全部通过。工作簿 SHA-256 为 `70735b23c8b2047ed80c38d9ee64cf96748e4669addd47d2e355ad4ed254ad42`。
- 运行注意：工作簿生成命令完成导出与核验后仍返回非零退出码，原因未定位；未掩盖退出状态，以独立 ZIP/XML、公式缓存、文件哈希和视觉验收确认成品，不把命令退出码当作成功证据。D: 存取异常已恢复。本任务交付 A–N 后停止，未运行 B/D 或文学史解释。

## 2026-09-03 B–C–D 发现层稳健性

- 已按来源 `finding_level_bd_eligible=True` 自动筛选9项，仅复用冻结矩阵和既有alignment；4项LOAO降级候选不进入主检验。共同集合43,014首，B/C/D分别损失2/1/0首，三套时期样本均无损失；C原始/共同效应最大差4.72e-16，native敏感性差0。
- 来源split采用无权概率和，并另存single-best敏感性；B为3 DIRECT+6 BUNDLE，D为2 DIRECT+6 BUNDLE+1 MERGED_PROXY。来源未记录正式human_review_status，明确保留NOT_RECORDED_IN_SOURCE，不能视为人工批准。
- 固定seed=20260902、5,000次时期分层作者cluster bootstrap，覆盖9 findings×3模型×3样本×2聚合=162行主效应；另存72行single-best、108行native、54行原C基准、32行Top1/Top3作者检查和4行条件CLR。优先四项32行排除无反向；T07为唯一CLR触发finding。
- 最终ROBUST_ACROSS_BCD：T08 E-L、T03 E-L、T03 M-L、T06 E-L；PARTIALLY_ROBUST：T16 E-L、T03 E-H、T11 M-L、T07 E-L、T19 E-H；其余三类均0。T16 E-L与T03 E-H为D支持/B区间不确定；T07有样本、聚合与bundle依赖；T19受D合并代理限制。C原有统计与LOAO标签均保留。
- 产物位于 `outputs/01a0628d-ef13-7961-85ce-146e0a14f60a/BCD_finding_level_robustness/`：19份CSV、6张PNG、19节报告、12-sheet工作簿及完整配置/哈希清单。20项不变量通过；重复运行19份CSV与bootstrap哈希完全一致；仓库tests 131项、stopwords 4项、segmentation 23项共158项通过。
- 工作簿全部12页、6张图与全量类型化单元格镜像完成检查；公式错误0。独立artifact-tool builder放在C盘工作目录（D盘不支持所需junction），实际路径及源码哈希见manifest implementation；导出完成后仍可能返回非零码，原因未定位，成品以独立检查为准且未掩盖退出状态。
- 本阶段止于A–N方法结果，不自动开展文学史解释。可进入下一阶段的4项仍须人工确认来源映射与代表诗，不等于LITERARY_HISTORY_CONFIRMED。
- 最终回归再次通过131项主测试；48项输入、31项登记产物及3个实现文件哈希独立核验通过。最终工作簿SHA-256为 `13cb45f42df695913aaf8dea16d087da95892ac6ceb4b0e9f30965b5008c9661`；导出进程实际退出码仍为1，未作为成品成功依据。

## 2026-09-03 人文学者审读报告

- 已在 `outputs/01a0650e-1708-7441-8842-44cfc4cd2cf4/humanities_review/` 生成 `QTS_BCD_Humanities_Review_Report.md`（权威文本）及同名DOCX，严格采用19节结构，覆盖4项核心、5项部分稳健、55条术语、常见误读与下一阶段审读清单；含17,730个中文字，开头阅读说明937个中文字。
- A—H材料均找到；9项发现的162个CORE差值/区间单元与效应底表一致，18条映射与原alignment一致，原统计/作者/发现分类保留；129项冻结来源与矩阵文件前后哈希不变。本任务未重训、重统计、改metadata或mapping，未开展代表诗细读及外部文学史解释。
- 报告区分旧“优先四项”（含T16 E-L）和最终稳健四项（含T03 M-L），明确T03 M-L未做B/D额外Top1/Top3检查。人工映射审核状态仍缺失，全语料代表诗不自动具备时期或CORE资格；原对齐卡同文异署仅列待校勘入口。
- 已核实趋势阶段数据字典将 `trend_shape` 描述为依据bootstrap支持方向，实际程序采用数值方向；报告记录释义不一致，未修改冻结字典或计算。是否后续勘误待用户决定。
- DOCX正文与Markdown同文、12张表格几何和编号清单结构均通过；canonical renderer因未安装LibreOffice/soffice返回WinError 2，未完成分页视觉核验，不应宣称已通过视觉验收。详细核验在报告目录 `_build/final_verification.json`。

## 2026-09-03 论文主题趋势图

- 已在 `historical_analysis/publication_figures/` 完成10个图组、13个核心图位、27个文件版本，均含600 dpi PNG、矢量PDF、SVG与逐图底表；另含完整图注、manifest、来源路径和复现脚本。
- 仅转录冻结均值、CI、BCD效应和GLOBAL_TOP排除结果；669项生成核验、2,369项独立核验通过，最终报告54行162数值一致。27份PDF全部经Poppler渲染并逐图检查，81个图文件哈希与manifest吻合，79个冻结来源前后不变。
- 字体采用Noto Serif SC常规400字重；T03保留盛唐至中唐回升，T08保留B/D作者均衡路径差异。图6因来源无B/D时期均值CI而只画已有点估计；T16强调B两种CI跨零，不误称B绝对区间始终最宽。
- 图9最终注释另通过不遮点、边界与源值定向核验。所有数据、mapping、taxonomy、时期、效应验证、LOAO及robustness分类保持冻结；正文选图建议与限制已写入README及图注。本任务完成后停止。

## 2026-09-07 C 语料 Lower-K 边界与 held-out perplexity

- 已完成 C 语料 K=5/10/15、三个固定 seed 的 9 个全语料 Batch LDA，并以固定 80/20 文档切分完成 A/B/C/D 共 69 个 held-out 训练；仅 K 与 seed 改变，沿用原词表、BoW、NPMI、Topic Diversity、跨 seed 稳定性和 LDA 参数，未改写冻结模型及下游分析。
- C 的 held-out perplexity 随 K 单调增大：K=5/10/15/20 分别为 7660.72/10171.69/12741.63/15568.24；NPMI 在 K=15 达到 0.102736，高于 K=20 的 0.101364，稳定性和多样性则以 K=5 最高。原先只比较 K=20–60 的 `SUPPORTS_K20` 结论现被限定为该候选区间，不能作为低 K 边界已排除的证据。
- K=5/10/15 对 C_K20 的自动 alignment 合并比例为 80.0%/70.0%/53.3%，虽随 K 上升下降，K=15 仍超过一半；该结果只作人工复核定位，不能替代语义判断。低 K 人工评分共 30 行保持空白，既有 C_K20 人工可解释性均分为 3.750。
- 自动边界状态为 `LOWER_K_COMPETITIVE`，综合状态为 `REVIEW_K15_VS_K20`：K15 值得与冻结 C_K20 做定向人工比较，但现有证据不足以改写 C_K20 taxonomy、主模型或任何文学史结果。
- 产物位于 `model_selection/lower_k_boundary_perplexity/`，含完整 CSV、30 张审核卡、8 幅 PNG、三份 Markdown 报告、输入输出哈希与三页人工复核工作簿。17 项任务不变量、154 项主测试、4 项 stopwords 测试和 23 项 segmentation 测试均通过；工作簿公式、校验、OpenXML 结构与逐页渲染通过。

## 2026-09-07 Lower-K 人文学语义审核

- 已逐题审读 C_K5、C_K10、C_K15 共 30 个 Topic 的 Top30 words、Top20 representative poems、author-balanced 样本及 C_K20 one-to-many alignment；所有结论均标为 `AI_REVIEW` / `PROVISIONAL_REVIEW` / `NEEDS_RESEARCHER_CONFIRMATION`。
- K5 的 CLEAR/BROAD/MIXED/NOISE 为 1/2/2/0，K10 为 3/3/4/0，K15 为 4/6/4/1；AI mean interpretability 分别为 3.800/3.800/3.867，mean overbreadth 为 3.200/3.000/2.600。
- Automatic merge ratio 为 80.0%/70.0%/53.3%，语义欠分率为 40.0%/50.0%/26.7%；K15 T01/T04/T14 暂列 `PROBLEMATIC_MERGE`，T08/T11 暂列 `MEANINGFUL_AGGREGATION`。
- 九个 C_K20 CLEAR fields 在 K15 均有可检测承载点，但只有 3 个清楚保留、4 个保留但更宽、2 个并入邻近语义；K20 新增分辨率判断为 5 meaningful split、2 minor refinement、1 fragmentation、7 uncertain。
- 综合结论为 K15 相对 K20 `COMPARABLE`、`K15_COMPETITIVE_WITH_CAVEATS`，建议 `REVIEW_K15_TO_K20_FINE_GRID`；本轮未运行 K16–19。
- 产物位于 `model_selection/lower_k_semantic_audit/`，含 30 张审核卡、核心 CSV、4 张 PNG、13 节报告及 6-sheet artifact-tool 工作簿。15 项不变量全部通过；工作簿 270 个 `Researcher_*` 单元格全部为空并完成逐页渲染核验。

## 2026-09-07 C 语料 K=2–30 逐整数穷举诊断

- 已直接复用冻结 `perplexity_split_C.csv`，完成 C corpus 的 K=2–30 × 3 seeds 共 87 个 train-only Batch LDA；NPMI 与 perplexity 来自同一训练模型，NPMI reference 仅为 TRAIN corpus，旧 full-corpus NPMI 未覆盖。
- `TRAIN_SPLIT_NPMI` 全局峰为 K12（0.0999330781±0.0001521897）；严格局部峰为 K5/K9/K12/K14/K21/K23/K26/K29，top-1% 高位带为 K11–K15，98% 连续主平台为 K11–K19，曲线判定 `BROAD_PLATEAU`。
- Held-out perplexity 在 K2–30 严格递增，K2 全局最低（6023.885442±11.259860），无内部局部谷；K15 相对 K20 的 NPMI 高 0.0023489422、perplexity 低 2808.035144。预测拟合系统偏好极低 K，但不能据此自动选择 K2。
- 新候选集合为 K5/K9/K11/K12/K13/K14/K15/K20/K21/K23/K26/K29；K12 为稳定的 NPMI 全局峰，K14 是 K13/14/16/17/18/19 中唯一高于 K15 者。状态为 `FINE_GRID_REVEALS_NEW_CANDIDATE`，只建议下一步定向语义审核，不修改冻结 C_K20。
- 产物位于 `model_selection/exhaustive_k2_30_scan/`，含 6 份主 CSV、6 张 PNG、14 节报告、检查点与哈希审计。15/15 不变量、7 项专用测试和 161 项全仓测试通过；K20/K30 的 6 个重叠 run 与旧诊断 perplexity 逐值精确一致。

## 2026-09-08 C 语料量化候选二次验证

- 已完成 K11/K12/K13/K14/K15/K20 × 10 seeds × ORIGINAL/FIXED_K20 prior 的 120/120 个 train-only Batch LDA；冻结 C corpus、35,063 词表、34,412/8,603 split、既有 C_K20 状态及 NPMI/Topic Diversity 定义均未改变。
- ORIGINAL_PRIOR NPMI 最高为 K12（0.0989849527±0.0022424664），FIXED_PRIOR_K20 最高为 K15（0.0984186340±0.0024978917）。K11–15 只构成 prior-dependent partial plateau；K12/K14 无实质差距，K12/K15 在 0.001 与 1% 尺度实用等价但固定先验下顺序反转。
- PRIOR_SENSITIVE 为 K11/K12/K13/K15，SEED_SENSITIVE 为 K20，REDUNDANCY_RISK 为空；两套 prior 下六个 K 均为 Pareto nondominated。最终量化 shortlist 为 K12/K15/K20，只建议下一步进入 Humanities Semantic Review，不自动执行。
- 预注册条件触发 K12/K13/K15/K20 的 max_iter=60 复核；完成 21 次唯一拟合并登记 24 行（K20 两套 prior 有效参数相同，3 行审计性复用）。受影响候选原先验排序保持 K12>K15>K13>K20；固定先验三种子排序由 K13>K12>K15>K20 变为 K12>K13>K15>K20，说明迭代上限会影响细小排序，不覆盖 10-seed 主结果。
- 主实验 15/15 不变量、迭代敏感性 21/21 unique fits 与 24/24 登记行、全仓 170 项测试及六幅图视觉检查均通过。

## 2026-09-08 四模型盲审输入门禁

- 已对 K12/K14/K15/K20 的冻结输入做递归盘点。K15 与 K20 具备 `model.joblib`、topic-word、document-topic 与代表诗产物；K12、K14 均未找到模型文件、逐诗 document-topic 矩阵或代表诗表。
- K12/K14 在量化验证中各有 20 份 NPZ（两套 prior × 10 seeds），40 份文件一致只含 `topic_word_unit` 与 `top_indices`；检查点只保存总体 prevalence 和 `components_sha256`，不足以反推出逐诗 topic probability 或原始 LDA components。
- 附件明确要求四模型均读取冻结 document-topic、不得重训且代表诗必须报告 `topic_probability`。因此四模型盲审尚未启动，也未创建映射、卡片、评分、工作簿或报告；等待用户提供 K12/K14 完整冻结产物，或明确修改“不得重训”的任务约束。

## 2026-09-09 K12/K14 模型产物恢复与盲审数据补齐

- 已按冻结 C 语料、35,063 词表、34,412/8,603 固定 split、原 K/seed/prior 和 sklearn 参数重新拟合 K12/K14 × 10 seeds × 2 prior 共 40 个 train-only 模型；先验证 topic-word，验收后才生成 theta。40/40 均为 `EXACT_REPRODUCTION`，components SHA-256、float32 topic-word 与 Top20 均精确一致；无 permutation、numerically close 或 mismatch。
- 40 个 run 均保存 `model.joblib`、float64 train/test/full document-topic、行索引、模型元数据和每 Topic 的 raw Top20/50/100 与 author-balanced Top20 代表诗。43,015 行映射唯一完整，theta shape、finite、非负及逐行和为 1 全部通过。
- 代表 seed 按预注册规则选择：排除 K 内最高 NPMI seed，从最接近该 K NPMI 中位数的五个非最高 seed 中取稳定性中心度最高者；K12=101，K14=31415。另以两 seed 在 43,015 首全语料上用 `ORIGINAL_PRIOR`、`max_iter=30` 建成盲审模型，不重算 held-out perplexity。
- K12/K14/K15/K20 均为 FULL_CORPUS、ORIGINAL_PRIOR、max_iter=30、同 C corpus 与词表；K15/K20 只读复用既有 seed=2026 模型，未重训。18/18 QA、6 项专项测试和全仓 176 项测试全部通过，最终状态 `BLIND_AUDIT_READY`。
- 产物位于 `model_selection/artifact_recovery_K12_K14/`；原量化 NPZ、既有 K15 与 frozen K20 哈希不变。任务在输入准备完成处停止，尚未启动 Four-Model Blind Humanities Semantic Resolution Audit。

## 2026-09-09 四模型盲审式人文学语义分辨率审核

- 已只读四套 `BLIND_AUDIT_READY` full-corpus 模型，生成并独立审读 61/61 张全局随机 Topic 卡；每卡含 Top30 words、Top20 真实概率代表诗及最多10首作者平衡样本。盲态材料未泄漏真实 K、量化指标、旧 taxonomy 或旧 K15 评分。
- 匿名模型 CLEAR/BROAD/MIXED/NOISE：A=6/4/0/2，B=7/6/4/3，C=5/6/3/1，D=7/2/3/2；盲态排名 `A > C > D > B`，无并列，信心 `MEDIUM`。文学可解释性均值为 4.417/3.950/4.333/3.857；欠分率为 0/20.0%/20.0%/21.43%，碎片化 Topic 率为 0/10.0%/0/14.29%。
- 盲审快照在解盲前冻结，SHA256=`7c7c33d5e581e5e7ebf67961cd415e2166862c503203ea529139f4b056739a8b`；解盲前后哈希一致。映射为 A→K12、B→K20、C→K15、D→K14。
- 七个由低分辨率到高分辨率的 SPLIT 裁决为 3 `MEANINGFUL_SPLIT`、1 `MINOR_REFINEMENT`、3 `FRAGMENTATION`、0 `UNCERTAIN`。K15→K20 为一项有意义拆分、一项轻微细化和两项碎片化，额外分辨率主要表现为过度碎片化/有限增益。
- 量化—语义证据分列而不加权汇总，最终状态 `PREFER_K12`；所有判断均为 `AI_PROVISIONAL`。8-sheet artifact-tool 研究者工作簿的 732 个 `Researcher_*` 单元格全空，3组数据验证、8页渲染和泄漏扫描通过；18/18 专项 QA 通过。
- 产物位于 `model_selection/four_model_blind_semantic_audit/`；冻结模型、旧 taxonomy 与历史下游分析均未修改或重跑。除 18/18 专项 QA 外，全仓既有 176 项 unittest 回归亦全部通过。

## 2026-09-09 K12/K15/K20 研究者主导人文学裁决准备

- 已只读确认三套代表模型完全可比：K12 seed=101、K15 seed=2026、K20 seed=2026，均为 C 语料、同词表、FULL_CORPUS、ORIGINAL_PRIOR、max_iter=30；没有重新训练或调用 `.fit()`。
- 已把 47 个 Topic 全局随机映射为 HR001–HR047，生成 47 张盲审卡、1,410 行 Top30 词证据和 1,410 行诗歌证据。每题含原始 Top20 与 author-balanced Top10，后者至少 5 位作者且每位最多 2 首。
- 第一阶段所有 14 个 `Researcher_*` 字段保持空白；盲审卡、公开 CSV 与工作簿 3 个可见页均不显示真实 K、模型身份、seed/prior/量化指标或旧 AI 标签。映射、候选和旧 AI 参考均隔离在 `_private`；工作簿后续 8 页为 `veryHidden`。
- 后台证据候选为 K12→K15 20 个事件（8 split、12 merge）、K15→K20 16 个事件（8 split、8 merge）及 K20 15 个碎片化候选对；所有判断/理由/信心字段均为空，候选不是人工裁决。
- 已生成 `QTS_K12_K15_K20_Human_Review.xlsx`、验证/冻结/解盲汇总脚本和 10 项专项测试。空白工作簿按预期返回 `TOPIC_REVIEW_INCOMPLETE`，未生成冻结或解盲结果；synthetic 47/47 仅在内存验证门禁可达，不写入正式数据。
- 工作簿 11 页均完成渲染复核，公式错误为 0；可见页仅 `Instructions`、`Blind_Topic_Review`、`Blind_Poem_Evidence`。10/10 专项、186 项主测试、4 项 stopwords 与 23 项 segmentation 测试均通过，冻结输入哈希未变。工作簿 SHA256=`04ae77e31b25691741a219ef3de953597803f1ab56f7c573952702d68ab7927a`。
- 当前状态为 `READY_FOR_RESEARCHER_REVIEW`。尚未冻结 Topic 级人工结果、未解盲、未进行模型级/拆并/碎片化/语义家族裁决，也未作最终 K 选择。

## 2026-09-09 C_K12 / C_K15 / C_K20 pyLDAvis 比较诊断

- 已只读使用 C_K12 seed=101、C_K15 seed=2026 与冻结 C_K20 seed=2026 的 FULL_CORPUS 工件；三者共用冻结 C corpus、43,015×35,063 DTM、相同词表与文档顺序。没有调用 LDA `fit` 或 `transform`。
- 三页均以 `pyLDAvis.prepare(..., sort_topics=False, mds="pcoa")` 从保存 phi/theta 生成；47/47 Topic ID 保持，离线 HTML、PreparedData、comparison 首页、相关词/坐标/prevalence/模型内相似度/alignment/诊断数据、八幅图、17节报告及研究者 XLSX 已输出到 `visualization/pyldavis_C_K12_K15_K20/`。
- sklearn 默认 prior 制度相同，但有效 alpha/eta 随 K 为 1/K；K12 的 seed 与代表 seed 选择法不同，均已登记为比较性警告，不构成硬阻断。K20 明确取 frozen reference。
- 数值结果：最大 prevalence K12 T02=12.5217%、K15 T06=11.0014%、K20 T10=8.5692%；mean nearest cosine 0.6091→0.5932→0.5878；cos≥0.85 pair 均为0，因此纯相似度碎片化信号记为 NO，不改写已有语义判断。
- 25/25 静态/数值 QA 和两页 XLSX 渲染通过，源工件哈希未变。浏览器自动化因 file:// URL 策略不能打开新文件；保留为用户本机交互查看事项。状态 `PYLDAVIS_THREE_MODEL_READY`。

## 2026-09-10 K12/K15/K20 AI 暂定逐卡审读

- 已按 HR001→HR047 逐卡阅读 Top30 词、Top20 代表诗与作者平衡 Top10；47/47 完成 15 个 `AI_Provisional_*` 字段，并在解盲前冻结为 `AI_topic_review_frozen.csv`，SHA256=`d9f17b9928e3ac58190673de59e15e2abb99b1ddf10cd57bb679d96db75cb9c9`。真实 K/topic_id 在冻结前保持盲态，但同一对话早期曾暴露少量旧 AI 内容，故不宣称对旧 AI 的严格独立隔离。
- 解盲汇总：K12 CLEAR/BROAD/MIXED/NOISE=7/2/2/1，K15=8/5/1/1，K20=10/4/4/2；文学可解释性均值为 4.5833/4.4000/4.2000，过宽度均值为 2.3333/2.3333/2.6000，Needs Splitting=YES 为 3/12、2/15、6/20。
- K12→K15 的 8 个 split 为 meaningful/minor/fragmentation/uncertain=5/1/1/1；K15→K20 为 4/2/2/0。15 个 K20 碎片化候选为 meaningful separation 14、uncertain 1、true fragmentation 0；所有判断基于卡片文本，量化相似度未作自动裁决。
- 原工作簿与逐字节备份 SHA256 均为 `04ae77e31b25691741a219ef3de953597803f1ab56f7c573952702d68ab7927a`；原件未改。已生成 16-sheet `QTS_K12_K15_K20_Human_Review_AI_Filled.xlsx`，保留原 11 页并新增 5 个 AI 页，`Researcher_*` 及正式人工页内容/公式/批注/验证/可见性全部保持不变。
- 15/15 专项验收与全仓 201 项 unittest 全部通过，5 个新增页完成视觉复核，公式错误为 0。AI 整体状态为 `AI_MORE_REVIEW_REQUIRED`；未填写最终人工决定、未修改最终 K、未重训 LDA。

## 2026-09-10 C_K12 作者时期历史分析完整重跑

- 经用户明确授权，以已恢复的 C_K12 FULL_CORPUS_FIT seed=101 `document_topic_full.npz` 独立重算 43,015×12 历史分析；未重新拟合 LDA，document_index 全量对齐，theta finite/非负/行和及 STRICT/CORE/EXPANDED 冻结样本数全部通过。
- 复用 C_K20 的趋势、代表性、CR1/HC3、BH-FDR、完整 12 维 CLR、作者 cluster bootstrap、LOAO 与两种 Top contributor 排除实现及阈值。CLEAR/BROAD/NOISE 严格分流；8 条 STRONG/SUPPORTED 进入 LOAO，结果为 0 AUTHOR_ROBUST、7 CONCENTRATED_BUT_ROBUST、1 AUTHOR_SENSITIVE，最终保留 7 条 `KEEP_WITH_CAUTION`。
- K12 盲态阶段先以 37 项核心文件生成 `SHA256_MANIFEST.txt` 与 `K12_BLIND_FREEZE.json`；逐项验 hash 后才读取旧 K20。旧 9 条保留 finding 的事后比较采用既有 AI_PROVISIONAL semantic family 与 K12→K15→K20 topic-word 路径：5 条正式复现、1 条仅在更宽 BROAD 主题中方向复现、2 条削弱/未复现、1 条题义重定义而不可单一比较。
- 完整输出位于 `history_analysis/C_K12/`，含 3 份阶段报告、18 节最终报告、K12/K20 比较报告、Humanistic Interpretation Queue、8 幅规定 PNG、20/20 最终 QA 和 59 项交付哈希。全仓 207 项 unittest 通过；8 幅图已逐张视觉复核。
- 状态为 `C_K12_HISTORY_ANALYSIS_COMPLETE`。本轮未执行 B_K12/D_K12 或 B/C/D finding-level robustness，未修改最终 K 选择、研究者工作簿或作者时期 metadata；跨模型语义对齐仍待研究者确认。
