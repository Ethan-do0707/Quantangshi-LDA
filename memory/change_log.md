# 变更日志

- 2026-09-07：新增独立的 held-out perplexity 诊断流水线、固定拆分配置、断点检查点、汇总与合并脚本、11项相关测试及完整审计。完成 A/B/C/D × K=20/30/40/50/60 × 3 seeds 共60次 Batch LDA 重训，生成5份主CSV、5张论文级PNG和中文诊断报告；全部完整性检查通过，既有输入与冻结模型哈希不变。四套语料的最低平均测试集困惑度均在 K=20，C 的 Perplexity、NPMI、Stability 一致支持 K20，结论为 `SUPPORTS_K20`，未修改既有模型选择。

- 2026-09-04：新增可重复运行的 pyLDAvis 脚本与4项异常处理测试；生成 C/B/D 三份独立离线HTML、逐主题编号审计及运行报告。复用各自冻结模型、原始DTM/词表/保存概率，哈希、shape、行列对齐和60个Topic编号/关键词/占比核验通过，无重训、重分词或原始文件修改。改用核心矩阵接口消除 sklearn helper 重复推断；记录发行版/内部版本字符串差异。HTML数据与JS语法通过；浏览器安全策略阻止file://交互实测，明确待用户本机验收。

## 2026-08-25

- 新增 SQL 解析、四路分词、断点推理、重构校验和边界比较流水线。
- 新增 4 项单元测试、运行说明和版本/限制元数据。
- 完成统一语料以及 jieba、Jiayan、bert-ancient-chinese 三套全量结果。
- 识别并隔离 sikufenci 官方 Google Drive 的错误 MaskedLM 权重，阻止其被误用为分词模型。
- 核验百度网盘官方 sikufenci 权重含 9 类分类头，并完成 43,030 条原精度 CPU 分词。
- 完成四套结果的行数、ID、原文重构与代表性样例复核。
- 生成分词粒度、两两边界一致性、高频词和差异样例的 Markdown、CSV 与 JSON 报告。
- 明确记录 sikufenci 耗时仅覆盖最后一次续跑进程，并保存断点起始记录数，避免把它误读为多次运行总耗时。

## 2026-08-27

- 新增《全唐诗》sikufenci 专用停用词生成器、探索性主题熵脚本和 4 项单元测试。
- 生成 105 词核心表、164 词扩展表、105 个主题保护词及含多维统计和人工理由的审计 CSV。
- 验证核心表删除 19.01% 纯汉字词元、扩展表删除 26.32%，两种口径均未新增空文档。
- 识别源 SQL 中 11 条空正文，并在元数据与报告中单独记录，避免归因于停用词过滤。
- 新增 `segmentation/lda_standard/README.md`，规定 LDA 分词单位、优先级、典故词准入、人地名保留、禁合并结构、处理流程和验收标准。
- 新增 97 条保护词种子、85 条命名实体种子和 12 条禁合并规则；用户指定的 `来仪`、`折腰`、`倾城`、`难再逢`、`良可惜`、`千钟` 均列为最高优先级强制合并项。
- 核查参考表 7,136 行及异形词噪声，明确其只作候选来源，不作自动词典或人工金标准。
- 校验三份 TSV 的字段完整性、重复项、跨表冲突、全唐诗语料命中和所列 `allusion_id`，全部通过。
- 在根 README 和分词说明中加入 LDA 人工覆盖规范入口。
- 新增 `segmentation/lda_standard/apply_standard.py` 和 7 项后处理测试，以字符偏移、最长匹配和原文重构校验实施规范。
- 生成全量 `sikufenci_lda.jsonl`、元数据和逐诗审计；43,030 首中 13,290 首词界变化，新增 14,957 个边界、移除 5,076 个边界。
- 独立复核 43,030 行 ID、字符重构、空 token、审计行数和输出 SHA-256，全部通过；六个指定表达全部成为完整 token。
- 发现统一语料与 sikufenci 基础结果仅有 1 处字符差异：ID 1 的“连薨”与“连甍”。另存结果以用户指定的 sikufenci 字符序列为准，差异写入元数据而未静默修改语料。

## 2026-08-30

- 使用工作簿读取工具核验 `tang_poetry 人工金标准.xlsx` 的 485 行主表，并抽取带工作簿及标注文件哈希的机器可读副本。
- 新增人工金标准覆盖脚本、使用说明和 6 项测试；完整标注、前缀标注、异文词界投射及不安全投射拒绝均有回归测试。
- 生成全量 `sikufenci_manual_gold.jsonl`、元数据和 485 行审计；426 首全文标注，59 首仅覆盖已标前缀，4 首异文只投射词界。
- 43,030 首中 412 首词界变化，新增 361 个边界、移除 913 个边界，总 token 数由 2,481,050 降至 2,480,498。
- 独立复核确认 42,545 首未标诗完全未动、全部输出可还原基础字符、完整和前缀人工词界均落实，输出 SHA-256 为 `0fa04536fa9a00ec13484f160069d399c9ebfd07ab8400263a08df474a826caf`。
- 从人工标注另存 Voyant 可直接读取的 `李白.txt` 与 `杜甫.txt`，分别含 11 首/957 词元和 24 首/1,782 词元；篇名未写入正文，诗篇以空行分隔。
- 两个文本均通过无 BOM UTF-8、逐段逐词一致、无残留斜线和省略号校验；同时记录李白 2 首、杜甫 7 首仅有前缀标注的覆盖限制。
- 依用户要求以全量 `sikufenci_manual_gold.jsonl` 重建 Voyant 文本：李白由 11 首扩充为 896 首/80,883 词元，杜甫由 24 首扩充为 1,158 首/101,076 词元；人工未标作品继续使用基础 sikufenci，前缀标注诗的后缀也已补全。
- 独立复核两个文本的语料作品数、唯一 ID、段落顺序和逐词内容均完全一致；无 UTF-8 BOM、无残留 `/` 或临时文件。李白/杜甫文件 SHA-256 分别为 `68464bd1298d21d458fc6a8610af7fad49313e43bace97467e180f300199fed0`、`1e640cc2df2b2fcc4413ff425d4ae67c56e57823a63b81b043a8d7ef90a0fc25`。
- 新增 `segmentation/manual_gold/build_preprocessing_resources.py`，以人工覆盖版 43,030 首结果重算全部汉字词元的 TF/DF，并生成可复现的五类预处理资源。
- 生成 4 条局部词形/异文记录、105/164 个核心/扩展停用词、182 个专名/典故/固定表达保护词面、默认/严格 TF/DF 规则以及 27 条可选语义映射；所有映射均保留原词，不作静默覆盖。
- 默认过滤档设为 `TF>=5 且 DF>=3`，严格敏感性档设为 `TF>=10 且 DF>=5`；不设自动最大 DF，避免误删高频诗歌意象，保护词豁免最小频率阈值。
- 新增 6 项资源测试，验证保护优先级、词表包含与互斥、局部异文不自动替换、高 DF 策略、输出哈希和高歧义语义标记；相关 23 项测试全部通过。
- 新增标准库实现的 `src/clean_corpus.py`、JSON 兼容 YAML 集中配置和 5 项关键约束测试，完整实现数据质量、规范化、保护优先、停用词、TF/DF、逐词删除原因、逐诗保留率、固定种子抽样、运行环境与哈希审计。
- 使用核心停用词和默认 `TF>=5、DF>=3` 档完成 43,030 首全量清洗，生成 `output/` 主语料、语义映射旁表以及 `reports/` 下全部必需和补充审计文件；未执行 Dictionary、BoW 或 LDA。
- 清洗后保留 1,485,707 个 token、23,123 个词型和 43,014 首非空诗；16 首零 token 诗仍按原顺序保留。检测到 25 行数据质量问题、0 个停用/保护冲突、14 个低频保护挽救词型和 4,985 个疑似保护词跨 token 实例。
- 复核主 CSV、token 文本和逐诗统计的 43,030 行顺序完全一致，200 首人工抽样固定，运行元数据中 17 个输出哈希全部匹配；相关 32 项测试全部通过。
- 新增 `src/run_tf_sensitivity.py`、集中配置和 5 项测试，在其他清洗规则完全冻结时生成 TF1、TF2、TF3、TF5 四组独立语料；既有 TF5 主输出只读保留。
- 生成阈值比较、累计 LOW_TF 归因、三组相邻差异、100,249 行纯频率低频候选、21 个案例词追踪、1,984 行极端长度审核和完整 Markdown 报告；当前环境无 matplotlib，按配置生成四张确定性学术风格 SVG 图。
- 实测固定 `DF>=3` 使 TF1/TF2/TF3 完全一致（35,063 型、1,525,932 次）；TF3→TF5 新损失 11,940 型、40,225 次，TF5 vocabulary/token retention 分别为 65.947010%/97.363906%。
- 核验 176 个现成保护词型、22,500 次出现跨四组全部保留；独立 TF5 与旧 TF5 的 43,030 行内容和 token 哈希完全一致，旧文件及全部输入哈希未变。37 项相关测试通过，27 个登记输出哈希复核无误；未运行 LDA。
- 新增 `src/run_tf_df_lda.py`、`config/tf_df_lda_config.yaml` 和 6 项测试，生成 A=(1,1)、B=(2,2)、C=(3,3)、D=(5,3) 四套 TF×DF 语料、词典、稀疏 BoW、文档索引及空诗清单。
- 完成 K=20/30/40/50/60、三个固定种子的 60 个 Batch LDA；保存逐模型 top-20 主题词、top-5 代表诗、document NPMI、Topic Diversity、训练 perplexity和可恢复内容哈希检查点。
- 生成 60 行模型指标、60 行匈牙利匹配跨种子稳定性、20 行配置汇总、48,000 行主题词、12,000 行代表诗、Markdown 报告和 215 个输出哈希元数据。
- 定向重跑 A/K=20 两个受系统休眠影响的计时，校正后 60 模型拟合时间合计 15,351.869 秒；模型指标与原检查点一致，旧计时文件留在日志目录供审计。
- 独立复核四组 43,030 行、词表/有效诗/空诗计数、矩阵形状与 nnz、A→B→C→D token 嵌套、D 与既有 TF5 哈希、全部登记输出哈希；全项目 43 项测试通过。
- 新增六候选人工 Topic 审核包生成器、集中配置和 4 项测试；按 NPMI 最接近三 seed 均值的规则选定六个主审核 seed，生成 140 张审核卡、人工评分表、模型汇总模板、代表诗概率诊断和缺失数据审计。
- 严格复用第一轮 top-20 主题词、top-5 代表诗和 soft prevalence；因完整模型与矩阵未保存，随机高概率诗、完整 prevalence、Topic similarity、跨模型匹配及 K 分裂映射明确标记为 unavailable，未重训或补造数据。
- 生成 11,940 行 C-only TF=3/4 词表及三组比较占位结构；核验 66 个输入与 205 个输出哈希零失败、六套卡片合计 140 张、评分初值全空、三个测试目录共 47 项通过。
- 新增 `src/rebuild_candidate_models.py` 与集中配置，按第一轮原参数精确重建六个 selected-seed 模型；六套 top-20 主题词及 top-5 代表诗 CSV 均与第一轮 SHA-256 完全一致，既有指标最大绝对差为 0。
- 为每个候选保存完整 `model.joblib`、components、归一化 Topic×Word、Document×Topic、行列索引、top-30、top-20 代表诗和复现审计；六个模型与三个 NPZ 均精确回读成功，语料及第一轮输入未改变。
- 新增完整审核包生成器，补齐每 Topic 20 首固定随机高概率诗、完整 prevalence、全分布 cosine、相似 Topic、C/D 匈牙利匹配、20 组配对审核页和 B/D K20→K30 Top-3 映射；140 行人工评分仍全空。
- 六模型高相似 pair 数为 28/34/42/28/26/18；C/D 20 对匹配均值 cosine=0.6803，B/D 各有 19 个 possible split 提示。重建层与审核层共登记 135 个输入哈希和 295 个输出哈希，零失败；全项目 54 项测试通过。
- 参照既有 D-K20、D-K30、C-K20 评分表的相同四维 1–5 锚点，逐题核读并完成 A-K20、B-K20、B-K30 共 70 个 Topic 的人工标签、评分、问题标记与证据备注；C/D 主表行保持空白。
- 新增 `human_topic_review/abc_human_topic_scores.json`、安全写回程序 `src/apply_manual_topic_scores.py` 及 2 项覆盖保护测试；执行 `--summary-only` 后三组平均总分为 2.5375、3.7500、3.091667。
- 生成三份与参照表同版式的评分工作簿，均含公式计算的 Topic 均分及模型汇总；逐表视觉检查和公式错误扫描通过。本轮 16 项可运行测试通过，完整套件受当前 Python 环境缺少 SciPy 限制未能重跑。
- 由 `segmentation/results/sikufenci_manual_gold.jsonl` 导出 `segmentation/manual_gold/全唐诗.txt`：无 BOM UTF-8、CRLF 换行、单空格分词、空行分诗，不含 ID、诗人或篇名；按原顺序保留 43,030 首、2,480,498 个词元及 11 条空正文。
- 反向拆分 TXT 后逐诗与源 JSONL 完全一致；文件大小 11,756,159 字节，SHA-256 为 `047826f8db1e3650e0d221eef4147bea99c656aa16c3435e80aea5cef350f506`。

## 2026-08-31

- 核验六候选共 140 个 Topic 的人工评分完整性，选定最新合并工作簿作为唯一人工来源，并记录四个输入文件的路径、修改时间与大小；全程只读既有评分和模型指标。
- 汇总 NPMI、稳定性、多样性、人工四维均分以及 high/low/mixed/noise/duplicate 比例，生成九指标排名、广义 Pareto 前沿及有向支配矩阵。
- 用 min-max 归一化建立平衡、人文学优先、数量优先三种权重情境；稳定前 3 为 B_K20、C_K20、D_K20，C_K20 在前两种情境居首，D_K20 在数量优先情境居首。
- 完成 K=20/30 与 A→B→C→D 过滤强度敏感性解释：K=30 主要增加拆分和噪声；B→C 改善稳定性与噪声控制，C→D 则是统计指标与文学低频信息保留的明确权衡。
- 生成 `model_comprehensive_evaluation.xlsx`、`paper_tables.xlsx`、完整 CSV、四张可视化和 `final_model_selection_report.md`；工作簿派生列使用公式，公式错误扫描为零，全部工作表和图像完成视觉复核。
- 最终建议 C_K20 为主模型，D_K20 为次模型/稳健性检验，B_K20 为宽词表对照；A_K20、B_K30、D_K30 不进入主分析。

## 2026-09-01

- 新增 `src/build_literary_history_dataset.py` 与集中配置，自动盘点输入、复制冻结 B_K20/C_K20/D_K20、生成 SHA-256 manifest，并以 33 项完整性检查作为下游构建门禁；全程未调用 LDA 训练或重新变换。
- 建立 C_K20 20-topic 正式 taxonomy、唯一 `topic_metadata` 主表和解释边界；分类为 9 CLEAR、4 BROAD、4 MIXED、3 NOISE。
- 在共享词表上计算 C↔B、C↔D 共 800 行 cosine/Top30/代表诗证据，保留 Hungarian 一对一基线并人工核定 split/merge/weak/unmatched；生成 20 份逐 Topic alignment 审核卡，LOW confidence 明示 `NEEDS_HUMAN_REVIEW`。
- 严格按冻结 document index 的 poem_id 合并 43,015×20 矩阵与权威 `poems.jsonl`/C 语料；0 个 matrix row 未匹配，15 首清洗后空诗单列，不以作者加篇名连接。
- 生成 43,015×54 的 CSV/XLSX 母数据集、Top500/Top100 反向索引、诗到主题摘要、metadata QC 与项目报告；六个派生连续字段在 XLSX 中含 258,090 个全行公式及经全量测试验证的缓存值。
- 15 项最终数据测试和主 XLSX XML 级结构/公式/首中末行缓存值核验全部通过；四份工作簿完成错误扫描与各 sheet 视觉检查。大型主表因 artifact-tool 在 8 GB 堆上限无法序列化，改用 bundled XlsxWriter 常量内存流式写入，taxonomy/alignment/topic_metadata 仍由 artifact-tool 生成。
- 新增 `src/build_historical_metadata.py`、集中配置、7 项单元测试和固定 CBDB 2026-08-29 SQLite 快照，完成 2,528 个 QTS 原始作者实体的保守标准化、CBDB 正名/别名候选审计、年代语义分离、双轨作者分期、字段级来源和人工审核队列；未修改 LDA 或原母表。
- CBDB 安全匹配 1,114 位作者，覆盖 28,381 首；初唐/盛唐/中唐/晚唐作者分别为 102/124/162/91 位，已知 author-period 覆盖 23,609 首。2,317 位待审核作者按身份歧义、分期未解、跨期/边界、LOW 和冲突排序。
- 生成 43,015 × 68 enriched master CSV、作者主表、候选/来源/覆盖率/QC/数据字典/方法报告和可复现 source snapshot；17 项数据一致性检查全 PASS，20 维 Topic probability 校验和完全不变。
- 作者主表、人工审核、数据字典三份 XLSX 经 artifact-tool 逐页渲染和独立 ZIP/XML 页签、维度、表头核验全部通过；新增工作簿验证脚本。经用户单项授权，43,015 × 68 enriched master XLSX 已用 XlsxWriter 3.2.9、`constant_memory=True` 从权威 CSV 流式生成。
- 大型 master XLSX 的 `Data!A1:BP43016`、ZIP/XML/relationships、43,015 个 poem_id 全量顺序、首中末行和固定随机 100 行全 68 字段均通过验证；共比较 7,004 个单元格，最大浮点绝对误差 `4.44e-16`，无截断、错位、重复或丢失。导出例外、时间戳和 CSV/XLSX SHA-256 已写入正式报告与版本 JSON。
- 使用含 SciPy 1.17.1 的系统 Python 复跑 tests、stopwords/tests、segmentation/tests；加入 6 项大型 XLSX 类型/容差测试后，42+4+23=69 项全部通过。打包 Python 缺少 SciPy 的首次导入失败已确认是环境依赖差异。
- 新增 `src/run_period_representativeness_audit.py`、集中配置与规则测试，对 43,015 首诗和 2,528 位作者完成 period KNOWN/UNKNOWN 覆盖、作者产量、20 Topic SMD、Topic type mass、dominant topic、entropy、CBDB 状态、缺失可预测性与三套未来样本代表性审计；未比较初盛中晚唐，未重训 LDA，未修改历史元数据。
- 以 1,000 次 multinomial count bootstrap（seed=20260901）生成 16 份核心审计 CSV、43,015 行样本标签、5 张学术图和正式报告；12 项 QC 全 PASS，权威 master SHA-256 前后保持 `93fa45a5d848d0708fd1f4fb44289c26d8e98b7eec5b56bc440dc0de95de801f`。
- 使用 artifact-tool 生成 14-sheet `Period_Metadata_Representativeness_Audit.xlsx`，含交叉引用 KPI、原生图表、完整底表、风险着色、方法与 QC；14 页全部渲染复核，公式错误为 0。新增独立 ZIP/XML/relationships、Excel 解析、16 张 CSV 表格镜像、5 张 PNG 和 manifest 哈希验证，6+4 项任务测试全部通过。
- 新增配置驱动的 C_K20 作者时期趋势流水线，直接复用审计冻结的 STRICT/CORE/EXPANDED 标签，形成 poem-weighted 与 author-balanced 六套规格；全程未修改权威 master、20维 Topic probability、taxonomy、作者时期或 CBDB 匹配。
- 以作者为 cluster 完成固定 seed 20260901、2,000次 bootstrap，生成 Period×Topic 均值与区间、作者 profile、peak/trough、完整 rank、数值 shape、bootstrap-supported direction、六规格稳健性、聚合敏感性、样本敏感性和作者 Topic mass 集中度。
- CORE poem/author peak 分布分别为初盛中晚 6/5/3/6 与 4/4/5/7，20 Topic 中 peak 一致10个；六规格稳健性 HIGH/MODERATE/LOW 为0/16/4。9个 CLEAR Topic 均为 MODERATE，并以 candidate robust trend 交付人工审核，不作文学史解释。
- 生成13份数据 CSV、六幅结构图、14节 Markdown 报告和12-sheet `QTS_Author_Period_Trend_Structure.xlsx`；15项不可变性检查、公式错误扫描、12页渲染与 OpenXML/缓存值核验通过，全仓库94项测试通过。

## 2026-09-02

- 新增 `src/run_author_period_effect_validation.py`、`config/effect_validation_config.yaml` 与16项核心/输出测试；用 NumPy/SciPy 显式实现时期 cell-means OLS、作者 cluster CR1、AUTHOR HC3、有限样本 Wald/t 检验、BH-FDR、Hedges' g、Cliff's delta、六规格敏感性与 CLR 临时转换。
- 对9个 CLEAR Topics 运行 CORE_POEM/CORE_AUTHOR 正式推断，并对20 Topics 保留六规格六对比描述性附录；5,000次作者 cluster bootstrap 固定 seed 20260901，未把 STRICT/EXPANDED 扩展为额外 p-value families。
- 生成12份数据 CSV、9张统计图、18节 `QTS_Author_Period_Effect_Size_Statistical_Validation_Report.md`、statistically supported candidate 清单与 Author Influence/LOAO 队列；15项数据不可变性检查全部通过。
- 使用 artifact-tool 生成13-sheet `QTS_Author_Period_Effect_Size_Statistical_Validation.xlsx`，Overview 公式核对 omnibus 5/9 与2/9、planned contrast 状态计数和13项 supported candidates；13页全部渲染，公式错误扫描为0，工作簿 SHA-256 为 `36185ca0e7f8c5f99dcedda957be31a0df0d4e5ee19dee31c1e05e6250c815fc`。
- 最终执行 `python -m unittest discover -s tests -v`，83项测试全部通过；权威 master、20维 probability、时期标签和 taxonomy 未修改。

## 2026-09-03

- 新增 `src/run_author_influence_loao.py`、`config/author_influence_config.yaml` 和工作簿构建器，完成 C_K20 作者贡献/集中度、LOAO、两方案 Top contributor 排除、156组 bootstrap CI、聚合冲突分解与13组候选作者稳健性诊断。
- 明确 GLOBAL_TOP 主分类与 PER_PERIOD_TOP 敏感性分列，新增方案差异与 Top 作者表别名测试；扩充报告中的规则优先级、全部阈值、近零反向解释和两方案差异，未改变冻结概率、作者时期或样本标签。
- D: 曾反复报“设备未就绪”，用户确认恢复后续跑成功；2,000次 bootstrap replicate 哈希与前次完全一致，15项不变量检查全部通过，13组主分类仍为1稳健/8集中但稳健/4敏感。
- 使用试算表核验流程修复作者 ID、数据字典和排除摘要截断，并在 Overview 明示阈值；13页、两处方案分列区域和5张图完成视觉复核，公式错误0。最新122项测试全部通过，8项输入/34项输出哈希全部吻合，工作簿哈希 `70735b23c8b2047ed80c38d9ee64cf96748e4669addd47d2e355ad4ed254ad42`。
- 记录生成器“导出与自检完成但命令退出码非零”的未解运行问题；成品另经 ZIP/XML、缓存计数、哈希测试独立验收，未掩盖退出状态。更新项目记忆，保留9组下一阶段候选并在 A–N 交付后停止；没有运行 B/D 模型或文学史解释。
- 新增BCD发现层配置、分析/报告模块与24项核心及输出测试，复用9项eligible来源、既有语义映射、43,014首共同集合及冻结历史标签；不重新训练或选择LDA模型。
- 完成5,000次作者bootstrap、split概率和/single-best、C原始/共同及native基准、条件CLR、优先四项Top1/Top3检查。最终4项ROBUST_ACROSS_BCD、5项PARTIALLY_ROBUST，原C统计与LOAO状态不变。
- 生成19份CSV、6张PNG、19节报告与12-sheet artifact-tool工作簿。20项不变量、重复运行19份CSV/抽样哈希一致性、全量XLSX数据镜像与158项仓库测试通过；修正长表头显示、空单元格比对以及删除后某时期无作者时的不可估计标记。
- 记录来源缺少正式人工审核状态、T07 split/样本/聚合限制、T19 D合并代理以及T16/T03 E-H的B区间不确定性。工作簿导出后的非零退出状态保留为运行风险；成品另经结构、公式、哈希与视觉核验。本阶段交付A–N后停止，未开展文学史解释。
- 完成19节人文学者审读报告及同文DOCX，含四项核心发现统一A—H模板、12张表、55条术语、误读说明与后续审读原则；正文为17,730个中文字，Markdown为权威文本。
- 转录与核对9项发现162个CORE差值/区间、18条原映射及所有既有分类，129项冻结来源哈希未变；记录旧优先四项与最终四项差别、代表诗时期资格及trend_shape字段释义不一致，不改冻结结果。
- DOCX同文及结构检查通过；LibreOffice/soffice缺失导致分页渲染不可用，明确保留视觉验收限制。完成后停止，未细读、查询外部研究或重新统计。
- 新增冻结结果论文绘图与独立验收脚本，完成10图组/13核心图位/27版本，输出PNG600dpi、矢量PDF、SVG、逐图CSV、图注和manifest；不修改既有研究结果。
- 669项生成核验、2,369项独立核验及图9最终定向检查通过；27份PDF逐张渲染与视觉复核、81项图文件哈希和79项冻结来源哈希均通过。修正图例间距和注释遮挡，记录图6均值CI缺失与T16区间宽度表述边界。
## 2026-09-07 QTS LDA K-value perplexity diagnostic

- 新增 `src/run_perplexity_diagnostic.py`、`src/build_perplexity_outputs.py`、两类单元/产物测试及独立 `model_selection/perplexity_diagnostic/` 输出树。
- 完成 4×5×3=60 次训练子集 Batch LDA 与测试集 perplexity/score，保存固定split、原始运行、三seed汇总、既有指标并列表、C专项表、5图、中文报告和哈希审计；未改写语料、冻结模型或既有模型选择结论。
- 60次及10项完整性检查通过；诊断结论 `SUPPORTS_K20`。A宽词表有15,271个测试词次未见于训练子集，作为冻结全语料词表设计限制明示。

## 2026-09-07 C 语料 Lower-K 边界与 held-out perplexity

- 新增 `src/run_lower_k_boundary_perplexity.py`、集中配置、工作簿构建器和 8 项单元测试；完成 C 的 K=5/10/15 三 seed 全语料模型及 A/B/C/D 固定 80/20 split 共 69 个 held-out 运行，支持检查点续跑与输入哈希门禁。
- 复用原 NPMI、Topic Diversity 和跨 seed 稳定性实现，补充低 K→C_K20 全 topic-word cosine、匈牙利基线、one-to-many 合并证据、30 张人工审核卡与空白评分表；未自动命名或评分。
- 生成低 K 指标、perplexity、相邻 K 变化、多指标诊断、alignment、split、17 项不变量、8 幅图、三份报告及三页 `lower_k_human_review.xlsx`。自动状态为 `LOWER_K_COMPETITIVE`，综合状态为 `REVIEW_K15_VS_K20`；冻结 C_K20 及全部下游结果未修改。
- 154 项主测试、4 项 stopwords 测试、23 项 segmentation 测试与 17 项任务不变量全部通过；30 行人工字段保持空白，工作簿公式/数据验证/OpenXML/渲染及登记输出哈希独立核验通过。工作簿生成器在成功导出后仍返回退出码 1，保留为工具运行风险，不据此否定独立验收结果。

## 2026-09-07 Lower-K Humanities Semantic Audit

- 新增 `src/prepare_lower_k_semantic_audit_evidence.py`、`src/run_lower_k_semantic_audit.py`、`src/verify_lower_k_semantic_audit.py` 与 artifact-tool 工作簿构建器；只读冻结模型和既有对齐数据，未重新训练或修改任何矩阵、词表及 C_K20 taxonomy。
- 对 K5/K10/K15 共 30 个 Topic 完成 Top30、Top20、author-balanced 与 K20 alignment 三类证据审读，生成 30 张统一结构审核卡、逐题/分 K/K15 专项/核心覆盖/30题优先队列数据和 13 节报告。
- 生成 4 张 300 dpi 图与 6-sheet `QTS_Lower_K_Researcher_Review.xlsx`；9 个 Researcher 字段共 270 个单元格保持空白，artifact-tool 检查、6 页渲染、公式错误扫描与 XLSX ZIP 验证通过。
- 15 项不变量与完整性测试全部通过，53 项优先来源无缺失且 SHA-256 未变。K15 暂定为 `COMPARABLE` / `K15_COMPETITIVE_WITH_CAVEATS`，下一步 `REVIEW_K15_TO_K20_FINE_GRID`；未运行 K16–19。

## 2026-09-07 C Corpus K=2–30 Exhaustive Scan

- 新增 `src/run_exhaustive_k_scan.py`、集中配置和 7 项专用测试，直接复用冻结 C split 与原 top-20 文档 NPMI 实现，完成 29×3=87 个 train-only LDA 检查点；只写新目录，不覆盖旧 NPMI、冻结模型或 C_K20 状态。
- 生成逐 run/逐 K 汇总、差分、局部极值、平台和候选共 6 份主 CSV，6 张 300 dpi PNG、14 节中文报告、输入/输出 manifest 与 15 项不变量审计。
- 定量结果为 K12 NPMI 全局峰、K11–K15 top-1% 带、K11–K19 98% 连续主平台；perplexity K2 全局最低且 K2–30 严格递增。K14 是指定 K13/14/16/17/18/19 中唯一超过 K15 者，最终状态 `FINE_GRID_REVEALS_NEW_CANDIDATE`。
- 15/15 不变量、161 项全仓测试与六图视觉检查通过；旧诊断中 K20/K30 的 6 个重叠 perplexity 值逐项精确复现。保留 sklearn 默认 prior 随 K 为 1/K 的限制与后续 fixed-prior sensitivity 建议，本轮未自动运行。

## 2026-09-08 QTS Quantitative Candidate Validation

- 新增 `src/run_quantitative_candidate_validation.py`、集中配置与 7 项核心测试，完成六候选、十种子、两套 prior 共 120 个 LDA 检查点；生成逐 run/汇总/跨种子 Hungarian stability/配对差异/prior sensitivity/Pareto/shortlist 数据、六幅图及 18 节报告。
- 新增模型内 topic cosine/Top20 Jaccard 冗余、相似 pair 数、topic prevalence 与低占比诊断；未生成加权总分，未执行主题命名或语义审核，未改冻结 C_K20。
- 量化 shortlist 为 K12/K15/K20；K12 是原先验 NPMI 领跑者，K15 是固定先验领跑者，K20 保留 frozen reference。主实验 15/15 不变量和六图视觉检查通过。
- 新增 `src/run_candidate_iteration_sensitivity.py` 与 2 项规则测试；按触发条件仅复核 K12/K13/K15/K20 的原 3 seeds，完成 max_iter=60 的 21 次唯一拟合、24 行登记、30→60 配对汇总和独立哈希审计。全仓 170 项测试通过。

## 2026-09-08 Four-Model Blind Audit Input Gate

- 只读核验四候选的冻结输入：K15/K20 完整，K12/K14 缺少逐诗 document-topic、原始 components/model 与代表诗；40 份 K12/K14 候选 NPZ 均只有单位化 topic-word 向量和 Top20 词索引。
- 因任务同时禁止重训并要求真实 `topic_probability`，未开始盲审，也未写入任何审计产物；记录为待补输入或待用户变更约束的阻断项。

## 2026-09-09 K12/K14 Model Artifact Recovery and Blind-Audit Data Completion

- 新增 `src/run_artifact_recovery_k12_k14.py`、集中配置和 6 项专项测试，建立 40-run 检查点续跑、严格 topic-word 复现门禁、theta/代表诗恢复、全语料模型生成、四模型口径审计和输出 manifest。
- 精确重建 K12/K14 × 10 seeds × ORIGINAL/FIXED_K20 prior 的 40 个 train-only 模型；40/40 为 `EXACT_REPRODUCTION`，0 permutation、0 numerically close、0 mismatch。全部生成 train/test/full theta、行索引、模型对象和四套真实概率代表诗。
- 按预注册的“非最高 NPMI、接近中位数、稳定性中心度优先”规则选定 K12 seed=101、K14 seed=31415，并生成两套 43,015 首 full-corpus ORIGINAL_PRIOR、max_iter=30 模型。
- 只读核对既有 K15/K20 seed=2026；四模型 corpus/vocabulary/fit scope/prior/max_iter 直接可比。原 K12/K14 NPZ、K15 与 frozen K20 哈希未变，18/18 QA、6 项专项测试及全仓 176 项测试通过，状态为 `BLIND_AUDIT_READY`；未开始语义盲审。

## 2026-09-09 QTS Four-Model Blind Humanities Semantic Resolution Audit

- 新增盲审准备、汇总/冻结/解盲、artifact-tool 工作簿构建与 18 项专项 QA 脚本；完成 61 张匿名随机卡、11 份规定数据 CSV、5 张 300 dpi 图、13节盲审报告、16节解盲报告和8-sheet研究者工作簿。
- 盲审快照 SHA256 为 `7c7c33d5e581e5e7ebf67961cd415e2166862c503203ea529139f4b056739a8b`，解盲前后完全一致；映射 A/B/C/D→K12/K20/K15/K14，盲态排名 A>C>D>B。
- 七项 SPLIT 为 meaningful/minor/fragmentation/uncertain=3/1/3/0；K20 相对 K14/K15 的额外分辨率主要为碎片化或有限增益。量化与语义证据不作任意加权，最终状态 `PREFER_K12`。
- 工作簿 732 个 Researcher 单元格全空，3组数据验证、8页渲染与真实K/量化泄漏扫描通过；18/18专项 QA 与全仓既有176项 unittest 回归通过。未重训模型、未修改冻结输入、未复用旧评分、未重跑历史下游分析。

## 2026-09-09 QTS K12/K15/K20 Researcher-Led Humanities Adjudication PREPARE

- 新增 `src/prepare_k12_k15_k20_human_adjudication.py`、`src/validate_human_review.py`、`src/freeze_human_review.py`、`src/summarize_human_review.py`、artifact-tool 工作簿构建器、集中配置及 10 项专项测试。
- 对可比的 K12/K15/K20 full-corpus 代表模型生成 47 张全局随机盲号卡、Top30 词与 30 首/题诗证据；所有 14 个 Topic 级研究者字段及后续裁决字段均为空，真实映射和历史 AI 参考隔离在 `_private`。
- 生成 20 个 K12→K15 split/merge 事件、16 个 K15→K20 split/merge 事件与 15 个 K20 碎片化候选对，仅作定量定位，不自动判断。
- 生成 11-sheet `QTS_K12_K15_K20_Human_Review.xlsx`：第一阶段仅 3 页可见，后续 8 页均为 `veryHidden`；11 页渲染、公式扫描、OpenXML 页签状态与字段空白核验通过。
- 专项 10/10、主测试 186、stopwords 4、segmentation 23 全部通过；空白工作簿冻结门禁按预期拒绝，未解盲或选择最终 K。状态 `READY_FOR_RESEARCHER_REVIEW`。

## 2026-09-09 QTS C_K12/K15/K20 pyLDAvis Comparative Visualization

- 新增 `src/generate_pyldavis_three_model_comparison.py`、`src/verify_pyldavis_three_model_outputs.py` 与集中配置，读取现有 full-corpus phi/theta、冻结 C DTM/词表和 frozen C_K20，生成三份保号离线 pyLDAvis 与 comparison 首页。
- 输出 43,015 行文档对齐 QC、47 行 Topic ID map、三套坐标/prevalence/relevant terms、模型内 pair/nearest similarity、55 条既有相邻 K alignment、标签参考、诊断汇总、八幅静态图、17节报告和 artifact-tool 研究者核查表。
- K12 的 `topic_word_unit.npz` 经检查为 cosine 单位向量，改读与模型逐元素相同的 `components.npz` 并逐行归一化；未调用 fit/transform，未修改源对象。
- 25/25 QA 与 XLSX 渲染通过；47/47 Topic ID、sort_topics=false、离线资源、入口链接和源哈希均通过。最终状态 `PYLDAVIS_THREE_MODEL_READY`；未作最佳 K 判断或文学自动命名。

## 2026-09-10 QTS K12/K15/K20 Card-by-Card AI Provisional Review

- 新增 `src/prepare_ai_card_review.py`、`src/freeze_ai_topic_review.py`、`src/build_ai_review_outputs.py`、artifact-tool 工作簿构建器与 15 项专项验收；完成 47 张匿名卡的逐题暂定标签、八项评分、类型、拆分判断、证据和说明。
- 第一阶段冻结 SHA256 为 `d9f17b9928e3ac58190673de59e15e2abb99b1ddf10cd57bb679d96db75cb9c9`，解盲后 15 个暂定字段逐值不变；记录同一对话旧 AI 上下文造成的严格独立盲审限制。
- 生成解盲主题表、三模型汇总、36 个 split/merge 事件、15 个碎片化候选、47 行研究者优先队列与 15 节报告；字段名统一为 `AI_Provisional_Split_Judgment` 与 `AI_Provisional_Fragmentation_Status`。
- 保留原工作簿与同哈希备份，另生成含 5 个可见 AI 页的 `QTS_K12_K15_K20_Human_Review_AI_Filled.xlsx`；原 11 页、全部 `Researcher_*`、公式、批注、数据验证和可见性经逐页对比不变。
- 专项 15/15、全仓 201 项 unittest、公式扫描与 5 页视觉核验全部通过。状态 `AI_MORE_REVIEW_REQUIRED`；未作最终 K 或人工裁决，也未重训模型。

## 2026-09-10 QTS C_K12 Historical Analysis Full Rerun

- 新增 `config/k12_historical_analysis_config.yaml`、`src/run_k12_historical_analysis.py`、`src/build_k12_k20_historical_comparison.py` 与 6 项交付测试；参数化复用既有 C_K20 trend/effect/representativeness/LOAO 代码和冻结阈值。
- 从 C_K12 43,015×12 full-corpus theta 生成三样本×两口径趋势、topic-specific known/unknown SMD、CLEAR primary CR1/HC3 与 BH-FDR、完整 12 维 CLR、8 个候选的 LOAO/Top1/3/5/10、7 条最终保留候选和 Humanistic Interpretation Queue。
- 在读取 K20 前冻结 37 项 K12 核心产物及 SHA-256；随后以既有语义家族和 K12→K15→K20 词项路径比较旧 9 条保留 finding，输出逐条证据 CSV、比较报告和 18 节总报告。旧 K20 reference 文件前后哈希一致。
- 交付目录 `history_analysis/C_K12/` 含 59 项已 hash 文件；20/20 QA、全仓 207 项 unittest 通过，8 幅 PNG 逐张视觉核验通过。最终状态 `C_K12_HISTORY_ANALYSIS_COMPLETE`；未运行 B/D 或 B/C/D robustness，未修改 K 选择与 metadata。

## 2026-10-31 GitHub Repository Initialization and Public Upload

- 将既有非 Git 项目目录初始化为 `main` 分支，连接 `https://github.com/Ethan-do0707/Quantangshi-LDA.git`，并以远端初始提交为父提交保留历史。
- 新增中文注释 `.gitignore`，排除本地环境、依赖副本、大型模型与数据库快照、可重建检查点、缓存、临时文件及盲审私有映射；未删除任何本地文件。
- 首次公开上传包含 2,278 个文件、约 500 万行新增内容；远端确认主提交为 `954aeadb1a44f98cf93e554d63d1468f5dd02350`，文件树完整且无大于或等于 100 MiB 的对象。
- 上传前凭据模式扫描无命中；分词 23/23、停用词 4/4 通过，主测试 205/207 通过。两项主测试错误定位为旧 `D:` 盘绝对路径残留，未在本次上传任务中改写历史 manifest。
