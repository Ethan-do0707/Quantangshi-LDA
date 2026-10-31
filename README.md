# 全唐诗 LDA 数据处理

当前仓库包含 `tang_poetry.sql` 及针对其中 43,030 条诗歌正文的四路分词比较流水线。

使用方法、算法口径、模型限制和输出说明见 [segmentation/README.md](segmentation/README.md)。

面向 LDA 的人工覆盖规则、典故词、人地名与禁合并口径见 [segmentation/lda_standard/README.md](segmentation/lda_standard/README.md)。

依据 `tang_poetry 人工金标准.xlsx` 对原始 sikufenci 作局部人工词界覆盖的口径、结果与审计见 [segmentation/manual_gold/README.md](segmentation/manual_gold/README.md)。

基于 sikufenci 结果建立的《全唐诗》专用停用词表、审计统计和使用说明见 [stopwords/README.md](stopwords/README.md)。

## LDA 语料清洗

严格使用人工金标准叠加版五类资源执行清洗：

```powershell
python src/clean_corpus.py --config config/cleaning_config.yaml
```

程序不会覆盖原始语料，不会应用未明确获准自动执行的词形/异文记录，也不会把语义映射写回主语料。LDA 主语料写入 `output/`，逐项审计、固定种子抽样、运行元数据与清洗报告写入 `reports/`，运行日志写入 `logs/`。人工审核阈值只标记异常诗歌，不触发删除。

### LOW_TF 阈值敏感性实验

在其余规则完全冻结的前提下，独立生成 TF1、TF2、TF3、TF5 四组对照语料：

```powershell
python src/run_tf_sensitivity.py --config config/tf_sensitivity_config.yaml
```

四组语料写入 `output/tf_sensitivity/`，比较表、差异词表、案例追踪、极端长度诗、图表和实验报告写入 `reports/tf_sensitivity/`。既有 `output/lda_corpus.csv` 与 `output/lda_tokens.txt` 只用于 TF5 一致性核验，不会被覆盖。

### TF×DF 与第一轮 LDA 敏感性实验

按 A=(TF≥1,DF≥1)、B=(2,2)、C=(3,3)、D=(5,3) 生成四组语料，并对每组运行 K=20/30/40/50/60、三个固定随机种子的 Batch LDA：

```powershell
python src/run_tf_df_lda.py --config config/tf_df_lda_config.yaml
```

程序为每个模型保存检查点，可在中断后用同一命令续跑。四组 CSV/TXT、词典、稀疏 BoW 和空诗清单位于 `output/tf_df_sensitivity/`；逐模型主题词与代表诗位于 `output/lda_sensitivity/models/`；NPMI、Topic Diversity、跨种子 Topic Stability 和汇总报告位于 `reports/lda_sensitivity/`。空诗保留在语料文件中，但不进入 LDA 矩阵。

### C 语料 Lower-K 边界与 held-out perplexity 诊断

复用第一轮冻结的四套 BoW、词表、文档索引和 Batch LDA 参数，对 C 语料补跑 K=5/10/15 的全语料模型，并以固定 80/20 文档切分计算 A/B/C/D 的训练集与测试集 perplexity：

```powershell
python src/run_lower_k_boundary_perplexity.py --phase all
```

全部新增模型、检查点、指标、跨 K 对齐、人工审核卡、工作簿、图表和报告均写入 `model_selection/lower_k_boundary_perplexity/`。程序不会覆盖第一轮模型或六候选冻结模型；人工主题标签和评分字段始终留给研究者填写，自动指标不代替主题可解释性判断。

### 六候选模型精确重建与人工 Topic 审核包

按第一轮原 BoW、原参数和已选 seed 精确重建 A-K20、B-K20、B-K30、C-K20、D-K20、D-K30，并保存完整模型及矩阵：

```powershell
python src/rebuild_candidate_models.py --config config/candidate_rebuild_config.yaml
```

每个模型只有在重建 top-20 主题词 CSV 与第一轮文件 SHA-256 完全一致、top-5 代表诗 CSV 也完全一致，并通过指标、矩阵和模型回读检查后，才写入 `output/lda_candidate_rebuild/`。每套目录包含 `model.joblib`、原始 components、归一化 Topic×Word、Document×Topic、行列索引和复现审计；命令支持复用已验证检查点。

随后使用完整矩阵生成审核卡和人工评分表：

```powershell
python src/build_human_topic_review.py --config config/human_topic_review_config.yaml
```

审核包位于 `human_topic_review/`。每个候选仍使用三个 seed 中 NPMI 最接近该配置三 seed 均值者，不选择单次最高值。完整包包含每主题 top-30 词、top-20 代表诗、固定随机高概率诗、完整 prevalence、模型内相似度、C/D 匈牙利匹配、B/D 的 K20→K30 Top-3 映射和 140 张审核卡。人工标签与评分始终留空，不由程序自动判断。

人工填写 `human_topic_review/human_topic_evaluation.csv` 后，只更新评分汇总而不训练 LDA：

```powershell
python src/build_human_topic_review.py --config config/human_topic_review_config.yaml --summary-only
```

## 作者历史年代元数据增强

第一期作者身份、CBDB 候选、生卒年与活动期、双轨作者分期、字段级来源、人工审核队列及诗级继承结果见 [historical_metadata/README.md](historical_metadata/README.md)。本阶段不修改任何 LDA 概率，也不把作者时期解释为单篇作品创作年代。

## 既有模型 pyLDAvis 可视化

运行 `python generate_pyldavis_qts.py`，按 C_K20、B_K20、D_K20 顺序读取各自冻结模型、原始 DTM、原始词表和保存的文档概率，输出保留 T01–T20 编号的离线 HTML。不会训练模型、分词或重建词表；`--check-only` 仅核验输入。文件路径、API 选择与验收限制见 [pyLDAvis 说明](outputs/pyldavis/README.md)。

## B–C–D 发现层稳健性

仅验证 C_K20 Author Influence/LOAO 保留的具体 Topic×contrast，读取冻结 B/C/D 矩阵与既有语义对齐，不重新训练或选择模型：

```powershell
python src/run_bcd_finding_robustness.py --config config/BCD_robustness_config.yaml
python -m unittest discover -s tests -p 'test_bcd_finding_robustness*.py' -v
```

共同文档、映射、六规格效应、作者/CLR/单目标敏感性、发现层阶梯与配置哈希见 [BCD结果说明](outputs/01a0628d-ef13-7961-85ce-146e0a14f60a/BCD_finding_level_robustness/README.md)。主分析输出CSV，工作簿由独立artifact-tool builder生成；builder路径和哈希记入manifest。4项跨模型稳健、5项部分稳健只构成方法证据，不能替代人工语义确认或文学史解释。
