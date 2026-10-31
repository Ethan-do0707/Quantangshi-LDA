# 《全唐诗》pyLDAvis

双击 `pyLDAvis_C_K20.html` 打开主模型；B、D 为各自独立的词表对照。HTML 内嵌 D3、LDAvis 和样式，无需启动服务器或连接 CDN。

运行项目根目录的 `python generate_pyldavis_qts.py` 可重新生成；`--check-only` 仅核验输入。详细路径、shape、输入/输出哈希及环境见 `run_report.json`，简洁终端报告见 `run_report.txt`。

## 已保存的对象

- 模型：`outputs/01a05061-9ee3-7610-a401-abfec81de1b2/literary_history_dataset/freeze/{C_K20,B_K20,D_K20}/model.joblib`。
- 词表：上述每个冻结目录各自的 `vocabulary.csv`；与对应 `output/tf_df_sensitivity/{C,B,D}/vocabulary.csv` 的内容和顺序哈希完全一致。
- DTM：`output/tf_df_sensitivity/{C,B,D}/bow_matrix.npz`，每个模型使用自己的矩阵。
- 文档概率：各自冻结目录的 `document_topic_probabilities.npz`；文档索引与原始 DTM 行索引完全相同。

训练代码直接使用原始稀疏 BoW 与 `term_index`，没有使用或保存 CountVectorizer。`SavedVocabulary` 只是读取原词表的只读接口，不拟合词表，也不分词或生成 DTM。

## 编号和阅读口径

固定 `sort_topics=False`、`start_index=1`、`R=30`、`lambda_step=0.01`。原模型 T03 对应 Topic 3，T01–T20 均逐项验证；结果保存在 `topic_id_audit_*.csv`。核验同时覆盖原模型各行的关键词概率与原审核 top-30，避免仅检查连续编号。

点击主题后，可使用 λ=1.0、0.6、0.2 比较相关词。圆面积使用文档词次加权主题占比，与逐诗等权的平均概率口径不同。距离图采用官方 PCoA 二维投影，重叠只作“建议人工复核”，不自动命名、合并或重训。

## 版本与 API

安装包版本为 pyLDAvis **3.4.1**；其内部 `__version__` 为 **3.4.0**，两者均记录。scikit-learn 为 **1.9.0**，与保存模型时一致。

已检查实际 `pyLDAvis.lda_model.prepare`：它无条件执行 `model.transform(dtm)`，即使传入已保存的文档概率也仍重复推断。本项目已经保存完整概率，最终使用官方 `pyLDAvis.prepare` 数组接口，直接传入原模型 components 的逐行归一化结果、原始 Document×Topic、DTM 行词次和列词频、原始有序词表。没有执行 `fit`、`transform` 或词表重建。接口定义见 [pyLDAvis 官方 API](https://pyldavis.readthedocs.io/en/latest/modules/API.html)。

HTML 先经 `pyLDAvis.save_html` 生成，再内嵌发行包自带 JS/CSS；保留上游资源版权信息。首次尝试 sklearn 便捷接口产生多进程开销并被中止，最终三份产物已全部由上述矩阵接口重新生成。

## 验收范围

三套 shape、哈希、DTM 逻辑指纹、文档行和词表列索引、词次与词频、模型 components、概率归一化、全部 60 个 Topic 的编号/关键词/占比核验通过。原始输入未变。

当前浏览器工具安全策略禁止打开本地 `file://` 页面，未完成浏览器内点击和滑块实测；不将静态检查宣称为交互测试。请在本机浏览器双击主模型 HTML，选择 Topic 3 并切换 λ 完成最后交互验收。
