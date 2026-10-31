# 《全唐诗》停用词表

本目录以 `segmentation/results/sikufenci.jsonl` 的 43,030 首分词结果为基础，建立面向古诗 LDA 的专用停用词表。

## 文件

- `tang_poetry_stopwords_core.txt`：默认使用的核心停用词，一词一行；
- `tang_poetry_stopwords_extended.txt`：核心词加审慎候选，供稳健性复算；
- `protected_theme_terms.txt`：即使高频也不应自动删除的主题词；
- `tang_poetry_stopword_audit.csv`：词频、文档频率、文档熵、诗人熵、决策和理由；
- `tang_poetry_stopword_metadata.json`：语料规模、阈值和词表数量；
- `pilot_topic_metrics.csv`：20 主题探索性 LDA 的词主题熵；
- `pilot_topic_metadata.json`：探索性 LDA 参数与版本；
- `stopword_report.md`：方法说明、代表性统计和使用建议；
- `build_stopwords.py`：可重复执行的生成程序。

## 生成

```powershell
python stopwords/pilot_topic_entropy.py
python stopwords/build_stopwords.py
```

第一条命令运行无停用词的探索性 LDA，只为估计各词在初始主题间的分布熵；第二条命令结合词频、文档频率、文档熵、诗人熵和主题熵生成分层词表，并逐行检查统一语料与 sikufenci 结果的 ID 对齐。统计门槛只生成候选，不会把“风、云、山、水、月、春、秋、愁、客、酒”等高频主题词自动视为停用词。

默认建议先使用核心表运行 LDA，再用扩展表复算一次。只有在主题关键词和文档主题分布基本稳定时，才应采用扩展表作为最终口径。
