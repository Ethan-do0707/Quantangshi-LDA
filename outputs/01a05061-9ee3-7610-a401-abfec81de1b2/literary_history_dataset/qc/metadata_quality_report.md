# Metadata Quality Report

母数据集包含 43,015 首 C_K20 有效诗歌。Document×Topic 每一行均通过冻结 index 的 poem_id 唯一连接；另有 15 条全语料 metadata 因 C 清洗后为空而不进入模型矩阵，已列入 unmatched_documents.csv。

## 权威来源

- poem_id、poet_id、author、title、text：`segmentation/data/poems.jsonl`。
- cleaned tokens：`output/tf_df_sensitivity/C/lda_corpus.csv`。
- matrix row：冻结的 `C_K20/document_index.csv`。

## 字段覆盖率

| 字段 | 非缺失 | 缺失 | 覆盖率 | 唯一值 | 说明 |
|---|---:|---:|---:|---:|---|
| poem_id | 43015 | 0 | 100.00% | 43015 |  |
| poet_id | 43015 | 0 | 100.00% | 2528 |  |
| author | 43015 | 0 | 100.00% | 2528 | unique_authors=2528;anonymous_or_unattributed=980 |
| title | 43015 | 0 | 100.00% | 37208 |  |
| text | 43015 | 0 | 100.00% | 42446 |  |
| cleaned_text | 43015 | 0 | 100.00% | 42366 |  |
| volume | 0 | 43015 | 0.00% | 0 | 当前 SQL/JSONL 无《全唐诗》卷次字段 |
| genre | 0 | 43015 | 0.00% | 0 | 当前权威 metadata 无文体字段 |
| form | 0 | 43015 | 0.00% | 0 | 当前权威 metadata 无文体字段 |
| poetic_form | 0 | 43015 | 0.00% | 0 | 当前权威 metadata 无文体字段 |
| date | 0 | 43015 | 0.00% | 0 | 无诗歌创作时间；SQL created_at/updated_at 为技术时间，禁止采用 |
| year | 0 | 43015 | 0.00% | 0 | 无诗歌创作时间；SQL created_at/updated_at 为技术时间，禁止采用 |
| period | 0 | 43015 | 0.00% | 0 | period metadata 尚需构建或校验；本阶段不作初盛中晚唐推断 |
| author_period | 0 | 43015 | 0.00% | 0 |  |
| dating_method | 0 | 43015 | 0.00% | 0 |  |
| place | 0 | 43015 | 0.00% | 0 | 当前权威 metadata 无空间字段 |
| region | 0 | 43015 | 0.00% | 0 | 当前权威 metadata 无空间字段 |
| location | 0 | 43015 | 0.00% | 0 | 当前权威 metadata 无空间字段 |

## 可用性结论

- poem_id、author、title、text 与清洗词元可用于作者聚合和文本回查。
- 没有可靠 date/year/period，**尚不具备时间或分期分析条件**。
- 没有 genre/form/poetic_form，**尚不具备文体比较条件**。
- 没有 place/region/location，**尚不具备空间分析条件**。
- 作者生卒年未进入本表，也绝不能被自动当作诗歌创作年份。
