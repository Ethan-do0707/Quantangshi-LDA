# 《全唐诗》作者历史年代元数据增强（第一期）

本目录保存作者身份标准化、CBDB 候选审计、生卒年、明示 floruit、有年事件、作者分期、字段级来源、人工审核队列，以及与 C_K20 文学史母表的非覆盖式连接结果。

运行：

```powershell
python src/build_historical_metadata.py --config config/historical_metadata_config.yaml
```

本阶段不训练或修改 LDA，不生成 `poem_year`/`poem_period`，也不执行 Topic × Period 分析。诗级表中的 `author_period_*` 只是继承的作者元数据。

## 当前交付状态

- 作者主表、CBDB 候选与来源审计、作者分期、人工审核队列、数据字典、覆盖率报告、17 项 QC，以及 43,015 × 68 的 enriched master CSV 均已生成。
- `QTS_author_historical_metadata.xlsx`、`author_metadata_human_review.xlsx` 与 `historical_metadata_data_dictionary.xlsx` 已完成逐页视觉检查和独立 ZIP/XML 结构核验。
- `C_K20_literary_history_master_with_author_history.xlsx` 已在用户单项授权后，以 XlsxWriter 3.2.9、`constant_memory=True` 从权威 CSV 逐行序列化。指定的 `artifact-tool` 对该 43,015 × 68 单表曾在 8 GB Node 堆上限下内存溢出；本例外及双端 SHA-256 已写入正式报告和版本记录。
- 大型工作簿为 `Data!A1:BP43016`，含 43,015 数据行与 68 列；ZIP/XML、workbook relationship、全量 `poem_id` 顺序、首中末行及固定随机 100 行全字段比较均通过。

工作簿结构复核：

```powershell
python src/verify_historical_metadata_workbooks.py
```
