# B–C–D Finding-Level Robustness

只验证C_K20既有eligible findings；不重训、不重新选模型、不改写C结论。

## 复运行

```powershell
python src/run_bcd_finding_robustness.py --config config/BCD_robustness_config.yaml
```

工作簿由独立 artifact-tool builder 从这些权威CSV生成；分析与XLSX制作分离。builder及两个分析模块的实际路径与哈希保存在 `analysis_manifest.json` 的 implementation 中；用配置好的Node运行builder，并把此输出目录的绝对路径作为唯一参数。

固定 seed=20260902，作者分层bootstrap=5000次。所有配置阈值在 `BCD_robustness_config.yaml`；输入/输出哈希在 `analysis_manifest.json`。

## 审计入口

- `data/finding_semantic_mapping.csv`：语义映射核心审计表。
- `data/BCD_finding_effects.csv`：共同集合、正式映射、六规格主效应。
- `data/C_original_vs_common.csv`：原C与共同集合基准，不覆盖原结果。
- `data/split_single_best_sensitivity.csv`：split单目标敏感性，与正式bundle分离。
- `data/model_native_sensitivity.csv`：模型原生文档辅助敏感性；缺失历史标签不重新推断。
- `data/clr_sensitivity.csv`、`priority_author_sensitivity.csv`：按触发条件运行的诊断。
- `data/finding_robustness_ladder.csv`：发现层排序与下一阶段候选。
- `data/invariance_tests.csv`：20项任务契约检查。

## 数据表行数

| dataset | rows |
|---|---|
| BCD_common_document_index | 43014 |
| BCD_common_universe_qc | 51 |
| finding_semantic_mapping | 18 |
| BCD_finding_effects | 162 |
| C_original_vs_common | 54 |
| model_native_sensitivity | 108 |
| split_single_best_sensitivity | 72 |
| sample_direction_summary | 54 |
| aggregation_sensitivity | 81 |
| shape_similarity | 108 |
| priority_author_sensitivity | 32 |
| clr_sensitivity | 4 |
| BCD_finding_robustness_comparison | 9 |
| model_sensitivity_diagnostics | 19 |
| finding_robustness_ladder | 9 |
| BCD_priority_four_results | 24 |
| excluded_findings_status | 4 |
| invariance_tests | 20 |
| data_dictionary | 539 |

## 解释边界

作者时期不是诗作年代；跨模型Topic语义范围不完全相同；来源未记录正式human_review_status。本阶段只增加方法证据，不进行文学史阐释。
