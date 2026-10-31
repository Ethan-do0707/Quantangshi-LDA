# QTS Author-Period Topic Trend Structure

本目录只分析按作者所属时期划分的 C_K20 Topic probability。`author_period_primary` 不等于单篇作品的创作年代。

## 运行

```powershell
python src/run_author_period_trend_structure.py --config config/trend_structure_config.yaml
```

工作簿由 `src/build_author_period_trend_workbook.mjs` 从已验证 CSV 生成，不参与趋势计算。

## 固定分析规格

- STRICT_POEM / STRICT_AUTHOR
- CORE_POEM / CORE_AUTHOR
- EXPANDED_POEM / EXPANDED_AUTHOR

样本标签直接读取 Period Metadata Representativeness Audit，不在本阶段重新定义。Bootstrap 以 author 为 cluster，固定 2,000 次、seed=20260901。

## 显式阈值

```json
{
  "numeric_flat_relative_delta": 0.01,
  "numeric_absolute_epsilon": 1e-12,
  "near_flat_normalized_range": 0.1,
  "normalized_range_min_mean": 1e-09,
  "peak_tie_tolerance": 1e-12,
  "peak_trough_ratio_min_denominator": 1e-06,
  "high_peak_consistency_count": 5,
  "low_peak_consistency_count": 3,
  "high_transition_consistency_count": 5,
  "high_min_consistent_transitions": 2,
  "low_global_direction_conflict_count": 2,
  "author_concentration_moderate_top1": 0.15,
  "author_concentration_moderate_top5": 0.4,
  "author_concentration_moderate_hhi": 0.08,
  "author_concentration_high_top1": 0.25,
  "author_concentration_high_top5": 0.6,
  "author_concentration_high_hhi": 0.15
}
```

## 核心表

- `data/period_counts.csv`：12 行。
- `data/period_topic_means.csv`：480 行。
- `data/author_topic_profiles.csv`：866 行。
- `data/period_topic_bootstrap_ci.csv`：480 行。
- `data/topic_trend_structure.csv`：120 行。
- `data/topic_trend_robustness.csv`：20 行。
- `data/poem_vs_author_trend_comparison.csv`：60 行。
- `data/period_topic_author_concentration.csv`：240 行。
- `data/clear_topic_trend_structure.csv`：9 行。
- `data/trend_candidate_topics.csv`：9 行。
- `data/sensitivity_flags.csv`：20 行。
- `data/invariance_tests.csv`：15 行。
- `data/data_dictionary.csv`：275 行。

六幅 PNG 位于 `figures/`，正式报告与工作簿位于本目录。`data/invariance_tests.csv` 必须全部为 PASS。
