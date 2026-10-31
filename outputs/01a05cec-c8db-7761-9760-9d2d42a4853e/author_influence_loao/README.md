# QTS Author Influence / Leave-One-Author-Out

本目录保存 C_K20 CORE 作者贡献、LOAO、Top-contributor exclusion、聚合冲突分解与作者稳健性分类。

## Reproduce

```powershell
python src/run_author_influence_loao.py --config config/author_influence_config.yaml
```

## Fixed scope

- Primary: CORE_POEM and CORE_AUTHOR
- 主分类：LOAO + GLOBAL_TOP Top1/3/5；PER_PERIOD_TOP 使用相同阈值独立报告敏感性状态
- Sensitivity: selected STRICT / EXPANDED pairs only
- Bootstrap seed: 20260902
- No new p-value search, no B_K20/D_K20 computation, no literary-history interpretation

## Data tables

- `data/author_period_topic_contributions.csv`: 1,863 rows
- `data/author_concentration_summary.csv`: 36 rows
- `data/loao_effects.csv`: 7,452 rows
- `data/loao_summary.csv`: 72 rows
- `data/author_influence_score.csv`: 7,452 rows
- `data/top_influential_authors.csv`: 720 rows
- `data/clear_topic_top_influential_authors.csv`: 720 rows
- `data/top_contributor_exclusion_effects.csv`: 576 rows
- `data/top_exclusion_bootstrap_ci.csv`: 156 rows
- `data/sample_influence_sensitivity.csv`: 882 rows
- `data/aggregation_conflict_summary.csv`: 7 rows
- `data/aggregation_conflict_decomposition.csv`: 786 rows
- `data/clear_topic_author_influence_summary.csv`: 36 rows
- `data/author_validated_candidate_trends.csv`: 13 rows
- `data/invariance_tests.csv`: 15 rows
- `data/data_dictionary.csv`: 312 rows
