# Author-Period Effect Size & Statistical Validation

本目录保存 C_K20 作者时期主题效应量、稳健推断、样本敏感性、CLR 敏感性与综合验证状态。

## Reproduce

```powershell
python src/run_author_period_effect_validation.py --config config/effect_validation_config.yaml
```

## Fixed design

- Bootstrap iterations: 5000
- Random seed: 20260901
- Formal inference: 9 CLEAR Topics under CORE_POEM and CORE_AUTHOR only
- Sensitivity: STRICT/CORE/EXPANDED × POEM/AUTHOR without extra p-value families
- Primary contrasts: E-H, H-M, M-L, E-L
- Secondary contrasts: E-M, H-L

## Data tables

- `data/clear_topic_period_omnibus.csv`: 18 rows
- `data/clear_topic_primary_contrasts.csv`: 36 rows
- `data/clear_topic_all_pairwise.csv`: 108 rows
- `data/clear_topic_effect_sensitivity_matrix.csv`: 36 rows
- `data/raw_vs_clr_direction_agreement.csv`: 72 rows
- `data/all_topic_effect_sizes.csv`: 720 rows
- `data/statistically_supported_candidate_trends.csv`: 13 rows
- `data/author_influence_check_queue.csv`: 36 rows
- `data/clear_topics_validation_summary.csv`: 9 rows
- `data/posthoc_peak_trough_effects.csv`: 18 rows
- `data/invariance_tests.csv`: 15 rows
- `data/data_dictionary.csv`: 269 rows
