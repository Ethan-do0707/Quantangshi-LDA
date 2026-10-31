# QTS Quantitative Candidate Validation

本目录保存 C corpus 的 K11–15 与 K20、十个固定 seed、两套 prior regime 共 120 个 train-only LDA fits，以及 NPMI、held-out perplexity、Topic Diversity、模型内冗余、Topic prevalence、跨 seed Hungarian 稳定性、成对差异、prior sensitivity、Pareto 辅助筛选和最终量化 shortlist。

复现或续跑：

```powershell
python src/run_quantitative_candidate_validation.py
```

只重新汇总已完成检查点：

```powershell
python src/run_quantitative_candidate_validation.py --summarize-only
```

Fixed-prior 只是一项 sensitivity；所有产物均不覆盖冻结 C_K20。权威解释见 `QTS_Quantitative_K_Candidate_Validation_Report.md`。
