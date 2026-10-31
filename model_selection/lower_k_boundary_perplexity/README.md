# QTS LDA Lower-K Boundary Test + Held-out Perplexity Diagnostic

本目录只包含补充诊断产物。原 C_K20、B_K20、D_K20、taxonomy、历史时期分析、效应验证、LOAO 与 B/C/D 稳健性均未改动。

- `lower_k_models/`：C_K5/C_K10/C_K15 三 seed 全语料诊断模型及完整矩阵。
- `heldout_models/`：69 个 held-out run 的可恢复指标检查点；未把测试集传入 fit。
- `splits/`：A/B/C/D 各自固定 80/20 poem_id 切分。
- `data/`：逐 run、汇总、alignment、边界表及 17 项检查。
- `review/`：审核卡、空白人工评分工作簿及其 CSV 数据源。
- `figures/`：八张诊断图。

当前自动边界状态：`LOWER_K_COMPETITIVE`；整合判断：`REVIEW_K15_VS_K20`。低 K 人工分数未由程序代填。

复现命令：

```powershell
python src/run_lower_k_boundary_perplexity.py --phase all
```
