# C Corpus K=2–30 穷举诊断

本目录保存同一冻结 80/20 split 上的 87 个 train-only LDA runs、逐 K 汇总、局部极值、NPMI 平台、候选 K 和六张诊断图。

复现命令：

```powershell
python src/run_exhaustive_k_scan.py
```

程序支持检查点续跑，不覆盖第一轮 full-corpus 模型、冻结 C_K20 或旧 NPMI。权威结论见 `QTS_Corpus_C_K2_30_Exhaustive_K_Scan_Report.md`。
