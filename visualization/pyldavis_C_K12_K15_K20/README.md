# 《全唐诗》C_K12 / C_K15 / C_K20 pyLDAvis 比较诊断

先打开 `html/C_K12_K15_K20_pyLDAvis_comparison.html`，再分别进入 K12、K15、K20。三个模型页面均为单文件离线 HTML。

阅读 relevant terms 时，建议先看 λ=1.0，再看 λ≈0.6，最后看 λ≈0.2。每个模型的 PCoA 坐标独立，不可跨模型直接比较。圆面积仅表示 prevalence；主题邻近、重叠或分离需要结合代表诗和既有语义审核人工复核。

复现命令：

```powershell
python src/generate_pyldavis_three_model_comparison.py
```

脚本仅读取配置列明的冻结/既有工件，不执行 LDA fit、transform、重新分词或词表构建。生成后请运行 `python src/verify_pyldavis_three_model_outputs.py` 做静态 QA，并查看 `data/pyldavis_three_model_qa.csv`。
