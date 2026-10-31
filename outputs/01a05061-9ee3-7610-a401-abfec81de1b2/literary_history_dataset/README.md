# 《全唐诗》LDA 文学史分析母数据集

本目录冻结 B_K20、C_K20、D_K20，并以 C_K20 作为唯一主模型。所有诗歌级统计必须优先使用完整 20 维 `theta_dk`；dominant topic 只用于浏览，不能把一首诗硬化为单一主题。

## 功能分区

- `freeze/`：模型复制、SHA-256、manifest 与完整性检查。
- `taxonomy/`：C_K20 正式 taxonomy 和下游唯一标签主表 `topic_metadata`。
- `alignment/`：B–C–D 共享词表相似度、多对多对齐和20份审核卡。
- `matrices/`：原始 Document×Topic、诗到主题摘要和每 Topic Top500 反向索引。
- `master/`：一行一首有效诗的文学史母数据集。
- `qc/`：输入盘点、document 对齐、metadata 覆盖率和15项最终测试。

## 禁止事项

不得重新训练、重新 `fit_transform`、改变 TF/DF 或 K、修改人工评分、平均 B/C/D 概率、按 Topic ID 直接跨模型对应、用作者加篇名代替 poem_id、补造年代/分期/文体/空间字段，或在本阶段生成文学史趋势结论。
