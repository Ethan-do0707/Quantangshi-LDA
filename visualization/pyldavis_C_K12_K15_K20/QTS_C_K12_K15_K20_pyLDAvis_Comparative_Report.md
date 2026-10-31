# 1. Objective

本任务比较 C_K12、C_K15 与冻结 C_K20 的 pyLDAvis 视觉诊断，用于检查主题占比、模型内部邻近/重叠和不同 λ 下的 relevant terms。它不选择 K，不生成综合评分，也不改写既有文学判断。

# 2. Model Comparability

三模型均为 `FULL_CORPUS`，共用冻结 C corpus（TF≥3、DF≥3）、43,015 行 DTM、35,063 词项及相同文档顺序。Batch LDA 和 max_iter=30 约定一致。审计警告：`REPRESENTATIVE_SEED_AND_SELECTION_METHOD_DIFFER；有效默认先验为 1/K`。K12 seed=101；K15/K20 seed=2026。K20 读取冻结 reference。

# 3. What pyLDAvis Shows

左侧 bubble 面积表示词次加权 prevalence；圆间位置是本模型 topic-term 距离的二维 PCoA 投影。右侧显示点击 Topic 的 relevant terms。λ=1.0 偏重主题内概率，λ≈0.6 作为主要观察点，λ≈0.2 偏重相对区分度；0.6 不是“最优 λ”。三页均设 `sort_topics=False`。

# 4. C_K12 Visualization

最大 prevalence T02=12.522%；最小 T04=2.825%；中位数 8.339%。mean/max nearest cosine=0.6091/0.6667。这些是概率与相似度事实；圆的视觉重叠需由研究者打开 HTML 后复核。

# 5. C_K15 Visualization

最大 prevalence T06=11.001%；最小 T09=2.812%；中位数 5.341%。mean/max nearest cosine=0.5932/0.6709。这些事实不构成对 K15 的优劣判断。

# 6. C_K20 Visualization

最大 prevalence T10=8.569%；最小 T14=1.928%；中位数 4.752%。mean/max nearest cosine=0.5878/0.6710。该页使用冻结 C_K20 工件，没有重新训练或重新推断 theta。

# 7. Topic Prevalence Comparison

K12/K15/K20 的最大 Topic mass 分别为 12.522%、11.001%、8.569%。小于 1% 的 Topic 数为 0、0、0；小于 2% 为 0、0、1。Prevalence 不是文学重要性。

# 8. High-Dimensional Topic Separation

模型内所有 Topic pair 的 mean/median/max cosine：K12=0.4566/0.4627/0.6667；K15=0.4371/0.4292/0.6709；K20=0.3981/0.4018/0.6710。Top20/Top30 Jaccard 已逐 pair 输出。

# 9. Nearest-Neighbor Structure

mean nearest-neighbor cosine 按 K12→K15→K20 为 0.6091→0.5932→0.5878。最高 pair 分别为 T03↔T11、T07↔T10、T08↔T19。

# 10. K12 → K15 Resolution Change

本报告引用既有 blind semantic audit 的 K12↔K15 alignment，并保留其 relation_type 和 AI_PROVISIONAL 状态。二维坐标各自独立，因此没有用 PC 坐标计算跨模型距离。研究者应结合 alignment、cosine、λ=0.6 terms 与代表诗填写 checklist。

# 11. K15 → K20 Resolution Change

K15↔K20 alignment 同样直接引用既有审核。已有 `MEANINGFUL_SPLIT`、`MINOR_REFINEMENT`、`FRAGMENTATION` 或其他事件时，pyLDAvis 仅提供辅助视觉证据，不覆盖原判断。

# 12. Potential Fragmentation Signals

诊断结论为 **NO**。量化依据：mean nearest cosine 为 0.6091→0.5932→0.5878；cos≥0.85 pair 数为 0→0→0。这只记为 `POTENTIAL_FRAGMENTATION_SIGNAL`，不等同文学意义上的碎片化。

# 13. Relation to Humanities Semantic Audit

alignment 来源为现有四模型盲审数据；K12/K15 标签保持 `AI_PROVISIONAL`，K20 标签来自已冻结的人工 taxonomy。事件在输出 alignment 中按原状态保留。带判断的 pair 计数为：{"FRAGMENTATION": 15, "PROBLEMATIC_MERGE": 2, "MINOR_REFINEMENT": 2, "MEANINGFUL_SPLIT": 2}。

# 14. Visual–Semantic Agreements

当前自动流程没有替研究者判断二维 bubble 是否构成视觉 agreement。高维 cosine、Top-word overlap 和现有语义判断已并列输出，供人工确认。

# 15. Visual–Semantic Disagreements

当前没有把二维距离自动解释为 `VISUAL_SEMANTIC_DISAGREEMENT`。如研究者观察到 FRAGMENTATION pair 的 bubble 较远，或 MEANINGFUL_SPLIT pair 高度邻近，应在 checklist 记录，且保留原语义判断。

# 16. Limitations

1. pyLDAvis 二维位置是高维距离投影。
2. 不同 K 的 PC 坐标不能直接跨模型比较。
3. bubble 大小只是 prevalence，不是质量或文学重要性。
4. Topic 接近不等于文学意义重复。
5. Topic 分离不等于文学意义必然不同。
6. relevant terms 不能替代代表诗阅读。
7. pyLDAvis 不能单独决定最优 K。

# 17. Researcher Review Checklist

研究者核查表位于 `researcher/K12_K15_K20_pyldavis_researcher_checklist.xlsx`。视觉分离、重叠、词项区分度和 notes 字段保持空白，等待研究者填写。运行状态：`PYLDAVIS_THREE_MODEL_READY`。
