# K selection integrated judgment

- Lower-K Boundary Test：**LOWER_K_COMPETITIVE**
- Integrated Judgment：**REVIEW_K15_VS_K20**

## 十一项回答

1. K5 是否明显过粗：见自动 merge ratio；人工语义确认待填。
2. K10 是否明显过粗：见自动 merge ratio；人工语义确认待填。
3. K15 是否仍欠分：见 K15 merge ratio；人工语义确认待填。
4. K20 相对 K15 的语义增益：自动 alignment 可量化，解释性增益待人工审核。
5. K30+ fragmentation：本轮未新增人工审核，不能作强结论。
6. Perplexity 最偏好 K5。
7. K20附近 diminishing returns：不支持；各相邻区间的“改善”均为负值，即 K 增加时测试集困惑度持续上升；不存在可称为 K20 附近 diminishing returns 的正向收益。
8. NPMI 最支持 K15。
9. Stability 最支持 K5。
10. Human interpretability：低 K 尚待填写，当前只有既有 C_K20 分数。
11. 是否保留 C_K20：REVIEW_K15_VS_K20。

## 300字以内理由

K15 在自动指标中仍具竞争力，但仍有 53.3% 的自动合并提示，且低 K 人工审核尚缺；当前证据不足以推翻冻结的 C_K20，应先完成人工比较，再决定是否调整模型选择。
