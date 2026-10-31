# K12/K15/K20 Researcher-Led Humanities Adjudication

本目录保存三模型研究者主导的人文学裁决材料。当前只完成 PREPARE；47 个 Topic 的 14 项 `Researcher_*` 字段全部为空，不得解盲或生成最终 K 决策。自动相似度只生成后台候选，不构成 split、merge 或 fragmentation 裁决。

## 第一阶段使用顺序

1. 打开 `QTS_K12_K15_K20_Human_Review.xlsx`；初始仅显示 `Instructions`、`Blind_Topic_Review`、`Blind_Poem_Evidence`。
2. 研究者逐一完成 HR001–HR047。每题含 Top30 词、Top20 原始代表诗和每位作者最多 2 首的 author-balanced Top10。
3. 运行 `python src/validate_human_review.py`。未完成时返回 `TOPIC_REVIEW_INCOMPLETE`，不会解盲。
4. 47/47 完成后运行 `python src/freeze_human_review.py`，生成冻结 CSV 与 SHA256。
5. 仅在冻结成功后运行 `python src/summarize_human_review.py`，再进入模型级、split/merge、碎片化、语义家族和最终裁决阶段。

## 已准备的后台候选

- K12→K15：20 个 split/merge 事件（8 split、12 merge）。
- K15→K20：16 个 split/merge 事件（8 split、8 merge）。
- K20：15 个碎片化候选对。

候选仅由 topic-word cosine、Top20 词重叠和代表诗证据生成；所有研究者判断列均为空。旧 AI 参考仅保存在 `_private` 及工作簿的 `Hidden_AI_Reference` 非常隐藏页，第一阶段不可见，也不参与候选生成。
