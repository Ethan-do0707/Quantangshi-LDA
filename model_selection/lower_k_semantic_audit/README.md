# QTS Lower-K Humanities Semantic Audit

本目录包含 C_K5、C_K10、C_K15 共 30 个 Topic 的 AI-assisted provisional 人文学语义审核。每题均核对 Top30 words、Top20 representative poems、author-balanced 诗例和 C_K20 one-to-many alignment。

## 状态与边界

- 全部 AI 判断均标为 `AI_REVIEW`、`PROVISIONAL_REVIEW`、`NEEDS_RESEARCHER_CONFIRMATION`。
- 没有重新训练 LDA，没有修改任何冻结矩阵、词表、K20 taxonomy 或历史分析。
- 本轮没有运行 K16、K17、K18、K19。

## 主要文件

- `review_cards/K5|K10|K15/`：30 份逐题证据卡。
- `data/lower_k_topic_ai_review.csv`：逐题 AI 暂定评分和证据路径。
- `data/lower_k_humanities_summary.csv`：自动指标与人文学语义指标并列总表。
- `data/K15_vs_K20_semantic_resolution_audit.csv`：K15 专项比较。
- `data/K15_core_semantic_coverage.csv`：九个 C_K20 CLEAR fields 的 K15 承载情况。
- `data/researcher_review_priority.csv`：研究者优先队列。
- `data/invariance_tests.csv`：不可变性与完整性测试。
- `QTS_Lower_K_Researcher_Review.xlsx`：保留 AI 建议、Researcher 字段全部空白的复核工作簿。
- `QTS_Lower_K_Humanities_Semantic_Audit_Report.md`：13 节审计报告。

## 判读限制

Cosine similarity 仅作关系提示，Topic 类型、合并质量和欠分风险均以 Top words 与代表诗的实际语义为主。K20 的既有 3.75 分与本轮 AI 分来自不同审核体系，比较只能作为暂定诊断。
