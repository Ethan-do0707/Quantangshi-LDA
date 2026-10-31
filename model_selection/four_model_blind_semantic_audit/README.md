# Four-Model Blind Semantic Audit

该目录保存四个冻结 LDA 模型的 AI-assisted blind semantic audit。61 个 Topic 以全局随机卡序独立审核；真实映射仅位于 `_private_unblind/`，盲审 CSV、图、报告与研究者工作簿均不显示真实 K 或量化指标。

## 执行与冻结顺序

1. `src/run_four_model_blind_semantic_audit.py --phase prepare`：只读冻结模型，生成 Top30、Top20 与作者平衡证据卡。
2. `src/finalize_four_model_blind_semantic_audit.py --phase blind`：汇总 61 项独立评分、模型内结构、跨模型 family 与 split/merge 裁决。
3. `--phase freeze`：生成 `data/blind_semantic_judgment_frozen.csv`；SHA256 记录在 `audit/blind_semantic_judgment_frozen.sha256`。
4. `--phase unblind`：先校验冻结哈希，再读取私有映射与既有量化验证；不回写任何盲审字段。
5. `src/build_four_model_blind_review_workbook.mjs`：从冻结盲审数据生成研究者复核工作簿，所有 `Researcher_*` 单元格留空。
6. `python tests/test_four_model_blind_semantic_audit.py`：执行 18 项专项 QA 并写入 `audit/qa_results.csv`。

## 当前结论边界

- 盲审排序：`MODEL_A > MODEL_C > MODEL_D > MODEL_B`，无并列，信心为 `MEDIUM`。
- 解盲综合状态：`PREFER_K12`。
- 所有语义评分均为 `AI_PROVISIONAL`，研究者尚未完成复核；优先检查 `data/blind_researcher_priority_queue.csv` 中的 Priority 1 与关键 split/merge。
- 本任务没有重训模型、修改冻结输入、重新运行历史下游分析或改写旧 taxonomy。
