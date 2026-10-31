# 1. Objective

恢复 K12/K14 的 40 个 train-only quantitative-validation 模型产物，并另建与既有 K15/K20 同口径的 full-corpus 盲审模型；本任务不执行语义盲审。

# 2. Why Topic-Word Arrays Cannot Recover Theta

`topic_word_unit` 只描述主题—词方向，缺少逐文档后验和 LDA components 的尺度信息，不能唯一反推出 P(topic|document)。本任务只从原模型对象或严格复现的模型执行 `transform`。

# 3. Existing Artifact Inventory

40/40 原 NPZ 与检查点存在；原有 model object 为 0，40 个 NPZ 均只有 `topic_word_unit`、`top_indices`，故 40 个 run 均需重新拟合。

# 4. Recovery Strategy

逐 run 复用冻结 C DTM、词表、train/test 行序、K、seed、prior 与原 sklearn 参数。先比较重建 topic-word；只有达到预注册 acceptance 才生成 theta、模型对象和代表诗。

# 5. Corpus / Vocabulary / Split Verification

DTM=(43015, 35063)，nnz=1464941，train=34412，test=8603；matrix、vocabulary、split、fit rows、test rows 哈希均与冻结记录一致。

# 6. Model Reproduction Validation

EXACT_REPRODUCTION=40；PERMUTATION_EQUIVALENT=0；NUMERICALLY_CLOSE=0；REPRODUCTION_MISMATCH=0

# 7. K12 Recovery

K12 共 20/20 个 run 登记，accepted=20/20，全部明确标注 TRAIN_ONLY。

# 8. K14 Recovery

K14 共 20/20 个 run 登记，accepted=20/20，全部明确标注 TRAIN_ONLY。

# 9. Document-Topic Matrices

accepted run 均保存 float64 `doc_topic_train`、`doc_topic_test`、`doc_topic_full` 及三套行索引；逐行 finite、非负、和为 1 的检查见 metadata 与 QA。

# 10. Representative Poems

每个 accepted Topic 均从真实 `doc_topic_full` 提取 Top20、Top50、Top100，以及同一作者每 Topic 最多 2 首的 author-balanced Top20；均含 poem_id、正文和 topic_probability。

# 11. Full-Corpus Blind-Audit Models

K12 representative seed=101；K14 representative seed=31415。两者均在 43,015 首冻结 C corpus 上以 ORIGINAL_PRIOR、max_iter=30 拟合，仅用于 Humanities Semantic Resolution Audit。

# 12. Model Comparability Audit

四模型直接可比=True。K12/K14 为本任务新建 full-corpus；K15/K20 只读复用 seed=2026 既有 full-corpus 模型。四者均为 ORIGINAL_PRIOR、max_iter=30、同 C DTM 与词表。

# 13. QA Results

PASS=18/18；FAIL=0/18。详细结果见 `data/qa_results.csv`。

# 14. Remaining Risks

joblib 模型对象依赖已登记的软件版本；full-corpus K12/K14 与 train-only quantitative runs 属不同 fit scope，不得把 full-corpus 模型用于 held-out perplexity 或改写既有量化结果。

# 15. Blind-Audit Readiness

最终状态：`BLIND_AUDIT_READY`。本任务在此停止，未开始 Four-Model Blind Humanities Semantic Resolution Audit，等待研究者确认。
