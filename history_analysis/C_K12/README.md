# C_K12 作者时期历史分析

版本：`C_K12_HISTORY_V2_PRIMARY_CANDIDATE`。

本目录从 FULL_CORPUS_FIT 的 43,015×12 Document×Topic 矩阵独立重算作者时期趋势、代表性、效应验证与作者影响。K12 正式统计先在 `BLIND_TO_OLD_TREND_DIRECTION` 条件下完成并写入 `SHA256_MANIFEST.txt`，之后才建立 C_K12 与旧 C_K20 的事后语义比较。

`author_period_primary` 表示作者所属时期，不表示单篇作品创作年代。统计候选不等于文学史确认。

- `00_input_audit`：theta、document_index、taxonomy、样本数、最终 20 项 QA 与配置。
- `01_trend_structure`：三样本、两 estimand、四时期的趋势结构。
- `02_representativeness`：已知/未知作者时期的 K12 Topic SMD 与风险。
- `03_statistical_validation`：CLEAR Topics 的 CR1/HC3、BH-FDR、效应量、样本与 CLR 敏感性。
- `04_author_influence`：仅 STRONG/SUPPORTED 候选的 LOAO 与 Top contributor 排除。
- `05_figures`：八幅规定 PNG。
- `06_final`：研究者解释队列、18 节最终报告、冻结后 K20 比较及逐条比较数据。

进入 LOAO 的候选数：8；最终保留：7。最终交付状态：`C_K12_HISTORY_ANALYSIS_COMPLETE`。
