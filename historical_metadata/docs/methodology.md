# 方法说明

本管线只增强作者历史元数据，不训练 LDA、不修改 Topic probability、不修改 taxonomy/alignment，也不推断单篇作品创作年代。

## 作者身份

作者原值先做 NFKC、首尾空格和控制字符清理。简繁字符表只生成独立候选键，绝不覆盖 `author_raw` 或 `author_normalized`；尾随数字作者不删除数字，也不进入自动匹配。正名与 CBDB 别名候选全部进入审计表。同名多位可行候选不自动选第一条。

## CBDB

数据来自固定 SQLite 快照 `cbdb_20260829.sqlite3`，检索时间为 2026-09-01T14:59:05+08:00。`birthyear`、`deathyear`、floruit 和 index year 保持不同语义。人物级来源另存，除非 SQLite 提供字段级外键，否则不把人物级来源冒充某年份字段的直接来源。

## 活动期与分期

`author_fl_start/end` 只承载 CBDB 明示 floruit。明确有年的事件、入仕/考试和任官记录另存 `observed_activity_start/end`。index year 只作低置信度代理。`QTS_PERIOD_V1` 是透明的操作性年代分段，不宣称文学史边界唯一客观。

传统文学史分期只接受 `traditional_period_mapping.csv` 中带来源的人工记录；空映射保持 NA。

## 分析限制

诗级新增字段名称均明确带 `author_`。未来可以在充分处理低置信、未知、跨期和边界人物后做 author-period analysis；不能把它称为 poem-level chronological analysis。
