# QTS C_K12 vs C_K20 Historical Findings Comparison

## 1. 比较前置条件

K12 统计输出于 `2026-09-10T07:53:49.059131+08:00` 冻结，状态为 `BLIND_TO_OLD_TREND_DIRECTION_COMPLETE`；冻结清单 SHA-256 为 `25273e82cf87bec7a8b4cbb2214c41c4b46b545a86d96c69a0c2d1d0eed8f6f4`。本报告仅在逐项验证冻结清单之后读取旧 K20 结果，因此 K20 趋势没有参与 K12 结果生成。

旧结果保留为 `C_K20_HISTORY_V1_REFERENCE`；新结果为 `C_K12_HISTORY_V2_PRIMARY_CANDIDATE`。旧 K20 核心 finding 文件哈希为 `1b8fda5403f5c1b530ec0f3a19830f2a882a4f879ae3f05c25ff85a3dcbb45b7`，未覆盖。

## 2. 比较方法

Topic ID 不作跨模型等同。语义对应来自既有四模型盲审的 K12→K20 `AI_PROVISIONAL` semantic family，并用既有 K12→K15→K20 topic-word cosine 最优路径作量化旁证。两套对齐都只是研究者判读线索；只有在此之后，才把冻结 K12 的同一 planned contrast（统一为 B−A）与旧 K20 finding 对读。

## 3. 逐项结果

| K20 finding | K12 对应 | K12 raw 方向 | K12 验证状态 | 比较结论 |
| --- | --- | --- | --- | --- |
| topic_03 E-H | topic_05;topic_08 | NEGATIVE;POSITIVE | STRONG;UNSUPPORTED | REPRODUCED_IN_REFINED_K12_COMPONENT |
| topic_03 E-L | topic_05;topic_08 | NEGATIVE;POEM_AUTHOR_CONFLICT | STRONG;CONFLICTING | REPRODUCED_IN_REFINED_K12_COMPONENT |
| topic_03 M-L | topic_05;topic_08 | NEGATIVE;NEGATIVE | STRONG;CONFLICTING | REPRODUCED_IN_REFINED_K12_COMPONENT |
| topic_06 E-L | topic_02 | POSITIVE | EXPLORATORY_BROAD | REPRODUCED_IN_BROADER_K12_EXPLORATORY |
| topic_07 E-L | topic_06 | POEM_AUTHOR_CONFLICT | CONFLICTING | MERGED_IN_K12_NOT_REPRODUCED |
| topic_08 E-L | topic_09 | POSITIVE | STRONG | REPRODUCED_IN_K12 |
| topic_11 M-L | topic_06 | POSITIVE | SUPPORTED | MERGED_IN_K12_REPRODUCED |
| topic_16 E-L | topic_11 | NEGATIVE | TENTATIVE | WEAKENED_IN_K12 |
| topic_19 E-H | topic_01;topic_09 | POEM_AUTHOR_CONFLICT;POSITIVE | EXPLORATORY_BROAD;CONFLICTING | INCOMPARABLE_TOPIC_REDEFINED |

- K20 topic_03 E-H：旧礼乐/应制复合主题在 K12 中细分；T05 的同一比较方向一致且经 LOAO 保留，T08 不支持同一结论。
- K20 topic_03 E-L：旧礼乐/应制复合主题在 K12 中细分；T05 的同一比较方向一致且经 LOAO 保留，T08 出现 POEM/AUTHOR 冲突。
- K20 topic_03 M-L：旧礼乐/应制复合主题在 K12 中细分；T05 的同一比较方向一致且经 LOAO 保留，T08 未形成候选。
- K20 topic_06 E-L：K12 T02 合并老病、诗酒与士人自况；POEM/AUTHOR 的 E-L 均为正，但该主题为 BROAD，未进入 primary inference。
- K20 topic_07 E-L：K20 的相思离愁与女性闺情在 K12 T06 合并；E-L 在 K12 的 POEM/AUTHOR 方向冲突，旧 finding 未保留。
- K20 topic_08 E-L：山寺/僧侣主题的直接语义对应最强；同一 E-L 比较为正且经 LOAO 保留。
- K20 topic_11 M-L：K20 的闺情主题与相思离愁在 K12 T06 合并；M-L 同向且经 LOAO 保留，但语义范围较宽。
- K20 topic_16 E-L：边塞主题为直接对应；E-L 两种 estimand 同为负，但 K12 仅为 TENTATIVE，未进入 LOAO 保留 finding。
- K20 topic_19 E-H：旧闲居山林主题分散到 K12 T01 与 T09；T01 出现聚合方向差异，T09 又有 CLR 冲突，无法归于单一等价 K12 finding。

## 4. 分类摘要

- 正式层面复现：5 条；K20 topic_03/E-H → K12 topic_05;topic_08; K20 topic_03/E-L → K12 topic_05;topic_08; K20 topic_03/M-L → K12 topic_05;topic_08; K20 topic_08/E-L → K12 topic_09; K20 topic_11/M-L → K12 topic_06。
- 更宽 K12 主题中方向复现但仅属探索性：1 条；K20 topic_06/E-L → K12 topic_02。
- 涉及 K12 合并主题：3 条；其中相同 contrast 是否保留须逐条区分。
- 削弱或未复现：2 条；K20 topic_07/E-L → K12 topic_06; K20 topic_16/E-L → K12 topic_11。
- 因题义重定义而不可作单一对应：1 条；K20 topic_19/E-H → K12 topic_01;topic_09。

## 5. 解释边界

这里的“复现”只指：在语义对齐后，同一作者时期比较的冻结 K12 raw probability 方向与旧 K20 相容，并且（正式复现时）K12 finding 经自身统计与作者影响流程保留。它不表示 Topic 完全同义，也不构成文学史事实确认。author period 表示作者所属时期，并非单首诗的创作年代。
