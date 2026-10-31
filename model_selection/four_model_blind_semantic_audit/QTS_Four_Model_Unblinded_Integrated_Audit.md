# 1 Blind Mapping

MODEL_A → K12；MODEL_B → K20；MODEL_C → K15；MODEL_D → K14。映射仅在盲审 CSV 完成冻结并通过 SHA256 校验后读取。

# 2 Blind Results Before Unblinding

盲态排序为 MODEL_A > MODEL_C > MODEL_D > MODEL_B，无并列，信心 MEDIUM。解盲后没有修改任何 Topic 类型、评分、模型位次或 split/merge 判断。

# 3 K12 Semantic Resolution

K12 对应 MODEL_A。CLEAR+BROAD 为 10/12，文学可解释性最高，本轮未识别独立 Topic 内欠分；主要问题是两个 NOISE 残余桶及部分 BROAD 边界，研究者仍需确认紧凑结构是否掩盖低频语义。

# 4 K14 Semantic Resolution

K14 对应 MODEL_D。七个 CLEAR 构成可读核心，但三个 MIXED Topic 显示局部欠分，另有一对残余 Topic 被判碎片化，因此未在盲态成为最平衡模型。

# 5 K15 Semantic Resolution

K15 对应 MODEL_C。语义特异性较高且只有一个 NOISE；两个有意义拆分带来解释收益，但三个 MIXED/欠分 Topic 说明若干文学场仍被压合，量化先验敏感性也需保留。

# 6 K20 Semantic Resolution

K20 对应 MODEL_B。新增了春日物候等可解释区分，但 MIXED、NOISE 与欠分负担最高；山寺/林泉重复与残余槽位扩张显示额外复杂度主要没有转化为稳定文学边界。

# 7 Cross-K Split / Merge Analysis

由低分辨率向高分辨率的七个 SPLIT 裁决为：MEANINGFUL_SPLIT=3，MINOR_REFINEMENT=1，FRAGMENTATION=3，UNCERTAIN=0；另有两个 MERGE 事件分别为 MEANINGFUL_AGGREGATION 与 PROBLEMATIC_MERGE。

# 8 Meaningful Resolution Gains

K14→K15 的闺情季节结构与政治伦理/社会悲愤区分构成主要有意义增益；K15→K20 仅有一项明确有意义拆分与一项轻微细化，同时出现两项碎片化。

# 9 Under-splitting vs Fragmentation

K12 的观测风险最低；K14、K15 有局部欠分，K14 另有残余重复；K20 同时具有最高欠分比例与明显残余/宗教题材碎片化。该结论限于本轮代表 seed 与 61 个 Topic 的 AI 初审。

# 10 Quantitative Evidence

量化与语义证据保持为分列矩阵，不生成加权总分。K12 为原先验 NPMI 与跨 seed 稳定性领先者，K15 为固定先验 NPMI 领先者；K20 的预测拟合与匹配稳定性在四候选中较弱。

# 11 Prior Robustness

K14 与 K20 为 PRIOR_ROBUST；K12 与 K15 为 PRIOR_SENSITIVE。先验稳健性是独立证据，不反向更改盲审语义结论。

# 12 Seed Robustness

K20 带有 SEED_SENSITIVE 警示；K12、K14、K15 在既有候选验证中无 seed warning。语义审核使用预先固定的代表 seed，而非单次最高 NPMI seed。

# 13 Quantitative–Semantic Trade-off

| K | blind_model_id | blind_rank | quantitative_coherence | predictive_fit | prior_robustness | seed_robustness | semantic_coherence_clear_broad_ratio | mean_literary_interpretability | under_split_ratio | fragmentation_topic_ratio | final_K_status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | MODEL_A | 1 | VERY_STRONG | BEST_AMONG_CANDIDATES | PRIOR_SENSITIVE | NO_SEED_WARNING | 0.833 | 4.417 | 0.000 | 0.000 | STRONG_FINAL_CANDIDATE|SEMANTICALLY_STRONG|QUANTITATIVELY_STRONG|PRIOR_SENSITIVE_CANDIDATE |
| 14 | MODEL_D | 3 | STRONG | SECOND | PRIOR_ROBUST | NO_SEED_WARNING | 0.643 | 3.857 | 0.214 | 0.143 | BALANCED_CANDIDATE |
| 15 | MODEL_C | 2 | VERY_STRONG | THIRD | PRIOR_SENSITIVE | NO_SEED_WARNING | 0.733 | 4.333 | 0.200 | 0.000 | SEMANTICALLY_STRONG|QUANTITATIVELY_STRONG|PRIOR_SENSITIVE_CANDIDATE |
| 20 | MODEL_B | 4 | WEAKER | WEAKEST_AMONG_CANDIDATES | PRIOR_ROBUST | SEED_SENSITIVE | 0.650 | 3.950 | 0.200 | 0.100 | OVER_FRAGMENTED|SEED_SENSITIVE_CANDIDATE|REFERENCE_ONLY |

以上各轴独立呈现；没有通过任意权重合成为单一分数。

# 14 Final Candidate Judgment

最终状态：PREFER_K12。其盲态语义平衡位居第一，同时具备强量化一致性与 seed 稳定性；代价是 prior sensitivity、两个残余桶及较紧凑分辨率。K15 是最接近的语义备选，但局部欠分与 prior sensitivity 仍在；K14 稳健却盲态语义质量较弱；K20 的额外主题以碎片化/轻微细化为主。

# 15 Researcher Review Queue

优先队列共 61 项，其中 Priority 1=14。研究者应先核验 MIXED、under-split、文本确认的 fragmentation pair，以及直接影响候选胜负的 split/merge。

# 16 Limitations

这是 AI-assisted blind audit，评分全部为 AI_PROVISIONAL，尚未完成研究者复核。代表 seed 不能穷尽每个候选的 seed 内部变异；family 与 split/merge 仍依赖解释判断。若最终采用非 K20，状态仅记录为 DOWNSTREAM_RERUN_WOULD_BE_REQUIRED；本任务没有重跑或修改任何历史下游分析。
