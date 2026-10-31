# 《全唐诗》六个候选 LDA 模型完整人工 Topic 审核包

## 1. 模型来源与复现

六个模型均使用第一轮原 BoW、原 scikit-learn Batch LDA 参数和已选 seed 重新拟合。每个模型只有在重建 top-20 主题词 CSV 与第一轮文件 SHA-256 完全一致、重建 top-5 代表诗 CSV 也完全一致，并通过指标与矩阵回读检查后，才进入本审核包。

## 2. Selected seed

每个 Corpus×K 仍按 NPMI 最接近三个 seed 平均值选择，距离并列时取较小 seed；没有改选单次最高值。

- `A_K20`：seed `20260830`；mean NPMI=0.0685453771；selected NPMI=0.0759173764；distance=0.0073719993；similarity≥0.50 pairs=28。
- `B_K20`：seed `20260830`；mean NPMI=0.0996023647；selected NPMI=0.0983576110；distance=0.0012447537；similarity≥0.50 pairs=34。
- `B_K30`：seed `2026`；mean NPMI=0.0948181883；selected NPMI=0.0945970171；distance=0.0002211712；similarity≥0.50 pairs=42。
- `C_K20`：seed `2026`；mean NPMI=0.1013637818；selected NPMI=0.1028332265；distance=0.0014694447；similarity≥0.50 pairs=28。
- `D_K20`：seed `42`；mean NPMI=0.1058211914；selected NPMI=0.1068058130；distance=0.0009846216；similarity≥0.50 pairs=26。
- `D_K30`：seed `42`；mean NPMI=0.1034422036；selected NPMI=0.1040689644；distance=0.0006267608；similarity≥0.50 pairs=18。

## 3. Top Words 与 Representative Poems

Top 30 Words 直接由完整归一化 Topic×Word 矩阵按概率降序提取；概率并列时按 term_index。Top 20 Representative Poems 由完整 Document×Topic 矩阵按 P(topic|poem) 降序提取；并列时按 matrix_row。

## 4. Random High-Probability Poems

每个 Topic 先取概率排名前 10% 且 P≥1/K 的诗，排除 Top 20 代表诗，再从候选池抽取至多 20 首。全局种子为 20260830，并通过 model_id、topic_id 派生稳定子种子。

## 5. Topic Prevalence

soft prevalence 为全部有效诗 P(topic|poem) 的均值；dominant prevalence 使用每诗最大概率 Topic（并列取较小 topic_id）。同时提供 ≥0.2、≥0.3、≥0.5、≥2/K、≥3/K 计数。

## 6. Topic Similarity 与比较材料

模型内及跨模型相似度均使用完整 Topic×Word 概率分布的 cosine similarity。跨语料比较先按 token 在词表并集上对齐；C-K20 vs D-K20 使用 Hungarian 一对一最佳匹配。K20 vs K30 为每个 K20 Topic 保留 Top 3，若其对至少两个 K30 Topic 的 similarity ≥0.40，则标记 possible_split。

## 7. 人工评分

Semantic Coherence、Poem Coherence、Interpretability、Distinctiveness 均为 1–5 分；具体锚点已写在每张审核卡。请把人工标签、评分、True/False 标记和备注填入 `human_topic_evaluation.csv`，程序不会自动命名或自动判断好坏。

## 8. 评分后汇总

```powershell
python src/build_human_topic_review.py --config config/human_topic_review_config.yaml --summary-only
```

只有四项分数完整且均在 1–5 时才纳入模型均值；interpretability_score≤2 计为 uninterpretable Topic。
