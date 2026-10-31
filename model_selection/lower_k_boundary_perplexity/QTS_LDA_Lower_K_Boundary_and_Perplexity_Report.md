# 《全唐诗》LDA 主题数下边界与测试集困惑度补充报告

## 1. 为什么需要补做 K<20？

原实验只考察 K=20/30/40/50/60，而最终主模型恰为搜索下界 K20。因此原结论只能说明 K20 在已测试范围中较合适，不能排除 K10 或 K15 更合适。本轮向下补测，专门检查边界效应。

## 2. 实验设计

C corpus 新增 K=5/10/15 的三 seed 全语料诊断，并对 K=5–60 全部八个 K 做固定 80/20 held-out 评估。A/B/D 只对原 K=20–60 做 held-out 评估。全轮 9 个低 K 全语料 run 与 69 个 held-out run；split random_state=20260904。

## 3. 什么是 Perplexity？

Perplexity 可理解为模型面对未参与训练的新诗时，对词语分布有多“困惑”；越低表示预测拟合越好。最低 perplexity 不等于最有文学解释力的 Topic model。

## 4. 为什么使用 Held-out Test Set？

训练集只负责拟合，测试集只用于 score 与 perplexity。这样可观察模型对未见诗歌的预测表现，避免把训练资料上的贴合程度误当作泛化能力。词表与 BoW 完全复用原语料定义。

## 5. C Corpus：K=5–60

| K | NPMI ↑ | Diversity ↑ | Stability ↑ | Test Perplexity ↓ | Human review |
| ---: | ---: | ---: | ---: | ---: | --- |
| 5 | 0.095839 | 0.666667 | 0.491281 | 7660.721 ± 16.504 | PENDING_RESEARCHER_REVIEW |
| 10 | 0.099631 | 0.561667 | 0.360343 | 10171.689 ± 65.037 | PENDING_RESEARCHER_REVIEW |
| 15 | 0.102736 | 0.511111 | 0.335408 | 12741.629 ± 39.005 | PENDING_RESEARCHER_REVIEW |
| 20 | 0.101364 | 0.456667 | 0.315420 | 15568.238 ± 161.371 | EXISTING_REVIEW_AVAILABLE |
| 30 | 0.100031 | 0.395556 | 0.285990 | 22104.343 ± 434.415 | NOT_REVIEWED_IN_THIS_AUDIT |
| 40 | 0.096310 | 0.383750 | 0.250383 | 30686.216 ± 420.014 | NOT_REVIEWED_IN_THIS_AUDIT |
| 50 | 0.091195 | 0.372000 | 0.239538 | 41638.464 ± 223.482 | NOT_REVIEWED_IN_THIS_AUDIT |
| 60 | 0.085035 | 0.373333 | 0.240014 | 55692.158 ± 333.715 | NOT_REVIEWED_IN_THIS_AUDIT |

## 6. Lower-K Automatic Metrics

NPMI 最高为 K15，Stability 最高为 K5。K20–60 的 NPMI、Diversity 与 Stability 直接读取原报告；K5/10/15 复用原函数计算。

## 7. Lower-K Topic Structure

Under-splitting 指模型可用的“主题盒子”太少，迫使边塞、送别、羁旅或仕宦等可区分语义场进入同一 Topic。自动 alignment 允许一个低 K Topic 对应多个 C_K20 Topics，不以匈牙利一对一结果代替合并诊断。

| K | LOWER_K_MERGE topics | Merge ratio | Mean best cosine |
| ---: | ---: | ---: | ---: |
| 5 | 4 | 80.0% | 0.8775 |
| 10 | 7 | 70.0% | 0.8114 |
| 15 | 8 | 53.3% | 0.8831 |

## 8. Lower-K Human Review

已生成逐 Topic 审核卡与可填写工作簿。依项目既有方法决策，proposed label、四项 1–5 分、mixed 与 under-split 标记不由程序自动填写；当前低 K 人评状态为 PENDING_RESEARCHER_REVIEW。

## 9. Perplexity Curve

以下“改善”定义为前一 K 的 perplexity 减去后一 K；负值表示预测表现恶化。

- K5→K10：绝对改善 -2510.968，相对改善 -32.777%
- K10→K15：绝对改善 -2569.940，相对改善 -25.266%
- K15→K20：绝对改善 -2826.609，相对改善 -22.184%
- K20→K30：绝对改善 -6536.105，相对改善 -41.984%
- K30→K40：绝对改善 -8581.873，相对改善 -38.824%
- K40→K50：绝对改善 -10952.248，相对改善 -35.691%
- K50→K60：绝对改善 -14053.694，相对改善 -33.752%

各相邻区间的“改善”均为负值，即 K 增加时测试集困惑度持续上升；不存在可称为 K20 附近 diminishing returns 的正向收益。

测试集困惑度最低的 K 为 K5。训练集困惑度只保存于逐 run 表，不用于主判断。

## 10. Perplexity vs NPMI

两项指标分别衡量预测拟合与主题词共现一致性，方向可能不同；不能用其中一项替代另一项。

## 11. Perplexity vs Stability

Stability 仍采用三 seed 两两匈牙利匹配后的 Top-20 词 Jaccard；其高低反映随机初始化下的可复现程度。

## 12. Perplexity vs Human Interpretability

C_K20 可读取既有人工 interpretability；K5/10/15 尚待研究者填写，因此现阶段不能完成严格的人评 trade-off 比较。对应图保留这一缺口，不补造分数。

## 13. K20 是否仍然合理？

K15 在自动指标中仍具竞争力，但仍有 53.3% 的自动合并提示，且低 K 人工审核尚缺；当前证据不足以推翻冻结的 C_K20，应先完成人工比较，再决定是否调整模型选择。

## 14. Boundary Test Conclusion

**LOWER_K_COMPETITIVE**（自动诊断状态；低 K 人工评分待补）。

## 15. Limitations

本轮固定使用原全语料词表，因此 TF/DF 词表本身由全语料确定；held-out 评估隔离的是 LDA 参数拟合。Alignment 阈值是诊断规则，不等于人工语义裁决。K30+ 的 fragmentation 未在本轮新增人工审核。人工评分完成前，整合判断保留限定。
