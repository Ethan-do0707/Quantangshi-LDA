# A–N 完成汇报

## A. C lower-K full-corpus runs

9/9，全部完成。

## B. Held-out perplexity runs

69/69，全部完成。

## C. C corpus K=5–60

| K | NPMI | Topic Diversity | Stability | Held-out Perplexity |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 0.095839 | 0.666667 | 0.491281 | 7660.721 ± 16.504 |
| 10 | 0.099631 | 0.561667 | 0.360343 | 10171.689 ± 65.037 |
| 15 | 0.102736 | 0.511111 | 0.335408 | 12741.629 ± 39.005 |
| 20 | 0.101364 | 0.456667 | 0.315420 | 15568.238 ± 161.371 |
| 30 | 0.100031 | 0.395556 | 0.285990 | 22104.343 ± 434.415 |
| 40 | 0.096310 | 0.383750 | 0.250383 | 30686.216 ± 420.014 |
| 50 | 0.091195 | 0.372000 | 0.239538 | 41638.464 ± 223.482 |
| 60 | 0.085035 | 0.373333 | 0.240014 | 55692.158 ± 333.715 |

## D. K5/10/15 人工审核

Mean Interpretability、Mean Overbreadth、Mixed Topic Ratio 与 Under-split Ratio：待研究者填写；程序未代填人评。

## E. Topic merge / under-splitting

按比例逐级下降，但未消失：K5 merge ratio=80.0%; K10 merge ratio=70.0%; K15 merge ratio=53.3%。K15 仍有超过一半 Topic 被自动标为 LOWER_K_MERGE，需人工语义确认。

## F. K15 vs K20

- NPMI：K15 更高。
- Stability：K15 更高。
- Diversity：K15 更高。
- Perplexity：K15 更低。
- Human Interpretability：K15 待评，不能比较。

## G. 相邻 K perplexity 变化

- K5→K10：绝对改善 -2510.968，相对改善 -32.777%
- K10→K15：绝对改善 -2569.940，相对改善 -25.266%
- K15→K20：绝对改善 -2826.609，相对改善 -22.184%
- K20→K30：绝对改善 -6536.105，相对改善 -41.984%
- K30→K40：绝对改善 -8581.873，相对改善 -38.824%
- K40→K50：绝对改善 -10952.248，相对改善 -35.691%
- K50→K60：绝对改善 -14053.694，相对改善 -33.752%

## H. K20 附近 diminishing returns

不支持。各相邻区间的“改善”均为负值，即 K 增加时测试集困惑度持续上升；不存在可称为 K20 附近 diminishing returns 的正向收益。

## I. Perplexity 最偏好的 K

K5。

## J. NPMI 最偏好的 K

K15。

## K. Stability 最偏好的 K

K5。

## L. Human Interpretability

低 K 尚待人工审核；现有可用值仅为 C_K20 平均 3.750，不能据此作跨 K 排名。

## M. Lower-K Boundary Test

LOWER_K_COMPETITIVE

## N. Integrated Judgment

REVIEW_K15_VS_K20

K15 在自动指标中仍具竞争力，但仍有 53.3% 的自动合并提示，且低 K 人工审核尚缺；当前证据不足以推翻冻结的 C_K20，应先完成人工比较，再决定是否调整模型选择。
