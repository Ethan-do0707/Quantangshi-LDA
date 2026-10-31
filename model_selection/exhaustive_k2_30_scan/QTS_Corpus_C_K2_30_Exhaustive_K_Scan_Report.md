# QTS C Corpus K=2–30 Exhaustive K Scan Report

## 1. Objective

旧实验只按 K=20/30/40/50/60 粗粒度搜索；补测 K=5/10/15 后，K15 的旧 full-corpus NPMI 超过 K20，而 held-out perplexity 偏好更低 K。本实验逐整数扫描 K=2–30，用统一 train-only 模型状态回答真实局部峰、平台与转折在哪里。旧 NPMI 保留不覆盖；本文数值统一标记为 `TRAIN_SPLIT_NPMI`。

## 2. Experimental Design

只使用 C corpus（TF>=3、DF>=3），冻结矩阵 43,015 首非空诗、词表 35,063 词。K=2–30 共 29 个整数，每 K 使用 seed 42、2026、20260830，共 87 runs。所有 run 直接复用旧 `perplexity_split_C.csv` 的 34,412/8,603 切分；CSV 保存成员关系，训练行顺序由旧配置恢复并以旧 `fit_rows_sha256`、`test_rows_sha256` 复核。每个模型只在 X_train 拟合；X_test 只用于评分；NPMI 共现 reference 也只用 X_train。

## 3. Perplexity

全局最低 held-out perplexity 位于 K=2：6023.885442 ± 11.259860。正式形状判断为 `STRICTLY_INCREASING`。Perplexity 越低，表示模型对未参与训练的新诗给出的概率预测更顺；这里由 sklearn `score` 的同一变分下界按 `exp(-score/词次)` 换算，并逐 run 验证为有限正值。Held-out predictive fit systematically favors very low-resolution topic models. 这显示预测拟合与人文学语义分辨率之间存在权衡；不能据此把最低 K 自动称为最佳模型。

## 4. NPMI

全局最高 `TRAIN_SPLIT_NPMI` 位于 K=12：0.0999330781 ± 0.0001521897；状态为 `NEW_QUANTITATIVE_CANDIDATE`。NPMI 越高，表示同一 Topic 的 top-20 重要词在训练诗歌中彼此共同出现得更自然。正式曲线形状判断为 `BROAD_PLATEAU`。

## 5. Local Peaks and Valleys

NPMI 严格局部峰：K5、K9、K12、K14、K21、K23、K26、K29。Perplexity 严格局部谷：无。端点只参加全局极值判断，不按给定的双邻点定义标记为局部极值。二阶差分仅保存在诊断表中，不据此自动认定最佳 K。

## 6. K10–20 Zoom

K10–20 内 NPMI 最高为 K=12（0.0999330781）。指定的 K13、K14、K16、K17、K18、K19 中，高于 K15 的是：K14。该比较只产生 quantitative candidate，不替代人工语义审核。

## 7. Metric Seed Variability

全体 K 的 NPMI SD 中位数为 0.0031930450，候选分层以此作相对稳定阈值；perplexity 另报告 SD 与 CV。NPMI 峰值附近 K=11–13 的最大 NPMI SD 为 0.0045492965，最大 perplexity CV 为 0.6394%。三 seed 波动描述重复训练敏感度，不等同跨 Topic 语义稳定性。

## 8. NPMI Plateau

达到全局最大值 99% 的 K：K11–K15；达到 98% 的 K：K9、K11–K19、K21、K23。连续主平台标签：`NPMI_PLATEAU_K11_K19`；K9、K21、K23虽也达到 98%，但与主平台不连续，故不并入平台名称。这些阈值表示近峰区间，不把极小数值差自动解释为显著优劣。

## 9. Perplexity Trend

K=2–30 的判断为 `STRICTLY_INCREASING`；所有真实一阶差分见 `k_scan_first_differences.csv`，3-point moving average 只在汇总表作为辅助，不替换原始均值。NPMI 与 perplexity 回答不同问题：前者衡量主题内部词语一致性，后者衡量对新诗的预测拟合，因此最优 K 不同并非实验失败。

## 10. Candidate K Window

去重候选为：K5、K9、K11–K15、K20–K21、K23、K26、K29。新增 quantitative candidate 为：K5、K9、K11–K14、K21、K23、K26、K29。`TIER_1` 要求进入 NPMI top-1% 且 NPMI SD 不高于全 K 中位数；其他由定量极值触发者为 `TIER_2`；仅因历史保留的 K15/K20 为 `REFERENCE`。候选数不作机械限制。

## 11. Comparison with K15 and K20

K15−K20 的 NPMI 差为 +0.0023489422；held-out perplexity 差为 -2808.035144。这只限定定量分辨率竞争关系；不修改冻结 C_K20 taxonomy、主模型或下游文学史结果。

## 12. Prior Audit

实际参数：`learning_method=batch`、`max_iter=30`、`learning_decay=0.7`、`learning_offset=10.0`、`max_doc_update_iter=100`、`mean_change_tol=0.001`、`evaluate_every=-1`、`perp_tol=0.1`、`n_jobs=-1`。原配置两项均为 None；scikit-learn 本次实际按每个 K 使用 1/K，故有效先验随 K 改变。主实验不修改 alpha/eta。若单调趋势很强，后续可另做 fixed-prior sensitivity；本轮未自动启动。由于 `evaluate_every=-1`，sklearn 不执行外层 perplexity 早停，`converged_flag=False` 只表示达到 30 轮上限，不能证明数学意义上的未收敛。

## 13. Limitations

本实验沿用冻结全语料词表，测试诗虽未参与 LDA fit，但词表预处理层来自全语料；held-out perplexity 是 sklearn 变分下界近似，不是 document-completion perplexity。NPMI 只看 top-20 文档共现，未做人工 Topic 标签、代表诗细读、文学解释或时期分析。三 seed 的指标 SD 也不能代替跨 seed Topic matching。

## 14. Next-step Recommendation

正式建议：`FINE_GRID_REVEALS_NEW_CANDIDATE`。下一阶段只应对本次候选或平台内 K 做定向人文学语义审核，并与 K15、冻结 K20 同表比较；在审核前不重新冻结主模型。
