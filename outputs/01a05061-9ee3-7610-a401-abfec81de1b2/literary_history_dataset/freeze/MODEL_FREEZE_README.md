# 最终 LDA 模型冻结说明

B_K20、C_K20、D_K20 已从通过精确复现验证的模型目录复制到本目录，原始文件未移动或删除。

- B_K20：broader-vocabulary control model。
- C_K20：primary model，是后续文学史研究唯一主模型。
- D_K20：stricter-filter robustness model。

**自此以后，任何文学史分析不得重新拟合这些模型。** 禁止重新训练、重新 `fit_transform`、改变 TF/DF、改变 K、平均三个模型的概率或修改人工评分。
