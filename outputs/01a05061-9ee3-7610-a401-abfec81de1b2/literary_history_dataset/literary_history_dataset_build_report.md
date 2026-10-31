# 文学史母数据集构建报告

## 1. 冻结结果

B_K20、C_K20、D_K20 均已成功复制冻结，原始模型未移动或删除；所有完整性检查通过。三个模型的 model.joblib SHA-256：

- B_K20：`d0907b37d5da7cb36e3487851627dbd9f159205050dd09ff164c3e2a8f54463a`
- C_K20：`4b23ee5661afa3127fd52f4938ca166446a9c0903cf8706c0985c964446d6d6f`
- D_K20：`7100c135964812ad224cfffbb20c869094be5961be3236cc315619a4b5ff0e95`

## 2. C_K20 20-Topic Taxonomy

topic_01. 宫廷宴乐与杂咏（噪声）（NOISE）
topic_02. 人生出处、忠义与悲慨（混合）（MIXED）
topic_03. 朝廷礼乐、应制与颂圣（CLEAR）
topic_04. 春景花木与岁时感怀（BROAD）
topic_05. 君子伦理、仁义与政治教化（BROAD）
topic_06. 老病身世、仕宦与人生感怀（CLEAR）
topic_07. 相思离愁、夜月与琴声（CLEAR）
topic_08. 山居寺院、僧侣与幽居（CLEAR）
topic_09. 泛化词主导杂咏（噪声）（NOISE）
topic_10. 江湖羁旅、送别与思归（CLEAR）
topic_11. 花香春情、女性与闺情（CLEAR）
topic_12. 春水花鸟与自然风物（BROAD）
topic_13. 清景物候、咏物与杂咏（混合）（MIXED）
topic_14. 寒景云物与杂咏（混合）（MIXED）
topic_15. 功名文士与颂赠杂糅（噪声）（NOISE）
topic_16. 边塞征战与军旅（CLEAR）
topic_17. 仙山道观、登临与神仙想象（BROAD）
topic_18. 飞鸟夜景、山水与行旅（混合）（MIXED）
topic_19. 闲居山林与隐逸生活（CLEAR）
topic_20. 饮酒酬赠、宴集与送别（CLEAR）

类型计数：{'NOISE': 3, 'MIXED': 4, 'CLEAR': 9, 'BROAD': 4}。可进入 primary/secondary interpretation：topic_03, topic_04, topic_05, topic_06, topic_07, topic_08, topic_10, topic_11, topic_12, topic_16, topic_17, topic_19, topic_20；limited：topic_02, topic_13, topic_14, topic_18；exclude：topic_01, topic_09, topic_15。

## 3. B–C–D Alignment

两侧共40个 alignment 的类型计数：{'weak': 4, 'split': 20, 'strong_1to1': 8, 'merge': 3, 'unmatched': 2, 'moderate_1to1': 3}。Semantic stability：{'PARAMETER_SENSITIVE': 5, 'PARTIALLY_ROBUST': 11, 'ROBUST': 4}。

ROBUST：topic_04, topic_08, topic_10, topic_16。
PARTIALLY_ROBUST：topic_02, topic_03, topic_05, topic_06, topic_07, topic_11, topic_12, topic_14, topic_17, topic_19, topic_20。
PARAMETER_SENSITIVE：topic_01, topic_09, topic_13, topic_15, topic_18。

## 4. Document×Topic 与母数据集

C_K20 Document×Topic shape 为 43,015×20。成功通过 poem_id 唯一匹配 43,015 首；unmatched matrix rows=0。全语料另有 15 首因 C 清洗后为空而未进入矩阵，已完整报告。

最终 master dataset 为 43,015 行 × 54 列：`D:/中大中文系工作相关/論文發表與投稿/全唐詩LDA/outputs/01a05061-9ee3-7610-a401-abfec81de1b2/literary_history_dataset/master/C_K20_literary_history_master.csv`。完整20维概率全部保留；dominant/second/third、entropy、normalized entropy 与 top1-top2 gap 为辅助字段。

## 5. Metadata QC

author 覆盖率 100.00%；title 100.00%；volume 0.00%；date/year 0.00%/0.00%；period 0.00%；genre/form 0.00%/0.00%；location 0.00%。

- author/title/text 完整，可直接开展作者级描述与文本回查。
- 没有可靠 date/year/period，尚不能开展时间趋势或初盛中晚唐分析。
- 没有 genre/form/poetic_form，尚不能开展文体比较。
- 没有 place/region/location，尚不能开展空间分析。

## 6. 数据风险与下一阶段边界

1. poem_id=1 的原文与分词来源仍存在“连薨/连甍”既知异文，母表保留权威 SQL 导出原文，模型概率保持冻结结果。
2. 15 首清洗后空诗不在 C_K20 Document×Topic 中，未静默删除，已列入 unmatched_documents.csv。
3. Topic 01、09、15 为 NOISE；Topic 02、13、14、18 为 MIXED，不得与 CLEAR Topic 等量齐观。
4. Cross-model alignment 是 topic-level semantic robustness，不是 historical trend robustness。
5. alignment 中 LOW confidence 项已在审核包标记 NEEDS_HUMAN_REVIEW。
6. 下一阶段可直接进行作者聚合、Topic 个案取诗与概率/熵描述；时间、文体和空间研究必须先补充并校验相应 metadata。

## 7. 自动测试与 XLSX 验证

- `qc/final_dataset_tests.csv`：15/15 PASS；Top1/Top2/Top3 对 43,015 行逐行回查未舍入冻结矩阵。
- `freeze/model_integrity_check.csv`：33/33 PASS。
- `master/C_K20_literary_history_master.xlsx`：Data dimension=`A1:BB43016`，共 43,015 个数据行、54 列；六个连续派生字段共有 258,090 个公式。
- `qc/master_xlsx_verification.json`：首行、中间行和末行的公式缓存值与 CSV 对应字段差值均为 0，整体状态 PASS。
