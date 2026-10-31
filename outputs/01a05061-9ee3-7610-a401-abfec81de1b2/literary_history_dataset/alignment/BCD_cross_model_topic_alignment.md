# B–C–D Cross-model Topic Alignment

以 C_K20 为唯一 canonical taxonomy。相似度先投影到 pairwise shared vocabulary；Hungarian 只作为一对一基线，最终类型结合 cosine、Top30 重合、代表诗重合、人工标签和审核阅读，不强制 1:1。

- C↔B shared vocabulary：35,063。
- C↔D shared vocabulary：23,123。
- Semantic stability：{'PARAMETER_SENSITIVE': 5, 'PARTIALLY_ROBUST': 11, 'ROBUST': 4}。
- Alignment type counts：{'B:weak': 3, 'D:split': 13, 'B:split': 7, 'B:strong_1to1': 5, 'D:strong_1to1': 3, 'B:merge': 2, 'B:unmatched': 1, 'D:unmatched': 1, 'B:moderate_1to1': 2, 'D:moderate_1to1': 1, 'D:weak': 1, 'D:merge': 1}。

| C Topic | C 名称 | B 对应 | B 类型 | D 对应 | D 类型 | 稳定性 |
|---|---|---|---|---|---|---|
| topic_01 | 宫廷宴乐与杂咏（噪声） | topic_01;topic_06;topic_04 | weak | topic_09;topic_17 | split | PARAMETER_SENSITIVE |
| topic_02 | 人生出处、忠义与悲慨（混合） | topic_14;topic_07 | split | topic_11;topic_04;topic_02 | split | PARTIALLY_ROBUST |
| topic_03 | 朝廷礼乐、应制与颂圣 | topic_08;topic_19 | split | topic_17;topic_08 | split | PARTIALLY_ROBUST |
| topic_04 | 春景花木与岁时感怀 | topic_10 | strong_1to1 | topic_06 | strong_1to1 | ROBUST |
| topic_05 | 君子伦理、仁义与政治教化 | topic_08 | merge | topic_11;topic_01;topic_04 | split | PARTIALLY_ROBUST |
| topic_06 | 老病身世、仕宦与人生感怀 | topic_14;topic_13 | split | topic_05;topic_11 | split | PARTIALLY_ROBUST |
| topic_07 | 相思离愁、夜月与琴声 | topic_18;topic_04;topic_07 | split | topic_16;topic_13;topic_04 | split | PARTIALLY_ROBUST |
| topic_08 | 山居寺院、僧侣与幽居 | topic_17 | strong_1to1 | topic_12 | strong_1to1 | ROBUST |
| topic_09 | 泛化词主导杂咏（噪声） | topic_13;topic_12;topic_14 | unmatched | topic_05;topic_16;topic_17 | unmatched | PARAMETER_SENSITIVE |
| topic_10 | 江湖羁旅、送别与思归 | topic_18 | strong_1to1 | topic_16;topic_15 | split | ROBUST |
| topic_11 | 花香春情、女性与闺情 | topic_04;topic_10 | split | topic_13;topic_06 | split | PARTIALLY_ROBUST |
| topic_12 | 春水花鸟与自然风物 | topic_10;topic_17;topic_18 | split | topic_06;topic_18 | split | PARTIALLY_ROBUST |
| topic_13 | 清景物候、咏物与杂咏（混合） | topic_08;topic_07;topic_17 | weak | topic_18;topic_01;topic_04 | split | PARAMETER_SENSITIVE |
| topic_14 | 寒景云物与杂咏（混合） | topic_17 | moderate_1to1 | topic_18 | moderate_1to1 | PARTIALLY_ROBUST |
| topic_15 | 功名文士与颂赠杂糅（噪声） | topic_08;topic_16;topic_01 | weak | topic_01;topic_17 | weak | PARAMETER_SENSITIVE |
| topic_16 | 边塞征战与军旅 | topic_06 | strong_1to1 | topic_19 | strong_1to1 | ROBUST |
| topic_17 | 仙山道观、登临与神仙想象 | topic_01 | strong_1to1 | topic_18;topic_14 | split | PARTIALLY_ROBUST |
| topic_18 | 飞鸟夜景、山水与行旅（混合） | topic_17;topic_05 | split | topic_18;topic_15;topic_14 | split | PARAMETER_SENSITIVE |
| topic_19 | 闲居山林与隐逸生活 | topic_13 | moderate_1to1 | topic_12 | merge | PARTIALLY_ROBUST |
| topic_20 | 饮酒酬赠、宴集与送别 | topic_13 | merge | topic_05;topic_02 | split | PARTIALLY_ROBUST |
