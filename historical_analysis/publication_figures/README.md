# 《全唐诗》作者时期主题趋势论文图

已生成10个图组、13个最低要求核心图位；共27个文件版本，每版均有600 dpi PNG、矢量PDF、SVG和对应plot_data CSV。重复/排版版本不代表新增发现。

## 来源与复现

脚本只读取既有结果CSV，不读取概率矩阵计算均值，不重新bootstrap、训练、分类、推断、平滑或更改映射。同名候选中选择最新已完成且manifest QC通过的目录；使用前验证冻结哈希。

```powershell
python historical_analysis/publication_figures/scripts/generate_publication_figures.py
```

仅核查来源和数值：命令末尾加 `--audit-only`。数值或映射核对不一致时脚本中止。

实际字体：Noto Serif SC（思源宋体，常规400字重）；字体文件：`D:\中大中文系工作相关\論文發表與投稿\全唐詩LDA\historical_analysis\publication_figures\qa\fonts\QTS_Noto_Regular.ttf`。PDF嵌入TrueType字体，SVG转路径。

## 使用说明

- 正文优先：图1、图5a/5b、图7；需要跨模型四期路径时选图6的AUTHOR版本。
- 补充材料：图2—4（正文已有图1时）、图6的POEM版、图7b、图8—10。图8或图10可用于正文方法讨论。
- 图1a—1c为独立权威图；图1及combined为排版合图。图2—4与其相应面板内容一致。
- 图6无后缀文件与AUTHOR版内容一致；图5和图10另有分面合图。
- 概率纵轴从0起；跨模型效应同图共轴，图5两聚合版本同尺度，图9两轴等比例。
- 两聚合并画时：POEM蓝实线实心圆，AUTHOR棕虚线空心方。三模型并画时：B蓝虚线三角、C绿实线圆、D紫点线菱形；聚合由标题/分面明确。图10用固定主题编码及聚合分面。黑白打印仍可区分。
- 每图来源、行数、生成时间和脚本见figure_manifest.csv；完整精度与原CSV行号见data目录。

## 需要保留的说明

- 图1—4均值CI来自趋势阶段2,000次bootstrap；图5/8效应CI来自BCD阶段5,000次bootstrap，不同对象及阶段未混用。
- 图6直接使用BCD表四期点估计；该表未保存B/D四期均值CI，所以图6不画CI，不新增计算。
- T03并非持续下降：盛唐至中唐有回升。T08的B/D AUTHOR路径也非持续上升；保留所有真实均值。
- T16六个点估计同向为负，B两种CI跨0；B区间不一定在绝对宽度上比C/D更宽。
- T03/T06的B/D是split主题概率和；T08是HIGH DIRECT。来源未记录正式human_review_status，不补称人工确认。
- 图10保留原GLOBAL_TOP方案及实际删除名单，不混同BCD额外检查的GLOBAL_TOP_WITHIN_CONTRAST。
- 原trend_shape数据字典释义不一致已在项目记忆记录，本任务不改冻结字典，图只读真实均值。

## 文件索引

| 文件基名 | 用途 |
|---|---|
| [Figure_01a_T03](png/Figure_01a_T03.png) | 独立权威图；正文图1组成面板 |
| [Figure_02_T03_Ritual_Court_Praise_Trend](png/Figure_02_T03_Ritual_Court_Praise_Trend.png) | 补充材料或单题讨论；与图1面板择一 |
| [Figure_01b_T06](png/Figure_01b_T06.png) | 独立权威图；正文图1组成面板 |
| [Figure_03_T06_Life_Career_Aging_Trend](png/Figure_03_T06_Life_Career_Aging_Trend.png) | 补充材料或单题讨论；与图1面板择一 |
| [Figure_01c_T08](png/Figure_01c_T08.png) | 独立权威图；正文图1组成面板 |
| [Figure_04_T08_Temple_Reclusion_Trend](png/Figure_04_T08_Temple_Reclusion_Trend.png) | 补充材料或单题讨论；与图1面板择一 |
| [Figure_01_Core_Robust_Topic_Trends](png/Figure_01_Core_Robust_Topic_Trends.png) | 正文优先 |
| [Figure_01_Core_Robust_Topic_Trends_combined](png/Figure_01_Core_Robust_Topic_Trends_combined.png) | 图1排版副本 |
| [Figure_05a_BCD_Effects_POEM](png/Figure_05a_BCD_Effects_POEM.png) | 正文优先，与另一聚合版本配对 |
| [Figure_05b_BCD_Effects_AUTHOR](png/Figure_05b_BCD_Effects_AUTHOR.png) | 正文优先，与另一聚合版本配对 |
| [Figure_05_BCD_Core_Finding_Effect_Comparison](png/Figure_05_BCD_Core_Finding_Effect_Comparison.png) | 图5a、5b的分面排版合图 |
| [Figure_06a_T03_BCD_Period_Profile_POEM](png/Figure_06a_T03_BCD_Period_Profile_POEM.png) | 补充材料 |
| [Figure_06a_T03_BCD_Period_Profile_AUTHOR](png/Figure_06a_T03_BCD_Period_Profile_AUTHOR.png) | 正文候选 |
| [Figure_06a_T03_BCD_Period_Profile](png/Figure_06a_T03_BCD_Period_Profile.png) | 正文候选；默认AUTHOR版 |
| [Figure_06b_T06_BCD_Period_Profile_POEM](png/Figure_06b_T06_BCD_Period_Profile_POEM.png) | 补充材料 |
| [Figure_06b_T06_BCD_Period_Profile_AUTHOR](png/Figure_06b_T06_BCD_Period_Profile_AUTHOR.png) | 正文候选 |
| [Figure_06b_T06_BCD_Period_Profile](png/Figure_06b_T06_BCD_Period_Profile.png) | 正文候选；默认AUTHOR版 |
| [Figure_06c_T08_BCD_Period_Profile_POEM](png/Figure_06c_T08_BCD_Period_Profile_POEM.png) | 补充材料 |
| [Figure_06c_T08_BCD_Period_Profile_AUTHOR](png/Figure_06c_T08_BCD_Period_Profile_AUTHOR.png) | 正文候选 |
| [Figure_06c_T08_BCD_Period_Profile](png/Figure_06c_T08_BCD_Period_Profile.png) | 正文候选；默认AUTHOR版 |
| [Figure_07_Finding_Robustness_Ladder](png/Figure_07_Finding_Robustness_Ladder.png) | 正文优先 |
| [Figure_07b_Finding_Evidence_Matrix](png/Figure_07b_Finding_Evidence_Matrix.png) | 补充材料 |
| [Figure_08_T16_Border_War_Partial_Robustness](png/Figure_08_T16_Border_War_Partial_Robustness.png) | 补充材料；方法讨论可选正文 |
| [Figure_09_POEM_vs_AUTHOR_Core_Findings](png/Figure_09_POEM_vs_AUTHOR_Core_Findings.png) | 补充材料 |
| [Figure_10a_Author_Influence_POEM](png/Figure_10a_Author_Influence_POEM.png) | 补充材料 |
| [Figure_10b_Author_Influence_AUTHOR](png/Figure_10b_Author_Influence_AUTHOR.png) | 补充材料 |
| [Figure_10_Author_Influence_Core_Findings](png/Figure_10_Author_Influence_Core_Findings.png) | 补充材料；方法讨论可选正文 |

图注：publication_figure_captions.md；数值和来源核验：qa目录。PDF渲染与逐图版面检查由独立验收脚本及人工检查记录。

下一步仅由用户选择正文图和排版位置；本任务完成后停止。

## 最终验收

全部27份PDF已逐张渲染并完成版面复核；669项生成核验与2,369项独立核验通过。图9最终调整另经定向核验。详见[最终验收记录](qa/FINAL_QA.md)。
