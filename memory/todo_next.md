# 下一步

- C_K12 作者时期历史分析已按用户明确授权完整重跑并冻结于 `history_analysis/C_K12/`；下一步优先由研究者审阅 7 条 `KEEP_WITH_CAUTION`、K12 taxonomy 与 K12/K20 事后语义对齐。不得把统计候选升级为 `LITERARY_HISTORY_CONFIRMED`，author period 不得解释为 poem date。
- 人工模型裁决仍待完成：由研究者打开 `model_selection/K12_K15_K20_human_adjudication/QTS_K12_K15_K20_Human_Review.xlsx`，仅在三个可见页独立完成 HR001–HR047 的 14 项人工判断。`QTS_K12_K15_K20_Human_Review_AI_Filled.xlsx` 仅供 AI 暂定意见复核，不得把其中结论批量转录为 `Researcher_*`。
- 47/47 完成后依次运行 `python src/validate_human_review.py` 与 `python src/freeze_human_review.py`；只有冻结 CSV 与 SHA256 成功后，才运行 `python src/summarize_human_review.py` 解盲并启用模型级、K12→K15、K15→K20、K20 碎片化、语义家族与最终证据矩阵复核。
- 旧四模型审计的 `PREFER_K12` 仍仅为 AI 初审，不得作为研究者裁决的先验答案。本次 K12 历史重跑是用户单独授权的 `PRIMARY_CANDIDATE` 分析，不等于研究者已最终确认 K；除非再次获授权，不执行 B_K12/D_K12 或 B/C/D finding-level robustness。

- pyLDAvis：本机双击 `outputs/pyldavis/pyLDAvis_C_K20.html`，选择 Topic 3 并比较 λ=1.0/0.6/0.2，完成受浏览器工具 file:// 限制而未执行的交互验收。B/D HTML已备妥；全部编号、原始概率与离线资源静态核验已通过，人工主题名称优先。
- 三模型 pyLDAvis：先本机打开 `visualization/pyldavis_C_K12_K15_K20/html/C_K12_K15_K20_pyLDAvis_comparison.html`，分别查看 K12/K15/K20；按 λ=1.0→约0.6→约0.2 人工检查，并在 `researcher/K12_K15_K20_pyldavis_researcher_checklist.xlsx` 填写末四列。不要据此单独选择 K 或覆盖既有语义裁决。

- GitHub 后续维护：把作者影响与 BCD 输出 manifest 中写死的旧 `D:` 盘绝对路径改为仓库相对路径或运行时项目根目录，再复跑主测试，目标为 207/207。
- 若仓库继续增长，优先把两份超过 50 MiB 的 CSV 迁移到 Git LFS 或发布附件；需要归档当前被忽略的模型权重、CBDB 快照或盲审私有映射时，应使用私有存储并单独核对许可与盲审状态，不直接加入公开 `main`。

1. 优先人工审核高产且 CBDB identity ambiguous 的作者（刘长卿、皎然、王建、温庭筠、权德舆、韦庄、王维等），并把批准映射写入版本化人工映射表后重跑管线。
2. 核对诗歌 ID 1 的“连薨”与“连甍”应以何种底本为准，并同步统一语料、SQL 来源或勘误记录。
3. 继续审核 `chuxueji_with_poem_example1.xlsx` 的典故候选，形成带 `allusion_id`、裁决和理由的正式典故词表；禁止自动接纳未审核异形。
4. 已完成 `sikufenci_manual_gold.jsonl` 的停用词 TF/DF 重算；尚需对 `sikufenci_lda.jsonl` 做同口径重算，再比较两种新词界下的词表稳定性。
5. B/C/D 已冻结，C_K20 taxonomy、跨模型对齐与文学史母数据集均已完成。等待用户人工确认 taxonomy 和 LOW-confidence alignment 后，再进入论文中的文学史解释；不得把 high-similarity、split 或 semantic robustness 直接当成人工结论或历史趋势证据。
6. 建模时明确处理清洗后的 16 条零 token 记录：11 条来自空正文，另 5 条由现有过滤规则清空；清洗输出阶段已全部保留，不得静默删除。
7. 用相同 LDA 参数比较基础、规则覆盖版和人工金标准覆盖版的主题稳定性与词表稀疏度，再冻结正式口径。
8. 扩充并分层抽样人工金标准；当前 485 首中有 59 首仅标前缀，现有覆盖可评价已标跨度，但不能代表 43,030 首全语料准确率。
9. `全唐诗.txt`、`李白.txt` 与 `杜甫.txt` 均已按人工覆盖版全量结果导出；如上传 Voyant，需统一停用词表和工具参数，并记录全语料及两位诗人的词频、搭配与主题比较口径。
10. 对 4 条局部词形/异文逐条做版本校勘，尤其确认 poem_id=217 的“水/永”；校勘前不得自动替换。
11. 若实际启用保护合并或 `05_semantic_mapping.tsv`，在最终词界/词面上重算 TF/DF，不直接沿用当前基线预估。
12. C_K20 author-period Trend Structure 已完成；经用户人工确认后，可进入 Period Effect Size & Statistical Validation。下一阶段必须继续保留 STRICT/CORE/EXPANDED、poem/author 两种 estimand、author cluster 结构与 metadata risk，不得把作者时期继承当作作品年代。
13. 当前六规格稳健性为 HIGH 0、MODERATE 16、LOW 4；所有 CLEAR Topics 均为 MODERATE。进入文学史解释前，应先完成效应量、统计验证及 B_K20/D_K20 finding-level robustness，不能因没有 HIGH Topic 而事后放宽趋势门槛。
14. C_K20 Author Influence / Leave-One-Author-Out 已完成并验收。保留 GLOBAL_TOP 主分类及 PER_PERIOD_TOP 敏感性差异，不以第二方案的较有利结果撤销降级；未来若重建工作簿，需继续独立验证导出文件，生成器非零退出码问题尚未定位。
15. B_K20/D_K20 finding-level robustness 已完成：9项中4项跨模型稳健、5项部分稳健。下一步仅在用户指示后，对 T08 E-L、T03 E-L、T03 M-L、T06 E-L 进入 Humanistic Interpretation / Representative Poems / Close Reading；先确认来源mapping，不能自动开始。其余5项保留方法限制，不列入跨模型稳健主叙述。
16. T10 E-L、T16 H-M、T19 E-L、T20 E-H 已降级，不作为 B/D 主候选；若未来复核，应保留为敏感性对照。所有候选均不等于 `LITERARY_HISTORY_CONFIRMED`，author period 不得当作 poem composition date，作者影响不等于文学史重要性。
17. 若重建BCD工作簿，读取manifest implementation中的C盘builder位置，用配置Node与输出目录参数运行；导出工具的末尾非零退出状态尚未定位，应继续独立验证ZIP/XML、全量数据、公式与哈希，不据退出码单独判断文件有效性。严格保持CSV统计与XLSX导出分离。
18. 人文学者审读报告已完成，按本次指示停止。后续由用户确认映射、代表诗时期/CORE资格和同文异署问题后再启动细读；T03 M-L的B/D额外作者排除尚未做，不补称已验证。趋势数据字典 `trend_shape` 的文字释义与实际数值方向分类不一致，待用户确认是否仅作释义勘误。DOCX仍待具备LibreOffice等分页渲染条件后视觉复核，Markdown为权威版。
19. 论文主题趋势图已完成并验收于 `historical_analysis/publication_figures/`。下一步仅由用户选择正文图（优先1、5a/5b、7，6的AUTHOR版可选）及补充图；无需重新统计。图6不含尚未保存的B/D时期均值CI；沿用来源映射的人工审核状态限制。按本次指示停止，不自动开始文学史解释。
