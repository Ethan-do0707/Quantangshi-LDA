"""生成跨模型发现层分析的报告、数据字典与静态科学图。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def display_finding(fid: str) -> str:
    """把稳定标识显示为可读的 Topic 与 contrast。"""
    topic, contrast = fid.split("_")
    return f"{topic} {contrast[0]}–{contrast[1]}"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """生成不依赖额外包的紧凑 Markdown 表。"""
    rows = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for record in frame[columns].to_dict("records"):
        values = []
        for value in record.values():
            if isinstance(value, (float, np.floating)):
                text = f"{value:.6f}" if math.isfinite(value) else "NA"
            else:
                text = str(value)
            values.append(text.replace("|", "/").replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def build_report(frames: dict[str, pd.DataFrame], config: dict[str, Any], audit: dict[str, list[str]]) -> str:
    """按任务指定十九节组织方法与结果，不开展文学史解释。"""
    comparison = frames["BCD_finding_robustness_comparison"]
    qc = frames["BCD_common_universe_qc"]
    mappings = frames["finding_semantic_mapping"]
    effects = frames["BCD_finding_effects"]
    benchmark = frames["C_original_vs_common"]
    counts = comparison["finding_level_robustness"].value_counts()
    text = ["# QTS B–C–D Finding-Level Robustness Report", "", "## 1. Objective", "",
            "本阶段只验证既有 C_K20 finding 对词汇过滤设置的稳健性，不重新选择模型、不重训 LDA、不改写已有 C effect validation 或 LOAO 状态。分析单位是语义场与具体时期 contrast 的组合，不是整个 Topic。",
            f"输入 `finding_level_bd_eligible=True` 实际得到 {len(comparison)} 项；预期清单差异：{audit}。另有 {len(frames['excluded_findings_status'])} 项仅保留在排除状态附表，没有进入主验证。", "",
            "## 2. Model Roles", "", "B_K20：broader-vocabulary control（TF≥2、DF≥2）；C_K20：primary model（TF≥3、DF≥3）；D_K20：stricter-filter robustness control（TF≥5、DF≥3）。三模型保持独立，不平均 probability 或 effect。", "",
            "## 3. Common Document Universe", "", markdown_table(qc.loc[qc["sample"].eq("ALL")], ["model", "total_documents", "common_documents", "lost_documents", "lost_poem_ids"]), "",
            "共同集合严格按 poem_id 取 B∩C∩D；matrix_row 仅用于连接后的矩阵取值。逐时期/STRICT/CORE/EXPANDED 损失明细见 `BCD_common_universe_qc.csv`。",
            f"三套时期样本的总损失为 {int(qc.loc[qc['sample'].isin(config['samples']) & qc['period'].eq('ALL'), 'lost_documents'].sum())}。B 的一个原生文档没有冻结 C-based 历史 master/sample 行，因此明确标为 metadata unavailable，不重推时期或样本资格；它不进入任何时期样本。",
            f"C_ORIGINAL→C_COMMON 最大 |Δ变化|={benchmark['absolute_delta_change'].max():.3g}，DOCUMENT_INTERSECTION_SENSITIVITY={int(benchmark['DOCUMENT_INTERSECTION_SENSITIVITY'].sum())} 行。原 C 区间使用旧 seed，COMMON 区间使用本阶段 seed，因此仅有 CI 差异不自动归因于交集。原 C 结果未覆盖。",
            f"B_NATIVE/D_NATIVE 仅作为辅助敏感性，最大 |Δ相对COMMON变化|={frames['model_native_sensitivity']['delta_difference_from_primary'].abs().max():.3g}。", "",
            "## 4. Semantic Mapping Method", "",
            "映射只读取既有 alignment。DIRECT 使用一个来源目标；BUNDLE 对来源明确 split 的全部目标概率直接求和，不使用相似度权重。single-best 取来源 candidate cosine 最高者，并与 bundle 分开保存。MERGED_PROXY 无法隔离 focal 语义场，结论最多为部分稳健。WEAK_PROXY 仅探索；unmatched 不强制替代。",
            "来源 CSV 未包含 human_review_status 或 NEEDS_HUMAN_REVIEW 字段，因此输出明确保留 NOT_RECORDED_IN_SOURCE，不能把缺失当作人工批准。若将来来源出现复核标记或 LOW confidence，自动标记 ALIGNMENT_REVIEW_REQUIRED 并限制结论等级。MEDIUM bundle 亦不升级为高置信稳健。",
            "同一 target 可能承载多个 C 语义场；这种重用在 overlapping_other_c_topics 中保留，既不强制一对一，也不把各 finding 相加。",
            f"统计：{config['bootstrap_iterations']:,} 次时期分层作者 cluster bootstrap，seed={config['random_seed']}；同一时期与样本使用相同作者抽签同时计算 B/C/D 与 POEM/AUTHOR。POEM 为每诗等权；AUTHOR 为先作者内均值、再时期内作者等权。只计算入选 contrasts，不建立新的 p-value family。",
            "标准化效应沿用前阶段作者级 Hedges g 与 Cliff delta；raw Δ ratio 仅诊断，|C Δ|<0.005 时不计算。四时期 Pearson/Spearman 只作描述，不报告 p-value。",
            "分类规则在配置中冻结：ROBUST 要求 B/D 的 CORE 两聚合同 C 同号、各目标至少一种 CORE 聚合的 CI 支持、三模型每个至少5/6规格维持 C 核心方向，且没有语义代理/待复核/明显bundle依赖/目标作者反转限制。稳定反向要求至少4/6数值反向且至少2/6反向CI支持。强DIRECT下两目标均稳定反向、无样本/聚合/CLR解释候选时才允许 NOT_REPLICATED。",
            "CLR 仅在目标与 C raw 方向不同、bundle/single方向不同或原 C compositional warning 时触发。bundle 采用 log(概率和)−mean(log(完整20维概率))，不把多个 CLR 坐标相加；无真零时不加 pseudocount。", "",
            "## 5. Overall Finding-level Robustness", "",
            "; ".join(f"{status}={int(counts.get(status, 0))}" for status in config["classification_order"]), "",
            markdown_table(comparison, ["finding_id", "B_mapping_type", "D_mapping_type", "semantic_mapping_assessment", "historical_effect_assessment", "finding_level_robustness"]), ""]
    order = config["anticipated_findings_for_audit_only"]
    actual_order = [fid for fid in order if fid in set(comparison["finding_id"])] + [fid for fid in comparison["finding_id"] if fid not in order]
    for number, fid in enumerate(actual_order, start=6):
        row = comparison.loc[comparison["finding_id"].eq(fid)].iloc[0]
        core = effects.loc[effects["finding_id"].eq(fid) & effects["sample"].eq("CORE")].copy()
        core["CI95"] = core.apply(lambda r: f"[{r.bootstrap_ci_low:.6f}, {r.bootstrap_ci_high:.6f}]", axis=1)
        map_rows = mappings.loc[mappings["finding_id"].eq(fid)]
        text += [f"## {number}. {display_finding(fid)}", "", f"语义场：{row['topic_label']}。原 C={row['C_original_status']}；LOAO={row['C_author_robustness_status']}（保持原状）。", "",
                 markdown_table(map_rows, ["target_model", "target_topic_ids", "alignment_relation_type", "alignment_confidence", "mapping_testability"]), "",
                 markdown_table(core, ["model", "aggregation", "raw_delta", "CI95", "bootstrap_supported_direction", "hedges_g_if_applicable", "hedges_magnitude"]), "",
                 f"样本方向：{row['sample_direction_consistency']}；方向模式：{row['cross_model_direction_pattern']}。最终：**{row['finding_level_robustness']}**。{row['robustness_reason']}。", ""]
    next_section = 6 + len(actual_order)
    priority = comparison.loc[comparison["finding_id"].isin(config["priority_findings"])]
    model_sensitive = comparison.loc[comparison["finding_level_robustness"].isin(["MODEL_SENSITIVE", "NOT_REPLICATED"])]
    unresolved = comparison.loc[comparison["finding_level_robustness"].eq("SEMANTICALLY_UNRESOLVED")]
    ladder = frames["finding_robustness_ladder"]
    exclusions = frames["priority_author_sensitivity"]
    clr = frames["clr_sensitivity"]
    only_b = comparison.loc[comparison["B_supports_C_finding"] & ~comparison["D_supports_C_finding"], "finding_id"].tolist()
    only_d = comparison.loc[comparison["D_supports_C_finding"] & ~comparison["B_supports_C_finding"], "finding_id"].tolist()
    text += [f"## {next_section}. Priority Four", "", markdown_table(priority, ["finding_id", "B_poem_delta", "C_poem_delta", "D_poem_delta", "B_author_delta", "C_author_delta", "D_author_delta", "finding_level_robustness"]), "",
             "优先级只影响展示与额外 Top1/Top3 检查，不改变统计阈值。四期 mean profile 同时展示 POEM 与 AUTHOR；中间路径或峰值不同不自动否定所选 contrast。作者检查按 contrast 涉及时期内的语义质量总和排名，删除作者整簇。", "",
             f"实际运行 {int(exclusions['check_run'].sum())} 行 B/D×Top1/Top3×POEM/AUTHOR 检查，符号反转 {int(exclusions['sign_reversal'].sum())} 行，不可估计 {int(exclusions['author_sensitivity_flag'].eq('NOT_ESTIMABLE_AFTER_EXCLUSION').sum())} 行。仅检查方向，不把这一步称为完整目标模型LOAO。", "",
             f"## {next_section + 1}. Model-sensitive Findings", "",
             markdown_table(model_sensitive, ["finding_id", "B_supports_C_finding", "D_supports_C_finding", "finding_level_robustness", "robustness_reason"]) if not model_sensitive.empty else "当前未发现达到预设系统反向阈值的 MODEL_SENSITIVE / NOT_REPLICATED finding。", "",
             f"B supports / D does not：{', '.join(only_b) or '无'}；D supports / B does not：{', '.join(only_d) or '无'}。supports 要求 CORE 两聚合同 C 同号且至少一个区间支持，不以单纯符号或语义相似替代。",
             "D 支持而 B 不支持，至多与宽词表下语义质量更异质、严格过滤下信号更集中这两种可能性 consistent with；当前不能区分它们，也不能排除区间精度或语义范围差异。B 支持而 D 不支持的情况本批次没有观察到，不补造反例。这些只是方法解释候选，不构成过滤机制的因果证据。",
             f"条件CLR共 {len(clr)} 行，涉及 {', '.join(sorted(clr['finding_id'].unique())) or '无'}；raw/CLR符号不同 {int((~clr['raw_clr_direction_agreement']).sum()) if not clr.empty else 0} 行。详见 `clr_sensitivity.csv`，不以CLR替换主概率效应。", "",
             f"## {next_section + 2}. Semantically Unresolved Findings", "",
             markdown_table(unresolved, ["finding_id", "semantic_mapping_assessment", "robustness_reason"]) if not unresolved.empty else "没有 finding 被最终分类为 SEMANTICALLY_UNRESOLVED。仍须区分高置信 bundle、MEDIUM bundle 与 MERGED_PROXY 的解释限制。",
             "来源 split/merge 与 single-best 依赖的逐项证据见 `split_single_best_sensitivity.csv` 和 `model_sensitivity_diagnostics.csv`；无法可靠同义比较时不称为未复现。", "",
             f"## {next_section + 3}. Robustness Ladder", "", markdown_table(ladder, ["robustness_rank", "finding_id", "finding_level_robustness", "eligible_for_next_stage", "next_stage_note"]), "",
             f"## {next_section + 4}. Limitations", "",
             "- 跨模型 Topic 不是完全相同的潜变量；split/merge 降低精确可比性。",
             "- raw probability magnitude 不在严格相同的语义尺度上；不按Δ相等要求复现。",
             "- 作者时期不等于诗歌创作年代，冻结 metadata 的代表性与缺失偏差仍存在。",
             "- B/D robustness 不是历史因果证据，词汇过滤本身改变语义分辨率。",
             "- 四时期形状相关只有四个点，不作显著性推断；峰值不同不自动推翻具体 contrast。",
             "- human_review_status 在来源缺失，不能声称本阶段获得新的人工语义确认；进入下一阶段前须核对映射与代表诗。",
             "- C 既有统计/LOAO状态保持不变；本阶段只是独立跨模型证据层，止于方法结果，不开始文学史阐释。", ""]
    return "\n".join(text)


def data_dictionary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """为每张数据表每个字段记录类型与关键语义。"""
    descriptions = {
        "finding_id": "由来源C Topic与指定contrast构成的唯一发现标识", "c_topic_id": "C_K20内部Topic编号；不得直接等同B/D编号", "topic": "原C Topic标识", "contrast": "只测试来源预选对比；Δ=后一期减前一期",
        "poem_id": "跨模型与历史metadata的唯一连接键", "mapping_testability": "DIRECT/BUNDLE/MERGED_PROXY/WEAK_PROXY/NOT_TESTABLE", "target_topic_ids": "既有alignment列出的全部目标，顺序保持来源",
        "analysis_topic_ids": "实际提取概率的来源目标；unmatched为空", "semantic_similarity": "来源alignment的最高候选cosine，不作为bundle权重", "human_review_status": "来源人工审核状态；缺失明确标记，不能视为批准",
        "alignment_review_required": "来源明确需审核、审核状态标记或LOW置信时为True", "raw_delta": "指定contrast的period_B_mean减period_A_mean；概率尺度", "direction": "数值符号；与bootstrap区间支持分开",
        "bootstrap_ci_low": "作者整簇、时期分层bootstrap的2.5%分位", "bootstrap_ci_high": "作者整簇、时期分层bootstrap的97.5%分位", "bootstrap_supported_direction": "CI全正/全负时支持，否则UNCERTAIN",
        "hedges_g_if_applicable": "AUTHOR作者均值分布的B-A Hedges g；POEM留空", "cliffs_delta_if_applicable": "AUTHOR作者均值B相对A的优势概率差；POEM留空",
        "semantic_mapping_assessment": "独立于历史效应的语义可比性评价", "historical_effect_assessment": "独立于语义可比性的方向/区间/规格评价", "finding_level_robustness": "按配置固定规则合成的最终跨模型证据层",
        "model_sample_positive_count": "该model/aggregation三样本中CI支持正向的数量", "model_sample_negative_count": "该model/aggregation三样本中CI支持负向的数量", "model_sample_uncertain_count": "该model/aggregation三样本中CI跨零的数量",
        "raw_positive_count": "三样本数值正号数量，与CI支持数量不同", "raw_negative_count": "三样本数值负号数量，与CI支持数量不同", "DOCUMENT_INTERSECTION_SENSITIVITY": "按配置阈值判断C原始与共同集合效应是否有非微小变化",
        "eligible_for_next_stage": "仅跨模型稳健finding进入人工解释/细读候选，不代表历史结论", "structural_replication": "目标重现C的POEM/AUTHOR方向分歧结构", "clr_delta": "临时log语义质量相对完整20维几何均值的contrast",
    }
    rows = []
    for name, frame in frames.items():
        for field in frame:
            description = descriptions.get(field)
            if description is None:
                if field.startswith("mean_"): description = "该样本与聚合方式下指定时期的语义概率均值"
                elif "ci_" in field or "CI" in field: description = "对应来源/模型的95% bootstrap区间端点或显示值"
                elif "delta" in field: description = "对应模型/估计量的指定contrast效应或差异；不是跨模型平均值"
                elif "shape" in field or "pearson" in field or "spearman" in field: description = "四时期均值向量的描述性形状比较，不含p-value"
                elif "count" in field or "documents" in field: description = "对应字段定义范围内的记录/样本数量"
                elif "direction" in field or "agreement" in field: description = "对应模型/样本/聚合的方向或一致性标记"
                else: description = "来源冻结字段或分析审计字段；具体口径见报告第4节与配置"
            rows.append({"dataset": name, "field": field, "dtype": str(frame[field].dtype), "description": description})
    return pd.DataFrame(rows)


def readme_text(config: dict[str, Any], frames: dict[str, pd.DataFrame]) -> str:
    """提供可复运行命令和文件定位。"""
    return "\n".join([
        "# B–C–D Finding-Level Robustness", "", "只验证C_K20既有eligible findings；不重训、不重新选模型、不改写C结论。", "",
        "## 复运行", "", "```powershell", "python src/run_bcd_finding_robustness.py --config config/BCD_robustness_config.yaml", "```", "",
        "工作簿由独立 artifact-tool builder 从这些权威CSV生成；分析与XLSX制作分离。builder及两个分析模块的实际路径与哈希保存在 `analysis_manifest.json` 的 implementation 中；用配置好的Node运行builder，并把此输出目录的绝对路径作为唯一参数。", "",
        f"固定 seed={config['random_seed']}，作者分层bootstrap={config['bootstrap_iterations']}次。所有配置阈值在 `BCD_robustness_config.yaml`；输入/输出哈希在 `analysis_manifest.json`。", "",
        "## 审计入口", "", "- `data/finding_semantic_mapping.csv`：语义映射核心审计表。",
        "- `data/BCD_finding_effects.csv`：共同集合、正式映射、六规格主效应。",
        "- `data/C_original_vs_common.csv`：原C与共同集合基准，不覆盖原结果。",
        "- `data/split_single_best_sensitivity.csv`：split单目标敏感性，与正式bundle分离。",
        "- `data/model_native_sensitivity.csv`：模型原生文档辅助敏感性；缺失历史标签不重新推断。",
        "- `data/clr_sensitivity.csv`、`priority_author_sensitivity.csv`：按触发条件运行的诊断。",
        "- `data/finding_robustness_ladder.csv`：发现层排序与下一阶段候选。",
        "- `data/invariance_tests.csv`：20项任务契约检查。", "",
        "## 数据表行数", "", markdown_table(pd.DataFrame([{"dataset": n, "rows": len(f)} for n, f in frames.items()]), ["dataset", "rows"]), "",
        "## 解释边界", "", "作者时期不是诗作年代；跨模型Topic语义范围不完全相同；来源未记录正式human_review_status。本阶段只增加方法证据，不进行文学史阐释。", "",
    ])


def build_figures(frames: dict[str, pd.DataFrame], root: Path, config: dict[str, Any]) -> list[Path]:
    """生成六张要求的静态科学图，不绘制无关Topic。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import ListedColormap
    for font in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if font.exists():
            font_manager.fontManager.addfont(str(font))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
            break
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
    comparison = frames["BCD_finding_robustness_comparison"]
    ids = [fid for fid in config["anticipated_findings_for_audit_only"] if fid in set(comparison["finding_id"])]
    ids += [fid for fid in comparison["finding_id"] if fid not in ids]
    effects = frames["BCD_finding_effects"]
    core = effects.loc[effects["sample"].eq("CORE")].set_index(["finding_id", "model", "aggregation"])
    colors = {"B": "#2478A0", "C": "#202C39", "D": "#C56B30"}
    paths = []
    def save(fig: Any, name: str) -> None:
        """统一保存与关闭图。"""
        path = root / name
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(path)
    for aggregation in ("POEM", "AUTHOR"):
        fig, ax = plt.subplots(figsize=(10.5, max(6.8, len(ids) * 0.62)))
        for model, offset in zip(("B", "C", "D"), (-0.22, 0, 0.22)):
            for index, fid in enumerate(ids):
                row = core.loc[(fid, model, aggregation)]
                y = index + offset
                ax.hlines(y, row["bootstrap_ci_low"], row["bootstrap_ci_high"], color=colors[model], linewidth=2)
                ax.scatter(row["raw_delta"], y, color=colors[model], s=32, label=model if index == 0 else None, zorder=3)
        ax.axvline(0, color="#9AA5B1", linewidth=1, linestyle="--")
        ax.set_yticks(range(len(ids)), [display_finding(fid) for fid in ids])
        ax.invert_yaxis()
        ax.set_xlabel("Raw Δ = period B − period A（probability mass）")
        ax.set_title(f"BCD finding effects · CORE_{aggregation}\n作者分层bootstrap 95% CI · common document universe", loc="left", pad=15)
        ax.legend(frameon=False, ncol=3, loc="lower right")
        ax.grid(axis="x", alpha=0.15)
        save(fig, f"BCD_finding_effect_comparison_{aggregation.lower()}.png")
    columns = [(m, a) for m in ("B", "C", "D") for a in ("POEM", "AUTHOR")]
    matrix = np.array([[1 if core.loc[(fid, m, a)]["bootstrap_supported_direction"] == "SUPPORTED_POSITIVE" else -1 if core.loc[(fid, m, a)]["bootstrap_supported_direction"] == "SUPPORTED_NEGATIVE" else 0 for m, a in columns] for fid in ids])
    fig, ax = plt.subplots(figsize=(11.4, 6.9))
    ax.imshow(matrix, cmap=ListedColormap(["#BDD9ED", "#F5E8CC", "#BFDCC7"]), vmin=-1, vmax=1, aspect="auto")
    for i in range(len(ids)):
        for j in range(len(columns)):
            ax.text(j, i, {1: "POSITIVE", -1: "NEGATIVE", 0: "UNCERTAIN"}[matrix[i, j]], ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(columns)), [f"{m}_{a}" for m, a in columns])
    ax.set_yticks(range(len(ids)), [display_finding(fid) for fid in ids])
    ax.set_title("CORE bootstrap-supported direction\nUNCERTAIN 表示CI跨零，不等于零效应", loc="left", pad=15)
    save(fig, "BCD_finding_direction_matrix.png")
    mapping = frames["finding_semantic_mapping"].set_index(["finding_id", "target_model"])
    fig, ax = plt.subplots(figsize=(15, max(8, len(ids) * 0.9)))
    ax.set_xlim(0, 1); ax.set_ylim(-0.8, len(ids) - 0.2); ax.axis("off")
    for i, fid in enumerate(ids):
        y = len(ids) - 1 - i
        ax.text(0.5, y, display_finding(fid), ha="center", va="center", bbox={"boxstyle": "round,pad=.55", "facecolor": "#E7EDF2", "edgecolor": "#526579"}, fontsize=11)
        for model, x, start, end in (("B", 0.15, 0.29, 0.43), ("D", 0.85, 0.57, 0.71)):
            row = mapping.loc[(fid, model)]
            label = " + ".join(f"{model}{int(t.split('_')[-1]):02d}" for t in row["target_topic_ids"].split(";") if t)
            ax.text(x, y, f"{label}\n{row['alignment_relation_type']} · {row['alignment_confidence']}", ha="center", va="center", fontsize=9, bbox={"boxstyle": "round,pad=.4", "facecolor": "#EAF3F7" if model == "B" else "#FAEFE7", "edgecolor": colors[model]})
            style = "--" if row["mapping_testability"] in {"MERGED_PROXY", "WEAK_PROXY", "NOT_TESTABLE"} else "-"
            ax.plot([start, end], [y, y], color=colors[model], linestyle=style, linewidth=2.5 if row["mapping_testability"] == "BUNDLE" else 1.3)
    ax.set_title("Finding semantic mapping · 来源alignment，非Topic同号对应\n粗线=BUNDLE概率和；虚线=代理限制；重复语义场按finding分别展示", loc="left", pad=20)
    save(fig, "BCD_finding_semantic_mapping.png")
    short = {"ROBUST_ACROSS_BCD": "ROBUST", "PARTIALLY_ROBUST": "PARTIAL", "MODEL_SENSITIVE": "MODEL-SENS", "SEMANTICALLY_UNRESOLVED": "UNRESOLVED", "NOT_REPLICATED": "NOT-REPL"}
    headers = ["Semantic\nAlignment", "B\nDirection", "D\nDirection", "Aggregation", "Sample", "Author\nInfluence", "Final\nRobustness"]
    cells = []
    for fid in ids:
        row = comparison.loc[comparison["finding_id"].eq(fid)].iloc[0]
        cells.append([f"{row['B_mapping_type']} /\n{row['D_mapping_type']}", "SUPPORT" if row["B_supports_C_finding"] else "LIMITED", "SUPPORT" if row["D_supports_C_finding"] else "LIMITED", "AGREE" if row["poem_direction_agreement"] and row["author_direction_agreement"] else "PARTIAL", f"min {row['sample_consistency_minimum']}/6", "SENSITIVE" if "TARGET_MODEL_AUTHOR_SENSITIVE" in (row["B_author_sensitivity_flag"], row["D_author_sensitivity_flag"]) else "CHECKED" if fid in config["priority_findings"] else "C LOAO", short[row["finding_level_robustness"]]])
    fig, ax = plt.subplots(figsize=(15, 7.5)); ax.axis("off")
    table = ax.table(cellText=cells, rowLabels=[display_finding(fid) for fid in ids], colLabels=headers, loc="center", cellLoc="center", colWidths=[.22, .105, .105, .13, .1, .12, .15])
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 2.7)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("#FFFFFF")
        if i == 0: cell.set_facecolor("#263B4F"); cell.set_text_props(color="white")
        elif j == 6: cell.set_facecolor("#C6E0CD" if cells[i - 1][6] == "ROBUST" else "#F2DFB8")
        else: cell.set_facecolor("#EEF2F5" if i % 2 else "#F7F9FA")
    ax.set_title("Finding robustness evidence matrix\nROBUST/PARTIAL等完整代码见CSV；作者检查仅优先四项B/D Top1/Top3", loc="left", pad=18)
    save(fig, "BCD_finding_robustness_matrix.png")
    priority_ids = [fid for fid in config["priority_findings"] if fid in ids]
    fig, axes = plt.subplots(len(priority_ids), 2, figsize=(12.5, 3.1 * len(priority_ids)), squeeze=False)
    for i, fid in enumerate(priority_ids):
        for j, aggregation in enumerate(("POEM", "AUTHOR")):
            ax = axes[i, j]
            for model in ("B", "C", "D"):
                row = core.loc[(fid, model, aggregation)]
                ax.plot(range(4), [row[f"mean_{p}"] for p in "EHML"], marker="o", color=colors[model], label=model, linewidth=1.8)
            ax.set_xticks(range(4), config["period_order"])
            ax.set_title(f"{display_finding(fid)} · CORE_{aggregation}", loc="left")
            ax.set_ylabel("Mean probability mass")
            ax.grid(alpha=.15)
            if i == 0: ax.legend(frameon=False, ncol=3)
    fig.suptitle("Priority four · 四时期均值profile（不是所有contrast检验）", x=.08, ha="left", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .97])
    save(fig, "BCD_priority_four_trends.png")
    return paths
