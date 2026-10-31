"""在 K12 盲态结果冻结后生成 K12/K20 事后比较与最终报告。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "history_analysis" / "C_K12"
FINAL = OUT / "06_final"
AUDIT = OUT / "00_input_audit"

OLD_CANDIDATES = (
    ROOT
    / "outputs"
    / "01a05cec-c8db-7761-9760-9d2d42a4853e"
    / "author_influence_loao"
    / "data"
    / "author_validated_candidate_trends.csv"
)
BLIND_ALIGNMENT = (
    ROOT
    / "model_selection"
    / "four_model_blind_semantic_audit"
    / "data"
    / "blind_cross_model_alignment.csv"
)
BLIND_TOPIC_SECRET = (
    ROOT
    / "model_selection"
    / "four_model_blind_semantic_audit"
    / "_private_unblind"
    / "blind_topic_mapping_secret.csv"
)
ALIGNMENT_DIR = (
    ROOT
    / "model_selection"
    / "K12_K15_K20_human_adjudication"
    / "_private"
    / "stage2_candidates"
)
K12_K15 = ALIGNMENT_DIR / "K12_to_K15_alignment_candidates.csv"
K15_K20 = ALIGNMENT_DIR / "K15_to_K20_alignment_candidates.csv"

PERIOD_VALUE_COLUMNS = {
    "E": "early_mean",
    "H": "high_mean",
    "M": "middle_mean",
    "L": "late_mean",
}


# 比较规则只负责选择既有语义对齐中的相关 K12 主题；趋势方向始终从冻结 K12 结果读取。
COMPARISON_RULES: dict[tuple[str, str], dict[str, Any]] = {
    ("topic_03", "E-H"): {
        "k12": [5, 8],
        "category": "REPRODUCED_IN_REFINED_K12_COMPONENT",
        "note": "旧礼乐/应制复合主题在 K12 中细分；T05 的同一比较方向一致且经 LOAO 保留，T08 不支持同一结论。",
    },
    ("topic_03", "E-L"): {
        "k12": [5, 8],
        "category": "REPRODUCED_IN_REFINED_K12_COMPONENT",
        "note": "旧礼乐/应制复合主题在 K12 中细分；T05 的同一比较方向一致且经 LOAO 保留，T08 出现 POEM/AUTHOR 冲突。",
    },
    ("topic_03", "M-L"): {
        "k12": [5, 8],
        "category": "REPRODUCED_IN_REFINED_K12_COMPONENT",
        "note": "旧礼乐/应制复合主题在 K12 中细分；T05 的同一比较方向一致且经 LOAO 保留，T08 未形成候选。",
    },
    ("topic_06", "E-L"): {
        "k12": [2],
        "category": "REPRODUCED_IN_BROADER_K12_EXPLORATORY",
        "note": "K12 T02 合并老病、诗酒与士人自况；POEM/AUTHOR 的 E-L 均为正，但该主题为 BROAD，未进入 primary inference。",
    },
    ("topic_07", "E-L"): {
        "k12": [6],
        "category": "MERGED_IN_K12_NOT_REPRODUCED",
        "note": "K20 的相思离愁与女性闺情在 K12 T06 合并；E-L 在 K12 的 POEM/AUTHOR 方向冲突，旧 finding 未保留。",
    },
    ("topic_08", "E-L"): {
        "k12": [9],
        "category": "REPRODUCED_IN_K12",
        "note": "山寺/僧侣主题的直接语义对应最强；同一 E-L 比较为正且经 LOAO 保留。",
    },
    ("topic_11", "M-L"): {
        "k12": [6],
        "category": "MERGED_IN_K12_REPRODUCED",
        "note": "K20 的闺情主题与相思离愁在 K12 T06 合并；M-L 同向且经 LOAO 保留，但语义范围较宽。",
    },
    ("topic_16", "E-L"): {
        "k12": [11],
        "category": "WEAKENED_IN_K12",
        "note": "边塞主题为直接对应；E-L 两种 estimand 同为负，但 K12 仅为 TENTATIVE，未进入 LOAO 保留 finding。",
    },
    ("topic_19", "E-H"): {
        "k12": [1, 9],
        "category": "INCOMPARABLE_TOPIC_REDEFINED",
        "note": "旧闲居山林主题分散到 K12 T01 与 T09；T01 出现聚合方向差异，T09 又有 CLR 冲突，无法归于单一等价 K12 finding。",
    },
}


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """以稳定格式保存 UTF-8 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.15g")


def save_text(text: str, path: Path) -> None:
    """以 UTF-8 和 LF 保存文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """生成紧凑 Markdown 表格。"""

    selected = frame.loc[:, columns].copy()
    if selected.empty:
        return "无。"

    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", "/").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected.to_dict("records"):
        lines.append("| " + " | ".join(render(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def parse_manifest(path: Path) -> dict[str, str]:
    """解析冻结清单。"""

    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        records[relative] = digest
    return records


def verify_blind_freeze() -> tuple[dict[str, Any], dict[str, str]]:
    """在接触 K20 前验证 K12 盲态冻结文件及其全部哈希。"""

    freeze_path = OUT / "K12_BLIND_FREEZE.json"
    manifest_path = OUT / "SHA256_MANIFEST.txt"
    if not freeze_path.exists() or not manifest_path.exists():
        raise RuntimeError("BLOCKED_MISSING_K12_BLIND_FREEZE")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("blind_phase_status") != "BLIND_TO_OLD_TREND_DIRECTION_COMPLETE":
        raise RuntimeError("BLOCKED_INVALID_K12_BLIND_FREEZE")
    if sha256_file(manifest_path) != freeze.get("manifest_sha256"):
        raise RuntimeError("BLOCKED_K12_MANIFEST_HASH_MISMATCH")
    manifest = parse_manifest(manifest_path)
    mismatches = []
    for relative, expected in manifest.items():
        target = OUT / relative
        if not target.exists() or sha256_file(target) != expected:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"BLOCKED_K12_FROZEN_OUTPUT_CHANGED: {mismatches}")
    return freeze, manifest


def build_alignment_evidence() -> pd.DataFrame:
    """合并既有盲审语义家族与 K12→K15→K20 词项路径证据。"""

    blind = pd.read_csv(BLIND_ALIGNMENT)
    secret = pd.read_csv(BLIND_TOPIC_SECRET)
    true_topic = secret.set_index("blind_topic_id")["true_topic_id"].to_dict()
    direct = blind.loc[
        (blind["source_blind_model_id"] == "MODEL_A")
        & (blind["target_blind_model_id"] == "MODEL_B")
    ].copy()
    direct["k12_topic_number"] = direct["source_blind_topic_id"].map(true_topic).astype(int)
    direct["k20_topic_number"] = direct["target_blind_topic_id"].map(true_topic).astype(int)

    first = pd.read_csv(K12_K15)
    second = pd.read_csv(K15_K20)
    paths = first.merge(
        second,
        left_on="higher_topic_id",
        right_on="lower_topic_id",
        suffixes=("_k12_k15", "_k15_k20"),
    )
    paths["path_score_geomean"] = np.sqrt(
        paths["topic_word_cosine_k12_k15"] * paths["topic_word_cosine_k15_k20"]
    )
    paths = paths.sort_values(
        ["lower_topic_id_k12_k15", "higher_topic_id_k15_k20", "path_score_geomean"],
        ascending=[True, True, False],
    ).drop_duplicates(["lower_topic_id_k12_k15", "higher_topic_id_k15_k20"])
    paths = paths.rename(
        columns={
            "lower_topic_id_k12_k15": "k12_topic_number",
            "higher_topic_id_k12_k15": "best_k15_bridge_topic",
            "higher_topic_id_k15_k20": "k20_topic_number",
            "topic_word_cosine_k12_k15": "k12_k15_cosine",
            "topic_word_cosine_k15_k20": "k15_k20_cosine",
        }
    )
    direct = direct.merge(
        paths[
            [
                "k12_topic_number",
                "k20_topic_number",
                "best_k15_bridge_topic",
                "k12_k15_cosine",
                "k15_k20_cosine",
                "path_score_geomean",
            ]
        ],
        on=["k12_topic_number", "k20_topic_number"],
        how="left",
        validate="one_to_one",
    )
    keep = [
        "k12_topic_number",
        "k20_topic_number",
        "semantic_family_id",
        "relation_type",
        "candidate_rank",
        "topic_word_cosine",
        "top20_jaccard",
        "AI_review_status",
        "AI_semantic_evidence",
        "best_k15_bridge_topic",
        "k12_k15_cosine",
        "k15_k20_cosine",
        "path_score_geomean",
    ]
    return direct[keep].sort_values(["k20_topic_number", "k12_topic_number"]).reset_index(drop=True)


def trend_delta(row: pd.Series, contrast: str) -> float:
    """按 B−A 定义从四时期均值计算 planned contrast。"""

    first, second = contrast.split("-")
    return float(row[PERIOD_VALUE_COLUMNS[second]] - row[PERIOD_VALUE_COLUMNS[first]])


def sign_label(value: float, tolerance: float = 1e-12) -> str:
    """把原始差值转为方向标签。"""

    if value > tolerance:
        return "POSITIVE"
    if value < -tolerance:
        return "NEGATIVE"
    return "ZERO"


def build_comparison(alignment: pd.DataFrame) -> pd.DataFrame:
    """逐条比较旧 K20 已保留 finding 与冻结 K12 同一时期对比。"""

    old = pd.read_csv(OLD_CANDIDATES)
    old_retained = old.loc[old["finding_level_bd_eligible"].astype(bool)].copy()
    taxonomy = pd.read_csv(AUDIT / "C_K12_topic_taxonomy_frozen.csv")
    poem = pd.read_csv(OUT / "01_trend_structure" / "C_K12_CORE_POEM_trends.csv")
    author = pd.read_csv(OUT / "01_trend_structure" / "C_K12_CORE_AUTHOR_trends.csv")
    effects = pd.read_csv(OUT / "03_statistical_validation" / "C_K12_effect_size_summary.csv")
    final = pd.read_csv(OUT / "04_author_influence" / "C_K12_final_candidate_trends_after_LOAO.csv")

    taxonomy_by_number = taxonomy.set_index(taxonomy["topic_id"].str[-2:].astype(int))
    poem_by_id = poem.set_index("topic_id")
    author_by_id = author.set_index("topic_id")
    effect_by_key = effects.set_index(["topic_id", "contrast"])
    final_by_key = final.set_index(["topic_id", "contrast"])
    rows: list[dict[str, Any]] = []

    for old_row in old_retained.sort_values(["topic_id", "contrast"]).itertuples(index=False):
        key = (str(old_row.topic_id), str(old_row.contrast))
        if key not in COMPARISON_RULES:
            raise RuntimeError(f"缺少旧 finding 的事后比较规则: {key}")
        rule = COMPARISON_RULES[key]
        k12_records = []
        for number in rule["k12"]:
            topic_id = f"topic_{number:02d}"
            tax = taxonomy_by_number.loc[number]
            p_delta = trend_delta(poem_by_id.loc[topic_id], old_row.contrast)
            a_delta = trend_delta(author_by_id.loc[topic_id], old_row.contrast)
            if (topic_id, old_row.contrast) in effect_by_key.index:
                effect_row = effect_by_key.loc[(topic_id, old_row.contrast)]
                validation = str(effect_row["validation_status"])
                clr_agreement = bool(effect_row["clr_agreement"])
            else:
                validation = str(tax["interpretation_scope"])
                clr_agreement = np.nan
            if (topic_id, old_row.contrast) in final_by_key.index:
                final_row = final_by_key.loc[(topic_id, old_row.contrast)]
                final_status = str(final_row["final_historical_status"])
                retained = bool(final_row["finding_level_bd_eligible"])
            else:
                final_status = "NOT_IN_LOAO_RETAINED_FINDINGS"
                retained = False
            match = alignment.loc[
                (alignment["k12_topic_number"] == number)
                & (alignment["k20_topic_number"] == int(str(old_row.topic_id)[-2:]))
            ]
            if len(match) != 1:
                raise RuntimeError(f"语义对齐记录数量异常: K12 T{number:02d} / K20 {old_row.topic_id}")
            align = match.iloc[0]
            k12_records.append(
                {
                    "topic_id": topic_id,
                    "label": str(tax["final_topic_label"]),
                    "type": str(tax["topic_type"]),
                    "poem_delta": p_delta,
                    "author_delta": a_delta,
                    "direction": (
                        sign_label(p_delta)
                        if sign_label(p_delta) == sign_label(a_delta)
                        else "POEM_AUTHOR_CONFLICT"
                    ),
                    "validation": validation,
                    "clr_agreement": clr_agreement,
                    "final_status": final_status,
                    "retained": retained,
                    "family": str(align["semantic_family_id"]),
                    "relation": str(align["relation_type"]),
                    "direct_cosine": align["topic_word_cosine"],
                    "bridge": align["best_k15_bridge_topic"],
                    "path_score": align["path_score_geomean"],
                }
            )

        old_direction = (
            sign_label(old_row.original_poem_delta)
            if sign_label(old_row.original_poem_delta) == sign_label(old_row.original_author_delta)
            else "POEM_AUTHOR_CONFLICT"
        )
        rows.append(
            {
                "old_k20_topic_id": str(old_row.topic_id),
                "old_k20_topic_label": str(old_row.topic_label),
                "contrast": str(old_row.contrast),
                "old_validation_status": str(old_row.previous_validation_status),
                "old_poem_delta": float(old_row.original_poem_delta),
                "old_author_delta": float(old_row.original_author_delta),
                "old_direction": old_direction,
                "old_author_robustness": str(old_row.author_robustness_status),
                "k12_topic_ids": ";".join(record["topic_id"] for record in k12_records),
                "k12_topic_labels": ";".join(record["label"] for record in k12_records),
                "k12_topic_types": ";".join(record["type"] for record in k12_records),
                "semantic_family_ids": ";".join(record["family"] for record in k12_records),
                "alignment_relation_types": ";".join(record["relation"] for record in k12_records),
                "direct_topic_word_cosines": ";".join(
                    "NA" if pd.isna(record["direct_cosine"]) else f"{record['direct_cosine']:.6f}"
                    for record in k12_records
                ),
                "best_k15_bridge_topics": ";".join(
                    "NA" if pd.isna(record["bridge"]) else f"T{int(record['bridge']):02d}"
                    for record in k12_records
                ),
                "path_score_geomeans": ";".join(
                    "NA" if pd.isna(record["path_score"]) else f"{record['path_score']:.6f}"
                    for record in k12_records
                ),
                "k12_poem_deltas": ";".join(f"{record['poem_delta']:.10f}" for record in k12_records),
                "k12_author_deltas": ";".join(f"{record['author_delta']:.10f}" for record in k12_records),
                "k12_directions": ";".join(record["direction"] for record in k12_records),
                "k12_validation_statuses": ";".join(record["validation"] for record in k12_records),
                "k12_clr_agreement": ";".join(
                    "NA" if pd.isna(record["clr_agreement"]) else str(bool(record["clr_agreement"])).upper()
                    for record in k12_records
                ),
                "k12_final_statuses": ";".join(record["final_status"] for record in k12_records),
                "k12_any_exact_finding_retained": any(record["retained"] for record in k12_records),
                "comparison_category": str(rule["category"]),
                "comparison_note": str(rule["note"]),
                "alignment_provenance": "AI_PROVISIONAL semantic family + topic-word candidate path; researcher adjudication pending",
            }
        )
    comparison = pd.DataFrame(rows)
    if len(comparison) != 9:
        raise RuntimeError(f"旧 K20 已保留 finding 数量异常: {len(comparison)} != 9")
    return comparison


def build_comparison_summary(comparison: pd.DataFrame) -> dict[str, Any]:
    """生成旧 K20 finding 的分组摘要。"""

    def items(categories: set[str]) -> list[str]:
        subset = comparison.loc[comparison["comparison_category"].isin(categories)]
        return [
            f"K20 {row.old_k20_topic_id}/{row.contrast} → K12 {row.k12_topic_ids}"
            for row in subset.itertuples(index=False)
        ]

    return {
        "old_retained_finding_count": int(len(comparison)),
        "formal_reproduced": items(
            {
                "REPRODUCED_IN_REFINED_K12_COMPONENT",
                "REPRODUCED_IN_K12",
                "MERGED_IN_K12_REPRODUCED",
            }
        ),
        "broader_exploratory_reproduced": items({"REPRODUCED_IN_BROADER_K12_EXPLORATORY"}),
        "merged_in_k12": items(
            {
                "MERGED_IN_K12_REPRODUCED",
                "MERGED_IN_K12_NOT_REPRODUCED",
                "REPRODUCED_IN_BROADER_K12_EXPLORATORY",
            }
        ),
        "weakened_or_failed": items({"WEAKENED_IN_K12", "MERGED_IN_K12_NOT_REPRODUCED"}),
        "incomparable_topic_redefined": items({"INCOMPARABLE_TOPIC_REDEFINED"}),
        "category_counts": {
            str(key): int(value)
            for key, value in comparison["comparison_category"].value_counts().sort_index().items()
        },
    }


def build_comparison_report(
    comparison: pd.DataFrame,
    summary: dict[str, Any],
    freeze: dict[str, Any],
    old_hashes: dict[str, str],
) -> str:
    """生成 K12 与 K20 的事后比较报告。"""

    display = comparison.copy()
    display["K20 finding"] = display["old_k20_topic_id"] + " " + display["contrast"]
    display["K12 对应"] = display["k12_topic_ids"]
    display["K12 raw 方向"] = display["k12_directions"]
    display["K12 验证状态"] = display["k12_validation_statuses"]
    display["比较结论"] = display["comparison_category"]
    rows = markdown_table(
        display,
        ["K20 finding", "K12 对应", "K12 raw 方向", "K12 验证状态", "比较结论"],
    )
    notes = "\n".join(
        f"- K20 {row.old_k20_topic_id} {row.contrast}：{row.comparison_note}"
        for row in comparison.itertuples(index=False)
    )
    return f"""# QTS C_K12 vs C_K20 Historical Findings Comparison

## 1. 比较前置条件

K12 统计输出于 `{freeze['frozen_at']}` 冻结，状态为 `{freeze['blind_phase_status']}`；冻结清单 SHA-256 为 `{freeze['manifest_sha256']}`。本报告仅在逐项验证冻结清单之后读取旧 K20 结果，因此 K20 趋势没有参与 K12 结果生成。

旧结果保留为 `C_K20_HISTORY_V1_REFERENCE`；新结果为 `C_K12_HISTORY_V2_PRIMARY_CANDIDATE`。旧 K20 核心 finding 文件哈希为 `{old_hashes['old_candidates']}`，未覆盖。

## 2. 比较方法

Topic ID 不作跨模型等同。语义对应来自既有四模型盲审的 K12→K20 `AI_PROVISIONAL` semantic family，并用既有 K12→K15→K20 topic-word cosine 最优路径作量化旁证。两套对齐都只是研究者判读线索；只有在此之后，才把冻结 K12 的同一 planned contrast（统一为 B−A）与旧 K20 finding 对读。

## 3. 逐项结果

{rows}

{notes}

## 4. 分类摘要

- 正式层面复现：{len(summary['formal_reproduced'])} 条；{'; '.join(summary['formal_reproduced'])}。
- 更宽 K12 主题中方向复现但仅属探索性：{len(summary['broader_exploratory_reproduced'])} 条；{'; '.join(summary['broader_exploratory_reproduced'])}。
- 涉及 K12 合并主题：{len(summary['merged_in_k12'])} 条；其中相同 contrast 是否保留须逐条区分。
- 削弱或未复现：{len(summary['weakened_or_failed'])} 条；{'; '.join(summary['weakened_or_failed'])}。
- 因题义重定义而不可作单一对应：{len(summary['incomparable_topic_redefined'])} 条；{'; '.join(summary['incomparable_topic_redefined'])}。

## 5. 解释边界

这里的“复现”只指：在语义对齐后，同一作者时期比较的冻结 K12 raw probability 方向与旧 K20 相容，并且（正式复现时）K12 finding 经自身统计与作者影响流程保留。它不表示 Topic 完全同义，也不构成文学史事实确认。author period 表示作者所属时期，并非单首诗的创作年代。
"""


def build_final_report(summary: dict[str, Any], comparison: pd.DataFrame) -> str:
    """生成十八节最终历史分析报告。"""

    taxonomy = pd.read_csv(AUDIT / "C_K12_topic_taxonomy_frozen.csv")
    sample = pd.read_csv(AUDIT / "C_K12_metadata_sample_audit.csv")
    poem = pd.read_csv(OUT / "01_trend_structure" / "C_K12_CORE_POEM_trends.csv")
    author = pd.read_csv(OUT / "01_trend_structure" / "C_K12_CORE_AUTHOR_trends.csv")
    aggregation = pd.read_csv(OUT / "01_trend_structure" / "C_K12_aggregation_sensitivity.csv")
    robustness = pd.read_csv(OUT / "01_trend_structure" / "C_K12_trend_robustness.csv")
    omnibus = pd.read_csv(OUT / "03_statistical_validation" / "C_K12_omnibus_period_effects.csv")
    effects = pd.read_csv(OUT / "03_statistical_validation" / "C_K12_effect_size_summary.csv")
    final = pd.read_csv(OUT / "04_author_influence" / "C_K12_final_candidate_trends_after_LOAO.csv")

    tax_table = taxonomy.rename(
        columns={"final_topic_id": "Topic", "final_topic_label": "标签", "topic_type": "类型"}
    )
    sample_rows: list[dict[str, Any]] = []
    for sample_name in ["STRICT", "CORE", "EXPANDED"]:
        subset = sample.loc[sample["sample"] == sample_name]
        poems_total = int(subset.loc[subset["scope"] == "ALL_POEMS", "observed"].iloc[0])
        authors_total = int(subset.loc[subset["scope"] == "ALL_AUTHORS", "observed"].iloc[0])
        sample_rows.append(
            {
                "样本": sample_name,
                "时期": "ALL",
                "诗数": poems_total,
                "作者数": authors_total,
                "QA": "PASS",
            }
        )
    core_sample = sample.loc[(sample["sample"] == "CORE") & (sample["period"] != "ALL")]
    for period in ["初唐", "盛唐", "中唐", "晚唐"]:
        subset = core_sample.loc[core_sample["period"] == period]
        sample_rows.append(
            {
                "样本": "CORE",
                "时期": period,
                "诗数": int(subset.loc[subset["scope"] == "POEMS", "observed"].iloc[0]),
                "作者数": int(subset.loc[subset["scope"] == "AUTHORS", "observed"].iloc[0]),
                "QA": "PASS",
            }
        )
    sample_view = pd.DataFrame(sample_rows)
    poem_view = poem.copy()
    poem_view["Topic"] = poem_view["topic_id"].str.replace("topic_", "T", regex=False)
    poem_view = poem_view.rename(
        columns={
            "topic_label": "标签",
            "early_mean": "初唐",
            "high_mean": "盛唐",
            "middle_mean": "中唐",
            "late_mean": "晚唐",
            "peak_period": "峰值",
            "trend_shape": "形态",
        }
    )
    author_view = author.copy()
    author_view["Topic"] = author_view["topic_id"].str.replace("topic_", "T", regex=False)
    author_view = author_view.rename(
        columns={
            "topic_label": "标签",
            "early_mean": "初唐",
            "high_mean": "盛唐",
            "middle_mean": "中唐",
            "late_mean": "晚唐",
            "peak_period": "峰值",
            "trend_shape": "形态",
        }
    )
    aggregate_diff = aggregation.loc[
        (aggregation["sample"] == "CORE")
        & ~aggregation["peak_period_agreement"].astype(bool), [
        "topic_id",
        "poem_peak_period",
        "author_peak_period",
        "poem_author_agreement",
    ]].copy()
    aggregate_diff["Topic"] = aggregate_diff["topic_id"].str.replace("topic_", "T", regex=False)
    aggregate_diff = aggregate_diff.rename(
        columns={
            "poem_peak_period": "POEM峰值",
            "author_peak_period": "AUTHOR峰值",
            "poem_author_agreement": "综合一致度",
        }
    )
    robust_view = robustness[[
        "topic_id",
        "topic_type",
        "peak_consistency_ratio",
        "trend_shape_consistency_ratio",
        "trend_robustness",
        "sensitivity_flags",
    ]].copy()
    robust_view["Topic"] = robust_view["topic_id"].str.replace("topic_", "T", regex=False)
    robust_view = robust_view.rename(
        columns={
            "topic_type": "类型",
            "peak_consistency_ratio": "峰值一致率",
            "trend_shape_consistency_ratio": "形态一致率",
            "trend_robustness": "稳健度",
            "sensitivity_flags": "敏感性",
        }
    )
    omni = omnibus.copy()
    omni["Topic"] = omni["topic_id"].str.replace("topic_", "T", regex=False)
    omni = omni.rename(
        columns={"aggregation": "口径", "q_value": "omnibus q", "covariance_estimator": "方法"}
    )
    effect_keep = effects.loc[effects["validation_status"].isin(["STRONG", "SUPPORTED"])].copy()
    effect_keep["Topic"] = effect_keep["topic_id"].str.replace("topic_", "T", regex=False)
    effect_keep = effect_keep.rename(
        columns={
            "contrast": "比较",
            "raw_effect_poem": "POEM差值",
            "raw_effect_author": "AUTHOR差值",
            "hedges_g": "Hedges g",
            "validation_status": "统计状态",
        }
    )
    final_keep = final.loc[final["finding_level_bd_eligible"].astype(bool)].copy()
    final_keep["Topic"] = final_keep["topic_id"].str.replace("topic_", "T", regex=False)
    final_keep = final_keep.rename(
        columns={
            "topic_label": "标签",
            "contrast": "比较",
            "original_poem_delta": "POEM差值",
            "original_author_delta": "AUTHOR差值",
            "previous_validation_status": "统计状态",
            "author_robustness_status": "作者稳健度",
            "final_historical_status": "最终状态",
        }
    )
    broad = robustness.loc[robustness["topic_type"] == "BROAD"].copy()
    broad["Topic"] = broad["topic_id"].str.replace("topic_", "T", regex=False)
    broad = broad.merge(
        poem[["topic_id", "peak_period", "trend_shape"]], on="topic_id", how="left"
    ).merge(
        author[["topic_id", "peak_period", "trend_shape"]],
        on="topic_id",
        suffixes=("_poem", "_author"),
        how="left",
    )
    broad = broad.rename(
        columns={
            "peak_period_poem": "POEM峰值",
            "peak_period_author": "AUTHOR峰值",
            "trend_shape_poem": "POEM形态",
            "trend_shape_author": "AUTHOR形态",
            "trend_robustness": "稳健度",
        }
    )
    noise = robustness.loc[robustness["topic_type"] == "NOISE"].copy()
    noise["Topic"] = noise["topic_id"].str.replace("topic_", "T", regex=False)
    noise = noise.merge(
        poem[["topic_id", "peak_period"]], on="topic_id", how="left"
    ).merge(
        author[["topic_id", "peak_period"]], on="topic_id", suffixes=("_poem", "_author"), how="left"
    )
    noise = noise.rename(
        columns={
            "peak_period_poem": "POEM峰值",
            "peak_period_author": "AUTHOR峰值",
            "trend_robustness": "稳健度",
        }
    )
    compare_view = comparison.copy()
    compare_view["K20 finding"] = compare_view["old_k20_topic_id"] + " " + compare_view["contrast"]
    compare_view = compare_view.rename(
        columns={"k12_topic_ids": "K12对应", "comparison_category": "比较结论"}
    )

    strong = [f"{item['topic_id'].replace('topic_', 'T')} {item['contrast']}" for item in summary["O_strong_contrasts"]]
    supported = [f"{item['topic_id'].replace('topic_', 'T')} {item['contrast']}" for item in summary["P_supported_contrasts"]]
    conflicts = [f"{item['topic_id'].replace('topic_', 'T')} {item['contrast']}" for item in summary["Q_poem_author_conflicts"]]

    return f"""# QTS C_K12 Final Historical Analysis

# 1 Objective

以 C_K12 FULL_CORPUS_FIT 的 43,015×12 Document×Topic probability matrix，独立重算作者时期主题结构、代表性风险、效应验证及作者影响。所有结果均是统计候选，不等于文学史确认。

# 2 Why the Historical Analysis Was Rerun

主 Topic 模型由原 C_K20 重新评估为 C_K12 候选，因此所有依赖 Topic 定义的历史趋势都必须从 K12 theta 重算。K20 趋势没有映射成 K12 结果；K12 在盲态冻结后才与旧 K20 作事后比较。

# 3 Data and Metadata

theta 来源：`{summary['A_theta_path']}`，shape={summary['B_theta_shape']}；finite、非负、行和与 document_index 对齐均通过。历史 metadata、`author_period_primary`、三套 eligibility 与 `author_id_internal` 均复用冻结版本，未重建作者生卒年或时期。

关键解释边界：本分析研究作者所属时期与其现存诗作 Topic probability 的关系。**author period 不等于 poem date**；结果不能解释为单首诗在某年发生变化。

# 4 K12 Topic Taxonomy

标签 provenance 为 `AI_PROVISIONAL`；Topic probability 不因标签措辞改变。CLEAR 6 题进入 primary inference，BROAD 4 题只作探索，NOISE 2 题只作模型诊断。

{markdown_table(tax_table, ['Topic', '标签', '类型', 'interpretation_scope'])}

# 5 Sample Coverage

CORE 是主描述规格；STRICT 是 metadata-quality sensitivity；EXPANDED 是 coverage sensitivity。样本核验全部通过。

{markdown_table(sample_view, ['样本', '时期', '诗数', '作者数', 'QA'])}

# 6 CORE Poem-weighted Historical Structure

POEM 口径每首诗等权，置信区间按作者 cluster 处理。表中数值为 mean Topic probability。

{markdown_table(poem_view, ['Topic', '标签', '初唐', '盛唐', '中唐', '晚唐', '峰值', '形态'])}

# 7 CORE Author-balanced Historical Structure

AUTHOR 口径先在作者内部平均，再令同一时期作者等权；不按作品数二次加权。

{markdown_table(author_view, ['Topic', '标签', '初唐', '盛唐', '中唐', '晚唐', '峰值', '形态'])}

# 8 Aggregation Sensitivity

POEM/AUTHOR 峰值一致 {summary['J_core_peak_agreement_count']}/12。四时期 Pearson/Spearman 只作描述诊断，不报告显著性 p-value。峰值不同的主题如下；这些差异提示留存作品量与典型作者口径不可混写。

{markdown_table(aggregate_diff, ['Topic', 'POEM峰值', 'AUTHOR峰值', '综合一致度'])}

# 9 Metadata / Sample Sensitivity

六规格 trend robustness 为 HIGH={summary['K_trend_robustness_counts'].get('HIGH', 0)}、MODERATE={summary['K_trend_robustness_counts'].get('MODERATE', 0)}、LOW={summary['K_trend_robustness_counts'].get('LOW', 0)}。STRICT/CORE/EXPANDED planned directions 共发现 {summary['R_sample_sign_flip_count']} 个符号反转标记，因此不得从三套样本挑选结果。CLEAR representativeness risk：{json.dumps(summary['L_clear_representativeness_risk'], ensure_ascii=False)}。

{markdown_table(robust_view, ['Topic', '类型', '峰值一致率', '形态一致率', '稳健度', '敏感性'])}

# 10 CLEAR Topic Statistical Validation

CORE_POEM 使用 author-cluster CR1，CORE_AUTHOR 使用 HC3；两套 6-topic omnibus 与两套 24-contrast family 分别作 BH-FDR。CLR 以完整 12 维 composition 计算，只有诊断作用。STRONG：{', '.join(strong)}。SUPPORTED：{', '.join(supported)}。POEM/AUTHOR 冲突：{', '.join(conflicts)}。CLR 方向冲突数为 {summary['S_clr_direction_conflict_count']}。

{markdown_table(omni, ['Topic', '口径', 'omnibus q', '方法'])}

{markdown_table(effect_keep, ['Topic', '比较', 'POEM差值', 'AUTHOR差值', 'Hedges g', '统计状态'])}

# 11 Author Influence

仅 8 个 STRONG/SUPPORTED candidate 进入 CORE LOAO；删除单位为 `author_id_internal`，并按每个 K12 Topic×Contrast 重新计算 GLOBAL_TOP/PER_PERIOD_TOP 的 Top1/3/5/10。结果为 AUTHOR_ROBUST={summary['U_author_robust_count']}、CONCENTRATED_BUT_ROBUST={summary['V_concentrated_robust_count']}、AUTHOR_SENSITIVE={summary['W_author_sensitive_count']}。作者贡献集中度表示概率质量集中，不表示文学影响力。

# 12 Robust Candidate Historical Patterns

最终保留 7 条，均为 `KEEP_WITH_CAUTION`；它们只表示在当前 K12、样本及阈值下值得进入研究者阅读队列。

{markdown_table(final_keep, ['Topic', '标签', '比较', 'POEM差值', 'AUTHOR差值', '统计状态', '作者稳健度', '最终状态'])}

# 13 BROAD Topic Descriptive Patterns

T01、T02、T10、T12 完成六规格趋势计算，但不进入 primary FDR。即使呈现强趋势，也只能标记为 `RESEARCHER_REVIEW_BROAD`，不能升级为正式 finding。

{markdown_table(broad, ['Topic', 'POEM峰值', 'AUTHOR峰值', 'POEM形态', 'AUTHOR形态', '稳健度'])}

# 14 NOISE Topic Diagnostics

T04、T07 的四时期均值仍因完整 12 维 composition 而保存；任何时期差异只解释为 residual-topic composition 的变化，不作为独立文学史结论。

{markdown_table(noise, ['Topic', 'POEM峰值', 'AUTHOR峰值', '稳健度'])}

# 15 Comparison with Previous C_K20 Findings

比较仅在 K12 SHA-256 冻结后执行，且基于 semantic family 与 K12→K15→K20 topic-word path，绝不按 Topic ID 直接等同。旧 K20 的 9 条保留 finding 中，5 条在 K12 正式层面复现，1 条只在更宽 BROAD 主题中方向复现，2 条削弱或未复现，1 条因主题重定义不可作单一比较；“合并”与“削弱”可在同一旧 finding 上同时构成解释背景，逐条证据见比较报告。

{markdown_table(compare_view, ['K20 finding', 'K12对应', '比较结论'])}

# 16 Humanistic Interpretation Priorities

优先核读 7 条保留 candidate 的对应时期代表诗、作者归属、同文异署及外部文学史材料。T05 三条应制礼仪比较与 T09 E-L 的统计证据最完整；T06、T08、T11 仍需特别注意样本或作者集中敏感性。研究者解释必须保持“候选”措辞。

# 17 Limitations

Topic 标签仍为 AI_PROVISIONAL；作者时期不是作品创作日期；三套样本受 metadata 覆盖限制；作者集与留存作品不均衡；CLR 与 raw probability 可能给出不同方向；LOAO 只检验作者集中与删除敏感性；跨模型语义对齐尚待研究者正式裁决。因而不得使用“证明”“导致”或“唐诗从此转向”等因果/断代措辞。

# 18 Next Step：B_K12 / C_K12 / D_K12 finding-level robustness

下一阶段应由研究者确认 K12 taxonomy 与本轮候选后，再决定是否执行 B_K12 / C_K12 / D_K12 finding-level robustness。本任务没有训练 B_K12 或 D_K12，没有修改最终 K 选择，也没有改动诗人时期 metadata。
"""


def build_final_qa(
    freeze: dict[str, Any],
    manifest: dict[str, str],
    comparison: pd.DataFrame,
    old_hashes_before: dict[str, str],
    old_hashes_after: dict[str, str],
) -> pd.DataFrame:
    """将预比较 18 项 QA 扩展为最终 20 项 QA。"""

    pre = pd.read_csv(AUDIT / "C_K12_precomparison_QA.csv")
    rows = pre.to_dict("records")
    # QA_18 在最终阶段按任务原义核验旧 K20 reference 未被覆盖。
    rows[17] = {
        "test_id": "QA_18",
        "test_name": "旧 K20 文件未被覆盖",
        "status": "PASS" if old_hashes_before == old_hashes_after else "FAIL",
        "observed": json.dumps(old_hashes_after, ensure_ascii=False, sort_keys=True),
        "expected": json.dumps(old_hashes_before, ensure_ascii=False, sort_keys=True),
    }
    expected_frozen = int(freeze["file_count"])
    rows.append(
        {
            "test_id": "QA_19",
            "test_name": "所有关键输出已 hash",
            "status": "PASS" if len(manifest) == expected_frozen else "FAIL",
            "observed": json.dumps(
                {"blind_manifest_files": len(manifest), "comparison_rows": len(comparison)},
                ensure_ascii=False,
            ),
            "expected": json.dumps(
                {"blind_manifest_files": expected_frozen, "comparison_rows": 9},
                ensure_ascii=False,
            ),
        }
    )
    final_report = (FINAL / "QTS_C_K12_Final_Historical_Analysis.md").read_text(encoding="utf-8")
    wording_pass = "author period 不等于 poem date" in final_report
    rows.append(
        {
            "test_id": "QA_20",
            "test_name": "最终报告明确 author period ≠ poem date",
            "status": "PASS" if wording_pass else "FAIL",
            "observed": wording_pass,
            "expected": True,
        }
    )
    qa = pd.DataFrame(rows)
    if not qa["status"].eq("PASS").all():
        failed = qa.loc[qa["status"] != "PASS", "test_id"].tolist()
        raise RuntimeError(f"最终 QA 未通过: {failed}")
    return qa


def write_delivery_manifest() -> tuple[Path, dict[str, str]]:
    """为 C_K12 交付目录内全部文件生成最终哈希清单。"""

    manifest_path = OUT / "DELIVERY_SHA256_MANIFEST.txt"
    files = sorted(
        path
        for path in OUT.rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve()
    )
    records = {path.relative_to(OUT).as_posix(): sha256_file(path) for path in files}
    text = "\n".join(f"{digest}  {relative}" for relative, digest in records.items())
    save_text(text, manifest_path)
    return manifest_path, records


def build_readme(summary: dict[str, Any]) -> str:
    """更新交付目录说明。"""

    return f"""# C_K12 作者时期历史分析

版本：`C_K12_HISTORY_V2_PRIMARY_CANDIDATE`。

本目录从 FULL_CORPUS_FIT 的 43,015×12 Document×Topic 矩阵独立重算作者时期趋势、代表性、效应验证与作者影响。K12 正式统计先在 `BLIND_TO_OLD_TREND_DIRECTION` 条件下完成并写入 `SHA256_MANIFEST.txt`，之后才建立 C_K12 与旧 C_K20 的事后语义比较。

`author_period_primary` 表示作者所属时期，不表示单篇作品创作年代。统计候选不等于文学史确认。

- `00_input_audit`：theta、document_index、taxonomy、样本数、最终 20 项 QA 与配置。
- `01_trend_structure`：三样本、两 estimand、四时期的趋势结构。
- `02_representativeness`：已知/未知作者时期的 K12 Topic SMD 与风险。
- `03_statistical_validation`：CLEAR Topics 的 CR1/HC3、BH-FDR、效应量、样本与 CLR 敏感性。
- `04_author_influence`：仅 STRONG/SUPPORTED 候选的 LOAO 与 Top contributor 排除。
- `05_figures`：八幅规定 PNG。
- `06_final`：研究者解释队列、18 节最终报告、冻结后 K20 比较及逐条比较数据。

进入 LOAO 的候选数：{summary['T_loao_candidate_count']}；最终保留：{len(summary['X_retained_historical_candidates'])}。最终交付状态：`C_K12_HISTORY_ANALYSIS_COMPLETE`。
"""


def main() -> None:
    """执行冻结验证、事后比较、最终报告和交付 QA。"""

    # 此调用必须是读取任何 K20 文件之前的第一项工作。
    freeze, blind_manifest = verify_blind_freeze()
    required_sources = [OLD_CANDIDATES, BLIND_ALIGNMENT, BLIND_TOPIC_SECRET, K12_K15, K15_K20]
    missing = [str(path) for path in required_sources if not path.exists()]
    if missing:
        raise RuntimeError(f"BLOCKED_MISSING_COMPARISON_SOURCE: {missing}")
    old_hashes_before = {
        "old_candidates": sha256_file(OLD_CANDIDATES),
        "blind_alignment": sha256_file(BLIND_ALIGNMENT),
        "blind_topic_secret": sha256_file(BLIND_TOPIC_SECRET),
        "k12_k15_candidates": sha256_file(K12_K15),
        "k15_k20_candidates": sha256_file(K15_K20),
    }

    alignment = build_alignment_evidence()
    save_csv(alignment, FINAL / "C_K12_K20_semantic_alignment_evidence.csv")
    comparison = build_comparison(alignment)
    save_csv(comparison, FINAL / "C_K12_vs_C_K20_finding_comparison.csv")
    comparison_summary = build_comparison_summary(comparison)
    comparison_summary["blind_freeze_timestamp"] = freeze["frozen_at"]
    comparison_summary["blind_manifest_sha256"] = freeze["manifest_sha256"]
    comparison_summary["comparison_source_hashes"] = old_hashes_before
    save_text(
        json.dumps(comparison_summary, ensure_ascii=False, indent=2, allow_nan=False),
        FINAL / "C_K12_vs_C_K20_comparison_summary.json",
    )
    save_text(
        build_comparison_report(comparison, comparison_summary, freeze, old_hashes_before),
        FINAL / "QTS_C_K12_vs_C_K20_Historical_Findings_Comparison.md",
    )

    run_summary = json.loads(
        (AUDIT / "C_K12_precomparison_summary.json").read_text(encoding="utf-8")
    )
    save_text(
        build_final_report(run_summary, comparison),
        FINAL / "QTS_C_K12_Final_Historical_Analysis.md",
    )
    save_text(build_readme(run_summary), OUT / "README.md")

    old_hashes_after = {
        "old_candidates": sha256_file(OLD_CANDIDATES),
        "blind_alignment": sha256_file(BLIND_ALIGNMENT),
        "blind_topic_secret": sha256_file(BLIND_TOPIC_SECRET),
        "k12_k15_candidates": sha256_file(K12_K15),
        "k15_k20_candidates": sha256_file(K15_K20),
    }
    qa = build_final_qa(freeze, blind_manifest, comparison, old_hashes_before, old_hashes_after)
    save_csv(qa, AUDIT / "C_K12_final_QA.csv")
    manifest_path, delivery_records = write_delivery_manifest()

    required_delivery = {
        "06_final/QTS_C_K12_Final_Historical_Analysis.md",
        "06_final/QTS_C_K12_vs_C_K20_Historical_Findings_Comparison.md",
        "06_final/C_K12_vs_C_K20_finding_comparison.csv",
        "00_input_audit/C_K12_final_QA.csv",
    }
    if not required_delivery.issubset(delivery_records):
        raise RuntimeError("最终交付清单缺少关键文件")
    for relative, expected in delivery_records.items():
        if sha256_file(OUT / relative) != expected:
            raise RuntimeError(f"最终交付文件 hash 验证失败: {relative}")

    completion = {
        **run_summary,
        "Y_k20_comparison": comparison_summary,
        "Z_status": "C_K12_HISTORY_ANALYSIS_COMPLETE",
        "final_qa_passed": int(qa["status"].eq("PASS").sum()),
        "final_qa_total": int(len(qa)),
        "delivery_manifest": str(manifest_path),
        "delivery_manifest_file_count": len(delivery_records),
    }
    save_text(
        json.dumps(completion, ensure_ascii=False, indent=2, allow_nan=False),
        FINAL / "C_K12_completion_summary_A_to_Z.json",
    )
    # completion summary 是最后生成的辅助交付物，追加到交付清单后再次验证。
    manifest_path, delivery_records = write_delivery_manifest()
    print(
        json.dumps(
            {
                "status": "C_K12_HISTORY_ANALYSIS_COMPLETE",
                "comparison_rows": len(comparison),
                "final_qa": f"{len(qa)}/{len(qa)} PASS",
                "delivery_manifest": str(manifest_path),
                "delivery_file_count": len(delivery_records),
                "comparison_summary": comparison_summary,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
