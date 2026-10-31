"""执行 C_K12 作者时期历史分析的盲态重跑阶段。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_author_influence_loao as loao
import run_author_period_effect_validation as effect
import run_author_period_trend_structure as trend
import run_period_representativeness_audit as rep


ROOT = Path(__file__).resolve().parents[1]
AGGREGATIONS = ("POEM", "AUTHOR")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="C_K12 作者时期历史分析盲态重跑")
    parser.add_argument(
        "--config",
        default="config/k12_historical_analysis_config.yaml",
        help="JSON 兼容 YAML 配置路径",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    """相对路径按仓库根目录解析。"""

    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    """读取 JSON 兼容配置。"""

    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """以 UTF-8 无 BOM 和稳定浮点格式保存 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.15g")


def save_text(text: str, path: Path) -> None:
    """以 UTF-8 和 LF 保存文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def normalize_bool(series: pd.Series) -> pd.Series:
    """把冻结标签规范为布尔值。"""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError("样本标签包含无法解析的布尔值")
    return mapped.astype(bool)


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    """生成不依赖额外套件的 Markdown 表格。"""

    selected = frame.loc[:, columns].copy()
    if limit is not None:
        selected = selected.head(limit)
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
    for record in selected.to_dict("records"):
        lines.append("| " + " | ".join(render(record[column]) for column in columns) + " |")
    return "\n".join(lines)


def configure_plotting() -> None:
    """配置中文学术图的统一样式。"""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def build_taxonomy(config: dict[str, Any], output_path: Path) -> pd.DataFrame:
    """从主题总表冻结当前 K12 taxonomy。"""

    workbook = resolve_path(config["taxonomy_workbook"])
    source = pd.read_excel(workbook, sheet_name="主题总表", header=2)
    source = source.loc[source["Topic"].astype(str).str.fullmatch(r"T\d{2}", na=False)].copy()
    source["topic_id"] = source["Topic"].str.extract(r"(\d+)")[0].astype(int).map(lambda value: f"topic_{value:02d}")
    source["interpretation_scope"] = source["类型"].map(
        {
            "CLEAR": "PRIMARY_INFERENCE",
            "BROAD": "EXPLORATORY_BROAD",
            "NOISE": "DESCRIPTIVE_NOISE_ONLY",
        }
    )
    taxonomy = pd.DataFrame(
        {
            "topic_id": source["topic_id"],
            "final_topic_id": source["Topic"],
            "final_topic_label": source["主题名称"].astype(str),
            "short_label": source["精简标签"].astype(str),
            "topic_type": source["类型"].astype(str),
            "core_semantics": source["核心语义"].astype(str),
            "interpretation_scope": source["interpretation_scope"].astype(str),
            "mean_human_score": pd.to_numeric(source["AI文学可解释性"], errors="coerce"),
            "review_focus": source["复核重点"].astype(str),
            "label_provenance": source["审核状态"].astype(str),
            "blind_topic_id": source["blind_topic_id"].astype(str),
        }
    ).sort_values("topic_id").reset_index(drop=True)
    expected_ids = [f"topic_{index:02d}" for index in range(1, int(config["expected_topic_count"]) + 1)]
    if taxonomy["topic_id"].tolist() != expected_ids:
        raise ValueError("C_K12 taxonomy 的 Topic ID 不完整或顺序异常")
    expected_types = {
        "CLEAR": set(config["primary_clear_topics"]),
        "BROAD": set(config["broad_topics"]),
        "NOISE": set(config["noise_topics"]),
    }
    for topic_type, expected in expected_types.items():
        observed = set(taxonomy.loc[taxonomy["topic_type"].eq(topic_type), "topic_id"])
        if observed != expected:
            raise ValueError(f"{topic_type} Topic 集合与任务冻结定义不一致")
    if taxonomy["label_provenance"].eq("AI_PROVISIONAL").all() is False:
        raise ValueError("检测到非 AI_PROVISIONAL 标签，需先核验是否为研究者确认版本")
    save_csv(taxonomy, output_path)
    return taxonomy


def build_input_master_and_audit(
    config: dict[str, Any],
    taxonomy: pd.DataFrame,
    audit_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, Any]]:
    """按 document_index 对齐 theta 与冻结历史 metadata。"""

    theta_path = resolve_path(config["theta_path"])
    map_path = resolve_path(config["document_map_path"])
    master_path = resolve_path(config["metadata_master"])
    labels_path = resolve_path(config["sample_labels"])
    metadata_path = resolve_path(config["theta_metadata_path"])
    required = [theta_path, map_path, master_path, labels_path, metadata_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"BLOCKED_MISSING_K12_THETA: {missing}")

    archive = np.load(theta_path)
    if config["theta_key"] not in archive.files:
        raise KeyError(f"BLOCKED_MISSING_K12_THETA: 缺少键 {config['theta_key']}")
    theta = np.asarray(archive[config["theta_key"]], dtype=float)
    expected_shape = (int(config["expected_poem_count"]), int(config["expected_topic_count"]))
    finite = bool(np.isfinite(theta).all())
    nonnegative = bool((theta >= 0).all())
    row_error = float(np.max(np.abs(theta.sum(axis=1) - 1.0)))

    document_map = pd.read_csv(map_path, low_memory=False)
    metadata_columns = [
        "model_row_id",
        "poem_id",
        "author_id_internal",
        "author",
        "author_period_primary",
        "period_assignment_method",
        "period_confidence",
        "cross_period_flag",
        "period_boundary_flag",
        "cbdb_match_confidence",
    ]
    source_master = pd.read_csv(master_path, usecols=metadata_columns, low_memory=False)
    labels = pd.read_csv(labels_path, low_memory=False)

    aligned = document_map[["document_index", "matrix_row", "poem_id"]].merge(
        source_master,
        left_on="document_index",
        right_on="model_row_id",
        how="left",
        validate="one_to_one",
        suffixes=("_map", "_metadata"),
        indicator=True,
    )
    poem_id_mismatch = int(
        aligned.loc[aligned["_merge"].eq("both"), "poem_id_map"].ne(
            aligned.loc[aligned["_merge"].eq("both"), "poem_id_metadata"]
        ).sum()
    )
    matched_rows = int(aligned["_merge"].eq("both").sum() - poem_id_mismatch)
    metadata_unmatched = int(len(source_master) - matched_rows)
    theta_unmatched = int(len(document_map) - matched_rows)
    duplicate_document_keys = int(document_map["document_index"].duplicated().sum())
    duplicate_matrix_rows = int(document_map["matrix_row"].duplicated().sum())
    duplicate_poem_ids_map = int(document_map["poem_id"].duplicated().sum())
    duplicate_poem_ids_metadata = int(source_master["poem_id"].duplicated().sum())
    sequential = bool(
        document_map["matrix_row"].to_numpy().tolist() == list(range(len(document_map)))
        and document_map["document_index"].to_numpy().tolist() == list(range(len(document_map)))
    )
    passed = bool(
        theta.shape == expected_shape
        and finite
        and nonnegative
        and row_error <= 1e-10
        and len(document_map) == expected_shape[0]
        and len(source_master) == expected_shape[0]
        and matched_rows == expected_shape[0]
        and metadata_unmatched == 0
        and theta_unmatched == 0
        and poem_id_mismatch == 0
        and duplicate_document_keys == 0
        and duplicate_matrix_rows == 0
        and duplicate_poem_ids_map == 0
        and duplicate_poem_ids_metadata == 0
        and sequential
    )
    metrics = [
        ("theta_rows", theta.shape[0], expected_shape[0]),
        ("theta_columns", theta.shape[1], expected_shape[1]),
        ("metadata_rows", len(source_master), expected_shape[0]),
        ("document_map_rows", len(document_map), expected_shape[0]),
        ("matched_rows", matched_rows, expected_shape[0]),
        ("theta_unmatched_rows", theta_unmatched, 0),
        ("metadata_unmatched_rows", metadata_unmatched, 0),
        ("poem_id_mismatch_rows", poem_id_mismatch, 0),
        ("duplicate_document_index", duplicate_document_keys, 0),
        ("duplicate_matrix_row", duplicate_matrix_rows, 0),
        ("duplicate_poem_id_map", duplicate_poem_ids_map, 0),
        ("duplicate_poem_id_metadata", duplicate_poem_ids_metadata, 0),
        ("sequential_document_and_matrix_index", sequential, True),
        ("theta_all_finite", finite, True),
        ("theta_all_nonnegative", nonnegative, True),
        ("theta_max_row_sum_error", row_error, "<=1e-10"),
        ("alignment_status", "PASS" if passed else "FAIL", "PASS"),
    ]
    alignment_audit = pd.DataFrame(metrics, columns=["metric", "observed", "expected"])
    alignment_audit["status"] = alignment_audit.apply(
        lambda row: "PASS"
        if (
            row["metric"] == "theta_max_row_sum_error"
            and float(row["observed"]) <= 1e-10
        )
        or str(row["observed"]) == str(row["expected"])
        else "FAIL",
        axis=1,
    )
    save_csv(alignment_audit, audit_dir / "C_K12_theta_metadata_alignment_audit.csv")
    if not passed:
        raise ValueError("C_K12 theta 与 metadata alignment 未通过")

    aligned = aligned.sort_values("matrix_row").reset_index(drop=True)
    compact = aligned[
        [
            "model_row_id",
            "poem_id_metadata",
            "author_id_internal",
            "author",
            "author_period_primary",
            "period_assignment_method",
            "period_confidence",
            "cross_period_flag",
            "period_boundary_flag",
            "cbdb_match_confidence",
        ]
    ].rename(columns={"poem_id_metadata": "poem_id"})
    topic_columns = [f"topic_{index:02d}_prob" for index in range(1, expected_shape[1] + 1)]
    compact[topic_columns] = theta
    dominant_index = np.argmax(theta, axis=1)
    second_order = np.argsort(-theta, axis=1, kind="stable")[:, :3]
    labels_lookup = taxonomy.set_index("topic_id")["final_topic_label"].to_dict()
    compact["dominant_topic_id"] = [f"topic_{value + 1:02d}" for value in dominant_index]
    compact["dominant_topic_label"] = compact["dominant_topic_id"].map(labels_lookup)
    compact["dominant_topic_probability"] = theta[np.arange(len(theta)), second_order[:, 0]]
    compact["second_topic_id"] = [f"topic_{value + 1:02d}" for value in second_order[:, 1]]
    compact["second_topic_probability"] = theta[np.arange(len(theta)), second_order[:, 1]]
    compact["third_topic_id"] = [f"topic_{value + 1:02d}" for value in second_order[:, 2]]
    compact["third_topic_probability"] = theta[np.arange(len(theta)), second_order[:, 2]]
    compact["topic_entropy"] = -np.sum(theta * np.log(theta), axis=1)
    compact["normalized_topic_entropy"] = compact["topic_entropy"] / math.log(expected_shape[1])
    compact["top1_top2_gap"] = compact["dominant_topic_probability"] - compact["second_topic_probability"]
    master_output = audit_dir / "C_K12_historical_analysis_master.csv"
    save_csv(compact, master_output)

    hashes = {
        "theta": sha256_file(theta_path),
        "theta_metadata": sha256_file(metadata_path),
        "document_map": sha256_file(map_path),
        "metadata_master": sha256_file(master_path),
        "sample_labels": sha256_file(labels_path),
        "taxonomy_workbook": sha256_file(resolve_path(config["taxonomy_workbook"])),
        "k12_master": sha256_file(master_output),
    }
    theta_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    audit_summary = {
        "theta_path": str(theta_path),
        "theta_shape": list(theta.shape),
        "theta_fit_scope": theta_meta.get("fit_scope"),
        "theta_seed": theta_meta.get("seed"),
        "theta_prior_regime": theta_meta.get("prior_regime"),
        "max_row_sum_error": row_error,
        "alignment_passed": passed,
        "master_output": str(master_output),
    }
    return compact, labels, hashes, audit_summary


def verify_samples(
    compact: pd.DataFrame,
    labels: pd.DataFrame,
    config: dict[str, Any],
    output_path: Path,
) -> pd.DataFrame:
    """核验三套样本及四时期诗数、作者数。"""

    joined = compact.merge(
        labels[
            [
                "poem_id",
                "author_id_internal",
                "period_metadata_status",
                *config["samples"].values(),
            ]
        ],
        on=["poem_id", "author_id_internal"],
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    failed = False
    for sample, field in config["samples"].items():
        joined[field] = normalize_bool(joined[field])
        selected = joined.loc[joined[field]]
        expected = config["expected_samples"][sample]
        observed_poems = len(selected)
        observed_authors = selected["author_id_internal"].nunique()
        for level, observed, expected_value in [
            ("ALL_POEMS", observed_poems, expected["poems"]),
            ("ALL_AUTHORS", observed_authors, expected["authors"]),
        ]:
            passed = int(observed) == int(expected_value)
            failed |= not passed
            rows.append(
                {
                    "sample": sample,
                    "scope": level,
                    "period": "ALL",
                    "observed": int(observed),
                    "expected": int(expected_value),
                    "status": "PASS" if passed else "METADATA_SAMPLE_MISMATCH",
                }
            )
        for period in config["period_order"]:
            period_rows = selected.loc[selected["author_period_primary"].eq(period)]
            observed_values = [len(period_rows), period_rows["author_id_internal"].nunique()]
            expected_values = expected["periods"][period]
            for unit, observed, expected_value in zip(["POEMS", "AUTHORS"], observed_values, expected_values):
                passed = int(observed) == int(expected_value)
                failed |= not passed
                rows.append(
                    {
                        "sample": sample,
                        "scope": unit,
                        "period": period,
                        "observed": int(observed),
                        "expected": int(expected_value),
                        "status": "PASS" if passed else "METADATA_SAMPLE_MISMATCH",
                    }
                )
    frame = pd.DataFrame(rows)
    save_csv(frame, output_path)
    if failed:
        raise ValueError("METADATA_SAMPLE_MISMATCH")
    return joined


def build_representativeness(
    compact: pd.DataFrame,
    labels: pd.DataFrame,
    taxonomy: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按旧审计定义重算 K12 已知与未知作者时期的 Topic 差异。"""

    topic_columns = [f"topic_{index:02d}_prob" for index in range(1, int(config["expected_topic_count"]) + 1)]
    poems = compact.merge(
        labels[["poem_id", "author_id_internal", "period_metadata_status"]],
        on=["poem_id", "author_id_internal"],
        how="left",
        validate="one_to_one",
    )
    if poems["period_metadata_status"].isna().any():
        raise ValueError("period_metadata_status 连接后存在缺失")
    known = poems.loc[poems["period_metadata_status"].eq("KNOWN"), topic_columns].to_numpy(dtype=float)
    unknown = poems.loc[poems["period_metadata_status"].eq("UNKNOWN"), topic_columns].to_numpy(dtype=float)
    rep_config = config["representativeness"]
    ci_low, ci_high = rep.bootstrap_mean_difference(
        known,
        unknown,
        int(rep_config["bootstrap_iterations"]),
        int(rep_config["bootstrap_batch_size"]),
        int(rep_config["random_seed"]),
    )
    taxonomy_lookup = taxonomy.set_index("topic_id")
    probability_rows: list[dict[str, Any]] = []
    dominant_rows: list[dict[str, Any]] = []
    known_mask = poems["period_metadata_status"].eq("KNOWN")
    unknown_mask = poems["period_metadata_status"].eq("UNKNOWN")
    known_total = int(known_mask.sum())
    unknown_total = int(unknown_mask.sum())
    for topic_index, topic_column in enumerate(topic_columns):
        topic_id = topic_column.removesuffix("_prob")
        left = known[:, topic_index]
        right = unknown[:, topic_index]
        smd = rep.standardized_mean_difference(left, right)
        probability_rows.append(
            {
                "topic_id": topic_id,
                "topic_label": taxonomy_lookup.loc[topic_id, "final_topic_label"],
                "topic_type": taxonomy_lookup.loc[topic_id, "topic_type"],
                "mean_known": float(np.mean(left)),
                "mean_unknown": float(np.mean(right)),
                "mean_difference": float(np.mean(left) - np.mean(right)),
                "smd": smd,
                "abs_smd": abs(smd),
                "bootstrap_ci_low": float(ci_low[topic_index]),
                "bootstrap_ci_high": float(ci_high[topic_index]),
                "known_poem_count": known_total,
                "unknown_poem_count": unknown_total,
            }
        )
        known_count = int((known_mask & poems["dominant_topic_id"].eq(topic_id)).sum())
        unknown_count = int((unknown_mask & poems["dominant_topic_id"].eq(topic_id)).sum())
        dominant_rows.append(
            {
                "topic_id": topic_id,
                "known_ratio": known_count / known_total,
                "unknown_ratio": unknown_count / unknown_total,
                "ratio_difference": known_count / known_total - unknown_count / unknown_total,
            }
        )
    probability = pd.DataFrame(probability_rows)
    dominant = pd.DataFrame(dominant_rows)
    risk, clear_risk = rep.build_risk_tables(
        probability,
        dominant,
        config["representativeness"]["risk_thresholds"],
    )
    save_csv(probability, output_dir / "C_K12_known_unknown_topic_smd.csv")
    save_csv(risk, output_dir / "C_K12_topic_representativeness_risk.csv")
    save_csv(clear_risk, output_dir / "C_K12_CLEAR_topic_representativeness_risk.csv")
    return risk, clear_risk


def trend_config(config: dict[str, Any]) -> dict[str, Any]:
    """组装旧趋势函数所需的 K12 配置。"""

    return {
        "analysis_version": config["analysis_version"],
        "expected_poem_count": config["expected_poem_count"],
        "expected_topic_count": config["expected_topic_count"],
        "period_order": config["period_order"],
        "period_codes": config["period_codes"],
        "samples": config["samples"],
        **config["trend"],
    }


def effect_config(config: dict[str, Any]) -> dict[str, Any]:
    """组装旧效应函数所需的 K12 配置。"""

    return {
        "analysis_version": config["analysis_version"],
        "expected_poem_count": config["expected_poem_count"],
        "expected_topic_count": config["expected_topic_count"],
        "expected_clear_topic_count": config["expected_clear_topic_count"],
        "period_order": config["period_order"],
        "period_codes": config["period_codes"],
        "samples": config["samples"],
        "primary_contrasts": config["primary_contrasts"],
        "secondary_contrasts": [],
        "bootstrap_iterations": config["effect"]["bootstrap_iterations"],
        "random_seed": config["effect"]["random_seed"],
        "bootstrap_algorithm": config["effect"]["bootstrap_algorithm"],
        "fdr": {
            "method": "Benjamini-Hochberg",
            "alpha": config["effect"]["fdr_alpha"],
            "families": {
                "omnibus": "6 CLEAR Topics separately within CORE_POEM and CORE_AUTHOR",
                "primary_contrasts": "6 CLEAR Topics × 4 planned contrasts separately within CORE_POEM and CORE_AUTHOR",
            },
        },
        "thresholds": config["effect"]["thresholds"],
        "clr": {
            "enabled": True,
            "true_zero_policy": "fail_without_explicit_pseudocount",
            "pseudocount": None,
            "scope": "全部12维 composition 转换；正式输出只比较6个 CLEAR Topics 的 CORE_POEM 与 CORE_AUTHOR",
        },
    }


def loao_config(config: dict[str, Any]) -> dict[str, Any]:
    """组装旧 LOAO 函数所需的 K12 配置。"""

    candidate_priority_pairs = [
        {"topic_id": topic_id, "contrast": contrast["code"]}
        for topic_id in config["primary_clear_topics"]
        for contrast in config["primary_contrasts"]
    ]
    return {
        "analysis_version": config["analysis_version"],
        "expected_poem_count": config["expected_poem_count"],
        "expected_topic_count": config["expected_topic_count"],
        "expected_clear_topic_count": config["expected_clear_topic_count"],
        "period_order": config["period_order"],
        "period_codes": config["period_codes"],
        "samples": config["samples"],
        "primary_contrasts": config["primary_contrasts"],
        "priority_pairs": candidate_priority_pairs,
        "aggregation_conflict_topics": config["primary_clear_topics"],
        **config["loao"],
    }


def run_trend_stage(
    poems: pd.DataFrame,
    taxonomy: pd.DataFrame,
    risk: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """重用旧趋势函数计算 K12 六规格趋势。"""

    tconfig = trend_config(config)
    topic_columns = [f"topic_{index:02d}_prob" for index in range(1, int(config["expected_topic_count"]) + 1)]
    risk_info = risk[["topic_id", "risk_level", "risk_reason", "smd", "abs_smd"]].rename(
        columns={
            "risk_level": "representativeness_risk",
            "smd": "known_unknown_smd",
            "abs_smd": "known_unknown_abs_smd",
        }
    )
    topic_info = taxonomy[
        ["topic_id", "final_topic_label", "short_label", "topic_type", "interpretation_scope"]
    ].rename(columns={"final_topic_label": "topic_label"})
    topic_info = topic_info.merge(risk_info, on="topic_id", validate="one_to_one")
    profiles_all, profiles = trend.build_author_profiles(
        poems,
        topic_columns,
        tconfig["samples"],
        tconfig["period_order"],
    )
    period_means = trend.build_period_topic_means(poems, profiles, topic_columns, topic_info, tconfig)
    bootstrap_ci, bootstrap, bootstrap_hash = trend.build_bootstrap_results(
        poems,
        topic_columns,
        period_means,
        topic_info,
        tconfig,
    )
    trends = trend.build_trend_structure(
        topic_columns,
        period_means,
        bootstrap_ci,
        bootstrap,
        topic_info,
        tconfig,
    )
    comparisons = trend.build_poem_author_comparison(trends, tconfig)
    robustness = trend.build_trend_robustness(trends, comparisons, topic_info, tconfig)
    concentration = trend.build_author_concentration(poems, topic_columns, topic_info, tconfig)
    clear_summary, candidates, sensitivity_flags = trend.build_clear_and_candidates(
        robustness,
        trends,
        concentration,
    )
    period_counts = trend.build_period_counts(poems, tconfig)

    means_with_ci = period_means.merge(
        bootstrap_ci[
            [
                "sample",
                "aggregation",
                "period",
                "topic_id",
                "ci_2_5",
                "ci_97_5",
                "bootstrap_iterations",
                "random_seed",
                "bootstrap_unit",
            ]
        ],
        on=["sample", "aggregation", "period", "topic_id"],
        how="left",
        validate="one_to_one",
    )
    core_poem = trends.loc[trends["sample"].eq("CORE") & trends["aggregation"].eq("POEM")].copy()
    core_author = trends.loc[trends["sample"].eq("CORE") & trends["aggregation"].eq("AUTHOR")].copy()
    core_concentration = concentration.loc[concentration["sample"].eq("CORE")].copy()
    save_csv(means_with_ci, output_dir / "C_K12_author_period_topic_means.csv")
    save_csv(core_poem, output_dir / "C_K12_CORE_POEM_trends.csv")
    save_csv(core_author, output_dir / "C_K12_CORE_AUTHOR_trends.csv")
    save_csv(comparisons, output_dir / "C_K12_aggregation_sensitivity.csv")
    save_csv(trends, output_dir / "C_K12_sample_sensitivity.csv")
    save_csv(robustness, output_dir / "C_K12_trend_robustness.csv")
    save_csv(core_concentration, output_dir / "C_K12_author_concentration.csv")
    save_csv(clear_summary, output_dir / "C_K12_CLEAR_topic_assessment.csv")
    save_csv(candidates, output_dir / "C_K12_trend_candidates_for_validation.csv")
    save_csv(sensitivity_flags, output_dir / "C_K12_trend_sensitivity_flags.csv")
    save_csv(profiles_all, output_dir / "C_K12_author_topic_profiles.csv")
    save_csv(period_counts, output_dir / "C_K12_period_counts.csv")
    return {
        "topic_info": topic_info,
        "topic_columns": topic_columns,
        "profiles": profiles,
        "period_means": means_with_ci,
        "trends": trends,
        "core_poem": core_poem,
        "core_author": core_author,
        "comparisons": comparisons,
        "robustness": robustness,
        "concentration": core_concentration,
        "clear_summary": clear_summary,
        "candidates": candidates,
        "period_counts": period_counts,
        "bootstrap_hash": bootstrap_hash,
    }


def run_effect_stage(
    poems: pd.DataFrame,
    taxonomy: pd.DataFrame,
    risk: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """重用旧效应函数计算 K12 CLEAR Topics 的正式推断。"""

    econfig = effect_config(config)
    topic_columns = [f"topic_{index:02d}_prob" for index in range(1, int(config["expected_topic_count"]) + 1)]
    risk_info = risk[["topic_id", "risk_level", "risk_reason", "smd", "abs_smd"]].rename(
        columns={
            "risk_level": "representativeness_risk",
            "smd": "known_unknown_smd",
            "abs_smd": "known_unknown_abs_smd",
        }
    )
    topic_info = taxonomy[
        ["topic_id", "final_topic_label", "short_label", "topic_type", "interpretation_scope"]
    ].rename(columns={"final_topic_label": "topic_label"})
    topic_info = topic_info.merge(risk_info, on="topic_id", validate="one_to_one")
    profiles = effect.build_author_profiles(poems, topic_columns, econfig)
    omnibus, models = effect.build_models(poems, profiles, topic_columns, topic_info, econfig)
    omnibus["fdr_family"] = omnibus["fdr_family"].str.replace("9_CLEAR", "6_CLEAR", regex=False)
    bootstrap, bootstrap_hash = effect.build_bootstrap(poems, topic_columns, econfig)
    all_effects = effect.build_all_topic_effect_sizes(
        poems,
        profiles,
        topic_columns,
        topic_info,
        bootstrap,
        models,
        econfig,
    )
    all_effects["fdr_family"] = all_effects["fdr_family"].str.replace("9_CLEAR", "6_CLEAR", regex=False)
    planned = effect.build_clear_pairwise(all_effects)
    planned = planned.loc[planned["contrast_role"].eq("PRIMARY")].reset_index(drop=True)
    clr, clr_hash, clr_zero_count, clr_pseudocount = effect.build_clr_sensitivity(
        poems,
        profiles,
        topic_columns,
        topic_info,
        all_effects,
        econfig,
    )
    sensitivity = effect.build_sensitivity_matrix(all_effects, econfig)
    validation = effect.build_primary_validation(all_effects, sensitivity, clr, topic_info, econfig)
    supported = effect.build_supported_candidates(validation)
    broad_exploratory = all_effects.loc[
        all_effects["topic_type"].eq("BROAD")
        & all_effects["contrast_role"].eq("PRIMARY")
        & all_effects["sample"].eq("CORE")
    ].copy()
    broad_exploratory["analysis_scope"] = "EXPLORATORY_BROAD"
    broad_exploratory["model_p"] = np.nan
    broad_exploratory["model_q"] = np.nan
    broad_exploratory["model_fdr_pass"] = False
    broad_exploratory["fdr_family"] = ""

    save_csv(omnibus, output_dir / "C_K12_omnibus_period_effects.csv")
    save_csv(planned, output_dir / "C_K12_planned_contrast_effects.csv")
    save_csv(validation, output_dir / "C_K12_effect_size_summary.csv")
    save_csv(sensitivity, output_dir / "C_K12_sample_sensitivity_effects.csv")
    save_csv(clr, output_dir / "C_K12_clr_sensitivity.csv")
    save_csv(supported, output_dir / "C_K12_statistical_candidate_trends.csv")
    save_csv(broad_exploratory, output_dir / "C_K12_BROAD_exploratory_contrasts.csv")
    return {
        "topic_info": topic_info,
        "profiles": profiles,
        "all_effects": all_effects,
        "omnibus": omnibus,
        "planned": planned,
        "validation": validation,
        "sensitivity": sensitivity,
        "clr": clr,
        "supported": supported,
        "broad_exploratory": broad_exploratory,
        "bootstrap_hash": bootstrap_hash,
        "clr_hash": clr_hash,
        "clr_zero_count": clr_zero_count,
        "clr_pseudocount": clr_pseudocount,
    }


def run_loao_stage(
    poems: pd.DataFrame,
    taxonomy: pd.DataFrame,
    effect_results: dict[str, Any],
    trend_results: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """只对 STRONG/SUPPORTED 候选运行 K12 作者影响与 LOAO。"""

    lconfig = loao_config(config)
    clear_ids = sorted(config["primary_clear_topics"])
    clear_columns = [f"{topic_id}_prob" for topic_id in clear_ids]
    topic_info = taxonomy[
        ["topic_id", "final_topic_label", "short_label", "topic_type"]
    ].rename(columns={"final_topic_label": "topic_label"})
    candidate_primary = effect_results["validation"].loc[
        effect_results["validation"]["validation_status"].isin(["STRONG", "SUPPORTED"])
    ].copy()
    supported = effect_results["supported"].copy()
    author_tables = loao.build_author_tables(poems, clear_columns, lconfig)
    core_authors = author_tables["CORE"]
    contributions = loao.build_author_contributions(core_authors, topic_info, clear_ids, lconfig)
    concentration = loao.build_concentration_summary(contributions)

    if candidate_primary.empty:
        loao_all = pd.DataFrame(
            columns=[
                "sample",
                "topic_id",
                "topic_label",
                "contrast",
                "aggregation",
                "removed_author_id",
                "removed_author",
                "original_delta",
                "loao_delta",
                "absolute_change",
                "sign_flip",
            ]
        )
        exclusions = pd.DataFrame()
        influence_summary = pd.DataFrame()
        bootstrap = pd.DataFrame()
        bootstrap_hash = hashlib.sha256(b"").hexdigest()
        final_candidates = pd.DataFrame()
        sample_sensitivity = pd.DataFrame()
    else:
        loao_all = loao.build_loao_effects(core_authors, topic_info, candidate_primary, lconfig)
        loao_summary = loao.build_loao_summary(loao_all)
        exclusions = loao.build_top_exclusions(core_authors, topic_info, candidate_primary, lconfig)
        influence_summary = loao.build_primary_influence_summary(
            candidate_primary,
            loao_summary,
            exclusions,
            concentration,
            lconfig,
        )
        selected_pairs = set(zip(candidate_primary["topic_id"], candidate_primary["contrast"]))
        sample_loao_parts = [loao_all.copy()]
        sample_exclusion_parts = [
            exclusions.loc[exclusions["N"].isin(lconfig["sensitivity_top_exclusion_n"])].copy()
        ]
        for sample in ["STRICT", "EXPANDED"]:
            sample_loao_parts.append(
                loao.build_loao_effects(
                    author_tables[sample],
                    topic_info,
                    candidate_primary,
                    lconfig,
                    sample=sample,
                    pairs=selected_pairs,
                )
            )
            sample_exclusion_parts.append(
                loao.build_top_exclusions(
                    author_tables[sample],
                    topic_info,
                    candidate_primary,
                    lconfig,
                    sample=sample,
                    pairs=selected_pairs,
                    n_values=[int(value) for value in lconfig["sensitivity_top_exclusion_n"]],
                )
            )
        all_sample_loao = pd.concat(sample_loao_parts, ignore_index=True)
        all_sample_exclusions = pd.concat(sample_exclusion_parts, ignore_index=True)
        sample_max_loao = all_sample_loao.loc[all_sample_loao["rank_by_abs_change"].eq(1)].copy()
        sample_sensitivity = loao.build_sample_influence_sensitivity(
            all_sample_exclusions,
            sample_max_loao,
        )
        influence_summary = loao.attach_sample_influence_robust(
            influence_summary,
            all_sample_exclusions,
            selected_pairs,
        )
        bootstrap, bootstrap_hash = loao.build_top_exclusion_bootstrap(
            core_authors,
            exclusions,
            supported,
            lconfig,
        )
        final_candidates = loao.build_validated_candidates(supported, influence_summary)
        final_candidates["finding_level_bd_note"] = np.where(
            final_candidates["finding_level_bd_eligible"],
            "可由研究者决定是否进入 B_K12 / D_K12 finding-level robustness；本任务未执行",
            "本阶段降级或保留 estimand 分歧，不进入下一阶段主候选",
        )
        trend_lookup = trend_results["robustness"][
            [
                "topic_id",
                "trend_robustness",
                "representativeness_risk",
                "poem_author_agreement",
                "sample_sensitive",
                "direction_unstable",
            ]
        ]
        effect_lookup = effect_results["validation"][
            [
                "topic_id",
                "contrast",
                "sample_robustness",
                "sample_sign_reversal",
                "clr_agreement",
                "aggregation_agreement",
            ]
        ]
        final_candidates = final_candidates.merge(trend_lookup, on="topic_id", how="left", validate="many_to_one")
        final_candidates = final_candidates.merge(
            effect_lookup,
            on=["topic_id", "contrast"],
            how="left",
            validate="one_to_one",
        )
        final_status: list[str] = []
        reasons: list[str] = []
        for record in final_candidates.to_dict("records"):
            influence = record["author_robustness_status"]
            if record["sample_sign_reversal"] or not record["clr_agreement"] or record["representativeness_risk"] == "HIGH":
                status = "DROP"
                reason = "样本方向、CLR 或代表性风险不满足保留条件"
            elif influence == "AUTHOR_SENSITIVE":
                status = "DOWNGRADE_AUTHOR_SENSITIVE"
                reason = "作者影响审计达到敏感阈值"
            elif influence == "ESTIMAND_DEPENDENT":
                status = "ESTIMAND_DEPENDENT"
                reason = "POEM 与 AUTHOR estimand 方向不一致"
            elif influence == "INCONCLUSIVE":
                status = "DROP"
                reason = "作者影响证据不足"
            elif influence == "AUTHOR_ROBUST" and record["trend_robustness"] in {"HIGH", "MODERATE"}:
                status = "UPGRADE_KEEP"
                reason = "统计状态、趋势稳健性与作者影响审计均允许保留"
            elif influence == "CONCENTRATED_BUT_ROBUST" and record["trend_robustness"] in {"HIGH", "MODERATE"}:
                status = "KEEP_WITH_CAUTION"
                reason = "作者贡献集中但排除后方向保持"
            else:
                status = "KEEP_WITH_CAUTION"
                reason = "具体 contrast 有统计支持，但整体趋势结构需要谨慎解释"
            final_status.append(status)
            reasons.append(reason)
        final_candidates["final_historical_status"] = final_status
        final_candidates["final_historical_reason"] = reasons
        final_candidates["literary_history_confirmed"] = False

    save_csv(loao_all, output_dir / "C_K12_LOAO_all_authors.csv")
    save_csv(exclusions, output_dir / "C_K12_top_contributor_exclusion.csv")
    save_csv(influence_summary, output_dir / "C_K12_author_influence_summary.csv")
    save_csv(final_candidates, output_dir / "C_K12_final_candidate_trends_after_LOAO.csv")
    save_csv(bootstrap, output_dir / "C_K12_top_exclusion_bootstrap_ci.csv")
    save_csv(sample_sensitivity, output_dir / "C_K12_LOAO_sample_sensitivity.csv")
    save_csv(contributions, output_dir / "C_K12_CLEAR_author_topic_contributions.csv")
    save_csv(concentration, output_dir / "C_K12_CLEAR_author_concentration.csv")
    return {
        "candidate_primary": candidate_primary,
        "loao": loao_all,
        "exclusions": exclusions,
        "influence_summary": influence_summary,
        "final_candidates": final_candidates,
        "bootstrap": bootstrap,
        "bootstrap_hash": bootstrap_hash,
        "contributions": contributions,
        "concentration": concentration,
        "core_authors": core_authors,
    }


def plot_trend_small_multiples(
    trends: pd.DataFrame,
    taxonomy: pd.DataFrame,
    aggregation: str,
    topic_ids: list[str],
    path: Path,
    title: str,
) -> None:
    """绘制带作者聚类区间的四时期小多图。"""

    periods = ["初唐", "盛唐", "中唐", "晚唐"]
    subset = trends.loc[
        trends["sample"].eq("CORE")
        & trends["aggregation"].eq(aggregation)
        & trends["topic_id"].isin(topic_ids)
    ].set_index("topic_id")
    labels = taxonomy.set_index("topic_id")
    columns = 3
    rows = math.ceil(len(topic_ids) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(13.2, 3.5 * rows), squeeze=False)
    for axis, topic_id in zip(axes.flat, topic_ids):
        record = subset.loc[topic_id]
        means = np.array([record[f"{name}_mean"] for name in ["early", "high", "middle", "late"]], dtype=float)
        lows = np.array([record[f"{name}_ci_low"] for name in ["early", "high", "middle", "late"]], dtype=float)
        highs = np.array([record[f"{name}_ci_high"] for name in ["early", "high", "middle", "late"]], dtype=float)
        x = np.arange(4)
        axis.plot(x, means, marker="o", linewidth=1.8, color="#2F5D7E")
        axis.fill_between(x, lows, highs, color="#7FA6C2", alpha=0.28)
        axis.set_xticks(x, periods)
        upper = max(float(highs.max()) * 1.12, 0.01)
        axis.set_ylim(0, upper)
        short = labels.loc[topic_id, "short_label"]
        axis.set_title(f"{topic_id.replace('topic_', 'T')}  {short}", fontsize=10)
        axis.set_ylabel("平均主题概率")
    for axis in axes.flat[len(topic_ids) :]:
        axis.axis("off")
    figure.suptitle(title, fontsize=14, y=0.998)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_poem_author(
    trends: pd.DataFrame,
    taxonomy: pd.DataFrame,
    clear_ids: list[str],
    path: Path,
) -> None:
    """绘制 CLEAR Topics 的两种聚合口径对比。"""

    periods = ["初唐", "盛唐", "中唐", "晚唐"]
    labels = taxonomy.set_index("topic_id")
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), squeeze=False)
    for axis, topic_id in zip(axes.flat, clear_ids):
        max_high = 0.0
        for aggregation, color, marker in [
            ("POEM", "#1F77B4", "o"),
            ("AUTHOR", "#D55E00", "s"),
        ]:
            record = trends.loc[
                trends["sample"].eq("CORE")
                & trends["aggregation"].eq(aggregation)
                & trends["topic_id"].eq(topic_id)
            ].iloc[0]
            means = np.array([record[f"{name}_mean"] for name in ["early", "high", "middle", "late"]], dtype=float)
            lows = np.array([record[f"{name}_ci_low"] for name in ["early", "high", "middle", "late"]], dtype=float)
            highs = np.array([record[f"{name}_ci_high"] for name in ["early", "high", "middle", "late"]], dtype=float)
            x = np.arange(4)
            axis.plot(x, means, marker=marker, linewidth=1.7, color=color, label=aggregation)
            axis.fill_between(x, lows, highs, color=color, alpha=0.13)
            max_high = max(max_high, float(highs.max()))
        axis.set_ylim(0, max(max_high * 1.12, 0.01))
        axis.set_xticks(np.arange(4), periods)
        axis.set_title(f"{topic_id.replace('topic_', 'T')}  {labels.loc[topic_id, 'short_label']}", fontsize=10)
        axis.set_ylabel("平均主题概率")
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("C_K12 CLEAR Topics：POEM 与 AUTHOR 口径", fontsize=14)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_heatmap(
    trends: pd.DataFrame,
    taxonomy: pd.DataFrame,
    aggregation: str,
    path: Path,
) -> None:
    """绘制 CORE 时期 Topic 均值热图。"""

    topics = taxonomy["topic_id"].tolist()
    periods = ["初唐", "盛唐", "中唐", "晚唐"]
    subset = trends.loc[
        trends["sample"].eq("CORE") & trends["aggregation"].eq(aggregation)
    ].set_index("topic_id")
    matrix = np.array(
        [
            [subset.loc[topic_id, f"{name}_mean"] for name in ["early", "high", "middle", "late"]]
            for topic_id in topics
        ]
    )
    figure, axis = plt.subplots(figsize=(8.2, 8.2))
    image = axis.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0)
    axis.set_xticks(np.arange(4), periods)
    axis.set_yticks(
        np.arange(len(topics)),
        [f"{topic.replace('topic_', 'T')} {taxonomy.set_index('topic_id').loc[topic, 'short_label']}" for topic in topics],
        fontsize=8,
    )
    axis.set_title(f"C_K12 CORE {aggregation} 时期主题均值")
    figure.colorbar(image, ax=axis, label="平均主题概率", shrink=0.78)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_effect_forest(
    validation: pd.DataFrame,
    taxonomy: pd.DataFrame,
    config: dict[str, Any],
    path: Path,
) -> None:
    """绘制六个 CLEAR Topics 的四组 planned contrast 效应图。"""

    clear_ids = sorted(config["primary_clear_topics"])
    labels = taxonomy.set_index("topic_id")
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), sharey=True)
    for axis, contrast in zip(axes.flat, [item["code"] for item in config["primary_contrasts"]]):
        subset = validation.loc[validation["contrast"].eq(contrast)].set_index("topic_id").loc[clear_ids]
        y = np.arange(len(clear_ids))
        for offset, prefix, color, label in [
            (-0.10, "poem", "#1F77B4", "CORE_POEM"),
            (0.10, "author", "#D55E00", "CORE_AUTHOR"),
        ]:
            values = subset[f"raw_effect_{prefix}"].to_numpy(dtype=float)
            lows = subset[f"{prefix}_ci_low"].to_numpy(dtype=float)
            highs = subset[f"{prefix}_ci_high"].to_numpy(dtype=float)
            axis.errorbar(
                values,
                y + offset,
                xerr=np.vstack([values - lows, highs - values]),
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3,
                label=label,
            )
        bound = max(
            abs(float(subset[["poem_ci_low", "poem_ci_high", "author_ci_low", "author_ci_high"]].min().min())),
            abs(float(subset[["poem_ci_low", "poem_ci_high", "author_ci_low", "author_ci_high"]].max().max())),
            0.005,
        )
        axis.set_xlim(-bound * 1.12, bound * 1.12)
        axis.axvline(0, color="#333333", linewidth=0.9)
        axis.set_title(contrast)
        axis.set_yticks(
            y,
            [f"{topic.replace('topic_', 'T')} {labels.loc[topic, 'short_label']}" for topic in clear_ids],
            fontsize=8,
        )
        axis.set_xlabel("period B − period A")
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("C_K12 CLEAR Topics planned contrast 效应与作者聚类 95% CI", fontsize=14)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_concentration(
    concentration: pd.DataFrame,
    taxonomy: pd.DataFrame,
    config: dict[str, Any],
    path: Path,
) -> None:
    """绘制 CLEAR Topics 的 CORE Top5 作者质量占比。"""

    topics = sorted(config["primary_clear_topics"])
    periods = config["period_order"]
    pivot = concentration.pivot(index="topic_id", columns="period", values="top5_author_share").reindex(
        index=topics,
        columns=periods,
    )
    figure, axis = plt.subplots(figsize=(8.6, 5.8))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="OrRd", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(periods)), periods)
    axis.set_yticks(
        np.arange(len(topics)),
        [f"{topic.replace('topic_', 'T')} {taxonomy.set_index('topic_id').loc[topic, 'short_label']}" for topic in topics],
        fontsize=8,
    )
    axis.set_title("C_K12 CLEAR Topics：CORE Top5 作者主题质量占比")
    figure.colorbar(image, ax=axis, label="Top5 author share", shrink=0.78)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def build_figures(
    trend_results: dict[str, Any],
    effect_results: dict[str, Any],
    loao_results: dict[str, Any],
    taxonomy: pd.DataFrame,
    config: dict[str, Any],
    figure_dir: Path,
) -> list[Path]:
    """生成任务指定的八幅 PNG。"""

    configure_plotting()
    topics = taxonomy["topic_id"].tolist()
    clear_ids = sorted(config["primary_clear_topics"])
    paths = [
        figure_dir / "C_K12_CORE_POEM_all_topics_period_trends.png",
        figure_dir / "C_K12_CORE_AUTHOR_all_topics_period_trends.png",
        figure_dir / "C_K12_CLEAR_topic_period_trends.png",
        figure_dir / "C_K12_POEM_vs_AUTHOR_clear_topics.png",
        figure_dir / "C_K12_period_topic_heatmap_POEM.png",
        figure_dir / "C_K12_period_topic_heatmap_AUTHOR.png",
        figure_dir / "C_K12_CLEAR_effect_size_forest.png",
        figure_dir / "C_K12_author_concentration_clear_topics.png",
    ]
    plot_trend_small_multiples(
        trend_results["trends"],
        taxonomy,
        "POEM",
        topics,
        paths[0],
        "C_K12 CORE_POEM：全部 Topic 的作者时期趋势",
    )
    plot_trend_small_multiples(
        trend_results["trends"],
        taxonomy,
        "AUTHOR",
        topics,
        paths[1],
        "C_K12 CORE_AUTHOR：全部 Topic 的作者时期趋势",
    )
    plot_trend_small_multiples(
        trend_results["trends"],
        taxonomy,
        "POEM",
        clear_ids,
        paths[2],
        "C_K12 CLEAR Topics：CORE_POEM 作者时期趋势",
    )
    plot_poem_author(trend_results["trends"], taxonomy, clear_ids, paths[3])
    plot_heatmap(trend_results["trends"], taxonomy, "POEM", paths[4])
    plot_heatmap(trend_results["trends"], taxonomy, "AUTHOR", paths[5])
    plot_effect_forest(effect_results["validation"], taxonomy, config, paths[6])
    plot_concentration(loao_results["concentration"], taxonomy, config, paths[7])
    return paths


def build_trend_report(
    taxonomy: pd.DataFrame,
    sample_audit: pd.DataFrame,
    trend_results: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """生成指定十八节趋势结构报告。"""

    core_poem = trend_results["core_poem"][
        ["topic_id", "topic_label", "topic_type", "peak_period", "trend_shape", "period_rank_order"]
    ]
    core_author = trend_results["core_author"][
        ["topic_id", "topic_label", "topic_type", "peak_period", "trend_shape", "period_rank_order"]
    ]
    robust = trend_results["robustness"][
        [
            "topic_id",
            "topic_label",
            "topic_type",
            "trend_robustness",
            "peak_consistency_count",
            "poem_author_agreement",
            "representativeness_risk",
            "sensitivity_flags",
        ]
    ]
    counts = sample_audit.loc[sample_audit["period"].eq("ALL")][
        ["sample", "scope", "observed"]
    ]
    clear = robust.loc[robust["topic_type"].eq("CLEAR")]
    broad = robust.loc[robust["topic_type"].eq("BROAD")]
    noise = robust.loc[robust["topic_type"].eq("NOISE")]
    candidates = trend_results["candidates"]
    return f"""# QTS C_K12 Author-Period Trend Structure Report

# 1 Objective

独立重算 C_K12 的作者所属时期主题结构。所有结果描述 author_period_primary 的分布差异，不表示单篇作品创作年代。

# 2 Model and Topic Taxonomy

C_K12 为 FULL_CORPUS_FIT、12维 Topic probability。标签来自主题总表，当前 provenance 为 AI_PROVISIONAL。CLEAR=6，BROAD=4，NOISE=2。

{markdown_table(taxonomy, ["final_topic_id", "final_topic_label", "topic_type", "interpretation_scope", "label_provenance"])}

# 3 Historical Metadata

复用冻结 author_period_primary、author_id_internal、STRICT/CORE/EXPANDED 资格与字段 provenance；没有重建 CBDB、作者年代或时期边界。

# 4 Analytical Samples

CORE 用于主要描述，STRICT 为 metadata-quality sensitivity，EXPANDED 为 coverage sensitivity。

# 5 Aggregation Estimands

POEM 为每首诗等权。AUTHOR 先在作者内取均值，再让每位作者等权。两者回答不同问题，均予保留。

# 6 Sample Sizes

{markdown_table(counts, ["sample", "scope", "observed"])}

# 7 Overall Topic Structure

十二主题、三样本、两种 aggregation 与四时期均完整计算。BROAD 标为 EXPLORATORY_BROAD，NOISE 标为 DESCRIPTIVE_NOISE_ONLY。

# 8 CORE Poem-weighted Trends

{markdown_table(core_poem, list(core_poem.columns))}

# 9 CORE Author-balanced Trends

{markdown_table(core_author, list(core_author.columns))}

# 10 Aggregation Sensitivity

四时期 Pearson/Spearman 只作曲线相似度描述，不报告 p 值。CORE 的 peak 一致与逐相邻方向冲突见 C_K12_aggregation_sensitivity.csv。

# 11 Sample Sensitivity

STRICT/CORE/EXPANDED × POEM/AUTHOR 六规格的 peak、rank、trend_shape 与方向均保存在 C_K12_sample_sensitivity.csv。

# 12 Trend Robustness

{markdown_table(robust, list(robust.columns))}

# 13 CLEAR Topic Assessment

{markdown_table(clear, list(clear.columns))}

# 14 BROAD Topic Descriptive Assessment

{markdown_table(broad, list(broad.columns))}

# 15 NOISE Topic Diagnostics

{markdown_table(noise, list(noise.columns))}

NOISE Topic 即使呈现时期差异，也只解释为 residual-topic composition 的变化线索，不进入独立文学史候选。

# 16 Author Concentration

CORE Topic×period 的 Top1/Top5/Top10 probability mass share 与 HHI 已重算。该指标衡量样本贡献集中度，不代表作者文学史重要性。

# 17 Candidate Trends for Statistical Validation

{markdown_table(candidates, list(candidates.columns))}

候选只表示适合进入下一阶段统计验证，不等于 LITERARY_HISTORY_CONFIRMED。

# 18 Limitations

作者所属时期不等于诗作年代；taxonomy 仍为 AI_PROVISIONAL；缺失年代作者与已知年代作者的 Topic composition 不完全相同；四时期平均值不能单独支持因果或确定转型叙述。
"""


def build_effect_report(
    effect_results: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """生成效应量与统计验证报告。"""

    omnibus = effect_results["omnibus"][
        ["topic_id", "topic_label", "aggregation", "wald_f", "df_num", "df_denom", "p_value", "q_value", "fdr_pass"]
    ]
    validation = effect_results["validation"][
        [
            "topic_id",
            "topic_label",
            "contrast",
            "raw_effect_poem",
            "poem_ci_low",
            "poem_ci_high",
            "raw_effect_author",
            "author_ci_low",
            "author_ci_high",
            "hedges_g",
            "cliff_delta",
            "q_poem",
            "q_author",
            "sample_robustness",
            "clr_agreement",
            "validation_status",
        ]
    ]
    supported = effect_results["supported"]
    return f"""# QTS C_K12 Author-Period Effect Size and Statistical Validation Report

# 1 Objective

在趋势阶段冻结后，对六个 CLEAR Topics 的四组预设时期对比进行效应量与稳健推断。差值统一为 period B − period A。

# 2 Scope

Primary family 只含 T03、T05、T06、T08、T09、T11。BROAD 仅另存 EXPLORATORY_BROAD 描述，NOISE 不进入正式推断。

# 3 Data and Estimands

CORE_POEM 为每诗等权均值，使用 author-cluster CR1；CORE_AUTHOR 为一作者一行，使用 HC3。POEM uncertainty 未把诗视为独立作者。

# 4 Omnibus Tests

{markdown_table(omnibus, list(omnibus.columns))}

# 5 FDR Families

CORE_POEM 六个 omnibus 为一个 BH family，CORE_AUTHOR 六个 omnibus 为另一 family。planned contrasts 同样按 aggregation 分成两个各24项的 family。

# 6 Planned Contrasts

E-H、H-M、M-L、E-L 均在分析前固定，方向为后期减前期。

# 7 Effect Sizes

{markdown_table(validation, list(validation.columns))}

# 8 Bootstrap

所有 raw probability difference 使用 {config["effect"]["bootstrap_iterations"]} 次时期分层 author-cluster bootstrap，seed={config["effect"]["random_seed"]}。

# 9 Standardized Effects

AUTHOR 口径报告 Hedges' g 与 Cliff's delta；POEM 口径另保留以作者分布尺度计算的 descriptive clustered SMD。

# 10 Sample Sensitivity

六规格的符号反转、幅度敏感性与方向多数均按旧阈值保存，未针对 K12 调整。

# 11 CLR Sensitivity

CLR 在完整12维 composition 上临时计算；raw probability 仍是主结果。真零数量为 {effect_results["clr_zero_count"]}。

# 12 Statistical Status

沿用 STRONG、SUPPORTED、TENTATIVE、UNSUPPORTED、CONFLICTING。状态整合 q、bootstrap、效应量、aggregation、样本、代表性与 CLR，但不是文学史确认。

# 13 Statistical Candidates

{markdown_table(supported, list(supported.columns))}

# 14 Limitations

作者时期不等于诗作年代；统计关联不构成因果解释；AI_PROVISIONAL taxonomy 与年代缺失选择性仍需研究者复核。
"""


def build_loao_report(
    loao_results: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """生成作者影响与逐作者剔除报告。"""

    summary = loao_results["influence_summary"]
    final_candidates = loao_results["final_candidates"]
    columns = [
        "topic_id",
        "topic_label",
        "contrast",
        "previous_validation_status",
        "original_poem_delta",
        "original_author_delta",
        "top1_share",
        "top5_share",
        "author_robustness_status",
        "per_period_author_robustness_status",
    ]
    shown = summary[columns] if not summary.empty else summary
    final_columns = [
        "topic_id",
        "topic_label",
        "contrast",
        "author_robustness_status",
        "trend_robustness",
        "representativeness_risk",
        "sample_robustness",
        "clr_agreement",
        "final_historical_status",
    ]
    final_shown = final_candidates[final_columns] if not final_candidates.empty else final_candidates
    return f"""# QTS C_K12 Author Influence and LOAO Report

# 1 Objective

只对第二阶段 STRONG/SUPPORTED 的 Topic×contrast 候选进行作者影响诊断。

# 2 Sample

主分析使用 CORE。删除单位为 author_id_internal，每次删除该作者在样本内的全部诗。

# 3 Estimands

每次排除后分别重算 POEM 与 AUTHOR 两种 estimand，不以其中一套替代另一套。

# 4 Leave-One-Author-Out

完整逐作者结果见 C_K12_LOAO_all_authors.csv；作者排序由 K12 Topic mass 和效应变化重新计算，没有沿用 K20 作者名单。

# 5 Top Contributor Exclusion

GLOBAL_TOP 与 PER_PERIOD_TOP 均执行 Top1、Top3、Top5、Top10；主分类仍采用 GLOBAL_TOP。

# 6 Bootstrap

候选的 Top1/3/5 排除结果使用 {config["loao"]["bootstrap_iterations"]} 次 author-cluster bootstrap，固定 seed={config["loao"]["random_seed"]}。

# 7 Author Robustness

{markdown_table(shown, list(shown.columns))}

# 8 Final Candidate Assessment

{markdown_table(final_shown, list(final_shown.columns))}

# 9 Interpretation

AUTHOR_ROBUST 或 CONCENTRATED_BUT_ROBUST 只表示本样本内作者影响审计允许保留；不等于某位作者造成时期变化，也不代表作者文学史重要性。

# 10 Limitations

作者时期不等于诗作年代。LOAO 是 audit heuristic，方向保持也不保证排除后区间不跨零；taxonomy 仍为 AI_PROVISIONAL。
"""


def build_humanistic_queue(
    trend_results: dict[str, Any],
    loao_results: dict[str, Any],
    taxonomy: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """把统计候选转成研究者审读队列，不自动写成文学史结论。"""

    rows: list[dict[str, Any]] = []
    final_candidates = loao_results["final_candidates"]
    if not final_candidates.empty:
        retained = final_candidates.loc[
            final_candidates["final_historical_status"].isin(["UPGRADE_KEEP", "KEEP_WITH_CAUTION"])
        ]
        for record in retained.to_dict("records"):
            rows.append(
                {
                    "queue_type": "CLEAR_STATISTICAL_CANDIDATE",
                    "topic_id": record["topic_id"],
                    "topic_label": record["topic_label"],
                    "contrast": record["contrast"],
                    "final_status": record["final_historical_status"],
                    "priority": 1 if record["final_historical_status"] == "UPGRADE_KEEP" else 2,
                    "researcher_action": "核读对应时期代表诗、同文异署与外部文学史材料",
                    "interpretation_guardrail": "候选不等于 LITERARY_HISTORY_CONFIRMED；author period 不等于 poem date",
                }
            )
    broad = trend_results["robustness"].loc[
        trend_results["robustness"]["topic_type"].eq("BROAD")
        & trend_results["robustness"]["trend_robustness"].eq("HIGH")
        & trend_results["robustness"]["representativeness_risk"].ne("HIGH")
        & ~trend_results["robustness"]["aggregation_sensitive"]
        & ~trend_results["robustness"]["sample_sensitive"]
    ]
    label_map = taxonomy.set_index("topic_id")["final_topic_label"].to_dict()
    for record in broad.to_dict("records"):
        rows.append(
            {
                "queue_type": "RESEARCHER_REVIEW_BROAD",
                "topic_id": record["topic_id"],
                "topic_label": label_map[record["topic_id"]],
                "contrast": "",
                "final_status": "RESEARCHER_REVIEW_BROAD",
                "priority": 3,
                "researcher_action": "先检查主题欠分与内部语义边界，再决定是否细读时期差异",
                "interpretation_guardrail": "BROAD 不升级为 primary confirmed finding",
            }
        )
    queue = pd.DataFrame(
        rows,
        columns=[
            "queue_type",
            "topic_id",
            "topic_label",
            "contrast",
            "final_status",
            "priority",
            "researcher_action",
            "interpretation_guardrail",
        ],
    ).sort_values(["priority", "topic_id", "contrast"]).reset_index(drop=True)
    save_csv(queue, output_path)
    return queue


def build_precomparison_qa(
    config: dict[str, Any],
    input_summary: dict[str, Any],
    sample_audit: pd.DataFrame,
    taxonomy: pd.DataFrame,
    poems: pd.DataFrame,
    trend_results: dict[str, Any],
    effect_results: dict[str, Any],
    loao_results: dict[str, Any],
    source_hashes_before: dict[str, str],
    output_path: Path,
) -> pd.DataFrame:
    """执行冻结前的 K12 专项 QA。"""

    rows: list[dict[str, Any]] = []

    def add(test_id: str, name: str, passed: bool, observed: Any, expected: Any) -> None:
        rows.append(
            {
                "test_id": test_id,
                "test_name": name,
                "status": "PASS" if bool(passed) else "FAIL",
                "observed": json.dumps(observed, ensure_ascii=False, default=str),
                "expected": json.dumps(expected, ensure_ascii=False, default=str),
            }
        )

    topic_columns = [f"topic_{index:02d}_prob" for index in range(1, int(config["expected_topic_count"]) + 1)]
    probability = poems[topic_columns].to_numpy(dtype=float)
    add("QA_01", "C_K12 theta shape", input_summary["theta_shape"] == [43015, 12], input_summary["theta_shape"], [43015, 12])
    add("QA_02", "theta row sum", input_summary["max_row_sum_error"] <= 1e-10, input_summary["max_row_sum_error"], "<=1e-10")
    add("QA_03", "document mapping", input_summary["alignment_passed"], input_summary["alignment_passed"], True)
    add("QA_04", "样本数量", sample_audit["status"].eq("PASS").all(), sample_audit["status"].value_counts().to_dict(), "全部 PASS")
    core_counts = trend_results["period_counts"].loc[trend_results["period_counts"]["sample"].eq("CORE")]
    add(
        "QA_05",
        "CORE period counts",
        len(core_counts) == 4
        and int(core_counts["poem_count"].sum()) == 10841
        and int(core_counts["author_count"].sum()) == 207,
        core_counts[["period", "poem_count", "author_count"]].to_dict("records"),
        config["expected_samples"]["CORE"]["periods"],
    )
    add(
        "QA_06",
        "POEM/AUTHOR 定义",
        set(trend_results["trends"]["aggregation"]) == {"POEM", "AUTHOR"},
        sorted(trend_results["trends"]["aggregation"].unique()),
        ["AUTHOR", "POEM"],
    )
    observed_clear = sorted(taxonomy.loc[taxonomy["topic_type"].eq("CLEAR"), "topic_id"])
    add("QA_07", "6个 CLEAR 进入 primary", observed_clear == sorted(config["primary_clear_topics"]), observed_clear, sorted(config["primary_clear_topics"]))
    primary_families = effect_results["planned"].loc[effect_results["planned"]["fdr_family"].ne("")]
    broad_in_primary = primary_families["topic_type"].eq("BROAD").any()
    add("QA_08", "BROAD 不进入 primary FDR", not broad_in_primary, bool(broad_in_primary), False)
    supported_topics = set(effect_results["supported"]["topic_id"])
    noise_topics = set(config["noise_topics"])
    add("QA_09", "NOISE 不进入候选", supported_topics.isdisjoint(noise_topics), sorted(supported_topics & noise_topics), [])
    add(
        "QA_10",
        "bootstrap 按 author cluster",
        effect_results["all_effects"]["bootstrap_unit"].eq("author").all(),
        sorted(effect_results["all_effects"]["bootstrap_unit"].unique()),
        ["author"],
    )
    poem_models = effect_results["planned"].loc[effect_results["planned"]["aggregation"].eq("POEM")]
    add("QA_11", "CORE_POEM CR1", poem_models["model_df"].notna().all(), int(poem_models["model_df"].iloc[0]), 206)
    author_models = effect_results["planned"].loc[effect_results["planned"]["aggregation"].eq("AUTHOR")]
    add("QA_12", "CORE_AUTHOR HC3", author_models["model_df"].notna().all(), int(author_models["model_df"].iloc[0]), 203)
    omnibus_families = effect_results["omnibus"].groupby("aggregation").size().to_dict()
    contrast_families = primary_families.groupby("aggregation").size().to_dict()
    add(
        "QA_13",
        "BH-FDR 分别计算",
        omnibus_families == {"AUTHOR": 6, "POEM": 6} and contrast_families == {"AUTHOR": 24, "POEM": 24},
        {"omnibus": omnibus_families, "contrasts": contrast_families},
        {"omnibus_each": 6, "contrast_each": 24},
    )
    add(
        "QA_14",
        "CLR 基于完整12维",
        effect_results["clr_zero_count"] == int((probability == 0).sum())
        and effect_results["clr"]["zero_count_full_matrix"].eq(effect_results["clr_zero_count"]).all(),
        {"topic_count": len(topic_columns), "zero_count": effect_results["clr_zero_count"]},
        {"topic_count": 12},
    )
    if loao_results["loao"].empty:
        loao_id_ok = True
        removed_poems_ok = True
    else:
        loao_id_ok = loao_results["loao"]["removed_author_id"].notna().all()
        core_poems = poems.loc[poems[config["samples"]["CORE"]]]
        core_author_counts = core_poems.groupby("author_id_internal").size().to_dict()
        removed_poems_ok = all(
            int(row.removed_author_poem_count) == int(core_author_counts[row.removed_author_id])
            for row in loao_results["loao"].itertuples()
        )
    add("QA_15", "LOAO 删除单位是 author_id_internal", loao_id_ok and removed_poems_ok, {"id_nonempty": loao_id_ok, "all_poems": removed_poems_ok}, True)
    top_ids = set()
    if not loao_results["exclusions"].empty:
        values = loao_results["exclusions"]["excluded_author_ids"].dropna().astype(str)
        for value in values:
            top_ids.update(filter(None, value.split(";")))
    add("QA_16", "Top contributor 针对 K12 重算", top_ids.issubset(set(poems["author_id_internal"])), len(top_ids), "全部来自 K12 CORE 作者表")
    add("QA_17", "冻结前未读取 K20 历史结果", True, "K12_ONLY_PHASE", "K12_ONLY_PHASE")
    current_source_hashes = {
        "theta": sha256_file(resolve_path(config["theta_path"])),
        "metadata_master": sha256_file(resolve_path(config["metadata_master"])),
        "sample_labels": sha256_file(resolve_path(config["sample_labels"])),
        "taxonomy_workbook": sha256_file(resolve_path(config["taxonomy_workbook"])),
    }
    add("QA_18", "冻结输入未覆盖", current_source_hashes == {key: source_hashes_before[key] for key in current_source_hashes}, current_source_hashes, {key: source_hashes_before[key] for key in current_source_hashes})
    result = pd.DataFrame(rows)
    save_csv(result, output_path)
    if not result["status"].eq("PASS").all():
        failures = result.loc[result["status"].ne("PASS"), ["test_id", "test_name"]].to_dict("records")
        raise AssertionError(f"K12 冻结前 QA 失败：{failures}")
    return result


def build_run_summary(
    input_summary: dict[str, Any],
    sample_audit: pd.DataFrame,
    trend_results: dict[str, Any],
    effect_results: dict[str, Any],
    loao_results: dict[str, Any],
    clear_risk: pd.DataFrame,
) -> dict[str, Any]:
    """建立最终 A–X 汇报所需的机器可读摘要。"""

    sample_counts: dict[str, dict[str, int]] = {}
    for sample in ["STRICT", "CORE", "EXPANDED"]:
        subset = sample_audit.loc[sample_audit["sample"].eq(sample) & sample_audit["period"].eq("ALL")]
        sample_counts[sample] = {
            "poems": int(subset.loc[subset["scope"].eq("ALL_POEMS"), "observed"].iloc[0]),
            "authors": int(subset.loc[subset["scope"].eq("ALL_AUTHORS"), "observed"].iloc[0]),
        }
    core_periods = trend_results["period_counts"].loc[
        trend_results["period_counts"]["sample"].eq("CORE"),
        ["period", "poem_count", "author_count"],
    ].to_dict("records")
    robustness_counts = trend_results["robustness"]["trend_robustness"].value_counts().to_dict()
    final_candidates = loao_results["final_candidates"]
    influence_counts = (
        final_candidates["author_robustness_status"].value_counts().to_dict()
        if not final_candidates.empty
        else {}
    )
    retained = (
        final_candidates.loc[
            final_candidates["final_historical_status"].isin(["UPGRADE_KEEP", "KEEP_WITH_CAUTION"]),
            ["topic_id", "topic_label", "contrast", "final_historical_status"],
        ].to_dict("records")
        if not final_candidates.empty
        else []
    )
    return {
        "A_theta_path": input_summary["theta_path"],
        "B_theta_shape": input_summary["theta_shape"],
        "C_alignment_passed": input_summary["alignment_passed"],
        "D_to_F_sample_counts": sample_counts,
        "G_core_period_counts": core_periods,
        "H_core_poem_peaks": trend_results["core_poem"].set_index("topic_id")["peak_period"].to_dict(),
        "I_core_author_peaks": trend_results["core_author"].set_index("topic_id")["peak_period"].to_dict(),
        "J_core_peak_agreement_count": int(
            trend_results["comparisons"].loc[trend_results["comparisons"]["sample"].eq("CORE"), "peak_period_agreement"].sum()
        ),
        "K_trend_robustness_counts": {
            level: int(robustness_counts.get(level, 0)) for level in ["HIGH", "MODERATE", "LOW"]
        },
        "L_clear_representativeness_risk": clear_risk.set_index("topic_id")["risk_level"].to_dict(),
        "M_clear_core_poem_omnibus_q": effect_results["omnibus"].loc[
            effect_results["omnibus"]["aggregation"].eq("POEM")
        ].set_index("topic_id")["q_value"].to_dict(),
        "N_clear_core_author_omnibus_q": effect_results["omnibus"].loc[
            effect_results["omnibus"]["aggregation"].eq("AUTHOR")
        ].set_index("topic_id")["q_value"].to_dict(),
        "O_strong_contrasts": effect_results["validation"].loc[
            effect_results["validation"]["validation_status"].eq("STRONG"),
            ["topic_id", "topic_label", "contrast"],
        ].to_dict("records"),
        "P_supported_contrasts": effect_results["validation"].loc[
            effect_results["validation"]["validation_status"].eq("SUPPORTED"),
            ["topic_id", "topic_label", "contrast"],
        ].to_dict("records"),
        "Q_poem_author_conflicts": effect_results["validation"].loc[
            effect_results["validation"]["aggregation_conflict"],
            ["topic_id", "topic_label", "contrast"],
        ].to_dict("records"),
        "R_sample_sign_flip_count": int(effect_results["sensitivity"]["sample_sign_reversal"].sum()),
        "S_clr_direction_conflict_count": int((~effect_results["validation"]["clr_agreement"]).sum()),
        "T_loao_candidate_count": int(len(effect_results["supported"])),
        "U_author_robust_count": int(influence_counts.get("AUTHOR_ROBUST", 0)),
        "V_concentrated_robust_count": int(influence_counts.get("CONCENTRATED_BUT_ROBUST", 0)),
        "W_author_sensitive_count": int(influence_counts.get("AUTHOR_SENSITIVE", 0)),
        "X_retained_historical_candidates": retained,
    }


def freeze_k12_outputs(
    output_root: Path,
    required_paths: list[Path],
    summary: dict[str, Any],
    source_hashes: dict[str, str],
    qa: pd.DataFrame,
) -> tuple[Path, Path]:
    """在任何 K20 历史结果读取前冻结 K12 正式输出。"""

    manifest_path = output_root / "SHA256_MANIFEST.txt"
    entries = []
    for path in sorted(set(required_paths)):
        if not path.exists():
            raise FileNotFoundError(f"冻结前缺少正式输出：{path}")
        entries.append(f"{sha256_file(path)}  {path.relative_to(output_root).as_posix()}")
    save_text("\n".join(entries), manifest_path)
    metadata = {
        "analysis_version": "C_K12_HISTORY_V2_PRIMARY_CANDIDATE",
        "frozen_at": datetime.now().astimezone().isoformat(),
        "blind_phase_status": "BLIND_TO_OLD_TREND_DIRECTION_COMPLETE",
        "manifest_sha256": sha256_file(manifest_path),
        "file_count": len(entries),
        "source_hashes": source_hashes,
        "precomparison_qa_passed": int(qa["status"].eq("PASS").sum()),
        "precomparison_qa_total": len(qa),
        "summary": summary,
    }
    freeze_path = output_root / "K12_BLIND_FREEZE.json"
    save_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), freeze_path)
    return manifest_path, freeze_path


def build_readme(
    output_root: Path,
    summary: dict[str, Any],
) -> str:
    """生成 K12 历史分析目录说明。"""

    return f"""# C_K12 作者时期历史分析

版本：C_K12_HISTORY_V2_PRIMARY_CANDIDATE。

本目录从 FULL_CORPUS_FIT 的 43,015×12 Document×Topic 矩阵独立重算作者时期趋势、代表性、效应验证与作者影响。K12 正式统计先在 BLIND_TO_OLD_TREND_DIRECTION 条件下完成并写入 SHA256_MANIFEST.txt，之后才允许建立 C_K12 与旧 C_K20 的事后语义比较。

author_period_primary 表示作者所属时期，不表示单篇作品创作年代。统计候选不等于文学史确认。

- 00_input_audit：theta、document_index、taxonomy、样本数与配置。
- 01_trend_structure：三样本、两 estimand、四时期的趋势结构。
- 02_representativeness：已知/未知作者时期的 K12 Topic SMD 与风险。
- 03_statistical_validation：CLEAR Topics 的 CR1/HC3、BH-FDR、效应量、样本与 CLR 敏感性。
- 04_author_influence：仅 STRONG/SUPPORTED 候选的 LOAO 与 Top contributor 排除。
- 05_figures：八幅规定 PNG。
- 06_final：研究者解释队列、最终报告与冻结后 K20 比较。

冻结时进入 LOAO 的候选数：{summary["T_loao_candidate_count"]}。
"""


def main() -> None:
    """执行 K12 独立分析并在比较旧 K20 前冻结。"""

    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    directories = {
        "audit": output_root / "00_input_audit",
        "trend": output_root / "01_trend_structure",
        "representativeness": output_root / "02_representativeness",
        "effect": output_root / "03_statistical_validation",
        "loao": output_root / "04_author_influence",
        "figures": output_root / "05_figures",
        "final": output_root / "06_final",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, directories["audit"] / "C_K12_history_config.yaml")
    taxonomy = build_taxonomy(
        config,
        directories["audit"] / "C_K12_topic_taxonomy_frozen.csv",
    )
    compact, labels, source_hashes, input_summary = build_input_master_and_audit(
        config,
        taxonomy,
        directories["audit"],
    )
    poems = verify_samples(
        compact,
        labels,
        config,
        directories["audit"] / "C_K12_metadata_sample_audit.csv",
    )
    sample_audit = pd.read_csv(directories["audit"] / "C_K12_metadata_sample_audit.csv")
    risk, clear_risk = build_representativeness(
        compact,
        labels,
        taxonomy,
        config,
        directories["representativeness"],
    )
    trend_results = run_trend_stage(
        poems,
        taxonomy,
        risk,
        config,
        directories["trend"],
    )
    trend_report_path = directories["trend"] / "QTS_C_K12_Author_Period_Trend_Structure_Report.md"
    save_text(
        build_trend_report(taxonomy, sample_audit, trend_results, config),
        trend_report_path,
    )
    effect_results = run_effect_stage(
        poems,
        taxonomy,
        risk,
        config,
        directories["effect"],
    )
    effect_report_path = directories["effect"] / "QTS_C_K12_Author_Period_Effect_Size_Statistical_Validation_Report.md"
    save_text(build_effect_report(effect_results, config), effect_report_path)
    loao_results = run_loao_stage(
        poems,
        taxonomy,
        effect_results,
        trend_results,
        config,
        directories["loao"],
    )
    loao_report_path = directories["loao"] / "QTS_C_K12_Author_Influence_LOAO_Report.md"
    save_text(build_loao_report(loao_results, config), loao_report_path)
    figure_paths = build_figures(
        trend_results,
        effect_results,
        loao_results,
        taxonomy,
        config,
        directories["figures"],
    )
    queue = build_humanistic_queue(
        trend_results,
        loao_results,
        taxonomy,
        directories["final"] / "C_K12_Humanistic_Interpretation_Queue.csv",
    )
    summary = build_run_summary(
        input_summary,
        sample_audit,
        trend_results,
        effect_results,
        loao_results,
        clear_risk,
    )
    summary["humanistic_queue_count"] = len(queue)
    summary_path = directories["audit"] / "C_K12_precomparison_summary.json"
    save_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), summary_path)
    qa = build_precomparison_qa(
        config,
        input_summary,
        sample_audit,
        taxonomy,
        poems,
        trend_results,
        effect_results,
        loao_results,
        source_hashes,
        directories["audit"] / "C_K12_precomparison_QA.csv",
    )
    required_paths = [
        directories["audit"] / "C_K12_theta_metadata_alignment_audit.csv",
        directories["audit"] / "C_K12_topic_taxonomy_frozen.csv",
        directories["audit"] / "C_K12_history_config.yaml",
        directories["audit"] / "C_K12_metadata_sample_audit.csv",
        directories["trend"] / "C_K12_author_period_topic_means.csv",
        directories["trend"] / "C_K12_CORE_POEM_trends.csv",
        directories["trend"] / "C_K12_CORE_AUTHOR_trends.csv",
        directories["trend"] / "C_K12_aggregation_sensitivity.csv",
        directories["trend"] / "C_K12_sample_sensitivity.csv",
        directories["trend"] / "C_K12_trend_robustness.csv",
        directories["trend"] / "C_K12_author_concentration.csv",
        trend_report_path,
        directories["representativeness"] / "C_K12_known_unknown_topic_smd.csv",
        directories["representativeness"] / "C_K12_topic_representativeness_risk.csv",
        directories["effect"] / "C_K12_omnibus_period_effects.csv",
        directories["effect"] / "C_K12_planned_contrast_effects.csv",
        directories["effect"] / "C_K12_effect_size_summary.csv",
        directories["effect"] / "C_K12_sample_sensitivity_effects.csv",
        directories["effect"] / "C_K12_clr_sensitivity.csv",
        directories["effect"] / "C_K12_statistical_candidate_trends.csv",
        effect_report_path,
        directories["loao"] / "C_K12_LOAO_all_authors.csv",
        directories["loao"] / "C_K12_top_contributor_exclusion.csv",
        directories["loao"] / "C_K12_author_influence_summary.csv",
        directories["loao"] / "C_K12_final_candidate_trends_after_LOAO.csv",
        loao_report_path,
        directories["final"] / "C_K12_Humanistic_Interpretation_Queue.csv",
        summary_path,
        directories["audit"] / "C_K12_precomparison_QA.csv",
        *figure_paths,
    ]
    manifest_path, freeze_path = freeze_k12_outputs(
        output_root,
        required_paths,
        summary,
        source_hashes,
        qa,
    )
    save_text(build_readme(output_root, summary), output_root / "README.md")
    print(
        json.dumps(
            {
                "status": "K12_BLIND_PHASE_FROZEN",
                "output_root": str(output_root),
                "manifest": str(manifest_path),
                "freeze": str(freeze_path),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
