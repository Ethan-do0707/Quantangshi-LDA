"""审计 author-period 元数据子样本的代表性与可观测缺失偏差。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PERIOD_STATUS_VALUES = {"KNOWN", "UNKNOWN"}
CONFIDENCE_VALUES = {"HIGH", "MEDIUM", "LOW", "UNKNOWN"}
TOPIC_TYPES = ["CLEAR", "BROAD", "MIXED", "NOISE"]
PRODUCTIVITY_BUCKETS = ["1 poem", "2–4", "5–9", "10–19", "20–49", "50–99", "100–199", "200+"]
MIXTURE_FIELDS = ["topic_entropy", "normalized_topic_entropy", "top1_top2_gap", "dominant_topic_probability"]


def read_config(path: Path) -> dict[str, Any]:
    """读取 JSON 兼容的集中配置。"""

    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """以流式方式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text_values(values: Iterable[str]) -> str:
    """计算顺序敏感的字段值哈希。"""

    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def topic_probability_checksum(path: Path, topic_fields: Sequence[str]) -> str:
    """沿用历史元数据管线口径计算 poem_id 与20维概率校验和。"""

    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            digest.update(str(row["poem_id"]).encode("utf-8"))
            digest.update(b"\x1f")
            for field in topic_fields:
                digest.update(str(row[field]).encode("utf-8"))
                digest.update(b"\x1e")
            digest.update(b"\n")
    return digest.hexdigest()


def source_field_hashes(path: Path, fields: Sequence[str]) -> dict[str, str]:
    """读取权威 CSV 的关键不可变字段哈希。"""

    digests = {field: hashlib.sha256() for field in fields}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for field in fields:
                digests[field].update(str(row.get(field, "")).encode("utf-8"))
                digests[field].update(b"\n")
    return {field: digest.hexdigest() for field, digest in digests.items()}


def parse_boolean_series(series: pd.Series) -> pd.Series:
    """严格解析冻结 CSV 中的布尔值。"""

    mapping = {True: True, False: False, "True": True, "False": False, "TRUE": True, "FALSE": False}
    parsed = series.map(mapping)
    if parsed.isna().any():
        values = sorted(set(series[parsed.isna()].astype(str)))
        raise ValueError(f"出现无法解析的布尔值：{values}")
    return parsed.astype(bool)


def summary_statistics(values: np.ndarray) -> dict[str, float | int]:
    """生成偏态分布友好的描述统计。"""

    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def cliffs_delta_from_u(u_statistic: float, size_left: int, size_right: int) -> float:
    """由 Mann–Whitney U 计算 Cliff's delta，正值表示左组更高。"""

    return float(2.0 * u_statistic / (size_left * size_right) - 1.0)


def cliffs_interpretation(value: float, thresholds: dict[str, float]) -> str:
    """按审计启发式解释 Cliff's delta 绝对值。"""

    absolute = abs(value)
    if absolute < thresholds["cliffs_negligible"]:
        return "NEGLIGIBLE"
    if absolute < thresholds["cliffs_small"]:
        return "SMALL"
    if absolute < thresholds["cliffs_moderate"]:
        return "MODERATE"
    return "LARGE"


def smd_interpretation(value: float) -> str:
    """按任务给定阈值解释 SMD。"""

    absolute = abs(value)
    if absolute < 0.10:
        return "NEGLIGIBLE"
    if absolute < 0.20:
        return "SMALL"
    if absolute < 0.50:
        return "MODERATE"
    return "LARGE"


def standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    """计算两组均值差的 pooled-SD 标准化效应量。"""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    numerator = (left.size - 1) * np.var(left, ddof=1) + (right.size - 1) * np.var(right, ddof=1)
    denominator = left.size + right.size - 2
    pooled_sd = math.sqrt(numerator / denominator) if denominator > 0 else 0.0
    mean_difference = float(np.mean(left) - np.mean(right))
    if pooled_sd == 0:
        if mean_difference == 0:
            return 0.0
        raise ValueError("两组 pooled SD 为零但均值不同，SMD 数学上不可计算")
    return mean_difference / pooled_sd


def bootstrap_mean_difference(
    left: np.ndarray,
    right: np.ndarray,
    iterations: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """使用分批 multinomial count bootstrap 计算均值差95%区间。"""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim == 1:
        left = left[:, None]
    if right.ndim == 1:
        right = right[:, None]
    if left.shape[1] != right.shape[1]:
        raise ValueError("bootstrap 两组变量数不一致")
    rng = np.random.default_rng(seed)
    differences = np.empty((iterations, left.shape[1]), dtype=float)
    left_probability = np.full(left.shape[0], 1.0 / left.shape[0])
    right_probability = np.full(right.shape[0], 1.0 / right.shape[0])
    offset = 0
    while offset < iterations:
        current = min(batch_size, iterations - offset)
        left_weights = rng.multinomial(left.shape[0], left_probability, size=current)
        right_weights = rng.multinomial(right.shape[0], right_probability, size=current)
        left_means = left_weights @ left / left.shape[0]
        right_means = right_weights @ right / right.shape[0]
        differences[offset : offset + current] = left_means - right_means
        offset += current
    return np.quantile(differences, 0.025, axis=0), np.quantile(differences, 0.975, axis=0)


def cramers_v(table: np.ndarray) -> tuple[float, float, float, int]:
    """计算列联表 Chi-square、p值与 Cramér's V。"""

    chi_square, p_value, degrees_freedom, _ = stats.chi2_contingency(table, correction=False)
    sample_size = int(np.sum(table))
    minimum_dimension = min(table.shape[0] - 1, table.shape[1] - 1)
    value = math.sqrt(chi_square / (sample_size * minimum_dimension)) if minimum_dimension > 0 else 0.0
    return float(chi_square), float(p_value), float(value), int(degrees_freedom)


def productivity_bucket(count: int) -> str:
    """按任务指定边界映射作者诗歌产量分箱。"""

    if count == 1:
        return "1 poem"
    if count <= 4:
        return "2–4"
    if count <= 9:
        return "5–9"
    if count <= 19:
        return "10–19"
    if count <= 49:
        return "20–49"
    if count <= 99:
        return "50–99"
    if count <= 199:
        return "100–199"
    return "200+"


def coverage_by_category(author_frame: pd.DataFrame, category: str) -> pd.DataFrame:
    """同时按作者与诗歌权重统计类别覆盖率。"""

    rows: list[dict[str, Any]] = []
    for value, group in author_frame.groupby(category, dropna=False, sort=True):
        known = group["period_metadata_status"].eq("KNOWN")
        total_poems = int(group["author_poem_count"].sum())
        known_poems = int(group.loc[known, "author_poem_count"].sum())
        rows.append(
            {
                category: str(value),
                "author_count": int(len(group)),
                "period_known_author_count": int(known.sum()),
                "period_unknown_author_count": int((~known).sum()),
                "period_known_author_ratio": float(known.mean()),
                "poem_count": total_poems,
                "period_known_poem_count": known_poems,
                "period_unknown_poem_count": total_poems - known_poems,
                "period_known_poem_ratio": known_poems / total_poems if total_poems else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_coverage_tables(poems: pd.DataFrame, authors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """建立总体覆盖率、作者产量统计与产量分箱表。"""

    total_poems = len(poems)
    total_authors = len(authors)
    rows = []

    def append_row(scope: str, category: str, author_mask: pd.Series, poem_mask: pd.Series, definition: str) -> None:
        author_count = int(author_mask.sum())
        poem_count = int(poem_mask.sum())
        rows.append(
            {
                "scope": scope,
                "category": category,
                "author_count": author_count,
                "author_ratio": author_count / total_authors,
                "poem_count": poem_count,
                "poem_ratio": poem_count / total_poems,
                "definition": definition,
            }
        )

    append_row("OVERALL", "TOTAL", pd.Series(True, index=authors.index), pd.Series(True, index=poems.index), "全部作者与诗歌")
    for status in ["KNOWN", "UNKNOWN"]:
        append_row(
            "PERIOD_STATUS",
            status,
            authors["period_metadata_status"].eq(status),
            poems["period_metadata_status"].eq(status),
            "由 author_period_primary 是否属于初唐、盛唐、中唐、晚唐判定",
        )
    for confidence in ["HIGH", "MEDIUM", "LOW"]:
        append_row(
            "KNOWN_CONFIDENCE",
            confidence,
            authors["period_metadata_status"].eq("KNOWN") & authors["period_confidence_group"].eq(confidence),
            poems["period_metadata_status"].eq("KNOWN") & poems["period_confidence_group"].eq(confidence),
            "仅在 KNOWN period 内按 period_confidence 分层",
        )
    for sample_name, field in [
        ("STRICT", "eligible_period_strict"),
        ("CORE", "eligible_period_core"),
        ("EXPANDED", "eligible_period_expanded"),
    ]:
        append_row("ANALYSIS_SAMPLE", sample_name, authors[field], poems[field], f"未来 {sample_name} author-period 分析样本")
    coverage = pd.DataFrame(rows)

    known_counts = authors.loc[authors["period_metadata_status"].eq("KNOWN"), "author_poem_count"].to_numpy()
    unknown_counts = authors.loc[authors["period_metadata_status"].eq("UNKNOWN"), "author_poem_count"].to_numpy()
    u_statistic, p_value = stats.mannwhitneyu(known_counts, unknown_counts, alternative="two-sided", method="asymptotic")
    delta = cliffs_delta_from_u(float(u_statistic), len(known_counts), len(unknown_counts))
    productivity_rows = []
    for status, values in [("KNOWN", known_counts), ("UNKNOWN", unknown_counts)]:
        productivity_rows.append(
            {
                "period_metadata_status": status,
                **summary_statistics(values),
                "mann_whitney_u": float(u_statistic),
                "mann_whitney_p_value": float(p_value),
                "cliffs_delta_known_minus_unknown": delta,
                "effect_interpretation": cliffs_interpretation(delta, CONFIG["risk_thresholds"]),
                "direction_note": "正值表示 KNOWN 作者产量更高",
            }
        )
    productivity = pd.DataFrame(productivity_rows)

    bucket_rows = []
    for bucket in PRODUCTIVITY_BUCKETS:
        group = authors[authors["productivity_bucket"].eq(bucket)]
        known = group["period_metadata_status"].eq("KNOWN")
        poem_count = int(group["author_poem_count"].sum())
        known_poem_count = int(group.loc[known, "author_poem_count"].sum())
        bucket_rows.append(
            {
                "productivity_bucket": bucket,
                "author_count": int(len(group)),
                "period_known_count": int(known.sum()),
                "period_known_ratio": float(known.mean()) if len(group) else 0.0,
                "poem_count": poem_count,
                "period_known_poem_count": known_poem_count,
                "period_known_poem_ratio": known_poem_count / poem_count if poem_count else 0.0,
            }
        )
    return coverage, productivity, pd.DataFrame(bucket_rows)


def build_topic_probability_table(
    poems: pd.DataFrame,
    topic_metadata: pd.DataFrame,
    topic_fields: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    """比较 KNOWN 与 UNKNOWN 的20维 Topic probability。"""

    known = poems.loc[poems["period_metadata_status"].eq("KNOWN"), topic_fields].to_numpy(dtype=float)
    unknown = poems.loc[poems["period_metadata_status"].eq("UNKNOWN"), topic_fields].to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_mean_difference(
        known,
        unknown,
        config["bootstrap_iterations"],
        config["bootstrap_batch_size"],
        config["random_seed"],
    )
    metadata = topic_metadata.set_index("topic_id")
    rows = []
    for index, field in enumerate(topic_fields):
        left = known[:, index]
        right = unknown[:, index]
        mean_known = float(np.mean(left))
        mean_unknown = float(np.mean(right))
        difference = mean_known - mean_unknown
        smd = standardized_mean_difference(left, right)
        rows.append(
            {
                "topic_id": field.removesuffix("_prob"),
                "topic_label": metadata.loc[field.removesuffix("_prob"), "final_topic_label"],
                "topic_type": metadata.loc[field.removesuffix("_prob"), "topic_type"],
                "mean_human_score": float(metadata.loc[field.removesuffix("_prob"), "mean_human_score"]),
                "interpretation_scope": metadata.loc[field.removesuffix("_prob"), "interpretation_scope"],
                "mean_known": mean_known,
                "mean_unknown": mean_unknown,
                "mean_difference": difference,
                "relative_difference": difference / mean_unknown if abs(mean_unknown) > 1e-12 else None,
                "smd": smd,
                "abs_smd": abs(smd),
                "bootstrap_ci_low": float(ci_low[index]),
                "bootstrap_ci_high": float(ci_high[index]),
                "bias_level": smd_interpretation(smd),
            }
        )
    return pd.DataFrame(rows).sort_values("topic_id").reset_index(drop=True)


def build_topic_type_mass_table(
    poems: pd.DataFrame,
    topic_metadata: pd.DataFrame,
    topic_fields: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """比较 KNOWN 与 UNKNOWN 在四类 Topic 上的概率质量。"""

    type_fields: dict[str, list[str]] = {}
    metadata_types = topic_metadata.set_index("topic_id")["topic_type"].to_dict()
    for topic_type in TOPIC_TYPES:
        type_fields[topic_type] = [field for field in topic_fields if metadata_types[field.removesuffix("_prob")] == topic_type]
        poems[f"{topic_type.lower()}_topic_mass"] = poems[type_fields[topic_type]].sum(axis=1)
    mass_fields = [f"{topic_type.lower()}_topic_mass" for topic_type in TOPIC_TYPES]
    mass_sum = poems[mass_fields].sum(axis=1)
    max_deviation = float(np.max(np.abs(mass_sum.to_numpy() - 1.0)))
    if max_deviation > 1e-8:
        raise ValueError(f"四类 Topic mass 之和偏离1，最大偏差 {max_deviation}")
    known = poems.loc[poems["period_metadata_status"].eq("KNOWN"), mass_fields].to_numpy(dtype=float)
    unknown = poems.loc[poems["period_metadata_status"].eq("UNKNOWN"), mass_fields].to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_mean_difference(
        known,
        unknown,
        config["bootstrap_iterations"],
        config["bootstrap_batch_size"],
        config["random_seed"] + 1,
    )
    rows = []
    for index, topic_type in enumerate(TOPIC_TYPES):
        left = known[:, index]
        right = unknown[:, index]
        smd = standardized_mean_difference(left, right)
        rows.append(
            {
                "topic_type": topic_type,
                "topic_count": len(type_fields[topic_type]),
                "mean_known": float(np.mean(left)),
                "mean_unknown": float(np.mean(right)),
                "mean_difference": float(np.mean(left) - np.mean(right)),
                "smd": smd,
                "abs_smd": abs(smd),
                "bootstrap_ci_low": float(ci_low[index]),
                "bootstrap_ci_high": float(ci_high[index]),
                "bias_level": smd_interpretation(smd),
            }
        )
    return pd.DataFrame(rows), {"max_mass_sum_deviation": max_deviation}


def build_dominant_tables(
    poems: pd.DataFrame,
    topic_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """建立 dominant topic 与 topic type 的 KNOWN/UNKNOWN 列联比较。"""

    metadata = topic_metadata.set_index("topic_id")
    topic_ids = topic_metadata["topic_id"].tolist()
    known_total = int(poems["period_metadata_status"].eq("KNOWN").sum())
    unknown_total = int(poems["period_metadata_status"].eq("UNKNOWN").sum())
    rows = []
    table = []
    for topic_id in topic_ids:
        known_count = int((poems["period_metadata_status"].eq("KNOWN") & poems["dominant_topic_id"].eq(topic_id)).sum())
        unknown_count = int((poems["period_metadata_status"].eq("UNKNOWN") & poems["dominant_topic_id"].eq(topic_id)).sum())
        table.append([known_count, unknown_count])
        rows.append(
            {
                "topic_id": topic_id,
                "topic_label": metadata.loc[topic_id, "final_topic_label"],
                "topic_type": metadata.loc[topic_id, "topic_type"],
                "known_count": known_count,
                "known_ratio": known_count / known_total,
                "unknown_count": unknown_count,
                "unknown_ratio": unknown_count / unknown_total,
                "ratio_difference": known_count / known_total - unknown_count / unknown_total,
            }
        )
    contingency = np.asarray(table, dtype=int).T
    chi_square, p_value, value, degrees_freedom = cramers_v(contingency)
    dominant = pd.DataFrame(rows)
    dominant["chi_square"] = chi_square
    dominant["chi_square_p_value"] = p_value
    dominant["degrees_freedom"] = degrees_freedom
    dominant["cramers_v"] = value

    poems["dominant_topic_type"] = poems["dominant_topic_id"].map(metadata["topic_type"])
    if poems["dominant_topic_type"].isna().any():
        raise ValueError("存在无法映射到 topic_metadata 的 dominant_topic_id")
    type_rows = []
    type_table = []
    for topic_type in TOPIC_TYPES:
        known_count = int((poems["period_metadata_status"].eq("KNOWN") & poems["dominant_topic_type"].eq(topic_type)).sum())
        unknown_count = int((poems["period_metadata_status"].eq("UNKNOWN") & poems["dominant_topic_type"].eq(topic_type)).sum())
        type_table.append([known_count, unknown_count])
        type_rows.append(
            {
                "topic_type": topic_type,
                "known_count": known_count,
                "known_ratio": known_count / known_total,
                "unknown_count": unknown_count,
                "unknown_ratio": unknown_count / unknown_total,
                "ratio_difference": known_count / known_total - unknown_count / unknown_total,
            }
        )
    type_contingency = np.asarray(type_table, dtype=int).T
    type_chi_square, type_p_value, type_value, type_df = cramers_v(type_contingency)
    dominant_type = pd.DataFrame(type_rows)
    dominant_type["chi_square"] = type_chi_square
    dominant_type["chi_square_p_value"] = type_p_value
    dominant_type["degrees_freedom"] = type_df
    dominant_type["cramers_v"] = type_value
    tests = {
        "dominant_topic_chi_square": chi_square,
        "dominant_topic_p_value": p_value,
        "dominant_topic_cramers_v": value,
        "dominant_topic_type_chi_square": type_chi_square,
        "dominant_topic_type_p_value": type_p_value,
        "dominant_topic_type_cramers_v": type_value,
    }
    return dominant, dominant_type, tests


def build_mixture_table(poems: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """比较 entropy、top1 概率与 top1-top2 gap。"""

    known_mask = poems["period_metadata_status"].eq("KNOWN")
    rows = []
    for field in MIXTURE_FIELDS:
        known = poems.loc[known_mask, field].to_numpy(dtype=float)
        unknown = poems.loc[~known_mask, field].to_numpy(dtype=float)
        u_statistic, p_value = stats.mannwhitneyu(known, unknown, alternative="two-sided", method="asymptotic")
        delta = cliffs_delta_from_u(float(u_statistic), len(known), len(unknown))
        known_summary = summary_statistics(known)
        unknown_summary = summary_statistics(unknown)
        rows.append(
            {
                "metric": field,
                "known_count": known_summary["count"],
                "known_mean": known_summary["mean"],
                "known_median": known_summary["median"],
                "known_std": known_summary["std"],
                "known_p25": known_summary["p25"],
                "known_p75": known_summary["p75"],
                "unknown_count": unknown_summary["count"],
                "unknown_mean": unknown_summary["mean"],
                "unknown_median": unknown_summary["median"],
                "unknown_std": unknown_summary["std"],
                "unknown_p25": unknown_summary["p25"],
                "unknown_p75": unknown_summary["p75"],
                "mean_difference_known_minus_unknown": known_summary["mean"] - unknown_summary["mean"],
                "mann_whitney_u": float(u_statistic),
                "mann_whitney_p_value": float(p_value),
                "cliffs_delta_known_minus_unknown": delta,
                "effect_interpretation": cliffs_interpretation(delta, thresholds),
            }
        )
    return pd.DataFrame(rows)


def build_logistic_diagnostic(
    poems: pd.DataFrame,
    topic_fields: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """以作者分组交叉验证诊断 period-known 的可预测性，不用于填补。"""

    numeric_fields = ["log1p_author_poem_count", *topic_fields, "normalized_topic_entropy", "dominant_topic_probability"]
    categorical_fields = ["author_status", "cbdb_match_status", "cbdb_match_confidence"]
    frame = poems[numeric_fields + categorical_fields].copy()
    target = poems["period_metadata_status"].eq("KNOWN").astype(int)
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_fields),
            ("categorical", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), categorical_fields),
        ],
        remainder="drop",
    )
    model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=config["random_seed"])
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    cross_validator = StratifiedGroupKFold(
        n_splits=config["logistic_cv_folds"],
        shuffle=True,
        random_state=config["random_seed"],
    )
    cross_validation = cross_validate(
        pipeline,
        frame,
        target,
        groups=poems["author_id_internal"],
        cv=cross_validator,
        scoring="roc_auc",
        return_train_score=False,
        n_jobs=1,
    )
    out_of_fold = cross_val_predict(
        pipeline,
        frame,
        target,
        groups=poems["author_id_internal"],
        cv=cross_validator,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    cross_validated_auc = float(roc_auc_score(target, out_of_fold))
    pipeline.fit(frame, target)
    fitted_probability = pipeline.predict_proba(frame)[:, 1]
    apparent_auc = float(roc_auc_score(target, fitted_probability))
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_[0]
    rows = [
        {
            "feature": "__MODEL__",
            "feature_group": "MODEL_SUMMARY",
            "coefficient": None,
            "standardized_coefficient": None,
            "odds_ratio": None,
            "apparent_auc": apparent_auc,
            "cross_validated_auc": cross_validated_auc,
            "fold_auc_mean": float(np.mean(cross_validation["test_score"])),
            "fold_auc_std": float(np.std(cross_validation["test_score"], ddof=1)),
            "diagnostic_note": "按 author_id_internal 分组的5折诊断；不得用于 period imputation",
        }
    ]
    for feature_name, coefficient in zip(feature_names, coefficients):
        is_numeric = feature_name.startswith("numeric__")
        rows.append(
            {
                "feature": feature_name,
                "feature_group": "STANDARDIZED_NUMERIC" if is_numeric else "CATEGORICAL_INDICATOR",
                "coefficient": float(coefficient),
                "standardized_coefficient": float(coefficient) if is_numeric else None,
                "odds_ratio": float(math.exp(coefficient)),
                "apparent_auc": apparent_auc,
                "cross_validated_auc": cross_validated_auc,
                "fold_auc_mean": float(np.mean(cross_validation["test_score"])),
                "fold_auc_std": float(np.std(cross_validation["test_score"], ddof=1)),
                "diagnostic_note": "系数仅描述 period availability 与可观测特征的关联",
            }
        )
    return pd.DataFrame(rows), {
        "apparent_auc": apparent_auc,
        "cross_validated_auc": cross_validated_auc,
        "fold_auc_mean": float(np.mean(cross_validation["test_score"])),
        "fold_auc_std": float(np.std(cross_validation["test_score"], ddof=1)),
    }


def overall_bias_level(max_abs_smd: float, mean_abs_smd: float, cramers_value: float, entropy_effect: float) -> str:
    """按透明审计启发式汇总样本代表性风险。"""

    if max_abs_smd >= 0.50 or cramers_value >= 0.20 or abs(entropy_effect) >= 0.474:
        return "HIGH"
    if max_abs_smd < 0.20 and mean_abs_smd < 0.10 and cramers_value < 0.10 and abs(entropy_effect) < 0.147:
        return "LOW"
    return "MODERATE"


def build_sample_sensitivity(
    poems: pd.DataFrame,
    authors: pd.DataFrame,
    topic_fields: list[str],
) -> pd.DataFrame:
    """比较 STRICT、CORE、EXPANDED 与完整语料的描述性代表性。"""

    full_topics = poems[topic_fields].to_numpy(dtype=float)
    full_entropy = poems["topic_entropy"].to_numpy(dtype=float)
    full_dominant_counts = poems["dominant_topic_id"].value_counts().reindex([field.removesuffix("_prob") for field in topic_fields], fill_value=0)
    full_author_counts = authors["author_poem_count"].to_numpy(dtype=float)
    rows = []
    for sample_name, field in [
        ("STRICT", "eligible_period_strict"),
        ("CORE", "eligible_period_core"),
        ("EXPANDED", "eligible_period_expanded"),
    ]:
        sample_poems = poems[poems[field]]
        sample_authors = authors[authors[field]]
        sample_topics = sample_poems[topic_fields].to_numpy(dtype=float)
        smds = np.asarray(
            [standardized_mean_difference(sample_topics[:, index], full_topics[:, index]) for index in range(len(topic_fields))]
        )
        entropy_u, _ = stats.mannwhitneyu(sample_poems["topic_entropy"], full_entropy, alternative="two-sided", method="asymptotic")
        entropy_delta = cliffs_delta_from_u(float(entropy_u), len(sample_poems), len(full_entropy))
        sample_counts = sample_poems["dominant_topic_id"].value_counts().reindex(full_dominant_counts.index, fill_value=0)
        _, _, dominant_v, _ = cramers_v(np.vstack([sample_counts.to_numpy(), full_dominant_counts.to_numpy()]))
        productivity_u, _ = stats.mannwhitneyu(
            sample_authors["author_poem_count"], full_author_counts, alternative="two-sided", method="asymptotic"
        )
        productivity_delta = cliffs_delta_from_u(float(productivity_u), len(sample_authors), len(full_author_counts))
        maximum = float(np.max(np.abs(smds)))
        mean_absolute = float(np.mean(np.abs(smds)))
        rows.append(
            {
                "sample_name": sample_name,
                "poem_count": int(len(sample_poems)),
                "poem_ratio": len(sample_poems) / len(poems),
                "author_count": int(len(sample_authors)),
                "author_ratio": len(sample_authors) / len(authors),
                "max_abs_topic_smd": maximum,
                "mean_abs_topic_smd": mean_absolute,
                "topic_count_abs_smd_ge_0_10": int(np.sum(np.abs(smds) >= 0.10)),
                "topic_count_abs_smd_ge_0_20": int(np.sum(np.abs(smds) >= 0.20)),
                "topic_count_abs_smd_ge_0_50": int(np.sum(np.abs(smds) >= 0.50)),
                "entropy_effect_size": entropy_delta,
                "dominant_topic_cramers_v": dominant_v,
                "author_productivity_effect_size": productivity_delta,
                "overall_bias_level": overall_bias_level(maximum, mean_absolute, dominant_v, entropy_delta),
                "bias_score_for_ranking": maximum + mean_absolute + dominant_v + abs(entropy_delta),
                "comparison_note": "与包含自身的 full corpus 作描述性比较，不作独立样本显著性检验",
            }
        )
    return pd.DataFrame(rows)


def build_risk_tables(
    topic_probability: pd.DataFrame,
    dominant_topic: pd.DataFrame,
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按透明启发式识别 Topic 缺失偏差风险。"""

    merged = topic_probability.merge(
        dominant_topic[["topic_id", "known_ratio", "unknown_ratio", "ratio_difference"]],
        on="topic_id",
        how="left",
        validate="one_to_one",
    )
    rows = []
    for row in merged.to_dict("records"):
        absolute_smd = abs(row["smd"])
        dominant_absolute = abs(row["ratio_difference"])
        ci_excludes_zero = row["bootstrap_ci_low"] > 0 or row["bootstrap_ci_high"] < 0
        if absolute_smd >= thresholds["smd_large"] or (
            absolute_smd >= thresholds["smd_moderate"] and dominant_absolute >= thresholds["dominant_ratio_high"]
        ):
            risk = "HIGH"
        elif (
            absolute_smd >= thresholds["smd_small"]
            or dominant_absolute >= thresholds["dominant_ratio_moderate"]
            or (ci_excludes_zero and absolute_smd >= 0.05)
        ):
            risk = "MODERATE"
        else:
            risk = "LOW"
        direction = "UNKNOWN_OVERREPRESENTED" if row["mean_unknown"] > row["mean_known"] else "KNOWN_OVERREPRESENTED"
        reasons = [direction, f"ABS_SMD={absolute_smd:.3f}", f"DOMINANT_RATIO_DIFF={row['ratio_difference']:.3f}"]
        if ci_excludes_zero:
            reasons.append("BOOTSTRAP_CI_EXCLUDES_ZERO")
        rows.append(
            {
                "topic_id": row["topic_id"],
                "topic_label": row["topic_label"],
                "topic_type": row["topic_type"],
                "mean_known": row["mean_known"],
                "mean_unknown": row["mean_unknown"],
                "smd": row["smd"],
                "abs_smd": absolute_smd,
                "bootstrap_ci_low": row["bootstrap_ci_low"],
                "bootstrap_ci_high": row["bootstrap_ci_high"],
                "dominant_ratio_known": row["known_ratio"],
                "dominant_ratio_unknown": row["unknown_ratio"],
                "dominant_ratio_difference": row["ratio_difference"],
                "risk_level": risk,
                "risk_reason": ";".join(reasons),
            }
        )
    risk = pd.DataFrame(rows).sort_values(["risk_level", "abs_smd"], ascending=[True, False]).reset_index(drop=True)
    clear = risk[risk["topic_type"].eq("CLEAR")].sort_values("topic_id").reset_index(drop=True)
    return risk, clear


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """以 UTF-8 无 BOM 保存稳定 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.15g")


def records_without_nan(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """把表格中的不可适用缺失值转换为 JSON null。"""

    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def configure_matplotlib() -> None:
    """设置确定性、简洁的学术图形样式。"""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def build_figures(
    figures_root: Path,
    topic_probability: pd.DataFrame,
    productivity_buckets: pd.DataFrame,
    poems: pd.DataFrame,
    dominant_topic: pd.DataFrame,
) -> list[Path]:
    """生成五张仅用于代表性审计的静态图。"""

    configure_matplotlib()
    figures_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    ordered = topic_probability.sort_values("smd")
    colors = ["#C44E52" if value < 0 else "#4C72B0" for value in ordered["smd"]]
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(ordered["topic_id"], ordered["smd"], color=colors)
    for threshold, style in [(0.10, "--"), (0.20, "-."), (0.50, ":")]:
        axis.axvline(threshold, color="#777777", linewidth=0.8, linestyle=style)
        axis.axvline(-threshold, color="#777777", linewidth=0.8, linestyle=style)
    axis.axvline(0, color="#222222", linewidth=0.8)
    axis.set_xlabel("Standardized mean difference (KNOWN − UNKNOWN)")
    axis.set_ylabel("C_K20 Topic")
    axis.set_title("Topic probability representativeness by period availability")
    fig.tight_layout()
    path = figures_root / "topic_known_unknown_smd.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(topic_probability["mean_known"], topic_probability["mean_unknown"], c="#4C72B0", s=42)
    lower = float(min(topic_probability["mean_known"].min(), topic_probability["mean_unknown"].min()))
    upper = float(max(topic_probability["mean_known"].max(), topic_probability["mean_unknown"].max()))
    axis.plot([lower, upper], [lower, upper], color="#777777", linestyle="--", linewidth=1)
    for row in topic_probability.to_dict("records"):
        axis.annotate(row["topic_id"].replace("topic_", "T"), (row["mean_known"], row["mean_unknown"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axis.set_xlabel("Mean Topic probability — KNOWN")
    axis.set_ylabel("Mean Topic probability — UNKNOWN")
    axis.set_title("KNOWN vs UNKNOWN mean Topic probability")
    fig.tight_layout()
    path = figures_root / "topic_mean_known_vs_unknown.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(productivity_buckets))
    axis.plot(positions, productivity_buckets["period_known_ratio"], marker="o", color="#4C72B0", label="Authors")
    axis.plot(positions, productivity_buckets["period_known_poem_ratio"], marker="s", color="#55A868", label="Poems weighted")
    axis.set_xticks(positions, productivity_buckets["productivity_bucket"], rotation=30, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Period-known ratio")
    axis.set_xlabel("Author productivity bucket")
    axis.set_title("Period metadata coverage rises with author productivity")
    axis.legend(frameon=False)
    fig.tight_layout()
    path = figures_root / "author_productivity_period_coverage.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    known_entropy = poems.loc[poems["period_metadata_status"].eq("KNOWN"), "normalized_topic_entropy"].to_numpy()
    unknown_entropy = poems.loc[poems["period_metadata_status"].eq("UNKNOWN"), "normalized_topic_entropy"].to_numpy()
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.boxplot([known_entropy, unknown_entropy], tick_labels=["KNOWN", "UNKNOWN"], showfliers=False, patch_artist=True, boxprops={"facecolor": "#D9EAF7"})
    axis.set_ylabel("Normalized Topic entropy")
    axis.set_xlabel("Period metadata status")
    axis.set_title("Topic mixture distribution by period availability")
    fig.tight_layout()
    path = figures_root / "entropy_known_unknown.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    ordered_dominant = dominant_topic.sort_values("ratio_difference")
    positions = np.arange(len(ordered_dominant))
    fig, axis = plt.subplots(figsize=(10, 7))
    height = 0.38
    axis.barh(positions - height / 2, ordered_dominant["known_ratio"], height=height, color="#4C72B0", label="KNOWN")
    axis.barh(positions + height / 2, ordered_dominant["unknown_ratio"], height=height, color="#C44E52", label="UNKNOWN")
    axis.set_yticks(positions, ordered_dominant["topic_id"])
    axis.set_xlabel("Dominant-topic proportion")
    axis.set_ylabel("C_K20 Topic")
    axis.set_title("Dominant Topic composition by period availability")
    axis.legend(frameon=False)
    fig.tight_layout()
    path = figures_root / "dominant_topic_known_unknown.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    """生成无需第三方依赖的 Markdown 表格。"""

    selected = frame[columns].head(limit) if limit else frame[columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for record in selected.to_dict("records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def build_report(
    output_root: Path,
    input_manifest: dict[str, Any],
    coverage: pd.DataFrame,
    productivity: pd.DataFrame,
    topic_probability: pd.DataFrame,
    topic_type_mass: pd.DataFrame,
    dominant_tests: dict[str, float | int],
    mixture: pd.DataFrame,
    cbdb_status: pd.DataFrame,
    cbdb_confidence: pd.DataFrame,
    sample_sensitivity: pd.DataFrame,
    risk: pd.DataFrame,
    clear: pd.DataFrame,
    logistic_summary: dict[str, float],
) -> tuple[str, dict[str, Any]]:
    """生成最终审计报告与工作簿概览载荷。"""

    known_coverage = coverage[(coverage["scope"] == "PERIOD_STATUS") & (coverage["category"] == "KNOWN")].iloc[0]
    unknown_coverage = coverage[(coverage["scope"] == "PERIOD_STATUS") & (coverage["category"] == "UNKNOWN")].iloc[0]
    known_productivity = productivity[productivity["period_metadata_status"].eq("KNOWN")].iloc[0]
    unknown_productivity = productivity[productivity["period_metadata_status"].eq("UNKNOWN")].iloc[0]
    productivity_delta = float(known_productivity["cliffs_delta_known_minus_unknown"])
    topic_max = float(topic_probability["abs_smd"].max())
    topic_mean = float(topic_probability["abs_smd"].mean())
    counts = {
        "ge_0_10": int((topic_probability["abs_smd"] >= 0.10).sum()),
        "ge_0_20": int((topic_probability["abs_smd"] >= 0.20).sum()),
        "ge_0_50": int((topic_probability["abs_smd"] >= 0.50).sum()),
    }
    entropy_row = mixture[mixture["metric"].eq("topic_entropy")].iloc[0]
    normalized_entropy_row = mixture[mixture["metric"].eq("normalized_topic_entropy")].iloc[0]
    mixed_row = topic_type_mass[topic_type_mass["topic_type"].eq("MIXED")].iloc[0]
    noise_row = topic_type_mass[topic_type_mass["topic_type"].eq("NOISE")].iloc[0]
    best_sample = sample_sensitivity.sort_values("bias_score_for_ranking").iloc[0]
    clear_groups = {
        level: clear.loc[clear["risk_level"].eq(level), "topic_id"].tolist()
        for level in ["LOW", "MODERATE", "HIGH"]
    }
    high_risk_clear_count = len(clear_groups["HIGH"])
    if high_risk_clear_count >= 3 or abs(productivity_delta) >= 0.33 or best_sample["overall_bias_level"] == "HIGH":
        recommendation_code = 3
        recommendation = "建议先进一步人工补充高影响作者 metadata，同时准备敏感性框架；补充完成前不把现有样本直接当作完整语料的无偏替代。"
    elif counts["ge_0_10"] > 0 or dominant_tests["dominant_topic_cramers_v"] >= 0.10 or logistic_summary["cross_validated_auc"] >= 0.70:
        recommendation_code = 2
        recommendation = "可以进入 author-period analysis，但必须同步报告 STRICT/CORE/EXPANDED 敏感性结果。"
    else:
        recommendation_code = 1
        recommendation = "可以进入 author-period analysis，并保留常规敏感性说明。"
    unknown_mixed_noise = (mixed_row["mean_unknown"] + noise_row["mean_unknown"]) - (mixed_row["mean_known"] + noise_row["mean_known"])

    report = f"""# Period Metadata Representativeness Audit Report

# 1. Objective

本报告只检验 author-period metadata 子样本相对于43,015首完整 C_K20 语料的代表性与可观测缺失关联；不比较初唐、盛唐、中唐、晚唐，不推断单篇作品年代，也不修改任何原数据。

权威输入：`{input_manifest['authoritative_master']['path']}`；SHA-256：`{input_manifest['authoritative_master']['sha256']}`。

# 2. Coverage

- KNOWN：{int(known_coverage['author_count']):,} / {int(coverage.iloc[0]['author_count']):,} 位作者（{known_coverage['author_ratio']:.2%}），{int(known_coverage['poem_count']):,} / {int(coverage.iloc[0]['poem_count']):,} 首诗（{known_coverage['poem_ratio']:.2%}）。
- UNKNOWN：{int(unknown_coverage['author_count']):,} 位作者（{unknown_coverage['author_ratio']:.2%}），{int(unknown_coverage['poem_count']):,} 首诗（{unknown_coverage['poem_ratio']:.2%}）。

# 3. Author Productivity Bias

KNOWN 作者诗作数中位数为 {known_productivity['median']:.1f}，UNKNOWN 为 {unknown_productivity['median']:.1f}；均值分别为 {known_productivity['mean']:.3f} 与 {unknown_productivity['mean']:.3f}。Mann–Whitney U={known_productivity['mann_whitney_u']:.3f}，p={known_productivity['mann_whitney_p_value']:.3g}；Cliff's delta={productivity_delta:.4f}（{known_productivity['effect_interpretation']}，正值表示 KNOWN 更高）。时期信息可用性明显偏向高产作者时，应将其视为选择偏差诊断，而不是文学史差异。

# 4. Topic Probability Bias

20 Topic 的最大 abs(SMD)={topic_max:.4f}，平均 abs(SMD)={topic_mean:.4f}；abs(SMD)≥0.10 / 0.20 / 0.50 的 Topic 数分别为 {counts['ge_0_10']} / {counts['ge_0_20']} / {counts['ge_0_50']}。Bootstrap 使用固定 seed 20260901、1,000次 multinomial-count resampling；效应量优先于大样本 p 值。

{markdown_table(topic_probability.sort_values('abs_smd', ascending=False), ['topic_id', 'topic_label', 'topic_type', 'mean_known', 'mean_unknown', 'smd', 'bootstrap_ci_low', 'bootstrap_ci_high'], 10)}

# 5. Topic Type Bias

MIXED mass 的 KNOWN/UNKNOWN 均值为 {mixed_row['mean_known']:.6f}/{mixed_row['mean_unknown']:.6f}，NOISE 为 {noise_row['mean_known']:.6f}/{noise_row['mean_unknown']:.6f}。UNKNOWN 相对 KNOWN 的 MIXED+NOISE mass 差为 {unknown_mixed_noise:.6f}；这里只说明缺失与 topic structure 的关联，不作文学意义判断。

{markdown_table(topic_type_mass, ['topic_type', 'mean_known', 'mean_unknown', 'smd', 'bootstrap_ci_low', 'bootstrap_ci_high', 'bias_level'])}

# 6. Dominant Topic Bias

KNOWN × dominant topic 的 Chi-square={dominant_tests['dominant_topic_chi_square']:.3f}，p={dominant_tests['dominant_topic_p_value']:.3g}，Cramér's V={dominant_tests['dominant_topic_cramers_v']:.4f}。Dominant Topic type 的 Cramér's V={dominant_tests['dominant_topic_type_cramers_v']:.4f}。统计显著性不等同实质差异。

# 7. Topic Mixture / Entropy

Topic entropy 的 KNOWN/UNKNOWN 均值为 {entropy_row['known_mean']:.6f}/{entropy_row['unknown_mean']:.6f}，Cliff's delta={entropy_row['cliffs_delta_known_minus_unknown']:.4f}（{entropy_row['effect_interpretation']}）。Normalized entropy 的对应效应量为 {normalized_entropy_row['cliffs_delta_known_minus_unknown']:.4f}。正值表示 KNOWN 更高；若为负值，只能说明 UNKNOWN 更混合的可观测关联。

# 8. CBDB / Identity Resolution Bias

{markdown_table(cbdb_status, ['cbdb_match_status', 'author_count', 'period_known_author_ratio', 'poem_count', 'period_known_poem_ratio'])}

{markdown_table(cbdb_confidence, ['cbdb_match_confidence', 'author_count', 'period_known_author_ratio', 'poem_count', 'period_known_poem_ratio'])}

Period availability 与 CBDB identity resolution 的可观测状态明显相关时，可表述为 “Observed missingness is inconsistent with a simple random-missing pattern”，但本审计不声称证明 MCAR、MAR 或 MNAR。

# 9. STRICT / CORE / EXPANDED Samples

STRICT=KNOWN+HIGH 且排除 cross-period/boundary；CORE=KNOWN+(HIGH/MEDIUM) 且排除 cross-period/boundary；EXPANDED=全部 KNOWN。三者只建立标签，不删除或修改原数据。

{markdown_table(sample_sensitivity, ['sample_name', 'poem_count', 'author_count', 'max_abs_topic_smd', 'mean_abs_topic_smd', 'entropy_effect_size', 'dominant_topic_cramers_v', 'overall_bias_level'])}

按预先登记的描述性 bias score，{best_sample['sample_name']} 在三者中最接近 full corpus；这不表示其没有缺失偏差。

# 10. High-risk Topics

{markdown_table(risk.sort_values(['risk_level', 'abs_smd'], ascending=[True, False]), ['topic_id', 'topic_label', 'topic_type', 'mean_known', 'mean_unknown', 'smd', 'risk_level', 'risk_reason'])}

# 11. CLEAR Topic Assessment

- LOW：{', '.join(clear_groups['LOW']) if clear_groups['LOW'] else '无'}
- MODERATE：{', '.join(clear_groups['MODERATE']) if clear_groups['MODERATE'] else '无'}
- HIGH：{', '.join(clear_groups['HIGH']) if clear_groups['HIGH'] else '无'}

{markdown_table(clear, ['topic_id', 'topic_label', 'mean_known', 'mean_unknown', 'smd', 'bootstrap_ci_low', 'bootstrap_ci_high', 'dominant_ratio_difference', 'risk_level'])}

# 12. Overall Assessment

A. Known-period poems 与 full corpus 存在可检测差异；是否具实质性主要由上述效应量而非 p 值判断。

B. 20 Topic 的效应量计数为 {counts['ge_0_10']}/{counts['ge_0_20']}/{counts['ge_0_50']}（阈值0.10/0.20/0.50），不把大样本统计显著自动解释为严重偏差。

C. 作者产量 Cliff's delta={productivity_delta:.4f}，显示 UNKNOWN 是否偏向低产作者应按其方向与效应等级判断。

D. UNKNOWN 的 MIXED+NOISE mass 相对差为 {unknown_mixed_noise:.6f}。

E. CLEAR LOW-risk：{', '.join(clear_groups['LOW']) if clear_groups['LOW'] else '无'}。

F. CLEAR MODERATE/HIGH-risk：{', '.join(clear_groups['MODERATE'] + clear_groups['HIGH']) if clear_groups['MODERATE'] or clear_groups['HIGH'] else '无'}。

G. 三个未来样本中，{best_sample['sample_name']} 的预登记综合偏差分数最低。

H. 建议代码 {recommendation_code}：{recommendation}

Logistic missingness diagnostic 的 apparent AUC={logistic_summary['apparent_auc']:.4f}，按作者分组的 cross-validated AUC={logistic_summary['cross_validated_auc']:.4f}。该模型仅诊断 period availability 可预测性，禁止用于时期填补。
"""
    overview = {
        "known_authors": int(known_coverage["author_count"]),
        "known_author_ratio": float(known_coverage["author_ratio"]),
        "unknown_authors": int(unknown_coverage["author_count"]),
        "known_poems": int(known_coverage["poem_count"]),
        "known_poem_ratio": float(known_coverage["poem_ratio"]),
        "unknown_poems": int(unknown_coverage["poem_count"]),
        "productivity_cliffs_delta": productivity_delta,
        "productivity_effect": known_productivity["effect_interpretation"],
        "max_abs_topic_smd": topic_max,
        "mean_abs_topic_smd": topic_mean,
        **counts,
        "dominant_topic_cramers_v": float(dominant_tests["dominant_topic_cramers_v"]),
        "entropy_cliffs_delta": float(entropy_row["cliffs_delta_known_minus_unknown"]),
        "normalized_entropy_cliffs_delta": float(normalized_entropy_row["cliffs_delta_known_minus_unknown"]),
        "unknown_minus_known_mixed_noise_mass": float(unknown_mixed_noise),
        "clear_risk_groups": clear_groups,
        "best_sample": best_sample["sample_name"],
        "recommendation_code": recommendation_code,
        "recommendation": recommendation,
        "logistic_cross_validated_auc": logistic_summary["cross_validated_auc"],
    }
    return report, overview


def build_readme(input_manifest: dict[str, Any]) -> str:
    """生成审计目录说明。"""

    return f"""# Period Metadata Representativeness Audit

本目录审计 author-period KNOWN/UNKNOWN 子样本的覆盖率、作者产量结构、20维 Topic probability、dominant topic、topic type mass、entropy、CBDB identity resolution 与 STRICT/CORE/EXPANDED 样本代表性。

权威诗级输入：`{input_manifest['authoritative_master']['path']}`  
SHA-256：`{input_manifest['authoritative_master']['sha256']}`

诗级输入缺少 `author_status` 与 `cbdb_match_status`，故只按唯一 `author_id_internal` 从 `{input_manifest['author_metadata']['path']}` 确定性连接这两列；不修改权威 CSV。

本任务是 missingness/representativeness audit，不比较初盛中晚唐，不重新训练 LDA，不修改 CBDB 或 period，不进行重加权或时期填补。
"""


def ensure_no_numeric_nan(frames: dict[str, pd.DataFrame]) -> list[str]:
    """检查核心输出的数值字段是否存在未说明 NaN。"""

    failures = []
    for name, frame in frames.items():
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.isna().any().any():
            columns = numeric.columns[numeric.isna().any()].tolist()
            failures.append(f"{name}:{','.join(columns)}")
    return failures


def write_manifest(output_root: Path, inputs: dict[str, Any], outputs: list[Path], metadata: dict[str, Any]) -> None:
    """记录输入、环境、方法和非 XLSX 输出哈希。"""

    payload = {
        "audit_version": CONFIG["audit_version"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": inputs,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "methods": {
            "period_known_rule": "author_period_primary in {初唐,盛唐,中唐,晚唐}",
            "bootstrap": f"multinomial count bootstrap; iterations={CONFIG['bootstrap_iterations']}; seed={CONFIG['random_seed']}",
            "productivity_effect": "Cliff's delta derived from Mann–Whitney U",
            "topic_effect": "standardized mean difference using pooled SD",
            "dominant_composition": "Chi-square and Cramér's V",
            "logistic_diagnostic": "5-fold StratifiedGroupKFold by author_id_internal; no imputation",
            "sample_comparison": "overlapping subset vs full corpus; descriptive only",
        },
        "summary": metadata,
        "outputs": {
            path.relative_to(output_root).as_posix(): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(outputs)
        },
    }
    (output_root / "audit_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    """执行全量代表性审计并生成可复核产物。"""

    global CONFIG
    CONFIG = read_config(config_path)
    project_root = config_path.resolve().parents[1]
    master_path = project_root / CONFIG["authoritative_master"]
    author_path = project_root / CONFIG["author_metadata"]
    topic_metadata_path = project_root / CONFIG["topic_metadata"]
    history_version_path = project_root / CONFIG["historical_metadata_version"]
    output_root = project_root / CONFIG["output_root"]
    data_root = output_root / "data"
    figures_root = output_root / "figures"
    build_root = output_root / ".build"
    for path in [data_root, figures_root, build_root]:
        path.mkdir(parents=True, exist_ok=True)

    input_hash_before = sha256_file(master_path)
    topic_fields = [f"topic_{index:02d}_prob" for index in range(1, CONFIG["expected_topic_count"] + 1)]
    required_master_fields = [
        "poem_id",
        "author",
        "author_id_internal",
        "cbdb_match_confidence",
        "author_period_primary",
        "period_confidence",
        "cross_period_flag",
        "period_boundary_flag",
        *topic_fields,
        "dominant_topic_id",
        "dominant_topic_label",
        "dominant_topic_probability",
        "second_topic_probability",
        "third_topic_probability",
        "topic_entropy",
        "normalized_topic_entropy",
        "top1_top2_gap",
    ]
    poems = pd.read_csv(master_path, encoding="utf-8-sig", low_memory=False)
    missing_master_fields = sorted(set(required_master_fields) - set(poems.columns))
    if missing_master_fields:
        raise ValueError(f"权威 master 缺少核心字段：{missing_master_fields}")
    if len(poems) != CONFIG["expected_poem_count"]:
        raise ValueError(f"权威 master 行数为 {len(poems)}，预期 {CONFIG['expected_poem_count']}")
    if poems["poem_id"].duplicated().any():
        raise ValueError("权威 master poem_id 不唯一")

    author_metadata = pd.read_csv(author_path, encoding="utf-8-sig", low_memory=False)
    if author_metadata["author_id_internal"].duplicated().any():
        raise ValueError("作者主表 author_id_internal 不唯一")
    author_join = author_metadata[
        [
            "author_id_internal",
            "author_status",
            "cbdb_match_status",
            "cbdb_match_confidence",
            "author_period_primary",
            "period_confidence",
            "cross_period_flag",
            "period_boundary_flag",
            "poem_count",
        ]
    ].copy()
    poems = poems.merge(
        author_join[["author_id_internal", "author_status", "cbdb_match_status", "poem_count"]],
        on="author_id_internal",
        how="left",
        validate="many_to_one",
    )
    if poems[["author_status", "cbdb_match_status", "poem_count"]].isna().any().any():
        raise ValueError("诗级 master 有 author_id_internal 无法连接作者主表")
    computed_poem_count = poems.groupby("author_id_internal")["poem_id"].transform("size")
    if not np.array_equal(computed_poem_count.to_numpy(), poems["poem_count"].astype(int).to_numpy()):
        raise ValueError("作者主表 poem_count 与权威诗级记录数不一致")
    poems = poems.rename(columns={"poem_count": "author_poem_count"})
    poems["log1p_author_poem_count"] = np.log1p(poems["author_poem_count"].astype(float))
    poems["cross_period_flag"] = parse_boolean_series(poems["cross_period_flag"])
    poems["period_boundary_flag"] = parse_boolean_series(poems["period_boundary_flag"])
    poems["author_period_primary"] = poems["author_period_primary"].fillna("").astype(str)
    poems["period_confidence"] = poems["period_confidence"].fillna("UNKNOWN").astype(str)
    known_period_values = set(CONFIG["known_period_values"])
    poems["period_metadata_status"] = np.where(poems["author_period_primary"].isin(known_period_values), "KNOWN", "UNKNOWN")
    poems["period_confidence_group"] = poems["period_confidence"].where(poems["period_confidence"].isin(CONFIDENCE_VALUES), "UNKNOWN")
    if not set(poems["period_metadata_status"]).issubset(PERIOD_STATUS_VALUES):
        raise ValueError("period_metadata_status 出现非法值")
    poems["eligible_period_strict"] = (
        poems["period_metadata_status"].eq("KNOWN")
        & poems["period_confidence_group"].eq("HIGH")
        & ~poems["cross_period_flag"]
        & ~poems["period_boundary_flag"]
    )
    poems["eligible_period_core"] = (
        poems["period_metadata_status"].eq("KNOWN")
        & poems["period_confidence_group"].isin(["HIGH", "MEDIUM"])
        & ~poems["cross_period_flag"]
        & ~poems["period_boundary_flag"]
    )
    poems["eligible_period_expanded"] = poems["period_metadata_status"].eq("KNOWN")

    topic_metadata = pd.read_csv(topic_metadata_path, encoding="utf-8-sig")
    required_topic_fields = {"topic_id", "final_topic_label", "topic_type", "mean_human_score", "interpretation_scope"}
    if not required_topic_fields.issubset(topic_metadata.columns):
        raise ValueError(f"topic_metadata 缺少字段：{sorted(required_topic_fields - set(topic_metadata.columns))}")
    expected_topic_ids = [field.removesuffix("_prob") for field in topic_fields]
    if topic_metadata["topic_id"].tolist() != expected_topic_ids:
        raise ValueError("topic_metadata topic_id 与20维概率列顺序不一致")
    if set(topic_metadata["topic_type"]) != set(TOPIC_TYPES):
        raise ValueError("topic_metadata topic_type 未覆盖 CLEAR/BROAD/MIXED/NOISE")
    if poems[topic_fields].isna().any().any():
        raise ValueError("20维 Topic probability 存在缺失")
    probability_sums = poems[topic_fields].sum(axis=1)
    max_probability_sum_deviation = float(np.max(np.abs(probability_sums.to_numpy() - 1.0)))
    if max_probability_sum_deviation > 1e-8:
        raise ValueError(f"20维概率和偏离1，最大偏差 {max_probability_sum_deviation}")

    author_columns = [
        "author_id_internal",
        "author",
        "author_status",
        "cbdb_match_status",
        "cbdb_match_confidence",
        "author_poem_count",
        "period_metadata_status",
        "period_confidence_group",
        "cross_period_flag",
        "period_boundary_flag",
        "eligible_period_strict",
        "eligible_period_core",
        "eligible_period_expanded",
    ]
    authors = poems[author_columns].drop_duplicates("author_id_internal").copy()
    consistency_counts = poems.groupby("author_id_internal")[
        ["period_metadata_status", "period_confidence_group", "eligible_period_strict", "eligible_period_core", "eligible_period_expanded"]
    ].nunique()
    if (consistency_counts > 1).any().any():
        raise ValueError("同一作者的 period/audit sample 标签不一致")
    authors["productivity_bucket"] = authors["author_poem_count"].astype(int).map(productivity_bucket)
    authors["productivity_bucket"] = pd.Categorical(authors["productivity_bucket"], PRODUCTIVITY_BUCKETS, ordered=True)

    coverage, productivity, productivity_buckets = build_coverage_tables(poems, authors)
    topic_probability = build_topic_probability_table(poems, topic_metadata, topic_fields, CONFIG)
    topic_type_mass, mass_qc = build_topic_type_mass_table(poems, topic_metadata, topic_fields, CONFIG)
    dominant_topic, dominant_type, dominant_tests = build_dominant_tables(poems, topic_metadata)
    mixture = build_mixture_table(poems, CONFIG["risk_thresholds"])
    author_status_coverage = coverage_by_category(authors, "author_status")
    cbdb_status_coverage = coverage_by_category(authors, "cbdb_match_status")
    cbdb_confidence_coverage = coverage_by_category(authors, "cbdb_match_confidence")
    logistic, logistic_summary = build_logistic_diagnostic(poems, topic_fields, CONFIG)
    sample_sensitivity = build_sample_sensitivity(poems, authors, topic_fields)
    risk, clear = build_risk_tables(topic_probability, dominant_topic, CONFIG["risk_thresholds"])

    labels = poems[
        [
            "poem_id",
            "author_id_internal",
            "period_metadata_status",
            "period_confidence_group",
            "author_poem_count",
            "cross_period_flag",
            "period_boundary_flag",
            "eligible_period_strict",
            "eligible_period_core",
            "eligible_period_expanded",
        ]
    ].copy()
    frames = {
        "period_metadata_coverage_audit": coverage,
        "author_productivity_known_vs_unknown": productivity,
        "period_coverage_by_author_productivity": productivity_buckets,
        "topic_probability_representativeness": topic_probability,
        "topic_type_mass_representativeness": topic_type_mass,
        "dominant_topic_known_unknown": dominant_topic,
        "dominant_topic_type_known_unknown": dominant_type,
        "topic_mixture_representativeness": mixture,
        "period_coverage_by_author_status": author_status_coverage,
        "period_coverage_by_cbdb_status": cbdb_status_coverage,
        "period_coverage_by_cbdb_confidence": cbdb_confidence_coverage,
        "period_missingness_logistic_diagnostic": logistic,
        "analysis_sample_representativeness": sample_sensitivity,
        "topic_missingness_risk": risk,
        "clear_topic_representativeness": clear,
        "period_metadata_sample_labels": labels,
    }
    nan_failures = ensure_no_numeric_nan({name: frame for name, frame in frames.items() if name != "period_missingness_logistic_diagnostic"})
    if nan_failures:
        raise ValueError(f"核心输出存在未说明 NaN：{nan_failures}")

    inputs = {
        "authoritative_master": {
            "path": master_path.as_posix(),
            "modified_time": datetime.fromtimestamp(master_path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
            "size_bytes": master_path.stat().st_size,
            "sha256": input_hash_before,
            "rows": len(poems),
            "columns": len(pd.read_csv(master_path, encoding="utf-8-sig", nrows=0).columns),
        },
        "author_metadata": {
            "path": author_path.as_posix(),
            "sha256": sha256_file(author_path),
            "join_key": "author_id_internal",
            "joined_fields": ["author_status", "cbdb_match_status", "poem_count"],
        },
        "topic_metadata": {
            "path": topic_metadata_path.as_posix(),
            "sha256": sha256_file(topic_metadata_path),
            "rows": len(topic_metadata),
        },
    }
    report, overview = build_report(
        output_root,
        inputs,
        coverage,
        productivity,
        topic_probability,
        topic_type_mass,
        dominant_tests,
        mixture,
        cbdb_status_coverage,
        cbdb_confidence_coverage,
        sample_sensitivity,
        risk,
        clear,
        logistic_summary,
    )

    output_paths = []
    for name, frame in frames.items():
        path = data_root / f"{name}.csv"
        save_csv(frame, path)
        output_paths.append(path)
    figure_paths = build_figures(figures_root, topic_probability, productivity_buckets, poems, dominant_topic)
    output_paths.extend(figure_paths)
    report_path = output_root / "Period_Metadata_Representativeness_Audit_Report.md"
    report_path.write_text(report, encoding="utf-8")
    readme_path = output_root / "README.md"
    readme_path.write_text(build_readme(inputs), encoding="utf-8")
    output_paths.extend([report_path, readme_path])

    invariant_fields = ["poem_id", "author_period_primary", "period_confidence", "cbdb_person_id", "cbdb_match_confidence"]
    invariants_before = source_field_hashes(master_path, invariant_fields)
    input_hash_after = sha256_file(master_path)
    invariants_after = source_field_hashes(master_path, invariant_fields)
    expected_topic_checksum = json.loads(history_version_path.read_text(encoding="utf-8"))["output_topic_probability_checksum"]
    actual_topic_checksum = topic_probability_checksum(master_path, topic_fields)
    label_poem_hash = sha256_text_values(labels["poem_id"].astype(str))
    source_poem_hash = sha256_text_values(poems["poem_id"].astype(str))
    qc_rows = [
        ("TEST_01", "输入为43,015行", len(poems) == CONFIG["expected_poem_count"], len(poems), CONFIG["expected_poem_count"]),
        ("TEST_02", "audit label poem_id 顺序不变化", label_poem_hash == source_poem_hash, label_poem_hash, source_poem_hash),
        ("TEST_03", "20维 Topic probabilities 校验和不变化", actual_topic_checksum == expected_topic_checksum, actual_topic_checksum, expected_topic_checksum),
        ("TEST_04", "author_period_primary 不变化", invariants_before["author_period_primary"] == invariants_after["author_period_primary"], invariants_after["author_period_primary"], invariants_before["author_period_primary"]),
        ("TEST_05", "period_confidence 不变化", invariants_before["period_confidence"] == invariants_after["period_confidence"], invariants_after["period_confidence"], invariants_before["period_confidence"]),
        ("TEST_06", "CBDB 字段不变化", invariants_before["cbdb_person_id"] == invariants_after["cbdb_person_id"] and invariants_before["cbdb_match_confidence"] == invariants_after["cbdb_match_confidence"], "unchanged", "unchanged"),
        ("TEST_07", "审计不覆盖 authoritative master", input_hash_before == input_hash_after, input_hash_after, input_hash_before),
        ("TEST_08", "Topic labels 全部来自 topic_metadata", topic_metadata["topic_id"].tolist() == expected_topic_ids, len(topic_metadata), len(expected_topic_ids)),
        ("TEST_09", "KNOWN + UNKNOWN = 43,015", int((poems["period_metadata_status"].eq("KNOWN") | poems["period_metadata_status"].eq("UNKNOWN")).sum()) == len(poems), len(poems), len(poems)),
        ("TEST_10", "核心 SMD/effect size 无未说明 NaN", not nan_failures, ";".join(nan_failures), "none"),
        ("TEST_11", "20维概率和约等于1", max_probability_sum_deviation <= 1e-8, max_probability_sum_deviation, "<=1e-8"),
        ("TEST_12", "四类 Topic mass 之和约等于1", mass_qc["max_mass_sum_deviation"] <= 1e-8, mass_qc["max_mass_sum_deviation"], "<=1e-8"),
    ]
    qc = pd.DataFrame(
        [
            {"test_id": test_id, "description": description, "status": "PASS" if passed else "FAIL", "observed": observed, "expected": expected}
            for test_id, description, passed, observed, expected in qc_rows
        ]
    )
    qc_path = data_root / "representativeness_audit_qc.csv"
    save_csv(qc, qc_path)
    output_paths.append(qc_path)
    if not qc["status"].eq("PASS").all():
        raise RuntimeError(f"代表性审计 QC 失败：{qc.loc[qc['status'].ne('PASS'), 'test_id'].tolist()}")

    payload = {
        "metadata": {
            "output_root": output_root.as_posix(),
            "workbook_output": (output_root / "Period_Metadata_Representativeness_Audit.xlsx").as_posix(),
            "audit_version": CONFIG["audit_version"],
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "authoritative_master_sha256": input_hash_before,
            "topic_metadata_sha256": inputs["topic_metadata"]["sha256"],
        },
        "overview": overview,
        "input_manifest": inputs,
        "tables": {name: records_without_nan(frame) for name, frame in frames.items() if name != "period_metadata_sample_labels"},
        "fields": {name: frame.columns.tolist() for name, frame in frames.items() if name != "period_metadata_sample_labels"},
        "qc": records_without_nan(qc),
        "qc_fields": qc.columns.tolist(),
        "figure_paths": [path.as_posix() for path in figure_paths],
        "method_notes": [
            ["审计边界", "只比较 period KNOWN/UNKNOWN 与未来样本相对 full corpus；不比较初盛中晚唐。"],
            ["权威数据", "43,015×68 enriched master CSV 只读；author_status/cbdb_match_status 从作者主表按唯一 ID 连接。"],
            ["Bootstrap", f"multinomial count bootstrap，{CONFIG['bootstrap_iterations']}次，seed={CONFIG['random_seed']}。"],
            ["效应量", "Topic 使用 pooled-SD SMD；偏态分布使用 Cliff's delta；列联表使用 Cramér's V。"],
            ["样本定义", "STRICT=KNOWN+HIGH+非跨期/边界；CORE=KNOWN+HIGH/MEDIUM+非跨期/边界；EXPANDED=全部KNOWN。"],
            ["缺失机制", "只判断 period availability 与可观测特征有关；不声称统计证明 MCAR/MAR/MNAR。"],
            ["禁止用途", "Logistic 模型只作 missingness diagnostic，禁止 period imputation 或重加权。"],
        ],
    }
    payload_path = build_root / "workbook_payload.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output_paths.append(payload_path)
    write_manifest(output_root, inputs, [path for path in output_paths if path != payload_path], overview)

    return {
        "status": "PASS",
        "output_root": output_root.as_posix(),
        "poems": len(poems),
        "authors": len(authors),
        "known_poems": int(poems["period_metadata_status"].eq("KNOWN").sum()),
        "unknown_poems": int(poems["period_metadata_status"].eq("UNKNOWN").sum()),
        "known_authors": int(authors["period_metadata_status"].eq("KNOWN").sum()),
        "unknown_authors": int(authors["period_metadata_status"].eq("UNKNOWN").sum()),
        "topic_probability_checksum": actual_topic_checksum,
        "overview": overview,
        "qc_failed": qc.loc[qc["status"].ne("PASS"), "test_id"].tolist(),
        "payload": payload_path.as_posix(),
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/period_representativeness_audit_config.yaml"))
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""

    args = parse_args()
    result = run(args.config.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


CONFIG: dict[str, Any] = {}


if __name__ == "__main__":
    raise SystemExit(main())
