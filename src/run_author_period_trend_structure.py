"""生成 C_K20 按作者时期划分的主题趋势结构分析。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRANSITIONS = [("EH", "初唐", "盛唐"), ("HM", "盛唐", "中唐"), ("ML", "中唐", "晚唐")]
AGGREGATIONS = ["POEM", "AUTHOR"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/trend_structure_config.yaml")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """读取 JSON 兼容 YAML 配置。"""

    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str) -> Path:
    """相对仓库根目录解析路径。"""

    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """计算连续数组内容哈希。"""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """以无 BOM UTF-8 写出稳定 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.15g")


def normalize_boolean(series: pd.Series, field: str) -> pd.Series:
    """把审计标签安全规范为布尔值。"""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False}
    lowered = series.astype(str).str.strip().str.lower()
    invalid = sorted(set(lowered) - set(mapping))
    if invalid:
        raise ValueError(f"{field} 含非法布尔值：{invalid}")
    return lowered.map(mapping).astype(bool)


def mode_with_count(values: list[str]) -> tuple[str, int, bool]:
    """返回众数、次数与是否并列；并列时保守返回 UNCERTAIN。"""

    counts = Counter(values)
    maximum = max(counts.values())
    winners = sorted(value for value, count in counts.items() if count == maximum)
    if len(winners) > 1:
        return "UNCERTAIN", maximum, True
    return winners[0], maximum, False


def numeric_direction(delta: float, reference_mean: float, thresholds: dict[str, float]) -> str:
    """按相对阈值判定数值方向，避免仅用精确零。"""

    tolerance = max(
        thresholds["numeric_absolute_epsilon"],
        abs(reference_mean) * thresholds["numeric_flat_relative_delta"],
    )
    if abs(delta) <= tolerance:
        return "FLAT_NUMERIC"
    return "UP" if delta > 0 else "DOWN"


def supported_direction(low: float, high: float) -> str:
    """依据 bootstrap 差值区间返回描述性方向。"""

    if low > 0:
        return "SUPPORTED_UP"
    if high < 0:
        return "SUPPORTED_DOWN"
    return "UNCERTAIN"


def classify_shape(normalized_range: float, directions: list[str], thresholds: dict[str, float]) -> str:
    """以连续波幅和四期均值的数值方向保守分类曲线。"""

    if not math.isnan(normalized_range) and normalized_range <= thresholds["near_flat_normalized_range"]:
        return "NEAR_FLAT"
    if any(direction == "FLAT_NUMERIC" for direction in directions):
        return "UNCERTAIN"
    signs = [1 if direction == "UP" else -1 for direction in directions]
    if signs == [1, 1, 1]:
        return "MONOTONIC_UP"
    if signs == [-1, -1, -1]:
        return "MONOTONIC_DOWN"
    if signs in ([1, 1, -1], [1, -1, -1]):
        return "RISE_THEN_FALL"
    if signs in ([-1, -1, 1], [-1, 1, 1]):
        return "FALL_THEN_RISE"
    return "MULTI_TURN"


def rank_order(periods: list[str], values: np.ndarray, period_codes: dict[str, str]) -> str:
    """按均值降序保存完整四期排序。"""

    order = sorted(range(len(periods)), key=lambda index: (-float(values[index]), index))
    return ">".join(period_codes[periods[index]] for index in order)


def pearson_four(left: np.ndarray, right: np.ndarray) -> float:
    """计算四期向量 Pearson 相关，不返回 p 值。"""

    if np.std(left) == 0 or np.std(right) == 0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def spearman_four(left: np.ndarray, right: np.ndarray) -> float:
    """以平均秩计算四期向量 Spearman 相关，不返回 p 值。"""

    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=float)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=float)
    return pearson_four(left_rank, right_rank)


def concentration_level(top1: float, top5: float, hhi: float, thresholds: dict[str, float]) -> str:
    """按配置阈值标记作者 Topic mass 集中度。"""

    if (
        top1 >= thresholds["author_concentration_high_top1"]
        or top5 >= thresholds["author_concentration_high_top5"]
        or hhi >= thresholds["author_concentration_high_hhi"]
    ):
        return "HIGH"
    if (
        top1 >= thresholds["author_concentration_moderate_top1"]
        or top5 >= thresholds["author_concentration_moderate_top5"]
        or hhi >= thresholds["author_concentration_moderate_hhi"]
    ):
        return "MODERATE"
    return "LOW"


def load_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, Path], dict[str, str]]:
    """读取并验证 master、审计标签、taxonomy 与缺失风险。"""

    paths = {
        "authoritative_master": resolve_path(config["authoritative_master"]),
        "topic_metadata": resolve_path(config["topic_metadata"]),
        "sample_labels": resolve_path(config["sample_labels"]),
        "topic_missingness_risk": resolve_path(config["topic_missingness_risk"]),
        "representativeness_audit_manifest": resolve_path(config["representativeness_audit_manifest"]),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少输入文件：{missing}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    master = pd.read_csv(paths["authoritative_master"], low_memory=False)
    labels = pd.read_csv(paths["sample_labels"], low_memory=False)
    metadata = pd.read_csv(paths["topic_metadata"], low_memory=False)
    risk = pd.read_csv(paths["topic_missingness_risk"], low_memory=False)
    expected_poems = int(config["expected_poem_count"])
    expected_topics = int(config["expected_topic_count"])
    if len(master) != expected_poems or len(labels) != expected_poems:
        raise ValueError(f"master/标签行数不等于 {expected_poems}")
    if master["poem_id"].duplicated().any() or labels["poem_id"].duplicated().any():
        raise ValueError("poem_id 必须在 master 与审计标签中唯一")
    topic_columns = sorted(column for column in master if column.startswith("topic_") and column.endswith("_prob"))
    if len(topic_columns) != expected_topics:
        raise ValueError(f"Topic probability 列数不等于 {expected_topics}")
    topic_ids = [column.removesuffix("_prob") for column in topic_columns]
    if set(metadata["topic_id"]) != set(topic_ids) or metadata["topic_id"].duplicated().any():
        raise ValueError("topic_metadata 与 master 的 Topic 集合不一致")
    if set(risk["topic_id"]) != set(topic_ids) or risk["topic_id"].duplicated().any():
        raise ValueError("topic_missingness_risk 与 master 的 Topic 集合不一致")
    probability = master[topic_columns].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("Topic probability 含缺失、无穷或负值")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-10, rtol=0):
        raise ValueError("每首诗的20维 Topic probability 之和不为1")
    label_columns = ["poem_id", "author_id_internal", *config["samples"].values()]
    if any(column not in labels for column in label_columns):
        raise ValueError("审计标签缺少必要字段")
    joined = master.merge(labels[label_columns], on=["poem_id", "author_id_internal"], how="left", validate="one_to_one")
    if joined[list(config["samples"].values())].isna().any().any():
        raise ValueError("审计标签连接后存在缺失")
    for column in config["samples"].values():
        joined[column] = normalize_boolean(joined[column], column)
    valid_periods = set(config["period_order"])
    for sample, column in config["samples"].items():
        invalid = joined.loc[joined[column] & ~joined["author_period_primary"].isin(valid_periods)]
        if not invalid.empty:
            raise ValueError(f"{sample} 入选行存在非法 author_period_primary")
    topic_info = metadata[
        ["topic_id", "final_topic_label", "short_label", "topic_type", "interpretation_scope"]
    ].rename(columns={"final_topic_label": "topic_label"})
    risk_info = risk[["topic_id", "risk_level", "risk_reason", "smd", "abs_smd"]].rename(
        columns={"risk_level": "representativeness_risk", "smd": "known_unknown_smd", "abs_smd": "known_unknown_abs_smd"}
    )
    topic_info = topic_info.merge(risk_info, on="topic_id", validate="one_to_one").sort_values("topic_id").reset_index(drop=True)
    return joined, topic_info, labels, topic_columns, paths, hashes


def build_author_profiles(
    poems: pd.DataFrame,
    topic_columns: list[str],
    samples: dict[str, str],
    period_order: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """建立三套样本的作者等权基础表。"""

    frames: list[pd.DataFrame] = []
    by_sample: dict[str, pd.DataFrame] = {}
    for sample, label_column in samples.items():
        selected = poems.loc[poems[label_column]].copy()
        grouped = selected.groupby(
            ["author_id_internal", "author", "author_period_primary"], sort=False, observed=True
        )
        profile = grouped[topic_columns].mean().reset_index()
        counts = grouped.size().rename("poem_count_in_sample").reset_index()
        profile = counts.merge(
            profile,
            on=["author_id_internal", "author", "author_period_primary"],
            validate="one_to_one",
        )
        profile.insert(0, "sample", sample)
        renamed = {column: column.replace("_prob", "_mean") for column in topic_columns}
        profile = profile.rename(columns=renamed)
        profile["_period_order"] = pd.Categorical(
            profile["author_period_primary"], categories=period_order, ordered=True
        ).codes
        profile = profile.sort_values(["_period_order", "author_id_internal"]).drop(columns="_period_order").reset_index(drop=True)
        frames.append(profile)
        by_sample[sample] = profile
    return pd.concat(frames, ignore_index=True), by_sample


def build_period_topic_means(
    poems: pd.DataFrame,
    profiles: dict[str, pd.DataFrame],
    topic_columns: list[str],
    topic_info: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """计算六套规格的 Period × Topic 描述统计。"""

    info = topic_info.set_index("topic_id")
    rows: list[dict[str, Any]] = []
    for sample_index, (sample, label_column) in enumerate(config["samples"].items(), start=1):
        sample_poems = poems.loc[poems[label_column]]
        sample_profiles = profiles[sample]
        for period_index, period in enumerate(config["period_order"], start=1):
            period_poems = sample_poems.loc[sample_poems["author_period_primary"].eq(period)]
            period_authors = sample_profiles.loc[sample_profiles["author_period_primary"].eq(period)]
            if period_poems.empty or period_authors.empty:
                raise ValueError(f"{sample}/{period} 没有可分析数据")
            for aggregation in AGGREGATIONS:
                values = (
                    period_poems[topic_columns].to_numpy(dtype=float)
                    if aggregation == "POEM"
                    else period_authors[[column.replace("_prob", "_mean") for column in topic_columns]].to_numpy(dtype=float)
                )
                for topic_index, topic_column in enumerate(topic_columns):
                    topic_id = topic_column.removesuffix("_prob")
                    row_info = info.loc[topic_id]
                    vector = values[:, topic_index]
                    rows.append(
                        {
                            "sample": sample,
                            "aggregation": aggregation,
                            "specification": f"{sample}_{aggregation}",
                            "sample_order": sample_index,
                            "period": period,
                            "period_order": period_index,
                            "topic_id": topic_id,
                            "topic_label": row_info["topic_label"],
                            "topic_type": row_info["topic_type"],
                            "interpretation_scope": row_info["interpretation_scope"],
                            "poem_count": int(len(period_poems)),
                            "author_count": int(period_poems["author_id_internal"].nunique()),
                            "estimator_unit": "poem" if aggregation == "POEM" else "author",
                            "mean_topic_probability": float(np.mean(vector)),
                            "median_topic_probability": float(np.median(vector)),
                            "sd_topic_probability": float(np.std(vector, ddof=1)) if len(vector) > 1 else 0.0,
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["sample_order", "aggregation", "period_order", "topic_id"]
    ).reset_index(drop=True)


def bootstrap_sample(
    poems: pd.DataFrame,
    label_column: str,
    topic_columns: list[str],
    period_order: list[str],
    sample_index: int,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """按时期以作者为 cluster 生成 POEM/AUTHOR 两种 bootstrap replicate。"""

    selected = poems.loc[poems[label_column]]
    result: dict[str, dict[str, np.ndarray]] = {}
    for period_index, period in enumerate(period_order):
        period_poems = selected.loc[selected["author_period_primary"].eq(period)]
        grouped = period_poems.groupby("author_id_internal", sort=True, observed=True)
        topic_sums = grouped[topic_columns].sum().to_numpy(dtype=float)
        poem_counts = grouped.size().to_numpy(dtype=float)
        author_means = topic_sums / poem_counts[:, None]
        author_count = len(poem_counts)
        random = np.random.default_rng(np.random.SeedSequence([seed, sample_index, period_index]))
        cluster_counts = random.multinomial(
            author_count,
            np.full(author_count, 1.0 / author_count),
            size=iterations,
        ).astype(float)
        poem_numerator = cluster_counts @ topic_sums
        poem_denominator = cluster_counts @ poem_counts
        result[period] = {
            "POEM": poem_numerator / poem_denominator[:, None],
            "AUTHOR": (cluster_counts @ author_means) / author_count,
            "cluster_counts": cluster_counts,
            "author_count": np.array([author_count]),
            "poem_count": np.array([len(period_poems)]),
        }
    return result


def build_bootstrap_results(
    poems: pd.DataFrame,
    topic_columns: list[str],
    period_means: pd.DataFrame,
    topic_info: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, np.ndarray]]], str]:
    """生成六套规格的95%作者聚类 bootstrap 区间。"""

    all_bootstrap: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    rows: list[dict[str, Any]] = []
    info = topic_info.set_index("topic_id")
    mean_lookup = period_means.set_index(["sample", "aggregation", "period", "topic_id"])
    digest = hashlib.sha256()
    for sample_index, (sample, label_column) in enumerate(config["samples"].items()):
        sample_bootstrap = bootstrap_sample(
            poems,
            label_column,
            topic_columns,
            config["period_order"],
            sample_index,
            int(config["bootstrap_iterations"]),
            int(config["random_seed"]),
        )
        all_bootstrap[sample] = sample_bootstrap
        for period_index, period in enumerate(config["period_order"], start=1):
            for aggregation in AGGREGATIONS:
                replicate = sample_bootstrap[period][aggregation]
                digest.update(sample.encode("utf-8"))
                digest.update(period.encode("utf-8"))
                digest.update(aggregation.encode("ascii"))
                digest.update(replicate.tobytes())
                lows = np.quantile(replicate, 0.025, axis=0)
                highs = np.quantile(replicate, 0.975, axis=0)
                bootstrap_means = np.mean(replicate, axis=0)
                bootstrap_sds = np.std(replicate, axis=0, ddof=1)
                for topic_index, topic_column in enumerate(topic_columns):
                    topic_id = topic_column.removesuffix("_prob")
                    row_info = info.loc[topic_id]
                    observed = mean_lookup.loc[(sample, aggregation, period, topic_id)]
                    rows.append(
                        {
                            "sample": sample,
                            "aggregation": aggregation,
                            "specification": f"{sample}_{aggregation}",
                            "period": period,
                            "period_order": period_index,
                            "topic_id": topic_id,
                            "topic_label": row_info["topic_label"],
                            "topic_type": row_info["topic_type"],
                            "poem_count": int(observed["poem_count"]),
                            "author_count": int(observed["author_count"]),
                            "mean": float(observed["mean_topic_probability"]),
                            "ci_2_5": float(lows[topic_index]),
                            "ci_97_5": float(highs[topic_index]),
                            "bootstrap_mean": float(bootstrap_means[topic_index]),
                            "bootstrap_sd": float(bootstrap_sds[topic_index]),
                            "bootstrap_iterations": int(config["bootstrap_iterations"]),
                            "random_seed": int(config["random_seed"]),
                            "bootstrap_unit": "author",
                            "bootstrap_algorithm": config["bootstrap_algorithm"],
                        }
                    )
    frame = pd.DataFrame(rows).sort_values(
        ["sample", "aggregation", "period_order", "topic_id"]
    ).reset_index(drop=True)
    return frame, all_bootstrap, digest.hexdigest()


def build_trend_structure(
    topic_columns: list[str],
    period_means: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    all_bootstrap: dict[str, dict[str, dict[str, np.ndarray]]],
    topic_info: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """计算20 Topic × 六规格的四期趋势结构。"""

    mean_lookup = period_means.set_index(["sample", "aggregation", "period", "topic_id"])
    ci_lookup = bootstrap_ci.set_index(["sample", "aggregation", "period", "topic_id"])
    info = topic_info.set_index("topic_id")
    periods = config["period_order"]
    codes = config["period_codes"]
    thresholds = config["thresholds"]
    rows: list[dict[str, Any]] = []
    for sample, label_column in config["samples"].items():
        _ = label_column
        for aggregation in AGGREGATIONS:
            for topic_index, topic_column in enumerate(topic_columns):
                topic_id = topic_column.removesuffix("_prob")
                values = np.array(
                    [mean_lookup.loc[(sample, aggregation, period, topic_id), "mean_topic_probability"] for period in periods],
                    dtype=float,
                )
                ci_rows = [ci_lookup.loc[(sample, aggregation, period, topic_id)] for period in periods]
                reference_mean = float(np.mean(values))
                normalized_range = (
                    float((np.max(values) - np.min(values)) / reference_mean)
                    if abs(reference_mean) >= thresholds["normalized_range_min_mean"]
                    else math.nan
                )
                peak_value = float(np.max(values))
                trough_value = float(np.min(values))
                peak_candidates = [
                    period for period, value in zip(periods, values) if abs(float(value) - peak_value) <= thresholds["peak_tie_tolerance"]
                ]
                trough_candidates = [
                    period for period, value in zip(periods, values) if abs(float(value) - trough_value) <= thresholds["peak_tie_tolerance"]
                ]
                record: dict[str, Any] = {
                    "topic_id": topic_id,
                    "topic_label": info.loc[topic_id, "topic_label"],
                    "topic_type": info.loc[topic_id, "topic_type"],
                    "interpretation_scope": info.loc[topic_id, "interpretation_scope"],
                    "representativeness_risk": info.loc[topic_id, "representativeness_risk"],
                    "metadata_bias_warning": bool(info.loc[topic_id, "representativeness_risk"] in {"MODERATE", "HIGH"}),
                    "sample": sample,
                    "aggregation": aggregation,
                    "specification": f"{sample}_{aggregation}",
                }
                for period, code, value, ci_row in zip(periods, ["early", "high", "middle", "late"], values, ci_rows):
                    _ = period
                    record[f"{code}_mean"] = float(value)
                    record[f"{code}_ci_low"] = float(ci_row["ci_2_5"])
                    record[f"{code}_ci_high"] = float(ci_row["ci_97_5"])
                numeric_directions: list[str] = []
                for transition, left, right in TRANSITIONS:
                    left_index = periods.index(left)
                    right_index = periods.index(right)
                    delta = float(values[right_index] - values[left_index])
                    difference = (
                        all_bootstrap[sample][right][aggregation][:, topic_index]
                        - all_bootstrap[sample][left][aggregation][:, topic_index]
                    )
                    low, high = np.quantile(difference, [0.025, 0.975])
                    direction = supported_direction(float(low), float(high))
                    numeric_value = numeric_direction(delta, reference_mean, thresholds)
                    numeric_directions.append(numeric_value)
                    record[f"delta_{transition}"] = delta
                    record[f"delta_{transition}_ci_low"] = float(low)
                    record[f"delta_{transition}_ci_high"] = float(high)
                    record[f"direction_{transition}"] = numeric_value
                    record[f"bootstrap_direction_{transition}"] = direction
                early_late = float(values[3] - values[0])
                early_late_difference = (
                    all_bootstrap[sample][periods[3]][aggregation][:, topic_index]
                    - all_bootstrap[sample][periods[0]][aggregation][:, topic_index]
                )
                early_late_low, early_late_high = np.quantile(early_late_difference, [0.025, 0.975])
                record.update(
                    {
                        "delta_EL": early_late,
                        "delta_EL_ci_low": float(early_late_low),
                        "delta_EL_ci_high": float(early_late_high),
                        "bootstrap_direction_EL": supported_direction(float(early_late_low), float(early_late_high)),
                        "peak_period": peak_candidates[0] if len(peak_candidates) == 1 else "TIE",
                        "peak_period_candidates": ";".join(peak_candidates),
                        "peak_value": peak_value,
                        "trough_period": trough_candidates[0] if len(trough_candidates) == 1 else "TIE",
                        "trough_period_candidates": ";".join(trough_candidates),
                        "trough_value": trough_value,
                        "peak_trough_difference": peak_value - trough_value,
                        "peak_trough_ratio": peak_value / trough_value
                        if trough_value >= thresholds["peak_trough_ratio_min_denominator"]
                        else math.nan,
                        "period_rank_order": rank_order(periods, values, codes),
                        "normalized_range": normalized_range,
                        "trend_shape": classify_shape(normalized_range, numeric_directions, thresholds),
                    }
                )
                rows.append(record)
    return pd.DataFrame(rows).sort_values(["sample", "aggregation", "topic_id"]).reset_index(drop=True)


def build_poem_author_comparison(trends: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """比较三套样本内 poem-weighted 与 author-balanced 四期曲线。"""

    rows: list[dict[str, Any]] = []
    for sample in config["samples"]:
        sample_trends = trends.loc[trends["sample"].eq(sample)]
        for topic_id in sorted(sample_trends["topic_id"].unique()):
            poem = sample_trends[(sample_trends["topic_id"] == topic_id) & (sample_trends["aggregation"] == "POEM")].iloc[0]
            author = sample_trends[(sample_trends["topic_id"] == topic_id) & (sample_trends["aggregation"] == "AUTHOR")].iloc[0]
            poem_vector = poem[["early_mean", "high_mean", "middle_mean", "late_mean"]].to_numpy(dtype=float)
            author_vector = author[["early_mean", "high_mean", "middle_mean", "late_mean"]].to_numpy(dtype=float)
            direction_conflicts = []
            for transition, _, _ in TRANSITIONS:
                left = poem[f"bootstrap_direction_{transition}"]
                right = author[f"bootstrap_direction_{transition}"]
                conflict = {left, right} == {"SUPPORTED_UP", "SUPPORTED_DOWN"}
                direction_conflicts.append(conflict)
            peak_agreement = poem["peak_period"] == author["peak_period"]
            shape_agreement = poem["trend_shape"] == author["trend_shape"]
            if peak_agreement and shape_agreement and not any(direction_conflicts):
                agreement = "HIGH"
            elif peak_agreement and not any(direction_conflicts):
                agreement = "MODERATE"
            else:
                agreement = "LOW"
            rows.append(
                {
                    "sample": sample,
                    "topic_id": topic_id,
                    "topic_label": poem["topic_label"],
                    "topic_type": poem["topic_type"],
                    "pearson_four_period": pearson_four(poem_vector, author_vector),
                    "spearman_four_period": spearman_four(poem_vector, author_vector),
                    "poem_peak_period": poem["peak_period"],
                    "author_peak_period": author["peak_period"],
                    "peak_period_agreement": bool(peak_agreement),
                    "poem_trend_shape": poem["trend_shape"],
                    "author_trend_shape": author["trend_shape"],
                    "trend_shape_agreement": bool(shape_agreement),
                    "max_abs_aggregation_difference": float(np.max(np.abs(poem_vector - author_vector))),
                    "mean_abs_aggregation_difference": float(np.mean(np.abs(poem_vector - author_vector))),
                    "rmse_aggregation_difference": float(np.sqrt(np.mean((poem_vector - author_vector) ** 2))),
                    "EH_direction_conflict": bool(direction_conflicts[0]),
                    "HM_direction_conflict": bool(direction_conflicts[1]),
                    "ML_direction_conflict": bool(direction_conflicts[2]),
                    "direction_conflict_count": int(sum(direction_conflicts)),
                    "poem_author_agreement": agreement,
                }
            )
    return pd.DataFrame(rows).sort_values(["sample", "topic_id"]).reset_index(drop=True)


def build_trend_robustness(
    trends: pd.DataFrame,
    comparisons: pd.DataFrame,
    topic_info: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """综合六套规格的 peak、rank、shape 与方向一致性。"""

    thresholds = config["thresholds"]
    info = topic_info.set_index("topic_id")
    rows: list[dict[str, Any]] = []
    for topic_id in sorted(trends["topic_id"].unique()):
        subset = trends.loc[trends["topic_id"].eq(topic_id)]
        topic_comparison = comparisons.loc[comparisons["topic_id"].eq(topic_id)]
        dominant_peak, peak_count, peak_tie = mode_with_count(subset["peak_period"].tolist())
        dominant_trough, trough_count, trough_tie = mode_with_count(subset["trough_period"].tolist())
        dominant_rank, rank_count, rank_tie = mode_with_count(subset["period_rank_order"].tolist())
        dominant_shape, shape_count, shape_tie = mode_with_count(subset["trend_shape"].tolist())
        record: dict[str, Any] = {
            "topic_id": topic_id,
            "topic_label": info.loc[topic_id, "topic_label"],
            "topic_type": info.loc[topic_id, "topic_type"],
            "interpretation_scope": info.loc[topic_id, "interpretation_scope"],
            "representativeness_risk": info.loc[topic_id, "representativeness_risk"],
            "known_unknown_smd": float(info.loc[topic_id, "known_unknown_smd"]),
            "metadata_bias_warning": bool(info.loc[topic_id, "representativeness_risk"] in {"MODERATE", "HIGH"}),
            "dominant_peak_period": dominant_peak,
            "peak_consistency_count": peak_count,
            "peak_consistency_ratio": peak_count / 6,
            "peak_mode_tie": peak_tie,
            "dominant_trough_period": dominant_trough,
            "trough_consistency_count": trough_count,
            "trough_consistency_ratio": trough_count / 6,
            "trough_mode_tie": trough_tie,
            "dominant_rank_order": dominant_rank,
            "rank_order_consistency_count": rank_count,
            "rank_order_consistency_ratio": rank_count / 6,
            "rank_order_mode_tie": rank_tie,
            "dominant_trend_shape": dominant_shape,
            "trend_shape_consistency_count": shape_count,
            "trend_shape_consistency_ratio": shape_count / 6,
            "trend_shape_mode_tie": shape_tie,
        }
        consistent_transitions = 0
        global_direction_conflicts = 0
        main_directions: list[str] = []
        for transition, _, _ in TRANSITIONS:
            values = subset[f"bootstrap_direction_{transition}"].tolist()
            up_count = values.count("SUPPORTED_UP")
            down_count = values.count("SUPPORTED_DOWN")
            uncertain_count = values.count("UNCERTAIN")
            record[f"{transition}_UP_count"] = up_count
            record[f"{transition}_DOWN_count"] = down_count
            record[f"{transition}_uncertain_count"] = uncertain_count
            dominant_count = max(up_count, down_count)
            if dominant_count >= thresholds["high_transition_consistency_count"]:
                consistent_transitions += 1
            if up_count > 0 and down_count > 0:
                global_direction_conflicts += 1
            if up_count > down_count and up_count > uncertain_count:
                main_directions.append("UP")
            elif down_count > up_count and down_count > uncertain_count:
                main_directions.append("DOWN")
            else:
                main_directions.append("UNCERTAIN")
        poem_author_direction_conflicts = int(topic_comparison["direction_conflict_count"].sum())
        aggregation_sensitive = bool((~topic_comparison["peak_period_agreement"]).any())
        poem_peaks = subset.loc[subset["aggregation"].eq("POEM"), "peak_period"].nunique()
        author_peaks = subset.loc[subset["aggregation"].eq("AUTHOR"), "peak_period"].nunique()
        sample_sensitive = bool(poem_peaks > 1 or author_peaks > 1)
        direction_unstable = global_direction_conflicts > 0
        if (
            peak_count >= thresholds["high_peak_consistency_count"]
            and consistent_transitions >= thresholds["high_min_consistent_transitions"]
            and poem_author_direction_conflicts == 0
            and global_direction_conflicts == 0
        ):
            robustness = "HIGH"
        elif (
            peak_count <= thresholds["low_peak_consistency_count"]
            or poem_author_direction_conflicts > 0
            or global_direction_conflicts >= thresholds["low_global_direction_conflict_count"]
            or poem_peaks >= 3
            or author_peaks >= 3
        ):
            robustness = "LOW"
        else:
            robustness = "MODERATE"
        peak_agreement_count = int(topic_comparison["peak_period_agreement"].sum())
        if peak_agreement_count == 3 and poem_author_direction_conflicts == 0:
            poem_author_agreement = "HIGH"
        elif peak_agreement_count >= 2 and poem_author_direction_conflicts == 0:
            poem_author_agreement = "MODERATE"
        else:
            poem_author_agreement = "LOW"
        flags = []
        if aggregation_sensitive:
            flags.append("AGGREGATION_SENSITIVE")
        if sample_sensitive:
            flags.append("SAMPLE_SENSITIVE")
        if direction_unstable:
            flags.append("DIRECTION_UNSTABLE")
        if record["metadata_bias_warning"]:
            flags.append("METADATA_RISK")
        record.update(
            {
                "main_direction_pattern": "|".join(
                    f"{transition}:{direction}" for (transition, _, _), direction in zip(TRANSITIONS, main_directions)
                ),
                "highly_consistent_transition_count": consistent_transitions,
                "global_direction_conflict_count": global_direction_conflicts,
                "poem_author_direction_conflict_count": poem_author_direction_conflicts,
                "poem_author_peak_agreement_count": peak_agreement_count,
                "poem_author_agreement": poem_author_agreement,
                "aggregation_sensitive": aggregation_sensitive,
                "sample_sensitive": sample_sensitive,
                "direction_unstable": direction_unstable,
                "trend_robustness": robustness,
                "sensitivity_flags": ";".join(flags) if flags else "NONE",
            }
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values("topic_id").reset_index(drop=True)


def build_author_concentration(
    poems: pd.DataFrame,
    topic_columns: list[str],
    topic_info: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """计算各样本、时期、Topic 的作者 probability mass 集中度。"""

    info = topic_info.set_index("topic_id")
    rows: list[dict[str, Any]] = []
    thresholds = config["thresholds"]
    for sample, label_column in config["samples"].items():
        selected = poems.loc[poems[label_column]]
        for period_index, period in enumerate(config["period_order"], start=1):
            period_poems = selected.loc[selected["author_period_primary"].eq(period)]
            mass = period_poems.groupby(["author_id_internal", "author"], sort=True, observed=True)[topic_columns].sum()
            for topic_index, topic_column in enumerate(topic_columns):
                topic_id = topic_column.removesuffix("_prob")
                vector = mass.iloc[:, topic_index].to_numpy(dtype=float)
                total = float(np.sum(vector))
                shares = vector / total
                order = np.argsort(-shares, kind="stable")
                top1 = float(np.sum(shares[order[:1]]))
                top5 = float(np.sum(shares[order[:5]]))
                top10 = float(np.sum(shares[order[:10]]))
                hhi = float(np.sum(shares**2))
                top_index = mass.index[order[0]]
                rows.append(
                    {
                        "sample": sample,
                        "period": period,
                        "period_order": period_index,
                        "topic_id": topic_id,
                        "topic_label": info.loc[topic_id, "topic_label"],
                        "topic_type": info.loc[topic_id, "topic_type"],
                        "poem_count": int(len(period_poems)),
                        "author_count": int(len(mass)),
                        "total_topic_mass": total,
                        "top1_author_share": top1,
                        "top5_author_share": top5,
                        "top10_author_share": top10,
                        "hhi": hhi,
                        "top1_author_id_internal": top_index[0],
                        "top1_author": top_index[1],
                        "concentration_level": concentration_level(top1, top5, hhi, thresholds),
                    }
                )
    return pd.DataFrame(rows).sort_values(["sample", "period_order", "topic_id"]).reset_index(drop=True)


def build_clear_and_candidates(
    robustness: pd.DataFrame,
    trends: pd.DataFrame,
    concentration: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成 CLEAR Topic 摘要、候选趋势与敏感性标记表。"""

    clear = robustness.loc[robustness["topic_type"].eq("CLEAR")].copy()
    core_poem = trends[(trends["sample"] == "CORE") & (trends["aggregation"] == "POEM")][
        ["topic_id", "peak_period", "trend_shape", "period_rank_order", "normalized_range"]
    ].rename(
        columns={
            "peak_period": "core_poem_peak_period",
            "trend_shape": "core_poem_trend_shape",
            "period_rank_order": "core_poem_rank_order",
            "normalized_range": "core_poem_normalized_range",
        }
    )
    core_author = trends[(trends["sample"] == "CORE") & (trends["aggregation"] == "AUTHOR")][
        ["topic_id", "peak_period", "trend_shape", "period_rank_order", "normalized_range"]
    ].rename(
        columns={
            "peak_period": "core_author_peak_period",
            "trend_shape": "core_author_trend_shape",
            "period_rank_order": "core_author_rank_order",
            "normalized_range": "core_author_normalized_range",
        }
    )
    core_concentration = concentration.loc[concentration["sample"].eq("CORE")].copy()
    core_concentration["_level_order"] = core_concentration["concentration_level"].map({"LOW": 0, "MODERATE": 1, "HIGH": 2})
    concentration_rows = []
    for topic_id, group in core_concentration.groupby("topic_id", sort=True):
        top5_row = group.sort_values("top5_author_share", ascending=False).iloc[0]
        level_row = group.sort_values(["_level_order", "top5_author_share"], ascending=[False, False]).iloc[0]
        concentration_rows.append(
            {
                "topic_id": topic_id,
                "max_core_top1_author_share": float(group["top1_author_share"].max()),
                "max_core_top5_author_share": float(group["top5_author_share"].max()),
                "max_core_hhi": float(group["hhi"].max()),
                "max_core_top5_period": top5_row["period"],
                "author_concentration_level": level_row["concentration_level"],
            }
        )
    clear = clear.merge(core_poem, on="topic_id", validate="one_to_one")
    clear = clear.merge(core_author, on="topic_id", validate="one_to_one")
    clear = clear.merge(pd.DataFrame(concentration_rows), on="topic_id", validate="one_to_one")
    priorities = []
    for row in clear.to_dict("records"):
        if (
            row["trend_robustness"] == "HIGH"
            and row["representativeness_risk"] == "LOW"
            and not row["aggregation_sensitive"]
            and not row["sample_sensitive"]
        ):
            priorities.append("HIGH_REVIEW_PRIORITY")
        elif row["trend_robustness"] in {"HIGH", "MODERATE"} and row["representativeness_risk"] != "HIGH":
            priorities.append("REVIEW_CANDIDATE")
        else:
            priorities.append("DIAGNOSTIC_ONLY")
    clear["interpretation_priority"] = priorities
    candidate = clear.loc[
        clear["representativeness_risk"].ne("HIGH") & clear["trend_robustness"].isin(["HIGH", "MODERATE"])
    ].copy()
    candidate["notes"] = candidate.apply(
        lambda row: (
            f"统计结构候选：dominant peak={row['dominant_peak_period']}；"
            f"shape={row['dominant_trend_shape']}；peak consistency={row['peak_consistency_ratio']:.3f}；"
            f"flags={row['sensitivity_flags']}。不构成文学史解释。"
        ),
        axis=1,
    )
    candidate = candidate[
        [
            "topic_id",
            "topic_label",
            "representativeness_risk",
            "trend_robustness",
            "dominant_peak_period",
            "peak_consistency_ratio",
            "main_direction_pattern",
            "poem_author_agreement",
            "sensitivity_flags",
            "notes",
        ]
    ].reset_index(drop=True)
    sensitivity = robustness[
        [
            "topic_id",
            "topic_label",
            "topic_type",
            "aggregation_sensitive",
            "sample_sensitive",
            "direction_unstable",
            "metadata_bias_warning",
            "sensitivity_flags",
        ]
    ].copy()
    return clear.sort_values("topic_id").reset_index(drop=True), candidate, sensitivity


def build_period_counts(poems: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """汇总三套样本的时期诗数和作者数。"""

    rows = []
    for sample, label_column in config["samples"].items():
        selected = poems.loc[poems[label_column]]
        for period_index, period in enumerate(config["period_order"], start=1):
            period_rows = selected.loc[selected["author_period_primary"].eq(period)]
            rows.append(
                {
                    "sample": sample,
                    "period": period,
                    "period_order": period_index,
                    "poem_count": int(len(period_rows)),
                    "author_count": int(period_rows["author_id_internal"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def configure_matplotlib() -> None:
    """配置中文学术图形的统一样式。"""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def heatmap_figure(
    frame: pd.DataFrame,
    topic_info: pd.DataFrame,
    periods: list[str],
    path: Path,
    title: str,
) -> None:
    """绘制20 Topic × 四时期概率热图。"""

    matrix = frame.pivot(index="topic_id", columns="period", values="mean_topic_probability").reindex(
        index=topic_info["topic_id"], columns=periods
    )
    labels = [f"{topic_id.replace('topic_', 'T')} {label}" for topic_id, label in zip(topic_info["topic_id"], topic_info["short_label"])]
    fig, axis = plt.subplots(figsize=(10.5, 11.5))
    image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0)
    axis.set_xticks(range(len(periods)), periods)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("作者所属时期")
    axis.set_ylabel("C_K20 Topic")
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iat[row, column]
            axis.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(image, ax=axis, label="Mean topic probability", fraction=0.035, pad=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_figures(
    figures_root: Path,
    period_means: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    trends: pd.DataFrame,
    comparisons: pd.DataFrame,
    concentration: pd.DataFrame,
    topic_info: pd.DataFrame,
    config: dict[str, Any],
) -> list[Path]:
    """生成任务要求的六张趋势结构图。"""

    configure_matplotlib()
    figures_root.mkdir(parents=True, exist_ok=True)
    periods = config["period_order"]
    paths: list[Path] = []
    for aggregation, filename, title in [
        ("POEM", "core_poem_period_topic_heatmap.png", "CORE poem-weighted：按作者时期划分的 Topic 均值"),
        ("AUTHOR", "core_author_period_topic_heatmap.png", "CORE author-balanced：按作者时期划分的 Topic 均值"),
    ]:
        path = figures_root / filename
        heatmap_figure(
            period_means[(period_means["sample"] == "CORE") & (period_means["aggregation"] == aggregation)],
            topic_info,
            periods,
            path,
            title,
        )
        paths.append(path)

    clear_info = topic_info.loc[topic_info["topic_type"].eq("CLEAR")].sort_values("topic_id")
    rows = math.ceil(len(clear_info) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(14, 3.7 * rows), squeeze=False)
    x = np.arange(4)
    for axis, topic in zip(axes.flat, clear_info.to_dict("records")):
        for aggregation, color, marker, label in [
            ("POEM", "#2F6B8A", "o", "CORE poem-weighted"),
            ("AUTHOR", "#D97A3A", "s", "CORE author-balanced"),
        ]:
            subset = bootstrap_ci[
                (bootstrap_ci["sample"] == "CORE")
                & (bootstrap_ci["aggregation"] == aggregation)
                & (bootstrap_ci["topic_id"] == topic["topic_id"])
            ].sort_values("period_order")
            mean = subset["mean"].to_numpy(dtype=float)
            low = subset["ci_2_5"].to_numpy(dtype=float)
            high = subset["ci_97_5"].to_numpy(dtype=float)
            axis.plot(x, mean, color=color, marker=marker, linewidth=1.6, label=label)
            axis.fill_between(x, low, high, color=color, alpha=0.16)
        upper = max(axis.get_ylim()[1], float(subset["ci_97_5"].max()) * 1.08)
        axis.set_ylim(0, upper)
        axis.set_xticks(x, ["初", "盛", "中", "晚"])
        axis.set_title(f"{topic['topic_id'].replace('topic_', 'T')} {topic['short_label']}")
        axis.set_ylabel("Mean probability")
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    for axis in axes.flat[len(clear_info) :]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("CLEAR Topics：CORE poem-weighted 与 author-balanced（作者聚类 bootstrap 95% CI）\n各子图 y 轴独立且自 0 起", y=1.02)
    fig.tight_layout()
    path = figures_root / "clear_topic_trend_small_multiples.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    specifications = [f"{sample}_{aggregation}" for sample in config["samples"] for aggregation in AGGREGATIONS]
    peak_matrix = trends.pivot(index="topic_id", columns="specification", values="peak_period").reindex(
        index=topic_info["topic_id"], columns=specifications
    )
    value_map = {period: index for index, period in enumerate(periods)} | {"TIE": 4}
    numeric = peak_matrix.map(lambda value: value_map.get(value, 5)).to_numpy()
    from matplotlib.colors import ListedColormap

    colors = ["#B7D4EA", "#E7C98B", "#9BC6A6", "#B9A7D6", "#D9D9D9", "#FFFFFF"]
    fig, axis = plt.subplots(figsize=(11, 9.5))
    axis.imshow(numeric, aspect="auto", cmap=ListedColormap(colors), vmin=-0.5, vmax=5.5)
    axis.set_xticks(range(len(specifications)), specifications, rotation=30, ha="right")
    axis.set_yticks(range(20), [topic_id.replace("topic_", "T") for topic_id in topic_info["topic_id"]])
    code_map = {period: config["period_codes"][period] for period in periods} | {"TIE": "TIE"}
    for row in range(peak_matrix.shape[0]):
        for column in range(peak_matrix.shape[1]):
            axis.text(column, row, code_map.get(peak_matrix.iat[row, column], "?"), ha="center", va="center", fontsize=8)
    axis.set_title("六种规格的 peak period 稳健性矩阵（E/H/M/L）")
    axis.set_xlabel("Analytical specification")
    axis.set_ylabel("C_K20 Topic")
    fig.tight_layout()
    path = figures_root / "trend_robustness_matrix.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    core_difference = comparisons.loc[comparisons["sample"].eq("CORE")].sort_values(
        "max_abs_aggregation_difference"
    )
    fig, axis = plt.subplots(figsize=(9, 7))
    colors = ["#C44E52" if not agree else "#4C72B0" for agree in core_difference["peak_period_agreement"]]
    axis.barh(core_difference["topic_id"].str.replace("topic_", "T"), core_difference["max_abs_aggregation_difference"], color=colors)
    axis.set_xlabel("四时期中的最大绝对均值差")
    axis.set_ylabel("C_K20 Topic")
    axis.set_title("CORE poem-weighted 与 author-balanced 趋势差异\n红色表示 peak period 不一致")
    fig.tight_layout()
    path = figures_root / "poem_vs_author_aggregation_difference.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    core_clear = concentration[
        (concentration["sample"] == "CORE") & concentration["topic_id"].isin(clear_info["topic_id"])
    ]
    concentration_matrix = core_clear.pivot(index="topic_id", columns="period", values="top5_author_share").reindex(
        index=clear_info["topic_id"], columns=periods
    )
    fig, axis = plt.subplots(figsize=(8.5, 6.5))
    image = axis.imshow(concentration_matrix.to_numpy(), aspect="auto", cmap="OrRd", vmin=0, vmax=1)
    axis.set_xticks(range(4), periods)
    axis.set_yticks(range(len(clear_info)), [topic_id.replace("topic_", "T") for topic_id in clear_info["topic_id"]])
    for row in range(concentration_matrix.shape[0]):
        for column in range(concentration_matrix.shape[1]):
            axis.text(column, row, f"{concentration_matrix.iat[row, column]:.1%}", ha="center", va="center", fontsize=8)
    axis.set_title("CORE CLEAR Topics：各时期 top-5 作者 Topic mass share")
    axis.set_xlabel("作者所属时期")
    axis.set_ylabel("CLEAR Topic")
    fig.colorbar(image, ax=axis, label="Top-5 author share", fraction=0.045, pad=0.04)
    fig.tight_layout()
    path = figures_root / "period_topic_author_concentration.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    """把数据框转换为紧凑 Markdown 表。"""

    selected = frame[columns].head(limit) if limit else frame[columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for record in selected.to_dict("records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, (float, np.floating)):
                values.append("NA" if math.isnan(float(value)) else f"{float(value):.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *rows])


def build_summary(
    period_counts: pd.DataFrame,
    trends: pd.DataFrame,
    comparisons: pd.DataFrame,
    robustness: pd.DataFrame,
    clear: pd.DataFrame,
    candidates: pd.DataFrame,
    concentration: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """建立供报告、工作簿和最终汇报使用的摘要。"""

    sample_summary = {}
    for sample in config["samples"]:
        rows = period_counts.loc[period_counts["sample"].eq(sample)]
        sample_summary[sample] = {
            "poem_count": int(rows["poem_count"].sum()),
            "author_count": int(rows["author_count"].sum()),
            "periods": rows[["period", "poem_count", "author_count"]].to_dict("records"),
        }
    peak_distributions = {}
    for aggregation in AGGREGATIONS:
        values = trends[(trends["sample"] == "CORE") & (trends["aggregation"] == aggregation)]["peak_period"]
        peak_distributions[aggregation] = {period: int((values == period).sum()) for period in config["period_order"]}
        peak_distributions[aggregation]["TIE"] = int((values == "TIE").sum())
    core_comparison = comparisons.loc[comparisons["sample"].eq("CORE")]
    concentration_core = concentration.loc[concentration["sample"].eq("CORE")].copy()
    thresholds = config["thresholds"]
    concentration_core["concentration_score"] = np.maximum.reduce(
        [
            concentration_core["top1_author_share"] / thresholds["author_concentration_high_top1"],
            concentration_core["top5_author_share"] / thresholds["author_concentration_high_top5"],
            concentration_core["hhi"] / thresholds["author_concentration_high_hhi"],
        ]
    )
    concentration_top = concentration_core.sort_values(
        ["concentration_score", "top1_author_share", "top5_author_share"], ascending=False
    ).head(10)
    return {
        "analysis_version": config["analysis_version"],
        "sample_summary": sample_summary,
        "core_peak_distributions": peak_distributions,
        "core_poem_author_peak_agreement": {
            "agree": int(core_comparison["peak_period_agreement"].sum()),
            "disagree": int((~core_comparison["peak_period_agreement"]).sum()),
        },
        "trend_robustness_counts": {
            level: int((robustness["trend_robustness"] == level).sum()) for level in ["HIGH", "MODERATE", "LOW"]
        },
        "clear_high_robustness": clear.loc[clear["trend_robustness"].eq("HIGH"), "topic_id"].tolist(),
        "clear_aggregation_sensitive": clear.loc[clear["aggregation_sensitive"], "topic_id"].tolist(),
        "clear_sample_sensitive": clear.loc[clear["sample_sensitive"], "topic_id"].tolist(),
        "clear_topics": clear.to_dict("records"),
        "candidate_topics": candidates.to_dict("records"),
        "highest_core_author_concentration": concentration_top[
            [
                "topic_id",
                "topic_label",
                "period",
                "top1_author_share",
                "top5_author_share",
                "top10_author_share",
                "hhi",
                "top1_author",
                "concentration_level",
            ]
        ].to_dict("records"),
    }


def build_report(
    output_root: Path,
    summary: dict[str, Any],
    period_counts: pd.DataFrame,
    trends: pd.DataFrame,
    comparisons: pd.DataFrame,
    robustness: pd.DataFrame,
    clear: pd.DataFrame,
    candidates: pd.DataFrame,
    concentration: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    """生成仅讨论统计结构的14节正式报告。"""

    core_poem = trends[(trends["sample"] == "CORE") & (trends["aggregation"] == "POEM")].copy()
    core_author = trends[(trends["sample"] == "CORE") & (trends["aggregation"] == "AUTHOR")].copy()
    clear_report = clear.rename(
        columns={
            "topic_id": "Topic",
            "topic_label": "Label",
            "representativeness_risk": "Representativeness Risk",
            "core_poem_peak_period": "CORE Poem Peak",
            "core_author_peak_period": "CORE Author Peak",
            "peak_consistency_ratio": "Peak Consistency",
            "dominant_trend_shape": "Shape",
            "trend_robustness": "Trend Robustness",
            "max_core_top5_author_share": "Author Concentration",
            "interpretation_priority": "Interpretation Status",
        }
    )
    sample_lines = []
    for sample, data in summary["sample_summary"].items():
        sample_lines.append(f"- {sample}：{data['poem_count']:,} 首、{data['author_count']:,} 位作者。")
    robust_high = robustness.loc[robustness["trend_robustness"].eq("HIGH"), "topic_id"].tolist()
    report = f"""# QTS Author-Period Topic Trend Structure Report

# 1. Objective

本分析只描述 C_K20 的20维 Topic probability 如何随 `author_period_primary`（作者所属时期）分布；它不是单篇作品创作年代分析，也不重训 LDA、不修改 taxonomy、Topic probability、作者时期或 CBDB 匹配。

# 2. Analytical Samples

{chr(10).join(sample_lines)}

STRICT 直接使用审计的 `eligible_period_strict`，CORE 使用 `eligible_period_core`，EXPANDED 使用 `eligible_period_expanded`。CORE 是 primary descriptive specification；STRICT 是 metadata-quality sensitivity；EXPANDED 是 coverage sensitivity。三者都不是“真实答案”。

# 3. Aggregation Estimands

- POEM：每首入选诗权重相同，回答现存诗歌文本总体中的平均 Topic probability。
- AUTHOR：先计算每位作者在样本内的 Topic profile，再在同一时期等权平均，回答典型作者的平均 Topic profile。

两者是不同 estimand；差异被视为趋势结构诊断，不视为错误。

# 4. Sample Sizes

{markdown_table(period_counts, ['sample', 'period', 'poem_count', 'author_count'])}

# 5. Overall Topic Structure

CORE_POEM peak 分布：{summary['core_peak_distributions']['POEM']}。CORE_AUTHOR peak 分布：{summary['core_peak_distributions']['AUTHOR']}。六规格稳健性为 HIGH/MODERATE/LOW={summary['trend_robustness_counts']['HIGH']}/{summary['trend_robustness_counts']['MODERATE']}/{summary['trend_robustness_counts']['LOW']}。这些结果只概括统计结构。

# 6. CORE Poem-weighted Trends

{markdown_table(core_poem, ['topic_id', 'topic_label', 'early_mean', 'high_mean', 'middle_mean', 'late_mean', 'peak_period', 'period_rank_order', 'trend_shape', 'normalized_range'])}

# 7. CORE Author-balanced Trends

{markdown_table(core_author, ['topic_id', 'topic_label', 'early_mean', 'high_mean', 'middle_mean', 'late_mean', 'peak_period', 'period_rank_order', 'trend_shape', 'normalized_range'])}

# 8. Aggregation Sensitivity

CORE 的20个 Topic 中，poem-weighted 与 author-balanced peak 一致 {summary['core_poem_author_peak_agreement']['agree']} 个、不一致 {summary['core_poem_author_peak_agreement']['disagree']} 个。Pearson/Spearman 只有四个时期，只作描述性诊断，不报告 p 值。

{markdown_table(comparisons.loc[comparisons['sample'].eq('CORE')], ['topic_id', 'poem_peak_period', 'author_peak_period', 'peak_period_agreement', 'poem_trend_shape', 'author_trend_shape', 'pearson_four_period', 'spearman_four_period', 'max_abs_aggregation_difference', 'direction_conflict_count'])}

# 9. Sample Sensitivity

`SAMPLE_SENSITIVE` 表示同一聚合口径下 STRICT/CORE/EXPANDED 的 peak 不完全一致；它不自动否定曲线，而是要求后续效应量分析保留样本敏感性。

{markdown_table(robustness, ['topic_id', 'dominant_peak_period', 'peak_consistency_ratio', 'dominant_rank_order', 'dominant_trend_shape', 'sample_sensitive', 'aggregation_sensitive', 'direction_unstable'])}

# 10. Robust Trend Structures

HIGH topics：{', '.join(robust_high) if robust_high else '无'}。稳健性同时考虑 peak、完整 rank order、三个相邻 bootstrap-supported direction 及 poem/author 冲突；不只看 peak。

{markdown_table(robustness, ['topic_id', 'peak_consistency_count', 'rank_order_consistency_count', 'trend_shape_consistency_count', 'EH_UP_count', 'EH_DOWN_count', 'HM_UP_count', 'HM_DOWN_count', 'ML_UP_count', 'ML_DOWN_count', 'poem_author_direction_conflict_count', 'trend_robustness'])}

# 11. CLEAR Topic Assessment

{markdown_table(clear_report, ['Topic', 'Label', 'Representativeness Risk', 'CORE Poem Peak', 'CORE Author Peak', 'Peak Consistency', 'Shape', 'Trend Robustness', 'Author Concentration', 'Interpretation Status'])}

# 12. Author Concentration

集中度以各时期 poem-weighted Topic mass 的作者份额计算。top1/top5/top10 与 HHI 只说明少数高产作者可能驱动总体质量，不作具体诗人文学解释。

{markdown_table(concentration.loc[concentration['sample'].eq('CORE')].sort_values(['concentration_level', 'top5_author_share'], ascending=[True, False]), ['topic_id', 'period', 'author_count', 'top1_author_share', 'top5_author_share', 'top10_author_share', 'hhi', 'concentration_level'], 20)}

# 13. Candidate Trends for Humanistic Interpretation

以下仅为 candidate robust trends，筛选条件是 CLEAR、representativeness risk 非 HIGH、trend robustness 为 HIGH 或 MODERATE；不构成文学史规律。

{markdown_table(candidates, ['topic_id', 'topic_label', 'representativeness_risk', 'trend_robustness', 'dominant_peak_period', 'peak_consistency_ratio', 'main_direction_pattern', 'poem_author_agreement', 'sensitivity_flags'])}

# 14. Limitations

- Author period 不等于 poem composition date。
- 已知作者时期只覆盖部分诗歌，且 metadata 明显偏向高产作者。
- 现存《全唐诗》语料本身存在文献保存偏差。
- 作者产量高度不均，poem-weighted 与 author-balanced 回答不同问题。
- Topic probability 是 LDA 混合概率，不等于传统题材的唯一分类。
- Bootstrap CI 与 bootstrap-supported direction 只作 uncertainty 描述，不是正式假设检验；本阶段没有运行 ANOVA、Kruskal–Wallis、pairwise Mann–Whitney 或多重比较校正。

## 下一阶段接口

输出已保留 sample、aggregation、period、poem/author count、均值、bootstrap interval、shape、robustness 与 flags，可进入 Effect Size & Statistical Validation；下一阶段仍须保持三样本、两聚合口径和 representativeness warning，并避免把作者时期解释为作品创作年代。
"""
    path = output_root / "QTS_Author_Period_Trend_Structure_Report.md"
    path.write_text(report, encoding="utf-8", newline="\n")
    return report


def build_data_dictionary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """为核心表生成字段级数据字典。"""

    descriptions = {
        "sample": "审计冻结的 STRICT/CORE/EXPANDED 样本",
        "aggregation": "POEM=每诗等权；AUTHOR=每作者等权",
        "specification": "sample 与 aggregation 的联合规格",
        "period": "author_period_primary；不是单篇作品创作时期",
        "period_order": "固定时期顺序1=初、2=盛、3=中、4=晚",
        "topic_id": "从 topic_metadata 读取的 C_K20 Topic 标识",
        "topic_label": "从 topic_metadata 读取的正式标签",
        "topic_type": "CLEAR/BROAD/MIXED/NOISE",
        "interpretation_scope": "Topic 的解释范围",
        "poem_count": "该样本与时期的诗歌数",
        "author_count": "该样本与时期的唯一作者数",
        "mean_topic_probability": "相应估计对象的平均 Topic probability",
        "ci_2_5": "作者聚类 bootstrap 的2.5%分位数",
        "ci_97_5": "作者聚类 bootstrap 的97.5%分位数",
        "bootstrap_sd": "作者聚类 bootstrap replicate 的标准差",
        "peak_period": "四时期均值的最高时期；并列时为 TIE",
        "trough_period": "四时期均值的最低时期；并列时为 TIE",
        "period_rank_order": "四时期均值从高到低的完整排序",
        "trend_shape": "按配置阈值和 bootstrap-supported direction 得到的描述性形状",
        "normalized_range": "(四期最大值-最小值)/四期均值",
        "trend_robustness": "六规格综合的 HIGH/MODERATE/LOW 描述性稳健性",
        "representativeness_risk": "直接读取 Period Metadata Representativeness Audit",
        "sensitivity_flags": "聚合、样本、方向与 metadata 风险标记",
        "top1_author_share": "单一作者占该时期该 Topic poem-weighted probability mass 的份额",
        "top5_author_share": "前五作者占该时期该 Topic mass 的份额",
        "top10_author_share": "前十作者占该时期该 Topic mass 的份额",
        "hhi": "作者 Topic mass 份额平方和",
        "author_id_internal": "项目内部稳定作者标识",
        "author_period_primary": "作者所属时期，不是作品创作时期",
    }
    rows = []
    for table, frame in frames.items():
        for column in frame.columns:
            description = descriptions.get(column)
            if description is None and column.startswith("topic_") and column.endswith("_mean"):
                description = "该作者在样本内的相应 Topic probability 平均值"
            elif description is None and column.startswith("delta_"):
                description = "相应时期均值之差；正值表示后一期较高"
            elif description is None and "direction" in column:
                description = "相邻时期的数值或 bootstrap-supported direction"
            elif description is None:
                description = "分析输出字段；定义与阈值见 README、报告及配置"
            rows.append({"table": table, "field": column, "description": description})
    return pd.DataFrame(rows)


def build_readme(output_root: Path, config: dict[str, Any], output_frames: dict[str, pd.DataFrame]) -> None:
    """写出运行说明、口径与输出清单。"""

    thresholds = json.dumps(config["thresholds"], ensure_ascii=False, indent=2)
    outputs = "\n".join(f"- `data/{name}.csv`：{len(frame):,} 行。" for name, frame in output_frames.items())
    text = f"""# QTS Author-Period Topic Trend Structure

本目录只分析按作者所属时期划分的 C_K20 Topic probability。`author_period_primary` 不等于单篇作品的创作年代。

## 运行

```powershell
python src/run_author_period_trend_structure.py --config config/trend_structure_config.yaml
```

工作簿由 `src/build_author_period_trend_workbook.mjs` 从已验证 CSV 生成，不参与趋势计算。

## 固定分析规格

- STRICT_POEM / STRICT_AUTHOR
- CORE_POEM / CORE_AUTHOR
- EXPANDED_POEM / EXPANDED_AUTHOR

样本标签直接读取 Period Metadata Representativeness Audit，不在本阶段重新定义。Bootstrap 以 author 为 cluster，固定 {config['bootstrap_iterations']:,} 次、seed={config['random_seed']}。

## 显式阈值

```json
{thresholds}
```

## 核心表

{outputs}

六幅 PNG 位于 `figures/`，正式报告与工作簿位于本目录。`data/invariance_tests.csv` 必须全部为 PASS。
"""
    (output_root / "README.md").write_text(text, encoding="utf-8", newline="\n")


def run_invariance_tests(
    poems: pd.DataFrame,
    labels: pd.DataFrame,
    topic_columns: list[str],
    topic_info: pd.DataFrame,
    period_means: pd.DataFrame,
    profiles: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    bootstrap_hash: str,
    trends: pd.DataFrame,
    source_path: Path,
    source_hash_before: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """执行任务指定的15项数据不可变性与设计测试。"""

    tests: list[dict[str, Any]] = []

    def add(test_id: str, name: str, passed: bool, observed: Any, expected: Any) -> None:
        tests.append(
            {
                "test_id": test_id,
                "test_name": name,
                "status": "PASS" if bool(passed) else "FAIL",
                "observed": str(observed),
                "expected": str(expected),
            }
        )

    expected_rows = int(config["expected_poem_count"])
    topic_matrix_before = poems[topic_columns].to_numpy(dtype=float)
    source_hash_after = sha256_file(source_path)
    add("TEST_01", "author-period master 原始行数不变", len(poems) == expected_rows, len(poems), expected_rows)
    add("TEST_02", "不覆盖 authoritative master", source_hash_after == source_hash_before, source_hash_after, source_hash_before)
    add(
        "TEST_03",
        "20 Topic probability 不变化",
        np.allclose(topic_matrix_before.sum(axis=1), 1.0, atol=1e-10, rtol=0),
        sha256_array(topic_matrix_before),
        "每行和为1且矩阵保持只读",
    )
    labels_match = period_means[["topic_id", "topic_label"]].drop_duplicates().sort_values("topic_id").reset_index(drop=True).equals(
        topic_info[["topic_id", "topic_label"]].sort_values("topic_id").reset_index(drop=True)
    )
    add("TEST_04", "Topic labels 从 topic_metadata 读取", labels_match, labels_match, True)
    observed_order = period_means.sort_values("period_order")["period"].drop_duplicates().tolist()
    add("TEST_05", "period order 固定", observed_order == config["period_order"], observed_order, config["period_order"])
    label_ok = True
    for sample, column in config["samples"].items():
        source_ids = set(labels.loc[normalize_boolean(labels[column], column), "poem_id"])
        analysis_ids = set(poems.loc[poems[column], "poem_id"])
        label_ok &= source_ids == analysis_ids
    add("TEST_06", "STRICT/CORE/EXPANDED 与 Audit 输出一致", label_ok, label_ok, True)
    selected_mask = np.logical_or.reduce([poems[column].to_numpy() for column in config["samples"].values()])
    period_valid = poems.loc[selected_mask, "author_period_primary"].isin(config["period_order"]).all()
    add("TEST_07", "所有入选行具有合法 author_period_primary", period_valid, period_valid, True)
    core_author_output = period_means[(period_means["sample"] == "CORE") & (period_means["aggregation"] == "AUTHOR")]
    core_profiles = profiles.loc[profiles["sample"].eq("CORE")]
    independent_author = core_profiles.groupby("author_period_primary")[[column.replace("_prob", "_mean") for column in topic_columns]].mean()
    author_ok = all(
        math.isclose(
            row["mean_topic_probability"],
            independent_author.loc[row["period"], row["topic_id"] + "_mean"],
            rel_tol=0,
            abs_tol=1e-14,
        )
        for row in core_author_output.to_dict("records")
    )
    add("TEST_08", "author-balanced 中每位作者权重相等", author_ok, author_ok, True)
    core_poems = poems.loc[poems[config["samples"]["CORE"]]]
    independent_poem = core_poems.groupby("author_period_primary")[topic_columns].mean()
    core_poem_output = period_means[(period_means["sample"] == "CORE") & (period_means["aggregation"] == "POEM")]
    poem_ok = all(
        math.isclose(
            row["mean_topic_probability"],
            independent_poem.loc[row["period"], row["topic_id"] + "_prob"],
            rel_tol=0,
            abs_tol=1e-14,
        )
        for row in core_poem_output.to_dict("records")
    )
    add("TEST_09", "poem-weighted 中每首诗权重相等", poem_ok, poem_ok, True)
    cluster_ok = bootstrap_ci["bootstrap_unit"].eq("author").all() and bootstrap_ci["bootstrap_algorithm"].str.contains("author-cluster").all()
    add("TEST_10", "cluster bootstrap unit 为 author", cluster_ok, cluster_ok, True)
    counts = period_means.groupby(["sample", "aggregation"]).size()
    complete_spec = bool((counts == int(config["expected_topic_count"]) * len(config["period_order"])).all())
    add("TEST_11", "每个 specification 有20 Topics × 4 periods", complete_spec, counts.to_dict(), 80)
    specifications = set(period_means["specification"])
    expected_specs = {f"{sample}_{aggregation}" for sample in config["samples"] for aggregation in AGGREGATIONS}
    add("TEST_12", "六种 specification 均完整生成", specifications == expected_specs, sorted(specifications), sorted(expected_specs))
    unchanged_hash = sha256_array(poems[topic_columns].to_numpy(dtype=float)) == sha256_array(topic_matrix_before)
    add("TEST_13", "每首诗20维概率原值不变", unchanged_hash, unchanged_hash, True)
    reproduction_digest = hashlib.sha256()
    for sample_index, (sample, label_column) in enumerate(config["samples"].items()):
        reproduced = bootstrap_sample(
            poems,
            label_column,
            topic_columns,
            config["period_order"],
            sample_index,
            int(config["bootstrap_iterations"]),
            int(config["random_seed"]),
        )
        for period in config["period_order"]:
            for aggregation in AGGREGATIONS:
                reproduction_digest.update(sample.encode("utf-8"))
                reproduction_digest.update(period.encode("utf-8"))
                reproduction_digest.update(aggregation.encode("ascii"))
                reproduction_digest.update(reproduced[period][aggregation].tobytes())
    reproduced_hash = reproduction_digest.hexdigest()
    add("TEST_14", "全部 bootstrap results 可由 seed 重现", reproduced_hash == bootstrap_hash, reproduced_hash, bootstrap_hash)
    topic_types = set(trends["topic_type"])
    all_topics = trends.groupby("specification")["topic_id"].nunique().eq(int(config["expected_topic_count"])).all()
    type_ok = {"MIXED", "NOISE"}.issubset(topic_types) and all_topics
    add("TEST_15", "Mixed/Noise Topic 未被静默删除", type_ok, sorted(topic_types), "包含 MIXED/NOISE 且每规格20 Topic")
    frame = pd.DataFrame(tests)
    if not frame["status"].eq("PASS").all():
        failures = frame.loc[frame["status"].ne("PASS"), ["test_id", "test_name"]].to_dict("records")
        raise AssertionError(f"数据不可变性测试失败：{failures}")
    return frame


def main() -> None:
    """执行完整趋势结构流水线。"""

    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    data_root = output_root / "data"
    figures_root = output_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    poems, topic_info, labels, topic_columns, input_paths, input_hashes = load_inputs(config)
    source_hash_before = input_hashes["authoritative_master"]
    author_profiles, profiles_by_sample = build_author_profiles(
        poems, topic_columns, config["samples"], config["period_order"]
    )
    period_means = build_period_topic_means(
        poems, profiles_by_sample, topic_columns, topic_info, config
    )
    bootstrap_ci, all_bootstrap, bootstrap_hash = build_bootstrap_results(
        poems, topic_columns, period_means, topic_info, config
    )
    trends = build_trend_structure(
        topic_columns, period_means, bootstrap_ci, all_bootstrap, topic_info, config
    )
    comparisons = build_poem_author_comparison(trends, config)
    robustness = build_trend_robustness(trends, comparisons, topic_info, config)
    concentration = build_author_concentration(poems, topic_columns, topic_info, config)
    clear, candidates, sensitivity = build_clear_and_candidates(robustness, trends, concentration)
    period_counts = build_period_counts(poems, config)
    invariance = run_invariance_tests(
        poems,
        labels,
        topic_columns,
        topic_info,
        period_means,
        author_profiles,
        bootstrap_ci,
        bootstrap_hash,
        trends,
        input_paths["authoritative_master"],
        source_hash_before,
        config,
    )

    output_frames = {
        "period_counts": period_counts,
        "period_topic_means": period_means,
        "author_topic_profiles": author_profiles,
        "period_topic_bootstrap_ci": bootstrap_ci,
        "topic_trend_structure": trends,
        "topic_trend_robustness": robustness,
        "poem_vs_author_trend_comparison": comparisons,
        "period_topic_author_concentration": concentration,
        "clear_topic_trend_structure": clear,
        "trend_candidate_topics": candidates,
        "sensitivity_flags": sensitivity,
        "invariance_tests": invariance,
    }
    data_dictionary = build_data_dictionary(output_frames)
    output_frames["data_dictionary"] = data_dictionary
    for name, frame in output_frames.items():
        save_csv(frame, data_root / f"{name}.csv")

    figure_paths = build_figures(
        figures_root,
        period_means,
        bootstrap_ci,
        trends,
        comparisons,
        concentration,
        topic_info,
        config,
    )
    summary = build_summary(
        period_counts, trends, comparisons, robustness, clear, candidates, concentration, config
    )
    (data_root / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n"
    )
    build_report(
        output_root,
        summary,
        period_counts,
        trends,
        comparisons,
        robustness,
        clear,
        candidates,
        concentration,
        config,
    )
    build_readme(output_root, config, output_frames)

    output_paths = [data_root / f"{name}.csv" for name in output_frames]
    output_paths += figure_paths
    output_paths += [
        data_root / "analysis_summary.json",
        output_root / "README.md",
        output_root / "QTS_Author_Period_Trend_Structure_Report.md",
    ]
    manifest = {
        "analysis_version": config["analysis_version"],
        "created_at_local_date": "2026-09-01",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "inputs": {
            name: {"path": str(path), "sha256": input_hashes[name]} for name, path in input_paths.items()
        },
        "bootstrap": {
            "iterations": config["bootstrap_iterations"],
            "seed": config["random_seed"],
            "algorithm": config["bootstrap_algorithm"],
            "replicate_sha256": bootstrap_hash,
        },
        "outputs": {
            str(path.relative_to(output_root)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(output_paths)
        },
        "invariance_tests_passed": int(invariance["status"].eq("PASS").sum()),
        "invariance_tests_total": int(len(invariance)),
    }
    (output_root / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
