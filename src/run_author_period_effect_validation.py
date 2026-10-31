"""生成 C_K20 作者时期主题效应量与统计验证结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
AGGREGATIONS = ["POEM", "AUTHOR"]
STATUS_ORDER = ["STRONG", "SUPPORTED", "TENTATIVE", "UNSUPPORTED", "CONFLICTING"]
STATUS_COLORS = {
    "STRONG": "#166534",
    "SUPPORTED": "#4D7C0F",
    "TENTATIVE": "#D97706",
    "UNSUPPORTED": "#64748B",
    "CONFLICTING": "#B91C1C",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/effect_validation_config.yaml")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    """相对仓库根目录解析路径。"""

    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    """读取 JSON 兼容 YAML 配置。"""

    return json.loads(path.read_text(encoding="utf-8"))


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
    """以 UTF-8 写出稳定 CSV。"""

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


def bh_adjust(values: np.ndarray) -> np.ndarray:
    """执行 Benjamini-Hochberg FDR 校正。"""

    p_values = np.asarray(values, dtype=float)
    result = np.full_like(p_values, np.nan)
    valid = np.isfinite(p_values)
    if not valid.any():
        return result
    observed = p_values[valid]
    order = np.argsort(observed)
    ranked = observed[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    unsorted = np.empty_like(adjusted)
    unsorted[order] = adjusted
    result[valid] = unsorted
    return result


def direction_from_delta(delta: float, tolerance: float = 1e-12) -> str:
    """按固定数值容差返回效应方向。"""

    if abs(delta) <= tolerance:
        return "NEAR_ZERO"
    return "POSITIVE" if delta > 0 else "NEGATIVE"


def bootstrap_direction(low: float, high: float) -> str:
    """根据效应区间判定支持方向。"""

    if low > 0:
        return "SUPPORTED_POSITIVE"
    if high < 0:
        return "SUPPORTED_NEGATIVE"
    return "UNCERTAIN"


def hedges_label(value: float, thresholds: dict[str, Any]) -> str:
    """按配置阈值标记 Hedges' g 大小。"""

    absolute = abs(value)
    if absolute < thresholds["hedges_small"]:
        return "VERY_SMALL"
    if absolute < thresholds["hedges_moderate"]:
        return "SMALL"
    if absolute < thresholds["hedges_large"]:
        return "MODERATE"
    return "LARGE"


def cliff_label(value: float, thresholds: dict[str, Any]) -> str:
    """按配置阈值标记 Cliff's delta 大小。"""

    absolute = abs(value)
    if absolute < thresholds["cliff_small"]:
        return "NEGLIGIBLE"
    if absolute < thresholds["cliff_medium"]:
        return "SMALL"
    if absolute < thresholds["cliff_large"]:
        return "MEDIUM"
    return "LARGE"


def hedges_g(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """计算 B-A 方向的独立组 Hedges' g。"""

    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return math.nan
    pooled_variance = ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (n_a + n_b - 2)
    if pooled_variance <= 0:
        return 0.0 if math.isclose(float(np.mean(b) - np.mean(a)), 0.0, abs_tol=1e-15) else math.nan
    d_value = float((np.mean(b) - np.mean(a)) / math.sqrt(pooled_variance))
    correction = 1.0 - 3.0 / (4.0 * (n_a + n_b) - 9.0)
    return d_value * correction


def cliffs_delta(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """计算 B-A 方向的 Cliff's delta，平局计半。"""

    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return math.nan
    u_value = float(stats.mannwhitneyu(b, a, alternative="two-sided", method="asymptotic").statistic)
    return 2.0 * u_value / (len(a) * len(b)) - 1.0


def relative_change(mean_a: float, delta: float) -> float:
    """以 A 期均值为分母计算辅助相对变化。"""

    return math.nan if math.isclose(mean_a, 0.0, abs_tol=1e-15) else delta / mean_a


def load_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, Path], dict[str, str]]:
    """读取并核验权威数据、审计标签、taxonomy 与上一阶段输出。"""

    trend_root = resolve_path(config["trend_output_root"])
    paths = {
        "authoritative_master": resolve_path(config["authoritative_master"]),
        "topic_metadata": resolve_path(config["topic_metadata"]),
        "sample_labels": resolve_path(config["sample_labels"]),
        "topic_missingness_risk": resolve_path(config["topic_missingness_risk"]),
        "clear_topic_representativeness": resolve_path(config["clear_topic_representativeness"]),
        "trend_period_means": trend_root / "data" / "period_topic_means.csv",
        "trend_author_profiles": trend_root / "data" / "author_topic_profiles.csv",
        "trend_bootstrap_ci": trend_root / "data" / "period_topic_bootstrap_ci.csv",
        "trend_structure": trend_root / "data" / "topic_trend_structure.csv",
        "trend_robustness": trend_root / "data" / "topic_trend_robustness.csv",
        "trend_poem_author": trend_root / "data" / "poem_vs_author_trend_comparison.csv",
        "trend_concentration": trend_root / "data" / "period_topic_author_concentration.csv",
        "trend_clear": trend_root / "data" / "clear_topic_trend_structure.csv",
        "trend_candidates": trend_root / "data" / "trend_candidate_topics.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少输入文件：{missing}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    master = pd.read_csv(paths["authoritative_master"], low_memory=False)
    labels = pd.read_csv(paths["sample_labels"], low_memory=False)
    metadata = pd.read_csv(paths["topic_metadata"], low_memory=False)
    risk = pd.read_csv(paths["topic_missingness_risk"], low_memory=False)
    clear_representativeness = pd.read_csv(paths["clear_topic_representativeness"], low_memory=False)
    if len(master) != int(config["expected_poem_count"]) or len(labels) != int(config["expected_poem_count"]):
        raise ValueError("权威 master 或审计标签行数不符合预期")
    if master["poem_id"].duplicated().any() or labels["poem_id"].duplicated().any():
        raise ValueError("poem_id 必须唯一")
    topic_columns = sorted(column for column in master if column.startswith("topic_") and column.endswith("_prob"))
    if len(topic_columns) != int(config["expected_topic_count"]):
        raise ValueError("Topic probability 列数不符合预期")
    topic_ids = [column.removesuffix("_prob") for column in topic_columns]
    if set(metadata["topic_id"]) != set(topic_ids) or metadata["topic_id"].duplicated().any():
        raise ValueError("taxonomy Topic 集合与 master 不一致")
    clear_ids = sorted(metadata.loc[metadata["topic_type"].eq("CLEAR"), "topic_id"])
    if len(clear_ids) != int(config["expected_clear_topic_count"]):
        raise ValueError("CLEAR Topic 数量不符合预期")
    if sorted(clear_representativeness["topic_id"]) != clear_ids:
        raise ValueError("CLEAR representativeness 分类与 taxonomy 不一致")
    probability = master[topic_columns].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("Topic probability 含缺失、无穷或负值")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-10, rtol=0):
        raise ValueError("每首诗20维 Topic probability 之和不为1")
    label_columns = ["poem_id", "author_id_internal", *config["samples"].values()]
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
    topic_info = metadata[["topic_id", "final_topic_label", "short_label", "topic_type", "interpretation_scope"]].rename(
        columns={"final_topic_label": "topic_label"}
    )
    risk_info = risk[["topic_id", "risk_level", "risk_reason", "smd", "abs_smd"]].rename(
        columns={
            "risk_level": "representativeness_risk",
            "smd": "known_unknown_smd",
            "abs_smd": "known_unknown_abs_smd",
        }
    )
    topic_info = topic_info.merge(risk_info, on="topic_id", validate="one_to_one").sort_values("topic_id").reset_index(drop=True)
    prior_clear = pd.read_csv(paths["trend_clear"])
    if sorted(prior_clear["topic_id"]) != clear_ids:
        raise ValueError("上一阶段 CLEAR Topic 集合与 taxonomy 不一致")
    prior_labels = prior_clear[["topic_id", "topic_label"]].sort_values("topic_id").reset_index(drop=True)
    current_labels = topic_info.loc[topic_info["topic_type"].eq("CLEAR"), ["topic_id", "topic_label"]].sort_values("topic_id").reset_index(drop=True)
    if not prior_labels.equals(current_labels):
        raise ValueError("上一阶段 CLEAR Topic 标签与 taxonomy 不一致")
    return joined, topic_info, labels, topic_columns, paths, hashes


def build_author_profiles(poems: pd.DataFrame, topic_columns: list[str], config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """按样本建立一行一作者的等权主题 profile。"""

    profiles: dict[str, pd.DataFrame] = {}
    for sample, label_column in config["samples"].items():
        selected = poems.loc[poems[label_column]]
        grouped = selected.groupby(["author_id_internal", "author", "author_period_primary"], sort=True, observed=True)
        profile = grouped[topic_columns].mean().reset_index()
        counts = grouped.size().rename("poem_count_in_sample").reset_index()
        profile = counts.merge(profile, on=["author_id_internal", "author", "author_period_primary"], validate="one_to_one")
        profile = profile.rename(columns={column: column.replace("_prob", "_mean") for column in topic_columns})
        profiles[sample] = profile
    return profiles


def fit_period_model(values: np.ndarray, periods: np.ndarray, groups: np.ndarray | None, period_order: list[str], covariance: str) -> dict[str, Any]:
    """拟合时期 cell-means OLS，并计算 CR1 或 HC3 协方差。"""

    y = np.asarray(values, dtype=float)
    period_values = np.asarray(periods)
    x = np.column_stack([period_values == period for period in period_order]).astype(float)
    n_obs, n_parameters = x.shape
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    if covariance == "CLUSTER_CR1":
        if groups is None:
            raise ValueError("CR1 需要作者 cluster")
        group_values = np.asarray(groups)
        unique_groups = np.unique(group_values)
        meat = np.zeros((n_parameters, n_parameters), dtype=float)
        for group in unique_groups:
            mask = group_values == group
            score = x[mask].T @ residual[mask]
            meat += np.outer(score, score)
        group_count = len(unique_groups)
        factor = (group_count / (group_count - 1)) * ((n_obs - 1) / (n_obs - n_parameters))
        covariance_matrix = factor * xtx_inv @ meat @ xtx_inv
        df_denom = group_count - 1
    elif covariance == "HC3":
        leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
        adjusted = residual / (1.0 - leverage)
        meat = x.T @ ((adjusted**2)[:, None] * x)
        covariance_matrix = xtx_inv @ meat @ xtx_inv
        group_count = n_obs
        df_denom = n_obs - n_parameters
    else:
        raise ValueError(f"未知 covariance：{covariance}")
    contrast_basis = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]], dtype=float)
    joint = contrast_basis @ beta
    joint_covariance = contrast_basis @ covariance_matrix @ contrast_basis.T
    wald_chi2 = float(joint.T @ np.linalg.pinv(joint_covariance) @ joint)
    wald_f = wald_chi2 / contrast_basis.shape[0]
    omnibus_p = float(stats.f.sf(wald_f, contrast_basis.shape[0], df_denom))
    return {
        "beta": beta,
        "covariance": covariance_matrix,
        "n_obs": n_obs,
        "n_parameters": n_parameters,
        "group_count": group_count,
        "df_denom": df_denom,
        "wald_chi2": wald_chi2,
        "wald_f": wald_f,
        "omnibus_df_num": contrast_basis.shape[0],
        "omnibus_p": omnibus_p,
    }


def model_contrast(model: dict[str, Any], period_a: str, period_b: str, period_order: list[str]) -> dict[str, float]:
    """从时期 cell-means 模型计算 B-A planned contrast。"""

    vector = np.zeros(len(period_order), dtype=float)
    vector[period_order.index(period_a)] = -1.0
    vector[period_order.index(period_b)] = 1.0
    estimate = float(vector @ model["beta"])
    variance = float(vector @ model["covariance"] @ vector)
    standard_error = math.sqrt(max(variance, 0.0))
    critical = float(stats.t.ppf(0.975, model["df_denom"]))
    t_value = estimate / standard_error if standard_error > 0 else math.nan
    p_value = float(2 * stats.t.sf(abs(t_value), model["df_denom"])) if math.isfinite(t_value) else math.nan
    return {
        "model_estimate": estimate,
        "model_se": standard_error,
        "model_ci_low": estimate - critical * standard_error,
        "model_ci_high": estimate + critical * standard_error,
        "model_t": t_value,
        "model_df": float(model["df_denom"]),
        "model_p": p_value,
    }


def bootstrap_sample(poems: pd.DataFrame, label_column: str, topic_columns: list[str], config: dict[str, Any], sample_index: int, transformed: np.ndarray | None = None) -> dict[str, dict[str, np.ndarray]]:
    """以时期内作者为 cluster 生成 POEM/AUTHOR bootstrap replicate。"""

    selected_mask = poems[label_column].to_numpy(dtype=bool)
    selected = poems.loc[selected_mask, ["author_id_internal", "author_period_primary"]].copy()
    if transformed is None:
        matrix = poems.loc[selected_mask, topic_columns].to_numpy(dtype=float)
    else:
        matrix = np.asarray(transformed, dtype=float)[selected_mask]
    for index, topic_column in enumerate(topic_columns):
        selected[topic_column] = matrix[:, index]
    result: dict[str, dict[str, np.ndarray]] = {}
    iterations = int(config["bootstrap_iterations"])
    for period_index, period in enumerate(config["period_order"]):
        period_frame = selected.loc[selected["author_period_primary"].eq(period)]
        grouped = period_frame.groupby("author_id_internal", sort=True, observed=True)
        topic_sums = grouped[topic_columns].sum().to_numpy(dtype=float)
        poem_counts = grouped.size().to_numpy(dtype=float)
        author_means = topic_sums / poem_counts[:, None]
        author_count = len(poem_counts)
        random = np.random.default_rng(np.random.SeedSequence([int(config["random_seed"]), sample_index, period_index]))
        counts = random.multinomial(author_count, np.full(author_count, 1.0 / author_count), size=iterations).astype(float)
        result[period] = {
            "POEM": (counts @ topic_sums) / (counts @ poem_counts)[:, None],
            "AUTHOR": (counts @ author_means) / author_count,
            "author_count": np.array([author_count]),
            "poem_count": np.array([len(period_frame)]),
        }
    return result


def build_bootstrap(poems: pd.DataFrame, topic_columns: list[str], config: dict[str, Any]) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], str]:
    """生成六规格 bootstrap，并记录可复现摘要。"""

    all_bootstrap: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    digest = hashlib.sha256()
    for sample_index, (sample, label_column) in enumerate(config["samples"].items()):
        generated = bootstrap_sample(poems, label_column, topic_columns, config, sample_index)
        all_bootstrap[sample] = generated
        for period in config["period_order"]:
            for aggregation in AGGREGATIONS:
                digest.update(sample.encode("utf-8"))
                digest.update(period.encode("utf-8"))
                digest.update(aggregation.encode("ascii"))
                digest.update(generated[period][aggregation].tobytes())
    return all_bootstrap, digest.hexdigest()


def contrast_definitions(config: dict[str, Any]) -> list[dict[str, str]]:
    """返回 primary 与 secondary 对比及其预设层级。"""

    rows: list[dict[str, str]] = []
    for item in config["primary_contrasts"]:
        rows.append({**item, "contrast_role": "PRIMARY"})
    for item in config["secondary_contrasts"]:
        rows.append({**item, "contrast_role": "SECONDARY"})
    return rows


def build_models(
    poems: pd.DataFrame,
    profiles: dict[str, pd.DataFrame],
    topic_columns: list[str],
    topic_info: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    """只对 CLEAR Topics 的两套 CORE estimand 生成正式模型与 omnibus。"""

    core_poems = poems.loc[poems[config["samples"]["CORE"]]]
    core_authors = profiles["CORE"]
    clear_info = topic_info.loc[topic_info["topic_type"].eq("CLEAR")].set_index("topic_id")
    models: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for aggregation in AGGREGATIONS:
        for topic_id, info in clear_info.iterrows():
            if aggregation == "POEM":
                frame = core_poems
                values = frame[f"{topic_id}_prob"].to_numpy(dtype=float)
                model = fit_period_model(
                    values,
                    frame["author_period_primary"].to_numpy(),
                    frame["author_id_internal"].to_numpy(),
                    config["period_order"],
                    "CLUSTER_CR1",
                )
                covariance = "AUTHOR_CLUSTER_CR1"
                unit = "poem"
            else:
                frame = core_authors
                values = frame[f"{topic_id}_mean"].to_numpy(dtype=float)
                model = fit_period_model(
                    values,
                    frame["author_period_primary"].to_numpy(),
                    None,
                    config["period_order"],
                    "HC3",
                )
                covariance = "HC3"
                unit = "author"
            models[(aggregation, topic_id)] = model
            rows.append(
                {
                    "topic_id": topic_id,
                    "topic_label": info["topic_label"],
                    "topic_type": info["topic_type"],
                    "aggregation": aggregation,
                    "specification": f"CORE_{aggregation}",
                    "estimator_unit": unit,
                    "covariance_estimator": covariance,
                    "n_observations": model["n_obs"],
                    "author_cluster_count": model["group_count"],
                    "wald_chi2": model["wald_chi2"],
                    "wald_f": model["wald_f"],
                    "df_num": model["omnibus_df_num"],
                    "df_denom": model["df_denom"],
                    "p_value": model["omnibus_p"],
                    "fdr_family": f"CORE_{aggregation}_9_CLEAR_OMNIBUS",
                }
            )
    omnibus = pd.DataFrame(rows).sort_values(["aggregation", "topic_id"]).reset_index(drop=True)
    omnibus["q_value"] = np.nan
    for aggregation in AGGREGATIONS:
        mask = omnibus["aggregation"].eq(aggregation)
        omnibus.loc[mask, "q_value"] = bh_adjust(omnibus.loc[mask, "p_value"].to_numpy())
    omnibus["fdr_pass"] = omnibus["q_value"].le(float(config["fdr"]["alpha"]))
    return omnibus, models


def period_values(
    poems: pd.DataFrame,
    profiles: dict[str, pd.DataFrame],
    sample: str,
    aggregation: str,
    topic_id: str,
    period: str,
    config: dict[str, Any],
) -> np.ndarray:
    """提取一套 specification 下某时期的 Topic 向量。"""

    if aggregation == "POEM":
        frame = poems.loc[poems[config["samples"][sample]] & poems["author_period_primary"].eq(period)]
        return frame[f"{topic_id}_prob"].to_numpy(dtype=float)
    frame = profiles[sample]
    return frame.loc[frame["author_period_primary"].eq(period), f"{topic_id}_mean"].to_numpy(dtype=float)


def build_all_topic_effect_sizes(
    poems: pd.DataFrame,
    profiles: dict[str, pd.DataFrame],
    topic_columns: list[str],
    topic_info: pd.DataFrame,
    all_bootstrap: dict[str, dict[str, dict[str, np.ndarray]]],
    models: dict[tuple[str, str], dict[str, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    """生成全部20 Topics、六规格与六对比的描述性效应量。"""

    info = topic_info.set_index("topic_id")
    definitions = contrast_definitions(config)
    rows: list[dict[str, Any]] = []
    tolerance = float(config["thresholds"]["numeric_zero_tolerance"])
    for sample in config["samples"]:
        for aggregation in AGGREGATIONS:
            for topic_index, topic_column in enumerate(topic_columns):
                topic_id = topic_column.removesuffix("_prob")
                topic_meta = info.loc[topic_id]
                for contrast_index, contrast in enumerate(definitions, start=1):
                    period_a = contrast["period_a"]
                    period_b = contrast["period_b"]
                    values_a = period_values(poems, profiles, sample, aggregation, topic_id, period_a, config)
                    values_b = period_values(poems, profiles, sample, aggregation, topic_id, period_b, config)
                    mean_a = float(np.mean(values_a))
                    mean_b = float(np.mean(values_b))
                    delta = mean_b - mean_a
                    replicate = (
                        all_bootstrap[sample][period_b][aggregation][:, topic_index]
                        - all_bootstrap[sample][period_a][aggregation][:, topic_index]
                    )
                    low, high = np.quantile(replicate, [0.025, 0.975])
                    bootstrap_se = float(np.std(replicate, ddof=1))
                    author_a = period_values(poems, profiles, sample, "AUTHOR", topic_id, period_a, config)
                    author_b = period_values(poems, profiles, sample, "AUTHOR", topic_id, period_b, config)
                    g_value = hedges_g(author_a, author_b)
                    cliff_value = cliffs_delta(author_a, author_b)
                    if aggregation == "POEM":
                        pooled_author_sd = math.sqrt(
                            ((len(author_a) - 1) * np.var(author_a, ddof=1) + (len(author_b) - 1) * np.var(author_b, ddof=1))
                            / (len(author_a) + len(author_b) - 2)
                        )
                        clustered_smd = delta / pooled_author_sd if pooled_author_sd > 0 else math.nan
                        standardized_effect = clustered_smd
                        standardized_effect_type = "DESCRIPTIVE_CLUSTERED_SMD"
                    else:
                        clustered_smd = math.nan
                        standardized_effect = g_value
                        standardized_effect_type = "HEDGES_G"
                    model_values: dict[str, float] = {
                        "model_estimate": math.nan,
                        "model_se": math.nan,
                        "model_ci_low": math.nan,
                        "model_ci_high": math.nan,
                        "model_t": math.nan,
                        "model_df": math.nan,
                        "model_p": math.nan,
                    }
                    formal = sample == "CORE" and topic_meta["topic_type"] == "CLEAR"
                    if formal:
                        model_values = model_contrast(models[(aggregation, topic_id)], period_a, period_b, config["period_order"])
                    rows.append(
                        {
                            "sample": sample,
                            "aggregation": aggregation,
                            "specification": f"{sample}_{aggregation}",
                            "topic_id": topic_id,
                            "topic_label": topic_meta["topic_label"],
                            "topic_type": topic_meta["topic_type"],
                            "interpretation_scope": topic_meta["interpretation_scope"],
                            "representativeness_risk": topic_meta["representativeness_risk"],
                            "contrast": contrast["code"],
                            "contrast_order": contrast_index,
                            "contrast_role": contrast["contrast_role"],
                            "period_a": period_a,
                            "period_b": period_b,
                            "effect_definition": "period_b_minus_period_a",
                            "mean_a": mean_a,
                            "mean_b": mean_b,
                            "raw_delta": delta,
                            "percentage_point_delta": delta * 100.0,
                            "relative_change": relative_change(mean_a, delta),
                            "raw_direction": direction_from_delta(delta, tolerance),
                            "bootstrap_ci_low": float(low),
                            "bootstrap_ci_high": float(high),
                            "bootstrap_se": bootstrap_se,
                            "bootstrap_direction": bootstrap_direction(float(low), float(high)),
                            "bootstrap_iterations": int(config["bootstrap_iterations"]),
                            "bootstrap_unit": "author",
                            "standardized_effect": standardized_effect,
                            "standardized_effect_type": standardized_effect_type,
                            "descriptive_clustered_smd": clustered_smd,
                            "hedges_g": g_value if aggregation == "AUTHOR" else math.nan,
                            "hedges_magnitude": hedges_label(g_value, config["thresholds"]) if aggregation == "AUTHOR" else "",
                            "cliff_delta": cliff_value if aggregation == "AUTHOR" else math.nan,
                            "cliff_magnitude": cliff_label(cliff_value, config["thresholds"]) if aggregation == "AUTHOR" else "",
                            "n_a": len(values_a),
                            "n_b": len(values_b),
                            "formal_inference_scope": formal,
                            **model_values,
                        }
                    )
    frame = pd.DataFrame(rows).sort_values(
        ["topic_id", "contrast_order", "sample", "aggregation"]
    ).reset_index(drop=True)
    frame["model_q"] = np.nan
    primary = frame["formal_inference_scope"] & frame["contrast_role"].eq("PRIMARY")
    for aggregation in AGGREGATIONS:
        mask = primary & frame["aggregation"].eq(aggregation)
        frame.loc[mask, "model_q"] = bh_adjust(frame.loc[mask, "model_p"].to_numpy())
    frame["model_fdr_pass"] = frame["model_q"].le(float(config["fdr"]["alpha"]))
    frame["fdr_family"] = ""
    for aggregation in AGGREGATIONS:
        mask = primary & frame["aggregation"].eq(aggregation)
        frame.loc[mask, "fdr_family"] = f"CORE_{aggregation}_9_CLEAR_X_4_PRIMARY"
    return frame


def build_clear_pairwise(all_effects: pd.DataFrame) -> pd.DataFrame:
    """整理 CLEAR Topics 的 CORE 六对比正式与描述性输出。"""

    columns = [
        "topic_id", "topic_label", "topic_type", "representativeness_risk", "aggregation",
        "specification", "contrast", "contrast_order", "contrast_role", "period_a", "period_b",
        "mean_a", "mean_b", "raw_delta", "percentage_point_delta", "relative_change", "raw_direction",
        "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_se", "bootstrap_direction",
        "standardized_effect", "standardized_effect_type", "descriptive_clustered_smd", "hedges_g",
        "hedges_magnitude", "cliff_delta", "cliff_magnitude", "model_estimate", "model_se",
        "model_ci_low", "model_ci_high", "model_t", "model_df", "model_p", "model_q",
        "model_fdr_pass", "fdr_family", "n_a", "n_b",
    ]
    return all_effects.loc[
        all_effects["sample"].eq("CORE") & all_effects["topic_type"].eq("CLEAR"), columns
    ].sort_values(["topic_id", "contrast_order", "aggregation"]).reset_index(drop=True)


def clr_transform(probabilities: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, int, float | None]:
    """临时执行 CLR 转换；真零必须按配置显式处理。"""

    matrix = np.asarray(probabilities, dtype=float)
    zero_count = int((matrix == 0).sum())
    pseudocount = config["clr"].get("pseudocount")
    if zero_count and pseudocount is None:
        raise ValueError("Topic probability 存在真零，且配置未授权 pseudocount")
    if zero_count:
        matrix = matrix + float(pseudocount)
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
    log_values = np.log(matrix)
    return log_values - log_values.mean(axis=1, keepdims=True), zero_count, pseudocount


def build_clr_sensitivity(
    poems: pd.DataFrame,
    profiles: dict[str, pd.DataFrame],
    topic_columns: list[str],
    topic_info: pd.DataFrame,
    all_effects: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, str, int, float | None]:
    """比较 CORE raw probability 与临时 CLR 对比方向。"""

    probability = poems[topic_columns].to_numpy(dtype=float)
    clr_matrix, zero_count, pseudocount = clr_transform(probability, config)
    clr_hash = sha256_array(clr_matrix)
    clr_bootstrap = bootstrap_sample(
        poems,
        config["samples"]["CORE"],
        topic_columns,
        config,
        list(config["samples"]).index("CORE"),
        transformed=clr_matrix,
    )
    core_mask = poems[config["samples"]["CORE"]].to_numpy(dtype=bool)
    core_meta = poems.loc[core_mask, ["author_id_internal", "author_period_primary"]].reset_index(drop=True)
    core_clr = pd.DataFrame(clr_matrix[core_mask], columns=topic_columns)
    core_clr = pd.concat([core_meta, core_clr], axis=1)
    clr_profiles = core_clr.groupby(["author_id_internal", "author_period_primary"], sort=True, observed=True)[topic_columns].mean().reset_index()
    clear_info = topic_info.loc[topic_info["topic_type"].eq("CLEAR")].set_index("topic_id")
    raw_lookup = all_effects.loc[
        all_effects["sample"].eq("CORE")
        & all_effects["topic_type"].eq("CLEAR")
        & all_effects["contrast_role"].eq("PRIMARY")
    ].set_index(["aggregation", "topic_id", "contrast"])
    rows: list[dict[str, Any]] = []
    tolerance = float(config["thresholds"]["numeric_zero_tolerance"])
    for aggregation in AGGREGATIONS:
        for topic_index, topic_column in enumerate(topic_columns):
            topic_id = topic_column.removesuffix("_prob")
            if topic_id not in clear_info.index:
                continue
            info = clear_info.loc[topic_id]
            for contrast in config["primary_contrasts"]:
                period_a, period_b = contrast["period_a"], contrast["period_b"]
                source = core_clr if aggregation == "POEM" else clr_profiles
                clr_a = source.loc[source["author_period_primary"].eq(period_a), topic_column].to_numpy(dtype=float)
                clr_b = source.loc[source["author_period_primary"].eq(period_b), topic_column].to_numpy(dtype=float)
                clr_delta = float(np.mean(clr_b) - np.mean(clr_a))
                replicate = clr_bootstrap[period_b][aggregation][:, topic_index] - clr_bootstrap[period_a][aggregation][:, topic_index]
                low, high = np.quantile(replicate, [0.025, 0.975])
                raw = raw_lookup.loc[(aggregation, topic_id, contrast["code"])]
                raw_direction = raw["raw_direction"]
                transformed_direction = direction_from_delta(clr_delta, tolerance)
                agreement = raw_direction == transformed_direction
                rows.append(
                    {
                        "topic_id": topic_id,
                        "topic_label": info["topic_label"],
                        "aggregation": aggregation,
                        "specification": f"CORE_{aggregation}",
                        "contrast": contrast["code"],
                        "period_a": period_a,
                        "period_b": period_b,
                        "raw_delta": raw["raw_delta"],
                        "raw_direction": raw_direction,
                        "clr_delta": clr_delta,
                        "clr_direction": transformed_direction,
                        "clr_ci_low": float(low),
                        "clr_ci_high": float(high),
                        "clr_bootstrap_direction": bootstrap_direction(float(low), float(high)),
                        "direction_agreement": agreement,
                        "zero_count_full_matrix": zero_count,
                        "pseudocount": pseudocount,
                        "clr_scope": config["clr"]["scope"],
                    }
                )
    return pd.DataFrame(rows).sort_values(["topic_id", "contrast", "aggregation"]).reset_index(drop=True), clr_hash, zero_count, pseudocount


def magnitude_gap(values: list[float], absolute_floor: float, relative_threshold: float) -> bool:
    """判断效应量跨度是否达到配置中的敏感性阈值。"""

    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if len(finite) < 2:
        return False
    observed_range = float(finite.max() - finite.min())
    scale = float(np.median(np.abs(finite)))
    return observed_range > max(absolute_floor, relative_threshold * scale)


def build_sensitivity_matrix(all_effects: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """汇总 CLEAR primary contrasts 的六规格符号、区间与样本敏感性。"""

    frame = all_effects.loc[
        all_effects["topic_type"].eq("CLEAR") & all_effects["contrast_role"].eq("PRIMARY")
    ]
    rows: list[dict[str, Any]] = []
    tolerance = float(config["thresholds"]["numeric_zero_tolerance"])
    thresholds = config["thresholds"]
    for (topic_id, topic_label, contrast), group in frame.groupby(["topic_id", "topic_label", "contrast"], sort=True):
        raw_directions = group["raw_direction"].tolist()
        positive_count = raw_directions.count("POSITIVE")
        negative_count = raw_directions.count("NEGATIVE")
        uncertain_count = raw_directions.count("NEAR_ZERO")
        bootstrap_directions = group["bootstrap_direction"].tolist()
        bootstrap_positive_count = bootstrap_directions.count("SUPPORTED_POSITIVE")
        bootstrap_negative_count = bootstrap_directions.count("SUPPORTED_NEGATIVE")
        bootstrap_uncertain_count = bootstrap_directions.count("UNCERTAIN")
        reversal_specs: list[str] = []
        sample_sign_reversal = False
        reversal_aggregation_count = 0
        magnitude_sensitive = False
        aggregation_notes: list[str] = []
        for aggregation in AGGREGATIONS:
            subset = group.loc[group["aggregation"].eq(aggregation)].sort_values("sample")
            signs = set(subset["raw_direction"]) - {"NEAR_ZERO"}
            if {"POSITIVE", "NEGATIVE"}.issubset(signs):
                sample_sign_reversal = True
                reversal_aggregation_count += 1
                reversal_specs.extend(subset["specification"].tolist())
                aggregation_notes.append(f"{aggregation}:SIGN_REVERSAL")
            if magnitude_gap(
                subset["raw_delta"].tolist(),
                float(thresholds["sample_magnitude_gap_absolute"]),
                float(thresholds["sample_magnitude_gap_relative"]),
            ):
                magnitude_sensitive = True
                aggregation_notes.append(f"{aggregation}:MAGNITUDE_SENSITIVE")
        if positive_count > negative_count and positive_count > uncertain_count:
            dominant = "POSITIVE"
            dominant_count = positive_count
        elif negative_count > positive_count and negative_count > uncertain_count:
            dominant = "NEGATIVE"
            dominant_count = negative_count
        else:
            dominant = "UNCERTAIN"
            dominant_count = max(positive_count, negative_count, uncertain_count)
        if sample_sign_reversal:
            robustness = "LOW"
        elif dominant_count >= 5 and not magnitude_sensitive:
            robustness = "HIGH"
        else:
            robustness = "MODERATE"
        row: dict[str, Any] = {
            "topic_id": topic_id,
            "topic_label": topic_label,
            "contrast": contrast,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "uncertain_count": uncertain_count,
            "bootstrap_positive_count": bootstrap_positive_count,
            "bootstrap_negative_count": bootstrap_negative_count,
            "bootstrap_uncertain_count": bootstrap_uncertain_count,
            "dominant_direction": dominant,
            "dominant_direction_count": dominant_count,
            "sample_sign_reversal": sample_sign_reversal,
            "systematic_sample_sign_reversal": reversal_aggregation_count == len(AGGREGATIONS),
            "sign_reversal_aggregation_count": reversal_aggregation_count,
            "sign_reversal_specifications": ";".join(sorted(set(reversal_specs))),
            "magnitude_sensitive": magnitude_sensitive,
            "sample_robustness": robustness,
            "sensitivity_reason": ";".join(aggregation_notes) if aggregation_notes else "NO_SIGN_REVERSAL",
        }
        for record in group.to_dict("records"):
            prefix = record["specification"].lower()
            row[f"{prefix}_delta"] = record["raw_delta"]
            row[f"{prefix}_ci_low"] = record["bootstrap_ci_low"]
            row[f"{prefix}_ci_high"] = record["bootstrap_ci_high"]
            row[f"{prefix}_direction"] = record["bootstrap_direction"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["topic_id", "contrast"]).reset_index(drop=True)


def aggregation_agreement(poem: pd.Series, author: pd.Series, config: dict[str, Any]) -> tuple[str, str, bool]:
    """按预设规则判定 CORE POEM/AUTHOR 聚合一致度。"""

    poem_direction = poem["raw_direction"]
    author_direction = author["raw_direction"]
    if poem_direction != author_direction and "NEAR_ZERO" not in {poem_direction, author_direction}:
        return "LOW", "CORE_POEM 与 CORE_AUTHOR 原始效应方向相反", True
    values = [float(poem["raw_delta"]), float(author["raw_delta"])]
    gap = abs(values[0] - values[1]) > max(
        float(config["thresholds"]["aggregation_magnitude_gap_absolute"]),
        float(config["thresholds"]["aggregation_magnitude_gap_relative"]) * max(abs(values[0]), abs(values[1])),
    )
    supported_same = (
        poem["bootstrap_direction"] == author["bootstrap_direction"]
        and poem["bootstrap_direction"] in {"SUPPORTED_POSITIVE", "SUPPORTED_NEGATIVE"}
    )
    if poem_direction == author_direction and supported_same and not gap:
        return "HIGH", "同号、bootstrap 支持方向一致且无巨大幅度差", False
    return "MODERATE", "同号但至少一项区间跨零，或存在配置定义的幅度差", False


def build_primary_validation(
    all_effects: pd.DataFrame,
    sensitivity: pd.DataFrame,
    clr: pd.DataFrame,
    topic_info: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """合并两种 CORE estimand，并按多维证据给出验证状态。"""

    core = all_effects.loc[
        all_effects["sample"].eq("CORE")
        & all_effects["topic_type"].eq("CLEAR")
        & all_effects["contrast_role"].eq("PRIMARY")
    ]
    info = topic_info.set_index("topic_id")
    sensitivity_lookup = sensitivity.set_index(["topic_id", "contrast"])
    clr_grouped = clr.groupby(["topic_id", "contrast"], sort=False)
    rows: list[dict[str, Any]] = []
    for (topic_id, contrast), group in core.groupby(["topic_id", "contrast"], sort=True):
        poem = group.loc[group["aggregation"].eq("POEM")].iloc[0]
        author = group.loc[group["aggregation"].eq("AUTHOR")].iloc[0]
        sample_row = sensitivity_lookup.loc[(topic_id, contrast)]
        clr_rows = clr_grouped.get_group((topic_id, contrast))
        clr_agreement = bool(clr_rows["direction_agreement"].all())
        agreement, agreement_reason, aggregation_conflict = aggregation_agreement(poem, author, config)
        g_value = float(author["hedges_g"])
        cliff_value = float(author["cliff_delta"])
        nontrivial = (
            abs(g_value) >= float(config["thresholds"]["hedges_small"])
            or abs(cliff_value) >= float(config["thresholds"]["cliff_small"])
        )
        same_raw = poem["raw_direction"] == author["raw_direction"] and poem["raw_direction"] != "NEAR_ZERO"
        supported_directions = {poem["bootstrap_direction"], author["bootstrap_direction"]}
        both_bootstrap_supported_same = (
            len(supported_directions) == 1
            and next(iter(supported_directions)) in {"SUPPORTED_POSITIVE", "SUPPORTED_NEGATIVE"}
        )
        at_least_one_bootstrap = any(value != "UNCERTAIN" for value in supported_directions)
        opposite_bootstrap = supported_directions == {"SUPPORTED_POSITIVE", "SUPPORTED_NEGATIVE"}
        q_pass = bool(poem["model_fdr_pass"] or author["model_fdr_pass"])
        risk = info.loc[topic_id, "representativeness_risk"]
        sample_reversal = bool(sample_row["sample_sign_reversal"])
        systematic_sample_reversal = bool(sample_row["systematic_sample_sign_reversal"])
        reasons: list[str] = []
        if aggregation_conflict or systematic_sample_reversal or not clr_agreement or opposite_bootstrap:
            status = "CONFLICTING"
            if aggregation_conflict:
                reasons.append("CORE_POEM 与 CORE_AUTHOR 方向冲突")
            if systematic_sample_reversal:
                reasons.append("STRICT/CORE/EXPANDED 在 POEM 与 AUTHOR 两种聚合中均出现符号反转")
            if not clr_agreement:
                reasons.append("raw 与 CLR 至少一套 CORE estimand 方向相反")
            if opposite_bootstrap:
                reasons.append("两种聚合的 bootstrap 支持方向相反")
        elif (
            same_raw
            and both_bootstrap_supported_same
            and nontrivial
            and q_pass
            and agreement == "HIGH"
            and risk in config["thresholds"]["strong_allowed_representativeness_risk"]
            and not sample_reversal
        ):
            status = "STRONG"
            reasons.append("两种 CORE estimand 同号且 bootstrap 同向支持")
            reasons.append("至少一个 planned contrast 通过相应 FDR family")
            reasons.append("作者等权标准化效应达到预设非微小阈值")
        elif (
            same_raw
            and nontrivial
            and at_least_one_bootstrap
            and not opposite_bootstrap
            and agreement in {"HIGH", "MODERATE"}
            and risk in config["thresholds"]["supported_allowed_representativeness_risk"]
            and not sample_reversal
        ):
            status = "SUPPORTED"
            reasons.append("两种 CORE estimand 同号且至少一项 bootstrap 支持该方向")
            reasons.append("作者等权标准化效应达到预设非微小阈值")
            if not q_pass:
                reasons.append("planned contrast FDR 未通过，结论依赖效应量与区间")
        elif same_raw and not nontrivial and not q_pass and not at_least_one_bootstrap:
            status = "UNSUPPORTED"
            reasons.append("两种 CORE estimand 虽同号，但区间均跨零且标准化效应很小")
        else:
            status = "TENTATIVE"
            reasons.append("证据方向占多数，但效应量、区间、FDR 或敏感性证据不完整")
        if risk == "MODERATE":
            reasons.append("metadata representativeness risk 为 MODERATE")
        if bool(sample_row["magnitude_sensitive"]):
            reasons.append("样本定义改变时幅度达到敏感阈值")
        rows.append(
            {
                "topic_id": topic_id,
                "topic_label": info.loc[topic_id, "topic_label"],
                "contrast": contrast,
                "period_a": poem["period_a"],
                "period_b": poem["period_b"],
                "direction": poem["raw_direction"] if same_raw else "CONFLICTING",
                "raw_effect_poem": poem["raw_delta"],
                "poem_ci_low": poem["bootstrap_ci_low"],
                "poem_ci_high": poem["bootstrap_ci_high"],
                "raw_effect_author": author["raw_delta"],
                "author_ci_low": author["bootstrap_ci_low"],
                "author_ci_high": author["bootstrap_ci_high"],
                "effect_size_author": g_value,
                "hedges_g": g_value,
                "hedges_magnitude": author["hedges_magnitude"],
                "cliff_delta": cliff_value,
                "cliff_magnitude": author["cliff_magnitude"],
                "descriptive_clustered_smd": poem["descriptive_clustered_smd"],
                "bootstrap_support_poem": poem["bootstrap_direction"],
                "bootstrap_support_author": author["bootstrap_direction"],
                "q_poem": poem["model_q"],
                "q_author": author["model_q"],
                "p_poem": poem["model_p"],
                "p_author": author["model_p"],
                "poem_model_se": poem["model_se"],
                "poem_model_ci_low": poem["model_ci_low"],
                "poem_model_ci_high": poem["model_ci_high"],
                "author_model_se": author["model_se"],
                "author_model_ci_low": author["model_ci_low"],
                "author_model_ci_high": author["model_ci_high"],
                "aggregation_agreement": agreement,
                "aggregation_agreement_reason": agreement_reason,
                "aggregation_conflict": aggregation_conflict,
                "sample_robustness": sample_row["sample_robustness"],
                "sample_sign_reversal": sample_reversal,
                "systematic_sample_sign_reversal": systematic_sample_reversal,
                "sample_magnitude_sensitive": bool(sample_row["magnitude_sensitive"]),
                "representativeness_risk": risk,
                "clr_agreement": clr_agreement,
                "nontrivial_standardized_effect": nontrivial,
                "validation_status": status,
                "validation_reason": "; ".join(reasons),
            }
        )
    contrast_order = {item["code"]: index for index, item in enumerate(config["primary_contrasts"], start=1)}
    result = pd.DataFrame(rows)
    result["contrast_order"] = result["contrast"].map(contrast_order)
    return result.sort_values(["topic_id", "contrast_order"]).reset_index(drop=True)


def build_peak_trough_effects(
    all_effects: pd.DataFrame,
    paths: dict[str, Path],
    config: dict[str, Any],
) -> pd.DataFrame:
    """按上一阶段 CORE peak/trough 生成 post-hoc 描述性比较。"""

    trends = pd.read_csv(paths["trend_structure"])
    rows: list[dict[str, Any]] = []
    for aggregation in AGGREGATIONS:
        subset = trends.loc[
            trends["sample"].eq("CORE")
            & trends["aggregation"].eq(aggregation)
            & trends["topic_type"].eq("CLEAR")
        ]
        for record in subset.to_dict("records"):
            topic_id = record["topic_id"]
            peak = record["peak_period"]
            trough = record["trough_period"]
            candidates = all_effects.loc[
                all_effects["sample"].eq("CORE")
                & all_effects["aggregation"].eq(aggregation)
                & all_effects["topic_id"].eq(topic_id)
                & (
                    (all_effects["period_a"].eq(trough) & all_effects["period_b"].eq(peak))
                    | (all_effects["period_a"].eq(peak) & all_effects["period_b"].eq(trough))
                )
            ]
            if candidates.empty:
                continue
            effect = candidates.iloc[0]
            sign = 1.0 if effect["period_a"] == trough else -1.0
            rows.append(
                {
                    "topic_id": topic_id,
                    "topic_label": record["topic_label"],
                    "aggregation": aggregation,
                    "peak_period": peak,
                    "trough_period": trough,
                    "peak_minus_trough_delta": sign * effect["raw_delta"],
                    "ci_low": sign * effect["bootstrap_ci_low"] if sign > 0 else -effect["bootstrap_ci_high"],
                    "ci_high": sign * effect["bootstrap_ci_high"] if sign > 0 else -effect["bootstrap_ci_low"],
                    "hedges_g": sign * effect["hedges_g"] if aggregation == "AUTHOR" else math.nan,
                    "cliff_delta": sign * effect["cliff_delta"] if aggregation == "AUTHOR" else math.nan,
                    "analysis_role": "POST_HOC_DESCRIPTIVE",
                    "primary_fdr_included": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["topic_id", "aggregation"]).reset_index(drop=True)


def build_author_queue(
    validation: pd.DataFrame,
    paths: dict[str, Path],
    config: dict[str, Any],
) -> pd.DataFrame:
    """建立下一阶段 Author Influence / LOAO 检查队列。"""

    concentration = pd.read_csv(paths["trend_concentration"])
    clear_trend = pd.read_csv(paths["trend_clear"]).set_index("topic_id")
    rows: list[dict[str, Any]] = []
    for record in validation.to_dict("records"):
        periods = [record["period_a"], record["period_b"]]
        subset = concentration.loc[
            concentration["sample"].eq("CORE")
            & concentration["topic_id"].eq(record["topic_id"])
            & concentration["period"].isin(periods)
        ]
        top1 = float(subset["top1_author_share"].max())
        top5 = float(subset["top5_author_share"].max())
        concentration_high = subset["concentration_level"].eq("HIGH").any()
        prior_sensitive = bool(clear_trend.loc[record["topic_id"], "aggregation_sensitive"])
        reasons: list[str] = []
        if concentration_high:
            reasons.append("CORE 对比时期作者集中度为 HIGH")
        if record["aggregation_conflict"]:
            reasons.append("CORE_POEM 与 CORE_AUTHOR 方向冲突")
        if record["topic_id"] == "topic_16":
            reasons.append("Topic 16 为预设 aggregation-sensitive 专项")
        if prior_sensitive:
            reasons.append("上一阶段标记 aggregation_sensitive")
        if record["validation_status"] in {"STRONG", "SUPPORTED"} and top5 >= 0.40:
            reasons.append("较高验证状态同时伴随较高 top5 author share")
        if not reasons:
            continue
        if concentration_high and record["validation_status"] in {"STRONG", "SUPPORTED"}:
            priority = 1
        elif record["aggregation_conflict"]:
            priority = 1
        elif record["topic_id"] == "topic_16" or prior_sensitive:
            priority = 2
        else:
            priority = 3
        rows.append(
            {
                "priority": priority,
                "topic_id": record["topic_id"],
                "topic_label": record["topic_label"],
                "contrast": record["contrast"],
                "reason": "; ".join(reasons),
                "top1_author_share": top1,
                "top5_author_share": top5,
                "poem_delta": record["raw_effect_poem"],
                "author_delta": record["raw_effect_author"],
                "validation_status": record["validation_status"],
            }
        )
    return pd.DataFrame(rows).sort_values(["priority", "topic_id", "contrast"]).reset_index(drop=True)


def build_supported_candidates(validation: pd.DataFrame) -> pd.DataFrame:
    """仅保留达到 STRONG/SUPPORTED 的 CLEAR planned contrasts。"""

    columns = [
        "topic_id", "topic_label", "contrast", "direction", "raw_effect_poem", "raw_effect_author",
        "effect_size_author", "hedges_g", "cliff_delta", "bootstrap_support_poem",
        "bootstrap_support_author", "q_poem", "q_author", "aggregation_agreement",
        "sample_robustness", "representativeness_risk", "clr_agreement", "validation_status",
        "validation_reason",
    ]
    return validation.loc[validation["validation_status"].isin({"STRONG", "SUPPORTED"}), columns].reset_index(drop=True)


def build_clear_topic_summary(validation: pd.DataFrame) -> pd.DataFrame:
    """按 Topic 汇总四个 planned contrasts 的验证状态。"""

    rows: list[dict[str, Any]] = []
    for (topic_id, topic_label), group in validation.groupby(["topic_id", "topic_label"], sort=True):
        counts = group["validation_status"].value_counts().to_dict()
        row: dict[str, Any] = {
            "topic_id": topic_id,
            "topic_label": topic_label,
            "topic_type": "CLEAR",
            "representativeness_risk": group["representativeness_risk"].iloc[0],
            "all_primary_poem_author_sign_consistent": not bool(group["aggregation_conflict"].any()),
            "aggregation_conflict_count": int(group["aggregation_conflict"].sum()),
            "sample_sign_reversal_count": int(group["sample_sign_reversal"].sum()),
            "clr_direction_disagreement_count": int((~group["clr_agreement"]).sum()),
        }
        for status in STATUS_ORDER:
            row[f"{status.lower()}_count"] = int(counts.get(status, 0))
        for record in group.to_dict("records"):
            row[f"{record['contrast'].replace('-', '').lower()}_status"] = record["validation_status"]
        rows.append(row)
    return pd.DataFrame(rows)


def run_invariance_tests(
    poems: pd.DataFrame,
    labels: pd.DataFrame,
    topic_columns: list[str],
    topic_info: pd.DataFrame,
    profiles: dict[str, pd.DataFrame],
    all_effects: pd.DataFrame,
    clr_hash: str,
    clr_zero_count: int,
    bootstrap_hash: str,
    paths: dict[str, Path],
    source_hash_before: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """执行任务指定的15项数据不可变性测试。"""

    tests: list[dict[str, Any]] = []

    def add(test_id: str, name: str, passed: bool, observed: Any, expected: Any) -> None:
        tests.append(
            {
                "test_id": test_id,
                "test_name": name,
                "status": "PASS" if bool(passed) else "FAIL",
                "observed": json.dumps(observed, ensure_ascii=False, default=str),
                "expected": json.dumps(expected, ensure_ascii=False, default=str),
            }
        )

    probability = poems[topic_columns].to_numpy(dtype=float)
    add("TEST_01", "43,015 authoritative rows 不变", len(poems) == int(config["expected_poem_count"]), len(poems), config["expected_poem_count"])
    source_hash_after = sha256_file(paths["authoritative_master"])
    add("TEST_02", "不覆盖 authoritative master", source_hash_after == source_hash_before, source_hash_after, source_hash_before)
    add(
        "TEST_03",
        "20 Topic probability 不变",
        len(topic_columns) == 20 and np.allclose(probability.sum(axis=1), 1.0, atol=1e-10, rtol=0),
        {"topic_count": len(topic_columns), "matrix_sha256": sha256_array(probability)},
        {"topic_count": 20, "row_sum": 1.0},
    )
    period_ok = True
    for sample, label_column in config["samples"].items():
        source_ids = set(labels.loc[normalize_boolean(labels[label_column], label_column), "poem_id"])
        observed_ids = set(poems.loc[poems[label_column], "poem_id"])
        period_ok &= source_ids == observed_ids
    add("TEST_04", "period metadata 与审计标签不变", period_ok, period_ok, True)
    clear_ids = sorted(topic_info.loc[topic_info["topic_type"].eq("CLEAR"), "topic_id"])
    prior_clear = sorted(pd.read_csv(paths["trend_clear"])["topic_id"])
    add("TEST_05", "CLEAR Topic classification 不变", clear_ids == prior_clear and len(clear_ids) == 9, clear_ids, prior_clear)
    observed_contrasts = all_effects.loc[all_effects["contrast_role"].eq("PRIMARY"), "contrast"].drop_duplicates().tolist()
    expected_contrasts = [item["code"] for item in config["primary_contrasts"]]
    add("TEST_06", "primary contrasts 固定", observed_contrasts == expected_contrasts, observed_contrasts, expected_contrasts)
    poem_formal = all_effects.loc[all_effects["formal_inference_scope"] & all_effects["aggregation"].eq("POEM")]
    add("TEST_07", "CORE_POEM cluster SE groups 为 author", poem_formal["model_df"].nunique() == 1 and int(poem_formal["model_df"].iloc[0]) == profiles["CORE"]["author_id_internal"].nunique() - 1, int(poem_formal["model_df"].iloc[0]), profiles["CORE"]["author_id_internal"].nunique() - 1)
    add("TEST_08", "CORE_AUTHOR 一行一作者", not profiles["CORE"]["author_id_internal"].duplicated().any(), profiles["CORE"]["author_id_internal"].nunique(), len(profiles["CORE"]))
    add("TEST_09", "bootstrap unit 为 author", all_effects["bootstrap_unit"].eq("author").all(), sorted(all_effects["bootstrap_unit"].unique()), ["author"])
    family_counts = all_effects.loc[all_effects["fdr_family"].ne("")].groupby("fdr_family").size().to_dict()
    add("TEST_10", "BH-FDR family 正确", set(family_counts.values()) == {36} and len(family_counts) == 2, family_counts, "两套各36个 planned contrasts")
    specs = sorted(all_effects["specification"].unique())
    expected_specs = sorted(f"{sample}_{aggregation}" for sample in config["samples"] for aggregation in AGGREGATIONS)
    add("TEST_11", "STRICT/CORE/EXPANDED 六规格完整", specs == expected_specs, specs, expected_specs)
    add("TEST_12", "raw vs CLR 只做临时分析", clr_hash != sha256_array(probability) and sha256_file(paths["authoritative_master"]) == source_hash_before, {"clr_sha256": clr_hash, "zero_count": clr_zero_count}, "CLR 独立临时矩阵且 master 未改")
    type_counts = all_effects.groupby("topic_type")["topic_id"].nunique().to_dict()
    add("TEST_13", "MIXED/NOISE 未静默删除", {"MIXED", "NOISE"}.issubset(type_counts), type_counts, "包含 MIXED 与 NOISE")
    reproduced, reproduced_hash = build_bootstrap(poems, topic_columns, config)
    del reproduced
    add("TEST_14", "所有随机过程可由 seed 重现", reproduced_hash == bootstrap_hash, reproduced_hash, bootstrap_hash)
    primary = all_effects.loc[all_effects["formal_inference_scope"] & all_effects["contrast_role"].eq("PRIMARY")]
    complete = primary[["raw_delta", "bootstrap_ci_low", "bootstrap_ci_high", "standardized_effect"]].notna().all().all()
    add("TEST_15", "所有 primary result 同时报 effect size 与 CI", complete and len(primary) == 72, len(primary), 72)
    result = pd.DataFrame(tests)
    if not result["status"].eq("PASS").all():
        failed = result.loc[result["status"].ne("PASS"), ["test_id", "test_name"]].to_dict("records")
        raise AssertionError(f"数据不可变性测试失败：{failed}")
    return result


def build_data_dictionary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """为正式 CSV 建立字段级数据字典。"""

    descriptions = {
        "raw_delta": "主要效应：period_b mean - period_a mean。",
        "bootstrap_ci_low": "作者 cluster bootstrap 95% 区间下限。",
        "bootstrap_ci_high": "作者 cluster bootstrap 95% 区间上限。",
        "hedges_g": "作者等权分布的 Hedges' g，方向为 B-A。",
        "cliff_delta": "作者等权分布的 Cliff's delta，方向为 B-A。",
        "model_p": "指定稳健协方差下 planned contrast 双侧 p 值。",
        "model_q": "对应36项 planned contrast family 的 BH-FDR q 值。",
        "validation_status": "多证据综合状态，不是文学史确认。",
        "validation_reason": "触发验证状态的可审计规则说明。",
        "direction_agreement": "raw probability 与 CLR 对比方向是否一致。",
    }
    rows: list[dict[str, str]] = []
    for file_name, frame in frames.items():
        for column in frame.columns:
            rows.append(
                {
                    "file": f"{file_name}.csv",
                    "field": column,
                    "dtype": str(frame[column].dtype),
                    "description": descriptions.get(column, "见报告方法与该表字段名；数值均为本流水线派生结果。"),
                }
            )
    return pd.DataFrame(rows)


def configure_plotting() -> None:
    """设置论文图所需的统一字体与导出参数。"""

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def build_figures(
    figures_root: Path,
    validation: pd.DataFrame,
    sensitivity: pd.DataFrame,
    clr: pd.DataFrame,
    config: dict[str, Any],
) -> list[Path]:
    """生成 forest、聚合对照、六规格、验证状态与 CLR 图。"""

    configure_plotting()
    figures_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    clear_order = sorted(validation["topic_id"].unique())
    label_lookup = validation.drop_duplicates("topic_id").set_index("topic_id")["topic_label"].to_dict()
    contrast_codes = [item["code"] for item in config["primary_contrasts"]]

    for contrast in contrast_codes:
        subset = validation.loc[validation["contrast"].eq(contrast)].set_index("topic_id").loc[clear_order]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2), sharey=True)
        for axis, aggregation, color in zip(axes, ["POEM", "AUTHOR"], ["#2563EB", "#D97706"]):
            values = subset[f"raw_effect_{aggregation.lower()}"]
            lows = subset[f"{aggregation.lower()}_ci_low"]
            highs = subset[f"{aggregation.lower()}_ci_high"]
            y = np.arange(len(subset))
            axis.errorbar(values, y, xerr=[values - lows, highs - values], fmt="o", color=color, ecolor=color, capsize=3)
            axis.axvline(0, color="#475569", linewidth=0.8)
            axis.set_title(f"CORE_{aggregation}")
            axis.set_xlabel("Raw probability Δ (B−A)")
            axis.grid(axis="x", color="#E2E8F0", linewidth=0.6)
        axes[0].set_yticks(np.arange(len(subset)), [f"{topic.replace('topic_', 'T')} {label_lookup[topic]}" for topic in clear_order])
        fig.suptitle(f"CLEAR Topics planned contrast {contrast}: author-cluster bootstrap 95% CI")
        fig.tight_layout()
        code = contrast.replace("-", "")
        path = figures_root / f"effect_forest_{code}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharey=True)
    for axis, contrast in zip(axes.ravel(), contrast_codes):
        subset = validation.loc[validation["contrast"].eq(contrast)].set_index("topic_id").loc[clear_order]
        y = np.arange(len(subset))
        axis.errorbar(subset["raw_effect_poem"], y - 0.12, xerr=[subset["raw_effect_poem"] - subset["poem_ci_low"], subset["poem_ci_high"] - subset["raw_effect_poem"]], fmt="o", color="#2563EB", label="CORE_POEM", capsize=2)
        axis.errorbar(subset["raw_effect_author"], y + 0.12, xerr=[subset["raw_effect_author"] - subset["author_ci_low"], subset["author_ci_high"] - subset["raw_effect_author"]], fmt="s", color="#D97706", label="CORE_AUTHOR", capsize=2)
        axis.axvline(0, color="#475569", linewidth=0.8)
        axis.set_title(contrast)
        axis.grid(axis="x", color="#E2E8F0", linewidth=0.5)
        axis.set_yticks(y, [topic.replace("topic_", "T") for topic in clear_order])
    axes[0, 0].legend(frameon=False)
    fig.suptitle("CLEAR Topic primary effects: CORE_POEM vs CORE_AUTHOR")
    fig.tight_layout()
    combined_forest = figures_root / "clear_topic_primary_effect_forest.png"
    fig.savefig(combined_forest, bbox_inches="tight")
    plt.close(fig)
    paths.append(combined_forest)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5))
    for axis, contrast in zip(axes.ravel(), contrast_codes):
        subset = validation.loc[validation["contrast"].eq(contrast)]
        for status, status_group in subset.groupby("validation_status"):
            axis.scatter(status_group["raw_effect_poem"], status_group["raw_effect_author"], label=status, color=STATUS_COLORS[status], s=42, alpha=0.85)
        limits = np.array([subset["raw_effect_poem"], subset["raw_effect_author"]])
        bound = max(abs(limits.min()), abs(limits.max())) * 1.12
        axis.plot([-bound, bound], [-bound, bound], color="#94A3B8", linestyle="--", linewidth=0.8)
        axis.axhline(0, color="#CBD5E1", linewidth=0.6)
        axis.axvline(0, color="#CBD5E1", linewidth=0.6)
        axis.set_title(contrast)
        axis.set_xlabel("CORE_POEM Δ")
        axis.set_ylabel("CORE_AUTHOR Δ")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=max(1, len(labels)), frameon=False)
    fig.suptitle("Poem-weighted vs author-balanced primary effects", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    path = figures_root / "clear_topic_poem_vs_author_effect.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    specs = [f"{sample}_{aggregation}" for sample in config["samples"] for aggregation in AGGREGATIONS]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)
    for axis, contrast in zip(axes.ravel(), contrast_codes):
        subset = sensitivity.loc[sensitivity["contrast"].eq(contrast)].set_index("topic_id").loc[clear_order]
        matrix = np.array([[subset.loc[topic, f"{spec.lower()}_delta"] for spec in specs] for topic in clear_order])
        bound = max(abs(float(matrix.min())), abs(float(matrix.max())))
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        axis.set_title(contrast)
        axis.set_xticks(np.arange(len(specs)), specs, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(np.arange(len(clear_order)), [topic.replace("topic_", "T") for topic in clear_order])
        fig.colorbar(image, ax=axis, shrink=0.72, label="Raw Δ")
    fig.suptitle("CLEAR Topics six-specification effect sensitivity")
    path = figures_root / "clear_topic_six_spec_effect_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    status_codes = {status: index for index, status in enumerate(STATUS_ORDER)}
    matrix = np.array(
        [
            [status_codes[validation.loc[(validation["topic_id"].eq(topic)) & (validation["contrast"].eq(contrast)), "validation_status"].iloc[0]] for contrast in contrast_codes]
            for topic in clear_order
        ]
    )
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap([STATUS_COLORS[status] for status in STATUS_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, len(STATUS_ORDER) + 0.5), cmap.N)
    fig, axis = plt.subplots(figsize=(7.8, 6.2))
    image = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    axis.set_xticks(np.arange(len(contrast_codes)), contrast_codes)
    axis.set_yticks(np.arange(len(clear_order)), [f"{topic.replace('topic_', 'T')} {label_lookup[topic]}" for topic in clear_order])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, STATUS_ORDER[matrix[row, column]][:4], ha="center", va="center", color="white", fontsize=8)
    colorbar = fig.colorbar(image, ax=axis, ticks=np.arange(len(STATUS_ORDER)), shrink=0.8)
    colorbar.ax.set_yticklabels(STATUS_ORDER)
    axis.set_title("CLEAR Topic validation status matrix")
    fig.tight_layout()
    path = figures_root / "clear_topic_validation_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    clr_matrix = np.zeros((len(clear_order), len(contrast_codes) * 2), dtype=int)
    clr_labels: list[str] = []
    for contrast_index, contrast in enumerate(contrast_codes):
        for aggregation_index, aggregation in enumerate(AGGREGATIONS):
            column = contrast_index * 2 + aggregation_index
            clr_labels.append(f"{contrast}\n{aggregation}")
            for row, topic in enumerate(clear_order):
                agreement = clr.loc[
                    clr["topic_id"].eq(topic) & clr["contrast"].eq(contrast) & clr["aggregation"].eq(aggregation),
                    "direction_agreement",
                ].iloc[0]
                clr_matrix[row, column] = 1 if agreement else 0
    fig, axis = plt.subplots(figsize=(10.5, 5.8))
    axis.imshow(clr_matrix, cmap=ListedColormap(["#B91C1C", "#166534"]), vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(clr_labels)), clr_labels, fontsize=8)
    axis.set_yticks(np.arange(len(clear_order)), [f"{topic.replace('topic_', 'T')} {label_lookup[topic]}" for topic in clear_order])
    for row in range(clr_matrix.shape[0]):
        for column in range(clr_matrix.shape[1]):
            axis.text(column, row, "一致" if clr_matrix[row, column] else "反向", ha="center", va="center", color="white", fontsize=7)
    axis.set_title("Raw probability vs CLR contrast direction agreement")
    fig.tight_layout()
    path = figures_root / "raw_vs_clr_direction_agreement.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def compact_effect(record: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """将 primary result 压缩为 A–M 汇报所需字段。"""

    return {
        "topic_id": record["topic_id"],
        "topic_label": record["topic_label"],
        "contrast": record["contrast"],
        "raw_delta_poem": float(record["raw_effect_poem"]),
        "raw_delta_author": float(record["raw_effect_author"]),
        "hedges_g": float(record["hedges_g"]),
        "cliff_delta": float(record["cliff_delta"]),
        "validation_status": record["validation_status"],
    }


def build_summary(
    omnibus: pd.DataFrame,
    validation: pd.DataFrame,
    sensitivity: pd.DataFrame,
    clr: pd.DataFrame,
    supported: pd.DataFrame,
    queue: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """构造机器可读摘要与最终 A–M 所需结果。"""

    omnibus_counts = omnibus.groupby("aggregation")["fdr_pass"].sum().astype(int).to_dict()
    status_counts: dict[str, dict[str, int]] = {}
    for contrast in [item["code"] for item in config["primary_contrasts"]]:
        counts = validation.loc[validation["contrast"].eq(contrast), "validation_status"].value_counts().to_dict()
        status_counts[contrast] = {status: int(counts.get(status, 0)) for status in STATUS_ORDER}
    early_late = validation.loc[validation["contrast"].eq("E-L")]
    poem_max = early_late.loc[early_late["raw_effect_poem"].abs().idxmax()]
    author_max = early_late.loc[early_late["raw_effect_author"].abs().idxmax()]
    sign_complete: list[str] = []
    aggregation_conflict: list[str] = []
    for topic_id, group in validation.groupby("topic_id"):
        if not group["aggregation_conflict"].any():
            sign_complete.append(topic_id)
        else:
            aggregation_conflict.append(topic_id)
    sample_reversals = sensitivity.loc[sensitivity["sample_sign_reversal"], ["topic_id", "topic_label", "contrast", "sign_reversal_specifications"]].to_dict("records")
    clr_disagreements = clr.loc[~clr["direction_agreement"], ["topic_id", "topic_label", "contrast", "aggregation", "raw_direction", "clr_direction"]].to_dict("records")
    special: dict[str, list[dict[str, Any]]] = {}
    for topic_id in ["topic_03", "topic_08", "topic_06", "topic_16"]:
        special[topic_id] = [compact_effect(row) for _, row in validation.loc[validation["topic_id"].eq(topic_id)].sort_values("contrast_order").iterrows()]
    queue_topics = (
        queue.sort_values(["priority", "topic_id"])
        .groupby(["topic_id", "topic_label"], as_index=False)["priority"].min()
        .sort_values(["priority", "topic_id"])[["priority", "topic_id", "topic_label"]]
        .to_dict("records")
        if not queue.empty
        else []
    )
    return {
        "omnibus_fdr_pass": omnibus_counts,
        "validation_status_counts": status_counts,
        "early_late_largest_absolute": {
            "CORE_POEM": compact_effect(poem_max),
            "CORE_AUTHOR": compact_effect(author_max),
        },
        "core_poem_author_sign_complete_topics": sign_complete,
        "aggregation_conflict_topics": aggregation_conflict,
        "sample_sign_reversals": sample_reversals,
        "raw_vs_clr_direction_disagreements": clr_disagreements,
        "special_topics": special,
        "topic_06_in_author_queue": bool((queue["topic_id"].eq("topic_06")).any()) if not queue.empty else False,
        "statistically_supported_candidate_trends": supported.to_dict("records"),
        "author_influence_priority_topics": queue_topics,
    }


def format_effect_table(frame: pd.DataFrame) -> str:
    """把 primary 对比压缩为报告 Markdown 表。"""

    lines = [
        "| Topic | 对比 | POEM Δ [95% CI] | AUTHOR Δ [95% CI] | g | Cliff δ | q(P/A) | 状态 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in frame.to_dict("records"):
        lines.append(
            f"| {record['topic_id'].replace('topic_', 'T')} {record['topic_label']} | {record['contrast']} | "
            f"{record['raw_effect_poem']:.5f} [{record['poem_ci_low']:.5f}, {record['poem_ci_high']:.5f}] | "
            f"{record['raw_effect_author']:.5f} [{record['author_ci_low']:.5f}, {record['author_ci_high']:.5f}] | "
            f"{record['hedges_g']:.3f} | {record['cliff_delta']:.3f} | "
            f"{record['q_poem']:.3g}/{record['q_author']:.3g} | {record['validation_status']} |"
        )
    return "\n".join(lines)


def build_report(
    output_root: Path,
    omnibus: pd.DataFrame,
    validation: pd.DataFrame,
    sensitivity: pd.DataFrame,
    clr: pd.DataFrame,
    supported: pd.DataFrame,
    queue: pd.DataFrame,
    peak_trough: pd.DataFrame,
    config: dict[str, Any],
) -> Path:
    """生成18节统计验证报告，不进入文学史解释。"""

    omnibus_lines = ["| Topic | Aggregation | F(df1, df2) | p | q | FDR |", "|---|---|---:|---:|---:|---|"]
    for record in omnibus.to_dict("records"):
        omnibus_lines.append(
            f"| {record['topic_id'].replace('topic_', 'T')} {record['topic_label']} | {record['aggregation']} | "
            f"{record['wald_f']:.3f} ({record['df_num']}, {record['df_denom']}) | {record['p_value']:.3g} | "
            f"{record['q_value']:.3g} | {'PASS' if record['fdr_pass'] else 'NO'} |"
        )
    disagreement = validation.loc[validation["aggregation_conflict"]]
    reversal = sensitivity.loc[sensitivity["sample_sign_reversal"]]
    clr_conflict = clr.loc[~clr["direction_agreement"]]
    sections: list[str] = [
        "# QTS Author-Period Effect Size & Statistical Validation Report",
        "",
        "## 1. Objective",
        "本阶段验证作者所属时期与 Topic probability 之间关联的效应幅度、区间与稳健性；目标不是搜寻 p-value，也不作因果或文学史确认。",
        "",
        "## 2. Analytical Framework",
        f"CORE 为主要样本；STRICT 与 EXPANDED 只作效应方向、幅度和作者聚类 bootstrap 敏感性。时期固定为 {'、'.join(config['period_order'])}。POEM estimand 以诗等权，AUTHOR estimand 以作者等权。随机种子为 {config['random_seed']}，每套作者 cluster bootstrap 为 {config['bootstrap_iterations']} 次。",
        "",
        "## 3. Omnibus Period Effects",
        "CORE_POEM 与 CORE_AUTHOR 分别构成含9个 CLEAR Topics 的 BH-FDR family。POEM 使用作者 cluster CR1；AUTHOR 使用一行一作者的 HC3。",
        "\n".join(omnibus_lines),
        "",
        "## 4. Planned Contrast Effect Sizes",
        "主要对比为 E-H、H-M、M-L、E-L，原始概率差均定义为 B−A；百分比变化只作辅助。",
        format_effect_table(validation),
        "",
        "## 5. Poem-weighted Results",
        "CORE_POEM 报告 raw Δ、作者 cluster bootstrap 95% CI、CR1 cluster-robust 模型区间与 FDR。DESCRIPTIVE_CLUSTERED_SMD 只作描述，不把诗视为独立 cluster。",
        "",
        "## 6. Author-balanced Results",
        "CORE_AUTHOR 每位作者一行且不按诗数加权；报告 raw Δ、作者 bootstrap 95% CI、Hedges' g、Cliff's delta 与 HC3 稳健推断。",
        "",
        "## 7. Aggregation Agreement",
        f"方向冲突的 Topic/contrast 共 {len(disagreement)} 项。HIGH 需同号、bootstrap 支持方向一致且无配置定义的巨大幅度差；同号但区间或幅度证据不完整为 MODERATE。",
        "",
        "## 8. Sample Sensitivity",
        f"STRICT/CORE/EXPANDED 内出现符号反转的 Topic/contrast 共 {len(reversal)} 项；六规格完整保留，未新增 sensitivity p-value families。",
        "",
        "## 9. Compositional Sensitivity",
        f"CLR 仅在 CORE 的 CLEAR Topics 临时执行；raw 与 CLR 方向不一致共 {len(clr_conflict)} 项。CLR 数值不覆盖 raw probability。",
        "",
        "## 10. Representativeness Warnings",
        "representativeness_risk 直接复用既有审计。MODERATE/HIGH 风险进入 validation_reason，不因较大的标准化效应而被静默忽略。",
        "",
        "## 11. Validation Status",
        "状态由 raw Δ、bootstrap CI、标准化效应、FDR、聚合一致性、六规格敏感性、CLR 与 metadata risk 共同生成。STATISTICALLY_SUPPORTED_CANDIDATE 仍不等于 LITERARY_HISTORY_CONFIRMED。",
        "",
    ]
    for number, topic_id in [(12, "topic_03"), (13, "topic_08"), (14, "topic_06"), (15, "topic_16")]:
        special = validation.loc[validation["topic_id"].eq(topic_id)].sort_values("contrast_order")
        sections.extend(
            [
                f"## {number}. Topic {topic_id[-2:]}",
                "以下只报告统计结构；不作文学史解释。" if topic_id != "topic_16" else "以下重点并列 POEM 与 AUTHOR estimand；两者回答不同加权问题，不裁定哪一个‘正确’。",
                format_effect_table(special),
                "",
            ]
        )
    candidate_text = "无。" if supported.empty else ", ".join(f"{row.topic_id.replace('topic_', 'T')} {row.contrast} ({row.validation_status})" for row in supported.itertuples())
    queue_text = "无。" if queue.empty else ", ".join(sorted(set(queue["topic_id"].str.replace("topic_", "T"))))
    sections.extend(
        [
            "## 16. Candidate Trends",
            f"达到 STATISTICALLY_SUPPORTED_CANDIDATE 门槛的 Topic/contrast：{candidate_text}",
            "",
            "## 17. Author Influence Queue",
            f"下一阶段需优先执行 Author Influence / Leave-One-Author-Out 的 Topic：{queue_text}",
            "",
            "## 18. Limitations",
            "- author period ≠ poem composition date。",
            "- metadata 仍不完整，representativeness warning 必须保留。",
            "- 作者在语料中的诗作数量高度不均。",
            "- Topic probabilities 为 compositional data；CLR 只作敏感性。",
            "- statistical significance ≠ literary importance。",
            "- post-hoc peak/trough comparisons 未纳入 primary FDR；见 `posthoc_peak_trough_effects.csv`。",
            "- 所有结果仅表示作者所属时期与 Topic probability 的关联，不是因果效应。",
        ]
    )
    path = output_root / "QTS_Author_Period_Effect_Size_Statistical_Validation_Report.md"
    path.write_text("\n".join(sections) + "\n", encoding="utf-8", newline="\n")
    return path


def build_readme(output_root: Path, config: dict[str, Any], frames: dict[str, pd.DataFrame]) -> Path:
    """写出目录级说明与复现命令。"""

    lines = [
        "# Author-Period Effect Size & Statistical Validation",
        "",
        "本目录保存 C_K20 作者时期主题效应量、稳健推断、样本敏感性、CLR 敏感性与综合验证状态。",
        "",
        "## Reproduce",
        "",
        "```powershell",
        "python src/run_author_period_effect_validation.py --config config/effect_validation_config.yaml",
        "```",
        "",
        "## Fixed design",
        "",
        f"- Bootstrap iterations: {config['bootstrap_iterations']}",
        f"- Random seed: {config['random_seed']}",
        "- Formal inference: 9 CLEAR Topics under CORE_POEM and CORE_AUTHOR only",
        "- Sensitivity: STRICT/CORE/EXPANDED × POEM/AUTHOR without extra p-value families",
        "- Primary contrasts: E-H, H-M, M-L, E-L",
        "- Secondary contrasts: E-M, H-L",
        "",
        "## Data tables",
        "",
    ]
    lines.extend(f"- `data/{name}.csv`: {len(frame):,} rows" for name, frame in frames.items())
    path = output_root / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def main() -> None:
    """执行完整效应量与统计验证流水线。"""

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
    profiles = build_author_profiles(poems, topic_columns, config)
    omnibus, models = build_models(poems, profiles, topic_columns, topic_info, config)
    all_bootstrap, bootstrap_hash = build_bootstrap(poems, topic_columns, config)
    all_effects = build_all_topic_effect_sizes(
        poems, profiles, topic_columns, topic_info, all_bootstrap, models, config
    )
    pairwise = build_clear_pairwise(all_effects)
    clr, clr_hash, clr_zero_count, clr_pseudocount = build_clr_sensitivity(
        poems, profiles, topic_columns, topic_info, all_effects, config
    )
    sensitivity = build_sensitivity_matrix(all_effects, config)
    validation = build_primary_validation(all_effects, sensitivity, clr, topic_info, config)
    primary = validation.copy()
    supported = build_supported_candidates(validation)
    clear_summary = build_clear_topic_summary(validation)
    queue = build_author_queue(validation, input_paths, config)
    peak_trough = build_peak_trough_effects(all_effects, input_paths, config)
    invariance = run_invariance_tests(
        poems,
        labels,
        topic_columns,
        topic_info,
        profiles,
        all_effects,
        clr_hash,
        clr_zero_count,
        bootstrap_hash,
        input_paths,
        source_hash_before,
        config,
    )

    output_frames = {
        "clear_topic_period_omnibus": omnibus,
        "clear_topic_primary_contrasts": primary,
        "clear_topic_all_pairwise": pairwise,
        "clear_topic_effect_sensitivity_matrix": sensitivity,
        "raw_vs_clr_direction_agreement": clr,
        "all_topic_effect_sizes": all_effects,
        "statistically_supported_candidate_trends": supported,
        "author_influence_check_queue": queue,
        "clear_topics_validation_summary": clear_summary,
        "posthoc_peak_trough_effects": peak_trough,
        "invariance_tests": invariance,
    }
    data_dictionary = build_data_dictionary(output_frames)
    output_frames["data_dictionary"] = data_dictionary
    for name, frame in output_frames.items():
        save_csv(frame, data_root / f"{name}.csv")

    figure_paths = build_figures(figures_root, validation, sensitivity, clr, config)
    summary = build_summary(omnibus, validation, sensitivity, clr, supported, queue, config)
    summary["clr"] = {
        "zero_count": clr_zero_count,
        "pseudocount": clr_pseudocount,
        "temporary_matrix_sha256": clr_hash,
    }
    summary["bootstrap"] = {
        "iterations": config["bootstrap_iterations"],
        "seed": config["random_seed"],
        "replicate_sha256": bootstrap_hash,
    }
    summary_path = data_root / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")
    report_path = build_report(output_root, omnibus, validation, sensitivity, clr, supported, queue, peak_trough, config)
    readme_path = build_readme(output_root, config, output_frames)

    output_paths = [data_root / f"{name}.csv" for name in output_frames]
    output_paths.extend(figure_paths)
    output_paths.extend([summary_path, report_path, readme_path])
    manifest = {
        "analysis_version": config["analysis_version"],
        "created_at_local_date": "2026-09-01",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": stats.__version__ if hasattr(stats, "__version__") else __import__("scipy").__version__,
        "matplotlib": matplotlib.__version__,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "inputs": {name: {"path": str(path), "sha256": input_hashes[name]} for name, path in input_paths.items()},
        "bootstrap": {
            "iterations": config["bootstrap_iterations"],
            "seed": config["random_seed"],
            "algorithm": config["bootstrap_algorithm"],
            "replicate_sha256": bootstrap_hash,
        },
        "clr": {
            "zero_count": clr_zero_count,
            "pseudocount": clr_pseudocount,
            "temporary_matrix_sha256": clr_hash,
        },
        "outputs": {
            str(path.relative_to(output_root)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(output_paths)
        },
        "invariance_tests_passed": int(invariance["status"].eq("PASS").sum()),
        "invariance_tests_total": int(len(invariance)),
    }
    manifest_path = output_root / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
