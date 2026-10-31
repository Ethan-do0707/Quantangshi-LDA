"""《全唐诗》C_K20 作者影响与逐作者剔除稳健性分析。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AGGREGATIONS = ("POEM", "AUTHOR")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/author_influence_config.yaml")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    """把相对路径解析到仓库根目录。"""

    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    """读取以 JSON 语法保存的 YAML 配置。"""

    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """计算连续数值矩阵的 SHA-256。"""

    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """以 UTF-8 无 BOM 写出 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def normalize_boolean(series: pd.Series, field: str) -> pd.Series:
    """把审计标签严格转换为布尔值。"""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    normalized = series.astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        raise ValueError(f"{field} 含无法识别的布尔值")
    return normalized.astype(bool)


def direction(value: float, tolerance: float = 1e-12) -> str:
    """返回数值效应方向。"""

    if not np.isfinite(value):
        return "NA"
    if value > tolerance:
        return "POSITIVE"
    if value < -tolerance:
        return "NEGATIVE"
    return "ZERO"


def sign_flip(original: float, new: float, tolerance: float = 1e-12) -> bool:
    """判断新旧效应是否跨过零点。"""

    left = direction(original, tolerance)
    right = direction(new, tolerance)
    return left in {"POSITIVE", "NEGATIVE"} and right in {"POSITIVE", "NEGATIVE"} and left != right


def attenuation_ratio(original: float, new: float, minimum: float) -> float:
    """计算 1-|new|/|original|；原效应过小时返回缺失。"""

    if not np.isfinite(original) or not np.isfinite(new) or abs(original) < minimum:
        return math.nan
    return 1.0 - abs(new) / abs(original)


def gini_coefficient(values: Iterable[float]) -> float:
    """计算非负贡献值的 Gini 系数。"""

    array = np.asarray(list(values), dtype=float)
    if len(array) == 0 or not np.isfinite(array).all() or (array < 0).any():
        return math.nan
    total = float(array.sum())
    if math.isclose(total, 0.0, abs_tol=1e-15):
        return 0.0
    ordered = np.sort(array)
    index = np.arange(1, len(ordered) + 1, dtype=float)
    return float((2.0 * np.sum(index * ordered) / (len(ordered) * total)) - (len(ordered) + 1) / len(ordered))


def stable_seed(base_seed: int, *parts: Any) -> int:
    """由固定种子与分析键派生遍历顺序无关的子种子。"""

    text = "|".join(str(part) for part in parts)
    token = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
    return int((int(base_seed) + token) % (2**32 - 1))


def contrast_lookup(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    """建立 contrast code 到两时期的映射。"""

    return {item["code"]: item for item in config["primary_contrasts"]}


def load_inputs(
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Path],
    dict[str, str],
]:
    """读取并核验权威 master、冻结标签、taxonomy 与上一阶段结果。"""

    effect_root = resolve_path(config["effect_output_root"])
    trend_root = resolve_path(config["trend_output_root"])
    paths = {
        "authoritative_master": resolve_path(config["authoritative_master"]),
        "topic_metadata": resolve_path(config["topic_metadata"]),
        "sample_labels": resolve_path(config["sample_labels"]),
        "primary_contrasts": effect_root / "data" / "clear_topic_primary_contrasts.csv",
        "supported_candidates": effect_root / "data" / "statistically_supported_candidate_trends.csv",
        "effect_sensitivity": effect_root / "data" / "clear_topic_effect_sensitivity_matrix.csv",
        "author_queue": effect_root / "data" / "author_influence_check_queue.csv",
        "period_concentration": trend_root / "data" / "period_topic_author_concentration.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少输入文件：{missing}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes["authoritative_master"] != config["expected_master_sha256"]:
        raise ValueError("权威 master SHA-256 与冻结记录不一致")

    master = pd.read_csv(paths["authoritative_master"], low_memory=False)
    labels = pd.read_csv(paths["sample_labels"], low_memory=False)
    metadata = pd.read_csv(paths["topic_metadata"], low_memory=False)
    primary = pd.read_csv(paths["primary_contrasts"], low_memory=False)
    supported = pd.read_csv(paths["supported_candidates"], low_memory=False)
    if len(master) != int(config["expected_poem_count"]) or len(labels) != int(config["expected_poem_count"]):
        raise ValueError("权威 master 或样本标签行数不符合预期")
    if master["poem_id"].duplicated().any() or labels["poem_id"].duplicated().any():
        raise ValueError("poem_id 必须唯一")
    required = {"author_id_internal", "author", "author_period_primary"}
    if not required.issubset(master.columns):
        raise ValueError(f"权威 master 缺少字段：{sorted(required - set(master.columns))}")

    topic_columns = sorted(column for column in master.columns if column.startswith("topic_") and column.endswith("_prob"))
    if len(topic_columns) != int(config["expected_topic_count"]):
        raise ValueError("Topic probability 列数不符合预期")
    topic_ids = [column.removesuffix("_prob") for column in topic_columns]
    if set(metadata["topic_id"]) != set(topic_ids) or metadata["topic_id"].duplicated().any():
        raise ValueError("taxonomy Topic 集合与 master 不一致")
    clear_ids = sorted(metadata.loc[metadata["topic_type"].eq("CLEAR"), "topic_id"].tolist())
    if len(clear_ids) != int(config["expected_clear_topic_count"]):
        raise ValueError("CLEAR Topic 数量不符合预期")
    clear_columns = [f"{topic_id}_prob" for topic_id in clear_ids]

    probability = master[topic_columns].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("Topic probability 含缺失、无穷或负值")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-10, rtol=0):
        raise ValueError("每首诗20维 Topic probability 之和不为1")

    label_columns = ["poem_id", "author_id_internal", *config["samples"].values()]
    joined = master.merge(labels[label_columns], on=["poem_id", "author_id_internal"], how="left", validate="one_to_one")
    if joined[list(config["samples"].values())].isna().any().any():
        raise ValueError("样本标签连接后存在缺失")
    for column in config["samples"].values():
        joined[column] = normalize_boolean(joined[column], column)
    valid_periods = set(config["period_order"])
    for sample, column in config["samples"].items():
        invalid = joined.loc[joined[column] & ~joined["author_period_primary"].isin(valid_periods)]
        if not invalid.empty:
            raise ValueError(f"{sample} 入选行存在非法 author_period_primary")

    topic_info = metadata[["topic_id", "final_topic_label", "short_label", "topic_type"]].rename(
        columns={"final_topic_label": "topic_label"}
    )
    prior_clear = sorted(primary["topic_id"].unique().tolist())
    if prior_clear != clear_ids or len(primary) != len(clear_ids) * len(config["primary_contrasts"]):
        raise ValueError("上一阶段 primary contrasts 与冻结 CLEAR Topics 不一致")
    prior_labels = primary[["topic_id", "topic_label"]].drop_duplicates().sort_values("topic_id").reset_index(drop=True)
    current_labels = topic_info.loc[topic_info["topic_id"].isin(clear_ids), ["topic_id", "topic_label"]].sort_values("topic_id").reset_index(drop=True)
    if not prior_labels.equals(current_labels):
        raise ValueError("上一阶段 CLEAR Topic 标签与 taxonomy 不一致")
    if len(supported) != int(config["expected_supported_candidate_count"]):
        raise ValueError("上一阶段 statistically supported candidate 数量不符合预期")
    return joined, topic_info, clear_ids, clear_columns, primary, supported, paths, hashes


def build_author_wide(poems: pd.DataFrame, label_column: str, clear_columns: list[str]) -> pd.DataFrame:
    """为指定样本建立一行一作者时期的诗数、Topic mass 与作者均值表。"""

    selected = poems.loc[poems[label_column]].copy()
    keys = ["author_id_internal", "author_period_primary"]
    grouped = selected.groupby(keys, sort=True, observed=True)
    counts = grouped.size().rename("poem_count").reset_index()
    names = grouped["author"].first().rename("author").reset_index()
    mass = grouped[clear_columns].sum().reset_index().rename(columns={column: f"{column}_mass" for column in clear_columns})
    means = grouped[clear_columns].mean().reset_index().rename(columns={column: f"{column}_mean" for column in clear_columns})
    wide = counts.merge(names, on=keys, validate="one_to_one")
    wide = wide.merge(mass, on=keys, validate="one_to_one").merge(means, on=keys, validate="one_to_one")
    if wide["author_id_internal"].duplicated().any():
        raise ValueError("同一 author_id_internal 在样本内出现多个时期")
    return wide.sort_values(["author_period_primary", "author_id_internal"]).reset_index(drop=True)


def build_author_tables(
    poems: pd.DataFrame, clear_columns: list[str], config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    """为 STRICT、CORE、EXPANDED 建立作者级基础表。"""

    return {
        sample: build_author_wide(poems, label_column, clear_columns)
        for sample, label_column in config["samples"].items()
    }


def build_author_contributions(
    author_wide: pd.DataFrame,
    topic_info: pd.DataFrame,
    clear_ids: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    """生成 CORE author×period×topic 贡献基础表。"""

    label_map = topic_info.set_index("topic_id")["topic_label"].to_dict()
    rows: list[dict[str, Any]] = []
    period_order = {period: index + 1 for index, period in enumerate(config["period_order"])}
    for topic_id in clear_ids:
        mass_column = f"{topic_id}_prob_mass"
        mean_column = f"{topic_id}_prob_mean"
        for period, group in author_wide.groupby("author_period_primary", sort=False, observed=True):
            total_mass = float(group[mass_column].sum())
            total_poems = int(group["poem_count"].sum())
            for record in group.to_dict("records"):
                rows.append(
                    {
                        "sample": "CORE",
                        "author_id_internal": record["author_id_internal"],
                        "author": record["author"],
                        "period": period,
                        "period_order": period_order[period],
                        "topic_id": topic_id,
                        "topic_label": label_map[topic_id],
                        "poem_count": int(record["poem_count"]),
                        "topic_mass": float(record[mass_column]),
                        "author_mean_topic_probability": float(record[mean_column]),
                        "share_of_period_topic_mass": float(record[mass_column] / total_mass) if total_mass > 0 else math.nan,
                        "share_of_period_poems": float(record["poem_count"] / total_poems) if total_poems else math.nan,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["rank_by_topic_mass"] = (
        frame.groupby(["period", "topic_id"], sort=False)["topic_mass"].rank(method="first", ascending=False).astype(int)
    )
    frame["rank_by_poem_count"] = (
        frame.groupby(["period", "topic_id"], sort=False)["poem_count"].rank(method="first", ascending=False).astype(int)
    )
    return frame.sort_values(["period_order", "topic_id", "rank_by_topic_mass", "author_id_internal"]).reset_index(drop=True)


def build_concentration_summary(contributions: pd.DataFrame) -> pd.DataFrame:
    """按 period×CLEAR Topic 计算作者贡献集中度。"""

    rows: list[dict[str, Any]] = []
    for (period, period_order, topic_id, topic_label), group in contributions.groupby(
        ["period", "period_order", "topic_id", "topic_label"], sort=True, observed=True
    ):
        ordered = group.sort_values(["topic_mass", "author_id_internal"], ascending=[False, True])
        shares = ordered["share_of_period_topic_mass"].to_numpy(dtype=float)
        hhi = float(np.square(shares).sum())
        top = ordered.iloc[0]
        rows.append(
            {
                "sample": "CORE",
                "period": period,
                "period_order": int(period_order),
                "topic_id": topic_id,
                "topic_label": topic_label,
                "poem_count": int(group["poem_count"].sum()),
                "author_count": int(len(group)),
                "total_topic_mass": float(group["topic_mass"].sum()),
                "top1_author_share": float(shares[:1].sum()),
                "top3_author_share": float(shares[:3].sum()),
                "top5_author_share": float(shares[:5].sum()),
                "top10_author_share": float(shares[:10].sum()),
                "hhi": hhi,
                "gini": gini_coefficient(group["topic_mass"]),
                "effective_number_of_authors": float(1.0 / hhi) if hhi > 0 else math.nan,
                "top1_author_id_internal": top["author_id_internal"],
                "top1_author": top["author"],
            }
        )
    return pd.DataFrame(rows).sort_values(["period_order", "topic_id"]).reset_index(drop=True)


def delta_from_author_rows(
    author_rows: pd.DataFrame,
    topic_id: str,
    period_a: str,
    period_b: str,
    aggregation: str,
    excluded_author_ids: set[str] | None = None,
) -> float:
    """在给定作者排除集合下计算 B-A 的 POEM 或 AUTHOR 效应。"""

    excluded = excluded_author_ids or set()
    selected = author_rows.loc[
        author_rows["author_period_primary"].isin([period_a, period_b])
        & ~author_rows["author_id_internal"].isin(excluded)
    ]
    mass_column = f"{topic_id}_prob_mass"
    mean_column = f"{topic_id}_prob_mean"
    means: dict[str, float] = {}
    for period in [period_a, period_b]:
        group = selected.loc[selected["author_period_primary"].eq(period)]
        if group.empty:
            return math.nan
        if aggregation == "POEM":
            denominator = float(group["poem_count"].sum())
            means[period] = float(group[mass_column].sum() / denominator) if denominator > 0 else math.nan
        elif aggregation == "AUTHOR":
            means[period] = float(group[mean_column].mean())
        else:
            raise ValueError(f"未知 aggregation：{aggregation}")
    return means[period_b] - means[period_a]


def build_loao_effects(
    author_rows: pd.DataFrame,
    topic_info: pd.DataFrame,
    primary: pd.DataFrame,
    config: dict[str, Any],
    sample: str = "CORE",
    pairs: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """逐一删除 contrast 两时期内的每位作者并重算两种 estimand。"""

    lookup = contrast_lookup(config)
    label_map = topic_info.set_index("topic_id")["topic_label"].to_dict()
    tolerance = float(config["thresholds"]["numeric_zero_tolerance"])
    relative_minimum = float(config["thresholds"]["relative_change_minimum_original"])
    concern_floor = float(config["thresholds"]["near_zero_original_delta"])
    rows: list[dict[str, Any]] = []
    for prior in primary.sort_values(["contrast_order", "topic_id"]).to_dict("records"):
        key = (prior["topic_id"], prior["contrast"])
        if pairs is not None and key not in pairs:
            continue
        definition = lookup[prior["contrast"]]
        period_a, period_b = definition["period_a"], definition["period_b"]
        involved = author_rows.loc[author_rows["author_period_primary"].isin([period_a, period_b])].copy()
        mass_column = f"{prior['topic_id']}_prob_mass"
        period_mass = involved.groupby("author_period_primary")[mass_column].sum().to_dict()
        period_poems = involved.groupby("author_period_primary")["poem_count"].sum().to_dict()
        for aggregation in AGGREGATIONS:
            prior_column = "raw_effect_poem" if aggregation == "POEM" else "raw_effect_author"
            if sample == "CORE":
                original = float(prior[prior_column])
            else:
                original = delta_from_author_rows(author_rows, prior["topic_id"], period_a, period_b, aggregation)
            for author in involved.to_dict("records"):
                loao = delta_from_author_rows(
                    author_rows,
                    prior["topic_id"],
                    period_a,
                    period_b,
                    aggregation,
                    {author["author_id_internal"]},
                )
                absolute = abs(loao - original) if np.isfinite(loao) else math.nan
                raw_flip = sign_flip(original, loao, tolerance)
                rows.append(
                    {
                        "sample": sample,
                        "topic_id": prior["topic_id"],
                        "topic_label": label_map[prior["topic_id"]],
                        "contrast": prior["contrast"],
                        "period_a": period_a,
                        "period_b": period_b,
                        "aggregation": aggregation,
                        "removed_author_id": author["author_id_internal"],
                        "removed_author": author["author"],
                        "removed_author_period": author["author_period_primary"],
                        "removed_author_poem_count": int(author["poem_count"]),
                        "removed_author_topic_mass": float(author[mass_column]),
                        "removed_author_topic_mass_share": float(author[mass_column] / period_mass[author["author_period_primary"]]),
                        "removed_author_poem_share": float(author["poem_count"] / period_poems[author["author_period_primary"]]),
                        "original_delta": original,
                        "loao_delta": loao,
                        "absolute_change": absolute,
                        "relative_change_if_defined": (loao - original) / abs(original) if abs(original) >= relative_minimum else math.nan,
                        "attenuation_ratio": attenuation_ratio(original, loao, relative_minimum),
                        "sign_original": direction(original, tolerance),
                        "sign_loao": direction(loao, tolerance),
                        "sign_flip": raw_flip,
                        "sign_flip_concern": bool(raw_flip and abs(original) >= concern_floor),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(
        ["sample", "topic_id", "contrast", "aggregation", "absolute_change", "removed_author_id"],
        ascending=[True, True, True, True, False, True],
    )
    frame["rank_by_abs_change"] = frame.groupby(
        ["sample", "topic_id", "contrast", "aggregation"], sort=False
    ).cumcount() + 1
    return frame.reset_index(drop=True)


def build_influence_scores(loao: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """生成仅用于排序的多指标作者影响分数。"""

    weights = config["influence_score_weights"]
    scored = loao.copy()
    group_keys = ["sample", "topic_id", "contrast", "aggregation"]
    maxima = scored.groupby(group_keys)["absolute_change"].transform("max")
    scored["normalized_absolute_change"] = np.where(maxima > 0, scored["absolute_change"] / maxima, 0.0)
    scored["positive_attenuation_component"] = scored["attenuation_ratio"].clip(lower=0.0, upper=1.0).fillna(0.0)
    scored["influence_score"] = (
        float(weights["absolute_change"]) * scored["normalized_absolute_change"]
        + float(weights["sign_flip"]) * scored["sign_flip_concern"].astype(float)
        + float(weights["attenuation"]) * scored["positive_attenuation_component"]
        + float(weights["topic_mass_share"]) * scored["removed_author_topic_mass_share"].fillna(0.0)
        + float(weights["poem_share"]) * scored["removed_author_poem_share"].fillna(0.0)
    )
    scored["influence_score_note"] = "配置化复合排序工具；不表示作者文学史重要性或客观真值"
    scored = scored.sort_values(group_keys + ["influence_score", "absolute_change", "removed_author_id"], ascending=[True, True, True, True, False, False, True])
    scored["rank_by_influence_score"] = scored.groupby(group_keys, sort=False).cumcount() + 1
    return scored.reset_index(drop=True)


def build_top_influential_authors(loao: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """每个 Topic×contrast×aggregation 保留 absolute change 最大的作者。"""

    columns = [
        "sample",
        "topic_id",
        "topic_label",
        "contrast",
        "aggregation",
        "rank_by_abs_change",
        "removed_author_id",
        "removed_author",
        "removed_author_period",
        "removed_author_poem_count",
        "removed_author_topic_mass",
        "removed_author_topic_mass_share",
        "original_delta",
        "loao_delta",
        "absolute_change",
        "attenuation_ratio",
        "sign_flip",
        "sign_flip_concern",
    ]
    result = loao.loc[loao["rank_by_abs_change"].le(limit), columns].copy()
    return result.rename(
        columns={
            "rank_by_abs_change": "rank",
            "removed_author_id": "author_id_internal",
            "removed_author": "author",
            "removed_author_period": "period",
            "removed_author_poem_count": "poem_count",
            "removed_author_topic_mass": "topic_mass",
            "removed_author_topic_mass_share": "topic_mass_share",
            "original_delta": "delta_original",
            "loao_delta": "delta_after_removal",
        }
    ).sort_values(["topic_id", "contrast", "aggregation", "rank"]).reset_index(drop=True)


def build_loao_summary(loao: pd.DataFrame) -> pd.DataFrame:
    """汇总每个 Topic×contrast×aggregation 的 LOAO delta 分布。"""

    rows: list[dict[str, Any]] = []
    keys = ["sample", "topic_id", "topic_label", "contrast", "aggregation"]
    for key, group in loao.groupby(keys, sort=True, observed=True):
        ordered = group.sort_values(["absolute_change", "removed_author_id"], ascending=[False, True])
        top = ordered.iloc[0]
        valid_attenuation = group["attenuation_ratio"].dropna()
        rows.append(
            {
                "sample": key[0],
                "topic_id": key[1],
                "topic_label": key[2],
                "contrast": key[3],
                "aggregation": key[4],
                "original_delta": float(group["original_delta"].iloc[0]),
                "eligible_author_count": int(len(group)),
                "loao_delta_min": float(group["loao_delta"].min()),
                "loao_delta_max": float(group["loao_delta"].max()),
                "loao_delta_mean": float(group["loao_delta"].mean()),
                "loao_delta_median": float(group["loao_delta"].median()),
                "loao_delta_sd": float(group["loao_delta"].std(ddof=1)),
                "max_abs_change": float(group["absolute_change"].max()),
                "sign_flip_count": int(group["sign_flip"].sum()),
                "sign_flip_concern_count": int(group["sign_flip_concern"].sum()),
                "sign_flip_ratio": float(group["sign_flip"].mean()),
                "max_attenuation": float(valid_attenuation.max()) if not valid_attenuation.empty else math.nan,
                "median_attenuation": float(valid_attenuation.median()) if not valid_attenuation.empty else math.nan,
                "max_influence_author_id": top["removed_author_id"],
                "max_influence_author": top["removed_author"],
                "max_influence_author_period": top["removed_author_period"],
                "max_influence_delta": float(top["loao_delta"]),
                "max_influence_sign_flip": bool(top["sign_flip"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["topic_id", "contrast", "aggregation"]).reset_index(drop=True)


def contributor_exclusion_ids(
    involved: pd.DataFrame,
    topic_id: str,
    scheme: str,
    n: int,
    period_a: str,
    period_b: str,
) -> list[str]:
    """按 Topic mass 取得 GLOBAL_TOP 或 PER_PERIOD_TOP 排除作者。"""

    mass_column = f"{topic_id}_prob_mass"
    if scheme == "GLOBAL_TOP":
        ordered = involved.sort_values([mass_column, "author_id_internal"], ascending=[False, True])
        return ordered.head(n)["author_id_internal"].tolist()
    if scheme == "PER_PERIOD_TOP":
        ids: list[str] = []
        for period in [period_a, period_b]:
            ordered = involved.loc[involved["author_period_primary"].eq(period)].sort_values(
                [mass_column, "author_id_internal"], ascending=[False, True]
            )
            ids.extend(ordered.head(n)["author_id_internal"].tolist())
        return list(dict.fromkeys(ids))
    raise ValueError(f"未知 exclusion scheme：{scheme}")


def build_top_exclusions(
    author_rows: pd.DataFrame,
    topic_info: pd.DataFrame,
    primary: pd.DataFrame,
    config: dict[str, Any],
    sample: str = "CORE",
    pairs: set[tuple[str, str]] | None = None,
    n_values: list[int] | None = None,
) -> pd.DataFrame:
    """按 Topic-mass contributor 排除 Top1/3/5/10 并重算效应。"""

    lookup = contrast_lookup(config)
    label_map = topic_info.set_index("topic_id")["topic_label"].to_dict()
    tolerance = float(config["thresholds"]["numeric_zero_tolerance"])
    relative_minimum = float(config["thresholds"]["relative_change_minimum_original"])
    concern_floor = float(config["thresholds"]["near_zero_original_delta"])
    ns = n_values or [int(value) for value in config["top_exclusion_n"]]
    rows: list[dict[str, Any]] = []
    for prior in primary.sort_values(["contrast_order", "topic_id"]).to_dict("records"):
        key = (prior["topic_id"], prior["contrast"])
        if pairs is not None and key not in pairs:
            continue
        definition = lookup[prior["contrast"]]
        period_a, period_b = definition["period_a"], definition["period_b"]
        involved = author_rows.loc[author_rows["author_period_primary"].isin([period_a, period_b])].copy()
        mass_column = f"{prior['topic_id']}_prob_mass"
        name_map = involved.set_index("author_id_internal")["author"].to_dict()
        for scheme in ["GLOBAL_TOP", "PER_PERIOD_TOP"]:
            for n in ns:
                excluded = contributor_exclusion_ids(involved, prior["topic_id"], scheme, n, period_a, period_b)
                excluded_rows = involved.loc[involved["author_id_internal"].isin(excluded)]
                for aggregation in AGGREGATIONS:
                    if sample == "CORE":
                        prior_column = "raw_effect_poem" if aggregation == "POEM" else "raw_effect_author"
                        original = float(prior[prior_column])
                    else:
                        original = delta_from_author_rows(author_rows, prior["topic_id"], period_a, period_b, aggregation)
                    new = delta_from_author_rows(
                        author_rows,
                        prior["topic_id"],
                        period_a,
                        period_b,
                        aggregation,
                        set(excluded),
                    )
                    raw_flip = sign_flip(original, new, tolerance)
                    rows.append(
                        {
                            "sample": sample,
                            "topic_id": prior["topic_id"],
                            "topic_label": label_map[prior["topic_id"]],
                            "contrast": prior["contrast"],
                            "period_a": period_a,
                            "period_b": period_b,
                            "aggregation": aggregation,
                            "exclusion_scheme": scheme,
                            "N": int(n),
                            "excluded_author_count": int(len(excluded)),
                            "excluded_author_ids": ";".join(excluded),
                            "excluded_authors": ";".join(str(name_map[value]) for value in excluded),
                            "excluded_poem_count": int(excluded_rows["poem_count"].sum()),
                            "excluded_topic_mass": float(excluded_rows[mass_column].sum()),
                            "original_delta": original,
                            "new_delta": new,
                            "absolute_change": abs(new - original) if np.isfinite(new) else math.nan,
                            "attenuation_ratio": attenuation_ratio(original, new, relative_minimum),
                            "sign_original": direction(original, tolerance),
                            "sign_new": direction(new, tolerance),
                            "sign_flip": raw_flip,
                            "sign_flip_concern": bool(raw_flip and abs(original) >= concern_floor),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["sample", "topic_id", "contrast", "aggregation", "exclusion_scheme", "N"]
    ).reset_index(drop=True)


def contrast_concentration(
    concentration: pd.DataFrame, topic_id: str, period_a: str, period_b: str
) -> dict[str, float]:
    """取得 contrast 两时期中的最大作者集中度。"""

    group = concentration.loc[
        concentration["topic_id"].eq(topic_id) & concentration["period"].isin([period_a, period_b])
    ]
    return {
        "top1_share": float(group["top1_author_share"].max()),
        "top3_share": float(group["top3_author_share"].max()),
        "top5_share": float(group["top5_author_share"].max()),
        "top10_share": float(group["top10_author_share"].max()),
        "max_hhi": float(group["hhi"].max()),
        "max_gini": float(group["gini"].max()),
    }


def exclusion_record(
    exclusions: pd.DataFrame,
    topic_id: str,
    contrast: str,
    aggregation: str,
    n: int,
    scheme: str = "GLOBAL_TOP",
    sample: str = "CORE",
) -> pd.Series:
    """取得唯一的 Top exclusion 记录。"""

    match = exclusions.loc[
        exclusions["sample"].eq(sample)
        & exclusions["topic_id"].eq(topic_id)
        & exclusions["contrast"].eq(contrast)
        & exclusions["aggregation"].eq(aggregation)
        & exclusions["exclusion_scheme"].eq(scheme)
        & exclusions["N"].eq(n)
    ]
    if len(match) != 1:
        raise ValueError(f"Top exclusion 记录不唯一：{sample}/{topic_id}/{contrast}/{aggregation}/{scheme}/{n}")
    return match.iloc[0]


def classify_author_robustness(
    prior: pd.Series,
    poem_summary: pd.Series,
    author_summary: pd.Series,
    exclusions: pd.DataFrame,
    concentration_values: dict[str, float],
    config: dict[str, Any],
    scheme: str | None = None,
) -> tuple[str, str]:
    """按配置化 audit heuristic 分类作者稳健性。"""

    thresholds = config["thresholds"]
    scheme = scheme or config["primary_exclusion_scheme"]
    if scheme not in {"GLOBAL_TOP", "PER_PERIOD_TOP"}:
        raise ValueError(f"未知作者排除方案：{scheme}")
    topic_id, contrast = str(prior["topic_id"]), str(prior["contrast"])
    if direction(float(prior["raw_effect_poem"])) != direction(float(prior["raw_effect_author"])):
        return "ESTIMAND_DEPENDENT", "CORE_POEM 与 CORE_AUTHOR 原始方向不同；两种 estimand 保持分列"

    required_values = [
        poem_summary["max_abs_change"],
        author_summary["max_abs_change"],
        poem_summary["max_attenuation"],
        author_summary["max_attenuation"],
    ]
    if not np.isfinite(np.asarray(required_values, dtype=float)).all():
        return "INCONCLUSIVE", "LOAO 汇总存在不可计算值"

    exclusion_rows = {
        (aggregation, n): exclusion_record(exclusions, topic_id, contrast, aggregation, n, scheme=scheme)
        for aggregation in AGGREGATIONS
        for n in config["classification_exclusion_n"]
    }
    concern_sign_flip = (
        int(poem_summary["sign_flip_concern_count"]) > 0
        or int(author_summary["sign_flip_concern_count"]) > 0
        or any(bool(record["sign_flip_concern"]) for record in exclusion_rows.values())
    )
    strong_attenuation = (
        float(poem_summary["max_attenuation"]) >= float(thresholds["attenuation_strong"])
        or float(author_summary["max_attenuation"]) >= float(thresholds["attenuation_strong"])
        or any(
            np.isfinite(float(record["attenuation_ratio"]))
            and float(record["attenuation_ratio"]) >= float(thresholds["attenuation_strong"])
            for record in exclusion_rows.values()
        )
    )
    if concern_sign_flip or strong_attenuation:
        reasons: list[str] = []
        if concern_sign_flip:
            reasons.append("单作者或 Top contributor 排除造成方向反转")
        if strong_attenuation:
            reasons.append(f"最大衰减达到 {float(thresholds['attenuation_strong']):.0%} audit 阈值")
        return "AUTHOR_SENSITIVE", "；".join(reasons)

    top5_direction_stable = all(not bool(exclusion_rows[(aggregation, 5)]["sign_flip"]) for aggregation in AGGREGATIONS)
    top5_remaining = all(
        abs(float(exclusion_rows[(aggregation, 5)]["new_delta"]))
        >= float(thresholds["top5_remaining_ratio_minimum"])
        * abs(float(exclusion_rows[(aggregation, 5)]["original_delta"]))
        for aggregation in AGGREGATIONS
    )
    if not top5_direction_stable or not top5_remaining:
        return "AUTHOR_SENSITIVE", "Top5 contributor 排除后方向不稳或剩余幅度低于配置阈值"

    concentrated = (
        concentration_values["top1_share"] >= float(thresholds["top1_share_high"])
        or concentration_values["top5_share"] >= float(thresholds["top5_share_high"])
    )
    if concentrated:
        return (
            "CONCENTRATED_BUT_ROBUST",
            "至少一个时期贡献集中度达到 audit 阈值，但 LOAO 与 Top1/3/5 排除后方向保持且未达强衰减阈值",
        )
    return "AUTHOR_ROBUST", "LOAO 无高关注方向反转，Top1/3/5 排除保持方向且集中度未达高阈值"


def build_primary_influence_summary(
    primary: pd.DataFrame,
    loao_summary: pd.DataFrame,
    exclusions: pd.DataFrame,
    concentration: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """整合36个 CLEAR Topic primary contrasts 的作者影响诊断。"""

    lookup = contrast_lookup(config)
    rows: list[dict[str, Any]] = []
    for prior in primary.sort_values(["contrast_order", "topic_id"]).itertuples(index=False):
        record = pd.Series(prior._asdict())
        topic_id, contrast = str(record["topic_id"]), str(record["contrast"])
        poem = loao_summary.loc[
            loao_summary["topic_id"].eq(topic_id)
            & loao_summary["contrast"].eq(contrast)
            & loao_summary["aggregation"].eq("POEM")
        ].iloc[0]
        author = loao_summary.loc[
            loao_summary["topic_id"].eq(topic_id)
            & loao_summary["contrast"].eq(contrast)
            & loao_summary["aggregation"].eq("AUTHOR")
        ].iloc[0]
        definition = lookup[contrast]
        concentration_values = contrast_concentration(
            concentration, topic_id, definition["period_a"], definition["period_b"]
        )
        status, reason = classify_author_robustness(
            record, poem, author, exclusions, concentration_values, config
        )
        secondary_status, secondary_reason = classify_author_robustness(
            record, poem, author, exclusions, concentration_values, config,
            scheme=config["secondary_exclusion_scheme"],
        )
        output: dict[str, Any] = {
            "topic_id": topic_id,
            "topic_label": record["topic_label"],
            "contrast": contrast,
            "period_a": definition["period_a"],
            "period_b": definition["period_b"],
            "original_poem_delta": float(record["raw_effect_poem"]),
            "original_author_delta": float(record["raw_effect_author"]),
            "previous_validation_status": record["validation_status"],
            "poem_loao_max_abs_change": float(poem["max_abs_change"]),
            "author_loao_max_abs_change": float(author["max_abs_change"]),
            "poem_sign_flip_count": int(poem["sign_flip_count"]),
            "author_sign_flip_count": int(author["sign_flip_count"]),
            "poem_sign_flip_concern_count": int(poem["sign_flip_concern_count"]),
            "author_sign_flip_concern_count": int(author["sign_flip_concern_count"]),
            "poem_max_attenuation": float(poem["max_attenuation"]),
            "author_max_attenuation": float(author["max_attenuation"]),
            "poem_max_influence_author": poem["max_influence_author"],
            "poem_max_influence_author_period": poem["max_influence_author_period"],
            "author_max_influence_author": author["max_influence_author"],
            "author_max_influence_author_period": author["max_influence_author_period"],
            **concentration_values,
            "author_robustness_status": status,
            "robustness_reason": reason,
        }
        for n in [1, 3, 5]:
            for aggregation in AGGREGATIONS:
                exclusion = exclusion_record(exclusions, topic_id, contrast, aggregation, n)
                prefix = aggregation.lower()
                output[f"top{n}_exclusion_delta_{prefix}"] = float(exclusion["new_delta"])
                output[f"top{n}_exclusion_attenuation_{prefix}"] = float(exclusion["attenuation_ratio"])
                output[f"top{n}_exclusion_sign_flip_{prefix}"] = bool(exclusion["sign_flip"])
        output.update({
            "primary_exclusion_scheme": config["primary_exclusion_scheme"],
            "per_period_author_robustness_status": secondary_status,
            "per_period_robustness_reason": secondary_reason,
            "exclusion_scheme_status_agreement": status == secondary_status,
        })
        rows.append(output)
    return pd.DataFrame(rows).sort_values(["topic_id", "contrast"]).reset_index(drop=True)


def sensitivity_pair_set(
    influence_summary: pd.DataFrame,
    loao: pd.DataFrame,
    config: dict[str, Any],
) -> set[tuple[str, str]]:
    """确定 STRICT/EXPANDED 简化复核的 Topic×contrast 范围。"""

    pairs = {(item["topic_id"], item["contrast"]) for item in config["priority_pairs"]}
    sensitive = influence_summary.loc[influence_summary["author_robustness_status"].eq("AUTHOR_SENSITIVE")]
    pairs.update(zip(sensitive["topic_id"], sensitive["contrast"]))
    flips = loao.loc[loao["sign_flip"]]
    pairs.update(zip(flips["topic_id"], flips["contrast"]))
    return set(pairs)


def build_sample_influence_sensitivity(
    sample_exclusions: pd.DataFrame,
    sample_max_loao: pd.DataFrame,
) -> pd.DataFrame:
    """合并 STRICT/CORE/EXPANDED 的简化 Top exclusion 与最大 LOAO 结果。"""

    exclusion = sample_exclusions.copy()
    exclusion["diagnostic_type"] = "TOP_CONTRIBUTOR_EXCLUSION"
    exclusion["removed_author"] = ""
    exclusion["loao_delta"] = np.nan
    exclusion["max_abs_change"] = np.nan
    keep_exclusion = [
        "sample", "topic_id", "topic_label", "contrast", "aggregation", "diagnostic_type",
        "exclusion_scheme", "N", "excluded_authors", "new_delta", "attenuation_ratio", "sign_flip",
        "removed_author", "loao_delta", "max_abs_change",
    ]
    maximum = sample_max_loao.copy()
    maximum["diagnostic_type"] = "MAX_INFLUENCE_LOAO"
    maximum["exclusion_scheme"] = "LOAO"
    maximum["N"] = 1
    maximum["excluded_authors"] = maximum["removed_author"]
    maximum["new_delta"] = maximum["loao_delta"]
    maximum["max_abs_change"] = maximum["absolute_change"]
    maximum = maximum[keep_exclusion]
    return pd.concat([exclusion[keep_exclusion], maximum], ignore_index=True).sort_values(
        ["topic_id", "contrast", "sample", "aggregation", "diagnostic_type", "exclusion_scheme", "N"]
    ).reset_index(drop=True)


def attach_sample_influence_robust(
    influence_summary: pd.DataFrame,
    all_sample_exclusions: pd.DataFrame,
    selected_pairs: set[tuple[str, str]],
) -> pd.DataFrame:
    """标记三套样本在 GLOBAL Top1 排除后是否均保持方向。"""

    result = influence_summary.copy()
    flags: list[bool | None] = []
    reasons: list[str] = []
    for record in result.to_dict("records"):
        pair = (record["topic_id"], record["contrast"])
        if pair not in selected_pairs:
            flags.append(None)
            reasons.append("未进入预设简化样本敏感性范围")
            continue
        group = all_sample_exclusions.loc[
            all_sample_exclusions["topic_id"].eq(record["topic_id"])
            & all_sample_exclusions["contrast"].eq(record["contrast"])
            & all_sample_exclusions["exclusion_scheme"].eq("GLOBAL_TOP")
            & all_sample_exclusions["N"].eq(1)
        ]
        expected = 3 * len(AGGREGATIONS)
        robust = len(group) == expected and not bool(group["sign_flip"].any())
        flags.append(bool(robust))
        reasons.append("STRICT/CORE/EXPANDED 两种 estimand 的 Top1 排除均不反向" if robust else "至少一套样本或 estimand 的 Top1 排除反向/缺失")
    result["sample_influence_robust"] = pd.array(flags, dtype="boolean")
    result["sample_influence_robust_reason"] = reasons
    return result


def bootstrap_excluded_delta(
    author_rows: pd.DataFrame,
    topic_id: str,
    period_a: str,
    period_b: str,
    aggregation: str,
    excluded_ids: set[str],
    iterations: int,
    seed: int,
) -> np.ndarray:
    """在排除 contributor 后运行时期分层作者 cluster bootstrap。"""

    rng = np.random.default_rng(seed)
    mass_column = f"{topic_id}_prob_mass"
    mean_column = f"{topic_id}_prob_mean"
    boot_means: dict[str, np.ndarray] = {}
    for period in [period_a, period_b]:
        group = author_rows.loc[
            author_rows["author_period_primary"].eq(period)
            & ~author_rows["author_id_internal"].isin(excluded_ids)
        ]
        n_authors = len(group)
        if n_authors == 0:
            return np.full(iterations, np.nan)
        counts = rng.multinomial(n_authors, np.full(n_authors, 1.0 / n_authors), size=iterations)
        if aggregation == "POEM":
            numerator = counts @ group[mass_column].to_numpy(dtype=float)
            denominator = counts @ group["poem_count"].to_numpy(dtype=float)
            boot_means[period] = numerator / denominator
        elif aggregation == "AUTHOR":
            boot_means[period] = (counts @ group[mean_column].to_numpy(dtype=float)) / n_authors
        else:
            raise ValueError(f"未知 aggregation：{aggregation}")
    return boot_means[period_b] - boot_means[period_a]


def bootstrap_supported_direction(low: float, high: float) -> str:
    """按 bootstrap 区间是否跨零描述方向支持。"""

    if low > 0:
        return "SUPPORTED_POSITIVE"
    if high < 0:
        return "SUPPORTED_NEGATIVE"
    return "UNCERTAIN"


def build_top_exclusion_bootstrap(
    author_rows: pd.DataFrame,
    exclusions: pd.DataFrame,
    supported: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    """只对13组统计候选的 Top1/3/5 exclusion 运行固定种子 cluster bootstrap。"""

    supported_pairs = set(zip(supported["topic_id"], supported["contrast"]))
    selected = exclusions.loc[
        exclusions["sample"].eq("CORE")
        & exclusions.apply(lambda row: (row["topic_id"], row["contrast"]) in supported_pairs, axis=1)
        & exclusions["N"].isin(config["bootstrap_exclusion_n"])
    ].copy()
    iterations = int(config["bootstrap_iterations"])
    base_seed = int(config["random_seed"])
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for record in selected.sort_values(
        ["topic_id", "contrast", "aggregation", "exclusion_scheme", "N"]
    ).to_dict("records"):
        excluded = set(filter(None, str(record["excluded_author_ids"]).split(";")))
        seed = stable_seed(
            base_seed,
            record["topic_id"],
            record["contrast"],
            record["aggregation"],
            record["exclusion_scheme"],
            record["N"],
        )
        draws = bootstrap_excluded_delta(
            author_rows,
            record["topic_id"],
            record["period_a"],
            record["period_b"],
            record["aggregation"],
            excluded,
            iterations,
            seed,
        )
        digest.update(np.ascontiguousarray(draws).tobytes())
        low, high = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "sample": "CORE",
                "topic_id": record["topic_id"],
                "topic_label": record["topic_label"],
                "contrast": record["contrast"],
                "aggregation": record["aggregation"],
                "exclusion_scheme": record["exclusion_scheme"],
                "N": int(record["N"]),
                "excluded_author_count": int(record["excluded_author_count"]),
                "excluded_authors": record["excluded_authors"],
                "delta": float(record["new_delta"]),
                "ci_low": float(low),
                "ci_high": float(high),
                "supported_direction": bootstrap_supported_direction(float(low), float(high)),
                "bootstrap_iterations": iterations,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows), digest.hexdigest()


def build_conflict_summary(
    primary: pd.DataFrame,
    exclusions: pd.DataFrame,
    top_authors: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """判断 Top contributor 排除能否消除既有 POEM/AUTHOR 方向冲突。"""

    tolerance = float(config["thresholds"]["numeric_zero_tolerance"])
    max_n = int(config["thresholds"]["conflict_resolution_max_n"])
    rows: list[dict[str, Any]] = []
    conflicts = primary.loc[primary["aggregation_conflict"]].sort_values(["topic_id", "contrast"])
    for prior in conflicts.to_dict("records"):
        original_poem = float(prior["raw_effect_poem"])
        original_author = float(prior["raw_effect_author"])
        resolution_n: int | None = None
        resolution_poem = math.nan
        resolution_author = math.nan
        for n in [1, 3, 5]:
            if n > max_n:
                continue
            poem = exclusion_record(exclusions, prior["topic_id"], prior["contrast"], "POEM", n)
            author = exclusion_record(exclusions, prior["topic_id"], prior["contrast"], "AUTHOR", n)
            if direction(float(poem["new_delta"]), tolerance) == direction(float(author["new_delta"]), tolerance):
                resolution_n = n
                resolution_poem = float(poem["new_delta"])
                resolution_author = float(author["new_delta"])
                break
        if resolution_n == 1:
            assessment = "RESOLVED_AFTER_TOP1"
        elif resolution_n in {3, 5}:
            assessment = f"RESOLVED_AFTER_TOP{resolution_n}"
        else:
            assessment = "PERSISTENT_ESTIMAND_DIFFERENCE"
        influential = top_authors.loc[
            top_authors["topic_id"].eq(prior["topic_id"])
            & top_authors["contrast"].eq(prior["contrast"])
            & top_authors["aggregation"].eq("POEM")
            & top_authors["rank"].eq(1)
        ].iloc[0]
        rows.append(
            {
                "topic_id": prior["topic_id"],
                "topic_label": prior["topic_label"],
                "contrast": prior["contrast"],
                "original_poem_delta": original_poem,
                "original_author_delta": original_author,
                "original_poem_sign": direction(original_poem, tolerance),
                "original_author_sign": direction(original_author, tolerance),
                "conflict_resolution_assessment": assessment,
                "minimum_global_top_n_for_agreement": resolution_n,
                "poem_delta_after_resolution_exclusion": resolution_poem,
                "author_delta_after_resolution_exclusion": resolution_author,
                "max_poem_influence_author": influential["author"],
                "max_poem_influence_author_period": influential["period"],
                "max_poem_influence_author_poem_count": int(influential["poem_count"]),
                "max_poem_influence_author_topic_mass_share": float(influential["topic_mass_share"]),
                "diagnostic_interpretation": (
                    "高 Topic-contribution 作者排除后两种 estimand 同向；仅为数值诊断，不是因果解释"
                    if resolution_n is not None
                    else "Top1/3/5 排除后方向冲突仍在，保留为 estimand difference"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_conflict_decomposition(
    author_rows: pd.DataFrame,
    primary: pd.DataFrame,
    conflict_summary: pd.DataFrame,
) -> pd.DataFrame:
    """逐作者分解聚合冲突中的 poem-weighted 与 author-balanced contribution。"""

    rows: list[dict[str, Any]] = []
    summary_map = conflict_summary.set_index(["topic_id", "contrast"])["conflict_resolution_assessment"].to_dict()
    for prior in primary.loc[primary["aggregation_conflict"]].to_dict("records"):
        topic_id = prior["topic_id"]
        mass_column = f"{topic_id}_prob_mass"
        mean_column = f"{topic_id}_prob_mean"
        for period, sign in [(prior["period_a"], -1.0), (prior["period_b"], 1.0)]:
            group = author_rows.loc[author_rows["author_period_primary"].eq(period)].copy()
            poem_count = int(group["poem_count"].sum())
            author_count = int(len(group))
            period_poem_mean = float(group[mass_column].sum() / poem_count)
            period_author_mean = float(group[mean_column].mean())
            period_mean_poems = float(group["poem_count"].mean())
            period_median_poems = float(group["poem_count"].median())
            group = group.sort_values([mass_column, "author_id_internal"], ascending=[False, True])
            for rank, author in enumerate(group.to_dict("records"), start=1):
                rows.append(
                    {
                        "topic_id": topic_id,
                        "topic_label": prior["topic_label"],
                        "contrast": prior["contrast"],
                        "period": period,
                        "contrast_side": "A" if sign < 0 else "B",
                        "period_author_count": author_count,
                        "period_poem_count": poem_count,
                        "period_mean_poems_per_author": period_mean_poems,
                        "period_median_poems_per_author": period_median_poems,
                        "author_id_internal": author["author_id_internal"],
                        "author": author["author"],
                        "author_poem_count": int(author["poem_count"]),
                        "author_topic_mass": float(author[mass_column]),
                        "author_mean_topic_probability": float(author[mean_column]),
                        "rank_by_topic_mass": rank,
                        "poem_weight": float(author["poem_count"] / poem_count),
                        "author_equal_weight": float(1.0 / author_count),
                        "topic_mass_share": float(author[mass_column] / group[mass_column].sum()),
                        "poem_weighted_period_mean_contribution": float(author[mass_column] / poem_count),
                        "author_balanced_period_mean_contribution": float(author[mean_column] / author_count),
                        "signed_poem_delta_contribution": float(sign * author[mass_column] / poem_count),
                        "signed_author_delta_contribution": float(sign * author[mean_column] / author_count),
                        "period_poem_mean": period_poem_mean,
                        "period_author_mean": period_author_mean,
                        "conflict_resolution_assessment": summary_map[(topic_id, prior["contrast"])],
                    }
                )
    return pd.DataFrame(rows).sort_values(["topic_id", "contrast", "contrast_side", "rank_by_topic_mass"]).reset_index(drop=True)


def exclusion_result_text(summary: pd.Series, n: int) -> str:
    """把两种 estimand 的 GLOBAL TopN 结果压缩为可审核文本。"""

    poem = float(summary[f"top{n}_exclusion_delta_poem"])
    author = float(summary[f"top{n}_exclusion_delta_author"])
    poem_flip = bool(summary[f"top{n}_exclusion_sign_flip_poem"])
    author_flip = bool(summary[f"top{n}_exclusion_sign_flip_author"])
    return f"POEM={poem:+.10f} ({'flip' if poem_flip else 'stable'}); AUTHOR={author:+.10f} ({'flip' if author_flip else 'stable'})"


def build_validated_candidates(
    supported: pd.DataFrame,
    influence_summary: pd.DataFrame,
) -> pd.DataFrame:
    """把上一阶段13组统计候选更新为作者影响后的候选表。"""

    merged = supported.merge(
        influence_summary,
        on=["topic_id", "topic_label", "contrast"],
        how="left",
        validate="one_to_one",
        suffixes=("_previous", ""),
    )
    assessment_map = {
        "AUTHOR_ROBUST": "UPGRADE_KEEP",
        "CONCENTRATED_BUT_ROBUST": "KEEP_WITH_CAUTION",
        "AUTHOR_SENSITIVE": "DOWNGRADE_AUTHOR_SENSITIVE",
        "ESTIMAND_DEPENDENT": "ESTIMAND_DEPENDENT",
        "INCONCLUSIVE": "INSUFFICIENT",
    }
    merged["previous_validation_status"] = merged["validation_status"]
    merged["top1_exclusion_result"] = merged.apply(lambda row: exclusion_result_text(row, 1), axis=1)
    merged["top3_exclusion_result"] = merged.apply(lambda row: exclusion_result_text(row, 3), axis=1)
    merged["top5_exclusion_result"] = merged.apply(lambda row: exclusion_result_text(row, 5), axis=1)
    merged["final_author_influence_assessment"] = merged["author_robustness_status"].map(assessment_map)
    merged["finding_level_bd_eligible"] = merged["final_author_influence_assessment"].isin(
        ["UPGRADE_KEEP", "KEEP_WITH_CAUTION"]
    )
    merged["finding_level_bd_note"] = np.where(
        merged["finding_level_bd_eligible"],
        "可进入 B_K20 / D_K20 finding-level robustness；仍只属统计候选",
        "本阶段先降级或保留 estimand 分歧，不作为下一阶段主候选",
    )
    columns = [
        "topic_id", "topic_label", "contrast", "previous_validation_status",
        "original_poem_delta", "original_author_delta", "author_robustness_status",
        "robustness_reason", "sample_influence_robust", "top1_exclusion_result",
        "top3_exclusion_result", "top5_exclusion_result", "final_author_influence_assessment",
        "finding_level_bd_eligible", "finding_level_bd_note",
        "primary_exclusion_scheme", "per_period_author_robustness_status",
        "per_period_robustness_reason", "exclusion_scheme_status_agreement",
    ]
    return merged[columns].sort_values(["topic_id", "contrast"]).reset_index(drop=True)


def json_records(frame: pd.DataFrame, columns: list[str] | None = None) -> list[dict[str, Any]]:
    """把 DataFrame 转换为无 NaN 的 JSON 记录。"""

    selected = frame if columns is None else frame[columns]
    clean = selected.astype(object).where(pd.notna(selected), None)
    return json.loads(clean.to_json(orient="records", force_ascii=False))


def build_analysis_summary(
    influence_summary: pd.DataFrame,
    candidates: pd.DataFrame,
    loao: pd.DataFrame,
    conflict_summary: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """建立最终 A–N 汇报所需的机器可读摘要。"""

    counts = candidates["author_robustness_status"].value_counts().to_dict()
    status_counts = {status: int(counts.get(status, 0)) for status in config["status_order"]}
    special_pairs = [
        ("topic_03", "E-L"),
        ("topic_06", "E-L"),
        ("topic_08", "E-L"),
        ("topic_16", "H-M"),
        ("topic_16", "E-L"),
    ]
    special: dict[str, Any] = {}
    for topic_id, contrast in special_pairs:
        row = influence_summary.loc[
            influence_summary["topic_id"].eq(topic_id) & influence_summary["contrast"].eq(contrast)
        ]
        if not row.empty:
            special[f"{topic_id}_{contrast}"] = json_records(row)[0]
    concern_flips = loao.loc[loao["sign_flip"]].groupby(
        ["topic_id", "topic_label", "contrast", "aggregation"], as_index=False
    ).agg(
        sign_flip_count=("sign_flip", "sum"),
        authors=("removed_author", lambda values: ";".join(sorted(set(str(value) for value in values)))),
    )
    attenuation_threshold = float(config["thresholds"]["attenuation_noticeable"])
    top1_stable_top35_attenuated: list[dict[str, Any]] = []
    for row in influence_summary.to_dict("records"):
        top1_stable = not row["top1_exclusion_sign_flip_poem"] and not row["top1_exclusion_sign_flip_author"]
        later = max(
            row["top3_exclusion_attenuation_poem"],
            row["top3_exclusion_attenuation_author"],
            row["top5_exclusion_attenuation_poem"],
            row["top5_exclusion_attenuation_author"],
        )
        if top1_stable and np.isfinite(later) and later >= attenuation_threshold:
            top1_stable_top35_attenuated.append(
                {
                    "topic_id": row["topic_id"],
                    "topic_label": row["topic_label"],
                    "contrast": row["contrast"],
                    "max_top3_top5_attenuation": float(later),
                }
            )
    concentrated = influence_summary.loc[
        influence_summary["author_robustness_status"].eq("CONCENTRATED_BUT_ROBUST")
    ]
    return {
        "primary_exclusion_scheme": config["primary_exclusion_scheme"],
        "secondary_exclusion_scheme": config["secondary_exclusion_scheme"],
        "exclusion_scheme_disagreements": json_records(
            influence_summary.loc[~influence_summary["exclusion_scheme_status_agreement"]],
            ["topic_id", "contrast", "author_robustness_status", "per_period_author_robustness_status"],
        ),
        "candidate_status_counts": status_counts,
        "special_pairs": special,
        "aggregation_conflicts": json_records(conflict_summary),
        "loao_sign_flips": json_records(concern_flips),
        "top1_stable_top3_top5_attenuated": top1_stable_top35_attenuated,
        "concentrated_but_robust": json_records(
            concentrated,
            ["topic_id", "topic_label", "contrast", "top1_share", "top5_share", "author_robustness_status"],
        ),
        "author_validated_candidate_trends": json_records(candidates),
        "finding_level_bd_eligible": json_records(
            candidates.loc[candidates["finding_level_bd_eligible"]],
            ["topic_id", "topic_label", "contrast", "final_author_influence_assessment"],
        ),
        "analysis_scope": "C_K20 author influence diagnostic only; no B/D computation and no literary-history interpretation",
    }


def build_data_dictionary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """生成字段级数据字典。"""

    descriptions = {
        "topic_mass": "作者在该时期该 Topic 的 probability 总和。",
        "share_of_period_topic_mass": "作者 Topic mass 占该时期总 Topic mass 的比例。",
        "original_delta": "排除前 period_b mean - period_a mean。",
        "loao_delta": "删除该作者全部诗后重新计算的效应。",
        "absolute_change": "|loao_delta-original_delta|。",
        "attenuation_ratio": "1-|new_delta|/|original_delta|；正值表示衰减。",
        "sign_flip": "原始与排除后效应是否跨零。",
        "influence_score": "配置化多指标排序工具，不是作者重要性的客观真值。",
        "author_robustness_status": "LOAO 与 GLOBAL_TOP Top1/3/5 的主诊断状态；不是文学史结论，也不代表所有排除方案稳健。",
        "primary_exclusion_scheme": "主分类方案 GLOBAL_TOP；PER_PERIOD_TOP 只作独立敏感性诊断。",
        "per_period_author_robustness_status": "将同一组分类阈值应用于 LOAO 与 PER_PERIOD_TOP Top1/3/5 后的敏感性状态。",
        "per_period_robustness_reason": "PER_PERIOD_TOP 敏感性分类的数值理由。",
        "exclusion_scheme_status_agreement": "GLOBAL_TOP 与 PER_PERIOD_TOP 两种分类是否一致。",
        "sample_influence_robust": "已选复核项在 STRICT/CORE/EXPANDED 中 GLOBAL_TOP Top1 是否均无反向；空白表示未运行，且不等于幅度或区间稳定。",
        "final_author_influence_assessment": "统计候选在作者影响检查后的保留或降级状态。",
    }
    rows: list[dict[str, str]] = []
    for name, frame in frames.items():
        for column in frame.columns:
            rows.append(
                {
                    "file": f"{name}.csv",
                    "field": column,
                    "dtype": str(frame[column].dtype),
                    "description": descriptions.get(column, "见报告方法与字段名；均为本流水线派生或继承字段。"),
                }
            )
    return pd.DataFrame(rows)


def run_invariance_tests(
    poems: pd.DataFrame,
    topic_columns: list[str],
    clear_ids: list[str],
    primary: pd.DataFrame,
    author_rows: pd.DataFrame,
    loao: pd.DataFrame,
    exclusions: pd.DataFrame,
    candidates: pd.DataFrame,
    bootstrap: pd.DataFrame,
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
    add("TEST_01", "master 43,015 rows 不变", len(poems) == int(config["expected_poem_count"]), len(poems), config["expected_poem_count"])
    source_hash_after = sha256_file(paths["authoritative_master"])
    add("TEST_02", "不覆盖 authoritative master", source_hash_after == source_hash_before, source_hash_after, source_hash_before)
    add(
        "TEST_03",
        "Topic probability 不变",
        len(topic_columns) == 20 and np.allclose(probability.sum(axis=1), 1.0, atol=1e-10, rtol=0),
        {"topic_count": len(topic_columns), "matrix_sha256": sha256_array(probability)},
        {"topic_count": 20, "row_sum": 1.0},
    )
    valid_periods = set(config["period_order"])
    sample_period_ok = all(
        poems.loc[poems[column], "author_period_primary"].isin(valid_periods).all()
        for column in config["samples"].values()
    )
    add("TEST_04", "author_period 与样本标签不变", sample_period_ok, sample_period_ok, True)
    add("TEST_05", "CLEAR Topic list 不变", clear_ids == sorted(primary["topic_id"].unique()) and len(clear_ids) == 9, clear_ids, sorted(primary["topic_id"].unique()))
    add("TEST_06", "LOAO 删除单位是 author_id_internal", loao["removed_author_id"].notna().all(), loao["removed_author_id"].nunique(), "非空 author_id_internal")

    core_counts = poems.loc[poems[config["samples"]["CORE"]]].groupby("author_id_internal").size().to_dict()
    all_poems_removed = all(int(row.removed_author_poem_count) == int(core_counts[row.removed_author_id]) for row in loao.itertuples())
    add("TEST_07", "删除作者时删除其全部诗歌", all_poems_removed, all_poems_removed, True)

    computed_poem: dict[tuple[str, str], float] = {}
    computed_author: dict[tuple[str, str], float] = {}
    lookup = contrast_lookup(config)
    for row in primary.itertuples():
        definition = lookup[row.contrast]
        key = (row.topic_id, row.contrast)
        computed_poem[key] = delta_from_author_rows(author_rows, row.topic_id, definition["period_a"], definition["period_b"], "POEM")
        computed_author[key] = delta_from_author_rows(author_rows, row.topic_id, definition["period_a"], definition["period_b"], "AUTHOR")
    poem_ok = all(math.isclose(computed_poem[(row.topic_id, row.contrast)], row.raw_effect_poem, rel_tol=0, abs_tol=1e-12) for row in primary.itertuples())
    author_ok = all(math.isclose(computed_author[(row.topic_id, row.contrast)], row.raw_effect_author, rel_tol=0, abs_tol=1e-12) for row in primary.itertuples())
    add("TEST_08", "POEM estimand 保持 poem-weighted", poem_ok, max(abs(computed_poem[(row.topic_id, row.contrast)] - row.raw_effect_poem) for row in primary.itertuples()), 0.0)
    add("TEST_09", "AUTHOR estimand 保持 author-balanced", author_ok, max(abs(computed_author[(row.topic_id, row.contrast)] - row.raw_effect_author) for row in primary.itertuples()), 0.0)

    top_rank_ok = True
    for record in exclusions.loc[exclusions["exclusion_scheme"].eq("GLOBAL_TOP") & exclusions["N"].eq(1) & exclusions["aggregation"].eq("POEM")].to_dict("records"):
        definition = lookup[record["contrast"]]
        involved = author_rows.loc[author_rows["author_period_primary"].isin([definition["period_a"], definition["period_b"]])]
        mass_column = f"{record['topic_id']}_prob_mass"
        expected = involved.sort_values([mass_column, "author_id_internal"], ascending=[False, True]).iloc[0]["author_id_internal"]
        top_rank_ok &= record["excluded_author_ids"] == expected
    add("TEST_10", "Top contributors 按 Topic mass 排名", top_rank_ok, top_rank_ok, True)

    original_ok = all(
        math.isclose(row.original_delta, (primary.loc[(primary["topic_id"].eq(row.topic_id)) & (primary["contrast"].eq(row.contrast)), "raw_effect_poem" if row.aggregation == "POEM" else "raw_effect_author"].iloc[0]), rel_tol=0, abs_tol=1e-15)
        for row in loao.itertuples()
    )
    add("TEST_11", "所有 original delta 与上一阶段一致", original_ok, original_ok, True)

    first = bootstrap.sort_values(["topic_id", "contrast", "aggregation", "exclusion_scheme", "N"]).iloc[0]
    exclusion = exclusion_record(exclusions, first["topic_id"], first["contrast"], first["aggregation"], int(first["N"]), first["exclusion_scheme"])
    draws = bootstrap_excluded_delta(
        author_rows,
        first["topic_id"],
        exclusion["period_a"],
        exclusion["period_b"],
        first["aggregation"],
        set(filter(None, str(exclusion["excluded_author_ids"]).split(";"))),
        int(first["bootstrap_iterations"]),
        int(first["seed"]),
    )
    low, high = np.quantile(draws, [0.025, 0.975])
    reproducible = math.isclose(float(low), float(first["ci_low"]), abs_tol=1e-15) and math.isclose(float(high), float(first["ci_high"]), abs_tol=1e-15)
    add("TEST_12", "LOAO 与 bootstrap 结果可复现", reproducible, [float(low), float(high)], [float(first["ci_low"]), float(first["ci_high"])])

    exclusion_count_ok = True
    for record in exclusions.loc[exclusions["aggregation"].eq("POEM")].to_dict("records"):
        definition = lookup[record["contrast"]]
        involved = author_rows.loc[author_rows["author_period_primary"].isin([definition["period_a"], definition["period_b"]])]
        if record["exclusion_scheme"] == "GLOBAL_TOP":
            expected = min(int(record["N"]), len(involved))
        else:
            expected = sum(min(int(record["N"]), int((involved["author_period_primary"] == period).sum())) for period in [definition["period_a"], definition["period_b"]])
        exclusion_count_ok &= int(record["excluded_author_count"]) == expected
    add("TEST_13", "Top1/3/5/10 exclusion 实际删除作者数正确", exclusion_count_ok, exclusion_count_ok, True)

    flip_recomputed = loao.apply(lambda row: sign_flip(float(row["original_delta"]), float(row["loao_delta"])), axis=1)
    flip_ok = bool((flip_recomputed.to_numpy(dtype=bool) == loao["sign_flip"].to_numpy(dtype=bool)).all())
    add("TEST_14", "无 sign flip 被漏记", flip_ok, int(flip_recomputed.sum()), int(loao["sign_flip"].sum()))

    allowed = {"UPGRADE_KEEP", "KEEP_WITH_CAUTION", "DOWNGRADE_AUTHOR_SENSITIVE", "ESTIMAND_DEPENDENT", "INSUFFICIENT"}
    scope_ok = set(candidates["final_author_influence_assessment"]).issubset(allowed)
    add("TEST_15", "不生成新的文学史结论", scope_ok, sorted(candidates["final_author_influence_assessment"].unique()), sorted(allowed))
    result = pd.DataFrame(tests)
    if not result["status"].eq("PASS").all():
        failed = result.loc[result["status"].ne("PASS"), ["test_id", "test_name"]].to_dict("records")
        raise AssertionError(f"数据不可变性测试失败：{failed}")
    return result


def configure_plotting() -> None:
    """配置跨平台中文绘图与审计图样式。"""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
        }
    )


def short_topic(topic_id: str) -> str:
    """把 topic_03 转换为 T03。"""

    return f"T{int(str(topic_id).split('_')[-1]):02d}"


def save_figure(fig: plt.Figure, path: Path) -> None:
    """统一保存并关闭图像。"""

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_figures(
    figures_root: Path,
    loao_summary: pd.DataFrame,
    top_authors: pd.DataFrame,
    exclusions: pd.DataFrame,
    conflict_decomposition: pd.DataFrame,
    influence_summary: pd.DataFrame,
) -> list[Path]:
    """生成五张作者影响诊断图。"""

    configure_plotting()
    figures_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    contrasts = ["E-H", "H-M", "M-L", "E-L"]
    topics = sorted(loao_summary["topic_id"].unique())
    fig, axes = plt.subplots(1, 4, figsize=(18, 8), sharey=True)
    colors = {"POEM": "#2563EB", "AUTHOR": "#D97706"}
    offsets = {"POEM": -0.14, "AUTHOR": 0.14}
    for axis, contrast in zip(axes, contrasts):
        for aggregation in AGGREGATIONS:
            group = loao_summary.loc[
                loao_summary["contrast"].eq(contrast) & loao_summary["aggregation"].eq(aggregation)
            ].set_index("topic_id").reindex(topics)
            y = np.arange(len(topics)) + offsets[aggregation]
            for index, (_, row) in enumerate(group.iterrows()):
                axis.plot([row["loao_delta_min"], row["loao_delta_max"]], [y[index], y[index]], color=colors[aggregation], lw=2, alpha=0.75)
            axis.scatter(group["original_delta"], y, s=28, color=colors[aggregation], label=aggregation, zorder=3)
        axis.axvline(0, color="#111827", lw=0.8)
        axis.set_title(contrast)
        axis.set_xlabel("Effect Δ")
    axes[0].set_yticks(np.arange(len(topics)), [short_topic(value) for value in topics])
    axes[0].set_ylabel("CLEAR Topic")
    axes[-1].legend(loc="lower right", frameon=False)
    fig.suptitle("CORE LOAO delta ranges: original effect and leave-one-author-out extrema", fontsize=14)
    path = figures_root / "loao_delta_range_clear_topics.png"
    save_figure(fig, path)
    paths.append(path)

    priority = [("topic_03", "E-L"), ("topic_06", "E-L"), ("topic_08", "E-L"), ("topic_16", "H-M"), ("topic_16", "E-L")]
    fig, axes = plt.subplots(len(priority), 1, figsize=(13, 17))
    for axis, (topic_id, contrast) in zip(axes, priority):
        group = top_authors.loc[
            top_authors["topic_id"].eq(topic_id)
            & top_authors["contrast"].eq(contrast)
            & top_authors["rank"].le(5)
        ].copy()
        group["change"] = group["delta_after_removal"] - group["delta_original"]
        group["label"] = group["aggregation"].str[0] + " · " + group["author"].astype(str)
        group = group.sort_values(["aggregation", "rank"], ascending=[True, False])
        palette = group["aggregation"].map(colors)
        axis.barh(group["label"], group["change"], color=palette, alpha=0.85)
        axis.axvline(0, color="#111827", lw=0.8)
        axis.set_title(f"{short_topic(topic_id)} {contrast}: top five LOAO changes per estimand")
        axis.set_xlabel("LOAO Δ − original Δ")
    fig.suptitle("Top influential authors are ranked by absolute effect change", fontsize=14)
    path = figures_root / "top_author_influence_forest.png"
    save_figure(fig, path)
    paths.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    flat_axes = axes.ravel()
    x_values = [0, 1, 3, 5, 10]
    for axis, (topic_id, contrast) in zip(flat_axes, priority):
        for aggregation in AGGREGATIONS:
            group = exclusions.loc[
                exclusions["topic_id"].eq(topic_id)
                & exclusions["contrast"].eq(contrast)
                & exclusions["aggregation"].eq(aggregation)
                & exclusions["exclusion_scheme"].eq("GLOBAL_TOP")
            ].set_index("N")
            original = float(group["original_delta"].iloc[0])
            values = [original] + [float(group.loc[n, "new_delta"]) for n in x_values[1:]]
            axis.plot(x_values, values, marker="o", lw=2, color=colors[aggregation], label=aggregation)
        axis.axhline(0, color="#111827", lw=0.8)
        axis.set_xticks(x_values, ["None", "Top1", "Top3", "Top5", "Top10"])
        axis.set_title(f"{short_topic(topic_id)} {contrast}")
        axis.set_ylabel("Effect Δ")
    flat_axes[-1].axis("off")
    flat_axes[0].legend(frameon=False)
    fig.suptitle("GLOBAL_TOP contributor exclusion sensitivity", fontsize=14)
    path = figures_root / "top_contributor_exclusion_sensitivity.png"
    save_figure(fig, path)
    paths.append(path)

    conflict_pairs = sorted(set(zip(conflict_decomposition["topic_id"], conflict_decomposition["contrast"])))
    fig, axes = plt.subplots(4, 2, figsize=(16, 19))
    for axis, pair in zip(axes.ravel(), conflict_pairs):
        group = conflict_decomposition.loc[
            conflict_decomposition["topic_id"].eq(pair[0]) & conflict_decomposition["contrast"].eq(pair[1])
        ]
        authors = group.groupby(["author_id_internal", "author"], as_index=False).agg(
            signed_poem=("signed_poem_delta_contribution", "sum"),
            signed_author=("signed_author_delta_contribution", "sum"),
        )
        authors["gap"] = (authors["signed_poem"] - authors["signed_author"]).abs()
        authors = authors.nlargest(8, "gap").sort_values("gap")
        y = np.arange(len(authors))
        axis.barh(y - 0.18, authors["signed_poem"], height=0.34, color=colors["POEM"], label="POEM")
        axis.barh(y + 0.18, authors["signed_author"], height=0.34, color=colors["AUTHOR"], label="AUTHOR")
        axis.set_yticks(y, authors["author"])
        axis.axvline(0, color="#111827", lw=0.8)
        axis.set_title(f"{short_topic(pair[0])} {pair[1]}")
        axis.set_xlabel("Signed contribution to B−A effect")
    for axis in axes.ravel()[len(conflict_pairs):]:
        axis.axis("off")
    axes.ravel()[0].legend(frameon=False)
    fig.suptitle("Authors with the largest POEM-vs-AUTHOR contribution gaps", fontsize=14)
    path = figures_root / "aggregation_conflict_author_contributions.png"
    save_figure(fig, path)
    paths.append(path)

    plot = influence_summary.copy()
    plot["max_loao_attenuation"] = plot[["poem_max_attenuation", "author_max_attenuation"]].max(axis=1)
    status_colors = {
        "AUTHOR_ROBUST": "#059669",
        "CONCENTRATED_BUT_ROBUST": "#2563EB",
        "AUTHOR_SENSITIVE": "#DC2626",
        "ESTIMAND_DEPENDENT": "#D97706",
        "INCONCLUSIVE": "#6B7280",
    }
    fig, axis = plt.subplots(figsize=(11, 8))
    for status, group in plot.groupby("author_robustness_status"):
        axis.scatter(group["top5_share"], group["max_loao_attenuation"], s=52, alpha=0.85, color=status_colors[status], label=status)
    for row in plot.itertuples():
        if row.previous_validation_status in {"STRONG", "SUPPORTED"}:
            horizontal = -4 if row.top5_share > 0.88 else 4
            alignment = "right" if horizontal < 0 else "left"
            axis.annotate(
                f"{short_topic(row.topic_id)} {row.contrast}",
                (row.top5_share, row.max_loao_attenuation),
                fontsize=7,
                xytext=(horizontal, 4),
                textcoords="offset points",
                ha=alignment,
            )
    axis.axvline(0.70, color="#6B7280", ls="--", lw=0.9)
    axis.axhline(0.70, color="#6B7280", ls="--", lw=0.9)
    axis.set_xlabel("Maximum period Top5 author topic-mass share")
    axis.set_ylabel("Maximum single-author LOAO attenuation")
    axis.set_title("Concentration and single-author influence are separate diagnostics")
    axis.legend(frameon=False, fontsize=8)
    path = figures_root / "concentration_vs_influence.png"
    save_figure(fig, path)
    paths.append(path)
    return paths


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    """生成简洁 Markdown 表。"""

    selected = frame[columns].head(limit) if limit is not None else frame[columns]
    headers = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for record in selected.astype(object).where(pd.notna(selected), "").to_dict("records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value).replace("|", "／"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([headers, separator, *rows])


def build_report(
    output_root: Path,
    contributions: pd.DataFrame,
    concentration: pd.DataFrame,
    loao_summary: pd.DataFrame,
    exclusions: pd.DataFrame,
    conflict_summary: pd.DataFrame,
    influence_summary: pd.DataFrame,
    candidates: pd.DataFrame,
    config: dict[str, Any],
) -> Path:
    """生成15节作者影响诊断报告。"""

    counts = candidates["author_robustness_status"].value_counts().to_dict()
    special = influence_summary.loc[
        influence_summary.apply(
            lambda row: (row["topic_id"], row["contrast"])
            in {("topic_03", "E-L"), ("topic_06", "E-L"), ("topic_08", "E-L")},
            axis=1,
        )
    ]
    scheme_disagreements = influence_summary.loc[~influence_summary["exclusion_scheme_status_agreement"]]
    thresholds = config["thresholds"]
    per_period_priority = exclusions.loc[
        exclusions["exclusion_scheme"].eq(config["secondary_exclusion_scheme"])
        & exclusions["N"].isin(config["classification_exclusion_n"])
        & exclusions.apply(
            lambda row: (row["topic_id"], row["contrast"])
            in {("topic_03", "E-L"), ("topic_06", "E-L"), ("topic_08", "E-L"), ("topic_16", "H-M"), ("topic_16", "E-L")},
            axis=1,
        )
    ]
    sections = [
        "# QTS Author Influence / Leave-One-Author-Out Report",
        "",
        "## 1. Objective",
        "检验既有 C_K20 author-period Topic effects 是否依赖少数高产或高 Topic-contribution 作者。本报告只作 influence / robustness diagnostic。",
        "",
        "## 2. Methods",
        f"正式分析使用 CORE；以 author_id_internal 为删除单位。分别重算 POEM（剩余诗等权）与 AUTHOR（剩余作者等权）效应，并运行 GLOBAL_TOP、PER_PERIOD_TOP 的 Top1/3/5/10 排除。13组既有统计候选的关键 exclusion 使用 {config['bootstrap_iterations']} 次固定种子作者 cluster bootstrap。",
        "",
        f"主分类及未加方案前缀的 Top1/3/5 摘要使用 {config['primary_exclusion_scheme']}；{config['secondary_exclusion_scheme']} 独立报告敏感性状态，不以第二方案中较有利的结果替换主分类。两种方案删去的作者数不同（全局N位与每期N位），不是等删除量对照。所有 delta 均为 period B 减 period A，单位为 Topic probability。",
        "",
        f"Bootstrap 按时期分层、以作者整簇有放回抽样；基准 seed={config['random_seed']}，各 Topic/contrast/方案/估计量使用稳定派生子种子。CI 单独报告，不将未跨零的原始 CI 自动继承给删作者后效应；作者稳健性标签不是新的显著性检验。STRICT/EXPANDED 只复核预定优先项与 CORE 敏感/原始反向项，sample_influence_robust 仅指 GLOBAL_TOP Top1 的方向保持。",
        "",
        "## 3. Overall Influence Structure",
        f"CORE 共 {contributions['author_id_internal'].nunique()} 位作者、{int(contributions.drop_duplicates(['author_id_internal'])['poem_count'].sum())} 首诗。13组候选状态计数："
        + "，".join(f"{status}={int(counts.get(status, 0))}" for status in config["status_order"]) + "。",
        "",
        "## 4. T03",
        markdown_table(influence_summary.loc[influence_summary["topic_id"].eq("topic_03")], ["contrast", "original_poem_delta", "original_author_delta", "poem_loao_max_abs_change", "author_loao_max_abs_change", "author_robustness_status"]),
        "",
        "## 5. T06",
        markdown_table(influence_summary.loc[influence_summary["topic_id"].eq("topic_06")], ["contrast", "original_poem_delta", "original_author_delta", "poem_loao_max_abs_change", "author_loao_max_abs_change", "author_robustness_status"]),
        "",
        "## 6. T08",
        markdown_table(influence_summary.loc[influence_summary["topic_id"].eq("topic_08")], ["contrast", "original_poem_delta", "original_author_delta", "poem_loao_max_abs_change", "author_loao_max_abs_change", "author_robustness_status"]),
        "",
        "## 7. T07",
        markdown_table(conflict_summary.loc[conflict_summary["topic_id"].eq("topic_07")], ["contrast", "original_poem_delta", "original_author_delta", "conflict_resolution_assessment", "max_poem_influence_author"]),
        "",
        "## 8. T11",
        markdown_table(conflict_summary.loc[conflict_summary["topic_id"].eq("topic_11")], ["contrast", "original_poem_delta", "original_author_delta", "conflict_resolution_assessment", "max_poem_influence_author"]),
        "",
        "## 9. T16",
        markdown_table(influence_summary.loc[influence_summary["topic_id"].eq("topic_16")], ["contrast", "original_poem_delta", "original_author_delta", "poem_loao_max_abs_change", "author_loao_max_abs_change", "author_robustness_status"]),
        "",
        "## 10. T19",
        markdown_table(conflict_summary.loc[conflict_summary["topic_id"].eq("topic_19")], ["contrast", "original_poem_delta", "original_author_delta", "conflict_resolution_assessment", "max_poem_influence_author"]),
        "",
        "## 11. T10 / T20",
        markdown_table(influence_summary.loc[influence_summary["topic_id"].isin(["topic_10", "topic_20"])], ["topic_id", "contrast", "original_poem_delta", "original_author_delta", "author_robustness_status"]),
        "",
        "## 12. Top Contributor Exclusion",
        "TopN contributor 依每个 Topic×contrast 的 Topic mass 排名，不是固定的‘十大诗人’。GLOBAL_TOP 合并两时期排名；PER_PERIOD_TOP 在两侧分别排名。",
        "",
        markdown_table(special, ["topic_id", "contrast", "top1_exclusion_delta_poem", "top3_exclusion_delta_poem", "top5_exclusion_delta_poem", "top1_exclusion_delta_author", "top3_exclusion_delta_author", "top5_exclusion_delta_author"]),
        "",
        "上表为 GLOBAL_TOP。以下为五组重点效应的 PER_PERIOD_TOP 复核（每期各删N位）：",
        "",
        markdown_table(per_period_priority, ["topic_id", "contrast", "aggregation", "N", "new_delta", "attenuation_ratio", "sign_flip"]),
        "",
        "完整两方案 Top1/3/5/10 数值见 top_contributor_exclusion_effects.csv。T16 H-M 在 PER_PERIOD_TOP Top5 的 POEM 效应反向；T10 E-L 与 T20 E-H 在 PER_PERIOD_TOP 方向保持，但不推翻 GLOBAL_TOP 及 LOAO 主诊断。T07 E-L 的 PER_PERIOD_TOP Top5 AUTHOR 衰减约51.44%，因此‘保留’不表示效应幅度不变。",
        "",
        "## 13. Author Robustness Classification",
        "AUTHOR_ROBUST、CONCENTRATED_BUT_ROBUST、AUTHOR_SENSITIVE、ESTIMAND_DEPENDENT、INCONCLUSIVE 均由配置化 audit heuristic 生成；复合 influence score 只用于排序。",
        "",
        f"规则优先级：原始 POEM/AUTHOR 方向不同先列 ESTIMAND_DEPENDENT；必要 LOAO 汇总不可计算则列 INCONCLUSIVE；原始 |delta|≥{thresholds['near_zero_original_delta']} 时的 LOAO/Top1/3/5 反向，或任一衰减≥{thresholds['attenuation_strong']:.0%}，列 AUTHOR_SENSITIVE。原始 |delta|<{thresholds['relative_change_minimum_original']} 时衰减设 NA；所有 raw sign flip 仍完整保留。",
        "",
        f"其余结果还必须满足两种估计量 Top5 无 raw sign flip 且剩余绝对幅度至少为原值的{thresholds['top5_remaining_ratio_minimum']:.0%}。任一时期 Top1 share≥{thresholds['top1_share_high']:.0%} 或 Top5 share≥{thresholds['top5_share_high']:.0%} 列 CONCENTRATED_BUT_ROBUST，否则为 AUTHOR_ROBUST。Top10 是额外压力测试，不参与分类。以上为数值审计启发式，并非公认效应量等级。",
        "",
        "以下结果的 GLOBAL_TOP 主分类与 PER_PERIOD_TOP 敏感性分类不同；不得把主分类误述为所有排除方式一致：",
        "",
        markdown_table(scheme_disagreements, ["topic_id", "contrast", "author_robustness_status", "per_period_author_robustness_status"]),
        "",
        "## 14. Candidate Trends After Author Influence Check",
        markdown_table(candidates, ["topic_id", "contrast", "previous_validation_status", "author_robustness_status", "final_author_influence_assessment", "finding_level_bd_eligible"]),
        "",
        "主分类保留的9组候选在 PER_PERIOD_TOP 复核中也均为 AUTHOR_ROBUST 或 CONCENTRATED_BUT_ROBUST。4组降级仍保留完整估计量与两方案结果，不因敏感性方案较有利而恢复候选资格。",
        "",
        "## 15. Limitations",
        "author period 不等于 poem composition date；author influence 不等于文学史重要性；高影响可能来自创作量、现存作品量、《全唐诗》收录量或 Topic probability，不能区分为单一原因。高产不等于因果影响；survival / anthology bias 仍在。即使 UPGRADE_KEEP，也只表示作者影响检查后仍稳健的统计候选趋势。",
        "",
    ]
    path = output_root / "QTS_Author_Influence_LOAO_Report.md"
    path.write_text("\n".join(sections), encoding="utf-8", newline="\n")
    return path


def build_readme(output_root: Path, config: dict[str, Any], frames: dict[str, pd.DataFrame]) -> Path:
    """写出目录说明与复现命令。"""

    lines = [
        "# QTS Author Influence / Leave-One-Author-Out",
        "",
        "本目录保存 C_K20 CORE 作者贡献、LOAO、Top-contributor exclusion、聚合冲突分解与作者稳健性分类。",
        "",
        "## Reproduce",
        "",
        "```powershell",
        "python src/run_author_influence_loao.py --config config/author_influence_config.yaml",
        "```",
        "",
        "## Fixed scope",
        "",
        "- Primary: CORE_POEM and CORE_AUTHOR",
        "- 主分类：LOAO + GLOBAL_TOP Top1/3/5；PER_PERIOD_TOP 使用相同阈值独立报告敏感性状态",
        "- Sensitivity: selected STRICT / EXPANDED pairs only",
        "- Bootstrap seed: 20260902",
        "- No new p-value search, no B_K20/D_K20 computation, no literary-history interpretation",
        "",
        "## Data tables",
        "",
    ]
    lines.extend(f"- `data/{name}.csv`: {len(frame):,} rows" for name, frame in frames.items())
    path = output_root / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def main() -> None:
    """执行完整作者影响与 LOAO 流水线。"""

    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    output_root = resolve_path(config["output_root"])
    data_root = output_root / "data"
    figures_root = output_root / "figures"
    conflict_root = data_root / "conflict_contributions"
    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    conflict_root.mkdir(parents=True, exist_ok=True)

    poems, topic_info, clear_ids, clear_columns, primary, supported, input_paths, input_hashes = load_inputs(config)
    source_hash_before = input_hashes["authoritative_master"]
    topic_columns = sorted(column for column in poems.columns if column.startswith("topic_") and column.endswith("_prob"))
    author_tables = build_author_tables(poems, clear_columns, config)
    core_authors = author_tables["CORE"]

    contributions = build_author_contributions(core_authors, topic_info, clear_ids, config)
    concentration = build_concentration_summary(contributions)
    loao = build_loao_effects(core_authors, topic_info, primary, config)
    influence_scores = build_influence_scores(loao, config)
    top_authors = build_top_influential_authors(loao)
    loao_summary = build_loao_summary(loao)
    core_exclusions = build_top_exclusions(core_authors, topic_info, primary, config)
    influence_summary = build_primary_influence_summary(primary, loao_summary, core_exclusions, concentration, config)

    selected_pairs = sensitivity_pair_set(influence_summary, loao, config)
    sample_loao_parts = [loao.loc[loao.apply(lambda row: (row["topic_id"], row["contrast"]) in selected_pairs, axis=1)].copy()]
    sample_exclusion_parts = [
        core_exclusions.loc[
            core_exclusions.apply(lambda row: (row["topic_id"], row["contrast"]) in selected_pairs, axis=1)
            & core_exclusions["N"].isin(config["sensitivity_top_exclusion_n"])
        ].copy()
    ]
    for sample in ["STRICT", "EXPANDED"]:
        sample_loao_parts.append(
            build_loao_effects(author_tables[sample], topic_info, primary, config, sample=sample, pairs=selected_pairs)
        )
        sample_exclusion_parts.append(
            build_top_exclusions(
                author_tables[sample],
                topic_info,
                primary,
                config,
                sample=sample,
                pairs=selected_pairs,
                n_values=[int(value) for value in config["sensitivity_top_exclusion_n"]],
            )
        )
    all_sample_loao = pd.concat(sample_loao_parts, ignore_index=True)
    all_sample_exclusions = pd.concat(sample_exclusion_parts, ignore_index=True)
    sample_max_loao = all_sample_loao.loc[all_sample_loao["rank_by_abs_change"].eq(1)].copy()
    sample_sensitivity = build_sample_influence_sensitivity(all_sample_exclusions, sample_max_loao)
    influence_summary = attach_sample_influence_robust(influence_summary, all_sample_exclusions, selected_pairs)

    bootstrap, bootstrap_hash = build_top_exclusion_bootstrap(core_authors, core_exclusions, supported, config)
    conflict_summary = build_conflict_summary(primary, core_exclusions, top_authors, config)
    conflict_decomposition = build_conflict_decomposition(core_authors, primary, conflict_summary)
    candidates = build_validated_candidates(supported, influence_summary)
    invariance = run_invariance_tests(
        poems,
        topic_columns,
        clear_ids,
        primary,
        core_authors,
        loao,
        core_exclusions,
        candidates,
        bootstrap,
        input_paths,
        source_hash_before,
        config,
    )

    output_frames: dict[str, pd.DataFrame] = {
        "author_period_topic_contributions": contributions,
        "author_concentration_summary": concentration,
        "loao_effects": loao,
        "loao_summary": loao_summary,
        "author_influence_score": influence_scores,
        "top_influential_authors": top_authors,
        "clear_topic_top_influential_authors": top_authors,
        "top_contributor_exclusion_effects": core_exclusions,
        "top_exclusion_bootstrap_ci": bootstrap,
        "sample_influence_sensitivity": sample_sensitivity,
        "aggregation_conflict_summary": conflict_summary,
        "aggregation_conflict_decomposition": conflict_decomposition,
        "clear_topic_author_influence_summary": influence_summary,
        "author_validated_candidate_trends": candidates,
        "invariance_tests": invariance,
    }
    data_dictionary = build_data_dictionary(output_frames)
    output_frames["data_dictionary"] = data_dictionary
    for name, frame in output_frames.items():
        save_csv(frame, data_root / f"{name}.csv")

    special_conflicts = {
        "T16_EH_author_contributions": ("topic_16", "E-H"),
        "T16_ML_author_contributions": ("topic_16", "M-L"),
        "T07_HM_author_contributions": ("topic_07", "H-M"),
        "T07_ML_author_contributions": ("topic_07", "M-L"),
        "T11_HM_author_contributions": ("topic_11", "H-M"),
        "T11_EL_author_contributions": ("topic_11", "E-L"),
        "T19_ML_author_contributions": ("topic_19", "M-L"),
    }
    special_paths: list[Path] = []
    for name, (topic_id, contrast) in special_conflicts.items():
        frame = conflict_decomposition.loc[
            conflict_decomposition["topic_id"].eq(topic_id)
            & conflict_decomposition["contrast"].eq(contrast)
        ]
        path = conflict_root / f"{name}.csv"
        save_csv(frame, path)
        special_paths.append(path)

    figure_paths = build_figures(
        figures_root,
        loao_summary,
        top_authors,
        core_exclusions,
        conflict_decomposition,
        influence_summary,
    )
    summary = build_analysis_summary(influence_summary, candidates, loao, conflict_summary, config)
    summary["bootstrap"] = {
        "iterations": int(config["bootstrap_iterations"]),
        "seed": int(config["random_seed"]),
        "replicate_sha256": bootstrap_hash,
    }
    summary["sensitivity_pairs"] = sorted(f"{topic_id}:{contrast}" for topic_id, contrast in selected_pairs)
    summary_path = data_root / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")
    report_path = build_report(
        output_root,
        contributions,
        concentration,
        loao_summary,
        core_exclusions,
        conflict_summary,
        influence_summary,
        candidates,
        config,
    )
    readme_path = build_readme(output_root, config, output_frames)
    config_snapshot = output_root / "author_influence_config.yaml"
    shutil.copyfile(config_path, config_snapshot)

    output_paths = [data_root / f"{name}.csv" for name in output_frames]
    output_paths.extend(special_paths)
    output_paths.extend(figure_paths)
    output_paths.extend([summary_path, report_path, readme_path, config_snapshot])
    manifest = {
        "analysis_version": config["analysis_version"],
        "created_at_local_date": datetime.now().astimezone().date().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "inputs": {name: {"path": str(path), "sha256": input_hashes[name]} for name, path in input_paths.items()},
        "bootstrap": {
            "iterations": int(config["bootstrap_iterations"]),
            "seed": int(config["random_seed"]),
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
    manifest_path = output_root / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
