"""仅验证冻结 C findings 的 B–C–D 跨模型稳健性，不拟合或选择模型。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AGGREGATIONS = ("POEM", "AUTHOR")
MODELS = ("B", "C", "D")
MAPPING_TYPES = ("DIRECT", "BUNDLE", "MERGED_PROXY", "WEAK_PROXY", "NOT_TESTABLE")


def resolve(value: str | Path) -> Path:
    """把配置路径解析到仓库内。"""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    """分块计算文件哈希。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed: int, *parts: Any) -> int:
    """派生与遍历顺序无关的固定随机种子。"""
    payload = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def bool_series(values: pd.Series, name: str) -> pd.Series:
    """严格转换冻结布尔值，拒绝把缺失当作已通过。"""
    lookup = {"true": True, "false": False, "1": True, "0": False}
    converted = values.astype(str).str.lower().map(lookup)
    if converted.isna().any():
        raise ValueError(f"布尔字段含非法值：{name}")
    return converted.astype(bool)


def finding_id(topic: str, contrast: str) -> str:
    """从来源 Topic 与 contrast 构造稳定标识。"""
    return f"T{int(topic.split('_')[-1]):02d}_{contrast.replace('-', '')}"


def sign(value: float, tolerance: float = 1e-12) -> str:
    """数值符号与 bootstrap 支持分开保存。"""
    if not np.isfinite(value):
        return "NOT_TESTABLE"
    if abs(value) <= tolerance:
        return "NEAR_ZERO"
    return "POSITIVE" if value > 0 else "NEGATIVE"


def supported(low: float, high: float) -> str:
    """仅在区间不跨零时标记支持方向。"""
    if not np.isfinite(low + high):
        return "NOT_TESTABLE"
    if low > 0:
        return "SUPPORTED_POSITIVE"
    if high < 0:
        return "SUPPORTED_NEGATIVE"
    return "UNCERTAIN"


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """按前阶段口径计算作者均值分布的 B-A Hedges g。"""
    if len(a) < 2 or len(b) < 2:
        return math.nan
    variance = ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2)
    delta = float(np.mean(b) - np.mean(a))
    if variance <= 0:
        return 0.0 if abs(delta) <= 1e-15 else math.nan
    return delta / math.sqrt(variance) * (1 - 3 / (4 * (len(a) + len(b)) - 9))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """直接计数作者均值的 B-A 优势概率，平局贡献为零。"""
    if len(a) == 0 or len(b) == 0:
        return math.nan
    return float(np.sign(b[:, None] - a[None, :]).mean())


def magnitude(value: float, kind: str, thresholds: dict[str, float]) -> str:
    """沿用前阶段标准化效应量分档。"""
    if not np.isfinite(value):
        return "UNAVAILABLE"
    keys = ["hedges_small", "hedges_moderate", "hedges_large"] if kind == "hedges" else ["cliff_small", "cliff_medium", "cliff_large"]
    labels = ["VERY_SMALL", "SMALL", "MODERATE", "LARGE"] if kind == "hedges" else ["NEGLIGIBLE", "SMALL", "MEDIUM", "LARGE"]
    return labels[sum(abs(value) >= thresholds[key] for key in keys)]


def mapping_type(relation: str, confidence: str) -> str:
    """只按来源关系决定可测试性，不用 Topic 编号猜测对应。"""
    if relation == "strong_1to1" or (relation == "moderate_1to1" and confidence in {"HIGH", "MEDIUM"}):
        return "DIRECT"
    return {"split": "BUNDLE", "merge": "MERGED_PROXY", "weak": "WEAK_PROXY", "unmatched": "NOT_TESTABLE", "moderate_1to1": "WEAK_PROXY"}.get(relation, "NOT_TESTABLE")


def source_best(ids: list[str], details: str) -> str:
    """只在既有目标集合中取来源 cosine 最大者作为单一目标敏感性。"""
    scores = {match[0]: float(match[1]) for match in re.findall(r"(topic_\d+):cos=([0-9.eE+-]+)", details)}
    return max(ids, key=lambda item: scores.get(item, -math.inf)) if ids else ""


def build_mapping(candidates: pd.DataFrame, alignment: pd.DataFrame) -> pd.DataFrame:
    """生成 finding×目标模型审计表，明确缺失的人工审核状态。"""
    if alignment["C_topic_id"].duplicated().any():
        raise ValueError("来源 alignment 的 C Topic 不唯一")
    source = alignment.set_index("C_topic_id")
    rows = []
    for finding in candidates.to_dict("records"):
        if finding["topic_id"] not in source.index:
            raise ValueError(f"来源 alignment 缺少 {finding['topic_id']}")
        record = source.loc[finding["topic_id"]]
        for model in ("B", "D"):
            ids = [x for x in str(record[f"{model}_topic_id"]).split(";") if x and x != "nan"]
            relation = str(record[f"{model}_alignment_type"])
            confidence = str(record[f"{model}_alignment_confidence"])
            testability = mapping_type(relation, confidence)
            best = source_best(ids, str(record.get(f"{model}_candidate_details", "")))
            status = str(record.get(f"{model}_human_review_status", record.get("human_review_status", "NOT_RECORDED_IN_SOURCE")))
            raw_review = record.get(f"{model}_NEEDS_HUMAN_REVIEW", record.get("NEEDS_HUMAN_REVIEW", False))
            review = str(raw_review).lower() == "true" or "NEEDS_HUMAN_REVIEW" in status or confidence == "LOW"
            analysis_ids = [] if testability == "NOT_TESTABLE" else ([best] if testability == "WEAK_PROXY" else ids)
            if testability in {"DIRECT", "MERGED_PROXY"} and len(ids) != 1:
                raise ValueError(f"{model}/{finding['finding_id']} 的单一代理映射含多个目标")
            if testability == "BUNDLE" and len(ids) < 2:
                raise ValueError("split 必须具有至少两个明确来源目标")
            overlap = []
            for other in alignment.to_dict("records"):
                if other["C_topic_id"] != finding["topic_id"] and set(ids) & set(str(other[f"{model}_topic_id"]).split(";")):
                    overlap.append(other["C_topic_id"])
            rows.append({
                "finding_id": finding["finding_id"], "c_topic_id": finding["topic_id"], "c_topic_label": finding["topic_label"], "contrast": finding["contrast"],
                "target_model": model, "target_topic_ids": ";".join(ids), "target_topic_labels": record[f"{model}_original_label"],
                "alignment_relation_type": relation, "alignment_confidence": confidence, "semantic_similarity": float(record[f"{model}_similarity"]),
                "semantic_notes": record["notes"], "mapping_testability": testability, "analysis_topic_ids": ";".join(analysis_ids),
                "single_best_topic_id": best, "single_best_selection_rule": "per-target confidence absent; maximum source candidate cosine; source-order tie break",
                "human_review_status": status, "alignment_review_required": review,
                "alignment_review_flag": "ALIGNMENT_REVIEW_REQUIRED" if review else "SOURCE_REVIEW_STATUS_NOT_RECORDED" if status == "NOT_RECORDED_IN_SOURCE" else "",
                "overlapping_other_c_topics": ";".join(overlap), "source_candidate_details": record.get(f"{model}_candidate_details", ""),
                "probability_rule": "UNWEIGHTED_SUM" if testability == "BUNDLE" else "SOURCE_BEST_WEAK_PROXY" if testability == "WEAK_PROXY" else "NONE" if testability == "NOT_TESTABLE" else "SINGLE_TARGET",
            })
    return pd.DataFrame(rows)


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """加载冻结来源并以 poem_id 唯一连接历史标签。"""
    names = ("authoritative_master", "sample_labels", "alignment", "eligible_findings", "original_effects", "original_clr")
    paths = {name: resolve(config[name]) for name in names}
    freeze = resolve(config["freeze_root"])
    for model in MODELS:
        for path in sorted((freeze / f"{model}_K20").iterdir()):
            if path.is_file():
                paths[f"freeze_{model}_{path.name}"] = path
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes["authoritative_master"] != config["expected_master_sha256"]:
        raise ValueError("权威历史 master 哈希与冻结值不一致")
    master = pd.read_csv(paths["authoritative_master"], low_memory=False)
    labels = pd.read_csv(paths["sample_labels"], low_memory=False)
    if master["poem_id"].duplicated().any() or labels["poem_id"].duplicated().any():
        raise ValueError("历史 master 与 sample labels 的 poem_id 必须唯一")
    label_columns = ["poem_id", "author_id_internal", *config["samples"].values()]
    metadata_columns = ["poem_id", "author", "title", "author_id_internal", "author_period_primary", "period_confidence", "cross_period_flag", "period_boundary_flag"]
    metadata = master[metadata_columns].merge(labels[label_columns], on="poem_id", how="left", validate="one_to_one", suffixes=("", "_label"))
    if not metadata["author_id_internal"].eq(metadata["author_id_internal_label"]).all():
        raise ValueError("poem_id 连接后作者身份不一致")
    metadata = metadata.drop(columns="author_id_internal_label")
    for field in config["samples"].values():
        metadata[field] = bool_series(metadata[field], field)
        if not metadata.loc[metadata[field], "author_period_primary"].isin(config["period_order"]).all():
            raise ValueError(f"{field} 中存在非法时期")
    original_candidates = pd.read_csv(paths["eligible_findings"], keep_default_na=False)
    eligibility = bool_series(original_candidates["finding_level_bd_eligible"], "finding_level_bd_eligible")
    original_candidates["finding_id"] = [finding_id(t, c) for t, c in zip(original_candidates["topic_id"], original_candidates["contrast"])]
    if original_candidates["finding_id"].duplicated().any():
        raise ValueError("来源 finding 不唯一")
    candidates = original_candidates.loc[eligibility].copy().reset_index(drop=True)
    excluded = original_candidates.loc[~eligibility].copy().reset_index(drop=True)
    if candidates.empty:
        raise ValueError("来源文件没有 eligible finding")
    alignment = pd.read_csv(paths["alignment"], keep_default_na=False)
    mapping = build_mapping(candidates, alignment)
    indexes, matrices = {}, {}
    for model in MODELS:
        index = pd.read_csv(freeze / f"{model}_K20" / "document_index.csv")
        matrix = np.load(freeze / f"{model}_K20" / "document_topic_probabilities.npz")["probabilities"]
        if index["poem_id"].duplicated().any() or set(index["matrix_row"]) != set(range(len(index))):
            raise ValueError(f"{model} document index 非唯一或非完整行索引")
        if matrix.shape != (len(index), 20) or not np.isfinite(matrix).all() or (matrix < 0).any() or not np.allclose(matrix.sum(axis=1), 1, atol=1e-10, rtol=0):
            raise ValueError(f"{model} 冻结概率矩阵校验失败")
        matrix.setflags(write=False)
        indexes[model], matrices[model] = index, matrix
    c_check = master[["poem_id", *[f"topic_{i:02d}_prob" for i in range(1, 21)]]].merge(indexes["C"][["poem_id", "matrix_row"]], on="poem_id", validate="one_to_one")
    if not np.allclose(c_check.iloc[:, 1:21].to_numpy(float), matrices["C"][c_check["matrix_row"].to_numpy(int)], atol=1e-14, rtol=0):
        raise ValueError("C master 与冻结 C 矩阵按 poem_id 对齐后概率不一致")
    return {"paths": paths, "hashes": hashes, "master": master, "metadata": metadata, "labels": labels, "candidates": candidates, "excluded": excluded, "alignment": alignment, "mapping": mapping, "indexes": indexes, "matrices": matrices,
            "original_effects": pd.read_csv(paths["original_effects"]), "original_clr": pd.read_csv(paths["original_clr"])}


def common_universe(inputs: dict[str, Any], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """构造精确 B∩C∩D，并分别保留全体和分样本损失账本。"""
    indexes = inputs["indexes"]
    common_ids = set.intersection(*(set(indexes[m]["poem_id"]) for m in MODELS))
    common = pd.DataFrame({"poem_id": sorted(common_ids)})
    native = {}
    for model in MODELS:
        common = common.merge(indexes[model][["poem_id", "matrix_row"]].rename(columns={"matrix_row": f"{model}_matrix_row"}), on="poem_id", validate="one_to_one")
        frame = indexes[model][["poem_id", "matrix_row"]].merge(inputs["metadata"], on="poem_id", how="left", validate="one_to_one", indicator=True)
        frame["historical_metadata_available"] = frame["_merge"].eq("both")
        frame = frame.drop(columns="_merge")
        for column in config["samples"].values():
            frame[column] = frame[column].eq(True)
        native[model] = frame
    common = common.merge(inputs["metadata"], on="poem_id", how="left", validate="one_to_one", indicator=True)
    if not common["_merge"].eq("both").all():
        raise ValueError("共同文档存在缺失历史 metadata，不能进行主比较")
    common = common.drop(columns="_merge")
    rows = []
    for model in MODELS:
        frame = native[model]
        scopes = [("ALL", "ALL", np.ones(len(frame), dtype=bool))]
        for sample, field in config["samples"].items():
            scopes.append((sample, "ALL", frame[field].to_numpy(bool)))
            for period in config["period_order"]:
                scopes.append((sample, period, (frame[field] & frame["author_period_primary"].eq(period)).to_numpy(bool)))
        scopes.append(("METADATA_MISSING", "UNKNOWN", ~frame["historical_metadata_available"].to_numpy(bool)))
        for sample, period, mask in scopes:
            selected = frame.loc[mask]
            in_common = selected["poem_id"].isin(common_ids)
            rows.append({"model": model, "sample": sample, "period": period, "total_documents": len(selected), "common_documents": int(in_common.sum()), "lost_documents": int((~in_common).sum()), "lost_poem_ids": ";".join(map(str, selected.loc[~in_common, "poem_id"])), "native_model_documents": len(frame), "historical_metadata_missing": int((~frame["historical_metadata_available"]).sum())})
    return common, pd.DataFrame(rows), native


def probability_sum(matrix: np.ndarray, topic_ids: list[str]) -> np.ndarray:
    """只对明确映射的 Topic probability mass 求无权重和。"""
    if not topic_ids or len(set(topic_ids)) != len(topic_ids):
        raise ValueError("映射 Topic 集合为空或重复")
    columns = [int(topic.split("_")[-1]) - 1 for topic in topic_ids]
    if min(columns) < 0 or max(columns) >= matrix.shape[1]:
        raise ValueError("映射 Topic 超出矩阵列范围")
    return matrix[:, columns].sum(axis=1)


def semantic_frame(metadata: pd.DataFrame, matrix: np.ndarray, model: str, mapping: pd.DataFrame, topics: list[str], row_column: str, variant: str = "PRIMARY", clr: bool = False) -> pd.DataFrame:
    """按 poem_id 已连接的显式 matrix_row 提取语义质量，不依赖输入行顺序。"""
    result = metadata.copy().reset_index(drop=True)
    theta = matrix[result[row_column].to_numpy(int)]
    if clr and (theta <= 0).any():
        raise ValueError("CLR 遇到真零；未授权 pseudocount")
    model_mapping = mapping.loc[mapping["target_model"].eq(model)].drop_duplicates("c_topic_id").set_index("c_topic_id") if model != "C" else None
    for topic in topics:
        if model == "C":
            ids = [topic]
        else:
            record = model_mapping.loc[topic]
            if record["mapping_testability"] == "NOT_TESTABLE":
                result[topic] = np.nan
                continue
            ids = [record["single_best_topic_id"]] if variant == "SINGLE_BEST" else record["analysis_topic_ids"].split(";")
        values = probability_sum(theta, ids)
        result[topic] = np.log(values) - np.log(theta).mean(axis=1) if clr else values
    return result


def author_table(frame: pd.DataFrame, topics: list[str]) -> pd.DataFrame:
    """建立一行一作者的诗数、语义质量与均值。"""
    grouped = frame.groupby(["author_id_internal", "author_period_primary"], sort=True, observed=True)
    mass = grouped[topics].sum(min_count=1)
    counts = grouped.size().rename("poem_count")
    output = pd.concat([counts, mass.add_suffix("_mass"), mass.div(counts, axis=0).add_suffix("_mean")], axis=1).reset_index()
    names = frame.drop_duplicates("author_id_internal").set_index("author_id_internal")["author"]
    output["author"] = output["author_id_internal"].map(names)
    return output


def bootstrap_period(authors: pd.DataFrame, topics: list[str], iterations: int, seed: int, batch: int = 500) -> dict[str, np.ndarray]:
    """作者整簇重采样，同时计算诗等权与作者等权均值。"""
    n = len(authors)
    if n == 0:
        raise ValueError("时期内没有作者")
    rng = np.random.default_rng(seed)
    mass = authors[[f"{t}_mass" for t in topics]].to_numpy(float)
    means = authors[[f"{t}_mean" for t in topics]].to_numpy(float)
    counts = authors["poem_count"].to_numpy(float)
    poem, author = [], []
    for start in range(0, iterations, batch):
        weights = rng.multinomial(n, np.full(n, 1 / n), size=min(batch, iterations - start))
        poem.append((weights @ mass) / (weights @ counts)[:, None])
        author.append((weights @ means) / n)
    return {"POEM": np.concatenate(poem), "AUTHOR": np.concatenate(author)}


def analyze(frame: pd.DataFrame, model: str, candidates: pd.DataFrame, mapping: pd.DataFrame, config: dict[str, Any], universe: str = "COMMON", variant: str = "PRIMARY") -> tuple[pd.DataFrame, dict[str, pd.DataFrame], str]:
    """仅计算已筛选 finding 的 contrast，同时保存四期均值。"""
    topics = sorted(candidates["topic_id"].unique())
    periods = config["period_order"]
    code_to_period = {v: k for k, v in config["period_codes"].items()}
    map_lookup = mapping.loc[mapping["target_model"].eq(model)].set_index("finding_id") if model != "C" else None
    rows, author_tables, digest = [], {}, hashlib.sha256()
    for sample, field in config["samples"].items():
        selected = frame.loc[frame[field]].copy()
        authors = author_table(selected, topics)
        author_tables[sample] = authors
        boots, point = {}, {}
        for period in periods:
            group = authors.loc[authors["author_period_primary"].eq(period)].sort_values("author_id_internal")
            seed = stable_seed(config["random_seed"], "SHARED_AUTHOR_DRAWS", sample, period)
            boots[period] = bootstrap_period(group, topics, config["bootstrap_iterations"], seed, config["bootstrap_batch_size"])
            point[period] = {
                "POEM": group[[f"{t}_mass" for t in topics]].sum().to_numpy(float) / group["poem_count"].sum(),
                "AUTHOR": group[[f"{t}_mean" for t in topics]].mean().to_numpy(float),
            }
            for aggregation in AGGREGATIONS:
                digest.update(boots[period][aggregation].tobytes())
        for finding in candidates.to_dict("records"):
            topic, fid = finding["topic_id"], finding["finding_id"]
            index = topics.index(topic)
            a_code, b_code = finding["contrast"].split("-")
            period_a, period_b = code_to_period[a_code], code_to_period[b_code]
            group_a = authors.loc[authors["author_period_primary"].eq(period_a)]
            group_b = authors.loc[authors["author_period_primary"].eq(period_b)]
            ga, gb = group_a[f"{topic}_mean"].to_numpy(float), group_b[f"{topic}_mean"].to_numpy(float)
            g, cliff = hedges_g(ga, gb), cliffs_delta(ga, gb)
            mapping_record = map_lookup.loc[fid] if model != "C" else None
            for aggregation in AGGREGATIONS:
                profile = np.array([point[p][aggregation][index] for p in periods])
                delta = float(point[period_b][aggregation][index] - point[period_a][aggregation][index])
                draws = boots[period_b][aggregation][:, index] - boots[period_a][aggregation][:, index]
                low, high = np.quantile(draws, [0.025, 0.975])
                testable = model == "C" or mapping_record["mapping_testability"] != "NOT_TESTABLE"
                if not testable:
                    profile[:] = np.nan
                    delta, low, high, g, cliff = [math.nan] * 5
                rows.append({
                    "finding_id": fid, "c_topic_id": topic, "c_topic_label": finding["topic_label"], "contrast": finding["contrast"],
                    "model": model, "model_label": "C_COMMON" if model == "C" and universe == "COMMON" else f"{model}_{universe}", "universe": universe,
                    "mapped_topic_ids": topic if model == "C" else mapping_record["single_best_topic_id"] if variant == "SINGLE_BEST" else mapping_record["analysis_topic_ids"],
                    "mapping_type": "DIRECT" if model == "C" else mapping_record["mapping_testability"], "mapping_variant": variant,
                    "sample": sample, "aggregation": aggregation, "specification": f"{sample}_{aggregation}",
                    "period_A": period_a, "period_B": period_b, "period_A_mean": float(profile[periods.index(period_a)]), "period_B_mean": float(profile[periods.index(period_b)]),
                    "raw_delta": delta, "bootstrap_ci_low": float(low), "bootstrap_ci_high": float(high), "direction": sign(delta), "bootstrap_supported_direction": supported(float(low), float(high)),
                    "hedges_g_if_applicable": g if aggregation == "AUTHOR" else math.nan, "hedges_magnitude": magnitude(g, "hedges", config["thresholds"]) if aggregation == "AUTHOR" else "",
                    "cliffs_delta_if_applicable": cliff if aggregation == "AUTHOR" else math.nan, "cliffs_magnitude": magnitude(cliff, "cliff", config["thresholds"]) if aggregation == "AUTHOR" else "",
                    **{f"mean_{config['period_codes'][p]}": float(profile[i]) for i, p in enumerate(periods)},
                    "peak_period": periods[int(np.argmax(profile))] if testable else "", "trough_period": periods[int(np.argmin(profile))] if testable else "",
                    "rank_order": ">".join(config["period_codes"][periods[i]] for i in np.argsort(-profile, kind="stable")) if testable else "",
                    "n_authors_A": len(group_a), "n_authors_B": len(group_b), "n_poems_A": int(group_a["poem_count"].sum()), "n_poems_B": int(group_b["poem_count"].sum()),
                    "bootstrap_iterations": config["bootstrap_iterations"], "bootstrap_unit": "author", "random_seed": config["random_seed"], "formal_p_value_family": "NONE_PRESELECTED_FINDINGS",
                })
    return pd.DataFrame(rows), author_tables, digest.hexdigest()


def compare_original_common(effects: pd.DataFrame, inputs: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    """原样保留既有 C 结果，并独立量化文档交集影响。"""
    original = inputs["original_effects"].copy()
    original["finding_id"] = [finding_id(t, c) for t, c in zip(original["topic_id"], original["contrast"])]
    original = original.loc[original["finding_id"].isin(inputs["candidates"]["finding_id"])]
    common = effects.loc[effects["model"].eq("C")]
    prior = original.set_index(["finding_id", "sample", "aggregation"])
    rows = []
    thresholds = config["thresholds"]
    for record in common.to_dict("records"):
        old = prior.loc[(record["finding_id"], record["sample"], record["aggregation"])]
        difference = record["raw_delta"] - float(old["raw_delta"])
        relative = abs(difference) / abs(float(old["raw_delta"])) if abs(float(old["raw_delta"])) > 1e-12 else math.nan
        changed = sign(float(old["raw_delta"])) != record["direction"] or abs(difference) >= thresholds["intersection_absolute_change"] or (abs(difference) >= thresholds["intersection_relative_absolute_floor"] and relative >= thresholds["intersection_relative_change"])
        rows.append({"finding_id": record["finding_id"], "sample": record["sample"], "aggregation": record["aggregation"], "C_original_delta": float(old["raw_delta"]), "C_common_delta": record["raw_delta"], "absolute_delta_change": abs(difference), "relative_delta_change": relative,
                     "C_original_ci_low": float(old["bootstrap_ci_low"]), "C_original_ci_high": float(old["bootstrap_ci_high"]), "C_common_ci_low": record["bootstrap_ci_low"], "C_common_ci_high": record["bootstrap_ci_high"],
                     "C_original_hedges_g": old["hedges_g"], "C_common_hedges_g": record["hedges_g_if_applicable"], "direction_agreement": sign(float(old["raw_delta"])) == record["direction"],
                     "DOCUMENT_INTERSECTION_SENSITIVITY": changed, "ci_comparison_note": "原结果seed=20260901；COMMON seed=20260902；CI差异包含Monte Carlo差异，不自动解释为样本变化"})
    return pd.DataFrame(rows)


def build_sensitivity_tables(effects: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """汇总样本、聚合方式与四时期形状，不生成额外时期检验。"""
    lookup = effects.set_index(["finding_id", "model", "sample", "aggregation"])
    sample_rows, aggregation_rows, shape_rows = [], [], []
    for (fid, model, aggregation), group in effects.groupby(["finding_id", "model", "aggregation"], sort=False):
        reference = lookup.loc[(fid, "C", "CORE", aggregation)]["direction"]
        sample_rows.append({"finding_id": fid, "model": model, "aggregation": aggregation,
                            "model_sample_positive_count": int(group["bootstrap_supported_direction"].eq("SUPPORTED_POSITIVE").sum()),
                            "model_sample_negative_count": int(group["bootstrap_supported_direction"].eq("SUPPORTED_NEGATIVE").sum()),
                            "model_sample_uncertain_count": int(group["bootstrap_supported_direction"].eq("UNCERTAIN").sum()),
                            "raw_positive_count": int(group["direction"].eq("POSITIVE").sum()), "raw_negative_count": int(group["direction"].eq("NEGATIVE").sum()),
                            "same_as_C_CORE_count": int(group["direction"].eq(reference).sum()), "C_CORE_reference_direction": reference,
                            **{f"{s}_direction": lookup.loc[(fid, model, s, aggregation)]["direction"] for s in config["samples"]},
                            **{f"{s}_bootstrap": lookup.loc[(fid, model, s, aggregation)]["bootstrap_supported_direction"] for s in config["samples"]}})
    for (fid, model, sample), group in effects.groupby(["finding_id", "model", "sample"], sort=False):
        poem, author = lookup.loc[(fid, model, sample, "POEM")], lookup.loc[(fid, model, sample, "AUTHOR")]
        c_poem, c_author = lookup.loc[(fid, "C", sample, "POEM")], lookup.loc[(fid, "C", sample, "AUTHOR")]
        c_difference = c_poem["direction"] != c_author["direction"]
        aggregation_rows.append({"finding_id": fid, "model": model, "sample": sample, "poem_delta": poem["raw_delta"], "author_delta": author["raw_delta"], "poem_direction": poem["direction"], "author_direction": author["direction"],
                                 "aggregation_direction_agreement": poem["direction"] == author["direction"], "C_estimand_difference": c_difference,
                                 "structural_replication": bool(c_difference and poem["direction"] == c_poem["direction"] and author["direction"] == c_author["direction"]), "absolute_estimand_gap": abs(poem["raw_delta"] - author["raw_delta"])})
        if model == "C":
            continue
        for aggregation in AGGREGATIONS:
            target = lookup.loc[(fid, model, sample, aggregation)]
            c = lookup.loc[(fid, "C", sample, aggregation)]
            left = np.array([target[f"mean_{p}"] for p in "EHML"], dtype=float)
            right = np.array([c[f"mean_{p}"] for p in "EHML"], dtype=float)
            valid = np.isfinite(left).all() and np.isfinite(right).all() and np.std(left) > 0 and np.std(right) > 0
            pearson = float(np.corrcoef(left, right)[0, 1]) if valid else math.nan
            spearman = float(np.corrcoef(pd.Series(left).rank(), pd.Series(right).rank())[0, 1]) if valid else math.nan
            shape_rows.append({"finding_id": fid, "target_model": model, "sample": sample, "aggregation": aggregation, "pearson_four_periods": pearson, "spearman_four_periods": spearman,
                               "C_peak_period": c["peak_period"], "target_peak_period": target["peak_period"], "C_trough_period": c["trough_period"], "target_trough_period": target["trough_period"], "C_rank_order": c["rank_order"], "target_rank_order": target["rank_order"], "correlation_inference": "DESCRIPTIVE_ONLY_FOUR_PERIODS_NO_P_VALUE"})
    return pd.DataFrame(sample_rows), pd.DataFrame(aggregation_rows), pd.DataFrame(shape_rows)


def minimal_author_checks(author_tables: dict[str, dict[str, pd.DataFrame]], effects: pd.DataFrame, candidates: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """仅对优先四项中核心方向复制成功的 B/D 运行 Top1/Top3 检查。"""
    lookup = effects.set_index(["finding_id", "model", "sample", "aggregation"])
    rows = []
    for finding in candidates.loc[candidates["finding_id"].isin(config["priority_findings"])].to_dict("records"):
        fid, topic = finding["finding_id"], finding["topic_id"]
        for model in ("B", "D"):
            core = [{**lookup.loc[(fid, model, "CORE", aggregation)].to_dict(), "aggregation": aggregation} for aggregation in AGGREGATIONS]
            replicated = all(record["direction"] == lookup.loc[(fid, "C", "CORE", record["aggregation"])]["direction"] for record in core)
            authors = author_tables[model]["CORE"]
            involved = authors.loc[authors["author_period_primary"].isin([core[0]["period_A"], core[0]["period_B"]])]
            ranked = involved.sort_values([f"{topic}_mass", "author_id_internal"], ascending=[False, True])
            for n in config["minimal_author_check"]["top_n"]:
                removed = ranked.head(n) if replicated else ranked.iloc[:0]
                remaining = involved.loc[~involved["author_id_internal"].isin(removed["author_id_internal"])]
                for record in core:
                    aggregation = record["aggregation"]
                    means = []
                    for period in (record["period_A"], record["period_B"]):
                        group = remaining.loc[remaining["author_period_primary"].eq(period)]
                        value = math.nan if group.empty else group[f"{topic}_mass"].sum() / group["poem_count"].sum() if aggregation == "POEM" else group[f"{topic}_mean"].mean()
                        means.append(float(value))
                    delta = means[1] - means[0] if replicated else math.nan
                    reversal = replicated and sign(delta) in {"POSITIVE", "NEGATIVE"} and sign(delta) != record["direction"]
                    rows.append({"finding_id": fid, "model": model, "sample": "CORE", "aggregation": aggregation, "N": n, "check_run": replicated, "exclusion_scheme": "GLOBAL_TOP_WITHIN_CONTRAST", "ranking_measure": "mapped semantic probability mass",
                                 "excluded_author_ids": ";".join(removed["author_id_internal"]), "excluded_authors": ";".join(removed["author"].astype(str)), "excluded_poems": int(removed["poem_count"].sum()), "original_delta": record["raw_delta"], "excluded_delta": delta,
                                 "original_direction": record["direction"], "excluded_direction": sign(delta), "sign_reversal": bool(reversal), "author_sensitivity_flag": "TARGET_MODEL_AUTHOR_SENSITIVE" if reversal else "NOT_ESTIMABLE_AFTER_EXCLUSION" if replicated and not math.isfinite(delta) else "NO_TOP1_TOP3_REVERSAL" if replicated else "NOT_RUN_CORE_REPLICATION_NOT_MET"})
    return pd.DataFrame(rows)


def run_triggered_clr(inputs: dict[str, Any], common: pd.DataFrame, effects: pd.DataFrame, singles: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    """只对预先定义触发条件运行临时 CLR 诊断。"""
    raw = effects.loc[effects["sample"].eq("CORE")].set_index(["finding_id", "model", "aggregation"])
    single = singles.loc[singles["sample"].eq("CORE")].set_index(["finding_id", "model", "aggregation"])
    old_clr = inputs["original_clr"].copy()
    old_clr["finding_id"] = [finding_id(t, c) for t, c in zip(old_clr["topic_id"], old_clr["contrast"])]
    warnings = set(old_clr.loc[~bool_series(old_clr["direction_agreement"], "direction_agreement"), "finding_id"])
    triggers = {}
    for fid in inputs["candidates"]["finding_id"]:
        for model in ("B", "D"):
            reasons = []
            if any(raw.loc[(fid, model, a)]["direction"] != raw.loc[(fid, "C", a)]["direction"] for a in AGGREGATIONS):
                reasons.append("TARGET_C_RAW_SIGN_DISAGREEMENT")
            if any((fid, model, a) in single.index and single.loc[(fid, model, a)]["direction"] != raw.loc[(fid, model, a)]["direction"] for a in AGGREGATIONS):
                reasons.append("BUNDLE_SINGLE_SIGN_DISAGREEMENT")
            if fid in warnings:
                reasons.append("PRIOR_C_COMPOSITIONAL_WARNING")
            if reasons:
                triggers[(fid, model)] = ";".join(reasons)
    parts, hashes = [], {}
    for model in ("B", "D"):
        ids = [fid for fid, target in triggers if target == model]
        selected = inputs["candidates"].loc[inputs["candidates"]["finding_id"].isin(ids)]
        if selected.empty:
            continue
        frame = semantic_frame(common, inputs["matrices"][model], model, inputs["mapping"], sorted(selected["topic_id"].unique()), f"{model}_matrix_row", clr=True)
        clr_config = {**config, "samples": {"CORE": config["samples"]["CORE"]}}
        transformed, _, digest = analyze(frame, model, selected, inputs["mapping"], clr_config, variant="CLR_DIAGNOSTIC")
        hashes[model] = digest
        for record in transformed.to_dict("records"):
            original = raw.loc[(record["finding_id"], model, record["aggregation"])]
            parts.append({"finding_id": record["finding_id"], "model": model, "aggregation": record["aggregation"], "sample": "CORE", "mapped_topic_ids": record["mapped_topic_ids"], "mapping_type": record["mapping_type"],
                          "trigger": triggers[(record["finding_id"], model)], "raw_delta": original["raw_delta"], "raw_direction": original["direction"], "clr_delta": record["raw_delta"], "clr_direction": record["direction"], "clr_ci_low": record["bootstrap_ci_low"], "clr_ci_high": record["bootstrap_ci_high"], "clr_bootstrap_direction": record["bootstrap_supported_direction"],
                          "raw_clr_direction_agreement": original["direction"] == record["direction"], "transform": config["clr"]["bundle_transform"], "pseudocount": "NONE", "bootstrap_iterations": config["bootstrap_iterations"], "bootstrap_unit": "author"})
    columns = ["finding_id", "model", "aggregation", "sample", "mapped_topic_ids", "mapping_type", "trigger", "raw_delta", "raw_direction", "clr_delta", "clr_direction", "clr_ci_low", "clr_ci_high", "clr_bootstrap_direction", "raw_clr_direction_agreement", "transform", "pseudocount", "bootstrap_iterations", "bootstrap_unit"]
    return pd.DataFrame(parts, columns=columns), hashes


def classify_robustness(evidence: dict[str, Any]) -> str:
    """按固定规则合成证据层，绝不依据 effect ratio 单独判定。"""
    if evidence["unresolved"]:
        return "SEMANTICALLY_UNRESOLVED"
    if evidence["both_strong_not_replicated"]:
        return "NOT_REPLICATED"
    if evidence["systematic_reversal"]:
        return "MODEL_SENSITIVE"
    if evidence["core_agreement"] and evidence["target_support"] and evidence["sample_consistency"] and not any(evidence[key] for key in ("semantic_limit", "review_required", "bundle_dependency", "author_sensitive")):
        return "ROBUST_ACROSS_BCD"
    return "PARTIALLY_ROBUST"


def build_comparison(inputs: dict[str, Any], effects: pd.DataFrame, singles: pd.DataFrame, shapes: pd.DataFrame, exclusions: pd.DataFrame, clr: pd.DataFrame, benchmark: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分开语义映射与历史效应评价，并保留所有不一致诊断。"""
    lookup = effects.set_index(["finding_id", "model", "sample", "aggregation"])
    maps = inputs["mapping"].set_index(["finding_id", "target_model"])
    rows, diagnostic_rows = [], []
    for finding in inputs["candidates"].to_dict("records"):
        fid = finding["finding_id"]
        record = {"finding_id": fid, "topic": finding["topic_id"], "topic_label": finding["topic_label"], "contrast": finding["contrast"], "C_original_status": finding["previous_validation_status"], "C_author_robustness_status": finding["author_robustness_status"], "C_original_poem_delta": float(finding["original_poem_delta"]), "C_original_author_delta": float(finding["original_author_delta"]), "priority": "VERY_HIGH" if fid in config["priority_findings"] else "STANDARD"}
        mapping_records = [maps.loc[(fid, m)] for m in ("B", "D")]
        unresolved = any(row["mapping_testability"] in {"WEAK_PROXY", "NOT_TESTABLE"} for row in mapping_records)
        semantic_limit = any(row["mapping_testability"] == "MERGED_PROXY" or (row["mapping_testability"] == "BUNDLE" and row["alignment_confidence"] not in config["semantic_policy"]["reliable_bundle_confidence"]) for row in mapping_records)
        review_required = any(bool(row["alignment_review_required"]) for row in mapping_records)
        record["semantic_mapping_assessment"] = "UNRESOLVED" if unresolved else "LIMITED_PROXY_OR_MEDIUM_BUNDLE" if semantic_limit else "ACCEPTABLE_HIGH_CONFIDENCE_BUNDLE" if any(row["mapping_testability"] == "BUNDLE" for row in mapping_records) else "STRONG_OR_MODERATE_DIRECT"
        target_support, core_agreements, sample_counts, stable_reversals, author_flags, bundle_flags, all_boot = [], [], {}, [], [], [], []
        for model in MODELS:
            group = effects.loc[effects["finding_id"].eq(fid) & effects["model"].eq(model)]
            same_reference = 0
            for aggregation in AGGREGATIONS:
                core = lookup.loc[(fid, model, "CORE", aggregation)]
                c = lookup.loc[(fid, "C", "CORE", aggregation)]
                record[f"{model}_{aggregation.lower()}_delta"] = core["raw_delta"]
                record[f"{model}_{aggregation.lower()}_ci_low"] = core["bootstrap_ci_low"]
                record[f"{model}_{aggregation.lower()}_ci_high"] = core["bootstrap_ci_high"]
                record[f"{model}_{aggregation.lower()}_bootstrap"] = core["bootstrap_supported_direction"]
                record[f"{model}_{aggregation.lower()}_direction"] = core["direction"]
                all_boot.append(core["bootstrap_supported_direction"] == f"SUPPORTED_{c['direction']}")
                same_reference += int(group.loc[group["aggregation"].eq(aggregation), "direction"].eq(c["direction"]).sum())
                if model != "C":
                    agreement = core["direction"] == c["direction"] and core["direction"] in {"POSITIVE", "NEGATIVE"}
                    record[f"{model.lower()}_{aggregation.lower()}_sign_agreement"] = agreement
                    core_agreements.append(agreement)
                    record[f"{model}_{aggregation.lower()}_effect_ratio_to_C"] = core["raw_delta"] / c["raw_delta"] if abs(c["raw_delta"]) >= config["thresholds"]["effect_ratio_minimum_c_delta"] else math.nan
            author = lookup.loc[(fid, model, "CORE", "AUTHOR")]
            record[f"{model}_author_hedges_g"] = author["hedges_g_if_applicable"]
            record[f"{model}_author_hedges_magnitude"] = author["hedges_magnitude"]
            record[f"{model}_author_cliffs_delta"] = author["cliffs_delta_if_applicable"]
            record[f"{model}_author_cliffs_magnitude"] = author["cliffs_magnitude"]
            sample_counts[model] = same_reference
            if model == "C":
                continue
            mapping_record = maps.loc[(fid, model)]
            record[f"{model}_mapping_type"] = mapping_record["mapping_testability"]
            record[f"{model}_mapped_topic_ids"] = mapping_record["analysis_topic_ids"]
            record[f"{model}_semantic_confidence"] = mapping_record["alignment_confidence"]
            record[f"{model}_alignment_review_required"] = bool(mapping_record["alignment_review_required"])
            correct_supported = sum(lookup.loc[(fid, model, "CORE", a)]["bootstrap_supported_direction"] == f"SUPPORTED_{lookup.loc[(fid, 'C', 'CORE', a)]['direction']}" for a in AGGREGATIONS)
            agrees = all(record[f"{model.lower()}_{a.lower()}_sign_agreement"] for a in AGGREGATIONS)
            target_support.append(agrees and correct_supported >= 1)
            opposite_raw = opposite_supported = 0
            for item in group.to_dict("records"):
                c = lookup.loc[(fid, "C", item["sample"], item["aggregation"])]
                opposite = item["direction"] in {"POSITIVE", "NEGATIVE"} and c["direction"] in {"POSITIVE", "NEGATIVE"} and item["direction"] != c["direction"]
                opposite_raw += int(opposite)
                opposite_supported += int(opposite and item["bootstrap_supported_direction"] == f"SUPPORTED_{item['direction']}")
            stable = opposite_raw >= config["thresholds"]["systematic_opposite_raw_minimum_of_six"] and opposite_supported >= config["thresholds"]["systematic_opposite_supported_minimum_of_six"]
            stable_reversals.append(stable)
            record[f"{model}_opposite_raw_spec_count"] = opposite_raw
            record[f"{model}_opposite_supported_spec_count"] = opposite_supported
            record[f"{model}_supports_C_finding"] = agrees and correct_supported >= 1
            single = singles.loc[singles["finding_id"].eq(fid) & singles["model"].eq(model) & singles["sample"].eq("CORE")]
            dependent = any(row["direction"] != lookup.loc[(fid, model, "CORE", row["aggregation"])]["direction"] for row in single.to_dict("records"))
            bundle_flags.append(dependent)
            record[f"{model}_bundle_single_direction_dependency"] = dependent
            author_check = exclusions.loc[exclusions["finding_id"].eq(fid) & exclusions["model"].eq(model)]
            sensitive = bool(author_check["sign_reversal"].any()) if not author_check.empty else False
            author_flags.append(sensitive)
            record[f"{model}_author_sensitivity_flag"] = "TARGET_MODEL_AUTHOR_SENSITIVE" if sensitive else "NOT_ESTIMABLE_AFTER_EXCLUSION" if not author_check.empty and author_check["author_sensitivity_flag"].eq("NOT_ESTIMABLE_AFTER_EXCLUSION").any() else "NO_TOP1_TOP3_REVERSAL" if not author_check.empty and author_check["check_run"].all() else "NOT_RUN_NOT_PRIORITY_OR_NOT_REPLICATED"
            shape = shapes.loc[shapes["finding_id"].eq(fid) & shapes["target_model"].eq(model) & shapes["sample"].eq("CORE")]
            for a in AGGREGATIONS:
                item = shape.loc[shape["aggregation"].eq(a)].iloc[0]
                record[f"{model}_C_{a.lower()}_shape_pearson"] = item["pearson_four_periods"]
                record[f"{model}_C_{a.lower()}_shape_spearman"] = item["spearman_four_periods"]
        core_agreement = all(core_agreements)
        sample_consistent = all(n >= config["thresholds"]["sample_same_sign_minimum_of_six"] for n in sample_counts.values())
        composition_flag = bool((~clr.loc[clr["finding_id"].eq(fid), "raw_clr_direction_agreement"]).any()) if not clr.empty else False
        both_strong = all(row["mapping_testability"] == "DIRECT" and row["alignment_confidence"] == "HIGH" for row in mapping_records)
        internally_stable = all(effects.loc[effects["finding_id"].eq(fid) & effects["model"].eq(m)].groupby("aggregation")["direction"].nunique().le(1).all() for m in MODELS)
        no_aggregation_artifact = all(lookup.loc[(fid, m, "CORE", "POEM")]["direction"] == lookup.loc[(fid, m, "CORE", "AUTHOR")]["direction"] for m in MODELS)
        evidence = {"unresolved": unresolved, "both_strong_not_replicated": both_strong and all(stable_reversals) and not composition_flag and internally_stable and no_aggregation_artifact,
                    "systematic_reversal": any(stable_reversals), "core_agreement": core_agreement, "target_support": all(target_support), "sample_consistency": sample_consistent,
                    "semantic_limit": semantic_limit, "review_required": review_required, "bundle_dependency": any(bundle_flags), "author_sensitive": any(author_flags)}
        classification = classify_robustness(evidence)
        direction_pattern = "UNTESTABLE" if unresolved else "SIGN_REVERSAL" if any(stable_reversals) else "FULL_AGREEMENT" if core_agreement and all(all_boot) else "MOSTLY_AGREEMENT" if core_agreement else "PARTIAL_AGREEMENT"
        record.update({"poem_direction_agreement": record["b_poem_sign_agreement"] and record["d_poem_sign_agreement"], "author_direction_agreement": record["b_author_sign_agreement"] and record["d_author_sign_agreement"],
                       "cross_model_direction_pattern": direction_pattern, "sample_direction_consistency": ";".join(f"{m}:{sample_counts[m]}/6" for m in MODELS),
                       "sample_consistency_minimum": min(sample_counts.values()), "historical_effect_assessment": "SYSTEMATIC_SIGN_REVERSAL" if any(stable_reversals) else "DIRECTIONALLY_SUPPORTED" if core_agreement and all(target_support) and sample_consistent else "PARTIAL_OR_UNCERTAIN",
                       "finding_level_robustness": classification, "DOCUMENT_INTERSECTION_SENSITIVITY": bool(benchmark.loc[benchmark["finding_id"].eq(fid), "DOCUMENT_INTERSECTION_SENSITIVITY"].any()), "compositional_sensitivity_flag": composition_flag})
        record["shape_similarity_BC"] = f"POEM Spearman={record['B_C_poem_shape_spearman']:.3f}; AUTHOR Spearman={record['B_C_author_shape_spearman']:.3f}"
        record["shape_similarity_CD"] = f"POEM Spearman={record['D_C_poem_shape_spearman']:.3f}; AUTHOR Spearman={record['D_C_author_shape_spearman']:.3f}"
        reasons = [f"CORE符号一致={core_agreement}", f"B/D至少一种聚合的CI支持={all(target_support)}", f"样本符号={record['sample_direction_consistency']}", f"语义={record['semantic_mapping_assessment']}"]
        if any(bundle_flags): reasons.append("bundle与single-best存在方向依赖")
        if any(author_flags): reasons.append("目标模型Top1/Top3排除反向")
        if review_required: reasons.append("ALIGNMENT_REVIEW_REQUIRED")
        if any(stable_reversals): reasons.append("目标模型多规格稳定反向")
        record["robustness_reason"] = "；".join(reasons)
        rows.append(record)
        if classification == "ROBUST_ACROSS_BCD":
            continue
        for model in ("B", "D"):
            mapping_record = maps.loc[(fid, model)]
            problems = []
            if mapping_record["mapping_testability"] == "BUNDLE": problems.append(("SEMANTIC_SPLIT", f"来源split={mapping_record['analysis_topic_ids']}；single方向依赖={record[f'{model}_bundle_single_direction_dependency']}"))
            if mapping_record["mapping_testability"] == "MERGED_PROXY": problems.append(("SEMANTIC_MERGE", mapping_record["semantic_notes"]))
            if mapping_record["mapping_testability"] in {"WEAK_PROXY", "NOT_TESTABLE"} or mapping_record["alignment_review_required"]: problems.append(("WEAK_ALIGNMENT", mapping_record["alignment_review_flag"]))
            if record[f"{model}_opposite_raw_spec_count"]: problems.append(("SIGN_REVERSAL", f"反向数值规格={record[f'{model}_opposite_raw_spec_count']}/6；反向CI支持={record[f'{model}_opposite_supported_spec_count']}/6"))
            if sample_counts[model] < 6: problems.append(("SAMPLE_SENSITIVITY", f"与C核心方向一致={sample_counts[model]}/6"))
            if record[f"{model}_poem_direction"] != record[f"{model}_author_direction"]: problems.append(("AGGREGATION_SENSITIVITY", f"POEM={record[f'{model}_poem_direction']}；AUTHOR={record[f'{model}_author_direction']}"))
            if abs(record[f"{model}_author_hedges_g"]) < config["thresholds"]["hedges_small"] and abs(record["C_author_hedges_g"]) >= config["thresholds"]["hedges_small"]: problems.append(("MAGNITUDE_COLLAPSE", f"作者g由C非微小档降至VERY_SMALL：C={record['C_author_hedges_g']:.4f}，{model}={record[f'{model}_author_hedges_g']:.4f}；不凭raw ratio判定"))
            clr_model = clr.loc[clr["finding_id"].eq(fid) & clr["model"].eq(model)]
            if not clr_model.empty and (~clr_model["raw_clr_direction_agreement"]).any(): problems.append(("COMPOSITIONAL_SENSITIVITY", ";".join(f"{r['aggregation']} raw={r['raw_direction']} CLR={r['clr_direction']}" for r in clr_model.to_dict("records"))))
            if record[f"{model}_author_sensitivity_flag"] == "TARGET_MODEL_AUTHOR_SENSITIVE": problems.append(("AUTHOR_SENSITIVITY", "Top1或Top3语义质量贡献作者排除后反向"))
            for problem, detail in problems:
                diagnostic_rows.append({"finding": fid, "model": model, "problem_type": problem, "evidence": detail, "interpretation_limit": "只诊断模型/样本/估计量敏感性；不能据此证明词汇过滤或文学史因果", "finding_level_robustness": classification})
    return pd.DataFrame(rows), pd.DataFrame(diagnostic_rows, columns=["finding", "model", "problem_type", "evidence", "interpretation_limit", "finding_level_robustness"])


def run_invariance_tests(inputs: dict[str, Any], common: pd.DataFrame, effects: pd.DataFrame, benchmark: pd.DataFrame, frames: dict[str, pd.DataFrame], semantic_frames: dict[str, pd.DataFrame], report_text: str, config: dict[str, Any]) -> pd.DataFrame:
    """逐项执行任务要求的二十项不可变性和方法契约检查。"""
    current_hashes = {name: sha256_file(path) for name, path in inputs["paths"].items()}
    rows = []
    def add(number: int, name: str, passed: bool, evidence: str) -> None:
        """记录可审计的测试结果。"""
        rows.append({"test_id": f"TEST_{number:02d}", "test": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})
    freeze_names = [name for name in current_hashes if name.startswith("freeze_")]
    add(1, "冻结模型文件不变", all(current_hashes[n] == inputs["hashes"][n] for n in freeze_names), f"{len(freeze_names)}个冻结文件逐项SHA-256前后比较")
    import ast
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    fit_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "fit_transform", "partial_fit"}]
    add(2, "不重新拟合LDA", len(fit_calls) == 0, "分析源代码AST无fit/fit_transform/partial_fit调用；只读取NPZ")
    add(3, "概率矩阵只读且不变", all(not value.flags.writeable for value in inputs["matrices"].values()) and all(current_hashes[f"freeze_{m}_document_topic_probabilities.npz"] == inputs["hashes"][f"freeze_{m}_document_topic_probabilities.npz"] for m in MODELS), "三个原始矩阵writeable=False且文件哈希不变")
    add(4, "历史metadata不变", current_hashes["authoritative_master"] == inputs["hashes"]["authoritative_master"], current_hashes["authoritative_master"])
    source = pd.read_csv(inputs["paths"]["eligible_findings"])
    expected_ids = {finding_id(r.topic_id, r.contrast) for r in source.loc[bool_series(source["finding_level_bd_eligible"], "eligible")].itertuples()}
    actual_ids = set(inputs["candidates"]["finding_id"])
    add(5, "finding来自eligible字段", actual_ids == expected_ids, f"来源eligible={len(expected_ids)}，实际={len(actual_ids)}；预期清单仅用于差异报告")
    add(6, "LOAO降级finding不复活", not (set(inputs["excluded"]["finding_id"]) & set(effects["finding_id"])), f"{len(inputs['excluded'])}项仅保留在排除状态附表")
    joins_correct = all(common[f"{m}_matrix_row"].tolist() == common["poem_id"].map(inputs["indexes"][m].set_index("poem_id")["matrix_row"]).tolist() for m in MODELS)
    add(7, "poem_id连接", joins_correct and not common["poem_id"].duplicated().any(), "逐模型poem_id→matrix_row映射逐项一致；未以作者标题或行序连接")
    intersection = set.intersection(*(set(inputs["indexes"][m]["poem_id"]) for m in MODELS))
    add(8, "共同集合精确为B交C交D", set(common["poem_id"]) == intersection, f"common={len(common)}")
    add(9, "C_ORIGINAL与C_COMMON差异报告", len(benchmark) == len(actual_ids) * len(config["samples"]) * 2, f"{len(benchmark)}行逐finding/sample/aggregation差异和阈值标记")
    alignment = inputs["alignment"].set_index("C_topic_id")
    mapping = inputs["mapping"]
    source_equal = all(row["target_topic_ids"] == str(alignment.loc[row["c_topic_id"], f"{row['target_model']}_topic_id"]) for row in mapping.to_dict("records"))
    add(10, "映射来自既有alignment", source_equal, "18行或动态finding×2目标均逐项核对来源目标集合")
    add(11, "不直接跨模型等同TopicID", source_equal and any(row["c_topic_id"] != row["target_topic_ids"] for row in mapping.to_dict("records")), "目标列始终由来源C-key映射解析；同号不构成匹配依据")
    bundle_equal = True
    for row in mapping.loc[mapping["mapping_testability"].eq("BUNDLE")].drop_duplicates(["c_topic_id", "target_model"]).to_dict("records"):
        theta = inputs["matrices"][row["target_model"]][common[f"{row['target_model']}_matrix_row"].to_numpy(int)]
        bundle_equal &= np.array_equal(semantic_frames[row["target_model"]][row["c_topic_id"]].to_numpy(float), probability_sum(theta, row["analysis_topic_ids"].split(";")))
    add(12, "split仅无权重概率求和", bool(bundle_equal) and mapping.loc[mapping["mapping_testability"].eq("BUNDLE"), "probability_rule"].eq("UNWEIGHTED_SUM").all(), "逐文档bundle与来源目标列直接求和精确一致")
    add(13, "merge明确为代理", mapping.loc[mapping["alignment_relation_type"].eq("merge"), "mapping_testability"].eq("MERGED_PROXY").all(), "所有来源merge均为MERGED_PROXY，不能升级为直接复现")
    add(14, "weak/unmatched不强制匹配", mapping.loc[mapping["alignment_relation_type"].eq("unmatched"), "analysis_topic_ids"].eq("").all() and mapping.loc[mapping["alignment_relation_type"].eq("weak"), "mapping_testability"].eq("WEAK_PROXY").all(), "unmatched不提取概率；weak仅来源代理；当前候选无weak/unmatched")
    poem_ok = author_ok = True
    for row in effects.loc[effects["sample"].eq("CORE")].to_dict("records"):
        frame = semantic_frames[row["model"]]
        selected = frame.loc[frame[config["samples"]["CORE"]] & frame["author_period_primary"].eq(row["period_A"])]
        expected = selected[row["c_topic_id"]].mean() if row["aggregation"] == "POEM" else selected.groupby("author_id_internal")[row["c_topic_id"]].mean().mean()
        correct = np.isclose(expected, row["period_A_mean"], atol=1e-14, rtol=0, equal_nan=True)
        if row["aggregation"] == "POEM": poem_ok &= correct
        else: author_ok &= correct
    add(15, "CORE_POEM诗等权", bool(poem_ok), "核心每finding/model时期均值与直接逐诗mean复核")
    add(16, "CORE_AUTHOR作者等权", bool(author_ok), "先作者内mean再时期内作者mean复核")
    add(17, "bootstrap单位为作者", effects["bootstrap_unit"].eq("author").all(), "时期内整作者multinomial权重；同抽签同时产生POEM/AUTHOR")
    sample_equal = current_hashes["sample_labels"] == inputs["hashes"]["sample_labels"]
    for field in config["samples"].values():
        original_values = inputs["metadata"].set_index("poem_id")[field]
        sample_equal &= common[field].tolist() == common["poem_id"].map(original_values).tolist()
    add(18, "STRICT_CORE_EXPANDED定义不变", bool(sample_equal), "样本标签源文件哈希与共同集合逐poem_id标签均不变")
    synthetic = pd.DataFrame({"poem_count": [1, 3, 5], "t_mass": [0.1, 1.2, 0.8], "t_mean": [0.1, 0.4, 0.16]})
    first = bootstrap_period(synthetic, ["t"], 200, config["random_seed"])
    second = bootstrap_period(synthetic, ["t"], 200, config["random_seed"])
    add(19, "随机过程固定seed可复现", all(np.array_equal(first[a], second[a]) for a in AGGREGATIONS) and effects["random_seed"].eq(20260902).all(), "同输入同seed整数组精确一致；正式seed=20260902；各运行replicate哈希另存")
    banned = ("三个模型共同证明", "证明了文学史规律", "LITERARY_HISTORY_CONFIRMED")
    add(20, "不输出文学史确认语句", not any(term in report_text for term in banned), "报告只陈述语义/效应/不确定性与方法限制")
    return pd.DataFrame(rows)


def workbook_payload(frames: dict[str, pd.DataFrame], summary: dict[str, Any], output: Path) -> Path:
    """只为工作簿投影与类型化数据，不重新计算统计结果。"""
    specs = [
        ("Common_Universe_QC", "BCD_common_universe_qc", None),
        ("Semantic_Mapping", "finding_semantic_mapping", ["finding_id", "c_topic_id", "contrast", "target_model", "target_topic_ids", "target_topic_labels", "alignment_relation_type", "alignment_confidence", "semantic_similarity", "mapping_testability", "human_review_status", "alignment_review_required", "semantic_notes"]),
        ("Finding_Effects", "BCD_finding_effects", ["finding_id", "model_label", "sample", "aggregation", "mapped_topic_ids", "mapping_type", "period_A_mean", "period_B_mean", "raw_delta", "bootstrap_ci_low", "bootstrap_ci_high", "direction", "bootstrap_supported_direction", "hedges_g_if_applicable", "hedges_magnitude", "cliffs_delta_if_applicable", "cliffs_magnitude", "mean_E", "mean_H", "mean_M", "mean_L", "peak_period", "trough_period", "rank_order", "n_authors_A", "n_authors_B", "n_poems_A", "n_poems_B"]),
        ("CORE_Comparison", "BCD_finding_robustness_comparison", ["finding_id", "topic_label", "C_original_status", "C_author_robustness_status", "B_mapping_type", "D_mapping_type", "B_poem_delta", "C_poem_delta", "D_poem_delta", "B_author_delta", "C_author_delta", "D_author_delta", "B_supports_C_finding", "D_supports_C_finding", "cross_model_direction_pattern", "sample_direction_consistency", "semantic_mapping_assessment", "historical_effect_assessment", "finding_level_robustness", "robustness_reason"]),
        ("Sample_Sensitivity", "sample_direction_summary", None),
        ("Aggregation_Sensitivity", "aggregation_sensitivity", None),
        ("Priority_Four", "BCD_priority_four_results", ["finding_id", "model", "aggregation", "mapped_topic_ids", "mean_E", "mean_H", "mean_M", "mean_L", "raw_delta", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_supported_direction", "sample_direction_consistency", "finding_level_robustness"]),
        ("Mapping_Diagnostics", "split_single_best_sensitivity", ["finding_id", "model", "sample", "aggregation", "mapped_topic_ids", "mapping_variant", "raw_delta", "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_supported_direction", "primary_common_delta", "direction_agreement_with_primary", "delta_difference_from_primary"]),
        ("Model_Sensitivity", "model_sensitivity_diagnostics", None),
        ("Robustness_Ladder", "finding_robustness_ladder", ["robustness_rank", "finding_id", "topic_label", "B_mapping_type", "D_mapping_type", "sample_direction_consistency", "finding_level_robustness", "eligible_for_next_stage", "next_stage_note"]),
        ("Data_Dictionary", "data_dictionary", None),
    ]
    tables = []
    for sheet, dataset, fields in specs:
        frame = frames[dataset] if fields is None else frames[dataset][fields]
        payload = json.loads(frame.to_json(orient="split", force_ascii=False, double_precision=15))
        tables.append({"sheet": sheet, "dataset": dataset, "headers": payload["columns"], "data": payload["data"]})
    work = output / ".artifact_work"
    work.mkdir(exist_ok=True)
    path = work / "workbook_payload.json"
    path.write_text(json.dumps({"summary": summary, "tables": tables}, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path


def main() -> None:
    """运行完整发现层稳健性分析并写出可审计产物。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/BCD_robustness_config.yaml")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["bootstrap_iterations"] < 2000 or config["random_seed"] != 20260902:
        raise ValueError("本任务要求至少2000次bootstrap且seed固定20260902")
    output = resolve(config["output_root"])
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    print("读取冻结输入与来源eligibility...", flush=True)
    inputs = load_inputs(config)
    common, qc, native = common_universe(inputs, config)
    topics = sorted(inputs["candidates"]["topic_id"].unique())
    primary_parts, single_parts, native_parts = [], [], []
    semantic_frames, author_tables, bootstrap_hashes = {}, {}, {}
    for model in MODELS:
        print(f"{model}_COMMON：{len(inputs['candidates'])} findings，三样本×两聚合，{config['bootstrap_iterations']}次作者bootstrap", flush=True)
        frame = semantic_frame(common, inputs["matrices"][model], model, inputs["mapping"], topics, f"{model}_matrix_row")
        semantic_frames[model] = frame
        result, authors, digest = analyze(frame, model, inputs["candidates"], inputs["mapping"], config)
        primary_parts.append(result)
        author_tables[model] = authors
        bootstrap_hashes[f"{model}_COMMON"] = digest
        if model == "C":
            continue
        split_ids = inputs["mapping"].loc[inputs["mapping"]["target_model"].eq(model) & inputs["mapping"]["mapping_testability"].eq("BUNDLE"), "finding_id"]
        split_candidates = inputs["candidates"].loc[inputs["candidates"]["finding_id"].isin(split_ids)]
        if not split_candidates.empty:
            single_frame = semantic_frame(common, inputs["matrices"][model], model, inputs["mapping"], sorted(split_candidates["topic_id"].unique()), f"{model}_matrix_row", variant="SINGLE_BEST")
            single_result, _, single_digest = analyze(single_frame, model, split_candidates, inputs["mapping"], config, variant="SINGLE_BEST")
            single_parts.append(single_result)
            bootstrap_hashes[f"{model}_SINGLE_BEST"] = single_digest
        native_frame = semantic_frame(native[model], inputs["matrices"][model], model, inputs["mapping"], topics, "matrix_row")
        native_result, _, native_digest = analyze(native_frame, model, inputs["candidates"], inputs["mapping"], config, universe="NATIVE")
        native_parts.append(native_result)
        bootstrap_hashes[f"{model}_NATIVE"] = native_digest
    effects = pd.concat(primary_parts, ignore_index=True)
    singles = pd.concat(single_parts, ignore_index=True) if single_parts else effects.iloc[:0].copy()
    native_effects = pd.concat(native_parts, ignore_index=True)
    primary_lookup = effects.set_index(["finding_id", "model", "sample", "aggregation"])
    for frame in (singles, native_effects):
        frame["primary_common_delta"] = [primary_lookup.loc[(r.finding_id, r.model, r.sample, r.aggregation)]["raw_delta"] for r in frame.itertuples()]
        frame["direction_agreement_with_primary"] = [sign(r.raw_delta) == sign(r.primary_common_delta) for r in frame.itertuples()]
        frame["delta_difference_from_primary"] = frame["raw_delta"] - frame["primary_common_delta"]
    benchmark = compare_original_common(effects, inputs, config)
    samples, aggregations, shapes = build_sensitivity_tables(effects, config)
    exclusions = minimal_author_checks(author_tables, effects, inputs["candidates"], config)
    print("运行按条件触发的CLR诊断与最终分类...", flush=True)
    clr, clr_hashes = run_triggered_clr(inputs, common, effects, singles, config)
    bootstrap_hashes.update({f"{m}_CLR": h for m, h in clr_hashes.items()})
    comparison, diagnostics = build_comparison(inputs, effects, singles, shapes, exclusions, clr, benchmark, config)
    ladder = comparison.copy()
    ladder["class_order"] = ladder["finding_level_robustness"].map({v: i for i, v in enumerate(config["classification_order"])})
    ladder["mapping_order"] = [0 if b == d == "DIRECT" else 1 if "MERGED_PROXY" not in (b, d) else 2 for b, d in zip(ladder["B_mapping_type"], ladder["D_mapping_type"])]
    ladder = ladder.sort_values(["class_order", "mapping_order", "sample_consistency_minimum", "finding_id"], ascending=[True, True, False, True]).reset_index(drop=True)
    ladder.insert(0, "robustness_rank", np.arange(1, len(ladder) + 1))
    ladder["eligible_for_next_stage"] = ladder["finding_level_robustness"].eq("ROBUST_ACROSS_BCD")
    ladder["next_stage_note"] = np.where(ladder["eligible_for_next_stage"], "可进入人工解释与代表诗细读候选；须先确认来源mapping，仍不是历史结论", "先处理映射或方法敏感性，不列入跨模型稳健主叙述")
    priority = effects.loc[effects["sample"].eq("CORE") & effects["finding_id"].isin(config["priority_findings"])].merge(comparison[["finding_id", "finding_level_robustness", "sample_direction_consistency"]], on="finding_id", validate="many_to_one")
    frames = {
        "BCD_common_document_index": common, "BCD_common_universe_qc": qc, "finding_semantic_mapping": inputs["mapping"], "BCD_finding_effects": effects,
        "C_original_vs_common": benchmark, "model_native_sensitivity": native_effects, "split_single_best_sensitivity": singles, "sample_direction_summary": samples,
        "aggregation_sensitivity": aggregations, "shape_similarity": shapes, "priority_author_sensitivity": exclusions, "clr_sensitivity": clr,
        "BCD_finding_robustness_comparison": comparison, "model_sensitivity_diagnostics": diagnostics, "finding_robustness_ladder": ladder, "BCD_priority_four_results": priority,
        "excluded_findings_status": inputs["excluded"],
    }
    from bcd_robustness_reporting import build_figures, build_report, data_dictionary, readme_text
    anticipated = set(config["anticipated_findings_for_audit_only"])
    actual = set(inputs["candidates"]["finding_id"])
    audit = {"expected_but_not_eligible": sorted(anticipated - actual), "eligible_beyond_expected": sorted(actual - anticipated)}
    report_text = build_report(frames, config, audit)
    invariance = run_invariance_tests(inputs, common, effects, benchmark, frames, semantic_frames, report_text, config)
    frames["invariance_tests"] = invariance
    frames["data_dictionary"] = data_dictionary(frames)
    for name, frame in frames.items():
        frame.to_csv(output / "data" / f"{name}.csv", index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g")
    if not invariance["status"].eq("PASS").all():
        raise AssertionError(invariance.loc[invariance["status"].ne("PASS")].to_string(index=False))
    figures = build_figures(frames, output / "figures", config)
    report_path = output / "QTS_BCD_Finding_Level_Robustness_Report.md"
    report_path.write_text(report_text, encoding="utf-8", newline="\n")
    (output / "README.md").write_text(readme_text(config, frames), encoding="utf-8", newline="\n")
    shutil.copyfile(config_path, output / "BCD_robustness_config.yaml")
    summary = {
        "analysis_version": config["analysis_version"], "common_documents": len(common), "universe_qc": qc.loc[qc["sample"].eq("ALL")].to_dict("records"),
        "eligible_findings": sorted(actual), "eligible_list_audit": audit, "excluded_findings": inputs["excluded"]["finding_id"].tolist(),
        "mapping_counts": {m: {t: int(((inputs["mapping"]["target_model"] == m) & (inputs["mapping"]["mapping_testability"] == t)).sum()) for t in MAPPING_TYPES} for m in ("B", "D")},
        "robustness_counts": {s: int(comparison["finding_level_robustness"].eq(s).sum()) for s in config["classification_order"]},
        "findings": json.loads(comparison.to_json(orient="records", force_ascii=False)), "robustness_ladder": ladder[["robustness_rank", "finding_id", "finding_level_robustness", "eligible_for_next_stage"]].to_dict("records"),
        "bootstrap": {"iterations": config["bootstrap_iterations"], "seed": config["random_seed"], "replicate_hashes": bootstrap_hashes},
        "invariance_tests_passed": int(invariance["status"].eq("PASS").sum()), "original_common_max_absolute_delta_change": float(benchmark["absolute_delta_change"].max()),
        "native_max_absolute_delta_change": float(native_effects["delta_difference_from_primary"].abs().max()), "source_human_review_status": "NOT_RECORDED_IN_SOURCE",
    }
    (output / "data" / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")
    workbook_payload(frames, summary, output)
    output_paths = [output / "data" / f"{name}.csv" for name in frames] + figures + [report_path, output / "README.md", output / "BCD_robustness_config.yaml", output / "data" / "analysis_summary.json"]
    manifest = {"analysis_version": config["analysis_version"], "created_at": datetime.now().astimezone().isoformat(), "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
                "config_sha256": sha256_file(config_path), "implementation": {p.name: {"path": str(p.resolve()), "sha256": sha256_file(p)} for p in [Path(__file__), Path(__file__).with_name("bcd_robustness_reporting.py")]}, "inputs": {n: {"path": str(p), "sha256": inputs["hashes"][n]} for n, p in inputs["paths"].items()}, "bootstrap": summary["bootstrap"],
                "outputs": {str(p.relative_to(output)): {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in output_paths}, "invariance_tests_total": 20, "invariance_tests_passed": 20, "lda_refit_performed": False}
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"output_root": str(output), "common_documents": len(common), "mapping_counts": summary["mapping_counts"], "robustness_counts": summary["robustness_counts"], "invariance_tests_passed": 20}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
