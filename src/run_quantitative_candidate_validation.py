"""对《全唐诗》C 语料的 K11–15 与 K20 做量化候选二次验证。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib.externals.loky import get_reusable_executor
from scipy import sparse
from scipy.optimize import linear_sum_assignment

import run_exhaustive_k_scan as scan
import run_lower_k_boundary_perplexity as lower
import run_tf_df_lda as original


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "model_selection/quantitative_candidate_validation/config/candidate_validation_config.yaml"
REQUIRED_K = [11, 12, 13, 14, 15, 20]
REQUIRED_SEEDS = [42, 2026, 20260830, 7, 17, 101, 31415, 271828, 20260907, 12345]
REQUIRED_REGIMES = ["ORIGINAL_PRIOR", "FIXED_PRIOR_K20"]
FOCUS_COMPARISONS = [(12, 11), (12, 13), (12, 14), (12, 15), (12, 20), (15, 20)]
ALL_RUN_FIELDS = [
    "K",
    "seed",
    "prior_regime",
    "NPMI",
    "test_perplexity",
    "train_perplexity",
    "test_score",
    "train_score",
    "topic_diversity",
    "mean_pairwise_topic_cosine",
    "max_pairwise_topic_cosine",
    "mean_pairwise_top20_jaccard",
    "max_pairwise_top20_jaccard",
    "high_cosine_090_pairs",
    "high_cosine_085_pairs",
    "high_jaccard_050_pairs",
    "min_topic_prevalence",
    "max_topic_prevalence",
    "min_dominant_topic_share",
    "max_dominant_topic_share",
    "low_prevalence_topic_count",
    "fit_time",
    "n_iter",
    "doc_topic_prior_effective",
    "topic_word_prior_effective",
    "n_train",
    "n_test",
    "vocab_size",
]
SUMMARY_METRICS = [field for field in ALL_RUN_FIELDS if field not in {"K", "seed", "prior_regime", "n_train", "n_test", "vocab_size"}]


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """读取并严格验证候选实验的预注册配置。"""

    config = scan.read_json(path)
    if list(config["primary_candidates"]) != REQUIRED_K:
        raise ValueError(f"主候选必须正好为 {REQUIRED_K}")
    if list(config["supplementary_candidates"]) != []:
        raise ValueError("本次主验证不加入可选 K21，避免改变 120 fits 口径")
    if list(config["seeds"]) != REQUIRED_SEEDS:
        raise ValueError("十个随机种子或其登记顺序不符")
    if list(config["prior_regimes"]) != REQUIRED_REGIMES:
        raise ValueError("必须完整登记 ORIGINAL_PRIOR 与 FIXED_PRIOR_K20")
    original_prior = config["prior_regimes"]["ORIGINAL_PRIOR"]
    fixed_prior = config["prior_regimes"]["FIXED_PRIOR_K20"]
    if original_prior["doc_topic_prior"] is not None or original_prior["topic_word_prior"] is not None:
        raise ValueError("ORIGINAL_PRIOR 必须保持两个 prior=None")
    if float(fixed_prior["doc_topic_prior"]) != 0.05 or float(fixed_prior["topic_word_prior"]) != 0.05:
        raise ValueError("FIXED_PRIOR_K20 必须固定为 0.05/0.05")
    metrics = config["metrics"]
    expected_metrics = (
        metrics["coherence"] == "document_npmi"
        and metrics["npmi_reference"] == "TRAIN"
        and int(metrics["top_words"]) == 20
        and metrics["topic_diversity"] == "top_20_unique_ratio"
        and metrics["topic_matching"] == "hungarian_max_full_topic_word_cosine"
    )
    if not expected_metrics:
        raise ValueError("NPMI、Topic Diversity 或 Hungarian matching 定义偏离预注册口径")
    expected = config["expected"]
    if (expected["corpus"], int(expected["min_tf"]), int(expected["min_df"])) != ("C", 3, 3):
        raise ValueError("本实验只允许 C corpus（TF>=3、DF>=3）")
    if int(expected["main_fit_count"]) != 120:
        raise ValueError("主运行数必须固定为 120")
    allowed = (ROOT / "model_selection/quantitative_candidate_validation").resolve()
    if scan.resolve(config["output_root"]).resolve() != allowed:
        raise ValueError("输出目录越界")
    return config


def regime_lda(base_lda: Mapping[str, Any], config: Mapping[str, Any], regime: str) -> dict[str, Any]:
    """只按登记的 prior regime 改写两项先验。"""

    lda = dict(base_lda)
    definition = config["prior_regimes"][regime]
    lda["doc_topic_prior"] = definition["doc_topic_prior"]
    lda["topic_word_prior"] = definition["topic_word_prior"]
    return lda


def tracked_inputs(config: Mapping[str, Any], config_path: Path) -> list[Path]:
    """登记本阶段必须保持只读的冻结输入与统计实现。"""

    paths = scan.tracked_inputs(config)
    paths.extend(
        [
            config_path.resolve(),
            Path(__file__).resolve(),
            (ROOT / "src/run_exhaustive_k_scan.py").resolve(),
            (ROOT / "src/run_lower_k_boundary_perplexity.py").resolve(),
        ]
    )
    return sorted(set(paths), key=str)


def establish_input_manifest(config: Mapping[str, Any], config_path: Path, output: Path) -> dict[str, str]:
    """建立冻结输入哈希清单；已有拟合时拒绝变更。"""

    current = {str(path): scan.file_sha256(path) for path in tracked_inputs(config, config_path)}
    manifest_path = output / "audit/input_manifest.json"
    if manifest_path.is_file():
        previous = scan.read_json(manifest_path)
        if previous != current and any((output / "checkpoints").glob("*.json")):
            raise ValueError("已有候选拟合后冻结输入或实现发生变化，拒绝混用")
    scan.atomic_json(manifest_path, current)
    return current


def prepare(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """完成全部只读前置核验并装载冻结训练、测试矩阵。"""

    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    output = scan.resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    for name in ["data", "figures", "checkpoints", "topic_matrices", "audit", "logs"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    base_lda, original_audit = scan.validate_lda_source(config)
    matrix, vocabulary, documents, metadata = scan.load_corpus(config)
    train, test, split_audit = scan.load_frozen_split(config, documents)
    manifest = establish_input_manifest(config, config_path, output)
    regime_parameters: dict[str, dict[str, Any]] = {}
    for regime in REQUIRED_REGIMES:
        model = lower.build_model(REQUIRED_K[0], REQUIRED_SEEDS[0], regime_lda(base_lda, config, regime))
        regime_parameters[regime] = model.get_params(deep=False)
    source_hashes = {
        "topic_top_indices": hashlib.sha256(inspect.getsource(original.topic_top_indices).encode("utf-8")).hexdigest(),
        "topic_npmi_scores": hashlib.sha256(inspect.getsource(original.topic_npmi_scores).encode("utf-8")).hexdigest(),
        "topic_diversity": hashlib.sha256(inspect.getsource(original.topic_diversity).encode("utf-8")).hexdigest(),
        "model_factory": hashlib.sha256(inspect.getsource(lower.build_model).encode("utf-8")).hexdigest(),
        "hungarian_match": hashlib.sha256(inspect.getsource(match_seed_pair).encode("utf-8")).hexdigest(),
        "evaluate_one": hashlib.sha256(inspect.getsource(evaluate_one).encode("utf-8")).hexdigest(),
    }
    identity_data = {
        "schema_version": config["schema_version"],
        "config": config,
        "base_lda": base_lda,
        "original_audited_parameters": original_audit,
        "regime_parameters": regime_parameters,
        "matrix_sha256": config["expected"]["matrix_sha256"],
        "vocabulary_sha256": scan.file_sha256(scan.resolve(config["corpus_root"]) / "vocabulary.csv"),
        "split_audit": split_audit,
        "source_hashes": source_hashes,
    }
    identity = scan.stable_hash(identity_data)
    identity_path = output / "audit/run_identity.json"
    payload = {"identity": identity, **identity_data}
    if identity_path.is_file() and scan.read_json(identity_path) != payload and any((output / "checkpoints").glob("*.json")):
        raise ValueError("统计实现或配置已改变，拒绝复用旧检查点")
    scan.atomic_json(identity_path, payload)
    return {
        "config": config,
        "config_path": config_path,
        "output": output,
        "base_lda": base_lda,
        "audited_params": regime_parameters,
        "matrix": matrix,
        "vocabulary": vocabulary,
        "documents": documents,
        "metadata": metadata,
        "train_indices": train,
        "test_indices": test,
        "x_train": matrix[train],
        "x_test": matrix[test],
        "split_audit": split_audit,
        "input_manifest": manifest,
        "identity": identity,
        "vocabulary_sha256": identity_data["vocabulary_sha256"],
        "source_hashes": source_hashes,
    }


def topic_matrix_path(output: Path, regime: str, topic_count: int, seed: int) -> Path:
    """返回跨 seed 稳定性所需的最小主题矩阵产物路径。"""

    return output / "topic_matrices" / f"{regime}_K{topic_count}_seed_{seed}.npz"


def checkpoint_path(output: Path, regime: str, topic_count: int, seed: int) -> Path:
    """返回单次候选拟合检查点路径。"""

    return output / "checkpoints" / f"{regime}_K{topic_count}_seed_{seed}.json"


def checkpoint_signature(context: Mapping[str, Any], regime: str, topic_count: int, seed: int) -> str:
    """生成包含 regime、K 与 seed 的单次拟合签名。"""

    return scan.stable_hash({"identity": context["identity"], "prior_regime": regime, "K": topic_count, "seed": seed})


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """原子保存压缩 NumPy 矩阵。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def pairwise_jaccard(top_indices: Sequence[Sequence[int]]) -> np.ndarray:
    """计算同一模型全部 Topic pair 的 top-20 Jaccard。"""

    values: list[float] = []
    sets = [set(map(int, topic)) for topic in top_indices]
    for first, second in itertools.combinations(sets, 2):
        union = first | second
        values.append(len(first & second) / len(union) if union else 1.0)
    return np.asarray(values, dtype=np.float64)


def off_diagonal_upper(matrix: np.ndarray) -> np.ndarray:
    """提取对称 Topic 相似度矩阵的严格上三角。"""

    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def evaluate_one(
    context: Mapping[str, Any], regime: str, topic_count: int, seed: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    """仅在 X_train 拟合一个候选模型并计算全部逐 run 指标。"""

    config = context["config"]
    lda = regime_lda(context["base_lda"], config, regime)
    model = lower.build_model(topic_count, seed, lda)
    fit_parameters = model.get_params(deep=False)
    started = time.perf_counter()
    model.fit(context["x_train"])
    fit_time = time.perf_counter() - started
    components_hash = hashlib.sha256(model.components_.tobytes()).hexdigest()
    topic_word = model.components_ / model.components_.sum(axis=1, keepdims=True)
    norms = np.linalg.norm(topic_word, axis=1, keepdims=True)
    topic_word_unit = np.divide(topic_word, norms, out=np.zeros_like(topic_word), where=norms > 0)
    top_indices = original.topic_top_indices(topic_word, int(config["metrics"]["top_words"]))
    _, mean_npmi = original.topic_npmi_scores(context["x_train"], top_indices)
    diversity = original.topic_diversity(top_indices)
    cosine_pairs = off_diagonal_upper(np.clip(topic_word_unit @ topic_word_unit.T, -1.0, 1.0))
    jaccard_pairs = pairwise_jaccard(top_indices)

    model.set_params(n_jobs=1)
    document_topic = model.transform(context["x_train"])
    prevalence = document_topic.mean(axis=0)
    dominant_counts = np.bincount(np.argmax(document_topic, axis=1), minlength=topic_count)
    dominant_shares = dominant_counts / document_topic.shape[0]
    train_score, train_perplexity = scan.score_and_perplexity(model, context["x_train"])
    test_score, test_perplexity = scan.score_and_perplexity(model, context["x_test"])
    if components_hash != hashlib.sha256(model.components_.tobytes()).hexdigest():
        raise ValueError("指标计算改变了 Topic-word 参数")
    numeric = [mean_npmi, diversity, *cosine_pairs.tolist(), *jaccard_pairs.tolist(), *prevalence.tolist()]
    if not np.isfinite(np.asarray(numeric, dtype=float)).all():
        raise ValueError("候选模型产生非有限指标")
    low_threshold = float(config["metrics"]["low_prevalence_threshold"])
    result = {
        "K": int(topic_count),
        "seed": int(seed),
        "prior_regime": regime,
        "NPMI": float(mean_npmi),
        "test_perplexity": float(test_perplexity),
        "train_perplexity": float(train_perplexity),
        "test_score": float(test_score),
        "train_score": float(train_score),
        "topic_diversity": float(diversity),
        "mean_pairwise_topic_cosine": float(cosine_pairs.mean()),
        "max_pairwise_topic_cosine": float(cosine_pairs.max()),
        "mean_pairwise_top20_jaccard": float(jaccard_pairs.mean()),
        "max_pairwise_top20_jaccard": float(jaccard_pairs.max()),
        "high_cosine_090_pairs": int(np.sum(cosine_pairs >= 0.90)),
        "high_cosine_085_pairs": int(np.sum(cosine_pairs >= 0.85)),
        "high_jaccard_050_pairs": int(np.sum(jaccard_pairs >= 0.50)),
        "min_topic_prevalence": float(prevalence.min()),
        "max_topic_prevalence": float(prevalence.max()),
        "min_dominant_topic_share": float(dominant_shares.min()),
        "max_dominant_topic_share": float(dominant_shares.max()),
        "low_prevalence_topic_count": int(np.sum(prevalence < low_threshold)),
        "fit_time": float(fit_time),
        "n_iter": int(model.n_iter_),
        "doc_topic_prior_effective": float(model.doc_topic_prior_),
        "topic_word_prior_effective": float(model.topic_word_prior_),
        "n_train": int(context["x_train"].shape[0]),
        "n_test": int(context["x_test"].shape[0]),
        "vocab_size": int(context["x_train"].shape[1]),
    }
    topic_details = [
        {
            "topic_id": index + 1,
            "mean_probability": float(prevalence[index]),
            "dominant_topic_document_count": int(dominant_counts[index]),
            "dominant_share": float(dominant_shares[index]),
            "low_prevalence_flag": bool(prevalence[index] < low_threshold),
        }
        for index in range(topic_count)
    ]
    audit = {
        "fit_scope": "X_train_only",
        "test_used_for_fit": False,
        "npmi_reference": "X_train_only",
        "npmi_top_words": int(config["metrics"]["top_words"]),
        "topic_diversity_definition": "top_20_unique_ratio",
        "matching_definition": config["metrics"]["topic_matching"],
        "fit_n_jobs": fit_parameters["n_jobs"],
        "evaluation_n_jobs": 1,
        "model_parameters": fit_parameters,
        "components_sha256": components_hash,
        "topic_prevalence": topic_details,
    }
    arrays = {
        "topic_word_unit": topic_word_unit.astype(np.float32),
        "top_indices": np.asarray(top_indices, dtype=np.int32),
    }
    del model, topic_word, topic_word_unit, document_topic
    return result, audit, arrays


def load_checkpoint(path: Path, expected_signature: str) -> dict[str, Any]:
    """严格读取检查点并复核其主题矩阵哈希。"""

    item = scan.read_json(path)
    if item.get("checkpoint_signature") != expected_signature:
        raise ValueError(f"检查点签名不符：{path}")
    if item.get("result_sha256") != scan.stable_hash(item.get("result")):
        raise ValueError(f"检查点结果哈希不符：{path}")
    matrix_path = scan.resolve(item["topic_matrix_path"])
    if not matrix_path.is_file() or scan.file_sha256(matrix_path) != item.get("topic_matrix_sha256"):
        raise ValueError(f"检查点主题矩阵缺失或哈希不符：{path}")
    return item


def collect_checkpoints(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """按预注册顺序收集所有有效检查点。"""

    rows: list[dict[str, Any]] = []
    for regime, topic_count, seed in itertools.product(REQUIRED_REGIMES, REQUIRED_K, REQUIRED_SEEDS):
        path = checkpoint_path(context["output"], regime, topic_count, seed)
        if path.is_file():
            rows.append(load_checkpoint(path, checkpoint_signature(context, regime, topic_count, seed))["result"])
    return rows


def write_all_runs(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """持续刷新逐 run 主表。"""

    frame = pd.DataFrame(rows, columns=ALL_RUN_FIELDS)
    if len(frame):
        frame = frame.sort_values(["prior_regime", "K", "seed"])
    scan.atomic_frame(output / "data/quantitative_candidate_all_runs.csv", frame)


def load_topic_arrays(context: Mapping[str, Any], regime: str, topic_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """读取经检查点哈希验证的主题单位向量与 top-20 索引。"""

    item = load_checkpoint(
        checkpoint_path(context["output"], regime, topic_count, seed),
        checkpoint_signature(context, regime, topic_count, seed),
    )
    with np.load(scan.resolve(item["topic_matrix_path"])) as data:
        topic_word_unit = np.asarray(data["topic_word_unit"], dtype=np.float64)
        top_indices = np.asarray(data["top_indices"], dtype=np.int32)
    if topic_word_unit.shape != (topic_count, int(context["config"]["expected"]["vocab_size"])):
        raise ValueError("跨 seed 稳定性矩阵 shape 不符")
    if top_indices.shape != (topic_count, 20):
        raise ValueError("跨 seed top-20 索引 shape 不符")
    return topic_word_unit, top_indices


def match_seed_pair(
    first_unit: np.ndarray,
    second_unit: np.ndarray,
    first_top: np.ndarray,
    second_top: np.ndarray,
) -> tuple[float, float]:
    """按完整 Topic-word cosine 做 Hungarian 匹配，再在同一配对算 Jaccard。"""

    similarity = np.clip(first_unit @ second_unit.T, -1.0, 1.0)
    first_rows, second_columns = linear_sum_assignment(-similarity)
    cosine_values = similarity[first_rows, second_columns]
    jaccard_values: list[float] = []
    for first_index, second_index in zip(first_rows, second_columns):
        first_set = set(map(int, first_top[first_index]))
        second_set = set(map(int, second_top[second_index]))
        union = first_set | second_set
        jaccard_values.append(len(first_set & second_set) / len(union) if union else 1.0)
    return float(np.mean(cosine_values)), float(np.mean(jaccard_values))


def build_cross_seed_stability(context: Mapping[str, Any]) -> pd.DataFrame:
    """生成同 K、同 prior regime 的全部 45 个 seed pair 稳定性。"""

    rows: list[dict[str, Any]] = []
    for regime, topic_count in itertools.product(REQUIRED_REGIMES, REQUIRED_K):
        loaded = {
            seed: load_topic_arrays(context, regime, topic_count, seed)
            for seed in REQUIRED_SEEDS
        }
        group_rows: list[dict[str, Any]] = []
        for seed_a, seed_b in itertools.combinations(REQUIRED_SEEDS, 2):
            first_unit, first_top = loaded[seed_a]
            second_unit, second_top = loaded[seed_b]
            matched_cosine, matched_jaccard = match_seed_pair(
                first_unit, second_unit, first_top, second_top
            )
            group_rows.append(
                {
                    "K": topic_count,
                    "prior_regime": regime,
                    "seed_A": seed_a,
                    "seed_B": seed_b,
                    "matched_cosine": matched_cosine,
                    "matched_jaccard": matched_jaccard,
                    "matching_method": "HUNGARIAN_MAX_FULL_TOPIC_WORD_COSINE",
                }
            )
        cosine = pd.Series([row["matched_cosine"] for row in group_rows], dtype=float)
        jaccard = pd.Series([row["matched_jaccard"] for row in group_rows], dtype=float)
        additions = {
            "mean_matched_cosine": float(cosine.mean()),
            "sd_matched_cosine": float(cosine.std(ddof=1)),
            "median_matched_cosine": float(cosine.median()),
            "minimum_pairwise_seed_stability": float(cosine.min()),
            "maximum_matched_cosine": float(cosine.max()),
            "mean_matched_jaccard": float(jaccard.mean()),
            "sd_matched_jaccard": float(jaccard.std(ddof=1)),
            "median_matched_jaccard": float(jaccard.median()),
            "minimum_matched_jaccard": float(jaccard.min()),
            "maximum_matched_jaccard": float(jaccard.max()),
            "seed_pair_count": len(group_rows),
        }
        for row in group_rows:
            row.update(additions)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["prior_regime", "K", "seed_A", "seed_B"]).reset_index(drop=True)


def five_number_summary(values: pd.Series, ddof: int) -> dict[str, float]:
    """返回 mean、median、SD、min、max 五类描述统计。"""

    numeric = values.astype(float)
    return {
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "sd": float(numeric.std(ddof=ddof)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def build_candidate_summary(
    all_runs: pd.DataFrame, stability: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """按 K×prior regime 汇总全部运行指标与跨 seed 稳定性。"""

    ddof = int(config["metrics"]["standard_deviation_ddof"])
    rows: list[dict[str, Any]] = []
    for (regime, topic_count), frame in all_runs.groupby(["prior_regime", "K"], sort=False):
        row: dict[str, Any] = {
            "K": int(topic_count),
            "prior_regime": regime,
            "seed_count": int(frame["seed"].nunique()),
        }
        for metric in SUMMARY_METRICS:
            for statistic, value in five_number_summary(frame[metric], ddof).items():
                row[f"{metric}_{statistic}"] = value
        stable = stability[(stability["prior_regime"] == regime) & (stability["K"] == topic_count)]
        for metric in ["matched_cosine", "matched_jaccard"]:
            for statistic, value in five_number_summary(stable[metric], ddof).items():
                row[f"{metric}_{statistic}"] = value
        row["mean_matched_cosine"] = float(stable["matched_cosine"].mean())
        row["sd_matched_cosine"] = float(stable["matched_cosine"].std(ddof=ddof))
        row["mean_matched_jaccard"] = float(stable["matched_jaccard"].mean())
        row["sd_matched_jaccard"] = float(stable["matched_jaccard"].std(ddof=ddof))
        row["minimum_pairwise_seed_stability"] = float(stable["matched_cosine"].min())
        row["minimum_matched_jaccard"] = float(stable["matched_jaccard"].min())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["NPMI_rank"] = summary.groupby("prior_regime")["NPMI_mean"].rank(method="min", ascending=False).astype(int)
    summary["test_perplexity_rank"] = summary.groupby("prior_regime")["test_perplexity_mean"].rank(method="min", ascending=True).astype(int)
    relative_band = float(config["practical_equivalence"]["relative_global_max_band"])
    maxima = summary.groupby("prior_regime")["NPMI_mean"].transform("max")
    summary["NPMI_within_1pct_global_max"] = summary["NPMI_mean"] >= maxima * (1.0 - relative_band)
    return summary.sort_values(["prior_regime", "K"]).reset_index(drop=True)


def practical_equivalence_status(
    mean_difference: float,
    both_in_relative_band: bool,
    config: Mapping[str, Any],
) -> str:
    """并列报告两个绝对尺度与一个相对全局峰尺度。"""

    bands = [float(value) for value in config["practical_equivalence"]["absolute_npmi_bands"]]
    labels = [
        f"ABS_LE_{band:.4f}" if abs(mean_difference) <= band else f"ABS_GT_{band:.4f}"
        for band in bands
    ]
    labels.append("BOTH_WITHIN_1PCT_GLOBAL_MAX" if both_in_relative_band else "NOT_BOTH_WITHIN_1PCT_GLOBAL_MAX")
    return "|".join(labels)


def build_pairwise_comparison(
    all_runs: pd.DataFrame, summary: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """生成六组预指定的 seed-matched 候选差异。"""

    rows: list[dict[str, Any]] = []
    for regime in REQUIRED_REGIMES:
        regime_summary = summary[summary["prior_regime"] == regime].set_index("K")
        for k_a, k_b in FOCUS_COMPARISONS:
            first = all_runs[(all_runs["prior_regime"] == regime) & (all_runs["K"] == k_a)].set_index("seed")
            second = all_runs[(all_runs["prior_regime"] == regime) & (all_runs["K"] == k_b)].set_index("seed")
            if set(first.index) != set(REQUIRED_SEEDS) or set(second.index) != set(REQUIRED_SEEDS):
                raise ValueError("seed-matched 差异缺少预注册 seed")
            npmi_diff = first.loc[REQUIRED_SEEDS, "NPMI"].astype(float) - second.loc[REQUIRED_SEEDS, "NPMI"].astype(float)
            ppl_diff = first.loc[REQUIRED_SEEDS, "test_perplexity"].astype(float) - second.loc[REQUIRED_SEEDS, "test_perplexity"].astype(float)
            diversity_diff = first.loc[REQUIRED_SEEDS, "topic_diversity"].astype(float) - second.loc[REQUIRED_SEEDS, "topic_diversity"].astype(float)
            tolerance = 1e-12
            ties = int(np.sum(np.abs(npmi_diff.to_numpy()) <= tolerance))
            both_relative = bool(
                regime_summary.loc[k_a, "NPMI_within_1pct_global_max"]
                and regime_summary.loc[k_b, "NPMI_within_1pct_global_max"]
            )
            mean_difference = float(npmi_diff.mean())
            rows.append(
                {
                    "K_A": k_a,
                    "K_B": k_b,
                    "prior_regime": regime,
                    "mean_npmi_difference": mean_difference,
                    "median_npmi_difference": float(npmi_diff.median()),
                    "seed_sign_A_gt_B": int(np.sum(npmi_diff.to_numpy() > tolerance)),
                    "seed_sign_B_gt_A": int(np.sum(npmi_diff.to_numpy() < -tolerance)),
                    "ties": ties,
                    "bootstrap_ci_low": "",
                    "bootstrap_ci_high": "",
                    "bootstrap_scope": "NOT_USED",
                    "perplexity_difference": float(ppl_diff.mean()),
                    "median_perplexity_difference": float(ppl_diff.median()),
                    "topic_diversity_difference": float(diversity_diff.mean()),
                    "practical_equivalence_status": practical_equivalence_status(mean_difference, both_relative, config),
                }
            )
    return pd.DataFrame(rows)


def build_prior_sensitivity(summary: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """比较每个 K 在两套 prior 下的均值、排名与顺序反转。"""

    original_rows = summary[summary["prior_regime"] == "ORIGINAL_PRIOR"].set_index("K")
    fixed_rows = summary[summary["prior_regime"] == "FIXED_PRIOR_K20"].set_index("K")
    threshold = int(config["status_rules"]["prior_sensitive_rank_change"])
    rows: list[dict[str, Any]] = []
    for topic_count in REQUIRED_K:
        original_rank = int(original_rows.loc[topic_count, "NPMI_rank"])
        fixed_rank = int(fixed_rows.loc[topic_count, "NPMI_rank"])
        reversals = 0
        for other in REQUIRED_K:
            if other == topic_count:
                continue
            original_sign = np.sign(original_rows.loc[topic_count, "NPMI_mean"] - original_rows.loc[other, "NPMI_mean"])
            fixed_sign = np.sign(fixed_rows.loc[topic_count, "NPMI_mean"] - fixed_rows.loc[other, "NPMI_mean"])
            reversals += int(original_sign != 0 and fixed_sign != 0 and original_sign != fixed_sign)
        platform_change = bool(
            original_rows.loc[topic_count, "NPMI_within_1pct_global_max"]
            != fixed_rows.loc[topic_count, "NPMI_within_1pct_global_max"]
        )
        rank_change = fixed_rank - original_rank
        sensitive = abs(rank_change) >= threshold or platform_change
        rows.append(
            {
                "K": topic_count,
                "original_mean_NPMI": float(original_rows.loc[topic_count, "NPMI_mean"]),
                "fixed_mean_NPMI": float(fixed_rows.loc[topic_count, "NPMI_mean"]),
                "original_rank": original_rank,
                "fixed_rank": fixed_rank,
                "rank_change": rank_change,
                "pairwise_order_reversals": reversals,
                "original_perplexity": float(original_rows.loc[topic_count, "test_perplexity_mean"]),
                "fixed_perplexity": float(fixed_rows.loc[topic_count, "test_perplexity_mean"]),
                "original_within_1pct": bool(original_rows.loc[topic_count, "NPMI_within_1pct_global_max"]),
                "fixed_within_1pct": bool(fixed_rows.loc[topic_count, "NPMI_within_1pct_global_max"]),
                "prior_sensitivity_status": "PRIOR_SENSITIVE" if sensitive else "PRIOR_ROBUST",
            }
        )
    return pd.DataFrame(rows)


def dominates(first: Mapping[str, float], second: Mapping[str, float], tolerance: float = 1e-12) -> bool:
    """判断 first 是否在全部多目标维度不劣且至少一项严格更优。"""

    directions = {
        "NPMI": 1.0,
        "test_perplexity": -1.0,
        "topic_diversity": 1.0,
        "cosine_stability": 1.0,
        "jaccard_stability": 1.0,
        "redundancy": -1.0,
        "low_prevalence": -1.0,
    }
    no_worse = True
    strictly_better = False
    for metric, direction in directions.items():
        first_value = direction * float(first[metric])
        second_value = direction * float(second[metric])
        no_worse &= first_value >= second_value - tolerance
        strictly_better |= first_value > second_value + tolerance
    return bool(no_worse and strictly_better)


def build_pareto_front(summary: pd.DataFrame) -> pd.DataFrame:
    """在每套 prior 内独立计算不加权的 Pareto 非支配集合。"""

    rows: list[dict[str, Any]] = []
    for regime in REQUIRED_REGIMES:
        frame = summary[summary["prior_regime"] == regime].set_index("K")
        vectors = {
            topic_count: {
                "NPMI": frame.loc[topic_count, "NPMI_mean"],
                "test_perplexity": frame.loc[topic_count, "test_perplexity_mean"],
                "topic_diversity": frame.loc[topic_count, "topic_diversity_mean"],
                "cosine_stability": frame.loc[topic_count, "mean_matched_cosine"],
                "jaccard_stability": frame.loc[topic_count, "mean_matched_jaccard"],
                "redundancy": frame.loc[topic_count, "mean_pairwise_topic_cosine_mean"],
                "low_prevalence": frame.loc[topic_count, "low_prevalence_topic_count_mean"],
            }
            for topic_count in REQUIRED_K
        }
        for topic_count in REQUIRED_K:
            dominators = [other for other in REQUIRED_K if other != topic_count and dominates(vectors[other], vectors[topic_count])]
            rows.append(
                {
                    "K": topic_count,
                    "prior_regime": regime,
                    "is_pareto_nondominated": not dominators,
                    "dominated_by": ";".join(f"K{value}" for value in dominators),
                }
            )
    return pd.DataFrame(rows)


def tukey_fence(values: Sequence[float], multiplier: float, upper: bool) -> float:
    """计算跨候选相对异常诊断的 Tukey fence。"""

    array = np.asarray(values, dtype=float)
    first, third = np.quantile(array, [0.25, 0.75])
    spread = third - first
    return float(third + multiplier * spread if upper else first - multiplier * spread)


def classify_candidate_statuses(
    summary: pd.DataFrame,
    prior: pd.DataFrame,
    pareto: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """按预注册相对异常与平台规则给每个 K 多标签状态。"""

    multiplier = float(config["status_rules"]["outlier_iqr_multiplier"])
    flags: dict[tuple[str, int], dict[str, bool]] = {}
    for regime in REQUIRED_REGIMES:
        frame = summary[summary["prior_regime"] == regime].set_index("K")
        npmi_sd_upper = tukey_fence(frame["NPMI_sd"], multiplier, upper=True)
        cosine_lower = tukey_fence(frame["mean_matched_cosine"], multiplier, upper=False)
        jaccard_lower = tukey_fence(frame["mean_matched_jaccard"], multiplier, upper=False)
        mean_redundancy_upper = tukey_fence(frame["mean_pairwise_topic_cosine_mean"], multiplier, upper=True)
        max_redundancy_upper = tukey_fence(frame["max_pairwise_topic_cosine_mean"], multiplier, upper=True)
        high_pairs_upper = tukey_fence(frame["high_cosine_085_pairs_mean"], multiplier, upper=True)
        cosine_median = float(frame["mean_matched_cosine"].median())
        jaccard_median = float(frame["mean_matched_jaccard"].median())
        for topic_count in REQUIRED_K:
            flags[(regime, topic_count)] = {
                "seed_sensitive": bool(
                    frame.loc[topic_count, "NPMI_sd"] > npmi_sd_upper
                    or frame.loc[topic_count, "mean_matched_cosine"] < cosine_lower
                    or frame.loc[topic_count, "mean_matched_jaccard"] < jaccard_lower
                ),
                "redundancy_risk": bool(
                    frame.loc[topic_count, "mean_pairwise_topic_cosine_mean"] > mean_redundancy_upper
                    or frame.loc[topic_count, "max_pairwise_topic_cosine_mean"] > max_redundancy_upper
                    or frame.loc[topic_count, "high_cosine_085_pairs_mean"] > high_pairs_upper
                ),
                "stable_at_or_above_median": bool(
                    frame.loc[topic_count, "mean_matched_cosine"] >= cosine_median
                    and frame.loc[topic_count, "mean_matched_jaccard"] >= jaccard_median
                ),
                "platform": bool(frame.loc[topic_count, "NPMI_within_1pct_global_max"]),
            }
    prior_map = prior.set_index("K")["prior_sensitivity_status"].to_dict()
    rows: list[dict[str, Any]] = []
    for topic_count in REQUIRED_K:
        seed_sensitive = any(flags[(regime, topic_count)]["seed_sensitive"] for regime in REQUIRED_REGIMES)
        redundancy_risk = any(flags[(regime, topic_count)]["redundancy_risk"] for regime in REQUIRED_REGIMES)
        platform_both = all(flags[(regime, topic_count)]["platform"] for regime in REQUIRED_REGIMES)
        platform_any = any(flags[(regime, topic_count)]["platform"] for regime in REQUIRED_REGIMES)
        stable_both = all(flags[(regime, topic_count)]["stable_at_or_above_median"] for regime in REQUIRED_REGIMES)
        prior_sensitive = prior_map[topic_count] == "PRIOR_SENSITIVE"
        pareto_any = bool(pareto[(pareto["K"] == topic_count)]["is_pareto_nondominated"].any())
        labels: list[str] = []
        if platform_both and stable_both and not prior_sensitive and not seed_sensitive and not redundancy_risk:
            labels.append("QUANT_STRONG")
        elif platform_any or pareto_any:
            labels.append("QUANT_COMPETITIVE")
        else:
            labels.append("WEAK_QUANT_CANDIDATE")
        if prior_sensitive:
            labels.append("PRIOR_SENSITIVE")
        if seed_sensitive:
            labels.append("SEED_SENSITIVE")
        if redundancy_risk:
            labels.append("REDUNDANCY_RISK")
        rows.append(
            {
                "K": topic_count,
                "quantitative_status": "|".join(labels),
                "platform_under_both_priors": platform_both,
                "seed_sensitive": seed_sensitive,
                "redundancy_risk": redundancy_risk,
                "prior_sensitive": prior_sensitive,
                "pareto_nondominated_in_any_regime": pareto_any,
            }
        )
    return pd.DataFrame(rows)


def classify_plateau(summary: pd.DataFrame) -> dict[str, Any]:
    """判断 K11–15 在两套 prior 下是否保持连续高值平台。"""

    result: dict[str, Any] = {}
    for regime in REQUIRED_REGIMES:
        frame = summary[(summary["prior_regime"] == regime) & (summary["K"].between(11, 15))]
        members = frame.loc[frame["NPMI_within_1pct_global_max"], "K"].astype(int).tolist()
        ranges = scan.contiguous_runs(members)
        longest = max((end - start + 1 for start, end in ranges), default=0)
        if members == [11, 12, 13, 14, 15]:
            status = "BROAD_PLATEAU"
        elif longest >= 3:
            status = "PARTIAL_PLATEAU"
        else:
            status = "DISTINCT_PEAK"
        result[regime] = {"status": status, "members": members, "longest_contiguous": longest}
    return result


def build_shortlist(
    summary: pd.DataFrame,
    statuses: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """用四个透明角色槽位生成最多四个 K 的量化 shortlist。"""

    indexed = summary.set_index(["prior_regime", "K"])
    status_index = statuses.set_index("K")
    reasons: dict[int, list[str]] = {}
    order: list[int] = []

    def add(topic_count: int, reason: str) -> None:
        reasons.setdefault(topic_count, []).append(reason)
        if topic_count not in order:
            order.append(topic_count)

    original_best = int(summary[summary["prior_regime"] == "ORIGINAL_PRIOR"].sort_values(["NPMI_mean", "K"], ascending=[False, True]).iloc[0]["K"])
    fixed_best = int(summary[summary["prior_regime"] == "FIXED_PRIOR_K20"].sort_values(["NPMI_mean", "K"], ascending=[False, True]).iloc[0]["K"])
    add(original_best, "ORIGINAL_PRIOR_NPMI_LEADER")
    add(fixed_best, "FIXED_PRIOR_NPMI_LEADER")
    stable_pool: list[tuple[float, float, int]] = []
    for topic_count in [11, 12, 13, 14, 15]:
        status = status_index.loc[topic_count]
        if bool(status["platform_under_both_priors"]) and not bool(status["seed_sensitive"]) and not bool(status["redundancy_risk"]):
            cosine_floor = min(float(indexed.loc[(regime, topic_count), "mean_matched_cosine"]) for regime in REQUIRED_REGIMES)
            jaccard_floor = min(float(indexed.loc[(regime, topic_count), "mean_matched_jaccard"]) for regime in REQUIRED_REGIMES)
            stable_pool.append((cosine_floor, jaccard_floor, topic_count))
    if not stable_pool:
        for topic_count in [11, 12, 13, 14, 15]:
            status = status_index.loc[topic_count]
            if not bool(status["seed_sensitive"]) and not bool(status["redundancy_risk"]):
                cosine_floor = min(float(indexed.loc[(regime, topic_count), "mean_matched_cosine"]) for regime in REQUIRED_REGIMES)
                jaccard_floor = min(float(indexed.loc[(regime, topic_count), "mean_matched_jaccard"]) for regime in REQUIRED_REGIMES)
                stable_pool.append((cosine_floor, jaccard_floor, topic_count))
    if stable_pool:
        stability_best = sorted(stable_pool, key=lambda item: (-item[0], -item[1], item[2]))[0][2]
        add(stability_best, "CROSS_SEED_STABILITY_LEADER_WITHOUT_WARNING")
    add(20, "FROZEN_PRIMARY_REFERENCE")
    maximum = int(config["status_rules"]["shortlist_maximum"])
    if len(order) > maximum:
        raise ValueError("预注册 shortlist 角色产生超过四个候选")
    rows: list[dict[str, Any]] = []
    for selection_order, topic_count in enumerate(order, 1):
        row: dict[str, Any] = {
            "selection_order": selection_order,
            "K": topic_count,
            "selection_reason": "|".join(reasons[topic_count]),
            "quantitative_status": status_index.loc[topic_count, "quantitative_status"],
        }
        for regime, prefix in [("ORIGINAL_PRIOR", "original"), ("FIXED_PRIOR_K20", "fixed")]:
            current = indexed.loc[(regime, topic_count)]
            row[f"{prefix}_mean_NPMI"] = float(current["NPMI_mean"])
            row[f"{prefix}_mean_test_perplexity"] = float(current["test_perplexity_mean"])
            row[f"{prefix}_mean_matched_cosine"] = float(current["mean_matched_cosine"])
            row[f"{prefix}_mean_matched_jaccard"] = float(current["mean_matched_jaccard"])
            row[f"{prefix}_mean_redundancy_cosine"] = float(current["mean_pairwise_topic_cosine_mean"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_topic_prevalence_details(context: Mapping[str, Any]) -> pd.DataFrame:
    """从检查点导出每个模型、每个 Topic 的 prevalence 诊断。"""

    rows: list[dict[str, Any]] = []
    for regime, topic_count, seed in itertools.product(REQUIRED_REGIMES, REQUIRED_K, REQUIRED_SEEDS):
        item = load_checkpoint(
            checkpoint_path(context["output"], regime, topic_count, seed),
            checkpoint_signature(context, regime, topic_count, seed),
        )
        for topic in item["audit"]["topic_prevalence"]:
            rows.append({"K": topic_count, "seed": seed, "prior_regime": regime, **topic})
    return pd.DataFrame(rows).sort_values(["prior_regime", "K", "seed", "topic_id"]).reset_index(drop=True)


def plot_candidate_errorbars(
    summary: pd.DataFrame,
    metric: str,
    sd_metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """绘制两套 prior 的候选均值与标准差。"""

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    offsets = {"ORIGINAL_PRIOR": -0.08, "FIXED_PRIOR_K20": 0.08}
    colors = {"ORIGINAL_PRIOR": "#245A7A", "FIXED_PRIOR_K20": "#B65C3A"}
    for regime in REQUIRED_REGIMES:
        frame = summary[summary["prior_regime"] == regime].set_index("K").loc[REQUIRED_K]
        x = np.arange(len(REQUIRED_K), dtype=float) + offsets[regime]
        ax.errorbar(x, frame[metric], yerr=frame[sd_metric], marker="o", capsize=4, linewidth=1.8, color=colors[regime], label=regime)
    ax.set_xticks(np.arange(len(REQUIRED_K)), [f"K{value}" for value in REQUIRED_K])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_figures(
    all_runs: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
) -> None:
    """生成任务指定的六张候选验证图。"""

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    plot_candidate_errorbars(
        summary,
        "NPMI_mean",
        "NPMI_sd",
        "Train-split NPMI (higher is better)",
        "Candidate NPMI: original vs fixed prior (mean ± SD, 10 seeds)",
        figures / "candidate_NPMI_original_vs_fixed_prior.png",
    )
    plot_candidate_errorbars(
        summary,
        "test_perplexity_mean",
        "test_perplexity_sd",
        "Held-out perplexity (lower is better)",
        "Candidate perplexity: original vs fixed prior (mean ± SD, 10 seeds)",
        figures / "candidate_perplexity_original_vs_fixed_prior.png",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    for regime, color, marker in [("ORIGINAL_PRIOR", "#245A7A", "o"), ("FIXED_PRIOR_K20", "#B65C3A", "s")]:
        frame = summary[summary["prior_regime"] == regime].set_index("K").loc[REQUIRED_K]
        axes[0].errorbar(REQUIRED_K, frame["mean_matched_cosine"], yerr=frame["sd_matched_cosine"], marker=marker, capsize=3, color=color, label=regime)
        axes[1].errorbar(REQUIRED_K, frame["mean_matched_jaccard"], yerr=frame["sd_matched_jaccard"], marker=marker, capsize=3, color=color, label=regime)
    axes[0].set_title("A. Hungarian matched cosine")
    axes[1].set_title("B. Matched top-20 Jaccard")
    for ax in axes:
        ax.set_xlabel("K")
        ax.set_xticks(REQUIRED_K)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Pairwise-seed stability (mean ± SD)")
    axes[1].legend(frameon=False)
    fig.suptitle("Cross-seed topic stability")
    fig.tight_layout()
    fig.savefig(figures / "candidate_cross_seed_stability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    colors = {"ORIGINAL_PRIOR": "#245A7A", "FIXED_PRIOR_K20": "#B65C3A"}
    for regime in REQUIRED_REGIMES:
        frame = summary[summary["prior_regime"] == regime].set_index("K").loc[REQUIRED_K]
        axes[0].plot(REQUIRED_K, frame["mean_pairwise_topic_cosine_mean"], marker="o", color=colors[regime], label=f"{regime} mean")
        axes[0].plot(REQUIRED_K, frame["max_pairwise_topic_cosine_mean"], marker="x", linestyle="--", color=colors[regime], label=f"{regime} max")
        axes[1].plot(REQUIRED_K, frame["high_cosine_085_pairs_mean"], marker="o", color=colors[regime], label=f"{regime} ≥.85")
        axes[1].plot(REQUIRED_K, frame["high_cosine_090_pairs_mean"], marker="x", linestyle="--", color=colors[regime], label=f"{regime} ≥.90")
        axes[2].plot(REQUIRED_K, frame["mean_pairwise_top20_jaccard_mean"], marker="o", color=colors[regime], label=f"{regime} mean")
        axes[2].plot(REQUIRED_K, frame["max_pairwise_top20_jaccard_mean"], marker="x", linestyle="--", color=colors[regime], label=f"{regime} max")
    axes[0].set_title("A. Topic-word cosine")
    axes[1].set_title("B. High-cosine pair count")
    axes[2].set_title("C. Top-20 Jaccard")
    for ax in axes:
        ax.set_xlabel("K")
        ax.set_xticks(REQUIRED_K)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Within-model topic redundancy diagnostics")
    fig.tight_layout()
    fig.savefig(figures / "candidate_topic_redundancy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    comparisons = [11, 13, 14, 15, 20]
    seed_positions = np.arange(len(REQUIRED_SEEDS))
    for axis, regime in zip(axes, REQUIRED_REGIMES):
        base = all_runs[(all_runs["prior_regime"] == regime) & (all_runs["K"] == 12)].set_index("seed")
        for other in comparisons:
            current = all_runs[(all_runs["prior_regime"] == regime) & (all_runs["K"] == other)].set_index("seed")
            differences = base.loc[REQUIRED_SEEDS, "NPMI"].astype(float) - current.loc[REQUIRED_SEEDS, "NPMI"].astype(float)
            axis.plot(seed_positions, differences, marker="o", linewidth=1.2, label=f"K12−K{other}")
        axis.axhline(0, color="#333333", linewidth=1)
        axis.set_title(regime)
        axis.set_xlabel("Seed")
        axis.set_xticks(seed_positions, [str(seed) for seed in REQUIRED_SEEDS], rotation=50, ha="right")
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Seed-matched NPMI difference")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("K12 seed-matched NPMI differences")
    fig.tight_layout()
    fig.savefig(figures / "K12_seed_matched_differences.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    averaged = summary.groupby("K", sort=True).mean(numeric_only=True).loc[REQUIRED_K]
    metric_definitions = [
        ("NPMI", averaged["NPMI_mean"].to_numpy(float), True),
        ("Perplexity", averaged["test_perplexity_mean"].to_numpy(float), False),
        ("Diversity", averaged["topic_diversity_mean"].to_numpy(float), True),
        ("Cosine Stability", averaged["mean_matched_cosine"].to_numpy(float), True),
        ("Jaccard Stability", averaged["mean_matched_jaccard"].to_numpy(float), True),
        ("Redundancy", averaged["mean_pairwise_topic_cosine_mean"].to_numpy(float), False),
    ]
    normalized_columns: list[np.ndarray] = []
    for _label, values, higher_is_better in metric_definitions:
        spread = float(values.max() - values.min())
        normalized = np.ones_like(values) if spread == 0 else (values - values.min()) / spread
        normalized_columns.append(normalized if higher_is_better else 1.0 - normalized)
    display_matrix = np.column_stack(normalized_columns)
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    image = ax.imshow(display_matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metric_definitions)), [item[0] for item in metric_definitions], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(REQUIRED_K)), [f"K{value}" for value in REQUIRED_K])
    for row in range(display_matrix.shape[0]):
        for column in range(display_matrix.shape[1]):
            ax.text(column, row, f"{display_matrix[row, column]:.2f}", ha="center", va="center", color="white" if display_matrix[row, column] > 0.55 else "#222222", fontsize=9)
    fig.colorbar(image, ax=ax, label="Directional min–max display scale")
    ax.set_title("Candidate multimetric matrix (mean across prior regimes; display only)")
    fig.tight_layout()
    fig.savefig(figures / "candidate_multimetric_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    """把小型结果表转换为不依赖额外库的 Markdown。"""

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def format_k_values(values: Sequence[int]) -> str:
    """格式化 K 集合。"""

    return "、".join(f"K{int(value)}" for value in values) if values else "无"


def metric_winner(value_15: float, value_20: float, higher_is_better: bool) -> str:
    """仅按单一指标方向描述 K15 与 K20，不合成总分。"""

    if math.isclose(value_15, value_20, rel_tol=0.0, abs_tol=1e-12):
        return "TIE"
    if higher_is_better:
        return "K15" if value_15 > value_20 else "K20"
    return "K15" if value_15 < value_20 else "K20"


def generate_report(
    all_runs: pd.DataFrame,
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    pairwise: pd.DataFrame,
    prior: pd.DataFrame,
    pareto: pd.DataFrame,
    statuses: pd.DataFrame,
    shortlist: pd.DataFrame,
    plateau: Mapping[str, Any],
    context: Mapping[str, Any],
) -> None:
    """生成任务规定的十八节量化候选验证报告。"""

    output = context["output"]
    indexed = summary.set_index(["prior_regime", "K"])
    status_index = statuses.set_index("K")
    best_original = summary[summary["prior_regime"] == "ORIGINAL_PRIOR"].sort_values(["NPMI_mean", "K"], ascending=[False, True]).iloc[0]
    best_fixed = summary[summary["prior_regime"] == "FIXED_PRIOR_K20"].sort_values(["NPMI_mean", "K"], ascending=[False, True]).iloc[0]
    comparison_index = pairwise.set_index(["prior_regime", "K_A", "K_B"])
    prior_sensitive = prior.loc[prior["prior_sensitivity_status"] == "PRIOR_SENSITIVE", "K"].astype(int).tolist()
    seed_sensitive = statuses.loc[statuses["seed_sensitive"], "K"].astype(int).tolist()
    redundancy_risk = statuses.loc[statuses["redundancy_risk"], "K"].astype(int).tolist()
    pareto_lists = {
        regime: pareto.loc[(pareto["prior_regime"] == regime) & (pareto["is_pareto_nondominated"]), "K"].astype(int).tolist()
        for regime in REQUIRED_REGIMES
    }
    plateau_both = all(plateau[regime]["status"] == "BROAD_PLATEAU" for regime in REQUIRED_REGIMES)
    plateau_label = "BROAD_PLATEAU_BOTH_PRIORS" if plateau_both else "PRIOR_DEPENDENT_OR_PARTIAL_PLATEAU"

    summary_rows: list[list[Any]] = []
    for topic_count in REQUIRED_K:
        row = indexed.loc[("ORIGINAL_PRIOR", topic_count)]
        summary_rows.append(
            [
                f"K{topic_count}",
                f"{row['NPMI_mean']:.10f} ± {row['NPMI_sd']:.10f}",
                f"{row['test_perplexity_mean']:.3f}",
                f"{row['topic_diversity_mean']:.4f}",
                f"{row['mean_matched_cosine']:.4f}",
                f"{row['mean_matched_jaccard']:.4f}",
            ]
        )
    fixed_rows: list[list[Any]] = []
    for topic_count in REQUIRED_K:
        row = indexed.loc[("FIXED_PRIOR_K20", topic_count)]
        fixed_rows.append(
            [
                f"K{topic_count}",
                f"{row['NPMI_mean']:.10f} ± {row['NPMI_sd']:.10f}",
                f"{row['test_perplexity_mean']:.3f}",
                f"{row['topic_diversity_mean']:.4f}",
                f"{row['mean_matched_cosine']:.4f}",
                f"{row['mean_matched_jaccard']:.4f}",
            ]
        )
    k12_rows: list[list[Any]] = []
    for regime in REQUIRED_REGIMES:
        for other in [11, 13, 14, 15, 20]:
            row = comparison_index.loc[(regime, 12, other)]
            k12_rows.append(
                [
                    regime,
                    f"K12−K{other}",
                    f"{row['mean_npmi_difference']:+.10f}",
                    f"{int(row['seed_sign_A_gt_B'])}/10",
                    row["practical_equivalence_status"],
                ]
            )
    k15_k20_rows: list[list[Any]] = []
    for regime in REQUIRED_REGIMES:
        row15 = indexed.loc[(regime, 15)]
        row20 = indexed.loc[(regime, 20)]
        values = [
            ("NPMI ↑", row15["NPMI_mean"], row20["NPMI_mean"], True),
            ("Perplexity ↓", row15["test_perplexity_mean"], row20["test_perplexity_mean"], False),
            ("Diversity ↑", row15["topic_diversity_mean"], row20["topic_diversity_mean"], True),
            ("Cosine stability ↑", row15["mean_matched_cosine"], row20["mean_matched_cosine"], True),
            ("Jaccard stability ↑", row15["mean_matched_jaccard"], row20["mean_matched_jaccard"], True),
            ("Redundancy cosine ↓", row15["mean_pairwise_topic_cosine_mean"], row20["mean_pairwise_topic_cosine_mean"], False),
        ]
        for metric, value15, value20, higher in values:
            k15_k20_rows.append([regime, metric, f"{float(value15):.10f}", f"{float(value20):.10f}", metric_winner(float(value15), float(value20), higher)])

    lines = [
        "# QTS Quantitative K Candidate Validation Report",
        "",
        "## 1. Objective",
        "",
        "K=2–30 穷举扫描定位了 K12 全局 NPMI 峰和 K11–15 top-1% 带，但三个随机初始化不足以解释 0.0001–0.001 量级差异。本阶段只对 K11–15 与冻结 K20 reference 做十 seed、两套 prior 的量化复核；不命名主题，不改写 C_K20 taxonomy、主模型或文学史结果。",
        "",
        "## 2. Candidate Set",
        "",
        "主候选正好为 K11、K12、K13、K14、K15、K20。可选的 K21 未加入主排名，也没有重跑 K2–10、K16–19 或 K22–30。所有结果继续使用冻结 C corpus（TF>=3、DF>=3；43,015 首非空诗；35,063 词）与同一 34,412/8,603 split。",
        "",
        "## 3. Why More Seeds?",
        "",
        "每个 K×prior 使用同一组十个固定 seed：42、2026、20260830、7、17、101、31415、271828、20260907、12345。seed 不按结果删减。SD、符号一致性与跨 seed Topic matching 描述随机初始化变异，不是对历史诗歌总体的统计推断；本报告未做大规模 p-value 搜索，也未使用 bootstrap CI。",
        "",
        "## 4. Original Prior Results",
        "",
        f"`ORIGINAL_PRIOR` 保持 sklearn 的 `doc_topic_prior=None`、`topic_word_prior=None`，有效 alpha=eta=1/K。十 seed mean NPMI 最高为 K{int(best_original['K'])}：{best_original['NPMI_mean']:.10f} ± {best_original['NPMI_sd']:.10f}。",
        "",
        *markdown_table(["K", "NPMI mean ± SD", "Test PPL", "Diversity", "Cosine stability", "Jaccard stability"], summary_rows),
        "",
        "## 5. Fixed Prior Sensitivity",
        "",
        f"`FIXED_PRIOR_K20` 对所有 K 固定 alpha=eta=0.05。十 seed mean NPMI 最高为 K{int(best_fixed['K'])}：{best_fixed['NPMI_mean']:.10f} ± {best_fixed['NPMI_sd']:.10f}。这是一项 prior sensitivity，不是新正式模型，不能据此覆盖现有 K20。",
        "",
        *markdown_table(["K", "NPMI mean ± SD", "Test PPL", "Diversity", "Cosine stability", "Jaccard stability"], fixed_rows),
        "",
        "## 6. NPMI Candidate Comparison",
        "",
        f"K11–15 平台总判断为 `{plateau_label}`。ORIGINAL_PRIOR 的 1% 带成员是 {format_k_values(plateau['ORIGINAL_PRIOR']['members'])}，状态 `{plateau['ORIGINAL_PRIOR']['status']}`；FIXED_PRIOR_K20 的成员是 {format_k_values(plateau['FIXED_PRIOR_K20']['members'])}，状态 `{plateau['FIXED_PRIOR_K20']['status']}`。绝对差 0.0005、0.0010 和相对全局峰 1% 三个尺度始终并列报告，不选其中一个冒充唯一显著性阈值。",
        "",
        f"K14 在 ORIGINAL_PRIOR 的十 seed 均值为 {indexed.loc[('ORIGINAL_PRIOR', 14), 'NPMI_mean']:.10f}，相邻 K13/K15 分别为 {indexed.loc[('ORIGINAL_PRIOR', 13), 'NPMI_mean']:.10f}/{indexed.loc[('ORIGINAL_PRIOR', 15), 'NPMI_mean']:.10f}；它是否维持局部优势据这些固定均值如实判断，不外推为文学解释。",
        "",
        "## 7. Held-out Perplexity",
        "",
        "Perplexity 仅作为未见测试诗的 predictive-fit 维度。此前全扫描已显示该指标系统偏好低分辨率 K；本轮不以 perplexity 单独淘汰任何候选。train/test score 与 perplexity 均由同一 train-only 模型计算，测试集未参与拟合。",
        "",
        "## 8. Cross-seed Topic Stability",
        "",
        "每个 K×prior 的十 seed 形成 45 个 seed pair。对完整 Topic×Word 分布的 K×K cosine matrix 做 Hungarian 最大总相似度匹配；Top20 Jaccard 在同一配对上另算，二者不加权、不合成。`candidate_cross_seed_stability.csv` 同时保存均值、SD、最小值与全部 540 个真实 pair。",
        "",
        "## 9. Topic Redundancy",
        "",
        f"模型内冗余分别由完整 Topic-word cosine 和 Top20 Jaccard 描述，并统计 cosine≥0.85/0.90 与 Jaccard≥0.50 的 pair 数。相对候选异常规则标出的 `REDUNDANCY_RISK` 为：{format_k_values(redundancy_risk)}。该标签只表示定量复核优先级，不能直接称为 over-splitting。",
        "",
        "## 10. Topic Prevalence",
        "",
        "每个 train document-topic matrix 均计算各 Topic 的 mean probability、dominant-topic document count 与 dominant share；mean probability<0.01 只标为 `LOW_PREVALENCE_TOPIC`。逐 Topic 明细见 `topic_prevalence_diagnostics.csv`，模型级 min/max 与小 Topic 数见主表和汇总表；该阈值不自动判定模型失败。",
        "",
        "## 11. K12 vs K11/13/14/15/20",
        "",
        *markdown_table(["Prior", "Comparison", "Mean NPMI diff", "K12 higher seeds", "Practical equivalence"], k12_rows),
        "",
        f"K12 在 ORIGINAL_PRIOR 是否仍为十 seed mean NPMI 最高：{'是' if int(best_original['K']) == 12 else '否'}；在 FIXED_PRIOR_K20 是否领先：{'是' if int(best_fixed['K']) == 12 else '否'}。其结构稳定性与冗余风险见上表和状态 `{status_index.loc[12, 'quantitative_status']}`。",
        "",
        "## 12. K15 vs K20",
        "",
        *markdown_table(["Prior", "Metric", "K15", "K20", "Single-metric leader"], k15_k20_rows),
        "",
        "每一行只回答一个指标；没有把这些指标加权成总分。K20 的量化代价如实呈现，但其 reference 身份和冻结状态不受改变。",
        "",
        "## 13. Prior Sensitivity",
        "",
        f"`PRIOR_SENSITIVE` 候选为：{format_k_values(prior_sensitive)}。预注册规则是 NPMI rank 绝对变化至少 2，或是否进入 1% 高值带发生变化；pairwise order reversals 另存表，不因 fixed prior 结果重写原模型。",
        "",
        *markdown_table(
            ["K", "Original rank", "Fixed rank", "Rank change", "Order reversals", "Status"],
            [[f"K{int(row.K)}", int(row.original_rank), int(row.fixed_rank), int(row.rank_change), int(row.pairwise_order_reversals), row.prior_sensitivity_status] for row in prior.itertuples()],
        ),
        "",
        "## 14. Seed Sensitivity",
        "",
        f"`SEED_SENSITIVE` 候选为：{format_k_values(seed_sensitive)}。规则使用每套 prior 内六候选的 Tukey 1.5×IQR 相对异常：NPMI SD 异常高，或 mean matched cosine/Jaccard 异常低。没有根据观察结果删除 seed 或事后移动阈值。",
        "",
        "## 15. Pareto Comparison",
        "",
        f"ORIGINAL_PRIOR 非支配候选：{format_k_values(pareto_lists['ORIGINAL_PRIOR'])}；FIXED_PRIOR_K20 非支配候选：{format_k_values(pareto_lists['FIXED_PRIOR_K20'])}。维度为 NPMI↑、perplexity↓、diversity↑、两类 stability↑、redundancy↓、low-prevalence count↓；不做标准化总分。Pareto 非支配只表示在本组量化维度未被另一候选全面压倒，不等于最佳文学模型。",
        "",
        "## 16. Quantitative Candidate Tier",
        "",
        *markdown_table(
            ["K", "Status", "Both-prior platform", "Pareto any"],
            [[f"K{int(row.K)}", row.quantitative_status, row.platform_under_both_priors, row.pareto_nondominated_in_any_regime] for row in statuses.itertuples()],
        ),
        "",
        f"最终量化 shortlist 为 {format_k_values(shortlist['K'].astype(int).tolist())}。角色槽位依次保留原先验 NPMI 领跑者、固定先验 NPMI 领跑者、双先验平台内无 seed/redundancy 警告的跨 seed stability 领跑者，并强制保留 K20 reference；重复角色合并，因此不会超过四个。",
        "",
        "## 17. What Quantitative Metrics Cannot Decide",
        "",
        "数值最高不等于文学语义最清楚。NPMI 只衡量 top-20 词的训练诗文档共现；perplexity 是 sklearn 变分近似下的 predictive fit；Topic Diversity、cosine/Jaccard、prevalence 阈值均不能判断某一语义拆分是否具有人文学意义。冻结词表来自全语料预处理，held-out 不是 document-completion perplexity。所有标签都必须接受代表诗、主题边界与语义可解释性的人工复核。",
        "",
        "## 18. Recommendation for Semantic Review",
        "",
        f"下一阶段只建议对 {format_k_values(shortlist['K'].astype(int).tolist())} 开展 Humanities Semantic Review，并与现有 K15/K20 审核材料并表。K20 必须继续作为 frozen reference。本任务到此停止，不自动运行语义审核。",
        "",
    ]
    (output / "QTS_Quantitative_K_Candidate_Validation_Report.md").write_text("\n".join(lines), encoding="utf-8")
    readme = [
        "# QTS Quantitative Candidate Validation",
        "",
        "本目录保存 C corpus 的 K11–15 与 K20、十个固定 seed、两套 prior regime 共 120 个 train-only LDA fits，以及 NPMI、held-out perplexity、Topic Diversity、模型内冗余、Topic prevalence、跨 seed Hungarian 稳定性、成对差异、prior sensitivity、Pareto 辅助筛选和最终量化 shortlist。",
        "",
        "复现或续跑：",
        "",
        "```powershell",
        "python src/run_quantitative_candidate_validation.py",
        "```",
        "",
        "只重新汇总已完成检查点：",
        "",
        "```powershell",
        "python src/run_quantitative_candidate_validation.py --summarize-only",
        "```",
        "",
        "Fixed-prior 只是一项 sensitivity；所有产物均不覆盖冻结 C_K20。权威解释见 `QTS_Quantitative_K_Candidate_Validation_Report.md`。",
        "",
    ]
    (output / "README.md").write_text("\n".join(readme), encoding="utf-8")


def verify(
    context: Mapping[str, Any],
    all_runs: pd.DataFrame | None = None,
    stability: pd.DataFrame | None = None,
    complete: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """执行任务规定的十五项不变量测试。"""

    if all_runs is None:
        all_runs = pd.DataFrame(collect_checkpoints(context), columns=ALL_RUN_FIELDS)
    expected = set(itertools.product(REQUIRED_REGIMES, REQUIRED_K, REQUIRED_SEEDS))
    actual = set(zip(all_runs["prior_regime"], all_runs["K"].astype(int), all_runs["seed"].astype(int))) if len(all_runs) else set()
    current_manifest = {str(path): scan.file_sha256(path) for path in tracked_inputs(context["config"], context["config_path"])}
    immutable = current_manifest == context["input_manifest"]
    matrix_hash_ok = scan.file_sha256(scan.resolve(context["config"]["corpus_root"]) / "bow_matrix.npz") == context["config"]["expected"]["matrix_sha256"]
    split_hash_ok = scan.file_sha256(scan.resolve(context["config"]["frozen_split"])) == context["config"]["expected"]["split_sha256"]
    parameters_ok = True
    fit_scope_ok = True
    npmi_scope_ok = True
    diversity_scope_ok = True
    checkpoint_inputs_ok = True
    prior_values_ok = True
    for regime, topic_count, seed in actual:
        item = load_checkpoint(
            checkpoint_path(context["output"], regime, topic_count, seed),
            checkpoint_signature(context, regime, topic_count, seed),
        )
        audit = item["audit"]
        expected_params = lower.build_model(
            topic_count,
            seed,
            regime_lda(context["base_lda"], context["config"], regime),
        ).get_params(deep=False)
        parameters_ok &= audit["model_parameters"] == expected_params
        fit_scope_ok &= audit["fit_scope"] == "X_train_only" and audit["test_used_for_fit"] is False
        npmi_scope_ok &= audit["npmi_reference"] == "X_train_only" and int(audit["npmi_top_words"]) == 20
        diversity_scope_ok &= audit["topic_diversity_definition"] == "top_20_unique_ratio"
        checkpoint_inputs_ok &= (
            item["matrix_sha256"] == context["config"]["expected"]["matrix_sha256"]
            and item["vocabulary_sha256"] == context["vocabulary_sha256"]
            and item["split_sha256"] == context["split_audit"]["split_sha256"]
            and item["fit_rows_sha256"] == context["split_audit"]["fit_rows_sha256"]
            and item["test_rows_sha256"] == context["split_audit"]["test_rows_sha256"]
        )
        result = item["result"]
        expected_prior = 1.0 / topic_count if regime == "ORIGINAL_PRIOR" else 0.05
        prior_values_ok &= math.isclose(float(result["doc_topic_prior_effective"]), expected_prior, rel_tol=0.0, abs_tol=1e-15)
        prior_values_ok &= math.isclose(float(result["topic_word_prior_effective"]), expected_prior, rel_tol=0.0, abs_tol=1e-15)
    complete_runs = len(all_runs) == 120 and actual == expected
    group_counts_ok = bool(
        len(all_runs)
        and all_runs.groupby(["prior_regime", "K"])["seed"].nunique().eq(10).all()
        and all_runs.groupby(["prior_regime", "seed"])["K"].nunique().eq(6).all()
    )
    regimes_ok = set(all_runs["prior_regime"]) == set(REQUIRED_REGIMES) if len(all_runs) else False
    candidates_ok = set(all_runs["K"].astype(int)) == set(REQUIRED_K) if len(all_runs) else False
    finite_ppl = bool(
        len(all_runs)
        and np.isfinite(all_runs[["train_perplexity", "test_perplexity"]].astype(float)).all().all()
        and (all_runs[["train_perplexity", "test_perplexity"]].astype(float) > 0).all().all()
    )
    finite_npmi = bool(len(all_runs) and np.isfinite(all_runs["NPMI"].astype(float)).all())
    identity = scan.read_json(context["output"] / "audit/run_identity.json")
    current_npmi_hash = hashlib.sha256(inspect.getsource(original.topic_npmi_scores).encode("utf-8")).hexdigest()
    current_diversity_hash = hashlib.sha256(inspect.getsource(original.topic_diversity).encode("utf-8")).hexdigest()
    current_matching_hash = hashlib.sha256(inspect.getsource(match_seed_pair).encode("utf-8")).hexdigest()
    if stability is None and (context["output"] / "data/candidate_cross_seed_stability.csv").is_file():
        stability = pd.read_csv(context["output"] / "data/candidate_cross_seed_stability.csv", encoding="utf-8-sig")
    hungarian_ok = False
    if stability is not None and len(stability):
        group_pairs = stability.groupby(["prior_regime", "K"]).size()
        hungarian_ok = bool(
            len(stability) == 540
            and len(group_pairs) == 12
            and group_pairs.eq(45).all()
            and set(stability["matching_method"]) == {"HUNGARIAN_MAX_FULL_TOPIC_WORD_COSINE"}
            and np.isfinite(stability[["matched_cosine", "matched_jaccard"]].astype(float)).all().all()
        )
    elif not complete:
        hungarian_ok = True
    checks = [
        ("TEST_01", "C corpus hash 不变", immutable and matrix_hash_ok and checkpoint_inputs_ok),
        ("TEST_02", "原 train/test split 不变", immutable and split_hash_ok and checkpoint_inputs_ok),
        ("TEST_03", "原 frozen C_K20 不变", immutable),
        ("TEST_04", "主候选正好为 11、12、13、14、15、20", candidates_ok and list(context["config"]["primary_candidates"]) == REQUIRED_K),
        ("TEST_05", "每候选每 prior 正好 10 seeds", group_counts_ok),
        ("TEST_06", "两个 prior regimes 完整且其余参数固定", regimes_ok and parameters_ok),
        ("TEST_07", "主运行总计 120 fits", complete_runs),
        ("TEST_08", "所有 perplexity finite positive", finite_ppl),
        ("TEST_09", "所有 NPMI finite", finite_npmi),
        ("TEST_10", "NPMI 实现与 train-only reference 不变", npmi_scope_ok and current_npmi_hash == identity["source_hashes"]["topic_npmi_scores"]),
        ("TEST_11", "Topic Diversity 定义不变", diversity_scope_ok and current_diversity_hash == identity["source_hashes"]["topic_diversity"]),
        ("TEST_12", "Hungarian matching 实现固定且 540 pairs 完整", hungarian_ok and current_matching_hash == identity["source_hashes"]["hungarian_match"]),
        ("TEST_13", "不根据结果删除 seed", list(context["config"]["seeds"]) == REQUIRED_SEEDS and actual == expected),
        ("TEST_14", "fixed prior 为 0.05/0.05，original prior 为 1/K", prior_values_ok),
        ("TEST_15", "不自动修改现有 K20 状态", immutable and fit_scope_ok and context["output"].resolve() == (ROOT / "model_selection/quantitative_candidate_validation").resolve()),
    ]
    payload = {
        "complete": complete_runs,
        "n_runs": len(all_runs),
        "passed": sum(int(passed) for _, _, passed in checks),
        "total": len(checks),
        "checks": [
            {"test": test, "description": description, "passed": bool(passed)}
            for test, description, passed in checks
        ],
    }
    scan.atomic_json(context["output"] / "audit/integrity_checks.json", payload)
    if complete and not all(passed for _, _, passed in checks):
        raise ValueError(f"不变量检查失败：{payload}")
    return all_runs.sort_values(["prior_regime", "K", "seed"]).reset_index(drop=True), payload["checks"]


def finalize(context: Mapping[str, Any]) -> dict[str, Any]:
    """生成全部表格、图、报告、清单并执行最终核验。"""

    all_runs = pd.DataFrame(collect_checkpoints(context), columns=ALL_RUN_FIELDS)
    if len(all_runs) != 120:
        raise ValueError("只有 120 个主拟合全部完成后才能汇总")
    stability = build_cross_seed_stability(context)
    summary = build_candidate_summary(all_runs, stability, context["config"])
    pairwise = build_pairwise_comparison(all_runs, summary, context["config"])
    prior = build_prior_sensitivity(summary, context["config"])
    pareto = build_pareto_front(summary)
    statuses = classify_candidate_statuses(summary, prior, pareto, context["config"])
    shortlist = build_shortlist(summary, statuses, context["config"])
    plateau = classify_plateau(summary)
    prevalence = build_topic_prevalence_details(context)
    output = context["output"]
    summary_output = summary.merge(statuses[["K", "quantitative_status"]], on="K", how="left")
    scan.atomic_frame(output / "data/quantitative_candidate_all_runs.csv", all_runs[ALL_RUN_FIELDS].sort_values(["prior_regime", "K", "seed"]))
    scan.atomic_frame(output / "data/quantitative_candidate_summary.csv", summary_output)
    scan.atomic_frame(output / "data/candidate_cross_seed_stability.csv", stability)
    scan.atomic_frame(output / "data/candidate_pairwise_comparison.csv", pairwise)
    scan.atomic_frame(output / "data/candidate_prior_sensitivity.csv", prior)
    scan.atomic_frame(output / "data/candidate_pareto_front.csv", pareto)
    scan.atomic_frame(output / "data/final_quantitative_candidate_shortlist.csv", shortlist)
    scan.atomic_frame(output / "data/topic_prevalence_diagnostics.csv", prevalence)
    generate_figures(all_runs, summary, output)
    generate_report(all_runs, summary, stability, pairwise, prior, pareto, statuses, shortlist, plateau, context)
    all_runs, checks = verify(context, all_runs, stability, complete=True)
    required_outputs = [
        output / "README.md",
        output / "config/candidate_validation_config.yaml",
        output / "data/quantitative_candidate_all_runs.csv",
        output / "data/quantitative_candidate_summary.csv",
        output / "data/candidate_cross_seed_stability.csv",
        output / "data/candidate_pairwise_comparison.csv",
        output / "data/candidate_prior_sensitivity.csv",
        output / "data/candidate_pareto_front.csv",
        output / "data/final_quantitative_candidate_shortlist.csv",
        output / "data/topic_prevalence_diagnostics.csv",
        output / "figures/candidate_NPMI_original_vs_fixed_prior.png",
        output / "figures/candidate_perplexity_original_vs_fixed_prior.png",
        output / "figures/candidate_cross_seed_stability.png",
        output / "figures/candidate_topic_redundancy.png",
        output / "figures/K12_seed_matched_differences.png",
        output / "figures/candidate_multimetric_matrix.png",
        output / "QTS_Quantitative_K_Candidate_Validation_Report.md",
        output / "audit/integrity_checks.json",
    ]
    missing = [str(path) for path in required_outputs if not path.is_file()]
    if missing:
        raise ValueError(f"缺少要求产物：{missing}")
    scan.atomic_json(
        output / "audit/output_manifest.json",
        {str(path.relative_to(ROOT)).replace("\\", "/"): scan.file_sha256(path) for path in required_outputs},
    )
    return {
        "all_runs": all_runs,
        "summary": summary,
        "stability": stability,
        "pairwise": pairwise,
        "prior": prior,
        "pareto": pareto,
        "statuses": statuses,
        "shortlist": shortlist,
        "plateau": plateau,
        "checks": checks,
    }


def main() -> None:
    """命令行入口：顺序拟合、原子检查点、可续跑与独立汇总。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--max-new-runs", type=int)
    args = parser.parse_args()
    context = prepare(args.config)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "primary_candidates": REQUIRED_K,
                    "seeds": REQUIRED_SEEDS,
                    "prior_regimes": REQUIRED_REGIMES,
                    "n_train": context["x_train"].shape[0],
                    "n_test": context["x_test"].shape[0],
                    "vocab_size": context["x_train"].shape[1],
                    "expected_fits": 120,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.verify_only:
        frame, checks = verify(context, complete=True)
        print(f"核验通过：{len(frame)}/120 runs，{sum(item['passed'] for item in checks)}/15 invariants")
        return
    if args.summarize_only:
        artifacts = finalize(context)
        print(f"汇总完成：shortlist={format_k_values(artifacts['shortlist']['K'].astype(int).tolist())}")
        return

    lock = context["output"] / "audit/runner.lock"
    if lock.exists():
        raise RuntimeError(f"已有运行锁：{lock}")
    lock.write_text(str(os.getpid()), encoding="utf-8")
    new_runs = 0
    try:
        existing_rows = collect_checkpoints(context)
        completed = {
            (row["prior_regime"], int(row["K"]), int(row["seed"]))
            for row in existing_rows
        }
        rows = list(existing_rows)
        for regime, topic_count, seed in itertools.product(REQUIRED_REGIMES, REQUIRED_K, REQUIRED_SEEDS):
            key = (regime, topic_count, seed)
            if key in completed:
                continue
            if args.max_new_runs is not None and new_runs >= args.max_new_runs:
                break
            scan.atomic_json(
                context["output"] / "audit/progress.json",
                {
                    "state": "RUNNING",
                    "pid": os.getpid(),
                    "completed": len(rows),
                    "current": {"prior_regime": regime, "K": topic_count, "seed": seed},
                },
            )
            print(f"开始 {regime} K={topic_count} seed={seed}（已完成 {len(rows)}/120）", flush=True)
            result, audit, arrays = evaluate_one(context, regime, topic_count, seed)
            matrix_path = topic_matrix_path(context["output"], regime, topic_count, seed)
            atomic_npz(matrix_path, **arrays)
            item = {
                "checkpoint_signature": checkpoint_signature(context, regime, topic_count, seed),
                "result": result,
                "result_sha256": scan.stable_hash(result),
                "audit": audit,
                "topic_matrix_path": str(matrix_path.relative_to(ROOT)).replace("\\", "/"),
                "topic_matrix_sha256": scan.file_sha256(matrix_path),
                "matrix_sha256": context["config"]["expected"]["matrix_sha256"],
                "vocabulary_sha256": context["vocabulary_sha256"],
                "split_sha256": context["split_audit"]["split_sha256"],
                "fit_rows_sha256": context["split_audit"]["fit_rows_sha256"],
                "test_rows_sha256": context["split_audit"]["test_rows_sha256"],
            }
            scan.atomic_json(checkpoint_path(context["output"], regime, topic_count, seed), item)
            rows.append(result)
            completed.add(key)
            new_runs += 1
            write_all_runs(context["output"], rows)
            print(
                f"完成 {len(rows)}/120 {regime} K={topic_count} seed={seed} "
                f"NPMI={result['NPMI']:.8f} testPPL={result['test_perplexity']:.4f} fit={result['fit_time']:.1f}s",
                flush=True,
            )
            get_reusable_executor().shutdown(wait=True, kill_workers=True)
            gc.collect()
        if len(rows) == 120:
            artifacts = finalize(context)
            scan.atomic_json(
                context["output"] / "audit/progress.json",
                {
                    "state": "COMPLETE",
                    "completed": 120,
                    "shortlist": artifacts["shortlist"]["K"].astype(int).tolist(),
                },
            )
            print("120/120 runs 与全部量化候选验证产物完成。", flush=True)
        else:
            verify(context, pd.DataFrame(rows, columns=ALL_RUN_FIELDS), complete=False)
            scan.atomic_json(context["output"] / "audit/progress.json", {"state": "PARTIAL", "completed": len(rows)})
            print(f"本次停止于 {len(rows)}/120 runs。", flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
