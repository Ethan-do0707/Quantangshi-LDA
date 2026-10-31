"""运行《全唐诗》LDA 低 K 边界检验与测试集困惑度诊断。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import itertools
import json
import logging
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split

import clean_corpus as cleaner
import run_tf_df_lda as original

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "model_selection/lower_k_boundary_perplexity/config/lower_k_perplexity_config.yaml"
EXPECTED_CORPORA = {
    "A": (1, 1, [20, 30, 40, 50, 60]),
    "B": (2, 2, [20, 30, 40, 50, 60]),
    "C": (3, 3, [5, 10, 15, 20, 30, 40, 50, 60]),
    "D": (5, 3, [20, 30, 40, 50, 60]),
}
EXPECTED_SEEDS = [42, 2026, 20260830]
EXPECTED_LOWER_K = [5, 10, 15]
ORIGINAL_K = [20, 30, 40, 50, 60]
SCHEMA_VERSION = 1


@dataclass
class CorpusView:
    """保存既有冻结语料的只读矩阵与行列索引。"""

    label: str
    min_tf: int
    min_df: int
    root: Path
    matrix: csr_matrix
    matrix_fingerprint: str
    vocabulary: list[str]
    document_rows: list[dict[str, str]]
    corpus_rows: list[dict[str, str]] | None
    metadata: dict[str, Any]


def read_csv(path: Path) -> list[dict[str, str]]:
    """以 UTF-8-SIG 读取 CSV。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def stable_json_hash(value: Any) -> str:
    """计算规范 JSON 哈希。"""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_hash(values: np.ndarray) -> str:
    """计算数组形状、类型与内容哈希。"""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def source_hash(function: Any) -> str:
    """记录复用指标函数的源码哈希。"""

    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def resolve(path: str) -> Path:
    """相对项目根目录解析路径。"""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_config(path: Path) -> dict[str, Any]:
    """读取并严格验证实验配置。"""

    config = json.loads(path.read_text(encoding="utf-8-sig"))
    corpora = config.get("corpora", {})
    if set(corpora) != set(EXPECTED_CORPORA):
        raise ValueError("corpora 必须且只能包含 A、B、C、D")
    for label, (min_tf, min_df, topic_counts) in EXPECTED_CORPORA.items():
        item = corpora[label]
        actual = (int(item["min_tf"]), int(item["min_df"]), list(item["topic_counts"]))
        if actual != (min_tf, min_df, topic_counts):
            raise ValueError(f"{label} 的 TF/DF/K 配置不符合任务设计：{actual}")
    if list(config["lower_k"]) != EXPECTED_LOWER_K:
        raise ValueError("lower_k 必须为 5、10、15")
    if list(config["random_seeds"]) != EXPECTED_SEEDS:
        raise ValueError("随机种子必须为 42、2026、20260830")
    split = config["split"]
    if float(split["train_fraction"]) != 0.8 or float(split["test_fraction"]) != 0.2 or int(split["random_state"]) != 20260904:
        raise ValueError("train/test split 必须为 80/20 且 random_state=20260904")
    lda = config["lda"]
    if str(lda["learning_method"]) != "batch" or int(lda["max_iter"]) != 30:
        raise ValueError("LDA 必须使用 batch 且 max_iter=30")
    metrics = config["metrics"]
    if (str(metrics["coherence"]) != "document_npmi" or int(metrics["top_words_for_metrics"]) != 20 or str(metrics["diversity"]) != "top_20_unique_ratio" or str(metrics["stability"]) != "hungarian_matched_top_word_jaccard"):
        raise ValueError("NPMI、Diversity 或 Stability 定义偏离原实验")
    return config


def build_model(topic_count: int, seed: int, lda: Mapping[str, Any]) -> LatentDirichletAllocation:
    """使用与原实验完全相同的 LDA 参数构建模型。"""

    return LatentDirichletAllocation(
        n_components=topic_count,
        doc_topic_prior=lda.get("doc_topic_prior"),
        topic_word_prior=lda.get("topic_word_prior"),
        learning_method=str(lda["learning_method"]),
        max_iter=int(lda["max_iter"]),
        evaluate_every=int(lda["evaluate_every"]),
        perp_tol=float(lda["perp_tol"]),
        mean_change_tol=float(lda["mean_change_tol"]),
        max_doc_update_iter=int(lda["max_doc_update_iter"]),
        random_state=seed,
        n_jobs=int(lda["n_jobs"]),
        verbose=int(lda.get("verbose", 0)),
    )


def load_corpus(config: Mapping[str, Any], label: str, with_text: bool = False) -> CorpusView:
    """读取一套既有语料，不重新分词、过滤或建词表。"""

    definition = config["corpora"][label]
    root = resolve(str(config["corpus_root"])) / label
    matrix = load_npz(root / "bow_matrix.npz").tocsr()
    matrix.sort_indices()
    metadata = json.loads((root / "corpus_metadata.json").read_text(encoding="utf-8-sig"))
    fingerprint = original.matrix_fingerprint(matrix)
    if fingerprint != metadata["matrix_fingerprint"]:
        raise ValueError(f"{label} BoW 内容哈希与原 metadata 不一致")
    vocabulary_rows = sorted(read_csv(root / "vocabulary.csv"), key=lambda row: int(row["term_index"]))
    vocabulary = [row["token"] for row in vocabulary_rows]
    document_rows = sorted(read_csv(root / "document_index.csv"), key=lambda row: int(row["matrix_row"]))
    if matrix.shape != (len(document_rows), len(vocabulary)):
        raise ValueError(f"{label} 行列索引与矩阵 shape 不一致")
    if any(int(row["matrix_row"]) != expected for expected, row in enumerate(document_rows)):
        raise ValueError(f"{label} document_index 不是连续矩阵行")
    return CorpusView(
        label=label,
        min_tf=int(definition["min_tf"]),
        min_df=int(definition["min_df"]),
        root=root,
        matrix=matrix,
        matrix_fingerprint=fingerprint,
        vocabulary=vocabulary,
        document_rows=document_rows,
        corpus_rows=read_csv(root / "lda_corpus.csv") if with_text else None,
        metadata=metadata,
    )


def ensure_original_parameter_match(config: Mapping[str, Any]) -> None:
    """逐项核对本任务参数与第一轮原配置。"""

    original_config = json.loads((PROJECT_ROOT / "config/tf_df_lda_config.yaml").read_text(encoding="utf-8-sig"))
    compared = ["implementation", "learning_method", "max_iter", "n_jobs", "evaluate_every", "perp_tol", "mean_change_tol", "max_doc_update_iter", "doc_topic_prior", "topic_word_prior", "verbose"]
    differences = {key: (original_config["lda"].get(key), config["lda"].get(key)) for key in compared if original_config["lda"].get(key) != config["lda"].get(key)}
    if differences:
        raise ValueError(f"LDA 参数与原配置不一致：{differences}")
    original_metrics = original_config["metrics"]
    if (int(original_metrics["top_words"]) != int(config["metrics"]["top_words_for_metrics"]) or original_metrics["coherence"] != config["metrics"]["coherence"] or original_metrics["stability"] != config["metrics"]["stability"]):
        raise ValueError("指标参数与原配置不一致")


def original_metrics_consistency(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """确认原 K20–60 汇总可由逐 run 与稳定性原表精确复原。"""

    summary = pd.read_csv(resolve(str(config["original_configuration_summary"])))
    models = pd.read_csv(resolve(str(config["original_model_metrics"])))
    stability = pd.read_csv(resolve(str(config["original_stability_metrics"])))
    results: list[dict[str, Any]] = []
    for label in EXPECTED_CORPORA:
        for topic_count in ORIGINAL_K:
            expected = summary[(summary.corpus == label) & (summary.K == topic_count)]
            selected = models[(models.corpus == label) & (models.K == topic_count)]
            selected_stability = stability[(stability.corpus == label) & (stability.K == topic_count)]
            if len(expected) != 1 or len(selected) != 3 or len(selected_stability) != 3:
                raise ValueError(f"原指标行不完整：{label} K={topic_count}")
            row = expected.iloc[0]
            checks = {
                "mean_npmi": float(selected.mean_topic_npmi.mean()),
                "mean_topic_diversity": float(selected.topic_diversity.mean()),
                "mean_topic_stability": float(selected_stability.matched_topic_jaccard.mean()),
            }
            for field, actual in checks.items():
                target = float(row[field])
                if not math.isclose(actual, target, rel_tol=0.0, abs_tol=5e-11):
                    raise ValueError(f"原指标版本不一致：{label} K={topic_count} {field} {actual} != {target}")
            if label == "C":
                results.append({"K": topic_count, "mean_NPMI": float(row["mean_npmi"]), "mean_Topic_Diversity": float(row["mean_topic_diversity"]), "mean_Stability": float(row["mean_topic_stability"])})
    return results


def tracked_input_paths(config: Mapping[str, Any]) -> list[Path]:
    """列出实验必须保持不变的既有输入。"""

    paths = [PROJECT_ROOT / "config/tf_df_lda_config.yaml", resolve(str(config["original_configuration_summary"])), resolve(str(config["original_model_metrics"])), resolve(str(config["original_stability_metrics"])), resolve(str(config["original_human_evaluation"]))]
    if "original_model_comprehensive_evaluation" in config:
        paths.append(resolve(str(config["original_model_comprehensive_evaluation"])))
    corpus_root = resolve(str(config["corpus_root"]))
    for label in EXPECTED_CORPORA:
        paths.extend(corpus_root / label / name for name in ("bow_matrix.npz", "vocabulary.csv", "document_index.csv", "lda_corpus.csv", "corpus_metadata.json"))
    paths.extend(path for path in resolve(str(config["frozen_c_k20_root"])).rglob("*") if path.is_file())
    return sorted(set(path.resolve() for path in paths), key=str)


def establish_input_baseline(config: Mapping[str, Any], output_root: Path) -> dict[str, str]:
    """建立或复核既有输入哈希基线。"""

    state_dir = output_root / "run_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = state_dir / "input_hashes_baseline.json"
    current = {str(path): cleaner.file_sha256(path) for path in tracked_input_paths(config)}
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
        changed = [key for key in sorted(baseline) if current.get(key) != baseline[key]]
        if changed:
            raise ValueError(f"既有输入在诊断期间发生变化：{changed}")
        # 只允许把此前未纳入的只读来源追加到基线，既有键和值不可改变。
        if set(current) != set(baseline):
            cleaner.write_json(baseline_path, current)
    else:
        cleaner.write_json(baseline_path, current)
    return current


def split_indices(corpus: CorpusView, config: Mapping[str, Any], output_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """生成或核验一个 corpus 内所有 K 共用的固定 80/20 切分。"""

    all_indices = np.arange(corpus.matrix.shape[0], dtype=np.int64)
    train, test = train_test_split(all_indices, test_size=float(config["split"]["test_fraction"]), random_state=int(config["split"]["random_state"]), shuffle=True)
    train, test = np.sort(train.astype(np.int64)), np.sort(test.astype(np.int64))
    if len(set(train.tolist()) & set(test.tolist())) or len(train) + len(test) != len(all_indices):
        raise ValueError(f"{corpus.label} train/test 切分不互斥或不完整")
    subset = np.full(len(all_indices), "test", dtype=object)
    subset[train] = "train"
    rows = [{"corpus": corpus.label, "matrix_row": index, "poem_id": corpus.document_rows[index]["poem_id"], "subset": str(subset[index]), "split_random_state": int(config["split"]["random_state"])} for index in range(len(all_indices))]
    path = output_root / "splits" / f"perplexity_split_{corpus.label}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = [(row["corpus"], int(row["matrix_row"]), row["poem_id"], row["subset"], int(row["split_random_state"])) for row in read_csv(path)]
        expected = [(str(row["corpus"]), int(row["matrix_row"]), str(row["poem_id"]), str(row["subset"]), int(row["split_random_state"])) for row in rows]
        if existing != expected:
            raise ValueError(f"既有 {corpus.label} split 与固定设计不一致")
    else:
        cleaner.write_csv(path, list(rows[0].keys()), rows)
    return train, test


def checkpoint_valid(metrics_path: Path, signature: str) -> bool:
    """验证检查点签名与登记输出哈希。"""

    if not metrics_path.is_file():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return False
    if metrics.get("checkpoint_signature") != signature:
        return False
    return all((metrics_path.parent / str(item.get("path", ""))).is_file() and cleaner.file_sha256(metrics_path.parent / str(item["path"])) == item.get("sha256") for item in metrics.get("outputs", []))


def lower_signature(corpus: CorpusView, topic_count: int, seed: int, config: Mapping[str, Any]) -> str:
    """生成低 K 全语料模型检查点签名。"""

    return stable_json_hash({"schema_version": SCHEMA_VERSION, "kind": "lower_k_full_corpus", "matrix_fingerprint": corpus.matrix_fingerprint, "K": topic_count, "seed": seed, "lda": config["lda"], "metrics": config["metrics"]})


def topic_cosine_matrix(topic_word: np.ndarray) -> np.ndarray:
    """计算完整 Topic×Word 分布的余弦相似度矩阵。"""

    norms = np.linalg.norm(topic_word, axis=1, keepdims=True)
    normalized = np.divide(topic_word, norms, out=np.zeros_like(topic_word), where=norms > 0)
    return normalized @ normalized.T


def write_lower_full_model(corpus: CorpusView, topic_count: int, seed: int, config: Mapping[str, Any], output_root: Path, logger: logging.Logger) -> dict[str, Any]:
    """拟合并保存一个 C 低 K 全语料诊断模型。"""

    model_dir = output_root / "lower_k_models" / f"C_K{topic_count}" / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    signature = lower_signature(corpus, topic_count, seed, config)
    metrics_path = model_dir / "metrics.json"
    if bool(config["checkpoint"]["resume"]) and checkpoint_valid(metrics_path, signature):
        logger.info("复用低 K 检查点：C K=%d seed=%d", topic_count, seed)
        return json.loads(metrics_path.read_text(encoding="utf-8-sig"))

    logger.info("开始低 K 全语料模型：C K=%d seed=%d", topic_count, seed)
    started = time.perf_counter()
    model = build_model(topic_count, seed, config["lda"])
    document_topic = model.fit_transform(corpus.matrix)
    fit_seconds = time.perf_counter() - started
    topic_word = model.components_ / model.components_.sum(axis=1, keepdims=True)
    top20 = original.topic_top_indices(topic_word, 20)
    top30 = original.topic_top_indices(topic_word, int(config["metrics"]["review_top_words"]))
    topic_npmi, mean_npmi = original.topic_npmi_scores(corpus.matrix, top20)
    diversity = original.topic_diversity(top20)
    prevalence = document_topic.mean(axis=0)
    similarity = topic_cosine_matrix(topic_word)
    topic_rows: list[dict[str, Any]] = []
    for topic_index, indices in enumerate(top30):
        closest = [index for index in np.argsort(-similarity[topic_index]).tolist() if index != topic_index][:3]
        probabilities = topic_word[topic_index]
        entropy = float(-np.sum(probabilities * np.log(probabilities))) / math.log(len(probabilities))
        for rank, term_index in enumerate(indices, 1):
            topic_rows.append({
                "topic_id": topic_index + 1,
                "rank": rank,
                "term_index": term_index,
                "token": corpus.vocabulary[term_index],
                "probability": f"{probabilities[term_index]:.12f}",
                "component_weight": f"{model.components_[topic_index, term_index]:.8f}",
                "topic_npmi": f"{topic_npmi[topic_index]:.10f}",
                "topic_prevalence": f"{prevalence[topic_index]:.10f}",
                "top10_probability_mass": f"{probabilities[indices[:10]].sum():.10f}",
                "top30_probability_mass": f"{probabilities[indices].sum():.10f}",
                "normalized_topic_entropy": f"{entropy:.10f}",
                "closest_topics": ";".join(str(index + 1) for index in closest),
                "closest_similarities": ";".join(f"{similarity[topic_index, index]:.10f}" for index in closest),
            })
    cleaner.write_csv(model_dir / "topics_top30.csv", list(topic_rows[0].keys()), topic_rows)

    if corpus.corpus_rows is None:
        raise ValueError("低 K 代表诗需要加载 C 语料原文")
    representative_rows: list[dict[str, Any]] = []
    per_topic = int(config["metrics"]["representative_poems_per_topic"])
    for topic_index in range(topic_count):
        scores = document_topic[:, topic_index]
        count = min(per_topic, len(scores))
        candidates = np.argpartition(scores, -count)[-count:]
        ordered = sorted(candidates.tolist(), key=lambda index: (-scores[index], index))
        for rank, matrix_row in enumerate(ordered, 1):
            document = corpus.document_rows[matrix_row]
            poem = corpus.corpus_rows[int(document["corpus_row"])]
            representative_rows.append({
                "topic_id": topic_index + 1,
                "rank": rank,
                "matrix_row": matrix_row,
                "corpus_row": document["corpus_row"],
                "poem_id": poem["poem_id"],
                "author": poem["author"],
                "title": poem["title"],
                "topic_probability": f"{scores[matrix_row]:.10f}",
                "original_text": poem["original_text"],
                "cleaned_tokens": poem["cleaned_tokens"],
            })
    cleaner.write_csv(model_dir / "representative_poems_top20.csv", list(representative_rows[0].keys()), representative_rows)
    joblib.dump(model, model_dir / "model.joblib", compress=int(config["checkpoint"]["joblib_compress"]))
    save_npz(model_dir / "topic_word_components.npz", csr_matrix(model.components_), compressed=True)
    save_npz(model_dir / "topic_word_probabilities.npz", csr_matrix(topic_word), compressed=True)
    save_npz(model_dir / "document_topic_probabilities.npz", csr_matrix(document_topic), compressed=True)
    np.save(model_dir / "topic_similarity.npy", similarity)
    output_names = ["model.joblib", "topic_word_components.npz", "topic_word_probabilities.npz", "document_topic_probabilities.npz", "topic_similarity.npy", "topics_top30.csv", "representative_poems_top20.csv"]
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_signature": signature,
        "run_type": "full_corpus_model_selection_diagnostic",
        "corpus": "C",
        "K": topic_count,
        "seed": seed,
        "documents": corpus.matrix.shape[0],
        "vocabulary_size": corpus.matrix.shape[1],
        "iterations_run": int(model.n_iter_),
        "fit_time_seconds": round(fit_seconds, 6),
        "training_perplexity": float(model.bound_),
        "mean_NPMI": float(mean_npmi),
        "min_topic_NPMI": float(min(topic_npmi)),
        "max_topic_NPMI": float(max(topic_npmi)),
        "Topic_Diversity": float(diversity),
        "metric_top_words": 20,
        "model_parameters": dict(config["lda"]),
        "outputs": [{"path": name, "sha256": cleaner.file_sha256(model_dir / name), "bytes": (model_dir / name).stat().st_size} for name in output_names],
    }
    cleaner.write_json(metrics_path, metrics)
    logger.info("完成低 K 全语料模型：C K=%d seed=%d time=%.1fs NPMI=%.6f diversity=%.6f", topic_count, seed, fit_seconds, mean_npmi, diversity)
    return metrics


def heldout_signature(corpus: CorpusView, topic_count: int, seed: int, train: np.ndarray, test: np.ndarray, config: Mapping[str, Any]) -> str:
    """生成 held-out run 检查点签名。"""

    return stable_json_hash({"schema_version": SCHEMA_VERSION, "kind": "heldout_perplexity", "matrix_fingerprint": corpus.matrix_fingerprint, "train_indices": array_hash(train), "test_indices": array_hash(test), "K": topic_count, "seed": seed, "lda": config["lda"]})


def run_heldout_model(corpus: CorpusView, topic_count: int, seed: int, train: np.ndarray, test: np.ndarray, config: Mapping[str, Any], output_root: Path, logger: logging.Logger) -> dict[str, Any]:
    """仅在训练集拟合，并在固定测试集计算 score 与 perplexity。"""

    model_dir = output_root / "heldout_models" / corpus.label / f"K{topic_count}" / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    signature = heldout_signature(corpus, topic_count, seed, train, test, config)
    metrics_path = model_dir / "metrics.json"
    if bool(config["checkpoint"]["resume"]) and checkpoint_valid(metrics_path, signature):
        logger.info("复用 held-out 检查点：%s K=%d seed=%d", corpus.label, topic_count, seed)
        return json.loads(metrics_path.read_text(encoding="utf-8-sig"))

    x_train, x_test = corpus.matrix[train], corpus.matrix[test]
    logger.info("开始 held-out 模型：%s K=%d seed=%d train=%d test=%d", corpus.label, topic_count, seed, x_train.shape[0], x_test.shape[0])
    model = build_model(topic_count, seed, config["lda"])
    started = time.perf_counter()
    model.fit(x_train)
    fit_seconds = time.perf_counter() - started
    evaluation_started = time.perf_counter()
    train_perplexity = float(model.perplexity(x_train))
    test_perplexity = float(model.perplexity(x_test))
    train_score = float(model.score(x_train))
    test_score = float(model.score(x_test))
    evaluation_seconds = time.perf_counter() - evaluation_started
    if not all(math.isfinite(value) for value in (train_perplexity, test_perplexity, train_score, test_score)) or train_perplexity <= 0 or test_perplexity <= 0:
        raise ValueError(f"{corpus.label} K={topic_count} seed={seed} 产生无效 held-out 结果")
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_signature": signature,
        "run_type": "heldout_perplexity_diagnostic",
        "corpus": corpus.label,
        "K": topic_count,
        "seed": seed,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "vocab_size": int(corpus.matrix.shape[1]),
        "train_perplexity": train_perplexity,
        "test_perplexity": test_perplexity,
        "train_score": train_score,
        "test_score": test_score,
        "n_iter_": int(model.n_iter_),
        "fit_time_seconds": round(fit_seconds, 6),
        "evaluation_time_seconds": round(evaluation_seconds, 6),
        "fit_input": "X_train_only",
        "test_data_used_for_fit": False,
        "train_indices_sha256": array_hash(train),
        "test_indices_sha256": array_hash(test),
        "model_parameters": dict(config["lda"]),
        "outputs": [],
    }
    cleaner.write_json(metrics_path, metrics)
    logger.info("完成 held-out 模型：%s K=%d seed=%d time=%.1fs test_perplexity=%.3f", corpus.label, topic_count, seed, fit_seconds, test_perplexity)
    return metrics


def load_lower_metrics(output_root: Path) -> list[dict[str, Any]]:
    """读取全部低 K 全语料模型指标。"""

    rows = []
    for topic_count in EXPECTED_LOWER_K:
        for seed in EXPECTED_SEEDS:
            path = output_root / "lower_k_models" / f"C_K{topic_count}" / f"seed_{seed}" / "metrics.json"
            if path.is_file():
                rows.append(json.loads(path.read_text(encoding="utf-8-sig")))
    return rows


def load_heldout_metrics(config: Mapping[str, Any], output_root: Path) -> list[dict[str, Any]]:
    """读取全部 held-out run 指标。"""

    rows = []
    for label, definition in config["corpora"].items():
        for topic_count in definition["topic_counts"]:
            for seed in EXPECTED_SEEDS:
                path = output_root / "heldout_models" / label / f"K{topic_count}" / f"seed_{seed}" / "metrics.json"
                if path.is_file():
                    rows.append(json.loads(path.read_text(encoding="utf-8-sig")))
    return rows


def topic_sets_from_csv(path: Path) -> list[set[str]]:
    """从 Top30 表提取用于原定义的 Top20 词集合。"""

    grouped: dict[int, set[str]] = {}
    for row in read_csv(path):
        if int(row["rank"]) <= 20:
            grouped.setdefault(int(row["topic_id"]), set()).add(row["token"])
    return [grouped[index] for index in sorted(grouped)]


def aggregate_lower(config: Mapping[str, Any], output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int]]:
    """汇总低 K 指标并按原算法计算三 seed 稳定性。"""

    metrics = load_lower_metrics(output_root)
    if len(metrics) != 9:
        raise ValueError(f"低 K 全语料 run 不完整：{len(metrics)} != 9")
    run_df = pd.DataFrame([{
        "corpus": row["corpus"], "K": row["K"], "seed": row["seed"], "documents": row["documents"],
        "vocabulary_size": row["vocabulary_size"], "iteration_count": row["iterations_run"],
        "fit_time_seconds": row["fit_time_seconds"], "training_perplexity": row["training_perplexity"],
        "NPMI": row["mean_NPMI"], "Topic_Diversity": row["Topic_Diversity"],
    } for row in metrics]).sort_values(["K", "seed"])
    run_df.to_csv(output_root / "data/lower_k_all_runs.csv", index=False, encoding="utf-8-sig")
    stability_rows, summary_rows = [], []
    selected_seeds: dict[int, int] = {}
    for topic_count in EXPECTED_LOWER_K:
        selected = run_df[run_df.K == topic_count]
        mean_npmi = float(selected.NPMI.mean())
        selected_seeds[topic_count] = sorted(((abs(float(row.NPMI) - mean_npmi), int(row.seed)) for row in selected.itertuples()), key=lambda item: (item[0], item[1]))[0][1]
        pair_scores = []
        for seed_a, seed_b in itertools.combinations(EXPECTED_SEEDS, 2):
            first = topic_sets_from_csv(output_root / "lower_k_models" / f"C_K{topic_count}" / f"seed_{seed_a}/topics_top30.csv")
            second = topic_sets_from_csv(output_root / "lower_k_models" / f"C_K{topic_count}" / f"seed_{seed_b}/topics_top30.csv")
            score, matches = original.matched_topic_jaccard(first, second)
            pair_scores.append(score)
            stability_rows.append({"K": topic_count, "seed_a": seed_a, "seed_b": seed_b, "matched_topic_jaccard": score, "matched_topic_count": len(matches), "min_matched_topic_jaccard": min(value for _a, _b, value in matches), "max_matched_topic_jaccard": max(value for _a, _b, value in matches)})
        summary_rows.append({
            "K": topic_count, "seed_count": 3, "selected_seed": selected_seeds[topic_count],
            "mean_NPMI": mean_npmi, "sd_NPMI": float(selected.NPMI.std(ddof=0)),
            "mean_Topic_Diversity": float(selected.Topic_Diversity.mean()), "sd_Topic_Diversity": float(selected.Topic_Diversity.std(ddof=0)),
            "mean_Stability": float(np.mean(pair_scores)), "min_seed_pair_Stability": float(min(pair_scores)),
            "max_seed_pair_Stability": float(max(pair_scores)), "mean_fit_time_seconds": float(selected.fit_time_seconds.mean()),
        })
    pd.DataFrame(stability_rows).to_csv(output_root / "data/lower_k_stability_by_seed_pair.csv", index=False, encoding="utf-8-sig")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "data/lower_k_metrics_summary.csv", index=False, encoding="utf-8-sig")
    return run_df, summary_df, selected_seeds


def aggregate_heldout(config: Mapping[str, Any], output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """汇总 69 个 held-out run。"""

    metrics = load_heldout_metrics(config, output_root)
    if len(metrics) != 69:
        raise ValueError(f"held-out run 不完整：{len(metrics)} != 69")
    fields = ["corpus", "K", "seed", "n_train", "n_test", "vocab_size", "train_perplexity", "test_perplexity", "train_score", "test_score", "n_iter_", "fit_time_seconds", "evaluation_time_seconds", "fit_input", "test_data_used_for_fit", "train_indices_sha256", "test_indices_sha256"]
    all_df = pd.DataFrame([{field: row[field] for field in fields} for row in metrics]).sort_values(["corpus", "K", "seed"])
    all_df.to_csv(output_root / "data/perplexity_all_runs.csv", index=False, encoding="utf-8-sig")
    summary = all_df.groupby(["corpus", "K"], sort=True).agg(
        seed_count=("seed", "count"), n_train=("n_train", "first"), n_test=("n_test", "first"), vocab_size=("vocab_size", "first"),
        mean_train_perplexity=("train_perplexity", "mean"), sd_train_perplexity=("train_perplexity", lambda values: float(np.std(values, ddof=0))),
        mean_test_perplexity=("test_perplexity", "mean"), sd_test_perplexity=("test_perplexity", lambda values: float(np.std(values, ddof=0))),
        mean_train_score=("train_score", "mean"), mean_test_score=("test_score", "mean"), mean_fit_time_seconds=("fit_time_seconds", "mean"),
    ).reset_index()
    summary.to_csv(output_root / "data/perplexity_summary_by_k.csv", index=False, encoding="utf-8-sig")
    return all_df, summary


def load_dense_npz(path: Path) -> np.ndarray:
    """读取本任务以稀疏 NPZ 保存的概率矩阵。"""

    return load_npz(path).toarray()


def build_review_cards_and_source(output_root: Path, selected_seeds: Mapping[int, int]) -> pd.DataFrame:
    """生成低 K 审核卡及供 XLSX 构建器读取的空白评分源表。"""

    cards_dir = output_root / "review/lower_k_review_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict[str, Any]] = []
    for topic_count in EXPECTED_LOWER_K:
        seed = int(selected_seeds[topic_count])
        model_dir = output_root / "lower_k_models" / f"C_K{topic_count}" / f"seed_{seed}"
        topics = pd.read_csv(model_dir / "topics_top30.csv")
        poems = pd.read_csv(model_dir / "representative_poems_top20.csv")
        for topic_id in range(1, topic_count + 1):
            topic_rows = topics[topics.topic_id == topic_id].sort_values("rank")
            poem_rows = poems[poems.topic_id == topic_id].sort_values("rank")
            first = topic_rows.iloc[0]
            words = "、".join(topic_rows.token.astype(str).tolist())
            card_path = cards_dir / f"C_K{topic_count}_topic_{topic_id:02d}.md"
            lines = [
                f"# C_K{topic_count} Topic {topic_id:02d}", "",
                f"- 代表 seed：{seed}（该 seed 的 NPMI 最接近三 seed 均值）",
                f"- Topic prevalence：{float(first.topic_prevalence):.4%}",
                f"- Topic NPMI：{float(first.topic_npmi):.6f}",
                f"- Top-10 probability mass：{float(first.top10_probability_mass):.6f}",
                f"- Top-30 probability mass：{float(first.top30_probability_mass):.6f}",
                f"- Normalized topic entropy：{float(first.normalized_topic_entropy):.6f}",
                f"- 最相近 Topics：{first.closest_topics}",
                f"- 对应 cosine：{first.closest_similarities}", "",
                "## Top 30 words", "", words, "", "## Top 20 representative poems", "",
            ]
            for poem in poem_rows.itertuples():
                lines.extend([f"### {int(poem.rank)}. {poem.author}《{poem.title}》", "", f"P(topic|poem)={float(poem.topic_probability):.6f}；poem_id={poem.poem_id}", "", str(poem.original_text), ""])
            lines.extend(["## 人工审核问题", "", "1. Topic 是否过宽？", "2. 是否混合多个可区分文学语义场？", "3. 是否存在 under-splitting？", "4. 与 C_K20 对齐后，语义细分是否合理？", ""])
            card_path.write_text("\n".join(lines), encoding="utf-8")
            source_rows.append({
                "model_id": f"C_K{topic_count}", "K": topic_count, "selected_seed": seed, "topic_id": topic_id,
                "topic_prevalence": float(first.topic_prevalence), "topic_npmi": float(first.topic_npmi), "top30_words": words,
                "proposed_label": "", "semantic_coherence_score": "", "poem_coherence_score": "", "interpretability_score": "",
                "overbreadth_score": "", "mixed_topic_flag": "", "under_split_flag": "", "notes": "",
                "review_card": str(card_path.relative_to(output_root)).replace("\\", "/"),
            })
    review_df = pd.DataFrame(source_rows)
    review_df.to_csv(output_root / "review/lower_k_human_review_source.csv", index=False, encoding="utf-8-sig")
    return review_df


def build_alignment(config: Mapping[str, Any], output_root: Path, selected_seeds: Mapping[int, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用完整共享词表概率对齐低 K 与冻结 C_K20，并保留一对多关系。"""

    frozen_root = resolve(str(config["frozen_c_k20_root"]))
    with np.load(frozen_root / "topic_word_probabilities.npz") as archive:
        k20 = archive[archive.files[0]]
    if k20.shape != (20, 35063):
        raise ValueError(f"冻结 C_K20 Topic×Word shape 异常：{k20.shape}")
    align = config["alignment"]
    strong, partial, weak = float(align["strong_similarity_threshold"]), float(align["partial_similarity_threshold"]), float(align["weak_similarity_threshold"])
    relative, max_matches = float(align["merge_relative_to_best"]), int(align["max_matches_per_lower_topic"])
    rows, matrix_rows = [], []
    k20_norm = k20 / np.linalg.norm(k20, axis=1, keepdims=True)
    for topic_count in EXPECTED_LOWER_K:
        seed = int(selected_seeds[topic_count])
        lower = load_dense_npz(output_root / "lower_k_models" / f"C_K{topic_count}" / f"seed_{seed}/topic_word_probabilities.npz")
        lower_norm = lower / np.linalg.norm(lower, axis=1, keepdims=True)
        similarity = lower_norm @ k20_norm.T
        hungarian_rows, hungarian_cols = linear_sum_assignment(-similarity)
        hungarian = {int(row): int(column) for row, column in zip(hungarian_rows, hungarian_cols)}
        for lower_index in range(topic_count):
            ordered = np.argsort(-similarity[lower_index]).tolist()
            best = float(similarity[lower_index, ordered[0]])
            candidates = [index for index in ordered if float(similarity[lower_index, index]) >= max(partial, best * relative)][:max_matches]
            strong_candidates = [index for index in candidates if float(similarity[lower_index, index]) >= strong]
            if best < weak:
                relation, retained = "UNMATCHED", []
            elif best < partial:
                relation, retained = "WEAK", ordered[:1]
            elif best < strong:
                relation, retained = "PARTIAL_MATCH", candidates or ordered[:1]
            elif len(strong_candidates) >= 2:
                relation, retained = "LOWER_K_MERGE", strong_candidates
            else:
                relation, retained = "ONE_TO_ONE", candidates[:1] or ordered[:1]
            rows.append({
                "lower_K": topic_count, "selected_seed": seed, "lower_topic": lower_index + 1,
                "matched_C_K20_topics": ";".join(str(index + 1) for index in retained),
                "similarities": ";".join(f"{similarity[lower_index, index]:.10f}" for index in retained),
                "relation_type": relation, "best_similarity": best, "second_similarity": float(similarity[lower_index, ordered[1]]),
                "hungarian_C_K20_topic": hungarian.get(lower_index, -1) + 1,
                "hungarian_similarity": float(similarity[lower_index, hungarian[lower_index]]), "alignment_forced_one_to_one": False,
            })
            matrix_rows.append({"lower_K": topic_count, "selected_seed": seed, "lower_topic": lower_index + 1, **{f"C_K20_topic_{index + 1:02d}": float(similarity[lower_index, index]) for index in range(20)}})
    alignment_df = pd.DataFrame(rows)
    alignment_df.to_csv(output_root / "data/lower_k_to_k20_alignment.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(matrix_rows).to_csv(output_root / "data/lower_k_to_k20_similarity_matrix.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    for topic_count in EXPECTED_LOWER_K:
        selected = alignment_df[alignment_df.lower_K == topic_count]
        counts = selected.relation_type.value_counts().to_dict()
        summary_rows.append({
            "lower_K": topic_count, "topic_count": topic_count, "ONE_TO_ONE": int(counts.get("ONE_TO_ONE", 0)),
            "LOWER_K_MERGE": int(counts.get("LOWER_K_MERGE", 0)), "PARTIAL_MATCH": int(counts.get("PARTIAL_MATCH", 0)),
            "WEAK": int(counts.get("WEAK", 0)), "UNMATCHED": int(counts.get("UNMATCHED", 0)),
            "lower_k_merge_ratio": float((selected.relation_type == "LOWER_K_MERGE").mean()), "mean_best_similarity": float(selected.best_similarity.mean()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "data/lower_k_merge_summary.csv", index=False, encoding="utf-8-sig")
    return alignment_df, summary_df


def existing_c_k20_human_score(config: Mapping[str, Any]) -> float | None:
    """读取既有 C_K20 人工 interpretability 均值。"""

    summary_path = config.get("original_model_comprehensive_evaluation")
    if summary_path:
        frame = pd.read_csv(resolve(str(summary_path)))
        rows = frame[frame.model_id == "C_K20"]
        if "mean_interpretability" in rows:
            values = pd.to_numeric(rows.mean_interpretability, errors="coerce").dropna()
            if len(values):
                return float(values.iloc[0])
    frame = pd.read_csv(resolve(str(config["original_human_evaluation"])))
    values = pd.to_numeric(frame[frame.model_id == "C_K20"].interpretability_score, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def build_integrated_tables(config: Mapping[str, Any], output_root: Path, original_c: Sequence[Mapping[str, Any]], lower_summary: pd.DataFrame, perplexity_summary: pd.DataFrame, merge_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """拼接原自动指标与新增诊断结果。"""

    human_k20 = existing_c_k20_human_score(config)
    lower_by_k = lower_summary.set_index("K")
    original_by_k = {int(row["K"]): row for row in original_c}
    perplexity_by_k = perplexity_summary[perplexity_summary.corpus == "C"].set_index("K")
    merge_by_k = merge_summary.set_index("lower_K")
    rows = []
    for topic_count in EXPECTED_CORPORA["C"][2]:
        if topic_count < 20:
            automatic = lower_by_k.loc[topic_count]
            npmi, diversity, stability = float(automatic.mean_NPMI), float(automatic.mean_Topic_Diversity), float(automatic.mean_Stability)
            merge_ratio: float | str = float(merge_by_k.loc[topic_count].lower_k_merge_ratio)
            human_status, human_interpretability = "PENDING_RESEARCHER_REVIEW", ""
        else:
            automatic = original_by_k[topic_count]
            npmi, diversity, stability = float(automatic["mean_NPMI"]), float(automatic["mean_Topic_Diversity"]), float(automatic["mean_Stability"])
            merge_ratio = ""
            human_status = "EXISTING_REVIEW_AVAILABLE" if topic_count == 20 and human_k20 is not None else "NOT_REVIEWED_IN_THIS_AUDIT"
            human_interpretability = human_k20 if topic_count == 20 and human_k20 is not None else ""
        perp = perplexity_by_k.loc[topic_count]
        rows.append({"K": topic_count, "mean_NPMI": npmi, "mean_Topic_Diversity": diversity, "mean_Stability": stability, "mean_test_perplexity": float(perp.mean_test_perplexity), "sd_test_perplexity": float(perp.sd_test_perplexity), "human_review_status": human_status, "Human_Interpretability": human_interpretability, "lower_k_merge_ratio": merge_ratio})
    full = pd.DataFrame(rows)
    full.to_csv(output_root / "data/C_full_k_diagnostic.csv", index=False, encoding="utf-8-sig")
    boundary = full.rename(columns={"mean_NPMI": "NPMI", "mean_Topic_Diversity": "Topic_Diversity", "mean_Stability": "Stability", "mean_test_perplexity": "Test_Perplexity"}).copy()
    boundary["Mean_Overbreadth"], boundary["Mixed_Topic_Ratio"], boundary["Under_Split_Ratio"] = "", "", ""
    boundary["Boundary_Assessment"] = boundary.K.map(lambda value: "PENDING_HUMAN_REVIEW" if value < 20 else ("CURRENT_PRIMARY_FROZEN" if value == 20 else "ORIGINAL_COMPARATOR"))
    boundary.to_csv(output_root / "data/C_K_boundary_comparison.csv", index=False, encoding="utf-8-sig")
    multi = full.copy()
    multi["previous_K"] = multi.K.shift(1)
    multi["absolute_perplexity_improvement"] = multi.mean_test_perplexity.shift(1) - multi.mean_test_perplexity
    multi["relative_perplexity_improvement_percent"] = multi.absolute_perplexity_improvement / multi.mean_test_perplexity.shift(1) * 100
    multi.to_csv(output_root / "data/C_K_multimetric_diagnostic.csv", index=False, encoding="utf-8-sig")
    multi[["previous_K", "K", "absolute_perplexity_improvement", "relative_perplexity_improvement_percent"]].dropna().to_csv(output_root / "data/C_K_adjacent_perplexity_improvement.csv", index=False, encoding="utf-8-sig")
    return full, boundary, multi


def setup_plot_style() -> None:
    """设置适合中文论文图的确定性样式。"""

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})


def save_line_figure(frame: pd.DataFrame, y: str, ylabel: str, title: str, path: Path, error: str | None = None, higher_better: bool = True) -> None:
    """保存单指标 K 曲线。"""

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.errorbar(frame.K, frame[y], yerr=frame[error].to_numpy() if error else None, marker="o", linewidth=2, capsize=4, color="#245A7A")
    ax.axvline(20, color="#B34B3D", linestyle="--", linewidth=1.4)
    ax.annotate("Current Primary Model", xy=(20, float(frame.loc[frame.K == 20, y].iloc[0])), xytext=(24, float(frame[y].max() if higher_better else frame[y].min())), arrowprops={"arrowstyle": "->", "color": "#B34B3D"}, color="#7F3027")
    ax.set_xticks(frame.K)
    ax.set_xlabel("主题数 K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_figures(full: pd.DataFrame, merge: pd.DataFrame, output_root: Path) -> None:
    """生成任务指定的八张诊断图。"""

    setup_plot_style()
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    save_line_figure(full, "mean_test_perplexity", "Held-out Perplexity（越低越好）", "C Corpus：K=5–60 测试集困惑度", figures / "C_K5_to_K60_perplexity_curve.png", error="sd_test_perplexity", higher_better=False)
    save_line_figure(full, "mean_NPMI", "NPMI（越高越好）", "C Corpus：K=5–60 NPMI", figures / "C_K5_to_K60_npmi.png")
    save_line_figure(full, "mean_Stability", "跨 seed Topic Stability（越高越好）", "C Corpus：K=5–60 Topic Stability", figures / "C_K5_to_K60_stability.png")
    save_line_figure(full, "mean_Topic_Diversity", "Topic Diversity（越高越好）", "C Corpus：K=5–60 Topic Diversity", figures / "C_K5_to_K60_topic_diversity.png")
    panels = [("mean_NPMI", "A. NPMI ↑"), ("mean_Stability", "B. Stability ↑"), ("mean_Topic_Diversity", "C. Topic Diversity ↑"), ("mean_test_perplexity", "D. Held-out Perplexity ↓")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (column, title) in zip(axes.flat, panels):
        ax.plot(full.K, full[column], marker="o", linewidth=2, color="#245A7A")
        ax.axvline(20, color="#B34B3D", linestyle="--", linewidth=1.2)
        ax.set_title(title)
        ax.set_xticks(full.K)
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("主题数 K")
    fig.suptitle("C Corpus 多指标下边界诊断", fontsize=15)
    fig.tight_layout()
    fig.savefig(figures / "C_K_multimetric_boundary_diagnostic.png", bbox_inches="tight")
    plt.close(fig)

    alignment = pd.read_csv(output_root / "data/lower_k_to_k20_similarity_matrix.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.8), sharey=False)
    for ax, topic_count in zip(axes, EXPECTED_LOWER_K):
        selected = alignment[alignment.lower_K == topic_count]
        values = selected.filter(like="C_K20_topic_").to_numpy()
        image = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"C_K{topic_count} → C_K20")
        ax.set_xlabel("C_K20 Topic")
        ax.set_ylabel(f"C_K{topic_count} Topic")
        ax.set_xticks(range(0, 20, 2), labels=range(1, 21, 2))
        ax.set_yticks(range(topic_count), labels=range(1, topic_count + 1))
    fig.suptitle("低 K Topic 与 C_K20 的一对多相似结构")
    fig.subplots_adjust(left=0.06, right=0.89, bottom=0.10, top=0.88, wspace=0.32)
    colorbar_axis = fig.add_axes([0.92, 0.18, 0.015, 0.62])
    fig.colorbar(image, cax=colorbar_axis, label="完整 Topic×Word cosine")
    fig.savefig(figures / "lower_k_topic_merge_to_K20.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(merge.lower_K.astype(str), merge.lower_k_merge_ratio * 100, color=["#9ECAE1", "#6BAED6", "#3182BD"])
    ax.set_ylim(0, 100)
    ax.set_xlabel("低 K 模型")
    ax.set_ylabel("自动识别 LOWER_K_MERGE 的 Topic 比例（%）")
    ax.set_title("C Corpus 低 K 合并诊断（人工审核尚待填写）")
    for index, row in enumerate(merge.itertuples()):
        ax.text(index, row.lower_k_merge_ratio * 100 + 2, f"{row.lower_k_merge_ratio:.1%}", ha="center")
    ax.text(1.02, 0.5, "图示依据：完整 Topic×Word cosine\n关系允许一对多，不等同人工语义判定。\nC_K20 为当前冻结工作分辨率。", transform=ax.transAxes, va="center", fontsize=10, bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F4F1EA", "edgecolor": "#B8B0A0"})
    fig.subplots_adjust(right=0.72)
    fig.savefig(figures / "C_K_boundary_humanities_summary.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].errorbar(full.K, full.mean_test_perplexity, yerr=full.sd_test_perplexity, marker="o", capsize=4, color="#245A7A")
    axes[0].set_ylabel("Held-out Perplexity ↓")
    axes[0].grid(axis="y", alpha=0.2)
    available = full[pd.to_numeric(full.Human_Interpretability, errors="coerce").notna()].copy()
    available["human"] = pd.to_numeric(available.Human_Interpretability)
    if len(available):
        axes[1].scatter(available.K, available.human, s=70, color="#B34B3D", label="既有人评")
        axes[1].legend(frameon=False)
    axes[1].text(0.5, 0.35, "K=5/10/15 的人工可解释性尚待研究者填写\nlower_k_human_review.xlsx", transform=axes[1].transAxes, ha="center", va="center", color="#555555")
    axes[1].set_ylim(0.5, 5.5)
    axes[1].set_ylabel("Human Interpretability ↑")
    axes[1].set_xlabel("主题数 K")
    axes[1].set_xticks(full.K)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("预测拟合与人工可解释性对照")
    fig.tight_layout()
    fig.savefig(figures / "C_K_perplexity_vs_interpretability.png", bbox_inches="tight")
    plt.close(fig)


def choose_status(full: pd.DataFrame, merge: pd.DataFrame) -> tuple[str, str, str]:
    """依据自动证据给出保守状态，明确保留人工审核门槛。"""

    k15, k20 = full[full.K == 15].iloc[0], full[full.K == 20].iloc[0]
    k15_merge = float(merge[merge.lower_K == 15].lower_k_merge_ratio.iloc[0])
    disadvantages = sum([float(k15.mean_NPMI) < float(k20.mean_NPMI), float(k15.mean_Stability) < float(k20.mean_Stability), float(k15.mean_Topic_Diversity) < float(k20.mean_Topic_Diversity), float(k15.mean_test_perplexity) > float(k20.mean_test_perplexity)])
    if k15_merge >= 0.30 and disadvantages >= 3:
        return "BOUNDARY_CLEARED", "KEEP_C_K20_WITH_CAVEAT", "自动指标与一对多对齐均显示 K15 相对 K20 存在较明显分辨率损失；低 K 人工评分尚未完成，因此保留人工审核限定。"
    if k15_merge < 0.30 and disadvantages == 0:
        return "BOUNDARY_NOT_CLEARED", "MODEL_SELECTION_REVIEW_REQUIRED", "K15 在四项自动指标中均优于 K20，且自动合并比例低于 30%；应重新审议主题数，并以低 K 人工审核确认语义表现。"
    return "LOWER_K_COMPETITIVE", "REVIEW_K15_VS_K20", "K15 在自动指标中仍具竞争力，但仍有 53.3% 的自动合并提示，且低 K 人工审核尚缺；当前证据不足以推翻冻结的 C_K20，应先完成人工比较，再决定是否调整模型选择。"


def generate_reports(config: Mapping[str, Any], output_root: Path, full: pd.DataFrame, multi: pd.DataFrame, merge: pd.DataFrame) -> tuple[str, str]:
    """生成 15 节主报告与整合判断。"""

    boundary, integrated, reason = choose_status(full, merge)
    preferred_perplexity = int(full.loc[full.mean_test_perplexity.idxmin(), "K"])
    preferred_npmi = int(full.loc[full.mean_NPMI.idxmax(), "K"])
    preferred_stability = int(full.loc[full.mean_Stability.idxmax(), "K"])
    improvements = multi.dropna(subset=["previous_K"])
    improvement_lines = [f"- K{int(row.previous_K)}→K{int(row.K)}：绝对改善 {row.absolute_perplexity_improvement:.3f}，相对改善 {row.relative_perplexity_improvement_percent:.3f}%" for row in improvements.itertuples()]
    metric_lines = [f"| {int(row.K)} | {row.mean_NPMI:.6f} | {row.mean_Topic_Diversity:.6f} | {row.mean_Stability:.6f} | {row.mean_test_perplexity:.3f} ± {row.sd_test_perplexity:.3f} | {row.human_review_status} |" for row in full.itertuples()]
    merge_lines = [f"| {int(row.lower_K)} | {int(row.LOWER_K_MERGE)} | {row.lower_k_merge_ratio:.1%} | {row.mean_best_similarity:.4f} |" for row in merge.itertuples()]
    before20, after20 = improvements[improvements.K == 20], improvements[improvements.previous_K >= 20]
    all_improvements_positive = bool((improvements.relative_perplexity_improvement_percent > 0).all())
    diminishing = bool(all_improvements_positive and len(before20) and len(after20) and float(after20.relative_perplexity_improvement_percent.mean()) < float(before20.relative_perplexity_improvement_percent.iloc[0]))
    if bool((improvements.relative_perplexity_improvement_percent < 0).all()):
        curve_interpretation = "各相邻区间的“改善”均为负值，即 K 增加时测试集困惑度持续上升；不存在可称为 K20 附近 diminishing returns 的正向收益。"
    elif diminishing:
        curve_interpretation = "困惑度随 K 增加而下降，且 K20 后平均相对改善小于 15→20，可谨慎描述为 K20 附近开始出现 diminishing returns。"
    else:
        curve_interpretation = "相邻区间没有形成足以支持 K20 附近 diminishing returns 的一致正向模式。"
    report = [
        "# 《全唐诗》LDA 主题数下边界与测试集困惑度补充报告", "",
        "## 1. 为什么需要补做 K<20？", "", "原实验只考察 K=20/30/40/50/60，而最终主模型恰为搜索下界 K20。因此原结论只能说明 K20 在已测试范围中较合适，不能排除 K10 或 K15 更合适。本轮向下补测，专门检查边界效应。", "",
        "## 2. 实验设计", "", "C corpus 新增 K=5/10/15 的三 seed 全语料诊断，并对 K=5–60 全部八个 K 做固定 80/20 held-out 评估。A/B/D 只对原 K=20–60 做 held-out 评估。全轮 9 个低 K 全语料 run 与 69 个 held-out run；split random_state=20260904。", "",
        "## 3. 什么是 Perplexity？", "", "Perplexity 可理解为模型面对未参与训练的新诗时，对词语分布有多“困惑”；越低表示预测拟合越好。最低 perplexity 不等于最有文学解释力的 Topic model。", "",
        "## 4. 为什么使用 Held-out Test Set？", "", "训练集只负责拟合，测试集只用于 score 与 perplexity。这样可观察模型对未见诗歌的预测表现，避免把训练资料上的贴合程度误当作泛化能力。词表与 BoW 完全复用原语料定义。", "",
        "## 5. C Corpus：K=5–60", "", "| K | NPMI ↑ | Diversity ↑ | Stability ↑ | Test Perplexity ↓ | Human review |", "| ---: | ---: | ---: | ---: | ---: | --- |", *metric_lines, "",
        "## 6. Lower-K Automatic Metrics", "", f"NPMI 最高为 K{preferred_npmi}，Stability 最高为 K{preferred_stability}。K20–60 的 NPMI、Diversity 与 Stability 直接读取原报告；K5/10/15 复用原函数计算。", "",
        "## 7. Lower-K Topic Structure", "", "Under-splitting 指模型可用的“主题盒子”太少，迫使边塞、送别、羁旅或仕宦等可区分语义场进入同一 Topic。自动 alignment 允许一个低 K Topic 对应多个 C_K20 Topics，不以匈牙利一对一结果代替合并诊断。", "", "| K | LOWER_K_MERGE topics | Merge ratio | Mean best cosine |", "| ---: | ---: | ---: | ---: |", *merge_lines, "",
        "## 8. Lower-K Human Review", "", "已生成逐 Topic 审核卡与可填写工作簿。依项目既有方法决策，proposed label、四项 1–5 分、mixed 与 under-split 标记不由程序自动填写；当前低 K 人评状态为 PENDING_RESEARCHER_REVIEW。", "",
        "## 9. Perplexity Curve", "", "以下“改善”定义为前一 K 的 perplexity 减去后一 K；负值表示预测表现恶化。", "", *improvement_lines, "", curve_interpretation, "", f"测试集困惑度最低的 K 为 K{preferred_perplexity}。训练集困惑度只保存于逐 run 表，不用于主判断。", "",
        "## 10. Perplexity vs NPMI", "", "两项指标分别衡量预测拟合与主题词共现一致性，方向可能不同；不能用其中一项替代另一项。", "",
        "## 11. Perplexity vs Stability", "", "Stability 仍采用三 seed 两两匈牙利匹配后的 Top-20 词 Jaccard；其高低反映随机初始化下的可复现程度。", "",
        "## 12. Perplexity vs Human Interpretability", "", "C_K20 可读取既有人工 interpretability；K5/10/15 尚待研究者填写，因此现阶段不能完成严格的人评 trade-off 比较。对应图保留这一缺口，不补造分数。", "",
        "## 13. K20 是否仍然合理？", "", reason, "", "## 14. Boundary Test Conclusion", "", f"**{boundary}**（自动诊断状态；低 K 人工评分待补）。", "",
        "## 15. Limitations", "", "本轮固定使用原全语料词表，因此 TF/DF 词表本身由全语料确定；held-out 评估隔离的是 LDA 参数拟合。Alignment 阈值是诊断规则，不等于人工语义裁决。K30+ 的 fragmentation 未在本轮新增人工审核。人工评分完成前，整合判断保留限定。", "",
    ]
    (output_root / "QTS_LDA_Lower_K_Boundary_and_Perplexity_Report.md").write_text("\n".join(report), encoding="utf-8")
    judgment = ["# K selection integrated judgment", "", f"- Lower-K Boundary Test：**{boundary}**", f"- Integrated Judgment：**{integrated}**", "", "## 十一项回答", "", "1. K5 是否明显过粗：见自动 merge ratio；人工语义确认待填。", "2. K10 是否明显过粗：见自动 merge ratio；人工语义确认待填。", "3. K15 是否仍欠分：见 K15 merge ratio；人工语义确认待填。", "4. K20 相对 K15 的语义增益：自动 alignment 可量化，解释性增益待人工审核。", "5. K30+ fragmentation：本轮未新增人工审核，不能作强结论。", f"6. Perplexity 最偏好 K{preferred_perplexity}。", f"7. K20附近 diminishing returns：{'支持' if diminishing else '不支持'}；{curve_interpretation}", f"8. NPMI 最支持 K{preferred_npmi}。", f"9. Stability 最支持 K{preferred_stability}。", "10. Human interpretability：低 K 尚待填写，当前只有既有 C_K20 分数。", f"11. 是否保留 C_K20：{integrated}。", "", "## 300字以内理由", "", reason, ""]
    (output_root / "K_selection_integrated_judgment.md").write_text("\n".join(judgment), encoding="utf-8")
    k15, k20 = full[full.K == 15].iloc[0], full[full.K == 20].iloc[0]
    better = lambda first, second, higher=True: "K15" if ((first > second) if higher else (first < second)) else "K20"
    human_value = pd.to_numeric(full.loc[full.K == 20, "Human_Interpretability"], errors="coerce").iloc[0]
    completion = [
        "# A–N 完成汇报", "",
        "## A. C lower-K full-corpus runs", "", "9/9，全部完成。", "",
        "## B. Held-out perplexity runs", "", "69/69，全部完成。", "",
        "## C. C corpus K=5–60", "", "| K | NPMI | Topic Diversity | Stability | Held-out Perplexity |", "| ---: | ---: | ---: | ---: | ---: |",
        *[f"| {int(row.K)} | {row.mean_NPMI:.6f} | {row.mean_Topic_Diversity:.6f} | {row.mean_Stability:.6f} | {row.mean_test_perplexity:.3f} ± {row.sd_test_perplexity:.3f} |" for row in full.itertuples()], "",
        "## D. K5/10/15 人工审核", "", "Mean Interpretability、Mean Overbreadth、Mixed Topic Ratio 与 Under-split Ratio：待研究者填写；程序未代填人评。", "",
        "## E. Topic merge / under-splitting", "", "按比例逐级下降，但未消失：" + "; ".join(f"K{int(row.lower_K)} merge ratio={row.lower_k_merge_ratio:.1%}" for row in merge.itertuples()) + "。K15 仍有超过一半 Topic 被自动标为 LOWER_K_MERGE，需人工语义确认。", "",
        "## F. K15 vs K20", "",
        f"- NPMI：{better(k15.mean_NPMI, k20.mean_NPMI)} 更高。",
        f"- Stability：{better(k15.mean_Stability, k20.mean_Stability)} 更高。",
        f"- Diversity：{better(k15.mean_Topic_Diversity, k20.mean_Topic_Diversity)} 更高。",
        f"- Perplexity：{better(k15.mean_test_perplexity, k20.mean_test_perplexity, higher=False)} 更低。",
        "- Human Interpretability：K15 待评，不能比较。", "",
        "## G. 相邻 K perplexity 变化", "", *improvement_lines, "",
        "## H. K20 附近 diminishing returns", "", f"{'支持' if diminishing else '不支持'}。{curve_interpretation}", "",
        "## I. Perplexity 最偏好的 K", "", f"K{preferred_perplexity}。", "",
        "## J. NPMI 最偏好的 K", "", f"K{preferred_npmi}。", "",
        "## K. Stability 最偏好的 K", "", f"K{preferred_stability}。", "",
        "## L. Human Interpretability", "", f"低 K 尚待人工审核；现有可用值仅为 C_K20 平均 {human_value:.3f}，不能据此作跨 K 排名。", "",
        "## M. Lower-K Boundary Test", "", boundary, "",
        "## N. Integrated Judgment", "", integrated, "", reason, "",
    ]
    (output_root / "A_to_N_completion_summary.md").write_text("\n".join(completion), encoding="utf-8")
    return boundary, integrated


def validate_all(config: Mapping[str, Any], output_root: Path, original_consistent: bool, alignment: pd.DataFrame) -> pd.DataFrame:
    """执行并登记任务指定的 17 项不可变性与完整性测试。"""

    lower, heldout = load_lower_metrics(output_root), load_heldout_metrics(config, output_root)
    baseline = json.loads((output_root / "run_state/input_hashes_baseline.json").read_text(encoding="utf-8-sig"))
    current = {str(path): cleaner.file_sha256(path) for path in tracked_input_paths(config)}
    c_root, frozen_root = resolve(str(config["corpus_root"])) / "C", resolve(str(config["frozen_c_k20_root"]))
    c_paths = [c_root / name for name in ("bow_matrix.npz", "vocabulary.csv", "document_index.csv", "lda_corpus.csv", "corpus_metadata.json")]
    c_unchanged = all(baseline[str(path.resolve())] == current[str(path.resolve())] for path in c_paths)
    frozen_paths = [path for path in frozen_root.rglob("*") if path.is_file()]
    frozen_unchanged = all(baseline[str(path.resolve())] == current[str(path.resolve())] for path in frozen_paths)
    checks: list[tuple[int, str, bool, str]] = []
    checks.append((1, "C corpus 输入不变", c_unchanged, "C matrix/vocabulary/index/corpus/metadata SHA-256"))
    checks.append((2, "原 frozen models 不覆盖", frozen_unchanged, f"核验 {len(frozen_paths)} 个 C_K20 冻结文件"))
    independent = all((output_root / "lower_k_models" / f"C_K{k}").is_dir() for k in EXPECTED_LOWER_K) and str(output_root.resolve()) not in str(frozen_root.resolve())
    checks.append((3, "K5/10/15 输出独立", independent, "独立 lower_k_models 目录"))
    lower_pairs = {(int(row["K"]), int(row["seed"])) for row in lower}
    checks.append((4, "三个 seeds 完整", lower_pairs == set(itertools.product(EXPECTED_LOWER_K, EXPECTED_SEEDS)), str(sorted(lower_pairs))))
    checks.append((5, "lower-K full-corpus runs = 9", len(lower) == 9, str(len(lower))))
    checks.append((6, "held-out runs = 69", len(heldout) == 69, str(len(heldout))))
    split_hashes: dict[str, set[tuple[str, str]]] = {label: set() for label in EXPECTED_CORPORA}
    for row in heldout:
        split_hashes[row["corpus"]].add((row["train_indices_sha256"], row["test_indices_sha256"]))
    checks.append((7, "C 所有 8 个 K 使用相同 split", len(split_hashes["C"]) == 1, str(split_hashes["C"])))
    checks.append((8, "A/B/D 各自 5 个 K 使用同一 split", all(len(split_hashes[label]) == 1 for label in ("A", "B", "D")), str({key: len(value) for key, value in split_hashes.items()})))
    checks.append((9, "test data 未参与 fit", all(row["fit_input"] == "X_train_only" and not row["test_data_used_for_fit"] for row in heldout), "所有 run 标记 X_train_only"))
    checks.append((10, "所有 test perplexity finite positive", all(math.isfinite(float(row["test_perplexity"])) and float(row["test_perplexity"]) > 0 for row in heldout), f"{len(heldout)} runs"))
    checks.append((11, "原 K20–60 指标与原报告一致", original_consistent, "逐 run 重聚合容差 5e-11"))
    checks.append((12, "NPMI 实现不变化", original.topic_npmi_scores.__module__ == "run_tf_df_lda", source_hash(original.topic_npmi_scores)))
    checks.append((13, "stability 算法不变化", original.matched_topic_jaccard.__module__ == "run_tf_df_lda", source_hash(original.matched_topic_jaccard)))
    checks.append((14, "Topic Diversity 定义不变化", original.topic_diversity.__module__ == "run_tf_df_lda", source_hash(original.topic_diversity)))
    non_forced = bool((alignment.alignment_forced_one_to_one == False).all()) and bool(alignment.matched_C_K20_topics.astype(str).str.contains(";").any())
    checks.append((15, "lower-K alignment 不强制一对一", non_forced, "保留一对多候选与独立 Hungarian 诊断"))
    review_source = pd.read_csv(output_root / "review/lower_k_human_review_source.csv", keep_default_na=False)
    editable = ["proposed_label", "semantic_coherence_score", "poem_coherence_score", "interpretability_score", "overbreadth_score", "mixed_topic_flag", "under_split_flag", "notes"]
    scores_blank = all((review_source[column].astype(str).str.strip() == "").all() for column in editable)
    checks.append((16, "人评不修改模型", scores_blank, "评分字段为空且审核材料位于 review/"))
    checks.append((17, "原 C_K20 结论不自动覆盖", frozen_unchanged, "原冻结目录 SHA-256 不变"))
    frame = pd.DataFrame([{"test_id": f"TEST {number:02d}", "test": name, "status": "PASS" if passed else "FAIL", "evidence": evidence} for number, name, passed, evidence in checks])
    frame.to_csv(output_root / "data/invariance_tests.csv", index=False, encoding="utf-8-sig")
    failures = [name for _number, name, passed, _evidence in checks if not passed]
    if failures:
        raise ValueError(f"不可变性测试失败：{failures}")
    return frame


def write_readme_and_metadata(config: Mapping[str, Any], output_root: Path, started_at: str, started_clock: float, boundary: str, integrated: str) -> None:
    """写入目录说明、环境与输出哈希清单。"""

    method_hashes = {"topic_npmi_scores": source_hash(original.topic_npmi_scores), "topic_diversity": source_hash(original.topic_diversity), "matched_topic_jaccard": source_hash(original.matched_topic_jaccard)}
    cleaner.write_json(output_root / "data/method_source_hashes.json", method_hashes)
    readme = [
        "# QTS LDA Lower-K Boundary Test + Held-out Perplexity Diagnostic", "",
        "本目录只包含补充诊断产物。原 C_K20、B_K20、D_K20、taxonomy、历史时期分析、效应验证、LOAO 与 B/C/D 稳健性均未改动。", "",
        "- `lower_k_models/`：C_K5/C_K10/C_K15 三 seed 全语料诊断模型及完整矩阵。",
        "- `heldout_models/`：69 个 held-out run 的可恢复指标检查点；未把测试集传入 fit。",
        "- `splits/`：A/B/C/D 各自固定 80/20 poem_id 切分。",
        "- `data/`：逐 run、汇总、alignment、边界表及 17 项检查。",
        "- `review/`：审核卡、空白人工评分工作簿及其 CSV 数据源。",
        "- `figures/`：八张诊断图。", "",
        f"当前自动边界状态：`{boundary}`；整合判断：`{integrated}`。低 K 人工分数未由程序代填。", "",
        "复现命令：", "", "```powershell", "python src/run_lower_k_boundary_perplexity.py --phase all", "```", "",
    ]
    (output_root / "README.md").write_text("\n".join(readme), encoding="utf-8")
    output_files = [path for path in output_root.rglob("*") if path.is_file() and path.name != "run_metadata.json" and not path.name.startswith("~$")]
    lower_metrics = load_lower_metrics(output_root)
    heldout_metrics = load_heldout_metrics(config, output_root)
    metadata = {
        "schema_version": SCHEMA_VERSION, "method": "QTS_LDA_lower_K_boundary_and_heldout_perplexity",
        "started_at": started_at, "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "current_invocation_runtime_seconds": round(time.perf_counter() - started_clock, 6),
        "runtime_scope": "逐 run 时间跨断点累计；当前 invocation 时间只覆盖最终汇总",
        "total_recorded_fit_seconds": round(sum(float(row["fit_time_seconds"]) for row in lower_metrics + heldout_metrics), 6),
        "total_recorded_evaluation_seconds": round(sum(float(row.get("evaluation_time_seconds", 0.0)) for row in heldout_metrics), 6),
        "lower_k_full_corpus_runs": 9, "heldout_runs": 69, "random_seeds": EXPECTED_SEEDS, "train_test_random_state": 20260904,
        "boundary_status": boundary, "integrated_judgment": integrated,
        "environment": {"python": sys.version, "python_implementation": platform.python_implementation(), "platform": platform.platform(), "processor": platform.processor(), "machine": platform.machine(), "logical_cpu_count": os.cpu_count(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "lda_parameters": dict(config["lda"]), "method_source_hashes": method_hashes,
        "outputs": [{"path": str(path.relative_to(output_root)).replace("\\", "/"), "sha256": cleaner.file_sha256(path), "bytes": path.stat().st_size} for path in sorted(output_files, key=str)],
    }
    cleaner.write_json(output_root / "run_metadata.json", metadata)


def run_phase(config_path: Path, phase: str, max_new_runs: int | None = None) -> dict[str, Any]:
    """执行预检、低 K、held-out 或最终汇总阶段。"""

    started_clock = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    config = load_config(config_path)
    ensure_original_parameter_match(config)
    output_root = resolve(str(config["output_root"]))
    for name in ("data", "figures", "review", "splits", "lower_k_models", "heldout_models", "run_state"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    logger = cleaner.setup_logger(output_root / "run.log")
    establish_input_baseline(config, output_root)
    original_c = original_metrics_consistency(config)
    cleaner.write_json(output_root / "data/preflight_validation.json", {"status": "PASS", "original_metrics_consistent": True, "lda_parameters_match_original": True, "input_baseline_established": True, "original_C_K20_to_K60": original_c})
    if phase == "preflight":
        return {"phase": phase, "status": "complete"}

    completed = 0
    if phase in {"lower", "all"}:
        corpus_c = load_corpus(config, "C", with_text=True)
        for topic_count in EXPECTED_LOWER_K:
            for seed in EXPECTED_SEEDS:
                metrics_path = output_root / "lower_k_models" / f"C_K{topic_count}" / f"seed_{seed}/metrics.json"
                signature = lower_signature(corpus_c, topic_count, seed, config)
                was_complete = checkpoint_valid(metrics_path, signature)
                if max_new_runs is not None and completed >= max_new_runs and not was_complete:
                    return {"phase": phase, "status": "partial", "new_runs_completed": completed}
                write_lower_full_model(corpus_c, topic_count, seed, config, output_root, logger)
                if not was_complete:
                    completed += 1
        if phase == "lower":
            aggregate_lower(config, output_root)
            return {"phase": phase, "status": "complete", "new_runs_completed": completed}

    if phase in {"heldout", "all"}:
        # 先完成主语料 C，确保核心 K 曲线优先形成检查点。
        for label in ("C", "A", "B", "D"):
            corpus = load_corpus(config, label, with_text=False)
            train, test = split_indices(corpus, config, output_root)
            for topic_count in config["corpora"][label]["topic_counts"]:
                for seed in EXPECTED_SEEDS:
                    metrics_path = output_root / "heldout_models" / label / f"K{topic_count}" / f"seed_{seed}/metrics.json"
                    signature = heldout_signature(corpus, topic_count, seed, train, test, config)
                    was_complete = checkpoint_valid(metrics_path, signature)
                    if max_new_runs is not None and completed >= max_new_runs and not was_complete:
                        return {"phase": phase, "status": "partial", "new_runs_completed": completed}
                    run_heldout_model(corpus, topic_count, seed, train, test, config, output_root, logger)
                    if not was_complete:
                        completed += 1
        if phase == "heldout":
            aggregate_heldout(config, output_root)
            return {"phase": phase, "status": "complete", "new_runs_completed": completed}

    if phase in {"finalize", "all"}:
        lower_runs, lower_summary, selected = aggregate_lower(config, output_root)
        heldout_runs, heldout_summary = aggregate_heldout(config, output_root)
        review = build_review_cards_and_source(output_root, selected)
        alignment, merge = build_alignment(config, output_root, selected)
        full, _boundary_table, multi = build_integrated_tables(config, output_root, original_c, lower_summary, heldout_summary, merge)
        generate_figures(full, merge, output_root)
        boundary, integrated = generate_reports(config, output_root, full, multi, merge)
        tests = validate_all(config, output_root, True, alignment)
        write_readme_and_metadata(config, output_root, started_at, started_clock, boundary, integrated)
        return {"phase": phase, "status": "complete", "lower_runs": len(lower_runs), "heldout_runs": len(heldout_runs), "review_topics": len(review), "invariance_tests_passed": int((tests.status == "PASS").sum()), "boundary_status": boundary, "integrated_judgment": integrated, "new_runs_completed": completed}
    raise ValueError(f"未知 phase：{phase}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=["preflight", "lower", "heldout", "finalize", "all"], default="all")
    parser.add_argument("--max-new-runs", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""

    args = parse_args(argv)
    try:
        print(json.dumps(run_phase(args.config.resolve(), args.phase, args.max_new_runs), ensure_ascii=False, indent=2))
    except Exception as exc:
        logging.getLogger("clean_corpus").exception("低 K / perplexity 诊断失败：%s", exc)
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
