"""生成 A/B/C/D 四套 TF×DF 语料并运行第一轮 LDA 敏感性实验。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csc_matrix, csr_matrix, load_npz, save_npz
from sklearn.decomposition import LatentDirichletAllocation

import clean_corpus as cleaner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "tf_df_lda_config.yaml"
REQUIRED_CORPORA = {
    "A": (1, 1),
    "B": (2, 2),
    "C": (3, 3),
    "D": (5, 3),
}
MODEL_SCHEMA_VERSION = 1
METRIC_SCHEMA_VERSION = 1


@dataclass
class CorpusArtifact:
    """保存一套 TF×DF 语料及其稀疏矩阵信息。"""

    label: str
    min_tf: int
    min_df: int
    directory: Path
    corpus_path: Path
    tokens_path: Path
    vocabulary_path: Path
    matrix_path: Path
    document_index_path: Path
    empty_poems_path: Path
    metadata_path: Path
    vocabulary: list[str]
    token_to_index: dict[str, int]
    matrix: csr_matrix
    valid_record_indices: list[int]
    decisions: dict[str, tuple[bool, str, tuple[str, ...], bool]]
    statistics: dict[str, Any]
    matrix_fingerprint: str


def json_sha256(value: Any) -> str:
    """对规范化 JSON 值计算稳定哈希。"""

    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def matrix_fingerprint(matrix: csr_matrix) -> str:
    """按稀疏矩阵逻辑内容计算与 NPZ 容器无关的哈希。"""

    canonical = matrix.copy()
    canonical.sort_indices()
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype=np.int64).tobytes())
    digest.update(canonical.indptr.astype(np.int64, copy=False).tobytes())
    digest.update(canonical.indices.astype(np.int64, copy=False).tobytes())
    digest.update(canonical.data.astype(np.int64, copy=False).tobytes())
    return digest.hexdigest()


def validate_corpus_definitions(definitions: Mapping[str, Any]) -> None:
    """确认 A/B/C/D 阈值与用户确认的默认设计完全一致。"""

    actual: dict[str, tuple[int, int]] = {}
    for label, expected in REQUIRED_CORPORA.items():
        value = cleaner.require_mapping(definitions.get(label), f"corpora.{label}")
        actual[label] = (int(value["min_tf"]), int(value["min_df"]))
        if actual[label] != expected:
            raise cleaner.CleaningError(
                f"{label} 阈值必须为 TF>={expected[0]}、DF>={expected[1]}，"
                f"当前为 {actual[label]}"
            )
    if set(definitions) != set(REQUIRED_CORPORA):
        raise cleaner.CleaningError("corpora 只能包含 A、B、C、D")


def build_decisions(
    token_tf: Counter[str],
    token_df: Counter[str],
    effective_poem_count: int,
    protected_words: set[str],
    stopwords: set[str],
    min_tf: int,
    min_df: int,
    automatic_max_df: Any,
) -> dict[str, tuple[bool, str, tuple[str, ...], bool]]:
    """按现有清洗优先级计算一套 TF×DF 判断。"""

    return {
        token: cleaner.token_decision(
            token,
            tf,
            token_df[token],
            cleaner.ratio(token_df[token], effective_poem_count),
            protected_words,
            stopwords,
            min_tf,
            min_df,
            automatic_max_df,
        )
        for token, tf in token_tf.items()
    }


def build_sparse_matrix(
    documents: Sequence[Sequence[str]], token_to_index: Mapping[str, int]
) -> csr_matrix:
    """按既定词典把非空 token 序列转为词频 CSR 矩阵。"""

    indptr = [0]
    indices: list[int] = []
    data: list[int] = []
    for document in documents:
        counts = Counter(document)
        ordered = sorted(
            ((token_to_index[token], count) for token, count in counts.items()),
            key=lambda item: item[0],
        )
        indices.extend(index for index, _count in ordered)
        data.extend(count for _index, count in ordered)
        indptr.append(len(indices))
    matrix = csr_matrix(
        (
            np.asarray(data, dtype=np.int32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(documents), len(token_to_index)),
        dtype=np.int32,
    )
    matrix.sort_indices()
    return matrix


def percentile(values: Sequence[int], probability: float) -> float:
    """使用线性插值计算文档长度百分位数。"""

    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_corpus_artifact(
    *,
    label: str,
    min_tf: int,
    min_df: int,
    records: Sequence[cleaner.PoemRecord],
    token_tf: Counter[str],
    token_df: Counter[str],
    effective_poem_count: int,
    protected_words: set[str],
    stopwords: set[str],
    automatic_max_df: Any,
    token_separator: str,
    output_root: Path,
) -> CorpusArtifact:
    """生成一套语料、词典、有效诗索引与稀疏 BoW。"""

    decisions = build_decisions(
        token_tf,
        token_df,
        effective_poem_count,
        protected_words,
        stopwords,
        min_tf,
        min_df,
        automatic_max_df,
    )
    cleaned_by_poem = [
        [token for token in record.normalized_tokens if decisions[token][0]]
        for record in records
    ]
    vocabulary = sorted(
        {token for tokens in cleaned_by_poem for token in tokens},
        key=lambda token: (-token_tf[token], token),
    )
    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    valid_record_indices = [
        index for index, tokens in enumerate(cleaned_by_poem) if tokens
    ]
    valid_documents = [cleaned_by_poem[index] for index in valid_record_indices]
    matrix = build_sparse_matrix(valid_documents, token_to_index)
    lengths = [len(tokens) for tokens in cleaned_by_poem]
    total_tokens = sum(lengths)
    protected_vocabulary = set(vocabulary) & protected_words
    stats = {
        "label": label,
        "min_tf": min_tf,
        "min_df": min_df,
        "automatic_max_df": automatic_max_df,
        "poem_count": len(records),
        "valid_poem_count": len(valid_record_indices),
        "empty_poem_count": len(records) - len(valid_record_indices),
        "total_tokens": total_tokens,
        "vocabulary_size": len(vocabulary),
        "matrix_shape": list(matrix.shape),
        "matrix_nnz": int(matrix.nnz),
        "document_length": {
            "min": min(lengths, default=0),
            "mean": cleaner.ratio(total_tokens, len(records)),
            "median": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": max(lengths, default=0),
        },
        "protected_vocabulary_size": len(protected_vocabulary),
        "protected_token_occurrences": sum(
            token_tf[token] for token in protected_vocabulary
        ),
        "empty_documents_excluded_from_lda": True,
    }

    directory = output_root / label
    directory.mkdir(parents=True, exist_ok=True)
    corpus_path = directory / "lda_corpus.csv"
    tokens_path = directory / "lda_tokens.txt"
    vocabulary_path = directory / "vocabulary.csv"
    matrix_path = directory / "bow_matrix.npz"
    document_index_path = directory / "document_index.csv"
    empty_poems_path = directory / "empty_poems.csv"
    metadata_path = directory / "corpus_metadata.json"

    cleaner.write_csv(
        corpus_path,
        [
            "poem_id",
            "author",
            "title",
            "original_text",
            "cleaned_tokens",
            "token_count",
            "empty_after_filter",
        ],
        (
            {
                "poem_id": record.poem_id,
                "author": record.author,
                "title": record.title,
                "original_text": record.original_text,
                "cleaned_tokens": token_separator.join(tokens),
                "token_count": len(tokens),
                "empty_after_filter": not tokens,
            }
            for record, tokens in zip(records, cleaned_by_poem)
        ),
    )
    with tokens_path.open("w", encoding="utf-8", newline="\n") as handle:
        for tokens in cleaned_by_poem:
            handle.write(token_separator.join(tokens) + "\n")
    cleaner.write_csv(
        vocabulary_path,
        ["term_index", "token", "TF", "DF", "DF_ratio", "protected"],
        (
            {
                "term_index": index,
                "token": token,
                "TF": token_tf[token],
                "DF": token_df[token],
                "DF_ratio": f"{cleaner.ratio(token_df[token], effective_poem_count):.8f}",
                "protected": str(token in protected_words).lower(),
            }
            for index, token in enumerate(vocabulary)
        ),
    )
    cleaner.write_csv(
        document_index_path,
        ["matrix_row", "corpus_row", "poem_id", "author", "title", "token_count"],
        (
            {
                "matrix_row": matrix_row,
                "corpus_row": record_index,
                "poem_id": records[record_index].poem_id,
                "author": records[record_index].author,
                "title": records[record_index].title,
                "token_count": len(cleaned_by_poem[record_index]),
            }
            for matrix_row, record_index in enumerate(valid_record_indices)
        ),
    )
    cleaner.write_csv(
        empty_poems_path,
        ["corpus_row", "poem_id", "author", "title", "original_text"],
        (
            {
                "corpus_row": index,
                "poem_id": record.poem_id,
                "author": record.author,
                "title": record.title,
                "original_text": record.original_text,
            }
            for index, record in enumerate(records)
            if not cleaned_by_poem[index]
        ),
    )
    save_npz(matrix_path, matrix, compressed=True)
    fingerprint = matrix_fingerprint(matrix)
    metadata = {
        **stats,
        "vocabulary_order": "TF descending, then token ascending",
        "matrix_dtype": str(matrix.dtype),
        "matrix_fingerprint": fingerprint,
        "outputs": {
            "lda_corpus": cleaner.file_sha256(corpus_path),
            "lda_tokens": cleaner.file_sha256(tokens_path),
            "vocabulary": cleaner.file_sha256(vocabulary_path),
            "bow_matrix_npz": cleaner.file_sha256(matrix_path),
            "document_index": cleaner.file_sha256(document_index_path),
            "empty_poems": cleaner.file_sha256(empty_poems_path),
        },
    }
    cleaner.write_json(metadata_path, metadata)
    return CorpusArtifact(
        label=label,
        min_tf=min_tf,
        min_df=min_df,
        directory=directory,
        corpus_path=corpus_path,
        tokens_path=tokens_path,
        vocabulary_path=vocabulary_path,
        matrix_path=matrix_path,
        document_index_path=document_index_path,
        empty_poems_path=empty_poems_path,
        metadata_path=metadata_path,
        vocabulary=vocabulary,
        token_to_index=token_to_index,
        matrix=matrix,
        valid_record_indices=valid_record_indices,
        decisions=decisions,
        statistics=stats,
        matrix_fingerprint=fingerprint,
    )


def topic_top_indices(
    topic_word: np.ndarray, top_words: int
) -> list[list[int]]:
    """稳定提取每个主题概率最高的词索引。"""

    result: list[list[int]] = []
    for weights in topic_word:
        candidate = np.argpartition(weights, -top_words)[-top_words:]
        ordered = sorted(candidate.tolist(), key=lambda index: (-weights[index], index))
        result.append(ordered)
    return result


def topic_npmi_scores(
    matrix: csr_matrix, top_indices: Sequence[Sequence[int]]
) -> tuple[list[float], float]:
    """以诗歌为文档，用 top-word 两两 NPMI 计算主题一致性。"""

    document_count = matrix.shape[0]
    unique_indices = sorted({index for topic in top_indices for index in topic})
    local_index = {global_index: index for index, global_index in enumerate(unique_indices)}
    binary = matrix[:, unique_indices].astype(np.int32, copy=True)
    binary.data[:] = 1
    binary_csc = csc_matrix(binary)
    document_frequency = np.asarray(binary_csc.sum(axis=0)).ravel().astype(np.float64)
    cooccurrence = (binary_csc.T @ binary_csc).toarray().astype(np.float64)
    topic_scores: list[float] = []
    for topic in top_indices:
        pair_scores: list[float] = []
        for first, second in itertools.combinations(topic, 2):
            first_local = local_index[first]
            second_local = local_index[second]
            pair_df = cooccurrence[first_local, second_local]
            if pair_df <= 0:
                pair_scores.append(-1.0)
                continue
            pair_probability = pair_df / document_count
            first_probability = document_frequency[first_local] / document_count
            second_probability = document_frequency[second_local] / document_count
            pmi = math.log(pair_probability / (first_probability * second_probability))
            pair_scores.append(pmi / -math.log(pair_probability))
        topic_scores.append(float(np.mean(pair_scores)) if pair_scores else 0.0)
    return topic_scores, float(np.mean(topic_scores)) if topic_scores else 0.0


def topic_diversity(top_indices: Sequence[Sequence[int]]) -> float:
    """计算 top words 中不同词所占比例。"""

    total = sum(len(topic) for topic in top_indices)
    unique = len({index for topic in top_indices for index in topic})
    return cleaner.ratio(unique, total)


def matched_topic_jaccard(
    first: Sequence[set[str]], second: Sequence[set[str]]
) -> tuple[float, list[tuple[int, int, float]]]:
    """用匈牙利匹配计算两次模型的主题词 Jaccard 稳定性。"""

    if len(first) != len(second):
        raise cleaner.CleaningError("稳定性比较的主题数不一致")
    size = len(first)
    similarity = np.zeros((size, size), dtype=np.float64)
    for first_index, first_words in enumerate(first):
        for second_index, second_words in enumerate(second):
            union = first_words | second_words
            similarity[first_index, second_index] = (
                len(first_words & second_words) / len(union) if union else 1.0
            )
    first_rows, second_columns = linear_sum_assignment(-similarity)
    matches = [
        (int(first_index), int(second_index), float(similarity[first_index, second_index]))
        for first_index, second_index in zip(first_rows, second_columns)
    ]
    return float(np.mean([value for _first, _second, value in matches])), matches


def representative_rows(
    document_topics: np.ndarray,
    artifact: CorpusArtifact,
    records: Sequence[cleaner.PoemRecord],
    per_topic: int,
) -> list[dict[str, Any]]:
    """提取每个主题后验概率最高的代表诗。"""

    rows: list[dict[str, Any]] = []
    for topic_index in range(document_topics.shape[1]):
        scores = document_topics[:, topic_index]
        count = min(per_topic, len(scores))
        candidates = np.argpartition(scores, -count)[-count:]
        ordered = sorted(candidates.tolist(), key=lambda index: (-scores[index], index))
        for rank, matrix_row in enumerate(ordered, 1):
            record_index = artifact.valid_record_indices[matrix_row]
            record = records[record_index]
            rows.append(
                {
                    "topic_id": topic_index + 1,
                    "rank": rank,
                    "matrix_row": matrix_row,
                    "corpus_row": record_index,
                    "poem_id": record.poem_id,
                    "author": record.author,
                    "title": record.title,
                    "topic_probability": f"{scores[matrix_row]:.10f}",
                    "original_text": record.original_text,
                }
            )
    return rows


def model_signature(
    artifact: CorpusArtifact,
    topic_count: int,
    seed: int,
    lda_config: Mapping[str, Any],
    metric_config: Mapping[str, Any],
) -> str:
    """生成模型检查点签名。"""

    return json_sha256(
        {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "matrix_fingerprint": artifact.matrix_fingerprint,
            "vocabulary": json_sha256(artifact.vocabulary),
            "topic_count": topic_count,
            "seed": seed,
            "lda": dict(lda_config),
            "metrics": dict(metric_config),
        }
    )


def checkpoint_is_valid(
    metrics_path: Path,
    topics_path: Path,
    representatives_path: Path,
    expected_signature: str,
) -> bool:
    """检查一个模型是否已有完整且匹配的结果。"""

    if not all(path.is_file() for path in (metrics_path, topics_path, representatives_path)):
        return False
    try:
        value = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return False
    if value.get("checkpoint_signature") != expected_signature:
        return False
    outputs = value.get("outputs")
    if not isinstance(outputs, Mapping):
        return False
    for output_name, output_path in (
        ("topics", topics_path),
        ("representative_poems", representatives_path),
    ):
        output_metadata = outputs.get(output_name)
        if not isinstance(output_metadata, Mapping):
            return False
        expected_hash = output_metadata.get("sha256")
        if not isinstance(expected_hash, str):
            return False
        try:
            if cleaner.file_sha256(output_path) != expected_hash:
                return False
        except OSError:
            return False
    return True


def fit_one_model(
    *,
    artifact: CorpusArtifact,
    records: Sequence[cleaner.PoemRecord],
    topic_count: int,
    seed: int,
    lda_config: Mapping[str, Any],
    metric_config: Mapping[str, Any],
    model_root: Path,
    resume: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    """拟合一个模型，写出主题、代表诗和可恢复检查点。"""

    model_dir = model_root / artifact.label / f"K{topic_count}" / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    topics_path = model_dir / "topics.csv"
    representatives_path = model_dir / "representative_poems.csv"
    metrics_path = model_dir / "metrics.json"
    signature = model_signature(
        artifact, topic_count, seed, lda_config, metric_config
    )
    if resume and checkpoint_is_valid(
        metrics_path, topics_path, representatives_path, signature
    ):
        logger.info(
            "跳过已有检查点：corpus=%s K=%d seed=%d",
            artifact.label,
            topic_count,
            seed,
        )
        return json.loads(metrics_path.read_text(encoding="utf-8-sig"))

    logger.info(
        "开始模型：corpus=%s K=%d seed=%d docs=%d vocab=%d",
        artifact.label,
        topic_count,
        seed,
        artifact.matrix.shape[0],
        artifact.matrix.shape[1],
    )
    started = time.perf_counter()
    model = LatentDirichletAllocation(
        n_components=topic_count,
        doc_topic_prior=lda_config.get("doc_topic_prior"),
        topic_word_prior=lda_config.get("topic_word_prior"),
        learning_method=str(lda_config["learning_method"]),
        max_iter=int(lda_config["max_iter"]),
        evaluate_every=int(lda_config["evaluate_every"]),
        perp_tol=float(lda_config["perp_tol"]),
        mean_change_tol=float(lda_config["mean_change_tol"]),
        max_doc_update_iter=int(lda_config["max_doc_update_iter"]),
        random_state=seed,
        n_jobs=int(lda_config["n_jobs"]),
        verbose=int(lda_config.get("verbose", 0)),
    )
    document_topics = model.fit_transform(artifact.matrix)
    fit_seconds = time.perf_counter() - started
    topic_word = model.components_ / model.components_.sum(axis=1, keepdims=True)
    top_word_count = int(metric_config["top_words"])
    top_indices = topic_top_indices(topic_word, top_word_count)
    topic_npmi, mean_npmi = topic_npmi_scores(artifact.matrix, top_indices)
    diversity = topic_diversity(top_indices)
    prevalence = document_topics.mean(axis=0)

    topic_rows: list[dict[str, Any]] = []
    for topic_index, indices in enumerate(top_indices):
        for rank, term_index in enumerate(indices, 1):
            topic_rows.append(
                {
                    "topic_id": topic_index + 1,
                    "rank": rank,
                    "term_index": term_index,
                    "token": artifact.vocabulary[term_index],
                    "probability": f"{topic_word[topic_index, term_index]:.12f}",
                    "component_weight": f"{model.components_[topic_index, term_index]:.8f}",
                    "topic_npmi": f"{topic_npmi[topic_index]:.10f}",
                    "topic_prevalence": f"{prevalence[topic_index]:.10f}",
                }
            )
    cleaner.write_csv(
        topics_path,
        [
            "topic_id",
            "rank",
            "term_index",
            "token",
            "probability",
            "component_weight",
            "topic_npmi",
            "topic_prevalence",
        ],
        topic_rows,
    )
    representatives = representative_rows(
        document_topics,
        artifact,
        records,
        int(metric_config["representative_poems_per_topic"]),
    )
    cleaner.write_csv(
        representatives_path,
        [
            "topic_id",
            "rank",
            "matrix_row",
            "corpus_row",
            "poem_id",
            "author",
            "title",
            "topic_probability",
            "original_text",
        ],
        representatives,
    )
    metrics = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "checkpoint_signature": signature,
        "corpus": artifact.label,
        "min_tf": artifact.min_tf,
        "min_df": artifact.min_df,
        "topic_count": topic_count,
        "random_seed": seed,
        "document_count": artifact.matrix.shape[0],
        "vocabulary_size": artifact.matrix.shape[1],
        "matrix_nnz": int(artifact.matrix.nnz),
        "max_iter": int(lda_config["max_iter"]),
        "iterations_run": int(model.n_iter_),
        "learning_method": str(lda_config["learning_method"]),
        "fit_seconds": round(fit_seconds, 6),
        "training_perplexity": float(model.bound_),
        "coherence_metric": "document_npmi",
        "mean_topic_npmi": mean_npmi,
        "min_topic_npmi": min(topic_npmi),
        "max_topic_npmi": max(topic_npmi),
        "topic_diversity": diversity,
        "top_words": top_word_count,
        "outputs": {
            "topics": {
                "path": cleaner.display_path(topics_path),
                "sha256": cleaner.file_sha256(topics_path),
            },
            "representative_poems": {
                "path": cleaner.display_path(representatives_path),
                "sha256": cleaner.file_sha256(representatives_path),
            },
        },
    }
    cleaner.write_json(metrics_path, metrics)
    logger.info(
        "完成模型：corpus=%s K=%d seed=%d time=%.2fs NPMI=%.6f diversity=%.6f perplexity=%.3f",
        artifact.label,
        topic_count,
        seed,
        fit_seconds,
        mean_npmi,
        diversity,
        model.bound_,
    )
    return metrics


def load_topic_sets(path: Path) -> list[set[str]]:
    """从主题词表读取每个主题的 top-word 集合。"""

    grouped: dict[int, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[int(row["topic_id"])].add(row["token"])
    return [grouped[index] for index in sorted(grouped)]


def aggregate_results(
    artifacts: Sequence[CorpusArtifact],
    topic_counts: Sequence[int],
    seeds: Sequence[int],
    model_root: Path,
    reports_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """汇总模型指标并计算跨种子稳定性。"""

    model_rows: list[dict[str, Any]] = []
    topic_rows: list[dict[str, Any]] = []
    representative_rows_all: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        for topic_count in topic_counts:
            topic_sets_by_seed: dict[int, list[set[str]]] = {}
            for seed in seeds:
                model_dir = model_root / artifact.label / f"K{topic_count}" / f"seed_{seed}"
                metrics = json.loads(
                    (model_dir / "metrics.json").read_text(encoding="utf-8-sig")
                )
                model_rows.append(
                    {
                        "corpus": artifact.label,
                        "min_tf": artifact.min_tf,
                        "min_df": artifact.min_df,
                        "K": topic_count,
                        "seed": seed,
                        "documents": metrics["document_count"],
                        "vocabulary_size": metrics["vocabulary_size"],
                        "iterations_run": metrics["iterations_run"],
                        "fit_seconds": f"{float(metrics['fit_seconds']):.6f}",
                        "training_perplexity": f"{float(metrics['training_perplexity']):.10f}",
                        "mean_topic_npmi": f"{float(metrics['mean_topic_npmi']):.10f}",
                        "min_topic_npmi": f"{float(metrics['min_topic_npmi']):.10f}",
                        "max_topic_npmi": f"{float(metrics['max_topic_npmi']):.10f}",
                        "topic_diversity": f"{float(metrics['topic_diversity']):.10f}",
                    }
                )
                topics_path = model_dir / "topics.csv"
                topic_sets_by_seed[seed] = load_topic_sets(topics_path)
                with topics_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    for row in csv.DictReader(handle):
                        topic_rows.append(
                            {
                                "corpus": artifact.label,
                                "min_tf": artifact.min_tf,
                                "min_df": artifact.min_df,
                                "K": topic_count,
                                "seed": seed,
                                **row,
                            }
                        )
                with (model_dir / "representative_poems.csv").open(
                    "r", encoding="utf-8-sig", newline=""
                ) as handle:
                    for row in csv.DictReader(handle):
                        representative_rows_all.append(
                            {
                                "corpus": artifact.label,
                                "min_tf": artifact.min_tf,
                                "min_df": artifact.min_df,
                                "K": topic_count,
                                "seed": seed,
                                **row,
                            }
                        )
            for first_seed, second_seed in itertools.combinations(seeds, 2):
                score, matches = matched_topic_jaccard(
                    topic_sets_by_seed[first_seed], topic_sets_by_seed[second_seed]
                )
                stability_rows.append(
                    {
                        "corpus": artifact.label,
                        "min_tf": artifact.min_tf,
                        "min_df": artifact.min_df,
                        "K": topic_count,
                        "seed_a": first_seed,
                        "seed_b": second_seed,
                        "matched_topic_jaccard": f"{score:.10f}",
                        "matched_topic_count": len(matches),
                        "min_matched_topic_jaccard": f"{min(value for _a, _b, value in matches):.10f}",
                        "max_matched_topic_jaccard": f"{max(value for _a, _b, value in matches):.10f}",
                    }
                )

    cleaner.write_csv(
        reports_root / "model_metrics.csv",
        list(model_rows[0].keys()),
        model_rows,
    )
    cleaner.write_csv(
        reports_root / "stability_metrics.csv",
        list(stability_rows[0].keys()),
        stability_rows,
    )
    cleaner.write_csv(
        reports_root / "topic_keywords.csv",
        list(topic_rows[0].keys()),
        topic_rows,
    )
    cleaner.write_csv(
        reports_root / "representative_poems.csv",
        list(representative_rows_all[0].keys()),
        representative_rows_all,
    )

    summary_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        for topic_count in topic_counts:
            models = [
                row
                for row in model_rows
                if row["corpus"] == artifact.label and int(row["K"]) == topic_count
            ]
            stabilities = [
                float(row["matched_topic_jaccard"])
                for row in stability_rows
                if row["corpus"] == artifact.label and int(row["K"]) == topic_count
            ]
            npmi_values = [float(row["mean_topic_npmi"]) for row in models]
            diversity_values = [float(row["topic_diversity"]) for row in models]
            perplexity_values = [float(row["training_perplexity"]) for row in models]
            fit_values = [float(row["fit_seconds"]) for row in models]
            summary_rows.append(
                {
                    "corpus": artifact.label,
                    "min_tf": artifact.min_tf,
                    "min_df": artifact.min_df,
                    "K": topic_count,
                    "seed_count": len(models),
                    "vocabulary_size": artifact.statistics["vocabulary_size"],
                    "valid_poems": artifact.statistics["valid_poem_count"],
                    "mean_npmi": f"{np.mean(npmi_values):.10f}",
                    "std_npmi": f"{np.std(npmi_values):.10f}",
                    "mean_topic_diversity": f"{np.mean(diversity_values):.10f}",
                    "std_topic_diversity": f"{np.std(diversity_values):.10f}",
                    "mean_topic_stability": f"{np.mean(stabilities):.10f}",
                    "min_seed_pair_stability": f"{min(stabilities):.10f}",
                    "max_seed_pair_stability": f"{max(stabilities):.10f}",
                    "mean_training_perplexity": f"{np.mean(perplexity_values):.10f}",
                    "mean_fit_seconds": f"{np.mean(fit_values):.6f}",
                }
            )
    cleaner.write_csv(
        reports_root / "configuration_summary.csv",
        list(summary_rows[0].keys()),
        summary_rows,
    )
    return model_rows, stability_rows, summary_rows


def build_report(
    artifacts: Sequence[CorpusArtifact],
    summary_rows: Sequence[Mapping[str, Any]],
    invocation_runtime: float,
    model_fit_seconds_total: float,
    topic_counts: Sequence[int],
    seeds: Sequence[int],
    lda_config: Mapping[str, Any],
) -> str:
    """生成第一轮 LDA 敏感性实验报告。"""

    corpus_table = [
        (
            artifact.label,
            artifact.min_tf,
            artifact.min_df,
            f"{artifact.statistics['vocabulary_size']:,}",
            f"{artifact.statistics['total_tokens']:,}",
            f"{artifact.statistics['valid_poem_count']:,}",
            artifact.statistics["empty_poem_count"],
        )
        for artifact in artifacts
    ]
    metric_table = [
        (
            row["corpus"],
            row["K"],
            f"{float(row['mean_npmi']):.5f}",
            f"{float(row['mean_topic_diversity']):.5f}",
            f"{float(row['mean_topic_stability']):.5f}",
            f"{float(row['mean_training_perplexity']):.2f}",
            f"{float(row['mean_fit_seconds']):.1f}",
        )
        for row in summary_rows
    ]
    descriptive_best = []
    for artifact in artifacts:
        rows = [row for row in summary_rows if row["corpus"] == artifact.label]
        best_npmi = max(rows, key=lambda row: float(row["mean_npmi"]))
        best_stability = max(rows, key=lambda row: float(row["mean_topic_stability"]))
        best_diversity = max(rows, key=lambda row: float(row["mean_topic_diversity"]))
        descriptive_best.append(
            (
                artifact.label,
                f"K={best_npmi['K']} ({float(best_npmi['mean_npmi']):.5f})",
                f"K={best_diversity['K']} ({float(best_diversity['mean_topic_diversity']):.5f})",
                f"K={best_stability['K']} ({float(best_stability['mean_topic_stability']):.5f})",
            )
        )
    lines = [
        "# 《全唐诗》TF×DF 第一轮 LDA 敏感性实验报告",
        "",
        "## Experimental Design",
        "",
        "本轮比较 A=(TF≥1,DF≥1)、B=(2,2)、C=(3,3)、D=(5,3)。四组共享相同分词、词形规范、核心停用词、保护词、非汉字规则和 token 顺序；保护词继续豁免最低 TF/DF。",
        "",
        f"每组分别运行 K={list(topic_counts)}，随机种子={list(seeds)}，共 {len(artifacts) * len(topic_counts) * len(seeds)} 个模型。"
        f"模型为 scikit-learn Batch LDA，`max_iter={lda_config['max_iter']}`、`n_jobs={lda_config['n_jobs']}`。",
        "",
        "空 token 诗保留在 CSV/TXT 中，但无法形成词袋，故不进入 LDA 矩阵；排除清单按组保存。",
        "",
        "## Corpora",
        "",
        cleaner.markdown_table(
            ["Corpus", "min TF", "min DF", "Vocabulary", "Tokens", "Valid poems", "Empty"],
            corpus_table,
        ),
        "",
        "## Model Metrics",
        "",
        cleaner.markdown_table(
            ["Corpus", "K", "Mean NPMI", "Topic diversity", "Topic stability", "Train perplexity", "Mean fit sec"],
            metric_table,
        ),
        "",
        "NPMI 以诗歌文档共现计算；Topic Diversity 为各主题 top-20 词的去重比例；Topic Stability 为三个种子两两之间、经匈牙利匹配后的 top-20 Jaccard 均值。",
        "",
        "## Descriptive Maxima",
        "",
        cleaner.markdown_table(
            ["Corpus", "Highest mean NPMI", "Highest diversity", "Highest stability"],
            descriptive_best,
        ),
        "",
        "这些只是各单项指标的描述性最大值，不构成最终 K 或语料选择。",
        "",
        "## Outputs",
        "",
        "- `reports/lda_sensitivity/model_metrics.csv`：60 个模型逐一指标",
        "- `reports/lda_sensitivity/configuration_summary.csv`：20 个 Corpus×K 汇总",
        "- `reports/lda_sensitivity/stability_metrics.csv`：种子两两稳定性",
        "- `reports/lda_sensitivity/topic_keywords.csv`：全部主题 top-20 词",
        "- `reports/lda_sensitivity/representative_poems.csv`：每主题 top-5 代表诗",
        "",
        "## Limitations",
        "",
        "当前环境没有 gensim，本轮按确认方案计算 document NPMI，不把它称作 C_v。主题词质量、代表诗合理性和 Human Interpretability 仍需人工审核。训练 perplexity 只可在同一语料/词典条件下辅助比较，不宜跨不同 vocabulary 直接判优。",
        "",
        f"60 个模型拟合时间合计：{model_fit_seconds_total:.3f} 秒；"
        f"本次命令墙钟时间（含语料重建、断点检查与汇总）：{invocation_runtime:.3f} 秒。",
        "",
    ]
    return "\n".join(lines)


def run_experiment(
    config_path: Path = DEFAULT_CONFIG,
    project_root: Path = PROJECT_ROOT,
    prepare_only: bool = False,
    max_new_models: int | None = None,
) -> dict[str, Any]:
    """生成四套语料并以检查点方式运行全部模型。"""

    started_clock = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    config = cleaner.load_config(config_path)
    corpus_definitions = cleaner.require_mapping(config.get("corpora"), "corpora")
    validate_corpus_definitions(corpus_definitions)
    base_config_path = cleaner.resolve_path(str(config["base_cleaning_config"]), project_root)
    base_config = cleaner.load_config(base_config_path)
    base_paths = cleaner.prepare_paths(base_config, project_root)
    experiment_paths = cleaner.require_mapping(config.get("paths"), "paths")
    corpus_output_root = cleaner.resolve_path(
        str(experiment_paths["corpus_output_root"]), project_root
    )
    model_output_root = cleaner.resolve_path(
        str(experiment_paths["model_output_root"]), project_root
    )
    reports_root = cleaner.resolve_path(str(experiment_paths["reports_root"]), project_root)
    log_path = cleaner.resolve_path(str(experiment_paths["log_path"]), project_root)
    for directory in (corpus_output_root, model_output_root, reports_root, log_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    logger = cleaner.setup_logger(log_path)
    logger.info("开始 TF×DF 语料与 LDA 实验：%s", config_path)

    lda_config = cleaner.require_mapping(config.get("lda"), "lda")
    metric_config = cleaner.require_mapping(config.get("metrics"), "metrics")
    execution_config = cleaner.require_mapping(config.get("execution"), "execution")
    topic_counts = [int(value) for value in lda_config["topic_counts"]]
    seeds = [int(value) for value in lda_config["random_seeds"]]
    if topic_counts != [20, 30, 40, 50, 60]:
        raise cleaner.CleaningError("K 必须为 20、30、40、50、60")
    if seeds != [42, 2026, 20260830]:
        raise cleaner.CleaningError("随机种子必须为 42、2026、20260830")
    if str(lda_config["learning_method"]) != "batch" or int(lda_config["max_iter"]) != 30:
        raise cleaner.CleaningError("本轮必须使用 Batch LDA 和 max_iter=30")

    cleaning_config = cleaner.require_mapping(base_config.get("cleaning"), "cleaning")
    profile_name = str(cleaning_config["profile"])
    token_separator = str(cleaning_config.get("token_separator", " "))
    automatic_only = bool(cleaning_config.get("apply_automatic_normalization_only", True))
    rules_document = json.loads(
        base_paths["filtering_rules"].read_text(encoding="utf-8-sig")
    )
    frozen_profile = cleaner.require_mapping(rules_document[profile_name], profile_name)
    stopword_tier = str(frozen_profile["stopword_tier"])
    stopword_path = base_paths["stopwords"][stopword_tier]
    automatic_max_df = frozen_profile["automatic_max_df"]

    input_paths = [
        Path(__file__).resolve(),
        (SRC_DIR / "clean_corpus.py").resolve(),
        config_path,
        base_config_path,
        base_paths["corpus"],
        base_paths["tokenized_corpus"],
        base_paths["normalization"],
        stopword_path,
        base_paths["protected_words"],
        base_paths["filtering_rules"],
        base_paths["semantic_mapping"],
    ]
    input_hashes_before = {str(path): cleaner.file_sha256(path) for path in input_paths}
    existing_tf5_paths = {
        "lda_corpus": base_paths["output_dir"] / "lda_corpus.csv",
        "lda_tokens": base_paths["output_dir"] / "lda_tokens.txt",
    }
    existing_tf5_hashes_before = {
        name: cleaner.file_sha256(path) for name, path in existing_tf5_paths.items()
    }

    corpus_rows = cleaner.read_jsonl(base_paths["corpus"])
    token_rows = cleaner.read_jsonl(base_paths["tokenized_corpus"])
    records, data_issues = cleaner.align_records(corpus_rows, token_rows)
    normalization_rules = cleaner.load_normalization_rules(base_paths["normalization"])
    normalization_events = []
    for record in records:
        normalized, events = cleaner.apply_normalization(
            record.poem_id, record.original_tokens, normalization_rules, automatic_only
        )
        record.normalized_tokens = normalized
        normalization_events.extend(events)
    stopwords = set(cleaner.read_terms(stopword_path))
    protected_rows = cleaner.read_tsv(base_paths["protected_words"])
    protected_words = {row["surface"].strip() for row in protected_rows}
    token_tf: Counter[str] = Counter()
    token_df: Counter[str] = Counter()
    for record in records:
        token_tf.update(record.normalized_tokens)
        token_df.update(set(record.normalized_tokens))
    effective_poem_count = sum(
        any(
            token in protected_words
            or (cleaner.is_han_token(token) and token not in stopwords)
            for token in record.normalized_tokens
        )
        for record in records
    )

    artifacts: list[CorpusArtifact] = []
    for label in REQUIRED_CORPORA:
        definition = cleaner.require_mapping(corpus_definitions[label], f"corpora.{label}")
        artifact = build_corpus_artifact(
            label=label,
            min_tf=int(definition["min_tf"]),
            min_df=int(definition["min_df"]),
            records=records,
            token_tf=token_tf,
            token_df=token_df,
            effective_poem_count=effective_poem_count,
            protected_words=protected_words,
            stopwords=stopwords,
            automatic_max_df=automatic_max_df,
            token_separator=token_separator,
            output_root=corpus_output_root,
        )
        artifacts.append(artifact)
        logger.info(
            "语料 %s 完成：TF>=%d DF>=%d vocab=%d tokens=%d valid=%d empty=%d matrix=%s nnz=%d",
            label,
            artifact.min_tf,
            artifact.min_df,
            artifact.statistics["vocabulary_size"],
            artifact.statistics["total_tokens"],
            artifact.statistics["valid_poem_count"],
            artifact.statistics["empty_poem_count"],
            artifact.matrix.shape,
            artifact.matrix.nnz,
        )

    for previous, current in zip(artifacts, artifacts[1:]):
        if not set(current.vocabulary) <= set(previous.vocabulary):
            raise cleaner.CleaningError(f"语料 {current.label} 不是 {previous.label} 的词表子集")
        for record in records:
            previous_tokens = [
                token for token in record.normalized_tokens if previous.decisions[token][0]
            ]
            current_tokens = [
                token for token in record.normalized_tokens if current.decisions[token][0]
            ]
            iterator = iter(previous_tokens)
            if not all(any(candidate == token for candidate in iterator) for token in current_tokens):
                raise cleaner.CleaningError(
                    f"语料 {current.label} 相对 {previous.label} 出现 token 次序异常"
                )
    present_protected = protected_words & token_tf.keys()
    if not all(
        artifact.decisions[token][0]
        for artifact in artifacts
        for token in present_protected
    ):
        raise cleaner.CleaningError("至少一个保护 token 在 A/B/C/D 中被删除")
    if cleaner.file_sha256(artifacts[-1].tokens_path) != existing_tf5_hashes_before["lda_tokens"]:
        raise cleaner.CleaningError("D 语料与既有 TF5 token 输出不一致")

    corpus_comparison_rows = [
        {
            "corpus": artifact.label,
            "min_tf": artifact.min_tf,
            "min_df": artifact.min_df,
            "poems": artifact.statistics["poem_count"],
            "valid_poems": artifact.statistics["valid_poem_count"],
            "empty_poems": artifact.statistics["empty_poem_count"],
            "total_tokens": artifact.statistics["total_tokens"],
            "vocabulary_size": artifact.statistics["vocabulary_size"],
            "matrix_rows": artifact.matrix.shape[0],
            "matrix_columns": artifact.matrix.shape[1],
            "matrix_nnz": artifact.matrix.nnz,
            "protected_vocabulary_size": artifact.statistics["protected_vocabulary_size"],
            "protected_token_occurrences": artifact.statistics["protected_token_occurrences"],
        }
        for artifact in artifacts
    ]
    cleaner.write_csv(
        reports_root / "corpus_comparison.csv",
        list(corpus_comparison_rows[0].keys()),
        corpus_comparison_rows,
    )

    if prepare_only:
        logger.info("已按 --prepare-only 停止；未运行 LDA")
        return {
            "prepared": True,
            "corpora": corpus_comparison_rows,
            "lda_executed": False,
        }

    resume = bool(execution_config.get("resume", True))
    new_model_count = 0
    all_metrics: list[dict[str, Any]] = []
    stopped_early = False
    for artifact in artifacts:
        for topic_count in topic_counts:
            for seed in seeds:
                model_dir = (
                    model_output_root
                    / artifact.label
                    / f"K{topic_count}"
                    / f"seed_{seed}"
                )
                signature = model_signature(
                    artifact, topic_count, seed, lda_config, metric_config
                )
                already_complete = checkpoint_is_valid(
                    model_dir / "metrics.json",
                    model_dir / "topics.csv",
                    model_dir / "representative_poems.csv",
                    signature,
                )
                if (
                    max_new_models is not None
                    and new_model_count >= max_new_models
                    and not already_complete
                ):
                    stopped_early = True
                    break
                metrics = fit_one_model(
                    artifact=artifact,
                    records=records,
                    topic_count=topic_count,
                    seed=seed,
                    lda_config=lda_config,
                    metric_config=metric_config,
                    model_root=model_output_root,
                    resume=resume,
                    logger=logger,
                )
                all_metrics.append(metrics)
                if not already_complete:
                    new_model_count += 1
            if stopped_early:
                break
        if stopped_early:
            break
    if stopped_early:
        logger.info(
            "达到 --max-new-models=%d；检查点已保存，可续跑",
            max_new_models,
        )
        return {
            "prepared": True,
            "new_models_completed": new_model_count,
            "stopped_early": True,
        }

    expected_model_count = len(artifacts) * len(topic_counts) * len(seeds)
    if len(all_metrics) != expected_model_count:
        raise cleaner.CleaningError(
            f"模型结果数不完整：{len(all_metrics)} != {expected_model_count}"
        )
    model_rows, stability_rows, summary_rows = aggregate_results(
        artifacts,
        topic_counts,
        seeds,
        model_output_root,
        reports_root,
    )
    runtime_seconds = time.perf_counter() - started_clock
    model_fit_seconds_total = sum(float(row["fit_seconds"]) for row in model_rows)
    report = build_report(
        artifacts,
        summary_rows,
        runtime_seconds,
        model_fit_seconds_total,
        topic_counts,
        seeds,
        lda_config,
    )
    (reports_root / "lda_sensitivity_report.md").write_text(report, encoding="utf-8")

    input_hashes_after = {str(path): cleaner.file_sha256(path) for path in input_paths}
    existing_tf5_hashes_after = {
        name: cleaner.file_sha256(path) for name, path in existing_tf5_paths.items()
    }
    if input_hashes_before != input_hashes_after:
        raise cleaner.CleaningError("实验过程中输入文件发生变化")
    if existing_tf5_hashes_before != existing_tf5_hashes_after:
        raise cleaner.CleaningError("实验过程中既有 TF5 输出发生变化")

    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    output_files = sorted(
        [path for path in corpus_output_root.rglob("*") if path.is_file()]
        + [path for path in model_output_root.rglob("*") if path.is_file()]
        + [
            path
            for path in reports_root.rglob("*")
            if path.is_file() and path.name != "run_metadata.json"
        ],
        key=lambda path: str(path),
    )
    metadata = {
        "schema_version": 1,
        "method": "tf_df_corpus_and_first_round_lda_sensitivity",
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime_seconds": round(runtime_seconds, 6),
        "runtime_scope": "current_invocation_including_corpus_build_checkpoint_scan_and_aggregation",
        "model_fit_seconds_total": round(model_fit_seconds_total, 6),
        "models_fitted_this_invocation": new_model_count,
        "models_reused_from_checkpoints": expected_model_count - new_model_count,
        "inputs": [
            {
                "path": cleaner.display_path(path),
                "sha256": input_hashes_before[str(path)],
            }
            for path in input_paths
        ],
        "corpus_definitions": {
            artifact.label: {
                "min_tf": artifact.min_tf,
                "min_df": artifact.min_df,
                **artifact.statistics,
                "matrix_fingerprint": artifact.matrix_fingerprint,
            }
            for artifact in artifacts
        },
        "lda": dict(lda_config),
        "metrics": dict(metric_config),
        "model_count": len(model_rows),
        "configuration_count": len(summary_rows),
        "stability_pair_count": len(stability_rows),
        "environment": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "validation": {
            "input_hashes_unchanged": input_hashes_before == input_hashes_after,
            "existing_tf5_hashes_unchanged": existing_tf5_hashes_before
            == existing_tf5_hashes_after,
            "D_tokens_match_existing_tf5": cleaner.file_sha256(
                artifacts[-1].tokens_path
            )
            == existing_tf5_hashes_before["lda_tokens"],
            "corpora_nested_and_order_preserved": True,
            "protected_tokens_retained_in_all_corpora": True,
            "empty_poems_excluded_only_from_lda_matrix": True,
            "semantic_mapping_applied": False,
            "model_count_complete": len(model_rows) == 60,
        },
        "outputs": [
            {
                "path": cleaner.display_path(path),
                "sha256": cleaner.file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in output_files
        ],
    }
    cleaner.write_json(reports_root / "run_metadata.json", metadata)
    logger.info(
        "全部完成：models=%d configurations=%d runtime=%.2fs",
        len(model_rows),
        len(summary_rows),
        runtime_seconds,
    )
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="生成 A/B/C/D TF×DF 语料并运行第一轮 LDA 敏感性实验"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--max-new-models",
        type=int,
        default=None,
        help="仅新增指定数量模型后停止，用于耗时试跑；下次可续跑",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""

    args = parse_args(argv)
    try:
        run_experiment(
            args.config.resolve(),
            prepare_only=args.prepare_only,
            max_new_models=args.max_new_models,
        )
    except Exception as exc:
        logging.getLogger("clean_corpus").exception("TF×DF LDA 实验失败：%s", exc)
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
