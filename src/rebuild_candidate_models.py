"""按第一轮原参数重建六个候选 LDA，并持久化完整模型与概率矩阵。"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import numpy as np
import scipy
import sklearn
from scipy.sparse import csr_matrix, load_npz
from sklearn.decomposition import LatentDirichletAllocation

import clean_corpus as cleaner
import run_tf_df_lda as original


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "candidate_rebuild_config.yaml"
ARTIFACT_SCHEMA_VERSION = 1
TOPIC_FIELDS = [
    "topic_id",
    "rank",
    "term_index",
    "token",
    "probability",
    "component_weight",
    "topic_npmi",
    "topic_prevalence",
]
REPRESENTATIVE_FIELDS = [
    "topic_id",
    "rank",
    "matrix_row",
    "corpus_row",
    "poem_id",
    "author",
    "title",
    "topic_probability",
    "original_text",
]


@dataclass
class ExistingCorpus:
    """保存只读语料矩阵及其行列对齐信息。"""

    label: str
    min_tf: int
    min_df: int
    directory: Path
    matrix: csr_matrix
    matrix_fingerprint: str
    vocabulary: list[str]
    vocabulary_rows: list[dict[str, str]]
    document_rows: list[dict[str, str]]
    corpus_rows: list[dict[str, str]]
    metadata: dict[str, Any]


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 Excel 兼容 UTF-8 CSV。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_path(project_root: Path, value: Any, name: str) -> Path:
    """解析并验证配置路径。"""

    if not isinstance(value, str) or not value.strip():
        raise cleaner.CleaningError(f"{name} 必须为非空路径")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def setup_logger(path: Path) -> logging.Logger:
    """建立重建过程日志。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("candidate_model_rebuild")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def environment_info() -> dict[str, Any]:
    """记录影响数值复现的核心运行环境。"""

    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def validate_environment(
    current: Mapping[str, Any], original_environment: Mapping[str, Any]
) -> dict[str, Any]:
    """逐项核对第一轮与本轮的数值环境。"""

    keys = (
        "python",
        "implementation",
        "platform",
        "logical_cpu_count",
        "numpy",
        "scipy",
        "scikit_learn",
    )
    comparisons = {
        key: {
            "original": original_environment.get(key),
            "current": current.get(key),
            "match": original_environment.get(key) == current.get(key),
        }
        for key in keys
    }
    return {
        "all_match": all(item["match"] for item in comparisons.values()),
        "fields": comparisons,
    }


def load_existing_corpus(
    corpus_root: Path,
    label: str,
    min_tf: int,
    min_df: int,
) -> ExistingCorpus:
    """加载并严格验证第一轮已冻结的语料、词表和 BoW。"""

    directory = corpus_root / label
    metadata_path = directory / "corpus_metadata.json"
    matrix_path = directory / "bow_matrix.npz"
    vocabulary_path = directory / "vocabulary.csv"
    document_path = directory / "document_index.csv"
    corpus_path = directory / "lda_corpus.csv"
    required = [metadata_path, matrix_path, vocabulary_path, document_path, corpus_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise cleaner.CleaningError(f"{label} 语料输入缺失：{missing}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if (int(metadata["min_tf"]), int(metadata["min_df"])) != (min_tf, min_df):
        raise cleaner.CleaningError(f"{label} TF/DF 与候选配置不一致")
    outputs = cleaner.require_mapping(metadata.get("outputs"), "corpus outputs")
    for name, path in (
        ("lda_corpus", corpus_path),
        ("vocabulary", vocabulary_path),
        ("bow_matrix_npz", matrix_path),
        ("document_index", document_path),
    ):
        if cleaner.file_sha256(path) != outputs.get(name):
            raise cleaner.CleaningError(f"{label} {name} 哈希与语料元数据不一致")

    matrix = load_npz(matrix_path).tocsr()
    matrix.sort_indices()
    fingerprint = original.matrix_fingerprint(matrix)
    if fingerprint != metadata.get("matrix_fingerprint"):
        raise cleaner.CleaningError(f"{label} BoW 逻辑指纹不一致")

    vocabulary_rows = read_csv(vocabulary_path)
    vocabulary_rows.sort(key=lambda row: int(row["term_index"]))
    indices = [int(row["term_index"]) for row in vocabulary_rows]
    if indices != list(range(len(vocabulary_rows))):
        raise cleaner.CleaningError(f"{label} vocabulary term_index 不连续")
    vocabulary = [row["token"] for row in vocabulary_rows]
    if len(set(vocabulary)) != len(vocabulary):
        raise cleaner.CleaningError(f"{label} vocabulary 存在重复 token")

    document_rows = read_csv(document_path)
    document_rows.sort(key=lambda row: int(row["matrix_row"]))
    matrix_rows = [int(row["matrix_row"]) for row in document_rows]
    if matrix_rows != list(range(len(document_rows))):
        raise cleaner.CleaningError(f"{label} document_index matrix_row 不连续")
    corpus_rows = read_csv(corpus_path)
    if matrix.shape != (len(document_rows), len(vocabulary)):
        raise cleaner.CleaningError(
            f"{label} matrix shape {matrix.shape} 与行列索引不一致"
        )
    for row in document_rows:
        corpus_row = int(row["corpus_row"])
        if corpus_row < 0 or corpus_row >= len(corpus_rows):
            raise cleaner.CleaningError(f"{label} corpus_row 越界")
        if corpus_rows[corpus_row]["poem_id"] != row["poem_id"]:
            raise cleaner.CleaningError(f"{label} document_index 与语料 poem_id 不一致")

    return ExistingCorpus(
        label=label,
        min_tf=min_tf,
        min_df=min_df,
        directory=directory,
        matrix=matrix,
        matrix_fingerprint=fingerprint,
        vocabulary=vocabulary,
        vocabulary_rows=vocabulary_rows,
        document_rows=document_rows,
        corpus_rows=corpus_rows,
        metadata=metadata,
    )


def original_model_signature(
    corpus: ExistingCorpus,
    topic_count: int,
    seed: int,
    lda_config: Mapping[str, Any],
    metric_config: Mapping[str, Any],
) -> str:
    """按第一轮完全相同的字段计算检查点签名。"""

    return original.json_sha256(
        {
            "model_schema_version": original.MODEL_SCHEMA_VERSION,
            "metric_schema_version": original.METRIC_SCHEMA_VERSION,
            "matrix_fingerprint": corpus.matrix_fingerprint,
            "vocabulary": original.json_sha256(corpus.vocabulary),
            "topic_count": topic_count,
            "seed": seed,
            "lda": dict(lda_config),
            "metrics": dict(metric_config),
        }
    )


def build_model(
    topic_count: int, seed: int, lda_config: Mapping[str, Any]
) -> LatentDirichletAllocation:
    """以第一轮原参数构造 sklearn Batch LDA。"""

    return LatentDirichletAllocation(
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


def topic_rows(
    model: LatentDirichletAllocation,
    topic_word: np.ndarray,
    document_topic: np.ndarray,
    corpus: ExistingCorpus,
    top_n: int,
    include_metrics: bool,
) -> tuple[list[dict[str, Any]], list[float], float, float]:
    """生成与第一轮格式兼容的主题词行。"""

    top_indices = original.topic_top_indices(topic_word, top_n)
    metric_indices = original.topic_top_indices(topic_word, 20)
    npmi_by_topic, mean_npmi = original.topic_npmi_scores(
        corpus.matrix, metric_indices
    )
    diversity = original.topic_diversity(metric_indices)
    prevalence = document_topic.mean(axis=0)
    rows: list[dict[str, Any]] = []
    for topic_index, indices in enumerate(top_indices):
        for rank, term_index in enumerate(indices, 1):
            row: dict[str, Any] = {
                "topic_id": topic_index + 1,
                "rank": rank,
                "term_index": term_index,
                "token": corpus.vocabulary[term_index],
                "probability": f"{topic_word[topic_index, term_index]:.12f}",
                "component_weight": f"{model.components_[topic_index, term_index]:.8f}",
            }
            if include_metrics:
                row["topic_npmi"] = f"{npmi_by_topic[topic_index]:.10f}"
                row["topic_prevalence"] = f"{prevalence[topic_index]:.10f}"
            rows.append(row)
    return rows, npmi_by_topic, mean_npmi, diversity


def representative_rows(
    document_topic: np.ndarray,
    corpus: ExistingCorpus,
    per_topic: int,
    include_cleaned_tokens: bool,
) -> list[dict[str, Any]]:
    """按 P(topic|poem) 稳定提取代表诗。"""

    rows: list[dict[str, Any]] = []
    for topic_index in range(document_topic.shape[1]):
        scores = document_topic[:, topic_index]
        count = min(per_topic, len(scores))
        candidates = np.argpartition(scores, -count)[-count:]
        ordered = sorted(candidates.tolist(), key=lambda index: (-scores[index], index))
        for rank, matrix_row in enumerate(ordered, 1):
            document = corpus.document_rows[matrix_row]
            corpus_row = int(document["corpus_row"])
            poem = corpus.corpus_rows[corpus_row]
            row: dict[str, Any] = {
                "topic_id": topic_index + 1,
                "rank": rank,
                "matrix_row": matrix_row,
                "corpus_row": corpus_row,
                "poem_id": poem["poem_id"],
                "author": poem["author"],
                "title": poem["title"],
                "topic_probability": f"{scores[matrix_row]:.10f}",
                "original_text": poem["original_text"],
            }
            if include_cleaned_tokens:
                row["cleaned_tokens"] = poem["cleaned_tokens"]
            rows.append(row)
    return rows


def compare_rows_exact(
    expected: Sequence[Mapping[str, str]],
    actual: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    label: str,
) -> None:
    """逐行逐字段验证格式化结果完全一致。"""

    if len(expected) != len(actual):
        raise cleaner.CleaningError(
            f"{label} 行数不一致：{len(actual)} != {len(expected)}"
        )
    for row_index, (expected_row, actual_row) in enumerate(zip(expected, actual), 1):
        for field in fields:
            if str(actual_row.get(field, "")) != str(expected_row.get(field, "")):
                raise cleaner.CleaningError(
                    f"{label} 第 {row_index} 行字段 {field} 不一致："
                    f"{actual_row.get(field)!r} != {expected_row.get(field)!r}"
                )


def matrix_checks(
    model: LatentDirichletAllocation,
    topic_word: np.ndarray,
    document_topic: np.ndarray,
    corpus: ExistingCorpus,
    topic_count: int,
) -> dict[str, Any]:
    """验证完整矩阵的形状、数值范围与归一化。"""

    expected_topic_shape = (topic_count, corpus.matrix.shape[1])
    expected_document_shape = (corpus.matrix.shape[0], topic_count)
    if model.components_.shape != expected_topic_shape:
        raise cleaner.CleaningError("components_ shape 不一致")
    if topic_word.shape != expected_topic_shape:
        raise cleaner.CleaningError("Topic×Word shape 不一致")
    if document_topic.shape != expected_document_shape:
        raise cleaner.CleaningError("Document×Topic shape 不一致")
    for name, matrix in (
        ("components", model.components_),
        ("topic_word", topic_word),
        ("document_topic", document_topic),
    ):
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise cleaner.CleaningError(f"{name} 含非有限值或负值")
    topic_error = float(np.max(np.abs(topic_word.sum(axis=1) - 1.0)))
    document_error = float(np.max(np.abs(document_topic.sum(axis=1) - 1.0)))
    if topic_error > 1e-12 or document_error > 1e-12:
        raise cleaner.CleaningError("概率矩阵行和超出 1e-12 容差")
    return {
        "components_shape": list(model.components_.shape),
        "topic_word_shape": list(topic_word.shape),
        "document_topic_shape": list(document_topic.shape),
        "topic_word_max_row_sum_error": topic_error,
        "document_topic_max_row_sum_error": document_error,
        "dtype": str(topic_word.dtype),
    }


def metrics_validation(
    model: LatentDirichletAllocation,
    original_metrics: Mapping[str, Any],
    mean_npmi: float,
    npmi_by_topic: Sequence[float],
    diversity: float,
    tolerance: float,
) -> dict[str, Any]:
    """核对第一轮已保存的模型级指标。"""

    actual = {
        "iterations_run": int(model.n_iter_),
        "training_perplexity": float(model.bound_),
        "mean_topic_npmi": float(mean_npmi),
        "min_topic_npmi": float(min(npmi_by_topic)),
        "max_topic_npmi": float(max(npmi_by_topic)),
        "topic_diversity": float(diversity),
    }
    comparisons: dict[str, Any] = {}
    for key, actual_value in actual.items():
        expected_value = original_metrics[key]
        if key == "iterations_run":
            passed = int(actual_value) == int(expected_value)
            difference = abs(int(actual_value) - int(expected_value))
        else:
            difference = abs(float(actual_value) - float(expected_value))
            passed = difference <= tolerance
        comparisons[key] = {
            "original": expected_value,
            "rebuilt": actual_value,
            "absolute_difference": difference,
            "passed": passed,
        }
    if not all(item["passed"] for item in comparisons.values()):
        raise cleaner.CleaningError(f"模型指标未通过复现容差：{comparisons}")
    return comparisons


def output_hash_rows(directory: Path, excluded: set[str]) -> list[dict[str, Any]]:
    """登记候选目录内全部稳定输出。"""

    return [
        {
            "path": path.name,
            "sha256": cleaner.file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name not in excluded
    ]


def checkpoint_valid(directory: Path, rebuild_signature: str) -> bool:
    """验证完整重建检查点及其全部文件哈希。"""

    manifest_path = directory / "artifact_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("rebuild_signature") != rebuild_signature:
        return False
    if not manifest.get("reproduction_validation_passed"):
        return False
    for output in manifest.get("outputs", []):
        path = directory / str(output.get("path", ""))
        if not path.is_file() or cleaner.file_sha256(path) != output.get("sha256"):
            return False
    return True


def rebuild_one(
    *,
    candidate: Mapping[str, Any],
    corpus: ExistingCorpus,
    lda_config: Mapping[str, Any],
    metric_config: Mapping[str, Any],
    original_model_root: Path,
    output_root: Path,
    rebuild_signature: str,
    tolerance: float,
    joblib_compress: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """重建一个候选；通过原结果复现验证后才写成正式检查点。"""

    model_id = str(candidate["model_id"])
    topic_count = int(candidate["K"])
    seed = int(candidate["selected_seed"])
    source_dir = original_model_root / corpus.label / f"K{topic_count}" / f"seed_{seed}"
    original_topics_path = source_dir / "topics.csv"
    original_representatives_path = source_dir / "representative_poems.csv"
    original_metrics_path = source_dir / "metrics.json"
    for path in (original_topics_path, original_representatives_path, original_metrics_path):
        if not path.is_file():
            raise cleaner.CleaningError(f"{model_id} 第一轮输出缺失：{path}")
    original_metrics = json.loads(original_metrics_path.read_text(encoding="utf-8-sig"))
    expected_signature = original_model_signature(
        corpus, topic_count, seed, lda_config, metric_config
    )
    if original_metrics.get("checkpoint_signature") != expected_signature:
        raise cleaner.CleaningError(f"{model_id} 第一轮检查点签名与当前输入不一致")

    logger.info(
        "开始精确重建 %s：corpus=%s K=%d seed=%d docs=%d vocab=%d",
        model_id,
        corpus.label,
        topic_count,
        seed,
        corpus.matrix.shape[0],
        corpus.matrix.shape[1],
    )
    started = time.perf_counter()
    model = build_model(topic_count, seed, lda_config)
    document_topic = model.fit_transform(corpus.matrix)
    fit_seconds = time.perf_counter() - started
    topic_word = model.components_ / model.components_.sum(axis=1, keepdims=True)
    checks = matrix_checks(model, topic_word, document_topic, corpus, topic_count)

    top20_rows, npmi_by_topic, mean_npmi, diversity = topic_rows(
        model, topic_word, document_topic, corpus, 20, True
    )
    top30_rows, _npmi_unused, _mean_unused, _diversity_unused = topic_rows(
        model, topic_word, document_topic, corpus, 30, False
    )
    top5_rows = representative_rows(document_topic, corpus, 5, False)
    top20_representatives = representative_rows(document_topic, corpus, 20, True)
    compare_rows_exact(read_csv(original_topics_path), top20_rows, TOPIC_FIELDS, "topics")
    compare_rows_exact(
        read_csv(original_representatives_path),
        top5_rows,
        REPRESENTATIVE_FIELDS,
        "representative poems",
    )
    metric_comparisons = metrics_validation(
        model,
        original_metrics,
        mean_npmi,
        npmi_by_topic,
        diversity,
        tolerance,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{model_id}.in_progress.", dir=output_root))
    final = output_root / model_id
    if final.exists():
        raise cleaner.CleaningError(f"{model_id} 目标目录已存在但检查点无效：{final}")
    try:
        cleaner.write_csv(temporary / "reproduced_topics_top20.csv", TOPIC_FIELDS, top20_rows)
        cleaner.write_csv(
            temporary / "reproduced_representative_poems_top5.csv",
            REPRESENTATIVE_FIELDS,
            top5_rows,
        )
        topic30_fields = [
            "topic_id",
            "rank",
            "term_index",
            "token",
            "probability",
            "component_weight",
        ]
        cleaner.write_csv(temporary / "topics_top30.csv", topic30_fields, top30_rows)
        cleaner.write_csv(
            temporary / "representative_poems_top20.csv",
            REPRESENTATIVE_FIELDS + ["cleaned_tokens"],
            top20_representatives,
        )

        exact_topic_hash = cleaner.file_sha256(
            temporary / "reproduced_topics_top20.csv"
        ) == cleaner.file_sha256(original_topics_path)
        exact_representative_hash = cleaner.file_sha256(
            temporary / "reproduced_representative_poems_top5.csv"
        ) == cleaner.file_sha256(original_representatives_path)
        if not exact_topic_hash or not exact_representative_hash:
            raise cleaner.CleaningError(
                f"{model_id} 逐字段一致但完整 CSV 哈希未精确复现"
            )

        np.savez_compressed(
            temporary / "topic_word_components.npz", components=model.components_
        )
        np.savez_compressed(
            temporary / "topic_word_probabilities.npz", probabilities=topic_word
        )
        np.savez_compressed(
            temporary / "document_topic_probabilities.npz", probabilities=document_topic
        )
        joblib.dump(model, temporary / "model.joblib", compress=joblib_compress)
        shutil.copy2(corpus.directory / "vocabulary.csv", temporary / "vocabulary.csv")
        shutil.copy2(corpus.directory / "document_index.csv", temporary / "document_index.csv")

        loaded_components = np.load(temporary / "topic_word_components.npz")["components"]
        loaded_topic_word = np.load(temporary / "topic_word_probabilities.npz")["probabilities"]
        loaded_document_topic = np.load(
            temporary / "document_topic_probabilities.npz"
        )["probabilities"]
        loaded_model = joblib.load(temporary / "model.joblib")
        if not np.array_equal(loaded_components, model.components_):
            raise cleaner.CleaningError(f"{model_id} components NPZ 回读不一致")
        if not np.array_equal(loaded_topic_word, topic_word):
            raise cleaner.CleaningError(f"{model_id} Topic×Word NPZ 回读不一致")
        if not np.array_equal(loaded_document_topic, document_topic):
            raise cleaner.CleaningError(f"{model_id} Document×Topic NPZ 回读不一致")
        if not np.array_equal(loaded_model.components_, model.components_):
            raise cleaner.CleaningError(f"{model_id} model.joblib 回读不一致")

        matrix_metadata = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_id": model_id,
            "corpus": corpus.label,
            "K": topic_count,
            "selected_seed": seed,
            "topic_word_components": {
                "path": "topic_word_components.npz",
                "array_key": "components",
                "shape": list(model.components_.shape),
                "dtype": str(model.components_.dtype),
                "row_alignment": "topic_id = zero-based row + 1",
                "column_alignment": "vocabulary.csv term_index",
            },
            "topic_word_probabilities": {
                "path": "topic_word_probabilities.npz",
                "array_key": "probabilities",
                "shape": list(topic_word.shape),
                "dtype": str(topic_word.dtype),
                "normalization": "each topic row sums to 1",
            },
            "document_topic_probabilities": {
                "path": "document_topic_probabilities.npz",
                "array_key": "probabilities",
                "shape": list(document_topic.shape),
                "dtype": str(document_topic.dtype),
                "row_alignment": "document_index.csv matrix_row",
                "column_alignment": "topic_id = zero-based column + 1",
                "normalization": "each document row sums to 1",
            },
        }
        cleaner.write_json(temporary / "matrix_metadata.json", matrix_metadata)

        reproduction = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_id": model_id,
            "status": "passed",
            "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "original_checkpoint_signature": expected_signature,
            "original_topics_sha256": cleaner.file_sha256(original_topics_path),
            "rebuilt_topics_sha256": cleaner.file_sha256(
                temporary / "reproduced_topics_top20.csv"
            ),
            "exact_top20_csv_hash_match": exact_topic_hash,
            "original_representatives_sha256": cleaner.file_sha256(
                original_representatives_path
            ),
            "rebuilt_representatives_sha256": cleaner.file_sha256(
                temporary / "reproduced_representative_poems_top5.csv"
            ),
            "exact_top5_representatives_csv_hash_match": exact_representative_hash,
            "metric_absolute_tolerance": tolerance,
            "metric_comparisons": metric_comparisons,
            "matrix_checks": checks,
            "fit_seconds": round(fit_seconds, 6),
            "model_object_reload_components_exact": True,
            "npz_roundtrip_exact": True,
        }
        cleaner.write_json(temporary / "reproduction_validation.json", reproduction)
        outputs = output_hash_rows(temporary, {"artifact_manifest.json"})
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_id": model_id,
            "rebuild_signature": rebuild_signature,
            "reproduction_validation_passed": True,
            "original_checkpoint_signature": expected_signature,
            "corpus_matrix_fingerprint": corpus.matrix_fingerprint,
            "outputs": outputs,
        }
        cleaner.write_json(temporary / "artifact_manifest.json", manifest)
        temporary.replace(final)
    except Exception:
        logger.error("%s 重建未通过；诊断目录保留在 %s", model_id, temporary)
        raise

    logger.info(
        "%s 精确复现通过：fit=%.2fs topics_hash=exact reps_hash=exact",
        model_id,
        fit_seconds,
    )
    return {
        "model_id": model_id,
        "corpus": corpus.label,
        "K": topic_count,
        "selected_seed": seed,
        "status": "passed",
        "fit_seconds": f"{fit_seconds:.6f}",
        "iterations_run": int(model.n_iter_),
        "training_perplexity": f"{float(model.bound_):.10f}",
        "exact_top20_csv_hash_match": True,
        "exact_top5_representatives_csv_hash_match": True,
        "topic_word_rows": topic_word.shape[0],
        "topic_word_columns": topic_word.shape[1],
        "document_topic_rows": document_topic.shape[0],
        "document_topic_columns": document_topic.shape[1],
        "checkpoint_reused": False,
    }


def run(config_path: Path, project_root: Path) -> dict[str, Any]:
    """执行六候选精确重建并生成总审计。"""

    started_clock = time.perf_counter()
    config = cleaner.load_config(config_path)
    base_config_path = resolve_path(
        project_root, config.get("base_lda_config"), "base_lda_config"
    )
    selection_path = resolve_path(
        project_root, config.get("selection_summary"), "selection_summary"
    )
    paths = cleaner.require_mapping(config.get("paths"), "paths")
    corpus_root = resolve_path(project_root, paths.get("corpus_root"), "paths.corpus_root")
    original_model_root = resolve_path(
        project_root, paths.get("original_model_root"), "paths.original_model_root"
    )
    output_root = resolve_path(project_root, paths.get("output_root"), "paths.output_root")
    log_path = resolve_path(project_root, paths.get("log_path"), "paths.log_path")
    logger = setup_logger(log_path)
    base_config = cleaner.load_config(base_config_path)
    lda_config = cleaner.require_mapping(base_config.get("lda"), "lda")
    metric_config = cleaner.require_mapping(base_config.get("metrics"), "metrics")
    corpus_definitions = cleaner.require_mapping(base_config.get("corpora"), "corpora")
    candidates_value = config.get("candidates")
    if not isinstance(candidates_value, list) or len(candidates_value) != 6:
        raise cleaner.CleaningError("candidates 必须恰含六个模型")
    candidates = [cleaner.require_mapping(value, "candidate") for value in candidates_value]
    validation = cleaner.require_mapping(config.get("validation"), "validation")
    persistence = cleaner.require_mapping(config.get("persistence"), "persistence")
    tolerance = float(validation["metric_absolute_tolerance"])
    resume = bool(persistence.get("resume", True))
    joblib_compress = int(persistence.get("joblib_compress", 3))

    selection_rows = read_csv(selection_path)
    selection_by_id = {row["model_id"]: row for row in selection_rows}
    for candidate in candidates:
        model_id = str(candidate["model_id"])
        selection = selection_by_id.get(model_id)
        if selection is None or int(selection["selected_seed"]) != int(
            candidate["selected_seed"]
        ):
            raise cleaner.CleaningError(f"{model_id} selected seed 与审核选择表不一致")

    original_run_metadata_path = project_root / "reports" / "lda_sensitivity" / "run_metadata.json"
    original_run_metadata = json.loads(
        original_run_metadata_path.read_text(encoding="utf-8-sig")
    )
    current_environment = environment_info()
    environment_validation = validate_environment(
        current_environment,
        cleaner.require_mapping(original_run_metadata.get("environment"), "environment"),
    )
    if validation.get("require_environment_match", True) and not environment_validation[
        "all_match"
    ]:
        raise cleaner.CleaningError("当前数值环境与第一轮环境不一致")

    output_root.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    rebuild_signature_base = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "script_sha256": cleaner.file_sha256(script_path),
        "config_sha256": cleaner.file_sha256(config_path),
        "base_config_sha256": cleaner.file_sha256(base_config_path),
    }
    input_paths: set[Path] = {
        script_path,
        config_path,
        base_config_path,
        selection_path,
        original_run_metadata_path,
    }
    corpora: dict[str, ExistingCorpus] = {}
    for candidate in candidates:
        label = str(candidate["corpus"])
        definition = cleaner.require_mapping(corpus_definitions.get(label), f"corpora.{label}")
        if label not in corpora:
            corpora[label] = load_existing_corpus(
                corpus_root,
                label,
                int(definition["min_tf"]),
                int(definition["min_df"]),
            )
        corpus_dir = corpus_root / label
        input_paths.update(
            {
                corpus_dir / "corpus_metadata.json",
                corpus_dir / "bow_matrix.npz",
                corpus_dir / "vocabulary.csv",
                corpus_dir / "document_index.csv",
                corpus_dir / "lda_corpus.csv",
            }
        )
        source_dir = (
            original_model_root
            / label
            / f"K{int(candidate['K'])}"
            / f"seed_{int(candidate['selected_seed'])}"
        )
        input_paths.update(
            {
                source_dir / "topics.csv",
                source_dir / "representative_poems.csv",
                source_dir / "metrics.json",
            }
        )
    input_hashes_before = {str(path): cleaner.file_sha256(path) for path in input_paths}

    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        model_id = str(candidate["model_id"])
        corpus = corpora[str(candidate["corpus"])]
        original_signature = original_model_signature(
            corpus,
            int(candidate["K"]),
            int(candidate["selected_seed"]),
            lda_config,
            metric_config,
        )
        rebuild_signature = original.json_sha256(
            {**rebuild_signature_base, "model_id": model_id, "original_signature": original_signature}
        )
        final = output_root / model_id
        if resume and checkpoint_valid(final, rebuild_signature):
            reproduction = json.loads(
                (final / "reproduction_validation.json").read_text(encoding="utf-8-sig")
            )
            matrix_metadata = json.loads(
                (final / "matrix_metadata.json").read_text(encoding="utf-8-sig")
            )
            summaries.append(
                {
                    "model_id": model_id,
                    "corpus": candidate["corpus"],
                    "K": candidate["K"],
                    "selected_seed": candidate["selected_seed"],
                    "status": "passed",
                    "fit_seconds": reproduction["fit_seconds"],
                    "iterations_run": reproduction["metric_comparisons"]["iterations_run"]["rebuilt"],
                    "training_perplexity": f"{float(reproduction['metric_comparisons']['training_perplexity']['rebuilt']):.10f}",
                    "exact_top20_csv_hash_match": True,
                    "exact_top5_representatives_csv_hash_match": True,
                    "topic_word_rows": matrix_metadata["topic_word_probabilities"]["shape"][0],
                    "topic_word_columns": matrix_metadata["topic_word_probabilities"]["shape"][1],
                    "document_topic_rows": matrix_metadata["document_topic_probabilities"]["shape"][0],
                    "document_topic_columns": matrix_metadata["document_topic_probabilities"]["shape"][1],
                    "checkpoint_reused": True,
                }
            )
            logger.info("复用已验证检查点：%s", model_id)
            continue
        summaries.append(
            rebuild_one(
                candidate=candidate,
                corpus=corpus,
                lda_config=lda_config,
                metric_config=metric_config,
                original_model_root=original_model_root,
                output_root=output_root,
                rebuild_signature=rebuild_signature,
                tolerance=tolerance,
                joblib_compress=joblib_compress,
                logger=logger,
            )
        )

    input_hashes_after = {str(path): cleaner.file_sha256(path) for path in input_paths}
    if input_hashes_before != input_hashes_after:
        raise cleaner.CleaningError("重建期间第一轮模型或语料输入发生变化")
    cleaner.write_csv(output_root / "rebuild_summary.csv", list(summaries[0].keys()), summaries)
    output_files = sorted(
        [path for path in output_root.rglob("*") if path.is_file() and path.name != "run_metadata.json"],
        key=lambda path: str(path),
    )
    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "method": "exact_candidate_rebuild_from_original_parameters",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "runtime_seconds": round(time.perf_counter() - started_clock, 6),
        "candidate_count": len(summaries),
        "all_reproduction_validations_passed": all(row["status"] == "passed" for row in summaries),
        "source_inputs_unchanged": input_hashes_before == input_hashes_after,
        "environment": current_environment,
        "environment_validation": environment_validation,
        "original_lda_parameters": dict(lda_config),
        "original_metric_parameters": dict(metric_config),
        "inputs": [
            {"path": cleaner.display_path(path), "sha256": input_hashes_before[str(path)]}
            for path in sorted(input_paths, key=lambda item: str(item))
        ],
        "outputs": [
            {
                "path": cleaner.display_path(path),
                "sha256": cleaner.file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in output_files
        ],
    }
    cleaner.write_json(output_root / "run_metadata.json", metadata)
    logger.info(
        "六候选精确重建完成：passed=%d/%d runtime=%.2fs",
        sum(row["status"] == "passed" for row in summaries),
        len(summaries),
        metadata["runtime_seconds"],
    )
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行。"""

    parser = argparse.ArgumentParser(
        description="按第一轮原参数精确重建六个候选模型并保存完整矩阵"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""

    args = parse_args(argv)
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    run(config_path, PROJECT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
