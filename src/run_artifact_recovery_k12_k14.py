"""恢复 K12/K14 候选模型产物，并准备同口径全语料盲审模型。"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib.externals.loky import get_reusable_executor
from scipy import sparse
from scipy.optimize import linear_sum_assignment

import run_exhaustive_k_scan as scan
import run_lower_k_boundary_perplexity as lower
import run_quantitative_candidate_validation as qcv
import run_tf_df_lda as original


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "model_selection/artifact_recovery_K12_K14/config/artifact_recovery_config.yaml"
EXPECTED_K = [12, 14]
EXPECTED_SEEDS = [42, 2026, 20260830, 7, 17, 101, 31415, 271828, 20260907, 12345]
EXPECTED_REGIMES = ["ORIGINAL_PRIOR", "FIXED_PRIOR_K20"]
ACCEPTED_STATUSES = {"EXACT_REPRODUCTION", "PERMUTATION_EQUIVALENT", "NUMERICALLY_CLOSE"}


def resolve(path: str | Path) -> Path:
    """把相对路径解析到项目根目录。"""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def array_sha256(values: np.ndarray) -> str:
    """计算数组形状、类型与连续字节的稳定哈希。"""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """原子保存压缩 NPZ。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    """原子保存 NPY。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, values)
    temporary.replace(path)


def atomic_joblib(path: Path, value: Any, compress: int) -> None:
    """原子保存 joblib 模型对象。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.joblib")
    joblib.dump(value, temporary, compress=compress)
    temporary.replace(path)


def relative(path: Path) -> str:
    """返回适合审计表的项目相对路径。"""

    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """读取并严格验证恢复任务的预注册配置。"""

    config = scan.read_json(resolve(path))
    if list(config["topic_counts"]) != EXPECTED_K:
        raise ValueError("恢复 K 必须正好为 K12、K14")
    if list(config["seeds"]) != EXPECTED_SEEDS:
        raise ValueError("恢复 seed 集合或顺序不符")
    if list(config["prior_regimes"]) != EXPECTED_REGIMES:
        raise ValueError("恢复 prior regime 必须完整覆盖两套口径")
    expected = config["expected"]
    if int(expected["quantitative_run_count"]) != 40:
        raise ValueError("quantitative recovery 必须正好登记 40 个 run")
    if int(config["quantitative_fit"]["max_iter"]) != 30:
        raise ValueError("quantitative recovery 只能恢复 max_iter=30 主实验")
    full = config["full_corpus_fit"]
    if full["fit_scope"] != "FULL_CORPUS" or full["prior_regime"] != "ORIGINAL_PRIOR" or int(full["max_iter"]) != 30:
        raise ValueError("full-corpus 盲审模型必须使用 ORIGINAL_PRIOR、max_iter=30")
    acceptance = config["reproduction_acceptance"]
    if float(acceptance["mean_matched_cosine_min"]) != 0.9999:
        raise ValueError("matched cosine 接受阈值必须预注册为 0.9999")
    if float(acceptance["mean_matched_top20_jaccard_min"]) != 0.95:
        raise ValueError("Top20 Jaccard 接受阈值必须预注册为 0.95")
    allowed = (ROOT / "model_selection/artifact_recovery_K12_K14").resolve()
    if resolve(config["output_root"]).resolve() != allowed:
        raise ValueError("恢复输出目录越界")
    return config


def environment_info() -> dict[str, Any]:
    """记录当前恢复所用软件环境。"""

    return {
        "python": sys.version,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "executable": sys.executable,
    }


def compare_environment(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    """逐字段比较当前环境与原实验登记环境。"""

    fields = ["numpy", "scipy", "scikit_learn", "joblib", "executable"]
    differences = {
        field: {"recorded": recorded.get(field), "current": current.get(field)}
        for field in fields
        if recorded.get(field) != current.get(field)
    }
    recorded_python = str(recorded.get("python", "")).split()[0]
    if recorded_python and recorded_python != current.get("python_version"):
        differences["python_version"] = {"recorded": recorded_python, "current": current.get("python_version")}
    return {
        "status": "MATCHES_RECORDED_ENVIRONMENT" if not differences else "ENVIRONMENT_VERSION_DIFFERENCE",
        "differences": differences,
        "recorded": dict(recorded),
        "current": dict(current),
    }


def prepare(config_path: str | Path = DEFAULT_CONFIG, with_text: bool = True) -> dict[str, Any]:
    """只读装载冻结矩阵、词表、文档、split 与原实验参数。"""

    config_path = resolve(config_path).resolve()
    config = load_config(config_path)
    output = resolve(config["output_root"])
    for name in ["inventory", "config", "quantitative_validation_runs", "full_corpus_blind_models", "representative_poems", "data", "audit", "logs"]:
        (output / name).mkdir(parents=True, exist_ok=True)

    qcv_config_path = resolve(config["candidate_validation_config"])
    qcv_config = qcv.load_config(qcv_config_path)
    base_lda, audited_lda = scan.validate_lda_source(qcv_config)
    matrix, vocabulary, documents, corpus_metadata = scan.load_corpus(qcv_config)
    train, test, split_audit = scan.load_frozen_split(qcv_config, documents)
    expected = config["expected"]
    checks = {
        "matrix_file_sha256": scan.file_sha256(resolve(config["corpus_root"]) / "bow_matrix.npz"),
        "split_file_sha256": scan.file_sha256(resolve(qcv_config["frozen_split"])),
        "matrix_fingerprint": original.matrix_fingerprint(matrix),
        "vocabulary_sha256": scan.file_sha256(resolve(config["corpus_root"]) / "vocabulary.csv"),
        "fit_rows_sha256": split_audit["fit_rows_sha256"],
        "test_rows_sha256": split_audit["test_rows_sha256"],
    }
    mismatches = {key: (value, expected[key]) for key, value in checks.items() if value != expected[key]}
    if mismatches:
        raise ValueError(f"冻结矩阵、词表或 split 哈希不符：{mismatches}")
    shape = (int(expected["document_count"]), int(expected["vocabulary_size"]))
    if matrix.shape != shape or len(train) != int(expected["train_count"]) or len(test) != int(expected["test_count"]):
        raise ValueError("冻结矩阵或 split shape 不符")

    corpus_rows: list[dict[str, str]] | None = None
    if with_text:
        corpus_rows = scan.read_csv(resolve(config["corpus_root"]) / "lda_corpus.csv")
        if len(corpus_rows) < matrix.shape[0]:
            raise ValueError("C lda_corpus 行数少于非空 document_index，无法完成正文回连")
        for document in documents:
            corpus_row = int(document["corpus_row"])
            if corpus_row >= len(corpus_rows) or corpus_rows[corpus_row]["poem_id"] != document["poem_id"]:
                raise ValueError("document_index 与 lda_corpus 无法按 corpus_row、poem_id 一一回连")

    recorded_environment = scan.read_json(resolve(config["recorded_environment"]))
    environment_audit = compare_environment(recorded_environment, environment_info())
    scan.atomic_json(output / "audit/environment_audit.json", environment_audit)
    scan.atomic_json(
        output / "audit/preflight.json",
        {
            "status": "PASS",
            "matrix_shape": list(matrix.shape),
            "matrix_nnz": int(matrix.nnz),
            "matrix_dtype": str(matrix.dtype),
            "train_size": len(train),
            "test_size": len(test),
            "vocabulary_size": len(vocabulary),
            "hashes": checks,
            "environment_status": environment_audit["status"],
            "audited_lda_parameters": audited_lda,
        },
    )
    return {
        "config": config,
        "config_path": config_path,
        "qcv_config": qcv_config,
        "qcv_config_path": qcv_config_path,
        "output": output,
        "base_lda": base_lda,
        "audited_lda": audited_lda,
        "matrix": matrix,
        "vocabulary": vocabulary,
        "documents": documents,
        "corpus_rows": corpus_rows,
        "corpus_metadata": corpus_metadata,
        "train": train,
        "test": test,
        "x_train": matrix[train],
        "x_test": matrix[test],
        "split_audit": split_audit,
        "environment_audit": environment_audit,
        "input_hashes": checks,
    }


def original_run_paths(context: Mapping[str, Any], topic_count: int, seed: int, regime: str) -> tuple[Path, Path]:
    """返回一个原量化 run 的 NPZ 与检查点路径。"""

    root = resolve(context["config"]["candidate_validation_root"])
    npz_path = root / "topic_matrices" / f"{regime}_K{topic_count}_seed_{seed}.npz"
    checkpoint_path = root / "checkpoints" / f"{regime}_K{topic_count}_seed_{seed}.json"
    return npz_path, checkpoint_path


def run_directory(context: Mapping[str, Any], topic_count: int, seed: int, regime: str) -> Path:
    """返回恢复后的 train-only run 目录。"""

    return context["output"] / "quantitative_validation_runs" / f"K{topic_count}" / regime / f"seed_{seed}"


def tracked_original_files(context: Mapping[str, Any]) -> list[tuple[str, Path]]:
    """列出本任务必须保持只读的原始证据。"""

    config = context["config"]
    items: list[tuple[str, Path]] = []
    for topic_count in EXPECTED_K:
        for regime in EXPECTED_REGIMES:
            for seed in EXPECTED_SEEDS:
                npz_path, checkpoint_path = original_run_paths(context, topic_count, seed, regime)
                items.extend([("K12_K14_ORIGINAL_NPZ", npz_path), ("K12_K14_ORIGINAL_CHECKPOINT", checkpoint_path)])
    for category, directory in [
        ("K15_EXISTING_FULL_CORPUS", resolve(config["k15_model_root"])),
        ("K20_FROZEN_REFERENCE", resolve(config["k20_model_root"])),
        ("K20_FROZEN_COPY", resolve(config["frozen_k20_copy"])),
    ]:
        if directory.is_dir():
            items.extend((category, path) for path in directory.rglob("*") if path.is_file())
    qcv_config = context["qcv_config"]
    corpus_root = resolve(config["corpus_root"])
    singleton = [
        context["config_path"],
        context["qcv_config_path"],
        resolve(qcv_config["original_lda_config"]),
        resolve(qcv_config["frozen_split"]),
        resolve(qcv_config["frozen_split_audit"]),
        resolve(qcv_config["frozen_perplexity_config"]),
        corpus_root / "bow_matrix.npz",
        corpus_root / "vocabulary.csv",
        corpus_root / "document_index.csv",
        corpus_root / "lda_corpus.csv",
        corpus_root / "corpus_metadata.json",
        resolve(config["candidate_all_runs"]),
        resolve(config["candidate_cross_seed_stability"]),
        resolve(config["recorded_environment"]),
        ROOT / "src/run_quantitative_candidate_validation.py",
        ROOT / "src/run_exhaustive_k_scan.py",
        ROOT / "src/run_lower_k_boundary_perplexity.py",
        ROOT / "src/run_tf_df_lda.py",
    ]
    items.extend(("METHOD_OR_FROZEN_INPUT", path) for path in singleton)
    unique: dict[Path, str] = {}
    for category, path in items:
        target = path.resolve()
        if not target.is_file():
            raise FileNotFoundError(f"只读来源不存在：{target}")
        unique.setdefault(target, category)
    return [(unique[path], path) for path in sorted(unique, key=str)]


def establish_original_hash_baseline(context: Mapping[str, Any]) -> pd.DataFrame:
    """建立或验证原始产物哈希基线。"""

    rows = [
        {"category": category, "path": relative(path), "sha256": scan.file_sha256(path), "bytes": path.stat().st_size}
        for category, path in tracked_original_files(context)
    ]
    frame = pd.DataFrame(rows).sort_values(["category", "path"]).reset_index(drop=True)
    baseline_path = context["output"] / "inventory/original_artifact_hashes.csv"
    if baseline_path.is_file():
        previous = pd.read_csv(baseline_path, dtype=str).fillna("")
        current = frame.astype({"bytes": str})
        if previous.to_dict("records") != current.to_dict("records"):
            raise ValueError("原始只读产物哈希与恢复任务起始基线不一致")
    else:
        scan.atomic_frame(baseline_path, frame)
    return frame


def inspect_original_npz(path: Path, topic_count: int, vocab_size: int) -> tuple[bool, bool, bool]:
    """核验原 NPZ 的键、shape 与类型。"""

    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        has_unit = "topic_word_unit" in keys
        has_top = "top_indices" in keys
        has_components = "components" in keys or "components_" in keys
        if has_unit and data["topic_word_unit"].shape != (topic_count, vocab_size):
            raise ValueError(f"topic_word_unit shape 不符：{path}")
        if has_top and data["top_indices"].shape != (topic_count, 20):
            raise ValueError(f"top_indices shape 不符：{path}")
    return has_unit, has_top, has_components


def build_inventory(context: Mapping[str, Any]) -> pd.DataFrame:
    """建立严格的 40-run 原始产物清单。"""

    rows: list[dict[str, Any]] = []
    vocab_size = int(context["config"]["expected"]["vocabulary_size"])
    for topic_count in EXPECTED_K:
        for regime in EXPECTED_REGIMES:
            for seed in EXPECTED_SEEDS:
                npz_path, checkpoint_path = original_run_paths(context, topic_count, seed, regime)
                if not npz_path.is_file() or not checkpoint_path.is_file():
                    raise FileNotFoundError(f"40-run 原始组合缺失：K{topic_count} {regime} seed={seed}")
                has_unit, has_top, has_components = inspect_original_npz(npz_path, topic_count, vocab_size)
                checkpoint = scan.read_json(checkpoint_path)
                expected_matrix_hash = checkpoint.get("topic_matrix_sha256")
                if expected_matrix_hash != scan.file_sha256(npz_path):
                    raise ValueError(f"原 NPZ 与检查点哈希不一致：{npz_path}")
                rows.append(
                    {
                        "K": topic_count,
                        "seed": seed,
                        "prior_regime": regime,
                        "existing_npz_path": relative(npz_path),
                        "has_topic_word_unit": has_unit,
                        "has_top_indices": has_top,
                        "has_components": has_components,
                        "has_model_object": False,
                        "has_doc_topic": False,
                        "has_representative_poems": False,
                        "has_metadata": bool(checkpoint.get("audit")),
                        "recovery_required": True,
                    }
                )
    frame = pd.DataFrame(rows).sort_values(["K", "prior_regime", "seed"]).reset_index(drop=True)
    if len(frame) != 40 or frame[["K", "seed", "prior_regime"]].duplicated().any():
        raise ValueError("40-run inventory 不是完整唯一网格")
    scan.atomic_frame(context["output"] / "inventory/k12_k14_artifact_inventory.csv", frame)
    return frame


def write_document_row_map(context: Mapping[str, Any]) -> pd.DataFrame:
    """保存 43,015 首诗的唯一矩阵行映射。"""

    rows = [
        {
            "document_index": int(row["matrix_row"]),
            "matrix_row": int(row["matrix_row"]),
            "corpus_row": int(row["corpus_row"]),
            "poem_id": row["poem_id"],
            "author": row["author"],
            "title": row["title"],
            "corpus": "C",
            "corpus_definition": "TF>=3;DF>=3;NON_EMPTY_POEMS",
        }
        for row in context["documents"]
    ]
    frame = pd.DataFrame(rows)
    if len(frame) != 43015 or frame.document_index.duplicated().any() or frame.poem_id.duplicated().any() or frame.isna().any().any():
        raise ValueError("C_document_row_map 不是 43,015 行 1:1 完整映射")
    scan.atomic_frame(context["output"] / "data/C_document_row_map.csv", frame)
    return frame


def cosine_rows(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """逐行计算余弦，显式重新标准化以容纳 float32 来源。"""

    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    denominator = first_norm * second_norm
    return np.divide(np.sum(first * second, axis=1), denominator, out=np.zeros(first.shape[0], dtype=np.float64), where=denominator > 0)


def flattened_cosine(first: np.ndarray, second: np.ndarray) -> float:
    """计算两个完整 Topic×Word 矩阵的全局余弦。"""

    left = first.ravel()
    right = second.ravel()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def top_jaccard(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """计算配对 Topic 的 Top20 Jaccard。"""

    values = []
    for left, right in zip(first, second):
        left_set = set(map(int, left))
        right_set = set(map(int, right))
        values.append(len(left_set & right_set) / len(left_set | right_set))
    return np.asarray(values, dtype=np.float64)


def validate_reproduction(
    saved_unit: np.ndarray,
    saved_top: np.ndarray,
    reproduced_unit: np.ndarray,
    reproduced_top: np.ndarray,
    saved_components_hash: str,
    reproduced_components_hash: str,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    """按 exact-order 后 Hungarian 的预注册顺序判定恢复状态。"""

    direct_rows = cosine_rows(saved_unit, reproduced_unit)
    direct_diff = np.abs(saved_unit - reproduced_unit)
    direct_exact = bool(
        np.array_equal(saved_unit.astype(np.float32), reproduced_unit.astype(np.float32))
        and np.array_equal(saved_top, reproduced_top)
        and saved_components_hash == reproduced_components_hash
    )
    similarity = np.clip(
        saved_unit / np.linalg.norm(saved_unit, axis=1, keepdims=True)
        @ (reproduced_unit / np.linalg.norm(reproduced_unit, axis=1, keepdims=True)).T,
        -1.0,
        1.0,
    )
    reference_rows, reproduced_columns = linear_sum_assignment(-similarity)
    order = np.empty(saved_unit.shape[0], dtype=np.int64)
    order[reference_rows] = reproduced_columns
    matched_unit = reproduced_unit[order]
    matched_top = reproduced_top[order]
    matched_rows = cosine_rows(saved_unit, matched_unit)
    matched_jaccard = top_jaccard(saved_top, matched_top)
    permutation_exact = bool(
        np.array_equal(saved_unit.astype(np.float32), matched_unit.astype(np.float32))
        and np.array_equal(saved_top, matched_top)
        and not np.array_equal(order, np.arange(len(order), dtype=np.int64))
    )
    thresholds = config["reproduction_acceptance"]
    numerically_close = bool(
        float(matched_rows.mean()) >= float(thresholds["mean_matched_cosine_min"])
        and float(matched_jaccard.mean()) >= float(thresholds["mean_matched_top20_jaccard_min"])
    )
    if direct_exact:
        status = "EXACT_REPRODUCTION"
    elif permutation_exact:
        status = "PERMUTATION_EQUIVALENT"
    elif numerically_close:
        status = "NUMERICALLY_CLOSE"
    else:
        status = "REPRODUCTION_MISMATCH"
    mapping_rows = [
        {
            "reference_topic_id": reference_index + 1,
            "reproduced_topic_id": int(order[reference_index]) + 1,
            "matched_cosine": float(matched_rows[reference_index]),
            "matched_top20_jaccard": float(matched_jaccard[reference_index]),
            "topic_float32_exact": bool(np.array_equal(saved_unit[reference_index].astype(np.float32), matched_unit[reference_index].astype(np.float32))),
            "top20_exact": bool(np.array_equal(saved_top[reference_index], matched_top[reference_index])),
        }
        for reference_index in range(len(order))
    ]
    validation = {
        "reproduction_status": status,
        "accepted": status in ACCEPTED_STATUSES,
        "direct_order_float32_exact": direct_exact,
        "direct_order_max_abs_diff": float(direct_diff.max()),
        "direct_order_mean_abs_diff": float(direct_diff.mean()),
        "direct_order_mean_row_cosine": float(direct_rows.mean()),
        "direct_order_min_row_cosine": float(direct_rows.min()),
        "direct_order_global_cosine": flattened_cosine(saved_unit, reproduced_unit),
        "matching_method": "HUNGARIAN_MAX_FULL_TOPIC_WORD_COSINE",
        "permutation_is_identity": bool(np.array_equal(order, np.arange(len(order), dtype=np.int64))),
        "matched_max_abs_diff": float(np.abs(saved_unit - matched_unit).max()),
        "matched_mean_abs_diff": float(np.abs(saved_unit - matched_unit).mean()),
        "mean_matched_cosine": float(matched_rows.mean()),
        "min_matched_cosine": float(matched_rows.min()),
        "matched_global_cosine": flattened_cosine(saved_unit, matched_unit),
        "mean_matched_top20_jaccard": float(matched_jaccard.mean()),
        "minimum_matched_top20_jaccard": float(matched_jaccard.min()),
        "exact_top20_topic_count": int(sum(row["top20_exact"] for row in mapping_rows)),
        "saved_components_sha256": saved_components_hash,
        "reproduced_components_sha256": reproduced_components_hash,
        "components_sha256_exact": saved_components_hash == reproduced_components_hash,
        "acceptance_thresholds": dict(thresholds),
    }
    return validation, mapping_rows, order


def theta_checks(theta: np.ndarray, expected_shape: tuple[int, int], tolerance: float) -> dict[str, Any]:
    """验证 document-topic 矩阵 shape、有限性、非负性与行和。"""

    row_error = np.abs(theta.sum(axis=1) - 1.0)
    result = {
        "shape": list(theta.shape),
        "dtype": str(theta.dtype),
        "shape_correct": theta.shape == expected_shape,
        "all_finite": bool(np.isfinite(theta).all()),
        "all_non_negative": bool((theta >= 0).all()),
        "max_row_sum_error": float(row_error.max()),
        "row_sum_tolerance": float(tolerance),
        "row_sums_valid": bool((row_error <= tolerance).all()),
    }
    result["passed"] = bool(result["shape_correct"] and result["all_finite"] and result["all_non_negative"] and result["row_sums_valid"])
    return result


def representative_rows(
    context: Mapping[str, Any],
    theta: np.ndarray,
    topic_count: int,
    seed: int,
    regime: str,
    fit_scope: str,
    limit: int,
    author_balanced: bool = False,
) -> list[dict[str, Any]]:
    """从真实 theta_full 按概率提取原始或作者平衡代表诗。"""

    documents = context["documents"]
    corpus_rows = context["corpus_rows"]
    if corpus_rows is None:
        raise ValueError("代表诗生成需要读取 lda_corpus 正文")
    dominant = np.argmax(theta, axis=1)
    dominant_probability = np.max(theta, axis=1)
    rows: list[dict[str, Any]] = []
    document_indices = np.arange(theta.shape[0], dtype=np.int64)
    max_per_author = int(context["config"]["representative_poems"]["author_maximum_per_topic"])
    for topic_index in range(topic_count):
        scores = theta[:, topic_index]
        order = np.lexsort((document_indices, -scores))
        author_counts: Counter[str] = Counter()
        chosen: list[int] = []
        for document_index in order:
            document = documents[int(document_index)]
            author = document["author"]
            if author_balanced and author_counts[author] >= max_per_author:
                continue
            chosen.append(int(document_index))
            author_counts[author] += 1
            if len(chosen) == limit:
                break
        for rank, document_index in enumerate(chosen, 1):
            document = documents[document_index]
            poem = corpus_rows[int(document["corpus_row"])]
            rows.append(
                {
                    "K": topic_count,
                    "prior_regime": regime,
                    "seed": seed,
                    "fit_scope": fit_scope,
                    "topic_id": topic_index + 1,
                    "rank": rank,
                    "document_index": document_index,
                    "poem_id": poem["poem_id"],
                    "author": poem["author"],
                    "title": poem["title"],
                    "topic_probability": float(scores[document_index]),
                    "dominant_topic": int(dominant[document_index]) + 1,
                    "dominant_probability": float(dominant_probability[document_index]),
                    "original_text": poem["original_text"],
                    "cleaned_tokens": poem["cleaned_tokens"],
                    "author_balanced": author_balanced,
                    "author_maximum_per_topic": max_per_author if author_balanced else "",
                }
            )
    return rows


def write_representatives(
    context: Mapping[str, Any],
    directory: Path,
    theta: np.ndarray,
    topic_count: int,
    seed: int,
    regime: str,
    fit_scope: str,
    mirror_prefix: str | None = None,
) -> list[Path]:
    """写出 Top20/50/100 与作者平衡 Top20 四套代表诗。"""

    outputs: list[Path] = []
    raw_top100 = representative_rows(context, theta, topic_count, seed, regime, fit_scope, 100, False)
    for limit in [20, 50, 100]:
        rows = [row for row in raw_top100 if int(row["rank"]) <= limit]
        path = directory / f"representative_poems_top{limit}.csv"
        scan.atomic_frame(path, pd.DataFrame(rows))
        outputs.append(path)
        if mirror_prefix:
            mirror = context["output"] / "representative_poems" / f"{mirror_prefix}_top{limit}.csv"
            shutil.copyfile(path, mirror)
            outputs.append(mirror)
    balanced_rows = representative_rows(
        context,
        theta,
        topic_count,
        seed,
        regime,
        fit_scope,
        int(context["config"]["representative_poems"]["author_balanced_count"]),
        True,
    )
    balanced = directory / "representative_poems_author_balanced_top20.csv"
    scan.atomic_frame(balanced, pd.DataFrame(balanced_rows))
    outputs.append(balanced)
    if mirror_prefix:
        mirror = context["output"] / "representative_poems" / f"{mirror_prefix}_author_balanced_top20.csv"
        shutil.copyfile(balanced, mirror)
        outputs.append(mirror)
    return outputs


def output_registration(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """登记输出文件的相对路径、大小与哈希。"""

    return [
        {"path": relative(path), "bytes": path.stat().st_size, "sha256": scan.file_sha256(path)}
        for path in sorted(set(paths), key=str)
    ]


def valid_completed_metadata(path: Path) -> bool:
    """验证已完成 run 的登记输出，以支持安全续跑。"""

    if not path.is_file():
        return False
    try:
        metadata = scan.read_json(path)
        if metadata.get("status") != "COMPLETE":
            return False
        for item in metadata.get("outputs", []):
            target = resolve(item["path"])
            if not target.is_file() or scan.file_sha256(target) != item["sha256"]:
                return False
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False
    return True


def recover_quantitative_run(context: Mapping[str, Any], topic_count: int, seed: int, regime: str) -> dict[str, Any]:
    """按原 train-only 条件重新拟合一个 run，并在验收后补齐 theta。"""

    directory = run_directory(context, topic_count, seed, regime)
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "model_metadata.json"
    if valid_completed_metadata(metadata_path):
        return scan.read_json(metadata_path)

    npz_path, checkpoint_path = original_run_paths(context, topic_count, seed, regime)
    checkpoint = scan.read_json(checkpoint_path)
    if checkpoint.get("topic_matrix_sha256") != scan.file_sha256(npz_path):
        raise ValueError(f"原 NPZ 哈希失配：{npz_path}")
    if checkpoint.get("fit_rows_sha256") != context["split_audit"]["fit_rows_sha256"] or checkpoint.get("test_rows_sha256") != context["split_audit"]["test_rows_sha256"]:
        raise ValueError(f"TRAIN_ORDER_MISMATCH：K{topic_count} {regime} seed={seed}")
    with np.load(npz_path, allow_pickle=False) as saved:
        saved_unit = np.asarray(saved["topic_word_unit"], dtype=np.float64)
        saved_top = np.asarray(saved["top_indices"], dtype=np.int32)

    lda = qcv.regime_lda(context["base_lda"], context["qcv_config"], regime)
    model = lower.build_model(topic_count, seed, lda)
    fit_parameters = model.get_params(deep=False)
    started = time.perf_counter()
    model.fit(context["x_train"])
    fit_seconds = time.perf_counter() - started

    components = np.asarray(model.components_, dtype=np.float64)
    reproduced_components_hash = hashlib.sha256(components.tobytes()).hexdigest()
    topic_word = components / components.sum(axis=1, keepdims=True)
    norms = np.linalg.norm(topic_word, axis=1, keepdims=True)
    reproduced_unit = np.divide(topic_word, norms, out=np.zeros_like(topic_word), where=norms > 0)
    reproduced_top = np.asarray(original.topic_top_indices(topic_word, 20), dtype=np.int32)
    validation, mapping_rows, order = validate_reproduction(
        saved_unit,
        saved_top,
        reproduced_unit,
        reproduced_top,
        str(checkpoint["audit"]["components_sha256"]),
        reproduced_components_hash,
        context["config"],
    )
    validation.update(
        {
            "K": topic_count,
            "seed": seed,
            "prior_regime": regime,
            "fit_scope": "TRAIN_ONLY",
            "fit_time_seconds": fit_seconds,
            "original_npz_path": relative(npz_path),
            "original_npz_sha256": scan.file_sha256(npz_path),
            "original_checkpoint_path": relative(checkpoint_path),
            "topic_order_for_enriched_arrays": "NATIVE_REPRODUCED_MODEL_ORDER",
            "reference_to_reproduced_topic_order": order.tolist(),
        }
    )
    mapping_frame = pd.DataFrame(mapping_rows)
    mapping_frame.insert(0, "prior_regime", regime)
    mapping_frame.insert(0, "seed", seed)
    mapping_frame.insert(0, "K", topic_count)
    scan.atomic_frame(directory / "topic_permutation_map.csv", mapping_frame)
    scan.atomic_json(directory / "reproduction_validation.json", validation)
    scan.atomic_json(
        directory / "original_reference.json",
        {
            "K": topic_count,
            "seed": seed,
            "prior_regime": regime,
            "original_npz_path": relative(npz_path),
            "original_npz_sha256": scan.file_sha256(npz_path),
            "original_checkpoint_path": relative(checkpoint_path),
            "original_checkpoint_sha256": scan.file_sha256(checkpoint_path),
            "original_result": checkpoint["result"],
            "original_audit": checkpoint["audit"],
        },
    )
    if not validation["accepted"]:
        atomic_npz(
            directory / "reproduced_topic_word_diagnostic.npz",
            components=components,
            topic_word_unit=reproduced_unit,
            top_indices=reproduced_top,
        )
        mismatch_metadata = {
            "schema_version": 1,
            "status": "REPRODUCTION_MISMATCH",
            "K": topic_count,
            "seed": seed,
            "prior_regime": regime,
            "fit_scope": "TRAIN_ONLY",
            "fit_parameters": fit_parameters,
            "reproduction_status": validation["reproduction_status"],
            "blocking_issue": "TOPIC_WORD_REPRODUCTION_FAILED_BEFORE_THETA",
            "outputs": output_registration(
                [
                    directory / "original_reference.json",
                    directory / "reproduction_validation.json",
                    directory / "topic_permutation_map.csv",
                    directory / "reproduced_topic_word_diagnostic.npz",
                ]
            ),
        }
        scan.atomic_json(metadata_path, mismatch_metadata)
        return mismatch_metadata

    model.set_params(n_jobs=int(context["config"]["quantitative_fit"]["inference_n_jobs"]))
    theta_train = np.asarray(model.transform(context["x_train"]), dtype=np.float64)
    theta_test = np.asarray(model.transform(context["x_test"]), dtype=np.float64)
    theta_full = np.asarray(model.transform(context["matrix"]), dtype=np.float64)
    tolerance = float(context["config"]["reproduction_acceptance"]["theta_row_sum_abs_tolerance"])
    checks = {
        "doc_topic_train": theta_checks(theta_train, (len(context["train"]), topic_count), tolerance),
        "doc_topic_test": theta_checks(theta_test, (len(context["test"]), topic_count), tolerance),
        "doc_topic_full": theta_checks(theta_full, (context["matrix"].shape[0], topic_count), tolerance),
    }
    if not all(item["passed"] for item in checks.values()):
        raise ValueError(f"theta 合法性检查失败：K{topic_count} {regime} seed={seed}")
    prevalence = theta_train.mean(axis=0)
    old_prevalence = np.asarray([row["mean_probability"] for row in checkpoint["audit"]["topic_prevalence"]], dtype=np.float64)
    prevalence_error = float(np.max(np.abs(prevalence - old_prevalence)))
    if prevalence_error > float(context["config"]["reproduction_acceptance"]["prevalence_abs_tolerance"]):
        raise ValueError(f"恢复 theta_train 与原 prevalence 不一致：{prevalence_error}")

    artifact_path = directory / "model_artifacts_enriched.npz"
    atomic_npz(
        artifact_path,
        components=components,
        topic_word_unit=reproduced_unit,
        top_indices=reproduced_top,
        doc_topic_train=theta_train,
        doc_topic_test=theta_test,
        doc_topic_full=theta_full,
        train_row_indices=np.asarray(context["train"], dtype=np.int64),
        test_row_indices=np.asarray(context["test"], dtype=np.int64),
        full_document_indices=np.arange(context["matrix"].shape[0], dtype=np.int64),
    )
    model.set_params(n_jobs=int(fit_parameters["n_jobs"]))
    model_path = directory / "lda_model.joblib"
    atomic_joblib(model_path, model, int(context["config"]["joblib_compress"]))
    representative_paths = write_representatives(context, directory, theta_full, topic_count, seed, regime, "TRAIN_ONLY")
    output_paths = [
        directory / "original_reference.json",
        model_path,
        artifact_path,
        directory / "reproduction_validation.json",
        directory / "topic_permutation_map.csv",
        *representative_paths,
    ]
    metadata = {
        "schema_version": 1,
        "status": "COMPLETE",
        "K": topic_count,
        "seed": seed,
        "prior_regime": regime,
        "fit_scope": "TRAIN_ONLY",
        "purpose": "QUANTITATIVE_VALIDATION_ARTIFACT_RECOVERY",
        "max_iter": 30,
        "fit_parameters": fit_parameters,
        "effective_alpha": float(model.doc_topic_prior_),
        "effective_eta": float(model.topic_word_prior_),
        "n_iter_": int(model.n_iter_),
        "fit_time_seconds": fit_seconds,
        "train_size": len(context["train"]),
        "test_size": len(context["test"]),
        "full_size": context["matrix"].shape[0],
        "vocab_size": context["matrix"].shape[1],
        "DTM_file_sha256": context["input_hashes"]["matrix_file_sha256"],
        "DTM_fingerprint": context["input_hashes"]["matrix_fingerprint"],
        "vocab_sha256": context["input_hashes"]["vocabulary_sha256"],
        "train_rows_sha256": context["input_hashes"]["fit_rows_sha256"],
        "test_rows_sha256": context["input_hashes"]["test_rows_sha256"],
        "existing_npz_path": relative(npz_path),
        "existing_npz_sha256": scan.file_sha256(npz_path),
        "reproduced_model_sha256": scan.file_sha256(model_path),
        "components_sha256": reproduced_components_hash,
        "theta_full_sha256": array_sha256(theta_full),
        "software_versions": context["environment_audit"]["current"],
        "environment_status": context["environment_audit"]["status"],
        "reproduction_status": validation["reproduction_status"],
        "prevalence_max_abs_difference_from_original_checkpoint": prevalence_error,
        "theta_checks": checks,
        "outputs": output_registration(output_paths),
    }
    scan.atomic_json(metadata_path, metadata)
    return metadata


def collect_quantitative_results(context: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """汇总已有恢复状态与 Topic permutation 记录。"""

    validations: list[dict[str, Any]] = []
    mappings: list[pd.DataFrame] = []
    for topic_count in EXPECTED_K:
        for regime in EXPECTED_REGIMES:
            for seed in EXPECTED_SEEDS:
                directory = run_directory(context, topic_count, seed, regime)
                validation_path = directory / "reproduction_validation.json"
                if validation_path.is_file():
                    validation = scan.read_json(validation_path)
                    validations.append(
                        {
                            "K": topic_count,
                            "seed": seed,
                            "prior_regime": regime,
                            "reproduction_status": validation["reproduction_status"],
                            "accepted": validation["accepted"],
                            "components_sha256_exact": validation["components_sha256_exact"],
                            "direct_order_max_abs_diff": validation["direct_order_max_abs_diff"],
                            "direct_order_mean_abs_diff": validation["direct_order_mean_abs_diff"],
                            "direct_order_mean_row_cosine": validation["direct_order_mean_row_cosine"],
                            "direct_order_global_cosine": validation["direct_order_global_cosine"],
                            "permutation_is_identity": validation["permutation_is_identity"],
                            "mean_matched_cosine": validation["mean_matched_cosine"],
                            "min_matched_cosine": validation["min_matched_cosine"],
                            "mean_matched_top20_jaccard": validation["mean_matched_top20_jaccard"],
                            "exact_top20_topic_count": validation["exact_top20_topic_count"],
                            "theta_generated": (directory / "model_artifacts_enriched.npz").is_file(),
                        }
                    )
                mapping_path = directory / "topic_permutation_map.csv"
                if mapping_path.is_file():
                    mappings.append(pd.read_csv(mapping_path))
    validation_frame = pd.DataFrame(validations)
    if len(validation_frame):
        validation_frame = validation_frame.sort_values(["K", "prior_regime", "seed"]).reset_index(drop=True)
    mapping_frame = pd.concat(mappings, ignore_index=True) if mappings else pd.DataFrame()
    if len(mapping_frame):
        mapping_frame = mapping_frame.sort_values(["K", "prior_regime", "seed", "reference_topic_id"]).reset_index(drop=True)
    scan.atomic_frame(context["output"] / "data/model_reproduction_validation.csv", validation_frame)
    scan.atomic_frame(context["output"] / "data/topic_permutation_maps.csv", mapping_frame)
    return validation_frame, mapping_frame


def recover_quantitative(context: Mapping[str, Any], max_new_runs: int | None = None) -> pd.DataFrame:
    """检查点式恢复 40 个 quantitative-validation runs。"""

    new_runs = 0
    total = 40
    for topic_count in EXPECTED_K:
        for regime in EXPECTED_REGIMES:
            for seed in EXPECTED_SEEDS:
                metadata_path = run_directory(context, topic_count, seed, regime) / "model_metadata.json"
                if valid_completed_metadata(metadata_path):
                    continue
                if max_new_runs is not None and new_runs >= max_new_runs:
                    validations, _ = collect_quantitative_results(context)
                    return validations
                print(f"RECOVERY_START K={topic_count} regime={regime} seed={seed}", flush=True)
                metadata = recover_quantitative_run(context, topic_count, seed, regime)
                new_runs += 1
                validations, _ = collect_quantitative_results(context)
                scan.atomic_json(
                    context["output"] / "audit/progress.json",
                    {
                        "registered": len(validations),
                        "complete": int(validations.theta_generated.sum()) if len(validations) else 0,
                        "expected": total,
                        "last_run": {"K": topic_count, "seed": seed, "prior_regime": regime, "status": metadata.get("reproduction_status")},
                    },
                )
                print(f"RECOVERY_DONE {len(validations)}/{total} K={topic_count} regime={regime} seed={seed} status={metadata.get('reproduction_status')}", flush=True)
                try:
                    get_reusable_executor().shutdown(wait=True, kill_workers=True)
                except Exception:
                    pass
                gc.collect()
    validations, _ = collect_quantitative_results(context)
    return validations


def choose_representative_seeds(context: Mapping[str, Any]) -> pd.DataFrame:
    """按预注册中位数邻域与跨 seed 中心性规则选择 full-corpus seed。"""

    runs = pd.read_csv(resolve(context["config"]["candidate_all_runs"]))
    pairs = pd.read_csv(resolve(context["config"]["candidate_cross_seed_stability"]))
    rows: list[dict[str, Any]] = []
    rule = context["config"]["representative_seed_rule"]
    for topic_count in EXPECTED_K:
        selected_runs = runs[(runs.K == topic_count) & (runs.prior_regime == rule["regime"])].copy()
        if set(selected_runs.seed.astype(int)) != set(EXPECTED_SEEDS) or len(selected_runs) != 10:
            raise ValueError(f"K{topic_count} ORIGINAL_PRIOR 十 seed NPMI 不完整")
        median = float(selected_runs.NPMI.median())
        maximum_npmi = float(selected_runs.NPMI.max())
        selected_pairs = pairs[(pairs.K == topic_count) & (pairs.prior_regime == rule["regime"])]
        if len(selected_pairs) != 45:
            raise ValueError(f"K{topic_count} seed pair 稳定性不是 45 行")
        centrality: defaultdict[int, list[float]] = defaultdict(list)
        for pair in selected_pairs.itertuples(index=False):
            centrality[int(pair.seed_A)].append(float(pair.matched_cosine))
            centrality[int(pair.seed_B)].append(float(pair.matched_cosine))
        candidates = []
        for run in selected_runs.itertuples(index=False):
            seed = int(run.seed)
            values = centrality[seed]
            if len(values) != 9:
                raise ValueError(f"K{topic_count} seed={seed} 中心性缺少 pair")
            candidates.append(
                {
                    "K": topic_count,
                    "seed": seed,
                    "NPMI": float(run.NPMI),
                    "K_specific_NPMI_median": median,
                    "NPMI_distance_to_median": abs(float(run.NPMI) - median),
                    "stability_centrality": float(np.mean(values)),
                    "is_maximum_NPMI_seed": bool(np.isclose(float(run.NPMI), maximum_npmi, rtol=0.0, atol=0.0)),
                }
            )
        eligible_pool = [item for item in candidates if not item["is_maximum_NPMI_seed"]]
        eligible_pool.sort(key=lambda item: (item["NPMI_distance_to_median"], item["seed"]))
        central_band = eligible_pool[: int(rule["central_band_size"])]
        band_seeds = {item["seed"] for item in central_band}
        winner = sorted(central_band, key=lambda item: (-item["stability_centrality"], item["NPMI_distance_to_median"], item["seed"]))[0]
        for item in candidates:
            in_band = item["seed"] in band_seeds
            selected = item["seed"] == winner["seed"]
            item.update(
                {
                    "eligible_central_band": in_band,
                    "selected": selected,
                    "selection_reason": (
                        "SELECTED: maximum-NPMI seed excluded; within five closest non-maximum seeds to K-specific NPMI median; highest stability centrality"
                        if selected
                        else (
                            "NOT_SELECTED: maximum-NPMI seed excluded by preregistration"
                            if item["is_maximum_NPMI_seed"]
                            else (
                                "NOT_SELECTED: outside preregistered median-near central band"
                                if not in_band
                                else "NOT_SELECTED: central-band seed with lower stability centrality"
                            )
                        )
                    ),
                }
            )
            rows.append(item)
    frame = pd.DataFrame(rows).sort_values(["K", "NPMI_distance_to_median", "seed"]).reset_index(drop=True)
    if frame.selected.sum() != 2 or frame[frame.selected].is_maximum_NPMI_seed.any():
        raise ValueError("representative seed 选择违反预注册规则")
    scan.atomic_frame(context["output"] / "data/K12_K14_representative_seed_selection.csv", frame)
    return frame


def full_model_directory(context: Mapping[str, Any], topic_count: int) -> Path:
    """返回 full-corpus 盲审模型目录。"""

    return context["output"] / "full_corpus_blind_models" / f"K{topic_count}"


def build_full_corpus_model(context: Mapping[str, Any], topic_count: int, seed: int) -> dict[str, Any]:
    """为盲审拟合一个 ORIGINAL_PRIOR、max_iter=30 的全语料模型。"""

    directory = full_model_directory(context, topic_count)
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "model_metadata.json"
    if valid_completed_metadata(metadata_path):
        return scan.read_json(metadata_path)
    lda = qcv.regime_lda(context["base_lda"], context["qcv_config"], "ORIGINAL_PRIOR")
    model = lower.build_model(topic_count, seed, lda)
    fit_parameters = model.get_params(deep=False)
    started = time.perf_counter()
    model.fit(context["matrix"])
    fit_seconds = time.perf_counter() - started
    components = np.asarray(model.components_, dtype=np.float64)
    topic_word = components / components.sum(axis=1, keepdims=True)
    norms = np.linalg.norm(topic_word, axis=1, keepdims=True)
    topic_word_unit = np.divide(topic_word, norms, out=np.zeros_like(topic_word), where=norms > 0)
    top_indices = np.asarray(original.topic_top_indices(topic_word, 20), dtype=np.int32)
    model.set_params(n_jobs=1)
    theta_full = np.asarray(model.transform(context["matrix"]), dtype=np.float64)
    tolerance = float(context["config"]["reproduction_acceptance"]["theta_row_sum_abs_tolerance"])
    theta_qc = theta_checks(theta_full, (context["matrix"].shape[0], topic_count), tolerance)
    if not theta_qc["passed"]:
        raise ValueError(f"K{topic_count} full-corpus theta QC 失败")
    model.set_params(n_jobs=int(fit_parameters["n_jobs"]))

    enriched = directory / "model_artifacts_enriched.npz"
    atomic_npz(
        enriched,
        components=components,
        topic_word_unit=topic_word_unit,
        top_indices=top_indices,
        doc_topic_full=theta_full,
        full_document_indices=np.arange(context["matrix"].shape[0], dtype=np.int64),
    )
    components_path = directory / "components.npz"
    topic_word_path = directory / "topic_word_unit.npz"
    theta_path = directory / "document_topic_full.npz"
    top_indices_path = directory / "top_indices.npy"
    atomic_npz(components_path, components=components)
    atomic_npz(topic_word_path, topic_word_unit=topic_word_unit)
    atomic_npz(theta_path, doc_topic_full=theta_full)
    atomic_npy(top_indices_path, top_indices)
    model_path = directory / "lda_model.joblib"
    atomic_joblib(model_path, model, int(context["config"]["joblib_compress"]))
    representative_paths = write_representatives(
        context,
        directory,
        theta_full,
        topic_count,
        seed,
        "ORIGINAL_PRIOR",
        "FULL_CORPUS",
        f"K{topic_count}",
    )

    prevalence = theta_full.mean(axis=0)
    dominant_counts = np.bincount(np.argmax(theta_full, axis=1), minlength=topic_count)
    small_threshold = float(context["config"]["small_topic_probability_threshold"])
    topic_qc_rows = [
        {
            "K": topic_count,
            "seed": seed,
            "topic_id": index + 1,
            "mean_probability": float(prevalence[index]),
            "dominant_document_count": int(dominant_counts[index]),
            "dominant_document_share": float(dominant_counts[index] / theta_full.shape[0]),
            "small_topic_warning": bool(prevalence[index] < small_threshold),
            "small_topic_probability_threshold": small_threshold,
        }
        for index in range(topic_count)
    ]
    topic_qc_path = directory / "topic_qc.csv"
    scan.atomic_frame(topic_qc_path, pd.DataFrame(topic_qc_rows))
    output_paths = [
        enriched,
        components_path,
        topic_word_path,
        theta_path,
        top_indices_path,
        model_path,
        topic_qc_path,
        *representative_paths,
    ]
    metadata = {
        "schema_version": 1,
        "status": "COMPLETE",
        "K": topic_count,
        "seed": seed,
        "prior_regime": "ORIGINAL_PRIOR",
        "fit_scope": "FULL_CORPUS",
        "purpose": context["config"]["full_corpus_fit"]["purpose"],
        "quantitative_metrics_recomputed": False,
        "max_iter": 30,
        "fit_parameters": fit_parameters,
        "effective_alpha": float(model.doc_topic_prior_),
        "effective_eta": float(model.topic_word_prior_),
        "n_iter_": int(model.n_iter_),
        "fit_time_seconds": fit_seconds,
        "doc_count_used_for_fit": context["matrix"].shape[0],
        "full_size": context["matrix"].shape[0],
        "vocab_size": context["matrix"].shape[1],
        "DTM_file_sha256": context["input_hashes"]["matrix_file_sha256"],
        "DTM_fingerprint": context["input_hashes"]["matrix_fingerprint"],
        "vocab_sha256": context["input_hashes"]["vocabulary_sha256"],
        "model_sha256": scan.file_sha256(model_path),
        "components_sha256": hashlib.sha256(components.tobytes()).hexdigest(),
        "theta_full_sha256": array_sha256(theta_full),
        "theta_checks": theta_qc,
        "topic_prevalence_min": float(prevalence.min()),
        "topic_prevalence_max": float(prevalence.max()),
        "small_topic_warning_count": int(np.sum(prevalence < small_threshold)),
        "software_versions": context["environment_audit"]["current"],
        "environment_status": context["environment_audit"]["status"],
        "reproduction_status": "NOT_APPLICABLE_DIFFERENT_FIT_SCOPE",
        "outputs": output_registration(output_paths),
    }
    scan.atomic_json(metadata_path, metadata)
    return metadata


def build_full_corpus_models(context: Mapping[str, Any], max_new_runs: int | None = None) -> pd.DataFrame:
    """按代表 seed 规则建立 K12、K14 两个 full-corpus 模型。"""

    selection = choose_representative_seeds(context)
    new_runs = 0
    for topic_count in EXPECTED_K:
        selected = selection[(selection.K == topic_count) & (selection.selected)]
        if len(selected) != 1:
            raise ValueError(f"K{topic_count} representative seed 不唯一")
        seed = int(selected.iloc[0].seed)
        metadata_path = full_model_directory(context, topic_count) / "model_metadata.json"
        if valid_completed_metadata(metadata_path):
            continue
        if max_new_runs is not None and new_runs >= max_new_runs:
            break
        print(f"FULL_CORPUS_START K={topic_count} seed={seed}", flush=True)
        metadata = build_full_corpus_model(context, topic_count, seed)
        new_runs += 1
        print(f"FULL_CORPUS_DONE K={topic_count} seed={seed} fit_seconds={metadata['fit_time_seconds']:.3f}", flush=True)
        try:
            get_reusable_executor().shutdown(wait=True, kill_workers=True)
        except Exception:
            pass
        gc.collect()
    return selection


def read_sparse_shape(path: Path) -> tuple[int, int]:
    """读取既有稀疏 NPZ 的 shape。"""

    return tuple(map(int, sparse.load_npz(path).shape))


def build_scope_and_ready_manifest(context: Mapping[str, Any], selection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """核验 K12/K14/K15/K20 的 full-corpus 可比口径与就绪状态。"""

    config = context["config"]
    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for topic_count in EXPECTED_K:
        metadata_path = full_model_directory(context, topic_count) / "model_metadata.json"
        metadata = scan.read_json(metadata_path) if valid_completed_metadata(metadata_path) else {}
        directory = full_model_directory(context, topic_count)
        seed_rows = selection[(selection.K == topic_count) & (selection.selected)]
        seed = int(seed_rows.iloc[0].seed) if len(seed_rows) == 1 else ""
        ready = bool(metadata)
        rows.append(
            {
                "K": topic_count,
                "model_path": relative(directory / "lda_model.joblib") if ready else "",
                "fit_scope": metadata.get("fit_scope", "SOURCE_MISSING"),
                "seed": seed,
                "prior": metadata.get("prior_regime", "SOURCE_MISSING"),
                "max_iter": metadata.get("max_iter", ""),
                "doc_count_used_for_fit": metadata.get("doc_count_used_for_fit", ""),
                "corpus": "C",
                "corpus_hash": metadata.get("DTM_fingerprint", ""),
                "vocab_hash": metadata.get("vocab_sha256", ""),
            }
        )
        manifest.append(
            {
                "K": topic_count,
                "fit_scope": metadata.get("fit_scope", "SOURCE_MISSING"),
                "seed": seed,
                "prior_regime": metadata.get("prior_regime", "SOURCE_MISSING"),
                "max_iter": metadata.get("max_iter", ""),
                "model_path": relative(directory / "lda_model.joblib") if ready else "",
                "components_path": relative(directory / "components.npz") if ready else "",
                "topic_word_path": relative(directory / "topic_word_unit.npz") if ready else "",
                "theta_full_path": relative(directory / "document_topic_full.npz") if ready else "",
                "representative_poems_path": relative(directory / "representative_poems_top100.csv") if ready else "",
                "corpus_hash": metadata.get("DTM_fingerprint", ""),
                "vocab_hash": metadata.get("vocab_sha256", ""),
                "ready_for_blind_audit": ready,
                "blocking_issue": "" if ready else "FULL_CORPUS_ARTIFACT_MISSING",
            }
        )

    k15_root = resolve(config["k15_model_root"])
    k15_metrics = scan.read_json(resolve(config["k15_metrics"]))
    k15_model = joblib.load(k15_root / "model.joblib")
    k15_theta_shape = read_sparse_shape(k15_root / "document_topic_probabilities.npz")
    k15_ready = bool(k15_theta_shape == (43015, 15) and k15_model.components_.shape == (15, 35063))
    rows.append(
        {
            "K": 15,
            "model_path": relative(k15_root / "model.joblib"),
            "fit_scope": "FULL_CORPUS",
            "seed": int(k15_metrics["seed"]),
            "prior": "ORIGINAL_PRIOR",
            "max_iter": int(k15_model.max_iter),
            "doc_count_used_for_fit": int(k15_metrics["documents"]),
            "corpus": "C",
            "corpus_hash": context["input_hashes"]["matrix_fingerprint"],
            "vocab_hash": context["input_hashes"]["vocabulary_sha256"],
        }
    )
    manifest.append(
        {
            "K": 15,
            "fit_scope": "FULL_CORPUS",
            "seed": int(k15_metrics["seed"]),
            "prior_regime": "ORIGINAL_PRIOR",
            "max_iter": int(k15_model.max_iter),
            "model_path": relative(k15_root / "model.joblib"),
            "components_path": relative(k15_root / "topic_word_components.npz"),
            "topic_word_path": relative(k15_root / "topic_word_probabilities.npz"),
            "theta_full_path": relative(k15_root / "document_topic_probabilities.npz"),
            "representative_poems_path": relative(k15_root / "representative_poems_top20.csv"),
            "corpus_hash": context["input_hashes"]["matrix_fingerprint"],
            "vocab_hash": context["input_hashes"]["vocabulary_sha256"],
            "ready_for_blind_audit": k15_ready,
            "blocking_issue": "" if k15_ready else "K15_ARTIFACT_GAP",
        }
    )
    del k15_model

    k20_root = resolve(config["k20_model_root"])
    k20_matrix_metadata = scan.read_json(resolve(config["k20_matrix_metadata"]))
    k20_model = joblib.load(k20_root / "model.joblib")
    k20_shape = tuple(k20_matrix_metadata["document_topic_probabilities"]["shape"])
    k20_ready = bool(k20_shape == (43015, 20) and k20_model.components_.shape == (20, 35063))
    rows.append(
        {
            "K": 20,
            "model_path": relative(k20_root / "model.joblib"),
            "fit_scope": "FULL_CORPUS",
            "seed": int(k20_matrix_metadata["selected_seed"]),
            "prior": "ORIGINAL_PRIOR",
            "max_iter": int(k20_model.max_iter),
            "doc_count_used_for_fit": int(k20_shape[0]),
            "corpus": "C",
            "corpus_hash": context["input_hashes"]["matrix_fingerprint"],
            "vocab_hash": scan.file_sha256(k20_root / "vocabulary.csv"),
        }
    )
    manifest.append(
        {
            "K": 20,
            "fit_scope": "FULL_CORPUS",
            "seed": int(k20_matrix_metadata["selected_seed"]),
            "prior_regime": "ORIGINAL_PRIOR",
            "max_iter": int(k20_model.max_iter),
            "model_path": relative(k20_root / "model.joblib"),
            "components_path": relative(k20_root / "topic_word_components.npz"),
            "topic_word_path": relative(k20_root / "topic_word_probabilities.npz"),
            "theta_full_path": relative(k20_root / "document_topic_probabilities.npz"),
            "representative_poems_path": relative(k20_root / "representative_poems_top20.csv"),
            "corpus_hash": context["input_hashes"]["matrix_fingerprint"],
            "vocab_hash": scan.file_sha256(k20_root / "vocabulary.csv"),
            "ready_for_blind_audit": k20_ready,
            "blocking_issue": "" if k20_ready else "K20_FROZEN_ARTIFACT_GAP",
        }
    )
    del k20_model
    scope = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)
    ready_manifest = pd.DataFrame(manifest).sort_values("K").reset_index(drop=True)
    scan.atomic_frame(context["output"] / "data/blind_audit_model_scope_audit.csv", scope)
    scan.atomic_frame(context["output"] / "data/blind_audit_ready_model_manifest.csv", ready_manifest)
    return scope, ready_manifest


def verify_original_hashes(context: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """确认所有只读来源与任务起始哈希一致。"""

    baseline = pd.read_csv(context["output"] / "inventory/original_artifact_hashes.csv", dtype=str).fillna("")
    changed: list[str] = []
    for row in baseline.itertuples(index=False):
        path = resolve(row.path)
        if not path.is_file() or scan.file_sha256(path) != row.sha256 or str(path.stat().st_size) != str(row.bytes):
            changed.append(row.path)
    return not changed, changed


def verify_representative_file(path: Path, topic_count: int, expected_per_topic: int, author_balanced: bool = False) -> bool:
    """核验代表诗行数、Topic 覆盖、正文与作者上限。"""

    if not path.is_file():
        return False
    frame = pd.read_csv(path)
    if len(frame) != topic_count * expected_per_topic or set(frame.topic_id.astype(int)) != set(range(1, topic_count + 1)):
        return False
    if frame[["document_index", "poem_id", "original_text", "topic_probability"]].isna().any().any():
        return False
    if author_balanced and int(frame.groupby(["topic_id", "author"]).size().max()) > 2:
        return False
    return True


def build_qa(
    context: Mapping[str, Any],
    inventory: pd.DataFrame,
    validations: pd.DataFrame,
    selection: pd.DataFrame,
    scope: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """执行文件、恢复、theta、mapping、代表诗与冻结保护 QA。"""

    qa: list[dict[str, Any]] = []

    def add(test_id: str, description: str, passed: bool, details: str) -> None:
        qa.append({"test_id": test_id, "description": description, "status": "PASS" if passed else "FAIL", "details": details})

    add("QA01", "40/40 原始组合存在", len(inventory) == 40, f"rows={len(inventory)}")
    add("QA02", "原 NPZ 均含 topic_word_unit 与 top_indices", bool(inventory.has_topic_word_unit.all() and inventory.has_top_indices.all()), "40 NPZ key audit")
    add("QA03", "40 run 均有恢复状态", len(validations) == 40, f"rows={len(validations)}")
    status_counts = validations.reproduction_status.value_counts().to_dict() if len(validations) else {}
    add("QA04", "40 run 均通过 reproduction acceptance", len(validations) == 40 and bool(validations.accepted.all()), json.dumps(status_counts, ensure_ascii=False, sort_keys=True))
    add("QA05", "accepted run 均生成 theta_full", len(validations) == 40 and bool(validations.theta_generated.all()), f"generated={int(validations.theta_generated.sum()) if len(validations) else 0}")

    theta_ok = True
    for topic_count in EXPECTED_K:
        for regime in EXPECTED_REGIMES:
            for seed in EXPECTED_SEEDS:
                path = run_directory(context, topic_count, seed, regime) / "model_artifacts_enriched.npz"
                if not path.is_file():
                    theta_ok = False
                    continue
                with np.load(path, allow_pickle=False) as data:
                    tolerance = float(context["config"]["reproduction_acceptance"]["theta_row_sum_abs_tolerance"])
                    checks = [
                        theta_checks(data["doc_topic_train"], (34412, topic_count), tolerance),
                        theta_checks(data["doc_topic_test"], (8603, topic_count), tolerance),
                        theta_checks(data["doc_topic_full"], (43015, topic_count), tolerance),
                    ]
                    theta_ok = theta_ok and all(item["passed"] for item in checks)
    add("QA06", "全部 quantitative theta shape 与概率合法", theta_ok, "train/test/full")

    row_map = pd.read_csv(context["output"] / "data/C_document_row_map.csv", dtype={"poem_id": str})
    row_map_ok = len(row_map) == 43015 and not row_map.document_index.duplicated().any() and not row_map.poem_id.duplicated().any() and not row_map.isna().any().any()
    add("QA07", "43,015 行 row mapping 唯一且无缺失", row_map_ok, f"rows={len(row_map)}")

    quantitative_reps_ok = True
    for topic_count in EXPECTED_K:
        for regime in EXPECTED_REGIMES:
            for seed in EXPECTED_SEEDS:
                directory = run_directory(context, topic_count, seed, regime)
                quantitative_reps_ok = quantitative_reps_ok and verify_representative_file(directory / "representative_poems_top20.csv", topic_count, 20)
                quantitative_reps_ok = quantitative_reps_ok and verify_representative_file(directory / "representative_poems_top50.csv", topic_count, 50)
                quantitative_reps_ok = quantitative_reps_ok and verify_representative_file(directory / "representative_poems_top100.csv", topic_count, 100)
                quantitative_reps_ok = quantitative_reps_ok and verify_representative_file(directory / "representative_poems_author_balanced_top20.csv", topic_count, 20, True)
    add("QA08", "accepted quantitative Topics 均有三套 raw 与作者平衡代表诗", quantitative_reps_ok, "Top20/50/100 + author-balanced Top20")

    full_ok = True
    full_rep_ok = True
    for topic_count in EXPECTED_K:
        directory = full_model_directory(context, topic_count)
        metadata_path = directory / "model_metadata.json"
        full_ok = full_ok and valid_completed_metadata(metadata_path)
        if metadata_path.is_file():
            metadata = scan.read_json(metadata_path)
            full_ok = full_ok and bool(metadata["theta_checks"]["passed"])
        full_rep_ok = full_rep_ok and verify_representative_file(directory / "representative_poems_top20.csv", topic_count, 20)
        full_rep_ok = full_rep_ok and verify_representative_file(directory / "representative_poems_top50.csv", topic_count, 50)
        full_rep_ok = full_rep_ok and verify_representative_file(directory / "representative_poems_top100.csv", topic_count, 100)
        full_rep_ok = full_rep_ok and verify_representative_file(directory / "representative_poems_author_balanced_top20.csv", topic_count, 20, True)
    add("QA09", "K12/K14 full-corpus 模型 2/2 READY", full_ok, "FULL_CORPUS ORIGINAL_PRIOR max_iter=30")
    add("QA10", "K12/K14 full-corpus 代表诗完整", full_rep_ok, "Top20/50/100 + author-balanced Top20")
    selected_ok = int(selection.selected.sum()) == 2 and not bool(selection[selection.selected].is_maximum_NPMI_seed.any())
    add("QA11", "representative seed 遵循预注册规则且非最高 NPMI", selected_ok, ";".join(f"K{int(row.K)}={int(row.seed)}" for row in selection[selection.selected].itertuples()))
    comparable_columns = ["fit_scope", "prior", "max_iter", "doc_count_used_for_fit", "corpus", "corpus_hash", "vocab_hash"]
    comparable = len(scope) == 4 and all(scope[column].nunique(dropna=False) == 1 for column in comparable_columns)
    add("QA12", "四模型 fit scope/prior/max_iter/corpus/vocab 可比", comparable, json.dumps({column: scope[column].astype(str).unique().tolist() for column in comparable_columns}, ensure_ascii=False))
    ready = len(manifest) == 4 and bool(manifest.ready_for_blind_audit.all())
    add("QA13", "四模型均 ready_for_blind_audit", ready, f"ready={int(manifest.ready_for_blind_audit.sum())}/4")
    immutable, changed = verify_original_hashes(context)
    add("QA14", "原 K12/K14 NPZ、K15 与 frozen K20 哈希不变", immutable, "changed=" + ";".join(changed))
    add("QA15", "当前软件环境匹配原实验登记", context["environment_audit"]["status"] == "MATCHES_RECORDED_ENVIRONMENT", context["environment_audit"]["status"])
    add("QA16", "full-corpus 模型未重算 held-out perplexity", all(not scan.read_json(full_model_directory(context, k) / "model_metadata.json")["quantitative_metrics_recomputed"] for k in EXPECTED_K), "quantitative_metrics_recomputed=false")
    add("QA17", "K20 未重训且引用冻结模型", str(scope[scope.K == 20].iloc[0].model_path) == relative(resolve(context["config"]["k20_model_root"]) / "model.joblib"), "frozen reference path")
    add("QA18", "K15 未重训且引用既有 full-corpus 模型", str(scope[scope.K == 15].iloc[0].model_path) == relative(resolve(context["config"]["k15_model_root"]) / "model.joblib"), "existing K15 path")
    frame = pd.DataFrame(qa)
    scan.atomic_frame(context["output"] / "data/qa_results.csv", frame)
    return frame


def generate_report(
    context: Mapping[str, Any],
    inventory: pd.DataFrame,
    validations: pd.DataFrame,
    selection: pd.DataFrame,
    scope: pd.DataFrame,
    manifest: pd.DataFrame,
    qa: pd.DataFrame,
) -> None:
    """生成严格 15 节恢复报告、README 与 A–P 机器摘要。"""

    counts = validations.reproduction_status.value_counts().to_dict()
    selected = {int(row.K): int(row.seed) for row in selection[selection.selected].itertuples()}
    all_ready = bool(len(manifest) == 4 and manifest.ready_for_blind_audit.all() and (qa.status == "PASS").all())
    comparability_fields = ["fit_scope", "prior", "max_iter", "doc_count_used_for_fit", "corpus", "corpus_hash", "vocab_hash"]
    comparable = all(scope[field].nunique(dropna=False) == 1 for field in comparability_fields)
    status = "BLIND_AUDIT_READY" if all_ready and comparable else "BLIND_AUDIT_BLOCKED"
    lines = [
        "# 1. Objective",
        "",
        "恢复 K12/K14 的 40 个 train-only quantitative-validation 模型产物，并另建与既有 K15/K20 同口径的 full-corpus 盲审模型；本任务不执行语义盲审。",
        "",
        "# 2. Why Topic-Word Arrays Cannot Recover Theta",
        "",
        "`topic_word_unit` 只描述主题—词方向，缺少逐文档后验和 LDA components 的尺度信息，不能唯一反推出 P(topic|document)。本任务只从原模型对象或严格复现的模型执行 `transform`。",
        "",
        "# 3. Existing Artifact Inventory",
        "",
        f"40/40 原 NPZ 与检查点存在；原有 model object 为 {int(inventory.has_model_object.sum())}，40 个 NPZ 均只有 `topic_word_unit`、`top_indices`，故 40 个 run 均需重新拟合。",
        "",
        "# 4. Recovery Strategy",
        "",
        "逐 run 复用冻结 C DTM、词表、train/test 行序、K、seed、prior 与原 sklearn 参数。先比较重建 topic-word；只有达到预注册 acceptance 才生成 theta、模型对象和代表诗。",
        "",
        "# 5. Corpus / Vocabulary / Split Verification",
        "",
        f"DTM={context['matrix'].shape}，nnz={context['matrix'].nnz}，train={len(context['train'])}，test={len(context['test'])}；matrix、vocabulary、split、fit rows、test rows 哈希均与冻结记录一致。",
        "",
        "# 6. Model Reproduction Validation",
        "",
        "；".join(f"{name}={counts.get(name, 0)}" for name in ["EXACT_REPRODUCTION", "PERMUTATION_EQUIVALENT", "NUMERICALLY_CLOSE", "REPRODUCTION_MISMATCH"]),
        "",
        "# 7. K12 Recovery",
        "",
        f"K12 共 {int((validations.K == 12).sum())}/20 个 run 登记，accepted={int(validations[validations.K == 12].accepted.sum())}/20，全部明确标注 TRAIN_ONLY。",
        "",
        "# 8. K14 Recovery",
        "",
        f"K14 共 {int((validations.K == 14).sum())}/20 个 run 登记，accepted={int(validations[validations.K == 14].accepted.sum())}/20，全部明确标注 TRAIN_ONLY。",
        "",
        "# 9. Document-Topic Matrices",
        "",
        "accepted run 均保存 float64 `doc_topic_train`、`doc_topic_test`、`doc_topic_full` 及三套行索引；逐行 finite、非负、和为 1 的检查见 metadata 与 QA。",
        "",
        "# 10. Representative Poems",
        "",
        "每个 accepted Topic 均从真实 `doc_topic_full` 提取 Top20、Top50、Top100，以及同一作者每 Topic 最多 2 首的 author-balanced Top20；均含 poem_id、正文和 topic_probability。",
        "",
        "# 11. Full-Corpus Blind-Audit Models",
        "",
        f"K12 representative seed={selected.get(12)}；K14 representative seed={selected.get(14)}。两者均在 43,015 首冻结 C corpus 上以 ORIGINAL_PRIOR、max_iter=30 拟合，仅用于 Humanities Semantic Resolution Audit。",
        "",
        "# 12. Model Comparability Audit",
        "",
        f"四模型直接可比={comparable}。K12/K14 为本任务新建 full-corpus；K15/K20 只读复用 seed=2026 既有 full-corpus 模型。四者均为 ORIGINAL_PRIOR、max_iter=30、同 C DTM 与词表。",
        "",
        "# 13. QA Results",
        "",
        f"PASS={int((qa.status == 'PASS').sum())}/{len(qa)}；FAIL={int((qa.status != 'PASS').sum())}/{len(qa)}。详细结果见 `data/qa_results.csv`。",
        "",
        "# 14. Remaining Risks",
        "",
        "joblib 模型对象依赖已登记的软件版本；full-corpus K12/K14 与 train-only quantitative runs 属不同 fit scope，不得把 full-corpus 模型用于 held-out perplexity 或改写既有量化结果。",
        "",
        "# 15. Blind-Audit Readiness",
        "",
        f"最终状态：`{status}`。本任务在此停止，未开始 Four-Model Blind Humanities Semantic Resolution Audit，等待研究者确认。",
        "",
    ]
    report_path = context["output"] / "QTS_K12_K14_Artifact_Recovery_Report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    readme = [
        "# K12 / K14 模型产物恢复",
        "",
        "本目录保存 40 个 train-only quantitative-validation run 的可复核恢复产物，以及 K12/K14 两个 full-corpus 盲审模型。原 K12/K14 NPZ、既有 K15 与 frozen C_K20 均保持只读。",
        "",
        f"最终状态：`{status}`。完整方法、QA 与风险见 `QTS_K12_K14_Artifact_Recovery_Report.md`。",
        "",
        "目录中的 `full_corpus_blind_models/` 只服务于后续人文学语义盲审，不得并入 held-out perplexity 表。",
        "",
    ]
    (context["output"] / "README.md").write_text("\n".join(readme), encoding="utf-8")
    summary = {
        "A_all_40_original_npz_found": len(inventory) == 40,
        "B_original_model_objects": int(inventory.has_model_object.sum()),
        "C_runs_requiring_refit": int(inventory.recovery_required.sum()),
        "D_exact_reproduction": int(counts.get("EXACT_REPRODUCTION", 0)),
        "E_permutation_equivalent": int(counts.get("PERMUTATION_EQUIVALENT", 0)),
        "F_numerically_close": int(counts.get("NUMERICALLY_CLOSE", 0)),
        "G_reproduction_mismatch": int(counts.get("REPRODUCTION_MISMATCH", 0)),
        "H_all_successful_doc_topic_full_generated": bool(len(validations) == 40 and validations[validations.accepted].theta_generated.all()),
        "I_all_theta_full_shapes_correct": str(qa.loc[qa.test_id == "QA06", "status"].iloc[0]) == "PASS",
        "J_row_mapping_100_percent_unique_complete": str(qa.loc[qa.test_id == "QA07", "status"].iloc[0]) == "PASS",
        "K_all_representative_poems_generated": bool(str(qa.loc[qa.test_id == "QA08", "status"].iloc[0]) == "PASS" and str(qa.loc[qa.test_id == "QA10", "status"].iloc[0]) == "PASS"),
        "L_K12_representative_seed": selected.get(12),
        "M_K14_representative_seed": selected.get(14),
        "N_full_corpus_K12_K14_generated": str(qa.loc[qa.test_id == "QA09", "status"].iloc[0]) == "PASS",
        "O_four_model_direct_comparability": comparable,
        "O_comparability_details": scope.to_dict("records"),
        "P_final_status": status,
        "blocking_issues": manifest.loc[~manifest.ready_for_blind_audit, "blocking_issue"].astype(str).tolist(),
    }
    scan.atomic_json(context["output"] / "audit/completion_summary.json", summary)


def write_output_manifest(context: Mapping[str, Any]) -> None:
    """登记恢复目录内除 manifest 自身外的全部文件。"""

    manifest_path = context["output"] / "audit/output_manifest.json"
    rows = []
    for path in context["output"].rglob("*"):
        if path.is_file() and path != manifest_path:
            rows.append({"path": relative(path), "bytes": path.stat().st_size, "sha256": scan.file_sha256(path)})
    scan.atomic_json(manifest_path, {"schema_version": 1, "files": sorted(rows, key=lambda row: row["path"])})


def finalize(context: Mapping[str, Any]) -> dict[str, Any]:
    """生成最终清单、QA、报告和完成摘要。"""

    inventory = build_inventory(context)
    validations, _ = collect_quantitative_results(context)
    if len(validations) != 40:
        raise ValueError(f"quantitative recovery 尚未完成：{len(validations)}/40")
    if not validations.accepted.all() or not validations.theta_generated.all():
        raise ValueError("存在 REPRODUCTION_MISMATCH 或 theta 未生成，禁止宣称完成")
    selection = choose_representative_seeds(context)
    for topic_count in EXPECTED_K:
        if not valid_completed_metadata(full_model_directory(context, topic_count) / "model_metadata.json"):
            raise ValueError(f"K{topic_count} full-corpus 盲审模型尚未完成")
    scope, ready_manifest = build_scope_and_ready_manifest(context, selection)
    qa = build_qa(context, inventory, validations, selection, scope, ready_manifest)
    generate_report(context, inventory, validations, selection, scope, ready_manifest, qa)
    write_output_manifest(context)
    if not (qa.status == "PASS").all():
        raise ValueError("最终 QA 存在 FAIL")
    return scan.read_json(context["output"] / "audit/completion_summary.json")


def initialize(context: Mapping[str, Any]) -> None:
    """执行 inventory、只读哈希基线、row map 与 seed 规则登记。"""

    establish_original_hash_baseline(context)
    build_inventory(context)
    write_document_row_map(context)
    choose_representative_seeds(context)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--phase", choices=["inventory", "quantitative", "full", "finalize", "all"], default="all")
    parser.add_argument("--max-new-runs", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """按指定阶段执行可续跑恢复。"""

    args = parse_args(argv)
    context = prepare(args.config, with_text=True)
    initialize(context)
    if args.phase == "inventory":
        print(json.dumps({"status": "INVENTORY_COMPLETE", "runs": 40}, ensure_ascii=False))
        return 0
    if args.phase in {"quantitative", "all"}:
        validations = recover_quantitative(context, args.max_new_runs)
        if len(validations) < 40 or not validations.theta_generated.all():
            print(json.dumps({"status": "QUANTITATIVE_PARTIAL", "registered": len(validations), "expected": 40}, ensure_ascii=False))
            return 0
    if args.phase in {"full", "all"}:
        build_full_corpus_models(context, args.max_new_runs)
        complete = sum(valid_completed_metadata(full_model_directory(context, topic_count) / "model_metadata.json") for topic_count in EXPECTED_K)
        if complete < 2:
            print(json.dumps({"status": "FULL_CORPUS_PARTIAL", "complete": complete, "expected": 2}, ensure_ascii=False))
            return 0
    if args.phase in {"finalize", "all"}:
        summary = finalize(context)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
