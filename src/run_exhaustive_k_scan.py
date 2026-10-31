"""对《全唐诗》C 语料运行 K=2–30 的 train-only LDA 穷举诊断。"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import inspect
import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict
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

import clean_corpus as cleaner
import run_lower_k_boundary_perplexity as lower
import run_perplexity_diagnostic as frozen_diagnostic
import run_tf_df_lda as original


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "model_selection/exhaustive_k2_30_scan/config/exhaustive_k_scan_config.yaml"
REQUIRED_K = list(range(2, 31))
REQUIRED_SEEDS = [42, 2026, 20260830]
ALL_RUN_FIELDS = [
    "K",
    "seed",
    "n_train",
    "n_test",
    "vocab_size",
    "train_score",
    "test_score",
    "train_perplexity",
    "test_perplexity",
    "npmi_coherence",
    "n_iter_",
    "fit_time_seconds",
    "doc_topic_prior_effective",
    "topic_word_prior_effective",
    "converged_flag",
    "finite_perplexity_flag",
]


def resolve(path: str | Path) -> Path:
    """把配置中的相对路径解析到项目根目录。"""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_json(path: str | Path) -> dict[str, Any]:
    """读取采用 JSON 语法的配置或审计文件。"""

    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """读取 UTF-8 CSV。"""

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: str | Path) -> str:
    """分块计算文件 SHA-256。"""

    target = Path(path)
    for attempt in range(30):
        try:
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except (OSError, PermissionError):
            if attempt == 29:
                raise
            time.sleep(2)
    raise RuntimeError(f"无法读取文件：{target}")


def stable_hash(value: Any) -> str:
    """计算稳定 JSON 哈希。"""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    """以 UTF-8 原子写入 JSON。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def atomic_frame(path: str | Path, frame: pd.DataFrame) -> None:
    """以 UTF-8-SIG 原子写入数据表。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
    temporary.replace(target)


def load_config(path: str | Path) -> dict[str, Any]:
    """读取并严格验证本次穷举扫描配置。"""

    config = read_json(path)
    topics = config["topic_counts"]
    actual_k = list(range(int(topics["start"]), int(topics["stop"]) + 1, int(topics["step"])))
    if actual_k != REQUIRED_K:
        raise ValueError(f"K 必须逐整数覆盖 2–30，当前为 {actual_k}")
    if list(config["seeds"]) != REQUIRED_SEEDS:
        raise ValueError("随机种子必须为 42、2026、20260830")
    metrics = config["metrics"]
    if metrics["coherence"] != "document_npmi" or int(metrics["top_words"]) != 20 or metrics["npmi_reference"] != "TRAIN":
        raise ValueError("NPMI 必须复用 top-20 诗歌文档共现定义，且 reference 只能是 TRAIN")
    expected = config["expected"]
    if (expected["corpus"], int(expected["min_tf"]), int(expected["min_df"])) != ("C", 3, 3):
        raise ValueError("本实验只允许 C corpus（TF>=3、DF>=3）")
    output = resolve(config["output_root"]).resolve()
    allowed = (ROOT / "model_selection/exhaustive_k2_30_scan").resolve()
    if output != allowed:
        raise ValueError("输出目录越界")
    return config


def validate_lda_source(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """从第一轮原配置读取 LDA 参数并记录 sklearn 实际参数。"""

    original_config = read_json(resolve(config["original_lda_config"]))
    lda = dict(original_config["lda"])
    if lda["learning_method"] != "batch" or int(lda["max_iter"]) != 30:
        raise ValueError("原 LDA 参数已不再是 batch、max_iter=30")
    if list(lda["random_seeds"]) != REQUIRED_SEEDS:
        raise ValueError("原配置随机种子与本实验不一致")
    model = lower.build_model(2, REQUIRED_SEEDS[0], lda)
    actual = model.get_params(deep=False)
    audited = {
        key: actual[key]
        for key in [
            "learning_method",
            "max_iter",
            "doc_topic_prior",
            "topic_word_prior",
            "learning_decay",
            "learning_offset",
            "max_doc_update_iter",
            "mean_change_tol",
            "evaluate_every",
            "perp_tol",
            "n_jobs",
            "verbose",
        ]
    }
    for key in [
        "learning_method",
        "max_iter",
        "doc_topic_prior",
        "topic_word_prior",
        "max_doc_update_iter",
        "mean_change_tol",
        "evaluate_every",
        "perp_tol",
        "n_jobs",
        "verbose",
    ]:
        if actual[key] != lda.get(key):
            raise ValueError(f"原配置参数未被模型准确采用：{key}")
    return lda, audited


def load_corpus(config: Mapping[str, Any]) -> tuple[sparse.csr_matrix, list[str], list[dict[str, str]], dict[str, Any]]:
    """只读加载冻结 C 语料矩阵、词表与文档索引。"""

    root = resolve(config["corpus_root"])
    metadata = read_json(root / "corpus_metadata.json")
    matrix_path = root / "bow_matrix.npz"
    if file_sha256(matrix_path) != config["expected"]["matrix_sha256"]:
        raise ValueError("C corpus BoW 文件哈希不符")
    matrix = sparse.load_npz(matrix_path).tocsr()
    matrix.sort_indices()
    vocabulary_rows = sorted(read_csv(root / "vocabulary.csv"), key=lambda row: int(row["term_index"]))
    vocabulary = [row["token"] for row in vocabulary_rows]
    documents = sorted(read_csv(root / "document_index.csv"), key=lambda row: int(row["matrix_row"]))
    expected_shape = (int(config["expected"]["n_documents"]), int(config["expected"]["vocab_size"]))
    if matrix.shape != expected_shape or len(vocabulary) != expected_shape[1] or len(documents) != expected_shape[0]:
        raise ValueError("C corpus 的矩阵、词表或文档索引 shape 不符")
    if original.matrix_fingerprint(matrix) != metadata["matrix_fingerprint"]:
        raise ValueError("C corpus 稀疏矩阵逻辑哈希不符")
    if [int(row["matrix_row"]) for row in documents] != list(range(matrix.shape[0])):
        raise ValueError("C corpus 文档索引不连续")
    if [int(row["term_index"]) for row in vocabulary_rows] != list(range(matrix.shape[1])):
        raise ValueError("C corpus 词表索引不连续")
    return matrix, vocabulary, documents, metadata


def load_frozen_split(
    config: Mapping[str, Any], documents: Sequence[Mapping[str, str]]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """直接复用冻结 split，并由旧配置恢复其已审计的训练行顺序。"""

    split_path = resolve(config["frozen_split"])
    if not split_path.is_file():
        raise FileNotFoundError(f"冻结 split 不存在：{split_path}")
    expected = config["expected"]
    if file_sha256(split_path) != expected["split_sha256"]:
        raise ValueError("冻结 C split 的 SHA-256 不符")
    rows = read_csv(split_path)
    if len(rows) != len(documents):
        raise ValueError("冻结 split 行数与 C corpus 不一致")
    poem_ids = [row["poem_id"] for row in documents]
    if [row["poem_id"] for row in rows] != poem_ids:
        raise ValueError("冻结 split 的诗歌顺序与 C document_index 不一致")
    labels = [row["split"] for row in rows]
    if set(labels) != {"TRAIN", "TEST"}:
        raise ValueError("冻结 split 含非法分组标签")

    frozen_config = read_json(resolve(config["frozen_perplexity_config"]))
    train, test = frozen_diagnostic.make_split(len(documents), frozen_config["split"])
    train_set = set(train.tolist())
    expected_labels = ["TRAIN" if index in train_set else "TEST" for index in range(len(documents))]
    if labels != expected_labels:
        raise ValueError("冻结 split 与旧诊断配置恢复的成员关系不一致")
    audit = read_json(resolve(config["frozen_split_audit"]))
    checks = {
        "n_train": len(train),
        "n_test": len(test),
        "split_sha256": file_sha256(split_path),
        "fit_rows_sha256": frozen_diagnostic.row_digest(train),
        "test_rows_sha256": frozen_diagnostic.row_digest(test),
    }
    for key, value in checks.items():
        if audit[key] != value:
            raise ValueError(f"冻结 split 审计字段不一致：{key}")
    if len(train) != int(expected["n_train"]) or len(test) != int(expected["n_test"]):
        raise ValueError("冻结 split 的训练/测试规模不符")
    return train, test, audit


def tracked_inputs(config: Mapping[str, Any]) -> list[Path]:
    """列出本实验必须保持不变的冻结输入。"""

    corpus = resolve(config["corpus_root"])
    paths = [
        resolve(config["original_lda_config"]),
        resolve(config["frozen_split"]),
        resolve(config["frozen_split_audit"]),
        resolve(config["frozen_perplexity_config"]),
        corpus / "bow_matrix.npz",
        corpus / "vocabulary.csv",
        corpus / "document_index.csv",
        corpus / "corpus_metadata.json",
        ROOT / "src/run_tf_df_lda.py",
    ]
    for directory in [ROOT / "output/lda_sensitivity/models/C/K20", ROOT / "output/lda_candidate_rebuild/C_K20"]:
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    return sorted({path.resolve() for path in paths}, key=str)


def establish_input_manifest(config: Mapping[str, Any], output: Path) -> dict[str, str]:
    """建立或复核冻结输入哈希清单。"""

    current = {str(path): file_sha256(path) for path in tracked_inputs(config)}
    path = output / "audit/input_manifest.json"
    if path.is_file() and read_json(path) != current:
        raise ValueError("冻结输入与本实验起始基线不一致")
    if not path.is_file():
        atomic_json(path, current)
    return current


def prepare(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """完成所有只读前置核验并返回训练上下文。"""

    config = load_config(config_path)
    output = resolve(config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    for name in ["data", "figures", "checkpoints", "audit", "logs"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    lda, audited_params = validate_lda_source(config)
    matrix, vocabulary, documents, metadata = load_corpus(config)
    train, test, split_audit = load_frozen_split(config, documents)
    manifest = establish_input_manifest(config, output)
    source_hashes = {
        "topic_top_indices": hashlib.sha256(inspect.getsource(original.topic_top_indices).encode("utf-8")).hexdigest(),
        "topic_npmi_scores": hashlib.sha256(inspect.getsource(original.topic_npmi_scores).encode("utf-8")).hexdigest(),
        "model_factory": hashlib.sha256(inspect.getsource(lower.build_model).encode("utf-8")).hexdigest(),
        "evaluate_one": hashlib.sha256(inspect.getsource(evaluate_one).encode("utf-8")).hexdigest(),
    }
    identity_data = {
        "schema_version": config["schema_version"],
        "config": config,
        "original_lda": lda,
        "audited_sklearn_parameters": audited_params,
        "matrix_sha256": config["expected"]["matrix_sha256"],
        "vocabulary_sha256": file_sha256(resolve(config["corpus_root"]) / "vocabulary.csv"),
        "split_audit": split_audit,
        "source_hashes": source_hashes,
    }
    vocabulary_sha256 = identity_data["vocabulary_sha256"]
    identity = stable_hash(identity_data)
    identity_path = output / "audit/run_identity.json"
    payload = {"identity": identity, **identity_data}
    if identity_path.is_file() and read_json(identity_path) != payload:
        raise ValueError("统计实现或配置已变化，拒绝混用旧检查点")
    atomic_json(identity_path, payload)
    return {
        "config": config,
        "lda": lda,
        "audited_params": audited_params,
        "output": output,
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
        "vocabulary_sha256": vocabulary_sha256,
    }


def score_and_perplexity(model: Any, matrix: sparse.csr_matrix) -> tuple[float, float]:
    """用一次 sklearn score 推断并按其定义换算 perplexity。"""

    score = float(model.score(matrix))
    perplexity = float(math.exp(-score / float(matrix.sum())))
    if not math.isfinite(score) or not math.isfinite(perplexity) or perplexity <= 0:
        raise ValueError("score 或 perplexity 非有限正值")
    return score, perplexity


def evaluate_one(
    topic_count: int,
    seed: int,
    x_train: sparse.csr_matrix,
    x_test: sparse.csr_matrix,
    lda: Mapping[str, Any],
    top_words: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """只在训练集拟合同一模型，并计算 train/test 指标与 train-only NPMI。"""

    model = lower.build_model(topic_count, seed, lda)
    params = model.get_params(deep=False)
    started = time.perf_counter()
    model.fit(x_train)
    fit_time = time.perf_counter() - started
    components_digest = hashlib.sha256(model.components_.tobytes()).hexdigest()
    topic_word = model.components_ / model.components_.sum(axis=1, keepdims=True)
    top_indices = original.topic_top_indices(topic_word, top_words)
    _, mean_npmi = original.topic_npmi_scores(x_train, top_indices)
    model.set_params(n_jobs=1)
    train_score, train_perplexity = score_and_perplexity(model, x_train)
    test_score, test_perplexity = score_and_perplexity(model, x_test)
    if components_digest != hashlib.sha256(model.components_.tobytes()).hexdigest():
        raise ValueError("评分过程改变了 Topic-word 参数")
    if not math.isfinite(mean_npmi):
        raise ValueError("NPMI 非有限值")
    result = {
        "K": topic_count,
        "seed": seed,
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "vocab_size": int(x_train.shape[1]),
        "train_score": train_score,
        "test_score": test_score,
        "train_perplexity": train_perplexity,
        "test_perplexity": test_perplexity,
        "npmi_coherence": float(mean_npmi),
        "n_iter_": int(model.n_iter_),
        "fit_time_seconds": float(fit_time),
        "doc_topic_prior_effective": float(model.doc_topic_prior_),
        "topic_word_prior_effective": float(model.topic_word_prior_),
        "converged_flag": bool(model.n_iter_ < int(lda["max_iter"])),
        "finite_perplexity_flag": bool(math.isfinite(train_perplexity) and math.isfinite(test_perplexity) and train_perplexity > 0 and test_perplexity > 0),
    }
    audit = {
        "fit_scope": "X_train_only",
        "test_used_for_fit": False,
        "npmi_reference": "X_train_only",
        "npmi_top_words": top_words,
        "fit_n_jobs": params["n_jobs"],
        "evaluation_n_jobs": 1,
        "model_parameters": params,
        "components_sha256": components_digest,
    }
    del model, topic_word, top_indices
    return result, audit


def checkpoint_signature(context: Mapping[str, Any], topic_count: int, seed: int) -> str:
    """生成单次 run 的统计签名。"""

    return stable_hash({"identity": context["identity"], "K": topic_count, "seed": seed})


def checkpoint_path(output: Path, topic_count: int, seed: int) -> Path:
    """返回一个 run 的检查点路径。"""

    return output / f"checkpoints/C_K{topic_count}_seed_{seed}.json"


def load_checkpoint(path: Path, expected_signature: str) -> dict[str, Any]:
    """严格读取并验证单次 run 检查点。"""

    item = read_json(path)
    if item.get("checkpoint_signature") != expected_signature:
        raise ValueError(f"检查点签名不符：{path}")
    if item.get("result_sha256") != stable_hash(item.get("result")):
        raise ValueError(f"检查点结果哈希不符：{path}")
    return item


def collect_checkpoints(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """按 K、seed 收集所有已验证检查点。"""

    rows: list[dict[str, Any]] = []
    for topic_count, seed in itertools.product(REQUIRED_K, REQUIRED_SEEDS):
        path = checkpoint_path(context["output"], topic_count, seed)
        if path.is_file():
            item = load_checkpoint(path, checkpoint_signature(context, topic_count, seed))
            rows.append(item["result"])
    return rows


def write_all_runs(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """持续写出当前全部逐 run 数据。"""

    frame = pd.DataFrame(rows, columns=ALL_RUN_FIELDS).sort_values(["K", "seed"]) if rows else pd.DataFrame(columns=ALL_RUN_FIELDS)
    atomic_frame(output / "data/k_scan_all_runs.csv", frame)


def contiguous_runs(values: Sequence[int]) -> list[tuple[int, int]]:
    """把整数集合压缩为连续区间。"""

    if not values:
        return []
    ordered = sorted(set(int(value) for value in values))
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            ranges.append((start, previous))
            start = value
        previous = value
    ranges.append((start, previous))
    return ranges


def classify_perplexity(values: Sequence[float], mostly_fraction: float = 0.8) -> str:
    """按预注册规则判断 perplexity 曲线形状。"""

    array = np.asarray(values, dtype=float)
    differences = np.diff(array)
    if np.all(differences > 0):
        return "STRICTLY_INCREASING"
    minimum = int(np.argmin(array))
    if 0 < minimum < len(array) - 1 and np.all(differences[:minimum] < 0) and np.all(differences[minimum:] > 0):
        return "U_SHAPED"
    if float(np.mean(differences > 0)) >= mostly_fraction:
        return "MOSTLY_INCREASING"
    if np.any(differences > 0) and np.any(differences < 0):
        return "NON_MONOTONIC"
    return "OTHER"


def classify_npmi(summary: pd.DataFrame, min_plateau_length: int = 3) -> str:
    """按局部峰和 98% 连续高位带判断 NPMI 曲线形状。"""

    plateau_k = summary.loc[summary["npmi_top_98_percent_band"], "K"].astype(int).tolist()
    if any(end - start + 1 >= min_plateau_length for start, end in contiguous_runs(plateau_k)):
        return "BROAD_PLATEAU"
    local_count = int(summary["npmi_local_max"].sum())
    values = summary["mean_npmi"].to_numpy(float)
    differences = np.diff(values)
    if np.all(differences >= 0) or np.all(differences <= 0):
        return "MONOTONIC"
    if local_count >= 2:
        return "MULTIPLE_LOCAL_PEAKS"
    if local_count == 1:
        return "SINGLE_PEAK"
    return "IRREGULAR"


def summarize_runs(all_runs: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    """生成逐 K 汇总、差分、局部极值、平台与候选集合。"""

    required = set(itertools.product(REQUIRED_K, REQUIRED_SEEDS))
    actual = set(zip(all_runs["K"].astype(int), all_runs["seed"].astype(int)))
    if len(all_runs) != 87 or actual != required:
        raise ValueError("只有 87 个 K×seed run 完整时才能汇总")
    grouped = all_runs.groupby("K", sort=True)
    rows: list[dict[str, Any]] = []
    ddof = int(config["metrics"]["standard_deviation_ddof"])
    for topic_count, frame in grouped:
        npmi = frame["npmi_coherence"].astype(float)
        perplexity = frame["test_perplexity"].astype(float)
        rows.append(
            {
                "K": int(topic_count),
                "mean_npmi": float(npmi.mean()),
                "median_npmi": float(npmi.median()),
                "sd_npmi": float(npmi.std(ddof=ddof)),
                "min_npmi": float(npmi.min()),
                "max_npmi": float(npmi.max()),
                "mean_test_perplexity": float(perplexity.mean()),
                "median_test_perplexity": float(perplexity.median()),
                "sd_test_perplexity": float(perplexity.std(ddof=ddof)),
                "min_test_perplexity": float(perplexity.min()),
                "max_test_perplexity": float(perplexity.max()),
                "mean_fit_time": float(frame["fit_time_seconds"].astype(float).mean()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)
    summary["npmi_rank"] = summary["mean_npmi"].rank(method="min", ascending=False).astype(int)
    summary["perplexity_rank"] = summary["mean_test_perplexity"].rank(method="min", ascending=True).astype(int)
    summary["cv_npmi_percent"] = summary["sd_npmi"] / summary["mean_npmi"].abs() * 100.0
    summary["cv_test_perplexity_percent"] = summary["sd_test_perplexity"] / summary["mean_test_perplexity"].abs() * 100.0
    summary["delta_npmi"] = summary["mean_npmi"].diff()
    summary["delta_perplexity"] = summary["mean_test_perplexity"].diff()
    summary["second_diff_npmi"] = summary["delta_npmi"].diff()
    summary["second_diff_perplexity"] = summary["delta_perplexity"].diff()
    summary["npmi_moving_average_3"] = summary["mean_npmi"].rolling(3, center=True, min_periods=3).mean()
    summary["perplexity_moving_average_3"] = summary["mean_test_perplexity"].rolling(3, center=True, min_periods=3).mean()
    summary["primary_diagnostic_window"] = summary["K"].between(*config["diagnostics"]["primary_window"])
    summary["npmi_local_max"] = False
    summary["perplexity_local_min"] = False
    for index in range(1, len(summary) - 1):
        summary.loc[index, "npmi_local_max"] = bool(summary.loc[index, "mean_npmi"] > summary.loc[index - 1, "mean_npmi"] and summary.loc[index, "mean_npmi"] > summary.loc[index + 1, "mean_npmi"])
        summary.loc[index, "perplexity_local_min"] = bool(summary.loc[index, "mean_test_perplexity"] < summary.loc[index - 1, "mean_test_perplexity"] and summary.loc[index, "mean_test_perplexity"] < summary.loc[index + 1, "mean_test_perplexity"])
    global_npmi = float(summary["mean_npmi"].max())
    summary["npmi_ratio_to_global_max"] = summary["mean_npmi"] / global_npmi
    summary["npmi_top_1_percent_band"] = summary["mean_npmi"] >= global_npmi * 0.99
    summary["npmi_top_98_percent_band"] = summary["mean_npmi"] >= global_npmi * 0.98

    differences = summary[["K", "delta_npmi", "delta_perplexity", "second_diff_npmi", "second_diff_perplexity"]].copy()
    extrema_rows: list[dict[str, Any]] = []
    for index in range(1, len(summary) - 1):
        row = summary.iloc[index]
        if bool(row["npmi_local_max"]):
            extrema_rows.append({"K": int(row["K"]), "extremum_type": "NPMI_LOCAL_MAX", "metric": "mean_npmi", "value": float(row["mean_npmi"]), "previous_value": float(summary.iloc[index - 1]["mean_npmi"]), "next_value": float(summary.iloc[index + 1]["mean_npmi"])})
        if bool(row["perplexity_local_min"]):
            extrema_rows.append({"K": int(row["K"]), "extremum_type": "PPL_LOCAL_MIN", "metric": "mean_test_perplexity", "value": float(row["mean_test_perplexity"]), "previous_value": float(summary.iloc[index - 1]["mean_test_perplexity"]), "next_value": float(summary.iloc[index + 1]["mean_test_perplexity"])})
    extrema = pd.DataFrame(extrema_rows, columns=["K", "extremum_type", "metric", "value", "previous_value", "next_value"])
    plateau = summary[["K", "mean_npmi", "npmi_ratio_to_global_max", "npmi_top_1_percent_band", "npmi_top_98_percent_band"]].copy()

    reasons: defaultdict[int, list[str]] = defaultdict(list)
    global_npmi_k = int(summary.loc[summary["mean_npmi"].idxmax(), "K"])
    reasons[global_npmi_k].append("NPMI_GLOBAL_MAX")
    for topic_count in summary.loc[summary["npmi_local_max"], "K"].astype(int):
        reasons[topic_count].append("NPMI_LOCAL_MAX")
    for topic_count in summary.loc[summary["npmi_top_1_percent_band"], "K"].astype(int):
        reasons[topic_count].append("NPMI_TOP_1_PERCENT_BAND")
    for topic_count in summary.loc[summary["perplexity_local_min"], "K"].astype(int):
        reasons[topic_count].append("PPL_LOCAL_MIN")
    reasons[15].append("HISTORICAL_K15_REFERENCE")
    reasons[20].append("CURRENT_K20_REFERENCE")
    stable_npmi_cutoff = float(summary["sd_npmi"].median())
    candidate_rows: list[dict[str, Any]] = []
    indexed = summary.set_index("K")
    for topic_count in sorted(reasons):
        row = indexed.loc[topic_count]
        stable = bool(float(row["sd_npmi"]) <= stable_npmi_cutoff)
        substantive = [reason for reason in reasons[topic_count] if not reason.endswith("REFERENCE")]
        if bool(row["npmi_top_1_percent_band"]) and stable:
            tier = "TIER_1"
        elif substantive:
            tier = "TIER_2"
        else:
            tier = "REFERENCE"
        candidate_rows.append({"K": topic_count, "candidate_tier": tier, "selection_reasons": ";".join(reasons[topic_count]), "mean_npmi": float(row["mean_npmi"]), "sd_npmi": float(row["sd_npmi"]), "mean_test_perplexity": float(row["mean_test_perplexity"]), "sd_test_perplexity": float(row["sd_test_perplexity"]), "npmi_seed_stable_relative_to_all_k_median_sd": stable})
    candidates = pd.DataFrame(candidate_rows)
    perplexity_shape = classify_perplexity(summary["mean_test_perplexity"], float(config["diagnostics"]["mostly_increasing_min_fraction"]))
    npmi_shape = classify_npmi(summary, int(config["diagnostics"]["broad_plateau_min_contiguous_k"]))
    return {"summary": summary, "differences": differences, "extrema": extrema, "plateau": plateau, "candidates": candidates, "perplexity_shape": perplexity_shape, "npmi_shape": npmi_shape, "stable_npmi_sd_cutoff": stable_npmi_cutoff}


def setup_plot_style() -> None:
    """设置稳定、适合中文论文图的绘图样式。"""

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})


def annotate_points(ax: Any, frame: pd.DataFrame, x_column: str, y_column: str, mask: pd.Series, prefix: str) -> None:
    """在曲线图标记全局或局部极值点。"""

    selected = frame.loc[mask]
    for index, row in enumerate(selected.itertuples(index=False)):
        x_value = getattr(row, x_column)
        y_value = getattr(row, y_column)
        horizontal_offset = 0 if len(selected) == 1 else (-24 if index % 2 == 0 else 24)
        vertical_offset = 10 if len(selected) == 1 else 10 + (index % 3) * 8
        ax.scatter([x_value], [y_value], s=45, color="#B34234", zorder=5)
        ax.annotate(f"{prefix} K={int(x_value)}", (x_value, y_value), xytext=(horizontal_offset, vertical_offset), textcoords="offset points", ha="center", fontsize=8, color="#7A2E26")


def generate_figures(summary: pd.DataFrame, output: Path) -> None:
    """生成任务要求的六张定量诊断图。"""

    setup_plot_style()
    figures = output / "figures"
    k = summary["K"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.errorbar(k, summary["mean_npmi"], yerr=summary["sd_npmi"], marker="o", markersize=4, linewidth=1.8, capsize=3, color="#245A7A")
    annotate_points(ax, summary, "K", "mean_npmi", summary["mean_npmi"].eq(summary["mean_npmi"].max()), "全局峰")
    annotate_points(ax, summary, "K", "mean_npmi", summary["npmi_local_max"] & ~summary["mean_npmi"].eq(summary["mean_npmi"].max()), "局部峰")
    ax.set(xlabel="主题数 K", ylabel="mean TRAIN_SPLIT_NPMI ↑", title="C Corpus：K=2–30 Train-only NPMI")
    ax.set_xticks(k)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout(); fig.savefig(figures / "C_K2_30_NPMI_curve.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.errorbar(k, summary["mean_test_perplexity"], yerr=summary["sd_test_perplexity"], marker="o", markersize=4, linewidth=1.8, capsize=3, color="#2D6A4F")
    annotate_points(ax, summary, "K", "mean_test_perplexity", summary["mean_test_perplexity"].eq(summary["mean_test_perplexity"].min()), "全局谷")
    annotate_points(ax, summary, "K", "mean_test_perplexity", summary["perplexity_local_min"] & ~summary["mean_test_perplexity"].eq(summary["mean_test_perplexity"].min()), "局部谷")
    ax.set(xlabel="主题数 K", ylabel="mean held-out perplexity ↓", title="C Corpus：K=2–30 Held-out Perplexity")
    ax.set_xticks(k)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout(); fig.savefig(figures / "C_K2_30_Perplexity_curve.png", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8.2), sharex=True)
    axes[0].errorbar(k, summary["mean_npmi"], yerr=summary["sd_npmi"], marker="o", markersize=3.8, capsize=2.5, color="#245A7A")
    axes[0].set_ylabel("NPMI ↑"); axes[0].set_title("A. Train-only NPMI"); axes[0].grid(axis="y", alpha=0.2)
    axes[1].errorbar(k, summary["mean_test_perplexity"], yerr=summary["sd_test_perplexity"], marker="o", markersize=3.8, capsize=2.5, color="#2D6A4F")
    axes[1].set_ylabel("Held-out Perplexity ↓"); axes[1].set_title("B. Held-out Perplexity"); axes[1].set_xlabel("主题数 K"); axes[1].set_xticks(k); axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("C Corpus K=2–30 双指标诊断", fontsize=15)
    fig.tight_layout(); fig.savefig(figures / "C_K2_30_NPMI_and_Perplexity.png", bbox_inches="tight"); plt.close(fig)

    zoom = summary[summary["K"].between(10, 20)]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8), sharex=True)
    axes[0].errorbar(zoom["K"], zoom["mean_npmi"], yerr=zoom["sd_npmi"], marker="o", capsize=3, color="#245A7A")
    axes[0].set_ylabel("NPMI ↑"); axes[0].set_title("A. K=10–20 Train-only NPMI"); axes[0].grid(axis="y", alpha=0.2)
    axes[1].errorbar(zoom["K"], zoom["mean_test_perplexity"], yerr=zoom["sd_test_perplexity"], marker="o", capsize=3, color="#2D6A4F")
    axes[1].set_ylabel("Held-out Perplexity ↓"); axes[1].set_title("B. K=10–20 Held-out Perplexity"); axes[1].set_xlabel("主题数 K"); axes[1].set_xticks(zoom["K"]); axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("C Corpus 主诊断窗口 K=10–20", fontsize=15)
    fig.tight_layout(); fig.savefig(figures / "C_K10_20_zoom.png", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.8), sharex=True)
    axes[0].axhline(0, color="#555555", linewidth=1); axes[0].plot(k, summary["delta_npmi"], marker="o", color="#245A7A"); axes[0].set_ylabel("ΔNPMI"); axes[0].set_title("A. NPMI 一阶变化"); axes[0].grid(axis="y", alpha=0.2)
    axes[1].axhline(0, color="#555555", linewidth=1); axes[1].plot(k, summary["delta_perplexity"], marker="o", color="#2D6A4F"); axes[1].set_ylabel("ΔPerplexity"); axes[1].set_title("B. Perplexity 一阶变化"); axes[1].set_xlabel("主题数 K"); axes[1].set_xticks(k); axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(figures / "C_K2_30_first_difference.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    width = 0.38
    ax.bar(k - width / 2, summary["npmi_rank"], width=width, label="NPMI rank", color="#5B8DB8")
    ax.bar(k + width / 2, summary["perplexity_rank"], width=width, label="Perplexity rank", color="#72A98F")
    ax.invert_yaxis(); ax.set_xlabel("主题数 K"); ax.set_ylabel("指标排名（1 为最佳）"); ax.set_title("C Corpus：NPMI 与 Perplexity 独立排名"); ax.set_xticks(k); ax.legend(frameon=False, ncol=2); ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(figures / "C_K2_30_metric_rank.png", bbox_inches="tight"); plt.close(fig)


def format_k_ranges(values: Sequence[int]) -> str:
    """把 K 集合格式化为紧凑中文区间。"""

    ranges = contiguous_runs(values)
    return "、".join(f"K{start}" if start == end else f"K{start}–K{end}" for start, end in ranges) if ranges else "无"


def generate_report(artifacts: Mapping[str, Any], all_runs: pd.DataFrame, config: Mapping[str, Any], audited_params: Mapping[str, Any], output: Path) -> None:
    """生成十四节中文主报告与目录说明。"""

    summary = artifacts["summary"]
    candidates = artifacts["candidates"]
    best_npmi = summary.loc[summary["mean_npmi"].idxmax()]
    best_ppl = summary.loc[summary["mean_test_perplexity"].idxmin()]
    local_npmi = summary.loc[summary["npmi_local_max"], "K"].astype(int).tolist()
    local_ppl = summary.loc[summary["perplexity_local_min"], "K"].astype(int).tolist()
    band99 = summary.loc[summary["npmi_top_1_percent_band"], "K"].astype(int).tolist()
    band98 = summary.loc[summary["npmi_top_98_percent_band"], "K"].astype(int).tolist()
    zoom = summary[summary["K"].between(10, 20)]
    zoom_best = zoom.loc[zoom["mean_npmi"].idxmax()]
    row15 = summary.loc[summary["K"] == 15].iloc[0]
    row20 = summary.loc[summary["K"] == 20].iloc[0]
    competitors = summary[summary["K"].isin([13, 14, 16, 17, 18, 19])]
    better_than_15 = competitors.loc[competitors["mean_npmi"] > row15["mean_npmi"], "K"].astype(int).tolist()
    new_candidates = candidates.loc[~candidates["K"].isin([15, 20]) & candidates["selection_reasons"].str.contains("NPMI_|PPL_")]
    if len(new_candidates):
        recommendation = "FINE_GRID_REVEALS_NEW_CANDIDATE"
    elif int(best_npmi["K"]) == 15:
        recommendation = "SEMANTIC_REVIEW_RECOMMENDED"
    else:
        recommendation = "NO_NEW_CANDIDATE"
    plateau_ranges = contiguous_runs(band98)
    if artifacts["npmi_shape"] == "BROAD_PLATEAU":
        plateau_start, plateau_end = max(plateau_ranges, key=lambda pair: pair[1] - pair[0])
        plateau_label = f"NPMI_PLATEAU_K{plateau_start}_K{plateau_end}"
        plateau_other = [value for start, end in plateau_ranges if (start, end) != (plateau_start, plateau_end) for value in range(start, end + 1)]
    else:
        plateau_label = "未形成预注册的 broad plateau"
        plateau_other = []
    peak_status = "K15_REMAINS_NPMI_PEAK" if int(best_npmi["K"]) == 15 else "NEW_QUANTITATIVE_CANDIDATE"
    prior_note = "原配置两项均为 None；scikit-learn 本次实际按每个 K 使用 1/K，故有效先验随 K 改变。"
    if artifacts["perplexity_shape"] in {"STRICTLY_INCREASING", "MOSTLY_INCREASING"} and int(best_ppl["K"]) <= 4:
        low_k_note = "Held-out predictive fit systematically favors very low-resolution topic models. 这显示预测拟合与人文学语义分辨率之间存在权衡；不能据此把最低 K 自动称为最佳模型。"
    else:
        low_k_note = "Held-out perplexity 未呈现对极低 K 的系统性单向偏好；仍不能仅凭该指标决定最终 K。"
    stable_near = summary[summary["K"].between(max(2, int(best_npmi["K"]) - 1), min(30, int(best_npmi["K"]) + 1))]

    lines = [
        "# QTS C Corpus K=2–30 Exhaustive K Scan Report",
        "",
        "## 1. Objective",
        "",
        "旧实验只按 K=20/30/40/50/60 粗粒度搜索；补测 K=5/10/15 后，K15 的旧 full-corpus NPMI 超过 K20，而 held-out perplexity 偏好更低 K。本实验逐整数扫描 K=2–30，用统一 train-only 模型状态回答真实局部峰、平台与转折在哪里。旧 NPMI 保留不覆盖；本文数值统一标记为 `TRAIN_SPLIT_NPMI`。",
        "",
        "## 2. Experimental Design",
        "",
        f"只使用 C corpus（TF>=3、DF>=3），冻结矩阵 {int(best_npmi['K']*0+43015):,} 首非空诗、词表 {int(all_runs['vocab_size'].iloc[0]):,} 词。K=2–30 共 29 个整数，每 K 使用 seed 42、2026、20260830，共 87 runs。所有 run 直接复用旧 `perplexity_split_C.csv` 的 34,412/8,603 切分；CSV 保存成员关系，训练行顺序由旧配置恢复并以旧 `fit_rows_sha256`、`test_rows_sha256` 复核。每个模型只在 X_train 拟合；X_test 只用于评分；NPMI 共现 reference 也只用 X_train。",
        "",
        "## 3. Perplexity",
        "",
        f"全局最低 held-out perplexity 位于 K={int(best_ppl['K'])}：{best_ppl['mean_test_perplexity']:.6f} ± {best_ppl['sd_test_perplexity']:.6f}。正式形状判断为 `{artifacts['perplexity_shape']}`。Perplexity 越低，表示模型对未参与训练的新诗给出的概率预测更顺；这里由 sklearn `score` 的同一变分下界按 `exp(-score/词次)` 换算，并逐 run 验证为有限正值。{low_k_note}",
        "",
        "## 4. NPMI",
        "",
        f"全局最高 `TRAIN_SPLIT_NPMI` 位于 K={int(best_npmi['K'])}：{best_npmi['mean_npmi']:.10f} ± {best_npmi['sd_npmi']:.10f}；状态为 `{peak_status}`。NPMI 越高，表示同一 Topic 的 top-20 重要词在训练诗歌中彼此共同出现得更自然。正式曲线形状判断为 `{artifacts['npmi_shape']}`。",
        "",
        "## 5. Local Peaks and Valleys",
        "",
        f"NPMI 严格局部峰：{format_k_ranges(local_npmi)}。Perplexity 严格局部谷：{format_k_ranges(local_ppl)}。端点只参加全局极值判断，不按给定的双邻点定义标记为局部极值。二阶差分仅保存在诊断表中，不据此自动认定最佳 K。",
        "",
        "## 6. K10–20 Zoom",
        "",
        f"K10–20 内 NPMI 最高为 K={int(zoom_best['K'])}（{zoom_best['mean_npmi']:.10f}）。指定的 K13、K14、K16、K17、K18、K19 中，高于 K15 的是：{format_k_ranges(better_than_15)}。该比较只产生 quantitative candidate，不替代人工语义审核。",
        "",
        "## 7. Metric Seed Variability",
        "",
        f"全体 K 的 NPMI SD 中位数为 {artifacts['stable_npmi_sd_cutoff']:.10f}，候选分层以此作相对稳定阈值；perplexity 另报告 SD 与 CV。NPMI 峰值附近 K={int(stable_near['K'].min())}–{int(stable_near['K'].max())} 的最大 NPMI SD 为 {stable_near['sd_npmi'].max():.10f}，最大 perplexity CV 为 {stable_near['cv_test_perplexity_percent'].max():.4f}%。三 seed 波动描述重复训练敏感度，不等同跨 Topic 语义稳定性。",
        "",
        "## 8. NPMI Plateau",
        "",
        f"达到全局最大值 99% 的 K：{format_k_ranges(band99)}；达到 98% 的 K：{format_k_ranges(band98)}。连续主平台标签：`{plateau_label}`；{format_k_ranges(plateau_other)}虽也达到 98%，但与主平台不连续，故不并入平台名称。这些阈值表示近峰区间，不把极小数值差自动解释为显著优劣。",
        "",
        "## 9. Perplexity Trend",
        "",
        f"K=2–30 的判断为 `{artifacts['perplexity_shape']}`；所有真实一阶差分见 `k_scan_first_differences.csv`，3-point moving average 只在汇总表作为辅助，不替换原始均值。NPMI 与 perplexity 回答不同问题：前者衡量主题内部词语一致性，后者衡量对新诗的预测拟合，因此最优 K 不同并非实验失败。",
        "",
        "## 10. Candidate K Window",
        "",
        f"去重候选为：{format_k_ranges(candidates['K'].astype(int).tolist())}。新增 quantitative candidate 为：{format_k_ranges(new_candidates['K'].astype(int).tolist())}。`TIER_1` 要求进入 NPMI top-1% 且 NPMI SD 不高于全 K 中位数；其他由定量极值触发者为 `TIER_2`；仅因历史保留的 K15/K20 为 `REFERENCE`。候选数不作机械限制。",
        "",
        "## 11. Comparison with K15 and K20",
        "",
        f"K15−K20 的 NPMI 差为 {row15['mean_npmi'] - row20['mean_npmi']:+.10f}；held-out perplexity 差为 {row15['mean_test_perplexity'] - row20['mean_test_perplexity']:+.6f}。这只限定定量分辨率竞争关系；不修改冻结 C_K20 taxonomy、主模型或下游文学史结果。",
        "",
        "## 12. Prior Audit",
        "",
        f"实际参数：`learning_method={audited_params['learning_method']}`、`max_iter={audited_params['max_iter']}`、`learning_decay={audited_params['learning_decay']}`、`learning_offset={audited_params['learning_offset']}`、`max_doc_update_iter={audited_params['max_doc_update_iter']}`、`mean_change_tol={audited_params['mean_change_tol']}`、`evaluate_every={audited_params['evaluate_every']}`、`perp_tol={audited_params['perp_tol']}`、`n_jobs={audited_params['n_jobs']}`。{prior_note}主实验不修改 alpha/eta。若单调趋势很强，后续可另做 fixed-prior sensitivity；本轮未自动启动。由于 `evaluate_every=-1`，sklearn 不执行外层 perplexity 早停，`converged_flag=False` 只表示达到 30 轮上限，不能证明数学意义上的未收敛。",
        "",
        "## 13. Limitations",
        "",
        "本实验沿用冻结全语料词表，测试诗虽未参与 LDA fit，但词表预处理层来自全语料；held-out perplexity 是 sklearn 变分下界近似，不是 document-completion perplexity。NPMI 只看 top-20 文档共现，未做人工 Topic 标签、代表诗细读、文学解释或时期分析。三 seed 的指标 SD 也不能代替跨 seed Topic matching。",
        "",
        "## 14. Next-step Recommendation",
        "",
        f"正式建议：`{recommendation}`。下一阶段只应对本次候选或平台内 K 做定向人文学语义审核，并与 K15、冻结 K20 同表比较；在审核前不重新冻结主模型。",
        "",
    ]
    (output / "QTS_Corpus_C_K2_30_Exhaustive_K_Scan_Report.md").write_text("\n".join(lines), encoding="utf-8")
    readme = [
        "# C Corpus K=2–30 穷举诊断",
        "",
        "本目录保存同一冻结 80/20 split 上的 87 个 train-only LDA runs、逐 K 汇总、局部极值、NPMI 平台、候选 K 和六张诊断图。",
        "",
        "复现命令：",
        "",
        "```powershell",
        "python src/run_exhaustive_k_scan.py",
        "```",
        "",
        "程序支持检查点续跑，不覆盖第一轮 full-corpus 模型、冻结 C_K20 或旧 NPMI。权威结论见 `QTS_Corpus_C_K2_30_Exhaustive_K_Scan_Report.md`。",
        "",
    ]
    (output / "README.md").write_text("\n".join(readme), encoding="utf-8")


def verify(context: Mapping[str, Any], complete: bool = True) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """执行任务指定的十五项不变量检查。"""

    rows = collect_checkpoints(context)
    frame = pd.DataFrame(rows, columns=ALL_RUN_FIELDS)
    expected_pairs = set(itertools.product(REQUIRED_K, REQUIRED_SEEDS))
    actual_pairs = set(zip(frame["K"].astype(int), frame["seed"].astype(int))) if len(frame) else set()
    current_manifest = {str(path): file_sha256(path) for path in tracked_inputs(context["config"])}
    immutable = current_manifest == context["input_manifest"]
    vocab_hash = file_sha256(resolve(context["config"]["corpus_root"]) / "vocabulary.csv")
    identity = read_json(context["output"] / "audit/run_identity.json")
    params_constant = True
    fit_scope_ok = True
    npmi_scope_ok = True
    matrix_ok = True
    split_ok = True
    for topic_count, seed in actual_pairs:
        item = load_checkpoint(checkpoint_path(context["output"], topic_count, seed), checkpoint_signature(context, topic_count, seed))
        audit = item["audit"]
        params = dict(audit["model_parameters"])
        expected = lower.build_model(topic_count, seed, context["lda"]).get_params(deep=False)
        params_constant &= params == expected
        fit_scope_ok &= audit["fit_scope"] == "X_train_only" and audit["test_used_for_fit"] is False
        npmi_scope_ok &= audit["npmi_reference"] == "X_train_only" and int(audit["npmi_top_words"]) == 20
        matrix_ok &= item["matrix_sha256"] == context["config"]["expected"]["matrix_sha256"] and item["vocabulary_sha256"] == vocab_hash
        split_ok &= item["split_sha256"] == context["split_audit"]["split_sha256"] and item["fit_rows_sha256"] == context["split_audit"]["fit_rows_sha256"] and item["test_rows_sha256"] == context["split_audit"]["test_rows_sha256"]
    complete_pairs = actual_pairs == expected_pairs and len(frame) == 87
    finite_ppl = bool(len(frame) and frame["finite_perplexity_flag"].map(lambda value: value is True or str(value).lower() == "true").all() and np.isfinite(frame[["train_perplexity", "test_perplexity"]].astype(float)).all().all() and (frame[["train_perplexity", "test_perplexity"]].astype(float) > 0).all().all())
    finite_npmi = bool(len(frame) and np.isfinite(frame["npmi_coherence"].astype(float)).all())
    counts_ok = bool(len(frame) and frame.groupby("K")["seed"].nunique().eq(3).all() and set(frame["K"].astype(int)) == set(REQUIRED_K))
    checks = [
        ("TEST_01", "C corpus 不变化", immutable and matrix_ok),
        ("TEST_02", "词表不变化", immutable and vocab_hash == identity["vocabulary_sha256"]),
        ("TEST_03", "train/test split 与旧 diagnostic 完全一致", split_ok and file_sha256(resolve(context["config"]["frozen_split"])) == context["config"]["expected"]["split_sha256"]),
        ("TEST_04", "K=2–30 共29个 K", set(frame["K"].astype(int)) == set(REQUIRED_K) if len(frame) else False),
        ("TEST_05", "每K 3 seeds", counts_ok),
        ("TEST_06", "总 run 数 = 87", complete_pairs),
        ("TEST_07", "所有 test perplexity finite positive", finite_ppl),
        ("TEST_08", "所有 NPMI finite", finite_npmi),
        ("TEST_09", "test set 未参与 fit", fit_scope_ok),
        ("TEST_10", "NPMI reference corpus = train only", npmi_scope_ok),
        ("TEST_11", "所有 K 使用相同 preprocessing", matrix_ok),
        ("TEST_12", "所有 K 使用相同 split", split_ok),
        ("TEST_13", "除 K 和 seed 外 LDA 参数不变化", params_constant),
        ("TEST_14", "不覆盖原 frozen model", immutable and context["output"] not in [ROOT / "output/lda_sensitivity", ROOT / "output/lda_candidate_rebuild"]),
        ("TEST_15", "不自动修改 C_K20 主模型状态", immutable),
    ]
    payload = {"complete": complete_pairs, "n_runs": len(frame), "passed": sum(int(passed) for _, _, passed in checks), "total": len(checks), "checks": [{"test": test, "description": description, "passed": bool(passed)} for test, description, passed in checks]}
    atomic_json(context["output"] / "audit/integrity_checks.json", payload)
    if complete and (not complete_pairs or not all(passed for _, _, passed in checks)):
        raise ValueError(f"不变量检查失败：{payload}")
    return frame.sort_values(["K", "seed"]).reset_index(drop=True), payload["checks"]


def finalize(context: Mapping[str, Any]) -> dict[str, Any]:
    """完成汇总表、图、报告、哈希清单和最终核验。"""

    all_runs, _ = verify(context, complete=True)
    artifacts = summarize_runs(all_runs, context["config"])
    output = context["output"]
    atomic_frame(output / "data/k_scan_all_runs.csv", all_runs[ALL_RUN_FIELDS])
    atomic_frame(output / "data/k_scan_summary.csv", artifacts["summary"])
    atomic_frame(output / "data/k_scan_first_differences.csv", artifacts["differences"])
    atomic_frame(output / "data/k_scan_local_extrema.csv", artifacts["extrema"])
    atomic_frame(output / "data/k_scan_plateau.csv", artifacts["plateau"])
    atomic_frame(output / "data/candidate_k_from_exhaustive_scan.csv", artifacts["candidates"])
    generate_figures(artifacts["summary"], output)
    generate_report(artifacts, all_runs, context["config"], context["audited_params"], output)
    required_outputs = [
        output / "README.md",
        output / "config/exhaustive_k_scan_config.yaml",
        output / "data/k_scan_all_runs.csv",
        output / "data/k_scan_summary.csv",
        output / "data/k_scan_first_differences.csv",
        output / "data/k_scan_local_extrema.csv",
        output / "data/k_scan_plateau.csv",
        output / "data/candidate_k_from_exhaustive_scan.csv",
        output / "figures/C_K2_30_NPMI_curve.png",
        output / "figures/C_K2_30_Perplexity_curve.png",
        output / "figures/C_K2_30_NPMI_and_Perplexity.png",
        output / "figures/C_K10_20_zoom.png",
        output / "figures/C_K2_30_first_difference.png",
        output / "figures/C_K2_30_metric_rank.png",
        output / "QTS_Corpus_C_K2_30_Exhaustive_K_Scan_Report.md",
    ]
    missing = [str(path) for path in required_outputs if not path.is_file()]
    if missing:
        raise ValueError(f"缺少要求产物：{missing}")
    atomic_json(output / "audit/output_manifest.json", {str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path) for path in required_outputs})
    verify(context, complete=True)
    return artifacts


def main() -> None:
    """命令行入口，按 K、seed 顺序训练并支持检查点续跑。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--max-new-runs", type=int)
    args = parser.parse_args()
    context = prepare(args.config)
    if args.prepare_only:
        print(json.dumps({"n_train": context["x_train"].shape[0], "n_test": context["x_test"].shape[0], "vocab_size": context["x_train"].shape[1], "split_sha256": context["split_audit"]["split_sha256"], "lda": context["audited_params"]}, ensure_ascii=False, indent=2))
        return
    if args.verify_only:
        frame, checks = verify(context, complete=True)
        print(f"核验通过：{len(frame)}/87 runs，{sum(item['passed'] for item in checks)}/15 invariants")
        return
    if args.summarize_only:
        artifacts = finalize(context)
        print(f"汇总完成：NPMI={artifacts['npmi_shape']}，Perplexity={artifacts['perplexity_shape']}")
        return

    lock = context["output"] / "audit/runner.lock"
    if lock.exists():
        raise RuntimeError(f"已有运行锁：{lock}")
    lock.write_text(str(os.getpid()), encoding="utf-8")
    new_runs = 0
    try:
        for topic_count, seed in itertools.product(REQUIRED_K, REQUIRED_SEEDS):
            path = checkpoint_path(context["output"], topic_count, seed)
            expected = checkpoint_signature(context, topic_count, seed)
            if path.is_file():
                load_checkpoint(path, expected)
                continue
            if args.max_new_runs is not None and new_runs >= args.max_new_runs:
                break
            completed_before = len(collect_checkpoints(context))
            atomic_json(context["output"] / "audit/progress.json", {"state": "RUNNING", "pid": os.getpid(), "completed": completed_before, "current": {"K": topic_count, "seed": seed}})
            print(f"开始 C K={topic_count} seed={seed}（已完成 {completed_before}/87）", flush=True)
            result, audit = evaluate_one(topic_count, seed, context["x_train"], context["x_test"], context["lda"], int(context["config"]["metrics"]["top_words"]))
            item = {
                "checkpoint_signature": expected,
                "result": result,
                "result_sha256": stable_hash(result),
                "audit": audit,
                "matrix_sha256": context["config"]["expected"]["matrix_sha256"],
                "vocabulary_sha256": context["vocabulary_sha256"],
                "split_sha256": context["split_audit"]["split_sha256"],
                "fit_rows_sha256": context["split_audit"]["fit_rows_sha256"],
                "test_rows_sha256": context["split_audit"]["test_rows_sha256"],
            }
            atomic_json(path, item)
            new_runs += 1
            rows = collect_checkpoints(context)
            write_all_runs(context["output"], rows)
            print(f"完成 {len(rows)}/87 C K={topic_count} seed={seed} NPMI={result['npmi_coherence']:.8f} testPPL={result['test_perplexity']:.6f} fit={result['fit_time_seconds']:.1f}s", flush=True)
            get_reusable_executor().shutdown(wait=True, kill_workers=True)
            gc.collect()
        rows = collect_checkpoints(context)
        if len(rows) == 87:
            artifacts = finalize(context)
            atomic_json(context["output"] / "audit/progress.json", {"state": "COMPLETE", "completed": 87, "npmi_shape": artifacts["npmi_shape"], "perplexity_shape": artifacts["perplexity_shape"]})
            print("87/87 runs 与全部汇总产物完成。", flush=True)
        else:
            verify(context, complete=False)
            atomic_json(context["output"] / "audit/progress.json", {"state": "PARTIAL", "completed": len(rows)})
            print(f"本次停止于 {len(rows)}/87 runs。", flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
