"""冻结最终 LDA 模型并构建《全唐诗》文学史分析母数据集。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment


class DatasetBuildError(RuntimeError):
    """数据集构建条件不满足。"""


@dataclass
class FrozenModel:
    """保存冻结模型及其全部对齐资源。"""

    model_id: str
    role: str
    corpus_id: str
    TF: int
    DF: int
    K: int
    selected_seed: int
    directory: Path
    model: Any
    topic_word: np.ndarray
    topic_word_components: np.ndarray
    document_topic: np.ndarray
    vocabulary: list[str]
    vocabulary_rows: list[dict[str, str]]
    document_index: list[dict[str, str]]
    top_words: dict[int, list[dict[str, str]]]
    representative_poems: dict[int, list[dict[str, str]]]
    reproduction_validation: dict[str, Any]


def file_sha256(path: Path) -> str:
    """计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 UTF-8 CSV。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """以 Excel 兼容 UTF-8 CSV 原子写入数据。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    """原子写入 UTF-8 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_npz_array(path: Path, key: str) -> np.ndarray:
    """读取 NPZ 中的指定数组并关闭文件。"""

    with np.load(path) as data:
        if key not in data:
            raise DatasetBuildError(f"{path} 缺少数组键 {key}")
        return np.asarray(data[key])


def resolve_path(project_root: Path, value: str) -> Path:
    """解析项目内相对路径。"""

    path = Path(value)
    return path if path.is_absolute() else project_root / path


def format_modified_time(path: Path) -> str:
    """按本地时区格式化修改时间。"""

    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def group_by_topic(rows: Sequence[Mapping[str, str]], topic_count: int) -> dict[int, list[dict[str, str]]]:
    """按 Topic ID 和 rank 稳定分组。"""

    grouped: dict[int, list[dict[str, str]]] = {topic_id: [] for topic_id in range(1, topic_count + 1)}
    for row in rows:
        topic_id = int(row["topic_id"])
        if topic_id not in grouped:
            raise DatasetBuildError(f"发现越界 Topic ID：{topic_id}")
        grouped[topic_id].append(dict(row))
    for topic_id in grouped:
        grouped[topic_id].sort(key=lambda row: int(row["rank"]))
    return grouped


def load_human_snapshot(path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    """加载由 artifact-tool 从最新人工审核工作簿提取的快照。"""

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = payload.get("rows", [])
    if len(rows) != 140:
        raise DatasetBuildError(f"人工审核快照应有 140 行，实际 {len(rows)}")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["model_id"]), int(row["topic_id"]))
        if key in indexed:
            raise DatasetBuildError(f"人工审核重复键：{key}")
        indexed[key] = dict(row)
    for model_id, topic_count in (("B_K20", 20), ("C_K20", 20), ("D_K20", 20)):
        for topic_id in range(1, topic_count + 1):
            row = indexed.get((model_id, topic_id))
            if not row or row.get("mean_human_score") in (None, "") or not row.get("proposed_label"):
                raise DatasetBuildError(f"{model_id} Topic {topic_id} 人工审核不完整")
    return indexed, payload


def inspect_inventory_path(path: Path, human_snapshot: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """为盘点报告提取路径规模和结构信息。"""

    result: dict[str, Any] = {
        "path": path.as_posix(),
        "size_bytes": "",
        "modified_time": "",
        "row_count": "",
        "column_count": "",
        "key_fields": "",
    }
    if not path.exists():
        result["key_fields"] = "MISSING"
        return result
    if path.is_dir():
        files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
        result.update(
            size_bytes=sum(candidate.stat().st_size for candidate in files),
            modified_time=max((format_modified_time(candidate) for candidate in files), default=format_modified_time(path)),
            row_count=len(files),
            column_count="N/A",
            key_fields="directory_file_count",
        )
        return result
    result["size_bytes"] = path.stat().st_size
    result["modified_time"] = format_modified_time(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            count = sum(1 for _ in reader)
        result.update(row_count=count, column_count=len(headers), key_fields=";".join(headers))
    elif suffix == ".jsonl":
        count = 0
        keys: list[str] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                if not keys:
                    keys = list(json.loads(line).keys())
        result.update(row_count=count, column_count=len(keys), key_fields=";".join(keys))
    elif suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        keys = list(value.keys()) if isinstance(value, dict) else []
        result.update(row_count=1, column_count=len(keys), key_fields=";".join(keys))
    elif suffix == ".npz":
        with np.load(path) as data:
            shapes = {key: list(data[key].shape) for key in data.files}
        result.update(row_count="N/A", column_count="N/A", key_fields=json.dumps(shapes, ensure_ascii=False))
    elif suffix == ".xlsx" and human_snapshot is not None:
        result.update(row_count=len(human_snapshot.get("rows", [])), column_count=15, key_fields="human_topic_evaluation A:O")
    elif suffix in {".md", ".txt"}:
        count = sum(1 for _ in path.open("r", encoding="utf-8-sig"))
        result.update(row_count=count, column_count="N/A", key_fields="text")
    elif suffix == ".joblib":
        result.update(row_count="N/A", column_count="N/A", key_fields="fitted sklearn LatentDirichletAllocation")
    else:
        result.update(row_count="N/A", column_count="N/A", key_fields=suffix.lstrip(".") or "binary")
    return result


def build_inventory(
    project_root: Path,
    config: Mapping[str, Any],
    human_snapshot: Mapping[str, Any],
    output_path: Path,
) -> list[dict[str, Any]]:
    """盘点模型、审核、选型结果和 metadata 候选。"""

    candidates: list[tuple[str, Path, bool, str]] = []
    rebuild_root = resolve_path(project_root, config["rebuild_root"])
    review_root = resolve_path(project_root, config["review_root"])
    for model_id in config["models"]:
        directory = rebuild_root / model_id
        for filename in (
            "model.joblib",
            "topic_word_components.npz",
            "topic_word_probabilities.npz",
            "document_topic_probabilities.npz",
            "vocabulary.csv",
            "document_index.csv",
            "matrix_metadata.json",
            "reproduction_validation.json",
            "artifact_manifest.json",
            "topics_top30.csv",
            "representative_poems_top20.csv",
        ):
            candidates.append(("frozen_model_source", directory / filename, True, f"{model_id} 冻结或完整性验证必需"))
        for filename in (
            "topic_top_words.csv",
            "topic_top_representative_poems.csv",
            "topic_random_high_probability_poems.csv",
            "topic_prevalence.csv",
            "topic_similarity_matrix.csv",
        ):
            candidates.append(("human_review_evidence", review_root / model_id / filename, True, f"{model_id} Topic 人工复核证据"))
        candidates.append(("human_review_cards", review_root / model_id / "review_cards", True, f"{model_id} 审核卡目录"))
    candidates.append(("human_evaluation", resolve_path(project_root, config["human_evaluation_xlsx"]), True, "最新六模型人工评分总表"))
    selection_root = resolve_path(project_root, config["model_selection_root"])
    for filename in (
        "model_comprehensive_evaluation.csv",
        "model_comprehensive_evaluation.xlsx",
        "final_model_selection_report.md",
        "final_candidate_models.csv",
        "scenario_weighted_scores.csv",
        "pareto_front.csv",
        "rank_robustness.csv",
    ):
        candidates.append(("model_selection", selection_root / filename, True, "已冻结模型角色的选型依据"))
    candidates.extend(
        [
            ("metadata_candidate", resolve_path(project_root, config["metadata_jsonl"]), True, "权威 poem_id、poet_id、作者、篇名和原文来源；由 SQL 无损导出"),
            ("metadata_candidate", project_root / "tang_poetry.sql", False, "原始数据库仅含诗人、篇名、正文和技术时间戳；由 poems.jsonl 作为可直接连接的权威导出替代"),
            ("metadata_candidate", project_root / "output" / "tf_df_sensitivity" / "C" / "lda_corpus.csv", True, "C 模型清洗词元和原始 corpus row 的权威来源"),
            ("metadata_candidate", project_root / "segmentation" / "results" / "sikufenci_manual_gold.jsonl", False, "分词层仅含 poem_id 和 tokens，不含作者、篇名等主 metadata"),
            ("metadata_candidate", project_root / "output" / "lda_corpus.csv", False, "TF5/D 口径清洗语料，不是主模型 C 的输入"),
        ]
    )
    rows: list[dict[str, Any]] = []
    human_path = resolve_path(project_root, config["human_evaluation_xlsx"])
    for category, path, adopted, reason in candidates:
        details = inspect_inventory_path(path, human_snapshot if path == human_path else None)
        rows.append({"category": category, **details, "adopted": adopted, "adoption_reason": reason})
    lines = [
        "# 项目文件盘点报告",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## 权威来源选择",
        "",
        "- `segmentation/data/poems.jsonl`：作为 poem_id、poet_id、author、title、text 的 authoritative source。它是项目已验证的 SQL 无损导出，43,030 个 poem_id 可直接唯一连接。",
        "- `output/tf_df_sensitivity/C/lda_corpus.csv`：作为 C_K20 的 cleaned tokens 与原始 corpus row 来源。",
        "- `output/lda_candidate_rebuild/C_K20/document_index.csv`：作为 Document×Topic matrix_row 到 poem_id/corpus_row 的唯一对齐依据。",
        "- 当前仓库不存在可靠 volume、genre、date/year、period 或 place/location 字段；SQL 中的 created_at/updated_at 是数据库技术时间，不是诗歌创作时间，未采用。",
        "",
        "## 文件清单",
        "",
        "| 类别 | 路径 | 大小(bytes) | 修改时间 | 行数/文件数 | 列数 | 关键字段 | 采用 | 理由 |",
        "|---|---|---:|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        fields = str(row["key_fields"]).replace("|", "\\|")
        lines.append(
            f"| {row['category']} | `{row['path']}` | {row['size_bytes']} | {row['modified_time']} | {row['row_count']} | {row['column_count']} | {fields} | {row['adopted']} | {row['adoption_reason']} |"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def freeze_models(project_root: Path, output_root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """只复制三个既有模型并生成冻结清单。"""

    rebuild_root = resolve_path(project_root, config["rebuild_root"])
    freeze_root = output_root / "freeze"
    copied_files = (
        "model.joblib",
        "topic_word_components.npz",
        "topic_word_probabilities.npz",
        "document_topic_probabilities.npz",
        "vocabulary.csv",
        "document_index.csv",
        "matrix_metadata.json",
        "reproduction_validation.json",
        "artifact_manifest.json",
        "topics_top30.csv",
        "representative_poems_top20.csv",
        "reproduced_topics_top20.csv",
        "reproduced_representative_poems_top5.csv",
    )
    freeze_rows: list[dict[str, Any]] = []
    for model_id, model_config in config["models"].items():
        source = rebuild_root / model_id
        destination = freeze_root / model_id
        destination.mkdir(parents=True, exist_ok=True)
        source_hashes: dict[str, str] = {}
        destination_hashes: dict[str, str] = {}
        for filename in copied_files:
            source_path = source / filename
            if not source_path.is_file():
                raise DatasetBuildError(f"冻结源文件缺失：{source_path}")
            destination_path = destination / filename
            shutil.copy2(source_path, destination_path)
            source_hashes[filename] = file_sha256(source_path)
            destination_hashes[filename] = file_sha256(destination_path)
        if source_hashes != destination_hashes:
            raise DatasetBuildError(f"{model_id} 冻结复制哈希不一致")
        matrix_metadata = json.loads((destination / "matrix_metadata.json").read_text(encoding="utf-8-sig"))
        fitted_model = joblib.load(destination / "model.joblib")
        params = fitted_model.get_params(deep=False)
        manifest = {
            "schema_version": 1,
            "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "model_id": model_id,
            "role": model_config["role"],
            "corpus_id": model_config["corpus_id"],
            "TF_threshold": model_config["TF"],
            "DF_threshold": model_config["DF"],
            "K": model_config["K"],
            "selected_seed": matrix_metadata["selected_seed"],
            "vocabulary_size": matrix_metadata["topic_word_probabilities"]["shape"][1],
            "valid_document_count": matrix_metadata["document_topic_probabilities"]["shape"][0],
            "n_topics": matrix_metadata["document_topic_probabilities"]["shape"][1],
            "source_directory": source.as_posix(),
            "frozen_directory": destination.as_posix(),
            "model_file_path": (destination / "model.joblib").as_posix(),
            "topic_word_matrix_path": (destination / "topic_word_probabilities.npz").as_posix(),
            "topic_word_components_path": (destination / "topic_word_components.npz").as_posix(),
            "document_topic_matrix_path": (destination / "document_topic_probabilities.npz").as_posix(),
            "vocabulary_path": (destination / "vocabulary.csv").as_posix(),
            "document_index_path": (destination / "document_index.csv").as_posix(),
            "training_parameter_summary": params,
            "source_file_modified_time": {filename: format_modified_time(source / filename) for filename in copied_files},
            "source_sha256": source_hashes,
            "frozen_sha256": destination_hashes,
            "freeze_status": "passed",
            "prohibition": "自此以后，任何文学史分析不得重新拟合这些模型。",
        }
        write_json(destination / "model_manifest.json", manifest)
        freeze_rows.append(
            {
                "model_id": model_id,
                "role": model_config["role"],
                "selected_seed": matrix_metadata["selected_seed"],
                "TF": model_config["TF"],
                "DF": model_config["DF"],
                "K": model_config["K"],
                "vocabulary_size": manifest["vocabulary_size"],
                "document_count": manifest["valid_document_count"],
                "model_hash": destination_hashes["model.joblib"],
                "topic_word_hash": destination_hashes["topic_word_probabilities.npz"],
                "document_topic_hash": destination_hashes["document_topic_probabilities.npz"],
                "vocab_hash": destination_hashes["vocabulary.csv"],
                "index_hash": destination_hashes["document_index.csv"],
                "freeze_status": "passed",
            }
        )
    fields = [
        "model_id", "role", "selected_seed", "TF", "DF", "K", "vocabulary_size", "document_count",
        "model_hash", "topic_word_hash", "document_topic_hash", "vocab_hash", "index_hash", "freeze_status",
    ]
    write_csv(freeze_root / "model_freeze_manifest.csv", fields, freeze_rows)
    (freeze_root / "MODEL_FREEZE_README.md").write_text(
        "# 最终 LDA 模型冻结说明\n\n"
        "B_K20、C_K20、D_K20 已从通过精确复现验证的模型目录复制到本目录，原始文件未移动或删除。\n\n"
        "- B_K20：broader-vocabulary control model。\n"
        "- C_K20：primary model，是后续文学史研究唯一主模型。\n"
        "- D_K20：stricter-filter robustness model。\n\n"
        "**自此以后，任何文学史分析不得重新拟合这些模型。** 禁止重新训练、重新 `fit_transform`、改变 TF/DF、改变 K、平均三个模型的概率或修改人工评分。\n",
        encoding="utf-8",
    )
    return freeze_rows


def load_frozen_model(output_root: Path, model_id: str, model_config: Mapping[str, Any]) -> FrozenModel:
    """从冻结目录加载模型；只读取，不重新拟合。"""

    directory = output_root / "freeze" / model_id
    matrix_metadata = json.loads((directory / "matrix_metadata.json").read_text(encoding="utf-8-sig"))
    vocabulary_rows = read_csv(directory / "vocabulary.csv")
    vocabulary_rows.sort(key=lambda row: int(row["term_index"]))
    document_index = read_csv(directory / "document_index.csv")
    document_index.sort(key=lambda row: int(row["matrix_row"]))
    return FrozenModel(
        model_id=model_id,
        role=str(model_config["role"]),
        corpus_id=str(model_config["corpus_id"]),
        TF=int(model_config["TF"]),
        DF=int(model_config["DF"]),
        K=int(model_config["K"]),
        selected_seed=int(model_config["selected_seed"]),
        directory=directory,
        model=joblib.load(directory / "model.joblib"),
        topic_word=load_npz_array(directory / "topic_word_probabilities.npz", "probabilities"),
        topic_word_components=load_npz_array(directory / "topic_word_components.npz", "components"),
        document_topic=load_npz_array(directory / "document_topic_probabilities.npz", "probabilities"),
        vocabulary=[row["token"] for row in vocabulary_rows],
        vocabulary_rows=vocabulary_rows,
        document_index=document_index,
        top_words=group_by_topic(read_csv(directory / "topics_top30.csv"), int(model_config["K"])),
        representative_poems=group_by_topic(read_csv(directory / "representative_poems_top20.csv"), int(model_config["K"])),
        reproduction_validation=json.loads((directory / "reproduction_validation.json").read_text(encoding="utf-8-sig")),
    )


def validate_models(models: Mapping[str, FrozenModel], output_root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """执行冻结模型完整性检查，任一失败即停止后续构建。"""

    tolerance = float(config["integrity"]["row_sum_tolerance"])
    check_rows: list[dict[str, Any]] = []

    def add(model_id: str, check: str, passed: bool, observed: Any, expected: Any, details: str) -> None:
        check_rows.append(
            {
                "model_id": model_id,
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "expected": expected,
                "details": details,
            }
        )

    for model_id, model in models.items():
        metadata = json.loads((model.directory / "matrix_metadata.json").read_text(encoding="utf-8-sig"))
        add(model_id, "K_equals_20", model.K == 20 and int(model.model.n_components) == 20, f"config={model.K};model={model.model.n_components}", 20, "三个冻结模型均须为 K=20")
        expected_topic_shape = (20, len(model.vocabulary))
        add(model_id, "topic_word_shape", model.topic_word.shape == expected_topic_shape and model.topic_word_components.shape == expected_topic_shape, f"prob={model.topic_word.shape};components={model.topic_word_components.shape}", expected_topic_shape, "Topic×Word 与 components 均检查")
        expected_doc_shape = (len(model.document_index), 20)
        add(model_id, "document_topic_shape", model.document_topic.shape == expected_doc_shape, model.document_topic.shape, expected_doc_shape, "Document×Topic 行数须匹配 index")
        row_sum_error = float(np.max(np.abs(model.document_topic.sum(axis=1) - 1.0)))
        add(model_id, "document_topic_row_sum", row_sum_error < tolerance, row_sum_error, f"<{tolerance}", "每行概率和约等于1")
        min_probability = float(np.min(model.document_topic))
        max_probability = float(np.max(model.document_topic))
        add(model_id, "probability_bounds", min_probability >= 0.0 and max_probability <= 1.0, f"min={min_probability};max={max_probability}", "0<=theta<=1", "不允许负值或大于1")
        finite = bool(np.isfinite(model.document_topic).all() and np.isfinite(model.topic_word).all())
        add(model_id, "no_nan_or_inf", finite, finite, True, "Document×Topic 与 Topic×Word 均检查")
        add(model_id, "vocabulary_length", len(model.vocabulary) == model.topic_word.shape[1], len(model.vocabulary), model.topic_word.shape[1], "vocabulary.csv 长度匹配")
        add(model_id, "document_index_length", len(model.document_index) == model.document_topic.shape[0], len(model.document_index), model.document_topic.shape[0], "document_index.csv 长度匹配")
        observed_seed = int(metadata["selected_seed"])
        add(model_id, "selected_seed", observed_seed == model.selected_seed and int(model.model.random_state) == model.selected_seed, f"metadata={observed_seed};model={model.model.random_state}", model.selected_seed, "从当前冻结文件重新核验")
        expected_rows = group_by_topic(read_csv(model.directory / "reproduced_topics_top20.csv"), model.K)
        derived_match = True
        for topic_id in range(1, model.K + 1):
            indices = np.argsort(-model.topic_word[topic_id - 1], kind="stable")[:20]
            derived = [model.vocabulary[index] for index in indices]
            expected = [row["token"] for row in expected_rows[topic_id]]
            if derived != expected:
                derived_match = False
                break
        validation = model.reproduction_validation
        historical_match = bool(validation.get("exact_top20_csv_hash_match") and validation.get("exact_top5_representatives_csv_hash_match"))
        file_hash_match = file_sha256(model.directory / "reproduced_topics_top20.csv") == validation.get("original_topics_sha256")
        add(model_id, "top20_and_representative_reproduction", derived_match and historical_match and file_hash_match, f"derived={derived_match};historical={historical_match};hash={file_hash_match}", True, "重新由冻结 Topic×Word 取 Top20，并核对第一轮 hash 与 top5 代表诗复现标记")
        add(model_id, "joblib_components_exact", np.array_equal(np.asarray(model.model.components_), model.topic_word_components), bool(np.array_equal(np.asarray(model.model.components_), model.topic_word_components)), True, "joblib 模型 components_ 与 NPZ 完全一致")
    fields = ["model_id", "check", "status", "observed", "expected", "details"]
    write_csv(output_root / "freeze" / "model_integrity_check.csv", fields, check_rows)
    failures = [row for row in check_rows if row["status"] != "PASS"]
    if failures:
        raise DatasetBuildError(f"冻结模型完整性检查失败 {len(failures)} 项；已停止后续构建")
    return check_rows


def representative_summary(row: Mapping[str, str]) -> str:
    """压缩一首代表诗为可审计单元格文本。"""

    return f"{row['poem_id']} | {row['author']}《{row['title']}》 | P={float(row['topic_probability']):.6f} | {row['original_text']}"


def build_taxonomy(
    project_root: Path,
    output_root: Path,
    config: Mapping[str, Any],
    c_model: FrozenModel,
    human_rows: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """建立 C_K20 正式 Topic Taxonomy。"""

    taxonomy_config = {int(row["topic_id"]): dict(row) for row in config["taxonomy"]}
    if set(taxonomy_config) != set(range(1, 21)):
        raise DatasetBuildError("taxonomy 配置必须包含且仅包含 Topic 1–20")
    prevalence = {
        int(row["topic_id"]): row
        for row in read_csv(resolve_path(project_root, config["review_root"]) / "C_K20" / "topic_prevalence.csv")
    }
    rows: list[dict[str, Any]] = []
    for topic_number in range(1, 21):
        human = dict(human_rows[("C_K20", topic_number)])
        manual = taxonomy_config[topic_number]
        words = [row["token"] for row in c_model.top_words[topic_number]]
        representatives = c_model.representative_poems[topic_number][:5]
        row: dict[str, Any] = {
            "topic_id": f"topic_{topic_number:02d}",
            "topic_number": topic_number,
            "machine_topic_id": topic_number - 1,
            "final_topic_label": manual["final_topic_label"],
            "short_label": manual["short_label"],
            "topic_type": manual["topic_type"],
            "source_proposed_label": human["proposed_label"],
            "mean_human_score": human["mean_human_score"],
            "semantic_coherence_score": human["semantic_coherence_score"],
            "poem_coherence_score": human["poem_coherence_score"],
            "interpretability_score": human["interpretability_score"],
            "distinctiveness_score": human["distinctiveness_score"],
            "mixed_topic_flag": human["mixed_topic_flag"],
            "noise_topic_flag": human["noise_topic_flag"],
            "duplicate_topic_flag": human["duplicate_topic_flag"],
            "top_10_words": ";".join(words[:10]),
            "top_20_words": ";".join(words[:20]),
            "top_30_words": ";".join(words[:30]),
            "interpretation": manual["interpretation"],
            "interpretation_confidence": manual["interpretation_confidence"],
            "use_for_primary_interpretation": manual["use_for_primary_interpretation"],
            "interpretation_scope": manual["interpretation_scope"],
            "soft_prevalence": prevalence[topic_number]["soft_prevalence"],
            "dominant_poem_count": prevalence[topic_number]["dominant_poem_count"],
            "dominant_poem_ratio": prevalence[topic_number]["dominant_poem_ratio"],
            "notes": f"人工审核：{human['notes']}；taxonomy 边界：{manual['taxonomy_note']}",
        }
        for rank in range(1, 6):
            row[f"representative_poem_{rank}"] = representative_summary(representatives[rank - 1])
        rows.append(row)
    fields = list(rows[0].keys())
    taxonomy_root = output_root / "taxonomy"
    write_csv(taxonomy_root / "C_K20_topic_taxonomy.csv", fields, rows)
    lines = [
        "# C_K20 正式 Topic Taxonomy",
        "",
        "C_K20 是唯一 Primary Model。名称只依据冻结模型的 Top30、代表诗、随机高概率诗、人工评分、相似 Topic 与 prevalence 形成；B/D 标签不替代 C。",
        "",
        "| Topic | 正式名称 | 类型 | 人工均分 | 主解释 | 范围 |",
        "|---|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['topic_id']} | {row['final_topic_label']} | {row['topic_type']} | {float(row['mean_human_score']):.2f} | {row['use_for_primary_interpretation']} | {row['interpretation_scope']} |")
    for row in rows:
        lines.extend(
            [
                "",
                f"## {row['topic_id']}　{row['final_topic_label']}",
                "",
                f"- 类型：{row['topic_type']}；人工均分：{float(row['mean_human_score']):.2f}；解释置信度：{row['interpretation_confidence']}。",
                f"- Top30：{row['top_30_words']}。",
                f"- 解释：{row['interpretation']}",
                f"- 使用范围：{row['interpretation_scope']}；主解释：{row['use_for_primary_interpretation']}。",
                f"- 说明：{row['notes']}",
            ]
        )
    (taxonomy_root / "C_K20_topic_taxonomy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def cross_similarity_shared(anchor: FrozenModel, target: FrozenModel) -> tuple[np.ndarray, int]:
    """在两模型共享词表上计算 Topic×Word cosine。"""

    anchor_index = {token: index for index, token in enumerate(anchor.vocabulary)}
    target_index = {token: index for index, token in enumerate(target.vocabulary)}
    shared = sorted(set(anchor_index) & set(target_index))
    if not shared:
        raise DatasetBuildError(f"{anchor.model_id} 与 {target.model_id} 无共享词表")
    anchor_matrix = anchor.topic_word[:, [anchor_index[token] for token in shared]]
    target_matrix = target.topic_word[:, [target_index[token] for token in shared]]
    anchor_norm = np.linalg.norm(anchor_matrix, axis=1)
    target_norm = np.linalg.norm(target_matrix, axis=1)
    if np.any(anchor_norm == 0) or np.any(target_norm == 0):
        raise DatasetBuildError("共享词表投影后出现零 Topic 向量")
    similarity = (anchor_matrix / anchor_norm[:, None]) @ (target_matrix / target_norm[:, None]).T
    return np.clip(similarity, 0.0, 1.0), len(shared)


def top_word_metrics(first: Sequence[Mapping[str, str]], second: Sequence[Mapping[str, str]]) -> tuple[float, float, list[str]]:
    """计算 Top30 Jaccard、倒数排名加权重合与共享词。"""

    first_tokens = [row["token"] for row in first]
    second_tokens = [row["token"] for row in second]
    first_set, second_set = set(first_tokens), set(second_tokens)
    jaccard = len(first_set & second_set) / len(first_set | second_set)
    first_weights = {token: 1.0 / rank for rank, token in enumerate(first_tokens, 1)}
    second_weights = {token: 1.0 / rank for rank, token in enumerate(second_tokens, 1)}
    union = first_set | second_set
    weighted = sum(min(first_weights.get(token, 0.0), second_weights.get(token, 0.0)) for token in union) / sum(max(first_weights.get(token, 0.0), second_weights.get(token, 0.0)) for token in union)
    shared = [token for token in first_tokens if token in second_set]
    return jaccard, weighted, shared


def representative_overlap(first: Sequence[Mapping[str, str]], second: Sequence[Mapping[str, str]]) -> tuple[float, int]:
    """计算 Top20 代表诗 poem_id Jaccard。"""

    first_ids = {row["poem_id"] for row in first}
    second_ids = {row["poem_id"] for row in second}
    return len(first_ids & second_ids) / len(first_ids | second_ids), len(first_ids & second_ids)


def build_alignment(
    output_root: Path,
    config: Mapping[str, Any],
    models: Mapping[str, FrozenModel],
    taxonomy_rows: Sequence[Mapping[str, Any]],
    human_rows: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """以 C_K20 为锚点建立 B–C–D 多证据主题对齐。"""

    c_model, b_model, d_model = models["C_K20"], models["B_K20"], models["D_K20"]
    cb_similarity, cb_shared = cross_similarity_shared(c_model, b_model)
    cd_similarity, cd_shared = cross_similarity_shared(c_model, d_model)
    cb_rows, cb_columns = linear_sum_assignment(-cb_similarity)
    cd_rows, cd_columns = linear_sum_assignment(-cd_similarity)
    cb_hungarian = dict(zip((row + 1 for row in cb_rows), (column + 1 for column in cb_columns)))
    cd_hungarian = dict(zip((row + 1 for row in cd_rows), (column + 1 for column in cd_columns)))
    taxonomy_by_number = {int(row["topic_number"]): row for row in taxonomy_rows}
    evidence_rows: list[dict[str, Any]] = []
    evidence_index: dict[tuple[str, int, int], dict[str, Any]] = {}
    for target_model, target, similarity, hungarian, shared_count in (
        ("B_K20", b_model, cb_similarity, cb_hungarian, cb_shared),
        ("D_K20", d_model, cd_similarity, cd_hungarian, cd_shared),
    ):
        reciprocal = np.argmax(similarity, axis=0)
        for c_topic in range(1, 21):
            for target_topic in range(1, 21):
                jaccard, weighted, shared_words = top_word_metrics(c_model.top_words[c_topic], target.top_words[target_topic])
                poem_jaccard, shared_poems = representative_overlap(c_model.representative_poems[c_topic], target.representative_poems[target_topic])
                row = {
                    "C_topic_id": f"topic_{c_topic:02d}",
                    "target_model": target_model,
                    "target_topic_id": f"topic_{target_topic:02d}",
                    "shared_vocabulary_size": shared_count,
                    "cosine_similarity": float(similarity[c_topic - 1, target_topic - 1]),
                    "top30_jaccard": jaccard,
                    "weighted_top30_overlap": weighted,
                    "representative_poem_jaccard": poem_jaccard,
                    "shared_representative_poem_count": shared_poems,
                    "shared_top_words": ";".join(shared_words),
                    "hungarian_match": target_topic == hungarian[c_topic],
                    "reciprocal_top1": int(reciprocal[target_topic - 1]) + 1 == c_topic and int(np.argmax(similarity[c_topic - 1])) + 1 == target_topic,
                }
                evidence_rows.append(row)
                evidence_index[(target_model, c_topic, target_topic)] = row
    evidence_fields = list(evidence_rows[0].keys())
    alignment_root = output_root / "alignment"
    write_csv(alignment_root / "BCD_pairwise_alignment_evidence.csv", evidence_fields, evidence_rows)

    alignment_rows: list[dict[str, Any]] = []
    overrides = config["alignment_overrides"]
    for c_topic in range(1, 21):
        override = overrides[str(c_topic)]
        taxonomy = taxonomy_by_number[c_topic]
        row: dict[str, Any] = {
            "canonical_topic_id": taxonomy["topic_id"],
            "C_topic_id": taxonomy["topic_id"],
            "C_final_label": taxonomy["final_topic_label"],
            "C_topic_type": taxonomy["topic_type"],
        }
        for prefix, target_model in (("B", "B_K20"), ("D", "D_K20")):
            topic_ids = [int(value) for value in override[f"{prefix}_topics"]]
            primary = evidence_index[(target_model, c_topic, topic_ids[0])]
            candidate_details = []
            labels = []
            for target_topic in topic_ids:
                evidence = evidence_index[(target_model, c_topic, target_topic)]
                labels.append(str(human_rows[(target_model, target_topic)]["proposed_label"]))
                candidate_details.append(
                    f"topic_{target_topic:02d}:cos={float(evidence['cosine_similarity']):.4f},jaccard={float(evidence['top30_jaccard']):.4f},weighted={float(evidence['weighted_top30_overlap']):.4f},poem={float(evidence['representative_poem_jaccard']):.4f}"
                )
            row.update(
                {
                    f"{prefix}_topic_id": ";".join(f"topic_{topic_id:02d}" for topic_id in topic_ids),
                    f"{prefix}_original_label": ";".join(labels),
                    f"{prefix}_similarity": primary["cosine_similarity"],
                    f"{prefix}_word_overlap": primary["weighted_top30_overlap"],
                    f"{prefix}_top30_jaccard": primary["top30_jaccard"],
                    f"{prefix}_poem_overlap": primary["representative_poem_jaccard"],
                    f"{prefix}_alignment_type": override[f"{prefix}_type"],
                    f"{prefix}_alignment_confidence": override[f"{prefix}_confidence"],
                    f"{prefix}_hungarian_topic": f"topic_{(cb_hungarian if prefix == 'B' else cd_hungarian)[c_topic]:02d}",
                    f"{prefix}_candidate_details": "; ".join(candidate_details),
                }
            )
        row["semantic_stability"] = override["semantic_stability"]
        row["notes"] = override["reason"]
        alignment_rows.append(row)
    alignment_fields = list(alignment_rows[0].keys())
    write_csv(alignment_root / "BCD_cross_model_topic_alignment.csv", alignment_fields, alignment_rows)
    type_counts = Counter()
    for row in alignment_rows:
        type_counts[f"B:{row['B_alignment_type']}"] += 1
        type_counts[f"D:{row['D_alignment_type']}"] += 1
    stability_counts = Counter(row["semantic_stability"] for row in alignment_rows)
    lines = [
        "# B–C–D Cross-model Topic Alignment",
        "",
        "以 C_K20 为唯一 canonical taxonomy。相似度先投影到 pairwise shared vocabulary；Hungarian 只作为一对一基线，最终类型结合 cosine、Top30 重合、代表诗重合、人工标签和审核阅读，不强制 1:1。",
        "",
        f"- C↔B shared vocabulary：{cb_shared:,}。",
        f"- C↔D shared vocabulary：{cd_shared:,}。",
        f"- Semantic stability：{dict(stability_counts)}。",
        f"- Alignment type counts：{dict(type_counts)}。",
        "",
        "| C Topic | C 名称 | B 对应 | B 类型 | D 对应 | D 类型 | 稳定性 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in alignment_rows:
        lines.append(f"| {row['C_topic_id']} | {row['C_final_label']} | {row['B_topic_id']} | {row['B_alignment_type']} | {row['D_topic_id']} | {row['D_alignment_type']} | {row['semantic_stability']} |")
    (alignment_root / "BCD_cross_model_topic_alignment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    review_root = alignment_root / "alignment_review"
    review_root.mkdir(parents=True, exist_ok=True)
    for row in alignment_rows:
        c_topic = int(str(row["C_topic_id"]).split("_")[-1])
        review_lines = [
            f"# C Topic {c_topic:02d} Alignment Review",
            "",
            f"- Final label：{row['C_final_label']}",
            f"- Topic type：{row['C_topic_type']}",
            f"- Top30：{', '.join(item['token'] for item in c_model.top_words[c_topic])}",
            "",
            "## C Top10 Representative Poems",
            "",
        ]
        for poem in c_model.representative_poems[c_topic][:10]:
            review_lines.append(f"{poem['rank']}. {poem['poem_id']} {poem['author']}《{poem['title']}》 P={float(poem['topic_probability']):.6f} — {poem['original_text']}")
        for prefix, model_id, target_model in (("B", "B_K20", b_model), ("D", "D_K20", d_model)):
            review_lines.extend(["", f"## {prefix} 对应 Topic", ""])
            for topic_text in str(row[f"{prefix}_topic_id"]).split(";"):
                target_topic = int(topic_text.split("_")[-1])
                evidence = evidence_index[(model_id, c_topic, target_topic)]
                review_lines.extend(
                    [
                        f"### {model_id} Topic {target_topic:02d} — {human_rows[(model_id, target_topic)]['proposed_label']}",
                        "",
                        f"- Cosine：{float(evidence['cosine_similarity']):.6f}",
                        f"- Top30 Jaccard：{float(evidence['top30_jaccard']):.6f}",
                        f"- Weighted overlap：{float(evidence['weighted_top30_overlap']):.6f}",
                        f"- Representative poem Jaccard：{float(evidence['representative_poem_jaccard']):.6f}",
                        f"- Top30：{', '.join(item['token'] for item in target_model.top_words[target_topic])}",
                        "",
                    ]
                )
                for poem in target_model.representative_poems[target_topic][:10]:
                    review_lines.append(f"{poem['rank']}. {poem['poem_id']} {poem['author']}《{poem['title']}》 P={float(poem['topic_probability']):.6f} — {poem['original_text']}")
        review_lines.extend(
            [
                "",
                "## Proposed Alignment",
                "",
                f"- B：{row['B_alignment_type']} / {row['B_alignment_confidence']}",
                f"- D：{row['D_alignment_type']} / {row['D_alignment_confidence']}",
                f"- Semantic stability：{row['semantic_stability']}",
                f"- Reason：{row['notes']}",
            ]
        )
        if row["B_alignment_confidence"] == "LOW" or row["D_alignment_confidence"] == "LOW":
            review_lines.extend(["", "**NEEDS_HUMAN_REVIEW**"])
        (review_root / f"C_topic_{c_topic:02d}_alignment.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    return alignment_rows, evidence_rows


def load_metadata_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 43,030 首权威 metadata JSONL。"""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_document_outputs(
    project_root: Path,
    output_root: Path,
    config: Mapping[str, Any],
    c_model: FrozenModel,
    taxonomy_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """严格连接 C Document×Topic 与诗歌 metadata 并生成母数据集。"""

    metadata_rows = load_metadata_jsonl(resolve_path(project_root, config["metadata_jsonl"]))
    corpus_rows = read_csv(resolve_path(project_root, config["corpus_root"]) / "C" / "lda_corpus.csv")
    metadata_by_id = {int(row["id"]): row for row in metadata_rows}
    corpus_by_id = {int(row["poem_id"]): row for row in corpus_rows}
    metadata_ids = [int(row["id"]) for row in metadata_rows]
    index_ids = [int(row["poem_id"]) for row in c_model.document_index]
    duplicate_metadata = len(metadata_ids) - len(set(metadata_ids))
    duplicate_index = len(index_ids) - len(set(index_ids))
    if duplicate_metadata or duplicate_index:
        raise DatasetBuildError("poem_id 或 document index 存在重复，已停止连接")
    if len(metadata_rows) != len(corpus_rows):
        raise DatasetBuildError("权威 metadata 与 C corpus 行数不一致")
    topic_labels = {int(row["topic_number"]): str(row["final_topic_label"]) for row in taxonomy_rows}
    topic_types = {int(row["topic_number"]): str(row["topic_type"]) for row in taxonomy_rows}
    human_scores = {int(row["topic_number"]): float(row["mean_human_score"]) for row in taxonomy_rows}
    matrix_root = output_root / "matrices"
    master_root = output_root / "master"
    qc_root = output_root / "qc"
    matrix_root.mkdir(parents=True, exist_ok=True)
    master_root.mkdir(parents=True, exist_ok=True)
    qc_root.mkdir(parents=True, exist_ok=True)
    probability_fields = [f"topic_{topic_id:02d}_prob" for topic_id in range(1, 21)]
    raw_fields = ["model_row_id", "poem_id", "original_corpus_index"] + [f"topic_{topic_id:02d}" for topic_id in range(1, 21)]
    master_fields = [
        "model_id", "model_row_id", "poem_id", "original_corpus_index", "poet_id", "author", "title", "text",
        "cleaned_text", "tokens", "volume", "genre", "form", "poetic_form", "date", "year", "period",
        "author_period", "dating_method", "place", "region", "location",
    ] + probability_fields + [
        "dominant_topic_id", "dominant_topic_label", "dominant_topic_probability",
        "second_topic_id", "second_topic_label", "second_topic_probability",
        "third_topic_id", "third_topic_label", "third_topic_probability",
        "topic_entropy", "normalized_topic_entropy", "top1_top2_gap",
    ]
    summary_fields = [
        "poem_id", "author", "title", "dominant_topic_id", "dominant_topic_label", "dominant_probability",
        "second_topic_id", "second_topic_label", "second_probability", "third_topic_id", "third_topic_label",
        "third_probability", "topic_entropy", "normalized_topic_entropy", "top1_top2_gap",
    ]
    raw_path = matrix_root / "C_K20_document_topic_matrix_raw.csv"
    master_path = master_root / "C_K20_literary_history_master.csv"
    summary_path = matrix_root / "C_K20_poem_topic_summary.csv"
    raw_temp = raw_path.with_suffix(".csv.tmp")
    master_temp = master_path.with_suffix(".csv.tmp")
    summary_temp = summary_path.with_suffix(".csv.tmp")
    matched_ids: set[int] = set()
    master_row_count = 0
    column_count = len(master_fields)
    master_records_for_qc: list[dict[str, Any]] = []
    unmatched_matrix: list[dict[str, Any]] = []
    with raw_temp.open("w", encoding="utf-8-sig", newline="") as raw_handle, master_temp.open("w", encoding="utf-8-sig", newline="") as master_handle, summary_temp.open("w", encoding="utf-8-sig", newline="") as summary_handle:
        raw_writer = csv.DictWriter(raw_handle, fieldnames=raw_fields)
        master_writer = csv.DictWriter(master_handle, fieldnames=master_fields)
        summary_writer = csv.DictWriter(summary_handle, fieldnames=summary_fields)
        raw_writer.writeheader()
        master_writer.writeheader()
        summary_writer.writeheader()
        for matrix_row, index_row in enumerate(c_model.document_index):
            if int(index_row["matrix_row"]) != matrix_row:
                raise DatasetBuildError(f"document_index matrix_row 不连续：{matrix_row}")
            poem_id = int(index_row["poem_id"])
            metadata = metadata_by_id.get(poem_id)
            corpus = corpus_by_id.get(poem_id)
            if metadata is None or corpus is None:
                unmatched_matrix.append({"unmatched_type": "matrix_without_unique_metadata", "model_row_id": matrix_row, "poem_id": poem_id, "original_corpus_index": index_row.get("corpus_row", ""), "reason": "metadata or C corpus missing"})
                continue
            if str(metadata.get("poet", "")) != corpus["author"] or str(metadata.get("title", "")) != corpus["title"] or str(metadata.get("text", "")) != corpus["original_text"]:
                raise DatasetBuildError(f"poem_id={poem_id} 的 author/title/text 跨权威来源不一致")
            theta = np.asarray(c_model.document_topic[matrix_row], dtype=np.float64)
            order = np.argsort(-theta, kind="stable")[:3]
            top1, top2, top3 = (int(index) + 1 for index in order)
            positive = theta[theta > 0]
            entropy = float(-np.sum(positive * np.log(positive)))
            normalized_entropy = entropy / math.log(c_model.K)
            token_string = corpus["cleaned_tokens"]
            token_list = token_string.split() if token_string else []
            raw = {
                "model_row_id": matrix_row,
                "poem_id": poem_id,
                "original_corpus_index": int(index_row["corpus_row"]),
                **{f"topic_{topic_id:02d}": format(float(theta[topic_id - 1]), ".17g") for topic_id in range(1, 21)},
            }
            raw_writer.writerow(raw)
            master = {
                "model_id": "C_K20",
                "model_row_id": matrix_row,
                "poem_id": poem_id,
                "original_corpus_index": int(index_row["corpus_row"]),
                "poet_id": metadata.get("poet_id", ""),
                "author": metadata.get("poet", ""),
                "title": metadata.get("title", ""),
                "text": metadata.get("text", ""),
                "cleaned_text": "".join(token_list),
                "tokens": token_string,
                "volume": "",
                "genre": "",
                "form": "",
                "poetic_form": "",
                "date": "",
                "year": "",
                "period": "",
                "author_period": "",
                "dating_method": "",
                "place": "",
                "region": "",
                "location": "",
                **{probability_fields[topic_id - 1]: format(float(theta[topic_id - 1]), ".17g") for topic_id in range(1, 21)},
                "dominant_topic_id": f"topic_{top1:02d}",
                "dominant_topic_label": topic_labels[top1],
                "dominant_topic_probability": format(float(theta[top1 - 1]), ".17g"),
                "second_topic_id": f"topic_{top2:02d}",
                "second_topic_label": topic_labels[top2],
                "second_topic_probability": format(float(theta[top2 - 1]), ".17g"),
                "third_topic_id": f"topic_{top3:02d}",
                "third_topic_label": topic_labels[top3],
                "third_topic_probability": format(float(theta[top3 - 1]), ".17g"),
                "topic_entropy": format(entropy, ".17g"),
                "normalized_topic_entropy": format(normalized_entropy, ".17g"),
                "top1_top2_gap": format(float(theta[top1 - 1] - theta[top2 - 1]), ".17g"),
            }
            master_writer.writerow(master)
            summary_writer.writerow(
                {
                    "poem_id": poem_id,
                    "author": metadata.get("poet", ""),
                    "title": metadata.get("title", ""),
                    "dominant_topic_id": master["dominant_topic_id"],
                    "dominant_topic_label": master["dominant_topic_label"],
                    "dominant_probability": master["dominant_topic_probability"],
                    "second_topic_id": master["second_topic_id"],
                    "second_topic_label": master["second_topic_label"],
                    "second_probability": master["second_topic_probability"],
                    "third_topic_id": master["third_topic_id"],
                    "third_topic_label": master["third_topic_label"],
                    "third_probability": master["third_topic_probability"],
                    "topic_entropy": master["topic_entropy"],
                    "normalized_topic_entropy": master["normalized_topic_entropy"],
                    "top1_top2_gap": master["top1_top2_gap"],
                }
            )
            matched_ids.add(poem_id)
            master_row_count += 1
            master_records_for_qc.append(master)
    os.replace(raw_temp, raw_path)
    os.replace(master_temp, master_path)
    os.replace(summary_temp, summary_path)
    if unmatched_matrix:
        unmatched_fields = ["unmatched_type", "model_row_id", "poem_id", "original_corpus_index", "reason"]
        write_csv(qc_root / "unmatched_documents.csv", unmatched_fields, unmatched_matrix)
        raise DatasetBuildError(f"存在 {len(unmatched_matrix)} 个 matrix row 无法唯一对应 metadata，已停止")
    metadata_only = []
    for poem_id in sorted(set(metadata_ids) - matched_ids):
        metadata = metadata_by_id[poem_id]
        corpus = corpus_by_id[poem_id]
        metadata_only.append(
            {
                "unmatched_type": "metadata_not_in_C_matrix",
                "model_row_id": "",
                "poem_id": poem_id,
                "original_corpus_index": metadata_ids.index(poem_id),
                "author": metadata.get("poet", ""),
                "title": metadata.get("title", ""),
                "reason": "C corpus empty_after_filter; retained in metadata but excluded from BoW/Document×Topic",
                "empty_after_filter": corpus.get("empty_after_filter", ""),
            }
        )
    unmatched_fields = ["unmatched_type", "model_row_id", "poem_id", "original_corpus_index", "author", "title", "reason", "empty_after_filter"]
    write_csv(qc_root / "unmatched_documents.csv", unmatched_fields, metadata_only)
    audit = {
        "model_id": "C_K20",
        "matrix_row_count": int(c_model.document_topic.shape[0]),
        "index_row_count": len(c_model.document_index),
        "metadata_row_count": len(metadata_rows),
        "corpus_row_count": len(corpus_rows),
        "matched_count": master_row_count,
        "unmatched_matrix_rows": 0,
        "unmatched_metadata_rows": len(metadata_only),
        "duplicate_poem_id_count": duplicate_metadata,
        "duplicate_index_count": duplicate_index,
        "join_key": "poem_id from frozen document_index.csv",
        "row_inflation": master_row_count - len(c_model.document_index),
        "status": "PASS",
    }
    write_csv(qc_root / "document_alignment_audit.csv", list(audit.keys()), [audit])

    top_poem_fields = ["topic_id", "topic_label", "rank", "poem_id", "author", "title", "topic_probability", "topic_type", "human_score"]
    top_poem_rows: list[dict[str, Any]] = []
    for topic_id in range(1, 21):
        ordered = np.argsort(-c_model.document_topic[:, topic_id - 1], kind="stable")[:500]
        for rank, matrix_row in enumerate(ordered, 1):
            index_row = c_model.document_index[int(matrix_row)]
            poem_id = int(index_row["poem_id"])
            metadata = metadata_by_id[poem_id]
            top_poem_rows.append(
                {
                    "topic_id": f"topic_{topic_id:02d}",
                    "topic_label": topic_labels[topic_id],
                    "rank": rank,
                    "poem_id": poem_id,
                    "author": metadata.get("poet", ""),
                    "title": metadata.get("title", ""),
                    "topic_probability": format(float(c_model.document_topic[int(matrix_row), topic_id - 1]), ".17g"),
                    "topic_type": topic_types[topic_id],
                    "human_score": human_scores[topic_id],
                }
            )
    write_csv(matrix_root / "C_K20_topic_top_poems.csv", top_poem_fields, top_poem_rows)
    write_csv(matrix_root / "C_K20_topic_top_100_poems.csv", top_poem_fields, [row for row in top_poem_rows if int(row["rank"]) <= 100])

    qc_fields = [
        "poem_id", "poet_id", "author", "title", "text", "cleaned_text", "volume", "genre", "form",
        "poetic_form", "date", "year", "period", "author_period", "dating_method", "place", "region", "location",
    ]
    source_map = {"poem_id": "poems.jsonl:id", "poet_id": "poems.jsonl:poet_id", "author": "poems.jsonl:poet", "title": "poems.jsonl:title", "text": "poems.jsonl:text", "cleaned_text": "C/lda_corpus.csv:cleaned_tokens"}
    qc_rows: list[dict[str, Any]] = []
    for field in qc_fields:
        values = [row.get(field, "") for row in master_records_for_qc]
        nonmissing = sum(value not in (None, "") for value in values)
        unique_count = len({str(value) for value in values if value not in (None, "")})
        note = ""
        if field == "author":
            anonymous = sum(str(value) in {"佚名", "无名氏", "无名"} for value in values)
            note = f"unique_authors={unique_count};anonymous_or_unattributed={anonymous}"
        elif field in {"date", "year"}:
            note = "无诗歌创作时间；SQL created_at/updated_at 为技术时间，禁止采用"
        elif field == "period":
            note = "period metadata 尚需构建或校验；本阶段不作初盛中晚唐推断"
        elif field in {"genre", "form", "poetic_form"}:
            note = "当前权威 metadata 无文体字段"
        elif field in {"place", "region", "location"}:
            note = "当前权威 metadata 无空间字段"
        elif field == "volume":
            note = "当前 SQL/JSONL 无《全唐诗》卷次字段"
        qc_rows.append(
            {
                "field": field,
                "source_field": source_map.get(field, "absent_in_authoritative_metadata"),
                "total_rows": master_row_count,
                "nonmissing_count": nonmissing,
                "missing_count": master_row_count - nonmissing,
                "coverage_rate": nonmissing / master_row_count,
                "unique_value_count": unique_count,
                "notes": note,
            }
        )
    qc_summary_fields = list(qc_rows[0].keys())
    write_csv(qc_root / "metadata_quality_summary.csv", qc_summary_fields, qc_rows)
    qc_by_field = {row["field"]: row for row in qc_rows}
    quality_lines = [
        "# Metadata Quality Report",
        "",
        f"母数据集包含 {master_row_count:,} 首 C_K20 有效诗歌。Document×Topic 每一行均通过冻结 index 的 poem_id 唯一连接；另有 {len(metadata_only):,} 条全语料 metadata 因 C 清洗后为空而不进入模型矩阵，已列入 unmatched_documents.csv。",
        "",
        "## 权威来源",
        "",
        "- poem_id、poet_id、author、title、text：`segmentation/data/poems.jsonl`。",
        "- cleaned tokens：`output/tf_df_sensitivity/C/lda_corpus.csv`。",
        "- matrix row：冻结的 `C_K20/document_index.csv`。",
        "",
        "## 字段覆盖率",
        "",
        "| 字段 | 非缺失 | 缺失 | 覆盖率 | 唯一值 | 说明 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in qc_rows:
        quality_lines.append(f"| {row['field']} | {row['nonmissing_count']} | {row['missing_count']} | {float(row['coverage_rate']):.2%} | {row['unique_value_count']} | {row['notes']} |")
    quality_lines.extend(
        [
            "",
            "## 可用性结论",
            "",
            "- poem_id、author、title、text 与清洗词元可用于作者聚合和文本回查。",
            "- 没有可靠 date/year/period，**尚不具备时间或分期分析条件**。",
            "- 没有 genre/form/poetic_form，**尚不具备文体比较条件**。",
            "- 没有 place/region/location，**尚不具备空间分析条件**。",
            "- 作者生卒年未进入本表，也绝不能被自动当作诗歌创作年份。",
        ]
    )
    (qc_root / "metadata_quality_report.md").write_text("\n".join(quality_lines) + "\n", encoding="utf-8")
    return {
        "matrix_shape": list(c_model.document_topic.shape),
        "master_rows": master_row_count,
        "master_columns": column_count,
        "matched_count": master_row_count,
        "unmatched_matrix_rows": 0,
        "unmatched_metadata_rows": len(metadata_only),
        "metadata_quality": qc_rows,
        "author_count": int(qc_by_field["author"]["unique_value_count"]),
        "master_path": master_path.as_posix(),
    }


def build_topic_metadata(
    output_root: Path,
    taxonomy_rows: Sequence[Mapping[str, Any]],
    alignment_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """生成所有下游程序唯一使用的 Topic Metadata 主表。"""

    alignment = {str(row["canonical_topic_id"]): row for row in alignment_rows}
    rows: list[dict[str, Any]] = []
    for taxonomy in taxonomy_rows:
        topic_id = str(taxonomy["topic_id"])
        aligned = alignment[topic_id]
        rows.append(
            {
                "topic_id": topic_id,
                "final_topic_label": taxonomy["final_topic_label"],
                "short_label": taxonomy["short_label"],
                "topic_type": taxonomy["topic_type"],
                "mean_human_score": taxonomy["mean_human_score"],
                "semantic_score": taxonomy["semantic_coherence_score"],
                "poem_score": taxonomy["poem_coherence_score"],
                "interpretability_score": taxonomy["interpretability_score"],
                "distinctiveness_score": taxonomy["distinctiveness_score"],
                "mixed_flag": taxonomy["mixed_topic_flag"],
                "noise_flag": taxonomy["noise_topic_flag"],
                "duplicate_flag": taxonomy["duplicate_topic_flag"],
                "use_for_primary_interpretation": taxonomy["use_for_primary_interpretation"],
                "interpretation_scope": taxonomy["interpretation_scope"],
                "interpretation": taxonomy["interpretation"],
                "cross_model_semantic_stability": aligned["semantic_stability"],
                "B_alignment": f"{aligned['B_topic_id']}|{aligned['B_alignment_type']}|{aligned['B_alignment_confidence']}",
                "D_alignment": f"{aligned['D_topic_id']}|{aligned['D_alignment_type']}|{aligned['D_alignment_confidence']}",
            }
        )
    fields = list(rows[0].keys())
    write_csv(output_root / "taxonomy" / "topic_metadata.csv", fields, rows)
    return rows


def run_final_tests(
    output_root: Path,
    c_model: FrozenModel,
    taxonomy_rows: Sequence[Mapping[str, Any]],
    alignment_rows: Sequence[Mapping[str, Any]],
    document_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """执行用户指定的十五项最终数据集测试。"""

    matrix = c_model.document_topic
    master_rows = read_csv(output_root / "master" / "C_K20_literary_history_master.csv")
    probability_fields = [f"topic_{topic_id:02d}_prob" for topic_id in range(1, 21)]
    dominant_checks: list[bool] = []
    rank_checks: list[bool] = []
    master_probabilities_finite = True
    for row in master_rows:
        probabilities = np.asarray([float(row[field]) for field in probability_fields], dtype=np.float64)
        model_probabilities = matrix[int(row["model_row_id"])]
        if not np.isfinite(probabilities).all():
            master_probabilities_finite = False
        order = np.argsort(-model_probabilities, kind="stable")[:3]
        dominant_checks.append(
            row["dominant_topic_id"] == f"topic_{int(order[0]) + 1:02d}"
            and math.isclose(float(row["dominant_topic_probability"]), float(model_probabilities[order[0]]), abs_tol=1e-10)
        )
        rank_checks.append(
            row["second_topic_id"] == f"topic_{int(order[1]) + 1:02d}"
            and row["third_topic_id"] == f"topic_{int(order[2]) + 1:02d}"
            and math.isclose(float(row["second_topic_probability"]), float(model_probabilities[order[1]]), abs_tol=1e-10)
            and math.isclose(float(row["third_topic_probability"]), float(model_probabilities[order[2]]), abs_tol=1e-10)
        )
    topic_ids = {str(row["topic_id"]) for row in taxonomy_rows}
    alignment_references = set()
    for row in alignment_rows:
        alignment_references.add(str(row["C_topic_id"]))
        for field in ("B_topic_id", "D_topic_id"):
            alignment_references.update(str(row[field]).split(";"))
    valid_topic_ids = {f"topic_{topic_id:02d}" for topic_id in range(1, 21)}
    test_values = [
        ("TEST_01", "C_K20 = K20", c_model.K == 20, c_model.K, 20),
        ("TEST_02", "Document×Topic columns = 20", matrix.shape[1] == 20, matrix.shape[1], 20),
        ("TEST_03", "matrix rows = valid document index rows", matrix.shape[0] == len(c_model.document_index), matrix.shape[0], len(c_model.document_index)),
        ("TEST_04", "每一概率在0与1之间", bool(np.all((matrix >= 0) & (matrix <= 1))), f"min={matrix.min()};max={matrix.max()}", "0<=theta<=1"),
        ("TEST_05", "每首诗20维概率和约等于1", float(np.max(np.abs(matrix.sum(axis=1) - 1))) < 1e-6, float(np.max(np.abs(matrix.sum(axis=1) - 1))), "<1e-6"),
        ("TEST_06", "dominant probability 等于20维最大值", all(dominant_checks), sum(dominant_checks), len(master_rows)),
        ("TEST_07", "second/third topic 排序正确", all(rank_checks), sum(rank_checks), len(master_rows)),
        ("TEST_08", "poem_id 连接唯一", len({int(row["poem_id"]) for row in c_model.document_index}) == len(c_model.document_index), len({int(row["poem_id"]) for row in c_model.document_index}), len(c_model.document_index)),
        ("TEST_09", "最终 master 行数等于 valid document count", len(master_rows) == matrix.shape[0], len(master_rows), matrix.shape[0]),
        ("TEST_10", "没有 merge 导致行数膨胀", len(master_rows) == len(c_model.document_index), len(master_rows) - len(c_model.document_index), 0),
        ("TEST_11", "taxonomy 有且只有20行", len(taxonomy_rows) == 20, len(taxonomy_rows), 20),
        ("TEST_12", "每个 Topic ID 唯一", len(topic_ids) == 20, len(topic_ids), 20),
        ("TEST_13", "C_K20 topic labels 可从 topic_metadata 连接", all(row.get("final_topic_label") for row in taxonomy_rows), sum(bool(row.get("final_topic_label")) for row in taxonomy_rows), 20),
        ("TEST_14", "B/C/D alignment 引用 Topic ID 真实存在", alignment_references <= valid_topic_ids, sorted(alignment_references - valid_topic_ids), []),
        ("TEST_15", "所有概率字段无 NaN", bool(np.isfinite(matrix).all()) and master_probabilities_finite, f"matrix={bool(np.isfinite(matrix).all())};master={master_probabilities_finite}", "matrix=True;master=True"),
    ]
    rows = [
        {"test_id": test_id, "description": description, "status": "PASS" if passed else "FAIL", "observed": observed, "expected": expected}
        for test_id, description, passed, observed, expected in test_values
    ]
    write_csv(output_root / "qc" / "final_dataset_tests.csv", list(rows[0].keys()), rows)
    write_json(output_root / "qc" / "final_dataset_test_summary.json", {"status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL", "tests": rows})
    failures = [row for row in rows if row["status"] != "PASS"]
    if failures:
        raise DatasetBuildError(f"最终数据集测试失败 {len(failures)} 项")
    return rows


def write_dataset_reports(
    output_root: Path,
    freeze_rows: Sequence[Mapping[str, Any]],
    taxonomy_rows: Sequence[Mapping[str, Any]],
    alignment_rows: Sequence[Mapping[str, Any]],
    document_summary: Mapping[str, Any],
) -> None:
    """生成数据目录 README 与最终项目状态报告。"""

    taxonomy_counts = Counter(str(row["topic_type"]) for row in taxonomy_rows)
    stability_counts = Counter(str(row["semantic_stability"]) for row in alignment_rows)
    alignment_counts = Counter()
    for row in alignment_rows:
        alignment_counts[str(row["B_alignment_type"])] += 1
        alignment_counts[str(row["D_alignment_type"])] += 1
    primary_topics = [str(row["topic_id"]) for row in taxonomy_rows if bool(row["use_for_primary_interpretation"])]
    limited_topics = [str(row["topic_id"]) for row in taxonomy_rows if row["interpretation_scope"] == "limited"]
    excluded_topics = [str(row["topic_id"]) for row in taxonomy_rows if row["interpretation_scope"] == "exclude"]
    (output_root / "README.md").write_text(
        "# 《全唐诗》LDA 文学史分析母数据集\n\n"
        "本目录冻结 B_K20、C_K20、D_K20，并以 C_K20 作为唯一主模型。所有诗歌级统计必须优先使用完整 20 维 `theta_dk`；dominant topic 只用于浏览，不能把一首诗硬化为单一主题。\n\n"
        "## 功能分区\n\n"
        "- `freeze/`：模型复制、SHA-256、manifest 与完整性检查。\n"
        "- `taxonomy/`：C_K20 正式 taxonomy 和下游唯一标签主表 `topic_metadata`。\n"
        "- `alignment/`：B–C–D 共享词表相似度、多对多对齐和20份审核卡。\n"
        "- `matrices/`：原始 Document×Topic、诗到主题摘要和每 Topic Top500 反向索引。\n"
        "- `master/`：一行一首有效诗的文学史母数据集。\n"
        "- `qc/`：输入盘点、document 对齐、metadata 覆盖率和15项最终测试。\n\n"
        "## 禁止事项\n\n"
        "不得重新训练、重新 `fit_transform`、改变 TF/DF 或 K、修改人工评分、平均 B/C/D 概率、按 Topic ID 直接跨模型对应、用作者加篇名代替 poem_id、补造年代/分期/文体/空间字段，或在本阶段生成文学史趋势结论。\n",
        encoding="utf-8",
    )
    hashes = "\n".join(f"- {row['model_id']}：`{row['model_hash']}`" for row in freeze_rows)
    labels = "\n".join(f"{row['topic_id']}. {row['final_topic_label']}（{row['topic_type']}）" for row in taxonomy_rows)
    qc_map = {row["field"]: row for row in document_summary["metadata_quality"]}
    report_lines = [
        "# 文学史母数据集构建报告",
        "",
        "## 1. 冻结结果",
        "",
        "B_K20、C_K20、D_K20 均已成功复制冻结，原始模型未移动或删除；所有完整性检查通过。三个模型的 model.joblib SHA-256：",
        "",
        hashes,
        "",
        "## 2. C_K20 20-Topic Taxonomy",
        "",
        labels,
        "",
        f"类型计数：{dict(taxonomy_counts)}。可进入 primary/secondary interpretation：{', '.join(primary_topics)}；limited：{', '.join(limited_topics)}；exclude：{', '.join(excluded_topics)}。",
        "",
        "## 3. B–C–D Alignment",
        "",
        f"两侧共40个 alignment 的类型计数：{dict(alignment_counts)}。Semantic stability：{dict(stability_counts)}。",
        "",
        "ROBUST：" + ", ".join(str(row["C_topic_id"]) for row in alignment_rows if row["semantic_stability"] == "ROBUST") + "。",
        "PARTIALLY_ROBUST：" + ", ".join(str(row["C_topic_id"]) for row in alignment_rows if row["semantic_stability"] == "PARTIALLY_ROBUST") + "。",
        "PARAMETER_SENSITIVE：" + ", ".join(str(row["C_topic_id"]) for row in alignment_rows if row["semantic_stability"] == "PARAMETER_SENSITIVE") + "。",
        "",
        "## 4. Document×Topic 与母数据集",
        "",
        f"C_K20 Document×Topic shape 为 {document_summary['matrix_shape'][0]:,}×{document_summary['matrix_shape'][1]}。成功通过 poem_id 唯一匹配 {document_summary['matched_count']:,} 首；unmatched matrix rows={document_summary['unmatched_matrix_rows']}。全语料另有 {document_summary['unmatched_metadata_rows']} 首因 C 清洗后为空而未进入矩阵，已完整报告。",
        "",
        f"最终 master dataset 为 {document_summary['master_rows']:,} 行 × {document_summary['master_columns']} 列：`{document_summary['master_path']}`。完整20维概率全部保留；dominant/second/third、entropy、normalized entropy 与 top1-top2 gap 为辅助字段。",
        "",
        "## 5. Metadata QC",
        "",
        f"author 覆盖率 {float(qc_map['author']['coverage_rate']):.2%}；title {float(qc_map['title']['coverage_rate']):.2%}；volume {float(qc_map['volume']['coverage_rate']):.2%}；date/year {float(qc_map['date']['coverage_rate']):.2%}/{float(qc_map['year']['coverage_rate']):.2%}；period {float(qc_map['period']['coverage_rate']):.2%}；genre/form {float(qc_map['genre']['coverage_rate']):.2%}/{float(qc_map['form']['coverage_rate']):.2%}；location {float(qc_map['location']['coverage_rate']):.2%}。",
        "",
        "- author/title/text 完整，可直接开展作者级描述与文本回查。",
        "- 没有可靠 date/year/period，尚不能开展时间趋势或初盛中晚唐分析。",
        "- 没有 genre/form/poetic_form，尚不能开展文体比较。",
        "- 没有 place/region/location，尚不能开展空间分析。",
        "",
        "## 6. 数据风险与下一阶段边界",
        "",
        "1. poem_id=1 的原文与分词来源仍存在“连薨/连甍”既知异文，母表保留权威 SQL 导出原文，模型概率保持冻结结果。",
        "2. 15 首清洗后空诗不在 C_K20 Document×Topic 中，未静默删除，已列入 unmatched_documents.csv。",
        "3. Topic 01、09、15 为 NOISE；Topic 02、13、14、18 为 MIXED，不得与 CLEAR Topic 等量齐观。",
        "4. Cross-model alignment 是 topic-level semantic robustness，不是 historical trend robustness。",
        "5. alignment 中 LOW confidence 项已在审核包标记 NEEDS_HUMAN_REVIEW。",
        "6. 下一阶段可直接进行作者聚合、Topic 个案取诗与概率/熵描述；时间、文体和空间研究必须先补充并校验相应 metadata。",
    ]
    (output_root / "literary_history_dataset_build_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def build_dataset(project_root: Path, config_path: Path, human_snapshot_path: Path) -> dict[str, Any]:
    """执行完整构建；冻结校验失败时不会继续生成母数据集。"""

    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    output_root = resolve_path(project_root, config["output_root"])
    for directory in ("freeze", "taxonomy", "alignment", "matrices", "master", "qc"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    human_rows, human_snapshot = load_human_snapshot(human_snapshot_path)
    snapshot_copy = output_root / "qc" / "human_topic_evaluation_snapshot.json"
    shutil.copy2(human_snapshot_path, snapshot_copy)
    inventory = build_inventory(project_root, config, human_snapshot, output_root / "qc" / "inventory_report.md")
    freeze_rows = freeze_models(project_root, output_root, config)
    models = {model_id: load_frozen_model(output_root, model_id, model_config) for model_id, model_config in config["models"].items()}
    integrity = validate_models(models, output_root, config)
    taxonomy_rows = build_taxonomy(project_root, output_root, config, models["C_K20"], human_rows)
    alignment_rows, evidence_rows = build_alignment(output_root, config, models, taxonomy_rows, human_rows)
    document_summary = build_document_outputs(project_root, output_root, config, models["C_K20"], taxonomy_rows)
    topic_metadata = build_topic_metadata(output_root, taxonomy_rows, alignment_rows)
    final_tests = run_final_tests(output_root, models["C_K20"], taxonomy_rows, alignment_rows, document_summary)
    write_dataset_reports(output_root, freeze_rows, taxonomy_rows, alignment_rows, document_summary)
    build_metadata = {
        "schema_version": 1,
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": project_root.as_posix(),
        "config_path": config_path.as_posix(),
        "human_evaluation_xlsx": resolve_path(project_root, config["human_evaluation_xlsx"]).as_posix(),
        "human_evaluation_xlsx_sha256": file_sha256(resolve_path(project_root, config["human_evaluation_xlsx"])),
        "human_snapshot_sha256": file_sha256(snapshot_copy),
        "inventory_rows": len(inventory),
        "freeze_models": len(freeze_rows),
        "integrity_checks": len(integrity),
        "taxonomy_rows": len(taxonomy_rows),
        "alignment_rows": len(alignment_rows),
        "pairwise_evidence_rows": len(evidence_rows),
        "topic_metadata_rows": len(topic_metadata),
        "final_tests": len(final_tests),
        "document_summary": document_summary,
        "prohibitions": ["no_fit", "no_fit_transform", "no_retraining", "no_score_changes", "no_TF_DF_K_changes", "no_literary_history_analysis"],
    }
    write_json(output_root / "build_metadata.json", build_metadata)
    return {"output_root": output_root.as_posix(), **build_metadata}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("config/literary_history_dataset_config.json"))
    parser.add_argument("--human-snapshot", type=Path, required=True, help="由 artifact-tool 从最新人工审核 XLSX 提取的 JSON 快照")
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""

    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    result = build_dataset(project_root, config_path, args.human_snapshot.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
