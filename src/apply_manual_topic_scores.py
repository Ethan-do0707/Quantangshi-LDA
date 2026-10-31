"""把已完成人工判断的 Topic 评分安全写回主评分表。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCORE_FIELDS = (
    "semantic_coherence_score",
    "poem_coherence_score",
    "interpretability_score",
    "distinctiveness_score",
)

INPUT_SCORE_FIELDS = (
    "semantic",
    "poem",
    "interpretability",
    "distinctiveness",
)

MANUAL_FIELDS = (
    "proposed_label",
    *SCORE_FIELDS,
    "mean_human_score",
    "duplicate_topic_flag",
    "mixed_topic_flag",
    "noise_topic_flag",
    "notes",
)

EXPECTED_TOPIC_COUNTS = {
    "A_K20": 20,
    "B_K20": 20,
    "B_K30": 30,
}


def _load_scores(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """读取并严格验证人工评分来源。"""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("人工评分 JSON 顶层必须是列表")

    scores: dict[tuple[str, int], dict[str, Any]] = {}
    per_model: dict[str, set[int]] = {model_id: set() for model_id in EXPECTED_TOPIC_COUNTS}
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("人工评分 JSON 的每一项必须是对象")
        model_id = str(item.get("model_id", ""))
        if model_id not in EXPECTED_TOPIC_COUNTS:
            raise ValueError(f"人工评分含未知模型：{model_id}")
        topic_id = int(item["topic_id"])
        key = (model_id, topic_id)
        if key in scores:
            raise ValueError(f"人工评分主键重复：{model_id}/Topic {topic_id}")
        for field in INPUT_SCORE_FIELDS:
            value = item.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
                raise ValueError(f"{model_id}/Topic {topic_id} 的 {field} 不是 1–5 整数")
        for field in ("duplicate", "mixed", "noise"):
            if not isinstance(item.get(field), bool):
                raise ValueError(f"{model_id}/Topic {topic_id} 的 {field} 不是布尔值")
        if not str(item.get("proposed_label", "")).strip() or not str(item.get("notes", "")).strip():
            raise ValueError(f"{model_id}/Topic {topic_id} 缺少标签或评分依据")
        scores[key] = item
        per_model[model_id].add(topic_id)

    for model_id, expected_count in EXPECTED_TOPIC_COUNTS.items():
        expected_ids = set(range(1, expected_count + 1))
        if per_model[model_id] != expected_ids:
            raise ValueError(f"{model_id} 的 Topic 编号或数量不完整")
    return scores


def apply_scores(evaluation_path: Path, score_path: Path, allow_overwrite: bool) -> int:
    """仅更新评分来源中明确列出的模型和 Topic。"""

    scores = _load_scores(score_path)
    with evaluation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("主评分表缺少表头")
        missing_fields = set(MANUAL_FIELDS).difference(reader.fieldnames)
        if missing_fields:
            raise ValueError(f"主评分表缺少字段：{sorted(missing_fields)}")
        rows = list(reader)
        fieldnames = reader.fieldnames

    matched_keys: set[tuple[str, int]] = set()
    for row in rows:
        key = (row["model_id"], int(row["topic_id"]))
        item = scores.get(key)
        if item is None:
            continue
        matched_keys.add(key)
        if not allow_overwrite and any(str(row.get(field, "")).strip() for field in MANUAL_FIELDS):
            raise ValueError(f"{key[0]}/Topic {key[1]} 已有人工内容；如需覆盖请显式使用 --allow-overwrite")

        numeric_scores = [int(item[field]) for field in INPUT_SCORE_FIELDS]
        row.update(
            {
                "proposed_label": str(item["proposed_label"]),
                "semantic_coherence_score": str(numeric_scores[0]),
                "poem_coherence_score": str(numeric_scores[1]),
                "interpretability_score": str(numeric_scores[2]),
                "distinctiveness_score": str(numeric_scores[3]),
                "mean_human_score": f"{sum(numeric_scores) / 4:.2f}",
                "duplicate_topic_flag": str(bool(item["duplicate"])),
                "mixed_topic_flag": str(bool(item["mixed"])),
                "noise_topic_flag": str(bool(item["noise"])),
                "notes": str(item["notes"]),
            }
        )

    missing_keys = set(scores).difference(matched_keys)
    if missing_keys:
        first = sorted(missing_keys)[0]
        raise ValueError(f"主评分表缺少评分目标：{first[0]}/Topic {first[1]}")

    temporary_path = evaluation_path.with_suffix(evaluation_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(evaluation_path)
    return len(matched_keys)


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="把已完成人工判断的 Topic 评分写回主评分表")
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("human_topic_review/human_topic_evaluation.csv"),
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("human_topic_review/abc_human_topic_scores.json"),
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()
    updated = apply_scores(args.evaluation, args.scores, args.allow_overwrite)
    print(f"已写入 {updated} 行人工 Topic 评分：{args.evaluation}")


if __name__ == "__main__":
    main()
