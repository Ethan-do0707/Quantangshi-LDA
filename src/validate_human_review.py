"""验证三模型研究者 Topic 级审核是否达到冻结条件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = ROOT / "model_selection/K12_K15_K20_human_adjudication/QTS_K12_K15_K20_Human_Review.xlsx"
RESEARCHER_FIELDS = [
    "Researcher_Label",
    "Researcher_Word_Coherence",
    "Researcher_Poem_Coherence",
    "Researcher_Word_Poem_Agreement",
    "Researcher_Labelability",
    "Researcher_Literary_Interpretability",
    "Researcher_Semantic_Specificity",
    "Researcher_Overbreadth",
    "Researcher_Internal_Mixedness",
    "Researcher_Topic_Type",
    "Researcher_Needs_Splitting",
    "Researcher_Suspected_Subthemes",
    "Researcher_Confidence",
    "Researcher_Notes",
]
SCORE_FIELDS = RESEARCHER_FIELDS[1:9]
REQUIRED_UNCONDITIONAL = [field for field in RESEARCHER_FIELDS if field != "Researcher_Suspected_Subthemes"]


def is_blank(value: object) -> bool:
    return value is None or str(value).strip() == ""


def load_topic_review(workbook_path: Path) -> pd.DataFrame:
    workbook = openpyxl.load_workbook(workbook_path, data_only=False, read_only=True)
    if "Blind_Topic_Review" not in workbook.sheetnames:
        raise ValueError("工作簿缺少 Blind_Topic_Review")
    sheet = workbook["Blind_Topic_Review"]
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        raise ValueError("Blind_Topic_Review 为空")
    headers = [str(value).strip() if value is not None else "" for value in rows[0]]
    if "blind_topic_id" not in headers:
        raise ValueError("Blind_Topic_Review 缺少 blind_topic_id")
    missing_headers = [field for field in RESEARCHER_FIELDS if field not in headers]
    if missing_headers:
        raise ValueError(f"工作簿缺少研究者字段：{missing_headers}")
    frame = pd.DataFrame(rows[1:], columns=headers)
    frame = frame[frame.blind_topic_id.notna() & frame.blind_topic_id.astype(str).str.strip().ne("")].copy()
    return frame[["blind_topic_id", *RESEARCHER_FIELDS]]


def validate_review_frame(frame: pd.DataFrame) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    expected_ids = [f"HR{index:03d}" for index in range(1, 48)]
    actual_ids = frame.blind_topic_id.astype(str).tolist()
    if len(frame) != 47 or sorted(actual_ids) != expected_ids or len(set(actual_ids)) != 47:
        issues.append({"blind_topic_id": "GLOBAL", "field": "blind_topic_id", "issue": "必须恰好包含 HR001–HR047"})
    for row in frame.itertuples(index=False):
        blind_id = str(row.blind_topic_id)
        values = row._asdict()
        for field in REQUIRED_UNCONDITIONAL:
            if is_blank(values[field]):
                issues.append({"blind_topic_id": blind_id, "field": field, "issue": "REQUIRED_BLANK"})
        for field in SCORE_FIELDS:
            value = values[field]
            if is_blank(value):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                issues.append({"blind_topic_id": blind_id, "field": field, "issue": "必须为1–5整数"})
                continue
            if numeric not in {1, 2, 3, 4, 5}:
                issues.append({"blind_topic_id": blind_id, "field": field, "issue": "必须为1–5整数"})
        if not is_blank(values["Researcher_Topic_Type"]) and str(values["Researcher_Topic_Type"]).upper() not in {"CLEAR", "BROAD", "MIXED", "NOISE"}:
            issues.append({"blind_topic_id": blind_id, "field": "Researcher_Topic_Type", "issue": "非法枚举"})
        needs = str(values["Researcher_Needs_Splitting"]).upper() if not is_blank(values["Researcher_Needs_Splitting"]) else ""
        if needs and needs not in {"YES", "NO", "UNCERTAIN"}:
            issues.append({"blind_topic_id": blind_id, "field": "Researcher_Needs_Splitting", "issue": "非法枚举"})
        if needs == "YES" and is_blank(values["Researcher_Suspected_Subthemes"]):
            issues.append({"blind_topic_id": blind_id, "field": "Researcher_Suspected_Subthemes", "issue": "Needs_Splitting=YES 时必填"})
        if not is_blank(values["Researcher_Confidence"]) and str(values["Researcher_Confidence"]).upper() not in {"HIGH", "MEDIUM", "LOW"}:
            issues.append({"blind_topic_id": blind_id, "field": "Researcher_Confidence", "issue": "非法枚举"})
    completed = sum(
        all(not is_blank(row[field]) for field in REQUIRED_UNCONDITIONAL)
        and (str(row["Researcher_Needs_Splitting"]).upper() != "YES" or not is_blank(row["Researcher_Suspected_Subthemes"]))
        for row in frame.to_dict("records")
    )
    return {
        "status": "TOPIC_REVIEW_COMPLETE" if not issues else "TOPIC_REVIEW_INCOMPLETE",
        "completed_topics": completed,
        "expected_topics": 47,
        "issue_count": len(issues),
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="验证47项研究者 Topic 级审核")
    parser.add_argument("--workbook", default=str(DEFAULT_WORKBOOK))
    parser.add_argument("--output")
    args = parser.parse_args()
    result = validate_review_frame(load_topic_review(Path(args.workbook)))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["status"] == "TOPIC_REVIEW_COMPLETE" else 2)


if __name__ == "__main__":
    main()
