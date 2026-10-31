"""核验 Lower-K 语义审核交付物的完整性和内部一致性。"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model_selection" / "lower_k_semantic_audit"


def verify() -> dict[str, object]:
    """执行不修改源模型的只读核验。"""

    reviews = pd.read_csv(OUT / "data" / "lower_k_topic_ai_review.csv")
    summary = pd.read_csv(OUT / "data" / "lower_k_humanities_summary.csv")
    audit = pd.read_csv(OUT / "data" / "K15_vs_K20_semantic_resolution_audit.csv")
    coverage = pd.read_csv(OUT / "data" / "K15_core_semantic_coverage.csv")
    queue = pd.read_csv(OUT / "data" / "researcher_review_priority.csv")
    tests = pd.read_csv(OUT / "data" / "invariance_tests.csv")
    workbook_source = pd.read_csv(OUT / "data" / "researcher_review_workbook_source.csv")
    report = (OUT / "QTS_Lower_K_Humanities_Semantic_Audit_Report.md").read_text(encoding="utf-8")
    workbook_path = OUT / "QTS_Lower_K_Researcher_Review.xlsx"
    workbook_check = json.loads((OUT / "_workbook_preview" / "verification.json").read_text(encoding="utf-8"))

    assert len(reviews) == 30
    assert reviews.groupby("K").size().to_dict() == {5: 5, 10: 10, 15: 15}
    assert len(list((OUT / "review_cards").glob("K*/K*_topic_*_review.md"))) == 30
    assert len(summary) == 3
    assert len(audit) == 15
    assert len(coverage) == 9
    assert len(queue) == 30
    assert queue.priority_level.value_counts().to_dict() == {
        "Priority 1": 10,
        "Priority 2": 5,
        "Priority 3": 10,
        "Priority 4": 5,
    }
    assert len(tests) >= 10 and (tests.status == "PASS").all()
    researcher_columns = [column for column in workbook_source if column.startswith("Researcher_")]
    assert len(researcher_columns) == 9
    assert all(workbook_source[column].isna().all() for column in researcher_columns)
    assert len(re.findall(r"^# (?:[1-9]|1[0-3])\.", report, re.MULTILINE)) == 13
    assert workbook_path.is_file() and zipfile.is_zipfile(workbook_path)
    assert workbook_check["researcherFieldsBlank"] is True
    assert workbook_check["researcherBlankCells"] == 270
    assert len(list((OUT / "figures").glob("*.png"))) == 4

    result = {
        "reviews": len(reviews),
        "review_cards": 30,
        "summary_rows": len(summary),
        "K15_audit_rows": len(audit),
        "core_coverage_rows": len(coverage),
        "priority_rows": len(queue),
        "priority_counts": queue.priority_level.value_counts().to_dict(),
        "invariance_tests": f"{int((tests.status == 'PASS').sum())}/{len(tests)}",
        "report_sections": 13,
        "workbook_bytes": workbook_path.stat().st_size,
        "researcher_blank_cells": 270,
        "figures": 4,
        "status": "PASS",
    }
    (OUT / "data" / "final_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
