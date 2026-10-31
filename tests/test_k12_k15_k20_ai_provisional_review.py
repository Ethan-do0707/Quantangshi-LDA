"""K12/K15/K20 AI 暂定逐卡审读产物验收。"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unittest
from collections import Counter
from pathlib import Path

import openpyxl


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "model_selection/K12_K15_K20_human_adjudication"
AI_DIR = TASK / "AI_review"
AUDIT_DIR = AI_DIR / "audit"
ORIGINAL = TASK / "QTS_K12_K15_K20_Human_Review.xlsx"
BACKUP = TASK / "QTS_K12_K15_K20_Human_Review.before_AI_review.xlsx"
OUTPUT = TASK / "QTS_K12_K15_K20_Human_Review_AI_Filled.xlsx"

ORIGINAL_SHEETS = [
    "Instructions",
    "Blind_Topic_Review",
    "Blind_Poem_Evidence",
    "Model_Level_Review",
    "K12_to_K15_Splits",
    "K15_to_K20_Splits",
    "Fragmentation_Review",
    "Semantic_Family_Coverage",
    "Final_Evidence_Matrix",
    "Final_Decision",
    "Hidden_AI_Reference",
]
AI_SHEETS = [
    "AI_Provisional_Review",
    "AI_Model_Summary",
    "AI_Split_Merge_Review",
    "AI_Fragmentation_Review",
    "AI_Researcher_Disagreement",
]
AI_PROVISIONAL_FIELDS = [
    "AI_Provisional_Label",
    "AI_Provisional_Word_Coherence",
    "AI_Provisional_Poem_Coherence",
    "AI_Provisional_Word_Poem_Agreement",
    "AI_Provisional_Labelability",
    "AI_Provisional_Literary_Interpretability",
    "AI_Provisional_Semantic_Specificity",
    "AI_Provisional_Overbreadth",
    "AI_Provisional_Internal_Mixedness",
    "AI_Provisional_Topic_Type",
    "AI_Provisional_Needs_Splitting",
    "AI_Provisional_Suspected_Subthemes",
    "AI_Provisional_Confidence",
    "AI_Provisional_Evidence",
    "AI_Provisional_Notes",
]


def read_csv(name: str) -> list[dict[str, str]]:
    with (AI_DIR / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nonempty(value: object) -> bool:
    return value is not None and str(value).strip() != ""


def sheet_payload(sheet: openpyxl.worksheet.worksheet.Worksheet) -> dict[str, object]:
    """抽取需要保真的单元格、批注、合并与数据验证。"""
    cells = []
    for row in sheet.iter_rows():
        for cell in row:
            if cell.value is not None or cell.comment is not None:
                cells.append(
                    (
                        cell.coordinate,
                        cell.value,
                        cell.data_type,
                        None if cell.comment is None else cell.comment.text,
                        None if cell.comment is None else cell.comment.author,
                    )
                )
    validations = []
    for validation in sheet.data_validations.dataValidation:
        validations.append(
            (
                str(validation.sqref),
                validation.type,
                validation.formula1,
                validation.formula2,
                validation.allowBlank,
                validation.operator,
            )
        )
    return {
        "state": sheet.sheet_state,
        "cells": cells,
        "merged": sorted(str(item) for item in sheet.merged_cells.ranges),
        "validations": sorted(validations, key=str),
    }


class AIProvisionalReviewTests(unittest.TestCase):
    """覆盖任务说明所列的十五项交付与完整性要求。"""

    def test_01_inventory_has_all_47_completed_cards(self) -> None:
        rows = read_csv("AI_review_card_inventory.csv")
        self.assertEqual([row["blind_topic_id"] for row in rows], [f"HR{i:03d}" for i in range(1, 48)])
        self.assertEqual({row["review_status"] for row in rows}, {"COMPLETED"})
        for field in ["card_exists", "top_words_present", "representative_poems_present", "author_balanced_poems_present"]:
            self.assertEqual({row[field] for row in rows}, {"YES"}, field)

    def test_02_each_card_contains_all_required_evidence_sections(self) -> None:
        cards = sorted((TASK / "human_review_cards").glob("HR*.md"))
        self.assertEqual(len(cards), 47)
        for card in cards:
            text = card.read_text(encoding="utf-8")
            self.assertIn("## Top30 topic words", text, card.name)
            self.assertIn("## Top20 representative poems", text, card.name)
            self.assertIn("## Top10 author-balanced representative poems", text, card.name)

    def test_03_protocol_records_blind_freeze_and_limitation(self) -> None:
        protocol = json.loads((AUDIT_DIR / "blind_review_protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["review_order"], "HR001-HR047")
        self.assertEqual(protocol["cards_completed"], 47)
        self.assertFalse(protocol["private_mapping_read_before_freeze"])
        self.assertFalse(protocol["quantitative_metrics_used_before_freeze"])
        self.assertTrue(nonempty(protocol["strict_blindness_limitation"]))

    def test_04_frozen_review_has_47_complete_provisional_rows(self) -> None:
        rows = read_csv("AI_topic_review_frozen.csv")
        self.assertEqual(len(rows), 47)
        self.assertEqual([row["blind_topic_id"] for row in rows], [f"HR{i:03d}" for i in range(1, 48)])
        for row in rows:
            for field in AI_PROVISIONAL_FIELDS:
                if field != "AI_Provisional_Suspected_Subthemes" or row["AI_Provisional_Needs_Splitting"] == "YES":
                    self.assertTrue(nonempty(row[field]), f"{row['blind_topic_id']} {field}")

    def test_05_unblinded_review_is_complete_without_researcher_fields(self) -> None:
        rows = read_csv("AI_topic_review_unblinded.csv")
        self.assertEqual(len(rows), 47)
        self.assertEqual(Counter(int(row["Unblinded_K"]) for row in rows), Counter({12: 12, 15: 15, 20: 20}))
        self.assertFalse(any(key.startswith("Researcher_") for key in rows[0]))
        for row in rows:
            required = [
                field
                for field in AI_PROVISIONAL_FIELDS
                if field != "AI_Provisional_Suspected_Subthemes" or row["AI_Provisional_Needs_Splitting"] == "YES"
            ]
            for field in [*required, "Unblinded_K", "Unblinded_Topic_ID", "AI_Challenge_Status"]:
                self.assertTrue(nonempty(row[field]), f"{row['blind_topic_id']} {field}")
            if row["AI_Challenge_Status"] != "NOT_REQUIRED":
                self.assertTrue(nonempty(row["AI_Challenge_Comment"]), row["blind_topic_id"])
                self.assertTrue(nonempty(row["AI_Missing_Semantic_Risk"]), row["blind_topic_id"])

    def test_06_original_sheets_and_researcher_fields_are_untouched(self) -> None:
        before = openpyxl.load_workbook(BACKUP, data_only=False, read_only=False)
        after = openpyxl.load_workbook(OUTPUT, data_only=False, read_only=False)
        self.assertEqual(after.sheetnames[: len(ORIGINAL_SHEETS)], ORIGINAL_SHEETS)
        for name in ORIGINAL_SHEETS:
            self.assertEqual(sheet_payload(before[name]), sheet_payload(after[name]), name)

    def test_07_frozen_review_hash_matches_record(self) -> None:
        recorded = (AUDIT_DIR / "AI_topic_review_frozen.sha256").read_text(encoding="utf-8").split()[0]
        protocol = json.loads((AUDIT_DIR / "blind_review_protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(sha256(AI_DIR / "AI_topic_review_frozen.csv"), recorded)
        self.assertEqual(recorded, protocol["frozen_csv_sha256"])

    def test_08_yes_split_rows_name_subthemes(self) -> None:
        rows = read_csv("AI_topic_review_frozen.csv")
        yes_rows = [row for row in rows if row["AI_Provisional_Needs_Splitting"] == "YES"]
        self.assertEqual(len(yes_rows), 11)
        self.assertTrue(all(nonempty(row["AI_Provisional_Suspected_Subthemes"]) for row in yes_rows))

    def test_09_mixed_and_noise_rows_have_evidence_and_notes(self) -> None:
        rows = read_csv("AI_topic_review_frozen.csv")
        risk_rows = [row for row in rows if row["AI_Provisional_Topic_Type"] in {"MIXED", "NOISE"}]
        self.assertEqual(len(risk_rows), 11)
        for row in risk_rows:
            self.assertTrue(nonempty(row["AI_Provisional_Evidence"]))
            self.assertTrue(nonempty(row["AI_Provisional_Notes"]))

    def test_10_all_split_merge_candidates_are_textually_adjudicated(self) -> None:
        rows = read_csv("AI_split_merge_review.csv")
        self.assertEqual(len(rows), 36)
        for row in rows:
            self.assertTrue(nonempty(row["AI_Provisional_Split_Judgment"]))
            self.assertTrue(nonempty(row["AI_Rationale"]))
            self.assertTrue(nonempty(row["AI_Text_Evidence"]))
            self.assertEqual(row["Quantitative_Metrics_Used_As_Decision"], "NO")

    def test_11_fragmentation_pairs_are_not_auto_decided(self) -> None:
        rows = read_csv("AI_fragmentation_review.csv")
        self.assertEqual(len(rows), 15)
        for row in rows:
            self.assertTrue(nonempty(row["AI_Provisional_Fragmentation_Status"]))
            self.assertTrue(nonempty(row["AI_Rationale"]))
            self.assertEqual(row["Quantitative_Metrics_Used_As_Decision"], "NO")

    def test_12_unblinding_did_not_change_frozen_semantic_review(self) -> None:
        frozen = {row["blind_topic_id"]: row for row in read_csv("AI_topic_review_frozen.csv")}
        unblinded = {row["blind_topic_id"]: row for row in read_csv("AI_topic_review_unblinded.csv")}
        self.assertEqual(set(frozen), set(unblinded))
        for blind_id, frozen_row in frozen.items():
            for field in AI_PROVISIONAL_FIELDS:
                self.assertEqual(frozen_row[field], unblinded[blind_id][field], f"{blind_id} {field}")

    def test_13_output_workbook_opens_with_expected_sheet_visibility(self) -> None:
        workbook = openpyxl.load_workbook(OUTPUT, data_only=False, read_only=False)
        self.assertEqual(workbook.sheetnames, ORIGINAL_SHEETS + AI_SHEETS)
        self.assertEqual([name for name in ORIGINAL_SHEETS if workbook[name].sheet_state == "visible"], ORIGINAL_SHEETS[:3])
        self.assertEqual({workbook[name].sheet_state for name in ORIGINAL_SHEETS[3:]}, {"veryHidden"})
        self.assertEqual({workbook[name].sheet_state for name in AI_SHEETS}, {"visible"})

    def test_14_new_ai_sheets_and_report_are_complete(self) -> None:
        workbook = openpyxl.load_workbook(OUTPUT, data_only=False, read_only=False)
        expected_rows = {
            "AI_Provisional_Review": 48,
            "AI_Model_Summary": 4,
            "AI_Split_Merge_Review": 37,
            "AI_Fragmentation_Review": 16,
            "AI_Researcher_Disagreement": 3,
        }
        for name, count in expected_rows.items():
            self.assertEqual(workbook[name].max_row, count, name)
        disagreement_headers = [cell.value for cell in workbook["AI_Researcher_Disagreement"][1]]
        self.assertEqual(disagreement_headers, ["blind_topic_id", "metric", "AI_value", "Researcher_value", "agreement_status"])
        report = (AI_DIR / "QTS_K12_K15_K20_AI_Provisional_Card_Review.md").read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"^##\s+\d+\.", report, flags=re.MULTILINE)), 15)
        for name in [
            "AI_model_summary.csv",
            "AI_split_merge_review.csv",
            "AI_fragmentation_review.csv",
            "AI_generated_researcher_priority_queue.csv",
        ]:
            self.assertTrue((AI_DIR / name).is_file(), name)

    def test_15_backup_is_exact_and_build_has_no_formula_error(self) -> None:
        baseline = json.loads((AUDIT_DIR / "workbook_preservation_baseline.json").read_text(encoding="utf-8"))
        self.assertEqual(sha256(ORIGINAL), sha256(BACKUP))
        self.assertEqual(sha256(ORIGINAL), baseline["original_sha256"])
        self.assertTrue(baseline["hashes_match"])
        result = json.loads((AUDIT_DIR / "post_edit/workbook_build_result.json").read_text(encoding="utf-8"))
        formula = json.loads((AUDIT_DIR / "post_edit/workbook_formula_errors.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "AI_FILLED_WORKBOOK_CREATED")
        self.assertEqual(result["formula_error_count"], 0)
        self.assertEqual(formula["error_count"], 0)


if __name__ == "__main__":
    unittest.main()
