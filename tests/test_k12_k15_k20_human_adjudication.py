"""K12/K15/K20 研究者主导盲审准备阶段验收。"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "model_selection/K12_K15_K20_human_adjudication"
WORKBOOK = TASK / "QTS_K12_K15_K20_Human_Review.xlsx"
sys.path.insert(0, str(ROOT / "src"))
from validate_human_review import RESEARCHER_FIELDS, load_topic_review, validate_review_frame  # noqa: E402


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class HumanAdjudicationPrepareTests(unittest.TestCase):
    """验证可比性、盲态、空白人工字段与冻结门禁。"""

    def test_01_models_comparable(self) -> None:
        frame = read_csv(TASK / "data/three_model_comparability_audit.csv")
        self.assertEqual(frame.K.astype(int).tolist(), [12, 15, 20])
        self.assertEqual(frame.representative_seed.astype(int).tolist(), [101, 2026, 2026])
        for column, expected in [("comparability_status", "READY"), ("fit_scope", "FULL_CORPUS"), ("prior_regime", "ORIGINAL_PRIOR")]:
            self.assertEqual(set(frame[column]), {expected})
        self.assertEqual(frame.corpus_hash.nunique(), 1)
        self.assertEqual(frame.vocabulary_hash.nunique(), 1)
        self.assertEqual(set(frame.max_iter_convention.astype(int)), {30})

    def test_02_all_topic_evidence_present(self) -> None:
        topic = read_csv(TASK / "data/researcher_topic_review_template.csv")
        words = read_csv(TASK / "data/blind_topic_words.csv")
        poems = read_csv(TASK / "data/blind_poem_evidence.csv")
        self.assertEqual(topic.blind_topic_id.tolist(), [f"HR{index:03d}" for index in range(1, 48)])
        self.assertEqual(len(words), 1410)
        self.assertEqual(len(poems), 1410)
        self.assertTrue((words.groupby("blind_topic_id").size() == 30).all())
        self.assertEqual(set(poems.groupby(["blind_topic_id", "sample_type"]).size()), {10, 20})

    def test_03_exactly_47_cards_exist(self) -> None:
        cards = sorted((TASK / "human_review_cards").glob("HR*.md"))
        self.assertEqual([card.stem for card in cards], [f"HR{index:03d}" for index in range(1, 48)])
        for card in cards:
            text = card.read_text(encoding="utf-8")
            for heading in ["## Top30 topic words", "## Top20 representative poems", "## Top10 author-balanced representative poems"]:
                self.assertIn(heading, text)

    def test_04_blind_materials_have_no_identity_leak(self) -> None:
        pattern = re.compile(r"\bK\s*\d+\b|representative_seed|random_state|prior_regime|perplexity|NPMI|AI_provisional_label|AI_topic_type|model_identity", re.I)
        paths = list((TASK / "human_review_cards").glob("HR*.md"))
        paths += [TASK / "data/blind_topic_words.csv", TASK / "data/blind_poem_evidence.csv", TASK / "data/researcher_topic_review_template.csv"]
        for path in paths:
            self.assertIsNone(pattern.search(path.read_text(encoding="utf-8-sig")), str(path))

    def test_05_workbook_visibility_and_no_visible_ai(self) -> None:
        workbook = openpyxl.load_workbook(WORKBOOK, read_only=False, data_only=False)
        visible = [name for name in workbook.sheetnames if workbook[name].sheet_state == "visible"]
        self.assertEqual(visible, ["Instructions", "Blind_Topic_Review", "Blind_Poem_Evidence"])
        hidden = {name for name in workbook.sheetnames if workbook[name].sheet_state == "veryHidden"}
        self.assertEqual(hidden, set(workbook.sheetnames) - set(visible))
        visible_text = "\n".join(str(cell.value) for name in visible for row in workbook[name].iter_rows() for cell in row if cell.value is not None)
        self.assertNotRegex(visible_text, re.compile(r"\bK\s*\d+\b|AI_provisional|AI_topic", re.I))

    def test_06_researcher_topic_fields_blank(self) -> None:
        frame = load_topic_review(WORKBOOK)
        self.assertEqual(len(frame), 47)
        for field in RESEARCHER_FIELDS:
            self.assertTrue(frame[field].isna().all() or frame[field].astype(str).str.strip().eq("").all(), field)
        summary = json.loads((TASK / "audit/prepare_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["researcher_fields_prefilled"], 0)

    def test_07_incomplete_review_cannot_unblind(self) -> None:
        result = validate_review_frame(load_topic_review(WORKBOOK))
        self.assertEqual(result["status"], "TOPIC_REVIEW_INCOMPLETE")
        self.assertEqual(result["completed_topics"], 0)
        for name in ["researcher_topic_review_frozen.csv", "researcher_topic_review_unblinded.csv", "researcher_model_summary.csv"]:
            self.assertFalse((TASK / "data" / name).exists())

    def test_08_synthetic_complete_review_passes_gate(self) -> None:
        rows = []
        for index in range(1, 48):
            row = {"blind_topic_id": f"HR{index:03d}", "Researcher_Label": "测试标签"}
            row.update({field: 3 for field in RESEARCHER_FIELDS[1:9]})
            row.update({"Researcher_Topic_Type": "CLEAR", "Researcher_Needs_Splitting": "NO", "Researcher_Suspected_Subthemes": "", "Researcher_Confidence": "MEDIUM", "Researcher_Notes": "仅验证门禁，不写入正式产物。"})
            rows.append(row)
        frame = pd.DataFrame(rows)[["blind_topic_id", *RESEARCHER_FIELDS]]
        self.assertEqual(validate_review_frame(frame)["status"], "TOPIC_REVIEW_COMPLETE")
        self.assertNotIn("blind_topic_mapping_secret", (ROOT / "src/freeze_human_review.py").read_text(encoding="utf-8"))

    def test_09_candidate_events_are_blank_evidence(self) -> None:
        specs = [("K12_to_K15", 20, {"SPLIT": 8, "MERGE": 12}), ("K15_to_K20", 16, {"SPLIT": 8, "MERGE": 8})]
        for prefix, size, counts in specs:
            frame = read_csv(TASK / f"_private/stage2_candidates/{prefix}_split_merge_candidates.csv")
            self.assertEqual(len(frame), size)
            self.assertEqual(Counter(frame.relation_candidate), Counter(counts))
            for field in ["Researcher_Split_Judgment", "Researcher_Split_Rationale", "Researcher_Confidence"]:
                self.assertTrue(frame[field].astype(str).str.strip().eq("").all())
        frame = read_csv(TASK / "_private/stage2_candidates/K20_fragmentation_candidates.csv")
        self.assertEqual(len(frame), 15)
        for field in ["Researcher_Fragmentation_Judgment", "Researcher_Fragmentation_Rationale", "Researcher_Confidence"]:
            self.assertTrue(frame[field].astype(str).str.strip().eq("").all())

    def test_10_balance_hashes_and_no_training(self) -> None:
        poems = read_csv(TASK / "data/blind_poem_evidence.csv")
        balanced = poems[poems.sample_type.eq("AUTHOR_BALANCED_TOP10")]
        for _, group in balanced.groupby("blind_topic_id"):
            self.assertEqual(len(group), 10)
            self.assertGreaterEqual(group.author.nunique(), 5)
            self.assertLessEqual(group.author.value_counts().max(), 2)
        for row in read_csv(TASK / "_private/input_hash_baseline.csv").itertuples(index=False):
            self.assertEqual(sha256(ROOT / row.path), row.sha256, row.role)
        formula = json.loads((TASK / "audit/workbook_formula_errors.json").read_text(encoding="utf-8"))
        self.assertEqual(formula["error_count"], 0)
        source = (ROOT / "src/prepare_k12_k15_k20_human_adjudication.py").read_text(encoding="utf-8")
        self.assertNotIn(".fit(", source)


if __name__ == "__main__":
    unittest.main()
