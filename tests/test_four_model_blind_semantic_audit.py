"""四模型盲审式语义分辨率审核的 18 项验收测试。"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from zipfile import ZipFile

import openpyxl
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model_selection/four_model_blind_semantic_audit"
DATA = OUT / "data"
AUDIT = OUT / "audit"
PRIVATE = OUT / "_private_unblind"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def blind_text_corpus() -> str:
    paths = list((OUT / "blind_review_cards").glob("*.md"))
    paths += list(DATA.glob("blind_*.csv"))
    paths += [OUT / "QTS_Four_Model_Blind_Semantic_Audit.md"]
    chunks = [path.read_text(encoding="utf-8-sig", errors="replace") for path in paths]
    workbook = openpyxl.load_workbook(
        OUT / "QTS_Four_Model_Blind_Researcher_Review.xlsx", data_only=False, read_only=True
    )
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows():
            chunks.extend(str(cell.value) for cell in row if isinstance(cell.value, str))
    return "\n".join(chunks)


def test_01_four_models_loaded() -> None:
    summary = json.loads((AUDIT / "prepare_summary.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(ROOT / "model_selection/artifact_recovery_K12_K14/data/blind_audit_ready_model_manifest.csv")
    assert summary["models_loaded"] == 4
    assert sorted(manifest.K.tolist()) == [12, 14, 15, 20]
    assert manifest.ready_for_blind_audit.all()


def test_02_no_model_retraining() -> None:
    prepare = json.loads((AUDIT / "prepare_summary.json").read_text(encoding="utf-8"))
    integrity = json.loads((AUDIT / "unblind_integrity.json").read_text(encoding="utf-8"))
    assert prepare["model_training_calls"] == 0
    assert integrity["model_training_calls"] == 0


def test_03_random_mapping_isolated() -> None:
    mapping = pd.read_csv(PRIVATE / "blind_mapping_secret.csv")
    assert list(mapping.columns) == ["blind_model_id", "true_K"]
    assert mapping.blind_model_id.tolist() == ["MODEL_A", "MODEL_B", "MODEL_C", "MODEL_D"]
    assert sorted(mapping.true_K.tolist()) == [12, 14, 15, 20]
    assert not (OUT / "blind_mapping_secret.csv").exists()


def test_04_no_true_k_in_blind_materials() -> None:
    corpus = blind_text_corpus()
    assert not re.search(r"K(?:12|14|15|20)\b", corpus, flags=re.IGNORECASE)


def test_05_no_quantitative_metrics_in_blind_materials() -> None:
    corpus = blind_text_corpus().lower()
    forbidden = ["npmi", "perplexity", "topic diversity", "cross-seed stability", "prior regime", "quantitative rank"]
    assert all(term not in corpus for term in forbidden)


def test_06_no_old_topic_labels_used() -> None:
    prepare = json.loads((AUDIT / "prepare_summary.json").read_text(encoding="utf-8"))
    baseline = pd.read_csv(PRIVATE / "input_hash_baseline.csv")
    assert prepare["legacy_review_sources_loaded"] == 0
    assert not baseline.path.str.contains("human_topic_review|lower_k_semantic_audit|topic_metadata|taxonomy", case=False).any()


def test_07_all_topics_reviewed() -> None:
    reviews = pd.read_csv(DATA / "blind_topic_ai_review.csv")
    score_columns = [column for column in reviews.columns if column.startswith("AI_") and column.endswith(("coherence", "agreement", "labelability", "interpretability", "specificity", "overbreadth", "mixedness"))]
    assert len(reviews) == 61
    assert reviews.blind_topic_id.nunique() == 61
    assert not reviews[score_columns + ["AI_topic_type", "AI_notes"]].isna().any().any()


def test_08_all_review_cards_exist() -> None:
    cards = list((OUT / "blind_review_cards").glob("card_*.md"))
    index = pd.read_csv(AUDIT / "blind_card_index.csv")
    assert len(cards) == 61
    assert len(index) == 61
    assert all((OUT / path).exists() for path in index.card_path)


def test_09_topic_order_randomized() -> None:
    index = pd.read_csv(AUDIT / "blind_card_index.csv").sort_values("card_order")
    assert index.blind_topic_id.tolist() != sorted(index.blind_topic_id.tolist())
    assert (index.blind_model_id != index.blind_model_id.shift()).sum() >= 20
    assert not index.blind_model_id.is_monotonic_increasing


def test_10_old_k20_taxonomy_not_used() -> None:
    prepare = json.loads((AUDIT / "prepare_summary.json").read_text(encoding="utf-8"))
    baseline = pd.read_csv(PRIVATE / "input_hash_baseline.csv")
    assert prepare["legacy_review_sources_loaded"] == 0
    assert not baseline.path.str.contains("taxonomy", case=False).any()


def test_11_old_k15_review_not_used() -> None:
    prepare = json.loads((AUDIT / "prepare_summary.json").read_text(encoding="utf-8"))
    baseline = pd.read_csv(PRIVATE / "input_hash_baseline.csv")
    assert prepare["legacy_review_sources_loaded"] == 0
    assert not baseline.path.str.contains("lower_k_semantic_audit|human_topic_review", case=False).any()


def test_12_researcher_fields_blank() -> None:
    workbook = openpyxl.load_workbook(OUT / "QTS_Four_Model_Blind_Researcher_Review.xlsx", data_only=False)
    sheet = workbook["Topic Review"]
    headers = [cell.value for cell in sheet[1]]
    researcher_indexes = [index + 1 for index, value in enumerate(headers) if str(value).startswith("Researcher_")]
    assert len(researcher_indexes) == 12
    assert all(sheet.cell(row=row, column=column).value in (None, "") for row in range(2, 63) for column in researcher_indexes)


def test_13_blind_results_frozen_and_hashed() -> None:
    frozen = DATA / "blind_semantic_judgment_frozen.csv"
    stored = (AUDIT / "blind_semantic_judgment_frozen.sha256").read_text(encoding="utf-8").split()[0]
    assert frozen.exists()
    assert file_sha256(frozen) == stored
    assert len(stored) == 64


def test_14_unblind_did_not_modify_blind_scores() -> None:
    integrity = json.loads((AUDIT / "unblind_integrity.json").read_text(encoding="utf-8"))
    assert integrity["blind_frozen_sha256_before"] == integrity["blind_frozen_sha256_after"]
    assert integrity["blind_scores_modified_after_unblind"] is False


def test_15_split_merge_judgments_have_text_evidence() -> None:
    judgments = pd.read_csv(DATA / "blind_split_merge_judgments.csv")
    assert set(judgments.relation_type) == {"SPLIT", "MERGE"}
    assert judgments.AI_text_evidence.str.len().ge(60).all()
    assert judgments.AI_review_status.eq("AI_PROVISIONAL").all()


def test_16_selection_not_based_on_single_metric() -> None:
    evidence = pd.read_csv(DATA / "unblinded_integrated_evidence.csv")
    required = {
        "original_NPMI_mean", "original_test_perplexity_mean", "original_matched_cosine_mean",
        "prior_robustness", "semantic_coherence_clear_broad_ratio", "mean_literary_interpretability",
        "under_split_ratio", "fragmentation_topic_ratio", "resolution_gain",
    }
    assert required.issubset(evidence.columns)
    assert evidence[list(required)].notna().all().all()


def test_17_no_arbitrary_weighted_score() -> None:
    evidence = pd.read_csv(DATA / "unblinded_integrated_evidence.csv")
    report = (OUT / "QTS_Four_Model_Unblinded_Integrated_Audit.md").read_text(encoding="utf-8")
    assert not any("weighted" in column.lower() or "综合得分" in column for column in evidence.columns)
    assert "没有通过任意权重合成为单一分数" in report


def test_18_frozen_inputs_unchanged() -> None:
    baseline = pd.read_csv(PRIVATE / "input_hash_baseline.csv")
    k20_rows = baseline[baseline.path.str.contains("output/lda_candidate_rebuild/C_K20", regex=False)]
    assert len(k20_rows) >= 4
    for row in k20_rows.itertuples(index=False):
        path = ROOT / row.path
        assert path.exists()
        assert file_sha256(path) == row.sha256


TESTS = [
    ("TEST 01", test_01_four_models_loaded),
    ("TEST 02", test_02_no_model_retraining),
    ("TEST 03", test_03_random_mapping_isolated),
    ("TEST 04", test_04_no_true_k_in_blind_materials),
    ("TEST 05", test_05_no_quantitative_metrics_in_blind_materials),
    ("TEST 06", test_06_no_old_topic_labels_used),
    ("TEST 07", test_07_all_topics_reviewed),
    ("TEST 08", test_08_all_review_cards_exist),
    ("TEST 09", test_09_topic_order_randomized),
    ("TEST 10", test_10_old_k20_taxonomy_not_used),
    ("TEST 11", test_11_old_k15_review_not_used),
    ("TEST 12", test_12_researcher_fields_blank),
    ("TEST 13", test_13_blind_results_frozen_and_hashed),
    ("TEST 14", test_14_unblind_did_not_modify_blind_scores),
    ("TEST 15", test_15_split_merge_judgments_have_text_evidence),
    ("TEST 16", test_16_selection_not_based_on_single_metric),
    ("TEST 17", test_17_no_arbitrary_weighted_score),
    ("TEST 18", test_18_frozen_inputs_unchanged),
]


def main() -> int:
    results = []
    for test_id, function in TESTS:
        try:
            function()
            results.append({"test_id": test_id, "status": "PASS", "details": function.__name__})
        except Exception as exc:  # noqa: BLE001
            results.append({"test_id": test_id, "status": "FAIL", "details": f"{type(exc).__name__}: {exc}"})
    path = AUDIT / "qa_results.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test_id", "status", "details"])
        writer.writeheader()
        writer.writerows(results)
    passed = sum(row["status"] == "PASS" for row in results)
    print(json.dumps({"passed": passed, "total": len(results), "output": str(path)}, ensure_ascii=False))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
