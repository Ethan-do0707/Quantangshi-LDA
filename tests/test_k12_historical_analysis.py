"""验证 C_K12 历史分析的冻结交付物。"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "history_analysis" / "C_K12"


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class K12HistoricalAnalysisOutputTests(unittest.TestCase):
    """核验完整重跑的结构、关键数字和冻结状态。"""

    def test_blind_manifest_is_intact(self) -> None:
        freeze = json.loads((OUT / "K12_BLIND_FREEZE.json").read_text(encoding="utf-8"))
        manifest_path = OUT / "SHA256_MANIFEST.txt"
        self.assertEqual(freeze["blind_phase_status"], "BLIND_TO_OLD_TREND_DIRECTION_COMPLETE")
        self.assertEqual(sha256_file(manifest_path), freeze["manifest_sha256"])
        lines = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(lines), 37)
        for line in lines:
            expected, relative = line.split("  ", 1)
            self.assertEqual(sha256_file(OUT / relative), expected)

    def test_completion_summary_and_qa(self) -> None:
        completion = json.loads(
            (OUT / "06_final" / "C_K12_completion_summary_A_to_Z.json").read_text(encoding="utf-8")
        )
        self.assertEqual(completion["Z_status"], "C_K12_HISTORY_ANALYSIS_COMPLETE")
        self.assertEqual(completion["B_theta_shape"], [43015, 12])
        self.assertEqual(completion["final_qa_passed"], 20)
        self.assertEqual(len(completion["X_retained_historical_candidates"]), 7)
        qa = pd.read_csv(OUT / "00_input_audit" / "C_K12_final_QA.csv")
        self.assertEqual(len(qa), 20)
        self.assertTrue(qa["status"].eq("PASS").all())

    def test_topic_roles_and_primary_family(self) -> None:
        taxonomy = pd.read_csv(OUT / "00_input_audit" / "C_K12_topic_taxonomy_frozen.csv")
        self.assertEqual(taxonomy["topic_type"].value_counts().to_dict(), {"CLEAR": 6, "BROAD": 4, "NOISE": 2})
        omnibus = pd.read_csv(OUT / "03_statistical_validation" / "C_K12_omnibus_period_effects.csv")
        self.assertEqual(set(omnibus["topic_type"]), {"CLEAR"})
        self.assertEqual(omnibus.groupby("aggregation").size().to_dict(), {"AUTHOR": 6, "POEM": 6})
        final = pd.read_csv(OUT / "04_author_influence" / "C_K12_final_candidate_trends_after_LOAO.csv")
        self.assertFalse(final["topic_id"].isin(["topic_04", "topic_07"]).any())

    def test_comparison_uses_alignment_and_has_nine_rows(self) -> None:
        comparison = pd.read_csv(OUT / "06_final" / "C_K12_vs_C_K20_finding_comparison.csv")
        self.assertEqual(len(comparison), 9)
        self.assertTrue(comparison["alignment_provenance"].str.contains("AI_PROVISIONAL").all())
        self.assertEqual(
            set(comparison["old_k20_topic_id"]),
            {"topic_03", "topic_06", "topic_07", "topic_08", "topic_11", "topic_16", "topic_19"},
        )
        self.assertFalse(comparison["comparison_category"].isna().any())

    def test_reports_and_figures_are_complete(self) -> None:
        report = (OUT / "06_final" / "QTS_C_K12_Final_Historical_Analysis.md").read_text(encoding="utf-8")
        for section in range(1, 19):
            self.assertIn(f"# {section} ", report)
        self.assertIn("author period 不等于 poem date", report)
        figure_names = {
            "C_K12_CORE_POEM_all_topics_period_trends.png",
            "C_K12_CORE_AUTHOR_all_topics_period_trends.png",
            "C_K12_CLEAR_topic_period_trends.png",
            "C_K12_POEM_vs_AUTHOR_clear_topics.png",
            "C_K12_period_topic_heatmap_POEM.png",
            "C_K12_period_topic_heatmap_AUTHOR.png",
            "C_K12_CLEAR_effect_size_forest.png",
            "C_K12_author_concentration_clear_topics.png",
        }
        actual = {path.name for path in (OUT / "05_figures").glob("*.png")}
        self.assertEqual(actual, figure_names)
        for name in figure_names:
            with Image.open(OUT / "05_figures" / name) as image:
                self.assertGreaterEqual(image.width, 1200)
                self.assertGreaterEqual(image.height, 700)

    def test_delivery_manifest_hashes_all_outputs(self) -> None:
        manifest_path = OUT / "DELIVERY_SHA256_MANIFEST.txt"
        lines = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
        current_files = {
            path.relative_to(OUT).as_posix()
            for path in OUT.rglob("*")
            if path.is_file() and path.resolve() != manifest_path.resolve()
        }
        manifested = set()
        for line in lines:
            expected, relative = line.split("  ", 1)
            manifested.add(relative)
            self.assertEqual(sha256_file(OUT / relative), expected)
        self.assertEqual(manifested, current_files)


if __name__ == "__main__":
    unittest.main()
