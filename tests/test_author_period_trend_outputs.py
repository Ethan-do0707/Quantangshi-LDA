"""作者时期 Topic 趋势结构正式输出测试。"""

from __future__ import annotations

import json
import unittest
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "01a05cec-c8db-7761-9760-9d2d42a4853e" / "author_period_trend_structure"


class AuthorPeriodTrendOutputTest(unittest.TestCase):
    """核验核心 CSV、图、报告与工作簿结构。"""

    def test_required_csv_shapes(self) -> None:
        """六规格、四时期与20 Topic 必须完整。"""

        means = pd.read_csv(OUTPUT / "data" / "period_topic_means.csv")
        trends = pd.read_csv(OUTPUT / "data" / "topic_trend_structure.csv")
        robust = pd.read_csv(OUTPUT / "data" / "topic_trend_robustness.csv")
        self.assertEqual(len(means), 480)
        self.assertEqual(len(trends), 120)
        self.assertEqual(len(robust), 20)
        self.assertEqual(means["specification"].nunique(), 6)

    def test_invariance_tests_all_pass(self) -> None:
        """任务指定的15项不可变性测试必须全部通过。"""

        checks = pd.read_csv(OUTPUT / "data" / "invariance_tests.csv")
        self.assertEqual(len(checks), 15)
        self.assertTrue(checks["status"].eq("PASS").all())

    def test_clear_topics_are_read_from_taxonomy(self) -> None:
        """CLEAR 输出数量必须与 taxonomy 当前内容一致。"""

        taxonomy = pd.read_csv(
            ROOT
            / "outputs"
            / "01a05061-9ee3-7610-a401-abfec81de1b2"
            / "literary_history_dataset"
            / "taxonomy"
            / "topic_metadata.csv"
        )
        clear = pd.read_csv(OUTPUT / "data" / "clear_topic_trend_structure.csv")
        expected = sorted(taxonomy.loc[taxonomy["topic_type"].eq("CLEAR"), "topic_id"])
        self.assertEqual(sorted(clear["topic_id"]), expected)

    def test_six_figures_exist_and_are_png(self) -> None:
        """六幅结构图必须存在且含 PNG 文件头。"""

        expected = [
            "core_poem_period_topic_heatmap.png",
            "core_author_period_topic_heatmap.png",
            "clear_topic_trend_small_multiples.png",
            "trend_robustness_matrix.png",
            "poem_vs_author_aggregation_difference.png",
            "period_topic_author_concentration.png",
        ]
        for name in expected:
            path = OUTPUT / "figures" / name
            self.assertTrue(path.exists(), name)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_summary_is_json_and_has_a_to_l_inputs(self) -> None:
        """机器可读摘要必须覆盖样本、peak、稳健性、CLEAR 与候选趋势。"""

        payload = json.loads((OUTPUT / "data" / "analysis_summary.json").read_text(encoding="utf-8"))
        for key in [
            "sample_summary",
            "core_peak_distributions",
            "core_poem_author_peak_agreement",
            "trend_robustness_counts",
            "clear_topics",
            "candidate_topics",
            "highest_core_author_concentration",
        ]:
            self.assertIn(key, payload)

    def test_workbook_has_twelve_named_sheets(self) -> None:
        """主工作簿必须是合法 XLSX 且包含要求的12张表。"""

        workbook = OUTPUT / "QTS_Author_Period_Trend_Structure.xlsx"
        self.assertTrue(workbook.exists())
        with zipfile.ZipFile(workbook) as archive:
            workbook_xml = archive.read("xl/workbook.xml").decode("utf-8")
        for name in [
            "Overview",
            "Period_Counts",
            "Period_Topic_Means",
            "Bootstrap_CI",
            "Trend_Structure",
            "Trend_Robustness",
            "Poem_vs_Author",
            "Author_Concentration",
            "CLEAR_Topics",
            "Candidate_Trends",
            "Sensitivity_Flags",
            "Data_Dictionary",
        ]:
            self.assertIn(f'name="{name}"', workbook_xml)


if __name__ == "__main__":
    unittest.main()
