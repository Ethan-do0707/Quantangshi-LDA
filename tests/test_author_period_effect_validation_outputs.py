"""作者时期效应量与统计验证正式输出测试。"""

from __future__ import annotations

import json
import unittest
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "01a05cec-c8db-7761-9760-9d2d42a4853e" / "author_period_effect_validation"


class AuthorPeriodEffectValidationOutputTest(unittest.TestCase):
    """核验 CSV、图、报告、工作簿与推断 family 范围。"""

    def test_required_table_shapes(self) -> None:
        """核心表必须覆盖9个 CLEAR、六规格和六对比。"""

        expected = {
            "clear_topic_period_omnibus.csv": 18,
            "clear_topic_primary_contrasts.csv": 36,
            "clear_topic_all_pairwise.csv": 108,
            "clear_topic_effect_sensitivity_matrix.csv": 36,
            "raw_vs_clr_direction_agreement.csv": 72,
            "all_topic_effect_sizes.csv": 720,
            "clear_topics_validation_summary.csv": 9,
            "invariance_tests.csv": 15,
        }
        for name, row_count in expected.items():
            frame = pd.read_csv(OUTPUT / "data" / name)
            self.assertEqual(len(frame), row_count, name)

    def test_formal_inference_is_core_clear_only(self) -> None:
        """p/q 值不得扩散到非 CORE 或非 CLEAR 的描述性规格。"""

        frame = pd.read_csv(OUTPUT / "data" / "all_topic_effect_sizes.csv")
        formal = frame.loc[frame["formal_inference_scope"]]
        self.assertEqual(len(formal), 108)
        self.assertTrue(formal["sample"].eq("CORE").all())
        self.assertTrue(formal["topic_type"].eq("CLEAR").all())
        nonformal = frame.loc[~frame["formal_inference_scope"]]
        self.assertTrue(nonformal["model_p"].isna().all())

    def test_primary_fdr_families_each_have_36_tests(self) -> None:
        """两套 planned contrast BH family 必须各含36项。"""

        frame = pd.read_csv(OUTPUT / "data" / "all_topic_effect_sizes.csv")
        counts = frame.loc[frame["fdr_family"].notna()].groupby("fdr_family").size().tolist()
        self.assertEqual(sorted(counts), [36, 36])

    def test_all_invariance_tests_pass(self) -> None:
        """任务指定的15项不可变性测试必须全部通过。"""

        frame = pd.read_csv(OUTPUT / "data" / "invariance_tests.csv")
        self.assertTrue(frame["status"].eq("PASS").all())

    def test_required_figures_are_png(self) -> None:
        """四张 forest 与五张综合图必须存在且为 PNG。"""

        names = [
            "effect_forest_EH.png",
            "effect_forest_HM.png",
            "effect_forest_ML.png",
            "effect_forest_EL.png",
            "clear_topic_primary_effect_forest.png",
            "clear_topic_poem_vs_author_effect.png",
            "clear_topic_six_spec_effect_heatmap.png",
            "clear_topic_validation_matrix.png",
            "raw_vs_clr_direction_agreement.png",
        ]
        for name in names:
            path = OUTPUT / "figures" / name
            self.assertTrue(path.exists(), name)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_report_has_eighteen_sections(self) -> None:
        """最终报告必须保留任务指定的18节结构。"""

        report = (OUTPUT / "QTS_Author_Period_Effect_Size_Statistical_Validation_Report.md").read_text(encoding="utf-8")
        for number in range(1, 19):
            self.assertIn(f"## {number}.", report)

    def test_workbook_has_thirteen_named_sheets(self) -> None:
        """主工作簿必须是合法 XLSX 且包含要求的13张表。"""

        workbook = OUTPUT / "QTS_Author_Period_Effect_Size_Statistical_Validation.xlsx"
        self.assertTrue(workbook.exists())
        with zipfile.ZipFile(workbook) as archive:
            xml = archive.read("xl/workbook.xml").decode("utf-8")
        for name in [
            "Overview", "Omnibus", "Primary_Contrasts", "All_Pairwise", "Effect_Sizes",
            "Bootstrap_CI", "Poem_vs_Author", "Sample_Sensitivity", "CLR_Sensitivity",
            "CLEAR_Topics", "Validation_Status", "Author_Check_Flags", "Data_Dictionary",
        ]:
            self.assertIn(f'name="{name}"', xml)

    def test_summary_has_a_to_m_inputs(self) -> None:
        """机器可读摘要必须覆盖最终 A–M 汇报所需字段。"""

        payload = json.loads((OUTPUT / "data" / "analysis_summary.json").read_text(encoding="utf-8"))
        for key in [
            "omnibus_fdr_pass", "validation_status_counts", "early_late_largest_absolute",
            "core_poem_author_sign_complete_topics", "aggregation_conflict_topics",
            "sample_sign_reversals", "raw_vs_clr_direction_disagreements", "special_topics",
            "topic_06_in_author_queue", "statistically_supported_candidate_trends",
            "author_influence_priority_topics",
        ]:
            self.assertIn(key, payload)


if __name__ == "__main__":
    unittest.main()
