"""作者影响与 LOAO 正式输出测试。"""

from __future__ import annotations

import json
import hashlib
import unittest
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "01a05cec-c8db-7761-9760-9d2d42a4853e" / "author_influence_loao"


class AuthorInfluenceLoaoOutputTest(unittest.TestCase):
    """核验核心 CSV、图、报告、工作簿与不变量测试。"""

    def test_core_tables_cover_required_scope(self) -> None:
        """正式摘要必须覆盖9个 Topic×4 contrasts×2 estimands。"""

        summary = pd.read_csv(OUTPUT / "data" / "loao_summary.csv")
        influence = pd.read_csv(OUTPUT / "data" / "clear_topic_author_influence_summary.csv")
        candidates = pd.read_csv(OUTPUT / "data" / "author_validated_candidate_trends.csv")
        self.assertEqual(len(summary), 72)
        self.assertEqual(len(influence), 36)
        self.assertEqual(len(candidates), 13)

    def test_loao_unit_and_ranking(self) -> None:
        """每组 LOAO 必须逐作者记录并有唯一影响排名。"""

        frame = pd.read_csv(OUTPUT / "data" / "loao_effects.csv")
        self.assertTrue(frame["removed_author_id"].notna().all())
        counts = frame.groupby(["topic_id", "contrast", "aggregation"])["rank_by_abs_change"].agg(["min", "max", "count"])
        self.assertTrue((counts["min"] == 1).all())
        self.assertTrue((counts["max"] == counts["count"]).all())

    def test_required_exclusion_schemes_and_n(self) -> None:
        """CORE Top exclusion 必须包含两种方案及1/3/5/10。"""

        frame = pd.read_csv(OUTPUT / "data" / "top_contributor_exclusion_effects.csv")
        self.assertEqual(set(frame["exclusion_scheme"]), {"GLOBAL_TOP", "PER_PERIOD_TOP"})
        self.assertEqual(set(frame["N"]), {1, 3, 5, 10})
        self.assertEqual(len(frame), 9 * 4 * 2 * 2 * 4)

    def test_bootstrap_scope_is_supported_candidates(self) -> None:
        """关键 bootstrap 只能覆盖13组候选的Top1/3/5排除。"""

        frame = pd.read_csv(OUTPUT / "data" / "top_exclusion_bootstrap_ci.csv")
        self.assertEqual(len(frame), 13 * 2 * 2 * 3)
        self.assertEqual(set(frame["N"]), {1, 3, 5})
        self.assertTrue(frame["bootstrap_iterations"].eq(2000).all())

    def test_all_invariance_tests_pass(self) -> None:
        """任务指定的15项不变量测试必须全部通过。"""

        frame = pd.read_csv(OUTPUT / "data" / "invariance_tests.csv")
        self.assertEqual(len(frame), 15)
        self.assertTrue(frame["status"].eq("PASS").all())

    def test_required_figures_are_png(self) -> None:
        """五张指定图必须存在且为 PNG。"""

        names = [
            "loao_delta_range_clear_topics.png",
            "top_author_influence_forest.png",
            "top_contributor_exclusion_sensitivity.png",
            "aggregation_conflict_author_contributions.png",
            "concentration_vs_influence.png",
        ]
        for name in names:
            path = OUTPUT / "figures" / name
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n", name)

    def test_report_has_fifteen_sections(self) -> None:
        """最终报告必须保留任务指定的15节。"""

        report = (OUTPUT / "QTS_Author_Influence_LOAO_Report.md").read_text(encoding="utf-8")
        for number in range(1, 16):
            self.assertIn(f"## {number}.", report)

    def test_workbook_has_thirteen_named_sheets(self) -> None:
        """主工作簿必须包含要求的13张表。"""

        workbook = OUTPUT / "QTS_Author_Influence_LOAO.xlsx"
        with zipfile.ZipFile(workbook) as archive:
            xml = archive.read("xl/workbook.xml").decode("utf-8")
        for name in [
            "Overview", "Influence_Summary", "LOAO_All", "Top_Influential_Authors",
            "Top_Exclusion", "Top_Exclusion_Bootstrap", "Aggregation_Conflict",
            "T03", "T06", "T08", "T16", "Final_Assessment", "Data_Dictionary",
        ]:
            self.assertIn(f'name="{name}"', xml)

    def test_summary_has_a_to_n_inputs(self) -> None:
        """机器可读摘要必须覆盖最终 A–N 汇报所需信息。"""

        payload = json.loads((OUTPUT / "data" / "analysis_summary.json").read_text(encoding="utf-8"))
        for key in [
            "candidate_status_counts", "special_pairs", "aggregation_conflicts", "loao_sign_flips",
            "top1_stable_top3_top5_attenuated", "concentrated_but_robust",
            "author_validated_candidate_trends", "finding_level_bd_eligible",
        ]:
            self.assertIn(key, payload)

    def test_scheme_disagreements_are_explicit(self) -> None:
        """两种排除方式的差异必须保留，且保留候选通过两方案检查。"""

        frame = pd.read_csv(OUTPUT / "data/clear_topic_author_influence_summary.csv")
        candidates = pd.read_csv(OUTPUT / "data/author_validated_candidate_trends.csv")
        self.assertTrue(frame["primary_exclusion_scheme"].eq("GLOBAL_TOP").all())
        self.assertTrue(frame["exclusion_scheme_status_agreement"].eq(
            frame["author_robustness_status"].eq(frame["per_period_author_robustness_status"])).all())
        retained = candidates.loc[candidates["finding_level_bd_eligible"]]
        self.assertEqual(len(retained), 9)
        self.assertTrue(retained["per_period_author_robustness_status"].isin(["AUTHOR_ROBUST", "CONCENTRATED_BUT_ROBUST"]).all())
        self.assertEqual(candidates["author_robustness_status"].value_counts().to_dict(),
                         {"CONCENTRATED_BUT_ROBUST": 8, "AUTHOR_SENSITIVE": 4, "AUTHOR_ROBUST": 1})

    def test_top_author_alias_is_identical(self) -> None:
        """任务两处指定的 Top 作者表名称应指向相同数据。"""

        self.assertEqual((OUTPUT / "data/top_influential_authors.csv").read_bytes(),
                         (OUTPUT / "data/clear_topic_top_influential_authors.csv").read_bytes())

    def test_manifest_hashes_match_current_files(self) -> None:
        """最终清单的全部输入输出哈希必须与磁盘一致。"""

        manifest = json.loads((OUTPUT / "analysis_manifest.json").read_text(encoding="utf-8"))
        for name, item in manifest["outputs"].items():
            self.assertEqual(hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest(), item["sha256"], name)
        for name, item in manifest["inputs"].items():
            self.assertEqual(hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest(), item["sha256"], name)

    def test_workbook_cached_counts_and_formula_scan(self) -> None:
        """工作簿缓存计数与候选表一致，且没有公式错误。"""

        with zipfile.ZipFile(OUTPUT / "QTS_Author_Influence_LOAO.xlsx") as archive:
            root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        cells = {cell.attrib["r"]: cell for cell in root.findall(".//x:sheetData/x:row/x:c", ns)}
        for column in ["B", "F"]:
            self.assertEqual([float(cells[f"{column}{row}"].find("x:v", ns).text) for row in range(5, 10)], [1, 8, 4, 0, 0])
            self.assertTrue(all(cells[f"{column}{row}"].find("x:f", ns) is not None for row in range(5, 10)))
        verification = json.loads((OUTPUT / "workbook_verification.json").read_text(encoding="utf-8"))
        self.assertEqual(verification["rendered_sheet_count"], 13)
        self.assertIn("Cell search matched 0 entries.", verification["formula_error_scan"])


if __name__ == "__main__":
    unittest.main()
