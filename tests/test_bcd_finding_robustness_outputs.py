"""跨模型发现层正式产物和工作簿的独立验收。"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/BCD_robustness_config.yaml").read_text(encoding="utf-8"))
OUTPUT = ROOT / CONFIG["output_root"]
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def file_hash(path: Path) -> str:
    """独立计算输入输出文件哈希。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_xlsx_cells(archive: zipfile.ZipFile, member: str, strings: list[str]) -> dict[str, object]:
    """使用标准库解码单元格，不依赖工作簿编辑库。"""
    root = ET.fromstring(archive.read(member))
    cells = {}
    for cell in root.findall(".//m:sheetData/m:row/m:c", NS):
        kind = cell.get("t")
        value = cell.find("m:v", NS)
        raw = value.text if value is not None else None
        if kind == "s": decoded = strings[int(raw)]
        elif kind == "inlineStr": decoded = "".join(cell.itertext())
        elif kind == "b": decoded = raw == "1"
        elif kind == "e": raise AssertionError(f"工作簿错误单元格 {member}:{cell.get('r')}={raw}")
        elif raw is None: decoded = None
        elif kind == "str": decoded = raw
        else: decoded = float(raw)
        cells[cell.get("r")] = decoded
    return cells


def column(index: int) -> str:
    """转换一基列号。"""
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


class BCDFindingOutputTest(unittest.TestCase):
    """核验任务范围、真实数据、图表和工作簿全表镜像。"""

    def test_eligibility_is_exact_source_filter(self) -> None:
        """主结果不能加入已降级候选。"""
        source = pd.read_csv(ROOT / CONFIG["eligible_findings"])
        expected = {(r.topic_id, r.contrast) for r in source.loc[source["finding_level_bd_eligible"]].itertuples()}
        effects = pd.read_csv(OUTPUT / "data/BCD_finding_effects.csv")
        actual = set(zip(effects["c_topic_id"], effects["contrast"]))
        self.assertEqual(actual, expected)
        self.assertEqual(len(effects), len(expected) * 3 * 3 * 2)

    def test_common_universe_is_exact_intersection(self) -> None:
        """共同诗歌集合不能混入native-only文档。"""
        sets = [set(pd.read_csv(ROOT / CONFIG["freeze_root"] / f"{m}_K20/document_index.csv")["poem_id"]) for m in "BCD"]
        common = pd.read_csv(OUTPUT / "data/BCD_common_document_index.csv")
        self.assertEqual(set(common["poem_id"]), set.intersection(*sets))
        self.assertFalse(common["poem_id"].duplicated().any())

    def test_invariance_twenty_pass(self) -> None:
        """用户要求的20项检查必须全部通过。"""
        checks = pd.read_csv(OUTPUT / "data/invariance_tests.csv")
        self.assertEqual(len(checks), 20)
        self.assertTrue(checks["status"].eq("PASS").all())

    def test_source_and_output_hashes(self) -> None:
        """再次独立核查所有来源及登记产物哈希。"""
        manifest = json.loads((OUTPUT / "analysis_manifest.json").read_text(encoding="utf-8"))
        for record in [*manifest["inputs"].values(), *manifest.get("implementation", {}).values()]:
            self.assertEqual(file_hash(Path(record["path"])), record["sha256"])
        for name, record in manifest["outputs"].items():
            self.assertEqual(file_hash(OUTPUT / name), record["sha256"], name)

    def test_bootstrap_and_no_new_pvalue_family(self) -> None:
        """5000次作者bootstrap且不新增探索性检验家族。"""
        effects = pd.read_csv(OUTPUT / "data/BCD_finding_effects.csv")
        self.assertTrue(effects["bootstrap_iterations"].eq(5000).all())
        self.assertTrue(effects["bootstrap_unit"].eq("author").all())
        self.assertTrue(effects["random_seed"].eq(20260902).all())
        self.assertTrue(effects["formal_p_value_family"].eq("NONE_PRESELECTED_FINDINGS").all())
        shape = pd.read_csv(OUTPUT / "data/shape_similarity.csv")
        self.assertFalse(any(name in shape for name in ["p_value", "pearson_p", "spearman_p"]))

    def test_split_and_native_are_separate_sensitivities(self) -> None:
        """single-best与native不能混入主矩阵。"""
        primary = pd.read_csv(OUTPUT / "data/BCD_finding_effects.csv")
        singles = pd.read_csv(OUTPUT / "data/split_single_best_sensitivity.csv")
        native = pd.read_csv(OUTPUT / "data/model_native_sensitivity.csv")
        self.assertTrue(primary["mapping_variant"].eq("PRIMARY").all())
        self.assertTrue(primary["universe"].eq("COMMON").all())
        self.assertTrue(singles["mapping_variant"].eq("SINGLE_BEST").all())
        self.assertTrue(native["universe"].eq("NATIVE").all())

    def test_required_figures_and_report(self) -> None:
        """六张PNG与十九节报告存在且未输出历史确认。"""
        for name in ["BCD_finding_effect_comparison_poem.png", "BCD_finding_effect_comparison_author.png", "BCD_finding_direction_matrix.png", "BCD_finding_semantic_mapping.png", "BCD_finding_robustness_matrix.png", "BCD_priority_four_trends.png"]:
            self.assertEqual((OUTPUT / "figures" / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        report = (OUTPUT / "QTS_BCD_Finding_Level_Robustness_Report.md").read_text(encoding="utf-8")
        for number in range(1, 20): self.assertIn(f"## {number}.", report)
        self.assertNotIn("三个模型共同证明", report)

    def test_original_status_preserved(self) -> None:
        """C原始统计/LOAO标签只能增加证据，不覆盖。"""
        source = pd.read_csv(ROOT / CONFIG["eligible_findings"]).set_index(["topic_id", "contrast"])
        result = pd.read_csv(OUTPUT / "data/BCD_finding_robustness_comparison.csv")
        for row in result.to_dict("records"):
            original = source.loc[(row["topic"], row["contrast"])]
            self.assertEqual(row["C_original_status"], original["previous_validation_status"])
            self.assertEqual(row["C_author_robustness_status"], original["author_robustness_status"])

    def test_workbook_full_data_mirror(self) -> None:
        """十二页结构、错误单元格及全部类型化数据独立核验。"""
        payload = json.loads((OUTPUT / ".artifact_work/workbook_payload.json").read_text(encoding="utf-8"))
        with zipfile.ZipFile(OUTPUT / "QTS_BCD_Finding_Level_Robustness.xlsx") as archive:
            self.assertIsNone(archive.testzip())
            xml = ET.fromstring(archive.read("xl/workbook.xml"))
            sheets = xml.findall("m:sheets/m:sheet", NS)
            self.assertEqual([s.get("name") for s in sheets], ["Overview", *[t["sheet"] for t in payload["tables"]]])
            strings = []
            if "xl/sharedStrings.xml" in archive.namelist():
                strings = ["".join(si.itertext()) for si in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", NS)]
            for index, table in enumerate(payload["tables"], start=2):
                actual = read_xlsx_cells(archive, f"xl/worksheets/sheet{index}.xml", strings)
                for row_index, row in enumerate([table["headers"], *table["data"]], start=1):
                    for col_index, expected in enumerate(row, start=1):
                        observed = actual.get(f"{column(col_index)}{row_index}")
                        if isinstance(expected, bool): self.assertEqual(observed, expected)
                        elif isinstance(expected, (int, float)):
                            self.assertTrue(math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12), (table["sheet"], row_index, col_index))
                        elif expected in (None, ""): self.assertIn(observed, (None, ""), (table["sheet"], row_index, col_index))
                        else: self.assertEqual(observed, expected, (table["sheet"], row_index, col_index))
            overview = read_xlsx_cells(archive, "xl/worksheets/sheet1.xml", strings)
            self.assertEqual(overview["B5"], payload["summary"]["common_documents"])
            self.assertEqual(overview["B6"], len(payload["summary"]["eligible_findings"]))
            self.assertEqual(overview["B14"], payload["summary"]["robustness_counts"]["ROBUST_ACROSS_BCD"])
        verification = json.loads((OUTPUT / "workbook_verification.json").read_text(encoding="utf-8"))
        self.assertEqual(verification["rendered_sheet_count"], 12)
        self.assertIn("matched 0 entries", verification["formula_error_scan"])


if __name__ == "__main__":
    unittest.main()
