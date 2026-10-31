from __future__ import annotations

import csv
import hashlib
import json
import math
import posixpath
import re
import unittest
import warnings
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from openpyxl import load_workbook
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "historical_metadata" / "representativeness_audit"
DATA_ROOT = AUDIT_ROOT / "data"
WORKBOOK = AUDIT_ROOT / "Period_Metadata_Representativeness_Audit.xlsx"

EXPECTED_SHEETS = [
    "Overview",
    "Coverage",
    "Author_Productivity",
    "Topic_Probability",
    "Topic_Type_Mass",
    "Dominant_Topic",
    "Entropy",
    "CBDB_Status",
    "Missingness_Model",
    "Sample_Sensitivity",
    "Clear_Topics",
    "High_Risk_Topics",
    "Method",
    "QC",
]

# CSV 文件、工作表、表头所在行（Excel 1-based）。
TABLE_SPECS = [
    ("period_metadata_coverage_audit.csv", "Coverage", 4),
    ("author_productivity_known_vs_unknown.csv", "Author_Productivity", 4),
    ("period_coverage_by_author_productivity.csv", "Author_Productivity", 10),
    ("topic_probability_representativeness.csv", "Topic_Probability", 4),
    ("topic_type_mass_representativeness.csv", "Topic_Type_Mass", 4),
    ("dominant_topic_known_unknown.csv", "Dominant_Topic", 4),
    ("dominant_topic_type_known_unknown.csv", "Dominant_Topic", 29),
    ("topic_mixture_representativeness.csv", "Entropy", 4),
    ("period_coverage_by_author_status.csv", "CBDB_Status", 5),
    ("period_coverage_by_cbdb_status.csv", "CBDB_Status", 12),
    ("period_coverage_by_cbdb_confidence.csv", "CBDB_Status", 30),
    ("period_missingness_logistic_diagnostic.csv", "Missingness_Model", 4),
    ("analysis_sample_representativeness.csv", "Sample_Sensitivity", 4),
    ("clear_topic_representativeness.csv", "Clear_Topics", 4),
    ("topic_missingness_risk.csv", "High_Risk_Topics", 5),
    ("representativeness_audit_qc.csv", "QC", 4),
]

NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_cell_matches(csv_value: str, excel_value: object) -> None:
    if csv_value == "":
        assert excel_value in (None, "")
        return
    if NUMERIC_RE.fullmatch(csv_value) and isinstance(excel_value, (int, float)):
        assert math.isclose(float(csv_value), float(excel_value), rel_tol=1e-11, abs_tol=1e-12)
        return
    assert str(excel_value) == csv_value


def test_workbook_zip_xml_and_relationships_are_valid() -> None:
    assert WORKBOOK.exists() and WORKBOOK.stat().st_size > 0
    with zipfile.ZipFile(WORKBOOK) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        assert "[Content_Types].xml" in names
        assert "xl/workbook.xml" in names
        assert any(name.startswith("xl/worksheets/sheet") and name.endswith(".xml") for name in names)
        for name in names:
            if name.endswith((".xml", ".rels")):
                ET.fromstring(archive.read(name))

        relation_namespace = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
        for rel_name in (name for name in names if name.endswith(".rels")):
            if rel_name == "_rels/.rels":
                owner = ""
            else:
                prefix, rel_file = rel_name.rsplit("/_rels/", 1)
                owner = f"{prefix}/{rel_file[:-5]}"
            root = ET.fromstring(archive.read(rel_name))
            for relation in root.findall("r:Relationship", relation_namespace):
                if relation.attrib.get("TargetMode") == "External":
                    continue
                target = relation.attrib["Target"]
                if target.startswith("/"):
                    resolved = target.lstrip("/")
                else:
                    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(owner), target))
                assert resolved in names, f"缺失关系目标：{rel_name} -> {resolved}"


def test_workbook_structure_formulas_and_cached_values() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        formula_book = load_workbook(WORKBOOK, data_only=False)
        value_book = load_workbook(WORKBOOK, data_only=True)
    assert formula_book.sheetnames == EXPECTED_SHEETS
    assert sum(len(sheet.tables) for sheet in formula_book.worksheets) == 17
    assert len(formula_book["Overview"]._charts) == 1
    assert formula_book["Overview"]["B4"].value == "=Coverage!D6"
    assert formula_book["Overview"]["E4"].value == "=Coverage!F6"
    assert formula_book["Overview"]["H4"].value == "=MAX(Topic_Probability!K5:K24)"
    assert math.isclose(value_book["Overview"]["B4"].value, 0.18947784810126583)
    assert math.isclose(value_book["Overview"]["E4"].value, 0.5488550505637568)
    assert math.isclose(value_book["Overview"]["H4"].value, 0.15166643272160263)


def test_workbook_tables_match_csv_outputs() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        book = load_workbook(WORKBOOK, data_only=True, read_only=False)
    for csv_name, sheet_name, header_row in TABLE_SPECS:
        with (DATA_ROOT / csv_name).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = reader.fieldnames
        assert fields is not None
        sheet = book[sheet_name]
        excel_headers = [sheet.cell(header_row, column).value for column in range(1, len(fields) + 1)]
        assert excel_headers == fields
        for row_offset, csv_row in enumerate(rows, start=1):
            for column, field in enumerate(fields, start=1):
                assert_cell_matches(csv_row[field], sheet.cell(header_row + row_offset, column).value)


def test_figures_and_manifest_are_valid() -> None:
    expected_figures = {
        "author_productivity_period_coverage.png",
        "dominant_topic_known_unknown.png",
        "entropy_known_unknown.png",
        "topic_known_unknown_smd.png",
        "topic_mean_known_vs_unknown.png",
    }
    for name in expected_figures:
        with Image.open(AUDIT_ROOT / "figures" / name) as image:
            image.verify()

    manifest = json.loads((AUDIT_ROOT / "audit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs"]["authoritative_master"]["rows"] == 43015
    assert manifest["inputs"]["authoritative_master"]["columns"] == 68
    for relative_name, metadata in manifest["outputs"].items():
        output = AUDIT_ROOT / relative_name
        assert output.exists()
        assert output.stat().st_size == metadata["size_bytes"]
        assert sha256(output) == metadata["sha256"]


class PeriodRepresentativenessAuditOutputTest(unittest.TestCase):
    """以标准库 unittest 运行导出物独立验证。"""

    def test_zip_xml_relationships(self) -> None:
        test_workbook_zip_xml_and_relationships_are_valid()

    def test_structure_formulas_cached_values(self) -> None:
        test_workbook_structure_formulas_and_cached_values()

    def test_csv_workbook_mirror(self) -> None:
        test_workbook_tables_match_csv_outputs()

    def test_figures_manifest(self) -> None:
        test_figures_and_manifest_are_valid()


if __name__ == "__main__":
    unittest.main()
