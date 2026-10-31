"""独立核验文学史母数据集 XLSX 的结构、公式与缓存值。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def verify(dataset_root: Path) -> dict[str, Any]:
    """验证主表 XLSX，并把结果写入 qc。"""

    xlsx_path = dataset_root / "master" / "C_K20_literary_history_master.xlsx"
    csv_path = dataset_root / "master" / "C_K20_literary_history_master.csv"
    expected_rows = 43015
    expected_columns = 54
    sample_excel_rows = {2, 12346, 43016}
    formula_columns = {"AS", "AV", "AY", "AZ", "BA", "BB"}
    cached_values: dict[str, float] = {}
    formula_text: dict[str, str] = {}

    with zipfile.ZipFile(xlsx_path) as archive:
        workbook_xml = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet_names = [node.attrib["name"] for node in workbook_xml.findall(f".//{{{MAIN_NS}}}sheet")]
        with archive.open("xl/worksheets/sheet1.xml") as sheet_stream:
            dimension = ""
            row_count = 0
            formula_count = 0
            for event, element in ET.iterparse(sheet_stream, events=("start", "end")):
                tag = element.tag.rsplit("}", 1)[-1]
                if event == "start" and tag == "dimension":
                    dimension = element.attrib.get("ref", "")
                elif event == "end" and tag == "row":
                    row_count += 1
                    element.clear()
                elif event == "end" and tag == "c":
                    reference = element.attrib.get("r", "")
                    formula = element.find(f"{{{MAIN_NS}}}f")
                    if formula is not None:
                        formula_count += 1
                    match = re.fullmatch(r"([A-Z]+)(\d+)", reference)
                    if match and int(match.group(2)) in sample_excel_rows and match.group(1) in formula_columns:
                        value = element.find(f"{{{MAIN_NS}}}v")
                        cached_values[reference] = float(value.text) if value is not None and value.text is not None else math.nan
                        formula_text[reference] = formula.text if formula is not None and formula.text is not None else ""
                    element.clear()

    csv_samples: dict[int, dict[str, str]] = {}
    target_data_indices = {row - 2 for row in sample_excel_rows}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        for data_index, row in enumerate(reader):
            if data_index in target_data_indices:
                csv_samples[data_index + 2] = row

    mapping = {
        "AS": "dominant_topic_probability",
        "AV": "second_topic_probability",
        "AY": "third_topic_probability",
        "AZ": "topic_entropy",
        "BA": "normalized_topic_entropy",
        "BB": "top1_top2_gap",
    }
    sample_differences: dict[str, float] = {}
    for excel_row in sorted(sample_excel_rows):
        for column, field in mapping.items():
            reference = f"{column}{excel_row}"
            sample_differences[reference] = abs(cached_values[reference] - float(csv_samples[excel_row][field]))

    checks = {
        "xlsx_exists": xlsx_path.exists(),
        "sheet_names": sheet_names == ["Data", "Topic_Metadata", "Build_QC", "Codebook"],
        "data_dimension": dimension == "A1:BB43016",
        "data_row_count": row_count == expected_rows + 1,
        "csv_column_count": len(headers) == expected_columns,
        "formula_count": formula_count == expected_rows * len(formula_columns),
        "sample_formula_presence": len(formula_text) == len(sample_excel_rows) * len(formula_columns) and all(formula_text.values()),
        "cached_values_match_csv": max(sample_differences.values()) < 1e-12,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "sheet_names": sheet_names,
        "data_dimension": dimension,
        "data_row_count_including_header": row_count,
        "formula_count": formula_count,
        "expected_formula_count": expected_rows * len(formula_columns),
        "sample_formula_refs": formula_text,
        "sample_cached_value_differences": sample_differences,
        "xlsx_size_bytes": xlsx_path.stat().st_size,
    }
    output_path = dataset_root / "qc" / "master_xlsx_verification.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise RuntimeError(f"主表 XLSX 核验失败：{checks}")
    return result


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""

    result = verify(parse_args().dataset_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
