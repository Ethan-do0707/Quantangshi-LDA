"""独立核验历史元数据工作簿的 ZIP、页签、维度与表头。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import posixpath
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CELL_REF_RE = re.compile(r"([A-Z]+)(\d+)")


@dataclass(frozen=True)
class SheetExpectation:
    """单一工作表的可复算预期。"""

    name: str
    headers: list[str]
    data_rows: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_shape(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader)
        return headers, sum(1 for _ in reader)


def column_name(number: int) -> str:
    if number < 1:
        raise ValueError("列数必须大于零")
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def expected_dimension(headers: list[str], data_rows: int) -> str:
    return f"A1:{column_name(len(headers))}{data_rows + 1}"


def cell_column_index(reference: str) -> int:
    match = CELL_REF_RE.fullmatch(reference)
    if not match:
        raise ValueError(f"无法解析单元格引用：{reference}")
    value = 0
    for character in match.group(1):
        value = value * 26 + ord(character) - 64
    return value - 1


def calculated_dimension(root: ET.Element) -> str:
    references = [
        cell.attrib["r"]
        for cell in root.findall(f".//{{{MAIN_NS}}}sheetData/{{{MAIN_NS}}}row/{{{MAIN_NS}}}c")
        if "r" in cell.attrib
    ]
    if not references:
        return ""
    parsed = []
    for reference in references:
        match = CELL_REF_RE.fullmatch(reference)
        if match is None:
            continue
        parsed.append((cell_column_index(reference), int(match.group(2))))
    if not parsed:
        return ""
    min_column = min(item[0] for item in parsed) + 1
    max_column = max(item[0] for item in parsed) + 1
    min_row = min(item[1] for item in parsed)
    max_row = max(item[1] for item in parsed)
    return f"{column_name(min_column)}{min_row}:{column_name(max_column)}{max_row}"


def shared_strings(archive: zipfile.ZipFile) -> list[str]:
    name = "xl/sharedStrings.xml"
    if name not in archive.namelist():
        return []
    root = ET.fromstring(archive.read(name))
    return ["".join(node.text or "" for node in item.findall(f".//{{{MAIN_NS}}}t")) for item in root]


def worksheet_map(archive: zipfile.ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in relationships.findall(f"{{{PKG_REL_NS}}}Relationship")
    }
    result: dict[str, str] = {}
    for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
        relation_id = sheet.attrib[f"{{{DOC_REL_NS}}}id"]
        target = targets[relation_id].lstrip("/")
        if target.startswith("xl/"):
            sheet_path = posixpath.normpath(target)
        else:
            sheet_path = posixpath.normpath(posixpath.join("xl", target))
        result[sheet.attrib["name"]] = sheet_path
    return result


def cell_text(cell: ET.Element, strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(f".//{{{MAIN_NS}}}t"))
    value_node = cell.find(f"{{{MAIN_NS}}}v")
    if value_node is None or value_node.text is None:
        return ""
    if cell_type == "s":
        return strings[int(value_node.text)]
    return value_node.text


def inspect_sheet(archive: zipfile.ZipFile, sheet_path: str, strings: list[str]) -> tuple[str, list[str]]:
    root = ET.fromstring(archive.read(sheet_path))
    dimension_node = root.find(f"{{{MAIN_NS}}}dimension")
    dimension = dimension_node.attrib.get("ref", "") if dimension_node is not None else calculated_dimension(root)
    row = root.find(f".//{{{MAIN_NS}}}sheetData/{{{MAIN_NS}}}row[@r='1']")
    if row is None:
        return dimension, []
    cells = list(row.findall(f"{{{MAIN_NS}}}c"))
    if not cells:
        return dimension, []
    max_index = max(cell_column_index(cell.attrib["r"]) for cell in cells)
    headers = [""] * (max_index + 1)
    for cell in cells:
        headers[cell_column_index(cell.attrib["r"])] = cell_text(cell, strings)
    return dimension, headers


def verify_workbook(path: Path, expectations: list[SheetExpectation]) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
        "status": "FAIL",
        "checks": [],
    }
    checks: list[dict[str, object]] = result["checks"]  # type: ignore[assignment]
    if not path.exists():
        checks.append({"check": "file_exists", "status": "FAIL"})
        return result
    result["size_bytes"] = path.stat().st_size
    result["sha256"] = sha256_file(path)
    try:
        with zipfile.ZipFile(path) as archive:
            corrupt_member = archive.testzip()
            checks.append({"check": "zip_integrity", "status": "PASS" if corrupt_member is None else "FAIL", "observed": corrupt_member})
            sheets = worksheet_map(archive)
            expected_names = [item.name for item in expectations]
            actual_names = list(sheets)
            checks.append({"check": "sheet_names", "status": "PASS" if actual_names == expected_names else "FAIL", "observed": actual_names, "expected": expected_names})
            strings = shared_strings(archive)
            sheet_results = []
            for item in expectations:
                if item.name not in sheets:
                    sheet_results.append({"sheet": item.name, "status": "FAIL", "reason": "MISSING_SHEET"})
                    continue
                dimension, headers = inspect_sheet(archive, sheets[item.name], strings)
                dimension_expected = expected_dimension(item.headers, item.data_rows)
                dimension_ok = dimension == dimension_expected
                headers_ok = headers == item.headers
                sheet_results.append(
                    {
                        "sheet": item.name,
                        "status": "PASS" if dimension_ok and headers_ok else "FAIL",
                        "dimension": dimension,
                        "expected_dimension": dimension_expected,
                        "headers_match": headers_ok,
                    }
                )
            result["sheets"] = sheet_results
            checks.append({"check": "sheet_dimensions_and_headers", "status": "PASS" if all(item["status"] == "PASS" for item in sheet_results) else "FAIL"})
    except (KeyError, OSError, ValueError, zipfile.BadZipFile, ET.ParseError) as error:
        checks.append({"check": "xlsx_parse", "status": "FAIL", "observed": f"{type(error).__name__}: {error}"})
    result["status"] = "PASS" if checks and all(item["status"] == "PASS" for item in checks) else "FAIL"
    return result


def expectation_from_csv(name: str, path: Path) -> SheetExpectation:
    headers, data_rows = read_csv_shape(path)
    return SheetExpectation(name=name, headers=headers, data_rows=data_rows)


def build_expectations(root: Path) -> dict[Path, list[SheetExpectation]]:
    output_root = root / "historical_metadata"
    return {
        output_root / "master" / "QTS_author_historical_metadata.xlsx": [
            expectation_from_csv("Author_Metadata", output_root / "master" / "QTS_author_historical_metadata.csv"),
            expectation_from_csv("Period_Coverage", output_root / "chronology" / "period_coverage.csv"),
            expectation_from_csv("Build_QC", output_root / "qc" / "historical_metadata_qc.csv"),
            SheetExpectation("Method_Notes", ["项目", "说明"], 6),
        ],
        output_root / "review" / "author_metadata_human_review.xlsx": [
            expectation_from_csv("Review_Queue", output_root / "review" / "author_metadata_human_review.csv"),
            expectation_from_csv("Candidate_Audit", output_root / "identity" / "cbdb_match_candidates.csv"),
            SheetExpectation("Instructions", ["项目", "说明"], 4),
        ],
        output_root / "docs" / "historical_metadata_data_dictionary.xlsx": [
            expectation_from_csv("Fields", output_root / "docs" / "historical_metadata_data_dictionary.csv"),
            SheetExpectation("Period_Scheme", ["label", "start", "end", "scheme_version", "scheme_note"], 4),
            SheetExpectation("Provenance_Notes", ["项目", "说明"], 6),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="核验历史元数据工作簿结构")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="项目根目录")
    parser.add_argument("--output", type=Path, default=None, help="核验结果 JSON")
    args = parser.parse_args()
    root = args.root.resolve()
    output_root = root / "historical_metadata"
    results = [verify_workbook(path, expected) for path, expected in build_expectations(root).items()]
    required_master = output_root / "master" / "C_K20_literary_history_master_with_author_history.xlsx"
    authoritative_csv = output_root / "master" / "C_K20_literary_history_master_with_author_history.csv"
    master_verification_path = output_root / "qc" / "enriched_master_xlsx_verification.json"
    recorded_master_verification = (
        json.loads(master_verification_path.read_text(encoding="utf-8"))
        if master_verification_path.exists()
        else {}
    )
    master_checks = {
        "file_exists": required_master.exists(),
        "verification_report_exists": master_verification_path.exists(),
        "verification_report_pass": recorded_master_verification.get("status") == "PASS",
        "xlsx_hash_matches_report": (
            required_master.exists()
            and sha256_file(required_master) == recorded_master_verification.get("output_xlsx_sha256")
        ),
        "source_csv_hash_matches_report": (
            authoritative_csv.exists()
            and sha256_file(authoritative_csv) == recorded_master_verification.get("source_csv_sha256")
        ),
        "worksheet_shape_verified": (
            recorded_master_verification.get("worksheet", {}).get("data_rows") == 43015
            and recorded_master_verification.get("worksheet", {}).get("spreadsheet_rows") == 43016
            and recorded_master_verification.get("worksheet", {}).get("columns") == 68
        ),
        "sample_values_verified": (
            recorded_master_verification.get("checks", {}).get("sample_values_match") == "PASS"
            and recorded_master_verification.get("comparisons", {}).get("random_sample_rows", 0) >= 100
        ),
    }
    master_status = {
        "path": str(required_master),
        "exists": required_master.exists(),
        "status": "PASS" if all(master_checks.values()) else "FAIL",
        "checks": {key: "PASS" if value else "FAIL" for key, value in master_checks.items()},
        "verification_report": str(master_verification_path),
    }
    completed_ok = all(item["status"] == "PASS" for item in results)
    payload = {
        "status": "PASS" if completed_ok and master_status["status"] == "PASS" else "FAIL",
        "completed_workbooks": results,
        "enriched_master_workbook": master_status,
    }
    output_path = args.output or output_root / "qc" / "historical_metadata_workbook_verification.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if completed_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
