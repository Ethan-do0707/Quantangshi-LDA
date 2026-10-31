"""从权威 CSV 流式生成并独立核验大型历史元数据 master XLSX。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import posixpath
import random
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

try:
    import xlsxwriter
except ModuleNotFoundError:
    xlsxwriter = None


EXPECTED_DATA_ROWS = 43_015
EXPECTED_COLUMNS = 68
SAMPLE_SIZE = 100
SAMPLE_SEED = 20_260_901
MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CELL_REF_RE = re.compile(r"([A-Z]+)(\d+)")

INTEGER_FIELDS = {
    "model_row_id",
    "poem_id",
    "original_corpus_index",
    "poet_id",
    "author_birth_year",
    "author_death_year",
    "author_fl_start",
    "author_fl_end",
}
FLOAT_FIELDS = {
    "dominant_topic_probability",
    "second_topic_probability",
    "third_topic_probability",
    "topic_entropy",
    "normalized_topic_entropy",
    "top1_top2_gap",
}
BOOLEAN_FIELDS = {"cross_period_flag", "period_boundary_flag"}
EXCEPTION_NOTE = (
    "Large enriched master XLSX was exported using XlsxWriter constant-memory streaming because "
    "artifact-tool exceeded the available 8 GB heap. The authoritative CSV had already passed all "
    "QC tests; the XLSX is a verified serialization of that CSV."
)


def sha256_file(path: Path) -> str:
    """以流式方式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_csv_field_size_limit() -> int:
    """设置当前平台可接受的最大 CSV 字段长度。"""

    limit = sys.maxsize
    while limit > 0:
        try:
            csv.field_size_limit(limit)
            return limit
        except OverflowError:
            limit //= 10
    raise RuntimeError("无法设置 CSV 字段长度上限")


def has_leading_zero(value: str) -> bool:
    """识别不可安全转成 Excel 数值的前导零整数。"""

    unsigned = value.lstrip("+-")
    return len(unsigned) > 1 and unsigned.startswith("0")


def typed_value(header: str, value: str) -> tuple[str, Any]:
    """按冻结字段语义决定 Excel 单元格类型。"""

    if value == "":
        return "blank", None
    if header in BOOLEAN_FIELDS:
        if value not in {"True", "False"}:
            raise ValueError(f"布尔字段 {header} 出现非法值：{value}")
        return "boolean", value == "True"
    if header in INTEGER_FIELDS:
        if has_leading_zero(value):
            return "text", value
        return "integer", int(value)
    if header.endswith("_prob") or header in FLOAT_FIELDS:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"数值字段 {header} 出现非有限值：{value}")
        return "float", number
    return "text", value


def column_index(reference: str) -> int:
    """把 A1 引用转换为零起始列号。"""

    match = CELL_REF_RE.fullmatch(reference)
    if match is None:
        raise ValueError(f"无法解析单元格引用：{reference}")
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - 64
    return result - 1


def column_name(number: int) -> str:
    """把一开始列号转换为 Excel 列名。"""

    if number < 1:
        raise ValueError("列号必须大于零")
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def build_workbook(source_csv: Path, temporary_xlsx: Path) -> dict[str, Any]:
    """逐行写入 CSV，不重新计算任何研究字段。"""

    if xlsxwriter is None:
        raise RuntimeError("当前 Python 环境缺少 XlsxWriter；请使用项目打包 Python 运行")
    workbook = xlsxwriter.Workbook(
        str(temporary_xlsx),
        {
            "constant_memory": True,
            "in_memory": False,
            "nan_inf_to_errors": False,
            "tmpdir": str(temporary_xlsx.parent),
        },
    )
    workbook.set_properties(
        {
            "title": "《全唐诗》C_K20 文学史母表（作者历史元数据增强）",
            "subject": "权威 CSV 的常量内存序列化镜像",
            "author": "《全唐诗》LDA 项目",
            "comments": EXCEPTION_NOTE,
        }
    )
    header_format = workbook.add_format(
        {
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#1F4E78",
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
            "border": 1,
            "border_color": "#D9D9D9",
        }
    )
    text_format = workbook.add_format({"font_color": "#404040", "valign": "top"})
    integer_format = workbook.add_format({"num_format": "0", "font_color": "#404040", "valign": "top"})
    float_format = workbook.add_format({"num_format": "0.000000000000", "font_color": "#404040", "valign": "top"})
    boolean_format = workbook.add_format({"font_color": "#404040", "valign": "top"})

    worksheet = workbook.add_worksheet("Data")
    worksheet.hide_gridlines(2)
    worksheet.freeze_panes(1, 7)
    worksheet.set_default_row(15)
    worksheet.set_row(0, 36)
    worksheet.set_column("A:E", 14)
    worksheet.set_column("F:G", 20)
    worksheet.set_column("H:H", 38)
    worksheet.set_column("I:J", 48)
    worksheet.set_column("K:V", 13)
    worksheet.set_column("W:AP", 14)
    worksheet.set_column("AQ:BB", 22)
    worksheet.set_column("BC:BP", 21)

    configure_csv_field_size_limit()
    type_counts = {"blank": 0, "boolean": 0, "integer": 0, "float": 0, "text": 0}
    data_rows = 0
    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader)
        if len(headers) != EXPECTED_COLUMNS:
            raise ValueError(f"权威 CSV 列数为 {len(headers)}，预期 {EXPECTED_COLUMNS}")
        worksheet.write_row(0, 0, headers, header_format)
        for data_rows, values in enumerate(reader, start=1):
            if len(values) != len(headers):
                raise ValueError(f"CSV 数据行 {data_rows} 列数为 {len(values)}，预期 {len(headers)}")
            for column, (header, raw_value) in enumerate(zip(headers, values)):
                value_type, value = typed_value(header, raw_value)
                type_counts[value_type] += 1
                if value_type == "blank":
                    continue
                if value_type == "boolean":
                    worksheet.write_boolean(data_rows, column, value, boolean_format)
                elif value_type == "integer":
                    worksheet.write_number(data_rows, column, value, integer_format)
                elif value_type == "float":
                    worksheet.write_number(data_rows, column, value, float_format)
                else:
                    if len(value) > 32_767:
                        raise ValueError(f"CSV 数据行 {data_rows} 字段 {header} 超过 Excel 32,767 字符限制")
                    worksheet.write_string(data_rows, column, value, text_format)
    if data_rows != EXPECTED_DATA_ROWS:
        raise ValueError(f"权威 CSV 数据行为 {data_rows}，预期 {EXPECTED_DATA_ROWS}")
    worksheet.autofilter(0, 0, data_rows, len(headers) - 1)
    close_result = workbook.close()
    if close_result:
        raise RuntimeError(f"XLSX 写入失败：{close_result}")
    return {
        "worksheet": "Data",
        "data_rows": data_rows,
        "spreadsheet_rows": data_rows + 1,
        "columns": len(headers),
        "type_counts": type_counts,
    }


def shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """读取可选 shared strings；常量内存模式通常使用 inline strings。"""

    name = "xl/sharedStrings.xml"
    if name not in archive.namelist():
        return []
    root = ET.fromstring(archive.read(name))
    return ["".join(node.text or "" for node in item.findall(f".//{{{MAIN_NS}}}t")) for item in root]


def xml_cell_value(cell: ET.Element, strings: list[str]) -> Any:
    """读取 worksheet XML 中一个单元格的序列化值。"""

    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(f".//{{{MAIN_NS}}}t"))
    value_node = cell.find(f"{{{MAIN_NS}}}v")
    if value_node is None or value_node.text is None:
        return None
    if cell_type == "s":
        return strings[int(value_node.text)]
    if cell_type == "b":
        return value_node.text == "1"
    return value_node.text


def collect_csv_evidence(source_csv: Path, sample_indices: set[int]) -> dict[str, Any]:
    """流式收集 CSV 的结构、样本和 poem_id 顺序证据。"""

    configure_csv_field_size_limit()
    poem_id_hash = hashlib.sha256()
    poem_ids: set[str] = set()
    duplicate_poem_ids: list[str] = []
    samples: dict[int, list[str]] = {}
    data_rows = 0
    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        headers = next(reader)
        poem_id_column = headers.index("poem_id")
        for data_index, values in enumerate(reader):
            data_rows += 1
            poem_id = values[poem_id_column]
            poem_id_hash.update(poem_id.encode("utf-8"))
            poem_id_hash.update(b"\n")
            if poem_id in poem_ids:
                duplicate_poem_ids.append(poem_id)
            poem_ids.add(poem_id)
            if data_index in sample_indices:
                samples[data_index] = values
    return {
        "headers": headers,
        "data_rows": data_rows,
        "columns": len(headers),
        "poem_id_sha256": poem_id_hash.hexdigest(),
        "duplicate_poem_ids": duplicate_poem_ids,
        "samples": samples,
    }


def workbook_sheet_path(archive: zipfile.ZipFile) -> tuple[list[str], str]:
    """核验 workbook relationships 并返回 Data worksheet 路径。"""

    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in relationships.findall(f"{{{PKG_REL_NS}}}Relationship")
    }
    sheet_nodes = workbook.findall(f".//{{{MAIN_NS}}}sheet")
    sheet_names = [sheet.attrib["name"] for sheet in sheet_nodes]
    if sheet_names != ["Data"]:
        raise ValueError(f"工作簿页签为 {sheet_names}，预期仅有 Data")
    relation_id = sheet_nodes[0].attrib[f"{{{DOC_REL_NS}}}id"]
    target = targets[relation_id].lstrip("/")
    sheet_path = target if target.startswith("xl/") else posixpath.normpath(posixpath.join("xl", target))
    if sheet_path not in archive.namelist():
        raise ValueError(f"workbook relationship 指向不存在的文件：{sheet_path}")
    return sheet_names, sheet_path


def collect_xlsx_evidence(xlsx_path: Path, sample_indices: set[int]) -> dict[str, Any]:
    """以 worksheet XML 流式核验行列、顺序和抽样值。"""

    required_parts = {
        "[Content_Types].xml",
        "_rels/.rels",
        "docProps/app.xml",
        "docProps/core.xml",
        "xl/styles.xml",
        "xl/workbook.xml",
        "xl/_rels/workbook.xml.rels",
    }
    with zipfile.ZipFile(xlsx_path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise zipfile.BadZipFile(f"损坏成员：{corrupt_member}")
        missing_parts = sorted(required_parts - set(archive.namelist()))
        if missing_parts:
            raise ValueError(f"XLSX 缺少必要部件：{missing_parts}")
        for name in sorted(required_parts - {"[Content_Types].xml"}):
            if name.endswith(".xml") or name.endswith(".rels"):
                ET.fromstring(archive.read(name))
        ET.fromstring(archive.read("[Content_Types].xml"))
        sheet_names, sheet_path = workbook_sheet_path(archive)
        strings = shared_strings(archive)
        samples: dict[int, dict[int, Any]] = {}
        poem_id_hash = hashlib.sha256()
        poem_ids: set[str] = set()
        duplicate_poem_ids: list[str] = []
        headers: list[Any] = []
        dimension = ""
        expected_row_number = 1
        row_count = 0
        max_column = 0
        with archive.open(sheet_path) as worksheet_xml:
            for _, element in ET.iterparse(worksheet_xml, events=("end",)):
                local_name = element.tag.rsplit("}", 1)[-1]
                if local_name == "dimension":
                    dimension = element.attrib.get("ref", "")
                if local_name != "row":
                    continue
                row_number = int(element.attrib["r"])
                if row_number != expected_row_number:
                    raise ValueError(f"worksheet 行号不连续：观测 {row_number}，预期 {expected_row_number}")
                expected_row_number += 1
                row_count += 1
                values_by_column: dict[int, Any] = {}
                for cell in element.findall(f"{{{MAIN_NS}}}c"):
                    index = column_index(cell.attrib["r"])
                    max_column = max(max_column, index + 1)
                    values_by_column[index] = xml_cell_value(cell, strings)
                if row_number == 1:
                    headers = [values_by_column.get(index) for index in range(EXPECTED_COLUMNS)]
                else:
                    data_index = row_number - 2
                    poem_id = str(values_by_column.get(2, ""))
                    poem_id_hash.update(poem_id.encode("utf-8"))
                    poem_id_hash.update(b"\n")
                    if poem_id in poem_ids:
                        duplicate_poem_ids.append(poem_id)
                    poem_ids.add(poem_id)
                    if data_index in sample_indices:
                        samples[data_index] = values_by_column
                element.clear()
    return {
        "zip_integrity": "PASS",
        "sheet_names": sheet_names,
        "sheet_path": sheet_path,
        "shared_strings_mode": "SHARED_STRINGS" if strings else "INLINE_STRINGS",
        "dimension": dimension,
        "spreadsheet_rows": row_count,
        "data_rows": row_count - 1,
        "columns": max_column,
        "headers": headers,
        "poem_id_sha256": poem_id_hash.hexdigest(),
        "duplicate_poem_ids": duplicate_poem_ids,
        "samples": samples,
    }


def compare_cell(header: str, expected_raw: str, actual: Any) -> tuple[bool, float, str]:
    """按目标 Excel 类型比较 CSV 与 XML 值。"""

    value_type, expected = typed_value(header, expected_raw)
    if value_type == "blank":
        return actual is None, 0.0, value_type
    if value_type == "boolean":
        return actual is expected, 0.0, value_type
    if value_type == "integer":
        try:
            return int(actual) == expected, 0.0, value_type
        except (TypeError, ValueError):
            return False, math.inf, value_type
    if value_type == "float":
        try:
            actual_number = float(actual)
        except (TypeError, ValueError):
            return False, math.inf, value_type
        difference = abs(actual_number - expected)
        return math.isclose(actual_number, expected, rel_tol=1e-12, abs_tol=1e-15), difference, value_type
    return actual == expected, 0.0, value_type


def verify_serialization(source_csv: Path, xlsx_path: Path) -> dict[str, Any]:
    """独立比较 CSV 与 XLSX 的结构、顺序和样本值。"""

    random_indices = set(random.Random(SAMPLE_SEED).sample(range(EXPECTED_DATA_ROWS), SAMPLE_SIZE))
    position_indices = {0, EXPECTED_DATA_ROWS // 2, EXPECTED_DATA_ROWS - 1}
    sample_indices = random_indices | position_indices
    csv_evidence = collect_csv_evidence(source_csv, sample_indices)
    xlsx_evidence = collect_xlsx_evidence(xlsx_path, sample_indices)

    mismatches = []
    max_float_abs_error = 0.0
    compared_cells = 0
    for data_index in sorted(sample_indices):
        expected_row = csv_evidence["samples"].get(data_index)
        actual_row = xlsx_evidence["samples"].get(data_index)
        if expected_row is None or actual_row is None:
            mismatches.append({"data_index": data_index, "reason": "MISSING_SAMPLE_ROW"})
            continue
        for column, (header, expected_raw) in enumerate(zip(csv_evidence["headers"], expected_row)):
            actual = actual_row.get(column)
            matches, difference, value_type = compare_cell(header, expected_raw, actual)
            compared_cells += 1
            if value_type == "float" and math.isfinite(difference):
                max_float_abs_error = max(max_float_abs_error, difference)
            if not matches and len(mismatches) < 50:
                mismatches.append(
                    {
                        "data_index": data_index,
                        "spreadsheet_row": data_index + 2,
                        "field": header,
                        "expected": expected_raw,
                        "observed": actual,
                    }
                )

    expected_dimension = f"A1:{column_name(EXPECTED_COLUMNS)}{EXPECTED_DATA_ROWS + 1}"
    checks = {
        "source_data_rows": csv_evidence["data_rows"] == EXPECTED_DATA_ROWS,
        "source_columns": csv_evidence["columns"] == EXPECTED_COLUMNS,
        "xlsx_data_rows": xlsx_evidence["data_rows"] == EXPECTED_DATA_ROWS,
        "xlsx_spreadsheet_rows": xlsx_evidence["spreadsheet_rows"] == EXPECTED_DATA_ROWS + 1,
        "xlsx_columns": xlsx_evidence["columns"] == EXPECTED_COLUMNS,
        "worksheet_dimension": xlsx_evidence["dimension"] == expected_dimension,
        "headers_exact": xlsx_evidence["headers"] == csv_evidence["headers"],
        "poem_id_order_exact": xlsx_evidence["poem_id_sha256"] == csv_evidence["poem_id_sha256"],
        "source_poem_id_unique": not csv_evidence["duplicate_poem_ids"],
        "xlsx_poem_id_unique": not xlsx_evidence["duplicate_poem_ids"],
        "sample_rows_complete": len(xlsx_evidence["samples"]) == len(sample_indices),
        "sample_values_match": not mismatches,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "status": status,
        "checks": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
        "worksheet": {
            "name": "Data",
            "dimension": xlsx_evidence["dimension"],
            "data_rows": xlsx_evidence["data_rows"],
            "spreadsheet_rows": xlsx_evidence["spreadsheet_rows"],
            "columns": xlsx_evidence["columns"],
            "shared_strings_mode": xlsx_evidence["shared_strings_mode"],
        },
        "comparisons": {
            "random_sample_rows": SAMPLE_SIZE,
            "random_seed": SAMPLE_SEED,
            "required_position_data_indices": sorted(position_indices),
            "total_distinct_sample_rows": len(sample_indices),
            "fields_per_sample_row": EXPECTED_COLUMNS,
            "compared_cells": compared_cells,
            "max_float_absolute_error": max_float_abs_error,
            "mismatches": mismatches,
        },
        "poem_id_sequence_sha256": csv_evidence["poem_id_sha256"],
        "zip_integrity": xlsx_evidence["zip_integrity"],
        "sheet_names": xlsx_evidence["sheet_names"],
    }


def record_exception(
    report_path: Path,
    version_path: Path,
    verification_path: Path,
    export_timestamp: str,
    source_csv: Path,
    source_hash: str,
    output_xlsx: Path,
    output_hash: str,
) -> None:
    """把获准的导出例外与哈希写入正式报告和版本记录。"""

    version = json.loads(version_path.read_text(encoding="utf-8"))
    version["large_enriched_master_xlsx_export"] = {
        "status": "PASS",
        "exception_note": EXCEPTION_NOTE,
        "method": "XlsxWriter constant_memory=True",
        "xlsxwriter_version": xlsxwriter.__version__,
        "export_timestamp": export_timestamp,
        "source_csv": source_csv.as_posix(),
        "source_csv_sha256": source_hash,
        "output_xlsx": output_xlsx.as_posix(),
        "output_xlsx_sha256": output_hash,
        "verification_report": verification_path.as_posix(),
    }
    version_path.write_text(json.dumps(version, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    start_marker = "<!-- LARGE_ENRICHED_MASTER_XLSX_EXPORT_START -->"
    end_marker = "<!-- LARGE_ENRICHED_MASTER_XLSX_EXPORT_END -->"
    section = "\n".join(
        [
            start_marker,
            "## Large enriched master XLSX 导出例外",
            "",
            EXCEPTION_NOTE,
            "",
            f"- XlsxWriter version：`{xlsxwriter.__version__}`",
            f"- export timestamp：`{export_timestamp}`",
            f"- source CSV SHA-256：`{source_hash}`",
            f"- output XLSX SHA-256：`{output_hash}`",
            f"- verification：`{verification_path.as_posix()}`（PASS）",
            end_marker,
        ]
    )
    report = report_path.read_text(encoding="utf-8")
    if start_marker in report and end_marker in report:
        prefix, remainder = report.split(start_marker, 1)
        _, suffix = remainder.split(end_marker, 1)
        report = prefix.rstrip() + "\n\n" + section + suffix
    else:
        report = report.rstrip() + "\n\n" + section + "\n"
    report_path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, default=Path("historical_metadata"))
    parser.add_argument("--overwrite", action="store_true", help="允许替换已存在的目标 XLSX")
    return parser.parse_args()


def main() -> int:
    """流式构建、核验并记录获准例外。"""

    args = parse_args()
    root = args.historical_root.resolve()
    master_dir = root / "master"
    source_csv = master_dir / "C_K20_literary_history_master_with_author_history.csv"
    output_xlsx = master_dir / "C_K20_literary_history_master_with_author_history.xlsx"
    temporary_xlsx = master_dir / "C_K20_literary_history_master_with_author_history.partial.xlsx"
    verification_path = root / "qc" / "enriched_master_xlsx_verification.json"
    report_path = root / "QTS_historical_metadata_enrichment_report.md"
    version_path = root / "historical_metadata_version.json"

    if not source_csv.exists():
        raise FileNotFoundError(f"找不到权威 CSV：{source_csv}")
    if output_xlsx.exists() and not args.overwrite:
        raise FileExistsError(f"目标已存在；如需按同一 CSV 重建，请显式传入 --overwrite：{output_xlsx}")
    if temporary_xlsx.exists():
        temporary_xlsx.unlink()

    source_hash_before = sha256_file(source_csv)
    export_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        build_result = build_workbook(source_csv, temporary_xlsx)
        verification = verify_serialization(source_csv, temporary_xlsx)
        source_hash_after = sha256_file(source_csv)
        verification["source_csv"] = source_csv.as_posix()
        verification["source_csv_sha256"] = source_hash_before
        verification["source_csv_unchanged_during_export"] = source_hash_before == source_hash_after
        verification["temporary_output_sha256"] = sha256_file(temporary_xlsx)
        verification["export"] = {
            "method": "XlsxWriter constant_memory=True",
            "xlsxwriter_version": xlsxwriter.__version__,
            "timestamp": export_timestamp,
            **build_result,
        }
        if verification["status"] != "PASS" or not verification["source_csv_unchanged_during_export"]:
            verification["status"] = "FAIL"
            verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(f"XLSX 序列化核验失败，详见：{verification_path}")
        os.replace(temporary_xlsx, output_xlsx)
        output_hash = sha256_file(output_xlsx)
        verification["output_xlsx"] = output_xlsx.as_posix()
        verification["output_xlsx_sha256"] = output_hash
        verification["installed_output_hash_matches_verified_temporary"] = output_hash == verification["temporary_output_sha256"]
        if not verification["installed_output_hash_matches_verified_temporary"]:
            verification["status"] = "FAIL"
            raise RuntimeError("正式输出哈希与已验证临时文件不一致")
        verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        record_exception(
            report_path,
            version_path,
            verification_path,
            export_timestamp,
            source_csv,
            source_hash_before,
            output_xlsx,
            output_hash,
        )
    finally:
        if temporary_xlsx.exists():
            temporary_xlsx.unlink()

    print(
        json.dumps(
            {
                "status": "PASS",
                "output": output_xlsx.as_posix(),
                "source_csv_sha256": source_hash_before,
                "output_xlsx_sha256": output_hash,
                "xlsxwriter_version": xlsxwriter.__version__,
                "export_timestamp": export_timestamp,
                "verification": verification_path.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
