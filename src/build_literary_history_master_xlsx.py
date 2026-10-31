"""以常量内存方式生成 C_K20 文学史母数据集 XLSX。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import xlsxwriter


def numeric_value(header: str, value: str) -> Any:
    """把确定的数值字段转换为 Excel 数值，其余字段保持原样。"""

    if value == "":
        return None
    if header in {"model_row_id", "poem_id", "original_corpus_index", "poet_id"}:
        return int(value)
    if header.endswith("_prob") or header in {
        "dominant_topic_probability",
        "second_topic_probability",
        "third_topic_probability",
        "topic_entropy",
        "normalized_topic_entropy",
        "top1_top2_gap",
    }:
        return float(value)
    return value


def build_workbook(dataset_root: Path) -> dict[str, Any]:
    """从已验证 CSV 流式写入完整工作簿。"""

    master_csv = dataset_root / "master" / "C_K20_literary_history_master.csv"
    topic_metadata_csv = dataset_root / "taxonomy" / "topic_metadata.csv"
    tests_csv = dataset_root / "qc" / "final_dataset_tests.csv"
    audit_csv = dataset_root / "qc" / "document_alignment_audit.csv"
    output_path = dataset_root / "master" / "C_K20_literary_history_master.xlsx"

    workbook = xlsxwriter.Workbook(
        output_path,
        {
            "constant_memory": True,
            "in_memory": False,
            "nan_inf_to_errors": True,
            "tmpdir": str(dataset_root / "master"),
        },
    )
    workbook.set_properties(
        {
            "title": "《全唐诗》C_K20 文学史母数据集",
            "subject": "冻结 LDA 模型的 20 维 Document×Topic 概率与权威 metadata",
            "author": "《全唐诗》LDA 项目",
            "comments": "未重训模型；Data 一行对应一首 C_K20 有效诗歌。",
        }
    )
    workbook.set_calc_mode("auto")

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
    probability_format = workbook.add_format({"num_format": "0.000000000000", "font_color": "#404040", "valign": "top"})
    formula_format = workbook.add_format({"num_format": "0.000000000000", "font_color": "#1F4E78", "valign": "top"})
    section_format = workbook.add_format({"bold": True, "font_color": "#1F4E78", "bg_color": "#F2F2F2"})
    note_format = workbook.add_format({"font_color": "#404040", "text_wrap": True, "valign": "top"})
    pass_format = workbook.add_format({"font_color": "#006100", "bg_color": "#C6EFCE", "bold": True})

    data_sheet = workbook.add_worksheet("Data")
    data_sheet.hide_gridlines(2)
    data_sheet.freeze_panes(1, 7)
    data_sheet.set_default_row(15)
    data_sheet.set_row(0, 36)
    data_sheet.set_column("A:E", 14)
    data_sheet.set_column("F:G", 20)
    data_sheet.set_column("H:H", 60)
    data_sheet.set_column("I:J", 52)
    data_sheet.set_column("K:V", 13)
    data_sheet.set_column("W:AP", 14)
    data_sheet.set_column("AQ:BB", 22)

    formula_by_header = {
        "dominant_topic_probability": lambda row: f'=INDEX($W{row}:$AP{row},1,VALUE(RIGHT($AQ{row},2)))',
        "second_topic_probability": lambda row: f'=INDEX($W{row}:$AP{row},1,VALUE(RIGHT($AT{row},2)))',
        "third_topic_probability": lambda row: f'=INDEX($W{row}:$AP{row},1,VALUE(RIGHT($AW{row},2)))',
        "topic_entropy": lambda row: f'=-SUMPRODUCT(IFERROR($W{row}:$AP{row}*LN($W{row}:$AP{row}),0))',
        "normalized_topic_entropy": lambda row: f'=$AZ{row}/LN(COUNTA($W{row}:$AP{row}))',
        "top1_top2_gap": lambda row: f'=$AS{row}-$AV{row}',
    }

    with master_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        data_sheet.write_row(0, 0, headers, header_format)
        for row_index, record in enumerate(reader, start=1):
            excel_row = row_index + 1
            for column_index, header in enumerate(headers):
                raw_value = record.get(header, "")
                if header in formula_by_header:
                    cached = float(raw_value)
                    data_sheet.write_formula(row_index, column_index, formula_by_header[header](excel_row), formula_format, cached)
                    continue
                value = numeric_value(header, raw_value)
                if value is None:
                    continue
                if isinstance(value, int):
                    data_sheet.write_number(row_index, column_index, value, integer_format)
                elif isinstance(value, float):
                    data_sheet.write_number(row_index, column_index, value, probability_format)
                else:
                    data_sheet.write_string(row_index, column_index, str(value), text_format)
        master_rows = row_index
    data_sheet.autofilter(0, 0, master_rows, len(headers) - 1)

    topic_sheet = workbook.add_worksheet("Topic_Metadata")
    topic_sheet.hide_gridlines(2)
    topic_sheet.freeze_panes(1, 4)
    topic_sheet.set_row(0, 36)
    topic_sheet.set_column("A:D", 26)
    topic_sheet.set_column("E:L", 13)
    topic_sheet.set_column("M:R", 32)
    with topic_metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        topic_headers = next(reader)
        topic_sheet.write_row(0, 0, topic_headers, header_format)
        topic_rows = 0
        for topic_rows, row in enumerate(reader, start=1):
            topic_sheet.write_row(topic_rows, 0, row, text_format)
    topic_sheet.autofilter(0, 0, topic_rows, len(topic_headers) - 1)

    qc_sheet = workbook.add_worksheet("Build_QC")
    qc_sheet.hide_gridlines(2)
    qc_sheet.set_column("A:A", 14)
    qc_sheet.set_column("B:B", 46)
    qc_sheet.set_column("C:E", 24)
    qc_sheet.write_row(0, 0, ["test_id", "description", "status", "observed", "expected"], header_format)
    with tests_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=1):
            formats = [text_format, text_format, pass_format if row["status"] == "PASS" else text_format, text_format, text_format]
            for column, (key, cell_format) in enumerate(zip(["test_id", "description", "status", "observed", "expected"], formats)):
                qc_sheet.write(row_number, column, row[key], cell_format)
    audit_start = row_number + 3
    qc_sheet.write(audit_start, 0, "Document alignment audit", section_format)
    with audit_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        audit = next(csv.DictReader(handle))
    for offset, (key, value) in enumerate(audit.items(), start=1):
        qc_sheet.write(audit_start + offset, 0, key, section_format)
        qc_sheet.write(audit_start + offset, 1, value, text_format)

    codebook = workbook.add_worksheet("Codebook")
    codebook.hide_gridlines(2)
    codebook.set_column("A:A", 28)
    codebook.set_column("B:B", 110)
    codebook.write_row(0, 0, ["项目", "说明"], header_format)
    notes = [
        ("数据粒度", "Data 每行对应一首进入 C_K20 的有效诗歌；poem_id 是连接权威 metadata 的唯一主键。"),
        ("完整概率", "W:AP 保存完整20维 theta_dk。正式聚合优先使用连续概率，dominant/second/third 仅用于浏览。"),
        ("公式字段", "AS、AV、AY、AZ、BA、BB 为全行公式，并写入经 Python 全量测试验证的缓存值。"),
        ("矩阵对齐", "43,015 条矩阵记录全部唯一匹配；另有15首清洗后空诗见 qc/unmatched_documents.csv。"),
        ("Metadata 边界", "author/title/text 覆盖完整；volume、date/year/period、genre/form、place/region/location 无可靠来源，不得补造。"),
        ("Topic 使用", "Topic_Metadata 是正式标签唯一主表；NOISE 与 MIXED Topic 不得同 CLEAR Topic 等量齐观。"),
        ("冻结边界", "本工作簿只读取冻结 C_K20 结果，没有调用 fit、fit_transform 或重新训练。"),
    ]
    for row_number, (name, note) in enumerate(notes, start=1):
        codebook.write(row_number, 0, name, section_format)
        codebook.write(row_number, 1, note, note_format)
        codebook.set_row(row_number, 34)

    close_result = workbook.close()
    if close_result:
        raise RuntimeError(f"XLSX 写入失败：{close_result}")
    return {
        "output": output_path.as_posix(),
        "master_rows": master_rows,
        "master_columns": len(headers),
        "topic_metadata_rows": topic_rows,
        "formula_columns": sorted(formula_by_header),
        "size_bytes": output_path.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""

    args = parse_args()
    result = build_workbook(args.dataset_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
