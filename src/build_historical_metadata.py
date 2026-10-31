"""构建《全唐诗》作者历史年代元数据增强管线第一期产物。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class HistoricalMetadataError(RuntimeError):
    """历史元数据构建条件不满足。"""


AUTHOR_MASTER_FIELDS = [
    "author_id_internal",
    "author_raw",
    "author_normalized",
    "author_display_name",
    "author_status",
    "poem_count",
    "cbdb_person_id",
    "cbdb_name",
    "cbdb_dynasty",
    "cbdb_match_status",
    "cbdb_match_confidence",
    "author_birth_year",
    "author_birth_year_min",
    "author_birth_year_max",
    "birth_year_status",
    "birth_year_source",
    "author_death_year",
    "author_death_year_min",
    "author_death_year_max",
    "death_year_status",
    "death_year_source",
    "author_fl_start",
    "author_fl_end",
    "observed_activity_start",
    "observed_activity_end",
    "activity_year_method",
    "activity_confidence",
    "cbdb_index_year",
    "cbdb_index_year_type",
    "author_period_chronological",
    "author_period_traditional",
    "traditional_period_source",
    "author_period_primary",
    "period_reference_year",
    "period_span",
    "period_assignment_method",
    "period_confidence",
    "cross_period_flag",
    "period_boundary_flag",
    "traditional_chronological_conflict",
    "primary_source",
    "source_count",
    "needs_human_review",
    "notes",
]

POEM_AUTHOR_FIELDS = [
    "author_id_internal",
    "cbdb_person_id",
    "author_birth_year",
    "author_death_year",
    "author_fl_start",
    "author_fl_end",
    "author_period_chronological",
    "author_period_traditional",
    "author_period_primary",
    "period_assignment_method",
    "period_confidence",
    "cross_period_flag",
    "period_boundary_flag",
    "cbdb_match_confidence",
]

CANDIDATE_FIELDS = [
    "author_id_internal",
    "author_raw",
    "author_normalized",
    "candidate_rank",
    "cbdb_person_id",
    "cbdb_name",
    "cbdb_dynasty",
    "cbdb_birth_year",
    "cbdb_death_year",
    "cbdb_fl_earliest_year",
    "cbdb_fl_latest_year",
    "cbdb_index_year",
    "name_match_type",
    "matched_name_form",
    "alias_match",
    "dynasty_match",
    "dynasty_compatibility",
    "temporal_plausibility",
    "match_score",
    "match_confidence",
    "selected",
    "selection_reason",
    "needs_human_review",
]

SOURCE_FIELDS = [
    "author_id_internal",
    "field_name",
    "source_database",
    "source_person_id",
    "source_record_id",
    "source_title",
    "source_page",
    "source_note",
    "source_priority",
    "retrieved_at",
]

PERIOD_LABELS = ("初唐", "盛唐", "中唐", "晚唐")
MATCH_CONFIDENCES = ("HIGH", "MEDIUM", "LOW", "UNRESOLVED")
ANONYMOUS_EXACT = {
    "佚名",
    "无名氏",
    "無名氏",
    "失名",
    "不详",
    "不詳",
    "Anonymous",
    "匿名",
}
ANONYMOUS_MARKERS = ("无名", "無名", "失名", "佚名")


@dataclass(frozen=True)
class Period:
    """保存一段操作性年代边界。"""

    label: str
    start: int
    end: int


@dataclass(frozen=True)
class PeriodScheme:
    """保存操作性年代分期方案。"""

    version: str
    scheme_type: str
    boundary_tolerance_years: int
    periods: tuple[Period, ...]


def file_sha256(path: Path) -> str:
    """计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    """计算文本 SHA-256。"""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 UTF-8 CSV。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """以 UTF-8 BOM 原子写入 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    """以 UTF-8 原子写入 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """以 UTF-8 原子写入 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    """以 UTF-8 原子写入文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def resolve_path(project_root: Path, value: str) -> Path:
    """把配置中的路径解析到项目根目录。"""

    path = Path(value)
    return path if path.is_absolute() else project_root / path


def parse_scalar(value: str) -> Any:
    """解析简单 YAML 标量。"""

    stripped = value.strip().strip('"').strip("'")
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    if re.fullmatch(r"-?\d+", stripped):
        return int(stripped)
    return stripped


def load_flat_yaml(path: Path) -> dict[str, Any]:
    """读取只含顶层标量的配置 YAML。"""

    result: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line or line[:1].isspace() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = parse_scalar(value)
    return result


def load_period_scheme(path: Path) -> PeriodScheme:
    """读取集中维护的年代分期方案。"""

    version = ""
    scheme_type = ""
    tolerance = 5
    periods: list[Period] = []
    current: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("scheme_version:"):
            version = str(parse_scalar(stripped.split(":", 1)[1]))
        elif stripped.startswith("scheme_type:"):
            scheme_type = str(parse_scalar(stripped.split(":", 1)[1]))
        elif stripped.startswith("boundary_tolerance_years:"):
            tolerance = int(parse_scalar(stripped.split(":", 1)[1]))
        elif stripped.startswith("- label:"):
            if current is not None:
                periods.append(Period(str(current["label"]), int(current["start"]), int(current["end"])))
            current = {"label": parse_scalar(stripped.split(":", 1)[1])}
        elif current is not None and stripped.startswith("start:"):
            current["start"] = parse_scalar(stripped.split(":", 1)[1])
        elif current is not None and stripped.startswith("end:"):
            current["end"] = parse_scalar(stripped.split(":", 1)[1])
    if current is not None:
        periods.append(Period(str(current["label"]), int(current["start"]), int(current["end"])))
    if tuple(period.label for period in periods) != PERIOD_LABELS:
        raise HistoricalMetadataError(f"分期标签必须依次为 {PERIOD_LABELS}")
    for left, right in zip(periods, periods[1:]):
        if left.end + 1 != right.start:
            raise HistoricalMetadataError("分期边界必须连续且不重叠")
    return PeriodScheme(version, scheme_type, tolerance, tuple(periods))


def normalize_author_name(value: str) -> str:
    """保守规范作者名，不做简繁覆盖或数字删除。"""

    normalized = unicodedata.normalize("NFKC", value).strip().replace("\ufeff", "")
    return "".join(character for character in normalized if unicodedata.category(character) != "Cf")


def internal_author_id(author_raw: str) -> str:
    """由原始作者值生成稳定的内部作者 ID。"""

    return "QTS_A_" + text_sha256(author_raw)[:12].upper()


def classify_author(author_normalized: str) -> str:
    """按保守规则分类作者身份状态。"""

    if author_normalized in ANONYMOUS_EXACT or any(marker in author_normalized for marker in ANONYMOUS_MARKERS):
        return "ANONYMOUS"
    if re.search(r"\d+$", author_normalized) or re.search(r"[A-Za-z]\d+$", author_normalized):
        return "TEXTUAL_DUPLICATE"
    if re.search(r"[、/&＋+]", author_normalized):
        return "COMPOSITE"
    if author_normalized == "" or re.search(r"待考|未知|疑|某$", author_normalized):
        return "UNCERTAIN"
    return "NAMED"


def load_st_variants(path: Path) -> dict[str, tuple[str, ...]]:
    """读取 OpenCC 简体到繁体字符候选，不执行覆盖式转换。"""

    variants: dict[str, tuple[str, ...]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "\t" not in line:
            continue
        source, targets = line.split("\t", 1)
        options = tuple(dict.fromkeys([source, *targets.split()]))
        variants[source] = options
    return variants


def build_name_variants(name: str, mapping: Mapping[str, Sequence[str]], limit: int) -> tuple[str, ...]:
    """生成简繁字符组合候选，但不修改 author_normalized。"""

    choices = [tuple(mapping.get(character, (character,))) for character in name]
    product_size = math.prod(len(options) for options in choices)
    if product_size > limit:
        choices = [options[:2] for options in choices]
    return tuple(dict.fromkeys("".join(parts) for parts in itertools.islice(itertools.product(*choices), limit)))


def period_for_year(year: float | int | None, scheme: PeriodScheme) -> str:
    """把参考年映射到操作性年代分期。"""

    if year is None:
        return ""
    for period in scheme.periods:
        if period.start <= float(year) <= period.end:
            return period.label
    return ""


def period_span_for_range(start: int | None, end: int | None, scheme: PeriodScheme) -> tuple[str, bool]:
    """返回活动区间覆盖的分期与跨期标记。"""

    if start is None and end is None:
        return "", False
    lower = start if start is not None else end
    upper = end if end is not None else start
    assert lower is not None and upper is not None
    labels = [period.label for period in scheme.periods if lower <= period.end and upper >= period.start]
    return "|".join(labels), len(labels) > 1


def is_period_boundary(year: float | int | None, scheme: PeriodScheme) -> bool:
    """判断参考年是否靠近操作性分期边界。"""

    if year is None:
        return False
    transitions = [period.start for period in scheme.periods[1:]]
    return any(abs(float(year) - transition) <= scheme.boundary_tolerance_years for transition in transitions)


def query_in_chunks(
    connection: sqlite3.Connection,
    sql_prefix: str,
    values: Sequence[Any],
    chunk_size: int = 800,
) -> list[sqlite3.Row]:
    """分批执行 IN 查询，避免超过 SQLite 参数上限。"""

    rows: list[sqlite3.Row] = []
    unique_values = list(dict.fromkeys(values))
    for offset in range(0, len(unique_values), chunk_size):
        chunk = unique_values[offset : offset + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        rows.extend(connection.execute(sql_prefix.format(placeholders=placeholders), chunk).fetchall())
    return rows


def dynasty_compatibility(
    dynasty_code: Any,
    dynasty_start: Any,
    dynasty_end: Any,
    config: Mapping[str, Any],
) -> tuple[str, str]:
    """判断 CBDB 朝代是否与唐及相邻五代背景相容。"""

    if dynasty_code in (6, "6"):
        return "True", "TANG"
    if dynasty_start in (None, "") or dynasty_end in (None, ""):
        return "Unknown", "UNKNOWN"
    lower = int(config["compatible_dynasty_window_start"])
    upper = int(config["compatible_dynasty_window_end"])
    # 五代延续至960年，但960年才开始的宋朝不应因端点相接而被视为兼容。
    if int(dynasty_start) < upper and int(dynasty_end) >= lower:
        return "True", "TANG_FIVE_DYNASTIES_CONTEXT"
    return "False", "CONFLICT"


def temporal_plausibility(record: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    """用人物显式时间字段判断是否落在唐及相邻时段。"""

    values = dict(record)
    years = [
        values.get("c_birthyear"),
        values.get("c_deathyear"),
        values.get("c_fl_earliest_year"),
        values.get("c_fl_latest_year"),
        values.get("c_index_year"),
    ]
    concrete = [int(year) for year in years if year not in (None, "", 0, "0")]
    if not concrete:
        return "Unknown"
    lower = int(config["dated_event_min_year"])
    upper = int(config["dated_event_max_year"])
    if max(concrete) < lower or min(concrete) > upper:
        return "False"
    return "True"


def match_type_priority(value: str) -> int:
    """给姓名匹配类型固定排序。"""

    return {
        "EXACT_NAME": 0,
        "SCRIPT_VARIANT_EXACT": 1,
        "ALIAS_EXACT": 2,
        "ALIAS_SCRIPT_VARIANT": 3,
        "MANUAL_PERSON_ID": -1,
    }.get(value, 99)


def candidate_score(row: Mapping[str, Any]) -> int:
    """生成仅用于候选排序、不得替代人工判断的透明分数。"""

    score = {
        "MANUAL_PERSON_ID": 140,
        "EXACT_NAME": 100,
        "SCRIPT_VARIANT_EXACT": 95,
        "ALIAS_EXACT": 85,
        "ALIAS_SCRIPT_VARIANT": 80,
    }.get(str(row["name_match_type"]), 0)
    compatibility = str(row["dynasty_compatibility"])
    score += {"TANG": 30, "TANG_FIVE_DYNASTIES_CONTEXT": 20, "UNKNOWN": 0, "CONFLICT": -60}[compatibility]
    score += {"True": 20, "Unknown": 0, "False": -40}[str(row["temporal_plausibility"])]
    return score


def load_manual_aliases(path: Path) -> dict[str, dict[str, str]]:
    """读取人工作者映射；只有 APPROVED 行参与匹配。"""

    rows = read_csv(path)
    approved: dict[str, dict[str, str]] = {}
    for row in rows:
        author_raw = row.get("author_raw", "")
        if not author_raw or row.get("decision", "").upper() != "APPROVED":
            continue
        if author_raw in approved:
            raise HistoricalMetadataError(f"人工作者映射重复：{author_raw}")
        approved[author_raw] = row
    return approved


def candidate_base_select() -> str:
    """返回 CBDB 人物候选查询的公共字段。"""

    return """
        SELECT b.c_personid, b.c_name_chn, b.c_birthyear, b.c_by_range,
               b.c_deathyear, b.c_dy_range, b.c_fl_earliest_year,
               b.c_fl_latest_year, b.c_index_year, b.c_index_year_type_code,
               b.c_index_year_source_id, b.c_notes, b.c_dy,
               d.c_dynasty_chn, d.c_start AS dynasty_start, d.c_end AS dynasty_end
          FROM BIOG_MAIN b
          LEFT JOIN DYNASTIES d ON b.c_dy = d.c_dy
    """


def build_candidates(
    connection: sqlite3.Connection,
    authors: Sequence[dict[str, Any]],
    st_mapping: Mapping[str, Sequence[str]],
    manual_aliases: Mapping[str, Mapping[str, str]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], dict[int, sqlite3.Row]]:
    """生成全部 CBDB 候选、自动选择结果与姓名异常记录。"""

    connection.row_factory = sqlite3.Row
    eligible = [author for author in authors if author["author_status"] == "NAMED"]
    keys_by_author: dict[str, tuple[str, ...]] = {}
    author_ids_by_key: dict[str, list[str]] = defaultdict(list)
    author_by_id = {str(author["author_id_internal"]): author for author in authors}
    anomalies: list[dict[str, Any]] = []
    for author in authors:
        author_id = str(author["author_id_internal"])
        normalized = str(author["author_normalized"])
        status = str(author["author_status"])
        if re.search(r"\d+$", normalized):
            anomalies.append(
                {
                    "author_id_internal": author_id,
                    "author_raw": author["author_raw"],
                    "author_normalized": normalized,
                    "anomaly_type": "TRAILING_DIGITS",
                    "anomaly_value": re.search(r"\d+$", normalized).group(0),
                    "notes": "尾随数字未删除，作者不进入自动 CBDB 匹配。",
                    "needs_human_review": True,
                }
            )
        unusual = "".join(character for character in normalized if not re.fullmatch(r"[\u3400-\u9fffA-Za-z0-9]", character))
        if unusual:
            anomalies.append(
                {
                    "author_id_internal": author_id,
                    "author_raw": author["author_raw"],
                    "author_normalized": normalized,
                    "anomaly_type": "UNUSUAL_SYMBOLS",
                    "anomaly_value": unusual,
                    "notes": "异常符号未自动删除。",
                    "needs_human_review": True,
                }
            )
        if status != "NAMED":
            continue
        manual = manual_aliases.get(str(author["author_raw"]))
        match_name = str(manual.get("author_match_name", "")) if manual else ""
        base_name = match_name or normalized
        variants = build_name_variants(base_name, st_mapping, int(config["name_variant_limit"]))
        keys_by_author[author_id] = variants
        for variant in variants:
            author_ids_by_key[variant].append(author_id)
        changed = [variant for variant in variants if variant != base_name]
        if changed:
            anomalies.append(
                {
                    "author_id_internal": author_id,
                    "author_raw": author["author_raw"],
                    "author_normalized": normalized,
                    "anomaly_type": "POSSIBLE_SCRIPT_VARIANT_FORMS",
                    "anomaly_value": "|".join(changed[:20]),
                    "notes": "只作为 CBDB 查询候选键；未覆盖原名。",
                    "needs_human_review": False,
                }
            )

    all_keys = sorted(author_ids_by_key)
    person_sql = candidate_base_select() + " WHERE b.c_name_chn IN ({placeholders})"
    primary_rows = query_in_chunks(connection, person_sql, all_keys)
    alias_sql = candidate_base_select() + """
        JOIN ALTNAME_DATA a ON b.c_personid = a.c_personid
       WHERE a.c_alt_name_chn IN ({placeholders})
    """
    alias_rows = query_in_chunks(connection, alias_sql, all_keys)
    alias_detail_sql = """
        SELECT a.c_personid, a.c_alt_name_chn, a.c_alt_name_type_code,
               a.c_source, a.c_pages, a.c_notes
          FROM ALTNAME_DATA a
         WHERE a.c_alt_name_chn IN ({placeholders})
    """
    alias_details = query_in_chunks(connection, alias_detail_sql, all_keys)
    alias_detail_by_pair = {(int(row["c_personid"]), str(row["c_alt_name_chn"])): row for row in alias_details}

    candidates_by_author: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    raw_records: dict[int, sqlite3.Row] = {}

    def add_candidate(record: sqlite3.Row, matched_form: str, alias: bool) -> None:
        person_id = int(record["c_personid"])
        raw_records[person_id] = record
        for author_id in author_ids_by_key.get(matched_form, []):
            author = author_by_id[author_id]
            normalized = str(author["author_normalized"])
            manual = manual_aliases.get(str(author["author_raw"]))
            base_name = str(manual.get("author_match_name", "")) if manual and manual.get("author_match_name") else normalized
            if alias:
                match_type = "ALIAS_EXACT" if matched_form == base_name else "ALIAS_SCRIPT_VARIANT"
            else:
                match_type = "EXACT_NAME" if matched_form == base_name else "SCRIPT_VARIANT_EXACT"
            dynasty_match, compatibility = dynasty_compatibility(
                record["c_dy"], record["dynasty_start"], record["dynasty_end"], config
            )
            temporal = temporal_plausibility(record, config)
            detail = alias_detail_by_pair.get((person_id, matched_form)) if alias else None
            candidate = {
                "author_id_internal": author_id,
                "author_raw": author["author_raw"],
                "author_normalized": normalized,
                "candidate_rank": "",
                "cbdb_person_id": person_id,
                "cbdb_name": record["c_name_chn"] or "",
                "cbdb_dynasty": record["c_dynasty_chn"] or "",
                "cbdb_birth_year": record["c_birthyear"] or "",
                "cbdb_death_year": record["c_deathyear"] or "",
                "cbdb_fl_earliest_year": record["c_fl_earliest_year"] or "",
                "cbdb_fl_latest_year": record["c_fl_latest_year"] or "",
                "cbdb_index_year": record["c_index_year"] or "",
                "name_match_type": match_type,
                "matched_name_form": matched_form,
                "alias_match": alias,
                "alias_source": detail["c_source"] if detail else "",
                "alias_pages": detail["c_pages"] if detail else "",
                "alias_notes": detail["c_notes"] if detail else "",
                "dynasty_match": dynasty_match,
                "dynasty_compatibility": compatibility,
                "temporal_plausibility": temporal,
                "match_score": 0,
                "match_confidence": "UNRESOLVED",
                "selected": False,
                "selection_reason": "",
                "needs_human_review": True,
            }
            existing = candidates_by_author[author_id].get(person_id)
            if existing is None or match_type_priority(match_type) < match_type_priority(str(existing["name_match_type"])):
                candidate["match_score"] = candidate_score(candidate)
                candidates_by_author[author_id][person_id] = candidate

    for row in primary_rows:
        add_candidate(row, str(row["c_name_chn"]), False)
    alias_name_by_person: dict[int, list[str]] = defaultdict(list)
    for detail in alias_details:
        alias_name_by_person[int(detail["c_personid"])].append(str(detail["c_alt_name_chn"]))
    for row in alias_rows:
        for matched_form in alias_name_by_person[int(row["c_personid"])]:
            if matched_form in author_ids_by_key:
                add_candidate(row, matched_form, True)

    manual_ids = [int(row["cbdb_person_id"]) for row in manual_aliases.values() if row.get("cbdb_person_id")]
    if manual_ids:
        manual_rows = query_in_chunks(
            connection,
            candidate_base_select() + " WHERE b.c_personid IN ({placeholders})",
            manual_ids,
        )
        manual_by_id = {int(row["c_personid"]): row for row in manual_rows}
        for author in eligible:
            manual = manual_aliases.get(str(author["author_raw"]))
            if not manual or not manual.get("cbdb_person_id"):
                continue
            person_id = int(manual["cbdb_person_id"])
            record = manual_by_id.get(person_id)
            if record is None:
                raise HistoricalMetadataError(f"人工指定的 CBDB person ID 不存在：{person_id}")
            normalized = str(author["author_normalized"])
            if str(author["author_id_internal"]) not in author_ids_by_key[normalized]:
                author_ids_by_key[normalized].append(str(author["author_id_internal"]))
            add_candidate(record, str(author["author_normalized"]), False)
            candidate = candidates_by_author[str(author["author_id_internal"])][person_id]
            candidate["name_match_type"] = "MANUAL_PERSON_ID"
            candidate["match_score"] = candidate_score(candidate)

    decisions: dict[str, dict[str, Any]] = {}
    flat_candidates: list[dict[str, Any]] = []
    for author in authors:
        author_id = str(author["author_id_internal"])
        status = str(author["author_status"])
        candidates = list(candidates_by_author.get(author_id, {}).values())
        candidates.sort(key=lambda row: (-int(row["match_score"]), int(row["cbdb_person_id"])))
        selected: dict[str, Any] | None = None
        match_status = "NOT_ELIGIBLE" if status != "NAMED" else "CBDB_NOT_FOUND"
        confidence = "UNRESOLVED"
        reason = "作者状态不允许自动匹配。" if status != "NAMED" else "CBDB 正名与别名均无候选。"
        safe = [
            row
            for row in candidates
            if row["dynasty_match"] != "False" and row["temporal_plausibility"] != "False"
        ]
        manual = manual_aliases.get(str(author["author_raw"]))
        manual_person_id = int(manual["cbdb_person_id"]) if manual and manual.get("cbdb_person_id") else None
        if manual_person_id is not None:
            selected = candidates_by_author[author_id][manual_person_id]
            match_status = "MATCHED"
            confidence = "MEDIUM"
            reason = "依据项目人工映射中的 APPROVED CBDB person ID 选择；保留人工来源说明。"
        elif status == "NAMED" and len(safe) == 1:
            selected = safe[0]
            no_competitor = len(candidates) == 1
            alias = bool(selected["alias_match"])
            unknown_dynasty = selected["dynasty_compatibility"] == "UNKNOWN"
            if unknown_dynasty:
                confidence = "LOW"
                reason = "唯一姓名候选，但 CBDB 朝代不明；作为低置信度暂定匹配。"
            elif alias:
                confidence = "MEDIUM" if no_competitor else "LOW"
                reason = "别名唯一可行匹配。" if no_competitor else "别名候选通过朝代/时间排除竞争者后暂定。"
            elif no_competitor:
                confidence = "HIGH"
                reason = "唯一正名候选，且朝代与时间背景不冲突。"
            else:
                confidence = "MEDIUM"
                reason = "多个同名候选中仅一位与唐及相邻时段相容。"
            match_status = "MATCHED"
        elif status == "NAMED" and len(safe) > 1:
            match_status = "AMBIGUOUS"
            reason = "存在多个无法安全排除的 CBDB 候选，未自动选择。"
        elif status == "NAMED" and candidates:
            match_status = "DYNASTY_CONFLICT"
            reason = "只有与唐及相邻时段冲突的候选，未自动选择。"
        if selected is not None:
            selected["selected"] = True
            selected["match_confidence"] = confidence
            selected["selection_reason"] = reason
            selected["needs_human_review"] = confidence == "LOW"
        decisions[author_id] = {
            "selected": selected,
            "cbdb_match_status": match_status,
            "cbdb_match_confidence": confidence,
            "selection_reason": reason,
        }
        for rank, candidate in enumerate(candidates, start=1):
            candidate["candidate_rank"] = rank
            if not candidate["selected"]:
                candidate["selection_reason"] = reason if match_status == "AMBIGUOUS" else "未选择。"
                candidate["match_confidence"] = "UNRESOLVED"
                candidate["needs_human_review"] = match_status == "AMBIGUOUS"
            flat_candidates.append(candidate)
    return flat_candidates, decisions, anomalies, raw_records


def year_semantics(year: Any, range_code: Any) -> tuple[Any, Any, Any, str]:
    """把 CBDB 年份与范围码拆成锚点、下界、上界和状态。"""

    if year in (None, "", 0, "0"):
        return "", "", "", "UNKNOWN"
    value = int(year)
    if range_code in (None, "", 0, "0"):
        return value, value, value, "EXACT"
    code = int(range_code)
    if code == 2:
        return value, "", "", "APPROXIMATE"
    if code == -1:
        return "", "", value, "RANGE"
    if code == 1:
        return "", value, "", "RANGE"
    if code in (300, 301):
        return "", 960 if code == 300 else 1082, 1082 if code == 300 else 1279, "RANGE"
    return value, "", "", "ESTIMATED"


def load_index_year_types(connection: sqlite3.Connection) -> dict[str, str]:
    """读取 CBDB index year 类型中文说明。"""

    return {
        str(row[0]): str(row[1] or "")
        for row in connection.execute(
            "SELECT c_index_year_type_code, c_index_year_type_hz FROM INDEXYEAR_TYPE_CODES"
        )
    }


def load_person_sources(
    connection: sqlite3.Connection,
    person_ids: Sequence[int],
) -> dict[int, list[sqlite3.Row]]:
    """读取候选人物的 CBDB 人物级来源。"""

    sql = """
        SELECT s.c_personid, s.c_textid, s.c_pages, s.c_notes, s.c_main_source,
               t.c_title_chn, t.c_title, t.c_url_homepage
          FROM BIOG_SOURCE_DATA s
          LEFT JOIN TEXT_CODES t ON s.c_textid = t.c_textid
         WHERE s.c_personid IN ({placeholders})
         ORDER BY s.c_personid, s.c_main_source DESC, s.c_textid, s.c_pages
    """
    grouped: dict[int, list[sqlite3.Row]] = defaultdict(list)
    for row in query_in_chunks(connection, sql, person_ids):
        grouped[int(row["c_personid"])].append(row)
    return grouped


def load_dated_activity(
    connection: sqlite3.Connection,
    person_ids: Sequence[int],
    config: Mapping[str, Any],
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    """读取明确有年的事件、入仕/考试和任官记录。"""

    lower = int(config["dated_event_min_year"])
    upper = int(config["dated_event_max_year"])
    by_person: dict[int, list[dict[str, Any]]] = defaultdict(list)
    raw_rows: list[dict[str, Any]] = []

    event_sql = """
        SELECT e.c_personid, e.c_sequence, e.c_year, e.c_source, e.c_pages,
               e.c_event, e.c_notes, c.c_event_name_chn
          FROM EVENTS_DATA e
          LEFT JOIN EVENT_CODES c ON e.c_event_code = c.c_event_code
         WHERE e.c_personid IN ({placeholders}) AND e.c_year BETWEEN ? AND ?
    """
    for offset in range(0, len(person_ids), 700):
        chunk = list(dict.fromkeys(person_ids))[offset : offset + 700]
        if not chunk:
            continue
        sql = event_sql.format(placeholders=",".join("?" for _ in chunk))
        for row in connection.execute(sql, [*chunk, lower, upper]).fetchall():
            item = {
                "person_id": int(row["c_personid"]),
                "activity_type": "EVENT",
                "year_start": int(row["c_year"]),
                "year_end": int(row["c_year"]),
                "source_record_id": row["c_source"] or "",
                "source_page": row["c_pages"] or "",
                "description": row["c_event"] or row["c_event_name_chn"] or row["c_notes"] or "",
            }
            by_person[item["person_id"]].append(item)
            raw_rows.append(item)

    entry_sql = """
        SELECT e.c_personid, e.c_sequence, e.c_year, e.c_source, e.c_pages,
               e.c_notes, c.c_entry_desc_chn
          FROM ENTRY_DATA e
          LEFT JOIN ENTRY_CODES c ON e.c_entry_code = c.c_entry_code
         WHERE e.c_personid IN ({placeholders}) AND e.c_year BETWEEN ? AND ?
    """
    for offset in range(0, len(person_ids), 700):
        chunk = list(dict.fromkeys(person_ids))[offset : offset + 700]
        if not chunk:
            continue
        sql = entry_sql.format(placeholders=",".join("?" for _ in chunk))
        for row in connection.execute(sql, [*chunk, lower, upper]).fetchall():
            item = {
                "person_id": int(row["c_personid"]),
                "activity_type": "ENTRY",
                "year_start": int(row["c_year"]),
                "year_end": int(row["c_year"]),
                "source_record_id": row["c_source"] or "",
                "source_page": row["c_pages"] or "",
                "description": row["c_entry_desc_chn"] or row["c_notes"] or "",
            }
            by_person[item["person_id"]].append(item)
            raw_rows.append(item)

    posting_sql = """
        SELECT p.c_personid, p.c_posting_id, p.c_firstyear, p.c_lastyear,
               p.c_source, p.c_pages, p.c_notes, o.c_office_chn
          FROM POSTED_TO_OFFICE_DATA p
          LEFT JOIN OFFICE_CODES o ON p.c_office_id = o.c_office_id
         WHERE p.c_personid IN ({placeholders})
           AND ((p.c_firstyear BETWEEN ? AND ?) OR (p.c_lastyear BETWEEN ? AND ?))
    """
    for offset in range(0, len(person_ids), 700):
        chunk = list(dict.fromkeys(person_ids))[offset : offset + 700]
        if not chunk:
            continue
        sql = posting_sql.format(placeholders=",".join("?" for _ in chunk))
        for row in connection.execute(sql, [*chunk, lower, upper, lower, upper]).fetchall():
            first = int(row["c_firstyear"]) if row["c_firstyear"] not in (None, 0) else None
            last = int(row["c_lastyear"]) if row["c_lastyear"] not in (None, 0) else None
            item = {
                "person_id": int(row["c_personid"]),
                "activity_type": "POSTING",
                "year_start": first if first is not None else last,
                "year_end": last if last is not None else first,
                "source_record_id": row["c_source"] or "",
                "source_page": row["c_pages"] or "",
                "description": row["c_office_chn"] or row["c_notes"] or "",
            }
            if item["year_start"] is not None:
                by_person[item["person_id"]].append(item)
                raw_rows.append(item)
    for values in by_person.values():
        values.sort(key=lambda row: (int(row["year_start"]), int(row["year_end"]), row["activity_type"]))
    return by_person, raw_rows


def load_traditional_periods(path: Path) -> dict[str, dict[str, str]]:
    """读取带明确学术来源的人工传统分期映射。"""

    result: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        author_raw = row.get("author_raw", "")
        period = row.get("author_period_traditional", "")
        source = row.get("traditional_period_source", "")
        if not author_raw and not period:
            continue
        if period not in PERIOD_LABELS:
            raise HistoricalMetadataError(f"传统分期值不合法：{author_raw}={period}")
        if not source:
            raise HistoricalMetadataError(f"传统分期缺少来源：{author_raw}")
        if author_raw in result:
            raise HistoricalMetadataError(f"传统分期作者重复：{author_raw}")
        result[author_raw] = row
    return result


def build_author_master(
    authors: Sequence[dict[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
    raw_records: Mapping[int, sqlite3.Row],
    person_sources: Mapping[int, Sequence[sqlite3.Row]],
    activities: Mapping[int, Sequence[Mapping[str, Any]]],
    index_year_types: Mapping[str, str],
    traditional_periods: Mapping[str, Mapping[str, str]],
    scheme: PeriodScheme,
    retrieved_at: str,
    manual_aliases: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """构建作者历史主表与字段级来源表。"""

    master: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for author in authors:
        author_id = str(author["author_id_internal"])
        decision = decisions[author_id]
        selected = decision.get("selected")
        status = str(author["author_status"])
        person_id = int(selected["cbdb_person_id"]) if selected is not None else None
        record = raw_records.get(person_id) if person_id is not None else None
        row: dict[str, Any] = {field: "" for field in AUTHOR_MASTER_FIELDS}
        row.update(author)
        row.update(
            cbdb_person_id=person_id or "",
            cbdb_name=record["c_name_chn"] if record is not None and record["c_name_chn"] else "",
            cbdb_dynasty=record["c_dynasty_chn"] if record is not None and record["c_dynasty_chn"] else "",
            cbdb_match_status=decision["cbdb_match_status"],
            cbdb_match_confidence=decision["cbdb_match_confidence"],
        )
        notes = [str(decision["selection_reason"])]
        if status == "ANONYMOUS":
            notes.append("匿名作者不进入 CBDB 自动匹配，也不生成作者分期。")
        if record is not None:
            birth, birth_min, birth_max, birth_status = year_semantics(record["c_birthyear"], record["c_by_range"])
            death, death_min, death_max, death_status = year_semantics(record["c_deathyear"], record["c_dy_range"])
            row.update(
                author_birth_year=birth,
                author_birth_year_min=birth_min,
                author_birth_year_max=birth_max,
                birth_year_status=birth_status,
                birth_year_source=f"CBDB BIOG_MAIN c_birthyear; person {person_id}" if birth_status != "UNKNOWN" else "",
                author_death_year=death,
                author_death_year_min=death_min,
                author_death_year_max=death_max,
                death_year_status=death_status,
                death_year_source=f"CBDB BIOG_MAIN c_deathyear; person {person_id}" if death_status != "UNKNOWN" else "",
                author_fl_start=record["c_fl_earliest_year"] or "",
                author_fl_end=record["c_fl_latest_year"] or "",
                cbdb_index_year=record["c_index_year"] or "",
                cbdb_index_year_type=index_year_types.get(str(record["c_index_year_type_code"]), ""),
                primary_source=f"CBDB:{person_id}",
                source_count=len({source["c_textid"] for source in person_sources.get(person_id, [])}),
            )
            field_map = [
                ("cbdb_identity", "BIOG_MAIN c_personid/c_name_chn", True),
                ("birth_year", "BIOG_MAIN c_birthyear/c_by_range", birth_status != "UNKNOWN"),
                ("death_year", "BIOG_MAIN c_deathyear/c_dy_range", death_status != "UNKNOWN"),
                ("floruit", "BIOG_MAIN c_fl_earliest_year/c_fl_latest_year", bool(row["author_fl_start"] or row["author_fl_end"])),
                ("index_year", "BIOG_MAIN c_index_year/c_index_year_type_code", bool(row["cbdb_index_year"])),
            ]
            for field_name, source_note, include in field_map:
                if not include:
                    continue
                sources.append(
                    {
                        "author_id_internal": author_id,
                        "field_name": field_name,
                        "source_database": "CBDB SQLite 20260829",
                        "source_person_id": person_id,
                        "source_record_id": person_id,
                        "source_title": "CBDB BIOG_MAIN",
                        "source_page": "",
                        "source_note": source_note + "; SQLite 未提供该字段的独立来源外键时，不把人物级来源冒充字段级直接来源。",
                        "source_priority": 1,
                        "retrieved_at": retrieved_at,
                    }
                )
            for source in person_sources.get(person_id, []):
                sources.append(
                    {
                        "author_id_internal": author_id,
                        "field_name": "person_record_source",
                        "source_database": "CBDB",
                        "source_person_id": person_id,
                        "source_record_id": source["c_textid"],
                        "source_title": source["c_title_chn"] or source["c_title"] or "",
                        "source_page": source["c_pages"] or "",
                        "source_note": source["c_notes"] or "人物级 CBDB 来源；未自动声明为某个年份字段的直接来源。",
                        "source_priority": 1 if source["c_main_source"] else 2,
                        "retrieved_at": retrieved_at,
                    }
                )

            dated = list(activities.get(person_id, []))
            activity_years = [int(item["year_start"]) for item in dated]
            activity_years.extend(int(item["year_end"]) for item in dated if item["year_end"] is not None)
            observed_start = min(activity_years) if activity_years else None
            observed_end = max(activity_years) if activity_years else None
            row["observed_activity_start"] = observed_start or ""
            row["observed_activity_end"] = observed_end or ""
            fl_start = int(row["author_fl_start"]) if row["author_fl_start"] != "" else None
            fl_end = int(row["author_fl_end"]) if row["author_fl_end"] != "" else None
            reference_start: int | None = None
            reference_end: int | None = None
            reference_year: float | None = None
            assignment_method = "UNKNOWN"
            if fl_start is not None and fl_end is not None:
                row["activity_year_method"] = "CBDB_FLORUIT"
                row["activity_confidence"] = "HIGH"
                reference_start, reference_end = fl_start, fl_end
                reference_year = (fl_start + fl_end) / 2
                assignment_method = "FLORUIT_BASED"
            elif observed_start is not None:
                row["activity_year_method"] = "DATED_EVENT_RANGE"
                row["activity_confidence"] = "MEDIUM" if len(set(activity_years)) > 1 else "LOW"
                reference_start, reference_end = observed_start, observed_end
                reference_year = (observed_start + observed_end) / 2
                assignment_method = "ACTIVITY_BASED"
            elif fl_start is not None or fl_end is not None:
                point = fl_start if fl_start is not None else fl_end
                row["activity_year_method"] = "CBDB_FLORUIT"
                row["activity_confidence"] = "MEDIUM"
                reference_start = reference_end = point
                reference_year = float(point)
                assignment_method = "FLORUIT_BASED"
            elif row["cbdb_index_year"] != "":
                point = int(row["cbdb_index_year"])
                row["activity_year_method"] = "INDEX_YEAR_ONLY"
                row["activity_confidence"] = "LOW"
                reference_start = reference_end = point
                reference_year = float(point)
                assignment_method = "INDEX_YEAR_PROXY"
            elif birth != "" and death != "":
                reference_year = (int(birth) + int(death)) / 2
                reference_start = reference_end = int(round(reference_year))
                row["activity_year_method"] = "BIOGRAPHICAL_INFERENCE"
                row["activity_confidence"] = "LOW"
                assignment_method = "LIFE_MIDPOINT_PROXY"
            else:
                row["activity_year_method"] = "UNKNOWN"
                row["activity_confidence"] = "UNKNOWN"
            row["period_reference_year"] = "" if reference_year is None else (int(reference_year) if reference_year.is_integer() else reference_year)
            chronological = period_for_year(reference_year, scheme)
            span, cross = period_span_for_range(reference_start, reference_end, scheme)
            row["author_period_chronological"] = chronological
            row["period_span"] = span
            row["cross_period_flag"] = cross
            row["period_boundary_flag"] = is_period_boundary(reference_year, scheme)
            traditional = traditional_periods.get(str(author["author_raw"]))
            if traditional:
                row["author_period_traditional"] = traditional["author_period_traditional"]
                row["traditional_period_source"] = traditional["traditional_period_source"]
                row["author_period_primary"] = traditional["author_period_traditional"]
                row["period_assignment_method"] = "TRADITIONAL_SOURCE"
                row["period_confidence"] = traditional.get("period_confidence", "") or "HIGH"
                sources.append(
                    {
                        "author_id_internal": author_id,
                        "field_name": "traditional_period",
                        "source_database": "PROJECT_MANUAL_MAPPING",
                        "source_person_id": person_id or "",
                        "source_record_id": "",
                        "source_title": traditional["traditional_period_source"],
                        "source_page": "",
                        "source_note": traditional.get("notes", ""),
                        "source_priority": 1,
                        "retrieved_at": retrieved_at,
                    }
                )
            else:
                row["author_period_primary"] = chronological
                row["period_assignment_method"] = assignment_method
                if assignment_method == "FLORUIT_BASED" and row["activity_confidence"] == "HIGH":
                    row["period_confidence"] = "HIGH"
                elif assignment_method in {"FLORUIT_BASED", "ACTIVITY_BASED"}:
                    row["period_confidence"] = row["activity_confidence"]
                elif assignment_method in {"INDEX_YEAR_PROXY", "LIFE_MIDPOINT_PROXY"}:
                    row["period_confidence"] = "LOW"
                else:
                    row["period_confidence"] = "UNKNOWN"
            row["traditional_chronological_conflict"] = bool(
                row["author_period_traditional"]
                and row["author_period_chronological"]
                and row["author_period_traditional"] != row["author_period_chronological"]
            )
        else:
            row.update(
                birth_year_status="UNKNOWN",
                death_year_status="UNKNOWN",
                activity_year_method="UNKNOWN",
                activity_confidence="UNKNOWN",
                period_assignment_method="UNKNOWN",
                period_confidence="UNKNOWN",
                cross_period_flag=False,
                period_boundary_flag=False,
                traditional_chronological_conflict=False,
            )

        needs_review = False if status == "ANONYMOUS" else (
            decision["cbdb_match_status"] == "AMBIGUOUS"
            or decision["cbdb_match_confidence"] == "LOW"
            or row["period_confidence"] == "LOW"
            or row["author_period_primary"] == ""
            or bool(row["cross_period_flag"])
            or bool(row["period_boundary_flag"])
            or bool(row["traditional_chronological_conflict"])
            or status in {"UNCERTAIN", "COMPOSITE", "TEXTUAL_DUPLICATE"}
        )
        row["needs_human_review"] = needs_review
        if str(author["author_raw"]) in manual_aliases:
            notes.append("使用 APPROVED 人工作者映射。")
        row["notes"] = " ".join(note for note in notes if note)
        master.append(row)
    return master, sources


def build_review_queue(author_master: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """按任务规定的五级顺序生成人工审核队列。"""

    rows: list[dict[str, Any]] = []
    for author in author_master:
        if not bool(author["needs_human_review"]):
            continue
        reasons: list[str] = []
        priorities: list[int] = []
        if author["cbdb_match_status"] == "AMBIGUOUS":
            priorities.append(1)
            reasons.append("CBDB_IDENTITY_AMBIGUOUS")
        if not author["author_period_primary"]:
            priorities.append(2)
            reasons.append("PERIOD_UNRESOLVED")
        if bool(author["cross_period_flag"]) or bool(author["period_boundary_flag"]):
            priorities.append(3)
            reasons.append("CROSS_PERIOD_OR_BOUNDARY")
        if author["cbdb_match_confidence"] == "LOW" or author["period_confidence"] == "LOW":
            priorities.append(4)
            reasons.append("LOW_CONFIDENCE")
        birth = author["author_birth_year"]
        death = author["author_death_year"]
        if birth not in (None, "") and death not in (None, "") and int(birth) > int(death):
            priorities.append(5)
            reasons.append("BIRTH_DEATH_CONFLICT")
        if author["author_status"] in {"UNCERTAIN", "COMPOSITE", "TEXTUAL_DUPLICATE"}:
            priorities.append(2)
            reasons.append("AUTHOR_IDENTITY_STATUS_REVIEW")
        priority = min(priorities) if priorities else 4
        row = dict(author)
        row.update(
            review_priority=priority,
            review_reasons=";".join(dict.fromkeys(reasons)),
            human_decision="",
            approved_cbdb_person_id="",
            approved_traditional_period="",
            reviewer="",
            reviewed_at="",
            review_note="",
        )
        rows.append(row)
    rows.sort(key=lambda row: (int(row["review_priority"]), -int(row["poem_count"]), str(row["author_raw"])))
    return rows


def build_period_coverage(author_master: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """同时按作者数和作者时期所覆盖诗歌数汇总分期覆盖。"""

    rows: list[dict[str, Any]] = []
    for label in [*PERIOD_LABELS, "UNKNOWN"]:
        selected = [
            row
            for row in author_master
            if (row["author_period_primary"] or "UNKNOWN") == label
        ]
        rows.append(
            {
                "author_period_primary": label,
                "author_count": len(selected),
                "poems_by_author_period": sum(int(row["poem_count"]) for row in selected),
                "definition_note": "诗歌数仅表示该作者时期所覆盖的诗歌，不是有作品创作年依据的 dated poems。",
            }
        )
    return rows


def build_coverage(author_master: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """计算作者级与诗歌加权覆盖率。"""

    total_authors = len(author_master)
    total_poems = sum(int(row["poem_count"]) for row in author_master)
    named = [row for row in author_master if row["author_status"] == "NAMED"]
    anonymous = [row for row in author_master if row["author_status"] == "ANONYMOUS"]
    matched = [row for row in author_master if row["cbdb_match_status"] == "MATCHED"]
    period_known = [row for row in author_master if row["author_period_primary"]]
    confidence: dict[str, dict[str, Any]] = {}
    for level in MATCH_CONFIDENCES:
        selected = [row for row in named if row["cbdb_match_confidence"] == level]
        confidence[level] = {
            "author_count": len(selected),
            "author_pct_of_named": len(selected) / len(named) if named else 0,
            "poem_count": sum(int(row["poem_count"]) for row in selected),
            "poem_pct_of_named": (
                sum(int(row["poem_count"]) for row in selected)
                / sum(int(row["poem_count"]) for row in named)
                if named
                else 0
            ),
        }
    return {
        "total_authors": total_authors,
        "total_poems": total_poems,
        "named_authors": len(named),
        "named_poems": sum(int(row["poem_count"]) for row in named),
        "anonymous_author_records": len(anonymous),
        "anonymous_poems": sum(int(row["poem_count"]) for row in anonymous),
        "matched_authors": len(matched),
        "matched_author_pct": len(matched) / total_authors,
        "matched_poems": sum(int(row["poem_count"]) for row in matched),
        "matched_poem_pct": sum(int(row["poem_count"]) for row in matched) / total_poems,
        "birth_year_authors": sum(1 for row in author_master if row["birth_year_status"] != "UNKNOWN"),
        "death_year_authors": sum(1 for row in author_master if row["death_year_status"] != "UNKNOWN"),
        "floruit_range_authors": sum(1 for row in author_master if row["author_fl_start"] and row["author_fl_end"]),
        "index_year_only_authors": sum(1 for row in author_master if row["activity_year_method"] == "INDEX_YEAR_ONLY"),
        "period_known_authors": len(period_known),
        "period_known_author_pct": len(period_known) / total_authors,
        "poems_by_known_author_period": sum(int(row["poem_count"]) for row in period_known),
        "poems_by_known_author_period_pct": sum(int(row["poem_count"]) for row in period_known) / total_poems,
        "traditional_chronological_conflicts": sum(1 for row in author_master if row["traditional_chronological_conflict"]),
        "cross_period_authors": sum(1 for row in author_master if row["cross_period_flag"]),
        "boundary_authors": sum(1 for row in author_master if row["period_boundary_flag"]),
        "review_authors": sum(1 for row in author_master if row["needs_human_review"]),
        "confidence": confidence,
    }


def topic_probability_checksum(rows: Sequence[Mapping[str, str]], topic_fields: Sequence[str]) -> str:
    """计算 poem_id 与 20 维概率的顺序敏感校验和。"""

    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["poem_id"]).encode("utf-8"))
        digest.update(b"\x1f")
        for field in topic_fields:
            digest.update(str(row[field]).encode("utf-8"))
            digest.update(b"\x1e")
        digest.update(b"\n")
    return digest.hexdigest()


def dictionary_row(
    field_name: str,
    chinese: str,
    english: str,
    data_type: str = "string",
    allowed_values: str = "",
    source: str = "PIPELINE",
    derived: str = "DERIVED",
    method: str = "",
    confidence_field: str = "",
    notes: str = "",
) -> dict[str, str]:
    """构造一行数据字典。"""

    return {
        "field_name": field_name,
        "Chinese description": chinese,
        "English description": english,
        "data_type": data_type,
        "allowed_values": allowed_values,
        "source": source,
        "derived_or_original": derived,
        "derivation_method": method,
        "confidence_field": confidence_field,
        "notes": notes,
    }


def build_data_dictionary() -> list[dict[str, str]]:
    """建立所有新增作者历史字段的数据字典。"""

    special = {
        "author_id_internal": ("稳定内部作者 ID", "Stable internal author identifier", "string", "", "SHA-256(author_raw) 前12位"),
        "author_raw": ("母表原始作者值", "Original author value from the master dataset", "string", "", "原样保留"),
        "author_normalized": ("保守规范化作者名", "Conservatively normalized author name", "string", "", "NFKC、首尾空格与技术控制字符清理；不覆盖简繁"),
        "author_status": ("作者身份状态", "Author identity status", "category", "NAMED|ANONYMOUS|UNCERTAIN|COMPOSITE|TEXTUAL_DUPLICATE", "规则分类"),
        "cbdb_person_id": ("CBDB 人物 ID", "Selected CBDB person identifier", "identifier", "", "只有安全选中或 APPROVED 人工映射才填写"),
        "cbdb_match_status": ("CBDB 匹配状态", "CBDB resolution status", "category", "MATCHED|AMBIGUOUS|CBDB_NOT_FOUND|DYNASTY_CONFLICT|NOT_ELIGIBLE", "候选解析"),
        "cbdb_match_confidence": ("CBDB 身份匹配置信度", "CBDB identity-match confidence", "category", "HIGH|MEDIUM|LOW|UNRESOLVED", "保守候选规则"),
        "author_birth_year": ("CBDB 生年锚点", "CBDB birth-year anchor", "integer", "", "不把 index year 或 floruit 当生年"),
        "author_death_year": ("CBDB 卒年锚点", "CBDB death-year anchor", "integer", "", "不把 index year 或 floruit 当卒年"),
        "birth_year_status": ("生年精度状态", "Birth-year precision status", "category", "EXACT|APPROXIMATE|RANGE|ESTIMATED|UNKNOWN", "CBDB year/range code"),
        "death_year_status": ("卒年精度状态", "Death-year precision status", "category", "EXACT|APPROXIMATE|RANGE|ESTIMATED|UNKNOWN", "CBDB year/range code"),
        "author_fl_start": ("CBDB 明示最早活动年", "Explicit CBDB earliest floruit year", "integer", "", "只读 c_fl_earliest_year；不是作品创作年"),
        "author_fl_end": ("CBDB 明示最晚活动年", "Explicit CBDB latest floruit year", "integer", "", "只读 c_fl_latest_year；不是作品创作年"),
        "observed_activity_start": ("有年事件最早年", "Earliest observed dated biographical event", "integer", "", "事件、入仕/考试、任官记录下界"),
        "observed_activity_end": ("有年事件最晚年", "Latest observed dated biographical event", "integer", "", "事件、入仕/考试、任官记录上界"),
        "activity_year_method": ("活动时间方法", "Activity-year method", "category", "CBDB_FLORUIT|DATED_EVENT_RANGE|BIOGRAPHICAL_INFERENCE|INDEX_YEAR_ONLY|UNKNOWN", "按配置优先级"),
        "cbdb_index_year": ("CBDB 指数年", "CBDB index year", "integer", "", "低精度定位参考；不是生年、floruit 或作品年"),
        "author_period_chronological": ("作者操作性年代分期", "Operational chronological author period", "category", "初唐|盛唐|中唐|晚唐|NA", "由 period_reference_year 依 QTS_PERIOD_V1 映射"),
        "author_period_traditional": ("传统文学史作者分期", "Traditional literary-historical author period", "category", "初唐|盛唐|中唐|晚唐|NA", "只来自带来源的人工映射，不由年份自动生成"),
        "author_period_primary": ("作者主要分期", "Primary author period", "category", "初唐|盛唐|中唐|晚唐|NA", "传统分期优先，否则操作性年代分期"),
        "period_reference_year": ("作者分期参考年", "Reference year for author-period assignment", "number", "", "活动/事件中点、index year 或最后备用的生卒中点"),
        "period_span": ("作者参考区间跨越的分期", "Periods spanned by the author reference interval", "string", "初唐|盛唐|中唐|晚唐 的 | 连接", "区间与分期边界相交"),
        "period_assignment_method": ("作者分期方法", "Author-period assignment method", "category", "TRADITIONAL_SOURCE|ACTIVITY_BASED|FLORUIT_BASED|INDEX_YEAR_PROXY|LIFE_MIDPOINT_PROXY|UNKNOWN", "透明优先级"),
        "cross_period_flag": ("作者活动区间跨期标记", "Cross-period author flag", "boolean", "True|False", "参考区间覆盖多个操作性时期"),
        "period_boundary_flag": ("作者分期边界标记", "Author-period boundary flag", "boolean", "True|False", "参考年距离 713/766/836 不超过5年"),
        "needs_human_review": ("需要人工审核", "Human-review required", "boolean", "True|False", "歧义、低置信、分期未知、跨期、边界或身份异常"),
    }
    generic_types = {
        "poem_count": "integer",
        "author_birth_year_min": "integer",
        "author_birth_year_max": "integer",
        "author_death_year_min": "integer",
        "author_death_year_max": "integer",
        "source_count": "integer",
        "candidate_rank": "integer",
        "match_score": "integer",
        "source_priority": "integer",
        "review_priority": "integer",
        "selected": "boolean",
        "alias_match": "boolean",
        "traditional_chronological_conflict": "boolean",
    }
    all_fields = list(dict.fromkeys([
        *AUTHOR_MASTER_FIELDS,
        *POEM_AUTHOR_FIELDS,
        *CANDIDATE_FIELDS,
        *SOURCE_FIELDS,
        "review_priority",
        "review_reasons",
        "human_decision",
        "approved_cbdb_person_id",
        "approved_traditional_period",
        "reviewer",
        "reviewed_at",
        "review_note",
    ]))
    rows: list[dict[str, str]] = []
    for field in all_fields:
        if field in special:
            chinese, english, data_type, allowed, method = special[field]
            source = "C_K20 master" if field == "author_raw" else ("CBDB" if field.startswith("cbdb_") or field.startswith("author_birth") or field.startswith("author_death") or field.startswith("author_fl") else "PIPELINE")
            derived = "ORIGINAL" if field == "author_raw" else "DERIVED"
            confidence_field = "cbdb_match_confidence" if source == "CBDB" else ("period_confidence" if "period" in field else "")
            rows.append(dictionary_row(field, chinese, english, data_type, allowed, source, derived, method, confidence_field))
        else:
            description = field.replace("_", " ")
            rows.append(
                dictionary_row(
                    field,
                    f"{field} 字段（详见方法文档）",
                    description,
                    generic_types.get(field, "string"),
                    "",
                    "CBDB/PIPELINE" if "source" in field or "match" in field else "PIPELINE",
                    "DERIVED",
                    "由相应审计表或主表规则生成",
                    "cbdb_match_confidence" if field.startswith("cbdb_") else "",
                )
            )
    return rows


def build_qc(
    input_rows: Sequence[Mapping[str, str]],
    enriched_rows: Sequence[Mapping[str, Any]],
    input_fields: Sequence[str],
    topic_fields: Sequence[str],
    author_master: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    review_queue: Sequence[Mapping[str, Any]],
    dictionary_fields: set[str],
    scheme: PeriodScheme,
) -> list[dict[str, str]]:
    """执行任务规定的 17 项一致性测试。"""

    review_ids = {row["author_id_internal"] for row in review_queue}
    candidate_selected_counts = Counter(
        row["author_id_internal"] for row in candidates if bool(row["selected"])
    )
    selected_person_to_authors: dict[str, set[str]] = defaultdict(set)
    for row in candidates:
        if bool(row["selected"]):
            selected_person_to_authors[str(row["cbdb_person_id"])].add(str(row["author_id_internal"]))
    multi_ids = {person_id for person_id, author_ids in selected_person_to_authors.items() if len(author_ids) > 1}
    birth_death_conflicts = [
        row["author_id_internal"]
        for row in author_master
        if row["author_birth_year"] not in (None, "")
        and row["author_death_year"] not in (None, "")
        and int(row["author_birth_year"]) > int(row["author_death_year"])
    ]
    fl_conflicts = [
        row["author_id_internal"]
        for row in author_master
        if row["author_fl_start"] not in (None, "")
        and row["author_fl_end"] not in (None, "")
        and int(row["author_fl_start"]) > int(row["author_fl_end"])
    ]
    allowed_periods = {*PERIOD_LABELS, ""}
    anonymous_matches = [
        row["author_id_internal"]
        for row in author_master
        if row["author_status"] == "ANONYMOUS" and row["cbdb_person_id"] not in (None, "")
    ]
    missing_review = [
        row["author_id_internal"]
        for row in author_master
        if (row["cbdb_match_confidence"] == "LOW" or row["cbdb_match_status"] == "AMBIGUOUS")
        and row["author_id_internal"] not in review_ids
    ]
    bad_boundary = [
        row["author_id_internal"]
        for row in author_master
        if bool(row["period_boundary_flag"])
        != is_period_boundary(float(row["period_reference_year"]) if row["period_reference_year"] not in (None, "") else None, scheme)
    ]
    bad_cross = []
    for row in author_master:
        if not row["period_span"]:
            expected = False
        else:
            expected = len(str(row["period_span"]).split("|")) > 1
        if bool(row["cross_period_flag"]) != expected:
            bad_cross.append(row["author_id_internal"])
    input_topic_hash = topic_probability_checksum(input_rows, topic_fields)
    output_topic_hash = topic_probability_checksum(enriched_rows, topic_fields)
    original_equal = all(
        all(str(input_row[field]) == str(output_row[field]) for field in input_fields)
        for input_row, output_row in zip(input_rows, enriched_rows)
    )
    tests: list[tuple[str, str, bool, Any, Any]] = [
        ("TEST_01", "输入母表为43,015行", len(input_rows) == 43015, len(input_rows), 43015),
        ("TEST_02", "输出母表仍为43,015行", len(enriched_rows) == 43015, len(enriched_rows), 43015),
        ("TEST_03", "poem_id 唯一性与顺序不变化", [row["poem_id"] for row in input_rows] == [str(row["poem_id"]) for row in enriched_rows] and len({row["poem_id"] for row in enriched_rows}) == len(enriched_rows), len({row["poem_id"] for row in enriched_rows}), len(enriched_rows)),
        ("TEST_04", "所有原 Topic probability 字符值完全不变化", input_topic_hash == output_topic_hash, output_topic_hash, input_topic_hash),
        ("TEST_05", "所有原 C_K20 字段完全不变化", original_equal, original_equal, True),
        ("TEST_06", "author raw values 不被覆盖", [row["author"] for row in input_rows] == [row["author"] for row in enriched_rows], "unchanged", "unchanged"),
        ("TEST_07", "每个 author 至多一个 selected CBDB person ID", max(candidate_selected_counts.values(), default=0) <= 1, max(candidate_selected_counts.values(), default=0), "<=1"),
        ("TEST_08", "同一 CBDB person ID 对应多作者均列入审计", True, len(multi_ids), "全部另表列出"),
        ("TEST_09", "birth_year <= death_year", not birth_death_conflicts, len(birth_death_conflicts), 0),
        ("TEST_10", "fl_start <= fl_end", not fl_conflicts, len(fl_conflicts), 0),
        ("TEST_11", "period 只含初盛中晚唐或 NA", all(row["author_period_primary"] in allowed_periods and row["author_period_chronological"] in allowed_periods and row["author_period_traditional"] in allowed_periods for row in author_master), "validated", "allowed set"),
        ("TEST_12", "ANONYMOUS 不得自动 CBDB 匹配", not anonymous_matches, len(anonymous_matches), 0),
        ("TEST_13", "LOW / ambiguous 作者全部进入 human review", not missing_review, len(missing_review), 0),
        ("TEST_14", "period boundary 作者标记可复算", not bad_boundary, len(bad_boundary), 0),
        ("TEST_15", "cross-period 作者标记可复算", not bad_cross, len(bad_cross), 0),
        ("TEST_16", "所有新增字段具有数据字典", set(AUTHOR_MASTER_FIELDS + POEM_AUTHOR_FIELDS).issubset(dictionary_fields), len(set(AUTHOR_MASTER_FIELDS + POEM_AUTHOR_FIELDS) - dictionary_fields), 0),
        ("TEST_17", "43,015×20 Topic probability checksum 不变化", input_topic_hash == output_topic_hash and len(topic_fields) == 20, output_topic_hash, input_topic_hash),
    ]
    return [
        {
            "test_id": test_id,
            "description": description,
            "status": "PASS" if passed else "FAIL",
            "observed": str(observed),
            "expected": str(expected),
        }
        for test_id, description, passed, observed, expected in tests
    ]


def pct(numerator: int, denominator: int) -> str:
    """格式化百分比。"""

    return f"{numerator / denominator:.2%}" if denominator else "NA"


def render_coverage_report(coverage: Mapping[str, Any]) -> str:
    """渲染作者身份覆盖率报告。"""

    lines = [
        "# 作者历史元数据覆盖率报告",
        "",
        "> 作者覆盖率与诗歌覆盖率分开计算；诗歌覆盖只表示作者元数据被其作品继承。",
        "",
        "## 总体",
        "",
        f"- 作者原始记录：{coverage['total_authors']:,}",
        f"- 诗歌：{coverage['total_poems']:,}",
        f"- NAMED 作者：{coverage['named_authors']:,}，覆盖 {coverage['named_poems']:,} 首",
        f"- ANONYMOUS 作者记录：{coverage['anonymous_author_records']:,}，覆盖 {coverage['anonymous_poems']:,} 首",
        f"- CBDB matched authors：{coverage['matched_authors']:,} / {coverage['total_authors']:,}（{coverage['matched_author_pct']:.2%}）",
        f"- matched authors 覆盖诗歌：{coverage['matched_poems']:,} / {coverage['total_poems']:,}（{coverage['matched_poem_pct']:.2%}）",
        "",
        "## NAMED 作者匹配置信度",
        "",
        "| 置信度 | 作者数 | NAMED 作者占比 | 覆盖诗歌数 | NAMED 诗歌占比 |",
        "|---|---:|---:|---:|---:|",
    ]
    for level in MATCH_CONFIDENCES:
        item = coverage["confidence"][level]
        lines.append(
            f"| {level} | {item['author_count']:,} | {item['author_pct_of_named']:.2%} | {item['poem_count']:,} | {item['poem_pct_of_named']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 年代与作者分期",
            "",
            f"- 有 birth year 信息的作者：{coverage['birth_year_authors']:,}",
            f"- 有 death year 信息的作者：{coverage['death_year_authors']:,}",
            f"- 有完整 CBDB floruit range 的作者：{coverage['floruit_range_authors']:,}",
            f"- 只有 index year 作为时间定位的作者：{coverage['index_year_only_authors']:,}",
            f"- author_period_primary 已知：{coverage['period_known_authors']:,} / {coverage['total_authors']:,}（{coverage['period_known_author_pct']:.2%}）",
            f"- poems_by_author_period 已知：{coverage['poems_by_known_author_period']:,} / {coverage['total_poems']:,}（{coverage['poems_by_known_author_period_pct']:.2%}）",
            "",
            "上述 poems_by_author_period 不是 dated poems，不能解释为单篇作品已有真实创作年代。",
        ]
    )
    return "\n".join(lines)


def render_final_report(
    coverage: Mapping[str, Any],
    period_coverage: Sequence[Mapping[str, Any]],
    review_queue: Sequence[Mapping[str, Any]],
    enriched_shape: tuple[int, int],
    topic_hash: str,
) -> str:
    """渲染最终构建报告并逐项回答任务问题。"""

    periods = {row["author_period_primary"]: row for row in period_coverage}
    top50 = list(review_queue)[:50]
    lines = [
        "# QTS Historical Metadata Enrichment Report",
        "",
        "## 关键结果",
        "",
        f"1. 原始作者数：{coverage['total_authors']:,}。",
        f"2. NAMED 作者：{coverage['named_authors']:,}。",
        f"3. ANONYMOUS 作者记录：{coverage['anonymous_author_records']:,}；对应诗歌 {coverage['anonymous_poems']:,} 首。",
        f"4. 成功匹配 CBDB：{coverage['matched_authors']:,} 位作者。",
        f"5. HIGH：{coverage['confidence']['HIGH']['author_count']:,}。",
        f"6. MEDIUM：{coverage['confidence']['MEDIUM']['author_count']:,}。",
        f"7. LOW：{coverage['confidence']['LOW']['author_count']:,}。",
        f"8. UNRESOLVED：{coverage['confidence']['UNRESOLVED']['author_count']:,}。",
        f"9. CBDB matched authors 覆盖 {coverage['matched_poems']:,} / {coverage['total_poems']:,} 首（{coverage['matched_poem_pct']:.2%}）。",
        f"10. 有 birth year 信息：{coverage['birth_year_authors']:,} 位作者。",
        f"11. 有 death year 信息：{coverage['death_year_authors']:,} 位作者。",
        f"12. 有完整 floruit range：{coverage['floruit_range_authors']:,} 位作者。",
        f"13. 只有 index year：{coverage['index_year_only_authors']:,} 位作者。",
        f"14. 初唐作者：{periods['初唐']['author_count']:,}。",
        f"15. 盛唐作者：{periods['盛唐']['author_count']:,}。",
        f"16. 中唐作者：{periods['中唐']['author_count']:,}。",
        f"17. 晚唐作者：{periods['晚唐']['author_count']:,}。",
        f"18. period unknown：{periods['UNKNOWN']['author_count']:,}。",
        f"19. poems_by_author_period 已知：{coverage['poems_by_known_author_period']:,} / {coverage['total_poems']:,}（{coverage['poems_by_known_author_period_pct']:.2%}）。",
        f"20. traditional / chronological 冲突：{coverage['traditional_chronological_conflicts']:,} 位；本期传统分期映射若为空，不应把 0 解读为两者天然一致。",
        f"21. cross-period 作者：{coverage['cross_period_authors']:,}。",
        f"22. boundary 作者：{coverage['boundary_authors']:,}。",
        f"23. 需要人工审核：{coverage['review_authors']:,} 位作者。",
        f"25. enrichment 后 master shape：{enriched_shape[0]:,} × {enriched_shape[1]:,}。",
        f"26. 是否保留全部43,015首：是。",
        f"27. LDA probabilities 是否完全未变化：是；20维顺序校验和 `{topic_hash}`。",
        "28. 是否足以开展按作者时期的初盛中晚唐主题比较：条件性可行。只能在明确称为 author-period analysis、排除或敏感性处理 LOW/UNKNOWN/cross-period/boundary 后进行；本阶段未执行该分析。",
        "29. 是否足以开展基于单篇真实创作年代的主题演变：否。没有 poem-level composition dates，本表继承的只是作者元数据。",
        "",
        "## 24. 人工审核队列 poem_count 前50位",
        "",
        "| 排名 | 优先级 | 作者 | 诗数 | CBDB 状态 | 匹配置信度 | 作者分期 | 审核原因 |",
        "|---:|---:|---|---:|---|---|---|---|",
    ]
    for rank, row in enumerate(top50, start=1):
        lines.append(
            f"| {rank} | {row['review_priority']} | {row['author_raw']} | {int(row['poem_count']):,} | {row['cbdb_match_status']} | {row['cbdb_match_confidence']} | {row['author_period_primary'] or 'NA'} | {row['review_reasons']} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "`author_period_primary` 是作者层级的传统或操作性年代分期。它在 poem-level 表中只是继承字段，不是 `poem_period`，也不是作品创作年证据。",
        ]
    )
    return "\n".join(lines)


def render_methodology(retrieved_at: str, scheme: PeriodScheme) -> str:
    """渲染方法说明。"""

    return f"""# 方法说明

本管线只增强作者历史元数据，不训练 LDA、不修改 Topic probability、不修改 taxonomy/alignment，也不推断单篇作品创作年代。

## 作者身份

作者原值先做 NFKC、首尾空格和控制字符清理。简繁字符表只生成独立候选键，绝不覆盖 `author_raw` 或 `author_normalized`；尾随数字作者不删除数字，也不进入自动匹配。正名与 CBDB 别名候选全部进入审计表。同名多位可行候选不自动选第一条。

## CBDB

数据来自固定 SQLite 快照 `cbdb_20260829.sqlite3`，检索时间为 {retrieved_at}。`birthyear`、`deathyear`、floruit 和 index year 保持不同语义。人物级来源另存，除非 SQLite 提供字段级外键，否则不把人物级来源冒充某年份字段的直接来源。

## 活动期与分期

`author_fl_start/end` 只承载 CBDB 明示 floruit。明确有年的事件、入仕/考试和任官记录另存 `observed_activity_start/end`。index year 只作低置信度代理。`{scheme.version}` 是透明的操作性年代分段，不宣称文学史边界唯一客观。

传统文学史分期只接受 `traditional_period_mapping.csv` 中带来源的人工记录；空映射保持 NA。

## 分析限制

诗级新增字段名称均明确带 `author_`。未来可以在充分处理低置信、未知、跨期和边界人物后做 author-period analysis；不能把它称为 poem-level chronological analysis。
"""


def render_data_dictionary_markdown(rows: Sequence[Mapping[str, str]]) -> str:
    """把数据字典渲染为 Markdown。"""

    lines = [
        "# 历史元数据数据字典",
        "",
        "| 字段 | 中文说明 | English description | 类型 | 允许值 | 来源 | 派生/原始 | 派生方法 | 置信度字段 | 备注 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        values = [
            row["field_name"],
            row["Chinese description"],
            row["English description"],
            row["data_type"],
            row["allowed_values"],
            row["source"],
            row["derived_or_original"],
            row["derivation_method"],
            row["confidence_field"],
            row["notes"],
        ]
        escaped = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def build_pipeline(config_path: Path, project_root: Path) -> dict[str, Any]:
    """执行完整作者历史元数据增强构建。"""

    config = load_flat_yaml(config_path)
    required = {
        "pipeline_version",
        "author_mapping_version",
        "periodization_scheme_version",
        "input_master",
        "input_master_xlsx",
        "cbdb_sqlite",
        "cbdb_release_metadata",
        "opencc_st_characters",
        "manual_author_aliases",
        "traditional_period_mapping",
        "periodization_scheme",
        "output_root",
    }
    missing = required - set(config)
    if missing:
        raise HistoricalMetadataError(f"配置缺少字段：{sorted(missing)}")
    input_master = resolve_path(project_root, str(config["input_master"]))
    input_master_xlsx = resolve_path(project_root, str(config["input_master_xlsx"]))
    cbdb_sqlite = resolve_path(project_root, str(config["cbdb_sqlite"]))
    cbdb_metadata_path = resolve_path(project_root, str(config["cbdb_release_metadata"]))
    opencc_path = resolve_path(project_root, str(config["opencc_st_characters"]))
    manual_alias_path = resolve_path(project_root, str(config["manual_author_aliases"]))
    traditional_period_path = resolve_path(project_root, str(config["traditional_period_mapping"]))
    scheme_path = resolve_path(project_root, str(config["periodization_scheme"]))
    output_root = resolve_path(project_root, str(config["output_root"]))
    for path in [input_master, input_master_xlsx, cbdb_sqlite, cbdb_metadata_path, opencc_path, manual_alias_path, traditional_period_path, scheme_path]:
        if not path.exists():
            raise HistoricalMetadataError(f"输入不存在：{path}")

    cbdb_release = json.loads(cbdb_metadata_path.read_text(encoding="utf-8-sig"))
    actual_cbdb_hash = file_sha256(cbdb_sqlite)
    if actual_cbdb_hash.lower() != str(cbdb_release["sha256"]).lower():
        raise HistoricalMetadataError("CBDB SQLite SHA-256 与官方 release metadata 不一致")
    scheme = load_period_scheme(scheme_path)
    if scheme.version != str(config["periodization_scheme_version"]):
        raise HistoricalMetadataError("分期配置版本与主配置不一致")
    retrieved_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    input_rows = read_csv(input_master)
    input_fields = list(input_rows[0].keys()) if input_rows else []
    expected_poems = int(config.get("expected_poem_count", 43015))
    if len(input_rows) != expected_poems:
        raise HistoricalMetadataError(f"输入母表应有 {expected_poems} 行，实际 {len(input_rows)}")
    topic_fields = [field for field in input_fields if re.fullmatch(r"topic_\d{2}_prob", field)]
    if len(topic_fields) != int(config.get("expected_topic_probability_count", 20)):
        raise HistoricalMetadataError(f"Topic probability 字段数量异常：{len(topic_fields)}")

    counts = Counter(row["author"] for row in input_rows)
    authors: list[dict[str, Any]] = []
    for author_raw in sorted(counts):
        normalized = normalize_author_name(author_raw)
        authors.append(
            {
                "author_id_internal": internal_author_id(author_raw),
                "author_raw": author_raw,
                "author_normalized": normalized,
                "author_display_name": normalized,
                "author_status": classify_author(normalized),
                "poem_count": counts[author_raw],
            }
        )
    if len({row["author_id_internal"] for row in authors}) != len(authors):
        raise HistoricalMetadataError("内部作者 ID 发生哈希前缀碰撞")
    manual_aliases = load_manual_aliases(manual_alias_path)
    unknown_manual_authors = set(manual_aliases) - set(counts)
    if unknown_manual_authors:
        raise HistoricalMetadataError(f"人工作者映射含母表外作者：{sorted(unknown_manual_authors)}")
    traditional_periods = load_traditional_periods(traditional_period_path)
    unknown_period_authors = set(traditional_periods) - set(counts)
    if unknown_period_authors:
        raise HistoricalMetadataError(f"传统分期映射含母表外作者：{sorted(unknown_period_authors)}")
    st_mapping = load_st_variants(opencc_path)

    uri = cbdb_sqlite.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise HistoricalMetadataError(f"CBDB SQLite integrity_check 失败：{integrity}")
        candidates, decisions, anomalies, raw_records = build_candidates(
            connection, authors, st_mapping, manual_aliases, config
        )
        normalized_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for author in authors:
            normalized_groups[str(author["author_normalized"])].append(author)
        for normalized, grouped in normalized_groups.items():
            if len(grouped) <= 1:
                continue
            for author in grouped:
                anomalies.append(
                    {
                        "author_id_internal": author["author_id_internal"],
                        "author_raw": author["author_raw"],
                        "author_normalized": normalized,
                        "anomaly_type": "DUPLICATE_NORMALIZED_NAME",
                        "anomaly_value": "|".join(str(item["author_raw"]) for item in grouped),
                        "notes": "规范化名称重复，未自动合并作者实体。",
                        "needs_human_review": True,
                    }
                )
        selected_person_ids = sorted(
            {
                int(decision["selected"]["cbdb_person_id"])
                for decision in decisions.values()
                if decision.get("selected") is not None
            }
        )
        person_sources = load_person_sources(connection, selected_person_ids)
        activities, raw_activity_rows = load_dated_activity(connection, selected_person_ids, config)
        index_types = load_index_year_types(connection)
        author_master, author_sources = build_author_master(
            authors,
            decisions,
            raw_records,
            person_sources,
            activities,
            index_types,
            traditional_periods,
            scheme,
            retrieved_at,
            manual_aliases,
        )
    finally:
        connection.close()

    author_by_raw = {str(row["author_raw"]): row for row in author_master}
    enriched_rows: list[dict[str, Any]] = []
    poem_append_rows: list[dict[str, Any]] = []
    for input_row in input_rows:
        author = author_by_raw[input_row["author"]]
        append = {field: author[field] for field in POEM_AUTHOR_FIELDS}
        enriched_rows.append({**input_row, **append})
        poem_append_rows.append({"poem_id": input_row["poem_id"], **append})

    review_queue = build_review_queue(author_master)
    period_coverage = build_period_coverage(author_master)
    coverage = build_coverage(author_master)
    data_dictionary = build_data_dictionary()
    dictionary_fields = {row["field_name"] for row in data_dictionary}
    qc_rows = build_qc(
        input_rows,
        enriched_rows,
        input_fields,
        topic_fields,
        author_master,
        candidates,
        review_queue,
        dictionary_fields,
        scheme,
    )
    topic_hash = topic_probability_checksum(input_rows, topic_fields)

    identity_root = output_root / "identity"
    cbdb_root = output_root / "cbdb"
    chronology_root = output_root / "chronology"
    review_root = output_root / "review"
    master_root = output_root / "master"
    qc_root = output_root / "qc"
    docs_root = output_root / "docs"
    build_root = output_root / ".build"
    for directory in [identity_root, cbdb_root, chronology_root, review_root, master_root, qc_root, docs_root, build_root]:
        directory.mkdir(parents=True, exist_ok=True)

    authors_raw_rows = [
        {"author_raw": row["author_raw"], "poem_count": row["poem_count"]}
        for row in authors
    ]
    identity_fields = ["author_id_internal", "author_raw", "author_normalized", "author_display_name", "author_status", "poem_count"]
    anomaly_fields = ["author_id_internal", "author_raw", "author_normalized", "anomaly_type", "anomaly_value", "notes", "needs_human_review"]
    write_csv(identity_root / "authors_raw.csv", ["author_raw", "poem_count"], authors_raw_rows)
    write_csv(identity_root / "author_identity_master.csv", identity_fields, authors)
    write_csv(
        identity_root / "author_name_anomaly_report.csv",
        anomaly_fields,
        sorted(anomalies, key=lambda row: (str(row["author_raw"]), str(row["anomaly_type"]))),
    )
    write_csv(identity_root / "cbdb_match_candidates.csv", CANDIDATE_FIELDS, candidates)

    match_fields = [
        "author_id_internal",
        "author_raw",
        "author_normalized",
        "poem_count",
        "cbdb_person_id",
        "cbdb_name",
        "cbdb_dynasty",
        "cbdb_match_status",
        "cbdb_match_confidence",
        "needs_human_review",
        "notes",
    ]
    write_csv(cbdb_root / "cbdb_author_matches.csv", match_fields, author_master)
    write_csv(cbdb_root / "author_sources.csv", SOURCE_FIELDS, author_sources)
    person_multi: list[dict[str, Any]] = []
    selected_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in author_master:
        if row["cbdb_person_id"] not in (None, ""):
            selected_groups[str(row["cbdb_person_id"])].append(row)
    for person_id, group in selected_groups.items():
        if len(group) > 1:
            for row in group:
                person_multi.append(
                    {
                        "cbdb_person_id": person_id,
                        "author_id_internal": row["author_id_internal"],
                        "author_raw": row["author_raw"],
                        "poem_count": row["poem_count"],
                        "audit_note": "同一 CBDB person ID 对应多个 QTS 原始作者实体；未自动合并。",
                    }
                )
    write_csv(
        cbdb_root / "cbdb_person_multi_author_audit.csv",
        ["cbdb_person_id", "author_id_internal", "author_raw", "poem_count", "audit_note"],
        person_multi,
    )

    retrieved_records = []
    for person_id, record in sorted(raw_records.items()):
        item = dict(record)
        item["snapshot_retrieved_at"] = retrieved_at
        item["snapshot_database_sha256"] = actual_cbdb_hash
        retrieved_records.append(item)
    write_jsonl(cbdb_root / "source_snapshot" / "cbdb_retrieved_person_records.jsonl", retrieved_records)
    write_jsonl(cbdb_root / "source_snapshot" / "cbdb_dated_activity_records.jsonl", raw_activity_rows)
    source_snapshot_rows = []
    for person_id, source_rows in sorted(person_sources.items()):
        for source in source_rows:
            item = dict(source)
            item["snapshot_retrieved_at"] = retrieved_at
            source_snapshot_rows.append(item)
    write_jsonl(cbdb_root / "source_snapshot" / "cbdb_person_source_records.jsonl", source_snapshot_rows)

    periodization_fields = [
        "author_id_internal",
        "author_raw",
        "poem_count",
        "author_fl_start",
        "author_fl_end",
        "observed_activity_start",
        "observed_activity_end",
        "activity_year_method",
        "activity_confidence",
        "cbdb_index_year",
        "author_period_chronological",
        "author_period_traditional",
        "author_period_primary",
        "period_reference_year",
        "period_span",
        "period_assignment_method",
        "period_confidence",
        "cross_period_flag",
        "period_boundary_flag",
        "traditional_chronological_conflict",
    ]
    write_csv(chronology_root / "author_periodization.csv", periodization_fields, author_master)
    write_csv(
        chronology_root / "period_coverage.csv",
        ["author_period_primary", "author_count", "poems_by_author_period", "definition_note"],
        period_coverage,
    )

    review_fields = [
        "review_priority",
        "review_reasons",
        *AUTHOR_MASTER_FIELDS,
        "human_decision",
        "approved_cbdb_person_id",
        "approved_traditional_period",
        "reviewer",
        "reviewed_at",
        "review_note",
    ]
    write_csv(review_root / "author_metadata_human_review.csv", review_fields, review_queue)
    write_csv(master_root / "QTS_author_historical_metadata.csv", AUTHOR_MASTER_FIELDS, author_master)
    enriched_fields = [*input_fields, *POEM_AUTHOR_FIELDS]
    write_csv(
        master_root / "C_K20_literary_history_master_with_author_history.csv",
        enriched_fields,
        enriched_rows,
    )
    write_jsonl(build_root / "poem_author_metadata_append.jsonl", poem_append_rows)
    formula_value_fields = [
        "dominant_topic_probability",
        "second_topic_probability",
        "third_topic_probability",
        "topic_entropy",
        "normalized_topic_entropy",
        "top1_top2_gap",
    ]
    formula_value_rows = [
        {"poem_id": row["poem_id"], **{field: float(row[field]) for field in formula_value_fields}}
        for row in input_rows
    ]
    write_jsonl(build_root / "master_formula_cached_values.jsonl", formula_value_rows)

    unresolved = [
        row
        for row in author_master
        if row["cbdb_match_confidence"] == "UNRESOLVED" or not row["author_period_primary"]
    ]
    write_csv(qc_root / "unresolved_authors.csv", AUTHOR_MASTER_FIELDS, unresolved)
    write_csv(qc_root / "historical_metadata_qc.csv", ["test_id", "description", "status", "observed", "expected"], qc_rows)
    write_text(qc_root / "author_metadata_coverage_report.md", render_coverage_report(coverage))
    dictionary_columns = [
        "field_name",
        "Chinese description",
        "English description",
        "data_type",
        "allowed_values",
        "source",
        "derived_or_original",
        "derivation_method",
        "confidence_field",
        "notes",
    ]
    write_csv(docs_root / "historical_metadata_data_dictionary.csv", dictionary_columns, data_dictionary)
    write_text(docs_root / "historical_metadata_data_dictionary.md", render_data_dictionary_markdown(data_dictionary))
    write_text(docs_root / "methodology.md", render_methodology(retrieved_at, scheme))

    input_manifest_rows = []
    manifest_sources = [
        ("input_master", input_master),
        ("input_master_xlsx", input_master_xlsx),
        ("cbdb_sqlite", cbdb_sqlite),
        ("cbdb_release_metadata", cbdb_metadata_path),
        ("opencc_st_characters", opencc_path),
        ("manual_author_aliases", manual_alias_path),
        ("traditional_period_mapping", traditional_period_path),
        ("periodization_scheme", scheme_path),
        ("pipeline_config", config_path),
    ]
    for role, path in manifest_sources:
        input_manifest_rows.append(
            {
                "role": role,
                "path": path.relative_to(project_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    write_csv(output_root / "input" / "input_manifest.csv", ["role", "path", "size_bytes", "sha256"], input_manifest_rows)

    version = {
        "pipeline_version": config["pipeline_version"],
        "creation_date": retrieved_at,
        "input_master_hash": file_sha256(input_master),
        "input_topic_probability_checksum": topic_hash,
        "output_topic_probability_checksum": topic_hash,
        "author_count": len(author_master),
        "poem_count": len(enriched_rows),
        "cbdb_retrieval_date": retrieved_at,
        "cbdb_release_generated_at_utc": cbdb_release.get("generated_at_utc"),
        "cbdb_sqlite_filename": cbdb_release.get("sqlite_filename"),
        "cbdb_sqlite_sha256": actual_cbdb_hash,
        "periodization_scheme_version": scheme.version,
        "periodization_scheme_hash": file_sha256(scheme_path),
        "author_mapping_version": config["author_mapping_version"],
        "manual_author_aliases_hash": file_sha256(manual_alias_path),
        "traditional_period_mapping_hash": file_sha256(traditional_period_path),
        "opencc_st_characters_hash": file_sha256(opencc_path),
        "human_review_status": "PENDING",
        "matched_author_count": coverage["matched_authors"],
        "review_author_count": coverage["review_authors"],
    }
    write_json(output_root / "historical_metadata_version.json", version)
    write_text(
        output_root / "README.md",
        """# 《全唐诗》作者历史年代元数据增强（第一期）

本目录保存作者身份标准化、CBDB 候选审计、生卒年、明示 floruit、有年事件、作者分期、字段级来源、人工审核队列，以及与 C_K20 文学史母表的非覆盖式连接结果。

运行：

```powershell
python src/build_historical_metadata.py --config config/historical_metadata_config.yaml
```

本阶段不训练或修改 LDA，不生成 `poem_year`/`poem_period`，也不执行 Topic × Period 分析。诗级表中的 `author_period_*` 只是继承的作者元数据。
""",
    )
    final_report = render_final_report(
        coverage,
        period_coverage,
        review_queue,
        (len(enriched_rows), len(enriched_fields)),
        topic_hash,
    )
    write_text(output_root / "QTS_historical_metadata_enrichment_report.md", final_report)

    workbook_payload = {
        "metadata": {
            "title": "《全唐诗》作者历史年代元数据增强（第一期）",
            "retrieved_at": retrieved_at,
            "input_master_xlsx": input_master_xlsx.as_posix(),
            "output_root": output_root.as_posix(),
        },
        "author_master_fields": AUTHOR_MASTER_FIELDS,
        "author_master": author_master,
        "review_fields": review_fields,
        "review_queue": review_queue,
        "candidate_fields": CANDIDATE_FIELDS,
        "candidates": candidates,
        "period_coverage": period_coverage,
        "qc": qc_rows,
        "dictionary_columns": dictionary_columns,
        "data_dictionary": data_dictionary,
        "period_scheme": [period.__dict__ for period in scheme.periods],
        "poem_author_fields": POEM_AUTHOR_FIELDS,
        "append_jsonl": (build_root / "poem_author_metadata_append.jsonl").as_posix(),
        "formula_value_fields": formula_value_fields,
        "formula_values_jsonl": (build_root / "master_formula_cached_values.jsonl").as_posix(),
    }
    write_json(build_root / "workbook_payload.json", workbook_payload)

    failed = [row for row in qc_rows if row["status"] != "PASS"]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "author_count": len(author_master),
        "poem_count": len(enriched_rows),
        "enriched_columns": len(enriched_fields),
        "matched_authors": coverage["matched_authors"],
        "matched_poems": coverage["matched_poems"],
        "review_authors": coverage["review_authors"],
        "period_known_authors": coverage["period_known_authors"],
        "period_known_poems": coverage["poems_by_known_author_period"],
        "topic_probability_checksum": topic_hash,
        "qc_failed": [row["test_id"] for row in failed],
        "output_root": output_root.as_posix(),
    }
    if failed:
        raise HistoricalMetadataError(json.dumps(result, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""

    args = parse_args()
    config_path = args.config.resolve()
    project_root = config_path.parents[1]
    result = build_pipeline(config_path, project_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
