"""严格依照现有资源清洗《全唐诗》分词语料，并生成完整审计文件。"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import importlib.metadata
import json
import logging
import platform
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "cleaning_config.yaml"
TRUE_VALUES = {"1", "true", "yes", "y", "是"}


class CleaningError(RuntimeError):
    """表示输入、规则或配置不能安全执行。"""


@dataclass(frozen=True)
class NormalizationRule:
    """保存一条明确的词形规范规则。"""

    source_form: str
    normalized_form: str
    relation: str
    scope: str
    status: str
    apply_automatically: bool
    row_number: int

    @property
    def rule_id(self) -> str:
        """返回可复核的规则标识。"""

        return (
            f"row={self.row_number}|{self.source_form}->{self.normalized_form}"
            f"|{self.relation}|{self.scope}"
        )


@dataclass
class PoemRecord:
    """保存一首诗在清洗各阶段的状态。"""

    poem_id: Any
    author: str
    title: str
    original_text: str
    original_tokens: list[str]
    normalized_tokens: list[str] = field(default_factory=list)
    normalization_events: list[dict[str, Any]] = field(default_factory=list)
    cleaned_tokens: list[str] = field(default_factory=list)
    removed_tokens: list[str] = field(default_factory=list)
    removal_counts: Counter[str] = field(default_factory=Counter)
    protected_token_count: int = 0
    protected_rescued_count: int = 0


def file_sha256(path: Path) -> str:
    """计算文件的 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """优先把仓库内路径显示为稳定的相对路径。"""

    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_config(path: Path) -> dict[str, Any]:
    """读取 JSON 兼容 YAML；必要时才使用可选的 PyYAML。"""

    text = path.read_text(encoding="utf-8-sig")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml
        except ImportError as exc:
            raise CleaningError(
                f"配置文件不是 JSON 兼容 YAML，且未安装 PyYAML：{path}；"
                f"JSON 解析错误：{json_error}"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise CleaningError(f"配置文件顶层必须是对象：{path}")
    return value


def resolve_path(value: str, project_root: Path = PROJECT_ROOT) -> Path:
    """把配置中的仓库相对路径解析为绝对路径。"""

    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """验证配置节点为对象。"""

    if not isinstance(value, Mapping):
        raise CleaningError(f"配置项 {label} 必须是对象")
    return value


def prepare_paths(
    config: Mapping[str, Any], project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """解析并验证全部输入输出路径。"""

    paths = require_mapping(config.get("paths"), "paths")
    required_inputs = (
        "corpus",
        "tokenized_corpus",
        "normalization",
        "protected_words",
        "filtering_rules",
        "semantic_mapping",
    )
    resolved: dict[str, Any] = {}
    for name in required_inputs:
        if name not in paths:
            raise CleaningError(f"配置缺少 paths.{name}")
        resolved[name] = resolve_path(str(paths[name]), project_root)
    stopword_paths = require_mapping(paths.get("stopwords"), "paths.stopwords")
    resolved["stopwords"] = {
        str(tier): resolve_path(str(value), project_root)
        for tier, value in stopword_paths.items()
    }
    for name in ("output_dir", "reports_dir", "logs_dir"):
        if name not in paths:
            raise CleaningError(f"配置缺少 paths.{name}")
        resolved[name] = resolve_path(str(paths[name]), project_root)

    for name in required_inputs:
        path = resolved[name]
        if not path.is_file():
            raise CleaningError(f"输入文件不存在：{path}")
    for tier, path in resolved["stopwords"].items():
        if not path.is_file():
            raise CleaningError(f"停用词文件不存在（{tier}）：{path}")

    input_paths = {
        resolved[name].resolve() for name in required_inputs
    } | {path.resolve() for path in resolved["stopwords"].values()}
    for directory_name in ("output_dir", "reports_dir", "logs_dir"):
        directory = resolved[directory_name]
        if directory.resolve() in input_paths:
            raise CleaningError(f"输出目录不得指向输入文件：{directory}")
        directory.mkdir(parents=True, exist_ok=True)
    return resolved


def setup_logger(log_path: Path) -> logging.Logger:
    """创建同时写入终端和文件的 UTF-8 日志。"""

    logger = logging.getLogger("clean_corpus")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，并在损坏行处给出明确错误。"""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise CleaningError(f"JSONL 存在空白行：{path} 第 {line_number} 行")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CleaningError(
                    f"JSONL 解析失败：{path} 第 {line_number} 行：{exc}"
                ) from exc
            if not isinstance(value, dict):
                raise CleaningError(f"JSONL 行必须是对象：{path} 第 {line_number} 行")
            rows.append(value)
    return rows


def read_terms(path: Path) -> list[str]:
    """读取一行一词的词表并拒绝重复项。"""

    values: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            term = line.strip()
            if not term or term.startswith("#"):
                continue
            if term in seen:
                raise CleaningError(
                    f"词表存在重复项：{path} 第 {line_number} 行“{term}”"
                )
            seen.add(term)
            values.append(term)
    return values


def read_tsv(path: Path) -> list[dict[str, str]]:
    """读取带表头的 TSV。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise CleaningError(f"TSV 缺少表头：{path}")
        return [dict(row) for row in reader]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """以 Excel 兼容的 UTF-8 编码写出 CSV。"""

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    """写出可读的 UTF-8 JSON。"""

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def is_han(char: str) -> bool:
    """判断字符是否属于常见或扩展汉字区。"""

    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x3134F
    )


def is_han_token(token: str) -> bool:
    """判断词元是否完全由汉字组成。"""

    return bool(token) and all(is_han(char) for char in token)


def parse_bool(value: Any) -> bool:
    """解析规则表中的布尔值。"""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUE_VALUES


def load_normalization_rules(path: Path) -> list[NormalizationRule]:
    """读取词形规范表，不推断任何额外映射。"""

    rows = read_tsv(path)
    required = {
        "source_form",
        "normalized_form",
        "relation",
        "scope",
        "status",
        "apply_automatically",
    }
    rules: list[NormalizationRule] = []
    for row_number, row in enumerate(rows, 2):
        missing = required - row.keys()
        if missing:
            raise CleaningError(f"规范表缺少字段 {sorted(missing)}：{path}")
        source = row["source_form"].strip()
        target = row["normalized_form"].strip()
        if not source or not target:
            raise CleaningError(f"规范表存在空词形：{path} 第 {row_number} 行")
        rules.append(
            NormalizationRule(
                source_form=source,
                normalized_form=target,
                relation=row["relation"].strip(),
                scope=row["scope"].strip(),
                status=row["status"].strip(),
                apply_automatically=parse_bool(row["apply_automatically"]),
                row_number=row_number,
            )
        )
    return rules


def scope_allows(scope: str, poem_id: Any) -> bool:
    """仅识别规则中明确写出的全局或 poem_id 范围。"""

    ids = {int(value) for value in re.findall(r"poem_id\s*=\s*(\d+)", scope)}
    if ids:
        try:
            return int(poem_id) in ids
        except (TypeError, ValueError):
            return False
    return "全局" in scope


def apply_normalization(
    poem_id: Any,
    tokens: Sequence[str],
    rules: Sequence[NormalizationRule],
    automatic_only: bool = True,
) -> tuple[list[str], list[dict[str, Any]]]:
    """只按明确、适用且获准自动执行的规则替换完整 token。"""

    by_source: dict[str, list[NormalizationRule]] = defaultdict(list)
    for rule in rules:
        if automatic_only and not rule.apply_automatically:
            continue
        by_source[rule.source_form].append(rule)

    normalized: list[str] = []
    events: list[dict[str, Any]] = []
    for token_index, token in enumerate(tokens):
        candidates = [
            rule for rule in by_source.get(token, ()) if scope_allows(rule.scope, poem_id)
        ]
        targets = {rule.normalized_form for rule in candidates}
        if len(targets) > 1:
            raise CleaningError(
                f"poem_id={poem_id} 的 token“{token}”同时命中不同规范结果："
                f"{sorted(targets)}"
            )
        if not candidates:
            normalized.append(token)
            continue
        rule = candidates[0]
        normalized.append(rule.normalized_form)
        if token != rule.normalized_form:
            events.append(
                {
                    "poem_id": poem_id,
                    "token_index": token_index,
                    "original_token": token,
                    "normalized_token": rule.normalized_form,
                    "rule_type": rule.relation,
                    "scope": rule.scope,
                    "status": rule.status,
                    "rule_id": rule.rule_id,
                }
            )
    return normalized, events


def make_issue(
    poem_id: Any,
    author: str,
    title: str,
    issue_type: str,
    original_text: str,
    tokens: Sequence[str],
    detail: str = "",
) -> dict[str, Any]:
    """构造统一的数据质量问题记录。"""

    return {
        "poem_id": poem_id,
        "author": author,
        "title": title,
        "issue_type": issue_type,
        "original_text": original_text,
        "tokenized_text": " ".join(tokens),
        "issue_detail": detail,
    }


def align_records(
    corpus_rows: Sequence[Mapping[str, Any]],
    token_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[PoemRecord], list[dict[str, Any]]]:
    """按 ID 和出现顺序对齐元数据与分词结果，同时保留异常记录。"""

    issues: list[dict[str, Any]] = []
    token_groups: dict[Any, deque[tuple[int, Mapping[str, Any]]]] = defaultdict(deque)
    token_id_counts: Counter[Any] = Counter()
    for line_number, row in enumerate(token_rows, 1):
        poem_id = row.get("id")
        token_id_counts[poem_id] += 1
        token_groups[poem_id].append((line_number, row))

    corpus_id_counts = Counter(row.get("id") for row in corpus_rows)
    records: list[PoemRecord] = []
    exact_seen: dict[tuple[str, str, str, tuple[str, ...]], Any] = {}
    for row_index, row in enumerate(corpus_rows, 1):
        poem_id = row.get("id")
        author_value = row.get("poet", row.get("author", ""))
        author = "" if author_value is None else str(author_value).strip()
        title_value = row.get("title", "")
        title = "" if title_value is None else str(title_value).strip()
        text_value = row.get("text", row.get("original_text", ""))
        original_text = "" if text_value is None else str(text_value)

        if token_groups.get(poem_id):
            token_line, token_row = token_groups[poem_id].popleft()
            raw_tokens = token_row.get("tokens")
            if not isinstance(raw_tokens, list) or not all(
                isinstance(token, str) for token in raw_tokens
            ):
                raise CleaningError(
                    f"tokens 必须是字符串数组：分词文件第 {token_line} 行，"
                    f"poem_id={poem_id}"
                )
            tokens = list(raw_tokens)
        else:
            tokens = []
            issues.append(
                make_issue(
                    poem_id,
                    author,
                    title,
                    "MISSING_TOKENIZED_RECORD",
                    original_text,
                    tokens,
                    f"语料第 {row_index} 行没有同 ID 的分词记录",
                )
            )

        record = PoemRecord(poem_id, author, title, original_text, tokens)
        records.append(record)
        if poem_id is None or str(poem_id).strip() == "":
            issues.append(
                make_issue(
                    poem_id, author, title, "MISSING_POEM_ID", original_text, tokens
                )
            )
        if corpus_id_counts[poem_id] > 1:
            issues.append(
                make_issue(
                    poem_id,
                    author,
                    title,
                    "DUPLICATE_POEM_ID",
                    original_text,
                    tokens,
                    f"语料中该 ID 出现 {corpus_id_counts[poem_id]} 次",
                )
            )
        if token_id_counts[poem_id] > 1:
            issues.append(
                make_issue(
                    poem_id,
                    author,
                    title,
                    "DUPLICATE_TOKENIZED_ID",
                    original_text,
                    tokens,
                    f"分词结果中该 ID 出现 {token_id_counts[poem_id]} 次",
                )
            )
        if not author:
            issues.append(
                make_issue(
                    poem_id, author, title, "MISSING_AUTHOR", original_text, tokens
                )
            )
        if not title:
            issues.append(
                make_issue(
                    poem_id, author, title, "MISSING_TITLE", original_text, tokens
                )
            )
        if not original_text.strip():
            issues.append(
                make_issue(
                    poem_id, author, title, "EMPTY_TEXT", original_text, tokens
                )
            )
        if not tokens:
            issues.append(
                make_issue(
                    poem_id, author, title, "EMPTY_TOKENS", original_text, tokens
                )
            )
        empty_indices = [index for index, token in enumerate(tokens) if token == ""]
        if empty_indices:
            issues.append(
                make_issue(
                    poem_id,
                    author,
                    title,
                    "EMPTY_TOKEN_ELEMENT",
                    original_text,
                    tokens,
                    f"空 token 索引：{empty_indices}",
                )
            )
        reconstructed = "".join(tokens)
        if reconstructed != original_text:
            issues.append(
                make_issue(
                    poem_id,
                    author,
                    title,
                    "TEXT_TOKEN_MISMATCH",
                    original_text,
                    tokens,
                    f"原文字符数={len(original_text)}；token 重构字符数={len(reconstructed)}",
                )
            )

        exact_key = (author, title, original_text, tuple(tokens))
        if exact_key in exact_seen:
            issues.append(
                make_issue(
                    poem_id,
                    author,
                    title,
                    "DUPLICATE_RECORD",
                    original_text,
                    tokens,
                    f"与 poem_id={exact_seen[exact_key]} 完全重复",
                )
            )
        else:
            exact_seen[exact_key] = poem_id

    for poem_id, queue in token_groups.items():
        while queue:
            token_line, token_row = queue.popleft()
            raw_tokens = token_row.get("tokens")
            tokens = raw_tokens if isinstance(raw_tokens, list) else []
            safe_tokens = [str(token) for token in tokens]
            issues.append(
                make_issue(
                    poem_id,
                    "",
                    "",
                    "ORPHAN_TOKENIZED_RECORD",
                    "",
                    safe_tokens,
                    f"分词文件第 {token_line} 行没有对应语料记录",
                )
            )
    return records, issues


def find_protected_segmentation_issues(
    record: PoemRecord, protected_rows: Sequence[Mapping[str, str]]
) -> list[dict[str, Any]]:
    """找出保护词字符连续出现、但被切成多个 token 的实例。"""

    tokens: list[str] = []
    original_indices: list[int] = []
    starts: list[int] = []
    position = 0
    for index, token in enumerate(record.normalized_tokens):
        if not token:
            continue
        tokens.append(token)
        original_indices.append(index)
        starts.append(position)
        position += len(token)
    text = "".join(tokens)
    issues: list[dict[str, Any]] = []
    if not text:
        return issues

    for term_row in protected_rows:
        surface = term_row.get("surface", "").strip()
        if len(surface) < 2:
            continue
        start = text.find(surface)
        while start >= 0:
            end = start + len(surface)
            left = bisect.bisect_right(starts, start) - 1
            right = bisect.bisect_right(starts, end - 1) - 1
            if 0 <= left < len(tokens) and 0 <= right < len(tokens) and left != right:
                issues.append(
                    {
                        "poem_id": record.poem_id,
                        "author": record.author,
                        "title": record.title,
                        "protected_word": surface,
                        "observed_tokens": " / ".join(tokens[left : right + 1]),
                        "start_char": start,
                        "end_char": end,
                        "start_token_index": original_indices[left],
                        "end_token_index": original_indices[right],
                        "decision": term_row.get("decision", ""),
                        "category": term_row.get("category", ""),
                        "scope": term_row.get("scope", ""),
                        "original_text": record.original_text,
                    }
                )
            start = text.find(surface, start + 1)
    return issues


def exceeds_max_df(max_df: Any, df: int, df_ratio: float) -> bool:
    """按照规则值判断是否超过自动最大 DF。"""

    if max_df is None:
        return False
    if isinstance(max_df, bool) or not isinstance(max_df, (int, float)):
        raise CleaningError("automatic_max_df 必须为 null、整数或数值比例")
    if max_df <= 0:
        raise CleaningError("automatic_max_df 必须大于 0")
    return df_ratio > float(max_df) if max_df <= 1 else df > int(max_df)


def token_decision(
    token: str,
    tf: int,
    df: int,
    df_ratio: float,
    protected_words: set[str],
    stopwords: set[str],
    min_tf: int,
    min_df: int,
    automatic_max_df: Any,
) -> tuple[bool, str, tuple[str, ...], bool]:
    """按固定优先级返回保留判断、主原因、全部潜在原因和保护挽救标记。"""

    reasons: list[str] = []
    if token in stopwords:
        reasons.append("STOPWORD")
    if not is_han_token(token):
        reasons.append("OTHER_RULE")
    if tf < min_tf:
        reasons.append("LOW_TF")
    if df < min_df:
        reasons.append("LOW_DF")
    if exceeds_max_df(automatic_max_df, df, df_ratio):
        reasons.append("HIGH_DF")

    if token in protected_words:
        return True, "PROTECTED", tuple(reasons), bool(reasons)
    precedence = ("STOPWORD", "OTHER_RULE", "LOW_TF", "LOW_DF", "HIGH_DF")
    for reason in precedence:
        if reason in reasons:
            return False, reason, tuple(reasons), False
    return True, "RETAINED", (), False


def ratio(numerator: int, denominator: int) -> float:
    """安全计算比例。"""

    return numerator / denominator if denominator else 0.0


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """生成简单 Markdown 表格并转义竖线。"""

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def dependency_versions() -> dict[str, str]:
    """记录本程序可能使用的主要依赖版本。"""

    versions = {"python": platform.python_version(), "standard_library_only": "true"}
    try:
        versions["PyYAML"] = importlib.metadata.version("PyYAML")
    except importlib.metadata.PackageNotFoundError:
        versions["PyYAML"] = "not installed; JSON-compatible YAML parser used"
    return versions


def build_summary(
    *,
    started_at: str,
    runtime_seconds: float,
    input_inventory: Sequence[Mapping[str, Any]],
    resource_counts: Mapping[str, int],
    profile_name: str,
    profile: Mapping[str, Any],
    records: Sequence[PoemRecord],
    data_issues: Sequence[Mapping[str, Any]],
    before_tf: Counter[str],
    after_tf: Counter[str],
    reason_occurrences: Counter[str],
    reason_types: Counter[str],
    normalization_events: Sequence[Mapping[str, Any]],
    protected_retained_occurrences: int,
    protected_rescue_rows: Sequence[Mapping[str, Any]],
    conflict_rows: Sequence[Mapping[str, Any]],
    segmentation_rows: Sequence[Mapping[str, Any]],
    low_retention_rows: Sequence[Mapping[str, Any]],
    removed_rows: Sequence[Mapping[str, Any]],
    token_stats_rows: Sequence[Mapping[str, Any]],
    empty_text_count: int,
) -> str:
    """生成可审计的 Markdown 清洗报告。"""

    poem_count = len(records)
    valid_poems = sum(bool(record.cleaned_tokens) for record in records)
    author_count = len({record.author for record in records if record.author})
    original_token_lengths = [len(record.original_tokens) for record in records]
    before_token_count = sum(before_tf.values())
    after_token_count = sum(after_tf.values())
    before_han_tf = Counter(
        {token: count for token, count in before_tf.items() if is_han_token(token)}
    )
    before_han_count = sum(before_han_tf.values())
    issue_counts = Counter(row["issue_type"] for row in data_issues)
    zero_count = sum(not record.cleaned_tokens for record in records)
    low_nonzero_count = sum(
        "LOW_TOKEN_COUNT" in str(row.get("review_reasons", ""))
        for row in low_retention_rows
    )
    low_ratio_count = sum(
        "LOW_RETENTION_RATIO" in str(row.get("review_reasons", ""))
        for row in low_retention_rows
    )

    top_removed = [
        (
            row["token"],
            row["affected_occurrences"],
            row["DF"],
            row["DF_ratio"],
            row["removal_reason"],
        )
        for row in removed_rows[:100]
    ]
    top_df = [
        (
            row["token"],
            row["TF"],
            row["DF"],
            row["DF_ratio"],
            row["protected"],
            row["stopword"],
            row["automatic_removal_reason"],
        )
        for row in sorted(
            token_stats_rows,
            key=lambda row: (-int(row["DF"]), -int(row["TF"]), str(row["token"])),
        )[:100]
    ]
    rescue_table = [
        (
            row["token"],
            row["affected_occurrences"],
            row["DF"],
            row["avoided_removal_reasons"],
        )
        for row in protected_rescue_rows
    ]

    lines = [
        "# 《全唐诗》LDA 语料清洗报告",
        "",
        "## Run",
        "",
        f"- 开始时间：{started_at}",
        f"- 本报告生成时耗时：{runtime_seconds:.3f} 秒",
        f"- 清洗档位：`{profile_name}`",
        f"- 最小 TF：{profile['min_tf']}",
        f"- 最小 DF：{profile['min_df']}",
        f"- 自动最大 DF：{profile.get('automatic_max_df')}",
        "- 语义映射：仅输出旁表，未改写主语料",
        "- 异常记录：全部保留，只写入人工审核文件",
        "",
        "### 实际读取文件",
        "",
        markdown_table(
            ["角色", "路径", "SHA-256"],
            [(row["role"], row["path"], row["sha256"]) for row in input_inventory],
        ),
        "",
        "### 规则资源规模",
        "",
        markdown_table(
            ["资源", "记录数"],
            [(name, count) for name, count in resource_counts.items()],
        ),
        "",
        "## Corpus",
        "",
        f"- 清洗前诗歌数量：{poem_count:,}",
        f"- 作者总数（非空作者去重）：{author_count:,}",
        f"- 每首诗平均原始 token 数：{statistics.fmean(original_token_lengths):.6f}",
        f"- 每首诗原始 token 数中位数：{statistics.median(original_token_lengths):.6f}",
        f"- 最短诗歌原始 token 数：{min(original_token_lengths, default=0):,}",
        f"- 最长诗歌原始 token 数：{max(original_token_lengths, default=0):,}",
        f"- 清洗后有效诗歌数量：{valid_poems:,}",
        f"- 清洗后零 token 诗歌数量：{zero_count:,}",
        f"- 空原文数量：{empty_text_count:,}",
        f"- 空分词记录数量：{issue_counts['EMPTY_TOKENS']:,}",
        f"- 完全重复记录数量（不含首次出现）：{issue_counts['DUPLICATE_RECORD']:,}",
        f"- 缺失作者数量：{issue_counts['MISSING_AUTHOR']:,}",
        f"- 缺失题目数量：{issue_counts['MISSING_TITLE']:,}",
        "- 主语料没有自动删除任何诗；零 token 行仍保留并与 `lda_tokens.txt` 对齐。",
        "",
        "## Vocabulary",
        "",
        f"- 清洗前 token 总数（含标点等非汉字 token）：{before_token_count:,}",
        f"- 清洗后 token 总数：{after_token_count:,}",
        f"- 清洗前 vocabulary size：{len(before_tf):,}",
        f"- 清洗后 vocabulary size：{len(after_tf):,}",
        f"- 清洗前纯汉字 token 总数：{before_han_count:,}",
        f"- 清洗前纯汉字 vocabulary size：{len(before_han_tf):,}",
        "",
        "## Cleaning",
        "",
        f"- normalization 实际替换次数：{len(normalization_events):,}",
        f"- STOPWORD 删除词次 / 词型：{reason_occurrences['STOPWORD']:,} / {reason_types['STOPWORD']:,}",
        f"- LOW_TF 删除词次 / 词型：{reason_occurrences['LOW_TF']:,} / {reason_types['LOW_TF']:,}",
        f"- LOW_DF 删除词次 / 词型：{reason_occurrences['LOW_DF']:,} / {reason_types['LOW_DF']:,}",
        f"- HIGH_DF 删除词次 / 词型：{reason_occurrences['HIGH_DF']:,} / {reason_types['HIGH_DF']:,}",
        f"- OTHER_RULE 删除词次 / 词型：{reason_occurrences['OTHER_RULE']:,} / {reason_types['OTHER_RULE']:,}",
        f"- protected words 保留词次：{protected_retained_occurrences:,}",
        f"- protected words 实际挽救词次 / 词型：{sum(int(row['affected_occurrences']) for row in protected_rescue_rows):,} / {len(protected_rescue_rows):,}",
        f"- 保护词与停用词冲突：{len(conflict_rows):,} 个词型",
        "",
        "`OTHER_RULE` 在当前规则中表示非纯汉字 token（主要为标点）；没有新增任何其他推断规则。"
        " `token_statistics.csv` 为保证所有删除可追溯，统计规范化视图中的全部 token，"
        "因此也保留已标为停用词或非汉字的统计行。",
        "",
        "## Retention",
        "",
        f"- Token Retention Rate：{ratio(after_token_count, before_token_count):.6%}",
        f"- 汉字 Token Retention Rate：{ratio(after_token_count, before_han_count):.6%}",
        f"- Vocabulary Retention Rate：{ratio(len(after_tf), len(before_tf)):.6%}",
        f"- 汉字 Vocabulary Retention Rate：{ratio(len(after_tf), len(before_han_tf)):.6%}",
        f"- Poem Retention Rate：{ratio(valid_poems, poem_count):.6%}",
        "",
        "## 数据格式与质量问题",
        "",
    ]
    if issue_counts:
        lines.append(
            markdown_table(
                ["issue_type", "记录数"], sorted(issue_counts.items())
            )
        )
    else:
        lines.append("未发现数据质量问题。")
    lines.extend(
        [
            "",
            "详细记录见 `reports/data_quality_issues.csv`；这些记录均未自动删除。",
            "",
            "## 规则冲突",
            "",
            f"保护词与停用词交集共有 {len(conflict_rows):,} 个词型，均按保护词优先保留。",
            "",
            "## 删除数量最多的前 100 个 token",
            "",
            markdown_table(
                ["token", "删除词次", "DF", "DF_ratio", "主删除原因"], top_removed
            ),
            "",
            "完整表见 `reports/removed_tokens.csv`；独立前 100 表见 `reports/top_100_removed_tokens.csv`。",
            "",
            "## DF 最高的前 100 个 token",
            "",
            markdown_table(
                ["token", "TF", "DF", "DF_ratio", "protected", "stopword", "处理结果"],
                top_df,
            ),
            "",
            "独立表见 `reports/top_100_df_tokens.csv`。",
            "",
            "## 被保护规则挽救的 token",
            "",
        ]
    )
    if rescue_table:
        lines.append(
            markdown_table(
                ["token", "词次", "DF", "避免的删除原因"], rescue_table
            )
        )
    else:
        lines.append("本次没有 token 需要由保护优先级挽救。")
    lines.extend(
        [
            "",
            "完整审计见 `reports/protected_rescues.csv`。",
            "",
            "## 疑似保护词错误切分",
            "",
            f"- 疑似实例数：{len(segmentation_rows):,}",
            f"- 涉及诗歌数：{len({row['poem_id'] for row in segmentation_rows}):,}",
            f"- 涉及保护词型数：{len({row['protected_word'] for row in segmentation_rows}):,}",
            "- 程序只报告跨多个既有 token 的连续字面命中，没有自动合并。",
            "- 详细实例见 `reports/protected_word_segmentation_issues.csv`。",
            "",
            "## 清洗后 token 过少的诗",
            "",
            f"- 清洗后 0 token：{zero_count:,}",
            f"- 非零但达到低 token 审核阈值：{low_nonzero_count:,}",
            f"- retention_ratio 低于审核阈值：{low_ratio_count:,}",
            f"- 汇总进入人工审核表的诗：{len(low_retention_rows):,}",
            "- 这些审核阈值只用于报告，不会触发删除。",
            "- 详细记录见 `reports/low_retention_poems.csv`。",
            "",
        ]
    )
    return "\n".join(lines)


def run_cleaning(
    config_path: Path = DEFAULT_CONFIG, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """执行完整清洗流程并返回核心统计。"""

    started_clock = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    config = load_config(config_path)
    paths = prepare_paths(config, project_root)
    logger = setup_logger(paths["logs_dir"] / "clean_corpus.log")
    logger.info("开始读取配置：%s", config_path)

    cleaning_config = require_mapping(config.get("cleaning"), "cleaning")
    review_config = require_mapping(config.get("quality_review"), "quality_review")
    profile_name = str(cleaning_config.get("profile", ""))
    if not profile_name:
        raise CleaningError("配置缺少 cleaning.profile")
    token_separator = str(cleaning_config.get("token_separator", " "))
    automatic_only = bool(
        cleaning_config.get("apply_automatic_normalization_only", True)
    )

    rules_document = json.loads(paths["filtering_rules"].read_text(encoding="utf-8-sig"))
    if profile_name not in rules_document:
        raise CleaningError(f"过滤规则不存在档位：{profile_name}")
    profile = require_mapping(rules_document[profile_name], profile_name)
    for key in ("stopword_tier", "min_tf", "min_df", "automatic_max_df"):
        if key not in profile:
            raise CleaningError(f"过滤档位缺少字段：{profile_name}.{key}")
    stopword_tier = str(profile["stopword_tier"])
    if stopword_tier not in paths["stopwords"]:
        raise CleaningError(f"配置没有提供 {stopword_tier} 停用词路径")
    stopword_path = paths["stopwords"][stopword_tier]
    min_tf = int(profile["min_tf"])
    min_df = int(profile["min_df"])
    if min_tf < 1 or min_df < 1:
        raise CleaningError("min_tf 和 min_df 必须至少为 1")
    automatic_max_df = profile["automatic_max_df"]
    high_df_audit_rate = float(profile.get("high_df_manual_audit_rate", 1.0))

    low_token_threshold = int(review_config["low_token_count_threshold"])
    low_retention_threshold = float(review_config["low_retention_ratio_threshold"])
    sample_size = int(review_config["manual_sample_size"])
    random_seed = int(review_config["random_seed"])
    if low_token_threshold < 0 or not 0 <= low_retention_threshold <= 1:
        raise CleaningError("人工审核阈值超出有效范围")
    if sample_size < 0:
        raise CleaningError("manual_sample_size 不得为负数")

    corpus_rows = read_jsonl(paths["corpus"])
    token_rows = read_jsonl(paths["tokenized_corpus"])
    normalization_rules = load_normalization_rules(paths["normalization"])
    stopwords = set(read_terms(stopword_path))
    protected_rows = read_tsv(paths["protected_words"])
    semantic_rows = read_tsv(paths["semantic_mapping"])
    protected_words = {row.get("surface", "").strip() for row in protected_rows}
    if "" in protected_words:
        raise CleaningError("保护词表存在空 surface")
    if len(protected_words) != len(protected_rows):
        raise CleaningError("保护词表 surface 存在重复项")
    logger.info(
        "已读取语料 %d 行、分词结果 %d 行、规范规则 %d 条、%s 停用词 %d 条、保护词 %d 条、语义映射 %d 条",
        len(corpus_rows),
        len(token_rows),
        len(normalization_rules),
        stopword_tier,
        len(stopwords),
        len(protected_words),
        len(semantic_rows),
    )

    records, data_issues = align_records(corpus_rows, token_rows)
    logger.info("数据对齐完成；发现 %d 条质量问题记录", len(data_issues))

    normalization_events: list[dict[str, Any]] = []
    normalization_hit_counts: Counter[str] = Counter()
    normalized_target_counts: Counter[str] = Counter()
    for record in records:
        normalized, events = apply_normalization(
            record.poem_id, record.original_tokens, normalization_rules, automatic_only
        )
        record.normalized_tokens = normalized
        record.normalization_events = events
        normalization_events.extend(events)
        for event in events:
            normalization_hit_counts[str(event["rule_id"])] += 1
            normalized_target_counts[str(event["normalized_token"])] += 1
    logger.info("词形规范完成；实际替换 %d 次", len(normalization_events))

    segmentation_rows: list[dict[str, Any]] = []
    for record in records:
        segmentation_rows.extend(
            find_protected_segmentation_issues(record, protected_rows)
        )
    segmentation_rows.sort(
        key=lambda row: (str(row["poem_id"]), int(row["start_char"]), row["protected_word"])
    )
    logger.info("保护词切分检查完成；发现 %d 个疑似实例", len(segmentation_rows))

    token_tf: Counter[str] = Counter()
    token_df: Counter[str] = Counter()
    for record in records:
        token_tf.update(record.normalized_tokens)
        token_df.update(set(record.normalized_tokens))
    effective_poem_count = sum(
        any(
            token in protected_words
            or (is_han_token(token) and token not in stopwords)
            for token in record.normalized_tokens
        )
        for record in records
    )
    logger.info(
        "TF/DF 统计完成；词型 %d，有效 DF 分母 %d 首", len(token_tf), effective_poem_count
    )

    decisions: dict[str, tuple[bool, str, tuple[str, ...], bool]] = {}
    for token, tf in token_tf.items():
        df = token_df[token]
        decisions[token] = token_decision(
            token,
            tf,
            df,
            ratio(df, effective_poem_count),
            protected_words,
            stopwords,
            min_tf,
            min_df,
            automatic_max_df,
        )

    conflict_tokens = sorted(protected_words & stopwords)
    conflict_rows = [
        {
            "token": token,
            "conflict_types": "PROTECTED_WORD|STOPWORD",
            "resolution": "保留（protected_words 优先）",
            "TF": token_tf[token],
            "DF": token_df[token],
            "DF_ratio": f"{ratio(token_df[token], effective_poem_count):.8f}",
        }
        for token in conflict_tokens
    ]

    protected_rescue_rows: list[dict[str, Any]] = []
    for token in sorted(protected_words & token_tf.keys()):
        keep, primary_reason, all_reasons, rescued = decisions[token]
        if keep and primary_reason == "PROTECTED" and rescued:
            protected_rescue_rows.append(
                {
                    "token": token,
                    "TF": token_tf[token],
                    "DF": token_df[token],
                    "DF_ratio": f"{ratio(token_df[token], effective_poem_count):.8f}",
                    "avoided_removal_reasons": "|".join(all_reasons),
                    "affected_poems": token_df[token],
                    "affected_occurrences": token_tf[token],
                }
            )
    protected_rescue_rows.sort(
        key=lambda row: (-int(row["affected_occurrences"]), str(row["token"]))
    )

    reason_occurrences: Counter[str] = Counter()
    protected_retained_occurrences = 0
    for record in records:
        cleaned: list[str] = []
        removed: list[str] = []
        counts: Counter[str] = Counter()
        protected_count = 0
        rescued_count = 0
        for token in record.normalized_tokens:
            keep, reason, _all_reasons, rescued = decisions[token]
            if keep:
                cleaned.append(token)
                if token in protected_words:
                    protected_count += 1
                    protected_retained_occurrences += 1
                    if rescued:
                        rescued_count += 1
            else:
                removed.append(token)
                counts[reason] += 1
                reason_occurrences[reason] += 1
        record.cleaned_tokens = cleaned
        record.removed_tokens = removed
        record.removal_counts = counts
        record.protected_token_count = protected_count
        record.protected_rescued_count = rescued_count

    token_stats_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    reason_types: Counter[str] = Counter()
    for token, tf in token_tf.items():
        df = token_df[token]
        df_ratio = ratio(df, effective_poem_count)
        keep, reason, all_reasons, _rescued = decisions[token]
        row = {
            "token": token,
            "TF": tf,
            "DF": df,
            "DF_ratio": f"{df_ratio:.8f}",
            "protected": str(token in protected_words).lower(),
            "stopword": str(token in stopwords).lower(),
            "normalization_status": (
                "NORMALIZED_TARGET" if normalized_target_counts[token] else "UNCHANGED"
            ),
            "normalization_hit_count": normalized_target_counts[token],
            "high_df_manual_audit": str(df_ratio >= high_df_audit_rate).lower(),
            "automatic_removal_reason": reason,
            "retained": str(keep).lower(),
        }
        token_stats_rows.append(row)
        if not keep:
            reason_types[reason] += 1
            removed_rows.append(
                {
                    "token": token,
                    "TF": tf,
                    "DF": df,
                    "DF_ratio": f"{df_ratio:.8f}",
                    "removal_reason": reason,
                    "all_removal_reasons": "|".join(all_reasons),
                    "affected_poems": df,
                    "affected_occurrences": tf,
                }
            )
    token_stats_rows.sort(key=lambda row: (-int(row["TF"]), str(row["token"])))
    removed_rows.sort(
        key=lambda row: (-int(row["affected_occurrences"]), str(row["token"]))
    )
    logger.info(
        "过滤完成；保留 %d 词次，删除 %d 词次",
        sum(len(record.cleaned_tokens) for record in records),
        sum(reason_occurrences.values()),
    )

    poem_statistics_rows: list[dict[str, Any]] = []
    low_retention_rows: list[dict[str, Any]] = []
    for record in records:
        original_count = len(record.original_tokens)
        cleaned_count = len(record.cleaned_tokens)
        retention = ratio(cleaned_count, original_count)
        review_reasons: list[str] = []
        if cleaned_count == 0:
            review_reasons.append("ZERO_CLEANED_TOKENS")
        elif cleaned_count <= low_token_threshold:
            review_reasons.append("LOW_TOKEN_COUNT")
        if original_count > 0 and retention < low_retention_threshold:
            review_reasons.append("LOW_RETENTION_RATIO")
        if record.removal_counts["HIGH_DF"]:
            review_reasons.append("HIGH_DF_FILTER_IMPACT")
        row = {
            "poem_id": record.poem_id,
            "author": record.author,
            "title": record.title,
            "original_token_count": original_count,
            "cleaned_token_count": cleaned_count,
            "removed_token_count": len(record.removed_tokens),
            "retention_ratio": f"{retention:.8f}",
            "stopword_removed_count": record.removal_counts["STOPWORD"],
            "low_tf_removed_count": record.removal_counts["LOW_TF"],
            "low_df_removed_count": record.removal_counts["LOW_DF"],
            "high_df_removed_count": record.removal_counts["HIGH_DF"],
            "other_rule_removed_count": record.removal_counts["OTHER_RULE"],
            "protected_token_count": record.protected_token_count,
            "protected_rescued_count": record.protected_rescued_count,
            "review_reasons": "|".join(review_reasons),
        }
        poem_statistics_rows.append(row)
        if review_reasons:
            low_retention_rows.append(
                {
                    **row,
                    "original_text": record.original_text,
                    "original_tokens": token_separator.join(record.original_tokens),
                    "cleaned_tokens": token_separator.join(record.cleaned_tokens),
                }
            )

    output_dir = paths["output_dir"]
    reports_dir = paths["reports_dir"]
    lda_rows = [
        {
            "poem_id": record.poem_id,
            "author": record.author,
            "title": record.title,
            "original_text": record.original_text,
            "cleaned_tokens": token_separator.join(record.cleaned_tokens),
            "token_count": len(record.cleaned_tokens),
        }
        for record in records
    ]
    write_csv(
        output_dir / "lda_corpus.csv",
        ["poem_id", "author", "title", "original_text", "cleaned_tokens", "token_count"],
        lda_rows,
    )
    with (output_dir / "lda_tokens.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(token_separator.join(record.cleaned_tokens) + "\n")

    semantic_expanded_rows: list[dict[str, Any]] = []
    for row in semantic_rows:
        surface = row.get("surface", "")
        semantic_expanded_rows.append(
            {
                **row,
                "TF": token_tf[surface],
                "DF": token_df[surface],
                "DF_ratio": f"{ratio(token_df[surface], effective_poem_count):.8f}",
                "present_as_token": str(surface in token_tf).lower(),
                "applied_to_lda_corpus": "false",
            }
        )
    semantic_headers = list(semantic_rows[0].keys()) if semantic_rows else []
    semantic_headers.extend(
        ["TF", "DF", "DF_ratio", "present_as_token", "applied_to_lda_corpus"]
    )
    write_csv(
        output_dir / "semantic_mapping_expanded.csv",
        semantic_headers,
        semantic_expanded_rows,
    )

    write_csv(
        reports_dir / "data_quality_issues.csv",
        [
            "poem_id",
            "author",
            "title",
            "issue_type",
            "original_text",
            "tokenized_text",
            "issue_detail",
        ],
        data_issues,
    )
    write_csv(
        reports_dir / "normalization_log.csv",
        [
            "poem_id",
            "token_index",
            "original_token",
            "normalized_token",
            "rule_type",
            "scope",
            "status",
            "rule_id",
        ],
        normalization_events,
    )
    normalization_stats_rows = [
        {
            "rule_id": rule.rule_id,
            "source_form": rule.source_form,
            "normalized_form": rule.normalized_form,
            "rule_type": rule.relation,
            "scope": rule.scope,
            "status": rule.status,
            "apply_automatically": str(rule.apply_automatically).lower(),
            "actual_hit_count": normalization_hit_counts[rule.rule_id],
        }
        for rule in normalization_rules
    ]
    write_csv(
        reports_dir / "normalization_rule_statistics.csv",
        [
            "rule_id",
            "source_form",
            "normalized_form",
            "rule_type",
            "scope",
            "status",
            "apply_automatically",
            "actual_hit_count",
        ],
        normalization_stats_rows,
    )
    write_csv(
        reports_dir / "rule_conflicts.csv",
        ["token", "conflict_types", "resolution", "TF", "DF", "DF_ratio"],
        conflict_rows,
    )
    write_csv(
        reports_dir / "protected_word_segmentation_issues.csv",
        [
            "poem_id",
            "author",
            "title",
            "protected_word",
            "observed_tokens",
            "start_char",
            "end_char",
            "start_token_index",
            "end_token_index",
            "decision",
            "category",
            "scope",
            "original_text",
        ],
        segmentation_rows,
    )
    write_csv(
        reports_dir / "token_statistics.csv",
        [
            "token",
            "TF",
            "DF",
            "DF_ratio",
            "protected",
            "stopword",
            "normalization_status",
            "normalization_hit_count",
            "high_df_manual_audit",
            "automatic_removal_reason",
            "retained",
        ],
        token_stats_rows,
    )
    write_csv(
        reports_dir / "removed_tokens.csv",
        [
            "token",
            "TF",
            "DF",
            "DF_ratio",
            "removal_reason",
            "all_removal_reasons",
            "affected_poems",
            "affected_occurrences",
        ],
        removed_rows,
    )
    write_csv(
        reports_dir / "protected_rescues.csv",
        [
            "token",
            "TF",
            "DF",
            "DF_ratio",
            "avoided_removal_reasons",
            "affected_poems",
            "affected_occurrences",
        ],
        protected_rescue_rows,
    )
    poem_stat_headers = [
        "poem_id",
        "author",
        "title",
        "original_token_count",
        "cleaned_token_count",
        "removed_token_count",
        "retention_ratio",
        "stopword_removed_count",
        "low_tf_removed_count",
        "low_df_removed_count",
        "high_df_removed_count",
        "other_rule_removed_count",
        "protected_token_count",
        "protected_rescued_count",
        "review_reasons",
    ]
    write_csv(
        reports_dir / "poem_cleaning_statistics.csv",
        poem_stat_headers,
        poem_statistics_rows,
    )
    write_csv(
        reports_dir / "low_retention_poems.csv",
        poem_stat_headers + ["original_text", "original_tokens", "cleaned_tokens"],
        low_retention_rows,
    )

    sample_indices = sorted(
        random.Random(random_seed).sample(
            range(len(records)), min(sample_size, len(records))
        )
    )
    sample_rows: list[dict[str, Any]] = []
    for index in sample_indices:
        record = records[index]
        normalized_descriptions = [
            f"{event['token_index']}:{event['original_token']}→{event['normalized_token']}"
            for event in record.normalization_events
        ]
        sample_rows.append(
            {
                "poem_id": record.poem_id,
                "author": record.author,
                "title": record.title,
                "original_text": record.original_text,
                "original_tokens": token_separator.join(record.original_tokens),
                "cleaned_tokens": token_separator.join(record.cleaned_tokens),
                "removed_tokens": token_separator.join(record.removed_tokens),
                "normalized_tokens": " | ".join(normalized_descriptions),
                "token_retention_ratio": f"{ratio(len(record.cleaned_tokens), len(record.original_tokens)):.8f}",
            }
        )
    write_csv(
        reports_dir / "manual_check_sample.csv",
        [
            "poem_id",
            "author",
            "title",
            "original_text",
            "original_tokens",
            "cleaned_tokens",
            "removed_tokens",
            "normalized_tokens",
            "token_retention_ratio",
        ],
        sample_rows,
    )

    top_removed_rows = removed_rows[:100]
    top_df_rows = sorted(
        token_stats_rows,
        key=lambda row: (-int(row["DF"]), -int(row["TF"]), str(row["token"])),
    )[:100]
    write_csv(
        reports_dir / "top_100_removed_tokens.csv",
        [
            "token",
            "TF",
            "DF",
            "DF_ratio",
            "removal_reason",
            "all_removal_reasons",
            "affected_poems",
            "affected_occurrences",
        ],
        top_removed_rows,
    )
    write_csv(
        reports_dir / "top_100_df_tokens.csv",
        [
            "token",
            "TF",
            "DF",
            "DF_ratio",
            "protected",
            "stopword",
            "normalization_status",
            "normalization_hit_count",
            "high_df_manual_audit",
            "automatic_removal_reason",
            "retained",
        ],
        top_df_rows,
    )

    input_files = [
        ("config", config_path),
        ("corpus", paths["corpus"]),
        ("tokenized_corpus", paths["tokenized_corpus"]),
        ("normalization", paths["normalization"]),
        (f"stopwords_{stopword_tier}", stopword_path),
        ("protected_words", paths["protected_words"]),
        ("filtering_rules", paths["filtering_rules"]),
        ("semantic_mapping", paths["semantic_mapping"]),
    ]
    input_inventory = [
        {"role": role, "path": display_path(path), "sha256": file_sha256(path)}
        for role, path in input_files
    ]
    resource_counts = {
        "normalization": len(normalization_rules),
        f"stopwords_{stopword_tier}": len(stopwords),
        "protected_words": len(protected_words),
        "filter_profiles": sum(
            name.endswith("_profile") for name in rules_document.keys()
        ),
        "semantic_mapping": len(semantic_rows),
    }
    after_tf = Counter(
        token for record in records for token in record.cleaned_tokens
    )
    runtime_at_summary = time.perf_counter() - started_clock
    summary = build_summary(
        started_at=started_at,
        runtime_seconds=runtime_at_summary,
        input_inventory=input_inventory,
        resource_counts=resource_counts,
        profile_name=profile_name,
        profile=profile,
        records=records,
        data_issues=data_issues,
        before_tf=token_tf,
        after_tf=after_tf,
        reason_occurrences=reason_occurrences,
        reason_types=reason_types,
        normalization_events=normalization_events,
        protected_retained_occurrences=protected_retained_occurrences,
        protected_rescue_rows=protected_rescue_rows,
        conflict_rows=conflict_rows,
        segmentation_rows=segmentation_rows,
        low_retention_rows=low_retention_rows,
        removed_rows=removed_rows,
        token_stats_rows=token_stats_rows,
        empty_text_count=sum(not record.original_text.strip() for record in records),
    )
    (reports_dir / "cleaning_summary.md").write_text(summary, encoding="utf-8")

    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    runtime_seconds = time.perf_counter() - started_clock
    output_paths = sorted(
        [path for path in output_dir.iterdir() if path.is_file()]
        + [
            path
            for path in reports_dir.iterdir()
            if path.is_file() and path.name != "run_metadata.json"
        ],
        key=lambda path: str(path),
    )
    metadata = {
        "schema_version": 1,
        "method": "rule_bound_lda_corpus_cleaning",
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime_seconds": round(runtime_seconds, 6),
        "inputs": input_inventory,
        "resource_counts": resource_counts,
        "rules": {
            "schema_version": rules_document.get("schema_version"),
            "profile": profile_name,
            "stopword_tier": stopword_tier,
            "min_tf": min_tf,
            "min_df": min_df,
            "automatic_max_df": automatic_max_df,
            "high_df_manual_audit_rate": high_df_audit_rate,
            "semantic_mapping_applied": False,
            "automatic_normalization_only": automatic_only,
            "protected_words_priority": True,
        },
        "quality_review": {
            "low_token_count_threshold": low_token_threshold,
            "low_retention_ratio_threshold": low_retention_threshold,
            "manual_sample_size": len(sample_rows),
            "random_seed": random_seed,
            "thresholds_affect_cleaning": False,
        },
        "environment": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
        },
        "statistics": {
            "poem_count": len(records),
            "author_count": len({record.author for record in records if record.author}),
            "original_tokens_per_poem_mean": statistics.fmean(
                len(record.original_tokens) for record in records
            ),
            "original_tokens_per_poem_median": statistics.median(
                len(record.original_tokens) for record in records
            ),
            "original_tokens_per_poem_min": min(
                (len(record.original_tokens) for record in records), default=0
            ),
            "original_tokens_per_poem_max": max(
                (len(record.original_tokens) for record in records), default=0
            ),
            "empty_text_count": sum(
                not record.original_text.strip() for record in records
            ),
            "empty_tokenized_record_count": sum(
                not record.original_tokens for record in records
            ),
            "duplicate_record_count": sum(
                row["issue_type"] == "DUPLICATE_RECORD" for row in data_issues
            ),
            "missing_author_count": sum(
                row["issue_type"] == "MISSING_AUTHOR" for row in data_issues
            ),
            "missing_title_count": sum(
                row["issue_type"] == "MISSING_TITLE" for row in data_issues
            ),
            "effective_df_denominator": effective_poem_count,
            "cleaned_nonempty_poem_count": sum(
                bool(record.cleaned_tokens) for record in records
            ),
            "input_token_count": sum(token_tf.values()),
            "cleaned_token_count": sum(after_tf.values()),
            "input_vocabulary_size": len(token_tf),
            "cleaned_vocabulary_size": len(after_tf),
            "normalization_occurrences": len(normalization_events),
            "data_quality_issue_rows": len(data_issues),
            "rule_conflict_types": len(conflict_rows),
            "protected_segmentation_issue_occurrences": len(segmentation_rows),
            "low_retention_review_poems": len(low_retention_rows),
            "removal_occurrences_by_primary_reason": dict(reason_occurrences),
            "removal_types_by_primary_reason": dict(reason_types),
        },
        "outputs": [
            {
                "path": display_path(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in output_paths
        ],
    }
    write_json(reports_dir / "run_metadata.json", metadata)
    logger.info("全部输出完成；总耗时 %.3f 秒", runtime_seconds)
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="严格依照既有资源清洗《全唐诗》LDA 分词语料"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="清洗配置文件路径",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口；所有异常均转换为明确错误信息。"""

    args = parse_args(argv)
    try:
        run_cleaning(args.config.resolve())
    except Exception as exc:
        logging.getLogger("clean_corpus").exception("清洗失败：%s", exc)
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
