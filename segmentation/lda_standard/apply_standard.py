"""把《全唐诗》LDA 分词规范确定性地应用到 sikufenci 基础结果。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
STANDARD_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "segmentation" / "data" / "poems.jsonl"
DEFAULT_INPUT = ROOT / "segmentation" / "results" / "sikufenci.jsonl"
DEFAULT_OUTPUT = ROOT / "segmentation" / "results" / "sikufenci_lda.jsonl"
DEFAULT_METADATA = ROOT / "segmentation" / "results" / "sikufenci_lda.metadata.json"
DEFAULT_AUDIT = ROOT / "segmentation" / "results" / "sikufenci_lda.audit.csv"
DEFAULT_PROTECTED = STANDARD_DIR / "protected_term_seed.tsv"
DEFAULT_ENTITIES = STANDARD_DIR / "named_entity_seed.tsv"
DEFAULT_NON_MERGE = STANDARD_DIR / "non_merge_rules.tsv"
RULE_VERSION = "1.0"
REQUIRED_TERMS = ("来仪", "折腰", "倾城", "难再逢", "良可惜", "千钟")
TRIE_RULES_KEY = "__rules__"
NUMERAL_CHARS = frozenset("〇零一二三四五六七八九十百千万亿两廿卅卌半数几")
PUNCTUATION = frozenset(
    "，。！？；：、,.!?;:（）()〔〕【】[]《》〈〉“”‘’\"'﹁﹂﹃﹄—…·"
)


@dataclass(frozen=True)
class MergeRule:
    """保存一个词形的合并优先级与审计信息。"""

    surface: str
    source: str
    decision: str
    scope: str
    priority: int


@dataclass(frozen=True)
class MatchedSpan:
    """保存原文上的已选择保护跨度。"""

    start: int
    end: int
    rule: MergeRule


@dataclass(frozen=True)
class SplitPattern:
    """保存禁合并表中带斜线示例的明确切分位置。"""

    surface: str
    offsets: tuple[int, ...]
    rule_id: str


@dataclass(frozen=True)
class PostprocessResult:
    """保存单首诗的后处理结果和边界变更。"""

    tokens: list[str]
    matched_spans: list[MatchedSpan]
    changed_terms: tuple[str, ...]
    added_boundaries: tuple[int, ...]
    removed_boundaries: tuple[int, ...]
    split_rule_hits: tuple[str, ...]


def is_han(char: str) -> bool:
    """判断字符是否属于常用或扩展汉字区。"""

    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x323AF
    )


def is_han_term(term: str) -> bool:
    """保护词必须全部由汉字组成。"""

    return bool(term) and all(is_han(char) for char in term)


def read_tsv(path: Path) -> list[dict[str, str]]:
    """读取 UTF-8 TSV，并拒绝空字段或残缺行。"""

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"规则表为空：{path}")
    for line_number, row in enumerate(rows, 2):
        if None in row or any(value is None or not value.strip() for value in row.values()):
            raise ValueError(f"规则表第 {line_number} 行存在空字段或多余字段：{path}")
    return rows


def load_merge_rules(protected_path: Path, entity_path: Path) -> list[MergeRule]:
    """载入保护词和实体词；条件项只保护字面边界，不进行语义归一化。"""

    rules: list[MergeRule] = []
    protected_rows = read_tsv(protected_path)
    if set(protected_rows[0]) != {"term", "decision", "category", "scope", "evidence", "note"}:
        raise ValueError(f"保护词表字段不符合规范：{protected_path}")
    for row in protected_rows:
        surface = row["term"].strip()
        if len(surface) < 2 or not is_han_term(surface):
            raise ValueError(f"保护词不是至少二字的纯汉字词形：{surface}")
        priority = 50 if row["decision"] == "强制合并" else 30
        rules.append(
            MergeRule(
                surface=surface,
                source="protected_term",
                decision=row["decision"],
                scope=row["scope"],
                priority=priority,
            )
        )

    entity_rows = read_tsv(entity_path)
    if set(entity_rows[0]) != {"surface", "canonical", "type", "scope", "ambiguity", "note"}:
        raise ValueError(f"实体词表字段不符合规范：{entity_path}")
    for row in entity_rows:
        surface = row["surface"].strip()
        if len(surface) < 2 or not is_han_term(surface):
            raise ValueError(f"实体词不是至少二字的纯汉字词形：{surface}")
        rules.append(
            MergeRule(
                surface=surface,
                source="named_entity",
                decision="保留实体边界",
                scope=row["scope"],
                priority=40,
            )
        )

    surfaces = [rule.surface for rule in rules]
    duplicates = sorted(term for term, count in Counter(surfaces).items() if count > 1)
    if duplicates:
        raise ValueError(f"保护词与实体词存在重复词形：{duplicates}")
    return rules


def load_split_patterns(path: Path) -> list[SplitPattern]:
    """只把禁合并表中明确写出斜线边界的例子转成确定性规则。"""

    rows = read_tsv(path)
    if set(rows[0]) != {
        "rule_id",
        "structure",
        "default_decision",
        "examples",
        "exceptions",
        "reason",
    }:
        raise ValueError(f"禁合并表字段不符合规范：{path}")
    patterns: list[SplitPattern] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for row in rows:
        if row["rule_id"] in {"R01", "R10", "R12"}:
            continue
        for example in row["examples"].split("；"):
            parts = [part.strip() for part in example.split("/")]
            if len(parts) < 2 or any(not part for part in parts):
                continue
            surface = "".join(parts)
            offsets: list[int] = []
            cursor = 0
            for part in parts[:-1]:
                cursor += len(part)
                offsets.append(cursor)
            key = (surface, tuple(offsets))
            if key in seen:
                continue
            seen.add(key)
            patterns.append(SplitPattern(surface, tuple(offsets), row["rule_id"]))
    return patterns


class LongestTermMatcher:
    """用字符 Trie 搜索词形，并按长度、优先级、起点选择非重叠跨度。"""

    def __init__(self, rules: Sequence[MergeRule]):
        self.trie: dict[str, object] = {}
        for rule in rules:
            node = self.trie
            for char in rule.surface:
                node = node.setdefault(char, {})  # type: ignore[assignment]
            node.setdefault(TRIE_RULES_KEY, []).append(rule)  # type: ignore[union-attr]

    def find(self, text: str) -> list[MatchedSpan]:
        """选择全文中互不重叠的最长词形。"""

        candidates: list[MatchedSpan] = []
        for start in range(len(text)):
            node: object = self.trie
            end = start
            while end < len(text) and isinstance(node, dict) and text[end] in node:
                node = node[text[end]]
                end += 1
                if isinstance(node, dict):
                    for rule in node.get(TRIE_RULES_KEY, []):
                        candidates.append(MatchedSpan(start, end, rule))
        candidates.sort(
            key=lambda span: (
                -(span.end - span.start),
                -span.rule.priority,
                span.start,
                span.rule.surface,
            )
        )
        occupied = [False] * len(text)
        selected: list[MatchedSpan] = []
        for span in candidates:
            if any(occupied[index] for index in range(span.start, span.end)):
                continue
            for index in range(span.start, span.end):
                occupied[index] = True
            selected.append(span)
        selected.sort(key=lambda span: (span.start, span.end))
        return selected


def validate_tokens(text: str, tokens: Sequence[str], poem_id: object, label: str) -> None:
    """验证 token 顺序拼接后严格还原原文。"""

    if "".join(tokens) != text:
        raise ValueError(f"{label} 无法还原诗歌 {poem_id}")
    if any(token == "" for token in tokens):
        raise ValueError(f"{label} 含空 token：诗歌 {poem_id}")


def token_boundaries(text: str, tokens: Sequence[str]) -> set[int]:
    """把 token 列表转换成含首尾的字符边界集合。"""

    validate_tokens(text, tokens, "未知", "输入")
    boundaries = {0}
    cursor = 0
    for token in tokens:
        cursor += len(token)
        boundaries.add(cursor)
    return boundaries


def hard_boundaries(text: str) -> set[int]:
    """生成标点、空白和联句编辑标记的硬边界。"""

    boundaries = {0, len(text)}
    for index, char in enumerate(text):
        if char in PUNCTUATION:
            boundaries.update((index, index + 1))
    for match in re.finditer(r"__+|\s+", text):
        boundaries.update((match.start(), match.end()))
    return boundaries


def iter_occurrences(text: str, surface: str) -> Iterable[int]:
    """返回允许重叠的字面匹配起点。"""

    cursor = 0
    while True:
        start = text.find(surface, cursor)
        if start < 0:
            return
        yield start
        cursor = start + 1


def split_numeric_prefixes(
    text: str,
    boundaries: set[int],
    protected_interiors: set[int],
) -> int:
    """把未受保护词覆盖的前置数词串与普通成分分开。"""

    added = 0
    ordered = sorted(boundaries)
    for start, end in zip(ordered, ordered[1:]):
        token = text[start:end]
        if len(token) < 2 or not all(is_han(char) for char in token):
            continue
        cursor = 0
        while cursor < len(token) and token[cursor] in NUMERAL_CHARS:
            cursor += 1
        position = start + cursor
        if 0 < cursor < len(token) and position not in protected_interiors and position not in boundaries:
            boundaries.add(position)
            added += 1
    return added


def postprocess_tokens(
    text: str,
    base_tokens: Sequence[str],
    matcher: LongestTermMatcher,
    split_patterns: Sequence[SplitPattern],
    poem_id: object = "未知",
) -> PostprocessResult:
    """按规范覆盖基础边界，并返回可审计的变更。"""

    validate_tokens(text, base_tokens, poem_id, "sikufenci")
    base_boundaries = token_boundaries(text, base_tokens)
    hard = hard_boundaries(text)
    boundaries = set(base_boundaries) | hard
    spans = matcher.find(text)
    protected_interiors: set[int] = set()
    changed_terms: set[str] = set()

    for span in spans:
        interiors = set(range(span.start + 1, span.end))
        protected_interiors.update(interiors)
        if interiors & base_boundaries or span.start not in base_boundaries or span.end not in base_boundaries:
            changed_terms.add(span.rule.surface)
        for position in interiors - hard:
            boundaries.discard(position)
        boundaries.update((span.start, span.end))

    split_rule_hits: list[str] = []
    for pattern in split_patterns:
        for start in iter_occurrences(text, pattern.surface):
            hit = False
            for offset in pattern.offsets:
                position = start + offset
                if position in protected_interiors or position in boundaries:
                    continue
                boundaries.add(position)
                hit = True
            if hit:
                split_rule_hits.append(pattern.rule_id)

    numeric_hits = split_numeric_prefixes(text, boundaries, protected_interiors)
    split_rule_hits.extend(["R05"] * numeric_hits)
    boundaries.update((0, len(text)))
    ordered = sorted(boundaries)
    tokens = [text[start:end] for start, end in zip(ordered, ordered[1:]) if start < end]
    validate_tokens(text, tokens, poem_id, "sikufenci_lda")
    return PostprocessResult(
        tokens=tokens,
        matched_spans=spans,
        changed_terms=tuple(sorted(changed_terms)),
        added_boundaries=tuple(sorted(boundaries - base_boundaries)),
        removed_boundaries=tuple(sorted(base_boundaries - boundaries)),
        split_rule_hits=tuple(split_rule_hits),
    )


def file_sha256(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """仓库内路径使用相对形式，外部测试路径保留绝对形式。"""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def summarize_text_mismatch(corpus_text: str, result_text: str, poem_id: object) -> dict[str, object]:
    """记录语料正文与基础分词字符序列的第一处差异，不静默改写任一来源。"""

    common = min(len(corpus_text), len(result_text))
    offset = next(
        (index for index in range(common) if corpus_text[index] != result_text[index]),
        common,
    )
    left = max(0, offset - 12)
    right_corpus = min(len(corpus_text), offset + 13)
    right_result = min(len(result_text), offset + 13)
    return {
        "id": poem_id,
        "offset": offset,
        "corpus_length": len(corpus_text),
        "source_result_length": len(result_text),
        "corpus_context": corpus_text[left:right_corpus],
        "source_result_context": result_text[left:right_result],
    }


def preview_tokens(tokens: Sequence[str], max_chars: int = 160) -> str:
    """生成审计表中的短分词预览。"""

    parts: list[str] = []
    size = 0
    for token in tokens:
        if size + len(token) > max_chars:
            parts.append("…")
            break
        parts.append(token)
        size += len(token)
    return " / ".join(parts)


def write_audit(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """写出带 BOM 的逐诗变更审计 CSV。"""

    fields = [
        "id",
        "poet",
        "title",
        "old_token_count",
        "new_token_count",
        "added_boundary_count",
        "removed_boundary_count",
        "added_boundaries",
        "removed_boundaries",
        "matched_terms",
        "changed_terms",
        "split_rule_hits",
        "old_preview",
        "new_preview",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ensure_new_outputs(paths: Sequence[Path], overwrite: bool) -> None:
    """默认拒绝覆盖已有派生结果。"""

    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"输出已存在；如需重建请显式使用 --overwrite：{existing}")


def atomic_replace(temp_path: Path, final_path: Path) -> None:
    """把本次生成的临时文件原子替换为正式输出。"""

    os.replace(temp_path, final_path)


def build(
    corpus_path: Path,
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    audit_path: Path,
    protected_path: Path,
    entity_path: Path,
    non_merge_path: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    """流式处理全语料并写出结果、元数据和审计表。"""

    outputs = (output_path, metadata_path, audit_path)
    ensure_new_outputs(outputs, overwrite)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    merge_rules = load_merge_rules(protected_path, entity_path)
    split_patterns = load_split_patterns(non_merge_path)
    matcher = LongestTermMatcher(merge_rules)
    output_temp = output_path.with_name(output_path.name + ".tmp")
    metadata_temp = metadata_path.with_name(metadata_path.name + ".tmp")
    audit_temp = audit_path.with_name(audit_path.name + ".tmp")
    for temp_path in (output_temp, metadata_temp, audit_temp):
        if temp_path.exists():
            temp_path.unlink()

    poem_count = 0
    changed_poem_count = 0
    old_token_count = 0
    new_token_count = 0
    added_boundary_count = 0
    removed_boundary_count = 0
    matched_term_counts: Counter[str] = Counter()
    changed_term_counts: Counter[str] = Counter()
    split_rule_counts: Counter[str] = Counter()
    corpus_text_mismatch_count = 0
    corpus_text_mismatches: list[dict[str, object]] = []
    required_occurrences: Counter[str] = Counter()
    required_before_tokens: Counter[str] = Counter()
    required_after_tokens: Counter[str] = Counter()
    audit_rows: list[dict[str, object]] = []

    try:
        with corpus_path.open("r", encoding="utf-8") as corpus_handle, input_path.open(
            "r", encoding="utf-8"
        ) as input_handle, output_temp.open("w", encoding="utf-8", newline="\n") as output_handle:
            pairs = zip_longest(corpus_handle, input_handle)
            for line_number, pair in enumerate(pairs, 1):
                corpus_line, input_line = pair
                if corpus_line is None or input_line is None:
                    raise ValueError("统一语料与 sikufenci 结果行数不一致")
                poem = json.loads(corpus_line)
                record = json.loads(input_line)
                if poem["id"] != record["id"]:
                    raise ValueError(f"第 {line_number} 行 ID 未对齐")
                base_tokens = [str(token) for token in record["tokens"]]
                corpus_text = str(poem["text"])
                text = "".join(base_tokens)
                if corpus_text != text:
                    corpus_text_mismatch_count += 1
                    if len(corpus_text_mismatches) < 100:
                        corpus_text_mismatches.append(
                            summarize_text_mismatch(corpus_text, text, poem["id"])
                        )
                result = postprocess_tokens(text, base_tokens, matcher, split_patterns, poem["id"])
                output_handle.write(
                    json.dumps(
                        {"id": poem["id"], "tokens": result.tokens},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

                poem_count += 1
                old_token_count += len(base_tokens)
                new_token_count += len(result.tokens)
                added_boundary_count += len(result.added_boundaries)
                removed_boundary_count += len(result.removed_boundaries)
                matched_term_counts.update(span.rule.surface for span in result.matched_spans)
                changed_term_counts.update(result.changed_terms)
                split_rule_counts.update(result.split_rule_hits)
                for term in REQUIRED_TERMS:
                    required_occurrences[term] += text.count(term)
                    required_before_tokens[term] += base_tokens.count(term)
                    required_after_tokens[term] += result.tokens.count(term)

                changed = bool(result.added_boundaries or result.removed_boundaries)
                if changed:
                    changed_poem_count += 1
                    audit_rows.append(
                        {
                            "id": poem["id"],
                            "poet": poem.get("poet", ""),
                            "title": poem.get("title", ""),
                            "old_token_count": len(base_tokens),
                            "new_token_count": len(result.tokens),
                            "added_boundary_count": len(result.added_boundaries),
                            "removed_boundary_count": len(result.removed_boundaries),
                            "added_boundaries": "|".join(map(str, result.added_boundaries)),
                            "removed_boundaries": "|".join(map(str, result.removed_boundaries)),
                            "matched_terms": "|".join(
                                span.rule.surface for span in result.matched_spans
                            ),
                            "changed_terms": "|".join(result.changed_terms),
                            "split_rule_hits": "|".join(result.split_rule_hits),
                            "old_preview": preview_tokens(base_tokens),
                            "new_preview": preview_tokens(result.tokens),
                        }
                    )

        required_validation = {}
        for term in REQUIRED_TERMS:
            occurrences = required_occurrences[term]
            after = required_after_tokens[term]
            required_validation[term] = {
                "text_occurrences": occurrences,
                "exact_tokens_before": required_before_tokens[term],
                "exact_tokens_after": after,
                "all_occurrences_protected": after == occurrences,
            }
            if after != occurrences:
                raise ValueError(f"指定词“{term}”未全部保护：正文 {occurrences} 次，结果 {after} 次")

        write_audit(audit_temp, audit_rows)
        atomic_replace(output_temp, output_path)
        atomic_replace(audit_temp, audit_path)
        metadata = {
            "method": "sikufenci_lda",
            "rule_version": RULE_VERSION,
            "source_method": "sikufenci",
            "text_authority": "source_result_token_concatenation",
            "source_result": display_path(input_path),
            "source_corpus": display_path(corpus_path),
            "output_result": display_path(output_path),
            "audit_file": display_path(audit_path),
            "rule_files": {
                "protected_terms": display_path(protected_path),
                "named_entities": display_path(entity_path),
                "non_merge_rules": display_path(non_merge_path),
            },
            "rule_file_sha256": {
                "protected_terms": file_sha256(protected_path),
                "named_entities": file_sha256(entity_path),
                "non_merge_rules": file_sha256(non_merge_path),
            },
            "source_sha256": file_sha256(input_path),
            "output_sha256": file_sha256(output_path),
            "poem_count": poem_count,
            "changed_poem_count": changed_poem_count,
            "unchanged_poem_count": poem_count - changed_poem_count,
            "old_token_count": old_token_count,
            "new_token_count": new_token_count,
            "net_token_change": new_token_count - old_token_count,
            "added_boundary_count": added_boundary_count,
            "removed_boundary_count": removed_boundary_count,
            "merge_rule_count": len(merge_rules),
            "explicit_split_pattern_count": len(split_patterns),
            "corpus_text_mismatch_count": corpus_text_mismatch_count,
            "corpus_text_mismatches": corpus_text_mismatches,
            "conditional_scope_note": "条件词只保护原文字面边界，不赋予规范名或典故语义。",
            "non_merge_implementation_note": "需要词性或语境判断的禁合并类别只执行 TSV 中明确写出的斜线示例；R05 另对未受保护的前置数词串执行通用切分，不以未经审核的启发式规则扩张。",
            "matched_term_counts": dict(matched_term_counts.most_common()),
            "changed_term_counts": dict(changed_term_counts.most_common()),
            "split_rule_counts": dict(split_rule_counts.most_common()),
            "required_term_validation": required_validation,
            "validation": {
                "line_count_and_id_alignment": True,
                "all_outputs_reconstruct_source_result_text": True,
                "all_required_terms_protected": True,
            },
        }
        metadata_temp.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        atomic_replace(metadata_temp, metadata_path)
        return metadata
    except Exception:
        for temp_path in (output_temp, metadata_temp, audit_temp):
            if temp_path.exists():
                temp_path.unlink()
        raise


def parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    cli.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    cli.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    cli.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    cli.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    cli.add_argument("--protected", type=Path, default=DEFAULT_PROTECTED)
    cli.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    cli.add_argument("--non-merge", type=Path, default=DEFAULT_NON_MERGE)
    cli.add_argument("--overwrite", action="store_true")
    return cli


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""

    args = parser().parse_args(argv)
    metadata = build(
        corpus_path=args.corpus,
        input_path=args.input,
        output_path=args.output,
        metadata_path=args.metadata,
        audit_path=args.audit,
        protected_path=args.protected,
        entity_path=args.entities,
        non_merge_path=args.non_merge,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
