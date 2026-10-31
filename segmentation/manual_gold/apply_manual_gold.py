"""把人工金标准词界覆盖到 sikufenci 基础结果，并保留基础结果的原字符。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
MANUAL_DIR = Path(__file__).resolve().parent
DEFAULT_WORKBOOK = ROOT / "tang_poetry 人工金标准.xlsx"
DEFAULT_ANNOTATIONS = MANUAL_DIR / "manual_gold.annotations.jsonl"
DEFAULT_EXTRACTION_METADATA = MANUAL_DIR / "manual_gold.annotations.metadata.json"
DEFAULT_CORPUS = ROOT / "segmentation" / "data" / "poems.jsonl"
DEFAULT_INPUT = ROOT / "segmentation" / "results" / "sikufenci.jsonl"
DEFAULT_OUTPUT = ROOT / "segmentation" / "results" / "sikufenci_manual_gold.jsonl"
DEFAULT_METADATA = ROOT / "segmentation" / "results" / "sikufenci_manual_gold.metadata.json"
DEFAULT_AUDIT = ROOT / "segmentation" / "results" / "sikufenci_manual_gold.audit.csv"
EXPECTED_HEADER = ["诗人", "篇名", "分词"]
MIN_VARIANT_SIMILARITY = 0.95


@dataclass(frozen=True)
class ManualAnnotation:
    """保存一行人工分词及其清洗后的词元。"""

    excel_row: int
    poet: str
    title: str
    tokens: tuple[str, ...]
    truncated: bool
    whitespace_removed_count: int

    @property
    def text(self) -> str:
        """返回去除分隔符、省略号与格式空白后的标注文本。"""

        return "".join(self.tokens)


@dataclass(frozen=True)
class Overlay:
    """保存一首诗的人工词界覆盖结果。"""

    poem_id: int
    annotation: ManualAnnotation
    tokens: tuple[str, ...]
    match_mode: str
    scope: str
    similarity: float
    added_boundaries: tuple[int, ...]
    removed_boundaries: tuple[int, ...]
    variant_ops: tuple[str, ...]


def file_sha256(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """仓库内路径使用相对形式，仓库外路径保留绝对形式。"""

    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """读取非空 JSONL，并在错误中保留行号。"""

    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"JSONL 第 {line_number} 行为空：{path}")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL 第 {line_number} 行不是对象：{path}")
            records.append(record)
    if not records:
        raise ValueError(f"JSONL 为空：{path}")
    return records


def parse_annotation(record: dict[str, object]) -> ManualAnnotation:
    """解析人工行；只清除格式空白，不改写任何其他字符。"""

    required = {"excel_row", "poet", "title", "segmentation"}
    if set(record) != required:
        raise ValueError(f"人工标注字段不符合预期：{sorted(record)}")
    excel_row = int(record["excel_row"])
    poet = str(record["poet"]).strip()
    title = str(record["title"]).strip()
    segmentation = str(record["segmentation"])
    if excel_row < 2 or not poet or not title or not segmentation.strip():
        raise ValueError(f"人工表第 {excel_row} 行存在空字段")

    tokens: list[str] = []
    whitespace_removed_count = 0
    for part in segmentation.split("/"):
        stripped = part.strip()
        cleaned = "".join(stripped.split())
        whitespace_removed_count += len(stripped) - len(cleaned)
        if not cleaned:
            raise ValueError(f"人工表第 {excel_row} 行存在空词元")
        tokens.append(cleaned)

    truncated = tokens[-1] == "…"
    if truncated:
        tokens.pop()
    if not tokens or any("…" in token for token in tokens):
        raise ValueError(f"人工表第 {excel_row} 行的省略号位置不合法")
    return ManualAnnotation(
        excel_row=excel_row,
        poet=poet,
        title=title,
        tokens=tuple(tokens),
        truncated=truncated,
        whitespace_removed_count=whitespace_removed_count,
    )


def load_annotations(path: Path) -> list[ManualAnnotation]:
    """载入人工行，并拒绝重复的 Excel 行号。"""

    annotations = [parse_annotation(record) for record in read_jsonl(path)]
    row_counts = Counter(annotation.excel_row for annotation in annotations)
    duplicates = sorted(row for row, count in row_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"人工标注存在重复 Excel 行号：{duplicates}")
    return annotations


def validate_extraction(
    workbook_path: Path,
    annotations_path: Path,
    metadata_path: Path,
    annotation_count: int,
) -> dict[str, object]:
    """核验机器可读标注确由当前工作簿抽取。"""

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("header") != EXPECTED_HEADER:
        raise ValueError("人工金标准抽取元数据的字段不符合预期")
    if int(metadata.get("record_count", -1)) != annotation_count:
        raise ValueError("人工标注行数与抽取元数据不一致")
    if metadata.get("source_workbook_sha256") != file_sha256(workbook_path):
        raise ValueError("人工金标准工作簿已变化，请重新抽取标注")
    if metadata.get("annotation_output_sha256") != file_sha256(annotations_path):
        raise ValueError("机器可读人工标注已变化，请重新抽取或修复元数据")
    return metadata


def token_boundaries(tokens: Sequence[str]) -> set[int]:
    """返回字符序列中的内部词界。"""

    result: set[int] = set()
    offset = 0
    for token in tokens[:-1]:
        offset += len(token)
        result.add(offset)
    return result


def tokens_from_boundaries(text: str, boundaries: Iterable[int]) -> tuple[str, ...]:
    """按严格递增的内部词界切分字符序列。"""

    positions = [0, *sorted(set(boundaries)), len(text)]
    if any(position <= 0 or position >= len(text) for position in positions[1:-1]):
        raise ValueError("词界超出正文范围")
    return tuple(
        text[start:end] for start, end in zip(positions, positions[1:]) if start < end
    )


def project_variant_boundaries(
    manual_text: str,
    source_text: str,
    manual_boundaries: set[int],
    excel_row: int,
) -> tuple[set[int], float, tuple[str, ...]]:
    """把少量异文字面上的人工词界安全投射到基础结果字符序列。"""

    matcher = SequenceMatcher(None, manual_text, source_text, autojunk=False)
    similarity = matcher.ratio()
    if similarity < MIN_VARIANT_SIMILARITY:
        raise ValueError(
            f"人工表第 {excel_row} 行与基础文本相似度过低：{similarity:.4f}"
        )
    opcodes = matcher.get_opcodes()
    projected: set[int] = set()
    for boundary in sorted(manual_boundaries):
        candidates: set[int] = set()
        unsafe = False
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal" and i1 <= boundary <= i2:
                candidates.add(j1 + boundary - i1)
            elif boundary == i1:
                candidates.add(j1)
            elif boundary == i2:
                candidates.add(j2)
            elif i1 < boundary < i2:
                unsafe = True
        if unsafe or len(candidates) != 1:
            raise ValueError(
                f"人工表第 {excel_row} 行的词界 {boundary} 落在无法安全投射的异文内部"
            )
        projected.add(candidates.pop())

    variant_ops = tuple(
        f"{tag}:人工[{i1}:{i2}]={manual_text[i1:i2]!r}->基础[{j1}:{j2}]={source_text[j1:j2]!r}"
        for tag, i1, i2, j1, j2 in opcodes
        if tag != "equal"
    )
    return projected, similarity, variant_ops


def annotation_matches_source(annotation: ManualAnnotation, source_text: str) -> bool:
    """判断人工行是完整匹配还是省略号前缀匹配。"""

    return (
        source_text.startswith(annotation.text)
        if annotation.truncated
        else source_text == annotation.text
    )


def make_overlay(
    poem_id: int,
    annotation: ManualAnnotation,
    base_tokens: Sequence[str],
) -> Overlay:
    """生成单首诗的人工词界覆盖，不改写基础文本字符。"""

    source_text = "".join(base_tokens)
    base_boundaries = token_boundaries(base_tokens)
    manual_boundaries = token_boundaries(annotation.tokens)
    variant_ops: tuple[str, ...] = ()

    if annotation_matches_source(annotation, source_text):
        similarity = 1.0
        if annotation.truncated:
            scope = "prefix"
            match_mode = "exact_prefix"
            annotated_end = len(annotation.text)
            if annotated_end >= len(source_text):
                raise ValueError(f"人工表第 {annotation.excel_row} 行省略号未省略正文")
            if annotated_end not in base_boundaries:
                raise ValueError(
                    f"人工表第 {annotation.excel_row} 行的前缀终点不是基础词界，无法确定续接方式"
                )
            new_boundaries = manual_boundaries | {
                boundary for boundary in base_boundaries if boundary >= annotated_end
            }
        else:
            scope = "full"
            match_mode = "exact_full"
            new_boundaries = manual_boundaries
    else:
        if annotation.truncated:
            raise ValueError(
                f"人工表第 {annotation.excel_row} 行同时含省略号和文本异文，拒绝自动投射"
            )
        scope = "full"
        match_mode = "variant_boundary_projection"
        new_boundaries, similarity, variant_ops = project_variant_boundaries(
            annotation.text,
            source_text,
            manual_boundaries,
            annotation.excel_row,
        )

    output_tokens = tokens_from_boundaries(source_text, new_boundaries)
    if "".join(output_tokens) != source_text:
        raise ValueError(f"诗歌 ID {poem_id} 的人工覆盖结果无法还原基础文本")
    return Overlay(
        poem_id=poem_id,
        annotation=annotation,
        tokens=output_tokens,
        match_mode=match_mode,
        scope=scope,
        similarity=similarity,
        added_boundaries=tuple(sorted(new_boundaries - base_boundaries)),
        removed_boundaries=tuple(sorted(base_boundaries - new_boundaries)),
        variant_ops=variant_ops,
    )


def align_annotations(
    annotations: Sequence[ManualAnnotation],
    poems: Sequence[dict[str, object]],
    source_by_id: dict[int, dict[str, object]],
) -> dict[int, Overlay]:
    """以诗人和篇名定位；同名诗必须再由文本唯一消歧。"""

    poems_by_key: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for poem in poems:
        poems_by_key[(str(poem.get("poet", "")), str(poem.get("title", "")))].append(poem)

    overlays: dict[int, Overlay] = {}
    for annotation in annotations:
        candidates = poems_by_key.get((annotation.poet, annotation.title), [])
        if not candidates:
            raise ValueError(
                f"人工表第 {annotation.excel_row} 行找不到诗人和篇名对应项："
                f"{annotation.poet}《{annotation.title}》"
            )
        exact_candidates = []
        for poem in candidates:
            poem_id = int(poem["id"])
            source_text = "".join(str(token) for token in source_by_id[poem_id]["tokens"])
            if annotation_matches_source(annotation, source_text):
                exact_candidates.append(poem)
        if len(exact_candidates) == 1:
            poem = exact_candidates[0]
        elif len(exact_candidates) > 1:
            raise ValueError(f"人工表第 {annotation.excel_row} 行文本仍对应多首同名诗")
        elif len(candidates) == 1:
            poem = candidates[0]
        else:
            candidate_ids = [int(candidate["id"]) for candidate in candidates]
            raise ValueError(
                f"人工表第 {annotation.excel_row} 行无法在同名诗中消歧：{candidate_ids}"
            )

        poem_id = int(poem["id"])
        if poem_id in overlays:
            raise ValueError(f"诗歌 ID {poem_id} 被多行人工标注重复覆盖")
        base_tokens = [str(token) for token in source_by_id[poem_id]["tokens"]]
        overlays[poem_id] = make_overlay(poem_id, annotation, base_tokens)
    return overlays


def summarize_text_mismatch(
    corpus_text: str, source_text: str, poem_id: int
) -> dict[str, object]:
    """记录统一语料与基础分词字符序列的第一处差异。"""

    common = min(len(corpus_text), len(source_text))
    offset = next(
        (index for index in range(common) if corpus_text[index] != source_text[index]),
        common,
    )
    left = max(0, offset - 12)
    return {
        "id": poem_id,
        "offset": offset,
        "corpus_length": len(corpus_text),
        "source_result_length": len(source_text),
        "corpus_context": corpus_text[left : offset + 13],
        "source_result_context": source_text[left : offset + 13],
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


def write_audit(
    path: Path,
    overlays: Sequence[Overlay],
    poems_by_id: dict[int, dict[str, object]],
    source_by_id: dict[int, dict[str, object]],
) -> None:
    """写出全部人工标注行的逐条审计 CSV。"""

    fields = [
        "excel_row",
        "id",
        "poet",
        "title",
        "scope",
        "match_mode",
        "similarity",
        "annotated_char_count",
        "whitespace_removed_count",
        "old_token_count",
        "new_token_count",
        "added_boundary_count",
        "removed_boundary_count",
        "added_boundaries",
        "removed_boundaries",
        "variant_ops",
        "old_preview",
        "new_preview",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for overlay in sorted(overlays, key=lambda item: item.annotation.excel_row):
            poem = poems_by_id[overlay.poem_id]
            base_tokens = [
                str(token) for token in source_by_id[overlay.poem_id]["tokens"]
            ]
            writer.writerow(
                {
                    "excel_row": overlay.annotation.excel_row,
                    "id": overlay.poem_id,
                    "poet": poem.get("poet", ""),
                    "title": poem.get("title", ""),
                    "scope": overlay.scope,
                    "match_mode": overlay.match_mode,
                    "similarity": f"{overlay.similarity:.6f}",
                    "annotated_char_count": len(overlay.annotation.text),
                    "whitespace_removed_count": overlay.annotation.whitespace_removed_count,
                    "old_token_count": len(base_tokens),
                    "new_token_count": len(overlay.tokens),
                    "added_boundary_count": len(overlay.added_boundaries),
                    "removed_boundary_count": len(overlay.removed_boundaries),
                    "added_boundaries": "|".join(map(str, overlay.added_boundaries)),
                    "removed_boundaries": "|".join(map(str, overlay.removed_boundaries)),
                    "variant_ops": "|".join(overlay.variant_ops),
                    "old_preview": preview_tokens(base_tokens),
                    "new_preview": preview_tokens(overlay.tokens),
                }
            )


def ensure_new_outputs(paths: Sequence[Path], overwrite: bool) -> None:
    """默认拒绝覆盖已有派生结果。"""

    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"输出已存在；如需重建请显式使用 --overwrite：{existing}")


def atomic_replace(temp_path: Path, final_path: Path) -> None:
    """原子替换本次生成的正式输出。"""

    os.replace(temp_path, final_path)


def build(
    workbook_path: Path,
    annotations_path: Path,
    extraction_metadata_path: Path,
    corpus_path: Path,
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    audit_path: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    """应用全部可安全对齐的人工词界，并写出结果、元数据和审计。"""

    outputs = (output_path, metadata_path, audit_path)
    ensure_new_outputs(outputs, overwrite)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(annotations_path)
    extraction_metadata = validate_extraction(
        workbook_path,
        annotations_path,
        extraction_metadata_path,
        len(annotations),
    )
    poems = read_jsonl(corpus_path)
    source_records = read_jsonl(input_path)
    if len(poems) != len(source_records):
        raise ValueError("统一语料与 sikufenci 结果行数不一致")

    poems_by_id: dict[int, dict[str, object]] = {}
    source_by_id: dict[int, dict[str, object]] = {}
    corpus_text_mismatch_count = 0
    corpus_text_mismatches: list[dict[str, object]] = []
    for line_number, (poem, source_record) in enumerate(zip(poems, source_records), 1):
        poem_id = int(poem["id"])
        source_id = int(source_record["id"])
        if poem_id != source_id:
            raise ValueError(f"第 {line_number} 行的统一语料与基础结果 ID 未对齐")
        if poem_id in poems_by_id or source_id in source_by_id:
            raise ValueError(f"诗歌 ID 重复：{poem_id}")
        base_tokens = [str(token) for token in source_record["tokens"]]
        if not base_tokens and str(poem.get("text", "")):
            raise ValueError(f"诗歌 ID {poem_id} 的基础 tokens 为空")
        source_text = "".join(base_tokens)
        corpus_text = str(poem.get("text", ""))
        if source_text != corpus_text:
            corpus_text_mismatch_count += 1
            if len(corpus_text_mismatches) < 100:
                corpus_text_mismatches.append(
                    summarize_text_mismatch(corpus_text, source_text, poem_id)
                )
        poems_by_id[poem_id] = poem
        source_by_id[source_id] = source_record

    overlays = align_annotations(annotations, poems, source_by_id)
    output_temp = output_path.with_name(output_path.name + ".tmp")
    metadata_temp = metadata_path.with_name(metadata_path.name + ".tmp")
    audit_temp = audit_path.with_name(audit_path.name + ".tmp")
    for temp_path in (output_temp, metadata_temp, audit_temp):
        if temp_path.exists():
            temp_path.unlink()

    old_token_count = 0
    new_token_count = 0
    try:
        with output_temp.open("w", encoding="utf-8", newline="\n") as handle:
            for source_record in source_records:
                poem_id = int(source_record["id"])
                base_tokens = [str(token) for token in source_record["tokens"]]
                output_tokens = list(overlays[poem_id].tokens) if poem_id in overlays else base_tokens
                if "".join(output_tokens) != "".join(base_tokens):
                    raise ValueError(f"诗歌 ID {poem_id} 的输出改写了基础文本字符")
                handle.write(
                    json.dumps(
                        {"id": poem_id, "tokens": output_tokens},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                old_token_count += len(base_tokens)
                new_token_count += len(output_tokens)

        overlay_values = list(overlays.values())
        write_audit(audit_temp, overlay_values, poems_by_id, source_by_id)
        atomic_replace(output_temp, output_path)
        atomic_replace(audit_temp, audit_path)

        changed_overlays = [
            overlay
            for overlay in overlay_values
            if overlay.added_boundaries or overlay.removed_boundaries
        ]
        variant_overlays = [
            overlay
            for overlay in overlay_values
            if overlay.match_mode == "variant_boundary_projection"
        ]
        metadata = {
            "method": "sikufenci_manual_gold_overlay",
            "source_method": "sikufenci",
            "text_authority": "source_result_token_concatenation",
            "source_result": display_path(input_path),
            "source_corpus": display_path(corpus_path),
            "manual_workbook": display_path(workbook_path),
            "manual_annotations": display_path(annotations_path),
            "manual_extraction_metadata": display_path(extraction_metadata_path),
            "output_result": display_path(output_path),
            "audit_file": display_path(audit_path),
            "source_sha256": file_sha256(input_path),
            "source_corpus_sha256": file_sha256(corpus_path),
            "manual_workbook_sha256": file_sha256(workbook_path),
            "manual_annotations_sha256": file_sha256(annotations_path),
            "output_sha256": file_sha256(output_path),
            "manual_sheet": extraction_metadata["sheet"],
            "manual_range": extraction_metadata["range"],
            "manual_header": extraction_metadata["header"],
            "poem_count": len(source_records),
            "annotated_poem_count": len(overlay_values),
            "unannotated_poem_count": len(source_records) - len(overlay_values),
            "full_annotation_count": sum(
                overlay.scope == "full" for overlay in overlay_values
            ),
            "prefix_annotation_count": sum(
                overlay.scope == "prefix" for overlay in overlay_values
            ),
            "exact_text_annotation_count": sum(
                overlay.match_mode != "variant_boundary_projection"
                for overlay in overlay_values
            ),
            "variant_boundary_projection_count": len(variant_overlays),
            "whitespace_normalized_annotation_count": sum(
                overlay.annotation.whitespace_removed_count > 0
                for overlay in overlay_values
            ),
            "annotated_character_count": sum(
                len(overlay.annotation.text) for overlay in overlay_values
            ),
            "changed_poem_count": len(changed_overlays),
            "unchanged_annotated_poem_count": len(overlay_values) - len(changed_overlays),
            "old_token_count": old_token_count,
            "new_token_count": new_token_count,
            "net_token_change": new_token_count - old_token_count,
            "added_boundary_count": sum(
                len(overlay.added_boundaries) for overlay in overlay_values
            ),
            "removed_boundary_count": sum(
                len(overlay.removed_boundaries) for overlay in overlay_values
            ),
            "corpus_text_mismatch_count": corpus_text_mismatch_count,
            "corpus_text_mismatches": corpus_text_mismatches,
            "variant_annotations": [
                {
                    "excel_row": overlay.annotation.excel_row,
                    "id": overlay.poem_id,
                    "poet": overlay.annotation.poet,
                    "title": overlay.annotation.title,
                    "similarity": round(overlay.similarity, 6),
                    "operations": list(overlay.variant_ops),
                }
                for overlay in sorted(
                    variant_overlays, key=lambda item: item.annotation.excel_row
                )
            ],
            "scope_note": (
                "人工金标准只覆盖 485 首，其中省略号行只覆盖已标前缀；"
                "未标诗与前缀之外的正文保留原 sikufenci 词界。"
            ),
            "variant_note": (
                "人工表中的异文字面只用于投射词界；输出始终保留基础 sikufenci 的字符序列。"
            ),
            "validation": {
                "extraction_hashes_match_workbook": True,
                "all_annotations_uniquely_aligned": True,
                "all_outputs_reconstruct_source_result_text": True,
                "line_count_and_id_alignment": True,
            },
        }
        metadata_temp.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
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
    cli.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    cli.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    cli.add_argument(
        "--extraction-metadata", type=Path, default=DEFAULT_EXTRACTION_METADATA
    )
    cli.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    cli.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    cli.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    cli.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    cli.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    cli.add_argument("--overwrite", action="store_true")
    return cli


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""

    args = parser().parse_args(argv)
    metadata = build(
        workbook_path=args.workbook,
        annotations_path=args.annotations,
        extraction_metadata_path=args.extraction_metadata,
        corpus_path=args.corpus,
        input_path=args.input,
        output_path=args.output,
        metadata_path=args.metadata,
        audit_path=args.audit,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
