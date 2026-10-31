#!/usr/bin/env python3
"""《全唐诗》四路分词与无金标准比较流水线。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import itertools
import json
import math
import re
import sys
import time
import types
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / ".segmentation-runtime"
DEFAULT_SQL = ROOT / "tang_poetry.sql"
DEFAULT_CORPUS = ROOT / "segmentation" / "data" / "poems.jsonl"
DEFAULT_RESULTS = ROOT / "segmentation" / "results"
SOURCES = {
    "sikufenci": "https://github.com/hsc748NLP/sikufenci",
    "bert_ancient": "https://github.com/Jihuai-wpy/bert-ancient-chinese",
    "jiayan": "https://github.com/jiaeyan/Jiayan",
    "jieba": "https://github.com/fxsjy/jieba",
}
METHODS = tuple(SOURCES)


def activate_runtime() -> None:
    """优先使用项目内安装的第三方包。"""
    if RUNTIME_DIR.exists():
        runtime = str(RUNTIME_DIR)
        if runtime not in sys.path:
            sys.path.insert(0, runtime)


def mysql_unescape_char(char: str) -> str:
    """还原 MySQL 文本导出中常见的反斜线转义。"""
    return {
        "0": "\0",
        "b": "\b",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "Z": "\x1a",
    }.get(char, char)


def iter_mysql_values(values: str) -> Iterator[list[object]]:
    """逐项解析一条 INSERT ... VALUES 后面的元组。"""
    index = 0
    size = len(values)
    while index < size:
        while index < size and (values[index].isspace() or values[index] in ",;"):
            index += 1
        if index >= size:
            return
        if values[index] != "(":
            raise ValueError(f"VALUES 在位置 {index} 缺少左括号")
        index += 1
        row: list[object] = []
        while True:
            while index < size and values[index].isspace():
                index += 1
            if index >= size:
                raise ValueError("VALUES 元组意外结束")
            if values[index] == "'":
                index += 1
                buffer: list[str] = []
                while index < size:
                    char = values[index]
                    if char == "\\":
                        index += 1
                        if index >= size:
                            raise ValueError("字符串以孤立反斜线结束")
                        buffer.append(mysql_unescape_char(values[index]))
                        index += 1
                    elif char == "'":
                        if index + 1 < size and values[index + 1] == "'":
                            buffer.append("'")
                            index += 2
                        else:
                            index += 1
                            break
                    else:
                        buffer.append(char)
                        index += 1
                else:
                    raise ValueError("字符串缺少右引号")
                value: object = "".join(buffer)
            else:
                start = index
                while index < size and values[index] not in ",)":
                    index += 1
                raw = values[start:index].strip()
                if raw.upper() == "NULL":
                    value = None
                elif re.fullmatch(r"-?\d+", raw):
                    value = int(raw)
                else:
                    value = raw
            row.append(value)
            while index < size and values[index].isspace():
                index += 1
            if index >= size:
                raise ValueError("VALUES 元组缺少右括号")
            if values[index] == ",":
                index += 1
                continue
            if values[index] == ")":
                index += 1
                yield row
                break
            raise ValueError(f"VALUES 在位置 {index} 出现未知分隔符")


def load_sql_dump(sql_path: Path) -> list[dict[str, object]]:
    """从 SQL 导出中读取诗人和诗歌，保持原始记录顺序。"""
    poets: dict[int, str] = {}
    poems: list[dict[str, object]] = []
    prefixes = {
        "poets": "INSERT INTO `poets` VALUES ",
        "poetries": "INSERT INTO `poetries` VALUES ",
    }
    with sql_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.startswith(prefixes["poets"]):
                values = line[len(prefixes["poets"]) :]
                for row in iter_mysql_values(values):
                    if len(row) != 4:
                        raise ValueError(f"第 {line_number} 行 poets 字段数不是 4")
                    poets[int(row[0])] = str(row[1])
            elif line.startswith(prefixes["poetries"]):
                values = line[len(prefixes["poetries"]) :]
                for row in iter_mysql_values(values):
                    if len(row) != 6:
                        raise ValueError(f"第 {line_number} 行 poetries 字段数不是 6")
                    poems.append(
                        {
                            "id": int(row[0]),
                            "poet_id": int(row[1]),
                            "text": str(row[2]),
                            "title": str(row[3]),
                        }
                    )
    if not poets or not poems:
        raise ValueError("未在 SQL 中解析到 poets 或 poetries 数据")
    missing = sorted({int(poem["poet_id"]) for poem in poems} - poets.keys())
    if missing:
        raise ValueError(f"有 {len(missing)} 个诗人 ID 无法关联")
    for poem in poems:
        poem["poet"] = poets[int(poem["poet_id"])]
    ids = [int(poem["id"]) for poem in poems]
    if len(ids) != len(set(ids)):
        raise ValueError("诗歌 ID 存在重复")
    return poems


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    """以 UTF-8 原子写入 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def export_corpus(sql_path: Path, corpus_path: Path) -> list[dict[str, object]]:
    """导出统一语料并验证总数。"""
    poems = load_sql_dump(sql_path)
    write_jsonl(corpus_path, poems)
    metadata = {
        "source": str(sql_path.resolve()),
        "record_count": len(poems),
        "first_id": poems[0]["id"],
        "last_id": poems[-1]["id"],
        "sha256": sha256_file(sql_path),
    }
    write_json(corpus_path.with_suffix(".metadata.json"), metadata)
    return poems


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, object]]:
    """读取 JSONL，可用于冒烟测试时限制条数。"""
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
    return records


def write_json(path: Path, payload: object) -> None:
    """以 UTF-8 写入缩进 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_tokens(text: str, tokens: Sequence[str], poem_id: int, method: str) -> None:
    """确保分词结果可以无损还原原文。"""
    reconstructed = "".join(tokens)
    if reconstructed != text:
        raise ValueError(
            f"{method} 无法还原诗歌 {poem_id}：原文 {len(text)} 字，结果 {len(reconstructed)} 字"
        )


def save_method_metadata(
    results_dir: Path,
    method: str,
    record_count: int,
    elapsed_seconds: float,
    implementation: str,
    package_version: str,
    limitations: list[str],
    extra: dict[str, object] | None = None,
) -> None:
    """保存每一路结果的运行口径。"""
    payload: dict[str, object] = {
        "method": method,
        "source": SOURCES[method],
        "record_count": record_count,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "implementation": implementation,
        "package_version": package_version,
        "limitations": limitations,
    }
    if extra:
        payload.update(extra)
    write_json(results_dir / f"{method}.metadata.json", payload)


def run_jieba(poems: Sequence[dict[str, object]], results_dir: Path) -> None:
    """运行 jieba 精确模式。"""
    activate_runtime()
    import jieba

    started = time.perf_counter()
    tokenizer = jieba.Tokenizer()
    tokenizer.initialize()

    def records() -> Iterator[dict[str, object]]:
        for poem in poems:
            text = str(poem["text"])
            tokens = list(tokenizer.cut(text, cut_all=False, HMM=True))
            validate_tokens(text, tokens, int(poem["id"]), "jieba")
            yield {"id": poem["id"], "tokens": tokens}

    write_jsonl(results_dir / "jieba.jsonl", records())
    save_method_metadata(
        results_dir,
        "jieba",
        len(poems),
        time.perf_counter() - started,
        "jieba.Tokenizer.cut(cut_all=False, HMM=True)，即精确模式",
        getattr(jieba, "__version__", "0.42.1"),
        ["面向通用现代汉语训练，并非唐诗或古汉语专用模型。"],
    )


def import_jiayan_word_tokenizer():
    """在无 KenLM 时加载 Jiayan 自带的词典路径分词器。"""
    activate_runtime()
    if "kenlm" not in sys.modules:
        sys.modules["kenlm"] = types.ModuleType("kenlm")
    from jiayan import WordNgramTokenizer

    return WordNgramTokenizer


def run_jiayan(poems: Sequence[dict[str, object]], results_dir: Path) -> None:
    """运行 Jiayan 官方最大概率路径分词器。"""
    started = time.perf_counter()
    tokenizer_class = import_jiayan_word_tokenizer()
    tokenizer = tokenizer_class()

    def records() -> Iterator[dict[str, object]]:
        for poem in poems:
            text = str(poem["text"])
            tokens = list(tokenizer.tokenize(text))
            validate_tokens(text, tokens, int(poem["id"]), "jiayan")
            yield {"id": poem["id"], "tokens": tokens}

    write_jsonl(results_dir / "jiayan.jsonl", records())
    save_method_metadata(
        results_dir,
        "jiayan",
        len(poems),
        time.perf_counter() - started,
        "jiayan.WordNgramTokenizer，词典 DAG + 最大概率路径",
        "0.0.21",
        [
            "未使用项目推荐的 CharHMMTokenizer：其 KenLM 扩展在 Python 3.11/Windows 无可用轮子，本机也无 MSVC/WSL 编译环境。",
            "官方文档说明该实现颗粒度较粗、基本以字为单位。",
        ],
    )


def wordpiece_with_offsets(tokenizer, text: str) -> tuple[list[str], list[str]]:
    """按官方 tokenizer 边界切分，同时以原文片段替代 [UNK] 以便无损还原。"""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    model_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    tokens: list[str] = []
    kept_model_tokens: list[str] = []
    cursor = 0
    for model_token, (start, end) in zip(model_tokens, offsets):
        start = int(start)
        end = int(end)
        if start > cursor:
            tokens.append(text[cursor:start])
            kept_model_tokens.append("[GAP]")
        if end > start:
            tokens.append(text[start:end])
            kept_model_tokens.append(str(model_token))
            cursor = max(cursor, end)
    if cursor < len(text):
        tokens.append(text[cursor:])
        kept_model_tokens.append("[GAP]")
    return tokens, kept_model_tokens


def run_bert_ancient(poems: Sequence[dict[str, object]], results_dir: Path) -> None:
    """运行 bert-ancient-chinese 官方 WordPiece tokenizer。"""
    activate_runtime()
    from transformers import AutoTokenizer, __version__ as transformers_version

    started = time.perf_counter()
    model_dir = ROOT / "models" / "bert-ancient-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    unknown_count = 0
    model_token_count = 0

    def records() -> Iterator[dict[str, object]]:
        nonlocal unknown_count, model_token_count
        for poem in poems:
            text = str(poem["text"])
            tokens, model_tokens = wordpiece_with_offsets(tokenizer, text)
            unknown_count += sum(token == tokenizer.unk_token for token in model_tokens)
            model_token_count += len(model_tokens)
            validate_tokens(text, tokens, int(poem["id"]), "bert_ancient")
            yield {"id": poem["id"], "tokens": tokens}

    write_jsonl(results_dir / "bert_ancient.jsonl", records())
    save_method_metadata(
        results_dir,
        "bert_ancient",
        len(poems),
        time.perf_counter() - started,
        "Jihuai/bert-ancient-chinese 官方 AutoTokenizer 的 WordPiece 边界；[UNK] 位置以原文片段保存",
        f"transformers {transformers_version}",
        [
            "官方仓库仅发布预训练 Fill-Mask 模型，没有发布可直接推理的 CWS 任务头。",
            "这一路是子词切分基线，不应解释为经过微调的古汉语词法分词。",
        ],
        {
            "vocabulary_size": len(tokenizer),
            "model_token_count": model_token_count,
            "unknown_token_count": unknown_count,
        },
    )


@dataclass
class SikuChunk:
    """sikufenci 单个 BERT 输入块。"""

    poem_index: int
    chunk_index: int
    chars: str
    pieces: list[str]
    char_piece_starts: list[int]


def build_siku_chunks(tokenizer, texts: Sequence[str], max_pieces: int = 510) -> list[SikuChunk]:
    """按子词上限切块，并保留每个原字符对应的首个子词。"""
    chunks: list[SikuChunk] = []
    for poem_index, text in enumerate(texts):
        chunk_index = 0
        chars: list[str] = []
        pieces: list[str] = []
        starts: list[int] = []

        def flush() -> None:
            nonlocal chunk_index, chars, pieces, starts
            if chars:
                chunks.append(
                    SikuChunk(poem_index, chunk_index, "".join(chars), pieces, starts)
                )
                chunk_index += 1
                chars = []
                pieces = []
                starts = []

        for char in text:
            char_pieces = tokenizer.tokenize(char) or ["[UNK]"]
            if pieces and len(pieces) + len(char_pieces) > max_pieces:
                flush()
            starts.append(len(pieces))
            chars.append(char)
            pieces.extend(char_pieces[:max_pieces])
        flush()
        if not text:
            chunks.append(SikuChunk(poem_index, 0, "", [], []))
    return chunks


def decode_siku_tags(chars: str, tags: Sequence[str]) -> list[str]:
    """依照 sikufenci 原实现，在 E 或 S 标签后断词。"""
    if not chars:
        return []
    tokens: list[str] = []
    current: list[str] = []
    for char, tag in zip(chars, tags):
        current.append(char)
        if tag in {"E", "S"}:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


class SikuSegmenter:
    """直接加载 sikufenci 官方微调权重的批量推理器。"""

    labels = {1: "X", 2: "O", 3: "B", 4: "I", 5: "E", 6: "S", 7: "[CLS]", 8: "[SEP]"}

    def __init__(self, model_dir: Path, batch_size: int = 8):
        activate_runtime()
        import torch
        from pytorch_pretrained_bert.modeling import BertConfig, BertForTokenClassification
        from pytorch_pretrained_bert.tokenization import BertTokenizer

        self.torch = torch
        self.batch_size = batch_size
        config = BertConfig(str(model_dir / "bert_config.json"))
        self.model = BertForTokenClassification(config, num_labels=9)
        state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
        if "classifier.weight" not in state or "classifier.bias" not in state:
            raise RuntimeError(
                "sikufenci 权重不含分词分类头；官方 Google Drive 文件实际是 MaskedLM。"
                "请从 README 所列百度网盘下载 415.6 MB 的分词 pytorch_model.bin。"
            )
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(str(model_dir), do_lower_case=False)

    def segment_many(self, texts: Sequence[str]) -> list[list[str]]:
        """对一批诗文按相近长度组批推理。"""
        torch = self.torch
        chunks = build_siku_chunks(self.tokenizer, texts)
        indexed = list(enumerate(chunks))
        indexed.sort(key=lambda item: len(item[1].pieces))
        predicted: dict[int, list[str]] = {}
        for offset in range(0, len(indexed), self.batch_size):
            batch = indexed[offset : offset + self.batch_size]
            max_length = max(len(chunk.pieces) + 2 for _, chunk in batch)
            input_ids: list[list[int]] = []
            masks: list[list[int]] = []
            for _, chunk in batch:
                tokens = ["[CLS]", *chunk.pieces, "[SEP]"]
                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                mask = [1] * len(ids)
                padding = max_length - len(ids)
                input_ids.append(ids + [0] * padding)
                masks.append(mask + [0] * padding)
            ids_tensor = torch.tensor(input_ids, dtype=torch.long)
            masks_tensor = torch.tensor(masks, dtype=torch.long)
            segments_tensor = torch.zeros_like(ids_tensor)
            with torch.inference_mode():
                logits = self.model(ids_tensor, segments_tensor, masks_tensor)
                label_ids = torch.argmax(logits, dim=2).cpu().tolist()
            for (original_index, chunk), row in zip(batch, label_ids):
                tags = [self.labels.get(row[start + 1], "S") for start in chunk.char_piece_starts]
                predicted[original_index] = decode_siku_tags(chunk.chars, tags)
        by_poem: dict[int, list[tuple[int, list[str]]]] = defaultdict(list)
        for original_index, chunk in enumerate(chunks):
            by_poem[chunk.poem_index].append((chunk.chunk_index, predicted[original_index]))
        results: list[list[str]] = []
        for poem_index in range(len(texts)):
            tokens: list[str] = []
            for _, chunk_tokens in sorted(by_poem[poem_index]):
                tokens.extend(chunk_tokens)
            results.append(tokens)
        return results


def completed_part_ids(part_path: Path, poems: Sequence[dict[str, object]]) -> int:
    """验证 sikufenci 断点文件并返回已完成行数。"""
    if not part_path.exists():
        return 0
    count = 0
    with part_path.open("r", encoding="utf-8") as handle:
        for count, line in enumerate(handle, 1):
            record = json.loads(line)
            if count > len(poems) or record["id"] != poems[count - 1]["id"]:
                raise ValueError("sikufenci 断点文件与当前语料顺序不一致")
    return count


def run_sikufenci(
    poems: Sequence[dict[str, object]],
    results_dir: Path,
    batch_size: int = 8,
    block_size: int = 256,
) -> None:
    """运行 sikufenci 官方微调模型，支持按完整 JSONL 行续跑。"""
    started = time.perf_counter()
    model_dir = ROOT / "models" / "sikufenci"
    weights = model_dir / "pytorch_model.bin"
    if not weights.exists():
        raise FileNotFoundError("缺少 models/sikufenci/pytorch_model.bin")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "sikufenci.jsonl"
    part_path = results_dir / "sikufenci.jsonl.part"
    if output_path.exists():
        output_path.unlink()
    completed = completed_part_ids(part_path, poems)
    segmenter = SikuSegmenter(model_dir, batch_size=batch_size)
    mode = "a" if completed else "w"
    with part_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for start in range(completed, len(poems), block_size):
            block = poems[start : start + block_size]
            texts = [str(poem["text"]) for poem in block]
            token_lists = segmenter.segment_many(texts)
            for poem, tokens in zip(block, token_lists):
                validate_tokens(str(poem["text"]), tokens, int(poem["id"]), "sikufenci")
                handle.write(json.dumps({"id": poem["id"], "tokens": tokens}, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            finished = min(start + len(block), len(poems))
            print(f"sikufenci：{finished}/{len(poems)}", flush=True)
    part_path.replace(output_path)
    save_method_metadata(
        results_dir,
        "sikufenci",
        len(poems),
        time.perf_counter() - started,
        "官方 BertForTokenClassification 微调权重；按原实现以 B/I/E/S 标签中的 E、S 为词尾",
        "sikufenci 0.0.28",
        [
            "仅 CPU 推理；长文本在 510 个子词处切块，块边界被视为词边界。",
            "模型面向繁体古籍，本 SQL 语料以简体为主，域与字形均存在偏移。",
        ],
        {
            "weights_sha256": sha256_file(weights),
            "batch_size": batch_size,
            "block_size": block_size,
            "resumed_from_records": completed,
            "elapsed_seconds_scope": "本次进程；若从断点续跑，不含此前进程耗时",
        },
    )


def is_han(char: str) -> bool:
    """判断字符是否属于常用或扩展汉字区。"""
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x323AF
    )


def candidate_boundaries(text: str) -> set[int]:
    """只比较相邻两个汉字之间的潜在词界。"""
    return {index for index in range(1, len(text)) if is_han(text[index - 1]) and is_han(text[index])}


def token_boundaries(text: str, tokens: Sequence[str]) -> set[int]:
    """把分词列表转换为原文字符偏移上的词界集合。"""
    candidates = candidate_boundaries(text)
    offsets: set[int] = set()
    cursor = 0
    for token in tokens[:-1]:
        cursor += len(token)
        if cursor in candidates:
            offsets.add(cursor)
    return offsets


def render_tokens(tokens: Sequence[str], max_chars: int = 120) -> str:
    """将词列表压缩为适合报告展示的短字符串。"""
    rendered: list[str] = []
    length = 0
    for token in tokens:
        if length + len(token) > max_chars:
            rendered.append("…")
            break
        rendered.append(token)
        length += len(token)
    return " / ".join(rendered)


def open_result_streams(results_dir: Path):
    """按固定顺序打开四个结果文件。"""
    return {method: (results_dir / f"{method}.jsonl").open("r", encoding="utf-8") for method in METHODS}


def compare_results(corpus_path: Path, results_dir: Path, report_dir: Path) -> dict[str, object]:
    """在没有人工金标准的前提下比较边界、粒度与词频。"""
    report_dir.mkdir(parents=True, exist_ok=True)
    counters = {method: Counter() for method in METHODS}
    method_stats = {
        method: {"tokens": 0, "lexical_tokens": 0, "han_chars": 0, "multi_char_tokens": 0}
        for method in METHODS
    }
    pair_stats = {
        pair: {"a": 0, "b": 0, "common": 0, "union": 0, "exact_poems": 0}
        for pair in itertools.combinations(METHODS, 2)
    }
    decision_total = 0
    unanimous_decisions = 0
    all_exact_poems = 0
    poem_count = 0
    disagreement_heap: list[tuple[float, int, dict[str, object]]] = []
    famous_titles = {"静夜思", "蜀道难", "将进酒", "春晓", "登鹳雀楼", "枫桥夜泊", "锦瑟", "琵琶行"}
    famous_rows: list[dict[str, object]] = []
    streams = open_result_streams(results_dir)
    try:
        with corpus_path.open("r", encoding="utf-8") as corpus_handle:
            for corpus_line in corpus_handle:
                if not corpus_line.strip():
                    continue
                poem = json.loads(corpus_line)
                method_records: dict[str, dict[str, object]] = {}
                for method, stream in streams.items():
                    line = stream.readline()
                    if not line:
                        raise ValueError(f"{method} 结果比语料更早结束")
                    method_records[method] = json.loads(line)
                    if method_records[method]["id"] != poem["id"]:
                        raise ValueError(f"{method} 与语料 ID 未对齐：{poem['id']}")
                poem_count += 1
                text = str(poem["text"])
                boundaries: dict[str, set[int]] = {}
                displays: dict[str, str] = {}
                for method, record in method_records.items():
                    tokens = [str(token) for token in record["tokens"]]
                    validate_tokens(text, tokens, int(poem["id"]), method)
                    boundaries[method] = token_boundaries(text, tokens)
                    displays[method] = render_tokens(tokens)
                    method_stats[method]["tokens"] += len(tokens)
                    for token in tokens:
                        lexical = "".join(char for char in token if is_han(char))
                        if lexical:
                            method_stats[method]["lexical_tokens"] += 1
                            method_stats[method]["han_chars"] += len(lexical)
                            if len(lexical) > 1:
                                method_stats[method]["multi_char_tokens"] += 1
                            counters[method][lexical] += 1
                boundary_variants = {frozenset(value) for value in boundaries.values()}
                if len(boundary_variants) == 1:
                    all_exact_poems += 1
                candidates = candidate_boundaries(text)
                decision_total += len(candidates)
                for candidate in candidates:
                    votes = sum(candidate in boundaries[method] for method in METHODS)
                    if votes in {0, len(METHODS)}:
                        unanimous_decisions += 1
                for method_a, method_b in itertools.combinations(METHODS, 2):
                    set_a = boundaries[method_a]
                    set_b = boundaries[method_b]
                    stats = pair_stats[(method_a, method_b)]
                    stats["a"] += len(set_a)
                    stats["b"] += len(set_b)
                    stats["common"] += len(set_a & set_b)
                    stats["union"] += len(set_a | set_b)
                    if set_a == set_b:
                        stats["exact_poems"] += 1
                if candidates:
                    disputed = sum(
                        1
                        for candidate in candidates
                        if 0 < sum(candidate in boundaries[method] for method in METHODS) < len(METHODS)
                    )
                    score = disputed / len(candidates)
                else:
                    score = 0.0
                row = {
                    "id": poem["id"],
                    "poet": poem["poet"],
                    "title": poem["title"],
                    "text": text[:120] + ("…" if len(text) > 120 else ""),
                    "disagreement_rate": round(score, 6),
                    **displays,
                }
                if len(disagreement_heap) < 30:
                    heapq.heappush(disagreement_heap, (score, int(poem["id"]), row))
                elif (score, int(poem["id"])) > disagreement_heap[0][:2]:
                    heapq.heapreplace(disagreement_heap, (score, int(poem["id"]), row))
                if str(poem["title"]) in famous_titles:
                    famous_rows.append(row)
        for method, stream in streams.items():
            if stream.readline():
                raise ValueError(f"{method} 结果比语料更长")
    finally:
        for stream in streams.values():
            stream.close()

    stats_rows: list[dict[str, object]] = []
    for method in METHODS:
        stats = method_stats[method]
        lexical = stats["lexical_tokens"]
        stats_rows.append(
            {
                "method": method,
                **stats,
                "types": len(counters[method]),
                "mean_han_chars_per_lexical_token": round(stats["han_chars"] / lexical, 6) if lexical else 0,
                "multi_char_token_rate": round(stats["multi_char_tokens"] / lexical, 6) if lexical else 0,
            }
        )
    pair_rows: list[dict[str, object]] = []
    for (method_a, method_b), stats in pair_stats.items():
        denominator = stats["a"] + stats["b"]
        pair_rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "boundaries_a": stats["a"],
                "boundaries_b": stats["b"],
                "common_boundaries": stats["common"],
                "boundary_dice": round(2 * stats["common"] / denominator, 6) if denominator else 1.0,
                "boundary_jaccard": round(stats["common"] / stats["union"], 6) if stats["union"] else 1.0,
                "exact_poem_rate": round(stats["exact_poems"] / poem_count, 6) if poem_count else 0,
            }
        )
    top_rows: list[dict[str, object]] = []
    for method in METHODS:
        for rank, (token, frequency) in enumerate(counters[method].most_common(100), 1):
            top_rows.append({"method": method, "rank": rank, "token": token, "frequency": frequency})
    disagreement_rows = [item[2] for item in sorted(disagreement_heap, reverse=True)]
    write_csv(report_dir / "method_statistics.csv", stats_rows)
    write_csv(report_dir / "pairwise_boundary_agreement.csv", pair_rows)
    write_csv(report_dir / "top_tokens.csv", top_rows)
    write_csv(report_dir / "highest_disagreement_poems.csv", disagreement_rows)
    write_csv(report_dir / "famous_poem_samples.csv", famous_rows)
    summary = {
        "poem_count": poem_count,
        "candidate_boundary_count": decision_total,
        "unanimous_boundary_decision_rate": round(unanimous_decisions / decision_total, 6) if decision_total else 1.0,
        "all_four_exact_poem_rate": round(all_exact_poems / poem_count, 6) if poem_count else 0,
        "method_statistics": stats_rows,
        "pairwise": pair_rows,
    }
    write_json(report_dir / "summary.json", summary)
    (report_dir / "comparison_report.md").write_text(
        build_markdown_report(summary, famous_rows[:8]), encoding="utf-8"
    )
    return summary


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """写入带 UTF-8 BOM 的 CSV，便于直接用 Excel 打开。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(value: object) -> str:
    """把零到一的小数格式化为百分比。"""
    return f"{float(value) * 100:.2f}%"


def build_markdown_report(summary: dict[str, object], samples: Sequence[dict[str, object]]) -> str:
    """根据机器统计生成中文比较报告。"""
    lines = [
        "# 《全唐诗》四套分词结果比较",
        "",
        "## 结论摘要",
        "",
        f"本次共处理 **{summary['poem_count']:,}** 条诗歌记录。四套方法在所有相邻汉字位置上的边界判断一致率为 **{percent(summary['unanimous_boundary_decision_rate'])}**；整首诗四套边界完全相同的比例为 **{percent(summary['all_four_exact_poem_rate'])}**。这些数字衡量的是方法间一致性，不是准确率。",
        "",
        "## 方法与口径",
        "",
        "- `sikufenci`：使用官方发布的 SikuBERT 分词微调权重和 B/I/E/S 标签。",
        "- `bert_ancient`：使用 `Jihuai/bert-ancient-chinese` 官方 WordPiece tokenizer。官方没有发布可直接推理的分词任务头，因此这一路是子词基线，不是 CWS 微调结果。",
        "- `jiayan`：使用官方 `WordNgramTokenizer`（词典 DAG + 最大概率路径）。推荐的 HMM 路径依赖 KenLM，但当前 Python 3.11/Windows 环境无法构建该扩展。",
        "- `jieba`：使用精确模式并启用 HMM。",
        "- 只比较原文中相邻两个汉字之间的词界，标点和空白位置不进入边界指标。语料没有人工金标准，因此不对四套方法作准确率排名。",
        "",
        "## 分词粒度",
        "",
        "| 方法 | 总 token 数 | 汉字词 token 数 | 词型数 | 平均汉字词长 | 多字词比例 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["method_statistics"]:
        lines.append(
            f"| {row['method']} | {row['tokens']:,} | {row['lexical_tokens']:,} | {row['types']:,} | {row['mean_han_chars_per_lexical_token']:.3f} | {percent(row['multi_char_token_rate'])} |"
        )
    lines.extend(
        [
            "",
            "平均词长和多字词比例越高，表示切分粒度总体越粗；它不直接代表效果更好。",
            "",
            "## 两两边界一致性",
            "",
            "| 方法 A | 方法 B | Dice | Jaccard | 整诗完全一致率 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["pairwise"]:
        lines.append(
            f"| {row['method_a']} | {row['method_b']} | {percent(row['boundary_dice'])} | {percent(row['boundary_jaccard'])} | {percent(row['exact_poem_rate'])} |"
        )
    if samples:
        lines.extend(["", "## 代表性样例", ""])
        for sample in samples:
            lines.extend(
                [
                    f"### {sample['poet']}《{sample['title']}》（ID {sample['id']}）",
                    "",
                    f"- sikufenci：{sample['sikufenci']}",
                    f"- bert_ancient：{sample['bert_ancient']}",
                    f"- jiayan：{sample['jiayan']}",
                    f"- jieba：{sample['jieba']}",
                    "",
                ]
            )
    lines.extend(
        [
            "## 输出文件",
            "",
            "四套完整 JSONL 结果位于 `segmentation/results/`，每行包含诗歌 `id` 和 `tokens` 数组。统计表、差异最高样例和高频词位于 `segmentation/comparison/`。每套 JSONL 都已验证 `''.join(tokens) == 原文`。",
            "",
            "## 官方来源",
            "",
            *[f"- [{method}]({url})" for method, url in SOURCES.items()],
            "",
        ]
    )
    return "\n".join(lines)


def require_corpus(corpus_path: Path, sql_path: Path) -> list[dict[str, object]]:
    """优先读取已导出语料，否则从 SQL 建立。"""
    if not corpus_path.exists():
        return export_corpus(sql_path, corpus_path)
    return load_jsonl(corpus_path)


def run_method(
    method: str,
    poems: Sequence[dict[str, object]],
    results_dir: Path,
    batch_size: int,
    block_size: int,
) -> None:
    """分派单一分词方法。"""
    if method == "sikufenci":
        run_sikufenci(poems, results_dir, batch_size=batch_size, block_size=block_size)
    elif method == "bert_ancient":
        run_bert_ancient(poems, results_dir)
    elif method == "jiayan":
        run_jiayan(poems, results_dir)
    elif method == "jieba":
        run_jieba(poems, results_dir)
    else:
        raise ValueError(f"未知方法：{method}")


def parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    cli = argparse.ArgumentParser(description=__doc__)
    subparsers = cli.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="从 SQL 导出统一语料")
    export_parser.add_argument("--sql", type=Path, default=DEFAULT_SQL)
    export_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    run_parser = subparsers.add_parser("run", help="运行一套或全部分词器")
    run_parser.add_argument("--method", choices=(*METHODS, "all"), required=True)
    run_parser.add_argument("--sql", type=Path, default=DEFAULT_SQL)
    run_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    run_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--batch-size", type=int, default=8)
    run_parser.add_argument("--block-size", type=int, default=256)

    compare_parser = subparsers.add_parser("compare", help="比较四套完整结果")
    compare_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    compare_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    compare_parser.add_argument("--report-dir", type=Path, default=ROOT / "segmentation" / "comparison")

    all_parser = subparsers.add_parser("all", help="导出、四路分词并比较")
    all_parser.add_argument("--sql", type=Path, default=DEFAULT_SQL)
    all_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    all_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    all_parser.add_argument("--report-dir", type=Path, default=ROOT / "segmentation" / "comparison")
    all_parser.add_argument("--batch-size", type=int, default=8)
    all_parser.add_argument("--block-size", type=int, default=256)
    return cli


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""
    args = parser().parse_args(argv)
    if args.command == "export":
        poems = export_corpus(args.sql, args.corpus)
        print(f"已导出 {len(poems)} 条诗歌到 {args.corpus}")
        return 0
    if args.command == "run":
        poems = require_corpus(args.corpus, args.sql)
        if args.limit is not None:
            poems = poems[: args.limit]
        methods = METHODS if args.method == "all" else (args.method,)
        for method in methods:
            print(f"开始运行 {method}，共 {len(poems)} 条", flush=True)
            run_method(method, poems, args.results_dir, args.batch_size, args.block_size)
        return 0
    if args.command == "compare":
        summary = compare_results(args.corpus, args.results_dir, args.report_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "all":
        poems = export_corpus(args.sql, args.corpus)
        for method in METHODS:
            print(f"开始运行 {method}，共 {len(poems)} 条", flush=True)
            run_method(method, poems, args.results_dir, args.batch_size, args.block_size)
        compare_results(args.corpus, args.results_dir, args.report_dir)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
