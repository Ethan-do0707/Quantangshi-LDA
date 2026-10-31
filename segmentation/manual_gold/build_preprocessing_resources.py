"""根据人工金标准叠加结果构建《全唐诗》LDA 预处理资源。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "segmentation" / "data" / "poems.jsonl"
DEFAULT_RESULTS = ROOT / "segmentation" / "results" / "sikufenci_manual_gold.jsonl"
DEFAULT_RESULT_METADATA = (
    ROOT / "segmentation" / "results" / "sikufenci_manual_gold.metadata.json"
)
DEFAULT_STOPWORD_DIR = ROOT / "stopwords"
DEFAULT_STANDARD_DIR = ROOT / "segmentation" / "lda_standard"
DEFAULT_OUTPUT_DIR = ROOT / "segmentation" / "manual_gold" / "resources"

DEFAULT_MIN_TF = 5
DEFAULT_MIN_DF = 3
STRICT_MIN_TF = 10
STRICT_MIN_DF = 5
HIGH_DF_AUDIT_RATE = 0.10


@dataclass(frozen=True)
class CorpusStatistics:
    """保存全量词频、文档频率和语料规模。"""

    poem_count: int
    han_token_count: int
    non_han_token_count: int
    token_frequency: Counter[str]
    document_frequency: Counter[str]
    poem_texts: tuple[str, ...]


VARIANT_ROWS: tuple[dict[str, str], ...] = (
    {
        "source_form": "连薨",
        "normalized_form": "连甍",
        "relation": "底本文字差异",
        "scope": "poem_id=1；excel_row=2；《帝京篇十首》已标前缀",
        "evidence_level": "人工标注与分词结果一致",
        "status": "建议仅在该诗规范视图采用",
        "apply_automatically": "否",
        "evidence": "manual_gold.annotations.jsonl 第2行；sikufenci_manual_gold.metadata.json 的 corpus_text_mismatches",
        "note": "统一语料作“连薨”，人工标注与基础分词结果作“连甍”；不扩展为全局字级映射。",
    },
    {
        "source_form": "水",
        "normalized_form": "永",
        "relation": "文本异文候选",
        "scope": "poem_id=217；excel_row=80；《唐明堂乐章。迎送王公》",
        "evidence_level": "人工标注与基础结果不一致",
        "status": "待版本校勘",
        "apply_automatically": "否",
        "evidence": "sikufenci_manual_gold.metadata.json 的 variant_annotations",
        "note": "人工标注作“永”，基础结果作“水”；现有材料不足以确定通行底本，禁止全局替换。",
    },
    {
        "source_form": "桑乾",
        "normalized_form": "桑干",
        "relation": "地名词形",
        "scope": "poem_id=778；excel_row=268；仅“桑乾”地名",
        "evidence_level": "人工标注直接支持",
        "status": "可用于独立规范视图",
        "apply_automatically": "否",
        "evidence": "sikufenci_manual_gold.metadata.json 的 variant_annotations",
        "note": "基础结果作“桑乾”，人工标注作“桑干”；不得据此把所有“乾”全局改为“干”。",
    },
    {
        "source_form": "走覃",
        "normalized_form": "趠",
        "relation": "罕见字拆分写法",
        "scope": "poem_id=919；excel_row=326；《相和歌辞。公无渡河》",
        "evidence_level": "人工标注直接支持",
        "status": "可用于独立规范视图",
        "apply_automatically": "否",
        "evidence": "sikufenci_manual_gold.metadata.json 的 variant_annotations",
        "note": "只规范该处字形；原分词结果仍保存基础字符序列以便审计。",
    },
)


SEMANTIC_CONCEPT_CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "surface": "皇帝",
        "canonical": "皇帝",
        "mapping_group": "帝王称谓",
        "relation": "概念中心词",
        "ambiguity": "低",
        "recommended_use": "可作为聚合标签",
        "source": "研究者候选",
        "note": "概念组标签，不改写原词。",
    },
    {
        "surface": "天子",
        "canonical": "皇帝",
        "mapping_group": "帝王称谓",
        "relation": "近义称谓",
        "ambiguity": "低",
        "recommended_use": "可选聚合",
        "source": "研究者候选",
        "note": "保留原词；只在需要合并帝王称谓时使用。",
    },
    {
        "surface": "帝王",
        "canonical": "皇帝",
        "mapping_group": "帝王称谓",
        "relation": "近义称谓",
        "ambiguity": "低",
        "recommended_use": "可选聚合",
        "source": "研究者候选",
        "note": "保留原词；只在需要合并帝王称谓时使用。",
    },
    {
        "surface": "君王",
        "canonical": "皇帝",
        "mapping_group": "帝王称谓",
        "relation": "近义称谓",
        "ambiguity": "中",
        "recommended_use": "语境确认后聚合",
        "source": "研究者候选",
        "note": "可能泛指诸侯或君主，不默认等同于皇帝。",
    },
    {
        "surface": "皇王",
        "canonical": "皇帝",
        "mapping_group": "帝王称谓",
        "relation": "近义称谓",
        "ambiguity": "中",
        "recommended_use": "语境确认后聚合",
        "source": "研究者候选",
        "note": "先核对具体诗句和指称对象。",
    },
    {
        "surface": "人主",
        "canonical": "皇帝",
        "mapping_group": "帝王称谓",
        "relation": "近义称谓",
        "ambiguity": "中",
        "recommended_use": "语境确认后聚合",
        "source": "研究者候选",
        "note": "先核对具体政治语境。",
    },
    {
        "surface": "圣主",
        "canonical": "皇帝",
        "mapping_group": "帝王称谓",
        "relation": "褒义称谓",
        "ambiguity": "中",
        "recommended_use": "语境确认后聚合",
        "source": "研究者候选",
        "note": "带评价色彩，不与中性称谓完全等价。",
    },
    {
        "surface": "明主",
        "canonical": "皇帝",
        "mapping_group": "帝王称谓",
        "relation": "褒义称谓",
        "ambiguity": "中",
        "recommended_use": "语境确认后聚合",
        "source": "研究者候选",
        "note": "带评价色彩，不与中性称谓完全等价。",
    },
    {
        "surface": "至尊",
        "canonical": "皇帝",
        "mapping_group": "帝王称谓",
        "relation": "尊称",
        "ambiguity": "高",
        "recommended_use": "必须语境确认",
        "source": "研究者候选",
        "note": "可能进入宗教或泛尊崇语境。",
    },
    {
        "surface": "帝",
        "canonical": "皇帝",
        "mapping_group": "帝王称谓",
        "relation": "简称候选",
        "ambiguity": "高",
        "recommended_use": "必须语境确认",
        "source": "研究者候选",
        "note": "可能指天帝、神祇或古帝，不默认聚合。",
    },
)


def file_sha256(path: Path) -> str:
    """计算文件的 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """仓库内文件输出相对路径。"""

    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


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


def read_terms(path: Path) -> list[str]:
    """读取一行一词的 UTF-8 词表。"""

    terms: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            term = line.strip()
            if not term or term.startswith("#"):
                continue
            if term in seen:
                raise ValueError(f"词表存在重复项：{path} 第 {line_number} 行“{term}”")
            seen.add(term)
            terms.append(term)
    return terms


def read_tsv(path: Path) -> list[dict[str, str]]:
    """读取带表头的 UTF-8 TSV。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def write_lines(path: Path, values: Iterable[str]) -> None:
    """写出一行一项的 UTF-8 文本。"""

    text = "\n".join(values) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def write_table(
    path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str], delimiter: str
) -> None:
    """按给定列顺序写出 CSV 或 TSV。"""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter=delimiter,
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def iter_aligned_records(corpus_path: Path, result_path: Path):
    """逐行读取统一语料和分词结果，并核验 ID 对齐。"""

    with corpus_path.open("r", encoding="utf-8") as corpus_handle, result_path.open(
        "r", encoding="utf-8"
    ) as result_handle:
        for line_number, pair in enumerate(zip_longest(corpus_handle, result_handle), 1):
            corpus_line, result_line = pair
            if corpus_line is None or result_line is None:
                raise ValueError("统一语料与人工金标准叠加结果行数不一致")
            poem = json.loads(corpus_line)
            result = json.loads(result_line)
            if poem["id"] != result["id"]:
                raise ValueError(f"第 {line_number} 行 ID 未对齐")
            yield poem, result


def collect_statistics(corpus_path: Path, result_path: Path) -> CorpusStatistics:
    """统计人工标注叠加结果的 TF、DF 和诗篇文本。"""

    token_frequency: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()
    poem_texts: list[str] = []
    han_token_count = 0
    non_han_token_count = 0
    poem_count = 0

    for _, result in iter_aligned_records(corpus_path, result_path):
        poem_count += 1
        raw_tokens = [str(token) for token in result["tokens"]]
        poem_texts.append("".join(raw_tokens))
        han_tokens = [token for token in raw_tokens if is_han_token(token)]
        han_token_count += len(han_tokens)
        non_han_token_count += len(raw_tokens) - len(han_tokens)
        token_frequency.update(han_tokens)
        document_frequency.update(set(han_tokens))

    return CorpusStatistics(
        poem_count=poem_count,
        han_token_count=han_token_count,
        non_han_token_count=non_han_token_count,
        token_frequency=token_frequency,
        document_frequency=document_frequency,
        poem_texts=tuple(poem_texts),
    )


def load_stopword_categories(audit_path: Path) -> dict[str, dict[str, str]]:
    """从既有审计表读取人工分类和判断依据。"""

    with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["token"]: dict(row) for row in csv.DictReader(handle)}


def build_stopword_rows(
    core_terms: Sequence[str],
    extended_terms: Sequence[str],
    previous_audit: Mapping[str, Mapping[str, str]],
    stats: CorpusStatistics,
) -> list[dict[str, object]]:
    """用叠加结果重算既有唐诗停用词的 TF 和 DF。"""

    core = set(core_terms)
    rows: list[dict[str, object]] = []
    for token in extended_terms:
        previous = previous_audit.get(token, {})
        tf = stats.token_frequency[token]
        df = stats.document_frequency[token]
        rows.append(
            {
                "token": token,
                "tier": "核心" if token in core else "扩展新增",
                "category": previous.get("category", "既有人工分类"),
                "tf": tf,
                "df": df,
                "df_rate": f"{df / stats.poem_count:.8f}",
                "previous_tf": previous.get("frequency", ""),
                "previous_df": previous.get("document_frequency", ""),
                "reason": previous.get(
                    "reason", "沿用既有唐诗专用停用词语义判断；本表只重算 TF/DF。"
                ),
            }
        )
    return rows


def phrase_counts(terms: Iterable[str], poem_texts: Sequence[str]) -> dict[str, tuple[int, int]]:
    """统计保护词在原字符序列中的出现次数和诗篇覆盖数。"""

    counts: dict[str, tuple[int, int]] = {}
    for term in sorted(set(terms)):
        occurrence = 0
        document_count = 0
        for text in poem_texts:
            current = text.count(term)
            if current:
                occurrence += current
                document_count += 1
        counts[term] = (occurrence, document_count)
    return counts


def build_protected_rows(
    protected_seed: Sequence[Mapping[str, str]],
    entity_seed: Sequence[Mapping[str, str]],
    stats: CorpusStatistics,
) -> list[dict[str, object]]:
    """合并固定表达、典故和专名种子，并补充实际频率。"""

    merged: dict[str, dict[str, str]] = {}
    for row in protected_seed:
        term = row["term"]
        merged[term] = {
            "surface": term,
            "canonical": term,
            "resource_type": "固定表达/典故/诗歌词语",
            "decision": row["decision"],
            "category": row["category"],
            "scope": row["scope"],
            "ambiguity": (
                "需语境判断"
                if row["decision"] == "条件保留" or row["scope"] != "全局"
                else "低"
            ),
            "source": "protected_term_seed.tsv",
            "evidence": row["evidence"],
            "note": row["note"],
        }

    for row in entity_seed:
        surface = row["surface"]
        if surface in merged:
            current = merged[surface]
            current["resource_type"] += "|专名"
            current["category"] += f"|{row['type']}"
            current["scope"] += f"|{row['scope']}"
            current["ambiguity"] += f"|{row['ambiguity']}"
            current["source"] += "|named_entity_seed.tsv"
            current["note"] += f"；{row['note']}"
            if row["canonical"] != surface:
                current["canonical"] = row["canonical"]
        else:
            merged[surface] = {
                "surface": surface,
                "canonical": row["canonical"],
                "resource_type": "专名",
                "decision": "实体语境保护",
                "category": row["type"],
                "scope": row["scope"],
                "ambiguity": row["ambiguity"],
                "source": "named_entity_seed.tsv",
                "evidence": "既有专名人工种子",
                "note": row["note"],
            }

    text_counts = phrase_counts(merged, stats.poem_texts)
    rows: list[dict[str, object]] = []
    for surface in sorted(merged):
        row = dict(merged[surface])
        text_tf, text_df = text_counts[surface]
        row.update(
            {
                "token_tf": stats.token_frequency[surface],
                "token_df": stats.document_frequency[surface],
                "text_tf": text_tf,
                "text_df": text_df,
            }
        )
        rows.append(row)
    return rows


def token_decision(
    token: str,
    tf: int,
    df: int,
    stopwords: set[str],
    protected: set[str],
    min_tf: int,
    min_df: int,
) -> tuple[bool, str]:
    """按保护优先、停用词其次、低频最后的顺序判断词元。"""

    if token in protected:
        return True, "保护词豁免"
    if token in stopwords:
        return False, "停用词删除"
    if tf < min_tf and df < min_df:
        return False, "TF与DF均低于阈值"
    if tf < min_tf:
        return False, "TF低于阈值"
    if df < min_df:
        return False, "DF低于阈值"
    return True, "达到TF/DF阈值"


def build_token_rows(
    stats: CorpusStatistics,
    core_stopwords: set[str],
    extended_stopwords: set[str],
    protected_terms: set[str],
) -> list[dict[str, object]]:
    """为全部汉字词元生成默认和严格过滤判断。"""

    rows: list[dict[str, object]] = []
    ordered_tokens = sorted(
        stats.token_frequency,
        key=lambda token: (-stats.token_frequency[token], token),
    )
    for rank, token in enumerate(ordered_tokens, 1):
        tf = stats.token_frequency[token]
        df = stats.document_frequency[token]
        default_keep, default_reason = token_decision(
            token,
            tf,
            df,
            core_stopwords,
            protected_terms,
            DEFAULT_MIN_TF,
            DEFAULT_MIN_DF,
        )
        strict_keep, strict_reason = token_decision(
            token,
            tf,
            df,
            extended_stopwords,
            protected_terms,
            STRICT_MIN_TF,
            STRICT_MIN_DF,
        )
        rows.append(
            {
                "token": token,
                "frequency_rank": rank,
                "tf": tf,
                "df": df,
                "df_rate": f"{df / stats.poem_count:.8f}",
                "is_core_stopword": token in core_stopwords,
                "is_extended_stopword": token in extended_stopwords,
                "is_protected": token in protected_terms,
                "high_df_audit_flag": df / stats.poem_count >= HIGH_DF_AUDIT_RATE,
                "default_keep": default_keep,
                "default_reason": default_reason,
                "strict_keep": strict_keep,
                "strict_reason": strict_reason,
            }
        )
    return rows


def build_semantic_rows(
    entity_seed: Sequence[Mapping[str, str]], stats: CorpusStatistics
) -> list[dict[str, object]]:
    """生成实体别称和少量待复核概念候选旁表。"""

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for entity in entity_seed:
        surface = entity["surface"]
        canonical = entity["canonical"]
        if surface == canonical:
            continue
        relation = entity["type"]
        key = (surface, canonical, relation)
        if key in seen:
            continue
        seen.add(key)
        ambiguity = entity["ambiguity"]
        rows.append(
            {
                "surface": surface,
                "canonical": canonical,
                "mapping_group": "专名规范",
                "relation": relation,
                "ambiguity": ambiguity,
                "recommended_use": (
                    "实体识别后可聚合" if ambiguity == "低" else "语境确认后聚合"
                ),
                "token_tf": stats.token_frequency[surface],
                "token_df": stats.document_frequency[surface],
                "source": "named_entity_seed.tsv",
                "note": entity["note"],
            }
        )

    for candidate in SEMANTIC_CONCEPT_CANDIDATES:
        surface = candidate["surface"]
        key = (surface, candidate["canonical"], candidate["relation"])
        if key in seen:
            continue
        seen.add(key)
        row: dict[str, object] = dict(candidate)
        row["token_tf"] = stats.token_frequency[surface]
        row["token_df"] = stats.document_frequency[surface]
        rows.append(row)

    return sorted(rows, key=lambda row: (str(row["mapping_group"]), str(row["canonical"]), str(row["surface"])))


def summarize_filter(
    rows: Sequence[Mapping[str, object]], keep_field: str, stats: CorpusStatistics
) -> dict[str, object]:
    """汇总某一过滤档位保留的词型和词次。"""

    kept = [row for row in rows if row[keep_field] is True]
    kept_token_count = sum(int(row["tf"]) for row in kept)
    return {
        "retained_type_count": len(kept),
        "removed_type_count": len(rows) - len(kept),
        "retained_token_count": kept_token_count,
        "retained_token_rate": round(kept_token_count / stats.han_token_count, 8),
    }


def build_rules(
    stats: CorpusStatistics,
    token_rows: Sequence[Mapping[str, object]],
    core_count: int,
    extended_count: int,
    protected_count: int,
    result_path: Path,
) -> dict[str, object]:
    """生成机器可读的 TF/DF 规则和实测保留规模。"""

    distribution = {}
    for threshold in (1, 2, 3, 5, 10, 20, 50, 100):
        distribution[str(threshold)] = {
            "types_with_tf_at_least": sum(
                1 for value in stats.token_frequency.values() if value >= threshold
            ),
            "types_with_df_at_least": sum(
                1 for value in stats.document_frequency.values() if value >= threshold
            ),
        }

    return {
        "schema_version": 1,
        "statistics_scope": "sikufenci_manual_gold 全量结果；TF/DF 均按诗篇文档计算",
        "statistics_stage": "当前分词基线，尚未执行本目录的保护合并或可选语义映射",
        "document_unit": "poem",
        "input": {
            "result": display_path(result_path),
            "poem_count": stats.poem_count,
            "han_token_count": stats.han_token_count,
            "han_type_count": len(stats.token_frequency),
            "non_han_token_count": stats.non_han_token_count,
        },
        "order": [
            "可选词形规范视图（不覆盖原文）",
            "先识别并锁定专名、典故和固定表达",
            "可选语义映射（默认关闭）",
            "若词界或词面发生变化，重新统计TF/DF",
            "删除非汉字词元",
            "删除所选档位的停用词",
            "应用最小TF与最小DF",
        ],
        "default_profile": {
            "stopword_tier": "core",
            "min_tf": DEFAULT_MIN_TF,
            "min_df": DEFAULT_MIN_DF,
            "protected_terms_bypass_minimums": True,
            "automatic_max_df": None,
            "high_df_manual_audit_rate": HIGH_DF_AUDIT_RATE,
            "single_character_policy": "不按长度删除；单字同样执行停用词和TF/DF规则",
            **summarize_filter(token_rows, "default_keep", stats),
        },
        "strict_sensitivity_profile": {
            "stopword_tier": "extended",
            "min_tf": STRICT_MIN_TF,
            "min_df": STRICT_MIN_DF,
            "protected_terms_bypass_minimums": True,
            "automatic_max_df": None,
            "high_df_manual_audit_rate": HIGH_DF_AUDIT_RATE,
            **summarize_filter(token_rows, "strict_keep", stats),
        },
        "resource_counts": {
            "core_stopword_count": core_count,
            "extended_stopword_count": extended_count,
            "protected_surface_count": protected_count,
        },
        "frequency_distribution": distribution,
        "rationale": {
            "minimums": "默认档 TF>=5 且 DF>=3，降低偶见切分误差和只见于极少诗篇的词型；严格档用于稳健性比较。",
            "maximum_df": "不自动按高DF删除，因为风、云、山、月等高频词仍有主题区分价值；高DF只触发人工审计。",
            "protection": "保护词优先级最高，避免低频典故、专名和固定表达被阈值误删。",
            "semantic_mapping": "默认关闭；实体别称与概念候选只作旁表，不回写原词。",
            "baseline_estimate": "本文件的保留规模按当前分词基线估算；实际执行保护合并或语义映射后应重新统计。",
        },
    }


def build_readme(rules: Mapping[str, object], semantic_count: int, variant_count: int) -> str:
    """生成资源说明文档。"""

    default = rules["default_profile"]
    strict = rules["strict_sensitivity_profile"]
    inputs = rules["input"]
    resources = rules["resource_counts"]
    return f"""# 《全唐诗》人工金标准叠加版预处理资源

本目录根据 `manual_gold.annotations.jsonl` 和全量 `sikufenci_manual_gold.jsonl` 构建。人工词界只覆盖 485 首诗（其中 59 首仅覆盖已标前缀）；因此这些文件是“人工金标准叠加版资源”，不能称为 43,030 首全人工金标准。

## 文件

- `01_variant_normalization.tsv`：{variant_count} 条可定位的文本/字形差异。所有映射均保留原文，默认不自动替换。
- `02_stopwords_core.txt`：{resources['core_stopword_count']} 个默认删除的唐诗专用核心停用词。
- `02_stopwords_extended.txt`：{resources['extended_stopword_count']} 个核心加敏感性分析候选词。
- `02_stopword_audit.csv`：在人工叠加结果上重新计算的 TF、DF，并保留既有语义分类理由。
- `03_protected_terms.tsv`：{resources['protected_surface_count']} 个专名、典故、固定表达或诗歌词语种子，同时给出“现成词元频率”和“原字符序列频率”。
- `04_tf_df_rules.json`：推荐过滤顺序、默认/严格阈值和实测保留规模。
- `04_token_statistics.csv`：当前分词基线中全部汉字词元的 TF、DF、停用/保护标记和两档预估判断。
- `05_semantic_mapping.tsv`：{semantic_count} 条可选实体别称或概念映射；默认关闭，不覆盖原词。
- `resource_metadata.json`：输入输出哈希、规模和一致性检查。

## 推荐流程

1. 原文与原分词结果只读保存；需要时另建规范视图。
2. 先按 `03_protected_terms.tsv` 识别专名、典故和固定表达，再删除停用词。
3. 默认使用核心停用词；扩展词表仅用于敏感性分析。
4. 若保护合并或语义映射改变了词界/词面，重新统计 TF/DF。
5. 默认保留 `TF >= {default['min_tf']}` 且 `DF >= {default['min_df']}` 的词；严格对照使用 `TF >= {strict['min_tf']}` 且 `DF >= {strict['min_df']}`。
6. 保护词不受最小 TF/DF 限制。单字不因长度自动删除。
7. 不设自动 `max_df`：风、云、山、月等高频词可能仍是主题词；DF 覆盖率达到 {HIGH_DF_AUDIT_RATE:.0%} 只进入人工审计。
8. 高歧义语义映射必须先做语境消歧。

## 实测规模

- 诗篇：{inputs['poem_count']:,}
- 汉字词次：{inputs['han_token_count']:,}
- 汉字词型：{inputs['han_type_count']:,}
- 默认档基线预估：{default['retained_type_count']:,} 个词型、{default['retained_token_rate']:.2%} 的汉字词次
- 严格档基线预估：{strict['retained_type_count']:,} 个词型、{strict['retained_token_rate']:.2%} 的汉字词次

## 证据边界

- 停用词的语义判断沿用本项目既有《全唐诗》专用审计结论，本次只在人工叠加结果上重算 TF/DF。
- `03_protected_terms.tsv` 保留原种子的“强制合并”“条件保留”与实体语境范围；“保护”不等于“已完成典故考证”。
- `01_variant_normalization.tsv` 中“待版本校勘”的行绝不可自动替换。
- `05_semantic_mapping.tsv` 的专名别称来自既有人工种子；“帝王称谓”是研究者候选组，不主张词义完全相同。

## 重建

在仓库根目录运行：

```powershell
python segmentation/manual_gold/build_preprocessing_resources.py
```
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--result-metadata", type=Path, default=DEFAULT_RESULT_METADATA)
    parser.add_argument("--stopword-dir", type=Path, default=DEFAULT_STOPWORD_DIR)
    parser.add_argument("--standard-dir", type=Path, default=DEFAULT_STANDARD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """构建五类资源并写出审计元数据。"""

    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    core_path = args.stopword_dir / "tang_poetry_stopwords_core.txt"
    extended_path = args.stopword_dir / "tang_poetry_stopwords_extended.txt"
    stopword_audit_path = args.stopword_dir / "tang_poetry_stopword_audit.csv"
    protected_seed_path = args.standard_dir / "protected_term_seed.tsv"
    entity_seed_path = args.standard_dir / "named_entity_seed.tsv"

    core_terms = read_terms(core_path)
    extended_terms = read_terms(extended_path)
    if not set(core_terms) <= set(extended_terms):
        raise ValueError("核心停用词必须是扩展停用词表的子集")

    previous_audit = load_stopword_categories(stopword_audit_path)
    protected_seed = read_tsv(protected_seed_path)
    entity_seed = read_tsv(entity_seed_path)
    stats = collect_statistics(args.corpus, args.results)

    stopword_rows = build_stopword_rows(
        core_terms, extended_terms, previous_audit, stats
    )
    protected_rows = build_protected_rows(protected_seed, entity_seed, stats)
    protected_terms = {str(row["surface"]) for row in protected_rows}
    conflicts = sorted(set(extended_terms) & protected_terms)
    if conflicts:
        raise ValueError(f"停用词与保护词发生冲突：{conflicts}")

    token_rows = build_token_rows(
        stats, set(core_terms), set(extended_terms), protected_terms
    )
    semantic_rows = build_semantic_rows(entity_seed, stats)
    rules = build_rules(
        stats,
        token_rows,
        len(core_terms),
        len(extended_terms),
        len(protected_terms),
        args.results,
    )

    output_paths = {
        "variants": args.output_dir / "01_variant_normalization.tsv",
        "core_stopwords": args.output_dir / "02_stopwords_core.txt",
        "extended_stopwords": args.output_dir / "02_stopwords_extended.txt",
        "stopword_audit": args.output_dir / "02_stopword_audit.csv",
        "protected_terms": args.output_dir / "03_protected_terms.tsv",
        "tf_df_rules": args.output_dir / "04_tf_df_rules.json",
        "token_statistics": args.output_dir / "04_token_statistics.csv",
        "semantic_mapping": args.output_dir / "05_semantic_mapping.tsv",
        "readme": args.output_dir / "README.md",
        "metadata": args.output_dir / "resource_metadata.json",
    }

    write_table(
        output_paths["variants"],
        VARIANT_ROWS,
        [
            "source_form",
            "normalized_form",
            "relation",
            "scope",
            "evidence_level",
            "status",
            "apply_automatically",
            "evidence",
            "note",
        ],
        "\t",
    )
    write_lines(output_paths["core_stopwords"], core_terms)
    write_lines(output_paths["extended_stopwords"], extended_terms)
    write_table(
        output_paths["stopword_audit"],
        stopword_rows,
        [
            "token",
            "tier",
            "category",
            "tf",
            "df",
            "df_rate",
            "previous_tf",
            "previous_df",
            "reason",
        ],
        ",",
    )
    write_table(
        output_paths["protected_terms"],
        protected_rows,
        [
            "surface",
            "canonical",
            "resource_type",
            "decision",
            "category",
            "scope",
            "ambiguity",
            "token_tf",
            "token_df",
            "text_tf",
            "text_df",
            "source",
            "evidence",
            "note",
        ],
        "\t",
    )
    output_paths["tf_df_rules"].write_text(
        json.dumps(rules, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_table(
        output_paths["token_statistics"],
        token_rows,
        [
            "token",
            "frequency_rank",
            "tf",
            "df",
            "df_rate",
            "is_core_stopword",
            "is_extended_stopword",
            "is_protected",
            "high_df_audit_flag",
            "default_keep",
            "default_reason",
            "strict_keep",
            "strict_reason",
        ],
        ",",
    )
    write_table(
        output_paths["semantic_mapping"],
        semantic_rows,
        [
            "surface",
            "canonical",
            "mapping_group",
            "relation",
            "ambiguity",
            "recommended_use",
            "token_tf",
            "token_df",
            "source",
            "note",
        ],
        "\t",
    )
    output_paths["readme"].write_text(
        build_readme(rules, len(semantic_rows), len(VARIANT_ROWS)),
        encoding="utf-8",
        newline="\n",
    )

    result_metadata = json.loads(args.result_metadata.read_text(encoding="utf-8"))
    metadata = {
        "schema_version": 1,
        "method": "manual_gold_overlay_preprocessing_resources",
        "inputs": {
            "corpus": {"path": display_path(args.corpus), "sha256": file_sha256(args.corpus)},
            "results": {"path": display_path(args.results), "sha256": file_sha256(args.results)},
            "result_metadata": {
                "path": display_path(args.result_metadata),
                "sha256": file_sha256(args.result_metadata),
            },
            "core_stopwords": {"path": display_path(core_path), "sha256": file_sha256(core_path)},
            "extended_stopwords": {
                "path": display_path(extended_path),
                "sha256": file_sha256(extended_path),
            },
            "protected_seed": {
                "path": display_path(protected_seed_path),
                "sha256": file_sha256(protected_seed_path),
            },
            "entity_seed": {
                "path": display_path(entity_seed_path),
                "sha256": file_sha256(entity_seed_path),
            },
        },
        "coverage": {
            "poem_count": stats.poem_count,
            "manual_annotated_poem_count": result_metadata["annotated_poem_count"],
            "full_annotation_count": result_metadata["full_annotation_count"],
            "prefix_annotation_count": result_metadata["prefix_annotation_count"],
            "unannotated_poem_count": result_metadata["unannotated_poem_count"],
        },
        "counts": {
            "han_token_count": stats.han_token_count,
            "han_type_count": len(stats.token_frequency),
            "variant_row_count": len(VARIANT_ROWS),
            "core_stopword_count": len(core_terms),
            "extended_stopword_count": len(extended_terms),
            "protected_surface_count": len(protected_terms),
            "semantic_mapping_count": len(semantic_rows),
        },
        "validation": {
            "line_count_and_id_alignment": True,
            "core_is_subset_of_extended": True,
            "stopword_protected_conflict_count": len(conflicts),
            "source_result_reconstructs_base_text": result_metadata["validation"][
                "all_outputs_reconstruct_source_result_text"
            ],
        },
        "outputs": {},
    }
    metadata["outputs"] = {
        key: {"path": display_path(path), "sha256": file_sha256(path)}
        for key, path in output_paths.items()
        if key != "metadata"
    }
    output_paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        json.dumps(
            {
                "poem_count": stats.poem_count,
                "han_token_count": stats.han_token_count,
                "han_type_count": len(stats.token_frequency),
                "core_stopword_count": len(core_terms),
                "extended_stopword_count": len(extended_terms),
                "protected_surface_count": len(protected_terms),
                "semantic_mapping_count": len(semantic_rows),
                "output_dir": display_path(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
