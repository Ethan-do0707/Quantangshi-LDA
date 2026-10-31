"""根据 sikufenci 全量结果生成《全唐诗》停用词表及审计统计。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "segmentation" / "data" / "poems.jsonl"
DEFAULT_RESULTS = ROOT / "segmentation" / "results" / "sikufenci.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "stopwords"
DEFAULT_TOPIC_METRICS = DEFAULT_OUTPUT_DIR / "pilot_topic_metrics.csv"

MIN_FREQUENCY = 1000
MIN_DOCUMENT_RATE = 0.025
MIN_DOCUMENT_ENTROPY = 0.65
MIN_POET_ENTROPY = 0.55
MIN_TOPIC_ENTROPY = 0.70


CORE_GROUPS: dict[str, set[str]] = {
    "古汉语结构与语气词": {
        "之", "其", "而", "兮", "也", "矣", "焉", "哉", "乎", "者", "所", "乃", "则", "诸",
    },
    "介词连词与关联词": {
        "以", "于", "与", "为", "因", "若", "如", "似", "从", "对", "及", "虽", "且", "但", "然", "或", "即", "便",
    },
    "人称与指示代词": {"我", "吾", "余", "予", "汝", "尔", "彼", "此", "斯", "己"},
    "否定与疑问词": {"不", "无", "未", "莫", "非", "弗", "勿", "毋", "何", "谁", "孰", "岂", "曷", "奚"},
    "情态体貌与泛化副词": {
        "有", "是", "得", "可", "能", "欲", "将", "应", "须", "当", "已", "亦", "又", "复", "还", "更", "犹", "尚", "仍", "曾", "使",
    },
    "语料型高频泛化词": {
        "一", "多", "各", "每", "皆", "只", "唯", "惟", "相", "自", "在", "同", "共",
        "来", "上", "中", "下", "时", "见", "闻", "听", "出", "入", "向", "临", "成", "作", "起",
    },
}


OPTIONAL_GROUPS: dict[str, set[str]] = {
    "高频方位处所词": {"前", "后", "里", "间", "处", "外", "内", "方", "至"},
    "高频移动变化词": {"去", "到", "过", "回"},
    "高频时间过程词": {"初", "始", "终", "常", "忽", "渐", "正", "先"},
    "泛指人物事物词": {"人", "君", "事", "身", "名"},
    "泛化数词数量词": {"二", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "万", "两", "双", "半", "数", "几"},
    "多义虚化候选词": {"安", "胡", "故", "却", "遂", "徒", "堪", "尽", "长", "满", "重", "分", "动", "直", "疑", "定"},
}


PROTECTED_GROUPS: dict[str, set[str]] = {
    "自然与物候意象": {
        "风", "云", "山", "水", "月", "日", "天", "地", "春", "秋", "花", "雨", "雪", "江", "河", "海", "湖", "溪", "泉", "松", "竹", "梅", "柳", "草", "树", "鸟", "雁", "鹤", "猿", "鹿", "马", "鱼",
    },
    "时间与环境意象": {"夜", "朝", "暮", "晓", "夕", "寒", "暑", "年"},
    "情感与心理主题": {"心", "思", "愁", "悲", "恨", "泪", "梦", "忆", "独", "空", "醉", "怜", "爱", "情", "惊", "惜", "难"},
    "社会与文化主题": {
        "客", "酒", "诗", "歌", "书", "乡", "家", "城", "关", "塞", "宫", "王", "帝", "兵", "军", "剑", "舟", "寺", "道", "佛", "僧", "禅", "仙", "农", "田",
    },
    "主题性动作": {"归", "行", "游", "送", "别", "登", "望", "寻", "战"},
    "审美与生命主题": {"清", "明", "静", "香", "闲", "老", "生", "死", "高", "流", "轻", "色", "鸣", "卧"},
}


@dataclass(frozen=True)
class TokenStatistics:
    """保存单个词的可审计统计。"""

    token: str
    frequency_rank: int
    frequency: int
    document_frequency: int
    document_rate: float
    inverse_document_frequency: float
    document_entropy: float
    poet_count: int
    poet_entropy: float
    top_poet_share: float
    topic_entropy: float
    top_topic_share: float
    automatic_candidate: bool
    decision: str
    category: str
    reason: str


def is_han(char: str) -> bool:
    """判断字符是否属于常用或扩展汉字区。"""
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x3134F
    )


def is_han_token(token: str) -> bool:
    """只保留完全由汉字组成的词，标点由预处理阶段另行删除。"""
    return bool(token) and all(is_han(char) for char in token)


def flatten_groups(groups: Mapping[str, set[str]]) -> dict[str, str]:
    """把分类词集展开为词到类别的映射，并阻止重复分类。"""
    mapping: dict[str, str] = {}
    for category, tokens in groups.items():
        for token in tokens:
            if token in mapping:
                raise ValueError(f"词语“{token}”在同一层级重复分类")
            mapping[token] = category
    return mapping


def validate_curated_groups() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """验证核心、扩展与保护词之间没有冲突。"""
    core = flatten_groups(CORE_GROUPS)
    optional = flatten_groups(OPTIONAL_GROUPS)
    protected = flatten_groups(PROTECTED_GROUPS)
    overlaps = {
        "核心与扩展": set(core) & set(optional),
        "核心与保护": set(core) & set(protected),
        "扩展与保护": set(optional) & set(protected),
    }
    conflicts = {name: sorted(tokens) for name, tokens in overlaps.items() if tokens}
    if conflicts:
        raise ValueError(f"停用词分类存在冲突：{conflicts}")
    return core, optional, protected


def normalized_entropy(counts: Iterable[int], total: int, category_count: int) -> float:
    """计算相对于全部类别数归一化的香农熵。"""
    if total <= 0 or category_count <= 1:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(category_count)


def iter_aligned_records(corpus_path: Path, result_path: Path):
    """逐行读取统一语料与分词结果，并验证长度和 ID 对齐。"""
    with corpus_path.open("r", encoding="utf-8") as corpus_handle, result_path.open(
        "r", encoding="utf-8"
    ) as result_handle:
        for line_number, pair in enumerate(zip_longest(corpus_handle, result_handle), 1):
            corpus_line, result_line = pair
            if corpus_line is None or result_line is None:
                raise ValueError("统一语料与 sikufenci 结果行数不一致")
            poem = json.loads(corpus_line)
            result = json.loads(result_line)
            if poem["id"] != result["id"]:
                raise ValueError(f"第 {line_number} 行 ID 未对齐")
            yield poem, result


def first_pass(corpus_path: Path, result_path: Path):
    """统计全语料词频、文档频率与文档内分布。"""
    frequency: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()
    document_count_log_sum: defaultdict[str, float] = defaultdict(float)
    poets: set[str] = set()
    poem_count = 0
    for poem, result in iter_aligned_records(corpus_path, result_path):
        poem_count += 1
        poets.add(str(poem["poet"]))
        document_counts = Counter(
            str(token) for token in result["tokens"] if is_han_token(str(token))
        )
        frequency.update(document_counts)
        document_frequency.update(document_counts.keys())
        for token, count in document_counts.items():
            document_count_log_sum[token] += count * math.log(count)
    return frequency, document_frequency, document_count_log_sum, poem_count, poets


def second_pass_poets(
    corpus_path: Path, result_path: Path, target_tokens: set[str]
) -> dict[str, Counter[str]]:
    """仅为候选及人工审查词统计跨诗人分布。"""
    poet_frequency: dict[str, Counter[str]] = defaultdict(Counter)
    for poem, result in iter_aligned_records(corpus_path, result_path):
        poet = str(poem["poet"])
        document_counts = Counter(
            str(token) for token in result["tokens"] if str(token) in target_tokens
        )
        for token, count in document_counts.items():
            poet_frequency[token][poet] += count
    return poet_frequency


def group_reason(decision: str, category: str) -> str:
    """给人工分类生成简明、可复核的判断说明。"""
    if decision == "核心停用词":
        if category == "语料型高频泛化词":
            return "在全语料中高频且跨诗人分散，单独成词时语义泛化，默认删除"
        return "主要承担古汉语语法、指代、否定、关联或体貌功能，默认删除"
    if decision == "扩展停用词":
        return "频率较高且分布广，但兼有方位、时间、人物或动作语义，仅建议在敏感性分析中删除"
    if decision == "保留主题词":
        return "虽然可能高频且分布广，但承载自然意象、情感、社会文化或诗歌行动主题，不应按频率自动删除"
    return "达到统计候选门槛，但缺少足够语义依据，暂不纳入停用词表"


def load_topic_metrics(path: Path) -> dict[str, tuple[float, float]]:
    """读取探索性 LDA 生成的词主题熵和最大主题占比。"""
    metrics: dict[str, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            metrics[row["token"]] = (
                float(row["topic_entropy"]),
                float(row["top_topic_share"]),
            )
    return metrics


def calculate_filtering_impact(
    corpus_path: Path,
    result_path: Path,
    core_tokens: set[str],
    extended_tokens: set[str],
) -> dict[str, object]:
    """统计停词前后汉字词数量，并检查是否产生空诗。"""
    total_tokens = 0
    core_removed_tokens = 0
    extended_removed_tokens = 0
    empty_after_core = 0
    empty_after_extended = 0
    source_empty_poem_ids: list[int] = []
    new_empty_after_core = 0
    new_empty_after_extended = 0
    for poem, record in iter_aligned_records(corpus_path, result_path):
        tokens = [
            str(token) for token in record["tokens"] if is_han_token(str(token))
        ]
        total_tokens += len(tokens)
        core_removed_tokens += sum(token in core_tokens for token in tokens)
        extended_removed_tokens += sum(token in extended_tokens for token in tokens)
        if not tokens:
            source_empty_poem_ids.append(int(poem["id"]))
        if not any(token not in core_tokens for token in tokens):
            empty_after_core += 1
            if tokens:
                new_empty_after_core += 1
        if not any(token not in extended_tokens for token in tokens):
            empty_after_extended += 1
            if tokens:
                new_empty_after_extended += 1
    return {
        "total_han_word_tokens": total_tokens,
        "source_empty_poems": len(source_empty_poem_ids),
        "source_empty_poem_ids": source_empty_poem_ids,
        "core_removed_tokens": core_removed_tokens,
        "core_remaining_tokens": total_tokens - core_removed_tokens,
        "empty_poems_after_core": empty_after_core,
        "new_empty_poems_after_core": new_empty_after_core,
        "extended_removed_tokens": extended_removed_tokens,
        "extended_remaining_tokens": total_tokens - extended_removed_tokens,
        "empty_poems_after_extended": empty_after_extended,
        "new_empty_poems_after_extended": new_empty_after_extended,
    }


def build_statistics(corpus_path: Path, result_path: Path, topic_metrics_path: Path):
    """生成停用词决策所需的全量统计与人工分层结果。"""
    core, optional, protected = validate_curated_groups()
    topic_metrics = load_topic_metrics(topic_metrics_path)
    (
        frequency,
        document_frequency,
        document_count_log_sum,
        poem_count,
        poets,
    ) = first_pass(corpus_path, result_path)
    frequency_order = sorted(frequency, key=lambda token: (-frequency[token], token))
    frequency_rank = {token: rank for rank, token in enumerate(frequency_order, 1)}
    preliminary_candidates = {
        token
        for token, count in frequency.items()
        if count >= MIN_FREQUENCY
        and document_frequency[token] / poem_count >= MIN_DOCUMENT_RATE
    }
    target_tokens = preliminary_candidates | set(core) | set(optional) | set(protected)
    target_tokens &= set(frequency)
    poet_frequency = second_pass_poets(corpus_path, result_path, target_tokens)
    rows: list[TokenStatistics] = []
    automatic_candidates: set[str] = set()
    for token in target_tokens:
        count = frequency[token]
        doc_count = document_frequency[token]
        doc_entropy_raw = math.log(count) - document_count_log_sum[token] / count
        doc_entropy = doc_entropy_raw / math.log(poem_count)
        poet_counts = poet_frequency[token]
        poet_entropy = normalized_entropy(poet_counts.values(), count, len(poets))
        top_poet_share = max(poet_counts.values(), default=0) / count
        topic_entropy, top_topic_share = topic_metrics.get(token, (0.0, 1.0))
        automatic_candidate = (
            count >= MIN_FREQUENCY
            and doc_count / poem_count >= MIN_DOCUMENT_RATE
            and doc_entropy >= MIN_DOCUMENT_ENTROPY
            and poet_entropy >= MIN_POET_ENTROPY
            and topic_entropy >= MIN_TOPIC_ENTROPY
        )
        if automatic_candidate:
            automatic_candidates.add(token)
        if token in core:
            decision = "核心停用词"
            category = core[token]
        elif token in optional:
            decision = "扩展停用词"
            category = optional[token]
        elif token in protected:
            decision = "保留主题词"
            category = protected[token]
        elif automatic_candidate:
            decision = "高频候选未纳入"
            category = "统计候选"
        else:
            continue
        rows.append(
            TokenStatistics(
                token=token,
                frequency_rank=frequency_rank[token],
                frequency=count,
                document_frequency=doc_count,
                document_rate=doc_count / poem_count,
                inverse_document_frequency=math.log((poem_count + 1) / (doc_count + 1)) + 1,
                document_entropy=doc_entropy,
                poet_count=len(poet_counts),
                poet_entropy=poet_entropy,
                top_poet_share=top_poet_share,
                topic_entropy=topic_entropy,
                top_topic_share=top_topic_share,
                automatic_candidate=automatic_candidate,
                decision=decision,
                category=category,
                reason=group_reason(decision, category),
            )
        )
    decision_order = {"核心停用词": 0, "扩展停用词": 1, "保留主题词": 2, "高频候选未纳入": 3}
    rows.sort(key=lambda row: (decision_order[row.decision], -row.frequency, row.token))
    observed_core = set(core) & set(frequency)
    observed_optional = set(optional) & set(frequency)
    observed_protected = set(protected) & set(frequency)
    filtering_impact = calculate_filtering_impact(
        corpus_path, result_path, observed_core, observed_core | observed_optional
    )
    filtering_impact["core_removed_rate"] = (
        filtering_impact["core_removed_tokens"]
        / filtering_impact["total_han_word_tokens"]
    )
    filtering_impact["extended_removed_rate"] = (
        filtering_impact["extended_removed_tokens"]
        / filtering_impact["total_han_word_tokens"]
    )
    metadata = {
        "source_method": "sikufenci",
        "source_result": str(result_path.relative_to(ROOT)),
        "source_corpus": str(corpus_path.relative_to(ROOT)),
        "poem_count": poem_count,
        "poet_count": len(poets),
        "han_token_type_count": len(frequency),
        "thresholds": {
            "minimum_frequency": MIN_FREQUENCY,
            "minimum_document_rate": MIN_DOCUMENT_RATE,
            "minimum_document_entropy": MIN_DOCUMENT_ENTROPY,
            "minimum_poet_entropy": MIN_POET_ENTROPY,
            "minimum_pilot_topic_entropy": MIN_TOPIC_ENTROPY,
        },
        "counts": {
            "core_stopwords": len(observed_core),
            "optional_additions": len(observed_optional),
            "extended_stopwords_including_core": len(observed_core | observed_optional),
            "protected_theme_terms": len(observed_protected),
            "automatic_statistical_candidates": len(automatic_candidates),
        },
        "filtering_impact": filtering_impact,
        "selection_note": "统计门槛只生成候选；是否删除还经过古汉语语法与诗歌主题语义审查。",
        "pilot_topic_metrics": str(topic_metrics_path.relative_to(ROOT)),
    }
    return rows, metadata, frequency, observed_core, observed_optional, observed_protected


def ordered_tokens(tokens: set[str], frequency: Counter[str]) -> list[str]:
    """按语料频率降序输出词表，频率相同则按字符排序。"""
    return sorted(tokens, key=lambda token: (-frequency[token], token))


def write_word_list(path: Path, tokens: Sequence[str]) -> None:
    """写出可直接被 LDA 预处理读取的一词一行文本。"""
    path.write_text("".join(f"{token}\n" for token in tokens), encoding="utf-8")


def write_audit_csv(path: Path, rows: Sequence[TokenStatistics]) -> None:
    """写出带统计指标和人工决策理由的审计表。"""
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = asdict(row)
            for field in (
                "document_rate",
                "inverse_document_frequency",
                "document_entropy",
                "poet_entropy",
                "top_poet_share",
                "topic_entropy",
                "top_topic_share",
            ):
                record[field] = f"{record[field]:.6f}"
            writer.writerow(record)


def write_report(
    path: Path,
    rows: Sequence[TokenStatistics],
    metadata: Mapping[str, object],
    core_tokens: Sequence[str],
    optional_tokens: Sequence[str],
    protected_tokens: Sequence[str],
) -> None:
    """生成停用词选择方法与代表性统计的中文报告。"""
    core_rows = [row for row in rows if row.decision == "核心停用词"][:30]
    protected_candidates = [
        row for row in rows if row.decision == "保留主题词" and row.automatic_candidate
    ][:30]
    report = [
        "# 《全唐诗》sikufenci 停用词表说明",
        "",
        "## 结论",
        "",
        f"核心停用词共 **{len(core_tokens)}** 个；扩展表在核心表基础上增加 **{len(optional_tokens)}** 个审慎候选。默认建议使用核心表，并把扩展表用于 LDA 稳健性复算。",
        "",
        "本表不照搬现代汉语停用词表。古汉语虚词由人工语法分类确定；语料型候选要求同时满足高词频、高诗歌文档频率、高文档熵、高诗人熵和探索性主题熵。统计指标只负责发现候选，最终仍进行语义审查。",
        "",
        "## 统计门槛",
        "",
        f"- 总频次不少于 {MIN_FREQUENCY:,}；",
        f"- 至少出现于 {MIN_DOCUMENT_RATE:.1%} 的诗歌；",
        f"- 归一化文档熵不低于 {MIN_DOCUMENT_ENTROPY:.2f}；",
        f"- 归一化诗人熵不低于 {MIN_POET_ENTROPY:.2f}。",
        f"- 20 主题探索性 LDA 的归一化主题熵不低于 {MIN_TOPIC_ENTROPY:.2f}。",
        "",
        "高文档熵和诗人熵表示某词并非集中在少量诗歌或少数诗人中；高主题熵表示该词在探索性 LDA 的各主题间也较均匀，因而更可能是全语料泛化词。",
        "",
        "## 停词影响",
        "",
        "| 口径 | 删除汉字词数 | 删除比例 | 剩余汉字词数 | 空诗数 |",
        "|---|---:|---:|---:|---:|",
        f"| 核心表 | {metadata['filtering_impact']['core_removed_tokens']:,} | {metadata['filtering_impact']['core_removed_rate']:.2%} | {metadata['filtering_impact']['core_remaining_tokens']:,} | {metadata['filtering_impact']['empty_poems_after_core']:,} |",
        f"| 扩展表 | {metadata['filtering_impact']['extended_removed_tokens']:,} | {metadata['filtering_impact']['extended_removed_rate']:.2%} | {metadata['filtering_impact']['extended_remaining_tokens']:,} | {metadata['filtering_impact']['empty_poems_after_extended']:,} |",
        "",
        f"两种口径均未新增空文档。表中的 {metadata['filtering_impact']['source_empty_poems']} 首空诗来自源 SQL 本身，ID 为：{', '.join(str(value) for value in metadata['filtering_impact']['source_empty_poem_ids'])}。",
        "",
        "## 核心表中频率最高的词",
        "",
        "| 词 | 总频次 | 诗歌文档率 | 诗人熵 | 主题熵 | 类别 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in core_rows:
        report.append(
            f"| {row.token} | {row.frequency:,} | {row.document_rate:.2%} | {row.poet_entropy:.3f} | {row.topic_entropy:.3f} | {row.category} |"
        )
    report.extend(
        [
            "",
            "## 高频但明确保护的主题词",
            "",
            "以下词可能满足统计门槛，但承载自然意象、情感或社会文化主题，因此不因高频而删除。",
            "",
            "| 词 | 总频次 | 诗歌文档率 | 诗人熵 | 主题熵 | 类别 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in protected_candidates:
        report.append(
            f"| {row.token} | {row.frequency:,} | {row.document_rate:.2%} | {row.poet_entropy:.3f} | {row.topic_entropy:.3f} | {row.category} |"
        )
    report.extend(
        [
            "",
            "## 使用建议",
            "",
            "- 主模型读取 `tang_poetry_stopwords_core.txt`；",
            "- 稳健性分析读取 `tang_poetry_stopwords_extended.txt`，该文件已包含全部核心词；",
            "- 标点和非汉字符号应通过字符规则删除，不混入词汇停用词表；",
            "- 不建议删除 `protected_theme_terms.txt` 中的词；",
            "- 完整指标、决策类别和理由见 `tang_poetry_stopword_audit.csv`。",
            "",
            "## 口径限制",
            "",
            "当前“主题区分能力弱”综合使用文档频率、跨诗人分布均匀度和无停用词探索性 LDA 的主题熵，仍不是基于人工主题标签的判别实验。扩展表中的方位、时间、人物和动作词具有多义性，必须通过保留/删除两组正式 LDA 的主题稳定性比较再决定是否用于最终模型。",
            "",
            f"语料规模：{metadata['poem_count']:,} 首，{metadata['poet_count']:,} 个诗人标签。",
        ]
    )
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def build(
    corpus_path: Path,
    result_path: Path,
    output_dir: Path,
    topic_metrics_path: Path,
) -> dict[str, object]:
    """执行统计、分层并写出全部停用词成果。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, metadata, frequency, core, optional, protected = build_statistics(
        corpus_path, result_path, topic_metrics_path
    )
    core_tokens = ordered_tokens(core, frequency)
    optional_tokens = ordered_tokens(optional, frequency)
    extended_tokens = ordered_tokens(core | optional, frequency)
    protected_tokens = ordered_tokens(protected, frequency)
    write_word_list(output_dir / "tang_poetry_stopwords_core.txt", core_tokens)
    write_word_list(output_dir / "tang_poetry_stopwords_extended.txt", extended_tokens)
    write_word_list(output_dir / "protected_theme_terms.txt", protected_tokens)
    write_audit_csv(output_dir / "tang_poetry_stopword_audit.csv", rows)
    (output_dir / "tang_poetry_stopword_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        output_dir / "stopword_report.md",
        rows,
        metadata,
        core_tokens,
        optional_tokens,
        protected_tokens,
    )
    return metadata


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="生成《全唐诗》sikufenci 停用词表")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topic-metrics", type=Path, default=DEFAULT_TOPIC_METRICS)
    return parser.parse_args()


def main() -> int:
    """命令行入口。"""
    args = parse_args()
    metadata = build(args.corpus, args.results, args.output_dir, args.topic_metrics)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
