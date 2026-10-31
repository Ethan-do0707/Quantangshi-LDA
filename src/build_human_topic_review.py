"""从既有第一轮输出生成六个候选模型的人工 Topic 审核包。"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import clean_corpus as cleaner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "human_topic_review_config.yaml"
EXPECTED_SEEDS = (42, 2026, 20260830)
SCORE_FIELDS = (
    "semantic_coherence_score",
    "poem_coherence_score",
    "interpretability_score",
    "distinctiveness_score",
)
FLAG_FIELDS = (
    "duplicate_topic_flag",
    "mixed_topic_flag",
    "noise_topic_flag",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 UTF-8 CSV，并保留原字段字符串。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def setup_logger(path: Path) -> logging.Logger:
    """建立同时写入终端和文件的日志器。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("human_topic_review")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def resolve_path(project_root: Path, value: Any, name: str) -> Path:
    """解析配置中的项目相对路径。"""

    if not isinstance(value, str) or not value.strip():
        raise cleaner.CleaningError(f"{name} 必须是非空路径")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def mean(values: Sequence[float]) -> float:
    """计算非空数列均值。"""

    if not values:
        raise cleaner.CleaningError("无法对空数列计算均值")
    return float(statistics.fmean(values))


def select_seed(
    candidate: Mapping[str, Any],
    model_metrics: Sequence[Mapping[str, str]],
    stability_metrics: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """选择 NPMI 最接近多种子均值的 seed，并整理量化依据。"""

    corpus = str(candidate["corpus"])
    topic_count = int(candidate["K"])
    matching_models = [
        row
        for row in model_metrics
        if row["corpus"] == corpus and int(row["K"]) == topic_count
    ]
    seeds = sorted(int(row["seed"]) for row in matching_models)
    if seeds != list(EXPECTED_SEEDS):
        raise cleaner.CleaningError(
            f"{candidate['model_id']} seed 不完整：{seeds} != {list(EXPECTED_SEEDS)}"
        )
    matching_stability = [
        row
        for row in stability_metrics
        if row["corpus"] == corpus and int(row["K"]) == topic_count
    ]
    if len(matching_stability) != 3:
        raise cleaner.CleaningError(
            f"{candidate['model_id']} 稳定性 seed pair 数不是 3"
        )
    npmi_by_seed = {
        int(row["seed"]): float(row["mean_topic_npmi"])
        for row in matching_models
    }
    diversity_by_seed = {
        int(row["seed"]): float(row["topic_diversity"])
        for row in matching_models
    }
    mean_npmi = mean(list(npmi_by_seed.values()))
    selected_seed = min(
        seeds,
        key=lambda seed: (abs(npmi_by_seed[seed] - mean_npmi), seed),
    )
    pairwise_values = [
        float(row["matched_topic_jaccard"]) for row in matching_stability
    ]
    stability_by_seed: dict[int, list[float]] = defaultdict(list)
    stability_rows: list[dict[str, Any]] = []
    for row in matching_stability:
        seed_a = int(row["seed_a"])
        seed_b = int(row["seed_b"])
        value = float(row["matched_topic_jaccard"])
        stability_by_seed[seed_a].append(value)
        stability_by_seed[seed_b].append(value)
        stability_rows.append(
            {
                "model_id": candidate["model_id"],
                "corpus": corpus,
                "K": topic_count,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "matched_topic_jaccard": f"{value:.10f}",
            }
        )
    seed_rows = []
    for seed in seeds:
        seed_rows.append(
            {
                "model_id": candidate["model_id"],
                "corpus": corpus,
                "K": topic_count,
                "seed": seed,
                "mean_topic_npmi": f"{npmi_by_seed[seed]:.10f}",
                "distance_from_group_mean_npmi": f"{abs(npmi_by_seed[seed] - mean_npmi):.10f}",
                "topic_diversity": f"{diversity_by_seed[seed]:.10f}",
                "mean_stability_involving_seed": f"{mean(stability_by_seed[seed]):.10f}",
                "selected_for_review": str(seed == selected_seed),
            }
        )
    selection = {
        **dict(candidate),
        "available_seeds": ";".join(str(seed) for seed in seeds),
        "selected_seed": selected_seed,
        "mean_npmi": f"{mean_npmi:.10f}",
        "selected_seed_npmi": f"{npmi_by_seed[selected_seed]:.10f}",
        "distance_from_mean": f"{abs(npmi_by_seed[selected_seed] - mean_npmi):.10f}",
        "selected_seed_topic_diversity": f"{diversity_by_seed[selected_seed]:.10f}",
        "mean_pairwise_stability": f"{mean(pairwise_values):.10f}",
        "selected_seed_mean_stability": f"{mean(stability_by_seed[selected_seed]):.10f}",
        "selection_rule": "NPMI closest to multi-seed mean; smaller seed breaks ties",
    }
    return selection, seed_rows, stability_rows


def group_topics(
    rows: Sequence[Mapping[str, str]], topic_count: int, expected_rank_count: int
) -> dict[int, list[dict[str, str]]]:
    """按 Topic 分组并验证现有 rank 数量与连续性。"""

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["topic_id"])].append(dict(row))
    if sorted(grouped) != list(range(1, topic_count + 1)):
        raise cleaner.CleaningError("Topic ID 不连续或数量不匹配")
    for topic_id, topic_rows in grouped.items():
        topic_rows.sort(key=lambda row: int(row["rank"]))
        ranks = [int(row["rank"]) for row in topic_rows]
        if ranks != list(range(1, expected_rank_count + 1)):
            raise cleaner.CleaningError(
                f"Topic {topic_id} rank 不完整：{ranks}"
            )
    return grouped


def load_corpus_rows(path: Path) -> list[dict[str, str]]:
    """读取逐诗清洗语料并验证 corpus_row 可按零起始行号定位。"""

    rows = read_csv(path)
    if not rows:
        raise cleaner.CleaningError(f"语料为空：{path}")
    return rows


def enrich_representatives(
    model_id: str,
    grouped_rows: Mapping[int, Sequence[Mapping[str, str]]],
    corpus_rows: Sequence[Mapping[str, str]],
    requested_count: int,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    """给既有 top-5 代表诗补入 cleaned_tokens，并保持原概率顺序。"""

    all_rows: list[dict[str, Any]] = []
    grouped_output: dict[int, list[dict[str, Any]]] = {}
    for topic_id in sorted(grouped_rows):
        topic_output: list[dict[str, Any]] = []
        for row in grouped_rows[topic_id]:
            corpus_row = int(row["corpus_row"])
            if corpus_row < 0 or corpus_row >= len(corpus_rows):
                raise cleaner.CleaningError(
                    f"{model_id} Topic {topic_id} corpus_row 越界：{corpus_row}"
                )
            corpus_record = corpus_rows[corpus_row]
            if str(corpus_record["poem_id"]) != str(row["poem_id"]):
                raise cleaner.CleaningError(
                    f"{model_id} Topic {topic_id} poem_id 与语料行不一致"
                )
            output_row = {
                "model_id": model_id,
                "topic_id": topic_id,
                "rank": int(row["rank"]),
                "poem_id": row["poem_id"],
                "author": row["author"],
                "title": row["title"],
                "original_text": row["original_text"],
                "cleaned_tokens": corpus_record["cleaned_tokens"],
                "topic_probability": row["topic_probability"],
                "requested_top_n": requested_count,
                "available_top_n": len(grouped_rows[topic_id]),
                "completeness_status": "partial_existing_top5_only",
            }
            topic_output.append(output_row)
            all_rows.append(output_row)
        grouped_output[topic_id] = topic_output
    return all_rows, grouped_output


def top_words_markdown(
    model_id: str,
    selected_seed: int,
    grouped: Mapping[int, Sequence[Mapping[str, str]]],
    requested_count: int,
) -> str:
    """生成现有主题词的人读 Markdown。"""

    lines = [
        f"# {model_id} Topic Top Words",
        "",
        f"Selected seed: `{selected_seed}`",
        "",
        f"> 缺失提示：第一轮只保存了每个 Topic 的 top-20，无法恢复要求的 top-{requested_count}；以下不补造 rank 21–{requested_count}。",
        "",
    ]
    for topic_id in sorted(grouped):
        lines.extend([f"## Topic {topic_id:02d}", ""])
        for row in grouped[topic_id]:
            lines.append(
                f"{row['rank']}. {row['token']} — {float(row['probability']):.12f}"
            )
        lines.append("")
    return "\n".join(lines)


def human_evaluation_section() -> list[str]:
    """返回统一的人工填写区。"""

    return [
        "## F. Human Evaluation",
        "",
        "### Topic Label",
        "",
        "人工命名：",
        "",
        "### Semantic Coherence",
        "",
        "Score: 1 / 2 / 3 / 4 / 5",
        "",
        "1 = 关键词几乎无共同语义；2 = 存在少量局部关系但整体松散；3 = 大致可辨认共同语义；4 = 大部分围绕清晰主题；5 = 高度一致并形成明确主题。",
        "",
        "### Poem Coherence",
        "",
        "Score: 1 / 2 / 3 / 4 / 5",
        "",
        "1 = 代表诗基本无共同主题；2 = 共同性较弱；3 = 大部分可以解释；4 = 明显共享主题；5 = Top poems 与随机高概率诗均高度一致。",
        "",
        "### Interpretability",
        "",
        "Score: 1 / 2 / 3 / 4 / 5",
        "",
        "1 = 无法命名；2 = 只能非常模糊命名；3 = 可给出宽泛名称；4 = 可给出明确文学主题名称；5 = 主题边界非常清晰且文学意义明确。",
        "",
        "### Distinctiveness",
        "",
        "Score: 1 / 2 / 3 / 4 / 5",
        "",
        "1 = 与其他 Topic 大量重复；2 = 重复严重；3 = 有一定独立性；4 = 基本独立；5 = 与其他 Topic 边界非常清楚。",
        "",
        "### Overall Notes",
        "",
        "人工自由备注：",
        "",
        "- 是否疑似多个主题混合？",
        "- 是否疑似主题过度拆分？",
        "- 是否由少数专名或低频词驱动？",
        "- 是否具有明确文学史意义？",
        "- 是否与其他 Topic 重复？",
        "- 是否存在明显噪音？",
        "",
    ]


def review_card(
    selection: Mapping[str, Any],
    topic_id: int,
    words: Sequence[Mapping[str, str]],
    representatives: Sequence[Mapping[str, Any]],
    soft_prevalence: str,
    requested_words: int,
    requested_representatives: int,
) -> str:
    """生成一张只呈现现有真实证据的 Topic 审核卡。"""

    title_model = str(selection["model_id"]).replace("_", "-")
    lines = [
        f"# Model: {title_model}",
        "",
        f"# Topic: {topic_id:02d}",
        "",
        f"Selected seed: `{selection['selected_seed']}`",
        "",
        "## A. Top 30 Words",
        "",
        f"> 数据可用性：仅有 {len(words)}/{requested_words} 个既有主题词；未补造缺失排名。",
        "",
    ]
    for row in words:
        lines.append(
            f"{row['rank']}. {row['token']} — {float(row['probability']):.12f}"
        )
    lines.extend(
        [
            "",
            "## B. Top 20 Representative Poems",
            "",
            f"> 数据可用性：仅有 {len(representatives)}/{requested_representatives} 首既有代表诗；完整 document-topic 矩阵未保存。",
            "",
        ]
    )
    for row in representatives:
        lines.extend(
            [
                f"### Rank {row['rank']} — {row['author']}《{row['title']}》",
                "",
                f"- Poem ID: {row['poem_id']}",
                f"- Topic probability: {float(row['topic_probability']):.10f}",
                f"- Original text: {row['original_text']}",
                f"- Cleaned tokens: {row['cleaned_tokens']}",
                "",
            ]
        )
    lines.extend(
        [
            "## C. Random High-Probability Poems",
            "",
            "> Unavailable：缺少完整 P(topic|poem)，无法定义前 10% 且 P≥1/K 的候选池，也无法执行不与 Top 20 重复的固定随机抽样。",
            "",
            "## D. Topic Prevalence",
            "",
            f"- soft prevalence: {float(soft_prevalence):.10f}",
            "- dominant poem count: Unavailable",
            "- dominant poem ratio: Unavailable",
            "- P(topic|poem) ≥ 0.2: Unavailable",
            "- P(topic|poem) ≥ 0.3: Unavailable",
            "- P(topic|poem) ≥ 0.5: Unavailable",
            "- P(topic|poem) ≥ 2/K: Unavailable",
            "- P(topic|poem) ≥ 3/K: Unavailable",
            "",
            "## E. Most Similar Topics",
            "",
            "> Unavailable：没有完整 topic-word 分布，也没有要求的 top-100 fallback；现有 top-20 不用于冒充合规相似度。",
            "",
        ]
    )
    lines.extend(human_evaluation_section())
    return "\n".join(lines)


def parse_score(value: str) -> float | None:
    """解析 1–5 分；空值返回 None，非法值明确失败。"""

    if not value.strip():
        return None
    number = float(value)
    if number < 1 or number > 5:
        raise cleaner.CleaningError(f"人工评分超出 1–5：{value}")
    return number


def parse_flag(value: str) -> bool | None:
    """解析人工布尔标记；空值表示尚未填写。"""

    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise cleaner.CleaningError(f"无法解析人工标记：{value}")


def summarize_human_scores(
    evaluation_rows: Sequence[Mapping[str, str]],
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """在人工填写后计算模型级汇总；未评分模型保持空值。"""

    result: list[dict[str, Any]] = []
    for candidate in candidates:
        model_id = str(candidate["model_id"])
        rows = [row for row in evaluation_rows if row["model_id"] == model_id]
        if len(rows) != int(candidate["K"]):
            raise cleaner.CleaningError(
                f"{model_id} 人工评分行数 {len(rows)} != K={candidate['K']}"
            )
        parsed_scores = [
            {field: parse_score(str(row.get(field, ""))) for field in SCORE_FIELDS}
            for row in rows
        ]
        complete_indices = [
            index
            for index, values in enumerate(parsed_scores)
            if all(values[field] is not None for field in SCORE_FIELDS)
        ]
        base = {
            "model_id": model_id,
            "corpus": candidate["corpus"],
            "K": candidate["K"],
        }
        if not complete_indices:
            result.append(
                {
                    **base,
                    "mean_semantic_coherence": "",
                    "mean_poem_coherence": "",
                    "mean_interpretability": "",
                    "mean_distinctiveness": "",
                    "mean_total_score": "",
                    "number_of_topics_scored": "",
                    "number_of_uninterpretable_topics": "",
                    "uninterpretable_topic_ratio": "",
                    "number_of_duplicate_topics": "",
                    "duplicate_topic_ratio": "",
                    "number_of_mixed_topics": "",
                    "number_of_noise_topics": "",
                    "summary_status": "awaiting_complete_four_score_rows",
                }
            )
            continue
        field_means = {
            field: mean(
                [float(parsed_scores[index][field]) for index in complete_indices]
            )
            for field in SCORE_FIELDS
        }
        row_means = [
            mean([float(parsed_scores[index][field]) for field in SCORE_FIELDS])
            for index in complete_indices
        ]
        interpretability = [
            float(parsed_scores[index]["interpretability_score"])
            for index in complete_indices
        ]
        flags = {
            field: [parse_flag(str(row.get(field, ""))) for row in rows]
            for field in FLAG_FIELDS
        }
        flag_counts = {
            field: sum(value is True for value in values)
            if any(value is not None for value in values)
            else None
            for field, values in flags.items()
        }
        uninterpretable_count = sum(value <= 2 for value in interpretability)
        duplicate_count = flag_counts["duplicate_topic_flag"]
        result.append(
            {
                **base,
                "mean_semantic_coherence": f"{field_means['semantic_coherence_score']:.6f}",
                "mean_poem_coherence": f"{field_means['poem_coherence_score']:.6f}",
                "mean_interpretability": f"{field_means['interpretability_score']:.6f}",
                "mean_distinctiveness": f"{field_means['distinctiveness_score']:.6f}",
                "mean_total_score": f"{mean(row_means):.6f}",
                "number_of_topics_scored": len(complete_indices),
                "number_of_uninterpretable_topics": uninterpretable_count,
                "uninterpretable_topic_ratio": f"{uninterpretable_count / len(complete_indices):.6f}",
                "number_of_duplicate_topics": "" if duplicate_count is None else duplicate_count,
                "duplicate_topic_ratio": ""
                if duplicate_count is None
                else f"{duplicate_count / len(rows):.6f}",
                "number_of_mixed_topics": ""
                if flag_counts["mixed_topic_flag"] is None
                else flag_counts["mixed_topic_flag"],
                "number_of_noise_topics": ""
                if flag_counts["noise_topic_flag"] is None
                else flag_counts["noise_topic_flag"],
                "summary_status": "partial_or_complete_human_scores",
            }
        )
    return result


def root_readme(
    selections: Sequence[Mapping[str, Any]], review_config: Mapping[str, Any]
) -> str:
    """生成审核包总说明。"""

    selection_lines = [
        f"- `{row['model_id']}`：selected seed `{row['selected_seed']}`；mean NPMI={row['mean_npmi']}；selected NPMI={row['selected_seed_npmi']}；distance={row['distance_from_mean']}。"
        for row in selections
    ]
    return "\n".join(
        [
            "# 《全唐诗》六个候选 LDA 模型人工 Topic 审核包",
            "",
            "## 1. 候选与目的",
            "",
            "本包用于 Human Interpretability Evaluation，候选为 A-K20、B-K20、B-K30、C-K20、D-K20、D-K30。它们覆盖宽松到较严格的 TF×DF 语料，并保留 B、D 的 K=20/30 比较。量化指标只用于定位审核对象，不直接决定最终模型。",
            "",
            "## 2. Selected seed",
            "",
            "每个 Corpus×K 从 42、2026、20260830 三个 seed 中，选择 mean topic NPMI 最接近三 seed 平均值者；若距离并列，选较小 seed。没有选择单次 NPMI 最高值。",
            "",
            *selection_lines,
            "",
            "## 3. 数据可用性边界",
            "",
            "> 第一轮没有保存完整 sklearn 模型、完整 topic-word 矩阵或完整 document-topic 矩阵。每个 Topic 只保存 top-20 words、top-5 representative poems 与 soft prevalence。本包严格禁止重训，也不以 top-20 冒充规定的 top-100 fallback。",
            "",
            "因此：",
            "",
            "- `topic_top_words.csv` 含真实 top-20，缺少的 rank 21–30 不补造；",
            "- `topic_top_representative_poems.csv` 含真实 top-5，并补入 cleaned_tokens；",
            "- 随机高概率样本、dominant/阈值 prevalence、合规 Topic similarity、跨模型 matching 与 split candidate 当前不可计算；",
            "- 对应 CSV 保留规范表头，`missing_data_report.csv` 与各比较目录的 `MISSING_DATA.md` 说明原因；空表不表示相似 pair 为 0。",
            "",
            "## 4. Top Words",
            "",
            "现有词按第一轮 `topics.csv` 的 topic probability 降序排列。完整要求为 top-30，但当前只能审计已保存的 top-20。",
            "",
            "## 5. Representative Poems",
            "",
            "现有代表诗按第一轮保存的 P(topic|poem) 降序排列。完整要求为 top-20，但当前只能审计已保存的 top-5；正文和 cleaned_tokens 均已包含。",
            "",
            "## 6. Random High-Probability Poems",
            "",
            f"原设计要求在概率前 {float(review_config['high_probability_pool_fraction']) * 100:.0f}% 且 P≥1/K 的候选池中，以 seed={review_config['random_seed']} 抽取 {review_config['requested_random_poems']} 首并排除 Top 20。缺少完整 document-topic 矩阵，故没有执行抽样。",
            "",
            "## 7. Topic Prevalence",
            "",
            "soft prevalence 是全体有效诗 P(topic|poem) 的均值，已从第一轮主题输出恢复。dominant poem count/ratio、固定阈值计数以及 2/K、3/K 计数需要完整 document-topic 矩阵，当前留空。",
            "",
            "## 8. Topic Similarity",
            "",
            f"原设计优先使用完整 topic-word 分布的 cosine similarity，缺失时用 top-{review_config['similarity_fallback_top_words']} 稀疏向量。当前只有 top-20，故不计算，也不自动判断重复 Topic。",
            "",
            "## 9. 人工评分",
            "",
            "每张卡预留 Semantic Coherence、Poem Coherence、Interpretability、Distinctiveness 四项 1–5 分。请把分数和标签同步填入 `human_topic_evaluation.csv`；布尔标记使用 `True/False`。不要把缺失的随机诗或相似度当作负面评分证据。",
            "",
            "- Semantic Coherence：1=关键词几乎无共同语义，2=局部相关但整体松散，3=大致可辨认共同语义，4=大部分围绕清晰主题，5=高度一致并形成明确主题；",
            "- Poem Coherence：1=代表诗基本无共同主题，2=共同性较弱，3=大部分可以解释，4=明显共享主题，5=Top poems 与随机高概率诗均高度一致；当前随机样本缺失时只按可见材料评分并在备注说明；",
            "- Interpretability：1=无法命名，2=只能非常模糊命名，3=可给出宽泛名称，4=可给出明确文学主题名称，5=主题边界清晰且文学意义明确；",
            "- Distinctiveness：1=与其他 Topic 大量重复，2=重复严重，3=有一定独立性，4=基本独立，5=与其他 Topic 边界非常清楚；当前相似度缺失时不得臆测量化证据。",
            "",
            "## 10. 评分后汇总",
            "",
            "填写完成后运行：",
            "",
            "```powershell",
            "python src/build_human_topic_review.py --config config/human_topic_review_config.yaml --summary-only",
            "```",
            "",
            "脚本会验证 1–5 分，按 interpretability_score≤2 统计 uninterpretable topics，并更新 `model_human_evaluation_summary.csv`。不会训练 LDA。",
            "",
            "## 11. 重要原则",
            "",
            "本包没有自动命名、自动评分、自动判好坏、自动删除 Topic、修改语料或重新训练模型。人工结论必须由研究者填写。",
            "",
        ]
    )


def write_missing_comparison(
    directory: Path,
    title: str,
    csv_name: str,
    fieldnames: Sequence[str],
    reason: str,
) -> None:
    """为不可恢复的比较生成空表头和明确缺失说明。"""

    directory.mkdir(parents=True, exist_ok=True)
    cleaner.write_csv(directory / csv_name, fieldnames, [])
    (directory / "MISSING_DATA.md").write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                "> 状态：Unavailable；空 CSV 不表示没有匹配、相似 pair 或 split candidate。",
                "",
                reason,
                "",
                "根据任务禁令，本阶段没有重新训练 LDA。",
                "",
            ]
        ),
        encoding="utf-8",
    )


def generate_package(config_path: Path, project_root: Path) -> dict[str, Any]:
    """生成审核包；只读取既有模型输出，不拟合或转换任何 LDA。"""

    config = cleaner.load_config(config_path)
    paths_config = cleaner.require_mapping(config.get("paths"), "paths")
    review_config = cleaner.require_mapping(config.get("review"), "review")
    candidates_value = config.get("candidates")
    if not isinstance(candidates_value, list) or len(candidates_value) != 6:
        raise cleaner.CleaningError("candidates 必须包含六个候选模型")
    candidates = [cleaner.require_mapping(value, "candidate") for value in candidates_value]
    model_metrics_path = resolve_path(
        project_root, paths_config.get("model_metrics"), "paths.model_metrics"
    )
    stability_metrics_path = resolve_path(
        project_root, paths_config.get("stability_metrics"), "paths.stability_metrics"
    )
    model_root = resolve_path(
        project_root, paths_config.get("model_root"), "paths.model_root"
    )
    corpus_root = resolve_path(
        project_root, paths_config.get("corpus_root"), "paths.corpus_root"
    )
    output_root = resolve_path(
        project_root, paths_config.get("output_root"), "paths.output_root"
    )
    log_path = resolve_path(project_root, paths_config.get("log_path"), "paths.log_path")
    logger = setup_logger(log_path)
    logger.info("开始生成六模型人工 Topic 审核包；禁止重新训练")

    model_metrics = read_csv(model_metrics_path)
    stability_metrics = read_csv(stability_metrics_path)
    selections: list[dict[str, Any]] = []
    seed_rows_all: list[dict[str, Any]] = []
    stability_rows_all: list[dict[str, Any]] = []
    for candidate in candidates:
        selection, seed_rows, stability_rows = select_seed(
            candidate, model_metrics, stability_metrics
        )
        selections.append(selection)
        seed_rows_all.extend(seed_rows)
        stability_rows_all.extend(stability_rows)

    input_paths = {config_path, Path(__file__).resolve(), model_metrics_path, stability_metrics_path}
    for selection in selections:
        corpus = str(selection["corpus"])
        topic_count = int(selection["K"])
        input_paths.update(
            {
                corpus_root / corpus / "lda_corpus.csv",
                corpus_root / corpus / "vocabulary.csv",
            }
        )
        for seed in EXPECTED_SEEDS:
            model_dir = model_root / corpus / f"K{topic_count}" / f"seed_{seed}"
            input_paths.update(
                {
                    model_dir / "metrics.json",
                    model_dir / "topics.csv",
                    model_dir / "representative_poems.csv",
                }
            )
    missing_inputs = [path for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise cleaner.CleaningError(
            "候选模型输入缺失：" + ", ".join(str(path) for path in missing_inputs)
        )
    input_hashes_before = {str(path): cleaner.file_sha256(path) for path in input_paths}

    output_root.mkdir(parents=True, exist_ok=True)
    cleaner.write_csv(
        output_root / "model_seed_metrics.csv",
        list(seed_rows_all[0].keys()),
        seed_rows_all,
    )
    cleaner.write_csv(
        output_root / "model_stability_metrics.csv",
        list(stability_rows_all[0].keys()),
        stability_rows_all,
    )

    requested_words = int(review_config["requested_top_words"])
    requested_representatives = int(review_config["requested_representative_poems"])
    requested_random = int(review_config["requested_random_poems"])
    similarity_fallback = int(review_config["similarity_fallback_top_words"])
    selection_output: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    review_card_counts: dict[str, int] = {}

    for selection in selections:
        model_id = str(selection["model_id"])
        corpus = str(selection["corpus"])
        topic_count = int(selection["K"])
        selected_seed = int(selection["selected_seed"])
        model_dir = model_root / corpus / f"K{topic_count}" / f"seed_{selected_seed}"
        topics_rows = read_csv(model_dir / "topics.csv")
        representative_source = read_csv(model_dir / "representative_poems.csv")
        grouped_words = group_topics(topics_rows, topic_count, 20)
        grouped_representatives = group_topics(representative_source, topic_count, 5)
        corpus_rows = load_corpus_rows(corpus_root / corpus / "lda_corpus.csv")
        representatives, grouped_representative_output = enrich_representatives(
            model_id,
            grouped_representatives,
            corpus_rows,
            requested_representatives,
        )
        model_output = output_root / model_id
        cards_dir = model_output / "review_cards"
        cards_dir.mkdir(parents=True, exist_ok=True)

        top_word_output: list[dict[str, Any]] = []
        soft_prevalence_by_topic: dict[int, str] = {}
        for topic_id in sorted(grouped_words):
            prevalence_values = {row["topic_prevalence"] for row in grouped_words[topic_id]}
            if len(prevalence_values) != 1:
                raise cleaner.CleaningError(
                    f"{model_id} Topic {topic_id} soft prevalence 不一致"
                )
            soft_prevalence_by_topic[topic_id] = next(iter(prevalence_values))
            for row in grouped_words[topic_id]:
                top_word_output.append(
                    {
                        "model_id": model_id,
                        "topic_id": topic_id,
                        "rank": int(row["rank"]),
                        "token": row["token"],
                        "weight": row["probability"],
                        "probability": row["probability"],
                        "component_weight": row["component_weight"],
                        "requested_top_n": requested_words,
                        "available_top_n": len(grouped_words[topic_id]),
                        "completeness_status": "partial_existing_top20_only",
                    }
                )
        cleaner.write_csv(
            model_output / "topic_top_words.csv",
            list(top_word_output[0].keys()),
            top_word_output,
        )
        (model_output / "topic_top_words.md").write_text(
            top_words_markdown(
                model_id, selected_seed, grouped_words, requested_words
            ),
            encoding="utf-8",
        )
        cleaner.write_csv(
            model_output / "topic_top_representative_poems.csv",
            list(representatives[0].keys()),
            representatives,
        )
        random_fields = [
            "model_id",
            "topic_id",
            "sample_rank",
            "poem_id",
            "author",
            "title",
            "original_text",
            "cleaned_tokens",
            "topic_probability",
        ]
        cleaner.write_csv(
            model_output / "topic_random_high_probability_poems.csv",
            random_fields,
            [],
        )
        prevalence_rows: list[dict[str, Any]] = []
        for topic_id in range(1, topic_count + 1):
            probabilities = [
                float(row["topic_probability"])
                for row in grouped_representative_output[topic_id]
            ]
            probability_rows.append(
                {
                    "model_id": model_id,
                    "topic_id": topic_id,
                    "available_representative_count": len(probabilities),
                    "top1_probability": f"{max(probabilities):.10f}",
                    "mean_available_top5_probability": f"{mean(probabilities):.10f}",
                    "min_available_top5_probability": f"{min(probabilities):.10f}",
                    "uniform_probability_baseline": f"{1 / topic_count:.10f}",
                    "top1_below_0_5": str(max(probabilities) < 0.5),
                    "mean_available_top5_below_0_5": str(mean(probabilities) < 0.5),
                    "diagnostic_scope": "available_saved_top5_only",
                }
            )
            prevalence_rows.append(
                {
                    "model_id": model_id,
                    "topic_id": topic_id,
                    "soft_prevalence": soft_prevalence_by_topic[topic_id],
                    "dominant_poem_count": "",
                    "dominant_poem_ratio": "",
                    "probability_ge_0_2_count": "",
                    "probability_ge_0_3_count": "",
                    "probability_ge_0_5_count": "",
                    "probability_ge_2_over_K_count": "",
                    "probability_ge_3_over_K_count": "",
                    "valid_poem_count": "",
                    "availability_status": "soft_only_full_document_topic_missing",
                }
            )
        cleaner.write_csv(
            model_output / "topic_prevalence.csv",
            list(prevalence_rows[0].keys()),
            prevalence_rows,
        )
        cleaner.write_csv(
            model_output / "topic_similarity_matrix.csv",
            ["model_id", "topic_id"],
            [],
        )
        cleaner.write_csv(
            model_output / "high_similarity_topic_pairs.csv",
            [
                "model_id",
                "topic_a",
                "topic_b",
                "cosine_similarity",
                "shared_top_words",
            ],
            [],
        )
        (model_output / "MISSING_DATA.md").write_text(
            "\n".join(
                [
                    f"# {model_id} Missing Data",
                    "",
                    f"- Top words：要求 {requested_words}，现有 20。",
                    f"- Representative poems：要求 {requested_representatives}，现有 5。",
                    f"- Random high-probability poems：要求 {requested_random}，缺少完整 document-topic 矩阵，无法抽样。",
                    "- Prevalence：只有 soft prevalence；dominant 与阈值计数不可恢复。",
                    f"- Similarity：缺少完整 topic-word 分布和 top-{similarity_fallback} fallback；不以 top-20 代替。",
                    "- 完整模型对象：不存在。",
                    "",
                    "本阶段没有重新训练 LDA。",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        for topic_id in range(1, topic_count + 1):
            card_path = cards_dir / f"{model_id}_topic_{topic_id:02d}.md"
            card_path.write_text(
                review_card(
                    selection,
                    topic_id,
                    grouped_words[topic_id],
                    grouped_representative_output[topic_id],
                    soft_prevalence_by_topic[topic_id],
                    requested_words,
                    requested_representatives,
                ),
                encoding="utf-8",
            )
            evaluation_rows.append(
                {
                    "model_id": model_id,
                    "corpus": corpus,
                    "K": topic_count,
                    "selected_seed": selected_seed,
                    "topic_id": topic_id,
                    "proposed_label": "",
                    "semantic_coherence_score": "",
                    "poem_coherence_score": "",
                    "interpretability_score": "",
                    "distinctiveness_score": "",
                    "mean_human_score": "",
                    "duplicate_topic_flag": "",
                    "mixed_topic_flag": "",
                    "noise_topic_flag": "",
                    "notes": "",
                }
            )
        review_card_counts[model_id] = topic_count
        selection_output.append(
            {
                **selection,
                "full_model_object_available": "False",
                "full_topic_word_distribution_available": "False",
                "full_document_topic_probability_available": "False",
                "available_top_words_per_topic": 20,
                "available_representative_poems_per_topic": 5,
                "review_card_count": topic_count,
                "high_similarity_pair_count": "",
                "high_similarity_pair_status": "unavailable_missing_similarity_inputs",
            }
        )
        for capability, requested, available, status, reason in (
            (
                "topic_top_words",
                requested_words,
                20,
                "partial",
                "first round saved top-20 only",
            ),
            (
                "representative_poems",
                requested_representatives,
                5,
                "partial",
                "first round saved top-5 only",
            ),
            (
                "random_high_probability_poems",
                requested_random,
                0,
                "unavailable",
                "full document-topic matrix and candidate pool are missing",
            ),
            (
                "topic_prevalence",
                "soft+dominant+thresholds",
                "soft only",
                "partial",
                "full document-topic matrix is missing",
            ),
            (
                "topic_similarity",
                f"full distribution or top-{similarity_fallback}",
                "top-20 only",
                "unavailable",
                "saved top-20 is below required fallback depth",
            ),
        ):
            missing_rows.append(
                {
                    "model_id": model_id,
                    "capability": capability,
                    "requested": requested,
                    "available": available,
                    "status": status,
                    "reason": reason,
                    "lda_retraining_performed": "False",
                }
            )
        logger.info(
            "%s 完成：selected_seed=%d cards=%d，完整矩阵缺失",
            model_id,
            selected_seed,
            topic_count,
        )

    cleaner.write_csv(
        output_root / "model_selection_summary.csv",
        list(selection_output[0].keys()),
        selection_output,
    )
    cleaner.write_csv(
        output_root / "missing_data_report.csv",
        list(missing_rows[0].keys()),
        missing_rows,
    )
    cleaner.write_csv(
        output_root / "available_representative_probability_diagnostics.csv",
        list(probability_rows[0].keys()),
        probability_rows,
    )

    evaluation_path = output_root / "human_topic_evaluation.csv"
    evaluation_fields = list(evaluation_rows[0].keys())
    if evaluation_path.is_file():
        existing_evaluation = read_csv(evaluation_path)
        expected_keys = {
            (row["model_id"], str(row["topic_id"])) for row in evaluation_rows
        }
        existing_keys = {
            (row["model_id"], str(row["topic_id"])) for row in existing_evaluation
        }
        if existing_keys != expected_keys:
            raise cleaner.CleaningError("既有人工评分表的 model/topic 行集合不一致")
        evaluation_for_summary = existing_evaluation
        logger.info("保留既有 human_topic_evaluation.csv，未覆盖人工填写")
    else:
        cleaner.write_csv(evaluation_path, evaluation_fields, evaluation_rows)
        evaluation_for_summary = [
            {key: str(value) for key, value in row.items()} for row in evaluation_rows
        ]
    summary_rows = summarize_human_scores(evaluation_for_summary, candidates)
    cleaner.write_csv(
        output_root / "model_human_evaluation_summary.csv",
        list(summary_rows[0].keys()),
        summary_rows,
    )

    comparison_root = output_root / "comparison"
    c_vocabulary = {row["token"]: row for row in read_csv(corpus_root / "C" / "vocabulary.csv")}
    d_vocabulary = {row["token"]: row for row in read_csv(corpus_root / "D" / "vocabulary.csv")}
    c_only_rows = [
        {
            "token": row["token"],
            "TF": int(row["TF"]),
            "DF": int(row["DF"]),
        }
        for token, row in c_vocabulary.items()
        if token not in d_vocabulary and int(row["TF"]) in {3, 4}
    ]
    c_only_rows.sort(key=lambda row: (row["TF"], row["DF"], row["token"]))
    c_d_dir = comparison_root / "C_K20_vs_D_K20"
    c_d_dir.mkdir(parents=True, exist_ok=True)
    cleaner.write_csv(
        c_d_dir / "C_only_low_frequency_tokens.csv",
        ["token", "TF", "DF"],
        c_only_rows,
    )
    cleaner.write_csv(
        c_d_dir / "topic_matching.csv",
        [
            "C_topic",
            "D_topic",
            "similarity",
            "shared_top_words",
            "C_only_top_words",
            "D_only_top_words",
        ],
        [],
    )
    (c_d_dir / "topic_pair_review").mkdir(parents=True, exist_ok=True)
    (c_d_dir / "MISSING_DATA.md").write_text(
        "\n".join(
            [
                "# C-K20 vs D-K20 Topic Matching",
                "",
                f"已生成 C 中存在而 D 中不存在的 TF=3/4 词共 {len(c_only_rows):,} 个。",
                "",
                "> Topic matching 与 pair review 不可生成：两个选定模型都只有 top-20 topic words，没有完整 topic-word 分布或 top-100 fallback。`topic_matching.csv` 只有表头，不表示没有可匹配主题。",
                "",
                "根据任务禁令，本阶段没有重新训练 LDA。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (c_d_dir / "topic_pair_review" / "README.md").write_text(
        "Topic pair review 等待合规 topic matching 输入；当前没有生成伪匹配。\n",
        encoding="utf-8",
    )

    write_missing_comparison(
        comparison_root / "B_K20_vs_B_K30",
        "B-K20 vs B-K30 Topic Mapping",
        "B_K20_vs_K30_topic_mapping.csv",
        [
            "K20_topic",
            "K30_topic_rank",
            "K30_topic",
            "similarity",
            "shared_words",
            "possible_split",
        ],
        "两个模型都只有 top-20 topic words，低于要求的 top-100 fallback，无法可靠计算 Top 3 mapping 或 possible_split。",
    )
    write_missing_comparison(
        comparison_root / "D_K20_vs_D_K30",
        "D-K20 vs D-K30 Topic Mapping",
        "D_K20_vs_K30_topic_mapping.csv",
        [
            "K20_topic",
            "K30_topic_rank",
            "K30_topic",
            "similarity",
            "shared_words",
            "possible_split",
        ],
        "两个模型都只有 top-20 topic words，低于要求的 top-100 fallback，无法可靠计算 Top 3 mapping 或 possible_split。",
    )
    cleaner.write_csv(
        comparison_root / "comparison_status.csv",
        ["comparison", "status", "reason", "empty_csv_means_zero_results"],
        [
            {
                "comparison": "C_K20_vs_D_K20_topic_matching",
                "status": "unavailable",
                "reason": "full topic-word distributions and top-100 fallback are missing",
                "empty_csv_means_zero_results": "False",
            },
            {
                "comparison": "B_K20_vs_B_K30_topic_mapping",
                "status": "unavailable",
                "reason": "full topic-word distributions and top-100 fallback are missing",
                "empty_csv_means_zero_results": "False",
            },
            {
                "comparison": "D_K20_vs_D_K30_topic_mapping",
                "status": "unavailable",
                "reason": "full topic-word distributions and top-100 fallback are missing",
                "empty_csv_means_zero_results": "False",
            },
        ],
    )
    (output_root / "README.md").write_text(
        root_readme(selection_output, review_config), encoding="utf-8"
    )

    input_hashes_after = {str(path): cleaner.file_sha256(path) for path in input_paths}
    if input_hashes_before != input_hashes_after:
        raise cleaner.CleaningError("生成审核包时既有模型、语料或指标文件发生变化")
    output_files = sorted(
        [
            path
            for path in output_root.rglob("*")
            if path.is_file() and path.name != "generation_metadata.json"
        ],
        key=lambda path: str(path),
    )
    metadata = {
        "schema_version": 1,
        "method": "existing_outputs_only_human_topic_review_package",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "lda_training_performed": False,
        "corpus_modified": False,
        "automatic_topic_labels_generated": False,
        "automatic_human_scores_generated": False,
        "candidate_count": len(candidates),
        "topic_review_card_count": sum(review_card_counts.values()),
        "review_card_counts": review_card_counts,
        "selected_seeds": {
            row["model_id"]: row["selected_seed"] for row in selection_output
        },
        "availability": {
            "top_words_per_topic": 20,
            "representative_poems_per_topic": 5,
            "soft_prevalence": True,
            "full_topic_word_distribution": False,
            "full_document_topic_probability": False,
            "model_objects": False,
            "topic_similarity": False,
            "cross_model_topic_matching": False,
        },
        "C_only_TF3_or_TF4_token_count": len(c_only_rows),
        "validation": {
            "input_hashes_unchanged": input_hashes_before == input_hashes_after,
            "all_candidate_seed_triplets_present": True,
            "selected_model_topics_have_exactly_20_saved_words": True,
            "selected_model_topics_have_exactly_5_saved_representatives": True,
            "review_card_count_matches_sum_K": sum(review_card_counts.values()) == 140,
            "unavailable_outputs_not_fabricated": True,
        },
        "inputs": [
            {
                "path": cleaner.display_path(path),
                "sha256": input_hashes_before[str(path)],
            }
            for path in sorted(input_paths, key=lambda path: str(path))
        ],
        "outputs": [
            {
                "path": cleaner.display_path(path),
                "sha256": cleaner.file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in output_files
        ],
    }
    cleaner.write_json(output_root / "generation_metadata.json", metadata)
    logger.info(
        "审核包完成：models=%d cards=%d C_only_TF3_4=%d；未重新训练",
        len(candidates),
        sum(review_card_counts.values()),
        len(c_only_rows),
    )
    return metadata


def update_summary(config_path: Path, project_root: Path) -> list[dict[str, Any]]:
    """只读取人工评分表并更新模型汇总，不改动其他审核材料。"""

    config = cleaner.load_config(config_path)
    paths_config = cleaner.require_mapping(config.get("paths"), "paths")
    candidates_value = config.get("candidates")
    if not isinstance(candidates_value, list):
        raise cleaner.CleaningError("candidates 配置缺失")
    candidates = [cleaner.require_mapping(value, "candidate") for value in candidates_value]
    output_root = resolve_path(
        project_root, paths_config.get("output_root"), "paths.output_root"
    )
    evaluation_path = output_root / "human_topic_evaluation.csv"
    if not evaluation_path.is_file():
        raise cleaner.CleaningError("人工评分表不存在，请先生成审核包")
    summary_rows = summarize_human_scores(read_csv(evaluation_path), candidates)
    cleaner.write_csv(
        output_root / "model_human_evaluation_summary.csv",
        list(summary_rows[0].keys()),
        summary_rows,
    )
    return summary_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="从已验证的完整 LDA 产物生成六模型人工 Topic 审核包"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""

    args = parse_args(argv)
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if args.summary_only:
        update_summary(config_path, PROJECT_ROOT)
    else:
        config = cleaner.load_config(config_path)
        paths_config = cleaner.require_mapping(config.get("paths"), "paths")
        if paths_config.get("rebuild_root"):
            from build_complete_human_topic_review import generate_complete_package

            generate_complete_package(config_path, PROJECT_ROOT)
        else:
            generate_package(config_path, PROJECT_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
