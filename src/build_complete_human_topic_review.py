"""使用已精确复现的完整矩阵生成六模型人工 Topic 审核包。"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from scipy.optimize import linear_sum_assignment

import build_human_topic_review as base
import clean_corpus as cleaner


@dataclass
class CompleteModelData:
    """保存一个候选模型生成审核材料所需的完整证据。"""

    selection: dict[str, Any]
    topic_word: np.ndarray
    document_topic: np.ndarray
    vocabulary: list[str]
    token_to_index: dict[str, int]
    document_rows: list[dict[str, str]]
    corpus_rows: list[dict[str, str]]
    words_by_topic: dict[int, list[dict[str, str]]]
    representatives_by_topic: dict[int, list[dict[str, Any]]]
    random_by_topic: dict[int, list[dict[str, Any]]]
    prevalence_by_topic: dict[int, dict[str, Any]]
    similarity: np.ndarray
    similar_by_topic: dict[int, list[dict[str, Any]]]
    high_similarity_pairs: list[dict[str, Any]]


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 UTF-8 CSV。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_path(project_root: Path, value: Any, name: str) -> Path:
    """解析项目相对路径。"""

    if not isinstance(value, str) or not value.strip():
        raise cleaner.CleaningError(f"{name} 必须为非空路径")
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_npz_array(path: Path, key: str) -> np.ndarray:
    """读取指定 NPZ 数组并关闭文件句柄。"""

    with np.load(path) as data:
        if key not in data:
            raise cleaner.CleaningError(f"{path} 缺少数组键 {key}")
        return np.asarray(data[key])


def validate_rebuild_checkpoint(directory: Path) -> dict[str, Any]:
    """验证候选完整模型检查点及全部登记哈希。"""

    manifest_path = directory / "artifact_manifest.json"
    validation_path = directory / "reproduction_validation.json"
    if not manifest_path.is_file() or not validation_path.is_file():
        raise cleaner.CleaningError(f"完整模型检查点缺失：{directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    validation = json.loads(validation_path.read_text(encoding="utf-8-sig"))
    if not manifest.get("reproduction_validation_passed") or validation.get("status") != "passed":
        raise cleaner.CleaningError(f"完整模型未通过复现验证：{directory.name}")
    for output in manifest.get("outputs", []):
        path = directory / str(output.get("path", ""))
        if not path.is_file() or cleaner.file_sha256(path) != output.get("sha256"):
            raise cleaner.CleaningError(f"完整模型文件哈希失效：{path}")
    return validation


def cosine_similarity_rows(matrix: np.ndarray) -> np.ndarray:
    """计算同一矩阵各行间余弦相似度。"""

    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise cleaner.CleaningError("Topic×Word 存在零向量")
    normalized = matrix / norms[:, None]
    similarity = normalized @ normalized.T
    similarity = np.clip(similarity, 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def cross_cosine_similarity(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """计算两组已对齐 Topic 向量的交叉余弦相似度。"""

    first_norms = np.linalg.norm(first, axis=1)
    second_norms = np.linalg.norm(second, axis=1)
    if np.any(first_norms == 0) or np.any(second_norms == 0):
        raise cleaner.CleaningError("跨模型 Topic 含零向量")
    return np.clip(
        (first / first_norms[:, None]) @ (second / second_norms[:, None]).T,
        0.0,
        1.0,
    )


def shared_words(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
) -> list[str]:
    """按第一主题顺序列出两组 top words 的交集。"""

    second_tokens = {str(row["token"]) for row in second}
    return [str(row["token"]) for row in first if str(row["token"]) in second_tokens]


def stable_topic_seed(base_seed: int, model_id: str, topic_id: int) -> int:
    """从全局种子稳定派生模型与 Topic 专用抽样种子。"""

    digest = hashlib.sha256(
        f"{base_seed}:{model_id}:{topic_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def sample_random_high_probability(
    *,
    model_id: str,
    topic_id: int,
    scores: np.ndarray,
    document_rows: Sequence[Mapping[str, str]],
    corpus_rows: Sequence[Mapping[str, str]],
    representative_matrix_rows: set[int],
    topic_count: int,
    pool_fraction: float,
    requested_count: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    """从前 10% 且不低于 1/K 的非代表诗中固定随机抽样。"""

    ordered = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    pool_size = max(1, math.ceil(len(scores) * pool_fraction))
    threshold = 1.0 / topic_count
    eligible = [
        matrix_row
        for matrix_row in ordered[:pool_size]
        if float(scores[matrix_row]) >= threshold
        and matrix_row not in representative_matrix_rows
    ]
    derived_seed = stable_topic_seed(base_seed, model_id, topic_id)
    rng = random.Random(derived_seed)
    selected = rng.sample(eligible, min(requested_count, len(eligible)))
    rows: list[dict[str, Any]] = []
    for sample_rank, matrix_row in enumerate(selected, 1):
        document = document_rows[matrix_row]
        corpus_row = int(document["corpus_row"])
        poem = corpus_rows[corpus_row]
        rows.append(
            {
                "model_id": model_id,
                "topic_id": topic_id,
                "sample_rank": sample_rank,
                "matrix_row": matrix_row,
                "corpus_row": corpus_row,
                "poem_id": poem["poem_id"],
                "author": poem["author"],
                "title": poem["title"],
                "original_text": poem["original_text"],
                "cleaned_tokens": poem["cleaned_tokens"],
                "topic_probability": f"{float(scores[matrix_row]):.10f}",
                "global_random_seed": base_seed,
                "derived_topic_seed": derived_seed,
                "pool_fraction": f"{pool_fraction:.6f}",
                "minimum_probability": f"{threshold:.10f}",
                "eligible_pool_size_after_excluding_top20": len(eligible),
            }
        )
    return rows


def compute_prevalence(
    model_id: str, document_topic: np.ndarray
) -> dict[int, dict[str, Any]]:
    """计算 soft、dominant 与各概率阈值 prevalence。"""

    document_count, topic_count = document_topic.shape
    dominant = np.argmax(document_topic, axis=1)
    dominant_counts = np.bincount(dominant, minlength=topic_count)
    result: dict[int, dict[str, Any]] = {}
    for topic_index in range(topic_count):
        scores = document_topic[:, topic_index]
        count = int(dominant_counts[topic_index])
        result[topic_index + 1] = {
            "model_id": model_id,
            "topic_id": topic_index + 1,
            "soft_prevalence": f"{float(scores.mean()):.10f}",
            "dominant_poem_count": count,
            "dominant_poem_ratio": f"{count / document_count:.10f}",
            "probability_ge_0_2_count": int(np.count_nonzero(scores >= 0.2)),
            "probability_ge_0_3_count": int(np.count_nonzero(scores >= 0.3)),
            "probability_ge_0_5_count": int(np.count_nonzero(scores >= 0.5)),
            "probability_ge_2_over_K_count": int(
                np.count_nonzero(scores >= 2.0 / topic_count)
            ),
            "probability_ge_3_over_K_count": int(
                np.count_nonzero(scores >= 3.0 / topic_count)
            ),
            "valid_poem_count": document_count,
            "availability_status": "complete_document_topic_matrix",
        }
    return result


def build_similarity_outputs(
    model_id: str,
    similarity: np.ndarray,
    words_by_topic: Mapping[int, Sequence[Mapping[str, Any]]],
    threshold: float,
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    """整理每主题最相似五项及高相似 pair。"""

    topic_count = similarity.shape[0]
    similar_by_topic: dict[int, list[dict[str, Any]]] = {}
    for topic_index in range(topic_count):
        ordered = sorted(
            (index for index in range(topic_count) if index != topic_index),
            key=lambda index: (-similarity[topic_index, index], index),
        )[:5]
        similar_by_topic[topic_index + 1] = [
            {
                "topic_id": other + 1,
                "similarity": float(similarity[topic_index, other]),
                "shared_words": shared_words(
                    words_by_topic[topic_index + 1], words_by_topic[other + 1]
                ),
            }
            for other in ordered
        ]
    pairs: list[dict[str, Any]] = []
    for first in range(topic_count):
        for second in range(first + 1, topic_count):
            score = float(similarity[first, second])
            if score >= threshold:
                pairs.append(
                    {
                        "model_id": model_id,
                        "topic_a": first + 1,
                        "topic_b": second + 1,
                        "cosine_similarity": f"{score:.10f}",
                        "shared_top_words": ";".join(
                            shared_words(
                                words_by_topic[first + 1], words_by_topic[second + 1]
                            )
                        ),
                    }
                )
    pairs.sort(key=lambda row: (-float(row["cosine_similarity"]), row["topic_a"], row["topic_b"]))
    return similar_by_topic, pairs


def complete_review_card(
    model: CompleteModelData,
    topic_id: int,
) -> str:
    """生成含完整六部分证据的 Topic 审核卡。"""

    selection = model.selection
    title_model = str(selection["model_id"]).replace("_", "-")
    words = model.words_by_topic[topic_id]
    representatives = model.representatives_by_topic[topic_id]
    random_poems = model.random_by_topic[topic_id]
    prevalence = model.prevalence_by_topic[topic_id]
    lines = [
        f"# Model: {title_model}",
        "",
        f"# Topic: {topic_id:02d}",
        "",
        f"Selected seed: `{selection['selected_seed']}`",
        "",
        "## A. Top 30 Words",
        "",
    ]
    for row in words:
        lines.append(f"{row['rank']}. {row['token']} — {float(row['probability']):.12f}")
    lines.extend(["", "## B. Top 20 Representative Poems", ""])
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
    lines.extend(["## C. Random High-Probability Poems", ""])
    for row in random_poems:
        lines.extend(
            [
                f"### Sample {row['sample_rank']} — {row['author']}《{row['title']}》",
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
            "## D. Topic Prevalence",
            "",
            f"- soft prevalence: {prevalence['soft_prevalence']}",
            f"- dominant poem count: {prevalence['dominant_poem_count']}",
            f"- dominant poem ratio: {prevalence['dominant_poem_ratio']}",
            f"- P(topic|poem) ≥ 0.2: {prevalence['probability_ge_0_2_count']}",
            f"- P(topic|poem) ≥ 0.3: {prevalence['probability_ge_0_3_count']}",
            f"- P(topic|poem) ≥ 0.5: {prevalence['probability_ge_0_5_count']}",
            f"- P(topic|poem) ≥ 2/K: {prevalence['probability_ge_2_over_K_count']}",
            f"- P(topic|poem) ≥ 3/K: {prevalence['probability_ge_3_over_K_count']}",
            "",
            "## E. Most Similar Topics",
            "",
        ]
    )
    for rank, row in enumerate(model.similar_by_topic[topic_id], 1):
        lines.append(
            f"{rank}. Topic {row['topic_id']:02d} — {row['similarity']:.10f} — shared: {', '.join(row['shared_words']) or '无 top-30 交词'}"
        )
    lines.append("")
    lines.extend(base.human_evaluation_section())
    return "\n".join(lines)


def top_words_markdown(model_id: str, seed: int, grouped: Mapping[int, Sequence[Mapping[str, str]]]) -> str:
    """生成完整 top-30 人读文件。"""

    lines = [f"# {model_id} Topic Top 30 Words", "", f"Selected seed: `{seed}`", ""]
    for topic_id in sorted(grouped):
        lines.extend([f"## Topic {topic_id:02d}", ""])
        lines.extend(
            f"{row['rank']}. {row['token']} — {float(row['probability']):.12f}"
            for row in grouped[topic_id]
        )
        lines.append("")
    return "\n".join(lines)


def load_complete_model(
    *,
    selection: dict[str, Any],
    rebuild_root: Path,
    corpus_root: Path,
    review: Mapping[str, Any],
    corpus_cache: dict[str, list[dict[str, str]]],
) -> CompleteModelData:
    """加载一个已验证完整模型并计算审核所需派生量。"""

    model_id = str(selection["model_id"])
    corpus_label = str(selection["corpus"])
    topic_count = int(selection["K"])
    directory = rebuild_root / model_id
    validation = validate_rebuild_checkpoint(directory)
    if int(selection["selected_seed"]) != int(
        json.loads((directory / "matrix_metadata.json").read_text(encoding="utf-8-sig"))["selected_seed"]
    ):
        raise cleaner.CleaningError(f"{model_id} selected seed 与完整模型不一致")
    if not validation.get("exact_top20_csv_hash_match") or not validation.get(
        "exact_top5_representatives_csv_hash_match"
    ):
        raise cleaner.CleaningError(f"{model_id} 未达到精确复现条件")

    topic_word = load_npz_array(directory / "topic_word_probabilities.npz", "probabilities")
    document_topic = load_npz_array(
        directory / "document_topic_probabilities.npz", "probabilities"
    )
    vocabulary_rows = read_csv(directory / "vocabulary.csv")
    vocabulary_rows.sort(key=lambda row: int(row["term_index"]))
    vocabulary = [row["token"] for row in vocabulary_rows]
    document_rows = read_csv(directory / "document_index.csv")
    document_rows.sort(key=lambda row: int(row["matrix_row"]))
    if corpus_label not in corpus_cache:
        corpus_cache[corpus_label] = read_csv(corpus_root / corpus_label / "lda_corpus.csv")
    corpus_rows = corpus_cache[corpus_label]
    if topic_word.shape != (topic_count, len(vocabulary)):
        raise cleaner.CleaningError(f"{model_id} Topic×Word shape 与索引不一致")
    if document_topic.shape != (len(document_rows), topic_count):
        raise cleaner.CleaningError(f"{model_id} Document×Topic shape 与索引不一致")

    words_by_topic = base.group_topics(
        read_csv(directory / "topics_top30.csv"), topic_count, 30
    )
    representative_raw = base.group_topics(
        read_csv(directory / "representative_poems_top20.csv"), topic_count, 20
    )
    representatives_by_topic: dict[int, list[dict[str, Any]]] = {}
    for topic_id, rows in representative_raw.items():
        representatives_by_topic[topic_id] = [
            {"model_id": model_id, **dict(row)} for row in rows
        ]

    random_by_topic: dict[int, list[dict[str, Any]]] = {}
    for topic_id in range(1, topic_count + 1):
        representative_rows = {
            int(row["matrix_row"]) for row in representatives_by_topic[topic_id]
        }
        random_by_topic[topic_id] = sample_random_high_probability(
            model_id=model_id,
            topic_id=topic_id,
            scores=document_topic[:, topic_id - 1],
            document_rows=document_rows,
            corpus_rows=corpus_rows,
            representative_matrix_rows=representative_rows,
            topic_count=topic_count,
            pool_fraction=float(review["high_probability_pool_fraction"]),
            requested_count=int(review["requested_random_poems"]),
            base_seed=int(review["random_seed"]),
        )
    prevalence_by_topic = compute_prevalence(model_id, document_topic)
    similarity = cosine_similarity_rows(topic_word)
    similar_by_topic, pairs = build_similarity_outputs(
        model_id,
        similarity,
        words_by_topic,
        float(review["high_similarity_threshold"]),
    )
    return CompleteModelData(
        selection=selection,
        topic_word=topic_word,
        document_topic=document_topic,
        vocabulary=vocabulary,
        token_to_index={token: index for index, token in enumerate(vocabulary)},
        document_rows=document_rows,
        corpus_rows=corpus_rows,
        words_by_topic=words_by_topic,
        representatives_by_topic=representatives_by_topic,
        random_by_topic=random_by_topic,
        prevalence_by_topic=prevalence_by_topic,
        similarity=similarity,
        similar_by_topic=similar_by_topic,
        high_similarity_pairs=pairs,
    )


def aligned_cross_similarity(
    first: CompleteModelData, second: CompleteModelData
) -> np.ndarray:
    """在两模型词表并集上对齐完整 Topic×Word 后计算余弦。"""

    union = sorted(set(first.vocabulary) | set(second.vocabulary))
    union_index = {token: index for index, token in enumerate(union)}
    first_aligned = np.zeros((first.topic_word.shape[0], len(union)), dtype=np.float64)
    second_aligned = np.zeros((second.topic_word.shape[0], len(union)), dtype=np.float64)
    first_columns = [union_index[token] for token in first.vocabulary]
    second_columns = [union_index[token] for token in second.vocabulary]
    first_aligned[:, first_columns] = first.topic_word
    second_aligned[:, second_columns] = second.topic_word
    return cross_cosine_similarity(first_aligned, second_aligned)


def write_c_d_comparison(
    comparison_root: Path,
    corpus_root: Path,
    c_model: CompleteModelData,
    d_model: CompleteModelData,
) -> tuple[int, int]:
    """生成 C-K20 与 D-K20 的词表差异、匈牙利匹配和配对审核页。"""

    directory = comparison_root / "C_K20_vs_D_K20"
    cards_dir = directory / "topic_pair_review"
    cards_dir.mkdir(parents=True, exist_ok=True)
    c_vocabulary = {row["token"]: row for row in read_csv(corpus_root / "C" / "vocabulary.csv")}
    d_vocabulary = {row["token"]: row for row in read_csv(corpus_root / "D" / "vocabulary.csv")}
    c_only = [
        {"token": row["token"], "TF": int(row["TF"]), "DF": int(row["DF"])}
        for token, row in c_vocabulary.items()
        if token not in d_vocabulary and int(row["TF"]) in {3, 4}
    ]
    c_only.sort(key=lambda row: (row["TF"], row["DF"], row["token"]))
    cleaner.write_csv(directory / "C_only_low_frequency_tokens.csv", ["token", "TF", "DF"], c_only)

    similarity = aligned_cross_similarity(c_model, d_model)
    c_indices, d_indices = linear_sum_assignment(-similarity)
    rows: list[dict[str, Any]] = []
    for c_index, d_index in sorted(zip(c_indices, d_indices), key=lambda item: item[0]):
        c_topic = int(c_index) + 1
        d_topic = int(d_index) + 1
        c_words = c_model.words_by_topic[c_topic]
        d_words = d_model.words_by_topic[d_topic]
        c_tokens = [row["token"] for row in c_words]
        d_tokens = [row["token"] for row in d_words]
        shared = shared_words(c_words, d_words)
        rows.append(
            {
                "C_topic": c_topic,
                "D_topic": d_topic,
                "similarity": f"{float(similarity[c_index, d_index]):.10f}",
                "shared_top_words": ";".join(shared),
                "C_only_top_words": ";".join(token for token in c_tokens if token not in set(d_tokens)),
                "D_only_top_words": ";".join(token for token in d_tokens if token not in set(c_tokens)),
            }
        )
        lines = [
            f"# C Topic {c_topic:02d} vs D Topic {d_topic:02d}",
            "",
            f"Full-distribution cosine similarity: {float(similarity[c_index, d_index]):.10f}",
            "",
            "## C Top 30 Words",
            "",
            ", ".join(c_tokens),
            "",
            "## D Top 30 Words",
            "",
            ", ".join(d_tokens),
            "",
            "## Shared / Difference",
            "",
            f"- Shared: {', '.join(shared)}",
            f"- C only: {', '.join(token for token in c_tokens if token not in set(d_tokens))}",
            f"- D only: {', '.join(token for token in d_tokens if token not in set(c_tokens))}",
            "",
            "## C Top 20 Representative Poems",
            "",
        ]
        for poem in c_model.representatives_by_topic[c_topic]:
            lines.append(
                f"{poem['rank']}. {poem['author']}《{poem['title']}》 — {poem['topic_probability']} — {poem['original_text']}"
            )
        lines.extend(["", "## D Top 20 Representative Poems", ""])
        for poem in d_model.representatives_by_topic[d_topic]:
            lines.append(
                f"{poem['rank']}. {poem['author']}《{poem['title']}》 — {poem['topic_probability']} — {poem['original_text']}"
            )
        (cards_dir / f"C_topic_{c_topic:02d}__D_topic_{d_topic:02d}.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    cleaner.write_csv(
        directory / "topic_matching.csv",
        ["C_topic", "D_topic", "similarity", "shared_top_words", "C_only_top_words", "D_only_top_words"],
        rows,
    )
    for obsolete in (directory / "MISSING_DATA.md", cards_dir / "README.md"):
        if obsolete.is_file():
            obsolete.unlink()
    return len(c_only), len(rows)


def write_k_mapping(
    directory: Path,
    filename: str,
    model_k20: CompleteModelData,
    model_k30: CompleteModelData,
    split_threshold: float,
) -> tuple[int, int]:
    """为每个 K20 Topic 输出最相似的三个 K30 Topic 并标记可能拆分。"""

    directory.mkdir(parents=True, exist_ok=True)
    similarity = aligned_cross_similarity(model_k20, model_k30)
    rows: list[dict[str, Any]] = []
    split_count = 0
    for topic_index in range(similarity.shape[0]):
        ordered = sorted(
            range(similarity.shape[1]),
            key=lambda index: (-similarity[topic_index, index], index),
        )[:3]
        possible_split = int(np.count_nonzero(similarity[topic_index] >= split_threshold)) >= 2
        split_count += int(possible_split)
        for rank, other in enumerate(ordered, 1):
            rows.append(
                {
                    "K20_topic": topic_index + 1,
                    "K30_topic_rank": rank,
                    "K30_topic": other + 1,
                    "similarity": f"{float(similarity[topic_index, other]):.10f}",
                    "shared_words": ";".join(
                        shared_words(
                            model_k20.words_by_topic[topic_index + 1],
                            model_k30.words_by_topic[other + 1],
                        )
                    ),
                    "possible_split": str(possible_split),
                    "split_threshold": f"{split_threshold:.4f}",
                }
            )
    cleaner.write_csv(directory / filename, list(rows[0].keys()), rows)
    obsolete = directory / "MISSING_DATA.md"
    if obsolete.is_file():
        obsolete.unlink()
    return len(rows), split_count


def complete_readme(
    selections: Sequence[Mapping[str, Any]],
    review: Mapping[str, Any],
    high_pair_counts: Mapping[str, int],
) -> str:
    """生成完整审核包说明。"""

    lines = [
        "# 《全唐诗》六个候选 LDA 模型完整人工 Topic 审核包",
        "",
        "## 1. 模型来源与复现",
        "",
        "六个模型均使用第一轮原 BoW、原 scikit-learn Batch LDA 参数和已选 seed 重新拟合。每个模型只有在重建 top-20 主题词 CSV 与第一轮文件 SHA-256 完全一致、重建 top-5 代表诗 CSV 也完全一致，并通过指标与矩阵回读检查后，才进入本审核包。",
        "",
        "## 2. Selected seed",
        "",
        "每个 Corpus×K 仍按 NPMI 最接近三个 seed 平均值选择，距离并列时取较小 seed；没有改选单次最高值。",
        "",
    ]
    for row in selections:
        lines.append(
            f"- `{row['model_id']}`：seed `{row['selected_seed']}`；mean NPMI={row['mean_npmi']}；selected NPMI={row['selected_seed_npmi']}；distance={row['distance_from_mean']}；similarity≥{float(review['high_similarity_threshold']):.2f} pairs={high_pair_counts[row['model_id']]}。"
        )
    lines.extend(
        [
            "",
            "## 3. Top Words 与 Representative Poems",
            "",
            "Top 30 Words 直接由完整归一化 Topic×Word 矩阵按概率降序提取；概率并列时按 term_index。Top 20 Representative Poems 由完整 Document×Topic 矩阵按 P(topic|poem) 降序提取；并列时按 matrix_row。",
            "",
            "## 4. Random High-Probability Poems",
            "",
            f"每个 Topic 先取概率排名前 {float(review['high_probability_pool_fraction']) * 100:.0f}% 且 P≥1/K 的诗，排除 Top 20 代表诗，再从候选池抽取至多 {review['requested_random_poems']} 首。全局种子为 {review['random_seed']}，并通过 model_id、topic_id 派生稳定子种子。",
            "",
            "## 5. Topic Prevalence",
            "",
            "soft prevalence 为全部有效诗 P(topic|poem) 的均值；dominant prevalence 使用每诗最大概率 Topic（并列取较小 topic_id）。同时提供 ≥0.2、≥0.3、≥0.5、≥2/K、≥3/K 计数。",
            "",
            "## 6. Topic Similarity 与比较材料",
            "",
            "模型内及跨模型相似度均使用完整 Topic×Word 概率分布的 cosine similarity。跨语料比较先按 token 在词表并集上对齐；C-K20 vs D-K20 使用 Hungarian 一对一最佳匹配。K20 vs K30 为每个 K20 Topic 保留 Top 3，若其对至少两个 K30 Topic 的 similarity ≥0.40，则标记 possible_split。",
            "",
            "## 7. 人工评分",
            "",
            "Semantic Coherence、Poem Coherence、Interpretability、Distinctiveness 均为 1–5 分；具体锚点已写在每张审核卡。请把人工标签、评分、True/False 标记和备注填入 `human_topic_evaluation.csv`，程序不会自动命名或自动判断好坏。",
            "",
            "## 8. 评分后汇总",
            "",
            "```powershell",
            "python src/build_human_topic_review.py --config config/human_topic_review_config.yaml --summary-only",
            "```",
            "",
            "只有四项分数完整且均在 1–5 时才纳入模型均值；interpretability_score≤2 计为 uninterpretable Topic。",
            "",
        ]
    )
    return "\n".join(lines)


def generate_complete_package(config_path: Path, project_root: Path) -> dict[str, Any]:
    """读取六个已验证完整模型并重新生成完整审核包。"""

    config = cleaner.load_config(config_path)
    paths = cleaner.require_mapping(config.get("paths"), "paths")
    review = cleaner.require_mapping(config.get("review"), "review")
    candidates_value = config.get("candidates")
    if not isinstance(candidates_value, list) or len(candidates_value) != 6:
        raise cleaner.CleaningError("candidates 必须恰含六个模型")
    candidates = [cleaner.require_mapping(value, "candidate") for value in candidates_value]
    model_metrics_path = resolve_path(project_root, paths.get("model_metrics"), "paths.model_metrics")
    stability_path = resolve_path(project_root, paths.get("stability_metrics"), "paths.stability_metrics")
    rebuild_root = resolve_path(project_root, paths.get("rebuild_root"), "paths.rebuild_root")
    rebuild_metadata_path = resolve_path(
        project_root, paths.get("rebuild_metadata"), "paths.rebuild_metadata"
    )
    corpus_root = resolve_path(project_root, paths.get("corpus_root"), "paths.corpus_root")
    output_root = resolve_path(project_root, paths.get("output_root"), "paths.output_root")
    log_path = resolve_path(project_root, paths.get("log_path"), "paths.log_path")
    logger = base.setup_logger(log_path)
    rebuild_metadata = json.loads(rebuild_metadata_path.read_text(encoding="utf-8-sig"))
    if not rebuild_metadata.get("all_reproduction_validations_passed"):
        raise cleaner.CleaningError("六候选重建总审计未通过")

    model_metrics = read_csv(model_metrics_path)
    stability_metrics = read_csv(stability_path)
    selections: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        selection, candidate_seed_rows, candidate_stability_rows = base.select_seed(
            candidate, model_metrics, stability_metrics
        )
        selections.append(selection)
        seed_rows.extend(candidate_seed_rows)
        stability_rows.extend(candidate_stability_rows)

    input_paths: set[Path] = {
        config_path,
        Path(__file__).resolve(),
        project_root / "src" / "build_human_topic_review.py",
        model_metrics_path,
        stability_path,
        rebuild_metadata_path,
    }
    for selection in selections:
        directory = rebuild_root / str(selection["model_id"])
        input_paths.update(path for path in directory.iterdir() if path.is_file())
        input_paths.add(corpus_root / str(selection["corpus"]) / "lda_corpus.csv")
        input_paths.add(corpus_root / str(selection["corpus"]) / "vocabulary.csv")
    input_hashes_before = {str(path): cleaner.file_sha256(path) for path in input_paths}

    output_root.mkdir(parents=True, exist_ok=True)
    cleaner.write_csv(output_root / "model_seed_metrics.csv", list(seed_rows[0].keys()), seed_rows)
    cleaner.write_csv(
        output_root / "model_stability_metrics.csv", list(stability_rows[0].keys()), stability_rows
    )
    corpus_cache: dict[str, list[dict[str, str]]] = {}
    models: dict[str, CompleteModelData] = {}
    for selection in selections:
        model = load_complete_model(
            selection=selection,
            rebuild_root=rebuild_root,
            corpus_root=corpus_root,
            review=review,
            corpus_cache=corpus_cache,
        )
        models[str(selection["model_id"])] = model

    evaluation_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    completeness: list[dict[str, Any]] = []
    high_pair_counts: dict[str, int] = {}
    review_card_counts: dict[str, int] = {}
    for model_id, model in models.items():
        topic_count = int(model.selection["K"])
        model_dir = output_root / model_id
        cards_dir = model_dir / "review_cards"
        cards_dir.mkdir(parents=True, exist_ok=True)
        top_word_rows = [
            {
                "model_id": model_id,
                "topic_id": int(row["topic_id"]),
                "rank": int(row["rank"]),
                "term_index": int(row["term_index"]),
                "token": row["token"],
                "weight": row["probability"],
                "probability": row["probability"],
                "component_weight": row["component_weight"],
                "requested_top_n": 30,
                "available_top_n": 30,
                "completeness_status": "complete_rebuilt_matrix",
            }
            for topic_id in sorted(model.words_by_topic)
            for row in model.words_by_topic[topic_id]
        ]
        cleaner.write_csv(model_dir / "topic_top_words.csv", list(top_word_rows[0].keys()), top_word_rows)
        (model_dir / "topic_top_words.md").write_text(
            top_words_markdown(model_id, int(model.selection["selected_seed"]), model.words_by_topic),
            encoding="utf-8",
        )
        representative_rows = [
            row
            for topic_id in sorted(model.representatives_by_topic)
            for row in model.representatives_by_topic[topic_id]
        ]
        representative_fields = [
            "model_id", "topic_id", "rank", "matrix_row", "corpus_row", "poem_id",
            "author", "title", "original_text", "cleaned_tokens", "topic_probability"
        ]
        cleaner.write_csv(
            model_dir / "topic_top_representative_poems.csv", representative_fields, representative_rows
        )
        random_rows = [
            row for topic_id in sorted(model.random_by_topic) for row in model.random_by_topic[topic_id]
        ]
        cleaner.write_csv(
            model_dir / "topic_random_high_probability_poems.csv",
            list(random_rows[0].keys()),
            random_rows,
        )
        prevalence_rows = [model.prevalence_by_topic[topic_id] for topic_id in sorted(model.prevalence_by_topic)]
        cleaner.write_csv(model_dir / "topic_prevalence.csv", list(prevalence_rows[0].keys()), prevalence_rows)
        similarity_fields = ["model_id", "topic_id"] + [f"topic_{index:02d}" for index in range(1, topic_count + 1)]
        similarity_rows = []
        for topic_index in range(topic_count):
            row = {"model_id": model_id, "topic_id": topic_index + 1}
            row.update(
                {
                    f"topic_{other + 1:02d}": f"{float(model.similarity[topic_index, other]):.10f}"
                    for other in range(topic_count)
                }
            )
            similarity_rows.append(row)
        cleaner.write_csv(model_dir / "topic_similarity_matrix.csv", similarity_fields, similarity_rows)
        pair_fields = ["model_id", "topic_a", "topic_b", "cosine_similarity", "shared_top_words"]
        cleaner.write_csv(
            model_dir / "high_similarity_topic_pairs.csv", pair_fields, model.high_similarity_pairs
        )
        high_pair_counts[model_id] = len(model.high_similarity_pairs)

        obsolete = model_dir / "MISSING_DATA.md"
        if obsolete.is_file():
            obsolete.unlink()
        for topic_id in range(1, topic_count + 1):
            (cards_dir / f"{model_id}_topic_{topic_id:02d}.md").write_text(
                complete_review_card(model, topic_id), encoding="utf-8"
            )
            probabilities = [
                float(row["topic_probability"])
                for row in model.representatives_by_topic[topic_id]
            ]
            diagnostics.append(
                {
                    "model_id": model_id,
                    "topic_id": topic_id,
                    "representative_count": len(probabilities),
                    "top1_probability": f"{max(probabilities):.10f}",
                    "mean_top20_probability": f"{float(np.mean(probabilities)):.10f}",
                    "min_top20_probability": f"{min(probabilities):.10f}",
                    "uniform_probability_baseline": f"{1 / topic_count:.10f}",
                    "top1_below_0_5": str(max(probabilities) < 0.5),
                    "mean_top20_below_0_5": str(float(np.mean(probabilities)) < 0.5),
                    "diagnostic_scope": "complete_top20_from_document_topic_matrix",
                }
            )
            evaluation_rows.append(
                {
                    "model_id": model_id,
                    "corpus": model.selection["corpus"],
                    "K": topic_count,
                    "selected_seed": model.selection["selected_seed"],
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
        selection_rows.append(
            {
                **model.selection,
                "full_model_object_available": "True",
                "full_topic_word_distribution_available": "True",
                "full_document_topic_probability_available": "True",
                "available_top_words_per_topic": 30,
                "available_representative_poems_per_topic": 20,
                "random_poems_per_topic": min(len(rows) for rows in model.random_by_topic.values()),
                "review_card_count": topic_count,
                "high_similarity_pair_count": len(model.high_similarity_pairs),
                "high_similarity_pair_status": "complete_full_topic_word_cosine",
                "reproduction_validation": "exact_top20_and_top5_csv_hash_match",
            }
        )
        for capability in (
            "model_object", "topic_top_words", "representative_poems",
            "random_high_probability_poems", "topic_prevalence", "topic_similarity"
        ):
            completeness.append(
                {
                    "model_id": model_id,
                    "capability": capability,
                    "status": "complete",
                    "source": "exactly_rebuilt_selected_seed_model",
                }
            )
        logger.info(
            "%s 完整审核材料完成：cards=%d high_similarity_pairs=%d",
            model_id,
            topic_count,
            len(model.high_similarity_pairs),
        )

    cleaner.write_csv(output_root / "model_selection_summary.csv", list(selection_rows[0].keys()), selection_rows)
    cleaner.write_csv(output_root / "data_completeness_report.csv", list(completeness[0].keys()), completeness)
    cleaner.write_csv(
        output_root / "missing_data_report.csv",
        ["model_id", "capability", "requested", "available", "status", "reason"],
        [],
    )
    cleaner.write_csv(
        output_root / "representative_probability_diagnostics.csv", list(diagnostics[0].keys()), diagnostics
    )
    old_diagnostics = output_root / "available_representative_probability_diagnostics.csv"
    if old_diagnostics.is_file():
        old_diagnostics.unlink()

    evaluation_path = output_root / "human_topic_evaluation.csv"
    if evaluation_path.is_file():
        existing = read_csv(evaluation_path)
        expected_keys = {(row["model_id"], str(row["topic_id"])) for row in evaluation_rows}
        existing_keys = {(row["model_id"], str(row["topic_id"])) for row in existing}
        if existing_keys != expected_keys:
            raise cleaner.CleaningError("既有人工评分表行集合与完整审核包不一致")
        evaluation_for_summary = existing
    else:
        cleaner.write_csv(evaluation_path, list(evaluation_rows[0].keys()), evaluation_rows)
        evaluation_for_summary = [{key: str(value) for key, value in row.items()} for row in evaluation_rows]
    summary_rows = base.summarize_human_scores(evaluation_for_summary, candidates)
    cleaner.write_csv(
        output_root / "model_human_evaluation_summary.csv", list(summary_rows[0].keys()), summary_rows
    )

    comparison_root = output_root / "comparison"
    c_only_count, matching_count = write_c_d_comparison(
        comparison_root, corpus_root, models["C_K20"], models["D_K20"]
    )
    b_mapping_rows, b_split_count = write_k_mapping(
        comparison_root / "B_K20_vs_B_K30",
        "B_K20_vs_K30_topic_mapping.csv",
        models["B_K20"],
        models["B_K30"],
        float(review["split_similarity_threshold"]),
    )
    d_mapping_rows, d_split_count = write_k_mapping(
        comparison_root / "D_K20_vs_D_K30",
        "D_K20_vs_K30_topic_mapping.csv",
        models["D_K20"],
        models["D_K30"],
        float(review["split_similarity_threshold"]),
    )
    comparison_status = [
        {
            "comparison": "C_K20_vs_D_K20_topic_matching",
            "status": "complete",
            "method": "Hungarian matching on full union-aligned Topic×Word cosine",
            "output_rows": matching_count,
            "split_candidates": "",
        },
        {
            "comparison": "B_K20_vs_B_K30_topic_mapping",
            "status": "complete",
            "method": "Top 3 full Topic×Word cosine per K20 topic",
            "output_rows": b_mapping_rows,
            "split_candidates": b_split_count,
        },
        {
            "comparison": "D_K20_vs_D_K30_topic_mapping",
            "status": "complete",
            "method": "Top 3 full Topic×Word cosine per K20 topic",
            "output_rows": d_mapping_rows,
            "split_candidates": d_split_count,
        },
    ]
    cleaner.write_csv(
        comparison_root / "comparison_status.csv", list(comparison_status[0].keys()), comparison_status
    )
    (output_root / "README.md").write_text(
        complete_readme(selection_rows, review, high_pair_counts), encoding="utf-8"
    )

    input_hashes_after = {str(path): cleaner.file_sha256(path) for path in input_paths}
    if input_hashes_before != input_hashes_after:
        raise cleaner.CleaningError("生成完整审核包时重建模型或语料输入发生变化")
    output_files = sorted(
        [path for path in output_root.rglob("*") if path.is_file() and path.name != "generation_metadata.json"],
        key=lambda path: str(path),
    )
    metadata = {
        "schema_version": 2,
        "method": "complete_human_topic_review_from_exactly_rebuilt_models",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "candidate_rebuild_validated": True,
        "all_six_exact_reproductions_passed": True,
        "lda_training_performed_during_package_generation": False,
        "corpus_modified": False,
        "automatic_topic_labels_generated": False,
        "automatic_human_scores_generated": False,
        "candidate_count": len(models),
        "topic_review_card_count": sum(review_card_counts.values()),
        "review_card_counts": review_card_counts,
        "selected_seeds": {row["model_id"]: row["selected_seed"] for row in selection_rows},
        "availability": {
            "top_words_per_topic": 30,
            "representative_poems_per_topic": 20,
            "random_high_probability_poems_per_topic": 20,
            "full_prevalence": True,
            "full_topic_word_distribution": True,
            "full_document_topic_probability": True,
            "model_objects": True,
            "topic_similarity": True,
            "cross_model_topic_matching": True,
        },
        "high_similarity_pair_counts": high_pair_counts,
        "C_only_TF3_or_TF4_token_count": c_only_count,
        "C_D_hungarian_match_count": matching_count,
        "B_possible_split_topic_count": b_split_count,
        "D_possible_split_topic_count": d_split_count,
        "validation": {
            "input_hashes_unchanged": input_hashes_before == input_hashes_after,
            "review_card_count_matches_sum_K": sum(review_card_counts.values()) == 140,
            "all_topics_have_top30": all(len(rows) == 30 for model in models.values() for rows in model.words_by_topic.values()),
            "all_topics_have_top20_representatives": all(len(rows) == 20 for model in models.values() for rows in model.representatives_by_topic.values()),
            "all_topics_have_random_sample": all(len(rows) == 20 for model in models.values() for rows in model.random_by_topic.values()),
            "all_complete_capabilities_present": True,
        },
        "inputs": [
            {"path": cleaner.display_path(path), "sha256": input_hashes_before[str(path)]}
            for path in sorted(input_paths, key=lambda item: str(item))
        ],
        "outputs": [
            {"path": cleaner.display_path(path), "sha256": cleaner.file_sha256(path), "bytes": path.stat().st_size}
            for path in output_files
        ],
    }
    cleaner.write_json(output_root / "generation_metadata.json", metadata)
    logger.info(
        "完整审核包完成：models=6 cards=140 C/D matches=%d B splits=%d D splits=%d",
        matching_count,
        b_split_count,
        d_split_count,
    )
    return metadata
