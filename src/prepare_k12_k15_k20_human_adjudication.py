"""准备 K12/K15/K20 研究者主导的人文学盲审材料。

本程序只读取已冻结的 full-corpus 模型矩阵。它不会导入训练器、调用 fit、
使用旧 AI 标签筛选证据，或填写任何 Researcher_* 字段。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "model_selection/K12_K15_K20_human_adjudication/config/human_adjudication_config.yaml"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def load_dense_npz(path: Path, preferred_keys: Iterable[str]) -> np.ndarray:
    try:
        return np.asarray(sparse.load_npz(path).toarray(), dtype=np.float64)
    except (ValueError, TypeError, KeyError):
        pass
    with np.load(path, allow_pickle=False) as data:
        for key in preferred_keys:
            if key in data.files:
                return np.asarray(data[key], dtype=np.float64)
        if len(data.files) == 1:
            return np.asarray(data[data.files[0]], dtype=np.float64)
    raise ValueError(f"无法确定 NPZ 数组键：{path}")


def normalize_rows(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("topic-word 分布存在非正行和")
    return values / totals


def unit_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("topic-word 分布存在零范数行")
    return values / norms


def stable_top_indices(values: np.ndarray, count: int) -> np.ndarray:
    return np.argsort(-values, axis=1, kind="mergesort")[:, :count]


@dataclass(frozen=True)
class FrozenModel:
    k: int
    seed: int
    topic_word: np.ndarray
    topic_unit: np.ndarray
    theta: np.ndarray
    manifest_row: dict[str, Any]


def load_corpus(corpus_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    vocabulary = pd.read_csv(corpus_root / "vocabulary.csv").sort_values("term_index").reset_index(drop=True)
    index = pd.read_csv(corpus_root / "document_index.csv")
    corpus = pd.read_csv(corpus_root / "lda_corpus.csv")
    documents = index.merge(
        corpus[["poem_id", "original_text", "cleaned_tokens"]],
        on="poem_id", how="left", validate="one_to_one",
    ).sort_values("matrix_row").reset_index(drop=True)
    if len(vocabulary) != 35063 or len(documents) != 43015:
        raise ValueError("冻结 C corpus 的 shape 不符合登记值")
    if not np.array_equal(documents.matrix_row.to_numpy(), np.arange(43015)):
        raise ValueError("冻结 C corpus 的 matrix_row 顺序异常")
    if documents.original_text.isna().any():
        raise ValueError("代表诗正文映射存在缺失")
    return vocabulary, documents


def load_models(config: dict[str, Any]) -> tuple[dict[int, FrozenModel], pd.DataFrame, list[dict[str, str]]]:
    manifest_path = ROOT / config["ready_manifest"]
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest.K.isin(config["candidate_K"])].sort_values("K").reset_index(drop=True)
    if manifest.K.tolist() != config["candidate_K"] or not manifest.ready_for_blind_audit.all():
        raise ValueError("三模型未全部处于 blind-audit-ready 状态")
    input_rows = [{"role": "ready_manifest", "path": relative(manifest_path), "sha256": sha256(manifest_path)}]
    models: dict[int, FrozenModel] = {}
    for record in manifest.to_dict("records"):
        k = int(record["K"])
        required = {
            "model": ROOT / record["model_path"],
            "components": ROOT / record["components_path"],
            "theta": ROOT / record["theta_full_path"],
            "representative_poems": ROOT / record["representative_poems_path"],
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"K{k} 缺少冻结产物：{missing}")
        for role, path in required.items():
            input_rows.append({"role": f"K{k}_{role}", "path": relative(path), "sha256": sha256(path)})
        components = load_dense_npz(required["components"], ["components"])
        theta = load_dense_npz(required["theta"], ["doc_topic_full", "probabilities"])
        if components.shape != (k, 35063) or theta.shape != (43015, k):
            raise ValueError(f"K{k} 冻结矩阵 shape 异常：{components.shape}, {theta.shape}")
        if not np.isfinite(components).all() or not np.isfinite(theta).all() or np.any(theta < 0):
            raise ValueError(f"K{k} 冻结矩阵含非法值")
        if float(np.max(np.abs(theta.sum(axis=1) - 1.0))) > 1e-10:
            raise ValueError(f"K{k} document-topic 行和异常")
        topic_word = normalize_rows(components)
        models[k] = FrozenModel(k, int(record["seed"]), topic_word, unit_rows(topic_word), theta, record)
    return models, manifest, input_rows


def build_comparability(manifest: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    corpus_hashes = set(manifest.corpus_hash.astype(str))
    vocab_hashes = set(manifest.vocab_hash.astype(str))
    scopes = set(manifest.fit_scope.astype(str))
    priors = set(manifest.prior_regime.astype(str))
    iterations = set(manifest.max_iter.astype(int))
    common_ready = (
        corpus_hashes == {"744baeb001f1c5fa43802f07bceb73c6142b605ff3bf69c12e3f14d5dc8eb779"}
        and vocab_hashes == {"a816c82fe379e2abfadc55b4624b056bb0a7b7c2e6a4e0cddff8e9f386c10084"}
        and scopes == {"FULL_CORPUS"}
        and priors == {"ORIGINAL_PRIOR"}
        and iterations == {30}
    )
    for row in manifest.itertuples(index=False):
        rows.append({
            "K": int(row.K),
            "representative_seed": int(row.seed),
            "corpus_hash": row.corpus_hash,
            "vocabulary_hash": row.vocab_hash,
            "preprocessing": "C_CORPUS_TF_GE_3_DF_GE_3_CORE_STOPWORDS_PROTECTED_TERMS",
            "fit_scope": row.fit_scope,
            "prior_regime": row.prior_regime,
            "max_iter_convention": int(row.max_iter),
            "representative_seed_selection_principle": "PREDETERMINED_FORMAL_REPRESENTATIVE_NOT_BEST_RUN_CHERRY_PICKING",
            "model_path": row.model_path,
            "comparability_status": "READY" if common_ready else "BLOCKED",
        })
    result = pd.DataFrame(rows)
    write_csv(result, output / "data/three_model_comparability_audit.csv")
    if not common_ready:
        raise ValueError("三模型口径审计未通过")
    return result


def make_mapping(models: dict[int, FrozenModel], config: dict[str, Any], private: Path) -> pd.DataFrame:
    path = private / "blind_topic_mapping_secret.csv"
    topics = [(k, topic_id) for k in config["candidate_K"] for topic_id in range(1, k + 1)]
    random.Random(int(config["topic_mapping_seed"])).shuffle(topics)
    rows = []
    for order, (k, topic_id) in enumerate(topics, start=1):
        rows.append({
            "blind_topic_id": f"HR{order:03d}",
            "true_K": k,
            "true_topic_id": topic_id,
            "representative_seed": models[k].seed,
            "mapping_seed": int(config["topic_mapping_seed"]),
        })
    expected = pd.DataFrame(rows)
    if len(expected) != 47:
        raise AssertionError("匿名 Topic 总数不是47")
    if path.exists():
        existing = pd.read_csv(path)
        if not existing.equals(expected):
            raise ValueError("既有匿名映射与固定 seed 不一致")
        return existing
    write_csv(expected, path)
    return expected


def poem_record(blind_id: str, sample_type: str, rank: int, matrix_row: int, model: FrozenModel, topic_index: int, documents: pd.DataFrame) -> dict[str, Any]:
    doc = documents.iloc[matrix_row]
    return {
        "blind_topic_id": blind_id,
        "sample_type": sample_type,
        "rank": rank,
        "poem_id": int(doc.poem_id),
        "author": str(doc.author),
        "title": str(doc.title),
        "topic_probability": float(model.theta[matrix_row, topic_index]),
        "original_text": str(doc.original_text),
    }


def build_blind_evidence(
    models: dict[int, FrozenModel], mapping: pd.DataFrame, vocabulary: pd.DataFrame,
    documents: pd.DataFrame, config: dict[str, Any], output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tokens = vocabulary.token.astype(str).to_numpy()
    mapping_lookup = {(int(row.true_K), int(row.true_topic_id)): str(row.blind_topic_id) for row in mapping.itertuples(index=False)}
    word_rows: list[dict[str, Any]] = []
    poem_rows: list[dict[str, Any]] = []
    topic_rows: list[dict[str, Any]] = []
    for k in config["candidate_K"]:
        model = models[k]
        word_indices = stable_top_indices(model.topic_word, int(config["top_word_count"]))
        for topic_index in range(k):
            blind_id = mapping_lookup[(k, topic_index + 1)]
            topic_words = []
            for rank, term_index in enumerate(word_indices[topic_index], start=1):
                word = str(tokens[term_index])
                topic_words.append(word)
                word_rows.append({
                    "blind_topic_id": blind_id,
                    "rank": rank,
                    "word": word,
                    "topic_weight": float(model.topic_word[topic_index, term_index]),
                })
            order = np.argsort(-model.theta[:, topic_index], kind="mergesort")
            raw_rows = [int(value) for value in order[: int(config["raw_poem_count"])]]
            balanced_rows: list[int] = []
            author_counts: Counter[str] = Counter()
            for matrix_row in order:
                author = str(documents.iloc[int(matrix_row)].author)
                if author_counts[author] >= int(config["author_maximum_per_topic"]):
                    continue
                balanced_rows.append(int(matrix_row))
                author_counts[author] += 1
                if len(balanced_rows) == int(config["author_balanced_poem_count"]):
                    break
            if len(balanced_rows) != 10 or len(author_counts) < 5:
                raise ValueError(f"{blind_id} 作者平衡样本不足")
            for rank, matrix_row in enumerate(raw_rows, start=1):
                poem_rows.append(poem_record(blind_id, "RAW_TOP20", rank, matrix_row, model, topic_index, documents))
            for rank, matrix_row in enumerate(balanced_rows, start=1):
                poem_rows.append(poem_record(blind_id, "AUTHOR_BALANCED_TOP10", rank, matrix_row, model, topic_index, documents))
            raw_summary = "；".join(
                f"{documents.iloc[matrix_row].author}《{documents.iloc[matrix_row].title}》"
                for matrix_row in raw_rows[:5]
            )
            topic_rows.append({
                "blind_topic_id": blind_id,
                "Top30_words": "、".join(topic_words),
                "representative_poem_summary": raw_summary,
                "Researcher_Label": "",
                "Researcher_Word_Coherence": "",
                "Researcher_Poem_Coherence": "",
                "Researcher_Word_Poem_Agreement": "",
                "Researcher_Labelability": "",
                "Researcher_Literary_Interpretability": "",
                "Researcher_Semantic_Specificity": "",
                "Researcher_Overbreadth": "",
                "Researcher_Internal_Mixedness": "",
                "Researcher_Topic_Type": "",
                "Researcher_Needs_Splitting": "",
                "Researcher_Suspected_Subthemes": "",
                "Researcher_Confidence": "",
                "Researcher_Notes": "",
            })
    words = pd.DataFrame(word_rows).sort_values(["blind_topic_id", "rank"])
    poems = pd.DataFrame(poem_rows).sort_values(["blind_topic_id", "sample_type", "rank"])
    topics = pd.DataFrame(topic_rows).sort_values("blind_topic_id")
    write_csv(words, output / "data/blind_topic_words.csv")
    write_csv(poems, output / "data/blind_poem_evidence.csv")
    write_csv(topics, output / "data/researcher_topic_review_template.csv")
    return topics, words, poems


def write_cards(topics: pd.DataFrame, words: pd.DataFrame, poems: pd.DataFrame, output: Path) -> None:
    cards = output / "human_review_cards"
    cards.mkdir(parents=True, exist_ok=True)
    for row in topics.itertuples(index=False):
        blind_id = str(row.blind_topic_id)
        topic_words = words[words.blind_topic_id == blind_id].sort_values("rank")
        raw = poems[(poems.blind_topic_id == blind_id) & (poems.sample_type == "RAW_TOP20")].sort_values("rank")
        balanced = poems[(poems.blind_topic_id == blind_id) & (poems.sample_type == "AUTHOR_BALANCED_TOP10")].sort_values("rank")
        lines = [
            f"# {blind_id}", "", "## Top30 topic words", "",
            "| Rank | Word | Topic weight |", "|---:|---|---:|",
        ]
        lines.extend(f"| {int(item.rank)} | {item.word} | {float(item.topic_weight):.10f} |" for item in topic_words.itertuples(index=False))
        lines.extend(["", "## Top20 representative poems", ""])
        for item in raw.itertuples(index=False):
            lines.extend([
                f"### {int(item.rank)}. poem_id={int(item.poem_id)}｜{item.author}｜{item.title}｜P={float(item.topic_probability):.6f}",
                "", str(item.original_text), "",
            ])
        lines.extend(["## Top10 author-balanced representative poems", "", "同一作者最多2首，至少覆盖5位作者。", ""])
        for item in balanced.itertuples(index=False):
            lines.extend([
                f"### {int(item.rank)}. poem_id={int(item.poem_id)}｜{item.author}｜{item.title}｜P={float(item.topic_probability):.6f}",
                "", str(item.original_text), "",
            ])
        (cards / f"{blind_id}.md").write_text("\n".join(lines), encoding="utf-8")


def jaccard(left: set[int], right: set[int]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def topic_summary(k: int, topic_id: int, mapping_lookup: dict[tuple[int, int], str], words: pd.DataFrame, poems: pd.DataFrame) -> dict[str, str]:
    blind_id = mapping_lookup[(k, topic_id)]
    topic_words = words[words.blind_topic_id == blind_id].sort_values("rank")
    raw = poems[(poems.blind_topic_id == blind_id) & (poems.sample_type == "RAW_TOP20")].sort_values("rank")
    balanced = poems[(poems.blind_topic_id == blind_id) & (poems.sample_type == "AUTHOR_BALANCED_TOP10")].sort_values("rank")
    poem_text = lambda frame: "；".join(
        f"{int(row.poem_id)}:{row.author}《{row.title}》" for row in frame.head(10).itertuples(index=False)
    )
    return {
        "blind_topic_id": blind_id,
        "top20_words": "、".join(topic_words.head(20).word.astype(str)),
        "top10_poems": poem_text(raw),
        "author_balanced_top10": poem_text(balanced),
    }


def build_transition_candidates(
    lower_k: int, higher_k: int, models: dict[int, FrozenModel], mapping: pd.DataFrame,
    words: pd.DataFrame, poems: pd.DataFrame, config: dict[str, Any], output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lower = models[lower_k]
    higher = models[higher_k]
    cosine = lower.topic_unit @ higher.topic_unit.T
    lower_top20 = stable_top_indices(lower.topic_word, 20)
    higher_top20 = stable_top_indices(higher.topic_word, 20)
    lower_top_poems = stable_top_indices(lower.theta.T, 20)
    higher_top_poems = stable_top_indices(higher.theta.T, 20)
    mapping_lookup = {(int(row.true_K), int(row.true_topic_id)): str(row.blind_topic_id) for row in mapping.itertuples(index=False)}
    pair_rows = []
    for lower_index in range(lower_k):
        rank_order = np.argsort(-cosine[lower_index], kind="mergesort")
        for rank, higher_index in enumerate(rank_order, start=1):
            top_word_overlap = jaccard(set(map(int, lower_top20[lower_index])), set(map(int, higher_top20[higher_index])))
            poem_overlap = jaccard(set(map(int, lower_top_poems[lower_index])), set(map(int, higher_top_poems[higher_index])))
            if rank <= int(config["alignment_top_n"]) or cosine[lower_index, higher_index] >= float(config["split_cosine_min"]):
                pair_rows.append({
                    "transition": f"K{lower_k}_TO_K{higher_k}",
                    "lower_topic_id": lower_index + 1,
                    "lower_blind_topic_id": mapping_lookup[(lower_k, lower_index + 1)],
                    "higher_topic_id": higher_index + 1,
                    "higher_blind_topic_id": mapping_lookup[(higher_k, higher_index + 1)],
                    "candidate_rank_within_lower_topic": rank,
                    "topic_word_cosine": float(cosine[lower_index, higher_index]),
                    "top20_word_jaccard": top_word_overlap,
                    "top20_representative_poem_jaccard": poem_overlap,
                    "candidate_use": "ALIGNMENT_HINT_ONLY_RESEARCHER_DECIDES",
                })
    pairs = pd.DataFrame(pair_rows).sort_values(["lower_topic_id", "candidate_rank_within_lower_topic"])
    events = []
    event_number = 1
    minimum = float(config["split_cosine_min"])
    relative = float(config["split_relative_to_best"])
    for lower_index in range(lower_k):
        values = cosine[lower_index]
        selected = np.where((values >= minimum) & (values >= relative * values.max()))[0]
        if len(selected) >= 2:
            selected = selected[np.argsort(-values[selected], kind="mergesort")]
            lower_summary = topic_summary(lower_k, lower_index + 1, mapping_lookup, words, poems)
            high_summaries = [topic_summary(higher_k, int(index) + 1, mapping_lookup, words, poems) for index in selected]
            events.append({
                "candidate_event_id": f"K{lower_k}K{higher_k}_C{event_number:02d}",
                "relation_candidate": "SPLIT",
                "lower_topic_ids": str(lower_index + 1),
                "lower_blind_topic_ids": lower_summary["blind_topic_id"],
                "higher_topic_ids": "|".join(str(int(index) + 1) for index in selected),
                "higher_blind_topic_ids": "|".join(item["blind_topic_id"] for item in high_summaries),
                "source_top20_words": lower_summary["top20_words"],
                "target_top20_words": " || ".join(item["top20_words"] for item in high_summaries),
                "source_top10_poems": lower_summary["top10_poems"],
                "target_top10_poems": " || ".join(item["top10_poems"] for item in high_summaries),
                "source_author_balanced_top10": lower_summary["author_balanced_top10"],
                "target_author_balanced_top10": " || ".join(item["author_balanced_top10"] for item in high_summaries),
                "maximum_topic_word_cosine": float(values[selected].max()),
                "minimum_selected_topic_word_cosine": float(values[selected].min()),
                "Researcher_Split_Judgment": "",
                "Researcher_Split_Rationale": "",
                "Researcher_Confidence": "",
            })
            event_number += 1
    for higher_index in range(higher_k):
        values = cosine[:, higher_index]
        selected = np.where((values >= minimum) & (values >= relative * values.max()))[0]
        if len(selected) >= 2:
            selected = selected[np.argsort(-values[selected], kind="mergesort")]
            low_summaries = [topic_summary(lower_k, int(index) + 1, mapping_lookup, words, poems) for index in selected]
            higher_summary = topic_summary(higher_k, higher_index + 1, mapping_lookup, words, poems)
            events.append({
                "candidate_event_id": f"K{lower_k}K{higher_k}_C{event_number:02d}",
                "relation_candidate": "MERGE",
                "lower_topic_ids": "|".join(str(int(index) + 1) for index in selected),
                "lower_blind_topic_ids": "|".join(item["blind_topic_id"] for item in low_summaries),
                "higher_topic_ids": str(higher_index + 1),
                "higher_blind_topic_ids": higher_summary["blind_topic_id"],
                "source_top20_words": " || ".join(item["top20_words"] for item in low_summaries),
                "target_top20_words": higher_summary["top20_words"],
                "source_top10_poems": " || ".join(item["top10_poems"] for item in low_summaries),
                "target_top10_poems": higher_summary["top10_poems"],
                "source_author_balanced_top10": " || ".join(item["author_balanced_top10"] for item in low_summaries),
                "target_author_balanced_top10": higher_summary["author_balanced_top10"],
                "maximum_topic_word_cosine": float(values[selected].max()),
                "minimum_selected_topic_word_cosine": float(values[selected].min()),
                "Researcher_Split_Judgment": "",
                "Researcher_Split_Rationale": "",
                "Researcher_Confidence": "",
            })
            event_number += 1
    events_frame = pd.DataFrame(events)
    prefix = f"K{lower_k}_to_K{higher_k}"
    write_csv(pairs, output / f"_private/stage2_candidates/{prefix}_alignment_candidates.csv")
    write_csv(events_frame, output / f"_private/stage2_candidates/{prefix}_split_merge_candidates.csv")
    return pairs, events_frame


def build_fragmentation_candidates(
    model: FrozenModel, mapping: pd.DataFrame, words: pd.DataFrame, poems: pd.DataFrame,
    config: dict[str, Any], output: Path,
) -> pd.DataFrame:
    cosine = model.topic_unit @ model.topic_unit.T
    top20 = stable_top_indices(model.topic_word, 20)
    top_poems = stable_top_indices(model.theta.T, 20)
    mapping_lookup = {(int(row.true_K), int(row.true_topic_id)): str(row.blind_topic_id) for row in mapping.itertuples(index=False)}
    all_rows = []
    for left in range(model.k):
        for right in range(left + 1, model.k):
            all_rows.append({
                "left_topic_id": left + 1,
                "right_topic_id": right + 1,
                "topic_word_cosine": float(cosine[left, right]),
                "top20_word_jaccard": jaccard(set(map(int, top20[left])), set(map(int, top20[right]))),
                "top20_representative_poem_jaccard": jaccard(set(map(int, top_poems[left])), set(map(int, top_poems[right]))),
            })
    ranked = sorted(all_rows, key=lambda row: (-row["topic_word_cosine"], -row["top20_word_jaccard"]))
    selected_keys = {
        (row["left_topic_id"], row["right_topic_id"])
        for row in ranked[: int(config["fragmentation_top_n_floor"])]
    }
    for row in ranked:
        if row["topic_word_cosine"] >= float(config["fragmentation_cosine_min"]) or row["top20_word_jaccard"] >= float(config["fragmentation_jaccard_min"]):
            selected_keys.add((row["left_topic_id"], row["right_topic_id"]))
    rows = []
    for rank, row in enumerate((item for item in ranked if (item["left_topic_id"], item["right_topic_id"]) in selected_keys), start=1):
        left = topic_summary(model.k, row["left_topic_id"], mapping_lookup, words, poems)
        right = topic_summary(model.k, row["right_topic_id"], mapping_lookup, words, poems)
        rows.append({
            "candidate_pair_id": f"K20_F{rank:02d}",
            "left_topic_id": row["left_topic_id"],
            "left_blind_topic_id": left["blind_topic_id"],
            "right_topic_id": row["right_topic_id"],
            "right_blind_topic_id": right["blind_topic_id"],
            "topic_word_cosine": row["topic_word_cosine"],
            "top20_word_jaccard": row["top20_word_jaccard"],
            "top20_representative_poem_jaccard": row["top20_representative_poem_jaccard"],
            "left_top20_words": left["top20_words"],
            "right_top20_words": right["top20_words"],
            "left_top10_poems": left["top10_poems"],
            "right_top10_poems": right["top10_poems"],
            "Researcher_Fragmentation_Judgment": "",
            "Researcher_Fragmentation_Rationale": "",
            "Researcher_Confidence": "",
        })
    result = pd.DataFrame(rows)
    write_csv(result, output / "_private/stage2_candidates/K20_fragmentation_candidates.csv")
    return result


def build_hidden_ai_reference(mapping: pd.DataFrame, output: Path, input_rows: list[dict[str, str]]) -> None:
    old_root = ROOT / "model_selection/four_model_blind_semantic_audit"
    old_mapping_path = old_root / "_private_unblind/blind_topic_mapping_secret.csv"
    old_review_path = old_root / "data/blind_topic_ai_review.csv"
    if not old_mapping_path.exists() or not old_review_path.exists():
        write_csv(pd.DataFrame(columns=["blind_topic_id", "AI_reference_status"]), output / "_private/hidden_ai_reference.csv")
        return
    old_mapping = pd.read_csv(old_mapping_path)
    old_review = pd.read_csv(old_review_path)
    reference = mapping.merge(
        old_mapping[["true_K", "true_topic_id", "blind_topic_id"]].rename(columns={"blind_topic_id": "old_blind_topic_id"}),
        on=["true_K", "true_topic_id"], how="left", validate="one_to_one",
    ).merge(
        old_review,
        left_on="old_blind_topic_id", right_on="blind_topic_id", how="left", suffixes=("", "_old"), validate="one_to_one",
    )
    keep = [
        "blind_topic_id", "AI_provisional_label", "AI_word_coherence", "AI_poem_coherence",
        "AI_word_poem_agreement", "AI_labelability", "AI_literary_interpretability",
        "AI_semantic_specificity", "AI_overbreadth", "AI_internal_mixedness", "AI_topic_type",
        "AI_under_split_flag", "AI_notes", "AI_review_status",
    ]
    write_csv(reference[keep].sort_values("blind_topic_id"), output / "_private/hidden_ai_reference.csv")
    for role, path in [("old_ai_mapping_background_only", old_mapping_path), ("old_ai_review_background_only", old_review_path)]:
        input_rows.append({"role": role, "path": relative(path), "sha256": sha256(path)})


def scan_blind_materials(output: Path) -> list[dict[str, str]]:
    failures = []
    forbidden = [
        "K12", "K15", "K20", "MODEL_A", "MODEL_B", "MODEL_C", "MODEL_D",
        "AI_provisional", "AI_topic_type", "under_split", "fragmentation", "NPMI", "Perplexity",
    ]
    paths = list((output / "human_review_cards").glob("HR*.md")) + [
        output / "data/researcher_topic_review_template.csv",
        output / "data/blind_topic_words.csv",
        output / "data/blind_poem_evidence.csv",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for term in forbidden:
            if term.lower() in text.lower():
                failures.append({"path": relative(path), "forbidden_term": term})
    return failures


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    output = ROOT / config["output_root"]
    for name in ["data", "audit", "_private", "_private/stage2_candidates", "human_review_cards"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    models, manifest, input_rows = load_models(config)
    comparability = build_comparability(manifest, output)
    corpus_root = ROOT / config["corpus_root"]
    vocabulary, documents = load_corpus(corpus_root)
    for name in ["vocabulary.csv", "document_index.csv", "lda_corpus.csv", "corpus_metadata.json"]:
        path = corpus_root / name
        input_rows.append({"role": "frozen_corpus", "path": relative(path), "sha256": sha256(path)})
    mapping = make_mapping(models, config, output / "_private")
    topics, words, poems = build_blind_evidence(models, mapping, vocabulary, documents, config, output)
    write_cards(topics, words, poems, output)
    pairs_12_15, events_12_15 = build_transition_candidates(12, 15, models, mapping, words, poems, config, output)
    pairs_15_20, events_15_20 = build_transition_candidates(15, 20, models, mapping, words, poems, config, output)
    fragmentation = build_fragmentation_candidates(models[20], mapping, words, poems, config, output)
    build_hidden_ai_reference(mapping, output, input_rows)
    failures = scan_blind_materials(output)
    if failures:
        write_csv(pd.DataFrame(failures), output / "audit/blind_leak_failures.csv")
        raise ValueError(f"第一阶段盲审材料发现 {len(failures)} 项泄漏")
    write_csv(pd.DataFrame(input_rows).drop_duplicates("path"), output / "_private/input_hash_baseline.csv")
    summary = {
        "status": "READY_FOR_RESEARCHER_REVIEW",
        "models_ready": int((comparability.comparability_status == "READY").sum()),
        "topics": len(topics),
        "cards": len(list((output / "human_review_cards").glob("HR*.md"))),
        "K12_to_K15_candidate_pairs": len(pairs_12_15),
        "K12_to_K15_split_merge_events": len(events_12_15),
        "K12_to_K15_split_events": int((events_12_15.relation_candidate == "SPLIT").sum()) if len(events_12_15) else 0,
        "K12_to_K15_merge_events": int((events_12_15.relation_candidate == "MERGE").sum()) if len(events_12_15) else 0,
        "K15_to_K20_candidate_pairs": len(pairs_15_20),
        "K15_to_K20_split_merge_events": len(events_15_20),
        "K15_to_K20_split_events": int((events_15_20.relation_candidate == "SPLIT").sum()) if len(events_15_20) else 0,
        "K15_to_K20_merge_events": int((events_15_20.relation_candidate == "MERGE").sum()) if len(events_15_20) else 0,
        "K20_fragmentation_candidate_pairs": len(fragmentation),
        "researcher_fields_prefilled": 0,
        "blind_material_leaks": 0,
        "model_training_calls": 0,
        "topic_review_freeze_status": "TOPIC_REVIEW_INCOMPLETE",
    }
    write_json(output / "audit/prepare_summary.json", summary)
    (output / "README.md").write_text(
        "# K12/K15/K20 Researcher-Led Humanities Adjudication\n\n"
        "本目录保存三模型研究者主导的人文学裁决材料。当前只完成 PREPARE；47 个 `Researcher_*` 记录仍为空，"
        "不得解盲或生成最终 K 决策。研究者应从工作簿 `Blind_Topic_Review` 开始，完成后运行"
        " `validate_human_review.py` 与 `freeze_human_review.py`。\n\n"
        "所有 Topic 级评分必须由研究者本人填写。自动相似度只生成后台候选，不构成 split、merge 或 fragmentation 裁决。\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="准备三模型研究者人文学盲审材料")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    result = prepare(read_json(Path(args.config)))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
