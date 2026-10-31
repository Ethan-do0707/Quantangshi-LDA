"""《全唐诗》四模型盲审式语义分辨率审核。

prepare 阶段只读冻结模型，生成匿名、全局乱序的 Topic 证据卡。
本程序从不调用 LDA ``fit``，也不读取旧 K15 审核或 K20 taxonomy。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "model_selection/four_model_blind_semantic_audit/config/blind_audit_config.yaml"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def ensure_dirs(out: Path) -> dict[str, Path]:
    paths = {
        "private": out / "_private_unblind",
        "config": out / "config",
        "cards": out / "blind_review_cards",
        "data": out / "data",
        "figures": out / "figures",
        "audit": out / "audit",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_dense_npz(path: Path, preferred_keys: Iterable[str]) -> np.ndarray:
    """同时兼容 scipy sparse NPZ 与普通 NumPy NPZ。"""
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
        raise KeyError(f"无法确定 {path} 的数组键：{data.files}")


def row_probability(components: np.ndarray) -> np.ndarray:
    totals = components.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("components 存在非正行和")
    return components / totals


def row_unit(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("topic-word 存在零范数行")
    return values / norms


def stable_top_indices(values: np.ndarray, count: int) -> np.ndarray:
    return np.argsort(-values, axis=1, kind="mergesort")[:, :count]


def excerpt(text: object, limit: int) -> str:
    value = "" if pd.isna(text) else str(text).strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip("，。；！？、 ") + "……"


@dataclass(frozen=True)
class ModelData:
    true_k: int
    blind_model_id: str
    components: np.ndarray
    topic_word: np.ndarray
    topic_unit: np.ndarray
    theta: np.ndarray
    manifest_row: dict[str, str]


def load_corpus(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    corpus_root = ROOT / config["corpus_root"]
    vocabulary = pd.read_csv(corpus_root / "vocabulary.csv")
    documents = pd.read_csv(corpus_root / "document_index.csv")
    corpus = pd.read_csv(corpus_root / "lda_corpus.csv")
    if not {"term_index", "token"}.issubset(vocabulary.columns):
        raise ValueError("C vocabulary 字段不完整")
    if not {"matrix_row", "corpus_row", "poem_id", "author", "title"}.issubset(documents.columns):
        raise ValueError("C document_index 字段不完整")
    if len(vocabulary) != 35063 or len(documents) != 43015:
        raise ValueError(f"冻结 C corpus shape 异常：vocab={len(vocabulary)}, documents={len(documents)}")
    joined = documents.merge(
        corpus[["poem_id", "original_text", "cleaned_tokens"]],
        on="poem_id",
        how="left",
        validate="one_to_one",
    ).sort_values("matrix_row")
    if len(joined) != 43015 or joined.original_text.isna().any():
        raise ValueError("C corpus 文档映射不完整")
    if not np.array_equal(joined.matrix_row.to_numpy(), np.arange(43015)):
        raise ValueError("C corpus matrix_row 顺序错误")
    return vocabulary.sort_values("term_index").reset_index(drop=True), joined.reset_index(drop=True)


def make_model_mapping(config: dict[str, Any], private_dir: Path) -> pd.DataFrame:
    path = private_dir / "blind_mapping_secret.csv"
    aliases = ["MODEL_A", "MODEL_B", "MODEL_C", "MODEL_D"]
    shuffled = list(map(int, config["expected_topic_counts"]))
    random.Random(int(config["model_mapping_seed"])).shuffle(shuffled)
    expected = pd.DataFrame({"blind_model_id": aliases, "true_K": shuffled})
    if path.exists():
        existing = pd.read_csv(path)
        if not existing.equals(expected):
            raise ValueError("既有 blind mapping 与固定 seed 生成结果不一致")
        return existing
    expected.to_csv(path, index=False, encoding="utf-8-sig")
    return expected


def load_models(config: dict[str, Any], mapping: pd.DataFrame) -> tuple[list[ModelData], list[dict[str, Any]]]:
    manifest_path = ROOT / config["ready_manifest"]
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    manifest["K"] = manifest.K.astype(int)
    if sorted(manifest.K.tolist()) != sorted(map(int, config["expected_topic_counts"])):
        raise ValueError("blind-audit-ready manifest 的 K 集合不符")
    if not manifest.ready_for_blind_audit.str.lower().eq("true").all():
        raise ValueError("至少一个模型未标记为 ready_for_blind_audit")
    input_rows: list[dict[str, Any]] = [
        {"role": "ready_manifest", "path": rel(manifest_path), "sha256": sha256(manifest_path)}
    ]
    model_rows: list[ModelData] = []
    k_to_alias = dict(zip(mapping.true_K.astype(int), mapping.blind_model_id))
    for row in manifest.sort_values("K").to_dict("records"):
        true_k = int(row["K"])
        source_paths = {
            name: ROOT / row[name]
            for name in ["model_path", "components_path", "topic_word_path", "theta_full_path", "representative_poems_path"]
        }
        missing = [name for name, path in source_paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"SOURCE_MISSING K={true_k}: {missing}")
        for name, path in source_paths.items():
            input_rows.append({"role": f"model_{name}", "path": rel(path), "sha256": sha256(path)})
        components = load_dense_npz(source_paths["components_path"], ["components"])
        theta = load_dense_npz(source_paths["theta_full_path"], ["doc_topic_full", "probabilities"])
        if components.shape != (true_k, 35063) or theta.shape != (43015, true_k):
            raise ValueError(f"模型矩阵 shape 异常 K={true_k}: {components.shape}, {theta.shape}")
        if not np.isfinite(components).all() or not np.isfinite(theta).all() or np.any(theta < 0):
            raise ValueError(f"模型矩阵含非法值 K={true_k}")
        max_error = float(np.max(np.abs(theta.sum(axis=1) - 1.0)))
        if max_error > 1e-10:
            raise ValueError(f"theta 行和异常 K={true_k}: {max_error}")
        topic_word = row_probability(components)
        model_rows.append(ModelData(true_k, k_to_alias[true_k], components, topic_word, row_unit(topic_word), theta, row))
    return model_rows, input_rows


def make_topic_mapping(models: list[ModelData], config: dict[str, Any], private_dir: Path) -> pd.DataFrame:
    path = private_dir / "blind_topic_mapping_secret.csv"
    rng = random.Random(int(config["topic_shuffle_seed"]))
    rows: list[dict[str, Any]] = []
    serial = 1
    for model in sorted(models, key=lambda item: item.blind_model_id):
        topic_ids = list(range(1, model.true_k + 1))
        rng.shuffle(topic_ids)
        for anonymous_slot, true_topic_id in enumerate(topic_ids, start=1):
            rows.append({
                "blind_topic_id": f"BT{serial:03d}",
                "blind_model_id": model.blind_model_id,
                "anonymous_topic_slot": anonymous_slot,
                "true_K": model.true_k,
                "true_topic_id": true_topic_id,
            })
            serial += 1
    expected = pd.DataFrame(rows)
    if len(expected) != int(config["expected_total_topics"]):
        raise AssertionError("匿名 Topic 总数不是 61")
    if path.exists():
        existing = pd.read_csv(path)
        if not existing.equals(expected):
            raise ValueError("既有 blind topic mapping 与固定 seed 生成结果不一致")
        return existing
    expected.to_csv(path, index=False, encoding="utf-8-sig")
    return expected


def build_blind_evidence(
    models: list[ModelData],
    topic_mapping: pd.DataFrame,
    vocabulary: pd.DataFrame,
    documents: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tokens = vocabulary.token.astype(str).to_numpy()
    lookup = {(int(row.true_K), int(row.true_topic_id)): row for row in topic_mapping.itertuples(index=False)}
    evidence_rows: list[dict[str, Any]] = []
    word_rows: list[dict[str, Any]] = []
    poem_rows: list[dict[str, Any]] = []
    for model in models:
        top_words = stable_top_indices(model.topic_word, int(config["top_word_count"]))
        for true_topic_id in range(1, model.true_k + 1):
            secret = lookup[(model.true_k, true_topic_id)]
            blind_id = str(secret.blind_topic_id)
            topic_idx = true_topic_id - 1
            for rank, term_idx in enumerate(top_words[topic_idx], start=1):
                word_rows.append({
                    "blind_topic_id": blind_id,
                    "blind_model_id": model.blind_model_id,
                    "rank": rank,
                    "word": tokens[term_idx],
                    "topic_weight": float(model.topic_word[topic_idx, term_idx]),
                })
            order = np.argsort(-model.theta[:, topic_idx], kind="mergesort")
            raw = list(map(int, order[: int(config["representative_poem_count"])]))
            balanced: list[int] = []
            author_counts: Counter[str] = Counter()
            for matrix_row in order:
                author = str(documents.iloc[int(matrix_row)].author)
                if author_counts[author] >= int(config["author_maximum_per_topic"]):
                    continue
                balanced.append(int(matrix_row))
                author_counts[author] += 1
                if len(balanced) >= int(config["author_balanced_count"]):
                    break
            if len(balanced) < int(config["author_balanced_count"]):
                raise ValueError(f"作者平衡代表诗不足：{blind_id}")
            for sample_type, selected in [("RAW_TOP20", raw), ("AUTHOR_BALANCED_TOP10", balanced)]:
                for rank, matrix_row in enumerate(selected, start=1):
                    doc = documents.iloc[matrix_row]
                    poem_rows.append({
                        "blind_topic_id": blind_id,
                        "blind_model_id": model.blind_model_id,
                        "sample_type": sample_type,
                        "rank": rank,
                        "document_index": matrix_row,
                        "poem_id": int(doc.poem_id),
                        "author": str(doc.author),
                        "title": str(doc.title),
                        "topic_probability": float(model.theta[matrix_row, topic_idx]),
                        "text_excerpt": excerpt(doc.original_text, int(config["excerpt_character_limit"])),
                        "cleaned_excerpt": excerpt(doc.cleaned_tokens, int(config["excerpt_character_limit"])),
                    })
            evidence_rows.append({
                "blind_topic_id": blind_id,
                "blind_model_id": model.blind_model_id,
                "anonymous_topic_slot": int(secret.anonymous_topic_slot),
                "review_status": "AWAITING_AI_PROVISIONAL_REVIEW",
            })
    evidence = pd.DataFrame(evidence_rows)
    words = pd.DataFrame(word_rows)
    poems = pd.DataFrame(poem_rows)
    if len(evidence) != 61 or len(words) != 61 * int(config["top_word_count"]):
        raise AssertionError("blind evidence 数量异常")
    evidence.to_csv(paths["data"] / "blind_topic_evidence_index.csv", index=False, encoding="utf-8-sig")
    words.to_csv(paths["data"] / "blind_topic_words.csv", index=False, encoding="utf-8-sig")
    poems.to_csv(paths["data"] / "blind_representative_poems.csv", index=False, encoding="utf-8-sig")
    return evidence, words, poems


def write_cards(
    evidence: pd.DataFrame,
    words: pd.DataFrame,
    poems: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
) -> pd.DataFrame:
    blind_ids = evidence.blind_topic_id.tolist()
    random.Random(int(config["card_order_seed"])).shuffle(blind_ids)
    index_rows: list[dict[str, Any]] = []
    for card_order, blind_id in enumerate(blind_ids, start=1):
        meta = evidence[evidence.blind_topic_id == blind_id].iloc[0]
        topic_words = words[words.blind_topic_id == blind_id].sort_values("rank")
        raw = poems[(poems.blind_topic_id == blind_id) & (poems.sample_type == "RAW_TOP20")].sort_values("rank")
        balanced = poems[(poems.blind_topic_id == blind_id) & (poems.sample_type == "AUTHOR_BALANCED_TOP10")].sort_values("rank")
        lines = [
            f"# Blind Topic Card {card_order:03d}",
            "",
            f"- Blind Topic ID: `{blind_id}`",
            f"- Blind Model ID: `{meta.blind_model_id}`",
            "- Status: `AI_PROVISIONAL_REVIEW_PENDING`",
            "",
            "## Top 30 topic words",
            "",
            "| Rank | Word | Topic weight |",
            "|---:|---|---:|",
        ]
        lines.extend(f"| {int(row.rank)} | {row.word} | {float(row.topic_weight):.10f} |" for row in topic_words.itertuples(index=False))
        lines.extend(["", "## Top 20 representative poems", ""])
        for row in raw.itertuples(index=False):
            lines.extend([
                f"### {int(row.rank)}. poem_id={int(row.poem_id)}｜{row.author}｜{row.title}｜P={float(row.topic_probability):.6f}",
                "",
                str(row.text_excerpt),
                "",
            ])
        lines.extend(["## Author-balanced representative poems", "", "最多 10 首；每位作者最多 1 首。", ""])
        for row in balanced.itertuples(index=False):
            lines.extend([
                f"### {int(row.rank)}. poem_id={int(row.poem_id)}｜{row.author}｜{row.title}｜P={float(row.topic_probability):.6f}",
                "",
                str(row.text_excerpt),
                "",
            ])
        filename = f"card_{card_order:03d}.md"
        (paths["cards"] / filename).write_text("\n".join(lines), encoding="utf-8")
        index_rows.append({
            "card_order": card_order,
            "blind_topic_id": blind_id,
            "blind_model_id": meta.blind_model_id,
            "card_path": f"blind_review_cards/{filename}",
        })
    card_index = pd.DataFrame(index_rows)
    card_index.to_csv(paths["audit"] / "blind_card_index.csv", index=False, encoding="utf-8-sig")
    return card_index


def jaccard(left: set[int], right: set[int]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def write_similarity_candidates(
    models: list[ModelData],
    topic_mapping: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    key_to_blind = {
        (int(row.true_K), int(row.true_topic_id)): str(row.blind_topic_id)
        for row in topic_mapping.itertuples(index=False)
    }
    top20 = {model.true_k: stable_top_indices(model.topic_word, 20) for model in models}
    within_rows: list[dict[str, Any]] = []
    for model in models:
        similarities = model.topic_unit @ model.topic_unit.T
        candidates: list[dict[str, Any]] = []
        for left in range(model.true_k):
            for right in range(left + 1, model.true_k):
                candidates.append({
                    "blind_model_id": model.blind_model_id,
                    "blind_topic_id_1": key_to_blind[(model.true_k, left + 1)],
                    "blind_topic_id_2": key_to_blind[(model.true_k, right + 1)],
                    "topic_word_cosine": float(similarities[left, right]),
                    "top20_jaccard": jaccard(set(map(int, top20[model.true_k][left])), set(map(int, top20[model.true_k][right]))),
                })
        candidates.sort(key=lambda item: (-item["topic_word_cosine"], -item["top20_jaccard"]))
        selected = [
            item for item in candidates
            if item["topic_word_cosine"] >= float(config["within_model_candidate_cosine_min"])
            or item["top20_jaccard"] >= float(config["within_model_candidate_jaccard_min"])
        ]
        for item in candidates[:3]:
            if item not in selected:
                selected.append(item)
        for item in selected:
            item["automatic_status"] = "CANDIDATE_ONLY_TEXT_REVIEW_REQUIRED"
            within_rows.append(item)
    pd.DataFrame(within_rows).sort_values(
        ["blind_model_id", "topic_word_cosine"], ascending=[True, False]
    ).to_csv(paths["data"] / "blind_within_model_similarity_candidates.csv", index=False, encoding="utf-8-sig")

    cross_rows: list[dict[str, Any]] = []
    ordered = sorted(models, key=lambda item: item.blind_model_id)
    for source_index, source in enumerate(ordered):
        for target in ordered[source_index + 1 :]:
            similarities = source.topic_unit @ target.topic_unit.T
            for topic_index in range(source.true_k):
                order = np.argsort(-similarities[topic_index], kind="mergesort")[: int(config["cross_model_candidates_per_target"])]
                for candidate_rank, target_index in enumerate(order, start=1):
                    cross_rows.append({
                        "source_blind_model_id": source.blind_model_id,
                        "source_blind_topic_id": key_to_blind[(source.true_k, topic_index + 1)],
                        "target_blind_model_id": target.blind_model_id,
                        "target_blind_topic_id": key_to_blind[(target.true_k, int(target_index) + 1)],
                        "candidate_rank": candidate_rank,
                        "topic_word_cosine": float(similarities[topic_index, target_index]),
                        "top20_jaccard": jaccard(
                            set(map(int, top20[source.true_k][topic_index])),
                            set(map(int, top20[target.true_k][target_index])),
                        ),
                        "automatic_status": "CANDIDATE_ONLY_SEMANTIC_REVIEW_REQUIRED",
                    })
    pd.DataFrame(cross_rows).to_csv(paths["data"] / "blind_cross_model_alignment_candidates.csv", index=False, encoding="utf-8-sig")


def scan_blind_leaks(paths: dict[str, Path], config: dict[str, Any]) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    files = list(paths["cards"].glob("*.md")) + list(paths["data"].glob("blind_*.csv"))
    patterns = [(term, re.compile(re.escape(term), re.IGNORECASE)) for term in config["forbidden_blind_terms"]]
    for path in files:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for term, pattern in patterns:
            if pattern.search(text):
                failures.append({"path": rel(path), "forbidden_term": term})
    return failures


def prepare(config: dict[str, Any], config_source: Path) -> dict[str, Any]:
    out = ROOT / config["output_root"]
    paths = ensure_dirs(out)
    config_target = paths["config"] / "blind_audit_config.yaml"
    if config_source.resolve() != config_target.resolve():
        shutil.copy2(config_source, config_target)
    mapping = make_model_mapping(config, paths["private"])
    models, input_rows = load_models(config, mapping)
    topic_mapping = make_topic_mapping(models, config, paths["private"])
    vocabulary, documents = load_corpus(config)
    for name in ["vocabulary.csv", "document_index.csv", "lda_corpus.csv"]:
        source = ROOT / config["corpus_root"] / name
        input_rows.append({"role": "frozen_corpus", "path": rel(source), "sha256": sha256(source)})
    pd.DataFrame(input_rows).drop_duplicates("path").to_csv(
        paths["private"] / "input_hash_baseline.csv", index=False, encoding="utf-8-sig"
    )
    evidence, words, poems = build_blind_evidence(models, topic_mapping, vocabulary, documents, config, paths)
    card_index = write_cards(evidence, words, poems, config, paths)
    write_similarity_candidates(models, topic_mapping, config, paths)
    leaks = scan_blind_leaks(paths, config)
    if leaks:
        pd.DataFrame(leaks).to_csv(paths["audit"] / "blind_leak_failures.csv", index=False, encoding="utf-8-sig")
        raise ValueError(f"盲审材料发现身份或量化指标泄漏：{len(leaks)} 项")
    summary = {
        "status": "BLIND_EVIDENCE_PREPARED",
        "models_loaded": len(models),
        "topics_prepared": len(evidence),
        "cards_generated": len(card_index),
        "model_training_calls": 0,
        "blind_leaks": 0,
        "legacy_review_sources_loaded": 0,
    }
    write_json(paths["audit"] / "prepare_summary.json", summary)
    (out / "README.md").write_text(
        "# Four-Model Blind Semantic Audit\n\n"
        "该目录保存四个冻结 LDA 模型的 AI-assisted blind semantic audit。"
        "盲审材料使用匿名模型和全局乱序 Topic；真实映射仅位于 `_private_unblind/`。\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="《全唐诗》四模型盲审式语义分辨率审核")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--phase", choices=["prepare"], default="prepare")
    parsed = parser.parse_args()
    config_path = Path(parsed.config)
    result = prepare(read_json(config_path), config_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
