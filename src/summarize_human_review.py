"""只根据已冻结的研究者评分生成解盲后的模型级统计。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model_selection/K12_K15_K20_human_adjudication"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    frozen_path = OUT / "data/researcher_topic_review_frozen.csv"
    hash_path = OUT / "audit/researcher_topic_review_frozen.sha256"
    if not frozen_path.exists() or not hash_path.exists():
        raise RuntimeError("TOPIC_REVIEW_INCOMPLETE：未找到 Topic 级冻结结果，禁止解盲")
    stored = hash_path.read_text(encoding="utf-8").split()[0]
    if sha256(frozen_path) != stored:
        raise RuntimeError("Topic 级冻结结果哈希失配，禁止解盲")
    frozen = pd.read_csv(frozen_path, encoding="utf-8-sig")
    mapping = pd.read_csv(OUT / "_private/blind_topic_mapping_secret.csv", encoding="utf-8-sig")
    merged = frozen.merge(mapping[["blind_topic_id", "true_K", "true_topic_id"]], on="blind_topic_id", validate="one_to_one")
    score_map = {
        "Researcher_Word_Coherence": "mean_word_coherence",
        "Researcher_Poem_Coherence": "mean_poem_coherence",
        "Researcher_Word_Poem_Agreement": "mean_word_poem_agreement",
        "Researcher_Labelability": "mean_labelability",
        "Researcher_Literary_Interpretability": "mean_literary_interpretability",
        "Researcher_Semantic_Specificity": "mean_semantic_specificity",
        "Researcher_Overbreadth": "mean_overbreadth",
        "Researcher_Internal_Mixedness": "mean_internal_mixedness",
    }
    rows = []
    for k, group in merged.groupby("true_K", sort=True):
        row = {"K": int(k), "n_topics": len(group)}
        for topic_type in ["CLEAR", "BROAD", "MIXED", "NOISE"]:
            count = int(group.Researcher_Topic_Type.eq(topic_type).sum())
            row[f"{topic_type}_count"] = count
            row[f"{topic_type}_ratio"] = count / len(group)
        for source, target in score_map.items():
            row[target] = float(pd.to_numeric(group[source]).mean())
        row["needs_split_count"] = int(group.Researcher_Needs_Splitting.eq("YES").sum())
        row["needs_split_ratio"] = row["needs_split_count"] / len(group)
        rows.append(row)
    unblinded_path = OUT / "data/researcher_topic_review_unblinded.csv"
    summary_path = OUT / "data/researcher_model_summary.csv"
    merged.sort_values(["true_K", "true_topic_id"]).to_csv(unblinded_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    if sha256(frozen_path) != stored:
        raise RuntimeError("解盲统计改变了冻结 Topic 评分")
    result = {
        "status": "TOPIC_REVIEW_SUMMARIZED_READY_FOR_MODEL_LEVEL_REVIEW",
        "frozen_sha256_unchanged": stored,
        "models": len(rows),
        "AI_scores_used": False,
        "researcher_scores_generated_by_codex": False,
    }
    (OUT / "audit/topic_unblind_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
