"""为 Lower-K 人文学语义审核建立只读、可追踪的证据索引。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_ROOT = ROOT / "model_selection" / "lower_k_boundary_perplexity"
OUTPUT_ROOT = ROOT / "model_selection" / "lower_k_semantic_audit"
DATA_ROOT = OUTPUT_ROOT / "data"
CORPUS_ROOT = ROOT / "output" / "tf_df_sensitivity" / "C"
K20_ROOT = (
    ROOT
    / "outputs"
    / "01a05061-9ee3-7610-a401-abfec81de1b2"
    / "literary_history_dataset"
)


def sha256(path: Path) -> str:
    """计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_text(value: object, limit: int = 96) -> str:
    """生成不截断句内字段结构的短正文证据。"""

    text = " ".join(str(value).replace("\r", " ").replace("\n", " ").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "……"


def join_weighted_words(rows: pd.DataFrame) -> str:
    """将 Top words 与原始概率合并为可审读字符串。"""

    ordered = rows.sort_values("rank")
    return ";".join(
        f"{row.token}|{float(row.probability):.12f}" for row in ordered.itertuples()
    )


def author_balanced_rows(
    matrix: np.ndarray,
    topic_index: int,
    documents: pd.DataFrame,
    poems: pd.DataFrame,
    n_authors: int = 5,
) -> pd.DataFrame:
    """每位作者只保留该 Topic 概率最高的一首，再取前若干作者。"""

    candidates = documents.copy()
    candidates["topic_probability"] = matrix[:, topic_index]
    candidates = candidates.sort_values(
        ["topic_probability", "matrix_row"], ascending=[False, True]
    )
    candidates = candidates.drop_duplicates("author", keep="first").head(n_authors)
    poem_columns = poems[["poem_id", "original_text"]].copy()
    return candidates.merge(poem_columns, on="poem_id", how="left", validate="one_to_one")


def build() -> dict[str, object]:
    """读取冻结来源并写出证据索引与紧凑审读底稿。"""

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    selected = (
        pd.read_csv(BOUNDARY_ROOT / "data" / "lower_k_metrics_summary.csv")
        .set_index("K")["selected_seed"]
        .astype(int)
        .to_dict()
    )
    alignment = pd.read_csv(BOUNDARY_ROOT / "data" / "lower_k_to_k20_alignment.csv")
    taxonomy_path = K20_ROOT / "taxonomy" / "C_K20_topic_taxonomy.csv"
    taxonomy = pd.read_csv(taxonomy_path).set_index("topic_number")
    k20_words_path = K20_ROOT / "freeze" / "C_K20" / "topics_top30.csv"
    k20_poems_path = K20_ROOT / "freeze" / "C_K20" / "representative_poems_top20.csv"
    k20_words = pd.read_csv(k20_words_path)
    k20_poems = pd.read_csv(k20_poems_path)
    documents = pd.read_csv(CORPUS_ROOT / "document_index.csv")
    poems = pd.read_csv(CORPUS_ROOT / "lda_corpus.csv")

    evidence_rows: list[dict[str, object]] = []
    balanced_frames: list[pd.DataFrame] = []
    source_paths: set[Path] = {
        BOUNDARY_ROOT / "data" / "lower_k_metrics_summary.csv",
        BOUNDARY_ROOT / "data" / "lower_k_to_k20_alignment.csv",
        BOUNDARY_ROOT / "data" / "lower_k_merge_summary.csv",
        BOUNDARY_ROOT / "data" / "C_K_boundary_comparison.csv",
        BOUNDARY_ROOT / "data" / "C_full_k_diagnostic.csv",
        BOUNDARY_ROOT / "QTS_LDA_Lower_K_Boundary_and_Perplexity_Report.md",
        taxonomy_path,
        k20_words_path,
        k20_poems_path,
        CORPUS_ROOT / "document_index.csv",
        CORPUS_ROOT / "lda_corpus.csv",
    }

    for topic_count in (5, 10, 15):
        seed = selected[topic_count]
        model_root = BOUNDARY_ROOT / "lower_k_models" / f"C_K{topic_count}" / f"seed_{seed}"
        words_path = model_root / "topics_top30.csv"
        reps_path = model_root / "representative_poems_top20.csv"
        matrix_path = model_root / "document_topic_probabilities.npz"
        metrics_path = model_root / "metrics.json"
        review_dir = BOUNDARY_ROOT / "review" / "lower_k_review_cards"
        source_paths.update({words_path, reps_path, matrix_path, metrics_path})
        words = pd.read_csv(words_path)
        reps = pd.read_csv(reps_path)
        matrix = load_npz(matrix_path).toarray()
        if matrix.shape != (len(documents), topic_count):
            raise ValueError(f"C_K{topic_count} Document×Topic 维度不一致：{matrix.shape}")
        dominant = np.asarray(matrix.argmax(axis=1), dtype=int)

        for topic_id in range(1, topic_count + 1):
            topic_words = words[words.topic_id == topic_id].sort_values("rank")
            topic_reps = reps[reps.topic_id == topic_id].sort_values("rank")
            if len(topic_words) != 30 or len(topic_reps) != 20:
                raise ValueError(f"C_K{topic_count} Topic {topic_id} 证据数量不足")
            align = alignment[
                (alignment.lower_K == topic_count) & (alignment.lower_topic == topic_id)
            ].iloc[0]
            mapped_topics = [int(value) for value in str(align.matched_C_K20_topics).split(";")]
            similarities = [float(value) for value in str(align.similarities).split(";")]
            mapped_labels = [str(taxonomy.loc[value, "final_topic_label"]) for value in mapped_topics]
            mapped_types = [str(taxonomy.loc[value, "topic_type"]) for value in mapped_topics]
            balanced = author_balanced_rows(matrix, topic_id - 1, documents, poems)
            balanced.insert(0, "topic_id", topic_id)
            balanced.insert(0, "K", topic_count)
            balanced.insert(2, "rank", range(1, len(balanced) + 1))
            balanced["text_snippet"] = balanced.original_text.map(short_text)
            balanced_frames.append(
                balanced[
                    [
                        "K",
                        "topic_id",
                        "rank",
                        "poem_id",
                        "author",
                        "title",
                        "topic_probability",
                        "text_snippet",
                    ]
                ]
            )
            review_card = review_dir / f"C_K{topic_count}_topic_{topic_id:02d}.md"
            source_paths.add(review_card)
            evidence_rows.append(
                {
                    "K": topic_count,
                    "selected_seed": seed,
                    "topic_id": topic_id,
                    "topic_prevalence": float(topic_words.iloc[0].topic_prevalence),
                    "dominant_topic_share": float((dominant == topic_id - 1).mean()),
                    "top30_words_with_probability": join_weighted_words(topic_words),
                    "top30_source": str(words_path.relative_to(ROOT)).replace("\\", "/"),
                    "top30_rows": f"{(topic_id - 1) * 30 + 2}-{topic_id * 30 + 1}",
                    "representative_poem_ids": ";".join(topic_reps.poem_id.astype(str)),
                    "representative_poem_source": str(reps_path.relative_to(ROOT)).replace("\\", "/"),
                    "representative_poem_rows": f"{(topic_id - 1) * 20 + 2}-{topic_id * 20 + 1}",
                    "top20_unique_authors": int(topic_reps.author.nunique()),
                    "top20_max_author_share": float(topic_reps.author.value_counts(normalize=True).max()),
                    "author_balanced_poem_ids": ";".join(balanced.poem_id.astype(str)),
                    "mapped_C_K20_topics": ";".join(f"topic_{value:02d}" for value in mapped_topics),
                    "mapped_C_K20_labels": ";".join(mapped_labels),
                    "mapped_C_K20_types": ";".join(mapped_types),
                    "mapped_similarities": ";".join(f"{value:.10f}" for value in similarities),
                    "automatic_relation_type": align.relation_type,
                    "alignment_source": str(
                        (BOUNDARY_ROOT / "data" / "lower_k_to_k20_alignment.csv").relative_to(ROOT)
                    ).replace("\\", "/"),
                    "alignment_row": int(align.name) + 2,
                    "prior_review_card": str(review_card.relative_to(ROOT)).replace("\\", "/"),
                }
            )

    evidence = pd.DataFrame(evidence_rows).sort_values(["K", "topic_id"])
    evidence_path = DATA_ROOT / "topic_evidence_index.csv"
    evidence.to_csv(evidence_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    balanced_all = pd.concat(balanced_frames, ignore_index=True)
    balanced_path = DATA_ROOT / "author_balanced_representative_poems.csv"
    balanced_all.to_csv(balanced_path, index=False, encoding="utf-8-sig")

    for topic_count in (5, 10, 15):
        seed = selected[topic_count]
        model_root = BOUNDARY_ROOT / "lower_k_models" / f"C_K{topic_count}" / f"seed_{seed}"
        words = pd.read_csv(model_root / "topics_top30.csv")
        reps = pd.read_csv(model_root / "representative_poems_top20.csv")
        lines = [f"# C_K{topic_count} 语义审核证据底稿", "", "仅供 AI provisional review；每题同时呈现 Top30、Top20 与 K20 alignment。", ""]
        for topic_id in range(1, topic_count + 1):
            row = evidence[(evidence.K == topic_count) & (evidence.topic_id == topic_id)].iloc[0]
            topic_words = words[words.topic_id == topic_id].sort_values("rank")
            topic_reps = reps[reps.topic_id == topic_id].sort_values("rank")
            lines.extend(
                [
                    f"## K{topic_count} Topic {topic_id:02d}",
                    "",
                    f"- Top30：{'、'.join(f'{r.token}({float(r.probability):.6f})' for r in topic_words.itertuples())}",
                    f"- Prevalence：{float(row.topic_prevalence):.6f}；dominant share：{float(row.dominant_topic_share):.6f}",
                    f"- Alignment：{row.mapped_C_K20_topics} → {row.mapped_C_K20_labels}；cosine={row.mapped_similarities}",
                    f"- Top20 作者数：{int(row.top20_unique_authors)}；单一作者最高占比：{float(row.top20_max_author_share):.1%}",
                    "",
                    "### Top20 representative poems",
                    "",
                ]
            )
            for rep in topic_reps.itertuples():
                lines.append(
                    f"{int(rep.rank)}. {int(rep.poem_id)}｜{rep.author}《{rep.title}》｜P={float(rep.topic_probability):.6f}｜{short_text(rep.original_text)}"
                )
            lines.extend(["", "### Author-balanced top 5", ""])
            selected_balanced = balanced_all[
                (balanced_all.K == topic_count) & (balanced_all.topic_id == topic_id)
            ]
            for rep in selected_balanced.itertuples():
                lines.append(
                    f"{int(rep.rank)}. {int(rep.poem_id)}｜{rep.author}《{rep.title}》｜P={float(rep.topic_probability):.6f}｜{rep.text_snippet}"
                )
            lines.extend([""])
        (DATA_ROOT / f"evidence_digest_K{topic_count}.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    k20_lines = ["# C_K20 冻结语义证据底稿", "", "标签仅作对齐参照，不向低 K 机械复制。", ""]
    for topic_id in range(1, 21):
        tax = taxonomy.loc[topic_id]
        topic_words = k20_words[k20_words.topic_id == topic_id].sort_values("rank")
        topic_reps = k20_poems[k20_poems.topic_id == topic_id].sort_values("rank").head(10)
        k20_lines.extend(
            [
                f"## C_K20 topic_{topic_id:02d}｜{tax.final_topic_label}｜{tax.topic_type}",
                "",
                f"- Top30：{'、'.join(f'{r.token}({float(r.probability):.6f})' for r in topic_words.itertuples())}",
                f"- 解释：{tax.interpretation}",
                "- 代表诗前10：",
            ]
        )
        for rep in topic_reps.itertuples():
            k20_lines.append(
                f"  - {int(rep.poem_id)}｜{rep.author}《{rep.title}》｜P={float(rep.topic_probability):.6f}｜{short_text(rep.original_text)}"
            )
        k20_lines.append("")
    (DATA_ROOT / "evidence_digest_C_K20.md").write_text("\n".join(k20_lines), encoding="utf-8")

    inventory_rows = []
    for path in sorted(source_paths, key=lambda value: str(value).lower()):
        inventory_rows.append(
            {
                "source_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "status": "FOUND" if path.is_file() else "SOURCE_MISSING",
                "bytes": path.stat().st_size if path.is_file() else "",
                "sha256": sha256(path) if path.is_file() else "",
            }
        )
    inventory = pd.DataFrame(inventory_rows)
    inventory_path = DATA_ROOT / "source_inventory.csv"
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    summary = {
        "topics": len(evidence),
        "top30_complete": bool(evidence.top30_rows.notna().all()),
        "top20_complete": bool(evidence.representative_poem_ids.map(lambda value: len(str(value).split(";")) == 20).all()),
        "author_balanced_rows": len(balanced_all),
        "sources_found": int((inventory.status == "FOUND").sum()),
        "sources_missing": int((inventory.status == "SOURCE_MISSING").sum()),
    }
    (DATA_ROOT / "evidence_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
