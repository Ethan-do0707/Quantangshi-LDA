"""汇总、冻结并解盲《全唐诗》四模型语义分辨率审核。

blind 阶段只读取匿名材料；freeze 阶段写入不可变盲审快照；
unblind 阶段先校验快照哈希，之后才读取真实模型映射与量化证据。
本程序不导入训练库，也不调用任何模型拟合函数。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT / "model_selection/four_model_blind_semantic_audit"
CONFIG = AUDIT_ROOT / "config"
DATA = AUDIT_ROOT / "data"
FIGURES = AUDIT_ROOT / "figures"
AUDIT = AUDIT_ROOT / "audit"
PRIVATE = AUDIT_ROOT / "_private_unblind"

TOPIC_FIELDS = [
    "AI_word_coherence",
    "AI_poem_coherence",
    "AI_word_poem_agreement",
    "AI_labelability",
    "AI_literary_interpretability",
    "AI_semantic_specificity",
    "AI_overbreadth",
    "AI_internal_mixedness",
]
TYPE_ORDER = ["CLEAR", "BROAD", "MIXED", "NOISE"]
MODEL_ORDER = ["MODEL_A", "MODEL_B", "MODEL_C", "MODEL_D"]
COVERAGE_ORDER = [
    "ABSENT",
    "WEAKLY_PRESENT",
    "MERGED_WITH_OTHER",
    "RESOLVED_BUT_BROAD",
    "CLEARLY_RESOLVED",
    "FRAGMENTED",
]


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def truthy(value: object) -> bool:
    return str(value).strip().upper() == "TRUE"


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_blind_inputs() -> dict[str, pd.DataFrame]:
    """仅载入匿名数据；本函数不得访问 private 目录。"""
    frames = {
        "reviews": read_csv(CONFIG / "blind_ai_topic_reviews.csv"),
        "evidence": read_csv(DATA / "blind_topic_evidence_index.csv"),
        "words": read_csv(DATA / "blind_topic_words.csv"),
        "poems": read_csv(DATA / "blind_representative_poems.csv"),
        "families": read_csv(CONFIG / "blind_semantic_family_assignments.csv"),
        "model_judgments": read_csv(CONFIG / "blind_model_ai_judgments.csv"),
        "fragmentation_pairs": read_csv(CONFIG / "blind_fragmentation_pairs.csv"),
        "within_candidates": read_csv(DATA / "blind_within_model_similarity_candidates.csv"),
        "cross_candidates": read_csv(DATA / "blind_cross_model_alignment_candidates.csv"),
        "card_index": read_csv(AUDIT / "blind_card_index.csv"),
    }
    reviews = frames["reviews"]
    if len(reviews) != 61 or reviews.blind_topic_id.nunique() != 61:
        raise ValueError("独立盲审必须恰好包含 61 个唯一 Topic")
    if reviews[TOPIC_FIELDS + ["AI_topic_type", "AI_provisional_label", "AI_notes"]].isna().any().any():
        raise ValueError("独立盲审存在缺失评分或文本判断")
    if not reviews[TOPIC_FIELDS].apply(lambda col: col.between(1, 5).all()).all():
        raise ValueError("存在超出 1–5 范围的盲审评分")
    if not set(reviews.AI_topic_type).issubset(TYPE_ORDER):
        raise ValueError("存在非法 Topic 类型")
    return frames


def build_topic_review(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    reviews = frames["reviews"].copy()
    review = frames["evidence"][["blind_topic_id", "blind_model_id"]].merge(
        reviews, on="blind_topic_id", how="inner", validate="one_to_one"
    )
    review["AI_review_status"] = "AI_PROVISIONAL"
    columns = [
        "blind_topic_id", "blind_model_id", "AI_provisional_label", *TOPIC_FIELDS,
        "AI_topic_type", "AI_under_split_flag", "AI_under_split_subgroups",
        "AI_notes", "AI_review_status",
    ]
    review = review[columns].sort_values("blind_topic_id").reset_index(drop=True)
    write_csv(review, DATA / "blind_topic_ai_review.csv")
    return review


def topic_text_evidence(topic_id: str, frames: dict[str, pd.DataFrame]) -> str:
    words = frames["words"]
    poems = frames["poems"]
    word_text = "、".join(
        words[words.blind_topic_id == topic_id].sort_values("rank").head(12).word.astype(str)
    )
    poem_slice = poems[
        (poems.blind_topic_id == topic_id) & (poems.sample_type == "AUTHOR_BALANCED_TOP10")
    ].sort_values("rank").head(4)
    poem_text = "；".join(
        f"poem_id {int(row.poem_id)}《{row.title}》" for row in poem_slice.itertuples(index=False)
    )
    return f"Top words：{word_text}。代表诗：{poem_text}。"


def build_under_split(topic_review: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    flagged = topic_review[topic_review.AI_under_split_flag.map(truthy)].copy()
    rows = []
    for row in flagged.itertuples(index=False):
        rows.append({
            "blind_model_id": row.blind_model_id,
            "blind_topic_id": row.blind_topic_id,
            "AI_under_split_status": "POTENTIAL_UNDER_SPLIT",
            "AI_semantic_subgroups": row.AI_under_split_subgroups,
            "AI_text_evidence": topic_text_evidence(row.blind_topic_id, frames) + str(row.AI_notes),
            "AI_review_status": "AI_PROVISIONAL",
        })
    result = pd.DataFrame(rows).sort_values(["blind_model_id", "blind_topic_id"]).reset_index(drop=True)
    write_csv(result, DATA / "blind_under_split_audit.csv")
    return result


def build_fragmentation(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pairs = frames["fragmentation_pairs"].copy()
    reviews = frames["reviews"].set_index("blind_topic_id")
    candidate_lookup = {}
    for row in frames["within_candidates"].itertuples(index=False):
        key = tuple(sorted([str(row.blind_topic_id_1), str(row.blind_topic_id_2)]))
        candidate_lookup[(str(row.blind_model_id), key)] = (float(row.topic_word_cosine), float(row.top20_jaccard))
    rows = []
    for row in pairs.itertuples(index=False):
        key = tuple(sorted([str(row.blind_topic_id_1), str(row.blind_topic_id_2)]))
        cosine, overlap = candidate_lookup.get((str(row.blind_model_id), key), (np.nan, np.nan))
        left = reviews.loc[row.blind_topic_id_1]
        right = reviews.loc[row.blind_topic_id_2]
        rows.append({
            "blind_model_id": row.blind_model_id,
            "blind_topic_id_1": row.blind_topic_id_1,
            "blind_topic_id_2": row.blind_topic_id_2,
            "topic_word_cosine_candidate": cosine,
            "top20_jaccard_candidate": overlap,
            "AI_fragmentation_status": "POTENTIAL_FRAGMENTATION",
            "AI_fragmentation_judgment": row.AI_fragmentation_judgment,
            "AI_text_evidence": (
                f"{left.AI_provisional_label} ↔ {right.AI_provisional_label}。"
                f"{row.AI_fragmentation_rationale}"
            ),
            "AI_review_status": "AI_PROVISIONAL",
        })
    result = pd.DataFrame(rows).sort_values(["blind_model_id", "blind_topic_id_1"]).reset_index(drop=True)
    write_csv(result, DATA / "blind_fragmentation_audit.csv")
    return result


def build_model_summary(
    topic_review: pd.DataFrame,
    under_split: pd.DataFrame,
    fragmentation: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    judgment_types = {
        "MODEL_A": ("BALANCED_RESOLUTION", "MEDIUM"),
        "MODEL_B": ("OVER_FRAGMENTED", "MEDIUM"),
        "MODEL_C": ("HIGH_RESOLUTION_BUT_USEFUL", "MEDIUM"),
        "MODEL_D": ("MIXED_QUALITY", "MEDIUM"),
    }
    rationale = frames["model_judgments"].set_index("blind_model_id")
    frag_topics = {
        model: set(group.blind_topic_id_1) | set(group.blind_topic_id_2)
        for model, group in fragmentation.groupby("blind_model_id")
    }
    rows = []
    for model in MODEL_ORDER:
        group = topic_review[topic_review.blind_model_id == model]
        n_topics = len(group)
        counts = group.AI_topic_type.value_counts().to_dict()
        under_count = int((under_split.blind_model_id == model).sum())
        pair_count = int((fragmentation.blind_model_id == model).sum())
        frag_topic_count = len(frag_topics.get(model, set()))
        row: dict[str, Any] = {"blind_model_id": model, "n_topics": n_topics}
        for topic_type in TYPE_ORDER:
            count = int(counts.get(topic_type, 0))
            row[f"{topic_type}_count"] = count
            row[f"{topic_type}_ratio"] = count / n_topics
        for field in TOPIC_FIELDS:
            row[f"mean_{field.removeprefix('AI_')}"] = float(group[field].mean())
        row.update({
            "under_split_topic_count": under_count,
            "under_split_ratio": under_count / n_topics,
            "fragmentation_pair_count": pair_count,
            "fragmentation_topic_count": frag_topic_count,
            "fragmentation_topic_ratio": frag_topic_count / n_topics,
            "AI_model_level_judgment": judgment_types[model][0],
            "AI_blind_rank": int(rationale.loc[model, "AI_blind_rank"]),
            "AI_rank_confidence": judgment_types[model][1],
            "AI_blind_rationale": rationale.loc[model, "AI_blind_rationale"],
            "AI_review_status": "AI_PROVISIONAL",
        })
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("AI_blind_rank").reset_index(drop=True)
    write_csv(result, DATA / "blind_model_summary.csv")
    return result


def coverage_status(group: pd.DataFrame, fragmented_ids: set[str], family_id: str) -> str:
    if group.empty:
        return "ABSENT"
    ids = set(group.blind_topic_id.astype(str))
    types = set(group.AI_topic_type.astype(str))
    if ids & fragmented_ids:
        return "FRAGMENTED"
    if family_id == "F09" or types.issubset({"MIXED", "NOISE"}):
        return "WEAKLY_PRESENT"
    if len(group) >= 2:
        return "CLEARLY_RESOLVED"
    topic_type = next(iter(types))
    if topic_type == "CLEAR":
        return "CLEARLY_RESOLVED"
    if topic_type == "BROAD":
        return "RESOLVED_BUT_BROAD"
    if topic_type == "MIXED":
        return "MERGED_WITH_OTHER"
    return "WEAKLY_PRESENT"


def build_family_matrix(
    topic_review: pd.DataFrame,
    fragmentation: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = frames["families"][["blind_topic_id", "primary_family_id", "AI_family_rationale"]]
    assigned = topic_review.merge(assignments, on="blind_topic_id", how="left", validate="one_to_one")
    if assigned.primary_family_id.isna().any():
        raise ValueError("至少一个 Topic 未分配跨模型语义 family")
    fragmented_ids = set(fragmentation.blind_topic_id_1) | set(fragmentation.blind_topic_id_2)
    family_ids = sorted(assigned.primary_family_id.unique())
    rows = []
    for family_id in family_ids:
        row: dict[str, Any] = {"semantic_family_id": f"Family_{int(family_id[1:]):02d}"}
        for model in MODEL_ORDER:
            group = assigned[(assigned.primary_family_id == family_id) & (assigned.blind_model_id == model)]
            row[model] = coverage_status(group, fragmented_ids, family_id)
            row[f"{model}_topic_ids"] = "|".join(group.blind_topic_id.astype(str))
            row[f"{model}_evidence"] = "；".join(group.AI_family_rationale.astype(str))
        rows.append(row)
    matrix = pd.DataFrame(rows)
    write_csv(matrix, DATA / "blind_semantic_family_matrix.csv")
    return matrix, assigned


def build_cross_alignment(assigned: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    candidate_lookup = {}
    for row in frames["cross_candidates"].itertuples(index=False):
        candidate_lookup[(str(row.source_blind_topic_id), str(row.target_blind_topic_id))] = (
            int(row.candidate_rank), float(row.topic_word_cosine), float(row.top20_jaccard)
        )
    rows = []
    for source_pos, source_model in enumerate(MODEL_ORDER):
        for target_model in MODEL_ORDER[source_pos + 1:]:
            source = assigned[assigned.blind_model_id == source_model]
            target = assigned[assigned.blind_model_id == target_model]
            for family_id in sorted(assigned.primary_family_id.unique()):
                source_group = source[source.primary_family_id == family_id]
                target_group = target[target.primary_family_id == family_id]
                if source_group.empty:
                    continue
                for source_row in source_group.itertuples(index=False):
                    if target_group.empty:
                        rows.append({
                            "source_blind_model_id": source_model,
                            "source_blind_topic_id": source_row.blind_topic_id,
                            "target_blind_model_id": target_model,
                            "target_blind_topic_id": "",
                            "semantic_family_id": f"Family_{int(family_id[1:]):02d}",
                            "relation_type": "UNMATCHED",
                            "candidate_rank": "",
                            "topic_word_cosine": "",
                            "top20_jaccard": "",
                            "AI_semantic_evidence": f"{source_row.AI_provisional_label}：目标模型无同 family 的稳定对应 Topic。",
                            "AI_review_status": "AI_PROVISIONAL",
                        })
                        continue
                    for target_row in target_group.itertuples(index=False):
                        if family_id == "F09":
                            relation = "WEAK"
                        elif len(source_group) == 1 and len(target_group) > 1:
                            relation = "SPLIT"
                        elif len(source_group) > 1 and len(target_group) == 1:
                            relation = "MERGE"
                        else:
                            relation = "DIRECT"
                        metrics = candidate_lookup.get((source_row.blind_topic_id, target_row.blind_topic_id), ("", "", ""))
                        rows.append({
                            "source_blind_model_id": source_model,
                            "source_blind_topic_id": source_row.blind_topic_id,
                            "target_blind_model_id": target_model,
                            "target_blind_topic_id": target_row.blind_topic_id,
                            "semantic_family_id": f"Family_{int(family_id[1:]):02d}",
                            "relation_type": relation,
                            "candidate_rank": metrics[0],
                            "topic_word_cosine": metrics[1],
                            "top20_jaccard": metrics[2],
                            "AI_semantic_evidence": (
                                f"{source_row.AI_provisional_label} ↔ {target_row.AI_provisional_label}；"
                                f"文本复核后归入同一归纳 family。"
                            ),
                            "AI_review_status": "AI_PROVISIONAL",
                        })
    result = pd.DataFrame(rows)
    write_csv(result, DATA / "blind_cross_model_alignment.csv")
    return result


def build_split_merge_judgments(topic_review: pd.DataFrame) -> pd.DataFrame:
    labels = topic_review.set_index("blind_topic_id").AI_provisional_label.to_dict()
    definitions = [
        ("SM01", "MODEL_A", "BT007|BT012", "MODEL_D", "BT048|BT055|BT059|BT060", "SPLIT", "FRAGMENTATION", "残余材料由两个槽位扩至四个，但新槽位仍以异质杂咏或弱边界材料为主，未形成稳定的新文学解释。"),
        ("SM02", "MODEL_D", "BT058", "MODEL_C", "BT038|BT039", "SPLIT", "MEANINGFUL_SPLIT", "单一闺情主题被区分为秋月夜声的离情愁怀与春花艳情的宫怨惜春；关键词、季节意象与代表诗均有可解释差异。"),
        ("SM03", "MODEL_D", "BT061", "MODEL_C", "BT033|BT037", "SPLIT", "MEANINGFUL_SPLIT", "生死身世与伦理议论被进一步区分为政治伦理讽谏、行路贫困和社会悲愤，两组代表诗的言说对象与文学功能不同。"),
        ("SM04", "MODEL_D", "BT048|BT055|BT059|BT060", "MODEL_C", "BT043|BT047", "MERGE", "MEANINGFUL_AGGREGATION", "多个弱残余槽位合并为两个仍不理想但更可管理的混合主题，减少了没有独立文学收益的重复边界。"),
        ("SM05", "MODEL_C", "BT040", "MODEL_B", "BT015|BT028", "SPLIT", "MEANINGFUL_SPLIT", "较宽的春水花鸟物候场被区分为新春岁序与异乡归思、春水花鸟与湖岸游赏，时间经验和景物组织不同。"),
        ("SM06", "MODEL_C", "BT043|BT046", "MODEL_B", "BT018|BT026|BT027", "SPLIT", "FRAGMENTATION", "宗教林泉方向虽增加槽位，但山寺幽居与林泉归隐的关键词和代表诗高度重合；新增边界的文学收益不足。"),
        ("SM07", "MODEL_C", "BT041", "MODEL_B", "BT025|BT031", "SPLIT", "MINOR_REFINEMENT", "老病退闲与诗酒酬唱被部分区分，存在语义重心差异，但士人日常、自况与酬答仍高度相邻。"),
        ("SM08", "MODEL_C", "BT043|BT047", "MODEL_B", "BT017|BT020|BT023|BT024|BT029|BT030", "SPLIT", "FRAGMENTATION", "两个残余主题扩为六个弱或混合槽位，多数仍需复合列举才能命名，额外复杂度主要表现为残余分散。"),
        ("SM09", "MODEL_C", "BT035|BT044", "MODEL_B", "BT032", "MERGE", "PROBLEMATIC_MERGE", "朝廷巡幸恩荣与祀典神圣空间两个可分别解释的礼仪方向在目标模型中收束为一个大主题，压缩了仪式功能与空间边界。"),
    ]
    rows = []
    for event_id, source_model, source_ids, target_model, target_ids, relation, judgment, rationale in definitions:
        all_ids = source_ids.split("|") + target_ids.split("|")
        missing = [topic_id for topic_id in all_ids if topic_id not in labels]
        if missing:
            raise ValueError(f"split/merge 事件引用未知 Topic：{missing}")
        label_evidence = "；".join(f"{topic_id}={labels[topic_id]}" for topic_id in all_ids)
        rows.append({
            "event_id": event_id,
            "source_blind_model_id": source_model,
            "source_blind_topic_ids": source_ids,
            "target_blind_model_id": target_model,
            "target_blind_topic_ids": target_ids,
            "relation_type": relation,
            "AI_semantic_judgment": judgment,
            "AI_text_evidence": f"{label_evidence}。{rationale}",
            "AI_review_status": "AI_PROVISIONAL",
        })
    result = pd.DataFrame(rows)
    write_csv(result, DATA / "blind_split_merge_judgments.csv")
    return result


def build_priority_queue(
    topic_review: pd.DataFrame,
    fragmentation: pd.DataFrame,
    split_merge: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    fragmentation_ids = set(fragmentation.blind_topic_id_1) | set(fragmentation.blind_topic_id_2)
    decisive_ids: set[str] = set()
    for row in split_merge.itertuples(index=False):
        decisive_ids.update(str(row.source_blind_topic_ids).split("|"))
        decisive_ids.update(str(row.target_blind_topic_ids).split("|"))
    card_lookup = frames["card_index"].set_index("blind_topic_id")
    rows = []
    for row in topic_review.itertuples(index=False):
        reasons = []
        if row.AI_topic_type == "MIXED":
            reasons.append("AI_MIXED")
        if truthy(row.AI_under_split_flag):
            reasons.append("AI_UNDER_SPLIT")
        if row.blind_topic_id in fragmentation_ids:
            reasons.append("TEXT_CONFIRMED_FRAGMENTATION_PAIR")
        if reasons:
            priority = 1
        elif row.blind_topic_id in decisive_ids or row.AI_topic_type in {"BROAD", "NOISE"}:
            priority = 2
            reasons.append("DECISIVE_SPLIT_MERGE_OR_BOUNDARY_CASE")
        else:
            priority = 3
            reasons.append("AI_CLEAR_CONFIRMATION")
        rows.append({
            "priority": priority,
            "blind_topic_id": row.blind_topic_id,
            "blind_model_id": row.blind_model_id,
            "AI_provisional_label": row.AI_provisional_label,
            "AI_topic_type": row.AI_topic_type,
            "review_reason": "|".join(reasons),
            "card_order": int(card_lookup.loc[row.blind_topic_id, "card_order"]),
            "card_path": card_lookup.loc[row.blind_topic_id, "card_path"],
            "researcher_status": "PENDING",
        })
    result = pd.DataFrame(rows).sort_values(["priority", "card_order"]).reset_index(drop=True)
    write_csv(result, DATA / "blind_researcher_priority_queue.csv")
    return result


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame[columns].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.3f}")
    header = "| " + " | ".join(display.columns) + " |"
    rule = "|" + "|".join(["---"] * len(display.columns)) + "|"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *body])


def write_blind_report(
    topic_review: pd.DataFrame,
    model_summary: pd.DataFrame,
    under_split: pd.DataFrame,
    fragmentation: pd.DataFrame,
    family_matrix: pd.DataFrame,
    split_merge: pd.DataFrame,
) -> None:
    summary_table = markdown_table(model_summary, [
        "blind_model_id", "n_topics", "CLEAR_ratio", "BROAD_ratio", "MIXED_ratio", "NOISE_ratio",
        "mean_literary_interpretability", "mean_overbreadth", "under_split_ratio",
        "fragmentation_topic_ratio", "AI_model_level_judgment", "AI_blind_rank", "AI_rank_confidence",
    ])
    family_table = markdown_table(family_matrix, ["semantic_family_id", *MODEL_ORDER])
    split_table = markdown_table(split_merge, [
        "event_id", "source_blind_model_id", "target_blind_model_id", "relation_type", "AI_semantic_judgment"
    ])
    model_sections = []
    for index, model in enumerate(MODEL_ORDER, start=5):
        row = model_summary[model_summary.blind_model_id == model].iloc[0]
        topics = topic_review[topic_review.blind_model_id == model]
        type_counts = ", ".join(f"{topic_type}={int((topics.AI_topic_type == topic_type).sum())}" for topic_type in TYPE_ORDER)
        model_sections.append(
            f"# {index} {model}\n\n{type_counts}。模型级判断：{row.AI_model_level_judgment}；"
            f"盲审位次 {int(row.AI_blind_rank)}，信心 {row.AI_rank_confidence}。{row.AI_blind_rationale}\n"
        )
    text = (
        "# 1 Objective\n\n以匿名证据判断哪一种主题分辨率最适合文学研究，平衡语义一致性、可解释性、欠分与碎片化。"
        "本审核为 AI-assisted blind audit，所有判断均为 AI_PROVISIONAL，不能视为 human review completed。\n\n"
        "# 2 Blinding Protocol\n\n模型以 MODEL_A–D 匿名，Topic 使用全局 BT 编号并按固定种子打乱；盲态材料不含真实身份、自动量化指标、旧标签或旧审核评分。独立 Topic 审核完成后才按匿名模型汇总。\n\n"
        "# 3 Review Rubric\n\n八项评分均为 1–5；一致性、可命名性、文学可解释性与特异性越高越好，overbreadth 与 internal mixedness 越高越差。类型限定为 CLEAR、BROAD、MIXED、NOISE。相似度仅用于候选提示，文本证据负责最终裁决。\n\n"
        "# 4 Independent Topic Review\n\n已完成 61/61 个匿名 Topic 的独立审核。每张卡同时检查 Top30 words、Top20 representative poems 与作者平衡样本，且所有标签均为盲态重新生成。\n\n"
        + "\n".join(model_sections) + "\n"
        "# 9 Under-splitting\n\n共识别 " + str(len(under_split)) + " 个 POTENTIAL_UNDER_SPLIT Topic。模型级比率见下表；每一项均在 CSV 中保留关键词、代表诗与子群文本证据。\n\n" + summary_table + "\n\n"
        "# 10 Fragmentation\n\n文本复核确认 " + str(len(fragmentation)) + " 组 POTENTIAL_FRAGMENTATION。候选相似度没有被当作自动裁决。跨模型结构事件如下：\n\n" + split_table + "\n\n"
        "# 11 Semantic Family Coverage\n\n由四模型共同结果归纳九个中性 family；覆盖状态不继承任何历史 taxonomy。\n\n" + family_table + "\n\n"
        "# 12 Blind Model Ranking\n\n盲审排序为 MODEL_A > MODEL_C > MODEL_D > MODEL_B，无并列；信心均为 MEDIUM，因为研究者尚未复核关键边界。排序依据是比例、质量、欠分和碎片化的并列证据，未使用主题绝对数或自动量化指标。\n\n"
        "# 13 Blind Recommendation\n\nMODEL_A 呈现最紧凑的质量平衡；MODEL_C 有较强核心但局部欠分；MODEL_D 的清晰核心伴随残余重复；MODEL_B 的额外分辨率主要落在混合与残余槽位。最终决策仍须研究者亲查 MIXED、split、merge 与模型差异最大的 Topic。\n"
    )
    forbidden = ["K12", "K14", "K15", "K20", "NPMI", "Perplexity", "Topic Diversity", "cross-seed stability", "prior regime", "quantitative rank"]
    hits = [term for term in forbidden if re.search(re.escape(term), text, flags=re.IGNORECASE)]
    if hits:
        raise ValueError(f"盲审报告出现禁用词：{hits}")
    (AUDIT_ROOT / "QTS_Four_Model_Blind_Semantic_Audit.md").write_text(text, encoding="utf-8")


def write_blind_figures(model_summary: pd.DataFrame, family_matrix: pd.DataFrame) -> None:
    configure_plotting()
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = model_summary.set_index("blind_model_id").loc[MODEL_ORDER]
    colors = {"CLEAR": "#0f766e", "BROAD": "#38bdf8", "MIXED": "#f59e0b", "NOISE": "#ef4444"}
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bottom = np.zeros(len(summary))
    for topic_type in TYPE_ORDER:
        values = summary[f"{topic_type}_ratio"].to_numpy()
        ax.bar(MODEL_ORDER, values, bottom=bottom, label=topic_type, color=colors[topic_type])
        bottom += values
    ax.set_ylabel("Topic proportion")
    ax.set_ylim(0, 1)
    ax.set_title("Blind topic-quality distribution")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "blind_model_topic_quality_distribution.png", bbox_inches="tight")
    plt.close(fig)

    metric_map = {
        "Interpretability": "mean_literary_interpretability",
        "Specificity": "mean_semantic_specificity",
        "Overbreadth (risk)": "mean_overbreadth",
        "Mixedness (risk)": "mean_internal_mixedness",
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(MODEL_ORDER))
    width = 0.19
    palette = ["#172554", "#0284c7", "#f59e0b", "#dc2626"]
    for offset, (label, column) in enumerate(metric_map.items()):
        ax.bar(x + (offset - 1.5) * width, summary[column], width, label=label, color=palette[offset])
    ax.set_xticks(x, MODEL_ORDER)
    ax.set_ylim(0, 5)
    ax.set_ylabel("Mean score (1–5)")
    ax.set_title("Blind semantic-quality metrics")
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "blind_semantic_quality_metrics.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.scatter(summary.under_split_ratio, summary.fragmentation_topic_ratio, s=95, color="#0284c7")
    for model, row in summary.iterrows():
        ax.annotate(model, (row.under_split_ratio, row.fragmentation_topic_ratio), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("Under-splitting topic ratio")
    ax.set_ylabel("Fragmentation topic ratio")
    ax.set_title("Under-splitting vs fragmentation")
    ax.set_xlim(left=-0.02)
    ax.set_ylim(bottom=-0.02)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "blind_under_split_vs_fragmentation.png", bbox_inches="tight")
    plt.close(fig)

    numeric = family_matrix[MODEL_ORDER].applymap(lambda value: COVERAGE_ORDER.index(value)).to_numpy()
    cmap = ListedColormap(["#f1f5f9", "#cbd5e1", "#f59e0b", "#7dd3fc", "#0f766e", "#ef4444"])
    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    ax.imshow(numeric, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(COVERAGE_ORDER) - 0.5)
    ax.set_xticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    ax.set_yticks(np.arange(len(family_matrix)), family_matrix.semantic_family_id)
    for i in range(len(family_matrix)):
        for j, model in enumerate(MODEL_ORDER):
            ax.text(j, i, family_matrix.iloc[i][model].replace("_", "\n"), ha="center", va="center", fontsize=7)
    ax.set_title("Blind semantic-family coverage")
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(FIGURES / "blind_semantic_family_coverage.png", bbox_inches="tight")
    plt.close(fig)


def blind_phase() -> None:
    if (AUDIT / "blind_semantic_judgment_frozen.sha256").exists():
        raise RuntimeError("盲审结果已冻结；禁止重新生成盲审评分或汇总")
    frames = load_blind_inputs()
    topic_review = build_topic_review(frames)
    under_split = build_under_split(topic_review, frames)
    fragmentation = build_fragmentation(frames)
    model_summary = build_model_summary(topic_review, under_split, fragmentation, frames)
    family_matrix, assigned = build_family_matrix(topic_review, fragmentation, frames)
    build_cross_alignment(assigned, frames)
    split_merge = build_split_merge_judgments(topic_review)
    build_priority_queue(topic_review, fragmentation, split_merge, frames)
    write_blind_report(topic_review, model_summary, under_split, fragmentation, family_matrix, split_merge)
    write_blind_figures(model_summary, family_matrix)
    print("BLIND_AUDIT_COMPLETE topics=61 models=4")


def build_frozen_frame() -> pd.DataFrame:
    sources = [
        ("TOPIC_REVIEW", DATA / "blind_topic_ai_review.csv", "blind_topic_id"),
        ("MODEL_SUMMARY", DATA / "blind_model_summary.csv", "blind_model_id"),
        ("UNDER_SPLIT", DATA / "blind_under_split_audit.csv", "blind_topic_id"),
        ("FRAGMENTATION", DATA / "blind_fragmentation_audit.csv", "blind_topic_id_1"),
        ("ALIGNMENT", DATA / "blind_cross_model_alignment.csv", "source_blind_topic_id"),
        ("FAMILY_MATRIX", DATA / "blind_semantic_family_matrix.csv", "semantic_family_id"),
        ("SPLIT_MERGE", DATA / "blind_split_merge_judgments.csv", "event_id"),
        ("PRIORITY_QUEUE", DATA / "blind_researcher_priority_queue.csv", "blind_topic_id"),
    ]
    rows = []
    for record_type, path, key_column in sources:
        frame = read_csv(path, keep_default_na=False)
        for index, row in frame.iterrows():
            payload = {column: row[column] for column in frame.columns}
            rows.append({
                "record_type": record_type,
                "entity_id": f"{row[key_column]}:{index + 1}",
                "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "AI_review_status": "AI_PROVISIONAL",
            })
    return pd.DataFrame(rows)


def freeze_phase() -> None:
    frozen_path = DATA / "blind_semantic_judgment_frozen.csv"
    hash_path = AUDIT / "blind_semantic_judgment_frozen.sha256"
    expected = build_frozen_frame()
    if frozen_path.exists() or hash_path.exists():
        if not (frozen_path.exists() and hash_path.exists()):
            raise RuntimeError("冻结文件与哈希文件状态不一致")
        stored = hash_path.read_text(encoding="utf-8").strip().split()[0]
        current = sha256(frozen_path)
        if current != stored:
            raise RuntimeError("冻结盲审文件哈希已变化")
        print(f"BLIND_ALREADY_FROZEN sha256={current}")
        return
    write_csv(expected, frozen_path)
    digest = sha256(frozen_path)
    hash_path.write_text(f"{digest}  data/blind_semantic_judgment_frozen.csv\n", encoding="utf-8")
    freeze_files = [
        DATA / "blind_topic_ai_review.csv", DATA / "blind_model_summary.csv",
        DATA / "blind_under_split_audit.csv", DATA / "blind_fragmentation_audit.csv",
        DATA / "blind_cross_model_alignment.csv", DATA / "blind_semantic_family_matrix.csv",
        DATA / "blind_split_merge_judgments.csv", DATA / "blind_researcher_priority_queue.csv",
        AUDIT_ROOT / "QTS_Four_Model_Blind_Semantic_Audit.md", frozen_path,
    ]
    manifest = pd.DataFrame([
        {"path": str(path.relative_to(AUDIT_ROOT)).replace("\\", "/"), "sha256": sha256(path)}
        for path in freeze_files
    ])
    write_csv(manifest, AUDIT / "blind_freeze_manifest.csv")
    print(f"BLIND_FROZEN sha256={digest}")


def verify_frozen() -> str:
    frozen_path = DATA / "blind_semantic_judgment_frozen.csv"
    hash_path = AUDIT / "blind_semantic_judgment_frozen.sha256"
    if not frozen_path.exists() or not hash_path.exists():
        raise RuntimeError("必须先冻结盲审结果才能解盲")
    stored = hash_path.read_text(encoding="utf-8").strip().split()[0]
    current = sha256(frozen_path)
    if current != stored:
        raise RuntimeError("解盲前冻结盲审文件哈希校验失败")
    return current


def build_seed_selection(mapping: pd.DataFrame) -> pd.DataFrame:
    manifest = read_csv(ROOT / "model_selection/artifact_recovery_K12_K14/data/blind_audit_ready_model_manifest.csv")
    rows = []
    for row in manifest.sort_values("K").itertuples(index=False):
        rows.append({
            "K": int(row.K),
            "representative_seed": int(row.seed),
            "fit_scope": row.fit_scope,
            "prior_regime": row.prior_regime,
            "selection_rule": "REUSE_PREVIOUSLY_FIXED_FORMAL_REPRESENTATIVE_SEED",
            "selection_reason": "审核代表性模型实例，不选择单次最高分运行",
            "model_path": row.model_path,
            "ready_for_blind_audit": bool(row.ready_for_blind_audit),
        })
    result = pd.DataFrame(rows)
    write_csv(result, DATA / "representative_seed_selection.csv")
    return result


def build_integrated_evidence(mapping: pd.DataFrame) -> pd.DataFrame:
    summary = read_csv(DATA / "blind_model_summary.csv")
    quant = read_csv(ROOT / "model_selection/quantitative_candidate_validation/data/quantitative_candidate_summary.csv")
    prior = read_csv(ROOT / "model_selection/quantitative_candidate_validation/data/candidate_prior_sensitivity.csv")
    mapping = mapping.copy()
    mapping["K"] = mapping.true_K.astype(int)
    candidates = [12, 14, 15, 20]
    rows = []
    final_status = {
        12: "STRONG_FINAL_CANDIDATE|SEMANTICALLY_STRONG|QUANTITATIVELY_STRONG|PRIOR_SENSITIVE_CANDIDATE",
        14: "BALANCED_CANDIDATE",
        15: "SEMANTICALLY_STRONG|QUANTITATIVELY_STRONG|PRIOR_SENSITIVE_CANDIDATE",
        20: "OVER_FRAGMENTED|SEED_SENSITIVE_CANDIDATE|REFERENCE_ONLY",
    }
    resolution_gain = {
        12: "BASELINE_COMPACT_RESOLUTION",
        14: "LIMITED_GAIN_WITH_RESIDUAL_FRAGMENTATION",
        15: "TWO_MEANINGFUL_SPLITS_WITH_LOCAL_UNDER_SPLITTING",
        20: "ONE_MEANINGFUL_SPLIT_ONE_MINOR_REFINEMENT_TWO_FRAGMENTATION_EVENTS",
    }
    strength = {12: "VERY_STRONG", 14: "STRONG", 15: "VERY_STRONG", 20: "WEAKER"}
    fit = {12: "BEST_AMONG_CANDIDATES", 14: "SECOND", 15: "THIRD", 20: "WEAKEST_AMONG_CANDIDATES"}
    for k in candidates:
        alias = mapping.loc[mapping.K == k, "blind_model_id"].iloc[0]
        blind = summary[summary.blind_model_id == alias].iloc[0]
        original = quant[(quant.K == k) & (quant.prior_regime == "ORIGINAL_PRIOR")].iloc[0]
        fixed = quant[(quant.K == k) & (quant.prior_regime == "FIXED_PRIOR_K20")].iloc[0]
        prior_row = prior[prior.K == k].iloc[0]
        rows.append({
            "K": k,
            "blind_model_id": alias,
            "blind_rank": int(blind.AI_blind_rank),
            "blind_rank_confidence": blind.AI_rank_confidence,
            "blind_model_judgment": blind.AI_model_level_judgment,
            "quantitative_coherence": strength[k],
            "predictive_fit": fit[k],
            "original_NPMI_mean": float(original.NPMI_mean),
            "fixed_NPMI_mean": float(fixed.NPMI_mean),
            "original_test_perplexity_mean": float(original.test_perplexity_mean),
            "fixed_test_perplexity_mean": float(fixed.test_perplexity_mean),
            "original_topic_diversity_mean": float(original.topic_diversity_mean),
            "fixed_topic_diversity_mean": float(fixed.topic_diversity_mean),
            "original_matched_cosine_mean": float(original.mean_matched_cosine),
            "fixed_matched_cosine_mean": float(fixed.mean_matched_cosine),
            "original_matched_jaccard_mean": float(original.mean_matched_jaccard),
            "fixed_matched_jaccard_mean": float(fixed.mean_matched_jaccard),
            "original_redundancy_cosine_mean": float(original.mean_pairwise_topic_cosine_mean),
            "fixed_redundancy_cosine_mean": float(fixed.mean_pairwise_topic_cosine_mean),
            "prior_robustness": prior_row.prior_sensitivity_status,
            "seed_robustness": "SEED_SENSITIVE" if "SEED_SENSITIVE" in str(original.quantitative_status) else "NO_SEED_WARNING",
            "semantic_coherence_clear_broad_ratio": float(blind.CLEAR_ratio + blind.BROAD_ratio),
            "mean_literary_interpretability": float(blind.mean_literary_interpretability),
            "under_split_ratio": float(blind.under_split_ratio),
            "fragmentation_topic_ratio": float(blind.fragmentation_topic_ratio),
            "resolution_gain": resolution_gain[k],
            "final_K_status": final_status[k],
            "AI_review_status": "AI_PROVISIONAL",
        })
    result = pd.DataFrame(rows)
    write_csv(result, DATA / "unblinded_integrated_evidence.csv")
    return result


def write_unblinded_figure(integrated: pd.DataFrame) -> None:
    configure_plotting()
    frame = integrated.set_index("K").loc[[12, 14, 15, 20]]
    npmi = frame.original_NPMI_mean
    quant_strength = (npmi - npmi.min()) / (npmi.max() - npmi.min())
    values = pd.DataFrame({
        "Quantitative\nstrength": quant_strength,
        "Semantic\ncoherence": frame.semantic_coherence_clear_broad_ratio,
        "Interpretability": frame.mean_literary_interpretability / 5,
        "Low under-\nsplitting risk": 1 - frame.under_split_ratio,
        "Low fragmentation\nrisk": 1 - frame.fragmentation_topic_ratio,
        "Prior\nrobustness": frame.prior_robustness.eq("PRIOR_ROBUST").astype(float),
        "Seed\nrobustness": frame.seed_robustness.eq("NO_SEED_WARNING").astype(float),
    }, index=frame.index)
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    image = ax.imshow(values.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(values.columns)), values.columns)
    ax.set_yticks(np.arange(len(values.index)), [f"K{k}" for k in values.index])
    for i in range(len(values.index)):
        for j in range(len(values.columns)):
            ax.text(j, i, f"{values.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if values.iloc[i, j] > 0.62 else "#172554")
    ax.set_title("Quantitative–semantic evidence matrix (separate axes; no weighted total)")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "unblinded_quant_vs_semantic_matrix.png", bbox_inches="tight")
    plt.close(fig)


def write_unblinded_report(mapping: pd.DataFrame, integrated: pd.DataFrame, priority: pd.DataFrame) -> None:
    map_text = "；".join(f"{row.blind_model_id} → K{int(row.true_K)}" for row in mapping.sort_values("blind_model_id").itertuples(index=False))
    evidence_table = markdown_table(integrated, [
        "K", "blind_model_id", "blind_rank", "quantitative_coherence", "predictive_fit",
        "prior_robustness", "seed_robustness", "semantic_coherence_clear_broad_ratio",
        "mean_literary_interpretability", "under_split_ratio", "fragmentation_topic_ratio", "final_K_status",
    ])
    by_k = integrated.set_index("K")
    text = (
        "# 1 Blind Mapping\n\n" + map_text + "。映射仅在盲审 CSV 完成冻结并通过 SHA256 校验后读取。\n\n"
        "# 2 Blind Results Before Unblinding\n\n盲态排序为 MODEL_A > MODEL_C > MODEL_D > MODEL_B，无并列，信心 MEDIUM。解盲后没有修改任何 Topic 类型、评分、模型位次或 split/merge 判断。\n\n"
        "# 3 K12 Semantic Resolution\n\nK12 对应 MODEL_A。CLEAR+BROAD 为 10/12，文学可解释性最高，本轮未识别独立 Topic 内欠分；主要问题是两个 NOISE 残余桶及部分 BROAD 边界，研究者仍需确认紧凑结构是否掩盖低频语义。\n\n"
        "# 4 K14 Semantic Resolution\n\nK14 对应 MODEL_D。七个 CLEAR 构成可读核心，但三个 MIXED Topic 显示局部欠分，另有一对残余 Topic 被判碎片化，因此未在盲态成为最平衡模型。\n\n"
        "# 5 K15 Semantic Resolution\n\nK15 对应 MODEL_C。语义特异性较高且只有一个 NOISE；两个有意义拆分带来解释收益，但三个 MIXED/欠分 Topic 说明若干文学场仍被压合，量化先验敏感性也需保留。\n\n"
        "# 6 K20 Semantic Resolution\n\nK20 对应 MODEL_B。新增了春日物候等可解释区分，但 MIXED、NOISE 与欠分负担最高；山寺/林泉重复与残余槽位扩张显示额外复杂度主要没有转化为稳定文学边界。\n\n"
        "# 7 Cross-K Split / Merge Analysis\n\n由低分辨率向高分辨率的七个 SPLIT 裁决为：MEANINGFUL_SPLIT=3，MINOR_REFINEMENT=1，FRAGMENTATION=3，UNCERTAIN=0；另有两个 MERGE 事件分别为 MEANINGFUL_AGGREGATION 与 PROBLEMATIC_MERGE。\n\n"
        "# 8 Meaningful Resolution Gains\n\nK14→K15 的闺情季节结构与政治伦理/社会悲愤区分构成主要有意义增益；K15→K20 仅有一项明确有意义拆分与一项轻微细化，同时出现两项碎片化。\n\n"
        "# 9 Under-splitting vs Fragmentation\n\nK12 的观测风险最低；K14、K15 有局部欠分，K14 另有残余重复；K20 同时具有最高欠分比例与明显残余/宗教题材碎片化。该结论限于本轮代表 seed 与 61 个 Topic 的 AI 初审。\n\n"
        "# 10 Quantitative Evidence\n\n量化与语义证据保持为分列矩阵，不生成加权总分。K12 为原先验 NPMI 与跨 seed 稳定性领先者，K15 为固定先验 NPMI 领先者；K20 的预测拟合与匹配稳定性在四候选中较弱。\n\n"
        "# 11 Prior Robustness\n\nK14 与 K20 为 PRIOR_ROBUST；K12 与 K15 为 PRIOR_SENSITIVE。先验稳健性是独立证据，不反向更改盲审语义结论。\n\n"
        "# 12 Seed Robustness\n\nK20 带有 SEED_SENSITIVE 警示；K12、K14、K15 在既有候选验证中无 seed warning。语义审核使用预先固定的代表 seed，而非单次最高 NPMI seed。\n\n"
        "# 13 Quantitative–Semantic Trade-off\n\n" + evidence_table + "\n\n以上各轴独立呈现；没有通过任意权重合成为单一分数。\n\n"
        "# 14 Final Candidate Judgment\n\n最终状态：PREFER_K12。其盲态语义平衡位居第一，同时具备强量化一致性与 seed 稳定性；代价是 prior sensitivity、两个残余桶及较紧凑分辨率。K15 是最接近的语义备选，但局部欠分与 prior sensitivity 仍在；K14 稳健却盲态语义质量较弱；K20 的额外主题以碎片化/轻微细化为主。\n\n"
        "# 15 Researcher Review Queue\n\n优先队列共 " + str(len(priority)) + " 项，其中 Priority 1=" + str(int((priority.priority == 1).sum())) + "。研究者应先核验 MIXED、under-split、文本确认的 fragmentation pair，以及直接影响候选胜负的 split/merge。\n\n"
        "# 16 Limitations\n\n这是 AI-assisted blind audit，评分全部为 AI_PROVISIONAL，尚未完成研究者复核。代表 seed 不能穷尽每个候选的 seed 内部变异；family 与 split/merge 仍依赖解释判断。若最终采用非 K20，状态仅记录为 DOWNSTREAM_RERUN_WOULD_BE_REQUIRED；本任务没有重跑或修改任何历史下游分析。\n"
    )
    (AUDIT_ROOT / "QTS_Four_Model_Unblinded_Integrated_Audit.md").write_text(text, encoding="utf-8")


def unblind_phase() -> None:
    before = verify_frozen()
    mapping = read_csv(PRIVATE / "blind_mapping_secret.csv")
    if sorted(mapping.true_K.astype(int).tolist()) != [12, 14, 15, 20]:
        raise ValueError("private mapping 的候选集合异常")
    build_seed_selection(mapping)
    integrated = build_integrated_evidence(mapping)
    priority = read_csv(DATA / "blind_researcher_priority_queue.csv")
    write_unblinded_figure(integrated)
    write_unblinded_report(mapping, integrated, priority)
    after = verify_frozen()
    if before != after:
        raise RuntimeError("解盲过程修改了冻结盲审结果")
    (AUDIT / "unblind_integrity.json").write_text(
        json.dumps({
            "blind_frozen_sha256_before": before,
            "blind_frozen_sha256_after": after,
            "blind_scores_modified_after_unblind": False,
            "model_training_calls": 0,
            "downstream_rerun_performed": False,
            "final_selection_status": "PREFER_K12",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"UNBLIND_COMPLETE blind_sha256_unchanged={before}")


def main() -> None:
    parser = argparse.ArgumentParser(description="四模型盲审汇总、冻结与解盲")
    parser.add_argument("--phase", required=True, choices=["blind", "freeze", "unblind"])
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    if args.phase == "blind":
        blind_phase()
    elif args.phase == "freeze":
        freeze_phase()
    else:
        unblind_phase()


if __name__ == "__main__":
    main()
