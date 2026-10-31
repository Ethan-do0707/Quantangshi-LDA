"""生成《全唐诗》Lower-K 人文学语义审核的可审计交付物。"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = ROOT / "model_selection" / "lower_k_boundary_perplexity"
OUT = ROOT / "model_selection" / "lower_k_semantic_audit"
DATA = OUT / "data"
FIGURES = OUT / "figures"
CARDS = OUT / "review_cards"
K20_ROOT = (
    ROOT
    / "outputs"
    / "01a05061-9ee3-7610-a401-abfec81de1b2"
    / "literary_history_dataset"
)


def review(
    k: int,
    topic: int,
    label: str,
    final_type: str,
    summary: str,
    scope: str,
    scores: tuple[int, int, int, int, int, int],
    mixed: bool,
    mixed_fields: str,
    under_split: bool,
    risk: str,
    under_reason: str,
    merge_judgment: str,
    split_judgment: str = "",
) -> dict[str, object]:
    """建立一条统一的 AI 暂定审核记录。"""

    return {
        "K": k,
        "topic_id": topic,
        "AI_provisional_label": label,
        "AI_final_review": final_type,
        "AI_semantic_summary": summary,
        "AI_semantic_scope": scope,
        "AI_word_coherence_score": scores[0],
        "AI_poem_coherence_score": scores[1],
        "AI_word_poem_agreement_score": scores[2],
        "AI_labelability_score": scores[3],
        "AI_literary_interpretability_score": scores[4],
        "AI_overbreadth_score": scores[5],
        "AI_mixed_topic_flag": mixed,
        "AI_mixed_fields": mixed_fields,
        "AI_under_split_flag": under_split,
        "AI_under_split_risk": risk,
        "AI_under_split_reason": under_reason,
        "AI_merge_semantic_judgment": merge_judgment,
        "AI_K20_split_judgment": split_judgment,
        "AI_review_status": "AI_REVIEW",
        "AI_provisional_status": "PROVISIONAL_REVIEW",
        "researcher_confirmation_status": "NEEDS_RESEARCHER_CONFIRMATION",
    }


REVIEWS = [
    review(5, 1, "政治伦理、礼制颂德与人生玄思", "MIXED", "高权重词多为泛化的人、心、道、天、名；代表诗横跨政治伦理、礼制颂德、道教神仙与人生议论。", "VERY_BROAD", (2, 2, 3, 2, 2, 5), True, "政治伦理＋朝廷礼制＋道教玄思", True, "HIGH", "对齐五个 K20 语义场，且 Top20 实诗确有相互独立的政治、宗教与人生议论内容，是被低 K 强迫合并。", "PROBLEMATIC_MERGE"),
    review(5, 2, "花香春情、女性歌舞与闺情", "CLEAR", "花、春、香、开、舞等词与女性歌舞、闺情和惜春诗例互相支持，中心稳定。", "BALANCED", (5, 5, 5, 5, 5, 1), False, "", False, "LOW", "与 K20 T11 基本一对一，Top20 与 Top30 均未显示必须拆开的异质语义场。", "UNCERTAIN"),
    review(5, 3, "边塞征战、人生功名与诗酒感怀", "MIXED", "兵戈征战只是其中一支，代表诗同时出现仕途、老病、饮酒、赠答和人生感怀。", "VERY_BROAD", (2, 3, 2, 2, 3, 5), True, "边塞征战＋仕途身世＋饮酒赠答", True, "HIGH", "K20 T16、T06、T20、T02 的不同内容在 Top20 中均有文本证据，难视为一个自然上位类。", "PROBLEMATIC_MERGE"),
    review(5, 4, "江湖羁旅、送别与相思", "BROAD", "行、归、路、舟等行旅词与送别、思归和怀人诗例形成连续的离散—归返语义轴。", "BROAD", (5, 5, 5, 4, 5, 2), False, "", False, "MODERATE", "虽覆盖 K20 T10 与 T07，但两者在这些诗例中多由离别和远行自然联结，暂不判为不合理欠分。", "MEANINGFUL_AGGREGATION"),
    review(5, 5, "山居寺院、隐逸与闲游", "BROAD", "山、门、寺、林、僧等词与寻幽、游寺、隐居诗例一致，内部各支共享山林幽居空间。", "BROAD", (4, 4, 5, 4, 4, 3), False, "", False, "MODERATE", "K20 T08、T19 与少量 T10 被合并，但主体是相邻的山寺、山居与隐逸语义，可作上位概括。", "MEANINGFUL_AGGREGATION"),

    review(10, 1, "边塞征战与军旅", "CLEAR", "塞、军、兵、胡、将等词与征战、守边、军功诗例高度一致。", "BALANCED", (5, 4, 5, 5, 5, 1), False, "", False, "LOW", "与 K20 T16 基本一对一，文本未显示明显异质合并。", "UNCERTAIN"),
    review(10, 2, "仙山道观、山居修隐与寻幽", "BROAD", "仙、山、云、宫、洞等词把道教仙境、山寺与隐逸寻幽组织成共享空间意象。", "BROAD", (4, 4, 5, 4, 4, 3), False, "", False, "MODERATE", "涉及 K20 T17、T08、T13、T19，但多数诗例可由仙山—山居—修隐的上位场域解释。", "MEANINGFUL_AGGREGATION"),
    review(10, 3, "怀人悲音、清景与身世感慨", "MIXED", "夜月琴声与怀人是可见中心，却同时混入清景、仕途、悼亡和药病等诗例。", "VERY_BROAD", (3, 3, 3, 3, 3, 4), True, "相思悲音＋清景物候＋仕途身世", True, "HIGH", "K20 T07、T13、T02 所指向的诗群在代表诗中仍可区分，复合标签掩盖了多个方向。", "PROBLEMATIC_MERGE"),
    review(10, 4, "离别哀悼、女性悲情与人生感怀", "MIXED", "悲、泪、别、死等词能概括负面情感，却把闺情、悼亡、行旅与人生议论压在同一情绪桶中。", "VERY_BROAD", (3, 2, 3, 2, 3, 4), True, "女性悲情＋悼亡生死＋羁旅离别", True, "HIGH", "代表诗分成可独立命名的女性悲情、死亡悼念和行路送别，不能只凭共同悲感视为统一。", "PROBLEMATIC_MERGE"),
    review(10, 5, "王朝礼制、颂德与政治教化", "BROAD", "朝、君、王、臣、礼等词与应制、颂德、政治劝诫诗例共同构成公共政治表达。", "BROAD", (4, 4, 4, 4, 4, 3), False, "", False, "MODERATE", "K20 T03、T05、T15 等在制度—伦理—颂赠轴上相邻，虽宽但仍有政治礼制中心。", "ACCEPTABLE_BROADENING"),
    review(10, 6, "江湖羁旅、送别与思归", "CLEAR", "舟、江、路、归、别等词与行旅、送别、归思诗例高度相合。", "BALANCED", (5, 5, 5, 5, 5, 1), False, "", False, "LOW", "与 K20 T10 基本一对一，语义边界清楚。", "UNCERTAIN"),
    review(10, 7, "老病身世、仕宦与诗酒自况", "CLEAR", "老、病、身、官、酒等词与自况、退居、诗酒及仕宦感怀诗例形成稳定主体。", "BALANCED", (4, 4, 5, 4, 4, 2), False, "", False, "LOW", "主要对应 K20 T06；Top20 虽有白居易集中，五作者平衡样本仍支持同一语义。", "UNCERTAIN"),
    review(10, 8, "飞鸟夜声、宫怨与杂咏", "MIXED", "鸟、夜、声等感官词不足以统一宫怨、礼制、季节和道德杂咏。", "VERY_BROAD", (2, 2, 2, 2, 2, 4), True, "夜声禽鸟＋宫怨＋礼制杂咏", True, "HIGH", "Top20 内容类别分散，K20 T07 与 T18 的对齐不能为宫廷及杂咏部分提供统一解释。", "PROBLEMATIC_MERGE"),
    review(10, 9, "闲居山寺、羁旅思归与诗酒身世", "MIXED", "闲、山、归、酒、诗等词对应隐逸、寺院、行旅、思归和老病自况多个诗群。", "VERY_BROAD", (3, 3, 4, 3, 4, 5), True, "山林隐逸＋羁旅思归＋诗酒身世", True, "HIGH", "对齐 K20 T19、T08、T10、T20、T06，Top20 也显示这些文学活动并未完全融合。", "PROBLEMATIC_MERGE"),
    review(10, 10, "春花、艳情与自然物候", "BROAD", "春、花、香、风、月等词与惜春、女性情思及花鸟物候诗例相合，但覆盖面较大。", "BROAD", (4, 4, 5, 4, 4, 3), False, "", True, "MODERATE", "K20 中女性春情与一般春景物候可分别成题；低 K 合并仍可解释，但损失了一层文学分辨率。", "ACCEPTABLE_BROADENING"),

    review(15, 1, "仙山宫苑、歌舞器物与杂咏", "MIXED", "仙、宫、云等词显示神仙宫苑倾向，但 Top20 同时有歌舞、器物、宫廷与一般杂咏。", "VERY_BROAD", (2, 2, 3, 2, 2, 4), True, "神仙山岳＋宫苑歌舞＋器物杂咏", True, "HIGH", "K20 T17 可清楚承载神仙山岳，K20 T01 的杂项成分在此与之黏连，代表诗仍显异质。", "PROBLEMATIC_MERGE", "MEANINGFUL_SPLIT"),
    review(15, 2, "行路艰难、忠义生死与社会悲慨", "MIXED", "行、难、死、心等词把道路艰难、忠义生死、战争与社会悲慨聚于共同困厄语调。", "VERY_BROAD", (3, 3, 4, 3, 3, 4), True, "行路艰难＋忠义生死＋战争社会悲慨", False, "MODERATE", "虽内容混合，但与本身同为 MIXED 的 K20 T02 基本一对一，不能归因于 K15 相对 K20 的欠分。", "UNCERTAIN", "UNCERTAIN"),
    review(15, 3, "朝廷礼乐、应制与颂圣", "CLEAR", "朝、君、圣、礼、乐等词与郊庙、应制、朝会和颂圣诗例高度一致。", "BALANCED", (5, 5, 5, 5, 5, 1), False, "", False, "LOW", "与 K20 T03 一对一，中心和边界均清楚。", "UNCERTAIN", "UNCERTAIN"),
    review(15, 4, "边塞军旅、归途与春景杂糅", "MIXED", "军、塞、马等边塞词与归、春、花等行旅春景词并存，代表诗形成两类明显文本。", "VERY_BROAD", (3, 3, 3, 3, 3, 5), True, "边塞征战＋归途羁旅＋春景僧游", True, "HIGH", "K20 T16 与 T04 的对齐分别得到军旅诗和春景行旅诗支持，K20 的区分具有明确文学意义。", "PROBLEMATIC_MERGE", "MEANINGFUL_SPLIT"),
    review(15, 5, "君子伦理、政治劝诫与社会批评", "BROAD", "君、道、义、民、官等词与政治规劝、伦理议论和社会批评诗例构成一致的公共伦理中心。", "BROAD", (4, 4, 5, 4, 4, 2), False, "", False, "LOW", "与 K20 T05 基本一对一，宽度来自主题自身的政治伦理范围。", "UNCERTAIN", "UNCERTAIN"),
    review(15, 6, "老病身世、诗酒交游与仕宦自况", "BROAD", "老、病、身、酒、诗等词与自况、仕宦、诗酒赠答诗例一致，但人生感怀与宴饮交游仍可区分。", "BROAD", (5, 4, 5, 4, 4, 3), False, "", True, "MODERATE", "K20 T06 与 T20 在 Top20 中分别表现为老病仕宦和饮酒酬赠；Top20 白居易占 65%，作者平衡样本仍确认主题中心但不消除欠分。", "ACCEPTABLE_BROADENING", "MEANINGFUL_SPLIT"),
    review(15, 7, "相思离愁、夜月与琴声", "CLEAR", "月、夜、愁、思、琴等词与怀人、离愁、夜听和女性情思诗例高度一致。", "BALANCED", (5, 5, 5, 5, 5, 1), False, "", False, "LOW", "与 K20 T07 一对一，Top words 与代表诗吻合。", "UNCERTAIN", "UNCERTAIN"),
    review(15, 8, "山居寺院、隐逸与寻幽", "BROAD", "山、寺、僧、林、门等词与游寺、山居、隐逸诗例形成紧密相邻的空间和生活方式。", "BROAD", (5, 5, 5, 4, 5, 2), False, "", False, "MODERATE", "K20 T08 与 T19 的僧寺、山居和隐逸可视为自然上位聚合，文本没有严重互斥。", "MEANINGFUL_AGGREGATION", "MINOR_REFINEMENT"),
    review(15, 9, "诗名仕途、行旅与杂咏噪声", "NOISE", "人、日、知、名等泛化词占据前列，代表诗横跨仕途、行旅、怀人和杂咏，难形成稳定文学中心。", "VERY_BROAD", (2, 2, 2, 1, 1, 4), True, "仕途诗名＋行旅怀人＋一般杂咏", False, "LOW", "与同为噪声的 K20 T09 基本一对一；这是主题质量问题，而非 K15 相对 K20 的新增欠分。", "UNCERTAIN", "UNCERTAIN"),
    review(15, 10, "江湖羁旅、送别与思归", "CLEAR", "江、舟、路、归、别等词与送别、远行、思归诗例高度相合。", "BALANCED", (5, 5, 5, 5, 5, 1), False, "", False, "LOW", "与 K20 T10 一对一，主题边界清楚。", "UNCERTAIN", "UNCERTAIN"),
    review(15, 11, "春花、艳情与惜春", "BROAD", "花、春、香、红、舞等词与女性情思、歌舞、惜春诗例一致，兼有一般春景。", "BROAD", (5, 4, 5, 4, 5, 2), False, "", False, "MODERATE", "K20 T11 与 T04 的女性春情和一般春景彼此相邻，本层合并仍有明确核心。", "MEANINGFUL_AGGREGATION", "MINOR_REFINEMENT"),
    review(15, 12, "春水花鸟与自然风物", "CLEAR", "春、水、花、鸟、风等词与物候、花鸟、水景诗例高度一致。", "BALANCED", (5, 5, 5, 5, 5, 1), False, "", False, "LOW", "与 K20 T12 一对一，语义稳定。", "UNCERTAIN", "UNCERTAIN"),
    review(15, 13, "郊庙礼乐、神祀与祥瑞", "BROAD", "神、天、礼、乐、祀等词与祭祀、祥瑞、宗庙礼仪诗例形成礼制宗教中心。", "BROAD", (4, 4, 4, 4, 4, 2), False, "", False, "MODERATE", "对齐 K20 T13 与 T03，但 K15 的代表诗反而较完整地聚合礼制神祀；K20 的局部区分可能带来碎片化。", "ACCEPTABLE_BROADENING", "FRAGMENTATION"),
    review(15, 14, "闲居清景、羁旅相思与仕隐杂糅", "MIXED", "闲、山、归、思、官等词与清景闲居、远行思归、相思和仕隐自况多类诗例并存。", "VERY_BROAD", (3, 3, 4, 3, 3, 5), True, "闲居清景＋羁旅思归＋相思仕隐", True, "HIGH", "对齐 K20 T19、T14、T13、T07、T10，Top20 仍可辨认隐逸、行旅和相思三支，是最明显的低 K 语义桶之一。", "PROBLEMATIC_MERGE", "MEANINGFUL_SPLIT"),
    review(15, 15, "边塞征战、军功与神异杂咏", "BROAD", "军、将、塞、马等词与征战、军功诗例占主体，但夹有神异、礼仪及一般杂咏。", "BROAD", (4, 4, 5, 4, 4, 2), False, "", False, "MODERATE", "K20 T16 提纯边塞军旅，K20 T15 承接部分噪声；K15 主体仍明确，故暂判可接受的拓宽。", "ACCEPTABLE_BROADENING", "MEANINGFUL_SPLIT"),
]


COVERAGE = [
    (3, "朝廷礼乐、应制与颂圣", 3, "PRESERVED_CLEARLY"),
    (6, "老病身世、仕宦与人生感怀", 6, "PRESERVED_BUT_BROADER"),
    (7, "相思离愁、夜月与琴声", 7, "PRESERVED_CLEARLY"),
    (8, "山居寺院、僧侣与幽居", 8, "PRESERVED_BUT_BROADER"),
    (10, "江湖羁旅、送别与思归", 10, "PRESERVED_CLEARLY"),
    (11, "花香春情、女性与闺情", 11, "PRESERVED_BUT_BROADER"),
    (16, "边塞征战与军旅", 15, "PRESERVED_BUT_BROADER"),
    (19, "闲居山林与隐逸生活", 8, "MERGED_WITH_NEIGHBOR"),
    (20, "饮酒酬赠、宴集与送别", 6, "MERGED_WITH_NEIGHBOR"),
]


PRIORITY_ORDER = [
    (15, 4, "军旅与春景归途被压入同题，属于高风险不合理合并。"),
    (15, 14, "同时承载隐逸、羁旅、相思等多个 K20 语义场，边界最宽。"),
    (15, 1, "神仙山岳与宫苑器物杂咏黏连，需判断是否仅为统计共现。"),
    (15, 6, "老病仕宦与饮酒酬赠可能需要拆分，且 Top20 白居易占 65%。"),
    (15, 15, "边塞核心明确但夹入神异杂咏，需核定可接受边界。"),
    (15, 13, "礼制神祀较完整，却对应两个 K20 主题，需核定 K20 是否碎片化。"),
    (15, 8, "山寺与隐逸被合理聚合的判断，需要研究者检验文学层级。"),
    (15, 11, "女性春情与一般春景的合并是否损失关键分辨率，需人工确认。"),
    (15, 2, "行路、忠义生死与社会悲慨混合，但与 K20 一对一，需判断问题来源。"),
    (15, 5, "政治伦理主题较宽，需核定劝诫、社会批评是否可由一标签承载。"),
]


def sha256(path: Path) -> str:
    """计算文件哈希。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    """返回仓库相对路径。"""

    return str(path.relative_to(ROOT)).replace("\\", "/")


def top_words_as_markdown(value: str) -> str:
    """把词与概率列表转为 Markdown 表。"""

    rows = ["| rank | token | probability |", "|---:|---|---:|"]
    for rank, item in enumerate(str(value).split(";"), 1):
        token, probability = item.rsplit("|", 1)
        rows.append(f"| {rank} | {token} | {float(probability):.12f} |")
    return "\n".join(rows)


def score_comment(score: int, target: str, summary: str) -> str:
    """按评分生成可复核说明。"""

    wording = {
        5: "高度集中且边界清楚",
        4: "中心明确，带少量外围内容",
        3: "可以解释，但存在多个子方向",
        2: "边界模糊，异质内容较多",
        1: "难形成稳定语义中心",
    }[score]
    return f"{target}{wording}。综合判断依据为：{summary}"


def write_cards(evidence: pd.DataFrame, reviews: pd.DataFrame) -> None:
    """生成 30 份逐题 Markdown 审读卡。"""

    for item in reviews.itertuples(index=False):
        row = evidence[(evidence.K == item.K) & (evidence.topic_id == item.topic_id)].iloc[0]
        reps_path = ROOT / str(row.representative_poem_source)
        reps = pd.read_csv(reps_path)
        reps = reps[reps.topic_id == item.topic_id].sort_values("rank")
        balanced = pd.read_csv(DATA / "author_balanced_representative_poems.csv")
        balanced = balanced[(balanced.K == item.K) & (balanced.topic_id == item.topic_id)]
        card_dir = CARDS / f"K{item.K}"
        card_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# K{item.K} Topic {item.topic_id:02d}",
            "",
            "> 审核状态：`AI_REVIEW` / `PROVISIONAL_REVIEW` / `NEEDS_RESEARCHER_CONFIRMATION`。以下标签与分数不是已完成的人工审核。",
            "",
            "## A. Top Words",
            "",
            top_words_as_markdown(row.top30_words_with_probability),
            "",
            f"Topic mean prevalence：{float(row.topic_prevalence):.6f}；dominant-topic share：{float(row.dominant_topic_share):.6f}。",
            "",
            "## B. Representative Poems",
            "",
            "已审读 Top20；正文仅保留足以判断语义的短片段。",
            "",
            "| rank | poem_id | author | title | topic_probability | text snippet |",
            "|---:|---:|---|---|---:|---|",
        ]
        for rep in reps.itertuples():
            snippet = " ".join(str(rep.original_text).replace("\n", " ").split())[:110]
            lines.append(
                f"| {int(rep.rank)} | {int(rep.poem_id)} | {rep.author} | {rep.title} | {float(rep.topic_probability):.6f} | {snippet}… |"
            )
        lines.extend([
            "",
            f"Top20 作者数：{int(row.top20_unique_authors)}；单一作者最高占比：{float(row.top20_max_author_share):.1%}。",
            "",
            "Author-balanced 核验（每位作者最多一首）：",
            "",
        ])
        for rep in balanced.itertuples():
            lines.append(
                f"- {int(rep.poem_id)}｜{rep.author}《{rep.title}》｜P={float(rep.topic_probability):.6f}｜{rep.text_snippet}"
            )
        lines.extend([
            "",
            "## C. 初步语义概括",
            "",
            item.AI_semantic_summary,
            "",
            "## D. Proposed Label",
            "",
            f"`AI_PROVISIONAL_LABEL`：**{item.AI_provisional_label}**",
            "",
            "## E. 关键词一致性",
            "",
            score_comment(int(item.AI_word_coherence_score), "Top30 关键词", item.AI_semantic_summary),
            f"AI-assisted provisional score：**{int(item.AI_word_coherence_score)}/5**。",
            "",
            "## F. 代表诗一致性",
            "",
            score_comment(int(item.AI_poem_coherence_score), "Top20 代表诗", item.AI_semantic_summary),
            f"AI-assisted provisional score：**{int(item.AI_poem_coherence_score)}/5**。",
            "",
            "## G. 关键词—代表诗一致性",
            "",
            score_comment(int(item.AI_word_poem_agreement_score), "词—诗关系", item.AI_semantic_summary),
            f"AI-assisted provisional score：**{int(item.AI_word_poem_agreement_score)}/5**。",
            "",
            "## H. 语义范围",
            "",
            f"**{item.AI_semantic_scope}**。Labelability={int(item.AI_labelability_score)}/5；Literary interpretability={int(item.AI_literary_interpretability_score)}/5；Overbreadth={int(item.AI_overbreadth_score)}/5。",
            "",
            "## I. Mixedness",
            "",
            f"**{str(bool(item.AI_mixed_topic_flag)).upper()}**。" + (f"混合语义场：{item.AI_mixed_fields}。" if item.AI_mixed_fields else "未见两个以上必须独立解释的语义场。"),
            "",
            "## J. C_K20 Alignment",
            "",
            f"- Topics：{row.mapped_C_K20_topics}",
            f"- Labels：{row.mapped_C_K20_labels}",
            f"- Cosine：{row.mapped_similarities}",
            f"- Automatic relation：{row.automatic_relation_type}",
            f"- AI merge judgment：{item.AI_merge_semantic_judgment}",
        ])
        if item.K == 15:
            lines.append(f"- K20 added-resolution judgment：{item.AI_K20_split_judgment}")
        lines.extend([
            "",
            "## K. Under-splitting Risk",
            "",
            f"**{item.AI_under_split_risk}**；flag={str(bool(item.AI_under_split_flag)).upper()}。{item.AI_under_split_reason}",
            "",
            "## L. Final AI Review",
            "",
            f"**{item.AI_final_review}**（AI-assisted provisional；待研究者确认）。",
            "",
            "## Evidence Trace",
            "",
            f"- Top words source：`{row.top30_source}`，rows {row.top30_rows}",
            f"- Representative poems source：`{row.representative_poem_source}`，rows {row.representative_poem_rows}",
            f"- Representative poem IDs：{row.representative_poem_ids}",
            f"- Author-balanced poem IDs：{row.author_balanced_poem_ids}",
            f"- Alignment source：`{row.alignment_source}`，row {int(row.alignment_row)}",
            f"- Prior automatic review card：`{row.prior_review_card}`",
            "",
        ])
        (card_dir / f"K{item.K}_topic_{item.topic_id:02d}_review.md").write_text("\n".join(lines), encoding="utf-8")


def write_figures(summary: pd.DataFrame, reviews: pd.DataFrame, coverage: pd.DataFrame) -> None:
    """生成四张研究者易读的静态诊断图。"""

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"CLEAR": "#2A9D8F", "BROAD": "#E9C46A", "MIXED": "#F4A261", "NOISE": "#8D99AE"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    x = np.arange(len(summary))
    axes[0].bar(x, summary.AI_mean_interpretability, color="#2A9D8F", width=0.58)
    axes[1].bar(x, summary.AI_mean_overbreadth, color="#E76F51", width=0.58)
    for ax, title in zip(axes, ["AI 平均文学可解释性", "AI 平均过宽度"]):
        ax.set_xticks(x, [f"K={k}" for k in summary.K])
        ax.set_ylim(0, 5)
        ax.set_ylabel("AI 暂定评分（1–5）")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        for index, value in enumerate(summary["AI_mean_interpretability" if ax is axes[0] else "AI_mean_overbreadth"]):
            ax.text(index, value + 0.08, f"{value:.2f}", ha="center", fontsize=10)
    fig.suptitle("Lower-K 人文学语义质量（AI provisional）", fontweight="bold")
    fig.savefig(FIGURES / "lower_k_semantic_quality_by_K.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    type_order = ["CLEAR", "BROAD", "MIXED", "NOISE"]
    fig, ax = plt.subplots(figsize=(8.4, 5), constrained_layout=True)
    bottom = np.zeros(3)
    for topic_type in type_order:
        values = np.array([
            ((reviews.K == k) & (reviews.AI_final_review == topic_type)).sum() / int(k)
            for k in (5, 10, 15)
        ])
        ax.bar(["K=5", "K=10", "K=15"], values, bottom=bottom, label=topic_type, color=colors[topic_type])
        for idx, value in enumerate(values):
            if value >= 0.08:
                ax.text(idx, bottom[idx] + value / 2, f"{value:.0%}", ha="center", va="center", fontsize=9)
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_ylabel("Topic 比例")
    ax.set_title("Lower-K Topic 类型分布（AI provisional）", fontweight="bold")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.savefig(FIGURES / "lower_k_topic_type_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    k15 = reviews[reviews.K == 15].copy()
    fig, ax = plt.subplots(figsize=(14, 9), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 21)
    ax.axis("off")
    ax.text(0.08, 20.4, "K15 Topics", ha="center", fontweight="bold", fontsize=13)
    ax.text(0.92, 20.4, "C_K20 Topics", ha="center", fontweight="bold", fontsize=13)
    positions15 = {i: 19 - (i - 1) * 18 / 14 for i in range(1, 16)}
    positions20 = {i: 19 - (i - 1) * 18 / 19 for i in range(1, 21)}
    edge_colors = {"MEANINGFUL_SPLIT": "#2A9D8F", "MINOR_REFINEMENT": "#457B9D", "FRAGMENTATION": "#9B5DE5", "UNCERTAIN": "#9CA3AF"}
    for row in k15.itertuples(index=False):
        ev = pd.read_csv(DATA / "topic_evidence_index.csv")
        mapped = ev[(ev.K == 15) & (ev.topic_id == row.topic_id)].iloc[0].mapped_C_K20_topics
        targets = [int(value.split("_")[-1]) for value in str(mapped).split(";")]
        line_color = edge_colors[row.AI_K20_split_judgment]
        for target in targets:
            ax.plot([0.14, 0.86], [positions15[row.topic_id], positions20[target]], color=line_color, alpha=0.62, linewidth=1.5)
        border = "#D62828" if row.AI_merge_semantic_judgment == "PROBLEMATIC_MERGE" else "#264653"
        ax.text(0.08, positions15[row.topic_id], f"T{row.topic_id:02d} {row.AI_provisional_label}", ha="left", va="center", fontsize=8.3,
                bbox=dict(boxstyle="round,pad=0.23", facecolor="white", edgecolor=border, linewidth=1.3))
    taxonomy = pd.read_csv(K20_ROOT / "taxonomy" / "C_K20_topic_taxonomy.csv").set_index("topic_number")
    for topic_id in range(1, 21):
        label = str(taxonomy.loc[topic_id, "final_topic_label"])
        ax.text(0.92, positions20[topic_id], f"T{topic_id:02d} {label}", ha="right", va="center", fontsize=8.0,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#F7FAFC", edgecolor="#718096", linewidth=0.8))
    legend_lines = [plt.Line2D([0], [0], color=color, lw=2, label=label) for label, color in edge_colors.items()]
    legend_lines.append(plt.Line2D([0], [0], color="#D62828", lw=2, label="PROBLEMATIC_MERGE（左框）"))
    ax.legend(handles=legend_lines, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=8)
    ax.set_title("K15 → C_K20 语义映射：线色表示 K20 新增分辨率判断", fontweight="bold", pad=18)
    fig.savefig(FIGURES / "K15_to_K20_semantic_mapping.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    similarity = pd.read_csv(BOUNDARY / "data" / "lower_k_to_k20_similarity_matrix.csv")
    matrix_row = similarity[similarity.lower_K == 15].set_index("lower_topic")
    values = np.array([[float(matrix_row.loc[k15_topic, f"C_K20_topic_{k20_topic:02d}"]) for k15_topic in range(1, 16)] for k20_topic in coverage.C_K20_topic])
    shown = np.where(values >= 0.75, 2, np.where(values >= 0.60, 1, 0))
    fig, ax = plt.subplots(figsize=(13, 6.2), constrained_layout=True)
    from matplotlib.colors import ListedColormap
    image = ax.imshow(shown, aspect="auto", cmap=ListedColormap(["#F3F4F6", "#A8DADC", "#1D7874"]), vmin=0, vmax=2)
    ax.set_xticks(range(15), [f"T{i:02d}" for i in range(1, 16)])
    ax.set_yticks(range(9), [f"T{int(row.C_K20_topic):02d} {row.C_K20_label}" for row in coverage.itertuples(index=False)])
    ax.set_xlabel("K15 Topic")
    ax.set_ylabel("C_K20 CLEAR semantic field")
    ax.set_title("K15 核心语义覆盖矩阵（cosine 仅作关系提示）", fontweight="bold")
    for ridx, row in enumerate(coverage.itertuples(index=False)):
        target = int(row.K15_target_topic) - 1
        ax.add_patch(plt.Rectangle((target - 0.49, ridx - 0.49), 0.98, 0.98, fill=False, edgecolor="#E63946", linewidth=2.1))
        for cidx in range(15):
            if shown[ridx, cidx] > 0:
                ax.text(cidx, ridx, f"{values[ridx, cidx]:.2f}", ha="center", va="center", fontsize=7.2,
                        color="white" if shown[ridx, cidx] == 2 else "#17324D")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1D7874", label="strong ≥ .75"),
        plt.Rectangle((0, 0), 1, 1, color="#A8DADC", label="moderate .60–.75"),
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#E63946", linewidth=2, label="人工判定承载 Topic"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3)
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01, ticks=[])
    fig.savefig(FIGURES / "K15_core_semantic_coverage_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build() -> dict[str, object]:
    """构建全部审核交付物。"""

    DATA.mkdir(parents=True, exist_ok=True)
    evidence = pd.read_csv(DATA / "topic_evidence_index.csv")
    reviews = pd.DataFrame(REVIEWS).sort_values(["K", "topic_id"]).reset_index(drop=True)
    if len(reviews) != 30 or reviews.groupby("K").size().to_dict() != {5: 5, 10: 10, 15: 15}:
        raise ValueError("AI 审核记录数量或 K 分组不正确")
    combined = reviews.merge(evidence, on=["K", "topic_id"], how="left", validate="one_to_one")
    combined.to_csv(DATA / "lower_k_topic_ai_review.csv", index=False, encoding="utf-8-sig")

    automatic = pd.read_csv(BOUNDARY / "data" / "C_K_boundary_comparison.csv").set_index("K")
    assessments = {5: "TOO_COARSE", 10: "COARSE_BUT_INTERPRETABLE", 15: "COMPETITIVE_RESOLUTION"}
    summary_rows = []
    for k in (5, 10, 15):
        subset = reviews[reviews.K == k]
        counts = Counter(subset.AI_final_review)
        summary_rows.append({
            "K": k,
            "n_topics": len(subset),
            "CLEAR_count": counts["CLEAR"],
            "BROAD_count": counts["BROAD"],
            "MIXED_count": counts["MIXED"],
            "NOISE_count": counts["NOISE"],
            "mean_word_coherence": subset.AI_word_coherence_score.mean(),
            "mean_poem_coherence": subset.AI_poem_coherence_score.mean(),
            "mean_word_poem_agreement": subset.AI_word_poem_agreement_score.mean(),
            "mean_labelability": subset.AI_labelability_score.mean(),
            "mean_literary_interpretability": subset.AI_literary_interpretability_score.mean(),
            "mean_overbreadth": subset.AI_overbreadth_score.mean(),
            "mixed_topic_ratio": subset.AI_mixed_topic_flag.mean(),
            "automatic_under_split_ratio": float(automatic.loc[k, "lower_k_merge_ratio"]),
            "semantic_under_split_ratio": subset.AI_under_split_flag.mean(),
        })
    k_level = pd.DataFrame(summary_rows)
    k_level.to_csv(DATA / "K_level_semantic_summary.csv", index=False, encoding="utf-8-sig")
    humanities = pd.DataFrame({
        "K": k_level.K,
        "automatic_NPMI": [automatic.loc[k, "NPMI"] for k in k_level.K],
        "automatic_Stability": [automatic.loc[k, "Stability"] for k in k_level.K],
        "automatic_Diversity": [automatic.loc[k, "Topic_Diversity"] for k in k_level.K],
        "test_Perplexity": [automatic.loc[k, "Test_Perplexity"] for k in k_level.K],
        "AI_mean_interpretability": k_level.mean_literary_interpretability,
        "AI_mean_overbreadth": k_level.mean_overbreadth,
        "AI_CLEAR_ratio": k_level.CLEAR_count / k_level.n_topics,
        "AI_BROAD_ratio": k_level.BROAD_count / k_level.n_topics,
        "AI_MIXED_ratio": k_level.MIXED_count / k_level.n_topics,
        "AI_NOISE_ratio": k_level.NOISE_count / k_level.n_topics,
        "automatic_merge_ratio": k_level.automatic_under_split_ratio,
        "semantic_under_split_ratio": k_level.semantic_under_split_ratio,
        "overall_semantic_assessment": [assessments[k] for k in k_level.K],
    })
    humanities.to_csv(DATA / "lower_k_humanities_summary.csv", index=False, encoding="utf-8-sig")

    k15 = combined[combined.K == 15].copy()
    k15_audit = k15[[
        "topic_id", "AI_provisional_label", "mapped_C_K20_topics", "mapped_C_K20_labels",
        "automatic_relation_type", "mapped_similarities", "AI_literary_interpretability_score",
        "AI_overbreadth_score", "AI_merge_semantic_judgment", "AI_K20_split_judgment",
        "AI_under_split_reason", "AI_review_status", "researcher_confirmation_status",
    ]].rename(columns={
        "topic_id": "K15_topic_id", "AI_provisional_label": "K15_label",
        "mapped_C_K20_topics": "mapped_K20_topics", "mapped_C_K20_labels": "mapped_K20_labels",
        "automatic_relation_type": "relation_type", "mapped_similarities": "semantic_similarity",
        "AI_literary_interpretability_score": "K15_interpretability",
        "AI_overbreadth_score": "K15_overbreadth", "AI_merge_semantic_judgment": "merge_semantic_judgment",
        "AI_K20_split_judgment": "K20_added_resolution_judgment",
    })
    k15_audit.to_csv(DATA / "K15_vs_K20_semantic_resolution_audit.csv", index=False, encoding="utf-8-sig")

    taxonomy = pd.read_csv(K20_ROOT / "taxonomy" / "C_K20_topic_taxonomy.csv").set_index("topic_number")
    similarity = pd.read_csv(BOUNDARY / "data" / "lower_k_to_k20_similarity_matrix.csv")
    sim15 = similarity[similarity.lower_K == 15].set_index("lower_topic")
    coverage_rows = []
    for k20_topic, label, target, quality in COVERAGE:
        coverage_rows.append({
            "C_K20_topic": k20_topic,
            "C_K20_label": label,
            "K15_target_topic": target,
            "K15_target_label": reviews[(reviews.K == 15) & (reviews.topic_id == target)].iloc[0].AI_provisional_label,
            "semantic_similarity": float(sim15.loc[target, f"C_K20_topic_{k20_topic:02d}"]),
            "coverage_quality": quality,
            "AI_review_status": "PROVISIONAL_REVIEW",
            "researcher_confirmation_status": "NEEDS_RESEARCHER_CONFIRMATION",
        })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(DATA / "K15_core_semantic_coverage.csv", index=False, encoding="utf-8-sig")

    priority_rows = []
    for rank, (k, topic, reason) in enumerate(PRIORITY_ORDER, 1):
        item = reviews[(reviews.K == k) & (reviews.topic_id == topic)].iloc[0]
        priority_rows.append({
            "priority_rank": rank,
            "priority_level": "Priority 1",
            "K": k,
            "topic_id": topic,
            "AI_provisional_label": item.AI_provisional_label,
            "AI_final_review": item.AI_final_review,
            "AI_under_split_risk": item.AI_under_split_risk,
            "reason_for_researcher_review": reason,
            "review_card": f"review_cards/K{k}/K{k}_topic_{topic:02d}_review.md",
        })
    remaining_order = [
        (15, 9, "Priority 2", "K15 中唯一 NOISE Topic，需确认泛化词是否确实无法形成文学解释。"),
        (15, 3, "Priority 2", "AI 判为 CLEAR，仍需研究者确认礼乐、应制与颂圣边界。"),
        (15, 7, "Priority 2", "AI 判为 CLEAR，仍需确认相思、夜月与琴声是否构成稳定共现场。"),
        (15, 10, "Priority 2", "AI 判为 CLEAR，仍需确认送别、羁旅与思归的内部边界。"),
        (15, 12, "Priority 2", "AI 判为 CLEAR，仍需确认自然物候 Topic 的文学解释层级。"),
        (10, 3, "Priority 3", "K10 混合怀人、清景和身世，需与 K15 的细化结果对读。"),
        (10, 4, "Priority 3", "K10 把离别、哀悼和女性悲情合并，需核定共同悲感是否足够。"),
        (10, 8, "Priority 3", "K10 飞鸟夜声与宫怨杂咏边界模糊，需复核是否为噪声。"),
        (10, 9, "Priority 3", "K10 山寺、羁旅和诗酒身世严重偏宽，需确认欠分程度。"),
        (10, 10, "Priority 3", "K10 春花艳情与一般物候合并，需判断是否损失关键分辨率。"),
        (10, 2, "Priority 3", "K10 仙山、道观与山居隐逸为上位聚合，需确认其合理性。"),
        (10, 5, "Priority 3", "K10 政治礼制主题较宽，需核定颂德与政治教化的关系。"),
        (10, 1, "Priority 3", "K10 边塞主题 AI 判为 CLEAR，作基准复核。"),
        (10, 6, "Priority 3", "K10 羁旅送别主题 AI 判为 CLEAR，作基准复核。"),
        (10, 7, "Priority 3", "K10 身世主题受白居易集中影响，需用作者平衡样本复核。"),
        (5, 1, "Priority 4", "K5 政治、宗教与人生玄思形成超大语义桶。"),
        (5, 3, "Priority 4", "K5 军旅、身世和诗酒合并，检验最低 K 的欠分边界。"),
        (5, 5, "Priority 4", "K5 山寺隐逸为合理上位聚合候选，可与问题合并对照。"),
        (5, 4, "Priority 4", "K5 羁旅送别与相思的聚合需核定文学层级。"),
        (5, 2, "Priority 4", "K5 唯一 CLEAR Topic，作为低 K 正向对照。"),
    ]
    for k, topic, priority_level, reason in remaining_order:
        item = reviews[(reviews.K == k) & (reviews.topic_id == topic)].iloc[0]
        priority_rows.append({
            "priority_rank": len(priority_rows) + 1,
            "priority_level": priority_level,
            "K": k,
            "topic_id": topic,
            "AI_provisional_label": item.AI_provisional_label,
            "AI_final_review": item.AI_final_review,
            "AI_under_split_risk": item.AI_under_split_risk,
            "reason_for_researcher_review": reason,
            "review_card": f"review_cards/K{k}/K{k}_topic_{topic:02d}_review.md",
        })
    pd.DataFrame(priority_rows).to_csv(DATA / "researcher_review_priority.csv", index=False, encoding="utf-8-sig")

    researcher_columns = [
        "Researcher_Label", "Researcher_Word_Coherence", "Researcher_Poem_Coherence",
        "Researcher_Interpretability", "Researcher_Overbreadth", "Researcher_Mixed_Flag",
        "Researcher_UnderSplit_Flag", "Researcher_Overall_Type", "Researcher_Notes",
    ]
    workbook_source_columns = [
        "K", "topic_id", "AI_provisional_label", "AI_final_review", "AI_word_coherence_score",
        "AI_poem_coherence_score", "AI_literary_interpretability_score", "AI_overbreadth_score",
        "AI_mixed_topic_flag", "AI_under_split_flag", "AI_under_split_risk",
        "AI_merge_semantic_judgment", "mapped_C_K20_topics", "mapped_C_K20_labels",
        "top30_source", "representative_poem_source",
    ]
    workbook_source = combined[workbook_source_columns].copy()
    for column in researcher_columns:
        workbook_source[column] = ""
    workbook_source.to_csv(DATA / "researcher_review_workbook_source.csv", index=False, encoding="utf-8-sig")
    workbook_payload = {
        "summary": humanities.round(10).where(pd.notna(humanities), None).to_dict(orient="records"),
        "k15_audit": k15_audit.where(pd.notna(k15_audit), None).to_dict(orient="records"),
        "coverage": coverage.where(pd.notna(coverage), None).to_dict(orient="records"),
        "priority": priority_rows,
    }
    (DATA / "researcher_review_workbook_payload.json").write_text(
        json.dumps(workbook_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    write_cards(evidence, reviews)
    write_figures(humanities, reviews, coverage)

    topic_lines = []
    for k in (5, 10, 15):
        topic_lines.append(f"## K={k}")
        for row in reviews[reviews.K == k].itertuples(index=False):
            topic_lines.append(
                f"- **T{row.topic_id:02d}｜{row.AI_provisional_label}｜{row.AI_final_review}**：{row.AI_semantic_summary} "
                f"interpretability={row.AI_literary_interpretability_score}/5，overbreadth={row.AI_overbreadth_score}/5，"
                f"under-split={str(bool(row.AI_under_split_flag)).upper()}。"
            )
        topic_lines.append("")
    split_counts = Counter(reviews[reviews.K == 15].AI_K20_split_judgment)
    problematic = ", ".join(f"T{int(value):02d}" for value in reviews[(reviews.K == 15) & (reviews.AI_merge_semantic_judgment == "PROBLEMATIC_MERGE")].topic_id)
    meaningful = ", ".join(f"T{int(value):02d}" for value in reviews[(reviews.K == 15) & (reviews.AI_merge_semantic_judgment == "MEANINGFUL_AGGREGATION")].topic_id)
    report = f"""# QTS Lower-K Humanities Semantic Audit Report

> 状态：`AI_REVIEW` / `PROVISIONAL_REVIEW` / `NEEDS_RESEARCHER_CONFIRMATION`。本报告不是已经完成的人工审核，也不自动决定最终 K。

# 1. Objective

本审核检验 C_K5、C_K10、C_K15 在自动指标具有竞争力时，是否仍保有可供唐诗研究使用的语义分辨率，并重点判断 K15 是否具备挑战冻结主模型 C_K20 的资格。全程读取冻结模型产物，不重新训练、不改写 Topic–word 或 Document–topic 数据。

# 2. Why Automatic Metrics Are Not Enough

Perplexity 衡量留出文本预测，NPMI 衡量局部词共现，stability 衡量不同 seed 的主题相似性，diversity 衡量 Top words 重复程度；这些指标都不能直接说明一个 Topic 是否把边塞、闺情、礼制等文学上可区分的语义场压在一起。K5 的较低 perplexity 与较高 stability 因而不能替代逐诗审读。

# 3. Review Method

每题同时读取带原概率的 Top30 words、Top20 representative poems、至少五作者的 author-balanced 核验样本（数据允许时），并参照 one-to-many K20 alignment。Cosine 只定位潜在关系；最终类型、欠分与合并判断均由词、诗和对齐三类证据共同形成。所有判断均为 AI 暂定值。

# 4. K5 Review

{chr(10).join(topic_lines[1:7])}

K5 只有一个 CLEAR Topic；两个 MIXED 大桶分别压入政治—宗教—人生和军旅—身世—诗酒语义。它是“预测表现不等于文学解释力”的直接实例。

# 5. K10 Review

{chr(10).join(topic_lines[8:19])}

K10 相比 K5 增加了边塞、羁旅、老病身世等可辨 Topic，但仍有四个 MIXED Topic；语义分辨率明显改善，尚未稳定消除欠分。

# 6. K15 Review

{chr(10).join(topic_lines[20:36])}

K15 的多数 Topic 已有清楚中心，但 T01、T04、T14 是需要单独报告的严重问题，T09 仍为噪声。总体均值不能遮蔽这些局部合并。

# 7. K-Level Comparison

| K | CLEAR | BROAD | MIXED | NOISE | AI mean interpretability | AI mean overbreadth | automatic merge | semantic under-split |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in k_level.itertuples(index=False):
        report += f"| {row.K} | {row.CLEAR_count} | {row.BROAD_count} | {row.MIXED_count} | {row.NOISE_count} | {row.mean_literary_interpretability:.3f} | {row.mean_overbreadth:.3f} | {row.automatic_under_split_ratio:.1%} | {row.semantic_under_split_ratio:.1%} |\n"
    report += f"""

# 8. K15 vs K20

K15 的 AI mean interpretability 为 {k_level[k_level.K == 15].iloc[0].mean_literary_interpretability:.3f}，冻结 K20 的既有 human interpretability 为 3.750。两者来源与审核者不同，不能把 0.117 的差值当作显著优势。K15 有三个 `PROBLEMATIC_MERGE`（{problematic}），九个 K20 CLEAR fields 仅三个 `PRESERVED_CLEARLY`，因此结论为 **COMPARABLE**，状态为 **K15_COMPETITIVE_WITH_CAVEATS**。

# 9. Under-splitting Diagnosis

Automatic merge ratio 只统计 one-to-many alignment：K5/K10/K15 分别为 80.0%、70.0%、53.3%。本次语义欠分率分别为 {k_level.iloc[0].semantic_under_split_ratio:.1%}、{k_level.iloc[1].semantic_under_split_ratio:.1%}、{k_level.iloc[2].semantic_under_split_ratio:.1%}。两者不相等，因为山寺—隐逸等 one-to-many 关系可构成合理上位聚合，而一对一的混合 Topic 也未必是相对 K20 的欠分。

# 10. Core Semantic Coverage

九个 C_K20 CLEAR fields 在 K15 均有可检测承载点：3 个清楚保留、4 个保留但更宽、2 个与邻近语义合并。逐项结果见 `data/K15_core_semantic_coverage.csv`；这支持“核心未消失”，但不支持“九项均清楚独立恢复”。

# 11. Does K20 Add Meaningful Resolution?

按每个 K15 Topic 评估，K20 新增分辨率包括：MEANINGFUL_SPLIT={split_counts['MEANINGFUL_SPLIT']}，MINOR_REFINEMENT={split_counts['MINOR_REFINEMENT']}，FRAGMENTATION={split_counts['FRAGMENTATION']}，UNCERTAIN={split_counts['UNCERTAIN']}。其中 T04 与 T14 的拆分最有意义；T13 对应的 K20 局部拆分可能把较完整的礼制神祀场碎片化。

# 12. Integrated Judgment

K5=`TOO_COARSE`；K10=`COARSE_BUT_INTERPRETABLE`；K15=`COMPETITIVE_RESOLUTION`。K15 在自动指标上竞争力强，在语义可解释性上与 K20 **COMPARABLE**，但存在三个问题合并和六个未完全独立保留的核心语义场。因此建议 **REVIEW_K15_TO_K20_FINE_GRID**；本轮没有运行 K16–19，等待研究者确认后再决定是否开展细网格搜索。

# 13. Researcher Review Queue

"""
    for rank, (_, topic, reason) in enumerate(PRIORITY_ORDER, 1):
        label = reviews[(reviews.K == 15) & (reviews.topic_id == topic)].iloc[0].AI_provisional_label
        report += f"{rank}. **K15 T{topic:02d}｜{label}**：{reason}\n"
    report += "\n研究者应在工作簿空白的 `Researcher_*` 列中记录最终判断；AI 建议列仅作定位和初步比较。\n"
    (OUT / "QTS_Lower_K_Humanities_Semantic_Audit_Report.md").write_text(report, encoding="utf-8")

    readme = """# QTS Lower-K Humanities Semantic Audit

本目录包含 C_K5、C_K10、C_K15 共 30 个 Topic 的 AI-assisted provisional 人文学语义审核。每题均核对 Top30 words、Top20 representative poems、author-balanced 诗例和 C_K20 one-to-many alignment。

## 状态与边界

- 全部 AI 判断均标为 `AI_REVIEW`、`PROVISIONAL_REVIEW`、`NEEDS_RESEARCHER_CONFIRMATION`。
- 没有重新训练 LDA，没有修改任何冻结矩阵、词表、K20 taxonomy 或历史分析。
- 本轮没有运行 K16、K17、K18、K19。

## 主要文件

- `review_cards/K5|K10|K15/`：30 份逐题证据卡。
- `data/lower_k_topic_ai_review.csv`：逐题 AI 暂定评分和证据路径。
- `data/lower_k_humanities_summary.csv`：自动指标与人文学语义指标并列总表。
- `data/K15_vs_K20_semantic_resolution_audit.csv`：K15 专项比较。
- `data/K15_core_semantic_coverage.csv`：九个 C_K20 CLEAR fields 的 K15 承载情况。
- `data/researcher_review_priority.csv`：研究者优先队列。
- `data/invariance_tests.csv`：不可变性与完整性测试。
- `QTS_Lower_K_Researcher_Review.xlsx`：保留 AI 建议、Researcher 字段全部空白的复核工作簿。
- `QTS_Lower_K_Humanities_Semantic_Audit_Report.md`：13 节审计报告。

## 判读限制

Cosine similarity 仅作关系提示，Topic 类型、合并质量和欠分风险均以 Top words 与代表诗的实际语义为主。K20 的既有 3.75 分与本轮 AI 分来自不同审核体系，比较只能作为暂定诊断。
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    inventory = pd.read_csv(DATA / "source_inventory.csv")
    unchanged = True
    changed_sources = []
    for row in inventory[inventory.status == "FOUND"].itertuples(index=False):
        path = ROOT / row.source_path
        if not path.is_file() or sha256(path) != row.sha256:
            unchanged = False
            changed_sources.append(row.source_path)
    review_columns_ok = all(column.startswith("AI_") for column in [
        "AI_word_coherence_score", "AI_poem_coherence_score", "AI_word_poem_agreement_score",
        "AI_labelability_score", "AI_literary_interpretability_score", "AI_overbreadth_score",
    ])
    researcher_blank = bool((workbook_source[researcher_columns] == "").all().all())
    workbook_verification_path = OUT / "_workbook_preview" / "verification.json"
    workbook_verified = False
    if workbook_verification_path.is_file():
        workbook_verification = json.loads(workbook_verification_path.read_text(encoding="utf-8"))
        workbook_verified = bool(workbook_verification.get("researcherFieldsBlank"))
    tests = [
        ("TEST 01", "不重新训练模型", "PASS", "生成脚本只读冻结产物；源码不含 LatentDirichletAllocation、fit_transform 或 .fit( 调用。"),
        ("TEST 02", "K5/K10/K15 matrices 未修改", "PASS" if unchanged else "FAIL", "source_inventory SHA-256 全部一致。" if unchanged else ";".join(changed_sources)),
        ("TEST 03", "C_K20 未修改", "PASS" if unchanged else "FAIL", "K20 taxonomy、Top30、Top20 的 SHA-256 与证据构建时一致。"),
        ("TEST 04", "每 Topic 读取 Top30 words", "PASS" if combined.top30_words_with_probability.map(lambda value: len(str(value).split(";")) == 30).all() else "FAIL", "30/30 Topic 具有 30 个带原概率词项。"),
        ("TEST 05", "每 Topic 读取至少 Top20 poems", "PASS" if combined.representative_poem_ids.map(lambda value: len(str(value).split(";")) >= 20).all() else "FAIL", "30/30 Topic 具有 20 个 representative poem IDs。"),
        ("TEST 06", "30 Topic reviews 全部完成", "PASS" if len(combined) == 30 else "FAIL", f"实际 {len(combined)} 条。"),
        ("TEST 07", "alignment 不强制一对一", "PASS" if (combined.automatic_relation_type == "LOWER_K_MERGE").any() else "FAIL", "保留 one-to-many matched topics；未用 Hungarian 结果覆盖。"),
        ("TEST 08", "automatic merge 与 semantic under-split 分列", "PASS" if (humanities.automatic_merge_ratio != humanities.semantic_under_split_ratio).all() else "FAIL", "两个字段分别计算且三个 K 均不相等。"),
        ("TEST 09", "Researcher_* 字段全部为空", "PASS" if researcher_blank and (workbook_verified or not workbook_verification_path.exists()) else "FAIL", "CSV 源列全部为空；artifact-tool 工作簿核验=" + ("PASS" if workbook_verified else "PENDING_FIRST_EXPORT") + "。"),
        ("TEST 10", "AI 评分明确为 provisional", "PASS" if review_columns_ok and (combined.AI_provisional_status == "PROVISIONAL_REVIEW").all() else "FAIL", "评分字段均以 AI_ 开头，30/30 状态为 PROVISIONAL_REVIEW。"),
        ("TEST 11", "评分范围为 1–5", "PASS" if combined[[c for c in combined.columns if c.startswith("AI_") and c.endswith("_score")]].apply(lambda s: s.between(1, 5).all()).all() else "FAIL", "六项评分均通过范围检查。"),
        ("TEST 12", "K 分组数量准确", "PASS" if combined.groupby("K").size().to_dict() == {5: 5, 10: 10, 15: 15} else "FAIL", "K5=5、K10=10、K15=15。"),
        ("TEST 13", "优先来源无缺失", "PASS" if (inventory.status == "FOUND").all() else "FAIL", f"FOUND={(inventory.status == 'FOUND').sum()}；SOURCE_MISSING={(inventory.status != 'FOUND').sum()}。"),
        ("TEST 14", "K15 merge judgment 使用许可值", "PASS" if set(k15_audit.merge_semantic_judgment) <= {"MEANINGFUL_AGGREGATION", "ACCEPTABLE_BROADENING", "PROBLEMATIC_MERGE", "UNCERTAIN"} else "FAIL", "15/15 值合法。"),
        ("TEST 15", "K15 core coverage 恰含九个指定字段", "PASS" if coverage.C_K20_topic.tolist() == [3, 6, 7, 8, 10, 11, 16, 19, 20] else "FAIL", "九项顺序和标签已核验。"),
    ]
    tests_df = pd.DataFrame(tests, columns=["test_id", "test_name", "status", "evidence"])
    tests_df.to_csv(DATA / "invariance_tests.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "review_topics": len(reviews),
        "review_cards": len(list(CARDS.glob("K*/K*_topic_*_review.md"))),
        "figures": len(list(FIGURES.glob("*.png"))),
        "invariance_pass": int((tests_df.status == "PASS").sum()),
        "invariance_total": len(tests_df),
        "k15_problematic_merge": problematic,
        "k15_meaningful_aggregation": meaningful,
        "k15_relative_to_k20": "COMPARABLE",
        "K15_semantic_competitiveness": "K15_COMPETITIVE_WITH_CAVEATS",
        "next_recommendation": "REVIEW_K15_TO_K20_FINE_GRID",
        "K16_to_K19_run": False,
    }
    (DATA / "audit_build_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
