"""解盲后生成模型汇总、挑战复核、分合并裁决和研究者优先队列。"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "model_selection" / "K12_K15_K20_human_adjudication"
AI_DIR = BASE / "AI_review"
STAGE2_DIR = BASE / "_private" / "stage2_candidates"
FROZEN_PATH = AI_DIR / "AI_topic_review_frozen.csv"
FROZEN_HASH_PATH = AI_DIR / "audit" / "AI_topic_review_frozen.sha256"
MAPPING_PATH = BASE / "_private" / "blind_topic_mapping_secret.csv"
UNBLINDED_PATH = AI_DIR / "AI_topic_review_unblinded.csv"
SUMMARY_PATH = AI_DIR / "AI_model_summary.csv"
SPLIT_MERGE_PATH = AI_DIR / "AI_split_merge_review.csv"
FRAGMENTATION_PATH = AI_DIR / "AI_fragmentation_review.csv"
PRIORITY_PATH = AI_DIR / "AI_generated_researcher_priority_queue.csv"
REPORT_PATH = AI_DIR / "QTS_K12_K15_K20_AI_Provisional_Card_Review.md"

SCORE_FIELDS = [
    "AI_Provisional_Word_Coherence",
    "AI_Provisional_Poem_Coherence",
    "AI_Provisional_Word_Poem_Agreement",
    "AI_Provisional_Labelability",
    "AI_Provisional_Literary_Interpretability",
    "AI_Provisional_Semantic_Specificity",
    "AI_Provisional_Overbreadth",
    "AI_Provisional_Internal_Mixedness",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CHALLENGE = {
    "HR002": (
        "RETAIN_WITH_CAUTION",
        "宫怨、失宠和花月闺情为稳定核心，但一般春景与纯咏花诗使边界偏宽；建议研究者核看宫怨是否应与花月词分层。",
        "宫怨与一般花月艳情可能被同一女性化意象遮蔽。",
    ),
    "HR026": (
        "RETAIN_WITH_CAUTION",
        "山水送别与归隐愿望连续，但夜别、寄友、退仕、仙道与新岁思归同时存在；建议核看送别和归隐是否应独立。",
        "友朋离别与山林出处可能被归、山、君等通用词混并。",
    ),
    "HR035": (
        "SPLIT_CANDIDATE",
        "瀑布危峰和寻仙登临是强核心；宴乐、人才品鉴、清冰物赋与笙磬并非同一语义机制，应优先复核。",
        "山水奇观可能掩盖应制、品鉴与音乐等异质成分。",
    ),
    "HR037": (
        "UNSTABLE_TOPIC",
        "隋宫、蜀道、仕宦送别、寺院仙迹、渔钓与史论互不统摄，不能以知、行、归、书等高频词建立可靠解释。",
        "多个独立文学主题被高频功能词汇吸附，缺少可复现的语义中心。",
    ),
    "HR045": (
        "SPLIT_CANDIDATE",
        "民生赋役、官吏讽刺与寓言劝戒有共同批判功能；歌乐、壁语、史传及仙游则明显偏离，建议分层检查。",
        "社会批判核心与音乐、史咏、仙游残余混合。",
    ),
}


SPLIT_JUDGMENTS = {
    "K12K15_C01": ("UNCERTAIN", "HR035 的瀑布、危峰和寻仙可与 HR033 的山林出处相接；HR018 礼祀与 HR030 月夜离愁仅共享云、心等词，现有去向兼有清理与漂移。", "42612《仙岩四瀑布即事》；8032《慈恩精舍南池作》；378《明堂乐章》；1339《宛转歌》", "MEDIUM"),
    "K12K15_C02": ("MINOR_REFINEMENT", "HR039 的老病、诗酒、闲居主体在 HR027 中延续；HR016 的政治讽谏只承接少量劝世成分，增益有限。", "21354《对酒闲吟》；21009《诗酒琴人》；10851《忝官引》", "HIGH"),
    "K12K15_C03": ("MEANINGFUL_SPLIT", "HR037 本为高频词驱动的噪声主题；去向把老境、险途、江舟、边塞、政治讽谏和山居拆成可辨语义群，虽 HR046 仍保留噪声。", "25776《隋宫》、1096《蜀道难》对照 13318《古别离》、13955《古从军》、23199《送元绪上人》", "HIGH"),
    "K12K15_C04": ("MEANINGFUL_SPLIT", "HR002 的宫闺花月语域在 HR019 中形成春花美人/宫怨，在 HR030 中形成秋夜月声/离愁，季节与触情机制更清楚。", "21931《长门怨》；28298《采莲歌》；21862《月中闻磬》", "HIGH"),
    "K12K15_C05": ("MEANINGFUL_SPLIT", "HR045 的民生、寓言、政治批判和历史战争混合，被分别引向讽谏、世路困厄、边塞等较可解释主题；个别宫苑与老境去向仍需注意。", "23324《庄居野行》、41026《苦热行》对照 10851《忝官引》、1055《北上行》、6926《胡马客歌》", "HIGH"),
    "K12K15_C06": ("MEANINGFUL_SPLIT", "HR020 的王朝功德语汇涵盖朝恩、郊庙、武功与贤臣，K15 对应为宫廷恩荣、祭祀礼乐、边塞功业和讽谏伦理，文学功能得到分层。", "4083《从猎渭川献诗》；4049《舞马词》；501《蜡百神乐章》；6926《胡马客歌》", "HIGH"),
    "K12K15_C07": ("MEANINGFUL_SPLIT", "HR026 的山水送别、友朋酬寄和归隐被分解为老境酬答、江舟羁旅、世路困厄与山居访道，原有宽主题获得可解释分辨率。", "8997《别王彻》、16816《送曹璩归旧隐》对照 40380《南归舟中》、31105《游天台》", "HIGH"),
    "K12K15_C08": ("FRAGMENTATION", "HR029 已是高度集中的塞外苦战主题；只有 HR007 保留边塞主轴，其余去向分散到江舟、一般行路和月夜离愁，属于对清楚主题的过度离散。", "12214《塞下曲》、5748《从军行》对照 13318《古别离》、1339《宛转歌》", "HIGH"),
    "K15K20_C01": ("MEANINGFUL_SPLIT", "HR038 将仙山、华清宫与歌舞华饰混在一起；HR036 提炼仙山道观和寻真，HR003 承接宫廷恩幸与华贵物象，虽后者仍混合但方向有意义。", "10949《经月岩山》、21648《华清宫望幸》对照 22577《怀仙》、4587《岐王席咏美人》", "HIGH"),
    "K15K20_C02": ("MINOR_REFINEMENT", "HR017 的关塞送行主体由 HR047 明确保留；HR012 仅承接春归、新岁和一般归程，形成季节性残余，属于有限清理而非完整语义分裂。", "13955《古从军》、23170《送李琮归灵州》对照 10344《燕歌行》、2762《新年作》", "HIGH"),
    "K15K20_C03": ("MEANINGFUL_SPLIT", "HR027 的老境诗酒兼含私人衰病与社交宴饮；HR010 聚焦老病身世，HR041 聚焦诗酒赠答和离席，分辨率提升清楚。", "21009《诗酒琴人》对照 20666《岁暮咏怀》、34448《离筵诉酒》", "HIGH"),
    "K15K20_C04": ("MEANINGFUL_SPLIT", "HR032 的山居隐逸兼有佛寺与一般退隐，HR042 明确寺院禅僧，HR024 明确山林闲居和出处选择，宗教与世俗隐逸得到有效区分。", "28715《留别二上人》、31944《送张逸人》对照 26923《野寺寓居》、24464《闲居即事》", "HIGH"),
    "K15K20_C05": ("MEANINGFUL_SPLIT", "HR019 的春花美人和一般季节惜时在 HR043 与 HR012 中分别突出，前者以女性、歌舞和宫闺为轴，后者以新岁、江南和乡思为轴。", "28298《采莲歌》、19585《惜牡丹》对照 1781《踏歌词》、1772《忆江南》", "HIGH"),
    "K15K20_C06": ("FRAGMENTATION", "HR018 已是清楚的郊庙礼祀主题；HR040 延续核心，而 HR011 将部分礼祀与清光物赋、禅道、祥瑞混合，新增的是残余杂糅而非稳定子题。", "501《蜡百神乐章》、2366《南郊舒和》对照 14945《月映清淮流》、16320《禅堂》", "HIGH"),
    "K15K20_C07": ("FRAGMENTATION", "HR033 的官居清游和归隐愿望虽偏宽，去向除 HR024 外分散到清寒杂咏、祥瑞物赋和秋夜离愁，两项仍混合，未形成稳定的多子题结构。", "8032《慈恩精舍南池作》、13322《归山招王逵》对照 38020《清风戒寒》、14945《月映清淮流》", "HIGH"),
    "K15K20_C08": ("MINOR_REFINEMENT", "HR007 的边塞核心在 HR047 中明显纯化，功名与朝廷声誉残余流入 HR005；后一主题仍混合，因此属于有限清理。", "6926《幽州胡马客歌》、7263《送梁公北征》对照 8348《走马川行》、26880《和主司王起》", "HIGH"),
}


MERGE_JUDGMENTS = {
    "K12K15_C09": ("PROBLEMATIC_MERGE", "民生批判、礼典、山水与宫闺汇入仙山宫阙杂题，目标仍为 MIXED，华饰词汇不足以支撑合并。", "HIGH"),
    "K12K15_C10": ("PROBLEMATIC_MERGE", "边塞、山水送别、社会批判和老境被压入一般世路困厄，牺牲多个清晰文学机制。", "HIGH"),
    "K12K15_C11": ("PROBLEMATIC_MERGE", "边塞战争、春日宴游和山水送别只共享行、归、马等表层词，目标关塞归程过宽。", "HIGH"),
    "K12K15_C12": ("PROBLEMATIC_MERGE", "老病诗酒与民生寓言不能由政治讽谏完全统摄，合并会遮蔽私人晚年书写。", "HIGH"),
    "K12K15_C13": ("ACCEPTABLE_BROADENING", "老境退居与山水友朋赠答可在诗酒知己和归隐框架内共存，目标仍保持清楚中心。", "HIGH"),
    "K12K15_C14": ("PROBLEMATIC_MERGE", "宫闺、边塞、楚江、瀑布和禅寺汇入月夜离愁，主要依赖云月思等高频意象。", "HIGH"),
    "K12K15_C15": ("PROBLEMATIC_MERGE", "六个来源跨送别、噪声、老境、功德、山水和春游，目标本身为 NOISE。", "HIGH"),
    "K12K15_C16": ("ACCEPTABLE_BROADENING", "楚江舟行与山水送别直接支持江舟羁旅；边塞来源仅是次级思归成分，目标仍保持清晰。", "MEDIUM"),
    "K12K15_C17": ("MEANINGFUL_AGGREGATION", "春日花酒与宫闺花月共同形成春花美人、赏花惜春和宫怨的连贯主题。", "HIGH"),
    "K12K15_C18": ("PROBLEMATIC_MERGE", "江峡舟行、花酒游乐和宫闺相思并非同一春水生态主题，目标清晰主要来自局部来源。", "HIGH"),
    "K12K15_C19": ("MEANINGFUL_AGGREGATION", "宫廷礼典与帝王功德共同汇成郊庙礼乐；山水来源中的神圣登临只是边缘补充。", "HIGH"),
    "K12K15_C20": ("PROBLEMATIC_MERGE", "边塞与王朝功业可相接，但山水、民生杂题和一般礼典扩大了边界，目标仍为 BROAD。", "HIGH"),
    "K15K20_C09": ("PROBLEMATIC_MERGE", "春花宫闺与关塞军旅仅由春、归等词相接，汇入新岁江南乡思会损失两端主题。", "HIGH"),
    "K15K20_C10": ("PROBLEMATIC_MERGE", "郊庙礼祀与官居清游汇入清光祥瑞杂题，目标为 MIXED，未形成有效聚合。", "HIGH"),
    "K15K20_C11": ("PROBLEMATIC_MERGE", "山林出处与秋夜琴歌离愁汇入清寒云景杂题，依赖清、云、夜等表层词。", "HIGH"),
    "K15K20_C12": ("PROBLEMATIC_MERGE", "边塞武功与宫廷恩荣只在功名话语上相交，目标的文章仕宦主题仍为 MIXED。", "HIGH"),
    "K15K20_C13": ("MEANINGFUL_AGGREGATION", "关塞骑行与边塞征战共同汇入汉胡战争、将军出塞和军旅凯旋，文本机制一致。", "HIGH"),
    "K15K20_C14": ("PROBLEMATIC_MERGE", "仙山、山居、江舟与边塞四种来源中只有前两者稳定支持寻真，其他来源属于语义漂移。", "HIGH"),
    "K15K20_C15": ("PROBLEMATIC_MERGE", "七个来源跨夜愁、春水、隐逸、仙山、宫廷和江舟，目标为 NOISE，典型过度聚合。", "HIGH"),
    "K15K20_C16": ("ACCEPTABLE_BROADENING", "山居访道与官居归思可共同支持山林闲居；江舟来源仅扩展水边清游，目标仍具清楚中心。", "MEDIUM"),
}


FRAGMENTATION_JUDGMENTS = {
    "K20_F01": ("MEANINGFUL_SEPARATION", "HR042 以寺院、僧侣、钟磬和禅居为核心；HR024 以世俗山居、退仕和自适为核心，宗教身份与出处功能不同。", "HIGH"),
    "K20_F02": ("MEANINGFUL_SEPARATION", "HR001 是月夜声景触发的相思离愁，HR021 是楚江舟路中的客愁思归；意象重叠但叙事场景不同。", "HIGH"),
    "K20_F03": ("MEANINGFUL_SEPARATION", "HR012 强调新岁、江南、归乡和惜时，HR043 强调女性、花、歌舞和宫闺相思，季节词不构成重复主题。", "HIGH"),
    "K20_F04": ("MEANINGFUL_SEPARATION", "HR025 是乱世、忠义和历史悲愤，HR010 是个人衰老、病困与诗友酬答，主体尺度和文学功能明显不同。", "HIGH"),
    "K20_F05": ("MEANINGFUL_SEPARATION", "HR021 聚焦移动中的江舟旅宿与送别，HR024 聚焦稳定的山林居处和退隐选择。", "HIGH"),
    "K20_F06": ("MEANINGFUL_SEPARATION", "HR043 的春花服务于女性歌舞与闺情，HR004 的春景服务于禽鸟、水岸和自然生态观察。", "HIGH"),
    "K20_F07": ("MEANINGFUL_SEPARATION", "山寺禅居与楚江羁旅的空间、人物和文体功能均不同，仅共享山云月等景物词。", "HIGH"),
    "K20_F08": ("MEANINGFUL_SEPARATION", "江舟归思与仙山寻真分别属于羁旅抒情和宗教超越，云山水等意象不足以判为碎片化。", "HIGH"),
    "K20_F09": ("MEANINGFUL_SEPARATION", "HR010 以老病身世为中心，HR041 以宴饮、赠答和离席为中心；诗酒与友朋是交集但不是同一主轴。", "HIGH"),
    "K20_F10": ("MEANINGFUL_SEPARATION", "HR025 通过战争与史迹表达忠魂悲愤，HR015 以史论、寓言和名节实施规范性劝戒，抒情与论理功能可区分。", "HIGH"),
    "K20_F11": ("MEANINGFUL_SEPARATION", "寺院清修与女性春花歌舞没有共同语义机制，相似度来自云月花等常用景物。", "HIGH"),
    "K20_F12": ("MEANINGFUL_SEPARATION", "HR042 的佛寺僧居与 HR036 的道观仙山、炼丹访真具有明确宗教传统和叙事目标差异。", "HIGH"),
    "K20_F13": ("MEANINGFUL_SEPARATION", "政治伦理劝戒与个人老病自咏在主体、语气和文体功能上不同。", "HIGH"),
    "K20_F14": ("MEANINGFUL_SEPARATION", "HR040 是郊庙献享和国家礼乐，HR005 是文章声名、仕宦荣典并夹杂宗教与哀挽，核心功能不同。", "HIGH"),
    "K20_F15": ("UNCERTAIN", "HR010 的老病身世清楚，HR022 本身为高频词驱动噪声；部分年时、出处与送归重叠无法证明重复，也难据噪声主题确认有效分离。", "MEDIUM"),
}


def main() -> None:
    expected_hash = FROZEN_HASH_PATH.read_text(encoding="utf-8").split()[0]
    assert sha256(FROZEN_PATH) == expected_hash, "冻结文件哈希已改变"
    frozen = read_csv(FROZEN_PATH)
    mapping = {row["blind_topic_id"]: row for row in read_csv(MAPPING_PATH)}
    unblinded: list[dict[str, object]] = []
    for row in frozen:
        card_id = row["blind_topic_id"]
        challenge = CHALLENGE.get(card_id, ("NOT_REQUIRED", "", ""))
        joined: dict[str, object] = dict(row)
        joined["Unblinded_K"] = int(mapping[card_id]["true_K"])
        joined["Unblinded_Topic_ID"] = int(mapping[card_id]["true_topic_id"])
        joined["AI_Challenge_Status"] = challenge[0]
        joined["AI_Challenge_Comment"] = challenge[1]
        joined["AI_Missing_Semantic_Risk"] = challenge[2]
        unblinded.append(joined)
    write_csv(UNBLINDED_PATH, unblinded, list(unblinded[0]))

    summary_rows: list[dict[str, object]] = []
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in unblinded:
        grouped.setdefault(int(row["Unblinded_K"]), []).append(row)
    for k in (12, 15, 20):
        rows = grouped[k]
        types = Counter(str(row["AI_Provisional_Topic_Type"]) for row in rows)
        needs = Counter(str(row["AI_Provisional_Needs_Splitting"]) for row in rows)
        result: dict[str, object] = {"K": k, "Topic_Count": len(rows)}
        for topic_type in ("CLEAR", "BROAD", "MIXED", "NOISE"):
            result[f"{topic_type}_Count"] = types[topic_type]
            result[f"{topic_type}_Ratio"] = round(types[topic_type] / len(rows), 6)
        for field in SCORE_FIELDS:
            result[f"Mean_{field.removeprefix('AI_Provisional_')}"] = round(mean(float(row[field]) for row in rows), 4)
        result["Needs_Splitting_Count"] = needs["YES"]
        result["Needs_Splitting_Ratio"] = round(needs["YES"] / len(rows), 6)
        result["Needs_Splitting_or_Uncertain_Count"] = needs["YES"] + needs["UNCERTAIN"]
        result["Needs_Splitting_or_Uncertain_Ratio"] = round((needs["YES"] + needs["UNCERTAIN"]) / len(rows), 6)
        result["AI_Overall_Status"] = "AI_MORE_REVIEW_REQUIRED"
        summary_rows.append(result)
    write_csv(SUMMARY_PATH, summary_rows, list(summary_rows[0]))

    split_merge_rows: list[dict[str, object]] = []
    for transition, filename in (
        ("K12_to_K15", "K12_to_K15_split_merge_candidates.csv"),
        ("K15_to_K20", "K15_to_K20_split_merge_candidates.csv"),
    ):
        for row in read_csv(STAGE2_DIR / filename):
            event_id = row["candidate_event_id"]
            if row["relation_candidate"] == "SPLIT":
                judgment, rationale, evidence, confidence = SPLIT_JUDGMENTS[event_id]
            else:
                judgment, rationale, confidence = MERGE_JUDGMENTS[event_id]
                evidence = f"来源 {row['lower_blind_topic_ids']}；目标 {row['higher_blind_topic_ids']}；已复核相应卡片关键词与代表诗正文。"
            split_merge_rows.append(
                {
                    "candidate_event_id": event_id,
                    "transition": transition,
                    "relation_candidate": row["relation_candidate"],
                    "lower_topic_ids": row["lower_topic_ids"],
                    "lower_blind_topic_ids": row["lower_blind_topic_ids"],
                    "higher_topic_ids": row["higher_topic_ids"],
                    "higher_blind_topic_ids": row["higher_blind_topic_ids"],
                    "AI_Provisional_Split_Judgment": judgment,
                    "AI_Rationale": rationale,
                    "AI_Text_Evidence": evidence,
                    "AI_Confidence": confidence,
                    "Quantitative_Metrics_Used_As_Decision": "NO",
                }
            )
    write_csv(SPLIT_MERGE_PATH, split_merge_rows, list(split_merge_rows[0]))

    fragmentation_rows: list[dict[str, object]] = []
    for row in read_csv(STAGE2_DIR / "K20_fragmentation_candidates.csv"):
        judgment, rationale, confidence = FRAGMENTATION_JUDGMENTS[row["candidate_pair_id"]]
        fragmentation_rows.append(
            {
                "candidate_pair_id": row["candidate_pair_id"],
                "left_topic_id": row["left_topic_id"],
                "left_blind_topic_id": row["left_blind_topic_id"],
                "right_topic_id": row["right_topic_id"],
                "right_blind_topic_id": row["right_blind_topic_id"],
                "AI_Provisional_Fragmentation_Status": judgment,
                "AI_Rationale": rationale,
                "AI_Confidence": confidence,
                "Quantitative_Metrics_Used_As_Decision": "NO",
            }
        )
    write_csv(FRAGMENTATION_PATH, fragmentation_rows, list(fragmentation_rows[0]))

    key_split_sources = {
        row["lower_blind_topic_ids"]
        for row in split_merge_rows
        if row["relation_candidate"] == "SPLIT" and row["AI_Provisional_Split_Judgment"] == "MEANINGFUL_SPLIT"
    }
    key_fragment_topics = {
        card_id
        for row in fragmentation_rows
        if row["AI_Provisional_Fragmentation_Status"] in {"TRUE_FRAGMENTATION", "UNCERTAIN"}
        for card_id in (row["left_blind_topic_id"], row["right_blind_topic_id"])
    }
    priority_rows: list[dict[str, object]] = []
    for row in unblinded:
        card_id = str(row["blind_topic_id"])
        topic_type = str(row["AI_Provisional_Topic_Type"])
        needs = str(row["AI_Provisional_Needs_Splitting"])
        reasons: list[str] = []
        if topic_type in {"MIXED", "NOISE"}:
            reasons.append(f"Topic_Type={topic_type}")
        if needs == "YES":
            reasons.append("Needs_Splitting=YES")
        if card_id in key_split_sources:
            reasons.append("关键有意义分裂的来源主题")
        if card_id in key_fragment_topics:
            reasons.append("碎片化候选仍不确定")
        if reasons:
            priority = 1
        elif topic_type == "BROAD" or needs == "UNCERTAIN":
            priority = 2
            reasons.append("BROAD 或 Needs_Splitting=UNCERTAIN")
        else:
            priority = 3
            reasons.append("CLEAR 且无需拆分")
        priority_rows.append(
            {
                "Priority": priority,
                "blind_topic_id": card_id,
                "Unblinded_K": row["Unblinded_K"],
                "Unblinded_Topic_ID": row["Unblinded_Topic_ID"],
                "AI_Provisional_Label": row["AI_Provisional_Label"],
                "AI_Provisional_Topic_Type": topic_type,
                "AI_Provisional_Needs_Splitting": needs,
                "Priority_Reason": "；".join(reasons),
            }
        )
    priority_rows.sort(key=lambda item: (int(item["Priority"]), int(item["Unblinded_K"]), int(item["Unblinded_Topic_ID"])))
    write_csv(PRIORITY_PATH, priority_rows, list(priority_rows[0]))

    by_k = {int(row["K"]): row for row in summary_rows}
    split_counts: dict[str, Counter[str]] = {}
    for transition in ("K12_to_K15", "K15_to_K20"):
        split_counts[transition] = Counter(
            str(row["AI_Provisional_Split_Judgment"])
            for row in split_merge_rows
            if row["transition"] == transition and row["relation_candidate"] == "SPLIT"
        )
    frag_counts = Counter(str(row["AI_Provisional_Fragmentation_Status"]) for row in fragmentation_rows)
    challenge_rows = [row for row in unblinded if row["AI_Challenge_Status"] != "NOT_REQUIRED"]
    attention = {
        k: [
            f"{row['blind_topic_id']}/T{row['Unblinded_Topic_ID']} {row['AI_Provisional_Label']}"
            for row in priority_rows
            if int(row["Unblinded_K"]) == k and int(row["Priority"]) <= 2
        ]
        for k in (12, 15, 20)
    }
    lines: list[str] = []
    lines += ["## 1. Objective", "", "对 47 张匿名主题卡进行 AI 暂定语义审读，保持 Researcher_* 字段原样；结果只作为研究者复核材料。", ""]
    lines += ["## 2. Card Inventory", "", "HR001–HR047 共 47 张均存在，Top30、Top20 代表诗与作者平衡 Top10 均完整。详见 `AI_review_card_inventory.csv`。", ""]
    lines += ["## 3. Blind Review Protocol", "", f"按 HR001→HR047 审读并在解盲前冻结；冻结 SHA256：`{expected_hash}`。本轮未提前读取真实 K、映射或量化指标。受同一对话早期上下文曾暴露少量旧 AI 内容影响，不宣称对旧 AI 的完全独立隔离。", ""]
    lines += ["## 4. Review Rubric", "", "八项 1–5 分包括词语凝聚、诗例凝聚、词诗一致、可命名、文学可解释、语义特定性、过宽度与内部混合度；后两项高分代表风险较高。类型限定 CLEAR/BROAD/MIXED/NOISE。", ""]
    lines += ["## 5. 47-Topic Review Completion", "", "47/47 主题的 15 个 AI_Provisional_* 字段均已填写；Needs_Splitting=YES 均列出疑似子主题，MIXED/NOISE 均有证据与说明。", ""]
    for number, k in ((6, 12), (7, 15), (8, 20)):
        row = by_k[k]
        lines += [f"## {number}. K{k} Summary", "", f"CLEAR {row['CLEAR_Count']}，BROAD {row['BROAD_Count']}，MIXED {row['MIXED_Count']}，NOISE {row['NOISE_Count']}；文学可解释性均值 {row['Mean_Literary_Interpretability']}，过宽度均值 {row['Mean_Overbreadth']}，Needs_Splitting=YES {row['Needs_Splitting_Count']}/{row['Topic_Count']}。", ""]
    lines += ["## 9. K12 Broad / Noise Challenge", "", "复核对象：" + "；".join(f"{row['blind_topic_id']}/T{row['Unblinded_Topic_ID']} {row['AI_Challenge_Status']}" for row in challenge_rows) + "。挑战意见另存于解盲主题表。", ""]
    c = split_counts["K12_to_K15"]
    lines += ["## 10. K12→K15 Split Review", "", f"8 个分裂候选：MEANINGFUL_SPLIT {c['MEANINGFUL_SPLIT']}，MINOR_REFINEMENT {c['MINOR_REFINEMENT']}，FRAGMENTATION {c['FRAGMENTATION']}，UNCERTAIN {c['UNCERTAIN']}。逐事件依据见 `AI_split_merge_review.csv`。", ""]
    c = split_counts["K15_to_K20"]
    lines += ["## 11. K15→K20 Split Review", "", f"8 个分裂候选：MEANINGFUL_SPLIT {c['MEANINGFUL_SPLIT']}，MINOR_REFINEMENT {c['MINOR_REFINEMENT']}，FRAGMENTATION {c['FRAGMENTATION']}，UNCERTAIN {c['UNCERTAIN']}。K20 虽增加若干清晰细分，也生成混合残余。", ""]
    lines += ["## 12. Fragmentation Review", "", f"15 对候选中 TRUE_FRAGMENTATION {frag_counts['TRUE_FRAGMENTATION']}，MEANINGFUL_SEPARATION {frag_counts['MEANINGFUL_SEPARATION']}，UNCERTAIN {frag_counts['UNCERTAIN']}。判断基于词与诗的文学功能，不以 cosine/Jaccard 自动裁决。", ""]
    lines += ["## 13. AI Provisional Model Comparison", "", "K12 的清晰主题比例较高但包含宽主题与噪声；K15 在清晰度、混合风险和语义分辨率之间较平衡；K20 提供若干有效细分，但 MIXED/NOISE 比例上升。鉴于旧 AI 隔离限制与二阶段残余风险，状态记为 `AI_MORE_REVIEW_REQUIRED`，不作最终 K 决策。", ""]
    lines += ["## 14. Items Requiring Researcher Attention", "", f"K12：{'；'.join(attention[12])}。\n\nK15：{'；'.join(attention[15])}。\n\nK20：{'；'.join(attention[20])}。", ""]
    lines += ["## 15. Limitations", "", "本结果是单一 AI 的暂定审读，不替代研究者裁决；同一对话早期曾出现少量旧 AI 内容，严格独立盲审受到限制。量化指标未用于修改冻结语义分数，候选关系也不被视为真实谱系。", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    assert sha256(FROZEN_PATH) == expected_hash, "解盲后冻结文件发生改变"
    print(f"unblinded={len(unblinded)} split_merge={len(split_merge_rows)} fragmentation={len(fragmentation_rows)} priority={len(priority_rows)}")
    print("summary", [(row["K"], row["CLEAR_Count"], row["BROAD_Count"], row["MIXED_Count"], row["NOISE_Count"], row["Mean_Literary_Interpretability"], row["Mean_Overbreadth"], row["Needs_Splitting_Count"]) for row in summary_rows])
    print("split_counts", {key: dict(value) for key, value in split_counts.items()})
    print("frag_counts", dict(frag_counts))


if __name__ == "__main__":
    main()
