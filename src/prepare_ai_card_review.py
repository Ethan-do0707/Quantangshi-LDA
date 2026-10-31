"""为 K12/K15/K20 匿名主题卡生成清单，并按盲审顺序输出审读包。"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "model_selection" / "K12_K15_K20_human_adjudication"
CARDS_DIR = BASE / "human_review_cards"
INVENTORY_PATH = BASE / "AI_review" / "AI_review_card_inventory.csv"


def split_sections(text: str) -> dict[str, str]:
    """按二级标题切分卡片内容。"""
    matches = list(re.finditer(r"^## (.+?)\s*$", text, flags=re.MULTILINE))
    result: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result[match.group(1).strip()] = text[start:end].strip()
    return result


def parse_words(section: str) -> list[str]:
    """提取关键词表中的词语。"""
    words: list[str] = []
    for line in section.splitlines():
        match = re.match(r"^\|\s*\d+\s*\|\s*(.*?)\s*\|", line)
        if match:
            words.append(match.group(1).strip())
    return words


def parse_poems(section: str) -> list[dict[str, str]]:
    """提取诗歌标题行与正文。"""
    heading = re.compile(
        r"^###\s+\d+\.\s+poem_id=(.*?)｜(.*?)｜(.*?)｜P=([^\s]+)\s*$",
        flags=re.MULTILINE,
    )
    matches = list(heading.finditer(section))
    poems: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section)
        body = re.sub(r"\s+", "", section[start:end].strip())
        poems.append(
            {
                "poem_id": match.group(1).strip(),
                "author": match.group(2).strip(),
                "title": match.group(3).strip(),
                "probability": match.group(4).strip(),
                "text": body,
            }
        )
    return poems


def card_data(card_id: str) -> dict[str, object]:
    """读取单张匿名卡，但不接触任何私有映射。"""
    path = CARDS_DIR / f"{card_id}.md"
    if not path.exists():
        return {
            "blind_topic_id": card_id,
            "path": path,
            "words": [],
            "representative": [],
            "balanced": [],
        }
    text = path.read_text(encoding="utf-8")
    sections = split_sections(text)
    return {
        "blind_topic_id": card_id,
        "path": path,
        "words": parse_words(sections.get("Top30 topic words", "")),
        "representative": parse_poems(sections.get("Top20 representative poems", "")),
        "balanced": parse_poems(sections.get("Top10 author-balanced representative poems", "")),
    }


def write_inventory() -> None:
    """写出 47 张卡的结构完整性清单。"""
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for number in range(1, 48):
        card_id = f"HR{number:03d}"
        data = card_data(card_id)
        exists = Path(data["path"]).exists()
        words_ok = len(data["words"]) >= 30
        representative_ok = len(data["representative"]) >= 20
        balanced_ok = len(data["balanced"]) >= 10
        rows.append(
            {
                "blind_topic_id": card_id,
                "card_path": Path(data["path"]).relative_to(ROOT).as_posix(),
                "card_exists": "YES" if exists else "NO",
                "top_words_present": "YES" if words_ok else "NO",
                "representative_poems_present": "YES" if representative_ok else "NO",
                "author_balanced_poems_present": "YES" if balanced_ok else "NO",
                "review_status": "READY" if all((exists, words_ok, representative_ok, balanced_ok)) else "INSUFFICIENT_EVIDENCE",
            }
        )
    with INVENTORY_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_packet(start: int, end: int) -> None:
    """打印连续卡片的关键词、Top20 正文及作者平衡样本。"""
    for number in range(start, end + 1):
        card_id = f"HR{number:03d}"
        data = card_data(card_id)
        print(f"\n===== {card_id} =====")
        print("TOP30:", "、".join(data["words"]))
        representative = data["representative"]
        print("-- TOP20 REPRESENTATIVE POEMS --")
        for index, poem in enumerate(representative, start=1):
            print(
                f"R{index:02d} poem_id={poem['poem_id']}｜{poem['author']}｜{poem['title']}｜P={poem['probability']}\n"
                f"{poem['text']}"
            )
        representative_ids = {str(poem["poem_id"]) for poem in representative}
        print("-- AUTHOR-BALANCED TOP10 --")
        for index, poem in enumerate(data["balanced"], start=1):
            suffix = "（正文已见 Top20）" if str(poem["poem_id"]) in representative_ids else f"\n{poem['text']}"
            print(
                f"B{index:02d} poem_id={poem['poem_id']}｜{poem['author']}｜{poem['title']}｜P={poem['probability']}"
                f"{suffix}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=1)
    args = parser.parse_args()
    if args.inventory:
        write_inventory()
    print_packet(args.start, args.end)


if __name__ == "__main__":
    main()
