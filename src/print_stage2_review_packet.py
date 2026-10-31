"""把二阶段候选与冻结 AI 标签并列，供人工语义裁决。"""

from __future__ import annotations

import csv
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "model_selection" / "K12_K15_K20_human_adjudication"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    frozen = {row["blind_topic_id"]: row for row in read_csv(BASE / "AI_review" / "AI_topic_review_frozen.csv")}

    def format_ids(value: str) -> str:
        return " | ".join(
            f"{card_id}:{frozen[card_id]['AI_Provisional_Label']}[{frozen[card_id]['AI_Provisional_Topic_Type']}]"
            for card_id in value.split("|")
        )

    stage2 = BASE / "_private" / "stage2_candidates"
    for filename in ("K12_to_K15_split_merge_candidates.csv", "K15_to_K20_split_merge_candidates.csv"):
        print(f"\n### {filename}")
        for row in read_csv(stage2 / filename):
            print(f"{row['candidate_event_id']} {row['relation_candidate']}")
            print(f"  LOWER: {format_ids(row['lower_blind_topic_ids'])}")
            print(f"  HIGHER: {format_ids(row['higher_blind_topic_ids'])}")
            if not args.compact:
                print(f"  source poems: {row['source_top10_poems']}")
                print(f"  target poems: {row['target_top10_poems']}")

    print("\n### K20 fragmentation")
    for row in read_csv(stage2 / "K20_fragmentation_candidates.csv"):
        left = row["left_blind_topic_id"]
        right = row["right_blind_topic_id"]
        print(
            f"{row['candidate_pair_id']} "
            f"{left}:{frozen[left]['AI_Provisional_Label']}[{frozen[left]['AI_Provisional_Topic_Type']}] <> "
            f"{right}:{frozen[right]['AI_Provisional_Label']}[{frozen[right]['AI_Provisional_Topic_Type']}] "
            f"poem_jaccard={row['top20_representative_poem_jaccard']}"
        )
        if not args.compact:
            print(f"  left poems: {row['left_top10_poems']}")
            print(f"  right poems: {row['right_top10_poems']}")


if __name__ == "__main__":
    main()
