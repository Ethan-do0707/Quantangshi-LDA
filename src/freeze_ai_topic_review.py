"""校验并冻结 47 个匿名主题的 AI 暂定审读结果。"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "model_selection" / "K12_K15_K20_human_adjudication"
AI_DIR = BASE / "AI_review"
WORKING_PATH = AI_DIR / "blind_review_working.json"
FROZEN_PATH = AI_DIR / "AI_topic_review_frozen.csv"
HASH_PATH = AI_DIR / "audit" / "AI_topic_review_frozen.sha256"
PROTOCOL_PATH = AI_DIR / "audit" / "blind_review_protocol.json"
BASELINE_PATH = AI_DIR / "audit" / "workbook_preservation_baseline.json"
INVENTORY_PATH = AI_DIR / "AI_review_card_inventory.csv"
BACKUP_PATH = BASE / "QTS_K12_K15_K20_Human_Review.before_AI_review.xlsx"

EXPECTED_ORIGINAL_HASH = "04ae77e31b25691741a219ef3de953597803f1ab56f7c573952702d68ab7927a"
AI_FIELDS = [
    "AI_Provisional_Label",
    "AI_Provisional_Word_Coherence",
    "AI_Provisional_Poem_Coherence",
    "AI_Provisional_Word_Poem_Agreement",
    "AI_Provisional_Labelability",
    "AI_Provisional_Literary_Interpretability",
    "AI_Provisional_Semantic_Specificity",
    "AI_Provisional_Overbreadth",
    "AI_Provisional_Internal_Mixedness",
    "AI_Provisional_Topic_Type",
    "AI_Provisional_Needs_Splitting",
    "AI_Provisional_Suspected_Subthemes",
    "AI_Provisional_Confidence",
    "AI_Provisional_Evidence",
    "AI_Provisional_Notes",
]
SCORE_FIELDS = AI_FIELDS[1:9]


def sha256(path: Path) -> str:
    """计算文件 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(records: list[dict[str, object]]) -> None:
    """执行冻结前的结构和业务规则校验。"""
    assert len(records) == 47, f"应有 47 条记录，实际为 {len(records)}"
    assert [record["blind_topic_id"] for record in records] == [f"HR{i:03d}" for i in range(1, 48)]
    for record in records:
        assert set(record) == {"blind_topic_id", *AI_FIELDS}, record["blind_topic_id"]
        for field in SCORE_FIELDS:
            assert isinstance(record[field], int) and 1 <= int(record[field]) <= 5, (record["blind_topic_id"], field)
        assert record["AI_Provisional_Topic_Type"] in {"CLEAR", "BROAD", "MIXED", "NOISE"}
        assert record["AI_Provisional_Needs_Splitting"] in {"YES", "NO", "UNCERTAIN"}
        assert record["AI_Provisional_Confidence"] in {"HIGH", "MEDIUM", "LOW"}
        assert str(record["AI_Provisional_Label"]).strip()
        assert str(record["AI_Provisional_Evidence"]).strip()
        assert str(record["AI_Provisional_Notes"]).strip()
        if record["AI_Provisional_Needs_Splitting"] == "YES":
            assert str(record["AI_Provisional_Suspected_Subthemes"]).strip()
        if record["AI_Provisional_Topic_Type"] in {"MIXED", "NOISE"}:
            assert str(record["AI_Provisional_Evidence"]).strip()
            assert str(record["AI_Provisional_Notes"]).strip()


def main() -> None:
    records = json.loads(WORKING_PATH.read_text(encoding="utf-8"))
    validate(records)
    AI_DIR.mkdir(parents=True, exist_ok=True)
    HASH_PATH.parent.mkdir(parents=True, exist_ok=True)

    with FROZEN_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blind_topic_id", *AI_FIELDS])
        writer.writeheader()
        writer.writerows(records)
    frozen_hash = sha256(FROZEN_PATH)
    HASH_PATH.write_text(f"{frozen_hash}  {FROZEN_PATH.name}\n", encoding="utf-8")

    inventory = list(csv.DictReader(INVENTORY_PATH.open("r", encoding="utf-8-sig", newline="")))
    for row in inventory:
        if row["review_status"] == "READY":
            row["review_status"] = "COMPLETED"
    with INVENTORY_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)

    backup_hash = sha256(BACKUP_PATH)
    assert backup_hash == EXPECTED_ORIGINAL_HASH
    baseline = {
        "original_workbook": "QTS_K12_K15_K20_Human_Review.xlsx",
        "original_sha256": EXPECTED_ORIGINAL_HASH,
        "backup_workbook": BACKUP_PATH.name,
        "backup_sha256": backup_hash,
        "hashes_match": True,
        "note": "原工作簿在冻结时由 Excel 占用；原件哈希此前以共享读取方式确认，备份为同哈希逐字节副本。",
    }
    BASELINE_PATH.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    protocol = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_order": "HR001-HR047",
        "cards_completed": 47,
        "private_mapping_read_before_freeze": False,
        "quantitative_metrics_used_before_freeze": False,
        "frozen_csv_sha256": frozen_hash,
        "strict_blindness_limitation": "本轮未读取真实 K 或私有映射；但同一对话的早期上下文曾暴露少量旧 AI 内容，因此不宣称对旧 AI 的完全独立隔离。",
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": 47, "frozen_sha256": frozen_hash, "backup_sha256": backup_hash}, ensure_ascii=False))


if __name__ == "__main__":
    main()
