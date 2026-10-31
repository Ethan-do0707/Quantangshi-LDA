"""冻结已完整填写的47项研究者 Topic 级审核。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from validate_human_review import DEFAULT_WORKBOOK, RESEARCHER_FIELDS, load_topic_review, validate_review_frame


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model_selection/K12_K15_K20_human_adjudication"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结研究者 Topic 级审核")
    parser.add_argument("--workbook", default=str(DEFAULT_WORKBOOK))
    args = parser.parse_args()
    workbook_path = Path(args.workbook)
    frame = load_topic_review(workbook_path)
    result = validate_review_frame(frame)
    if result["status"] != "TOPIC_REVIEW_COMPLETE":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    frozen_path = OUT / "data/researcher_topic_review_frozen.csv"
    hash_path = OUT / "audit/researcher_topic_review_frozen.sha256"
    export = frame[["blind_topic_id", *RESEARCHER_FIELDS]].sort_values("blind_topic_id")
    if frozen_path.exists() or hash_path.exists():
        if not (frozen_path.exists() and hash_path.exists()):
            raise RuntimeError("冻结 CSV 与哈希文件状态不一致")
        stored = hash_path.read_text(encoding="utf-8").split()[0]
        if sha256(frozen_path) != stored:
            raise RuntimeError("既有 Topic 级冻结文件哈希失配")
        current_bytes = export.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
        if hashlib.sha256(current_bytes).hexdigest() != stored:
            raise RuntimeError("工作簿研究者评分与既有冻结结果不一致；禁止覆盖")
        print(json.dumps({"status": "TOPIC_REVIEW_ALREADY_FROZEN", "sha256": stored}, ensure_ascii=False))
        return
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(frozen_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    digest = sha256(frozen_path)
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(f"{digest}  data/researcher_topic_review_frozen.csv\n", encoding="utf-8")
    metadata = {
        "status": "TOPIC_REVIEW_FROZEN_READY_FOR_UNBLIND",
        "topics": 47,
        "frozen_sha256": digest,
        "source_workbook_sha256": sha256(workbook_path),
        "researcher_scores_generated_by_codex": False,
    }
    (OUT / "audit/topic_review_freeze.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
