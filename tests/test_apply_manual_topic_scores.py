"""验证人工评分导入只改目标行并保护既有人工内容。"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "apply_manual_topic_scores.py"
SPEC = importlib.util.spec_from_file_location("apply_manual_topic_scores", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("无法加载人工评分导入程序")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ApplyManualTopicScoresTest(unittest.TestCase):
    """覆盖评分写回和覆盖保护。"""

    @staticmethod
    def _scores() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for model_id, topic_count in MODULE.EXPECTED_TOPIC_COUNTS.items():
            for topic_id in range(1, topic_count + 1):
                rows.append(
                    {
                        "model_id": model_id,
                        "topic_id": topic_id,
                        "proposed_label": f"{model_id}-主题-{topic_id}",
                        "semantic": 4,
                        "poem": 3,
                        "interpretability": 4,
                        "distinctiveness": 3,
                        "duplicate": False,
                        "mixed": topic_id == 1,
                        "noise": False,
                        "notes": "人工评分依据",
                    }
                )
        return rows

    @staticmethod
    def _write_evaluation(path: Path) -> None:
        fieldnames = [
            "model_id",
            "corpus",
            "K",
            "selected_seed",
            "topic_id",
            "proposed_label",
            "semantic_coherence_score",
            "poem_coherence_score",
            "interpretability_score",
            "distinctiveness_score",
            "mean_human_score",
            "duplicate_topic_flag",
            "mixed_topic_flag",
            "noise_topic_flag",
            "notes",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for model_id, topic_count in {**MODULE.EXPECTED_TOPIC_COUNTS, "C_K20": 1}.items():
                for topic_id in range(1, topic_count + 1):
                    writer.writerow(
                        {
                            "model_id": model_id,
                            "corpus": model_id[0],
                            "K": topic_count,
                            "selected_seed": 20260830,
                            "topic_id": topic_id,
                        }
                    )

    def test_apply_scores_preserves_non_target_rows(self) -> None:
        """评分只能写入 JSON 明确列出的 A/B Topic。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation_path = root / "evaluation.csv"
            score_path = root / "scores.json"
            self._write_evaluation(evaluation_path)
            score_path.write_text(json.dumps(self._scores(), ensure_ascii=False), encoding="utf-8")

            updated = MODULE.apply_scores(evaluation_path, score_path, allow_overwrite=False)
            self.assertEqual(updated, 70)
            with evaluation_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            first = rows[0]
            untouched = rows[-1]
            self.assertEqual(first["mean_human_score"], "3.50")
            self.assertEqual(first["mixed_topic_flag"], "True")
            self.assertEqual(untouched["model_id"], "C_K20")
            self.assertEqual(untouched["proposed_label"], "")

    def test_existing_manual_content_requires_explicit_overwrite(self) -> None:
        """默认不得静默覆盖已经存在的人工判断。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation_path = root / "evaluation.csv"
            score_path = root / "scores.json"
            self._write_evaluation(evaluation_path)
            score_path.write_text(json.dumps(self._scores(), ensure_ascii=False), encoding="utf-8")
            MODULE.apply_scores(evaluation_path, score_path, allow_overwrite=False)
            with self.assertRaisesRegex(ValueError, "已有人工内容"):
                MODULE.apply_scores(evaluation_path, score_path, allow_overwrite=False)


if __name__ == "__main__":
    unittest.main()
