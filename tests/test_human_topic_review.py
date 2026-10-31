"""验证人工 Topic 审核包的选 seed、缺失标记和评分汇总。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "src" / "build_human_topic_review.py"


def load_review_module():
    """从文件路径加载审核包模块。"""

    spec = importlib.util.spec_from_file_location(
        "build_human_topic_review", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载人工 Topic 审核包程序")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HumanTopicReviewTest(unittest.TestCase):
    """检查审核包不会挑最好 seed 或自动填人工判断。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.review = load_review_module()

    def test_seed_is_closest_to_mean_not_highest(self) -> None:
        """主审核 seed 必须最接近均值，而不是 NPMI 最高。"""

        candidate = {
            "model_id": "X_K20",
            "corpus": "X",
            "min_tf": 1,
            "min_df": 1,
            "K": 20,
        }
        metrics = [
            {"corpus": "X", "K": "20", "seed": "42", "mean_topic_npmi": "0.1", "topic_diversity": "0.4"},
            {"corpus": "X", "K": "20", "seed": "2026", "mean_topic_npmi": "0.2", "topic_diversity": "0.5"},
            {"corpus": "X", "K": "20", "seed": "20260830", "mean_topic_npmi": "0.3", "topic_diversity": "0.6"},
        ]
        stability = [
            {"corpus": "X", "K": "20", "seed_a": "42", "seed_b": "2026", "matched_topic_jaccard": "0.2"},
            {"corpus": "X", "K": "20", "seed_a": "42", "seed_b": "20260830", "matched_topic_jaccard": "0.3"},
            {"corpus": "X", "K": "20", "seed_a": "2026", "seed_b": "20260830", "matched_topic_jaccard": "0.4"},
        ]
        selection, _seed_rows, _stability_rows = self.review.select_seed(
            candidate, metrics, stability
        )
        self.assertEqual(selection["selected_seed"], 2026)
        self.assertEqual(selection["selected_seed_npmi"], "0.2000000000")

    def test_unfilled_scores_remain_uncomputed(self) -> None:
        """未填写评分时模型汇总必须保持空值。"""

        candidates = [{"model_id": "X_K1", "corpus": "X", "K": 1}]
        rows = [
            {
                "model_id": "X_K1",
                "topic_id": "1",
                **{field: "" for field in self.review.SCORE_FIELDS},
                **{field: "" for field in self.review.FLAG_FIELDS},
            }
        ]
        summary = self.review.summarize_human_scores(rows, candidates)[0]
        self.assertEqual(summary["mean_total_score"], "")
        self.assertEqual(summary["number_of_uninterpretable_topics"], "")

    def test_completed_scores_are_summarized(self) -> None:
        """人工评分完成后应按既定定义计算模型汇总。"""

        candidates = [{"model_id": "X_K2", "corpus": "X", "K": 2}]
        rows = [
            {
                "model_id": "X_K2",
                "topic_id": str(topic_id),
                "semantic_coherence_score": str(score),
                "poem_coherence_score": str(score),
                "interpretability_score": str(score),
                "distinctiveness_score": str(score),
                "duplicate_topic_flag": "True" if topic_id == 1 else "False",
                "mixed_topic_flag": "False",
                "noise_topic_flag": "False",
            }
            for topic_id, score in ((1, 2), (2, 4))
        ]
        summary = self.review.summarize_human_scores(rows, candidates)[0]
        self.assertEqual(summary["mean_total_score"], "3.000000")
        self.assertEqual(summary["number_of_uninterpretable_topics"], 1)
        self.assertEqual(summary["number_of_duplicate_topics"], 1)

    def test_review_card_marks_unavailable_sections(self) -> None:
        """审核卡不得把未保存的数据伪装成完整输出。"""

        selection = {"model_id": "X_K1", "selected_seed": 42}
        words = [{"rank": "1", "token": "山", "probability": "0.1"}]
        representatives = [
            {
                "rank": 1,
                "author": "甲",
                "title": "诗",
                "poem_id": "1",
                "topic_probability": "0.8",
                "original_text": "山。",
                "cleaned_tokens": "山",
            }
        ]
        card = self.review.review_card(
            selection, 1, words, representatives, "0.1", 30, 20
        )
        self.assertIn("仅有 1/30", card)
        self.assertIn("完整 document-topic 矩阵未保存", card)
        self.assertIn("不用于冒充合规相似度", card)


if __name__ == "__main__":
    unittest.main()
