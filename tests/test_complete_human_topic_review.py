"""完整人工 Topic 审核包的概率、相似度和抽样测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import build_complete_human_topic_review as complete_review


class CompleteHumanTopicReviewTest(unittest.TestCase):
    """验证完整矩阵派生结果的核心定义。"""

    def test_full_topic_similarity_uses_cosine(self) -> None:
        """完整 Topic×Word 相似度应按行计算 cosine。"""

        matrix = np.asarray([[1.0, 0.0], [1.0, 1.0]])
        similarity = complete_review.cosine_similarity_rows(matrix)
        self.assertAlmostEqual(similarity[0, 1], 2 ** -0.5)
        self.assertEqual(similarity[0, 0], 1.0)

    def test_random_sample_excludes_representatives_and_is_reproducible(self) -> None:
        """随机高概率样本不得与代表诗重复，且固定输入应完全复现。"""

        scores = np.linspace(1.0, 0.1, 20)
        documents = [{"corpus_row": str(index)} for index in range(len(scores))]
        corpus = [
            {
                "poem_id": str(index),
                "author": "作者",
                "title": "题目",
                "original_text": "正文",
                "cleaned_tokens": "正文",
            }
            for index in range(len(scores))
        ]
        kwargs = {
            "model_id": "X_K20",
            "topic_id": 1,
            "scores": scores,
            "document_rows": documents,
            "corpus_rows": corpus,
            "representative_matrix_rows": {0},
            "topic_count": 20,
            "pool_fraction": 0.10,
            "requested_count": 20,
            "base_seed": 20260830,
        }
        first = complete_review.sample_random_high_probability(**kwargs)
        second = complete_review.sample_random_high_probability(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual([row["matrix_row"] for row in first], [1])

    def test_complete_prevalence_counts_dominant_and_thresholds(self) -> None:
        """完整 prevalence 应同时给出 soft、dominant 和阈值计数。"""

        matrix = np.asarray([[0.8, 0.2], [0.4, 0.6]])
        rows = complete_review.compute_prevalence("X_K2", matrix)
        self.assertEqual(rows[1]["dominant_poem_count"], 1)
        self.assertEqual(rows[2]["dominant_poem_count"], 1)
        self.assertEqual(rows[1]["probability_ge_0_5_count"], 1)


if __name__ == "__main__":
    unittest.main()
