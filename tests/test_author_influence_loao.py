"""作者影响与 LOAO 核心规则测试。"""

from __future__ import annotations

import sys
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_author_influence_loao as influence


class AuthorInfluenceLoaoTest(unittest.TestCase):
    """验证删除单位、两种估计量、贡献排名与重采样。"""

    def setUp(self) -> None:
        """建立两个时期、四位作者的最小作者表。"""

        self.authors = pd.DataFrame(
            [
                {"author_id_internal": "A1", "author": "甲", "author_period_primary": "初唐", "poem_count": 2, "topic_03_prob_mass": 0.6, "topic_03_prob_mean": 0.3},
                {"author_id_internal": "A2", "author": "乙", "author_period_primary": "初唐", "poem_count": 1, "topic_03_prob_mass": 0.1, "topic_03_prob_mean": 0.1},
                {"author_id_internal": "B1", "author": "丙", "author_period_primary": "晚唐", "poem_count": 3, "topic_03_prob_mass": 0.3, "topic_03_prob_mean": 0.1},
                {"author_id_internal": "B2", "author": "丁", "author_period_primary": "晚唐", "poem_count": 1, "topic_03_prob_mass": 0.4, "topic_03_prob_mean": 0.4},
            ]
        )

    def test_direction_and_sign_flip(self) -> None:
        """方向反转只在正负号互换时成立。"""

        self.assertEqual(influence.direction(0.2), "POSITIVE")
        self.assertTrue(influence.sign_flip(0.2, -0.1))
        self.assertFalse(influence.sign_flip(0.2, 0.0))

    def test_attenuation_ratio(self) -> None:
        """效应缩小一半时 attenuation 必须为0.5。"""

        self.assertAlmostEqual(influence.attenuation_ratio(0.2, 0.1, 1e-6), 0.5)

    def test_gini_equal_contributions_is_zero(self) -> None:
        """完全均等贡献的 Gini 必须为零。"""

        self.assertAlmostEqual(influence.gini_coefficient([1, 1, 1]), 0.0)

    def test_poem_estimand_is_poem_weighted(self) -> None:
        """POEM estimand 必须用 Topic mass 除以诗数。"""

        observed = influence.delta_from_author_rows(self.authors, "topic_03", "初唐", "晚唐", "POEM")
        expected = (0.7 / 4) - (0.7 / 3)
        self.assertAlmostEqual(observed, expected)

    def test_author_estimand_is_author_balanced(self) -> None:
        """AUTHOR estimand 必须让每位作者等权。"""

        observed = influence.delta_from_author_rows(self.authors, "topic_03", "初唐", "晚唐", "AUTHOR")
        expected = np.mean([0.1, 0.4]) - np.mean([0.3, 0.1])
        self.assertAlmostEqual(observed, expected)

    def test_deleting_author_removes_whole_cluster(self) -> None:
        """删除 A1 后不得保留其任何诗作权重。"""

        observed = influence.delta_from_author_rows(self.authors, "topic_03", "初唐", "晚唐", "POEM", {"A1"})
        expected = (0.7 / 4) - 0.1
        self.assertAlmostEqual(observed, expected)

    def test_global_top_uses_topic_mass(self) -> None:
        """GLOBAL_TOP 必须按 Topic mass 而非诗数排序。"""

        ids = influence.contributor_exclusion_ids(self.authors, "topic_03", "GLOBAL_TOP", 1, "初唐", "晚唐")
        self.assertEqual(ids, ["A1"])

    def test_per_period_top_removes_each_side(self) -> None:
        """PER_PERIOD_TOP 1 必须从两时期各删除一位作者。"""

        ids = influence.contributor_exclusion_ids(self.authors, "topic_03", "PER_PERIOD_TOP", 1, "初唐", "晚唐")
        self.assertEqual(set(ids), {"A1", "B2"})

    def test_bootstrap_is_reproducible(self) -> None:
        """相同 seed 的 cluster bootstrap 必须逐项一致。"""

        first = influence.bootstrap_excluded_delta(self.authors, "topic_03", "初唐", "晚唐", "POEM", set(), 100, 42)
        second = influence.bootstrap_excluded_delta(self.authors, "topic_03", "初唐", "晚唐", "POEM", set(), 100, 42)
        np.testing.assert_array_equal(first, second)

    def test_bootstrap_supported_direction(self) -> None:
        """区间跨零时不得声明支持方向。"""

        self.assertEqual(influence.bootstrap_supported_direction(0.1, 0.3), "SUPPORTED_POSITIVE")
        self.assertEqual(influence.bootstrap_supported_direction(-0.2, 0.1), "UNCERTAIN")

    def test_classification_keeps_exclusion_schemes_separate(self) -> None:
        """主方案稳健但分期方案反向时，必须分别给出状态。"""

        config = json.loads((ROOT / "config/author_influence_config.yaml").read_text(encoding="utf-8"))
        prior = pd.Series({"topic_id": "topic_03", "contrast": "E-L", "raw_effect_poem": 0.1, "raw_effect_author": 0.1})
        summary = pd.Series({"max_abs_change": 0.01, "max_attenuation": 0.1, "sign_flip_concern_count": 0})
        rows = []
        for scheme in ["GLOBAL_TOP", "PER_PERIOD_TOP"]:
            for aggregation in ["POEM", "AUTHOR"]:
                for n in [1, 3, 5]:
                    flipped = scheme == "PER_PERIOD_TOP" and aggregation == "POEM" and n == 5
                    rows.append({"sample": "CORE", "topic_id": "topic_03", "contrast": "E-L", "aggregation": aggregation,
                                 "exclusion_scheme": scheme, "N": n, "original_delta": 0.1,
                                 "new_delta": -0.02 if flipped else 0.09, "attenuation_ratio": 0.8 if flipped else 0.1,
                                 "sign_flip": flipped, "sign_flip_concern": flipped})
        exclusions = pd.DataFrame(rows)
        concentration = {"top1_share": 0.2, "top5_share": 0.5}
        primary, _ = influence.classify_author_robustness(prior, summary, summary, exclusions, concentration, config)
        secondary, _ = influence.classify_author_robustness(prior, summary, summary, exclusions, concentration, config, scheme="PER_PERIOD_TOP")
        self.assertEqual(primary, "AUTHOR_ROBUST")
        self.assertEqual(secondary, "AUTHOR_SENSITIVE")


if __name__ == "__main__":
    unittest.main()
