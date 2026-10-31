"""作者时期 Topic 趋势结构关键规则测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_author_period_trend_structure as trend


THRESHOLDS = {
    "numeric_flat_relative_delta": 0.01,
    "numeric_absolute_epsilon": 1e-12,
    "near_flat_normalized_range": 0.10,
    "normalized_range_min_mean": 1e-9,
    "peak_tie_tolerance": 1e-12,
    "peak_trough_ratio_min_denominator": 1e-6,
    "author_concentration_moderate_top1": 0.15,
    "author_concentration_moderate_top5": 0.40,
    "author_concentration_moderate_hhi": 0.08,
    "author_concentration_high_top1": 0.25,
    "author_concentration_high_top5": 0.60,
    "author_concentration_high_hhi": 0.15,
}


class AuthorPeriodTrendStructureTest(unittest.TestCase):
    """验证方向、形状、相关与作者聚类抽样。"""

    def test_numeric_direction_uses_recorded_relative_tolerance(self) -> None:
        """小于四期参考均值1%的变化必须标记为数值近似平。"""

        self.assertEqual(trend.numeric_direction(0.0005, 0.1, THRESHOLDS), "FLAT_NUMERIC")
        self.assertEqual(trend.numeric_direction(0.002, 0.1, THRESHOLDS), "UP")

    def test_supported_direction_uses_difference_interval(self) -> None:
        """bootstrap 差值区间跨零时不得标记为受支持方向。"""

        self.assertEqual(trend.supported_direction(0.01, 0.03), "SUPPORTED_UP")
        self.assertEqual(trend.supported_direction(-0.03, -0.01), "SUPPORTED_DOWN")
        self.assertEqual(trend.supported_direction(-0.01, 0.02), "UNCERTAIN")

    def test_shape_classification_uses_four_period_mean_directions(self) -> None:
        """趋势形状由四期均值方向定义，bootstrap 支持另列保存。"""

        self.assertEqual(trend.classify_shape(0.5, ["UP", "UP", "UP"], THRESHOLDS), "MONOTONIC_UP")
        self.assertEqual(trend.classify_shape(0.5, ["UP", "DOWN", "DOWN"], THRESHOLDS), "RISE_THEN_FALL")
        self.assertEqual(trend.classify_shape(0.5, ["UP", "DOWN", "UP"], THRESHOLDS), "MULTI_TURN")

    def test_near_flat_precedes_direction_shape(self) -> None:
        """normalized range 达到配置近似平阈值时优先标记 NEAR_FLAT。"""

        self.assertEqual(trend.classify_shape(0.09, ["UP", "DOWN", "UP"], THRESHOLDS), "NEAR_FLAT")

    def test_mode_tie_is_not_forced(self) -> None:
        """六规格众数并列时必须保守返回 UNCERTAIN。"""

        value, count, tied = trend.mode_with_count(["初唐", "初唐", "盛唐", "盛唐"])
        self.assertEqual(value, "UNCERTAIN")
        self.assertEqual(count, 2)
        self.assertTrue(tied)

    def test_rank_order_keeps_all_periods(self) -> None:
        """时期排序必须保留四个时期而非只保留 peak。"""

        observed = trend.rank_order(
            ["初唐", "盛唐", "中唐", "晚唐"],
            np.array([0.2, 0.4, 0.3, 0.1]),
            {"初唐": "E", "盛唐": "H", "中唐": "M", "晚唐": "L"},
        )
        self.assertEqual(observed, "H>M>E>L")

    def test_spearman_is_descriptive_without_p_value(self) -> None:
        """四期完全同序向量的 Spearman 描述值必须为1。"""

        self.assertAlmostEqual(trend.spearman_four(np.array([1, 2, 3, 4]), np.array([10, 20, 30, 40])), 1.0)

    def test_author_cluster_bootstrap_is_reproducible(self) -> None:
        """固定 seed 时作者 cluster replicate 必须逐值复现。"""

        rows = []
        for period in ["初唐", "盛唐", "中唐", "晚唐"]:
            for author_index in range(2):
                for poem_index in range(author_index + 1):
                    rows.append(
                        {
                            "eligible": True,
                            "author_period_primary": period,
                            "author_id_internal": f"{period}_{author_index}",
                            "topic_01_prob": 0.2 + 0.1 * author_index,
                            "topic_02_prob": 0.8 - 0.1 * author_index,
                        }
                    )
        frame = pd.DataFrame(rows)
        first = trend.bootstrap_sample(frame, "eligible", ["topic_01_prob", "topic_02_prob"], ["初唐", "盛唐", "中唐", "晚唐"], 0, 20, 7)
        second = trend.bootstrap_sample(frame, "eligible", ["topic_01_prob", "topic_02_prob"], ["初唐", "盛唐", "中唐", "晚唐"], 0, 20, 7)
        for period in first:
            np.testing.assert_array_equal(first[period]["POEM"], second[period]["POEM"])
            np.testing.assert_array_equal(first[period]["AUTHOR"], second[period]["AUTHOR"])

    def test_concentration_thresholds_are_explicit(self) -> None:
        """作者集中度等级必须只按配置阈值判定。"""

        self.assertEqual(trend.concentration_level(0.10, 0.30, 0.05, THRESHOLDS), "LOW")
        self.assertEqual(trend.concentration_level(0.16, 0.30, 0.05, THRESHOLDS), "MODERATE")
        self.assertEqual(trend.concentration_level(0.26, 0.30, 0.05, THRESHOLDS), "HIGH")


if __name__ == "__main__":
    unittest.main()
