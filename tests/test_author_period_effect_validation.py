"""作者时期效应量与稳健推断核心规则测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_author_period_effect_validation as effect


class AuthorPeriodEffectValidationTest(unittest.TestCase):
    """验证方向、效应量、FDR、稳健协方差与 CLR。"""

    def test_bh_adjust_is_monotone_in_rank(self) -> None:
        """BH 校正必须按秩单调且不超过1。"""

        p_values = np.array([0.04, 0.001, 0.03, 0.8])
        adjusted = effect.bh_adjust(p_values)
        order = np.argsort(p_values)
        self.assertTrue(np.all(np.diff(adjusted[order]) >= -1e-15))
        self.assertTrue(np.all(adjusted <= 1.0))

    def test_hedges_g_uses_b_minus_a_direction(self) -> None:
        """Hedges' g 的正号必须表示 B 高于 A。"""

        value = effect.hedges_g(np.array([0.0, 1.0, 2.0]), np.array([2.0, 3.0, 4.0]))
        self.assertGreater(value, 0)

    def test_cliffs_delta_uses_b_minus_a_direction(self) -> None:
        """Cliff's delta 的正号必须表示 B 倾向高于 A。"""

        value = effect.cliffs_delta(np.array([0.0, 1.0]), np.array([2.0, 3.0]))
        self.assertAlmostEqual(value, 1.0)

    def test_period_cell_means_equal_group_means(self) -> None:
        """无截距时期 dummy 模型系数必须等于四组均值。"""

        periods = np.array(["初唐", "初唐", "盛唐", "盛唐", "中唐", "中唐", "晚唐", "晚唐"])
        values = np.array([1, 3, 2, 4, 4, 6, 8, 10], dtype=float)
        groups = np.arange(len(values))
        model = effect.fit_period_model(values, periods, groups, ["初唐", "盛唐", "中唐", "晚唐"], "CLUSTER_CR1")
        np.testing.assert_allclose(model["beta"], [2, 3, 5, 9])

    def test_hc3_model_contrast_matches_raw_difference(self) -> None:
        """AUTHOR HC3 planned contrast 点估计必须等于组均值差。"""

        periods = np.repeat(["初唐", "盛唐", "中唐", "晚唐"], 3)
        values = np.array([1, 2, 3, 2, 3, 4, 4, 5, 6, 8, 9, 10], dtype=float)
        model = effect.fit_period_model(values, periods, None, ["初唐", "盛唐", "中唐", "晚唐"], "HC3")
        contrast = effect.model_contrast(model, "初唐", "晚唐", ["初唐", "盛唐", "中唐", "晚唐"])
        self.assertAlmostEqual(contrast["model_estimate"], 7.0)
        self.assertGreater(contrast["model_se"], 0)

    def test_clr_rows_sum_to_zero(self) -> None:
        """CLR 临时矩阵每行必须以数值精度和为零。"""

        config = {"clr": {"pseudocount": None}}
        transformed, zero_count, pseudocount = effect.clr_transform(np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]]), config)
        np.testing.assert_allclose(transformed.sum(axis=1), 0.0, atol=1e-12)
        self.assertEqual(zero_count, 0)
        self.assertIsNone(pseudocount)

    def test_true_zero_requires_explicit_pseudocount(self) -> None:
        """遇到真零且未授权 pseudocount 时必须停止。"""

        with self.assertRaises(ValueError):
            effect.clr_transform(np.array([[0.0, 1.0]]), {"clr": {"pseudocount": None}})

    def test_bootstrap_direction_uses_interval(self) -> None:
        """bootstrap 区间跨零时不得宣称受支持方向。"""

        self.assertEqual(effect.bootstrap_direction(0.01, 0.03), "SUPPORTED_POSITIVE")
        self.assertEqual(effect.bootstrap_direction(-0.03, -0.01), "SUPPORTED_NEGATIVE")
        self.assertEqual(effect.bootstrap_direction(-0.01, 0.02), "UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
