"""作者时期元数据代表性审计关键规则测试。"""

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

import run_period_representativeness_audit as audit


class PeriodRepresentativenessAuditTest(unittest.TestCase):
    """验证时期缺失审计的效应量、分箱与风险规则。"""

    def test_productivity_bucket_boundaries(self) -> None:
        """作者产量分箱必须严格符合任务给定边界。"""

        cases = {1: "1 poem", 2: "2–4", 4: "2–4", 5: "5–9", 9: "5–9", 10: "10–19", 49: "20–49", 99: "50–99", 199: "100–199", 200: "200+"}
        self.assertEqual({value: audit.productivity_bucket(value) for value in cases}, cases)

    def test_smd_zero_for_equal_distributions(self) -> None:
        """完全相同分布的 SMD 必须为零。"""

        values = np.array([0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(audit.standardized_mean_difference(values, values), 0.0)

    def test_cliffs_delta_direction(self) -> None:
        """KNOWN 全部高于 UNKNOWN 时 Cliff's delta 应为正一。"""

        self.assertEqual(audit.cliffs_delta_from_u(6.0, 2, 3), 1.0)

    def test_bias_heuristic_is_conservative(self) -> None:
        """低风险须同时满足 Topic、构成与 entropy 条件。"""

        self.assertEqual(audit.overall_bias_level(0.19, 0.09, 0.09, 0.14), "LOW")
        self.assertEqual(audit.overall_bias_level(0.21, 0.09, 0.09, 0.14), "MODERATE")
        self.assertEqual(audit.overall_bias_level(0.51, 0.09, 0.09, 0.14), "HIGH")

    def test_cramers_v_zero_for_identical_proportions(self) -> None:
        """两组构成完全一致时 Cramér's V 应为零。"""

        _, _, value, _ = audit.cramers_v(np.array([[10, 20], [20, 40]]))
        self.assertAlmostEqual(value, 0.0)

    def test_json_records_replace_nan_with_null(self) -> None:
        """不可适用统计量必须进入 JSON null，而不是非法 NaN。"""

        records = audit.records_without_nan(pd.DataFrame([{"value": np.nan, "label": "model"}]))
        self.assertIsNone(records[0]["value"])


if __name__ == "__main__":
    unittest.main()
