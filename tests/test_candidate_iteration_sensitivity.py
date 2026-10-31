"""测试候选 max_iter=60 敏感性复核的范围与 K20 复用规则。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SPEC = importlib.util.spec_from_file_location(
    "run_candidate_iteration_sensitivity",
    SRC / "run_candidate_iteration_sensitivity.py",
)
assert SPEC and SPEC.loader
sensitivity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sensitivity)


class CandidateIterationSensitivityTests(unittest.TestCase):
    """覆盖触发后的 21 次唯一拟合和 24 行登记口径。"""

    def test_unique_plan_has_21_fits(self) -> None:
        plan = sensitivity.unique_fit_plan()
        self.assertEqual(21, len(plan))
        self.assertNotIn(("FIXED_PRIOR_K20", 20, 42), plan)
        self.assertIn(("ORIGINAL_PRIOR", 20, 42), plan)

    def test_k20_equivalent_prior_rows_are_explicitly_reused(self) -> None:
        rows = []
        for seed in sensitivity.ORIGINAL_SEEDS:
            rows.append(
                {
                    "K": 20,
                    "seed": seed,
                    "prior_regime": "ORIGINAL_PRIOR",
                    "unique_fit": True,
                    "reused_from_prior_regime": "",
                }
            )
        expanded = sensitivity.expand_reused_k20(rows)
        self.assertEqual(6, len(expanded))
        reused = [row for row in expanded if not row["unique_fit"]]
        self.assertEqual(3, len(reused))
        self.assertTrue(all(row["prior_regime"] == "FIXED_PRIOR_K20" for row in reused))


if __name__ == "__main__":
    unittest.main()
