"""K12/K14 模型产物恢复核心规则测试。"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/run_artifact_recovery_k12_k14.py"
SPEC = importlib.util.spec_from_file_location("artifact_recovery_k12_k14", MODULE_PATH)
assert SPEC and SPEC.loader
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


class ArtifactRecoveryTests(unittest.TestCase):
    """验证网格、复现、概率与代表 seed 规则。"""

    def test_registered_grid_is_exactly_forty(self) -> None:
        """两 K、十 seed、两 prior 必须正好组成 40 个 run。"""

        grid = {(k, seed, regime) for k in recovery.EXPECTED_K for seed in recovery.EXPECTED_SEEDS for regime in recovery.EXPECTED_REGIMES}
        self.assertEqual(len(grid), 40)

    def test_exact_reproduction_requires_component_hash_and_top_indices(self) -> None:
        """float32 topic-word 一致仍须同时核对 components hash 与 Top20。"""

        unit = np.eye(2, dtype=np.float64)
        top = np.asarray([[0, 1], [1, 0]], dtype=np.int32)
        config = {"reproduction_acceptance": {"mean_matched_cosine_min": 0.9999, "mean_matched_top20_jaccard_min": 0.95}}
        validation, _, order = recovery.validate_reproduction(unit, top, unit.copy(), top.copy(), "same", "same", config)
        self.assertEqual(validation["reproduction_status"], "EXACT_REPRODUCTION")
        np.testing.assert_array_equal(order, np.arange(2))

    def test_hungarian_recovers_topic_permutation(self) -> None:
        """Topic 顺序改变时必须由 Hungarian 明确恢复映射。"""

        unit = np.eye(3, dtype=np.float64)
        top = np.asarray([[0, 3], [1, 4], [2, 5]], dtype=np.int32)
        permutation = np.asarray([2, 0, 1])
        config = {"reproduction_acceptance": {"mean_matched_cosine_min": 0.9999, "mean_matched_top20_jaccard_min": 0.95}}
        validation, _, order = recovery.validate_reproduction(unit, top, unit[permutation], top[permutation], "old", "new", config)
        self.assertEqual(validation["reproduction_status"], "PERMUTATION_EQUIVALENT")
        np.testing.assert_array_equal(unit[permutation][order], unit)

    def test_mismatch_is_not_accepted(self) -> None:
        """未达到预注册 cosine/Jaccard 阈值时不得生成 theta。"""

        saved = np.eye(2, dtype=np.float64)
        reproduced = np.asarray([[0.8, 0.6], [0.6, 0.8]], dtype=np.float64)
        saved_top = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
        reproduced_top = np.asarray([[4, 5], [6, 7]], dtype=np.int32)
        config = {"reproduction_acceptance": {"mean_matched_cosine_min": 0.9999, "mean_matched_top20_jaccard_min": 0.95}}
        validation, _, _ = recovery.validate_reproduction(saved, saved_top, reproduced, reproduced_top, "old", "new", config)
        self.assertEqual(validation["reproduction_status"], "REPRODUCTION_MISMATCH")
        self.assertFalse(validation["accepted"])

    def test_theta_checks_reject_bad_rows(self) -> None:
        """theta 必须同时满足 shape、有限、非负和行和约束。"""

        good = np.asarray([[0.4, 0.6], [0.1, 0.9]], dtype=np.float64)
        self.assertTrue(recovery.theta_checks(good, (2, 2), 1e-10)["passed"])
        bad = np.asarray([[0.4, 0.7], [-0.1, 1.1]], dtype=np.float64)
        self.assertFalse(recovery.theta_checks(bad, (2, 2), 1e-10)["passed"])

    def test_seed_rule_excludes_maximum_npmi(self) -> None:
        """代表 seed 规则不得选中 K 内 NPMI 最高 run。"""

        seeds = recovery.EXPECTED_SEEDS
        runs = []
        pairs = []
        for topic_count in recovery.EXPECTED_K:
            for index, seed in enumerate(seeds):
                runs.append({"K": topic_count, "seed": seed, "prior_regime": "ORIGINAL_PRIOR", "NPMI": 0.09 + index * 0.001})
            for left_index, left in enumerate(seeds):
                for right in seeds[left_index + 1 :]:
                    central_bonus = 0.02 if left == seeds[4] or right == seeds[4] else 0.0
                    pairs.append({"K": topic_count, "prior_regime": "ORIGINAL_PRIOR", "seed_A": left, "seed_B": right, "matched_cosine": 0.7 + central_bonus})
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            runs_path = directory_path / "runs.csv"
            pairs_path = directory_path / "pairs.csv"
            output = directory_path / "output"
            (output / "data").mkdir(parents=True)
            pd.DataFrame(runs).to_csv(runs_path, index=False)
            pd.DataFrame(pairs).to_csv(pairs_path, index=False)
            context = {
                "config": {
                    "candidate_all_runs": str(runs_path),
                    "candidate_cross_seed_stability": str(pairs_path),
                    "representative_seed_rule": {"regime": "ORIGINAL_PRIOR", "central_band_size": 5},
                },
                "output": output,
            }
            selected = recovery.choose_representative_seeds(context)
            self.assertEqual(int(selected.selected.sum()), 2)
            self.assertFalse(bool(selected[selected.selected].is_maximum_NPMI_seed.any()))


if __name__ == "__main__":
    unittest.main()
