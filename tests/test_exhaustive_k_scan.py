"""测试 C corpus K=2–30 穷举诊断的核心规则。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SPEC = importlib.util.spec_from_file_location("run_exhaustive_k_scan", SRC / "run_exhaustive_k_scan.py")
assert SPEC and SPEC.loader
scan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scan)


class ExhaustiveKScanTests(unittest.TestCase):
    """覆盖形状分类、连续区间、汇总与模型参数审计。"""

    def test_required_grid_has_87_runs(self) -> None:
        self.assertEqual(list(range(2, 31)), scan.REQUIRED_K)
        self.assertEqual(87, len(scan.REQUIRED_K) * len(scan.REQUIRED_SEEDS))

    def test_perplexity_shape_rules(self) -> None:
        self.assertEqual("STRICTLY_INCREASING", scan.classify_perplexity([1, 2, 3, 4]))
        self.assertEqual("U_SHAPED", scan.classify_perplexity([4, 2, 1, 3, 5]))
        self.assertEqual("NON_MONOTONIC", scan.classify_perplexity([1, 3, 2, 4], mostly_fraction=0.8))

    def test_contiguous_ranges(self) -> None:
        self.assertEqual([(2, 4), (7, 7), (9, 10)], scan.contiguous_runs([10, 3, 2, 4, 9, 7]))
        self.assertEqual("K2–K4、K7、K9–K10", scan.format_k_ranges([10, 3, 2, 4, 9, 7]))

    def test_npmi_shape_detects_broad_plateau(self) -> None:
        frame = pd.DataFrame({"K": list(range(2, 7)), "mean_npmi": [0.08, 0.099, 0.1, 0.0995, 0.08], "npmi_local_max": [False, False, True, False, False], "npmi_top_98_percent_band": [False, True, True, True, False]})
        self.assertEqual("BROAD_PLATEAU", scan.classify_npmi(frame, 3))

    def test_summary_marks_strict_local_extrema_and_plateau(self) -> None:
        rows = []
        for topic_count in scan.REQUIRED_K:
            for offset, seed in enumerate(scan.REQUIRED_SEEDS):
                npmi = 0.1 - abs(topic_count - 15) * 0.001 + offset * 0.00001
                perplexity = 1000.0 + topic_count * 10 + offset
                rows.append({
                    "K": topic_count, "seed": seed, "n_train": 8, "n_test": 2, "vocab_size": 4,
                    "train_score": -10.0, "test_score": -3.0, "train_perplexity": 3.0,
                    "test_perplexity": perplexity, "npmi_coherence": npmi, "n_iter_": 30,
                    "fit_time_seconds": 1.0, "doc_topic_prior_effective": 1 / topic_count,
                    "topic_word_prior_effective": 1 / topic_count, "converged_flag": False,
                    "finite_perplexity_flag": True,
                })
        artifacts = scan.summarize_runs(pd.DataFrame(rows), scan.load_config(scan.DEFAULT_CONFIG))
        summary = artifacts["summary"]
        self.assertEqual([15], summary.loc[summary["npmi_local_max"], "K"].tolist())
        self.assertEqual([], summary.loc[summary["perplexity_local_min"], "K"].tolist())
        self.assertEqual(2, int(summary.loc[summary["perplexity_rank"] == 1, "K"].iloc[0]))
        self.assertIn(15, artifacts["candidates"]["K"].tolist())
        self.assertIn(20, artifacts["candidates"]["K"].tolist())

    def test_original_npmi_function_is_reused(self) -> None:
        source = Path(scan.original.__file__).read_text(encoding="utf-8")
        self.assertIn("def topic_npmi_scores", source)
        self.assertIs(scan.original.topic_npmi_scores, scan.original.topic_npmi_scores)

    def test_effective_default_priors_follow_one_over_k(self) -> None:
        config = scan.load_config(scan.DEFAULT_CONFIG)
        lda, audited = scan.validate_lda_source(config)
        self.assertIsNone(audited["doc_topic_prior"])
        self.assertIsNone(audited["topic_word_prior"])
        self.assertEqual(0.7, audited["learning_decay"])
        self.assertEqual(10.0, audited["learning_offset"])
        model = scan.lower.build_model(4, 42, lda)
        tiny = np.array([[2, 0, 1], [0, 2, 1], [1, 1, 0], [3, 0, 0]], dtype=float)
        model.set_params(max_iter=1, n_jobs=1, verbose=0).fit(tiny)
        self.assertEqual(0.25, model.doc_topic_prior_)
        self.assertEqual(0.25, model.topic_word_prior_)


if __name__ == "__main__":
    unittest.main()
