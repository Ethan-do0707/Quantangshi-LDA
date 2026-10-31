"""测试量化候选二次验证的预注册规则与核心计算。"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SPEC = importlib.util.spec_from_file_location(
    "run_quantitative_candidate_validation",
    SRC / "run_quantitative_candidate_validation.py",
)
assert SPEC and SPEC.loader
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


class QuantitativeCandidateValidationTests(unittest.TestCase):
    """覆盖候选、先验、匹配、等价、Pareto 与 shortlist 规则。"""

    def test_registered_design_has_exactly_120_fits(self) -> None:
        config = validation.load_config()
        self.assertEqual([11, 12, 13, 14, 15, 20], validation.REQUIRED_K)
        self.assertEqual(10, len(validation.REQUIRED_SEEDS))
        self.assertEqual(120, len(validation.REQUIRED_K) * len(validation.REQUIRED_SEEDS) * len(validation.REQUIRED_REGIMES))
        self.assertEqual([], config["supplementary_candidates"])

    def test_prior_regimes_only_change_two_priors(self) -> None:
        config = validation.load_config()
        base = validation.scan.read_json(ROOT / "config/tf_df_lda_config.yaml")["lda"]
        original = validation.regime_lda(base, config, "ORIGINAL_PRIOR")
        fixed = validation.regime_lda(base, config, "FIXED_PRIOR_K20")
        differing = {key for key in base if original.get(key) != fixed.get(key)}
        self.assertEqual({"doc_topic_prior", "topic_word_prior"}, differing)
        self.assertEqual(0.05, fixed["doc_topic_prior"])
        self.assertEqual(0.05, fixed["topic_word_prior"])

    def test_hungarian_cosine_alignment_reuses_pairs_for_jaccard(self) -> None:
        first = np.eye(3, dtype=float)
        permutation = [2, 0, 1]
        second = first[permutation]
        first_top = np.array([[1, 2], [3, 4], [5, 6]])
        second_top = first_top[permutation]
        cosine, jaccard = validation.match_seed_pair(first, second, first_top, second_top)
        self.assertAlmostEqual(1.0, cosine)
        self.assertAlmostEqual(1.0, jaccard)

    def test_practical_equivalence_reports_all_three_scales(self) -> None:
        config = validation.load_config()
        status = validation.practical_equivalence_status(0.0004, True, config)
        self.assertIn("ABS_LE_0.0005", status)
        self.assertIn("ABS_LE_0.0010", status)
        self.assertIn("BOTH_WITHIN_1PCT_GLOBAL_MAX", status)

    def test_pareto_dominance_respects_metric_directions(self) -> None:
        better = {
            "NPMI": 0.10,
            "test_perplexity": 100.0,
            "topic_diversity": 0.9,
            "cosine_stability": 0.8,
            "jaccard_stability": 0.5,
            "redundancy": 0.1,
            "low_prevalence": 0.0,
        }
        worse = {
            "NPMI": 0.09,
            "test_perplexity": 110.0,
            "topic_diversity": 0.8,
            "cosine_stability": 0.7,
            "jaccard_stability": 0.4,
            "redundancy": 0.2,
            "low_prevalence": 1.0,
        }
        self.assertTrue(validation.dominates(better, worse))
        self.assertFalse(validation.dominates(worse, better))

    def test_shortlist_is_capped_and_keeps_k20(self) -> None:
        config = validation.load_config()
        rows = []
        for regime in validation.REQUIRED_REGIMES:
            for topic_count in validation.REQUIRED_K:
                rows.append(
                    {
                        "K": topic_count,
                        "prior_regime": regime,
                        "NPMI_mean": 1.0 - abs(topic_count - 12) * 0.001,
                        "mean_matched_cosine": 0.9 - abs(topic_count - 14) * 0.01,
                        "mean_matched_jaccard": 0.6 - abs(topic_count - 14) * 0.01,
                        "test_perplexity_mean": 100 + topic_count,
                        "mean_pairwise_topic_cosine_mean": 0.1,
                    }
                )
        summary = pd.DataFrame(rows)
        statuses = pd.DataFrame(
            {
                "K": validation.REQUIRED_K,
                "quantitative_status": ["QUANT_COMPETITIVE"] * 6,
                "platform_under_both_priors": [True] * 5 + [False],
                "seed_sensitive": [False] * 6,
                "redundancy_risk": [False] * 6,
            }
        )
        shortlist = validation.build_shortlist(summary, statuses, config)
        self.assertLessEqual(len(shortlist), 4)
        self.assertIn(20, shortlist["K"].tolist())
        self.assertIn(12, shortlist["K"].tolist())

    def test_synthetic_summary_figures_and_report(self) -> None:
        config = validation.load_config()
        run_rows = []
        for regime in validation.REQUIRED_REGIMES:
            for topic_count in validation.REQUIRED_K:
                for seed_index, seed in enumerate(validation.REQUIRED_SEEDS):
                    row = {field: 0.0 for field in validation.ALL_RUN_FIELDS}
                    row.update(
                        {
                            "K": topic_count,
                            "seed": seed,
                            "prior_regime": regime,
                            "NPMI": 0.1 - abs(topic_count - 12) * 0.0002 + seed_index * 0.000001,
                            "test_perplexity": 1000 + topic_count * 10 + seed_index,
                            "train_perplexity": 900 + topic_count * 10 + seed_index,
                            "test_score": -100.0,
                            "train_score": -500.0,
                            "topic_diversity": 0.9 - topic_count * 0.001,
                            "mean_pairwise_topic_cosine": 0.10 + topic_count * 0.001,
                            "max_pairwise_topic_cosine": 0.20 + topic_count * 0.001,
                            "mean_pairwise_top20_jaccard": 0.05,
                            "max_pairwise_top20_jaccard": 0.10,
                            "fit_time": 1.0,
                            "n_iter": 30,
                            "doc_topic_prior_effective": 1 / topic_count if regime == "ORIGINAL_PRIOR" else 0.05,
                            "topic_word_prior_effective": 1 / topic_count if regime == "ORIGINAL_PRIOR" else 0.05,
                            "n_train": 34412,
                            "n_test": 8603,
                            "vocab_size": 35063,
                        }
                    )
                    run_rows.append(row)
        all_runs = pd.DataFrame(run_rows)
        stability_rows = []
        for regime in validation.REQUIRED_REGIMES:
            for topic_count in validation.REQUIRED_K:
                for seed_a, seed_b in __import__("itertools").combinations(validation.REQUIRED_SEEDS, 2):
                    stability_rows.append(
                        {
                            "K": topic_count,
                            "prior_regime": regime,
                            "seed_A": seed_a,
                            "seed_B": seed_b,
                            "matched_cosine": 0.8 - abs(topic_count - 14) * 0.001,
                            "matched_jaccard": 0.5 - abs(topic_count - 14) * 0.001,
                        }
                    )
        stability = pd.DataFrame(stability_rows)
        summary = validation.build_candidate_summary(all_runs, stability, config)
        pairwise = validation.build_pairwise_comparison(all_runs, summary, config)
        prior = validation.build_prior_sensitivity(summary, config)
        pareto = validation.build_pareto_front(summary)
        statuses = validation.classify_candidate_statuses(summary, prior, pareto, config)
        shortlist = validation.build_shortlist(summary, statuses, config)
        plateau = validation.classify_plateau(summary)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "figures").mkdir()
            validation.generate_figures(all_runs, summary, output)
            validation.generate_report(
                all_runs,
                summary,
                stability,
                pairwise,
                prior,
                pareto,
                statuses,
                shortlist,
                plateau,
                {"output": output},
            )
            self.assertEqual(6, len(list((output / "figures").glob("*.png"))))
            self.assertTrue((output / "QTS_Quantitative_K_Candidate_Validation_Report.md").is_file())
        self.assertEqual(12, len(summary))
        self.assertEqual(12, len(pairwise))
        self.assertLessEqual(len(shortlist), 4)


if __name__ == "__main__":
    unittest.main()
