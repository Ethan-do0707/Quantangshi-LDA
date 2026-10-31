"""跨模型发现层稳健性核心规则测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_bcd_finding_robustness as pipeline


class BCDFindingRobustnessTest(unittest.TestCase):
    """验证映射、估计量、作者重采样和保守分类。"""

    def test_finding_key_preserves_specific_contrast(self) -> None:
        """同一Topic不同contrast必须是不同发现。"""
        self.assertNotEqual(pipeline.finding_id("topic_03", "E-H"), pipeline.finding_id("topic_03", "E-L"))

    def test_invalid_boolean_rejected(self) -> None:
        """缺失eligibility不能自动通过。"""
        with self.assertRaises(ValueError):
            pipeline.bool_series(pd.Series(["True", ""]), "eligible")

    def test_split_bundle_is_unweighted_sum(self) -> None:
        """bundle不使用cosine权重。"""
        matrix = np.array([[.2, .3, .5], [.1, .7, .2]])
        np.testing.assert_array_equal(pipeline.probability_sum(matrix, ["topic_01", "topic_03"]), matrix[:, 0] + matrix[:, 2])

    def test_duplicate_bundle_topic_rejected(self) -> None:
        """同一目标不得重复计入概率质量。"""
        with self.assertRaises(ValueError):
            pipeline.probability_sum(np.ones((2, 3)), ["topic_01", "topic_01"])

    def test_single_best_is_from_source_candidates(self) -> None:
        """单一目标敏感性不使用同号或匈牙利编号。"""
        self.assertEqual(pipeline.source_best(["topic_08", "topic_19"], "topic_08:cos=0.7559; topic_19:cos=0.6292; topic_03:cos=0.9999"), "topic_08")

    def test_mapping_categories(self) -> None:
        """merge、weak、unmatched保留各自限制。"""
        self.assertEqual(pipeline.mapping_type("strong_1to1", "HIGH"), "DIRECT")
        self.assertEqual(pipeline.mapping_type("moderate_1to1", "MEDIUM"), "DIRECT")
        self.assertEqual(pipeline.mapping_type("split", "HIGH"), "BUNDLE")
        self.assertEqual(pipeline.mapping_type("merge", "MEDIUM"), "MERGED_PROXY")
        self.assertEqual(pipeline.mapping_type("weak", "LOW"), "WEAK_PROXY")
        self.assertEqual(pipeline.mapping_type("unmatched", "LOW"), "NOT_TESTABLE")

    def test_mapping_reads_source_and_review_flag(self) -> None:
        """来源目标和待审核标记必须保留。"""
        candidate = pd.DataFrame([{"finding_id": "T03_EL", "topic_id": "topic_03", "topic_label": "礼乐", "contrast": "E-L"}])
        row = {"C_topic_id": "topic_03", "notes": "测试", "NEEDS_HUMAN_REVIEW": True}
        for model in ("B", "D"):
            row.update({f"{model}_topic_id": "topic_08;topic_19", f"{model}_original_label": "甲;乙", f"{model}_similarity": .8, f"{model}_alignment_type": "split", f"{model}_alignment_confidence": "HIGH", f"{model}_candidate_details": "topic_08:cos=0.8; topic_19:cos=0.6"})
        result = pipeline.build_mapping(candidate, pd.DataFrame([row]))
        self.assertTrue(result["alignment_review_required"].all())
        self.assertTrue(result["target_topic_ids"].eq("topic_08;topic_19").all())

    def test_hedges_and_cliff_orientation(self) -> None:
        """标准化效应均以B-A为方向。"""
        a, b = np.array([.1, .2, .3]), np.array([.4, .5, .6])
        self.assertGreater(pipeline.hedges_g(a, b), 0)
        self.assertEqual(pipeline.cliffs_delta(a, b), 1)
        self.assertAlmostEqual(pipeline.hedges_g(a, b), -pipeline.hedges_g(b, a))

    def test_bootstrap_reproducible_and_distinct_estimands(self) -> None:
        """作者整簇抽签可复现，诗等权不等于作者等权。"""
        authors = pd.DataFrame({"poem_count": [1, 5], "t_mass": [.1, 4], "t_mean": [.1, .8]})
        first = pipeline.bootstrap_period(authors, ["t"], 300, 42)
        second = pipeline.bootstrap_period(authors, ["t"], 300, 42)
        for aggregation in pipeline.AGGREGATIONS:
            np.testing.assert_array_equal(first[aggregation], second[aggregation])
        self.assertFalse(np.array_equal(first["POEM"], first["AUTHOR"]))

    def test_author_table_keeps_whole_cluster(self) -> None:
        """作者诗数与质量之和必须同时保留。"""
        frame = pd.DataFrame({"author_id_internal": ["a", "a", "b"], "author": ["甲", "甲", "乙"], "author_period_primary": ["初唐"] * 3, "t": [.2, .4, .1]})
        table = pipeline.author_table(frame, ["t"]).set_index("author_id_internal")
        self.assertEqual(table.loc["a", "poem_count"], 2)
        self.assertAlmostEqual(table.loc["a", "t_mass"], .6)
        self.assertAlmostEqual(table.loc["a", "t_mean"], .3)

    def test_core_estimands_match_definitions(self) -> None:
        """分析行必须分别实现诗mean和作者mean-of-means。"""
        config = {"period_order": ["初唐", "盛唐", "中唐", "晚唐"], "period_codes": dict(zip(["初唐", "盛唐", "中唐", "晚唐"], "EHML")), "samples": {"CORE": "eligible"}, "random_seed": 42, "bootstrap_iterations": 100, "bootstrap_batch_size": 50,
                  "thresholds": {"hedges_small": .2, "hedges_moderate": .5, "hedges_large": .8, "cliff_small": .147, "cliff_medium": .33, "cliff_large": .474}}
        rows = []
        for i, period in enumerate(config["period_order"]):
            for author, values in (("a", [.1 + i * .1, .3 + i * .1]), ("b", [.2 + i * .05])):
                rows.extend({"author_id_internal": f"{i}{author}", "author": author, "author_period_primary": period, "eligible": True, "topic_03": value} for value in values)
        frame = pd.DataFrame(rows)
        candidates = pd.DataFrame([{"finding_id": "T03_EL", "topic_id": "topic_03", "topic_label": "测试", "contrast": "E-L"}])
        effects, authors, _ = pipeline.analyze(frame, "C", candidates, pd.DataFrame(), config)
        for row in effects.to_dict("records"):
            early = frame.loc[frame["author_period_primary"].eq("初唐")]
            expected = early["topic_03"].mean() if row["aggregation"] == "POEM" else early.groupby("author_id_internal")["topic_03"].mean().mean()
            self.assertAlmostEqual(row["period_A_mean"], expected)
        combined = pd.concat([effects.assign(model=model) for model in pipeline.MODELS], ignore_index=True)
        config["priority_findings"] = ["T03_EL"]
        config["minimal_author_check"] = {"top_n": [1, 3]}
        checks = pipeline.minimal_author_checks({model: authors for model in pipeline.MODELS}, combined, candidates, config)
        self.assertEqual(len(checks), 8)
        self.assertTrue(checks["check_run"].all())
        empty_period = checks.loc[checks["N"].eq(3)]
        self.assertTrue(empty_period["excluded_delta"].isna().all())
        self.assertTrue(empty_period["author_sensitivity_flag"].eq("NOT_ESTIMABLE_AFTER_EXCLUSION").all())

    def evidence(self) -> dict[str, bool]:
        """构造全部通过的最小分类输入。"""
        return {"unresolved": False, "both_strong_not_replicated": False, "systematic_reversal": False, "core_agreement": True, "target_support": True, "sample_consistency": True, "semantic_limit": False, "review_required": False, "bundle_dependency": False, "author_sensitive": False}

    def test_robust_requires_all_evidence_layers(self) -> None:
        """语义、作者或bundle限制均不能升级为高稳健。"""
        self.assertEqual(pipeline.classify_robustness(self.evidence()), "ROBUST_ACROSS_BCD")
        for flag in ("semantic_limit", "review_required", "bundle_dependency", "author_sensitive"):
            evidence = self.evidence(); evidence[flag] = True
            self.assertEqual(pipeline.classify_robustness(evidence), "PARTIALLY_ROBUST")

    def test_unresolved_not_called_nonreplication(self) -> None:
        """弱语义不能误判为未复现。"""
        evidence = self.evidence(); evidence["unresolved"] = True; evidence["systematic_reversal"] = True
        self.assertEqual(pipeline.classify_robustness(evidence), "SEMANTICALLY_UNRESOLVED")

    def test_systematic_reversal_is_visible(self) -> None:
        """达到固定阈值的反向必须得到模型敏感分类。"""
        evidence = self.evidence(); evidence["systematic_reversal"] = True
        self.assertEqual(pipeline.classify_robustness(evidence), "MODEL_SENSITIVE")
        evidence["both_strong_not_replicated"] = True
        self.assertEqual(pipeline.classify_robustness(evidence), "NOT_REPLICATED")

    def test_uncertain_ci_not_zero_effect(self) -> None:
        """区间不确定与数值正负分开。"""
        self.assertEqual(pipeline.sign(.01), "POSITIVE")
        self.assertEqual(pipeline.supported(-.02, .03), "UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
