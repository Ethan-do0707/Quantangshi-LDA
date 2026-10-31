"""验证低 K 边界与 held-out 困惑度流水线的关键约束。"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "run_lower_k_boundary_perplexity.py"
CONFIG = ROOT / "model_selection/lower_k_boundary_perplexity/config/lower_k_perplexity_config.yaml"


def load_module():
    """从文件路径加载诊断程序。"""

    spec = importlib.util.spec_from_file_location("run_lower_k_boundary_perplexity", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载低 K 诊断程序")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LowerKBoundaryPerplexityTest(unittest.TestCase):
    """检查配置、固定切分、模型参数与原指标读取。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        cls.config = cls.module.load_config(CONFIG)

    def test_exact_experiment_grid(self) -> None:
        """C 含 8 个 K，A/B/D 各含 5 个 K，总计 69 个 held-out run。"""

        count = sum(len(item["topic_counts"]) * 3 for item in self.config["corpora"].values())
        self.assertEqual(count, 69)
        self.assertEqual(self.config["lower_k"], [5, 10, 15])

    def test_model_parameters_match_original(self) -> None:
        """诊断 LDA 的非 K/seed 参数必须与原实验一致。"""

        self.module.ensure_original_parameter_match(self.config)
        model = self.module.build_model(5, 42, self.config["lda"])
        self.assertEqual(model.learning_method, "batch")
        self.assertEqual(model.max_iter, 30)
        self.assertEqual(model.n_jobs, -1)

    def test_fixed_split_is_disjoint_complete_and_reusable(self) -> None:
        """同 corpus 固定切分必须互斥、完整且重复读取一致。"""

        matrix = csr_matrix(np.eye(10, dtype=np.int32))
        corpus = self.module.CorpusView(
            label="C",
            min_tf=3,
            min_df=3,
            root=Path("."),
            matrix=matrix,
            matrix_fingerprint="test",
            vocabulary=[str(index) for index in range(10)],
            document_rows=[{"matrix_row": str(index), "poem_id": str(index)} for index in range(10)],
            corpus_rows=None,
            metadata={},
        )
        with tempfile.TemporaryDirectory() as directory:
            first_train, first_test = self.module.split_indices(corpus, self.config, Path(directory))
            second_train, second_test = self.module.split_indices(corpus, self.config, Path(directory))
        np.testing.assert_array_equal(first_train, second_train)
        np.testing.assert_array_equal(first_test, second_test)
        self.assertFalse(set(first_train) & set(first_test))
        self.assertEqual(len(first_train) + len(first_test), 10)

    def test_original_metrics_reconcile(self) -> None:
        """原 K20–60 指标必须由权威逐 run 表精确复原。"""

        rows = self.module.original_metrics_consistency(self.config)
        self.assertEqual([row["K"] for row in rows], [20, 30, 40, 50, 60])

    def test_metric_functions_are_directly_reused(self) -> None:
        """NPMI、Diversity 与 Stability 必须直接来自原模块。"""

        self.assertEqual(self.module.original.topic_npmi_scores.__module__, "run_tf_df_lda")
        self.assertEqual(self.module.original.topic_diversity.__module__, "run_tf_df_lda")
        self.assertEqual(self.module.original.matched_topic_jaccard.__module__, "run_tf_df_lda")

    def test_existing_c_k20_human_score_uses_authoritative_summary(self) -> None:
        """既有 C_K20 人评读取权威模型汇总，不把空白低 K 人评自动补值。"""

        self.assertEqual(self.module.existing_c_k20_human_score(self.config), 3.75)

    def test_boundary_not_cleared_status_is_reachable(self) -> None:
        """K15 自动指标全面占优且合并率低时必须触发重新审议。"""

        full = self.module.pd.DataFrame([
            {"K": 15, "mean_NPMI": 0.12, "mean_Stability": 0.40, "mean_Topic_Diversity": 0.60, "mean_test_perplexity": 9000},
            {"K": 20, "mean_NPMI": 0.10, "mean_Stability": 0.30, "mean_Topic_Diversity": 0.50, "mean_test_perplexity": 10000},
        ])
        merge = self.module.pd.DataFrame([{"lower_K": 15, "lower_k_merge_ratio": 0.20}])
        boundary, integrated, _reason = self.module.choose_status(full, merge)
        self.assertEqual(boundary, "BOUNDARY_NOT_CLEARED")
        self.assertEqual(integrated, "MODEL_SELECTION_REVIEW_REQUIRED")

    def test_heldout_fit_receives_train_rows_only(self) -> None:
        """held-out 执行路径不得把测试行传给 fit。"""

        calls = []

        class FakeModel:
            """记录矩阵用途的轻量测试模型。"""

            n_iter_ = 30

            def fit(self, matrix):
                calls.append(("fit", tuple(matrix.shape), matrix.toarray().copy()))
                return self

            def perplexity(self, matrix):
                calls.append(("perplexity", tuple(matrix.shape), None))
                return 10.0 + matrix.shape[0]

            def score(self, matrix):
                calls.append(("score", tuple(matrix.shape), None))
                return -100.0 - matrix.shape[0]

        matrix = csr_matrix(np.arange(18, dtype=np.int32).reshape(6, 3))
        corpus = self.module.CorpusView(
            label="C",
            min_tf=3,
            min_df=3,
            root=Path("."),
            matrix=matrix,
            matrix_fingerprint="test",
            vocabulary=["一", "二", "三"],
            document_rows=[{"matrix_row": str(index), "poem_id": str(index)} for index in range(6)],
            corpus_rows=None,
            metadata={},
        )
        train, test = np.array([0, 1, 2, 3]), np.array([4, 5])
        logger = self.module.logging.getLogger("heldout-test")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            self.module, "build_model", return_value=FakeModel()
        ):
            result = self.module.run_heldout_model(
                corpus, 5, 42, train, test, self.config, Path(directory), logger
            )
        fit_calls = [item for item in calls if item[0] == "fit"]
        self.assertEqual(len(fit_calls), 1)
        self.assertEqual(fit_calls[0][1], (4, 3))
        np.testing.assert_array_equal(fit_calls[0][2], matrix[train].toarray())
        self.assertFalse(result["test_data_used_for_fit"])


if __name__ == "__main__":
    unittest.main()
