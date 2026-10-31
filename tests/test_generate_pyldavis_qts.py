# -*- coding: utf-8 -*-
"""验证文件缺失与输入错配的停止策略，不训练任何模型。"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation

import generate_pyldavis_qts as qts


class InputSafetyTests(unittest.TestCase):
    def test_shape_mismatch_reports_paths_and_dimensions(self):
        """维度错配须在调用可视化或尝试修补数据之前报告。"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "models/C_K20"
            corpus_dir = root / "corpora/C"
            model_dir.mkdir(parents=True)
            corpus_dir.mkdir(parents=True)
            for name in ("model.joblib", "vocabulary.csv", "document_index.csv",
                         "document_topic_probabilities.npz", "topic_word_components.npz",
                         "matrix_metadata.json", "artifact_manifest.json", "topics_top30.csv"):
                (model_dir / name).write_text("", encoding="utf-8")
            for name in ("bow_matrix.npz", "vocabulary.csv", "document_index.csv", "corpus_metadata.json"):
                (corpus_dir / name).write_text("", encoding="utf-8")
            (model_dir / "vocabulary.csv").write_text("term_index,token\n0,云\n1,月\n", encoding="utf-8")
            # 只构造测试对象的属性，不执行拟合。
            model = LatentDirichletAllocation(n_components=20)
            model.components_ = np.ones((20, 2))
            report = {}
            with patch.object(qts.joblib, "load", return_value=model), patch.object(qts, "load_npz", return_value=csr_matrix((3, 4))):
                with self.assertRaises(qts.ConsistencyError) as failure:
                    qts.load_inputs("C_K20", root / "models", root / "corpora", report)
            message = str(failure.exception)
            for detail in ("model.joblib", "bow_matrix.npz", "vocabulary.csv", "(20, 2)", "(3, 4)", "词项数=2", "seed=2026"):
                self.assertIn(detail, message)

    def run_failure_case(self, failures):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config").mkdir()
            (root / "config/literary_history_dataset_config.json").write_text(
                json.dumps({"output_root": "frozen", "corpus_root": "corpora"}), encoding="utf-8")
            with patch.object(qts, "ROOT", root), patch.object(qts, "load_inputs", side_effect=failures) as loader:
                with contextlib.redirect_stdout(io.StringIO()):
                    status = qts.main(["--check-only"])
                report = json.loads((root / "outputs/pyldavis/input_check.json").read_text(encoding="utf-8"))
            return status, report["models"], loader.call_count

    def test_missing_model_does_not_abort_other_models(self):
        status, reports, calls = self.run_failure_case([FileNotFoundError("缺少模型文件")] * 3)
        self.assertEqual(status, 1)
        self.assertEqual(calls, 3)
        self.assertEqual([row["model_id"] for row in reports], ["C_K20", "B_K20", "D_K20"])
        self.assertTrue(all(row["status"] == "缺少文件" for row in reports))

    def test_main_model_mismatch_stops_following_models(self):
        status, reports, calls = self.run_failure_case([qts.ConsistencyError("原始词表顺序不匹配")])
        self.assertEqual(status, 1)
        self.assertEqual(calls, 1)
        self.assertEqual([row["status"] for row in reports], ["一致性检查失败", "未运行", "未运行"])

    def test_comparison_mismatch_still_checks_remaining_comparison(self):
        failures = [FileNotFoundError("缺少 C"), qts.ConsistencyError("B 词表错配"), FileNotFoundError("缺少 D")]
        _, reports, calls = self.run_failure_case(failures)
        self.assertEqual(calls, 3)
        self.assertEqual([row["status"] for row in reports], ["缺少文件", "一致性检查失败", "缺少文件"])


if __name__ == "__main__":
    unittest.main()
