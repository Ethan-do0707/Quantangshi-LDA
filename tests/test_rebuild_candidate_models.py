"""六候选精确重建程序的轻量回归测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import clean_corpus as cleaner
import rebuild_candidate_models as rebuild


class RebuildCandidateModelsTest(unittest.TestCase):
    """验证精确比较、矩阵检查和检查点哈希。"""

    def test_exact_row_comparison_rejects_formatted_difference(self) -> None:
        """概率字符串末位变化也必须判为复现失败。"""

        with self.assertRaises(cleaner.CleaningError):
            rebuild.compare_rows_exact(
                [{"topic_id": "1", "probability": "0.100000000000"}],
                [{"topic_id": 1, "probability": "0.100000000001"}],
                ["topic_id", "probability"],
                "topics",
            )

    def test_probability_matrix_checks_accept_valid_shapes(self) -> None:
        """完整矩阵必须具有正确形状且逐行归一。"""

        components = np.asarray([[2.0, 1.0], [1.0, 2.0]])
        topic_word = components / components.sum(axis=1, keepdims=True)
        document_topic = np.asarray([[0.7, 0.3], [0.2, 0.8]])
        corpus = SimpleNamespace(matrix=csr_matrix(np.eye(2)))
        result = rebuild.matrix_checks(
            SimpleNamespace(components_=components),
            topic_word,
            document_topic,
            corpus,
            2,
        )
        self.assertEqual(result["topic_word_shape"], [2, 2])
        self.assertEqual(result["document_topic_shape"], [2, 2])

    def test_environment_validation_reports_each_field(self) -> None:
        """环境差异必须可定位到具体字段。"""

        original = {
            "python": "a",
            "implementation": "CPython",
            "platform": "x",
            "logical_cpu_count": 1,
            "numpy": "1",
            "scipy": "1",
            "scikit_learn": "1",
        }
        current = {**original, "numpy": "2"}
        result = rebuild.validate_environment(current, original)
        self.assertFalse(result["all_match"])
        self.assertFalse(result["fields"]["numpy"]["match"])

    def test_checkpoint_requires_all_registered_hashes(self) -> None:
        """检查点任一文件被改写后都不得复用。"""

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            data_path = directory / "data.bin"
            data_path.write_bytes(b"original")
            manifest = {
                "rebuild_signature": "signature",
                "reproduction_validation_passed": True,
                "outputs": [
                    {
                        "path": data_path.name,
                        "sha256": cleaner.file_sha256(data_path),
                    }
                ],
            }
            (directory / "artifact_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertTrue(rebuild.checkpoint_valid(directory, "signature"))
            data_path.write_bytes(b"changed")
            self.assertFalse(rebuild.checkpoint_valid(directory, "signature"))


if __name__ == "__main__":
    unittest.main()
