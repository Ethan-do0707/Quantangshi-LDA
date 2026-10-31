"""验证 TF×DF 语料和 LDA 指标的关键逻辑。"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "src" / "run_tf_df_lda.py"


def load_experiment():
    """从文件路径加载实验模块。"""

    spec = importlib.util.spec_from_file_location("run_tf_df_lda", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 TF×DF LDA 程序")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TfDfLdaTest(unittest.TestCase):
    """检查阈值、保护优先、BoW、NPMI 和稳定性。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = load_experiment()

    def test_default_corpus_definitions(self) -> None:
        """A/B/C/D 必须使用用户确认的四组阈值。"""

        definitions = {
            label: {"min_tf": values[0], "min_df": values[1]}
            for label, values in self.experiment.REQUIRED_CORPORA.items()
        }
        self.experiment.validate_corpus_definitions(definitions)

    def test_protected_word_bypasses_both_minimums(self) -> None:
        """保护词必须同时豁免最低 TF 与最低 DF。"""

        decisions = self.experiment.build_decisions(
            Counter({"函谷关": 1}),
            Counter({"函谷关": 1}),
            10,
            {"函谷关"},
            set(),
            5,
            3,
            None,
        )
        self.assertTrue(decisions["函谷关"][0])
        self.assertEqual(decisions["函谷关"][1], "PROTECTED")

    def test_sparse_matrix_preserves_counts(self) -> None:
        """稀疏 BoW 必须保持文档顺序和词频。"""

        matrix = self.experiment.build_sparse_matrix(
            [["山", "山", "月"], ["月"]], {"山": 0, "月": 1}
        )
        np.testing.assert_array_equal(matrix.toarray(), np.array([[2, 1], [0, 1]]))
        self.assertEqual(matrix.dtype, np.int32)

    def test_npmi_rewards_consistent_cooccurrence(self) -> None:
        """总是共现的一对词应获得正 NPMI。"""

        matrix = self.experiment.build_sparse_matrix(
            [["山", "月"], ["山", "月"], ["水"]],
            {"山": 0, "月": 1, "水": 2},
        )
        topic_scores, mean_score = self.experiment.topic_npmi_scores(
            matrix, [[0, 1]]
        )
        self.assertGreater(topic_scores[0], 0)
        self.assertEqual(topic_scores[0], mean_score)

    def test_hungarian_stability_matches_permuted_topics(self) -> None:
        """主题顺序置换不应降低完全一致主题的稳定性。"""

        first = [{"山", "月"}, {"江", "水"}]
        second = [{"江", "水"}, {"山", "月"}]
        score, matches = self.experiment.matched_topic_jaccard(first, second)
        self.assertEqual(score, 1.0)
        self.assertEqual(len(matches), 2)

    def test_checkpoint_rejects_tampered_output(self) -> None:
        """检查点必须同时匹配签名与两个结果文件的哈希。"""

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            topics_path = directory / "topics.csv"
            representatives_path = directory / "representative_poems.csv"
            metrics_path = directory / "metrics.json"
            topics_path.write_text("topic_id,token\n1,山\n", encoding="utf-8")
            representatives_path.write_text(
                "topic_id,poem_id\n1,1\n", encoding="utf-8"
            )
            metrics_path.write_text(
                json.dumps(
                    {
                        "checkpoint_signature": "signature",
                        "outputs": {
                            "topics": {
                                "sha256": self.experiment.cleaner.file_sha256(
                                    topics_path
                                )
                            },
                            "representative_poems": {
                                "sha256": self.experiment.cleaner.file_sha256(
                                    representatives_path
                                )
                            },
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                self.experiment.checkpoint_is_valid(
                    metrics_path,
                    topics_path,
                    representatives_path,
                    "signature",
                )
            )
            topics_path.write_text("topic_id,token\n1,水\n", encoding="utf-8")
            self.assertFalse(
                self.experiment.checkpoint_is_valid(
                    metrics_path,
                    topics_path,
                    representatives_path,
                    "signature",
                )
            )


if __name__ == "__main__":
    unittest.main()
