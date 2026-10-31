"""验证 TF 阈值敏感性实验的唯一变量与统计逻辑。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "src" / "run_tf_sensitivity.py"


def load_experiment():
    """从文件路径加载实验模块。"""

    spec = importlib.util.spec_from_file_location("run_tf_sensitivity", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 TF 敏感性实验程序")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TfSensitivityTest(unittest.TestCase):
    """检查档位构造、DF 交互、保护优先和百分位数。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = load_experiment()

    def test_only_min_tf_changes(self) -> None:
        """四组规则只能改变 min_tf。"""

        frozen = {
            "stopword_tier": "core",
            "min_tf": 5,
            "min_df": 3,
            "automatic_max_df": None,
        }
        profiles = {
            f"TF{threshold}": self.experiment.build_threshold_profile(
                frozen, threshold
            )
            for threshold in (1, 2, 3, 5)
        }
        self.experiment.validate_threshold_profiles(profiles, frozen)
        self.assertEqual([profile["min_tf"] for profile in profiles.values()], [1, 2, 3, 5])
        self.assertTrue(all(profile["min_df"] == 3 for profile in profiles.values()))

    def test_fixed_df_collapses_tf1_tf2_tf3(self) -> None:
        """DF≥3 时 TF=1、2 的非保护词在前三组都无法进入语料。"""

        tf = Counter({"孤词": 1, "双见": 2, "三见": 3})
        df = Counter({"孤词": 1, "双见": 2, "三见": 3})
        vocabularies = []
        for threshold in (1, 2, 3):
            decisions = self.experiment.build_decisions(
                tf,
                df,
                10,
                set(),
                set(),
                {
                    "min_tf": threshold,
                    "min_df": 3,
                    "automatic_max_df": None,
                },
            )
            vocabularies.append({token for token, decision in decisions.items() if decision[0]})
        self.assertEqual(vocabularies, [{"三见"}, {"三见"}, {"三见"}])

    def test_protected_word_survives_all_thresholds(self) -> None:
        """TF=1、DF=1 的保护词必须在四组全部保留。"""

        tf = Counter({"函谷关": 1})
        df = Counter({"函谷关": 1})
        for threshold in (1, 2, 3, 5):
            decisions = self.experiment.build_decisions(
                tf,
                df,
                10,
                {"函谷关"},
                set(),
                {
                    "min_tf": threshold,
                    "min_df": 3,
                    "automatic_max_df": None,
                },
            )
            self.assertTrue(decisions["函谷关"][0])
            self.assertEqual(decisions["函谷关"][1], "PROTECTED")

    def test_percentile_uses_linear_interpolation(self) -> None:
        """百分位数必须使用明确的线性插值口径。"""

        self.assertEqual(self.experiment.percentile([0, 10], 0.25), 2.5)
        self.assertEqual(self.experiment.percentile([0, 10], 0.50), 5.0)
        self.assertEqual(self.experiment.percentile([0, 10], 0.99), 9.9)

    def test_subsequence_preserves_token_order(self) -> None:
        """更严格语料只能从宽松语料删除 token，不得改序。"""

        self.assertTrue(self.experiment.is_subsequence(["山", "月"], ["山", "高", "月"]))
        self.assertFalse(self.experiment.is_subsequence(["月", "山"], ["山", "月"]))


if __name__ == "__main__":
    unittest.main()
