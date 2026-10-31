"""验证 LDA 语料清洗程序的关键约束。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "src" / "clean_corpus.py"


def load_cleaner():
    """从文件路径加载清洗模块。"""

    spec = importlib.util.spec_from_file_location("clean_corpus", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载清洗程序")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CleanCorpusTest(unittest.TestCase):
    """检查保护优先级、规范范围和切分审计。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cleaner = load_cleaner()

    def test_protected_word_overrides_every_filter(self) -> None:
        """保护词必须覆盖停用、低频和非汉字判断。"""

        keep, reason, all_reasons, rescued = self.cleaner.token_decision(
            "测试",
            1,
            1,
            0.01,
            {"测试"},
            {"测试"},
            5,
            3,
            None,
        )
        self.assertTrue(keep)
        self.assertEqual(reason, "PROTECTED")
        self.assertIn("STOPWORD", all_reasons)
        self.assertTrue(rescued)

    def test_filter_reason_precedence_is_deterministic(self) -> None:
        """同一词触发多条规则时必须给出稳定主原因并保留完整原因。"""

        keep, reason, all_reasons, rescued = self.cleaner.token_decision(
            "，", 1, 1, 0.9, set(), set(), 5, 3, 0.5
        )
        self.assertFalse(keep)
        self.assertEqual(reason, "OTHER_RULE")
        self.assertEqual(all_reasons, ("OTHER_RULE", "LOW_TF", "LOW_DF", "HIGH_DF"))
        self.assertFalse(rescued)

    def test_automatic_normalization_requires_explicit_permission(self) -> None:
        """未获自动执行许可的词形记录不得改写 token。"""

        rule = self.cleaner.NormalizationRule(
            "牀", "床", "异体字", "全局", "待审核", False, 2
        )
        tokens, events = self.cleaner.apply_normalization(1, ["牀"], [rule], True)
        self.assertEqual(tokens, ["牀"])
        self.assertEqual(events, [])

    def test_normalization_is_exact_and_scope_bound(self) -> None:
        """规范化只匹配完整 token 和明确 poem_id。"""

        rule = self.cleaner.NormalizationRule(
            "桑乾", "桑干", "地名词形", "poem_id=778", "可用", True, 2
        )
        matched, events = self.cleaner.apply_normalization(
            778, ["桑乾", "桑乾河"], [rule], True
        )
        unmatched, _ = self.cleaner.apply_normalization(779, ["桑乾"], [rule], True)
        self.assertEqual(matched, ["桑干", "桑乾河"])
        self.assertEqual(len(events), 1)
        self.assertEqual(unmatched, ["桑乾"])

    def test_split_protected_word_is_reported_without_merging(self) -> None:
        """保护词跨多个 token 时只报告，不自动合并。"""

        record = self.cleaner.PoemRecord(1, "甲", "题", "彩凤来仪", ["彩凤", "来", "仪"])
        record.normalized_tokens = ["彩凤", "来", "仪"]
        issues = self.cleaner.find_protected_segmentation_issues(
            record,
            [
                {
                    "surface": "来仪",
                    "decision": "强制合并",
                    "category": "典故",
                    "scope": "全局",
                }
            ],
        )
        self.assertEqual(record.normalized_tokens, ["彩凤", "来", "仪"])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["observed_tokens"], "来 / 仪")


if __name__ == "__main__":
    unittest.main()
