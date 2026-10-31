"""测试《全唐诗》停用词生成器的基础规则。"""

import json
import tempfile
import unittest
from pathlib import Path

from stopwords.build_stopwords import (
    calculate_filtering_impact,
    is_han_token,
    normalized_entropy,
    validate_curated_groups,
)


class StopwordBuilderTests(unittest.TestCase):
    """验证字符、熵和人工分层规则。"""

    def test_han_token_filter(self):
        """汉字词应保留，标点和混合词应排除。"""
        self.assertTrue(is_han_token("山水"))
        self.assertFalse(is_han_token("山，"))
        self.assertFalse(is_han_token("。"))

    def test_normalized_entropy(self):
        """均匀分布的归一化熵应高于集中分布。"""
        even = normalized_entropy([5, 5], 10, 2)
        concentrated = normalized_entropy([9, 1], 10, 2)
        self.assertAlmostEqual(even, 1.0)
        self.assertGreater(even, concentrated)

    def test_curated_groups_do_not_overlap(self):
        """核心、扩展和保护词不得相互冲突。"""
        core, optional, protected = validate_curated_groups()
        self.assertFalse(set(core) & set(optional))
        self.assertFalse(set(core) & set(protected))
        self.assertFalse(set(optional) & set(protected))

    def test_filtering_impact_distinguishes_source_empty_poems(self):
        """源语料空诗和停词后新增空诗应分别计数。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            results = root / "results.jsonl"
            corpus_records = [
                {"id": 1, "poet": "甲", "title": "空", "text": ""},
                {"id": 2, "poet": "乙", "title": "句", "text": "之"},
            ]
            result_records = [
                {"id": 1, "tokens": []},
                {"id": 2, "tokens": ["之"]},
            ]
            corpus.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in corpus_records),
                encoding="utf-8",
            )
            results.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result_records),
                encoding="utf-8",
            )
            impact = calculate_filtering_impact(corpus, results, {"之"}, {"之"})
            self.assertEqual(impact["source_empty_poems"], 1)
            self.assertEqual(impact["new_empty_poems_after_core"], 1)


if __name__ == "__main__":
    unittest.main()
