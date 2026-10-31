import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


STANDARD_PATH = Path(__file__).resolve().parents[1] / "lda_standard" / "apply_standard.py"
SPEC = importlib.util.spec_from_file_location("lda_standard_apply", STANDARD_PATH)
STANDARD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STANDARD
SPEC.loader.exec_module(STANDARD)


class LdaStandardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        standard_dir = STANDARD_PATH.parent
        rules = STANDARD.load_merge_rules(
            standard_dir / "protected_term_seed.tsv",
            standard_dir / "named_entity_seed.tsv",
        )
        cls.matcher = STANDARD.LongestTermMatcher(rules)
        cls.split_patterns = STANDARD.load_split_patterns(standard_dir / "non_merge_rules.tsv")

    def apply(self, text, tokens):
        return STANDARD.postprocess_tokens(
            text,
            tokens,
            self.matcher,
            self.split_patterns,
            poem_id="测试",
        ).tokens

    def test_required_terms_are_isolated_and_merged(self):
        text = "彩凤肃来仪，欢乐难再逢，芳辰良可惜。千钟圣酒"
        tokens = [
            "彩凤",
            "肃",
            "来",
            "仪",
            "，",
            "欢乐",
            "难",
            "再",
            "逢",
            "，",
            "芳辰",
            "良",
            "可",
            "惜",
            "。",
            "千",
            "钟",
            "圣酒",
        ]
        self.assertEqual(
            self.apply(text, tokens),
            ["彩凤", "肃", "来仪", "，", "欢乐", "难再逢", "，", "芳辰", "良可惜", "。", "千钟", "圣酒"],
        )

    def test_longest_entity_match_does_not_absorb_suffix(self):
        self.assertEqual(self.apply("洞庭湖上", ["洞庭湖上"]), ["洞庭湖", "上"])

    def test_explicit_non_merge_examples_split(self):
        self.assertEqual(
            self.apply("不知山中高楼", ["不知", "山中", "高楼"]),
            ["不", "知", "山", "中", "高", "楼"],
        )

    def test_numeric_rule_respects_protected_terms(self):
        self.assertEqual(
            self.apply("千寻百雉千钟万里", ["千寻", "百雉", "千钟", "万里"]),
            ["千", "寻", "百", "雉", "千钟", "万里"],
        )

    def test_conditional_term_only_changes_surface_boundary(self):
        self.assertEqual(self.apply("归来采薇", ["归来", "采", "薇"]), ["归来", "采薇"])

    def test_reconstruction_is_preserved(self):
        text = "长安城里见明月。"
        result = self.apply(text, ["长安城", "里", "见", "明", "月", "。"])
        self.assertEqual("".join(result), text)
        self.assertIn("长安", result)
        self.assertIn("明月", result)

    def test_build_uses_source_result_text_and_audits_corpus_mismatch(self):
        standard_dir = STANDARD_PATH.parent
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            corpus = temp / "corpus.jsonl"
            source = temp / "source.jsonl"
            output = temp / "output.jsonl"
            metadata = temp / "metadata.json"
            audit = temp / "audit.csv"
            corpus.write_text(
                json.dumps({"id": 1, "text": "连薨", "poet": "测试", "title": "测试"}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            source.write_text(
                json.dumps({"id": 1, "tokens": ["连", "甍"]}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            result = STANDARD.build(
                corpus,
                source,
                output,
                metadata,
                audit,
                standard_dir / "protected_term_seed.tsv",
                standard_dir / "named_entity_seed.tsv",
                standard_dir / "non_merge_rules.tsv",
            )
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("".join(saved["tokens"]), "连甍")
            self.assertEqual(result["corpus_text_mismatch_count"], 1)
            self.assertEqual(result["corpus_text_mismatches"][0]["offset"], 1)


if __name__ == "__main__":
    unittest.main()
