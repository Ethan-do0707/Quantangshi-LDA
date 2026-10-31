import importlib.util
import sys
import unittest
from pathlib import Path


PIPELINE_PATH = Path(__file__).resolve().parents[1] / "pipeline.py"
SPEC = importlib.util.spec_from_file_location("segmentation_pipeline", PIPELINE_PATH)
PIPELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)


class PipelineTests(unittest.TestCase):
    def test_mysql_values_support_escapes(self):
        source = "(1,97,'太白\\'诗\\n二行','题'),(2,97,NULL,'次题');"
        rows = list(PIPELINE.iter_mysql_values(source))
        self.assertEqual(rows[0], [1, 97, "太白'诗\n二行", "题"])
        self.assertEqual(rows[1], [2, 97, None, "次题"])

    def test_boundaries_ignore_punctuation(self):
        text = "床前明月光，疑是地上霜。"
        tokens = ["床前", "明月", "光", "，", "疑是", "地上", "霜", "。"]
        self.assertEqual(PIPELINE.token_boundaries(text, tokens), {2, 4, 8, 10})

    def test_siku_tag_decoding(self):
        chars = "床前明月光"
        tags = ["B", "E", "B", "E", "S"]
        self.assertEqual(PIPELINE.decode_siku_tags(chars, tags), ["床前", "明月", "光"])

    def test_token_reconstruction(self):
        PIPELINE.validate_tokens("白日依山尽", ["白日", "依", "山", "尽"], 1, "test")
        with self.assertRaises(ValueError):
            PIPELINE.validate_tokens("白日依山尽", ["白日", "山", "尽"], 1, "test")


if __name__ == "__main__":
    unittest.main()
