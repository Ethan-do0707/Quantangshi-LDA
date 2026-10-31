"""验证人工金标准叠加版预处理资源。"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESOURCE_DIR = ROOT / "segmentation" / "manual_gold" / "resources"
BUILDER_PATH = (
    ROOT / "segmentation" / "manual_gold" / "build_preprocessing_resources.py"
)


def load_builder():
    """从文件路径加载构建模块。"""

    spec = importlib.util.spec_from_file_location("build_preprocessing_resources", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载预处理资源构建模块")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_terms(path: Path) -> set[str]:
    """读取一行一词的词表。"""

    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def sha256(path: Path) -> str:
    """计算测试文件哈希。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


class ManualGoldResourceTest(unittest.TestCase):
    """检查五类资源的关键约束。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load_builder()
        cls.metadata = json.loads(
            (RESOURCE_DIR / "resource_metadata.json").read_text(encoding="utf-8")
        )
        cls.rules = json.loads(
            (RESOURCE_DIR / "04_tf_df_rules.json").read_text(encoding="utf-8")
        )

    def test_filter_precedence(self) -> None:
        """保护词必须优先于停用词和低频阈值。"""

        keep, reason = self.builder.token_decision(
            "测试", 1, 1, {"测试"}, {"测试"}, 5, 3
        )
        self.assertTrue(keep)
        self.assertEqual(reason, "保护词豁免")

    def test_stopword_sets_do_not_conflict_with_protection(self) -> None:
        """核心词表应包含于扩展词表，且不与保护词面冲突。"""

        core = read_terms(RESOURCE_DIR / "02_stopwords_core.txt")
        extended = read_terms(RESOURCE_DIR / "02_stopwords_extended.txt")
        with (RESOURCE_DIR / "03_protected_terms.tsv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            protected = {row["surface"] for row in csv.DictReader(handle, delimiter="\t")}
        self.assertLessEqual(core, extended)
        self.assertFalse(extended & protected)

    def test_variants_are_not_global_automatic_replacements(self) -> None:
        """局部异文不得被误设为全局自动替换。"""

        with (RESOURCE_DIR / "01_variant_normalization.tsv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["apply_automatically"] == "否" for row in rows))
        self.assertIn("待版本校勘", {row["status"] for row in rows})

    def test_rules_preserve_poetic_high_df_terms(self) -> None:
        """高 DF 不应触发无条件自动删除。"""

        self.assertIsNone(self.rules["default_profile"]["automatic_max_df"])
        self.assertEqual(self.rules["default_profile"]["min_tf"], 5)
        self.assertEqual(self.rules["default_profile"]["min_df"], 3)
        self.assertEqual(self.rules["input"]["poem_count"], 43030)

    def test_metadata_hashes_match_outputs(self) -> None:
        """元数据中的输出哈希必须与当前文件一致。"""

        for output in self.metadata["outputs"].values():
            path = ROOT / output["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(sha256(path), output["sha256"])

    def test_semantic_mapping_keeps_ambiguity_labels(self) -> None:
        """高歧义帝王称谓必须要求语境确认。"""

        with (RESOURCE_DIR / "05_semantic_mapping.tsv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = {row["surface"]: row for row in csv.DictReader(handle, delimiter="\t")}
        self.assertEqual(rows["帝"]["ambiguity"], "高")
        self.assertEqual(rows["帝"]["recommended_use"], "必须语境确认")
        self.assertEqual(rows["仲尼"]["canonical"], "孔子")


if __name__ == "__main__":
    unittest.main()
