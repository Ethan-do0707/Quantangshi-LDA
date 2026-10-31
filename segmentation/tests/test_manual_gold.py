import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "manual_gold" / "apply_manual_gold.py"
SPEC = importlib.util.spec_from_file_location("manual_gold_apply", SCRIPT_PATH)
MANUAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MANUAL
SPEC.loader.exec_module(MANUAL)


def annotation(segmentation, truncated=False):
    return MANUAL.ManualAnnotation(
        excel_row=2,
        poet="测试",
        title="测试",
        tokens=tuple(segmentation.split(" / ")),
        truncated=truncated,
        whitespace_removed_count=0,
    )


class ManualGoldTests(unittest.TestCase):
    def test_full_annotation_replaces_all_boundaries(self):
        result = MANUAL.make_overlay(
            1,
            annotation("来仪 / 难再逢"),
            ["来", "仪", "难", "再", "逢"],
        )
        self.assertEqual(result.tokens, ("来仪", "难再逢"))
        self.assertEqual(result.match_mode, "exact_full")

    def test_prefix_annotation_preserves_unannotated_suffix(self):
        result = MANUAL.make_overlay(
            1,
            annotation("彩凤 / 来仪", truncated=True),
            ["彩", "凤", "来", "仪", "清歌", "一曲"],
        )
        self.assertEqual(result.tokens, ("彩凤", "来仪", "清歌", "一曲"))
        self.assertEqual(result.scope, "prefix")

    def test_variant_substitution_projects_boundaries_without_rewriting(self):
        result = MANUAL.make_overlay(
            1,
            annotation("君臣德合 / 鱼水斯同 / 睿图 / 方 / 永 / 周历长隆 / 万国来同"),
            ["君臣", "德合", "鱼水", "斯同", "睿", "图方", "水", "周历", "长隆", "万国", "来同"],
        )
        self.assertEqual(
            result.tokens,
            ("君臣德合", "鱼水斯同", "睿图", "方", "水", "周历长隆", "万国来同"),
        )
        self.assertEqual("".join(result.tokens), "君臣德合鱼水斯同睿图方水周历长隆万国来同")
        self.assertEqual(result.match_mode, "variant_boundary_projection")

    def test_manual_punctuation_inside_token_can_be_projected(self):
        result = MANUAL.make_overlay(
            1,
            annotation("卧骨将军已殁功 / 苦、战 / 沙间 / 卧箭痕 / 戍楼闲上望星文"),
            ["卧骨", "将军", "已", "殁", "功", "苦", "战", "沙", "间", "卧", "箭痕", "戍楼", "闲", "上", "望", "星文"],
        )
        self.assertIn("苦战", result.tokens)
        self.assertIn("沙间", result.tokens)

    def test_boundary_inside_variant_is_rejected(self):
        prefix = "天地玄黄宇宙洪荒日月盈昃辰宿列张" * 2
        suffix = "寒来暑往秋收冬藏闰馀成岁律吕调阳" * 2
        unsafe_annotation = MANUAL.ManualAnnotation(
            excel_row=2,
            poet="测试",
            title="测试",
            tokens=(prefix, "甲", "乙", suffix),
            truncated=False,
            whitespace_removed_count=0,
        )
        with self.assertRaisesRegex(ValueError, "无法安全投射"):
            MANUAL.make_overlay(
                1,
                unsafe_annotation,
                [prefix, "丙丁", suffix],
            )

    def test_build_preserves_unannotated_poem(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            workbook = temp / "gold.xlsx"
            annotations = temp / "annotations.jsonl"
            extraction = temp / "annotations.metadata.json"
            corpus = temp / "corpus.jsonl"
            source = temp / "source.jsonl"
            output = temp / "output.jsonl"
            metadata = temp / "output.metadata.json"
            audit = temp / "output.audit.csv"

            workbook.write_bytes(b"test workbook")
            annotation_record = {
                "excel_row": 2,
                "poet": "甲",
                "title": "诗一",
                "segmentation": "来仪 / 千钟",
            }
            annotations.write_text(
                json.dumps(annotation_record, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            extraction.write_text(
                json.dumps(
                    {
                        "source_workbook_sha256": MANUAL.file_sha256(workbook),
                        "annotation_output_sha256": MANUAL.file_sha256(annotations),
                        "header": MANUAL.EXPECTED_HEADER,
                        "record_count": 1,
                        "sheet": "Sheet1",
                        "range": "A1:C2",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            corpus.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {"id": 1, "poet": "甲", "title": "诗一", "text": "来仪千钟"},
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {"id": 2, "poet": "乙", "title": "诗二", "text": "明月"},
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            source.write_text(
                "\n".join(
                    [
                        json.dumps({"id": 1, "tokens": ["来", "仪", "千", "钟"]}, ensure_ascii=False),
                        json.dumps({"id": 2, "tokens": ["明", "月"]}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = MANUAL.build(
                workbook,
                annotations,
                extraction,
                corpus,
                source,
                output,
                metadata,
                audit,
            )
            saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(saved[0]["tokens"], ["来仪", "千钟"])
            self.assertEqual(saved[1]["tokens"], ["明", "月"])
            self.assertEqual(result["annotated_poem_count"], 1)
            self.assertEqual(result["unannotated_poem_count"], 1)


if __name__ == "__main__":
    unittest.main()
