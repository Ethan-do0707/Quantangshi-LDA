"""验证作者历史年代元数据增强管线的关键约束。"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "src" / "build_historical_metadata.py"


def load_module():
    """从文件路径加载待测模块。"""

    spec = importlib.util.spec_from_file_location("build_historical_metadata", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载历史元数据构建程序")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HistoricalMetadataTest(unittest.TestCase):
    """检查身份、候选选择、年代语义和哈希不变性。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        cls.scheme = cls.module.PeriodScheme(
            "QTS_PERIOD_V1",
            "operational_chronological_scheme",
            5,
            (
                cls.module.Period("初唐", 618, 712),
                cls.module.Period("盛唐", 713, 765),
                cls.module.Period("中唐", 766, 835),
                cls.module.Period("晚唐", 836, 907),
            ),
        )

    def test_author_normalization_preserves_script_and_digits(self) -> None:
        """规范化不得删除尾随数字或覆盖简繁。"""

        self.assertEqual(self.module.normalize_author_name(" 许坚1\ufeff "), "许坚1")
        self.assertEqual(self.module.classify_author("许坚1"), "TEXTUAL_DUPLICATE")
        self.assertEqual(self.module.classify_author("佚名"), "ANONYMOUS")
        self.assertEqual(self.module.normalize_author_name("劉禹錫"), "劉禹錫")

    def test_year_semantics_do_not_turn_ranges_into_exact_years(self) -> None:
        """前后与约年不得错误标记为 EXACT。"""

        self.assertEqual(self.module.year_semantics(700, None), (700, 700, 700, "EXACT"))
        self.assertEqual(self.module.year_semantics(700, 2), (700, "", "", "APPROXIMATE"))
        self.assertEqual(self.module.year_semantics(700, -1), ("", "", 700, "RANGE"))
        self.assertEqual(self.module.year_semantics(700, 1), ("", 700, "", "RANGE"))

    def test_period_span_and_boundary_are_explicit(self) -> None:
        """跨期与边界标记必须可以独立复算。"""

        span, cross = self.module.period_span_for_range(760, 770, self.scheme)
        self.assertEqual(span, "盛唐|中唐")
        self.assertTrue(cross)
        self.assertTrue(self.module.is_period_boundary(766, self.scheme))
        self.assertFalse(self.module.is_period_boundary(750, self.scheme))

    def test_song_start_endpoint_is_not_tang_five_dynasties_compatible(self) -> None:
        """960年才开始的宋朝不得因端点相接被判为唐五代相容。"""

        config = {
            "compatible_dynasty_window_start": 581,
            "compatible_dynasty_window_end": 960,
        }
        self.assertEqual(
            self.module.dynasty_compatibility(15, 960, 1279, config),
            ("False", "CONFLICT"),
        )
        self.assertEqual(
            self.module.dynasty_compatibility(7, 907, 960, config),
            ("True", "TANG_FIVE_DYNASTIES_CONTEXT"),
        )

    def test_unique_tang_candidate_selected_but_ambiguous_not_selected(self) -> None:
        """唯一唐代候选可选，多位可行候选不得自动选第一条。"""

        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            CREATE TABLE BIOG_MAIN (
                c_personid INTEGER PRIMARY KEY, c_name_chn TEXT, c_birthyear INTEGER,
                c_by_range INTEGER, c_deathyear INTEGER, c_dy_range INTEGER,
                c_fl_earliest_year INTEGER, c_fl_latest_year INTEGER,
                c_index_year INTEGER, c_index_year_type_code TEXT,
                c_index_year_source_id INTEGER, c_notes TEXT, c_dy INTEGER
            );
            CREATE TABLE DYNASTIES (
                c_dy INTEGER PRIMARY KEY, c_dynasty_chn TEXT, c_start INTEGER, c_end INTEGER
            );
            CREATE TABLE ALTNAME_DATA (
                c_personid INTEGER, c_alt_name_chn TEXT, c_alt_name_type_code INTEGER,
                c_source INTEGER, c_pages TEXT, c_notes TEXT
            );
            INSERT INTO DYNASTIES VALUES (6, '唐', 618, 907);
            INSERT INTO BIOG_MAIN VALUES (1, '甲', 700, NULL, 760, NULL, NULL, NULL, 700, '01', NULL, '', 6);
            INSERT INTO BIOG_MAIN VALUES (2, '乙', 710, NULL, 770, NULL, NULL, NULL, 710, '01', NULL, '', 6);
            INSERT INTO BIOG_MAIN VALUES (3, '乙', 720, NULL, 780, NULL, NULL, NULL, 720, '01', NULL, '', 6);
            """
        )
        authors = [
            {"author_id_internal": "A", "author_raw": "甲", "author_normalized": "甲", "author_display_name": "甲", "author_status": "NAMED", "poem_count": 5},
            {"author_id_internal": "B", "author_raw": "乙", "author_normalized": "乙", "author_display_name": "乙", "author_status": "NAMED", "poem_count": 4},
            {"author_id_internal": "C", "author_raw": "佚名", "author_normalized": "佚名", "author_display_name": "佚名", "author_status": "ANONYMOUS", "poem_count": 3},
        ]
        config = {
            "name_variant_limit": 32,
            "compatible_dynasty_window_start": 581,
            "compatible_dynasty_window_end": 960,
            "dated_event_min_year": 550,
            "dated_event_max_year": 1000,
        }
        _, decisions, _, _ = self.module.build_candidates(connection, authors, {}, {}, config)
        self.assertEqual(decisions["A"]["cbdb_match_confidence"], "HIGH")
        self.assertEqual(decisions["A"]["selected"]["cbdb_person_id"], 1)
        self.assertEqual(decisions["B"]["cbdb_match_status"], "AMBIGUOUS")
        self.assertIsNone(decisions["B"]["selected"])
        self.assertEqual(decisions["C"]["cbdb_match_status"], "NOT_ELIGIBLE")
        connection.close()

    def test_topic_checksum_ignores_appended_author_fields(self) -> None:
        """新增作者字段不能改变20维概率校验和。"""

        fields = ["topic_01_prob", "topic_02_prob"]
        original = [{"poem_id": "1", "topic_01_prob": "0.25", "topic_02_prob": "0.75"}]
        enriched = [{**original[0], "author_period_primary": "盛唐"}]
        self.assertEqual(
            self.module.topic_probability_checksum(original, fields),
            self.module.topic_probability_checksum(enriched, fields),
        )

    def test_dictionary_covers_all_primary_new_fields(self) -> None:
        """作者主表与诗级新增字段必须都有数据字典。"""

        fields = {row["field_name"] for row in self.module.build_data_dictionary()}
        self.assertTrue(set(self.module.AUTHOR_MASTER_FIELDS).issubset(fields))
        self.assertTrue(set(self.module.POEM_AUTHOR_FIELDS).issubset(fields))


if __name__ == "__main__":
    unittest.main()
