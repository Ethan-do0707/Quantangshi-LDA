"""大型历史元数据 master XLSX 流式导出规则测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import build_historical_metadata_master_xlsx as exporter


class HistoricalMetadataMasterXlsxTest(unittest.TestCase):
    """验证序列化类型与容差不改变字段语义。"""

    def test_leading_zero_identifier_remains_text(self) -> None:
        """存在前导零的整数型标识不得被 Excel 数值化。"""

        self.assertEqual(exporter.typed_value("poem_id", "00123"), ("text", "00123"))

    def test_known_integer_float_and_boolean_types(self) -> None:
        """明确的整数、概率和布尔字段应写入相应 Excel 类型。"""

        self.assertEqual(exporter.typed_value("author_birth_year", "701"), ("integer", 701))
        self.assertEqual(exporter.typed_value("topic_01_prob", "0.125"), ("float", 0.125))
        self.assertEqual(exporter.typed_value("cross_period_flag", "False"), ("boolean", False))

    def test_missing_value_remains_blank(self) -> None:
        """CSV 空值必须保持为空白单元格。"""

        self.assertEqual(exporter.typed_value("author_death_year", ""), ("blank", None))

    def test_float_comparison_allows_only_serialization_noise(self) -> None:
        """正常 Excel 浮点序列化误差可接受，实质变化必须拒绝。"""

        close = exporter.compare_cell("topic_entropy", "1.234567890123456", "1.23456789012346")
        changed = exporter.compare_cell("topic_entropy", "1.234567890123456", "1.2345")
        self.assertTrue(close[0])
        self.assertFalse(changed[0])

    def test_boolean_validation_rejects_unregistered_values(self) -> None:
        """布尔字段不得把任意文本静默转换为真值。"""

        with self.assertRaises(ValueError):
            exporter.typed_value("period_boundary_flag", "UNKNOWN")

    def test_csv_field_limit_is_platform_safe(self) -> None:
        """CSV 字段上限必须兼容当前平台的 C long 范围。"""

        self.assertGreater(exporter.configure_csv_field_size_limit(), 32_767)


if __name__ == "__main__":
    unittest.main()
