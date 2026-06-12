"""
Unit tests for padding_calculator.py
"""

import os
import sys
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_get_best_config_path = os.path.join(_project_root, "get_best_config")
if _get_best_config_path not in sys.path:
    sys.path.insert(0, _get_best_config_path)

from padding_calculator import PaddingCalculator, PaddingTag, estimate_bandwidth, calc_padding_tags


class TestPaddingTag(unittest.TestCase):
    """Tests for PaddingTag enum."""

    def test_enum_values(self):
        self.assertEqual(PaddingTag.PADDING_NONE.value, 0)
        self.assertEqual(PaddingTag.PADDING_ND.value, 1)
        self.assertEqual(PaddingTag.PADDING_BLOCK_ND.value, 2)
        self.assertEqual(PaddingTag.PADDING_NZ.value, 3)

    def test_enum_count(self):
        self.assertEqual(len(list(PaddingTag)), 4)

    def test_enum_int_conversion(self):
        self.assertEqual(int(PaddingTag.PADDING_NONE), 0)
        self.assertEqual(int(PaddingTag.PADDING_NZ), 3)


class TestEstimateBandwidth(unittest.TestCase):
    """Tests for estimate_bandwidth function."""

    def test_basic_call(self):
        result = estimate_bandwidth(n_value=16, d_value=128, src_d_value=128)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)

    def test_small_n_value(self):
        result = estimate_bandwidth(n_value=1, d_value=16, src_d_value=16)
        self.assertIsInstance(result, float)

    def test_large_d_value(self):
        result = estimate_bandwidth(n_value=16, d_value=256, src_d_value=256)
        self.assertIsInstance(result, float)

    def test_aligned_src_d_value(self):
        result_256 = estimate_bandwidth(n_value=16, d_value=128, src_d_value=256)
        result_128 = estimate_bandwidth(n_value=16, d_value=128, src_d_value=128)
        result_64 = estimate_bandwidth(n_value=16, d_value=64, src_d_value=64)
        self.assertIsInstance(result_256, float)
        self.assertIsInstance(result_128, float)
        self.assertIsInstance(result_64, float)

    def test_very_large_src_d_value(self):
        result = estimate_bandwidth(n_value=16, d_value=128, src_d_value=65536)
        self.assertIsInstance(result, float)

    def test_zero_n_value(self):
        result = estimate_bandwidth(n_value=0, d_value=128, src_d_value=128)
        self.assertIsInstance(result, float)

    def test_class_method_equivalence(self):
        result_func = estimate_bandwidth(n_value=16, d_value=128, src_d_value=128)
        result_class = PaddingCalculator.estimate_bandwidth(n_value=16, d_value=128, src_d_value=128)
        self.assertEqual(result_func, result_class)


class TestCalcPaddingTags(unittest.TestCase):
    """Tests for calc_padding_tags function."""

    def test_basic_call(self):
        result = calc_padding_tags(
            m=128, n=256, k=512,
            m1=128, n1=256, k1=256,
            layout_tag_a=0, layout_tag_b=0,
            splitk_factor=1, core_num=20
        )
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], PaddingTag)
        self.assertIsInstance(result[1], PaddingTag)
        self.assertIsInstance(result[2], PaddingTag)

    def test_row_major_layouts(self):
        padding_a, padding_b, padding_c = calc_padding_tags(
            m=128, n=256, k=512,
            m1=128, n1=256, k1=256,
            layout_tag_a=0, layout_tag_b=0,
            splitk_factor=1, core_num=20
        )
        self.assertIn(padding_a, [PaddingTag.PADDING_NONE, PaddingTag.PADDING_NZ])
        self.assertIn(padding_b, [PaddingTag.PADDING_NONE, PaddingTag.PADDING_NZ])
        self.assertIn(padding_c, [PaddingTag.PADDING_NONE, PaddingTag.PADDING_ND])

    def test_column_major_layouts(self):
        padding_a, padding_b, padding_c = calc_padding_tags(
            m=128, n=256, k=512,
            m1=128, n1=256, k1=256,
            layout_tag_a=1, layout_tag_b=1,
            splitk_factor=1, core_num=20
        )
        self.assertIsInstance(padding_a, PaddingTag)
        self.assertIsInstance(padding_b, PaddingTag)

    def test_mixed_layouts(self):
        padding_a, padding_b, padding_c = calc_padding_tags(
            m=128, n=256, k=512,
            m1=128, n1=256, k1=256,
            layout_tag_a=0, layout_tag_b=1,
            splitk_factor=1, core_num=20
        )
        self.assertIsInstance(padding_a, PaddingTag)
        self.assertIsInstance(padding_b, PaddingTag)

    def test_small_shape(self):
        padding_a, padding_b, padding_c = calc_padding_tags(
            m=16, n=16, k=16,
            m1=16, n1=16, k1=16,
            layout_tag_a=0, layout_tag_b=0,
            splitk_factor=1, core_num=20
        )
        self.assertEqual(padding_a, PaddingTag.PADDING_NONE)
        self.assertEqual(padding_b, PaddingTag.PADDING_NONE)

    def test_large_shape(self):
        padding_a, padding_b, padding_c = calc_padding_tags(
            m=8192, n=8192, k=8192,
            m1=128, n1=256, k1=256,
            layout_tag_a=0, layout_tag_b=0,
            splitk_factor=1, core_num=20
        )
        self.assertIsInstance(padding_a, PaddingTag)
        self.assertIsInstance(padding_b, PaddingTag)

    def test_splitk_factor(self):
        result_no_splitk = calc_padding_tags(
            m=128, n=256, k=512,
            m1=128, n1=256, k1=256,
            layout_tag_a=0, layout_tag_b=0,
            splitk_factor=1, core_num=20
        )
        result_splitk = calc_padding_tags(
            m=128, n=256, k=512,
            m1=128, n1=256, k1=256,
            layout_tag_a=0, layout_tag_b=0,
            splitk_factor=4, core_num=20
        )
        self.assertEqual(len(result_no_splitk), 3)
        self.assertEqual(len(result_splitk), 3)

    def test_different_core_nums(self):
        result_20 = calc_padding_tags(
            m=128, n=256, k=512,
            m1=128, n1=256, k1=256,
            layout_tag_a=0, layout_tag_b=0,
            splitk_factor=1, core_num=20
        )
        result_32 = calc_padding_tags(
            m=128, n=256, k=512,
            m1=128, n1=256, k1=256,
            layout_tag_a=0, layout_tag_b=0,
            splitk_factor=1, core_num=32
        )
        self.assertEqual(len(result_20), 3)
        self.assertEqual(len(result_32), 3)

    def test_class_method_equivalence(self):
        result_func = calc_padding_tags(
            m=128, n=256, k=512,
            m1=128, n1=256, k1=256,
            layout_tag_a=0, layout_tag_b=0,
            splitk_factor=1, core_num=20
        )
        result_class = PaddingCalculator.calc_padding_tags(
            m=128, n=256, k=512,
            m1=128, n1=256, k1=256,
            layout_tag_a=0, layout_tag_b=0,
            splitk_factor=1, core_num=20
        )
        self.assertEqual(result_func, result_class)

    def test_zero_splitk_factor(self):
        result = calc_padding_tags(
            m=128, n=256, k=512,
            m1=128, n1=256, k1=256,
            layout_tag_a=0, layout_tag_b=0,
            splitk_factor=0, core_num=20
        )
        self.assertEqual(len(result), 3)

    def test_padding_c_for_large_output(self):
        padding_a, padding_b, padding_c = calc_padding_tags(
            m=4096, n=2049, k=1024,
            m1=128, n1=256, k1=256,
            layout_tag_a=0, layout_tag_b=0,
            splitk_factor=1, core_num=20
        )
        self.assertIn(padding_c, [PaddingTag.PADDING_NONE, PaddingTag.PADDING_ND])


if __name__ == "__main__":
    unittest.main()