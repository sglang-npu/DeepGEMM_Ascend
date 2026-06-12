"""
Unit tests for tiling_calculator.py
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "get_best_config"))

from tiling_calculator import (
    MatmulTilingCalculator,
    PlatformInfo,
    LayoutTag,
    OperatorType,
)


class TestPlatformInfo(unittest.TestCase):
    """Tests for PlatformInfo class."""

    def test_default_values(self):
        platform = PlatformInfo()
        self.assertEqual(platform.coreNum, 20)
        self.assertEqual(platform.ubSize, 192 * 1024)
        self.assertEqual(platform.l1Size, 512 * 1024)
        self.assertEqual(platform.l0ASize, 64 * 1024)
        self.assertEqual(platform.l0BSize, 64 * 1024)
        self.assertEqual(platform.l0CSize, 128 * 1024)

    def test_custom_values(self):
        platform = PlatformInfo()
        platform.coreNum = 32
        platform.ubSize = 256 * 1024
        self.assertEqual(platform.coreNum, 32)
        self.assertEqual(platform.ubSize, 256 * 1024)


class TestLayoutTag(unittest.TestCase):
    """Tests for LayoutTag enum."""

    def test_enum_values(self):
        self.assertEqual(LayoutTag.TagRowMajor.value, 0)
        self.assertEqual(LayoutTag.TagColumnMajor.value, 1)

    def test_int_conversion(self):
        self.assertEqual(int(LayoutTag.TagRowMajor), 0)
        self.assertEqual(int(LayoutTag.TagColumnMajor), 1)


class TestOperatorType(unittest.TestCase):
    """Tests for OperatorType class."""

    def test_operator_types(self):
        self.assertEqual(OperatorType.SMALL_MATMUL, "SmallMatmul")
        self.assertEqual(OperatorType.COMMON_MATMUL, "CommonMatmul")
        self.assertEqual(OperatorType.PADDING_COMMON_MATMUL, "PaddingCommonMatmul")
        self.assertEqual(OperatorType.PADDING_MULTICORE_SPLITK_MATMUL, "PaddingMultiCoreSplitkMatmul")
        self.assertEqual(OperatorType.PADDING_STREAMK_MATMUL, "PaddingStreamkMatmul")


class TestMatmulTilingCalculator(unittest.TestCase):
    """Tests for MatmulTilingCalculator class."""

    def setUp(self):
        self.calculator = MatmulTilingCalculator()

    def test_init_default_platform(self):
        calc = MatmulTilingCalculator()
        self.assertEqual(calc.platform.coreNum, 20)

    def test_init_custom_platform(self):
        custom_platform = PlatformInfo()
        custom_platform.coreNum = 32
        calc = MatmulTilingCalculator(platform_info=custom_platform)
        self.assertEqual(calc.platform.coreNum, 32)

    def test_layout_map(self):
        self.assertEqual(self.calculator.layout_map["RowMajor"], LayoutTag.TagRowMajor)
        self.assertEqual(self.calculator.layout_map["ColumnMajor"], LayoutTag.TagColumnMajor)
        self.assertEqual(self.calculator.layout_map["row"], LayoutTag.TagRowMajor)
        self.assertEqual(self.calculator.layout_map["col"], LayoutTag.TagColumnMajor)
        self.assertEqual(self.calculator.layout_map["0"], LayoutTag.TagRowMajor)
        self.assertEqual(self.calculator.layout_map["1"], LayoutTag.TagColumnMajor)

    def test_judge_space_true(self):
        result = self.calculator.judge_space(m1=128, n1=128, k1=256, dtype_size=2)
        self.assertTrue(result)

    def test_judge_space_false_large_values(self):
        result = self.calculator.judge_space(m1=512, n1=512, k1=512, dtype_size=2)
        self.assertFalse(result)

    def test_judge_space_boundary(self):
        result = self.calculator.judge_space(m1=128, n1=256, k1=256, dtype_size=2)
        self.assertTrue(result)

    def test_get_max_k1(self):
        k1 = self.calculator.get_max_k1(m1=128, n1=128, dtype_size=2)
        self.assertIn(k1, [1024, 512, 256, 128])

    def test_get_max_k1_small_tiles(self):
        k1 = self.calculator.get_max_k1(m1=16, n1=16, dtype_size=2)
        self.assertGreater(k1, 0)

    def test_calculate_tiling_layout00(self):
        m1, n1, k1 = self.calculator.calculate_tiling(128, 256, 512, 0, 0)
        self.assertGreater(m1, 0)
        self.assertGreater(n1, 0)
        self.assertGreater(k1, 0)
        self.assertEqual(m1 % 16, 0)
        self.assertEqual(n1 % 16, 0)

    def test_calculate_tiling_layout01(self):
        m1, n1, k1 = self.calculator.calculate_tiling(128, 256, 512, 0, 1)
        self.assertGreater(m1, 0)
        self.assertGreater(n1, 0)
        self.assertGreater(k1, 0)
        self.assertEqual(m1 % 16, 0)
        self.assertEqual(n1 % 16, 0)

    def test_calculate_tiling_layout10(self):
        m1, n1, k1 = self.calculator.calculate_tiling(128, 256, 512, 1, 0)
        self.assertGreater(m1, 0)
        self.assertGreater(n1, 0)
        self.assertGreater(k1, 0)
        self.assertEqual(m1 % 16, 0)
        self.assertEqual(n1 % 16, 0)

    def test_calculate_tiling_layout11(self):
        m1, n1, k1 = self.calculator.calculate_tiling(128, 256, 512, 1, 1)
        self.assertGreater(m1, 0)
        self.assertGreater(n1, 0)
        self.assertGreater(k1, 0)
        self.assertEqual(m1 % 16, 0)
        self.assertEqual(n1 % 16, 0)

    def test_calculate_tiling_invalid_layout(self):
        with self.assertRaises(ValueError):
            self.calculator.calculate_tiling(128, 256, 512, 2, 0)

    def test_calculate_basic(self):
        result = self.calculator.calculate(m=8192, n=8192, k=8192)
        self.assertIn("tiling", result)
        self.assertIn("operator_type", result)
        self.assertIn("shape", result)
        self.assertIn("layout", result)
        self.assertIn("blockDim", result)
        self.assertIn("splitkFactor", result)
        self.assertEqual(result["shape"], (8192, 8192, 8192))

    def test_calculate_with_layouts(self):
        result = self.calculator.calculate(m=128, n=256, k=512, layout_tag_a=0, layout_tag_b=1)
        self.assertEqual(result["layout"], (0, 1))

    def test_calculate_str(self):
        result = self.calculator.calculate_str(m=128, n=256, k=512, layout_a="RowMajor", layout_b="ColumnMajor")
        self.assertEqual(result["layout"], ("RowMajor", "ColumnMajor"))

    def test_calculate_str_invalid_layout(self):
        result = self.calculator.calculate_str(m=128, n=256, k=512, layout_a="invalid", layout_b="RowMajor")
        self.assertEqual(result["layout"][0], "invalid")

    def test_balance_workload(self):
        m1, n1 = self.calculator.balance_workload(m=8192, n=8192, m1=128, n1=256, threshold=32)
        self.assertGreater(m1, 0)
        self.assertGreater(n1, 0)
        self.assertEqual(m1 % 16, 0)
        self.assertEqual(n1 % 16, 0)

    def test_balance_workload_small_matrix(self):
        m1, n1 = self.calculator.balance_workload(m=64, n=128, m1=128, n1=256, threshold=32)
        self.assertGreater(m1, 0)
        self.assertGreater(n1, 0)

    def test_get_kernel_serial(self):
        self.assertEqual(self.calculator.get_kernel_serial(OperatorType.SMALL_MATMUL), 1)
        self.assertEqual(self.calculator.get_kernel_serial(OperatorType.PADDING_COMMON_MATMUL), 2)
        self.assertEqual(self.calculator.get_kernel_serial(OperatorType.COMMON_MATMUL), 0)

    def test_get_dispatch_policy_tag(self):
        self.assertEqual(self.calculator.get_dispatch_policy_tag(OperatorType.SMALL_MATMUL), "DynamicSmall")
        self.assertEqual(self.calculator.get_dispatch_policy_tag(OperatorType.COMMON_MATMUL), "DynamicCommon")
        self.assertEqual(self.calculator.get_dispatch_policy_tag(OperatorType.PADDING_STREAMK_MATMUL), "DynamicStreamk")

    def test_set_platform_info(self):
        new_platform = PlatformInfo()
        new_platform.coreNum = 40
        self.calculator.set_platform_info(new_platform)
        self.assertEqual(self.calculator.platform.coreNum, 40)

    def test_get_platform_info(self):
        platform = self.calculator.get_platform_info()
        self.assertEqual(platform.coreNum, 20)

    def test_select_kernel_type_returns_correct_types(self):
        result = self.calculator.calculate(m=128, n=128, k=128)
        valid_types = [
            OperatorType.SMALL_MATMUL,
            OperatorType.COMMON_MATMUL,
            OperatorType.PADDING_COMMON_MATMUL,
            OperatorType.PADDING_STREAMK_MATMUL,
            OperatorType.PADDING_MULTICORE_SPLITK_MATMUL,
        ]
        self.assertIn(result["operator_type"], valid_types)

    def test_small_shape_small_matmul(self):
        result = self.calculator.calculate(m=128, n=128, k=128)
        self.assertEqual(result["operator_type"], OperatorType.SMALL_MATMUL)

    def test_large_shape(self):
        result = self.calculator.calculate(m=8192, n=8192, k=8192)
        self.assertIn(result["operator_type"], [
            OperatorType.COMMON_MATMUL,
            OperatorType.PADDING_COMMON_MATMUL,
            OperatorType.PADDING_STREAMK_MATMUL,
        ])

    def test_single_dimension(self):
        result_m1 = self.calculator.calculate(m=1, n=128, k=128)
        result_n1 = self.calculator.calculate(m=128, n=1, k=128)
        self.assertIn("operator_type", result_m1)
        self.assertIn("operator_type", result_n1)

    def test_calculate_block_dim(self):
        block_dim = self.calculator.calculate_block_dim(
            m=128, n=256, k=512,
            m1=128, n1=256, k1=256,
            layout_tag_a=0, layout_tag_b=0,
            padding_tag_a=0, padding_tag_b=0,
            splitk_factor=1
        )
        self.assertGreater(block_dim, 0)
        self.assertLessEqual(block_dim, self.calculator.platform.coreNum)

    def test_calculate_block_dim_with_padding(self):
        block_dim = self.calculator.calculate_block_dim(
            m=512, n=1024, k=2048,
            m1=128, n1=256, k1=256,
            layout_tag_a=0, layout_tag_b=0,
            padding_tag_a=3, padding_tag_b=3,
            splitk_factor=1
        )
        self.assertGreater(block_dim, 0)


if __name__ == "__main__":
    unittest.main()