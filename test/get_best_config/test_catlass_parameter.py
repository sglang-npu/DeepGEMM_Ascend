"""
Unit tests for catlass_parameter.py
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "get_best_config"))

from catlass_parameter import CatlassParameter
from padding_calculator import PaddingTag


class TestCatlassParameterInit(unittest.TestCase):
    """Tests for CatlassParameter initialization."""

    def test_init_with_all_params(self):
        catlass = CatlassParameter(
            operator_type="SmallMatmul",
            core_num=20,
            layout_tag_a=0,
            layout_tag_b=1
        )
        self.assertEqual(catlass.operator_type, "SmallMatmul")
        self.assertEqual(catlass.core_num, 20)
        self.assertEqual(catlass.layout_tag_a, 0)
        self.assertEqual(catlass.layout_tag_b, 1)

    def test_init_default_core_num(self):
        catlass = CatlassParameter(
            operator_type="CommonMatmul",
            layout_tag_a=0,
            layout_tag_b=0
        )
        self.assertEqual(catlass.core_num, 20)

    def test_init_missing_operator_type_raises(self):
        with self.assertRaises(ValueError) as ctx:
            CatlassParameter(
                operator_type=None,
                layout_tag_a=0,
                layout_tag_b=1
            )
        self.assertIn("operator_type", str(ctx.exception))

    def test_init_missing_layout_tags_raises(self):
        with self.assertRaises(ValueError) as ctx:
            CatlassParameter(
                operator_type="SmallMatmul",
                layout_tag_a=None,
                layout_tag_b=None
            )
        self.assertIn("layout_tag_a", str(ctx.exception))
        self.assertIn("layout_tag_b", str(ctx.exception))

    def test_init_missing_all_raises(self):
        with self.assertRaises(ValueError) as ctx:
            CatlassParameter(
                operator_type=None,
                layout_tag_a=None,
                layout_tag_b=None
            )
        self.assertIn("operator_type", str(ctx.exception))
        self.assertIn("layout_tag_a", str(ctx.exception))
        self.assertIn("layout_tag_b", str(ctx.exception))


class TestCatlassParameterGrid(unittest.TestCase):
    """Tests for grid parameter generation."""

    def setUp(self):
        self.catlass = CatlassParameter(
            operator_type="SmallMatmul",
            core_num=20,
            layout_tag_a=0,
            layout_tag_b=0
        )

    def test_grid_generate_parameters_returns_list(self):
        params = self.catlass.grid_parameters
        self.assertIsInstance(params, list)
        self.assertGreater(len(params), 0)

    def test_grid_parameters_format(self):
        params = self.catlass.grid_parameters
        for param in params:
            self.assertIn("mTile", param)
            self.assertIn("nTile", param)
            self.assertIn("kTile", param)
            self.assertIsInstance(param["mTile"], int)
            self.assertIsInstance(param["nTile"], int)
            self.assertIsInstance(param["kTile"], int)

    def test_grid_parameters_values_in_range(self):
        params = self.catlass.grid_parameters
        for param in params:
            self.assertGreaterEqual(param["mTile"], 1)
            self.assertLessEqual(param["mTile"], 64)
            self.assertGreaterEqual(param["nTile"], 1)
            self.assertLessEqual(param["nTile"], 64)
            self.assertGreaterEqual(param["kTile"], 1)
            self.assertLessEqual(param["kTile"], 128)

    def test_generate_mn_tile_values(self):
        mn_values = self.catlass.generate_mn_tile_values()
        self.assertIsInstance(mn_values, list)
        self.assertIn(1, mn_values)
        self.assertIn(16, mn_values)
        self.assertIn(64, mn_values)

    def test_generate_k_tile_values(self):
        k_values = self.catlass.generate_k_tile_values()
        self.assertIsInstance(k_values, list)
        self.assertIn(1, k_values)
        self.assertIn(128, k_values)

    def test_generate_mnk_tiles_linear_yields_valid_combinations(self):
        count = 0
        for m_tile, n_tile, k_tile in self.catlass.generate_mnk_tiles_linear():
            self.assertGreaterEqual(m_tile, 1)
            self.assertGreaterEqual(n_tile, 1)
            self.assertGreaterEqual(k_tile, 1)
            count += 1
        self.assertGreater(count, 0)


class TestCatlassParameterConstraints(unittest.TestCase):
    """Tests for constraint checking methods."""

    def setUp(self):
        self.catlass = CatlassParameter(
            operator_type="SmallMatmul",
            core_num=20,
            layout_tag_a=0,
            layout_tag_b=0
        )

    def test_check_smallmatmul_constraints_true(self):
        result = self.catlass.check_smallmatmul_constraints(
            m=128, n=128, k=128,
            m1=128, n1=128, k1=256,
            layout_tag_a=0, layout_tag_b=0
        )
        self.assertTrue(result)

    def test_check_smallmatmul_constraints_false_padding(self):
        result = self.catlass.check_smallmatmul_constraints(
            m=17, n=17, k=17,
            m1=128, n1=128, k1=256,
            layout_tag_a=0, layout_tag_b=0
        )
        self.assertIsInstance(result, bool)

    def test_check_smallmatmul_constraints_false_k_exceeds(self):
        result = self.catlass.check_smallmatmul_constraints(
            m=128, n=128, k=512,
            m1=128, n1=128, k1=256,
            layout_tag_a=0, layout_tag_b=0
        )
        self.assertFalse(result)

    def test_check_padding_common_matmul_constraints_true(self):
        result = self.catlass.check_padding_common_matmul_constraints(
            m=17, n=17, k=17,
            m1=128, n1=128, k1=256,
            layout_tag_a=0, layout_tag_b=0
        )
        self.assertIsInstance(result, bool)

    def test_check_padding_common_matmul_constraints_false(self):
        result = self.catlass.check_padding_common_matmul_constraints(
            m=128, n=128, k=128,
            m1=128, n1=128, k1=256,
            layout_tag_a=0, layout_tag_b=0
        )
        self.assertFalse(result)

    def test_check_padding_multicore_splitk_constraints(self):
        result = self.catlass.check_padding_multicore_splitk_constraints(
            m=128, n=256, k=6000,
            m1=128, n1=256, k1=256,
            layout_tag_a=0, layout_tag_b=0
        )
        self.assertIsInstance(result, bool)

    def test_check_padding_streamk_constraints(self):
        result = self.catlass.check_padding_streamk_constraints(
            m=128, n=256, k=4000,
            m1=128, n1=256, k1=256,
            layout_tag_a=0, layout_tag_b=0
        )
        self.assertIsInstance(result, bool)

    def test_check_commonmatmul_constraints(self):
        result = self.catlass.check_commonmatmul_constraints(
            m=128, n=128, k=512,
            m1=128, n1=128, k1=256,
            layout_tag_a=0, layout_tag_b=0
        )
        self.assertIsInstance(result, bool)


class TestCatlassParameterFilter(unittest.TestCase):
    """Tests for filter_parameters method."""

    def setUp(self):
        self.catlass_small = CatlassParameter(
            operator_type="SmallMatmul",
            core_num=20,
            layout_tag_a=0,
            layout_tag_b=0
        )
        self.catlass_common = CatlassParameter(
            operator_type="CommonMatmul",
            core_num=20,
            layout_tag_a=0,
            layout_tag_b=0
        )
        self.catlass_padding = CatlassParameter(
            operator_type="PaddingCommonMatmul",
            core_num=20,
            layout_tag_a=0,
            layout_tag_b=0
        )

    def test_filter_parameters_smallmatmul(self):
        params = self.catlass_small.filter_parameters([128, 128, 128])
        self.assertIsInstance(params, list)
        for param in params:
            self.assertIn("mTile", param)
            self.assertIn("nTile", param)
            self.assertIn("kTile", param)

    def test_filter_parameters_empty_result_warning(self):
        params = self.catlass_small.filter_parameters([1, 1, 100000])
        self.assertIsInstance(params, list)

    def test_filter_parameters_with_explicit_layout(self):
        params = self.catlass_small.filter_parameters(
            [128, 128, 128],
            layout_tag_a=0,
            layout_tag_b=1
        )
        self.assertIsInstance(params, list)

    def test_filter_parameters_uses_instance_layout(self):
        catlass = CatlassParameter(
            operator_type="SmallMatmul",
            core_num=20,
            layout_tag_a=0,
            layout_tag_b=1
        )
        params = catlass.filter_parameters([128, 128, 128])
        self.assertIsInstance(params, list)

    def test_filter_parameters_commonmatmul(self):
        params = self.catlass_common.filter_parameters([8192, 8192, 8192])
        self.assertIsInstance(params, list)

    def test_filter_parameters_padding_common(self):
        params = self.catlass_padding.filter_parameters([100, 100, 100])
        self.assertIsInstance(params, list)

    def test_filter_parameters_unknown_type_returns_all(self):
        catlass = CatlassParameter(
            operator_type="UnknownType",
            core_num=20,
            layout_tag_a=0,
            layout_tag_b=0
        )
        params = catlass.filter_parameters([128, 128, 128])
        self.assertEqual(params, catlass.grid_parameters)


class TestCatlassParameterGetParams(unittest.TestCase):
    """Tests for get_params_with_idx method."""

    def setUp(self):
        self.catlass = CatlassParameter(
            operator_type="SmallMatmul",
            core_num=20,
            layout_tag_a=0,
            layout_tag_b=0
        )

    def test_get_params_with_idx_valid(self):
        params = self.catlass.filter_parameters([128, 128, 128])
        if len(params) > 0:
            param = self.catlass.get_params_with_idx([128, 128, 128], 0)
            self.assertIn("mTile", param)
            self.assertIn("nTile", param)
            self.assertIn("kTile", param)

    def test_get_params_with_idx_out_of_range_raises(self):
        params = self.catlass.filter_parameters([128, 128, 128])
        with self.assertRaises(IndexError):
            self.catlass.get_params_with_idx([128, 128, 128], len(params) + 100)


class TestCatlassParameterOperatorTypes(unittest.TestCase):
    """Tests for different operator types."""

    def test_smallmatmul_operator(self):
        catlass = CatlassParameter(
            operator_type="SmallMatmul",
            core_num=20,
            layout_tag_a=0,
            layout_tag_b=0
        )
        self.assertEqual(catlass.operator_type, "SmallMatmul")

    def test_commonmatmul_operator(self):
        catlass = CatlassParameter(
            operator_type="CommonMatmul",
            core_num=20,
            layout_tag_a=0,
            layout_tag_b=0
        )
        self.assertEqual(catlass.operator_type, "CommonMatmul")

    def test_padding_common_matmul_operator(self):
        catlass = CatlassParameter(
            operator_type="PaddingCommonMatmul",
            core_num=20,
            layout_tag_a=0,
            layout_tag_b=0
        )
        self.assertEqual(catlass.operator_type, "PaddingCommonMatmul")

    def test_padding_multicore_splitk_operator(self):
        catlass = CatlassParameter(
            operator_type="PaddingMultiCoreSplitkMatmul",
            core_num=20,
            layout_tag_a=0,
            layout_tag_b=0
        )
        self.assertEqual(catlass.operator_type, "PaddingMultiCoreSplitkMatmul")

    def test_padding_streamk_operator(self):
        catlass = CatlassParameter(
            operator_type="PaddingStreamkMatmul",
            core_num=20,
            layout_tag_a=0,
            layout_tag_b=0
        )
        self.assertEqual(catlass.operator_type, "PaddingStreamkMatmul")


class TestCatlassParameterLayouts(unittest.TestCase):
    """Tests for different layout combinations."""

    def test_row_major_both(self):
        catlass = CatlassParameter(
            operator_type="SmallMatmul",
            core_num=20,
            layout_tag_a=0,
            layout_tag_b=0
        )
        self.assertEqual(catlass.layout_tag_a, 0)
        self.assertEqual(catlass.layout_tag_b, 0)

    def test_column_major_both(self):
        catlass = CatlassParameter(
            operator_type="SmallMatmul",
            core_num=20,
            layout_tag_a=1,
            layout_tag_b=1
        )
        self.assertEqual(catlass.layout_tag_a, 1)
        self.assertEqual(catlass.layout_tag_b, 1)

    def test_mixed_layouts(self):
        catlass = CatlassParameter(
            operator_type="SmallMatmul",
            core_num=20,
            layout_tag_a=0,
            layout_tag_b=1
        )
        self.assertEqual(catlass.layout_tag_a, 0)
        self.assertEqual(catlass.layout_tag_b, 1)


class TestCatlassParameterCoreNum(unittest.TestCase):
    """Tests for different core_num values."""

    def test_default_core_num(self):
        catlass = CatlassParameter(
            operator_type="SmallMatmul",
            layout_tag_a=0,
            layout_tag_b=0
        )
        self.assertEqual(catlass.core_num, 20)

    def test_custom_core_num(self):
        catlass = CatlassParameter(
            operator_type="SmallMatmul",
            core_num=32,
            layout_tag_a=0,
            layout_tag_b=0
        )
        self.assertEqual(catlass.core_num, 32)

    def test_core_num_affects_constraints(self):
        catlass_20 = CatlassParameter(
            operator_type="SmallMatmul",
            core_num=20,
            layout_tag_a=0,
            layout_tag_b=0
        )
        catlass_32 = CatlassParameter(
            operator_type="SmallMatmul",
            core_num=32,
            layout_tag_a=0,
            layout_tag_b=0
        )
        params_20 = catlass_20.filter_parameters([512, 512, 128])
        params_32 = catlass_32.filter_parameters([512, 512, 128])
        self.assertIsInstance(params_20, list)
        self.assertIsInstance(params_32, list)


if __name__ == "__main__":
    unittest.main()