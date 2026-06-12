"""
Unit tests for utils/common.py
"""

import os
import sys
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_get_best_config_path = os.path.join(_project_root, "get_best_config")
if _get_best_config_path not in sys.path:
    sys.path.insert(0, _get_best_config_path)

from utils.common import ceil_div, round_up


class TestCeilDiv(unittest.TestCase):
    """Tests for ceil_div function."""

    def test_basic_positive(self):
        self.assertEqual(ceil_div(10, 3), 4)
        self.assertEqual(ceil_div(9, 3), 3)
        self.assertEqual(ceil_div(8, 3), 3)
        self.assertEqual(ceil_div(7, 3), 3)
        self.assertEqual(ceil_div(6, 3), 2)

    def test_exact_division(self):
        self.assertEqual(ceil_div(100, 10), 10)
        self.assertEqual(ceil_div(64, 8), 8)
        self.assertEqual(ceil_div(256, 16), 16)

    def test_dividend_smaller_than_divisor(self):
        self.assertEqual(ceil_div(1, 10), 1)
        self.assertEqual(ceil_div(5, 100), 1)
        self.assertEqual(ceil_div(15, 20), 1)

    def test_zero_dividend(self):
        self.assertEqual(ceil_div(0, 10), 0)
        self.assertEqual(ceil_div(0, 1), 0)
        self.assertEqual(ceil_div(0, 100), 0)

    def test_large_numbers(self):
        self.assertEqual(ceil_div(1000000, 1000), 1000)
        self.assertEqual(ceil_div(8192, 128), 64)
        self.assertEqual(ceil_div(65536, 256), 256)

    def test_negative_divisor_raises_error(self):
        with self.assertRaises(ValueError):
            ceil_div(10, -1)
        with self.assertRaises(ValueError):
            ceil_div(10, -5)

    def test_zero_divisor_raises_error(self):
        with self.assertRaises(ValueError):
            ceil_div(10, 0)
        with self.assertRaises(ValueError):
            ceil_div(0, 0)

    def test_one_as_divisor(self):
        self.assertEqual(ceil_div(100, 1), 100)
        self.assertEqual(ceil_div(0, 1), 0)
        self.assertEqual(ceil_div(1, 1), 1)


class TestRoundUp(unittest.TestCase):
    """Tests for round_up function."""

    def test_basic_alignment(self):
        self.assertEqual(round_up(10, 16), 16)
        self.assertEqual(round_up(17, 16), 32)
        self.assertEqual(round_up(32, 16), 32)
        self.assertEqual(round_up(33, 16), 48)

    def test_exact_alignment(self):
        self.assertEqual(round_up(16, 16), 16)
        self.assertEqual(round_up(64, 16), 64)
        self.assertEqual(round_up(128, 16), 128)
        self.assertEqual(round_up(256, 32), 256)

    def test_value_smaller_than_alignment(self):
        self.assertEqual(round_up(1, 16), 16)
        self.assertEqual(round_up(5, 16), 16)
        self.assertEqual(round_up(15, 16), 16)

    def test_zero_value(self):
        self.assertEqual(round_up(0, 16), 0)
        self.assertEqual(round_up(0, 1), 0)
        self.assertEqual(round_up(0, 100), 0)

    def test_large_numbers(self):
        self.assertEqual(round_up(8191, 16), 8192)
        self.assertEqual(round_up(65535, 256), 65536)
        self.assertEqual(round_up(1000, 512), 1024)

    def test_different_alignments(self):
        self.assertEqual(round_up(15, 1), 15)
        self.assertEqual(round_up(15, 2), 16)
        self.assertEqual(round_up(15, 4), 16)
        self.assertEqual(round_up(15, 8), 16)
        self.assertEqual(round_up(15, 32), 32)

    def test_negative_alignment_raises_error(self):
        with self.assertRaises(ValueError):
            round_up(10, -1)
        with self.assertRaises(ValueError):
            round_up(10, -16)

    def test_zero_alignment_raises_error(self):
        with self.assertRaises(ValueError):
            round_up(10, 0)
        with self.assertRaises(ValueError):
            round_up(0, 0)


if __name__ == "__main__":
    unittest.main()