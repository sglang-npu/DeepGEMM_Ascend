"""
Run all unit tests for get_best_config module.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py -v           # Run with verbose output
    python run_tests.py --module utils  # Run specific module tests
"""

import argparse
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "get_best_config"))

TEST_MODULES = {
    "utils": "test_utils_common",
    "padding": "test_padding_calculator",
    "tiling": "test_tiling_calculator",
    "catlass": "test_catlass_parameter",
    "model": "test_model",
    "predictor": "test_get_best_config",
    "all": None,
}


def run_tests(module="all", verbose=False):
    """
    Run unit tests.

    Args:
        module: Which module to test (utils, padding, tiling, catlass, model, predictor, all)
        verbose: Whether to show verbose output
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_dir = os.path.dirname(os.path.abspath(__file__))

    if module == "all":
        suite.addTests(loader.discover(test_dir, pattern="test_*.py"))
    else:
        test_file = TEST_MODULES.get(module)
        if test_file is None:
            print(f"Unknown module: {module}")
            print(f"Available modules: {list(TEST_MODULES.keys())}")
            return False
        test_path = os.path.join(test_dir, f"{test_file}.py")
        if not os.path.exists(test_path):
            print(f"Test file not found: {test_path}")
            return False
        suite.addTests(loader.discover(test_dir, pattern=f"{test_file}.py"))

    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)

    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description="Run unit tests for get_best_config")
    parser.add_argument(
        "--module",
        "-m",
        choices=list(TEST_MODULES.keys()),
        default="all",
        help="Which module to test",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output",
    )
    args = parser.parse_args()

    success = run_tests(module=args.module, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()