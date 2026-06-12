"""
Unit tests for get_best_config.py
"""

import unittest
import sys
import os
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "get_best_config"))

import torch
from get_best_config import (
    TilingPredictor,
    parse_args,
    parse_hidden_dims,
    EXTENDED_FEATURES,
    VERSION_CONFIG,
)


class TestParseHiddenDims(unittest.TestCase):
    """Tests for parse_hidden_dims function."""

    def test_single_value(self):
        result = parse_hidden_dims("64")
        self.assertEqual(result, [64])

    def test_multiple_values(self):
        result = parse_hidden_dims("64, 128, 256")
        self.assertEqual(result, [64, 128, 256])

    def test_values_with_spaces(self):
        result = parse_hidden_dims(" 64 , 128 , 256 ")
        self.assertEqual(result, [64, 128, 256])

    def test_empty_string(self):
        result = parse_hidden_dims("")
        self.assertEqual(result, [])

    def test_mixed_spacing(self):
        result = parse_hidden_dims("64,128, 256,512")
        self.assertEqual(result, [64, 128, 256, 512])


class TestVersionConfig(unittest.TestCase):
    """Tests for VERSION_CONFIG."""

    def test_a2_config_exists(self):
        self.assertIn("A2", VERSION_CONFIG)
        self.assertIn("hidden_dims_small", VERSION_CONFIG["A2"])
        self.assertIn("hidden_dims_common", VERSION_CONFIG["A2"])
        self.assertIn("hidden_dims_padding", VERSION_CONFIG["A2"])

    def test_a3_config_exists(self):
        self.assertIn("A3", VERSION_CONFIG)
        self.assertIn("hidden_dims_small", VERSION_CONFIG["A3"])
        self.assertIn("hidden_dims_common", VERSION_CONFIG["A3"])
        self.assertIn("hidden_dims_padding", VERSION_CONFIG["A3"])

    def test_config_values_are_strings(self):
        for version in ["A2", "A3"]:
            for key in ["hidden_dims_small", "hidden_dims_common", "hidden_dims_padding"]:
                self.assertIsInstance(VERSION_CONFIG[version][key], str)


class TestExtendedFeatures(unittest.TestCase):
    """Tests for EXTENDED_FEATURES."""

    def test_extended_features_list(self):
        self.assertEqual(EXTENDED_FEATURES, ["M", "N", "K", "m_tile", "n_tile", "k_tile"])

    def test_extended_features_count(self):
        self.assertEqual(len(EXTENDED_FEATURES), 6)


class TestTilingPredictorInit(unittest.TestCase):
    """Tests for TilingPredictor initialization with mocked models."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "model.pth")
        self.scaler_path = os.path.join(self.temp_dir, "scaler.npz")

        model = torch.nn.Linear(6, 1)
        torch.save(model.state_dict(), self.model_path)

        mean = np.zeros(6, dtype=np.float32)
        std = np.ones(6, dtype=np.float32)
        np.savez(self.scaler_path, mean=mean, std=std)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_init_basic(self):
        predictor = TilingPredictor(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            hidden_dims=[64, 32],
        )
        self.assertIsNotNone(predictor.predictor_ctx)

    def test_init_with_operator_type(self):
        predictor = TilingPredictor(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            hidden_dims=[64, 32],
            operator_type="SmallMatmul",
        )
        self.assertEqual(predictor.operator_type, "SmallMatmul")

    def test_init_with_core_num(self):
        predictor = TilingPredictor(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            hidden_dims=[64, 32],
            core_num=32,
        )
        self.assertEqual(predictor.core_num, 32)

    def test_init_with_selection_method(self):
        predictor = TilingPredictor(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            hidden_dims=[64, 32],
            selection_method="topk_median",
        )
        self.assertEqual(predictor.selection_method, "topk_median")

    def test_init_with_selection_topk(self):
        predictor = TilingPredictor(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            hidden_dims=[64, 32],
            selection_topk=20,
        )
        self.assertEqual(predictor.selection_topk, 20)

    def test_init_with_min_tiling(self):
        predictor = TilingPredictor(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            hidden_dims=[64, 32],
            min_tiling=50,
        )
        self.assertEqual(predictor.min_tiling, 50)

    def test_init_with_time_diff_threshold(self):
        predictor = TilingPredictor(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            hidden_dims=[64, 32],
            time_diff_threshold=0.1,
        )
        self.assertEqual(predictor.time_diff_threshold, 0.1)


class TestTilingPredictorLoadScaler(unittest.TestCase):
    """Tests for load_scaler_arrays method."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.scaler_path = os.path.join(self.temp_dir, "scaler.npz")

        mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        std = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        np.savez(self.scaler_path, mean=mean, std=std)

        self.model_path = os.path.join(self.temp_dir, "model.pth")
        model = torch.nn.Linear(6, 1)
        torch.save(model.state_dict(), self.model_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_scaler_arrays_success(self):
        predictor = TilingPredictor(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            hidden_dims=[64, 32],
        )
        mean, std = predictor.load_scaler_arrays(self.scaler_path)
        self.assertEqual(len(mean), 6)
        self.assertEqual(len(std), 6)

    def test_load_scaler_arrays_wrong_format(self):
        wrong_path = os.path.join(self.temp_dir, "scaler.txt")
        with open(wrong_path, "w") as f:
            f.write("invalid")
        predictor = TilingPredictor(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            hidden_dims=[64, 32],
        )
        with self.assertRaises(ValueError):
            predictor.load_scaler_arrays(wrong_path)

    def test_load_scaler_arrays_missing_fields(self):
        bad_scaler_path = os.path.join(self.temp_dir, "bad_scaler.npz")
        np.savez(bad_scaler_path, other=np.zeros(6))
        predictor = TilingPredictor(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            hidden_dims=[64, 32],
        )
        with self.assertRaises(KeyError):
            predictor.load_scaler_arrays(bad_scaler_path)


class TestTilingPredictorBuildFeatureMatrix(unittest.TestCase):
    """Tests for build_feature_matrix method."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "model.pth")
        self.scaler_path = os.path.join(self.temp_dir, "scaler.npz")

        model = torch.nn.Linear(6, 1)
        torch.save(model.state_dict(), self.model_path)

        mean = np.zeros(6, dtype=np.float32)
        std = np.ones(6, dtype=np.float32)
        np.savez(self.scaler_path, mean=mean, std=std)

        self.predictor = TilingPredictor(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            hidden_dims=[64, 32],
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_build_feature_matrix_single_param(self):
        params = [{"mTile": 8, "nTile": 16, "kTile": 32}]
        features = self.predictor.build_feature_matrix([128, 256, 512], params, 6)
        self.assertEqual(features.shape, (1, 6))

    def test_build_feature_matrix_multiple_params(self):
        params = [
            {"mTile": 8, "nTile": 16, "kTile": 32},
            {"mTile": 16, "nTile": 32, "kTile": 64},
        ]
        features = self.predictor.build_feature_matrix([128, 256, 512], params, 6)
        self.assertEqual(features.shape, (2, 6))

    def test_build_feature_matrix_values(self):
        params = [{"mTile": 8, "nTile": 16, "kTile": 32}]
        features = self.predictor.build_feature_matrix([128, 256, 512], params, 6)
        self.assertEqual(features[0, 0], 128.0)
        self.assertEqual(features[0, 1], 256.0)
        self.assertEqual(features[0, 2], 512.0)

    def test_build_feature_matrix_wrong_dim_raises(self):
        params = [{"mTile": 8, "nTile": 16, "kTile": 32}]
        with self.assertRaises(ValueError):
            self.predictor.build_feature_matrix([128, 256, 512], params, 10)


class TestTilingPredictorSelectTilingStrategy(unittest.TestCase):
    """Tests for select_tiling_strategy method."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "model.pth")
        self.scaler_path = os.path.join(self.temp_dir, "scaler.npz")

        model = torch.nn.Linear(6, 1)
        torch.save(model.state_dict(), self.model_path)

        mean = np.zeros(6, dtype=np.float32)
        std = np.ones(6, dtype=np.float32)
        np.savez(self.scaler_path, mean=mean, std=std)

        self.predictor = TilingPredictor(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            hidden_dims=[64, 32],
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_select_greedy_returns_best(self):
        params = [
            {"mTile": 8, "nTile": 16, "kTile": 32},
            {"mTile": 16, "nTile": 32, "kTile": 64},
        ]
        preds = np.array([100.0, 50.0])
        best_param, best_time = self.predictor.select_tiling_strategy(
            params, preds, method="greedy"
        )
        self.assertEqual(best_param, params[1])
        self.assertEqual(best_time, 50.0)

    def test_select_topk_median(self):
        params = [
            {"mTile": 8, "nTile": 16, "kTile": 32},
            {"mTile": 16, "nTile": 32, "kTile": 64},
            {"mTile": 32, "nTile": 64, "kTile": 128},
        ]
        preds = np.array([100.0, 50.0, 75.0])
        best_param, best_time = self.predictor.select_tiling_strategy(
            params, preds, method="topk_median", topk=3
        )
        self.assertIn(best_param, params)

    def test_select_empty_params_returns_none(self):
        best_param, best_time = self.predictor.select_tiling_strategy([], None)
        self.assertIsNone(best_param)
        self.assertIsNone(best_time)

    def test_select_empty_preds_returns_none(self):
        params = [{"mTile": 8, "nTile": 16, "kTile": 32}]
        best_param, best_time = self.predictor.select_tiling_strategy(params, np.array([]))
        self.assertIsNone(best_param)
        self.assertIsNone(best_time)


class TestTilingPredictorPredictBatch(unittest.TestCase):
    """Tests for predict_batch method."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "model.pth")
        self.scaler_path = os.path.join(self.temp_dir, "scaler.npz")

        mean = np.zeros(6, dtype=np.float32)
        std = np.ones(6, dtype=np.float32)
        np.savez(self.scaler_path, mean=mean, std=std)

        self.predictor = TilingPredictor(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            hidden_dims=[64, 32],
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_predict_batch_single_sample(self):
        features = np.random.randn(1, 6).astype(np.float32)
        preds = self.predictor.predict_batch(self.predictor.predictor_ctx, features)
        self.assertEqual(len(preds), 1)

    def test_predict_batch_multiple_samples(self):
        features = np.random.randn(100, 6).astype(np.float32)
        preds = self.predictor.predict_batch(self.predictor.predictor_ctx, features)
        self.assertEqual(len(preds), 100)

    def test_predict_batch_large_batch(self):
        features = np.random.randn(5000, 6).astype(np.float32)
        preds = self.predictor.predict_batch(
            self.predictor.predictor_ctx, features, max_batch_size=1024
        )
        self.assertEqual(len(preds), 5000)

    def test_predict_batch_empty_raises(self):
        with self.assertRaises(ValueError):
            self.predictor.predict_batch(self.predictor.predictor_ctx, np.zeros((0, 6)))


class TestTilingPredictorDetectDevice(unittest.TestCase):
    """Tests for detect_device method."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "model.pth")
        self.scaler_path = os.path.join(self.temp_dir, "scaler.npz")

        model = torch.nn.Linear(6, 1)
        torch.save(model.state_dict(), self.model_path)

        mean = np.zeros(6, dtype=np.float32)
        std = np.ones(6, dtype=np.float32)
        np.savez(self.scaler_path, mean=mean, std=std)

        self.predictor = TilingPredictor(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            hidden_dims=[64, 32],
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_detect_device_returns_cpu(self):
        device = self.predictor.detect_device()
        self.assertEqual(device.type, "cpu")


class TestTilingPredictorSetCatlassParamGenerator(unittest.TestCase):
    """Tests for set_catlass_param_generator method."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "model.pth")
        self.scaler_path = os.path.join(self.temp_dir, "scaler.npz")

        model = torch.nn.Linear(6, 1)
        torch.save(model.state_dict(), self.model_path)

        mean = np.zeros(6, dtype=np.float32)
        std = np.ones(6, dtype=np.float32)
        np.savez(self.scaler_path, mean=mean, std=std)

        self.predictor = TilingPredictor(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            hidden_dims=[64, 32],
            operator_type="SmallMatmul",
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_set_catlass_param_generator(self):
        self.predictor.set_catlass_param_generator(layout_tag_a=0, layout_tag_b=1)
        self.assertIsNotNone(self.predictor.catlass_param_generator)
        self.assertEqual(self.predictor.catlass_param_generator.layout_tag_a, 0)
        self.assertEqual(self.predictor.catlass_param_generator.layout_tag_b, 1)


class TestParseArgs(unittest.TestCase):
    """Tests for parse_args function."""

    def test_default_args(self):
        with patch("sys.argv", ["script"]):
            args = parse_args()
            self.assertEqual(args.model_version, "A2")
            self.assertEqual(args.selection_method, "greedy")
            self.assertEqual(args.selection_topk, 10)
            self.assertEqual(args.min_tiling, 60)
            self.assertEqual(args.time_diff_threshold, 0.03)

    def test_custom_model_version(self):
        with patch("sys.argv", ["script", "--model-version", "A3"]):
            args = parse_args()
            self.assertEqual(args.model_version, "A3")

    def test_custom_selection_method(self):
        with patch("sys.argv", ["script", "--selection-method", "topk_median"]):
            args = parse_args()
            self.assertEqual(args.selection_method, "topk_median")

    def test_custom_selection_topk(self):
        with patch("sys.argv", ["script", "--selection-topk", "20"]):
            args = parse_args()
            self.assertEqual(args.selection_topk, 20)

    def test_custom_min_tiling(self):
        with patch("sys.argv", ["script", "--min-tiling", "100"]):
            args = parse_args()
            self.assertEqual(args.min_tiling, 100)

    def test_custom_time_diff_threshold(self):
        with patch("sys.argv", ["script", "--time-diff-threshold", "0.05"]):
            args = parse_args()
            self.assertEqual(args.time_diff_threshold, 0.05)


if __name__ == "__main__":
    unittest.main()