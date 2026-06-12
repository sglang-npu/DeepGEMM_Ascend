"""
Unit tests for model.py
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "get_best_config"))

import torch
from model import TimePredictMLP


class TestTimePredictMLPInit(unittest.TestCase):
    """Tests for TimePredictMLP initialization."""

    def test_init_default_params(self):
        model = TimePredictMLP()
        self.assertEqual(model.layers[0].in_features, 6)

    def test_init_custom_input_dim(self):
        model = TimePredictMLP(input_dim=12)
        self.assertEqual(model.layers[0].in_features, 12)

    def test_init_custom_hidden_dims(self):
        hidden_dims = [64, 32, 16]
        model = TimePredictMLP(input_dim=6, hidden_dims=hidden_dims)
        self.assertEqual(model.layers[0].in_features, 6)
        self.assertEqual(model.layers[0].out_features, hidden_dims[0])

    def test_init_single_hidden_layer(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64])
        self.assertIsInstance(model, TimePredictMLP)

    def test_init_multiple_hidden_layers(self):
        hidden_dims = [128, 64, 32, 16]
        model = TimePredictMLP(input_dim=6, hidden_dims=hidden_dims)
        self.assertIsInstance(model, TimePredictMLP)

    def test_init_large_hidden_dims(self):
        hidden_dims = [1024, 512, 256]
        model = TimePredictMLP(input_dim=6, hidden_dims=hidden_dims)
        self.assertIsInstance(model, TimePredictMLP)


class TestTimePredictMLPStructure(unittest.TestCase):
    """Tests for TimePredictMLP network structure."""

    def test_network_has_linear_layers(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        self.assertGreater(len(linear_layers), 0)

    def test_network_has_batchnorm_layers(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        self.assertGreater(len(bn_layers), 0)

    def test_network_has_relu_activation(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        relu_layers = [m for m in model.modules() if isinstance(m, torch.nn.ReLU)]
        self.assertGreater(len(relu_layers), 0)

    def test_output_layer_is_linear(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        last_layer = model.layers[-1]
        self.assertIsInstance(last_layer, torch.nn.Linear)

    def test_output_dimension_is_one(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        last_linear = None
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                last_linear = m
        self.assertEqual(last_linear.out_features, 1)


class TestTimePredictMLPForward(unittest.TestCase):
    """Tests for TimePredictMLP forward pass."""

    def test_forward_single_sample(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        model.eval()
        x = torch.randn(1, 6)
        output = model(x)
        self.assertEqual(output.shape, (1, 1))

    def test_forward_batch(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        x = torch.randn(32, 6)
        output = model(x)
        self.assertEqual(output.shape, (32, 1))

    def test_forward_large_batch(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        x = torch.randn(1024, 6)
        output = model(x)
        self.assertEqual(output.shape, (1024, 1))

    def test_forward_custom_input_dim(self):
        model = TimePredictMLP(input_dim=10, hidden_dims=[64, 32])
        x = torch.randn(16, 10)
        output = model(x)
        self.assertEqual(output.shape, (16, 1))

    def test_forward_output_is_float(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        model.eval()
        x = torch.randn(1, 6)
        output = model(x)
        self.assertEqual(output.dtype, torch.float32)

    def test_forward_positive_time_prediction(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        model.eval()
        x = torch.randn(10, 6)
        output = model(x)
        self.assertIsInstance(output.item() if output.numel() == 1 else output[0].item(), float)


class TestTimePredictMLPWeights(unittest.TestCase):
    """Tests for weight initialization."""

    def test_init_weights_called(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                self.assertIsNotNone(m.weight)
                self.assertIsNotNone(m.bias)

    def test_linear_weights_not_zero(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        for layer in linear_layers:
            self.assertFalse(torch.all(layer.weight == 0))

    def test_bias_initialized_to_zero(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        for layer in linear_layers:
            self.assertTrue(torch.all(layer.bias == 0))


class TestTimePredictMLPEvalMode(unittest.TestCase):
    """Tests for eval mode behavior."""

    def test_eval_mode_changes_batchnorm(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        model.eval()
        bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        for bn in bn_layers:
            self.assertFalse(bn.training)

    def test_eval_mode_consistent_output(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        model.eval()
        x = torch.randn(10, 6)
        output1 = model(x)
        output2 = model(x)
        self.assertTrue(torch.allclose(output1, output2))

    def test_train_mode_different_output_with_same_input(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        model.train()
        x = torch.randn(100, 6)
        output1 = model(x)
        output2 = model(x)
        self.assertTrue(torch.allclose(output1, output2))


class TestTimePredictMLPDevice(unittest.TestCase):
    """Tests for device placement."""

    def test_model_on_cpu(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        self.assertEqual(next(model.parameters()).device.type, "cpu")

    def test_forward_on_cpu(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        x = torch.randn(10, 6)
        output = model(x)
        self.assertEqual(output.device.type, "cpu")


class TestTimePredictMLPGradient(unittest.TestCase):
    """Tests for gradient computation."""

    def test_gradient_enabled_in_train_mode(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        model.train()
        x = torch.randn(10, 6)
        output = model(x)
        loss = output.sum()
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_no_gradient_in_eval_mode_with_no_grad(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        model.eval()
        x = torch.randn(10, 6)
        with torch.no_grad():
            output = model(x)
            self.assertFalse(output.requires_grad)


class TestTimePredictMLPSaveLoad(unittest.TestCase):
    """Tests for model save and load."""

    def test_save_load_state_dict(self):
        model1 = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        state_dict = model1.state_dict()
        model2 = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        model2.load_state_dict(state_dict)
        x = torch.randn(10, 6)
        model1.eval()
        model2.eval()
        output1 = model1(x)
        output2 = model2(x)
        self.assertTrue(torch.allclose(output1, output2))

    def test_state_dict_keys(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64, 32])
        state_dict = model.state_dict()
        self.assertGreater(len(state_dict), 0)
        for key in state_dict:
            self.assertTrue(key.startswith("layers.") or key.startswith("layers."))


class TestTimePredictMLPParameters(unittest.TestCase):
    """Tests for model parameters count."""

    def test_parameter_count(self):
        model = TimePredictMLP(input_dim=6, hidden_dims=[64])
        param_count = sum(p.numel() for p in model.parameters())
        self.assertGreater(param_count, 0)

    def test_larger_hidden_dims_more_params(self):
        model_small = TimePredictMLP(input_dim=6, hidden_dims=[64])
        model_large = TimePredictMLP(input_dim=6, hidden_dims=[128])
        params_small = sum(p.numel() for p in model_small.parameters())
        params_large = sum(p.numel() for p in model_large.parameters())
        self.assertGreater(params_large, params_small)

    def test_more_layers_more_params(self):
        model_1layer = TimePredictMLP(input_dim=6, hidden_dims=[64])
        model_3layers = TimePredictMLP(input_dim=6, hidden_dims=[64, 32, 16])
        params_1 = sum(p.numel() for p in model_1layer.parameters())
        params_3 = sum(p.numel() for p in model_3layers.parameters())
        self.assertGreater(params_3, params_1)


if __name__ == "__main__":
    unittest.main()