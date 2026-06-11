#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import pytest
import numpy as np
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from verify_result import verify_result as verify_result_v1
from verify import verify_result as verify_result_v2
from gen_golden import gen_golden_data as gen_golden_v1
from gen_data import gen_golden_data as gen_golden_v2


class TestVerifyResultV1:
    def test_verify_result_identical_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.uniform(1, 10, [64, 64]).astype(np.float32)
            output_path = os.path.join(tmpdir, 'output.bin')
            golden_path = os.path.join(tmpdir, 'golden.bin')
            
            data.tofile(output_path)
            data.tofile(golden_path)
            
            result = verify_result_v1(output_path, golden_path)
            assert result is True

    def test_verify_result_small_difference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            golden = np.random.uniform(1, 10, [64, 64]).astype(np.float32)
            output = golden + np.random.uniform(-1e-7, 1e-7, [64, 64]).astype(np.float32)
            
            output_path = os.path.join(tmpdir, 'output.bin')
            golden_path = os.path.join(tmpdir, 'golden.bin')
            
            output.tofile(output_path)
            golden.tofile(golden_path)
            
            result = verify_result_v1(output_path, golden_path)
            assert result is True

    def test_verify_result_large_difference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            golden = np.random.uniform(1, 10, [64, 64]).astype(np.float32)
            output = golden * 2
            
            output_path = os.path.join(tmpdir, 'output.bin')
            golden_path = os.path.join(tmpdir, 'golden.bin')
            
            output.tofile(output_path)
            golden.tofile(golden_path)
            
            result = verify_result_v1(output_path, golden_path)
            assert result is False

    def test_verify_result_nan_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            golden = np.random.uniform(1, 10, [64, 64]).astype(np.float32)
            golden[0, 0] = np.nan
            output = golden.copy()
            
            output_path = os.path.join(tmpdir, 'output.bin')
            golden_path = os.path.join(tmpdir, 'golden.bin')
            
            output.tofile(output_path)
            golden.tofile(golden_path)
            
            result = verify_result_v1(output_path, golden_path)
            assert result is True

    def test_verify_result_different_sizes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = np.random.uniform(1, 10, [64, 64]).astype(np.float32)
            golden = np.random.uniform(1, 10, [32, 32]).astype(np.float32)
            
            output_path = os.path.join(tmpdir, 'output.bin')
            golden_path = os.path.join(tmpdir, 'golden.bin')
            
            output.tofile(output_path)
            golden.tofile(golden_path)
            
            result = verify_result_v1(output_path, golden_path)


class TestVerifyResultV2:
    def test_verify_result_identical_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.uniform(1, 10, [128, 128]).astype(np.float32)
            output_path = os.path.join(tmpdir, 'output.bin')
            golden_path = os.path.join(tmpdir, 'golden.bin')
            
            data.tofile(output_path)
            data.tofile(golden_path)
            
            result = verify_result_v2(output_path, golden_path)
            assert result is True

    def test_verify_result_size_check(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = np.random.uniform(1, 10, [64, 64]).astype(np.float32)
            golden = np.random.uniform(1, 10, [128, 128]).astype(np.float32)
            
            output_path = os.path.join(tmpdir, 'output.bin')
            golden_path = os.path.join(tmpdir, 'golden.bin')
            
            output.tofile(output_path)
            golden.tofile(golden_path)
            
            result = verify_result_v2(output_path, golden_path)
            assert result is False

    def test_verify_result_small_difference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            golden = np.random.uniform(1, 10, [100, 100]).astype(np.float32)
            output = golden + np.random.uniform(-1e-7, 1e-7, [100, 100]).astype(np.float32)
            
            output_path = os.path.join(tmpdir, 'output.bin')
            golden_path = os.path.join(tmpdir, 'golden.bin')
            
            output.tofile(output_path)
            golden.tofile(golden_path)
            
            result = verify_result_v2(output_path, golden_path)
            assert result is True

    def test_verify_result_zero_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            golden = np.random.uniform(1, 10, [64, 64]).astype(np.float32)
            golden[0, 0] = 0.0
            output = golden.copy()
            output[0, 0] = 1e-10
            
            output_path = os.path.join(tmpdir, 'output.bin')
            golden_path = os.path.join(tmpdir, 'golden.bin')
            
            output.tofile(output_path)
            golden.tofile(golden_path)
            
            result = verify_result_v2(output_path, golden_path)
            assert result is True

    def test_verify_result_error_ratio_calculation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            golden = np.ones([100, 100]).astype(np.float32)
            output = golden.copy()
            output[0:10, 0:10] = 100
            
            output_path = os.path.join(tmpdir, 'output.bin')
            golden_path = os.path.join(tmpdir, 'golden.bin')
            
            output.tofile(output_path)
            golden.tofile(golden_path)
            
            result = verify_result_v2(output_path, golden_path)
            assert result is False


class TestGenGoldenV1:
    def test_gen_golden_data_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                gen_golden_v1(64, 128, 32)
                
                assert os.path.exists('input/x1_gm.bin')
                assert os.path.exists('input/x2_gm.bin')
                assert os.path.exists('output/golden.bin')
                
                x1 = np.fromfile('input/x1_gm.bin', dtype=np.float16)
                x2 = np.fromfile('input/x2_gm.bin', dtype=np.float16)
                golden = np.fromfile('output/golden.bin', dtype=np.float32)
                
                assert x1.shape == (64 * 32,)
                assert x2.shape == (32 * 128,)
                assert golden.shape == (64 * 128,)
                
                x1_reshaped = x1.reshape(64, 32)
                x2_reshaped = x2.reshape(32, 128)
                expected_golden = np.matmul(x1_reshaped.astype(np.float32), 
                                           x2_reshaped.astype(np.float32))
                actual_golden = golden.reshape(64, 128)
                
                assert np.allclose(actual_golden, expected_golden)
            finally:
                os.chdir(old_cwd)

    def test_gen_golden_data_large_matrix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                gen_golden_v1(512, 512, 512)
                
                assert os.path.exists('input/x1_gm.bin')
                assert os.path.exists('input/x2_gm.bin')
                assert os.path.exists('output/golden.bin')
                
                x1 = np.fromfile('input/x1_gm.bin', dtype=np.float16)
                assert x1.shape == (512 * 512,)
            finally:
                os.chdir(old_cwd)

    def test_gen_golden_data_small_matrix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                gen_golden_v1(16, 16, 16)
                
                assert os.path.exists('input/x1_gm.bin')
                assert os.path.exists('input/x2_gm.bin')
                assert os.path.exists('output/golden.bin')
                
                golden = np.fromfile('output/golden.bin', dtype=np.float32)
                assert golden.shape == (16 * 16,)
            finally:
                os.chdir(old_cwd)

    def test_gen_golden_data_different_dimensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                gen_golden_v1(1, 512, 128)
                
                x1 = np.fromfile('input/x1_gm.bin', dtype=np.float16).reshape(1, 128)
                x2 = np.fromfile('input/x2_gm.bin', dtype=np.float16).reshape(128, 512)
                golden = np.fromfile('output/golden.bin', dtype=np.float32).reshape(1, 512)
                
                assert x1.shape == (1, 128)
                assert x2.shape == (128, 512)
                assert golden.shape == (1, 512)
            finally:
                os.chdir(old_cwd)


class TestGenGoldenV2:
    def test_gen_golden_data_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                gen_golden_v2()
                
                assert os.path.exists('input/x1_gm.bin')
                assert os.path.exists('input/x2_gm.bin')
                assert os.path.exists('output/golden.bin')
                
                x1 = np.fromfile('input/x1_gm.bin', dtype=np.float16)
                x2 = np.fromfile('input/x2_gm.bin', dtype=np.float16)
                golden = np.fromfile('output/golden.bin', dtype=np.float32)
                
                assert x1.shape == (32 * 32,)
                assert x2.shape == (32 * 32,)
                assert golden.shape == (32 * 32,)
                
                x1_reshaped = x1.reshape(32, 32)
                x2_reshaped = x2.reshape(32, 32)
                expected_golden = np.matmul(x1_reshaped.astype(np.float32), 
                                           x2_reshaped.astype(np.float32))
                actual_golden = golden.reshape(32, 32)
                
                assert np.allclose(actual_golden, expected_golden)
            finally:
                os.chdir(old_cwd)

    def test_gen_golden_data_fixed_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                gen_golden_v2()
                
                golden = np.fromfile('output/golden.bin', dtype=np.float32)
                assert golden.size == 32 * 32
            finally:
                os.chdir(old_cwd)

    def test_gen_golden_data_data_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                gen_golden_v2()
                
                x1 = np.fromfile('input/x1_gm.bin', dtype=np.float16)
                x2 = np.fromfile('input/x2_gm.bin', dtype=np.float16)
                
                assert np.all(x1 >= 1) and np.all(x1 <= 10)
                assert np.all(x2 >= 1) and np.all(x2 <= 10)
            finally:
                os.chdir(old_cwd)


class TestFileOperations:
    def test_directory_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                gen_golden_v1(64, 64, 64)
                
                assert os.path.isdir('input')
                assert os.path.isdir('output')
            finally:
                os.chdir(old_cwd)

    def test_file_permissions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                gen_golden_v1(64, 64, 64)
                
                input_file = Path('input/x1_gm.bin')
                output_file = Path('output/golden.bin')
                
                assert input_file.exists()
                assert output_file.exists()
                assert os.access(input_file, os.R_OK)
                assert os.access(output_file, os.R_OK)
            finally:
                os.chdir(old_cwd)

    def test_binary_file_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                gen_golden_v1(64, 64, 64)
                
                with open('input/x1_gm.bin', 'rb') as f:
                    data = f.read()
                    assert len(data) == 64 * 64 * 2
                
                with open('output/golden.bin', 'rb') as f:
                    data = f.read()
                    assert len(data) == 64 * 64 * 4
            finally:
                os.chdir(old_cwd)


class TestEdgeCases:
    def test_verify_result_empty_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'output.bin')
            golden_path = os.path.join(tmpdir, 'golden.bin')
            
            np.array([], dtype=np.float32).tofile(output_path)
            np.array([], dtype=np.float32).tofile(golden_path)
            
            result = verify_result_v2(output_path, golden_path)
            assert result is True

    def test_verify_result_single_element(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            golden = np.array([5.0], dtype=np.float32)
            output = np.array([5.0], dtype=np.float32)
            
            output_path = os.path.join(tmpdir, 'output.bin')
            golden_path = os.path.join(tmpdir, 'golden.bin')
            
            output.tofile(output_path)
            golden.tofile(golden_path)
            
            result = verify_result_v1(output_path, golden_path)
            assert result is True

    def test_gen_golden_square_matrix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            
            try:
                gen_golden_v1(128, 128, 128)
                
                x1 = np.fromfile('input/x1_gm.bin', dtype=np.float16).reshape(128, 128)
                x2 = np.fromfile('input/x2_gm.bin', dtype=np.float16).reshape(128, 128)
                golden = np.fromfile('output/golden.bin', dtype=np.float32).reshape(128, 128)
                
                assert x1.shape == (128, 128)
                assert x2.shape == (128, 128)
                assert golden.shape == (128, 128)
            finally:
                os.chdir(old_cwd)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])