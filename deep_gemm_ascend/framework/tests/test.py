#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
import numpy as np
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import sys, os
import acl
import deep_gemm_ascend

torch.npu.config.allow_internal_format = False
relative_tol = 2e-4
absolute_tol = 1e-9
error_tol = 1e-4

def gen_golden_data():
    M = 1
    N = 512
    K = 128

    rng = np.random.default_rng()

    def heavy_tail(shape):
        v = rng.lognormal(mean=1.0, sigma=1.2, size=shape)
        return np.clip(v, 1, 10).astype(np.float32)

    x1_gm = heavy_tail([M, K])
    x2_gm = heavy_tail([K, N])

    golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.float32)
    return x1_gm, x2_gm, golden

