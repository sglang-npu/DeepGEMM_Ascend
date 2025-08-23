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
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import sys, os

sys.path.append(os.getcwd())
import deep_gemm_cpp

torch.npu.config.allow_internal_format = False


class TestCustomAdd(TestCase):

    def test_mmad_custom_ops(self):
        print("============test api kernel==============")
        length_x = [96, 5952]
        length_y = [5952, 1536]
        length_z = [96, 1536]
        x = torch.rand(length_x, device='cpu', dtype=torch.float16)
        y = torch.rand(length_y, device='cpu', dtype=torch.float16)
        z = torch.empty(length_z, device='cpu', dtype=torch.float16)

        x_npu = x.npu()
        y_npu = y.npu()
        z_npu = z.npu()
        deep_gemm_cpp.run_mmad_custom(x_npu, y_npu, z_npu)
        cpuout = x.float() @ y.float()
        print(f"{cpuout=}")
        print(f"{z_npu.float()=}")

    def test_mmad_cache_ops(self):
        print("============test cache kernel==============")
        length_x = [96, 5952]
        length_y = [5952, 1536]
        length_z = [96, 1536]
        x = torch.rand(length_x, device='cpu', dtype=torch.float16)
        y = torch.rand(length_y, device='cpu', dtype=torch.float16)
        z = torch.empty(length_z, device='cpu', dtype=torch.float16)

        bin_path = os.environ.get("KERNEL_BIN_PATH")
        if not bin_path:
            assert False

        x_npu = x.npu()
        y_npu = y.npu()
        z_npu = z.npu()
        deep_gemm_cpp.run_mmad_cache(x_npu, y_npu, z_npu, bin_path)
        cpuout = x.float() @ y.float()
        print(f"{cpuout=}")
        print(f"{z_npu.float()=}")


if __name__ == "__main__":
    run_tests()
