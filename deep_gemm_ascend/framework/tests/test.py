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
print('lt cur path: ',os.getcwd)
import deep_gemm_ascend
print('lt get dga module path', deep_gemm_ascend.__file__)

torch.npu.config.allow_internal_format = False


class TestCustomAdd(TestCase):

    def test_mmad_custom_ops(self):
        return
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
        deep_gemm_ascend.run_mmad_custom(x_npu, y_npu, z_npu)
        cpuout = x.float() @ y.float()
        print(f"cpu_out: {cpuout=}")
        print(f"npu_out: {z_npu.float()=}")

    def test_mmad_rtc_ops(self):
        print("============test runtime compile kernel==============")
        length_x = [1, 96, 5952]
        length_y = [1, 5952, 1536]
        length_z = [1, 96, 1536]
        x = torch.rand(length_x, device='cpu', dtype=torch.float16)
        y = torch.rand(length_y, device='cpu', dtype=torch.float16)
        z = torch.empty(length_z, device='cpu', dtype=torch.float16)

        x_npu = x.npu()
        y_npu = y.npu()
        z_npu = z.npu()
        deep_gemm_ascend.run_mmad_rtc(x_npu, y_npu, z_npu)
        cpuout = x.float() @ y.float()
        print(f"cpu_out: {cpuout=}")
        print(f"npu_out: {z_npu.float()=}")


if __name__ == "__main__":
    run_tests()
