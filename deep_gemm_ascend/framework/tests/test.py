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

def gen_golden_data():
    M = 96 
    N = 1536
    K = 5952

    rng = np.random.default_rng()

    def heavy_tail(shape):
        v = rng.lognormal(mean=1.0, sigma=1.2, size=shape)
        return np.clip(v, 1, 10).astype(np.float16)
    
    x1_gm = heavy_tail([M, K])
    x2_gm = heavy_tail([K, N])

    golden = (npu.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.float32)
    return x1_gm, x2_gm, golden 

def verify_result(output, golden):
    output = output.reshape(-1)
    print(f"{output=}")
    printf(f"{output.shape=}")

    golden = golden.reshape(-1)
    diff_ele_result = np.isclose(output,
                                 golden, 
                                 rtol=relative_tol,
                                 atol=absolute_tol,
                                 equal_nan=True)
    diff_ele_idxs = np.where(diff_ele_result == False)[0]
    for idx in range(len(diff_ele_idxs)):
        real_idx = diff_ele_idxs[idx]
        golden_data = golden[real_idx]
        output_data = output[real_idx]
        print(
            "data index: %06d, excepted: %-.9f， actual: %-.9f，rdiff: %-.6f" % (real_idx, golden_data, output_data, abs(output_data - golden_data) / golden_data)
        )

        if idx == 100:
            break
    
    error_ratio = float(diff_ele_idxs.size) / golden.size 
    print("error ratio: %.4f， tolerance： %.4f" % (error_ratio, error_tol))
    return error_ratio <= error_tol

class TestCustomAdd(TestCase):

    def test_mmad_custom_ops(self):
        # return
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

    def test_mmad_rtc_ops_2(self):
        print("============test runtime compile kernel again==============")
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
