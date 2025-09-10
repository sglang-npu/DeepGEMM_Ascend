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

sys.path.append(os.getcwd())
print('lt cur path: ',os.getcwd)
import deep_gemm_ascend
print('lt get dga module path', deep_gemm_ascend.__file__)

torch.npu.config.allow_internal_format = False
relative_tol = 1e-6
absolute_tol = 1e-9
error_tol = 1e-4

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

    golden = (np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))).astype(np.float32)
    return x1_gm, x2_gm, golden 

def verify_result(output, golden):
    output = output.reshape(-1)
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
            "data index: %06d, excepted: %-.9f， actual: %-.9f，rdiff: %-.6f" % (real_idx, golden_data, output_data, 
            abs(output_data - golden_data) / golden_data)
        )

        if idx == 10:
            break
    
    error_ratio = float(diff_ele_idxs.size) / golden.size 
    print("error ratio: %.4f， tolerance： %.4f" % (error_ratio, error_tol))
    return error_ratio <= error_tol

class TestCustomAdd(TestCase):

    def test_mmad_custom_ops(self):
        # return
        print("============test api kernel==============")
        x1_gm, x2_gm, golden = gen_golden_data()

        x_npu = torch.tensor(x1_gm, device='npu')
        y_npu = torch.tensor(x2_gm, device='npu')
        
        length_z = [96, 1536]
       
        z_npu = torch.empty(length_z, device='npu', dtype=torch.float32)  

        deep_gemm_ascend.run_mmad_custom(x_npu, y_npu, z_npu)
        
        verify_result(z_npu.cpu().numpy(), golden)

    def test_mmad_rtc_ops(self):
        print("============test runtime compile kernel==============")
        x1_gm, x2_gm, golden = gen_golden_data()

        # two ways to expend torch tensor
        batch = 1
        x_npu = torch.tensor(x1_gm, device='npu').unsqueeze(0).repeat(batch, 1, 1)
        y_npu = torch.stack([torch.tensor(x2_gm, device='npu')] * batch, dim=0)
       
        length_z = [96, 1536]
     
        z_npu = torch.empty(length_z, device='npu', dtype=torch.float32)
        deep_gemm_ascend.run_mmad_rtc(x_npu, y_npu, z_npu)
        verify_result(z_npu.cpu().numpy(), golden)

    def test_mmad_rtc_ops_2(self):
        print("============test runtime compile kernel again==============")
        x1_gm, x2_gm, golden = gen_golden_data()

        # two ways to expend torch tensor
        batch = 1
        x_npu = torch.tensor(x1_gm, device='npu').unsqueeze(0).repeat(batch, 1, 1)
        y_npu = torch.stack([torch.tensor(x2_gm, device='npu')] * batch, dim=0)
     
        length_z = [96, 1536]
 
        z_npu = torch.empty(length_z, device='npu', dtype=torch.float32) 
        deep_gemm_ascend.run_mmad_rtc(x_npu, y_npu, z_npu)
        verify_result(z_npu.cpu().numpy(), golden)


if __name__ == "__main__":
    run_tests()
