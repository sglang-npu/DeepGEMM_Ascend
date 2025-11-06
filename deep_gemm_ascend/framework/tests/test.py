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
    def test_mmad_rtc_ops(self):
        # return
        print("============test runtime compile kernel==============")
        x1_gm, x2_gm, golden = gen_golden_data()

        # two ways to expend torch tensor
        batch = 1
        x_npu = torch.tensor(x1_gm, device='npu', dtype=torch.bfloat16).unsqueeze(0).repeat(batch, 1, 1)
        y_npu = torch.stack([torch.tensor(x2_gm, device='npu', dtype=torch.bfloat16)] * batch, dim=0)

        length_z = [batch, x_npu.size(1), y_npu.size(2)]
     
        z_npu = torch.zeros(length_z, device='npu', dtype=torch.float32)
        deep_gemm_ascend.run_mmad_rtc(x_npu, y_npu, z_npu)

        bmm_out = torch.zeros(length_z, device='npu', dtype=torch.bfloat16)
        torch.bmm(x_npu, y_npu, out=bmm_out)

        matmul_out = torch.zeros(length_z, device='npu', dtype=torch.bfloat16)
        torch.matmul(x_npu, y_npu, out=matmul_out)

        print("compare znpu to golden")
        verify_result(z_npu.cpu().numpy(), np.concatenate([golden] * batch, axis=0))
        print("compare bmm_out to golden")
        verify_result(bmm_out.to(torch.float32).cpu().numpy(), np.concatenate([golden] * batch, axis=0))
        print("compare matmul_out to golden")
        verify_result(matmul_out.to(torch.float32).cpu().numpy(), np.concatenate([golden] * batch, axis=0))

    def test_mmad_rtc_ops_from_pt(self):
        return
        print("============test runtime compile kernel==============")
        x1_gm = torch.load('./A_tensor.pt', map_location='npu')
        x2_gm = torch.load('./B_tensor.pt', map_location='npu')
        golden = torch.load('./golden_tensor.pt', map_location='npu')

        # 获取shape参数
        batch = x1_gm.size(0)
        x_npu = x1_gm.clone()
        y_npu = x2_gm.clone()

        length_z = [batch, x_npu.size(1), y_npu.size(2)]
     
        z_npu = torch.zeros(length_z, device='npu', dtype=torch.float32)
        deep_gemm_ascend.run_mmad_rtc(x_npu, y_npu, z_npu)

        bmm_out = torch.zeros(length_z, device='npu', dtype=torch.bfloat16)
        torch.bmm(x_npu, y_npu, out=bmm_out)

        matmul_out = torch.zeros(length_z, device='npu', dtype=torch.bfloat16)
        torch.matmul(x_npu, y_npu, out=matmul_out)
        # np.concatenate [golden] * batch 会创建一个包含 batch 个 golden 数组的列表（例如，若 batch=3，则列表为 [golden, golden, golden]）。
        print("compare znpu to golden")
        verify_result(z_npu.cpu().numpy(), np.concatenate([golden] * batch, axis=0))

        print("compare bmm_out to golden")
        verify_result(bmm_out.to(torch.float32).cpu().numpy(), np.concatenate([golden] * batch, axis=0))

        print("compare matmul_out to golden")
        verify_result(matmul_out.to(torch.float32).cpu().numpy(), np.concatenate([golden] * batch, axis=0))

if __name__ == "__main__":
    run_tests()
