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
relative_tol = 1e-6
absolute_tol = 1e-9
error_tol = 1e-4

def gen_golden_data(m, n, k):
    rng = np.random.default_rng()

    def heavy_tail(shape):
        v = rng.lognormal(mean=1.0, sigma=1.2, size=shape)
        return np.clip(v, 1, 10).astype(np.float16)
    
    x1_gm = heavy_tail([m, k])
    x2_gm = heavy_tail([k, n])

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

class BenchDeepGemmAscend(TestCase):
    def test_bench_dga(self):
        print("============bench deepgemm for ascend==============")
        x1_gm, x2_gm, golden = gen_golden_data(96, 1536, 5952)
        x_npu = torch.tensor(x1_gm, device='npu')
        y_npu = torch.tensor(x2_gm, device='npu')

        # 28 params
        params_list = [1, 1, 3, 8, 20, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        params_npu = torch.tensor(params_list, device='npu', dtype=torch.int32)

        length_z = [x_npu.size(0), y_npu.size(1)]
 
        z_npu = torch.zeros(length_z, device='npu', dtype=torch.float32)
        # default: m_sections=1, n_sections=1, m_sec_o_blocks=3, n_sec_o_blocks=8, k_o_iter_blocks=20, db_o_blocks=10
        deep_gemm_ascend.run_mmad_bench(x_npu, y_npu, z_npu, params_npu)
        verify_result(z_npu.cpu().numpy(), golden)


if __name__ == "__main__":
    acl.init()
    acl.rt.set_device(0)
    device, _ = acl.rt.get_device()
    print(f"now use {device=}")
    run_tests()
    acl.rt.reset_device(0)
    acl.finalize()
