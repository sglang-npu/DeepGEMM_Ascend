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
import sys, os, math
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


def align16(x):
    return (x + 15) & ~15


def get_best_config(m, n, k, m_sections, n_sections, m_sec_o_blocks, n_sec_o_blocks, k_o_iter_blocks, db_o_blocks):
    param_dic = {}
    param_dic["m_sections"] = m_sections
    param_dic["n_sections"] = n_sections
    param_dic["m_sec_o_blocks"] = m_sec_o_blocks
    param_dic["n_sec_o_blocks"] = n_sec_o_blocks
    param_dic["k_o_iter_blocks"] = k_o_iter_blocks
    param_dic["db_o_blocks"] = db_o_blocks

    param_dic["m_blocks"] = align16(m) // 16
    param_dic["n_blocks"] = align16(n) // 16
    param_dic["k_blocks"] = align16(k) // 16
    param_dic["m_o_fix"] = align16(m) - m
    param_dic["n_o_fix"] = align16(n) - n
    param_dic["k_o_fix"] = align16(k) - k

    param_dic["db_o_num"] = k_o_iter_blocks // db_o_blocks

    param_dic["r_m_blocks"] = param_dic["m_blocks"] % m_sec_o_blocks
    param_dic["r_n_blocks"] = param_dic["n_blocks"] % n_sec_o_blocks
    if param_dic["r_m_blocks"] == 0:
        param_dic["r_m_blocks"] = m_sec_o_blocks
    if param_dic["r_n_blocks"] == 0:
        param_dic["r_n_blocks"] = n_sec_o_blocks

    param_dic["k_iters"] = math.ceil(param_dic["k_blocks"] / k_o_iter_blocks)

    k_tail_blocks = param_dic["k_blocks"] % k_o_iter_blocks
    if k_tail_blocks == 0:
        param_dic["r_db_num"] = param_dic["db_o_num"]
        param_dic["r_k_blocks"] = db_o_blocks
    else:
        param_dic["r_db_num"] = math.ceil(k_tail_blocks / db_o_blocks)
        param_dic["r_k_blocks"] = k_tail_blocks - ((param_dic["r_db_num"] - 1) * db_o_blocks)

    m_iters = math.ceil(param_dic["m_blocks"] / m_sec_o_blocks)
    n_iters = math.ceil(param_dic["n_blocks"] / n_sec_o_blocks)
    param_dic["m_parts"] = math.ceil(m_iters / param_dic["m_sections"])
    param_dic["n_parts"] = math.ceil(n_iters / param_dic["n_sections"])

    param_dic["m_sc_blocks"] = param_dic["m_parts"] * m_sec_o_blocks
    param_dic["n_sc_blocks"] = param_dic["n_parts"] * n_sec_o_blocks
    param_dic["r_m_parts"] = m_iters - ((m_sections - 1) * param_dic["m_parts"] )
    param_dic["r_n_parts"] = n_iters - ((n_sections - 1) * param_dic["n_parts"] )

    param_dic["batch"] = 1
    param_dic["m"] = m
    param_dic["n"] = n
    param_dic["k"] = k
    print(f"{param_dic=}")


class BenchDeepGemmAscend(TestCase):
    def test_bench_dga(self):
        print("============bench deepgemm for ascend==============")
        x1_gm, x2_gm, golden = gen_golden_data(8, 4096, 7168)
        x_npu = torch.tensor(x1_gm, device='npu')
        y_npu = torch.tensor(x2_gm, device='npu')

        # 28 params
        get_best_config(8, 4096, 7168, 1, 1, 96, 24, 1, 1,)
        # params_list = [1, 4, 2, 128, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        params_list = [1, 1, 96, 24, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        params_npu = torch.tensor(params_list, device='npu', dtype=torch.int32)

        length_z = [x_npu.size(0), y_npu.size(1)]
 
        z_npu = torch.zeros(length_z, device='npu', dtype=torch.float32)
        # default: m_sections=1, n_sections=1, m_sec_o_blocks=3, n_sec_o_blocks=8, k_o_iter_blocks=20, db_o_blocks=10
        try:
            deep_gemm_ascend.run_mmad_bench(x_npu, y_npu, z_npu, params_npu)
        except Exception as e:
            print(f"run error {e}")
        verify_result(z_npu.cpu().numpy(), golden)


if __name__ == "__main__":
    acl.init()
    acl.rt.set_device(0)
    device, _ = acl.rt.get_device()
    print(f"now use {device=}")
    run_tests()
    acl.rt.reset_device(0)
    acl.finalize()
