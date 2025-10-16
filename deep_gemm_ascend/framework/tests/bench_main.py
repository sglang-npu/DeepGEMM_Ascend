#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

"""
用法：
    python3 bench_main.py \
        --m 96 --n 1536 --k 5952  --m_sections 1 --n_sections 1  --m_sec_o_blocks 3 --n_sec_o_blocks 8 --k_o_iter_blocks 20 --db_o_blocks 10
调试：
    export LAUNCH_KERNEL_PATH=$DGA_ROOT_DIR/deep_gemm_ascend/cache/kernel_m1n1k1_type1/mmad_kernels.o
    msdebug python3
    (msdebug) setting set -- target.run-args "bench_main.py" "--m" 8 "--n" 1536 "--k" 5952 "--m_sections" 1 "--n_sections" 1 "--m_sec_o_blocks" 3 "--n_sec_o_blocks" 8 "--k_o_iter_blocks" 20 "--db_o_blocks" 10
依赖：
    cann ≥ 8.3.RC1
"""
import torch
import numpy as np
import torch_npu
import sys, os, math, argparse
import acl
import deep_gemm_ascend

torch.npu.config.allow_internal_format = False
relative_tol = 1.5e-6
absolute_tol = 1e-9
error_tol = 1e-4

def gen_golden_data(m, n, k):
    # 定义目标范围 [a, b)
    min_num = 1.0  # 最小值
    max_num = 10.0  # 最大值

    a_npu = min_num + (max_num - min_num) * torch.rand((m, k), device='npu', dtype=torch.float16)
    b_npu = min_num + (max_num - min_num) * torch.rand((k, n), device='npu', dtype=torch.float16)

    x_npu = a_npu.to(torch.float32)
    y_npu = b_npu.to(torch.float32)
    golden = torch.matmul(x_npu, y_npu)
    return a_npu, b_npu, golden

def verify_result(output, golden):
    output = output.reshape(-1)
    golden = golden.reshape(-1)

    diff_ele_result = np.isclose(output,
                                 golden,
                                 rtol=relative_tol,
                                 atol=absolute_tol,
                                 equal_nan=True)
    diff_ele_idxs = np.where(diff_ele_result == False)[0]
    for index in range(len(diff_ele_idxs)):
        real_index = diff_ele_idxs[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.8f" %
            (real_index, golden_data, output_data,
             abs(output_data - golden_data) / golden_data))
        if index == 10:
            break

    error_ratio = float(diff_ele_idxs.size) / golden.size
    print("error ratio: %.6f, tolerance: %.4f" % (error_ratio, error_tol))
    return error_ratio <= error_tol
