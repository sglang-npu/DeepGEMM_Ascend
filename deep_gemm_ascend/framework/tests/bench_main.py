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

def try_parse_args():
    parser = argparse.ArgumentParser(
        usage='%(prog)s --m [num] --n [num] --k [num] \
            --m_sections [num] --n_sections [num] --m_sec_o_blocks [num] \
            --n_sec_o_blocks [num] --k_o_iter_blocks [num] --db_o_blocks [num]'
    )
    parser.add_argument('--m', required=True, type=int)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--m_sections", required=True, type=int)
    parser.add_argument("--n_sections", required=True, type=int)
    parser.add_argument("--m_sec_o_blocks", required=True, type=int)
    parser.add_argument("--n_sec_o_blocks", required=True, type=int)
    parser.add_argument("--k_o_iter_blocks", required=True, type=int)
    parser.add_argument("--db_o_blocks", required=True, type=int)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return parser.parse_args()

def test_bench_dga(args):
    x_npu, y_npu, golden = gen_golden_data(args.m, args.n, args.k)

    # 28 params
    params_list = [
        args.m_sections,
        args.n_sections,
        args.m_sec_o_blocks,
        args.n_sec_o_blocks,
        args.k_o_iter_blocks,
        args.db_o_blocks,
    ]
    params_list.extend([0] * 22)
    params_npu = torch.tensor(params_list, device='npu', dtype=torch.int32)

    length_z = [x_npu.size(0), y_npu.size(1)]

    z_npu = torch.zeros(length_z, device='npu', dtype=torch.float32)
    deep_gemm_ascend.run_mmad_bench(x_npu, y_npu, z_npu, params_npu)
    verify_result(z_npu.cpu().numpy(), golden.cpu().numpy())

if __name__ == "__main__":
    args = try_parse_args()
    test_bench_dga(args)

