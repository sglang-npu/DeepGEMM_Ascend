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
import deep_gemm_ascend

torch.npu.config.allow_internal_format = False


def gen_golden_data(m, n, k):
    # 定义目标范围 [a, b)
    min_num = 1.0  # 最小值
    max_num = 10.0  # 最大值

    a_npu = min_num + (max_num - min_num) * torch.rand((m, k), device='npu', dtype=torch.float16)
    b_npu = min_num + (max_num - min_num) * torch.rand((k, n), device='npu', dtype=torch.float16)

    return a_npu, b_npu


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
    parser.add_argument("--rank_id", required=True, type=int)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return parser.parse_args()


def test_bench_dga(args):
    x_npu, y_npu = gen_golden_data(args.m, args.n, args.k)

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


if __name__ == "__main__":
    args = try_parse_args()
    torch.npu.set_device(args.rank_id)
    test_bench_dga(args)