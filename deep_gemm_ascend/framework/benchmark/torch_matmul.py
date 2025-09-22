#!/usr/bin/env python3
"""
用法：
    python torch_verify_npu.py 128 256 64

依赖：
    torch-npu ≥ 2.0
"""
import argparse
import torch
import torch_npu   # 只需 import 一次即可注册设备
import numpy as np


def gen_data(shape: list):
    device = torch.device("npu" if torch.npu.is_available() else "cpu")
    if device.type == "cpu":
        assert False

    M, N, K = shape

    A = torch.rand((M, K), device=device, dtype=torch.float16)
    B = torch.rand((K, N), device=device, dtype=torch.float16)

    x_npu = A.cpu().to(torch.float32).to(device)
    y_npu = B.cpu().to(torch.float32).to(device)

    return x_npu, y_npu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    args = parser.parse_args()

    # 1. 生成数据
    x_npu, y_npu = gen_data([args.M, args.N, args.K])

    # 2. NPU 上做矩阵乘
    out = torch.matmul(x_npu, y_npu)

    # 3. 
    print("NPU 计算完成")


if __name__ == "__main__":
    main()