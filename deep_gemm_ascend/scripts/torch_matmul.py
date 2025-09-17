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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    args = parser.parse_args()

    # 1. NumPy 读二进制
    x1_np = np.fromfile("input/x1_gm.bin", dtype=np.float16).reshape(args.M, args.K)
    x2_np = np.fromfile("input/x2_gm.bin", dtype=np.float16).reshape(args.K, args.N)

    # 2. 转到 NPU（float32 精度）
    device = torch.device("npu" if torch.npu.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: NPU 不可用，将在 CPU 上运行！")

    x1 = torch.from_numpy(x1_np.astype(np.float16)).to(device)
    x2 = torch.from_numpy(x2_np.astype(np.float16)).to(device)

    # 3. NPU 上做矩阵乘
    out = torch.matmul(x1, x2)

    # 4. 拷回 CPU 并写文件
    out_cpu = out.cpu().numpy().astype(np.float32)
    out_cpu.tofile("output/torch_out.bin")
    print("NPU 计算完成，结果已写回 output/torch_out.bin")

if __name__ == "__main__":
    main()