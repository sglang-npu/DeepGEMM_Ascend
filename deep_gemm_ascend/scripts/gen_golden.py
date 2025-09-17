#!/usr/bin/env python3
"""
用法:
    python gen_golden.py 64 128 32
"""
import argparse
import numpy as np
import os

def gen_golden_data(M: int, N: int, K: int):
    x1 = np.random.uniform(1, 10, [M, K]).astype(np.float16)
    x2 = np.random.uniform(1, 10, [K, N]).astype(np.float16)

    golden = np.matmul(x1.astype(np.float32),
                       x2.astype(np.float32)).astype(np.float32)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    x1.tofile("input/x1_gm.bin")
    x2.tofile("input/x2_gm.bin")
    golden.tofile("output/golden.bin")
    print(f"已生成 M={M}, N={N}, K={K} 的测试数据")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    args = parser.parse_args()
    gen_golden_data(args.M, args.N, args.K)