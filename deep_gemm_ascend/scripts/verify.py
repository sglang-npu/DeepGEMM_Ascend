#!/usr/bin/env python3
"""
用法:
    python verify.py output/torch_out.bin output/golden.bin
"""
import sys
import numpy as np

# 误差阈值
relative_tol = 1e-6
absolute_tol = 1e-9
error_tol = 1e-4

def verify_result(output_path: str, golden_path: str) -> bool:
    output = np.fromfile(output_path, dtype=np.float32).reshape(-1)
    golden = np.fromfile(golden_path, dtype=np.float32).reshape(-1)

    if output.size != golden.size:
        print(f"长度不一致 output={output.size}, golden={golden.size}")
        return False

    close = np.isclose(output, golden, rtol=relative_tol, atol=absolute_tol,
                       equal_nan=True)
    diff_idx = np.where(~close)[0]

    for i, idx in enumerate(diff_idx):
        g, o = golden[idx], output[idx]
        rdiff = abs(o - g) / abs(g) if g != 0 else abs(o)
        print(f"index={idx:06d}  expect={g:-.9f}  actual={o:-.9f}  rdiff={rdiff:-.6f}")
        if i == 99:          # 最多打印前 100 个差异
            break

    err_ratio = diff_idx.size / golden.size
    print(f"error ratio: {err_ratio:.6f}  (tolerance: {error_tol})")
    return err_ratio <= error_tol

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage: python verify.py <output.bin> <golden.bin>")
        sys.exit(1)

    ok = verify_result(sys.argv[1], sys.argv[2])
    if ok:
        print("test pass")
    else:
        print("[ERROR] result error")
        sys.exit(1)