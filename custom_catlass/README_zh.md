# Custom Catlass

本目录可以编译产出以 catlass 的 dynamic matmul 为算子基础得到的自定义验证二进制文件。

## 1. 编译

```bash
bash build_custom_catlass.sh ./changes.patch
```

## 2. 使用

编译日志 `build.log` 中会有 `export LD_LIBRARY_PATH` 的提示，按提示执行。

编译产物在 `./catlass/output/bin` 目录下。

```bash
cd ./catlass/output/bin
./102_dynamic_optimized_matmul <device_id> <run_mode> <m> <n> <k> <layoutA> <layoutB> <checkAcc> <mTile> <nTile> <kTile>
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `device_id` | 设备卡号 |
| `run_mode` | 运行模式：`0` 为原生 catlass，`1` 为自定义 tiling |
| `m`, `n`, `k` | 运算 shape |
| `layoutA`, `layoutB` | 运算格式：`0` 为行主序，`1` 为列主序 |
| `checkAcc` | 是否检查算子计算结果：`0` 为否，`1` 为是 |
| `mTile`, `nTile`, `kTile` | 运算自定义 tiling |

### 示例

```bash
./102_dynamic_optimized_matmul 8 1 16 578 2014 0 0 1 16 64 512
```

## 3. 注意事项

- shape 巨大的情况下，检查计算结果非常缓慢
- catlass 版本更新可能导致 patch 不可用
