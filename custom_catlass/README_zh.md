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
./102_dynamic_optimized_matmul <device_id> <run_mode> <checkAcc> <m> <n> <k> <layoutA> <layoutB> <mTile> <nTile> <kTile>
```

### 参数说明

| 参数                        | 说明                                                     |
|---------------------------|--------------------------------------------------------|
| `device_id`               | 设备卡号                                                   |
| `run_mode`                | 运行模式：`0` 为原生 catlass，`1` 为自定义 tiling，`2`为在线预测，`3`为离线缓存 |
| `m`, `n`, `k`             | 运算 shape                                               |
| `layoutA`, `layoutB`      | 运算格式：`0` 为行主序，`1` 为列主序                                 |
| `checkAcc`                | 是否检查算子计算结果：`0` 为否，`1` 为是                               |
| `mTile`, `nTile`, `kTile` | 运算自定义 tiling，仅模式1需要                                    |

### 示例
#### 自定义tiling
```bash
./102_dynamic_optimized_matmul 0 1 1 16 578 2014 0 0 16 64 512
```
#### 在线预测
```bash
export $PYTHONPATH=<get_best_config_path>:$PYTHONPATH
./102_dynamic_optimized_matmul 0 2 1 16 578 2014 0 0
```
#### 离线缓存
要求缓存文件中存在m、n、k、layoutA、layoutB匹配的数据条目，需要手动输入
```bash
export DGA_CACHE_FILE_PATH=<cache_file_path>
./102_dynamic_optimized_matmul 0 3 1 16 578 2014 0 0
```

## 3. 性能采集
利用CANN中自带的工具msprof进行性能数据收集：
```bash
msprof op ./102_dynamic_optimized_matmul 0 1 1 16 578 2014 0 0 16 64 512
```
输出中包含算子名称、算子类型、算子耗时等信息，可自行分析。

## 4. 注意事项

- shape 巨大的情况下，开启checkRes可能导致检查计算结果非常缓慢
- catlass 版本更新可能导致 patch 不可用
- 离线缓存模式自行创建csv文件，表头可以先执行一次程序自动生成
