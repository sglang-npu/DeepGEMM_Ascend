# DeepGEMM benchmark_v2 使用手册

> 多 NPU / 多进程 Catlass GEMM Benchmarks —— 参数生成、批量执行、结果采集与可视化的一站式方案。

---

## 目录

1. [总体介绍](#总体介绍)
2. [目录结构](#目录结构)
3. [核心流程](#核心流程)
4. [快速开始](#快速开始)
5. [命令行参数](#命令行参数)
6. [MsProf 批量执行与异常处理](#msprof-批量执行与异常处理)
7. [结果文件与数据转换](#结果文件与数据转换)
8. [Tiling 参数与算子约束](#tiling-参数与算子约束)
9. [常见问题](#常见问题)

---

## 总体介绍

`benchmark_v2` 是 DeepGEMM 在 Ascend 平台上针对 Catlass GEMM 算子打造的新版基准测试流水线，具备：

- **多进程多 NPU 调度**：轮询分配 shapes，`rank_id` 对应 `npu_ids[rank_id]`
- **批量 MsProf 执行**：同一 shape 的所有 tiling 写入 CSV，批量执行减少进程切换
- **递归异常处理**：异常时自动缩小区间，定位到具体 tiling；无法恢复则标记 `time=-1`
- **断点续跑**：每个 shape / NPU 组合独立 checkpoint，进程异常重启后从中断处继续
- **Shape 筛选与切片**：支持从 Excel 筛选并切片，灵活控制测试范围

适用场景：对大量 shape / 算子进行规模化评测；需要稳定的批量运行、日志与结果产出。

---

## 目录结构

```
benchmark_v2/
├── catlass_benchmark_cli.py          # CLI 入口：参数解析 + Runner 装配
├── distributed_benchmark_runner.py   # 调度器：shape 分发、批量执行、异常处理
├── execution_simulation.md           # 批量 & 异常流程的示例推演
├── launch_benchmark_jobs.sh          # 多 rank 启动脚本
├── catlass_parameter.py              # tiling 采样、约束校验
├── padding_calculator.py             # Padding 策略与带宽估算
├── utils/
│   ├── common.py                     # ceil_div、round_up 等通用工具
│   ├── logger.py                     # 轻量日志封装
│   ├── msprof_executor.py            # MsProf 包装器
│   └── result_parse.py                # MsProf 输出 + CSV 解析
├── file_io.py                        # CheckpointManager / ResultWriter / shape loader
├── models.py                         # `CatlassResult` 等数据类
├── jsonl2excel.py                    # JSON/CSV 转 Excel
├── __init__.py                       # 兼容旧引用
└── README.md                         # 本文
```

---

## 核心流程

### 1. CLI 启动（`catlass_benchmark_cli.py`）
- 解析命令行参数：`rank_id` / `process_num` / `npu_ids` / `operator_type` / `shapes_file` 等
- 从 Excel 文件筛选 shapes（可选：按算子类型、LayoutTag 筛选，支持切片）
- 构造 `GEMMBenchmarkRunner` 并调用 `run_benchmarks()`

### 2. 进程调度（`distributed_benchmark_runner.py`）
- **轮询分配**：`shape[i]` 分配给 `rank_id = i % num_processes` 的进程
- 每个 shape 的处理流程：
  1. 生成/读取 `shape_<M>_<N>_<K>.csv`（包含所有 tiling 参数）
  2. 加载 checkpoint，定位下一条 tiling index
  3. 以跨度 100 分批执行，异常时递归减半（100→50→25→12→6→3→1）
  4. 将结果写入 `shape_<M>_<N>_<K>_npu_<id>.jsonl`

### 3. MsProf 执行（`utils/msprof_executor.py`）
```bash
msprof op --output=<rank_msp_dir> --aic-metrics=PipeUtilization \
          --kernel-name=_Z --launch-count=<span> \
          <catlass_bin> <npu_id> 2 <tiling_csv> <start_idx> <end_idx> 0
```
- `launch-count` 动态等于当前跨度 `span`（最少 1，最多等于本批 tiling 数）
- `timeout` 动态等于 `3 * span` 秒（至少 1s）
- `mode` 固定为 `2`
- 超时会 Kill 进程组并返回空字符串

### 4. 结果解析（`utils/result_parse.py`）
1. 从 MsProf 输出中提取 `Kernel Func Name`
2. 读取 `OpBasicInfo_*.csv`（耗时 / Block / Mix Block）
3. 组合为 `(kernel_name, accuracy, duration, block_dim, mix_block_dim)` 列表
4. 若解析失败或数量不匹配，抛异常触发递归拆分

---

## 快速开始

### 1. 单进程（单 NPU）

```bash
cd DeepGEMM_Ascend/deep_gemm_ascend

python -m framework.benchmark_v2.catlass_benchmark_cli \
  --rank_id 0 \
  --process_num 1 \
  --npu_ids 0 \
  --catlass_bin_path /path/to/catlass/bin \
  --result_dir ./catlass_results \
  --msp_dir ./catlass_msp \
  --operator_type SmallMatmulKernel \
  --core_num 20 \
  --shapes_file ./shapes.xlsx \
  --layout_tag_a 0 \
  --layout_tag_b 0
```

### 2. 多进程 / 多 NPU

```bash
./framework/benchmark_v2/launch_benchmark_jobs.sh \
  framework/benchmark_v2/catlass_benchmark_cli.py 4 0,1,2,3 \
  --operator_type SmallMatmulKernel \
  --core_num 20 \
  --catlass_bin_path /path/to/catlass/bin \
  --result_dir ./catlass_results \
  --msp_dir ./catlass_msp \
  --shapes_file ./shapes.xlsx
```

> `num_runs` (=4) 必须与 `npu_ids` 数量一致。每个 rank 通过轮询分配处理部分 shapes。

### 3. Shape 筛选与切片

```bash
# 从 Excel 筛选并切片（从索引 10 到 20，不包含 20）
python -m framework.benchmark_v2.catlass_benchmark_cli \
  --rank_id 0 --process_num 1 --npu_ids 0 \
  --shapes_file ./shapes.xlsx \
  --operator_type SmallMatmulKernel \
  --layout_tag_a 0 --layout_tag_b 0 \
  --start_idx 10 --end_idx 20 \
  --catlass_bin_path /path/to/bin
```

### 4. 结果转换

```bash
# 单个文件转换
python -m framework.benchmark_v2.jsonl2excel \
  --input ./catlass_results/shape_1279_5003_7681_npu_0.jsonl \
  --output ./reports/shape_1279_5003_7681.xlsx

# 合并多个 NPU 的结果
python -m framework.benchmark_v2.jsonl2excel \
  --input ./catlass_results \
  --shape 1279_5003_7681 \
  --type merge \
  --output ./reports/shape_1279_5003_7681.xlsx
```

---

## 命令行参数

### 必需参数

| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `--rank_id` | 进程的 rank ID，从 0 开始 | `0` |
| `--process_num` | 总进程数 | `4` |
| `--npu_ids` | NPU 设备 ID 列表，逗号分隔，数量必须与 `process_num` 一致 | `0,1,2,3` |

### 可选参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--catlass_bin_path` | `/home/q30063557/code/cutlass/21_dynamic_tiling_matmul` | Catlass 可执行文件路径 |
| `--result_dir` | `./catlass_results` | 结果保存目录 |
| `--msp_dir` | `./catlass_msp` | msprof 输出目录 |
| `--operator_type` | `None` | 算子类型：`SmallMatmulKernel` / `PaddingMatmulKernel` / `PaddingCommonMatmulKernel` |
| `--core_num` | `20` | AI Core 数量 |
| `--shapes_file` | `None` | shapes.xlsx 文件路径，不提供则使用默认 shape_group |
| `--layout_tag_a` | `None` | 从 shapes_file 读取时筛选的 LayoutTagA（默认 0，传入负值代表不过滤） |
| `--layout_tag_b` | `None` | 从 shapes_file 读取时筛选的 LayoutTagB（默认 0，传入负值代表不过滤） |
| `--start_idx` | `None` | 对筛选后的 shapes 进行切片的起始位置（从 0 开始，包含该位置） |
| `--end_idx` | `None` | 对筛选后的 shapes 进行切片的结束位置（不包含该位置，None 表示到末尾） |

> **注意**：`--start_idx` 和 `--end_idx` 必须在筛选完成后使用，系统会检查参数合法性（`start_idx >= 0`、`end_idx <= len(shapes)`、`start_idx < end_idx`）。

---

## MsProf 批量执行与异常处理

> 更详细的推演案例见 `execution_simulation.md`。

### 批量执行策略
- 每批默认 100 条 tiling（左闭右开区间）
- `launch-count = span`（当前批次大小）
- `timeout = 3 × span` 秒（至少 1s）

### 异常检测
- `msprof_executor.process()` 返回空字符串
- `ResultParse` 返回 `None` 或数量不匹配
- 解析过程抛出异常

### 递归处理策略
异常区间自动减半继续尝试：`100 → 50 → 25 → 12 → 6 → 3 → 1`

### 异常标记
当 `span=1` 仍异常时，返回 `("", -1, -1, -1, -1)`；上层写入 `time=-1, diff=inf`。

### 部分成功处理
只要局部区间成功，就保留该部分结果；不会因为单个 tiling 失败而丢弃整个区间。

---

## 结果文件与数据转换

### 输出文件

| 文件 | 说明 |
| --- | --- |
| `result_dir/shape_<M>_<N>_<K>_npu_<id>.jsonl` | 每条记录为 `CatlassResult`：`idx, M, N, K, time, diff, kernel_func_name, parameters, pipe_utilization` |
| `result_dir/tiling_csvs/shape_<M>_<N>_<K>.csv` | MsProf 批量输入，表头 `M, N, K, mTile, nTile, kTile, None` |
| `msp_dir/npu_<id>/OPPROF_xxx` | MsProf 原始采集目录，可用于查看 PipeUtilization 等 |
| Excel 报表 | `jsonl2excel.py` 生成，便于数据分析或分享 |

### 特殊值说明
- `time=-1`：MsProf 失败或无法解析
- `diff=inf`：未做精度校验或失败
- `pipe_utilization`：若 MsProf 生成 PipeUtilization.xlsx / .csv，会解析到该字段

---

## Tiling 参数与算子约束

详见 `catlass_parameter.py` / `padding_calculator.py`：

| 算子类型 | 约束摘要 |
| --- | --- |
| `SmallMatmulKernel` | L0C: `mTile×nTile×1024 ≤ 128KB`<br>L0A: `mTile×kTile×1024 ≤ 64KB`<br>L0B: `nTile×kTile×1024 ≤ 64KB`<br>L1: `(mTile+nTile)×kTile×32 ≤ 256KB`<br>`ceil(m/mTile)×ceil(n/nTile) ≤ core_num`<br>PaddingTagA/B/C 均为 `NONE` |
| `PaddingMatmulKernel` / `PaddingCommonMatmulKernel` | 缓存约束 + 至少一个 PaddingTag 非 `NONE` |

### 参数生成
- 默认生成约 300~500 个有效组合（受约束限制）
- `filter_parameters(shape)` 根据 `operator_type` 自动筛选
- `mTile` / `nTile` / `kTile` 范围：1-64，采样密度优化

---

## 常见问题

1. **导入报错**：如果直接 `python framework/benchmark_v2/catlass_benchmark_cli.py`，可能因相对导入失败；换用 `python -m framework.benchmark_v2.catlass_benchmark_cli` 即可。

2. **NPU 不可见**：无需设置 `ASCEND_RT_VISIBLE_DEVICES`，Runner 会自动把 `npu_ids[rank_id]` 写入 MsProf 命令。

3. **进程异常退出**：`launch_benchmark_jobs.sh` 会根据退出码判断是否重启，已完成的 rank 不会重复执行。

4. **结果缺失**：`time=-1` 即为 MsProf 失败；可单独用 `csv_path` + `start_idx` 重跑该 tiling。

5. **PipeUtilization 未生成**：MsProf 可能未生成 OPPROF 文件夹，可调大 `launch_count` 或确认环境变量。

6. **切片参数错误**：`--start_idx` 和 `--end_idx` 必须在筛选后的 shapes 范围内，系统会自动检查并报错。

---

## 更多参考

- `execution_simulation.md`：完整的批量/异常执行推演
- `distributed_benchmark_runner.py`：批量执行与递归拆分实现
- `jsonl2excel.py`：结果加工脚本

欢迎根据业务需求扩展脚本，并持续在 README 补充实践经验。
