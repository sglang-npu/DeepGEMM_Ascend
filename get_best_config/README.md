# get_best_config 模块

基于机器学习的矩阵乘法最优 Tiling 配置预测模块。

## 概述

本模块使用训练好的 MLP 模型预测给定矩阵维度 (M, N, K, layout_tag_a, layout_tag_b) 下的最优 tiling 参数 (m1, n1, k1)，以优化矩阵乘法算子的执行性能。

## 模块结构

```
get_best_config/
├── get_best_config.py      # 主模块：TilingPredictor 和 GetBestConfig 类
├── model.py                # MLP 模型定义 (TimePredictMLP)
├── catlass_parameter.py    # Catlass 参数生成器
├── tiling_calculator.py    # Tiling 计算器
├── padding_calculator.py   # Padding 计算器
├── utils/                  # 工具函数
│   ├── common.py
│   └── __init__.py
└── model_A2/               # A2 版本模型权重（需准备）
└── model_A3/               # A3 版本模型权重（需准备）
```

## 主要组件

### GetBestConfig

主入口类，整合三种算子类型的预测器：

- `predictor_small`: SmallMatmul 算子预测器
- `predictor_common`: CommonMatmul 算子预测器
- `predictor_padding`: PaddingCommonMatmul 算子预测器

### TilingPredictor

核心预测类，封装模型加载、特征构建和预测逻辑。

### TimePredictMLP

MLP 神经网络模型，输入特征 (M, N, K, m_tile, n_tile, k_tile)，输出预测执行时间。

## 使用方式

### Python API 调用

```python
from get_best_config import GetBestConfig, parse_args

# 初始化
args = parse_args()
args.model_version = "A2"  # 选择模型版本
config = GetBestConfig(args)

# 预测最优配置
result = config.predict(m=72, n=7392, k=8192, layout_tag_a=0, layout_tag_b=1)

# 返回结果包含：
# - predict_tiling: {"m1": x, "n1": y, "k1": z} 最优 tiling
# - operator_type: 算子类型
# - block_dim: block 维度
# - paddingTagA/B/C: padding 标签
# ...
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-version` | A2 | 模型版本 (A2/A3) |

## 工作流程

1. 根据矩阵维度计算算子类型
2. 生成符合硬件约束的 tiling 参数候选集
3. 使用 MLP 模型预测每个候选的执行时间
4. 根据选择策略选取最优 tiling
5. 回退机制：若模型预测不佳，回退到默认配置

## 依赖

- PyTorch
- NumPy
- scikit-learn (可选，用于 DBSCAN 策略)

## 算子类型

| 类型 | 适用场景 |
|------|----------|
| SmallMatmul | 小规模矩阵，无需 padding |
| CommonMatmul | 通用矩阵乘法 |
| PaddingCommonMatmul | 需要 padding 的矩阵 |
| PaddingMultiCoreSplitkMatmul | 多核 SplitK |
| PaddingStreamkMatmul | StreamK 调度 |