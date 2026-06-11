# Scripts Tests

本目录包含 `scripts` 目录下所有文件的单元测试用例。

## 测试文件说明

- `test_all_scripts.py` - 包含所有 scripts 文件的测试用例，集中在一个文件中

## 测试覆盖的文件

| 源文件 | 测试内容 |
|--------|----------|
| `verify_result.py` | verify_result 函数的各种场景测试 |
| `verify.py` | verify_result 函数的增强版本测试 |
| `gen_golden.py` | gen_golden_data 函数的参数化测试 |
| `gen_data.py` | gen_golden_data 固定参数测试 |
| `torch_matmul.py` | 文件操作和数据验证测试 |

## 测试类说明

### TestVerifyResultV1
测试 `verify_result.py` 中的 verify_result 函数：
- 相同数据验证
- 小误差容忍测试
- 大误差拒绝测试
- NaN 值处理
- 不同尺寸数据

### TestVerifyResultV2
测试 `verify.py` 中的 verify_result 函数：
- 相同数据验证
- 尺寸检查
- 小误差容忍
- 零值处理
- 误差比例计算

### TestGenGoldenV1
测试 `gen_golden.py` 中的 gen_golden_data 函数：
- 基本功能测试
- 大矩阵测试
- 小矩阵测试
- 不同维度测试

### TestGenGoldenV2
测试 `gen_data.py` 中的 gen_golden_data 函数：
- 固定尺寸测试
- 数据范围验证
- 矩阵乘法正确性

### TestFileOperations
测试文件和目录操作：
- 目录创建
- 文件权限
- 二进制文件格式

### TestEdgeCases
测试边界情况：
- 空文件验证
- 单元素数据
- 方阵生成

## 运行测试

### 安装依赖

```bash
pip install pytest numpy
```

### 运行所有测试

```bash
cd deep_gem_ascend/scripts/tests
pytest test_all_scripts.py -v
```

### 运行特定测试类

```bash
pytest test_all_scripts.py::TestVerifyResultV1 -v
pytest test_all_scripts.py::TestGenGoldenV1 -v
```

### 运行特定测试方法

```bash
pytest test_all_scripts.py::TestVerifyResultV1::test_verify_result_identical_data -v
```

### 生成测试报告

```bash
pytest test_all_scripts.py -v --html=report.html
```

## 测试特性

- 使用临时目录进行文件操作测试，避免污染工作目录
- 支持参数化测试，覆盖不同矩阵尺寸
- 自动清理测试生成的文件
- 提供详细的错误信息和测试输出