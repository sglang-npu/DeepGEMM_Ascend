# Catlass Dynamic Matmul Op Host Tests

本目录包含 `op_host` 下业务代码的单元测试用例。

## 测试文件说明

- `utils_test.cpp` - 测试 utils.cpp 中的工具函数
- `do_tiling_test.cpp` - 测试 do_tiling.cpp 中的分块策略函数
- `select_kernel_test.cpp` - 测试 select_kernel.cpp 中的内核选择函数
- `cache_test.cpp` - 测试 cache.cpp 中的缓存功能
- `csv_test.cpp` - 测试 csv.cpp 中的CSV文档处理功能
- `predictor_test.cpp` - 测试 predictor.cpp 中的预测器功能
- `main_test.cpp` - 测试主入口

## 编译和运行

### 编译测试

```bash
cd aclnn_catlass_dynamic_matmul/tests/op_host/op_tiling
mkdir build && cd build
cmake ..
make
```

### 运行所有测试

```bash
./main_test
```

### 运行单个测试

```bash
./utils_test
./do_tiling_test
./select_kernel_test
./cache_test
./csv_test
./predictor_test
```

## 测试覆盖的函数

### utils.cpp
- CeilDiv
- RoundUp
- BalanceWorkload
- SetTile
- IsExStrideLimit
- JudgeSpace
- GetMaxK1

### do_tiling.cpp
- DoTilingLayout00
- DoTilingLayout01
- DoTilingLayout10
- DoTilingLayout11

### select_kernel.cpp
- SelectKernel
- SelectKernelWithCache

### cache.cpp
- TilingCache::GetInstance
- TilingCache::GetTiling
- TilingCache::SetTiling

### csv.cpp
- Document::InitFromFile
- Document::GetRowCount
- Document::GetCell
- Document::SaveRow
- Document::InitRowHead

### predictor.cpp
- Predictor::GetInstance
- Predictor::Predict