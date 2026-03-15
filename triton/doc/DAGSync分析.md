# DAGSync.cpp 详细分析文档

## 1. 概述

### 1.1 功能定位

DAGSync.cpp 是 Triton 亲和性优化流程中的**第一个 Pass**（最先执行），其核心职责是：

> **在 Vector 核心和 Cube 核心之间插入同步指令和数据搬运操作，确保跨核心数据依赖的正确性。**

### 1.2 在整体流程中的位置

```
┌─────────────────────────────────────────────────────────────┐
│                   Triton IR (TTIR)                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Pass 1: DAGSync  ★ 本文档分析对象                           │
│  - 分析 DAG 依赖                                              │
│  - 插入 SyncBlockSet/SyncBlockWait 同步指令                 │
│  - 插入数据搬运操作 (fixpipe, copy)                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Pass 2: DAGScope                                           │
│  - 封装 Scope，划分为 AIV (Vector) 和 AIC (Cube) 区域       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Pass 3: DAGSSBuffer                                        │
│  - 将向量操作转换为共享存储缓冲区操作                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   目标代码                                  │
│  - 包含正确的 Vector<->Cube 同步和数据搬运                   │
└─────────────────────────────────────────────────────────────┘
```

> **注意**: Pass 执行顺序依据 `third_party/ascend/backend/compiler.py` 中 `add_auto_scheduling` 配置下的实际调用顺序：
> ```python
> ascend.passes.ttir.add_dag_sync(pm)      # 1. 首先
> ascend.passes.ttir.add_dag_scope(pm)      # 2. 其次
> ascend.passes.ttir.add_dag_ssbuffer(pm)   # 3. 最后
> ```

### 1.3 核心概念

在分析代码前，需要理解昇腾 NPU 的两个核心类型：

| 核心类型 | 说明 | 典型操作 |
|---------|------|----------|
| **Vector (AIV)** | 向量计算核心，适合非矩阵乘法的张量操作 | load, store, element-wise ops, reduce |
| **Cube (AIC)** | 矩阵计算核心，适合矩阵乘法操作 | tt.dot |

**关键洞察**：当一个操作的输出被另一个核心类型的操作消费时，必须插入同步和数据搬运。

---

## 2. 代码结构总览

### 2.1 文件组织

DAGSync.cpp (约 960 行) 可以分为以下几个部分：

```
DAGSync.cpp
├── 头文件包含
├── 全局/命名空间变量
├── 辅助函数 (5个)
│   ├── newCbubAllocShape()      - 计算新形状
│   └── rewriteCopyChainForCbub() - 重写Copy链
├── DAGSyncPass 类 (主Pass)
│   ├── 成员变量
│   ├── 辅助方法 (9个)
│   │   ├── getNodeDeviceType()        - 获取节点设备类型
│   │   ├── needVectorCubeSync()       - 判断是否需要同步
│   │   ├── getTensorType()            - 获取张量类型
│   │   ├── replaceOperandWithNewValue() - 替换操作数
│   │   ├── getOrCreateAllocation()    - 获取/创建内存分配
│   │   ├── insertCubeToVectorDataMovement()  - CUBE→VECTOR 搬运
│   │   ├── insertVectorToCubeDataMovement()  - VECTOR→CUBE 搬运
│   │   ├── insertSyncAndMovement()     - 同步+搬运主函数
│   │   ├── insertSyncAndMovementForCrossBlock() - 跨块同步
│   │   └── processScfForSync()         - 处理 scf.for 循环
│   └── runOnOperation()               - 主入口
└── 独立函数
    ├── LegalizeDot()                  - 合法化 Dot 操作
    └── createDAGSyncPass()            - Pass 工厂函数
```

### 2.2 阅读顺序建议

**推荐阅读顺序**：按照数据流和调用关系

```
1. runOnOperation()           ← 入口点，理解整体流程
   │
   ├── 构建 DAG 图
   │   └── Graph::fromMultiBlockFunc()
   │
   ├── 标记核心类型
   │   └── main_graph.markCore()
   │
   ├── 遍历操作，插入同步
   │   ├── getNodeDeviceType()       ← 辅助：获取类型
   │   ├── needVectorCubeSync()      ← 辅助：判断需求
   │   ├── insertSyncAndMovement()   ← 核心逻辑
   │   │   ├── insertCubeToVectorDataMovement()
   │   │   └── insertVectorToCubeDataMovement()
   │   └── processScfForSync()       ← 处理循环
   │
   └── 后续处理
       └── rewriteCopyChainForCbub()
```

---

## 3. 数据流分析

### 3.1 输入示例（来自 fwd.ttir）

以下是一个简化的 Flash Attention 内层循环结构：

```mlir
// 外层循环 (i 维度)
scf.for %arg8 = %29 to %31 step %c64_i32 {
  // ===== VECTOR 核心区域 =====
  %45 = tt.load %arg14      // 加载 Q 数据 (VECTOR)
  %46 = tt.trans %45        // 转置 (VECTOR)

  // ===== 跨核心依赖点! =====
  // %46 产生于 VECTOR，但被 %47 (dot) 消费 - 需要 CUBE
  %47 = tt.dot %27, %46, %cst_4   // 矩阵乘法 (CUBE)

  // ===== VECTOR 核心区域 =====
  %48 = arith.mulf %47, %cst_3    // 缩放 (VECTOR)
  %49 = tt.reduce %48              // 归约 (VECTOR)
  %53 = math.exp %53               // 指数 (VECTOR)
  %55 = tt.load %arg13             // 加载 K 数据 (VECTOR)
  // ...
}
```

### 3.2 DAGSync 处理流程图

```
┌──────────────────────────────────────────────────────────────────┐
│                        runOnOperation()                          │
│                                                                  │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐ │
│  │ 1. 构建 DAG    │───▶│ 2. 标记核心类型 │───▶│ 3. 遍历操作    │ │
│  │ Graph::from    │    │ markCore()      │    │ 插入同步       │ │
│  │ MultiBlockFunc │    │                 │    │                │ │
│  └────────────────┘    └────────────────┘    └────────────────┘ │
│                                                      │           │
│                                                      ▼           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     遍历每个 Operation                     │   │
│  │                                                          │   │
│  │   For each Op:                                          │   │
│  │     1. 找到对应的 Node                                   │   │
│  │     2. 获取当前操作的核心类型 (VECTOR/CUBE)              │   │
│  │     3. 遍历所有输入节点                                   │   │
│  │        ├─ 如果输入类型 ≠ 当前类型                         │   │
│  │        │   └─→ 需要同步! 插入 Sync + Data Movement        │   │
│  │        └─ 如果输入类型 = 当前类型                         │   │
│  │            └─→ 无需处理                                  │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.3 同步插入示例

基于上述 ttir 例子，当检测到 VECTOR → CUBE 依赖时：

```
原始代码:
  %46 = tt.trans %45      // VECTOR 产生
  %47 = tt.dot ... %46    // CUBE 消费

处理后:
  %46 = tt.trans %45      // VECTOR 产生
  ────────────────────────── 新增 ─────────────────────────
  // 1. to_memref: VECTOR tensor → UB memref
  %ub_memref = bufferization.to_memref %46 : tensor<...> to memref<..., UB>

  // 2. copy: UB memref → L1 memref
  %l1_memref = hivm.copy %ub_memref : memref<..., UB> to memref<..., L1>

  // 3. convert_layout: 调整布局
  %cbuf = hivm.convert_layout %l1_memref : memref<..., L1> to memref<...>

  // 4. memory_space_cast: 转换为普通 memref
  %plain = memref.memory_space_cast %cbuf : ...

  // 5. to_tensor: memref → tensor
  %cube_tensor = bufferization.to_tensor %plain : tensor<...>

  // 6. Sync: 插入同步指令
  hivm.sync_block_set ...  // 设置同步点
  ─────────────────────────────────────────────────────────
  %47 = tt.dot ... %cube_tensor    // CUBE 消费
  hivm.sync_block_wait ...          // 等待同步
```

---

## 4. 函数详细分析

### 4.1 主入口: runOnOperation()

**位置**: 第 808-955 行

**功能**: Pass 的主入口，遍历所有函数并处理同步

**核心逻辑**:

```cpp
void DAGSyncPass::runOnOperation() {
    // 1. 遍历所有 triton 函数
    for (auto funcOp : llvm::make_early_inc_range(module.getOps<triton::FuncOp>())) {
        // 2. 构建 DAG 依赖图
        auto [shared_graph, _] = Graph::fromMultiBlockFunc(funcOp);
        auto& main_graph = *shared_graph;

        // 3. 标记每个操作的核心类型 (VECTOR/CUBE/SCALAR)
        main_graph.markCore();

        // 4. 遍历每个操作，插入同步
        funcOp.walk([&](mlir::Operation *op) {
            // 获取当前操作的核心类型
            CoreType currentType = getNodeDeviceType(currentNode, valueTypes);

            // 遍历所有输入依赖
            for (Node *inputNode : currentNode->ins) {
                CoreType inputType = getNodeDeviceType(inputNode, valueTypes);

                // 如果类型不同，插入同步
                if (needVectorCubeSync(inputType, currentType)) {
                    insertSyncAndMovement(...);
                }
            }
        });

        // 5. 后处理：重写 copy 链
        funcOp.walk([&](hivm::CopyOp copyOp) {
            rewriteCopyChainForCbub(...);
        });

        // 6. 注册图供后续 pass 使用
        GraphManager::getInstance().registerGraph(funcName, shared_graph);
    }
}
```

**与 ttir 的对应**:
- 对于 fwd.ttir 中的 `scf.for` 循环，会调用 `processScfForSync()` 特殊处理
- 对于 `tt.dot` (CUBE) 消费 `tt.trans` (VECTOR) 产出的情况，会触发 `insertSyncAndMovement()`

---

### 4.2 辅助判断函数

#### 4.2.1 getNodeDeviceType()

**位置**: 第 202-227 行

**功能**: 获取操作对应的核心类型

```cpp
CoreType DAGSyncPass::getNodeDeviceType(Node *node, llvm::DenseMap<mlir::Value, CoreType> *valueTypes) {
    if (!node || !node->op) {
        return CoreType::SCALAR;  // 默认
    }

    // 尝试从操作结果获取类型
    if (node->op->getNumResults() > 0) {
        mlir::Value result = node->op->getResult(0);
        auto it = valueTypes->find(result);
        if (it != valueTypes->end()) {
            return it->second;  // VECTOR / CUBE / SCALAR
        }
    }

    return CoreType::SCALAR;  // 默认
}
```

**对应 ttir 示例**:
```
tt.trans %45 → VECTOR (因为操作向量张量)
tt.dot %27, %46 → CUBE (矩阵乘法)
arith.addf %a, %b → SCALAR (标量运算)
```

#### 4.2.2 needVectorCubeSync()

**位置**: 第 230-234 行

**功能**: 判断两个核心类型之间是否需要同步

```cpp
bool DAGSyncPass::needVectorCubeSync(CoreType src, CoreType dst) {
    return (src == CoreType::VECTOR && dst == CoreType::CUBE) ||
           (src == CoreType::CUBE && dst == CoreType::VECTOR);
}
```

**真值表**:

| 源类型 (src) | 目标类型 (dst) | 需要同步? |
|-------------|---------------|----------|
| VECTOR      | CUBE          | ✅ 是    |
| CUBE        | VECTOR        | ✅ 是    |
| VECTOR      | VECTOR        | ❌ 否    |
| CUBE        | CUBE          | ❌ 否    |
| SCALAR      | 任意          | ❌ 否    |

---

### 4.3 数据搬运函数

#### 4.3.1 insertCubeToVectorDataMovement()

**位置**: 第 296-359 行

**功能**: 将数据从 CUBE 核心搬运到 Vector 核心

**典型场景**: `tt.dot` 的结果被后续的 `arith.mulf` 消费

```
数据流:
CUBE tensor → memref(UB) → fixpipe → memref(CBUF) → to_tensor → VECTOR tensor
```

**代码逻辑**:
```cpp
void DAGSyncPass::insertCubeToVectorDataMovement(...) {
    // 1. 在 srcOp 后创建 UB 空间的 memref.alloc
    mlir::Value ubAlloc = getOrCreateAllocation(srcOp, srcTensorType,
                                                hivm::AddressSpace::UB, builder, loc);

    // 2. 创建 fixpipe 指令 (CUBE → UB)
    FixpipeDMAModeAttr dmaModeAttr = FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2ND);
    auto fixpipeOp = builder.create<hivm::FixpipeOp>(loc, TypeRange{},
        srcResult, ubAlloc, /*options*/...);

    // 3. memory_space_cast (如果需要)
    // 4. to_tensor
    auto toTensorOp = builder.create<bufferization::ToTensorOp>(loc, srcTensorType, plainMemref, ...);

    // 5. 替换 dstOp 的操作数
    replaceOperandWithNewValue(dstOp, srcResult, toTensorOp.getResult());
}
```

#### 4.3.2 insertVectorToCubeDataMovement()

**位置**: 第 383-503 行

**功能**: 将数据从 Vector 核心搬运到 Cube 核心

**典型场景**: `tt.trans` 的结果被 `tt.dot` 消费

```
数据流:
VECTOR tensor → to_memref → memref(UB) → copy → memref(L1) → convert_layout → to_tensor → CUBE tensor
```

**关键处理步骤**:

1. **to_memref**: VECTOR tensor → UB memref
2. **alloc L1**: 创建 L1 空间的目标内存
3. **copy**: UB memref → L1 memref (使用 hivm.copy)
4. **convert_layout**: 调整数据布局适配 CUBE
5. **memory_space_cast**: 转换内存空间
6. **to_tensor**: memref → tensor
7. **替换操作数**: 使用新产生的 tensor

---

### 4.4 同步插入函数

#### 4.4.1 insertSyncAndMovement()

**位置**: 第 505-602 行

**功能**: 主同步插入函数，协调同步指令和数据搬运

```cpp
void DAGSyncPass::insertSyncAndMovement(
    mlir::Operation *srcOp,      // 源操作
    mlir::Operation *dstOp,      // 目标操作
    CoreType srcType,            // 源核心类型
    CoreType dstType,           // 目标核心类型
    mlir::OpBuilder &builder,
    int flag,
    llvm::DenseMap<Value, CoreType>* valueMap
)
```

**处理逻辑**:

```
┌─────────────────────────────────────────────────────────────┐
│                  insertSyncAndMovement()                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  判断方向:                                                  │
│  ┌──────────────────┐    ┌──────────────────┐             │
│  │ src=CUBE         │    │ src=VECTOR       │             │
│  │ dst=VECTOR       │    │ dst=CUBE         │             │
│  └────────┬─────────┘    └────────┬─────────┘             │
│           │                       │                        │
│           ▼                       ▼                        │
│  ┌──────────────────────────────────────────┐              │
│  │ 1. 插入数据搬运                           │              │
│  │   insertCubeToVectorDataMovement()       │              │
│  │   或                                       │              │
│  │   insertVectorToCubeDataMovement()        │              │
│  └────────────────────┬─────────────────────┘              │
│                       │                                       │
│                       ▼                                       │
│  ┌──────────────────────────────────────────┐              │
│  │ 2. 插入同步指令                           │              │
│  │                                           │              │
│  │   // CUBE → VECTOR:                      │              │
│  │   set: PIPE_FIX → wait: PIPE_V           │              │
│  │   位置: srcOp之后，dstOp之前             │              │
│  │                                           │              │
│  │   // VECTOR → CUBE:                       │              │
│  │   set: PIPE_MTE3 → wait: PIPE_MTE1       │              │
│  │   位置: srcOp之后，dstOp之前             │              │
│  └──────────────────────────────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**同步指令类型**:

| 方向 | Set Pipe | Wait Pipe | 说明 |
|------|----------|-----------|------|
| CUBE → VECTOR | PIPE_FIX | PIPE_V | Cube 写完成后等待 |
| VECTOR → CUBE | PIPE_MTE3 | PIPE_MTE1 | Vector 写完成后等待 |

**SyncBlockSetOp 和 SyncBlockWaitOp 示例**:
```mlir
// CUBE → VECTOR 同步
%set = hivm.sync_block_set {static_flag_id = 1 : i64, tcore_type = #hivm.tcore_type<CUBE>}
           PIPE_FIX, PIPE_V
%wait = hivm.sync_block_wait {static_flag_id = 1 : i64, tcore_type = #hivm.tcore_type<VECTOR>}
           PIPE_V, PIPE_FIX

// 使用 flag 关联 set 和 wait，确保数据可用
```

---

#### 4.4.2 insertSyncAndMovementForCrossBlock()

**位置**: 第 605-687 行

**功能**: 处理跨基本块（Block）边界的同步和数据搬运

**与 insertSyncAndMovement() 的区别**:

| 特性 | `insertSyncAndMovement` | `insertSyncAndMovementForCrossBlock` |
|------|------------------------|-----------------------------------|
| 处理范围 | 同一Block内的操作 | 跨Block的操作（如外层到内层循环） |
| Wait插入位置 | 直接插入在dstOp之前 | 插入在内层Block的入口前 |
| Set插入位置 | srcOp之后 | srcOp之后（外层） |
| 复杂度 | 简单 | 复杂（需要查找Block层级关系） |

**函数签名**:
```cpp
void insertSyncAndMovementForCrossBlock(
    mlir::Operation *srcOp,      // 源操作（产生数据）
    mlir::Operation *dstOp,      // 目标操作（消费数据）
    CoreType srcType,            // 源核心类型（CUBE/VECTOR）
    CoreType dstType,           // 目标核心类型
    mlir::OpBuilder &builder,
    int flag,                   // 同步标志ID
    bool dstIsInnerBlock,       // 目标是否在内层Block
    llvm::DenseMap<Value, CoreType>* valueMap
)
```

**处理逻辑**:

```
┌─────────────────────────────────────────────────────────────┐
│    insertSyncAndMovementForCrossBlock()                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 参数验证                                                 │
│     if (!dstIsInnerBlock)                                   │
│        → 退化为普通 insertSyncAndMovement()                 │
│                                                             │
│  2. 判断同步方向                                             │
│     ┌──────────────────┐    ┌──────────────────┐            │
│     │ src=CUBE         │    │ src=VECTOR       │            │
│     │ dst=VECTOR       │    │ dst=CUBE         │            │
│     └────────┬─────────┘    └────────┬─────────┘            │
│              │                       │                     │
│              ▼                       ▼                     │
│     ┌──────────────────────────────────────────┐           │
│     │ 2.1 插入数据搬运                         │           │
│     │   insertCubeToVectorDataMovement()      │           │
│     │   或                                     │           │
│     │   insertVectorToCubeDataMovement()       │           │
│     └────────────────────┬─────────────────────┘           │
│                          │                                  │
│                          ▼                                  │
│     ┌──────────────────────────────────────────┐           │
│     │ 2.2 插入同步指令（跨Block特殊处理）       │           │
│     │                                           │           │
│     │   Set: 在 srcOp 之后（外层）             │           │
│     │   Wait: 在内层Block入口前（非dstOp前）   │           │
│     │                                           │           │
│     │   关键：确保整个内层Block看到同步状态   │           │
│     └──────────────────────────────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**CUBE → VECTOR 场景** (AIC到AIV):

用于AIC计算完成后，结果需要传递给AIV进行后续处理：

```cpp
// AIC端设置完成标志
builder.setInsertionPointAfter(srcOp);
builder.create<SyncBlockSetOp>(loc, CUBE, PIPE_FIX, PIPE_V, flagId);

// 插入数据搬运（创建memref，处理地址空间转换）
insertCubeToVectorDataMovement(srcOp, dstOp, srcResult, builder, loc, nullptr);

// 查找内层Block的parentOp，在入口前插入Wait
builder.setInsertionPoint(parentOp);  // 关键：在Block入口
builder.create<SyncBlockWaitOp>(loc, VECTOR, PIPE_FIX, PIPE_V, flagId);
```

**同步示例**:
```mlir
// 外层Block (AIC)
%result = tt.dot %a, %b, %c  // CUBE类型
hivm.sync_block_set {flag = 1, tcore_type=CUBE, set_pipe=PIPE_FIX, wait_pipe=PIPE_V}

// 内层Block入口 (AIV)  <- Wait插入位置
hivm.sync_block_wait {flag = 1, tcore_type=VECTOR, set_pipe=PIPE_FIX, wait_pipe=PIPE_V}
scf.for %i = ... {
  %processed = math.exp %result  // VECTOR类型
}
```

**VECTOR → CUBE 场景** (AIV到AIC):

用于AIV准备数据后，传递给AIC进行矩阵计算：

```cpp
// AIV端设置数据ready标志
builder.setInsertionPointAfter(srcOp);
builder.create<SyncBlockSetOp>(loc, VECTOR, PIPE_MTE3, PIPE_MTE1, flagId);

// 插入数据搬运（处理内存布局转换）
insertVectorToCubeDataMovement(srcOp, dstOp, srcResult, builder, loc, valueMap);

// 在内层Block入口前等待
builder.setInsertionPoint(parentOp);
builder.create<SyncBlockWaitOp>(loc, CUBE, PIPE_MTE3, PIPE_MTE1, flagId);
```

**调用时机**:

此函数在 `insertSyncAndMovement` 中检测到跨Block依赖时被调用：

```cpp
// 检测是否跨Block
bool dstIsInnerBlock = false;
mlir::Operation *dstParentOp = dstBlock->getParentOp();

// 向上查找dstOp的Block是否在srcOp的Block内部
while (dstParentOp) {
    if (dstParentOp->getBlock() == srcBlock) {
        dstIsInnerBlock = true;  // 找到！dst在内层
        break;
    }
    dstParentOp = dstParentOp->getBlock()->getParentOp();
}

if (dstIsInnerBlock) {
    // 调用跨Block版本
    insertSyncAndMovementForCrossBlock(srcOp, dstOp, srcType, dstType,
                                      builder, flag, true, valueMap);
}
```

**实际应用场景**:

1. **外层循环到内层循环的数据传递**
```mlir
// 外层计算（AIC）
%result = tt.dot %a, %b, %c  // CUBE类型

// 内层循环需要%result（AIV）
scf.for %i = %c0 to %n step %c1 {
  // 跨Block依赖！
  %processed = math.exp %result  // VECTOR类型
}
```

2. **循环迭代参数的跨Block同步**
在 `processScfForSync` 函数中，处理迭代参数的跨核心传递：

```cpp
// 循环yield的值类型是CUBE，但首次使用是VECTOR
if (yieldType == CUBE && iterType == VECTOR) {
    insertSyncAndMovementForCrossBlock(
        yieldDefiningOp,    // src: CUBE操作
        firstUser,          // dst: VECTOR操作（在内层Block）
        CUBE, VECTOR,
        builder, flag,
        true,  // dstIsInnerBlock=true
        nullptr
    );
}
```

**具体 TTIR 示例**:

```mlir
// 场景：循环迭代参数的跨 Block 同步
// 问题：yield返回CUBE类型，但下一轮迭代首次使用时是VECTOR类型

scf.for %i = %c0 to %c16 step %c1
    iter_args(%cube_buffer = %init_cube)  // 初始为CUBE类型
    -> (tensor<64x64xf32>) {             // yield类型：CUBE

  // 【首次使用】%cube_buffer 被一个 VECTOR 操作使用
  // ⚠️ 这触发了 CUBE -> VECTOR 同步需求
  %memref_vec = bufferization.to_memref %cube_buffer
               : tensor<64x64xf32> -> memref<64x64xf32, ...>
  // to_memref 是 VECTOR 类型操作！

  // 准备数据给 CUBE 操作
  %processed = hivm.fixpipe %memref_vec  // VECTOR -> CUBE 转换

  // CUBE 计算
  %k = tt.load %ptr_k : !tt.ptr<tensor<64x64xf32>>
  %kt = tt.trans %k {order = array<i32: 1, 0>}
  %q = tt.load %ptr_q : !tt.ptr<tensor<64x64xf32>>
  %result = tt.dot %q, %kt, %processed

  // yield返回CUBE类型
  scf.yield %result : tensor<64x64xf32>
}

// processScfForSync 的分析过程：
// 1. iterArg = %cube_buffer (BlockArgument)
// 2. firstUser = bufferization.to_memref (VECTOR类型)
// 3. yieldDefiningOp = tt.dot (CUBE类型)
// 4. iterType = VECTOR (来自firstUser)
// 5. yieldType = CUBE (来自yield)
// 6. 条件触发：yieldType == CUBE && iterType == VECTOR
// 7. 调用 insertSyncAndMovementForCrossBlock(
//        srcOp = tt.dot,        // CUBE 操作
//        dstOp = to_memref,     // VECTOR 操作（在下一轮迭代的Block中）
//        srcType = CUBE,
//        dstType = VECTOR,
//        dstIsInnerBlock = true  // 关键：dstOp在新的一轮迭代Block中
//    )

// 生成的同步代码：
scf.for %i = ... iter_args(%buffer = %init) -> (...) {
  // 上一轮迭代的 CUBE 计算
  %result = tt.dot ...

  // 【同步1】在 CUBE 操作后 Set（本轮迭代）
  hivm.sync_block_set {flag=42, tcore_type=CUBE,
                      set_pipe=PIPE_FIX, wait_pipe=PIPE_V}
  scf.yield %result

  // 内层 Block 开始（下一轮迭代入口）
  // 【同步2】在 Block 入口前 Wait
  hivm.sync_block_wait {flag=42, tcore_type=VECTOR,
                       set_pipe=PIPE_FIX, wait_pipe=PIPE_V}

  // firstUser: to_memref
  %memref = bufferization.to_memref %buffer
  ...
}
```

**为什么 Wait 要插在 Block 入口而不是 dstOp 前？**

```mlir
// ❌ 错误的做法：Wait 直接插在 dstOp 前
scf.for ... {
  ...
  scf.yield %result
}
scf.for ... {  // 下一轮迭代
  // 其他操作...
  // 【Wait在这里】
  hivm.sync_block_wait {flag=42}
  %memref = to_memref %buffer  // firstUser
}
// 问题：如果第一轮迭代快速完成，第二轮开始时 Wait 才发出
// 可能导致流水线气泡

// ✅ 正确的做法：Wait 插在 Block 入口
scf.for ... {
  ...
  // 【Set在这里】
  hivm.sync_block_set {flag=42}
  scf.yield %result
}
// 【Wait在这里】在 Block 入口，父操作前
scf.for ... {  // 下一轮迭代
  // 立即等待，确保 Block 内所有操作都能看到同步状态
  hivm.sync_block_wait {flag=42}
  // 然后执行后续操作...
  %memref = to_memref %buffer
}
// 好处：整个 Block 的执行都会受到同步保护
```

**关键洞察**：循环迭代参数的同步发生在**循环边界**，跨越的是**时间维度**（本轮迭代到下一轮迭代），而不是空间维度（外层到内层Block）。但插入逻辑类似，都是确保数据在不同执行上下文间的正确传递。


**与 DAGScope Pass 的关系**:

`insertSyncAndMovementForCrossBlock` 在 **DAGSync Pass** 中使用（最先执行），而不是 DAGScope Pass：

```
DAGSync Pass (插入同步，最先执行)
    ↓
insertSyncAndMovementForCrossBlock (处理跨Block依赖)
    ↓
生成带同步原语的IR
    ↓
DAGScope Pass (封装Scope，第二执行)
    ↓
生成 AIV Scope 和 AIC Scope
```

**关键设计决策**:

1. **为什么Wait要插在Block入口而不是dstOp前？**
   - 保证Block内所有操作都能看到同步完成状态
   - 避免在Block内多次插入Wait（如果多个操作依赖同一数据）
   - 更符合控制流图的结构

2. **如何找到正确的插入位置？**
   ```cpp
   mlir::Operation *parentOp = dstBlock->getParentOp();
   while (srcOp->getBlock() != parentOp->getBlock()) {
       parentOp = parentOp->getBlock()->getParentOp();
   }
   builder.setInsertionPoint(parentOp);
   ```
   通过向上遍历Block层级，找到内层Block的父操作，在其之前插入Wait。

3. **与普通同步的区别**:
   - 普通同步：Set(src后) → Wait(dst前) → dstOp执行
   - 跨Block同步：Set(src后) → Wait(内层Block入口) → 内层Block执行

**总结**:

`insertSyncAndMovementForCrossBlock` 是处理跨核心、跨Block边界数据依赖的关键函数，它通过在内层Block入口前插入Wait，确保整个内层Block都能看到同步完成状态，从而正确实现AIV和AIC核心之间的数据传递。

---

### 4.5 循环处理函数

#### 4.5.1 processScfForSync()

**位置**: 第 93-199 行

**功能**: 特殊处理 scf.for 循环中的迭代参数同步

**问题背景**:
在循环中，迭代参数 (iter_args) 跨越循环边界传递数据：
- 循环起始的迭代参数
- yield 操作返回的值
- 下一次迭代使用的参数

这些值的核心类型可能在循环内发生变化，需要特别处理。

**代码逻辑**:

```cpp
void DAGSyncPass::processScfForSync(
    mlir::scf::ForOp forOp,
    Node* forNode,
    llvm::DenseMap<mlir::Value, CoreType> *valueTypes,
    mlir::OpBuilder &builder,
    int &flag
) {
    // 1. 获取循环体和 yield 操作
    mlir::Block* loopBody = forOp.getBody();
    mlir::scf::YieldOp yieldOp = ...;

    // 2. 遍历所有迭代参数
    for (int i = 0; i < forOp.getInitArgs().size(); i++) {
        mlir::BlockArgument iterArg = loopBody->getArgument(i+1);

        // 3. 找到首次使用
        mlir::Operation* firstUser = ...;

        // 4. 获取类型信息
        CoreType iterType = ...;   // 循环内首次使用的类型
        CoreType yieldType = ...; // yield 返回的类型

        // 5. 判断是否需要同步
        // CUBE → VECTOR
        if (yieldType == CoreType::CUBE && iterType == CoreType::VECTOR) {
            // 插入同步: 在 yieldDefiningOp 后 set，在 firstUser 前 wait
        }
        // VECTOR → CUBE
        else if (yieldType == CoreType::VECTOR && iterType == CoreType::CUBE) {
            // 插入同步
        }
    }
}
```

**示例 (来自 fwd.ttir)**:
```mlir
// 循环迭代参数
%28:5 = scf.for %arg8 = ... iter_args(%arg9 = %cst, %arg10 = %cst_4, ...) {
  // 循环内计算...
  // yield 返回的值
  scf.yield %60, %64, %50, %65, %66 : ...
}
```

如果 `yield` 产出的类型与下一次迭代使用的类型不同，需要同步。

---

### 4.6 其他辅助函数

#### 4.6.1 getOrCreateAllocation()

**位置**: 第 256-294 行

**功能**: 获取或创建内存分配，确保在函数入口处分配

```cpp
mlir::Value DAGSyncPass::getOrCreateAllocation(
    mlir::Operation *op,
    mlir::Type tensorType,
    hivm::AddressSpace addressSpace,
    mlir::OpBuilder &builder,
    mlir::Location loc
)
```

**地址空间**:
| 地址空间 | 说明 |
|---------|------|
| UB | Unified Buffer，通用缓冲 |
| L1 | L1 Cache，一级缓存 |
| CBUF | Cube Buffer，矩阵乘法专用 |

#### 4.6.2 rewriteCopyChainForCbub()

**位置**: 第 741-806 行

**功能**: 重写 copy 操作链以适配 CUBE 格式

**背景**: CUBE 核心对数据布局有特殊要求 (16x16 分块)，需要 reshape 和 transpose

```cpp
// 原始: [M, N] 2D tensor
// 变换: [M/16, N/16, 16, 16] 4D tensor (分块)
// 转置: [N/16, M/16, 16, 16]
```

---

### 4.7 LegalizeDot()

**位置**: 第 689-739 行

**功能**: 合法化 Dot 操作

**问题**: 如果 dot 操作的累加器 (第三个参数) 不是全零，需要将其拆分

```cpp
// 原始: dot(a, b, c)  where c != 0
// 转换:
//   1. zero = constant(0)
//   2. new_dot = dot(a, b, zero)
//   3. result = add(new_dot, c)
```

---

## 5. 完整数据流示例

结合 fwd.ttir 的完整处理流程：

```
输入 (TTIR):
┌─────────────────────────────────────────────────────────────┐
│ scf.for 外层循环                                            │
│   scf.for 内层循环 (Flash Attention)                        │
│     %45 = tt.load  (VECTOR)                                 │
│     %46 = tt.trans (VECTOR) ─┐                               │
│     %47 = tt.dot   (CUBE)   │ 需要同步!                     │
│     %48 = arith.mulf(VECTOR) ◄┘                             │
│     %49 = tt.reduce(VECTOR)                                 │
│     %53 = math.exp  (VECTOR)                                │
│     %55 = tt.load  (VECTOR)                                 │
│     ...                                                     │
│   scf.yield                                                 │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
处理过程:
┌─────────────────────────────────────────────────────────────┐
│ 1. 构建 DAG 图                                             │
│    - 分析操作依赖关系                                        │
│    - 创建 Node 和边                                         │
│                                                              │
│ 2. 标记核心类型                                             │
│    - %45, %46, %48, %49, %53, %55, ... → VECTOR            │
│    - %47 (tt.dot) → CUBE                                    │
│                                                              │
│ 3. 遍历并插入同步                                           │
│    - 检测 %46 (VECTOR) → %47 (CUBE) 依赖                    │
│    - 插入 Vector→Cube 数据搬运                              │
│    - 插入 sync_block_set/wait                              │
│                                                              │
│ 4. 后处理                                                   │
│    - rewriteCopyChainForCbub                               │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
输出 (带同步的 TTIR):
┌─────────────────────────────────────────────────────────────┐
│ scf.for 内层循环                                            │
│   %45 = tt.load  (VECTOR)                                   │
│   %46 = tt.trans (VECTOR)                                   │
│   ─────────────────────────────────────────                  │
│   %ub_mem = bufferization.to_memref %46                     │
│   %l1_mem = hivm.copy %ub_mem : ... to ...                 │
│   %cube_tensor = bufferization.to_tensor %l1_mem            │
│   %set = hivm.sync_block_set ... flag=1                    │
│   ─────────────────────────────────────────                  │
│   %47 = tt.dot %27, %cube_tensor, ... (CUBE)               │
│   %wait = hivm.sync_block_wait ... flag=1                  │
│   ─────────────────────────────────────────                  │
│   %48 = arith.mulf %47, ... (VECTOR)                       │
│   ...                                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 如何阅读 DAGSync.cpp

### 6.1 阅读路线图

```
第一次阅读 (了解整体):
1. runOnOperation()        → 理解 Pass 入口和流程骨架
2. 快速浏览辅助函数         → 了解有哪些工具可用

第二次阅读 (理解核心逻辑):
3. needVectorCubeSync()    → 理解同步触发条件
4. getNodeDeviceType()     → 理解类型获取方式
5. insertSyncAndMovement() → 核心同步逻辑

第三次阅读 (深入细节):
6. insertCubeToVectorDataMovement()  → CUBE→VECTOR 搬运
7. insertVectorToCubeDataMovement()  → VECTOR→CUBE 搬运
8. processScfForSync()               → 循环特殊处理
9. rewriteCopyChainForCbub()         → Copy 链重写
```

### 6.2 关键数据流追踪

阅读时建议追踪以下数据流：

1. **valueTypes Map**:
   - 谁创建? → `Graph::markCore()` 在 DAG.cpp 中
   - 谁使用? → `getNodeDeviceType()` 读取
   - 存储什么? → Value → CoreType 映射

2. **同步 flag**:
   - 递增逻辑? → `syncFlag++`
   - 作用? → 关联 set 和 wait，确保配对

3. **操作数替换**:
   - 为什么替换? → 数据搬运后，新产生的值替代原值
   - 替换函数? → `replaceOperandWithNewValue()`

### 6.3 调试技巧

代码中有大量 `llvm::outs()` 输出，可以用于调试：

```cpp
// 关键输出位置:
// - 第 172-173 行: 循环同步信息
// - 第 250-251 行: 操作数替换信息
// - 第 327-328 行: fixpipe 插入信息
// - 第 463-464 行: copy 插入信息
// - 第 573/600 行: 同步插入信息
```

---

## 6. DAGSync插入的同步在后续Pass中的使用

DAGSync Pass 插入的同步原语（`SyncBlockSetOp` 和 `SyncBlockWaitOp`）在后续的 Pass 中被广泛使用，主要包括：

### 6.1 在 DAGScope Pass 中的使用

DAGScope Pass（第二个执行的 Pass）会查找和使用 DAGSync 插入的同步操作，主要用于：

#### 1. 查找现有的同步操作

在 `addSyncOpsForBufferWait` 函数中（DAGScope.cpp:1041-1073）：

```cpp
/// 查找下一个可用的 flag
static int64_t findNextAvailableFlag(triton::FuncOp funcOp) {
  int64_t maxFlag = -1;

  // 遍历所有 SyncBlockSetOp 和 SyncBlockWaitOp
  funcOp.walk([&](Operation *op) {
    IntegerAttr flagAttr;
    if (isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op)) {
      flagAttr = op->getAttrOfType<IntegerAttr>("static_flag_id");
    }
    if (flagAttr && flagAttr.getInt() > maxFlag) {
      maxFlag = flagAttr.getInt();
    }
  });

  return maxFlag + 1;  // 从现有最大值+1开始
}
```

**使用场景**：为新增同步操作分配不重复的 flag ID

#### 2. 在 AIC→AIV 同步处理中查找 SetOp

在 `processFixpipeOpsInAIC` 函数中（DAGScope.cpp:894-928）：

```cpp
/// 处理 AIC 中的 FixpipeOp
static void processFixpipeOpsInAIC(
    Region *aicRegion, Region *aivRegion, int64_t &nextFlag) {

  aicRegion->walk([&](hivm::FixpipeOp fixpipeOp) {
    int64_t newflag = nextFlag++;

    // 1. 在 FixpipeOp 前插入 Wait
    builder.create<SyncBlockWaitOp>(...);

    // 2. 在 AIC Region 末尾插入 Wait
    insertWaitBeforeFinalReturn(aicRegion, builder, newflag, ...);

    // 3. 在 AIV Region 开头插入 Set
    insertSetAtRegionStart(aivRegion, builder, newflag, ...);

    // 4. 【查找 DAGSync 插入的 SetOp】
    if (auto *nextSetOp = findNextSyncBlockSetAfter(fixpipeOp)) {
      auto setFlagAttr = nextSetOp->getAttrOfType<IntegerAttr>("static_flag_id");
      int64_t setflag = setFlagAttr.getInt();

      // 5. 在 AIV Region 中找对应的 WaitOp
      auto targetWait = findWaitOpInRegionWithFlag(aivRegion, setflag);

      // 6. 在该 Wait 后插入新的 Set(newflag)
      if (auto *insertPt = findInsertionPointAfterWaitForAIV(targetWait)) {
        builder.setInsertionPoint(insertPt);
        builder.create<SyncBlockSetOp>(... newflag);
      }
    }
  });
}
```

**使用场景**：双向同步增强 - 查找 DAGSync 插入的 SetOp 作为锚点，在其后插入新的同步操作

#### 3. 在 AIV→AIC 同步处理中查找 SetOp

在 `processToMemrefOpsInAIV` 函数中（DAGScope.cpp:977-1018）：

类似地，查找 DAGSync 插入的 SetOp，并在 AIC Region 相应位置插入 Wait

```cpp
// 4. 在 aivRegion 向后找 SyncBlockSetOp
if (auto *nextSetOp = findNextSyncBlockSetAfter(toMemrefOp)) {
  auto setFlagAttr = nextSetOp->getAttrOfType<IntegerAttr>("static_flag_id");
  int64_t setflag = setFlagAttr.getInt();

  // 5. 在 aicRegion 中找 flag=setflag 的 WaitOp
  auto targetWait = findWaitOpInRegionWithFlag(aicRegion, setflag);

  // 6. 在该 Wait 后插入 Set(newflag)
  if (auto *insertPt = findInsertionPointAfterWaitForAIC(targetWait)) {
    builder.setInsertionPoint(insertPt);
    builder.create<SyncBlockSetOp>(..., newflag);
  }
}
```

**使用场景**：AIV→AIC 方向的双向同步增强

#### 4. 查找特定 flag 的 WaitOp

在 `findWaitOpInRegionWithFlag` 函数中（DAGScope.cpp:808-819）：

```cpp
static hivm::SyncBlockWaitOp findWaitOpInRegionWithFlag(Region *region, int64_t flag) {
  hivm::SyncBlockWaitOp result;
  region->walk([&](hivm::SyncBlockWaitOp op) {
    auto flagAttr = op->getAttrOfType<IntegerAttr>("static_flag_id");
    if (flagAttr && flagAttr.getInt() == flag) {
      result = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}
```

**使用场景**：根据 flag ID 查找对应的 Wait 操作，用于匹配 Set-Wait 对

#### 5. 查找 SetOp 后的插入点

在 `findInsertionPointAfterWaitForAIV` 和 `findInsertionPointAfterWaitForAIC` 中（DAGScope.cpp:821-847）：

```cpp
static Operation *findInsertionPointAfterWaitForAIV(Operation *waitOp) {
  Block *block = waitOp->getBlock();
  auto it = ++waitOp->getIterator();
  for (; it != block->end(); ++it) {
    if (isa<bufferization::ToMemrefOp>(*it) || isa<scf::YieldOp>(*it))
      return &*it;
  }
  return nullptr;
}
```

**使用场景**：找到 DAGSync 插入的 WaitOp 后的正确插入位置，确保新的同步操作插入在合适的位置

#### 6. 在 AIC 和 AIV 的边界插入额外的同步

在 `addSyncOpsForBufferWait` 函数中（DAGScope.cpp:1041-1073）：

```cpp
void addSyncOpsForBufferWait(ModuleOp module) {
  for (auto funcOp : module.getOps<triton::FuncOp>()) {
    // 查找 AIC Region 和 AIV Region
    Region *aicRegion = nullptr, *aivRegion = nullptr;
    funcOp.walk([&](scope::ScopeOp scopeOp) {
      if (coreType == CUBE) aicRegion = &scopeOp.getRegion();
      if (coreType == VECTOR) aivRegion = &scopeOp.getRegion();
    });

    if (!aicRegion || !aivRegion) continue;

    // 从 DAGSync 插入的 flag 之后继续分配
    int64_t nextFlag = findNextAvailableFlag(funcOp);

    // 处理 AIC 中的 FixpipeOp（会使用 DAGSync 的 SetOp 作为锚点）
    processFixpipeOpsInAIC(aicRegion, aivRegion, nextFlag);

    // 处理 AIV 中的 ToMemrefOp（会使用 DAGSync 的 SetOp 作为锚点）
    processToMemrefOpsInAIV(aivRegion, aicRegion, nextFlag);
  }
}
```

**使用场景**：在 DAGSync 的基础上，增强 AIC↔AIV 的双向同步，形成完整的同步链

#### 7. Pass 执行顺序的协同

```
原始 TTIR
    ↓
【DAGSync Pass】 (最先执行)
  - 分析 DAG
  - 插入同步 (Set/Wait)
  - 插入数据搬运 (Copy/Fixpipe)
    ↓
【DAGScope Pass】 (第二执行)
  - 读取 DAGSync 插入的同步操作
  - 查找 SetOp/WaitOp 作为锚点
  - 增强同步 (双向同步)
  - 封装 Scope
    ↓
【DAGSSBuffer Pass】 (第三执行)
  - 读取 DAGSync/DAGScope 插入的同步操作
  - 基于同步信息控制 SSBUffer
  - 转换操作到共享存储缓冲区
```

### 6.2 在 DAGSSBuffer Pass 中的使用

DAGSSBuffer Pass（第三个执行的 Pass）也会遍历和使用 DAGSync 插入的同步操作：

在 `ControlSsbufV2` 函数中（DAGSSBuffer.cpp:88-124）：

```cpp
void ControlSsbufV2(ModuleOp module) {
  mlir::OpBuilder builder(module.getContext());

  // 收集所有包含 SyncBlockWaitOp 的 for 循环
  llvm::DenseSet<mlir::Operation*> processedScopes2;
  module->walk([&](SyncBlockWaitOp waitOp) {
    // 向上查找父 scope.scope 操作
    mlir::Operation* parentOp = waitOp->getParentOp();
    mlir::Operation* scopeOp = nullptr;
    mlir::Operation* forOp = nullptr;

    // 向上遍历查找 scope.scope 操作
    while (parentOp) {
      if (dyn_cast<scope::ScopeOp>(parentOp)) {
        scopeOp = parentOp;
        break;
      }
      parentOp = parentOp->getParentOp();
    }
    // 查找父 scf.for 操作
    parentOp = waitOp->getParentOp();
    while (parentOp) {
      if (dyn_cast<scf::ForOp>(parentOp)) {
        forOp = parentOp;
        break;
      }
      parentOp = parentOp->getParentOp();
    }

    if (!scopeOp || !forOp) return;

    // 收集该 for 循环（去重）
    if (processedScopes2.count(forOp) > 0) return;
    processedScopes2.insert(forOp);
  });

  // 为收集到的每个 for 循环插入额外的同步控制
  for (auto forOp : processedScopes2) {
    // 查找父 scope.scope 操作
    mlir::Operation* scopeOp = nullptr;
    // ... 遍历查找 scopeOp ...

    bool isAIC = false;
    if (scopeOp->hasAttr("hivm.tcore_type")) {
      auto attr = scopeOp->getAttr("hivm.tcore_type");
      if (attr == aiCAttr) {
        isAIC = true;  // 确定为 AIC Scope
      }
    }

    if (isAIC) {
      // 在 AIC Scope 的 for 循环开头插入 Wait
      builder.setInsertionPointToStart(&forOp->getRegion(0).front());
      builder.create<SyncBlockWaitOp>(forOp->getLoc(),
        hivm::TCoreType::CUBE, PIPE::PIPE_S, PIPE::PIPE_S, flagId);

      // 在 for 循环末尾插入 Set
      auto &loopBody = forOp->getRegion(0).front();
      auto *terminator = loopBody.getTerminator();
      builder.setInsertionPoint(terminator);
      builder.create<SyncBlockSetOp>(forOp->getLoc(),
        hivm::TCoreType::CUBE, PIPE::PIPE_S, PIPE::PIPE_S, flagId);
    }
    else {
      // 在 AIV Scope 中处理...
    }
  }
}
```

**使用场景**：
- 遍历所有 DAGSync 插入的 `SyncBlockWaitOp`
- 找到其所在的 `scf.for` 循环和 `scope.scope` 区域
- 根据 Scope 的核心类型（AIC 或 AIV）插入额外的 PIPE_S 同步控制
- 基于 DAGSync 提供的同步基础设施，增强对 SSBUffer 的控制

### 6.3 协同工作模式

三个 Pass 通过同步操作形成协同工作：

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Pass 协作流程                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DAGSync Pass (第一)                                                │
│  ├─ 分析 Vector↔Cube 数据依赖                                       │
│  ├─ 插入 SyncBlockSetOp/SyncBlockWaitOp (flag=0,1,2...)            │
│  ├─ 插入 CopyOp/FixpipeOp 完成数据搬运                               │
│  └─ 输出：带有基础同步的 IR                                          │
│                                                                     │
│  DAGScope Pass (第二)                                               │
│  ├─ 读取 DAGSync 插入的同步操作                                     │
│  ├─ 查找 SetOp/WaitOp 作为锚点（findNextSyncBlockSetAfter）         │
│  ├─ 在锚点后插入额外的同步（双向同步增强）                            │
│  ├─ 收集 flag 分配新 ID（findNextAvailableFlag）                     │
│  ├─ 查找对应 flag 的 WaitOp（findWaitOpInRegionWithFlag）            │
│  ├─ 封装 AIV Scope 和 AIC Scope                                     │
│  └─ 输出：同步增强 + 核心分离的 IR                                   │
│                                                                     │
│  DAGSSBuffer Pass (第三)                                            │
│  ├─ 遍历 DAGSync/DAGScope 插入的 SyncBlockWaitOp                   │
│  ├─ 收集包含同步的 for 循环（processedScopes2）                      │
│  ├─ 基于核心类型（AIC/AIV）插入 PIPE_S 同步控制                       │
│  ├─ 控制 Shared Storage Buffer 的读写顺序                           │
│  └─ 输出：带有 SSBUffer 控制的 IR                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.4 具体使用示例

#### 示例：完整的 Flash Attention 同步链

```mlir
// 原始 TTIR
scf.for %i = ... iter_args(%buffer = %init) -> (...) {
  %scores = tt.dot %q, %k, %buffer  // CUBE 操作
  scf.yield %scores
}

// ======================================
// 经过 DAGSync Pass 后（第一遍）
// ======================================
scf.for %i = ... iter_args(%buffer = %init) -> (...) {
  // 数据准备（VECTOR）
  %k_load = tt.load %ptr_k
  %k_trans = tt.trans %k_load

  // AIV→AIC 同步（DAGSync 插入）
  hivm.sync_block_set {flag=0, tcore_type=VECTOR,
                      set_pipe=PIPE_MTE3, wait_pipe=PIPE_MTE1}
  hivm.sync_block_wait {flag=0, tcore_type=CUBE,
                       set_pipe=PIPE_MTE3, wait_pipe=PIPE_MTE1}

  // CUBE 计算
  %scores = tt.dot %q, %k_trans, %buffer  // CUBE

  // AIC→AIV 同步（DAGSync 插入）
  hivm.sync_block_set {flag=1, tcore_type=CUBE,
                      set_pipe=PIPE_FIX, wait_pipe=PIPE_V}
  hivm.sync_block_wait {flag=1, tcore_type=VECTOR,
                       set_pipe=PIPE_FIX, wait_pipe=PIPE_V}

  scf.yield %scores
}

// ======================================
// 经过 DAGScope Pass 后（第二遍）
// ======================================
func @_attn_fwd(...) {
  // AIV Scope
  scope.scope {
    %k_load = tt.load %ptr_k
    %k_trans = tt.trans %k_load

    // 找到 DAGSync 插入的 Set(flag=0)
    // 在其后插入新的 Set(flag=2)
    hivm.sync_block_set {flag=0, ...}
    hivm.sync_block_set {flag=2, ...}  // DAGScope 增强

    scope.return
  } {tcore_type = #hivm.tcore_type<VECTOR>}

  // AIC Scope
  scope.scope {
    // 双向同步
    hivm.sync_block_wait {flag=0, ...}  // 等待 AIV
    hivm.sync_block_wait {flag=2, ...}  // DAGScope 增强

    %scores = tt.dot %q, %k_trans, %buffer

    hivm.sync_block_set {flag=1, ...}
    scope.return
  } {tcore_type = #hivm.tcore_type<CUBE>}
}

// ======================================
// 经过 DAGSSBuffer Pass 后（第三遍）
// ======================================
func @_attn_fwd(...) {
  // 遍历 DAGSync/DAGScope 插入的所有 SyncBlockWaitOp
  // 在 AIC Scope 的 for 循环中插入 PIPE_S 控制

  // AIC Scope
  scope.scope {
    scf.for ... {
      // 在循环开头（基于 DAGSync 的 WaitOp 信息）
      hivm.sync_block_wait {flag=12, tcore_type=CUBE,
                           set_pipe=PIPE_S, wait_pipe=PIPE_S}

      %scores = tt.dot ...

      // 在循环末尾
      hivm.sync_block_set {flag=13, tcore_type=CUBE,
                          set_pipe=PIPE_S, wait_pipe=PIPE_S}

      scf.yield %scores
    }
  } {tcore_type = CUBE}
}
```

### 6.5 关键洞察

1. **基础设施提供者**: DAGSync 为整个优化流程提供基础的同步原语
   - 插入初始的 Set/Wait 对
   - 分配 flag ID
   - 建立数据搬运通道

2. **信息消费者**: DAGScope 和 DAGSSBuffer 读取并利用 DAGSync 提供的信息
   - 查找同步操作作为锚点
   - 基于现有 flag 继续分配
   - 增强同步机制

3. **分层增强**: 每个 Pass 在 DAGSync 基础上添加新的价值
   - DAGSync: 基础同步 + 数据搬运
   - DAGScope: 核心分配 + 双向同步增强
   - DAGSSBuffer: 存储控制 + PIPE_S 同步

4. **松耦合设计**: 三个 Pass 通过同步操作松耦合，各司其职
   - DAGSync 不依赖后续 Pass
   - DAGScope 依赖 DAGSync 的同步操作
   - DAGSSBuffer 依赖前两者的同步基础设施

---

## 7. 总结

DAGSync.cpp 是昇腾 NPU Triton 后端的关键 Pass，其核心价值在于：

1. **自动分析依赖**: 通过 DAG 分析操作间的数据依赖
2. **智能同步**: 在 Vector↔Cube 边界自动插入同步指令
3. **数据格式转换**: 处理不同核心间的数据布局差异
4. **循环特殊处理**: 正确处理循环迭代参数的跨核心传递
5. **基础设施**: 为后续 Pass 提供可扩展的同步原语

理解这个文件需要：
- 熟悉 MLIR 的遍历和转换模式
- 理解昇腾 NPU 的 Vector/Cube 架构
- 掌握数据搬运 (fixpipe, copy) 和同步 (sync_block_set/wait) 的使用场景
- 理解三个 Pass 之间的协作关系

---

*文档生成日期: 2026-03-15*
*分析源码版本: DAGSync.cpp (约960行)*
