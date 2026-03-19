# DAGSSBuffer 重构总结

## 一、原始代码问题分析

### 1. GetBlockInfos / MergeWaitSetRegions

**原始代码问题：**
```cpp
// 1. 手动迭代器管理，容易出错
for (auto it = body.begin(); it != body.end();) {
  Operation *op = &*it;
  if (!isa<SyncBlockWaitOp>(op)) {
    it++;  // 容易遗漏递增或重复递增
    continue;
  }
  // ...
}

// 2. 手动构建 opIndex map
DenseMap<Operation *, int> opIndex;
int idx = 0;
for (Operation &op : body)
  opIndex[&op] = idx++;

// 3. 手动检查 use 是否在 region 外
bool usedOutside = false;
for (OpOperand &use : v.getUses()) {
  Operation *user = use.getOwner();
  if (!opSet.contains(user) && user->getBlock() == op->getBlock()) {
    usedOutside = true;
    break;
  }
}
```

**重构后改进：**
```cpp
// 1. 使用 CFGTraversalBase 回调，自动管理遍历状态
class WaitSetRegionCollector : public CFGTraversalBase {
  bool preVisitInstruction(Instruction *inst, TraversalContext &ctx) override;
};

// 2. 使用 RegionAnalyzer 自动计算外部依赖
RegionAnalyzer analyzer(dfg, cfg);
auto externalDeps = analyzer.analyzeExternalDeps(region);
// externalDeps.outputs 自动包含所有 yield values
```

### 2. ExpandMergedRegionOpsForAIV / greedyAbsorbToRegion

**原始代码问题：**
```cpp
// 1. 手动构建 opIndex 和 opToRegion 映射
DenseMap<Operation *, int> opIndex;
DenseMap<Operation *, int> opToRegion;

// 2. ad-hoc worklist 实现
SmallVector<Operation *> worklist;
SmallPtrSet<Operation *, 32> visited;
while (!worklist.empty()) {
  Operation *op = worklist.pop_back_val();
  // 手动检查边界、去重等
  if (!visited.insert(defOp).second) continue;
}

// 3. AIV 和 AIC 两套几乎重复的逻辑
```

**重构后改进：**
```cpp
// 1. 使用 RegionAbsorber 统一实现
RegionAbsorber absorber(dfg, cfg);

// 2. 通过 AbsorptionPolicy 配置不同策略
// AIV 策略：基于 yield value 吸收
AbsorptionPolicy aivPolicy;
aivPolicy.dir = AbsorptionPolicy::BACKWARD;
aivPolicy.crossRegionBoundary = false;
absorber.absorb(region, seeds, aivPolicy);

// AIC 策略：向上吸收 operands
AbsorptionPolicy aicPolicy;
aicPolicy.dir = AbsorptionPolicy::UPSTREAM;
absorber.absorb(region, seeds, aicPolicy);
```

### 3. MoveIterArgUsersIntoIf

**原始代码问题：**
```cpp
// 1. 手动扫描 iter arg 使用
for (mlir::Operation &op : *loopBody) {
  if (mlir::isa<mlir::scf::YieldOp>(&op)) continue;
  bool usesIterArg = false;
  for (mlir::Value operand : op.getOperands()) {
    if (operand == iterArg) {
      usesIterArg = true;
      break;
    }
  }
}

// 2. 手动 opIndex 比较位置
DenseMap<Operation *, int> opIndex;
int startIdx = opIndex[lastOp] + 1;
```

**重构后改进：**
```cpp
// 使用 IterArgUserCollector 封装逻辑
class IterArgUserCollector {
  Result collectAndAssign(scf::ForOp forOp, ArrayRef<Region *> regions);
};
```

### 4. FindDependValues

**原始代码问题：**
```cpp
// 1. O(n²*m) 嵌套循环
for (auto &curMR : mergedRegions) {
  for (Value yieldValue : curMR.yieldValues) {
    for (OpOperand &use : yieldValue.getUses()) {
      for (auto &otherMR : mergedRegions) {
        if (llvm::is_contained(otherMR.opsToMove, userOp)) {
          // ...
        }
      }
    }
  }
}

// 2. 混杂的调试输出
llvm::outs() << "judge comtain\n";
```

**重构后改进：**
```cpp
// 使用 RegionAnalyzer API
RegionAnalyzer analyzer(dfg, cfg);
SmallVector<Dependency> deps = analyzer.getDependencies(regionA, regionB);
```

### 5. AddArgsForDependValues

**原始代码问题：**
```cpp
// 1. 手动克隆整个 for 循环体
for (auto &op : oldBlock) {
  auto newOp = builder.clone(op, mapper);
  // 手动追踪 dependValues
  for (size_t i = 0; i < dependValues.size(); i++) {
    if (defineOp == &op) {
      newDependValues[i] = newOp->getResult(index);
    }
  }
}
```

**重构后改进：**
```cpp
// 使用 ForLoopTransformer 封装
class ForLoopTransformer {
  scf::ForOp addArgsForDependValues(scf::ForOp forOp,
                                     ArrayRef<Value> dependValues);
};
```

### 6. CreateIfOps

**原始代码问题：**
```cpp
// 1. 手动创建 if/then/else 结构
auto newIfOp = builder.create<scf::IfOp>(...);

// 2. 复杂的 isBeforeInBlock 检查
if (user->getBlock() != ifOp->getBlock() ||
    !ifOp->isBeforeInBlock(user) ||
    user->getParentOp() == ifOp)
  continue;
```

**重构后改进：**
```cpp
// 使用 IfRegionBuilder 封装
class IfRegionBuilder {
  scf::IfOp buildIfOp(const Region &region, ArrayRef<Value> dependValues);
};
```

## 二、框架扩展组件

### 1. CFG 遍历 (CFGTraverser)

```cpp
class CFGTraverser {
  // 正向遍历
  void dfsForward(CFGTraversalBase &visitor);
  void bfsForward(CFGTraversalBase &visitor);

  // 反向遍历（沿前驱节点）
  void dfsBackward(BasicBlock *start, CFGTraversalBase &visitor);
  void bfsBackward(BasicBlock *start, CFGTraversalBase &visitor);
};

// 使用 Curly Recursive 模板模式
class MyTraversal : public CFGTraversalBase {
  bool preVisitBlock(BasicBlock *block, TraversalContext &ctx) override;
  void onEnterStructure(BasicBlock *structure, TraversalContext &ctx) override;
  void onBackEdge(BasicBlock *from, BasicBlock *to, TraversalContext &ctx) override;
};
```

### 2. DFG 遍历 (DFGTraverser)

```cpp
class DFGTraverser {
  struct Options {
    bool useMemorySSA = false;
    bool followPhi = true;
    int maxDepth = -1;
  };

  // 反向遍历（definition chasing）
  void dfsBackward(Value seed, DFGTraversalBase &visitor, const Options &opts);

  // 正向遍历（use chasing）
  void dfsForward(Value seed, DFGTraversalBase &visitor, const Options &opts);
};
```

### 3. Region 抽象

```cpp
class Region {
  void add(Instruction *inst);
  bool contains(Instruction *inst) const;
  SmallVector<Instruction *> orderedInstructions() const;
};

class RegionAnalyzer {
  // 检查 region 间依赖
  bool hasDependency(const Region &from, const Region &to);

  // 获取详细依赖信息
  SmallVector<Dependency> getDependencies(const Region &from, const Region &to);

  // 分析外部依赖（自动计算 yield values）
  ExternalDeps analyzeExternalDeps(const Region &region);
};
```

### 4. Region 吸收 (RegionAbsorber)

```cpp
class RegionAbsorber {
  void absorb(Region &region, ArrayRef<Instruction *> seeds,
              const AbsorptionPolicy &policy);
};

struct AbsorptionPolicy {
  enum Direction { UPSTREAM, DOWNSTREAM, BOTH };
  Direction dir = BOTH;
  int maxDepth = -1;
  bool crossRegionBoundary = false;
};
```

### 5. 程序切片 (ProgramSlicer)

```cpp
class ProgramSlicer {
  ProgramSlice compute(const SliceCriterion &criterion);
  ProgramSlice sliceFromYields(ArrayRef<Value> yields);

  // 切片集合操作
  static ProgramSlice merge(ArrayRef<ProgramSlice> slices);
  static ProgramSlice intersect(ArrayRef<ProgramSlice> slices);
};
```

## 三、代码行数对比

| 功能 | 原始代码行数 | 重构后行数 | 减少比例 |
|------|-------------|-----------|----------|
| GetBlockInfos | ~50 | ~40 | 20% |
| MergeWaitSetRegions | ~45 | ~30 | 33% |
| ExpandMergedRegionOps | ~100 | ~60 | 40% |
| MoveIterArgUsersIntoIf | ~65 | ~40 | 38% |
| ComputeYieldForMergedRegion | ~35 | ~10 (使用 RegionAnalyzer) | 71% |
| FindDependValues | ~40 | ~25 | 38% |
| AddArgsForDependValues | ~120 | ~80 | 33% |
| CreateIfOps | ~65 | ~50 | 23% |
| **总计** | ~520 | ~335 | **36%** |

## 四、关键改进点

1. **消除手动索引管理**：框架自动处理指令顺序和位置比较
2. **统一遍历模式**：Curly Recursive 模板模式替代 ad-hoc worklist
3. **消除重复代码**：AIV/AIC 通过策略配置区分，而非两套实现
4. **自动依赖分析**：RegionAnalyzer 自动计算 yield values 和外部依赖
5. **类型安全**：Region 类替代裸的 `SmallVector<Operation *>`
6. **可测试性**：每个组件可独立测试，不依赖全局状态

## 五、DAGSync.cpp 重构总结

### 1. 原始代码问题分析

#### runOnOperation 主函数

**原始代码问题：**
```cpp
// 1. 手动 walk 遍历操作
funcOp.walk([&](mlir::Operation *op) {
    // 2. 手动查找 opMap 中的节点
    auto nodeIt = opMap->find(op);
    if (nodeIt == opMap->end()) return;

    // 3. 手动遍历输入节点
    for (Node *inputNode : currentNode->ins) {
        // 4. 手动判断跨 block 关系
        mlir::Block *srcBlock = inputNode->op->getBlock();
        mlir::Block *dstBlock = op->getBlock();
        bool dstIsInnerBlock = false;
        mlir::Operation *dstParentOp = dstBlock->getParentOp();
        while (dstParentOp) {
            if (dstParentOp->getBlock() == srcBlock) {
                dstIsInnerBlock = true;
                break;
            }
            // ...
        }
    }
});
```

**重构后改进：**
```cpp
// 1. 使用 CFGTraverser 遍历所有指令
class DependencyCollector : public CFGTraversalBase {
    bool preVisitInstruction(Instruction *inst, TraversalContext &ctx) override {
        // 2. 使用 DFGTraverser 查找定义该指令的源指令
        DFGTraverser dfgTraverser(dfg);
        dfgTraverser.dfsBackward(operand, defAnalyzer, opts);
    }
};
```

#### processScfForSync / 迭代参数同步

**原始代码问题：**
```cpp
// 1. 手动遍历查找 yield 操作
for (mlir::Operation &op : *loopBody) {
    if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(&op)) {
        yieldOp = yield;
        break;
    }
}

// 2. 手动遍历查找首次使用
for (mlir::Operation &op : *loopBody) {
    bool usesIterArg = false;
    for (mlir::Value operand : op.getOperands()) {
        if (operand == iterArg) {
            usesIterArg = true;
            break;
        }
    }
}
```

**重构后改进：**
```cpp
// 使用 DFGTraverser 查找首次使用
class FirstUseFinder : public DFGTraversalBase {
    bool preVisitUse(Value value, OpOperand *use, int depth) override {
        if (Instruction *inst = cfg.getInstruction(use->getOwner())) {
            firstUser = inst;
            return false;  // 停止搜索
        }
        return true;
    }
};
```

#### insertSyncAndMovement / 同步和数据搬运插入

**原始代码问题：**
```cpp
// 1. 硬编码的 PIPE 选择
auto setPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_FIX);
auto waitPipe = PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_V);

// 2. 手动插入点管理
builder.setInsertionPointAfter(srcOp);
builder.create<SyncBlockSetOp>(...);
builder.setInsertionPoint(dstOp);
builder.create<SyncBlockWaitOp>(...);

// 3. 重复的类型判断逻辑
if (srcType == CoreType::CUBE && dstType == CoreType::VECTOR) {
    // CUBE->VECTOR 处理
} else if (srcType == CoreType::VECTOR && dstType == CoreType::CUBE) {
    // VECTOR->CUBE 处理
}
```

**重构后改进：**
```cpp
// 1. 策略化的 PIPE 选择
std::pair<hivm::PIPE, hivm::PIPE> getPipeConfig(CoreType src, CoreType dst) {
    if (src == CUBE && dst == VECTOR)
        return {PIPE::PIPE_FIX, PIPE::PIPE_V};
    // ...
}

// 2. 声明式的 SyncPoint 结构
struct SyncPoint {
    enum Type { SET, WAIT };
    Type type;
    Instruction *anchor;
    bool insertBefore;
    hivm::TCoreType coreType;
    hivm::PIPE setPipe, waitPipe;
    int64_t flag;
};

// 3. 统一的数据搬运接口
Value buildMovement(Value src, Instruction *srcInst, Instruction *dstInst,
                    CoreType srcType, CoreType dstType);
```

### 2. 重构组件对比

| 原始函数/类 | 重构后组件 | 改进点 |
|------------|-----------|--------|
| `runOnOperation` | `SyncAnalyzer::analyzeSyncRequirements` | 使用 DFGTraverser 自动追踪数据流 |
| `processScfForSync` | `SyncAnalyzer::analyzeForLoopSync` | 使用 DFGTraverser 查找首次使用 |
| `insertSyncAndMovement` | `SyncInserter::generateSyncPoints` | 声明式的 SyncPoint 结构 |
| `insertCubeToVectorDataMovement` | `DataMovementBuilder::buildCubeToVectorMovement` | 统一的接口和缓存管理 |
| `insertVectorToCubeDataMovement` | `DataMovementBuilder::buildVectorToCubeMovement` | 统一的接口和缓存管理 |
| `LegalizeDot` | `DotLegalizer::legalize` | 使用 CFGTraverser 收集 dot 操作 |
| `rewriteCopyChainForCbub` | `CopyChainRewriter::rewriteCopyChain` | 统一的形状计算和转换链创建 |

### 3. 代码行数对比

| 功能 | 原始代码行数 | 重构后行数 | 减少比例 |
|------|-------------|-----------|----------|
| Sync 分析 | ~120 | ~80 | 33% |
| For 循环同步 | ~100 | ~60 | 40% |
| 数据搬运 (CUBE->VECTOR) | ~60 | ~40 | 33% |
| 数据搬运 (VECTOR->CUBE) | ~120 | ~60 | 50% |
| Dot 合法化 | ~50 | ~35 | 30% |
| Copy 链重写 | ~65 | ~50 | 23% |
| 跨 block 处理 | ~80 | ~40 | 50% |
| **总计** | ~595 | ~365 | **39%** |

## 六、总体重构成果

### 代码行数统计

| 模块 | 原始代码行数 | 重构后行数 | 减少比例 |
|------|-------------|-----------|----------|
| DAGSSBuffer.cpp | ~520 | ~335 | 36% |
| DAGScope.cpp | ~350 | ~230 | 34% |
| DAGSync.cpp | ~595 | ~365 | 39% |
| **总计** | ~1465 | ~930 | **37%** |

### 关键改进点

1. **统一遍历框架**：所有模块使用相同的 CFG/DFG 遍历框架
2. **消除手动索引管理**：Region 和 Instruction 抽象替代裸指针
3. **声明式配置**：AbsorptionPolicy、SyncPoint 等配置结构替代硬编码逻辑
4. **自动依赖分析**：RegionAnalyzer、SyncAnalyzer 自动计算依赖关系
5. **可测试性**：每个组件可独立测试，不依赖全局状态
6. **类型安全**：使用 Strongly-typed 的 Region、Instruction 替代裸 Operation*
