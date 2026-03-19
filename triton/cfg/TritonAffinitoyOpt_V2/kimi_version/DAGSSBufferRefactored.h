/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * 重构后的 DAGSSBuffer 组件 - 使用新的 CFG/DFG 框架
 */

#ifndef TRITON_AFFINITY_OPT_DAG_SS_BUFFER_REFACTORED_H
#define TRITON_AFFINITY_OPT_DAG_SS_BUFFER_REFACTORED_H

#include "TritonToGraph/GraphAnalysis.h"
#include "TritonToGraph/ControlFlowGraph.h"
#include "TritonToGraph/DataflowGraph.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace triton;
using namespace cfg;

//===----------------------------------------------------------------------===//
// Wait-Set Region Detection (重构 GetBlockInfos)
//===----------------------------------------------------------------------===//

/// WaitSetRegion - 等待-设置区域
struct WaitSetRegion {
  Instruction *waitInst;
  Instruction *setInst;
  Region ops;
  bool hasCopyOrFixpipe = false;
};

/// WaitSetRegionCollector - 使用 CFG 遍历收集 Wait-Set Region
class WaitSetRegionCollector : public CFGTraversalBase {
public:
  explicit WaitSetRegionCollector(ControlFlowGraph &cfg) : cfg(cfg) {}

  // 收集指定块中的 wait-set regions
  SmallVector<WaitSetRegion> collect(BasicBlock *block);

  // 实现 CFGTraversalBase 回调
  bool preVisitInstruction(Instruction *inst, TraversalContext &ctx) override;

private:
  ControlFlowGraph &cfg;
  SmallVector<WaitSetRegion> regions_;
  Instruction *currentWait_ = nullptr;
  Instruction *lastSet_ = nullptr;
  SmallVector<Instruction *> opsInCurrentRegion_;
  int setCount_ = 0;
};

/// WaitSetRegionMerger - 合并相邻的 wait-set regions
class WaitSetRegionMerger {
public:
  struct MergedRegion {
    SmallVector<WaitSetRegion *> sourceRegions;
    Region ops;
    SmallVector<Value> yieldValues;
    SmallVector<Type> resultTypes;
  };

  explicit WaitSetRegionMerger(DataFlowGraph &dfg) : dfg(dfg) {}

  // 合并 regions（重构 MergeWaitSetRegions）
  SmallVector<MergedRegion> merge(ArrayRef<WaitSetRegion> regions);

private:
  void computeYieldValues(MergedRegion &mr);
  DataFlowGraph &dfg;
};

//===----------------------------------------------------------------------===//
// Region Expansion (重构 ExpandMergedRegionOpsForAIV/AIC)
//===----------------------------------------------------------------------===//

/// RegionExpander - 使用 DFG 遍历扩展 region
class RegionExpander {
public:
  explicit RegionExpander(DataFlowGraph &dfg, ControlFlowGraph &cfg)
      : dfg(dfg), cfg(cfg), absorber(dfg, cfg) {}

  // 为 AIV 扩展 region（基于 yield value）
  void expandForAIV(Region &region, scf::ForOp forOp,
                    ArrayRef<Value> yieldValues);

  // 为 AIC 扩展 region（向上吸收 operand）
  void expandForAIC(Region &region, scf::ForOp forOp);

private:
  DataFlowGraph &dfg;
  ControlFlowGraph &cfg;
  RegionAbsorber absorber;
};

//===----------------------------------------------------------------------===//
// Iter Arg Users Processing (重构 MoveIterArgUsersIntoIf)
//===----------------------------------------------------------------------===//

/// IterArgUserCollector - 收集 iter arg 的使用并分配到对应 region
class IterArgUserCollector {
public:
  struct Result {
    DenseMap<BlockArgument, Region *> iterArgToRegion;
    SmallVector<Instruction *> usersToMove;
  };

  explicit IterArgUserCollector(DataFlowGraph &dfg, ControlFlowGraph &cfg)
      : dfg(dfg), cfg(cfg) {}

  // 收集 iter arg 用户并分配到 regions
  Result collectAndAssign(scf::ForOp forOp,
                          ArrayRef<Region *> regions);

private:
  DataFlowGraph &dfg;
  ControlFlowGraph &cfg;
};

//===----------------------------------------------------------------------===//
// Dependency Analysis (重构 FindDependValues)
//===----------------------------------------------------------------------===//

/// InterRegionDependencyAnalyzer - 使用 DFG 分析 region 间依赖
class InterRegionDependencyAnalyzer {
public:
  struct Dependency {
    Region *fromRegion;
    Region *toRegion;
    Value value;
    Instruction *defInst;
    Instruction *useInst;
  };

  explicit InterRegionDependencyAnalyzer(DataFlowGraph &dfg,
                                          ControlFlowGraph &cfg)
      : dfg(dfg), cfg(cfg) {}

  // 查找所有 region 间的依赖值（重构 FindDependValues）
  SmallVector<Value> findDependentValues(ArrayRef<Region> regions);

  // 获取详细的依赖信息
  SmallVector<Dependency> analyze(ArrayRef<Region> regions);

private:
  DataFlowGraph &dfg;
  ControlFlowGraph &cfg;
};

//===----------------------------------------------------------------------===//
// For Loop Transformation (重构 AddArgsForDependValues)
//===----------------------------------------------------------------------===//

/// ForLoopTransformer - 使用框架 API 转换 for 循环
class ForLoopTransformer {
public:
  explicit ForLoopTransformer(DataFlowGraph &dfg, ControlFlowGraph &cfg,
                               ModuleOp module)
      : dfg(dfg), cfg(cfg), module(module) {}

  // 为依赖值添加新的迭代参数
  scf::ForOp addArgsForDependValues(scf::ForOp forOp,
                                    ArrayRef<Value> dependValues,
                                    ArrayRef<Region *> regions);

private:
  Value createInitTensor(Type type, Location loc, OpBuilder &builder);

  DataFlowGraph &dfg;
  ControlFlowGraph &cfg;
  ModuleOp module;
};

//===----------------------------------------------------------------------===//
// If Region Builder (重构 CreateIfOps)
//===----------------------------------------------------------------------===//

/// IfRegionBuilder - 构建 if region
class IfRegionBuilder {
public:
  struct IfRegion {
    Region thenRegion;
    SmallVector<Value> yieldValues;
    SmallVector<Type> resultTypes;
  };

  explicit IfRegionBuilder(DataFlowGraph &dfg, ControlFlowGraph &cfg,
                            OpBuilder &builder)
      : dfg(dfg), cfg(cfg), builder(builder) {}

  // 从 region 构建 if（重构 CreateIfOps）
  scf::IfOp buildIfOp(const Region &region,
                      ArrayRef<Value> dependValues);

  // 计算 else branch 的 yield values（重构 ComputeElseYieldValues）
  SmallVector<Value> computeElseYields(scf::ForOp forOp,
                                        const IfRegion &ifRegion,
                                        ArrayRef<Value> dependValues);

private:
  DataFlowGraph &dfg;
  ControlFlowGraph &cfg;
  OpBuilder &builder;
};

#endif // TRITON_AFFINITY_OPT_DAG_SS_BUFFER_REFACTORED_H
