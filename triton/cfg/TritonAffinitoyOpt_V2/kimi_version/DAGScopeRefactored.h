/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * 重构后的 DAGScope 组件 - 使用新的 CFG/DFG 框架
 */

#ifndef TRITON_AFFINITY_OPT_DAG_SCOPE_REFACTORED_H
#define TRITON_AFFINITY_OPT_DAG_SCOPE_REFACTORED_H

#include "TritonToGraph/GraphAnalysis.h"
#include "TritonToGraph/ControlFlowGraph.h"
#include "TritonToGraph/DataflowGraph.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace triton;
using namespace cfg;

//===----------------------------------------------------------------------===//
// Scope Partitioning (重构 encapsulateWithScope / SplitScope)
//===----------------------------------------------------------------------===//

/// ScopePartitioner - 将函数分区到 AIV/AIC scope
///
/// 原始代码问题：
/// 1. encapsulateWithScope: 手动遍历 block 找依赖于函数参数的操作
/// 2. SplitScope: 手动收集 opsToMove，复杂的分类逻辑
/// 3. 手动维护 parentFor 映射
///
/// 重构后改进：
/// 1. 使用 DFGTraverser 自动追踪数据流依赖
/// 2. 使用 Region 抽象管理指令集合
/// 3. 自动处理嵌套结构上下文
class ScopePartitioner {
public:
  struct PartitionResult {
    Region aivOps;
    Region aicOps;
    // 需要在两个 scope 中复制的操作（如 for/if 结构）
    Region sharedOps;
  };

  explicit ScopePartitioner(DataFlowGraph &dfg, ControlFlowGraph &cfg,
                            AffinityDAG::Graph &dagGraph)
      : dfg(dfg), cfg(cfg), dagGraph(dagGraph) {}

  /// 分析函数并分区
  PartitionResult partition(triton::FuncOp funcOp);

  /// 使用 DFG 从种子值收集相关操作到指定 region
  void collectRelatedOps(Value seed, Region &targetRegion,
                         AffinityDAG::CoreType targetType);

private:
  /// 判断操作应该属于哪个 scope
  AffinityDAG::CoreType classifyOperation(Operation *op);

  DataFlowGraph &dfg;
  ControlFlowGraph &cfg;
  AffinityDAG::Graph &dagGraph;
};

//===----------------------------------------------------------------------===//
// Sync Operation Insertion (重构 addSyncOpsForBufferWait 及相关函数)
//===----------------------------------------------------------------------===//

/// SyncOperationInserter - 在 AIV/AIC scope 之间插入同步操作
///
/// 原始代码问题：
/// 1. processFixpipeOpsInAIC: 手动 walk 查找 fixpipe
/// 2. processToMemrefOpsInAIV: 手动 walk 查找 to_memref
/// 3. 手动查找下一个/上一个 sync 操作
/// 4. 复杂的插入点计算
///
/// 重构后改进：
/// 1. 使用 CFGTraverser 遍历 scope
/// 2. 使用 RegionAnalyzer 识别数据依赖边界
/// 3. 自动计算最佳插入点
class SyncOperationInserter {
public:
  struct SyncPoint {
    enum Type { WAIT, SET };
    Type type;
    Instruction *anchor;      // 插入参考点
    bool insertBefore;        // true=在 anchor 前插入, false=后插入
    hivm::TCoreType coreType;
    hivm::PIPE setPipe;
    hivm::PIPE waitPipe;
    int64_t flag;
  };

  explicit SyncOperationInserter(DataFlowGraph &dfg, ControlFlowGraph &cfg)
      : dfg(dfg), cfg(cfg) {}

  /// 为 buffer 等待插入同步操作
  void insertSyncForBufferWait(triton::FuncOp funcOp,
                               Region *aicRegion,
                               Region *aivRegion);

  /// 生成 AIC scope 的同步点（围绕 fixpipe）
  SmallVector<SyncPoint> generateAICSyncPoints(Region *aicRegion,
                                               Region *aivRegion);

  /// 生成 AIV scope 的同步点（围绕 to_memref）
  SmallVector<SyncPoint> generateAIVSyncPoints(Region *aivRegion,
                                               Region *aicRegion);

  /// 执行同步点插入
  void applySyncPoints(ArrayRef<SyncPoint> syncPoints, OpBuilder &builder);

private:
  /// 查找下一个可用的 flag ID
  int64_t findNextAvailableFlag(triton::FuncOp funcOp);

  /// 创建 sync_block_set 操作
  hivm::SyncBlockSetOp createSetOp(OpBuilder &builder, Location loc,
                                   hivm::TCoreType coreType,
                                   hivm::PIPE setPipe,
                                   hivm::PIPE waitPipe,
                                   int64_t flag);

  /// 创建 sync_block_wait 操作
  hivm::SyncBlockWaitOp createWaitOp(OpBuilder &builder, Location loc,
                                     hivm::TCoreType coreType,
                                     hivm::PIPE setPipe,
                                     hivm::PIPE waitPipe,
                                     int64_t flag);

  DataFlowGraph &dfg;
  ControlFlowGraph &cfg;
};

//===----------------------------------------------------------------------===//
// Scope Builder - 构建最终的 scope.scope 操作
//===----------------------------------------------------------------------===//

/// ScopeBuilder - 构建 scope 操作
///
/// 原始代码问题：
/// 1. 手动移动操作到 scope body
/// 2. 手动创建 return op
/// 3. 手动设置 tcore_type 属性
///
/// 重构后改进：
/// 1. 使用 Region 抽象批量移动
/// 2. 自动处理外部依赖（参数/返回值）
class ScopeOpBuilder {
public:
  explicit ScopeOpBuilder(OpBuilder &builder, ControlFlowGraph &cfg)
      : builder(builder), cfg(cfg) {}

  /// 从 region 构建 scope 操作
  scope::ScopeOp buildScope(Region &ops, hivm::TCoreType coreType,
                            Location loc);

  /// 构建 AIV scope（VECTOR）
  scope::ScopeOp buildAIVScope(Region &aivOps, Location loc) {
    return buildScope(aivOps, hivm::TCoreType::VECTOR, loc);
  }

  /// 构建 AIC scope（CUBE）
  scope::ScopeOp buildAICScope(Region &aicOps, Location loc) {
    return buildScope(aicOps, hivm::TCoreType::CUBE, loc);
  }

private:
  OpBuilder &builder;
  ControlFlowGraph &cfg;
};

//===----------------------------------------------------------------------===//
// High-Level Orchestrator - 替换 DAGScopePass::runOnOperation
//===----------------------------------------------------------------------===//

/// ScopeOptimizationDriver - Scope 优化主驱动
///
/// 整合上述所有组件，提供清晰的优化流程
class ScopeOptimizationDriver {
public:
  ScopeOptimizationDriver(DataFlowGraph &dfg, ControlFlowGraph &cfg,
                          AffinityDAG::Graph &dagGraph)
      : partitioner(dfg, cfg, dagGraph),
        syncInserter(dfg, cfg),
        scopeBuilder(builder, cfg),
        dfg(dfg), cfg(cfg) {}

  /// 执行完整的 scope 优化
  void run(ModuleOp module, OpBuilder &builder);

private:
  /// 步骤1: 移动所有 alloc 操作到函数入口
  void hoistAllocs(triton::FuncOp funcOp);

  /// 步骤2: 分区操作到 AIV/AIC
  void partitionScopes(triton::FuncOp funcOp);

  /// 步骤3: 插入同步操作
  void insertSyncOps(triton::FuncOp funcOp);

  /// 步骤4: 构建最终的 scope 操作
  void buildFinalScopes(triton::FuncOp funcOp);

  ScopePartitioner partitioner;
  SyncOperationInserter syncInserter;
  ScopeOpBuilder scopeBuilder;
  DataFlowGraph &dfg;
  ControlFlowGraph &cfg;
  OpBuilder builder;
};

#endif // TRITON_AFFINITY_OPT_DAG_SCOPE_REFACTORED_H
