/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * 重构后的 DAGSync 组件 - 使用新的 CFG/DFG 框架
 */

#ifndef TRITON_AFFINITY_OPT_DAG_SYNC_REFACTORED_H
#define TRITON_AFFINITY_OPT_DAG_SYNC_REFACTORED_H

#include "TritonToGraph/GraphAnalysis.h"
#include "TritonToGraph/ControlFlowGraph.h"
#include "TritonToGraph/DataflowGraph.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace triton;
using namespace cfg;

//===----------------------------------------------------------------------===//
// Sync Analysis - 分析需要同步的跨类型依赖
//===----------------------------------------------------------------------===//

/// CrossCoreDependency - 跨核心类型依赖
struct CrossCoreDependency {
  Instruction *srcInst;
  Instruction *dstInst;
  AffinityDAG::CoreType srcType;
  AffinityDAG::CoreType dstType;
  Value value;  // 被传递的值
  bool isCrossBlock;
  bool dstIsInnerBlock;
};

/// SyncAnalyzer - 使用 DFG 分析需要同步的依赖
///
/// 原始代码问题：
/// 1. 手动遍历 DAG 节点检查依赖
/// 2. 手动判断跨 block 关系
/// 3. 重复的分类逻辑（VECTOR <-> CUBE）
///
/// 重构后改进：
/// 1. 使用 DFGTraverser 自动追踪数据流
/// 2. 使用 CFGTraverser 判断 block 关系
/// 3. 统一返回结构化的依赖信息
class SyncAnalyzer {
public:
  explicit SyncAnalyzer(DataFlowGraph &dfg, ControlFlowGraph &cfg,
                        AffinityDAG::Graph &dagGraph)
      : dfg(dfg), cfg(cfg), dagGraph(dagGraph) {}

  /// 分析函数中所有需要同步的跨类型依赖
  SmallVector<CrossCoreDependency> analyzeSyncRequirements(triton::FuncOp funcOp);

  /// 分析单个 scf.for 循环内的同步需求
  SmallVector<CrossCoreDependency> analyzeForLoopSync(scf::ForOp forOp);

private:
  /// 判断是否需要 VECTOR <-> CUBE 同步
  bool needVectorCubeSync(AffinityDAG::CoreType src, AffinityDAG::CoreType dst);

  /// 获取指令的核心类型
  AffinityDAG::CoreType getCoreType(Instruction *inst);

  /// 检查是否跨 block 依赖
  bool isCrossBlockDependency(Instruction *src, Instruction *dst,
                              bool &dstIsInner);

  DataFlowGraph &dfg;
  ControlFlowGraph &cfg;
  AffinityDAG::Graph &dagGraph;
};

//===----------------------------------------------------------------------===//
// Data Movement Builder - 构建数据搬运操作
//===----------------------------------------------------------------------===//

/// DataMovementBuilder - 构建 CUBE <-> VECTOR 之间的数据搬运
///
/// 原始代码问题：
/// 1. insertCubeToVectorDataMovement: 硬编码的 fixpipe 参数
/// 2. insertVectorToCubeDataMovement: 复杂的 layout 转换逻辑
/// 3. 重复的类型检查和 memref 操作
///
/// 重构后改进：
/// 1. 统一的数据搬运接口
/// 2. 自动类型推导和内存空间管理
/// 3. 可配置的数据搬运策略
class DataMovementBuilder {
public:
  struct MovementConfig {
    hivm::FixpipeDMAMode dmaMode = hivm::FixpipeDMAMode::NZ2ND;
    hivm::DataLayout srcLayout = hivm::DataLayout::ND;
    hivm::DataLayout dstLayout = hivm::DataLayout::ND;
  };

  explicit DataMovementBuilder(OpBuilder &builder, ModuleOp module,
                               AffinityDAG::Graph &dagGraph)
      : builder(builder), module(module), dagGraph(dagGraph) {}

  /// CUBE -> VECTOR 数据搬运（使用 fixpipe）
  Value buildCubeToVectorMovement(Value srcValue, Instruction *dstInst,
                                  const MovementConfig &config);

  /// VECTOR -> CUBE 数据搬运（使用 copy）
  Value buildVectorToCubeMovement(Value srcValue, Instruction *dstInst,
                                  const MovementConfig &config);

  /// 自动根据类型选择搬运方式
  Value buildMovement(Value srcValue, Instruction *srcInst, Instruction *dstInst,
                      AffinityDAG::CoreType srcType, AffinityDAG::CoreType dstType);

private:
  /// 获取或创建 memref.alloc
  Value getOrCreateAllocation(Type tensorType, hivm::AddressSpace addrSpace,
                              Location loc);

  /// 创建 fixpipe 操作
  void createFixpipeOp(Value src, Value dst, Location loc,
                       const MovementConfig &config);

  /// 创建 copy 操作链
  void createCopyOpChain(Value src, Value dst, Location loc,
                         const MovementConfig &config);

  /// 计算 reshape 后的形状
  std::optional<SmallVector<int64_t, 4>> computeReshapedShape(
      RankedTensorType tensorType);

  OpBuilder &builder;
  ModuleOp module;
  AffinityDAG::Graph &dagGraph;
  DenseMap<std::pair<Type, hivm::AddressSpace>, Value> allocationCache;
};

//===----------------------------------------------------------------------===//
// Sync Operation Inserter - 插入同步操作
//===----------------------------------------------------------------------===//

/// SyncInserter - 插入 sync_block_set/sync_block_wait 操作
///
/// 原始代码问题：
/// 1. insertSyncAndMovement: 复杂的参数计算
/// 2. 手动管理插入点（src 后/dst 前）
/// 3. 硬编码的 PIPE 选择
///
/// 重构后改进：
/// 1. 基于 SyncPoint 的声明式插入
/// 2. 自动计算最佳插入位置
/// 3. 策略化的 PIPE 选择
class SyncInserter {
public:
  struct SyncPoint {
    enum Type { SET, WAIT };
    Type type;
    Instruction *anchor;      // 插入参考点
    bool insertBefore;        // true=在 anchor 前, false=后
    hivm::TCoreType coreType;
    hivm::PIPE setPipe;
    hivm::PIPE waitPipe;
    int64_t flag;
  };

  explicit SyncInserter(OpBuilder &builder, DataFlowGraph &dfg,
                        ControlFlowGraph &cfg)
      : builder(builder), dfg(dfg), cfg(cfg) {}

  /// 为跨核心依赖生成同步点
  SmallVector<SyncPoint> generateSyncPoints(
      const CrossCoreDependency &dep, int64_t flag);

  /// 批量执行同步点插入
  void applySyncPoints(ArrayRef<SyncPoint> syncPoints);

  /// 为 scf.for 迭代参数生成同步点
  SmallVector<SyncPoint> generateForLoopSyncPoints(
      scf::ForOp forOp, ArrayRef<CrossCoreDependency> deps, int64_t startFlag);

private:
  /// 根据核心类型确定 PIPE 配置
  std::pair<hivm::PIPE, hivm::PIPE> getPipeConfig(AffinityDAG::CoreType srcType,
                                                   AffinityDAG::CoreType dstType);

  /// 创建单个 sync 操作
  void createSyncOp(const SyncPoint &sp);

  OpBuilder &builder;
  DataFlowGraph &dfg;
  ControlFlowGraph &cfg;
};

//===----------------------------------------------------------------------===//
// Cross-Block Sync Handler - 处理跨 block 同步
//===----------------------------------------------------------------------===//

/// CrossBlockSyncHandler - 处理跨 block 的同步特殊逻辑
///
/// 原始代码问题：
/// 1. insertSyncAndMovementForCrossBlock: 复杂的 parent block 查找
/// 2. 手动处理内外层 block 关系
/// 3. 特殊的插入点计算
///
/// 重构后改进：
/// 1. 使用 CFG 层级分析 API
/// 2. 统一的跨 block 同步策略
/// 3. 自动化的插入点定位
class CrossBlockSyncHandler {
public:
  explicit CrossBlockSyncHandler(ControlFlowGraph &cfg, SyncInserter &syncInserter)
      : cfg(cfg), syncInserter(syncInserter) {}

  /// 处理跨 block 同步
  void handleCrossBlockSync(const CrossCoreDependency &dep, int64_t flag);

  /// 找到最佳的 wait 插入点
  Instruction *findOptimalWaitLocation(const CrossCoreDependency &dep);

private:
  /// 检查 block 层级关系
  bool isAncestorBlock(Block *ancestor, Block *descendant);

  /// 获取 block 的嵌套深度
  int getBlockNestingDepth(Block *block);

  ControlFlowGraph &cfg;
  SyncInserter &syncInserter;
};

//===----------------------------------------------------------------------===//
// Dot Operation Legalizer - 处理 Dot 操作合法性
//===----------------------------------------------------------------------===//

/// DotLegalizer - 规范化 dot 操作的累加器
///
/// 原始代码问题：
/// 1. LegalizeDot: 手动 walk 查找 dot 操作
/// 2. 复杂的累加器零值检查
/// 3. 手动创建替换链
///
/// 重构后改进：
/// 1. 使用 CFGTraverser 收集所有 dot 操作
/// 2. 统一的累加器检查逻辑
/// 3. 自动化的 dot 替换
class DotLegalizer {
public:
  explicit DotLegalizer(OpBuilder &builder, ControlFlowGraph &cfg)
      : builder(builder), cfg(cfg) {}

  /// 处理函数中所有 dot 操作
  void legalize(triton::FuncOp funcOp);

private:
  /// 检查累加器是否为零值
  bool isZeroAccumulator(Value acc);

  /// 创建合法的 dot 操作替换
  void legalizeDotOp(triton::DotOp dotOp);

  OpBuilder &builder;
  ControlFlowGraph &cfg;
};

//===----------------------------------------------------------------------===//
// Copy Chain Rewriter - 重写 copy 链
//===----------------------------------------------------------------------===//

/// CopyChainRewriter - 重写 copy 链用于 CBUB 优化
///
/// 原始代码问题：
/// 1. rewriteCopyChainForCbub: 硬编码的 reshape/trans 顺序
/// 2. 手动计算新形状
/// 3. 类型更新分散在多处
///
/// 重构后改进：
/// 1. 统一的形状计算
/// 2. 可配置的转换链
/// 3. 自动类型更新
class CopyChainRewriter {
public:
  explicit CopyChainRewriter(OpBuilder &builder, AffinityDAG::Graph &dagGraph)
      : builder(builder), dagGraph(dagGraph) {}

  /// 重写所有符合条件的 copy 操作
  void rewriteAllCopyChains(triton::FuncOp funcOp);

  /// 重写单个 copy 操作
  bool rewriteCopyChain(hivm::CopyOp copyOp);

private:
  /// 计算 ND 到 NZ 转换的形状
  std::optional<SmallVector<int64_t, 4>> computeNDToNZShape(
      RankedTensorType tensorType);

  /// 计算中间 reshape 形状
  std::optional<SmallVector<int64_t, 4>> computeIntermediateShape(
      RankedTensorType tensorType);

  /// 创建 reshape + trans 链
  Value createReshapeTransChain(Value input,
                                const SmallVector<int64_t, 4> &finalShape,
                                Location loc);

  OpBuilder &builder;
  AffinityDAG::Graph &dagGraph;
};

//===----------------------------------------------------------------------===//
// High-Level Orchestrator - 主驱动
//===----------------------------------------------------------------------===//

/// DAGSyncOptimizationDriver - DAGSync 优化主驱动
///
/// 整合上述所有组件，提供清晰的优化流程
class DAGSyncOptimizationDriver {
public:
  DAGSyncOptimizationDriver(DataFlowGraph &dfg, ControlFlowGraph &cfg,
                            AffinityDAG::Graph &dagGraph,
                            OpBuilder &builder, ModuleOp module)
      : analyzer(dfg, cfg, dagGraph),
        dataMovementBuilder(builder, module, dagGraph),
        syncInserter(builder, dfg, cfg),
        crossBlockHandler(cfg, syncInserter),
        dotLegalizer(builder, cfg),
        copyRewriter(builder, dagGraph),
        dfg(dfg), cfg(cfg), dagGraph(dagGraph), builder(builder) {}

  /// 执行完整的 sync 优化
  void run(triton::FuncOp funcOp);

private:
  /// 步骤1: 合法化 dot 操作
  void legalizeDots(triton::FuncOp funcOp);

  /// 步骤2: 分析同步需求
  SmallVector<CrossCoreDependency> analyzeSyncNeeds(triton::FuncOp funcOp);

  /// 步骤3: 插入数据搬运
  void insertDataMovements(ArrayRef<CrossCoreDependency> deps);

  /// 步骤4: 插入同步操作
  void insertSyncOperations(ArrayRef<CrossCoreDependency> deps);

  /// 步骤5: 重写 copy 链
  void rewriteCopyChains(triton::FuncOp funcOp);

  SyncAnalyzer analyzer;
  DataMovementBuilder dataMovementBuilder;
  SyncInserter syncInserter;
  CrossBlockSyncHandler crossBlockHandler;
  DotLegalizer dotLegalizer;
  CopyChainRewriter copyRewriter;
  DataFlowGraph &dfg;
  ControlFlowGraph &cfg;
  AffinityDAG::Graph &dagGraph;
  OpBuilder &builder;
  int nextFlag = 1;
  DenseSet<std::pair<Instruction *, Instruction *>> processedPairs;
};

#endif // TRITON_AFFINITY_OPT_DAG_SYNC_REFACTORED_H
