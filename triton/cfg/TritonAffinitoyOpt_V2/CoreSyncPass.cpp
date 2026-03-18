/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * 基于TritonToGraph控制流数据流分析框架的Core同步Pass
 * 合并了DAGSync和DAGScope的核心功能
 */

#include "TritonAffinityOpt/Passes.h"
#include "TritonToGraph/ControlFlowGraphBuilder.h"
#include "TritonToGraph/DataflowGraph.h"
#include "CoreSyncAnalyzer.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "core-sync-pass"

using namespace mlir;
using namespace triton;
using namespace affinity;

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_CORESYNC
#include "ascend/include/TritonAffinityOpt/Passes.h.inc"

} // namespace triton
} // namespace mlir

namespace {

struct CoreSyncPass : public mlir::triton::impl::CoreSyncBase<CoreSyncPass> {
  void runOnOperation() override;

private:
  // 处理单个函数
  void processFunction(triton::FuncOp func, ModuleOp module);

  // 初始化CFG和DataFlow分析
  std::unique_ptr<cfg::ControlFlowGraph> buildCFG(triton::FuncOp func);

  // 使用原始DAG数据初始化valueTypes
  void initializeCoreTypes(triton::FuncOp func,
                           llvm::DenseMap<Value, CoreType>& types);
};

} // namespace

void CoreSyncPass::runOnOperation() {
  auto module = getOperation();

  LLVM_DEBUG(llvm::dbgs() << "=== CoreSyncPass Starting ===\n");
  LLVM_DEBUG(llvm::dbgs() << module << "\n\n");

  for (auto funcOp : llvm::make_early_inc_range(module.getOps<triton::FuncOp>())) {
    if (funcOp.getBody().empty())
      continue;

    processFunction(funcOp, module);
  }

  LLVM_DEBUG(llvm::dbgs() << module << "\n\n");
  LLVM_DEBUG(llvm::dbgs() << "=== CoreSyncPass Complete ===\n");
}

void CoreSyncPass::processFunction(triton::FuncOp func, ModuleOp module) {
  LLVM_DEBUG(llvm::dbgs() << "Processing function: " << func.getName() << "\n");

  // 1. 使用原始DAG框架分析核心类型（保持兼容性）
  llvm::DenseMap<Value, CoreType> coreTypes;
  initializeCoreTypes(func, coreTypes);

  // 2. 构建CFG（使用TritonToGraph框架）
  auto cfg = buildCFG(func);
  if (!cfg) {
    llvm::errs() << "Failed to build CFG for " << func.getName() << "\n";
    return;
  }

  // 3. 构建DataFlow图（包含Memory SSA分析）
  cfg::DataFlowGraph dataFlowGraph(*cfg);
  dataFlowGraph.build();

  // 4. 分析同步点
  CoreSyncAnalyzer analyzer(*cfg, dataFlowGraph.getDataFlowInfo(), &coreTypes);
  analyzer.analyze();

  // 5. 构建AIV/AIC Scope
  ScopeBuilder scopeBuilder(*cfg, analyzer, &coreTypes);
  auto [aivScope, aicScope] = scopeBuilder.buildScopes(func);

  if (!aivScope || !aicScope) {
    LLVM_DEBUG(llvm::dbgs() << "No scopes created for " << func.getName() << "\n");
    return;
  }

  // 6. 插入同步操作
  SyncInserter inserter(analyzer, &coreTypes);
  inserter.insertSyncOps(aivScope, aicScope);

  // 7. 处理buffer wait同步增强
  inserter.addSyncForBufferWait(func);

  LLVM_DEBUG(llvm::dbgs() << "Function " << func.getName()
                          << " processed, sync points: "
                          << analyzer.getCurrentFlag() - 1 << "\n");
}

std::unique_ptr<cfg::ControlFlowGraph> CoreSyncPass::buildCFG(triton::FuncOp func) {
  cfg::ControlFlowGraphBuilder builder;
  return builder.build(func);
}

void CoreSyncPass::initializeCoreTypes(triton::FuncOp func,
                                       llvm::DenseMap<Value, CoreType>& types) {
  // 使用原始AffinityDAG框架分析
  auto [shared_graph, _] = AffinityDAG::Graph::fromMultiBlockFunc(func);
  if (!shared_graph)
    return;

  shared_graph->markCore();

  // 复制分析结果
  for (const auto& pair : *shared_graph->valueTypes) {
    types[pair.first] = pair.second;
  }

  // 注册图以便后续使用
  AffinityDAG::GraphManager::getInstance().registerGraph(func.getName(), shared_graph);
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createCoreSyncPass() {
  return std::make_unique<CoreSyncPass>();
}
