/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_TO_CFG_CONTROL_FLOW_GRAPH_BUILDER_H
#define TRITON_TO_CFG_CONTROL_FLOW_GRAPH_BUILDER_H

#include "TritonToCFG/ControlFlowGraph.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <stack>

namespace mlir {
namespace triton {

// 构建控制流图的 Pass
class BuildCFGPass
    : public PassWrapper<BuildCFGPass, OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BuildCFGPass)

  BuildCFGPass() = default;

  StringRef getArgument() const override { return "build-cfg"; }
  StringRef getDescription() const override {
    return "Build Control Flow Graph from TTIR";
  }

  void runOnOperation() override;

  // Pass 选项
  std::string outputDir = ".";

  // 获取依赖的方言
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::TritonDialect, scf::SCFDialect, cf::ControlFlowDialect>();
  }

protected:
  // 构建单个函数的 CFG
  std::unique_ptr<cfg::ControlFlowGraph> buildForFunction(triton::FuncOp func);
};

// 独立的 CFG 构建器类（用于非 Pass 场景）
class ControlFlowGraphBuilder {
public:
  // 为函数构建 CFG
  std::unique_ptr<cfg::ControlFlowGraph> build(triton::FuncOp func);

  // 为模块构建所有函数的 CFG
  std::vector<std::unique_ptr<cfg::ControlFlowGraph>> buildForModule(ModuleOp module);

  // 处理一个 region，返回该 region 的入口块和出口块
  struct RegionBlocks {
    cfg::BasicBlock *entryBlock;
    cfg::BasicBlock *exitBlock;
  };

  RegionBlocks buildForRegion(Region &region, cfg::ControlFlowGraph &cfg,
                               cfg::BasicBlock *entryBlock,
                               cfg::BasicBlock *parentStructure = nullptr);

  // 处理 block 中的操作，返回最后处理的基本块
  cfg::BasicBlock *processBlock(Block &block, cfg::ControlFlowGraph &cfg,
                                 cfg::BasicBlock *currentBB,
                                 cfg::BasicBlock *parentStructure = nullptr);

  // 处理 scf.if 操作，返回 if 后面的基本块
  cfg::BasicBlock *handleIfOp(scf::IfOp ifOp, cfg::ControlFlowGraph &cfg,
                               cfg::BasicBlock *currentBB,
                               cfg::BasicBlock *parentStructure = nullptr);

  // 处理 scf.for 操作，返回 for 后面的基本块
  cfg::BasicBlock *handleForOp(scf::ForOp forOp, cfg::ControlFlowGraph &cfg,
                                cfg::BasicBlock *currentBB,
                                cfg::BasicBlock *parentStructure = nullptr);

  // 处理 scf.while 操作，返回 while 后面的基本块
  cfg::BasicBlock *handleWhileOp(scf::WhileOp whileOp, cfg::ControlFlowGraph &cfg,
                                  cfg::BasicBlock *currentBB,
                                  cfg::BasicBlock *parentStructure = nullptr);

  // 创建一个新的指令并添加到 basic block
  cfg::Instruction *createInstruction(Operation *op, cfg::BasicBlock *parentBlock, cfg::ControlFlowGraph &cfg);

  // 获取下一个指令 ID
  size_t getNextInstructionId() { return nextInstructionId++; }

private:
  size_t nextInstructionId = 0;      // 下一个指令 ID
};

// 创建 Pass 的工厂函数
std::unique_ptr<OperationPass<mlir::ModuleOp>> createBuildCFGPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_TO_CFG_CONTROL_FLOW_GRAPH_BUILDER_H
