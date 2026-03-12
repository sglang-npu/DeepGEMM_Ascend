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

  // 获取依赖的方言
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::TritonDialect, scf::SCFDialect, cf::ControlFlowDialect>();
  }

protected:
  // 构建单个函数的 CFG
  std::unique_ptr<cfg::ControlFlowGraph> buildForFunction(triton::FuncOp func);

  // 遍历区域构建 CFG
  void buildForRegion(Region &region, cfg::ControlFlowGraph &cfg,
                      size_t entryId);

  // 处理终止符操作
  void handleTerminator(Operation *terminator, cfg::ControlFlowGraph &cfg,
                        size_t blockId);

  // 处理 scf.if 操作
  void handleIfOp(Operation *op, cfg::ControlFlowGraph &cfg, size_t blockId);

  // 处理 scf.for 操作
  void handleForOp(Operation *op, cfg::ControlFlowGraph &cfg, size_t blockId);

  // 处理 scf.while 操作
  void handleWhileOp(Operation *op, cfg::ControlFlowGraph &cfg, size_t blockId);
};

// 创建 Pass 的工厂函数
std::unique_ptr<OperationPass<mlir::ModuleOp>> createBuildCFGPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_TO_CFG_CONTROL_FLOW_GRAPH_BUILDER_H
