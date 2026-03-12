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

#include "TritonToCFG/ControlFlowGraphBuilder.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "build-cfg"

using namespace mlir;
using namespace mlir::triton;

//===----------------------------------------------------------------------===//
// BuildCFGPass Implementation
//===----------------------------------------------------------------------===//

void BuildCFGPass::runOnOperation() {
  auto module = getOperation();
  llvm::errs() << "Building CFG for module\n";

  // 遍历模块中的所有 Triton 函数 (tt.func)
  for (triton::FuncOp func : module.getOps<triton::FuncOp>()) {
    llvm::errs() << "Processing function: " << func.getName() << "\n";

    auto cfg = buildForFunction(func);
    if (!cfg) {
      func.emitError() << "Failed to build CFG for function";
      signalPassFailure();
      return;
    }

    // 打印 CFG 到标准输出
    cfg->print(llvm::outs());
  }
}

std::unique_ptr<cfg::ControlFlowGraph>
BuildCFGPass::buildForFunction(triton::FuncOp func) {
  auto cfg = std::make_unique<cfg::ControlFlowGraph>(func);

  if (func.getBody().empty()) {
    return cfg;
  }

  // 创建入口块
  size_t entryId = cfg->addBasicBlock(nullptr, cfg::BlockType::ENTRY);

  // 构建函数体的 CFG
  buildForRegion(func.getBody(), *cfg, entryId);

  return cfg;
}

void BuildCFGPass::buildForRegion(Region &region, cfg::ControlFlowGraph &cfg,
                                  size_t entryId) {
  if (!region.hasOneBlock()) {
    LLVM_DEBUG(llvm::dbgs() << "Region has multiple blocks\n");
  }

  // 遍历所有块
  for (Block &block : region) {
    // 检查是否已经处理过这个块
    if (cfg.findBasicBlock(&block)) {
      continue;
    }

    // 添加基本块
    size_t blockId = cfg.addBasicBlock(&block, cfg::BlockType::NORMAL);

    // 如果是第一个块，连接到入口
    if (&block == &region.front()) {
      cfg.addEdge(entryId, blockId);
    }

    // 处理块中的所有操作
    for (Operation &op : block) {
      // 检查是否是控制流操作
      if (op.getDialect()) {
        StringRef dialectName = op.getDialect()->getNamespace();
        if (dialectName == "scf") {
          // 处理 scf dialect 的操作
          if (isa<scf::IfOp>(op)) {
            handleIfOp(&op, cfg, blockId);
          } else if (isa<scf::ForOp>(op)) {
            handleForOp(&op, cfg, blockId);
          } else if (isa<scf::WhileOp>(op)) {
            handleWhileOp(&op, cfg, blockId);
          }
        }
      }
    }

    // 处理终止符
    if (auto terminator = block.getTerminator()) {
      handleTerminator(terminator, cfg, blockId);
    }
  }
}

void BuildCFGPass::handleTerminator(Operation *terminator,
                                    cfg::ControlFlowGraph &cfg,
                                    size_t blockId) {
  LLVM_DEBUG(llvm::dbgs() << "Handling terminator: " << *terminator << "\n");

  // 处理返回操作
  if (isa<triton::ReturnOp>(terminator)) {
    // 创建出口块（如果还没有）
    bool hasExit = false;
    size_t exitId = 0;

    for (size_t i = 0; i < cfg.getNumBlocks(); ++i) {
      if (cfg.getBasicBlock(i).getType() == cfg::BlockType::EXIT) {
        hasExit = true;
        exitId = i;
        break;
      }
    }

    if (!hasExit) {
      exitId = cfg.addBasicBlock(nullptr, cfg::BlockType::EXIT);
    }

    cfg.addEdge(blockId, exitId);
  }
  // 处理分支操作
  else if (auto brOp = dyn_cast<cf::BranchOp>(terminator)) {
    Block *destBlock = brOp.getDest();
    if (auto *destNode = cfg.findBasicBlock(destBlock)) {
      cfg.addEdge(blockId, destNode->getId());
    }
  }
  // 处理条件分支
  else if (auto condBrOp = dyn_cast<cf::CondBranchOp>(terminator)) {
    Block *trueDest = condBrOp.getTrueDest();
    Block *falseDest = condBrOp.getFalseDest();

    if (auto *trueNode = cfg.findBasicBlock(trueDest)) {
      cfg.addEdge(blockId, trueNode->getId());
    }

    if (auto *falseNode = cfg.findBasicBlock(falseDest)) {
      cfg.addEdge(blockId, falseNode->getId());
    }
  }
}

void BuildCFGPass::handleIfOp(Operation *op, cfg::ControlFlowGraph &cfg,
                              size_t blockId) {
  auto ifOp = cast<scf::IfOp>(op);
  LLVM_DEBUG(llvm::dbgs() << "Handling scf.if: " << ifOp << "\n");

  // 处理 then 分支
  if (ifOp.getThenRegion().empty()) {
    // 空的 then 分支，直接连接到 if 操作后的块
  } else {
    // 为 then 分支的第一个块创建 CFG 节点
    buildForRegion(ifOp.getThenRegion(), cfg, blockId);
  }

  // 处理 else 分支
  if (ifOp.getElseRegion().empty()) {
    // 空的 else 分支
  } else {
    // 为 else 分支的第一个块创建 CFG 节点
    buildForRegion(ifOp.getElseRegion(), cfg, blockId);
  }
}

void BuildCFGPass::handleForOp(Operation *op, cfg::ControlFlowGraph &cfg,
                               size_t blockId) {
  auto forOp = cast<scf::ForOp>(op);
  LLVM_DEBUG(llvm::dbgs() << "Handling scf.for: " << forOp << "\n");

  // 创建循环头节点
  if (!forOp.getRegion().empty()) {
    buildForRegion(forOp.getRegion(), cfg, blockId);
  }
}

void BuildCFGPass::handleWhileOp(Operation *op, cfg::ControlFlowGraph &cfg,
                                 size_t blockId) {
  auto whileOp = cast<scf::WhileOp>(op);
  LLVM_DEBUG(llvm::dbgs() << "Handling scf.while: " << whileOp << "\n");

  // 处理 while 的 before 和 after 区域
  if (!whileOp.getBefore().empty()) {
    buildForRegion(whileOp.getBefore(), cfg, blockId);
  }

  if (!whileOp.getAfter().empty()) {
    // 处理 after 区域，需要连接到 before 的出口
    // 这里简化处理，实际需要更复杂的逻辑
    buildForRegion(whileOp.getAfter(), cfg, blockId);
  }
}

std::unique_ptr<OperationPass<ModuleOp>> triton::createBuildCFGPass() {
  return std::unique_ptr<OperationPass<ModuleOp>>(new BuildCFGPass());
}
