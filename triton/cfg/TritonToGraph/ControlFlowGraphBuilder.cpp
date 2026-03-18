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

#include "ControlFlowGraphBuilder.h"
#include "DataflowGraph.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "build-cfg"

using namespace mlir;
using namespace mlir::triton;
using namespace cfg;

//===----------------------------------------------------------------------===//
// BuildCFGPass Implementation
//===----------------------------------------------------------------------===//

void BuildCFGPass::runOnOperation() {
  auto module = getOperation();
  llvm::errs() << "Building CFG for module\n";

  // 获取输出目录（从命令行选项，默认为当前目录）
  // TableGen 生成的选项成员变量名为 outputDir
  std::string outputDir = this->outputDir;

  // 创建输出目录
  llvm::SmallString<128> outputPath(outputDir);
  llvm::sys::fs::create_directories(outputPath);

  llvm::errs() << "CFG output directory: " << outputPath << "\n";

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

    // 导出到文件
    std::string baseName = func.getName().str();

    // 导出文本格式
    llvm::SmallString<128> textPath(outputPath);
    llvm::sys::path::append(textPath, baseName + "_cfg.txt");
    if (auto err = cfg->exportToFile(textPath)) {
      llvm::errs() << "Failed to export CFG to " << textPath << "\n";
    } else {
      llvm::errs() << "Exported CFG to " << textPath << "\n";
    }

    // 导出 DOT 格式
    llvm::SmallString<128> dotPath(outputPath);
    llvm::sys::path::append(dotPath, baseName + "_cfg.dot");
    if (auto err = cfg->exportToFile(dotPath)) {
      llvm::errs() << "Failed to export CFG to " << dotPath << "\n";
    } else {
      llvm::errs() << "Exported CFG to " << dotPath << "\n";
    }

    // 导出 JSON 格式
    llvm::SmallString<128> jsonPath(outputPath);
    llvm::sys::path::append(jsonPath, baseName + "_cfg.json");
    if (auto err = cfg->exportToFile(jsonPath)) {
      llvm::errs() << "Failed to export CFG to " << jsonPath << "\n";
    } else {
      llvm::errs() << "Exported CFG to " << jsonPath << "\n";
    }

    // 导出 HTML 格式（网页可视化）
    llvm::SmallString<128> htmlPath(outputPath);
    llvm::sys::path::append(htmlPath, baseName + "_cfg.html");
    if (auto err = cfg->exportToHTML(htmlPath)) {
      llvm::errs() << "Failed to export CFG to " << htmlPath << "\n";
    } else {
      llvm::errs() << "Exported CFG to " << htmlPath << "\n";
    }

    // 构建 DataFlowGraph（包含 Memory SSA 分析）
    llvm::errs() << "  Building DataFlowGraph with Memory SSA...\n";
    DataFlowGraph dataFlowGraph(*cfg);
    dataFlowGraph.build();

    // 导出 DataFlowGraph
    std::error_code ec;
    llvm::SmallString<128> dataflowPath(outputPath);
    llvm::sys::path::append(dataflowPath, baseName + "_dataflow.json");
    llvm::raw_fd_ostream dfOs(dataflowPath, ec);
    if (!ec) {
      dataFlowGraph.exportToJSON(dfOs);
      llvm::errs() << "  Exported DataFlowGraph to " << dataflowPath << "\n";
    }
  }
}

std::unique_ptr<cfg::ControlFlowGraph>
BuildCFGPass::buildForFunction(triton::FuncOp func) {
  ControlFlowGraphBuilder cfgBuilder;
  return cfgBuilder.build(func);
}

ControlFlowGraphBuilder::RegionBlocks
ControlFlowGraphBuilder::buildForRegion(Region &region, cfg::ControlFlowGraph &cfg,
                              cfg::BasicBlock *entryBlock,
                              cfg::BasicBlock *parentStructure) {
  cfg::BasicBlock *currentBB = entryBlock;
  cfg::BasicBlock *lastBlock = entryBlock;

  // 遍历 region 中的所有 block
  for (Block &block : region) {
    // 处理当前 block 的操作
    currentBB = processBlock(block, cfg, currentBB, parentStructure);
    if (currentBB) {
      lastBlock = currentBB;
    }
  }

  return {entryBlock, lastBlock};
}

cfg::BasicBlock *ControlFlowGraphBuilder::processBlock(Block &block, cfg::ControlFlowGraph &cfg,
                                             cfg::BasicBlock *currentBB,
                                             cfg::BasicBlock *parentStructure) {
  if (!currentBB) return nullptr;

  // 遍历 block 中的所有操作
  for (Operation &op : block) {
    // 检查是否是控制流操作
    if (isa<scf::IfOp>(op)) {
      // 为 if 条件创建单独的 basic block
      auto *ifCondBB = cfg.createBasicBlock(BlockType::IF_COND, parentStructure);

      // 将当前 if 指令添加到 ifCondBB
      createInstruction(&op, ifCondBB, cfg);

      // 连接当前块到 if 条件块
      cfg.addEdge(currentBB, ifCondBB);

      // 处理 if 操作，返回 if 后面的块
      currentBB = handleIfOp(cast<scf::IfOp>(op), cfg, ifCondBB, parentStructure);
    }
    else if (isa<scf::ForOp>(op)) {
      // 为 for 条件创建单独的 basic block
      auto *forCondBB = cfg.createBasicBlock(BlockType::FOR_COND, parentStructure);

      // 将当前 for 指令添加到 forCondBB
      createInstruction(&op, forCondBB, cfg);

      // 连接当前块到 for 条件块
      cfg.addEdge(currentBB, forCondBB);

      // 处理 for 操作，返回 for 后面的块
      currentBB = handleForOp(cast<scf::ForOp>(op), cfg, forCondBB, parentStructure);
    }
    else if (isa<scf::WhileOp>(op)) {
      // 为 while 条件创建单独的 basic block
      auto *whileCondBB = cfg.createBasicBlock(BlockType::WHILE_COND, parentStructure);

      // 将当前 while 指令添加到 whileCondBB
      createInstruction(&op, whileCondBB, cfg);

      // 连接当前块到 while 条件块
      cfg.addEdge(currentBB, whileCondBB);

      // 处理 while 操作，返回 while 后面的块
      currentBB = handleWhileOp(cast<scf::WhileOp>(op), cfg, whileCondBB, parentStructure);
    }
    else if (isa<scf::YieldOp>(op)) {
      // yield 操作：创建指令并继续（后续由循环处理逻辑连接）
      createInstruction(&op, currentBB, cfg);
    }
    else if (isa<scf::ConditionOp>(op)) {
      // condition 操作（while 循环条件）：创建指令并继续
      createInstruction(&op, currentBB, cfg);
    }
    else if (isa<cf::CondBranchOp>(op)) {
      // cf.cond_br 条件分支
      createInstruction(&op, currentBB, cfg);

      // 注意：cf.cond_br 的目标块会在后续处理中连接
      // 这里我们依赖于 MLIR 的 block 结构已经建立
    }
    else if (isa<cf::BranchOp>(op)) {
      // cf.br 无条件跳转
      createInstruction(&op, currentBB, cfg);
    }
    else if (isa<triton::ReturnOp>(op)) {
      // return 操作
      createInstruction(&op, currentBB, cfg);
    }
    else if (op.getNumRegions() > 0) {
      // 有内部区域的 Triton 操作 (如 tt.reduce, tt.scan 等)
      // 先创建指令
      auto *inst = createInstruction(&op, currentBB, cfg);

      // 为该操作创建子图
      auto subGraph = std::make_unique<cfg::ControlFlowGraph>(cfg.getFunction());
      auto *subEntry = subGraph->createBasicBlock(cfg::BlockType::ENTRY);
      subGraph->setEntryBlock(subEntry);
      auto *subExit = subGraph->createBasicBlock(cfg::BlockType::EXIT);
      subGraph->setExitBlock(subExit);

      // 用于子图中生成唯一指令 ID
      size_t subInstId = 0;

      // 遍历所有区域
      for (size_t regionIdx = 0; regionIdx < op.getNumRegions(); ++regionIdx) {
        Region &region = op.getRegion(regionIdx);
        if (!region.empty()) {
          // 为每个区域构建 CFG
          cfg::BasicBlock *regionEntryBB = nullptr;
          cfg::BasicBlock *regionLastBB = nullptr;

          for (Block &regionBlock : region) {
            auto *bb = subGraph->createBasicBlock(cfg::BlockType::NORMAL);
            if (!regionEntryBB) regionEntryBB = bb;

            // 将区域中的操作添加到子图的基本块
            for (Operation &regionOp : regionBlock) {
              auto regionInst = std::make_unique<cfg::Instruction>(
                  subInstId++, &regionOp, bb);
              cfg::Instruction *instPtr = regionInst.get();
              bb->addInstruction(std::move(regionInst));

              // 添加到op到instruction的映射
              subGraph->addOpToInstruction(&regionOp, instPtr);
            }

            // 连接基本块
            if (regionLastBB) {
              subGraph->addEdge(regionLastBB, bb);
            }
            regionLastBB = bb;
          }

          // 连接区域入口到子图入口
          if (regionEntryBB) {
            subGraph->addEdge(subEntry, regionEntryBB);
          }
          // 连接区域出口到子图出口
          if (regionLastBB) {
            subGraph->addEdge(regionLastBB, subExit);
          }
        }
      }

      // 设置子图
      inst->setSubGraph(std::move(subGraph));
    }
    else {
      // 普通操作，直接添加到当前 basic block
      createInstruction(&op, currentBB, cfg);
    }
  }

  return currentBB;
}

cfg::BasicBlock *ControlFlowGraphBuilder::handleIfOp(scf::IfOp ifOp, cfg::ControlFlowGraph &cfg,
                                           cfg::BasicBlock *ifCondBB,
                                           cfg::BasicBlock *parentStructure) {
  // 创建 if 后面的汇合块
  auto *mergeBB = cfg.createBasicBlock(BlockType::NORMAL, parentStructure);

  // 处理 then 分支
  cfg::BasicBlock *thenExitBB = nullptr;
  if (!ifOp.getThenRegion().empty()) {
    // 创建 then 区域的入口块
    auto *thenEntryBB = cfg.createBasicBlock(BlockType::NORMAL, ifCondBB);
    cfg.addEdge(ifCondBB, thenEntryBB);

    // 构建 then 区域的 CFG
    auto result = buildForRegion(ifOp.getThenRegion(), cfg, thenEntryBB, ifCondBB);
    thenExitBB = result.exitBlock;
  }

  // 处理 else 分支
  cfg::BasicBlock *elseExitBB = nullptr;
  bool hasElse = !ifOp.getElseRegion().empty();

  if (hasElse) {
    // 创建 else 区域的入口块
    auto *elseEntryBB = cfg.createBasicBlock(BlockType::NORMAL, ifCondBB);
    cfg.addEdge(ifCondBB, elseEntryBB);

    // 构建 else 区域的 CFG
    auto result = buildForRegion(ifOp.getElseRegion(), cfg, elseEntryBB, ifCondBB);
    elseExitBB = result.exitBlock;
  }

  // 连接 then 分支到汇合块
  if (thenExitBB) {
    cfg.addEdge(thenExitBB, mergeBB);
  } else {
    // 空的 then 分支，直接从 ifCondBB 连接到 mergeBB
    cfg.addEdge(ifCondBB, mergeBB);
  }

  // 连接 else 分支到汇合块
  if (hasElse) {
    if (elseExitBB) {
      cfg.addEdge(elseExitBB, mergeBB);
    } else {
      // 空的 else 分支，直接从 ifCondBB 连接到 mergeBB
      cfg.addEdge(ifCondBB, mergeBB);
    }
  } else {
    // 没有 else 分支，ifCondBB 直接连接到 mergeBB（else 路径）
    cfg.addEdge(ifCondBB, mergeBB);
  }

  return mergeBB;
}

cfg::BasicBlock *ControlFlowGraphBuilder::handleForOp(scf::ForOp forOp, cfg::ControlFlowGraph &cfg,
                                            cfg::BasicBlock *forCondBB,
                                            cfg::BasicBlock *parentStructure) {
  // 创建循环体入口块
  auto *loopBodyEntryBB = cfg.createBasicBlock(BlockType::LOOP_BODY, forCondBB);
  cfg.addEdge(forCondBB, loopBodyEntryBB);

  // 创建循环出口块
  auto *loopExitBB = cfg.createBasicBlock(BlockType::LOOP_EXIT, parentStructure);

  // 构建循环体的 CFG
  auto result = buildForRegion(forOp.getRegion(), cfg, loopBodyEntryBB, forCondBB);

  // 循环体结束需要回到循环头（通过 yield 操作）
  if (result.exitBlock) {
    cfg.addEdge(result.exitBlock, forCondBB);
  }

  // 循环出口
  cfg.addEdge(forCondBB, loopExitBB);

  return loopExitBB;
}

cfg::BasicBlock *ControlFlowGraphBuilder::handleWhileOp(scf::WhileOp whileOp, cfg::ControlFlowGraph &cfg,
                                              cfg::BasicBlock *whileCondBB,
                                              cfg::BasicBlock *parentStructure) {
  // while 操作有两个区域：before（条件）和 after（循环体）
  // 控制流：
  //   whileCondBB (包含 scf.while 指令)
  //       ↓
  //   beforeEntryBB (条件计算区域)
  //       ↓
  //   scf.condition 分支：真 → afterEntryBB, 假 → loopExitBB
  //       ↓
  //   afterEntryBB (循环体)
  //       ↓
  //   scf.yield
  //       ↓
  //   whileCondBB (回到 while 头，重新进入 before)

  // 创建 before 区域的入口块（条件计算）
  auto *beforeEntryBB = cfg.createBasicBlock(BlockType::LOOP_BODY, whileCondBB);
  cfg.addEdge(whileCondBB, beforeEntryBB);

  // 创建 after 区域的入口块（循环体）
  auto *afterEntryBB = cfg.createBasicBlock(BlockType::LOOP_BODY, whileCondBB);

  // 创建循环出口块
  auto *loopExitBB = cfg.createBasicBlock(BlockType::LOOP_EXIT, parentStructure);

  // 构建 before 区域的 CFG（条件计算区域）
  // before 区域以一个 scf.condition 操作结束
  auto beforeResult = buildForRegion(whileOp.getBefore(), cfg, beforeEntryBB, whileCondBB);

  // 构建 after 区域的 CFG（循环体区域）
  // after 区域以 scf.yield 结束，yield 的参数会传递给 before 区域的参数
  auto afterResult = buildForRegion(whileOp.getAfter(), cfg, afterEntryBB, whileCondBB);

  // 处理 before 区域结束后的分支
  // before 区域应该以一个 scf.condition 操作结束
  // 该操作决定是进入 after 区域还是退出循环
  if (beforeResult.exitBlock) {
    // 从 before 出口连接到 after 入口（条件为真时）
    cfg.addEdge(beforeResult.exitBlock, afterEntryBB);

    // 从 before 出口连接到循环出口（条件为假时）
    // 注意：在实际的 scf.condition 中，条件为假会直接退出循环
    cfg.addEdge(beforeResult.exitBlock, loopExitBB);
  }

  // after 区域结束后回到 whileCondBB（重新进入 before 区域进行条件检查）
  if (afterResult.exitBlock) {
    cfg.addEdge(afterResult.exitBlock, whileCondBB);
  }

  return loopExitBB;
}

cfg::Instruction *ControlFlowGraphBuilder::createInstruction(Operation *op, cfg::BasicBlock *parentBlock, cfg::ControlFlowGraph &cfg) {
  if (!op || !parentBlock) return nullptr;

  auto inst = std::make_unique<cfg::Instruction>(getNextInstructionId(), op, parentBlock);
  cfg::Instruction *instPtr = inst.get();
  parentBlock->addInstruction(std::move(inst));

  // 将 Operation 到 Instruction 的映射添加到 CFG
  cfg.addOpToInstruction(op, instPtr);

  return instPtr;
}

//===----------------------------------------------------------------------===//
// ControlFlowGraphBuilder Implementation
//===----------------------------------------------------------------------===//

std::unique_ptr<cfg::ControlFlowGraph>
ControlFlowGraphBuilder::build(triton::FuncOp func) {
  auto cfg = std::make_unique<cfg::ControlFlowGraph>(func);

  if (func.getBody().empty()) {
    // 空函数，只创建入口和出口
    auto *entry = cfg->createBasicBlock(BlockType::ENTRY);
    auto *exit = cfg->createBasicBlock(BlockType::EXIT);
    cfg->addEdge(entry, exit);
    return cfg;
  }

  // 创建入口块
  auto *entryBlock = cfg->createBasicBlock(BlockType::ENTRY);
  cfg->setEntryBlock(entryBlock);

  // 创建出口块
  auto *exitBlock = cfg->createBasicBlock(BlockType::EXIT);
  cfg->setExitBlock(exitBlock);

  // 为函数体构建 CFG
  auto result = buildForRegion(func.getBody(), *cfg, entryBlock, nullptr);

  // 连接函数体出口到函数出口
  if (result.exitBlock) {
    cfg->addEdge(result.exitBlock, exitBlock);
  }

  return cfg;
}

std::vector<std::unique_ptr<cfg::ControlFlowGraph>>
ControlFlowGraphBuilder::buildForModule(ModuleOp module) {
  std::vector<std::unique_ptr<cfg::ControlFlowGraph>> cfgs;

  for (triton::FuncOp func : module.getOps<triton::FuncOp>()) {
    auto cfg = build(func);
    if (cfg) {
      cfgs.push_back(std::move(cfg));
    }
  }

  return cfgs;
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::cfg::createBuildCFGPass() {
  return std::unique_ptr<OperationPass<ModuleOp>>(new BuildCFGPass());
}
