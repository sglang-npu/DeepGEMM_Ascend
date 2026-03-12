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

#include "TritonToCFG/ControlFlowGraph.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>

using namespace mlir;
using namespace triton;
using namespace cfg;

//===----------------------------------------------------------------------===//
// BasicBlockNode
//===----------------------------------------------------------------------===//

StringRef BasicBlockNode::getName() const {
  if (!block) {
    return "<null>";
  }
  // 尝试获取块名称
  if (block->getParent()) {
    Operation *parentOp = block->getParent()->getParentOp();
    if (parentOp) {
      if (auto funcOp = dyn_cast<triton::FuncOp>(parentOp)) {
        return funcOp.getName();
      }
    }
  }
  return "bb";
}

void BasicBlockNode::print(raw_ostream &os) const {
  os << "Block " << id << " [";
  switch (type) {
  case BlockType::ENTRY:
    os << "ENTRY";
    break;
  case BlockType::EXIT:
    os << "EXIT";
    break;
  case BlockType::NORMAL:
    os << "NORMAL";
    break;
  case BlockType::LOOP_HEADER:
    os << "LOOP_HEADER";
    break;
  case BlockType::LOOP_BODY:
    os << "LOOP_BODY";
    break;
  case BlockType::LOOP_EXIT:
    os << "LOOP_EXIT";
    break;
  case BlockType::IF_THEN:
    os << "IF_THEN";
    break;
  case BlockType::IF_ELSE:
    os << "IF_ELSE";
    break;
  }
  os << "]";

  // 打印前驱和后继
  if (!predecessors.empty()) {
    os << " Preds: [";
    for (size_t i = 0; i < predecessors.size(); ++i) {
      if (i > 0)
        os << ", ";
      os << predecessors[i];
    }
    os << "]";
  }

  if (!successors.empty()) {
    os << " Succs: [";
    for (size_t i = 0; i < successors.size(); ++i) {
      if (i > 0)
        os << ", ";
      os << successors[i];
    }
    os << "]";
  }
}

void BasicBlockNode::dump() const { print(llvm::errs()); }

//===----------------------------------------------------------------------===//
// ControlFlowGraph
//===----------------------------------------------------------------------===//

ControlFlowGraph::ControlFlowGraph(triton::FuncOp func) : function(func) {}

ControlFlowGraph::~ControlFlowGraph() = default;

size_t ControlFlowGraph::addBasicBlock(Block *block, BlockType type) {
  size_t id = basicBlocks.size();
  auto bbNode = std::make_unique<BasicBlockNode>(id, block, type);
  basicBlocks.push_back(std::move(bbNode));

  if (block) {
    blockToId[block] = id;
  }

  return id;
}

BasicBlockNode *ControlFlowGraph::findBasicBlock(Block *block) {
  auto it = blockToId.find(block);
  if (it != blockToId.end()) {
    return basicBlocks[it->second].get();
  }
  return nullptr;
}

const BasicBlockNode *ControlFlowGraph::findBasicBlock(Block *block) const {
  auto it = blockToId.find(block);
  if (it != blockToId.end()) {
    return basicBlocks[it->second].get();
  }
  return nullptr;
}

void ControlFlowGraph::addEdge(size_t fromId, size_t toId) {
  if (fromId >= basicBlocks.size() || toId >= basicBlocks.size()) {
    return;
  }
  basicBlocks[fromId]->addSuccessor(toId);
  basicBlocks[toId]->addPredecessor(fromId);
}

void ControlFlowGraph::traverse(BlockVisitor visitor) {
  for (auto &bb : basicBlocks) {
    visitor(*bb);
  }
}

void ControlFlowGraph::print(raw_ostream &os) const {
  auto funcName = const_cast<triton::FuncOp&>(function).getName();
  os << "Control Flow Graph for function '" << (funcName.empty() ? "unnamed" : funcName) << "'\n";
  os << "========================================\n";
  os << "Number of blocks: " << basicBlocks.size() << "\n\n";

  for (const auto &bb : basicBlocks) {
    bb->print(os);
    os << "\n";
  }
}

void ControlFlowGraph::dump() const { print(llvm::errs()); }

void ControlFlowGraph::exportDOT(raw_ostream &os) const {
  auto funcName = const_cast<triton::FuncOp&>(function).getName();
  std::string funcNameStr = funcName.empty() ? "unnamed" : funcName.str();
  os << "digraph CFG_" << funcNameStr << " {\n";
  os << "  label=\"CFG for " << funcNameStr << "\";\n";
  os << "  labelloc=t;\n";
  os << "  rankdir=TB;\n\n";

  // 设置节点样式
  for (const auto &bb : basicBlocks) {
    os << "  " << bb->getId() << " [";
    os << "label=\"Block " << bb->getId() << "\\n";
    os << "(type: ";
    switch (bb->getType()) {
    case BlockType::ENTRY:
      os << "ENTRY";
      break;
    case BlockType::EXIT:
      os << "EXIT";
      break;
    case BlockType::NORMAL:
      os << "NORMAL";
      break;
    case BlockType::LOOP_HEADER:
      os << "LOOP_HEADER";
      break;
    case BlockType::LOOP_BODY:
      os << "LOOP_BODY";
      break;
    case BlockType::LOOP_EXIT:
      os << "LOOP_EXIT";
      break;
    case BlockType::IF_THEN:
      os << "IF_THEN";
      break;
    case BlockType::IF_ELSE:
      os << "IF_ELSE";
      break;
    }
    os << ")\", ";

    // 根据类型设置颜色
    switch (bb->getType()) {
    case BlockType::ENTRY:
      os << "style=filled, fillcolor=lightgreen";
      break;
    case BlockType::EXIT:
      os << "style=filled, fillcolor=lightcoral";
      break;
    case BlockType::LOOP_HEADER:
      os << "style=filled, fillcolor=lightblue";
      break;
    default:
      os << "shape=box";
      break;
    }
    os << "];\n";
  }

  os << "\n";

  // 输出边
  for (const auto &bb : basicBlocks) {
    for (auto succId : bb->getSuccessors()) {
      os << "  " << bb->getId() << " -> " << succId << ";\n";
    }
  }

  os << "}\n";
}

llvm::Error ControlFlowGraph::exportToFile(StringRef filename) const {
  std::error_code ec;
  llvm::raw_fd_ostream os(filename, ec, llvm::sys::fs::OF_Text);

  if (ec) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to open file: " + filename);
  }

  // 判断文件扩展名
  if (filename.ends_with(".dot")) {
    exportDOT(os);
  } else {
    print(os);
  }

  os.close();
  return llvm::Error::success();
}
