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

#ifndef TRITON_TO_CFG_CONTROL_FLOW_GRAPH_H
#define TRITON_TO_CFG_CONTROL_FLOW_GRAPH_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>

namespace mlir {
namespace triton {
namespace cfg {

// 基本块类型
enum class BlockType {
  ENTRY,       // 入口块
  EXIT,        // 出口块
  NORMAL,      // 普通块
  LOOP_HEADER, // 循环头
  LOOP_BODY,   // 循环体
  LOOP_EXIT,   // 循环出口
  IF_THEN,     // if-then 分支
  IF_ELSE,     // if-else 分支
};

// 基本块节点
class BasicBlockNode {
public:
  BasicBlockNode(size_t id, Block *block, BlockType type)
      : id(id), block(block), type(type) {}

  // 获取基本信息
  size_t getId() const { return id; }
  Block *getBlock() const { return block; }
  BlockType getType() const { return type; }
  void setType(BlockType t) { type = t; }

  // 获取后继块列表（用于遍历）
  ArrayRef<size_t> getSuccessors() const { return successors; }
  ArrayRef<size_t> getPredecessors() const { return predecessors; }

  // 获取名称
  StringRef getName() const;

  // 边的操作
  void addSuccessor(size_t succId) { successors.push_back(succId); }
  void addPredecessor(size_t predId) { predecessors.push_back(predId); }

  size_t getNumSuccessors() const { return successors.size(); }
  size_t getNumPredecessors() const { return predecessors.size(); }

  // 获取后继和前驱的访问器
  SmallVector<size_t>::const_iterator successor_begin() const {
    return successors.begin();
  }
  SmallVector<size_t>::const_iterator successor_end() const {
    return successors.end();
  }
  SmallVector<size_t>::const_iterator predecessor_begin() const {
    return predecessors.begin();
  }
  SmallVector<size_t>::const_iterator predecessor_end() const {
    return predecessors.end();
  }

  // 打印
  void print(raw_ostream &os) const;
  void dump() const;

private:
  size_t id;                          // 唯一ID
  Block *block;                       // 对应的 MLIR Block
  BlockType type;                     // 块类型
  SmallVector<size_t> successors;     // 后继块ID列表
  SmallVector<size_t> predecessors;   // 前驱块ID列表
};

// 控制流图
class ControlFlowGraph {
public:
  explicit ControlFlowGraph(triton::FuncOp func);
  ~ControlFlowGraph();

  // 获取函数
  triton::FuncOp getFunction() const { return function; }

  // 基本块操作
  size_t addBasicBlock(Block *block, BlockType type = BlockType::NORMAL);

  BasicBlockNode &getBasicBlock(size_t id) { return *basicBlocks[id]; }
  const BasicBlockNode &getBasicBlock(size_t id) const {
    return *basicBlocks[id];
  }

  BasicBlockNode *findBasicBlock(Block *block);
  const BasicBlockNode *findBasicBlock(Block *block) const;

  size_t getNumBlocks() const { return basicBlocks.size(); }

  // 边的操作
  void addEdge(size_t fromId, size_t toId);

  // 遍历
  using BlockVisitor = llvm::function_ref<void(BasicBlockNode &)>;
  void traverse(BlockVisitor visitor);

  // 打印
  void print(raw_ostream &os) const;
  void dump() const;

  // 导出为 DOT 格式
  void exportDOT(raw_ostream &os) const;

  // 导出到文件
  llvm::Error exportToFile(StringRef filename) const;

private:
  triton::FuncOp function;                                      // 所属函数
  SmallVector<std::unique_ptr<BasicBlockNode>> basicBlocks; // 基本块节点
  llvm::MapVector<Block *, size_t> blockToId;               // Block 到 ID 的映射
};

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_CFG_CONTROL_FLOW_GRAPH_H
