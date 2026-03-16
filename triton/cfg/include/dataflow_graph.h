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

#ifndef TRITON_TO_CFG_DATAFLOW_GRAPH_H
#define TRITON_TO_CFG_DATAFLOW_GRAPH_H

#include "TritonToCFG/memory_ssa.h"
#include "TritonToCFG/ControlFlowGraph.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace mlir {
namespace triton {
namespace cfg {

// 前向声明
class AliasAnalysis;
class MemorySSABuilder;

// DataFlowInfo - 统一的数据流信息
class DataFlowInfo {
public:
  // 为函数入口创建参数定义
  void createParameterDefinitions(triton::FuncOp func);

  // Memory SSA接口
  Definition* getMemoryDefinition(Value value) const;
  void addMemoryDefinition(Value value, Definition* def);

  SmallVector<Use> getMemoryUses(Value value) const;
  void addMemoryUse(Value value, const Use& use);

  void removeMemoryDefinition(Value value);
  void clearMemoryUses(Value value);

  // 传统SSA接口（复用MLIR原生功能）
  Operation* getSSADefinition(Value value) const {
    return value.getDefiningOp();
  }

  SmallVector<OpOperand*> getSSAUses(Value value) const {
    SmallVector<OpOperand*> result;
    for (OpOperand& use : value.getUses()) {
      result.push_back(&use);
    }
    return result;
  }

  // 循环Phi接口
  void addLoopPhi(Value value, const LoopPhiInfo& phiInfo) {
    loopPhis[value] = phiInfo;
  }

  LoopPhiInfo& getLoopPhi(Value value) {
    return loopPhis[value];
  }

  bool hasLoopPhi(Value value) const {
    return loopPhis.count(value) > 0;
  }

  // 统一查询接口
  struct DataFlowResult {
    enum class ResultKind {
      MEMORY_SSA,     // Memory SSA结果（tensor/pointer）
      SSA,            // 传统SSA结果（标量）
      NONE            // 无数据流信息
    };

    ResultKind kind;
    Operation* originOp;          // 定义该值的操作
    Definition* definition;       // MEMORY_SSA时使用
    Operation* ssaDefinition;     // SSA时使用
    SmallVector<OpOperand*> uses; // 所有uses
    std::optional<LoopPhiInfo> loopPhi; // 循环phi信息（如果有）
  };

  DataFlowResult queryDataFlow(Value value) const;

  // 查询某个定义的所有使用
  SmallVector<Use> getUses(Definition* def) const;

  // 查询某个操作的memory使用
  SmallVector<Use> getUsesByUserOp(Operation* userOp) const;

  // 遍历接口
  void forEachDefinition(llvm::function_ref<void(Value, Definition*)> func) const;
  void forEachUse(llvm::function_ref<void(const Use&)> func) const;

  // 获取所有Memory SSA definitions
  const DenseMap<Value, Definition*>& getMemoryDefinitions() const {
    return memoryDefinitions;
  }

  // 获取所有循环phi信息
  const DenseMap<Value, LoopPhiInfo>& getLoopPhis() const {
    return loopPhis;
  }

  // 打印信息（调试用）
  void print(llvm::raw_ostream& os) const;

  // 导出到JSON
  void exportToJSON(llvm::raw_ostream& os) const;

private:
  // Memory SSA映射
  DenseMap<Value, Definition*> memoryDefinitions;
  DenseMap<Value, SmallVector<Use>> memoryUses;

  // Loop Phi映射
  DenseMap<Value, LoopPhiInfo> loopPhis;

  // Use-Def映射缓存（def -> uses）
  mutable DenseMap<Definition*, SmallVector<Use>> defUseCache;
  mutable bool defUseCacheValid = false;

  // 构建def-use缓存
  void buildDefUseCache() const;
  void invalidateDefUseCache() { defUseCacheValid = false; defUseCache.clear(); }
};

// DataFlowGraph - 数据流图
class DataFlowGraph {
public:
  explicit DataFlowGraph(ControlFlowGraph& cfg)
      : cfg(cfg) {}

  ~DataFlowGraph() = default;

  // 构建完整的数据流信息
  void build();

  // 查询Value的数据流信息
  DataFlowInfo::DataFlowResult queryDataFlow(Value value) const {
    return dataFlowInfo.queryDataFlow(value);
  }

  // 获取所有Memory SSA definitions
  SmallVector<Definition*> getAllDefinitions() const {
    SmallVector<Definition*> result;
    for (const auto& entry : dataFlowInfo.getMemoryDefinitions()) {
      result.push_back(entry.second);
    }
    return result;
  }

  // 获取definition的所有uses
  SmallVector<Use> getUses(Definition* def) const {
    return dataFlowInfo.getUses(def);
  }

  // 获取操作的所有uses
  SmallVector<Use> getUsesByUserOp(Operation* userOp) const {
    return dataFlowInfo.getUsesByUserOp(userOp);
  }

  // 获取CFG
  ControlFlowGraph& getCFG() { return cfg; }
  const ControlFlowGraph& getCFG() const { return cfg; }

  // 获取DataFlowInfo
  DataFlowInfo& getDataFlowInfo() { return dataFlowInfo; }
  const DataFlowInfo& getDataFlowInfo() const { return dataFlowInfo; }

  // 导出数据流信息到JSON
  void exportToJSON(llvm::raw_ostream& os) const;

  // 导出def-use链到DOT格式
  void exportDefUseToDOT(llvm::raw_ostream& os) const;

  // 打印所有数据流信息（调试用）
  void print(llvm::raw_ostream& os) const;
  void dump() const;

private:
  ControlFlowGraph& cfg;

  // 组件
  std::unique_ptr<AliasAnalysis> aliasAnalysis;
  std::unique_ptr<MemorySSABuilder> memorySSABuilder;

  // 数据流信息
  DataFlowInfo dataFlowInfo;

  // 收集所有definitions
  void collectDefinitions();

  // 构建def-use图
  void buildDefUseGraph();
};

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_CFG_DATAFLOW_GRAPH_H
