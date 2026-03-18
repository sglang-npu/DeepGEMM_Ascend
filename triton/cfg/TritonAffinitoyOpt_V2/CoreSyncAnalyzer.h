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

#ifndef TRITON_AFFINITY_CORE_SYNC_ANALYZER_H
#define TRITON_AFFINITY_CORE_SYNC_ANALYZER_H

#include "TritonToGraph/ControlFlowGraph.h"
#include "TritonToGraph/DataflowGraph.h"
#include "TritonToGraph/AliasAnalysis.h"
#include "TritonAffinityOpt/DAG.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace triton {
namespace affinity {

using CoreType = AffinityDAG::CoreType;
using namespace cfg;

// 核心类型分类器 - 统一处理操作的核心类型识别
class CoreTypeClassifier {
public:
  CoreTypeClassifier(llvm::DenseMap<Value, CoreType>* types)
      : valueTypes(types) {}

  CoreType getType(Value value) const;
  CoreType getType(Operation* op) const;

  bool isVector(Value value) const { return getType(value) == CoreType::VECTOR; }
  bool isCube(Value value) const { return getType(value) == CoreType::CUBE; }
  bool isScalar(Value value) const { return getType(value) == CoreType::SCALAR; }

  bool needsSync(CoreType src, CoreType dst) const {
    return (src == CoreType::VECTOR && dst == CoreType::CUBE) ||
           (src == CoreType::CUBE && dst == CoreType::VECTOR);
  }

  bool isSyncOp(Operation* op) const {
    return isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op);
  }

  bool isControlFlow(Operation* op) const {
    return isa<scf::ForOp>(op) || isa<scf::IfOp>(op) ||
           isa<scf::WhileOp>(op) || isa<scf::YieldOp>(op);
  }

private:
  llvm::DenseMap<Value, CoreType>* valueTypes;
};

// 同步点信息
struct SyncPoint {
  Operation* srcOp;
  Operation* dstOp;
  CoreType srcType;
  CoreType dstType;
  bool crossBlock;
  int flag;
};

// 数据搬运信息
struct DataMovement {
  enum Type { CUBE_TO_VECTOR, VECTOR_TO_CUBE };
  Type type;
  Value source;
  Value target;
  Operation* insertAfter;
  Operation* insertBefore;
};

// 核心同步分析器 - 基于CFG和DataFlowGraph分析同步需求
class CoreSyncAnalyzer {
public:
  CoreSyncAnalyzer(ControlFlowGraph& cfg, DataFlowInfo& dfi,
                   llvm::DenseMap<Value, CoreType>* types)
      : cfg(cfg), dataFlowInfo(dfi), classifier(types), nextFlag(1) {}

  // 分析所有同步点
  void analyze();

  // 获取分析结果
  const SmallVector<SyncPoint>& getSyncPoints() const { return syncPoints; }
  const SmallVector<DataMovement>& getDataMovements() const { return dataMovements; }

  // 获取下一个可用的flag
  int getNextFlag() { return nextFlag++; }
  int getCurrentFlag() const { return nextFlag; }

private:
  ControlFlowGraph& cfg;
  DataFlowInfo& dataFlowInfo;
  CoreTypeClassifier classifier;
  int nextFlag;

  SmallVector<SyncPoint> syncPoints;
  SmallVector<DataMovement> dataMovements;

  // 分析基本块内的同步需求
  void analyzeBlock(BasicBlock& bb);

  // 分析控制流结构内的同步需求
  void analyzeControlFlow(Operation* op);

  // 分析scf.for的迭代参数同步
  void analyzeIterArgs(scf::ForOp forOp);

  // 检查跨block依赖
  bool isCrossBlockDependency(Operation* src, Operation* dst) const;

  // 查找数据搬运需求
  void analyzeDataMovement(Operation* src, Operation* dst,
                           CoreType srcType, CoreType dstType);
};

// Scope构建器 - 使用CFG构建AIV/AIC Scope
class ScopeBuilder {
public:
  ScopeBuilder(ControlFlowGraph& cfg, CoreSyncAnalyzer& analyzer,
               llvm::DenseMap<Value, CoreType>* types)
      : cfg(cfg), analyzer(analyzer), classifier(types) {}

  // 构建AIV和AIC Scope
  std::pair<scope::ScopeOp, scope::ScopeOp> buildScopes(triton::FuncOp func);

private:
  ControlFlowGraph& cfg;
  CoreSyncAnalyzer& analyzer;
  CoreTypeClassifier classifier;

  // 收集需要移动到各scope的操作
  void collectOpsToMove(BasicBlock& bb,
                       SmallVector<Operation*>& aivOps,
                       SmallVector<Operation*>& cubeOps);

  // 创建Scope操作
  scope::ScopeOp createScope(OpBuilder& builder, Location loc,
                             hivm::TCoreType coreType);

  // 移动操作到scope
  void moveOpsToScope(SmallVector<Operation*>& ops,
                      scope::ScopeOp scope,
                      IRMapping& mapper);
};

// 同步操作插入器 - 在适当位置插入sync和wait操作
class SyncInserter {
public:
  SyncInserter(CoreSyncAnalyzer& analyzer,
               llvm::DenseMap<Value, CoreType>* types)
      : analyzer(analyzer), classifier(types) {}

  // 在scope中插入所有同步操作
  void insertSyncOps(scope::ScopeOp aivScope, scope::ScopeOp cubeScope);

  // 处理buffer wait的同步增强
  void addSyncForBufferWait(triton::FuncOp func);

private:
  CoreSyncAnalyzer& analyzer;
  CoreTypeClassifier classifier;

  // 插入单个同步点
  void insertSync(const SyncPoint& point, OpBuilder& builder);

  // 插入数据搬运操作
  void insertDataMovement(const DataMovement& dm, OpBuilder& builder);

  // 在region末尾插入wait
  void insertWaitAtRegionEnd(Region& region, int flag,
                             hivm::TCoreType coreType,
                             OpBuilder& builder);

  // 在region开头插入set
  void insertSetAtRegionStart(Region& region, int flag,
                              hivm::TCoreType coreType,
                              OpBuilder& builder);

  // 创建sync操作
  void createSyncBlockSet(OpBuilder& builder, Location loc,
                          hivm::TCoreType coreType,
                          hivm::PIPE setPipe,
                          hivm::PIPE waitPipe,
                          int64_t flag);

  void createSyncBlockWait(OpBuilder& builder, Location loc,
                           hivm::TCoreType coreType,
                           hivm::PIPE setPipe,
                           hivm::PIPE waitPipe,
                           int64_t flag);
};

} // namespace affinity
} // namespace triton
} // namespace mlir

#endif // TRITON_AFFINITY_CORE_SYNC_ANALYZER_H
