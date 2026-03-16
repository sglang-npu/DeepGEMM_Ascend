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

#include "TritonToCFG/memory_ssa_builder.h"
#include "TritonToCFG/ControlFlowGraph.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Ops.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "memory-ssa-builder"

using namespace mlir;
using namespace triton;
using namespace cfg;

//===----------------------------------------------------------------------===//
// MemorySSABuilder
//===----------------------------------------------------------------------===//

MemorySSABuilder::~MemorySSABuilder() {
  // 清理所有创建的tensor objects（如果有动态分配的）
  // 这里假设TensorObject由外部管理生命周期
}

void MemorySSABuilder::build() {
  LLVM_DEBUG(llvm::dbgs() << "=== Starting Memory SSA Build ===\n");

  // 步骤1: 创建函数的参数定义
  createParameterDefinitions();

  LLVM_DEBUG(llvm::dbgs() << "Created parameter definitions\n");

  // 步骤2: 遍历所有基本块，构建Memory SSA
  size_t processedBlocks = 0;
  cfg.traverse([&](BasicBlock& bb) {
    LLVM_DEBUG(llvm::dbgs() << "Processing BB" << bb.getId() << "\n");

    // 在block入口处合并前驱数据流（处理phi节点）
    if (bb.getNumPredecessors() > 1) {
      mergePredecessorDataFlow(&bb);
    }

    // 处理当前block
    processBasicBlock(&bb);

    processedBlocks++;
  });

  LLVM_DEBUG(llvm::dbgs()
             << "=== Memory SSA Build Complete ===\n"
             << "Processed blocks: " << processedBlocks << "\n"
             << "Total definitions: " << allDefinitions.size() << "\n");
}

void MemorySSABuilder::processBasicBlock(BasicBlock* bb) {
  if (!bb) return;

  // 处理block内的所有指令
  for (auto& instPtr : bb->getInstructions()) {
    Instruction* inst = instPtr.get();
    processInstruction(inst);
  }
}

void MemorySSABuilder::processInstruction(Instruction* inst) {
  if (!inst) return;

  Operation* op = inst->getOperation();
  if (!op) return;

  LLVM_DEBUG(llvm::dbgs() << "Processing: " << op->getName() << "\n");

  MemorySSAInfo& ssaInfo = inst->getMemorySSAInfo();

  // 1. 处理operands：创建uses
  LLVM_DEBUG(llvm::dbgs() << "  Processing operands...\n");

  for (OpOperand& operand : op->getOpOperands()) {
    Value operandValue = operand.get();
    unsigned operandIdx = operand.getOperandNumber();

    // 检查是否是tensor或pointer类型
    if (isTensorType(operandValue.getType()) ||
        aliasAnalysis.isPointerType(operandValue.getType())) {

      // 查找operand的definition
      Definition* def = dataFlowInfo.getMemoryDefinition(operandValue);

      if (def) {
        // 创建use
        Use use = createUse(def, op, operandIdx);
        ssaInfo.uses.push_back(use);

        // 记录到全局map
        dataFlowInfo.addMemoryUse(operandValue, use);

        LLVM_DEBUG(llvm::dbgs() << "    Use: " << def->getId() << " in "
                                << op->getName() << " [operand #"
                                << operandIdx << "]\n");
      }
    }
  }

  // 2. 处理results：创建definitions
  LLVM_DEBUG(llvm::dbgs() << "  Processing results...\n");

  for (Value result : op->getResults()) {
    Type resultType = result.getType();

    // 检查是否是tensor类型
    if (isTensorType(resultType)) {
      // 判断是否需要创建新的definition
      if (shouldCreateNewTensorObject(op)) {
        // 创建tensor对象
        TensorObject* tensor = createTensorObject(op);

        // 判断是否可以是写操作
        if (isMemoryWriter(op)) {
          // 创建新definition
          Definition* newDef = createDefinition(tensor, op);
          ssaInfo.definitions.push_back(newDef);

          // 记录到全局map
          dataFlowInfo.addMemoryDefinition(result, newDef);

          LLVM_DEBUG(llvm::dbgs()
                     << "    Definition: " << newDef->getId() << " created by "
                     << op->getName() << "\n");
        } else if (isAliasOp(op)) {
          // 别名操作，复用base pointer的definition
          Value basePtr = aliasAnalysis.getBasePointer(result);
          Definition* baseDef = dataFlowInfo.getMemoryDefinition(basePtr);

          if (baseDef) {
            // result与base使用相同的definition
            ssaInfo.definitions.push_back(baseDef);
            dataFlowInfo.addMemoryDefinition(result, baseDef);

            LLVM_DEBUG(llvm::dbgs()
                       << "    Alias Definition: " << baseDef->getId()
                       << " for " << result << "\n");
          }
        }
      }
    }
  }

  // 3. 特殊处理控制流操作
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    processIfOp(ifOp, inst, nullptr, nullptr);
  } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    processForOp(forOp, inst, nullptr);
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    processWhileOp(whileOp, inst, nullptr, nullptr);
  }

  LLVM_DEBUG(llvm::dbgs() << "  Done\n");
}

void MemorySSABuilder::createParameterDefinitions() {
  triton::FuncOp func = cfg.getFunction();

  LLVM_DEBUG(llvm::dbgs()
             << "Creating parameter definitions for " << func.getName()
             << "\n");

  // 遍历函数参数
  for (BlockArgument arg : func.getArguments()) {
    Type argType = arg.getType();

    // 检查是否是我们关心的类型
    if (isTensorType(argType)) {
      // 创建参数名称
      std::string paramName = "param_" + std::to_string(arg.getArgNumber());

      // 如果是指针类型，从aliasAnalysis获取tensor对象
      TensorObject* tensor = nullptr;
      if (aliasAnalysis.isPointerType(argType)) {
        tensor = aliasAnalysis.getTensorObject(arg);
      }

      // 如果没有找到tensor对象，创建一个新的
      if (!tensor) {
        // 从类型推断shape
        SmallVector<int64_t> shape;
        if (auto ptrType = argType.dyn_cast<triton::PointerType>()) {
          Type pointeeType = ptrType.getPointeeType();
          if (auto rankedType = pointeeType.dyn_cast<RankedTensorType>()) {
            shape.append(rankedType.getShape().begin(),
                         rankedType.getShape().end());
          }
        } else if (auto rankedType = argType.dyn_cast<RankedTensorType>()) {
          shape.append(rankedType.getShape().begin(),
                       rankedType.getShape().end());
        }

        tensor = new TensorObject(paramName, shape, argType,
                                 TensorObject::TensorKind::GLOBAL_MEMORY);
      }

      // 为入参创建definition
      Definition* def = createDefinition(tensor, nullptr);

      // 记录到dataFlowInfo
      dataFlowInfo.addMemoryDefinition(arg, def);

      LLVM_DEBUG(llvm::dbgs()
                 << "  Created parameter definition: " << def->getId()
                 << "\n");
    }
  }
}

Definition* MemorySSABuilder::createDefinition(TensorObject* tensor,
                                               Operation* op) {
  unsigned version = isParameter(op) ? 0 : nextVersionId++;
  auto* def = new Definition(tensor, op, version);
  allDefinitions.push_back(def);
  return def;
}

Use MemorySSABuilder::createUse(Definition* def, Operation* userOp,
                                 unsigned operandIdx) {
  return Use(def, userOp, operandIdx);
}

TensorObject* MemorySSABuilder::createTensorObject(Operation* op) {
  if (!op) return nullptr;

  // 根据操作创建tensor对象
  std::string name = getOpName(op);
  Type resultType = op->getResultTypes().front();

  SmallVector<int64_t> shape;
  if (auto rankedType = resultType.dyn_cast<RankedTensorType>()) {
    shape.append(rankedType.getShape().begin(), rankedType.getShape().end());
  } else if (auto ptrType = resultType.dyn_cast<triton::PointerType>()) {
    Type pointeeType = ptrType.getPointeeType();
    if (auto rankedType = pointeeType.dyn_cast<RankedTensorType>()) {
      shape.append(rankedType.getShape().begin(),
                   rankedType.getShape().end());
    }
  }

  // 设置默认的kind（可以根据操作类型推断）
  TensorObject::TensorKind kind = TensorObject::TensorKind::GLOBAL_MEMORY;

  auto* tensor = new TensorObject(name, shape, resultType, kind);

  // 缓存tensor对象
  if (!op->getResults().empty()) {
    tensorObjectCache[op->getResult(0)] = tensor;
  }

  return tensor;
}

std::string MemorySSABuilder::getOpName(Operation* op) const {
  if (!op) return "unknown";

  // 基于操作类型和ID生成名称
  std::string opName = op->getName().getStringRef().str();
  std::replace(opName.begin(), opName.end(), '.', '_');

  return opName + "_" + std::to_string(nextVersionId);
}

bool MemorySSABuilder::shouldCreateNewTensorObject(Operation* op) const {
  if (!op) return false;

  // 检查是否有可能产生tensor结果的操作
  if (op->getNumResults() == 0) return false;

  // 只处理有结果的operations
  Type resultType = op->getResultTypes().front();
  return isTensorType(resultType);
}

//===----------------------------------------------------------------------===//
// 简单的数据处理
//===----------------------------------------------------------------------===//

namespace MemorySSABuilderHelper {

Type getResultType(Operation* op, unsigned resultIdx) {
  if (!op || resultIdx >= op->getNumResults()) return Type();
  return op->getResultTypes()[resultIdx];
}

SmallVector<int64_t> getShapeFromValue(Value value) {
  Type type = value.getType();
  SmallVector<int64_t> shape;

  if (auto rankedType = type.dyn_cast<RankedTensorType>()) {
    shape.append(rankedType.getShape().begin(),
                 rankedType.getShape().end());
  }

  return shape;
}

bool shapesEqual(ArrayRef<int64_t> shape1, ArrayRef<int64_t> shape2) {
  if (shape1.size() != shape2.size()) return false;
  return std::equal(shape1.begin(), shape1.end(), shape2.begin());
}

Operation* getYieldOp(Region& region) {
  if (region.empty()) return nullptr;

  Block& block = region.back();
  if (block.empty()) return nullptr;

  Operation& lastOp = block.back();
  if (isa<scf::YieldOp>(&lastOp)) {
    return &lastOp;
  }

  return nullptr;
}

std::string createUniqueTensorName(StringRef prefix, size_t id) {
  return prefix.str() + "_" + std::to_string(id);
}

bool shouldCreateNewVersion(Operation* op, Definition* currentDef) {
  if (!op || !currentDef) return true;

  // 如果操作会修改tensor内容，则创建新版本
  // 例如：tt.store、tt.trans等
  if (isa<triton::StoreOp>(op)) return true;
  if (isa<triton::TransOp>(op)) return true;

  // 其他操作可能复用当前definition
  return false;
}

} // namespace MemorySSABuilderHelper
