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

#include "AliasAnalysis.h"
#include "MemorySSA.h"
#include "ControlFlowGraph.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "alias-analysis"

using namespace mlir;
using namespace triton;
using namespace cfg;

//===----------------------------------------------------------------------===//
// AliasAnalysis Implementation
//===----------------------------------------------------------------------===//

void AliasAnalysis::analyzePointerAliases(ControlFlowGraph& cfg) {
  LLVM_DEBUG(llvm::dbgs() << "=== Starting Pointer Alias Analysis ===\n");

  // 步骤1: 识别全局内存参数
  triton::FuncOp func = cfg.getFunction();

  LLVM_DEBUG(llvm::dbgs() << "Analyzing function: " << func.getName() << "\n");

  for (BlockArgument arg : func.getArguments()) {
    Type argType = arg.getType();

    if (isPointerType(argType) && isGlobalMemoryType(argType)) {
      // 为入参创建tensor对象
      std::string paramName = "param_" + std::to_string(arg.getArgNumber());

      SmallVector<int64_t> shape;
      Type elementType;

      // 使用辅助函数提取shape和element type
      extractShapeAndElementType(argType, shape, elementType);

      TensorObject* tensor = new TensorObject(
          paramName, shape, argType, elementType,
          TensorObject::TensorKind::GLOBAL_MEMORY);

      // 记录alias关系 [param, param, tensor]
      addAlias(arg, arg, tensor);

      LLVM_DEBUG(llvm::dbgs() << "  Found global memory parameter "
                              << arg.getArgNumber() << ": " << tensor->getName() << "\n");
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Found " << aliasMap.size() << " global memory parameters\n");

  // 步骤2: 分析alias操作
  size_t aliasOpsFound = 0;
  cfg.traverse([&](BasicBlock& bb) {
    for (const auto& instPtr : bb.getInstructions()) {
      const Instruction* inst = instPtr.get();
      Operation* op = inst->getOperation();

      if (!op) continue;

      if (auto addptrOp = dyn_cast<triton::AddPtrOp>(op)) {
        analyzeAddPtrOp(addptrOp);
        aliasOpsFound++;
      } else if (auto makeTensorPtrOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
        analyzeMakeTensorPtrOp(makeTensorPtrOp);
        aliasOpsFound++;
      } else if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
        analyzeLoadOp(loadOp);
      } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
        analyzeStoreOp(storeOp);
      }
    }
  });

  LLVM_DEBUG(llvm::dbgs()
             << "=== Alias Analysis Complete ===\n"
             << "Total tracked aliases: " << aliasMap.size() << "\n"
             << "Alias operations found: " << aliasOpsFound << "\n");
}

Value AliasAnalysis::getBasePointer(Value ptr) const {
  // 递归查找base pointer
  auto it = aliasMap.find(ptr);
  if (it == aliasMap.end()) {
    // 没有找到，返回自身（可能是原始base pointer）
    return ptr;
  }

  Value base = it->second;

  // 如果base不是自己，继续递归查找
  if (base != ptr) {
    return getBasePointer(base);
  }

  return base;
}

void AliasAnalysis::analyzeAddPtrOp(mlir::triton::AddPtrOp addptrOp) {
  // %new = tt.addptr %ptr, %offset
  Value ptr = addptrOp.getPtr();
  Value result = addptrOp.getResult();

  // result是ptr的alias，指向同一个tensor
  Value basePtr = getBasePointer(ptr);
  TensorObject* tensor = getTensorObject(basePtr);

  if (tensor) {
    addAlias(result, basePtr, tensor);

    LLVM_DEBUG(llvm::dbgs() << "  AddPtr: " << result
                            << " -> " << basePtr
                            << " [" << tensor->getName() << "]\n");
  }
}

void AliasAnalysis::analyzeMakeTensorPtrOp(mlir::triton::MakeTensorPtrOp op) {
  // %tensor_ptr = tt.make_tensor_ptr %base, ...
  Value basePtr = op.getBase();
  Value result = op.getResult();

  TensorObject* tensor = getTensorObject(basePtr);

  if (tensor) {
    addAlias(result, basePtr, tensor);

    LLVM_DEBUG(llvm::dbgs() << "  MakeTensorPtr: " << result
                            << " -> " << basePtr
                            << " [" << tensor->getName() << "]\n");
  }
}

void AliasAnalysis::analyzeLoadOp(mlir::triton::LoadOp loadOp) {
  // tt.load操作通常不改变alias关系，
  // 但可以验证load的ptr是否被正确跟踪

  Value ptr = loadOp.getPtr();

  // 验证ptr是否被alias analysis跟踪
  TensorObject* tensor = getTensorObject(ptr);

  if (tensor) {
    LLVM_DEBUG(llvm::dbgs() << "  Load from tracked tensor: "
                            << tensor->getName() << "\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "  Load from untracked pointer\n");
  }
}

void AliasAnalysis::analyzeStoreOp(mlir::triton::StoreOp storeOp) {
  // tt.store操作不改变alias关系，
  // 但可以验证store的目标是否被正确跟踪

  Value ptr = storeOp.getPtr();

  // 验证ptr是否被alias analysis跟踪
  TensorObject* tensor = getTensorObject(ptr);

  if (tensor) {
    LLVM_DEBUG(llvm::dbgs() << "  Store to tracked tensor: "
                            << tensor->getName() << "\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "  Store to untracked pointer\n");
  }
}
