/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * RegionSelector实现
 * 负责选择需要yield的值和计算else分支的返回值
 */

#include "PipelineAnalyzer.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace triton {
namespace affinity {

// ==================== RegionSelector ====================

void RegionSelector::selectYieldValues(MergedRegion &region) {
  // yield values已经在PipelineAnalyzer::computeYieldValues中计算
  // 这里可以添加额外的选择逻辑
}

Value RegionSelector::findIterArgSource(Value v, Type expectedType) {
  SmallVector<Value> worklist = {v};
  SmallPtrSet<Value, 16> visited;

  while (!worklist.empty()) {
    Value cur = worklist.front();
    worklist.erase(worklist.begin());
    if (!visited.insert(cur).second)
      continue;

    // 匹配scf.for原始迭代参数, 直接返回
    if (auto b = dyn_cast<BlockArgument>(cur)) {
      auto forOp = dyn_cast<scf::ForOp>(b.getOwner()->getParentOp());
      if (forOp && b.getType() == expectedType) {
        for (Value iterArg : forOp.getRegionIterArgs()) {
          if (iterArg.getAsOpaquePointer() == b.getAsOpaquePointer()) {
            return b;
          }
        }
      }
    }

    Operation *defOp = cur.getDefiningOp();
    if (!defOp)
      continue;

    // 核心逻辑：如果当前值是scf.if的结果
    // 进入then块找源头
    if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      Block &thenBlock = ifOp.getThenRegion().front();
      // 找到then块最后一个op（scf.yield）
      // 取其operands（即ifOp结果的源头值）
      for (auto &innerOp : llvm::reverse(thenBlock)) {
        if (auto yieldOp = dyn_cast<scf::YieldOp>(&innerOp)) {
          // 按索引匹配: cur是ifOp的第n个结果, 取yieldOp的第n个operand
          for (auto [idx, res] : llvm::enumerate(ifOp.getResults())) {
            if (res.getAsOpaquePointer() == cur.getAsOpaquePointer()) {
              Value srcVal = yieldOp.getOperand(idx);
              if (!visited.count(srcVal))
                worklist.push_back(srcVal);
              break;
            }
          }
          break; // 找到yield即退出, 无需遍历其他op
        }
      }
    } else {
      // 非if结果值，正常往前追溯operands
      for (Value operand : defOp->getOperands()) {
        if (!visited.count(operand))
          worklist.push_back(operand);
      }
    }
  }

  return v;
}

void RegionSelector::computeElseYieldValues(
    const MergedRegion &region,
    SmallVector<Value> &elseValues,
    ArrayRef<Value> dependValues) {
  // 获取forOp
  if (region.yieldValues.empty())
    return;

  Operation *defOp = region.yieldValues[0].getDefiningOp();
  if (!defOp)
    return;

  auto forOp = dyn_cast<scf::ForOp>(defOp->getBlock()->getParentOp());
  if (!forOp)
    return;

  auto iterArgs = forOp.getRegionIterArgs();
  auto forYieldValues = forOp.getYieldedValues();

  // 新增的与dependvalue相关的initarg是接在原本for循环args后面
  int baseDependIdx = iterArgs.size() - dependValues.size();

  int idx = 0;
  for (Value v : region.yieldValues) {
    Type yieldType = region.resultTypes[idx];

    // yieldValue中是dependvalue的情况下
    // else yield value使用对应的新增iterargs
    if (llvm::is_contained(dependValues, v)) {
      int dependIdx = 0;
      for (; dependIdx < static_cast<int>(dependValues.size()); dependIdx++) {
        if (v == dependValues[dependIdx]) {
          break;
        }
      }
      elseValues.push_back(iterArgs[baseDependIdx + dependIdx]);
    } else {
      elseValues.push_back(findIterArgSource(v, yieldType));
    }
    idx++;
  }
}

bool RegionSelector::isUsedOutsideRegion(Value v, const MergedRegion &region) {
  SmallPtrSet<Operation *, 16> opSet(region.opsToMove.begin(),
                                     region.opsToMove.end());

  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    if (!opSet.contains(user)) {
      return true;
    }
  }
  return false;
}

} // namespace affinity
} // namespace triton
} // namespace mlir
