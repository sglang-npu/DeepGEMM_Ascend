/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * PipelineAnalyzer实现
 * 基于TritonToGraph框架的Pipeline分析
 */

#include "PipelineAnalyzer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace triton {
namespace affinity {

using namespace cfg;

// ==================== PipelineAnalyzer ====================

PipelineAnalysisResult PipelineAnalyzer::analyze(scf::ForOp forOp) {
  PipelineAnalysisResult result;
  Block& body = forOp.getRegion().front();

  // 1. 分析基本块中的wait-set regions
  for (auto& bb : cfg_.getBlocks()) {
    if (bb->getMLIRBlock() == &body) {
      analyzeBlockRegions(*bb, result.waitSetRegions);
      break;
    }
  }

  if (result.waitSetRegions.empty())
    return result;

  // 2. 合并wait-set regions
  mergeRegions(result.waitSetRegions, result.mergedRegions);

  // 3. 确定core类型
  CoreKind coreKind = getCoreKindForLoop(forOp);
  for (auto& mr : result.mergedRegions) {
    mr.coreKind = coreKind;
  }

  // 4. 扩展region操作（AIV/AIC不同策略）
  if (coreKind == CoreKind::VECTOR) {
    expandRegionsForAIV(forOp, result.mergedRegions);
  } else {
    expandRegionsForAIC(forOp, result.mergedRegions);
  }

  // 5. 计算yield values
  for (auto& mr : result.mergedRegions) {
    computeYieldValues(mr, body);
  }

  // 6. 识别跨region依赖
  identifyCrossRegionDeps(result.mergedRegions, result.dependValues);

  // 7. 建立op到region的映射
  result.opToRegion = computeOpIndices(body);
  for (size_t i = 0; i < result.mergedRegions.size(); ++i) {
    for (Operation* op : result.mergedRegions[i].opsToMove) {
      result.opToRegion[op] = static_cast<int>(i);
    }
  }

  return result;
}

CoreKind PipelineAnalyzer::getCoreKind(Operation* op) {
  // 检查操作是否在scope中，根据scope属性判断
  auto scopeOp = op->getParentOfType<scope::ScopeOp>();
  if (!scopeOp)
    return CoreKind::UNKNOWN;

  auto coreTypeAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(
      hivm::TCoreTypeAttr::name);
  if (!coreTypeAttr)
    return CoreKind::UNKNOWN;

  return coreTypeAttr.getTcoretype() == hivm::TCoreType::CUBE
             ? CoreKind::CUBE
             : CoreKind::VECTOR;
}

CoreKind PipelineAnalyzer::getCoreKindForLoop(scf::ForOp forOp) {
  auto scopeOp = forOp->getParentOfType<scope::ScopeOp>();
  if (!scopeOp)
    return CoreKind::UNKNOWN;

  auto coreTypeAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(
      hivm::TCoreTypeAttr::name);
  if (!coreTypeAttr)
    return CoreKind::UNKNOWN;

  return coreTypeAttr.getTcoretype() == hivm::TCoreType::CUBE
             ? CoreKind::CUBE
             : CoreKind::VECTOR;
}

bool PipelineAnalyzer::isTransferOp(Operation* op) {
  return isa<hivm::CopyOp>(op) || isa<hivm::FixpipeOp>(op);
}

bool PipelineAnalyzer::isSyncOp(Operation* op) {
  return isa<hivm::SyncBlockWaitOp>(op) || isa<hivm::SyncBlockSetOp>(op);
}

bool PipelineAnalyzer::isWaitOp(Operation* op) {
  return isa<hivm::SyncBlockWaitOp>(op);
}

bool PipelineAnalyzer::isSetOp(Operation* op) {
  return isa<hivm::SyncBlockSetOp>(op);
}

void PipelineAnalyzer::analyzeBlockRegions(
    BasicBlock& bb, SmallVector<WaitSetRegion>& regions) {
  Block* mlirBlock = bb.getMLIRBlock();
  if (!mlirBlock)
    return;

  // 遍历块中的所有操作，识别wait-set pairs
  for (auto it = mlirBlock->begin(); it != mlirBlock->end();) {
    Operation* op = &*it;

    // 找到wait操作作为起点
    if (!isWaitOp(op)) {
      ++it;
      continue;
    }

    Operation* waitOp = op;
    Operation* lastSetOp = nullptr;

    // 扫描到下一个wait，收集所有set
    auto curIt = std::next(it);
    auto endIt = curIt;
    int setOpCount = 0;
    SmallVector<Operation*> opsInRegion;

    for (; curIt != mlirBlock->end(); ++curIt) {
      Operation* curOp = &*curIt;
      if (isWaitOp(curOp) && setOpCount >= 1)
        break;
      if (isSetOp(curOp)) {
        setOpCount++;
        endIt = curIt;
        lastSetOp = curOp;
      }
    }

    if (!lastSetOp) {
      it = curIt;
      continue;
    }

    // 收集[wait, ..., lastSet]之间的所有操作
    bool hasTransferOp = false;
    for (auto it2 = it; it2 != std::next(endIt); ++it2) {
      Operation* curOp = &*it2;
      opsInRegion.push_back(curOp);
      if (isTransferOp(curOp)) {
        hasTransferOp = true;
      }
    }

    it = endIt;
    ++it;

    WaitSetRegion region;
    region.waitOp = waitOp;
    region.setOp = lastSetOp;
    region.ops = std::move(opsInRegion);
    region.hasTransferOp = hasTransferOp;
    region.coreKind = CoreKind::UNKNOWN;
    regions.push_back(std::move(region));
  }
}

void PipelineAnalyzer::mergeRegions(
    SmallVector<WaitSetRegion>& waitSetRegions,
    SmallVector<MergedRegion>& mergedRegions) {
  for (size_t i = 0; i < waitSetRegions.size();) {
    MergedRegion mr;
    mr.sourceRegions.push_back(&waitSetRegions[i]);
    mr.opsToMove.append(waitSetRegions[i].ops);

    size_t j = i;
    // 合并规则：如果当前region没有transfer op，则与下一个合并
    while (!waitSetRegions[j].hasTransferOp && j + 1 < waitSetRegions.size()) {
      j++;
      mr.sourceRegions.push_back(&waitSetRegions[j]);
      mr.opsToMove.append(waitSetRegions[j].ops);
    }

    mergedRegions.push_back(std::move(mr));
    i = j + 1;
  }
}

void PipelineAnalyzer::expandRegionsForAIV(
    scf::ForOp forOp, SmallVector<MergedRegion>& mergedRegions) {
  Block& body = forOp.getRegion().front();

  // 记录block中op的顺序
  DenseMap<Operation*, int> opIndex;
  int idx = 0;
  for (Operation& op : body)
    opIndex[&op] = idx++;

  // 建立op -> region映射
  DenseMap<Operation*, int> opToRegion;
  for (int r = 0; r < static_cast<int>(mergedRegions.size()); ++r) {
    for (Operation* op : mergedRegions[r].opsToMove) {
      opToRegion[op] = r;
    }
  }

  // 获取scf.yield
  auto yieldOp = cast<scf::YieldOp>(body.getTerminator());

  // 依次处理每个yield value（按编号顺序）
  for (Value yv : yieldOp.getOperands()) {
    Operation* defOp = yv.getDefiningOp();
    if (!defOp || defOp->getBlock() != &body)
      continue;

    int targetRegion = -1;

    // 如果已经在region中
    auto it = opToRegion.find(defOp);
    if (it != opToRegion.end()) {
      targetRegion = it->second;
    } else {
      // 否则向前搜索确定归属
      targetRegion = findTargetRegion(defOp, body, opToRegion);
    }

    if (targetRegion == -1)
      continue;

    // 计算边界lowerBound
    int lowerBound = 0;
    if (targetRegion > 0) {
      Operation* prevLast = mergedRegions[targetRegion - 1].opsToMove.back();
      lowerBound = opIndex[prevLast] + 1;
    }

    // 贪心吸收
    greedyAbsorbToRegion(defOp, targetRegion, lowerBound, body, opIndex,
                         opToRegion, mergedRegions);
  }

  // 每个region内按block顺序排序
  for (auto& mr : mergedRegions) {
    llvm::sort(mr.opsToMove, [&](Operation* a, Operation* b) {
      return opIndex[a] < opIndex[b];
    });
  }
}

void PipelineAnalyzer::expandRegionsForAIC(
    scf::ForOp forOp, SmallVector<MergedRegion>& mergedRegions) {
  Block& body = forOp.getRegion().front();

  // 记录每个mergedRegion的起始op index
  DenseMap<Operation*, int> opIndex;
  int idx = 0;
  for (Operation& op : body) {
    opIndex[&op] = idx++;
  }

  for (size_t r = 0; r < mergedRegions.size(); ++r) {
    MergedRegion& mr = mergedRegions[r];

    // 本mergedRegion的最早op
    Operation* firstOp = mr.opsToMove.front();
    int lowerBound = 0;

    // 边界：前一个mergedRegion的最后一个op
    if (r > 0) {
      Operation* prevLast = mergedRegions[r - 1].opsToMove.back();
      lowerBound = opIndex[prevLast] + 1;
    }

    SmallVector<Operation*> worklist(mr.opsToMove.begin(), mr.opsToMove.end());
    SmallPtrSet<Operation*, 32> visited(mr.opsToMove.begin(), mr.opsToMove.end());

    while (!worklist.empty()) {
      Operation* op = worklist.pop_back_val();

      // 往前吸收operand
      for (Value operand : op->getOperands()) {
        if (mlir::isa<BlockArgument>(operand))
          continue;

        Operation* defOp = operand.getDefiningOp();
        if (!defOp)
          continue;

        // 不在for body
        if (defOp->getBlock() != &body)
          continue;

        int defIdx = opIndex[defOp];

        // 超出允许向前吸收的边界
        if (defIdx < lowerBound)
          continue;

        // 已经在opsToMove
        if (!visited.insert(defOp).second)
          continue;

        // 吸收这个defOp
        mr.opsToMove.push_back(defOp);
        worklist.push_back(defOp);
      }
    }

    // 最后按原block顺序排序
    llvm::sort(mr.opsToMove, [&](Operation* a, Operation* b) {
      return opIndex[a] < opIndex[b];
    });
  }

  // AIC特殊处理：处理for op末尾的iter_arg users
  moveIterArgUsersForAIC(forOp, mergedRegions);
}

void PipelineAnalyzer::moveIterArgUsersForAIC(
    scf::ForOp forOp, SmallVector<MergedRegion>& mergedRegions) {
  Block& body = forOp.getRegion().front();

  // 建立iter_arg -> 使用它的region索引的映射
  DenseMap<BlockArgument, int> iterArgToRegion;

  for (size_t r = 0; r < mergedRegions.size(); ++r) {
    for (Operation* op : mergedRegions[r].opsToMove) {
      for (Value operand : op->getOperands()) {
        if (auto barg = dyn_cast<BlockArgument>(operand)) {
          if (barg.getOwner() == &body) {
            iterArgToRegion[barg] = static_cast<int>(r);
          }
        }
      }
    }
  }

  if (iterArgToRegion.empty())
    return;

  // 找最后一个mergedRegion的最后一个op
  Operation* lastOp = nullptr;
  for (MergedRegion& mr : mergedRegions) {
    lastOp = mr.opsToMove.back();
  }

  if (!lastOp)
    return;

  DenseMap<Operation*, int> opIndex;
  int idx = 0;
  for (Operation& op : body)
    opIndex[&op] = idx++;

  int startIdx = opIndex[lastOp] + 1;

  // 扫描for body尾部的op
  for (Operation& op : body) {
    if (opIndex[&op] < startIdx)
      continue;

    llvm::SmallDenseSet<int, 2> usedRegions;
    for (Value v : op.getOperands()) {
      if (auto barg = dyn_cast<BlockArgument>(v)) {
        auto it = iterArgToRegion.find(barg);
        if (it != iterArgToRegion.end())
          usedRegions.insert(it->second);
      }
    }

    // 必须且只能依赖一个mergedRegion
    if (usedRegions.size() != 1)
      continue;

    int target = *usedRegions.begin();
    mergedRegions[target].opsToMove.push_back(&op);
  }
}

int PipelineAnalyzer::findTargetRegion(
    Operation* startOp, Block& body,
    DenseMap<Operation*, int>& opToRegion) {
  SmallVector<Operation*> worklist{startOp};
  SmallPtrSet<Operation*, 16> visited;

  while (!worklist.empty()) {
    Operation* op = worklist.pop_back_val();
    if (!visited.insert(op).second)
      continue;

    auto it = opToRegion.find(op);
    if (it != opToRegion.end())
      return it->second;

    for (Value operand : op->getOperands()) {
      if (isa<BlockArgument>(operand))
        continue;

      Operation* defOp = operand.getDefiningOp();
      if (defOp && defOp->getBlock() == &body)
        worklist.push_back(defOp);
    }
  }

  return -1;
}

void PipelineAnalyzer::greedyAbsorbToRegion(
    Operation* startOp, int regionIdx, int lowerBound, Block& body,
    DenseMap<Operation*, int>& opIndex,
    DenseMap<Operation*, int>& opToRegion,
    SmallVector<MergedRegion>& mergedRegions) {
  auto& mr = mergedRegions[regionIdx];

  SmallVector<Operation*> worklist;
  SmallPtrSet<Operation*, 32> visited(mr.opsToMove.begin(), mr.opsToMove.end());

  // 先把startOp本身吸收（如果还没被吸收）
  if (!opToRegion.count(startOp)) {
    mr.opsToMove.push_back(startOp);
    opToRegion[startOp] = regionIdx;
    visited.insert(startOp);
  }

  worklist.push_back(startOp);

  while (!worklist.empty()) {
    Operation* op = worklist.pop_back_val();

    for (Value operand : op->getOperands()) {
      if (isa<BlockArgument>(operand))
        continue;

      Operation* defOp = operand.getDefiningOp();
      if (!defOp || defOp->getBlock() != &body)
        continue;

      int defIdx = opIndex[defOp];

      // 超过前一个region的末尾
      if (defIdx < lowerBound)
        continue;

      auto it = opToRegion.find(defOp);

      // 不能跨到其他region
      if (it != opToRegion.end() && it->second != regionIdx)
        continue;

      // 去重
      if (!visited.insert(defOp).second)
        continue;

      // 吸收defOp
      mr.opsToMove.push_back(defOp);
      opToRegion[defOp] = regionIdx;
      worklist.push_back(defOp);
    }
  }
}

void PipelineAnalyzer::computeYieldValues(MergedRegion& mr, Block& body) {
  mr.yieldValues.clear();
  mr.resultTypes.clear();

  SmallPtrSet<Operation*, 32> inRegion(mr.opsToMove.begin(), mr.opsToMove.end());

  for (Operation* op : mr.opsToMove) {
    for (Value res : op->getResults()) {
      bool usedOutside = false;

      for (OpOperand& use : res.getUses()) {
        Operation* user = use.getOwner();

        // 不在同一个for body，交给外层处理
        if (user->getBlock() != &body)
          continue;

        // 只要有一个use在region外，就必须yield
        if (!inRegion.contains(user)) {
          usedOutside = true;
          break;
        }
      }

      if (usedOutside) {
        mr.yieldValues.push_back(res);
        mr.resultTypes.push_back(res.getType());
      }
    }
  }
}

void PipelineAnalyzer::identifyCrossRegionDeps(
    SmallVector<MergedRegion>& mergedRegions,
    SmallVector<Value>& dependValues) {
  dependValues.clear();

  for (auto& curMR : mergedRegions) {
    for (Value yieldValue : curMR.yieldValues) {
      // 遍历当前区域的yieldValue的所有user OP
      for (OpOperand& use : yieldValue.getUses()) {
        Operation* userOp = use.getOwner();

        bool isUserInOtherRegion = false;
        for (auto& otherMR : mergedRegions) {
          // 跳过当前区域，只检查yieldValue是否被其他区域使用
          if (&otherMR == &curMR)
            continue;

          // 只要有一个userOp在otherMR的opsToMove列表中，就认为是dependValue
          if (llvm::is_contained(otherMR.opsToMove, userOp)) {
            isUserInOtherRegion = true;
            break;
          }
        }

        // 无重复的添加依赖变量
        if (isUserInOtherRegion) {
          if (!llvm::is_contained(dependValues, yieldValue)) {
            dependValues.push_back(yieldValue);
          }
          break;
        }
      }
    }
  }
}

DenseMap<Operation*, int> PipelineAnalyzer::computeOpIndices(Block& body) {
  DenseMap<Operation*, int> opIndex;
  int idx = 0;
  for (Operation& op : body) {
    opIndex[&op] = idx++;
  }
  return opIndex;
}

} // namespace affinity
} // namespace triton
} // namespace mlir
