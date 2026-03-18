/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * PipelineTransformer实现
 * 执行实际的代码变换
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

// ==================== PipelineTransformer ====================

scf::ForOp PipelineTransformer::addIterArgsForDeps(scf::ForOp forOp,
                                                    ArrayRef<Value> dependValues) {
  if (dependValues.empty())
    return forOp;

  // 收集依赖值的类型
  SmallVector<Type> valueTypes;
  for (Value v : dependValues) {
    valueTypes.push_back(v.getType());
  }

  // 为每个dependValue创建初始值
  SmallVector<Value> initTensors;
  OpBuilder moduleBuilder(forOp.getContext());

  // 寻找合适的插入点并创建零张量初始化
  auto module = forOp->getParentOfType<ModuleOp>();
  if (module) {
    module.walk([&](Operation *op) {
      if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
        moduleBuilder.setInsertionPoint(constOp);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }

  // 获取原循环的信息
  Value originalLowerBound = forOp.getLowerBound();
  Value originalUpperBound = forOp.getUpperBound();
  Value originalStep = forOp.getStep();

  SmallVector<Value> newInitArgs;
  for (auto arg : forOp.getInitArgs()) {
    newInitArgs.push_back(arg);
  }

  // 查找零张量作为初始值
  for (Type valueType : valueTypes) {
    if (auto tensorType = dyn_cast<RankedTensorType>(valueType)) {
      auto zeroAttr = moduleBuilder.getZeroAttr(tensorType);
      Value zeroTensor = moduleBuilder.create<arith::ConstantOp>(
          forOp.getLoc(), tensorType, zeroAttr);
      newInitArgs.push_back(zeroTensor);
    }
  }

  // 创建新的ForOp
  OpBuilder builder(forOp);
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), originalLowerBound, originalUpperBound,
      originalStep, newInitArgs);

  // 获取新循环的region块
  Block &newBlock = newForOp.getRegion().front();
  Block &oldBlock = forOp.getRegion().front();

  // 建立块参数的映射
  IRMapping mapper;
  for (unsigned i = 0; i < oldBlock.getNumArguments(); ++i) {
    if (i < newBlock.getNumArguments()) {
      mapper.map(oldBlock.getArgument(i), newBlock.getArgument(i));
    }
  }

  // 将原循环体中的操作克隆到新块中
  builder.setInsertionPointToStart(&newBlock);
  for (auto &op : oldBlock) {
    builder.clone(op, mapper);
  }

  // 创建新的循环yield操作
  auto oldYield = cast<scf::YieldOp>(newBlock.getTerminator());
  SmallVector<Value> newYieldOps(oldYield.getOperands());

  // 找到新的dependValues
  SmallVector<Value> newDependValues;
  for (size_t i = 0; i < dependValues.size(); i++) {
    Value v = dependValues[i];
    Operation *defineOp = v.getDefiningOp();
    if (defineOp) {
      Operation *newOp = mapper.lookupOrNull(defineOp);
      if (newOp) {
        unsigned int index = cast<OpResult>(v).getResultNumber();
        newDependValues.push_back(newOp->getResult(index));
      }
    }
  }

  // 按顺序增加找到的dependvalue到yield
  for (Value v : newDependValues) {
    newYieldOps.push_back(v);
  }

  builder.setInsertionPointToEnd(&newBlock);
  builder.create<scf::YieldOp>(oldYield.getLoc(), newYieldOps);
  oldYield.erase();

  // 将原forOp的所有使用替换为新forOp
  int oldResultNum = forOp->getResults().size();
  for (auto it : llvm::zip(forOp->getResults(),
                           newForOp->getResults().take_front(oldResultNum))) {
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
  }
  forOp.erase();

  return newForOp;
}

void PipelineTransformer::createIfRegions(
    scf::ForOp forOp, SmallVector<MergedRegion> &mergedRegions,
    ArrayRef<Value> dependValues) {
  for (auto &region : mergedRegions) {
    if (region.opsToMove.empty())
      continue;

    Operation *insertPt = region.opsToMove.front();
    OpBuilder builder(insertPt);
    Location loc = insertPt->getLoc();

    // 创建恒为true的条件（后续会被替换）
    Value cond = builder.create<arith::ConstantOp>(
        loc, builder.getI1Type(), builder.getBoolAttr(true));

    bool needsYield = !region.yieldValues.empty();
    scf::IfOp ifOp;
    if (needsYield)
      ifOp = builder.create<scf::IfOp>(loc, region.resultTypes, cond, true);
    else
      ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond, false);

    // 获取else yield values
    RegionSelector selector(analysis_);
    SmallVector<Value> elseYieldValues;
    if (needsYield) {
      selector.computeElseYieldValues(region, elseYieldValues, dependValues);
    }

    // 将op移进then块
    Block &thenBlock = ifOp.getThenRegion().front();
    for (Operation *m : llvm::reverse(region.opsToMove)) {
      m->moveBefore(&thenBlock, thenBlock.begin());
    }

    // 创建then/else yield
    if (needsYield) {
      OpBuilder thenBuilder(builder.getContext());
      thenBuilder.setInsertionPointToEnd(&thenBlock);
      thenBuilder.create<scf::YieldOp>(loc, region.yieldValues);

      // else block
      Block &elseBlock = ifOp.getElseRegion().front();
      OpBuilder elseBuilder(&elseBlock, elseBlock.end());
      elseBuilder.create<scf::YieldOp>(loc, elseYieldValues);

      // 替换外部使用
      Block *block = ifOp->getBlock();

      for (size_t i = 0; i < region.yieldValues.size(); ++i) {
        Value oldVal = region.yieldValues[i];
        Value newVal = ifOp.getResult(i);

        SmallVector<OpOperand *> usesToReplace;

        for (OpOperand &use : llvm::make_early_inc_range(oldVal.getUses())) {
          Operation *user = use.getOwner();
          // 同一个block, user必须在ifOp之后, 不能在ifOp内部
          if (user->getBlock() != ifOp->getBlock() ||
              !ifOp->isBeforeInBlock(user) || user->getParentOp() == ifOp)
            continue;
          usesToReplace.push_back(&use);
        }

        for (OpOperand *use : usesToReplace)
          use->set(newVal);
      }
    }
  }
}

void PipelineTransformer::moveIterArgUsers(scf::ForOp forOp,
                                           SmallVector<MergedRegion> &mergedRegions) {
  // 这个逻辑已经在PipelineAnalyzer::moveIterArgUsersForAIC中处理
  // 这里可以添加额外的处理
}

void PipelineTransformer::applyDoubleBuffering(scf::ForOp forOp,
                                              int bufferDepth) {
  // 双缓冲实现
  // 1. 扩展循环边界 (upperBound *= bufferDepth)
  // 2. 添加计数器迭代参数
  // 3. 修改if条件

  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();

  Value originalUpperBound = forOp.getUpperBound();
  Type ubType = originalUpperBound.getType();

  // 创建bufferDepth常数
  Value bufferConst;
  if (ubType.isIndex()) {
    bufferConst = builder.create<arith::ConstantIndexOp>(loc, bufferDepth);
  } else if (auto intType = dyn_cast<IntegerType>(ubType)) {
    bufferConst = builder.create<arith::ConstantIntOp>(loc, bufferDepth, intType);
  }

  // 新上界 = originalUpperBound * bufferDepth
  builder.setInsertionPoint(forOp);
  auto newUpperBound = builder.create<arith::MulIOp>(
      loc, originalUpperBound, bufferConst);

  // 获取原循环的迭代参数
  SmallVector<Value> newIterArgs;
  for (auto arg : forOp.getInitArgs()) {
    newIterArgs.push_back(arg);
  }

  // 添加计数器初始值
  auto i32Type = builder.getI32Type();
  Value counterInit = builder.create<arith::ConstantIntOp>(loc, 0, i32Type);
  newIterArgs.push_back(counterInit);

  // 创建新循环
  auto newForOp = builder.create<scf::ForOp>(
      loc, forOp.getLowerBound(), newUpperBound, forOp.getStep(), newIterArgs);

  // 映射和克隆循环体
  Block &newBlock = newForOp.getRegion().front();
  Block &oldBlock = forOp.getRegion().front();

  IRMapping mapper;
  mapper.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (auto [oldArg, newArg] :
       llvm::zip(forOp.getRegionIterArgs(), newForOp.getRegionIterArgs())) {
    mapper.map(oldArg, newArg);
  }

  builder.setInsertionPointToStart(&newBlock);
  for (auto &op : oldBlock.without_terminator()) {
    builder.clone(op, mapper);
  }

  // 克隆yield操作，添加计数器
  if (auto yieldOp = dyn_cast<scf::YieldOp>(oldBlock.getTerminator())) {
    SmallVector<Value> newYieldOperands;
    for (auto operand : yieldOp.getOperands()) {
      newYieldOperands.push_back(mapper.lookupOrDefault(operand));
    }
    // 添加计数器（自增）
    Value counter = newForOp.getRegionIterArgs().back();
    Value one = builder.create<arith::ConstantIntOp>(loc, 1, i32Type);
    Value newCounter = builder.create<arith::AddIOp>(loc, counter, one);
    newYieldOperands.push_back(newCounter);

    builder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
  }

  // 替换原循环
  unsigned numOriginalResults = forOp.getNumResults();
  SmallVector<Value> originalResults;
  for (unsigned i = 0; i < numOriginalResults; i++) {
    originalResults.push_back(newForOp.getResult(i));
  }
  forOp.replaceAllUsesWith(originalResults);
  forOp.erase();
}

void PipelineTransformer::insertSSBufferControl(scf::ForOp forOp,
                                               CoreKind coreKind) {
  // SSBuffer控制流插入
  // 根据core类型(AIC/AIV)插入不同的同步控制代码

  OpBuilder builder(forOp);

  if (coreKind == CoreKind::CUBE) {
    // AIC模式：插入双buffer指针和状态检查
    insertAICBufferControl(forOp, builder);
  } else {
    // AIV模式：插入单buffer控制
    insertAIVBufferControl(forOp, builder);
  }
}

void PipelineTransformer::insertAICBufferControl(scf::ForOp forOp,
                                                OpBuilder &builder) {
  Location loc = forOp.getLoc();
  auto i32Type = builder.getI32Type();
  auto i64Type = builder.getI64Type();

  // 在循环开头插入buffer指针创建
  builder.setInsertionPointToStart(&forOp.getRegion().front());

  // 创建常量32和64
  Value c32 = builder.create<arith::ConstantIntOp>(loc, 32, 64);
  Value c64 = builder.create<arith::ConstantIntOp>(loc, 64, 64);

  // 创建inttoptr操作
  Value ssb_vec0_ptr = builder.create<LLVM::IntToPtrOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext(), 11), c32);
  Value ssb_vec1_ptr = builder.create<LLVM::IntToPtrOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext(), 11), c64);

  // 创建load操作获取状态
  Value status_vec0 = builder.create<LLVM::LoadOp>(loc, i32Type, ssb_vec0_ptr);
  Value status_vec1 = builder.create<LLVM::LoadOp>(loc, i32Type, ssb_vec1_ptr);

  // 存储buffer指针供后续使用
  // 注意：实际使用时需要确保这些值可以在循环体内访问
}

void PipelineTransformer::insertAIVBufferControl(scf::ForOp forOp,
                                                OpBuilder &builder) {
  Location loc = forOp.getLoc();
  auto i32Type = builder.getI32Type();
  auto i64Type = builder.getI64Type();

  // 在scope开头插入buffer地址计算
  auto scopeOp = forOp.getParentOfType<scope::ScopeOp>();
  if (!scopeOp)
    return;

  builder.setInsertionPointToStart(&scopeOp.getRegion().front());

  // 获取sub_block_idx
  Value subIdOp = builder.create<hivm::GetSubBlockIdxOp>(loc, i64Type);

  // 计算buffer地址
  Value c32 = builder.create<arith::ConstantIntOp>(loc, 32, 64);
  Value ssbAddrOffset = builder.create<arith::MulIOp>(loc, subIdOp, c32);
  Value ssbAddr = builder.create<arith::AddIOp>(loc, ssbAddrOffset, c32);

  // 在循环开头插入inttoptr和load
  builder.setInsertionPointToStart(&forOp.getRegion().front());
  Value ssbCubePtr = builder.create<LLVM::IntToPtrOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext(), 11), ssbAddr);
  Value statusCube = builder.create<LLVM::LoadOp>(loc, i32Type, ssbCubePtr);
}

scf::IfOp PipelineTransformer::createIfOpForRegion(
    const MergedRegion &region, Operation *insertPoint,
    ArrayRef<Value> elseValues) {
  OpBuilder builder(insertPoint);
  Location loc = insertPoint->getLoc();

  // 创建条件
  Value cond = builder.create<arith::ConstantOp>(
      loc, builder.getI1Type(), builder.getBoolAttr(true));

  bool needsYield = !region.yieldValues.empty();
  scf::IfOp ifOp;
  if (needsYield)
    ifOp = builder.create<scf::IfOp>(loc, region.resultTypes, cond, true);
  else
    ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond, false);

  return ifOp;
}

void PipelineTransformer::updateRegionOpsWithMapping(
    SmallVector<MergedRegion> &mergedRegions, IRMapping &mapper) {
  for (auto &mr : mergedRegions) {
    // 更新opsToMove列表
    SmallVector<Operation *> newOpsToMove;
    for (Operation *op : mr.opsToMove) {
      if (op) {
        Operation *newOp = mapper.lookupOrNull(op);
        if (newOp)
          newOpsToMove.push_back(newOp);
      }
    }
    mr.opsToMove = newOpsToMove;

    // 更新yieldValues列表
    SmallVector<Value> newYieldValues;
    for (Value v : mr.yieldValues) {
      if (v) {
        Value newV = mapper.lookupOrNull(v);
        if (newV)
          newYieldValues.push_back(newV);
      }
    }
    mr.yieldValues = newYieldValues;
  }
}

Value PipelineTransformer::createBufferCondition(OpBuilder &builder, Location loc,
                                                int bufferIdx,
                                                CoreKind coreKind) {
  // 创建buffer条件检查
  // 根据buffer索引和core类型创建适当的条件

  auto i32Type = builder.getI32Type();

  // 创建buffer mask
  auto maskAttr = builder.getI32IntegerAttr(1 << bufferIdx);
  Value mask = builder.create<LLVM::ConstantOp>(loc, i32Type, maskAttr);

  // 这里简化处理，实际需要根据具体状态值创建条件
  Value c0 = builder.create<arith::ConstantIntOp>(loc, 0, i32Type);
  Value cond = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, c0, mask);

  return cond;
}

} // namespace affinity
} // namespace triton
} // namespace mlir
