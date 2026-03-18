/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#include "CoreSyncAnalyzer.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "core-sync-analyzer"

using namespace mlir;
using namespace triton;
using namespace affinity;

//===----------------------------------------------------------------------===//
// CoreTypeClassifier
//===----------------------------------------------------------------------===//

CoreType CoreTypeClassifier::getType(Value value) const {
  auto it = valueTypes->find(value);
  return (it != valueTypes->end()) ? it->second : CoreType::SCALAR;
}

CoreType CoreTypeClassifier::getType(Operation* op) const {
  if (!op || op->getNumResults() == 0)
    return CoreType::SCALAR;

  // 检查所有结果的核心类型
  CoreType result = CoreType::UNDETERMINED;
  for (Value res : op->getResults()) {
    CoreType ct = getType(res);
    if (ct != CoreType::UNDETERMINED) {
      if (result == CoreType::UNDETERMINED)
        result = ct;
      else
        result = static_cast<CoreType>(result | ct);
    }
  }

  // 特殊操作类型检查
  if (isa<hivm::CopyOp>(op))
    return CoreType::VECTOR;
  if (isa<hivm::FixpipeOp>(op))
    return CoreType::CUBE;

  return (result != CoreType::UNDETERMINED) ? result : CoreType::SCALAR;
}

//===----------------------------------------------------------------------===//
// CoreSyncAnalyzer
//===----------------------------------------------------------------------===//

void CoreSyncAnalyzer::analyze() {
  LLVM_DEBUG(llvm::dbgs() << "=== Starting Core Sync Analysis ===\n");

  // 遍历CFG的所有基本块
  cfg.traverse([&](BasicBlock& bb) {
    analyzeBlock(bb);
  });

  LLVM_DEBUG(llvm::dbgs() << "Found " << syncPoints.size()
                          << " sync points\n");
}

void CoreSyncAnalyzer::analyzeBlock(BasicBlock& bb) {
  // 遍历基本块内的所有指令
  for (auto& instPtr : bb.getInstructions()) {
    Instruction* inst = instPtr.get();
    Operation* op = inst->getOperation();

    if (!op || classifier.isSyncOp(op))
      continue;

    // 检查是否是控制流操作
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      analyzeIterArgs(forOp);
      continue;
    }

    // 获取当前操作的核心类型
    CoreType currentType = classifier.getType(op);
    if (currentType == CoreType::UNDETERMINED ||
        currentType == CoreType::SCALAR)
      continue;

    // 遍历Memory SSA uses来查找需要同步的前驱
    auto& memSSAInfo = inst->getMemorySSAInfo();
    for (const auto& use : memSSAInfo.uses) {
      MemorySSADef* def = use.getDefinition();
      if (!def || !def->getDefOp())
        continue;

      Operation* srcOp = def->getDefOp();
      CoreType srcType = classifier.getType(srcOp);

      if (!classifier.needsSync(srcType, currentType))
        continue;

      // 检查是否已经处理过
      bool alreadyExists = llvm::any_of(syncPoints, [&](const SyncPoint& sp) {
        return sp.srcOp == srcOp && sp.dstOp == op;
      });

      if (alreadyExists)
        continue;

      // 记录同步点
      SyncPoint point;
      point.srcOp = srcOp;
      point.dstOp = op;
      point.srcType = srcType;
      point.dstType = currentType;
      point.crossBlock = isCrossBlockDependency(srcOp, op);
      point.flag = getNextFlag();

      syncPoints.push_back(point);

      // 分析数据搬运需求
      analyzeDataMovement(srcOp, op, srcType, currentType);
    }
  }
}

void CoreSyncAnalyzer::analyzeIterArgs(scf::ForOp forOp) {
  Block* loopBody = forOp.getBody();
  scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(loopBody->getTerminator());
  if (!yieldOp)
    return;

  // 遍历所有迭代参数
  for (unsigned i = 0; i < forOp.getInitArgs().size(); ++i) {
    BlockArgument iterArg = loopBody->getArgument(i + 1);
    Value yieldValue = yieldOp.getOperand(i);

    CoreType iterType = classifier.getType(iterArg);
    CoreType yieldType = classifier.getType(yieldValue);

    if (!classifier.needsSync(yieldType, iterType))
      continue;

    // 查找yield值的定义操作
    Operation* yieldDef = yieldValue.getDefiningOp();
    Operation* firstUser = nullptr;

    // 在循环体内查找第一个使用iterArg的操作
    for (auto& op : *loopBody) {
      if (isa<scf::YieldOp>(&op))
        continue;

      if (llvm::any_of(op.getOperands(), [&](Value v) { return v == iterArg; })) {
        firstUser = &op;
        break;
      }
    }

    if (!yieldDef || !firstUser)
      continue;

    SyncPoint point;
    point.srcOp = yieldDef;
    point.dstOp = firstUser;
    point.srcType = yieldType;
    point.dstType = iterType;
    point.crossBlock = false;
    point.flag = getNextFlag();

    syncPoints.push_back(point);
  }
}

bool CoreSyncAnalyzer::isCrossBlockDependency(Operation* src, Operation* dst) const {
  Block* srcBlock = src->getBlock();
  Block* dstBlock = dst->getBlock();

  if (srcBlock == dstBlock)
    return false;

  // 检查dst是否在src的内层block中
  Operation* parentOp = dstBlock->getParentOp();
  while (parentOp) {
    if (parentOp->getBlock() == srcBlock)
      return true;
    parentOp = parentOp->getBlock() ? parentOp->getBlock()->getParentOp() : nullptr;
  }

  return false;
}

void CoreSyncAnalyzer::analyzeDataMovement(Operation* src, Operation* dst,
                                           CoreType srcType, CoreType dstType) {
  if (src->getNumResults() == 0)
    return;

  DataMovement dm;
  dm.source = src->getResult(0);
  dm.insertAfter = src;
  dm.insertBefore = dst;

  if (srcType == CoreType::CUBE && dstType == CoreType::VECTOR) {
    dm.type = DataMovement::CUBE_TO_VECTOR;
    dm.target = dst->getOperand(0);
    dataMovements.push_back(dm);
  } else if (srcType == CoreType::VECTOR && dstType == CoreType::CUBE) {
    dm.type = DataMovement::VECTOR_TO_CUBE;
    dm.target = dst->getOperand(0);
    dataMovements.push_back(dm);
  }
}

//===----------------------------------------------------------------------===//
// ScopeBuilder
//===----------------------------------------------------------------------===//

std::pair<scope::ScopeOp, scope::ScopeOp>
ScopeBuilder::buildScopes(triton::FuncOp func) {
  OpBuilder builder(func);
  Block& entryBlock = func.getBody().front();

  // 收集所有alloc操作并提前到函数开头
  SmallVector<Operation*> allocOps;
  func.walk([&](memref::AllocOp op) { allocOps.push_back(op); });

  for (auto* op : allocOps) {
    if (op->getBlock() != &entryBlock)
      op->moveBefore(&entryBlock, entryBlock.begin());
  }

  // 收集需要移动到各scope的操作
  SmallVector<Operation*> aivOps;
  SmallVector<Operation*> cubeOps;

  cfg.traverse([&](BasicBlock& bb) {
    collectOpsToMove(bb, aivOps, cubeOps);
  });

  if (aivOps.empty())
    return {nullptr, nullptr};

  // 创建AIV Scope (VECTOR)
  builder.setInsertionPoint(&entryBlock, entryBlock.end());
  auto aivScope = createScope(builder, func.getLoc(), hivm::TCoreType::VECTOR);

  // 移动操作到AIV Scope
  IRMapping aivMapper;
  moveOpsToScope(aivOps, aivScope, aivMapper);

  // 创建AIC Scope (CUBE)
  builder.setInsertionPointAfter(aivScope);
  auto aicScope = createScope(builder, func.getLoc(), hivm::TCoreType::CUBE);

  // 创建terminator
  OpBuilder aicBuilder(aicScope.getBodyRegion().front().getTerminator());
  aicBuilder.create<scope::ReturnOp>(aicScope.getLoc());

  return {aivScope, aicScope};
}

void ScopeBuilder::collectOpsToMove(BasicBlock& bb,
                                    SmallVector<Operation*>& aivOps,
                                    SmallVector<Operation*>& cubeOps) {
  for (auto& instPtr : bb.getInstructions()) {
    Operation* op = instPtr.get()->getOperation();
    if (!op || isa<arith::ConstantOp>(op) || isa<triton::GetProgramIdOp>(op))
      continue;

    CoreType ct = classifier.getType(op);

    // 控制流操作需要特殊处理
    if (classifier.isControlFlow(op)) {
      // 根据循环体内的操作决定放在哪个scope
      bool hasVector = false, hasCube = false;
      for (auto& region : op->getRegions()) {
        region.walk([&](Operation* innerOp) {
          CoreType innerCt = classifier.getType(innerOp);
          if (innerCt == CoreType::VECTOR) hasVector = true;
          if (innerCt == CoreType::CUBE) hasCube = true;
        });
      }

      if (hasVector) aivOps.push_back(op);
      if (hasCube) cubeOps.push_back(op);
      continue;
    }

    // 普通操作根据类型分类
    if (ct == CoreType::VECTOR || ct == CoreType::SCALAR)
      aivOps.push_back(op);
    if (ct == CoreType::CUBE || ct == CoreType::SCALAR)
      cubeOps.push_back(op);
  }
}

scope::ScopeOp ScopeBuilder::createScope(OpBuilder& builder, Location loc,
                                         hivm::TCoreType coreType) {
  auto scopeOp = builder.create<scope::ScopeOp>(loc, TypeRange{});
  scopeOp.getBodyRegion().emplaceBlock();

  auto coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), coreType);
  scopeOp->setAttr(hivm::TCoreTypeAttr::name, coreAttr);

  // 添加return
  OpBuilder scopeBuilder(scopeOp.getBodyRegion().front());
  scopeBuilder.create<scope::ReturnOp>(loc);

  return scopeOp;
}

void ScopeBuilder::moveOpsToScope(SmallVector<Operation*>& ops,
                                  scope::ScopeOp scope,
                                  IRMapping& mapper) {
  Block* scopeBody = &scope.getBodyRegion().front();
  OpBuilder builder(scopeBody, scopeBody->end());

  // 按原始顺序排序
  llvm::sort(ops, [](Operation* a, Operation* b) {
    return a->isBeforeInBlock(b);
  });

  for (Operation* op : ops) {
    if (!op || op->use_empty())
      continue;

    SmallVector<Value> originalResults = op->getResults();
    op->remove();
    builder.insert(op);

    // 更新映射
    for (size_t i = 0; i < originalResults.size(); ++i) {
      mapper.map(originalResults[i], op->getResult(i));
    }
  }
}

//===----------------------------------------------------------------------===//
// SyncInserter
//===----------------------------------------------------------------------===//

void SyncInserter::insertSyncOps(scope::ScopeOp aivScope,
                                 scope::ScopeOp cubeScope) {
  if (!aivScope || !cubeScope)
    return;

  OpBuilder builder(aivScope);

  for (const auto& point : analyzer.getSyncPoints()) {
    insertSync(point, builder);
  }

  for (const auto& dm : analyzer.getDataMovements()) {
    insertDataMovement(dm, builder);
  }
}

void SyncInserter::insertSync(const SyncPoint& point, OpBuilder& builder) {
  Location loc = point.srcOp->getLoc();

  // 确定pipe类型
  hivm::PIPE setPipe, waitPipe;
  hivm::TCoreType setCore, waitCore;

  if (point.srcType == CoreType::CUBE && point.dstType == CoreType::VECTOR) {
    setPipe = hivm::PIPE::PIPE_FIX;
    waitPipe = hivm::PIPE::PIPE_V;
    setCore = hivm::TCoreType::CUBE;
    waitCore = hivm::TCoreType::VECTOR;
  } else {
    setPipe = hivm::PIPE::PIPE_MTE3;
    waitPipe = hivm::PIPE::PIPE_MTE1;
    setCore = hivm::TCoreType::VECTOR;
    waitCore = hivm::TCoreType::CUBE;
  }

  auto flagId = builder.getIntegerAttr(builder.getI64Type(), point.flag);

  // 插入set
  builder.setInsertionPointAfter(point.srcOp);
  createSyncBlockSet(builder, loc, setCore, setPipe, waitPipe, point.flag);

  // 插入wait
  builder.setInsertionPoint(point.dstOp);
  if (point.crossBlock) {
    // 跨block时，wait需要插在内层block入口
    Block* dstBlock = point.dstOp->getBlock();
    Operation* parentOp = dstBlock->getParentOp();
    if (parentOp)
      builder.setInsertionPoint(parentOp);
  }
  createSyncBlockWait(builder, loc, waitCore, setPipe, waitPipe, point.flag);
}

void SyncInserter::insertDataMovement(const DataMovement& dm, OpBuilder& builder) {
  if (dm.type == DataMovement::CUBE_TO_VECTOR) {
    // CUBE -> VECTOR: 使用fixpipe
    builder.setInsertionPointAfter(dm.insertAfter);

    // 创建UB alloc
    auto tensorType = dyn_cast<RankedTensorType>(dm.source.getType());
    if (!tensorType)
      return;

    auto ubSpace = hivm::AddressSpaceAttr::get(builder.getContext(),
                                               hivm::AddressSpace::UB);
    auto memrefType = MemRefType::get(tensorType.getShape(),
                                      tensorType.getElementType(),
                                      nullptr, ubSpace);
    auto alloc = builder.create<memref::AllocOp>(dm.insertAfter->getLoc(),
                                                 memrefType);

    // 创建fixpipe
    auto dmaMode = hivm::FixpipeDMAModeAttr::get(builder.getContext(),
                                                  hivm::FixpipeDMAMode::NZ2ND);
    builder.create<hivm::FixpipeOp>(dm.insertAfter->getLoc(), TypeRange{},
                                    dm.source, alloc, nullptr, dmaMode,
                                    nullptr, nullptr, nullptr, nullptr, nullptr);

    // 在dst前插入to_tensor
    builder.setInsertionPoint(dm.insertBefore);
    auto toTensor = builder.create<bufferization::ToTensorOp>(
        dm.insertBefore->getLoc(), tensorType, alloc, true, true);

    // 替换操作数
    dm.insertBefore->replaceUsesOfWith(dm.source, toTensor.getResult());
  }
}

void SyncInserter::createSyncBlockSet(OpBuilder& builder, Location loc,
                                      hivm::TCoreType coreType,
                                      hivm::PIPE setPipe,
                                      hivm::PIPE waitPipe,
                                      int64_t flag) {
  auto coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), coreType);
  auto setPipeAttr = hivm::PipeAttr::get(builder.getContext(), setPipe);
  auto waitPipeAttr = hivm::PipeAttr::get(builder.getContext(), waitPipe);
  auto flagAttr = builder.getIntegerAttr(builder.getI64Type(), flag);

  builder.create<hivm::SyncBlockSetOp>(loc, coreAttr, setPipeAttr, waitPipeAttr,
                                       flagAttr);
}

void SyncInserter::createSyncBlockWait(OpBuilder& builder, Location loc,
                                       hivm::TCoreType coreType,
                                       hivm::PIPE setPipe,
                                       hivm::PIPE waitPipe,
                                       int64_t flag) {
  auto coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), coreType);
  auto setPipeAttr = hivm::PipeAttr::get(builder.getContext(), setPipe);
  auto waitPipeAttr = hivm::PipeAttr::get(builder.getContext(), waitPipe);
  auto flagAttr = builder.getIntegerAttr(builder.getI64Type(), flag);

  builder.create<hivm::SyncBlockWaitOp>(loc, coreAttr, setPipeAttr, waitPipeAttr,
                                        flagAttr);
}

void SyncInserter::addSyncForBufferWait(triton::FuncOp func) {
  // 查找AIV和AIC region
  Region* aivRegion = nullptr;
  Region* aicRegion = nullptr;

  func.walk([&](scope::ScopeOp scopeOp) {
    auto coreAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(
        hivm::TCoreTypeAttr::name);
    if (!coreAttr)
      return;

    if (coreAttr.getTcoretype() == hivm::TCoreType::CUBE)
      aicRegion = &scopeOp.getRegion();
    else if (coreAttr.getTcoretype() == hivm::TCoreType::VECTOR)
      aivRegion = &scopeOp.getRegion();
  });

  if (!aicRegion || !aivRegion)
    return;

  int64_t nextFlag = 1;
  OpBuilder builder(func.getContext());

  // 处理AIC中的FixpipeOp
  aicRegion->walk([&](hivm::FixpipeOp fixpipeOp) {
    int64_t flag = nextFlag++;

    // 在fixpipe前插入wait
    builder.setInsertionPoint(fixpipeOp);
    createSyncBlockWait(builder, fixpipeOp.getLoc(), hivm::TCoreType::CUBE,
                        hivm::PIPE::PIPE_V, hivm::PIPE::PIPE_FIX, flag);

    // 在AIC末尾插入wait
    insertWaitAtRegionEnd(*aicRegion, flag, hivm::TCoreType::CUBE, builder);

    // 在AIV开头插入set
    insertSetAtRegionStart(*aivRegion, flag, hivm::TCoreType::VECTOR, builder);
  });

  // 处理AIV中的ToMemrefOp
  aivRegion->walk([&](bufferization::ToMemrefOp toMemrefOp) {
    int64_t flag = nextFlag++;

    // 在toMemref前插入wait
    builder.setInsertionPoint(toMemrefOp);
    createSyncBlockWait(builder, toMemrefOp.getLoc(), hivm::TCoreType::VECTOR,
                        hivm::PIPE::PIPE_M, hivm::PIPE::PIPE_MTE3, flag);

    // 在AIV末尾插入wait
    insertWaitAtRegionEnd(*aivRegion, flag, hivm::TCoreType::VECTOR, builder);

    // 在AIC开头插入set
    insertSetAtRegionStart(*aicRegion, flag, hivm::TCoreType::CUBE, builder);
  });
}

void SyncInserter::insertWaitAtRegionEnd(Region& region, int flag,
                                         hivm::TCoreType coreType,
                                         OpBuilder& builder) {
  for (auto& block : region) {
    if (auto returnOp = dyn_cast<scope::ReturnOp>(block.getTerminator())) {
      builder.setInsertionPoint(returnOp);
      hivm::PIPE setPipe = (coreType == hivm::TCoreType::CUBE) ?
                           hivm::PIPE::PIPE_V : hivm::PIPE::PIPE_M;
      hivm::PIPE waitPipe = (coreType == hivm::TCoreType::CUBE) ?
                            hivm::PIPE::PIPE_FIX : hivm::PIPE::PIPE_MTE3;
      createSyncBlockWait(builder, returnOp->getLoc(), coreType, setPipe,
                          waitPipe, flag);
    }
  }
}

void SyncInserter::insertSetAtRegionStart(Region& region, int flag,
                                          hivm::TCoreType coreType,
                                          OpBuilder& builder) {
  if (region.empty())
    return;

  Block& entry = region.front();
  Location loc = entry.empty() ? region.getParentOp()->getLoc()
                               : entry.front().getLoc();
  builder.setInsertionPointToStart(&entry);

  hivm::PIPE setPipe = (coreType == hivm::TCoreType::CUBE) ?
                       hivm::PIPE::PIPE_M : hivm::PIPE::PIPE_V;
  hivm::PIPE waitPipe = (coreType == hivm::TCoreType::CUBE) ?
                        hivm::PIPE::PIPE_MTE3 : hivm::PIPE::PIPE_FIX;

  createSyncBlockSet(builder, loc, coreType, setPipe, waitPipe, flag);
}
