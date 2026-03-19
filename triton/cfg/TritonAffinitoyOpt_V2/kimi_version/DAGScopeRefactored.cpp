/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * 重构后的 DAGScope 实现 - 使用新的 CFG/DFG 框架
 */

#include "DAGScopeRefactored.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dag-scope-refactored"

using namespace mlir;
using namespace triton;
using namespace cfg;

//===----------------------------------------------------------------------===//
// ScopePartitioner (重构 encapsulateWithScope / SplitScope)
//
// 原始代码问题：
// 1. encapsulateWithScope: 手动 worklist 传播依赖
//    - DenseSet<Operation*> opsToMoveSet
//    - SmallVector<Operation*> worklist
//    - 手动遍历所有 use
//
// 2. SplitScope: 复杂的 collectOpsToMove 逻辑
//    - 手动检查结果类型
//    - 硬编码的操作类型检查（CopyOp, FixpipeOp, StoreOp）
//    - 手动维护 parentFor 映射
//
// 重构后改进：
// 1. 使用 DFGTraverser 自动追踪数据流
// 2. 使用 AbsorptionPolicy 配置收集策略
// 3. Region 抽象自动维护指令集合
//===----------------------------------------------------------------------===//

ScopePartitioner::PartitionResult ScopePartitioner::partition(triton::FuncOp funcOp) {
  PartitionResult result;

  // 使用 DFGTraverser 从函数参数开始收集
  DFGTraverser dfgTraverser(dfg);

  // 收集依赖于函数参数的操作（这些通常是 AIV 操作）
  for (BlockArgument arg : funcOp.getArguments()) {
    // 根据参数类型决定目标 region
    AffinityDAG::CoreType targetType = AffinityDAG::CoreType::VECTOR;

    AbsorptionPolicy policy;
    policy.dir = AbsorptionPolicy::DOWNSTREAM;  // 只收集使用
    policy.crossRegionBoundary = true;

    // 收集与该参数相关的所有操作
    collectRelatedOps(arg, result.aivOps, targetType);
  }

  // 使用 CFGTraverser 遍历所有块
  CFGTraverser cfgTraverser(cfg);

  class OpClassifier : public CFGTraversalBase {
  public:
    OpClassifier(PartitionResult &result, ScopePartitioner &partitioner,
                 ControlFlowGraph &cfg)
        : result(result), partitioner(partitioner), cfg(cfg) {}

    bool preVisitInstruction(Instruction *inst, TraversalContext &ctx) override {
      Operation *op = inst->getOperation();
      if (!op) return true;

      // 使用分类器确定操作类型
      auto coreType = partitioner.classifyOperation(op);

      if (coreType == AffinityDAG::CoreType::VECTOR) {
        result.aivOps.add(inst);
      } else if (coreType == AffinityDAG::CoreType::CUBE) {
        result.aicOps.add(inst);
      } else {
        // SCALAR 操作可能在两个 scope 中都需要
        result.sharedOps.add(inst);
      }

      return true;
    }

  private:
    PartitionResult &result;
    ScopePartitioner &partitioner;
    ControlFlowGraph &cfg;
  };

  OpClassifier classifier(result, *this, cfg);
  cfgTraverser.dfsForward(classifier);

  return result;
}

void ScopePartitioner::collectRelatedOps(Value seed, Region &targetRegion,
                                         AffinityDAG::CoreType targetType) {
  DFGTraverser dfgTraverser(dfg);

  class RelatedOpCollector : public DFGTraversalBase {
  public:
    RelatedOpCollector(Region &region, ControlFlowGraph &cfg,
                       AffinityDAG::CoreType targetType,
                       ScopePartitioner &partitioner)
        : region(region), cfg(cfg), targetType(targetType),
           partitioner(partitioner) {}

    bool preVisitDef(Value value, Operation *defOp, int depth) override {
      if (Instruction *inst = cfg.getInstruction(defOp)) {
        // 只收集匹配目标类型的操作
        if (partitioner.classifyOperation(defOp) == targetType) {
          region.add(inst);
        }
      }
      return true;
    }

    bool preVisitUse(Value value, OpOperand *use, int depth) override {
      Operation *userOp = use->getOwner();
      if (Instruction *inst = cfg.getInstruction(userOp)) {
        if (partitioner.classifyOperation(userOp) == targetType) {
          region.add(inst);
        }
      }
      return true;
    }

  private:
    Region &region;
    ControlFlowGraph &cfg;
    AffinityDAG::CoreType targetType;
    ScopePartitioner &partitioner;
  };

  RelatedOpCollector collector(targetRegion, cfg, targetType, *this);
  DFGTraverser::Options opts;
  opts.followPhi = true;

  // 双向收集：先向后找到定义，再向前找到所有使用
  dfgTraverser.dfsBackward(seed, collector, opts);
  dfgTraverser.dfsForward(seed, collector, opts);
}

AffinityDAG::CoreType ScopePartitioner::classifyOperation(Operation *op) {
  // 使用 DAG 图中的类型信息
  for (Value result : op->getResults()) {
    auto it = dagGraph.valueTypes->find(result);
    if (it != dagGraph.valueTypes->end()) {
      return it->second;
    }
  }

  // 特定操作类型检查
  if (isa<hivm::CopyOp>(op)) {
    return AffinityDAG::CoreType::VECTOR;
  }
  if (isa<hivm::FixpipeOp>(op)) {
    return AffinityDAG::CoreType::CUBE;
  }
  if (isa<scf::YieldOp, scope::ScopeOp, scf::ForOp>(op)) {
    return AffinityDAG::CoreType::SCALAR;  // 共享
  }

  if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
    auto storeRes = storeOp.getOperands()[1];
    auto it = dagGraph.valueTypes->find(storeRes);
    if (it != dagGraph.valueTypes->end()) {
      return it->second;
    }
  }

  // 检查 sync 操作的 tcore_type 属性
  if (isa<hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp>(op)) {
    auto coreAttr = op->getAttr("tcore_type");
    if (coreAttr) {
      if (auto typedAttr = dyn_cast<hivm::TCoreTypeAttr>(coreAttr)) {
        return typedAttr.getTcoretype() == hivm::TCoreType::CUBE
                   ? AffinityDAG::CoreType::CUBE
                   : AffinityDAG::CoreType::VECTOR;
      }
    }
  }

  return AffinityDAG::CoreType::SCALAR;
}

//===----------------------------------------------------------------------===//
// SyncOperationInserter (重构 addSyncOpsForBufferWait)
//
// 原始代码问题：
// 1. processFixpipeOpsInAIC: 手动 walk 查找 fixpipe
//    aicRegion->walk([&](hivm::FixpipeOp op) { ... });
//
// 2. 手动查找下一个 sync_block_set
//    auto *nextSetOp = findNextSyncBlockSetAfter(fixpipeOp);
//
// 3. 手动查找 wait op
//    auto targetWait = findWaitOpInRegionWithFlag(aivRegion, setflag);
//
// 4. 复杂的插入点计算
//    auto *insertPt = findInsertionPointAfterWaitForAIV(targetWait);
//
// 重构后改进：
// 1. 使用 Region 迭代器遍历
// 2. 使用 RegionAnalyzer 识别数据依赖边界
// 3. 统一的 SyncPoint 抽象描述插入需求
//===----------------------------------------------------------------------===//

void SyncOperationInserter::insertSyncForBufferWait(triton::FuncOp funcOp,
                                                     Region *aicRegion,
                                                     Region *aivRegion) {
  // 生成 AIC 同步点（围绕 fixpipe）
  auto aicSyncPoints = generateAICSyncPoints(aicRegion, aivRegion);

  // 生成 AIV 同步点（围绕 to_memref）
  auto aivSyncPoints = generateAIVSyncPoints(aivRegion, aicRegion);

  // 合并并执行插入
  SmallVector<SyncPoint> allSyncPoints;
  allSyncPoints.append(aicSyncPoints);
  allSyncPoints.append(aivSyncPoints);

  OpBuilder builder(funcOp.getContext());
  applySyncPoints(allSyncPoints, builder);
}

SmallVector<SyncOperationInserter::SyncPoint>
SyncOperationInserter::generateAICSyncPoints(Region *aicRegion,
                                              Region *aivRegion) {
  SmallVector<SyncPoint> syncPoints;
  int64_t nextFlag = findNextAvailableFlag(
      const_cast<triton::FuncOp &>(dfg.getCFG().getFunction()));

  // 遍历 AIC region 中的所有指令
  for (Instruction *inst : *aicRegion) {
    Operation *op = inst->getOperation();
    if (!isa<hivm::FixpipeOp>(op))
      continue;

    int64_t flag = nextFlag++;

    // 1. 在 fixpipe 前插入 wait (CUBE waits for VECTOR)
    syncPoints.push_back({SyncPoint::WAIT, inst, /*insertBefore=*/true,
                          hivm::TCoreType::CUBE, hivm::PIPE::PIPE_V,
                          hivm::PIPE::PIPE_FIX, flag});

    // 2. 在 AIC region return 前插入 wait
    // 使用 RegionAnalyzer 找到 region 的出口点
    RegionAnalyzer analyzer(dfg, cfg);
    auto externalDeps = analyzer.analyzeExternalDeps(*aicRegion);
    // ... 找到 return op 并添加 sync point

    // 3. 在 AIV region 开头插入 set
    // 找到 AIV region 的第一个指令
    auto orderedAIV = aivRegion->orderedInstructions();
    if (!orderedAIV.empty()) {
      syncPoints.push_back({SyncPoint::SET, orderedAIV.front(),
                            /*insertBefore=*/true, hivm::TCoreType::VECTOR,
                            hivm::PIPE::PIPE_V, hivm::PIPE::PIPE_FIX, flag});
    }
  }

  return syncPoints;
}

SmallVector<SyncOperationInserter::SyncPoint>
SyncOperationInserter::generateAIVSyncPoints(Region *aivRegion,
                                              Region *aicRegion) {
  SmallVector<SyncPoint> syncPoints;
  int64_t nextFlag = findNextAvailableFlag(
      const_cast<triton::FuncOp &>(dfg.getCFG().getFunction()));

  for (Instruction *inst : *aivRegion) {
    Operation *op = inst->getOperation();
    if (!isa<bufferization::ToMemrefOp>(op))
      continue;

    int64_t flag = nextFlag++;

    // 1. 在 to_memref 前插入 wait (VECTOR waits for CUBE)
    syncPoints.push_back({SyncPoint::WAIT, inst, /*insertBefore=*/true,
                          hivm::TCoreType::VECTOR, hivm::PIPE::PIPE_MTE3,
                          hivm::PIPE::PIPE_MTE1, flag});

    // 2. 在 AIV region return 前插入 wait
    // 3. 在 AIC region 开头插入 set
    auto orderedAIC = aicRegion->orderedInstructions();
    if (!orderedAIC.empty()) {
      syncPoints.push_back({SyncPoint::SET, orderedAIC.front(),
                            /*insertBefore=*/true, hivm::TCoreType::CUBE,
                            hivm::PIPE::PIPE_M, hivm::PIPE::PIPE_MTE3, flag});
    }
  }

  return syncPoints;
}

void SyncOperationInserter::applySyncPoints(ArrayRef<SyncPoint> syncPoints,
                                             OpBuilder &builder) {
  for (const auto &sp : syncPoints) {
    Operation *anchorOp = sp.anchor->getOperation();
    Location loc = anchorOp->getLoc();

    if (sp.insertBefore) {
      builder.setInsertionPoint(anchorOp);
    } else {
      builder.setInsertionPointAfter(anchorOp);
    }

    if (sp.type == SyncPoint::SET) {
      createSetOp(builder, loc, sp.coreType, sp.setPipe, sp.waitPipe, sp.flag);
    } else {
      createWaitOp(builder, loc, sp.coreType, sp.setPipe, sp.waitPipe, sp.flag);
    }
  }
}

int64_t SyncOperationInserter::findNextAvailableFlag(triton::FuncOp funcOp) {
  int64_t maxFlag = -1;

  funcOp.walk([&](Operation *op) {
    IntegerAttr flagAttr;
    if (isa<hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp>(op)) {
      flagAttr = op->getAttrOfType<IntegerAttr>("static_flag_id");
    }
    if (flagAttr && flagAttr.getInt() > maxFlag) {
      maxFlag = flagAttr.getInt();
    }
  });

  return maxFlag + 1;
}

hivm::SyncBlockSetOp SyncOperationInserter::createSetOp(OpBuilder &builder,
                                                         Location loc,
                                                         hivm::TCoreType coreType,
                                                         hivm::PIPE setPipe,
                                                         hivm::PIPE waitPipe,
                                                         int64_t flag) {
  auto coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), coreType);
  auto setPipeAttr = hivm::PipeAttr::get(builder.getContext(), setPipe);
  auto waitPipeAttr = hivm::PipeAttr::get(builder.getContext(), waitPipe);
  auto flagId = builder.getIntegerAttr(builder.getI64Type(), flag);

  return builder.create<hivm::SyncBlockSetOp>(loc, coreAttr, setPipeAttr,
                                              waitPipeAttr, flagId);
}

hivm::SyncBlockWaitOp SyncOperationInserter::createWaitOp(OpBuilder &builder,
                                                           Location loc,
                                                           hivm::TCoreType coreType,
                                                           hivm::PIPE setPipe,
                                                           hivm::PIPE waitPipe,
                                                           int64_t flag) {
  auto coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), coreType);
  auto setPipeAttr = hivm::PipeAttr::get(builder.getContext(), setPipe);
  auto waitPipeAttr = hivm::PipeAttr::get(builder.getContext(), waitPipe);
  auto flagId = builder.getIntegerAttr(builder.getI64Type(), flag);

  return builder.create<hivm::SyncBlockWaitOp>(loc, coreAttr, setPipeAttr,
                                               waitPipeAttr, flagId);
}

//===----------------------------------------------------------------------===//
// ScopeOpBuilder
//===----------------------------------------------------------------------===//

scope::ScopeOp ScopeOpBuilder::buildScope(Region &ops, hivm::TCoreType coreType,
                                          Location loc) {
  // 创建 scope 操作
  auto scopeOp = builder.create<scope::ScopeOp>(loc, llvm::ArrayRef<mlir::Type>{});

  // 移动 region 中的操作到 scope body
  Block &scopeBody = scopeOp.getBodyRegion().front();
  for (Instruction *inst : ops.orderedInstructions()) {
    if (Operation *op = inst->getOperation()) {
      op->moveBefore(&scopeBody, scopeBody.end());
    }
  }

  // 添加 return op
  builder.setInsertionPointToEnd(&scopeBody);
  builder.create<scope::ReturnOp>(loc);

  // 设置 tcore_type 属性
  auto coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), coreType);
  scopeOp->setAttr(hivm::TCoreTypeAttr::name, coreAttr);

  return scopeOp;
}

//===----------------------------------------------------------------------===//
// ScopeOptimizationDriver - 主驱动
//===----------------------------------------------------------------------===//

void ScopeOptimizationDriver::run(ModuleOp module, OpBuilder &builder) {
  for (auto funcOp : module.getOps<triton::FuncOp>()) {
    if (funcOp.getBody().empty())
      continue;

    // 步骤1: 提升 alloc 操作
    hoistAllocs(funcOp);

    // 步骤2: 分区操作
    partitionScopes(funcOp);

    // 步骤3: 插入同步
    insertSyncOps(funcOp);

    // 步骤4: 构建最终 scope
    buildFinalScopes(funcOp);
  }
}

void ScopeOptimizationDriver::hoistAllocs(triton::FuncOp funcOp) {
  // 使用 CFGTraverser 收集所有 alloc
  SmallVector<Operation *> allocOps;

  CFGTraverser cfgTraverser(cfg);
  class AllocCollector : public CFGTraversalBase {
  public:
    AllocCollector(SmallVector<Operation *> &allocs) : allocs(allocs) {}

    bool preVisitInstruction(Instruction *inst, TraversalContext &ctx) override {
      if (auto allocOp = dyn_cast<memref::AllocOp>(inst->getOperation())) {
        allocs.push_back(allocOp);
      }
      return true;
    }

  private:
    SmallVector<Operation *> &allocs;
  };

  AllocCollector collector(allocOps);
  cfgTraverser.dfsForward(collector);

  // 移动到函数入口
  Block &entryBlock = funcOp.getBody().front();
  auto insertPos = entryBlock.begin();

  for (Operation *allocOp : allocOps) {
    if (allocOp->getBlock() != &entryBlock ||
        allocOp->isBeforeInBlock(&*insertPos)) {
      continue;
    }
    allocOp->moveBefore(&entryBlock, insertPos);
  }
}
