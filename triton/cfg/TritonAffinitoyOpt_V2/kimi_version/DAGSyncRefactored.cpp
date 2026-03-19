/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * 重构后的 DAGSync 实现 - 使用新的 CFG/DFG 框架
 */

#include "DAGSyncRefactored.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dag-sync-refactored"

using namespace mlir;
using namespace triton;
using namespace cfg;

//===----------------------------------------------------------------------===//
// SyncAnalyzer (重构手动 DAG 遍历)
//
// 原始代码问题：
// 1. 手动遍历 opMap 查找节点
// 2. 手动遍历 inputNode->ins 检查依赖
// 3. 手动判断跨 block 关系（向上查找 parentOp）
//
// 重构后改进：
// 1. 使用 DFGTraverser 自动追踪数据流依赖
// 2. 使用 CFGTraverser 判断 block 层级关系
// 3. 统一返回结构化的依赖信息
//===----------------------------------------------------------------------===//

SmallVector<CrossCoreDependency>
SyncAnalyzer::analyzeSyncRequirements(triton::FuncOp funcOp) {
  SmallVector<CrossCoreDependency> deps;

  // 使用 CFGTraverser 遍历所有指令
  CFGTraverser cfgTraverser(cfg);

  class DependencyCollector : public CFGTraversalBase {
  public:
    DependencyCollector(SmallVector<CrossCoreDependency> &deps,
                        SyncAnalyzer &analyzer, DataFlowGraph &dfg)
        : deps(deps), analyzer(analyzer), dfg(dfg) {}

    bool preVisitInstruction(Instruction *inst, TraversalContext &ctx) override {
      // 获取当前指令的核心类型
      auto dstType = analyzer.getCoreType(inst);

      // 使用 DFGTraverser 查找所有定义该指令使用的值的源指令
      DFGTraverser dfgTraverser(dfg);

      class DefUseAnalyzer : public DFGTraversalBase {
      public:
        DefUseAnalyzer(Instruction *dstInst, AffinityDAG::CoreType dstType,
                       SyncAnalyzer &analyzer,
                       SmallVector<CrossCoreDependency> &deps)
            : dstInst(dstInst), dstType(dstType), analyzer(analyzer),
              deps(deps) {}

        bool preVisitDef(Value value, Operation *defOp, int depth) override {
          if (!defOp)
            return true;

          auto cfg = analyzer.cfg;
          Instruction *srcInst = cfg.getInstruction(defOp);
          if (!srcInst)
            return true;

          auto srcType = analyzer.getCoreType(srcInst);

          // 检查是否需要 VECTOR <-> CUBE 同步
          if (analyzer.needVectorCubeSync(srcType, dstType)) {
            bool isCrossBlock = false;
            bool dstIsInner = false;

            // 检查是否跨 block
            if (srcInst->getParentBlock() != dstInst->getParentBlock()) {
              isCrossBlock = true;
              dstIsInner = analyzer.isCrossBlockDependency(srcInst, dstInst, dstIsInner);
            }

            deps.push_back({srcInst, dstInst, srcType, dstType,
                           value, isCrossBlock, dstIsInner});
          }

          return true;
        }

      private:
        Instruction *dstInst;
        AffinityDAG::CoreType dstType;
        SyncAnalyzer &analyzer;
        SmallVector<CrossCoreDependency> &deps;
      };

      DefUseAnalyzer defAnalyzer(inst, dstType, analyzer, deps);

      // 遍历该指令的所有操作数
      if (auto op = inst->getOperation()) {
        for (Value operand : op->getOperands()) {
          DFGTraverser::Options opts;
          opts.followPhi = true;
          dfgTraverser.dfsBackward(operand, defAnalyzer, opts);
        }
      }

      return true;
    }

  private:
    SmallVector<CrossCoreDependency> &deps;
    SyncAnalyzer &analyzer;
    DataFlowGraph &dfg;
  };

  DependencyCollector collector(deps, *this, dfg);
  cfgTraverser.dfsForward(collector);

  return deps;
}

SmallVector<CrossCoreDependency>
SyncAnalyzer::analyzeForLoopSync(scf::ForOp forOp) {
  SmallVector<CrossCoreDependency> deps;

  Block *loopBody = forOp.getBody();
  scf::YieldOp yieldOp = nullptr;

  // 查找 yield 操作
  for (Operation &op : *loopBody) {
    if (auto yield = dyn_cast<scf::YieldOp>(&op)) {
      yieldOp = yield;
      break;
    }
  }

  if (!yieldOp)
    return deps;

  // 遍历所有迭代参数
  for (int i = 0; i < forOp.getInitArgs().size(); i++) {
    BlockArgument iterArg = loopBody->getArgument(i + 1);

    // 使用 DFGTraverser 查找首次使用
    DFGTraverser dfgTraverser(dfg);

    class FirstUseFinder : public DFGTraversalBase {
    public:
      FirstUseFinder(BlockArgument iterArg, ControlFlowGraph &cfg,
                     Instruction *&firstUser)
          : iterArg(iterArg), cfg(cfg), firstUser(firstUser),
            found(false) {}

      bool preVisitUse(Value value, OpOperand *use, int depth) override {
        if (found)
          return false;

        Operation *userOp = use->getOwner();
        if (isa<scf::YieldOp>(userOp))
          return true;

        if (Instruction *inst = cfg.getInstruction(userOp)) {
          firstUser = inst;
          found = true;
          return false;
        }

        return true;
      }

    private:
      BlockArgument iterArg;
      ControlFlowGraph &cfg;
      Instruction *&firstUser;
      bool found;
    };

    Instruction *firstUser = nullptr;
    FirstUseFinder finder(iterArg, cfg, firstUser);

    DFGTraverser::Options opts;
    opts.followPhi = true;
    dfgTraverser.dfsForward(iterArg, finder, opts);

    if (!firstUser)
      continue;

    // 获取类型
    AffinityDAG::CoreType iterType = getCoreType(firstUser);
    Value yieldOperand = yieldOp->getOperand(i);
    AffinityDAG::CoreType yieldType = AffinityDAG::CoreType::SCALAR;

    if (auto defOp = yieldOperand.getDefiningOp()) {
      if (Instruction *yieldDefInst = cfg.getInstruction(defOp)) {
        yieldType = getCoreType(yieldDefInst);
      }
    }

    // 检查是否需要同步
    if (needVectorCubeSync(yieldType, iterType)) {
      deps.push_back({cfg.getInstruction(yieldOperand.getDefiningOp()),
                     firstUser, yieldType, iterType, iterArg,
                     false, false});
    }
  }

  return deps;
}

bool SyncAnalyzer::needVectorCubeSync(AffinityDAG::CoreType src,
                                      AffinityDAG::CoreType dst) {
  return (src == AffinityDAG::CoreType::VECTOR &&
          dst == AffinityDAG::CoreType::CUBE) ||
         (src == AffinityDAG::CoreType::CUBE &&
          dst == AffinityDAG::CoreType::VECTOR);
}

AffinityDAG::CoreType SyncAnalyzer::getCoreType(Instruction *inst) {
  if (!inst || !inst->getOperation())
    return AffinityDAG::CoreType::SCALAR;

  Operation *op = inst->getOperation();

  // 尝试从 DAG 的类型映射中获取
  if (op->getNumResults() > 0) {
    mlir::Value result = op->getResult(0);
    auto it = dagGraph.valueTypes->find(result);
    if (it != dagGraph.valueTypes->end()) {
      return it->second;
    }
  }

  return AffinityDAG::CoreType::SCALAR;
}

bool SyncAnalyzer::isCrossBlockDependency(Instruction *src, Instruction *dst,
                                          bool &dstIsInner) {
  BasicBlock *srcBlock = src->getParentBlock();
  BasicBlock *dstBlock = dst->getParentBlock();

  if (srcBlock == dstBlock)
    return false;

  // 检查 dstBlock 是否在 srcBlock 内部
  Block *dstParentBlock = dstBlock->getMLIRBlock();
  Operation *dstParentOp = dstParentBlock->getParentOp();

  while (dstParentOp) {
    if (dstParentOp->getBlock() == srcBlock->getMLIRBlock()) {
      dstIsInner = true;
      return true;
    }
    if (dstParentOp->getBlock()) {
      dstParentOp = dstParentOp->getBlock()->getParentOp();
    } else {
      break;
    }
  }

  dstIsInner = false;
  return true;
}

//===----------------------------------------------------------------------===//
// DataMovementBuilder (重构 insertCubeToVector/VectorToCubeDataMovement)
//
// 原始代码问题：
// 1. 硬编码的 fixpipe/copy 参数
// 2. 重复的类型检查和内存空间管理
// 3. 手动 operand 替换
//
// 重构后改进：
// 1. 统一的数据搬运接口
// 2. 自动类型推导和缓存
// 3. 使用 DFG 更新 valueTypes
//===----------------------------------------------------------------------===//

Value DataMovementBuilder::buildCubeToVectorMovement(Value srcValue,
                                                     Instruction *dstInst,
                                                     const MovementConfig &config) {
  auto srcTensorType = dyn_cast<RankedTensorType>(srcValue.getType());
  if (!srcTensorType)
    return nullptr;

  Location loc = dstInst->getOperation()->getLoc();

  // 1. 获取 UB 空间的 allocation
  Value ubAlloc = getOrCreateAllocation(srcTensorType, hivm::AddressSpace::UB, loc);

  // 2. 在 srcOp 之后创建 fixpipe
  builder.setInsertionPointAfter(srcValue.getDefiningOp());
  createFixpipeOp(srcValue, ubAlloc, loc, config);

  // 3. 在 dstOp 前创建 to_tensor
  builder.setInsertionPoint(dstInst->getOperation());

  // memory_space_cast（如果需要）
  Value plainMemref = ubAlloc;
  auto memrefType = cast<MemRefType>(ubAlloc.getType());
  if (memrefType.getMemorySpace()) {
    auto plainMemrefType = MemRefType::get(memrefType.getShape(),
                                           memrefType.getElementType());
    plainMemref = builder.create<memref::MemorySpaceCastOp>(loc, plainMemrefType, ubAlloc);
    (*dagGraph.valueTypes)[plainMemref] = AffinityDAG::CoreType::VECTOR;
  }

  // 创建 to_tensor
  auto toTensorOp = builder.create<bufferization::ToTensorOp>(
      loc, srcTensorType, plainMemref, /*restrict=*/true, /*writable=*/true);

  (*dagGraph.valueTypes)[toTensorOp.getResult()] = AffinityDAG::CoreType::VECTOR;

  return toTensorOp.getResult();
}

Value DataMovementBuilder::buildVectorToCubeMovement(Value srcValue,
                                                     Instruction *dstInst,
                                                     const MovementConfig &config) {
  auto srcTensorType = dyn_cast<RankedTensorType>(srcValue.getType());
  if (!srcTensorType)
    return nullptr;

  Location loc = dstInst->getOperation()->getLoc();

  // 1. 创建 UB 空间的 to_memref
  builder.setInsertionPointAfter(srcValue.getDefiningOp());

  auto ubSpaceAttr = hivm::AddressSpaceAttr::get(builder.getContext(), hivm::AddressSpace::UB);
  auto ubMemrefType = MemRefType::get(srcTensorType.getShape(),
                                      srcTensorType.getElementType(),
                                      /*layout=*/nullptr, ubSpaceAttr);

  auto toMemrefOp = builder.create<bufferization::ToMemrefOp>(loc, ubMemrefType, srcValue);

  // 2. 创建 CBUF 空间的 allocation
  Value cbufAlloc = getOrCreateAllocation(srcTensorType, hivm::AddressSpace::L1, loc);

  // 3. 创建 copy 操作
  builder.setInsertionPointAfter(toMemrefOp);
  auto copyOp = builder.create<hivm::CopyOp>(loc, TypeRange{}, toMemrefOp.getResult(), cbufAlloc);

  // 4. 在 dstOp 前创建 convert_layout
  builder.setInsertionPoint(dstInst->getOperation());

  auto ndLayout = hivm::DataLayoutAttr::get(builder.getContext(), hivm::DataLayout::ND);
  auto convertLayoutOp = builder.create<hivm::ConvertLayoutOp>(
      loc, cbufAlloc.getType(), cbufAlloc, ndLayout, ndLayout);

  (*dagGraph.valueTypes)[convertLayoutOp.getResult()] = AffinityDAG::CoreType::CUBE;

  // 5. 创建 memory_space_cast
  auto cbufMemrefType = cast<MemRefType>(convertLayoutOp.getType());
  auto plainMemrefType = MemRefType::get(cbufMemrefType.getShape(),
                                         cbufMemrefType.getElementType());

  auto memspaceCastOp = builder.create<memref::MemorySpaceCastOp>(
      loc, plainMemrefType, convertLayoutOp.getResult());

  (*dagGraph.valueTypes)[memspaceCastOp.getResult()] = AffinityDAG::CoreType::CUBE;

  // 6. 创建 to_tensor
  auto toTensorOp = builder.create<bufferization::ToTensorOp>(
      loc, srcTensorType, memspaceCastOp.getResult(),
      /*restrict=*/true, /*writable=*/true);

  (*dagGraph.valueTypes)[toTensorOp.getResult()] = AffinityDAG::CoreType::CUBE;

  return toTensorOp.getResult();
}

Value DataMovementBuilder::buildMovement(Value srcValue, Instruction *srcInst,
                                         Instruction *dstInst,
                                         AffinityDAG::CoreType srcType,
                                         AffinityDAG::CoreType dstType) {
  MovementConfig config;

  if (srcType == AffinityDAG::CoreType::CUBE &&
      dstType == AffinityDAG::CoreType::VECTOR) {
    return buildCubeToVectorMovement(srcValue, dstInst, config);
  } else if (srcType == AffinityDAG::CoreType::VECTOR &&
             dstType == AffinityDAG::CoreType::CUBE) {
    return buildVectorToCubeMovement(srcValue, dstInst, config);
  }

  return nullptr;
}

Value DataMovementBuilder::getOrCreateAllocation(Type tensorType,
                                                 hivm::AddressSpace addrSpace,
                                                 Location loc) {
  auto rankedTensorType = dyn_cast<RankedTensorType>(tensorType);
  if (!rankedTensorType)
    return nullptr;

  // 检查缓存
  auto cacheKey = std::make_pair(tensorType, addrSpace);
  auto it = allocationCache.find(cacheKey);
  if (it != allocationCache.end()) {
    return it->second;
  }

  auto elementType = rankedTensorType.getElementType();
  auto shape = rankedTensorType.getShape();

  auto addressSpaceAttr = hivm::AddressSpaceAttr::get(builder.getContext(), addrSpace);
  auto memrefType = MemRefType::get(shape, elementType, /*layout=*/nullptr, addressSpaceAttr);

  // 查找函数入口 block
  Operation *funcOp = builder.getInsertionBlock()->getParentOp();
  while (funcOp && !isa<triton::FuncOp>(funcOp)) {
    funcOp = funcOp->getParentOp();
  }

  Value alloc;
  if (auto func = dyn_cast<triton::FuncOp>(funcOp)) {
    // 在函数入口创建 allocation
    Block &entryBlock = func.getBody().front();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&entryBlock);
    alloc = builder.create<memref::AllocOp>(loc, memrefType);
  } else {
    // 回退到当前位置
    alloc = builder.create<memref::AllocOp>(loc, memrefType);
  }

  allocationCache[cacheKey] = alloc;
  return alloc;
}

void DataMovementBuilder::createFixpipeOp(Value src, Value dst, Location loc,
                                          const MovementConfig &config) {
  FixpipeDMAModeAttr dmaModeAttr = FixpipeDMAModeAttr::get(builder.getContext(), config.dmaMode);

  builder.create<hivm::FixpipeOp>(
      loc, TypeRange{}, src, dst,
      /*unit_flag_cond=*/nullptr,
      /*dma_mode=*/dmaModeAttr,
      /*dual_dst_mode=*/nullptr,
      /*pre_quant=*/nullptr,
      /*pre_relu=*/nullptr,
      /*channel_split=*/nullptr,
      /*unit_flag_mode=*/nullptr);
}

//===----------------------------------------------------------------------===//
// SyncInserter (重构 insertSyncAndMovement)
//
// 原始代码问题：
// 1. 复杂的参数计算和硬编码 PIPE 选择
// 2. 手动管理插入点
// 3. 跨 block 特殊处理逻辑分散
//
// 重构后改进：
// 1. 声明式的 SyncPoint 结构
// 2. 自动计算最佳插入位置
// 3. 统一的 applySyncPoints 接口
//===----------------------------------------------------------------------===//

SmallVector<SyncInserter::SyncPoint>
SyncInserter::generateSyncPoints(const CrossCoreDependency &dep, int64_t flag) {
  SmallVector<SyncPoint> syncPoints;

  auto [setPipe, waitPipe] = getPipeConfig(dep.srcType, dep.dstType);

  hivm::TCoreType setCoreType = (dep.srcType == AffinityDAG::CoreType::VECTOR)
                                    ? hivm::TCoreType::VECTOR
                                    : hivm::TCoreType::CUBE;
  hivm::TCoreType waitCoreType = (dep.dstType == AffinityDAG::CoreType::VECTOR)
                                     ? hivm::TCoreType::VECTOR
                                     : hivm::TCoreType::CUBE;

  if (dep.isCrossBlock && dep.dstIsInnerBlock) {
    // 跨 block 处理：set 在 src 后，wait 在 block 入口处
    syncPoints.push_back({SyncPoint::SET, dep.srcInst, /*insertBefore=*/false,
                          setCoreType, setPipe, waitPipe, flag});

    // 找到内层 block 的入口点
    BasicBlock *dstBlock = dep.dstInst->getParentBlock();
    if (Block *mlirBlock = dstBlock->getMLIRBlock()) {
      if (Operation *parentOp = mlirBlock->getParentOp()) {
        if (Instruction *parentInst = cfg.getInstruction(parentOp)) {
          syncPoints.push_back({SyncPoint::WAIT, parentInst, /*insertBefore=*/true,
                                waitCoreType, setPipe, waitPipe, flag});
        }
      }
    }
  } else {
    // 同 block 处理
    syncPoints.push_back({SyncPoint::SET, dep.srcInst, /*insertBefore=*/false,
                          setCoreType, setPipe, waitPipe, flag});
    syncPoints.push_back({SyncPoint::WAIT, dep.dstInst, /*insertBefore=*/true,
                          waitCoreType, setPipe, waitPipe, flag});
  }

  return syncPoints;
}

void SyncInserter::applySyncPoints(ArrayRef<SyncPoint> syncPoints) {
  for (const auto &sp : syncPoints) {
    createSyncOp(sp);
  }
}

SmallVector<SyncInserter::SyncPoint>
SyncInserter::generateForLoopSyncPoints(scf::ForOp forOp,
                                        ArrayRef<CrossCoreDependency> deps,
                                        int64_t startFlag) {
  SmallVector<SyncPoint> syncPoints;
  int64_t flag = startFlag;

  for (const auto &dep : deps) {
    auto [setPipe, waitPipe] = getPipeConfig(dep.srcType, dep.dstType);

    hivm::TCoreType setCoreType = (dep.srcType == AffinityDAG::CoreType::VECTOR)
                                      ? hivm::TCoreType::VECTOR
                                      : hivm::TCoreType::CUBE;
    hivm::TCoreType waitCoreType = (dep.dstType == AffinityDAG::CoreType::VECTOR)
                                       ? hivm::TCoreType::VECTOR
                                       : hivm::TCoreType::CUBE;

    // set 在 yieldDefiningOp 后
    syncPoints.push_back({SyncPoint::SET, dep.srcInst, /*insertBefore=*/false,
                          setCoreType, setPipe, waitPipe, flag});

    // wait 在 firstUser 前
    syncPoints.push_back({SyncPoint::WAIT, dep.dstInst, /*insertBefore=*/true,
                          waitCoreType, setPipe, waitPipe, flag});

    flag++;
  }

  return syncPoints;
}

std::pair<hivm::PIPE, hivm::PIPE>
SyncInserter::getPipeConfig(AffinityDAG::CoreType srcType,
                            AffinityDAG::CoreType dstType) {
  if (srcType == AffinityDAG::CoreType::CUBE &&
      dstType == AffinityDAG::CoreType::VECTOR) {
    return {hivm::PIPE::PIPE_FIX, hivm::PIPE::PIPE_V};
  } else if (srcType == AffinityDAG::CoreType::VECTOR &&
             dstType == AffinityDAG::CoreType::CUBE) {
    return {hivm::PIPE::PIPE_MTE3, hivm::PIPE::PIPE_MTE1};
  }

  // 默认配置
  return {hivm::PIPE::PIPE_MTE3, hivm::PIPE::PIPE_MTE1};
}

void SyncInserter::createSyncOp(const SyncPoint &sp) {
  Operation *anchorOp = sp.anchor->getOperation();
  Location loc = anchorOp->getLoc();

  if (sp.insertBefore) {
    builder.setInsertionPoint(anchorOp);
  } else {
    builder.setInsertionPointAfter(anchorOp);
  }

  auto coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), sp.coreType);
  auto setPipeAttr = hivm::PipeAttr::get(builder.getContext(), sp.setPipe);
  auto waitPipeAttr = hivm::PipeAttr::get(builder.getContext(), sp.waitPipe);
  auto flagId = builder.getIntegerAttr(builder.getI64Type(), sp.flag);

  if (sp.type == SyncPoint::SET) {
    builder.create<hivm::SyncBlockSetOp>(loc, coreAttr, setPipeAttr, waitPipeAttr, flagId);
  } else {
    builder.create<hivm::SyncBlockWaitOp>(loc, coreAttr, setPipeAttr, waitPipeAttr, flagId);
  }
}

//===----------------------------------------------------------------------===//
// DotLegalizer (重构 LegalizeDot)
//
// 原始代码问题：
// 1. 手动 walk 查找 dot 操作
// 2. 复杂的累加器零值检查
// 3. 手动创建替换链
//
// 重构后改进：
// 1. 使用 CFGTraverser 收集所有 dot 操作
// 2. 统一的累加器检查
// 3. 自动替换链创建
//===----------------------------------------------------------------------===//

void DotLegalizer::legalize(triton::FuncOp funcOp) {
  CFGTraverser cfgTraverser(cfg);

  class DotCollector : public CFGTraversalBase {
  public:
    DotCollector(SmallVector<triton::DotOp> &dots) : dots(dots) {}

    bool preVisitInstruction(Instruction *inst, TraversalContext &ctx) override {
      if (auto dotOp = dyn_cast<triton::DotOp>(inst->getOperation())) {
        dots.push_back(dotOp);
      }
      return true;
    }

  private:
    SmallVector<triton::DotOp> &dots;
  };

  SmallVector<triton::DotOp> dotOps;
  DotCollector collector(dotOps);
  cfgTraverser.dfsForward(collector);

  // 处理每个 dot 操作
  for (auto dotOp : dotOps) {
    legalizeDotOp(dotOp);
  }
}

bool DotLegalizer::isZeroAccumulator(Value acc) {
  if (auto constantOp = acc.getDefiningOp<arith::ConstantOp>()) {
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValue())) {
      if (denseAttr.isSplat()) {
        if (auto floatAttr = dyn_cast<FloatAttr>(denseAttr.getSplatValue<Attribute>())) {
          return floatAttr.getValueAsDouble() == 0.0;
        }
      }
    }
  }
  return false;
}

void DotLegalizer::legalizeDotOp(triton::DotOp dotOp) {
  Value a = dotOp.getOperands()[0];
  Value b = dotOp.getOperands()[1];
  Value c = dotOp.getOperands()[2];

  // 检查累加器是否为零
  if (isZeroAccumulator(c)) {
    return;  // 已经是合法的
  }

  Location loc = dotOp.getLoc();
  auto resultType = dotOp.getResult().getType();

  builder.setInsertionPoint(dotOp);

  // 创建全零张量
  auto zeroAttr = DenseElementsAttr::get(
      dyn_cast<RankedTensorType>(resultType), APFloat(0.0f));
  auto zeroConstant = builder.create<arith::ConstantOp>(loc, zeroAttr);

  // 创建新的 dot 操作，使用零作为累加器
  auto newDot = builder.create<triton::DotOp>(loc, resultType, a, b, zeroConstant);

  // 创建加法操作
  auto addOp = builder.create<arith::AddFOp>(loc, newDot, c);

  // 替换原来的 dotOp
  dotOp.getResult().replaceAllUsesWith(addOp.getResult());

  // 删除原 dotOp
  if (dotOp.use_empty()) {
    dotOp.erase();
  }
}

//===----------------------------------------------------------------------===//
// CopyChainRewriter (重构 rewriteCopyChainForCbub)
//
// 原始代码问题：
// 1. 硬编码的 reshape/trans 顺序
// 2. 手动形状计算
// 3. 类型更新分散
//
// 重构后改进：
// 1. 统一的形状计算
// 2. 自动转换链创建
// 3. 集中类型更新
//===----------------------------------------------------------------------===//

void CopyChainRewriter::rewriteAllCopyChains(triton::FuncOp funcOp) {
  SmallVector<hivm::CopyOp> copyOps;

  // 收集所有 copy 操作
  funcOp.walk([&](hivm::CopyOp copyOp) {
    copyOps.push_back(copyOp);
  });

  // 重写每个 copy 操作
  for (auto copyOp : copyOps) {
    rewriteCopyChain(copyOp);
  }
}

bool CopyChainRewriter::rewriteCopyChain(hivm::CopyOp copyOp) {
  // 获取 copy 的输入（ins），应为 to_memref 的结果
  Value insVal = copyOp.getOperands()[0];
  auto toMemRefOp = insVal.getDefiningOp<bufferization::ToMemrefOp>();
  if (!toMemRefOp)
    return false;

  Value inputTensor = toMemRefOp.getTensor();
  auto inputTensorType = dyn_cast<RankedTensorType>(inputTensor.getType());
  if (!inputTensorType || inputTensorType.getRank() != 2)
    return false;

  // 计算新形状
  auto newShapeOpt = computeNDToNZShape(inputTensorType);
  if (!newShapeOpt)
    return false;

  auto newShape = *newShapeOpt;
  auto elementType = inputTensorType.getElementType();
  auto finalTensorType = RankedTensorType::get(newShape, elementType);

  Location loc = inputTensor.getLoc();

  // 在 toMemRefOp 前插入转换链
  builder.setInsertionPoint(toMemRefOp);

  // 创建 reshape + trans 链
  Value converted = createReshapeTransChain(inputTensor, newShape, loc);
  if (!converted)
    return false;

  // 创建新的 to_memref
  auto newMemRefType = MemRefType::get(
      newShape, elementType, AffineMap{}, toMemRefOp.getType().getMemorySpace());

  auto newToMemRefOp = builder.create<bufferization::ToMemrefOp>(
      toMemRefOp.getLoc(), newMemRefType, converted);

  (*dagGraph.valueTypes)[newToMemRefOp.getResult()] = AffinityDAG::CoreType::VECTOR;

  // 创建新的 copyOp
  builder.setInsertionPoint(copyOp);
  auto resultTypes = copyOp->getResultTypes();
  auto newCopyOp = builder.create<hivm::CopyOp>(
      copyOp.getLoc(), resultTypes, newToMemRefOp.getResult(),
      copyOp.getOperands()[1]);

  // 替换 uses 并清理旧 op
  copyOp.replaceAllUsesWith(newCopyOp);
  copyOp.erase();
  toMemRefOp.erase();

  return true;
}

std::optional<SmallVector<int64_t, 4>>
CopyChainRewriter::computeNDToNZShape(RankedTensorType tensorType) {
  if (tensorType.getRank() != 2)
    return std::nullopt;

  auto shape = tensorType.getShape();
  int64_t M = shape[0];
  int64_t N = shape[1];

  // 必须是静态且 16 对齐
  if (ShapedType::isDynamic(M) || ShapedType::isDynamic(N))
    return std::nullopt;
  if (M % 16 != 0 || N % 16 != 0)
    return std::nullopt;

  // 新 shape: (N/16, M/16, 16, 16)
  return SmallVector<int64_t, 4>{N / 16, M / 16, 16, 16};
}

std::optional<SmallVector<int64_t, 4>>
CopyChainRewriter::computeIntermediateShape(RankedTensorType tensorType) {
  if (tensorType.getRank() != 2)
    return std::nullopt;

  auto shape = tensorType.getShape();
  int64_t M = shape[0];
  int64_t N = shape[1];

  if (ShapedType::isDynamic(M) || ShapedType::isDynamic(N))
    return std::nullopt;
  if (M % 16 != 0 || N % 16 != 0)
    return std::nullopt;

  // 中间形状: [M/16, 16, N/16, 16]
  return SmallVector<int64_t, 4>{M / 16, 16, N / 16, 16};
}

Value CopyChainRewriter::createReshapeTransChain(Value input,
                                                  const SmallVector<int64_t, 4> &finalShape,
                                                  Location loc) {
  auto inputTensorType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputTensorType)
    return nullptr;

  // 计算中间形状
  auto interShapeOpt = computeIntermediateShape(inputTensorType);
  if (!interShapeOpt)
    return nullptr;

  auto elementType = inputTensorType.getElementType();
  auto interTensorType = RankedTensorType::get(*interShapeOpt, elementType);
  auto finalTensorType = RankedTensorType::get(finalShape, elementType);

  // 创建 reshape
  auto reshapeOp = builder.create<triton::ReshapeOp>(loc, interTensorType, input);
  (*dagGraph.valueTypes)[reshapeOp.getResult()] = AffinityDAG::CoreType::VECTOR;

  // 创建 trans: order = [2, 0, 1, 3]
  SmallVector<int32_t, 4> order = {2, 0, 1, 3};
  auto orderAttr = builder.getDenseI32ArrayAttr(order);

  auto transOp = builder.create<triton::TransOp>(loc, finalTensorType,
                                                  reshapeOp.getResult(), orderAttr);
  (*dagGraph.valueTypes)[transOp.getResult()] = AffinityDAG::CoreType::VECTOR;

  return transOp.getResult();
}

//===----------------------------------------------------------------------===//
// DAGSyncOptimizationDriver - 主驱动
//===----------------------------------------------------------------------===//

void DAGSyncOptimizationDriver::run(triton::FuncOp funcOp) {
  // 步骤1: 合法化 dot 操作
  legalizeDots(funcOp);

  // 步骤2: 分析同步需求
  auto deps = analyzeSyncNeeds(funcOp);

  // 步骤3: 插入数据搬运
  insertDataMovements(deps);

  // 步骤4: 插入同步操作
  insertSyncOperations(deps);

  // 步骤5: 重写 copy 链
  rewriteCopyChains(funcOp);
}

void DAGSyncOptimizationDriver::legalizeDots(triton::FuncOp funcOp) {
  dotLegalizer.legalize(funcOp);
}

SmallVector<CrossCoreDependency>
DAGSyncOptimizationDriver::analyzeSyncNeeds(triton::FuncOp funcOp) {
  SmallVector<CrossCoreDependency> allDeps;

  // 分析一般的跨核心依赖
  auto deps = analyzer.analyzeSyncRequirements(funcOp);
  allDeps.append(deps);

  // 分析 scf.for 循环内的特殊依赖
  funcOp.walk([&](scf::ForOp forOp) {
    auto forDeps = analyzer.analyzeForLoopSync(forOp);
    allDeps.append(forDeps);
  });

  return allDeps;
}

void DAGSyncOptimizationDriver::insertDataMovements(
    ArrayRef<CrossCoreDependency> deps) {
  for (const auto &dep : deps) {
    // 检查是否已经处理过
    auto pair = std::make_pair(dep.srcInst, dep.dstInst);
    if (processedPairs.contains(pair))
      continue;
    processedPairs.insert(pair);

    // 获取需要搬运的值
    Value srcValue = dep.value;
    if (!srcValue) {
      if (dep.srcInst->getOperation()->getNumResults() > 0) {
        srcValue = dep.srcInst->getOperation()->getResult(0);
      }
    }

    if (!srcValue)
      continue;

    // 构建数据搬运
    Value newValue = dataMovementBuilder.buildMovement(
        srcValue, dep.srcInst, dep.dstInst, dep.srcType, dep.dstType);

    // 更新 DAG 中的类型信息
    if (newValue) {
      // 替换 dst 操作的操作数
      if (auto dstOp = dep.dstInst->getOperation()) {
        for (unsigned i = 0; i < dstOp->getNumOperands(); ++i) {
          if (dstOp->getOperand(i) == srcValue) {
            dstOp->setOperand(i, newValue);
          }
        }
      }
    }
  }
}

void DAGSyncOptimizationDriver::insertSyncOperations(
    ArrayRef<CrossCoreDependency> deps) {
  for (const auto &dep : deps) {
    // 检查是否已经处理过
    auto pair = std::make_pair(dep.srcInst, dep.dstInst);
    if (processedPairs.contains(pair))
      continue;

    // 生成同步点
    auto syncPoints = syncInserter.generateSyncPoints(dep, nextFlag);

    // 应用同步点
    syncInserter.applySyncPoints(syncPoints);

    nextFlag++;
  }
}

void DAGSyncOptimizationDriver::rewriteCopyChains(triton::FuncOp funcOp) {
  copyRewriter.rewriteAllCopyChains(funcOp);
}
