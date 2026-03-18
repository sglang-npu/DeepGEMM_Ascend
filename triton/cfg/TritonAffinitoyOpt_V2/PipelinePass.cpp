/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * PipelinePass实现
 * Pass入口，协调分析器和变换器
 */

#include "PipelineAnalyzer.h"
#include "TritonAffinityOpt/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {
namespace affinity {

#define GEN_PASS_DEF_PIPELINEANALYSIS
#include "ascend/include/TritonAffinityOpt/Passes.h.inc"

// ==================== PipelinePassCoordinator ====================

void PipelinePassCoordinator::processFunction(triton::FuncOp func) {
  // 构建CFG
  auto cfg = buildCFG(func);
  if (!cfg)
    return;

  // 构建数据流图
  cfg::DataFlowGraph dataFlowGraph(*cfg);
  dataFlowGraph.build();

  // 处理所有for循环
  func.walk([&](scf::ForOp forOp) {
    processLoopWithDeps(forOp, *cfg, dataFlowGraph.getDataFlowInfo());
  });
}

void PipelinePassCoordinator::processLoop(scf::ForOp forOp) {
  // 获取包含该循环的函数
  auto func = forOp->getParentOfType<triton::FuncOp>();
  if (!func)
    return;

  // 构建CFG
  auto cfg = buildCFG(func);
  if (!cfg)
    return;

  // 构建数据流图
  cfg::DataFlowGraph dataFlowGraph(*cfg);
  dataFlowGraph.build();

  processLoopWithDeps(forOp, *cfg, dataFlowGraph.getDataFlowInfo());
}

void PipelinePassCoordinator::processLoopWithDeps(
    scf::ForOp forOp, ControlFlowGraph& cfg, DataFlowInfo& dfi) {
  // 1. 分析pipeline结构
  PipelineAnalyzer analyzer(cfg, dfi);
  auto analysisResult = analyzer.analyze(forOp);

  if (analysisResult.mergedRegions.empty())
    return;

  // 2. 处理依赖值，更新for循环
  OpBuilder builder(forOp);
  PipelineTransformer transformer(analysisResult, builder);

  scf::ForOp currentForOp = forOp;
  if (!analysisResult.dependValues.empty()) {
    currentForOp = transformer.addIterArgsForDeps(forOp, analysisResult.dependValues);

    // 重新分析更新后的循环
    PipelineAnalyzer newAnalyzer(cfg, dfi);
    analysisResult = newAnalyzer.analyze(currentForOp);
  }

  // 3. 创建if regions
  if (config_.enableIfCondition) {
    transformer.createIfRegions(currentForOp, analysisResult.mergedRegions,
                                analysisResult.dependValues);
  }

  // 4. 应用SSBuffer控制
  if (config_.enableSSBuffer) {
    CoreKind coreKind = analysisResult.mergedRegions.empty()
                           ? CoreKind::UNKNOWN
                           : analysisResult.mergedRegions[0].coreKind;
    transformer.insertSSBufferControl(currentForOp, coreKind);
  }

  // 5. 应用双缓冲
  if (config_.enableDoubleBuffer) {
    transformer.applyDoubleBuffering(currentForOp, config_.bufferDepth);
  }
}

std::unique_ptr<ControlFlowGraph> PipelinePassCoordinator::buildCFG(
    triton::FuncOp func) {
  auto cfg = std::make_unique<ControlFlowGraph>(func);
  cfg->build();
  return cfg;
}

// ==================== PipelinePass ====================

struct PipelinePass
    : public impl::PipelineAnalysisBase<PipelinePass> {
  void runOnOperation() override {
    auto func = getOperation();

    PipelineConfig config;
    // 可以从命令行选项读取配置
    config.enableIfCondition = true;
    config.enableDoubleBuffer = true;
    config.enableSSBuffer = true;
    config.bufferDepth = 2;

    PipelinePassCoordinator coordinator(config);
    coordinator.processFunction(func);
  }
};

// ==================== Helper Functions ====================

// 处理advance操作形式转换
void transformAdvanceOpForm(scf::ForOp forOp) {
  Block &body = forOp.getRegion().front();

  SmallVector<scf::IfOp, 4> ifOps;
  for (Operation &op : body) {
    if (auto ifOp = dyn_cast<scf::IfOp>(&op))
      ifOps.push_back(ifOp);
  }

  for (scf::IfOp ifOp : ifOps) {
    // 找then region中的advance
    triton::AdvanceOp advanceOp;
    for (Operation &thenOp : ifOp.getThenRegion().front()) {
      if (auto adv = dyn_cast<triton::AdvanceOp>(thenOp)) {
        advanceOp = adv;
        break;
      }
    }
    if (!advanceOp)
      continue;

    // base必须是for的iter_arg
    Value base = advanceOp.getPtr();
    auto barg = dyn_cast<BlockArgument>(base);
    if (!barg || barg.getOwner() != &body)
      continue;

    // yield去掉advance的返回值
    auto thenYield =
        cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
    auto elseYield =
        cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());

    int advanceIdx = -1;
    for (auto it : llvm::enumerate(thenYield.getOperands())) {
      if (it.value() == advanceOp.getResult()) {
        advanceIdx = it.index();
        break;
      }
    }

    if (advanceIdx == -1)
      continue;

    // 删除advance
    SmallVector<Value> thenOps(thenYield.getOperands().begin(),
                               thenYield.getOperands().end());
    SmallVector<Value> elseOps(elseYield.getOperands().begin(),
                               elseYield.getOperands().end());

    thenOps.erase(thenOps.begin() + advanceIdx);
    elseOps.erase(elseOps.begin() + advanceIdx);

    thenYield->setOperands(thenOps);
    elseYield->setOperands(elseOps);

    // 重建ifOp（去掉advance对应的result）
    OpBuilder ifBuilder(ifOp);
    ifBuilder.setInsertionPoint(ifOp);

    // 构造新的result types
    SmallVector<Type> newResultTypes;
    for (int i = 0; i < static_cast<int>(ifOp.getNumResults()); ++i) {
      if (i != advanceIdx)
        newResultTypes.push_back(ifOp.getResult(i).getType());
    }

    // 创建新的if
    auto newIf = ifBuilder.create<scf::IfOp>(ifOp.getLoc(), newResultTypes,
                                            ifOp.getCondition(),
                                            /*withElseRegion=*/true);

    // 把已经修改过yield的region搬过去
    newIf.getThenRegion().takeBody(ifOp.getThenRegion());
    newIf.getElseRegion().takeBody(ifOp.getElseRegion());

    // 替换if result的user
    int newIdx = 0;
    for (int oldIdx = 0; oldIdx < static_cast<int>(ifOp.getNumResults());
         ++oldIdx) {
      if (oldIdx == advanceIdx)
        continue;
      ifOp.getResult(oldIdx).replaceAllUsesWith(newIf.getResult(newIdx++));
    }

    OpBuilder builder(newIf);
    builder.setInsertionPointAfter(newIf);

    Value flag = newIf.getCondition();

    SmallVector<Value, 4> newOffsets;
    for (Value off : advanceOp.getOffsets()) {
      auto intTy = cast<IntegerType>(off.getType());
      auto zero = builder.create<arith::ConstantIntOp>(newIf.getLoc(), 0,
                                                       intTy.getWidth());
      auto sel = builder.create<arith::SelectOp>(newIf.getLoc(), flag, off, zero);
      newOffsets.push_back(sel);
    }

    auto newAdvance = builder.create<triton::AdvanceOp>(
        newIf.getLoc(), base.getType(), base, newOffsets);

    // 原if的advance result的users，接到newAdvance
    ifOp.getResult(advanceIdx).replaceAllUsesWith(newAdvance.getResult());

    // 删除旧的ifOp和advance
    advanceOp.erase();
    ifOp.erase();
  }
}

} // namespace affinity
} // namespace triton

// ==================== Pass Registration ====================

std::unique_ptr<OperationPass<triton::FuncOp>>
triton::createPipelineAnalysisPass() {
  return std::make_unique<triton::affinity::PipelinePass>();
}

} // namespace mlir
