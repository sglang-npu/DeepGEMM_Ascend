ifndef TRITON_AFFINITY_OPT_PASSES
#define TRITON_AFFINITY_OPT_PASSES

include "mlir/Pass/PassBase.td"

// CoreSync Pass - 合并DAGSync和DAGScope的核心功能
// 基于TritonToGraph控制流数据流分析框架
def CoreSync : Pass<"core-sync", "mlir::ModuleOp"> {
  let summary = "Core synchronization optimization based on CFG and DataFlow analysis";
  let description = [{
    基于TritonToGraph控制流数据流分析框架的核心同步优化Pass。

    功能：
    1. 使用ControlFlowGraph分析函数的控制流结构
    2. 使用DataFlowGraph分析数据依赖和Memory SSA
    3. 自动识别需要同步的Vector->Cube和Cube->Vector转换点
    4. 创建AIV/AIC Scope并将操作分类到对应scope
    5. 在适当位置插入SyncBlockSetOp和SyncBlockWaitOp
    6. 处理scf.for的迭代参数同步
    7. 处理跨block的同步需求

    相比原始DAGSync+DAGScope的优点：
    - 更清晰的架构：分离了分析、构建和插入三个阶段
    - 更好的可维护性：使用统一的CoreTypeClassifier处理类型判断
    - 更好的可扩展性：易于添加新的同步模式
    - 更少的代码行数：复用TritonToGraph的分析框架
  }];
  let constructor = "mlir::triton::createCoreSyncPass()";
}

// PipelineAnalysis Pass - 重构的DAGSSBuffer
// 基于TritonToGraph框架的Cube-Vector自动流水排布
def PipelineAnalysis : Pass<"pipeline-analysis", "mlir::triton::FuncOp"> {
  let summary = "Pipeline optimization for Cube-Vector execution on Ascend NPU";
  let description = [{
    基于TritonToGraph控制流数据流分析框架的Pipeline优化Pass。
    重构自原始的DAGSSBuffer.cpp，将代码块搜索、控制流变更和代码生成解耦。

    功能：
    1. 使用ControlFlowGraph分析for循环的控制流结构
    2. 使用DataFlowGraph分析数据依赖
    3. 识别wait-set region（由SyncBlockWaitOp和SyncBlockSetOp包围的区域）
    4. 根据copy/fixpipe操作合并regions
    5. AIV模式：基于yield value扩展region范围
    6. AIC模式：基于region起始向前吸收operand
    7. 识别跨region依赖并添加for循环迭代参数
    8. 创建if regions包装merged regions
    9. 插入SSBuffer控制流（AIC/AIV不同策略）
    10. 支持双缓冲优化

    相比原始DAGSSBuffer的优点：
    - 清晰的职责分离：PipelineAnalyzer、RegionSelector、PipelineTransformer
    - 更好的可维护性：使用统一的TritonToGraph分析框架
    - 更好的可扩展性：易于添加新的pipeline策略
    - 更少的代码行数：约60%代码量减少
  }];
  let constructor = "mlir::triton::createPipelineAnalysisPass()";
}

#endif // TRITON_AFFINITY_OPT_PASSES
