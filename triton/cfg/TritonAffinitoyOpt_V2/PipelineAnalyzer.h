/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * 基于TritonToGraph的Pipeline分析器
 * 用于华为昇腾NPU Cube-Vector自动流水排布
 */

#ifndef TRITON_AFFINITY_PIPELINE_ANALYZER_H
#define TRITON_AFFINITY_PIPELINE_ANALYZER_H

#include "TritonToGraph/ControlFlowGraph.h"
#include "TritonToGraph/DataflowGraph.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace triton {
namespace affinity {

using namespace cfg;

// Pipeline阶段类型
enum class PipelineStage {
  WAIT,       // 等待阶段
  COMPUTE,    // 计算阶段
  SET,        // 设置阶段
  TRANSFER,   // 数据传输阶段 (copy/fixpipe)
};

// Core类型
enum class CoreKind {
  CUBE,       // AIC (矩阵计算)
  VECTOR,     // AIV (向量计算)
  UNKNOWN
};

// Wait-Set Region 定义
struct WaitSetRegion {
  Operation* waitOp = nullptr;           // 起始wait操作
  Operation* setOp = nullptr;            // 结束set操作
  SmallVector<Operation*> ops;           // 区域内的所有操作
  bool hasTransferOp = false;            // 是否包含copy/fixpipe
  CoreKind coreKind = CoreKind::UNKNOWN; // 所属Core类型

  bool isValid() const { return waitOp && setOp; }
};

// MergedRegion - 合并后的执行区域
struct MergedRegion {
  SmallVector<WaitSetRegion*> sourceRegions;  // 来源的wait-set regions
  SmallVector<Operation*> opsToMove;          // 需要移动的操作
  SmallVector<Value> yieldValues;             // 需要yield的值
  SmallVector<Type> resultTypes;              // 结果类型
  CoreKind coreKind = CoreKind::UNKNOWN;      // 所属Core类型
};

// Pipeline分析结果
struct PipelineAnalysisResult {
  SmallVector<WaitSetRegion> waitSetRegions;     // 所有wait-set区域
  SmallVector<MergedRegion> mergedRegions;       // 合并后的区域
  SmallVector<Value> dependValues;               // 跨region依赖值
  DenseMap<Operation*, int> opToRegion;          // 操作到region的映射
};

// ==================== PipelineAnalyzer ====================
// 职责：基于CFG分析Pipeline结构，识别wait-set region和合并机会

class PipelineAnalyzer {
public:
  PipelineAnalyzer(ControlFlowGraph& cfg, DataFlowInfo& dfi)
      : cfg_(cfg), dataFlowInfo_(dfi) {}

  // 主分析入口
  PipelineAnalysisResult analyze(scf::ForOp forOp);

  // 获取操作所在的Core类型
  static CoreKind getCoreKind(Operation* op);

  // 判断是否为核心传输操作
  static bool isTransferOp(Operation* op);

  // 判断是否为核心同步操作
  static bool isSyncOp(Operation* op);

  // 判断是否为wait操作
  static bool isWaitOp(Operation* op);

  // 判断是否为set操作
  static bool isSetOp(Operation* op);

  // 获取循环的Core类型
  static CoreKind getCoreKindForLoop(scf::ForOp forOp);

private:
  ControlFlowGraph& cfg_;
  DataFlowInfo& dataFlowInfo_;

  // 分析单个基本块的wait-set regions
  void analyzeBlockRegions(BasicBlock& bb, SmallVector<WaitSetRegion>& regions);

  // 合并wait-set regions
  void mergeRegions(SmallVector<WaitSetRegion>& waitSetRegions,
                    SmallVector<MergedRegion>& mergedRegions);

  // 扩展region包含更多操作（AIV模式：基于yield value）
  void expandRegionsForAIV(scf::ForOp forOp,
                           SmallVector<MergedRegion>& mergedRegions);

  // 扩展region包含更多操作（AIC模式：基于region起始）
  void expandRegionsForAIC(scf::ForOp forOp,
                           SmallVector<MergedRegion>& mergedRegions);

  // 识别跨region依赖
  void identifyCrossRegionDeps(SmallVector<MergedRegion>& mergedRegions,
                               SmallVector<Value>& dependValues);

  // 工具函数：查找操作的index
  DenseMap<Operation*, int> computeOpIndices(Block& body);

  // 计算yield values
  void computeYieldValues(MergedRegion& mr, Block& body);

  // AIC特殊处理：移动iter_arg users
  void moveIterArgUsersForAIC(scf::ForOp forOp,
                              SmallVector<MergedRegion>& mergedRegions);

  // 查找target region
  int findTargetRegion(Operation* startOp, Block& body,
                       DenseMap<Operation*, int>& opToRegion);

  // 贪心吸收操作到region
  void greedyAbsorbToRegion(Operation* startOp, int regionIdx, int lowerBound,
                            Block& body, DenseMap<Operation*, int>& opIndex,
                            DenseMap<Operation*, int>& opToRegion,
                            SmallVector<MergedRegion>& mergedRegions);
};

// ==================== RegionSelector ====================
// 职责：决定哪些操作应该被移动到if region中

class RegionSelector {
public:
  RegionSelector(const PipelineAnalysisResult& analysis)
      : analysis_(analysis) {}

  // 选择需要yield的值
  void selectYieldValues(MergedRegion& region);

  // 查找迭代参数依赖
  Value findIterArgSource(Value v, Type expectedType);

  // 计算else分支的yield值
  void computeElseYieldValues(const MergedRegion& region,
                              SmallVector<Value>& elseValues,
                              ArrayRef<Value> dependValues);

private:
  const PipelineAnalysisResult& analysis_;

  // 检查值是否在region外被使用
  bool isUsedOutsideRegion(Value v, const MergedRegion& region);
};

// ==================== PipelineTransformer ====================
// 职责：执行实际的代码变换

class PipelineTransformer {
public:
  PipelineTransformer(const PipelineAnalysisResult& analysis,
                      OpBuilder& builder)
      : analysis_(analysis), builder_(builder) {}

  // 为依赖值添加for循环参数
  scf::ForOp addIterArgsForDeps(scf::ForOp forOp,
                                 ArrayRef<Value> dependValues);

  // 创建if包装region
  void createIfRegions(scf::ForOp forOp,
                       SmallVector<MergedRegion>& mergedRegions,
                       ArrayRef<Value> dependValues);

  // 移动iter_arg用户到if region
  void moveIterArgUsers(scf::ForOp forOp,
                        SmallVector<MergedRegion>& mergedRegions);

  // 应用双缓冲
  void applyDoubleBuffering(scf::ForOp forOp, int bufferDepth = 2);

  // 插入SSBuffer控制流
  void insertSSBufferControl(scf::ForOp forOp, CoreKind coreKind);

private:
  const PipelineAnalysisResult& analysis_;
  OpBuilder& builder_;

  // 创建单个if region
  scf::IfOp createIfOpForRegion(const MergedRegion& region,
                                Operation* insertPoint,
                                ArrayRef<Value> elseValues);

  // 更新region内的操作引用（使用IRMapping）
  void updateRegionOpsWithMapping(SmallVector<MergedRegion>& mergedRegions,
                                  IRMapping& mapper);

  // 创建buffer条件检查
  Value createBufferCondition(OpBuilder& builder, Location loc,
                              int bufferIdx, CoreKind coreKind);

  // 插入AIC buffer控制
  void insertAICBufferControl(scf::ForOp forOp, OpBuilder& builder);

  // 插入AIV buffer控制
  void insertAIVBufferControl(scf::ForOp forOp, OpBuilder& builder);
};

// ==================== PipelinePass ====================
// 职责：Pass入口，协调分析器和变换器

struct PipelineConfig {
  bool enableIfCondition = true;    // 启用if条件优化
  bool enableDoubleBuffer = true;   // 启用双缓冲
  bool enableSSBuffer = true;       // 启用SSBuffer控制
  int bufferDepth = 2;              // 缓冲深度
};

class PipelinePassCoordinator {
public:
  PipelinePassCoordinator(const PipelineConfig& config)
      : config_(config) {}

  // 处理单个函数
  void processFunction(triton::FuncOp func);

  // 处理单个循环
  void processLoop(scf::ForOp forOp);

private:
  PipelineConfig config_;

  // 初始化CFG和DataFlow分析
  std::unique_ptr<ControlFlowGraph> buildCFG(triton::FuncOp func);
};

} // namespace affinity
} // namespace triton
} // namespace mlir

#endif // TRITON_AFFINITY_PIPELINE_ANALYZER_H
