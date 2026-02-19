#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

// Triton 核心 Dialect
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

// 昇腾 Dialect
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "ascend/include/Dialect/Scope/IR/ScopeDialect.h"
#include "ascend/include/Dialect/HIVM/IR/HIVMDialect.h"

// 昇腾 Pass
#include "ascend/include/TritonToStructured/Passes.h"
#include "ascend/include/TritonToAnnotation/Passes.h"
#include "ascend/include/TritonToLinalg/Passes.h"
#include "ascend/include/TritonToUnstructure/Passes.h"
#include "ascend/include/TritonToHIVM/Passes.h"
#include "ascend/include/TritonToHFusion/Passes.h"
#include "ascend/include/TritonToLLVM/Passes.h"
#include "ascend/include/DiscreteMaskAccessConversion/Passes.h"
#include "ascend/include/BubbleUpOperation/Passes.h"

using namespace mlir;

// 注册所有昇腾 Pass
void registerAscendPasses() {
  // 结构化和语义转换
  mlir::triton::registerTritonToStructuredPass();
  mlir::triton::registerTritonToAnnotationPass();
  mlir::triton::registerTritonToUnstructurePass();

  // IR 转换
  mlir::triton::registerTritonToLinalgPass();
  mlir::triton::registerTritonToHIVMPass();
  mlir::triton::registerTritonToHFusionPass();
  mlir::triton::registerTritonToLLVMPass();

  // 优化 Pass
  mlir::triton::registerDiscreteMaskAccessConversionPass();
  mlir::triton::registerBubbleUpOperationPass();
}

int main(int argc, char **argv) {
  DialectRegistry registry;

  // 注册核心 MLIR Dialects
  registry.insert<bufferization::BufferizationDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<gpu::GPUDialect>();
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<vector::VectorDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<complex::ComplexDialect>();

  // 注册 Triton Dialects
  registry.insert<mlir::triton::TritonDialect>();
  registry.insert<mlir::triton::gpu::TritonGPUDialect>();

  // 注册昇腾 Dialects
  registry.insert<mlir::triton::ascend::TritonAscendDialect>();
  registry.insert<mlir::triton::ascend::ScopeDialect>();
  registry.insert<mlir::triton::ascend::HIVMDialect>();

  // 注册所有 Pass
  registerAscendPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Triton Ascend Adapter optimizer driver\n", registry));
}
