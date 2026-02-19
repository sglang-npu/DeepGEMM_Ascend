# Triton Ascend Adapter Opt Tool
# MLIR-based tool for debugging and testing IR transformations

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

# triton-adapter-opt
add_llvm_executable(triton-adapter-opt
  triton-adapter-opt.cpp

  PARTIAL_SOURCES_INTENDED
)

# 设置编译标志
llvm_update_compile_flags(triton-adapter-opt)

# 链接昇腾相关的库
set(ASCEND_LIBS
  # 核心 IR
  ascend_lib_ascend_ir

  # Dialects
  TritonAscendDialect
  ScopeDialect
  HIVMDialect

  # Passes
  TritonToStructured
  TritonToAnnotation
  TritonToLinalg
  TritonToUnstructure
  TritonToHIVM
  TritonToHFusion
  TritonToLLVM
  DiscreteMaskAccessConversion
  BubbleUpOperation

  # Triton 核心
  TritonLLVMIR
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR

  # Analysis
  TritonAnalysis
  TritonGPUTransforms
  TritonTransforms
  TritonNvidiaGPUTransforms
)

# 链接所有库
target_link_libraries(triton-adapter-opt PRIVATE
  ${ASCEND_LIBS}
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}

  # MLIR core
  MLIROptLib
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRLLVMIR
  MLIRTransforms
)

# 检查链接
triton_check_all_link_libraries(triton-adapter-opt)

# 安装规则
install(TARGETS triton-adapter-opt
  RUNTIME DESTINATION bin
)

# 输出信息
message(STATUS "Configured triton-adapter-opt tool")
