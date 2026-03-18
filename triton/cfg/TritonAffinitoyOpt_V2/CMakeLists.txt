add_triton_library(TritonAffinityOptNew
  CoreSyncAnalyzer.cpp
  CoreSyncPass.cpp
  PipelineAnalyzer.cpp
  RegionSelector.cpp
  PipelineTransformer.cpp
  PipelinePass.cpp

  DEPENDS
  TritonToGraph

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRBufferizationDialect
  TritonIR
  TritonToGraph
)

target_include_directories(TritonAffinityOptNew PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/third_party/ascend/lib
)
