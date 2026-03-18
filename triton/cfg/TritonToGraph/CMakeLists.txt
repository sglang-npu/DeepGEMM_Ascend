add_triton_library(TritonToCFG
  AliasAnalysis.cpp
  ControlFlowGraph.cpp
  ControlFlowGraphBuilder.cpp
  DataflowGraph.cpp
  InterProceduralCFG.cpp
  MemorySsaBuilder.cpp
  Passes.cpp

  DEPENDS
  TritonToCFGPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRDialectUtils
  MLIRIR
  MLIRFuncDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  TritonIR
  TritonTransforms
)
