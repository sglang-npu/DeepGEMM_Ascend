add_triton_library(TritonToCFG
  ControlFlowGraph.cpp
  ControlFlowGraphBuilder.cpp
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
