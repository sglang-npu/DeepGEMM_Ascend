add_triton_library(TritonToCFG
  ControlFlowGraph.cpp
  ControlFlowGraphBuilder.cpp
  InterProceduralCFG.cpp
  Passes.cpp
  tensor.cpp
  memory_ssa.cpp
  alias_analysis.cpp
  memory_ssa_builder.cpp
  dataflow_graph.cpp

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
