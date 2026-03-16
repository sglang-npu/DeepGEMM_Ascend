/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonToCFG/ControlFlowGraphBuilder.h"
#include "TritonToCFG/dataflow_graph.h"

using namespace mlir;
using namespace triton::cfg;

//===----------------------------------------------------------------------===//
// Example usage of Memory SSA
//===----------------------------------------------------------------------===//

/*
Example: Analyze tensor dependencies in a matmul kernel

Input MLIR:
```mlir
func.func @matmul_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32

  // scf.for with iter_args
  scf.for %i = %c0 to %c64 step %c1 iter_args(%ptr0 = %arg0) {
    %new_ptr = tt.addptr %ptr0, %c64 : ...
    scf.yield %new_ptr : ...
  }

  return
}
```

Memory SSA output:
  - %arg0 -> [tensor_0, param]
  - %ptr0 (iter_arg) -> [tensor_0, param] (LoopPhi)
  - %new_ptr -> [tensor_0, param] (alias of %ptr0)
*/

void analyzeFunction(triton::FuncOp func) {
  llvm::outs() << "Analyzing function: " << func.getName() << "\n";

  // 1. 构建CFG
  llvm::outs() << "Step 1: Building Control Flow Graph...\n";

  ControlFlowGraphBuilder builder;
  auto cfg = builder.build(func);

  if (!cfg) {
    llvm::errs() << "Failed to build CFG\n";
    return;
  }

  llvm::outs() << "  - CFG has " << cfg->getNumBlocks() << " basic blocks\n";

  // 2. 构建DataFlowGraph（包含AliasAnalysis和MemorySSABuilder）
  llvm::outs() << "Step 2: Building Data Flow Graph with Memory SSA...\n";

  DataFlowGraph dataFlowGraph(*cfg);
  dataFlowGraph.build();

  // 3. 查询和分析
  llvm::outs() << "Step 3: Querying data flow information...\n\n";

  // 遍历函数中的所有值
  func.walk([&](Operation* op) {
    // 查询每个result的数据流信息
    for (Value result : op->getResults()) {
      auto dataFlow = dataFlowGraph.queryDataFlow(result);

      llvm::outs() << "Value: " << result << "\n";
      llvm::outs() << "  Operation: " << op->getName() << "\n";
      llvm::outs() << "  Result: " << result.getType() << "\n";

      switch (dataFlow.kind) {
        case DataFlowInfo::DataFlowResult::ResultKind::MEMORY_SSA:
          llvm::outs() << "  DataFlow: MEMORY_SSA\n";
          llvm::outs() << "  Definition: " << dataFlow.definition->getId() << "\n";
          if (dataFlow.definition->isParameter()) {
            llvm::outs() << "  Type: Function Parameter\n";
          }
          break;

        case DataFlowInfo::DataFlowResult::ResultKind::SSA:
          llvm::outs() << "  DataFlow: Traditional SSA\n";
          break;

        case DataFlowInfo::DataFlowResult::ResultKind::NONE:
          llvm::outs() << "  DataFlow: None\n";
          break;
      }

      // 查询uses
      auto uses = dataFlow.uses;
      if (!uses.empty()) {
        llvm::outs() << "  Uses (" << uses.size() << "):\n";
        for (auto* use : uses) {
          llvm::outs() << "    - " << use->getOwner()->getName() << "\n";
        }
      }

      llvm::outs() << "\n";
    }
  });

  // 4. 导出可视化结果
  llvm::outs() << "Step 4: Exporting visualization...\n";

  std::string funcName = func.getName().str();
  if (funcName.empty()) funcName = "unknown";

  // 导出CFG
  std::string cfgFile = funcName + "_cfg.json";
  std::error_code ec;
  llvm::raw_fd_ostream cfgOs(cfgFile, ec);
  if (!ec) {
    cfg->exportToJSON(cfgOs);
    llvm::outs() << "  - CFG exported to " << cfgFile << "\n";
  }

  // 导出数据流图
  std::string dfFile = funcName + "_dataflow.json";
  llvm::raw_fd_ostream dfOs(dfFile, ec);
  if (!ec) {
    dataFlowGraph.exportToJSON(dfOs);
    llvm::outs() << "  - DataFlow exported to " << dfFile << "\n";
  }

  llvm::outs() << "\nAnalysis complete!\n";
}

//===----------------------------------------------------------------------===//
// Integration with existing BuildCFGPass
//===----------------------------------------------------------------------===//

/*
To integrate Memory SSA into the existing BuildCFGPass:

1. In BuildCFGPass::runOnOperation():
   - After building CFG, create DataFlowGraph
   - Call dataFlowGraph.build()
   - Optionally export or analyze

2. Example:

void BuildCFGPass::runOnOperation() {
  auto module = getOperation();

  for (triton::FuncOp func : module.getOps<triton::FuncOp>()) {
    // Build CFG (existing code)
    auto cfg = buildForFunction(func);

    // Build DataFlowGraph with Memory SSA (new)
    DataFlowGraph dataFlowGraph(*cfg);
    dataFlowGraph.build();

    // Analyze or export
    analyzeDataFlow(dataFlowGraph);
  }
}

3. Benefits:
   - Tracks tensor/pointer definitions and uses
   - Handles aliases (addptr, make_tensor_ptr)
   - Supports control flow (if, for, while)
   - Unified interface for memory operations
*/
