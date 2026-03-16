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

#include "TritonToCFG/dataflow_graph.h"
#include "TritonToCFG/alias_analysis.h.h"
#include "TritonToCFG/ControlFlowGraph.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "dataflow-graph"

using namespace mlir;
using namespace triton;
using namespace cfg;

//===----------------------------------------------------------------------===//
// DataFlowInfo
//===----------------------------------------------------------------------===//

DataFlowInfo::DataFlowResult DataFlowInfo::queryDataFlow(Value value) const {
  // 1. 优先查询Memory SSA
  if (Definition* def = getMemoryDefinition(value)) {
    return {
      .kind = DataFlowResult::ResultKind::MEMORY_SSA,
      .originOp = def->getDefOp(),
      .definition = def,
      .uses = getSSAUses(value)
    };
  }

  // 2. 查询传统SSA
  if (Operation* defOp = value.getDefiningOp()) {
    return {
      .kind = DataFlowResult::ResultKind::SSA,
      .originOp = defOp,
      .ssaDefinition = defOp,
      .uses = getSSAUses(value)
    };
  }

  // 3. 入参
  return {
    .kind = DataFlowResult::ResultKind::SSA,
    .originOp = nullptr,
    .ssaDefinition = nullptr,
    .uses = getSSAUses(value)
  };
}

SmallVector<Use> DataFlowInfo::getUses(Definition* def) const {
  SmallVector<Use> result;

  // 遍历所有uses，查找使用该definition的
  for (const auto& entry : memoryUses) {
    for (const Use& use : entry.second) {
      if (use.getDefinition() == def) {
        result.push_back(use);
      }
    }
  }

  return result;
}

SmallVector<Use> DataFlowInfo::getUsesByUserOp(Operation* userOp) const {
  SmallVector<Use> result;

  for (const auto& entry : memoryUses) {
    for (const Use& use : entry.second) {
      if (use.getUserOp() == userOp) {
        result.push_back(use);
      }
    }
  }

  return result;
}

void DataFlowInfo::buildDefUseCache() const {
  if (defUseCacheValid) return;

  defUseCache.clear();

  for (const auto& entry : memoryUses) {
    for (const Use& use : entry.second) {
      Definition* def = use.getDefinition();
      if (def) {
        defUseCache[def].push_back(use);
      }
    }
  }

  defUseCacheValid = true;
}

void DataFlowInfo::forEachDefinition(llvm::function_ref<void(Value, Definition*)> func) const {
  for (const auto& entry : memoryDefinitions) {
    func(entry.first, entry.second);
  }
}

void DataFlowInfo::forEachUse(llvm::function_ref<void(const Use&)> func) const {
  for (const auto& entry : memoryUses) {
    for (const Use& use : entry.second) {
      func(use);
    }
  }
}

void DataFlowInfo::print(llvm::raw_ostream& os) const {
  os << "=== Data Flow Information ===" << "\n";

  os << "Memory Definitions: " << memoryDefinitions.size() << "\n";
  for (const auto& entry : memoryDefinitions) {
    os << "  " << entry.first << " -> ";
    entry.second->print(os);
    os << "\n";
  }

  os << "Memory Uses: " << "\n";
  for (const auto& entry : memoryUses) {
    os << "  " << entry.first << ": ";
    for (const Use& use : entry.second) {
      os << "[" << use.getDefinition()->getId() << "] ";
    }
    os << "\n";
  }

  os << "Loop Phis: " << loopPhis.size() << "\n";
  for (const auto& entry : loopPhis) {
    os << "  " << entry.first << ": ";
    switch (entry.second.type) {
      case LoopPhiInfo::ITER_ARG: os << "ITER_ARG"; break;
      case LoopPhiInfo::IF_RESULT: os << "IF_RESULT"; break;
      case LoopPhiInfo::WHILE_ARG: os << "WHILE_ARG"; break;
    }
    os << "\n";
  }
}

void DataFlowInfo::exportToJSON(llvm::raw_ostream& os) const {
  os << "{\n";
  os << "  \"memoryDefinitions\": {\n";
  bool first = true;
  for (const auto& entry : memoryDefinitions) {
    if (!first) os << ",\n";
    first = false;
    os << "    \"" << entry.first << "\": {\n";
    os << "      \"id\": \"" << entry.second->getId() << "\",\n";
    os << "      \"tensor\": \"" << entry.second->getTensor()->getName() << "\",\n";
    os << "      \"version\": " << entry.second->getVersion();
    os << "\n    }";
  }
  os << "\n  },\n";

  os << "  \"loopPhis\": {\n";
  first = true;
  for (const auto& entry : loopPhis) {
    if (!first) os << ",\n";
    first = false;
    os << "    \"" << entry.first << "\": {\n";
    os << "      \"type\": " << entry.second.type << "\n";
    os << "    }";
  }
  os << "\n  }\n";
  os << "}\n";
}

//===----------------------------------------------------------------------===//
// DataFlowGraph
//===----------------------------------------------------------------------===//

void DataFlowGraph::build() {
  LLVM_DEBUG(llvm::dbgs() << "=== Starting Data Flow Graph Build ===" << "\n");

  // 步骤1: 构建Alias分析
  aliasAnalysis = std::make_unique<AliasAnalysis>();
  aliasAnalysis->analyzePointerAliases(cfg);

  LLVM_DEBUG(llvm::dbgs() << "Alias analysis complete" << "\n");

  // 步骤2: 构建Memory SSA
  memorySSABuilder = std::make_unique<MemorySSABuilder>(
      cfg, *aliasAnalysis, dataFlowInfo);
  memorySSABuilder->build();

  LLVM_DEBUG(llvm::dbgs() << "Memory SSA build complete" << "\n");

  // 步骤3: 收集所有definitions
  collectDefinitions();

  LLVM_DEBUG(llvm::dbgs() << "Definitions collected" << "\n");

  // 步骤4: 构建def-use图
  buildDefUseGraph();

  LLVM_DEBUG(llvm::dbgs()
             << "=== Data Flow Graph Build Complete ===" << "\n");
}

void DataFlowGraph::collectDefinitions() {
  // 遍历所有指令，收集definitions
  cfg.traverse([&](BasicBlock& bb) {
    for (const auto& instPtr : bb.getInstructions()) {
      const Instruction* inst = instPtr.get();
      const MemorySSAInfo& ssaInfo = inst->getMemorySSAInfo();

      for (Definition* def : ssaInfo.definitions) {
        if (def) {
          LLVM_DEBUG(llvm::dbgs() << "Collected definition: " << def->getId() << "\n");
        }
      }
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "Total memory operations tracked" << "\n");
}

void DataFlowGraph::buildDefUseGraph() {
  // 构建def-use图（在DataFlowInfo中实现）
  dataFlowInfo.buildDefUseCache();

  LLVM_DEBUG(llvm::dbgs() << "Def-use graph built" << "\n");
}

void DataFlowGraph::print(llvm::raw_ostream& os) const {
  os << "=== Data Flow Graph ===" << "\n";
  dataFlowInfo.print(os);
}

void DataFlowGraph::dump() const {
  print(llvm::errs());
}

void DataFlowGraph::exportToJSON(llvm::raw_ostream& os) const {
  auto funcName = const_cast<triton::FuncOp&>(cfg.getFunction()).getName();
  std::string funcNameStr = funcName.empty() ? "unnamed" : funcName.str();

  os << "{\n";
  os << "  \"function\": \"" << funcNameStr << "\"," << "\n";
  os << "  \"dataFlow\": ";
  dataFlowInfo.exportToJSON(os);
  os << "}\n";
}

void DataFlowGraph::exportDefUseToDOT(llvm::raw_ostream& os) const {
  os << "digraph DefUseGraph {\n";
  os << "  rankdir=TB;\n";
  os << "  node [shape=box];\n\n";

  // 遍历所有definitions并导出
  size_t nodeId = 0;
  DenseMap<const Definition*, size_t> defToNode;

  cfg.traverse([&](const BasicBlock& bb) {
    for (const auto& instPtr : bb.getInstructions()) {
      const Instruction* inst = instPtr.get();
      const MemorySSAInfo& ssaInfo = inst->getMemorySSAInfo();

      for (Definition* def : ssaInfo.definitions) {
        if (def && !defToNode.count(def)) {
          defToNode[def] = nodeId++;
          os << "  node_" << defToNode[def] << " [label=\""
             << def->getId() << "\"];\n";
        }
      }
    }
  });

  os << "}\n";
}

} // namespace cfg
} // namespace triton
} // namespace mlir
