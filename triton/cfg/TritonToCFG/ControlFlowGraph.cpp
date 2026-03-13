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

#include "TritonToCFG/ControlFlowGraph.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <sstream>

using namespace mlir;
using namespace triton;
using namespace cfg;

//===----------------------------------------------------------------------===//
// Instruction
//===----------------------------------------------------------------------===//

std::string Instruction::getAsString() const {
  if (!operation) return "<null operation>";
  std::string str;
  llvm::raw_string_ostream os(str);
  operation->print(os);
  return str;
}

void Instruction::print(raw_ostream &os, unsigned indent) const {
  os.indent(indent) << "Inst[" << id << "]: ";
  if (operation) {
    os << getAsString() << "\n";
  } else {
    os << "<null>\n";
  }
}

void Instruction::dump() const {
  print(llvm::errs());
}

//===----------------------------------------------------------------------===//
// BasicBlock
//===----------------------------------------------------------------------===//

void BasicBlock::addInstruction(std::unique_ptr<Instruction> inst) {
  instructions.push_back(std::move(inst));
}

Instruction *BasicBlock::getInstruction(size_t idx) const {
  if (idx < instructions.size()) {
    return instructions[idx].get();
  }
  return nullptr;
}

void BasicBlock::addSuccessor(BasicBlock *succ) {
  if (!succ) return;
  // 避免重复添加
  for (auto *s : successors) {
    if (s == succ) return;
  }
  successors.push_back(succ);
  succ->addPredecessor(this);
}

void BasicBlock::addPredecessor(BasicBlock *pred) {
  if (!pred) return;
  // 避免重复添加
  for (auto *p : predecessors) {
    if (p == pred) return;
  }
  predecessors.push_back(pred);
}

std::string BasicBlock::getName() const {
  std::string name = "BB";
  name += std::to_string(id);
  return name;
}

StringRef BasicBlock::getTypeString() const {
  switch (type) {
  case BlockType::NORMAL: return "NORMAL";
  case BlockType::ENTRY: return "ENTRY";
  case BlockType::EXIT: return "EXIT";
  case BlockType::IF_COND: return "IF_COND";
  case BlockType::FOR_COND: return "FOR_COND";
  case BlockType::WHILE_COND: return "WHILE_COND";
  case BlockType::LOOP_BODY: return "LOOP_BODY";
  case BlockType::LOOP_EXIT: return "LOOP_EXIT";
  }
  return "UNKNOWN";
}

void BasicBlock::print(raw_ostream &os) const {
  os << "============================================================\n";
  os << "BasicBlock " << getName() << " [" << getTypeString() << "]\n";
  if (parentStructure) {
    os << "  Parent Structure: " << parentStructure->getName() << "\n";
  }

  // 打印前驱
  if (!predecessors.empty()) {
    os << "  Predecessors: [";
    for (size_t i = 0; i < predecessors.size(); ++i) {
      if (i > 0) os << ", ";
      os << predecessors[i]->getName();
    }
    os << "]\n";
  }

  // 打印后继
  if (!successors.empty()) {
    os << "  Successors: [";
    for (size_t i = 0; i < successors.size(); ++i) {
      if (i > 0) os << ", ";
      os << successors[i]->getName();
    }
    os << "]\n";
  }

  // 打印指令
  os << "  Instructions (" << instructions.size() << "):\n";
  for (const auto &inst : instructions) {
    inst->print(os, 4);
  }
  os << "\n";
}

void BasicBlock::dump() const {
  print(llvm::errs());
}

void BasicBlock::exportToJSON(raw_ostream &os, unsigned indent) const {
  std::string ind(indent, ' ');
  os << ind << "{\n";
  os << ind << "  \"id\": " << id << ",\n";
  os << ind << "  \"name\": \"" << getName() << "\",\n";
  os << ind << "  \"type\": \"" << getTypeString() << "\",\n";
  os << ind << "  \"parentStructure\": " << (parentStructure ? std::to_string(parentStructure->getId()) : "null") << ",\n";

  // 前驱
  os << ind << "  \"predecessors\": [";
  for (size_t i = 0; i < predecessors.size(); ++i) {
    if (i > 0) os << ", ";
    os << predecessors[i]->getId();
  }
  os << "],\n";

  // 后继
  os << ind << "  \"successors\": [";
  for (size_t i = 0; i < successors.size(); ++i) {
    if (i > 0) os << ", ";
    os << successors[i]->getId();
  }
  os << "],\n";

  // 指令
  os << ind << "  \"instructions\": [\n";
  for (size_t i = 0; i < instructions.size(); ++i) {
    os << ind << "    {\n";
    os << ind << "      \"id\": " << instructions[i]->getId() << ",\n";
    // 转义字符串用于 JSON
    std::string instStr = instructions[i]->getAsString();
    // 简单的 JSON 字符串转义
    std::string escaped;
    for (char c : instStr) {
      if (c == '"') escaped += "\\\"";
      else if (c == '\\') escaped += "\\\\";
      else if (c == '\n') escaped += "\\n";
      else if (c == '\r') escaped += "\\r";
      else if (c == '\t') escaped += "\\t";
      else if ((unsigned char)c < 0x20) {
        char buf[8];
        snprintf(buf, sizeof(buf), "\\u%04x", c);
        escaped += buf;
      } else {
        escaped += c;
      }
    }
    os << ind << "      \"operation\": \"" << escaped << "\"\n";
    os << ind << "    }";
    if (i < instructions.size() - 1) os << ",";
    os << "\n";
  }
  os << ind << "  ]\n";
  os << ind << "}";
}

//===----------------------------------------------------------------------===//
// ControlFlowGraph
//===----------------------------------------------------------------------===//

ControlFlowGraph::ControlFlowGraph(triton::FuncOp func) : function(func) {}

ControlFlowGraph::~ControlFlowGraph() = default;

BasicBlock *ControlFlowGraph::createBasicBlock(BlockType type, BasicBlock *parentStructure) {
  auto bb = std::make_unique<BasicBlock>(nextBlockId++, type, parentStructure);
  BasicBlock *bbPtr = bb.get();
  basicBlocks.push_back(std::move(bb));

  // 自动设置入口/出口块
  if (type == BlockType::ENTRY) entryBlock = bbPtr;
  if (type == BlockType::EXIT) exitBlock = bbPtr;

  return bbPtr;
}

void ControlFlowGraph::addEdge(BasicBlock *from, BasicBlock *to) {
  if (!from || !to) return;
  from->addSuccessor(to);
}

void ControlFlowGraph::traverse(BlockVisitor visitor) {
  for (auto &bb : basicBlocks) {
    visitor(*bb);
  }
}

void ControlFlowGraph::print(raw_ostream &os) const {
  auto funcName = const_cast<triton::FuncOp&>(function).getName();
  os << "=================================================================\n";
  os << "Control Flow Graph for function '" << (funcName.empty() ? "unnamed" : funcName) << "'\n";
  os << "Number of blocks: " << basicBlocks.size() << "\n";
  os << "=================================================================\n\n";

  for (const auto &bb : basicBlocks) {
    bb->print(os);
  }
}

void ControlFlowGraph::dump() const {
  print(llvm::errs());
}

void ControlFlowGraph::exportDOT(raw_ostream &os) const {
  auto funcName = const_cast<triton::FuncOp&>(function).getName();
  std::string funcNameStr = funcName.empty() ? "unnamed" : funcName.str();

  // 清理函数名用于 DOT 标识符
  std::string cleanFuncName;
  for (char c : funcNameStr) {
    if (isalnum(c) || c == '_') cleanFuncName += c;
    else cleanFuncName += '_';
  }

  os << "digraph CFG_" << cleanFuncName << " {\n";
  os << "  label=\"CFG for " << funcNameStr << "\";\n";
  os << "  labelloc=t;\n";
  os << "  rankdir=TB;\n\n";

  // 设置节点样式
  for (const auto &bb : basicBlocks) {
    os << "  \"" << bb->getName() << "\" [";
    os << "label=\"" << bb->getName() << "\\n";
    os << "(" << bb->getTypeString() << ")\\n";

    // 添加指令摘要（最多3条）
    size_t numInsts = bb->getNumInstructions();
    if (numInsts > 0) {
      os << "\\n";
      for (size_t i = 0; i < std::min(numInsts, (size_t)3); ++i) {
        std::string instStr = bb->getInstruction(i)->getAsString();
        // 截断并转义
        if (instStr.length() > 40) instStr = instStr.substr(0, 37) + "...";
        // 替换特殊字符
        std::string escaped;
        for (char c : instStr) {
          if (c == '"') escaped += "\\\"";
          else if (c == '\\') escaped += "\\\\";
          else if (c == '\n') escaped += "\\n";
          else escaped += c;
        }
        os << escaped << "\\n";
      }
      if (numInsts > 3) os << "... (" << (numInsts - 3) << " more)\\n";
    }

    os << "\", ";

    // 根据类型设置颜色
    switch (bb->getType()) {
    case BlockType::ENTRY:
      os << "style=filled, fillcolor=lightgreen, shape=ellipse";
      break;
    case BlockType::EXIT:
      os << "style=filled, fillcolor=lightcoral, shape=ellipse";
      break;
    case BlockType::IF_COND:
      os << "style=filled, fillcolor=lightyellow, shape=diamond";
      break;
    case BlockType::FOR_COND:
    case BlockType::WHILE_COND:
      os << "style=filled, fillcolor=lightblue, shape=diamond";
      break;
    case BlockType::LOOP_BODY:
      os << "style=filled, fillcolor=lightcyan, shape=box";
      break;
    case BlockType::LOOP_EXIT:
      os << "style=filled, fillcolor=lightpink, shape=box";
      break;
    default:
      os << "shape=box";
      break;
    }
    os << "];\n";
  }

  os << "\n";

  // 输出边
  for (const auto &bb : basicBlocks) {
    for (auto *succ : bb->getSuccessors()) {
      os << "  \"" << bb->getName() << "\" -> \"" << succ->getName() << "\";\n";
    }
  }

  os << "}\n";
}

llvm::Error ControlFlowGraph::exportToFile(StringRef filename) const {
  std::error_code ec;
  llvm::raw_fd_ostream os(filename, ec, llvm::sys::fs::OF_Text);

  if (ec) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to open file: " + filename);
  }

  // 判断文件扩展名
  if (filename.ends_with(".dot")) {
    exportDOT(os);
  } else if (filename.ends_with(".json")) {
    exportToJSON(os);
  } else {
    print(os);
  }

  os.close();
  return llvm::Error::success();
}

void ControlFlowGraph::exportToJSON(raw_ostream &os) const {
  auto funcName = const_cast<triton::FuncOp&>(function).getName();
  std::string funcNameStr = funcName.empty() ? "unnamed" : funcName.str();

  os << "{\n";
  os << "  \"functionName\": \"" << funcNameStr << "\",\n";
  os << "  \"numBlocks\": " << basicBlocks.size() << ",\n";
  os << "  \"blocks\": [\n";

  for (size_t i = 0; i < basicBlocks.size(); ++i) {
    basicBlocks[i]->exportToJSON(os, 4);
    if (i < basicBlocks.size() - 1) os << ",";
    os << "\n";
  }

  os << "  ]\n";
  os << "}\n";
}

llvm::Error ControlFlowGraph::exportToHTML(StringRef filename) const {
  std::error_code ec;
  llvm::raw_fd_ostream os(filename, ec, llvm::sys::fs::OF_Text);

  if (ec) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to open file: " + filename);
  }

  auto funcName = const_cast<triton::FuncOp&>(function).getName();
  std::string funcNameStr = funcName.empty() ? "unnamed" : funcName.str();

  // 生成嵌入 vis.js 的 HTML 页面
  os << R"html(<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>CFG for )html" << funcNameStr << R"html(</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            height: 100vh;
        }
        #mynetwork {
            flex: 1;
            border: 1px solid lightgray;
            background: #fafafa;
        }
        #sidebar {
            width: 400px;
            border-left: 1px solid #ddd;
            background: white;
            padding: 20px;
            overflow-y: auto;
            box-shadow: -2px 0 5px rgba(0,0,0,0.1);
        }
        h1 {
            margin-top: 0;
            color: #333;
            font-size: 20px;
        }
        h2 {
            color: #666;
            font-size: 16px;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }
        .block-info {
            background: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .instruction {
            background: white;
            border: 1px solid #ddd;
            border-radius: 3px;
            padding: 8px;
            margin: 5px 0;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .instruction-id {
            color: #888;
            font-size: 10px;
        }
        #default-msg {
            color: #999;
            text-align: center;
            margin-top: 50px;
        }
    </style>
</head>
<body>
    <div id="mynetwork"></div>
    <div id="sidebar">
        <h1>CFG: )html" << funcNameStr << R"html(</h1>
        <div id="default-msg">
            <p>Click on a node to view block details</p>
        </div>
        <div id="block-details" style="display:none;">
            <div class="block-info">
                <strong>Block:</strong> <span id="block-name"></span><br>
                <strong>Type:</strong> <span id="block-type"></span><br>
                <strong>ID:</strong> <span id="block-id"></span>
            </div>
            <h2>Predecessors</h2>
            <div id="preds"></div>
            <h2>Successors</h2>
            <div id="succs"></div>
            <h2>Instructions (<span id="inst-count"></span>)</h2>
            <div id="instructions"></div>
        </div>
    </div>

<script type="text/javascript">
    // CFG data
    var cfgData = )html";

  // 输出 JSON 数据
  exportToJSON(os);

  os << R"html(;

    // Create nodes and edges
    var nodes = new vis.DataSet();
    var edges = new vis.DataSet();

    // Color mapping
    var colorMap = {
        'ENTRY': '#90EE90',
        'EXIT': '#F08080',
        'IF_COND': '#FFFACD',
        'FOR_COND': '#ADD8E6',
        'WHILE_COND': '#ADD8E6',
        'LOOP_BODY': '#E0FFFF',
        'LOOP_EXIT': '#FFB6C1',
        'NORMAL': '#FFFFFF'
    };

    // Shape mapping
    var shapeMap = {
        'ENTRY': 'ellipse',
        'EXIT': 'ellipse',
        'IF_COND': 'diamond',
        'FOR_COND': 'diamond',
        'WHILE_COND': 'diamond',
        'LOOP_BODY': 'box',
        'LOOP_EXIT': 'box',
        'NORMAL': 'box'
    };

    // Add nodes
    cfgData.blocks.forEach(function(block) {
        var label = block.name + '\n(' + block.type + ')';
        if (block.instructions.length > 0) {
            label += '\n\n';
            block.instructions.slice(0, 3).forEach(function(inst) {
                var op = inst.operation.substring(0, 40);
                if (inst.operation.length > 40) op += '...';
                label += op + '\n';
            });
            if (block.instructions.length > 3) {
                label += '... (' + (block.instructions.length - 3) + ' more)';
            }
        }

        nodes.add({
            id: block.id,
            label: label,
            color: {
                background: colorMap[block.type] || '#FFFFFF',
                border: '#666666',
                highlight: {
                    background: '#FFD700',
                    border: '#FF4500'
                }
            },
            shape: shapeMap[block.type] || 'box',
            font: {
                face: 'Consolas, Monaco, monospace',
                size: 11
            },
            margin: 10,
            blockData: block
        });
    });

    // Add edges
    cfgData.blocks.forEach(function(block) {
        block.successors.forEach(function(succId) {
            edges.add({
                from: block.id,
                to: succId,
                arrows: 'to',
                color: { color: '#666666' },
                smooth: { type: 'cubicBezier' }
            });
        });
    });

    // Create network
    var container = document.getElementById('mynetwork');
    var data = {
        nodes: nodes,
        edges: edges
    };
    var options = {
        layout: {
            hierarchical: {
                direction: 'UD',
                sortMethod: 'directed',
                levelSeparation: 150,
                nodeSpacing: 200
            }
        },
        physics: {
            enabled: false
        },
        interaction: {
            hover: true,
            selectConnectedEdges: true
        }
    };

    var network = new vis.Network(container, data, options);

    // Click handler
    network.on("click", function (params) {
        if (params.nodes.length > 0) {
            var blockId = params.nodes[0];
            var block = cfgData.blocks.find(b => b.id === blockId);
            showBlockDetails(block);
        }
    });

    function showBlockDetails(block) {
        document.getElementById('default-msg').style.display = 'none';
        document.getElementById('block-details').style.display = 'block';

        document.getElementById('block-name').textContent = block.name;
        document.getElementById('block-type').textContent = block.type;
        document.getElementById('block-id').textContent = block.id;

        // Predecessors
        var predsHtml = '';
        if (block.predecessors.length === 0) {
            predsHtml = '<em>None</em>';
        } else {
            block.predecessors.forEach(function(predId) {
                var pred = cfgData.blocks.find(b => b.id === predId);
                predsHtml += '<div>→ ' + (pred ? pred.name : 'BB' + predId) + '</div>';
            });
        }
        document.getElementById('preds').innerHTML = predsHtml;

        // Successors
        var succsHtml = '';
        if (block.successors.length === 0) {
            succsHtml = '<em>None</em>';
        } else {
            block.successors.forEach(function(succId) {
                var succ = cfgData.blocks.find(b => b.id === succId);
                succsHtml += '<div>→ ' + (succ ? succ.name : 'BB' + succId) + '</div>';
            });
        }
        document.getElementById('succs').innerHTML = succsHtml;

        // Instructions
        document.getElementById('inst-count').textContent = block.instructions.length;
        var instHtml = '';
        block.instructions.forEach(function(inst, idx) {
            instHtml += '<div class="instruction">';
            instHtml += '<span class="instruction-id">[' + inst.id + ']</span> ';
            instHtml += escapeHtml(inst.operation);
            instHtml += '</div>';
        });
        document.getElementById('instructions').innerHTML = instHtml;
    }

    function escapeHtml(text) {
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
</script>
</body>
</html>)html";

  os.close();
  return llvm::Error::success();
}
