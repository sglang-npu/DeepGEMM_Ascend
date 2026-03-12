# Triton Graph-Based Compilation Optimization Framework (C++ Implementation)

## 1. 架构概述

### 1.1 设计理念

本框架基于MLIR C++ API实现，直接在IR层面构建图表示，避免了Python与C++之间的数据转换开销。核心思想是通过分析TTIR中的cube和vector操作依赖关系，将它们从串行执行转换为流水线并行执行。

### 1.2 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input: TTIR Module (MLIR)                   │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              MLIR Pass (triton-ascend-pipeline-opt)            │
│  - Entry point for the optimization                             │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Phase 1: 控制流分析                                │
│  - TTIR Parser: 提取并分类操作                                  │
│  - Control Flow Graph Builder: 构建CFG和ICFG                    │
└──────────────┬──────────────────┬──────────────────┬─────────────┘
               ↓                  ↓                  ↓
      ┌────────┴────────┐  ┌─────┴──────┐  ┌──────┴─────┐
      │  Pointer Analysis │  │ Memory SSA │  │ Sparse VFG │
      │  - 识别Memory Object │  │  Builder   │  │  Builder   │
      │  - 分析指针别名    │  └────────────┘  └────────────┘
      └──────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Phase 2: 依赖分析                                  │
│  基于稀疏VFG进行分析：                                          │
│  - 第一类依赖: Tensor/Buffer (通过Memory SSA)                   │
│  - 第二类依赖: 普通SSA值 (通过use-def链)                         │
│  - 识别cube-vector pipeline机会                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Phase 3: Pipeline调度                              │
│  - List scheduling重排序操作                                    │
│  - 插入sync primitives                                           │
│  - Multi-buffer优化                                              │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              TTIR Generator                                     │
│  - 生成优化后的TTIR                                             │
│  - 插入同步和buffer管理                                        │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Output: Optimized TTIR Module                      │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 两类数据依赖的显式建模

本框架的核心创新是**显式区分并建模两类数据依赖**：

#### **第一类：Tensor/Buffer 内存依赖（隐式依赖）**

这类依赖涉及通过指针访问的内存对象（tensor、buffer），依赖关系不是通过SSA值直接表达，而是通过**内存访问模式**确定。

**示例**:
```mlir
%tensor = tt.load %ptr1           // 加载tensor（创建一个SSA值）
tt.store %ptr2, %tensor            // 将tensor写入另一个内存位置
%result = tt.load %ptr2            // 稍后从相同位置加载
```

**依赖关系**:
```
store %ptr2 -> [内存对象: buffer2]  // 生成memory def (版本+1)
load %ptr2  -> [内存对象: buffer2]  // 使用memory def，建立依赖
```

**实现机制**:
1. **内存对象识别**: 所有kernel参数和`tt.alloc`分配的buffer都标记为`MemoryObject`
2. **指针分析**: 跟踪`tt.addptr`等指针运算，确定load/store访问哪个`MemoryObject`
3. **Memory SSA**: 每个store生成唯一的版本号(def)，后续load引用该版本
4. **Sparse VFG**: 通过def-use边连接store和load，即使它们在文中相距很远

#### **第二类：普通SSA值依赖（显式依赖）**

这类依赖是MLIR SSA形式直接表达的，通过值的use-def链可见。

**示例**:
```mlir
%a = arith.constant 1.0 : f32
%b = arith.addf %a, %c : f32
```

**依赖关系**:
```
constant -> %a -> addf -> %b
```

**实现机制**: 直接使用MLIR的`Value::getUsers()`和`Value::getDefiningOp()`遍历

#### 两类依赖在VFG中的表示

```
稀疏值流图 (Sparse VFG):

第一类依赖（内存）:          第二类依赖（SSA）:
┌─────────┐                  ┌──────────┐
│ StoreOp │──memory def─┐    │ Constant │──┐
└─────────┘              │    └──────────┘  │
                         ↓                  ↓
                    ┌────────┐          ┌────────┐
                    │ Memory │def-use   │  Addf  │
                    │  Phi   │───────→  │        │
                    └────────┘          └────────┘
                         ↑                  ↑
┌─────────┐              │                  │
│ LoadOp  │──memory use──┘                  │
└─────────┘                                 │
    ↑                                       │
    └───────────────────────────────────────┘
              control dependence
```

### 1.4 核心优势

1. **原生性能**: C++实现，无Python-C++序列化开销
2. **MLIR集成**: 直接使用MLIR API，与TTIR互操作无缝
3. **Pass架构**: 作为标准MLIR Pass集成，符合编译器架构
4. **类型安全**: 编译时类型检查，减少运行时错误
5. **可扩展性**: 模块化设计，易于添加新分析和优化

---

## 2. 核心数据结构设计

### 2.1 节点和边的设计

```cpp
// File: third_party/ascend/include/TritonGraph/IR/TritonGraph.h

namespace mlir {
namespace triton {
namespace ascend {

// 节点ID类型
using NodeId = uint32_t;

// 边的类型
enum class EdgeType : uint8_t {
  DATAFLOW    = 0x01,  // 数据流依赖 (def-use链)
  CONTROL     = 0x02,  // 控制流依赖
  PIPELINE    = 0x04,  // 可pipeline的数据依赖
  SYNC        = 0x08,  // 同步依赖
};

// 节点类型
enum class NodeType : uint8_t {
  OPERATION,    // MLIR Operation
  VALUE,        // SSA Value (中间结果)
  BASIC_BLOCK,  // 基本块 (用于控制流分析)
};

// 操作类型 (cube/vector分类)
enum class OpType : uint8_t {
  UNKNOWN,      // 未知或未分类
  CUBE,         // AI Core操作 (矩阵乘等)
  VECTOR,       // Vector Core操作 (elementwise)
  MEMORY,       // 内存操作 (load/store)
  SYNC,         // 同步操作
  CONTROL,      // 控制流
};

// 依赖类型
enum class DependencyType : uint8_t {
  TRUE_DEPENDENCY,     // RAW (Read-After-Write)
  ANTI_DEPENDENCY,     // WAR (Write-After-Read)
  OUTPUT_DEPENDENCY,   // WAW (Write-After-Write)
  CONTROL_DEPENDENCY,  // 控制依赖
};

// 节点属性 (存储在MLIR Attribute中)
struct NodeAttrs : public AttributeStorage {
  NodeAttrs(
    NodeType type,
    OpType opType,
    StringRef name,
    ArrayRef<Value> operands,
    ArrayRef<Value> results,
    Location location,
    DictionaryAttr attributes
  ) : type(type), opType(opType), name(name),
      operands(operands.vec()), results(results.vec()),
      location(location), attributes(attributes) {}

  NodeType type;
  OpType opType;
  StringRef name;
  SmallVector<Value, 4> operands;
  SmallVector<Value, 4> results;
  Location location;
  DictionaryAttr attributes;
};

// 边属性
struct EdgeAttrs {
  EdgeType type;
  DependencyType depType;
  SmallVector<Value, 2> values;  // 传递的值
  unsigned latency;  // 依赖的延迟 (用于调度)
};

// 主图类 (基于MLIR的GraphRegion)
class TritonGraph : public llvm::Graph<TritonGraph*> {
public:
  // 构造函数
  explicit TritonGraph(MLIRContext* context)
      : context(context) {}

  // 添加节点 (包装MLIR Operation)
  NodeId addOperation(Operation* op, OpType opType);

  // 添加节点 (包装SSA Value)
  NodeId addValue(Value value);

  // 添加边
  void addEdge(NodeId from, NodeId to, EdgeAttrs attrs);

  // 查询节点
  Node* getNode(NodeId id) { return nodes.lookup(id); }
  NodeId getId(Node* node) { return nodeIds.lookup(node); }

  // 根据类型获取节点
  SmallVector<Node*> getNodesByType(OpType type);

  // 获取依赖关系
  SmallVector<Node*> getPredecessors(NodeId node);
  SmallVector<Node*> getSuccessors(NodeId node);

  // 获取边的属性
  EdgeAttrs getEdgeAttrs(NodeId from, NodeId to);

  // 图转换
  void contractNodes(SmallVector<NodeId> nodes, NodeId newNode);
  void expandNode(NodeId node, SmallVector<NodeId> newNodes);

  // 可视化
  void dumpToDot(StringRef filename);

  // 遍历所有节点
  using iterator = llvm::DenseMap<NodeId, Node*>::iterator;
  iterator begin() { return nodes.begin(); }
  iterator end() { return nodes.end(); }

private:
  MLIRContext* context;

  // 节点存储
  llvm::DenseMap<NodeId, Node*> nodes;
  llvm::DenseMap<Node*, NodeId> nodeIds;
  NodeId nextId = 0;

  // 边存储 (邻接表)
  llvm::DenseMap<NodeId, SmallVector<Edge>> adjacencyList;
  llvm::DenseMap<std::pair<NodeId, NodeId>, EdgeAttrs> edgeAttrs;
};

} // namespace ascend
} // namespace triton
} // namespace mlir
```

### 2.2 操作分类规则

```cpp
// third_party/ascend/lib/TritonGraph/Utils/OpClassifier.cpp

OpType classifyOperation(Operation* op) {
  // 1. Cube操作 (运行在AI Core)
  if (isa<triton::DotOp>(op) ||
      op->hasTrait<OpTrait::DotLike>()) {
    return OpType::CUBE;
  }

  // Ascend特定的cube接口
  if (auto attr = op->getAttrOfType<StringAttr>("triton.ascend.execution_unit")) {
    if (attr.getValue() == "cube") {
      return OpType::CUBE;
    }
  }

  // 2. Vector操作 (运行在Vector Core)
  if (op->hasTrait<OpTrait::Elementwise>() ||
      op->hasTrait<OpTrait::Broadcastable>()) {
    // Elementwise操作 (add, mul, exp等)
    return OpType::VECTOR;
  }

  // Arith Dialect的操作通常是vectorizable
  if (isa<arith::ArithDialect>(op)) {
    return OpType::VECTOR;
  }

  // Math Dialect (sqrt, exp, log等)
  if (isa<math::MathDialect>(op)) {
    return OpType::VECTOR;
  }

  // 3. 内存操作
  if (isa<triton::LoadOp>(op) ||
      isa<triton::StoreOp>(op) ||
      isa<triton::AdvanceOp>(op)) {
    return OpType::MEMORY;
  }

  // 4. 同步操作
  if (isa<triton::SyncBlockWaitOp>(op) ||
      isa<triton::SyncBlockSetOp>(op)) {
    return OpType::SYNC;
  }

  // 5. 控制流
  if (isa<scf::ForOp>(op) ||
      isa<scf::IfOp>(op) ||
      isa<scf::YieldOp>(op)) {
    return OpType::CONTROL;
  }

  return OpType::UNKNOWN;
}
```

---

## 3. 核心组件详细设计

### 3.1 TTIR Parser

Responsible for extracting operations and their dependencies from TTIR.

```cpp
// third_party/ascend/include/TritonGraph/Parser/TTIRParser.h

class TTIRParser {
public:
  TTIRParser(ModuleOp module, MLIRContext* context);

  // 解析模块中的所有函数
  LogicalResult parseModule(TritonGraph& graph);

  // 解析单个函数
  LogicalResult parseFunction(triton::FuncOp func, TritonGraph& graph);

  // 解析基本块
  LogicalResult parseBasicBlock(Block* block, TritonGraph& graph);

  // 解析操作并添加到图中
  LogicalResult parseOperation(Operation* op, TritonGraph& graph);

  // 获取解析统计信息
  struct ParseStats {
    size_t numOperations = 0;
    size_t numCubeOps = 0;
    size_t numVectorOps = 0;
    size_t numMemoryOps = 0;
    size_t numEdges = 0;
  };
  ParseStats getStats() const { return stats; }

private:
  ModuleOp module;
  MLIRContext* context;
  ParseStats stats;

  // 操作的use-def链分析
  void analyzeUseDefChains(Operation* op, TritonGraph& graph);

  // 检测循环带有的依赖
  bool detectLoopCarriedDependency(Operation* op, LoopLikeOpInterface loop);

  // 提取常量信息
  void extractConstants(Operation* op, TritonGraph& graph);
};
```

**实现细节**:
- 遍历模块中的所有函数
- 对每个函数遍历基本块
- 对每个操作调用 parseOperation
- 在 parseOperation 中:
  1. 创建节点并分类操作类型
  2. 创建值节点 (SSA value)
  3. 添加use-def边
  4. 记录统计信息

### 3.2 Sparse Dataflow Analysis and Memory SSA

**基于CFG的稀疏数据流分析，构建Memory SSA和值流图（VFG）**。

#### 3.2.1 核心概念

在传统的SSA中，每个变量只有一个定义点。但对于内存操作（load/store），我们需要**Memory SSA**来追踪内存位置的定义-使用链。

**Memory Object**: 表示一个内存区域，通常是tensor或buffer
- Kernel参数传入的tensor
- `tt.alloc`分配的内部buffer
- 所有通过指针运算（tt.addptr）访问的内存区域

**Memory SSA**: 为每个内存对象维护一个定义计数
- 每个store操作生成一个新的memory definition
- 后续的load操作依赖于该定义
- 通过phi节点合并来自不同路径的定义

#### 3.2.2 核心数据结构

```cpp
// third_party/ascend/include/TritonGraph/Analysis/MemoryObject.h

namespace ascend::triton {

// 内存对象（Memory Object）表示一个可访问的内存区域
class MemoryObject {
public:
  enum class Type {
    KERNEL_ARG,      // 内核参数（外部传入的tensor）
    ALLOCATED,       // 内部分配的buffer (tt.alloc)
    SLICE_VIEW,      // 切片视图 (tt.make_tensor_ptr)
    UNKNOWN          // 无法确定来源的指针
  };

  MemoryObject(Type type, Type elementType, ArrayRef<int64_t> shape);

  size_t getId() const { return id; }
  Type getType() const { return type; }

  // 获取内存对象的标识字符串
  std::string getName() const;

  // 维度信息
  ArrayRef<int64_t> getShape() const { return shape; }
  Type getElementType() const { return elementType; }

  // 记录所有访问此内存对象的操作
  void addAccess(Operation* op, bool isWrite);
  const SmallVector<Operation*>& getReadOps() const { return readOps; }
  const SmallVector<Operation*>& getWriteOps() const { return writeOps; }

  // Memory SSA: 管理定义计数
  unsigned getCurrentVersion() const { return currentVersion; }
  unsigned allocateNewVersion() { return ++currentVersion; }

  // 别名分析：检查是否与另一个内存对象可能重叠
  bool mayAlias(const MemoryObject* other) const;

  void print(raw_ostream& os) const;

private:
  size_t id;                          // 唯一ID
  Type type;                          // 内存对象类型
  Type elementType;                   // 元素类型
  SmallVector<int64_t> shape;         // 形状

  SmallVector<Operation*> readOps;    // 读取操作
  SmallVector<Operation*> writeOps;   // 写入操作

  unsigned currentVersion = 0;        // 当前版本号（用于Memory SSA）
};

// 内存访问上下文（用于指针分析）
struct PointerInfo {
  MemoryObject* baseObject = nullptr; // 基址对象
  Value offsetValue;                   // 偏移量（可以是动态值或常量）
  bool isDynamicOffset;                // 是否动态偏移

  // 是否精确指向一个对象（无别名）
  bool isPrecise() const { return baseObject != nullptr && !isDynamicOffset; }

  // 是否可能访问此对象
  bool mayAccess(MemoryObject* obj) const;
};

// 内存SSA定义点
struct MemoryDef {
  unsigned version;      // 版本号（内存对象的计数）
  Operation* defOp;      // 定义操作（store）
  MemoryObject* memoryObject; // 操作的内存对象
  BasicBlock* block;     // 定义所在的基本块

  // 如果是phi节点
  bool isPhi = false;
  SmallVector<MemoryDef*> phiArgs; // phi的参数（来自前驱的定义）
};

// 内存SSA使用点
struct MemoryUse {
  Operation* useOp;      // 使用操作（load）
  MemoryObject* memoryObject; // 使用的内存对象
  const MemoryDef* reachingDef; // 到达定义（通过数据流分析得到）

  // SSA值的使用（传统依赖）
  SmallVector<Value> usedSSAValues;
};

// 基于CFG的值流图节点
struct VFGNode {
  enum class Kind {
    // 操作节点
    OPERATION,     // 普通操作

    // 内存相关
    MEMORY_PHI,    // 内存phi节点（合并不同路径的内存状态）
    MEMORY_DEF,    // 内存定义节点（store操作）
    MEMORY_USE,    // 内存使用节点（load操作）

    // 控制流相关
    CONTROL_PHI,   // 控制phi节点（合并控制依赖）
    REGION_ENTRY,  // 函数/区域入口
    REGION_EXIT    // 函数/区域出口
  };

  Kind kind;
  Operation* op = nullptr;          // 对应的MLIR操作
  BasicBlock* block = nullptr;      // 所在基本块
  VFGNode* next = nullptr;          // 值流图中的下一个节点

  union {
    struct {
      MemoryObject* memoryObject;
      unsigned version;
    } memoryInfo;

    struct {
      SmallVector<VFGNode*> predecessors;
      SmallVector<VFGNode*> successors;
    } flowInfo;
  };
};

// 稀疏值流图 (Sparse Value Flow Graph)
class SparseValueFlowGraph {
public:
  explicit SparseValueFlowGraph(const ControlFlowGraph& cfg);
  ~SparseValueFlowGraph();

  // 节点管理
  VFGNode* addOperationNode(Operation* op, BasicBlock* block);
  VFGNode* addMemoryPhiNode(BasicBlock* block, MemoryObject* obj);
  VFGNode* addMemoryDefNode(Operation* storeOp, MemoryObject* obj, unsigned version);
  VFGNode* addMemoryUseNode(Operation* loadOp, MemoryObject* obj, const MemoryDef* def);

  // 边管理
  void addDefUseEdge(VFGNode* from, VFGNode* to);
  void addMemoryDefUseEdge(MemoryDef* def, MemoryUse* use);
  void addPhiEdge(VFGNode* phi, VFGNode* arg);

  // 遍历接口
  using NodeVisitor = llvm::function_ref<bool(VFGNode*)>;
  void traverseDefUseChains(NodeVisitor visitor);

  // 查询接口
  SmallVector<VFGNode*> getMemoryDefs(MemoryObject* obj) const;
  SmallVector<VFGNode*> getMemoryUses(MemoryObject* obj) const;
  const MemoryDef* getReachingDef(Operation* loadOp, MemoryObject* obj) const;

  void print(raw_ostream& os) const;

private:
  const ControlFlowGraph& cfg;
  SmallVector<std::unique_ptr<VFGNode>> nodes;

  // 内存对象到其定义的映射
  DenseMap<MemoryObject*, SmallVector<MemoryDef*>> memoryDefs;
  DenseMap<Operation*, SmallVector<MemoryUse*>> operationMemoryUses;

  // 值流边
  DenseMap<VFGNode*, SmallVector<VFGNode*>> defUseEdges;
};

} // namespace ascend::triton
```

#### 3.2.3 指针分析

```cpp
// third_party/ascend/include/TritonGraph/Analysis/PointerAnalysis.h

// 指针分析：将指针表达式映射到内存对象
class PointerAnalysis {
public:
  explicit PointerAnalysis(const ControlFlowGraph& cfg);

  // 分析模块中的所有指针操作
  LogicalResult analyze(ModuleOp module);

  // 查询指针信息
  PointerInfo getPointerInfo(Value pointer) const;

  // 获取指针的基址对象
  MemoryObject* getBaseObject(Value pointer) const;

  // 指针别名查询
  bool mayAlias(Value ptr1, Value ptr2) const;
  bool mayAlias(Value pointer, MemoryObject* obj) const;

  // 获取所有可能访问的内存对象
  SmallVector<MemoryObject*> getMayAccessedObjects(Value pointer) const;

private:
  const ControlFlowGraph& cfg;
  DenseMap<Value, PointerInfo> pointerInfoMap;

  // 分析不同类型的指针
  void analyzeKernelArg(Value arg);
  void analyzeAllocOp(triton::AllocOp alloc);
  void analyzeMakeTensorPtrOp(triton::MakeTensorPtrOp makePtr);
  void analyzeAddPtrOp(triton::AddPtrOp addPtr);

  // 从addptr回溯到基址
  Value traceToBasePointer(Value ptr) const;
};

// 指针分析算法实现
LogicalResult PointerAnalysis::analyze(ModuleOp module) {
  // 1. 识别所有内存对象
  module.walk([&](Operation* op) {
    TypeSwitch<Operation*>(op)
        .Case<triton::AllocOp>([&](auto alloc) { analyzeAllocOp(alloc); })
        .Case<triton::MakeTensorPtrOp>([&](auto makePtr) {
          analyzeMakeTensorPtrOp(makePtr);
        })
        .Default([](auto) {});
  });

  // 2. 分析指针运算 (tt.addptr)
  module.walk([&](triton::AddPtrOp addPtr) { analyzeAddPtrOp(addPtr); });

  // 3. 记录kernel参数
  module.walk([&](triton::FuncOp func) {
    if (func.isPublic()) { // 内核入口
      for (auto arg : func.getArguments()) {
        if (arg.getType().isa<triton::PointerType>()) {
          analyzeKernelArg(arg);
        }
      }
    }
  });

  return success();
}

void PointerAnalysis::analyzeAddPtrOp(triton::AddPtrOp addPtr) {
  Value ptr = addPtr.getPointer();
  Value offset = addPtr.getOffset();

  // 回溯到基址
  Value basePtr = traceToBasePointer(ptr);

  // 检查是否是常量偏移
  bool isDynamicOffset = true;
  if (auto constOp = offset.getDefiningOp<arith::ConstantOp>()) {
    isDynamicOffset = false;
  }

  // 建立指针信息
  PointerInfo info;
  info.baseObject = getBaseObject(basePtr);
  info.offsetValue = offset;
  info.isDynamicOffset = isDynamicOffset;

  pointerInfoMap[addPtr.getResult()] = info;
}

Value PointerAnalysis::traceToBasePointer(Value ptr) const {
  // 回溯指针链，直到找到非addptr的结果
  while (auto addPtr = ptr.getDefiningOp<triton::AddPtrOp>()) {
    ptr = addPtr.getPointer();
  }
  return ptr;
}
```

#### 3.2.4 Memory SSA 构建

```cpp
// third_party/ascend/include/TritonGraph/Analysis/MemorySSABuilder.h

// Memory SSA构建器
class MemorySSABuilder {
public:
  MemorySSABuilder(const ControlFlowGraph& cfg,
                   PointerAnalysis& ptrAnalysis);

  // 为函数构建Memory SSA
  LogicalResult build(triton::FuncOp func);

  // 获取指定内存对象在基本块入口的定义
  const MemoryDef* getMemoryDefAtEntry(BasicBlock* block,
                                       MemoryObject* obj) const;

  // 获取指定内存对象在操作前的定义
  const MemoryDef* getMemoryDefBefore(Operation* op,
                                      MemoryObject* obj) const;

  // 获取load操作对应的MemoryUse
  const MemoryUse* getMemoryUse(Operation* loadOp) const;

  // 获取store操作对应的MemoryDef
  const MemoryDef* getMemoryDef(Operation* storeOp) const;

  void print(raw_ostream& os) const;

private:
  const ControlFlowGraph& cfg;
  PointerAnalysis& ptrAnalysis;

  // 内存定义和使用点
  DenseMap<std::pair<MemoryObject*, Operation*>, std::unique_ptr<MemoryDef>>
      defs;
  DenseMap<Operation*, std::unique_ptr<MemoryUse>> uses;

  // 基本块入口的内存状态 (phi的输入)
  DenseMap<std::pair<BasicBlock*, MemoryObject*>, SmallVector<const MemoryDef*>>
      blockEntryDefs;

  // 构建算法
  void buildMemoryDefsAndUses(triton::FuncOp func);
  void insertMemoryPhis(triton::FuncOp func);
  void renameMemoryDefs(triton::FuncOp func);

  // 在基本块边插入phi
  void computePhiArgs(triton::FuncOp func);
};

// Memory SSA构建算法
LogicalResult MemorySSABuilder::build(triton::FuncOp func) {
  // 阶段1：识别所有内存定义和使用
  buildMemoryDefsAndUses(func);

  // 阶段2：在合并点插入phi节点
  insertMemoryPhis(func);

  // 阶段3：重命名定义，分配版本号
  renameMemoryDefs(func);

  return success();
}

void MemorySSABuilder::buildMemoryDefsAndUses(triton::FuncOp func) {
  // 遍历函数中的所有操作
  func.walk([&](Operation* op) {
    TypeSwitch<Operation*>(op)
        .Case<triton::LoadOp>([&](auto loadOp) {
          // 分析load的指针
          Value ptr = loadOp.getPtr();
          MemoryObject* obj = ptrAnalysis.getBaseObject(ptr);

          if (obj) {
            // 创建MemoryUse
            auto use = std::make_unique<MemoryUse>();
            use->useOp = loadOp;
            use->memoryObject = obj;
            uses[loadOp] = std::move(use);

            // 记录访问
            obj->addAccess(loadOp, /*isWrite=*/false);
          }
        })
        .Case<triton::StoreOp>([&](auto storeOp) {
          // 分析store的指针
          Value ptr = storeOp.getPtr();
          MemoryObject* obj = ptrAnalysis.getBaseObject(ptr);

          if (obj) {
            // 创建MemoryDef
            auto def = std::make_unique<MemoryDef>();
            def->defOp = storeOp;
            def->memoryObject = obj;
            def->block = storeOp->getBlock();
            def->version = obj->allocateNewVersion(); // 分配新版本

            defs[{obj, storeOp}] = std::move(def);

            // 记录访问
            obj->addAccess(storeOp, /*isWrite=*/true);
          }
        })
        .Default([](auto) {});
  });
}

void MemorySSABuilder::insertMemoryPhis(triton::FuncOp func) {
  // 在控制流合并点插入phi节点
  // 算法：遍历所有基本块，如果多个前驱对同一内存对象有不同的定义，则插入phi

  const ControlFlowGraph* cfgPtr = &cfg;
  auto& cfgRef = *cfgPtr;

  // 为每个基本块和内存对象创建phi
  for (auto& block : func.getBody()) {
    size_t blockId = cfgRef.findBasicBlock(&block)->id;
    const auto& blockNode = cfgRef.getBasicBlock(blockId);

    // 如果有多个前驱，可能需要phi
    if (blockNode.preds.size() > 1) {
      // 收集所有前驱定义的不同内存对象
      DenseSet<MemoryObject*> objectsInPreds;

      for (size_t predId : blockNode.preds) {
        BasicBlock* predBlock = cfgRef.getBasicBlock(predId).mlirBlock;

        // 获取该前驱的所有内存定义
        predBlock->walk([&](Operation* op) {
          if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
            MemoryObject* obj = ptrAnalysis.getBaseObject(storeOp.getPtr());
            if (obj) {
              objectsInPreds.insert(obj);
            }
          }
        });
      }

      // 为这些内存对象创建phi
      for (MemoryObject* obj : objectsInPreds) {
        // 检查前驱是否有不同的定义版本
        SmallVector<const MemoryDef*> predDefs;
        for (size_t predId : blockNode.preds) {
          if (auto def = getMemoryDefAtEntry(cfgRef.getBasicBlock(predId).mlirBlock, obj)) {
            predDefs.push_back(def);
          }
        }

        // 如果有多个不同的定义，需要phi节点
        if (predDefs.size() > 1) {
          // 创建phi节点 (在Triton IR中用scf.if/scf.for的argument模拟)
          auto phiDef = std::make_unique<MemoryDef>();
          phiDef->isPhi = true;
          phiDef->memoryObject = obj;
          phiDef->block = &block;
          phiDef->version = obj->allocateNewVersion();
          phiDef->phiArgs = predDefs; // 记录phi的参数

          // 将phi节点存储在基本块入口
          blockEntryDefs[{&block, obj}].push_back(phiDef.get());

          // 保存phi定义
          // 注意：phi节点不对应实际的操作，所以key用nullptr或特殊标记
          defs[{obj, nullptr}] = std::move(phiDef);
        }
      }
    }
  }
}

void MemorySSABuilder::renameMemoryDefs(triton::FuncOp func) {
  // 重命名定义，确保版本的唯一性和正确性
  // 使用支配树遍历顺序

  // 这类似于传统的SSA重命名算法
  DenseMap<MemoryObject*, const MemoryDef*> currentDefMap;

  // 深度优先遍历支配树
  std::function<void(BasicBlock*)> rename = [&](BasicBlock* block) {
    // 保存当前状态（用于回溯）
    DenseMap<MemoryObject*, const MemoryDef*> savedDefs = currentDefMap;

    // 处理块内的phi节点（如果有）
    if (auto it = blockEntryDefs.find({block, nullptr}); it != blockEntryDefs.end()) {
      for (const MemoryDef* phiDef : it->second) {
        currentDefMap[phiDef->memoryObject] = phiDef;
      }
    }

    // 处理块内的所有store操作
    for (auto& op : *block) {
      if (isa<triton::StoreOp>(op)) {
        MemoryObject* obj = ptrAnalysis.getBaseObject(
            cast<triton::StoreOp>(op).getPtr());
        if (obj && defs.count({obj, &op})) {
          currentDefMap[obj] = defs[{obj, &op}].get();
        }
      }

      // 为load设置到达定义
      if (isa<triton::LoadOp>(op)) {
        MemoryObject* obj = ptrAnalysis.getBaseObject(
            cast<triton::LoadOp>(op).getPtr());
        if (obj && uses.count(&op)) {
          uses[&op]->reachingDef = currentDefMap[obj];
        }
      }
    }

    // 递归处理后继
    size_t blockId = cfg.findBasicBlock(block)->id;
    for (size_t succId : cfg.getBasicBlock(blockId).succs) {
      BasicBlock* succ = cfg.getBasicBlock(succId).mlirBlock;
      rename(succ);
    }

    // 恢复状态
    currentDefMap = savedDefs;
  };

  // 从入口块开始
  BasicBlock* entry = &func.getBody().front();
  rename(entry);
}
```

#### 3.2.5 稀疏值流图（Sparse VFG）构建

```cpp
// third_party/ascend/include/TritonGraph/Builder/SparseVFGBuilder.h

// 稀疏值流图构建器
class SparseVFGBuilder {
public:
  SparseVFGBuilder(const ControlFlowGraph& cfg,
                  MemorySSABuilder& memorySSA,
                  PointerAnalysis& ptrAnalysis);

  // 构建稀疏值流图
  std::unique_ptr<SparseValueFlowGraph> build(triton::FuncOp func);

private:
  const ControlFlowGraph& cfg;
  MemorySSABuilder& memorySSA;
  PointerAnalysis& ptrAnalysis;

  // 节点创建辅助函数
  VFGNode* createOperationNode(Operation* op,
                               SparseValueFlowGraph& vfg,
                               DenseMap<Operation*, VFGNode*>& opToNode);

  // 为内存操作创建VFG节点
  void createMemoryVFGNodes(SparseValueFlowGraph& vfg);

  // 连接def-use链
  void connectDefUseChains(SparseValueFlowGraph& vfg,
                          DenseMap<Operation*, VFGNode*>& opToNode);

  // 处理第一类依赖：Memory依赖（基于Memory SSA）
  void buildMemoryDependencies(SparseValueFlowGraph& vfg);

  // 处理第二类依赖：普通SSA依赖
  void buildSSADependencies(SparseValueFlowGraph& vfg,
                           DenseMap<Operation*, VFGNode*>& opToNode);
};

// 构建稀疏值流图
std::unique_ptr<SparseValueFlowGraph> SparseVFGBuilder::build(
    triton::FuncOp func) {
  auto vfg = std::make_unique<SparseValueFlowGraph>(cfg);

  DenseMap<Operation*, VFGNode*> opToNode;

  // 阶段1：为所有操作创建基础VFG节点
  func.walk([&](Operation* op) {
    auto node = createOperationNode(op, *vfg, opToNode);
    if (node) {
      opToNode[op] = node;
    }
  });

  // 阶段2：构建Memory依赖（基于Memory SSA）
  buildMemoryDependencies(*vfg);

  // 阶段3：构建普通SSA依赖
  buildSSADependencies(*vfg, opToNode);

  return vfg;
}

void SparseVFGBuilder::buildMemoryDependencies(
    SparseValueFlowGraph& vfg) {
  // 遍历所有函数中的操作
  func.walk([&](Operation* op) {
    TypeSwitch<Operation*>(op)
        .Case<triton::LoadOp>([&](auto loadOp) {
          // Load操作：创建MemoryUse节点
          Value ptr = loadOp.getPtr();
          MemoryObject* obj = ptrAnalysis.getBaseObject(ptr);

          if (obj) {
            // 获取到达定义
            const MemoryUse* memUse = memorySSA.getMemoryUse(loadOp);
            if (memUse && memUse->reachingDef) {
              const MemoryDef* def = memUse->reachingDef;

              // 在VFG中创建MemoryUse节点
              auto useNode = vfg.addMemoryUseNode(loadOp, obj, def);

              // 查找对应的定义节点
              if (auto defNodeIter = opToNode.find(def->defOp);
                  defNodeIter != opToNode.end()) {
                VFGNode* defNode = defNodeIter->second;

                // 添加def-use边：store -> load
                vfg.addMemoryDefUseEdge(
                    const_cast<MemoryDef*>(def), // NOLINT
                    const_cast<MemoryUse*>(memUse)); // NOLINT
              }
            }
          }
        })
        .Case<triton::StoreOp>([&](auto storeOp) {
          // Store操作：创建MemoryDef节点
          Value ptr = storeOp.getPtr();
          MemoryObject* obj = ptrAnalysis.getBaseObject(ptr);

          if (obj) {
            const MemoryDef* memDef = memorySSA.getMemoryDef(storeOp);
            if (memDef) {
              vfg.addMemoryDefNode(storeOp, obj, memDef->version);
            }
          }
        });
  });

  // 添加Memory Phi节点（用于合并不同控制流路径的内存状态）
  for (auto& [key, phiDefs] : memorySSA.getBlockEntryDefs()) {
    BasicBlock* block = key.first;
    MemoryObject* obj = key.second;

    for (const MemoryDef* phiDef : phiDefs) {
      if (phiDef->isPhi) {
        VFGNode* phiNode = vfg.addMemoryPhiNode(block, obj);

        // 连接phi的参数（来自前驱的定义）
        for (const MemoryDef* argDef : phiDef->phiArgs) {
          if (auto argNodeIter = opToNode.find(argDef->defOp);
              argNodeIter != opToNode.end()) {
            vfg.addPhiEdge(phiNode, argNodeIter->second);
          }
        }
      }
    }
  }
}

void SparseVFGBuilder::buildSSADependencies(
    SparseValueFlowGraph& vfg,
    DenseMap<Operation*, VFGNode*>& opToNode) {
  // 构建普通SSA值的def-use链
  for (auto& [op, node] : opToNode) {
    // 1. 为操作的结果创建def边
    for (Value result : op->getResults()) {
      for (Operation* user : result.getUsers()) {
        if (user && opToNode.count(user)) {
          VFGNode* userNode = opToNode[user];
          vfg.addDefUseEdge(node, userNode);
        }
      }
    }

    // 2. 为操作的操作数创建use边
    for (Value operand : op->getOperands()) {
      if (Operation* defOp = operand.getDefiningOp()) {
        if (opToNode.count(defOp)) {
          VFGNode* defNode = opToNode[defOp];
          vfg.addDefUseEdge(defNode, node);
        }
      }
    }
  }
}
```

**关键设计点**:

1. **两类数据依赖明确区分**:
   - **第一类（Tensor/Buffer依赖）**: 通过`MemoryObject`、`MemoryDef`/`MemoryUse`追踪，基于Memory SSA的版本号
   - **第二类（SSA依赖）**: 通过MLIR的SSA use-def链直接追踪

2. **稀疏分析（Sparse Analysis）**:
   - 不在每个程序点存储数据流信息
   - 只在**定义点**和**使用点**创建节点
   - 通过def-use边直接连接相关的节点
   - 复杂度：O(|def| + |use|)，而不是O(|program points| × |variables|)

3. **Memory SSA的优势**:
   - 准确定义内存依赖关系
   - 支持复杂的控制流（循环、条件分支）
   - 为每个内存访问提供精确的到达定义信息
   - 便于实现优化（如store-to-load forwarding、死存储删除）

### 3.3 Control Flow Graph Builder

**设计和实现显式的控制流图（CFG）和过程间控制流图（ICFG）**。

#### 3.3.1 CFG/ICFG 数据结构

```cpp
// third_party/ascend/include/TritonGraph/Graph/ControlFlowGraph.h

namespace ascend::triton {

// 基本块节点 (Basic Block)
struct BasicBlockNode {
  // 基本块的唯一标识符
  size_t id;

  // 对应MLIR中的Block
  Block* mlirBlock = nullptr;

  // 包含的操作序列
  SmallVector<Operation*> operations;

  // 基本块类型
  enum class BlockType {
    ENTRY,      // 函数入口
    EXIT,       // 函数出口
    NORMAL,     // 普通基本块
    CONDITIONAL // 条件分支（对应scf.if的header）
    LOOP_HEADER, // 循环头部
    LOOP_LATCH,  // 循环latch
    LOOP_BODY    // 循环体
  };
  BlockType type = BlockType::NORMAL;

  // 控制流图相关数据
  bool visited = false;        // 遍历标记
  unsigned depth = 0;          // DFS深度
  unsigned postOrder = 0;      // 后序遍历编号
  SmallVector<size_t> preds;   // 前驱基本块ID列表
  SmallVector<size_t> succs;   // 后继基本块ID列表

  // 支配树（Dominator Tree）信息
  size_t immediateDominator = std::numeric_limits<size_t>::max(); // 直接支配节点
  SmallVector<size_t> dominates; // 支配的基本块列表

  // 循环相关
  LoopInfo* parentLoop = nullptr; // 所属的循环

  // 静态分析辅助数据
  void* analysisData = nullptr; // 用于存储分析结果（类型擦除）
};

// 控制流边 (CFG Edge)
struct ControlFlowEdge {
  size_t from;                // 源基本块ID
  size_t to;                  // 目标基本块ID

  // 边类型
  enum class EdgeType {
    DIRECT,          // 直接跳转（无条件）
    TRUE_BRANCH,     // 条件为真的分支
    FALSE_BRANCH,    // 条件为假的分支
    BACK_EDGE,       // 回边（循环）
    CALL_EDGE,       // 函数调用边
    RETURN_EDGE,     // 函数返回边
    EXCEPTION_EDGE   // 异常处理边
  };
  EdgeType type = EdgeType::DIRECT;

  // 条件值（如果是条件分支）
  Value conditionValue;       // branch condition

  // 边的标签（用于可视化）
  std::string label;
};

// 循环信息
struct LoopInfo {
  size_t headerId;            // 循环头基本块ID
  SmallVector<size_t> body;    // 循环体基本块ID列表
  SmallVector<size_t> latches; // latch基本块列表（回到循环头）
  size_t exitId = 0;          // 出口基本块ID（循环结束后）

  // 循环特性
  bool isNaturalLoop = true;  // 是否自然循环（一个入口，多个出口）
  bool isReducible = true;    // 是否可归约
  unsigned nestingLevel = 0;  // 嵌套深度
  LoopInfo* parentLoop = nullptr; // 父循环
  SmallVector<LoopInfo*> childLoops; // 子循环

  // 迭代变量信息（用于循环携带依赖分析）
  struct InductionVar {
    Value inductionVar;       // 迭代变量
    Value stepValue;          // 步长
    Value lowerBound;         // 下限
    Value upperBound;         // 上限
  };
  std::optional<InductionVar> inductionInfo;
};

// 函数级控制流图 (CFG)
class ControlFlowGraph {
public:
  explicit ControlFlowGraph(triton::FuncOp func);
  ~ControlFlowGraph();

  // 获取函数
  triton::FuncOp getFunction() const { return function; }

  // 基本块操作
  size_t addBasicBlock(Block* block, BasicBlockNode::BlockType type =
                       BasicBlockNode::BlockType::NORMAL);

  BasicBlockNode& getBasicBlock(size_t id) { return *basicBlocks[id]; }
  const BasicBlockNode& getBasicBlock(size_t id) const { return *basicBlocks[id]; }

  BasicBlockNode* findBasicBlock(Block* block);
  const BasicBlockNode* findBasicBlock(Block* block) const;

  size_t getNumBlocks() const { return basicBlocks.size(); }

  // 边的操作
  void addEdge(size_t from, size_t to, ControlFlowEdge::EdgeType type =
               ControlFlowEdge::EdgeType::DIRECT, const std::string& label = "");

  const ControlFlowEdge* getEdge(size_t from, size_t to) const;

  const SmallVector<std::unique_ptr<ControlFlowEdge>>& getEdges() const {
    return edges;
  }

  // 循环分析
  LoopInfo* findOrCreateLoop(size_t headerId);
  SmallVector<LoopInfo*> getTopLevelLoops() const;
  LoopInfo* get innerMostLoop(size_t blockId) const;

  // 遍历接口
  using BlockVisitor = llvm::function_ref<bool(BasicBlockNode&)>;
  using ConstBlockVisitor = llvm::function_ref<bool(const BasicBlockNode&)>;

  void traverseDFS(BlockVisitor visitor); // 深度优先遍历
  void traverseReversePostOrder(BlockVisitor visitor); // 逆后序遍历

  // 控制流分析
  void computeDominatorTree(); // 构建支配树
  void computePostDominatorTree(); // 构建后支配树
  void identifyNaturalLoops(); // 识别自然循环
  void analyzeControlFlow(); // 完整的控制流分析

  // 查询接口
  SmallVector<size_t> getEntryBlocks() const;  // 获取入口基本块
  SmallVector<size_t> getExitBlocks() const;   // 获取出口基本块
  bool dominates(size_t dominator, size_t dominated) const; // 支配查询
  bool isBackEdge(size_t from, size_t to) const; // 是否回边

  // 可视化
  void dumpToDot(const std::string& filename) const; // 输出到dot文件
  void print(raw_ostream& os) const;

private:
  triton::FuncOp function; // 对应的MLIR函数

  // 基本块映射
  DenseMap<Block*, size_t> blockToIdMap; // MLIR Block到节点ID的映射
  SmallVector<std::unique_ptr<BasicBlockNode>> basicBlocks;

  // 控制流边
  DenseMap<std::pair<size_t, size_t>, ControlFlowEdge*> edgeMap;
  SmallVector<std::unique_ptr<ControlFlowEdge>> edges;

  // 循环信息
  DenseMap<size_t, std::unique_ptr<LoopInfo>> loops; // headerId -> LoopInfo
  SmallVector<LoopInfo*> topLevelLoops; // 顶层循环

  // 支配树
  std::unique_ptr<DominatorTreeBase<BasicBlockNode>> domTree;
  std::unique_ptr<DominatorTreeBase<BasicBlockNode>> postDomTree;

  // 遍历相关
  SmallVector<size_t> postOrderBlocks; // 后序遍历顺序
  SmallVector<size_t> reversePostOrderBlocks; // 逆后序遍历
};

} // namespace ascend::triton
```

#### 3.3.2 CFG Builder 实现

```cpp
// third_party/ascend/include/TritonGraph/Builder/ControlFlowGraphBuilder.h

class ControlFlowGraphBuilder {
public:
  using CFGPtr = std::unique_ptr<ControlFlowGraph>;

  // 构建单个函数的CFG
  CFGPtr build(triton::FuncOp func);

  // 构建模块的ICFG（过程间控制流图）
  std::unique_ptr<InterProceduralCFG> buildModule(ModuleOp module);

private:
  // 遍历基本块构建CFG
  void buildCFGForRegion(Region& region, ControlFlowGraph& cfg);

  // 处理不同类型的终止操作
  void handleTerminator(Operation* terminator, ControlFlowGraph& cfg);

  // 处理scf.if操作
  void handleIfOp(scf::IfOp ifOp, size_t currentBlock, ControlFlowGraph& cfg);

  // 处理scf.for/while循环
  void handleLoopOp(LoopLikeOpInterface loop, size_t currentBlock,
                    ControlFlowGraph& cfg);

  // 识别循环结构
  void identifyLoops(ControlFlowGraph& cfg);
};
```

**构建算法实现**:
```cpp
std::unique_ptr<ControlFlowGraph> ControlFlowGraphBuilder::build(
    triton::FuncOp func) {
  auto cfg = std::make_unique<ControlFlowGraph>(func);

  // 区域可能有多个基本块
  if (!func.getBody().hasOneBlock()) {
    // 多基本块函数：需要构建完整的CFG
    buildCFGForRegion(func.getBody(), *cfg);
  } else {
    // 单基本块函数：只有入口和出口
    auto& entryBlock = func.getBody().getBlocks().front();
    size_t entryId = cfg->addBasicBlock(&entryBlock,
                                        BasicBlockNode::BlockType::ENTRY);

    // 遍历遍历操作
    for (auto& op : entryBlock) {
      cfg->getBasicBlock(entryId).operations.push_back(&op);
    }
  }

  // 执行控制流分析
  cfg->analyzeControlFlow();

  return cfg;
}

void ControlFlowGraphBuilder::buildCFGForRegion(Region& region,
                                                ControlFlowGraph& cfg) {
  // 1. 为每个Block创建基本块节点
  for (auto& block : region.getBlocks()) {
    size_t blockId = cfg.addBasicBlock(&block,
                                       BasicBlockNode::BlockType::NORMAL);

    // 添加块内的所有操作
    for (auto& op : block) {
      cfg.getBasicBlock(blockId).operations.push_back(&op);
    }
  }

  // 2. 遍历所有Block，构建控制流边
  for (auto& block : region.getBlocks()) {
    size_t fromId = cfg.findBasicBlock(&block)->id;

    // 处理终止操作
    Operation* terminator = block.getTerminator();
    if (!terminator) continue;

    handleTerminator(terminator, cfg);
  }
}

void ControlFlowGraphBuilder::handleTerminator(Operation* terminator,
                                               ControlFlowGraph& cfg) {
  size_t currentBlock = cfg.findBasicBlock(terminator->getBlock())->id;

  if (auto branchOp = dyn_cast<cf::BranchOp>(terminator)) {
    // 无条件跳转
    Block* dest = branchOp.getDest();
    size_t destId = cfg.findBasicBlock(dest)->id;
    cfg.addEdge(currentBlock, destId,
                ControlFlowEdge::EdgeType::DIRECT, "branch");

  } else if (auto condOp = dyn_cast<cf::CondBranchOp>(terminator)) {
    // 条件分支
    Block* trueDest = condOp.getTrueDest();
    Block* falseDest = condOp.getFalseDest();

    size_t trueId = cfg.findBasicBlock(trueDest)->id;
    size_t falseId = cfg.findBasicBlock(falseDest)->id;

    // true分支
    cfg.addEdge(currentBlock, trueId,
                ControlFlowEdge::EdgeType::TRUE_BRANCH, "true");

    // false分支
    cfg.addEdge(currentBlock, falseId,
                ControlFlowEdge::EdgeType::FALSE_BRANCH, "false");

  } else if (auto returnOp = dyn_cast<triton::ReturnOp>(terminator)) {
    // 函数返回
    // 标记当前基本块为出口
    cfg.getBasicBlock(currentBlock).type =
        BasicBlockNode::BlockType::EXIT;
  } else if (isa<scf::YieldOp>(terminator)) {
    // scf.yield是scf结构的一部分，在上层处理
  }
}

void ControlFlowGraphBuilder::handleIfOp(scf::IfOp ifOp, size_t currentBlock,
                                         ControlFlowGraph& cfg) {
  // scf.if 转换为条件分支
  Value condition = ifOp.getCondition();

  // then区域
  Region& thenRegion = ifOp.getThenRegion();
  if (!thenRegion.empty()) {
    Block* thenEntry = &thenRegion.front();
    size_t thenId = cfg.findBasicBlock(thenEntry)->id;

    // 标记为条件基本块
    cfg.getBasicBlock(thenId).type =
        BasicBlockNode::BlockType::CONDITIONAL;

    cfg.addEdge(currentBlock, thenId,
                ControlFlowEdge::EdgeType::TRUE_BRANCH, "then");

    // 递归构建then区域的CFG
    buildCFGForRegion(thenRegion, cfg);
  }

  // else区域（可选）
  Region& elseRegion = ifOp.getElseRegion();
  if (!elseRegion.empty()) {
    Block* elseEntry = &elseRegion.front();
    size_t elseId = cfg.findBasicBlock(elseEntry)->id;

    cfg.addEdge(currentBlock, elseId,
                ControlFlowEdge::EdgeType::FALSE_BRANCH, "else");

    // 递归构建else区域的CFG
    buildCFGForRegion(elseRegion, cfg);
  }
}

void ControlFlowGraphBuilder::identifyLoops(ControlFlowGraph& cfg) {
  // 识别回边 (back edges)
  SmallVector<std::pair<size_t, size_t>> backEdges;

  for (auto& edge : cfg.getEdges()) {
    size_t from = edge->from;
    size_t to = edge->to;

    // DFS序判断回边
    if (cfg.dominates(to, from)) {
      // to支配from，且存在从from到to的边 => 回边
      backEdges.push_back({from, to});
      cfg.addEdge(from, to, ControlFlowEdge::EdgeType::BACK_EDGE);
    }
  }

  // 为每个回边构建自然循环
  for (auto [tail, head] : backEdges) {
    LoopInfo* loop = cfg.findOrCreateLoop(head);

    // 收集循环体中的所有基本块
    SmallVector<size_t> workList = {tail};
    SmallVector<size_t> loopBlocks = {head};

    while (!workList.empty()) {
      size_t block = workList.pop_back_val();

      if (llvm::find(loopBlocks, block) != loopBlocks.end())
        continue;

      loopBlocks.push_back(block);

      // 添加前驱到工作列表
      for (size_t pred : cfg.getBasicBlock(block).preds) {
        if (pred != head) { // 不是循环头
          workList.push_back(pred);
        }
      }
    }

    loop->body = std::move(loopBlocks);
    loop->latches.push_back(tail);

    // 标记循环内的基本块
    for (size_t blockId : loop->body) {
      cfg.getBasicBlock(blockId).parentLoop = loop;
    }
  }
}
```

#### 3.3.3 ICFG (Inter-procedural CFG) 实现

```cpp
// third_party/ascend/include/TritonGraph/Graph/InterProceduralCFG.h

// 过程间控制流图 (ICFG)
class InterProceduralCFG {
public:
  explicit InterProceduralCFG(ModuleOp module);

  // 为每个函数构建CFG
  void build();

  // 获取函数的CFG
  ControlFlowGraph* getFunctionCFG(triton::FuncOp func);
  const ControlFlowGraph* getFunctionCFG(triton::FuncOp func) const;

  // 在调用点连接调用者和被调用者的CFG
  struct CallSite {
    Operation* callOp;          // 调用操作
    triton::FuncOp caller;      // 调用者函数
    triton::FuncOp callee;      // 被调用者函数
    size_t callBlockId;         // 调用点的基本块ID
    size_t returnBlockId;       // 返回后的基本块ID
  };

  // 获取所有调用点
  const SmallVector<CallSite>& getCallSites() const { return callSites; }

  // 连接ICFG中的调用边
  void connectCallGraph();

  // 查询函数调用关系
  SmallVector<triton::FuncOp> getCallees(triton::FuncOp caller) const;
  SmallVector<triton::FuncOp> getCallers(triton::FuncOp callee) const;

  // 全局可达性分析
  void computeReachability();
  bool isReachable(triton::FuncOp from, triton::FuncOp to) const;

  // 可视化
  void dumpToDot(const std::string& filename) const;

private:
  ModuleOp module;

  // 函数到CFG的映射
  DenseMap<triton::FuncOp, std::unique_ptr<ControlFlowGraph>> functionCFGs;

  // 调用点列表
  SmallVector<CallSite> callSites;

  // 调用图 (函数级别)
  DenseMap<triton::FuncOp, SmallVector<triton::FuncOp>> callGraph;
  DenseMap<triton::FuncOp, SmallVector<triton::FuncOp>> reverseCallGraph;

  // 可达性矩阵
  // functionA -> 到达的函数集合
  DenseMap<triton::FuncOp, DenseSet<triton::FuncOp>> reachability;
};
```

**ICFG构建算法**:
```cpp
std::unique_ptr<InterProceduralCFG> ControlFlowGraphBuilder::buildModule(
    ModuleOp module) {
  auto icfg = std::make_unique<InterProceduralCFG>(module);

  // 1. 为每个函数构建CFG
  icfg->build();

  // 2. 连接调用点
  icfg->connectCallGraph();

  // 3. 执行全局分析
  icfg->computeReachability();

  return icfg;
}

void InterProceduralCFG::build() {
  ControlFlowGraphBuilder builder;

  // 遍历模块中的所有函数
  module.walk([&](triton::FuncOp func) {
    // 为每个函数构建CFG
    auto cfg = builder.build(func);
    functionCFGs[func] = std::move(cfg);
  });
}

void InterProceduralCFG::connectCallGraph() {
  // 遍历所有调用操作
  module.walk([&](Operation* op) {
    // 检查是否是调用操作
    if (auto callOp = dyn_cast<triton::CallOp>(op)) {
      StringRef calleeName = callOp.getCallee();

      // 在模块中查找被调用函数
      if (auto calleeFunc = module.lookupSymbol<triton::FuncOp>(calleeName)) {
        // 获取调用者的CFG
        Operation* parentRegion = op->getParentWithTrait<OpTrait::IsIsolatedFromAbove>();
        if (auto callerFunc = dyn_cast<triton::FuncOp>(parentRegion)) {
          auto callerCFG = getFunctionCFG(callerFunc);

          if (callerCFG) {
            // 找到调用点的基本块
            size_t callBlockId = callerCFG->findBasicBlock(op->getBlock())->id;

            // 创建调用点记录
            CallSite callSite;
            callSite.callOp = op;
            callSite.caller = callerFunc;
            callSite.callee = calleeFunc;
            callSite.callBlockId = callBlockId;

            // 找到返回后的基本块（调用点之后的块）
            Block* nextBlock = op->getBlock()->splitBlock(op);
            if (nextBlock) {
              callSite.returnBlockId = callerCFG->findBasicBlock(nextBlock)->id;
            }

            callSites.push_back(callSite);

            // 添加到调用图
            callGraph[callerFunc].push_back(calleeFunc);
            reverseCallGraph[calleeFunc].push_back(callerFunc);

            // TODO: 在callerCFG中添加CALL_EDGE和RETURN_EDGE
            // 这对于简单的function inlining分析很有用
          }
        }
      }
    }
  });
}
```

### 3.4 Dependency Analyzer (高级版本)

**基于SVFG、CFG和循环分析的高级依赖分析器**。能够识别复杂的pipeline模式，包括循环内流水线、多迭代重叠以及特定算子优化。

**注意**: 本文档提供基础框架设计。针对FlashAttention、HSTU等复杂算子的**高级pipeline调度策略**，请参考：
- `triton_graph_scheduler_advanced.md` - 包含软件流水线和多迭代重叠的详细设计
- 特别是对于**FlashAttention反向传播**，实际模式不是简单的"5个matmul"，而是包含recomputation的复杂链式计算

**更新**: 基于2025-2026年FlashAttention-4最新研究（[官方发布](https://eu.36kr.com/en/p/3711195049046148)，[技术解析](https://modal.com/blog/reverse-engineer-flash-4)），反向传播的特点是：
- Five chained MMAs with full pipeline overlap
- Recompute S and P on-the-fly (not stored during forward)
- Transposed tile storage in TMEM (or L2 buffer on Ascend)
- 2-CTA mode for cooperative tile processing

#### 3.4.1 核心数据结构

```cpp
// third_party/ascend/include/TritonGraph/Analyzer/DependencyAnalyzer.h

namespace ascend::triton {

// ---- 依赖类型细分 ----
enum class DependencyType {
  // 基本依赖类型
  TRUE_DEPENDENCY,        // 真依赖 (RAW)
  ANTI_DEPENDENCY,         // 反依赖 (WAR)
  OUTPUT_DEPENDENCY,       // 输出依赖 (WAW)
  CONTROL_DEPENDENCY,      // 控制依赖

  // Memory依赖子类型
  MEMORY_RAW,              // 内存RAW (store->load)
  MEMORY_WAR,              // 内存WAR (load->store)
  MEMORY_WAW,              // 内存WAW (store->store)

  // 循环携带依赖
  LOOP_CARRIED_TRUE,       // 循环携带真依赖
  LOOP_CARRIED_MEMORY,     // 循环携带内存依赖

  // 间接依赖
  INDIRECT_DEPENDENCY,     // 通过其他操作传递的依赖
};

// ---- Pipeline模式识别 ----
enum class PipelinePattern {
  UNKNOWN,                 // 未知或未识别
  FLASH_ATTENTION_FWD,     // FlashAttention正向 (MHA)
  FLASH_ATTENTION_BWD,     // FlashAttention反向 (5个matmul)
  HSTU_ATTENTION_FWD,      // HSTU正向 (QK: cube, attention: vector, ScoreV: cube)
  HSTU_ATTENTION_BWD,      // HSTU反向
  NSA_FUSION_FWD,          // NSA融合正向
  NSA_FUSION_BWD,          // NSA融合反向
  MLA_FUSION_FWD,          // MLA融合正向
  MLA_FUSION_BWD,          // MLA融合反向
  DSA_FUSION_FWD,          // DSA融合正向
  DSA_FUSION_BWD,          // DSA融合反向
  SIMPLE_GEMM_BIAS,        // MatMul + bias (cube + vector)
  CHAIN_COMPUTE,           // 链式计算 (cube->vector->cube->vector)
};

// ---- 依赖关系的完整表示 ----
struct DependencyEdge {
  Operation* from;           // 源操作
  Operation* to;             // 目标操作
  DependencyType type;       // 依赖类型

  // ---- 位置信息 ----
  BasicBlock* fromBlock;     // 源操作所在基本块
  BasicBlock* toBlock;       // 目标操作所在基本块

  // ---- 循环相关信息 ----
  bool isLoopCarried;        // 是否是循环携带依赖
  LoopInfo* loop;            // 所属的循环（如果是循环携带依赖）
  int loopCarriedDistance;   // 循环携带距离（跨多少迭代）

  // ---- 依赖距离 ----
  // 在同一个基本块内的操作距离（相隔多少个操作）
  int intraBlockDistance;

  // 对于循环内依赖：表示在DAG中的最短路径长度
  int dagDistance;

  // ---- Memory依赖专用 ----
  MemoryObject* memoryObject; // 涉及的内存对象（如果type是MEMORY_*）
  const MemoryDef* memoryDef; // 到达定义

  // ---- 是否可pipeline ----
  bool canPipeline;          // 是否可以通过重排序pipeline
  unsigned minPipelineDepth; // 最小pipeline深度

  // ---- 同步需求 ----
  bool needsSync;            // 是否需要插入同步
  SmallVector<Value> syncValues; // 需要同步的值

  // 是否可以通过multi-buffer消除依赖
  bool canEliminateByMultiBuffer;
};

// ---- 循环的Pipeline分析 ----
struct LoopPipelineInfo {
  LoopInfo* loop;            // 循环信息
  PipelinePattern pattern;   // 识别的模式

  // ---- 循环内的操作序列 ----
  SmallVector<Operation*> operationSequence; // 按执行顺序排列

  // ---- 每类操作的计数 ----
  int numCubeOps;            // Cube操作数量
  int numVectorOps;          // Vector操作数量
  int numMemoryOps;          // Memory操作数量

  // ---- 依赖分析 ----
  SmallVector<DependencyEdge> dependencies;   // 循环内的依赖
  SmallVector<DependencyEdge> loopCarriedDeps; // 循环携带依赖

  // ---- Pipeline可行性 ----
  bool canPipeline;          // 是否可以pipeline整个循环
  unsigned optimalDepth;     // 最优pipeline深度
  unsigned maxOverlappingIters; // 最大重叠迭代数

  // ---- 依赖距离分析 ----
  // 关键：计算依赖距离，判断是否可以多迭代重叠
  // 例如：如果cube1->vector1的距离 >= MIN_DISTANCE，则i和i+1可以重叠
  bool canMultiIterationOverlap; // 是否可以多迭代重叠
  int minInterIterationDistance; // 迭代间最小距离

  // ---- 切分因子 ----
  // 是否可以将循环拆分为多个阶段来pipeline
  bool needsLoopSplitting;
  SmallVector<Operation*> stage1Ops; // 第一阶段操作
  SmallVector<Operation*> stage2Ops; // 第二阶段操作
  SmallVector<Operation*> stage3Ops; // 第三阶段操作

  // ---- 特殊模式标记 ----
  bool hasBackToBackCube;    // 连续的cube操作（如FA反向的5个matmul）
  bool hasLongVectorChain;   // 长vector链（softmax等）
};

// ---- Pipeline机会 ----
struct PipelineOpportunity {
  // 涉及的操作
  SmallVector<Operation*> operations;  // 所有相关操作
  SmallVector<Operation*> cubeOps;     // 所有cube操作
  SmallVector<Operation*> vectorOps;   // 所有vector操作

  // ---- 位置信息 ----
  scf::ForOp loop;           // 如果是在循环内
  BasicBlock* block;         // 基本块

  // ---- 依赖分析 ----
  DenseMap<Operation*, SmallVector<DependencyEdge*>> inDeps;  // 入依赖
  DenseMap<Operation*, SmallVector<DependencyEdge*>> outDeps; // 出依赖
  SmallVector<DependencyEdge*> criticalPath; // 关键路径

  // ---- Pipeline模式 ----
  PipelinePattern pattern;   // 识别的模式
  bool isPatternMatched;     // 是否匹配特定模式

  // ---- 分析结果 ----
  bool isProfitable;         // 是否有收益
  double estimatedSpeedup;   // 估计加速比
  unsigned optimalDepth;     // 最优pipeline深度 (2, 3, 4)

  // 最大重叠迭代数
  // 例如：如果设置为3，则同时执行迭代i, i+1, i+2
  unsigned maxOverlappingIterations;

  // ---- 优化策略 ----
  enum Strategy {
    DIRECT_PIPELINE,         // 直接pipeline（无循环）
    SOFTWARE_PIPELINE,       // 软件pipeline（循环展开+重排序）
    MULTI_ITERATION_OVERLAP, // 多迭代重叠（不展开）
    STAGED_EXECUTION,        // 分阶段执行（如：prefetch + compute）
  };
  Strategy strategy;

  // ---- 软流水参数 ----
  // 提前发射的操作数（prologue）
  SmallVector<Operation*> prologueOps;  // 内核前的prologue

  // 内核中的操作（循环内）
  SmallVector<SmallVector<Operation*, 3>> kernelPipeline; // 每阶段3个迭代

  // epilogue操作
  SmallVector<Operation*> epilogueOps;  // 内核后的epilogue

  // ---- 资源需求 ----
  struct ResourceRequirement {
    int numLoadBuffers;      // 需要的load buffer数量
    int numStoreBuffers;     // 需要的store buffer数量
    int numIntermediateBuffers; // 中间buffer
  };
  ResourceRequirement resources;
};

// ---- 依赖分析器 ----
class DependencyAnalyzer {
public:
  DependencyAnalyzer(const SparseValueFlowGraph& vfg,
                    const ControlFlowGraph& cfg);

  // ---- 主分析接口 ----
  // 分析所有pipeline机会
  SmallVector<PipelineOpportunity> findAllOpportunities();

  // 分析特定循环
  std::optional<LoopPipelineInfo> analyzeLoop(scf::ForOp loop);

  // 分析特定基本块（非循环）
  std::optional<PipelineOpportunity> analyzeBlock(BasicBlock* block);

  // ---- 模式识别 ----
  // 识别特定算子模式
  PipelinePattern recognizePattern(const SmallVector<Operation*>& ops);

  // 针对特定模式的分析
  LoopPipelineInfo analyzeFlashAttention(scf::ForOp loop);
  LoopPipelineInfo analyzeHSTU(scf::ForOp loop);
  LoopPipelineInfo analyzeNSA(scf::ForOp loop);
  LoopPipelineInfo analyzeMLA(scf::ForOp loop);
  LoopPipelineInfo analyzeDSA(scf::ForOp loop);
  LoopPipelineInfo analyzeChainPattern(scf::ForOp loop);

  // ---- 核心分析函数 ----
  // 构建依赖图
  SmallVector<DependencyEdge> buildDependencyGraph(
      const SmallVector<Operation*>& ops);

  // 计算依赖距离
  int computeDependencyDistance(Operation* from, Operation* to,
                                const SmallVector<Operation*>& ops);

  // 识别循环携带依赖
  SmallVector<DependencyEdge> identifyLoopCarriedDependencies(scf::ForOp loop);

  // 计算关键路径
  SmallVector<DependencyEdge*> computeCriticalPath(
      const SmallVector<DependencyEdge>& deps);

  // ---- 依赖查询 ----
  const SmallVector<DependencyEdge>& getDependencies() const {
    return allDependencies;
  }

  const DenseMap<scf::ForOp, LoopPipelineInfo>& getLoopInfos() const {
    return loopInfos;
  }

  // ---- 成本模型 ----
  double estimateOpLatency(Operation* op) const;
  double estimateLoopLatency(scf::ForOp loop) const;
  double estimateSpeedup(const PipelineOpportunity& opp) const;

  // ---- 其他查询 ----
  // 查询最繁忙的core类型
  CoreType getBusiestCore(const SmallVector<Operation*>& ops) const;

  // 查询内存带宽需求
  double estimateMemoryBandwidth(const SmallVector<Operation*>& ops) const;

private:
  const SparseValueFlowGraph& vfg;
  const ControlFlowGraph& cfg;

  // 依赖数据库
  SmallVector<DependencyEdge> allDependencies;
  DenseMap<scf::ForOp, LoopPipelineInfo> loopInfos;

  // 辅助函数
  bool hasLoopIndependentDependency(Operation* from,
                                   Operation* to,
                                   scf::ForOp loop) const;

  bool hasLoopCarriedDependency(Operation* from,
                               Operation* to,
                               scf::ForOp loop) const;

  bool areOperationsIndependent(Operation* a, Operation* b) const;

  int getIterationDistance(Operation* from, Operation* to) const;
};

} // namespace ascend::triton
```

#### 3.4.2 模式识别算法

```cpp
// third_party/ascend/lib/TritonGraph/Analyzer/DependencyAnalyzer.cpp

// ---- 主分析入口 ----
SmallVector<PipelineOpportunity>
DependencyAnalyzer::findAllOpportunities() {
  SmallVector<PipelineOpportunity> opportunities;

  // 遍历模块中的所有函数
  module.walk([&](triton::FuncOp func) {
    // 遍历函数中的所有循环
    func.walk([&](scf::ForOp loop) {
      auto loopInfo = analyzeLoop(loop);
      if (loopInfo && loopInfo->canPipeline) {
        // 转换为PipelineOpportunity
        PipelineOpportunity opp = convertToOpportunity(*loopInfo);
        if (opp.isProfitable && opp.estimatedSpeedup > 1.05) {
          opportunities.push_back(opp);
        }
      }
    });

    // 如果函数内没有循环，分析基本块
    if (opportunities.empty()) {
      func.walk([&](Block* block) {
        if (auto blockOpportunity = analyzeBlock(block)) {
          if (blockOpportunity->isProfitable) {
            opportunities.push_back(*blockOpportunity);
          }
        }
      });
    }
  });

  return opportunities;
}

// ---- 循环分析主函数 ----
std::optional<LoopPipelineInfo>
DependencyAnalyzer::analyzeLoop(scf::ForOp loop) {
  LoopPipelineInfo info;
  info.loop = loop;

  // 1. 收集循环内的所有操作
  SmallVector<Operation*> opsInLoop;
  loop.getBody()->walk([&](Operation* op) {
    if (!isa<scf::YieldOp>(op) && !isa<scf::ForOp>(op)) {
      opsInLoop.push_back(op);
    }
  });

  if (opsInLoop.empty()) {
    return std::nullopt; // 空循环
  }

  info.operationSequence = opsInLoop;

  // 2. 统计各类操作
  info.numCubeOps = llvm::count_if(opsInLoop, [&](Operation* op) {
    return classifyOperation(op) == OpType::CUBE;
  });
  info.numVectorOps = llvm::count_if(opsInLoop, [&](Operation* op) {
    return classifyOperation(op) == OpType::VECTOR;
  });
  info.numMemoryOps = llvm::count_if(opsInLoop, [&](Operation* op) {
    return classifyOperation(op) == OpType::MEMORY;
  });

  // 需要至少一个cube和一个vector才值得pipeline
  if (info.numCubeOps == 0 || info.numVectorOps == 0) {
    return std::nullopt;
  }

  // 3. 识别模式
  info.pattern = recognizePattern(opsInLoop);
  info.isPatternMatched = (info.pattern != PipelinePattern::UNKNOWN);

  // 4. 如果匹配特定模式，使用专门的分析
  switch (info.pattern) {
    case PipelinePattern::FLASH_ATTENTION_FWD:
      info = analyzeFlashAttention(loop);
      break;
    case PipelinePattern::HSTU_ATTENTION_FWD:
      info = analyzeHSTU(loop);
      break;
    case PipelinePattern::NSA_FUSION_FWD:
      info = analyzeNSA(loop);
      break;
    case PipelinePattern::MLA_FUSION_FWD:
      info = analyzeMLA(loop);
      break;
    case PipelinePattern::DSA_FUSION_FWD:
      info = analyzeDSA(loop);
      break;
    case PipelinePattern::CHAIN_COMPUTE:
    default:
      // 通用链式计算模式
      info = analyzeChainPattern(loop);
      break;
  }

  // 5. 估算加速比
  info.canPipeline = computePipelineFeasibility(info);
  if (info.canPipeline) {
    info.estimatedSpeedup = estimateSpeedupFromInfo(info);
    info.optimalDepth = computeOptimalDepth(info);
  }

  return info;
}

// ---- FlashAttention正向分析 ----
// 模式: load Q, K, V -> compute QK (cube) -> softmax (vector, 多项式exp)
//       -> compute O (cube) -> store O
// 特点: 2个cube操作，中间有复杂的softmax (vector计算)
// 流水线策略:
//   - Prologue: 预取Q_i, K_i, V_i
//   - Kernel:
//     - iteration i:   compute QK_i (cube) -> softmax_i (vector)
//     - iteration i+1: compute O_i (cube) (与softmax_{i+1} vector并行)
//     - load Q_{i+2}, K_{i+2}, V_{i+2} (memory)
//   - 深度: 2-3
LoopPipelineInfo
DependencyAnalyzer::analyzeFlashAttention(scf::ForOp loop) {
  LoopPipelineInfo info;
  info.loop = loop;
  info.pattern = PipelinePattern::FLASH_ATTENTION_FWD;

  // 1. 识别操作序列
  // FlashAttention典型的4阶段:
  // Stage 1: Load/Memory (tt.load Q, K, V)
  // Stage 2: Cube compute (tt.dot QK^T)
  // Stage 3: Vector compute (softmax: exp, sum, div)
  // Stage 4: Cube compute (tt.dot ScoreV)
  // Stage 5: Store (tt.store O)

  info.operationSequence.clear();

  // 遍历循环体内的操作，按阶段分类
  SmallVector<Operation*> loadOps;   // 加载Q, K, V
  SmallVector<Operation*> qkOps;     // QK计算 (cube)
  SmallVector<Operation*> softmaxOps; // softmax (vector)
  SmallVector<Operation*> oOps;      // O计算 (cube)
  SmallVector<Operation*> storeOps;  // store O

  loop.getBody()->walk([&](Operation* op) {
    if (isa<triton::LoadOp>(op)) {
      loadOps.push_back(op);
      info.operationSequence.push_back(op);
    } else if (isa<triton::DotOp>(op) && qkOps.empty()) {
      // 第一个dot: QK^T
      qkOps.push_back(op);
      info.operationSequence.push_back(op);
    } else if (classifyOperation(op) == OpType::VECTOR &&
               llvm::any_of(op->getOperands().getTypes(),
                           [&](Type t) { return t.isF32(); })) {
      // softmax相关的vector操作 (exp, div等)
      softmaxOps.push_back(op);
      info.operationSequence.push_back(op);
    } else if (isa<triton::DotOp>(op) && !qkOps.empty()) {
      // 第二个dot: Score @ V
      oOps.push_back(op);
      info.operationSequence.push_back(op);
    } else if (isa<triton::StoreOp>(op)) {
      storeOps.push_back(op);
      info.operationSequence.push_back(op);
    }
  });

  // 2. 统计操作数量
  info.numCubeOps = qkOps.size() + oOps.size();
  info.numVectorOps = softmaxOps.size();
  info.numMemoryOps = loadOps.size() + storeOps.size();

  // 3. 分析依赖关系
  info.dependencies = buildDependencyGraph(info.operationSequence);

  // 4. 计算关键路径
  info.criticalPath = computeCriticalPath(info.dependencies);

  // 5. 识别循环携带依赖
  info.loopCarriedDeps = identifyLoopCarriedDependencies(loop);

  // 6. 检查是否可以多迭代重叠
  // 关键：检查迭代间依赖距离
  // 如果criticalPathLength < loopTripCount * iterationLatency，则可以重叠
  info.canMultiIterationOverlap = true;
  info.minInterIterationDistance = 2; // 默认可以重叠2个迭代

  // 7. 根据依赖距离确定最优pipeline深度
  // 通过profiling和性能模型确定
  // FlashAttention典型的dependency distances:
  // - load -> compute QK: ~50 cycles (内存延迟)
  // - compute QK -> softmax: ~200 cycles (cube计算)
  // - softmax -> compute O: ~100 cycles (vector计算)
  info.optimalDepth = 3; // 推荐3级pipeline
  info.maxOverlappingIters = 3; // 3个迭代同时执行

  // 8. 阶段划分
  // 根据时间线重新排序：
  // 时隙 0:   load i
  // 时隙 50:  compute QK i
  // 时隙 250: softmax i, load i+1
  // 时隙 350: compute O i, compute QK i+1
  // 时隙 550: store O i, softmax i+1, load i+2
  info.stage1Ops = loadOps;  // Memory阶段
  info.stage2Ops = qkOps;    // QK计算阶段
  info.stage3Ops = softmaxOps; // Softmax阶段
  info.stage4Ops = oOps;     // O计算阶段
  info.stage5Ops = storeOps; // Store阶段

  // 9. 资源需求
  info.canPipeline = true;

  return info;
// ---- Multi-head Latent Attention (MLA) 正向分析 ----
// 基于2025-2026年最新研究（DeepSeek-V2/V3 MLA实现）
// 参考：
// - MLA详细解析: https://pyimagesearch.com/2025/10/13/kv-cache-optimization-via-multi-head-latent-attention/
// - 实现细节: https://liorsinai.github.io/machine-learning/2025/02/22/mla.html
// - FlashMLA: https://www.shashankshekhar.com/blog/flashmla/flashmla-1-mla
//
// MLA核心特点（相比标准MHA）：
// - KV Cache压缩64x：从共享latent space重新生成K,V
// - 操作序列：Down-projection -> Up-projection (on-the-fly) -> Q projection -> Attention
// - 关键优化：Weight absorption减少矩阵乘法，Decoupled RoPE
// - 计算密度：在latent space操作，减少memory bandwidth需求

LoopPipelineInfo
DependencyAnalyzer::analyzeMLA(scf::ForOp loop) {
  LoopPipelineInfo info;
  info.loop = loop;
  info.pattern = PipelinePattern::MLA_FUSION_FWD;

  // 1. 识别MLA操作序列（按执行顺序）
  // 标准的MLA前向包含以下阶段：
  // Stage 1: Load X (input) -> Down-projection: C_kv = X @ W_dkv (Cube)
  // Stage 2: Load/cache C_kv, Up-project: K = C_kv @ W_uk, V = C_kv @ W_uv (Cube)
  // Stage 3: Query path: X -> C_q -> Q (Cube + Vector for RMSNorm)
  // Stage 4: Load Q/K/V -> Attention: S = Q @ K^T (Cube)
  // Stage 5: Softmax -> P = softmax(S) (Vector)
  // Stage 6: O = P @ V (Cube)
  // Stage 7: Output projection (Cube)

  SmallVector<Operation*> downProjOps;    // X -> C_kv
  SmallVector<Operation*> upProjOps;      // C_kv -> K, V
  SmallVector<Operation*> queryOps;       // Q projection path
  SmallVector<Operation*> attentionOps;   // QK^T, Softmax, O = P@V
  SmallVector<Operation*> outputProjOps;  // Output projection

  loop.getBody()->walk([&](Operation* op) {
    info.operationSequence.push_back(op);

    // 分类操作类型
    if (isa<triton::DotOp>(op) || classifyOperation(op) == OpType::CUBE) {
      info.cubeOps.push_back(op);
      info.numCubeOps++;

      // 根据位置识别不同阶段的matmul
      // 需要额外的模式匹配来确定具体阶段
    } else if (classifyOperation(op) == OpType::VECTOR) {
      info.vectorOps.push_back(op);
      info.numVectorOps++;
      // RMSNorm, softmax的exp/div等elementwise操作
    } else if (isa<triton::LoadOp>(op)) {
      info.numMemoryOps++;
      // Load input X, load cached C_kv (decode阶段)
    } else if (isa<triton::StoreOp>(op)) {
      info.numMemoryOps++;
      // Store output, store C_kv (prefill阶段)
    }
  });

  // 2. MLA特定的pipeline策略
  //
  // MLA的两个关键优化特点：
  // a) Weight Absorption: 预计算 W_qk = W_uq^T @ W_uk, 在latent space计算
  //    减少一次matmul: Q, K from X -> S from C_q, C_kv
  //
  // b) KV Latent Cache: 只缓存C_kv (512维)，不缓存完整的K/V (16384维)
  //    Up-projection在每次forward时重新计算
  //
  // Pipeline机会：
  // - Stage1: Down-projection (if not using weight absorption)
  // - Stage2: Up-projection K, V (2 parallel matmuls)
  // - Stage3: Query projection Q (parallel to K/V)
  // - Stage4: Attention QK^T, softmax, O (FlashAttention-style pipeline)
  //
  // 深度分析：由于KV up-projection只是小矩阵乘latent(512) -> K/V(16384)
  // 计算密度较高，适合与Q计算overlap

  bool hasWeightAbsorption = false; // 是否检测到weight absorption
  bool hasLatentCache = false;      // 是否cache C_kv

  // 启发式检测：根据操作数和tensor维度
  if (info.cubeOps.size() >= 4) {
    // 至少有：down-proj, up-K, up-V, Q-proj, QK^T, O-proj
    // 说明是标准的MLA路径
    hasLatentCache = true;

    // 检查是否有weight absorption（减少的matmul数量）
    if (info.cubeOps.size() <= 6) {
      // 如果只有6个matmul（vs标准7个），可能应用了weight absorption
      hasWeightAbsorption = true;
    }
  }

  // 3. 基于MLA特点的pipeline优化策略
  //
  // 主要优化方向：
  // a) 并行化up-projection: K和V可以同时在不同核心计算
  // b) 重叠query和KV: Q投影可以与K/V up-projection并行
  // c) FlashAttention pipeline: QK^T -> softmax -> OV可以软件pipelining
  //
  // 推荐pipeline深度：4（适用于大多数场景）
  // Stage 1: Load + Down-projection (if needed)
  // Stage 2: Up-project K,V + Q-projection (parallel)
  // Stage 3: Attention QK^T + softmax
  // Stage 4: OV + Output projection

  info.optimalDepth = 4;
  info.maxOverlappingIters = 4;

  // 如果应用了weight absorption，减少depth到3
  if (hasWeightAbsorption) {
    info.optimalDepth = 3;  // 更少的matmul，更浅的pipeline
    info.maxOverlappingIters = 3;
  }

  // 4. 阶段划分
  // 需要根据实际操作序列进行更精确的分析
  info.stage1Ops = {};  // Down-project X -> C_kv
  info.stage2Ops = {};  // Up-project C_kv -> K,V and X -> Q
  info.stage3Ops = {};  // QK^T, softmax
  info.stage4Ops = {};  // OV, output projection

  // 5. Resource requirement
  // MLA的中间结果较大（K,V up-projected），需要更多buffer
  info.canPipeline = true;
  info.canEliminateByMultiBuffer = true;  // 多buffer管理关键

  return info;
}

// ---- HSTU正向分析 ----
LoopPipelineInfo
DependencyAnalyzer::analyzeHSTU(scf::ForOp loop) {
  // HSTU特点:
  // 1. compute QK (cube) -> attention score (vector) -> compute output (cube)
  // 2. 中间的vector操作可能有限，导致cube操作间隔较短
  // 3. 需要更激进的pipeline策略，可能需要提前发射多个迭代

  LoopPipelineInfo info;
  info.loop = loop;
  info.pattern = PipelinePattern::HSTU_ATTENTION_FWD;

  // 识别操作
  loop.getBody()->walk([&](Operation* op) {
    info.operationSequence.push_back(op);
    if (isa<triton::DotOp>(op)) {
      info.cubeOps.push_back(op);
      info.numCubeOps++;
    } else if (classifyOperation(op) == OpType::VECTOR) {
      info.vectorOps.push_back(op);
      info.numVectorOps++;
    } else if (isa<triton::LoadOp>(op) || isa<triton::StoreOp>(op)) {
      info.numMemoryOps++;
    }
  });

  // HSTU的模式: cube1 -> vector1 -> cube2
  // 关键挑战: cube操作间隔短，需要提前启动多个迭代
  if (info.cubeOps.size() >= 2) {
    // 分析cube1到cube2的距离
    auto deps = buildDependencyGraph(info.operationSequence);

    // 如果没有数据依赖，可以激进overlap
    bool backToBackCube = llvm::none_of(deps, [&](const DependencyEdge& dep) {
      return dep.from == info.cubeOps[0] && dep.to == info.cubeOps[1] &&
             dep.type == DependencyType::TRUE_DEPENDENCY;
    });

    if (backToBackCube) {
      info.hasBackToBackCube = true;
      // 可以深度pipeline，提前发射2-3个迭代
      info.optimalDepth = 4; // 更深
      info.maxOverlappingIters = 4;
      info.minInterIterationDistance = 1; // 距离可以是1

      // 由于cube间隔短，需要平衡各核心负载
      info.strategy = PipelineOpportunity::MULTI_ITERATION_OVERLAP;
    }
  }

  info.canPipeline = true;
  return info;
}

// ---- FlashAttention反向分析 (基于FlashAttention-4最新研究) ----
// 注意：这不是简单的"5个matmul"，而是复杂的链式计算，包括recomputation
// 详细设计请参考: third_party/ascend/docs/en/triton_graph_scheduler_advanced.md
// 研究文献：
// - FlashAttention-4: https://eu.36kr.com/en/p/3711195049046148
// - FA原理: https://mathfirst.github.io/files/from_online_softmax_to_FlashAttention_2.pdf
//
// 真实操作序列（块级别）：
//   Load -> Sij=Qi@Kj^T -> Pij=softmax(Sij) -> dVj+=Pij.T@dOi -> dPij=dOi@Vj.T
//   -> dSij=Pij*(dPij-Di) -> dQi+=dSij@Kj -> dKj+=dSij.T@Qi -> Store

LoopPipelineInfo
DependencyAnalyzer::analyzeFlashAttentionBwd(scf::ForOp loop) {
  LoopPipelineInfo info;
  info.loop = loop;
  info.pattern = PipelinePattern::FLASH_ATTENTION_BWD;

  // 识别操作序列和依赖关系
  loop.getBody()->walk([&](Operation* op) {
    info.operationSequence.push_back(op);

    if (isa<triton::DotOp>(op) || classifyOperation(op) == OpType::CUBE) {
      info.cubeOps.push_back(op);
      info.numCubeOps++;
    } else if (classifyOperation(op) == OpType::VECTOR) {
      info.vectorOps.push_back(op);
      info.numVectorOps++;
    } else if (isa<triton::LoadOp>(op) || isa<triton::StoreOp>(op)) {
      info.numMemoryOps++;
    }
  });

  // 策略：双层pipeline
  // 1. 内层Q循环：在单次迭代内重叠matmul和vector操作
  // 2. 外层KV循环：在多个KV块之间重叠计算
  // 归因于FA-4的研究：https://modal.com/blog/reverse-engineer-flash-4

  info.optimalDepth = 4; // 4级pipeline达到最佳性能
  info.maxOverlappingIters = 4; // 同时重叠4个迭代

  // 需要multi-buffer避免读后写冲突
  info.canEliminateByMultiBuffer = true;
  info.canEliminateByMultiBuffer = true;

  info.canPipeline = true;
  return info;
}

// ---- 通用链式计算模式 ----
LoopPipelineInfo
DependencyAnalyzer::analyzeChainPattern(scf::ForOp loop) {
  // 通用模式: 任意数量的cube和vector交替
  // 分析策略:
  // 1. 构建依赖图
  // 2. 计算依赖距离
  // 3. 判断是否可以overlap
  // 4. 确定最优pipeline深度

  LoopPipelineInfo info;
  info.loop = loop;

  // 收集所有操作
  SmallVector<Operation*> ops;
  loop.getBody()->walk([&](Operation* op) {
    if (classifyOperation(op) != OpType::CONTROL &&
        !isa<scf::YieldOp>(op)) {
      ops.push_back(op);
    }
  });

  info.operationSequence = ops;

  // 构建完整的依赖图
  info.dependencies = buildDependencyGraph(ops);

  // 识别cube和vector
  for (Operation* op : ops) {
    if (classifyOperation(op) == OpType::CUBE) {
      info.cubeOps.push_back(op);
      info.numCubeOps++;
    } else if (classifyOperation(op) == OpType::VECTOR) {
      info.vectorOps.push_back(op);
      info.numVectorOps++;
    } else if (isa<triton::LoadOp>(op) || isa<triton::StoreOp>(op)) {
      info.numMemoryOps++;
    }
  }

  // 分析依赖距离
  // 计算最短依赖路径长度
  info.minInterIterationDistance = computeMinDistanceBetweenIters(info);

  // 如果没有循环携带依赖，可以深度pipeline
  info.loopCarriedDeps = identifyLoopCarriedDependencies(loop);
  if (info.loopCarriedDeps.empty()) {
    // 循环独立的计算，可以自由overlap
    info.canPipeline = true;
    info.canMultiIterationOverlap = true;

    // 根据cube op的数量确定pipeline深度
    // 考虑到Ascend NPU的硬件并行度，通常最多3-4个迭代同时重叠
    info.optimalDepth = std::min(info.numCubeOps, 3);
    info.maxOverlappingIters = info.optimalDepth;
  } else {
    // 有循环携带依赖，限制重叠
    info.canPipeline = true;
    info.canMultiIterationOverlap = false;
    info.optimalDepth = 2; // 保守的2级pipeline
  }

  return info;
}

// ---- Native Sparse Attention (NSA) 分析 ----
// 基于ACL 2025 Best Paper: "Hardware-Aligned and Natively Trainable Sparse Attention"
// 参考: https://aclanthology.org/2025.acl-long.1126/
//
// NSA特点（2025年突破性设计）：
// - 动态分层策略：粗粒度token压缩 + 细粒度token选择 + 滑动窗口
// - 可学习的稀疏模式：在预训练中学习，非训练后处理
// - 与GQA集成：支持多头并行
// - 算术强度平衡：针对现代GPU/TPU内存层次优化
//
// 操作序列：
// 1. Token重要性评分（MLP-based）
// 2. Top-k块选择（chunk-level）
// 3. 细粒度token选择（在每个选中的块内）
// 4. 滑动窗口（固定local窗口）
// 5. 合并选择的token: concat(compressed, selected, local)
// 6. 在选择的子集上执行标准attention: Q @ K^T, softmax, @V

LoopPipelineInfo
DependencyAnalyzer::analyzeNSA(scf::ForOp loop) {
  LoopPipelineInfo info;
  info.loop = loop;
  info.pattern = PipelinePattern::NSA_FUSION_FWD;

  // 1. 识别NSA操作序列
  // NSA的关键：选择性计算，非所有token都参与
  SmallVector<Operation*> scoringOps;     // Token重要性评分
  SmallVector<Operation*> selectionOps;   // Top-k选择
  SmallVector<Operation*> gatherOps;      // 从选中的token收集K,V
  SmallVector<Operation*> attnOps;        // 在选择的子集上计算attention

  loop.getBody()->walk([&](Operation* op) {
    info.operationSequence.push_back(op);

    if (isa<triton::DotOp>(op) || classifyOperation(op) == OpType::CUBE) {
      info.cubeOps.push_back(op);
      info.numCubeOps++;
      // 注意：由于稀疏性，matmul数量减少
    } else if (classifyOperation(op) == OpType::VECTOR) {
      info.vectorOps.push_back(op);
      info.numVectorOps++;
      // MLP评分、softmax等
    } else if (isa<triton::LoadOp>(op) || isa<triton::StoreOp>(op)) {
      info.numMemoryOps++;
      // Gather操作：从原始序列加载选中的token
    }
  });

  // 2. NSA的稀疏性分析
  // 关键指标：稀疏率（计算token的比例）
  // - NSA典型稀疏率：70-75% token移除
  // - 剩余token：压缩块 + 选择token + local窗口
  //
  // Pipeline机会：
  // a) 评分与选择并行：token重要性评分可以与前一迭代的gather并行
  // b) 动态负载平衡：不同迭代可能选择不同数量的token，需要均衡
  // c) Gather-Compute流水线：Gather K,V时，可以开始计算QK^T

  info.hasSparsePattern = true;
  info.sparseRatioEstimate = 0.3;  // 保留30% token，移除70%

  // 3. 基于稀疏性的pipeline策略
  //
  // 与传统attention的区别：
  // - 不规则的内存访问（gather）
  // - 计算量动态变化 per iteration
  // - 需要在评分完成后才知道依赖关系
  //
  // 推荐策略：
  // - 3级pipeline足够
  // - Stage 1: 评分 + 选择
  // - Stage 2: Gather K,V, Q projection
  // - Stage 3: Attention计算 (QK^T, softmax, OV)

  info.optimalDepth = 3;
  info.maxOverlappingIters = 3;

  // 4. 特殊考虑：动态负载均衡
  // 由于稀疏选择，不同iteration的计算量可能不同
  // 需要运行时动态调整
  info.needsLoadBalancing = true;
  info.isDynamicWorkload = true;

  // 5. 内存访问模式
  // - 不规则的gather（indirect loads）
  // - 需要使用特殊指令（如tt.gather）
  // - Tiling策略：按选中的token block tiling
  info.hasIrregularMemoryAccess = true;

  // 6. 性能预期
  // 理论加速比：1 / (1 - sparseRatio) = 1 / 0.3 ≈ 3.3x
  // 实际加速比：2-2.5x（考虑gather开销、负载不均衡）
  info.canPipeline = true;
  info.canEliminateByMultiBuffer = true;  // Multi-buffer管理关键

  return info;
}

// ---- Dynamic Sparse Attention (DSA) 分析 ----
// 基于Dynamic Sparse Attention各种变体（2025-2026）
// 参考:
// - Hardware-Aligned Sparse Attention: https://aclanthology.org/2025.acl-long.1126/
// - DeepSeek-V3.2 DSA: 动态token级稀疏性
// - PADE: Predictor-Free Stage Fusion: https://quantumzeitgeist.com/1x-attention-sparse-accelerator-achieves-speedup-predictor-free-stage-fusion/
//
// DSA特点：
// - 动态token选择：每个head独立选择重要的token
// - 可学习的gating：使用Gumbel-Softmax学习稀疏模式
// - 运行时适应性：稀疏率可动态调整（30%-70%）
// - 细粒度并行：不同head在不同token上计算

LoopPipelineInfo
DependencyAnalyzer::analyzeDSA(scf::ForOp loop) {
  LoopPipelineInfo info;
  info.loop = loop;
  info.pattern = PipelinePattern::DSA_FUSION_FWD;

  // 1. 识别DSA操作序列
  // DSA的关键：每个head独立动态选择token
  SmallVector<Operation*> gatingOps;      // Gating network（计算重要性得分）
  SmallVector<Operation*> gumbelOps;      // Gumbel-Softmax采样
  SmallVector<Operation*> selectionOps;   // 动态token选择
  SmallVector<Operation*> perHeadOps;     // 每个head独立计算
  SmallVector<Operation*> reductionOps;   // 跨head结果聚合

  loop.getBody()->walk([&](Operation* op) {
    info.operationSequence.push_back(op);

    if (isa<triton::DotOp>(op) || classifyOperation(op) == OpType::CUBE) {
      info.cubeOps.push_back(op);
      info.numCubeOps++;
      // 注意：由于稀疏性和多head，实际计算量动态变化
    } else if (classifyOperation(op) == OpType::VECTOR) {
      info.vectorOps.push_back(op);
      info.numVectorOps++;
      // Gating network、Gumbel-Softmax、softmax等
    } else if (isa<triton::LoadOp>(op) || isa<triton::StoreOp>(op)) {
      info.numMemoryOps++;
      // Dynamic gather：基于选择mask加载token
    }
  });

  // 2. DSA的稀疏性特点
  // - 更细粒度：per-head稀疏模式（vs NSA的uniform）
  // - 动态变化：不同layer/iteration可能不同
  // - 可微分：Gumbel-Softmax允许梯度传播
  //
  // Pipeline机会：
  // a) Gating与计算overlap：计算当前head的gating时，并行计算其他head
  // b) Head级并行：不同head分配到不同core
  // c) 选择-计算融合：选择token后立即开始计算，减少内存占用

  info.hasSparsePattern = true;
  info.hasPerHeadSparsity = true;  // 每个head独立稀疏
  info.sparseRatioEstimate = 0.4;  // 视具体实现而定，典型40%保留

  // 3. DSA特定的挑战和策略
  //
  // 挑战：
  // - 不规则计算：不同head计算量不同（动态稀疏）
  // - Gating网络额外开销：需要计算重要性得分
  // - Gather操作更复杂：per-head不同的索引
  //
  // 策略：
  // - 4级pipeline，更好的隐藏延迟
  // - Stage 1: Gating network + selection
  // - Stage 2: Gather Q,K,V for selected tokens
  // - Stage 3: Per-head attention computation (parallel)
  // - Stage 4: Reduction across heads

  info.optimalDepth = 4;  // DSA需要更深的pipeline
  info.maxOverlappingIters = 4;

  // 4. 特殊优化：Head并行化
  // 由于各head独立，可以分配到多个AI Core
  info.enablePerHeadParallelism = true;
  info.numCoresForHeads = std::min(numHeads, 16);  // 使用最多16个core

  // 5. Gumbel-Softmax优化
  // - 采样操作有随机性，可以预计算
  // - Straight-through estimator允许梯度反向传播
  // - 在前向传播时可以并行化
  info.hasStochasticGating = true;

  // 6. 性能预期
  // 理论加速：1 / (1 - sparseRatio) = 1 / 0.4 = 2.5x
  // 实际加速：1.8-2.2x（考虑gating开销、gather不规则性）
  // 对比NSA：DSA更灵活但开销更大
  info.canPipeline = true;
  info.canEliminateByMultiBuffer = true;

  return info;
}

// ---- 计算Pipeline可行性 ----
bool DependencyAnalyzer::computePipelineFeasibility(
    const LoopPipelineInfo& info) {
  // 判断是否值得pipeline

  // 标准1: 至少2个cube操作
  if (info.numCubeOps < 2) return false;

  // 标准2: 有足够的依赖距离
  if (info.minInterIterationDistance < 1) return false;

  // 标准3: 估计的加速比
  double speedup = estimateSpeedupFromInfo(info);
  if (speedup < 1.1) return false; // 至少10%提升

  // 标准4: 资源限制检查
  // Ascend NPU通常有限制，例如:
  // - 最大3-4个cube tile同时运行
  // - 最大8-16个vector task同时运行
  // - Buffer大小限制
  if (info.maxOverlappingIters > 4) {
    // 限制最大重叠迭代数
    info.maxOverlappingIters = 4;
  }

  return true;
}

// ---- 计算最优Pipeline深度 ----
unsigned DependencyAnalyzer::computeOptimalDepth(
    const LoopPipelineInfo& info) {
  // 基于依赖距离和计算时间确定深度

  // 默认深度2
  unsigned depth = 2;

  // 如果依赖距离长，可以增加深度
  if (info.minInterIterationDistance >= 3) {
    depth = 3;
  }
  if (info.minInterIterationDistance >= 5) {
    depth = 4;
  }

  // 针对特定模式调整
  switch (info.pattern) {
    case PipelinePattern::FLASH_ATTENTION_FWD:
      depth = 3; // FA正向深度3最合适
      break;
    case PipelinePattern::FLASH_ATTENTION_BWD:
      depth = 4; // FA反向深度4-5
      break;
    case PipelinePattern::HSTU_ATTENTION_FWD:
      depth = 3;
      break;
    case PipelinePattern::CHAIN_COMPUTE:
      // 根据操作数量
      depth = std::min(info.numCubeOps, 3);
      break;
    default:
      break;
  }

  // 资源约束：Ascend NPU通常限制同时运行的cube任务数
  if (depth > 4) {
    depth = 4;
  }

  return depth;
}

// ---- 成本模型 ----
double DependencyAnalyzer::estimateOpLatency(Operation* op) const {
  // 基于Ascend NPU特性估算延迟
  auto type = classifyOperation(op);

  switch (type) {
    case OpType::CUBE:
      // tt.dot在Ascend上的大致延迟
      // 取决于block size和精度，通常在100-500 cycles
      return 200.0;
    case OpType::VECTOR:
      // elementwise操作，延迟较低
      return 10.0;
    case OpType::MEMORY:
      // load/store延迟，包括内存访问
      return 50.0;
    default:
      return 5.0;
  }
}

double DependencyAnalyzer::estimateSpeedupFromInfo(
    const LoopPipelineInfo& info) const {
  // 计算串行执行时间
  double serialTime = 0.0;
  for (Operation* op : info.operationSequence) {
    serialTime += estimateOpLatency(op);
  }

  // 计算流水线执行时间
  // 约等于: (criticalPathLength * tripCount) / depth
  double criticalPathLength = 0.0;
  for (const auto* edge : info.criticalPath) {
    criticalPathLength += estimateOpLatency(edge->from);
  }

  // 假设循环次数为N
  // 流水线执行时间 = (N + depth - 1) * (criticalPathLength / depth)
  // 保守估计使用N=100
  double N = 100.0;
  double pipelineTime = (N + info.optimalDepth - 1.0) *
                        (criticalPathLength / info.optimalDepth);

  return serialTime / pipelineTime;
}



  unsigned numIterations;            // 当前块的迭代数
  unsigned iterationIdOffset;        // 为kernel开始的迭代ID

  // 每个迭代的内部分配
  SmallVector<IterationSchedule> iterations;

  // buffer分配（区分不同迭代）
  DenseMap<Value, SmallVector<Value, 4>> multiBufferMap; // original -> buffers

  // 同步操作
  SmallVector<std::pair<SoftwarePipelineStage, SyncOp>> syncOps;

  // 代码生成辅助信息
  struct CodeGenInfo {
    // Kernel循环需要修改trip count
    // 原tripCount = N
    // 实际循环次数 = N - maxOverlappingIters + 1
    bool adjustTripCount;
    IntegerAttr originalTripCount;
  } codeGenInfo;
};

// ---- 完整的调度计划 ----
struct PipelineSchedule {
  // 针对特定模式的配置
  PipelinePattern pattern;  // 省略后续内容，包含下一部分

};

    readyQueue.remove(op);

    // 分配到合适的stage
    unsigned stage = assignStage(op, options);
    schedule.stages[stage].ops.push_back(op);
    scheduled.push_back(op);

    // 更新就绪队列: 添加新就绪的操作
    for (Operation* user : getUsers(op)) {
      if (isReady(user, scheduled)) {
        readyQueue.insert(user);
      }
    }

    // 重新排序就绪队列
    llvm::sort(readyQueue, [&](Operation* a, Operation* b) {
      return priorities[a] > priorities[b];
    });
  }

  // 5. 插入同步原语
  if (options.minimizeSync) {
    insertSyncPrimitives(schedule);
  }

  // 6. 应用多buffer
  if (options.enableMultiBuffer) {
    applyMultiBuffer(schedule);
  }

  // 7. 平衡pipeline
  if (options.balanceStages) {
    balancePipeline(schedule);
  }
}
```

### 3.6 TTIR Generator

```cpp
class TTIRGenerator {
public:
  TTIRGenerator(MLIRContext* context) : builder(context) {}

  // 从schedule生成TTIR
  LogicalResult generate(const Schedule& schedule, ModuleOp module);

  // 生成特定region的TTIR
  LogicalResult generateRegion(const Schedule& schedule, Region* region);

  // 生成同步操作
  void generateSyncOps(ArrayRef<SyncOp> syncOps, OpBuilder& builder);

  // 生成buffer管理
  void generateBufferMgmt(const Schedule& schedule, OpBuilder& builder);

private:
  OpBuilder builder;

  // 克隆操作到新位置
  Operation* cloneOperation(Operation* op, OpBuilder& builder);

  // 替换值为新buffer
  Value replaceWithBuffer(Value oldValue, BufferInfo buffer);

  // 生成循环 (pipeline main loop)
  scf::ForOp generatePipelineLoop(const Schedule& schedule, Location loc);

  // 生成prologue (填充pipeline)
  void generatePrologue(const Schedule& schedule, OpBuilder& builder);

  // 生成epilogue (清空pipeline)
  void generateEpilogue(const Schedule& schedule, OpBuilder& builder);
};
```

生成多buffer访问:
```cpp
// 为pipeline实现多buffer访问
// bufferIdx = (iteration + stage) % numBuffers
Value generateBufferAccess(Value original, unsigned stage,
                          unsigned numBuffers, OpBuilder& builder) {
  // 计算buffer索引
  // bufferIdx = (iteration + stage) % numBuffers
  auto iteration = getCurrentIteration();
  auto bufferIdx = builder.create<arith::AddIOp>(
      loc, iteration, builder.create<arith::ConstantIndexOp>(loc, stage));
  bufferIdx = builder.create<arith::RemSIOp>(
      loc, bufferIdx, builder.create<arith::ConstantIndexOp>(
                           loc, static_cast<int64_t>(numBuffers)));

  // 计算偏移
  auto offset = builder.create<arith::MulIOp>(
      loc, bufferIdx, builder.create<arith::ConstantIndexOp>(
                           loc, bufferSize));

  // 生成地址
  auto newPtr = builder.create<triton::AddPtrOp>(loc, originalPtr, offset);

  return newPtr;
}
```

---

## 4. MLIR Pass集成

### 4.1 Pass定义

```cpp
// third_party/ascend/include/TritonGraph/Passes/PipelineOptPass.h

class TritonPipelineOptPass
    : public PassWrapper<TritonPipelineOptPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonPipelineOptPass)

  TritonPipelineOptPass() = default;

  StringRef getArgument() const override {
    return "triton-ascend-pipeline-opt";
  }

  StringRef getDescription() const override {
    return "Optimize TTIR for pipeline parallelism on Ascend NPUs";
  }

  // Pass选项
  Option<unsigned> pipelineDepth{
      *this, "pipeline-depth",
      llvm::cl::desc("Pipeline depth (2, 3, or 4)"),
      llvm::cl::init(2)};

  Option<bool> enableMultiBuffer{
      *this, "enable-multibuffer",
      llvm::cl::desc("Enable multi-buffer optimization"),
      llvm::cl::init(true)};

  Option<bool> enableSyncOptimization{
      *this, "enable-sync-optimization",
      llvm::cl::desc("Optimize sync primitive placement"),
      llvm::cl::init(true)};

  // 入口函数
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // 1. 构建控制流图 (CFG)
    //    这是所有后续分析的基础
    ControlFlowGraphBuilder cfgBuilder;
    std::unique_ptr<InterProceduralCFG> icfg = cfgBuilder.buildModule(module);

    if (!icfg) {
      module.emitError("Failed to build CFG/ICFG");
      return signalPassFailure();
    }

    // 2. 指针分析：识别Memory Object和指针别名
    for (auto& funcCFG : icfg->getFunctionCFGs()) {
      PointerAnalysis ptrAnalysis(*funcCFG.second);
      if (failed(ptrAnalysis.analyze(module))) {
        module.emitWarning("Pointer analysis failed");
        continue;
      }

      // 3. 构建Memory SSA
      MemorySSABuilder memorySSA(*funcCFG.second, ptrAnalysis);
      if (failed(memorySSA.build(funcCFG.first))) {
        module.emitWarning("Memory SSA building failed");
        continue;
      }

      // 4. 构建稀疏值流图 (Sparse VFG)
      //    包含两类依赖：
      //    - 第一类：Memory依赖（基于Memory SSA）
      //    - 第二类：SSA依赖（直接use-def链）
      SparseVFGBuilder vfgBuilder(*funcCFG.second, memorySSA, ptrAnalysis);
      auto sparseVFG = vfgBuilder.build(funcCFG.first);

      if (!sparseVFG) {
        module.emitWarning("Failed to build sparse VFG");
        continue;
      }

      // 5. Pipeline机会分析（基于稀疏VFG）
      //    分析cube和vector操作之间的两类依赖
      DependencyAnalyzer analyzer(*sparseVFG, *funcCFG.second);
      auto opportunities = analyzer.findPipelineOpportunities();

      if (opportunities.empty()) {
        // 此函数没有优化机会，跳过
        continue;
      }

      // 6. Pipeline调度
      PipelineScheduler::Options opts;
      opts.pipelineDepth = pipelineDepth;
      opts.enableMultiBuffer = enableMultiBuffer;
      opts.minimizeSync = enableSyncOptimization;

      PipelineScheduler scheduler(*sparseVFG, *funcCFG.second, opts);
      auto schedule = scheduler.schedule(opportunities);
      if (failed(schedule)) {
        module.emitWarning("Failed to schedule pipeline");
        continue;
      }

      // 7. 生成优化的TTIR
      TTIRGenerator generator(&getContext());
      if (failed(generator.generate(*schedule, funcCFG.first))) {
        module.emitError("Failed to generate optimized TTIR");
        return signalPassFailure();
      }
    }

    // 标记Pass成功
    markAllAnalysesPreserved();
  }
};
```

### 4.2 Pass注册

```cpp
// third_party/ascend/lib/TritonGraph/Passes/PassRegistration.cpp

void registerTritonGraphPasses() {
  PassRegistration<TritonPipelineOptPass> registerPipelineOpt(
    []() -> std::unique_ptr<Pass> {
      return std::make_unique<TritonPipelineOptPass>();
    });
}

// 在third_party/ascend/triton_ascend.cc中注册
void registerAscendPasses() {
  // ... 其他pass

  // 注册图优化pass
  registerTritonGraphPasses();
}
```

### 4.3 在编译pipeline中使用

```cpp
// 在third_party/ascend/backend/compiler.cpp的ttir_to_linalg中

// 在标准优化pass之后，linalg转换之前添加pipeline优化
ascend.passes.ttir.add_triton_to_structure(...);
ascend.passes.ttir.add_discrete_mask_access_conversion(...);
ascend.passes.ttir.add_triton_to_annotation(...);

// 添加图优化pass (条件编译)
if (metadata["enable_pipeline_opt"]) {
  ascend.passes.ttir.add_pipeline_optimization(
      pm,
      metadata["pipeline_depth"],
      metadata["enable_multibuffer"]);
}

ascend.passes.ttir.add_triton_to_unstructure(...);
```

---

## 5. 构建系统配置

### 5.1 CMake配置

```cmake
# third_party/ascend/lib/TritonGraph/CMakeLists.txt

# TritonGraph库
add_mlir_dialect_library(TritonGraph
  # Parser
  Parser/TTIRParser.cpp

  # 核心数据结构
  IR/TritonGraph.cpp
  IR/Node.cpp
  IR/Edge.cpp
  Graph/ControlFlowGraph.cpp
  Graph/SparseValueFlowGraph.cpp

  # 解析器
  Parser/TTIRParser.cpp

  # 分析组件
  Analysis/MemoryObject.cpp
  Analysis/MemorySSA.cpp
  Analysis/PointerAnalysis.cpp

  # 构建器
  Builder/ControlFlowGraphBuilder.cpp
  Builder/MemorySSABuilder.cpp
  Builder/SparseVFGBuilder.cpp

  # Analyzer
  Analyzer/DependencyAnalyzer.cpp
  Analyzer/PipelineAnalyzer.cpp

  # Scheduler
  Scheduler/PipelineScheduler.cpp
  Scheduler/MultiBuffer.cpp

  # Generator
  Generator/TTIRGenerator.cpp
  Generator/SyncOpGenerator.cpp

  # Utils
  Utils/OpClassifier.cpp
  Utils/GraphUtils.cpp
  Utils/MLIRUtils.cpp

  DEPENDS
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRSupport
  MLIRAnalysis
  MLIRTritonDialect
  MLIRTritonGPUDialect
  TritonDialectIR
  TritonAscendDialect

  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRAnalysis
  MLIRTransformUtils
  TritonDialectIR
  TritonAscendDialect
)

# Pass库
add_mlir_dialect_library(TritonGraphPasses
  Passes/PipelineOptPass.cpp
  Passes/PassRegistration.cpp

  DEPENDS
  TritonGraph

  LINK_LIBS
  TritonGraph
  MLIRPass
)

# 测试
add_subdirectory(tests)

# 安装
install(TARGETS TritonGraph TritonGraphPasses
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
```

### 5.2 头文件组织

```
third_party/ascend/include/TritonGraph/
├── IR/
│   ├── TritonGraph.h          // 主图数据结构
│   ├── Node.h                 // 节点定义
│   └── Edge.h                 // 边定义
├── Graph/
│   ├── ControlFlowGraph.h     // CFG和ICFG
│   └── SparseValueFlowGraph.h // 稀疏值流图
├── Analysis/
│   ├── MemoryObject.h         // 内存对象定义
│   ├── MemorySSA.h            // Memory SSA结构
│   └── PointerAnalysis.h      // 指针和别名分析
├── Parser/
│   └── TTIRParser.h           // TTIR解析器
├── Builder/
│   ├── ControlFlowGraphBuilder.h   // CFG构建器
│   ├── MemorySSABuilder.h          // Memory SSA构建器
│   └── SparseVFGBuilder.h          // 稀疏VFG构建器
├── Analyzer/
│   ├── DependencyAnalyzer.h   // 依赖分析器
│   └── PipelineAnalyzer.h     // Pipeline机会分析
├── Scheduler/
│   ├── PipelineScheduler.h    // Pipeline调度器
│   └── MultiBuffer.h          // Multi-buffer优化
├── Generator/
│   └── TTIRGenerator.h        // TTIR代码生成器
├── Utils/
│   ├── OpClassifier.h         // 操作分类器
│   ├── GraphUtils.h           // 图算法工具
│   └── MLIRUtils.h            // MLIR辅助函数
└── Passes/
    └── PipelineOptPass.h      // MLIR优化Pass
```

---

## 6. 使用示例

### 6.1 作为MLIR Pass使用

```bash
# 1. 编译triton-ascend (包含TritonGraph)
cd /gemini/code/huawei/triton-ascend
mkdir build && cd build
cmake .. -G Ninja \
  -DTRITON_BUILD_PYTHON_MODULE=ON \
  -DTRITON_BUILD_TUTORIALS=OFF

# 编译
ninja triton-ascend-pipeline-opt

# 2. 使用pass优化TTIR
# 先dump出kernel.ttir.mlir
triton-adapter-opt kernel.ttir.mlir \
  --triton-to-structure \
  --triton-to-unstructure \
  -o kernel.before_pipeline.mlir

# 应用pipeline优化
triton-ascend-pipeline-opt kernel.before_pipeline.mlir \
  --pipeline-depth=2 \
  --enable-multibuffer=true \
  -o kernel.after_pipeline.mlir
```

### 6.2 作为库使用

```cpp
// 在Ascend后端编译器中使用

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "ascend/include/TritonGraph/Graph/ControlFlowGraph.h"
#include "ascend/include/TritonGraph/Graph/SparseValueFlowGraph.h"
#include "ascend/include/TritonGraph/Analysis/PointerAnalysis.h"
#include "ascend/include/TritonGraph/Analysis/MemorySSA.h"
#include "ascend/include/TritonGraph/Builder/SparseVFGBuilder.h"
#include "ascend/include/TritonGraph/Analyzer/DependencyAnalyzer.h"
#include "ascend/include/TritonGraph/Scheduler/PipelineScheduler.h"
#include "ascend/include/TritonGraph/Generator/TTIRGenerator.h"

LogicalResult optimizeForPipeline(triton::FuncOp func) {
  MLIRContext* ctx = func.getContext();

  // 1. 构建控制流图 (CFG)
  ControlFlowGraphBuilder cfgBuilder;
  auto cfg = cfgBuilder.build(func);

  if (!cfg) {
    return func.emitError("Failed to build CFG");
  }

  // 2. 指针分析：识别Memory Objects
  //    区分kernel参数、alloc buffer、指针运算
  PointerAnalysis ptrAnalysis(*cfg);
  if (failed(ptrAnalysis.analyze(func))) {
    return func.emitError("Pointer analysis failed");
  }

  // 3. 构建Memory SSA
  //    为每个内存对象维护def-use链
  MemorySSABuilder memorySSA(*cfg, ptrAnalysis);
  if (failed(memorySSA.build(func))) {
    return func.emitError("Memory SSA building failed");
  }

  // 4. 构建稀疏值流图 (Sparse VFG)
  //    包含两类依赖：
  //    - 第一类：Memory依赖（基于Memory SSA）
  //    - 第二类：SSA依赖（直接use-def链）
  SparseVFGBuilder vfgBuilder(*cfg, memorySSA, ptrAnalysis);
  auto sparseVFG = vfgBuilder.build(func);

  if (!sparseVFG) {
    return func.emitError("Failed to build sparse VFG");
  }

  // 5. 依赖分析：识别pipeline机会
  //    分析cube和vector操作之间的数据流
  DependencyAnalyzer analyzer(*sparseVFG, *cfg);
  auto opportunities = analyzer.findPipelineOpportunities();

  if (opportunities.empty()) {
    // 此函数没有优化机会
    return success();
  }

  // 6. Pipeline调度
  PipelineScheduler::Options opts;
  opts.pipelineDepth = 2;
  opts.enableMultiBuffer = true;

  PipelineScheduler scheduler(*sparseVFG, *cfg, opts);
  auto schedule = scheduler.schedule(opportunities);
  if (failed(schedule)) {
    return func.emitError("Failed to schedule pipeline");
  }

  // 7. 生成优化的TTIR
  TTIRGenerator generator(ctx);
  if (failed(generator.generate(*schedule, func))) {
    return func.emitError("Failed to generate optimized TTIR");
  }

  return success();
}
```

### 6.3 Triton Python API扩展

```python
# 在python/triton/backends/ascend/backend/compiler.py中添加

def ttir_to_pipeline(self, module, metadata, opt):
    """调用C++ pipeline优化"""
    import os

    # 检查是否启用pipeline优化
    enable_pipeline = metadata.get("enable_pipeline_opt", False)
    if not enable_pipeline:
        return str(module)

    pipeline_depth = metadata.get("pipeline_depth", 2)
    enable_multibuffer = metadata.get("enable_multibuffer", True)

    # 调用C++接口
    pm = ir.pass_manager(module.context)
    pm.enable_debug()

    ascend.passes.ttir.add_pipeline_optimization(
        pm,
        pipeline_depth,
        enable_multibuffer
    )
    pm.run(module)

    return str(module)
```

### 6.4 调试验证

```bash
# 1. 编译时启用调试信息
cmake .. -DCMAKE_BUILD_TYPE=Debug

# 2. 运行测试
./bin/triton-ascend-pipeline-opt \
  --pipeline-depth=2 \
  --enable-multibuffer \
  --print-ir-after-all \
  input.mlir -o output.mlir

# 3. 验证
# - 检查是否插入了 tt.sync_block_* 操作
# - 检查操作是否按pipeline stage重排
# - 验证正确性: 依赖关系是否保持

# 4. 性能测试
# 编译到二进制并运行，对比加速比
```

---

## 7. 开发计划

### 第一阶段: 基础框架和控制流分析 (预计: 1-2周)

**目标**: 实现TTIR解析、控制流分析和Memory SSA基础

任务:
1. 搭建项目结构和CMake配置
2. 实现OpClassifier (操作分类)
   - 识别cube操作 (triton::DotOp)
   - 识别vector操作 (elementwise, math, arith)
   - 识别memory操作 (load, store, alloc)
3. 实现TTIRParser基础版
   - 解析ModuleOp中的所有Operation
   - 按函数遍历基本块
   - 构建操作列表和统计
4. **实现ControlFlowGraphBuilder**
   - 构建函数级CFG
   - 识别基本块和边
   - 支持scf.if、scf.for等控制流结构
   - 创建ICFG (过程间控制流图)
5. **实现PointerAnalysis** (核心组件)
   - 识别MemoryObject (kernel参数、alloc buffer)
   - 分析指针别名 (base pointer + offset)
   - 支持tt.addptr等指针运算

交付物:
- 可以成功解析TTIR并构建CFG
- 提供CFG dump到dot的功能
- PointerAnalysis可以识别所有内存对象
- 单元测试覆盖主要代码路径

### 第二阶段: Memory SSA和稀疏值流图 (预计: 2-3周)

**目标**: 实现基于Memory SSA的稀疏数据流分析，构建值流图

任务:
1. **实现MemorySSABuilder** (核心创新)
   - 为每个store生成唯一的版本号（def）
   - 在控制流合并点插入Memory Phi节点
   - 为每个load确定到达定义（reaching def）
   - 支持循环结构的phi插入
2. **实现SparseVFGBuilder**
   - 将Memory SSA转换为稀疏值流图
   - 创建MemoryDef和MemoryUse节点
   - 连接def-use边（store -> load）
   - 构建SSA依赖边（传统use-def链）
3. 实现DependencyAnalyzer更新
   - 基于SparseVFG分析两类依赖
   - 识别cube-op和vector-op之间的数据流
   - 检测pipeline阻塞点（无法pipleling的依赖）
4. 依赖可视化
   - 将VFG导出为dot文件
   - 区分Memory依赖和SSA依赖的边样式

交付物:
- 完整的SparseVFG表示
- 准确的Memory依赖分析（store-load链）
- 可以可视化两类依赖关系
- 单元测试验证Memory SSA正确性

### 第三阶段: 调度算法 (预计: 3-4周)

**目标**: 实现pipeline调度器，生成执行计划

任务:
1. 实现List Scheduling算法
   - 优先级计算 (关键路径)
   - 就绪队列管理
   - Stage分配
2. 实现多buffer优化
   - Buffer分配策略
   - Ping-pong buffer生成
3. 实现同步原语插入
   - 识别需要同步的点
   - 插入tt.sync_block_wait/set
   - 优化同步位置 (最小化开销)
4. 调度验证
   - 验证调度保持依赖关系
   - 验证资源约束

交付物:
- 可以生成PipelineSchedule
- Schedule可视化 (各stage内容, 同步点)
- 性能估计报告

### 第四阶段: 代码生成和集成 (预计: 4-5周)

**目标**: 实现TTIR生成，集成到编译pipeline

任务:
1. 实现TTIRGenerator
   - Operation克隆
   - Buffer访问生成 (多buffer索引)
   - 同步操作生成
2. 实现MLIR Pass
   - 实现Pass接口
   - 注册到triton-ascend
3. 集成到编译器
   - 在ttir_to_linalg之前调用
   - 添加选项控制 (enable_pipeline_opt)
4. 端到端测试
   - 编译完整kernel
   - 验证正确性
   - 性能对比 (串行 vs pipeline)

交付物:
- 完整的端到端实现
- 可以使用triton-ascend-pipeline-opt工具
- 至少1个完整示例 (FlashAttention或MatMul)

### 第五阶段: 测试和优化 (预计: 5-6周)

**目标**: 全面测试和性能调优

任务:
1. 实现完整测试套件
   - 单元测试 (每个组件)
   - 集成测试 (端到端)
   - 边界情况测试
2. 性能benchmark
   - matmul + bias
   - FlashAttention
   - 自定义kernel
3. 调优和修复bug
   - 多buffer策略调优
   - 同步开销优化
4. 文档和示例
   - 详细文档
   - Tutorial
   - Best practices

交付物:
- 完整可用的框架
- 详细的测试报告
- 性能数据 (加速比)
- 用户使用文档

---

## 8. 关键技术问题

### 8.1 如何区分Cube和Vector操作

问题: TTIR中不是所有操作都有明确的属性标识运行在哪类核心

方案:

1. **Dialect-based分类**:
   ```cpp
   // 优先使用Dialect标识
   if (isa<triton::TritonDialect>(op)) {
     if (isa<triton::DotOp>(op)) return OpType::CUBE;
     if (isa<triton::LoadOp>(op) || isa<triton::StoreOp>(op))
       return OpType::MEMORY;
   }

   if (isa<arith::ArithDialect>(op) || isa<math::MathDialect>(op))
     return OpType::VECTOR;
   ```

2. **Attribute-based分类**:
   ```cpp
   // Ascend后端可以添加自定义属性
   if (auto attr = op->getAttrOfType<StringAttr>("triton.ascend.execution_unit")) {
     if (attr.getValue() == "cube") return OpType::CUBE;
     if (attr.getValue() == "vector") return OpType::VECTOR;
   }
   ```

3. **Pattern-based分类**:
   ```cpp
   // 基于操作模式识别
   // 例如: 大尺寸矩阵乘 -> cube
   //       elementwise小操作 -> vector
   ```

4. **可配置的分类表**:
   ```yaml
   # triton_graph_config.yaml
   cube_ops:
     - triton.dot
     - triton.matmul
     - ascend.cube.*

   vector_ops:
     - arith.addf
     - arith.mulf
     - math.exp
     - math.sqrt
     - triton.load
     - triton.store
   ```

### 8.2 如何处理循环

问题: Loop-carried dependency会阻塞pipeline

方案:

1. **循环展开**: 小循环直接展开，消除依赖
2. **循环分区**: 将循环分区，不同迭代的独立部分并行
3. **循环重排序**: 改变访问模式，消除依赖
4. **识别可减少依赖**: 例如: 加法可以重排，乘加链可以保留

关键: `hasLoopCarriedDependency(Operation* op, scf::ForOp loop)`

```cpp
// 检测操作是否依赖循环的迭代参数
bool hasLoopCarriedDependency(Operation* op, scf::ForOp loop) {
  // 获取循环的迭代参数 (iter_args)
  SmallVector<Value> iterArgs;
  loop.getRegionIterArgs(iterArgs);

  // 检查操作是否使用迭代参数
  for (Value operand : op->getOperands()) {
    if (is_contained(iterArgs, operand)) {
      // 操作依赖于迭代参数，可能有loop-carried dependency

      // 进一步检查: 是否是减少操作 (reduction)
      if (isReductionOp(op, loop)) {
        // 减少操作不能pipeline
        return true;
      }

      // 其他类型的依赖需要具体分析
      return true;
    }
  }
  return false;
}
```

### 8.3 如何插入同步原语

问题: 需要插入 `tt.sync_block_wait` 和 `tt.sync_block_set`

方案:

1. **Sync Block ID管理**:
   ```cpp
   // 为每个需要同步的依赖分配唯一的sync id
   struct SyncInfo {
     unsigned senderId;    // 发送者core id
     unsigned receiverId;  // 接收者core id
     unsigned eventId;     // 事件id
     Value value;          // 同步的值
   };
   ```

2. **生成SyncOp**:
   ```cpp
   // 在producer后生成 sync.set
   builder.create<triton::SyncBlockSetOp>(
       loc, sender, receiver, eventId);

   // 在consumer前生成 sync.wait
   builder.create<triton::SyncBlockWaitOp>(
       loc, sender, receiver, eventId);
   ```

3. **Sync优化**:
   - 合并多个同步 (如果同一个值被多次使用)
   - 移动sync位置以最大化重叠
   - 消除不需要的sync (如果consumer可以异步执行)

### 8.4 多buffer索引计算

问题: Pipeline中如何计算buffer索引

方案:

迭代 `i` 执行的操作使用 buffer: `buffer[(i + stage) % numBuffers]`

```cpp
// 计算buffer索引的MLIR代码
// bufferIdx = (iteration + stage) % numBuffers

Value computeBufferIdx(Value iteration, unsigned stage,
                      unsigned numBuffers, OpBuilder& builder) {
  Location loc = builder.getUnknownLoc();

  // stage offset
  auto stageVal = builder.create<arith::ConstantIndexOp>(loc, stage);
  auto stageIdx = builder.create<arith::AddIOp>(loc, iteration, stageVal);

  // wrap around
  auto numBufVal = builder.create<arith::ConstantIndexOp>(
      loc, static_cast<int64_t>(numBuffers));
  auto bufferIdx = builder.create<arith::RemUIOp>(loc, stageIdx, numBufVal);

  return bufferIdx;
}

// 访问buffer
// address = baseAddr + bufferIdx * bufferSize
Value accessMultiBuffer(Value baseAddr, Value bufferIdx,
                       unsigned bufferSize, OpBuilder& builder) {
  Location loc = builder.getUnknownLoc();

  auto bufferSizeVal = builder.create<arith::ConstantIndexOp>(
      loc, static_cast<int64_t>(bufferSize));
  auto offset = builder.create<arith::MulIOp>(loc, bufferIdx, bufferSizeVal);

  // 地址计算 (Triton AddPtrOp)
  auto newAddr = builder.create<triton::AddPtrOp>(loc, baseAddr, offset);

  return newAddr;
}
```

---

## 9. 测试策略

### 9.1 单元测试

使用MLIR的测试框架(mlir-opt + FileCheck):

```mlir
// test/TritonGraph/diag.mm.mlir

// RUN: triton-ascend-pipeline-opt %s \
// RUN:   --pipeline-depth=2 \
// RUN:   --enable-multibuffer \
// RUN:   | FileCheck %s

// 输入: 简单的matmul + bias
module {
  tt.func @matmul_bias(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>,
                      %bias: !tt.ptr<f32>, %C: !tt.ptr<f32>) {
    // tt.load A
    %a = tt.load %A : tensor<128x128xf16>

    // tt.load B
    %b = tt.load %B : tensor<128x128xf16>

    // tt.dot (cube)
    %c = tt.dot %a, %b : tensor<128x128xf32>

    // tt.load bias
    %bias_val = tt.load %bias : tensor<128xf32>

    // add bias (vector)
    %result = arith.addf %c, %bias_val : tensor<128x128xf32>

    // tt.store
    tt.store %C, %result : tensor<128x128xf32>

    tt.return
  }
}

// CHECK: tt.sync_block_set
// CHECK: tt.sync_block_wait
// CHECK-SAME: pipeline_depth = 2
```

### 9.2 集成测试

端到端编译和运行:

```python
# python/test/unit/test_graph_pipeline.py

def test_matmul_pipeline():
    """测试matmul的pipeline优化"""

    # 定义kernel
    @triton.jit
    def matmul_kernel(A, B, bias, C, M, N, K):
        # ... matmul + bias kernel ...
        pass

    # 启用pipeline优化
    os.environ["TRITON_ENABLE_PIPELINE_OPT"] = "1"
    os.environ["TRITON_PIPELINE_DEPTH"] = "2"

    # 编译并运行
    # 对比: 1) 不优化 2) pipeline优化
    # 验证结果相同
    # 验证性能提升
```

### 9.3 性能基准测试

```bash
# benchmark/bench_pipeline.py

# 测试用例:
# 1. FlashAttention
# 2. MatMul + BiasAdd + Gelu
# 3. Conv + BN + ReLU
# 4. 自定义融合kernel

# 测量指标:
# - 端到端执行时间
# - Cube Core利用率
# - Vector Core利用率
# - 内存带宽利用率
# - 加速比 (vs 串行版本)
```

---

## 10. 风险和缓解措施

### 10.1 技术风险

**风险1**: 依赖分析错误导致错误优化
- **缓解**: 严格的验证机制，对比优化前后语义
- **缓解**: 完整的测试覆盖，包括边界情况

**风险2**: 调度算法性能差，无法找到最优解
- **缓解**: 实现多种调度算法 (list scheduling, CP-based)
- **缓解**: 提供启发式参数调优

**风险3**: Sync primitive开销超过收益
- **缓解**: 精确的成本模型
- **缓解**: 只在高价值场景使用pipeline

### 10.2 进度风险

**风险1**: MLIR API使用复杂，学习曲线陡峭
- **缓解**: 参考MLIR现有pass实现
- **缓解**: 迭代开发，从简单例子开始

**风险2**: 图分析算法复杂度超过预期
- **缓解**: 使用成熟的图算法库 (LLVM GraphTraits)
- **缓解**: 简化第一版实现

---

## 11. 结论

本方案提供了一个完整的C++实现计划，通过MLIR API直接操作TTIR，构建图表示并进行流水线优化。关键亮点:

1. **原生集成**: 作为MLIR Pass集成到Triton-Ascend，符合编译器架构
2. **模块化设计**: Parser/Builder/Analyzer/Scheduler/Generator清晰分离
3. **实用性**: 直接解决cube/vector并行问题，有明确的性能收益
4. **可扩展**: 框架支持后续添加更多分析和优化

通过5-6周的开发，可以实现一个完整可用的图编译优化框架，预期为关键kernel (FlashAttention, MatMul等)带来15-25%的性能提升。

---

## 附录A: 参考资料

1. **MLIR官方文档**: https://mlir.llvm.org/docs/
2. **MLIR Pass开发**: https://mlir.llvm.org/docs/Tutorials/Quickstart/
3. **Triton MLIR Dialect**: ./include/triton/Dialect/Triton/IR/
4. **Ascend MLIR Dialect**: ./third_party/ascend/include/Dialect/
5. **LLVM GraphTraits**: https://llvm.org/docs/ProgrammersManual.html#iterating-over-nodes-in-a-dfs
6. **List Scheduling**: https://en.wikipedia.org/wiki/List_scheduling
7. **Software Pipelining**: https://en.wikipedia.org/wiki/Software_pipelining

---

## 附录B: 术语表

- **TTIR**: Triton Tensor IR, Triton的中间表示
- **MLIR**: Multi-Level Intermediate Representation, LLVM的多级IR框架
- **Cube**: Ascend NPU的AI Core, 用于密集型计算
- **Vector**: Ascend NPU的Vector Core, 用于向量操作
- **DFG**: Dataflow Graph, 数据流图
- **Pass**: MLIR的编译优化遍
- **Sync Primitive**: 同步原语 (如tt.sync_block_wait/set)
- **Multi-Buffer**: 多buffer技术, 用于重叠计算和通信
- **Pipeline**: 流水线并行，多个操作重叠执行
- **Stage**: Pipeline的一个阶段
- **Schedule**: 调度计划，操作到stage的分配
- **RAW**: Read-After-Write依赖
- **WAR**: Write-After-Read依赖
- **WAW**: Write-After-Write依赖
