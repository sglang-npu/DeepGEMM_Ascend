# Memory SSA 设计方案

## 版本信息
- 创建日期：2026-03-16
- 版本：v1.0
- 作者：Claude Code

## 1. 概述

### 1.1 目标
为Triton IR构建Memory SSA形式的数据流图，用于分析tensor/pointer的数据依赖关系，支持编译器的依赖分析、优化和并行化调度。

### 1.2 主要贡献
- 在Control Flow Graph基础上构建Memory SSA
- 统一处理传统SSA和Memory SSA的查询接口
- 支持scf.if和scf.for的控制流处理
- 保持与传统SSA的兼容性

## 2. 核心数据结构

### 2.1 TensorObject - Tensor对象定义

```cpp
class TensorObject {
public:
  // Tensor类型分类
  enum class TensorKind {
    GLOBAL_MEMORY,  // 全局内存tensor（如gm_obj）
    L1,             // CUBE片上1级缓存
    L0,             // CUBE片上0级缓存
    UB,             // VECTOR片上缓存
  };

  // Compute类型（用于指令分类）
  enum class ComputeType {
    CUBE,    // shape为2维
    VECTOR,  // shape为1维
    SCALAR   // 其他标量
  };

  // 构造函数
  TensorObject(StringRef name, ArrayRef<int64_t> shape, Type type,
               TensorKind kind = TensorKind::GLOBAL_MEMORY)
      : name(name.str()), shape(shape.begin(), shape.end()),
        type(type), kind(kind) {}

  // 获取tensor名称
  const std::string& getName() const { return name; }

  // 获取shape
  ArrayRef<int64_t> getShape() const { return shape; }

  // 获取MLIR类型
  Type getType() const { return type; }

  // 获取tensor种类
  TensorKind getKind() const { return kind; }

  // 根据shape确定compute类型
  ComputeType getComputeType() const {
    if (shape.size() == 2) return ComputeType::CUBE;
    if (shape.size() == 1) return ComputeType::VECTOR;
    return ComputeType::SCALAR;
  }

  // 获取维度数
  size_t getRank() const { return shape.size(); }

private:
  std::string name;          // Tensor名称，如"gm_obj_0"
  SmallVector<int64_t> shape;
  Type type;
  TensorKind kind;
};
```

### 2.2 Definition - Memory Definition

```cpp
class Definition {
public:
  // 构造函数
  Definition(TensorObject* tensor, Operation* defOp, unsigned version = 0)
      : tensor(tensor), defOp(defOp), version(version) {}

  // 获取tensor对象
  TensorObject* getTensor() const { return tensor; }

  // 获取创建该definition的操作
  // - 对于入参，返回nullptr
  // - 对于其他操作，返回对应的Operation指针
  Operation* getDefOp() const { return defOp; }

  // 获取版本号（函数内唯一序数）
  unsigned getVersion() const { return version; }

  // 获取唯一标识
  std::string getId() const {
    return tensor->getName() + "," +
           (defOp ? std::to_string(version) : "param");
  }

  // 判断是否是入参
  bool isParameter() const { return defOp == nullptr; }

private:
  TensorObject* tensor;      // 对应的tensor对象
  Operation* defOp;         // 创建该definition的操作
  unsigned version;         // 版本号（函数内递增）
};
```

### 2.3 Use - Memory Use

```cpp
class Use {
public:
  // 构造函数
  Use(Definition* definition, Operation* userOp, unsigned operandIdx)
      : definition(definition), userOp(userOp), operandIdx(operandIdx) {}

  // 获取使用的definition
  Definition* getDefinition() const { return definition; }

  // 获取使用该definition的操作
  Operation* getUserOp() const { return userOp; }

  // 获取operand序号
  unsigned getOperandIdx() const { return operandIdx; }

  // 获取Value（从userOp的operand）
  Value getValue() const {
    return userOp->getOperand(operandIdx);
  }

private:
  Definition* definition;    // 使用的definition
  Operation* userOp;        // 使用该definition的操作
  unsigned operandIdx;      // operand序号
};
```

### 2.4 MemorySSAInfo - 指令的Memory SSA信息

```cpp
struct MemorySSAInfo {
  // 指令的operands使用的definitions
  SmallVector<Use> uses;

  // 指令的results创建的definitions
  SmallVector<Definition*> definitions;

  // Alias信息（仅对pointer相关操作）
  struct AliasInfo {
    Value aliasee;              // 别名的源value
    TensorObject* baseTensor;   // 对应的tensor对象
  };
  std::optional<AliasInfo> aliasInfo;

  // 快速查询接口
  bool hasDefinition(Value value) const;
  Definition* getDefinition(Value value) const;
  bool hasUse(Value value) const;
  SmallVector<Use> getUses(Value value) const;

  // 判断是否创建了新的definition
  bool hasNewDefinitions() const { return !definitions.empty(); }

  // 遍定义
  void forEachDefinition(llvm::function_ref<void(Definition*)> func) const {
    for (Definition* def : definitions) {
      if (def) func(def);
    }
  }

  void forEachUse(llvm::function_ref<void(const Use&)> func) const {
    for (const Use& use : uses) {
      if (use.getDefinition()) func(use);
    }
  }
};
```

### 2.5 LoopPhiInfo - 循环Phi信息（scf.for iter_args专用）

```cpp
struct LoopPhiInfo {
  // Phi类型
  enum Type {
    ITER_ARG,   // scf.for的iter_arg
    IF_RESULT,  // scf.if的result
    WHILE_ARG   // scf.while的arg
  };

  Type type;
  BasicBlock *loopHeader;  // 循环头基本块

  // Phi值的来源
  struct {
    Definition *initialValue;  // 初始值（初始iteration）
    Definition *yieldValue;    // yield的值（后续iteration）
  } comingFrom;

  // 是否是第一次迭代
  bool isInitial() const { return comingFrom.yieldValue == nullptr; }

  // 获取当前definition（根据上下文决定）
  Definition* getCurrentDefinition(int iteration) const {
    return (iteration == 0) ? comingFrom.initialValue
                           : comingFrom.yieldValue;
  }
};
```

### 2.6 DataFlowInfo - 统一数据流信息

```cpp
class DataFlowInfo {
public:
  // 为函数入口创建参数定义
  void createParameterDefinitions(triton::FuncOp func);

  // Memory SSA接口
  Definition* getMemoryDefinition(Value value) const;
  void addMemoryDefinition(Value value, Definition* def);

  SmallVector<Use> getMemoryUses(Value value) const;
  void addMemoryUse(Value value, const Use& use);

  // 传统SSA接口（复用MLIR原生功能）
  Operation* getSSADefinition(Value value) const {
    return value.getDefiningOp();
  }

  SmallVector<OpOperand*> getSSAUses(Value value) const {
    SmallVector<OpOperand*> result;
    for (OpOperand& use : value.getUses()) {
      result.push_back(&use);
    }
    return result;
  }

  // 循环Phi接口（scf.for iter_args）
  void addLoopPhi(Value value, const LoopPhiInfo& phiInfo) {
    loopPhis[value] = phiInfo;
  }

  LoopPhiInfo& getLoopPhi(Value value) {
    return loopPhis[value];
  }

  // 统一查询接口
  struct DataFlowResult {
    enum class ResultKind {
      MEMORY_SSA,     // Memory SSA结果（tensor/pointer）
      SSA,            // 传统SSA结果（标量）
      NONE            // 无数据流信息
    };

    ResultKind kind;
    Operation* originOp;         // 查询的操作
    Definition* definition;      // MEMORY_SSA时使用
    Operation* ssaDefinition;    // SSA时使用
    SmallVector<OpOperand*> uses; // 所有uses
  };

  DataFlowResult queryDataFlow(Value value) const {
    // 1. 优先查询Memory SSA
    if (Definition* def = getMemoryDefinition(value)) {
      return {
        .kind = DataFlowResult::MEMORY_SSA,
        .originOp = def->getDefOp(),
        .definition = def,
        .uses = getSSAUses(value)
      };
    }

    // 2. 查询传统SSA
    if (Operation* defOp = value.getDefiningOp()) {
      return {
        .kind = DataFlowResult::SSA,
        .originOp = defOp,
        .ssaDefinition = defOp,
        .uses = getSSAUses(value)
      };
    }

    // 3. 入参
    return {
      .kind = DataFlowResult::SSA,
      .originOp = nullptr,
      .ssaDefinition = nullptr,
      .uses = getSSAUses(value)
    };
  }

private:
  // Memory SSA映射
  DenseMap<Value, Definition*> memoryDefinitions;
  DenseMap<Value, SmallVector<Use>> memoryUses;

  // Loop Phi映射（用于scf.for iter_args）
  DenseMap<Value, LoopPhiInfo> loopPhis;

  // Tensor对象映射
  DenseMap<Value, TensorObject*> tensorObjects;
};
```

### 2.7 扩展Instruction结构

```cpp
// 在ControlFlowGraph.h中扩展Instruction类
class Instruction {
public:
  // ... 现有方法 ...

  // 新增：数据流信息
  const MemorySSAInfo& getMemorySSAInfo() const { return memorySSAInfo; }
  MemorySSAInfo& getMemorySSAInfo() { return memorySSAInfo; }

private:
  // ... 现有字段 ...

  // 新增
  MemorySSAInfo memorySSAInfo;
};
```

## 3. 核心组件

### 3.1 AliasAnalysis - Alias分析器

#### 3.1.1 功能说明
分析pointer的alias关系，跟踪全局内存对象（如gm_obj）的别名链。

#### 3.1.2 数据结构

```cpp
class AliasAnalysis {
public:
  // 为整个CFG分析pointer别名
  void analyzePointerAliases(ControlFlowGraph &cfg);

  // 获取Value的base pointer
  Value getBasePointer(Value ptr) const;

  // 获取Value对应的TensorObject
  TensorObject* getTensorObject(Value value) const {
    auto it = baseTensorMap.find(value);
    return (it != baseTensorMap.end()) ? it->second : nullptr;
  }

  // 判断是否指向同一tensor
  bool mayAlias(Value ptr1, Value ptr2) const {
    return getBasePointer(ptr1) == getBasePointer(ptr2);
  }

  // 添加alias关系
  void addAlias(Value ptr, Value base, TensorObject* tensor) {
    aliasMap[ptr] = base;
    baseTensorMap[ptr] = tensor;
  }

  // 判断是否是指针类型
  static bool isPointerType(Type type) {
    return type.isa<triton::PointerType>();
  }

private:
  DenseMap<Value, Value> aliasMap;              // ptr -> base ptr
  DenseMap<Value, TensorObject*> baseTensorMap; // value -> tensor
};
```

#### 3.1.3 分析算法

```cpp
void AliasAnalysis::analyzePointerAliases(ControlFlowGraph& cfg) {
  // 步骤1: 识别全局内存参数
  for (BlockArgument arg : cfg.getFunction().getArguments()) {
    Type argType = arg.getType();
    if (isPointerType(argType) && isGlobalMemory(argType)) {
      // 为入参创建tensor对象
      TensorObject* tensor = new TensorObject(
          getTensorName(arg), getShape(arg), argType,
          TensorObject::TensorKind::GLOBAL_MEMORY);

      // 记录[gm_obj, param]
      addAlias(arg, arg, tensor);
    }
  }

  // 步骤2: 分析alias操作
  cfg.traverse([&](BasicBlock& bb) {
    for (const auto& instPtr : bb.getInstructions()) {
      Instruction* inst = instPtr.get();
      Operation* op = inst->getOperation();

      if (auto addptrOp = dyn_cast<triton::AddPtrOp>(op)) {
        // %new = tt.addptr %ptr, %offset
        Value ptr = addptrOp.getPtr();
        Value result = addptrOp.getResult();

        // result是ptr的alias，指向同一个tensor
        Value basePtr = getBasePointer(ptr);
        TensorObject* tensor = getTensorObject(basePtr);

        if (tensor) {
          addAlias(result, basePtr, tensor);

          // 记录alias信息到instruction
          inst->getMemorySSAInfo().aliasInfo = {
              .aliasee = ptr,
              .baseTensor = tensor
          };
        }
      }
      else if (auto makeTensorPtrOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
        // %tensor_ptr = tt.make_tensor_ptr %base, ...
        Value basePtr = makeTensorPtrOp.getBase();
        Value result = makeTensorPtrOp.getResult();

        TensorObject* tensor = getTensorObject(basePtr);

        if (tensor) {
          addAlias(result, basePtr, tensor);

          // 记录alias信息
          inst->getMemorySSAInfo().aliasInfo = {
              .aliasee = basePtr,
              .baseTensor = tensor
          };
        }
      }
    }
  });
}
```

### 3.2 MemorySSABuilder - Memory SSA构建器

#### 3.2.1 功能说明
构建整个CFG的Memory SSA信息，包括：
- 创建tensor definitions
- 创建uses
- 处理scf.if的phi节点
- 处理scf.for的iter_args

#### 3.2.2 数据结构

```cpp
class MemorySSABuilder {
public:
  MemorySSABuilder(ControlFlowGraph &cfg, AliasAnalysis &aliasAnalysis,
                   DataFlowInfo &dataFlowInfo)
      : cfg(cfg), aliasAnalysis(aliasAnalysis),
        dataFlowInfo(dataFlowInfo), nextVersionId(1) {}

  // 构建整个CFG的Memory SSA
  void build();

  // 处理单个BasicBlock
  void processBasicBlock(BasicBlock *bb);

  // 处理scf.if
  void processIfBlock(BasicBlock *ifCondBB, BasicBlock *thenExitBB,
                      BasicBlock *elseExitBB, BasicBlock *mergeBB);

  // 处理scf.for
  void processForBlock(scf::ForOp forOp, BasicBlock *forCondBB,
                       BasicBlock *loopBodyEntryBB, BasicBlock *loopExitBB);

private:
  ControlFlowGraph &cfg;
  AliasAnalysis &aliasAnalysis;
  DataFlowInfo &dataFlowInfo;
  size_t nextVersionId;  // version递增

  // 创建tensor definition
  Definition* createTensorDefinition(TensorObject* tensor, Operation* op) {
    unsigned version = isParameter(op) ? 0 : nextVersionId++;
    auto* def = new Definition(tensor, op, version);
    allDefinitions.push_back(def);
    return def;
  }

  // 创建use
  Use createUse(Definition* def, Operation* userOp, unsigned operandIdx) {
    return Use(def, userOp, operandIdx);
  }

  // 判断是否是写入操作
  bool isTensorWriter(Operation* op) const {
    // 包括：tt.load（读取）、tt.store（写入）、tt.trans、tt.dot等
    return isa<triton::LoadOp, triton::StoreOp, triton::TransOp,
               triton::DotOp, triton::MakeTensorPtrOp>(op);
  }

  // 判断是否读取操作
  bool isTensorReader(Operation* op) const {
    return isa<triton::LoadOp, triton::DotOp>(op);
  }

  // 根据操作创建tensor对象
  TensorObject* createTensorObject(Operation* op);

  // 判断是否是入参
  bool isParameter(Operation* op) const {
    return op == nullptr || isa<BlockArgument>(op);
  }

  // 判断是否需要创建新的tensor name
  bool shouldCreateNewTensorName(Operation* op) const;

  // 获取新tensor name
  std::string getNewTensorName(Operation* op) const;

  // 所有definitions
  SmallVector<Definition*> allDefinitions;
};
```

#### 3.2.3 构建流程

```cpp
void MemorySSABuilder::build() {
  // 步骤1: 拓扑排序，确定处理顺序
  std::vector<BasicBlock*> topoOrder = getTopologicalOrder(cfg);

  // 步骤2: 初始化函数的参数
  createParameterDefinitions();

  // 步骤3: 按拓扑序遍历每个BasicBlock
  for (BasicBlock *bb : topoOrder) {
    // 在block入口处合并前驱数据流
    if (bb->getNumPredecessors() > 1) {
      mergePredecessorDataFlow(bb);
    }

    // 处理block
    processBasicBlock(bb);
  }
}

void MemorySSABuilder::createParameterDefinitions() {
  triton::FuncOp func = cfg.getFunction();

  for (BlockArgument arg : func.getArguments()) {
    Type argType = arg.getType();

    // 检查是否是tensor类型
    if (isTensorType(argType)) {
      // 创建tensor对象
      std::string tensorName = "tensor_" + std::to_string(arg.getArgNumber());
      TensorObject* tensor = new TensorObject(
          tensorName, getShape(argType), argType);

      // 为入参创建definition
      // 入参的defOp为nullptr，version为0
      Definition* def = createTensorDefinition(tensor, nullptr);

      // 记录到dataFlowInfo
      dataFlowInfo.addMemoryDefinition(arg, def);

      llvm::errs() << "Created parameter definition: " << def->getId() << "\n";
    }
  }
}

void MemorySSABuilder::processBasicBlock(BasicBlock *bb) {
  // 处理block内的所有指令
  for (auto& instPtr : bb->getInstructions()) {
    Instruction* inst = instPtr.get();
    processInstruction(inst);
  }
}

void MemorySSABuilder::processInstruction(Instruction *inst) {
  Operation* op = inst->getOperation();
  MemorySSAInfo& ssaInfo = inst->getMemorySSAInfo();

  // 1. 处理operands：创建uses
  for (OpOperand& operand : op->getOpOperands()) {
    Value operandValue = operand.get();
    unsigned operandIdx = operand.getOperandNumber();

    // 检查是否是tensor/pointer类型
    if (isTensorType(operandValue.getType()) ||
        aliasAnalysis.isPointerType(operandValue.getType())) {

      // 查找operand的definition
      Definition* def = dataFlowInfo.getMemoryDefinition(operandValue);

      if (def) {
        // 创建use
        Use use = createUse(def, op, operandIdx);
        ssaInfo.uses.push_back(use);

        // 记录到全局map
        dataFlowInfo.addMemoryUse(operandValue, use);

        llvm::errs() << "  Use: " << def->getId() << " in " << op->getName()
                     << " [operand #" << operandIdx << "]\n";
      }
    }
  }

  // 2. 处理results：创建definitions
  for (Value result : op->getResults()) {
    Type resultType = result.getType();

    // 检查是否是tensor类型
    if (isTensorType(resultType)) {
      // 创建tensor对象
      TensorObject* tensor = createTensorObject(op);

      // 判断是否是写入操作
      if (isTensorWriter(op)) {
        // 创建新definition
        Definition* newDef = createTensorDefinition(tensor, op);
        ssaInfo.definitions.push_back(newDef);

        // 记录到全局map
        dataFlowInfo.addMemoryDefinition(result, newDef);

        llvm::errs() << "  Definition: " << newDef->getId() << " created by "
                     << op->getName() << "\n";

        // 判断是否需要新tensor name
        if (shouldCreateNewTensorName(op)) {
          tensor->setName(getNewTensorName(op));
        }
      }
    } else {
      // 标量result，不创建Memory SSA definition
      ssaInfo.definitions.push_back(nullptr);
    }
  }

  // 3. 特殊处理alias操作
  if (auto addptrOp = dyn_cast<triton::AddPtrOp>(op)) {
    handleAddPtrOp(addptrOp, inst);
  }
  else if (auto makeTensorPtrOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
    handleMakeTensorPtrOp(makeTensorPtrOp, inst);
  }
  else if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    handleLoadOp(loadOp, inst);
  }
}

void MemorySSABuilder::handleAddPtrOp(triton::AddPtrOp op, Instruction* inst) {
  Value ptr = op.getPtr();
  Value result = op.getResult();

  // result是ptr的alias，指向同一个tensor
  Value basePtr = aliasAnalysis.getBasePointer(ptr);
  TensorObject* tensor = aliasAnalysis.getTensorObject(basePtr);

  if (tensor) {
    // result与ptr指向同一个definition
    Definition* baseDef = dataFlowInfo.getMemoryDefinition(ptr);
    if (baseDef) {
      dataFlowInfo.addMemoryDefinition(result, baseDef);

      // 记录alias信息
      inst->getMemorySSAInfo().aliasInfo = {
          .aliasee = ptr,
          .baseTensor = tensor
      };

      llvm::errs() << "  Alias: " << result << " -> " << ptr
                   << " [" << tensor->getName() << "]\n";
    }
  }
}

void MemorySSABuilder::handleMakeTensorPtrOp(triton::MakeTensorPtrOp op,
                                              Instruction* inst) {
  Value basePtr = op.getBase();
  Value result = op.getResult();

  TensorObject* tensor = aliasAnalysis.getTensorObject(basePtr);

  if (tensor) {
    // result与basePtr指向同一个tensor
    Definition* baseDef = dataFlowInfo.getMemoryDefinition(basePtr);
    if (baseDef) {
      dataFlowInfo.addMemoryDefinition(result, baseDef);

      // 记录alias信息
      inst->getMemorySSAInfo().aliasInfo = {
          .aliasee = basePtr,
          .baseTensor = tensor
      };
    }
  }
}

void MemorySSABuilder::handleLoadOp(triton::LoadOp op, Instruction* inst) {
  Value ptr = op.getPtr();
  Value result = op.getResult();

  // load操作：从ptr加载数据到result tensor
  // result是ptr指向的tensor的equivalence

  if (auto tensorPtrType = ptr.getType().dyn_cast<triton::PointerType>()) {
    if (tensorPtrType.getPointeeType().isa<RankedTensorType>()) {
      // 这是tt.load %tensor_ptr操作
      TensorObject* tensor = aliasAnalysis.getTensorObject(ptr);

      if (tensor) {
        // result是[ptr的base_tensor, 0]的equivalence
        Definition* ptrDef = dataFlowInfo.getMemoryDefinition(ptr);
        if (ptrDef) {
          // result使用与ptr相同的definition
          dataFlowInfo.addMemoryDefinition(result, ptrDef);

          llvm::errs() << "  Load: " << result << " is equivalence of "
                       << ptrDef->getId() << "\n";
        }
      }
    }
  }
}

void MemorySSABuilder::processIfBlock(BasicBlock *ifCondBB,
                                      BasicBlock *thenExitBB,
                                      BasicBlock *elseExitBB,
                                      BasicBlock *mergeBB) {
  // 获取if指令
  scf::IfOp ifOp = cast<scf::IfOp>(ifCondBB->getInstruction(0)->getOperation());

  // 为每个result创建phi节点
  for (size_t i = 0; i < ifOp.getNumResults(); ++i) {
    Value ifResult = ifOp.getResult(i);

    // 从then区域获取yield的value
    Operation *thenYield = getYieldOp(ifOp.getThenRegion());
    Value thenValue = thenYield->getOperand(i);
    Definition *thenDef = dataFlowInfo.getMemoryDefinition(thenValue);

    // 从else区域获取yield的value
    Operation *elseYield = getYieldOp(ifOp.getElseRegion());
    Value elseValue = elseYield->getOperand(i);
    Definition *elseDef = dataFlowInfo.getMemoryDefinition(elseValue);

    // 在mergeBB创建phi，合并两个前驱
    if (thenDef || elseDef) {
      TensorObject* tensor = thenDef ? thenDef->getTensor()
                                      : elseDef->getTensor();
      Definition* phiDef = createTensorDefinition(tensor, ifOp.getOperation());

      // 记录if result的definition
      dataFlowInfo.addMemoryDefinition(ifResult, phiDef);

      // 为phi节点的operands创建uses
      if (thenDef) {
        Use thenUse(thenDef, ifOp.getOperation(), /*operandIdx=*/i);
        dataFlowInfo.addMemoryUse(thenValue, thenUse);
      }
      if (elseDef) {
        Use elseUse(elseDef, ifOp.getOperation(), /*operandIdx=*/i);
        dataFlowInfo.addMemoryUse(elseValue, elseUse);
      }

      llvm::errs() << "  Phi: " << phiDef->getId() << " for if result #"
                   << i << "\n";
    }
  }
}

void MemorySSABuilder::processForBlock(scf::ForOp forOp,
                                       BasicBlock *forCondBB,
                                       BasicBlock *loopBodyEntryBB,
                                       BasicBlock *loopExitBB) {
  // 处理scf.for的iter_args
  unsigned numIterArgs = forOp.getNumIterOperands();

  for (unsigned i = 0; i < numIterArgs; i++) {
    Value iterArg = forOp.getRegionIterArg(i);
    Value initValue = forOp.getIterOperands()[i];

    // 查找initValue的definition
    Definition *initDef = dataFlowInfo.getMemoryDefinition(initValue);

    if (initDef) {
      // iter_arg使用同一个definition，不创建新版本
      dataFlowInfo.addMemoryDefinition(iterArg, initDef);

      // 创建LoopPhiInfo（用于跟踪循环依赖）
      LoopPhiInfo phiInfo = {
          .type = LoopPhiInfo::ITER_ARG,
          .loopHeader = forCondBB,
          .comingFrom = {.initialValue = initDef, .yieldValue = nullptr}
      };

      dataFlowInfo.addLoopPhi(iterArg, phiInfo);

      // 为initValue创建use
      Use initUse(initDef, forOp, /*operandIdx=*/i);
      dataFlowInfo.addMemoryUse(initValue, initUse);

      llvm::errs() << "  IterArg #" << i << ": " << iterArg
                   << " -> " << initDef->getId() << "\n";
    }
  }

  // 处理yield操作（在循环体中）
  Operation *yieldOp = getYieldOp(forOp.getRegion());
  if (yieldOp) {
    for (unsigned i = 0; i < yieldOp->getNumOperands(); i++) {
      Value yieldedValue = yieldOp->getOperand(i);
      Value iterArg = forOp.getRegionIterArg(i);

      Definition *yieldedDef = dataFlowInfo.getMemoryDefinition(yieldedValue);

      if (yieldedDef) {
        // 更新LoopPhiInfo
        LoopPhiInfo &phiInfo = dataFlowInfo.getLoopPhi(iterArg);
        phiInfo.comingFrom.yieldValue = yieldedDef;

        // 为yieldedValue创建use
        Use yieldUse(yieldedDef, yieldOp, /*operandIdx=*/i);
        dataFlowInfo.addMemoryUse(yieldedValue, yieldUse);

        llvm::errs() << "    Yield #" << i << ": " << yieldedValue
                     << " updates iter_arg\n";
      }
    }
  }
}

void MemorySSABuilder::mergePredecessorDataFlow(BasicBlock *bb) {
  // 合并前驱的基本块数据流
  // 对于多前驱的基本块，需要处理phi节点

  llvm::errs() << "Merging data flow for BB" << bb->getId() << "\n";

  for (BasicBlock *pred : bb->getPredecessors()) {
    llvm::errs() << "  from BB" << pred->getId() << "\n";
  }
}
```

### 3.3 DataFlowGraph - 数据流图

#### 3.3.1 功能说明
构建统一的数据流图，提供传统SSA和Memory SSA的统一查询接口。

#### 3.3.2 数据结构

```cpp
class DataFlowGraph {
public:
  explicit DataFlowGraph(ControlFlowGraph &cfg)
      : cfg(cfg) {}

  // 构建数据流信息
  void build();

  // 查询Value的数据流信息
  DataFlowInfo::DataFlowResult queryDataFlow(Value value) const {
    return dataFlowInfo.queryDataFlow(value);
  }

  // 导出数据流信息到JSON
  void exportDataFlowToJSON(raw_ostream &os) const;

  // 获取所有Memory SSA definitions
  SmallVector<Definition*> getAllDefinitions() const {
    return allDefinitions;
  }

  // 获取use-def链
  SmallVector<Use> getUses(Definition* def) const {
    return defUseMap.lookup(def);
  }

  // 获取def-use链
  SmallVector<Use> getUsesByUserOp(Operation* userOp) const {
    SmallVector<Use> result;
    for (const auto& entry : defUseMap) {
      for (const Use& use : entry.second) {
        if (use.getUserOp() == userOp) {
          result.push_back(use);
        }
      }
    }
    return result;
  }

private:
  ControlFlowGraph &cfg;

  // 组件
  std::unique_ptr<AliasAnalysis> aliasAnalysis;
  std::unique_ptr<MemorySSABuilder> memorySSABuilder;

  // 数据
  DataFlowInfo dataFlowInfo;
  SmallVector<Definition*> allDefinitions;

  // Use-Def映射（def -> uses）
  DenseMap<Definition*, SmallVector<Use>> defUseMap;

  // 拓扑排序遍历
  void traverseInTopologicalOrder();
};
```

#### 3.3.3 构建流程

```cpp
void DataFlowGraph::build() {
  // 步骤1: 构建Alias分析
  aliasAnalysis = std::make_unique<AliasAnalysis>();
  aliasAnalysis->analyzePointerAliases(cfg);

  llvm::errs() << "=== Alias Analysis Complete ===\n";

  // 步骤2: 构建Memory SSA
  memorySSABuilder = std::make_unique<MemorySSABuilder>(
      cfg, *aliasAnalysis, dataFlowInfo);
  memorySSABuilder->build();

  llvm::errs() << "=== Memory SSA Build Complete ===\n";

  // 步骤3: 收集所有definitions
  collectDefinitions();

  // 步骤4: 构建def-use图
  buildDefUseGraph();

  llvm::errs() << "=== Data Flow Graph Build Complete ===\n";
}

void DataFlowGraph::collectDefinitions() {
  // 遍历所有指令，收集definitions
  cfg.traverse([&](BasicBlock& bb) {
    for (const auto& instPtr : bb.getInstructions()) {
      const Instruction* inst = instPtr.get();
      const MemorySSAInfo& ssaInfo = inst->getMemorySSAInfo();

      for (Definition* def : ssaInfo.definitions) {
        if (def) {
          allDefinitions.push_back(def);
        }
      }
    }
  });

  llvm::errs() << "Total definitions: " << allDefinitions.size() << "\n";
}

void DataFlowGraph::buildDefUseGraph() {
  // 构建def -> uses的映射
  cfg.traverse([&](BasicBlock& bb) {
    for (const auto& instPtr : bb.getInstructions()) {
      const Instruction* inst = instPtr.get();
      const MemorySSAInfo& ssaInfo = inst->getMemorySSAInfo();

      // 遍历所有uses
      for (const Use& use : ssaInfo.uses) {
        Definition* def = use.getDefinition();
        if (def) {
          defUseMap[def].push_back(use);
        }
      }
    }
  });

  // 打印统计信息
  size_t totalUses = 0;
  for (const auto& entry : defUseMap) {
    totalUses += entry.second.size();
  }

  llvm::errs() << "Total uses: " << totalUses << "\n";
  llvm::errs() << "Def-use chains: " << defUseMap.size() << "\n";
}
```

## 4. 使用示例

### 4.1 构建CFG和Memory SSA

```cpp
#include "TritonToCFG/ControlFlowGraph.h"
#include "TritonToCFG/DataFlowGraph.h"

void analyzeFunction(triton::FuncOp func) {
  // 1. 构建CFG
  cfg::ControlFlowGraphBuilder builder;
  auto cfg = builder.build(func);

  // 2. 构建Memory SSA
  cfg::DataFlowGraph dataFlowGraph(*cfg);
  dataFlowGraph.build();

  // 3. 查询数据流信息
  func.walk([&](Operation* op) {
    for (Value result : op->getResults()) {
      auto dataFlow = dataFlowGraph.queryDataFlow(result);

      if (dataFlow.kind == DataFlowInfo::DataFlowResult::MEMORY_SSA) {
        llvm::outs() << "Memory SSA: " << result << " -> "
                     << dataFlow.definition->getId() << "\n";
      }
    }
  });

  // 4. 导出可视化
  cfg->exportToFile("cfg.dot");
  cfg->exportToFile("cfg.json");
  cfg->exportToHTML("cfg.html");
}
```

### 4.2 分析scf.if的数据流

```cpp
void analyzeIfOp(scf::IfOp ifOp) {
  // ifOp有results，每个result都创建了phi definition
  for (size_t i = 0; i < ifOp.getNumResults(); i++) {
    Value result = ifOp.getResult(i);

    DataFlowInfo::DataFlowResult dataFlow =
        dataFlowInfo.queryDataFlow(result);

    if (dataFlow.kind == DataFlowInfo::DataFlowResult::MEMORY_SSA) {
      Definition* def = dataFlow.definition;

      // 查询uses
      SmallVector<Use> uses = defUseMap[def];

      llvm::outs() << "If result #" << i << " definition: "
                   << def->getId() << "\n";
      llvm::outs() << "  Uses: " << uses.size() << "\n";

      for (const Use& use : uses) {
        llvm::outs() << "    - " << use.getUserOp()->getName() << "\n";
      }
    }
  }
}
```

### 4.3 分析scf.for的iter_args

```cpp
void analyzeForOp(scf::ForOp forOp) {
  // 获取iter_args
  for (unsigned i = 0; i < forOp.getNumIterOperands(); i++) {
    Value iterArg = forOp.getRegionIterArg(i);

    // 查询definition
    Definition* def = dataFlowInfo.getMemoryDefinition(iterArg);

    if (def) {
      llvm::outs() << "IterArg #" << i << " definition: "
                   << def->getId() << "\n";

      // 查询loop phi信息
      if (LoopPhiInfo* phi = dataFlowInfo.getLoopPhi(iterArg)) {
        llvm::outs() << "  Loop phi: initial="
                     << (phi->comingFrom.initialValue ?
                         phi->comingFrom.initialValue->getId() : "null")
                     << ", yield="
                     << (phi->comingFrom.yieldValue ?
                         phi->comingFrom.yieldValue->getId() : "null")
                     << "\n";
      }
    }
  }
}
```

## 5. 关键技术点

### 5.1 scf.if的Phi节点处理

```mlir
%x = scf.if %cond -> (tensor<64x64xf32>) {
  %t = tt.load %ptr1 : tensor<64x64xf32>
  scf.yield %t : tensor<64x64xf32>
} else {
  %e = tt.load %ptr2 : tensor<64x64xf32>
  scf.yield %e : tensor<64x64xf32>
}
```

**处理步骤：**

1. 在ifCondBB创建if指令
2. 在then区域，%t的definition是[tensor_0, 0]
3. 在else区域，%e的definition是[tensor_1, 0]
4. 在mergeBB创建phi definition：
   - 新definition：[phi_tensor, 1]
   - operands：thenDef和elseDef
5. if result %x的definition是[phi_tensor, 1]

### 5.2 scf.for的IterArgs处理

```mlir
scf.for %i = %lb to %ub step %step iter_args(%arg = %init) {
  %data = tt.load %arg : tensor<64x64xf32>
  %new_arg = tt.addptr %arg, %offset
  scf.yield %new_arg : !tt.ptr<tensor<64x64xf32>>
}
```

**处理方案（保持同一definition）：**

1. %arg在所有迭代中共享同一个definition：[tensor_0, 0]
2. 创建LoopPhiInfo：
   - initialValue: [tensor_0, 0]（来自%init）
   - yieldValue: 初始为nullptr，后续更新
3. 循环体中所有使用%arg的地方都是[tensor_0, 0]
4. yield %new_arg更新了%arg的值，但不创建新definition
5. 记录更新：phiInfo.comingFrom.yieldValue = [tensor_0, 0]

**优点：**
- 避免创建大量definition版本
- 保留循环依赖信息
- 简化依赖分析

### 5.3 Alias跟踪

```mlir
// 示例1: addptr chain
%16 = tt.addptr %arg0, %15   // %16是%arg0的alias
%18 = tt.make_tensor_ptr %16, ...  // %18是%16的alias

// 都指向同一个base tensor: [gm_obj_0, param]
```

```mlir
// 示例2: load的equivalence
%27 = tt.load %18 : !tt.ptr<tensor<64x64xf32>>

// %27是[gm_obj_0, param]的equivalence
// 使用与%18相同的definition
```

**Alias分析算法：**

1. 识别全局内存参数（gm_obj_X）
2. 跟踪addptr链，每次addptr都保持base pointer
3. make_tensor_ptr指向base pointer
4. load操作返回与pointer相同的definition

## 6. 文件结构

```
third_party/ascend/include/TritonToCFG/
├── ControlFlowGraph.h          # 控制流图（已有，需要扩展Instruction）
├── ControlFlowGraphBuilder.h   # CFG构建器（已有）
├── InterProceduralCFG.h        # 过程间CFG（已有）
├── tensor.h                    # TensorObject定义（NEW）
├── memory_ssa.h                # Memory SSA定义（NEW）
├── alias_analysis.h            # Alias分析（NEW）
├── memory_ssa_builder.h        # Memory SSA构建器（NEW）
└── dataflow_graph.h            # 数据流图（NEW）

third_party/ascend/lib/TritonToCFG/
├── ControlFlowGraph.cpp        # CFG实现（已有，需要扩展）
├── ControlFlowGraphBuilder.cpp # CFG构建实现（已有）
├── tensor.cpp                  # TensorObject实现（NEW）
├── memory_ssa.cpp              # Memory SSA实现（NEW）
├── alias_analysis.cpp          # Alias分析实现（NEW）
├── memory_ssa_builder.cpp      # Memory SSA构建实现（NEW）
└── dataflow_graph.cpp          # 数据流图实现（NEW）
```

## 7. 开发计划

### 阶段1：基础数据结构 (2天)
- [ ] 实现TensorObject、Definition、Use
- [ ] 实现MemorySSAInfo
- [ ] 扩展Instruction结构
- [ ] 单元测试

### 阶段2：Alias分析 (2天)
- [ ] 实现AliasAnalysis
- [ ] 支持addptr和make_tensor_ptr
- [ ] 支持load的equivalence
- [ ] 单元测试

### 阶段3：Memory SSA构建 (3天)
- [ ] 实现MemorySSABuilder
- [ ] 支持基本指令处理
- [ ] 支持scf.if的phi节点
- [ ] 支持scf.for的iter_args（保持同一definition）
- [ ] 集成测试

### 阶段4：数据流接口 (2天)
- [ ] 实现DataFlowInfo和DataFlowGraph
- [ ] 统一查询接口
- [ ] 导出功能（JSON/DOT）
- [ ] 端到端测试

### 阶段5：与现有流程集成 (1天)
- [ ] 集成到ControlFlowGraphBuilder
- [ ] 添加Pass选项
- [ ] 实际IR测试

**总计：10天**

## 8. 待解决问题

1. **scf.while支持**：当前设计主要关注scf.if和scf.for，scf.while需要额外设计

2. **嵌套循环**：嵌套循环的iter_args处理需要验证

3. **TensorObject生命周期**：当前使用new分配，需要考虑内存管理

4. **操作分类**：isTensorWriter/Reader的判断需要完善，可能需要操作白名单

5. **版本号管理**：当前version在构建时递增，需要确保唯一性

6. **循环迭代分析**：iter_args的yieldValue如何在静态分析中使用

## 9. 参考资料

- [SSA Book](https://pfalcon.github.io/ssabook/latest/book-full.pdf) - SSA形式理论基础
- MLIR文档：https://mlir.llvm.org/
- Triton IR文档：https://triton-lang.org/
- Memory SSA论文："Memory SSA: A Unified Approach for Sparsely Representing Memory Operations"

## 10. 附录

### 10.1 术语表

- **Memory SSA**：扩展SSA以处理内存操作的表示形式
- **Definition**：Memory SSA的"定义"，表示值的创建点
- **Use**：Memory SSA的"使用"，表示值的使用点
- **Tensor Object**：表示内存中的tensor对象（如gm_obj_0）
- **Alias**：指针别名，不同指针指向同一内存位置
- **Loop Phi**：循环的phi节点，用于合并iter_args的多个来源

### 10.2 DSL示例

```mlir
// 输入IR
func.func @matmul_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i32

  // scf.for with iter_args
  scf.for %i = %c0 to %c64 step %c1 iter_args(%ptr0 = %arg0) {
    %new_ptr = tt.addptr %ptr0, %c64 : !tt.ptr<f32>, i32
    scf.yield %new_ptr : !tt.ptr<f32>
  }

  return
}

// Memory SSA构建后：
// %arg0 -> [gm_obj_0, param]
// %ptr0 (iter_arg) -> [gm_obj_0, param] (LoopPhi)
// %new_ptr -> [gm_obj_0, param] (alias of %ptr0)
// yield updates iter_arg, but no new definition
```

---

**文档版本历史**
- v1.0 (2026-03-16): 初始版本
