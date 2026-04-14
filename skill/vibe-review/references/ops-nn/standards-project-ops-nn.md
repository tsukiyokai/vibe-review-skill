# 【项目级】ops-nn项目规则

基于ops-nn仓库（1474次提交/380条缺陷，25.8%）和ops-nn-dev仓库（2571次提交/612条缺陷，23.8%）的完整git历史分析，从992条缺陷提交中提炼的55条高价值审查规则。每条规则均有commit证据和实际代码支撑。

ops-nn/ops-nn-dev是NN算子仓库，覆盖matmul、conv、pooling、norm、quant、scatter、foreach等模块。缺陷模式与通信库有本质差异——算子仓库侧重精度/tiling/shape/数据类型/kernel参数/流水线同步。matmul模块是绝对缺陷震中，两个仓库中均排名第一。

### 严重等级定义

| 等级 | 含义 | 影响范围                                    |
| ---- | ---- | ------------------------------------------- |
| P0   | 致命 | CoreDump、数据损坏、内存越界、死锁/无限循环 |
| P1   | 严重 | 特定配置崩溃、静默精度错误、功能不可用      |
| P2   | 一般 | 边界条件异常、构建失败、测试不通过          |
| P3   | 建议 | 代码质量、可维护性、潜在隐患                |

---

## 1整数溢出与类型安全

### 规则1.1 shape维度乘法必须使用int64_t

严重等级：P0

缺陷描述：shape维度值或元素总数使用int32_t/uint32_t存储，当维度乘积超过2^31-1时溢出。大shape场景（大batch x大sequence）下buffer分配不足或计算错误。排序索引硬编码DT_INT32、buffer偏移计算、batch维度累乘均为高发场景。

典型代码示例：

```cpp
// 缺陷代码 — 8f6ccaea (addlayernormgrad)
uint32_t roundUpNumLastDimFloat = ROUND_UP(numLastDim, ...);
// numLastDim > 2^31时ROUND_UP结果已溢出

// 修复代码
uint64_t roundUpNumLastDimFloat = ROUND_UP(static_cast<uint64_t>(numLastDim), ...);
```

```cpp
// 缺陷代码 — e0ddd962 (ops-nn-dev)
int32_t offset = addUbSize.GetValue(8) * shapeDim_;  // 两个int32相乘溢出

// 修复代码
uint64_t offset = static_cast<uint64_t>(addUbSize.GetValue(8)) * shapeDim_;
```

审查检查方法：
- shape维度乘法变量必须使用int64_t/uint64_t
- tiling结构体中shape/元素数相关字段的类型是否足够宽
- ROUND_UP/CeilAlign等宏的输入值和输出值类型一致性
- GetValue()返回值参与乘法时，操作数是否显式cast为64位

关联commit: `8f6ccaea`, `acadb4c7`, `fa1305c56e`, `e0ddd962`, `6ac2bba4`

---

### 规则1.2 GM偏移量在小类型域计算溢出

严重等级：P0

缺陷描述：GlobalTensor下标表达式`index * maxDataCount`中，当index为uint16_t/uint32_t时乘法在小类型域完成后溢出，结果才赋给uint64_t。此模式在foreach算子系列中批量出现（10个文件涉及8-9个算子）。

典型代码示例：

```cpp
// 缺陷代码 — 50df91e7 (foreach系列10个文件)
DataCopy(dataLocal, inTensorsGM[index * maxDataCount], dataCount);
// 乘法在uint32_t域完成后溢出，然后才被提升为指针偏移的uint64_t

// 修复代码
DataCopy(dataLocal, inTensorsGM[1ULL * index * maxDataCount], dataCount);
```

审查检查方法：
- GM内存偏移计算表达式中至少一个操作数必须为64位类型
- 批量审查：搜索`Gm[expr * expr]`或`inputGm[expr * expr]`模式

关联commit: `50df91e7`

---

### 规则1.3硬件指令参数位宽限制校验缺失

严重等级：P0

缺陷描述：DMA/datacopy指令的stride/blockLen参数为uint16_t类型（最大65535）。当cin或数据量超过65535时参数截断。load3d指令的kStartPt字段上限65535（16位），超限时溢出翻转指向错误内存位置。

典型代码示例：

```cpp
// 缺陷代码 — 34217db7 (Conv3DBackpropInput)
uint16_t stride = cin * sizeof(float);  // cin > 65535时截断

// 修复代码 — tiling层校验并降级
if (cin > MAX_DATACOPY_STRIDE) {
    useFallbackPath = true;
}
```

```cpp
// 缺陷代码 — 952313ace0 (ops-nn-dev, load3d)
int kStartPt = k0 * hk * wk;  // 超过65536时溢出翻转

// 修复代码
if (load3dK > LOAD3D_KSTART_MAX) {
    UseLargeKernelSplit();
}
```

审查检查方法：
- datacopy/DMA stride参数使用前校验不超过65535
- 涉及硬件指令参数时确认所有字段位宽限制
- tiling代码中对硬件参数做边界校验

关联commit: `34217db7`, `cf9ea8c9`, `952313ace0`

---

### 规则1.4有符号/无符号混比与uint64_t减法下溢

严重等级：P1

缺陷描述：无符号常量与有符号shape值比较引发-Wsign-compare告警；size_t/uint64_t做减法时被减数小于减数，结果下溢为极大正整数，后续用于内存分配或循环控制导致OOM或越界。

典型代码示例：

```cpp
// 缺陷代码 — 941d3c7f
if (shape.GetDim(i) > kSupportedInnerAxis) { ... }
// kSupportedInnerAxis为uint64_t，shape维度值为int64_t → 混合比较

// 修复代码
static constexpr int64_t kSupportedInnerAxis = ...;  // 类型一致
```

```cpp
// 缺陷代码 — hotspot aclnn_matmul.cpp (ops-nn-dev)
size_t loopDims = dimNum - 2;  // dimNum < 2时下溢为极大值

// 修复代码
if (dimNum < 2) { return ERROR; }
size_t loopDims = dimNum - 2;
```

审查检查方法：
- 开启`-Wsign-compare`并视为错误
- size_t/uint64_t做减法前是否判断被减数>=减数
- 常量类型与比较对象类型一致

关联commit: `941d3c7f`, `0253078d`, hotspot `aclnn_matmul.cpp`, `group_norm_silu_tiling.cpp`

---

## 2 Tiling/Buffer/Workspace计算

### 规则2.1 tiling侧buffer计算与kernel侧InitBuffer不一致

严重等级：P0

缺陷描述：tiling侧用公式估算buffer大小，但kernel侧实际buffer分配策略不同（buffer数量、double buffer策略），导致UB溢出或数据覆盖。这是tiling/buffer类缺陷中最具代表性的模式。

典型代码示例：

```cpp
// 缺陷代码 — e25e6e60c7 (ops-nn-dev)
bufferSize = ubSize_ / DOUBLE_BUFFER / bufferCount;
// 但kernel侧实际：x和y各占2份buffer(double buffer)，
// scale/shift又有不同策略(NO_DOUBLE_BUFFER)

// 修复 — 统一tiling侧公式
bufferSize = ubSize_ / (X_Y_BUFFER_NUM * DOUBLE_BUFFER + 1 + hasScaleShift);
```

审查检查方法：
- tiling侧buffer计算必须与kernel侧InitBuffer做交叉比对
- 检查buffer数量和double buffer策略是否两端一致
- 建议用共享常量定义buffer数量

关联commit: `e25e6e60c7`

---

### 规则2.2 tiling参数未区分normal核和tail核

严重等级：P0

缺陷描述：多核tiling中normal核和tail核使用相同参数（如buffer大小），但tail核的数据量通常小于normal核。用normal核的参数分配tail核的workspace可能不足，导致OOM。

典型代码示例：

```cpp
// 缺陷代码 — MaxPoolGradWithArgmaxV3 (ops-nn-dev)
void CalcGradArgmaxInner(int highAxisInner) {
    bufferSize = highAxisInner * elementSize;  // tail核highAxisTail可能远小
}

// 修复 — 拆分为normal版和tail版
void CalcGradArgmaxInner(int highAxisInner) { ... }     // normal
void CalcGradArgmaxInnerTail(int highAxisTail) { ... }   // tail
```

审查检查方法：
- 多核tiling中normal核和tail核的buffer尺寸是否分别计算
- tiling key是否区分normal/tail模式

关联commit: `90649ac299`

---

### 规则2.3 TilingData结构体host-kernel不一致

严重等级：P0

缺陷描述：TilingData结构体的字段布局、类型在host和kernel端不完全一致，导致数据解析错位。字段重排、类型缩窄(uint64->uint32)、value字段冗余等都会引发精度错误。

典型代码示例：

```cpp
// 缺陷代码 — 0d32207040 (ops-nn-dev)
// Host端TilingData
struct TilingData {
    int64_t scale;   // 应为float，int64_t导致精度丢失
    int64_t value;   // 冗余字段，kernel端不使用
};
// Kernel端期望 float scale; → 字段偏移错位
```

审查检查方法：
- TilingData结构体定义在host和kernel端是否完全一致（字段数、字段类型、字段顺序）
- 建议使用共享头文件定义TilingData

关联commit: `0d32207040`, revert `74cdae15d`

---

### 规则2.4 workspace计算公式中错误应用双buffer系数

严重等级：P1

缺陷描述：workspace大小计算时对每项无差别乘BUFFER_NUM（双buffer系数），但部分buffer实际只需单份空间。

典型代码示例：

```cpp
// 缺陷代码 — e32a5eaae3 (ops-nn-dev)
userWorkspaceByteSize += BT * H * IN_BYTE_SIZE * BUFFER_NUM;  // 只需单份

// 修复 — 去掉不需要双buffer的项
userWorkspaceByteSize += BT * H * IN_BYTE_SIZE;
```

审查检查方法：
- workspace计算中每项乘BUFFER_NUM时逐项注释说明原因
- 代码审查时逐行核对公式各项物理含义

关联commit: `e32a5eaae3`

---

### 规则2.5 tiling key参数传递错误与dead code

严重等级：P1

缺陷描述：tiling key参数传递错误导致kernel选择了错误的计算策略；return语句后的SetAtomicNone等代码永远不执行(dead code)。

审查检查方法：
- tiling key参数应有枚举约束而非裸字符串/魔法数
- return语句后不应有可执行代码
- tiling key值与kernel侧的switch-case必须一一对应

关联commit: `6eee5478`

---

## 3条件判断与边界处理

### 规则3.1 !=A || !=B恒真逻辑错误

严重等级：P0

缺陷描述： `x != A || x != B`（当A != B时）恒为true，属于De Morgan律的经典误用。在ops-nn中多次独立出现，导致合法输入被拒绝或错误路径被选中。

典型代码示例：

```cpp
// 缺陷代码 — 67c665fd (avg_pool_v2_grad)
if (inputDimNum != CHW_DIMS || inputDimNum != NCHW_DIMS) {
    return GRAPH_PARAM_INVALID;  // 恒为true，所有输入被拒绝
}
// 修复代码
if (inputDimNum != CHW_DIMS && inputDimNum != NCHW_DIMS) {
    return GRAPH_PARAM_INVALID;
}
```

审查检查方法：
- `x != A || x != B`是经典恒真逻辑错误
- 条件表达式中多个！=用||连接时99%应该用&&
- 建议作为clang-tidy静态检查规则

关联commit: `67c665fd`, `692f43a9`

---

### 规则3.2空tensor输入未正确处理

严重等级：P1

缺陷描述：空tensor输入时算子未正确early return并设置空输出shape，导致后续计算对零长度维度执行非预期操作。输入空和输出空语义不同，不能用`||`合并处理。此模式在aclnn_batch_matmul.cpp中被9次触及。

典型代码示例：

```cpp
// 缺陷代码 — aclnn_batch_matmul.cpp (hotspot, ops-nn-dev)
auto graph = CreateBatchmmEmptyTensorGraph(...);  // 空tensor分支
graph = CreateBatchMatmulExecBmmOpGraph(...);      // 无条件覆盖！死代码

// 修复
if (IsEmptyTensor(...)) {
    graph = CreateBatchmmEmptyTensorGraph(...);
} else {
    graph = CreateBatchMatmulExecBmmOpGraph(...);
}
```

审查检查方法：
- 空tensor判断后是否有return或else保护后续逻辑
- 输入空和输出空是否分别处理
- 空tensor分支是否正确设置输出shape

关联commit: hotspot `aclnn_batch_matmul.cpp`, `95c2fbb6bf`

---

### 规则3.3多核任务分配未区分tail core和idle core

严重等级：P0

缺陷描述：group卷积/分核场景下，`>=`判断将合法尾部core和超范围idle core合并处理，导致超范围core使用错误数据量计算，产生精度问题或越界访问。

典型代码示例：

```cpp
// 缺陷代码 — e2fc1fecb1 (ops-nn-dev)
if (coreIdx >= normalCoreNum) {
    ProcessData(tailDataSize);  // idle core不应执行此操作
}

// 修复 — 拆分为两个分支
if (coreIdx == normalCoreNum) {
    ProcessData(tailDataSize);
} else if (coreIdx > normalCoreNum) {
    return;  // idle core直接返回
}
```

审查检查方法：
- 多核任务分配中`>=`条件是否应拆分为`==`(tail)和`>`(idle)
- idle core是否有明确的early return

关联commit: `e2fc1fecb1`

---

### 规则3.4尾块/边界条件处理遗漏

严重等级：P1

缺陷描述：数据不能被tile/block整除时尾块处理遗漏、除零防护缺失、动态shape维度为0时的特殊处理遗漏、CeilDiv结果缺少上界clamp。

典型代码示例：

```cpp
// 缺陷代码（通用模式）
uint32_t tileCount = totalSize / tileSize;  // 不能整除时尾块丢失

// 修复代码
uint32_t tileCount = CeilDiv(totalSize, tileSize);
uint32_t lastTileSize = totalSize - (tileCount - 1) * tileSize;
```

```cpp
// 缺陷代码 — a70d8b16d5 (ops-nn-dev)
hoAL1min = CeilDiv(m0, wo);  // 结果可能超过实际ho

// 修复 — 添加上界约束
hoAL1min = std::min(CeilDiv(m0, wo), ho);
```

审查检查方法：
- tile/分块循环中最后一块的大小是否与其他块不同
- 所有除法的除数是否可能为零
- CeilDiv/CeilAlign结果是否需要与实际shape维度做min

关联commit: 多条， `a70d8b16d5`, `22f84c0e`

---

### 规则3.5设置特殊状态后缺少return

严重等级：P0

缺陷描述：SetUnknownRank/SetUnknownShape后没有return语句，代码继续执行后续shape推导逻辑，对UnknownRank的shape调用GetDimNum可能返回异常值导致越界访问。

典型代码示例：

```cpp
// 缺陷代码 — 2d424df8e6 (ops-nn-dev)
if (condition) {
    outputDesc->SetUnknownRank();
    // 缺少 return ge::GRAPH_SUCCESS;
}
int dimNum = outputDesc->GetShape().GetDimNum();  // dimNum可能是垃圾值

// 修复
if (condition) {
    outputDesc->SetUnknownRank();
    return ge::GRAPH_SUCCESS;
}
```

审查检查方法：
- 设置特殊状态(UnknownRank/UnknownShape)的分支必须包含return
- InferShape函数应做静态分析确保所有路径正确返回

关联commit: `2d424df8e6`

---

### 规则3.6新增类型/分支改变默认路径行为

严重等级：P1

缺陷描述：在条件判断中新增一个类型分支时，if-else结构的默认(else)路径原本服务于已有类型，但新分支的插入改变了条件匹配顺序，导致原有类型走入错误路径。

典型代码示例：

```cpp
// 缺陷代码 — 4d315436 (QuantBatchMatmulV4)
int k0 = (dtype == DT_INT4) ? INT8_K0 : INT4_K0;
// 条件搞反了: int4应该用INT4_K0

// 修复代码
int k0 = (dtype == DT_INT4) ? INT4_K0 : INT8_K0;
```

审查检查方法：
- 三元表达式的true分支应对应"新增的特殊情况"
- else（默认分支）必须保持原有行为不变
- if-else-if链中新增条件后，检查所有已有case是否仍匹配正确分支

关联commit: `4d315436`

---

### 规则3.7维度扩展后判断条件未适配

严重等级：P1

缺陷描述：2D场景tensor扩展为5D(D=1)后，CheckOutputAllZero要求所有维度为0才推断输出shape，D=1导致返回false，infershape流程失败。

典型代码示例：

```cpp
// 缺陷代码 — 08b9ebb112 (ops-nn-dev)
bool CheckOutputAllZero(const Shape& shape) {
    for (int i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) != 0) return false;  // D=1时返回false
    }
    return true;
}

// 修复 — 新增CheckOutputAllZeroFrom2D处理扩展维度
```

审查检查方法：
- 同时支持2D和3D且通过维度扩展统一处理时，所有依赖维度值的判断需考虑扩展维度默认值

关联commit: `08b9ebb112`

---

## 4空指针与资源管理

### 规则4.1 OP_CHECK宏缺少return语句

严重等级：P0

缺陷描述：OP_CHECK宏的第三参数写成裸值`nullptr`而非`return nullptr`。OP_CHECK失败时不会return而继续向下执行，使用空指针导致crash。此模式在单个文件中可批量出现（如aclnn_median.cpp中14处）。

典型代码示例：

```cpp
// 缺陷代码 — 7b4a1b53 (aclnnNanMedian)
OP_CHECK(condition, "error msg", nullptr);
// 失败时执行 nullptr; 作为表达式语句 -> 什么都不做

// 修复代码
OP_CHECK(condition, "error msg", return nullptr);
```

审查检查方法：
- OP_CHECK宏的第三参数必须包含return语句
- 静态分析：全文搜索`OP_CHECK.*,\s*nullptr\)`

关联commit: `7b4a1b53`

---

### 规则4.2结构体指针成员未初始化

严重等级：P0

缺陷描述：C++结构体中指针成员未在声明处初始化，某些代码路径下成员值为野指针。后续无条件使用（如OP_LOGE("%s", result.logMessage)）导致崩溃。

典型代码示例：

```cpp
// 缺陷代码 — 15e40a48 (addmm matmul_util.cpp)
struct PromoteResult {
    const char* logMessage;  // 未初始化 -> 野指针
};

// 修复代码
struct PromoteResult {
    const char* logMessage = nullptr;
};
```

审查检查方法：
- C++结构体指针成员必须在声明处初始化为nullptr
- OP_LOGE中`%s`对应的参数必须保证非空

关联commit: `15e40a48`

---

### 规则4.3先解引用后判空（先用后查）

严重等级：P0

缺陷描述：指针先被解引用使用，之后才检查是否为nullptr。此"先用后查"模式在op_api层多处存在。更严重的变体：CHECK_RET中检查的变量不是刚被赋值的变量。

典型代码示例：

```cpp
// 缺陷代码 — hotspot (aclnn_addmm.cpp:298-299)
auto selfContiguous = l0op::Contiguous(addmmTensor.self, uniqueExecutor);
if (addmmTensor.self != nullptr && ...)  // 检查太晚

// 缺陷代码 — hotspot (aclnn_quant_matmul_v5.cpp:907-908)
reformatedX1 = l0op::TransData(reformatedX1, ...);
CHECK_RET(x1 != nullptr, ...);  // 检查错误变量
```

审查检查方法：
- 指针解引用必须在nullptr检查之后
- CHECK_RET中检查的变量必须是刚被赋值的变量
- 搜索模式： `= l0op::Xxx(ptr); if (ptr != nullptr)`

关联commit: hotspot `aclnn_addmm.cpp`, `aclnn_quant_matmul_v5.cpp`

---

### 规则4.4可选输出tensor（outputMask控制）访问前必须检查null

严重等级：P0

缺陷描述：反向传播中dx/dw/dbias由outputMask控制是否计算，为null时仍被解引用取shape或dtype导致crash。

典型代码示例：

```cpp
// 缺陷代码 — e49d5d27c0 (ops-nn-dev)
auto mmDwOutShape2d = CalcShape(gradInput->GetShape());  // gradInput为nullptr时crash

// 修复 — 用outputMask守护
if (outputMask[0]) {
    PreConv1DBackwardTo2D(gradInput, ...);
}
```

审查检查方法：
- 反向传播中dx/dw/dbias解引用前需null检查或对应mask位检查
- dtype兼容性判断优先使用必选张量而非可选张量

关联commit: `e49d5d27c0`, `59a561e5`, `2463697d99`

---

### 规则4.5禁止在op_host/*.cpp中使用非const全局容器

严重等级：P0

缺陷描述：在.cpp文件作用域定义全局std:vector等容器，多实例并发调用时共享导致数据竞争和AIC错误。

典型代码示例：

```cpp
// 缺陷代码 — 2a15cd4d (ops-nn-dev)
std::vector<gert::Stride> indexstrideList;  // 全局！并发不安全

// 修复 — 移入类成员
class IndexNonContinuousTiling {
private:
    std::vector<gert::Stride> indexstrideList_;  // 每个实例独立
};
```

审查检查方法：
- op_host/*.cpp中是否有非const全局变量/容器声明
- tiling相关状态必须封装在类成员中

关联commit: `2a15cd4d`, revert `2b00825d3`

---

## 5流水线同步与硬件事件

### 规则5.1 PipeBarrier与FreeTensor时序颠倒

严重等级：P0

缺陷描述：PipeBarrier<PIPE_V>放在FreeTensor之后，导致Cast/向量运算未完成就释放了输入buffer。运行时表现为随机精度错误或数据损坏。

典型代码示例：

```cpp
// 缺陷代码 — cf84e222 (AddRmsNorm)
FreeTensor(inputBuffer);    // 先释放
PipeBarrier<PIPE_V>();      // 后等待

// 修复代码
PipeBarrier<PIPE_V>();      // 先等待
FreeTensor(inputBuffer);    // 后释放
```

审查检查方法：
- PipeBarrier必须在FreeTensor之前
- 搜索FreeTensor调用，确认前面都有PipeBarrier

关联commit: `cf84e222`

---

### 规则5.2 SetFlag/WaitFlag必须成对出现且无条件执行

严重等级：P0

缺陷描述：WaitFlag被条件化（如if ni>0）导致某些迭代路径跳过必要的同步等待；SetFlag/WaitFlag配对位置不当或提前return路径遗漏event平衡；CopyOut前使用了错误通道的同步事件。

典型代码示例：

```cpp
// 缺陷代码 — a79b527fe8 (ops-nn-dev)
if (ni > 0) {
    WaitFlag<MTE3_MTE2>(eventId);  // ni==0时跳过等待！
}
SetFlag<MTE3_MTE2>(eventId);       // 但SetFlag无条件执行

// 修复 — 无条件执行WaitFlag
WaitFlag<MTE3_MTE2>(eventId);
SetFlag<MTE3_MTE2>(eventId);
```

```cpp
// 缺陷代码 — 51f8247aee (ops-nn-dev)
WaitFlag<S_MTE2>(eventId);       // CopyOut前使用MTE2同步 — 错误通道
CopyOut(dst, src, count);        // CopyOut走MTE3通道

// 修复 — 使用正确的MTE3通道
WaitFlag<S_MTE3>(eventId);
WaitFlag<V_MTE3>(eventId2);
CopyOut(dst, src, count);
```

审查检查方法：
- SetFlag和WaitFlag是否成对出现且无条件执行
- CopyOut前的同步是否用MTE3通道（不是MTE2）
- pipe方向必须与实际数据流向一致

关联commit: `ed0bd5a1`, `a79b527fe8`, `51f8247aee`

---

### 规则5.3 DataCopy前后缺少流水线同步事件

严重等级：P0

缺陷描述：DataCopy(MTE2)前缺少MTE3_MTE2同步，向量运算前缺少MTE2_V同步。运行时表现为相同输入产生不同输出（竞态条件）。

典型代码示例：

```cpp
// 缺陷代码 — 74d43ef3 (scatter_sub)
DataCopy(localBuffer, gmSrc, dataLen);
// 缺少WaitFlag<PIPE_MTE2, PIPE_V>
VecAdd(result, localBuffer, localOther);  // 可能读到未就绪数据

// 修复代码
DataCopy(localBuffer, gmSrc, dataLen);
SetFlag<PIPE_MTE2, PIPE_V>(eventId);
WaitFlag<PIPE_MTE2, PIPE_V>(eventId);
VecAdd(result, localBuffer, localOther);
```

审查检查方法：
- 数据流每个阶段转换(MTE2->V->MTE3)必须有对应同步事件
- DataCopy(MTE2)前确保前序MTE3完成

关联commit: `74d43ef3`

---

### 规则5.4连续matmul操作间必须插入PipeBarrier

严重等级：P1

缺陷描述：两次matmul操作间缺少PipeBarrier<PIPE_ALL>()同步，matmulR结果可能尚未完成matmulL就读取。

典型代码示例：

```cpp
// 缺陷代码 — 6beab61dd0 (ops-nn-dev)
matmulL.Iterate();
// 缺少 PipeBarrier<PIPE_ALL>();
matmulR.Iterate();

// 修复
matmulL.Iterate();
PipeBarrier<PIPE_ALL>();
matmulR.Iterate();
```

审查检查方法：
- 连续matmul操作间是否插入PipeBarrier
- 同一kernel中eventID是否有冲突

关联commit: `6beab61dd0`

---

### 规则5.5 SetScheduleMode缺失

严重等级：P1

缺陷描述：tiling函数中缺少SetScheduleMode(1)调用，多核间存在数据依赖但未设置batch mode同步。

审查检查方法：
- 所有tiling函数必须显式设置SetScheduleMode
- 多核场景下是否存在数据依赖
- SetScheduleMode的参数值(0/1/2)是否与实际同步需求匹配

关联commit: `afb09c78`

---

## 6复制粘贴与变量引用

### 规则6.1函数调用参数重复f(a,a)

严重等级：P1

缺陷描述：函数调用中两个参数使用了同一个变量，实际上应使用不同变量。两个仓库均独立出现此模式（addbmm场景）。

典型代码示例：

```cpp
// 缺陷代码 — b2e2cada / dec58d1b24
bool isEmpty = isAddBmmProcessEmptyTensor(batch1, batch1);
// 第二参数应为batch2

// 修复代码
bool isEmpty = isAddBmmProcessEmptyTensor(batch1, batch2);
```

审查检查方法：
- 函数调用中两个参数完全相同时必须确认是否为copy-paste错误
- 尤其关注名称只差一个字符的参数(batch1/batch2, self/other, input/output)

关联commit: `b2e2cada`, `dec58d1b24`

---

### 规则6.2矩阵k/n维度索引粘贴错误

严重等级：P0

缺陷描述：矩阵维度索引的k和n在转置/非转置时必须互补。copy-paste后k维度的公式和n维度完全相同。

典型代码示例：

```cpp
// 缺陷代码 — 1c2de786 (matmul_common_infershape.cpp)
int64_t k_x2_dim = transB ? x2Shape.GetDim(dimNum - 1) : x2Shape.GetDim(dimNum - 2);
int64_t n_x2_dim = transB ? x2Shape.GetDim(dimNum - 1) : x2Shape.GetDim(dimNum - 2);
// n_x2_dim公式与k_x2_dim完全相同

// 修复代码
int64_t n_x2_dim = transB ? x2Shape.GetDim(dimNum - 2) : x2Shape.GetDim(dimNum - 1);
```

审查检查方法：
- 矩阵k/n维度索引在转置/非转置情况下必须互补
- 相邻行的赋值公式完全相同时高度疑似copy-paste错误

关联commit: `1c2de786`

---

### 规则6.3 GM偏移量公式维度混淆

严重等级：P1

缺陷描述：多维偏移量计算将不同维度索引合并后统一乘同一步长，公式语义错误。

典型代码示例：

```cpp
// 缺陷代码 — 910dac63 (QuantUpdateScatter)
gmVarOffset_ = (batchIdx * outerDim + axisOffset) * stride;
// axisOffset和outerDim的步长不同，不能合并乘

// 修复代码
gmVarOffset_ = batchIdx * outerDim * varDim3 + axisOffset * varDim3;
```

审查检查方法：
- 多维偏移量应逐维展开（各维索引x对应步长）
- 括号中合并多个维度索引后乘以统一步长是危险模式

关联commit: `910dac63`

---

### 规则6.4错误日志引用的变量与判断条件不一致

严重等级：P2

缺陷描述：CHECK/LOG调用中，报错信息引用的变量与实际校验的变量不一致，导致日志误导调试。

典型代码示例：

```cpp
// 缺陷代码 — d1832c87c9 (ops-nn-dev)
if (scaleShape->GetStorageShape().GetDim(0) != expectedDim) {
    LOG("dim mismatch: %d", offsetShape->GetStorageShape().GetDim(0));
    //                      ^^^^^^^^^^^^ 应为scaleShape
}
```

审查检查方法：
- 错误日志中引用的变量是否与上文条件判断中的变量一致
- 连续相似CHECK/LOG调用中逐行对比检查对象

关联commit: `d1832c87c9`

---

### 规则6.5删除"未使用变量"前检查是否原本应该被使用

严重等级：P2

缺陷描述：删除编译器报告的"未使用变量"时，该变量可能原本应该被使用——真正的bug是上游代码copy-paste后忘记替换变量名。盲目删除变量会掩盖真实逻辑缺陷。

典型代码示例：

```cpp
// 缺陷代码 — d818b4d4 (ops-nn-dev)
// 编译器报告sourceStorageFormat未使用
// 实际bug是条件判断copy-paste错误：
if (maskStorageFormat != FORMAT_ND || maskStorageFormat != FORMAT_ND) {
    //                                应为sourceStorageFormat
}
// 该commit只删除了sourceStorageFormat来消除告警，逻辑bug被掩盖
```

审查检查方法：
- 删除"未使用变量"前检查该变量是否原本应该被使用
- 搜索同一代码块中是否有copy-paste导致的变量名重复

关联commit: `d818b4d4`

---

## 7 Shape与接口一致性

### 规则7.1 StorageShape/ViewShape混淆

严重等级：P1

缺陷描述：非连续张量的StorageShape（物理内存布局）与ViewShape（用户视角逻辑shape）不同。shape校验使用GetStorageShape()时，非连续张量的StorageShape可能大于ViewShape，导致合法输入被错误拦截。

典型代码示例：

```cpp
// 缺陷代码 — 6b52fdcf (LSTM)
auto shape = input.GetStorageShape();  // 非连续张量StorageShape != ViewShape

// 修复代码
auto shape = input.GetViewShape();     // 优先使用逻辑shape
```

审查检查方法：
- shape校验优先使用GetViewShape()，仅在需要物理布局时用GetStorageShape()
- 非连续张量场景（view/transpose/slice后）必须区分两者

关联commit: `6b52fdcf`, `28ab6d70`

---

### 规则7.2接口调用默认参数与实际数据不一致

严重等级：P1

缺陷描述：调用下层接口时未传入实际数据类型，使用了接口的默认参数值。当用户传入的数据类型与默认值不同时产生错误结果。

典型代码示例：

```cpp
// 缺陷代码 — c8ca6bec (AdaptiveMaxPool3d)
MaxPool3D(input, output, indices);  // indicesDtype默认int32
// 但用户可能传int64的indices

// 修复代码
MaxPool3D(input, output, indices, indicesDtype);  // 显式传入
```

审查检查方法：
- 调用具有默认参数的接口时，检查实际数据类型是否可能与默认值不同
- 显式传参优于依赖默认值

关联commit: `c8ca6bec`

---

### 规则7.3属性索引硬编码顺序依赖

严重等级：P1

缺陷描述：通过`GetAttrPointer<T>(index)`硬编码索引获取算子属性，但不同算子类型的attr排列顺序不同。

典型代码示例：

```cpp
// 缺陷代码 — 07e77ddd (InplaceAddmm走GemmV3路径)
auto transA = GetAttrPointer<bool>(0);  // 期望读transpose_a
// 但GemmV3的attr顺序是: [alpha, beta, transpose_a, transpose_b]
// 索引0实际读到了alpha

// 修复代码 — GemmV3路径使用索引2/3
auto transA = GetAttrPointer<bool>(2);
```

审查检查方法：
- 优先使用属性名获取(GetAttrByName)而非索引
- 复用通用代码但attr布局不同时必须做分支处理

关联commit: `07e77ddd`

---

### 规则7.4变体接口复用通用dtype校验

严重等级：P1

缺陷描述：变体接口（如WeightNz版本）复用通用dtype校验函数，但变体接口有更严格的类型限制。通用校验放行了变体不支持的类型，运行时crash。

审查检查方法：
- 变体接口应使用独立dtype校验而非复用通用函数
- aclnn函数入口第一步应做所有必选输入的nullptr检查
- 校验函数的允许列表应与接口实际支持的类型严格一致

关联commit: `0bd3fe0b`

---

### 规则7.5修改Init签名/模板参数后调用点未全量同步

严重等级：P1

缺陷描述：修改kernel类Init签名或模板参数后，未全局搜索所有调用点同步更新。宏展开后的实际调用与当前函数签名不匹配。

典型代码示例：

```cpp
// 缺陷代码 — b14f5a03ca (ops-nn-dev)
GENERAL_OP_IMPL(op, OpType<T>);
// 宏展开: op.Init(tilingData);
// 但Init签名已改为: Init(tilingData, workspace);
```

审查检查方法：
- 修改Init签名/模板参数后，全局搜索所有调用点确认同步更新
- 宏展开后的实际调用是否与当前函数签名匹配

关联commit: `b14f5a03ca`, `22979c4afe`

---

### 规则7.6 TilingParse注册类型与实际CompileInfo类型不一致

严重等级：P1

缺陷描述：TilingParse<T>的模板参数T使用旧类型，但tiling实际依赖新类型，导致tiling数据解析失败。

审查检查方法：
- TilingParse<T>的T与TilingPrepare操作的CompileInfo类型是否一致
- 重构后grep旧类型名确认无残留引用

关联commit: `368607bd`

---

### 规则7.7新增数据类型支持时三处同步更新

严重等级：P1

缺陷描述：新增数据类型支持时，tiling层dtype校验、kernel层预编译条件、binary配置三处必须同步更新，遗漏任一处导致新类型走入错误分支或不被编译。

审查检查方法：
- 新增数据类型时同步检查tiling层、kernel层、binary配置三处
- 选择计算模板/优化路径时检查是否对所有支持的数据类型兼容

关联commit: `9150a7cca8`, `84ab237ad3`

---

## 8编译告警隐藏真bug

### 规则8.1无符号整数>=0恒真导致无限循环

严重等级：P0

缺陷描述： `for(uint64_t i = N; i >= 0UL; i--)`中`i >= 0`对无符号整数恒为true。当i减到0后再减1会wrap around到UINT64_MAX，循环永不终止。

典型代码示例：

```cpp
// 缺陷代码 — 0e00f88c (avg_pool3d_grad等5处)
for (uint64_t i = singleCoreWo; i >= 0UL; i--) {
    ProcessTile(i);  // 无限循环
}

// 修复代码
for (uint64_t i = singleCoreWo + 1; i > 0; i--) {
    ProcessTile(i - 1);
}
```

审查检查方法：
- 无符号整数循环中`>=0`条件是无限循环bug
- 开启`-Wtautological-compare`可自动检测

关联commit: `0e00f88c`

---

### 规则8.2链式比较与运算符优先级错误

严重等级：P1

缺陷描述： `a == b == 1`在C++中被解析为`(a == b) == 1`; `a < b < c`被解析为`(a < b) < c`即`bool < c`; `&&`优先级高于`||`，`X && Y || Z`实际为`(X && Y) || Z`。

典型代码示例：

```cpp
// 缺陷代码 — 50854088 (约14个文件)
if (tensor->GetViewShape().GetDim(firstLastDim) ==
    tensor->GetViewShape().GetDim(secondLastDim) == 1) { ... }
// 解析为 (dim1 == dim2) == 1

// 缺陷代码 — 7ad3e89e (ops-nn-dev)
if (lower < value < upper) { ... }
// 解析为 (lower < value) < upper，upper > 1时永远true
```

审查检查方法：
- `a == b == c`和`a < b < c`在C++中语义错误
- `&&`和`||`混合使用时必须加括号
- 开启`-Wlogical-op-parentheses -Wparentheses`

关联commit: `50854088`, `7ad3e89e`

---

## 9构建系统与配置

### 规则9.1 ascendc_config.json重复/冲突配置

严重等级：P1

缺陷描述：ascendc_config.json中存在完全重复的条目或同名但配置内容冲突的条目（如compute_units列表不同）。哪一条生效取决于JSON解析器实现，构建行为不确定。新增算子未在ascendc_config.json中注册导致kernel二进制不被打包。

典型代码示例：

```json
// 完全重复: AdaptiveAvgPool3d(2份), LayerNormQuant(3份)
// 同名冲突: AdaptiveAvgPool3dGrad/STFT的compute_units不一致
// 无效引用: compile_options中引用"ascend910_95"非已知平台标识
```

审查检查方法：
- 修改ascendc_config.json时检查同名算子是否已有条目
- 新增算子PR必须包含ascendc_config.json条目变更
- CI中应加入重复条目检测脚本

关联commit: hotspot_analysis, `c00e940bfa`, `0d47b59fdd`

---

### 规则9.2 CMake target依赖声明缺失

严重等级：P2

缺陷描述：CMake中custom_command/custom_target的DEPENDS未包含前置target，构建时序不确定导致间歇性构建失败。

典型代码示例：

```cmake
# 缺陷代码 — 640a1683
add_custom_target(ascendc_impl_gen
    DEPENDS ${impl_gen_outputs}  # 缺少对ops_info_gen_*的依赖
)
```

审查检查方法：
- CMake custom_command/custom_target的DEPENDS必须包含所有输入依赖
- foreach循环结束后不应使用循环变量（取值为最后一次迭代的值）

关联commit: `640a1683`, `fb3b8365b9`

---

### 规则9.3新平台binary/compute_units配置遗漏

严重等级：P1

缺陷描述：新增算子或新增硬件平台时，遗漏目标平台的binary.json配置或ascendc_config.json中compute_units列表缺少某个目标平台。编译阶段不报错，运行时找不到kernel。

典型代码示例：

```json
// 缺陷 — db31f7273 (ops-nn-dev, TransQuantParamV2)
{
    "op_name": "TransQuantParamV2",
    "compute_units": ["ascend910b", "ascend910_93"]
    // 缺少 "ascend310p"
}
```

审查检查方法：
- 新增算子时检查是否为所有目标平台都提供了binary.json和compute_units
- 对照平台支持矩阵逐一确认

关联commit: `d6acf37e`, `db31f7273`

---

### 规则9.4 shell脚本语法/逻辑错误

严重等级：P1

缺陷描述：shell脚本中`[`/`]`前后缺空格导致条件判断永远false/true、数组变量未正确展开、选项flag赋值错误、缺少`set -e`。

典型代码示例：

```bash
# 缺陷代码 — adec83fb (check_example.sh)
[ $status -ne 0]
# ']'与'0'粘连，条件永远false

# 缺陷代码 — build.sh
mssanitizer) ENABLE_MSSANITIZER=FALSE ;;  # 启用选项却赋值FALSE
```

审查检查方法：
- CI脚本必须通过shellcheck静态分析
- `[`和`]`前后必须有空格
- 数组变量使用`"${arr[@]}"`展开
- 构建脚本首行是否有`set -e`/`set -o pipefail`

关联commit: `adec83fb`, `9add7e2316`

---

### 规则9.5平台特定编译选项遗漏

严重等级：P1

缺陷描述：特定平台（如ascend910_95）需要额外编译选项(如`-mllvm -cce-aicore-dcci-before-kernel-end=false`)，但大量算子配置中遗漏。

审查检查方法：
- ascend910_95平台算子是否配置dcci相关编译选项
- 新增算子时对比同族算子的compile_options确认一致性

关联commit: `ac64e3d38b`

---

### 规则9.6 output paramType(REQUIRED/OPTIONAL)配置错误

严重等级：P1

缺陷描述：def.cpp中output的paramType标记与binary.json不一致，或与算子实际行为矛盾（实际为可选输出但标记为REQUIRED）。

审查检查方法：
- def.cpp的paramType与binary.json是否双向一致
- REQUIRED输出是否在所有代码路径中都有值

关联commit: `9c46dc5f19`

---

## 10平台适配与硬件约束

### 规则10.1 `__CCE_AICORE__`精确匹配应改为范围比较

严重等级：P1

缺陷描述：平台版本判断使用精确匹配`==`，新平台上线时指令集向下兼容，但精确匹配导致新平台不匹配任何分支走入default错误路径。

典型代码示例：

```cpp
// 缺陷代码 — 4f87bea9ab (ops-nn-dev)
#if __CCE_AICORE__ == 220
    UseAdvancedInstruction();
#else
    UseFallback();
#endif

// 修复 — 范围比较
#if __CCE_AICORE__ >= 220
    UseAdvancedInstruction();
#endif
```

审查检查方法：
- `__CCE_AICORE__ == xxx`和`__NPU_ARCH__ == xxx`精确匹配是否应改为`>=`
- 新增平台时有checklist检查所有相关算子的条件分支

关联commit: `4f87bea9ab`, `c1469b68`

---

### 规则10.2硬件宏条件编译遗漏

严重等级：P2

缺陷描述：特定芯片专有指令（如Fixpipe）未用`__NPU_ARCH__`等硬件宏条件编译保护，在不支持该指令的芯片上编译/运行失败。

审查检查方法：
- 特定芯片指令需用`__NPU_ARCH__`等硬件宏条件编译保护
- binary.json中不支持的平台配置应删除而非留空

关联commit: `c1469b68`

---

## 11计算逻辑错误

### 规则11.1多核场景全局内存初始化存在竞争

严重等级：P0

缺陷描述：多个核独立对全局内存做零初始化存在竞争条件。跨循环累积计数器语义混淆导致后续loop有效数据量不正确。

典型代码示例：

```cpp
// 缺陷代码 — fa03888b0d (ops-nn-dev)
if (isEmpty) {
    for (int i = 0; i < size; i++) {
        globalMem[i] = 0;  // 多核同时写，结果不确定
    }
}

// 修复 — 指定核初始化 + 同步屏障
if (isEmpty && GetBlockIdx() == 0) {
    for (int i = 0; i < size; i++) { globalMem[i] = 0; }
}
SyncAll();
```

审查检查方法：
- 多核场景全局内存写入是否由指定核(core 0)完成并加SyncAll

关联commit: `fa03888b0d`

---

### 规则11.2 splitK场景Bias加载时机错误

严重等级：P1

缺陷描述：splitK模式下Bias应在第一次K迭代(kIndex==0)加载参与累加，但错误地放在最后一次迭代加载。

典型代码示例：

```cpp
// 缺陷代码 — e7048cff4f (ops-nn-dev)
if (kIndex == splitKRound - 1) {
    LoadBias();  // 错误：应在首次累加时加入
}

// 修复
if (kIndex == 0) {
    LoadBias();
}
```

审查检查方法：
- splitK场景Bias/残差加载是否在kIndex==0
- 累加顺序是否符合数学定义

关联commit: `e7048cff4f`

---

### 规则11.3 L1全载优化在循环中的状态失效检查

严重等级：P1

缺陷描述：L1/L0缓存全载优化在多batch/多group循环中，当参数（如B矩阵偏移）变化时未切换全载状态，之前全载加载的数据已失效但仍被使用。

审查检查方法：
- L1全载优化在循环中，检查全载条件是否因参数变化而失效
- 状态变量(enableFullLoad)是否在循环迭代间正确更新

关联commit: `46f5817ef1`

---

## 12文档与示例代码

### 规则12.1示例代码未经编译/运行验证

严重等级：P2

缺陷描述：示例代码包含编译错误（重复函数定义、变量未声明）或运行时输出全0/垃圾数据。用户拷贝示例代码后直接编译失败或得到错误结果。

审查检查方法：
- 示例代码PR必须在CI中包含编译验证步骤
- 新增算子的示例应有至少一个非零输出的验证

关联commit: `501d38ab`, `4063f8f9`

---

### 规则12.2示例代码数据类型与API不匹配

严重等级：P1

缺陷描述：示例中host数据容器类型(如`vector<int32_t>`)与创建tensor时传入的aclDataType（如ACL_FLOAT）不匹配，数据被错误解释。

审查检查方法：
- 示例中`vector<T>`的T必须与`aclDataType`对应
- 检查aclrtMalloc/aclrtFree是否配对出现

关联commit: `964be4dd`

---

## 13功能回退与流程

### 规则13.1搭车提交造成revert附带伤害

严重等级：P2

缺陷描述：MR中混入与标题功能不相关的改动，revert时不相关改动被一并回滚。

审查检查方法：
- MR中每个文件改动应与标题描述的功能直接相关
- 不相关改动必须拆分为独立MR

关联commit: `d91e1757a`, `4482238c0`

---

### 规则13.2大规模重构应分步提交

严重等级：P2

缺陷描述：大体量提交（>20文件/>1000行）一次性合入后紧急revert。跨三层(op_api/tiling/kernel)的级联影响未充分验证。qbmm事件6.5万行一次性合入导致3次循环revert。

审查检查方法：
- 大feature合入（>20文件或>1000行）前需全流程CI + 集成测试
- 单次合入超过3000行的PR应要求拆分
- 大规模重构不应同时改变数据结构布局和计算逻辑
- 开启`-Wshadow`拦截变量遮蔽

关联commit: `852f21d6b`, revert `74cdae15d7`, `79623db1a`(qbmm)

---

### 规则13.3公共基础设施耦合在算子PR中

严重等级：P2

缺陷描述：公共头文件(op_util.h, error_util.h, op_api_def.h)和cmake修改耦合在算子PR中。公共修改影响面远超单算子，但review时容易被算子逻辑掩盖。

审查检查方法：
- 公共文件修改必须独立PR先行验证

关联commit: revert `79623db1a`, `9de027499`, `61a1f1583`

---

### 规则13.4移除binary.json算子变体前确认无下游调用

严重等级：P1

缺陷描述：过早移除binary.json中的算子变体或编译配置，但下游场景仍有调用，运行时找不到对应kernel。

审查检查方法：
- 移除binary.json变体前grep所有下游场景确认无调用
- 强制禁用已有模板(`return false`)必须注释原因

关联commit: `e682154870`

---

## 14日志与错误处理

### 规则14.1错误码区分PARAM_*和INNER_*

严重等级：P3

缺陷描述：对用户传入参数的校验失败使用INNER_*错误码（内部错误），应使用PARAM_*错误码。

审查检查方法：
- 错误码区分PARAM_*（用户输入）和INNER_*（内部不可达状态）

关联commit: `797589989a`

---

### 规则14.2宏定义中隐藏return语句

严重等级：P2

缺陷描述：宏中使用return语句隐藏控制流，调用者不知道宏可能提前退出函数。

审查检查方法：
- 宏定义中是否包含return语句
- 是否有更清晰的替代方案（内联函数、直接展开）

关联commit: `6beab61dd0`

---

## 15单元测试

### 规则15.1禁止注释EXPECT_*/ASSERT_*断言

严重等级：P3

缺陷描述：大量EXPECT_EQ断言被批量注释掉，UT失去验证能力沦为编译检查。

审查检查方法：
- grep检测`// EXPECT_`和`// ASSERT_`模式
- tiling参数应由tiling函数自动生成而非手工硬编码

关联commit: `91c29129`, `ee685a90`

---

## 跨类别系统性风险

### 规则SYS-01: matmul是绝对缺陷震中

严重等级：P0

缺陷描述：matmul/bmm模块在两个仓库中均是最高危模块。ops-nn中贡献67条缺陷(17.6%)跨9个类别；ops-nn-dev热点Top30文件中matmul占20席。量化矩阵乘(quant_batch_matmul_v3/v4, weight_quant_batch_matmul_v2)尤为集中。当前代码仍存在确认bug: 空指针先用后查、CHECK_RET检查错变量、copy-paste公式相同、除零风险等。

审查检查方法：
- matmul相关PR应提高审查标准（多人review）
- 重点检查空指针、copy-paste、维度索引、类型溢出
- matmul模块的代码复杂度高，新增功能需完整的回归测试矩阵

关联commit: hotspot分析，revert `9b3d2d76b`, `2e1ef4b41`, qbmm事件（3次循环revert）

---

### 规则SYS-02: 整数溢出是最普遍残留风险

严重等级：P0

缺陷描述：当前代码中仍存在多处整数溢出/精度损失风险，横跨matmul/conv/quant/norm四个模块。根因：int32类型tiling参数相乘、batch维度累乘、buffer偏移计算缺少溢出保护，以及int64_t转float精度丢失。

审查检查方法：
- 编写lint规则，强制shape维度算术运算使用int64_t
- 审查Lcm/GCD中乘法溢出、uint64_t下溢、int64->float精度丢失

---

### 规则SYS-03: 编译warning隐藏真bug

严重等级：P1

缺陷描述：编译告警修复提交中存在真正的逻辑bug: 无符号>=0恒真无限循环、`a==b==1`链式比较语义错误、`&&`/`||`优先级导致条件判断错误。编译warning不是噪声，而是真bug的信号。

审查检查方法：
- 开启`-Wall -Werror -Wsign-compare -Wshadow -Wtautological-compare -Wparentheses -Wconversion`
- warning修复提交应认真review——可能暴露新的bug

关联commit: `0e00f88c`, `50854088`, `d818b4d4`

---

### 规则SYS-04: 提交粒度过大导致质量失控

严重等级：P2

缺陷描述：qbmm事件6.5万行一次性合入导致3次循环revert; weight_quant事件5.7万行合入修改了公共error_util.h导致其他算子编译失败；70%的revert在3天内发生，说明合入前review不充分。

审查检查方法：
- 单次合入超过3000行的PR必须拆分
- 公共文件修改必须独立PR先行验证

---

## 附录：规则速查表

| 规则ID | 规则名称                                     | 等级 |                 |     |
| ------ | -------------------------------------------- | ---- | --------------- | --- |
| 1.1    | shape维度乘法必须使用int64_t                 | P0   |                 |     |
| 1.2    | GM偏移量在小类型域计算溢出                   | P0   |                 |     |
| 1.3    | 硬件指令参数位宽限制校验缺失                 | P0   |                 |     |
| 1.4    | 有符号/无符号混比与uint64_t减法下溢          | P1   |                 |     |
| 2.1    | tiling侧buffer计算与kernel侧InitBuffer不一致 | P0   |                 |     |
| 2.2    | tiling参数未区分normal核和tail核             | P0   |                 |     |
| 2.3    | TilingData结构体host-kernel不一致            | P0   |                 |     |
| 2.4    | workspace计算公式中错误应用双buffer系数      | P1   |                 |     |
| 2.5    | tiling key参数传递错误与dead code            | P1   |                 |     |
| 3.1    | !=A \                                        | \    | !=B恒真逻辑错误 | P0  |
| 3.2    | 空tensor输入未正确处理                       | P1   |                 |     |
| 3.3    | 多核任务分配未区分tail core和idle core       | P0   |                 |     |
| 3.4    | 尾块/边界条件处理遗漏                        | P1   |                 |     |
| 3.5    | 设置特殊状态后缺少return                     | P0   |                 |     |
| 3.6    | 新增类型/分支改变默认路径行为                | P1   |                 |     |
| 3.7    | 维度扩展后判断条件未适配                     | P1   |                 |     |
| 4.1    | OP_CHECK宏缺少return语句                     | P0   |                 |     |
| 4.2    | 结构体指针成员未初始化                       | P0   |                 |     |
| 4.3    | 先解引用后判空（先用后查）                   | P0   |                 |     |
| 4.4    | 可选输出tensor访问前必须检查null             | P0   |                 |     |
| 4.5    | 禁止op_host中非const全局容器                 | P0   |                 |     |
| 5.1    | PipeBarrier与FreeTensor时序颠倒              | P0   |                 |     |
| 5.2    | SetFlag/WaitFlag必须成对且无条件执行         | P0   |                 |     |
| 5.3    | DataCopy前后缺少流水线同步事件               | P0   |                 |     |
| 5.4    | 连续matmul操作间必须插入PipeBarrier          | P1   |                 |     |
| 5.5    | SetScheduleMode缺失                          | P1   |                 |     |
| 6.1    | 函数调用参数重复f(a,a)                       | P1   |                 |     |
| 6.2    | 矩阵k/n维度索引粘贴错误                      | P0   |                 |     |
| 6.3    | GM偏移量公式维度混淆                         | P1   |                 |     |
| 6.4    | 错误日志引用变量与判断条件不一致             | P2   |                 |     |
| 6.5    | 删除"未使用变量"前检查是否原应被使用         | P2   |                 |     |
| 7.1    | StorageShape/ViewShape混淆                   | P1   |                 |     |
| 7.2    | 接口调用默认参数与实际数据不一致             | P1   |                 |     |
| 7.3    | 属性索引硬编码顺序依赖                       | P1   |                 |     |
| 7.4    | 变体接口复用通用dtype校验                    | P1   |                 |     |
| 7.5    | 修改Init签名后调用点未全量同步               | P1   |                 |     |
| 7.6    | TilingParse注册类型与实际CompileInfo不一致   | P1   |                 |     |
| 7.7    | 新增数据类型支持时三处同步更新               | P1   |                 |     |
| 8.1    | 无符号整数>=0恒真导致无限循环                | P0   |                 |     |
| 8.2    | 链式比较与运算符优先级错误                   | P1   |                 |     |
| 9.1    | ascendc_config.json重复/冲突配置             | P1   |                 |     |
| 9.2    | CMake target依赖声明缺失                     | P2   |                 |     |
| 9.3    | 新平台binary/compute_units配置遗漏           | P1   |                 |     |
| 9.4    | shell脚本语法/逻辑错误                       | P1   |                 |     |
| 9.5    | 平台特定编译选项遗漏                         | P1   |                 |     |
| 9.6    | output paramType配置错误                     | P1   |                 |     |
| 10.1   | 平台版本精确匹配应改为范围比较               | P1   |                 |     |
| 10.2   | 硬件宏条件编译遗漏                           | P2   |                 |     |
| 11.1   | 多核场景全局内存初始化竞争                   | P0   |                 |     |
| 11.2   | splitK场景Bias加载时机错误                   | P1   |                 |     |
| 11.3   | L1全载优化状态失效检查                       | P1   |                 |     |
| 12.1   | 示例代码未经编译/运行验证                    | P2   |                 |     |
| 12.2   | 示例代码数据类型与API不匹配                  | P1   |                 |     |
| 13.1   | 搭车提交造成revert附带伤害                   | P2   |                 |     |
| 13.2   | 大规模重构应分步提交                         | P2   |                 |     |
| 13.3   | 公共基础设施耦合在算子PR中                   | P2   |                 |     |
| 13.4   | 移除binary.json变体前确认无下游调用          | P1   |                 |     |
| 14.1   | 错误码区分PARAM_*和INNER_*                   | P3   |                 |     |
| 14.2   | 宏定义中隐藏return语句                       | P2   |                 |     |
| 15.1   | 禁止注释EXPECT_*/ASSERT_*断言                | P3   |                 |     |
| SYS-01 | matmul是绝对缺陷震中                         | P0   |                 |     |
| SYS-02 | 整数溢出是最普遍残留风险                     | P0   |                 |     |
| SYS-03 | 编译warning隐藏真bug                         | P1   |                 |     |
| SYS-04 | 提交粒度过大导致质量失控                     | P2   |                 |     |

### P0规则汇总（致命级，需立即关注）

| 规则   | 规则名称                                     |     |                 |
| ------ | -------------------------------------------- | --- | --------------- |
| 1.1    | shape维度乘法必须使用int64_t                 |     |                 |
| 1.2    | GM偏移量在小类型域计算溢出                   |     |                 |
| 1.3    | 硬件指令参数位宽限制校验缺失                 |     |                 |
| 2.1    | tiling侧buffer计算与kernel侧InitBuffer不一致 |     |                 |
| 2.2    | tiling参数未区分normal核和tail核             |     |                 |
| 2.3    | TilingData结构体host-kernel不一致            |     |                 |
| 3.1    | !=A \                                        | \   | !=B恒真逻辑错误 |
| 3.3    | 多核任务分配未区分tail core和idle core       |     |                 |
| 3.5    | 设置特殊状态后缺少return                     |     |                 |
| 4.1    | OP_CHECK宏缺少return语句                     |     |                 |
| 4.2    | 结构体指针成员未初始化                       |     |                 |
| 4.3    | 先解引用后判空（先用后查）                   |     |                 |
| 4.4    | 可选输出tensor访问前必须检查null             |     |                 |
| 4.5    | 禁止op_host中非const全局容器                 |     |                 |
| 5.1    | PipeBarrier与FreeTensor时序颠倒              |     |                 |
| 5.2    | SetFlag/WaitFlag必须成对且无条件执行         |     |                 |
| 5.3    | DataCopy前后缺少流水线同步事件               |     |                 |
| 6.2    | 矩阵k/n维度索引粘贴错误                      |     |                 |
| 8.1    | 无符号整数>=0恒真导致无限循环                |     |                 |
| 11.1   | 多核场景全局内存初始化竞争                   |     |                 |
| SYS-01 | matmul是绝对缺陷震中                         |     |                 |
| SYS-02 | 整数溢出是最普遍残留风险                     |     |                 |

### 数据来源

- 仓库：ops-nn（1474次提交，380条缺陷，25.8%） + ops-nn-dev（2571次提交，612条缺陷，23.8%）
- 合计：4045次提交，992条缺陷提交
- 分析方法：逐条git show分析缺陷提交diff + 热点文件结构性审查 + Revert专项分析
- ops-nn分析周期：2025-09 ~ 2026-03
- ops-nn-dev Revert事件：17个独立事件（20条revert提交），70%在3天内回退
