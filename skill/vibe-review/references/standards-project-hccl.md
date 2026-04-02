# 【项目级】HCCL项目规则

## 1 HCCL项目级编码规则

- tokenId/tokenValue禁止入日志（网络安全红线）。RDMA rkey/lkey不属于敏感信息（避免误报）
- 返回值：用 `CHK_RET()` 检查，仅日志用 `CHK_PRT()`
- 日志：必须用 `HCCL_DEBUG`/`HCCL_INFO`/`HCCL_WARNING`/`HCCL_ERROR`/`HCCL_RUN_INFO`，禁止printf/cout
- 内存分配：堆上用 `NEW_NOTHROW`，智能指针用 `CHK_SMART_PTR_NULL()` 检查
- 错误上报：输入 `RPT_INPUT_ERR`，环境 `RPT_ENV_ERR`，内部 `RPT_INNER_ERR_PRT`，外调 `RPT_CALL_ERR`

## 2高价值缺陷模式

HCCL实际审查中发现的高频严重缺陷，保持高度敏感：

1. sizeof（容器）误用：`sizeof(std::vector<T>)` = 对象大小(24)，非数据大小。用 `.size()`
2. 值传递悬垂指针：`void F(Type p) { m_ = &p; }` — 值拷贝返回后销毁，m_悬垂
3. 秒转毫秒溢出：`uint32_t ms = seconds * 1000` — seconds > ~4.3M时溢出
4. 格式字符串不匹配：HCCL_INFO/ERROR中 `%` 占位符数/类型与实参不一致 = UB
5. 跨文件遗漏清理：删除结构体成员但未清理其他文件引用 → 内存布局错误。必须grep
6. 构造失败后析构崩溃：构造中途失败 → 析构清理未初始化成员（空指针delete）
7. 局部变量指针逃逸：返回局部变量指针/引用；lambda捕获局部变量但异步执行时已销毁
8. thread_local + dlclose：dlopen的.so中thread_local → dlclose时析构crash
9. gm偏移用int32：gm内存偏移/大小必须用int64，int32溢出
10. 整数不转浮点：可整数计算时禁止转浮点，必要时转更高精度整数类型
11. 通信算子融合同步缺失：多轮计算和集合通信之间需增加核间同步
12. 差一错误：循环边界、`\0`终结符、数组长度计算中的off-by-one

基于hcomm仓库428次提交（84次缺陷，19.6%）、hcomm-dev仓库488次提交（162次缺陷，33.1%）、hccl-dev仓库133次提交（10次缺陷，7.5%）的完整git历史分析，共1049次提交中256次缺陷相关提交，提炼的80条高价值审查规则。每条规则均有commit证据和实际代码支撑。

### 严重等级定义

| 等级 | 含义 | 影响范围                                       |
| ---- | ---- | ---------------------------------------------- |
| P0   | 致命 | 进程crash、数据损坏、安全漏洞、集群级故障      |
| P1   | 严重 | 功能错误、静默精度劣化、资源泄漏、性能严重退化 |
| P2   | 一般 | 边界条件异常、可观测性缺失、日志误导           |
| P3   | 建议 | 代码质量、可维护性、潜在隐患                   |

---

### 类别一：算法正确性（21次，25.0%）

集合通信算法实现中逻辑错误的各种表现形式，是HCCL项目最高频的缺陷类别。

#### 规则ALG-01: 变量名遮蔽导致成员未赋值

严重等级：P0

缺陷描述：构造函数或成员函数中，局部变量与成员变量同名，赋值写入局部变量而非成员变量。成员变量保持未初始化状态，后续使用产生未定义行为。

典型代码示例：

```cpp
// 缺陷代码 — src/platform/common/hccl_ip_address.cc
HcclIpAddress::HcclIpAddress(const Eid &eidInput)
{
    family = AF_INET6;        // 写入了同名局部变量，成员变量 this->family 未赋值
}

// 修复代码
HcclIpAddress::HcclIpAddress(const Eid &eidInput)
{
    this->family = AF_INET6;  // 显式指定成员变量
}
```

审查检查方法：
- 编译选项开启 `-Wshadow -Werror=shadow`，CI强制执行
- 构造函数体内赋值语句逐一确认目标是否为成员变量
- 优先使用初始化列表代替构造函数体内赋值

关联commit: `ef766683`

---

#### 规则ALG-02: 变量自比较(tautological compare)

严重等级：P0

缺陷描述：条件表达式中同一变量与自身比较（如 `x > x`），结果恒为常量，边界保护形同虚设。

典型代码示例：

```cpp
// 缺陷代码 — src/platform/task/dispatcher_aicpu.cc
CHK_PRT_RET(sqeContextBuffer->tailSqeIdx > sqeContextBuffer->tailSqeIdx,  // 自比较，恒false
    HCCL_ERROR("[DispatcherAicpu][MemcpyRtsq] tailSqeIdx[%u] > HCCL_SQE_MAX_CNT[%u]",
        sqeContextBuffer->tailSqeIdx, HCCL_SQE_MAX_CNT),
    HCCL_E_INTERNAL);

// 修复代码
CHK_PRT_RET(sqeContextBuffer->tailSqeIdx > HCCL_SQE_MAX_CNT,  // 与上限常量比较
    HCCL_ERROR("[DispatcherAicpu][MemcpyRtsq] tailSqeIdx[%u] > HCCL_SQE_MAX_CNT[%u]",
        sqeContextBuffer->tailSqeIdx, HCCL_SQE_MAX_CNT),
    HCCL_E_INTERNAL);
```

审查检查方法：
- 编译选项开启 `-Wtautological-compare`
- CHK_PRT_RET/CHK_RET条件表达式须与错误消息字符串交叉验证：消息说比较A和B，条件也应是A和B的比较

关联commit: `e1880dc1`

---

#### 规则ALG-03: 结构体/参数赋值遗漏

严重等级：P1

缺陷描述：填充参数结构体时遗漏了某个字段的赋值。特别是成对字段(dataType/outputDataType、sendType/recvType)，只赋值了一个而遗漏另一个。遗漏字段保持零值或默认值，导致运行时行为错误。

典型代码示例：

```cpp
// 缺陷代码 — src/legacy/framework/communicator/aicpu/aicpu_utils.cc
// FillKernelParam 函数中只设置了 outputDataType，遗漏了 dataType
kernelParam_->op.algOperator.outputDataType = HcclDataTypeToDataType(data->outputDataType);
CHECK_DATA_TYPE(kernelParam_->op.algOperator.outputDataType);
// dataType 未赋值！MC2场景下输入输出类型不同时精度错误

// 修复代码 — 补上 dataType 赋值
kernelParam_->op.algOperator.dataType = HcclDataTypeToDataType(data->dataType);
CHECK_DATA_TYPE(kernelParam_->op.algOperator.dataType);
kernelParam_->op.algOperator.outputDataType = HcclDataTypeToDataType(data->outputDataType);
CHECK_DATA_TYPE(kernelParam_->op.algOperator.outputDataType);
```

审查检查方法：
- 填充参数结构体时，列出结构体全部字段，逐一对比赋值语句，标记未赋值字段
- 成对字段（input/output、src/dst、send/recv前缀）确保不遗漏其一
- if/else分支中填充参数时，检查每个分支是否都做了完整赋值

关联commit: `7bc9e850`（dataType遗漏）， `1d171a92`（ipcMemDataSize遗漏）

---

#### 规则ALG-04: 同族executor/子类一致性缺陷

严重等级：P1

缺陷描述：继承体系中N个子类共享某段逻辑，但有1个子类遗漏。典型表现：N-1个executor有某个调用而1个没有；同一业务条件在不同子类中判断逻辑不一致。

典型代码示例：

```cpp
// 缺陷代码 — coll_all_to_all_v_direct_fullmesh_executor.cc
// 所有其他executor（AllGather、AllReduce、Broadcast等）的Orchestrate末尾都有：
//   CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
// 唯独 AlltoAllDirectFullmesh 遗漏了这一调用，导致aicpu cache功能异常

// 修复代码 — 在 Orchestrate 末尾补上
CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
```

审查检查方法：
- 新增executor子类时，用同族其他executor作为checklist逐项对比
- N-1个子类有某段逻辑而1个没有，应标记为"遗漏"而非"不需要"，除非有明确注释说明原因
- 日志字符串中的类名/tag前缀必须与当前文件名或类名匹配（检测copy-paste错误）

关联commit: `f7183c87`（LaunchTaskExtend遗漏）， `bb681c5c`（preloadCopyOpt条件不一致）， `71fd0b86`（numBlocks校验不一致）， `12f3680c`（日志类名copy-paste错误）

---

#### 规则ALG-05: 边界条件缺失

严重等级：P1

缺陷描述：算法选择或执行路径未覆盖所有有效配置。典型场景：pipeline算法假设至少3卡但未排除2卡场景；参数校验上限值与实际类型范围不一致。

典型代码示例：

```cpp
// 缺陷代码 — src/algorithm/impl/operator/all_reduce_operator.cc
// Pipeline算法至少需要3卡，但选择器未排除2卡场景
if (isOpbase && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
    algName = "AllReduceDeterPipelineExecutor";  // 2卡时也进入pipeline路径
}

// 修复代码 — 增加卡数守卫
if (isOpbase && algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE
    && deviceNumPerAggregation_ > DEVICE_TWO) {
    algName = "AllReduceDeterPipelineExecutor";
}
```

审查检查方法：
- 算法选择分支必须覆盖所有设备拓扑配置（1卡/2卡/多卡），每种配置标注预期行为
- 参数校验上限值须与参数实际类型范围一致（u8上限255、u16上限65535）
- Builder/fluent API构造对象时确认所有必要setter都被调用

关联commit: `666b6ab5`（2卡边界）， `7990e3f3`（repeat上限）， `8959e766`（selector遗漏SetRankSize）

---

#### 规则ALG-06: Get函数不应有Set副作用

严重等级：P1

缺陷描述：命名为Get*/Query*的查询函数内部调用了Set/Update/Insert操作，违反查询函数的纯净性约束。调用方无法预期查询操作会修改系统状态。

审查检查方法：
- Get*前缀函数体内不应包含Set/Update/Insert/Modify调用
- 方向对称性检查：`dst.*GetSource()` 或 `src.*GetTarget()` 模式标记为可疑

关联commit: `3323cecd`（GetEndpointDesc调用SetEndpointToIface）， `e4f59213`（查询函数修改成员变量curAlgName）

---

#### 规则ALG-07: 多阶段/多版本流程变量混淆

严重等级：P1

缺陷描述：多阶段流水线或多版本兼容代码中，阶段标识变量名相似导致混淆(intra vs inter、prepare vs launch、v1 vs v2)。典型表现：用错误的阶段变量作为条件判断。

典型代码示例：

```cpp
// 缺陷描述 — src/algorithm中CP pipeline实现
// intraState 和 interState 变量名相似
// 等待接收信息的代码本应检查 interState 却检查了 intraState
// 导致在错误的循环阶段执行等待操作
```

审查检查方法：
- 同一函数通过bool参数区分完全不同行为时，应拆分为两个独立函数
- 流水线多阶段的变量名须有明确区分前缀，审查条件是否指向正确阶段
- 多版本兼容路径中检查初始化顺序是否符合版本优先级

关联commit: `9bcb1bdc`（intra/inter混淆）， `af109f94`（prepare/launch阶段字段未区分）， `1a840065`（1.0/2.0初始化顺序）

---

#### 规则ALG-08: alltoallv等非均匀集合通信的per-rank偏移逻辑

严重等级：P1

缺陷描述：非均匀集合通信(alltoallv/allgatherv)中，每个rank的发送/接收偏移量和数据量不同。repeat模式下偏移量须逐轮累加，不能用通用标量运算代替。

典型代码示例：

```cpp
// 缺陷代码 — src/framework/device/aicpu_kfc/decoupler/comm_kfc_aicpu_server.cc
// alltoallv repeat模式下，每轮的offset未累加，始终操作相同数据段
for (u32 i = 0U; i < repeatCnt; ++i) {
    FormatOpData(msg, extMsg, i, data);  // extMsg中offset未变化
}

// 修复代码 — 每轮累加per-rank的offset
for (u32 i = 0U; i < repeatCnt; ++i) {
    FormatOpData(msg, extMsg, rankNum_, i, data);
    // FormatOpData内部：
    // for (u32 j = 0; j < rankNum; ++j) {
    //     extMsg.sendOffset[j] += extMsg.sendCounts[j];
    //     extMsg.recvOffset[j] += extMsg.recvCounts[j];
    // }
}
```

审查检查方法：
- alltoallv/allgatherv等非均匀算子的repeat/分片逻辑须逐rank独立处理偏移
- 检查循环体内是否正确更新了所有per-rank的状态变量

关联commit: `fab6dbb7`

---

### 类别二：配置与兼容性（13次，15.5%）

多设备类型、多编译环境、多版本协议的适配问题。

#### 规则CFG-01: 跨版本协议OpCode修改须保持向后兼容

严重等级：P0

缺陷描述：修改已发布的OpCode对应的数据结构时，直接复用原OpCode编号。新版本发送新结构体给旧版本，旧版本按老结构体解析导致协议不兼容。

典型代码示例：

```cpp
// 缺陷代码 — src/platform/hccp/inc/private/network/ra_rs_comm.h
// RA_RS_TLV_INIT = 87 的数据结构增加了 moduleType 字段，但OpCode编号未变
// 旧版本按老结构体解析新数据，字段错位

// 修复代码 — 旧版本保持原编号，新版本分配新编号
RA_RS_TLV_INIT_V1 = 87,     // 旧版本保持原编号
RA_RS_TLV_INIT = 110,       // 新版本分配新编号

// 发送前查询对端版本
ret = RaHdcGetInterfaceVersion(phyId, RA_RS_TLV_INIT, &interfaceVersion);
if (ret == 0 && interfaceVersion >= RA_RS_OPCODE_BASE_VERSION) {
    opCode = RA_RS_TLV_INIT;        // 对端支持新版本
} else {
    opCode = RA_RS_TLV_INIT_V1;     // 对端仅支持旧版本
}
```

审查检查方法：
- 修改已发布OpCode的数据结构时，必须分配新OpCode编号
- 保留旧OpCode的处理函数（至少一个版本周期）
- 发送前须查询对端版本能力，选择兼容的OpCode

关联commit: `5ad2ced6`

---

#### 规则CFG-02: 设备类型分支须全面覆盖

严重等级：P1

缺陷描述：新增硬件资源初始化或引擎类型时，仅在部分设备类型分支中处理，其他设备类型分支遗漏。或者新增引擎类型后未全局检查所有引擎分支代码。

审查检查方法：
- 新增硬件资源初始化须审查是否适用所有设备类型（910_93/910_95/A3等）
- 新增引擎类型后须全局grep所有按引擎分支的代码确认归属
- 按设备类型分支时审查各分支功能等价性

关联commit: `099fe2a8`（心跳socket类型判断不一致）， `07c52708`（910_95 profiling不兼容）， `91fbd1d6`（A3 AIV跨节点回退1700行）， `9d75f557`（AIV引擎归入CPU分支），hcomm-dev `802c0411`/`c6f8cc36`（枚举/类型分支遗漏7例），hccl-dev `19d20206`（设备类型判断遗漏环境变量组合条件）

---

#### 规则CFG-03: 常量不应承担多重语义

严重等级：P2

缺陷描述：同一个常量既作为默认值又作为哨兵值（标识"未配置"），甚至其数值本身超出合法范围。

典型代码示例：

```cpp
// 缺陷代码 — src/legacy/framework/topo/new_topo_builder/rank_table_info/new_rank_info.h
constexpr unsigned int MAX_VALUE_DEVICEPORT = 65536;  // 既做默认值又做哨兵值，且超过合法端口上限65535
u32 devicePort{MAX_VALUE_DEVICEPORT};

// 修复代码 — 分离默认值与上限
constexpr unsigned int DEFAULT_VALUE_DEVICEPORT = 60001;  // 明确的默认端口
constexpr unsigned int MAX_VALUE_DEVICEPORT = 65535;       // 合法端口上限
u32 devicePort{DEFAULT_VALUE_DEVICEPORT};
```

审查检查方法：
- 一个常量只承担一个语义
- 端口号上限为65535而非65536
- 硬件常量修改须在注释中标注取值依据（spec文档引用）

关联commit: `2bfa02d1`（端口默认值/哨兵值混淆）， `fb56d64b`（CCU loop count常量值错误64->128）

---

#### 规则CFG-04: v1/v2接口分发完整性

严重等级：P1

缺陷描述：op_base.cc中的公开API须同时有v1和v2版本的分发入口。新增或修改API时遗漏HCCLV2_FUNC_RUN分发，导致v2调用路径无法到达。

审查检查方法：
- op_base.cc中所有公开API须有HCCLV2_FUNC_RUN分发（可自动化扫描）
- 字符串到枚举映射表中key须与枚举值名称一致

关联commit: `6383b2bc`（HcclGetCommConfigCapability缺分发）， `edf73e80`（协议名字符串映射不匹配），hcomm-dev `e25c6484`（HcclGetHcclBuffer等多个API缺少V2分发，5例）

---

### 类别三：并发问题（10次，11.9%）

通信库最核心的缺陷类型，涵盖竞态、死锁、原子性违反、同步时序。

#### 规则CON-01: 共享数据写后更新标志须有内存屏障

严重等级：P0

缺陷描述："先写数据、后更新标志位"的生产者-消费者模式中，缺少memory fence。CPU或编译器可能重排store指令，导致消费者看到标志位更新但数据还未写入。

典型代码示例：

```cpp
// 缺陷代码 — src/platform/resource/mem/hdc.cc
// HDCommunicate::Write 函数
CHK_RET(memcpy_s(hostMem_.data() + offset, hostMem_.size(), value, length));
// 缺少 memory fence！
u32 tail = *tailCntAddr_;
tail++;
*tailCntAddr_ = tail;   // 消费者可能看到tail更新但数据未就绪

// 修复代码
CHK_RET(memcpy_s(hostMem_.data() + offset, hostMem_.size(), value, length));
std::atomic_thread_fence(std::memory_order_seq_cst);  // 保证数据对其他线程可见
u32 tail = *tailCntAddr_;
tail++;
*tailCntAddr_ = tail;
```

审查检查方法：
- "先写数据、后更新标志位"模式必须在两步之间插入memory fence
- 跨线程/跨进程的共享内存通信，标志位更新前须有acquire-release语义保证

关联commit: `0947b660`, hcomm-dev `a014e919`（HDCommunicate:Write中memcpy与tailCnt更新间缺屏障）

---

#### 规则CON-02: Record/Wait同步顺序——先Wait依赖方再Record通知上游

严重等级：P0

缺陷描述：集合通信broadcast中，中继rank先Record通知root"数据已取走"，然后才Wait其他rank完成。root可能在下游未完成时就覆盖buffer。

典型代码示例：

```cpp
// 缺陷代码 — src/algorithm/base/alg_aiv_template/aiv_broadcast_910b_bigdata.h
// 中继rank的同步逻辑
Record(tag, root, AivNotifyType::DataSignal, 0, ifPingpong);  // 先通知root
PipeBarrier<PIPE_ALL>();
for (uint32_t remoteRank = 0; remoteRank < rankSize_; remoteRank += 1) {
    if (remoteRank != root && remoteRank != rank_) {
        Wait(tag, remoteRank, AivNotifyType::DataSignal, 0, ifPingpong);  // 后等待下游
    }
}

// 修复代码 — 先等待所有下游完成，最后通知root
for (uint32_t remoteRank = 0; remoteRank < rankSize_; remoteRank += 1) {
    if (remoteRank != root && remoteRank != rank_) {
        Wait(tag, remoteRank, AivNotifyType::DataSignal, 0, ifPingpong);
        PipeBarrier<PIPE_ALL>();
    }
}
PipeBarrier<PIPE_ALL>();
Record(tag, root, AivNotifyType::DataSignal, 0, ifPingpong);  // 最后通知root
```

审查检查方法：
- Record/Wait顺序原则：先Wait依赖方完成，再Record通知上游
- 集合通信中root/coordinator应是最后退出的角色

关联commit: `a0f05be2`

---

#### 规则CON-03: publish ordering——先完成工作再修改对外可见状态

严重等级：P0

缺陷描述：多线程环境下先重置状态标记再清理资源。其他线程看到状态已重置就认为可以使用，但资源实际还未清理完成。

典型代码示例：

```cpp
// 缺陷代码 — src/framework/device/framework/aicpu_communicator.cc
// Retry路径
ret = ResetOpRetryException(param.opType);       // 先reset状态
dfxExtendInfo_.pollStatus = PollStatus::kDefault; // 此时其他线程已看到状态重置
dfxExtendInfo_.cqeStatus = dfx::CqeStatus::kDefault;

// 修复代码 — 先清理资源再reset状态
dfxExtendInfo_.pollStatus = PollStatus::kDefault;
dfxExtendInfo_.cqeStatus = dfx::CqeStatus::kDefault;
ret = ResetOpRetryException(param.opType);        // 最后reset状态
```

审查检查方法：
- publish ordering原则：先完成实际清理/初始化工作，再修改对外可见的状态标志
- 状态重置函数调用应在所有资源清理操作之后

关联commit: `75659d24`, hcomm-dev `4d1a3a60`/`08b2658a`/`87da187b`（先启动线程后设标志、executor_过早置nullptr等执行顺序错误5例）

---

#### 规则CON-04: 并发delete须加锁保护

严重等级：P0

缺陷描述：多线程可能并发调用含delete操作的销毁函数，无锁保护导致double-free。常见模式：check-then-act（先判空再delete）在并发下不安全。

典型代码示例：

```cpp
// 缺陷代码 — src/platform/resource/dispatcher_ctx/dispatcher_ctx.cc
HcclResult DispatcherCtx::Destroy()
{
    if (dispatcher_ != nullptr) {    // 两个线程可能同时通过此检查
        delete dispatcher;           // double-free
        dispatcher_ = nullptr;
    }
}

// 修复代码 — 加互斥锁
HcclResult DispatcherCtx::Destroy()
{
    const std::lock_guard<std::mutex> lock(destroyMutex_);
    if (dispatcher_ != nullptr) {
        delete dispatcher;
        dispatcher_ = nullptr;
    }
}
```

审查检查方法：
- delete后置nullptr的模式在多线程环境下不安全，必须加锁
- 审查所有Destroy/Cleanup函数是否可能被并发调用

关联commit: `36994739`, hcomm-dev `7be3b8b8`/`cd056a5e`（DispatcherCtx:Destroy并发delete、reqHandle异步引用时直接free等3例）

---

#### 规则CON-05: atomic变量的所有读取必须用.load()

严重等级：P1

缺陷描述： `std::atomic`变量用隐式转换（如 `==` 操作符）读取时，在某些编译器/优化级别下可能不产生原子load指令。

典型代码示例：

```cpp
// 缺陷代码 — src/framework/communicator/impl/hccl_communicator_host.cc
if (g_enableBackupLinkCommCount == 0) {  // 隐式非原子读

// 修复代码
if (g_enableBackupLinkCommCount.load() == 0) {  // 显式原子load
```

审查检查方法：
- 所有`std::atomic`变量的读取点必须使用`.load()`显式调用
- 可用clang-tidy的`bugprone-use-after-move`等checker辅助检测

关联commit: `6b354394`

---

#### 规则CON-06: 单例/引用计数Init方法须区分"只需一次"和"每次都需"

严重等级：P1

缺陷描述：单例模式的Init方法中，部分逻辑只需首次执行（如注册回调），部分逻辑每次创建通信域都需执行（如设置白名单）。引用计数>1时直接跳过整个Init，导致后续通信域缺少必要配置。

审查检查方法：
- 单例Init方法中明确标注哪些操作是once-only、哪些是every-time
- 引用计数递增路径也须检查是否有必须每次执行的逻辑

关联commit: `eb9be21b`

---

#### 规则CON-07: thread_local变量须在线程入口点统一设置

严重等级：P1

缺陷描述： `thread_local`变量（如`gDispatcherCtx`）的设置逻辑被嵌在某个业务函数深处，新线程如果没有经过该函数就调用了依赖此变量的其他函数，访问到未初始化的thread_local值。

典型代码示例：

```cpp
// 缺陷代码 — src/framework/device/framework/aicpu_communicator.cc
// SetDispatcherCtx(dispatcherCtx_) 的调用嵌在 RegisterOpInfo 内部
HcclResult HcclCommAicpu::RegisterOpInfo(void* opInfo, u32 size)
{
    CHK_RET(SetDispatcherCtx(dispatcherCtx_));   // thread_local设置埋在此
    CHK_RET(taskExecption_.RegisterOpInfo(opInfo, size));
}
// 但 HcommAcquireComm 在另一个线程调用时，没走 RegisterOpInfo，gDispatcherCtx 未初始化

// 修复代码 — 抽取为独立方法，在每个线程入口显式调用
HcclResult HcclCommAicpu::SetDispatcherCtxOnThread()
{
    CHK_RET(SetDispatcherCtx(dispatcherCtx_));
    return HCCL_SUCCESS;
}

// hccl_api_data.cc — 线程入口处显式调用
int32_t HcommAcquireComm(const char* commId)
{
    HcclCommAicpu *hcclComm = AicpuHcclProcess::AicpuGetCommbyGroup(commId);
    CHK_RET(hcclComm->SetDispatcherCtxOnThread());   // 确保当前线程设置了ctx
    ...
}
```

审查检查方法：
- thread_local变量的设置须作为线程入口的第一步操作，不应埋在业务函数内部
- 新增线程入口时须审查是否依赖了已有的thread_local变量
- thread_local变量的所有使用点须确认该线程已经过设置路径

关联commit: `a6e4d199`

---

#### 规则CON-08: 全局/静态变量无锁访问

严重等级：P0

缺陷描述：static全局变量在多线程/多流场景下被并发读写，无任何同步机制。通信框架天然多线程（多device、多通信域、主线程+后台轮询线程），18个热点文件中15个存在此类问题。

典型代码示例：

```cpp
// 缺陷 — static全局数组在MC2多流场景下被多线程并发写
static uint8_t g_expectPrepareId[MAX_QUE_NUM];  // 多线程写同一位置
// 线程A: g_expectPrepareId[queueId] = idA;
// 线程B: g_expectPrepareId[queueId] = idB;  // 覆盖线程A的写入 -> 消息ID不匹配 -> hang

// 修复 — 改为thread_local
static thread_local uint8_t g_expectPrepareId[MAX_QUE_NUM];
```

审查检查方法：
- static全局变量在多线程场景下是否需要thread_local或显式同步？
- 对static/全局变量的新增或修改，必须标注线程安全声明（thread_local/互斥锁/仅单线程访问）

关联commit: hcomm-dev `341e7893`

---

#### 规则CON-09: TOCTOU竞态

严重等级：P1

缺陷描述：持锁检查后释放锁再操作，检查结果在释放锁后失效。两线程可同时通过"不存在"检查，各自创建同名资源导致覆盖。

典型代码示例：

```cpp
// 缺陷 — 持锁检查group不存在后解锁再创建，两线程可同时通过检查
{
    std::lock_guard<std::mutex> lock(groupMutex_);
    if (groupMap_.find(groupName) != groupMap_.end()) {
        return HCCL_E_EXIST;
    }
}  // 解锁
// 窗口期: 另一个线程也通过了检查
CHK_RET(CreateGroup(groupName));  // 两个线程都创建同名group

// 修复 — 检查和创建在同一临界区内
{
    std::lock_guard<std::mutex> lock(groupMutex_);
    if (groupMap_.find(groupName) != groupMap_.end()) {
        return HCCL_E_EXIST;
    }
    CHK_RET(CreateGroup(groupName));
    groupMap_[groupName] = group;
}
```

审查检查方法：
- 持锁检查结果是否在解锁后被使用(TOCTOU)? 操作和检查是否在同一个临界区内？

关联commit: hcomm-dev `7b12579d`

---

### 类别四：日志与调试（8次，9.5%）

可审查性高，多数可通过编译器警告或lint工具自动捕获。

#### 规则LOG-01: 格式化字符串占位符须与实参类型匹配

严重等级：P2

缺陷描述：printf-style格式化中，`%s`传入int、`%u`传入u64、参数与占位符错位。可能导致日志输出垃圾值或crash。

审查检查方法：
- 编译选项开启 `-Wformat -Werror=format`
- `std::string`传入C格式化函数须用`.c_str()`
- 禁止字符串字面量中用 `\` 续行（会将下一行前导空格纳入字符串）

关联commit: `35e32c7c`（%s传入int）， `6a6eac0f`（u64用%u、u32用%llu），hcomm-dev `5e446705`/`f715f167`（%s传int导致段错误、%u打印指针等4例）

---

#### 规则LOG-02: 日志前缀/tag必须与所在类名/函数名匹配

严重等级：P3

缺陷描述：copy-paste代码后未更新日志中的模块名/类名标识，误导定位。

审查检查方法：
- 日志前缀/tag必须与所在函数名或类名匹配
- 日志模块ID必须在枚举中有明确定义
- 可用脚本自动检测日志tag与代码位置的一致性

关联commit: `7ff92abc`, `112766de`（缺少[HCCL_ENV]前缀）， `12f3680c`（日志类名copy-paste错误）， `ed50e7eb`（日志模块ID未在枚举中定义）

---

#### 规则LOG-03: 快速路径必须与标准路径保持功能对等

严重等级：P2

缺陷描述：cache hit或fast path跳过了profiling/tracing/DFX信息采集，导致性能分析和问题定位时信息缺失。

审查检查方法：
- 所有快速路径(cache hit/fast path)必须与标准路径保持profiling/tracing/logging功能对等
- 新增fast path时逐项对比标准路径的DFX调用

关联commit: `891665ac`（SQE缓存命中缺profiling）， `3a9b61f2`（CCU DFX结构体缺字段）

---

#### 规则LOG-04: 日志洪泛/变量引用错误

严重等级：P3

缺陷描述：高频路径无日志抑制导致刷屏；日志中引用已被修改的累积值而非原始值，导致日志信息误导排查。

典型代码示例：

```cpp
// 缺陷 — 高频opcode每次调用都打印debug日志
HcclResult RaGetOpRight(u32 opcode) {
    HCCL_DEBUG("get op right for opcode[%u]", opcode);  // 高频刷屏
    ...
}

// 修复 — 引入日志抑制机制
HcclResult RaGetOpRight(u32 opcode) {
    if (!RaIsOpcodeLogSuppressed(opcode)) {
        HCCL_DEBUG("get op right for opcode[%u]", opcode);
    }
    ...
}
```

```cpp
// 缺陷 — 日志打印使用了已被累加额外预留空间的scratchBufSize
scratchBufSize += extraReserve;
HCCL_INFO("cclBufferSize[%llu]", scratchBufSize);  // 打印的是累加后的值

// 修复 — 在累加前记录原始值
HCCL_INFO("cclBufferSize[%llu]", cclBufferSize);   // 使用原始变量名
scratchBufSize += extraReserve;
```

审查检查方法：
- 日志中引用的变量是否是当前值而非已被修改的累积值？
- 高频调用路径上的日志是否有抑制/采样机制？

关联commit: hcomm-dev `4425a342`, `802c0411`

---

### 类别五：资源生命周期（8次，9.5%）

资源的创建、使用、销毁三阶段管理不当。

#### 规则RES-01: early return跳过必要初始化/清理

严重等级：P1

缺陷描述：函数包含多个early return路径，某个return跳过了后续必要的初始化或清理逻辑。

审查检查方法：
- 函数包含多个early return时，逐一审查每个return路径是否遗漏后续必要初始化/清理
- 调试用的`return HCCL_SUCCESS`须在提交前移除（lint可检测unreachable code）

关联commit: `82989927`（early return跳过TpManager:Init）， `c5443da0`（调试遗留return跳过整个逻辑），hcomm-dev `d8400fd2`/`bc8c5036`（hrtMalloc在CHK_RET失败路径未释放、buffer插入后Init失败残留等4例）

---

#### 规则RES-02: map/cache的key必须能唯一标识资源

严重等级：P1

缺陷描述：map/cache的key设计不充分，未包含所有区分资源的关键属性。不同语义的资源请求命中同一key，复用了错误的资源。

典型代码示例：

```cpp
// 缺陷描述 — netDevCtxMap_ 的key仅用IP，同IP不同端口复用错误NetDevCtx
// BatchSendRecv 资源key仅用algName/opTag，不同remote rank组合复用错误资源
// 修复：将端口/remote rank信息编入key
```

审查检查方法：
- 设计map/cache的key时，问"同一key下是否可能出现不同语义的资源请求？"
- key须覆盖所有区分资源的关键属性（IP+端口、algName+remoteRank等）
- 资源销毁顺序须按依赖关系图反序

关联commit: `43069da0`（IP不含端口）， `18752141`（key不含remote rank）

---

#### 规则RES-03: context切换附近的内存操作须审查执行context

严重等级：P1

缺陷描述：代码中切换到不同的设备context后分配/释放内存，但操作应在原context下执行。

审查检查方法：
- context切换(SetDevice/SwitchContext)附近的内存分配须审查执行context是否正确
- 枚举值switch/if匹配需覆盖所有有效值

关联commit: `3b8ac218`（DPU context下错误分配NPU内存）， `6b9278e4`（图模式remote memory更新遗漏）

---

#### 规则RES-04: 析构路径须完整——引用计数递减也须检查局部资源释放

严重等级：P1

缺陷描述：引用计数递减到非零值时直接返回，但该路径上可能有局部获取的资源（如映射handle）需要释放。

审查检查方法：
- 分配/映射的资源在释放函数中须有明确释放路径
- 引用计数递减的每个分支都须检查是否有局部资源需释放
- 析构函数中的日志/DFX调用应在所有资源释放操作之前（避免日志依赖先被销毁）

关联commit: `a9fea200`（refCount>1分支遗漏释放handle）， `7d914168`（析构顺序错误导致日志依赖先销毁）

---

#### 规则RES-05: 资源释放顺序错误/释放后未置空

严重等级：P0

缺陷描述：组件间存在资源依赖，析构顺序不正确导致UAF或报错。释放后handle/指针未置空导致重入时double free。

典型代码示例：

```cpp
// 缺陷 — communicator析构时先销毁AlgResource，
// 但ChannelManager中持有的transport link依赖底层资源已被清理
HcclCommunicator::~HcclCommunicator() {
    DestroyAlgResource();       // 先释放算法资源
    // channelManager_析构时transport link访问已释放资源 -> 报错
}

// 修复 — 新增releaseChannel_()回调注入析构流程
HcclCommunicator::~HcclCommunicator() {
    DestroyAlgResource();
    releaseChannel_();  // 通过回调释放channel transport，避免UAF
}
```

```cpp
// 缺陷 — UnimportAllJetty释放资源后handle仍保留旧值，重入时double free
void CcuComponent::UnimportAllJetty() {
    for (auto &jetty : jettyList_) {
        UrmaUnimportJetty(jetty.handle);
        // 遗漏: jetty.handle = INVALID_HANDLE;
    }
}

// 修复 — 释放后立即置空
void CcuComponent::UnimportAllJetty() {
    for (auto &jetty : jettyList_) {
        UrmaUnimportJetty(jetty.handle);
        jetty.handle = INVALID_HANDLE;  // 防止重入double free
    }
}
```

审查检查方法：
- 当组件A持有组件B创建的资源引用时，析构流程是否保证A的引用先释放？
- 资源释放后是否立即将handle/指针置空？

关联commit: hcomm-dev `f45581ee`, `e17cfff4`

---

### 类别六：错误处理（7次，8.3%）

#### 规则ERR-01: 异常上报操作须受前置条件保护

严重等级：P2

缺陷描述：异常上报/错误通知在正常路径上也无条件执行，导致正常操作触发虚假的异常打印。

审查检查方法：
- 异常上报操作（SendTaskExceptionByMBox等）必须受前置条件判断保护
- 重试成功路径不应触发异常上报

关联commit: `96087ffb`（重试成功也触发异常打印）

---

#### 规则ERR-02: 项目有统一异常处理宏时不允许裸try-catch

严重等级：P2

缺陷描述：项目提供了统一异常处理宏（如TRY_CATCH_PROCESS_THROW），但开发者手写try-catch吞掉异常只打日志return，导致异常类型信息丢失。

典型代码示例：

```cpp
// 缺陷代码 — src/legacy/unified_platform/resource/ccu_transport/ccu_transport.cpp
try {
    std::unique_lock<std::shared_timed_mutex> lock(transMutex);
    status = StateMachine();
} catch (HcclException &e) {
    HCCL_ERROR(e.what());
    return CcuTransport::TransStatus::CONNECT_FAILED;  // 吞掉异常
} catch (...) {
    HCCL_ERROR("Unknown error");
    return CcuTransport::TransStatus::CONNECT_FAILED;  // 吞掉异常
}

// 修复代码 — 使用统一宏
auto lockAndStatuMachine = [&]() {
    std::unique_lock<std::shared_timed_mutex> lock(transMutex);
    status = StateMachine();
};
TRY_CATCH_PROCESS_THROW(InternalException, lockAndStatuMachine(),
    "CcuTransport GetStatus() Error", { transStatus = CONNECT_FAILED; });
```

审查检查方法：
- 搜索裸try-catch块，确认是否应使用项目统一宏
- `catch(...)` 块不应默默return，至少需要向上传播异常类型信息

关联commit: `69c73166`

---

#### 规则ERR-03: C接口函数不应抛C++异常

严重等级：P0

缺陷描述：对外接口函数内部调用了含`THROW`的inline函数（如`MC2DataType()`、`MC2ReduceType()`），当参数非法时抛出C++异常。若调用方无法处理异常（如通过dlsym加载或跨语言调用），异常穿越接口边界导致`std::terminate()`直接abort进程。

典型代码示例：

```cpp
// 缺陷代码 — src/legacy/framework/entrance/op_base/op_base_v2.cc
// HcclSetOpSrcDataTypeV2 对外接口函数
DataType mc2DataType = MC2DataType(static_cast<HcclDataType>(srcDataType));
// MC2DataType() 内部含 THROW<Hccl::CcuApiException>，参数非法时抛异常

// 修复代码 — 先边界检查，直接查表，返回错误码
if (srcDataType >= (sizeof(MC2_DATA_TYPE) / sizeof(MC2_DATA_TYPE[0]))) {
    HCCL_ERROR("srcDataType[%u] invalid.", srcDataType);
    return HCCL_E_PARA;    // 返回错误码而非throw
}
opArgsPtr->srcDataType = MC2_DATA_TYPE[srcDataType];
```

审查检查方法：
- 对外接口函数体内不应有未catch的throw路径（含间接调用的inline函数中的THROW）
- 接口边界处使用错误码返回而非异常

关联commit: `9939a862`

---

#### 规则ERR-04: CHK_RET调用顺序须与业务逻辑匹配

严重等级：P2

缺陷描述：带CHK_RET宏包裹的函数调用放在条件判断之前，非相关场景的函数失败会阻断整个函数。

审查检查方法：
- CHK_RET包裹的函数调用应延迟到确认需要其返回值处
- if-else分支设置错误状态后检查是否遗漏return

关联commit: `9476c6df`（CHK_RET调用顺序不当）， `e025b6c5`（if-else缺return导致fallthrough）

---

#### 规则ERR-05: 预期内返回值不应用error日志级别

严重等级：P2

缺陷描述：调用底层接口时，某些返回值（如`-ENOTSUPP`）是预期内的"功能不支持"语义，但代码统一用error级别日志打印，产生大量误导性错误日志，掩盖真正的错误。

典型代码示例：

```c
// 缺陷代码 — src/platform/hccp/rdma_agent/peer/ra_peer.c
ret = RsSetQpLbValue(qpHandle->phyId, qpHandle->rdevIndex, qpHandle->qpn, lbValue);
if (ret != 0) {
    hccp_err("[set][lbValue]RsSetQpLbValue failed ret:%d", ret);  // -ENOTSUPP也打error
}

// 修复代码 — 区分预期内的"不支持"和真正的错误
if (ret != 0) {
    if (ret == -ENOTSUPP) {
        hccp_run_warn("[set][lbValue]RsSetQpLbValue unsuccessful ret:%d", ret);  // 降为warn
    } else {
        hccp_err("[set][lbValue]RsSetQpLbValue failed ret:%d", ret);
    }
}
```

审查检查方法：
- 调用可能返回"不支持/不可用"语义值的底层接口时，须区分日志级别
- `-ENOTSUPP`、`-EOPNOTSUPP`、`HCCL_E_NOT_SUPPORT`等返回值应用warn而非error
- error级别日志应保留给真正需要人工介入的错误

关联commit: `2fcde546`, hcomm-dev `6972024c`/`2adb85d7`（CHK_RET宏在可恢复场景隐式打ERROR、AIV fallback等3例）

---

### 类别七：缓存一致性（6次，7.1%）

缓存/复用机制中key生成不完整或缓存数据过期的问题。HCCL项目特有的高频模式。

#### 规则CACHE-01: cache key须覆盖所有影响执行路径的状态维度

严重等级：P1

缺陷描述：缓存key的维度不够，新增的分支条件未反映在key中，导致不同执行路径命中同一缓存。

典型代码示例：

```cpp
// 缺陷代码 — src/framework/device/framework/aicpu_cache_manager.cc
// AlltoAllV cache key没有区分 isBigCount，big/small count两种不同SQE编排方案命中同一缓存

// 修复代码 — 将 isBigCount 编码进 inputSize 字段作为key的一部分
bool isBigCountForAlltoallv = false;
CHK_RET(IsBigCountForAlltoallv(param, topoinfo, isBigCountForAlltoallv));
inputSize = isBigCountForAlltoallv ? 1 : 0;  // 编码进cache key
```

审查检查方法：
- 新增if/switch分支逻辑如果影响执行路径，须检查是否反映在cache key中
- cache key维度checklist：opType、algName、dataType、count/size、拓扑信息、特殊标志位
- 分支条件与缓存key必须使用相同数据源

关联commit: `e0744b7b`（缺isBigCount维度）， `18752141`（缺remote rank信息）

---

#### 规则CACHE-02: 缓存复用前须逐字段确认是否需要刷新

严重等级：P1

缺陷描述：缓存命中后直接复用所有缓存字段，但某些字段（如stream handle）在每次调用时可能变化，复用过期值导致错误。

典型代码示例：

```cpp
// 缺陷代码 — src/framework/communicator/impl/hccl_communicator_host.cc
// ExecOpCache 缓存命中时复用了旧的stream，没有用当前OpParam中的新stream

// 修复代码 — 在缓存复用时刷新可变字段
cacheInfo.resourceArgs.stream = opParam.stream.ptr();  // 刷新stream
```

审查检查方法：
- 缓存复用前逐字段确认哪些可能变化，对可变字段加刷新逻辑(staleness checklist)
- 句柄类字段(stream、context、device handle)几乎总需要刷新
- 缓存用于恢复逻辑时检查是否保存了所有依赖的输入状态

关联commit: `43dab3e2`（stream未刷新）， `dd053d5b`（algName未随缓存恢复），hcomm-dev `3c24c0fe`/`b3975b85`（ExecOpCache中stream字段和workflowMode维度缺失3例）

---

#### 规则CACHE-03: 函数调用前后使用同一成员变量时须确认中间调用是否有副作用

严重等级：P1

缺陷描述：在调用某函数前后都使用了同一成员变量，但该函数有副作用会修改这个成员变量。后续代码使用的是被修改后的值而非预期的原始值。

典型代码示例：

```cpp
// 缺陷代码 — src/legacy/framework/communicator/communicator_impl.cc
OpAcceleratorStateFallback();  // 副作用：修改了 curAlgName
// ...
opAcceStateCache.insert({{curOpParams.opType, curAlgName}, ...});  // 用了被修改后的值

// 修复代码 — 先保存原始值
string needFallBackAlgName = curAlgName;   // 保存原始值
OpAcceleratorStateFallback();              // 可能修改 curAlgName
// ...
opAcceStateCache.insert({{curOpParams.opType, needFallBackAlgName}, ...});  // 用保存的值
```

审查检查方法：
- 函数调用前后使用同一成员变量时，审查被调用函数是否修改该变量
- 有副作用的函数须在注释或命名中体现（如OpAcceleratorStateFallback暗示会修改状态）

关联commit: `e4f59213`

---

#### 规则CACHE-04: 运行时分支条件与缓存key必须使用相同数据源

严重等级：P1

缺陷描述：运行时判断走哪条执行路径用了变量A，但生成缓存key时用了变量B。A和B在大多数情况下一致，但在边界情况（如最后一轮最后一张卡）下不一致，导致缓存复用时走错路径。

审查检查方法：
- 运行时分支条件和缓存key生成必须使用完全相同的数据源
- 用虚函数或统一getter抽象数据源，避免两处分别计算

关联commit: `b0e6a8b7`（运行时用curSize_判断，key用perRankAvgDataSize_）

---

### 类别八：内存管理（4次，4.8%）

#### 规则MEM-01: 禁止将局部容器的内部指针暴露给外部

严重等级：P0

缺陷描述：将栈上局部vector的`.data()`指针通过输出参数返回。函数返回后vector析构，调用方拿到悬垂指针。

典型代码示例：

```cpp
// 缺陷代码 — src/framework/hcom/hcom.cc
HcclResult HcomGetandClearOverFlowTasks(const char *group,
    hccl::HcclDumpInfo **hcclDumpInfoPtr, u32 *len)
{
    std::vector<hccl::HcclDumpInfo> hcclDumpInfo;  // 局部vector
    CHK_RET(hcclComm->GetandClearOverFlowTasks(hcclDumpInfo));
    *hcclDumpInfoPtr = hcclDumpInfo.data();  // 返回局部vector内部指针
    *len = hcclDumpInfo.size();
    return HCCL_SUCCESS;
}  // hcclDumpInfo析构，*hcclDumpInfoPtr成为悬垂指针

// 修复代码 — malloc独立内存，memcpy数据
if (hcclDumpInfo.size() > 0) {
    *hcclDumpInfoPtr = static_cast<hccl::HcclDumpInfo*>(
        malloc(hcclDumpInfo.size() * sizeof(hccl::HcclDumpInfo)));
    CHK_PTR_NULL(*hcclDumpInfoPtr);
    memcpy_s(*hcclDumpInfoPtr, ..., hcclDumpInfo.data(), ...);
}
```

审查检查方法：
- 禁止将局部容器(vector/string/array)的内部数据指针通过输出参数或返回值暴露给外部
- 需要返回数据所有权时，使用malloc+memcpy或std:unique_ptr

关联commit: `6784944a`（此缺陷由`4e82ec25`的修复引入——缺陷修复引入新缺陷的典型案例）

---

#### 规则MEM-02: IPC共享内存注册须检查重复注册

严重等级：P1

缺陷描述：P2P传输中同一块物理内存被重复注册IPC映射，大规模通信下资源耗尽OOM。

审查检查方法：
- IPC共享内存注册前检查是否已注册同一块内存
- bool成员变量默认false且在多处以`!flag`守护资源分配时，检查所有初始化路径是否都有flag设置

关联commit: `3ec7410b`（IPC重复注册OOM）， `b2e74aee`（SetMemIncludeFlag遗漏调用），hcomm-dev `f64f6498`/`9277e023`（P2P传输中input/output在CCL buffer范围内仍独立IPC注册等4例）

---

#### 规则MEM-03: 可选初始化路径的成员指针使用前必须判空

严重等级：P1

缺陷描述：成员变量仅在特定配置条件下创建（如`symmetricMemory_`仅在跨超节点场景初始化），但使用该成员的函数没有判空保护，其他配置下调用时空指针解引用crash。

典型代码示例：

```cpp
// 缺陷代码 — src/framework/communicator/impl/hccl_communicator_host.cc
bool HcclCommunicator::IsSupportSymmetricMemory(HcclCMDType opType, OpParam &opParam)
{
    // symmetricMemory_ 在非跨超节点场景下为nullptr
    // 此处直接使用，无判空保护 -> crash
    HCCL_INFO("[%s] aicpuUnfold[%d], workflowMode[%d], ...", ...);
}

// 修复代码 — 入口判空
bool HcclCommunicator::IsSupportSymmetricMemory(HcclCMDType opType, OpParam &opParam)
{
    CHK_PRT_RET(symmetricMemory_ == nullptr,
        HCCL_DEBUG("symmetricMemory_ is a nullptr"), false);  // 提前返回
    HCCL_INFO("[%s] aicpuUnfold[%d], workflowMode[%d], ...", ...);
}
```

审查检查方法：
- 成员变量仅在特定条件下初始化时，所有使用点须有判空保护
- 新增对可选成员的使用时，须确认所有初始化路径是否都覆盖了该成员的创建
- 构造函数/Init中有条件分支跳过某些成员初始化时，标记这些成员为"可选"并在使用处强制判空

关联commit: `4e0bd97b`, hcomm-dev `a2b86b3d`/`384d78c0`（可选指针/新增参数校验缺失2例）

---

### 类别九：整数溢出/截断（2次，2.4%）

可审查性极高，编译器警告即可捕获。

#### 规则INT-01: 宽类型向窄类型转换必须检查溢出

严重等级：P0

缺陷描述：u64赋值给u32时高32位截断，导致内存大小计算远小于实际需要，后续buffer overflow。

典型代码示例：

```cpp
// 缺陷代码 — src/framework/hcom/hcom.cc
u32 count = hcomOpParam->count;  // hcomOpParam->count 是 u64，截断为 u32
// 当count > 4GB个元素时，scratch memory远小于实际需要

// 修复代码
u64 count = hcomOpParam->count;  // 保持u64
```

审查检查方法：
- 所有`static_cast`从宽类型向窄类型的转换必须检查溢出/截断风险
- 涉及内存大小计算的表达式必须使用u64
- 编译选项开启 `-Wconversion -Werror=conversion`

关联commit: `9c1f957b`, hcomm-dev `4ebb11d1`/`7a3d6fa6`（u32超时值通过u8返回类型截断等4例）

---

#### 规则INT-02: 加法后窄化须有饱和逻辑

严重等级：P1

缺陷描述：两个值相加后cast到窄类型，和超过窄类型上限时回绕到极小值。对于timeout场景，回绕导致超时时间从"很长"变成"几乎为零"。

典型代码示例：

```cpp
// 缺陷代码 — src/framework/communicator/impl/hccl_communicator_host.cc
u16 timeOut = static_cast<u16>(notifyWaitTime + AICPU_KERNEL_TIMEOUT_INC);
// 当和超过65535时，timeout回绕到极小值

// 修复代码 — 饱和算术
if (notifyWaitTime + AICPU_KERNEL_TIMEOUT_INC >= MAX_VALUE_U16) {
    timeOut = MAX_VALUE_U16;  // 饱和到上限
} else {
    timeOut = notifyWaitTime + AICPU_KERNEL_TIMEOUT_INC;
}
```

审查检查方法：
- 加法/乘法后窄化的表达式须有饱和逻辑或溢出检查
- 特别关注timeout、size、count类变量的窄化

关联commit: `6baf33c4`

---

### 类别十：C++语言特性（1次，1.2%）

#### 规则CPP-01: 跨SO边界的多态类虚析构不应使用 = default

严重等级：P1

缺陷描述： `= default`的虚析构函数由编译器在每个翻译单元隐式内联生成。当基类和派生类位于不同SO时，链接器可能找不到虚析构函数的符号。

典型代码示例：

```cpp
// 缺陷代码 — pkg_inc/hcomm/ccu/ccu_kernel.h (头文件, 被多个SO include)
~CcuKernel() override = default;  // 每个TU内联生成，跨SO时找不到符号

// 修复代码 — 头文件声明，.cc文件定义
// ccu_kernel.h
~CcuKernel() override;

// ccu_kernel.cc
CcuKernel::~CcuKernel() {}  // 显式定义在单一TU，确保符号导出
```

审查检查方法：
- 在pkg_inc或公共头文件中导出的多态类，虚析构函数不应使用`= default`
- 应在.cc文件中提供显式定义

关联commit: `bb490dc2`

---

### 类别十一：构建系统与Revert（2次，2.4%）

两次revert均代表缺陷逃逸到主干后紧急撤回，平均合入到revert间隔3.5小时。

#### 规则BUILD-01: 大变更合入须有充分的pre-merge验证

严重等级：P2

缺陷描述：24文件的构建系统重构合入1小时后被revert；硬件常量修改同时修改了UT期望值适配变更。两次PR描述均为空模板。

审查检查方法：
- PR描述为空模板的变更不应合入主干
- 同时修改源码和对应UT期望值时，审查UT修改的合理性——UT应独立验证而非适配变更
- 硬件常量修改须注明取值依据（spec文档引用）
- 构建系统大变更（>5个文件）应分步合入，每步独立验证
- 一个commit只做一件事：逻辑变更和风格变更必须分离

关联commit: `72cdf80e`（revert fb56d64b, CCU loop count常量修改无spec依据，合入6小时后撤回）， `753ba8c2`（revert 05b38411, 24文件构建重构，合入1小时后撤回）

---

### 跨类别系统性风险

以下模式跨越多个缺陷类别，是code review时应优先关注的系统性风险。

#### 规则SYS-01: 缺陷修复提交须额外审查是否引入新缺陷

严重等级：P1

缺陷描述： `4e82ec25`修复API参数方向问题时引入了use-after-free（将局部vector的data指针返回），被`6784944a`再次修复。修复提交的审查重心往往只在"原缺陷是否被修"，忽略了修复代码本身的正确性。

审查检查方法：
- 缺陷修复提交须有额外审查关注点：修复代码本身是否引入新问题
- 修复提交中新增的代码应按新代码标准审查，而非仅关注是否解决了原bug

关联commit: `4e82ec25` -> `6784944a`

---

#### 规则SYS-02: 修复一处须搜索重复代码段

严重等级：P2

缺陷描述：热点文件中存在大量代码重复（如ExecOp vs ExecOpAlltoAll、AllocTransportResource三路重复），修复一处时忘记修复重复代码中的同样问题。

审查检查方法：
- 修复一处缺陷后，全局搜索相似代码段是否有同样问题
- 对重复代码优先重构为共享函数，从根本上消除"修一漏一"

关联证据：hccl_communicator_host.cc（ExecOp重复），aicpu_communicator.cc（AllocTransportResource三路重复），communicator_impl.cc（5个初始化路径），hcomm-dev `36c8449d`/`fd23d1b6`（全局重命名后测试代码未同步5例），hccl-dev `2d8d548a`（API参数名blockDim→numBlocks后调用侧未同步）

---

#### 规则SYS-03: God Object文件修改须加强审查

严重等级：P3

缺陷描述：三个热点文件占全部缺陷修复的36.9%:

| 文件                      | 方法数/行数     | 缺陷修复次数 |
| ------------------------- | --------------- | ------------ |
| hccl_communicator_host.cc | 312方法/9152行  | 13次         |
| aicpu_communicator.cc     | 213函数/5550行  | 9次          |
| communicator_impl.cc      | ~100函数/3789行 | 9次          |

审查检查方法：
- 任何对这3个文件的修改应要求至少2个reviewer
- 长期策略：拆分God Object为职责单一的小类

---

#### 规则SYS-04: 返回值系统性忽略

严重等级：P1

缺陷描述：热点文件中多处忽略关键函数返回值：
- `hrtMalloc`返回值未检查(hccl_communicator_host.cc L1803)
- `malloc` 100MB未检查返回值(communicator_impl.cc L3115)
- `getenv`返回nullptr直接解引用(communicator_impl.cc L3025)

审查检查方法：
- 内存分配函数(malloc/hrtMalloc)返回值必须检查
- getenv()返回值必须判空
- 编译选项开启 `-Wunused-result`

---

#### 规则SYS-05: 编译器可捕获的缺陷须在CI中强制开启

严重等级：P2

缺陷描述：以下缺陷本可由编译器警告捕获但仍进入主干：

| 编译选项                  | 可捕获的缺陷 | 关联commit         |
| ------------------------- | ------------ | ------------------ |
| `-Wshadow`                | 变量遮蔽     | ef766683           |
| `-Wtautological-compare`  | 自比较       | e1880dc1           |
| `-Wformat`                | 格式串不匹配 | 35e32c7c, 6a6eac0f |
| `-Wconversion`            | 整数截断     | 9c1f957b, 6baf33c4 |
| `-Werror=enum-conversion` | 枚举隐式转换 | d953cbf3           |

审查检查方法：
- CI中强制开启上述警告并设为 `-Werror`
- 新增代码不允许suppress这些警告

---

#### 规则SYS-06: 热点文件中仍存在的已知高危风险

严重等级：P1

以下是热点文件中当前仍存在的高危问题（截至分析时点）：

| 风险          | 位置                          | 描述                         |
| ------------- | ----------------------------- | ---------------------------- |
| 悬空指针      | communicator_impl.cc L601     | 临时unique_ptr.get()传出     |
| malloc未检查  | communicator_impl.cc L3115    | malloc 100MB未检查返回值     |
| getenv空指针  | communicator_impl.cc L3025    | getenv返回nullptr直接解引用  |
| TLV解析死循环 | aicpu_communicator.cc L659    | length==0时while循环无法前进 |
| 锁策略不一致  | aicpu_communicator.cc resMap_ | 同一数据结构两种锁保护       |

#### 规则SYS-07: 公共API/ABI设计债务

严重等级：P1

缺陷描述：公共头文件中存在严重的ABI兼容性问题：
- hcom.h: extern "C"块内使用namespace、公共函数签名含C++默认参数、头文件中定义non-inline std:string/std:map全局变量
- hccl_comm_pub.h: 条件编译(CCL_KERNEL_AICPU/HCCD)改变类内存布局，跨模块传指针越界
- 多个API返回悬垂指针： `*algo = const_cast<char*>(str.c_str())`将局部变量内部指针返回（确定性UAF）
- 动态库导出函数用std:string&参数，跨ABI不兼容

审查检查方法：
- 公共头文件(pkg_inc/)的任何修改必须通过ABI兼容性checklist审查
- 公共头文件仅使用标准C/C++类型，禁止内部typedef
- 动态库导出函数(extern "C")的参数和返回值必须为POD类型
- stub符号与实际函数签名是否匹配

关联commit: hcomm-dev `30e25e50`/`7fccc101`/`e6d12932`（ABI兼容性/符号导出5例）

---

### 类别十二：构建/编译/链接缺陷

通信框架构建系统复杂度随多编译目标(host/device/kernel/daemon)和多构建模式(open/closed/HCCD/CCL_KERNEL)增长而爆炸，是hcomm-dev中频次最高的缺陷类别（24条，14.8%）。

#### 规则BUILD-02: CMakeLists源文件遗漏/重复

严重等级：P2

缺陷描述：新增.cc文件后忘记在CMakeLists.txt中添加，导致功能不被编译链接。另有源文件同时被添加到两个CMakeLists.txt中导致符号重复定义。

典型代码示例：

```cmake
# 缺陷 — alltoallv continuous pipeline的两个.cc文件遗漏
set(SOURCES
    ...
    coll_all_to_all_v_executor.cc
    # 遗漏: alltoallv_continuous_pipeline.cc
    # 遗漏: coll_all_to_all_v_continuous_pipeline_executor.cc
    ...
)

# 修复 — 同步添加新增源文件
set(SOURCES
    ...
    coll_all_to_all_v_executor.cc
    alltoallv_continuous_pipeline.cc
    coll_all_to_all_v_continuous_pipeline_executor.cc
    ...
)
```

审查检查方法：
- 新增.cc文件的PR是否同步修改了CMakeLists.txt?
- 同一.cc文件是否出现在多个CMakeLists.txt中（符号重复）？
- 新增add_library时，逐个检查源文件中的#include和函数调用是否有对应.cc文件在target中

关联commit: hcomm-dev `8c424d41`/`20125a56`, hccl-dev `beb0ed54`（scatter_aicpu_kernel库遗漏5个依赖源文件）

---

#### 规则BUILD-03: 条件编译/编译目标不兼容

严重等级：P2

缺陷描述：在特定编译目标下引入不可用的外部依赖，或条件编译块覆盖不完整。常见模式：函数调用了ACL API但CCL_KERNEL_AICPU/HCCD目标下ACL不可用。

典型代码示例：

```cpp
// 缺陷 — ParseCannVersion()调用ACL API，HCCD目标下不可用
HcclResult ParseCannVersion() {
    const char *version = aclGetVersion();  // CCL_KERNEL_AICPU/HCCD下链接失败
    ...
}

// 修复 — 条件编译包裹
HcclResult ParseCannVersion() {
#if !defined(CCL_KERNEL_AICPU) && !defined(HCCD)
    const char *version = aclGetVersion();
    ...
#else
    return HCCL_E_NOT_SUPPORT;
#endif
}
```

审查检查方法：
- 新增外部API依赖时，是否检查了所有编译目标(host/device/kernel/daemon)的可用性？
- 条件编译块（BUILD_OPEN_PROJECT/KERNEL_MODE等）的open和closed路径是否对称？
- 不同编译环境(Host/AICPU)下API可用性不同，需要条件编译隔离

关联commit: hcomm-dev `89fd99e0`/`947460b2`

---

#### 规则BUILD-04: ABI兼容性/符号导出

严重等级：P1

缺陷描述：公共头文件中使用内部类型、导出C++符号到C linkage、动态库导出函数使用std:string参数等。

典型代码示例：

```cpp
// 缺陷 — 公共头文件hcomm_primitives.h中uint32_t改为内部typedef u32
// 外部用户编译时找不到u32定义，ABI破坏
typedef unsigned int u32;  // 内部头文件
void HcommFunc(u32 param);  // pkg_inc/中使用内部typedef

// 修复 — 公共头文件仅使用标准C/C++类型
void HcommFunc(uint32_t param);
```

审查检查方法：
- 公共头文件(pkg_inc/)中是否仅使用标准C/C++类型，禁止内部typedef?
- 动态库导出函数(extern "C")的参数和返回值是否为POD类型？
- stub符号与实际函数签名是否匹配？

关联commit: hcomm-dev `30e25e50`/`7fccc101`/`e6d12932`

---

#### 规则BUILD-05: CMake版本兼容性与参数格式

严重等级：P2

缺陷描述：CMake脚本中使用了高版本特性（如CMAKE_CURRENT_FUNCTION_LIST_DIR需要3.17+），或`-D`参数后加空格导致解析失败。

典型代码示例：

```cmake
# 缺陷 — CMAKE_CURRENT_FUNCTION_LIST_DIR需要CMake 3.17+
-P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/_pack_stage.cmake"

# 修复 — 在文件作用域保存路径（兼容低版本CMake）
set(_FUNC_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
-P "${_FUNC_CMAKE_DIR}/_pack_stage.cmake"

# 缺陷 — -D后有空格导致某些版本解析失败
set(manifest_arg "-D _MANIFEST_FILE=${staging_dir}/${ARG_MANIFEST}")

# 修复 — -D后不加空格
set(manifest_arg -D_MANIFEST_FILE=${staging_dir}/${ARG_MANIFEST})
```

审查检查方法：
- 对照项目cmake_minimum_required确认使用的CMake特性版本要求
- `-D`后紧跟变量名，不加空格
- 函数内引用路径变量时，确认变量在函数作用域内可用

关联commit: hccl-dev `1f13573a`

---

### 类别十三：初始化/赋值缺陷

通信框架对象生命周期早期阶段的脆弱性。核心模式是多分支初始化路径中的遗漏和成员变量初始化缺失。hcomm-dev中17条，占10.5%。

#### 规则INIT-01: 多分支初始化路径不对称

严重等级：P1

缺陷描述：同一函数存在V1/V2、图模式/单算子模式等多条初始化分支，其中某条遗漏了必要的初始化步骤。

典型代码示例：

```cpp
// 缺陷 — V2路径创建communicator后遗漏HcomSetGroupTopoInfo
// V2路径
auto lambda = [&]() {
    CHK_RET(CreateCommunicator(...));
    // 遗漏: CHK_RET(HcomSetGroupTopoInfo(...));
    return HCCL_SUCCESS;
};
HCCLV2_FUNC_RUN(lambda);

// 非V2路径(已正确调用)
CHK_RET(CreateCommunicator(...));
CHK_RET(HcomSetGroupTopoInfo(...));

// 修复 — V2路径补充与非V2对称的调用
auto lambda = [&]() {
    CHK_RET(CreateCommunicator(...));
    CHK_RET(HcomSetGroupTopoInfo(...));  // 补充
    return HCCL_SUCCESS;
};
```

审查检查方法：
- 同一函数的多条初始化分支是否执行了相同的必要初始化步骤（对称性检查）？
- V2适配路径是否完整覆盖了V1路径的所有初始化步骤？
- 对比两个分支的函数调用序列，寻找不对称项

关联commit: hcomm-dev `65f0c2ba`/`b54e3852`

---

#### 规则INIT-02: 成员变量未初始化/默认值错误

严重等级：P0

缺陷描述：C++内置类型成员未显式初始化，运行时包含垃圾值导致非确定性行为。

典型代码示例：

```cpp
// 缺陷 — 多个成员缺少类内初始化器
class HcclCommunicator {
    bool isGroupMode_;       // 垃圾值 -> group模式判断异常
    u32 iSend;               // 垃圾值
    u32 iRecv;
    u32 nSend;
    u32 nRecv;
    u32 bufferSliceNum;
};

// 修复 — 显式初始化
class HcclCommunicator {
    bool isGroupMode_ = false;
    u32 iSend = 0;
    u32 iRecv = 0;
    u32 nSend = 0;
    u32 nRecv = 0;
    u32 bufferSliceNum = 0;
};
```

审查检查方法：
- C++类的内置类型成员（bool/int/u32/指针）是否有显式初始化（in-class initializer或构造函数初始化列表）？
- Init函数是否具备幂等性？重复调用是否会覆盖外部配置？

关联commit: hcomm-dev `2bad014b`/`103755111cff`/`6596fc1f`

---

#### 规则INIT-03: offload/lambda变量作用域泄漏

严重等级：P1

缺陷描述：HCCLV2_FUNC_RUN宏的lambda外部提前做了cast和使用，offload模式下控制流未正确进入V2分支，变量在错误作用域被访问。

典型代码示例：

```cpp
// 缺陷 — lambda外部使用了应在V2分支内的变量
auto *comm = static_cast<hcclComm*>(handle);  // lambda外部cast
comm->DoSomething();                            // 非V2模式下不应执行
auto lambda = [&]() {
    HCCLV2_FUNC_RUN(comm->V2Method());
    return HCCL_SUCCESS;
};

// 修复 — cast和使用都移入lambda内部
auto lambda = [&]() {
    auto *comm = static_cast<hcclComm*>(handle);
    CHK_RET(comm->V2Method());
    return HCCL_SUCCESS;
};
HCCLV2_FUNC_RUN(lambda);
```

审查检查方法：
- HCCLV2_FUNC_RUN宏前后的变量使用是否明确在正确的作用域内？
- lambda捕获列表中的变量是否可能在外部被提前消费？

关联commit: hcomm-dev `b54e3852`

---

### 类别十四：数据类型/枚举缺陷

通信框架中类型安全问题突出，包括枚举值误用为算术因子、枚举变体混淆、map:at()异常、参数对象混淆等。

#### 规则TYPE-01: 枚举值语义误用

严重等级：P0

缺陷描述：将枚举值当作字节大小、数组索引等用于算术运算。枚举值代表类型标识而非数值量。

典型代码示例：

```cpp
// 缺陷 — dataType枚举值被直接用作除数
// reduceInfo.dataType是枚举值(如HCCL_DATA_TYPE_FP32=3)，不是字节数(4)
u64 count = src->len / reduceInfo.dataType;  // 用枚举值做除法，结果错误

// 修复 — 通过SIZE_TABLE查表获取字节数
u64 count = src->len / SIZE_TABLE[reduceInfo.dataType];  // 查表获取实际字节数
```

审查检查方法：
- dataType枚举值是否被直接用作算术运算的操作数？应通过SIZE_TABLE查表获取字节数
- 枚举常量比较时，常量名是否与期望语义一致？

关联commit: hcomm-dev `5422c95d`/`20824546`

---

#### 规则TYPE-02: 通信引擎枚举变体混淆

严重等级：P0

缺陷描述：CANN平台存在多个语义相近的通信引擎枚举值（COMM_ENGINE_AICPU / COMM_ENGINE_AICPU_TS / COMM_ENGINE_CPU等），使用了错误的变体会导致执行路径、资源分配和launch模式全部错误。

典型代码示例：

```cpp
// 缺陷 — 使用AICPU而非AICPU_TS（6处全部用错）
param.engine = CommEngine::COMM_ENGINE_AICPU;

// 修复 — 使用正确的AICPU_TS变体
param.engine = CommEngine::COMM_ENGINE_AICPU_TS;
```

审查检查方法：
- 搜索所有CommEngine枚举使用点，确认变体选择与设计文档一致
- 同一文件/函数内的引擎类型应前后一致
- 新增引擎类型使用时，要求开发者说明选择理由

关联commit: hccl-dev `7347fee3`

---

#### 规则TYPE-03: 平台字符串字面量拼写错误

严重等级：P1

缺陷描述：平台对launch mode等字符串做精确匹配，拼写不一致（如"AICPU" vs "AI_CPU"）会导致功能静默失效。

典型代码示例：

```cpp
// 缺陷 — 平台期望的是"AI_CPU"（带下划线）
launchMode = "AICPU";

// 修复
launchMode = "AI_CPU";
```

审查检查方法：
- 所有平台相关字符串字面量应有对应常量定义，禁止裸字符串
- 审查时将字符串字面量与平台文档/头文件中的定义逐字符比对
- 搜索相似但不同的字符串（如"AICPU"/"AI_CPU"/"AiCpu"）是否混用

关联commit: hccl-dev `13b6032d`

---

#### 规则TYPE-04: 参数对象混淆/引用错误

严重等级：P0

缺陷描述：传入的参数使用了错误的对象，或参数被声明但未使用。多个同类型参数的函数容易传参错误。

典型代码示例：

```cpp
// 缺陷 — 应从dstThreadPtr获取notify，实际从threadPtr(源线程)获取
void HcommThreadNotifyRecordOnThread(Thread *threadPtr, Thread *dstThreadPtr) {
    auto notify = threadPtr->GetNotify();    // 错误: 从源线程获取
    // 参数dstThreadPtr传入却未使用
    ...
}

// 修复 — 使用正确的参数
void HcommThreadNotifyRecordOnThread(Thread *threadPtr, Thread *dstThreadPtr) {
    auto notify = dstThreadPtr->GetNotify();  // 从目标线程获取
    ...
}
```

审查检查方法：
- 函数参数被声明但未使用时是否告警(unused parameter)?
- 多个同类型参数的函数，调用处传参顺序是否正确？

关联commit: hcomm-dev `d89396c9`

---

#### 规则TYPE-05: map:at()对未收录key抛异常

严重等级：P0

缺陷描述：std:map:at()在key不存在时抛std:out_of_range异常，C++异常未被捕获导致coredump。新增枚举值后未更新对应map。

典型代码示例：

```cpp
// 缺陷 — errorMap未收录新增错误码，at()抛异常
std::string msg = errorMap.at(code);  // HCCL_E_SUSPENDING等未收录 -> 异常 -> coredump

// 修复 — 改用find()+迭代器检查
auto it = errorMap.find(code);
if (it != errorMap.end()) {
    std::string msg = it->second;
} else {
    std::string msg = "unknown error code: " + std::to_string(code);
}
```

审查检查方法：
- 禁止在可能接收不可控输入的场景使用std:map:at()，应使用find()+迭代器检查
- 新增枚举值时，是否全量搜索所有使用该枚举的map/switch/数组并同步更新？

关联commit: hcomm-dev `a84b98a0`

---

### 类别十五：代码质量/命名/拼写

命名不一致和拼写错误在通信框架的算法名称匹配场景中可能造成严重功能故障。hcomm-dev中15条，占9.3%。

#### 规则QUAL-01: 算法名称/变量名拼写错误

严重等级：P1

缺陷描述：Copy-Paste导致的算法名称错误、变量名多/少字符。通信框架通过字符串匹配选择executor，拼写错误直接导致选错算法。

典型代码示例：

```cpp
// 缺陷 — ReduceScatterV的算法名写成ReduceScatter(缺少"V"后缀)
algName = "AlignedReduceScatterDoubleRingFor91093Executor";

// 修复 — 补全"V"后缀
algName = "AlignedReduceScatterVDoubleRingFor91093Executor";
```

审查检查方法：
- ReduceScatterV相关文件中的算法名称是否包含"V"后缀？
- Copy-Paste新函数后，函数名/算法名/变量名中的标识符是否全部替换？
- 算法名称字符串是否有拼写检查机制（建议引入cspell）

关联commit: hcomm-dev `5bfaa1e5`/`1820333425906`, hccl-dev `8222fcf8`/`11b7211a`（复制粘贴后变量名未替换）

---

#### 规则QUAL-02: 全代码库拼写错误渗入API

严重等级：P3

缺陷描述：常见拼写错误（invaild/vaild, recevie, at lest等）渗入枚举值、函数名、日志，部分影响API兼容性。

典型代码示例：

```cpp
// 缺陷 — 枚举值拼写错误
enum OpType {
    OP_SEND = 0,
    OP_RECV,
    OP_INVAILD    // invalid拼成invaild
};

// 修复
enum OpType {
    OP_SEND = 0,
    OP_RECV,
    OP_INVALID
};
```

审查检查方法：
- CI中引入cspell或codespell工具扫描代码注释和字符串字面量
- 枚举值/函数名中的单词是否拼写正确？

关联commit: hcomm-dev `345be6c4`/`02d62eef`

---

#### 规则QUAL-03: 结构体/类字段变更未同步到所有使用方

严重等级：P1

缺陷描述：修改结构体定义（增删字段）后，部分使用方（特别是测试代码、mock、stub）未同步更新，导致编译失败。

典型代码示例：

```cpp
// 缺陷 — HcclCommConfig中commEngine/threadNum/notifyNumPerThread已被移除
// 但测试代码仍在设置这些字段
commConfig.commEngine = HCCL_COMM_ENGINE_CONFIG_NOT_SET;
commConfig.threadNum  = HCCL_COMM_THREADNUM_CONFIG_NOT_SET;
```

审查检查方法：
- 修改struct/class定义时，全局grep所有使用点，范围必须包含test/目录
- 提交变更前在本地完成全量编译
- 注意同一struct可能在.cc和。h中都有使用

关联commit: hccl-dev `c11a5289`/`8222fcf8`

---

#### 规则QUAL-04: 公共API函数名与头文件声明不匹配

严重等级：P1

缺陷描述：公共API的.cc实现使用了错误的函数名（与内部函数同名），与。h声明不一致，导致链接失败或无限递归。

典型代码示例：

```cpp
// 缺陷 — 头文件声明的是HcclBatchSendRecv，但实现写成了HcclBatchSendRecvInner
HcclResult HcclBatchSendRecvInner(HcclSendRecvItem* sendRecvInfo, ...) {
    return HcclBatchSendRecvInner(sendRecvInfo, itemNum, comm, stream);
    // 这还会导致无限递归（自己调自己）
}

// 修复
HcclResult HcclBatchSendRecv(HcclSendRecvItem* sendRecvInfo, ...) {
    return HcclBatchSendRecvInner(sendRecvInfo, itemNum, comm, stream);
}
```

审查检查方法：
- 公共API函数的.cc定义必须与。h声明的函数签名精确匹配
- 包装函数的名字不应与被包装函数相同（否则变成递归调用）

关联commit: hccl-dev `cae52923`

---

### 类别十六：空指针/入参校验缺陷

集中在framework层的公共API入口和内部组件的指针使用。hcomm-dev中8条，占4.9%。

#### 规则PTR-01: 空指针检查顺序错误

严重等级：P0

缺陷描述：先解引用再检查null，或日志中%s传可能为nullptr的指针(UB)。

典型代码示例：

```cpp
// 缺陷 — 先解引用再检查null
auto val = ptr->GetValue();
if (ptr == nullptr) {
    return HCCL_E_PTR;
}

// 修复 — 检查前置于解引用
if (ptr == nullptr) {
    return HCCL_E_PTR;
}
auto val = ptr->GetValue();
```

```cpp
// 缺陷 — nullptr传给%s是UB
void DestroyDispatcherCtx(const char *commId) {
    HCCL_INFO("[DestroyCtx] commId[%s]", commId);  // commId可能为nullptr
    if (commId == nullptr) { return; }

// 修复 — 空指针检查前置
void DestroyDispatcherCtx(const char *commId) {
    if (commId == nullptr) { return; }
    HCCL_INFO("[DestroyCtx] commId[%s]", commId);
```

审查检查方法：
- 空指针检查是否在解引用之前（不是之后）？
- %s格式化的参数是否保证非空？

关联commit: hcomm-dev `dd744786`/`384d78c0`

---

#### 规则PTR-02: API入口参数校验缺失

严重等级：P1

缺陷描述：公共API中reinterpret_cast后未检查null，用户传入无效句柄则crash。

典型代码示例：

```cpp
// 缺陷 — HcclGetHcclBuffer检查了buffer但漏检comm
HcclResult HcclGetHcclBuffer(HcclComm comm, void **buffer) {
    CHK_PRT_RET(buffer == nullptr,
        HCCL_ERROR("buffer is nullptr"), HCCL_E_PTR);
    hccl::hcclComm *hcclComm = static_cast<hccl::hcclComm *>(comm);
    // comm为nullptr时hcclComm也是nullptr -> 下一行crash
    std::string identifier = hcclComm->GetIdentifier();

// 修复 — 补充comm指针校验
HcclResult HcclGetHcclBuffer(HcclComm comm, void **buffer) {
    CHK_PRT_RET(buffer == nullptr,
        HCCL_ERROR("buffer is nullptr"), HCCL_E_PTR);
    CHK_PRT_RET(comm == nullptr,
        HCCL_ERROR("comm is nullptr"), HCCL_E_PTR);
```

审查检查方法：
- reinterpret_cast/static_cast后是否检查null? 特别是来自用户输入的句柄
- 公共API入口是否对所有指针参数做了null/有效性校验？

关联commit: hcomm-dev `58d588bf`

---

### 类别十七：接口/API设计与适配缺陷

通信框架经历V1->V2迁移、ACL->RT接口替换、open/closed双模式等架构演进，接口适配层面的设计缺陷频繁暴露。hcomm-dev中8条，占4.9%。

#### 规则API-01: 抢跑依赖未发布API

严重等级：P2

缺陷描述：在依赖的SDK正式发布新接口前就在消费端使用，链接失败。12天内对同一组文件进行5次方向性变更（乒乓Revert）。

典型代码示例：

```cpp
// 缺陷 — rtModelGetId在runtime SDK中尚未发布
uint64_t modelId;
rtModelGetId(rtModel, &modelId);  // 链接失败: undefined symbol

// 修复 — dlopen双路径fallback
void *handle = dlopen("libruntime.so", RTLD_LAZY);
auto func = dlsym(handle, "rtModelGetId");
if (func) {
    func(rtModel, &modelId);
} else {
    modelId = reinterpret_cast<uint64_t>(rtModel);  // fallback
}
```

审查检查方法：
- 跨团队接口迁移是否前置确认目标API在所有受支持版本已发布？
- 是否有dlopen/weak symbol的兼容性降级方案？

关联commit: hcomm-dev `30e25e50`/`1d8e2c14`

---

#### 规则API-02: 返回值语义差异未适配

严重等级：P1

缺陷描述：底层接口替换后返回值语义变化(bitmask vs enum)，调用方解析逻辑未同步适配。

典型代码示例：

```cpp
// 缺陷 — ACL返回bitmask，RT返回enum，切换后解析逻辑未改
u32 linkType = rtGetPairPhyDevicesInfo(devA, devB);
if (linkType & LINK_TYPE_HCCS) {  // 按bitmask解析enum -> 逻辑错误
    ...
}

// 修复 — 适配enum语义
u32 linkType = rtGetPairPhyDevicesInfo(devA, devB);
if (linkType == LINK_TYPE_HCCS) {  // 按enum单值比较
    ...
}
```

审查检查方法：
- 返回值类型差异(bitmask vs enum, signed vs unsigned)在迁移时调用方解析逻辑是否同步适配？
- API返回裸指针时，所有权（谁分配谁释放）是否在注释中明确文档化？

关联commit: hcomm-dev `1d8e2c14`

---

#### 规则API-03: 封装层能力不完整

严重等级：P2

缺陷描述：适配层未覆盖底层完整能力，或V2 weak symbol函数声明与实际签名不匹配。

审查检查方法：
- 封装层新增接口后，是否覆盖了底层完整能力？
- V2 weak symbol函数的签名是否与实际函数完全一致？

关联commit: hcomm-dev `47020bd5`/`26f5813f`

---

### 类别十八：硬件/平台适配缺陷

通信库需要适配多种芯片（910B/910_93/910_95/A5等）和多种连接协议(PCIE/RoCE/HCCS/URMA)，硬件协议约束和设备差异性是低可审查性缺陷的主要来源。hcomm-dev中6条，占3.7%。

#### 规则HW-01: 硬件协议乘数因子遗漏

严重等级：P0

缺陷描述：URMA约束每个SQE含4个WQEBB，多处代码计算SQ buffer大小和VA映射长度时遗漏WQEBB_NUM_PER_SQE(=4)乘数因子，device侧VA空间只有实际需要的1/4，内存越界。

典型代码示例：

```cpp
// 缺陷 — sqDepth未乘WQEBB_NUM_PER_SQE
UbConnLite::UbConnLite(..., u32 sqDepth, ...) {
    sqDepth_ = sqDepth;  // 遗漏乘数因子
}
u64 sqBufferSize = sqDepth_ * WQE_BB_SIZE;  // 缺 * WQEBB_NUM_PER_SQE
// VA空间只有实际需要的1/4 -> 内存越界

// 修复 — 统一乘WQEBB_NUM_PER_SQE
UbConnLite::UbConnLite(..., u32 sqDepth, ...) {
    sqDepth_ = sqDepth * WQEBB_NUM_PER_SQE;  // 每个SQE含4个WQEBB
}
```

审查检查方法：
- 涉及硬件协议约束的乘数因子是否在协议层统一封装为宏/函数？避免多处手动乘算
- SQ深度等协议参数的"单位"（SQE数/WQEBB数/字节）是否在类型或命名中体现？

关联commit: hcomm-dev `df96667c`

---

#### 规则HW-02: 设备类型特殊路径未处理

严重等级：P1

缺陷描述：新增设备类型的初始化/资源管理路径与已有设备不同。

典型代码示例：

```cpp
// 缺陷 — 910_95设备notify初始化不需要SetIpc()
CHK_RET(notify->SetIpc());  // 910_95上SetIpc()失败 -> notify申请失败

// 修复 — 按设备类型分支处理
if (deviceType_ != DEV_TYPE_910_95) {
    CHK_RET(notify->SetIpc());
}
```

审查检查方法：
- 新增设备类型时，是否审查了所有资源初始化路径的分支处理？
- 是否有设备兼容性矩阵文档？

关联commit: hcomm-dev `b9f46705`/`a0e420e9`

---

#### 规则HW-03: 硬件寄存器/地址映射宏定义冲突

严重等级：P0

缺陷描述：宏常量值重复导致寄存器访问错误。读DB_STATUS实际读到PI_TYPE寄存器。

典型代码示例：

```cpp
// 缺陷 — 宏定义值与另一个宏重复
#define URMA_JFS_PI_TYPE     0x0011
#define URMA_JFS_DB_STATUS   0x0011  // 错误: 与PI_TYPE重复

// 修复
#define URMA_JFS_DB_STATUS   0x000f  // 正确值
```

审查检查方法：
- 同一组寄存器偏移宏是否通过static_assert确保无重复值？
- 新增寄存器映射宏时是否检查了已有定义？

关联commit: hcomm-dev `39ad6c07`

---

### 类别十九：状态/超时管理缺陷

通信框架中大量使用重试超时、环境变量配置等机制。超时值不一致和状态不刷新是常见根因。hcomm-dev中10条，占6.2%。

#### 规则STATE-01: 超时值硬编码/配置不一致

严重等级：P1

缺陷描述：多层超时值未统一联动，内外层超时语义不一致。

典型代码示例：

```cpp
// 缺陷 — OpRetry超时硬编码205秒，用户配置更大值时先超时误判
constexpr u32 OP_RETRY_SEND_RECV_TIMEOUT = 205;
future.wait_for(std::chrono::seconds(OP_RETRY_SEND_RECV_TIMEOUT));
// 用户HCCL_LINK_TIMEOUT=300时，OpRetry在205秒先超时

// 修复 — 取max(用户配置, 默认值) + 裕量
u32 timeout = std::max(GetExternalInputHcclLinkTimeOut(),
                       OP_RETRY_SEND_RECV_TIMEOUT) + OP_RETRY_WAIT_AICPU_TIMEOUT;
```

```cpp
// 缺陷 — execTimeOut=0语义为"不超时"，代码按字面值处理
u64 timeoutUs = execTimeOut * 1000000;  // 0 * 1000000 = 0微秒 -> 立即超时

// 修复 — 显式处理零值特殊语义
u64 timeoutUs = (execTimeOut == 0) ? UINT64_MAX : execTimeOut * 1000000;
```

审查检查方法：
- 超时参数是否从外层逐层传递到底层阻塞调用？
- 超时值0是否有"不限制"的特殊语义？转换函数中是否显式处理了零值？
- 不同组件的超时值是否与用户可配置值联动（取max而非硬编码）？

关联commit: hcomm-dev `b8de68b9`/`1479e7ba`/`f058f2d9`

---

#### 规则STATE-02: 环境变量/运行时状态一致性

严重等级：P2

缺陷描述：每次调用都getenv而非缓存初始化时读取，运行时修改环境变量导致行为不一致。

典型代码示例：

```cpp
// 缺陷 — 每次调用都读环境变量，运行时修改导致行为不一致
bool IsIndependentOp() {
    const char *env = getenv("HCCL_INDEPENDENT_OP");
    return (env != nullptr && strcmp(env, "1") == 0);
}

// 修复 — 初始化时一次性读取并缓存
static bool g_isIndependentOp = false;
void InitConfig() {
    const char *env = getenv("HCCL_INDEPENDENT_OP");
    g_isIndependentOp = (env != nullptr && strcmp(env, "1") == 0);
}
bool IsIndependentOp() { return g_isIndependentOp; }
```

审查检查方法：
- 环境变量是否在初始化时一次性读取缓存？运行时重复读取是否有意为之？

关联commit: hcomm-dev `578c16b4`

---

### 类别二十：条件判断逻辑缺陷

#### 规则COND-01: 条件方向错误/条件编译覆盖运行时分支

严重等级：P0

缺陷描述：条件判断写反，或#ifdef块无条件覆写了上方的运行时分支判断。

典型代码示例：

```c
// 缺陷 — #ifdef CONFIG_CONTEXT块无条件覆写运行时分支
phyId = (attr->protocol == PROTOCOL_RDMA) ?
        attr->dev.rdma.phyId : attr->ub.phy_id;

#ifdef CONFIG_CONTEXT
    phyId = attr->ub.phy_id;  // 无条件覆写，RDMA时使用了错误的phyId
#endif

// 修复 — 条件编译块内保持与上方一致的分支语义
#ifdef CONFIG_CONTEXT
    phyId = (attr->protocol == PROTOCOL_RDMA) ?
            attr->dev.rdma.phyId : attr->ub.phy_id;
#endif
```

审查检查方法：
- 条件表达式的方向(>= vs <=, && vs ||)是否与注释/上下文语义一致？
- #ifdef块是否无条件覆写了上方的运行时判断？
- 基类和子类存在相同语义条件判断时，是否通过virtual方法统一？

关联commit: hcomm-dev `54e5e41b`/`32ff3df8`/`66f33b83`

---

### 类别二十一：网络安全

通信库中涉及敏感信息保护的审查规则，违反即触碰华为产品网络安全红线。

#### 规则SEC-01: token认证信息禁止打印到日志

严重等级：P0

缺陷描述：token认证类信息(tokenId、tokenValue)属于华为公司产品网络安全红线中规定的敏感信息，禁止通过日志、调试输出等任何途径打印。在UB芯片上需重点关注tokenId和tokenValue两个字段。

典型代码示例：

```cpp
// 缺陷代码
HCCL_INFO("token info: id=%u, value=%s", tokenId, tokenValue);  // 红线违规：敏感信息入日志

// 修复代码
HCCL_INFO("token info: id=***, value=***");  // 脱敏处理
```

审查检查方法：
- grep所有日志输出点(HCCL_INFO/DEBUG/ERROR/WARNING/RUN_INFO)，检查是否包含tokenId或tokenValue
- 新增日志语句时逐条确认无敏感字段泄露
- UB芯片相关代码重点检查token相关字段

---

#### 规则SEC-02: RDMA rkey/lkey不属于敏感信息

严重等级：信息（误报抑制）

缺陷描述：RDMA协议中使用的rkey和lkey信息不用于鉴权功能，仅作为索引辅助硬件找到注册的内存，因此不属于敏感信息。审查时不应将rkey/lkey的日志打印标记为网络安全违规。

审查检查方法：
- 遇到rkey/lkey打印时不报"敏感信息泄露"
- 参见references/false-positives.md中的对应条目

---

### 附录：规则速查表

| 规则ID   | 规则名称                                 | 等级 | 类别            |
| -------- | ---------------------------------------- | ---- | --------------- |
| ALG-01   | 变量名遮蔽导致成员未赋值                 | P0   | 算法正确性      |
| ALG-02   | 变量自比较                               | P0   | 算法正确性      |
| ALG-03   | 结构体/参数赋值遗漏                      | P1   | 算法正确性      |
| ALG-04   | 同族executor一致性缺陷                   | P1   | 算法正确性      |
| ALG-05   | 边界条件缺失                             | P1   | 算法正确性      |
| ALG-06   | Get函数不应有Set副作用                   | P1   | 算法正确性      |
| ALG-07   | 多阶段/多版本流程变量混淆                | P1   | 算法正确性      |
| ALG-08   | 非均匀集合通信per-rank偏移               | P1   | 算法正确性      |
| CFG-01   | 跨版本协议OpCode兼容                     | P0   | 配置与兼容性    |
| CFG-02   | 设备类型分支全面覆盖                     | P1   | 配置与兼容性    |
| CFG-03   | 常量不应承担多重语义                     | P2   | 配置与兼容性    |
| CFG-04   | v1/v2接口分发完整性                      | P1   | 配置与兼容性    |
| CON-01   | 共享数据写后更新标志须有内存屏障         | P0   | 并发问题        |
| CON-02   | Record/Wait同步顺序                      | P0   | 并发问题        |
| CON-03   | publish ordering                         | P0   | 并发问题        |
| CON-04   | 并发delete须加锁                         | P0   | 并发问题        |
| CON-05   | atomic变量须用.load()读取                | P1   | 并发问题        |
| CON-06   | 单例Init区分once/every-time              | P1   | 并发问题        |
| CON-07   | thread_local变量须在线程入口点设置       | P1   | 并发问题        |
| CON-08   | 全局/静态变量无锁访问                    | P0   | 并发问题        |
| CON-09   | TOCTOU竞态                               | P1   | 并发问题        |
| LOG-01   | 格式化字符串类型匹配                     | P2   | 日志与调试      |
| LOG-02   | 日志tag须匹配类名/函数名                 | P3   | 日志与调试      |
| LOG-03   | 快速路径功能对等                         | P2   | 日志与调试      |
| LOG-04   | 日志洪泛/变量引用错误                    | P3   | 日志与调试      |
| RES-01   | early return跳过初始化/清理              | P1   | 资源生命周期    |
| RES-02   | map/cache key唯一标识资源                | P1   | 资源生命周期    |
| RES-03   | context切换附近内存操作审查              | P1   | 资源生命周期    |
| RES-04   | 析构路径完整性                           | P1   | 资源生命周期    |
| RES-05   | 资源释放顺序错误/释放后未置空            | P0   | 资源生命周期    |
| ERR-01   | 异常上报须受前置条件保护                 | P2   | 错误处理        |
| ERR-02   | 不允许裸try-catch吞异常                  | P2   | 错误处理        |
| ERR-03   | 对外接口不应抛C++异常                    | P0   | 错误处理        |
| ERR-04   | CHK_RET调用顺序                          | P2   | 错误处理        |
| ERR-05   | 预期内返回值日志级别不当                 | P2   | 错误处理        |
| CACHE-01 | cache key覆盖所有状态维度                | P1   | 缓存一致性      |
| CACHE-02 | 缓存复用前逐字段确认刷新                 | P1   | 缓存一致性      |
| CACHE-03 | 函数副作用污染成员变量                   | P1   | 缓存一致性      |
| CACHE-04 | 分支条件与key同数据源                    | P1   | 缓存一致性      |
| MEM-01   | 禁止暴露局部容器内部指针                 | P0   | 内存管理        |
| MEM-02   | IPC共享内存重复注册检查                  | P1   | 内存管理        |
| MEM-03   | 可选初始化路径成员指针须判空             | P1   | 内存管理        |
| INT-01   | 宽类型向窄类型转换溢出检查               | P0   | 整数溢出        |
| INT-02   | 加法后窄化须饱和逻辑                     | P1   | 整数溢出        |
| CPP-01   | 跨SO虚析构不应= default                  | P1   | C++语言特性     |
| BUILD-01 | 大变更pre-merge验证                      | P2   | 构建系统        |
| SYS-01   | 缺陷修复须审查是否引入新缺陷             | P1   | 系统性风险      |
| SYS-02   | 修复一处须搜索重复代码段                 | P2   | 系统性风险      |
| SYS-03   | God Object文件加强审查                   | P3   | 系统性风险      |
| SYS-04   | 返回值系统性忽略                         | P1   | 系统性风险      |
| SYS-05   | CI编译器警告强制开启                     | P2   | 系统性风险      |
| SYS-06   | 热点文件已知高危风险                     | P1   | 系统性风险      |
| SYS-07   | 公共API/ABI设计债务                      | P1   | 系统性风险      |
| BUILD-02 | CMakeLists源文件遗漏/重复                | P2   | 构建/编译/链接  |
| BUILD-03 | 条件编译/编译目标不兼容                  | P2   | 构建/编译/链接  |
| BUILD-04 | ABI兼容性/符号导出                       | P1   | 构建/编译/链接  |
| BUILD-05 | CMake版本兼容性与参数格式                | P2   | 构建/编译/链接  |
| INIT-01  | 多分支初始化路径不对称                   | P1   | 初始化/赋值     |
| INIT-02  | 成员变量未初始化/默认值错误              | P0   | 初始化/赋值     |
| INIT-03  | offload/lambda变量作用域泄漏             | P1   | 初始化/赋值     |
| TYPE-01  | 枚举值语义误用                           | P0   | 数据类型/枚举   |
| TYPE-02  | 通信引擎枚举变体混淆                     | P0   | 数据类型/枚举   |
| TYPE-03  | 平台字符串字面量拼写错误                 | P1   | 数据类型/枚举   |
| TYPE-04  | 参数对象混淆/引用错误                    | P0   | 数据类型/枚举   |
| TYPE-05  | map:at()对未收录key抛异常                | P0   | 数据类型/枚举   |
| QUAL-01  | 算法名称/变量名拼写错误                  | P1   | 代码质量/命名   |
| QUAL-02  | 全代码库拼写错误渗入API                  | P3   | 代码质量/命名   |
| QUAL-03  | 结构体/类字段变更未同步                  | P1   | 代码质量/命名   |
| QUAL-04  | 公共API函数名与头文件不匹配              | P1   | 代码质量/命名   |
| PTR-01   | 空指针检查顺序错误                       | P0   | 空指针/入参校验 |
| PTR-02   | API入口参数校验缺失                      | P1   | 空指针/入参校验 |
| API-01   | 抢跑依赖未发布API                        | P2   | 接口/API设计    |
| API-02   | 返回值语义差异未适配                     | P1   | 接口/API设计    |
| API-03   | 封装层能力不完整                         | P2   | 接口/API设计    |
| HW-01    | 硬件协议乘数因子遗漏                     | P0   | 硬件/平台适配   |
| HW-02    | 设备类型特殊路径未处理                   | P1   | 硬件/平台适配   |
| HW-03    | 硬件寄存器/地址映射宏冲突                | P0   | 硬件/平台适配   |
| STATE-01 | 超时值硬编码/配置不一致                  | P1   | 状态/超时管理   |
| STATE-02 | 环境变量/运行时状态一致性                | P2   | 状态/超时管理   |
| COND-01  | 条件方向错误/条件编译覆盖运行时分支      | P0   | 条件判断逻辑    |
| SEC-01   | token认证信息禁止打印到日志              | P0   | 网络安全        |
| SEC-02   | RDMA rkey/lkey不属于敏感信息（误报抑制） | 信息 | 网络安全        |

---

### 数据来源

- 仓库：hcomm（428次提交，84次缺陷），hcomm-dev（488次提交，162次缺陷），hccl-dev（133次提交，10次缺陷）
- 分析范围：共1049次提交，256次缺陷相关提交
- 分析方法：逐条git show分析缺陷提交diff + 热点文件结构性审查 + Revert专项分析
- 原48条规则来自hcomm仓库；新增31条规则来自hcomm-dev（通信框架开发仓）和hccl-dev（开源开发仓）的增量缺陷模式；10条已有规则通过dev仓数据补充了额外evidence
