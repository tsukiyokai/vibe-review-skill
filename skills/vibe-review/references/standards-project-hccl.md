# 【项目级】HCCL项目规则

## 1 网络安全

### 规则 1.1 token认证类信息属于华为公司产品网络安全红线中规定的敏感信息，禁止打印。

### 规则 1.2 在UB芯片上，当前只需要关注tokenId和tokenValue不能打印。

### 规则 1.3 当前RDMA协议中使用的rkey和lkey信息不用于鉴权功能，仅作为索引辅助硬件找到注册的内存，因此不属于敏感信息。

## 2 HCCL项目级编码规则

- tokenId/tokenValue禁止入日志（网络安全红线）。RDMA rkey/lkey不属于敏感信息（避免误报）
- 返回值：用 `CHK_RET()` 检查，仅日志用 `CHK_PRT()`
- 日志：必须用 `HCCL_DEBUG`/`HCCL_INFO`/`HCCL_WARNING`/`HCCL_ERROR`/`HCCL_RUN_INFO`，禁止printf/cout
- 内存分配：堆上用 `NEW_NOTHROW`，智能指针用 `CHK_SMART_PTR_NULL()` 检查
- 错误上报：输入 `RPT_INPUT_ERR`，环境 `RPT_ENV_ERR`，内部 `RPT_INNER_ERR_PRT`，外调 `RPT_CALL_ERR`

## 3 高价值缺陷模式

HCCL实际审查中发现的高频严重缺陷，保持高度敏感：

1. sizeof(容器)误用：`sizeof(std::vector<T>)` = 对象大小（24），非数据大小。用 `.size()`
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

基于hcomm仓库428次提交的完整git历史分析，从84次缺陷相关提交（占比19.6%）中提炼的48条高价值审查规则。每条规则均有commit证据和实际代码支撑。84条缺陷的直接覆盖率77%，含间接覆盖约90%。

### 严重等级定义

| 等级 | 含义 | 影响范围 |
|------|------|----------|
| P0 | 致命 | 进程crash、数据损坏、安全漏洞、集群级故障 |
| P1 | 严重 | 功能错误、静默精度劣化、资源泄漏、性能严重退化 |
| P2 | 一般 | 边界条件异常、可观测性缺失、日志误导 |
| P3 | 建议 | 代码质量、可维护性、潜在隐患 |

---

### 类别一：算法正确性（21次，25.0%）

集合通信算法实现中逻辑错误的各种表现形式，是HCCL项目最高频的缺陷类别。

#### 规则 ALG-01: 变量名遮蔽导致成员未赋值

严重等级: P0

缺陷描述: 构造函数或成员函数中，局部变量与成员变量同名，赋值写入局部变量而非成员变量。成员变量保持未初始化状态，后续使用产生未定义行为。

典型代码示例:

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

审查检查方法:
- 编译选项开启 `-Wshadow -Werror=shadow`，CI强制执行
- 构造函数体内赋值语句逐一确认目标是否为成员变量
- 优先使用初始化列表代替构造函数体内赋值

关联commit: `ef766683`

---

#### 规则 ALG-02: 变量自比较（tautological compare）

严重等级: P0

缺陷描述: 条件表达式中同一变量与自身比较（如 `x > x`），结果恒为常量，边界保护形同虚设。

典型代码示例:

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

审查检查方法:
- 编译选项开启 `-Wtautological-compare`
- CHK_PRT_RET/CHK_RET条件表达式须与错误消息字符串交叉验证：消息说比较A和B，条件也应是A和B的比较

关联commit: `e1880dc1`

---

#### 规则 ALG-03: 结构体/参数赋值遗漏

严重等级: P1

缺陷描述: 填充参数结构体时遗漏了某个字段的赋值。特别是成对字段（dataType/outputDataType、sendType/recvType），只赋值了一个而遗漏另一个。遗漏字段保持零值或默认值，导致运行时行为错误。

典型代码示例:

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

审查检查方法:
- 填充参数结构体时，列出结构体全部字段，逐一对比赋值语句，标记未赋值字段
- 成对字段（input/output、src/dst、send/recv前缀）确保不遗漏其一
- if/else分支中填充参数时，检查每个分支是否都做了完整赋值

关联commit: `7bc9e850`（dataType遗漏）, `1d171a92`（ipcMemDataSize遗漏）

---

#### 规则 ALG-04: 同族executor/子类一致性缺陷

严重等级: P1

缺陷描述: 继承体系中N个子类共享某段逻辑，但有1个子类遗漏。典型表现：N-1个executor有某个调用而1个没有；同一业务条件在不同子类中判断逻辑不一致。

典型代码示例:

```cpp
// 缺陷代码 — coll_all_to_all_v_direct_fullmesh_executor.cc
// 所有其他executor（AllGather、AllReduce、Broadcast等）的Orchestrate末尾都有：
//   CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
// 唯独 AlltoAllDirectFullmesh 遗漏了这一调用，导致aicpu cache功能异常

// 修复代码 — 在 Orchestrate 末尾补上
CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));
```

审查检查方法:
- 新增executor子类时，用同族其他executor作为checklist逐项对比
- N-1个子类有某段逻辑而1个没有，应标记为"遗漏"而非"不需要"，除非有明确注释说明原因
- 日志字符串中的类名/tag前缀必须与当前文件名或类名匹配（检测copy-paste错误）

关联commit: `f7183c87`（LaunchTaskExtend遗漏）, `bb681c5c`（preloadCopyOpt条件不一致）, `71fd0b86`（numBlocks校验不一致）, `12f3680c`（日志类名copy-paste错误）

---

#### 规则 ALG-05: 边界条件缺失

严重等级: P1

缺陷描述: 算法选择或执行路径未覆盖所有有效配置。典型场景：pipeline算法假设至少3卡但未排除2卡场景；参数校验上限值与实际类型范围不一致。

典型代码示例:

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

审查检查方法:
- 算法选择分支必须覆盖所有设备拓扑配置（1卡/2卡/多卡），每种配置标注预期行为
- 参数校验上限值须与参数实际类型范围一致（u8上限255、u16上限65535）
- Builder/fluent API构造对象时确认所有必要setter都被调用

关联commit: `666b6ab5`（2卡边界）, `7990e3f3`（repeat上限）, `8959e766`（selector遗漏SetRankSize）

---

#### 规则 ALG-06: Get函数不应有Set副作用

严重等级: P1

缺陷描述: 命名为Get*/Query*的查询函数内部调用了Set/Update/Insert操作，违反查询函数的纯净性约束。调用方无法预期查询操作会修改系统状态。

审查检查方法:
- Get*前缀函数体内不应包含Set/Update/Insert/Modify调用
- 方向对称性检查：`dst.*GetSource()` 或 `src.*GetTarget()` 模式标记为可疑

关联commit: `3323cecd`（GetEndpointDesc调用SetEndpointToIface）, `e4f59213`（查询函数修改成员变量curAlgName）

---

#### 规则 ALG-07: 多阶段/多版本流程变量混淆

严重等级: P1

缺陷描述: 多阶段流水线或多版本兼容代码中，阶段标识变量名相似导致混淆（intra vs inter、prepare vs launch、v1 vs v2）。典型表现：用错误的阶段变量作为条件判断。

典型代码示例:

```cpp
// 缺陷描述 — src/algorithm中CP pipeline实现
// intraState 和 interState 变量名相似
// 等待接收信息的代码本应检查 interState 却检查了 intraState
// 导致在错误的循环阶段执行等待操作
```

审查检查方法:
- 同一函数通过bool参数区分完全不同行为时，应拆分为两个独立函数
- 流水线多阶段的变量名须有明确区分前缀，审查条件是否指向正确阶段
- 多版本兼容路径中检查初始化顺序是否符合版本优先级

关联commit: `9bcb1bdc`（intra/inter混淆）, `af109f94`（prepare/launch阶段字段未区分）, `1a840065`（1.0/2.0初始化顺序）

---

#### 规则 ALG-08: alltoallv等非均匀集合通信的per-rank偏移逻辑

严重等级: P1

缺陷描述: 非均匀集合通信（alltoallv/allgatherv）中，每个rank的发送/接收偏移量和数据量不同。repeat模式下偏移量须逐轮累加，不能用通用标量运算代替。

典型代码示例:

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

审查检查方法:
- alltoallv/allgatherv等非均匀算子的repeat/分片逻辑须逐rank独立处理偏移
- 检查循环体内是否正确更新了所有per-rank的状态变量

关联commit: `fab6dbb7`

---

### 类别二：配置与兼容性（13次，15.5%）

多设备类型、多编译环境、多版本协议的适配问题。

#### 规则 CFG-01: 跨版本协议OpCode修改须保持向后兼容

严重等级: P0

缺陷描述: 修改已发布的OpCode对应的数据结构时，直接复用原OpCode编号。新版本发送新结构体给旧版本，旧版本按老结构体解析导致协议不兼容。

典型代码示例:

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

审查检查方法:
- 修改已发布OpCode的数据结构时，必须分配新OpCode编号
- 保留旧OpCode的处理函数（至少一个版本周期）
- 发送前须查询对端版本能力，选择兼容的OpCode

关联commit: `5ad2ced6`

---

#### 规则 CFG-02: 设备类型分支须全面覆盖

严重等级: P1

缺陷描述: 新增硬件资源初始化或引擎类型时，仅在部分设备类型分支中处理，其他设备类型分支遗漏。或者新增引擎类型后未全局检查所有引擎分支代码。

审查检查方法:
- 新增硬件资源初始化须审查是否适用所有设备类型（910_93/910_95/A3等）
- 新增引擎类型后须全局grep所有按引擎分支的代码确认归属
- 按设备类型分支时审查各分支功能等价性

关联commit: `099fe2a8`（心跳socket类型判断不一致）, `07c52708`（910_95 profiling不兼容）, `91fbd1d6`（A3 AIV跨节点回退1700行）, `9d75f557`（AIV引擎归入CPU分支）

---

#### 规则 CFG-03: 常量不应承担多重语义

严重等级: P2

缺陷描述: 同一个常量既作为默认值又作为哨兵值（标识"未配置"），甚至其数值本身超出合法范围。

典型代码示例:

```cpp
// 缺陷代码 — src/legacy/framework/topo/new_topo_builder/rank_table_info/new_rank_info.h
constexpr unsigned int MAX_VALUE_DEVICEPORT = 65536;  // 既做默认值又做哨兵值，且超过合法端口上限65535
u32 devicePort{MAX_VALUE_DEVICEPORT};

// 修复代码 — 分离默认值与上限
constexpr unsigned int DEFAULT_VALUE_DEVICEPORT = 60001;  // 明确的默认端口
constexpr unsigned int MAX_VALUE_DEVICEPORT = 65535;       // 合法端口上限
u32 devicePort{DEFAULT_VALUE_DEVICEPORT};
```

审查检查方法:
- 一个常量只承担一个语义
- 端口号上限为65535而非65536
- 硬件常量修改须在注释中标注取值依据（spec文档引用）

关联commit: `2bfa02d1`（端口默认值/哨兵值混淆）, `fb56d64b`（CCU loop count常量值错误64->128）

---

#### 规则 CFG-04: v1/v2接口分发完整性

严重等级: P1

缺陷描述: op_base.cc中的公开API须同时有v1和v2版本的分发入口。新增或修改API时遗漏HCCLV2_FUNC_RUN分发，导致v2调用路径无法到达。

审查检查方法:
- op_base.cc中所有公开API须有HCCLV2_FUNC_RUN分发（可自动化扫描）
- 字符串到枚举映射表中key须与枚举值名称一致

关联commit: `6383b2bc`（HcclGetCommConfigCapability缺分发）, `edf73e80`（协议名字符串映射不匹配）

---

### 类别三：并发问题（10次，11.9%）

通信库最核心的缺陷类型，涵盖竞态、死锁、原子性违反、同步时序。

#### 规则 CON-01: 共享数据写后更新标志须有内存屏障

严重等级: P0

缺陷描述: "先写数据、后更新标志位"的生产者-消费者模式中，缺少memory fence。CPU或编译器可能重排store指令，导致消费者看到标志位更新但数据还未写入。

典型代码示例:

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

审查检查方法:
- "先写数据、后更新标志位"模式必须在两步之间插入memory fence
- 跨线程/跨进程的共享内存通信，标志位更新前须有acquire-release语义保证

关联commit: `0947b660`

---

#### 规则 CON-02: Record/Wait同步顺序——先Wait依赖方再Record通知上游

严重等级: P0

缺陷描述: 集合通信broadcast中，中继rank先Record通知root"数据已取走"，然后才Wait其他rank完成。root可能在下游未完成时就覆盖buffer。

典型代码示例:

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

审查检查方法:
- Record/Wait顺序原则：先Wait依赖方完成，再Record通知上游
- 集合通信中root/coordinator应是最后退出的角色

关联commit: `a0f05be2`

---

#### 规则 CON-03: publish ordering——先完成工作再修改对外可见状态

严重等级: P0

缺陷描述: 多线程环境下先重置状态标记再清理资源。其他线程看到状态已重置就认为可以使用，但资源实际还未清理完成。

典型代码示例:

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

审查检查方法:
- publish ordering原则：先完成实际清理/初始化工作，再修改对外可见的状态标志
- 状态重置函数调用应在所有资源清理操作之后

关联commit: `75659d24`

---

#### 规则 CON-04: 并发delete须加锁保护

严重等级: P0

缺陷描述: 多线程可能并发调用含delete操作的销毁函数，无锁保护导致double-free。常见模式：check-then-act（先判空再delete）在并发下不安全。

典型代码示例:

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

审查检查方法:
- delete后置nullptr的模式在多线程环境下不安全，必须加锁
- 审查所有Destroy/Cleanup函数是否可能被并发调用

关联commit: `36994739`

---

#### 规则 CON-05: atomic变量的所有读取必须用.load()

严重等级: P1

缺陷描述: `std::atomic`变量用隐式转换（如 `==` 操作符）读取时，在某些编译器/优化级别下可能不产生原子load指令。

典型代码示例:

```cpp
// 缺陷代码 — src/framework/communicator/impl/hccl_communicator_host.cc
if (g_enableBackupLinkCommCount == 0) {  // 隐式非原子读

// 修复代码
if (g_enableBackupLinkCommCount.load() == 0) {  // 显式原子load
```

审查检查方法:
- 所有`std::atomic`变量的读取点必须使用`.load()`显式调用
- 可用clang-tidy的`bugprone-use-after-move`等checker辅助检测

关联commit: `6b354394`

---

#### 规则 CON-06: 单例/引用计数Init方法须区分"只需一次"和"每次都需"

严重等级: P1

缺陷描述: 单例模式的Init方法中，部分逻辑只需首次执行（如注册回调），部分逻辑每次创建通信域都需执行（如设置白名单）。引用计数>1时直接跳过整个Init，导致后续通信域缺少必要配置。

审查检查方法:
- 单例Init方法中明确标注哪些操作是once-only、哪些是every-time
- 引用计数递增路径也须检查是否有必须每次执行的逻辑

关联commit: `eb9be21b`

---

#### 规则 CON-07: thread_local变量须在线程入口点统一设置

严重等级: P1

缺陷描述: `thread_local`变量（如`gDispatcherCtx`）的设置逻辑被嵌在某个业务函数深处，新线程如果没有经过该函数就调用了依赖此变量的其他函数，访问到未初始化的thread_local值。

典型代码示例:

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

审查检查方法:
- thread_local变量的设置须作为线程入口的第一步操作，不应埋在业务函数内部
- 新增线程入口时须审查是否依赖了已有的thread_local变量
- thread_local变量的所有使用点须确认该线程已经过设置路径

关联commit: `a6e4d199`

---

### 类别四：日志与调试（8次，9.5%）

可审查性高，多数可通过编译器警告或lint工具自动捕获。

#### 规则 LOG-01: 格式化字符串占位符须与实参类型匹配

严重等级: P2

缺陷描述: printf-style格式化中，`%s`传入int、`%u`传入u64、参数与占位符错位。可能导致日志输出垃圾值或crash。

审查检查方法:
- 编译选项开启 `-Wformat -Werror=format`
- `std::string`传入C格式化函数须用`.c_str()`
- 禁止字符串字面量中用 `\` 续行（会将下一行前导空格纳入字符串）

关联commit: `35e32c7c`（%s传入int）, `6a6eac0f`（u64用%u、u32用%llu）

---

#### 规则 LOG-02: 日志前缀/tag必须与所在类名/函数名匹配

严重等级: P3

缺陷描述: copy-paste代码后未更新日志中的模块名/类名标识，误导定位。

审查检查方法:
- 日志前缀/tag必须与所在函数名或类名匹配
- 日志模块ID必须在枚举中有明确定义
- 可用脚本自动检测日志tag与代码位置的一致性

关联commit: `7ff92abc`, `112766de`（缺少[HCCL_ENV]前缀）, `12f3680c`（日志类名copy-paste错误）, `ed50e7eb`（日志模块ID未在枚举中定义）

---

#### 规则 LOG-03: 快速路径必须与标准路径保持功能对等

严重等级: P2

缺陷描述: cache hit或fast path跳过了profiling/tracing/DFX信息采集，导致性能分析和问题定位时信息缺失。

审查检查方法:
- 所有快速路径（cache hit/fast path）必须与标准路径保持profiling/tracing/logging功能对等
- 新增fast path时逐项对比标准路径的DFX调用

关联commit: `891665ac`（SQE缓存命中缺profiling）, `3a9b61f2`（CCU DFX结构体缺字段）

---

### 类别五：资源生命周期（8次，9.5%）

资源的创建、使用、销毁三阶段管理不当。

#### 规则 RES-01: early return跳过必要初始化/清理

严重等级: P1

缺陷描述: 函数包含多个early return路径，某个return跳过了后续必要的初始化或清理逻辑。

审查检查方法:
- 函数包含多个early return时，逐一审查每个return路径是否遗漏后续必要初始化/清理
- 调试用的`return HCCL_SUCCESS`须在提交前移除（lint可检测unreachable code）

关联commit: `82989927`（early return跳过TpManager::Init）, `c5443da0`（调试遗留return跳过整个逻辑）

---

#### 规则 RES-02: map/cache的key必须能唯一标识资源

严重等级: P1

缺陷描述: map/cache的key设计不充分，未包含所有区分资源的关键属性。不同语义的资源请求命中同一key，复用了错误的资源。

典型代码示例:

```cpp
// 缺陷描述 — netDevCtxMap_ 的key仅用IP，同IP不同端口复用错误NetDevCtx
// BatchSendRecv 资源key仅用algName/opTag，不同remote rank组合复用错误资源
// 修复：将端口/remote rank信息编入key
```

审查检查方法:
- 设计map/cache的key时，问"同一key下是否可能出现不同语义的资源请求？"
- key须覆盖所有区分资源的关键属性（IP+端口、algName+remoteRank等）
- 资源销毁顺序须按依赖关系图反序

关联commit: `43069da0`（IP不含端口）, `18752141`（key不含remote rank）

---

#### 规则 RES-03: context切换附近的内存操作须审查执行context

严重等级: P1

缺陷描述: 代码中切换到不同的设备context后分配/释放内存，但操作应在原context下执行。

审查检查方法:
- context切换（SetDevice/SwitchContext）附近的内存分配须审查执行context是否正确
- 枚举值switch/if匹配需覆盖所有有效值

关联commit: `3b8ac218`（DPU context下错误分配NPU内存）, `6b9278e4`（图模式remote memory更新遗漏）

---

#### 规则 RES-04: 析构路径须完整——引用计数递减也须检查局部资源释放

严重等级: P1

缺陷描述: 引用计数递减到非零值时直接返回，但该路径上可能有局部获取的资源（如映射handle）需要释放。

审查检查方法:
- 分配/映射的资源在释放函数中须有明确释放路径
- 引用计数递减的每个分支都须检查是否有局部资源需释放
- 析构函数中的日志/DFX调用应在所有资源释放操作之前（避免日志依赖先被销毁）

关联commit: `a9fea200`（refCount>1分支遗漏释放handle）, `7d914168`（析构顺序错误导致日志依赖先销毁）

---

### 类别六：错误处理（7次，8.3%）

#### 规则 ERR-01: 异常上报操作须受前置条件保护

严重等级: P2

缺陷描述: 异常上报/错误通知在正常路径上也无条件执行，导致正常操作触发虚假的异常打印。

审查检查方法:
- 异常上报操作（SendTaskExceptionByMBox等）必须受前置条件判断保护
- 重试成功路径不应触发异常上报

关联commit: `96087ffb`（重试成功也触发异常打印）

---

#### 规则 ERR-02: 项目有统一异常处理宏时不允许裸try-catch

严重等级: P2

缺陷描述: 项目提供了统一异常处理宏（如TRY_CATCH_PROCESS_THROW），但开发者手写try-catch吞掉异常只打日志return，导致异常类型信息丢失。

典型代码示例:

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

审查检查方法:
- 搜索裸try-catch块，确认是否应使用项目统一宏
- `catch(...)` 块不应默默return，至少需要向上传播异常类型信息

关联commit: `69c73166`

---

#### 规则 ERR-03: C接口函数不应抛C++异常

严重等级: P0

缺陷描述: 对外接口函数内部调用了含`THROW`的inline函数（如`MC2DataType()`、`MC2ReduceType()`），当参数非法时抛出C++异常。若调用方无法处理异常（如通过dlsym加载或跨语言调用），异常穿越接口边界导致`std::terminate()`直接abort进程。

典型代码示例:

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

审查检查方法:
- 对外接口函数体内不应有未catch的throw路径（含间接调用的inline函数中的THROW）
- 接口边界处使用错误码返回而非异常

关联commit: `9939a862`

---

#### 规则 ERR-04: CHK_RET调用顺序须与业务逻辑匹配

严重等级: P2

缺陷描述: 带CHK_RET宏包裹的函数调用放在条件判断之前，非相关场景的函数失败会阻断整个函数。

审查检查方法:
- CHK_RET包裹的函数调用应延迟到确认需要其返回值处
- if-else分支设置错误状态后检查是否遗漏return

关联commit: `9476c6df`（CHK_RET调用顺序不当）, `e025b6c5`（if-else缺return导致fallthrough）

---

#### 规则 ERR-05: 预期内返回值不应用error日志级别

严重等级: P2

缺陷描述: 调用底层接口时，某些返回值（如`-ENOTSUPP`）是预期内的"功能不支持"语义，但代码统一用error级别日志打印，产生大量误导性错误日志，掩盖真正的错误。

典型代码示例:

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

审查检查方法:
- 调用可能返回"不支持/不可用"语义值的底层接口时，须区分日志级别
- `-ENOTSUPP`、`-EOPNOTSUPP`、`HCCL_E_NOT_SUPPORT`等返回值应用warn而非error
- error级别日志应保留给真正需要人工介入的错误

关联commit: `2fcde546`

---

### 类别七：缓存一致性（6次，7.1%）

缓存/复用机制中key生成不完整或缓存数据过期的问题。HCCL项目特有的高频模式。

#### 规则 CACHE-01: cache key须覆盖所有影响执行路径的状态维度

严重等级: P1

缺陷描述: 缓存key的维度不够，新增的分支条件未反映在key中，导致不同执行路径命中同一缓存。

典型代码示例:

```cpp
// 缺陷代码 — src/framework/device/framework/aicpu_cache_manager.cc
// AlltoAllV cache key没有区分 isBigCount，big/small count两种不同SQE编排方案命中同一缓存

// 修复代码 — 将 isBigCount 编码进 inputSize 字段作为key的一部分
bool isBigCountForAlltoallv = false;
CHK_RET(IsBigCountForAlltoallv(param, topoinfo, isBigCountForAlltoallv));
inputSize = isBigCountForAlltoallv ? 1 : 0;  // 编码进cache key
```

审查检查方法:
- 新增if/switch分支逻辑如果影响执行路径，须检查是否反映在cache key中
- cache key维度checklist：opType、algName、dataType、count/size、拓扑信息、特殊标志位
- 分支条件与缓存key必须使用相同数据源

关联commit: `e0744b7b`（缺isBigCount维度）, `18752141`（缺remote rank信息）

---

#### 规则 CACHE-02: 缓存复用前须逐字段确认是否需要刷新

严重等级: P1

缺陷描述: 缓存命中后直接复用所有缓存字段，但某些字段（如stream handle）在每次调用时可能变化，复用过期值导致错误。

典型代码示例:

```cpp
// 缺陷代码 — src/framework/communicator/impl/hccl_communicator_host.cc
// ExecOpCache 缓存命中时复用了旧的stream，没有用当前OpParam中的新stream

// 修复代码 — 在缓存复用时刷新可变字段
cacheInfo.resourceArgs.stream = opParam.stream.ptr();  // 刷新stream
```

审查检查方法:
- 缓存复用前逐字段确认哪些可能变化，对可变字段加刷新逻辑（staleness checklist）
- 句柄类字段（stream、context、device handle）几乎总需要刷新
- 缓存用于恢复逻辑时检查是否保存了所有依赖的输入状态

关联commit: `43dab3e2`（stream未刷新）, `dd053d5b`（algName未随缓存恢复）

---

#### 规则 CACHE-03: 函数调用前后使用同一成员变量时须确认中间调用是否有副作用

严重等级: P1

缺陷描述: 在调用某函数前后都使用了同一成员变量，但该函数有副作用会修改这个成员变量。后续代码使用的是被修改后的值而非预期的原始值。

典型代码示例:

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

审查检查方法:
- 函数调用前后使用同一成员变量时，审查被调用函数是否修改该变量
- 有副作用的函数须在注释或命名中体现（如OpAcceleratorStateFallback暗示会修改状态）

关联commit: `e4f59213`

---

#### 规则 CACHE-04: 运行时分支条件与缓存key必须使用相同数据源

严重等级: P1

缺陷描述: 运行时判断走哪条执行路径用了变量A，但生成缓存key时用了变量B。A和B在大多数情况下一致，但在边界情况（如最后一轮最后一张卡）下不一致，导致缓存复用时走错路径。

审查检查方法:
- 运行时分支条件和缓存key生成必须使用完全相同的数据源
- 用虚函数或统一getter抽象数据源，避免两处分别计算

关联commit: `b0e6a8b7`（运行时用curSize_判断，key用perRankAvgDataSize_）

---

### 类别八：内存管理（4次，4.8%）

#### 规则 MEM-01: 禁止将局部容器的内部指针暴露给外部

严重等级: P0

缺陷描述: 将栈上局部vector的`.data()`指针通过输出参数返回。函数返回后vector析构，调用方拿到悬垂指针。

典型代码示例:

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

审查检查方法:
- 禁止将局部容器（vector/string/array）的内部数据指针通过输出参数或返回值暴露给外部
- 需要返回数据所有权时，使用malloc+memcpy或std::unique_ptr

关联commit: `6784944a`（此缺陷由`4e82ec25`的修复引入——缺陷修复引入新缺陷的典型案例）

---

#### 规则 MEM-02: IPC共享内存注册须检查重复注册

严重等级: P1

缺陷描述: P2P传输中同一块物理内存被重复注册IPC映射，大规模通信下资源耗尽OOM。

审查检查方法:
- IPC共享内存注册前检查是否已注册同一块内存
- bool成员变量默认false且在多处以`!flag`守护资源分配时，检查所有初始化路径是否都有flag设置

关联commit: `3ec7410b`（IPC重复注册OOM）, `b2e74aee`（SetMemIncludeFlag遗漏调用）

---

#### 规则 MEM-03: 可选初始化路径的成员指针使用前必须判空

严重等级: P1

缺陷描述: 成员变量仅在特定配置条件下创建（如`symmetricMemory_`仅在跨超节点场景初始化），但使用该成员的函数没有判空保护，其他配置下调用时空指针解引用crash。

典型代码示例:

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

审查检查方法:
- 成员变量仅在特定条件下初始化时，所有使用点须有判空保护
- 新增对可选成员的使用时，须确认所有初始化路径是否都覆盖了该成员的创建
- 构造函数/Init中有条件分支跳过某些成员初始化时，标记这些成员为"可选"并在使用处强制判空

关联commit: `4e0bd97b`

---

### 类别九：整数溢出/截断（2次，2.4%）

可审查性极高，编译器警告即可捕获。

#### 规则 INT-01: 宽类型向窄类型转换必须检查溢出

严重等级: P0

缺陷描述: u64赋值给u32时高32位截断，导致内存大小计算远小于实际需要，后续buffer overflow。

典型代码示例:

```cpp
// 缺陷代码 — src/framework/hcom/hcom.cc
u32 count = hcomOpParam->count;  // hcomOpParam->count 是 u64，截断为 u32
// 当count > 4GB个元素时，scratch memory远小于实际需要

// 修复代码
u64 count = hcomOpParam->count;  // 保持u64
```

审查检查方法:
- 所有`static_cast`从宽类型向窄类型的转换必须检查溢出/截断风险
- 涉及内存大小计算的表达式必须使用u64
- 编译选项开启 `-Wconversion -Werror=conversion`

关联commit: `9c1f957b`

---

#### 规则 INT-02: 加法后窄化须有饱和逻辑

严重等级: P1

缺陷描述: 两个值相加后cast到窄类型，和超过窄类型上限时回绕到极小值。对于timeout场景，回绕导致超时时间从"很长"变成"几乎为零"。

典型代码示例:

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

审查检查方法:
- 加法/乘法后窄化的表达式须有饱和逻辑或溢出检查
- 特别关注timeout、size、count类变量的窄化

关联commit: `6baf33c4`

---

### 类别十：C++语言特性（1次，1.2%）

#### 规则 CPP-01: 跨SO边界的多态类虚析构不应使用 = default

严重等级: P1

缺陷描述: `= default`的虚析构函数由编译器在每个翻译单元隐式内联生成。当基类和派生类位于不同SO时，链接器可能找不到虚析构函数的符号。

典型代码示例:

```cpp
// 缺陷代码 — pkg_inc/hcomm/ccu/ccu_kernel.h (头文件, 被多个SO include)
~CcuKernel() override = default;  // 每个TU内联生成，跨SO时找不到符号

// 修复代码 — 头文件声明，.cc文件定义
// ccu_kernel.h
~CcuKernel() override;

// ccu_kernel.cc
CcuKernel::~CcuKernel() {}  // 显式定义在单一TU，确保符号导出
```

审查检查方法:
- 在pkg_inc或公共头文件中导出的多态类，虚析构函数不应使用`= default`
- 应在.cc文件中提供显式定义

关联commit: `bb490dc2`

---

### 类别十一：构建系统与Revert（2次，2.4%）

两次revert均代表缺陷逃逸到主干后紧急撤回，平均合入到revert间隔3.5小时。

#### 规则 BUILD-01: 大变更合入须有充分的pre-merge验证

严重等级: P2

缺陷描述: 24文件的构建系统重构合入1小时后被revert；硬件常量修改同时修改了UT期望值适配变更。两次PR描述均为空模板。

审查检查方法:
- PR描述为空模板的变更不应合入主干
- 同时修改源码和对应UT期望值时，审查UT修改的合理性——UT应独立验证而非适配变更
- 硬件常量修改须注明取值依据（spec文档引用）
- 构建系统大变更(>5个文件)应分步合入，每步独立验证
- 一个commit只做一件事：逻辑变更和风格变更必须分离

关联commit: `72cdf80e`（revert fb56d64b, CCU loop count常量修改无spec依据, 合入6小时后撤回）, `753ba8c2`（revert 05b38411, 24文件构建重构, 合入1小时后撤回）

---

### 跨类别系统性风险

以下模式跨越多个缺陷类别，是code review时应优先关注的系统性风险。

#### 规则 SYS-01: 缺陷修复提交须额外审查是否引入新缺陷

严重等级: P1

缺陷描述: `4e82ec25`修复API参数方向问题时引入了use-after-free（将局部vector的data指针返回），被`6784944a`再次修复。修复提交的审查重心往往只在"原缺陷是否被修"，忽略了修复代码本身的正确性。

审查检查方法:
- 缺陷修复提交须有额外审查关注点：修复代码本身是否引入新问题
- 修复提交中新增的代码应按新代码标准审查，而非仅关注是否解决了原bug

关联commit: `4e82ec25` -> `6784944a`

---

#### 规则 SYS-02: 修复一处须搜索重复代码段

严重等级: P2

缺陷描述: 热点文件中存在大量代码重复（如ExecOp vs ExecOpAlltoAll、AllocTransportResource三路重复），修复一处时忘记修复重复代码中的同样问题。

审查检查方法:
- 修复一处缺陷后，全局搜索相似代码段是否有同样问题
- 对重复代码优先重构为共享函数，从根本上消除"修一漏一"

关联证据: hccl_communicator_host.cc（ExecOp重复）, aicpu_communicator.cc（AllocTransportResource三路重复）, communicator_impl.cc（5个初始化路径）

---

#### 规则 SYS-03: God Object文件修改须加强审查

严重等级: P3

缺陷描述: 三个热点文件占全部缺陷修复的36.9%:

| 文件 | 方法数/行数 | 缺陷修复次数 |
|------|------------|-------------|
| hccl_communicator_host.cc | 312方法/9152行 | 13次 |
| aicpu_communicator.cc | 213函数/5550行 | 9次 |
| communicator_impl.cc | ~100函数/3789行 | 9次 |

审查检查方法:
- 任何对这3个文件的修改应要求至少2个reviewer
- 长期策略：拆分God Object为职责单一的小类

---

#### 规则 SYS-04: 返回值系统性忽略

严重等级: P1

缺陷描述: 热点文件中多处忽略关键函数返回值:
- `hrtMalloc`返回值未检查（hccl_communicator_host.cc L1803）
- `malloc` 100MB未检查返回值（communicator_impl.cc L3115）
- `getenv`返回nullptr直接解引用（communicator_impl.cc L3025）

审查检查方法:
- 内存分配函数（malloc/hrtMalloc）返回值必须检查
- getenv()返回值必须判空
- 编译选项开启 `-Wunused-result`

---

#### 规则 SYS-05: 编译器可捕获的缺陷须在CI中强制开启

严重等级: P2

缺陷描述: 以下缺陷本可由编译器警告捕获但仍进入主干:

| 编译选项 | 可捕获的缺陷 | 关联commit |
|---------|-------------|-----------|
| `-Wshadow` | 变量遮蔽 | ef766683 |
| `-Wtautological-compare` | 自比较 | e1880dc1 |
| `-Wformat` | 格式串不匹配 | 35e32c7c, 6a6eac0f |
| `-Wconversion` | 整数截断 | 9c1f957b, 6baf33c4 |
| `-Werror=enum-conversion` | 枚举隐式转换 | d953cbf3 |

审查检查方法:
- CI中强制开启上述警告并设为 `-Werror`
- 新增代码不允许suppress这些警告

---

#### 规则 SYS-06: 热点文件中仍存在的已知高危风险

严重等级: P1

以下是热点文件中当前仍存在的高危问题（截至分析时点）:

| 风险 | 位置 | 描述 |
|------|------|------|
| 悬空指针 | communicator_impl.cc L601 | 临时unique_ptr.get()传出 |
| malloc未检查 | communicator_impl.cc L3115 | malloc 100MB未检查返回值 |
| getenv空指针 | communicator_impl.cc L3025 | getenv返回nullptr直接解引用 |
| TLV解析死循环 | aicpu_communicator.cc L659 | length==0时while循环无法前进 |
| 锁策略不一致 | aicpu_communicator.cc resMap_ | 同一数据结构两种锁保护 |

---

### 附录：规则速查表

| 规则ID | 规则名称 | 等级 | 类别 |
|--------|---------|------|------|
| ALG-01 | 变量名遮蔽导致成员未赋值 | P0 | 算法正确性 |
| ALG-02 | 变量自比较 | P0 | 算法正确性 |
| ALG-03 | 结构体/参数赋值遗漏 | P1 | 算法正确性 |
| ALG-04 | 同族executor一致性缺陷 | P1 | 算法正确性 |
| ALG-05 | 边界条件缺失 | P1 | 算法正确性 |
| ALG-06 | Get函数不应有Set副作用 | P1 | 算法正确性 |
| ALG-07 | 多阶段/多版本流程变量混淆 | P1 | 算法正确性 |
| ALG-08 | 非均匀集合通信per-rank偏移 | P1 | 算法正确性 |
| CFG-01 | 跨版本协议OpCode兼容 | P0 | 配置与兼容性 |
| CFG-02 | 设备类型分支全面覆盖 | P1 | 配置与兼容性 |
| CFG-03 | 常量不应承担多重语义 | P2 | 配置与兼容性 |
| CFG-04 | v1/v2接口分发完整性 | P1 | 配置与兼容性 |
| CON-01 | 共享数据写后更新标志须有内存屏障 | P0 | 并发问题 |
| CON-02 | Record/Wait同步顺序 | P0 | 并发问题 |
| CON-03 | publish ordering | P0 | 并发问题 |
| CON-04 | 并发delete须加锁 | P0 | 并发问题 |
| CON-05 | atomic变量须用.load()读取 | P1 | 并发问题 |
| CON-06 | 单例Init区分once/every-time | P1 | 并发问题 |
| CON-07 | thread_local变量须在线程入口点设置 | P1 | 并发问题 |
| LOG-01 | 格式化字符串类型匹配 | P2 | 日志与调试 |
| LOG-02 | 日志tag须匹配类名/函数名 | P3 | 日志与调试 |
| LOG-03 | 快速路径功能对等 | P2 | 日志与调试 |
| RES-01 | early return跳过初始化/清理 | P1 | 资源生命周期 |
| RES-02 | map/cache key唯一标识资源 | P1 | 资源生命周期 |
| RES-03 | context切换附近内存操作审查 | P1 | 资源生命周期 |
| RES-04 | 析构路径完整性 | P1 | 资源生命周期 |
| ERR-01 | 异常上报须受前置条件保护 | P2 | 错误处理 |
| ERR-02 | 不允许裸try-catch吞异常 | P2 | 错误处理 |
| ERR-03 | 对外接口不应抛C++异常 | P0 | 错误处理 |
| ERR-04 | CHK_RET调用顺序 | P2 | 错误处理 |
| ERR-05 | 预期内返回值日志级别不当 | P2 | 错误处理 |
| CACHE-01 | cache key覆盖所有状态维度 | P1 | 缓存一致性 |
| CACHE-02 | 缓存复用前逐字段确认刷新 | P1 | 缓存一致性 |
| CACHE-03 | 函数副作用污染成员变量 | P1 | 缓存一致性 |
| CACHE-04 | 分支条件与key同数据源 | P1 | 缓存一致性 |
| MEM-01 | 禁止暴露局部容器内部指针 | P0 | 内存管理 |
| MEM-02 | IPC共享内存重复注册检查 | P1 | 内存管理 |
| MEM-03 | 可选初始化路径成员指针须判空 | P1 | 内存管理 |
| INT-01 | 宽类型向窄类型转换溢出检查 | P0 | 整数溢出 |
| INT-02 | 加法后窄化须饱和逻辑 | P1 | 整数溢出 |
| CPP-01 | 跨SO虚析构不应= default | P1 | C++语言特性 |
| BUILD-01 | 大变更pre-merge验证 | P2 | 构建系统 |
| SYS-01 | 缺陷修复须审查是否引入新缺陷 | P1 | 系统性风险 |
| SYS-02 | 修复一处须搜索重复代码段 | P2 | 系统性风险 |
| SYS-03 | God Object文件加强审查 | P3 | 系统性风险 |
| SYS-04 | 返回值系统性忽略 | P1 | 系统性风险 |
| SYS-05 | CI编译器警告强制开启 | P2 | 系统性风险 |
| SYS-06 | 热点文件已知高危风险 | P1 | 系统性风险 |

---

### 数据来源

- 仓库: /Users/shanshan/repo/cann/hcomm
- 分析范围: 428次提交，84次缺陷相关提交（19.6%）
- 分析方法: 逐条git show分析缺陷提交diff + 热点文件结构性审查 + Revert专项分析
- 缺陷类别分布: 算法正确性(25%) > 配置与兼容性(15.5%) > 并发问题(11.9%) > 日志与调试(9.5%) = 资源生命周期(9.5%) > 错误处理(8.3%) > 缓存一致性(7.1%) > 内存管理(4.8%) > 整数溢出(2.4%) > 构建系统(2.4%) > C++语言特性(1.2%)
