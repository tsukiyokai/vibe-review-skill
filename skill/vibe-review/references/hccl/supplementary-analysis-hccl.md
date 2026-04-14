# HCCL/HCOMM 补充分析

数据来源: hccl(hcomm仓库, 428提交) + hccl-dev(133提交) + hcomm-dev(488提交)

---

## Part 1: 热点文件风险分析

数据来源: hccl + hccl-dev + hcomm-dev

---

### hccl (hcomm仓库)


### 1. 缺陷热点文件排名

基于84次缺陷修复提交的git diff-tree统计，以下文件被缺陷修复最频繁地触及：

| 排名 | 文件路径 | 缺陷修复次数 | 行数 |
|------|---------|------------|------|
| 1 | src/framework/communicator/impl/hccl_communicator_host.cc | 13 | 9152 |
| 2 | src/framework/device/framework/aicpu_communicator.cc | 9 | 5550 |
| 3 | src/legacy/framework/communicator/communicator_impl.cc | 9 | 3789 |
| 4 | src/legacy/framework/communicator/communicator_impl.h | 6 | 585 |
| 5 | src/framework/hcom/hcom.cc | 4 | - |
| 6 | src/platform/resource/transport/host/transport_p2p.cc | 3 | - |
| 6 | src/legacy/service/collective/alg/interface/host/coll_alg_component.cc | 3 | - |
| 6 | src/legacy/framework/resource_manager/socket/socket_manager.cc | 3 | - |
| 6 | src/legacy/framework/entrance/op_base/op_base_v2.cc | 3 | - |
| 6 | src/framework/op_base/src/op_base.cc | 3 | - |
| 6 | src/algorithm/impl/operator/coll_alg_operator.cc | 3 | - |

前3名文件占全部缺陷修复的36.9%（31/84），且都是通信框架的核心实现——God Object特征明显。

---

### 2. 热点文件结构性风险详细分析

#### 2.1 hccl_communicator_host.cc（缺陷修复13次，9152行，312个方法）

这是全仓库最大、缺陷最密集的文件。

##### 高危风险

R1-1 资源泄漏——hrtMalloc返回值未检查
- 位置：L1803
- `hrtMalloc(&sendAlgParamMemPtr, sizeof(AivSuperKernelArgs))` 返回值被丢弃，分配失败后sendAlgParamMemPtr为nullptr，后续hrtMemSyncCopy和赋值均为UB

R1-2 返回值忽略——流分配失败后继续
- 位置：L6205
- `Mc2AiCpuStreamAllocAndGet(streamMode, aicpuStream)` 返回值未用CHK_RET检查，流分配失败后aicpuStream为未初始化变量

R1-3 构造函数OOM后继续运行
- 位置：L126-134, L159-167
- `new (std::nothrow) MrManager()` 返回nullptr时仅打印ERROR继续执行，后续使用产生空指针解引用

R1-4 构造函数初始化列表不一致
- 位置：L105-136 vs L138-169
- 两个构造函数约20个成员的初始化列表不完全相同，第二个漏了 `multiSuperPodDiffDeviceNumMode_(false)`

R1-5 数组越界——devIpAddr_[0]无前置空检查
- 位置：L1207, L2580
- 直接使用 `devIpAddr_[0]` 而不检查vector是否非空

##### 中危风险

R1-6 ExecOp与ExecOpAlltoAll大段重复（243行 vs 278行）
- 位置：L4356-4597 vs L4609-4885
- cache查找、算法选择、资源创建、心跳注册、DFX注册、计数器等逻辑大量重复。历史缺陷反复证实"修一处漏另一处"模式

R1-7 g_enableBackupLinkCommCount的TOCTOU竞态
- 位置：L214-218 vs L1210
- 全局原子变量的check-then-act操作非原子，多通信域并发销毁时计数可能错乱

R1-8 Init函数锁粒度不足
- 位置：L346-358
- g_hcomInitMutex只保护一小段初始化，之后的RegisterKernel、LoadAICPUKernel等在锁外执行

R1-9 reinterpret_cast大量使用
- 位置：L746, L5126, L5167, L5180, L7925, L8001-8029
- 指针转u64在host/device通信参数中传递，依赖平台指针宽度，跳过类型检查

R1-10 resMap_的tag管理与captureCnt_共享
- 位置：L4440-4458, L4706-4722
- 多算子类型共享同一计数器captureCnt_，图模式capture场景下可能造成tag冲突

R1-11 忙等待——Snapshot处理中无sleep忙循环
- 位置：L9003-9031, L9045-9073
- 多个while(true)循环无sleep/yield，100% CPU空转

R1-12 析构函数149行，调用大量可失败操作
- 位置：L171-318
- 中间操作失败后续资源不释放

R1-13 有符号/无符号混用
- 位置：L120, L1257, L1344, L1369
- devicePhyId_(u32) 用 static_cast<s32> 比较，大值时为实现定义行为

R1-14 const_cast丢弃const
- 位置：L2694-2700
- sendBuf/recvBuf的const被移除后存入OpParam，下游若修改则为UB

---

#### 2.2 aicpu_communicator.cc（缺陷修复9次，5550行，213个函数）

##### 高危风险

R2-1 Orchestrate状态机过于复杂
- 位置：L2109-L2300（192行）
- while(true)循环内12个case分支，状态转换路径组合爆炸，测试覆盖极难

R2-2 资源分配三路重复
- 位置：L1580-L1671 vs L1539-L1578 vs L1624-L1671
- AllocTransportResource / ReAllocTransportResource / IncreAllocTransportResource 几乎相同的三层for循环嵌套，修一处易忘另外两处

R2-3 两对Notify初始化函数重复
- 位置：L1132-L1184 vs L5409-L5461, L1064-L1109 vs L5286-L5330
- Transport路径和Channel路径各有一套几乎相同的Notify初始化逻辑

R2-4 resMap_用两种不同的锁保护
- 位置：L2001 (std::mutex) vs L1808/L1965/L4993 (PetersonLock)
- 同一数据结构被不同锁保护，存在竞态窗口

R2-5 TLV解析未防御length==0
- 位置：L659-L709, L711-L767
- 循环用commonTlv->length步进，length为0时死循环

##### 中危风险

R2-6 63处reinterpret_cast
- 全文件63处将u64转指针或反向转换，若host侧传入地址已失效则为UB

R2-7 isOpLaunch在错误路径未完全复位
- 位置：L2291-L2296
- isDeviceMode_为true时错误路径不复位isOpLaunch，后续操作可能误判

R2-8 OrchestrateHcclOp职责过重
- 位置：L3300-L3404（105行）
- 同时承担cache查询、算子展开、profiling、counter、notify同步、task下发

R2-9 map拷贝而非引用
- 位置：L1313
- `auto tempLinkRes = isBackup ? linkRdmaResBackUp_ : linkRdmaRes_` 拷贝了整个map

R2-10 errMessageReport_ static无同步保护
- 位置：L62, L4636-L4651
- 多通信域实例共享的static bool无任何同步，并发CQE异常时data race

R2-11 有符号/无符号混用
- 位置：L1850
- commIndex声明为s32但作为map key使用，负值(-1)和u32正值混在同一map

---

#### 2.3 communicator_impl.cc（缺陷修复9次，3789行，约100个函数）

##### 高危风险

R3-1 悬空指针——临时unique_ptr.get()
- 位置：L601
- `slaveStreams[i] = static_cast<rtStream_t>(std::make_unique<Stream>(true).get())` 临时对象在行末析构，slaveStreams[i]立即悬空

R3-2 malloc 100MB未检查返回值
- 位置：L3115
- `hostShareBuf = malloc(SHARE_HBM_MEMORY_SIZE)` (100MB) 无nullptr检查

R3-3 getenv返回nullptr解引用
- 位置：L3025
- `std::string getPath = getenv("ASCEND_HOME_PATH")` 当getenv返回nullptr时为UB。同文件L1297和L1400有正确的null检查，此处遗漏

##### 中危风险

R3-4 5个初始化/恢复路径各自独立
- 位置：L124-159 vs L220-265 vs L267-306 vs L2044-2104 vs L2107-2166
- 15-25个Init子调用的顺序和覆盖范围各不相同，新增初始化步骤极易遗漏某个路径

R3-5 thread_local变量跨实例污染
- 位置：L491-495
- static thread_local变量绑定线程而非CommunicatorImpl实例，同一线程多个实例共享timeout等配置

R3-6 initFlag无并发保护
- 位置：L101-102, L197-198, L222-223
- 普通bool做check-then-set，多线程同时Init存在double-init竞态

R3-7 status状态机缺原子性
- 位置：L639, L652, L676, L678
- status在多函数中读写无同步，弱内存模型架构上busy-wait可能永远看不到更新

R3-8 DPU kernel上下文切换异常安全
- 位置：L3104-3147
- 切换到DPU context后若后续调用失败提前返回，不会切回原context，线程永久停留在DPU context

R3-9 HrtSetDevice等返回值被忽略
- 位置：L126, L225, L228, L423, L2958-2961

R3-10 u64→u32截断用于cache key
- 位置：L438-445
- opParams.count(u64)被static_cast<u32>截断后作为cache key，count超2^32时不同值映射到相同key

R3-11 ExecAlgSelect递归调用无深度检查
- 位置：L2566
- 若缓存状态形成环，无限递归栈溢出

---

### 3. 结构性风险模式归纳

#### 3.1 God Object反模式

三个热点文件都是典型的God Object：
- hccl_communicator_host.cc: 312个方法/9152行
- aicpu_communicator.cc: 213个函数/5550行
- communicator_impl.cc: ~100个函数/3789行

单个类承载初始化、销毁、资源管理、算法调度、状态机、profiling、DFX等多种职责。职责过度集中是缺陷密度高的根本原因。

#### 3.2 代码重复导致"修一漏一"

三个文件都有大段重复代码：
- ExecOp vs ExecOpAlltoAll (hccl_communicator_host.cc)
- AllocTransportResource三路重复 (aicpu_communicator.cc)
- Transport/Channel两套Notify初始化 (aicpu_communicator.cc)
- 5个初始化路径 (communicator_impl.cc)

这与缺陷分析阶段发现的"修一处漏另一处"模式（如4e82ec25修复引入新缺陷被6784944a再次修复）完全一致。

#### 3.3 返回值忽略是系统性问题

三个文件都有忽略关键函数返回值的问题：
- hrtMalloc、Mc2AiCpuStreamAllocAndGet (hccl_communicator_host.cc)
- SetOpExecStatus、InvokeKfcHandler (aicpu_communicator.cc)
- malloc、HrtSetDevice、HrtMemcpy (communicator_impl.cc)

与缺陷分析中e025b6c5（缺return）、9476c6df（CHK_RET调用顺序不当）属同一系列问题。

#### 3.4 并发保护不一致

- 同一数据结构被不同锁保护（aicpu_communicator.cc的resMap_）
- initFlag等状态标志为普通bool而非atomic（communicator_impl.cc）
- 全局原子变量的非原子复合操作（hccl_communicator_host.cc）

#### 3.5 类型安全缺陷仍然广泛存在

- u64→u32截断用于cache key（communicator_impl.cc L438-445）
- s32/u32混用于map key（aicpu_communicator.cc L1850）
- 大量reinterpret_cast（三个文件均有）

与缺陷分析中9c1f957b（u64→u32截断致scratch mem不足）、6baf33c4（u16截断溢出）属同一系列。

---

### 4. 审查建议

#### 针对热点文件的审查策略

1. 任何对这3个文件的修改应要求至少2个reviewer
2. 新增初始化步骤时，须在checklist中列出所有初始化路径并逐一确认
3. 修复缺陷时，必须搜索重复代码段是否有同样问题
4. 所有runtime API调用（hrtMalloc/HrtSetDevice/HrtMemcpy等）的返回值必须检查

#### 长期重构建议

1. 拆分God Object：将资源管理、算法调度、状态机、DFX等职责分离
2. 消除ExecOp/ExecOpAlltoAll的代码重复——提取公共模板
3. 统一资源分配的三路重复为单一参数化函数
4. 统一初始化路径为builder模式，确保所有步骤不遗漏
5. 对resMap_统一使用同一种锁机制

---

### hccl-dev


### 热点文件排名

| 文件 | 缺陷触及次数 | 缺陷commit |
|------|-------------|-----------|
| src/ops/scatter/scatter_op.cc | 4 | 2d8d548a, 13b6032d, 7347fee3, 19d20206 |
| test/.../sim_communicator.cc  | 2 | c11a5289, 8222fcf8 |
| build.sh                      | 2 | 11b7211a, 8222fcf8 |

### 热点1：src/ops/scatter/scatter_op.cc（813行）

4/10的缺陷集中于此文件。当前代码中仍存在的结构性风险：

#### 风险1：错误日志引用错误变量（当前bug）

第356行:
```cpp
aclError aclRet = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, param.stream, &cfg, argsHandle, nullptr);
CHK_PRT_RET(aclRet != ACL_SUCCESS,
    HCCL_ERROR("[LoadCustomKernel][aclrtLaunchKernelWithConfig]errNo[0x%016llx] launch kernel failed", ret),
    HCCL_E_OPEN_FILE_FAILURE);
```

错误信息打印的是`ret`（第328行aclrtKernelArgsInit的返回值），应该是`aclRet`。这意味着当LaunchKernel失败时，日志中记录的错误码是错误的，会误导调试。

#### 风险2：全局状态不安全

第275行:
```cpp
aclrtNotify g_notifies[AICPU_CONTROL_NOTIFY_NUM];
```

全局notify数组在多线程/多算子并发调用时存在竞争风险。如果多个scatter操作同时执行，g_notifies会被覆盖。

#### 风险3：函数内宏定义

第715行:
```cpp
#define ACL_NOTIFY_DEFAULT          0x00000000U
```

在函数体内定义宏，宏的作用域不受函数限制，可能导致命名冲突。应改为constexpr常量。

#### 风险4：拼写错误（"faled"）

第307、610、623行均出现`"faled"`拼写错误（应为`"failed"`），说明这些代码段是复制粘贴产生，未经仔细审查。

#### 风险5：CommEngine枚举命名变迁的残留不一致

文件中混用了多代API命名：
- 第250行使用`COMM_ENGINE_CPU_TS`
- 第376行使用`COMM_ENGINE_CPU_TS`
- 但历史上同一语义曾用`COMM_ENGINE_HOSTCPU_TS`

枚举经历了多次重命名（AICPU→AICPU_TS, CommXxx→HcclXxx），每次重命名都是缺陷源。

#### 风险6：AICPU launch代码块过长且内联

第301-362行的AICPU kernel launch逻辑（~60行）直接内联在ExecOp函数中，混合了kernel参数准备、notify管理、错误处理，难以独立审查和测试。

### 热点2：test/.../sim_communicator.cc

2次缺陷均因主代码API变更（结构体字段增删）未同步到测试代码。测试代码作为API的消费者，是API变更传播不完整的典型受害者。

### 热点3：build.sh

2次缺陷：一次是复制粘贴引入的变量名错误，另一次是同一错误的修复前提。构建脚本中的函数复制是高风险操作。

---

### hcomm-dev


### 一、热点文件统计

数据来源: defect_analysis.md，共157条缺陷条目（含5条无涉及文件记录），涉及239个不同文件，总触及370次。

#### 1.1 Top 20 缺陷磁铁文件

| 排名 | 触及次数 | 文件路径 |
|------|----------|----------|
| 1  | 14 | src/framework/communicator/impl/hccl_communicator_host.cc |
| 2  | 14 | src/framework/hcom/hcom.cc |
| 3  | 12 | src/framework/device/framework/aicpu_communicator.cc |
| 4  | 11 | src/framework/op_base/src/op_base.cc |
| 5  | 7  | src/platform/common/adapter/adapter_rts.cc |
| 6  | 6  | pkg_inc/hccl/hcom.h |
| 7  | 6  | src/CMakeLists.txt |
| 8  | 5  | src/framework/communicator/impl/independent_op/hccl_independent_rank_graph.cc |
| 9  | 5  | src/algorithm/impl/operator/coll_alg_operator.cc |
| 10 | 5  | src/framework/CMakeLists.txt |
| 11 | 5  | src/framework/communicator/impl/hccl_communicator.cc |
| 12 | 4  | src/framework/communicator/impl/independent_op/hccl_independent_op_mem.cc |
| 13 | 4  | src/platform/comm_primitive/hccl_dispatcher_ctx.cc |
| 14 | 4  | src/algorithm/impl/operator/alltoall_operator.cc |
| 15 | 3  | src/framework/communicator/impl/independent_op/hccl_independent_op_channel.cc |
| 16 | 3  | src/framework/inc/hccl_comm_pub.h |
| 17 | 3  | src/platform/hccp/rdma_service/rs.c |
| 18 | 3  | src/platform/task/dispatcher_aicpu.cc |
| 19 | 3  | src/framework/hcom/hcom_common.cc |
| 20 | 3  | (多个文件并列3次) |

- 频次 >= 2 的文件: 52个
- 频次 = 1 的文件: 187个
- 头部集中度: Top 4 文件合计被51次缺陷修复触及，占总触及次数的14%

#### 1.2 模块维度聚合

| 触及次数 | 文件数 | 模块 | 平均触及/文件 |
|----------|--------|------|--------------|
| 143 | 64 | src/framework    | 2.23 |
| 92  | 69 | src/platform     | 1.33 |
| 36  | 26 | src/algorithm    | 1.38 |
| 14  | 13 | test/ut          | 1.08 |
| 13  | 11 | cmake            | 1.18 |
| 11  | 10 | src/orion        | 1.10 |
| 8   | 3  | pkg_inc          | 2.67 |
| 4   | 4  | src/pub_inc      | 1.00 |

framework模块的缺陷密度(2.23)远高于其他模块，是系统中最大的缺陷聚集区。

#### 1.3 framework子模块细分

| 触及次数 | 文件数 | 子模块 | 平均触及/文件 |
|----------|--------|--------|--------------|
| 53 | 20 | src/framework/communicator          | 2.65 |
| 23 | 11 | src/framework/device                | 2.09 |
| 20 | 4  | src/framework/hcom                  | 5.00 |
| 14 | 4  | src/framework/op_base               | 3.50 |
| 12 | 12 | src/framework/next                  | 1.00 |
| 7  | 6  | src/framework/cluster_maintenance   | 1.17 |
| 6  | 5  | src/framework/common                | 1.20 |

hcom子模块只有4个文件但平均触及5.0次/文件，是密度最高的缺陷聚集点。

#### 1.4 platform子模块细分

| 触及次数 | 文件数 | 子模块 | 平均触及/文件 |
|----------|--------|--------|--------------|
| 31 | 26 | src/platform/hccp           | 1.19 |
| 17 | 16 | src/platform/resource       | 1.06 |
| 16 | 7  | src/platform/common         | 2.29 |
| 11 | 10 | src/platform/legacy         | 1.10 |
| 7  | 3  | src/platform/comm_primitive | 2.33 |
| 6  | 4  | src/platform/task           | 1.50 |

platform的缺陷较为分散（hccp 31次/26文件 = 1.19），而common/adapter和comm_primitive密度偏高。

---

### 二、热点文件结构性风险审查

#### 2.1 hccl_communicator_host.cc (8818行, 14次缺陷修复)

主要职责: HCCL通信域的host侧实现，负责集合通信算子的参数校验、资源分配、算法选择与执行编排。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | 错误码吞没 | AicpuUnfold/AllReduceAicpuUnfold中Mc2AiCpuStreamAllocAndGet返回值被后续CreateCommResource覆盖，stream分配失败被静默忽略 | 2695, 2953 |
| 2 | 资源泄漏 | HcclGetAlgExecParam中hrtMalloc分配的device内存在CHK_RET失败提前返回时无法释放 | 1742-1812 |
| 3 | 函数复杂度 | ExecOp(230+行)和ExecOpAlltoAll(280+行)承担算法选择/资源创建/AIV处理/AICPU编排/DFX注册等多重职责 | 4280-4513, 4525-4800 |
| 4 | 错误处理 | SwitchNic中清理操作的CHK_RET可能丢弃原始错误码 | 8297-8302 |
| 5 | 并发安全 | SnapshotCheckPreProcess/PostProcess中busy-wait循环无sleep/yield，100%占用CPU | 8724-8806 |
| 6 | 死代码 | BatchSendRecv中IsAtomicInit()重复检查，第二次永远不触发（复制粘贴遗留） | 3864, 3878 |
| 7 | 并发安全 | g_enableBackupLinkCommCount的read-then-decrement非原子操作，并发析构可能underflow | 212-217 |
| 8 | 空指针 | implAlg_在AllGather/AllGatherOutPlace/AllReduceOutPlace等多个公共API中未做null检查就调用 | 2576, 2741, 3017等 |
| 9 | 资源管理 | 构造函数中mrManager_/zeroCopyAclGraph_的new(std::nothrow)分配失败仅记日志，对象带nullptr存活 | 124-132, 157-165 |
| 10 | 函数复杂度 | 析构函数超过140行，资源释放顺序依赖隐式假设 | 169-316 |
| 11 | 错误处理 | SwitchNic在ParseSwitchRanks失败后仍向设备发送可能包含垃圾值的changeLinkInfo | 8236-8303 |
| 12 | 并发安全 | Init函数的g_hcomInitMutex仅覆盖部分初始化操作，子通信域Init完全不使用该锁 | 344-405 |

#### 2.2 hcom.cc (4068行, 14次缺陷修复)

主要职责: Hcom层入口，封装AllReduce/AllGather/Broadcast/Send/Receive等集合通信操作及通信域初始化、Group管理。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | 并发安全(TOCTOU) | HcomCreateGroupImplHeterog: 持锁检查group不存在→解锁→CreateGroup→再加锁插入map，两个线程可同时通过检查导致重复创建 | 1488-1514 |
| 2 | 并发安全 | HcomReleaseSubComms: 直接遍历hcomInfo.hcomGroupMap而未持groupParamsLock，并发CreateGroup/DestroyGroup会导致迭代器失效 | 2089-2104 |
| 3 | 悬垂指针(确定性UAF) | HcomGetAlgorithm: `*algo = const_cast<char *>(str.c_str())`将局部变量str的内部指针返回给调用者，函数返回后即悬垂 | 2076 |
| 4 | 悬垂指针 | GetGroupNameByOpBaseHcom: `*groupname = const_cast<char *>(hcclComm->GetIdentifier().c_str())`若返回临时对象则立即悬垂 | 2800-2801 |
| 5 | 空指针(23处) | HcclCommGraph系列函数中reinterpret_cast<hcclComm*>(opBaseHcom)后大多不做null检查，仅2处有检查。用户传入0/无效句柄则crash | 888-2776(23处) |
| 6 | 内存泄漏 | HcomGetandClearOverFlowTasks: malloc后memcpy_s失败直接返回错误码，已分配内存未free | 2485-2494 |
| 7 | 内存泄漏 | HcomGetSplitStrategy: `*segmentIdxPtr = new u32[*len]`裸new交由调用者管理，无释放责任文档 | 1657 |
| 8 | 数据竞争 | g_rankTableSetInfo: 文件级static全局变量，Set/Get操作无任何锁保护，RankTable_t赋值非原子 | 2405 |
| 9 | 初始化不对称 | HcomInit: 部分backlogged groups已创建成功但HcomDestroy可能无法正确处理部分初始化状态 | 67-162 |
| 10 | 日志敏感信息 | HcomSetRankTableImpl和HcomInitByMasterInfo将完整rankTable/IP/端口以RUN_INFO级别明文打印 | 2408, 316-317 |

#### 2.3 aicpu_communicator.cc (5809行, 12次缺陷修复)

主要职责: AICPU侧通信域管理器，负责初始化/销毁通信资源、编排集合通信算子执行、管理链路重试状态机和设备stream生命周期。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | 数据错乱(确定性Bug) | UpdateSqStatus中SQ_TAIL查询结果写入head、SQ_HEAD查询结果写入tail，head/tail赋值反转导致重执行路径stream状态恢复错误 | 3385-3386 |
| 2 | 并发安全 | errMessageReport_作为static bool被多实例跨线程无锁读写（C++标准层面是UB） | 61, 2989, 4925, 4940 |
| 3 | 并发安全 | dfxExtendInfo_在背景线程HandleCqeException中写入、主线程CheckOpExecStatus中读取，无同步保护 | 4889-4892写, 4041读 |
| 4 | 并发安全 | isOpLaunch/endStopLaunch/needsResponseStopLaunch_/printTaskExceptionForErr_等bool标志无锁跨线程读写 | 2586, 2653, 2986-2988 |
| 5 | 并发安全 | GetAlgResponseRes中resMap_的double-check locking: 锁外读取resMap_时其他线程可能正在rehash | 2297-2318 |
| 6 | 资源管理 | 析构函数清理了dispatcher_但未考虑Transport对象析构对dispatcher_的依赖（析构顺序问题） | 78-106 |
| 7 | 资源管理 | opUnfoldCachePtr_使用裸new/delete而非RAII，Init()中间步骤失败可能导致泄漏 | 100-103, 160-161 |
| 8 | 函数复杂度 | Orchestrate状态机约190行，12-case switch内局部变量跨case隐式共享 | 2407-2594 |
| 9 | 错误处理 | ReportErrCqe中QuerySqStatusByType返回值被忽略，失败时head/tail保持初始值0导致后续越界 | 4904-4905 |
| 10 | 类型安全 | HcclOpSupportRetry返回bool但内部CHK_RET会返回HcclResult，非零错误码被隐式转为true（"支持重试"），语义完全相反 | 3230 |

#### 2.4 op_base.cc (4587行, 11次缺陷修复)

主要职责: HCCL通信域的初始化/销毁及各类集合通信算子的单算子模式入口实现。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | 并发安全 | g_oneSidedCommHcomInfos和g_oneSidedCommSet全局变量无锁访问（IsOneSidedComm/InitOneSidedHcomInfo/DeInitOneSidedHcomInfo等多处） | 267-334, 717, 2832 |
| 2 | 并发安全 | opGroup2CommMap的find/insert操作部分路径未持锁（CheckOpBasedHcom/HcclCreateSubCommConfigInner/InitCommRootInfo/HcclSetConfig） | 343, 900, 1353, 1829 |
| 3 | 并发安全 | HcclSendInner/HcclRecvInner/HcclAlltoAllInner/HcclAlltoAllVInner/HcclAlltoAllVCInner/HcclReduceInner缺少operatorlock_ | 2667-2794, 3247-3582 |
| 4 | 死代码 | HcclCommInitClusterInfoWrapper: return HCCL_SUCCESS后面还有return ret，永远不执行 | 582-584 |
| 5 | 状态泄漏 | HcclReduceInner/HcclAlltoAllInner/HcclAlltoAllVInner/HcclAlltoAllVCInner缺少HcclResetIfProfile()恢复调用，profiling开关泄漏 | 3287, 3409, 3527, 3634 |
| 6 | 代码重复 | InitCommRootInfo(212行)与InitCommClusterInfo(136行)的配置序列几乎完全重复，HcclCreateSubCommConfigInner亦然，三处拷贝 | 1344-1555, 395-530, 888-1025 |
| 7 | 并发安全 | GetHcclExistDeviceOpInfoCtx锁内修改thread_local的g_hcclDeviceId，后续无锁路径可能访问错误设备上下文 | 116-117 |
| 8 | 资源泄漏 | HcclOneSidedCommDestroy: opGroup2CommMap.find失败时提前return，isUsed标记永远不被清除，槽位无法复用 | 2843 |
| 9 | 悬垂引用 | HcclGetCommAll: 线程创建失败时已启动的线程未join，这些线程持有comms[]的引用，主线程return后引用失效 | 188-193 |
| 10 | 并发安全 | HcclCommDestroy和HcclCommDestroyWrapper对IsOneSidedComm的锁保护策略不一致（前者持锁后者无锁） | 3010-3014 vs 2905-2907 |

#### 2.5 adapter_rts.cc (2487行, 7次缺陷修复)

主要职责: 封装ACL/Runtime底层API，为HCCL提供跨平台适配层。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | 死代码(特性失效) | REPLACE_NOTIFY_WITH_EVENT宏中result硬编码为0，if(result!=0)永远不执行，event替换功能完全失效 | 37-45 |
| 2 | 并发安全 | g_localDeviceType/g_localDeviceLogicId/g_deviceSatMode等普通全局变量多线程无锁读写 | 94-96, 207, 449, 458 |
| 3 | 资源管理 | hrtDevMemAlignWithPage: hrtGetPointAttr失败且hrtFreeHost也失败时ptrAttr泄漏 | 1148-1196 |
| 4 | 函数复杂度 | hrtMalloc: 多层嵌套的设备类型判断+level2Address分支，76行，设备类型分支组合易遗漏 | 558-634 |

#### 2.6 hccl_communicator.cc (3110行, 5次缺陷修复)

主要职责: HCCL通信域核心实现，包含初始化、通信操作执行、NS恢复(Suspend/StopExec/Clean)等全生命周期管理。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | 并发安全(竞态) | DestroyOpTransportResponse: 持锁erase后释放锁，在无锁状态下逐个DeInit()，其他线程可能同时操作同一transport对象 | 634-678 |
| 2 | 初始化顺序缺陷 | InitOpResPara: memset_s清零后使用opResDeviceParaPtr_初始化链表，但该指针在下面CreateWorkSpace才赋值 | 123-139 |
| 3 | 代码重复 | Suspend/StopExec/Clean三个函数(约190行)包含几乎完全相同的while(true)轮询循环和超时处理 | 1166-1353 |
| 4 | 死代码 | DestroyAicpuComm: while(true)循环后的return HCCL_SUCCESS不可达 | 881 |
| 5 | 并发安全 | isSuspending成员变量在多个方法中设置为true，无atomic声明或锁保护 | 1176, 1236, 1299 |

#### 2.7 hccl_independent_rank_graph.cc (490行, 5次缺陷修复)

主要职责: RankGraph查询的公共API入口，支持V1/V2双版本分发。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | 错误处理 | HcclGetTopoInstsByLayer/HcclGetTopoType/HcclGetRanksByTopoInst: HCCLV2_FUNC_RUN后直接return HCCL_SUCCESS，V1路径永远不执行 | 251-323 |
| 2 | 日志格式错误 | 日志字符串"netLayer%u]"缺少左方括号 | 70, 80 |
| 3 | 状态一致性 | 每次调用都getenv("HCCL_INDEPENDENT_OP")而非缓存，运行时修改环境变量导致行为不一致 | 60, 102等11处 |

#### 2.8 coll_alg_operator.cc (1081行, 5次缺陷修复)

主要职责: 集合通信算法算子基类，负责算法选择、executor管理、资源请求计算和执行编排。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | 死代码 | CalNumBlocks: else分支后有永远不可达的return HCCL_SUCCESS | 97-100 |
| 2 | 状态管理 | executor_惰性初始化+SelectAlgFor91093WithCoreLimit中置nullptr重建，模式脆弱 | 67, 78, 140, 215 |
| 3 | 函数复杂度 | GetDefaultAlgoLevel1V2: 多个跨4行的复合布尔条件，可读性差 | 471-526 |

#### 2.9 alltoall_operator.cc (712行, 4次缺陷修复)

主要职责: AlltoAll系列集合通信算子的算法选择、参数预处理和执行编排。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | 空指针 | SetExecutorAttr/SetExcutorExtraInfo/CheckNeedRecreateComm: dynamic_cast结果未检查nullptr | 454, 458, 491 |
| 2 | 条件方向错误 | IsSatisfyAlltoAllAivCondition: CHK_PRT_RET(userRankSize_ > 1, ...)的条件方向与上下文逻辑矛盾 | 549 |
| 3 | 陈旧数据 | allMeshAggregationSendRecvInfo_在SelectAlg提前return时不会更新，后续SetExcutorExtraInfo使用上次遗留数据 | 243-349 |

#### 2.10 hccl_independent_op_mem.cc (142行, 4次缺陷修复)

主要职责: 独立算子模式下的内存注册/注销和CCL Buffer获取。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | 错误吞没 | HcclCommDeregMem: HCCL_E_NOT_FOUND时返回HCCL_SUCCESS，掩盖内存泄漏调试线索 | 90-91 |
| 2 | 控制流依赖宏 | HcclGetHcclBuffer: V1/V2路径选择依赖HCCLV2_FUNC_RUN宏内部实现，可读性差 | 105-142 |

#### 2.11 hccl_dispatcher_ctx.cc (231行, 4次缺陷修复)

主要职责: DispatcherCtx生命周期管理，维护commId与DispatcherCtx的全局映射。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | 资源泄漏 | AcquireDispatcherCtx: 创建新DispatcherCtx但不注册到g_ctx全局映射，无法被其他线程找到或按commId清理 | 205-231 |
| 2 | 并发安全(TOCTOU) | DestroyDispatcherCtx: deleteMutex_和g_mtx的嵌套获取+保护粒度不一致 | 127-161 |
| 3 | 悬垂指针 | DestroyDispatcherCtx: ctx=nullptr只修改局部参数副本，调用者持有的指针不会被置空 | 159 |

#### 2.12 pkg_inc/hccl/hcom.h (387行, 6次缺陷修复)

主要职责: HCOM公共API头文件，定义集合通信对外接口和数据结构。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | ABI兼容性 | extern "C"块内使用namespace(hccl::HcclDumpInfo)，C编译器无法处理 | 28-39 |
| 2 | ABI兼容性 | 公共API函数签名包含C++默认参数，在C linkage中无意义且影响ABI稳定性 | 261, 275, 284, 317 |
| 3 | 符号膨胀 | 头文件中定义非inline的const std::string/std::map全局变量，每个翻译单元独立副本+静态初始化顺序问题 | 112-141 |
| 4 | 内部实现泄露 | 直接include了workflow.h/dtype_common.h/hccl_rank_graph.h等内部头文件 | 20-22 |
| 5 | 类型安全 | HcclRtStream和rtStream_t均为void*，编译器无法区分误用 | 143-144 |

#### 2.13 src/CMakeLists.txt (697行, 6次缺陷修复)

主要职责: 顶层构建配置。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | 路径硬编码 | HCCL_THIRD_PARTY_DIR使用7层../回溯 | 31 |
| 2 | 代码重复 | ccl_kernel_plf和ccl_kernel_plf_a的include/compile/link配置几乎完全重复 | 306-376, 493-573 |
| 3 | 编译选项冲突 | ccl_kernel_plf_a同时指定-O3和-O2，后者覆盖前者 | 515 |
| 4 | 安全选项不一致 | hccd用-fstack-protector-strong，ccl_kernel_plf/ccl_kernel_plf_a用-fstack-protector-all | 243, 502, 517 |
| 5 | 依赖管理 | GLOB_RECURSE动态收集源文件目录作为include路径 | 624-629 |

#### 2.14 src/framework/CMakeLists.txt (997行, 5次缺陷修复)

主要职责: framework层构建配置。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | 路径硬编码 | 8层../回溯，且与顶层CMakeLists同名变量值不同 | 16 |
| 2 | 代码重复 | ccl_kernel的include路径在BUILD_OPEN_PROJECT分支内外被添加两次 | 689-706, 835-846 |
| 3 | 全局副作用 | CMAKE_CXX_FLAGS在BUILD_OPEN_PROJECT分支中被清空，影响后续所有target | 927 |
| 4 | 硬编码构建产物路径 | open分支用硬编码.a路径引用ccl_kernel_plf | 943 |
| 5 | 条件分支不对称 | KERNEL_MODE=OFF时ccl_kernel target不存在但后续分支仍对其配置 | 267, 785 |

#### 2.15 hccl_comm_pub.h (454行, 3次缺陷修复)

主要职责: hcclComm类定义，集合通信域的核心内部接口。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | 封装泄露 | planner/barrierSendBuf/barrierRecvBuf/operatorlock_作为public成员暴露 | 338-341 |
| 2 | 条件编译导致内存布局变化 | 不同编译配置下hcclComm类的成员集合不同(CCL_KERNEL_AICPU/HCCD宏)，跨模块传递指针可能越界 | 366-375, 441-448 |
| 3 | 类职责过重 | hcclComm类承载约80个public方法，单类承担初始化/通信/资源/group/拓扑/profiling等全部功能 | 全文件 |

#### 2.16 rs.c (2558行, 3次缺陷修复)

主要职责: RDMA服务核心实现，管理RDMA设备初始化、QP创建、连接管理。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | UAF | RsInit错误路径free(rscb)后gRsCb(thread-local)仍指向已释放内存 | 554-622 |
| 2 | 条件编译爆炸 | CONFIG_TLV/CONFIG_CONTEXT/CUSTOM_INTERFACE三层宏组合，每个变体初始化/反初始化路径不同 | 38-521 |
| 3 | 不可靠退出 | RsDeinitRscbCfg用usleep+tryAgain忙等待线程退出，超时后仅warning继续执行 | 529-549 |

#### 2.17 dispatcher_aicpu.cc (1265行, 3次缺陷修复)

主要职责: AICPU任务分发器，负责SQE编排、下发和RTSQ管理。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | 函数复杂度 | LaunchTask(216行)和LaunchNewTask(145行)包含多层嵌套的ring buffer索引计算 | 354-718 |
| 2 | 活锁风险 | LaunchTask忙等待RTSQ空间时遍历其他stream调LaunchTask(false)，互相等待可能活锁 | 529-567 |
| 3 | 代码重复 | LaunchTask和WaitRtsq+MemcpyRtsq的RTSQ等待/超时/拷贝逻辑大量重复 | 502-718 vs 902-1145 |
| 4 | 错误码误用 | 多处用HCCL_E_PTR表示"超出范围"错误，应为HCCL_E_PARA | 516, 913, 984 |

#### 2.18 hcom_common.cc (1613行, 3次缺陷修复)

主要职责: HCOM通用功能实现，包括初始化、group管理、集合通信操作的入口封装。

| # | 风险类别 | 描述 | 行号 |
|---|---------|------|------|
| 1 | 并发安全(TOCTOU) | HcomGetCurHcomCtx: 加锁查找+返回引用后锁即释放，引用在锁外使用但内容可能被并发修改 | 119-148 |
| 2 | 无限阻塞 | HcomDestroyGroupImpl: while(ref!=0)忙等待轮询group引用计数，无超时保护 | 533-538 |
| 3 | 全局状态过多 | 文件中定义了10+个全局/static变量，锁获取顺序无明确约定 | 74-117 |
| 4 | 日志错误 | HcomCreateCommCCLbuffer: 错误报告中函数名写成了"HcomGetDevType" | 1468-1481 |

---

### 三、跨文件系统性风险总结

#### 3.1 并发安全是最大的系统性风险

在审查的18个热点文件中，15个存在并发安全问题。模式包括：

- TOCTOU竞态: hcom.cc(HcomCreateGroupImplHeterog), hcom_common.cc(HcomGetCurHcomCtx), hccl_dispatcher_ctx.cc(DestroyDispatcherCtx)
- 无锁访问共享状态: op_base.cc(g_oneSidedCommHcomInfos), aicpu_communicator.cc(errMessageReport_), adapter_rts.cc(g_localDeviceType), hcom.cc(g_rankTableSetInfo)
- 锁范围不足/不一致: op_base.cc(operatorlock_缺失), hccl_communicator_host.cc(Init锁范围), hccl_communicator.cc(DestroyOpTransportResponse)
- 忙等待无超时/无yield: hccl_communicator_host.cc(SnapshotCheck), hcom_common.cc(HcomDestroyGroupImpl), rs.c(RsDeinitRscbCfg)

根因: 通信框架天然是多线程环境（多device、多通信域、主线程+背景轮询线程），但锁的使用缺乏统一的层级设计和文档化约定。

#### 3.2 错误处理链断裂

- 返回值覆盖: hccl_communicator_host.cc中stream分配返回值被后续调用覆盖
- 错误码类型冲突: aicpu_communicator.cc中HcclResult被隐式转bool，语义反转
- 错误路径资源泄漏: hcom.cc(malloc后memcpy_s失败未free), hccl_communicator_host.cc(hrtMalloc失败路径), op_base.cc(isUsed标记未清除)
- 死代码掩盖逻辑错误: op_base.cc(return后的return), hccl_communicator.cc(while后的return)

#### 3.3 God Class与代码重复

- hcclComm(80+方法)、hccl_communicator_host.cc(8818行)、aicpu_communicator.cc(5809行)、op_base.cc(4587行)、hcom.cc(4068行) 均为超大文件
- 三处InitCommConfig序列重复(op_base.cc)，两处RTSQ下发逻辑重复(dispatcher_aicpu.cc)，两处CMake target配置重复
- 代码重复导致"改一漏一"是hcomm-dev缺陷的高频来源

#### 3.4 公共API/ABI设计缺陷

- hcom.h混用C/C++ linkage、暴露内部头、在头文件定义std容器全局变量
- hccl_comm_pub.h暴露public成员变量、条件编译改变类内存布局
- 多个API返回裸指针(悬垂指针: hcom.cc两处, 资源所有权不清: hcom.cc HcomGetSplitStrategy)

#### 3.5 构建系统脆弱性

- 7-8层相对路径回溯
- GLOB_RECURSE动态收集include路径
- 编译选项/安全选项在不同target间不一致
- BUILD_OPEN_PROJECT/KERNEL_MODE条件分支不完整

---

### 四、缺陷热点分布可视化

```
src/framework/                          ████████████████████████████████████████ 143次
  communicator/                         ██████████████████████████ 53次
    impl/hccl_communicator_host.cc      ███████ 14次  ← 最高频
    impl/hccl_communicator.cc           ██ 5次
    impl/independent_op/                █████ 11次
  hcom/                                 ██████████ 20次
    hcom.cc                             ███████ 14次  ← 并列最高频
    hcom_common.cc                      █ 3次
  device/framework/                     ██████████ 23次
    aicpu_communicator.cc               ██████ 12次
  op_base/src/                          ███████ 14次
    op_base.cc                          █████ 11次

src/platform/                           ██████████████████████████ 92次
  hccp/                                 █████████ 31次(分散)
  common/adapter/                       ███ 7次
  comm_primitive/                       ██ 4次
  task/                                 ██ 3次

src/algorithm/                          ██████████ 36次
  impl/operator/                        █████████ 9次

cmake + CMakeLists                      █████ 13+6+5=24次
pkg_inc/hccl/                           ██ 8次
```

---

## Part 2: Revert事件分析

数据来源: hccl(2个Revert) + hccl-dev(2个Revert) + hcomm-dev(4个Revert), 共8个Revert事件

---

### hccl (hcomm仓库)

#### 概述

hcomm仓库428次提交中仅有2次revert（0.47%），但每一次revert都代表缺陷逃逸到主干后被紧急撤回，是高价值的流程缺陷信号。

| 编号 | Revert Commit | 原始 Commit | 合入到Revert间隔 | 影响范围 |
|------|--------------|-------------|-----------------|---------|
| R1 | 72cdf80e | fb56d64b | ~6小时 | 2文件，硬件常量+UT |
| R2 | 753ba8c2 | 05b38411 | ~1小时 | 24文件，构建系统核心 |

---

#### R1: 72cdf80e Revert "fix loopnum = 128"

##### 原始提交分析 (fb56d64b)

时间线：
- 原始合入：2026-02-11 11:37
- Revert：2026-02-11 17:39
- 间隔：约6小时

变更内容：
- `ccu_microcode.h`: `CCU_MS_DEFAULT_LOOP_COUNT` 从 64 改为 128
- `ut_ccu_dfx.cpp`: 大量UT期望值从 `size() == 3` 改为 `size() == 2`，同时混入代码风格变更（花括号位置、空格格式化）

##### 缺陷根因

1. 无依据的常量修改：commit message仅写"fix loopnum = 128"，PR描述完全为空，没有任何设计文档或硬件spec引用说明为何128是正确值
2. UT被修改为适配变更而非独立验证：原UT期望profiling info返回3条记录，修改后改为2条——这意味着修改者同时改了被测代码和测试，让测试"通过"而非让测试"验证"
3. 逻辑变更与代码风格混合：同一commit中既修改了CCU loop count常量，又重新格式化了大量UT代码的花括号和空格，增加了review难度

##### 逃逸原因分析

- pre-merge CI通过（UT已被同步修改适配）
- PR描述为空模板，reviewer无法判断变更合理性
- 硬件相关常量缺乏取值依据文档化
- 逻辑改动与风格改动混合，干扰了reviewer对核心变更的注意力

##### 审查教训

- 硬件相关常量修改必须在commit message或PR中注明取值依据（datasheet/spec引用）
- UT期望值大面积同步修改是强烈的审查红旗——测试应独立于被测代码
- 逻辑变更不应与代码风格变更混在同一commit

---

#### R2: 753ba8c2 Revert "[Build] Add support for offline compile [2/2]"

##### 原始提交分析 (05b38411)

时间线：
- Part [1/2] (b3c0ad3e) 合入：2026-02-12 16:18（14天前）
- Part [2/2] (05b38411) 合入：2026-02-26 22:22
- Revert：2026-02-26 23:21
- 间隔：约1小时

变更内容：
- 24个文件，640行新增/499行删除
- 核心变更：将 `cmake/utils.cmake`（308行）替换为 `cmake/hcomm_utils.cmake`（347行），引入远程下载hcomm-utils预编译包的逻辑
- 涉及多个third_party cmake文件的依赖获取方式重构
- openssl编译逻辑大幅修改
- 文档(docs/build.md)也有配套修改

##### 缺陷根因

1. 大规模构建变更未经充分验证：24个文件的构建系统重构，合入后1小时即revert，极可能连基本编译都未通过
2. 分步合入策略失效：[1/2]在14天前合入，[2/2]在后续合入，但两者之间的时间差可能导致代码基已经演进，[2/2]与当前代码不兼容
3. PR描述为空模板：与R1相同的问题

##### 逃逸原因分析

- 构建系统变更缺乏独立的CI验证pipeline（可能只做了编译不做链接/运行）
- 单次合入影响过大（24个文件），难以在review中全面覆盖
- 夜间合入（22:22）可能意味着review不够充分
- 两部分提交间隔14天，期间代码演进可能导致不兼容

##### 审查教训

- 构建系统重大变更应有独立的staging验证，不能仅依赖常规CI
- 单次PR不应同时修改20+个构建文件，应分步骤合入以便定位问题
- 分步提交（[1/2], [2/2]）间隔不宜过长，否则[2/2]可能与已演进的代码不兼容
- 晚间合入大变更应有更严格的审查门槛

---

#### 共性模式归纳

模式1: PR描述为空

两次被revert的提交的PR描述均为空模板（仅保留了模板占位符，未填写任何内容）。这直接降低了reviewer对变更合理性的判断能力。

审查规则：PR描述为空的变更不应合入主干。至少应包含：变更原因、影响范围、测试方法。

模式2: 测试未能阻止缺陷

- R1: UT被修改为适配错误变更，失去了独立验证功能
- R2: 构建系统变更缺乏端到端验证（编译+链接+运行）

审查规则：当PR同时修改被测代码和测试代码时，reviewer应格外关注测试修改的合理性。构建系统变更必须有完整的端到端CI验证。

模式3: 变更范围与审查强度不匹配

- R1: 虽然只改2个文件，但涉及硬件常量语义，影响全部使用该常量的算法路径
- R2: 24个文件的构建系统重构，影响所有模块的编译链接

审查规则：影响范围大的变更应有对应强度的审查和测试覆盖。硬件常量修改虽然行数少但影响面广，应按"高影响"级别审查。

模式4: 快速Revert（<24小时）

两次revert都在合入后数小时内发生，说明问题在合入后立即被发现。这意味着：
- 问题本可以在pre-merge阶段被CI捕获
- 缺少有效的pre-merge门禁机制

审查规则：高影响变更（构建系统、硬件常量、核心算法）应有更严格的pre-merge检查。

模式5: 变更类型混合

R1将逻辑变更（常量值修改）和风格变更（花括号格式化）混在同一commit中。

审查规则：一个commit只做一件事。逻辑变更和风格变更必须分离。

---

#### Revert信号在Code Review中的应用

高风险信号清单（参考本仓库revert历史）

1. PR描述为空或仅保留模板 → 强制要求补充
2. 同时修改源码和对应UT的期望值 → 审查UT修改是否合理
3. 硬件常量修改无spec引用 → 要求补充依据
4. 构建系统大范围修改（>5个文件） → 要求分步合入+staging验证
5. 分步提交的后续部分与前序间隔超过1周 → 审查与代码演进的兼容性
6. 逻辑变更夹带风格变更 → 要求拆分commit

量化指标

- 仓库revert率：2/428 = 0.47%
- 原始提交到revert的平均间隔：3.5小时
- 两次revert的PR描述均为空（100%）
- 两次revert均在合入当天被执行（100% <24小时）

---

### hccl-dev

#### 概览

共发现4个revert相关提交，涉及2个独立事件。

#### 事件1：打包目录结构反复变更（share/info flip-flop）

时间线：
1. MR !18 (b6aec59, 2025-11-26)：将打包路径从`hccl/`改为`share/info/hccl/`，新增`--use-share-info`命令行参数，影响18个文件
2. MR !20 (d67b6a5/52bda68, 2025-11-27)：24小时内revert MR !18，回退到`hccl/`路径
3. MR !22 (9e7ea4f, 2025-11-27)：同日再次revert MR !20（即恢复MR !18的修改），重新使用`share/info/hccl/`路径

分析：
- 2天内对同一个18文件的大规模修改执行了3次（加-撤-再加），说明打包目录结构的设计决策在合入前未充分确认
- 每次变更涉及cmake、shell脚本、xml配置等多种文件类型，反复修改增加了遗漏风险
- 该事件本身不是代码缺陷，但反映了架构决策确认不足的流程问题

审查启示：
- 大规模路径/目录结构变更应有明确的设计评审记录，避免合入后反悔
- 涉及多种配置文件的变更应有完整的回归测试

#### 事件2：API命名批量重命名的premature merge

时间线：
1. MR !34 (0148fc4, 2025-11-29)：API函数批量重命名，影响6个文件
   - `CommGetEngineCtx` → `HcclGetEngineCtx`
   - `CommCreateEngineCtx` → `HcclCreateEngineCtx`
   - `CommAllocThreadResByStream` → `HcclAllocThreadResByStream`
   - `CommChannelCreate` → `HcclChannelCreate`
   - `CommGetRankGraph` → `HcclGetRankGraph`
   - `CommLocalBareNotifyRecord` → `HcommInterOpNotifyRecordOnThread`
   - `CommLocalBareNotifyWait` → `HcommInterOpNotifyWaitOnThread`
   - 还添加了`__attribute__((weak))`前向声明
2. MR !44 (29c59f3, 2025-11-29)：同日revert MR !34

分析：
- API重命名从`Comm`前缀统一到`Hccl`前缀，方向合理（统一命名空间），但实现时机不对
- 被revert可能原因：(1)下游依赖尚未同步更新；(2)weak symbol方案引入的不确定性；(3)重命名范围未覆盖所有调用点
- 同日合入又revert说明变更未经充分的集成测试验证

审查启示：
- 批量API重命名需制定迁移计划，确认所有调用方已准备好
- 使用`__attribute__((weak))`做过渡的方案需要明确weak symbol的生命周期和回退策略
- 大规模重命名应在独立分支充分测试后再合入

#### 与缺陷提交的关联

事件2的revert与缺陷提交c11a5289、8222fcf8有间接关联：这些编译错误修复涉及HcclCommConfig结构体字段的变更（commEngine/threadNum/notifyNumPerThread的增删），属于同一时期的API迭代。频繁的API变更-revert-再变更增加了不同提交之间的接口不一致风险。

---

### hcomm-dev

共4条Revert提交，涉及3个独立回退事件（Revert#2和Revert#3属于同一runtime接口迁移系列）。

---

#### Revert#1: a37e6cf1 — Revert HB check opinconsistent

| 属性 | 值 |
|------|-----|
| Revert hash | a37e6cf15c75b6db581a07f433eeea45bf3c21f7 |
| 日期 | 2025-12-03 |
| 作者 | dingzhiqiang |
| 原始提交 | b51f8cc4（2025-11-18, yanyefeng, "update"） |
| 涉及文件 | 7个（heartbeat.cc/h, hccl_communicator_host.cc等） |

##### 原始变更内容

为心跳模块引入算子不一致性检测(Op Inconsistency Check)功能：

1. 重构心跳帧数据结构：从`OpInfoDesc opInfoList[16]`扁平数组改为按tag分组的两层嵌套结构`OpInfoTagQueueFrame`（内含`OpInfoTagQueue[10]`，每个tag下可存`OpInfoDesc[500]`）
2. 新增`inconsistentOpMap_`和`srTagMap_`两个map，增加`RegisterSROpIdentifier()`/`AddInconsistentOpRecord()`/`CheckOpInconsistentError()`三个函数
3. 将opInfo注册条件从仅`SEND/RECEIVE`扩大到所有集合通信算子
4. 发送逻辑重构：引入`opInfoQueueForSend_`发送缓冲队列，增加`HBFRAME_SEND_LOOP_MAX_NUM=120`的发送重试循环（带100us sleep）
5. 新增`CommCheckOpInconsistentError`公开接口

##### Revert根因

HeartBeatFrame结构体膨胀导致性能和内存灾难：

- 帧大小膨胀约30倍：原版`OpInfoDesc opInfoList[16]`约6-7KB，修改后`OpInfoTagQueue[10] * OpInfoDesc[500]`约200KB
- 心跳每50ms广播一次，发送缓冲区`sendBuffer`（最大3072帧）内存占用从约20MB暴增到约600MB
- 120次重试循环（每次sleep 100us = 最多30ms）阻塞心跳线程，50ms周期内60%时间被sleep占用
- `inconsistentOpMap_`和`srTagMap_`只增不清理，长时间运行存在内存泄漏风险
- 功能直接全量生效，无灰度开关（后续重新引入ebfcde5c加了"with switch"证实此教训）

##### 暴露的缺陷模式

1. 数据结构膨胀无约束：嵌套定长数组`[500]*[10]=5000`个元素嵌入每帧，远超实际需要，设计时未考虑网络I/O中结构体大小的乘法效应
2. 巨型提交掩盖问题：原始功能隐藏在678文件的"update"大批量同步提交中，审查者无法在海量无关变更中发现帧膨胀30倍
3. 阻塞式重试设计：在周期性心跳线程中引入sleep重试循环，违反非阻塞心跳的设计原则
4. 缺少feature flag：影响跨节点协议的功能变更直接全量生效，无灰度开关

##### 审查教训

- 改变跨节点传输帧格式的提交，必须单独提交并附带帧大小分析（before/after sizeof）
- 包含嵌套定长数组的结构体出现在网络传输路径上时，必须有sizeof的审查检查项
- 大批量同步提交不应包含功能变更——同步应是纯机械操作，功能修改必须独立提交
- 在周期性线程中引入sleep/重试时，必须评估对周期的影响（30ms vs 50ms = 60%占用）
- 影响跨节点协议的功能上线应具备灰度能力（环境变量开关）

---

#### Revert#2: 30e25e50 — Revert Runtime interfaces

| 属性 | 值 |
|------|-----|
| Revert hash | 30e25e509b127c077358877259046992f7111d34 |
| 日期 | 2025-12-16 |
| 作者 | jiyuanhao |
| 原始变更 | 来自d38afa51等多个commit的runtime接口替换 |
| 涉及文件 | 9个（adapter_rts.cc/h, stream_utils.cc, op_base_host.cc等） |

##### 原始变更内容

将ACL高层接口替换为RT层直接接口：

1. `reinterpret_cast<uint64_t>(rtModel)`(hack写法) → `rtModelGetId()`(正规API)
2. `aclrtSetExceptionInfoCallback` → `rtRegTaskFailCallbackByModule`(引入模块名"HCCL"参数)
3. 公共头文件`hcomm_primitives.h`中`uint32_t` → `u32`(内部typedef)

##### Revert根因

1. 对未发布API的抢跑依赖：`rtModelGetId`和`rtRegTaskFailCallbackByModule`在当时的runtime SDK中尚未发布，构建时链接失败
2. 公共头文件ABI破坏：`uint32_t`改为`u32`出现在对外头文件中，破坏外部用户ABI兼容性
3. 引入非法头文件依赖：`adapter_rts.cc`中引入了`acl/error_codes/rt_error_codes.h`内部头文件

##### 暴露的缺陷模式

1. 抢跑依赖未发布API：在runtime SDK正式发布新接口前就在消费端代码中使用，导致链接失败
2. 公共头文件类型不一致：对外API头文件必须使用标准C/C++类型，不能使用内部typedef
3. "看起来是改进"的重构放松审查警惕：去掉hack、使用正式API看起来"显然更好"，容易让审查者忽略前置条件是否满足

---

#### Revert#3: 1d8e2c14 — [Master] Revert rts interfaces

| 属性 | 值 |
|------|-----|
| Revert hash | 1d8e2c143905a5f2f035df01d7e0b12304e3fc13 |
| 日期 | 2025-12-23 |
| 作者 | jiyuanhao |
| 前序事件 | f9d03829(12/12, C25 Revert) → 9968572e(12/21, 重新提交) → 本次(12/23) |
| 涉及文件 | 7个（adapter_rts.cc/h, p2p_mgmt.cc, UT头文件等） |

注意：Revert#2和Revert#3属于同一runtime接口迁移系列，涉及同一组文件的反复修改。

##### 原始变更内容

commit message名为"Revert"，但实际是正向引入rt底层接口替换ACL接口：

1. `aclrtGetDeviceInfo` → `rtGetPhyDeviceInfo`（设备信息查询）
2. 引入`hrtGetPairDeviceLinkTypeRaw`双路径函数（先尝试dlopen加载`rtGetPairPhyDevicesInfo`，失败fallback到ACL的`aclrtGetDevicesTopo`）
3. P2P管理接口从ACL(`aclrtDeviceEnablePeerAccess`等) → RT(`rtEnableP2P`等)，函数签名从`(u32 peerDevPhyId)`改为`(u32 deviceLogicId, u32 devicePhyId)`

##### "乒乓Revert"时间线

```
12/11  d38afa51  原始runtime接口替换
12/12  f9d03829  [C25] Revert rts interfaces
12/16  30e25e50  [Master] Revert Runtime interfaces    <- Revert#2
12/21  9968572e  [C25] Use external runtime interfaces  (部分切回ACL+混合变更)
12/23  1d8e2c14  [Master] Revert rts interfaces        <- Revert#3(实际是再次引入rt接口)
```

12天内对同一组文件进行了5次方向性变更。

##### Revert根因

1. rt接口可用性不确定：某些rt接口在特定runtime版本中不可用，不得不引入dlopen动态探测+fallback机制
2. 返回值语义差异：ACL的`aclrtGetDevicesTopo`返回bitmask，rt的`rtGetPairPhyDevicesInfo`返回enum值，切换时调用方解析逻辑未同步适配
3. 缺少前期接口兼容性验证：在切换底层runtime依赖前未确认目标接口在所有部署环境都可用

##### 暴露的缺陷模式

1. 命名与行为严重脱节：多个commit message都叫"Revert rts interfaces"，但实际方向完全是"引入rt接口"。后续维护者无法从提交历史推断意图
2. 乒乓Revert模式：同一功能12天内5次方向性变更，说明缺少前期的接口兼容性验证和完整迁移方案
3. 混合提交：`9968572e`同时做了code hygiene(删extern改头文件)、功能回退(切回ACL)、UT更新，应拆为独立提交
4. 缺少运行时兼容性设计：dlopen双路径fallback在第一次引入时就应考虑，而非经过两轮revert后才补上

##### 审查教训（Revert#2和Revert#3共享）

- 跨层接口迁移需要兼容性验证前置：切换runtime依赖前需checklist确认目标接口在所有受支持版本已发布
- "Revert"一词仅用于真正的回退操作，不应描述正向迁移
- 返回值语义差异（bitmask vs enum）极易引入silent bug，审查时重点检查
- 反复revert是设计不充分的信号：审查者应暂停合并，要求提供完整迁移方案（含兼容性矩阵和rollback策略）
- 同一功能在多分支(Master/C25)反复revert，说明需要跨分支集成验证

---

#### Revert#4: 1844b823 — Revert "ars code"

| 属性 | 值 |
|------|-----|
| Revert hash | 1844b8232208fcffe2d5e2676282ddf4ad700d0a |
| 日期 | 2026-01-08 |
| 作者 | liuwanke152 |
| 原始提交 | 99d2a2b3（2026-01-07, MR !713, "ars code"） |
| 涉及文件 | 51个（跨topo/executor/operator/platform四层） |

##### 原始变更内容

为HCCL在910_93芯片上引入ARS(Adaptive Ring Size)算法，核心思想是在server内卡数不对称拓扑下动态计算最优环内大小。包含：

1. 新增3个ARS executor文件（AllReduce/AllGather/ReduceScatter各一个）
2. 新增`CalcOptimalIntraRingsize()`带宽代价模型优化器——但在`coll_comm_executor.cc`和`coll_alg_operator.cc`中被完整复制了两份，且存在微妙的成员变量引用差异
3. 修改`topo_matcher`新增COMM_LEVEL0_LOGICAL/COMM_LEVEL1_LOGICAL逻辑通信平面
4. 重构`GetBandWidthPerNPU()`从switch-case改为查表法
5. 修改`MultiRingAllGather`和`MultiRingReduceScatter`函数签名（新增`CommPlane levelIndex`参数）
6. 基类`CollAllReduceRingFor91093Executor`侵入式修改，新增protected成员
7. `isARSDoubleRing`初始化为`true`——新特性默认开启
8. 净增2104行（2283+, 179-），commit message仅两个单词"ars code"

##### Revert根因

提交1天后即被revert（1/7提交 → 1/8 revert），推断原因：

1. 侵入性过强：51个文件跨4个代码层次的全栈修改，函数签名变更影响所有调用者，未同步更新的调用点导致编译失败
2. 代码重复：`CalcOptimalIntraRingsize()`复制两份，一份用`topoAttr_.userRankSize`另一份用`userRankSize_`，一份用`topoAttr_.isARSDoubleRing`另一份用`isARSDoubleRing_`
3. 公共接口非向后兼容修改：`GetBandWidthPerNPU()`语义变更，`SetRankMap()`从private变public
4. 默认值陷阱：`isARSDoubleRing = true`让新特性默认开启，所有910_93场景都可能走到ARS路径

##### 暴露的缺陷模式

1. 大爆炸提交(Big Bang Commit)：51个文件、2000+行、跨4层的变更压缩在commit message只有"ars code"的单个提交中
2. Copy-Paste代码：核心算法函数被完整复制到两个类中，带有微妙的成员变量引用差异，典型DRY违反
3. 函数签名的破坏性变更：`MultiRingAllGather`/`MultiRingReduceScatter`新增参数改变签名，影响所有调用者
4. 默认值陷阱：新特性flag默认true，违反feature flag安全引入原则（新特性应默认关闭）
5. 缺少commit message：两个单词的message无法传达变更意图、影响范围、测试策略

##### 审查教训

- 拆分提交是有效审查的前提：至少应拆为5个独立提交（拓扑发现/带宽重构/优化算法提取/executor实现/operator选择逻辑）
- 函数签名变更需单独审查，配合全量搜索确认所有调用点已更新
- 重复代码应在审查阶段被拦截，要求提取为共享实现
- 新增布尔flag的默认值应被质疑：新特性默认开启是否安全？
- 快速revert（仅1天）暴露CI/测试覆盖不足：编译或功能回归应在MR阶段被拦截
- commit message是审查入口，reviewer应要求提交者提供足够的上下文说明

---

#### 跨Revert共性模式总结

模式1：大批量/巨型提交掩盖关键变更
- Revert#1：功能隐藏在678文件的"update"同步提交中
- Revert#4：51文件跨4层的全栈变更压缩在"ars code"中
- 教训：功能变更必须独立提交，大批量同步不应包含功能修改

模式2：对未就绪依赖的抢跑集成
- Revert#2/#3：依赖未发布的runtime API导致链接失败，12天内5次方向性变更
- 教训：跨团队接口迁移需前置兼容性验证，确认目标API在所有环境可用

模式3：缺少灰度/开关机制
- Revert#1：心跳帧格式变更直接全量生效，无法线上回退
- Revert#4：`isARSDoubleRing = true`默认开启新特性
- 教训：影响跨节点协议或全局行为的功能需feature flag

模式4：数据结构/接口的破坏性变更未隔离审查
- Revert#1：帧结构膨胀30倍未做sizeof分析
- Revert#2：公共头文件`uint32_t`→`u32`破坏ABI
- Revert#4：函数签名变更未全量检查调用点
- 教训：涉及ABI/协议/签名的变更必须单独提交并配套影响分析

模式5：commit message质量低下
- Revert#1原始提交："update"（678文件）
- Revert#3："Revert rts interfaces"（实际是正向引入）
- Revert#4："ars code"（51文件2000+行）
- 教训：commit message是审查第一入口，应准确描述变更意图和影响范围

---

### 跨仓库共性模式

综合hccl(2个Revert)、hccl-dev(2个Revert)、hcomm-dev(4个Revert)共8个Revert事件，提炼以下跨仓库共性：

1. PR/MR描述缺失是revert的头号预测因子：hccl的两次revert PR描述均为空，hcomm-dev的Revert#4原始commit仅两个单词"ars code"。缺乏上下文的变更无法被有效审查，本质上是将质量门禁从code review转嫁给了post-merge验证。审查规则：PR描述为空或commit message少于10个单词的变更不应合入。

2. 大批量提交是缺陷的天然庇护所：hcomm-dev Revert#1隐藏在678文件的"update"中，hccl-dev事件1在18文件中反复操作，hcomm-dev Revert#4跨51个文件。巨型提交让reviewer无法聚焦关键变更，功能缺陷被无关变更稀释。审查规则：功能变更必须从机械性同步/重命名中分离，单次PR功能变更不超过500行。

3. 跨组件/跨版本接口变更缺少前置验证：hcomm-dev Revert#2/#3对未发布runtime API的抢跑依赖，hccl-dev事件2的API重命名未确认下游同步。接口消费方和提供方的变更时序不协调是反复revert的根因。审查规则：跨组件接口变更需附带兼容性矩阵，确认所有消费方已就绪。

4. 新特性缺少灰度开关和回退能力：hcomm-dev Revert#1的心跳帧格式变更、Revert#4的ARS算法均直接全量生效。一旦发现问题只能整体revert，无法灰度验证或线上关闭。审查规则：影响跨节点协议、核心算法路径的新特性必须具备环境变量开关，默认关闭。

5. 快速revert(<24小时)暴露CI覆盖不足：8个revert事件中，hccl两次均在数小时内revert，hcomm-dev Revert#4仅1天。问题在合入后立即暴露说明pre-merge CI未覆盖到关键场景（构建系统端到端验证、多设备类型编译、性能回归）。审查规则：构建系统变更、硬件常量修改、核心算法变更应有专项CI pipeline。

---

## Part 3: 跨类别系统性风险

以下模式不属于单一缺陷类别，而是贯穿多个类别的系统性问题，综合自hccl(模式A-F)、hcomm-dev(风险1-5)和hccl-dev(跨类别观察)的交叉分析。

### 风险1: God Class与代码重复导致"改一漏一"

三个仓库的核心文件均为God Object，是缺陷反复出现的根本原因:

hccl热点文件:
- hccl_communicator_host.cc: 312方法/9152行, 13次缺陷修复
- aicpu_communicator.cc: 213函数/5550行, 9次缺陷修复
- communicator_impl.cc: ~100函数/3789行, 9次缺陷修复

hcomm-dev热点文件:
- hccl_communicator_host.cc: 8818行, 14次缺陷修复
- hcom.cc: 4068行, 14次缺陷修复
- aicpu_communicator.cc: 5809行, 12次缺陷修复
- op_base.cc: 4587行, 11次缺陷修复

hccl-dev:
- scatter_op.cc集中了枚举混淆和条件判断两类缺陷，是该仓库的核心风险文件

代码重复的典型形式:
- ExecOp vs ExecOpAlltoAll重复（hccl_communicator_host.cc）
- AllocTransportResource三路重复（aicpu_communicator.cc）
- 5个初始化路径（communicator_impl.cc）
- op_base.cc中InitCommRootInfo(212行)/InitCommClusterInfo(136行)/HcclCreateSubCommConfigInner三处InitCommConfig序列几乎完全重复
- hccl_communicator.cc中Suspend/StopExec/Clean三个函数约190行包含几乎完全相同的while(true)轮询循环
- dispatcher_aicpu.cc中LaunchTask和WaitRtsq的RTSQ等待/超时/拷贝逻辑大量重复

hccl前3名文件占全部缺陷修复的36.9%。任何对这些文件的修改应要求至少2个reviewer。

### 风险2: 缺陷修复引入新缺陷

- `4e82ec25`修复API参数方向 -> 引入use-after-free -> `6784944a`再次修复
- 缺陷修复提交须有额外审查关注点：修复是否引入新问题

### 风险3: 并发安全缺乏系统性设计

hcomm-dev 18个热点文件中15个存在并发安全问题。模式包括:
- TOCTOU竞态: hcom.cc(HcomCreateGroupImplHeterog), hcom_common.cc(HcomGetCurHcomCtx)
- 无锁全局变量: op_base.cc(g_oneSidedCommHcomInfos), aicpu_communicator.cc(errMessageReport_), adapter_rts.cc(g_localDeviceType), hcom.cc(g_rankTableSetInfo)
- 锁范围不一致: op_base.cc中HcclSendInner/HcclRecvInner等6个函数缺少operatorlock_

根因: 锁的使用缺乏统一的层级设计和文档化约定。

### 风险4: 公共API/ABI设计债务

- hcom.h: extern "C"块内使用namespace、公共函数签名含C++默认参数、头文件中定义non-inline std::string/std::map全局变量
- hccl_comm_pub.h: 条件编译(CCL_KERNEL_AICPU/HCCD)改变类内存布局，跨模块传指针越界
- 多个API返回悬垂指针: hcom.cc:2076 `*algo = const_cast<char*>(str.c_str())`将局部变量内部指针返回(确定性UAF)

### 风险5: 返回值系统性忽略

三个热点文件都有忽略关键函数返回值的问题:
- hrtMalloc、Mc2AiCpuStreamAllocAndGet（hccl_communicator_host.cc L1803, L6205）
- malloc 100MB未检查返回值（communicator_impl.cc L3115）
- getenv返回nullptr解引用（communicator_impl.cc L3025）

### 风险6: 编译器可捕获的缺陷仍逃逸

以下缺陷本可由编译器警告捕获但仍进入主干:
- `-Wshadow`: ef766683 变量遮蔽
- `-Wtautological-compare`: e1880dc1 自比较
- `-Wformat`: 35e32c7c, 6a6eac0f 格式串不匹配
- `-Wconversion`: 9c1f957b, 6baf33c4 整数截断
- `-Werror=enum-conversion`: d953cbf3 枚举隐式转换

建议CI中强制开启上述警告并设为error。

### 风险7: 大爆炸提交(Big Bang Commit)掩盖关键变更

hccl Revert分析:
- `72cdf80e` 对fb56d64b的revert，合入6小时后撤回。CCU loop count常量修改无spec依据，UT期望值被同步修改适配
- `753ba8c2` 对05b38411的revert，合入1小时后撤回。24文件构建系统重构未经充分验证
- 两次PR描述均为空模板，平均合入到revert间隔3.5小时
- UT被修改为适配变更而非独立验证
- 逻辑变更夹带风格变更增加review难度

hcomm-dev Revert分析:
- Revert#1原始提交: "update"(678文件)中隐藏心跳帧结构膨胀30倍
- Revert#4: "ars code"(51文件, 2000+行, 跨4层)仅两个单词commit message
- 功能变更混入大批量同步提交，审查者无法在海量无关变更中发现关键问题

hccl-dev:
- API重命名(MR !34)同日被revert，打包路径结构(MR !18)2天内经历3次变更
- 类别1和类别2合计占60%，均与API快速迭代相关，反映了开发分支侧重API设计/重构的特点

### 风险8: 确定性Bug仍存在于当前代码

hccl热点文件中仍存在的高危风险:
- 悬空指针: communicator_impl.cc L601 临时unique_ptr.get()
- malloc未检查: communicator_impl.cc L3115
- getenv空指针: communicator_impl.cc L3025
- TLV解析死循环: aicpu_communicator.cc L659 length==0
- 同一数据结构两种锁: aicpu_communicator.cc resMap_

hcomm-dev热点文件中发现的确定性bug:
- aicpu_communicator.cc:3385-3386 UpdateSqStatus中SQ_TAIL查询结果写入head、SQ_HEAD写入tail，head/tail赋值反转
- hcom.cc:2076 `*algo = const_cast<char*>(str.c_str())`返回局部变量内部指针，函数返回后即悬垂(确定性UAF)
- adapter_rts.cc:37-45 REPLACE_NOTIFY_WITH_EVENT宏中result硬编码为0，if(result!=0)永远不执行，event替换功能完全失效

---

### 附: C++语言特性 (1条) [hccl]

- `bb490dc2` CcuKernel虚析构函数声明为`= default`，跨SO边界时不生成out-of-line定义导致链接符号缺失
- 跨动态库边界的多态类（pkg_inc导出），虚析构函数不应使用`= default`，应在.cc中提供显式定义（C++ ODR + 动态链接经典陷阱）
