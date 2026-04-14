# ops-transformer 补充分析

数据来源: ops-transformer主仓(1323提交, 243条缺陷) + ops-transformer-dev(2822提交, 528条缺陷)

---

## Part 1: 热点文件风险分析

数据来源: ops-transformer主仓 + ops-transformer-dev

---

### ops-transformer主仓


### 一、缺陷热点文件统计

#### 1.1 按文件频次排序 (Top 20)

| 排名 | 文件路径 | 缺陷触及次数 | 代码层 |
|------|---------|-------------|--------|
| 1 | attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp | 22 | tiling |
| 2 | cmake/custom_build.cmake | 7 | 构建 |
| 3 | build.sh | 7 | 构建 |
| 4 | mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h | 6 | kernel |
| 5 | attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_v2.cpp | 6 | tiling |
| 6 | attention/common/op_kernel/arch35/infer_flash_attention_kvcache.h | 6 | kernel |
| 7 | mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h | 4 | kernel |
| 8 | gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul.cpp | 4 | API |
| 9 | cmake/func.cmake | 4 | 构建 |
| 10 | attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp | 4 | tiling |
| 11 | attention/common/op_kernel/arch35/flash_attention_score_block_vec_base.h | 4 | kernel |
| 12 | mc2/moe_distribute_combine/op_kernel/moe_distribute_combine_a2_layered.h | 3 | kernel |
| 13 | mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2.h | 3 | kernel |
| 14 | mc2/moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_v2_tiling.cpp | 3 | tiling |
| 15 | mc2/allto_all_matmul/op_host/op_tiling/arch32/allto_all_matmul_tiling_910b.cpp | 3 | tiling |
| 16 | mc2/allto_all_matmul/op_api/aclnn_allto_all_quant_matmul.cpp | 3 | API |
| 17 | attention/incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp | 3 | tiling |
| 18 | attention/fused_infer_attention_score/op_host/fused_infer_attention_score_infershape.cpp | 3 | infershape |
| 19 | attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp | 3 | tiling |
| 20 | attention/flash_attention_score_grad/op_api/aclnn_flash_attention_score_grad.cpp | 3 | API |

#### 1.2 按模块聚合

| 模块 | 缺陷触及总次数 | 占比 |
|------|--------------|------|
| attention | 178 | 42.3% |
| mc2 | 78 | 18.5% |
| moe | 56 | 13.3% |
| gmm | 36 | 8.6% |
| cmake/build | 19 | 4.5% |
| posembedding | 16 | 3.8% |
| tests | 9 | 2.1% |
| scripts | 6 | 1.4% |
| common | 6 | 1.4% |
| ffn | 5 | 1.2% |
| 其他 | 12 | 2.9% |

#### 1.3 按代码层聚合

| 代码层 | 缺陷触及次数 | 占比 | 说明 |
|--------|-------------|------|------|
| op_host (tiling层) | 180 | 42.9% | tiling参数计算是首要缺陷集中区 |
| op_kernel (算子核心) | 140 | 33.3% | kernel代码是第二大缺陷区 |
| 其他 (含docs/scripts) | 58 | 13.8% | |
| cmake (构建系统) | 19 | 4.5% | |
| op_api (接口层) | 15 | 3.6% | |
| common (公共库) | 5 | 1.2% | |
| tests (测试) | 3 | 0.7% | |

#### 1.4 缺陷根因类别 Top 10

| 根因类别 | 出现次数 | 占比 |
|---------|---------|------|
| 构建配置缺陷 | 39 | 16.1% |
| 除零错误(边界条件缺失) | 30+ | 12.4% |
| 条件分支遗漏/缺陷 | 20+ | 8.3% |
| 整数类型错误/溢出 | 15+ | 6.2% |
| 输入校验缺失 | 12+ | 5.0% |
| workspace/buffer计算错误 | 10+ | 4.1% |
| 硬件流水线同步缺失 | 10+ | 4.1% |
| API误用/接口参数错误 | 8+ | 3.3% |
| 赋值遗漏/复制粘贴错误 | 6+ | 2.5% |
| 空tensor场景处理缺失 | 5+ | 2.1% |

---

### 二、热点文件结构性风险审查

#### 2.1 prompt_flash_attention_tiling_v2.cpp (22次缺陷, 5163行)

核心功能: PromptFlashAttention算子的host侧tiling参数计算。根据输入shape/dtype/layout及20+个特性开关(PA/MLA/量化/Rope/Prefix等)计算分块策略、多核分配、MatMul配置、workspace大小。

这是整个仓库中缺陷密度最高的单文件，5163行承担了过多职责。

##### 高严重度风险

风险H1 -- int64_t到uint32_t截断导致数据错误 (行2901-2935)
- ParseActualSeqLengths中，actSeqLenData->GetData<int64_t>()的值通过static_cast<uint32_t>()各自截断后再做减法。TND layout下cumulative序列的差值计算会因先截断再减产生错误结果(应先减再截断)。负值输入被截断后变为巨大正数。
- 类别: 整数溢出/截断

风险H2 -- ifaBlockSizeBase被成员变量除法污染 (行2422)
- `ifaBlockSizeBase /= static_cast<int32_t>(dataTypeSize)`: 直接除以dataTypeSize，如果为0则除零。更严重的是如果类实例被复用，该成员变量会被反复除以dataTypeSize导致值越来越小直到为0。
- 类别: 除零 + 状态污染

风险H3 -- GetPFAWorkSpaceSize返回ge::GRAPH_FAILED作为size_t (行4120-4122)
- 函数返回类型是size_t，错误时返回ge::GRAPH_FAILED(枚举值通常为1)。调用方会把1当作合法workspace大小分配1字节，后续kernel执行时越界写入。
- 类别: 类型不匹配 / 错误传播

风险H4 -- 成员变量状态在多次调用间累积 (全类)
- 超过80个成员变量，没有统一的reset机制。如果实例被复用于不同参数的tiling调用，前次遗留状态影响后续调用。特别是enableXXX布尔标志和计算中间值(faRunFlag_, ifaBlockSizeBase等)。
- 类别: 状态污染 / 可重入性

##### 中严重度风险

风险M1 -- quantScale2ShapeSize / n 无零保护 (行1190)
- CheckPostQuantParams中`quantD = quantScale2ShapeSize / n`，n来自queryShapeInfo.n，异常输入路径下可能为0。

风险M2 -- int64_t到uint32_t截断(shape维度) (行545-550, 609-614)
- SetShape中h = n * d的结果是int64_t，当n和d都很大时超出uint32_t范围，截断后产生错误值。

风险M3 -- SetTilingData中4处除法依赖成员变量初始值 (行3130-3133)
- typeByteNum = BYTE_BLOCK / dataTypeSize等，如果CheckIODataType中inputType不在allowedDtypes中，dataTypeSize保持默认初始值(可能为0)。

风险M4 -- sinnerBlocknum除零 (行3880-3881)
- 空tensor场景下seqInnerSize可能为0，导致sinnerBlocknum为0，后续除法除零。

风险M5 -- gSize除零链 (行847, 2727, 5036)
- gSize = nQ / nKV在nQ本身为0时(前面只检查了nQ < 0)，gSize为0，后续多处用gSize做除数。

风险M6 -- GetMaxSeq无空/大小检查 (行398-405)
- 直接访问actualSeqLength->GetData<int64_t>()[0]，未检查GetShapeSize()>0和GetData()非空。

风险M7 -- workspace大小计算整数溢出 (行4127-4186)
- 多处大整数乘法链，uint32/int64混合运算，极端参数下中间结果可能溢出。

风险M8 -- innerPrecise被无条件覆盖导致死代码 (行2516-2529)
- innerPrecise = HIGH_PRECISION后，后续的innerPrecise == HIGH_PERFORMANCE分支永远为false，softmaxDataTypeSize永远不会被设为FLOAT16SIZE。

风险M9 -- allowedDtypes与dataTypeSizeArray长度不匹配 (行326-328, 1355, 1433)
- 5元素 vs 6元素，索引对应关系脆弱，新增类型时极易错位。此模式在文件中重复3次。

风险M10 -- CeilDiv与CeilDivision边界行为不一致 (行163-188)
- 两个功能相同的函数在除数为0时行为不同(返回n1 vs 返回0)，同一文件中共存是维护陷阱。

##### 低严重度风险

- blockTableStr.pop_back()在空字符串时的未定义行为 (行1502-1507)
- strideQ/ratio除法已有保护但负值未考虑 (行3288-3294)

##### 复杂度观察

- 5163行单文件，约80个成员函数
- 20+个互相关联的enableXXX布尔特性开关，交叉约束分散在多个Check函数中
- allowedDtypes查表模式重复3次未提取公共函数
- 5种layout(TND/NTD/BSH/BSND/BNSD)的条件分支在全文反复出现
- 最深嵌套4层(ComputeSplitNBSeq的三重循环+条件)

---

#### 2.2 moe_distribute_dispatch_v2_full_mesh.h (6次缺陷, 1611行)

核心功能: MoE分布式dispatch的full mesh通信实现。通过RDMA窗口在多卡间进行token级all-to-all数据分发。

##### 高严重度风险

风险H1 -- moeExpertRankNum_为0时除零 (行366)
- `moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_`，moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_。当所有卡都部署共享专家时为0。elastic scaling场景下这些值来自runtime tensor，不受tiling静态保证约束。

风险H2 -- rankNumPerSharedExpert_为0时取模除零 (行363, 568)
- sharedExpertRankNum_ / sharedExpertNum_计算得到rankNumPerSharedExpert_，结果为0时在568行被用作取模运算除数。

风险H3 -- moeUsedAivNum_为0时除零 (行621)
- `sendNum_ = moeExpertNum_ / moeUsedAivNum_`，moeUsedAivNum_ = aivUsedAllToAll_ - sharedUsedAivNum_，当所有核分配给共享专家时为0。

风险H4 -- 3处无限循环无超时退出 (行933-944, 987-996, 1146-1155)
- GetCumSum、WaitDispatch、WaitCumSumFlag中的while(true)忙等循环没有超时机制。远端rank故障或通信异常时整个AIV核挂死。

##### 中严重度风险

- axisMaxBS_ = globalBS_ / epWorldSizeOriginal_ 缺零保护 (行350)
- 窗口地址偏移计算中uint32乘法可能在32位空间截断后再提升为uint64 (行404-405)
- startId_/endId_/sendNum_/statusCntAlign_ 4个成员变量未初始化 (行254-257)
- elastic scaling路径下epRankId_自引用索引无边界检查 (行321)

##### 复杂度观察

- 1611行单体header，80+成员变量
- 通信/计算/内存管理全混合在一个类中
- Process()三阶段流水线通过共享成员变量隐式通信
- 圈复杂度估计200+

---

#### 2.3 fused_infer_attention_score_tiling_v2.cpp (6次缺陷, 1148行)

核心功能: FusedInferAttentionScore算子的host侧tiling。根据运行时参数决定走IFA还是PFA模板路径。

##### 高严重度风险

风险H1 -- GetAttrPointer返回nullptr时直接解引用 (行741-742, 760)
- `*attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX)` 未检查返回值。相比之下行466-478的对比代码只保存指针不解引用。lseFlag在行760也有同样问题。

##### 中严重度风险

- int64_t到uint32_t强转: GetDim返回-1(无效维度)时转为UINT32_MAX (行159, 213, 265)
- GetDynamicInputShape返回值未检查 (行439-440)
- layout字符串匹配仅覆盖BSH/BNSD/BSND三种，TND/NTD等未覆盖 (行994)

##### 复杂度观察

- DoOpTiling函数单体448行，14+种layout组合的if-else if链
- layout字符串比较分散在多处，没有统一的枚举映射

---

#### 2.4 infer_flash_attention_kvcache.h (6次缺陷, 886行)

核心功能: Flash Attention推理阶段的KV cache参数计算。处理不同layout下的query/key/value内存偏移。

##### 高严重度风险

风险H1 -- constInfo.gSize在6处被用作除数无零保护 (行38, 451, 545, 546, 668, 670-671)
- GQA未启用时gSize理论上为1，但初始化异常时为0则多处除零。

风险H2 -- GetKeyCoreOffsetParam中NTD layout分支缺失 (行281-313)
- 只处理BSH、TND和else(BNSD)，NTD走else分支。但NTD的key布局与BNSD不同。对比GetValueCoreOffsetParam有NTD独立分支(行337-346)，两个函数不对称。

##### 中严重度风险

- actualSeqQlenAddr数组越界: bIdx-1在unsigned下溢时为UINT32_MAX (行104, 108, 476)
- 偏移量计算中int32*uint64的乘法链可能中间溢出 (行288, 306, 484)
- enableKVPrefix路径下TND/NTD/BNSD统一处理但布局语义不同 (行365-377)

##### 低严重度风险

- 自赋值语句runParam.actualS1Size = runParam.actualS1Size (行191)
- halfS1RealSize除零依赖||短路求值 (行628-629)

---

#### 2.5 moe_distribute_dispatch_v2.h (4次缺陷)

核心功能: MoE分布式dispatch的kernel侧主逻辑。

##### 高严重度风险

- 行329/337: kernel侧无保护除法，tiling参数异常时导致NPU硬件异常

---

#### 2.6 aclnn_grouped_matmul.cpp (4次缺陷)

核心功能: GroupedMatMul算子的aclnn API层实现。

##### 中严重度风险

- 输入tensor列表的校验分散在多处，空列表场景下部分路径缺少前置检查

---

#### 2.7 incre_flash_attention_tiling.cpp (4次缺陷)

核心功能: Incremental Flash Attention的host侧tiling。

##### 高严重度风险

风险H1 -- increGcd函数b==0时除零 (行125-131)
- `a % b`在b为0时是未定义行为。该函数被tiling分块逻辑调用。

风险H2 -- GetMaxSeqLength未校验tensor数据有效性 (行327-333)
- 直接访问`actualSeqLength->GetData<int64_t>()[0]`，未检查GetData返回nullptr、GetShapeSize()为0。

---

#### 2.8 flash_attention_score_block_vec_base.h (4次缺陷)

核心功能: Flash Attention前向的AIV核基础实现。

##### 高严重度风险

风险H1 -- deScaleKvOffset - 1的unsigned underflow (行933/942/948/956/1095/1100/1105/1110/1235/1240/1245/1250)
- 12处相同模式: 当deScaleKvOffset为0时，无符号减法下溢产生巨大值，影响FP8 attention精度。

---

#### 2.9 incre_flash_attention_tiling_v2.cpp (3次缺陷, 2500+行)

核心功能: IFA V2的host侧tiling策略。覆盖从输入校验到tiling数据生成的完整流程。

##### 高严重度风险

- increGcd函数b==0除零 (行125-131)，与2.7同源
- GetMaxSeqLength未校验tensor空/null (行327-333)

##### 中严重度风险

- 循环变量int与GetShapeSize()的size_t类型不匹配 (行329)

---

#### 2.10 fused_infer_attention_score_infershape.cpp (3次缺陷, 428行)

核心功能: FusedInferAttentionScore的InferShape与InferDataType。

##### 高严重度风险

风险H1 -- BSH layout下numHeadsPtr值为0时除零 (行130)
- 仅检查指针非null，未检查指向值是否为0。

风险H2 -- GetValueD中numKeyValueHeads为0时除零 (行180, 197)
- PA路径和非PA BSH路径都存在。

##### 中严重度风险

- GetQueryBSND返回值被忽略 (行222, 229, 235)，失败时b/s1/n1/d1保持默认值0，静默错误。

---

#### 2.11 flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp (3次缺陷, 1500+行)

核心功能: Flash Attention Score反向梯度算子的tiling策略。

##### 高严重度风险

风险H1 -- queryRope在null检查前被解引用 (行597-598)
- 先调用->GetStorageShape()取地址，下一行才检查queryRope != nullptr。keyRope同理(行599-601)。

风险H2 -- headNum/g/n2除零传播链 (行614-631)
- queryDim2 / keyDim2 -> g, headNum / g -> n2, queryDim2 / headNum -> d, valueDim2 / n2 -> d1。上游任一维度为0导致下游连续多次除零。

##### 中严重度风险

- 手动new[]/delete[]存在异常安全问题 (行996)，中间return导致内存泄漏。

---

#### 2.12 aclnn_flash_attention_score_grad.cpp (3次缺陷, ~1000行)

核心功能: aclnn API层的Flash Attention Score Grad实现。

##### 中严重度风险

- dDim为0后n2Dim除法链 (行348-362): 有部分保护但语义上dDim==0时继续执行不正确。
- GetSumIntArrayMaxValue未处理空数组 (行136-140): Size()==0时访问[0]越界。

---

#### 2.13 flash_attention_score_block_vec_infer.h (3次缺陷, 988行)

核心功能: Flash Attention推理侧的AIV核实现，负责输出初始化、softmax LSE、Flash Decode多核合并。

##### 高严重度风险

风险H1 -- InitGlobalBuffer中workspace指针回退溢出 (行266)
- `workspace -= singleCoreOffset * preloadTimes * (aicIdx + 1)` 三个无符号整数乘积可能超uint64范围导致回绕。回退后指针无下界检查。

风险H2 -- InitOutputSingleCore中uint32减法下溢 (行561-562)
- `totalOutputSize - aivIdx * singleCoreSize`当后者更大时下溢为巨大正数。后续`> 0`判断对无符号数永远为true。

---

#### 2.14 attenmask.h (3次缺陷, 607行)

核心功能: Attention Mask的kernel侧实现。处理各种compress模式的mask数据DMA拷入和偏移计算。

##### 中严重度风险

- BoolCopyInRegbase中blockBytes可能为0导致CeilDiv除零 (行69-72)
- MergeBandModeMask中s2BaseSize为0时除零 (行277)
- halfS1RealSize强转uint16后+1可能溢出(65535+1=0) (行280)

---

#### 2.15 mc2模块其他热点文件

##### moe_distribute_combine_a2_layered.h (3次缺陷)
- 通信同步busy-wait无超时保护
- 除法运算缺零保护(与dispatch系列同类问题)

##### moe_distribute_combine_v2.h (3次缺陷)
- 空指针解引用风险(tensor数据访问前缺校验)
- 成员变量初始化不完整

##### moe_distribute_combine_v2_tiling.cpp (3次缺陷)
- 除法运算缺零保护
- workspace大小计算中整数溢出风险

##### allto_all_matmul_tiling_910b.cpp (3次缺陷)
- tiling参数计算中的除法链缺零保护
- 条件分支覆盖不完整

##### aclnn_allto_all_quant_matmul.cpp (3次缺陷)
- 输入校验分散，部分路径缺少前置检查
- 空指针解引用风险

---

### 三、跨文件系统性风险模式

#### 3.1 除法零保护缺失 (系统性, 覆盖16/20热点文件)

这是整个仓库最突出的系统性风险。除数来源包括:
- shape维度(headNum, n, nKV, gSize, d): 来自用户输入tensor，异常输入可能为0
- tiling参数: 通过tiling data传递，理论上由tiling侧保证非零，但kernel侧无本地校验
- 计算中间结果(sinnerBlocknum, moeExpertRankNum_): 上游计算异常导致为0

典型传播链: H / headNum -> d, H / d -> n2, H / n2 -> d1。一个上游维度为0触发下游连续多次除零。

检查点: 每个除法运算的除数是否来自可信来源? 是否有本地零值校验?

#### 3.2 整数类型截断与溢出 (系统性, 覆盖10/20热点文件)

三类子模式:
1. int64_t -> uint32_t截断: shape维度GetDim()返回int64_t，被static_cast<uint32_t>()截断。-1(无效维度)变为UINT32_MAX。
2. uint32_t无符号减法下溢: a - b当b > a时下溢为巨大正数而非负数。特别在kernel代码中因性能使用窄类型(uint16_t, uint32_t)时频发。
3. 乘法链中间溢出: 偏移量/workspace大小计算中多个uint32相乘，中间结果在32位空间溢出后再提升为64位。

检查点: 跨类型赋值是否有范围检查? 无符号减法是否可能下溢? 乘法链是否在最宽类型空间中计算?

#### 3.3 条件分支不完整 (系统性, 覆盖8/20热点文件)

三类子模式:
1. Layout分支遗漏: 5种layout(BSH/BSND/BNSD/TND/NTD)的处理中NTD最容易被遗漏。GetKeyCoreOffsetParam vs GetValueCoreOffsetParam的不对称是典型案例。
2. 可选输入null检查时序: 先解引用后判空(flash_attention_score_grad_tiling中queryRope)。
3. 函数返回值忽略: GetQueryBSND等返回ge::graphStatus，调用方未检查导致错误静默传播。

检查点: 新增layout时是否同步更新了所有相关函数? 可选输入是否先判空再使用? 返回错误码的函数调用是否检查了返回值?

#### 3.4 状态管理缺陷 (局部但高危, 覆盖3/20热点文件)

主要出现在大型tiling类中:
- prompt_flash_attention_tiling_v2.cpp: 80+成员变量无统一reset，复用实例导致状态污染
- moe_distribute_dispatch_v2_full_mesh.h: 80+成员变量，三阶段流水线通过共享状态隐式通信
- GetPFAWorkSpaceSize返回ge::GRAPH_FAILED(值=1)作为size_t，调用方误认为合法大小

检查点: tiling类实例是否被复用? 成员变量是否在每次调用前重置? 错误返回值的类型是否与函数签名匹配?

#### 3.5 并发/通信同步无超时 (局部, 覆盖mc2模块)

mc2模块的分布式通信代码中，while(true)忙等循环缺乏超时退出机制。远端故障或通信异常时AIV核挂死。

涉及文件: moe_distribute_dispatch_v2_full_mesh.h, moe_distribute_combine_a2_layered.h

检查点: 分布式同步循环是否有超时退出? 超时后是否有错误上报机制?

---

### 四、热点文件复杂度指标

| 文件 | 行数 | 估计圈复杂度 | 最深嵌套 | 成员变量数 | 核心问题 |
|------|------|-------------|---------|-----------|---------|
| prompt_flash_attention_tiling_v2.cpp | 5163 | 300+ | 4层 | 80+ | 单文件过大, 职责过多, 20+特性开关交叉 |
| moe_distribute_dispatch_v2_full_mesh.h | 1611 | 200+ | 4层 | 80+ | 通信/计算/内存混合, 状态耦合极紧 |
| flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp | 1500+ | 150+ | 4层 | - | 除零传播链, 手动内存管理 |
| incre_flash_attention_tiling_v2.cpp | 2500+ | 200+ | 5-6层 | - | 多模式状态依赖, 分支嵌套深 |
| fused_infer_attention_score_tiling_v2.cpp | 1148 | 80+ | 3层 | - | 448行单函数, layout枚举爆炸 |
| infer_flash_attention_kvcache.h | 886 | 60+ | 4层 | - | 模板实例化组合数百种, 偏移计算分散 |

---

### 五、风险优先级排序 (建议修复顺序)

#### P0 (立即修复 -- 可导致NPU硬件异常或数据错误)

1. flash_attention_score_block_vec_base.h: deScaleKvOffset-1 unsigned underflow (12处)
2. flash_attention_score_block_vec_infer.h: uint32减法下溢导致tailSize错误 (行561)
3. prompt_flash_attention_tiling_v2.cpp: GetPFAWorkSpaceSize返回ge::GRAPH_FAILED作为size_t (行4120)
4. moe_distribute_dispatch_v2_full_mesh.h: moeExpertRankNum_/rankNumPerSharedExpert_/moeUsedAivNum_除零 (行363-621)

#### P1 (高优先级 -- 特定配置下触发崩溃)

5. flash_attention_score_grad_tiling: queryRope null检查前解引用 (行597)
6. flash_attention_score_grad_tiling: headNum/g/n2除零传播链 (行614-631)
7. fused_infer_attention_score_infershape: numHeadsPtr/numKeyValueHeads值为0除零 (行130, 180, 197)
8. fused_infer_attention_score_tiling_v2: GetAttrPointer返回nullptr解引用 (行741-742)
9. infer_flash_attention_kvcache.h: gSize在6处做除数无零保护 (行38-671)
10. incre_flash_attention_tiling: increGcd函数b==0除零 (行125)

#### P2 (中优先级 -- 边界场景下的潜在问题)

11. prompt_flash_attention_tiling_v2: ParseActualSeqLengths类型截断 (行2901)
12. prompt_flash_attention_tiling_v2: 成员变量状态累积/无reset (全类)
13. prompt_flash_attention_tiling_v2: workspace大小计算溢出 (行4127-4186)
14. moe_distribute_dispatch_v2_full_mesh: busy-wait无超时 (3处)
15. aclnn_flash_attention_score_grad: GetSumIntArrayMaxValue空数组越界 (行136)

#### P3 (低优先级 -- 代码质量/维护性)

16. prompt_flash_attention_tiling_v2: CeilDiv/CeilDivision行为不一致
17. prompt_flash_attention_tiling_v2: innerPrecise覆盖导致死代码
18. prompt_flash_attention_tiling_v2: allowedDtypes/dataTypeSizeArray长度不匹配
19. 各tiling文件: layout分支覆盖完整性

---

### ops-transformer-dev


### 一、缺陷热点文件 Top-30

按被缺陷修复提交触及次数排序（788条缺陷提交中各文件被触及次数）：

| 排名 | 文件 | 触及次数 | 模块 |
|------|------|---------|------|
| 1 | cmake/custom_build.cmake | 23 | Build |
| 2 | attention/prompt_flash_attention/.../tiling_v2.cpp | 19 | PFA |
| 3 | cmake/func.cmake | 17 | Build |
| 4 | mc2/matmul_all_reduce/op_kernel/matmul_all_reduce.cpp | 14 | MC2 |
| 5 | gmm/grouped_matmul/.../grouped_matmul_tiling.cpp | 14 | GMM |
| 6 | CMakeLists.txt | 14 | Build |
| 7 | mc2/moe_distribute_dispatch_v2/.../tiling.cpp | 13 | MC2/MoE |
| 8 | cmake/obj_func.cmake | 12 | Build |
| 9 | classify_rule.yaml | 12 | Build |
| 10 | build.sh | 12 | Build |
| 11 | tests/test_config.yaml | 11 | Test |
| 12 | mc2/moe_distribute_combine_v2/.../combine_v2.h | 11 | MC2/MoE |
| 13 | mc2/moe_distribute_combine_add_rms_norm/.../kernel.h | 11 | MC2/MoE |
| 14 | mc2/matmul_all_reduce/.../quant_tiling_910_95.cpp | 11 | MC2 |
| 15 | attention/flash_attention_score_grad/.../regbase.cpp | 11 | FAG |
| 16 | gmm/grouped_matmul/.../aclnn_grouped_matmul.cpp | 10 | GMM |
| 17 | attention/incre_flash_attention/.../tiling_v2.cpp | 10 | IFA |
| 18 | attention/common/op_kernel/arch32/fia_kernel_nonquant_mla.h | 10 | FIA/MLA |
| 19 | mc2/moe_distribute_dispatch_v2/.../dispatch_v2.h | 9 | MC2/MoE |
| 20 | mc2/moe_distribute_combine_v2/.../tiling.cpp | 9 | MC2/MoE |

模块分布: Build系统(6/20) > MC2/MoE(7/20) > Attention(5/20) > GMM(2/20)

### 二、构建系统热点 (5文件, 3308行)

#### cmake/custom_build.cmake (23次)
- 847行
- 风险1: BUILD_OPEN_PROJECT与非BUILD_OPEN_PROJECT两条分支大量重复代码。aclnn/aclnnInner/aclnnExc三组target创建、aclnn源码生成循环、opbuild命令生成几乎是三份拷贝，修一处漏一处。
- 风险2: 第540-561行target_sources/target_link_libraries在if/else块外无条件执行，引用的`${OPHOST_NAME}_opapi_obj`等target仅在BUILD_OPEN_PROJECT=ON时存在，依赖`$<TARGET_EXISTS:...>`防护，去掉guard即崩溃。
- 风险3: 第330-346行硬编码`".*attention.*"`和`".*moe_inplace_index_add_with_sorted.*"`特殊算子名正则匹配，算子增减时容易遗忘更新。

#### cmake/func.cmake (17次)
- 846行
- 风险1: `add_modules_sources`和`add_modules_sources_with_soc`两个macro代码重复率约80%。
- 风险2: 两者是macro而非function，局部变量(如SOURCE_DIR)会污染调用者作用域。
- 风险3: `add_bin_compile_target`(501-738行)237行长函数，含深层嵌套循环和无注释的"group"格式解析(`1-0`字符串拆分为step/start_index)。
- 风险4: `A5_OPS_BLACK_LIST`每个元素带末尾分号如`"mla_prolog;"`，不加分号的元素无法被`list(FIND ...)`匹配。

#### CMakeLists.txt (14次)
- 607行
- 风险1(已确认bug): 第60行变量名拼写错误`INDXE`(应为INDEX)。`list(FIND ARCH_DIRECTORY "arch32" INDXE)`存入INDXE，但第61行`if(NOT INDEX EQUAL -1)`检查的是INDEX(上方foreach残留值)。arch22是否被追加完全取决于上次循环残留状态。
- 风险2: 第289-296行硬编码特定算子名和路径，绕过统一发现机制。
- 风险3: 第322-607行BUILD_OPS_RTY_KERNEL块(285行)与custom_build.cmake几乎平行实现，增减feature需两处同步。

#### cmake/obj_func.cmake (12次)
- 697行
- 风险1: `add_opapi_modules`/`add_infer_modules`/`add_tiling_modules`三个function的include_directories大量重复，新增公共include需3处同步。
- 风险2: `AICPU_INCLUDE`变量在variables.cmake第252-264行通过`list(APPEND ...)`扩展后，被第267行`set(AICPU_INCLUDE ...)`无条件覆盖，append白做。
- 风险3: 所有GLOB调用(48/66/73/93-99行)不会在文件增减时自动重新配置。

#### cmake/variables.cmake (7次)
- 311行
- 风险1: 第259行include路径使用通配符`*.h`，include_directories期望目录路径而非文件通配符，静默失效。
- 风险2: 第302-311行用`grep -Po`(Perl正则)获取版本号，macOS不支持`-P`选项。
- 风险3: OPAPI_INCLUDE/OP_TILING_INCLUDE/OP_PROTO_INCLUDE三组include列表存在大量重复路径，缺乏公共部分提取。

构建系统小结: 核心问题是custom_build.cmake与CMakeLists.txt(RTY)的平行实现、func.cmake两个macro的高重复、variables.cmake三组include列表的冗余。23次缺陷触及的根因是"修一处漏一处"的结构性问题。

### 三、Tiling文件热点 (5文件, ~15000行)

#### prompt_flash_attention_tiling_v2.cpp (19次)
- 4762行
- 风险1: 巨型单文件4762行，`RunBigKernelTilingWithParams`含20+串行校验步骤。
- 风险2: workspace计算`batchSize * gSize * headNumSize * kvSplitPart * vHeadSize * sizeof(float)`六值连乘，大batch+大head场景溢出风险。
- 风险3: `CeilDivision`和`AlignUp`在除数为0时静默返回0，下游使用错误tiling参数可能导致kernel访存越界。
- 风险4: `dVBasicBlock`在dSize>512时保持为0(未赋值)，导致`bmm2ResBlockSize=0`，workspace计算偏小。
- 风险5: `PFATilingDataconvert`100+行逐字段手工搬运，新增字段容易遗漏。

#### grouped_matmul_tiling.cpp (14次)
- 2103行
- 风险1: `SixteenAlign`的int64_t版本`a & ~15`对有符号数的行为在C++20前是实现定义的。
- 风险2: `AlignUp`对num2==0返回0而非报错，`ratio = float(curTaskNum) / AlignUp(uint32_t(curTaskNum), aicNum)`在aicNum=0时产生NaN。
- 风险3: `IsFixedAxisMoveCondition`中FIXAXISMOVE_K1/K2/N1/N2等magic number与kernel侧寄存器分配紧耦合。

#### moe_distribute_dispatch_v2_tiling.cpp (13次)
- 1833行
- 风险1: window size计算`maxBs * tokenNeedSize * epWorldSize * localMoeExpertNum * DOUBLE_DATA_BUFFER`在大集群(epWorldSize=768)场景溢出风险。
- 风险2: `globalBs / epWorldSize`计算maxBs，epWorldSize=0时除零无防御。
- 风险3: socVersion通过字符串比较("Ascend910_95")路由，添加新硬件型号时极易遗漏分支。
- 风险4: if分支数(69)远大于else分支数(19)，大量条件缺少else fallthrough处理。

#### incre_flash_attention_tiling_v2.cpp (10次)
- 4351行
- 风险1(已确认bug): `SetLayoutTypefaRun`中map有重复key: `IfaLayout::BSH_BSND`映射了两次(分别到LAYOUT_BSH和LAYOUT_BSND)，std::map覆盖前值，BSH_BSND实际映射到LAYOUT_BSND而非LAYOUT_BSH。
- 风险2: `RunBigKernelTiling`中`EmptyTensorProcess() != ge::GRAPH_SUCCESS`时才走正常tiling流程，"失败才继续"的控制流极易误读。
- 风险3: `CheckUbSpace()`返回`ge::graphStatus`但函数体`return false/true`，bool到graphStatus隐式转换导致语义反转(false=0=GRAPH_SUCCESS)。
- 风险4: `UpdateTilingKeyConfig()`的else分支只打LOGE但没return错误码，函数返回void，调用链无法感知不支持的参数组合。

#### fused_infer_attention_score_tiling.cpp (8次)
- 2002行
- 风险1: `GetQueryD`中BSH layout分支`queryD = queryH / numHeads`，numHeads未校验是否为0。
- 风险2: layout字符串"BSH"/"BNSD"/"TND"等硬编码散布在20+个函数中，缺少集中enum映射，拼写错误无编译期保护。
- 风险3: 路由函数`TilingFusedInferAttentionScore`400+行mega-function同时处理IFA/PFA/FAI三条路径。

Tiling文件小结: 核心系统性风险: (1) 整数溢出(5个文件的workspace/buffer计算都涉及多维度连乘); (2) 静默失败的除零防御(AlignUp/CeilDivision返回0); (3) 手工字段搬运(TilingDataconvert); (4) magic number分散无集中定义。

### 四、Kernel/Host文件热点 (9文件)

#### matmul_all_reduce.cpp (14次)
- 585行
- 风险1: 310P与910B两套分支通过`#if __CCE_AICORE__ == 220`隔离，各含20+个`if constexpr`分支。新增量化模式需同步修改两处。
- 风险2: 宏参数传递(`INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL_COMM_INT8`等)嵌套在条件编译内部，参数顺序/类型依赖调用点的`using`定义，跨文件一致性难以静态验证。

#### moe_distribute_combine_v2.h (11次)
- 1318行
- 风险1: `WaitDispatch`中用float比较判断状态完成，浮点精度误差可能导致永不满足条件，且无超时保护，异常时死循环hang住。
- 风险2: `epOffset * hAlignWinSize_`地址计算，两个uint32乘积在大规模MoE场景可能溢出。
- 风险3: `LocalWindowCopy`循环体超130行，GM地址通过`base + offset1 + offset2`重计算，难以审计一致性。

#### moe_distribute_combine_add_rms_norm.h (11次)
- 1321行
- 风险1: 与combine_v2.h约80%代码相似但存在关键差异(HcclOpResParam vs HcclOpParam, GetWinAddrByRankId实现不同)。修bug不同步到另一侧就会行为不一致。
- 风险2: 缺少v2中的`bufferNum_`和shared expert相关成员，若启用shared expert + AddRmsNorm组合将触发未定义行为。

#### quant_matmul_all_reduce_tiling_910_95.cpp (11次)
- 839行
- 风险1: `GetDynamicQuantTempBuffSize`中`procRows = ubSize / ubDenomQuant`，ubDenomQuant为0时除零。
- 风险2: `GetWorkspaceSize`中`rankNum * M * N * sizeof(type)`在大模型场景可能溢出uint64。
- 风险3: `GetTilingKey`使用`GET_TPL_TILING_KEY`宏按位组装tilingKey，参数顺序必须与kernel侧严格一致，仅靠人工保证。

#### flash_attention_score_grad_tiling_...regbase.cpp (11次)
- 4202行（本仓最大单文件）
- 风险1: 函数调用链深度超5层(CalcleDeterParam -> CalcleTNDDeterParam -> CalcleTNDCausalDeterParam -> ...)，每层含大量数学计算和条件分支。
- 风险2: `std::copy`到固定大小数组`deterPrefix0/1/2`，源vector大小取决于运行时batch数，可能写越界。
- 风险3: `GetTilingKey`组装18个参数到uint64 tilingKey，参数位域排列通过宏完成，增删参数必须同步kernel侧。
- 风险4: `(optiling::DtypeEnum)4/5/6`强转magic number，枚举定义变更将silent错误。

#### fia_kernel_nonquant_mla.h (10次)
- 1026行
- 风险1: `Init`中workspace每个buffer的offset依赖前一个buffer的`sizeof(type) * size`(类型链half->float->half交替)。tiling侧修改任一buffer大小但kernel侧未同步，所有后续buffer地址错位且不会报错。
- 风险2: 模板参数爆炸(layout/dtype/PA/rope多维度)，PA+MLA+非标准layout交叉场景测试覆盖可能不足。

#### memory_copy.h (8次)
- 2775行
- 风险1: 8+种格式的OffsetCalculator特化，新增格式需同时修改特化和所有Copy类分支。
- 风险2: `SafeStrideCopy`处理`dstStride > UINT32_MAX`的退化路径，假设dstStride是字节单位，非标准类型(int4b_t等)的sizeof可能不符合预期。
- 风险3: 语义混淆的接口命名: `GetStrideBlockSize()`返回的是stride而非blockSize。

#### aclnn_grouped_matmul.cpp (10次)
- 2440行
- 风险1: V1/V2/V3/V4/V5/WeightNz共6个`GetWorkspaceSize`入口，参数子集不同但主体逻辑相似，新增参数/修check需同步6处。
- 风险2: `TransWeightToNz`中310P/非310P分支的break/continue控制流语义不直观(只检查第一个weight就break vs 检查所有)。

### 五、已确认的存量bug

1. CMakeLists.txt:60-61 -- `INDXE`变量名拼写错误导致arch32的检测逻辑使用了错误变量(上次循环残留值)
2. incre_flash_attention_tiling_v2.cpp -- `SetLayoutTypefaRun`中`BSH_BSND`重复key导致映射到错误layout
3. incre_flash_attention_tiling_v2.cpp -- `CheckUbSpace()`返回bool而声明为ge::graphStatus，语义反转
4. variables.cmake:267 -- `set(AICPU_INCLUDE ...)`无条件覆盖前序`list(APPEND ...)`结果

### 六、跨文件系统性风险模式

#### 模式1: 代码克隆/平行实现
- cmake/custom_build.cmake vs CMakeLists.txt(RTY分支): 285行平行实现
- cmake/func.cmake两个macro: 80%重复
- moe_distribute_combine_v2.h vs _add_rms_norm.h: 80%相似但有关键差异
- aclnn_grouped_matmul.cpp 6个版本入口
- PFA/IFA各自独立的TilingDataconvert

影响: "修一处漏一处"是23次cmake缺陷和11次MC2 kernel缺陷的根因。

#### 模式2: 整数溢出/除零
- workspace/buffer大小计算涉及4-6个维度连乘，普遍缺少溢出检查
- AlignUp/CeilDivision在除数为0时静默返回0
- 地址计算(uint32乘积)在大规模场景溢出

涉及文件: 几乎所有tiling文件和kernel文件。

#### 模式3: 跨文件隐式一致性约束
- tilingKey按位组装(host) vs 模板参数解析(kernel): 顺序必须严格一致
- workspace内存布局(host计算offset) vs kernel读取offset: sizeof链必须匹配
- 宏参数顺序(matmul_all_reduce.cpp) vs 宏定义: 仅靠人工保证

影响: tilingKey不匹配是本仓高频缺陷模式之一(阶段2分析中~20+条)。

#### 模式4: 硬件平台分支不完整
- socVersion字符串比较路由("Ascend910_95")
- `#if __CCE_AICORE__ == 220`预处理分支
- A2/A3/A5平台常量分散定义

影响: 新增硬件型号时多处遗漏是常见缺陷来源。

---

## Part 2: Revert事件分析

数据来源: ops-transformer主仓 + ops-transformer-dev

---

### ops-transformer主仓


共发现10条Revert相关提交(含1条空提交)，涉及9个独立的回退事件。

### 原始提交追溯汇总

| Revert hash | 原始提交 | 间隔commit数 | 回退原因类别 |
|-------------|---------|-------------|------------|
| 613ae40b | 8df8fe65 | ~19 | API兼容性/编译依赖耦合 |
| 349083a7 | cc85a74e | ~20 | 测试框架迁移不成熟 |
| a6aa6f17 | 54df45bb | ~4 | 编译依赖不完整 |
| 8ade8c3e | 81ef4922 | ~7 | 并发同步正确性 |
| e5988e2b | 5c46cb51 | ~84 | 状态管理逻辑错误(revert of revert) |
| f8ab5505 | 59c9c74f+ec00dc96+f61275e1 | ~30+ | 示例代码集成问题 |
| c8f181eb | ff21f09b | ~1 | 框架API不可用 |
| 4fd755b6 | b3b1f377 | ~7 | 模板编译超时 |
| d6dd8838 | (空提交) | -- | 操作失误 |

---

### 逐条分析

#### 613ae40b revert//新增ROPEV2接口支持辅助矩阵输入

- 原始提交: `8df8fe65` "新增ROPEV2接口支持辅助矩阵输入"
- revert原因: 原始提交是一个大型特性变更(27个文件, +2930/-47行)，引入了aclnnRotaryPositionEmbeddingV2新接口，支持辅助旋转矩阵(rotate)输入。关键问题：原始提交同时修改了interleave_rope的调用方式——将本地声明的`aclnnInnerRotaryPositionEmbedding`替换为通过头文件引入的公共API `aclnnRotaryPositionEmbedding`，并在CMakeLists中新增了对rotary_position_embedding模块的依赖(`set(interleave_rope_depends ...)`)。导致API层面的兼容性或编译/链接依赖问题。由cann-robot自动创建revert MR(!1862)，说明是CI门禁触发的紧急回退。
- 缺陷类型: 接口设计缺陷/编译依赖耦合
- 严重程度: 高
- 涉及文件: `posembedding/rotary_position_embedding/`下27个文件；`posembedding/interleave_rope/op_host/CMakeLists.txt`，`posembedding/interleave_rope/op_host/op_api/aclnn_interleave_rope.cpp`
- 审查规则:
  1. 新增公开API(aclnn接口)应与内部实现解耦，不应在同一MR中同时修改依赖方的调用方式
  2. 超过25个文件/3000行的大型特性应分阶段提交，降低回退代价
  3. 修改现有算子对其他算子的调用路径(Inner -> Public API)属于高风险变更，需全平台集成测试

---

#### 349083a7 Revert "修改了all_gather_matmul算子ut的op_host组件的用例输入方式，改成csv表格"

- 原始提交: `cc85a74e` "修改了all_gather_matmul算子ut的op_host组件的用例输入方式，改成csv表格"
- revert原因: 原始提交将all_gather_matmul算子的UT用例从C++硬编码改为CSV表格驱动方式。改造引入了CSV解析框架依赖，新增了.csv数据文件和参数解析结构体(param.h)，同时删除了原有的infershape独立测试文件。CSV框架本身存在稳定性问题，或改造不完整(删除了原测试但CSV未完全覆盖所有场景)，导致UT运行失败或覆盖率下降。
- 缺陷类型: 测试基础设施迁移不成熟/测试覆盖回退
- 严重程度: 中
- 涉及文件: `mc2/all_gather_matmul/tests/ut/op_api/`和`mc2/all_gather_matmul/tests/ut/op_host/`下的测试文件、CSV文件、param.h等
- 审查规则:
  1. 测试框架迁移(硬编码 -> CSV驱动)应确保新框架经过充分验证后再推广
  2. 迁移过程中不应删除原有测试文件，应先并行运行验证新框架完全覆盖后再清理
  3. 涉及删除独立测试文件时，审查者应确认新框架中已包含等价的测试覆盖

---

#### a6aa6f17 revert//优化头文件

- 原始提交: `54df45bb` "优化头文件"
- revert原因: 原始提交对FIA/IFA核心kernel代码进行了头文件细粒度化优化，将粗粒度的`kernel_operator.h`拆分为更精确的子头文件(`kernel_vec_intf.h`, `kernel_cube_intf.h`, `adv_api/activation/softmax.h`等)。拆分遗漏了某些隐式依赖(某些类型定义或宏通过kernel_operator.h间接引入，拆分后不再可达)，导致在某些编译配置下编译失败。由cann-robot自动创建revert MR(!1862)，CI编译失败触发。距原始提交仅4个commit。
- 缺陷类型: 编译依赖不完整/头文件管理错误
- 严重程度: 中
- 涉及文件: `attention/fused_infer_attention_score/op_kernel/`和`attention/incre_flash_attention/op_kernel/`和`attention/prompt_flash_attention/op_kernel/`下共8个头文件
- 审查规则:
  1. 头文件细粒度化(将kernel_operator.h拆分为子头文件)必须经过全平台、全编译配置的编译验证
  2. 格式清理(trailing whitespace)不应与功能性修改混在同一个commit中
  3. 同时修改多个核心算子(FIA + IFA + PFA)的头文件属于广影响范围变更，应逐模块提交

---

#### 8ade8c3e Revert "dispatch优化syncall"

- 原始提交: `81ef4922` "dispatch优化syncall"
- revert原因: 原始提交对moe_distribute_dispatch_v2的同步机制和token计数逻辑进行了优化，将expertTokenNums的计算从单核(lastCore)改为FIRST_CORE(aivId == 0)执行，并使用ReduceSum+循环方式替代原来的直接累加，同时修改多处同步原语(PipeBarrier改为SyncFunc)。这些变更导致运行时正确性问题(数据竞争或结果错误)。距原始提交仅7个commit。
- 缺陷类型: 并发同步/正确性问题
- 严重程度: 高
- 涉及文件: `mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h`(主逻辑)，`mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_constant.h`(常量定义)
- 详细分析:
  - 新增`SetExpertTokenNums()`方法，仅在FIRST_CORE执行，使用statusBuf_从windowInstatusFp32Tensor_复制状态，对每个localExpert逐个调用ReduceSum计算token数。相比原来在lastCore上通过直接读取sendCounts累加，依赖更多同步点
  - 同步修改：将`PipeBarrier<PIPE_MTE3>()`替换为`SyncFunc<AscendC::HardEvent::S_MTE3>()`和`SyncFunc<AscendC::HardEvent::MTE3_MTE2>()`
  - revert恢复了统一的`UpdateTokenNumsOut()`处理方式，在lastCore直接读取GM上的sendCounts值
- 审查规则:
  1. 修改多核同步机制(SyncAll/PipeBarrier/SyncFunc)必须有严格的并发正确性测试，包括多卡多专家场景
  2. 将计算从一个核移到另一个核时，需证明该核在执行时刻已拥有完整的数据依赖
  3. 同步原语的替换(PipeBarrier -> SyncFunc)改变了全局同步语义，需证明等价性

---

#### e5988e2b revert

- 原始提交: `5c46cb51`(对`57884a9b` "support dfx: mc2_win"的不当回退)
- revert原因: commit message只写了"revert"一个词。本质上是"revert of revert"——`5c46cb51`尝试将fullmesh的窗口状态管理逻辑从使用公共基类改回手动实现，但这个改动本身是错误的，所以e5988e2b又把它revert回来。
- 缺陷类型: 状态管理逻辑反复 + commit message缺失
- 严重程度: 高
- 涉及文件: `mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h`
- 详细分析:
  - diff涉及MOE dispatch核心路径：移除基类头文件include，恢复本地UB_ALIGN常量定义(重复定义)
  - 将`SetDataStatus()`从使用`InitWinState()`函数改回手动读写方式
  - 后续提交(64e33c46 "fix win addr and sync bug", 662f162c "fix a synchronization issue")证明MOE dispatch v2 fullmesh的状态管理是持续痛点
- 审查规则:
  1. commit message必须说明回退什么、为什么回退
  2. 对"revert of revert"场景，说明原始变更反复的原因比代码本身更重要
  3. 窗口状态管理(SetDataStatus)涉及设备侧GM状态读写，属于高风险代码，修改前应有状态机文档

---

#### f8ab5505 revert add_example

- 原始提交: 多个提交的累积效果：`59c9c74f`("add_example mod"), `ec00dc96`("add_example 编译修改"), `f61275e1`("add_example sup 910_93")
- revert原因: 新版add_example引入的图模式调用(GE IR)在开源仓场景下不适用(需GE图引擎依赖)，CMakeLists的BUILD_OPEN_PROJECT分支处理与构建系统冲突，且多轮修改导致路径管理和命名空间混乱。
- 缺陷类型: 示例代码过度扩展/编译集成问题/多轮修改积累的技术债
- 严重程度: 低
- 涉及文件: `examples/add_example/`下17个文件(+28/-658行)
- 审查规则:
  1. 示例代码中引入的能力(如图模式GE IR调用)应先确认是否适用于目标仓库(开源仓 vs 内部仓)
  2. 多轮修改(mod -> 编译修改 -> sup 910_93)说明原始方案不成熟，应在设计阶段明确需求再一次性实现
  3. 命名空间(Math vs Transformer)和include路径(相对 vs 绝对)的反复切换说明缺少统一编码规范

---

#### c8f181eb revert//schedule

- 原始提交: `ff21f09b` "set schedule mode"
- revert原因: 原始提交为FIA非量化MLA分支添加了schedule mode设置功能，依赖框架的`context_->SetScheduleMode()`接口。该接口在当时的框架版本中不可用或未稳定。revert仅间隔1个commit，由turing_project1自行创建并合入，说明问题明确且紧急。MR名称中的"-auto"表明是自动触发。
- 缺陷类型: 框架API依赖/接口不成熟
- 严重程度: 中
- 涉及文件: `attention/common/op_host/arch32/fia_tiling_nonquant_mla.cpp`，`attention/common/op_host/arch32/fia_tiling_nonquant_mla.h`，`attention/common/op_host/fia_tiling_base.h`
- 详细分析:
  - 新增`ScheduleMode`枚举(NORMAL_MODE=0, BATCH_MODE=1, SYNC_MODE=2)和`SetScheduleMode()`方法
  - 硬编码`scheduleMode_ = ScheduleMode::BATCH_MODE`
  - 后续在多个提交中(0e1f7c2d, 9ac275b5, bc5e43c1)重新引入了此功能，说明功能本身是必要的，只是首次提交时机不对
- 审查规则:
  1. 依赖框架新API前应确认API已在目标框架版本中稳定发布
  2. 仅间隔1个commit即被revert，说明提交前没有经过集成编译验证

---

#### 4fd755b6 FIA IFA PFA tilingkey revert

- 原始提交: `b3b1f377` "FIA PFA IFA模板化tiling key"
- revert原因: MR描述明确："FIA IFA PFA算子tilingkey整改回退, 编译超时导致线上ci阻塞"，关联Issue #48。模板化tilingkey导致C++编译器需实例化海量模板组合，编译时间超过CI限制。
- 缺陷类型: 编译性能问题——模板膨胀(template bloat)导致编译超时
- 严重程度: 高
- 涉及文件: 32个文件, +7717/-12346行，涉及`attention/fused_infer_attention_score/`、`attention/incre_flash_attention/`、`attention/prompt_flash_attention/`下的op_host和op_kernel
- 详细分析:
  - 核心目标：将FIA、IFA、PFA三大attention算子的tilingkey路由从硬编码if/else改为C++模板化
  - 问题：每个attention算子有数十个tilingkey维度(head_dim, seq_len, batch_size, quant_mode, mask_type, softmax_mode等)，模板参数的笛卡尔积导致实例化数量呈指数级增长
  - 后续通过`535a2749`("FIA IFA PFA tilingkey模版化整改")和`959643c6`("fia ifa pfa tilingkey整改 ut 适配")重新完成了模板化
- 审查规则:
  1. 大规模模板化重构必须在提交前测量编译时间，并设置编译时间预算作为CI门禁
  2. 超过10000行的diff必须分阶段提交(先FIA单独验证，通过后再IFA，最后PFA)
  3. 模板化tilingkey时应考虑使用显式实例化控制实例化数量，或使用tag dispatch等编译时间更友好的技术

---

#### d6dd8838 tilingkey revert

- 原始提交: (空提交，tree hash与parent完全相同)
- revert原因: 这是`4fd755b6`的前置revert尝试。b3b1f377合入后CI编译超时，团队紧急组织回退。d6dd8838(MR!191)是第一次revert尝试，但merge后产生了空结果(没有任何文件变更)，随后4fd755b6(MR!195)才完成了真正的revert。
- 缺陷类型: 操作失误/空提交
- 严重程度: 低
- 涉及文件: 无(空提交，0个文件变更)
- 审查规则:
  1. revert操作前应先`git diff`确认回退内容非空
  2. 紧急回退应有标准操作手册(SOP)，避免在匆忙中产生无效操作
  3. "merge master into master"这种source/target相同的MR不应被允许合入

---

### 跨Revert模式总结

#### 模式1：编译问题是首要revert原因(3/9)

- 头文件依赖不完整(a6aa6f17)
- 模板编译超时(4fd755b6)
- 框架API不可用(c8f181eb)

暗示代码审查中"全配置编译验证"环节薄弱。

#### 模式2：并发/同步修改是高风险区域(2/9)

- dispatch的syncall优化(8ade8c3e)
- fullmesh状态管理(e5988e2b)

MOE dispatch v2在后续又有多个同步修复提交，说明是持续技术痛点。

#### 模式3：大型变更缺乏分阶段策略

- ROPEV2: 27文件/3000行，一次性提交
- tilingkey模板化: 32文件/20000行，一次性提交

一次性提交导致回退代价极高，应拆分为多个独立可验证的小PR。

#### 模式4：commit message质量差

- e5988e2b只写了"revert"
- d6dd8838是空提交
- 多个revert的MR描述使用默认模板未填写

给后续维护者带来不必要的考古成本。

#### 模式5：CI门禁有效但不够前置

cann-robot自动创建revert说明有CI门禁机制，但"合入后发现问题再回退"的成本远高于"PR阶段就拦截"。应加强pre-merge编译检查和编译时间监控。

---

## Part 3: 跨类别系统性风险

### 风险1: 代码克隆/平行实现 -- "修一处漏一处"

本仓23次cmake缺陷和11次MC2 kernel缺陷的根因。

| 克隆对                                    | 相似度   | 影响                        |
|------------------------------------------|----------|-----------------------------|
| custom_build.cmake vs CMakeLists.txt(RTY) | 285行平行 | feature增减需两处同步        |
| func.cmake两个macro                       | 80%      | 新include需两处同步          |
| combine_v2.h vs _add_rms_norm.h            | 80%      | bug修复不同步致行为不一致    |
| aclnn_grouped_matmul.cpp 6个版本入口       | 主体相似  | 新参数/check需6处同步        |
| PFA/IFA各自TilingDataconvert              | 独立实现  | 新tiling字段遗漏             |

### 风险2: 跨文件隐式一致性约束

无编译期保证的跨文件约束是两仓高频缺陷源。

| 约束                                   | 涉及文件             | 失败频次 |
|----------------------------------------|---------------------|---------|
| tilingKey编码(host) vs 模板匹配(kernel) | tiling.cpp + kernel.h | ~20条   |
| workspace offset(host) vs 读取offset(kernel) | tiling.cpp + kernel.h | ~8条    |
| 宏参数顺序 vs 宏定义                    | matmul_all_reduce.cpp等 | ~5条   |
| host(tiling)侧与kernel侧计算不一致 [主仓] | op_host/ vs op_kernel/ | ~6条   |

### 风险3: GQA gSize缩放因子遗漏

跨6条独立commit反复出现(`d452dde6`, `81213d24`, `8a852918`, `3bb75678`, `e44028b1`, `16bc59a1`)。两仓库最频繁的单一具体根因。Q和KV的head数不同时，所有涉及Q-KV维度交叉计算的地方都需要gSize缩放。 [主仓]

### 风险4: 空tensor全链路处理

空tensor问题跨越API校验层和内部处理层。必须aclnn/infershape/tiling/kernel四层联动，任何一层遗漏都会导致问题。`<=0` vs `<0`的校验边界差异是最常见的具体表现。 [主仓]

### 风险5: 整数溢出/除零静默失败

几乎所有tiling和kernel文件都涉及多维度连乘或除法，普遍缺少溢出/除零检查。
- workspace计算: batch * head * seq * dim * sizeof(type) 六值连乘
- AlignUp/CeilDivision: 除数为0时静默返回0，下游使用错误参数
- 地址计算: uint32乘积在大规模场景溢出

### 风险6: 硬件平台分支不完整

新增硬件型号时多处遗漏是常见缺陷来源。
- socVersion字符串比较路由("Ascend910_95")
- `#if __CCE_AICORE__ == 220`预处理分支
- 编译选项/对齐要求/c:v比例平台差异

### 风险7: 大型Tiling类状态累积 [主仓]

prompt_flash_attention_tiling_v2.cpp有80+成员变量无统一reset机制，不同tiling路径可能读到上一次计算的残留状态。hotspot分析中排名第一的结构性风险。

### 风险8: 构建模式组合爆炸 [主仓]

ENABLE_EXPERIMENTAL、开源/闭源、多平台(310P/910B/Ascend950)组合下，CMake路径/条件编译/依赖关系难以维护。是revert的首要触发因素。

---

### 缺陷模式与Revert/热点的交叉验证

| 缺陷类别  | 阶段2频次 | 阶段3(Revert)佐证                   | 阶段4(热点)佐证                       |
|----------|----------|--------------------------------------|---------------------------------------|
| BUILD    | ~99      | 事件群7(3次CI/构建revert)             | custom_build.cmake Top1(23次)          |
| COND     | ~105     | 事件群6(cleancode删错误处理)           | tiling_v2.cpp dVBasicBlock=0未赋值     |
| BUF      | ~74      | 事件群4(性能优化buffer公式)            | fia_kernel_nonquant_mla.h sizeof链     |
| SYNC     | ~55      | 事件群4(cross core同步)               | combine_v2.h WaitDispatch浮点比较      |
| TKEY     | ~20      | 事件群1(6次系统性失败)                 | regbase.cpp 18参数tilingKey组装        |
| PLAT     | ~28      | 事件群2(新算子平台不兼容)              | matmul_all_reduce.cpp #if分支          |
| CPASTE   | ~18      | 事件群5(423572ba大杂烩MR)              | aclnn_grouped_matmul 6版本入口         |

---

### 缺陷分布可视化(主仓)

#### 按模块分布
```
attention  ====================================  42.3%  (103)
mc2        ================                      18.5%  (45)
moe        ==============                        13.3%  (32)
gmm        ========                               8.6%  (21)
build/cfg  ==============                        14.0%  (34)
other      ===                                    3.3%  (8)
```

#### 按代码层分布
```
op_host/tiling  ====================================  42.9%
op_kernel       ==========================            33.3%
aclnn/infershape ========                              9.5%
build/test cfg   ============                          14.3%
```

---

### UT/测试代码缺陷(附录, 主仓16条)

主仓特有的独立类别，dev版未单独分出。

#### 子模式

| 子模式             | 频次 | 说明                                      |
|-------------------|------|-------------------------------------------|
| 数据类型不匹配     | 6    | host类型与aclDataType不一致               |
| 路径/依赖错误      | 5    | UT include引用闭源路径、CMake路径错误      |
| UT与production不同步| 3   | 头文件名、命名空间、结构体名变更后UT未更新 |
| tiling数据未初始化  | 2    | GmAlloc后未赋值                           |

#### 典型案例

- `b9a02e9d` gen_data.py中position用float而非int64; sizeof(half)应为sizeof(float) [主仓]
- `29b08607` include引用level2/闭源路径, system()拷贝闭源脚本 [主仓]
- `8d72712c` 头文件路径/命名空间/期望输出字符串全部过期 [主仓]
- `f1a24bd` GmAlloc后tiling结构体21个字段未初始化 [主仓]

#### 审查检查点

- UT中CreateAclTensor的dataType与host数据的C++类型是否匹配
- sizeof()参数的类型是否与实际测试数据类型一致
- UT的include路径是否使用相对路径(而非闭源绝对路径)
- UT中tiling结构体的所有字段是否初始化
- production代码重命名/重构后，对应UT是否同步更新
