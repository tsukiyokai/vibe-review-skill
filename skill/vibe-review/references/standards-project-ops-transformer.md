## ops-transformer项目缺陷模式与审查规则

基于ops-transformer主仓（1323次提交，243条缺陷，占比18.4%）及dev分支（2820次非merge提交，528条代码缺陷，占比18.7%）的全量分析，提炼的76条高价值审查规则。每条规则均有commit证据和实际代码支撑。dev分支与主仓提交历史完全独立（零共享hash），缺陷规模更大(528 vs 243)，新增了TilingKey一致性(TKEY)、硬件平台兼容性(PLAT)两个主仓不显著的类别，反映dev分支迭代速度更快、跨平台适配更密集的特征。

ops-transformer是算子仓库，缺陷模式与通信库(hcomm/hccl)有本质差异。通信库侧重并发/协议/资源生命周期，算子仓库侧重精度/tiling/shape/数据类型/kernel参数。本文档的缺陷类别从缺陷数据中自然涌现，未套用通信库框架。

### 严重等级定义

| 等级 | 含义 | 影响范围                                    |
| ---- | ---- | ------------------------------------------- |
| P0   | 致命 | 硬件异常/CoreDump、数据损坏、内存越界、死锁 |
| P1   | 严重 | 特定配置崩溃、静默精度错误、功能不可用      |
| P2   | 一般 | 边界条件异常、构建失败、测试不通过          |
| P3   | 建议 | 代码质量、可维护性、潜在隐患                |

### 缺陷分布总览

| 序号 | 缺陷类别                 | 频次  | 占比  | 规则数 | 规则ID前缀 |
| ---- | ------------------------ | ----- | ----- | ------ | ---------- |
| 1    | 计算/Tiling参数错误      | 35+20 | 14.4% | 10     | CALC       |
| 2    | 构建/配置/部署缺陷       | 34+65 | 14.0% | 9      | BUILD      |
| 3    | 条件分支/逻辑覆盖不完整  | 30+75 | 12.3% | 7      | LOGIC      |
| 4    | 输入/参数校验缺陷        | 28    | 11.5% | 3      | VALID      |
| 5    | 空值/边界/特殊值处理缺失 | 25+60 | 10.3% | 7      | BOUND      |
| 6    | 硬件流水线同步缺陷       | 20+35 | 8.2%  | 4      | SYNC       |
| 7    | 整数类型安全缺陷         | 18+30 | 7.4%  | 4      | INT        |
| 8    | UT/测试代码缺陷          | 16    | 6.6%  | 3      | TEST       |
| 9    | API误用/接口错误         | 12+25 | 4.9%  | 7      | API        |
| 10   | 状态管理/赋值/传参缺陷   | 12+30 | 4.9%  | 8      | STATE      |
| 11   | Revert/回归引入          | 6+16  | 2.5%  | 3      | REV        |
| 12   | 硬件平台兼容性           | ~25   | -     | 2      | PLAT       |
| 13   | TilingKey一致性          | ~20   | -     | 2      | TKEY       |
| -    | 跨类别系统性风险         | -     | -     | 7      | SYS        |

---

### 类别一：计算/Tiling参数错误（35条，14.4%）

算子仓库最高频的缺陷类别。根本原因是Tiling-Kernel两阶段架构下，host侧计算的参数与kernel侧实际使用之间缺乏强一致性约束。覆盖workspace/buffer大小计算、地址偏移、除零、公式错误。

#### 规则CALC-01: workspace/buffer大小计算单位混淆

严重等级：P0

缺陷描述：对齐计算后的值已经是字节数，再除以sizeof(type)导致单位从"字节"退化为"元素数"，buffer分配不足引发内存越界。或反过来，元素数被当作字节数使用。维度因子遗漏、数据类型宽度错误也属此类。14条案例中错误源头分布：维度因子遗漏(3)、字节/元素/block数单位混淆(3)、数据类型宽度错误(2)、对齐遗漏(2)。dev仓中Duplicate的count传字节偏移（4倍越界）、InitOutput参数语义为元素数但传入字节数等案例进一步印证此模式的普遍性。

典型代码示例：

```cpp
// 缺陷代码 — attention相关tiling文件
// AlignUp后的值已是字节数，再除sizeof(half)使单位从字节变为元素数
bufferSize = AlignUp(rawSize, BLOCK_SIZE) / sizeof(half);  // 单位错误：字节 -> 元素

// 修复代码
bufferSize = AlignUp(rawSize, BLOCK_SIZE);  // AlignUp后直接使用，已是字节数
```

另一典型案例：

```cpp
// 缺陷代码 — 69698404
// 偏移量是int64(8字节)但按FP32_BYTES(4字节)计算，少一半
// 乘NUM_TWO(2)但实际有3组偏移量
workspaceSize = offsetCount * FP32_BYTES * NUM_TWO;

// 修复代码
workspaceSize = offsetCount * INT64_BYTES * NUM_THREE;
```

审查检查方法：
- 追踪每个buffer size变量的单位链：原始值是字节/元素/block数？每步运算后单位是否正确？
- AlignUp/CeilAlign后的值通常是字节数，不应再除以sizeof(type)
- 每个乘法因子的语义（维度数x类型宽度x组数）是否与实际数据对应
- 每个API调用核对文档确认参数单位（字节数需除以sizeof, 元素数不需要）
- Duplicate/InitOutput/DataCopy: 参数名含"count/num"则为元素数，含"size/bytes"则为字节数

关联commit: `9691bcc3`, `69698404`, `e233e106`, `84ca56e9`；dev仓： `798597da0a`, `45cfad32`

---

#### 规则CALC-02: tiling(host)侧与kernel侧计算不一致

严重等级：P0

缺陷描述：workspace大小在tiling侧用原始值计算，kernel侧用对齐后的值使用，两侧公式不一致导致kernel写入超出tiling分配的空间。tiling和kernel物理分离在不同目录/编译单元，缺乏编译期一致性约束。

典型代码示例：

```cpp
// 缺陷代码 — tiling侧 (op_host/)
workspaceSize += vHeadSize * sizeof(float);  // 用原始vHeadSize

// kernel侧 (op_kernel/)
bufOffset += AlignUp(vHeadSize, BYTE_BLOCK) * sizeof(float);  // 用对齐后的值
// tiling分配的空间 < kernel实际使用的空间 → 越界
```

```cpp
// 修复代码 — tiling侧统一使用对齐值
workspaceSize += AlignUp(vHeadSize, BYTE_BLOCK) * sizeof(float);
```

审查检查方法：
- workspace/buffer大小计算公式在tiling(op_host/)和kernel(op_kernel/)两侧是否完全一致
- 比较两侧代码时逐项对齐：维度值、对齐方式、类型宽度、段数
- 若一侧使用AlignUp，另一侧必须也使用相同的AlignUp

关联commit: `e233e106`

---

#### 规则CALC-03: CeilDiv与CeilAlign语义混淆

严重等级：P1

缺陷描述：CeilDiv(x, n)返回"x需要多少个n大小的块"（分块数），CeilAlign(x, n)返回"x对齐到n倍后的大小"（对齐后字节数）。两者差n倍，混用导致buffer分配偏小或循环次数偏大。dev仓中ubFactor = CeilDiv(rawSize, alignment)应为CeilAlign，结果差异达130016 vs 4063。

典型代码示例：

```cpp
// 缺陷代码 — 84ca56e9
// 需要对齐后的字节数，但错误使用CeilDiv得到block数
bufferSize = CeilDiv(rawSize, BLOCK_SIZE);  // 返回块数，比实际字节数小BLOCK_SIZE倍

// 修复代码
bufferSize = CeilAlign(rawSize, BLOCK_SIZE);  // 返回对齐后字节数
```

审查检查方法：
- 赋给buffer size/offset的变量应使用CeilAlign（对齐后大小）
- 赋给loop count/block count的变量应使用CeilDiv（分块数）
- 命名约定：xxxSize/xxxLen用CeilAlign，xxxCount/xxxNum用CeilDiv
- early return: 确认return前所有输出字段都已被赋值

关联commit: `84ca56e9`；dev仓： `586493dc`, `66337ddb6b`

---

#### 规则CALC-04: workspace多段地址偏移重叠

严重等级：P0

缺陷描述：workspace分为多个段（如softmax段、sink段、alibi段），新增条件性段后未更新所有下游段的起始地址计算，导致相邻段地址重叠、数据互踩。dev仓中workspace offset链(fia_kernel_nonquant_mla.h)的sizeof类型交替(half->float->half)时任一类型错误导致后续全部错位，进一步印证此模式的危险性。

典型代码示例：

```cpp
// 缺陷代码 — 3ca57b44
// 新增dsinksum段占用空间，但pseAlibi的起始地址未加上dsinksum大小
pseAlibiAddr = workspaceBase + softmaxSize;  // 遗漏了dsinksum段

// 修复代码
pseAlibiAddr = workspaceBase + softmaxSize + dsinkSumSize;  // 累加所有前序段
```

审查检查方法：
- 画出workspace内存布局图：段0起始 | 段0大小 | 段1起始 | 段1大小 | ...
- 新增段后，检查所有后续段的偏移计算是否都加上了新段大小
- 条件性段（仅某些模式下存在）需要在偏移计算中也做条件判断
- sizeof链：核对每段sizeof(type)与实际数据类型是否匹配（type交替half/float/half时尤其注意）
- offset累加：验证每个offset = 前一个offset + 前一个size，链条无断裂

关联commit: `3ca57b44`；dev仓： `e5817c02`, `29edf728`

---

#### 规则CALC-05: GQA场景gSize缩放因子遗漏

严重等级：P1

缺陷描述：GQA(Grouped Query Attention)中Q head数是KV head数的gSize倍。涉及Q-KV维度交叉计算时（preTokens、nextTokens、actualSeqLengthKV、workspace偏移等），遗漏gSize缩放因子。本仓库最频繁的单一具体根因，跨6条独立commit反复出现。

典型代码示例：

```cpp
// 缺陷代码 — d452dde6
// MLA/GQA路径中KV维度未乘gSize
tilingData.preTokens = preTokens;            // 缺少 * gSize
tilingData.nextTokens = nextTokens;          // 缺少 * gSize

// 修复代码
tilingData.preTokens = preTokens * gSize;
tilingData.nextTokens = nextTokens * gSize;
```

```cpp
// 缺陷代码 — 8a852918
// if-else只覆盖了isIFAMLA和非GQA，遗漏"GQA+非MLA+非BNSD+s1==1"组合
// actualS1Size缺少gSize缩放

// 修复代码
if (isGQA && !isIFAMLA && layoutType != BNSD && s1Size == 1) {
    actualS1Size = s1Size * gSize;
}
```

审查检查方法：
- 全局搜索preTokens/nextTokens/actualSeqLength等Q-KV交叉变量，逐一确认GQA场景下是否乘了gSize
- MLA/GQA的条件分支是否完整覆盖了所有layout x mode组合
- workspace偏移中涉及KV head维度的计算是否考虑了gSize

关联commit: `d452dde6`, `81213d24`, `8a852918`, `3bb75678`, `e44028b1`, `16bc59a1`

---

#### 规则CALC-06: GM偏移量整数溢出

严重等级：P0

缺陷描述：GM(Global Memory)偏移量使用uint32_t，当progress*tileLength超过4GB（2^32字节）时溢出，kernel读写错误地址。大shape或多核场景下易触发。

典型代码示例：

```cpp
// 缺陷代码 — 6b029d5f
uint32_t ind = progress * tileLength;  // 超过4GB时溢出
Gm_x[ind] = ...;

// 修复代码
uint64_t ind = static_cast<uint64_t>(progress) * tileLength;
Gm_x[ind] = ...;
```

审查检查方法：
- GM地址偏移计算变量不应使用32位整型
- 包含乘法的offset/size计算，中间结果是否可能超过uint32_t(4GB)
- 优先使用uint64_t存储GM偏移

关联commit: `6b029d5f`, `831ab170`

---

#### 规则CALC-07: tiling除数为零

严重等级：P0

缺陷描述：tiling计算中从UB容量或shape维度推导的中间变量可能为零，后续CeilDiv或除法运算触发CoreDump。常见场景：极端headDim导致UB容纳行数为0；空tensor的shape维度为0。dev仓中AlignUp/CeilDivision在除数为0时静默返回0，下游使用错误参数。

典型代码示例：

```cpp
// 缺陷代码 — 234c1c8b
// Ascend950+headDim>8192时，maxNPerLoopForUb计算结果为0
uint32_t maxNPerLoopForUb = ubSize / headDim / sizeof(float);  // = 0
uint32_t loopCount = CeilDiv(totalN, maxNPerLoopForUb);  // 除零CoreDump

// 修复代码
uint32_t maxNPerLoopForUb = ubSize / headDim / sizeof(float);
if (maxNPerLoopForUb == 0) {
    // fallback处理或报错
    return ge::GRAPH_FAILED;
}
```

审查检查方法：
- 所有除法/CeilDiv的除数变量，追溯其来源是否可能为零
- 特别关注从UB容量推导的中间变量(ubSize / headDim / typeSize)在极端参数下的值
- shape解析中空tensor（维度为0）场景必须在除法前拦截
- AlignUp/CeilDivision调用的第二参数必须>0
- 无符号减法(a-b)检查a>=b，否则下溢为极大值

关联commit: `234c1c8b`, `1f3291bf`；dev仓： `b7a0e9fced`, `479084fc`

---

#### 规则CALC-08: 整数除法不满足分配律

严重等级：P1

缺陷描述：整数除法/向上取整对求和不满足分配律：`CeilDiv(a+b, n)` != `CeilDiv(a, n) + CeilDiv(b, n)`。将各batch的seqlen简单累加后做一次CeilDiv，结果与逐batch独立CeilDiv再累加不同。

典型代码示例：

```cpp
// 缺陷代码 — 16bc59a1
// NTD下每batch seqLen不同，简单右移得到错误scale offset
scaleOffset = totalSeqLen >> log2BlockSize;  // sum(a_i)/B

// 修复代码 — 逐batch独立计算再累加
for (int b = 0; b < batchSize; b++) {
    scaleOffset += CeilDiv(seqLen[b], blockSize);  // sum(CeilDiv(a_i, B))
}
```

审查检查方法：
- 涉及整数除法的复合表达式，检查是否依赖了分配律
- NTD/变长序列场景中，确认是逐batch计算还是全局计算
- 用括号显式标注运算顺序，避免整数乘除的截断误差放大

关联commit: `16bc59a1`, `1ceed275`

---

#### 规则CALC-09: 数据搬运对齐不满足硬件要求

严重等级：P1

缺陷描述：DataCopy/DataCopyPad搬运长度未满足32B(256bit)对齐；NZ格式行数未2行对齐；Mmad的m/n/k维度未BLOCK_SIZE对齐。dev仓约18条此类缺陷。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 3cfe2f3b64)
copyTotalS = rawTotalS;  // 未32对齐, NZ格式scale搬运失败

// 修复
copyTotalS = CeilAlign(rawTotalS, ALIGN_32);
```

```cpp
// 缺陷代码 (dev仓 c439c4d7)
// Mmad的m维度未BLOCK_SIZE对齐, L0C矩阵乘精度问题
Mmad(dst, src, m, n, k);  // m未对齐

// 修复: m = CeilAlign(m, BLOCK_SIZE)
```

审查检查方法：
- DataCopy/DataCopyPad: 搬运长度是否32B对齐；不满足时用DataCopyPad（自动padding）
- NZ格式：行数需2行（16元素行）对齐
- Mmad: m/n/k需BLOCK_SIZE对齐

关联commit: dev仓 `3cfe2f3b64`, `446d15c9`, `c439c4d7`

---

#### 规则CALC-10: 不必要的对齐破坏地址计算

严重等级：P2

缺陷描述：对不需要对齐的值执行对齐操作，对齐后值偏大，破坏后续偏移量计算。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 73a82c36)
perTokenSize_ = CeilAlign(rawPerTokenSize, 512);  // 不需要512对齐
// 对齐后偏大 -> 后续地址 = base + perTokenSize_ * idx 偏移错误

// 修复: 去掉不必要的对齐
perTokenSize_ = rawPerTokenSize;
```

审查检查方法：
- 每个CeilAlign/RoundUp: 确认对齐是否确实需要（有硬件/DMA要求）
- 对齐后值作为偏移量乘数时：偏大会导致后续所有地址错位

关联commit: dev仓 `73a82c36`, `ac141d52`

---

### 类别二：构建/配置/部署缺陷（34条，14.0%）

非功能性缺陷中占比最大。核心矛盾是CMake构建系统的复杂度随算子数量增长而爆炸，加上experimental/开源/闭源多种构建模式，配置遗漏几乎不可避免。dev仓（~65条，12.3%）中cmake/custom_build.cmake（23次触及）和cmake/func.cmake（17次触及）是Top1和Top3缺陷热点文件，核心根因是平行实现导致的"修一处漏一处"。

#### 规则BUILD-01: 新算子CMake集成遗漏

严重等级：P2

缺陷描述：新算子集成时CMakeLists.txt缺少add_subdirectory、list(APPEND)、依赖声明、源文件引用，导致算子没有被编译进产物。CI的operator_list.yaml也可能遗漏。运行时dlopen报undefined symbol。

典型代码示例：

```cmake
# 缺陷代码 — 80329f3a (scatter_pa_cache)
# 1. CMakeLists.txt缺少依赖声明(set(...CACHE INTERNAL))
# 2. kernel源文件命名不符约定(scatter_pa_cache_apt.cpp应为scatter_pa_cache.cpp)
# 3. ops_transformer_operator_list.yaml未包含新算子

# 修复代码
# CMakeLists.txt — 添加依赖声明
set(scatter_pa_cache_depends attention/scatter_pa_kv_cache
    CACHE INTERNAL "Dependencies for scatter_pa_cache")
# kernel文件重命名为符合约定的名称
# operator_list.yaml中新增scatter_pa_cache条目
```

审查检查方法：
- 新算子PR必须包含CMakeLists.txt的构建配置修改（依赖声明、源文件注册等）
- kernel文件命名是否符合项目约定（前缀、后缀）
- CI配置（operator_list.yaml或等价文件）是否包含新算子

关联commit: `80329f3a`

---

#### 规则BUILD-02: CMake宏参数缺失

严重等级：P1

缺陷描述：add_modules_sources()等CMake宏的必选参数(OPTYPE/ACLNNTYPE)缺失，编译不报错但生成的符号名不完整，运行时dlopen报undefined symbol。

典型代码示例：

```cmake
# 缺陷代码 — a54af6d2e
add_modules_sources(${OP_NAME})  # 缺少OPTYPE和ACLNNTYPE参数

# 修复代码
add_modules_sources(${OP_NAME} OPTYPE ${OP_TYPE} ACLNNTYPE ${ACLNN_TYPE})
```

审查检查方法：
- CMake宏调用的参数是否完整，对照宏定义的必选参数列表逐项检查
- 编译通过不代表集成正确，需要运行时验证dlopen/符号解析

关联commit: `a54af6d2e`

---

#### 规则BUILD-03: ENABLE_EXPERIMENTAL模式路径适配错误

严重等级：P2

缺陷描述：ENABLE_EXPERIMENTAL模式下源文件在experimental/子目录中，但EXISTS检查和include路径拼接缺少experimental/前缀。同一个函数连续两次修复说明此路径问题极易遗漏。

典型代码示例：

```cmake
# 缺陷代码 — fbade778b
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_dir")  # 缺少experimental/前缀
    op_add_depend_directory("op_dir")

# 修复代码
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/experimental/op_dir")
    op_add_depend_directory("experimental/op_dir")
```

审查检查方法：
- ENABLE_EXPERIMENTAL模式下所有路径拼接是否正确加了experimental/前缀
- include路径的-I顺序是否正确（项目路径优先于系统路径）

关联commit: `fbade778b`, `b0bf9fa2`, `ef22ac8f1`

---

#### 规则BUILD-04: 安装脚本误删其他组件文件

严重等级：P1

缺陷描述：安装脚本中entity="true"或类似选项导致安装时先清除目标目录，误删该目录下其他组件的文件。

典型代码示例：

```xml
<!-- 缺陷代码 — 83be205f -->
<install entity="true" dest="/usr/lib64/">  <!-- entity=true会清除目录 -->
    <file src="libop.so"/>
</install>

<!-- 修复代码 -->
<install dest="/usr/lib64/">  <!-- 去掉entity，不清除目录 -->
    <file src="libop.so"/>
</install>
```

审查检查方法：
- 安装/卸载脚本是否会影响目标目录下其他组件的文件
- entity/clean类选项是否必要

关联commit: `83be205f`

---

#### 规则BUILD-05: CMake OP_NAME/target名复制粘贴错误

严重等级：P2

缺陷描述：新增算子时从已有算子复制CMakeLists.txt，OP_NAME等关键标识符未替换为新算子名。dev仓约12条此类缺陷。

典型代码示例：

```cmake
# 缺陷代码 (dev仓 5508fbb6c9)
# moe_init_routing/op_host/CMakeLists.txt
set(OP_NAME MoeFinalizeRouting)  # 从finalize复制未改, 应为MoeInitRouting

# 修复代码
set(OP_NAME MoeInitRouting)
```

审查检查方法：
- CMakeLists中OP_NAME、OPTYPE与所在目录名/算子实际名称逐字核对
- 新增算子PR中搜索diff是否包含其他算子名（复制来源的残留）
- 检查target名是否与source文件名、namespace一致

关联commit: dev仓 `5508fbb6c9`, `d69eec6dab`

---

#### 规则BUILD-06: 新平台编译选项/分支未覆盖

严重等级：P1

缺陷描述：新增硬件平台（如ascend910_95/A5）时，多处CMakeLists/cmake中的平台条件分支(ASCEND_COMPUTE_UNIT)未添加新平台。dev仓中8个moe CMakeLists一次性遗漏A5平台分支(eeba666256)是典型案例。约10条。

典型代码示例：

```cmake
# 缺陷代码 (dev仓 eeba666256)
# 8个moe相关CMakeLists均缺少ascend910_95分支
if(ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
    target_compile_options(...)
elseif(ASCEND_COMPUTE_UNIT STREQUAL "ascend310p")
    target_compile_options(...)
# 缺少: elseif(ASCEND_COMPUTE_UNIT STREQUAL "ascend910_95")

# 修复: 添加910_95分支及其编译选项
```

审查检查方法：
- 新平台PR: 全局搜索ASCEND_COMPUTE_UNIT/socVersion条件，逐一确认新平台已覆盖
- 统计搜索命中数与实际修改数是否一致（遗漏=命中但未修改）
- 检查新平台的编译选项是否与最近似的已有平台一致

关联commit: dev仓 `eeba666256`, `e9ec0142`

---

#### 规则BUILD-07: auto-sync编译选项与kernel代码冲突

严重等级：P1

缺陷描述： --cce-auto-sync=on时编译器自动插入同步屏障，但kernel代码中已手动插入同步指令。两者叠加导致重复同步（性能下降）或auto-sync被关闭但手动同步不完整（数据竞争）。

典型代码示例：

```cmake
# 缺陷代码 (dev仓 38606b2b)
# auto-sync=on, 但kernel中手动调用SetFlag/WaitFlag
target_compile_options(... --cce-auto-sync=on)
# kernel: SetFlag<HardEvent::V_MTE3>(event); WaitFlag(event);
# 冲突: 编译器也会插入同步, 可能导致死锁或不必要停顿

# 修复: 手动同步的kernel必须设置 --cce-auto-sync=off
```

审查检查方法：
- diff中出现auto-sync设置变更时，检查对应kernel是否有手动SetFlag/WaitFlag
- kernel中新增手动同步时，检查CMakeLists中auto-sync是否为off
- 同一算子的不同平台分支auto-sync设置是否一致

关联commit: dev仓 `38606b2b`, `e9ec0142`

---

#### 规则BUILD-08: include路径变更未同步

严重等级：P2

缺陷描述：头文件路径重组后，引用该头文件的cmake/源码未同步更新。include guard名冲突、依赖缺失也属此类。dev仓约12条。

审查检查方法：
- include路径变更时grep所有引用该头文件的文件，确认路径同步更新
- 新增include guard时搜索同名guard是否已存在
- 跨仓头文件路径变更须确认上游变更已就绪

关联commit: dev仓 `05c16e686c`, `bb90fa511`

---

#### 规则BUILD-09: 黑名单/分包/注册配置未更新

严重等级：P2

缺陷描述：新增算子后operator_list.yaml、classify_rule.yaml、A5_OPS_BLACK_LIST等配置文件未同步更新。dev仓约10条。

审查检查方法：
- 新增算子PR中检查：classify_rule.yaml、operator_list.yaml是否已添加新算子
- 算子从黑名单中移除时确认对应平台编译/测试已通过
- A5_OPS_BLACK_LIST元素末尾分号格式是否正确（func.cmake中list(FIND)依赖此格式）

关联commit: dev仓 `eeba666256`, `422548259b`

---

### 类别三：条件分支/逻辑覆盖不完整（30条，12.3%）

算子仓库特有的高频模式。根因是多layout(BNSD/BSND/TND/NTD) x多模式(GQA/MHA/MQA) x多量化(FP8/INT8/Perblock) x多sparse mode的组合空间爆炸，开发者通常只验证常见组合。dev仓（~75条，14.2%）进一步印证此类缺陷的普遍性，OP_CHECK_IF宏的语义反转和逻辑运算符混淆是dev仓新增的高频子模式。

#### 规则LOGIC-01: layout分支遗漏

严重等级：P1

缺陷描述：新增layout类型（如NTD）时未在所有layout判断分支做系统排查。NTD是最容易被遗漏的layout——多处代码只检查了TND而遗漏NTD。遗漏点涵盖stride计算、workspace大小、IFA开关、isGqa设置等。dev仓中三处遗漏NTD(b4b9198148)是典型案例。

典型代码示例：

```cpp
// 缺陷代码 — ad627275f
// SoftmaxLseCopyOut中Lse输出stride的GQA处理仅在TND生效，遗漏NTD
if (layoutType == TND) {
    lseStride *= gSize;  // NTD也需要此处理但被遗漏
}
// CheckIO中NTD不应走IFA路径但缺少排除条件

// 修复代码
if (layoutType == TND || layoutType == NTD) {
    lseStride *= gSize;
}
```

审查检查方法：
- 引入新layout类型时，全局搜索所有layout判断分支（包括workspace、stride、offset、InferShape、Check函数）
- 确认if-else if链或switch-case是否覆盖了所有有效layout
- NTD和TND通常共享大部分逻辑，新增TND处理时检查NTD是否同步
- 使用if constexpr限制Layout专属逻辑只在目标Layout下执行

关联commit: `ad627275f`, `fcff1be7`, `e5dbe5b5c`；dev仓： `b4b9198148`, `a0077ae9`, `82dbcd14`

---

#### 规则LOGIC-02: 模式组合未覆盖

严重等级：P1

缺陷描述：多维枚举（GQA/MHA x BNSD/TND x量化/非量化x对齐/非对齐）的组合空间中遗漏"边角组合"。if-else只处理了常见路径，特殊组合穿透到默认分支产生错误。dev仓中新增枚举值（sparseMode、quantMode、Layout等）或运行模式后，代码中多处if/switch分支未覆盖新值的模式约20条。

典型代码示例：

```cpp
// 缺陷代码 — 8a852918
if (isIFAMLA) {
    // MLA路径
} else if (!isGQA) {
    // 非GQA路径
} else {
    // 默认路径 — 但遗漏了GQA+s1Size==1+非BNSD组合
}

// 修复代码 — 显式枚举GQA边角组合
if (isIFAMLA) { ... }
else if (isGQA && s1Size == 1 && layoutType != BNSD) {
    actualS1Size = s1Size * gSize;  // 新增分支
}
else if (!isGQA) { ... }
else { ... }
```

审查检查方法：
- 多模式组合用表格系统枚举所有有效组合，对照if-else链确认全覆盖
- if-else中更特化的条件必须在更泛化的条件之前
- 默认分支(else)须明确是"兜底"还是"不应到达"，后者应加assert/log
- 新增枚举值时全局搜索该枚举类型的所有switch/if-else，确认新值已覆盖
- 检查default/else分支是否会对新值产生意外行为

关联commit: `8a852918`, `b48e75b7c`；dev仓： `c8cf3f011d`, `b4b9198148`, `a0077ae9`

---

#### 规则LOGIC-03: 条件守卫缺失

严重等级：P1

缺陷描述：可选输入/输出的null检查、tpWorldSize==1退化场景、不满足对齐条件时的fallback路径缺失。代码直接操作可能不存在的tensor或走不适用的公式化路径。

典型代码示例：

```cpp
// 缺陷代码 — f91e8dee
// kernel中alltoallOut数据拷贝缺少isAlltoallOut条件判断
DataCopy(alltoallOut, ...);  // alltoallOut可能不存在

// 修复代码
if (isAlltoallOut) {
    DataCopy(alltoallOut, ...);
}
```

```cpp
// 缺陷代码 — a75a92e2
// orgMValue不满足PERBLOCK_SIZE*rankDim整数倍时仍走公式化切分
splitSize = orgMValue / (PERBLOCK_SIZE * rankDim);  // 不整除时结果错误

// 修复代码
if (orgMValue % (PERBLOCK_SIZE * rankDim) != 0) {
    // fallback到通用切分逻辑
}
```

审查检查方法：
- 可选输入/输出在kernel和host是否都有null/flag守卫
- tpWorldSize==1时是否需要豁免通信操作
- 不满足对齐/整除条件时是否有fallback路径

关联commit: `f91e8dee`, `a75a92e2`

---

#### 规则LOGIC-04: 循环内布尔赋值覆盖

严重等级：P1

缺陷描述：循环内对布尔标志用`=`赋值，每次迭代覆盖前一次的值。语义应为"是否存在某条件"（需要`|=`），但实际只保留了最后一次迭代的结果。

典型代码示例：

```cpp
// 缺陷代码 — f5f79a3e
for (int b = 0; b < batchSize; b++) {
    isSeqExistZero = (qLen == 0 || kvLen == 0);  // 每次覆盖，前面的true被后面的false冲掉
}

// 修复代码
for (int b = 0; b < batchSize; b++) {
    isSeqExistZero |= (qLen == 0 || kvLen == 0);  // 累积，只要有一个batch为空就为true
}
```

审查检查方法：
- 循环内对布尔标志的赋值是`=`还是`|=`
- 语义是"最后一次迭代的结果"还是"存在某条件"——后者必须用`|=`
- 同理，"所有迭代都满足"需要用`&=`

关联commit: `f5f79a3e`

---

#### 规则LOGIC-05: OP_CHECK_IF/OP_CHECK条件语义反转

严重等级：P0

缺陷描述：OP_CHECK_IF宏的设计语义是"参数为true时触发失败"，但开发者常将"合法条件"直接传入（不加取反），导致合法输入被拒绝、非法输入被放行。dev仓528条缺陷中此模式出现约12次，是条件类缺陷的第一大子类。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 b6ec4bfe86)
// CheckSameShape返回true表示shape合法
OP_CHECK_IF(CheckSameShape(qShape, kShape), ...);  // true=合法 -> 触发失败 -> 合法被拒

// 修复代码
OP_CHECK_IF(!CheckSameShape(qShape, kShape), ...);  // true=不合法 -> 触发失败 -> 正确
```

审查检查方法：
- 逐个核对OP_CHECK_IF参数的语义方向：条件为true应表示"错误状态"
- 当参数是函数调用（如CheckXxx()）时，确认函数返回true的含义是否与CHECK宏的fail语义一致
- 搜索diff中所有OP_CHECK_IF/OP_CHECK调用，特别关注直接传入bool函数（未取反）的用法

关联commit: dev仓 `b6ec4bfe86`, `479084fc`, `c8cf3f011d`

---

#### 规则LOGIC-06: 逻辑运算符OR/AND混淆导致恒真/恒假

严重等级：P0

缺陷描述： `(x != A || x != B)` 恒为true、`(x == A && x == B)` 恒为false是经典逻辑错误。dev仓多次出现在多枚举值校验场景中。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 67f1fffc)
if (dimNum != DIM3 || dimNum != DIM4) {  // 恒true: 不等于3或不等于4，任何值都满足其一
    return GRAPH_FAILED;
}

// 修复代码
if (dimNum != DIM3 && dimNum != DIM4) {  // 既不是3也不是4时才报错
    return GRAPH_FAILED;
}
```

审查检查方法：
- 搜索diff中`!= ... ||`和`== ... &&`模式，逐一验证逻辑意图
- 多枚举值OR连接时确认每个值是否属于同一约束集
- 两个不等式用OR连接时考虑是否应为AND

关联commit: dev仓 `67f1fffc`, `b6bdcb97`

---

#### 规则LOGIC-07: Layout专属逻辑在错误Layout下执行

严重等级：P1

缺陷描述：某些计算逻辑仅对特定Layout有效（如累加batch偏移仅BSH需要），但条件范围过宽导致对所有Layout执行。dev仓约6条此类缺陷。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 0166982e)
// BSND不需累加batch偏移, 但对所有layout都执行了
gmOffset += batchIdx * seqLen * headDim;  // BSND下batchIdx已编码在shape中

// 修复: 仅BSH/BNSD下执行
if (layout == BSH || layout == BNSD) {
    gmOffset += batchIdx * seqLen * headDim;
}
```

审查检查方法：
- 偏移量/stride计算：确认计算公式是否对当前Layout成立
- 不同Layout下batch/seq/head维度的位置不同，核对取值来源
- TND下tSize应用GetTSize()而非batchSize*qSeqSize

关联commit: dev仓 `0166982e`, `ec9278e0`

---

### 类别四：输入/参数校验缺陷（28条，11.5%）

包含校验缺失（未拦截不合法输入）和过度校验（拦截了合法输入）两个对立方向。核心矛盾是算子支持的feature组合快速增长时校验代码未同步更新。

#### 规则VALID-01: dtype白名单校验缺失

严重等级：P1

缺陷描述：算子入口对输入tensor的dtype没有做合法性前置检查。不支持的dtype穿透校验层进入计算，产生未定义行为或误导性错误信息。

典型代码示例：

```cpp
// 缺陷代码 — 8ceb72b6
// 量化路径入口直接进入分支逻辑，未校验x/weight的dtype是否在支持列表中
if (isPerTokenQuant) { ... }
else if (isPerBlockQuant) { ... }
// 不支持的dtype静默走入错误分支

// 修复代码 — 入口处新增白名单校验
if (std::find(X_DTYPE_SUPPORT_LIST.begin(), X_DTYPE_SUPPORT_LIST.end(), xDtype)
    == X_DTYPE_SUPPORT_LIST.end()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x dtype %s is not supported", ...);
    return false;
}
```

审查检查方法：
- 算子入口是否有全量dtype白名单校验（而非仅在子分支内局部检查）
- 白名单是否覆盖了所有新增的dtype支持
- 不支持的dtype是否有明确的错误信息

关联commit: `8ceb72b6`, `f8882f78`

---

#### 规则VALID-02: 空tensor过度拦截

严重等级：P1

缺陷描述：校验条件用`<=0`拦截了合法的N=0空tensor场景。伪量化等场景下空tensor是合法输入，应放行而非报错。同时IsNonEmpty()判断也可能将空shape误判为不存在。dev仓中校验过严误拦合法输入约15条，表现为新功能/新数据类型上线后被已有校验误拦。

典型代码示例：

```cpp
// 缺陷代码 — 54692072
OP_CHECK_IF(weightNDim_ <= 0,  // <=0 拦截了合法的 N=0
    OP_LOGE(context->GetNodeName(), "The n dim value should be positive..."),
    return false);

// 修复代码
OP_CHECK_IF(weightNDim_ < 0,   // <0 只拦截负值，放行 N=0
    OP_LOGE(context->GetNodeName(), "The n dim value should not be negative..."),
    return false);
```

审查检查方法：
- 校验条件是`<0`还是`<=0`？0是否是合法值？
- 空tensor（维度=0）是否作为合法输入被正确处理而非过度拦截
- 新增feature/mode后，现有校验路径是否需要放宽
- 新增数据类型/格式时检查已有校验是否需要扩展白名单

关联commit: `54692072`, `38c7162c`, `aa9faa8c`；dev仓： `c8cf3f011d`, `1eb3f0ce04`

---

#### 规则VALID-03: 校验顺序依赖错误

严重等级：P2

缺陷描述：校验函数的执行顺序不满足逻辑依赖。典型案例：batch dim==1的校验在收集tensorlist的循环中执行，但空tensor的dim(0)==0 != 1会触发误拦截——应先做空tensor判断再做维度值校验。

典型代码示例：

```cpp
// 缺陷代码 — 44e84d52
while (GetDynamicInputShape(KEY_INDEX, i) != nullptr) {
    kTensorList[i] = GetDynamicInputShape(KEY_INDEX, i);
    OP_CHECK_IF(kTensorList[i]->GetDim(0) != 1, ...);  // 空tensor的dim(0)==0，误拦截
    i++;
}

// 修复代码 — 先收集再判空再校验
while (GetDynamicInputShape(KEY_INDEX, i) != nullptr) {
    kTensorList[i] = GetDynamicInputShape(KEY_INDEX, i);
    i++;
}
if (CheckEmptyTensorList(kTensorList, i)) return true;  // 空tensor先放行
for (int j = 0; j < i; j++) {
    OP_CHECK_IF(kTensorList[j]->GetDim(0) != 1, ...);  // 此时确认非空
}
```

审查检查方法：
- 校验函数的执行顺序是否满足逻辑依赖
- 空tensor判断须在维度值判断之前
- CHECK宏的条件语义（true=通过vs true=拦截）是否清晰一致

关联commit: `44e84d52`, `9b12d59b`

---

### 类别五：空值/边界/特殊值处理缺失（25条，10.3%）

与类别四（输入校验）的区别：类别四是API边界处的参数合法性检查，类别五是计算逻辑内部对极端值的处理。空tensor问题跨越两个类别。dev仓（~60条，11.4%）中GM访问前缺边界检查、UB buffer复用缺同步/清零、DataCopy参数越界是新增的高频子模式。

#### 规则BOUND-01: 空tensor全链路处理缺失

严重等级：P0

缺陷描述：空tensor处理不是单层能解决的，必须aclnn/infershape/tiling/kernel四层全链路配套。任何一层遗漏都会导致问题：infershape未赋输出shape、tiling触发除零、kernel输出含脏数据。

典型代码示例：

```cpp
// 缺陷代码 — b5d7c0fe
// InferShape层：空rope tensor的shape未拷贝到输出
if (queryRopeShape != nullptr && queryRopeShape->GetShapeSize() != 0) {
    *dqRopeShape = *queryRopeShape;  // 空tensor的ShapeSize==0，跳过
}
// Tiling层：完全缺少rope输出的清零逻辑
// Kernel层：输出tensor含脏数据

// 修复代码 — 四层联动
// InferShape层：去掉GetShapeSize()!=0条件
if (queryRopeShape != nullptr) {
    *dqRopeShape = *queryRopeShape;
}
// Tiling层：新增rope输出的分核清零逻辑
// Kernel层：空tensor时跳过计算，输出已由tiling层处理清零
```

审查检查方法：
- 空tensor（维度=0）是否在aclnn/infershape/tiling/kernel四层全链路处理
- infershape中输出shape是否被正确赋值（即使输入为空）
- tiling中空tensor是否触发除零（shape维度为0作为除数）
- kernel中空tensor输出是否初始化（ZerosLike或清零）

关联commit: `b5d7c0fe`, `236a5fdb`, `78e07300`

---

#### 规则BOUND-02: 空指针格式化崩溃

严重等级：P0

缺陷描述：日志宏OP_LOGE中%s格式化传入nullptr导致崩溃。常见于将null检查与后续校验合并为一个条件表达式，日志在null时仍尝试格式化指针。

典型代码示例：

```cpp
// 缺陷代码 — d3aa4960
// nullptr检查和strcmp合并为一个条件，但OP_LOGE始终对commModePtr求值
OP_TILING_CHECK((commModePtr == nullptr || !(strcmp(commModePtr, "aiv") == 0)),
    OP_LOGE("commMode is %s", commModePtr),  // commModePtr为nullptr时崩溃
    return ge::GRAPH_FAILED);

// 修复代码 — 拆分为两个独立CHECK
OP_TILING_CHECK(commModePtr == nullptr,
    OP_LOGE("commMode is nullPtr."),           // 不引用指针
    return ge::GRAPH_FAILED);
OP_TILING_CHECK(strcmp(commModePtr, "aiv") != 0,
    OP_LOGE("commMode is %s", commModePtr),    // 此时指针安全
    return ge::GRAPH_FAILED);
```

审查检查方法：
- OP_LOGE/OP_LOGW中%s是否可能传入nullptr
- 指针null检查不应与后续使用合并到同一个CHECK宏中
- null判断通过后才能在日志中引用指针

关联commit: `d3aa4960`

---

#### 规则BOUND-03: 尾块边界条件处理错误

严重等级：P1

缺陷描述：分块处理中最后一块（尾块）的大小应为min(remainSize, blockSize)，但代码使用固定blockSize或未考虑lastBlockSize，导致MM参数超出有效范围。dev仓中尾块搬运用固定块大小而非实际剩余大小，超界覆写相邻scale数据的案例进一步印证此模式。

典型代码示例：

```cpp
// 缺陷代码 — b59cf016
// singleN未考虑lastBlockSize，MM参数超出有效范围
mmParams.N = singleN;  // 尾块时singleN > remainN

// 修复代码
mmParams.N = (isLastBlock) ? lastBlockN : singleN;
```

审查检查方法：
- 循环中最后一块的大小是否min(remainSize, blockSize)
- 分块参数传递给下游时，尾块是否使用了实际剩余大小
- 减法后的值是否做了下界保护（负值clamp到0）

关联commit: `b59cf016`, `ccd9bc9b`；dev仓： `2efc569c`

---

#### 规则BOUND-04: 特殊值(inf/NaN)未处理

严重等级：P1

缺陷描述：多SP域场景下LogSumExp(lse)可能为+inf，exp(+inf - (+inf)) = NaN污染输出。需要检测+inf并替换为安全值（-inf或0）。

典型代码示例：

```cpp
// 缺陷代码 — 4801ecb2
float result = exp(lse_a - lse_b);  // lse_a=+inf, lse_b=+inf时 → exp(NaN) → NaN

// 修复代码
if (lse_a == INFINITY) lse_a = -INFINITY;  // +inf替换为-inf，exp(-inf - x)=0安全
float result = exp(lse_a - lse_b);
```

审查检查方法：
- exp/log/div/sqrt等数学运算是否处理了0/+inf/-inf/NaN边界
- 多SP域合并场景中，各域的中间结果是否可能为inf

关联commit: `4801ecb2`

---

#### 规则BOUND-05: GM访问前缺边界检查(check-before-access)

严重等级：P0

缺陷描述：GetValue/SetValue等GM访问操作在边界检查之前执行，index越界时直接访问非法地址。dev仓约10条此类缺陷。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 af72ccb938)
auto val = blockTablesGm_.GetValue(offset);  // 先读
if (offset >= blockTableWidth_) { break; }   // 后检查 -> 已越界

// 修复代码
if (offset >= blockTableWidth_) { break; }   // 先检查
auto val = blockTablesGm_.GetValue(offset);  // 后读
```

审查检查方法：
- 每个GM GetValue/SetValue调用前是否有边界检查(index < size)
- 边界检查与GM访问之间无条件跳转（不会被跳过）
- 循环中的GM访问：循环条件是否包含边界约束

关联commit: dev仓 `af72ccb938`

---

#### 规则BOUND-06: UB buffer复用缺同步/清零

严重等级：P1

缺陷描述：两个逻辑buffer映射同一物理UB区间时缺同步屏障；buffer复用前未清零导致脏数据残留；清零长度小于分配大小。dev仓约12条。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 f55d2a35)
// statusTensor清零长度 < buffer大小, 尾部脏数据
Duplicate(statusTensor, (half)0, partialSize);  // partialSize < bufferSize

// 修复: 清零整个buffer
Duplicate(statusTensor, (half)0, bufferSize);
```

审查检查方法：
- buffer复用：新用途开始前是否已清零或完全覆写
- 清零范围：Duplicate/memset的长度 >= buffer实际分配大小
- 同一物理UB被两个逻辑buffer使用时，中间必须有PipeBarrier

关联commit: dev仓 `f55d2a35`, `64f59aae`, `949f8193fd`

---

#### 规则BOUND-07: DataCopy参数/边界越界

严重等级：P0

缺陷描述：DataCopy搬运长度超buffer分配大小、stride超uint16/uint32上限截断、尾块搬运用固定块大小而非实际剩余大小。按RoundUp(headSize)搬运，非32B对齐时超出GM连续空间。dev仓约12条。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 1dbd7856)
// 按RoundUp(headSize)搬运, 非32B对齐时超出GM连续空间
DataCopy(ubBuf, gmBuf, RoundUp(headSize, 32));

// 修复: 使用DataCopyPad处理非对齐尾部
```

审查检查方法：
- DataCopy搬运长度 <= 目标buffer分配大小
- stride参数不超uint16(65535)/uint32上限；超出时用DataCopyExtParams
- 尾块/边界块搬运需用min(blockSize, remainSize)

关联commit: dev仓 `2efc569c`, `1dbd7856`, `cefdc8776e`

---

### 类别六：硬件流水线同步缺陷（20条，8.2%）

AscendC NPU特有的高危缺陷类别。可审查性最低——需要理解MTE2/MTE3/Vector/Scalar四条流水线的数据依赖关系。但模式高度可归纳。dev仓（~35条，6.6%）中流水线屏障缺失、HardEvent类型错误、SetFlag/WaitFlag不配对是共性模式，与主仓高度一致。

#### 规则SYNC-01: 流水线切换缺失barrier

严重等级：P0

缺陷描述：Vector/MTE2/MTE3/Scalar流水线切换时遗漏对应的PipeBarrier或SyncFunc。最常见遗漏点是DataCopy/DataCopyPad前后。典型场景：处理完query(Vector)后切换到key的DataCopy(MTE2)，只做了MTE3→V同步而遗漏V→MTE2同步，Vector可能仍在写UB导致数据竞争。主仓13条，dev仓约10条。

典型代码示例：

```cpp
// 缺陷代码 — 984d8a72
this->MTE3ToVSync();
// 紧接DataCopy从GM读key到UB（MTE2操作）
DataCopy(inQQueBeforeCastLocal, key_in_GM[key_in_offset], ...);
// Vector可能仍在处理query，MTE2读同一块UB → 数据竞争

// 修复代码
this->MTE3ToVSync();
this->VToMTE2Sync();  // 新增：确保Vector写完UB后MTE2才读
DataCopy(inQQueBeforeCastLocal, key_in_GM[key_in_offset], ...);
```

审查检查方法：
- DataCopy/DataCopyPad操作前后是否有对应的pipeline barrier
  - V→MTE2: VToMTE2Sync()
  - V→MTE3: PipeBarrier<PIPE_V>
  - MTE3完成后：PipeBarrier<PIPE_MTE3>
- 切换处理不同tensor并复用同一块local memory时，是否补全了所有必要sync
- 在kernel代码中搜索DataCopy，确认每个DataCopy前后的pipeline状态
- 特别注意：buffer被写入GM后立即复用的场景

关联commit: `984d8a72`, `368c7325`, `d1d95dfb`；dev仓： `d2601de86d`, `949f8193fd`, `37105bc0`

---

#### 规则SYNC-02: 同步方向反转

严重等级：P0

缺陷描述：SyncFunc模板参数的方向语义：`SyncFunc<A_B>`表示"A完成后B可以开始"。写反方向导致完全无效的同步——程序等待了错误的流水线。dev仓中HardEvent类型指定了生产者->消费者的管道对，类型选错导致同步的是错误的管道对。

典型代码示例：

```cpp
// 缺陷代码 — 662f162c (moe_distribute_dispatch_v2_full_mesh.h)
// else分支中放置了SyncFunc<V_MTE2>()，语义为"Vector完成后MTE2可以开始"
// 但数据流是MTE2先加载数据、Vector后消费，应该是MTE2→V方向
SyncFunc<AscendC::HardEvent::V_MTE2>();  // 方向反了，等待了错误的流水线

// 修复代码 — 删除方向错误的同步，在正确位置补上PipeBarrier
// if分支: 保留已有的SyncFunc<MTE2_V>()，新增PipeBarrier<PIPE_V>()
// else分支: 删除错误的SyncFunc<V_MTE2>()
```

审查检查方法：
- SyncFunc模板参数的方向（A_B表示"A完成后B可以开始"）是否与数据流一致
- 画出当前位置的数据流向图：哪条流水线产出数据，哪条消费数据
- 产出者→消费者的方向必须与SyncFunc模板参数匹配
- 方向错误的同步不仅无效，还可能掩盖真正需要的同步点
- HardEvent类型：核对模板参数中的管道对是否与实际数据流方向一致

关联commit: `662f162c`；dev仓： `d5c79c44c7`

---

#### 规则SYNC-03: event信号Set/Wait不配对

严重等级：P0

缺陷描述：SetFlag和WaitFlag不严格配对，导致event信号积压或丢失。常见于if-else分支中两个分支分别SetFlag不同event，但WaitFlag只等了其中一个。未Set即Wait导致死锁；多余Wait导致流水线不必要停顿；Set后遗漏对应Wait导致同步缺失。

典型代码示例：

```cpp
// 缺陷代码 — a58c8a0b9
// 分配了eventIdVToMte2A和eventIdVToMte2Sink两个event
// hasSink分支SetFlag(eventIdVToMte2Sink)
// 非hasSink分支SetFlag(eventIdVToMte2A)
// 但WaitFlag只等eventIdVToMte2Sink → 非hasSink路径的信号积压

// 修复代码 — 统一使用一个eventID
// hasSink和非hasSink做互斥SetFlag（通过条件控制）
// 统一WaitFlag(eventIdVToMte2A)，保证Set/Wait一一配对
if (loopIdx < total - 1 && !hasSink) {
    SetFlag<V_MTE2>(eventIdVToMte2A);
}
if (loopIdx < total - 1 && hasSink) {
    SetFlag<V_MTE2>(eventIdVToMte2A);  // 统一用同一个event
}
```

审查检查方法：
- SetFlag和WaitFlag是否严格配对——每个Set都有且仅有一个对应的Wait
- 条件分支中不同分支的Set/Wait是否一致
- 避免分配多余的eventID，同一对生产者-消费者用一个event
- 同一eventId的Set和Wait之间无条件跳转（某分支只Set不Wait或反之）
- 条件分支中的event使用：if分支Set了event则else分支也必须处理

关联commit: `a58c8a0b9`；dev仓： `ee07e5dd25`

---

#### 规则SYNC-04: SyncAll模板参数误用

严重等级：P0

缺陷描述：SyncAll<true>和SyncAll<false>语义不同，用错导致同步不充分或多流死锁。使用SyncAll的kernel在host tiling侧必须设置SetScheduleMode(BATCH_MODE_SCHEDULE)，遗漏会导致部分路径死锁。

典型代码示例：

```cpp
// 缺陷代码 — 5dce387d
SyncAll<true>();   // 含义A
SyncAll<false>();  // 含义B — 混淆使用导致同步不充分

// 缺陷代码 — 65cafae0
// kernel中使用SyncAll，但tiling侧部分路径缺少SetScheduleMode设置
// → 多流模式下SyncAll行为异常导致死锁
```

审查检查方法：
- SyncAll<true>和SyncAll<false>的语义差异是否被正确理解
- 使用SyncAll的kernel，host tiling是否设置了SetScheduleMode(BATCH_MODE_SCHEDULE)
- 同步ID/flag常量值是否全局唯一（特别是super-kernel场景）

关联commit: `5dce387d`, `65cafae0`, `5644a7b`

---

### 类别七：整数类型安全缺陷（18条，7.4%）

三种系统性子模式，每种都有固定的检测和修复方法。dev仓（~30条，5.7%）进一步印证此类缺陷的普遍性。

#### 规则INT-01: 无符号整数减法下溢

严重等级：P0

缺陷描述： `a - b`当a < b时，uint32_t/uint64_t回绕为极大正数。多核分核场景中尾核分不到数据时最易触发。下溢后的极大值作为GM偏移导致硬件异常，或作为循环边界导致死循环。

典型代码示例：

```cpp
// 缺陷代码 — 99a876f9
uint32_t tailSize = initParams.totalOutputSize - constInfo.aivIdx * initParams.singleCoreSize;
// aivIdx * singleCoreSize > totalOutputSize时下溢为接近2^32

// 修复代码
uint32_t product = constInfo.aivIdx * initParams.singleCoreSize;
uint32_t tailSize = (initParams.totalOutputSize > product) ?
    (initParams.totalOutputSize - product) : 0;
```

系统性案例（6处同模式修复）：

```cpp
// 缺陷代码 — b4baa0d2 (split_core.h)
assignInfo.batchLeftCost -= assignInfo.s1GLeftCost;      // 可能下溢
assignInfo.batchLeftBlock -= assignInfo.s1GLeftBlock;     // 可能下溢
assignInfo.s1GLeftCost -= curCost;                        // 可能下溢

// 修复代码 — 统一安全减法模式
assignInfo.batchLeftCost = assignInfo.batchLeftCost > assignInfo.s1GLeftCost ?
    assignInfo.batchLeftCost - assignInfo.s1GLeftCost : 0U;
assignInfo.batchLeftBlock = assignInfo.batchLeftBlock > assignInfo.s1GLeftBlock ?
    assignInfo.batchLeftBlock - assignInfo.s1GLeftBlock : 0U;
assignInfo.s1GLeftCost = assignInfo.s1GLeftCost > curCost ?
    assignInfo.s1GLeftCost - curCost : 0U;
```

审查检查方法：
- 所有uint减法`a - b`：a是否可能小于b
- 固定修复模式：`a > b ? a - b : 0U`
- 重点关注多核分核(SplitCore)场景的uint减法

关联commit: `99a876f9`, `b4baa0d2`

---

#### 规则INT-02: uint16参数截断

严重等级：P0

缺陷描述：DataCopy参数(blockLen/stride)和硬件指令repeat参数使用uint16_t，最大值65535。大数据量时截断导致实际拷贝/计算长度远小于预期。dev仓中DataCopyParams的srcStride/dstStride为uint16_t（上限65535），当输入shape较大时stride超限被截断的案例进一步印证。

典型代码示例：

```cpp
// 缺陷代码 — e13d1ec19
uint16_t cpInLen = size * sizeof(bfloat16_t);  // 超过65535时截断
DataCopyParams cpInParams{1, cpInLen, 0, 0};
DataCopyPad(xLocal, gmSrc, cpInParams, padParams);

// 修复代码 — 扩宽到uint32_t + 使用Ext版本API
uint32_t cpInLen = size * sizeof(bfloat16_t);
DataCopyExtParams cpInExtParams{1, cpInLen, 0, 0, 0};
DataCopyPadExtParams<bfloat16_t> padExtParams{false, 0, 0, 0};
DataCopyPad(xLocal, gmSrc, cpInExtParams, padExtParams);
```

审查检查方法：
- DataCopy参数(blockLen/stride)赋值是否可能超过uint16_t(65535)
- 是否应使用Ext版本API(DataCopyExtParams/DataCopyPadExtParams)
- repeat参数在大数据量时是否溢出
- 大shape场景(seq > 65535): 优先使用DataCopyExtParams

关联commit: `e13d1ec19`, `e96b7b2a`；dev仓： `cefdc8776e`, `025c1734`

---

#### 规则INT-03: unsigned/signed类型不匹配

严重等级：P1

缺陷描述：框架API用int64_t（-1表示动态shape），结构体字段用uint64_t导致-1隐式转为18446744073709551615。后续校验和计算逻辑全部出错。

典型代码示例：

```cpp
// 缺陷代码 — 78282635
struct QuantAllReduceShapeInfo {
    uint64_t b;           // 框架返回int64_t的-1(动态shape)
    uint64_t s;           // 隐式转为18446744073709551615
    uint64_t hiddenSize;
};

// 修复代码
struct QuantAllReduceShapeInfo {
    int64_t b;            // 与框架API类型匹配
    int64_t s;
    int64_t hiddenSize;
};
```

审查检查方法：
- 结构体字段类型是否与上游API的类型一致(int64_t vs uint64_t)
- shape相关的维度值应使用int64_t（因为-1表示动态维度）
- 编译选项开启-Wsign-conversion可检测隐式符号转换

关联commit: `78282635`

---

#### 规则INT-04: 乘法溢出

严重等级：P0

缺陷描述：uint32_t乘法在offset/size计算中溢出。常见于`a * b`其中a和b各自不大但乘积超过2^32。dev仓中地址偏移 = uint32 * uint32在大数据场景(>4GB)溢出的案例进一步印证。

典型代码示例：

```cpp
// 缺陷代码 — 831ab170
uint32_t offset = s1BottomTok * params.gSize;  // gSize较大时溢出

// 修复代码
uint64_t offset = static_cast<uint64_t>(s1BottomTok) * params.gSize;
```

审查检查方法：
- offset/size计算中的乘法是否可能溢出当前类型
- 至少一个操作数cast到目标宽度后再相乘
- GM偏移、workspace大小计算优先使用uint64_t
- 3个以上维度连乘：至少一个操作数cast到uint64/int64
- 特别关注TND/大batch/大seq场景

关联commit: `831ab170`, `6b029d5f`；dev仓： `29edf728`, `e5817c02`

---

### 类别八：UT/测试代码缺陷（16条，6.6%）

测试代码本身的缺陷。核心问题是UT编写质量标准低于production代码。

#### 规则TEST-01: UT数据类型不匹配

严重等级：P2

缺陷描述：UT中host数据类型与aclDataType不一致，或sizeof用错类型。导致内存分配只有一半、数据解析错误、精度对比永远失败。

典型代码示例：

```cpp
// 缺陷代码 — b9a02e9d
// gen_data.py中position用float生成，但算子期望int64
position = np.random.randn(...).astype(np.float32)  // 应为int64

// test_case_fp32中sizeof用错类型
size_t bufSize = count * sizeof(half);  // FP32测试用例应为sizeof(float)
```

审查检查方法：
- UT中CreateAclTensor的dataType与host数据的C++类型是否匹配
- sizeof()参数的类型是否与实际测试数据类型一致
- Python数据生成脚本中的dtype是否与算子预期一致

关联commit: `b9a02e9d`, `a911707e`

---

#### 规则TEST-02: UT路径依赖闭源组件

严重等级：P2

缺陷描述：UT include引用闭源路径（level2/等）、system()拷贝闭源仓库脚本，导致开源仓库无法编译UT。

典型代码示例：

```cpp
// 缺陷代码 — 29b08607
#include "level2/some_internal_header.h"      // 闭源路径
system("cp /path/to/closed_source/script.sh .");  // 依赖闭源仓库

// 修复代码
#include "relative/path/to/header.h"           // 使用相对路径
// 移除对闭源脚本的依赖
```

审查检查方法：
- UT的include路径是否使用相对路径（而非闭源绝对路径）
- UT是否有system()调用依赖闭源组件
- 开源环境下UT是否可独立编译运行

关联commit: `29b08607`, `79cd3244`

---

#### 规则TEST-03: UT与production不同步

严重等级：P3

缺陷描述：production代码重命名（头文件名、命名空间、结构体名）后UT未同步更新，UT编译失败或测试的是旧接口。tiling结构体字段未初始化也属此类。

典型代码示例：

```cpp
// 缺陷代码 — 8d72712c
#include "old_header_name.h"    // 头文件已重命名
using namespace old_ns;          // 命名空间已变更
EXPECT_EQ(output, "old_string"); // 期望输出已过期

// 修复代码
#include "new_header_name.h"
using namespace new_ns;
EXPECT_EQ(output, "new_string");
```

审查检查方法：
- production代码重命名/重构后，搜索UT中是否有旧名称引用
- UT中tiling结构体的所有字段是否初始化
- UT的期望输出是否与当前production行为一致

关联commit: `8d72712c`, `f1a24bd`

---

### 类别九：API误用/接口错误（12条，4.9%）

使用了错误的API或传递了错误的参数，通常因为API名称相似或语义不够直观。dev仓（~25条，4.7%）中Check函数返回值被丢弃、return false在graphStatus函数中产生语义反转是新增的高频子模式。

#### 规则API-01: GetInputShape与GetOptionalInputShape混淆

严重等级：P0

缺陷描述：对可选输入使用了必选输入的访问接口(GetInputShape/GetInputDesc)。可选输入不存在时，GetInputShape按必选索引访问会越界或返回无效指针，后续解引用产生未定义行为（越界报错或崩溃）。同类混淆还有Size() vs GetShapeSize()、CeilDiv vs CeilAlign。

典型代码示例：

```cpp
// 缺陷代码 — 73c4fdaa (fused_infer_attention_score_tiling.cpp)
auto* shape = context->GetInputShape(LEARNABLE_SINK_INDEX);  // 可选输入不存在时越界访问
int64_t dim0 = shape->GetDim(0);  // 未定义行为

// 修复代码
auto* shape = context->GetOptionalInputShape(OPTIONAL_INPUT_IDX);  // 不存在时返回nullptr
if (shape != nullptr) {
    int64_t dim0 = shape->GetDim(0);
}
```

审查检查方法：
- 可选输入是否使用GetOptionalInputShape/GetOptionalInputTensor（而非GetInputShape/GetInputTensor）
- 检查算子注册中哪些输入标记为optional，对照访问代码逐一确认
- 非连续tensor使用Size()还是GetShapeSize()——前者返回含stride的逻辑大小

关联commit: `73c4fdaa`, `d0ede159b`, `a7010fc9`, `c2250fc9`

---

#### 规则API-02: 接口注册参数顺序错误

严重等级：P1

缺陷描述：INFER_SHAPE注册、Attr注册等宏中参数顺序与函数签名或aclnn接口参数顺序不匹配。编译不报错但运行时参数被错误绑定。

典型代码示例：

```cpp
// 缺陷代码 — 6978efa7
// INFER_SHAPE注册时参数顺序与函数签名不匹配
IMPL_INFER_SHAPE(OpName, paramB, paramA)  // 反了

// 修复代码
IMPL_INFER_SHAPE(OpName, paramA, paramB)  // 与函数签名一致
```

审查检查方法：
- 接口注册（INFER_SHAPE等）的参数顺序是否与函数签名匹配
- Attr注册顺序是否与aclnn接口参数顺序一致
- 编译通过不代表正确——需要运行时验证

关联commit: `6978efa7`, `e40a4176`

---

#### 规则API-03: DataCopy参数结构体类型错误

严重等级：P1

缺陷描述：DataCopy系列API有多个参数结构体(DataCopyParams / DataCopyExtParams / DataCopyPadParams)，成员不同但部分重叠。传错类型编译可能不报错但运行时行为未定义。

典型代码示例：

```cpp
// 缺陷代码 — ac0860f7
DataCopyExtParams extParams{...};       // 用了ExtParams
DataCopyPad(dst, src, extParams, ...);  // DataCopyPad需要DataCopyPadParams

// 修复代码
DataCopyPadParams padParams{...};       // 正确的参数结构体类型
DataCopyPad(dst, src, padParams, ...);
```

审查检查方法：
- DataCopy系列API的参数结构体类型是否正确
- DataCopyParams用于DataCopy，DataCopyExtParams用于DataCopyExt，DataCopyPadParams用于DataCopyPad
- 不依赖编译器隐式转换——显式使用正确的结构体类型

关联commit: `ac0860f7`

---

#### 规则API-04: Check函数返回值被丢弃

严重等级：P0

缺陷描述：调用CheckXxx()函数后不检查返回值(ge:graphStatus)，即使返回GRAPH_FAILED也继续执行后续逻辑，非法输入不被拦截。dev仓约8条。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 0f363800)
CheckFAIIsTND(context);  // 返回值被完全丢弃
// GRAPH_FAILED也继续返回SUCCESS

// 修复代码
auto ret = CheckFAIIsTND(context);
if (ret != ge::GRAPH_SUCCESS) { return ret; }
```

审查检查方法：
- diff中所有Check/Verify函数调用：返回值是否被检查并传播
- 考虑对Check函数添加[[nodiscard]]属性
- LOGE后必须紧跟return错误码（否则仅打日志不生效）

关联commit: dev仓 `0f363800`, `e7278f18`

---

#### 规则API-05: return false在graphStatus函数中（语义反转）

严重等级：P1

缺陷描述：函数声明返回ge:graphStatus（整型），但函数体中return false/true。bool->int隐式转换导致false=0=GRAPH_SUCCESS，本意返回失败实际返回成功。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 479084fc, incre_flash_attention_tiling_v2.cpp存量bug)
ge::graphStatus CheckUbSpace(...) {
    if (condition) {
        return false;  // 意图返回失败, 但 false=0=GRAPH_SUCCESS
    }
    return true;  // 意图返回成功, 但 true=1=GRAPH_FAILED
}

// 修复: 使用正确的枚举值
ge::graphStatus CheckUbSpace(...) {
    if (condition) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
```

审查检查方法：
- 返回ge:graphStatus的函数中搜索return true/false
- 静态分析规则：检测graphStatus函数中的bool返回值
- LOGE后检查是否紧跟return ge:GRAPH_FAILED（而非缺少return）

关联commit: dev仓 `479084fc`

---

#### 规则API-06: 函数签名/参数不匹配

严重等级：P1

缺陷描述：extern声明与实际定义参数列表不一致；指针值传递应为引用；SetOutputDataType参数写反。dev仓约6条。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 0c941da3)
void ProcessData(SomeType* ptr) {  // 值传递指针
    ptr = newPtr;  // 修改局部副本, 调用方不可见
}

// 修复: 引用传递
void ProcessData(SomeType*& ptr) {
    ptr = newPtr;  // 修改调用方的指针
}
```

审查检查方法：
- extern声明与实际函数定义参数列表逐一比对
- 函数内修改指针本身（非指向内容）时：参数应为指针引用或二级指针
- API版本升级：V(N+1)新增参数不应在V(N)接口中调用

关联commit: dev仓 `0c941da3`, `7eee21c7`, `051f878c`

---

#### 规则API-07: API参数语义/顺序混淆

严重等级：P1

缺陷描述：函数参数位置调换（如rowOuterIdx与rowInnerIdx）；参数语义与实际传值相反（elementLengths和srcList赋值互换）。参数>3个时尤其易错。dev仓约25条参数值/顺序/魔数类缺陷。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 e4b3ba6d)
// elementLengths[i]与srcList.src[i+1]的Dst/Src对应关系反了
params.elementLengths[0] = mrgDstNum;  // 应为mrgSrcNum(对应src1)
params.elementLengths[1] = mrgSrcNum;  // 应为mrgDstNum(对应src2)
srcList.src1 = mrgDst;                 // 应为mrgSrc
srcList.src2 = mrgSrc;                 // 应为mrgDst

// 修复: Dst/Src的index顺序互换, 使elementLengths与srcList对应
params.elementLengths[0] = mrgSrcNum;
params.elementLengths[1] = mrgDstNum;
srcList.src1 = mrgSrc;
srcList.src2 = mrgDst;
```

审查检查方法：
- API调用核对参数与函数声明中参数名的语义对应（不仅看类型）
- 函数参数>3个：考虑用结构体封装避免位置混淆
- 同名参数（如两个int）：特别注意顺序

关联commit: dev仓 `e4b3ba6d`, `7075b6143f`

---

### 类别十：状态管理/赋值/传参缺陷（12条，4.9%）

dev仓（~30条初始化/赋值遗漏 + ~18条复制粘贴）进一步扩展了此类别的覆盖范围，新增了条件分支遗漏赋值、构造/基类初始化缺失、循环状态残留、平行函数修改不同步等子模式。

#### 规则STATE-01: 结构体字段赋值遗漏

严重等级：P1

缺陷描述：数据传递链路中某个字段未赋值，下游读到未初始化值或零值。常见于新增字段时遗漏了某条初始化路径，或tiling结构体未满足硬件对齐要求。

典型代码示例：

```cpp
// 缺陷代码 — 8e9fe171
// tilingData新增了fieldX字段，但SetTilingData函数中漏掉了赋值
tilingData.fieldA = valueA;
tilingData.fieldB = valueB;
// tilingData.fieldX = valueX;  ← 遗漏

// 修复代码
tilingData.fieldA = valueA;
tilingData.fieldB = valueB;
tilingData.fieldX = valueX;  // 补上
```

审查检查方法：
- 新增struct字段后，所有初始化/赋值路径是否都设置了该字段
- tiling结构体是否满足硬件对齐要求（64位对齐）
- 列出结构体全部字段，逐一对比赋值语句

关联commit: `8e9fe171`, `312d30359`, `90cc4df64`

---

#### 规则STATE-02: 复制粘贴变量名未全替换

严重等级：P1

缺陷描述：从相似函数/分支复制代码后，变量名未全部替换完毕。计算使用了错误的输入源或输出目标。dev仓约18条复制粘贴/拼写类缺陷，从已有文件/函数复制代码后关键标识符（变量名、函数名、buffer索引）未替换为新上下文的正确值。

典型代码示例：

```cpp
// 缺陷代码 — 64be6dd6
// 从处理query的代码复制到处理key，但部分变量名仍是query
keyOffset = queryStride * batchIdx;  // 应为keyStride

// 修复代码
keyOffset = keyStride * batchIdx;
```

```cpp
// 缺陷代码 (dev仓 120106d9)
// L0A/L0B/L0C索引全写成ASCEND_CB, 从一行复制未改
InitBuffer(inQueueL0A_, ASCEND_CB, ...);  // 应为ASCEND_L0A
InitBuffer(inQueueL0B_, ASCEND_CB, ...);  // 应为ASCEND_L0B
InitBuffer(inQueueL0C_, ASCEND_CB, ...);  // 应为ASCEND_L0C
```

审查检查方法：
- 复制粘贴的代码块中，变量名是否全部替换完毕
- diff中查看新增代码是否与相邻已有代码高度相似——若相似则逐行比对变量名
- 使用IDE的"查找替换"而非手动替换
- 新增文件/函数：如果是从已有代码复制，逐行检查所有标识符是否已替换
- grep源算子/函数名：diff中不应残留复制来源的标识符

关联commit: `64be6dd6`；dev仓： `a02041efea`, `e8bdf8f2`, `120106d9`

---

#### 规则STATE-03: 变量名拼写错误/重命名不同步

严重等级：P2

缺陷描述：变量名typo导致编译器创建新变量而非引用目标变量；函数/头文件重命名后调用点/include未同步更新。

典型代码示例：

```cpp
// 缺陷代码 — b96fd79d
auto consInfo = GetConstInfo();  // typo: 应为constInfo
// consInfo是新的局部变量，constInfo(成员变量)未被更新

// 修复代码
auto constInfo = GetConstInfo();
```

审查检查方法：
- 编译选项开启-Wshadow检测变量遮蔽
- 函数/变量重命名后，全局搜索确认所有调用点同步更新
- 头文件名修改后grep确认所有#include同步

关联commit: `b96fd79d`, `6e16f3dc`, `69f8dff2`

---

#### 规则STATE-04: 条件分支遗漏赋值（if赋值else未赋值）

严重等级：P1

缺陷描述：变量仅在if分支赋值，else分支保持默认值（通常为0），后续公共路径使用该变量。unsigned类型的0减1下溢为UINT32_MAX是典型后果。dev仓约10条。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 b7a0e9fced)
uint32_t kvSLoopNumTotal = 0;
if (condition) {
    kvSLoopNumTotal = calcValue;
}
// else分支未赋值, kvSLoopNumTotal保持0
uint32_t loopEnd = kvSLoopNumTotal - 1U;  // 0-1 = UINT32_MAX -> 循环爆炸
```

审查检查方法：
- if赋值的变量：检查else/default分支是否也赋值
- unsigned变量：减法操作前确认变量>0
- 新增分支（如新模式/新Layout）时确认所有输出变量都被赋值

关联commit: dev仓 `b7a0e9fced`, `3aacdac4`, `66337ddb6b`

---

#### 规则STATE-05: 构造/基类初始化缺失

严重等级：P1

缺陷描述：子类构造函数未调用基类InitCompileInfo()等初始化方法，或结构体字段声明时未给初始值，导致使用未初始化数据。dev仓约6条。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 1fab7bab45)
class DerivedTiling : public BaseTiling {
    DerivedTiling() {
        // 未调用 BaseTiling::InitCompileInfo()
        // platformInfo等基类成员未初始化
    }
};

// 修复: 构造函数中调用 InitCompileInfo()
```

审查检查方法：
- 新增子类：是否调用基类Init方法
- 结构体新增字段：是否有默认初始值或在所有构造路径被赋值
- 成员变量声明：是否在类定义处赋默认值

关联commit: dev仓 `1fab7bab45`, `3aacdac4`

---

#### 规则STATE-06: 循环/batch切换时状态残留

严重等级：P1

缺陷描述：多batch/多循环迭代间，上一轮的中间状态（偏移量、累加值、标志位）未被正确reset，导致后续迭代使用残留值。dev仓约5条。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 1d256572)
// batch切换时ROPE偏移量未更新, 使用上一batch的残留值
for (int b = 0; b < batchSize; b++) {
    // ropeOffset未在batch切换时重新计算
    ProcessBatch(b, ropeOffset);  // 第二个batch用了第一个batch的offset
}

// 修复: 每个batch开始时重新计算offset
```

审查检查方法：
- 循环体：上一迭代的中间变量是否在新迭代开始时reset
- batch处理：状态变量(offset/flag/accumulator)是否在batch切换时清零/重算
- buffer复用：跨迭代buffer是否需要清零

关联commit: dev仓 `1d256572`, `64f59aae`

---

#### 规则STATE-07: 平行函数组修改不同步

严重等级：P1

缺陷描述：V1/V2/V3等平行函数或combine_v2.h与_add_rms_norm.h等克隆文件，仅修改一处而未同步到平行代码。aclnn_grouped_matmul.cpp的6个版本入口是典型热点。dev仓约5条。

审查检查方法：
- diff中只修改了某函数的一个版本：检查平行版本是否也需同样修改
- combine_v2.h与_add_rms_norm.h: 同一bug修复必须同步到两个文件
- aclnn接口V1-V5: 新增参数/check需同步6处入口

关联commit: dev仓 `5508fbb6c9`, `e8bdf8f2`

---

#### 规则STATE-08: 常量名值不匹配/魔法数字

严重等级：P2

缺陷描述：常量名与值矛盾(ZERO_DIM=1, TWO_DIM=3); 边界值硬编码无推导来源注释；非标准无穷大表示（3e+99而非IEEE 754）。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 cfe618f7)
constexpr int ZERO_DIM = 1;  // 名为ZERO实际值为1
constexpr int TWO_DIM = 3;   // 名为TWO实际值为3

// 修复: 名称与值匹配
constexpr int ZERO_DIM = 0;
constexpr int TWO_DIM = 2;
```

审查检查方法：
- 常量定义：名称与值必须语义一致
- 魔法数字：边界值/阈值需定义为命名常量并附注推导来源
- 无穷大：使用std:numeric_limits<T>:infinity()而非字面量

关联commit: dev仓 `cfe618f7`, `cb1c8346`, `fb58b43e98`

---

### 类别十一：Revert/回归引入（6条，2.5%）

6个独立Revert事件暴露的系统性流程缺陷。dev仓（35个Revert, 8个事件群）进一步扩展了对Revert模式的认识，新增了架构方案首次失败后应暂停、大型新算子合入门禁两条规则。

#### 规则REV-01: 大型变更缺乏分阶段提交策略

严重等级：P3

缺陷描述：大型变更（>10文件/1000行）一次性提交，回退代价极高。ROPEV2（27文件/3000行）和tilingkey模板化（32文件/20000行）都一次性提交后被revert，最终改进后重新合入。触发原因分布：编译/构建问题(3)、并发/同步正确性(2)、示例代码集成(1)。

审查检查方法：
- 大型变更（>10文件/1000行）是否可以分阶段提交
- 同步原语的"优化"是否证明了等价性（而非依赖CI验证）
- 依赖的框架API是否已稳定发布（而非内部staging版本）
- CI门禁是pre-merge还是post-merge——post-merge拦截成本远高于pre-merge

关联commit: Revert相关commit见revert_analysis.md

---

#### 规则REV-02: 架构方案首次失败后应暂停评审

严重等级：P1

缺陷描述：dev仓中tilingKey模板化整改在MatmulAllReduce首次失败后未暂停，继续推进到其他MC2算子(MatmulReduceScatterV2、MoeDistributeCombine/Dispatch)，5天内连续6次Revert。

审查检查方法：
- 同一架构方案在首个模块失败并revert后：要求根因分析和方案修正再继续
- 跨模块推广：确认已有至少一个模块稳定运行

关联commit: dev仓 `2001261f7c`, `8ead68ccfa`, `20f4d5412b`, `58a7ef184e`, `a2b032dcb2`, `02505c922e`

---

#### 规则REV-03: 大型新算子合入门禁

严重等级：P2

缺陷描述：dev仓10个premature merge中7个在合入后2小时内revert，问题在本地编译/CI即可发现。commit message"接口调试"、MR描述空白是预警信号。

审查检查方法：
- 新增算子（>10个新文件）：必须填写测试栏并提供CI通过证据
- commit message含"调试"/"临时"/"WIP": 不应合入主干
- MR描述/branch名异常（如"merge master into master"）：需要额外审查

关联commit: dev仓 `1788aaceea`, `8611973812`, `1ff006a430`, `dfa78b4b93`

---

### 类别十二：硬件平台兼容性（~25条）

dev仓中新增的显著类别。dev分支承担新平台(A5/910_95)首发适配，硬件平台兼容性缺陷约25条。主仓不突出但在SYS-05（构建模式组合爆炸）中有所体现。

#### 规则PLAT-01: 新平台运行时/编译期分支未覆盖

严重等级：P1

缺陷描述：socVersion字符串比较路由("Ascend910_95")、`#if __CCE_AICORE__ == 220`预处理分支、ASCEND_COMPUTE_UNIT条件——新增硬件型号时多处遗漏。硬编码硬件参数（cvRatio/对齐粒度/核数）未通过API获取也属此类。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 4bc16847)
constexpr int cvRatio_ = 2;  // 硬编码c:v比例
// A5平台c:v=1:1, 此constexpr在编译期确定, 运行时无法区分

// 修复: 运行时获取
int cvRatio_ = GetSubBlockNum();  // API返回实际硬件比例
```

审查检查方法：
- 新平台适配PR: 搜索socVersion/ASCEND_COMPUTE_UNIT/__CCE_AICORE__所有出现位置
- 硬编码硬件参数（cvRatio/对齐粒度/核数）：应通过API获取而非hardcode
- constexpr用于平台参数时确认是否需要改为运行时获取

关联commit: dev仓 `4bc16847`, `bf74834`, `b85709b25e`

---

#### 规则PLAT-02: 平台特有寄存器/配置遗漏

严重等级：P2

缺陷描述：新arch适配时遗漏平台特有的SPR/控制寄存器设置；InitGlobalMemory对齐要求跨平台不同。

审查检查方法：
- 新arch kernel: 检查是否需要特殊SPR/控制寄存器设置（参考已有arch适配）
- 平台切换时：InitGlobalMemory/对齐参数是否需要调整

关联commit: dev仓 `b85709b25e`

---

### 类别十三：TilingKey一致性（~20条）

dev仓中新增的显著类别。host侧tiling生成的key值与kernel侧模板匹配常量不一致，是dev仓最具特征性的缺陷模式，也是Revert事件群1（6次tilingKey模板化整改连续失败）的根因。主仓仅零星出现。

#### 规则TKEY-01: host/kernel tilingKey常量值不一致

严重等级：P0

缺陷描述：host侧通过GET_TPL_TILING_KEY宏或手动计算生成tilingKey，kernel侧通过TILING_KEY_IS/ASCENDC_TPL_SEL匹配。两侧值不一致导致kernel走入错误的模板分支或无法匹配任何分支。dev仓约8条。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 02505c922e系列)
// host侧: tilingKey = ... (新编码方式)
// kernel侧: TILING_KEY_IS(oldValue) -> 无法匹配host下发的newValue
// 结果: 无法命中正确kernel分支, 可能走default fallback或行为未定义
```

审查检查方法：
- tilingKey新增/修改的diff中：必须同时出现host侧(tiling.cpp)和kernel侧(kernel.h/.cpp)的对应修改
- GET_TPL_TILING_KEY宏参数顺序与kernel侧ASCENDC_TPL_SEL严格一致
- 建议：host/kernel共享tilingKey常量定义头文件，避免手动维护两份

关联commit: dev仓 `02505c922e`, `ee638d9b`, `31044c91`

---

#### 规则TKEY-02: tilingKey编码维度缺失

严重等级：P1

缺陷描述：新增feature/配置参数时未编入tilingKey，不同配置映射到同一key，共用同一kernel模板分支。

典型代码示例：

```cpp
// 缺陷代码 (dev仓 31044c91)
// tilingKey未编码mte2Config参数
// 不同vec_config映射到同一key, kernel无法区分配置差异

// 修复: 将mte2Config编入key的某一位
tilingKey |= (mte2Config << BIT_OFFSET);
```

审查检查方法：
- 新增配置参数时：检查是否需要编入tilingKey（否则不同配置共用同一kernel）
- 位域顺序：核对每一位的含义注释与实际传值
- tilingKey位数：确认未溢出uint64范围

关联commit: dev仓 `31044c91`, `aba3bd8f`, `b41315b3`, `13ae078c`

---

### 跨类别系统性风险

以下7个风险跨越多个缺陷类别，需要系统性应对而非逐条修复。

#### 规则SYS-01: GQA gSize缩放因子遗漏

严重等级：P1（跨类别1/3，6条独立commit）

缺陷描述：本仓库最频繁的单一具体根因。GQA中Q head数是KV head数的gSize倍，所有涉及Q-KV维度交叉计算的地方都需要gSize缩放。问题反复出现的根因是：gSize影响的变量散布在tiling/kernel的多个函数中，没有一个统一的"GQA适配层"。

审查检查方法：
- 涉及attention算子的PR，全局搜索preTokens/nextTokens/actualSeqLength/s1Size等变量
- GQA路径中每个Q-KV维度交叉计算是否乘了gSize
- 新增layout或模式时，GQA组合是否被覆盖

关联commit: `d452dde6`, `81213d24`, `8a852918`, `3bb75678`, `e44028b1`, `16bc59a1`

---

#### 规则SYS-02: 空tensor全链路联动

严重等级：P1（跨类别4/5）

缺陷描述：空tensor问题跨越API校验层（类别4）和内部处理层（类别5）。`<=0` vs `<0`的校验边界差异是最常见的具体表现。四层必须联动：
- aclnn层：空tensor不拦截，直接返回成功
- infershape层：输出shape正确赋值（即使为空）
- tiling层：空tensor不触发除零，设置标志位跳过计算
- kernel层：空tensor输出初始化为零

审查检查方法：
- 新增算子或修改校验逻辑时，用空tensor(N=0/B=0)做端到端测试
- 检查四层是否全部覆盖（任何一层遗漏都会导致问题）
- 校验条件是`<0`而非`<=0`

关联commit: `b5d7c0fe`, `236a5fdb`, `78e07300`, `54692072`, `44e84d52`

---

#### 规则SYS-03: host(tiling)侧与kernel侧不一致

严重等级：P0（跨类别1/10/13）

缺陷描述：workspace大小、buffer对齐、struct类型均出现tiling/kernel两侧独立计算不匹配。根因是两侧代码物理分离（op_host/ vs op_kernel/，不同编译单元）缺乏编译期一致性约束。dev仓中tilingKey按位组装(host) vs模板参数解析(kernel)约20条、workspace内存布局(host offset) vs kernel读取offset约8条、宏参数顺序vs宏定义约5条，进一步印证此系统性风险。

审查检查方法：
- workspace/buffer大小计算的tiling侧和kernel侧是否完全一致
- tiling struct的定义和使用两侧是否从同一个头文件引入
- 新增tiling字段时，两侧是否都做了对应修改
- tilingKey修改：diff必须同时包含host和kernel两侧
- workspace offset变更：验证host侧sizeof链与kernel侧读取逻辑一致
- 宏调用参数>5个：逐一比对宏定义的参数名
- 考虑将共享计算逻辑抽取到公共头文件（如tiling_key.h）

关联commit: `e233e106`, `8e9fe171`；dev仓涵盖TKEY全部案例和BUF-01部分案例

---

#### 规则SYS-04: 大型Tiling类状态累积

严重等级：P2（跨类别10/热点分析）

缺陷描述：prompt_flash_attention_tiling_v2.cpp有80+成员变量无统一reset机制。不同tiling路径可能读到上一次计算的残留状态。hotspot分析中排名第一的结构性风险。

审查检查方法：
- 大型Tiling类（>30成员变量）是否有统一的reset/clear方法
- 不同tiling路径是否可能读到前一路径设置的残留状态
- 新增成员变量时是否在reset方法中同步添加

关联commit: 热点文件分析(hotspot_analysis.md)

---

#### 规则SYS-05: 构建模式组合爆炸

严重等级：P2（跨类别2/11）

缺陷描述：ENABLE_EXPERIMENTAL、开源/闭源、多平台(310P/910B/Ascend950)组合下，CMake路径/条件编译/依赖关系难以维护。是revert的首要触发因素。

审查检查方法：
- CMake修改是否在所有构建模式（experimental/normal, 开源/闭源）下都正确
- 条件编译宏的组合是否有遗漏
- 平台相关代码的架构标识符是否正确

关联commit: `fbade778b`, `b0bf9fa2`, `ef22ac8f1`

---

#### 规则SYS-06: 代码克隆/平行实现"修一处漏一处"

严重等级：P1

缺陷描述：dev仓23次cmake缺陷和11次MC2 kernel缺陷的根因是"修一处漏一处"。已识别克隆对：custom_build.cmake vs CMakeLists.txt(RTY) 285行平行；func.cmake两个macro 80%重复；combine_v2.h vs _add_rms_norm.h 80%相似；aclnn_grouped_matmul.cpp 6个版本入口；PFA/IFA各自独立的TilingDataconvert。

审查检查方法：
- diff仅修改一个克隆体时：搜索对应克隆体是否需同步修改
- cmake/custom_build.cmake修改时：检查CMakeLists.txt（RTY分支）是否需同步
- func.cmake: add_modules_sources修改时检查add_modules_sources_with_soc
- MC2 MoE: combine_v2.h修改时检查_add_rms_norm.h

关联commit: dev仓高频模式，贯穿多个类别

---

#### 规则SYS-07: 硬件平台分支不完整

严重等级：P1

缺陷描述：新增硬件型号时多处遗漏是常见缺陷来源。散布在：socVersion字符串比较路由、#if __CCE_AICORE__预处理分支、CMake ASCEND_COMPUTE_UNIT条件、平台常量（cvRatio/对齐粒度）。

审查检查方法：
- 新平台适配PR: 搜索所有平台分支入口(socVersion/ASCEND_COMPUTE_UNIT/__CCE_AICORE__)
- 平台参数：优先用API获取(GetSubBlockNum())而非hardcode
- checklist: 编译选项、运行时分支、寄存器配置、对齐要求四项逐一确认

关联commit: dev仓涵盖PLAT和BUILD-06全部案例

---

### 附录：规则速查表

| 规则ID   | 规则名称                                 | 严重等级 | 关联commit                          |
| -------- | ---------------------------------------- | -------- | ----------------------------------- |
| CALC-01  | workspace/buffer大小计算单位混淆         | P0       | 9691bcc3, 69698404; dev: 798597da0a |
| CALC-02  | tiling侧与kernel侧计算不一致             | P0       | e233e106                            |
| CALC-03  | CeilDiv与CeilAlign语义混淆               | P1       | 84ca56e9; dev: 586493dc             |
| CALC-04  | workspace多段地址偏移重叠                | P0       | 3ca57b44; dev: e5817c02             |
| CALC-05  | GQA场景gSize缩放因子遗漏                 | P1       | d452dde6, 81213d24, 8a852918        |
| CALC-06  | GM偏移量整数溢出                         | P0       | 6b029d5f, 831ab170                  |
| CALC-07  | tiling除数为零                           | P0       | 234c1c8b, 1f3291bf; dev: b7a0e9fced |
| CALC-08  | 整数除法不满足分配律                     | P1       | 16bc59a1, 1ceed275                  |
| CALC-09  | 数据搬运对齐不满足硬件要求               | P1       | dev: 3cfe2f3b64, c439c4d7           |
| CALC-10  | 不必要的对齐破坏地址计算                 | P2       | dev: 73a82c36                       |
| BUILD-01 | 新算子CMake集成遗漏                      | P2       | 80329f3a                            |
| BUILD-02 | CMake宏参数缺失                          | P1       | a54af6d2e                           |
| BUILD-03 | ENABLE_EXPERIMENTAL模式路径适配错误      | P2       | fbade778b, b0bf9fa2                 |
| BUILD-04 | 安装脚本误删其他组件文件                 | P1       | 83be205f                            |
| BUILD-05 | CMake OP_NAME复制粘贴错误                | P2       | dev: 5508fbb6c9                     |
| BUILD-06 | 新平台编译选项分支遗漏                   | P1       | dev: eeba666256                     |
| BUILD-07 | auto-sync编译选项与kernel冲突            | P1       | dev: 38606b2b                       |
| BUILD-08 | include路径变更未同步                    | P2       | dev: 05c16e686c                     |
| BUILD-09 | 黑名单/分包配置未更新                    | P2       | dev: eeba666256                     |
| LOGIC-01 | layout分支遗漏                           | P1       | ad627275f; dev: b4b9198148          |
| LOGIC-02 | 模式组合未覆盖                           | P1       | 8a852918; dev: c8cf3f011d           |
| LOGIC-03 | 条件守卫缺失                             | P1       | f91e8dee, a75a92e2                  |
| LOGIC-04 | 循环内布尔赋值覆盖                       | P1       | f5f79a3e                            |
| LOGIC-05 | OP_CHECK_IF条件语义反转                  | P0       | dev: b6ec4bfe86                     |
| LOGIC-06 | 逻辑运算符OR/AND混淆                     | P0       | dev: 67f1fffc                       |
| LOGIC-07 | Layout专属逻辑范围错误                   | P1       | dev: 0166982e                       |
| VALID-01 | dtype白名单校验缺失                      | P1       | 8ceb72b6, f8882f78                  |
| VALID-02 | 空tensor过度拦截                         | P1       | 54692072; dev: c8cf3f011d           |
| VALID-03 | 校验顺序依赖错误                         | P2       | 44e84d52, 9b12d59b                  |
| BOUND-01 | 空tensor全链路处理缺失                   | P0       | b5d7c0fe, 236a5fdb                  |
| BOUND-02 | 空指针格式化崩溃                         | P0       | d3aa4960                            |
| BOUND-03 | 尾块边界条件处理错误                     | P1       | b59cf016; dev: 2efc569c             |
| BOUND-04 | 特殊值(inf/NaN)未处理                    | P1       | 4801ecb2                            |
| BOUND-05 | GM访问前缺边界检查                       | P0       | dev: af72ccb938                     |
| BOUND-06 | UB buffer复用缺同步/清零                 | P1       | dev: f55d2a35                       |
| BOUND-07 | DataCopy参数/边界越界                    | P0       | dev: 2efc569c, 1dbd7856             |
| SYNC-01  | 流水线切换缺失barrier                    | P0       | 984d8a72; dev: d2601de86d           |
| SYNC-02  | 同步方向反转                             | P0       | 662f162c; dev: d5c79c44c7           |
| SYNC-03  | event信号Set/Wait不配对                  | P0       | a58c8a0b9; dev: ee07e5dd25          |
| SYNC-04  | SyncAll模板参数误用                      | P0       | 5dce387d, 65cafae0                  |
| INT-01   | 无符号整数减法下溢                       | P0       | 99a876f9, b4baa0d2                  |
| INT-02   | uint16参数截断                           | P0       | e13d1ec19; dev: cefdc8776e          |
| INT-03   | unsigned/signed类型不匹配                | P1       | 78282635                            |
| INT-04   | 乘法溢出                                 | P0       | 831ab170; dev: 29edf728             |
| TEST-01  | UT数据类型不匹配                         | P2       | b9a02e9d, a911707e                  |
| TEST-02  | UT路径依赖闭源组件                       | P2       | 29b08607                            |
| TEST-03  | UT与production不同步                     | P3       | 8d72712c, f1a24bd                   |
| API-01   | GetInputShape与GetOptionalInputShape混淆 | P0       | 73c4fdaa, d0ede159b                 |
| API-02   | 接口注册参数顺序错误                     | P1       | 6978efa7, e40a4176                  |
| API-03   | DataCopy参数结构体类型错误               | P1       | ac0860f7                            |
| API-04   | Check函数返回值被丢弃                    | P0       | dev: 0f363800                       |
| API-05   | return false在graphStatus函数中          | P1       | dev: 479084fc                       |
| API-06   | 函数签名/参数不匹配                      | P1       | dev: 0c941da3                       |
| API-07   | API参数语义/顺序混淆                     | P1       | dev: e4b3ba6d                       |
| STATE-01 | 结构体字段赋值遗漏                       | P1       | 8e9fe171, 312d30359                 |
| STATE-02 | 复制粘贴变量名未全替换                   | P1       | 64be6dd6; dev: a02041efea           |
| STATE-03 | 变量名拼写错误/重命名不同步              | P2       | b96fd79d, 6e16f3dc                  |
| STATE-04 | 条件分支遗漏赋值                         | P1       | dev: b7a0e9fced                     |
| STATE-05 | 构造/基类初始化缺失                      | P1       | dev: 1fab7bab45                     |
| STATE-06 | 循环/batch切换状态残留                   | P1       | dev: 1d256572                       |
| STATE-07 | 平行函数组修改不同步                     | P1       | dev: 5508fbb6c9                     |
| STATE-08 | 常量名值不匹配/魔法数字                  | P2       | dev: cfe618f7                       |
| REV-01   | 大型变更缺乏分阶段提交策略               | P3       | revert_analysis.md                  |
| REV-02   | 架构方案首次失败后应暂停                 | P1       | dev: 2001261f7c                     |
| REV-03   | 大型新算子合入门禁                       | P2       | dev: 1788aaceea                     |
| PLAT-01  | 新平台分支未覆盖                         | P1       | dev: 4bc16847                       |
| PLAT-02  | 平台特有寄存器/配置遗漏                  | P2       | dev: b85709b25e                     |
| TKEY-01  | host/kernel tilingKey不一致              | P0       | dev: 02505c922e                     |
| TKEY-02  | tilingKey编码维度缺失                    | P1       | dev: 31044c91, aba3bd8f             |
| SYS-01   | GQA gSize缩放因子遗漏                    | P1       | d452dde6, 81213d24等6条             |
| SYS-02   | 空tensor全链路联动                       | P1       | b5d7c0fe, 236a5fdb等                |
| SYS-03   | host(tiling)侧与kernel侧不一致           | P0       | e233e106; dev: TKEY全部             |
| SYS-04   | 大型Tiling类状态累积                     | P2       | hotspot_analysis.md                 |
| SYS-05   | 构建模式组合爆炸                         | P2       | fbade778b等                         |
| SYS-06   | 代码克隆"修一处漏一处"                   | P1       | dev仓高频模式                       |
| SYS-07   | 硬件平台分支不完整                       | P1       | dev仓PLAT+BUILD-06                  |

### 数据来源

- 主仓： ~/repo/cann/ops-transformer/
  - 提交总数：1323（已排除merge commit）
  - 缺陷提交数：243（占比18.4%）
  - 分析周期：全量git历史
  - 中间产物：defect_analysis.md, revert_analysis.md, hotspot_analysis.md, pattern_summary.md
- dev仓： ~/repo/cann/ops-transformer-dev/
  - 提交总数：2822（非merge: 2820）
  - 确认缺陷：788条（阶段1关键词初筛）
  - 代码缺陷：528条（阶段2全量diff分析确认）
  - 分析周期：2025-08至2026-01
  - Revert事件：35条（阶段3专项分析）
  - 缺陷类别：15个 + 4个跨类别风险（阶段5归纳）
