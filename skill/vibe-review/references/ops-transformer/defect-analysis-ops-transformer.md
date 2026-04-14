# ops-transformer 缺陷逐条分析

数据来源: ops-transformer主仓(1323提交, 243条缺陷) + ops-transformer-dev(2822提交, 528条缺陷), 合计771条缺陷

---

## ops-transformer主仓 - 243条缺陷


## 批次1: 提交 #1-#20 (2026-02-26 ~ 2026-03-02)

---

### fe0bee0d 修复dequant_quant_kvcache算子

非缺陷提交(代码清理)。删除一行已完成的TODO注释，不涉及运行时逻辑。从defect_commits中排除。

---

### 5079260e 修复pfa合轴精度问题
- 根因类别：边界条件缺失(tiling优化路径准入条件不完整)
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：PFA合轴优化路径的`CheckPFAMerge`函数缺少关键前置校验。当query序列长度s>256且head维度d不能被64整除时(`d%64!=0`)，合轴路径产生精度问题，但原代码未拦截。
- 修复模式：新增条件`if (queryShapeInfo.s > 256U && (queryShapeInfo.d % 64) != 0) return false;`，不满足对齐约束时回退到不合轴路径。
- 可审查性：中(需理解PFA合轴的数学约束)
- 审查规则建议：tiling优化路径准入条件变更时，要求附带各维度参数的对齐要求文档，并用UT覆盖边界值(d=63/64, s=256/257)。

---

### 6a573aa2 修复ROPEV2中json的可选项设置
- 根因类别：算子描述配置与代码不一致
- 涉及文件：posembedding/rotary_position_embedding/op_host/config/kirin9030/rotary_position_embedding_binary.json, posembedding/rotary_position_embedding/op_host/config/kirinx90/rotary_position_embedding_binary.json, posembedding/rotary_position_embedding/op_host/op_api/aclnn_rotary_position_embedding.cpp
- 缺陷描述：ROPEV2新增`rotate`可选输入，但两个平台的binary JSON配置缺少index=3的`rotate`声明。同时aclnn API缺少`rotate!=nullptr`时的SOC架构校验(仅DAV_2201支持)。
- 修复模式：补全两个平台JSON中`rotate`可选输入声明；aclnn层新增SOC架构校验guard。
- 可审查性：高(JSON与代码参数列表不一致可自动化校验)
- 审查规则建议：新增/修改算子参数时，要求同时更新所有平台JSON配置。CI加入一致性校验。可选参数有平台限制时API入口必须有平台校验。

---

### a882f8c0 alltoallvgmm 修复量化模板转置问题 & aclnn共享转置问题
- 根因类别：变量遮蔽(variable shadowing) + 模板参数位置错误
- 涉及文件：mc2/allto_allv_grouped_mat_mul/op_api/aclnn_allto_allv_quant_grouped_mat_mul.cpp, mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_apt.cpp
- 缺陷描述：两处独立bug。(1) if块内`auto mmWeightOptional = transMmWeightOptional`声明了同名局部变量遮蔽外层参数，if块结束后外层变量未被更新。(2) `QuantGroupedMatmul`模板实例化时`TILINGKEY_GMM_WEIGHT_TRANSPOSE`和`TILINGKEY_MM_WEIGHT_TRANSPOSE`两个模板参数位置填反。
- 修复模式：(1) 去掉`auto`关键字直接赋值。(2) 修正模板参数位置对应关系。
- 可审查性：高(编译器`-Wshadow`可捕获变量遮蔽；模板参数审查可对照签名)
- 审查规则建议：启用`-Werror=shadow`编译选项。多模板参数实例化时要求注释说明每个参数语义，或用具名常量代替裸bool。

---

### c3cf4d96 修复maskshape支持范围
- 根因类别：条件分支逻辑错误(硬编码覆盖动态shape值)
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：`CheckMaskTypeAndShape`中当`enableIFAMask=true`时强制将`maskQsSize`硬编码为1，覆盖了从attenMask实际shape读取的值。IFA模式下mask的Q维度不一定为1，导致tiling数据与真实shape不匹配。
- 修复模式：删除`if (enableIFAMask) { maskQsSize = 1; }`整个条件块。
- 可审查性：中(需理解IFA模式下mask shape语义)
- 审查规则建议：看到"读取shape后又用常量覆盖"的模式应重点质疑。硬编码覆盖动态值必须附带注释说明原因。

---

### e8ffb3b9 sfa添加aclnn接口

非缺陷提交(新增接口)。为SparseFlashAttention新增aclnn外壳接口，处理softmaxMax/softmaxSum空指针场景。属于接口设计补全而非bug修复。从defect_commits中排除。

---

### 1216e10a grouped_matmul_finalize_routing：去掉pertoken_scale输入必须存在拦截
- 根因类别：参数校验过严(可选参数被当必选校验)
- 涉及文件：gmm/grouped_matmul_finalize_routing/op_host/grouped_matmul_finalize_routing_infershape.cpp
- 缺陷描述：`ValidateScaleAndBias`中当scale为三维(E,1,N)时，`OP_CHECK_IF`强制要求`pertoken_scale`不为nullptr，但规格定义其为可选输入。不传pertoken_scale时infershape直接报错。
- 修复模式：删除对pertoken_scale的强制判空检查(2行)。
- 可审查性：高(对照算子规格文档即可发现可选参数被强制校验)
- 审查规则建议：通过`GetOptionalInputShape`获取的输入，nullptr应为合法路径。凡对可选输入判空后返回GRAPH_FAILED，需检查是否与规格矛盾。

---

### 984d8a72 修复rotaryDim!=headSize时同步问题
- 根因类别：硬件流水线同步缺失
- 涉及文件：posembedding/rope_with_sin_cos_cache/op_kernel/rope_with_sin_cos_cache_f_bf16.h, posembedding/rope_with_sin_cos_cache/op_kernel/rope_with_sin_cos_cache_fp32.h
- 缺陷描述：RoPE kernel中处理完query转处理key时，DataCopy(MTE2搬入)前只调了`MTE3ToVSync()`缺少`VToMTE2Sync()`。当rotaryDim!=headSize时Vector单元可能仍在处理query，MTE2提前搬入key导致数据竞争。
- 修复模式：在key的DataCopy前、MTE3ToVSync()之后各增加一行`this->VToMTE2Sync()`。bf16和fp32各改2处共4行。
- 可审查性：中(需深入理解Ascend NPU MTE2/MTE3/Vector流水线同步模型)
- 审查规则建议：切换处理不同tensor并复用同一块local memory时，检查DataCopy前是否补全所有必要sync调用。建议lint规则：每个DataCopy前检查是否有覆盖所有写入来源的sync屏障。

---

### 53c21a1d npu_ops_transformer_ext算子与老写法算子隔离

非缺陷提交(构建工程改进)。引入CMake白名单机制控制打包范围，从"扫描所有子目录"改为"只打包白名单算子"。从defect_commits中排除。

---

### f14d0284 [FIA] pse,mask 非对齐场景拷贝越界修复
- 根因类别：非对齐场景DataCopy越界
- 涉及文件：attention/common/op_kernel/arch35/attenmask.h, attention/common/op_kernel/arch35/pse.h, attention/common/op_kernel/arch35/infer_flash_attention_kvcache.h
- 缺陷描述：attenmask/pse的stride模式DataCopy仅检查`totalS2Size%blockBytes==0`但未检查`s2Size`是否也对齐blockBytes，blockLen向上取整后越界。kvcache中GQA场景下actualS1Size计算遗漏乘以gSize。
- 修复模式：(1) attenmask/pse增加`s2Size%blockBytes==0`检查，不满足走逐行拷贝fallback。(2) kvcache中GQA+s1Size==1+非BNSD场景使用`nextTokensPerBatch*gSize`修正计算。
- 可审查性：中(需理解DataCopy stride模式对齐要求和GQA的s1/gSize关系)
- 审查规则建议：使用DataCopy stride参数且涉及CeilDiv取整时，必须同时验证源数据实际宽度也满足blockBytes对齐。否则走逐行拷贝。

---

### 8e9fe171 fix antiquant BNSD_BSND bug
- 根因类别：数据传递链路断裂(结构体字段遗漏)
- 涉及文件：attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_v2.cpp, attention/incre_flash_attention/op_host/incre_flash_attention_tiling_context.h, attention/incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp
- 缺陷描述：`IncreFlashAttentionContext`结构体缺少`transposeLayout`成员，`ConvertContextToParamsIFA`未调用`GetTransposeLayout`赋值。BNSD_BSND layout场景下tiling参数传递链断裂，kernel侧收到未初始化的layout信息。
- 修复模式：context新增`uint32_t transposeLayout=0`成员；ConvertContextToParamsIFA返回前赋值；IFATilingDataconvert中传递该值。补全数据传递链路。
- 可审查性：中(需理解tiling参数传递完整链路，单文件审查难发现)
- 审查规则建议：context/params结构体新增字段时，检查所有构造路径是否显式赋值，所有转换函数是否读取该字段。

---

### 151ca802 fix combineARN精度问题
- 根因类别：buffer空间分配不足
- 涉及文件：mc2/moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_v2_tiling.cpp, mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2.h
- 缺陷描述：ReduceSum临时buffer分配`NUM_PER_REP_FP32*sizeof(float)`(256字节)，但H=8192时ReduceSum接口实际需要512字节。空间不足导致写越界，污染相邻buffer，表现为精度问题。
- 修复模式：buffer大小改为`NUM_PER_REP_FP32*sizeof(float)*2`，tiling侧和kernel侧同步修改。附带将gamma加载从循环内提到循环外。
- 可审查性：低(ReduceSum对buffer的最小空间需求是隐式的，需查API文档公式)
- 审查规则建议：调用ReduceSum/ReduceMax等归约API时，要求注释标明buffer大小计算依据(引用API文档公式)，并校验覆盖最大shape下的需求。建议封装CalcReduceSumBufferSize工具函数。

---

### 63f9fff6 fix gmm no quant l2 cache
- 根因类别：提前返回时状态未重置
- 涉及文件：gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_no_quant_matmul_tiling.cpp
- 缺陷描述：`SetDisableL2Cache`中当totalSize<l2Size时直接return，但未将`weightNoL2Cache_`设为false。该字段可能在之前流程中被设为true，导致不需要禁用L2的场景误走双页表路径，950非量化性能劣化。
- 修复模式：early return前补加`weightNoL2Cache_=false;`。1行代码。
- 可审查性：高(Set类函数的early return路径遗漏状态更新，模式清晰)
- 审查规则建议：Set/Configure类函数的所有return路径必须对目标状态变量有显式赋值。审查early return是否遗漏状态重置。

---

### ae4c91e3 追加伪量化责任田

非缺陷提交(配置更新)。仅修改classify_rule.yaml添加文件归属配置。从defect_commits中排除。

---

### 91ef89b4 GroupedmatmulFinalizerouting pertoken量化，weightnz模式aclnn通路

非缺陷提交(新功能)。为GMR算子增加pertoken量化+weightNz模式aclnn通路。PR标签为"新特性"。从defect_commits中排除。

---

### 7fbc7bc3 GMMSQ算子异常场景提交

非缺陷提交(功能增强/参数校验补充)。PR标签"新特性"。但有防御性价值：补充了dequantMode和quantMode必须相等的校验。记录参数一致性校验模式供后续归纳。

---

### 108b4dd3 [experimental FIA] 编译报错修复&&cmakelist整改

非缺陷提交(构建系统整改)。CMakeLists简化、tiling注册统一化、kernel冗余条件分支清理。从defect_commits中排除。

---

### 234c1c8b rope_with_sin_cos_cache fix 950 devide 0 error
- 根因类别：除零错误(tiling计算除数为零)
- 涉及文件：posembedding/rope_with_sin_cos_cache/op_host/rope_with_sin_cos_cache_tiling.cpp
- 缺陷描述：`maxNPerLoopForUb`和`numHeadsForUb`在Ascend950+headDim>8192场景下可能为0。代码4处用这两个变量做除数进行向上取整计算，除零触发CoreDump。
- 修复模式：4处除法前增加三元表达式`var==0 ? 0 : (a+var-1)/var`零值保护。
- 可审查性：高(除零防护是基本审查项)
- 审查规则建议：所有除法运算必须确保除数非零，特别是UB大小计算得来的变量(不同硬件平台UB容量差异可能导致零值)。可自动扫描`/variable`形式除法检查零值保护。

---

### af418dfc 魔鬼数字问题修改

非缺陷提交(代码规范)。将字面量128替换为命名常量NUM_128。静态分析告警消除。从defect_commits中排除。

---

### c2250fc9 修复FAG aclnn非连续场景下获取ShapeSize问题
- 根因类别：API误用(非连续tensor的Size()语义错误)
- 涉及文件：attention/flash_attention_score_grad/op_api/aclnn_flash_attention_score_grad.cpp
- 缺陷描述：`GetInputShapeInfo`中用`query->Size()`获取tensor大小，但Size()返回StorageShape大小。非连续tensor的StorageShape可能大于逻辑元素数。调用发生在转contiguous之前，导致各轴维度推导有误。
- 修复模式：将`query->Size()`改为`queryShape.GetShapeSize()`(基于逻辑shape)，key同理。
- 可审查性：高(Size() vs GetShapeSize()语义差异明确)
- 审查规则建议：aclnn代码中获取tensor元素数优先用`GetViewShape().GetShapeSize()`而非`->Size()`。标记所有`->Size()`调用检查是否在contiguous转换之前。

---

## 批次1统计

总计20条提交:
- 实际缺陷修复: 12条
- 非缺陷(排除): 8条 (fe0bee0d, e8ffb3b9, 53c21a1d, ae4c91e3, 91ef89b4, 7fbc7bc3, 108b4dd3, af418dfc)

缺陷根因初步分类:
| 根因类别 | 数量 | commit |
|---------|------|--------|
| 边界条件缺失/tiling约束 | 2 | 5079260e, c3cf4d96 |
| 参数校验问题 | 1 | 1216e10a |
| 配置与代码不一致 | 1 | 6a573aa2 |
| 变量遮蔽/模板参数错误 | 1 | a882f8c0 |
| 硬件流水线同步缺失 | 1 | 984d8a72 |
| DataCopy非对齐越界 | 1 | f14d0284 |
| 结构体字段/数据传递遗漏 | 1 | 8e9fe171 |
| buffer分配不足 | 1 | 151ca802 |
| 提前返回状态未重置 | 1 | 63f9fff6 |
| 除零错误 | 1 | 234c1c8b |
| API误用(tensor语义) | 1 | c2250fc9 |

---

## 批次2: 提交 #21-#40 (2026-02-26 ~ 2026-02-28)

---

### 28671df1 ROPE 修复kernel侧拦截问题

非缺陷提交(新功能)。大规模新增约2956行，为ROPE算子添加rotate matrix功能支持和V2 API。新增rotate可选输入、aclnnRotaryPositionEmbeddingV2接口、rotate_matrix.h(681行)、tiling文件(497行)。本质是功能扩展+API重构。从defect_commits中排除。

---

### 8a09dcd0 修改classify_rule路径配置错误

非缺陷提交(CI配置修正)。classify_rule.yaml中路径缺少`ops/`前缀，从`ops-transformer/posembedding/...`改为`ops/ops-transformer/posembedding/...`。与ae4c91e3同类，属构建分类配置。从defect_commits中排除。

---

### 14a289de fix constraint of ratio between numHeads and numKeyValueHeads

非缺陷提交(文档修改)。所有修改均在.md文档文件中(共12个文件)，移除"numHeads与numKeyValueHeads比值不能大于64"的约束描述。无运行时代码变更。从defect_commits中排除。

---

### f73c0505 输入异常分析

非缺陷提交(新功能)。为mc2/tools/dump_analysis/诊断工具新增输入异常检测分析能力(check_dis_com、check_mask、check_topk等)，新增754行。属于工具层新功能开发。从defect_commits中排除。

---

### 831ab170 S1G invalidrows bug code review
- 根因类别：数据类型溢出(uint32乘法溢出)
- 涉及文件：attention/common/op_kernel/vector_common.h
- 缺陷描述：`DealInvalidRowsBelow`函数中`dealRowOffset`声明为`uint32_t`，但通过`s1BottomTok * params.gSize`计算，当两值较大时uint32溢出(最大~43亿)，导致偏移量错误、越界写入或逻辑异常。附带问题：while循环条件中`s1 >= s1BottomTok`对uint32_t恒成立，属冗余条件。
- 修复模式：将`dealRowOffset`从`uint32_t`提升为`uint64_t`；移除循环中冗余条件。
- 可审查性：中
- 审查规则建议：kernel代码中偏移量/索引计算变量，审查类型是否足以承载最大值。两个uint32_t相乘或结果用作数组偏移时，建议使用uint64_t。

---

### 967fed57 编译头文件包含路径bugfix

非缺陷提交(编译路径兼容性)。添加`__has_include`预处理条件判断，头文件路径不存在时回退到备选路径。编译期适配，不涉及运行时逻辑。从defect_commits中排除。

---

### f38d4a49 NTD_TND Dsize拦截信息修复

非缺陷提交(错误消息文案修正)。将错误报告中硬编码"NTD"替换为动态变量`layoutStr.c_str()`，仅影响日志输出内容，不改变校验逻辑。从defect_commits中排除。

---

### 613ae40b revert//新增ROPEV2接口支持辅助矩阵输入

非缺陷提交(功能revert)。cann-robot自动执行revert，撤销ROPEV2接口及rotate matrix功能。删除V2头文件、实现文件、tiling文件、rotate_matrix.h，从proto/json/kernel中移除rotate参数。此revert将在阶段3专项分析。从defect_commits中排除。

---

### da61be50 fa infershape修改
- 根因类别：InferShape边界条件逻辑错误
- 涉及文件：attention/flash_attention_score/op_host/flash_attention_score_infershape.cpp
- 缺陷描述：FlashAttentionScore的InferShape中，当key的N2维度计算结果为0时(h2/D1==0，发生在h2<D1场景)，输出attention_out的第3维被错误设置为0。正确值应为N1*D1。输出shape为0导致下游shape不匹配或计算异常。
- 修复模式：将边界条件(N2==0)下的输出shape从硬编码0修正为N1*D1。
- 可审查性：高
- 审查规则建议：InferShape函数中所有`SetDim(idx, 0)`代码路径需重点审查，特别是除法结果为0的边界条件。输出shape为0应极少见，出现时需确认是否确实应为空tensor。

---

### 1065427e fix_transformer_graph_extension

非缺陷提交(新功能/重构)。虽以"fix"开头，实际是为moe_distribute算子添加graph mode(torch.compile)支持：新增GE Converter、将op_module.load()提升到模块级、重写Meta函数签名。从defect_commits中排除。

---

### a44e3e7e 修复使能PA时获取VD错误
- 根因类别：多模式shape解析错误(未区分PA/非PA模式)
- 涉及文件：attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_v2.cpp
- 缺陷描述：PA(PagedAttention)模式下value tensor的storage shape格式不同(BBH 3维或BND1BD0 5维)，但各layout分支统一用`GetStorageShape().GetDim(VALUE_DIM_X)`取valueD，PA模式下得到错误值。
- 修复模式：前置检测是否启用PA(BLOCK_TABLE_INDEX是否存在)，若启用则根据V维度数(3维/5维)用不同公式计算正确valueD；后续各分支用条件选择。
- 可审查性：中
- 审查规则建议：处理多种tensor格式/模式时，所有`GetStorageShape().GetDim()`路径必须考虑各模式下shape语义差异。存在模式标志时，每处shape获取应有模式guard。

---

### 79a2aced fix: fix warnings

非缺陷提交(编译告警消除+代码规范化)。修改7个文件700+行，全部为：代码格式统一(花括号位置)、添加const限定符、删除未使用include/参数、static改static inline。从defect_commits中排除。

---

### 988a83a9 aclnnGroupedMatmulV3接口兼容int8输入,int8输出场景

非缺陷提交(新功能)。PR标签"新特性"。新增V3版本int8输入/输出校验方法(CheckInputAndOutputDtypeForV3Version等)、新增UT测试、更新文档。从defect_commits中排除。

---

### 28daf0ab feature: infer FA, Ascend950, deprecate api

非缺陷提交(新功能/API废弃拦截)。在Ascend950上拦截旧版API(FIAS V1-V4、IFA V1-V3、PFA V1-V2)，升级fallback路径到V5，更新接口文档。从defect_commits中排除。

---

### 5e8817a7 fix bug for TND
- 根因类别：条件判断逻辑错误(多模式统一条件不正确)
- 涉及文件：attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_varlen.cpp
- 缺陷描述：varlen tiling分核优化判断中，`LEFT_UP_CAUSAL`和`RIGHT_DOWN_CAUSAL`两种sparse mode使用统一条件`actualSeqLenData[i] >= thresholdS2Size && actualSeqLenKvData[i] > thresholdS2Size`。但LEFT_UP_CAUSAL模式下有效计算量取决于`min(seqLen, kvLen)`，不是两者都必须超阈值。导致特定序列长度组合下分核优化被错误跳过或错误开启。
- 修复模式：对LEFT_UP_CAUSAL取min(seqLen, kvLen)与阈值比较，对RIGHT_DOWN_CAUSAL仍用kvLen。分模式特化条件。
- 可审查性：高
- 审查规则建议：同一代码路径处理多个枚举/模式值时，审查每个模式的数学语义是否一致。不同模式语义不同则应设独立条件分支，不用统一条件覆盖。

---

### be6bcad7 【日志优化】gmm算子AlogCheckDebugLevel废弃日志接口替换

非缺陷提交(API迁移)。将4个文件中`AlogCheckDebugLevel`机械替换为`CheckLogLevel`，函数签名/参数/逻辑完全不变。从defect_commits中排除。

---

### 4d3bbf03 编译头文件包含路径bugfix

非缺陷提交(编译路径兼容性)。与967fed57同类，在两个头文件中添加`__has_include`预处理回退路径。从defect_commits中排除。

---

### 498342e0 GQA perblock全量化添加拦截
- 根因类别：输入校验缺失(多维度遗漏)
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：`CheckPerblockQuantParams`存在两处遗漏：(1) 缺少对deqScale1、quantScale1、deqScale2、antiquantScale等不支持参数的非空拦截，传入不支持参数时不报错走入未定义行为；(2) 不支持layout列表不完整，原仅拦截TND，实际BNSD_NBSD、BSH_NBSD、BSH_BNSD、BSND_BNSD、BSND_NBSD、NTD也不支持。
- 修复模式：增加额外参数非空检查；将layout拦截从单一枚举扩展为不支持列表匹配。
- 可审查性：中
- 审查规则建议：多量化模式下所有输入参数合法性校验需逐项检查完备性。layout枚举拦截应使用白名单(支持列表)而非黑名单(不支持列表)以减少遗漏。

---

### 0104fc64 修复GQA非量化tiling下沉误拦截
- 根因类别：条件判断逻辑错误(模式标志位遗漏)
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：两处逻辑缺陷。(1) `CheckPseShiftTypeAndShape`中pseShift shape校验未考虑`isMaxWorkspace`标志——MaxWorkspace模式下shape用最大值占位，校验条件不适用但被触发，导致合法请求被误拦截。(2) `GetMaxWorkspaceFlag`遗漏对`actualSharedPrefixLen`的空数据指针判断，该tensor存在但数据为空时应判定为MaxWorkspace模式，否则后续访问空指针。
- 修复模式：校验包裹在`!isMaxWorkspace`条件下；补充空数据指针判断。
- 可审查性：中
- 审查规则建议：存在"模式标志位"控制路径时，所有依赖该模式的校验需检查是否在正确模式分支下。新增tensor输入时检查所有引用"同类tensor列表"的地方是否需同步更新。

---

### f5f79a3e 【FAG】fix tnd s1s2 exist zero
- 根因类别：循环内布尔标志覆盖(应为累积)
- 涉及文件：attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp
- 缺陷描述：`GetShapeAttrsInfo`循环中，`tndBaseInfo.isSeqExistZero`使用直接赋值`=`而非累积或`|=`。循环遍历多batch的qLen/kvLen检测零长度序列，但每次迭代覆盖前次结果，仅最后一个batch的判断生效。前面batch出现零长度序列信息被丢失。
- 修复模式：从`= (qLen == 0 || kvLen == 0)`改为`= (isSeqExistZero || (qLen == 0 || kvLen == 0))`。
- 可审查性：高
- 审查规则建议：循环内对布尔标志赋值，若语义为"是否存在某条件"必须用`|=`或`= (old || new)`。直接赋值`=`仅适用于"最终迭代即最终结果"。可编写静态规则：循环体内布尔变量直接赋值(非`|=/&=`)发出警告。

---

## 批次2统计

总计20条提交:
- 实际缺陷修复: 7条
- 非缺陷(排除): 13条 (28671df1, 8a09dcd0, 14a289de, f73c0505, 967fed57, f38d4a49, 613ae40b, 1065427e, 79a2aced, 988a83a9, 28daf0ab, be6bcad7, 4d3bbf03)

缺陷根因分类:
| 根因类别 | 数量 | commit |
|---------|------|--------|
| 数据类型溢出(uint32) | 1 | 831ab170 |
| InferShape边界条件 | 1 | da61be50 |
| 多模式shape解析错误 | 1 | a44e3e7e |
| 条件判断逻辑错误(多模式) | 1 | 5e8817a7 |
| 输入校验缺失(多维度) | 1 | 498342e0 |
| 模式标志位遗漏(误拦截) | 1 | 0104fc64 |
| 循环内布尔标志覆盖 | 1 | f5f79a3e |

## 批次3: 提交 #41-#60 (2026-02-14 ~ 2026-02-26)

---

### fcff1be7 [bugfix] 修复TND问题&修复资料问题
- 根因类别：workspace内存分配大小计算错误(多layout分支遗漏)
- 涉及文件：attention/dense_lightning_indexer_grad_kl_loss/op_kernel/dense_lightning_indexer_grad_kl_loss_base.h
- 缺陷描述：TND layout场景下，`dKeyIndexFloatSzie`计算使用了`bSize * s2Size`，但TND格式下token维度是连续展开的(T维度)，应从`actualSeqLengthsKeyGm`取实际序列长度`t2Size`来计算。错误的workspace大小可能导致内存越界或计算错误。
- 修复模式：增加`if constexpr (LAYOUT_T == DLILayout::TND)`编译期分支，TND场景用实际`t2Size`替代`bSize * s2Size`。
- 可审查性：中
- 审查规则建议：多layout算子中涉及shape维度乘积计算的地方，检查每种layout是否都有正确的处理分支。特别关注workspace/buffer大小计算是否遗漏layout分支。

---

### e46032f7 update error msg process

非缺陷(日志/可调试性改进)。文档示例代码中增加error message获取方法，不涉及运行时功能代码。

---

### 05e3ba28 ifa pfa kernel_operator.h 最后清理

非缺陷(编译优化/重构)。将`kernel_operator.h`大头文件替换为精确的`kernel_vec_intf.h`+`kernel_cube_intf.h`，无运行时行为变化。

---

### e48d4172 fix bug for eigen 5.0.0's line endings

非缺陷(构建配置)。修改eigen第三方库下载URL以解决换行符问题，不影响运行时行为。

---

### 89a58c3a 修复拦截问题
- 根因类别：参数校验(拦截)逻辑不完整，遗漏多种layout组合和功能互斥场景
- 涉及文件：attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_v2.cpp, attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp, attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.h
- 缺陷描述：三类bug：(1) `CheckNormalTensorList`中BSH检查遗漏`BSH_BNSD`和`BSH_NBSD`组合layout，BNSD检查遗漏`BNSD_NBSD`，导致这些layout下tensor list shape一致性校验被跳过。(2) 缺少`CheckRopeDataType`函数，rope tensor数据类型与query/key不一致时未拦截。(3) `CheckQuant`遗漏MLA不支持antiquant的拦截，`CheckPseShiftTypeAndShape`遗漏PFARope下MLA不支持pseShift的拦截。
- 修复模式：扩展layout字符串匹配范围，新增`CheckRopeDataType`校验函数，补充遗漏的互斥条件。
- 可审查性：中
- 审查规则建议：新增layout类型或功能特性时，搜索所有引用相关layout/flag的条件分支确保覆盖。建立layout与feature兼容性矩阵自动校验。

---

### 38c7162c 【PFA】删除keydim1 < blockNumValid的校验
- 根因类别：错误校验导致合法输入被拒绝(over-validation)
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp
- 缺陷描述：PagedAttention模式下，`CheckPAWhenBaseApi`和`CheckPATypeAndShape`校验了`keyDim1 < blockNumValid`，但key的block pool可以共享/复用，keyDim1完全可以小于blockNumValid。多余校验导致合法输入被拒绝。
- 修复模式：直接删除两处`OP_CHECK_IF(keyDim1 < blockNumValid)`校验。
- 可审查性：中
- 审查规则建议：删除校验逻辑时检查是否存在对称位置的相同错误校验(copy-paste pattern)。新增校验时需附带该约束的技术依据。

---

### 68af3c15 aclnnGroupedMatmulV5接口支持量化场景

非缺陷(文档+测试新增)。文档补充量化场景参数描述，测试文件补充量化测试用例。

---

### 1ffefbce add inferGroupSize for mmar mmrs & agmm
- 根因类别：缺失参数自动推断逻辑 + 模板类型不匹配
- 涉及文件：mc2/all_gather_matmul_v2/op_host/op_tiling/arch35/all_gather_quant_bmm_tiling.cpp, mc2/common/inc/tiling/mc2_tiling_utils.h, mc2/common/src/mc2_tiling_utils.cpp, mc2/matmul_all_reduce/op_host/op_tiling/arch35/quant_matmul_all_reduce_tiling_950.cpp, mc2/matmul_all_reduce/op_host/op_tiling/matmul_all_reduce_tiling_base.cpp, mc2/matmul_all_reduce/op_host/op_tiling/matmul_all_reduce_tiling_base.h, mc2/matmul_reduce_scatter_v2/op_host/op_tiling/arch35/quant_bmm_reduce_scatter_tiling.cpp
- 缺陷描述：三个MC2算子处理量化groupSize参数(打包的uint64，M/N/K各占16bit)时，部分维度为0表示"由算子自动推断"，但原代码无推断逻辑，直接用0值做校验/计算导致合法输入被拒或tiling计算错误。另有`GetAttrPointer<uint64_t>`应为`int64_t`的类型不匹配。
- 修复模式：新增`Mc2TilingUtils::InferGroupSize()`公共函数做自动推断；三算子统一调用后再校验；修复类型不匹配。
- 可审查性：低
- 审查规则建议：输入参数部分字段可能为0/空时，检查是否有默认值推断或fallback逻辑。`GetAttrPointer`模板参数类型与属性注册类型一致性校验。

---

### 57a30f3a 修复GroupedMatmulAdd算子参数不完全匹配问题

非缺陷(纯API文档修正)。修正文档表格中的参数名和方向描述。

---

### a0733b24 repair S1G InvalidRows boundary condition
- 根因类别：嵌套循环边界条件逻辑错误
- 涉及文件：attention/common/op_kernel/vector_common.h
- 缺陷描述：`DealInvalidRowsBelow`函数用嵌套while+for循环对无效行清零，存在多处逻辑缺陷：(1) 内层while用`s1Stride`做步进判断不精确，可能跳过需清零的行或多清零有效行。(2) 内层`if (s1 == s1BottomTok) break`在`s1 < s1BottomTok`条件下是死代码。(3) 迭代变量`i`在内外层循环中被共同修改，难以保证正确覆盖所有case。
- 修复模式：重写为先算偏移再单层while循环的线性扫描模式，消除嵌套循环的边界条件错误。
- 可审查性：中
- 审查规则建议：内外层循环共同修改同一迭代变量的模式极易出错。while循环条件含`&&`但循环体首行break同一条件的死代码检测。

---

### 1a62a495 [FIA] aclnnFusedInferAttentionScoreV4 (splitfusepa) 算子MHA场景功能泛化

非缺陷(功能泛化)。放宽MHA场景支持条件，扩大算子支持范围，非bug修复。

---

### cf2a33d1 修复GQA perblock 全量化qs=1下的拦截信息不匹配问题
- 根因类别：输入校验缺失 + 错误信息不准确
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：(1) `CheckPerTensorQuantParams`缺少对rope的拦截 — pertensor量化场景下使用rope会导致功能异常，但未拦截，用户传入rope时静默产生错误结果。(2) perblock场景的错误消息写"PFARope"应为"Rope"，拦截信息与实际约束不匹配。
- 修复模式：新增`OP_CHECK_IF(enablePFARope, ...)`校验；修正错误消息文本。
- 可审查性：高
- 审查规则建议：新增输入约束时检查所有相关校验函数是否都覆盖；错误消息应与用户可见的参数名/概念一致。

---

### b0182bb3 [FIX]修复s1RealSize为0时没有特殊处理的问题
- 根因类别：除零错误(边界条件缺失)
- 涉及文件：attention/common/op_kernel/arch35/infer_flash_attention_kvcache.h
- 缺陷描述：`ComputeParamS1`函数中，`s1RealSize`为0时`halfS1RealSize`也为0，导致`(runParam.nextTokensPerBatch * (-1)) / runParam.halfS1RealSize`除以零，device kernel中可能崩溃。
- 修复模式：条件判断最前面加入`runParam.s1RealSize == 0`短路判断(用`||`)，为0时直接返回true跳过后续计算。
- 可审查性：高
- 审查规则建议：作为除数的变量必须检查零值边界；静态分析时追踪分母变量取值范围确保不存在零值路径。

---

### 9081122a fix invalid link

非缺陷(纯文档链接修复)。修复markdown链接路径，不涉及运行时代码。

---

### 0162cad4 [mc2] fix wqmmar supported data type in pertensor scenario

非缺陷(纯文档描述修正)。修正数据类型支持场景的文档描述，不修改代码。

---

### 7af7c963 kv fix check shape
- 根因类别：数据类型枚举遗漏 + 空tensor未校验
- 涉及文件：posembedding/kv_rms_norm_rope_cache/op_host/kv_rms_norm_rope_cache_base_tiling.cpp, posembedding/kv_rms_norm_rope_cache/op_host/kv_rms_norm_rope_cache_regbase_full_load_tiling.cpp, posembedding/kv_rms_norm_rope_cache/op_host/kv_rms_norm_rope_cache_regbase_recompute_tiling.cpp, posembedding/kv_rms_norm_rope_cache/op_host/kv_rms_norm_rope_cache_tiling.h
- 缺陷描述：(1) 对齐校验中`kcacheDtype`/`vcacheDtype`判断仅覆盖`DT_INT8`，遗漏`DT_HIFLOAT8`/`DT_FLOAT8_E5M2`/`DT_FLOAT8_E4M3FN`三种量化类型，走错误的FP16对齐分支(16B vs 32B)导致数据对齐错误。(2) `DoOpTiling`入口缺少空tensor校验，传入空tensor会导致tiling访问无效shape。
- 修复模式：扩展dtype判断条件覆盖所有量化类型(四处同步修改)；新增`CheckInputShapeIsEmpty()`前置校验。
- 可审查性：高
- 审查规则建议：基于枚举值的switch/if分支检查是否覆盖所有有效值，特别关注后续新增的枚举成员。tensor输入算子检查是否有空tensor前置校验。

---

### b5d7c0fe [FAG] fix rope empty tensor bug
- 根因类别：空tensor场景多层联动处理缺失
- 涉及文件：attention/flash_attention_score_grad/op_api/aclnn_flash_attention_score_grad.cpp, attention/flash_attention_score_grad/op_host/flash_attention_score_grad_infershape.cpp, attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling.cpp, attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_empty_tensor_regbase.h, attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_tiling_data_regbase.h, attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad_apt.cpp
- 缺陷描述：三个关联bug：(1) BSH/SBH layout下`dDim`为0时直接报错，但rope空tensor场景下这是合法输入。(2) InferShape对rope shape的判断条件`!= nullptr && GetShapeSize() != 0`导致空tensor时输出shape未赋值。(3) empty tensor kernel缺少dqRope/dkRope的清零处理，输出含脏数据。
- 修复模式：op_api层增加D=0分支；infershape层放宽shape赋值条件；tiling/kernel层增加rope输出清零逻辑。跨4层联动修复。
- 可审查性：中
- 审查规则建议：empty tensor/边界case下检查所有输出tensor是否被正确初始化(清零)；可选输入的对应可选输出在各组合场景下的处理完备性。

---

### 3e056c8d 修复GroupedMatmulSwigluQuantV2、GroupedMatmulV4、GroupedMatmulV3算子问题
- 根因类别：示例代码中的VLA使用不当 + 空数组越界访问 + 非法内存释放
- 涉及文件：gmm/grouped_matmul/docs/aclnnGroupedMatmulV3.md, gmm/grouped_matmul/docs/aclnnGroupedMatmulV4.md, gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantV2.md
- 缺陷描述：文档示例代码中三处bug：(1) `aclTensor* tensors[size]`是C99 VLA，C++中不标准。(2) `aclCreateIntArray(data, 1)`但data为空vector，越界读取。(3) 对栈上数组指针调用`aclrtFree`属非法释放。
- 修复模式：VLA改`std::vector`；修正size参数；删除非法`aclrtFree`调用。
- 可审查性：高
- 审查规则建议：示例代码避免VLA；API调用size参数与实际数据长度匹配；`aclrtFree`参数确认为device侧分配的内存。
- 备注：此commit主体为文档示例代码修复，同时包含拼写修正等纯文档变更。

---

### 5c48d0be 补充groupListType描述

非缺陷(纯文档修改)。补充`aclnnGroupedMatmulWeightNz.md`中groupListType约束描述。

---

### 8f3a5747 修改整仓错误链接及错误产品名称

非缺陷(纯文档修改)。覆盖89个md文件修正链接路径、产品名称、术语统一。

---

## 批次3新增缺陷类别

| 类别 | 本批新增数 | hash |
|------|-----------|------|
| workspace/buffer大小计算错误(多layout) | 1 | fcff1be7 |
| 参数校验遗漏(layout组合/功能互斥) | 1 | 89a58c3a |
| 错误校验导致合法输入被拒(over-validation) | 1 | 38c7162c |
| 缺失参数自动推断逻辑 + 类型不匹配 | 1 | 1ffefbce |
| 嵌套循环边界条件逻辑错误 | 1 | a0733b24 |
| 输入校验缺失(量化+rope互斥) | 1 | cf2a33d1 |
| 除零错误(边界条件缺失) | 1 | b0182bb3 |
| 数据类型枚举遗漏 + 空tensor未校验 | 1 | 7af7c963 |
| 空tensor场景多层联动处理缺失 | 1 | b5d7c0fe |
| 示例代码缺陷(VLA/越界/非法释放) | 1 | 3e056c8d |

---

## 批次4: 提交 #61-#80 (2026-02-12 ~ 2026-02-14)

---

### 1f3291bf fix FA empty tensor n2=0 and d=0 div 0 problem
- 根因类别：除零错误(空tensor边界)
- 涉及文件：attention/flash_attention_score/op_api/aclnn_flash_attention_score.cpp
- 缺陷描述：AnalysisAxisForBsh和AnalysisAxisForSbh函数中，当n2=0时用n2做除数计算dk和dv会触发除零崩溃。原代码对dSize==0只做了提前return但没有保护n2==0的除法路径。
- 修复模式：去掉dSize==0的提前return，改为在每个除法前加三元运算符保护——当d==0时n2赋0，当n2==0时dk/dv回退为d值。
- 可审查性：高
- 审查规则建议：所有除法运算前检查除数是否可能为零；shape解析函数中空tensor(维度为0)的边界case是否被覆盖。

---

### 44e84d52 修复tensorlist场景B全为0被误拦截，添加非量化4维mask第2维异常校验
- 根因类别：校验逻辑顺序错误 + 校验条件缺失
- 涉及文件：attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_v2.cpp, attention/incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp, attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：(1) CheckTensorList中遍历tensorlist读取shape的while循环内就检查Batch维是否为1，但B全为0是合法空tensor场景，会被误拦截(CheckEmptyTensorList在后面才调用)。(2) IFA和PFA的CheckMaskShape中，对非量化4维mask的第2维(N维)没有做!=1的校验。
- 修复模式：(1) 将Batch==1校验移到CheckEmptyTensorList之后。(2) 增加attenMaskN==1的校验条件。
- 可审查性：中
- 审查规则建议：输入校验逻辑是否在合法提前退出(如空tensor判断)之后才执行；mask shape每个维度是否都有对应校验。

---

### 0c90fb3c 修复CMake构建脚本中JSON库头文件包含路径设置错误的问题
- 根因类别：构建配置错误(include路径指向错误)
- 涉及文件：cmake/third_party/json.cmake
- 缺陷描述：原代码通过set_target_properties设置INTERFACE_INCLUDE_DIRECTORIES指向了安装路径，但该路径在ExternalProject构建完成前可能不存在或内容不正确。
- 修复模式：删除错误的set_target_properties调用。
- 可审查性：中
- 审查规则建议：CMake ExternalProject的头文件路径应指向实际源码/构建产物路径，而非尚未生成的安装路径。

---

### 8ceb72b6 grouped_matmul_swiglu_quant_v2 修复日志提示不合理的问题
- 根因类别：输入校验缺失(dtype白名单)
- 涉及文件：gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/aclnn_grouped_matmul_swiglu_quant_v2_utils.h
- 缺陷描述：量化场景下，当x或weight的dtype不在任何支持列表(FP8/MXFP4/Pertoken)中时，原代码没有提前拦截，会穿透到后续组合dtype匹配逻辑，产生未定义行为。
- 修复模式：在进入具体分支之前增加对x dtype和weight dtype的全量支持列表检查。
- 可审查性：高
- 审查规则建议：校验函数应在入口处对所有输入参数做白名单检查，而不仅在分支内部做局部检查。

---

### 368c7325 fix ScatterPaKvCache 同步问题
- 根因类别：硬件流水线同步缺失
- 涉及文件：attention/scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_rope_fully_load.h, attention/scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_rope_not_fully_load.h
- 缺陷描述：(1) fully_load中Cast运算后紧接DataCopyPad写出GM，缺少V_MTE3同步屏障。(2) not_fully_load中V_MTE3同步位置错误，放在Div之后而非Cast之后。(3) 循环内重复调用slotMappingLocal.GetValue(iter)。
- 修复模式：在Cast后DataCopyPad前插入V_MTE3同步；将同步位置修正到正确依赖点；提取循环不变量。
- 可审查性：低
- 审查规则建议：Vector计算结果写出GM(DataCopyPad)前必须有V_MTE3同步屏障；审查同步原语位置是否紧邻被保护的操作对。

---

### 349083a7 Revert "修改了all_gather_matmul算子ut的op_host组件的用例输入方式，改成csv表格"

非缺陷提交。被revert的内容是UT测试代码的输入方式重构(csv改回硬编码)，不涉及产品代码逻辑。

---

### 44338c14 修复alltoallmatmul的api校验
- 根因类别：校验逻辑条件分支错误(量化模式互斥)
- 涉及文件：mc2/allto_all_matmul/op_api/aclnn_allto_all_quant_matmul.cpp
- 缺陷描述：CheckAllDtypesValid中，DYN_PERTOKEN_QUANT场景的if块结束后，紧接着对所有x1ScaleOptional!=nullptr的情况都用SCALE_DTYPE_SUPPORT_LIST做校验，覆盖了DYN_PERTOKEN_QUANT的特殊处理。同时缺少PERTOKEN_QUANT场景的校验。
- 修复模式：将独立的if改为else if，使DYN_PERTOKEN_QUANT和PERTOKEN_QUANT各走独立校验路径。
- 可审查性：高
- 审查规则建议：多个量化模式的校验分支应用if-else if确保互斥，避免后续校验覆盖前面的特殊处理。

---

### bd65dfcc 修改GmmSwigluQuantV2算子aclnn通路调用宏函数输出报错信息不明确的异常

非缺陷提交。纯代码重构，将成员变量访问改为参数传递以改善宏展开时的报错信息可读性，校验逻辑行为完全不变。

---

### 4efbd43d 空tensor初始化添加同步
- 根因类别：硬件流水线同步缺失
- 涉及文件：attention/prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_zero_output.h
- 缺陷描述：PromptFlashAttentionZeroOutPut::Process中，先对output做DuplicateZero/DuplicateInf初始化写出GM，再对softMaxLse做初始化写出GM。两段写出之间及之后都缺少MTE3_V同步屏障，可能导致数据竞争。
- 修复模式：在两段GM写出操作之间和最后一段之后各插入MTE3_V的SetFlag/WaitFlag同步。
- 可审查性：低
- 审查规则建议：所有对GM的写出操作之后，如果后续还有对GM的操作，必须插入对应的同步屏障。

---

### eb8f0ac7 MLA非量化mask bugfix
- 根因类别：赋值遗漏 + 错误的提前返回条件
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_kernel_noquant_mla.h, attention/common/op_kernel/arch35/flash_attention_score_block_vec_base.h
- 缺陷描述：(1) ComputeC函数初始化attenMaskInfo时遗漏attenMaskS1Size赋值，MLA非量化路径中mask的S1维度信息丢失。(2) MlaBoolCopyInRegbase中totalS2Size%blockBytes!=0时直接return，非对齐情况也应正常处理。
- 修复模式：补充attenMaskS1Size赋值；删除错误的提前返回判断。
- 可审查性：中
- 审查规则建议：初始化结构体时应逐字段检查是否有遗漏；提前返回条件是否真代表不可处理的情况。

---

### 9bb73d4c [FAG] error info modify

非缺陷提交。只修改了错误信息字符串中的数字(ceil(S2/128)->ceil(S2/256))，校验逻辑条件表达式完全没变，纯日志文本修正。

---

### 4573c2e ActSeqLen拦截信息修复
- 根因类别：校验逻辑条件错误(多余条件放行非法配置)
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：MLA场景下ActSeqLen的layout校验条件 `inputLayout != InputLayout::TND && inputLayout != InputLayout::NTD` 等价于放行了TND和NTD，但NTD不应被放行——正确行为是只允许TND(以及TND_NTD变体)。
- 修复模式：将条件改为 `inputLayout != InputLayout::TND`，只允许TND通过校验。
- 可审查性：中
- 审查规则建议：当校验逻辑使用多个"不等于"条件组合时，逐一确认每个被排除的值是否确实应该被排除；检查错误提示文本与实际校验逻辑是否一致。

---

### 46bdb193 补充grouplistType描述

非缺陷提交。只修改了.md文档文件，补充groupListType说明，纯文档更新。

---

### 3881d5c gmm算子aclnn异常场景拦截打印日志修复
- 根因类别：校验逻辑不完整(缺少上界检查) + printf格式符类型不匹配
- 涉及文件：gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul.cpp
- 缺陷描述：(1) 对x tensor的维度数只检查了下界 `xDimNum >= MIN_FM_DIM`，缺少上界检查，超过MAX_FM_DIM(6)时不会被拦截。(2) weight维度日志中使用%d打印size_t类型的GetDimNum()返回值，64位平台可能打印错误值。
- 修复模式：增加上界校验 `xDimNum <= MAX_FM_DIM`；修正%d为%zu。
- 可审查性：高
- 审查规则建议：数值范围校验必须同时检查上界和下界；printf格式符必须与实参类型严格匹配。

---

### a6aa6f17 revert//优化头文件

非缺陷提交。被revert的原始PR内容是将细粒度头文件include替换为聚合头文件及尾部空格清理，纯重构/清理性质。

---

### 0c3b9b38 Fix Debug: Min HCCL_BUFFSIZE may result in cleaning bufferId
- 根因类别：硬编码常量导致缓冲区越界清理
- 涉及文件：mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_a2_layered.h
- 缺陷描述：清理token end flag时使用硬编码常量MAX_BS_NUM=256，但实际最大batch size是运行时根据globalBs_/worldSize_动态计算的。当HCCL_BUFFSIZE为极限最小值时，buffer按实际maxBs分配，但cleanup固定按256清理，越界覆盖相邻区域的bufferId等元数据。
- 修复模式：删除硬编码MAX_BS_NUM=256，用运行时计算的maxBs_替换，确保清理范围与分配大小一致。
- 可审查性：中
- 审查规则建议：buffer分配大小和使用/清理大小必须来自同一数据源，禁止分配用动态值、清理用硬编码常量的模式。

---

### 1734ac98 修正pr_1711对moe算子的修改错误
- 根因类别：前序PR引入多处逻辑错误(CMake条件取反、芯片型号硬编码、batch size上限不区分模式、缺少调度模式设置)
- 涉及文件：mc2/moe_distribute_combine_v2/op_host/CMakeLists.txt, mc2/moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_v2_tiling.cpp, mc2/moe_distribute_dispatch_v2/op_host/CMakeLists.txt, mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp
- 缺陷描述：PR !1711引入多处错误：(1) CMake条件NOT取反写反；(2) 使用socVersion字符串比较判断芯片架构，应用GetNpuArch()枚举比较；(3) layered模式batch size上限应为512而非256；(4) dispatch/combine的A2 tiling缺少SetScheduleMode(batch_mode=1)，涉及SyncAll操作可能死锁。
- 修复模式：修正CMake条件方向；改用arch枚举比较；增加layered模式max batch size常量；添加batch mode调度设置。
- 可审查性：高
- 审查规则建议：CMake条件使用NOT时特别审查方向；芯片平台判断用架构枚举而非版本字符串；涉及SyncAll的算子必须检查batch mode设置。

---

### 974e5560 修复确定性场景FAG算子在SparseMode=1情况下的越界访问
- 根因类别：掩码尺寸硬编码导致缓冲区越界访问
- 涉及文件：attention/flash_attention_score_grad/op_kernel/arch32/basic_modules/vec_op_det.h
- 缺陷描述：SubGrapA函数中处理attention mask时，掩码张量尺寸硬编码为64x128，但sparseMode==1下实际尺寸由s1Extend和s2Extend决定，可能不同，导致CopyInAttenMaskBool和CalcAttenMaskBool越界读取。
- 修复模式：根据sparseMode动态计算掩码尺寸，sparseMode==1时用s1Extend和s2Extend，否则回退固定值；最后一轴做32字节对齐。
- 可审查性：中
- 审查规则建议：kernel中访问tensor时尺寸参数禁止与运行时配置无关的硬编码值；新增分支路径时须检查所有依赖参数是否在该路径下仍正确。

---

### d650ef8f 修复GMM V4问题

非缺陷提交。只修改了markdown文档文件中API签名的格式(两参数拆为两行显示)，纯文档格式调整。

---

### 5023d7a8 moe_finalize_routing_v2 md fix

非缺陷提交。修改.md文档中冗余分号和示例.cpp中的注释文本修正，不影响实际算子逻辑。

---

## 批次4新增缺陷类别

| 类别 | 本批新增数 | hash |
|------|-----------|------|
| 除零错误(空tensor边界) | 1 | 1f3291bf |
| 校验逻辑顺序错误(空tensor误拦截) | 1 | 44e84d52 |
| 构建配置错误(CMake include路径) | 1 | 0c90fb3c |
| 输入校验缺失(dtype白名单) | 1 | 8ceb72b6 |
| 硬件流水线同步缺失 | 2 | 368c7325, 4efbd43d |
| 校验逻辑条件分支错误(量化模式互斥) | 1 | 44338c14 |
| 赋值遗漏+错误的提前返回 | 1 | eb8f0ac7 |
| 校验逻辑条件错误(多余条件放行非法配置) | 1 | 4573c2e |
| 校验逻辑不完整(缺少上界)+格式符类型不匹配 | 1 | 3881d5c |
| 硬编码常量导致缓冲区越界清理 | 1 | 0c3b9b38 |
| 前序PR引入多处逻辑错误(回归缺陷) | 1 | 1734ac98 |
| 掩码尺寸硬编码导致越界访问 | 1 | 974e5560 |

---

## 批次5: 提交 #81-#99 (2026-02-10 ~ 2026-02-12)

---

### 853ce34a 【FAG】empty tensor bug
- 根因类别：整数类型错误(int64_t vs uint64_t) + 宏参数数量错误
- 涉及文件：attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling.cpp, attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_kernel_base.h
- 缺陷描述：两处修复: (1) SetTilingKey调用的GET_TPL_TILING_KEY宏参数个数不对(18个→19个)，empty tensor场景tiling key生成错误导致走到错误kernel分支; (2) s2SparseLeft/s2SparseRight类型从int64_t改为uint64_t，这些变量参与位移运算(>>6 <<6)和Max(...,0)比较，有符号负值右移是算术右移保留符号，修复后用无符号类型配合Max确保稀疏窗口左边界为负时clip到0。
- 修复模式：数据类型修正(int64_t→uint64_t) + 宏参数数量修正
- 可审查性：中
- 审查规则建议：检测宏调用参数数量是否与宏定义一致；涉及位运算的变量检查有符号/无符号类型是否匹配语义预期。

---

### d3aa4960 修复matmulReduceScatter/matmulReduceScatterV2文档和参数校验更正
- 根因类别：空指针解引用(null pointer dereference)
- 涉及文件：mc2/matmul_reduce_scatter_v2/op_host/op_tiling/arch32/matmul_reduce_scatter_v2.cpp
- 缺陷描述：原校验逻辑将空指针检查和值检查合并为一个条件`commModePtr == nullptr || !(strcmp(commModePtr, "aiv") == 0)`，当commModePtr为nullptr时，OP_LOGE的%s格式化可能对null指针产生未定义行为(某些平台崩溃)。修复后拆分为两个独立的OP_TILING_CHECK: 先检查nullptr再检查值。
- 修复模式：空指针检查前置，避免对null pointer调用strcmp和printf %s
- 可审查性：高
- 审查规则建议：检测对可能为null的指针在同一条件中既检查null又调用strcmp/printf %s的模式——应将null检查拆分为独立前置校验。

---

### 54692072 gmm伪量化放开n=0场景拦截
- 根因类别：空tensor场景过度拦截(over-validation)
- 涉及文件：gmm/grouped_matmul/ 下9个文件(infershape/aclnn/tiling/kernel四层)
- 缺陷描述：GMM算子伪量化场景对N=0空tensor输入过度拦截(<=0改为<0)。多层面问题: (1) infershape/aclnn/tiling层用<=0拦截N和K，阻止了合法N=0空tensor; (2) 含batch轴时空tensor判断只看最后第二维未累乘batch轴; (3) bias等tensorlist为[(0)]时IsNonEmpty判断不正确导致nullptr解引用; (4) kernel层N=0未skip计算导致非法内存访问。
- 修复模式：放宽校验条件 + 修正空tensor判断逻辑 + 增加kernel层N=0跳过逻辑
- 可审查性：中
- 审查规则建议：维度校验中<=0拦截时检查是否存在合法的维度为0空tensor；含batch轴的tensor判空应累乘所有非K维度。

---

### 046aef30 FusedFloydAttention算子workspace大小修正
- 根因类别：workspace大小计算遗漏因子
- 涉及文件：attention/fused_floyd_attention/op_host/fused_floyd_attention_tiling_general.cpp
- 缺陷描述：GetWorkspaceSize计算stage2所需空间时，公式遗漏了n2BaseSize乘法因子。原公式`s1BaseSize * alignedD * calcTypeSize`应为`n2BaseSize * s1BaseSize * alignedD * calcTypeSize`。workspace估算偏小导致NZND/FP32场景内存越界。此外还删除了调试代码`workspaces[0] += 300*1024*1024;`(硬编码300MB)，说明之前用此临时掩盖了workspace不足。
- 修复模式：在workspace公式中补充遗漏的维度因子
- 可审查性：高
- 审查规则建议：workspace计算应覆盖所有参与维度；硬编码大数值内存偏移(如300MB)应作为code smell标记。

---

### 64dedf37 修复scatterPaKvCache算子aicore问题
- 根因类别：参数传递错误(分块处理量 vs 总大小)
- 涉及文件：attention/scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_rope_not_fully_load.h
- 缺陷描述：CastToOrigin函数第三个参数(处理数据量)原传kHeadSize/vHeadSize，应传handleNum。在not fully load场景下每次处理量由handleNum决定(分块后实际处理量)，当kHeadSize较大时handleNum < kHeadSize，传错参数导致处理超出buffer范围触发aicore错误。
- 修复模式：函数参数从固定总大小改为实际分块处理量
- 可审查性：高
- 审查规则建议：分块处理模式下数据操作函数的长度参数应使用当前分块大小而非总大小。

---

### aa9faa8c 修复mla全量化场景qs>1输入sparsemode非3的报错提示
- 根因类别：校验条件范围过大(over-validation)
- 涉及文件：attention/incre_flash_attention/op_host/incre_flash_attention_tiling_check.cpp
- 缺陷描述：CheckMaskShapeWithQSeq中条件`if (antiQuantFlag_ || quantFlag_)`过于宽泛。quantFlag_不只MLA全量化还包含其他量化场景，而"qs>1时sparseMode必须为3"约束只适用于MLA场景。使用quantFlag_导致非MLA量化场景在qs>1且sparseMode!=3时被错误拦截。修复后条件改为精确匹配MLA场景(dequantScaleQuery != nullptr && ropeFlag_)。
- 修复模式：缩小条件范围，使校验仅作用于目标场景
- 可审查性：高
- 审查规则建议：输入校验guard condition应精确匹配适用场景，避免宽泛标志位；新增校验应验证所有已有场景兼容性。

---

### 021f8352 add error message for invalid quantmode parameters
- 根因类别：缺失错误处理/静默失败
- 涉及文件：mc2/allto_all_matmul/op_api/aclnn_allto_all_quant_matmul.cpp, mc2/matmul_allto_all/op_api/aclnn_quant_matmul_allto_all.cpp
- 缺陷描述：CheckDtypesValid函数中quantMode不匹配时存在两个问题: (1) allto_all_quant_matmul: quantMode不匹配时isAllDtypesValid保持false直接返回但无任何错误日志，用户无法得知失败原因; (2) quant_matmul_allto_all: 错误日志打印所有tensor数据类型，对quantMode不匹配场景产生误导。修复后在else分支添加OP_LOGE输出具体quantMode值。
- 修复模式：添加缺失错误日志 + 修正误导性错误信息
- 可审查性：高
- 审查规则建议：参数校验函数返回false的路径必须伴随OP_LOGE调用，禁止静默失败。

---

### 14824774 BugFix: KvRmsNormRopeCache算子大shape AicError问题修改
- 根因类别：整数溢出(循环索引类型不足)
- 涉及文件：posembedding/kv_rms_norm_rope_cache/op_kernel/arch35/kv_rms_norm_rope_cache_regbase_recompute.h
- 缺陷描述：5个Rope处理函数中for循环索引ubIdx声明为uint16_t(最大65535)，而循环上界ubFactorDkLoopCountCeil在D维度极大时超过uint16_t范围，导致循环索引溢出回绕产生AIC硬件错误。
- 修复模式：循环索引类型从uint16_t扩展为int64_t
- 可审查性：高
- 审查规则建议：循环索引类型应覆盖循环上界最大可能取值；当上界来源于外部shape计算时索引应用int64_t而非窄类型。

---

### b59cf016 [Fix] fix small s2 tail basic block
- 根因类别：尾块边界条件处理错误
- 涉及文件：attention/sparse_flash_attention_grad/basic_modules/cube_modules/cube1-5.h, cube_op.h (6个文件)
- 缺陷描述：sparse_flash_attention_grad处理s2维度尾部basic block时MM参数计算有误: (1) cube1/cube2的Sparse路径singleN固定为selectedBlockSize*blockOffset，未考虑最后block的lastBlockSize可能更小; (2) cube3中totalSel计算和isLastBasicBlock判断只在isLastLoop时生效导致非末次迭代singleK不正确; (3) cube4/5的singleM和GM拷贝长度未考虑lastBlockSize。s2不能被block大小整除时MM维度参数超出有效数据范围，偶现MTE1硬件报错。
- 修复模式：将lastBlockSize/isLastBasicBlock传入各cube函数，循环内用min()动态约束MM参数和数据搬运长度
- 可审查性：中
- 审查规则建议：分块处理时必须检查尾块实际大小是否被正确传递和使用；MM参数应取min(块大小, 剩余有效长度)。

---

### 7cab8a6a 修复MoeGatingTopKSoftmaxV2算子finished参数为true时expertIdxOut取值不对问题
- 根因类别：操作时序错误(EnQue位置不当导致break跳过入队)
- 涉及文件：moe/moe_gating_top_k_softmax_v2/op_kernel/arch35/moe_gating_top_k_softmax_v2_k_renorm.h
- 缺陷描述：indicesOutLocal的EnQue原放在ComputeSoftmax内部。当finished=true时代码在ComputeTopK之后ComputeSoftmax之前break跳出循环，直接进入CopyOut。break跳过了EnQue调用，CopyOut读到未入队的脏/旧数据，expertIdxOut输出不正确。
- 修复模式：将EnQue从ComputeSoftmax内部移到主循环CopyOut前，确保所有控制流路径都能正确入队
- 可审查性：高
- 审查规则建议：存在提前退出(break/continue/return)时，检查队列EnQue/DeQue是否在所有路径上正确配对；资源管理操作放在不会被跳过的位置。

---

### 非缺陷提交(本批次9条)

| hash | message | 排除原因 |
|------|---------|---------|
| d3460fbf | 回退 ROPEV2 文档 | 纯文档删除 |
| 7084b812 | 回退 ROPE V2接口 | 功能回退(revert)，非缺陷修复 |
| 2d497549 | 修复GMM问题，提升易用性 | 纯文档修改(docs/目录) |
| a9a25159 | fix alltoallquantmatmul A5 example | 示例代码修复(examples/目录) |
| af854dcd | fix GMMSQ && GMMweightNZ && GMMV5算子问题 | 纯文档修改(.md文件) |
| 6b173d1a | 将blockDim修改为numBlocks | 纯变量重命名重构 |
| 4143b07a | 修复 GMM && MOEInitRouting算子问题 | 纯文档修改(图片+链接) |
| 5980e181 | [FIA IFA PFA]将blockDim修改为numBlocks | 纯变量重命名重构 |
| 11e82892 | 修复GMMV4报错信息未打印数据类型 | DFX改进(错误信息增强) |

---

## 批次5缺陷类别统计

| 缺陷类别 | 数量 | 涉及hash |
|----------|------|---------|
| 整数类型错误/宏参数错误 | 1 | 853ce34a |
| 空指针解引用 | 1 | d3aa4960 |
| 空tensor过度拦截(over-validation) | 1 | 54692072 |
| workspace大小计算遗漏因子 | 1 | 046aef30 |
| 分块处理参数传递错误 | 1 | 64dedf37 |
| 校验条件范围过大(over-validation) | 1 | aa9faa8c |
| 缺失错误处理/静默失败 | 1 | 021f8352 |
| 整数溢出(循环索引类型不足) | 1 | 14824774 |
| 尾块边界条件处理错误 | 1 | b59cf016 |
| 操作时序错误(EnQue位置不当) | 1 | 7cab8a6a |

---

## 批次1+2+3+4+5累计统计

总计99条提交:
- 实际缺陷修复: 52条 (52.5%)
- 非缺陷(排除): 47条 (47.5%)

批次5: 10条缺陷 + 9条非缺陷 (52.6%)

累计非缺陷hash: fe0bee0d, e8ffb3b9, 53c21a1d, ae4c91e3, 91ef89b4, 7fbc7bc3, 108b4dd3, af418dfc, 28671df1, 8a09dcd0, 14a289de, f73c0505, 967fed57, f38d4a49, 613ae40b, 1065427e, 79a2aced, 988a83a9, 28daf0ab, be6bcad7, 4d3bbf03, e46032f7, 05e3ba28, e48d4172, 68af3c15, 57a30f3a, 1a62a495, 9081122a, 0162cad4, 5c48d0be, 8f3a5747, 349083a7, bd65dfcc, 9bb73d4c, 46bdb193, a6aa6f17, d650ef8f, 5023d7a8, d3460fbf, 7084b812, 2d497549, a9a25159, af854dcd, 6b173d1a, 4143b07a, 5980e181, 11e82892

---

## 批次6: 提交 #100-#118 (2026-02-07 ~ 2026-02-10)

---

### 80329f3a fix: scatter_pa_cache not compiling
- 根因类别：构建配置缺陷(新算子集成遗漏)
- 涉及文件：attention/scatter_pa_cache/op_host/CMakeLists.txt, attention/scatter_pa_cache/op_kernel/scatter_pa_cache.cpp(原名scatter_pa_cache_apt.cpp), scripts/ci/ascend950/ops_transformer_operator_list.yaml
- 缺陷描述：scatter_pa_cache算子存在三个问题导致无法编译：(1) CMakeLists.txt缺少`set(scatter_pa_cache_depends ...)`声明依赖关系；(2) kernel源文件命名为scatter_pa_cache_apt.cpp不符合约定(应为scatter_pa_cache.cpp)；(3) include路径仅写了一种相对路径未兼容另一种构建场景；(4) CI的operator_list.yaml未列入该算子。代码写好了但构建配置不完整，算子根本没有被编译进产物。
- 修复模式：重命名文件为标准名称；添加CMake依赖声明；用`#if __has_include()`做条件编译兼容两种路径；加入CI编译列表。
- 可审查性：中
- 审查规则建议：新增算子时应检查checklist——源文件命名是否符合约定、CMakeLists是否声明跨算子依赖、CI的operator_list是否包含新算子。

---

### 3c508d74 删除冗余函数，修复在非910b的CANN环境中ut用例执行失败的问题

非缺陷提交(UT测试修复)。修改tests/ut/目录下测试代码，将断言从`EXPECT_EQ(ACLNN_SUCCESS, ...)`改为`EXPECT_NE(ACLNN_ERR_PARAM_NULLPTR, ...)`，是对测试断言逻辑的修正。

---

### fd0ed91f Change Doc, single machine dc

非缺陷提交(纯文档更新)。所有8个文件均为README.md和API文档，扩展epWorldSize支持范围、删除约束说明。

---

### 3ea26544 fix(kv_rms_norm_rope_cache)：修改910B平台上，tilingkey判断分支

非缺陷提交(性能优化)。收紧DeepSeek优化分支的准入条件，增加batchSize > coreNum * BATCHES_FOR_EACH_CORE条件，避免非目标场景被错误分发到该分支导致性能损耗。

---

### 15ff1476 gmm_add aclnn description fix

非缺陷提交(纯文档注释修正)。修改.h文件中Doxygen注释的函数名引用笔误，不影响编译和运行时行为。

---

### 4e4f9168 【fix】moe_token_unpermute_with_routing_map_grad约束公式添加括号
- 根因类别：文档约束公式运算优先级错误(括号遗漏)
- 涉及文件：moe/moe_token_unpermute_with_routing_map_grad/docs/aclnnMoeTokenUnpermuteWithRoutingMapGrad.md
- 缺陷描述：约束公式缺少关键括号，运算符优先级导致语义错误。原公式`196608 - (probTypeLen + 1) * numExpertAlign-(tokenTypeLen + 8) * 256 / (6 * tokenTypeLen + 12) >= 1`按优先级`256/(6*tokenTypeLen+12)`会先计算。正确意图是分子为`196608 - (probTypeLen+1)*numExpertAlign - (tokenTypeLen+8)*256`，整体除以`6*tokenTypeLen+12`后`>=1`。用户按错误公式校验参数会得到错误结果。
- 修复模式：添加外层括号修正运算优先级。
- 可审查性：低
- 审查规则建议：API文档中约束公式应与代码实现中的校验逻辑交叉验证；review checklist中加入"验证公式括号是否正确反映运算优先级"。

---

### 613510ad 修复MoeInitRoutingV2 geir问题
- 根因类别：算子注册基础设施缺失(geir通路不可用)
- 涉及文件：moe/moe_init_routing_v2/op_graph/CMakeLists.txt(新增), moe/moe_init_routing_v2/op_graph/moe_init_routing_v2_proto.h(新增), moe/moe_init_routing_v2/op_host/moe_init_routing_v2_def.cpp(修改)
- 缺陷描述：MoeInitRoutingV2算子缺少op_graph目录下的proto头文件和CMakeLists.txt，导致geir通路(图引擎IR离线推理路径)无法运行。同时def.cpp缺少DynamicCompileStaticFlag、DynamicRankSupportFlag等配置，导致ascend950平台动态shape编译失败。
- 修复模式：补充缺失的proto.h注册文件和CMake配置；在def.cpp中添加动态编译配置项。
- 可审查性：低
- 审查规则建议：新增算子时检查op_graph目录是否包含proto.h和CMakeLists.txt；检查regbaseConfig是否配置动态shape相关flag。

---

### 08fbc029 [fix]自适应install_deps.sh的sudo, 避免找不到sudo报错
- 根因类别：环境兼容性缺陷(脚本硬编码sudo)
- 涉及文件：install_deps.sh
- 缺陷描述：install_deps.sh中所有包管理命令前硬编码了`sudo`，在没有安装sudo的环境(如精简Docker镜像)或已是root用户的环境下，脚本因"sudo: command not found"中断执行，依赖安装全部失败。
- 修复模式：脚本开头通过`command -v sudo`检测sudo是否存在，结合`$EUID`判断是否root，设置`try_sudo`变量替换硬编码的sudo。注意：修复不完整，有一处`sudo tee`未替换。
- 可审查性：高
- 审查规则建议：Shell脚本中不应硬编码sudo；应检测是否需要提权。

---

### 03f73405 sfag/slig infershape及校验修改
- 根因类别：输入校验缺失 + infershape输出dtype设置错误(多重缺陷)
- 涉及文件：attention/sparse_flash_attention_grad/op_host/arch35/sparse_flash_attention_grad_tiling_bs1_regbase.cpp, attention/sparse_flash_attention_grad/op_host/sparse_flash_attention_grad_infershape.cpp, attention/sparse_lightning_indexer_grad_kl_loss/op_host/arch35/sparse_lightning_indexer_grad_kl_loss_tiling_general_regbase.cpp, attention/sparse_lightning_indexer_grad_kl_loss/op_host/sparse_lightning_indexer_grad_kl_loss_infershape.cpp
- 缺陷描述：多重缺陷：(1) SFAG tiling中sparse_mode原本允许0和3但实际只支持3；head_dim校验条件不正确；缺少inputLayout校验、qRope/kRope维度校验、batchsize一致性校验。(2) SLIG infershape中loss输出shape未设置(缺SetDimNum/SetDim)；输出dtype只设了index 0，其他输出未设置类型。注意：修复中新引入bug——`dimDq != D_SIZE && dimDq != D_SIZE`两个条件相同，其中一个应为dimDk。
- 修复模式：添加参数范围校验；修正infershape输出shape计算和dtype映射。
- 可审查性：中
- 审查规则建议：infershape函数中所有output都必须有SetOutputDataType调用；注意检测相同子表达式的逻辑运算(copy-paste错误)。

---

### ac0860f7 Fix A3 fullmesh bug
- 根因类别：Pipeline同步缺失 + API参数类型错误
- 涉及文件：mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h
- 缺陷描述：A3 16P fullmesh_v2通路下动态量化场景expandx精度错误。(1) PerToken动态量化函数中Cast操作使用PIPE_V，紧接着的Copy依赖该数据但中间没有PipeBarrier<PIPE_V>()同步屏障，Copy可能读到脏数据。(2) `DataCopyExtParams`被误用于`DataCopyPad`参数(应为`DataCopyParams`，字段类型从uint32_t改为uint16_t)；`DataCopyPadExtParams<T>`改为`DataCopyPadParams`。
- 修复模式：插入PipeBarrier<PIPE_V>()保证时序；修正API参数结构体类型匹配。
- 可审查性：低
- 审查规则建议：跨pipe数据依赖(PIPE_V产生→Copy消费)前必须有PipeBarrier；DataCopy系列API参数结构体类型应与API签名严格匹配。

---

### 1fa23e28 modify information for layout and blocksize constrain

非缺陷提交(纯文档修改)。只改了一个.md资料文件，更新blockSize和layout约束描述。

---

### 73c4fdaa FIA的learnablesink的host越界报错修复
- 根因类别：API误用(对可选输入使用必选输入访问接口)
- 涉及文件：attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp
- 缺陷描述：CheckFAILearnableSink函数中，learnableSink是可选输入(optional input)，但代码使用GetInputDesc和GetInputShape来获取其描述和形状。当该可选输入未提供时，通过必选接口访问导致越界访问，触发host侧报错。
- 修复模式：将GetInputDesc替换为GetOptionalInputDesc，将GetInputShape替换为GetOptionalInputShape，共2处。
- 可审查性：高
- 审查规则建议：对OPTIONAL参数索引的访问必须使用GetOptionalInputDesc/GetOptionalInputShape系列API。可建立静态规则扫描。

---

### 096526e6 修复mc2算子ut的opapi因打桩变更引起的问题

非缺陷提交(UT测试适配)。所有50个文件改动均在tests/ut/op_api/目录，适配打桩(mock)接口变更。

---

### bd4b2d3e 修复BNSD_NBSD尾块搬出异常
- 根因类别：计算逻辑错误(尾块DataCopy的blockLen和偏移量计算错误)
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_block_vec_base.h, attention/common/op_kernel/arch35/infer_flash_attention_kvcache.h, attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：BNSD/NBSD layout转置输出场景中MlaTranspose2DataCopy函数存在多个bug：(1) subBlockIdx==1时curGIdx/curS1Idx计算公式有变量依赖错误；(2) 尾块blockLen使用了tailBlock*s1Size而非正确的tailBlock*dSizeV(维度用错)；(3) 尾块attentionOutOffset偏移计算公式错误。infer_flash_attention_kvcache.h中也有对应的偏移计算错误。
- 修复模式：删除错误的subBlockIdx条件分支；修正blockLen维度参数(s1Size→dSizeV)；修正偏移量累加公式；统一使用sOuterOffset代替cubeSOuterOffset。
- 可审查性：中
- 审查规则建议：DataCopy的blockLen涉及维度乘积时检查各维度变量是否与实际数据排布一致；转置layout变换的offset计算需逐维度验证步长。

---

### 71798b1b fix code spelling error

非缺陷提交(代码清理/变量重命名)。变量名拼写纠正(tmpRst→tmpRes, reduceGlobaLoop→reduceGlobalLoop等)、魔数替换为命名常量(3e+99→FLOAT_INF)、移除未使用头文件。所有改动不影响运行时行为。

---

### 6131adfc fix the inaccurate error description for alltoallmatmul&matmulalltoall

非缺陷提交(纯错误信息文案修正)。修改OP_LOGE中的字符串文案，MAX_GROUP_NAME_LEN从128改为127属于报错描述与实际行为对齐(C字符串末尾\0)。

---

### 5080903c 【fix】moe_token_unpermute_with_routing_map_grad补充参数描述

非缺陷提交(纯文档更新)。所有改动在README.md和docs/目录的.md文件，将参数名paddedMode统一替换为dropAndPad。

---

### 9aa05a4b fix gmmsqv2
- 根因类别：空指针解引用 + 入参校验缺失(场景未区分)
- 涉及文件：gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/aclnn_gmm_dsq_base.h, gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/aclnn_grouped_matmul_swiglu_quant_utils.h, gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/aclnn_grouped_matmul_swiglu_quant_v2.cpp
- 缺陷描述：grouped_matmul_swiglu_quant_v2算子在A4W4场景下，代码无条件对weightAssistMatrix进行解引用(`(*gmmDsqParams_.weightAssistMatrix)[i]`)，但A4W4场景weightAssistMatrix应为nullptr(只有A8W4才需要辅助矩阵)。导致A4W4场景传入nullptr时直接崩溃。原代码未区分A4W4和A8W4两种场景。
- 修复模式：新增isA8W4/isA4W4布尔标识通过SetScenario()自动判定场景；添加前置校验(A4W4要求weightAssistMatrix为nullptr)；所有解引用位置增加空值保护。
- 可审查性：中
- 审查规则建议：对可选参数(nullable指针)解引用前必须检查空值；当同一接口支持多种数据类型组合时校验逻辑应明确区分各场景对可选参数的要求。

---

### 0d2c731b sfag增加sparse_block_size校验拦截
- 根因类别：入参校验缺失(参数值未校验导致静默错误)
- 涉及文件：attention/sparse_flash_attention_grad/op_host/arch35/sparse_flash_attention_grad_tiling_bs1_regbase.cpp
- 缺陷描述：SFAG算子tiling中从属性读取selected_block_size参数但缺少合法性校验。当前只支持sparse_block_size=1，若传入非1值后续计算产生错误结果(静默错误)而非报错拦截。
- 修复模式：增加`if (selected_block_size != 1)`校验，不满足时OP_LOGE并返回GRAPH_FAILED。仅4行代码。
- 可审查性：高
- 审查规则建议：从属性/配置读取参数值后，应检查是否有合法范围校验；算子仅支持特定参数值时必须在入口显式拦截不支持的值。

---

## 批次6排除的非缺陷提交

| hash前8位 | commit message | 排除原因 |
|-----------|---------------|---------|
| 3c508d74 | 删除冗余函数，修复UT | UT测试修复 |
| fd0ed91f | Change Doc, single machine dc | 纯文档更新 |
| 3ea26544 | fix(kv_rms_norm_rope_cache) tilingkey分支 | 性能优化 |
| 15ff1476 | gmm_add aclnn description fix | 纯文档注释修正 |
| 1fa23e28 | modify information for layout and blocksize | 纯文档修改 |
| 096526e6 | 修复mc2算子ut的opapi | UT测试适配 |
| 71798b1b | fix code spelling error | 代码清理/重命名 |
| 6131adfc | fix inaccurate error description | 错误信息文案修正 |
| 5080903c | 补充参数描述 | 纯文档更新 |

---

## 批次6缺陷类别统计

| 缺陷类别 | 数量 | 涉及hash |
|----------|------|---------|
| 构建配置缺陷(算子集成遗漏) | 1 | 80329f3a |
| 文档约束公式运算优先级错误 | 1 | 4e4f9168 |
| 算子注册基础设施缺失 | 1 | 613510ad |
| 环境兼容性缺陷(脚本硬编码sudo) | 1 | 08fbc029 |
| 输入校验缺失+infershape输出dtype错误(多重) | 1 | 03f73405 |
| Pipeline同步缺失+API参数类型错误 | 1 | ac0860f7 |
| API误用(可选输入用必选接口访问) | 1 | 73c4fdaa |
| 计算逻辑错误(尾块搬出偏移/blockLen维度错误) | 1 | bd4b2d3e |
| 空指针解引用+场景未区分 | 1 | 9aa05a4b |
| 入参校验缺失(参数值未校验) | 1 | 0d2c731b |

---

## 批次7: 提交 #119-#138 (2026-02-05 ~ 2026-02-07)

---

### c4385241 [FIA] fix oom bug for antiquant
- 根因类别：tiling数据结构选择错误导致OOM
- 涉及文件：fused_infer_attention_score_template_tiling_key.h, incre_flash_attention_entry_regbase.h, incre_flash_attention_template_tiling_key.h
- 缺陷描述：FIA/IFA的antiquant场景tiling key配置中，`ASCENDC_TPL_TILING_STRUCT_SEL`选择了`IncreFlashAttentionTilingDataV2`，而实际应使用`FlashAttentionScoreSimplifiedTilingData`。两个struct内存大小/布局不匹配，导致kernel读取tiling数据时产生OOM。涉及两个tiling_key头文件约90处替换。
- 修复模式：将所有`IncreFlashAttentionTilingDataV2`替换为`FlashAttentionScoreSimplifiedTilingData`
- 可审查性：低
- 审查规则建议：tiling key中的struct类型必须与host侧tiling计算写入的struct一致。可建立自动化检查确保tiling_key.h中的struct与host侧填充的struct匹配。

---

### 1ceed275 fix debug for single machine
- 根因类别：运算符优先级错误(整数除法)
- 涉及文件：mc2/moe_distribute_dispatch_v2/op_host/moe_distribute_dispatch_v2_infershape.cpp
- 缺陷描述：表达式`globalBsReal * 2 * k * (*epWorldSize) / RANK_NUM_PER_NODE`中，整数乘除左结合导致`/`最后执行。语义上`(*epWorldSize) / RANK_NUM_PER_NODE`应先计算，但实际`globalBsReal * 2 * k * (*epWorldSize)`先乘完再除，整数截断导致结果与预期不同。单机模式下infershape输出尺寸与PTA不一致，是PR#1229引入的回归bug。
- 修复模式：添加括号`((*epWorldSize) / RANK_NUM_PER_NODE)`强制先做除法
- 可审查性：高
- 审查规则建议：涉及整数除法的复合表达式，必须确认运算顺序是否符合语义。乘除混合时强制使用括号消除歧义。

---

### d452dde6 修复MLA非量化sparse0/4分核错误
- 根因类别：GQA维度缩放(gSize)遗漏
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：MLA非量化路径的`GetPreNextTokensLeftUp`函数中，preTokens/nextTokens/actualSeqLengthKV在计算时未乘以gSize。MLA模式下KV维度经过group-size压缩，Q侧与KV侧序列长度存在gSize倍数关系，直接比较维度不匹配导致分核数计算错误。sparse mode 0和4下均受影响。
- 修复模式：增加`enableIFAMLA`分支，MLA模式下对preTokens、nextTokens、actualSeqLengthKV乘以gSize
- 可审查性：中
- 审查规则建议：MLA/GQA场景中Q-KV维度交叉计算处都须检查gSize缩放。

---

### d78096ed fa_fag_add_input_error_log
非缺陷提交(纯日志增强)。将FA/FAG的aclnn API层`CHECK_RET`替换为`OP_CHECK`+`OP_LOGE`，增加参数名称日志，不改变运行时逻辑。

---

### 156446e3 MlaProlog fix tilingkey-not-found issue in per-tile scenario
- 根因类别：编译宏过度约束导致tiling key缺失
- 涉及文件：attention/mla_prolog/op_kernel/mla_prolog_template_tiling_key.h
- 缺陷描述：MlaProlog per-tile量化场景的tiling key编译宏中，对`ORIG_DTYPE_KR_CACHE`增加了`== DT_BF16`约束。per-tile场景下kr_cache dtype不一定是BF16，使用其他dtype时编译宏不满足，tiling key不被编译，运行时匹配失败。
- 修复模式：移除编译宏中对`ORIG_DTYPE_KR_CACHE`的dtype约束
- 可审查性：中
- 审查规则建议：tiling key的`#if`编译宏新增dtype约束时，必须枚举该输入所有合法dtype组合，避免遗漏。

---

### 81213d24 fix leftpadding
- 根因类别：GQA维度缩放(gSize)遗漏 + 初始化flag遗漏(双重缺陷)
- 涉及文件：attention/common/op_kernel/arch35/infer_flash_attention_kvcache.h, attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：(1) `InitQueryLeftPaddingSize`中，GQA合轴场景下`actualS1Size`包含group维度倍数，直接用`s1Size - actualS1Size`计算padding大小得到错误值。(2) `SetAttributeInfo`中，`enableLeftPadding`为true时未设置`needInit = 1`，导致非量化left padding场景下output未刷零、LSE未初始化为inf，残留脏数据。
- 修复模式：(1) GQA场景下用`actualS1Size / constInfo.gSize`替代。(2) `enableLeftPadding`时显式设置`needInit = 1`。
- 可审查性：中
- 审查规则建议：多模式(GQA/MHA)共用计算路径时，审查维度语义一致性。涉及buffer初始化flag的新feature分支须检查初始化标志是否正确设置。

---

### 8adc6e89 AllGatherAdd bugfix
- 根因类别：函数重复定义导致编译错误
- 涉及文件：examples/mc2/all_gather_add/op_host/op_tiling/all_gather_add_tiling.cpp
- 缺陷描述：`GetWorkspaceSize`函数单独定义为static函数，与其他编译单元同名函数重复定义，导致CI编译失败。
- 修复模式：删除独立函数定义，将逻辑内联到调用处
- 可审查性：高
- 审查规则建议：新增函数时搜索全局是否已有同名定义。

---

### 3cbec10d mlaprologblocksize泛化
非缺陷提交(功能增强)。移除BlockSize硬编码限制{16,128}，泛化为16~1024且16的倍数。

---

### 17362b03 同步qbmmia算子规范资料到transfomer仓
非缺陷提交(纯文档更新)。仅修改.md文档。

---

### 62be3454 修改GQA伪量化支持prefix中的prefixloopcount判断条件
- 根因类别：循环索引偏移遗漏
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h, flash_attention_score_antiquant_kernel.h
- 缺陷描述：GQA伪量化KV prefix场景下，判断`runInfo.s2LoopCount < constInfo.prefixLoopCount`未加起始偏移`runInfo.s2StartIdx / constInfo.s2BaseSize`。s2LoopCount是局部计数，不能直接与全局prefixLoopCount比较。遗漏偏移导致误判为prefix循环，读取错误的KV数据源。4处相同逻辑均受影响。
- 修复模式：将`runInfo.s2LoopCount`替换为`(runInfo.s2LoopCount + runInfo.s2StartIdx / constInfo.s2BaseSize)`
- 可审查性：中
- 审查规则建议：循环计数与全局阈值比较时，审查计数是局部还是全局语义，是否需要加起始偏移。同一模式多处出现时确认逻辑一致性。

---

### 8a852918 GQA非量化非BNSD格式actualS1Size计算bugfix
- 根因类别：条件分支遗漏(GQA模式组合未覆盖)
- 涉及文件：attention/common/op_kernel/arch35/infer_flash_attention_kvcache.h
- 缺陷描述：`GetSingleCoreParam`中if-else只处理了hasRope+Aligned576+非BNSD和默认两种情况，缺少GQA+s1Size==1+非BNSD的分支。此场景下正确公式应为`(actualS2Size + preTokensPerBatch) * gSize`，缺失导致actualS1Size被截断为错误值。
- 修复模式：插入`else if`分支处理GQA+s1==1+非BNSD组合
- 可审查性：低
- 审查规则建议：多模式组合(GQA/MHA x BNSD/非BNSD x 量化/非量化)的分支逻辑须系统性枚举所有组合，检查是否遗漏。

---

### 49c60620 优化aclnnFusedInferAttentionScoreV4约束说明格式
非缺陷提交(纯文档格式调整)。将约束说明改为折叠格式。

---

### 8f4dbd41 修复GMM激活函数精度问题，补充tiling处acttype枚举值
- 根因类别：短路逻辑跳过了必要的状态设置
- 涉及文件：gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp, .h
- 缺陷描述：`CheckActiveMode`中，`wScale`为2维+shape=(g,1)+nSize==1时，外层if短路跳过了所有scale维度校验。但这导致`bQuantMode`未被正确设置为`PERCHANNEL_MODE`。GMM激活函数N=1场景下量化模式错误导致精度出错。
- 修复模式：(1) 删除nSize==1时的短路分支，统一走校验逻辑；(2) nSize==1时显式设置`bQuantMode = PERCHANNEL_MODE`；(3) 补充`GMMActType`枚举定义
- 可审查性：中
- 审查规则建议："满足条件X则跳过校验"的逻辑，须追问跳过后是否有副作用(如状态设置)也被跳过。

---

### 15eccf03 测试程序环境变量警告信息指引性改善
非缺陷提交(日志/提示信息增强)。增加指引用户查看文档的LOG_PRINT。

---

### 11031c07 修复examples目录下示例执行失败问题
- 根因类别：构建配置遗漏(CMake列表未同步)
- 涉及文件：cmake/custom_build.cmake, docs/zh/develop/aicore_develop_guide.md
- 缺陷描述：CMake中`add_subdirectory`添加example后，缺少`list(APPEND OP_DIR_LIST ...)`，导致示例虽编译但执行阶段找不到构建产物。
- 修复模式：add_subdirectory后增加list(APPEND OP_DIR_LIST ...)
- 可审查性：高
- 审查规则建议：CMake中add_subdirectory与列表变量存在配对使用模式时，新增调用须同步维护相关列表。

---

### 7e8c30fa moe_gating_top_k_softmax精度问题处理
- 根因类别：向量mask merge模式隐式推导导致精度错误
- 涉及文件：moe/moe_gating_top_k_softmax/op_kernel/arch35/moe_gating_top_k_softmax_fullload_generalized_regbase.h
- 缺陷描述：调用`AscendC::MicroAPI::Max`时未显式指定`MaskMergeMode::MERGING`模板参数。带mask的Max归约中，非MERGING模式下mask为0的lane可能被置零或写入脏值，导致`reduceMidRreg`混入脏数据，softmax精度异常。
- 修复模式：显式添加`<float, AscendC::MicroAPI::MaskMergeMode::MERGING>`模板参数
- 可审查性：低
- 审查规则建议：所有带mask的向量intrinsic调用须显式指定MaskMergeMode。可建立lint规则：MicroAPI::Max/Min传入mask参数时必须显式指定模板参数。

---

### e747156a 【fix】moe_token_unpermute_with_routing_map_grad约束公式修复
非缺陷提交(纯文档修复)。修改README和API文档中约束公式括号位置和表格内容。

---

### 16bc59a1 [FIX] Modify the GQA per-block fullquant offset calculation
- 根因类别：per-block量化offset计算公式错误(CeilDiv不满足分配律)
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_block_vec_base.h, flash_attention_score_kernel_base.h, flash_attention_score_kernel_infer.h, util_regbase.h
- 缺陷描述：NTD layout下FP8 per-block fullquant场景，deScale offset用`runInfo.s1SizeAcc >> 7`(累积seqlen/128)计算。但GQA多batch场景下，per-block量化scale个数应为逐batch `CeilDiv(seqlen, blockSize)`后求和。例如batch0 seqlen=129,batch1 seqlen=127，正确结果=CeilDiv(129,128)+CeilDiv(127,128)=3，而原写法(129+127)>>7=2。
- 修复模式：新增`s1ScaleNumAcc`/`s2ScaleNumAcc`累积变量，逐batch做CeilDiv后累加，替代简单右移
- 可审查性：中
- 审查规则建议：`sum(a_i)/B`与`sum(CeilDiv(a_i,B))`不等价。向上取整/整数除法对求和不满足分配律。per-block分块场景中偏移量应逐元素CeilDiv后累加。

---

### c25b4a43 FIA splitFuse 场景算子scalar耗时增加问题修复
- 根因类别：性能回退(变量存储位置不当导致间接寻址开销)
- 涉及文件：attention/fused_infer_attention_score/op_kernel/flash_attention_regular.h
- 缺陷描述：splitFuse重构PR(!1175)中，16个`GlobalTensor`变量从局部变量改为类private成员变量。热循环中访问成员变量需通过`this`指针间接寻址，比局部变量多一层间接跳转。小batch短kvseqlen场景下scalar耗时显著劣化。
- 修复模式：将GlobalTensor恢复为局部变量，通过`GlobalTensorBundle`结构体引用传递
- 可审查性：中
- 审查规则建议：AscendC kernel热路径上的GlobalTensor等频繁访问对象优先用局部变量。"局部变量提升为成员变量"的重构须关注性能影响。

---

### 5b3eca4a aclnnGroupedMatmulWeightNz伪量化参数删除actType
非缺陷提交(文档修正)。从API文档中移除actType的"为空"描述。

---

### 批次7统计

批次7: 13条缺陷 + 7条非缺陷 (65.0%)

| 根因类别 | 频次 | commit hash |
|---------|------|-------------|
| GQA维度缩放(gSize)遗漏 | 3 | d452dde6, 81213d24, 8a852918 |
| 构建配置缺陷 | 2 | 8adc6e89, 11031c07 |
| tiling数据结构选择错误(OOM) | 1 | c4385241 |
| 运算符优先级错误(整数除法) | 1 | 1ceed275 |
| 编译宏过度约束(tiling key缺失) | 1 | 156446e3 |
| 初始化flag遗漏 | 1 | 81213d24(双重缺陷) |
| 循环索引偏移遗漏 | 1 | 62be3454 |
| 短路逻辑跳过必要状态设置 | 1 | 8f4dbd41 |
| 向量mask merge模式隐式推导 | 1 | 7e8c30fa |
| per-block量化offset公式错误(CeilDiv) | 1 | 16bc59a1 |
| 性能回退(变量存储位置) | 1 | c25b4a43 |

---

## 批次1-7累计统计

总计138条提交:
- 实际缺陷修复: 75条 (54.3%)
- 非缺陷(排除): 63条 (45.7%)

批次7: 13条缺陷 + 7条非缺陷 (65.0%)

## 批次8: 提交 #139-#158 (2026-02-04 ~ 2026-02-05)

---

### a5f8edd2637010fdbaeecd953d5cd04a62ea022e fix gmmsq sync
- 根因类别：硬件流水线同步缺陷
- 涉及文件：gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant_a8w4_msd_post.h
- 缺陷描述：A8W4后处理流水线中，多个函数共用同一个mmOutQueue，通过EnQue/DeQue在函数间传递tensor。问题：(1) customDataCopyIn中对同一buffer先做fp16 DataCopy再Cast为fp32，中间用EnQue/DeQue"切换"数据类型视图，但queue同步语义无法保证Cast写完再读，实际需要PIPE_V屏障；(2) MulPertokenScale中对同一HardEvent S_V连续做3组SetFlag/WaitFlag，其中2组完全冗余，消耗事件ID且引入不必要流水线停顿，极端情况下事件ID耗尽导致死锁。
- 修复模式：queue-based隐式同步改为PipeBarrier显式同步；tensor从函数局部提升到类成员变量；删除冗余SetFlag/WaitFlag
- 可审查性：低
- 审查规则建议：检测TQue在相邻函数间"EnQue后紧跟DeQue"且中间无其他生产者的模式——queue仅被用作同步屏障而非真正生产-消费；检测同一HardEvent重复SetFlag/WaitFlag超过2次

---

### 236a5fdb7a7d76cfcfcd307d0c831a106ca541c5 gmm add empty tensor check
- 根因类别：输入校验缺失（空tensor零维度未检查）
- 涉及文件：gmm/common/cgmct/kernel/kernel_grouped_matmul.h, gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul.cpp, gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_no_quant_matmul_tiling.cpp/.h
- 缺陷描述：GMM算子三层均缺空tensor处理。kernel层：仅SPLIT_M检查M<=0、SPLIT_K检查K<=0，N维度完全未处理。host API层：CheckZeroShape只遍历x列表不查weight列表，weight为空时继续走matmul。tiling层：5个shape解析路径均未检查K=0，K=0进入matmul tiling计算导致除零或无效配置。
- 修复模式：三层防御——kernel统一"M/K/N任一<=0即跳过"；API新增CheckEmptyTensor入口校验；tiling新增kZero标志位在5条路径中追踪K=0
- 可审查性：中
- 审查规则建议：对矩阵运算算子检测GetDim()返回值未做零值校验就参与tiling/除法的路径；检测groupType分支中只对部分维度做零值检查而遗漏其他维度

---

### f8882f78bf19bf13726a2f11560c70d9e6d250b3 [FIA]fixbug:float8_e5m2拦截
- 根因类别：输入校验缺失（不支持数据类型静默通过）
- 涉及文件：attention/fused_infer_attention_score/op_host/fused_infer_attention_score_infershape.cpp
- 缺陷描述：InferDataType通过TORCH_DTYPE_ENUM_VALUE_TO_GE_DTYPE_MAP做类型映射，float8_e5m2(enum=23)不在map中，find miss后代码以默认类型(fp16)静默继续，不报错。但float8_e5m2在当前算子不支持，用默认类型替代导致输出精度错误或后续算子类型不匹配崩溃，用户无感知。
- 修复模式：在映射查找前增加不支持类型黑名单校验，命中则返回GRAPH_FAILED
- 可审查性：高
- 审查规则建议：检测map.find()在miss分支没有显式错误返回而fall-through使用默认值的路径；对用户可控枚举输入应有白名单+黑名单双重校验

---

### 99a876f975e25dd848d5c332cee5fa4e0be13eee 修复计算attenOut的singleCoreSize时可能出现减翻的问题
- 根因类别：无符号整数下溢（uint32_t减法溢出）
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_block_vec_infer.h
- 缺陷描述：InitOutputSingleCore用`totalOutputSize - aivIdx * singleCoreSize`计算tailSize(uint32_t)。当输入shape较小(如[1,3,297])时后面核分不到数据，aivIdx*singleCoreSize > totalOutputSize，uint32_t减法下溢为接近2^32的巨大值。随后取min(tailSize, singleCoreSize)=singleCoreSize，该核在错误GM偏移处写入数据，触发aiverr硬件访存异常。
- 修复模式：减法前判断差值>0，否则clamp to 0
- 可审查性：高
- 审查规则建议：检测uint32_t/uint64_t减法未预先判断"被减数>=减数"的模式；多核场景特别关注"total - coreIdx * perCoreSize"形式

---

### 662f162cf8bf14a755e180c5503661cc30d536ee fix a synchronization issue of dispatch v2 fullmesh
- 根因类别：硬件同步事件方向错误
- 涉及文件：mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h
- 缺陷描述：MoE dispatch v2 fullmesh中if/else两条路径处理expertIds。if分支DataCopyPad(MTE2)→SyncFunc<MTE2_V>()→Select(V)正确。else分支原代码用SyncFunc<V_MTE2>()，语义是"等V完成后启MTE2"，但实际需要的是"MTE2搬入完成后做V计算"，方向完全反。实际else分支复用已有UB数据，需要的是PipeBarrier<PIPE_V>()保证前序V计算完成。
- 修复模式：删除错误方向同步原语，替换为正确的PIPE_V屏障
- 可审查性：中
- 审查规则建议：检测SyncFunc模板参数方向是否与数据流一致；检测if/else分支同步原语类型不对称且缺少注释说明

---

### 65cafae00727f428756ac277a02ca4dc6e90a5d5 fix syncAll
- 根因类别：调度模式设置时序/路径遗漏（导致多流死锁）
- 涉及文件：attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp, attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp, attention/incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp, attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_arch38.cpp, attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：IFA/PFA/FIA kernel使用SyncAll全核同步，要求batch mode调度。原代码SetScheduleMode(BATCH_MODE_SCHEDULE)存在：(1) 时序错误——放在DoSubOpTiling调用之后，但框架在DoSubOpTiling返回后即可能读取调度配置；(2) 路径遗漏——FIA SplitFuse和PFA arch38中完全没有SetScheduleMode调用。导致多流调度下核因调度延迟未启动，先到达SyncAll的核永久死锁。
- 修复模式：将SetScheduleMode移到DoSubOpTiling入口处；补全缺失路径
- 可审查性：中
- 审查规则建议：检测kernel使用SyncAll但host tiling缺少SetScheduleMode(BATCH_MODE_SCHEDULE)的路径；检测SetScheduleMode在DoSubOpTiling/SetTilingData之后（时序过晚）

---

### 7a03291632b83e4e592779c2cdceb88236d3cd79 [FIA] fix GQA per-block omm bug
- 根因类别：数组越界 + tiling struct类型不匹配
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_block_vec_base.h, attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score_template_tiling_key.h, attention/prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_entry_regbase.h, attention/prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_template_tiling_key.h
- 缺陷描述：两个独立缺陷——(1) GQA per-block场景deScaleKvOffset为0时`deScaleKvOffset-1`产生下溢/越界访问；(2) FP8/HiFloat8 GQA非MLA场景使用PFAFullQuantTilingData而非FlashAttentionScoreSimplifiedTilingData，tiling数据结构不匹配引发OOM。
- 修复模式：(1) 增加边界值判断保护；(2) 更正模板tiling struct选择和tiling key分发条件
- 可审查性：中
- 审查规则建议：检查无符号整数减法是否有下溢保护；检查模板特化中tiling struct类型是否与实际场景一致

---

### 473010ebb375745d34258735b95978f616ccf5fd FIA constinfo struct optimize
- 非缺陷。结构体成员按类型大小重排(uint64_t→uint32_t→bool)减少编译器对齐填充的内存浪费，纯内存布局优化。

---

### 8c1f71ff535b6372d9d8ff197f0a85d70d99d9d8 修复moe_init_routing_v2_grad累加精度丢失问题
- 根因类别：数值精度缺陷（浮点累加顺序导致精度损失）
- 涉及文件：moe/moe_init_routing_v2_grad/op_kernel/arch35/moe_init_routing_v2_grad_base.h
- 缺陷描述：SequenceReduceSum中K维度顺序逐个累加(sum+=x[k])，K较大时累加器sum不断增大，后续小值浮点数加到大sum上发生精度截断（"大数吃小数"），导致moe_init_routing_v2_grad梯度精度不达标。
- 修复模式：顺序累加改为4路分块二分归约(binary tree reduction)——两两先加再合并，减少量级差异带来的精度损失
- 可审查性：中
- 审查规则建议：检查循环累加浮点数的场景是否使用精度友好归约方式（Kahan summation/tree reduction），特别在GPU/NPU向量化代码中

---

### 6978efa7716c1bcb7187420c91b007205cf2bfa8 整改L0接口内QuantGroupedMatmulInplaceAdd的Infershape入参顺序
- 根因类别：接口调用参数顺序错误
- 涉及文件：gmm/quant_grouped_matmul_inplace_add/op_host/op_api/quant_grouped_matmul_inplace_add.cpp
- 缺陷描述：INFER_SHAPE宏OP_INPUT参数顺序与算子注册input顺序不一致——scale1Optional(可选参数)被放在scale2前面，OOM框架按错误位置解析tensor shape，推断出错误的输出大小，OOM模式下触发越界检查报错。
- 修复模式：将scale1Optional移到OP_INPUT列表末尾，与算子注册顺序一致
- 可审查性：高
- 审查规则建议：自动比对INFER_SHAPE中OP_INPUT参数列表与算子proto定义的input顺序是否完全一致，可选参数应在末尾

---

### 17de35a6c2edd6f01c152a57c93679acb705a36c (tiling+gentask+infershape+register)replace socVersion with npuArch
- 非缺陷。大规模API迁移重构，mc2模块全域平台判断从socVersion替换为npuArch，涉及74个文件，提升平台抽象层级。

---

### fbf471b1a66c04e46f718697ec2c262ff9a53924 SFAG example问题修复
- 非缺陷（文档修复）。API文档示例代码中actSeqQLenshape/actSeqKvLenshape从int32_t改为int64_t，不影响生产代码。

---

### 6cce8b90633c999e6810531c758915f0302bbfc1 fix scatter_pa_kv_cache ub tiling
- 根因类别：buffer大小计算未对齐（RoundUp遗漏）
- 涉及文件：attention/scatter_pa_kv_cache/op_host/scatter_pa_kv_cache_tiling_arch35.cpp
- 缺陷描述：UB切分计算时numKHeadSize/numVHeadSize直接用原始headSize乘seqLen，但实际kernel运行时headSize需按dtypeByteSize做RoundUp对齐。reduceBuf/divideBuf/castBuf大小计算同理使用未对齐headSize。导致tiling阶段UB需求偏小，kernel执行时UB空间不够引发数据越界。
- 修复模式：在size计算路径中加入缺失的RoundUp对齐步骤
- 可审查性：高
- 审查规则建议：buffer大小计算涉及硬件对齐要求时，检查所有参与维度是否已正确对齐；搜索seqLen*headSize模式确认headSize是否需先对齐

---

### e61e5b81219ebc0bc532f4b9ca0ec20be6cffe8e fix tensorlist bug
- 根因类别：条件分支遗漏（多模式区分不当）
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：(1) tensorlist模式下emptyTensor判定检查keyInputShape/valueInputShape的StorageShape是否为0，但tensorlist首batch shape可能为0并非空tensor，误判导致跳过该batch计算；(2) 序列长度校验s<=0应为s<0，tensorlist首batch seq=0是合法场景被误拦截。
- 修复模式：tensorlist和非tensorlist场景分别判断emptyTensor；边界条件放宽允许seq=0
- 可审查性：高
- 审查规则建议：同一代码服务多种输入模式(tensorlist vs 普通tensor)时检查条件是否正确区分模式差异；输入校验边界值(<=0 vs <0)确认0是否合法

---

### 10009be89313c38b33a654b8e8b8b8317a68d993 fix license
- 非缺陷。OAT.xml许可证扫描配置维护，合并年份特定的license matcher为通用版本。

---

### ccd9bc9bcac4d6553327b792f005108c4cb54e9b fix rowinvalid
- 根因类别：边界条件保护缺失（负值未clamp）
- 涉及文件：attention/common/op_kernel/arch35/infer_flash_attention_kvcache.h
- 缺陷描述：FlashAttention推理KVCache路径中actualS1Size由原始值减去preToken/nextToken无效部分。当全部被裁剪时actualS1Size变负数，后续作为循环次数或内存偏移导致未定义行为/越界/kernel崩溃。
- 修复模式：减法后增加if(actualS1Size<0) actualS1Size=0下界保护
- 可审查性：高
- 审查规则建议：涉及减法得到的size/count/length变量检查是否有负值可能并做下界保护；关键模式：size=a-b后未检查size<0

---

### d76a776a5203587cbf393f1d7566385b1d972135 fix pipe V_S
- 根因类别：流水线同步缺失（Vector-Scalar pipeline间缺屏障）
- 涉及文件：mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h
- 缺陷描述：DataCopyPad(Vector pipeline)将数据写入workspace后紧接调用UpdateTokenNumsOut(可能使用Scalar pipeline读取)，两个pipeline间缺少V_S同步屏障，Scalar侧可能读到Vector侧未写完的数据，产生数据竞争造成间歇性计算错误。
- 修复模式：在V写和S读之间插入SyncFunc<HardEvent::V_S>()硬件同步
- 可审查性：中
- 审查规则建议：DataCopyPad后紧跟非同pipeline数据消费时检查是否有对应SyncFunc/PipeBarrier

---

### 9691bcc3fb03ef0d564d44992976529884964c16 dispatch v2 fullmesh 2-dim mask fix bug
- 根因类别：buffer大小计算单位错误 + 分配不足
- 涉及文件：mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h, mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h
- 缺陷描述：三个相关缺陷——(1) bsAlign256/bsKAlign256计算中Ceil(x*sizeof(half), ALIGNED_LEN_256)*ALIGNED_LEN_256已是字节数，再除sizeof(half)使单位变为元素数，后续比较和分配用错大小；(2) fullmesh MaxSizeCal缺少bsKAlign256对maxSize_的比较更新；(3) subExpBuf_ InitBuffer用expertIdsSize_但2-dim mask场景需更大空间，分配不足导致buffer越界。
- 修复模式：去除多余除法修正单位；buffer大小统一取max；InitBuffer改用maxSize_
- 可审查性：高
- 审查规则建议：buffer大小计算链路中检查对齐后值是否被多余类型大小除法破坏单位一致性；多用途复用buffer时检查是否取所有用途最大值

---

### e3ba65148240223012acda57a2a9c368d158a1c0 【bugfix/update】ffag支持D为特殊值/ffa examples修改
- 根因类别：输出格式配置错误 + 数据类型不匹配（混合提交）
- 涉及文件：attention/fused_floyd_attention_grad/op_host/fused_floyd_attention_grad_tiling_s1s2_bn2gs1s2.cpp, attention/fused_floyd_attention/examples/test_aclnn_fused_floyd_attention.cpp, attention/fused_floyd_attention/docs/aclnnFusedFloydAttention.md
- 缺陷描述：(1) FusedFloydAttentionGrad中mm2输出被条件配置为NZ格式(D=72/80/88/96时)，但NZ格式在这些case下产生不正确结果(5个case失败)。修复为统一ND格式。(2) example中attentionOut tensor创建类型ACL_FLOAT应为ACL_FLOAT16，类型不匹配导致内存大小错误和计算异常。(3) 文档重写属非缺陷部分。
- 修复模式：条件分支简化为固定安全值(统一ND格式)；数据类型修正
- 可审查性：中
- 审查规则建议：输出格式(NZ/ND)选择的条件分支中确认所有D维度值经过正确性验证；tensor创建时检查aclDataType是否与算子输出类型一致

---

### 69698404f178faa18bcbf848f6b924325737ab0e [FAG] fix old deter workspace size bug
- 根因类别：workspace大小计算错误（类型宽度+数量因子错误）
- 涉及文件：attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp
- 缺陷描述：确定性计算路径workspace存3组int64偏移量(query/key/value GmOffset)。旧代码用maxValidBBLen*aicNum*FP32_BYTES*NUM_TWO计算大小，两处错误：(1) 偏移量是int64(8字节)但按FP32_BYTES(4字节)算，少一半；(2) 乘NUM_TWO(2)但实际3组偏移量，少一组空间。workspace申请不足导致AIC运行时内存越界引发硬件异常。
- 修复模式：修正数据类型宽度(4→8)、数量因子(2→3)和对齐因子
- 可审查性：高
- 审查规则建议：workspace大小计算中检查每个乘法因子语义是否对应实际数据类型和数据组数；关键模式：sizeof类型与实际写入数据类型不匹配

---

### 批次8统计

总计20条提交(#139-#158):
- 实际缺陷修复: 16条 (80.0%)
- 非缺陷(排除): 4条 (20.0%)

非缺陷: 473010eb(结构体优化), 17de35a6(API重构), fbf471b1(文档修复), 10009be8(license配置)

批次8新增缺陷类别分布:
- 硬件流水线同步缺陷(3): a5f8edd2, 662f162c, d76a776a
- workspace/buffer大小计算错误(3): 6cce8b90, 9691bcc3, 69698404
- 输入校验缺失(2): 236a5fdb, f8882f78
- 无符号整数下溢/边界保护缺失(2): 99a876f9, ccd9bc9b
- 调度模式设置时序/路径遗漏(1): 65cafae0
- GQA相关(数组越界+tiling struct不匹配)(1): 7a032916
- 数值精度缺陷(浮点累加)(1): 8c1f71ff
- 接口参数顺序错误(1): 6978efa7
- 条件分支遗漏/模式区分不当(1): e61e5b81
- 输出格式配置错误(1): e3ba6514

---

### 累计统计 (批次1-8)

总计158条提交:
- 实际缺陷修复: 107条 (60.5%)
- 非缺陷(排除): 70条 (39.5%)

批次8: 16条缺陷 + 4条非缺陷 (80.0%)
批次9: 16条缺陷 + 3条非缺陷 (84.2%)

累计非缺陷hash: fe0bee0d, e8ffb3b9, 53c21a1d, ae4c91e3, 91ef89b4, 7fbc7bc3, 108b4dd3, af418dfc, 28671df1, 8a09dcd0, 14a289de, f73c0505, 967fed57, f38d4a49, 613ae40b, 1065427e, 79a2aced, 988a83a9, 28daf0ab, be6bcad7, 4d3bbf03, e46032f7, 05e3ba28, e48d4172, 68af3c15, 57a30f3a, 1a62a495, 9081122a, 0162cad4, 5c48d0be, 8f3a5747, 349083a7, bd65dfcc, 9bb73d4c, 46bdb193, a6aa6f17, d650ef8f, 5023d7a8, d3460fbf, 7084b812, 2d497549, a9a25159, af854dcd, 6b173d1a, 4143b07a, 5980e181, 11e82892, 3c508d74, fd0ed91f, 3ea26544, 15ff1476, 1fa23e28, 096526e6, 71798b1b, 6131adfc, 5080903c, d78096ed, 3cbec10d, 17362b03, 49c60620, 15eccf03, e747156a, 5b3eca4a, 473010eb, 17de35a6, fbf471b1, 10009be8, 650bed1a, 12821635, b115d199

---

## 批次9: 提交 #159-#177 (2026-02-02 ~ 2026-02-04)

---

### e5dc9ee0 fix bug : empty tensor——init post quant output size
- 根因类别：计算逻辑错误(整数除法顺序)
- 涉及文件：attention/common/op_host/arch32/fia_tiling_empty_tensor.cpp
- 缺陷描述：FIA空tensor场景下，计算singleCoreSize时先用totalOutputSize除以2*usedCoreNum再除以2(量化缩半)，整数除法截断误差在先除后除的链路上被放大。正确做法是先缩半再分核。
- 修复模式：将isOutQuantEnable判断提前到singleCoreSize计算之前，先对totalOutputSize做/2UL处理，再执行分核的向上取整除法。
- 可审查性：中
- 审查规则建议：多步整数除法运算时，检查除法顺序是否影响精度/截断；先缩减再分配 vs 先分配再缩减的差异。

---

### 9b12d59b 【update/bugfix】ffag输入校验判断修复
- 根因类别：校验宏语义混淆(条件取反错误)
- 涉及文件：attention/fused_floyd_attention_grad/op_host/fused_floyd_attention_grad_tiling_common.cpp
- 缺陷描述：CheckSupportShape函数中4处OP_CHECK_IF调用传入CheckSameShape(...)无取反。OP_CHECK_IF语义是"条件为true时报错"，原代码导致shape相同时报错、不同时通过——逻辑完全反了。
- 修复模式：在4处条件参数前加!取反，即OP_CHECK_IF(!CheckSameShape(...),...)。
- 可审查性：高
- 审查规则建议：使用OP_CHECK_IF等断言宏时必须核实条件语义——条件为true时是"通过"还是"失败"？逐个确认每个调用点的条件方向。

---

### e233e106 修复PFA tiling侧accumOutSize计算公式不配套的问题
- 根因类别：workspace大小计算不一致(tiling侧对齐遗漏)
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：GetPFAWorkSpaceSize中accumOutSize计算使用vHeadSize，但kernel侧实际用AlignUp(vHeadSize, BYTE_BLOCK)(32字节对齐)。tiling侧算出的workspace小于kernel侧实际使用量，导致潜在内存越界。
- 修复模式：将两处vHeadSize替换为AlignUp(vHeadSize, BYTE_BLOCK)即headDimAlign。
- 可审查性：高
- 审查规则建议：workspace/buffer大小计算必须与kernel侧实际使用保持一致，特别是对齐要求。建议将对齐后尺寸抽为共用常量或公共函数。

---

### dfd1838a fix isPerformanceFlag
- 根因类别：操作时序错误(循环内vs循环外) + 校验逻辑错误
- 涉及文件：mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2.h, mc2/moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_v2_tiling.cpp
- 缺陷描述：(1) kernel侧isPerformanceFlag_的性能信息写回(SetAtomicMax+DataCopyPad)放在for循环内，每次迭代都执行sync+atomic+copy，浪费性能且可能产生中间不正确数据。(2) host tiling侧对performanceInfoStorageShape做"不为空则报错"的检查逻辑有误，在合法场景下会误拦截。
- 修复模式：将isPerformanceFlag_代码块从for循环体内移到循环之后；删除tiling侧错误的校验。
- 可审查性：高
- 审查规则建议：写回/输出操作是否应在循环内还是循环外，特别是涉及atomic操作和sync的代码；新增校验上线前确认所有合法path不会误拦截。

---

### 650bed1a gmm部分日志打印语法错误

非缺陷提交(日志文案修正)。将"inputs is not empty"修正为"inputs are not empty"，纯文案改动。

---

### 05f535a7 [FIA]回退代码
- 根因类别：前序PR引入多处回归错误(Revert)
- 涉及文件：attention/fused_infer_attention_score/op_host/fused_infer_attention_score_infershape.cpp, attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：被回退的PR1283引入三个问题：(1) post quant输出dtype推导删掉了默认fallback DT_INT8，map.at()在key不存在时抛异常崩溃；(2) PFA的CheckPrefix错误放宽了tensorlist场景拦截，s==1时也不支持；(3) 新增MLA不支持pseShift的约束是错误的，MLA实际可用pseShift。
- 修复模式：完整回退PR1283所有变更。
- 可审查性：中
- 审查规则建议：修改infershape/dtype推导时确认所有合法dtype都能处理；std::map::at()必须确保key存在否则用find()；放宽或收紧校验前与算子实际能力对齐。

---

### 3ca57b44 修复sink PSE场景下的精度问题
- 根因类别：workspace地址偏移计算遗漏(多段workspace地址重叠)
- 涉及文件：attention/flash_attention_score_grad/op_host/arch32/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.cpp, attention/flash_attention_score_grad/op_kernel/arch32/flash_attention_score_grad_s1s2_bn2gs1s2_sab.h
- 缺陷描述：FlashAttentionScoreGrad在sink场景下workspace需额外分配dsinksum空间。Tiling侧正确推进了workspaceOffsets，但kernel侧计算pseAlibiAddr时完全未考虑dsinksum占用的空间，导致sink+PSE场景下地址重叠、数据互相覆盖。
- 修复模式：tiling数据结构新增sinkDataSize字段，tiling侧计算填充，kernel侧pseAlibiAddr加上该偏移量。
- 可审查性：高
- 审查规则建议：workspace多段使用时，每段起始地址必须考虑所有前序段的累积偏移。新增条件性workspace段时检查所有下游地址计算是否都累加了该段大小。

---

### 12821635 Modify the aclnn issues

非缺陷提交(纯文档修改)。仅修改aclnnQuantGroupedMatmulDequant.md的参数说明表格。

---

### c4fa9e4f 修复GQA非量化支持AttentionSink特性的拦截误改
- 根因类别：输入校验缺失(特性互斥约束遗漏)
- 涉及文件：attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_v2.cpp, attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：开发AttentionSink特性时遗漏不支持场景的拦截：(1) IFA伪量化模板不支持learnable sink但未拦截；(2) PFA路径缺少CheckLearnSink函数，未检查与量化模式/pse/alibi/leftpadding/prefix的互斥关系。
- 修复模式：IFA中增加OP_CHECK_IF校验；PFA中新增CheckLearnSink系统性检查互斥关系。
- 可审查性：高
- 审查规则建议：新增算子特性时，检查所有模板路径(IFA/PFA/GQA/MLA等)是否都添加了互斥特性拦截校验。

---

### d865fe6d MLA全量化FD负载均衡出口条件修复
- 根因类别：数值溢出/边界条件判断错误(硬编码精确值匹配)
- 涉及文件：attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp
- 缺陷描述：FD负载均衡出口条件使用精确值匹配(s2!=55002)，无法覆盖整网场景s2的合理变化范围。s2过长时计算过程出现Inf(浮点溢出)。
- 修复模式：改为范围判断(SEQ_LEN_MIN_V2=37000, SEQ_LEN_MAX_V2=65536)，通过上界防止数值溢出。
- 可审查性：中
- 审查规则建议：精确值比较(!=某常量)作为动态参数校验时应审查是否改为范围判断；涉及浮点运算的tiling参数检查极端输入下的Inf/NaN风险。

---

### 6ed5ed33 fix call
- 根因类别：短路求值顺序错误
- 涉及文件：gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul.cpp
- 缺陷描述：GMM INT8量化校验中，CHECK_COND宏内先调用CheckIsEnabledActive(gmmParams)再OR isNoActivation。当isNoActivation为true时仍执行了不必要的CheckIsEnabledActive调用，该函数在前置条件不满足时可能行为异常。
- 修复模式：改为isNoActivation || CheckIsEnabledActive(gmmParams)，利用短路求值避免不必要的校验。
- 可审查性：高
- 审查规则建议：CHECK_COND宏中多条件OR组合时，轻量级/无副作用条件放前面，避免前置条件不满足时执行可能有问题的校验函数。

---

### 8c74b0bb fix path 6
- 根因类别：条件分支遗漏(模式豁免缺失)
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：(1) CheckPerblockQuantParams中isMaxWorkspace为true时缺少提前返回，导致max workspace场景下误报shape不匹配；(2) CheckNTDLayoutCrossover中GQA head dim限制未豁免enablePerblockQuant && layout=="NTD_TND"组合，导致合法场景被错误拦截；(3) 错误信息硬编码"NTD"而非动态获取实际layout。
- 修复模式：增加isMaxWorkspace提前返回；增加perblock量化NTD_TND豁免条件；错误信息改用动态layoutStr。
- 可审查性：高
- 审查规则建议：多种workspace/layout模式的校验函数中，新增模式后逐一检查现有校验路径是否需要豁免或调整；错误信息参数值应动态获取。

---

### b115d199 mc2算子ut文件整改

非缺陷提交(代码风格整改)。命名风格、缩进、花括号统一，纯格式变更。

---

### a75a92e2 perblock量化场景串行修改
- 根因类别：条件分支遗漏(fallback路径缺失)
- 涉及文件：mc2/matmul_reduce_scatter_v2/op_host/op_tiling/arch35/quant_bmm_reduce_scatter_tiling.cpp, mc2/matmul_reduce_scatter_v2/op_kernel/arch35/quant_bmm_reduce_scatter_fp8_hif8.h
- 缺陷描述：perblock量化场景下orgMValue不满足PERBLOCK_SIZE*rankDim整数倍时，原代码仍走公式化切分路径，SetBatch()中无条件将batch4_设为rankDim。不满足对齐条件时tiling切分错误。
- 修复模式：新增isSerial_标志位，不满足对齐时回退串行模式；SetBatch()增加!isSerial_条件守卫。
- 可审查性：中
- 审查规则建议：有对齐/整除前提的计算路径是否有不满足条件的fallback处理；tiling参数设置是否有缺少条件守卫的情况。

---

### 8ade8c3e Revert "dispatch优化syncall"
- 根因类别：硬件流水线同步缺陷(优化引入正确性问题，Revert)
- 涉及文件：mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h
- 缺陷描述：被revert的提交对MoE dispatch的syncall进行"优化"：(1) UpdateTokenNumsOut()从lastCore_改到FIRST_CORE执行并用ReduceSum重新聚合，改变执行核和数据读取时序；(2) 同步原语从PipeBarrier改为SyncFunc。导致多核场景下数据竞争或读取未就绪数据。
- 修复模式：完整revert，恢复到优化前实现。
- 可审查性：低
- 审查规则建议：多核/多流水线同步优化审查要点：修改执行核是否影响数据依赖；同步原语替换是否保证相同ordering语义；此类优化必须附多卡多专家场景回归测试。

---

### f91e8dee alltoallmatmul修复
- 根因类别：条件守卫缺失 + 多平台分支逻辑错误
- 涉及文件：mc2/allto_all_matmul/op_kernel/arch32/allto_all_matmul.h, mc2/allto_all_matmul/op_api/aclnn_allto_all_quant_matmul.cpp, mc2/allto_all_matmul/op_host/op_tiling/arch32/allto_all_matmul_tiling_910b.cpp
- 缺陷描述：(1) kernel中alltoallOut数据拷贝分支缺少isAlltoallOut条件判断，不需要输出时仍拷贝，访问未分配buffer导致越界；(2) aclnn层CheckAllDtypesValid缺少x1ScaleOptional的dtype校验，对nullptr做dtype检查导致空指针；(3) A2平台yDtype和all2AllOutFlag被错误放在DAV_3510分支内，A2平台上始终为默认值。
- 修复模式：增加isAlltoallOut条件判断；增加空指针守卫；将公共赋值提到平台判断之前。
- 可审查性：中
- 审查规则建议：可选输出/输入在kernel和host都必须有null/flag守卫；多平台代码中公共逻辑不应被错误放在特定平台分支内。

---

### f69aa254 fix bugs : PSE feature, qs==1 pseshifts1 > qs, copy falut
- 根因类别：边界条件处理缺失(qs==1索引错误)
- 涉及文件：attention/common/op_kernel/arch32/fia_block_vec_nonquant.h, attention/common/op_kernel/memory_copy.h
- 缺陷描述：PSE搬运逻辑中qs==1但PSE的s1维度大于qs时，使用GetDimS1()获取s1Size(大于1)导致索引计算错位。同时stride应使用GetStrideG()而非GetStrideS1()。
- 修复模式：新增qsEqualOne参数，为true时强制s1Size=1并使用GetStrideG()。
- 可审查性：中
- 审查规则建议：tensor维度与实际处理长度不一致时(broadcast场景)，检查索引计算和stride选择是否基于实际处理长度而非tensor维度。

---

### 4b6e703c ffn2attn fix Sync bug
- 根因类别：硬件流水线同步缺失(MTE3 barrier缺失)
- 涉及文件：mc2/ffn_to_attention/op_kernel/ffn_to_attention.h
- 缺陷描述：FFNToAttention::Process()中：(1) 首次迭代(tokenCnt==0)缺少SyncFunc<HardEvent::S_MTE3>()，scalar可能在MTE3写操作完成前准备下一次搬运；(2) DataCopyPad搬运token数据后缺少PipeBarrier<PIPE_MTE3>()，状态位可能在token数据落地前被更新。
- 修复模式：首次迭代增加SyncFunc<HardEvent::S_MTE3>()；token数据DataCopyPad后增加PipeBarrier<PIPE_MTE3>()。
- 可审查性：低
- 审查规则建议：连续DataCopyPad(MTE3)调用之间，后者作为前者的"完成信号"时必须插PipeBarrier<PIPE_MTE3>；循环首次迭代检查前序同步完整性。

---

### 78282635 fixup bug uint64_t to int64_t for quant_all_reduce
- 根因类别：整数类型错误(unsigned vs signed mismatch)
- 涉及文件：mc2/quant_all_reduce/op_host/quant_all_reduce_infershape.cpp
- 缺陷描述：QuantAllReduceShapeInfo中b/s/bs/hiddenSize/rankNum声明为uint64_t，但来自API的int64_t赋值时负值(-1表示动态shape)会隐式转为极大正数，后续shape计算和校验失效。
- 修复模式：将字段类型从uint64_t改为int64_t，与API类型一致。
- 可审查性：高
- 审查规则建议：infershape/tiling代码中存储shape的变量应使用int64_t，框架API普遍用有符号类型(-1表示动态维度)；审查所有uint64_t用于shape字段的场景。

## 批次10: 提交 #178-#196 (2026-01-30 ~ 2026-01-31)

---

### 93e71beb [FIA]fixbug:打印信息不清晰
非缺陷提交(日志格式修正)。将OP_LOGE中std::string对象改为.c_str()传给%s格式符。虽然传std::string给%s是UB，但属于日志打印层面修正，不影响业务逻辑正确性。

---

### cea3f28f [FIA] fix GQA per-block fullquant pse bug
- 根因类别：条件判断逻辑错误(指针非空检查 vs 布尔标志语义混淆)
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：CheckPerblockQuantParams中per-block量化场景判断是否存在pse(位置偏移)时，使用了`pseType != nullptr`(指针非空检查)，但正确语义应检查`enablePseShift`(布尔标志)。当pseType指针非空但值为0(未启用pse)时，旧代码错误进入检查分支并可能拦截合法输入；反之enablePseShift为true但pseType指针为空时又跳过检查。
- 修复模式：将`pseType != nullptr`替换为语义正确的布尔标志`enablePseShift`
- 可审查性：高
- 审查规则建议：当存在语义明确的布尔标志(如enableXxx)时，优先使用布尔标志而非指针非空检查来判断特性是否启用；审查中应关注"指针非空 vs 功能开关"的语义差异。

---

### c7d7f6b9 [FIA]: fixbug pse\tensorlist\fp8e5m2相关场景拦截
- 根因类别：输入校验缺失 + 类型映射不完整 + 拦截条件不精确
- 涉及文件：attention/fused_infer_attention_score/op_host/fused_infer_attention_score_infershape.cpp, attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：三个子问题：(1) infershape中TORCH_DTYPE_ENUM_VALUE_TO_GE_DTYPE_MAP缺少{1, DT_INT8}映射，删除硬编码改为map查找，新增不支持类型(FP8E5M2)的错误返回；(2) MLA场景+pse/alibi未做拦截，缺少OP_CHECK_IF校验；(3) sys_prefix在Q_S=1时应支持tensorlist，拦截条件从enableTensorList改为enableTensorList && (queryShapeInfo.s > 1)，避免误拦截。
- 修复模式：补充类型映射、增加不支持场景的拦截校验、放宽合法场景的拦截条件
- 可审查性：中
- 审查规则建议：新增算子特性时需同步审查所有相关校验路径，确保不支持的组合被拦截、支持的组合不被误拦截；类型映射表变更时检查所有依赖路径的一致性。

---

### e5988e2b revert
- 根因类别：前序PR引入回归(Revert) + 初始化时序依赖
- 涉及文件：mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h
- 缺陷描述：对MoeDistributeDispatchV2FullMesh进行实质性重构：(1) 移除对moe_distribute_v2_base.h的依赖及InitWinState辅助函数调用，改为内联直接读写GM状态(DataCacheCleanAndInvalid+直接赋值0/1翻转)；(2) 删除globalBS_成员变量，将axisMaxBS_计算从Init末尾提前到赋值epWorldSizeOriginal_之后修正初始化顺序依赖；(3) 移除函数末尾对selfDataStatusGMTensor_的DataCopyPad写状态操作及多个相关成员变量。
- 修复模式：简化GM状态读写逻辑、消除对外部基类工具函数的依赖、修正变量初始化时序
- 可审查性：中
- 审查规则建议：commit message为"revert"的提交需关注实际内容是否为简单回退还是包含新逻辑；涉及多核/通信场景的状态管理变更需审查并发安全性；成员变量的初始化顺序应与声明顺序和依赖关系一致。

---

### 33202735 修复GMM算子日志打印的若干问题
非缺陷提交(纯日志文本修饰)。仅修改aclnn_grouped_matmul.cpp中大量日志字符串的首字母大写(如"x[%lu] is null" -> "X[%lu] is null")，不涉及任何逻辑、条件或数据流变更。

---

### 60550452 alltoallmatmul 修复int4 8卡性能
- 根因类别：属性推断方式错误(shape推断 vs 属性驱动)
- 涉及文件：mc2/allto_all_matmul/op_host/op_tiling/arch32/allto_all_matmul_tiling_910b.cpp, mc2/allto_all_matmul/op_host/op_tiling/arch32/allto_all_matmul_tiling_910b.h
- 缺陷描述：x2的转置判断使用了`bool isTrans = info.K * info.rankSize == x2Dim1`(基于shape推断)，而非直接使用用户传入的x2Transpose属性标志。当shape恰好满足等式但实际未转置时(或反之)会导致N维度取值错误，影响后续tiling计算和tilingKey生成。
- 修复模式：将shape推断式转置判断替换为属性驱动判断；调整8卡int4特定shape下的核心分配参数
- 可审查性：高
- 审查规则建议：矩阵转置等属性应直接使用用户传入的属性值判断，而非通过shape反推(shape推断存在歧义性)；tiling参数中涉及核心数分配的变更应要求性能测试数据佐证。

---

### 99342fa4 win区dump解析工具bugfix
- 根因类别：API误用(logging格式占位符缺失)
- 涉及文件：mc2/tools/dump_analysis/dump_analysis.py
- 缺陷描述：`logging.info("... shape:", len(int32_dis_0_status))`使用print风格的逗号拼接传参，但logging.info的格式字符串中没有%占位符消费第二个参数，导致shape信息不出现在日志输出中。
- 修复模式：改为`logging.info("... shape:%d", len(...))`，使用%d格式化占位符正确输出整数值
- 可审查性：高
- 审查规则建议：logging.info/debug/error等调用中若有额外参数，格式字符串必须包含对应数量的%占位符；可通过pylint的logging-format-interpolation规则自动检测。

---

### 731c9344 GMMSwigluQuant pertoken量化模式 支持 aclnn通路
非缺陷提交(新功能开发)。为GMMSwigluQuant算子的pertoken量化模式增加aclnn通路适配，包含新增测试文件、参数校验逻辑扩展、文档更新等。

---

### 78e07300 修复算子MoeTokenUnpermuteWithRoutingMap在空tensor临界情况与标杆输出存在差异的问题
- 根因类别：空tensor场景处理缺失
- 涉及文件：moe/moe_token_unpermute_with_routing_map/op_host/op_api/aclnn_moe_token_unpermute_with_routing_map.cpp
- 缺陷描述：空tensor(如hidden_size==0)临界情况下，原代码空tensor判断条件不够精确(直接判断permutedTokens->IsEmpty() || sortedIndices->IsEmpty())，且空tensor路径下未对输出tensor做零值初始化，导致输出结果与标杆不一致。paddedMode为true时若permutedTokens为空，未跳过后续Reshape/Mul和InplaceIndexAdd操作。
- 修复模式：修正空tensor判断条件为`sortedIndices->IsEmpty() || (paddedMode == false && permutedTokens->IsEmpty())`；空tensor路径下对所有输出执行ZerosLike+ViewCopy初始化；paddedMode路径增加`!permutedTokens->IsEmpty()`守卫
- 可审查性：高
- 审查规则建议：对算子的空tensor/零维度输入路径进行专项审查，确保所有输出tensor在边界条件下都有确定性初始化值而非返回未定义内容。

---

### fcba7723 MLA非量化支持sparse 0,3,4，G泛化支持1,2,4,8,16,32,64,128
非缺陷提交(新功能开发)。大型特性提交(26个文件、1300+行变更)，为MLA非量化flash attention添加sparse mode 0/3/4支持、G参数泛化支持更多值、layout泛化支持转置。

---

### 0b621cb6 [FAG] fix bn2 sprasemode3 bug
- 根因类别：条件分支遗漏(边界变量clamp缺失)
- 涉及文件：attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_kernel_base.h
- 缺陷描述：FlashAttentionScoreGrad反向kernel在bn2多块分块模式下，sparseMode为RIGHT_DOWN_CAUSAL(mode 3)时，s2EndLen未被正确限制在s2Size范围内。原代码仅在有prefixN的分支中对s2EndLen做了Min clamp，但在else分支(无prefixN、sparseMode==3)遗漏了上界约束，导致反向计算中S2维度的end长度可能超出实际s2Size，产生越界访问或计算错误。
- 修复模式：在else分支添加`if (constInfo.sparseMode == RIGHT_DOWN_CAUSAL) { s2EndLen = Min(s2EndLen, constInfo.commonConstInfo.s2Size); }`
- 可审查性：高
- 审查规则建议：对含多个sparseMode分支的kernel逻辑，审查每个分支下的边界变量(s2EndLen、s1StartLen等)是否都做了合法范围clamp，尤其注意if/else分支中某一分支有约束而另一分支遗漏的情况。

---

### d1d95dfb [mc2]fix matmul_all_reduce ub conflict
- 根因类别：硬件流水线同步缺失(UB数据竞争)
- 涉及文件：mc2/matmul_all_reduce/op_kernel/arch35/matmul_all_reduce_quant_pertoken_comm_int8.h
- 缺陷描述：MatmulAllReduce的pertoken comm int8通路中，matmul计算和低bit通信都使用vec单元且共享UB空间，缺少全核同步屏障导致前一步vec计算结果可能被后一步通信操作覆盖，造成数据竞争和计算结果错误。
- 修复模式：在hccl_.Commit之后、下一轮matmul开始之前插入SyncAll<false>()全核同步调用，确保vec计算和通信操作之间的流水正确串行化
- 可审查性：高
- 审查规则建议：在涉及多流水(matmul+通信)共享UB/向量单元的算子kernel中，审查每个流水阶段切换点是否有恰当的同步屏障(SyncAll)，特别是当不同操作复用同一硬件单元或同一buffer区域时。

---

### 5dce387d MatmulAllReduce CommFp8 Sync Fix
- 根因类别：硬件流水线同步缺失(SyncAll模板参数错误+跨迭代同步缺失)
- 涉及文件：mc2/matmul_all_reduce/op_kernel/arch35/matmul_all_reduce_quant_commfp8_mixed_calc.h, mc2/matmul_all_reduce/op_kernel/arch35/matmul_all_reduce_quant_pertile_comm_fp8.h
- 缺陷描述：与d1d95dfb同类问题但发生在CommFp8通路。两处缺陷：(1) ElementWiseAdd之后SyncAll()使用默认模板参数(SyncAll<true>())，同步语义不正确应使用SyncAll<false>()；(2) StepOneTurn(matmul+quant+通信一个完整流水步骤)执行完毕后缺少SyncAll<false>()同步屏障，当前轮次的通信/量化操作可能与下一轮matmul产生UB冲突。
- 修复模式：将SyncAll()改为SyncAll<false>()修正模板参数；在StepOneTurn调用后新增SyncAll<false>()确保流水阶段间数据一致性
- 可审查性：高
- 审查规则建议：对SyncAll的模板参数使用进行审查，确认<true>和<false>的语义是否符合场景需求；在多阶段流水(matmul -> quant -> comm)的循环体中审查每轮迭代结束时是否有充分的同步屏障。

---

### a3a76213 kvrmsnormropecache support fp8 hif8 quant and recompute fix
非缺陷提交(新特性为主体)。为kvrmsnormropecache算子新增fp8/hifloat8量化类型支持，将原本硬编码的int8_t扩展为模板参数，涵盖proto注册、tiling校验、kernel计算逻辑等全链路改动。虽然包含若干recompute修复(tensor alloc/free位置调整、RopeWithoutQuant增加kCacheRowOffset参数等)，但这些修复与新特性代码深度耦合，整体归类为新特性。

---

### d9caa9d7 增加FIAv4 ATK工程看护用例，并扩展golden中的部分场景未适配问题
非缺陷提交(测试/CI维护)。改动仅涉及测试目录下的json配置文件和python测试执行器，属于ATK自动化测试框架的CI看护用例增补。

---

### 372b8e80 修改prefix拦截&&FIAV4mask资料修改
- 根因类别：参数传递错误 + 校验执行顺序错误
- 涉及文件：attention/fused_infer_attention_score/op_host/arch32/fused_infer_attention_score_tiling_check_consistency.cpp
- 缺陷描述：两个独立缺陷：(1) CheckSystemPrefixShape中创建FiaTilingShapeCompare时传入错误的name常量KEY_NAME，实际应为KEY_SHARED_PREFIX_NAME(校验的是prefix key而非普通key)，错误标识名导致校验日志和错误提示指向错误张量；(2) systemPrefixLen > systemPrefixMaxLen的长度校验被放在CompareShape之前执行，但shape校验通过是长度校验有意义的前提，存在逻辑依赖关系错误。
- 修复模式：修正KEY_NAME -> KEY_SHARED_PREFIX_NAME；调整校验执行顺序(先shape校验再长度校验)；显式返回成功状态码
- 可审查性：高
- 审查规则建议：函数参数是具有相似命名的常量时(KEY_NAME vs KEY_SHARED_PREFIX_NAME)审查应确认参数语义与上下文一致；多个校验步骤之间存在依赖关系时应按依赖顺序排列。

---

### c4ececa0 fix log and check
- 根因类别：校验条件范围错误(过度拦截)
- 涉及文件：gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul.cpp, gmm/grouped_matmul/tests/ut/op_host/test_grouped_matmu_tiling.cpp
- 缺陷描述：ASCEND950平台上，原代码在进入量化类型分支判断之前无条件执行CHECK_COND(isNoActivation, ...)拦截，导致任何带activation的场景被直接拒绝。然而INT8量化的pertoken-perchannel和pertensor-perchannel模式实际支持activation，这个过早的全局拦截导致合法的INT8+activation场景被错误拒绝。
- 修复模式：删除错误的提前全局拦截，让校验下沉到具体分支内执行精确判断；修正UT平台版本配置
- 可审查性：高
- 审查规则建议：校验逻辑不应在分支判断之前做全局拦截，当存在"某些子场景合法、某些子场景非法"时校验应下沉到具体分支内各自执行。

---

### 3f21ad63 dispatch v2 fix winIn addr and sync bug
- 根因类别：GM地址来源错误 + 硬件流水线同步缺失(多处HardEvent缺失) + PipeBarrier位置错误
- 涉及文件：mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h, mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h
- 缺陷描述：三类缺陷：(1) winIn地址错误：核间同步标志的读写使用windowInstatusFp32Tensor_(指向远端window状态地址)，但核间同步标志属于本地rank操作应使用本地rank的winIn地址。(2) 缺失硬件事件同步：SendToMoeExpert中CalTokenSendExpertCnt前缺S_V同步、DataCopyPad前缺V_MTE2同步、statusCleanFp32Tensor_写入前缺V_MTE3同步。(3) PipeBarrier<PIPE_ALL>位置错误：原放在循环结束后，实际需保护的是LocalWindowCopy(含reset操作)。
- 修复模式：修正GM地址来源(远端 -> 本地rank窗口地址)；补充缺失的硬件事件同步(S_V, V_MTE2, V_MTE3)；将PipeBarrier移至正确位置
- 可审查性：低
- 审查规则建议：多核/多rank通信场景中对窗口地址的读写操作必须确认目标是本地rank还是远端rank；所有DMA操作前应确认前序计算阶段已通过对应HardEvent完成同步；PipeBarrier应与其保护的操作紧邻。

---

### 35d0716d alltoallmatmul tiling与A4W4性能修复
- 根因类别：tiling参数缺陷(int4位宽未适配) + 维度校验错误 + GM地址偏移缺失 + 输入校验缺失
- 涉及文件：mc2/allto_all_matmul/op_host/op_tiling/arch32/allto_all_matmul_tiling_910b.cpp, mc2/allto_all_matmul/op_kernel/arch32/allto_all_matmul_a4w4.h
- 缺陷描述：五个缺陷：(1) A4W4的L1/L0 TileShape中k轴切分粒度过小，int4位宽仅fp16的1/4但原配置未利用此特性；(2) pValue(peermem通信buffer容量)未考虑int4的4倍压缩比，三个rank函数均存在；(3) x1Scale维度校验逻辑错误，perTokenScale的第一维应对应alltoall前的完整M轴而非缩小后的M/rankSize；(4) perTokenScaleGM_地址偏移缺失，未加当前rank偏移量导致所有rank读取同一段scale数据；(5) 缺少k轴tokenSize范围校验。
- 修复模式：调整TileShape适配int4位宽；增加pValue的4倍系数；修正x1Scale维度校验条件；增加GM地址rank偏移；新增tokenSize范围校验
- 可审查性：中
- 审查规则建议：tiling参数TileShape应根据数据类型实际位宽调整；通信buffer容量计算必须考虑数据类型压缩比；维度校验条件中涉及rankSize除法时确认是alltoall前还是操作后的维度；GM地址在多rank场景下必须加rank偏移。

## 批次11: 提交 #197-#215 (2026-01-28 ~ 2026-01-30)

---

### a8bc9667 GMMFR，文档格式修复

非缺陷提交(纯文档格式修改)。仅修改5个.md文件的中文格式和表格排布。

---

### c04bfe85 修复archname
- 根因类别：硬件架构标识符错误
- 涉及文件：gmm/grouped_matmul_swiglu_quant_v2/op_kernel/arch35/grouped_matmul_swiglu_quant_v2_pertoken_quant.h
- 缺陷描述：`GmmSwigluAswtPertokenKernel`中`TileCopy`模板参数使用了内部架构代号`Arch::DAV_3510`，应使用对外平台标识`Arch::Ascend950`。错误的架构标识符可能导致kernel编译或运行时选择了错误的tiling/copy策略。
- 修复模式：将`Arch::DAV_3510`替换为`Arch::Ascend950`
- 可审查性：高
- 审查规则建议：Arch枚举值使用时内部代号与对外名称不应混用，统一使用对外平台标识。

---

### 3bb75678 fix fia bug
- 根因类别：多处逻辑缺陷(GQA offset计算错误 + layout分支冗余 + 校验缺失 + 条件判断错误)
- 涉及文件：attention/common/op_kernel/arch35/attenmask.h, attention/common/op_kernel/arch35/flash_attention_score_block_cube.h, attention/common/op_kernel/arch35/flash_attention_score_block_vec_infer.h, attention/common/op_kernel/arch35/flash_attention_score_kernel_base.h, attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp, attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.h
- 缺陷描述：复合修复，涉及Flash Attention多处独立缺陷：(1) attenmask.h中GQA场景`s1Offset`取模计算错误+多余模板参数；(2) flash_attention_score_block_cube.h中冗余layout分支使用了错误的queryOffset；(3) flash_attention_score_block_vec_infer.h中`PostQuant`的`perChannelQuantGQAOffset`遗漏`vec2S1BaseSize`和`subBlockIdx`维度偏移；(4) tiling层缺少n维度和nQ>256校验，GQA ratio检查逻辑冗余，actualSeqLengths解析条件误加enableIFA。
- 修复模式：删除错误GQA取模逻辑；统一用offsetCalculator替代冗余layout分支；补全多维偏移量计算；tiling层添加维度校验；简化GQA ratio检查；修正解析条件
- 可审查性：低(6文件batch fix)
- 审查规则建议：Flash Attention的offset计算应有UT覆盖GQA多head场景；tiling层对所有shape维度均需边界校验。

---

### 4f8bc97c fix matmulallReduce

非缺陷提交(功能调整/特性裁剪)。删除FP8_E5M2和FLOAT4_E1M2等数据类型支持，简化bias类型检查，属于接口收敛。

---

### f195e8f6 group list upper 128 and weight l2 cache on 950PR/DT

非缺陷提交(功能增强+性能优化)。950PR/DT非量化grouped_matmul支持tensorList长度从128放开至1024，新增weight L2 cache控制。

---

### c6d4e407 gather pa kv cache fix
- 根因类别：越界访问(GM读取在边界检查前) + DMA对齐缺失 + 类型精度丢失
- 涉及文件：attention/gather_pa_kv_cache/op_host/gather_pa_kv_cache_tiling_arch35.cpp, attention/gather_pa_kv_cache/op_kernel/arch35/gather_pa_kv_cache_nd.h, attention/gather_pa_kv_cache/op_kernel/arch35/gather_pa_kv_cache_nz.h
- 缺陷描述：GatherPaKvCache算子seqLens超长时aicore error(issue #608)。三处缺陷：(1) nd/nz kernel中`blockTableOffset`越界时仍先执行`blockTablesGm_.GetValue()`读取GM导致非法内存访问；(2) tiling层`maxUbHiddenSize`未对齐32B导致DMA传输错误；(3) `seqLensCopyParams.blockLen`缺少uint32_t转换，DataCopy系列API应使用带类型的Ext版本。
- 修复模式：将GM读取移至越界判断之后(越界时blockId=0)；maxUbHiddenSize做CeilAlign 32B对齐；修正类型转换和API参数类型
- 可审查性：高
- 审查规则建议：GM访问必须在边界检查之后执行；tiling中DMA相关size参数应保证对齐。

---

### acd150e2 GMM切K场景BASEM小于M时校验报错
- 根因类别：边界条件缺失(tiling base block size未与实际shape取min)
- 涉及文件：gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp
- 缺陷描述：GMM SPLIT_K场景tiling中`baseM_`硬编码为SPLITK_BASEM_256(256)，当实际`maxM_`<256时，后续矩阵分块调度因baseM>M导致校验报错或计算异常。
- 修复模式：baseM_赋值后增加判断：若baseM_>maxM_则调整为maxM_向上16对齐值
- 可审查性：高
- 审查规则建议：tiling中所有base block size赋值后应与实际shape维度做min/clamp，防止base超过实际大小。

---

### 826d92e0 修复后量化gm init
- 根因类别：数据类型大小计算错误(reinterpret_cast后size/offset不一致)
- 涉及文件：attention/common/op_kernel/arch32/fia_kernel_nonquant.h
- 缺陷描述：FIA非量化kernel中，输出类型为int8_t(后量化)时，output GM使用half指针初始化。`totalOutputSize`按元素数量计算但未换算为half元素个数(int8是1字节，half是2字节)，导致`singleCoreSize`过大。同时InitOutput调用处又错误地对offset/size除以2，结果只初始化了前半部分输出，后半部分含脏数据造成精度问题。
- 修复模式：计算totalOutputSize时提前除以2(换算为half元素数)，InitOutput调用处去掉多余除2，统一到half元素粒度
- 可审查性：高
- 审查规则建议：output GM使用reinterpret_cast改变数据类型时，所有基于该指针的size/offset必须统一到同一类型元素粒度。

---

### d0a870ea 修复MXFP8量化Cast时SAT_MODE不生效导致的精度差异
- 根因类别：硬件控制寄存器(SPR)配置遗漏
- 涉及文件：moe/moe_init_routing_v3/op_kernel/arch35/moe_v3_common.h, moe/moe_init_routing_v3/op_kernel/arch35/moe_v3_gather_mxfp8_quant.h, moe/moe_init_routing_v3/op_kernel/moe_init_routing_v3_apt.cpp
- 缺陷描述：Ascend 3101上MoE V3算子MXFP8量化Cast需要SAT_MODE(饱和模式)，但kernel未设置溢出控制寄存器`OVERFLOW_MODE_CTRL`(SPR #60)，导致Cast使用默认溢出模式，FP32->MXFP8转换产生精度差异。
- 修复模式：Init中通过SetCtrlSpr设溢出模式为0(saturation mode)；算子入口保存原始溢出模式，结束后恢复
- 可审查性：高
- 审查规则建议：低精度量化(FP8/MXFP8)kernel应检查是否需要显式设置溢出模式寄存器；修改SPR后必须在算子退出前恢复原值。

---

### 454df379 fix kernel select bugs of grouped_matmul

非缺陷提交(空提交)。tree与parent完全相同，无实际文件变更。

---

### 8b84d6df 解决MoeInitRouting/MoeInitRoutingQuantV2算子精度问题
- 根因类别：流水线同步缺失(SyncAll + PipeBarrier + TQueSync)
- 涉及文件：moe/moe_init_routing/op_kernel/moe_gather_out_small_activate_row.h, moe/moe_init_routing_quant_v2/op_kernel/arch35/moe_v2_gather_dynamic_quant_droppad.h, moe/moe_init_routing_quant_v2/op_kernel/arch35/moe_v2_gather_quant_simt.h, moe/moe_init_routing_quant_v2/op_host/moe_init_routing_quant_v2_tiling_base.cpp, moe/moe_init_routing/op_host/CMakeLists.txt, moe/moe_init_routing_quant_v2/op_host/CMakeLists.txt
- 缺陷描述：MoeInitRouting和MoeInitRoutingQuantV2在ascend910_95平台精度问题。三处：(1) 多核场景缺少SyncAll()导致核间数据竞争；(2) 量化kernel中Duplicate后缺PipeBarrier<PIPE_V>()导致Div读取未就绪数据；(3) bfloat16 Cast后缺TQueSync<PIPE_MTE3, PIPE_V>同步。
- 修复模式：在关键计算点插入SyncAll/PipeBarrier/TQueSync同步原语；为910_95平台条件添加编译选项
- 可审查性：中
- 审查规则建议：多核kernel和向量流水线连续操作中，检查是否缺少SyncAll/PipeBarrier等同步原语。

---

### 4b905603 fix ScatterPaKvCache tiling dual-in-out
- 根因类别：tiling逻辑分支遗漏(早期return跳过必要参数计算)
- 涉及文件：attention/scatter_pa_kv_cache/op_host/scatter_pa_kv_cache_tiling_arch35.cpp
- 缺陷描述：ScatterPaKvCache tiling在DUAL_IN_OUT模式下直接return，跳过了v维度(vHandleNumPerLoop/vLoopNum/vTailHandleNum)的tiling计算，导致该模式下v相关tiling参数未初始化。
- 修复模式：DUAL_IN_OUT分支内也执行v维度tiling计算后统一return
- 可审查性：高
- 审查规则建议：tiling函数中不同模式的早期return需确认所有必要tiling参数已被正确计算。

---

### 69f8dff2 fix GmmAddCgmct
- 根因类别：函数调用错误(重命名后调用点未同步更新)
- 涉及文件：gmm/grouped_matmul_add/op_kernel/grouped_matmul_add.cpp
- 缺陷描述：split_k场景tiling key对应的分支仍调用已重命名的`GmmAddAct`函数(已在f2173684中改为`GmmAddCgmct`)。
- 修复模式：将`GmmAddAct`调用替换为`GmmAddCgmct`
- 可审查性：高
- 审查规则建议：函数/类重命名后应全局搜索所有调用点确保一致性更新。

---

### b96fd79d fix g generalization
- 根因类别：变量名拼写错误(typo)
- 涉及文件：attention/common/op_kernel/arch35/attenmask.h
- 缺陷描述：GQA模式计算s1Offset时使用了`consInfo.s1Size`(少了t)，应为`constInfo.s1Size`，导致编译失败或运行时访问错误的结构体成员。
- 修复模式：将`consInfo`修正为`constInfo`
- 可审查性：高
- 审查规则建议：结构体/变量名引用应通过编译器严格检查，启用-Werror。

---

### f2173684 cgmct_fix

非缺陷提交(重构/重命名)。将grouped_matmul_add kernel从Act框架迁移到Cgmct框架，纯命名空间和文件名变更。

---

### 4eca5019 修复sink NZ场景下的精度问题
- 根因类别：计算逻辑错误(归约累加用赋值代替累加 + 符号错误)
- 涉及文件：attention/flash_attention_score_grad/op_kernel/arch32/flash_attention_score_grad_post.h
- 缺陷描述：FlashAttentionScoreGradPost在NZ格式处理dsink时两处算法错误：(1) 循环内`dsinkCalc = -vecOut.GetValue(0)`是赋值而非累加，多个数据块归约结果被覆盖；(2) 写入dsinkGm时应取负但直接写了dsinkCalc。两处共同导致NZ场景梯度缩放因子计算错误。
- 修复模式：赋值改累加`dsinkCalc += vecOut.GetValue(0)`；写入时取负`-dsinkCalc`；新增NZ格式sink场景完整处理分支
- 可审查性：中
- 审查规则建议：归约累加循环应检查是否误用赋值(=)代替累加(+=)；最终结果的符号正确性需交叉验证。

---

### b3ebb5b8 barrier的example运行出现问题，进行修复

非缺陷提交(测试示例参数调整)。仅修改example中K和moeExpertNum参数值。

---

### 79c5d19c 修正gmm activetype报错描述不清晰的问题

非缺陷提交(日志/错误信息改进)。改善错误提示信息的可读性。

---

### e21a3ec1 GMMFR MX aclnn通路提交

非缺陷提交(新功能开发)。GMMFR算子新增MX量化模式aclnn通路支持，+2043行新代码。

---

## 批次12: 提交 #216-#235 (2026-01-23 ~ 2026-01-28)

---

### 9cc118d5 fix: remove common/act

非缺陷提交(构建系统清理/重构)。删除common/act和common/groupedmatmul_act目录安装路径，替换为统一的common/cgmct路径。目录结构整理，非缺陷修复。

---

### bb1ca6f9 【gmm】【groupedmatmulinplaceadd】修改报错信息存在空行
- 根因类别：字符串字面量格式错误
- 涉及文件：gmm/grouped_matmul/op_host/grouped_matmul_infershape_quant_checker.cpp, gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul_910_95_checker.cpp, gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp, gmm/quant_grouped_matmul_inplace_add/下2个文件
- 缺陷描述：C/C++中用`\`行续接时，下一行开头的缩进空格被视为字符串内容，导致OP_LOGE输出的报错信息中间嵌入大段空白（几十个空格），可读性极差。
- 修复模式：将`\`续接后下一行的缩进空格全部移除，让字符串内容紧贴行首。
- 可审查性：高
- 审查规则建议：检测字符串字面量中`\`续接后下一行是否以空白字符开头。

---

### e2ba8a5d 修改低错问题
- 根因类别：Git merge冲突标记残留
- 涉及文件：moe/moe_init_routing_v3/README.md, moe/moe_init_routing_v3/docs/aclnnMoeInitRoutingV3.md, moe/moe_gating_top_k/docs/aclnnMoeGatingTopK.md
- 缺陷描述：README.md和API文档中残留了Git merge冲突标记（`<<<<<<< HEAD`、`=======`、`>>>>>>>`），导致文档内容异常，产品支持状态出现矛盾（一处写`√`另一处写`×`）。
- 修复模式：删除merge冲突标记，保留正确分支内容。
- 可审查性：高
- 审查规则建议：CI/pre-commit hook应检测`<<<<<<<`/`=======`/`>>>>>>>`merge冲突标记。

---

### 9b60bd2a GMM 类算子blockdim相关概念统一为numBlocks

非缺陷提交(命名重构)。将blockDim、cubeBlockDimN等统一重命名为numBlocks、cubeNumBlocksN。纯术语统一的机械替换。

---

### 422763a5 fix gmmsqV1

非缺陷提交(文档增强)。补充产品支持矩阵、术语优化、补充groupListType=count示例说明等。虽然commit message带fix但实际全为文档勘误。

---

### c2d880ba Fix SyncFunc EventId Type
- 根因类别：API返回值类型错误
- 涉及文件：mc2/common/inc/kernel/mc2_kernel_utils.h(新建), mc2下19个头文件
- 缺陷描述：`SyncFunc`中`FetchEventID(event)`返回`AscendC::TEventID`，但原代码用`static_cast<int32_t>`强转为`int32_t eventID`。这是类型不匹配问题——直接cast为int32_t可能导致信息丢失或与SetFlag/WaitFlag的参数类型不匹配。
- 修复模式：用`AscendC::TEventID eventID`接收返回值，去掉static_cast。同时将19个文件中重复的SyncFunc定义统一提取到mc2_kernel_utils.h。
- 可审查性：中
- 审查规则建议：对FetchEventID返回值检查是否用原生TEventID类型接收；检测static_cast将SDK专用类型转为基础类型的模式。

---

### bc5e43c1 涉及核间同步的算子必须设置schedule_mode为1,修复空tensor场景
- 根因类别：调度配置遗漏 + 空tensor判断条件错误
- 涉及文件：moe/moe_init_routing/op_host/moe_init_routing_tiling.cpp, moe/moe_token_permute_with_routing_map/下3个文件
- 缺陷描述：两个独立缺陷：(1) moe_init_routing和moe_token_permute_with_routing_map涉及核间同步但tiling阶段没有SetScheduleMode(1)独占全核，调度器可能将核分配给其他算子破坏同步语义。(2) 空tensor判断条件`if (routingMap->IsEmpty() || permuteTokensOut->IsEmpty())`误将output的IsEmpty也纳入判断，正确逻辑应只在输入routingMap为空时提前返回。同时缺少probs与routingMap的shape一致性校验。
- 修复模式：添加SetScheduleMode(1) + 收紧空tensor判断(去掉output的IsEmpty) + 增加shape校验。
- 可审查性：高
- 审查规则建议：所有使用核间同步的算子检查是否有SetScheduleMode(1)；空tensor提前返回条件不应混淆输入/输出。

---

### e9830dee dispatch、combine算子文档添加performanceInfoOptional

非缺陷提交(纯文档更新)。为dispatch/combine V4算子添加performanceInfoOptional参数文档说明。

---

### ada9091d 修复MoeTokenUnpermute
- 根因类别：workspace大小计算错误/硬编码
- 涉及文件：moe/moe_token_unpermute/op_host/moe_token_unpermute_tiling.cpp
- 缺陷描述：TilingMoeTokenUnpermute函数硬编码16MB workspace(sysWorkspaceSize = 16*1024*1024)，但后续TilingCompute内通过GetLibApiWorkSpaceSize()动态获取正确大小并覆盖。两处都调用GetWorkspaceSizes(1)写入workspaces[0]，前者硬编码被后者覆盖。若动态值不正确或代码顺序变化，会导致workspace分配不正确。
- 修复模式：删除冗余硬编码workspace设置，统一由动态计算决定。
- 可审查性：高
- 审查规则建议：检测workspace大小是否存在硬编码magic number；检查GetWorkspaceSizes是否被多次写入导致覆盖。

---

### c03c66cd dispatch&combine性能打点问题修复
- 根因类别：轮询逻辑缺陷导致重复计数
- 涉及文件：mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch_a2.h
- 缺陷描述：WaitDispatch函数的while循环轮询各rank通信完成标志时，对已完成的rank也会重复执行DataCopy+SyncFunc读取状态。原代码用DCCI(DataCacheCleanAndInvalid)清零已完成rank标志位来避免重复累计recvFlagNum，但远端内存清零方式不可靠。
- 修复模式：引入isVisited布尔数组，已完成的rank标记为true并continue跳过。用本地visited标记替代远端内存清零。
- 可审查性：中
- 审查规则建议：轮询完成标志的循环中检查是否有对已完成项的重复处理；用远端内存清零防止重复计数标记为高风险模式。

---

### 5202e30b 修复MoeTokenPermuteWithRoutingMapGrad

非缺陷提交(功能增强)。为MoeTokenPermuteWithRoutingMapGrad算子新增BF16+FLOAT混合精度支持。虽然commit message写"修复"但实际是扩展数据类型支持。

---

### e44028b1 fix deter casual GQA
- 根因类别：GQA维度缩放/offset计算
- 涉及文件：attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp
- 缺陷描述：CalcleCausalDeterParam()计算rUpper时未考虑GQA场景（g!=1）的修正量。当fBaseParams.g!=1时需额外加上(m+m1+1)*t1修正项rm3。缺少此修正导致GQA+causal+deterministic模式下tiling划分不正确。
- 修复模式：根据g!=1条件计算rm3修正项并加入rUpper累加。覆盖ell==0和ell!=0两个分支。
- 可审查性：中
- 审查规则建议：tiling参数计算涉及MHA/MQA/GQA多模式组合时，检查是否所有模式都被覆盖，特别是g(group数)相关条件分支。

---

### 8a64e103 【FAG】swizzle block num optimize

非缺陷提交(性能优化)。将swizzle连续块数量从固定值改为根据S1大小动态计算。

---

### b481f7c3 fix sfa race bug
- 根因类别：硬件流水线同步缺失(DMA后缺同步屏障)
- 涉及文件：attention/sparse_flash_attention/op_kernel/sparse_flash_attention_service_vector_mla.h
- 缺陷描述：MergeKv函数中DataCopyPad将UB数据通过MTE3通道拷贝到GM，该操作是异步的，但之后没有SetFlag/WaitFlag同步。后续代码可能在DMA完成前读写源缓冲区，产生数据竞争。
- 修复模式：在DataCopyPad后添加SetFlag<MTE3_S>和WaitFlag<MTE3_S>配对，使用新flag编号SYNC_INPUT_V0BUF_FLAG=6。
- 可审查性：高
- 审查规则建议：所有DataCopy/DataCopyPad后，若存在对源或目标缓冲区的复用/读取，必须确保有对应SetFlag/WaitFlag同步对。

---

### 4217f6d3 bugfix
- 根因类别：循环边界条件错误(变量名用错)
- 涉及文件：attention/lightning_indexer/op_kernel/lightning_indexer_kernel.h
- 缺陷描述：SplitCore函数内层for循环使用s2BaseNum作为上界，但实际应使用s2Loop。s2Loop在isSparseCountOver2K为true时被设为0或1，使用s2BaseNum意味着本应跳过的循环仍执行s2BaseNum次，导致访问无效数据。
- 修复模式：将`s2Idx < s2BaseNum`改为`s2Idx < s2Loop`。
- 可审查性：高
- 审查规则建议：当变量被赋值后在紧邻循环中未使用（而使用了名称相似的另一变量），应标记为可疑。

---

### 685562cc 修复aclnn storageShape导致的问题
- 根因类别：变量作用域错误 + 条件判断对象混淆
- 涉及文件：gmm/grouped_matmul_finalize_routing/op_host/op_api/aclnn_grouped_matmul_finalize_routing.cpp
- 缺陷描述：两个独立bug：(1) storageShape变量声明在if(INT32)分支内，SetStorageShape也只在该分支执行，但语义上所有路径都应设置storageShape。(2) 空指针校验条件`x2->GetDataType()==DT_INT4`错误地检查了源tensor x2而非目标tmpWeight的数据类型。
- 修复模式：变量声明提升作用域到if之前，SetStorageShape移到if之后；条件判断对象从x2改为tmpWeight。
- 可审查性：高
- 审查规则建议：检查SetStorageShape等属性设置是否被不必要地限制在条件分支内；条件判断中引用的tensor对象是否与语义一致(源vs目标混淆)。

---

### d2f14888 sycn dev code
- 根因类别：多类混合(边界检查缺失+变量未初始化+溢出防护缺失)
- 涉及文件：gmm/grouped_matmul/下2个文件, gmm/grouped_matmul_swiglu_quant_v2/下3个文件
- 缺陷描述：三个可识别缺陷：(1) groupNum通过static_cast<int32_t>强转赋值但缺少上界校验，修复新增groupNum>GMM_MAX_GROUP_LIST_SIZE(1024)越界检查。(2) AnalyzeAttrs函数缺少inputParams_.groupType=SPLIT_M初始化，tiling计算依赖groupType决定L1分块策略。(3) kernel函数新增浮点溢出模式控制SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL>(0)，防止nan/inf。
- 修复模式：边界检查增强 + 变量初始化补全 + 数值稳定性防护。
- 可审查性：中(多种改动混在sync dev code中)
- 审查规则建议：外部输入维度/数量值必须有上下界校验；tiling参数结构体所有字段使用前必须显式初始化。

---

### 78272790 fix AttentionToFFN/FFNToAttention aclnn demo and cmake
- 根因类别：API误用 + 构建配置错误
- 涉及文件：mc2/attention_to_ffn/examples/test_aclnn_attention_to_ffn.cpp, mc2/attention_to_ffn/op_host/CMakeLists.txt, mc2/ffn_to_attention/op_host/CMakeLists.txt
- 缺陷描述：三类问题：(1) HcclCommInitAll被错误地放在for循环中多次调用，但该函数语义是一次性初始化所有communicator。(2) CMakeLists.txt中OPTYPE和ACLNNTYPE参数存在重复值。(3) FFN Worker等待超时10秒不够，增加到30秒。
- 修复模式：去除冗余循环 + CMake参数去重 + 超时值调整。
- 可审查性：高
- 审查规则建议：InitAll/InitGroup语义的集合初始化函数不应在循环中调用；CMake宏参数列表检查重复项。

---

### b58c6cbf aclnnGroupedMatmul多多多说明

非缺陷提交(纯文档更新)。新增关于多tensor场景下shape约束说明。

---

### 96e8ecff sfag算子修复精度问题，算子性能优化，放开k=2048的限制
- 根因类别：多核并发写冲突 + 维度参数错用 + 尾块精度逻辑错误(compound fix)
- 涉及文件：attention/sparse_flash_attention_grad/下10个文件(约800行变更)
- 缺陷描述：多层面精度/正确性问题：(1) cube2中V矩阵读取使用错误stride维度dimDv应为dimDTotal，且数据源引用了错误的GM地址。(2) cube4/cube5中matmul结果通过ScatterFixOut直接scatter写到dkWorkspaceGm/dvWorkspaceGm，多核并行写同一地址产生竞争。isFixOut原为false导致Fixpipe未执行累加。修复改为每core独立workspace+SetAtomicAdd原子累加。(3) CalSoftmax/CalSoftmaxGrad的actualSelS2在尾块未选场景下未正确更新，修复为每次根据isLastBasicBlock和lastBlockSize重新计算。(4) dK的scaleValue乘法从post阶段移到ScatterAdd阶段，修正计算顺序。
- 修复模式：每core独立workspace+原子累加替代直接scatter + 维度参数修正 + 尾块逻辑局部化。
- 可审查性：低(精度修复与性能优化深度耦合，跨10个文件约800行变更)
- 审查规则建议：多核并行写同一输出地址必须使用原子操作或独立workspace再归约；scatter/gather维度参数必须与实际数据layout严格对应。

## 批次13: 提交 #236-#255 (2026-01-21 ~ 2026-01-23)

---

### 33d0b7bccf 修复MoeFinalizeRouting算子问题

非缺陷提交(纯文档格式修复)。diff仅修改aclnnMoeFinalizeRouting.md，将markdown标题行末尾误带的``` ``` ```删除，不影响任何运行时行为。

---

### e670c5622 代码规范修改，拦截遗漏修改
- 根因类别：条件分支遗漏(新layout/参数未同步更新)
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_kernel_noquant_mla.h, attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_v2.cpp, attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：三处独立缺陷：(1) MLA kernel中constInfo.isTNDOut未从inputParamsRegbase.isTNDOut赋值，TND输出layout标志始终为默认值；(2) fused_infer_attention_score中layout判断只处理"NTD"而遗漏"NTD_TND"变体，导致Lse shape校验缺失；(3) per-block quant参数校验中dequant scale的s维度除数硬编码为128U/256U，应使用变量fp8QBlockSize/fp8KVBlockSize，当实际block size不等于128/256时校验结果错误。
- 修复模式：补充遗漏赋值 + 扩展条件分支覆盖新layout + 硬编码常量替换为运行时变量
- 可审查性：中
- 审查规则建议：引入新layout类型时全局搜索所有layout判断分支确保新类型被覆盖；参数校验中与配置相关的数值(如blockSize)不应硬编码，应使用对应的运行时变量。

---

### 26daba7c3 gmm add check on 950PR and gmm add v1 support 950PR

非缺陷提交(新增平台支持+输入校验增强)。为950PR平台添加empty tensor校验、放开weight transpose限制、更新文档。属于新特性支持。

---

### dcc515a44 优化GroupedMatmul算子A8W4,A4W4

非缺陷提交(性能优化+新功能)。新增动态tiling函数优化核间负载均衡、L2Cache禁用优化、A4W4右矩阵转置支持等。均为性能调优和新功能扩展。

---

### 99755a1bb Layered Dispatch And Combine Bug Fix
- 根因类别：并发通信DMA参数错误 + flag残留误判
- 涉及文件：mc2/moe_distribute_combine/op_kernel/moe_distribute_combine_a2_layered.h, mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch_a2_layered.h
- 缺陷描述：两处缺陷：(1) Combine侧写flag时DataCopy的offset乘4、copy长度为4均不正确(flag实际类型uint64_t=8字节)，导致flag写入位置和大小与读取端不匹配，大BS场景下部分token的flag写到错误地址，接收端永远读不到到达标志造成token丢失。(2) Dispatch侧前后两次dispatch若属性不一致，UB中残留的旧flag值可能恰好等于SHOULD_SEND_FLAG_VALUE，导致接收端误判token已到达。
- 修复模式：改用DataCopyPad并使用sizeof(uint64_t)精确控制拷贝大小 + flag值中叠加magicVal_(时间轮)防止残留误判
- 可审查性：低(需理解RDMA多卡通信协议和分层MoE token传输机制)
- 审查规则建议：DataCopy/DataCopyPad的copy长度必须与实际数据类型sizeof一致；多轮通信中的flag/信号量必须包含轮次标识防止残留数据被误读。

---

### 1c635788c fix-debug combine layered
- 根因类别：计算逻辑错误(IPC内存可用大小多减了无关偏移量)
- 涉及文件：mc2/moe_distribute_combine/op_kernel/moe_distribute_combine_a2_layered.h
- 缺陷描述：GM2IPC函数中计算每张卡IPC slice可存储的最大token数时，原代码`maxBsInRankSizeOnIpc = (ipcSliceSize - IPC_DATA_OFFSET) / localMoeExpertNum_ / tokenSize`中IPC_DATA_OFFSET与当前分配逻辑无关不应被减去，导致可用空间被低估，大BS场景下可能提前触发空间不足或数据截断。
- 修复模式：移除错误的偏移量减法
- 可审查性：高
- 审查规则建议：内存可用空间计算中的每一个减法/偏移量都需要有明确的对应用途；审查时重点关注被减去的常量是否确实属于当前分配上下文。

---

### a54af6d2e bugfix: fix libopai undefined symbol
- 根因类别：构建配置缺陷(CMakeLists.txt参数缺失和路径错误)
- 涉及文件：attention/attention_worker_scheduler/op_host/CMakeLists.txt, ffn/ffn_worker_scheduler/CMakeLists.txt, ffn/ffn_worker_scheduler/op_host/CMakeLists.txt
- 缺陷描述：三处构建配置问题：(1)(3) add_modules_sources()缺少OPTYPE和ACLNNTYPE参数，模块源文件未被正确注册到构建系统，符号不会编入libopai.so；(2) file(GLOB RELATIVE . ...)的RELATIVE基准路径使用`.`而非`${CMAKE_CURRENT_SOURCE_DIR}`，out-of-source构建时路径解析错误导致子目录未被发现。最终libopai.so缺少相关符号，运行时dlopen失败。
- 修复模式：补全CMake宏参数 + 修正file(GLOB)的RELATIVE基准路径
- 可审查性：高
- 审查规则建议：所有add_modules_sources()调用必须包含OPTYPE和ACLNNTYPE参数；file(GLOB RELATIVE ...)的基准路径应始终使用${CMAKE_CURRENT_SOURCE_DIR}，禁止使用`.`。

---

### ad627275f NTD 叠加特性修复
- 根因类别：条件分支遗漏(NTD layout在多个代码路径中未被正确处理)
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_block_vec_infer.h, attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：三处缺陷：(1) SoftmaxLseCopyOut中Lse输出stride的GQA特殊处理仅在LAYOUT_TND分支生效，遗漏LAYOUT_NTD，NTD layout下Lse的dstStride计算走错分支产生数据输出错位；(2) CheckIO中当queryShapeInfo.s==1(decode场景)时无条件启用enableIFAMask和enableIFA，但NTD layout不应走IFA路径；(3) PFATilingDataconvert中isGqa设置条件人为排除NTD但NTD下isGqa也应根据isIFA设置。
- 修复模式：在条件分支中正确纳入或排除NTD layout的特殊处理
- 可审查性：中
- 审查规则建议：新增layout类型时逐一审查所有layout相关条件分支(特别是IFA/GQA/stride计算等关键路径)；decode场景(s==1)的特殊优化路径需枚举验证所有layout组合的正确性。

---

### e13d1ec19 modify datacopy overflow
- 根因类别：整数类型错误/溢出(uint16_t截断)
- 涉及文件：mc2/common/inc/kernel/mc2_nd_to_nz.h
- 缺陷描述：CastBFtoFloat函数中cpInLen(size*sizeof(bfloat16_t))和cpOutLen(size*sizeof(float))均声明为uint16_t，uint16_t最大值65535，当size超过32767个bfloat16元素或16383个float元素时长度被截断，DataCopy只拷贝部分数据导致计算结果错误。DataCopyParams的blockLen字段本身也是uint16_t。
- 修复模式：将uint16_t改为uint32_t，同时将DataCopyParams/DataCopyPadParams替换为Ext版本支持uint32_t的blockLen
- 可审查性：高
- 审查规则建议：DataCopy的长度计算结果不应赋值给uint16_t变量；当数据量可能超过64KB时必须使用Ext版本的DataCopy参数结构体。

---

### bfef435e6 remove blocksize and layout constrain for no quant

非缺陷提交(约束放宽/新特性支持)。为非量化场景放宽blockSize和layout约束条件(blockSize对齐要求从128降到16，范围扩展到[16,1024])，属于有意扩展算子支持范围。

---

### 2df5e03c7 fix pse datacopy error due to large s2
- 根因类别：整数类型错误/溢出(uint16_t截断)
- 涉及文件：attention/common/op_kernel/arch35/pse.h
- 缺陷描述：当actualS2Len很大时，计算出的srcStride可能超过uint16_t最大值65535，但DataCopyParams.srcStride字段是uint16_t类型。溢出后静默截断导致数据拷贝地址错误产生错误结果。
- 修复模式：先将srcStride计算为int64_t中间变量，增加srcStride <= UINT16_MAX的溢出检查条件，超出范围时走DataCopyExtParams慢路径
- 可审查性：高
- 审查规则建议：当赋值给宽度较小的整数类型字段(如uint16_t)时，检查右侧表达式是否可能超出目标类型范围；所有DataCopyParams的stride/blockLen字段赋值前应有显式范围检查。

---

### e5dbe5b5c moe_finalize_routing_v2 DB分支问题修复
- 根因类别：双缓冲(Double Buffer)条件分支逻辑错误
- 涉及文件：moe/moe_finalize_routing_v2/op_kernel/moe_finalize_routing_v2_fp_cuth_k2.h, moe/moe_finalize_routing_v2/op_kernel/moe_finalize_routing_v2_fp_cuth_k4.h
- 缺陷描述：两个问题：(1) k2版本中ISBIASEXIST=false时else分支执行Adds(x, x, 0, dataLen)，无意义的向量操作可能导致双缓冲DB0/DB1间数据竞争或pipeline同步问题；(2) k4版本中bias的Add操作被错误放在第二个DB分支统一处理DB0-DB3，但各DB的invalid row判断是独立的。当DB0有效但DB1无效(被Duplicate为0)时，DB0的bias加法被跳过。
- 修复模式：k2中删除无意义的Adds(x, x, 0) + k4中将bias Add拆分到各自DB的else分支中独立执行
- 可审查性：中
- 审查规则建议：双缓冲场景中每个buffer的计算操作应在其对应的条件分支中独立完成，不应合并不同buffer的操作；检查Adds(x, x, 0)或Muls(x, x, 1)这类no-op向量操作是否为无意义代码。

---

### 07dfd9c3e GQA per-block全量化支持NTD_TND

非缺陷提交(新功能)。为GQA per-block全量化增加NTD/TND layout支持，属于功能扩展。

---

### 3c413a90f bugfix: empty tensor tilingdata reset
- 根因类别：tiling数据结构不一致(host/kernel结构体不匹配)
- 涉及文件：attention/flash_attention_score/op_host/flash_attention_score_tiling.cpp, attention/flash_attention_score/op_kernel/arch32/flash_attention_score_template_tiling_key.h, attention/flash_attention_score/op_kernel/arch32/flash_attention_score_tiling.h
- 缺陷描述：empty tensor场景下host侧使用FlashAttentionScoreEmptyInputTilingData结构体(独立小结构)，kernel侧使用FlashAttentionScoreGeneralTilingData，两者内存布局不一致——host侧只填充少数字段，kernel侧按完整大结构体解析，读到未初始化垃圾值引发aicore error。
- 修复模式：删除独立的EmptyInputTiling类，改用完整TilingData结构体+reset()方法归零所有字段，确保host/kernel一致
- 可审查性：高
- 审查规则建议：tiling host侧写入的结构体类型必须与kernel侧读取的结构体类型严格一致；通过context->GetTilingData<T>()获取的类型T应与kernel侧声明的tiling struct匹配。

---

### 82013a396 sink tiling bug
- 根因类别：结构体内存布局/字段偏移错误
- 涉及文件：attention/flash_attention_score/op_kernel/arch32/flash_attention_score_tiling.h, attention/flash_attention_score_grad/op_kernel/arch32/flash_attention_score_grad_tiling.h
- 缺陷描述：InputParams结构体中needSinkOp字段位置不当，导致host侧写入和kernel侧读取的偏移量不匹配。梯度tiling结构体中padding数组大小错误(7应为3)，多余padding导致后续字段偏移错位。
- 修复模式：调整needSinkOp字段位置+修正padding数组大小，使内存布局与host侧一致
- 可审查性：高
- 审查规则建议：tiling结构体字段顺序和padding必须在host和kernel两侧严格对应；结构体添加新字段后应验证总大小和每个字段的offset是否与另一侧一致；建议使用static_assert或offsetof校验关键字段偏移。

---

### 70898673b FAG sink精度修复
- 根因类别：workspace偏移错误+初始化错误+DataCopyPad padding硬编码+索引计算错误+padding区域脏数据(compound fix)
- 涉及文件：attention/flash_attention_score_grad/op_host/arch32/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.cpp, attention/flash_attention_score_grad/op_kernel/arch32/下3个文件
- 缺陷描述：FAG Sink功能存在多个互相关联的缺陷：(1) dsinksum的workspace分配和偏移计算放在错误位置，offset与实际分配不匹配；(2) InitOutput初始化dsinksum时使用错误base地址且缺少核类型/核索引条件判断；(3) DataCopyPad的isPad和rightPadding硬编码，数据已8对齐时硬补8个float导致读取到workspace外脏数据；(4) dsinksum写入偏移计算公式错误；(5) Mul运算前未清零padding区域引入脏数据。
- 修复模式：重新组织workspace分配顺序 + 修正地址和条件 + 动态决定padding + 修正偏移公式 + padding区域清零
- 可审查性：低(涉及多个文件、多处关联修改、复杂的workspace内存布局)
- 审查规则建议：workspace的分配和使用必须严格按照相同顺序和条件；DataCopyPad的isPad和rightPadding应根据实际数据大小动态计算避免硬编码；向量运算前应确保padding区域已初始化为0。

---

### 03909dce5 Fix FAG Sink MM12 Nzout Bug
- 根因类别：操作时序错误(执行顺序/数据依赖)
- 涉及文件：attention/flash_attention_score_grad/op_kernel/arch32/flash_attention_score_grad_s1s2_bn2gs1s2_sab.h
- 缺陷描述：SubGrapB函数中将vecCopyOutBuffer拷贝到GM的DataCopyPad操作被放在Sink处理逻辑之前，但Sink逻辑会修改相关中间数据，在Nzout格式下数据拷贝和Sink计算产生竞争导致输出结果不正确。
- 修复模式：将DataCopyPad到GM的代码块整体移到Sink处理逻辑之后执行
- 可审查性：中
- 审查规则建议：在硬件pipeline架构下数据输出到GM的操作应放在所有依赖同一buffer的向量计算之后；代码块移动时应验证所有event信号的set/wait配对是否正确。

---

### 5f8f4d5fa fix gmmsqV1 ut
- 根因类别：UT测试数据类型错误
- 涉及文件：gmm/grouped_matmul_swiglu_quant/tests/ut/op_host/op_api/test_aclnn_grouped_matmul_swiglu_quant.cpp, gmm/grouped_matmul_swiglu_quant/tests/ut/op_host/op_api/test_aclnn_grouped_matmul_swiglu_quant_weight_nz.cpp
- 缺陷描述：UT中weightScale的tensor类型声明为ACL_INT64，但实际weight scale应为ACL_FLOAT类型，导致UT使用错误数据类型构造测试输入。
- 修复模式：将weightScale的TensorDesc类型从ACL_INT64改为ACL_FLOAT
- 可审查性：高
- 审查规则建议：UT中tensor的数据类型应与算子规格/接口文档中定义的类型一致；检查scale类tensor是否使用了浮点类型。

---

### ab7c48a6c fix scatter_pa_kv_cache oom aic
- 根因类别：DataCopy对齐要求导致越界读取(OOM/AIC异常)
- 涉及文件：attention/scatter_pa_kv_cache/op_kernel/arch35/下4个文件
- 缺陷描述：原代码DataCopy(dst, src, RoundUp(size))将size向上对齐到32B边界，但当源GM地址处的实际有效数据长度不足对齐后的长度时，DataCopy从GM读取超出有效数据范围的内存触发OOM或AIC异常。headSize不是32B对齐的倍数时必然发生。
- 修复模式：将所有DataCopy替换为DataCopyPad，使用DataCopyExtParams指定精确blockLen，通过DataCopyPadExtParams配置isPad=0避免越界读取
- 可审查性：高
- 审查规则建议：当GM数据的实际长度可能不满足DataCopy的对齐要求时应使用DataCopyPad替代DataCopy；检查所有使用RoundUp作为DataCopy长度参数的调用点。

---

### a248cefc1 fix CheckFeatureLse function

非缺陷提交(空提交)。该commit的tree与parent的tree完全一致，无任何文件变更。是merge产生的空commit。

---

## 批次14: 提交 #256-#275 (2026-01-15 ~ 2026-01-21)

---

### d0ede159b fix v3 infershape
- 根因类别：API误用/接口参数错误(GetInputShape vs GetOptionalInputShape)
- 涉及文件：moe/moe_init_routing_v3/op_host/moe_init_routing_v3_infershape.cpp
- 缺陷描述：对optional的scale和offset输入使用了GetInputShape()而非GetOptionalInputShape()。GetInputShape()在输入为空时返回异常结果或触发错误，而这两个输入在设计上允许为nullptr。
- 修复模式：将GetInputShape替换为GetOptionalInputShape
- 可审查性：高
- 审查规则建议：检测所有GetInputShape调用，对照算子定义文件中标记为optional的输入，确认是否应使用GetOptionalInputShape。

---

### ef22ac8f1 修复cust example头文件引用优先级问题
- 根因类别：构建配置缺陷(头文件include路径顺序)
- 涉及文件：build.sh
- 缺陷描述：g++编译命令中-I ${INCLUDE_PATH}在-I ${CUST_INCLUDE_PATH}之前，导致自定义算子的头文件被系统内置头文件覆盖。当cust算子与内置算子头文件同名时，编译器优先找到内置版本，产生链接或行为错误。
- 修复模式：交换两个-I参数的顺序，将${CUST_INCLUDE_PATH}放在${INCLUDE_PATH}前面
- 可审查性：高
- 审查规则建议：检测构建脚本中-I参数的顺序，cust/vendor路径应优先于系统路径。

---

### a58c8a0b9 FA算子B模板同步问题bugfix
- 根因类别：硬件流水线同步缺失(event信号使用错误导致数据竞争)
- 涉及文件：attention/flash_attention_score/op_kernel/arch32/flash_attention_score_bn2gs1s2_b.h
- 缺陷描述：FlashAttention算子B模板的hasSink场景中，原代码额外分配了独立的eventIdVToMte2Sink同步事件。hasSink时第一个if触发了eventIdVToMte2A的SetFlag，第二个if又触发了eventIdVToMte2Sink的SetFlag，但对应的WaitFlag只等了eventIdVToMte2Sink，没有消费eventIdVToMte2A，导致event信号积压引起死锁或数据竞争。
- 修复模式：删除冗余的event变量，统一使用eventIdVToMte2A；修正条件分支为互斥(!hasSink vs hasSink)
- 可审查性：低(需深入理解Ascend硬件event同步机制)
- 审查规则建议：检测同一代码块中分配多个相同类型HardEvent的情况(如多次AllocEventID<HardEvent::V_MTE2>)，可能暗示同步逻辑冗余或错误。

---

### 25c72d064 Fix RecurrentGatedDeltaRule UT

非缺陷提交(UT测试修复)。修复UT断言逻辑中||运算符优先级导致条件恒真的问题，以及CMakeLists中tiling文件路径引用。属于测试代码自身缺陷，不影响产品功能代码。

---

### e5aced81a RainFusion support BNSD kvcache nullptr && TND check

非缺陷提交(功能增强)。为RainFusionAttention算子新增BNSD格式下kvcache可传nullptr的支持能力，新增TND格式seqlen总和校验，是完整的特性开发而非缺陷修复。

---

### b48e75b7c AllReduceMM/MMReduceScatter_fix
- 根因类别：条件分支遗漏/缺陷(模板分支未覆盖所有参数组合导致ND/NZ格式混淆)
- 涉及文件：mc2/all_gather_matmul/op_kernel/all_gather_matmul.cpp, mc2/matmul_reduce_scatter/op_kernel/matmul_reduce_scatter.cpp
- 缺陷描述：原代码只有BIAS_CAST的true/false两个分支，两个分支内bType都固定使用NZ格式。当模板参数ND2NZ_OPT为false时(B矩阵本身是ND格式)，bType仍被设为NZ格式，导致Matmul引擎按NZ布局读取ND数据，产生静默的数值错误。同时使用普通if而非if constexpr，编译期无法剪枝。
- 修复模式：用if constexpr展开IsFullMesh x IsNd2Nz x IsBias的全部4种组合，每种组合正确设置bType(NZ vs ND)
- 可审查性：中
- 审查规则建议：检测模板函数中使用普通if分发模板布尔参数的代码；检测模板参数存在但分支内对应类型不随之变化的情况。

---

### b5023d8fe 拦截sink场景下QKV float32入参的情况
- 根因类别：输入校验缺失(sink场景下不支持float32但未做拦截)
- 涉及文件：attention/flash_attention_score_grad/op_api/aclnn_flash_attention_score_grad.cpp, 对应docs/
- 缺陷描述：FlashAttentionScoreGrad在sink场景(sinkInOptional != nullptr)下不支持float32数据类型的QKV输入，但aclnn层未做校验拦截，用户传入float32+sink时产生未定义的计算结果或崩溃。文档也错误地将FLOAT32列为支持类型。
- 修复模式：aclnn层新增当sinkInOptional非空且query dtype为DT_FLOAT时返回ACLNN_ERR_PARAM_INVALID；同步更新文档移除sink场景下FLOAT32类型声明
- 可审查性：高
- 审查规则建议：对每个aclnn算子的数据类型校验逻辑，检查是否覆盖了所有可选参数(optional tensor)非空时的类型约束；文档中声明的支持类型列表应与代码校验逻辑一致。

---

### c79715514 nsa compress with cache 校验修复
- 根因类别：输入校验缺失(空tensor和非ND格式未做拦截)
- 涉及文件：attention/nsa_compress_with_cache/op_host/op_api/aclnn_nsa_compress_with_cache.cpp, 对应docs/
- 缺陷描述：NsaCompressWithCache算子在aclnn层缺少两项关键校验：(1) input/weight/outputCache为空tensor(ShapeSize为0)时未拦截，会导致后续tiling或kernel阶段异常；(2) 输入tensor为非ND格式时未拦截，算子仅支持ND格式。文档中也有"使用该功能可传入nullptr"应为"不使用"的笔误。
- 修复模式：新增CheckIsEmptyTensor和CheckNDFormat校验函数，在GetWorkspaceSize入口处拦截；同步修正文档
- 可审查性：高
- 审查规则建议：对所有aclnn算子的GetWorkspaceSize入口，检查是否存在空tensor校验和格式校验。

---

### f736fef3f modify gsAxis performance debug

非缺陷提交(性能优化)。将循环内重复计算的表达式提取为局部变量，纯公共子表达式提取(CSE)，功能语义不变。

---

### 312d30359 Post tilingdata 结构体64位对齐
- 根因类别：赋值遗漏/结构体字段遗漏(tiling结构体未满足硬件64位对齐要求)
- 涉及文件：attention/flash_attention_score_grad/op_kernel/arch32/flash_attention_score_grad_tiling.h
- 缺陷描述：tiling数据结构中最后一个字段baseMN是uint32_t(4字节)，导致结构体总大小不是8字节整数倍。Ascend芯片DMA引擎要求传输数据大小8字节对齐，未对齐时DMA多搬或少搬字节，导致读取未初始化内存数据或覆盖相邻内存。
- 修复模式：在baseMN字段后添加4字节padding数组uint8_t PostParamsPH[4] = {}使结构体满足8字节对齐
- 可审查性：高
- 审查规则建议：对所有tiling数据结构体，静态检查sizeof是否为8的倍数；可编写编译期static_assert(sizeof(TilingData) % 8 == 0)。

---

### c3d301162 添加unlikely，减少sink分支可能带来的性能影响

非缺陷提交(性能优化)。将sink分支条件用unlikely()包裹，告诉编译器该分支大概率为false以优化分支预测。功能正确性不受影响。

---

### 23d4b8e93 select_attention_operators

非缺陷提交(新特性开发)。引入基于Quest论文的block-sparse mask predictor完整实现，约100个新增文件，不修改已有代码。

---

### 10f8a80b7 fix gather_pa_kv_cache ut

非缺陷提交(测试代码重构)。将UT从旧框架迁移到新框架，不涉及产品代码修改。

---

### 9be387c56 fix fia aclnn sample for opensource master

非缺陷提交(文档示例代码修正)。修改4个md文件中的示例代码(attenMask shape、变量名规范化)，不影响任何编译产出。

---

### edd2099ff 补充过滤脚本遗漏的内容

非缺陷提交(文档补充与产品名称整改)。约100个md文件变更，纯文档内容补齐和名称规范化。

---

### 7da5e5332 同步grouped_matmul_add代码

非缺陷提交(示例代码质量改进)。将示例代码的裸指针手动释放改为RAII智能指针模式，不涉及产品代码。

---

### 3639e55e7 fix moe_token_permute_with_routing_map example
- 根因类别：UT测试数据类型错误(示例代码CreateAclTensor参数与宿主数据类型不匹配)
- 涉及文件：moe/moe_token_permute_with_routing_map/examples/test_aclnn_moe_token_permute_with_routing_map.cpp
- 缺陷描述：三处数据类型不匹配：(1) indicesHostData为std::vector<int>(4字节)但CreateAclTensor指定ACL_BOOL(1字节)，内存布局完全不匹配；(2) x的宿主数据为std::vector<float>但创建tensor时指定ACL_BF16，float32与bfloat16内存表示不同导致数据截断；(3) expandedXOut同样从ACL_BF16改为ACL_FLOAT。
- 修复模式：统一宿主数据容器类型与aclDataType枚举值的语义映射
- 可审查性：高
- 审查规则建议：对所有CreateAclTensor调用，静态检查vector模板类型与aclDataType参数的一致性。

---

### c65fdcacb fix opapi ut bugs
- 根因类别：构建配置缺陷(测试退出码被后续清理命令覆盖 + 多芯片打桩不完整)
- 涉及文件：tests/ut/framework_normal/op_api/CMakeLists.txt, tests/ut/framework_normal/op_api/scripts/clean_opapi_stub.py, tests/ut/framework_normal/op_api/scripts/generate_opapi_stub.py
- 缺陷描述：两个独立缺陷：(1) CMake中bash -c用分号连接UT执行和清理脚本，整个bash -c退出码取决于最后一条命令(清理脚本)，当UT失败但清理成功时CMake接收到退出码0误认为UT通过，导致CI无法拦截UT失败；(2) generate_opapi_stub.py仅为当前芯片生成打桩，缺少其他芯片(ascend910b/ascend910_93/ascend910_95)路径下的符号链接。
- 修复模式：(1) 保存UT退出码并在最后返回: OP_API_UT_EXE_RET=$?; cleanup; exit $OP_API_UT_EXE_RET; (2) generate脚本为非当前芯片创建符号链接，clean脚本清理这些链接
- 可审查性：高
- 审查规则建议：CMake add_custom_command的bash -c中含测试执行后接清理命令时，检查是否正确保存并返回了测试退出码。

---

### 90cc4df64 Bugfix: A2 dispatch&combine add SetBlockDim for gentask
- 根因类别：赋值遗漏/结构体字段遗漏(aicpu task缺少SetBlockDim配置)
- 涉及文件：mc2/common/src/mc2_moe_gen_task_ops_utils.cpp
- 缺陷描述：A2平台上MoeDistributeDispatch/Combine及V2版本四种算子的gentask流程中，创建aicpu task时没有调用SetBlockDim设置block维度。缺少此设置导致aicpu任务使用默认配置，在A2平台上造成任务调度异常或性能严重劣化。462行改动中绝大部分是格式重排，实质性改动仅为新增NEED_SET_BLOCK_SET集合和SetBlockDim调用。
- 修复模式：在aicpu task创建后对特定算子类型调用SetBlockDim(默认6，可通过节点属性覆盖)
- 可审查性：中(大量格式重排掩盖真正逻辑改动)
- 审查规则建议：对创建aicpu task的代码路径检查是否根据平台/算子类型正确配置了blockDim；格式化和功能修复应拆分为独立提交。

---

### fbade778b fix experimental depends
- 根因类别：构建配置缺陷(ENABLE_EXPERIMENTAL模式下CMake路径拼接缺少前缀)
- 涉及文件：cmake/func.cmake
- 缺陷描述：两个函数受影响：(1) op_add_depend_directory中EXISTS检查使用不带experimental/前缀的路径，导致依赖存在但检查返回false被跳过；(2) add_bin_compile_target中源码拷贝路径同样缺少前缀。后果是开启ENABLE_EXPERIMENTAL编译时算子间依赖关系无法正确解析，构建失败。关联Issue #427。
- 修复模式：引入depend_info_update变量，在ENABLE_EXPERIMENTAL时统一加experimental/前缀
- 可审查性：高
- 审查规则建议：当编译模式开关影响源码目录结构时，静态检查所有file(EXISTS)、add_subdirectory、路径拼接是否都使用了带开关前缀的路径变量。

## 批次15: 提交 #276-#295 (2025-12-31 ~ 2026-01-14)

---

### c0224370 fix: opensource, fix mlaprolog A5 example
非缺陷提交(示例代码整理)。删除mla_prolog旧版example文件，替换为mla_prolog_v3新示例。示例/文档维护，不涉及production代码逻辑。

---

### 70e5be5f 修复UT批跑编译失败问题
- 根因类别：构建配置缺陷(CMake脚本缺少文件存在性检查)
- 涉及文件：cmake/custom_build.cmake, cmake/func.cmake
- 缺陷描述：执行`bash build.sh -u`批跑UT时，CMake脚本对每个算子目录无条件执行`add_subdirectory(${OP_UT_LIST}/tests)`和`file(READ "${OP_DIR}/tests/CMakeLists.txt" ...)`。当某些算子目录下不存在tests/CMakeLists.txt时，CMake直接报错导致整个编译流程失败。
- 修复模式：在3处关键位置添加`if (EXISTS "${...}/tests/CMakeLists.txt")`守卫检查，不存在时continue()跳过。
- 可审查性：高
- 审查规则建议：CMake中对add_subdirectory或file(READ)调用前，检查是否有路径存在性校验。

---

### b0bf9fa2 fix experimental depend bug
- 根因类别：构建配置缺陷(experimental模式下依赖路径硬编码错误)
- 涉及文件：cmake/func.cmake
- 缺陷描述：op_add_depend_directory函数中，ENABLE_EXPERIMENTAL开启时算子依赖目录应拼接为`${CMAKE_CURRENT_SOURCE_DIR}/experimental/${depend_info}`，但原代码始终使用`${CMAKE_CURRENT_SOURCE_DIR}/${depend_info}`，导致experimental模式下找不到依赖目录。
- 修复模式：添加if(ENABLE_EXPERIMENTAL)分支，根据编译模式选择不同路径前缀。
- 可审查性：高
- 审查规则建议：检查CMake中引用ENABLE_EXPERIMENTAL变量的代码块，是否在路径拼接逻辑中正确区分了experimental和非experimental路径。

---

### 611e9f1f fix spell problem for splitFuse at openSource code
非缺陷提交(标识符拼写修正)。将命名空间FaiKenel全局重命名为FaiKernel，所有引用处一致修正，不改变运行时行为。

---

### 7fb813a9 dispatch combine v4 demo fix
- 根因类别：API误用/接口参数错误(文档示例代码与API签名不匹配)
- 涉及文件：mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV4.md, mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV4.md
- 缺陷描述：V4版本API新增performanceInfoOptional参数，但文档示例代码未传递该参数，同时include头文件名大小写错误。用户复制示例代码编译直接失败。
- 修复模式：补充nullptr参数，修正include头文件名大小写。
- 可审查性：中
- 审查规则建议：API签名变更时自动检测docs/examples目录下的示例代码是否同步更新了函数调用参数。

---

### c3c78777 修改version info
非缺陷提交(版本依赖配置变更)。修改version.info中运行时依赖包列表，正常迭代调整。

---

### 8d72712c fix ring_attention_update ophost ut
- 根因类别：UT测试与production代码不同步
- 涉及文件：attention/ring_attention_update/tests/ut/op_host/test_ring_attention_update_infershape.cpp, attention/ring_attention_update/tests/ut/op_host/test_ring_attention_update_tiling.cpp
- 缺陷描述：两处问题：(1) infershape UT的include头文件名从infershape_context_faker.h更名为infer_shape_context_faker.h，UT未同步导致编译失败；(2) tiling UT中命名空间从Ops::Math::AnyValue改为Ops::Transformer::AnyValue，且期望的tiling输出末尾缺少字段(production代码已新增tiling字段，UT断言未同步)。
- 修复模式：更新include路径、命名空间引用和期望输出字符串。
- 可审查性：中
- 审查规则建议：tiling数据结构新增字段时，检查UT断言是否覆盖；production代码重构后UT引用路径是否同步更新。

---

### 7ee592d1 fix: opensource, fix soc default value
非缺陷提交(文档修正)。补充soc_version参数默认值说明。

---

### 8a72fdf2 关闭cmake默认编译选项；安装校验失败ERROR级别改为WARNING级别
- 根因类别：构建/安装配置缺陷
- 涉及文件：CMakeLists.txt, scripts/package/common/sh/version_compatiable.inc
- 缺陷描述：两个独立问题：(1) CMake Release模式默认注入-O3 -DNDEBUG等选项，与项目自身编译配置冲突；(2) 安装脚本version_compatiable.inc中版本兼容性检查失败时ERROR+return 1导致安装中断，在某些兼容场景下过于严格。
- 修复模式：清空CMake默认Release编译选项；将校验失败从ERROR降为WARNING并移除return 1。
- 可审查性：中
- 审查规则建议：安装脚本中将校验失败从阻断降为警告时，需确认不会导致不兼容环境下的运行时问题。

---

### a740c5c8 fix: opensource, fix fia/ifa warnings
非缺陷提交(编译警告清理)。删除未使用的constexpr变量、删除不存在soc型号的AddConfig注册、重命名参数消除shadowing警告。

---

### aac27c3e fix:add conditional constraints in tnd module
- 根因类别：条件分支遗漏/缺陷(layout模式条件守卫缺失)
- 涉及文件：attention/nsa_selected_attention_infer/op_kernel/nsa_selected_attention_infer.h
- 缺陷描述：curTotalQSeqLenOffset的累加循环在所有layout模式下都会执行，但该偏移量计算仅对TND layout有意义。非TND模式下执行该循环会读取无意义的actualQSeqLengthsGm数据，导致偏移量计算错误，进而造成数据访问异常。
- 修复模式：用`if constexpr (LAYOUT_T == LAYOUT::TND)`包裹偏移量累加循环，编译期排除不适用路径。
- 可审查性：中
- 审查规则建议：当代码存在模板参数区分多种数据布局时，检查所有布局相关偏移计算是否有对应的条件守卫。

---

### 83be205f fix rt_kb
- 根因类别：安装/打包配置缺陷(覆盖安装误删其他组件文件)
- 涉及文件：scripts/package/module/ascend/OpsTransformer.xml
- 缺陷描述：transformer包覆盖安装时，rtkb相关目录的安装配置缺少install_type="all"和正确的install_mod权限设置，且使用了entity="true"属性，导致覆盖安装时知识库目录下其他仓库的文件被误删。
- 修复模式：添加install_type="all"，移除entity="true"，设置正确权限(550/555)。
- 可审查性：低
- 审查规则建议：打包配置中带entity="true"的目录项需确认覆盖安装场景下不会清除其他组件文件。

---

### 34a2a609 fix memory leak
非缺陷提交(空提交)。commit tree与parent完全相同，本仓库无实际文件变更。实际修改可能在关联dev子仓库中。

---

### d8681ad1 fix AttentionToFFN Addr
- 根因类别：GM地址来源/偏移错误(误用remote指针获取local地址)
- 涉及文件：mc2/attention_to_ffn/op_kernel/attention_to_ffn.h
- 缺陷描述：获取本rank地址时，原代码通过winContext_->userMemType判断走两条路径(remoteRes[rankId_].nextDevicePtr或userMemRes[rankId_].addr)，某些场景下取到的地址不正确。正确做法是直接使用winContext_->localWindowsIn获取本rank本地窗口输入地址。
- 修复模式：将条件分支简化为直接取localWindowsIn，消除类型判断导致的地址获取错误。
- 可审查性：高
- 审查规则建议：获取本rank地址时检查是否误用了remote资源指针代替local资源指针；条件分支中获取同一语义资源的不同路径应确保语义一致性。

---

### ab0dceb8 gpt-oss sink doc modify
非缺陷提交(纯文档修改)。在API文档中新增learnableSinkOptional输入参数说明。

---

### a911707e bugfix: ffn example support fp16
- 根因类别：UT测试数据类型错误(host数据类型与aclDataType不匹配)
- 涉及文件：ffn/ffn/docs/aclnnFFN.md, ffn/ffn/docs/aclnnFFNV2.md, ffn/ffn/docs/aclnnFFNV3.md
- 缺陷描述：FFN示例代码声明aclDataType为ACL_FLOAT16，但host端数据容器使用std::vector<float>(4字节)，CreateAclTensor中用sizeof(T)即sizeof(float)=4计算device内存大小而非实际fp16的2字节。导致aclrtMalloc分配2倍内存、aclrtMemcpy拷贝多余数据、结果读取时数据错误。
- 修复模式：将host容器从std::vector<float>改为std::vector<op::fp16_t>；内存大小计算从sizeof(T)改为aclDataTypeSize(dataType)。
- 可审查性：高
- 审查规则建议：CreateAclTensor的dataType参数与模板类型T的sizeof不一致时应报警；示例代码host数据类型应与aclDataType声明严格对应。

---

### b9e99acc fix: opensource, fix pfa warnings
非缺陷提交(编译警告修复)。变量重命名消除shadowing警告，不改变运行时行为。

---

### 181d25f5 fix issue 修复genop生成目录问题
- 根因类别：构建配置缺陷(代码生成工具链不完整)
- 涉及文件：scripts/opgen/opgen_standalone.py, cmake/custom_build.cmake, 及多个template文件
- 缺陷描述：genop脚本创建新算子工程时，若算子分类(op_class)是全新的，脚本只复制算子模板目录但不会生成上层CMakeLists.txt，导致新创建的算子工程无法被CMake发现和编译。此外模板代码本身是空壳无法编译运行。
- 修复模式：在_copy_template方法中增加逻辑——若目标目录上级不存在CMakeLists.txt则从模板复制一份；新增CMakeLists.txt模板文件；补全模板算子代码使其可编译。
- 可审查性：中
- 审查规则建议：代码生成工具在创建新目录结构时，应检查构建系统所需的所有入口文件是否已存在或需同步创建。

---

### 6b029d5f 修复GM偏移量的数据类型为int64
- 根因类别：整数类型错误/溢出(GM偏移用uint32导致大数据量溢出)
- 涉及文件：moe/moe_token_unpermute_with_routing_map/op_kernel/masked_select.h
- 缺陷描述：GM偏移量ind使用uint32_t类型，计算公式为progress*tileLength。当两者都较大时乘积超过uint32最大值(~42亿)，发生整数溢出导致GM地址偏移错误，访问错误内存位置。
- 修复模式：将ind类型从uint32_t改为uint64_t，CopyInData和CopyInMask两处均修改。
- 可审查性：高
- 审查规则建议：GM地址偏移计算变量不应使用32位整型；当偏移量由两个变量相乘得到时，结果类型应足够宽以避免溢出。

---

### 84ca56e9 修复moe_init_routing_v3/moe_re_routing子包编译问题
- 根因类别：API误用(CeilDiv与CeilAlign语义混淆)
- 涉及文件：moe/moe_re_routing/op_host/moe_re_routing_r_tiling.cpp, moe/moe_re_routing/op_host/moe_re_routing_re_tiling.cpp, moe/moe_re_routing/op_kernel/arch35/moe_re_routing_r_regbase.h, moe/moe_re_routing/op_kernel/arch35/moe_re_routing_re_regbase.h, moe/moe_re_routing/tests/ut/op_host/test_moe_re_routing_tiling.cpp
- 缺陷描述：多处混淆CeilDiv和CeilAlign语义。CeilDiv(a,b)=ceil(a/b)是向上取整除法，CeilAlign(a,b)=ceil(a/b)*b是向上对齐。在计算UB buffer大小、InitBuffer的size参数、shift size等需要对齐到block边界的场景中，错误使用CeilDiv(得到block数)而非CeilAlign(得到对齐后字节数)，导致分配buffer远小于实际需要，引发越界或数据错误。UT期望值从4063/4064改为130016/130048反映了修复后buffer大小的数量级变化。
- 修复模式：区分需要"对齐"和"除法"的场景，将应该对齐的地方从CeilDiv改为CeilAlign。
- 可审查性：高
- 审查规则建议：CeilDiv和CeilAlign调用需审查上下文语义——赋给buffer size的应用CeilAlign(对齐)，赋给loop count的应用CeilDiv(除法)；当buffer大小从tiling值直接传入InitBuffer时确认单位一致(字节vs block数)。

## 批次16 (#296-#314)

### 66b12bf2 fix cv1:1 sk
- 根因类别：条件分支遗漏/缺陷 + constexpr误用
- 涉及文件：attention/mla_prolog/op_host/mla_prolog_tiling.cpp, attention/mla_prolog/op_kernel/kernel_mla_prolog_split_n.h, attention/mla_prolog/op_kernel/service_dynamic_quant_qn_mul_qr.h
- 缺陷描述：MlaProlog算子CV1:1模式多个bug：(1)tiling校验逻辑仅在非EMPTY_QUERY路径执行，EMPTY_QUERY路径遗漏校验，且用黑名单方式（列举不支持的cacheMode），修复改为白名单。(2)cvRatio_是static constexpr编译态常量跟随模板参数，但CV1:1编译的kernel运行时可能跑在CV1:2硬件上。修复改为运行时变量通过GetSubBlockNum()获取。(3)两处if constexpr改为if运行时判断。(4)DynamicQuantQnWithMulQr的cvRatio从模板参数改为函数参数。
- 修复模式：编译期常量改运行时变量，增加运行时硬件配置检测；校验提前到公共路径并改用白名单
- 可审查性：中
- 审查规则建议：硬件配置参数（cvRatio等）应优先使用运行时查询而非constexpr；当编译态和运行态可能不一致时，检查所有constexpr/模板参数是否在所有运行场景下恒定

### d2ae9c6b all_gather_add编译修复
- 根因类别：构建配置缺陷 + 资源释放顺序错误
- 涉及文件：cmake/custom_build.cmake, examples/mc2/all_gather_add/op_host/op_api/aclnn_all_gather_add.h, examples/mc2/all_gather_add/examples/test_aclnn_all_gather_add.cpp
- 缺陷描述：(1)CMake构建系统缺少all_gather_add的add_subdirectory入口；(2)include路径缺少aclnnop/子目录前缀致编译找不到头文件；(3)HcclCommDestroy在main函数中aclrtResetDevice之后才调用，此时aclBinaryUnload找不到free内存（use-after-free类）。
- 修复模式：CMake增加构建入口；修正include路径；将HcclCommDestroy移到线程内device资源释放前执行
- 可审查性：高
- 审查规则建议：HCCL通信资源销毁必须在aclrtResetDevice之前；资源释放遵循LIFO顺序；新增算子必须确保CMake入口完整

### d27eab56 修复experimental下的attention目录
- 根因类别：构建配置缺陷
- 涉及文件：cmake/custom_build.cmake, cmake/func.cmake, experimental/attention/CMakeLists.txt
- 缺陷描述：ENABLE_EXPERIMENTAL模式下attention目录构建配置错误：custom_build.cmake无条件add_subdirectory(attention)未切换到experimental路径；func.cmake重复包含experimental/attention的glob与CMakeLists.txt自身遍历冲突；CMakeLists.txt缺少按ASCEND_OP_NAME过滤能力。
- 修复模式：增加ENABLE_EXPERIMENTAL条件分支；删除重复glob；重写CMakeLists支持按算子名过滤
- 可审查性：高
- 审查规则建议：experimental和非experimental路径应互斥且完整；构建目录遍历逻辑应与其他模块保持一致

### 9cff0e48 fix_gmm
- 分类：非缺陷
- 原因：纯文档修改（markdown内容清理和锚点修正）

### 1bc7f8dc fix: update version for FIA/IFA/PFA/MlaProlog readme && delete FIA_vx
- 分类：非缺陷
- 原因：文档更新 + 废弃VX版本算子代码清理

### b11cce04 fix FA compile error for arch32
- 根因类别：赋值遗漏/重命名不同步
- 涉及文件：attention/common/op_host/fia_tiling_info.h, attention/common/op_host/arch32/*.cpp, attention/common/op_kernel/arch32/*.h, attention/fused_infer_attention_score/op_host/arch32/*.cpp/*.h, common/src/tiling_sink/CMakeLists.txt
- 缺陷描述：attenMaskSize字段在主线代码已重命名为attenMaskBatchStride（语义更准确：batch stride而非size），但arch32分支代码未同步更新致编译失败。另tiling_sink的CMakeLists.txt缺少v2版本和arch35的3个源文件。
- 修复模式：全局重命名attenMaskSize→attenMaskBatchStride；CMakeLists补充缺失源文件
- 可审查性：高
- 审查规则建议：重命名字段时全局搜索所有引用点（含不同arch分支）；CI覆盖所有目标架构编译

### 10697b77 fix delete setenv.sh bug
- 根因类别：构建/部署配置缺陷
- 涉及文件：scripts/package/common/sh/script_operator.inc
- 缺陷描述：add_setenv()/del_setenv()函数包含对setenv.sh的完整操作逻辑，但ops-transformer包不应管理setenv.sh，会导致安装/卸载时错误操作setenv.sh。
- 修复模式：将两个函数体清空为return 0
- 可审查性：中
- 审查规则建议：安装/卸载脚本的环境变量管理逻辑需评估对用户环境的影响

### bfbbec83 修改moe_gating_top_k_softmax算子的BlockReduceSum/MaxIntrinsicsImpl接口
- 根因类别：API误用/底层接口误用
- 涉及文件：moe/moe_gating_top_k_softmax/op_kernel/moe_gating_top_k_softmax_perf.h
- 缺陷描述：使用底层intrinsics接口BlockReduceMaxIntrinsicsImpl/BlockReduceSumIntrinsicsImpl（直接操作裸指针__ubuf__ float*），不符合AscendC编程规范，存在类型安全风险。6处调用。
- 修复模式：替换为高层模板接口BlockReduceMax<float,false>/BlockReduceSum<float,false>，使用LocalTensor和MASK_PLACEHOLDER
- 可审查性：高
- 审查规则建议：禁止直接调用*IntrinsicsImpl底层接口；对直接操作__ubuf__指针的代码标记告警

### 32732476 fix nQueryIndex = 8 MTE address out of range bug
- 根因类别：DataCopy对齐越界
- 涉及文件：attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/sparse_lightning_indexer_grad_kl_loss_vector.h
- 缺陷描述：gSizeQueryIndex==8时，DataCopy用AlignTo将长度从8对齐到16，但GM中实际只有8个元素，DMA引擎读取超出范围的后8个元素地址，造成MTE地址越界。
- 修复模式：对gSizeQueryIndex==8使用DataCopyPad替代DataCopy，通过DataCopyExtParams指定实际搬运字节数，DataCopyPadExtParams指定rightPadding=8填充值0
- 可审查性：高
- 审查规则建议：DataCopy搬运长度必须满足对齐要求（通常32字节）；不对齐场景应使用DataCopyPad；对MTE搬运长度参数做边界检查

### 0387ed9a 修复pfa example执行失败 && static脚本不支持3.7
- 根因类别：构建配置缺陷(链接库缺失 + Python版本兼容性)
- 涉及文件：build.sh, scripts/util/build_opp_kernel_static.py
- 缺陷描述：(1)example编译链接命令缺少-lc_sec致链接失败；(2)Path.unlink(missing_ok=True)是Python 3.8+才有的参数，Python 3.7下抛TypeError。
- 修复模式：g++命令添加-lc_sec；替换为try/except FileNotFoundError兼容3.7
- 可审查性：高
- 审查规则建议：新增依赖库引用同步所有编译命令；项目声明最低Python版本并检查新版API兼容性

### 19de45cc [MlaProlog] Fix precision issue when queryNormFlag is set to true
- 根因类别：workspace/buffer偏移计算错误
- 涉及文件：attention/mla_prolog/op_kernel/kernel_mla_prolog_split_n.h
- 缺陷描述：WorkspaceInit中，mmCqResGm_后的workspace偏移逻辑放在queryNormFlag条件判断外无条件执行。当queryNormFlag=true时，rmsNormCq结果直接输出到gm不经workspace，但后续mmQcQrResGm_等buffer仍需跳过mmCqResGm_空间。原代码此偏移仅在dtype不同时才执行，queryNormFlag=true下偏移缺失，导致workspace地址计算错误，后续buffer重叠引发精度问题。
- 修复模式：将偏移逻辑拆分到queryNormFlag==0和else两个分支，else分支无条件偏移mmCqOutputType大小
- 可审查性：中
- 审查规则建议：workspace中多buffer共享连续内存时，必须检查所有条件分支下偏移量计算一致性；review时应验证每个分支下的内存布局确保buffer不重叠

### f8ab5505 revert add_example
- 分类：非缺陷
- 原因：回退之前添加的add_example样例代码，代码整理

### 4138bd36 version.info add required_package info
- 分类：非缺陷
- 原因：版本配置/元信息更新，新增依赖包版本信息

### f05c0242 修复tiling下沉编译失败
- 根因类别：构建配置缺陷
- 涉及文件：common/src/tiling_sink/CMakeLists.txt
- 缺陷描述：CMakeLists.txt的src_files列表引用了4个已不存在的cpp文件（v2版本和arch38的tiling文件），文件在重构中被删除但CMakeLists未同步更新，tiling下沉模式编译失败。
- 修复模式：从src_files中移除4个不存在的源文件引用
- 可审查性：高
- 审查规则建议：删除/重命名源文件时必须同步更新所有CMakeLists.txt；CI应覆盖所有构建配置

### 8ab6568b aclnnFlashAttentionScoreV3部分链接修复
- 分类：非缺陷
- 原因：纯文档链接路径修正（docs/context → docs/zh/context）

### 2ff66c1c docs资料描述错误需要更改
- 分类：非缺陷
- 原因：纯文档修改（补充链接依赖说明、删除重复表格、修正安装路径）

### 4801ecb2 attentionupdate支持lse inf输入
- 根因类别：特殊值(inf/NaN)输入未处理
- 涉及文件：attention/attention_update/op_kernel/decode_update.h
- 缺陷描述：DecodeUpdate算子对lse输入执行max/exp/sum运算。多SP域场景下某些域无有效token，lse为+inf。exp(+inf - +inf) = exp(NaN) = NaN，污染最终输出。原代码未对+inf做特殊处理。
- 修复模式：新增ProcessLseInfReplacement函数，用CompareScalar检测+inf元素，Select替换为-inf。后续max运算中-inf不影响结果，exp(-inf - max)=0而非NaN。同时调整UB内存分配增加selMaskBuffer。
- 可审查性：中
- 审查规则建议：数学运算类算子review时应系统检查exp/log/div/sqrt在边界值(0/+inf/-inf/NaN)下的行为；UT应包含inf/NaN边界测试用例

### d8e8c382 opapi_math 替换opapi.so
- 分类：非缺陷
- 原因：上游依赖库名变更后的全局适配(opapi→opapi_math)，非代码逻辑缺陷

### 6d8abc8d fix:aclnnFusedInferAttentionScoreV4GetWorkspaceSize add learnableSinkOptional
- 分类：非缺陷
- 原因：纯文档修改，API函数原型声明补充遗漏参数

## 批次17: 提交 #315-#334 (2025-12-23 ~ 2025-12-01)

---

### 2b848593 修改version.info版本号
- 分类：非缺陷
- 原因：纯版本号变更(8.5.0.alpha001→8.5.0-beta.1)

---

### e40a4176 修复aclnn接口入参顺序
- 根因类别：API误用/接口参数错误
- 涉及文件：moe/moe_token_unpermute_with_ep_grad/op_host/moe_token_unpermute_with_ep_grad_def.cpp
- 缺陷描述：算子def.cpp中Attr属性注册顺序与aclnn接口参数顺序不匹配。range和topk_num排在padded_mode和restore_shape之前，但aclnn接口要求Attr注册顺序与接口入参顺序严格一致。
- 修复模式：交换Attr声明行顺序，使padded_mode/restore_shape在前，range/topk_num在后。
- 可审查性：中(需对照aclnn接口声明确认正确顺序)
- 审查规则建议：编写lint规则将_def.cpp中Attr注册顺序与对应aclnn接口声明自动比对

---

### 0a278590 修复FlashAttentionScore和FlashAttentionScoreGrad文件中aclnn文档问题
- 分类：非缺陷
- 原因：纯文档修复(12个md文件的参数描述补充)

---

### 88c2d46d 修复gmm和attention文件中aclnn文档问题
- 分类：非缺陷
- 原因：纯文档修复(清理git merge冲突标记残留、删除重复表格)

---

### befd8bcd fix bug for datacopysoftmaxlse bsnd layout
- 根因类别：多处关联缺陷(API误用+条件分支遗漏+功能限制误判)
- 涉及文件：attention/common/op_kernel/arch32/fia_block_vec_nonquant_mla.h, attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_check_feature.cpp, attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_v3.cpp
- 缺陷描述：三个关联问题。(1) BSND布局下调用DataCopySoftmaxLseBSND时多传了2个参数(qActSeqLensParser和info.bIdx)，该函数BSND布局仅需6参数。(2) CheckFeatureMlaNoquantUnsupported中错误阻止了BSND布局下的softmaxLse输出。(3) CheckGqaConstrain和CheckMlaConstrain函数体仅`return false`(占位符代码)，未实现完整的GQA/MLA约束判断逻辑，导致所有GQA/MLA场景均被错误拦截。
- 修复模式：删除多余参数；移除错误的功能限制检查；补全约束判断逻辑(+61行)，新增isNotLegacyGQA辅助函数区分legacy/non-legacy GQA模式。
- 可审查性：低(跨3文件3种修复类型，需深入理解FIA算子tiling策略)
- 审查规则建议：静态分析检测函数体仅含`return false/true`的约束检查函数，标记为可疑占位符代码；函数调用参数数量校验

---

### 881ed56a MlaPrologV3算子拦截问题修复
- 根因类别：多处校验缺陷(sizeof计算+InferShape维度+平台注册+空指针)
- 涉及文件：attention/mla_prolog/op_host/mla_prolog_tiling.cpp, mla_prolog_tiling.h, mla_prolog_tiling_check.cpp, attention/mla_prolog_v3/op_host/mla_prolog_v3_def.cpp, mla_prolog_v3_infershape.cpp, aclnn_mla_prolog_v2_weight_nz.cpp, aclnn_mla_prolog_v3.cpp
- 缺陷描述：(1) sizeof(char_array)-1导致strncmp比较长度不足，前缀相同的opType名称会误匹配。(2) InferShape将可选输出"不使用"标记为dim=1而非dim=0，下游误认为有有效数据。(3) V3算子未注册ascend910b平台，该平台完全无法调用。(4) actualSeqLen在特定cacheMode下可能为空指针直接解引用。(5) CheckTensorNotNullWhen/CheckTensorNullWhen两个独立判断存在逻辑不一致。
- 修复模式：修正sizeof计算、dim=1改dim=0、补全平台注册、加空指针保护、合并互斥校验函数。
- 可审查性：中
- 审查规则建议：sizeof(char_array)-1与strncmp配合检查是否正确处理终止符；算子def.cpp中AddConfig是否覆盖所有目标平台；InferShape中可选输出空语义应用dim=0而非dim=1

---

### e71e9b7d Fix cust example
- 根因类别：构建配置缺陷
- 涉及文件：build.sh
- 缺陷描述：构建example时CUST_LIBRARY_PATH/CUST_INCLUDE_PATH始终使用ASCEND_OPP_PATH下的路径。当用户设置了ASCEND_CUSTOM_OPP_PATH(自定义算子包非默认路径)时，脚本未使用该自定义路径，导致编译链接失败。
- 修复模式：增加ASCEND_CUSTOM_OPP_PATH环境变量判断，存在时优先使用。
- 可审查性：高
- 审查规则建议：涉及ASCEND_OPP_PATH路径拼接时检查是否需要考虑ASCEND_CUSTOM_OPP_PATH覆盖

---

### 88a950a2 修复quick_install描述
- 分类：非缺陷
- 原因：纯文档更新(安装指南md文件)

---

### 615303d6 move nn norm/kv_rms_norm_rope_cache to ops-transformer posembedding
- 分类：非缺陷
- 原因：代码迁移操作(37个新增文件，0删除)

---

### a8ac47bb issue of combineARN docs
- 分类：非缺陷
- 原因：纯文档修复(2个md文件的API参数表格调整)

---

### 1005700c rm version.info required_package
- 分类：非缺陷
- 原因：构建配置清理(移除version.info中内部依赖版本声明)，非实际构建问题修复

---

### 29b08607 修复moe_token_unpermute_grad和moe_token_unpermute_with_routing_map中op_api的UT
- 根因类别：UT测试代码缺陷+构建配置缺陷
- 涉及文件：moe/moe_token_unpermute_grad/op_host/CMakeLists.txt, tests/ut/op_host/op_api/test_aclnn_moe_token_unpermute_grad.cpp, tests/ut/op_kernel/test_moe_token_unpermute_grad.cpp, moe/moe_token_unpermute_with_routing_map/tests/ut/op_host/op_api/(新增CMakeLists.txt+test文件)
- 缺陷描述：(1) UT include路径引用了闭源仓库路径`level2/aclnn_moe_token_unpermute_grad.h`，开源仓库中不存在，UT编译失败。(2) UT通过system()调用拷贝闭源路径下的gen_data脚本，路径不存在导致运行失败。(3) CMakeLists.txt缺少依赖声明，moe_token_unpermute_with_routing_map完全缺少op_api级UT。
- 修复模式：路径改为相对路径；删除闭源路径的硬编码system调用；补充缺失的CMake配置和UT文件。
- 可审查性：高
- 审查规则建议：扫描UT中include路径和system()调用是否引用了仓库外路径

---

### a4e02b39 tp issue
- 根因类别：条件分支遗漏/缺陷
- 涉及文件：mc2/moe_distribute_combine/op_host/op_tiling/moe_distribute_combine_tiling.cpp, mc2/moe_distribute_combine_add_rms_norm同名文件, mc2/moe_distribute_combine_v2同名文件, mc2/moe_distribute_dispatch/op_host/op_tiling/moe_distribute_dispatch_tiling.cpp, mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch.h, mc2/moe_distribute_dispatch_v2同名文件
- 缺陷描述：MoE distribute/combine/dispatch系列算子在tpWorldSize==1(无需TP并行)时仍无条件初始化TP通信组和获取HCCL通信上下文。(1) tiling层无条件对不存在的TP通信组做SetGroupName/SetOpType/GetTiling，可能导致通信初始化异常。(2) kernel层无条件GetHcclContext<1>()，当只有一个通信组时该上下文无效，导致非法内存访问。(3) SetHCommCfg在获取tpWorldSize值之前就被调用，参数依赖未满足。
- 修复模式：给TP操作加`if (tpWorldSize > 1)`守卫；kernel中将GetHcclContext<1>()移入条件块；调整SetHCommCfg调用顺序。
- 可审查性：中
- 审查规则建议：HCCL通信上下文获取必须有worldSize>1守卫；函数参数依赖变量应在调用前完成初始化

---

### c362f74b modify fd invalidrows And fix compile issue
- 根因类别：多处缺陷(编译+运行时逻辑)
- 涉及文件：attention/common/op_kernel/arch32/fia_block_vec_flashdecode.h, fia_block_vec_nonquant.h, fia_block_vec_nonquant_mla.h, fia_kernel_nonquant.h, fia_kernel_nonquant_mla.h, memory_copy.h, vector_common.h
- 缺陷描述：综合性修复包含6类问题。(1) 头文件中`using namespace fa_base_vector`导致命名空间污染，引入新同名符号后编译二义性。(2) 变量在if块内声明但在块外使用，某些编译路径报"未声明标识符"。(3) DealInvalidRowsBelow循环条件`s1RealEnd > 0`漏处理第0行，应为`>= 0`。(4) FlashDecode的invalidRows处理时机错误，应在reduce之后、最终输出之前。(5) fdLseMaxUbBuf/fdLseSumUbBuf单buffer但按cntM%2做ping-pong切换导致数据覆盖，需改双buffer。(6) GetSafeActToken重复定义且mla版本用magic number，提取为公共函数。
- 修复模式：命名空间显式限定；修复变量作用域；循环边界>=0；调整处理时序；双buffer替代单buffer；公共函数提取。
- 可审查性：低(7个文件6种修复混在一个提交)
- 审查规则建议：禁止头文件using namespace；ping-pong双流模式共享buffer必须为双buffer；for循环处理index时注意0边界

---

### af587384 fix moe_token_unpermute_with_routing_map_grad md
- 分类：非缺陷
- 原因：纯文档修复(md文件公式符号修正+错误复制内容删除)

---

### 2adb2526 fix cmake for custom
- 根因类别：构建配置缺陷
- 涉及文件：27个op_host/CMakeLists.txt(gmm/moe/rope等多个算子目录)
- 缺陷描述：add_ops_compile_options被放在if(BUILD_OPS_RTY_KERNEL)分支下，与if(BUILD_OPEN_PROJECT)平级。CUSTOM包编译路径(BUILD_OPEN_PROJECT=true)时kernel编译选项(--cce-auto-sync/-Wno-deprecated-declarations/-Werror)不会被添加，导致CUSTOM包编译失败或行为异常。
- 修复模式：将add_ops_compile_options移入BUILD_OPEN_PROJECT条件块内，27个文件统一修改。
- 可审查性：中(单文件改动简单但需逐一确认27个文件一致性)
- 审查规则建议：add_ops_compile_options应在预期的编译目标条件作用域内

---

### 578ed128 fix: deqScale2 dataType name
- 分类：非缺陷
- 原因：纯文档拼写修正(FLOAT32r→FLOAT32)

---

### 054c31af nsacompressattentioninfer算子tiling侧actualQSeqLengths校验修复
- 根因类别：输入校验缺失(可选tensor判空逻辑错误)
- 涉及文件：attention/nsa_compress_attention_infer/op_host/nsa_compress_attention_infer_tiling.cpp
- 缺陷描述：SplitBN()和GetMaxQSeqlen()中用`actualQSeqLengths.tensor != nullptr`判断可选参数是否传入，但tensor对象可能存在(指针非null)但内部数据为空。应检查`tensor->GetData<int64_t>() != nullptr`。错误判断导致读取空数据，tiling计算错误。
- 修复模式：将判空从"检查tensor指针"改为"检查tensor内部数据指针"，2处同模式修改。
- 可审查性：高
- 审查规则建议：可选tensor参数判空应统一使用GetData()返回值；静态分析对`.tensor != nullptr`在可选参数上下文中发出警告

---

### 9f5283f9 fix ut and permute indices size
- 根因类别：workspace/buffer大小计算错误
- 涉及文件：moe/moe_token_permute/op_host/moe_token_permute_tiling.cpp, moe/moe_token_permute_with_ep/op_host/moe_token_permute_with_ep_tiling.cpp及其op_kernel/moe_index_copy.h和moe_index_copy_spilt_d.h, moe/moe_token_permute_with_routing_map/op_host/moe_token_permute_with_routing_map_tiling.cpp
- 缺陷描述：indices buffer大小计算时`UpAlign(onceIndices, ONE_BLOCK_BYTE)`按元素数量而非字节数对齐，应为`UpAlign(onceIndices * INT32_DTYPE_SIZE, ONE_BLOCK_BYTE)`。导致分配的字节数只有实际需要的1/4。kernel侧原来用`indicesUB * 4`做补偿，tiling/kernel间buffer大小语义不一致。另外moe_token_permute_with_routing_map中topK校验缺少paddedMode前置条件导致误报。
- 修复模式：tiling侧乘以INT32_DTYPE_SIZE统一按字节计算；kernel侧去掉补偿乘法；增加paddedMode前置条件。
- 可审查性：中
- 审查规则建议：UpAlign()第一个参数应始终为字节数；tiling和kernel之间buffer大小语义需保持一致，避免一侧元素另一侧字节的补偿模式

---

### 62ff8d25 修复首页md目录问题
- 分类：非缺陷
- 原因：纯文档更新(README目录树+开发指南md)

## 批次18: 提交 #335-#354 (2025-11-26 ~ 2025-11-10)

---

### 942d5acd 128P问题gmmalltoallv修复
- 根因类别：校验条件范围错误(硬编码白名单不完整)
- 涉及文件：mc2/grouped_mat_mul_allto_allv/op_host/op_tiling/grouped_mat_mul_allto_allv_tiling.cpp
- 缺陷描述：epWorldSize合法值白名单{8,16,32,64}缺少128，导致128P(128个EP节点)场景下tiling校验返回false，算子无法执行。
- 修复模式：扩展白名单枚举值加入128，同步更新错误日志文本。
- 可审查性：高
- 审查规则建议：硬编码白名单/枚举值校验应与上游规格文档一致；std::find模式的校验分支需验证白名单完备性

---

### eb8dda97 fix moe_init_routing md
- 分类：非缺陷
- 原因：纯文档格式修改(README.md表格布局调整)

---

### a46890ac fix moe_init_routing_quant md
- 分类：非缺陷
- 原因：纯文档格式修改(README.md表格布局调整)

---

### 64be6dd6 nsa_compress_attention_infer fix example by fixing mem.h InitBuffer
- 根因类别：赋值遗漏/结构体字段遗漏(复制粘贴错误)
- 涉及文件：common/include/kernel/mem.h
- 缺陷描述：AsdopsBuffer构造函数中初始化tensor数组时，L0A/L0B/L0C三种buffer类型的数组索引全部错误地写成了ASCEND_CB。连续3行`tensor[(uint32_t)BufferType::ASCEND_CB] = ...`不断覆盖同一个slot(tensor[1])，而tensor[2](L0A)、tensor[3](L0B)、tensor[4](L0C)从未被初始化。后续通过GetBuffer<ASCEND_L0A>等访问时得到未初始化tensor。两个条件编译分支各3处，共6处相同错误。
- 修复模式：将数组索引修正为对应的BufferType枚举(ASCEND_L0A/ASCEND_L0B/ASCEND_L0C)。
- 可审查性：高
- 审查规则建议：静态分析检测"同一数组相同索引被连续赋值多次"的模式，几乎总是复制粘贴遗漏修改的bug

---

### 0b12dd13 fix moe_gating_top_k_softmax_v2 md
- 分类：非缺陷
- 原因：纯文档修改(README.md参数分类和格式修正)

---

### cb655e0a fix moe_compute_expert_tokens md
- 分类：非缺陷
- 原因：纯文档修改(README.md表格格式调整)

---

### ce60c7d8 fix moe_finalize_routing_v2_grad md
- 分类：非缺陷
- 原因：纯文档修改(README.md表格样式+参数分类修正)

---

### d2e6b827 fix ut bug & support infer_datatype/infer_shaperange
- 根因类别：多处缺陷(UT框架+构建系统+API适配)
- 涉及文件：build.sh, cmake/ut.cmake, cmake/obj_func.cmake, common/stub/op_api/CMakeLists.txt, tests/ut/framework_normal/下多个文件，共32个文件
- 缺陷描述：包含多类修复。(1) build.sh中变量名拼写错误`UT_TARGES`应为`UT_TARGETS`(4处)，导致UT构建目标无法追加，UT不会被编译执行。(2) cmake/ut.cmake中`${CMAKE_CURRENT_SOURCE_DIR}`误用应为`${MODULE_DIR}`(多处)，路径解析错误。(3) cmake/ut.cmake中空字符串比较`""`应为`"ALL"`(5处)，--ops过滤逻辑失效。(4) UT框架中EXPECT_EQ被注释掉，infershape测试校验被跳过。(5) AscendString未加ge::命名空间限定。(6) op_kernel的tiling so路径硬编码错误。(7) 头文件guard从OPS_MATH_DEV改为OPS_TRANSFORMER_DEV(仓库复制遗留)。
- 修复模式：变量名拼写修正+路径修正+条件逻辑修正+取消注释断言+命名空间限定+API适配。
- 可审查性：低(32个文件，混合bug修复/API适配/新功能，未拆分)
- 审查规则建议：禁止提交被注释的EXPECT_EQ/ASSERT_EQ测试断言；shell变量名引用一致性检查；头文件guard应与当前仓库名一致

---

### 6e16f3dc nsa_selected_attention_infer fix example
- 根因类别：变量名typo(头文件名拼写错误)
- 涉及文件：attention/nsa_selected_attention_infer/examples/test_aclnn_nsa_selected_attention_infer.cpp
- 缺陷描述：include头文件名`aclnn_nsa_select_attention_infer.h`缺少`ed`后缀，应为`aclnn_nsa_selected_attention_infer.h`，导致example编译失败。
- 修复模式：修正头文件名拼写。
- 可审查性：高
- 审查规则建议：example代码提交应有编译验证CI gate

---

### 8b73026b fix example
- 根因类别：构建配置缺陷
- 涉及文件：build.sh
- 缺陷描述：build_example()和build_example_group_eager()中，-I ${EAGER_INCLUDE_OPP_ACLNNOP_PATH}头文件搜索路径原本只在MC2条件分支添加，但所有算子example编译都需要该路径。非MC2算子example编译时找不到aclnnop头文件报错。
- 修复模式：将-I路径从MC2条件分支移到通用g++编译命令行。
- 可审查性：高
- 审查规则建议：build脚本中通用头文件路径不应放入条件分支

---

### c29bc106 nsa_select_attention_infer: rename
- 分类：非缺陷
- 原因：纯重命名重构(select→selected批量替换)

---

### 47926d5a fix gmm_add_clean_code
- 根因类别：计算逻辑错误(L1 partA/B值与使用关系反转)
- 涉及文件：gmm/grouped_matmul_add/op_host/grouped_matmul_add_tiling.cpp
- 缺陷描述：原代码BEST_L1_PARTA=256K/PARTB=128K，但mmStepKa用PARTB计算、mmStepKb用PARTA计算，partA/B的值分配和使用关系是反的。修复交换值+纠正使用关系(stepKa用partA, stepKb用partB)。该缺陷被CleanCode标签掩盖，实际改变了stepKa和stepKb的计算结果。
- 修复模式：常量值纠正+变量使用关系纠正+显式类型转换增强。
- 可审查性：低(被CleanCode修改掩盖)
- 审查规则建议：常量定义值的变更应作为高优先级审查项；常量值和使用位置同时修改时需验证净效果

---

### 12461973 修复rope_with_sin_cos_cache example
- 根因类别：构建配置缺陷(头文件路径错误)
- 涉及文件：posembedding/rope_with_sin_cos_cache/examples/test_aclnn_rope_with_sin_cos_cache.cpp
- 缺陷描述：include路径`aclnnop/level2/aclnn_rope_with_sin_cos_cache.h`多了一层level2/子目录，应为`aclnnop/aclnn_rope_with_sin_cos_cache.h`。导致example编译失败。
- 修复模式：修正include路径。
- 可审查性：高
- 审查规则建议：include路径应与实际SDK头文件安装路径自动一致性校验

---

### 807caa4c NpuOpsTransformerExt 支持
- 分类：非缺陷
- 原因：纯新feature(新增experimental/npu_ops_transformer_ext工程模板)

---

### 4f5aeb89 mc2 ut kernel fix
- 根因类别：UT测试代码缺陷+构建配置缺陷
- 涉及文件：mc2目录下33个文件(all_gather_matmul, allto_all_all_gather_batch_mat_mul, matmul_all_reduce等多个算子)
- 缺陷描述：(1) 算子kernel中的相对include路径在UT构建目录下找不到头文件，通过__has_include宏增加备选路径fallback。(2) UT中tiling结构体名(如Mc2BatchMatmulTilingData)与主代码结构体名(BatchMatmulTilingData)不一致，UT编译失败。(3) 多个kernel入口函数在__CCE_KT_TEST__宏下缺少REGISTER_TILING_DEFAULT注册，UT运行时无法获取tiling数据。
- 修复模式：条件编译路径兼容+结构体命名同步+UT基础设施补全。
- 可审查性：中(33个文件但模式高度重复)
- 审查规则建议：UT中手写tiling_def.h应与主代码tiling结构体自动同步

---

### 7b44fa41 冒烟功能的看护优化
- 分类：非缺陷
- 原因：CI/CD测试看护优化(新增PR变更文件解析和自动关联UT/example)

---

### e96b7b2a 修复PFA的非32B对齐情况下的精度问题
- 根因类别：整数类型错误/溢出(硬件指令repeat参数溢出uint16)
- 涉及文件：attention/prompt_flash_attention/op_kernel/prompt_flash_attention_s1s2_bns1_x310_base.h
- 缺陷描述：CopyND2NZOnThe函数中对目标buffer做零初始化时，Duplicate的repeat参数=calcWidth*calcHeightAlign可能超过255(硬件InitConstValue的repeatTime字段上限为uint16)。当Query_seqlen在69~80之间且未16对齐时触发溢出，初始化不完整，buffer残留脏数据引发精度错误。
- 修复模式：将单次Duplicate改为循环调用InitConstValue，每次最多处理255个repeat(MAX_REPEATS_PER_BATCH)。
- 可审查性：中
- 审查规则建议：硬件intrinsic的repeat/repeatTime参数来源于运行时计算时，应检查是否可能超过255/65535上限，强制要求分批或加断言

---

### 55adcbeb 移除冗余输出
- 分类：非缺陷
- 原因：删除一行调试用std::cout输出，纯代码清理

---

### 24d5f4bb 回黄和builtin add_subdirectory统一，以attention为例
- 分类：非缺陷
- 原因：构建系统(CMake)重构统一，非运行时代码缺陷修复

---

### b4aada9f fix sequential bug in gqa antiquant kvnz
- 根因类别：硬件流水线同步缺失
- 涉及文件：attention/incre_flash_attention/op_kernel/incre_flash_attention_preload_dd.h
- 缺陷描述：DealAntiqBmm2ResBaseBlock函数中，DataCopy(bmm2ResPreUb, vec2ResUb, ...)读取Vector流水线的输出buffer，但之前没有PipeBarrier<PIPE_V>()。Vector写操作可能尚未完成，DataCopy读取到不完整/旧数据。GQA antiquant KV NZ场景下的RAW数据竞争。
- 修复模式：在DataCopy前插入PipeBarrier<PIPE_V>()。
- 可审查性：低
- 审查规则建议：DataCopy源操作数来自Vector计算结果时，必须确保前方存在PipeBarrier<PIPE_V>()

## 批次19: 提交 #355-#374 (2025-11-08 ~ 2025-10-31)

---

### 23b7944c support more ranknum and fix aivnum
- 根因类别：硬编码常量导致越界+调度模式配置错误
- 涉及文件：mc2/elastic_receivable_test/op_host/op_tiling/elastic_receivable_test_tiling.cpp, mc2/elastic_receivable_test/op_kernel/elastic_receivable_test.h
- 缺陷描述：(1) AIV_NUM_USED硬编码为6，但实际应为1(每个核独立处理)，导致sendRankNum_=rankNum_/aivNum_除法结果错误，任务分配不正确。(2) SetScheduleMode(1)(batch mode)应为SetScheduleMode(0)(normal mode)，调度模式不匹配算子执行场景。
- 修复模式：修正常量值6→1；修正调度模式1→0。
- 可审查性：中
- 审查规则建议：AIV_NUM_USED等硬编码核数常量应有取值依据注释；SetScheduleMode取值应与算子核间同步语义匹配

---

### c8f181eb revert//schedule
- 根因类别：前序PR引入回归错误(Revert)
- 涉及文件：attention/common/op_host/arch32/fia_tiling_nonquant_mla.cpp, fia_tiling_nonquant_mla.h, fia_tiling_base.h
- 缺陷描述：撤销MR!231(commit ff21f09b)引入的CalcScheduleMode功能。原始提交方法定义为FiaTilingNonQuant::CalcScheduleMode()但类声明在FiaTilingNonQuantMla中——类名不匹配，导致方法绑定到错误的类。
- 修复模式：Revert整个有缺陷的提交。
- 可审查性：高
- 审查规则建议：方法定义的类名与声明所在类必须一致

---

### 2090000e fix:修复mla_prolog_v3 资料 & exampe 执行报错
- 分类：非缺陷
- 原因：文档和example代码的数据类型/参数修正(BF16→INT8等)，不涉及算子运行时代码

---

### f66e305f fix md jump link
- 分类：非缺陷
- 原因：纯文档修改(22个md文件的相对路径跳转链接修正)

---

### ea47405a elastic_receivable_info_collect功能
- 分类：非缺陷
- 原因：纯新feature(全部19个文件均为新建，+2071行)

---

### 3037d282 修正mc2下部分example缺失打印信息的问题
- 分类：非缺陷
- 原因：example示例代码完善(补充打印信息、_exit改return等)

---

### 4fd31731 add:nsa_select_attention_infer_example
- 分类：非缺陷
- 原因：新增example调用示例

---

### 79cd3244 bugfix:nsa_select_infer_ut
- 根因类别：UT测试代码缺陷+构建配置缺陷
- 涉及文件：attention/nsa_select_attention_infer/tests/CMakeLists.txt, tests/utest/ts_nsa_select_attention_infer.h
- 缺陷描述：(1) CMakeLists.txt中file(GLOB_RECURSE)路径缺少算子目录前缀，找不到测试源文件。(2) message()和file()末尾多余反斜杠续行符导致CMake解析异常。(3) 测试类继承Ts_WithParam时缺少模板参数`<NsaSelectAttentionInferCase>`，编译失败。
- 修复模式：修正路径+删除多余续行符+补全模板参数。
- 可审查性：高
- 审查规则建议：CMake GLOB路径应包含完整算子目录前缀；模板类继承必须指定模板参数

---

### c05e6505 Barrier支持故障检测新增elasticInfo和timeOut可选输入
- 分类：非缺陷
- 原因：纯新feature(为DistributeBarrier算子新增可选输入)

---

### a7010fc9 fix ut for moe_init_routing_v3, moe_re_routing, moe_gating_top_k, interleave_rope
- 根因类别：多处生产代码缺陷(空指针校验对象写错+参数传递错误+API误用)
- 涉及文件：moe/moe_gating_top_k/op_host/moe_gating_top_k_tiling.cpp, moe_gating_top_k_tiling_arch35.cpp, moe_gating_top_k_proto.h, moe_gating_top_k_tiling.h
- 缺陷描述：虽标题为"fix ut"，但同时修了多个生产代码bug。(1) 获取expertIdxPtr后却校验了yShapePtr(copy-paste错误)，expertIdxPtr为空时不被捕获，后续解引用崩溃。(2) OP_LOGE的printf参数顺序与格式串不匹配(groupSelectMode_和perGroupExpertCount_传反)。(3) bias是optional输入却用了GetInputShape而非GetOptionalInputShape，未提供bias时异常。(4) 浮点字面量1e-20应为1e-20f(Float属性类型不匹配)。
- 修复模式：修正校验对象名+调换printf参数顺序+改用GetOptionalInputShape+加f后缀。
- 可审查性：高
- 审查规则建议：OP_CHECK_NULL校验对象应与上一行获取的指针变量名一致；optional输入必须用GetOptionalInputShape

---

### fc29fc0c fix:nsa_select_attention_infer_ut_example
- 分类：非缺陷
- 原因：UT框架搭建+example补充

---

### 0941b78c 修复GMMFR算子文档中变量名错误以及描述不通顺的问题
- 分类：非缺陷
- 原因：纯文档修改(5个md文件的变量名和描述修正)

---

### e03da471 [bugfix]NsaCompressAttentionWithCache UT 整改
- 分类：非缺陷
- 原因：UT工程结构整改(目录重组+CMake重写)

---

### 61ca689a nsa_compress_attention_infer fix ut
- 分类：非缺陷
- 原因：UT结构整改+tiling UT补充

---

### 644978ae nsa_compress_attention_infer and nsa_compress_with_cache example fix
- 分类：非缺陷
- 原因：example补全(新增2个example文件+文档)

---

### b9a02e9d fix rope_with_sin_cos_cache ut
- 根因类别：UT测试数据类型错误
- 涉及文件：posembedding/rope_with_sin_cos_cache/tests/ut/op_kernel/gen_data.py, test_case_fp32.cpp, test_case_fp16.cpp, op_host/test_rope_with_sin_cos_cache_infershape.cpp
- 缺陷描述：多处UT数据类型错误。(1) gen_data.py中position数据用传入的dtype(float)而非np.int64，位置索引必须是整数。(2) int(numQHeads * headSize)在字符串参数时做字符串重复而非数值乘法。(3) test_case_fp32中所有buffer size用sizeof(half)而非sizeof(float)，内存分配只有实际需要的一半，越界读写。(4) fp32/fp16测试都传了'bf16' dtype给gen_data.py。(5) infershape头文件名不匹配导致编译失败。
- 修复模式：修正数据类型+修正sizeof+修正dtype参数+修正头文件名。
- 可审查性：高
- 审查规则建议：UT中sizeof类型必须与测试数据类型一致；gen_data脚本的dtype参数应与测试用例对应

---

### 55ea73d2 fix_rope_quant_kvcache_ut
- 分类：非缺陷
- 原因：UT补充(取消注释启用tiling UT编译+新增tiling测试用例)

---

### 68b44c1e fix_moe_token_permute_with_routing_map_grad_ut
- 分类：非缺陷
- 原因：UT补充(取消注释启用UT编译+格式化)

---

### 1e927dbf [Bug-Report|缺陷反馈]: PromptFlashAttention缺少kernel部分UT用例
- 分类：非缺陷
- 原因：UT补充+CMake构建脚本重构(新增tiling/kernel/opapi测试用例)

---

### 8bfb5c84 修复算子的kernel_ut：moe_token_unpermute_grad等
- 根因类别：UT测试代码缺陷(路径依赖+类型错误)
- 涉及文件：moe/moe_token_unpermute_grad/tests/ut/op_kernel/, moe/moe_token_unpermute_with_routing_map/tests/ut/op_kernel/, moe/moe_token_unpermute_with_ep_grad/tests/ut/op_kernel/, common/include/kernel/masked_select.h
- 缺陷描述：(1) kernel UT通过system()调用依赖外部路径`ops/built-in/tests/ut/fast_op_test/`下的Python脚本生成测试数据，独立构建环境下路径不存在导致UT异常。(2) masked_select.h中模板类型约束错误(uint16_t应为half，int8_t应为bool)导致编译问题。
- 修复模式：移除外部路径依赖，改为C++内联tiling数据；修正模板类型。
- 可审查性：高
- 审查规则建议：UT不应依赖外部仓库绝对路径；模板特化的类型应与实际使用场景匹配

## 批次20: 提交 #375-#394 (2025-10-31 ~ 2025-10-23)

---

### 7bf5bee CMAKE_BUILD_MODE 调试
- 分类：非缺陷
- 原因：构建系统功能增强(debug/优化参数传递方式改进)

---

### 6aa4c33 解决magicValue整数反转问题
- 根因类别：整数类型错误/溢出
- 涉及文件：mc2/moe_distribute_combine/op_kernel/moe_distribute_combine_a2_layered.h, mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch_a2_layered.h, mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch_a2_layered_aicpu.h
- 缺陷描述：magicValue用于标识Dispatch/Combine算子调用轮次，每次递增1。类型为int32_t，超过INT32_MAX(约21亿次)后有符号溢出，值变负。下游通过magicValue+常量(12345/123/321)作为IPC同步标志区分轮次，溢出后标志值错乱导致核间同步失败，大模型推理输出胡言乱语或为空。
- 修复模式：类型扩宽int32→uint64；有符号改无符号；提取魔术数字为命名常量(GM2IPC_SYNC_FLAG等)。
- 可审查性：中
- 审查规则建议：持续递增的计数器变量应检查类型是否覆盖最大运行周期；IPC/核间同步标志应使用无符号类型

---

### 8dc40bd 修正example编译问题
- 分类：非缺陷
- 原因：文档和example代码修正(注释文件名+CMake示例补充链接库)

---

### 5cd9e41 fix UT of moeinitrouting moetokenpermute 系列算子
- 分类：非缺陷
- 原因：UT适配(更新期望tiling数据+修正路径+更新参数以匹配算子实现变化)

---

### 233466d bugfix_for_gmm_gragh
- 根因类别：边界条件/tiling约束(动态shape未处理)
- 涉及文件：gmm/grouped_matmul/op_host/grouped_matmul_infershape.cpp
- 缺陷描述：GetDim0函数循环累加每个tensor的第0维大小，当shape包含-1(动态/未知维度)时直接参与累加，结果变成无意义负数(多个-1累加得-3)，后续shape推断全部出错。
- 修复模式：遇到任何tensor的dim0为负值时立即返回该负值(传播-1表示"未知")，不再继续累加。
- 可审查性：高
- 审查规则建议：infershape中对维度值的算术操作前应检查是否为-1(动态shape标记)

---

### 8ef7993 修复ut报错
- 根因类别：UT测试代码缺陷(include路径+tiling未初始化+API签名+期望值错误)
- 涉及文件：moe/moe_finalize_routing_v2_grad/tests/ut/(op_host/op_kernel多文件), moe/moe_token_unpermute_with_ep/tests/ut/(多文件), moe/moe_compute_expert_tokens/tests/(新增)
- 缺陷描述：(1) include路径错误导致编译失败。(2) system()调用外部不存在路径。(3) infershape测试NodeOutputTd多传DType参数，API签名错误。(4) kernel测试tiling数据结构未初始化(GmAlloc后无赋值)，读取随机值crash。(5) tiling参数值错误(hidden从262144改为2621)。
- 修复模式：修正路径+删除外部依赖+修正API签名+补充tiling初始化+更新期望值。
- 可审查性：中
- 审查规则建议：GmAlloc后必须显式初始化所有字段；UT不应system()调用外部路径

---

### f1a24bd fix moe_token_unpermute_with_routing_map_grad ut
- 根因类别：UT测试代码缺陷(tiling数据未初始化+GET_TILING_DATA宏错误)
- 涉及文件：moe/moe_token_unpermute_with_routing_map_grad/tests/ut/op_kernel/test文件及头文件
- 缺陷描述：kernel UT通过GmAlloc分配tiling内存后未初始化任何字段，kernel读取随机值行为异常。GET_TILING_DATA宏使用memcpy从uint8_t*拷贝到结构体，存在对齐问题。
- 修复模式：补充21个字段显式初始化；重写宏为__ubuf__指针逐字段赋值。
- 可审查性：中
- 审查规则建议：GET_TILING_DATA宏应统一使用标准模板；tiling内存分配后所有字段必须初始化

---

### 5644a7b fix precision error under super-kernel
- 根因类别：硬件流水线同步缺失(同步ID冲突)
- 涉及文件：attention/mla_prolog/op_kernel/mla_prolog_comm.h
- 缺陷描述：super-kernel场景下MlaProlog算子的同步标志(sync flag)常量值(0x0/0x1)与前序算子同步ID冲突，导致错误同步行为引发精度问题。
- 修复模式：将所有同步常量值提升到0x6/0x7/0x8范围，避免ID冲突。
- 可审查性：低(需了解super-kernel场景所有算子的同步ID分配)
- 审查规则建议：同步ID分配应有全局统一管理机制而非各算子头文件硬编码

---

### b8cad11 PFA、IFA冗余模板删除
- 分类：非缺陷
- 原因：死代码清理(删除4个不可达的.h模板文件)

---

### 601bce1 fix_gmm_add_ut
- 分类：非缺陷
- 原因：UT适配(删除对外部路径的system调用依赖)

---

### b22f886 fix op-host cmake for [interleave_rope/moe_init_routing_v3/moe_re_routing]
- 根因类别：构建配置缺陷
- 涉及文件：posembedding/interleave_rope/op_host/CMakeLists.txt, moe/moe_init_routing_v3/op_host/CMakeLists.txt, moe/moe_re_routing/op_host/CMakeLists.txt
- 缺陷描述：add_modules_sources和add_ops_compile_options在if/else嵌套中被互斥化。"回黄kernel"模式只加编译选项不注册模块源码，"custom"模式反之。特定构建配置组合下编译失败或链接错误。
- 修复模式：提升add_ops_compile_options到BUILD_OPEN_PROJECT内无条件执行；add_modules_sources改为非BUILD_OPS_RTY_KERNEL时独立执行。
- 可审查性：中
- 审查规则建议：CMake条件分支中add_modules_sources和add_ops_compile_options不应互斥

---

### e677067 PFA IFA fix example
- 分类：非缺陷
- 原因：example代码改进(_exit改return)

---

### 4fd755b FIA IFA PFA tilingkey revert
- 根因类别：前序PR引入回归错误(Revert)
- 涉及文件：attention/fused_infer_attention_score/多文件, attention/incre_flash_attention/多文件, attention/prompt_flash_attention/多文件
- 缺陷描述：MR!99的tilingkey模板化整改引入大量模板化tilingkey头文件(4286行+585行+1662行)，导致编译时间爆炸，CI编译超时阻塞线上流水线。
- 修复模式：Revert恢复手动tilingkey计算方式。
- 可审查性：中
- 审查规则建议：模板化代码变更应评估编译时间影响；CI应有编译超时基线监控

---

### d6dd883 tilingkey revert
- 分类：非缺陷
- 原因：空提交(无文件变更，merge master操作)

---

### 6866957 解决部分算子编译告警问题
- 根因类别：参数传递错误(printf格式化说明符不匹配)
- 涉及文件：moe/moe_finalize_routing_v2/op_host/moe_finalize_routing_v2_tiling_arch35.cpp, posembedding/dequant_rope_quant_kvcache/op_host/dequant_rope_quant_kvcache_tiling.cpp, moe/moe_init_routing_v2_grad/op_host/moe_init_routing_v2_grad_tiling.cpp等
- 缺陷描述：多处OP_LOGE/OP_LOGD的printf格式化说明符与参数类型不匹配(%lu用于int64_t, %ld用于uint64_t, %zu用于int64_t等)。在C/C++标准层面构成未定义行为，错误路径触发时输出错误诊断信息。
- 修复模式：修正格式化说明符与参数类型一致。
- 可审查性：高
- 审查规则建议：CI加入-Wformat-security或同等静态分析

---

### b9d7361 修复GMM A8W4 tiling key路由错误
- 根因类别：条件分支遗漏/缺陷(tiling key路由优先级错误)
- 涉及文件：gmm/grouped_matmul/op_host/grouped_matmul_tiling.cpp, grouped_matmul_tiling.h
- 缺陷描述：isA8W4FakeA8W8_的tiling key判断被放在isA8W8_分支之后，且isA8W8_标志被isA8W4FakeA8W8_污染(通过`||`合并)。A8W4请求永远走进A8W8分支，被错误路由到A8W8的tiling key，无法到达TILING_KEY_A8W4_FAKE_A8W8。
- 修复模式：将isA8W4FakeA8W8_判断提到isA8W8_之前(优先匹配更特化路径)；去掉isA8W8_中的`||isA8W4FakeA8W8_`让两个标志独立。
- 可审查性：中
- 审查规则建议：多个互斥条件分支应先匹配更特化路径；布尔标志不应通过||合并不同语义

---

### a8c22c3 新增处理keyAntiquantScale和valueAntiquantScale输入(B, S)的情况
- 根因类别：条件分支遗漏/缺陷(输入shape维度处理遗漏)
- 涉及文件：attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp
- 缺陷描述：GetAntiquantSeqLength()中antiquantSIdx只考虑了gqaKvNZFlag_一种情况。当kvAntiParamSplitFlag_=true且antiquantScale tensor为2维(B,S)时，序列长度维度index应为1，但代码默认走到index=2。2维tensor没有index=2，越界访问或返回错误值。
- 修复模式：条件中增加2维shape判断，index=1。
- 可审查性：中
- 审查规则建议：对tensor维度索引访问应校验dimNum是否足够

---

### 4251efa 修复tnd模板gs1合轴精度问题
- 根因类别：计算逻辑错误(向量乘法repeat长度缺乘数因子)
- 涉及文件：attention/prompt_flash_attention/op_kernel/prompt_flash_attention_s1s2_bns1_mla_baseapi.h
- 缺陷描述：MLA kernel中bmm2结果与softmax临时结果做Mul时，repeat count参数应为`vec2S1RealSize * gBaseSize`，但缺少`* gBaseSize`。tnd模板gs1合轴场景下group维度合并到数据维度，缺少乘数导致只处理了部分数据，精度错误。
- 修复模式：Mul的repeat count参数增加`* extraInfo.gBaseSize`。
- 可审查性：中
- 审查规则建议：合轴场景的向量操作repeat/长度参数应包含合轴维度因子

---

### b5ce77b PFA、IFA、FIA - fix warning
- 分类：非缺陷
- 原因：编译告警消除+防御性null检查(代码健壮性改进)

---

### 4aee01d 修正编译问题
- 根因类别：构建配置缺陷(路径依赖+cmake宏调用错误)
- 涉及文件：attention/mla_prolog_v2/op_host/CMakeLists.txt, attention/mla_prolog_v2/op_kernel/mla_prolog_v2.cpp, cmake/obj_func.cmake, cmake/custom_build.cmake
- 缺陷描述：(1) mla_prolog_v2的CMake依赖变量在错误的条件分支设置，开源构建时变量未设置。(2) include路径在不同构建模式下不同导致编译失败。(3) add_library直接调用缺少add_opapi_modules()宏内的必要初始化。(4) 依赖算子kernel文件未安装到目标目录。
- 修复模式：修正条件分支+__has_include条件编译+替换为正确的cmake宏+新增依赖安装逻辑。
- 可审查性：中
- 审查规则建议：跨算子依赖需在cmake/custom_build中同步安装

## 批次21: 提交 #395-#416 (2025-10-22 ~ 2025-09-30)

---

### 98579637 fix warn
- 分类：非缺陷
- 原因：参数名shadow告警消除(doubleBufferFlag→doubleBufferFlagLocal)

---

### 4d24f605 fix moeinitroutingv2 310p
- 根因类别：硬件平台兼容性缺陷(内存对齐+workspace偏移+废弃API)
- 涉及文件：moe/moe_init_routing_v2/op_kernel/moe_v2_sort_multi_core.h, moe_v2_sort_one_core.h
- 缺陷描述：310P平台三类问题。(1) InitGlobalMemory传入长度未按sizeof(int32_t)对齐，310P硬件要求内存操作对齐。(2) workspace中expert索引偏移量使用sortTotalLength计算，与310P上实际每核分配元素数perCoreElements不一致，多核间workspace地址重叠或越界。(3) pipe_barrier(PIPE_ALL)是310P废弃API，需替换为PipeBarrier<PIPE_ALL>()。
- 修复模式：条件编译增加310P平台专用逻辑(Align对齐+独立perCoreOffset+替换API)。
- 可审查性：低
- 审查规则建议：InitGlobalMemory长度参数应经过对齐处理；workspace多核偏移应与tiling的每核元素数一致

---

### c3933b09 weightMatmulAllreduceFixed
- 根因类别：硬件流水线同步缺失(barrier位置不正确)
- 涉及文件：mc2/matmul_all_reduce/op_kernel/common.h
- 缺陷描述：CastBFtoFloatOnAiv0Impl函数末尾做MTE3_V事件同步，但真正需要同步的是循环迭代间——前一次DataCopyPad写出的数据可能与下次迭代向量计算产生RAW冲突。最后一次迭代的同步是无效等待。
- 修复模式：将同步从Impl内部移到调用方循环中，仅在还有后续迭代时插入PipeBarrier<PIPE_ALL>()。
- 可审查性：中
- 审查规则建议：循环调用含DMA操作的函数时，检查迭代间是否正确设置pipeline barrier

---

### fe350c5a ut support infer_datatype/infer_shaperange
- 分类：非缺陷
- 原因：UT框架功能增强(新增infer_datatype/infer_shaperange支持)

---

### b4baa0d2 [SplitCore] Sync bugfix code
- 根因类别：多处逻辑缺陷(无符号整数下溢+边界条件+分核循环无终止保护)
- 涉及文件：attention/common/op_host/split_core.h
- 缺陷描述：SplitCore分核算法多处缺陷(DTS2025092902423)。(1) uint32_t减法下溢：batchLeftCost-=s1GLeftCost等操作被减数<减数时下溢为极大值，分配逻辑错乱。(2) 跳过空batch的while循环与后续bIdx++叠加可能多跳有效batch。(3) ==改>=：索引因跳过逻辑超过边界时无法终止。(4) 分核主循环缺核数上限保护：unassignedCost>0但uint32下溢后curCoreIdx超过coreNum数组越界。(5) coreUse为0时coreUse-1越界。(6) minMaxCost初始值用totalCost不合适。
- 修复模式：安全减法(a>b?a-b:0U)+isComplete状态标记+>=边界判断+核数保护+std::max(coreUse,1U)+UINT32_MAX初始值。
- 可审查性：中
- 审查规则建议：uint32_t减法操作必须检查下溢风险；循环对索引的边界判断优先用>=而非==

---

### ec503a50 fix mem.h InitBuffer
- 根因类别：API变更适配错误(InitBuffer接口废弃后复制粘贴引入新缺陷)
- 涉及文件：common/include/kernel/mem.h
- 缺陷描述：原InitBuffer+手动设logicPos模式改为LocalTensor构造函数。但修复本身引入复制粘贴错误：L0A/L0B/L0C的tensor全部赋值给tensor[ASCEND_CB]索引，后续被64be6dd6修复。
- 修复模式：替换为构造函数直接初始化(但引入了索引错误)。
- 可审查性：高
- 审查规则建议：数组索引与语义标识匹配检查

---

### 4bbc13de fix tilingKey
- 分类：非缺陷
- 原因：新增tilingkey模板文件(纯新增3786行)

---

### 50aaff27 graph修改
- 分类：非缺陷
- 原因：变量命名改善(file→files)，功能无变化

---

### b7ff6fa2 fix_add_2dims
- 根因类别：多处缺陷(buffer复用冲突+变量初始化顺序+UB内存不足+同步缺失)
- 涉及文件：mc2/moe_distribute_combine_add_rms_norm/op_kernel/moe_distribute_combine_add_rms_norm.h, mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2.h, mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h
- 缺陷描述：(1) 多个LocalTensor从同一TBuf Get不同类型数据导致覆盖(expertScalesBuf_和rowTmpFloatBuf_)。(2) epWorldSize_/moeExpertPerRankNum_只在!hasElasticInfoFlag_分支赋值，但后续无条件使用。(3) 总buffer超限未降级为单buffer。(4) DataCopyPad后缺SyncFunc同步。(5) WaitDispatch中对同一tensor先Duplicate再DataCopy存在竞态。
- 修复模式：新增独立buffer解除复用+提前赋值+动态降级+补同步+引入stateResetBuf消除竞态。
- 可审查性：低
- 审查规则建议：同一TBuf被Get为多种类型tensor时标记复用冲突；条件分支内赋值的变量不得在分支外无条件使用

---

### 214c47f5 modify GQA antiquant 8 buffer and fix bugs for gqa antiquant pertoken
- 根因类别：多处缺陷(参数校验缺失+循环逻辑错误+buffer大小不足)
- 涉及文件：attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp, attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp, attention/incre_flash_attention/op_kernel/incre_flash_attention_preload_dd.h
- 缺陷描述：(1) SetupPerfMode中gqaKvNZFlag_分支return导致后续ropeFlag_不可达(dead code)。(2) CopyInMm1AToL1中msdIterNum循环与地址偏移计算错误(mIter*copyIterNum+i应为i)。(3) CopyInMm1BToL1ForPA中baseN=128/sizeof(KV_T)应为256/sizeof(KV_T)，拷贝宽度不足。(4) GQA KV NZ场景缺少antiquantMode参数校验和维度校验。
- 修复模式：if-else链式判断修复dead code+循环重构+baseN修正+校验补全。
- 可审查性：低
- 审查规则建议：if分支中的return导致后续代码不可达应被静态分析标记

---

### 6cef41c6 fix ut bug
- 根因类别：UT测试代码缺陷(cmake配置+宏参数名typo+链接库缺失)
- 涉及文件：cmake/config.cmake, cmake/func_utest.cmake, cmake/ut.cmake, tests/ut/framework_normal/common/(多文件), tests/ut/framework_normal/op_api/CMakeLists.txt
- 缺陷描述：(1) ASCEND_OP_NAME的逗号→分号替换在条件分支内部，其他分支未执行导致多算子名list操作失败。(2) DO_TILING宏参数名tilingPara与宏体内tilingContextPara不一致，编译错误。(3) ut.cmake缺少json/gtest链接库。(4) socVersion硬编码Ascend910_95改为可配置。
- 修复模式：cmake配置提前执行+宏参数名修正+补全链接库+参数化。
- 可审查性：中
- 审查规则建议：C++宏定义形参名必须与宏体内使用的变量名一致

---

### 9c461b25 mc2单词拼写错误修改
- 分类：非缺陷
- 原因：纯拼写修正(AllGahter→AllGather, Exector→Executor等34个文件)

---

### 0708f156 build_lib支持-j, --jit支持自定义算子
- 分类：非缺陷
- 原因：新feature(构建脚本功能增强)

---

### 1a22fcb9 secure_c使用新版本，protobuf使用gitcode链接
- 分类：非缺陷
- 原因：第三方依赖升级和镜像切换

---

### 60a57476 修正custom包名
- 根因类别：构建配置缺陷(包命名格式错误)
- 涉及文件：cmake/custom_build.cmake
- 缺陷描述：自定义算子包文件名`-linux.${ARCH}`不符合CANN命名规范，应为`_linux-${ARCH}`。下游安装脚本或部署流程无法正确识别。
- 修复模式：修正命名格式为_linux-。
- 可审查性：高
- 审查规则建议：CPACK_PACKAGE_FILE_NAME应通过正则校验符合标准模式

---

### bd4cc701 修复example找不到头文件的错误
- 分类：非缺陷
- 原因：example编译脚本完善(补充include路径和链接库)

---

### 9c6d1284 fix interleave_rope,moe_gating_top_k,moe_re_routing,moe_init_routing_v3
- 根因类别：多类缺陷(格式化字符串不匹配+变量未赋值+平台信息获取错误+shadowing)
- 涉及文件：moe/moe_gating_top_k/op_host/(tiling.cpp, tiling_arch35.cpp), moe/moe_re_routing/op_host/moe_re_routing_tiling_base.cpp, moe/moe_init_routing_v3/op_host/(infershape.cpp, tiling.cpp, tiling_base.cpp, op_api/aclnn.cpp), posembedding/interleave_rope/op_host/interleave_rope_tiling.h
- 缺陷描述：(1) OP_LOGE中%lu用于int64_t等格式化说明符不匹配(UB)。(2) moe_init_routing_v3 infershape中quantMode声明后从未从attrs获取值(初始为-1)，InferDataType逻辑永远走错分支。(3) tiling中平台信息在错误阶段获取。(4) aclnn中不必要的V2回退逻辑导致错误路由。(5) 1e-20应为1e-20f。(6) 构造函数参数名shadow成员变量。
- 修复模式：格式符修正+补充attrs读取+平台信息获取前移+删除回退逻辑+加f后缀+改参数名。
- 可审查性：低(多个不相关缺陷混在一个提交)
- 审查规则建议：启用-Wformat和-Wshadow；infershape中从attrs获取的属性值应检查"声明后是否赋值再使用"

---

### 5c38f9c5 fa compile waring fix
- 根因类别：构建配置缺陷(参数shadow在-Werror下编译失败)
- 涉及文件：attention/flash_attention_score/op_host/flash_attention_score_tiling.cpp, attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad_tiling.h
- 缺陷描述：setter函数参数名与类成员变量同名触发-Wshadow告警，在-Werror编译选项下导致编译失败。
- 修复模式：参数重命名加_val后缀+[[maybe_unused]]标注。
- 可审查性：高(但6000+行格式化diff掩盖实际修改)
- 审查规则建议：编译时启用-Wshadow -Werror；格式化变更应与逻辑变更分离提交

---

### 7d1f015a ut support noexec
- 根因类别：构建脚本参数解析缺陷
- 涉及文件：build.sh, cmake/func_utest.cmake
- 缺陷描述：(1) build.sh中--noexec分支缺少shift，后续参数全部错位。(2) cmake中变量名UT_NO_EXEC与build.sh传入的ENABLE_UT_EXEC不一致，noexec功能完全失效。
- 修复模式：补充shift+统一变量名为ENABLE_UT_EXEC。
- 可审查性：高
- 审查规则建议：shell脚本参数解析每个case必须包含shift；CMake -D变量名应与脚本传入名一致

---

### 7ed8fea3 -j命令取消空格
- 根因类别：构建脚本参数解析缺陷
- 涉及文件：build.sh
- 缺陷描述：-j参数只匹配精确字符串-j(空格分隔形式)，不支持-j8(紧跟数字)形式。用户输入-j8时匹配不到，并行编译数未被设置。
- 修复模式：匹配模式从-j改为-j*通配+提取数字部分。
- 可审查性：高
- 审查规则建议：带值的短选项应同时支持紧跟和空格两种形式

---

### f314395a aclnnGroupedMatmulFinalizeRoutingV3修复文档和报错信息
- 根因类别：输入校验缺失(多芯片场景参数校验未解耦+维度一致性校验缺失)
- 涉及文件：gmm/grouped_matmul_finalize_routing/op_host/op_api/aclnn_grouped_matmul_finalize_routing.cpp, aclnn_grouped_matmul_finalize_routing_MX_checker.h, docs/aclnnGroupedMatmulFinalizeRoutingV3.md
- 缺陷描述：(1) 910/95芯片路径多余CheckFormat调用，在不适用芯片上做错误format校验。(2) DAV_3510芯片weight非NZ格式处理逻辑不正确。(3) CheckDimRange未按芯片区分。(4) weight类型检查逻辑在非DAV_3510芯片上报错信息不准确。(5) 缺少x和weight的k维一致性前置校验。(6) weight/weightScale非ND场景无format校验。
- 修复模式：按芯片架构解耦校验+增加前置一致性校验+将format校验移入芯片专用checker。
- 可审查性：中
- 审查规则建议：多芯片校验应在架构层面解耦；维度一致性校验应作为shape check前置步骤

---

### ef8ac892 修复aclnn文档与头文件参数不匹配的问题
- 分类：非缺陷
- 原因：纯文档修复(const aclrtStream声明与头文件对齐)

---

## ops-transformer-dev - 528条缺陷


## 分析进度

- 总缺陷提交：788 (实际783，5条纯文档漏过滤: fbc5b72884, 50d822939c, c3df1cb505, a125659b8e, 3beec5f90e)
- 已分析：788/788 (全部完成，257条阶段2排除)
- 代码缺陷：528
- 剩余：0 (阶段2完成)

---

### b85709b25e MoeInitRoutingV3(David) MXFP8修复overflow问题
- 根因类别：硬件平台寄存器配置遗漏
- 涉及文件：moe/moe_init_routing_v3/op_kernel/arch35/moe_v3_common.h, moe_v3_gather_mxfp8_quant.h, moe_init_routing_v3_apt.cpp
- 缺陷描述：NPU_ARCH 3101(David)平台执行MXFP8量化时，未设置溢出模式控制寄存器(SPR第60号)，MoeGatherOutMxfp8Quant::Init缺少SetCtrlSpr<60,60>(0)调用，导致浮点overflow
- 修复模式：save-modify-restore寄存器管理——kernel入口保存原值，Init中设0，退出时恢复
- 可审查性：低
- 审查规则建议：新增arch适配时检查是否需要特殊寄存器配置；MXFP8量化路径必须检查溢出模式设置

### 66337ddb6b fix ScatterPaKvCache tiling dual-in-out
- 根因类别：early return跳过必要计算
- 涉及文件：attention/scatter_pa_kv_cache/op_host/scatter_pa_kv_cache_tiling_arch35.cpp
- 缺陷描述：TemplateRope()中DUAL_IN_OUT模式直接return GRAPH_SUCCESS，跳过了vHandleNumPerLoop_、vLoopNum_、vTailHandleNum_的计算，导致后续kernel使用未初始化的tiling值
- 修复模式：将early return改为条件内执行相同tiling计算逻辑
- 可审查性：高
- 审查规则建议：early return路径必须检查所有成员变量/tiling参数是否已赋值；新增模式时验证所有tiling字段都被填充

### eeba666256 moe算子A5编译选项修正
- 根因类别：编译选项缺少平台条件分支
- 涉及文件：8个moe算子的CMakeLists.txt
- 缺陷描述：moe系列CMakeLists未区分A5平台(ascend910_95)，对所有平台统一使用--cce-auto-sync=on且缺少-mllvm -cce-aicore-dcci-before-kernel-end=false选项
- 修复模式：CMakeLists中添加ASCEND_COMPUTE_UNIT平台判断，为ascend910_95单独配置编译选项
- 可审查性：中
- 审查规则建议：新增平台支持时检查所有相关算子CMakeLists是否包含该平台编译选项分支

### c8cf3f011d NTD场景全量化拦截误改修复
- 根因类别：条件检查范围过宽误拦截合法场景
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：CheckNTDLayoutCrossover中全量化拦截对所有NTD场景无条件生效，实际只有Prefill MLA场景(enablePFAMLA||enablePFARope)下NTD不支持全量化
- 修复模式：将拦截检查包裹在enablePFAMLA||enablePFARope条件内，缩小拦截范围
- 可审查性：中
- 审查规则建议：参数校验修改时附带UT覆盖所有受影响的layout-feature组合

### 1979ed3cfe 解决推理FA ATK工程CI开启缓存后cache丢失问题
- 根因类别：测试中间数据持久化方式不当
- 涉及文件：attention/fused_infer_attention_score/tests/st/executor_aclnnFusedInferAttentionScoreV4.py
- 缺陷描述：k/v_cache通过np.save写入磁盘再np.load读取，CI缓存开启后临时文件被清理导致找不到文件；overwrite_structured_mask函数条件不满足时return缺少返回值；tensor在NPU上缺少.cpu()调用
- 修复模式：文件传递改为内存传递(input_data.kwargs字典)；补全return值；添加.cpu()调用
- 可审查性：中
- 审查规则建议：测试代码禁止用文件系统传递中间数据；torch tensor转numpy前必须.cpu()；所有函数分支必须有明确return值

### af72ccb938 gather pa kv cache fix
- 根因类别：越界内存访问 + DataCopy API容量不足
- 涉及文件：attention/gather_pa_kv_cache/op_host/gather_pa_kv_cache_tiling_arch35.cpp, op_kernel/arch35/gather_pa_kv_cache_nd.h, gather_pa_kv_cache_nz.h
- 缺陷描述：(1)blockTablesGm_.GetValue()在blockTableOffset>=blockTableWidth_越界时仍被调用；(2)DataCopyParams不支持大数据seqLens拷贝需用Ext版本；(3)blockLen赋值缺少static_cast<uint32_t>；(4)maxUbHiddenSize未做CeilAlign对齐
- 修复模式：GM读取移到边界检查后；替换为DataCopyExtParams；添加类型转换；添加CeilAlign
- 可审查性：高
- 审查规则建议：GM访问必须在边界检查通过后执行("check-before-access")；DataCopy参数选型需根据数据量上限选择对应API版本

### ab0eb69c00 修正gmm actType资料描述不清晰的问题
- 根因类别：文档/错误信息与代码不一致
- 涉及文件：gmm/grouped_matmul/docs/aclnnGroupedMatmulV4.md, op_api/aclnn_grouped_matmul.cpp
- 缺陷描述：actType文档枚举值遗漏合法值0(无激活函数)；错误日志变量名"ActiveType"与接口名"ActType"不一致
- 修复模式：文档补上0；错误日志字段名修正
- 可审查性：高
- 审查规则建议：API文档枚举值列表应与代码校验范围自动对比一致；错误信息中参数名应与接口定义完全匹配

### 11dc74d461 [FAG] fix bn2 sprasemode3 bug
- 根因类别：稀疏模式下序列长度上界未截断
- 涉及文件：attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_kernel_base.h
- 缺陷描述：CheckS2EndLen中无prefixN且sparseMode==RIGHT_DOWN_CAUSAL(mode 3)时，s2EndLen未截断到s2Size，BN2多block场景下可能越界
- 修复模式：else分支添加sparseMode==RIGHT_DOWN_CAUSAL时Min(s2EndLen, s2Size)截断
- 可审查性：中
- 审查规则建议：所有sparseMode分支都应对序列长度变量做上界截断；新增sparseMode需检查所有使用s2EndLen的路径

### fb58b43e98 fix matmulReduceScatterV2
- 根因类别：校验条件边界值过严
- 涉及文件：mc2/matmul_reduce_scatter_v2/op_host/op_tiling/arch32/matmul_reduce_scatter_v2_aiv_mode_tiling.cpp
- 缺陷描述：checkAndResetTilingData_SmallM中swizzlCount上界写死9(应16)、ubMoveNum下界写死6(应4)，smallM场景合法参数被拒绝
- 修复模式：放宽校验边界
- 可审查性：中
- 审查规则建议：tiling校验魔数边界值应定义为常量并附注推导来源；校验范围变更需与tiling生成实际输出范围对齐

### b6ec4bfe86 bugfix:ffag shape校验修正
- 根因类别：条件判断逻辑取反错误
- 涉及文件：attention/fused_floyd_attention_grad/op_host/fused_floyd_attention_grad_tiling_common.cpp
- 缺陷描述：CheckSupportShape中4处OP_CHECK_IF调用条件缺少取反——CheckSameShape返回true表示合法，但OP_CHECK_IF(condition)为true时触发失败，导致合法shape被拒绝、非法shape被放行
- 修复模式：4处条件从CheckSameShape(...)改为!CheckSameShape(...)
- 可审查性：高
- 审查规则建议：OP_CHECK_IF宏参数应表示"错误条件"，审查时检查条件语义是否与fail语义一致；可建lint规则检测直接调用bool函数时是否缺少取反

### 75d961c44b int4(int32) 伪量化场景拦截(回退)
- 根因类别：参数校验条件过严误拦截合法输入
- 涉及文件：attention/incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp, 3个文档md
- 缺陷描述：CheckAntiQuantParam中PER_TENSOR_HEAD_MODE被错误包含在仅允许INT8的校验条件中，导致INT4(INT32)伪量化在per-tensor和per-tensor叠加per-head模式下被误拦截；还有冗余的per-tensor shape校验逻辑同样误拦截
- 修复模式：缩小拦截条件范围(移除PER_TENSOR_HEAD_MODE)，删除冗余校验代码块
- 可审查性：中
- 审查规则建议：OR连接多个枚举值时逐一确认每个枚举值的数据类型约束是否一致；新增数据类型时同步审查所有相关校验分支

### 242ee29f4e 修复算子在空tensor临界情况与标杆输出存在差异的问题
- 根因类别：空tensor边界条件处理缺失
- 涉及文件：moe/moe_token_unpermute_with_routing_map/op_host/op_api/aclnn_moe_token_unpermute_with_routing_map.cpp
- 缺陷描述：空tensor分支直接返回未将输出置零，含垃圾数据；paddedMode=true时空tensor判断条件不正确——将permutedTokens为空也拦截了但paddedMode下仍需计算；后续代码段缺少空tensor保护
- 修复模式：修正空tensor判断条件(区分paddedMode)；早退路径对输出执行ZerosLike+ViewCopy置零；补充保护分支
- 可审查性：中
- 审查规则建议：空tensor早退路径必须确保所有输出tensor正确初始化；含模式分支的空tensor处理需分别验证

### 69e49494d5 修正gmm activetype报错描述不清晰的问题
- 根因类别：错误信息不准确(缺少实际参数值)
- 涉及文件：gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul.cpp
- 缺陷描述：DAV_3510不支持activation时报错硬编码，未输出实际activeType值且暴露内部平台名称
- 修复模式：报错改为格式化字符串输出具体值，平台名改为"this platform"
- 可审查性：高
- 审查规则建议：CHECK/LOGE中报错应包含导致错误的实际参数值；避免在用户可见信息中暴露内部平台名

### c3df1cb505 fix gmmsqv1 [注：纯文档修改，应在阶段1过滤]
- 根因类别：文档缺陷(仅md文档修改)
- 涉及文件：3个文档md文件
- 缺陷描述：仅重构API文档参数说明和格式，无代码逻辑变更
- 修复模式：文档重写/格式修正
- 可审查性：低
- 审查规则建议：N/A (应从缺陷列表中排除)

### f28c491f24 回退ROPE V2相关改动
- 根因类别：接口签名变更破坏向后兼容
- 涉及文件：24个文件(核心：aclnn_interleave_rope.cpp, aclnn_rotary_position_embedding.cpp, 删除V2文件等)
- 缺陷描述：ROPE V2修改了aclnnInnerRotaryPositionEmbeddingGetWorkspaceSize签名(增加rotate参数)，但ACLNNINNER层未适配version分发，导致mindspore通过V1接口调用时签名不匹配
- 修复模式：完整回退V2所有改动，恢复V1签名
- 可审查性：中
- 审查规则建议：修改公共内部函数签名时必须确认所有调用方已适配；版本化API用version分发而非直接修改已有签名

### 7275ae854e 修改错误的拦截逻辑
- 根因类别：shape校验使用错误维度来源 + tiling层缺少校验
- 涉及文件：aclnn_grouped_matmul_finalize_routing_910_95_checker.h, grouped_matmul_finalize_routing_quant_tiling.cpp/.h
- 缺陷描述：checker中outputExpectShape用{m,n}但output的batch维应取自output自身shape(outputBS)而非input推导的m；tiling层完全缺少FP4场景k/n约束校验
- 修复模式：用outputBS替换m；tiling层新增CheckFp4Shape()补全MXFP4校验
- 可审查性：高
- 审查规则建议：shape校验期望值应直接取自对应tensor实际shape；aclnn层和tiling层校验逻辑保持一致

### 16bb2424c1 修复sink场景下的精度问题
- 根因类别：累加符号逻辑错误
- 涉及文件：attention/flash_attention_score_grad/op_kernel/arch32/flash_attention_score_grad_post.h
- 缺陷描述：循环内用dsinkCalc = -vecOut.GetValue(0)(赋值取负)，每次覆盖而非累加，多block结果只保留最后一个负值；BNGS1S2模式完全缺少dsink计算逻辑
- 修复模式：循环内改为+=累加，写出时统一取负；为BNGS1S2路径补充完整sink计算
- 可审查性：高
- 审查规则建议：循环内累加用+=不用=；正负号在最终写出时统一处理；多代码路径功能完整性检查

### 3c061c5749 barrier example改成双卡之后运行不起来
- 根因类别：示例参数与硬件配置不匹配
- 涉及文件：mc2/distribute_barrier/examples/test_aclnn_distribute_barrier.cpp
- 缺陷描述：单卡改双卡后K=3、moeExpertNum=7在EP_WORLD_SIZE=2下不兼容(专家数无法被卡数整除/topK超出范围)
- 修复模式：K从3改1，moeExpertNum从7改1
- 可审查性：高
- 审查规则建议：示例代码修改硬件配置时同步调整所有依赖参数

### 70a262adf6 Fix Dispatch and Combine, rank < 8
- 根因类别：硬编码对齐假设(假设worldSize为8的倍数)
- 涉及文件：moe_distribute_combine_v2_tiling.cpp, moe_distribute_combine_a2.h, moe_distribute_dispatch_v2_infershape.cpp, moe_distribute_dispatch_v2_tiling.cpp, moe_distribute_dispatch_a2.h
- 缺陷描述：(1)infershape中epRecvCounts的*epWorldSize/RANK_NUM_PER_NODE运算优先级错误，epWorldSize<8时整除得0；(2)kernel中buffer未按Vector指令256字节对齐；(3)GatherMask的mask参数localMoeExpertNum可能超过32(uint32 bit数上限)；(4)dataSizePerRank_未对齐；(5)单服务器模式bufferChosen同步缺失
- 修复模式：RoundUp对齐；GatherMask分批循环(每批BITS_PER_U32)；修正运算优先级(加括号)；epWorldSize<=8强制fullmesh且aicpuBlockDim=1；DataCopy改DataCopyPad
- 可审查性：中
- 审查规则建议：Vector/DMA指令buffer分配检查对齐约束(256字节/32字节)；位掩码操作mask宽度不超过32bit；整数除法注意运算优先级；硬编码rankPerNode=8需条件分支处理

### dea1311af5 fix aclnnMoeTokenPermuteWithRoutingMap empty token and add check
- 根因类别：空tensor判断条件过宽 + 输入校验缺失
- 涉及文件：moe/moe_token_permute_with_routing_map/op_host/op_api/aclnn_moe_token_permute_with_routing_map.cpp, UT文件
- 缺陷描述：空tensor判断包含permuteTokensOut(输出)为空时也跳过计算，但应仅在routingMap(输入)为空时跳过；缺少probs与routingMap的dim0/dim1 shape一致性校验
- 修复模式：空tensor判断简化为仅检查routingMap；新增shape一致性校验；补充UT
- 可审查性：高
- 审查规则建议：空tensor早退条件不应包含输出tensor(输出shape由输入决定)；多输入tensor间shape一致性约束必须显式检查

### 2cea0f9aed [FAG] BN2S2 adapt dkdv fix ub
- 根因类别：模板参数误用 + 跨核同步遗漏
- 涉及文件：flash_attention_score_grad多个文件(tiling, block_cube, block_vec, kernel, kernel_base, tiling_key)
- 缺陷描述：BN2S2模式适配dk/dv写UB路径时存在3类缺陷：(1)IS_DK_WRITE_UB被错误用于DQ和DV路径(应分别使用IS_DQ_WRITE_UB/IS_DV_WRITE_UB)；(2)Fixpipe输出缺少isFixpOut条件守卫导致无条件执行；(3)BN2S2 UB写模式下缺少CrossCoreWaitFlag同步，L0C累加结果可能被覆盖；(4)DqkvMulsAndCastFromUB早退条件错误(用s1Size==1判断而非halfS1RealSize/halfS2RealSize==0)
- 修复模式：逐一修正模板参数为正确的IS_DQ_WRITE_UB/IS_DV_WRITE_UB/IS_DK_WRITE_UB；Fixpipe加isFixpOut守卫；新增needSyncDkDvFixUb标志管理跨核同步；早退条件改为检查realSize
- 可审查性：中
- 审查规则建议：新增硬件buffer写路径(如UB/GM)时须逐一验证所有模板参数是否与目标tensor匹配；跨核数据搬运须有配对的Set/WaitFlag

### 5508fbb6c9 修复MoeInitRouting算子cmakelists
- 根因类别：复制粘贴错误
- 涉及文件：moe/moe_init_routing/op_host/CMakeLists.txt
- 缺陷描述：CMakeLists.txt中OP_NAME写成MoeFinalizeRouting，实际应为MoeInitRouting，导致编译选项应用到了错误的算子
- 修复模式：两处OP_NAME改为MoeInitRouting
- 可审查性：高
- 审查规则建议：CMakeLists中OP_NAME必须与所在目录算子名一致；copy-paste新算子构建配置后须检查OP_NAME

### 949f8193fd 算子精确度问题定位(MoeInitRouting)
- 根因类别：流水线同步遗漏 + 平台编译选项缺失
- 涉及文件：moe_init_routing/CMakeLists.txt, moe_gather_out_small_activate_row.h, moe_init_routing_quant_v2/CMakeLists.txt, moe_v2_gather_dynamic_quant_droppad.h, moe_v2_gather_quant_simt.h, tiling_base.cpp
- 缺陷描述：(1)多核场景Process()入口缺少SyncAll()，核间数据未同步即开始计算；(2)Duplicate写完dynamicQuantLocal后未加PipeBarrier<PIPE_V>()即执行Div读取同一buffer，存在RAW hazard；(3)bf16->half转换路径缺少MTE3到V pipe的同步(TQueSync)，Cast读到未完成的MTE3写入数据；(4)CMake未区分A5平台(ascend910_95)缺少-mllvm -cce-aicore-dcci-before-kernel-end=false选项
- 修复模式：添加SyncAll/PipeBarrier/TQueSync；CMake区分平台加编译选项
- 可审查性：中
- 审查规则建议：kernel中Duplicate/DataCopy后同一buffer被读取前须有对应PipeBarrier；bf16与half间Cast须检查pipe同步；新增平台编译选项须检查是否需按平台分支

### 3cfe2f3b64 修复PANZ per-token-group下的伪量化参数对齐
- 根因类别：硬件对齐要求未满足
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_antiquant_processor.h
- 缺陷描述：CopyAntiqScaleE8M0Nz的copyTotalS参数未32对齐，per-token-group量化场景下scale数据搬运长度不满足NZ格式32对齐要求
- 修复模式：(taskParam.copyTotalS + 31) / 32 * 32 向上对齐
- 可审查性：高
- 审查规则建议：所有NZ格式的DataCopy/CopyPad调用的size参数须检查是否满足对齐约束(通常16或32)

### 345ac22dca 误拦截问题修复(RotaryPositionEmbedding)
- 根因类别：校验条件过严 + 初始化顺序错误
- 涉及文件：posembedding/rotary_position_embedding/op_host/rope_rotate_matrix_tiling.cpp
- 缺陷描述：(1)CheckMode()通过匹配tilingMode名称(BNSD_BROADCAST_TWODIM等)判断是否支持interleaved模式，但未覆盖所有合法的维度广播情况，导致合法输入被误拦截；(2)GetDimLen()在CheckMode()之后调用，CheckMode()使用的xFirstDim等字段尚未初始化
- 修复模式：CheckMode()改为基于实际维度关系判断(rFirstDim==1 && rSecondDim==1等)；GetDimLen()调用移到CheckMode()之前
- 可审查性：高
- 审查规则建议：校验函数所依赖的数据必须在调用前完成初始化；输入校验条件应基于语义约束而非实现细节(如tiling mode名称)

### 5a2fbab09d [kernel] attention_update optimize exp precision
- 根因类别：数值精度不足
- 涉及文件：attention/attention_update/op_kernel/arch35/attention_update_with_lse_regbase.h, attention_update_without_lse_regbase.h
- 缺陷描述：Exp函数使用默认ZEROING模式，精度不够(>1ULP误差)，在attention update的lse指数计算中累积误差影响最终结果精度
- 修复模式：指定ExpSpecificMode为PRECISION_1ULP_FTZ_FALSE，确保1ULP精度且不flush-to-zero
- 可审查性：低
- 审查规则建议：attention/softmax路径中的Exp调用须评估精度要求，高精度场景使用PRECISION_1ULP模式

### 6c0b8ee1ab fix deter casual GQA
- 根因类别：公式遗漏GQA分支
- 涉及文件：flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp
- 缺陷描述：CalcleCausalDeterParam()计算rUpper时未考虑GQA(g!=1)场景的额外资源需求rm3项，导致确定性计算资源分配不足
- 修复模式：当fBaseParams.g != 1时加入rm3 = (m + m1 + 1) * t1修正项
- 可审查性：中
- 审查规则建议：tiling参数计算中涉及GQA/MQA/MHA分支的公式须逐一验证各head分组场景

### 798597da0a fix dispatch oom bug
- 根因类别：字节数/元素数混淆
- 涉及文件：mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_a2.h
- 缺陷描述：Duplicate<int32_t>(statusTensor_, 0, worldSize_ * DATA_OFFSET)中DATA_OFFSET是字节偏移，但Duplicate的count参数期望元素数，实际写入量为预期的4倍(sizeof(int32_t))，越界写导致OOM
- 修复模式：改为worldSize_ * DATA_OFFSET / sizeof(int32_t)
- 可审查性：高
- 审查规则建议：Duplicate/DataCopy的size参数须区分字节数和元素数；混用字节偏移常量和元素计数API时须显式除以sizeof

### d2601de86d fix sfa race bug
- 根因类别：MTE3流水线同步遗漏
- 涉及文件：attention/sparse_flash_attention/op_kernel/sparse_flash_attention_service_vector_mla.h
- 缺陷描述：MergeKv()中DataCopyPad将v0ValidSizeUb_写到GM后，未等待MTE3完成即返回，后续代码可能复用v0ValidSizeUb_ buffer导致数据竞争
- 修复模式：DataCopyPad后添加SetFlag/WaitFlag<MTE3_S>
- 可审查性：高
- 审查规则建议：DataCopy/DataCopyPad到GM后，若源buffer将被复用，须有MTE3同步barrier

### 0fb800dc41 [FAG] fix old deter 28 core error
- 根因类别：多缺陷组合(运算符优先级+类型+越界+核数对齐)
- 涉及文件：flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp, flash_attention_score_grad_s1s2_bn2gs1s2_regbase.h, tiling_data_regbase.h
- 缺陷描述：(1)ProcessQuantInfo()末尾缺少return GRAPH_SUCCESS，函数返回未定义值；(2)运算符优先级：||与&&混用无括号(如 a || b && c)导致逻辑错误；(3)for循环中int64_t与size_t比较产生有符号/无符号警告；(4)确定性V核数对非64核场景未做32对齐(如28核)导致除法结果为0或不对齐；(5)DataCopy/DataCopyPad未检查vBlockIdx是否在有效V核范围内，超出的V核执行无效DMA访问
- 修复模式：补return；加括号；改size_t；V核数做32取整；加vBlockIdx边界守卫
- 可审查性：高(前3项) / 中(后2项)
- 审查规则建议：非void函数所有路径须有return；||和&&混用须加括号；for循环遍历容器.size()须用size_t；多核场景DataCopy须检查当前核是否在有效核数范围内

### 0fee64b967 fix AttentionToFFN/FFNToAttention aclnn demo and cmake
- 根因类别：API误用(循环调用一次性初始化) + CMake参数重复
- 涉及文件：mc2/attention_to_ffn/docs, examples, op_host/CMakeLists.txt, mc2/ffn_to_attention/op_host/CMakeLists.txt
- 缺陷描述：(1)HcclCommInitAll应只调用一次，但被放在for循环内重复调用WORLD_SIZE次；(2)CMakeLists中OPTYPE和ACLNNTYPE参数重复写(attention_to_ffn attention_to_ffn / aclnn aclnn_inner)导致编译注册异常；(3)FFN Worker等待超时10s过短改为30s
- 修复模式：HcclCommInitAll移出循环；去除CMake参数重复
- 可审查性：高
- 审查规则建议：集合通信初始化函数(如HcclCommInitAll)不应放在循环内；CMake宏参数列表不应有重复项

### 28a4126215 dispatch cumsum分核功能代码回退
- 根因类别：功能回退(原实现过于复杂导致可靠性问题)
- 涉及文件：moe_distribute_dispatch_v2/op_host, op_kernel/moe_distribute_dispatch_v2_full_mesh.h, tiling.h
- 缺陷描述：原cumsum多核实现涉及复杂的核间状态同步(GatherMask+Sum+标志位轮询)，删除了~200行代码回退到简化方案，改用AscendC::GetCumSumMaxMinTmpSize获取正确的UB大小
- 修复模式：删除BufferInit/WaitDispatchClearStatus/GatherSumRecvCnt/CalRecvAndSetFlag/GetCumSum等复杂实现，改用平台API
- 可审查性：低(设计决策)
- 审查规则建议：优先使用平台提供的API而非自行实现复杂的核间同步；核间轮询同步方案须有超时和错误恢复机制

### 51b73426534 Fix SyncFunc EventId Type
- 根因类别：类型转换错误 + 代码重复
- 涉及文件：mc2/下26个文件，新增mc2/common/inc/kernel/mc2_kernel_utils.h
- 缺陷描述：SyncFunc中FetchEventID返回TEventID类型，但被强转为int32_t传给SetFlag/WaitFlag，类型不匹配可能导致事件ID截断；26个文件各自定义相同的SyncFunc模板函数
- 修复模式：统一提取到mc2_kernel_utils.h，使用正确的TEventID类型
- 可审查性：高
- 审查规则建议：事件ID类型须使用TEventID/event_t，不应强转为int32_t；公共kernel工具函数应放在common目录避免各文件重复定义

### c51f0736fc dispatch&combine性能打点问题修复
- 根因类别：轮询逻辑低效(重复检查已完成项)
- 涉及文件：mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_a2.h
- 缺陷描述：WaitDispatch轮询循环中，已接收flag的rank在下一轮仍被重新DataCopy+检查，且每次清零后用DCCI+MTE2 sync保序，既浪费带宽又增加延迟
- 修复模式：新增isVisited LCM bool数组跳过已完成rank；移除不必要的DCCI清零保序
- 可审查性：中
- 审查规则建议：轮询等待循环中须标记已完成项避免重复检查；DCCI指令开销大，仅在确实需要cache一致性时使用

### 69d0fee987 fix PipeBarrier(rope_with_sin_cos_cache)
- 根因类别：PipeBarrier<PIPE_ALL>滥用导致流水线停顿
- 涉及文件：posembedding/rope_with_sin_cos_cache/op_kernel下3个文件
- 缺陷描述：大量使用PipeBarrier<PIPE_ALL>作为万能同步，但PIPE_ALL会阻塞所有流水线，既是性能瓶颈也掩盖了真正的依赖关系，还可能在某些场景下因过宽同步导致死锁
- 修复模式：替换为精确的per-pipe同步函数(VToMTE2Sync, VToMTE3Sync, MTE2ToVSync, MTE3ToVSync)
- 可审查性：高
- 审查规则建议：禁止使用PipeBarrier<PIPE_ALL>作为默认同步手段；须分析实际的生产者-消费者pipe关系使用精确同步

### 1fab7bab45 MatmulAllReduce伪量化Tiling GetPlatformInfo错误修复
- 根因类别：基类初始化遗漏
- 涉及文件：mc2/matmul_all_reduce/op_host/op_tiling/arch35/weight_quant_matmul_all_reduce_tiling_910_95.h
- 缺陷描述：两个Tiling子类的构造函数未调用基类的InitCompileInfo()，导致平台信息(如芯片型号、核数等)未初始化，后续tiling计算使用未初始化值
- 修复模式：构造函数体内添加Base::InitCompileInfo()调用
- 可审查性：高
- 审查规则建议：继承Tiling基类时须检查是否需要调用基类的Init系列方法；构造函数中基类初始化不应遗漏

### d640283d1d fa clean code fix
- 根因类别：C风格类型转换 + 缺少override
- 涉及文件：flash_attention_score/op_host/arch35/下3个tiling cpp文件
- 缺陷描述：(1)REGISTER_TILING_TEMPLATE_WITH_ARCH宏参数中使用C风格强转(int32_t)而非static_cast；(2)SetSplitCoreModeParam()覆盖虚函数但缺少override关键字
- 修复模式：改用static_cast<int32_t>；添加override
- 可审查性：高(编译器可检测)
- 审查规则建议：禁止C风格类型转换，统一使用static_cast；虚函数覆盖须标记override

### a125659b8e fix aclnn.md
- 纯文档/注释修改，已排除。主要改动是3个.md文档 + 删除hpp中残留的调试打印注释。

### d69eec6dab bugfix: fix libopapi undefined symbol(ffn_worker_scheduler)
- 根因类别：构建配置错误(CMakeLists参数缺失)
- 涉及文件：attention/attention_worker_scheduler/op_host/CMakeLists.txt, ffn/ffn_worker_scheduler/CMakeLists.txt, ffn/ffn_worker_scheduler/op_host/CMakeLists.txt
- 缺陷描述：add_modules_sources()调用缺少OPTYPE和ACLNNTYPE参数，导致符号未导出，链接时undefined symbol。另外file(GLOB RELATIVE .)使用了错误的相对路径基准
- 修复模式：补充OPTYPE/ACLNNTYPE参数；将RELATIVE .改为RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
- 可审查性：高
- 审查规则建议：add_modules_sources()必须带OPTYPE和ACLNNTYPE参数；file(GLOB RELATIVE .)标记warning应使用CMAKE_CURRENT_SOURCE_DIR

### a02041efea 修复NTD_TND拦截
- 根因类别：复制粘贴字段混用
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：NTD layout下校验valueAntiquantScaleShape第0维时，错误使用valueShapeInfo.b(batch)而非valueShapeInfo.n(num heads)，导致shape校验失效
- 修复模式：两处valueShapeInfo.b改为valueShapeInfo.n
- 可审查性：高
- 审查规则建议：多layout分支校验时逐维核对layout定义与校验字段的一致性；NTD维度顺序(N,T,D)第0维必须用.n

### 61bbcf9a3f fix splitcore mode bug
- 根因类别：函数重命名不完整
- 涉及文件：attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_varlen.cpp
- 缺陷描述：函数IsUsesplit被重命名为IsUseSpliteCoreMode，但只改了定义侧未同步所有引用点，导致编译/链接错误
- 修复模式：将函数名从IsUsesplit改为IsUseSpliteCoreMode
- 可审查性：高
- 审查规则建议：重命名应使用IDE全局重构功能，确保声明、定义、调用一次性修改完毕

### 7075b6143f InitRoutingV2 & FinalizeRoutingV2性能优化-修复revert PR5178
- 根因类别：多重缺陷(参数顺序错误+UB内存计算遗漏+硬件事件类型错误+k=1场景未特化)
- 涉及文件：moe/moe_finalize_routing_v2/下3个文件, moe/moe_init_routing_v2/下3个文件
- 缺陷描述：(1)SetExpertIdxOffset参数rowOuterIdx/rowInnerIdx顺序与调用侧不一致(2)IsKHFullLoad()遗漏expandedRowIdx和expertIdx的空间(3)MTE3_S应为MTE2_S(事件类型写错)(4)k=1场景UB分配策略与k>1混用
- 修复模式：k=1特化路径、修正UB计算、修正事件类型、参数顺序修复
- 可审查性：低
- 审查规则建议：性能优化和bug修复应分开提交；硬件事件类型使用需团队级checklist；UB内存计算变更需附带推导说明

### cefdc8776e fix pse datacopy error due to large s2
- 根因类别：整数溢出(uint16截断)
- 涉及文件：attention/common/op_kernel/arch35/pse.h
- 缺陷描述：s2很大时srcStride超过uint16_t最大值65535，赋给DataCopyParams.srcStride(uint16_t)时截断溢出，数据搬运错误
- 修复模式：先用int64_t计算srcStride，分支条件增加srcStride <= UINT16_MAX判断，超出时走DataCopyExtParams分支
- 可审查性：高
- 审查规则建议：外部输入(shape维度)参与计算的stride/offset赋给窄类型前必须做范围校验；DataCopy参数优先使用Ext版本

### ffe61ca1c1 modify cast overflow
- 根因类别：整数溢出(uint16截断)
- 涉及文件：mc2/common/inc/kernel/mc2_nd_to_nz.h
- 缺陷描述：cpInLen/cpOutLen声明为uint16_t，计算size*sizeof(bfloat16_t)/sizeof(float)时乘积超过65535截断溢出
- 修复模式：uint16_t改为uint32_t；DataCopyParams/DataCopyPadParams替换为Ext版本
- 可审查性：高
- 审查规则建议：与cefdc877同类问题，DataCopy参数由运行时变量计算得出时应优先使用ExtParams版本

### b4b9198148 NTD叠加特性修复
- 根因类别：新layout枚举值遗漏/条件分支覆盖不全
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_block_vec_infer.h, attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：三处遗漏NTD layout处理：(1)SoftmaxLseCopyOut的dstStride只对TND特殊处理(2)NTD下不应开enableIFAMask(3)set_isGqa中错误排除NTD
- 修复模式：增加LAYOUT_NTD条件分支；NTD不启用IFA mask；移除isGqa中多余的NTD排除
- 可审查性：中
- 审查规则建议：新增layout/枚举值时必须全局搜索所有引用现有枚举的条件分支；建立checklist逐项确认新值是否需覆盖

### 34347d5c37 fix debug, dcci and mte2 sync
- 根因类别：硬件流水线同步缺失(DCCI与MTE2未同步)
- 涉及文件：mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_a2.h
- 缺陷描述：windowInstatusTensor_清零写入后发出DCCI指令，但DCCI和后续MTE2操作无同步屏障，MTE2在DCCI未完成时读到脏数据，导致recvFlagNum重复累计，下游combine死锁
- 修复模式：DCCI后插入SyncFunc<AscendC::HardEvent::S_MTE2>()
- 可审查性：低
- 审查规则建议：每个DataCacheCleanAndInvalid调用后若有MTE2操作访问同一内存区域必须插入SyncFunc<S_MTE2>()；建立DCCI使用模板

### a3ff13e7c2 dispatchV2、combineV2文档、代码修正
- 根因类别：初始化顺序错误
- 涉及文件：mc2/moe_distribute_combine_add_rms_norm/op_kernel/moe_distribute_combine_add_rms_norm.h, mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2.h
- 缺陷描述：Init函数中winDataSizeOffsetTp_的计算被放在GetWinAddrByRankId()/CheckWindowSize()调用之后，但这些函数依赖winDataSizeOffsetTp_的值，导致使用未初始化的偏移量
- 修复模式：将winDataSizeOffsetTp_赋值上移到被依赖函数调用之前；两个combine变体做相同调整
- 可审查性：中
- 审查规则建议：Init函数中变量赋值顺序必须保证被调用函数的依赖在调用前已初始化；同构代码修改一处时同步检查所有变体

### d5c79c44c7 [bugfix] solve dkey occasional issue
- 根因类别：硬件同步事件类型错误
- 涉及文件：attention/dense_lightning_indexer_grad_kl_loss/op_kernel/dense_lightning_indexer_grad_kl_loss_vector2.h
- 缺陷描述：ProcessVectorDk()中保护MTE2搬入和MTE3搬出的同步事件用了V_MTE2(向量到MTE2)，实际数据流是MTE3搬出->MTE2搬入下一轮，应用MTE3_MTE2。SetFlag位置也从Cast后移到MTE3 DataCopy后
- 修复模式：V_MTE2替换为MTE3_MTE2；调整SetFlag位置
- 可审查性：低
- 审查规则建议：审查时画数据流图确认Event类型匹配生产者-消费者关系；ping-pong双缓冲同步建立模板

### 094362a49a moe_finalize_routing_v2 db fix
- 根因类别：double buffer条件分支逻辑错误
- 涉及文件：moe/moe_finalize_routing_v2/op_kernel/moe_finalize_routing_v2_fp_cuth_k2.h, moe_finalize_routing_v2_fp_cuth_k4.h
- 缺陷描述：k4文件中bias的Add操作被错误放在两个double buffer分支的共同else里，导致Db0/Db2的bias只有在Db1/Db3不是INVALID_ROW_INDEX时才被加上；同时k2中ISBIASEXIST=false时有无意义的Adds(x,x,0)
- 修复模式：将bias Add从共享else拆分到每个buffer各自的else分支；删除无效加0操作
- 可审查性：中
- 审查规则建议：多路double buffer逻辑审查要验证每条buffer通路操作是否独立；无操作分支(Adds(x,x,0))应标记可疑

### 7da919d4a2 [mc2]matmulAllReduce fix mxfp
- 根因类别：新数据类型场景条件覆盖不全
- 涉及文件：mc2/matmul_all_reduce/op_host/op_tiling/arch35/quant_matmul_all_reduce_tiling_910_95.cpp, mc2/matmul_all_reduce/op_host/op_tiling/matmul_all_reduce_tiling_base.cpp
- 缺陷描述：matmulAllReduce tiling处理3维输入时只对perBlock量化在batch>1且m非128对齐时保持3维，但MXFP8/MXFP4的量化参数shape结构不同，合batch轴后shape无法正确变换，MXFP场景batch>1必须保持3维
- 修复模式：在perBlock条件分支中追加MXFP8/MXFP4在batch>1时的判断条件
- 可审查性：中
- 审查规则建议：新增量化类型时逐一排查所有涉及shape变换和tiling决策的代码路径；建立"量化类型×shape场景"覆盖矩阵

### 3beec5f90e aclnnGroupedMatmulWeightNz md fix
- 纯文档修改，已排除。修正md示例代码中API名称从V5复制过来遗留的名称不一致。

### 70423c4b18 修复TND causal场景下s1小于s2的确定性精度问题
- 根因类别：数学公式隐含假设未校验(假设s1>=s2)
- 涉及文件：attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp, attention/flash_attention_score_grad/op_kernel/arch35/deter.h
- 缺陷描述：确定性FA反向传播的坐标映射公式round_batch=(2m-n+1)*n/2隐含m>=n(即s1>=s2)，当s1<s2时2m-n+1为负数导致坐标计算错误。left_up_causal场景m<n时超出m范围的n无有效计算块
- 修复模式：新增UpdataCausalMN()做n=min(m,n)裁剪；host和kernel两端约10处插入裁剪逻辑
- 可审查性：低
- 审查规则建议：数学公式隐含假设必须在代码中用assert或显式校验；坐标映射代码建立boundary case测试矩阵覆盖s1<s2/s1==s2/s1>s2

### 81898f83c6 fix-debug combine layered
- 根因类别：计算逻辑错误(偏移量重复扣减)
- 涉及文件：mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_a2_layered.h
- 缺陷描述：计算maxBsInRankSizeOnIpc时ipcSliceSize已经是IPC数据区可用大小，又多减了一次IPC_DATA_OFFSET，导致可用空间被低估
- 修复模式：去掉多余的偏移量减法
- 可审查性：低
- 审查规则建议：共享内存布局计算增加断言验证偏移量/大小不超出buffer边界；IPC内存布局UT覆盖边界值场景

### 7ac3c7b2b5 fix intercept
- 根因类别：输入校验缺失
- 涉及文件：gmm/grouped_matmul_swiglu_quant_v2/op_host/下多个文件
- 缺陷描述：缺少多项输入合法性校验：x和weight数据类型不一致(fp4/fp8)未拦截；MXFP4场景K=2不支持未校验；MXFP4输出FP4时N>=4且为偶数未校验
- 修复模式：新增CheckDims()方法；CheckDtype()增加类型一致性校验；新增3个UT
- 可审查性：高
- 审查规则建议：新增数据类型组合时review checklist包含host侧是否增加对应校验拦截；维度约束应有文档化约束表与代码检查一一对应

### ae5a1e33fb 解决MoeInitRoutingQuantV2算子在某个用例的精度问题
- 根因类别：对齐长度与实际长度混用
- 涉及文件：moe/moe_init_routing_quant_v2/op_kernel/arch35/moe_v2_gather_quant_simt.h
- 缺陷描述：bf16->float的Cast操作错误使用Align对齐后的elements而非原始colsTileLength_，多处理了padding区域的无效值，经后续乘加运算产生精度偏差
- 修复模式：第一个Cast长度参数从elements改为colsTileLength_
- 可审查性：低
- 审查规则建议：涉及Align操作的代码段区分alignedLength和actualLength使用场景；变量命名明确区分；量化算子增加边界case精度回归测试

### c71a40a64f 暂时关闭opapi ut，待框架修复后恢复
- 根因类别：外部依赖问题临时规避
- 涉及文件：mc2/下约20个算子的tests/ut/op_api/CMakeLists.txt, tests/test_config.yaml
- 缺陷描述：框架侧打桩机制问题导致mc2下所有op_api层UT挂掉，非算子本身bug
- 修复模式：注释掉约20个CMakeLists.txt的UT编译指令
- 可审查性：高
- 审查规则建议：任何注释掉UT的PR必须关联恢复issue并设deadline；大面积关闭UT属高风险操作需专门审批流程

### 84aa954c73 fix CheckFeatureLse function
- 根因类别：约束条件错误删除/原有校验代码本身有bug
- 涉及文件：attention/fused_infer_attention_score/op_host/arch32/fused_infer_attention_score_tiling_check_feature.cpp
- 缺陷描述：CheckFeatureLse()中ROPE_SPLIT+vHeadDim=512场景的layout限制被整段删除。原代码中"BNSD_NBSD, TND_NTD"被写成单个字符串(逗号在引号内)，导致拦截从未生效
- 修复模式：直接删除整个约束代码块(11行)
- 可审查性：中
- 审查规则建议：删除校验/拦截逻辑的PR需说明原因；字符串匹配的枚举校验注意逗号是否在引号内

### 5d1fe7fba8 空tensor修复lse和错误拦截
- 根因类别：边界条件缺失 / 空输入场景未处理
- 涉及文件：attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_v2.cpp, attention/incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp, incre_flash_attention_tiling_v2.h
- 缺陷描述：q和attention output都是空tensor(shape size为0)时，softmax LSE的shape校验仍执行导致误报GRAPH_FAILED。CheckEmptyTensor接受外部loopTimes参数但应由kCache/vCache自身size决定循环次数，且未做nullptr检查直接解引用
- 修复模式：增加qOutEmptyTensor标志位包裹LSE校验；CheckEmptyTensor改用容器size并增加nullptr检查；CheckLse开头增加emptyTensor_提前返回
- 可审查性：中
- 审查规则建议：shape校验逻辑须覆盖空tensor/零size边界case；函数循环次数应与容器size一致；解引用前必须做nullptr检查

### 6e91c04253 fix special expert and elastic
- 根因类别：变量别名/值被运行时覆写导致语义错误
- 涉及文件：mc2/moe_distribute_combine_add_rms_norm/op_kernel/moe_distribute_combine_add_rms_norm.h, mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2.h
- 缺陷描述：moeExpertNum_在Init阶段赋值后被elasticInst_.InitElasticInfo以引用方式传入并覆写为弹性调度后的值，但后续MaskSpecialExpert和LocalWindowCombine中判断expert类型边界时需要原始值，导致special expert边界判断错误
- 修复模式：新增moeExpertOriginalNum_成员变量在Init时保存原始值，所有special expert边界判断处替换为moeExpertOriginalNum_
- 可审查性：中
- 审查规则建议：被引用传递修改的成员变量，如后续逻辑仍使用，需确认使用的是修改前还是修改后的语义；引用传递参数的副作用须在review时明确标注

### b7a0e9fced bugfixfia算子tnd+gqa+nomask精度error
- 根因类别：变量未初始化(条件分支遗漏赋值) + 无符号整数下溢
- 涉及文件：attention/fused_infer_attention_score/op_kernel/flash_attention_regular.h
- 缺陷描述：(1) kvSLoopNumTotal仅在if(maskType!=0 && sparseMode!=4)分支赋值，else分支(tnd+nomask)保持初始值0，后续0-1U下溢为UINT32_MAX导致条件永假，stackSeqTile不更新致精度错误；(2) kvSIdx-preKVNum在kvSIdx<preKVNum时无符号下溢
- 修复模式：else分支补充赋值；差值比较改为加法比较避免下溢
- 可审查性：高
- 审查规则建议：if/else中变量仅一个分支赋值但后续公共路径使用必须所有分支赋值；uint32_t减法须确保被减数>=减数否则改用加法等价形式

### f1fa74193b revert example
- 根因类别：配置变更回退
- 涉及文件：tests/test_config.yaml
- 缺陷描述：为grouped_matmul_swiglu_quant等3个gmm算子关闭example测试，回退之前添加的有问题的example测试
- 修复模式：yaml中添加test: examples: False
- 可审查性：低
- 审查规则建议：新增example测试应在合入前通过CI验证

### 73b5c51266 fix SetScheduleMode
- 根因类别：API调用遗漏
- 涉及文件：gmm/quant_grouped_matmul_dequant/op_host/quant_grouped_matmul_dequant_tiling.cpp, .h
- 缺陷描述：TilingForQuantGroupedMatmulDequant在调用runTiling前遗漏了context->SetScheduleMode(BATCH_MODE)调用，导致Grouped类算子使用默认调度模式
- 修复模式：runTiling前增加SetScheduleMode(BATCH_MODE)调用，头文件增加BATCH_MODE常量定义
- 可审查性：中
- 审查规则建议：Grouped/Batch类算子的tiling入口函数须在runTiling前调用SetScheduleMode

### c22d7c92ad fix matmulAllReduce UT
- 根因类别：UT与产品代码不同步(namespace变更未传播)
- 涉及文件：mc2模块下matmul_all_reduce和matmul_all_reduce_add_rms_norm的kernel/UT/tiling头文件，tests/test_config.yaml
- 缺陷描述：产品代码将tiling结构体迁移到Mc2Tiling命名空间，UT未同步适配致编译失败。UT中重复定义产品结构体、REGISTER_TILING_DEFAULT缺少namespace前缀、include路径在UT环境不正确、断言方向写反、yaml中ut:False掩盖编译问题
- 修复模式：删除UT重复定义改为include产品头文件；加namespace前缀；条件编译切换include路径；修正断言方向；移除ut:False
- 可审查性：高
- 审查规则建议：禁止UT重复定义产品结构体应直接include；namespace变更时CI须确保UT编译通过；禁止长期ut:False跳过UT

### d47678892d 修正opapi用例
- 根因类别：UT用例与算子接口定义不匹配
- 涉及文件：mc2/all_gather_matmul_v2/tests/ut/op_api/, mc2/matmul_reduce_scatter_v2/tests/ut/op_api/
- 缺陷描述：(1) 输出参数含多余amaxOut应传nullptr；(2) 缺少SetPlatformSocVersion调用；(3) tensor shape维度过小不满足最小要求(至少256)；(4) 负面测试断言方向写反(EXPECT_EQ应为EXPECT_NE)
- 修复模式：移除多余参数；添加平台版本设置；修正shape维度；修正断言方向
- 可审查性：高
- 审查规则建议：UT参数列表须与算子接口严格一致；负面测试须断言返回非SUCCESS；op_api UT的SetUpTestCase应显式设置目标平台

### 1886e6be44 fix RowInvalid bug
- 根因类别：条件分支逻辑错误(MLA场景覆盖不全)
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_block_vec_base.h
- 缺陷描述：RowInvalid函数中if-else if结构仅处理isMlaNoQuant场景，遗漏isMlaFullQuant场景，导致MLA全量化时误入非MLA路径的return分支，行无效化处理被跳过
- 修复模式：重构为先判断非MLA路径再独立判断MLA路径，两路径return条件彻底分离
- 可审查性：低
- 审查规则建议：FlashAttention核心计算路径(特别是MLA变体分支)强制双人review；多模式函数优先使用早期互斥分支避免else-if链条件遗漏

### 3b689fd480 revert-gmm-example
- 根因类别：测试配置回退
- 涉及文件：tests/test_config.yaml
- 缺陷描述：grouped_matmul和rope_with_sin_cos_cache的example测试需要禁用
- 修复模式：yaml中追加test: examples: False
- 可审查性：低
- 审查规则建议：测试配置变更PR须说明影响范围

### f6a3e574d6 cmake修复 - moe_distribute_combine_add_rms_norm
- 根因类别：构建依赖遗漏
- 涉及文件：mc2/moe_distribute_combine_add_rms_norm/op_host/CMakeLists.txt, scripts/ci/ascend910b/ops_transformer_operator_list.yaml
- 缺陷描述：CMakeLists依赖列表漏掉mc2/moe_distribute_dispatch_v2，导致编译找不到dispatch_v2的符号
- 修复模式：补充依赖项；暂时从CI算子列表移除该算子
- 可审查性：中
- 审查规则建议：同时依赖xxx和xxx_v2的combine版本时检查是否也需要dispatch_v2

### f8efb77ac7 fix opapi ut bugs
- 根因类别：返回值吞没 + 多芯片兼容性缺失
- 涉及文件：tests/ut/framework_normal/op_api/CMakeLists.txt, scripts/clean_opapi_stub.py, scripts/generate_opapi_stub.py
- 缺陷描述：(1) bash -c中UT执行和清理命令串联，UT退出码被清理脚本退出码覆盖，CI无法检测UT失败；(2) stub脚本只处理当前芯片配置目录，缺少其他芯片型号的symlink
- 修复模式：显式保存并返回UT进程退出码；添加多芯片配置目录symlink创建/清理
- 可审查性：高
- 审查规则建议：bash -c执行多命令时须检查关键命令退出码是否正确传递；用;连接命令最终退出码只反映最后一条

### faaa5e9686 Bugfix: A2 dispatch & combine Add SetBlockDim for gentask
- 根因类别：任务配置遗漏(缺少SetBlockDim)
- 涉及文件：mc2/common/src/mc2_moe_gen_task_ops_utils.cpp
- 缺陷描述：A2芯片上MoeDistributeDispatch/Combine及V2变体创建aicpu task时未调用SetBlockDim，导致使用默认block dim在A2上行为不正确
- 修复模式：新增NEED_SET_BLOCK_SET集合，创建aicpu task后检查并调用SetBlockDim(从节点属性读取，默认6)
- 可审查性：中
- 审查规则建议：新增算子类型/平台时checklist须包含是否需配置block dim；GenTask中创建aicpu task后检查是否遗漏平台相关配置

### f14ef205d8 修复TND sparseMode=3场景性能劣化
- 根因类别：枚举场景覆盖不全(新增sparseMode未处理)
- 涉及文件：attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_varlen.cpp, attention/flash_attention_score/op_kernel/arch35/flash_attention_score_kernel_train.h
- 缺陷描述：GetS2RealSize缺少RIGHT_DOWN_CAUSAL(sparseMode=3)处理分支，且CAUSAL分支多余s1Size==s2Size限制导致s1!=s2时不生效。host和kernel两侧都未对齐添加新枚举处理
- 修复模式：CAUSAL去掉多余条件；新增RIGHT_DOWN_CAUSAL分支计算公式；kernel侧同步新增对应分支
- 可审查性：高
- 审查规则建议：枚举值新增时检查所有switch/if-else链是否都处理了新值；可lint规则：枚举分支判断若不含default/exhaustive处理则报警

### 67a5e12c47 fix moe_token_unpermute_with_routing_map_grad mix prob
- 根因类别：混合精度处理错误 + 硬件同步事件错配 + sizeof跨平台问题
- 涉及文件：moe/moe_token_unpermute_with_routing_map_grad/op_host/tiling.cpp, op_kernel/base.h, prob_not_none_drop_pad_true.h
- 缺陷描述：(1) sizeof(float)在host/device可能不一致，应用显式常量SIZE_OF_FLOAT=4；size_t与int64_t混合运算隐式转换；(2) prob梯度用标量GetValue+手动cast应改为向量化Copy/Cast指令；(3) 同步原语V->S->MTE3应为V->MTE3，且需补充MTE3->V反向同步
- 修复模式：常量类型统一int64_t+显式SIZE_OF_FLOAT；标量操作替换为向量化指令；同步原语修正+补充反向同步
- 可审查性：中
- 审查规则建议：kernel同步原语须匹配实际producer-consumer数据流；tiling代码避免sizeof用显式常量；标量GetValue+手动cast标记为code smell

### d2cbbb763c enable overflow mode
- 根因类别：浮点溢出模式未设置 + 死代码
- 涉及文件：gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_apt.cpp, gmm/common/cgmct/epilogue/block_epilogue_swiglu_mx_quant.h
- 缺陷描述：(1) SwiGLU+MX量化kernel未设SPR60溢出模式控制寄存器导致NaN/Inf；(2) if constexpr(IsSameType<bfloat16_t,half>)恒假分支为死代码
- 修复模式：kernel入口save-set-restore SPR60三步式管理；删除恒假if constexpr分支和未使用变量
- 可审查性：中
- 审查规则建议：MX/FP8量化代码检查是否正确设置overflow模式(SPR60)；if constexpr中IsSameType恒假/恒真应被lint检测

### f3345d80 [FAG] fp8 datacopypad fix
- 根因类别：硬件指令参数类型溢出 + API参数约束不匹配
- 涉及文件：attention/common/op_kernel/matmul.h, attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_block_vec.h
- 缺陷描述：FP8场景两处缺陷。(1) LoadData从L1到L0A搬运时mStep较大时单次LoadData无法支持，需循环拆分(按l0bLoop=(mStep+1)>>1分批)，修复前直接一次性调用导致硬件行为未定义。(2) DataCopyPad的srcStride和dstBlockStride参数为uint16_t，当transpose_stride值较大时16位宽截断溢出，改为uint32_t并配合ReinterpretCast<uint8_t>匹配字节级搬运语义
- 修复模式：循环拆分(LoadData) + 参数类型提升uint16->uint32(DataCopyPad)
- 可审查性：中
- 审查规则建议：对硬件搬运指令参数检查stride/step是否存在uint16_t截断风险，凡涉及sizeof(T)*dimension的表达式作为uint16_t参数传递时标记可疑

### 31044c91 fix DTS (vec_config配置兼容)
- 根因类别：tiling key编码不完整 / 模板参数维度缺失
- 涉及文件：mc2/matmul_all_reduce/op_host/op_tiling/arch35/weight_quant_matmul_all_reduce_tiling_910_95.cpp/.h, mc2/matmul_all_reduce/op_kernel/matmul_all_reduce_apt.cpp, mc2/matmul_all_reduce/op_kernel/matmul_all_reduce_apt_tiling_key.h
- 缺陷描述：tiling key只编码了ANTIQUANT_TYPE/QUANTTYPE/HAS_ANTIQUANT_OFFSET/BAIS_IS_FP32/WEIGHTFORMAT，未包含mte2Config_(MTE2 inner size / buf num配置)信息，导致不同vec_config配置映射到同一tiling key，kernel侧无法区分。修复新增TEMPLATE_CUSTOM模板参数(4-bit宽，5种取值)编码进tiling key
- 修复模式：扩展tiling key维度以区分硬件配置
- 可审查性：低
- 审查规则建议：新增影响kernel行为的配置字段时应有checklist确认是否需要编入tiling key；host侧GetTilingKey()生成的key必须能在kernel侧注册表中找到匹配

### 52fd17df NsaCompressWithCache拦截错误码修复
- 根因类别：错误码语义误用
- 涉及文件：attention/nsa_compress_with_cache/op_host/op_api/aclnn_nsa_compress_with_cache.cpp, docs
- 缺陷描述：空tensor输入时返回ACLNN_ERR_INNER_TILING_ERROR(561002内部tiling错误)，但空tensor本质是参数校验失败应返回ACLNN_ERR_PARAM_INVALID(161002参数无效)。错误码选择错误导致上层框架误归类和用户排查成本增加
- 修复模式：错误码替换(INNER_TILING_ERROR -> PARAM_INVALID)
- 可审查性：高
- 审查规则建议：参数校验阶段只允许返回PARAM_INVALID或INNER_NULLPTR类错误码，INNER_TILING_ERROR仅允许在实际tiling计算逻辑中使用

### ca4ed1e3 修复GQA非量化支持prefix分核计算逻辑
- 根因类别：计算逻辑遗漏 / 分段取整数学错误
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：GQA支持prefix场景后，prefix和origin的KV序列分开切块，但innerBlockNums仍把actualSeqLengthKV+actualSharedPrefixLen作为整体做一次ceil除法。实际应分别对两段各自ceil后相加(ceil(a+b) != ceil(a)+ceil(b))。调用处也不应预先求和
- 修复模式：修改计算公式使其与分块语义一致，单次ceil(a+b)改为ceil(a)+ceil(b)
- 可审查性：中
- 审查规则建议：当存在分段切块语义时，检查块数计算是否对每段分别做向上取整；函数调用处和函数体内部的参数含义应保持一致

### e7e6e543 修复数据同步问题
- 根因类别：硬件同步屏障缺失
- 涉及文件：moe/moe_gating_top_k/op_kernel/arch35/moe_gating_top_k_regbase.h
- 缺陷描述：SelectTopKGroupIndex函数中循环通过DataCopy向outputAddr写入(VEC_STORE)，循环结束后紧接Duplicate填充padding值写入同一buffer。无显式LocalMemBar导致后续写入可能在前一批未落地时就开始，产生WAW冲突
- 修复模式：在两组异步写操作之间插入LocalMemBar<VEC_STORE, VEC_STORE>
- 可审查性：低
- 审查规则建议：同一段local memory先后被两组VEC_STORE写入时，检查是否存在LocalMemBar同步，特别是循环体内DataCopy结束后紧接非循环的DataCopy/Duplicate写同一buffer的模式

### 514caa09 新算子原型修正
- 根因类别：算子接口定义错误
- 涉及文件：mc2/allto_all_matmul/op_host/allto_all_matmul_def.cpp, mc2/allto_all_matmul/op_host/config/ascend910_95/allto_all_matmul_binary.json, mc2/matmul_allto_all/op_graph/matmul_allto_all_proto.h, mc2/matmul_allto_all/op_host/matmul_allto_all_def.cpp
- 缺陷描述：(1) all2all_out输出定义为REQUIRED但实际并非总需要，应为OPTIONAL。(2) world_size属性声明REQUIRED但给了默认值-1(.Int(-1))，语义矛盾——必选属性不应有默认值，用户可不传获得无效值-1绕过约束
- 修复模式：修正REQUIRED/OPTIONAL属性和默认值与语义一致
- 可审查性：高
- 审查规则建议：REQUIRED属性不应携带默认值；输出参数的REQUIRED/OPTIONAL应与实际使用场景匹配

### 2b381c57 fix int4 quant MoeV2GatherDynamicQuant
- 根因类别：buffer误用 / 复制粘贴错误
- 涉及文件：moe/moe_init_routing_quant_v2/op_kernel/moe_v2_gather_dynamic_quant.h
- 缺陷描述：Init函数int4量化分支中依次分配tempScaleBuf/maxValueBuf/mulBuf三个独立buffer。constScaleTensor从tempScaleBuf获取正确，maxValueTensor从maxValueBuf获取正确，但mulTensor错误地也从tempScaleBuf.Get<float>()获取而非mulBuf。导致mulTensor和constScaleTensor指向同一内存互相覆盖
- 修复模式：将tempScaleBuf.Get改为mulBuf.Get
- 可审查性：高
- 审查规则建议：连续多行xxxTensor=xxxBuf.Get<T>()时，检查tensor是否对应同名buffer；同一buffer被Get两次赋给不同变量时告警

### ef78c18f TND seq相等场景确定性问题修复
- 根因类别：条件逻辑缺陷(布尔表达式语义错误)
- 涉及文件：attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp
- 缺陷描述：原条件(layoutType==TND || isAllSame)在isAllSame==true且isDeterministic==true时也进入BN2S2分支，但该分支不支持确定性计算。修复改为(layoutType==TND || isAllSame && !isDeterministic)
- 修复模式：收窄条件分支进入条件，对特定场景增加排除约束
- 可审查性：低
- 审查规则建议：涉及deterministic flag的条件分支变更要覆盖所有布局类型x确定性开关组合；混合||和&&不加括号的复合条件发出可读性警告

### 06f330f9 修复行无效+不传mask的异常拦截
- 根因类别：空指针校验缺失
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：CheckMaskCrossover函数检查FP16 mask不支持行无效校正时直接访问maskDataType，未先检查attentionMask是否为nullptr。不传mask时maskDataType来源不确定，可能误匹配FP16触发错误拦截
- 修复模式：前置空指针守卫(null guard)，利用短路求值跳过不适用的校验
- 可审查性：中
- 审查规则建议：对可选输入tensor属性的访问必须先校验tensor指针非空

### 0f363800 fix ifa and splitfuse interception for TND
- 根因类别：返回值忽略 + 对称校验遗漏
- 涉及文件：attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp, attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp
- 缺陷描述：(1) CheckFAIQKV中调用CheckFAIIsTND()后完全忽略返回值，即使返回GRAPH_FAILED也继续返回GRAPH_SUCCESS，导致TND下非法输入未被拦截。(2) QKVPreProcess4TND中只校验Q的actual sequence length与T(query)一致，未校验KV的actual sequence length与T(key)一致
- 修复模式：检查并传播子函数返回值；补全QKV对称校验
- 可审查性：中(返回值忽略) / 低(对称校验)
- 审查规则建议：返回ge::graphStatus的函数调用强制要求检查返回值(类似nodiscard)；对Q/K/V某一个做维度校验时确认其余是否需同样校验

### 12b3902e mc2 a5 gentask fix
- 根因类别：映射表注册遗漏 + 常量命名不一致
- 涉及文件：mc2/common/src/mc2_gen_task_ops_utils_arch35.cpp
- 缺陷描述：GROUP_INFO_MAP_ARCH35映射表中缺少MatmulAllReduce算子类型的注册，导致该算子在A5架构上无法获取通信域task信息。同时常量名缺少V2后缀与实际算子类型名不一致
- 修复模式：补全映射表条目，重命名常量对齐V2语义
- 可审查性：中
- 审查规则建议：新增算子类型时检查所有包含算子类型映射表的文件是否同步更新

### 967dc86e 修复SFAG潜在问题
- 根因类别：多缺陷合集(pipe类型错误 + 尾块越界 + 整数溢出)
- 涉及文件：attention/sparse_flash_attention_grad/basic_modules/vec_op.h, attention/sparse_flash_attention_grad/op_kernel/sparse_flash_attention_grad_bs1_basic.h
- 缺陷描述：(1) CrossCoreWaitFlag的pipe类型误用PIPE_FIX应为PIPE_MTE2，C核等待V核时指定错误流水线导致同步可能提前通过。(2) GatherKV尾块搬运时blockLen仍用selectedBlockSize而非实际尾块大小导致DDR越界。(3) CeilDiv当参数超过65535时内部类型溢出导致结果变0，改为int64_t手动计算
- 修复模式：修正枚举值 + 尾块特殊分支 + 避免CeilDiv溢出
- 可审查性：低
- 审查规则建议：CrossCoreWaitFlag/SetFlag的pipe类型与实际数据搬运引擎一致性审查；循环DMA搬运强制审查尾块处理；CeilDiv参数可能超16位时审查内部实现

### b9ccd936 修复空tensor场景TND用例被错误拦截
- 根因类别：边界条件校验缺失(空tensor合法场景)
- 涉及文件：attention/incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp
- 缺陷描述：ProcessBaseTensors中TND layout下校验kv的T维度时直接判断tOfkv!=actualSeqLastSize即报错，但kv为空tensor(tOfkv==0)是合法输入不应拦截。校验headDim时空tensor也会误判
- 修复模式：在校验条件中增加对零值(空tensor)的豁免判断
- 可审查性：高
- 审查规则建议：tensor维度/shape校验逻辑必须显式考虑空tensor是否为合法输入

### 6a906f9f [FAG] get qkvStartIdx fix and sparse5+null prefix fix
- 根因类别：early return截断独立逻辑 + nullable输入fallback缺失
- 涉及文件：attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp, docs
- 缺陷描述：(1) SetQKVStartIdx()中获取qStartIdx过程中任何一步失败都直接return，导致后续kvStartIdx的获取逻辑被完全跳过。用户不传qStartIdx时kvStartIdx永远为默认值0。(2) sparseMode=5且prefixN为nullptr时缺少降级为ALL_MASK的fallback处理
- 修复模式：将串行early-return重构为独立if-block；新增sparseMode降级fallback
- 可审查性：高
- 审查规则建议：多个独立可选参数的读取禁止使用early return模式；所有nullable输入tensor为nullptr时必须有明确fallback路径

### 5d311e60 修复cleancode问题
- 根因类别：代码规范(const correctness)
- 涉及文件：attention/common/op_host/arch32/fia_tiling_nonquant.cpp/.h
- 缺陷描述：GetSafeActToken方法不修改成员变量但缺少const修饰符
- 修复模式：添加const修饰符
- 可审查性：高
- 审查规则建议：启用clang-tidy readability-make-member-function-const检查

### e9ec0142 修复mmallreduce的精度fail的问题
- 根因类别：编译选项缺失 / 构建配置错误
- 涉及文件：mc2/matmul_all_reduce/op_host/CMakeLists.txt
- 缺陷描述：ascend910_95平台编译时缺少--cce-auto-sync=off和-mllvm -cce-aicore-dcci-before-kernel-end=false，导致编译器自动插入不必要的同步/缓存清理指令破坏数据一致性时序产生精度问题。CMake条件判断中ASCEND_COMPUTE_UNIT匹配存在分号残留问题
- 修复模式：补充编译选项 + 清理CMake条件判断
- 可审查性：低
- 审查规则建议：为每个平台建立编译选项checklist，算子集成测试加入精度回归测试

### 7c31bb71 fix: opensource, fix fia/ifa warnings
- 根因类别：未使用变量 + 变量名遮蔽(shadowing)
- 涉及文件：attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_v2.cpp, attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp, attention/incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp
- 缺陷描述：(1) FIA两个文件中定义了MAX_BLOCK_SIZE和FROM_FUSED_FLAG局部变量但从未使用。(2) IFA文件中函数参数名prefixSSize_和ifaContext_使用下划线后缀与类成员变量同名造成遮蔽
- 修复模式：删除未使用变量；重命名参数消除遮蔽
- 可审查性：高
- 审查规则建议：开启-Wunused-variable和-Wshadow警告作为CI门禁

### 3edff194 [FAG] Fix the bug in Gm that was not cleaned up
- 根因类别：稀疏计算分支逻辑遗漏 + GM残留数据未清零
- 涉及文件：attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp/.h, tests
- 缺陷描述：(1) DoBn2MultiBlkSparse中TND格式处理被嵌套在isSparse条件内部，但TND不论是否稀疏都需要调用GetBlockInfoOfTNDForBn2，非稀疏TND场景会跳过TND专用逻辑。(2) 稀疏attention中无效行/列(由稀疏mask导致某些block无计算核覆盖)的GM梯度数据不会被写入但残留脏数据。修复新增isInvalidRow/isInvalidCol检测，检测到时降级tiling策略并执行GM清零
- 修复模式：调整条件分支嵌套顺序 + 新增无效行/列检测 + 降级tiling策略
- 可审查性：中
- 审查规则建议：稀疏计算tiling需关注"块有效性"，确保所有输出GM区域都有写入路径覆盖

### 0166982e nsa_selected_attention_infer的BSND_case报错修复
- 根因类别：Layout条件缺失(非TND格式执行了TND专属逻辑)
- 涉及文件：attention/nsa_selected_attention_infer/op_kernel/nsa_selected_attention_infer.h
- 缺陷描述：GetBatchInfo中累加前序batch Q序列长度的循环对所有layout格式执行，但该偏移量只在TND格式有意义。BSND格式下执行此累加得到错误的非零偏移量导致GM访问越界
- 修复模式：用if constexpr (LAYOUT_T == LAYOUT::TND)限制为仅TND执行
- 可审查性：高
- 审查规则建议：kernel模板支持多种layout时，所有与数据寻址/偏移计算相关的代码必须审查layout适用性

### 47e4b7c2 修复SFAG尾块未选场景精度问题
- 根因类别：尾块长度计算未区分"是否被选中"
- 涉及文件：attention/sparse_flash_attention_grad/basic_modules/vec_op.h, attention/sparse_flash_attention_grad/op_kernel/sparse_flash_attention_grad_bs1_basic.h
- 缺陷描述：(1) lastBlockSize始终使用curMaxS2%selectedBlockSize，但当尾块未被topk选中时所有被选block都是完整大小，lastBlockSize应为selectedBlockSize。(2) CalSoftmax/CalSoftmaxGrad中actualSelS2未考虑包含尾块时实际数据量应减少，导致读取垃圾数据参与softmax归一化
- 修复模式：新增isLastBlockSelected标志，lastBlockSize和actualSelS2加入尾块选中条件判断
- 可审查性：中
- 审查规则建议：分块+topk选择的算子必须审查尾块在"被选中"和"未被选中"两种场景下的处理逻辑

### 025c1734 antiquant fix aicore error due to large s2
- 根因类别：整数类型溢出/截断(uint16)
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_antiquant_kernel.h, attention/common/op_kernel/arch35/flash_attention_score_antiquant_processor.h
- 缺陷描述：(1) CeilDivision函数AIV端返回uint16，大s2时sInnerLoopSize计算结果超65535被截断。(2) AlignUp32参数和返回值都是uint16，大值截断。本质相同：选用了返回值uint16的工具函数未考虑大shape溢出
- 修复模式：替换为返回uint64的等价函数(CeilDivision->CeilDiv)；扩大辅助函数参数类型
- 可审查性：高
- 审查规则建议：kernel代码中CeilDivision(AIV端返回uint16)当输入可能超65535时应使用CeilDiv；检测uint16函数但输入来源为uint64的类型收窄

### 8ac086d8 fix aicerr due to large s2
- 根因类别：整数溢出 + workspace地址偏移计算错误(off-by-one)
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_block_vec_infer.h, attention/common/op_kernel/arch35/flash_attention_score_kernel_infer.h
- 缺陷描述：(1) 同025c1734，CeilDivision返回uint16溢出。(2) workspace回退基地址时InitCommonGlobalBuffer内部已将workspace前移了totalOffset+mm2Offset*3+ve2Offset*3，但原代码只回退aicIdx倍偏移少了一个singleCoreOffset，修复改为(aicIdx+1)
- 修复模式：CeilDivision->CeilDiv + 偏移因子修正aicIdx->(aicIdx+1)
- 可审查性：中
- 审查规则建议：接收已被偏移过的指针并回退到基地址时应有注释标明偏移量完整来源，建议用单一baseAddress变量

### 6b43f57c gmm fix 1,1 input in v1
- 根因类别：平台兼容性条件缺失
- 涉及文件：gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul.cpp
- 缺陷描述：V1版GroupedMatmul中NO_SPLIT模式无条件禁止weight转置，但Ascend910_95平台上该组合合法。CHECK_COND只检查"V1&&transposeWeight"就报错，缺少对910_95平台的例外判断
- 修复模式：在参数校验否定条件中增加平台白名单豁免
- 可审查性：低
- 审查规则建议：参数校验涉及硬件能力限制时显式列出所有受限/豁免平台；新平台上线时checklist回顾所有硬编码平台限制

### 014e93e7 Revert "MoeInitRoutingV2 & MoeFinalizeRoutingV2性能优化PR3935"
- 根因类别：性能优化引入功能回退(多处tiling/kernel不一致)
- 涉及文件：moe/moe_finalize_routing_v2/op_host/...tiling_arch35.cpp, moe/moe_finalize_routing_v2/op_kernel/arch35/...k_h_full_load.h, moe/moe_init_routing_v2/op_host/...tiling.cpp, moe/moe_init_routing_v2/op_kernel/arch35/...gather_out_for_simt.h, ...sort_multi_core.h
- 缺陷描述：被revert的PR3935引入多处问题：(1) K=1特化路径中UB内存估算tiling/kernel不一致 + SetOffsetOfExpertIdx参数顺序交换导致bias偏移错误。(2) GatherOut buffer从1改2但kernel侧同步机制未匹配。(3) Sort中SetWaitFlag从MTE3_S误改为MTE2_S导致错误硬件事件等待
- 修复模式：完整revert
- 可审查性：中
- 审查规则建议：tiling/kernel联动修改要求两侧内存估算保持对称；SetWaitFlag事件类型修改重点标注；函数参数顺序变更触发所有调用点核对

### b09cb10f fix rt_kb
- 根因类别：打包安装配置错误
- 涉及文件：scripts/package/module/ascend/OpsTransformer.xml
- 缺陷描述：rtkb打包配置XML缺少install_type="all"属性导致文件在某些安装模式下不部署；install_mod权限设置不正确；entity属性误用
- 修复模式：修正XML配置属性
- 可审查性：中
- 审查规则建议：打包配置变更时检查file_info节点是否包含install_type属性

### 48695f3d 修复pfa_tiling侧代码误改
- 根因类别：复制粘贴/误合入(条件分支外重复赋值)
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：SetShape函数中，对b和s的赋值(b=actualSequenceLengthQ->GetShapeSize(), s=GetMaxSeq)在if分支内已正确赋值，但if块外又无条件重复执行同样两行赋值，导致无论条件是否成立b/s都被覆盖为同一路径的值，使条件分支失去意义
- 修复模式：删除if块外部的重复赋值语句(2行)
- 可审查性：高
- 审查规则建议：检测if/else分支内的赋值在分支外被无条件重复执行的模式(duplicate-assignment-after-branch)

### a350f640 fix spell problem for splitFuse
- 根因类别：标识符拼写错误(namespace typo)
- 涉及文件：attention/fused_infer_attention_score/op_kernel/kernel_common.hpp(定义处), flash_attention_interface.cpp, flash_attention_regular.h, fused_infer_attention_score.cpp(使用处)
- 缺陷描述：namespace FaiKenel(缺字母'r'，应为FaiKernel)在kernel_common.hpp定义，被SplitFuse模块3个文件大量引用(FaiKenel::MaskType、FaiKenel::inputLayout等)
- 修复模式：全局标识符重命名(namespace rename跨4文件)
- 可审查性：中
- 审查规则建议：spell checker/naming convention工具对标识符做拼写检查

### 1d256572 deter rope精度修复
- 根因类别：初始化/赋值遗漏(条件编译守护的字段)
- 涉及文件：attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_kernel_deter.h
- 缺陷描述：SetRunInfoDeter函数中，lastBatchIdx!=bIdx(batch切换)时，缺少对ROPE偏移量的更新——lastBatchTotalS1BRopeOffset和lastBatchTotalS2BRopeOffset未赋值，导致deter模式下ROPE编码使用上一batch残留值，引发精度问题
- 修复模式：在if constexpr (IS_ROPE)守护下新增2行偏移量赋值
- 可审查性：高
- 审查规则建议：if分支内对struct多字段赋值时检查if constexpr守护的相关字段是否遗漏

### e8bdf8f2 vec2 stage bugfix
- 根因类别：复制粘贴错误(函数调用qualifier不匹配)
- 涉及文件：attention/common/op_kernel/arch35/vf/vf_div_cast.h
- 缺陷描述：DivCastImplGeneral函数(通用路径)内部错误调用了DivCastImpl128VF(128专用路径)，应为DivCastImplGeneralVF。General路径可能从128路径复制而来但忘改内部调用
- 修复模式：修正被调函数名(一行)
- 可审查性：高
- 审查规则建议：外层函数名含qualifier(General/Aligned/128等)时，内部调用的同族函数应包含相同qualifier

### f9c949e1 antiq pz_nz 连续点位精度失败
- 根因类别：VF硬件指令POST_MODE_UPDATE与NZ格式不兼容
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_antiquant_processor.h, vf/vf_antiquant_w4.h, vf/vf_antiquant_w8.h
- 缺陷描述：(1) colDstStride计算中(dealRowCount+(dealRowCount%2))*colBaseSize的2行对齐是多余的/错误的。(2) 内层row循环使用POST_MODE_UPDATE(地址自增)模式，但NZ格式下地址不连续，自增导致写入位置错误。(3) srcStride在PA+NZ下从dealRowCount%2改为0
- 修复模式：从"地址自增遍历"改为"显式地址+反向遍历"，修正步长公式，影响5个VF函数
- 可审查性：低
- 审查规则建议：POST_MODE_UPDATE与NZ格式同时使用时应重点审查；VF kernel中自增模式与非连续数据格式组合需警告

### 5c701aee fix AttentionToFFN Addr invalid
- 根因类别：数据源选择错误(通过remote数组获取local地址)
- 涉及文件：mc2/attention_to_ffn/op_kernel/attention_to_ffn.h
- 缺陷描述：SetFlagInAttn()获取selfRankAddr时通过remoteRes[rankId_].nextDevicePtr或userMemRes[rankId_].addr间接获取本rank地址，存在地址无效风险。应直接使用winContext_->localWindowsIn
- 修复模式：将6行条件分支简化为一行直接访问localWindowsIn
- 可审查性：中
- 审查规则建议：访问本地/本rank资源应优先使用local字段，不应通过remoteRes[selfRankId]间接获取

### cfac8ab4 CCU Sqe Num Fix
- 根因类别：task配置参数遗漏
- 涉及文件：mc2/common/src/mc2_a5_gen_task_utils.cpp
- 缺陷描述：CreateCcuFusionTask未设置sqe_num字段。mc2算子需1个AIC+4个CCU共5个SQE，未设置则运行时使用默认值导致SQE不足
- 修复模式：增加ccu_fusion_task.set_sqe_num(5)
- 可审查性：中
- 审查规则建议：创建硬件task时应有checklist确保所有必要字段已设置；对比同类task创建流程发现遗漏

### 91f11e9e fia maskcheck fix
- 根因类别：(1)OP_LOGE后缺return导致错误检查无效 (2)mask检查条件不完整
- 涉及文件：attention/fused_infer_attention_score/op_host/arch32/fused_infer_attention_score_tiling_check_feature.cpp
- 缺陷描述：(1) attenMaskFlag_==false且sparseMode!=NO_MASK时只打OP_LOGE但缺return GRAPH_FAILED，函数继续执行可能访问不存在的mask tensor。(2) 二维mask检查遗漏SPARSE_MODE_ALL_MASK和有rope场景，非法配置未被拦截
- 修复模式：(1)补return (2)扩展条件覆盖更多sparseMode×layout×rope组合
- 可审查性：高(缺陷1)/中(缺陷2)
- 审查规则建议：OP_LOGE后必须跟return GRAPH_FAILED；参数校验应维护合法组合矩阵

### 56ec691c fix-debug, make the send flag different
- 根因类别：UB数据残留导致通信flag误判
- 涉及文件：mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch_a2_layered.h
- 缺陷描述：分层Dispatch中SHOULD_SEND_FLAG_VALUE是固定常量，当前后调用属性不一致时UB中残留旧数据可能与固定flag值重叠，接收端误判flag已到达
- 修复模式：引入时间轮变量magicVal_，flag值改为SHOULD_SEND_FLAG_VALUE+magicVal_避免与旧残留冲突
- 可审查性：中
- 审查规则建议：分布式通信中使用固定magic number做同步标志存在重入/残留风险；检测"常量直接比较用于通信同步"的反模式

### 49d98876 rope问题修复
- 根因类别：多缺陷综合(变量遮蔽+索引计算错误+默认值错误+分支遗漏)
- 涉及文件：attention/flash_attention_score_grad/op_host/arch35/...tiling_s1s2_bn2gs1s2_regbase.cpp/.h, op_kernel/arch35/deter.h, flash_attention_score_grad_kernel_deter.h, flash_attention_score_grad_tiling_data_regbase.h
- 缺陷描述：(1) CalGQADenseIndex缺b_id重计算(b_id=ID%N;b_id=Ceil(b_id,g))。(2) CalcleCausalDeterParam中RIGHT_DOWN_CAUSAL&&m>n时直接m=n丢失跳过行信息，应算mGap。(3) CalGQACausalIndex中局部变量g遮蔽外层参数g(group数)。(4) CalTNDDenseIndex传batchId应传w。(5) isS1S2Same默认true改false。(6) Band稀疏模式下确定性参数计算分支遗漏
- 修复模式：参数传递纠错、变量重命名(g->gTail)、默认值修正、分支补全
- 可审查性：低
- 审查规则建议：-Wshadow捕获变量遮蔽；bool xxx=true形式默认初始化警惕"乐观默认"；单commit修单缺陷便于追溯

### 79af25ff 修复reducescatterv2图模式精度问题
- 根因类别：API调用缺少必要参数
- 涉及文件：mc2/common/new_mc2_mm/kernel/mc2_quant_batch_matmul.h
- 缺陷描述：SetMMParaAndCompute中SetTensorScaleA/SetTensorScaleB在MxType分支缺少ATrans/BTrans转置参数，导致图模式下Scale转置状态丢失
- 修复模式：补齐第二个参数ATrans/BTrans(两行各加一参数)
- 可审查性：高
- 审查规则建议：SetTensorScale系列API检查是否传入转置参数；API签名变更时全量搜索调用点确认参数补齐

### ac141d52 Fix precision issues of static quant on A5 Dispatch
- 根因类别：DataCopy对齐截断导致数据污染
- 涉及文件：mc2/moe_distribute_dispatch/op_kernel/arch35/moe_distribute_dispatch_arch35.h
- 缺陷描述：QuantStatic中DataCopy搬运scalesGT_到scalesLT_时，axisH_非32对齐时DataCopy向下截断，尾部包含UB随机残留值导致int8量化精度异常
- 修复模式：DataCopy替换为DataCopyPad精确搬运axisH_*sizeof(float)字节
- 可审查性：高
- 审查规则建议：DataCopy用于非对齐长度数据搬运时检查是否满足对齐要求，不满足应使用DataCopyPad

### cc37880e [FAG] fp8 fix
- 根因类别：模板特化不支持大尺寸数据块
- 涉及文件：attention/flash_attention_score_grad/op_kernel/arch35/vector_api/vf_cast_transdata_deconflict.h
- 缺陷描述：CastTransdataDeconflict的bfloat16_t分支只处理srcN<=128，FP8场景64x256基本块的srcN=256超出处理范围
- 修复模式：按srcN拆分if constexpr分支——<=128保持原逻辑，<=256新增分支将256宽度拆为左右128子块分别处理
- 可审查性：中
- 审查规则建议：模板函数中数据块尺寸的编译期分支应覆盖所有合法输入范围；else兜底加static_assert防遗漏

### c5c8bacd 修复genop生成目录问题
- 根因类别：工具逻辑缺失(未生成上级CMakeLists.txt)
- 涉及文件：scripts/opgen/opgen_standalone.py(核心), cmake/custom_build.cmake, scripts/opgen/template/CMakeLists.txt(新增模板)
- 缺陷描述：opgen_standalone.py的_copy_template中shutil.copytree只复制算子目录，但新算子分类的父目录缺CMakeLists.txt导致编译系统无法发现新算子
- 修复模式：copytree后增加父目录CMakeLists.txt存在性检查+条件复制
- 可审查性：中
- 审查规则建议：工具脚本创建目录结构时验证所有上游依赖文件的存在性

### 3181c5fd fix gmmfr opapi
- 根因类别：三重错误(多余参数+变量作用域错误+判断对象错误)
- 涉及文件：gmm/grouped_matmul_finalize_routing/op_host/op_api/aclnn_grouped_matmul_finalize_routing.cpp
- 缺陷描述：(1) CheckParams签名多了未使用的executor参数。(2) storageShape声明在if(DT_INT32)块内但SetStorageShape需在if外执行。(3) INT4判断条件用x2->GetDataType()但应用tmpWeight->GetDataType()(x2原始INT32被拆包为INT4)
- 修复模式：删多余参数、变量作用域提升、判断对象修正
- 可审查性：高
- 审查规则建议：未使用参数应产生编译警告；变量声明应在所有使用点的最小公共作用域；类型转换后条件判断应基于转换后对象

### dd8f7a67 修复bmm2 GQA Nd合轴场景误改
- 根因类别：边界条件缺失(GQA合轴+奇数s1)
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_block_cube.h
- 缺陷描述：IterateBmm2L1SplitN和IterateBmm2中fixpipeParams.mSize计算(s1RealSize+1)>>1<<1在BSNGD/TNGD格式且isPfaGS1Merge且s1奇数时不正确，应为s1RealSize+gSize以对齐g维度
- 修复模式：条件分支补全(isInfer编译期分支+format/合轴/奇数条件)
- 可审查性：中
- 审查规则建议：fixpipe参数的mSize计算在合轴场景应有专门case覆盖；奇偶对齐逻辑需边界测试

### a0eb3526 修复流水问题和增加缓冲块性能加强
- 根因类别：多核流水线同步缺陷+缓冲区不足+资源释放顺序错误
- 涉及文件：posembedding/rotary_position_embedding/op_host/rope_rotate_matrix_tiling.cpp, op_kernel/rotate_matrix.h
- 缺陷描述：(1) CV_PARALL_NUM从4改16，原缓冲区并行度不足致AIC写/AIV读workspace数据竞争。(2) blockIdx/cvParall局部变量在BN循环每次重初始化致跨BN状态丢失，改为CVConfig结构体管理。(3) 同步阈值从硬编码改为随缓冲块数量动态调整。(4) CrossCoreSetFlag从PIPE_MTE3改PIPE_MTE2且位置移到DataCopy后。(5) FreeTensor从乘法后移到CopyOut后防止未消费即释放
- 修复模式：状态管理重构+同步原语修正+缓冲区扩容+资源释放顺序修正
- 可审查性：低
- 审查规则建议：多核同步阈值不应硬编码应与缓冲区大小关联；跨迭代状态显式管理；CrossCoreSetFlag pipeline阶段与实际数据传输匹配；FreeTensor在所有消费者完成后调用

### 99b635a3 [FAG] fix some case perf
- 根因类别：分核策略条件不足(大量无效块未触发按块分核)
- 涉及文件：attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp/.h
- 缺陷描述：isSplitByBlockIdx启用条件原只看isExceedL2Cache，但LEFT_UP_CAUSAL模式下s2Outer>s1Outer且(s2Outer-s1Outer)*s1Outer>=3072时存在大量无效计算块，即使不超L2缓存也应启用按块分核
- 修复模式：新增CheckIsLargeInvalidBlk()函数，扩展启用条件为isExceedL2Cache||isLargeInvalidBlk
- 可审查性：高
- 审查规则建议：分核策略启用条件应覆盖所有已知性能退化场景；sparse模式下无效块比例应纳入分核决策

### cad55ec7 修复非FD场景的workspace计算异常
- 根因类别：条件分支缺失——非FlashDecode场景下错误读取FD专属tiling参数
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：GetPFAWorkSpaceSize函数中无论是否处于FlashDecode场景都读取kvSplitPart参数，非FD场景下该值未正确初始化，导致workspace计算异常
- 修复模式：kvSplitPart默认值设为1，仅在enableFlashDecode为true时从tiling数据读取
- 可审查性：高
- 审查规则建议：访问运行模式相关参数前确认当前场景是否适用；tiling参数在所有模式分支下都有合理默认值

### 06ac9d9f all_gather_matmul/matmul_reduce_scatter tilingkey修复
- 根因类别：模板分支覆盖不全——多bool模板参数组合的if-else分支只覆盖部分情况
- 涉及文件：mc2/all_gather_matmul/op_kernel/all_gather_matmul.cpp, mc2/matmul_reduce_scatter/op_kernel/matmul_reduce_scatter.cpp
- 缺陷描述：原代码只用BIAS_CAST做二分支判断，未根据ND2NZ_OPT参数区分bType的CubeFormat(NZ vs ND)，ND2NZ=false时bType仍使用NZ格式导致数据格式不匹配
- 修复模式：将3个bool模板参数组合展开为4个if constexpr分支，每个分支正确设置bType和biasType；运行时if改编译期if constexpr
- 可审查性：高
- 审查规则建议：多bool模板参数影响类型选择时审查所有参数组合是否有对应分支；涉及ND/NZ格式选择的场景需特别关注

### 44cd5c87 l0 op bugfix
- 根因类别：返回值未检查——kernel启动API调用返回值被丢弃
- 涉及文件：attention/flash_attention_score/op_api/flash_attention_score.cpp
- 缺陷描述：ADD_TO_LAUNCHER_LIST_AICORE宏启动kernel时返回值未赋给ret也未检查，kernel启动失败时函数继续返回看似正常的tensor指针
- 修复模式：将宏调用返回值赋给ret，失败时打印错误日志返回空指针
- 可审查性：高
- 审查规则建议：所有可能失败的API调用(特别是kernel launch)必须检查返回值

### cfe618f7 修改bias的错误类型
- 根因类别：数据类型定义错误——常量值和类型校验列表与实际规格不一致
- 涉及文件：gmm/grouped_matmul_finalize_routing/op_host/op_api/aclnn_grouped_matmul_finalize_routing_910_95_checker.h, 示例代码, 文档
- 缺陷描述：三类错误：(1)ZERO_DIM=1(应为0)、TWO_DIM=3(应为2)维度常量名值不匹配；(2)MX_BIAS_TYPE_SUPPORT_LIST中bias类型写成DT_FLOAT实际应为DT_BF16；(3)示例代码和文档中bias的aclDataType也对应错误
- 修复模式：修正常量定义值使其与语义一致，修正类型校验白名单，同步更新示例和文档
- 可审查性：高
- 审查规则建议：常量命名与其值应明确对应(ZERO_DIM=0等)可通过静态规则校验；类型支持白名单变更时同步检查文档和示例代码

### b2831bc0 A5 dispatch combine v2 MTE方式win区修复
- 根因类别：内存区域划分/偏移量错误——A5平台MTE通信方式的window区偏移和大小分配不正确
- 涉及文件：mc2/common/inc/kernel/moe_distribute_base.h, mc2/moe_distribute_combine_v2/op_host/, mc2/moe_distribute_dispatch_v2/op_kernel/多个文件
- 缺陷描述：(1)A5_MTE_STATE_WIN_SIZE被设为4MB实际应为1MB；(2)A5获取EP window size时未扣除状态区导致数据区与状态区重叠；(3)WIN_STATE_OFFSET(500K→450K)、STATE_WIN_OFFSET(950K→900K)偏移错位导致dispatch/combine状态互相覆盖
- 修复模式：统一A3/A5状态区为1MB；A5场景从总buffer扣减1MB得到数据区；调整区内偏移量避免重叠；提取GetEpWinSize统一窗口大小获取逻辑
- 可审查性：低
- 审查规则建议：内存区域划分常量修改应附带布局示意图和边界计算验证；共享内存区域偏移量定义应系统性验证无重叠

### e5817c02 修复GM偏移量的数据类型为int64
- 根因类别：整数溢出——GM偏移量使用uint32导致大数据场景溢出
- 涉及文件：moe/moe_token_unpermute_with_routing_map/op_kernel/masked_select.h
- 缺陷描述：CopyIn和CopyInMask中ind=progress*tileLength声明为uint32_t，超过4GB地址范围时溢出导致读取错误内存位置
- 修复模式：ind类型从uint32_t改为uint64_t
- 可审查性：高
- 审查规则建议：所有内存偏移/地址计算变量应使用uint64_t/int64_t；乘法表达式结果类型进行溢出风险检查

### 3aacdac4 fix tiling data
- 根因类别：成员变量未初始化——tiling计算使用未赋值的字段
- 涉及文件：gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_host_utils.h, op_tiling/arch35/grouped_matmul_swiglu_quant_v2_basic_tiling.cpp
- 缺陷描述：AnalyzeAttrs未设置inputParams_.groupType，但后续L1 tiling计算依赖该字段，未初始化值导致tiling计算结果不可预测
- 修复模式：新增常量SPLIT_M=0，在AnalyzeAttrs中显式赋值inputParams_.groupType=SPLIT_M
- 可审查性：高
- 审查规则建议：结构体所有字段在使用前应确保已初始化；新增使用已有字段的代码路径时审查该字段的所有初始化点

### 710ccbe8 fix MX_QUANT dynamicscale fail
- 根因类别：DMA搬运padding参数错误
- 涉及文件：mc2/moe_distribute_dispatch/op_kernel/arch35/moe_distribute_dispatch_arch35.h, mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h
- 缺陷描述：DataCopyPad的isPad被设为false，MX量化场景下末尾数据搬运未进行padding填充，非对齐部分数据残留导致精度错误
- 修复模式：6处DataCopyPadParams的isPad从false改为true启用padding
- 可审查性：高
- 审查规则建议：DataCopyPad用于非对齐数据搬运场景时检查isPad是否应为true；对所有DataCopyPadParams{false,...}审查是否遗漏padding需求

### 85579b8f fix kv_rms_norm_rope_cache compile error
- 根因类别：头文件引用路径错误
- 涉及文件：posembedding/kv_rms_norm_rope_cache/op_kernel/arch35/kv_rms_norm_rope_cache_regbase_base.h, platform.h
- 缺陷描述：引用了../inc/platform.h但该路径不存在(应为同目录platform.h)，且platform.h包含冗余平台抽象导致编译失败
- 修复模式：修正include路径为platform.h并加入kernel_operator.h；简化platform.h移除冗余函数
- 可审查性：高
- 审查规则建议：include路径变更后必须验证编译通过；相对路径引用头文件需确认目标文件存在

### 9449d9bb d不等长修复layout拦截
- 根因类别：过度校验——错误拦截合法输入
- 涉及文件：attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_v2.cpp
- 缺陷描述：tensorlist模式下对Key/Value的H维和D维做相等性校验，D不等长的合法场景被错误拦截
- 修复模式：删除standardKH!=standardVH和standardKD!=standardVD的校验逻辑(共10行)
- 可审查性：高
- 审查规则建议：输入合法性校验新增时确认拦截条件覆盖所有合法场景；维度校验需review是否存在合法不等长场景

### 09ce5437 伪量化修复同步问题和后量化搬运偏移错误问题
- 根因类别：(1)同步时序错误 (2)地址偏移计算遗漏
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h
- 缺陷描述：Bug1：CrossCoreSetFlag放在FreeTensor之后导致竞态——其他核可能在本核释放tensor前访问共享数据。Bug2：后量化per-channel模式下perChannelQuantGQAOffset缺少goIdx*s1BaseSize*dSizeV偏移项
- 修复模式：(1)CrossCoreSetFlag移到FreeTensor之前；(2)偏移计算增加goIdx相关项
- 可审查性：中
- 审查规则建议：CrossCoreSetFlag/WaitFlag与FreeTensor顺序必须review，同步信号在资源释放前设置；多维偏移计算检查是否遗漏维度分量

### dde680c6 修复mc2编译3-8包编不过的问题
- 根因类别：条件编译缺失——无条件include仅在开源构建可用的头文件
- 涉及文件：mc2/common/inc/mc2_hcom_topo_info.h, mc2/common/src/mc2_hcom_topo_info.cpp, UT文件
- 缺陷描述：无条件include了hccl_rank_graph.h及相关符号，但该头文件仅在BUILD_OPEN_PROJECT下可用，3-8版本包编译失败
- 修复模式：用#ifdef BUILD_OPEN_PROJECT宏保护相关include、声明和实现
- 可审查性：高
- 审查规则建议：引入新外部头文件依赖时检查在所有目标构建版本是否可用；条件可用的头文件需用#ifdef保护

### 446d15c9 antiquant pa_nz_pc e4m3精度问题
- 根因类别：内存stride计算未对齐——NZ格式下行数需2行对齐但未做
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_antiquant_processor.h, vf/vf_antiquant_w4.h, vf/vf_antiquant_w8.h
- 缺陷描述：PA NZ格式下colDstStride=dealRowCount*colBaseSize，硬件要求2行对齐，dealRowCount为奇数时stride偏小导致数据重叠或错位
- 修复模式：6处colDstStride改为(dealRowCount+(dealRowCount%2))*colBaseSize实现2行对齐；dataCopyParams.srcStride改用isKvCacheNz标志
- 可审查性：中
- 审查规则建议：NZ格式stride/offset计算须满足硬件对齐要求(2行对齐、32B对齐)；dealRowCount可能为奇数时所有依赖行数的计算需检查对齐

### 0fdf4d2b nsa compress with cache校验修复
- 根因类别：输入校验缺失——空tensor和非ND格式未检查
- 涉及文件：attention/nsa_compress_with_cache/op_host/op_api/aclnn_nsa_compress_with_cache.cpp, 文档
- 缺陷描述：aclnn接口未检查输入tensor是否为空(shapeSize为0)和格式是否为ND，导致后续计算异常或结果错误
- 修复模式：新增CheckIsEmptyTensor和CheckNDFormat函数拦截非法输入
- 可审查性：高
- 审查规则建议：aclnn接口GetWorkspaceSize阶段应完整校验所有输入tensor有效性(空tensor、格式、dtype)

### cb1c8346 attention update INF值修复
- 根因类别：魔法数字——使用3e+99作为非标准无穷大表示
- 涉及文件：attention/attention_update/op_kernel/decode_update.h
- 缺陷描述：POS_INF/NEG_INF使用3e+99/-3e+99而非IEEE 754标准infinity，某些计算路径下(如softmax数值稳定性)比较逻辑或数学运算结果不正确
- 修复模式：改用std::numeric_limits<float>::infinity()
- 可审查性：高
- 审查规则建议：禁止用魔法数字(3e+99、-3.4e+38等)表示无穷大，统一使用std::numeric_limits<float>::infinity()或INFINITY宏

### deca2f75 fix group 1024 limit
- 根因类别：输入参数上限校验缺失
- 涉及文件：gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/aclnn_grouped_matmul_swiglu_quant_v2_utils.h, gmm/quant_grouped_matmul_inplace_add/op_host/op_api/aclnn_quant_grouped_matmul_inplace_add_910_95_checker.cpp
- 缺陷描述：两个算子的aclnn接口未对groupList长度做上限校验，超过1024时底层kernel无法正确处理
- 修复模式：新增groupListLen>MAX_GROUP_LIST_SIZE(1024)的校验
- 可审查性：高
- 审查规则建议：算子接口对输入维度/长度等参数应有明确上下限校验；kernel资源限制须在host侧提前拦截

### 7327fc7c [FAG] fp8 fix
- 根因类别：数据类型错误 + 寄存器布局(RegLayout)混淆 + 多余内存屏障
- 涉及文件：attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_kernel_base.h, attention/flash_attention_score_grad/op_kernel/arch35/vector_api/vf_cast_transdata_deconflict.h, attention/flash_attention_score_grad/op_kernel/arch35/vector_api/vf_muls_sel_simple_softmax_aligned256.h
- 缺陷描述：三处问题。(1)pL1Buf的buffer初始化使用sizeof(INPUT_TYPE)而非sizeof(OUTDTYPE)，FP8输入时大小不同导致L1 buffer分配错误。(2)castTraitFp322Fp16Even原为模板但FP32->FP16不需要区分HIFP8舍入模式，FP8路径错误地复用了FP16的CastTrait，应使用独立的castTraitFp322Fp8Zero(RegLayout::ZERO)。(3)softmax对齐256路径中移除不必要的LocalMemBar。
- 修复模式：修正sizeof中的类型参数；重构CastTrait常量定义将FP8和FP16正确分离；删除多余内存屏障
- 可审查性：中
- 审查规则建议：多数据类型路径(FP8/FP16/FP32)时审查sizeof和CastTrait是否与实际数据路径匹配；L1/L0 buffer的Init中sizeof的类型应与实际存储的数据类型一致

### 651df9de 修复子包与base包包含路径冲突
- 根因类别：头文件include路径组织错误
- 涉及文件：mc2/all_gather_matmul_v2/op_kernel/all_gather_matmul_v2_apt.cpp, mc2/all_gather_matmul_v2/op_kernel/arch35/all_gather_quant_bmm_perblock.h
- 缺陷描述：顶层cpp直接include第三方子包头文件../3rd/...相对路径，子包与base包同时存在时路径冲突
- 修复模式：将include从顶层cpp移到实际使用该头文件的.h中，让依赖关系更局部化
- 可审查性：中
- 审查规则建议：第三方/子包头文件的include应放在直接使用它的头文件中；审查../3rd/等相对路径在不同打包模式下是否能正确解析

### 7eee21c7 fix aclnnInnerRotaryPositionEmbedding
- 根因类别：函数签名不匹配（缺少参数）
- 涉及文件：posembedding/interleave_rope/op_host/op_api/aclnn_interleave_rope.cpp
- 缺陷描述：aclnnInnerRotaryPositionEmbeddingGetWorkspaceSize的extern声明缺少const aclTensor* rotate参数，与实际定义不一致
- 修复模式：在extern声明中补充缺失参数，调用处传入nullptr
- 可审查性：高
- 审查规则建议：extern "C"前向声明的函数签名必须与实际定义完全一致；上游API添加新参数时所有前向声明和调用点都需同步更新

### be81c263 fix aclnn trans
- 根因类别：参数校验逻辑错误——转置校验位置不当 + 退化case未豁免
- 涉及文件：gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/aclnn_grouped_matmul_swiglu_quant_v2_utils.h
- 缺陷描述：转置校验放在weight转置处理之前且未考虑[1,1] shape时转置无区别的退化case；输出dtype校验用宏检查过于死板
- 修复模式：调整校验顺序，增加[1,1] shape豁免逻辑；用显式dtype判断替代宏调用
- 可审查性：中
- 审查规则建议：参数校验应考虑退化case（维度为1时转置等价）；校验逻辑的先后顺序需注意是否依赖前面的数据准备步骤

### 2efc569c [DTS2025121803100] oom包内存越界问题修复
- 根因类别：内存越界——尾块数据搬运超出实际数据边界 + OOM检测地址范围缺失
- 涉及文件：mc2/quant_all_reduce/和mc2/quant_reduce_scatter/下多个文件(tiling、kernel、通信层、工具函数)
- 缺陷描述：数据分块搬运时尾块按固定大小搬运但实际数据量非整数倍，导致尾块越过实际数据边界覆写scale数据；alignedXSize_对齐到1024B的padding方式不可靠；OOM检测框架未注册Win区地址范围
- 修复模式：新增BlockAlignMod计算尾块实际元素数；检测当前是否为最后一个核的最后一个块用实际尾块大小；scale偏移改用xSize_；注册OOM检测范围
- 可审查性：高
- 审查规则建议：分块循环的最后一块必须检查是否为尾块并使用实际剩余大小；DataCopy的count不应超过实际数据范围；对外部地址访问应注册OOM检测范围

### cbd02341 修正custom模式
- 根因类别：CMake条件判断不兼容尾部分号
- 涉及文件：mc2/matmul_all_reduce/op_host/CMakeLists.txt
- 缺陷描述：ASCEND_COMPUTE_UNIT变量在某些构建配置下带有尾部分号(ascend910_95;)，导致STREQUAL不匹配
- 修复模式：增加带分号后缀的匹配
- 可审查性：高
- 审查规则建议：CMake中list类型变量STREQUAL比较时可能带分号分隔符，应使用IN_LIST或正则匹配

### ffe0d37f fix注释&& oom包问题
- 根因类别：OOM地址注册错误（使用了错误的地址常量）
- 涉及文件：gmm/grouped_matmul/op_kernel/a16w4_msd/grouped_matmul_weight_quant_a16w4_msd_basic_block.h, gmm/grouped_matmul/op_kernel/a16w4_msd/grouped_matmul_weight_quant_a16w4_msd_cube_service.h
- 缺陷描述：OOMCheckAddrRange调用中将C1_EYE_DIAG传入作为地址，但实际GM上的对角矩阵使用的是C1C2_EYE_DIAG，二者指向不同地址，导致OOM检测地址范围不一致
- 修复模式：将C1_EYE_DIAG改为C1C2_EYE_DIAG；修正头文件保护宏注释
- 可审查性：高
- 审查规则建议：OOMCheckAddrRange的地址参数应与紧邻的SetGlobalBuffer调用使用相同的地址常量

### 13a8a1dd 修复rank4精度问题
- 根因类别：内存偏移计算错误(多rank stride)
- 涉及文件：mc2/allto_all_matmul/op_kernel/arch32/allto_all_matmul_util.h
- 缺陷描述：CopyUbufToGmAlignB16的目标stride参数写成k*sizeof(T)，但rank4场景下peer memory中数据按rank交错排列，正确的stride应为(rank_size-1)*k*sizeof(T)
- 修复模式：将dstStride参数从k*sizeof(T)改为(rank_size-1)*k*sizeof(T)
- 可审查性：中
- 审查规则建议：多rank/多卡场景DMA搬运时，审查stride参数是否正确包含rank_size因子

### 428e7130 修复后量化per-channel shape为[H]拦截
- 根因类别：输入校验逻辑不完备（缺少合法分支）
- 涉及文件：attention/incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp, attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：quantScale2Dim==1时只检查了per-tensor(shape为[1])的情况，未处理shape为[H]但维度数为1的合法per-channel输入，导致合法输入被错误拦截
- 修复模式：在quantScale2Dim==1分支中新增判断：shapeSize==perChannel时认定为per-channel模式
- 可审查性：高
- 审查规则建议：输入参数校验时枚举所有合法shape组合，确保校验逻辑覆盖完整

### fc6f5cfa alltoallmatmul tilingkey整改与AIC精度修复
- 根因类别：多处缺陷——(1)pong buffer偏移计算错误 (2)AIC到AIV同步信号类型错误 (3)数据搬运使用错误的对齐尺寸
- 涉及文件：mc2/allto_all_matmul/op_kernel/arch32/allto_all_matmul.h, mc2/allto_all_matmul/op_kernel/arch32/allto_all_matmul_util.h, mc2/allto_all_matmul/op_kernel/arch32/matmul.hpp, mc2/allto_all_matmul/op_kernel/allto_all_matmul.cpp
- 缺陷描述：(1)MoveResult函数参数传入data_len(含k维度总元素数)但函数内部已按token粒度乘k导致搬运量放大k倍。(2)peer_mem_k_size用512B对齐但实际应使用mid_output_k_size不对齐。(3)CrossCoreSetFlag使用PIPE_MTE3应为PIPE_FIX。(4)AlltoAll结束时缺少WaitEvent同步。
- 修复模式：修正参数语义从元素总数改为token数；去掉512B对齐；修正同步标志；增加WaitEvent
- 可审查性：中
- 审查规则建议：函数参数语义必须明确(token数vs元素总数)；同步原语pipe类型须与实际数据路径匹配；ping-pong循环结束后须确保最后一个buffer操作完成

### b7dc95f5 修复mxquant模板在inf/nan/scale补齐到2的场景下的偶现精度问题
- 根因类别：两个独立缺陷——(1)fp16转bf16时inf/nan指数信息丢失 (2)scale补齐到偶数后padding区域脏数据参与计算
- 涉及文件：moe/moe_init_routing_v3/op_kernel/arch35/moe_v3_gather_mxfp8_quant.h
- 缺陷描述：(1)fp16的inf(0x7c00)转bf16后变成正常大数(0x7f00)而非bf16的inf(0x7f80)，导致inf/nan检测失败。(2)CeilAlign到2补齐的那一列用maskLoop覆盖了包括padding的所有元素，未初始化数据参与运算产生不确定结果。
- 修复模式：(1)先用fp16指数mask提取inf/nan位置，转bf16后用Select强制设为bf16 inf值。(2)新增validScaleElemNum参数用maskValid排除padding区域。
- 可审查性：中
- 审查规则建议：浮点格式转换时审查特殊值(inf/nan/denormal)处理是否正确；CeilAlign/padding场景审查计算mask是否排除了padding区域

### 0540e51b fix AttentionToFFN accuracy
- 根因类别：数据搬运与核间分工的顺序依赖错误
- 涉及文件：mc2/attention_to_ffn/op_kernel/attention_to_ffn.h
- 缺陷描述：SplitToCore根据totalSendNum_分配任务，某些核startId>=totalSendNum_时提前return，但关键数据(expertIdsTensor_的GM到UB搬运)放在了提前return之后，导致提前return的核从未执行数据搬运
- 修复模式：将DataCopyPad移到SplitToCore和提前return之前
- 可审查性：高
- 审查规则建议：早期return语句之前检查是否有必要的初始化/搬运操作被跳过；多核场景下共享数据准备必须在任务分派和提前退出逻辑之前完成

### 4bc16847 fix 1:1 adapt sk 1:2
- 根因类别：编译态常量在运行态不匹配的兼容性缺陷
- 涉及文件：attention/mla_prolog/op_kernel/kernel_mla_prolog_split_n.h, attention/mla_prolog/op_kernel/service_dynamic_quant_qn_mul_qr.h
- 缺陷描述：cvRatio_为static constexpr编译期常量，按cv1:1编译但运行时实际硬件是cv1:2时，编译态constexpr无法反映运行态真实值，导致cubeBlockIdx_除以错误的cvRatio、vectorCoreNum_未修正、if constexpr走错分支
- 修复模式：将cvRatio_从constexpr改为运行时变量通过GetSubBlockNum()获取；将if constexpr改为运行时if；将模板参数改为函数参数
- 可审查性：高
- 审查规则建议：编译态常量(constexpr/模板参数)如果依赖硬件配置，必须验证编译态与运行态是否可能不一致；if constexpr仅用于真正的编译期分支

### 14987487 修复qbmmv4 mxa8w4 ND精度问题
- 根因类别：双buffer模式下L1处理循环缺失
- 涉及文件：common/act/prologue/block_prologue_b_cast_scsc.h
- 缺陷描述：ND格式2-buffer模式下nL1Len可能大于nUbSize(L1数据需分多次搬到UB处理)，但VectorProcess只处理了一轮UB大小的数据，缺少遍历所有N/K分块的循环
- 修复模式：将VectorProcess分为4-buffer路径(使用aiv分工偏移)和通用路径(nK双重循环遍历所有分块)
- 可审查性：中
- 审查规则建议：多buffer模式下数据处理，当L1数据量大于UB容量时必须有循环机制保证所有数据被处理

### 6c1b46d8 更改MatmulAllreduce精度Fail问题
- 根因类别：构建配置缺失导致编译宏未启用
- 涉及文件：mc2/matmul_all_reduce/op_host/CMakeLists.txt
- 缺陷描述：ascend910_95平台缺少-DENABLE_CV_COMM_VIA_SSBUF=true编译选项，依赖此宏的CV通信路径未启用导致精度问题
- 修复模式：新增add_ops_compile_options设置对应编译选项
- 可审查性：高
- 审查规则建议：新增算子或平台支持时检查所有平台特定编译宏是否正确配置

### 5eb3fef2 修复rope_with_sin_cos_cache算子在线打包A5失败
- 根因类别：(1)编译选项拼写错误 (2)A5子包配置遗漏算子
- 涉及文件：posembedding/rope_with_sin_cos_cache/op_host/CMakeLists.txt, scripts/ci/ascend910_95/ops_transformer_operator_list.yaml
- 缺陷描述：编译选项-cce-aicore-dcci-before-kernle-end中kernle拼写错误(应为kernel)导致选项未生效；A5算子列表yaml缺少该算子
- 修复模式：修正拼写；在yaml中添加算子
- 可审查性：高
- 审查规则建议：编译器选项字符串应做spell check或用常量定义；新增算子应有checklist确认所有目标平台配置已更新

### 479084fc fix grouped_matmul_swiglu_quant_v2 DFX
- 根因类别：错误返回值类型(return false在graphStatus函数中)
- 涉及文件：gmm/grouped_matmul_swiglu_quant_v2/op_host/op_tiling/grouped_matmul_swiglu_quant_v2_base_tiling.cpp, gmm/grouped_matmul_swiglu_quant_v2/op_host/op_tiling/grouped_matmul_swiglu_quant_v2_fusion_tiling.cpp
- 缺陷描述：ParseInputAndAttr返回类型为ge::graphStatus，但OP_CHECK_IF宏中写return false。false隐式转换为0即GRAPH_SUCCESS，校验失败时返回了"成功"状态
- 修复模式：将3处return false改为return ge::GRAPH_FAILED
- 可审查性：高
- 审查规则建议：函数返回类型为枚举/状态码时禁止使用bool字面量作为返回值；可编写静态分析规则检测graphStatus函数中的return false

### 66306863 修复A5子包缺失算子_apt.py脚本问题
- 根因类别：CMake install命令条件判断错误(configure阶段判断build阶段产物)
- 涉及文件：cmake/custom_build.cmake
- 缺陷描述：对_apt.py文件的install用if(EXISTS)包裹，但该判断在configure阶段执行此时文件未生成，EXISTS永远为false导致install永不执行
- 修复模式：去掉if(EXISTS)条件，直接使用带OPTIONAL的install命令
- 可审查性：高
- 审查规则建议：CMake中对构建产物使用if(EXISTS)是反模式，应使用install(... OPTIONAL)替代

### aa6aad87 FFNToAttention H对齐修正
- 根因类别：条件校验过严 + DMA搬运对齐
- 涉及文件：mc2/ffn_to_attention/op_host/op_tiling/ffn_to_attention_tilling.cpp, mc2/ffn_to_attention/op_kernel/ffn_to_attention.h
- 缺陷描述：(1)Tiling侧HS下界校验用H_MIN+SCALE_SIZE，实际H_MIN本身即合法下界(不含scale的场景)，导致合法输入被拒绝。(2)Kernel侧DataCopy按axisH_搬运，当axisH_非32B对齐时MTE越界；改为DataCopyPad按实际字节搬运并补零
- 修复模式：下界改为H_MIN；DataCopy→DataCopyPad
- 可审查性：高
- 审查规则建议：校验边界值时区分是否包含附加字段(如scale)；所有DataCopy对非对齐维度必须使用DataCopyPad

### 5a0210bf 修复A2AMM与MMA2A编译错误
- 根因类别：数据类型字面量混用
- 涉及文件：mc2/allto_all_matmul/op_host/op_tiling/arch32/allto_all_matmul_tiling_910.cpp, mc2/matmul_allto_all/op_host/op_tiling/arch32/matmul_allto_all_tiling_910.cpp
- 缺陷描述：std::map<int, ...>的key使用浮点字面量(2.0, 3.0, 40.0等)，类型不匹配导致编译错误
- 修复模式：将所有浮点字面量改为int字面量(2, 3, 40等)
- 可审查性：高
- 审查规则建议：map key类型必须与声明一致；编译器warning应当开启-Wfloat-conversion

### f44e77cf fix nQueryIndex = 8 MTE address out of range bug
- 根因类别：DMA搬运对齐/MTE地址越界
- 涉及文件：attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/sparse_lightning_indexer_grad_kl_loss_vector.h
- 缺陷描述：weight搬运使用AlignTo(gSizeQueryIndex, 16)按16对齐，当gSizeQueryIndex=8时对齐到16，超出GM实际数据范围导致MTE越界。修复：特判gSizeQueryIndex==8时用DataCopyPad按实际字节搬运并pad
- 修复模式：DataCopy→DataCopyPad，按实际字节长度搬运+padding
- 可审查性：中
- 审查规则建议：DataCopy的len参数若经过AlignTo处理，必须验证对齐后长度不超过GM实际可访问范围

### 71d681fc 日志修正
- 根因类别：日志级别错误
- 涉及文件：mc2/all_gather_matmul_v2/op_graph/fallback_all_gather_matmul_v2.cpp, mc2/matmul_reduce_scatter_v2/op_graph/fallback_matmul_reduce_scatter_v2.cpp
- 缺陷描述：fallback执行路径中使用OPS_LOG_E(error级别)打印commMode信息，这是正常执行流而非错误，错误日志干扰诊断
- 修复模式：删除多余的OPS_LOG_E调用
- 可审查性：高
- 审查规则建议：正常执行路径不应使用ERROR级别日志；OPS_LOG_E仅用于真正的错误条件

### 6e701375 修改GetSoftMaxFlashV2MaxTmpSize接口问题
- 根因类别：平台API接口差异/函数签名不匹配
- 涉及文件：moe/moe_gating_top_k_softmax_v2/op_host/moe_gating_top_k_softmax_v2_tiling_k_fullload.cpp
- 缺陷描述：GetSoftMaxFlashV2MaxTmpSize在A5平台(910_95)签名为(shape, size, bool, bool)，其他平台签名为(shape, size, size, bool)。原代码统一用A5签名，非A5平台参数类型不匹配导致计算错误
- 修复模式：按socVersion分支调用不同签名的API
- 可审查性：中
- 审查规则建议：调用platform-specific API时必须确认各平台签名一致性；若不一致需条件分支处理

### ef25ced5 bugfix: empty tensor tilingdata reset
- 根因类别：空tensor路径tiling数据结构不匹配
- 涉及文件：attention/flash_attention_score/op_host/flash_attention_score_tiling.cpp, op_kernel/arch32/flash_attention_score_template_tiling_key.h, op_kernel/arch32/flash_attention_score_tiling.h
- 缺陷描述：Empty tensor路径使用独立的FlashAttentionScoreEmptyInputTilingData结构填充tiling，但kernel侧模板选择绑定了FlashAttentionScoreTilingData主结构，结构不匹配导致kernel读取到错误tiling数据。修复：去掉独立结构，用主结构+reset()清零所有字段后再填充empty路径字段
- 修复模式：统一tiling数据结构，empty路径先reset再填充
- 可审查性：高
- 审查规则建议：empty tensor路径的tiling数据结构必须与kernel侧期望的结构一致；新增分支路径时检查tiling struct选择

### 1dbd7856 fix scatter_pa_kv_cache oom aic
- 根因类别：DMA搬运对齐/MTE地址越界(OOM)
- 涉及文件：attention/scatter_pa_kv_cache/op_kernel/arch35/scatter_pa_kv_cache_alibi_fully_load.h, scatter_pa_kv_cache_normal_fully_load.h, scatter_pa_kv_cache_rope_fully_load.h
- 缺陷描述：DataCopy按RoundUp(headSize)搬运，当headSize非32B对齐时RoundUp后的长度超出GM实际连续数据范围，导致MTE越界(OOM/AIC异常)。3个文件(alibi/normal/rope fully_load)均存在此问题
- 修复模式：DataCopy→DataCopyPad，按实际字节长度搬运，不做对齐扩展
- 可审查性：高
- 审查规则建议：DataCopy(src, dst, RoundUp(size))模式需验证GM侧连续空间是否足够；非对齐尺寸统一用DataCopyPad

### 7fe18da5 QuantMatmulAllReduce Fix GetDynamicQuantTempBuffSize
- 根因类别：复制粘贴/变量名混用
- 涉及文件：mc2/matmul_all_reduce/op_host/op_tiling/arch35/quant_matmul_all_reduce_tiling_910_95.cpp
- 缺陷描述：Dequant路径从Quant路径复制代码后，procRowTileCnt计算仍使用ubDenomQuant而非ubDenomDequant，导致Dequant的BroadCast tiling shape计算错误
- 修复模式：将ubDenomQuant改为ubDenomDequant；同时修正变量命名(ubDenomDeQuant→ubDenomDequant)
- 可审查性：高
- 审查规则建议：复制代码后搜索所有旧变量名是否已全部替换；对称路径(quant/dequant)的变量应一一对应检查

### ec0365bb fix AttentionToFFN/FFNToAttenttion graph
- 根因类别：新算子类型注册遗漏
- 涉及文件：mc2/common/src/mc2_moe_gen_task_ops_utils.cpp
- 缺陷描述：新增AttentionToFFN/FFNToAttention算子未加入NO_AI_CPU_SET集合和Mc2MoeGenTaskCallbackV2的过滤条件，导致这两个算子被错误地走AICPU插入task路径
- 修复模式：将两个新算子类型加入unordered_set和条件过滤链
- 可审查性：高
- 审查规则建议：新增算子类型时必须检查所有需要注册/过滤的集合和条件；建议用checklist管理算子注册点

### 77445c2b 修复静态库脚本不支持3.7
- 根因类别：Python版本兼容性
- 涉及文件：scripts/util/build_opp_kernel_static.py
- 缺陷描述：Path.unlink(missing_ok=True)参数是Python 3.8+特性，Python 3.7环境调用报错
- 修复模式：改为try/except FileNotFoundError方式兼容3.7
- 可审查性：中
- 审查规则建议：构建脚本需声明Python最低版本要求；使用新API特性前检查最低支持版本

### f55d2a35 Bugfix for statustensor
- 根因类别：初始化不完整/buffer清零范围不足
- 涉及文件：mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h
- 缺陷描述：statusTensor_初始化用Duplicate清零时，长度参数为recvWinBlockNum_*8，但实际buffer分配大小为statusBufCntAlign*8。当statusBufCntAlign > recvWinBlockNum_时，尾部区域未清零含脏数据
- 修复模式：将recvWinBlockNum_改为statusBufCntAlign
- 可审查性：高
- 审查规则建议：Duplicate清零的长度必须与buffer分配大小一致；分配和初始化应使用同一变量

### a692ffe6 修正fullmesh tiling侧对齐
- 根因类别：对齐处理缺失 + 条件分支遗漏
- 涉及文件：mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp
- 缺陷描述：(1)lastDim未做8对齐(CEIL_ALIGN32)导致GetCumSumMaxMinTmpSize接口调用失败。(2)非fullmesh场景(isSetCommAlg=false)不需要cumSum计算，但原代码无条件执行导致非fullmesh路径也做了不必要计算
- 修复模式：lastDim做CEIL_ALIGN32对齐；添加isSetCommAlg条件保护
- 可审查性：高
- 审查规则建议：调用AscendC tiling接口前确认shape参数满足对齐要求；条件功能(fullmesh)的tiling计算应有对应条件保护

### caa50884 修复dispatch MX量化场景下部分用例dynamic_scales精度问题
- 根因类别：对齐处理不一致/量化精度
- 涉及文件：mc2/moe_distribute_dispatch/op_kernel/arch35/moe_distribute_dispatch_arch35.h
- 缺陷描述：MX量化(MXFP8)用Align64填零，PERTILE量化用Align128填零，但PERTILE实际也需要Align128。当Hidden_Size非128对齐时，PERTILE路径搬入数据含脏数据影响scale计算精度
- 修复模式：合并条件分支，MX和PERTILE统一使用Align128填零
- 可审查性：中
- 审查规则建议：同一量化框架内的padding/对齐策略应保持一致；新增量化模式时检查对齐要求

### 4a63cf4b fix online compile
- 根因类别：构建配置遗漏(多处)
- 涉及文件：mc2/allto_all_matmul/op_host/CMakeLists.txt, op_host/allto_all_matmul_def.cpp, op_host/op_tiling/arch35/allto_all_fp_matmul_tiling_base.cpp, op_kernel/allto_all_matmul_apt.cpp等
- 缺陷描述：在线编译场景下多处配置遗漏：(1)CMakeLists缺少apt目标的depends声明 (2)算子定义缺少opFile.value配置 (3)通信引擎未指定withCommEngine (4)kernel模板参数签名从单一TemplateId改为三参数(QUANT_MODE, X2_TRANSPOSE, BIAS_DTYPE)
- 修复模式：补全各处缺失配置
- 可审查性：中
- 审查规则建议：新增在线编译支持时需checklist验证(CMake依赖、算子定义、通信引擎配置、kernel模板签名)

### 0c941da3 fix memory leak
- 根因类别：函数参数传递错误(值传递导致内存泄漏)
- 涉及文件：attention/fused_infer_attention_score/op_host/op_api/aclnn_fused_infer_attention_score_inner.cpp, .h
- 缺陷描述：FusedInferAttentionScoreProcessSoftmaxLse的tempTensor参数声明为const aclTensor *tempTensor(值传递)，函数内部可能修改指针(如创建新tensor)，但修改无法传回调用方。调用方持有的原始指针变为悬空或新对象无法释放
- 修复模式：参数改为const aclTensor *&tempTensor(引用传递)
- 可审查性：高
- 审查规则建议：函数内需要修改指针本身(而非指向内容)时，参数必须为引用或二级指针

### 60b8caa9 [MlaProlog] fix precision issue when queryNormFlag is set to true
- 根因类别：workspace偏移量计算条件分支缺陷
- 涉及文件：attention/mla_prolog/op_kernel/kernel_mla_prolog_split_n.h
- 缺陷描述：mmCqResGm_后的workspaceOffset偏移在queryNormFlag分支外无条件执行，导致queryNormFlag=true时mmCqResGm_的空间未被跳过，后续mmQcQrResGm_等buffer与之重叠。修复：将偏移逻辑移入if/else分支内，queryNormFlag=0时偏移mmCqResGm_大小后放rmsNorm结果，queryNormFlag=1时偏移mmCqResGm_大小但rmsNorm结果直接输出GM
- 修复模式：workspace offset计算逻辑按queryNormFlag分支重组
- 可审查性：中
- 审查规则建议：workspace布局中多个buffer的offset计算必须覆盖所有条件分支路径；建议画workspace layout图辅助审查

### 1eaf5af25f QuantMatmulAllReduce 拦截逻辑和日志修正
- 根因类别：条件判断逻辑错误 + 日志信息错误
- 涉及文件：mc2/matmul_all_reduce/op_host/op_tiling/arch35/weight_quant_matmul_all_reduce_tiling_910_95.cpp, mc2/matmul_all_reduce/op_host/op_tiling/matmul_all_reduce_tiling_base.cpp
- 缺陷描述：(1) CheckAxisSize中k轴超限的报错日志误写为"n-axis"，导致报错信息误导；(2) pergroup对齐校验未区分A16W8/F8(对齐32)和A16W4(对齐64)两种场景，统一check导致拦截条件不准确；(3) CheckDequantScaleShape中per-tensor场景(scaleShapeSize==1)直接return true跳过后续校验；(4) CheckPertokenScaleShape中MX场景下x1为三维时mValue包含batch乘积，应除以batch得到实际m轴
- 修复模式：修正日志字符串 + 拆分条件分支 + 修正shape计算公式
- 可审查性：中
- 审查规则建议：错误报告中引用的变量名/轴名应与实际打印的变量一致；校验逻辑应覆盖所有量化模式的分支并分别处理

### 928969a0d3 修复后量化per-tensor拦截问题
- 根因类别：输入校验逻辑缺陷(维度判断缺失)
- 涉及文件：attention/incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp, attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp/.h
- 缺陷描述：后量化场景中per-tensor/per-channel区分仅靠shapeSize判断(shapeSize==1为per-tensor)，当N*D恰好等于1时无法区分。修复改用tensor维度数(dimNum)区分：dimNum==1为per-tensor。同时新增quantScale2和quantOffset2的维度一致性校验
- 修复模式：引入维度数(dimNum)作为per-tensor/per-channel判别依据，拆分混合校验
- 可审查性：中
- 审查规则建议：tensor语义区分不应依赖shapeSize特定值，应优先使用维度数(rank)等结构化属性；scale/offset参数应做维度一致性交叉校验

### 249707c2b4 fix sync problem
- 根因类别：初始化顺序错误
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_kernel_base.h
- 缺陷描述：kernel初始化流程中InitLocalBuffer()被放在InitGlobalBuffer()之后调用，但GlobalBuffer初始化依赖LocalBuffer已就绪，导致运行时数据同步问题。修复将InitLocalBuffer()提前到InitGlobalBuffer()之前
- 修复模式：调整初始化函数调用顺序
- 可审查性：低
- 审查规则建议：kernel初始化函数间的依赖关系应有明确注释或通过设计约束体现，而非依赖隐式调用顺序

### efa215ed23 Fix inconsistencies between documentation and API interfaces
- 根因类别：文档与API签名不一致
- 涉及文件：attention/lightning_indexer_grad/op_host/op_api/aclnn_lightning_indexer_grad.cpp/.h, 多个docs/.md文件
- 缺陷描述：多个API文档中函数签名与实际代码不一致：(1)文档中aclrtStream参数写为const aclrtStream但实际无const；(2)SparseLightningIndexerGradKLLoss文档中executor参数缺一个指针层级(应为**)
- 修复模式：修正文档签名使其与代码一致
- 可审查性：高
- 审查规则建议：API文档中的函数签名应自动从代码头文件生成或通过CI校验一致性

### e700254d71 [fix] fix oom
- 根因类别：条件逻辑反转(三元表达式条件写反)
- 涉及文件：attention/lightning_indexer/op_kernel/lightning_indexer_kernel.h, attention/quant_lightning_indexer/op_kernel/quant_lightning_indexer_kernel.h
- 缺陷描述：ProcessInvalid中计算dealSize时，三元表达式条件写反：当baseSize+singleCoreSize>totalOutputSize时应取剩余量(totalOutputSize-baseSize)，但原代码在此条件下取了完整的singleCoreSize，导致越界写入引发OOM
- 修复模式：修正三元表达式条件(反转比较运算符)
- 可审查性：高
- 审查规则建议：三元表达式中边界计算review时应逐case验证；处理"剩余量"逻辑时优先使用min(singleCoreSize, totalOutputSize-baseSize)这类更直观的表达

### 1496b77e76 修复sink fa tilingdata对齐问题
- 根因类别：结构体字段布局/对齐错误
- 涉及文件：attention/flash_attention_score/op_kernel/arch32/flash_attention_score_tiling.h, attention/flash_attention_score_grad/op_kernel/arch32/flash_attention_score_grad_tiling.h
- 缺陷描述：tiling数据结构中needSinkOp字段(uint8_t)位置与host端序列化顺序不一致，导致kernel侧读取tiling参数时数据错位。修复将needSinkOp移到正确位置并调整placeholder数组大小保持总大小不变
- 修复模式：调整结构体字段顺序使其与序列化布局一致
- 可审查性：低
- 审查规则建议：tiling结构体字段顺序应与host端写入顺序严格一一对应；新增字段时应有checklist确认host/kernel两侧对齐

### 86a006ad1c 修改分包json文件合并问题
- 根因类别：合并算法逻辑不足(浅合并vs深合并)
- 涉及文件：scripts/package/common/py/merge_binary_info_config.py
- 缺陷描述：update_config函数使用Python dict.update()浅合并，当两个分包json包含同一算子配置时后者完全覆盖前者，导致binaryList等信息丢失。修复改为深合并：对同名算子字段级合并，binaryList做拼接
- 修复模式：将dict浅合并替换为自定义深合并函数
- 可审查性：高
- 审查规则建议：合并嵌套配置时应明确区分"覆盖"和"合并"语义；包含列表字段的配置合并需验证列表是追加还是替换

### 6802b0274 SFA block_table拦截修复
- 根因类别：可选输入属性获取遗漏
- 涉及文件：attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp
- 缺陷描述：GetOptionalInputParaInfo中获取blockTable可选输入时只取了tensor但遗漏了desc，导致后续对blockTable做shape校验时无法获取描述信息
- 修复模式：补充一行GetOptionalInputDesc调用
- 可审查性：高
- 审查规则建议：可选输入的获取应统一模式(tensor+desc成对获取)，review时用checklist检查所有可选输入是否完整获取

### e6a2cec801 fix moe_token_unpermute_with_routing_map_grad tiling and md
- 根因类别：tiling计算中类型长度使用错误
- 涉及文件：moe/moe_token_unpermute_with_routing_map_grad/op_host/moe_token_unpermute_with_routing_map_grad_tiling.cpp
- 缺陷描述：tiling代码中prob缓冲区的类型长度错误使用了inputTypeLength，但prob的dtype可能与input不同(prob支持FLOAT)，应使用prob自身的类型长度。当prob为FLOAT(4字节)而input为FLOAT16(2字节)时，UB计算偏大
- 修复模式：新增probTypeLength变量从prob的desc获取正确类型长度
- 可审查性：中
- 审查规则建议：UB/workspace分配中每个buffer的类型长度应从对应tensor的dtype获取，不应复用其他tensor的类型长度

### 5a01d2eb24 Fix FAG Sink MM12 Nzout Bug
- 根因类别：代码逻辑顺序错误/执行时序缺陷
- 涉及文件：attention/flash_attention_score_grad/op_kernel/arch32/flash_attention_score_grad_s1s2_bn2gs1s2_sab.h
- 缺陷描述：SubGrapB函数中copyOut操作被放在sink计算之前，当NZ格式输出时copyOut会使用vecCopyOutBuffer做ND2NZ转换并写出，但随后的sink计算也依赖同一buffer，导致sink计算使用了被copyOut改写的数据
- 修复模式：将copyOut操作整块移到sink计算完成之后执行
- 可审查性：中
- 审查规则建议：同一buffer在多个计算阶段被读写时，审查buffer生命周期和读写顺序确认无write-after-read冲突

### bbed8e5c3f fix vecin
- 根因类别：队列方向声明错误(VECOUT应为VECIN)
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h
- 缺陷描述：postQuantScaleQue和postQuantOffsetQue两个TQue声明使用了QuePosition::VECOUT，但这两个队列用于从GM加载post-quant参数到vector计算单元，应为VECIN
- 修复模式：将TQue<QuePosition::VECOUT, 1>改为TQue<QuePosition::VECIN, 1>
- 可审查性：高
- 审查规则建议：TQue声明时检查VECIN/VECOUT方向与实际数据流方向一致；输入参数应使用VECIN

### 9388f7e6b9 fix example bug
- 根因类别：变量引用错误(使用了错误的变量名)
- 涉及文件：build.sh
- 缺陷描述：build_example()中设置CUST_LIBRARY_PATH和CUST_INCLUDE_PATH时路径使用了${VENDOR}(原始输入)，但应使用${vendor_name}(经处理后的小写变量)，导致路径拼接错误
- 修复模式：将${VENDOR}替换为${vendor_name}
- 可审查性：高
- 审查规则建议：shell脚本中存在经过条件处理/变换的变量时，后续应统一使用处理后的变量

### 484867f563 alibi mask bugfix
- 根因类别：标志位语义混淆/重载
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp/.h
- 缺陷描述：enableIFA标志位同时控制mask shape处理逻辑和IFA计算路径，当enableAlibiPse开启且query的s维度为1时直接设置enableIFA=true，但IFA模式会跳过mask数据处理。修复引入新标志位enableIFAMask将mask shape处理与IFA计算路径解耦
- 修复模式：引入新布尔标志位拆分原有标志位的双重语义
- 可审查性：中
- 审查规则建议：当一个布尔标志位在多个不同功能点被使用时，审查是否存在语义过载；新增功能与现有标志位交互时需检查所有使用点

### 57b23414 FAG sink精度修复
- 根因类别：多重缺陷(workspace偏移计算+padding硬编码+初始化地址错误+索引计算公式错误)
- 涉及文件：attention/flash_attention_score_grad/op_host/arch32/*.cpp, attention/flash_attention_score_grad/op_kernel/arch32/*.h (4个文件)
- 缺陷描述：(1) dsink workspace的大小计算和偏移量放在总workspace赋值之后，dsink未被包含导致偏移越界；(2) DataCopyPad的isPad参数无条件设true且rightPadding固定为8，对齐时产生不必要padding影响精度；(3) pre阶段初始化dsinksum workspace使用错误偏移地址且缺少core type检查；(4) dsinksum位置索引计算公式中s1Outer/s2Outer乘法逻辑错误
- 修复模式：workspace分配移入正确位置+padding改条件判断+修复初始化地址+修复索引公式
- 可审查性：低
- 审查规则建议：workspace分配和偏移量计算必须在同一代码块完成所有size累加和offset设置；DataCopyPad的padding参数应基于实际对齐需求动态计算

### ee07e5dd25 b syn bug fix
- 根因类别：pipeline同步事件管理错误
- 涉及文件：attention/flash_attention_score/op_kernel/arch32/flash_attention_score_bn2gs1s2_b.h
- 缺陷描述：sink场景分配独立的eventIdVToMte2Sink事件，但loopIdxNew>0时无条件wait该事件，若前一轮未走sink路径则wait未set的事件导致死锁。修复删除独立事件统一使用eventIdVToMte2A
- 修复模式：消除冗余同步事件，统一使用已有事件并修正set/wait条件分支
- 可审查性：中
- 审查规则建议：pipeline同步事件的SetFlag/WaitFlag必须成对出现且在所有控制流路径上平衡；新增场景不应引入独立同步事件

### bc2ff19572 fix:ComputeMm2中V的同步引起的性能劣化问题
- 根因类别：同步/事件管理错误(硬件同步时序)
- 涉及文件：attention/common/op_kernel/arch32/fia_block_cube_nonquant_gqa.h
- 缺陷描述：V buffer的L1复用逻辑中SetFlag(MTE1_MTE2)放在"下一次需要搬入V时才Set"，导致Set/Wait顺序不正确引发性能劣化。修复取消V buffer复用(reuseVBuf强制false)，重构同步事件的Set/Wait位置
- 修复模式：重构同步事件Set/Wait位置，确保正确生命周期管理
- 可审查性：低
- 审查规则建议：HardEvent的SetFlag/WaitFlag配对应静态检查确保成对出现；buffer复用优化引入的同步变更需增加性能回归测试

### 7398eb6802 fix combine accuracy
- 根因类别：内存越界/对齐约束未满足
- 涉及文件：moe/moe_distribute_combine/op_kernel/moe_distribute_combine_a2_layered.h
- 缺陷描述：Combine算子中RDMA flag写入使用DataCopy要求32字节对齐，每个flag占4个uint64_t(32字节)。当token数>240时标志位区域(1024个uint64_t)越界写入。修复改为DataCopyPad每个flag只占1个uint64_t
- 修复模式：将DataCopy替换为DataCopyPad消除对齐导致的空间浪费
- 可审查性：中
- 审查规则建议：使用DataCopy写入小数据(<32字节)时审查是否因对齐padding导致实际写入范围超出buffer

### 9191fbc2d5 grouped_matmul_swiglu_quant_v2 A4W4 增强泛化精度
- 根因类别：数值精度错误/workspace地址计算错误
- 涉及文件：gmm/grouped_matmul_swiglu_quant_v2/ 下多个文件(aclnn, tiling, kernel mid/post/pre/utils共8个文件)
- 缺陷描述：多处问题：(1) workspace偏移计算错误导致mid/post阶段读写位置不一致；(2) 量化scale计算中1/quantScale精度不够，改为QUANT_SCALE_INT8/reduceRes避免除法精度损失；(3) sizeof(half)在某些上下文被错误使用
- 修复模式：修正workspace偏移+修正量化scale计算精度+统一类型大小常量
- 可审查性：中
- 审查规则建议：workspace offset计算应有注释说明数据布局；量化/反量化scale计算避免1/x形式倒数，优先用原始分子分母直接除

### d6ee381c4d s1 syn bug fix
- 根因类别：同步事件使用错误
- 涉及文件：attention/flash_attention_score/op_kernel/arch32/flash_attention_score_s1_bn2gs1.h
- 缺陷描述：S1 BN2GS1 kernel中为sink场景分配独立eventIdVToMte2Sink，应复用eventIdVToMte2B。多分配事件且WaitFlag在不正确位置导致同步不正确
- 修复模式：合并冗余同步事件，统一使用已有事件ID
- 可审查性：低
- 审查规则建议：新增HardEvent时审查是否可复用已有事件；确保SetFlag和WaitFlag在所有分支路径上配对

### 6ecd2224fb fix oom
- 根因类别：参数超限/OOM
- 涉及文件：attention/common/op_kernel/memory_copy.h
- 缺陷描述：CopyMultiMatrixNDToNZ中srcNdMatrixStride超过ND_MATRIX_STRIDE_LIMIT时原先在调用点通过if-else做for循环逐个搬运。修复将边界处理内聚到函数内部：超限时循环调用CopySingleMatrixNDToNZ
- 修复模式：将边界条件处理内聚到底层函数，消除调用方重复的fallback逻辑
- 可审查性：中
- 审查规则建议：硬件API参数有范围限制时应在封装函数中统一处理边界

### 95e5eff07e 修复moe_init_routing_v3/moe_re_routing子包编译问题
- 根因类别：编译错误/API迁移不兼容
- 涉及文件：cmake/func.cmake, moe/moe_re_routing/op_kernel/arch35/*.h (2个), scripts/ci/ascend910_95/ops_transformer_operator_list.yaml
- 缺陷描述：moe_re_routing kernel使用旧命名空间函数(ops::CeilAlign等)需迁移到新命名空间(Ops::Base::CeilAlign)；yaml中算子被注释掉导致不编译
- 修复模式：API命名空间迁移 + 取消yaml编译排除
- 可审查性：高
- 审查规则建议：API命名空间迁移时应全局搜索所有使用旧命名空间的文件确保一次性完成

### 8b03f15583 Deter fix GQA BAND problem And L2 problem
- 根因类别：C++链式比较表达式错误 + 变量未赋值 + 边界条件缺失
- 涉及文件：attention/flash_attention_score_grad/op_host/*.cpp, attention/flash_attention_score_grad/op_kernel/arch35/deter.h, kernel_base.h
- 缺陷描述：(1) CalGQABandIndex中使用C++链式比较if(x-p+1<=y<=x+q-1)，C++按左结合律先算得bool再与右侧比较导致逻辑错误，应改为&&连接；(2) bandInfo.n未赋值；(3) length未处理负数(xMax<xMin)导致超L2时aicore error
- 修复模式：修正链式比较为逻辑与表达式+补充变量赋值+增加负值边界保护
- 可审查性：高
- 审查规则建议：启用编译器warning检测链式比较(如-Wparentheses)；range计算结果用于循环/索引前必须检查非负

### 3d2910d234 修复NSAG basic模板精度问题
- 根因类别：并发竞态/同步缺失
- 涉及文件：attention/nsa_selected_attention_grad/op_kernel/nsa_selected_attention_grad_bs1_basic.h
- 缺陷描述：部分AI Core(aic)已开始往dk/dv workspace写数据，而AI Vector(aiv)还在执行atomicClean(清零)，导致aic写入的有效数据被aiv清零。修复在vecOp.Init后增加SyncAll()全局同步
- 修复模式：增加SyncAll同步屏障消除aic/aiv竞态
- 可审查性：低
- 审查规则建议：多核场景下对共享workspace的写入前/清零后必须有显式同步屏障(SyncAll)

### 8994257f76 修改qSeqLen和kvSeqLen只有一个值为0时有问题bug
- 根因类别：初始化时序错误
- 涉及文件：attention/flash_attention_score_grad/op_kernel/arch32/basic_modules/addr_compute_det.h
- 缺陷描述：UpdateSeqLen中lastBatchQSum/lastBatchKSum的计算被放在bIdx自增之后通过getTotalLen(bIdx-1)获取，当qSeqLen或kvSeqLen只有一个为0时bIdx先自增导致获取的lastBatch累积长度不正确
- 修复模式：调整初始化语句位置，确保在bIdx自增前获取旧值
- 可审查性：中
- 审查规则建议：函数中对索引自增后立即通过idx-1回溯获取旧状态的模式应重构，改为先保存旧值再自增

### b7a83e7a20 fix aclnn
- 根因类别：参数校验缺失/接口实现不完整
- 涉及文件：gmm/grouped_matmul_swiglu_quant_v2/ 下aclnn相关文件(utils.h, .cpp)
- 缺陷描述：多处缺陷：(1) 结构体成员dequantMode/quantMode等未初始化(无默认值)；(2) 缺少空tensor检查(CheckEmptyTensor)；(3) MX量化模式下缺少属性校验(CheckMXAttrs)和shape校验(CheckMXShape)；(4) bias未通过SetBias传入builder
- 修复模式：补充参数默认初始化+增加多层校验逻辑
- 可审查性：高
- 审查规则建议：struct成员变量必须有默认初始化值；aclnn接口必须对所有输入tensor做空检查和empty检查

### ee7721a2fe 修复新分核方式由于全遍历基本块导致性能劣化
- 根因类别：算法/分核策略缺陷
- 涉及文件：attention/flash_attention_score/op_host/flash_attention_score_tiling_regbase.cpp, attention/flash_attention_score/op_kernel/arch35/flash_attention_score_kernel_train.h
- 缺陷描述：新分核模式(splitCoreMode=1)中每个核遍历所有基本块(totalSize)通过if跳过非自己的块，导致无效遍历开销大。修复改为预先计算每核分配的基本块数量，核只遍历自己的份额
- 修复模式：从"全遍历+过滤"改为"预计算分配+直接映射"
- 可审查性：低
- 审查规则建议：分核循环中避免全遍历+continue跳过的模式，应预先计算各核工作范围

### 23d96204da fia full quant aclgraph bug
- 根因类别：条件判断缺失
- 涉及文件：attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp
- 缺陷描述：CalcInnerSize中rope场景下对sInnerSize_的调整逻辑缺少!isWorkspace_条件，导致aclgraph(workspace)模式下FlashDecode时tiling错误
- 修复模式：增加!isWorkspace_前置条件
- 可审查性：高
- 审查规则建议：已有条件分支上新增场景时需审查所有受影响的条件路径是否需要增加新场景的排除/包含逻辑

### 11c0ab3b7b kernel编译选项修复
- 根因类别：构建配置错误(编译选项在错误的条件分支内)
- 涉及文件：attention/kv_quant_sparse_flash_attention/op_host/CMakeLists.txt等4个CMakeLists.txt
- 缺陷描述：add_ops_compile_options被错误放在if(BUILD_OPS_RTY_KERNEL)分支内，导致非RTY kernel构建时不会设置编译选项(如--cce-auto-sync=off)
- 修复模式：将编译选项从条件分支中提取为无条件设置
- 可审查性：高
- 审查规则建议：编译选项和模块注册应分开管理，编译选项不应依赖于构建模式条件

### a6999b032f fix bug : init output out of range
- 根因类别：数组越界/边界检查缺失
- 涉及文件：attention/common/op_kernel/arch32/fia_kernel_nonquant.h, fia_kernel_nonquant_mla.h
- 缺陷描述：InitOutput初始化输出时tmpBlockIdx*singleCoreSize可能超过totalOutputSize，导致对GM地址越界初始化。修复增加边界检查条件
- 修复模式：增加越界保护条件
- 可审查性：高
- 审查规则建议：对GM地址的写操作前必须验证偏移量在有效范围内；InitOutput/memset类操作必须检查size>0

### 330ff015fa revert add py
- 根因类别：构建配置错误(文件安装范围过大)
- 涉及文件：CMakeLists.txt, cmake/custom_build.cmake
- 缺陷描述：file(GLOB dynamic_impl)将dynamic目录下所有.py文件glob出来，foreach循环中对每个op_name都install全部.py文件导致每个算子安装了全部dynamic实现文件。修复改为按算子名精确匹配
- 修复模式：将glob全量匹配改为按算子名精确匹配安装文件
- 可审查性：高
- 审查规则建议：CMake中file(GLOB)配合install时审查安装范围是否与预期一致

### 5d6a3b1ce6 sink s1 bug fix
- 根因类别：变量错误赋值/UB地址覆盖
- 涉及文件：attention/flash_attention_score/op_kernel/arch32/flash_attention_score_s1_bn2gs1.h
- 缺陷描述：非sink场景(else分支)错误地将expUb重新赋值为maskTBufPing地址并重设ShapeInfo，覆盖了expUb原来的有效地址导致后续SoftmaxFlashV2计算使用错误buffer
- 修复模式：删除错误的变量重新赋值
- 可审查性：中
- 审查规则建议：条件分支中对buffer/tensor变量的重新赋值应审查是否破坏了变量在后续代码中的预期语义

### d1258036 Bugfix for Stabilize dK gradient computation in FAG SameAB template
- 根因类别：硬件流水线同步缺失
- 涉及文件：attention/flash_attention_score_grad/op_kernel/arch32/flash_attention_score_grad_s1s2_bn2gs1s2_sab.h
- 缺陷描述：CalcDkvR函数中DataCopyPad/DataCopy(MTE2搬运)执行前缺少V_MTE2同步屏障，MTE2搬入操作可能与Vector单元对同一buffer的前序写操作产生数据竞争，使dK梯度值不确定
- 修复模式：在4处DataCopy/DataCopyPad调用前各插入SetFlag<V_MTE2>+WaitFlag<V_MTE2>同步原语
- 可审查性：中
- 审查规则建议：DataCopy/DataCopyPad调用前检查是否存在对应流水线同步事件，尤其当buffer被多个硬件单元共享时

### 850e623f fix intercept error
- 根因类别：宏副作用/校验逻辑反转
- 涉及文件：gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/aclnn_grouped_matmul_swiglu_quant_v2_utils.h
- 缺陷描述：checkMxfp4InputShape函数中CHECK_COND宏的条件方向与预期相反，导致校验形同虚设。CHECK_COND(kValue != MXFP4_K_CONSTRAINT, ...)在条件为false时才报错，与if判断语义相反
- 修复模式：将CHECK_COND宏替换为显式if+OP_LOGE+return false，消除宏语义歧义
- 可审查性：高
- 审查规则建议：CHECK_COND等校验宏的条件方向需仔细审查；优先使用显式if+return替代带隐式return的校验宏

### 4ac6230d IFA非TND场景误拦截修复
- 根因类别：条件判断过度约束
- 涉及文件：attention/incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp
- 缺陷描述：ProcessActualSeqLen中actualSeqLengthsQ处理被isPFAFlag_条件守护，非TND(如BNSD)场景下isPFAFlag_为false导致即使传入合法actualSeqLengthsQ也被跳过，后续使用未初始化的actualSeqLenQFlag_
- 修复模式：移除isPFAFlag_&&前置条件限制
- 可审查性：高
- 审查规则建议：新增条件守护时需Review所有调用路径确认不遗漏合法场景；optional输入tensor处理不应依赖不相关的模式标记

### 39b245e1 A5matmulallreduce在低比特通信下修复datacopy对齐问题
- 根因类别：DMA对齐粒度错误
- 涉及文件：mc2/matmul_all_reduce/op_kernel/arch35/matmul_all_reduce_dequant_perchannel.h等3个文件
- 缺陷描述：DataCopyParams的stride计算使用BYTE512(512字节)作为对齐基准，但int8低比特通信场景下应按BYTE32(32字节)对齐；allgatherOutCopyParams的srcStride固定为0，遗漏了多行搬运时的输入stride
- 修复模式：修正对齐粒度常量(512->32)，补全DataCopy双向stride计算
- 可审查性：中
- 审查规则建议：DataCopyParams中stride对齐粒度必须与实际数据类型的最小搬运块大小匹配

### 075d34fc fix scatterPaKvCache
- 根因类别：循环控制流错误(continue应为break)
- 涉及文件：attention/scatter_pa_kv_cache/op_kernel/arch35/下4个文件(alibi_fully/not_fully_load, rope_fully/not_fully_load)
- 缺陷描述：CopyInKey/CopyInValue/CopyOutKey/CopyOutValue函数循环内k>=seqLen_时使用continue，由于k单调递增一旦条件成立后续全部空转。offsetIndex远大于seqLen_时大量无用迭代
- 修复模式：将continue改为break立即退出循环
- 可审查性：高
- 审查规则建议：单调递增循环变量的越界检查必须使用break而非continue；检测"单调变量+范围检查+continue"反模式

### a0077ae9 TND精度问题修改
- 根因类别：Layout分支遗漏
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_block_cube.h, flash_attention_score_kernel_base.h, infer_flash_attention_kvcache.h
- 缺陷描述：FlashAttention kernel中GS1合轴(isPfaGS1Merge)逻辑仅处理BSNGD格式，遗漏语义等价的TNGD格式(6处条件需扩展)。attentionOutStride在isPfaGS1Merge场景需特殊处理但原代码只走isGqa分支导致TND格式输出stride错误
- 修复模式：在Layout判断条件中补充TNGD/TND分支，新增isPfaGS1Merge优先判断
- 可审查性：中
- 审查规则建议：新增Layout格式时全局搜索所有涉及已有格式的条件分支确认新格式是否需相同处理

### 60e20190 修复 cust example 问题
- 根因类别：构建配置缺陷-环境变量处理遗漏
- 涉及文件：build.sh
- 缺陷描述：build_example()中设置CUST_LIBRARY_PATH/CUST_INCLUDE_PATH时只从ASCEND_OPP_PATH派生，未考虑ASCEND_CUSTOM_OPP_PATH已设置的场景，导致自定义路径下编译example失败
- 修复模式：增加ASCEND_CUSTOM_OPP_PATH非空时的条件分支
- 可审查性：中
- 审查规则建议：构建脚本使用路径环境变量时检查是否有对应的自定义/覆盖路径变量

### b4ba1e01 修复新GMM A8W4 MSD kernel的精度问题
- 根因类别：tiling参数常量错误+workspace大小计算错误
- 涉及文件：gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp, gmm/grouped_matmul/op_kernel/grouped_matmul_autotiling_a8w4.h
- 缺陷描述：(1)SetA8W4HPTiling中K维split阈值TOTAL_K_THRESHOLD设为7168但实际硬件约束要求6656(不对齐)；(2)workspace大小使用aic*固定常量*sizeof(uint32_t)与实际输出M*N*sizeof(int16_t)不匹配
- 修复模式：修正tiling阈值常量；workspace计算改为基于实际输出tensor尺寸
- 可审查性：中
- 审查规则建议：workspace/buffer大小计算应基于实际tensor维度而非硬编码常量；tiling阈值修改应有硬件规格文档佐证

### b0eb19f4 sink b and s1 bug fix
- 根因类别：并发同步缺陷-buffer复用缺少同步+buffer地址错误
- 涉及文件：attention/flash_attention_score/op_kernel/arch32/flash_attention_score_bn2gs1s2_b.h, flash_attention_score_s1_bn2gs1.h
- 缺陷描述：(1)hasSink场景SoftMaxCompute中expUb使用maskTBufPing空间，但主循环MTE2和V流水间缺少eventIdVToMte2Sink同步事件导致数据竞争；(2)hasSink分支调用SoftmaxFlashV2前未将expUb重新指向maskTBufPing地址，expUb仍指向之前buffer导致指数值写入错误位置
- 修复模式：新增eventIdVToMte2Sink同步事件；显式将expUb重绑定到maskTBufPing地址
- 可审查性：低
- 审查规则建议：多pipeline stage共用buffer时必须有对应SetFlag/WaitFlag同步；buffer复用场景应注释标注所有使用者

### 0667453c fix MoeInitRoutingQuantV2
- 根因类别：接口参数缺陷-值传递应为引用+模板参数错误+off-by-one
- 涉及文件：moe/moe_init_routing_quant_v2/op_kernel/moe_v2_gather_dynamic_quant.h
- 缺陷描述：(1)OnceCopyOut的curLoopRow和initialRow参数用值传递但函数内需修改并反映到调用方，应为引用传递；(2)TQue模板参数用T(输入类型)应为quantType(量化类型)导致queue深度错误；(3)CopyOutXQuant1H调用OnceCopyOut传入row但compute已处理row+1，应传row-1
- 修复模式：改为引用传递；修正模板参数；修正row索引为row-1
- 可审查性：高
- 审查规则建议：函数内修改参数值时检查是否需要引用传递；模板参数确认使用正确的类型参数；流水线CopyOut索引应与Compute索引匹配

### d79f1ea0 移除重复的DFX日志定义，修复MTE越界问题
- 根因类别：DFX宏参数列表错误
- 涉及文件：attention/flash_attention_score_grad/op_api/aclnn_flash_attention_score_grad.cpp
- 缺陷描述：DFX_IN宏参数列表末尾多传了dsinkOut(输出参数)，导致对未初始化的输出tensor进行越界读取(MTE越界)
- 修复模式：从DFX_IN参数列表移除dsinkOut
- 可审查性：高
- 审查规则建议：DFX_IN和DFX_OUT参数应分别严格对应输入和输出参数

### b305130f dispatch fullmesh 代码修正
- 根因类别：阈值比较条件错误
- 涉及文件：mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h
- 缺陷描述：aivUsedCumSum_上限clamp条件与aivNum_比较，但实际意图是CumSum最多使用一半AIV核(另一半留给AllToAll)，应与aivNum_/2比较。错误导致CumSum占用过多核AllToAll核数不足
- 修复模式：比较条件从>=aivNum_改为>=(aivNum_/2)
- 可审查性：高
- 审查规则建议：资源分配上限检查应与实际约束(而非总量)比较

### e7278f18 gmm error code return fix
- 根因类别：返回值未检查
- 涉及文件：gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul.cpp
- 缺陷描述：CheckFunctionParams调用CheckNonQuantMatmulDataType后未检查返回值，直接丢弃aclnnStatus导致非法数据类型组合不被拦截
- 修复模式：用CHECK_COND宏包裹调用检查返回值
- 可审查性：高
- 审查规则建议：返回状态码的函数调用必须检查返回值；使用[[nodiscard]]强制编译期检查

### da0b254a 修复FA sparseMode=1精度问题
- 根因类别：DataCopy stride溢出
- 涉及文件：common/include/kernel/util.h
- 缺陷描述：BoolCopyIn函数中DataCopyParams的srcStride字段(uint16_t最大65535)，当totalS2Size-s2Size>UINT16_MAX时stride溢出，导致MTE按错误stride搬运。sparseMode=1场景attenMask总列数大触发溢出
- 修复模式：增加溢出判断分支，溢出时改用DataCopyExtParams(stride字段更大类型)
- 可审查性：中
- 审查规则建议：使用DataCopyParams时检查stride/blockLen是否可能超uint16_t；大尺寸tensor应优先使用ExtParams版本

### 7faeede8 bugfix : mlapo N>8k
- 根因类别：buffer偏移计算错误
- 涉及文件：attention/mla_preprocess/op_kernel/mla_preprocess_bf16.h, mla_preprocess_fp16.h
- 缺陷描述：量化路径中tmpfp16 buffer偏移使用OFFSET_SUM*num_col_align_withStride_fp32*2，正确偏移应为OFFSET_GAMMA*num_col_align_withStride_fp32。N>8k时错误偏移导致tmpfp16与其他buffer区域重叠覆盖有效数据
- 修复模式：修正偏移为OFFSET_GAMMA*num_col_align_withStride_fp32
- 可审查性：低
- 审查规则建议：UB buffer空间规划应有明确layout图标注每个偏移量用途和生命周期

### 717a4593 fix mm5/mm6 reluGrad sync bug
- 根因类别：并发同步缺陷-硬件事件同步遗漏
- 涉及文件：attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/sparse_lightning_indexer_grad_kl_loss_service_cube.h
- 缺陷描述：ComputeMm5和ComputeMm6共享L1上reluGrad数据但缺少同步事件：(1)mm5在CopyInMm5AToL1前未等待上一轮mm6使用完毕；(2)mm5完成后未通知mm6可以开始；(3)IS_RELUGRAD_REUSE=false时RELU_GRAD_EVENT的SetFlag位置错误(循环内应为循环外)
- 修复模式：新增SYNC_MM_RELU_GRAD_EVENT(EVENT_ID4)和SYNC_MM5_MM6_EVENT(EVENT_ID5)同步事件，在mm5/mm6数据搬运前后正确插入Wait/Set配对
- 可审查性：低
- 审查规则建议：共享buffer的生产者-消费者路径必须有配对的同步事件；SetFlag在循环内外的位置需与数据访问模式一致

### 2a2c5259 [FAG] fix outer drop bug
- 根因类别：初始化遗漏-GM buffer未设置
- 涉及文件：attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_block_vec.h
- 缺陷描述：IS_DROP=true且dropoutIsDivisibleBy8==0时，dropMaskWorkspaceGm缺少SetGlobalBuffer调用，后续使用该GM tensor访问未初始化地址
- 修复模式：在Init中增加条件判断补充SetGlobalBuffer调用
- 可审查性：中
- 审查规则建议：if constexpr控制的GM buffer初始化需检查所有条件组合(如IS_DROP+divisibleBy8)是否都有覆盖

### e4b3ba6d 主线仓[LI\QLI][-inf排序异常]
- 根因类别：参数顺序错误-实参与形参不匹配
- 涉及文件：attention/lightning_indexer/op_kernel/lightning_indexer_vector.h, attention/quant_lightning_indexer/op_kernel/quant_lightning_indexer_vector.h
- 缺陷描述：MergeSort函数中MrgSort4Info的elementLengths[0]/[1]和MrgSortSrcList的src1/src2赋值与API语义相反，mrgDst(已排序)和mrgSrc(新数据)的对应关系被颠倒，-inf输入时排序异常。LI和QLI两个算子存在相同bug
- 修复模式：交换elementLengths和srcList的赋值
- 可审查性：中
- 审查规则建议：调用排序/合并API时检查source参数和length参数顺序是否与API定义一致；相似算子的相同函数保持同步

### 5accdd66 assert bug fixes
- 根因类别：assert使用不当导致DFX信息丢失
- 涉及文件：mc2/distribute_barrier/op_kernel/distribute_barrier.h
- 缺陷描述：超时检测中直接assert(duration<timeOut_)，但assert触发时pipeline中可能有未完成指令导致DFX采集不完整
- 修复模式：assert前先PipeBarrier<PIPE_ALL>()确保所有pipeline完成再触发assert
- 可审查性：高
- 审查规则建议：AI Core kernel中assert前需先PipeBarrier<PIPE_ALL>()刷新pipeline

### 7655b1c5 MlaProlog修改打印格式错误的问题
- 根因类别：日志格式符类型不匹配
- 涉及文件：attention/mla_prolog/op_host/mla_prolog_tiling_check.cpp
- 缺陷描述：OP_LOGE中5处用%s打印GetDimNum()返回的整数值，%s将整数解释为字符串指针可能导致崩溃或乱码
- 修复模式：%s改为%d
- 可审查性：高
- 审查规则建议：静态分析检测printf格式符与实参类型不匹配

### c9b8cfbe DFX返回问题
- 根因类别：错误码语义误用
- 涉及文件：gmm/grouped_matmul_swiglu_quant/op_host/op_api/aclnn_grouped_matmul_swiglu_quant_utils.h
- 缺陷描述：CheckInputOutShape/CheckDtypeValid/CheckFormat三个校验函数失败时用ACLNN_ERR_PARAM_NULLPTR(空指针)错误码，但校验的是shape/dtype/format有效性应返回ACLNN_ERR_PARAM_INVALID
- 修复模式：替换为ACLNN_ERR_PARAM_INVALID
- 可审查性：高
- 审查规则建议：错误码须与实际校验语义匹配；ACLNN_ERR_PARAM_NULLPTR仅用于空指针检查

### a374ae05 fix prof
- 根因类别：条件判断数据源错误+冗余SyncAll
- 涉及文件：attention/flash_incre_attention/common/op_kernel/arch32/fia_kernel_nonquant_mla.h
- 缺陷描述：IsInitAttentionOutGm中使用constInfo.attenMaskFlag判断但应使用tilingData->maskParams.attenMaskFlag(constInfo可能过时)；skipInitOutputFlag=true时未启用核仍执行SyncAll()可能导致superkernel挂死
- 修复模式：修正数据源为tilingData；新增skipInitOutputFlag避免不必要的初始化和同步
- 可审查性：低
- 审查规则建议：条件判断中数据源(constInfo vs tilingData)需使用最新值；多核SyncAll需所有核保持参与一致性

### 586493dc MoeReRouting dsv3网络问题修复
- 根因类别：API误用-CeilDiv与CeilAlign语义混淆
- 涉及文件：moe/moe_re_routing/op_host/moe_re_routing_r_tiling.cpp等5个文件
- 缺陷描述：从gitee迁移到gitcode时所有CeilAlign调用被错误替换为CeilDiv。CeilAlign(x,align)=ceil(x/align)*align(对齐后值)而CeilDiv(x,align)=ceil(x/align)(块数)，ubFactor_从130016变为4063导致buffer分配严重不足，ST通过率仅15%
- 修复模式：12处CeilDiv恢复为CeilAlign
- 可审查性：高
- 审查规则建议：代码迁移时CeilAlign和CeilDiv等语义相近但结果差异巨大的函数需逐个确认

### 16623a49 [fix] fix li liq kvaSFA
- 根因类别：DataCopy stride和寻址计算错误
- 涉及文件：attention/kv_quant_sparse_flash_attention/op_kernel/kv_quant_sparse_flash_attention_service_vector_mla.h
- 缺陷描述：MergeKv函数尾部数据搬运中DataCopyExtParams的blockCount/blockLen/dstStride与GM实际数据布局不匹配(连续单块应为多块带stride)；GM地址计算乘数headDim应为blockElementNum
- 修复模式：改为多块带stride拷贝，修正地址乘数
- 可审查性：低
- 审查规则建议：DataCopy的blockCount/blockLen/stride组合需与GM数据物理排布匹配

### 8b58f7c1 Revert "optimize gmm split_k"
- 根因类别：性能优化引入功能回退(Revert)
- 涉及文件：gmm/grouped_matmul/op_host/grouped_matmul_host_util.h等4个文件
- 缺陷描述：原始优化引入split_k场景下的baseM/baseN/baseK特殊设置(256/128/64)和IsOutputDisableL2Cache函数，在某些网络场景下导致功能异常被完整revert
- 修复模式：完整revert
- 可审查性：中
- 审查规则建议：性能优化需充分多场景回归测试；split_k tiling参数修改需验证多种M/N/K组合

### 007311e4 add memoryCopy && fix
- 根因类别：缓存复用逻辑错误
- 涉及文件：fia_block_cube_nonquant_gqa.h, axis.h, memory_copy.h
- 缺陷描述：ComputeMm1中Q的L1缓存复用未判断canFullLoadQ就标记reuseQBuf=true，Q无法全部装入L1时使用无效缓存；qL1Snapshot.firstBufId双重取模错误
- 修复模式：增加canFullLoadQ前置条件校验+引入snapshot缓存机制
- 可审查性：低
- 审查规则建议：缓存复用逻辑需验证buf生命周期覆盖所有使用路径

### 26fea31d fix mlaprolog tilingkey
- 根因类别：模板配置选项错误
- 涉及文件：mla_prolog_tiling.cpp, mla_prolog_template_tiling_key.h
- 缺陷描述：CV_MODE模板同时列出AIC_1_1和AIC_1_2两种模式但算子只支持1:2，缺少aicNum:aivNum比例校验
- 修复模式：移除不支持的模板选项+增加核心配比校验
- 可审查性：中
- 审查规则建议：模板选择宏参数列表变更应与host端tiling校验保持一致

### aa8a875d 修复告警
- 根因类别：结构体位域压缩+缺失return路径
- 涉及文件：util_regbase.h, flash_attention_score_tiling_regbase.cpp/.h
- 缺陷描述：CVSharedParams结构体splitCoreMode字段导致超CacheLine限制风险，dSizeRope位域从12位缩为11位腾出空间；IsUseSpliteCoreMode缺少else return false路径
- 修复模式：位域压缩+补充缺失return
- 可审查性：中
- 审查规则建议：位域修改需验证缩减位宽不导致数据截断；函数所有分支须有return

### 5889364f dropmask percision bugfix
- 根因类别：workspace内存地址计算错误
- 涉及文件：flash_attention_score_block_vec_base/infer/train.h等9个文件
- 缺陷描述：dropmask boolMode下workspace地址布局错误——dropMaskGm设在workspace起始地址，但CalcWorkspaceSize中WORK_SPACE_RESERVE_SIZE已计算了偏移，导致dropmask数据与bmm2/vec2 workspace区域地址重叠数据互相覆盖
- 修复模式：修正workspace地址布局+通过tiling参数传递dropMaskAddrOffset
- 可审查性：低
- 审查规则建议：workspace内存布局变更必须同时审查host端size计算和kernel端offset使用的一致性

### c66f5c5b 修复GQA模板CopyQToL1同步问题
- 根因类别：硬件事件同步时序错误
- 涉及文件：fia_block_cube_nonquant_gqa.h
- 缺陷描述：ComputeMm1中Q搬入L1的事件同步存在时序错误——AllocEventID中对所有Q_EVENT预先SetFlag<MTE1_MTE2>，但Q_L1 buf生命周期可能跨越多轮MM1，预设SetFlag导致新数据覆盖正在使用的Q_L1数据
- 修复模式：事件同步从全局预分配改为按需设置(lazy set)
- 可审查性：低
- 审查规则建议：MTE事件SetFlag/WaitFlag配对需验证Set在数据使用完毕后、Wait在下次搬运前

### 5350798d fix bmm2fdout in flash_attention_score
- 根因类别：数据搬出地址偏移计算错误
- 涉及文件：flash_attention_score_block_vec_infer.h
- 缺陷描述：Bmm2FDOut中(1)blockCount使用halfS1RealSize而非vec2S1RealSize；(2)输出GM地址偏移缺少vec2S1Idx*vec2S1BaseSize*dSizeV项导致多次迭代写同一地址后覆盖前
- 修复模式：修正blockCount+补充分块索引偏移
- 可审查性：中
- 审查规则建议：DataCopy输出GM地址计算必须包含所有循环维度偏移量

### 25c11cf6 fix softmax copy and tensor empty check
- 根因类别：DataCopy参数语义错误+缺少空tensor校验
- 涉及文件：aclnn_ring_attention_update.cpp, ring_attention_update_tnd.h
- 缺陷描述：(1)softmax搬运DataCopyExtParams中blockCount设为headNumLoopEach(head数)但应为1(一整块连续)，blockLen应为headNumLoopEach*softmaxTailSize*floatDataSize；(2)AnalysisAxis缺少空tensor校验
- 修复模式：修正blockCount/blockLen语义+新增CheckNullTensor
- 可审查性：中
- 审查规则建议：DataCopyExtParams的blockCount=搬运块数/blockLen=每块字节数语义需严格验证

### 2e044c2a fix rope problem
- 根因类别：缺少前置条件校验
- 涉及文件：sparse_lightning_indexer_grad_kl_loss_tiling_general.cpp
- 缺陷描述：CrossShapeVerify中cross attention场景要求ROPE必须使能，但hasRope==0时未报错拦截
- 修复模式：增加OP_CHECK_IF(hasRope==0)校验
- 可审查性：高
- 审查规则建议：可选输入在某些场景为必选时应在场景入口立即校验

### cefbad33 修复PFA GQA GS1合轴 不叠加后量化
- 根因类别：功能开关遗漏
- 涉及文件：prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：CheckPFAMerge判断GQA GS1合轴兼容性时检查了多个互斥特性但遗漏enablePostQuant，后量化使能时GS1合轴路径无对应处理逻辑但仍返回true
- 修复模式：条件中补充||enablePostQuant
- 可审查性：高
- 审查规则建议：特性兼容性检查的排斥列表每次新增特性时须同步更新；考虑用白名单替代黑名单

### 6502ae00 matmul_all_reduce 拦截逻辑修正
- 根因类别：条件判断逻辑错误-维度推算错误
- 涉及文件：quant_matmul_all_reduce_tiling_910_95.cpp, weight_quant_matmul_all_reduce_tiling_910_95.cpp
- 缺陷描述：CheckAxisSize中通过GetNValue和isBTrans推算x2维度，但转置状态与实际存储shape不一致时推算错误导致校验被错误拦截/漏过；WeightQuant路径缺少pertensor场景下fp8/hif8数据类型校验
- 修复模式：改为直接从StorageShape读取实际维度；补充缺失校验
- 可审查性：高
- 审查规则建议：存在shape维度推算逻辑时审查是否应直接读取实际shape

### c310d26c fix cann-xx dir right error bug
- 根因类别：安装脚本目录权限处理错误
- 涉及文件：opp_install.sh
- 缺陷描述：install_opp()中stat -c保存目录权限是GNU特有语法，跨平台不兼容；chmod恢复逻辑在异常路径下可能导致权限错误
- 修复模式：删除冗余的权限保存恢复逻辑
- 可审查性：中
- 审查规则建议：shell脚本中stat -c存在跨平台兼容性问题

### 47f80395 FAG确定性计算优化算子问题修改
- 根因类别：计算逻辑错误+同步缺失
- 涉及文件：flash_attention_score_grad_tiling_s1s2_bn2gs1s2_basic_det.cpp, vec_op_det.h
- 缺陷描述：(1)dqPostAbsorb在非sparseMode==4时错误设为1；(2)IsDropMskCapable返回true但确定性模式不支持dropmask应返回false(还有双分号;;)；(3)OrderAccumDq末块处理缺少MTE3到MTE2的同步事件导致数据竞争
- 修复模式：修正tiling参数；添加硬件同步；修复返回值
- 可审查性：高
- 审查规则建议：确定性计算路径GM读写必须有同步事件；return true;;双分号应被lint捕获

### f99a34dd 回退mm_rs优化
- 根因类别：功能回退(Revert)-优化引入缺陷
- 涉及文件：matmul_reduce_scatter_v2目录下13个文件(~3500行)
- 缺陷描述：matmul_reduce_scatter smallM优化存在问题被整体回退，包括MatmulReduceScatterAivModeSmallM类及CRTP继承模式
- 修复模式：整体回退优化特性代码
- 可审查性：中
- 审查规则建议：大规模功能优化合入前应有充分泛化测试覆盖

### 6eaf8f29 修复SFAG S2非sparseBlockSize整数倍精度问题
- 根因类别：尾块边界处理缺失
- 涉及文件：cube3.h, cube4.h, cube5.h, cube_op.h, matmul.h, vec_op.h, sparse_flash_attention_grad_bs1_basic.h
- 缺陷描述：S2不是sparseBlockSize整数倍时最后一个block按满块计算/写出：(1)cube的mmParam.singleK越界读取无效数据；(2)ScatterFixOut按满块写覆盖有效数据；(3)attenMskEnd==64时1ULL<<64为UB
- 修复模式：新增lastBlockSize参数逐层传递，尾块用实际大小替代满块；修复64位移位UB
- 可审查性：高
- 审查规则建议：分块计算最后一块需有尾块处理逻辑；1ULL<<N当N可能等于64时为UB

### 7c7ffdc1 [fix] fix amla
- 根因类别：首轮循环初始化缺失+AtomicAdd条件错误
- 涉及文件：sparse_flash_attention_service_cube_mla.h, sparse_flash_attention_service_vector_mla.h
- 缺陷描述：BMM2 Fixpipe在isFirstSInnerLoop时跳过SetAtomicAdd，但GM workspace可能含残留数据导致首轮输出与残留直接覆盖；修复为所有轮次都AtomicAdd+首轮预填充极小偏置值
- 修复模式：统一AtomicAdd行为+初始化偏置值
- 可审查性：高
- 审查规则建议：AtomicAdd目标内存须正确初始化；首轮特殊处理应有明确初始化机制

### 64f59aae fix problem for mla softmaxlse when flashdecode
- 根因类别：初始化遗漏
- 涉及文件：fia_kernel_nonquant_mla.h
- 缺陷描述：flash decode模式初始化时softmaxLseFlag=true但缺少InitSoftmaxLseGm调用，softmaxLse输出写入未初始化GM地址
- 修复模式：增加softmaxLseFlag条件判断并调用InitSoftmaxLseGm
- 可审查性：高
- 审查规则建议：新增功能flag时审查所有kernel入口路径是否初始化对应资源

### 3c089d7e Fix some bugs and cleancode
- 根因类别：混合缺陷-L1拷贝尺寸参数错误+初始值语义不兼容
- 涉及文件：fia_tiling_shape.h/.cpp, fia_kernel_nonquant.h, fia_block_cube_nonquant_gqa.h等9个文件
- 缺陷描述：(1)CopyPToL1使用mL1.AlignedSize()应为mL1.sizeAct(实际行数)，可能L1溢出；(2)softmaxLse初始值IFA需-3.4e38而PFA需3e+99，统一用FLOAT_INF导致IFA精度异常
- 修复模式：修正L1拷贝size参数；新增isLegacyIfa区分初始值
- 可审查性：高
- 审查规则建议：L1/UB拷贝size须审查Aligned还是Actual值；初始值选择需考虑不同场景数值语义

### 3ba92827 fix 52 pfa;del redundant files
- 根因类别：不必要的跨核同步导致死锁
- 涉及文件：flash_attention_score_block_cube.h, flash_attention_score_block_vec_base.h, prompt_flash_attention_tiling_arch38.cpp等
- 缺陷描述：arch38 PFA中BMM1后SetCrossCore/BMM2前WaitCrossCore在非跨核流水场景不需要，引入死锁或性能问题
- 修复模式：移除不必要的跨核同步；补充缺失include
- 可审查性：高
- 审查规则建议：跨核同步原语必须与实际跨核数据流一致

### 780bb3c9 fix markdown
- 根因类别：tiling数据赋值遗漏
- 涉及文件：all_gather_matmul_aiv_mode_tiling.cpp等
- 缺陷描述：coctiling.rankSize = rankSize赋值遗漏导致AIV模式下rankSize未传入tiling数据
- 修复模式：补充tiling字段赋值
- 可审查性：高
- 审查规则建议：新增API参数后全局搜索所有调用点同步更新；tiling结构每个字段都应有赋值

### 6f5a8d95 Bugfix for FAG deterministic accumulation problem
- 根因类别：成员变量副作用导致计算错误
- 涉及文件：flash_attention_score_grad_s1s2_bn2gs1s2_sab.h
- 缺陷描述：CalcDqReduce/CalcDkvReduce用成员变量vecBlockNum计算分配，fallback路径直接修改vecBlockNum=coreNum产生副作用影响后续计算
- 修复模式：将成员变量改为显式函数参数消除副作用
- 可审查性：高
- 审查规则建议：函数内修改成员变量作为参数传递属于反模式，应改为显式参数

### 5bc15d9e Revert "add alibi mask feature for split fuse"
- 根因类别：功能回退(Revert)-新特性兼容性/正确性问题
- 涉及文件：fused_infer_attention_score相关21个文件
- 缺陷描述：alibi mask for split fuse特性引入tiling data结构二进制不兼容(bool替代padding字段)等问题被完整revert
- 修复模式：完整回退feature代码，恢复padding字段
- 可审查性：中
- 审查规则建议：tiling data结构修改需注意二进制兼容性

### 2b64db01 修复m=1场景Matmul精度错误问题
- 根因类别：边界条件处理缺失-硬件指令最小值约束
- 涉及文件：attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/sparse_lightning_indexer_grad_kl_loss_service_cube.h
- 缺陷描述：Mmad指令m参数直接使用singleM，当singleM=1时不满足硬件最小值要求(16)导致精度错误
- 修复模式：singleM==1时替换为16
- 可审查性：高
- 审查规则建议：检查Mmad指令m/n/k参数是否可能小于硬件最小约束

### 0c29b4ee bugfix
- 根因类别：初始化值语义错误+DMA对齐缺失
- 涉及文件：lightning_indexer_kernel.h, lightning_indexer_service_vector.h, sparse_flash_attention_service_vector_mla.h
- 缺陷描述：(1)invalid output初始化用0但语义应为负无穷(fp16:0xFC00,bf16:0xFF80)导致TopK误判；(2)DataCopy size未做32B对齐导致DMA越界
- 修复模式：初始化值改为类型相关负无穷；DataCopy替换为DataCopyPad+alignedSize
- 可审查性：中
- 审查规则建议：padding/invalid场景初始化值应符合语义(比较排序用极值非0)；DataCopy size须满足32B对齐

### fdd896e2 GMMDSQV2 fix bug
- 根因类别：多类复合缺陷(变量误用+内存对齐+同步时序+ReduceMax参数)
- 涉及文件：grouped_matmul_swiglu_quant_v2相关3个文件
- 缺陷描述：(1)N维检查用realMSize应为realNSize(拷贝错误)；(2)scaleOut offset未做32B对齐；(3)CrossCoreWaitFlag/SetFlag在for循环内导致同步时序错误可能挂死；(4)WholeReduceMax的repeatStride参数和边界处理错误
- 修复模式：修正变量名；offset做Ceil对齐；重构核间同步逻辑；对ReduceMax参数做min保护
- 可审查性：中
- 审查规则建议：边界检查维度变量与被检查维度一致；UB offset须32B对齐；核间同步flag的wait/set须在同一循环层级成对出现

### 45cfad32 修复NSAG非base模板精度问题
- 根因类别：单位混淆(字节数vs元素数)
- 涉及文件：nsa_selected_attention_grad/op_kernel/nsa_selected_attention_grad_bs1.h
- 缺陷描述：InitOutput参数语义为"元素个数"但传入的selectedKWorkspaceLen是字节数，导致初始化元素数为实际sizeof(T1)倍可能越界
- 修复模式：除以sizeof(T1)得到正确元素数
- 可审查性：高
- 审查规则建议：检查InitOutput调用length参数单位(元素vs字节)与变量语义一致

### c03d6694 修复sink下确定性精度问题
- 根因类别：条件分支缺失
- 涉及文件：flash_attention_score_grad_s1s2_bn2gs1s2_sab.h
- 缺陷描述：dsinksum计算和写入逻辑在非sink场景也被执行，向dsinksumWorkSpaceGm写入不必要数据影响确定性精度
- 修复模式：用if(sink==1)包裹dsinksum逻辑
- 可审查性：高
- 审查规则建议：特性相关workspace写入应被对应开关保护

### dfb42dfc fix sparseMode attentionOutGm 0
- 根因类别：编译期判断过度(constexpr遗漏运行时条件)
- 涉及文件：fia_kernel_nonquant.h, fia_kernel_nonquant_mla.h
- 缺陷描述：InitOutputSingleCore用if constexpr跳过TND/NTD场景初始化，但sparseMode下TND/NTD仍需初始化attentionOutGm为0
- 修复模式：替换为运行时函数调用IsInitAttentionOutGm()增加attenMaskFlag判断
- 可审查性：中
- 审查规则建议：编译期模板条件与运行时参数交叉影响时不应仅用if constexpr

### 8009dc00 [FIA] 修复actual_seq全为0的任务未分发问题
- 根因类别：边界条件处理缺失(全零输入)
- 涉及文件：attention/common/op_host/split_core.cpp
- 缺陷描述：所有batch的actual_seq_len均为0时核分配逻辑可能异常(未分发任务到任何核)
- 修复模式：检测isKvSeqAllZero后设usedCoreNum=1提前返回
- 可审查性：高
- 审查规则建议：tiling/分核逻辑需处理全零/全空退化case

### a65d88a1 解决sink场景下MTE越界问题
- 根因类别：DMA数据量超UB容量+索引计算错误
- 涉及文件：flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.cpp, flash_attention_score_grad_post.h, flash_attention_score_grad_s1s2_bn2gs1s2_sab.h
- 缺陷描述：(1)一次性DataCopy整个dsinksum数据量可能超UB容量导致MTE越界；(2)SubGrapB中dsinksum GM索引公式错误；(3)循环内高频小数据GM写入
- 修复模式：大数据量DMA拆分为分块循环；改为本地累加+循环后批量写入；修正索引公式
- 可审查性：中
- 审查规则建议：DataCopy到UB的数据量须验证不超buffer大小；循环内GM写入应改为本地累加后批量写

### b960534e fix 310p compile bug
- 根因类别：头文件目录结构错误
- 涉及文件：attention/incre_flash_attention/op_kernel/incre_flash_attention_obp.h, incre_flash_attention_tilingkey.h
- 缺陷描述：tilingkey.h放在arch32子目录下但非arch32场景(310P)也需include，导致编译找不到头文件
- 修复模式：将公共头文件移至op_kernel根目录
- 可审查性：高
- 审查规则建议：被多个架构引用的公共头文件不应在特定架构子目录；CI须覆盖所有目标平台

### 37105bc0 修复fag bn模板偶现nan问题
- 根因类别：流水线同步缺失(V->MTE3)
- 涉及文件：flash_attention_score_grad_ngs1s2_bn.h
- 缺陷描述：Cast(V管道)输出castedDvDropResPad未完成就被DataCopyPad(MTE3)读取，缺少V->MTE3同步屏障导致偶现nan
- 修复模式：Cast后添加PipeBarrier<PIPE_V>和SetFlag/WaitFlag<V_MTE3>
- 可审查性：高
- 审查规则建议：跨管道数据传递前必须有SetFlag/WaitFlag同步；V管道指令后紧跟DataCopy时检查V_MTE3同步

### 0146fcc7 [fix] actualSingleProcessSInnerSize should not align
- 根因类别：不当的对齐操作
- 涉及文件：sparse_flash_attention_kernel_mla.h
- 缺陷描述：actualSingleProcessSInnerSize表示实际S内维大小被错误做了sparseBlockSize对齐，对齐后值超实际范围使用无效数据导致精度问题
- 修复模式：删除多余对齐操作
- 可审查性：高
- 审查规则建议：名称含"actual"的变量不应做对齐向上取整

### 4eb9456 fix build bugs
- 根因类别：构建配置错误 — 误注释关键函数调用
- 涉及文件：cmake/obj_func.cmake
- 缺陷描述：add_infer_modules()被错误注释掉，导致推理模块未被添加，构建失败
- 修复模式：取消注释恢复被误删的代码行
- 可审查性：高
- 审查规则建议：对cmake中注释掉函数调用的变更进行标记，特别是没有替代实现的情况下

### 6b66cbd fix ARN compile
- 根因类别：宏注册遗漏 — tiling数据类未注册
- 涉及文件：mc2/3rd/add_rms_norm/op_host/add_rms_norm_tiling.h
- 缺陷描述：MC2AddRMSNormTilingData定义了tiling数据结构但缺少REGISTER_TILING_DATA_CLASS宏调用，导致编译时无法找到该tiling类型
- 修复模式：补充遗漏的宏注册语句
- 可审查性：高
- 审查规则建议：检查END_TILING_DATA_DEF之后是否紧跟对应的REGISTER_TILING_DATA_CLASS

### 8b7d9da fix add_rms_norm
- 根因类别：命名冲突/注册错误 — tiling数据结构名称不一致
- 涉及文件：mc2/3rd/add_rms_norm/op_host/add_rms_norm_tiling.h
- 缺陷描述：AddRMSNormRegbaseTilingData命名前缀缺少MC2前缀，同时删除了不应在此头文件中出现的REGISTER_TILING_DATA_CLASS注册
- 修复模式：重命名数据结构+删除错误位置的注册宏
- 可审查性：高
- 审查规则建议：tiling数据结构命名应遵循统一的模块前缀规范；REGISTER_TILING_DATA_CLASS应与数据定义就近放置

### 14c9c25 tiling修复布尔类型
- 根因类别：数据类型错误 — tiling字段使用不兼容的bool类型
- 涉及文件：attention/lightning_indexer/op_host/lightning_indexer_tiling.h
- 缺陷描述：returnValue字段定义为bool类型，但tiling data框架不正确支持bool的序列化/对齐(bool在不同平台大小不确定)，导致tiling数据传输或解析异常
- 修复模式：将bool改为uint32_t
- 可审查性：高
- 审查规则建议：TILING_DATA_FIELD_DEF中禁止使用bool类型，应使用uint32_t代替

### a2b2d3b empty tilingkey bug fix
- 根因类别：tiling key计算错误 — host/kernel侧条件编译差异
- 涉及文件：attention/flash_attention_score/op_host/flash_attention_score_tiling.cpp
- 缺陷描述：空输入场景下GET_TPL_TILING_KEY宏依赖__CCE_AICORE__条件编译，host侧和kernel侧选择不同的模板参数组合文件导致tiling key不匹配
- 修复模式：用固定常量FA_EMPTY_TILING_KEY=1替代宏生成的tiling key
- 可审查性：中
- 审查规则建议：空输入/特殊路径的tiling key不应依赖模板宏生成；检查__CCE_AICORE__条件编译在host侧代码中的使用

### 1f2208c 修复100左右值域下精度问题
- 根因类别：数值计算遗漏 — 缺少下界截断
- 涉及文件：attention/sparse_lightning_indexer_grad_kl_loss/op_kernel/sparse_lightning_indexer_grad_kl_loss_vector.h
- 缺陷描述：reduceSumPTmpUb在执行后续运算前未做Maxs截断，而相邻的reduceSumYResTmpUb已有此操作。当输入值域在100左右时可能出现极小负值导致精度异常
- 修复模式：对遗漏的tensor增加对称的Maxs下界截断操作
- 可审查性：高
- 审查规则建议：同一计算流程中对称的tensor应有一致的数值截断处理

### 4d35a80 [bugfix] fix race condition between two continuous repeats of a vector instruction
- 根因类别：并发竞争 — 向量指令repeat间内存竞争+越界访问
- 涉及文件：attention/mla_preprocess/op_kernel/mla_preprocess_bf16.h, mla_preprocess_fp16.h
- 缺陷描述：(1)bf16中PER_TENSOR_ASYMM_QUANT分支相邻repeat间存在内存竞争(stride重叠)；(2)fp16中PreloadB()缺少core_idx<num_core边界检查
- 修复模式：重构循环结构消除竞争+添加边界守卫
- 可审查性：低
- 审查规则建议：向量指令repeat stride必须确保不同repeat间无内存重叠；使用GetBlockIdx()后应检查是否超出有效核数

### b3aa078 SFA异常拦截
- 根因类别：输入校验遗漏 — 缺少维度一致性检查
- 涉及文件：attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp
- 缺陷描述：GetRopeHeadDim()按qLayout取D轴维度值但未校验query和query_rope维度数是否一致，维度数不同时按相同layout取轴会越界
- 修复模式：增加维度数一致性前置校验
- 可审查性：高
- 审查规则建议：对多个tensor按相同layout索引前必须校验维度数一致

### 677b14c fix mmrs
- 根因类别：条件编译不完整 — host侧缺少tiling模板参数
- 涉及文件：mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_v2_tiling_key.h
- 缺陷描述：ASCENDC_TPL_SEL宏只有A5和A2/A3两个条件分支，host侧编译时两个宏都不满足导致缺少模板参数定义
- 修复模式：增加#else分支覆盖host侧场景
- 可审查性：中
- 审查规则建议：平台条件编译宏必须有#else兜底分支覆盖host侧；tiling key模板选择需同时考虑kernel和host侧

### bf74834 fix cv1:1 band initoutput bug
- 根因类别：硬编码错误 — c:v比例硬编码为1:2
- 涉及文件：attention/common/op_kernel/arch32/fia_kernel_nonquant_mla.h
- 缺陷描述：InitOutputSingleCore()中硬编码2*usedCoreNum(假设c:v=1:2)，c:v=1:1配置下AIV核数计算错误导致部分输出未被初始化
- 修复模式：用constInfo.subBlockNum替代硬编码的2
- 可审查性：高
- 审查规则建议：禁止硬编码c:v比例值，统一使用配置参数subBlockNum

### 42817b8 修复KV最大维度的误改
- 根因类别：常量值错误 — shape校验上限被误改
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：INPUT_QKV_SHAPE_MAX_DIMS从5被误改为4，导致5维shape(合法输入)被校验拦截
- 修复模式：恢复正确的常量值(4->5)
- 可审查性：高
- 审查规则建议：修改约束常量时需同步检查所有合法输入是否仍在范围内

### f41a009 修复sinck场景下ROPE报错
- 根因类别：条件判断作用域错误 — 校验放在了错误的条件分支外
- 涉及文件：attention/flash_attention_score_grad/op_api/aclnn_flash_attention_score_grad.cpp
- 缺陷描述：sparseMode==6时禁止使用ROPE的检查放在了无论是否有query_rope/key_rope都会执行的位置，导致sinck场景即使不使用ROPE也报错
- 修复模式：将校验逻辑移入queryRope!=nullptr的条件分支内
- 可审查性：高
- 审查规则建议：参数互斥校验应放在相关参数非空的条件分支内部

### 166b74d gmm/gmm_weight_nz core28 fix
- 根因类别：多核调度逻辑错误 — tail split条件不当+对齐函数误用
- 涉及文件：gmm/common/groupedmatmul_act/matmul/block/block_scheduler_grouped_matmul_aswt.h等
- 缺陷描述：(1)tail split默认启用但groupNum==1时不应split；(2)weightNz场景singleCoreNSplit用CeilDiv应为CeilAlign；(3)循环体缺少shape合法性检查(m/n<=0应skip)
- 修复模式：新增tailSplit控制参数+修正对齐函数+增加shape守卫
- 可审查性：低
- 审查规则建议：CeilDiv和CeilAlign是不同操作需明确选择；多核调度tail split应有明确启用条件；循环处理tile shape需检查各维度>0

### 1cc6b847 [fix] add limit of sparseblockSizev
- 根因类别：输入校验缺失 — 缺少2的幂次校验
- 涉及文件：kv_quant_sparse_flash_attention_tiling.cpp, sparse_flash_attention_tiling.cpp
- 缺陷描述：sparseBlockSize只校验了范围[1,16]或(0,128]，缺少2的幂次校验，非2的幂次值导致后续计算逻辑异常
- 修复模式：增加(val & (val-1)) != 0的位运算校验
- 可审查性：高
- 审查规则建议：对齐/幂次约束的tiling参数应做2的幂次校验

### 44816b5d dispatchV2 打点数据搬出 bug fix
- 根因类别：执行时序/初始化遗漏 — 清零操作位于early return之后
- 涉及文件：moe_distribute_dispatch_a2.h
- 缺陷描述：WaitDispatch()中performanceInfoI32Tensor_清零操作位于taskNum==0的提前返回判断之后，未分配任务的核直接return导致未清零，后续CopyPerformanceInfo()将脏数据叠加到GM
- 修复模式：将清零操作移至taskNum判断之前
- 可审查性：高
- 审查规则建议：当函数存在early return路径时，审查初始化/清理操作是否覆盖所有分支

### 0b186faa MlaProlog算子拦截问题修复
- 根因类别：输入校验缺失+字符串长度计算错误
- 涉及文件：mla_prolog_tiling.cpp, mla_prolog_tiling.h, mla_prolog_tiling_check.cpp/h
- 缺陷描述：(1)CACHE_MODE_LEN用sizeof()-1排除\0导致strncmp比较长度不足可能误匹配；(2)SetScenarioInfo缺少actualSeqLen空指针校验；(3)V3版本属性缺少值域校验
- 修复模式：修正sizeof计算+增加null校验+新增CheckAttrsRange白名单校验
- 可审查性：中
- 审查规则建议：strncmp配合sizeof使用时确认是否需要包含\0；可选输入tensor解引用前必须做空指针校验

### 5474fdf4 revert//cleancode
- 根因类别：cleancode引入回归 — 函数返回值被错误删除
- 涉及文件：fused_infer_attention_score_infershape.cpp
- 缺陷描述：cleancode将infershape函数返回类型从ge::graphStatus改为void并删除return和错误分支返回，导致不支持的layout无法正确报错
- 修复模式：Revert — 恢复返回类型为ge::graphStatus并补上GRAPH_FAILED返回
- 可审查性：高
- 审查规则建议：重构中修改函数签名时必须确认所有调用点是否依赖返回值做错误处理

### a2b032dc Revert "tilingkey rectification for redecescatterv2_and_A5_isolation"
- 根因类别：代码整改引入回归 — tilingkey重构导致功能问题
- 涉及文件：matmul_reduce_scatter_v2_def.cpp, matmul_reduce_scatter_v2_aiv_mode_tiling.cpp等
- 缺陷描述：tilingkey整改为matmul_reduce_scatter_v2引入A5 isolation和apt文件拆分导致功能回归
- 修复模式：Revert — 删除apt文件和tilingkey头文件，恢复原始单文件逻辑
- 可审查性：低
- 审查规则建议：大规模重构应分算子逐步提交并保证完整回归测试

### 67f1fffc bug fix
- 根因类别：逻辑运算符错误 — ||误用为&&
- 涉及文件：scatter_pa_kv_cache_tiling_arch35.cpp
- 缺陷描述：OP_CHECK_IF(inputKeyDimNum != DIM3 || inputKeyDimNum != DIM4)中||永远为true(一个数不可能同时等于3和4)，导致维度校验失效
- 修复模式：将||改为&&
- 可审查性：高
- 审查规则建议：!= A || != B是经典逻辑错误(恒true)，可编写静态分析规则检测

### 58a7ef18 Revert "mc2 tilingkey rectification for A2/A3 and A5_isolation"
- 根因类别：代码整改引入回归 — 同351号同源的tilingkey重构
- 涉及文件：moe_distribute_combine/dispatch系列20个文件
- 缺陷描述：moe_distribute系列4个算子的tilingkey计算和kernel拆分到apt文件的重构引入功能问题
- 修复模式：Revert — 全量恢复原始逻辑
- 可审查性：低
- 审查规则建议：同351，大规模重构应分算子逐步提交

### 2af1bc37 fix qsfa 异常场景拦截
- 根因类别：字段错用+校验逻辑不精确
- 涉及文件：kv_quant_sparse_flash_attention_tiling.cpp/h
- 缺陷描述：(1)headDimAlign_使用qkHeadDim但MLA场景q和k的headDim不同，应使用独立的qHeadDim；(2)参数存在性检查过于粗糙需按layout分场景精确校验
- 修复模式：拆分qkHeadDim为独立的qHeadDim和kHeadDim字段；重写为按场景精确校验
- 可审查性：中
- 审查规则建议：当新增模式使原有假设(q/k共用headDim)不成立时，检查所有使用该假设的代码

### a83a9fb4 修复dropout场景下dsink精度问题和softmax layout报错
- 根因类别：数据流时序错误+参数传递错误
- 涉及文件：aclnn_flash_attention_score_grad.cpp, flash_attention_score_grad_s1s2_bn2gs1s2_sab.h
- 缺陷描述：(1)dropout场景dyvBuffer在dropout mask应用之前获取了原始数据而非dropout后数据；(2)softmaxInLayout参数传递时错用defaultSoftmaxInLayout
- 修复模式：将dyvBuffer的DataCopy移到ComputeDropMask之后；修正参数传递
- 可审查性：中
- 审查规则建议：数据变换pipeline中审查下游取用点是否在变换之后；变量名相似(xxx vs defaultXxx)应重点审查

### e261e798 fix error for op def file
- 根因类别：接口定义不完整 — op def数据类型/格式映射缺失
- 涉及文件：gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_def.cpp
- 缺陷描述：算子定义文件中weight仅支持DT_INT8+FRACTAL_NZ单一类型映射，缺少INT4及不同weight_scale类型(BF16/FP16/UINT64)的支持，导致合法输入被拒绝
- 修复模式：扩展DataType/Format列表从单一类型到5组类型映射
- 可审查性：中
- 审查规则建议：op def文件中输入输出的类型映射数量应一致，且需覆盖所有合法类型组合

### 8351a306 fix mm_reduce_scatter_v2 custom compile errors
- 根因类别：编译依赖/头文件路径错误 — 自定义编译路径不可达
- 涉及文件：mc2目录下12个文件(qbmm_asw_block.h, qbmm_mix_online_dynamic.h等)及CMakeLists.txt
- 缺陷描述：自定义编译场景下引用了不可达的头文件路径(../inc/platform.h, mc2_common_def.h)
- 修复模式：内联替代外部依赖+__has_include条件编译兼容+新增mc2_common_def.h
- 可审查性：低
- 审查规则建议：新增编译模式时应验证所有头文件路径在该模式下可达；CI应覆盖所有编译模式

### 809b164a 【bugfix】MatmulAllReduce修复mxfp4精度问题
- 根因类别：数据类型特殊处理遗漏 — 4-bit类型地址偏移计算错误
- 涉及文件：mc2/matmul_all_reduce/op_kernel/arch35/matmul_all_reduce_base.h
- 缺陷描述：sizeof(XType)计算字节偏移未处理fp4x2类型(4-bit实际长度应除2)，导致mxfp4场景地址偏移翻倍读取错误数据；tail分支还有copy-paste错误(tailInfo_.aAddrOffset误写为tileInfo_)
- 修复模式：增加IsSameType<XType,fp4x2_e2m1_t>类型判断做除2处理
- 可审查性：高
- 审查规则建议：sizeof计算地址偏移时需检查非标准字节宽度数据类型；tail分支与主分支变量名一致性检查

### ef42e436 修复SFAG infershape以及支持topk=8k
- 根因类别：变量名错用 — buffer分配使用错误变量
- 涉及文件：attention/sparse_flash_attention_grad/下3个文件(vec_op.h, infershape.cpp, sparse_flash_attention_grad.cpp)
- 缺陷描述：topk buffer分配使用selectedBlockCount而非正确的selectedCountOffset，导致buffer大小计算错误
- 修复模式：变量名修正+接口宏迁移
- 可审查性：中
- 审查规则建议：buffer分配大小应与实际数据量对应的变量一致

### e2eaac95 fix n is transpose;add ut comments
- 根因类别：矩阵维度取值逻辑错误 — 转置场景下维度索引未适配
- 涉及文件：gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/aclnn_grouped_matmul_swiglu_quant_v2_utils.h
- 缺陷描述：mxfp4输入shape校验中n值从weightScale第1维获取，但weight转置场景下应从weight本身的dim(1)或dim(2)获取
- 修复模式：根据transposeWeight标志选择正确的维度索引
- 可审查性：高
- 审查规则建议：涉及矩阵转置时，所有维度索引访问都需区分转置/非转置

### e8b3df03 fix nsa_compress_attention_infer MIN_ACTUALSEQLEN
- 根因类别：常量值错误
- 涉及文件：attention/nsa_compress_attention_infer/op_host/nsa_compress_attention_infer_tiling_utils.h
- 缺陷描述：MIN_ACTUALQSEQLEN定义为2但实际最小序列长度应为1，导致actualSeqLen=1时校验不通过
- 修复模式：常量值修正(2->1)
- 可审查性：高
- 审查规则建议：边界常量定义应与算子规格文档约束一致

### ec9278e0 fix tnd tsize
- 根因类别：条件分支遗漏 — TND/NTD layout下tSize计算错误
- 涉及文件：attention/common/op_kernel/arch32/fia_kernel_nonquant.h, fia_kernel_nonquant_mla.h
- 缺陷描述：tSize统一用batchSize*qSeqSize计算，但TND/NTD layout下应通过qActSeqLensParser.GetTSize()获取实际值
- 修复模式：增加TND/NTD layout条件分支用实际序列长度替代静态计算
- 可审查性：高
- 审查规则建议：不同layout(BNSD/TND/NTD)的代码路径中所有依赖shape的计算都应区分layout

### 1eb3f0ce revert//修改拦截报错信息
- 根因类别：过度校验 — 维度检查过于严格拒绝合法输入
- 涉及文件：fused_infer_attention_score_tiling.cpp, incre_flash_attention_tiling.cpp
- 缺陷描述：(1)FIA中TND/BNSD分支增加的维度数量检查误拦截合法输入；(2)IFA中MLA场景对int8+keyRope+headDim==512的NZ format限制过于特殊化
- 修复模式：revert过度校验逻辑恢复到更宽松的合法校验
- 可审查性：高
- 审查规则建议：新增输入校验应充分验证所有合法输入组合不被误拦截

### 96f64955 static bug修改
- 根因类别：静态编译模式构建配置缺陷
- 涉及文件：attention/下4个CMakeLists.txt, scripts/util/build_opp_kernel_static.py
- 缺陷描述：(1)ENABLE_STATIC分支缺少--op_relocatable_kernel_binary选项；(2)filePath字段缺少soc前缀；(3)anonymous namespace符号未过滤导致注册失败
- 修复模式：CMake条件分支简化+Python脚本路径修正+符号过滤
- 可审查性：低
- 审查规则建议：新增编译模式时应在CI中增加端到端构建验证

### f800032f fix A5 blacklist compile fail
- 根因类别：编译黑名单配置错误
- 涉及文件：cmake/func.cmake
- 缺陷描述：A5芯片黑名单错误包含grouped_matmul_swiglu_quant和quant_grouped_matmul_dequant两个算子，导致链接失败(其他模块依赖这些算子)
- 修复模式：从黑名单中移除不应被排除的算子
- 可审查性：高
- 审查规则建议：修改编译黑名单时需验证被排除算子不被其他算子依赖

### 9f9bfd88 fix mmAllreduce quant
- 根因类别：量化逻辑修复+大规模重构混合
- 涉及文件：23个文件(mc2/matmul_all_reduce, mc2/3rd/weight_quant_batch_matmul_v2等)
- 缺陷描述：tiling key冗余模板特化裁剪+matmul_all_reduce量化场景tiling逻辑调整
- 修复模式：tiling key重构+量化路径逻辑调整
- 可审查性：低
- 审查规则建议：此类大规模修改应拆分为独立的重构PR和bug fix PR

### b026d20e fix kernel src copy
- 根因类别：构建脚本文件拷贝逻辑缺陷 — 未递归处理子目录
- 涉及文件：cmake/custom_build.cmake, cmake/func.cmake
- 缺陷描述：kernel源码安装只拷贝op_kernel/和op_kernel/${ARCH}/两层，未递归处理更深子目录(如arch35/vf/)
- 修复模式：引入filter_copy_files函数通过install(DIRECTORY)递归拷贝
- 可审查性：中
- 审查规则建议：文件安装/拷贝脚本应有目录结构验证确保新增子目录不被遗漏

### 70349172 fix attention_update cmake
- 根因类别：CMake算子注册位置错误
- 涉及文件：attention/attention_update/op_host/CMakeLists.txt
- 缺陷描述：add_op_to_compiled_list()放在条件块内部，在线编译场景下不执行导致算子未注册到编译列表
- 修复模式：将注册语句移到文件顶部无条件执行
- 可审查性：高
- 审查规则建议：add_op_to_compiled_list()应始终在CMakeLists.txt顶部无条件调用

### 20f4d541 Revert "mc2 tilingkey rectification (MatmulReduceScatter V2)"
- 根因类别：功能回退/Revert — tilingkey整改系列
- 涉及文件：mc2/matmul_reduce_scatter_v2/下tiling和kernel文件
- 缺陷描述：ReduceScatterV2的tilingkey从模板参数方案回退到手动编码数值方案，原整改引入功能异常
- 修复模式：Revert — 删除tiling_key.h，还原tiling计算和kernel分发逻辑
- 可审查性：中
- 审查规则建议：tilingkey方案切换需完整端到端验证

### 8ead68cc Revert "mc2 tilingkey rectification for A2/A3 (combine,dispatch)"
- 根因类别：功能回退/Revert — tilingkey整改系列
- 涉及文件：moe_distribute_combine和dispatch的v1/v2版本12个文件
- 缺陷描述：combine/dispatch的A2/A3平台tilingkey从模板参数方案回退，同351/355/382同源
- 修复模式：Revert — 批量删除tiling_key.h还原逻辑
- 可审查性：低
- 审查规则建议：大规模重构需分模块逐步合入验证

### c0dd2d8f 【MlaProlog】fix infershape
- 根因类别：空指针解引用+条件判断错误
- 涉及文件：attention/mla_prolog_v3/op_host/mla_prolog_v3_infershape.cpp
- 缺陷描述：(1)GetAttrPointer返回值可能为nullptr直接解引用；(2)dequantScaleQNormShape推导用dtype推断应用weightQuantMode属性判断
- 修复模式：增加空指针检查+修正条件判断依据
- 可审查性：高
- 审查规则建议：所有GetAttrPointer返回值必须判空；infershape逻辑应基于属性参数而非推断dtype

### 07b5d3af [fix] fix presionFail acs2=0
- 根因类别：边界条件缺失 — actS2Size=0时除零/非法计算
- 涉及文件：attention/lightning_indexer/op_kernel/lightning_indexer_kernel.h, quant_lightning_indexer_kernel.h
- 缺陷描述：GetS2BaseBlockNumOnMask函数在actS2Size=0时缺少提前返回，后续计算产生负数导致精度问题
- 修复模式：增加actS2Size==0时返回0的guard clause
- 可审查性：高
- 审查规则建议：涉及长度/大小参数的函数入口需检查零值边界

### 7e400e34 fix lse out
- 根因类别：控制流错误 — TND/NTD场景提前return跳过必要初始化
- 涉及文件：attention/common/op_kernel/arch32/fia_kernel_nonquant.h, fia_kernel_nonquant_mla.h
- 缺陷描述：InitOutputSingleCore中TND/NTD场景在函数入口return跳过了后续softmaxLse初始化
- 修复模式：将跳过范围从函数级return缩小到块级if constexpr
- 可审查性：中
- 审查规则建议：提前return需审查是否跳过后续必要的初始化/同步操作

### 1064a387 fix bug set comm engine
- 根因类别：调用时序错误 — SetCommEngine在GetTiling之后
- 涉及文件：mc2/quant_all_reduce/op_host/op_tiling/quant_all_reduce_tiling.cpp
- 缺陷描述：SetCommEngine(AIV_TYPE)放在GetTiling之后导致comm engine配置未生效
- 修复模式：将SetCommEngine调用移到GetTiling之前
- 可审查性：高
- 审查规则建议：配置类API(Set*)必须在获取结果API(Get*)之前调用

### 73a82c36 combinev2用例精度失败修复
- 根因类别：错误的对齐计算
- 涉及文件：mc2/moe_distribute_combine/op_kernel/arch35/moe_distribute_combine_arch35.h
- 缺陷描述：perTokenSize_的计算使用Ceil做512字节对齐但实际不需要，对齐后偏大导致数据搬移地址错位
- 修复模式：移除不必要的对齐计算使用实际大小
- 可审查性：高
- 审查规则建议：数据对齐需确认是否真正需要(通信对齐vs计算对齐)

### b9259a86 修改pta版本过低导致的拦截问题
- 根因类别：兼容性缺陷 — 旧版PTA传入shape=0的tensor而非nullptr
- 涉及文件：attention/fused_infer_attention_score/op_host/arch32/下tiling_check_existence.cpp和tiling_info_parser.cpp
- 缺陷描述：torch_npu 2.1中不传actualSharedPrefixLen时传入shape=0的tensor而非nullptr，原代码仅检查!=nullptr导致误识别
- 修复模式：将actualSharedPrefixLen从强制存在性校验中移除，改为shape非零时才校验
- 可审查性：中
- 审查规则建议：可选参数存在性判断不能仅依赖nullptr，需同时检查shape有效性

### db554b11 fix performance info
- 根因类别：变量未初始化 — UB内存复用后脏值
- 涉及文件：mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch_a2.h
- 缺陷描述：performanceInfoI32Tensor_在AllocTensor时初始化，但UB内存复用后实际使用时值已被覆盖为脏值
- 修复模式：将初始化移到实际使用前(WaitDispatch中)+添加SyncFunc
- 可审查性：中
- 审查规则建议：UB内存复用场景下tensor初始化时机需在最后一次覆盖之后

### 2001261f Revert "MatmulAllreduce tilingKey整改【A2+310P】"
- 根因类别：功能回退/Revert — tilingkey整改系列
- 涉及文件：matmul_all_reduce的各平台23个文件(+7467/-2250行)
- 缺陷描述：MatmulAllreduce A2+310P tilingkey从模板方案回退到手动编码，与351/355/382/383同系列
- 修复模式：Revert — 大规模代码回退
- 可审查性：低
- 审查规则建议：超大规模重构(近万行变更)需有分批验证策略

### b4f258a5 修改NBSD和TND_NTD场景下softmaxlse输出
- 根因类别：逻辑错误 — 输出布局条件判断用运行时变量应为编译时模板参数
- 涉及文件：attention/common/op_kernel/arch32/fia_block_vec_nonquant_mla.h, memory_copy.h
- 缺陷描述：CopySoftmaxLseToGmByLayout用constInfo.outputLayout而非模板参数LAYOUT_T做条件判断，NBSD/NTD专用lse拷贝函数地址计算有误
- 修复模式：统一layout分支使用编译时模板参数
- 可审查性：中
- 审查规则建议：layout条件判断优先使用编译时模板参数而非运行时变量

### 2ecee98a add sync after LocalCompute to solve accuracy issue
- 根因类别：并发同步缺失 — 计算完成后缺少核间同步
- 涉及文件：mc2/all_gather_matmul_v2/op_kernel/all_gather_quant_bmm.h
- 缺陷描述：QuantMatmulLocalCompute完成后缺少Mc2SyncAll，部分核还在写入时其他核已开始通信导致数据竞争
- 修复模式：在计算阶段结束后添加核间同步屏障
- 可审查性：高
- 审查规则建议：多核异构计算的阶段转换点(compute->communication)必须有显式同步

### cd0436d7 fix MoeFusedTopk dtype check
- 根因类别：条件检查缺失 — 未根据feature开关做校验
- 涉及文件：moe/moe_fused_topk/op_host/op_api/aclnn_moe_fused_topk.cpp
- 缺陷描述：enableExpertMapping=false时mappingNum/mappingTable可能为nullptr，不应做dtype校验
- 修复模式：将校验包在if(enableExpertMapping)条件内
- 可审查性：高
- 审查规则建议：可选功能的关联参数校验必须在功能开关条件内执行

### b6bdcb97 bugfix in splitFuse D le 256
- 根因类别：逻辑运算符错误 — ||应为&&
- 涉及文件：attention/fused_infer_attention_score/op_host/下2个tiling文件
- 缺陷描述：isFAIDSize条件中(D<=256) || (D相等)应为&&，||导致维度不相等但都小于256时误判为true进入错误路径
- 修复模式：||改为&&(4处)
- 可审查性：高
- 审查规则建议：复合布尔条件的||与&&选择需审查语义；建议对复杂条件拆分为命名变量

### 46ee82ed 修复字节对齐
- 根因类别：数据结构对齐缺陷
- 涉及文件：attention/common/op_kernel/arch35/flash_attention_score_tiling_regbase.h
- 缺陷描述：splitCoreMode(uint8_t)后无padding导致结构体字节对齐不正确，影响host/device间tiling数据传递
- 修复模式：添加reserve[3]补齐4字节对齐
- 可审查性：高
- 审查规则建议：host/device共享的tiling结构体必须确保字节对齐一致性

### 71d9b81b 修复infershape中没有对dimnum校验
- 根因类别：输入校验缺失 — 缺少维度数和维度一致性校验
- 涉及文件：gmm/quant_grouped_matmul_inplace_add/op_host/下checker和infershape文件
- 缺陷描述：checker缺少y tensor的dimNum==3校验和M/N维度一致性校验；infershape假设固定3维硬编码取dim
- 修复模式：增加dimNum和维度一致性校验+简化infershape
- 可审查性：高
- 审查规则建议：所有tensor的dimNum需做显式校验，不能假设固定维度

### fb6a84b5 [MC2] fix the round mode of output cast
- 根因类别：参数错误 — 错误的舍入模式
- 涉及文件：mc2/quant_reduce_scatter/op_kernel/quant_reduce_scatter_mte.h
- 缺陷描述：output cast用CAST_ROUND(四舍五入)但量化场景应为CAST_RINT(银行家舍入)，CAST_ROUND在.5处产生系统性偏差
- 修复模式：更换RoundMode枚举值
- 可审查性：高
- 审查规则建议：量化相关Cast操作应统一使用CAST_RINT

### bfbca7e1 fix修复mmallreduce-310P
- 根因类别：变量名拼写错误 — M_TYPE应为MM_TYPE
- 涉及文件：mc2/matmul_all_reduce/op_kernel/matmul_all_reduce.cpp
- 缺陷描述：条件分支中M_TYPE应为MM_TYPE(MatMul类型)，M_TYPE指向错误的模板参数导致310P平台条件判断错误
- 修复模式：修正变量名
- 可审查性：高
- 审查规则建议：模板参数命名应避免单字母前缀歧义；启用更严格的shadow告警

### 3ac05af5 [MlaProlog] fix tiling
- 根因类别：输出shape维度错误
- 涉及文件：attention/mla_prolog/op_host/mla_prolog_tiling_check.cpp
- 缺陷描述：dequantScaleQNorm输出期望shape从1维{tSize}修正为2维{tSize,1}，与384号(V3版本)同类问题
- 修复模式：修正期望shape维度
- 可审查性：高
- 审查规则建议：同一算子不同版本需同步修复同类问题；tiling check期望shape需与infershape和kernel一致

### 9029cbf6 fix bugs rowInvalid
- 根因类别：off-by-one错误
- 涉及文件：attention/common/op_kernel/vector_common.h
- 缺陷描述：DealInvalidRowsBelow中循环终止条件s1RealEnd > 0应为>= 0，导致s1RealEnd==0时最后一行无效行未被处理
- 修复模式：> 0改为>= 0
- 可审查性：高
- 审查规则建议：循环终止条件>0需审查是否应为>=0，特别是索引遍历到第0个元素的场景

### a929f9dc IFA CV1:1 all quant bugfix
- 根因类别：初始化时序/平台适配逻辑错误
- 涉及文件：attention/common/op_host/arch32/fia_tiling_nonquant.cpp/h, incre_flash_attention_tiling.cpp/h
- 缺陷描述：(1)cvRatio_未初始化为成员变量，每次重复计算aivNum_/aicNum_有除零风险；(2)310P平台上cvRatio_在赋值前可能被使用
- 修复模式：提取为成员变量并修正初始化时序
- 可审查性：中
- 审查规则建议：成员变量在所有平台分支中都需有正确初始化路径；涉及除法的重复计算应缓存

### 97f88dbd allgatherv2 mxfp异常拦截修复
- 根因类别：输入校验逻辑错误/分支遗漏
- 涉及文件：mc2/all_gather_matmul_v2/op_host/op_tiling/all_gather_quant_bmm_tiling.cpp等
- 缺陷描述：(1)pertensor检查中错误混入mxfp逻辑；(2)SetQuantScene缺少scale维度校验；(3)pertensor模式未区分float与float8_e8m0；(4)proto缺少DT_FLOAT_E8M0
- 修复模式：重构校验逻辑按数据类型分路径+增加维度一致性校验
- 可审查性：中
- 审查规则建议：多模式(pertensor/perblock/mxfp)校验函数需正确覆盖所有分支；proto类型列表需与实际一致

### 6840b3b8 fix mlaprolog conflict
- 根因类别：模板参数缺失 — 合并冲突遗漏
- 涉及文件：attention/mla_prolog/op_kernel/mla_prolog_comm.h, mla_prolog_template_tiling_key.h, mla_prolog_v3.cpp
- 缺陷描述：MLAPType的FP8特化缺少CV_RATIO模板参数，FP8实例化未传cvRatio，tiling key缺少CV_MODE
- 修复模式：补全模板参数和实例化参数
- 可审查性：中
- 审查规则建议：新增模板参数时审查所有特化版本和实例化点是否同步更新

### ce09ff7c 修复gmm若干问题
- 根因类别：多类缺陷集合 — 校验缺失+逻辑表达式错误+空指针风险
- 涉及文件：gmm/grouped_matmul/下infershape/checker/aclnn/tiling 5个文件
- 缺陷描述：(1)shape索引计算未检查unknown shape/维度数；(2)weight->Size()>=1前置校验缺失；(3)||应为&&的逻辑表达式(isNoActivation||yDtype!=DT_INT8||yDtype!=DT_INT32恒true)；(4)CheckZeroShape语义倒置；(5)perTokenScaleOptional未判空；(6)tiling tensor维度未校验
- 修复模式：增加defensive校验+修正逻辑表达式+空指针保护
- 可审查性：低
- 审查规则建议：a!=X||a!=Y恒true是典型模式；解引用可选tensor前必须判空判size

### c88c54f2 修正GMM Tiling对mxfp4拦截的错误
- 根因类别：条件判断取反 — !=应为==
- 涉及文件：gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp
- 缺陷描述：OP_CHECK_IF(kSize != 2)应为== 2，!=导致k=2未拦截而所有k!=2的合法输入被错误拦截
- 修复模式：!=改为==
- 可审查性：高
- 审查规则建议：OP_CHECK_IF的条件是失败条件(true时报错)，审查条件语义与错误消息是否一致

### 707ab21c fix GMMFR: tiling and kernel about scatter_add
- 根因类别：废弃字段导致逻辑分支错误+空指针校验缺失
- 涉及文件：grouped_matmul_finalize_routing相关8个文件
- 缺陷描述：(1)tiling->scatterAdd字段控制combine模式分支但应无条件执行scatter_add逻辑；(2)logit输入为空时未校验；(3)printf格式%ld应为%lld
- 修复模式：移除废弃tiling字段用if constexpr替代+增加空指针校验
- 可审查性：中
- 审查规则建议：tiling字段增删时检查kernel侧所有引用；int64_t用%lld或PRId64

### a7c6d232 修复perfix和mask的拦截问题
- 根因类别：参数使用错误+初始化逻辑错误
- 涉及文件：fused_infer_attention_score_tiling_check_feature.cpp, tiling_info_parser.cpp
- 缺陷描述：(1)CheckFeatureGqaPrefix中totalLen使用s2Size应为maxActualseq；(2)maxActualseq_被无条件初始化为s2Size_但当actualSeqLengths存在时应从中计算
- 修复模式：修正变量引用+修正初始化逻辑
- 可审查性：高
- 审查规则建议：s2Size和maxActualseq含义相近时审查使用场景；"先初始化默认值后覆盖"模式检查默认值是否干扰覆盖逻辑

### 69df532e fix bugs softmax
- 根因类别：数值精度/极值处理错误
- 涉及文件：attention/common/op_kernel/arch32/fia_block_vec_nonquant.h, fia_block_vec_nonquant_mla.h
- 缺陷描述：SOFTMAX_MIN_NUM原值-2e38不是真正负无穷(float32最小约-3.4e38)，极端case下mask位置仍有微小概率
- 修复模式：替换为T(-1.0/0.0)即-inf
- 可审查性：高
- 审查规则建议：softmax/attention mask中极小值应使用-inf而非手写大负数(-1e9/-2e38)

### 235444dc MlaPrologV3解决拦截问题
- 根因类别：条件判断逻辑缺陷+参数校验遗漏
- 涉及文件：attention/mla_prolog/op_host/mla_prolog_tiling.cpp/h, mla_prolog_tiling_check.cpp/h
- 缺陷描述：(1)CheckPANZPerTile中&&应为||导致不合法组合不拦截；(2)else-if链条件重叠PER_CHANNEL分支进不去；(3)cacheMode未判空；(4)V3 GetQuantizationMode缺NO_QUANT分支
- 修复模式：重构参数校验逻辑+提取独立函数+增加ERROR_MODE枚举+空指针校验
- 可审查性：中
- 审查规则建议：多条件&&/||组合需逐条验证真值表；else-if链各分支条件不应重叠

### 1ae5a4f0 修复PFA GQA GS1合轴不叠加高阶特性
- 根因类别：条件判断遗漏 — 新增特性flag未纳入互斥检查
- 涉及文件：attention/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：CheckPFAMerge遗漏enablePFARope/enablePerblockQuant/enablePertensorQuant三个flag，这些特性开启时仍允许合轴导致计算错误
- 修复模式：在||链中补充三个遗漏的特性flag
- 可审查性：高
- 审查规则建议：新增特性flag时需全局搜索所有互斥/兼容性检查点

### 9124fa1f grouped_mamtul_swiglu_quant_v2 bugfix
- 根因类别：维度索引错误+矩阵格式错误+tiling参数遗漏
- 涉及文件：gmm/grouped_matmul_swiglu_quant_v2/op_host/op_tiling/grouped_matmul_swiglu_quant_base_tiling.cpp
- 缺陷描述：(1)quantGroupNum_取wScaleDimNum-1应为-2(KGroupCount在倒数第二维)；(2)SetBType的CubeFormat为ND应为NZ；(3)tiling缺少baseM/baseN字段
- 修复模式：修正维度索引+矩阵格式+补充tiling字段
- 可审查性：高
- 审查规则建议：tensor shape维度索引与注释中shape语义严格对应；ND/NZ格式需与数据布局匹配

### d16de174 修复末端dynamicScalesTensor越界导致MPU error
- 根因类别：内存地址计算错误 — buffer越界
- 涉及文件：mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch_a2.h
- 缺陷描述：dynamicScalesTensor从UB末端反向分配，使用编译时估计的expertIdsCnt_为元素数，实际token数超预期时越界触发MPU error
- 修复模式：改为紧随xOutTensor之后顺序分配+用运行时实际tokenNum
- 可审查性：中
- 审查规则建议：UB内存应顺序递增分配避免反向重叠；buffer大小应使用运行时实际值

### 9e08a82c gmm 310p bugfix
- 根因类别：分支匹配错误 — TRANS_A/TRANS_B实现体互换
- 涉及文件：gmm/grouped_matmul/op_kernel/grouped_matmul.cpp
- 缺陷描述：if constexpr模板分支中TRANS_A==0&&TRANS_B==1分支执行了TRANS_A==1的逻辑，两个分支实现体互换
- 修复模式：将分支体归还给正确的条件
- 可审查性：高
- 审查规则建议：constexpr分支条件与体内transpose参数应严格对应

### 707335ee fix indices size of moetokenpermute
- 根因类别：UB内存大小计算错误 — 缺少类型size乘数
- 涉及文件：moe/moe_token_permute/下tiling和kernel共5个文件
- 缺陷描述：indicesUB=UpAlign(onceIndices,ONE_BLOCK_BYTE)中onceIndices是元素个数但对齐按字节，缺INT32_DTYPE_SIZE乘数导致只分配1/4空间，kernel侧用*4补偿hack
- 修复模式：tiling侧补*INT32_DTYPE_SIZE+kernel侧移除*4
- 可审查性：高
- 审查规则建议：UpAlign/对齐参数单位必须为字节，元素个数须乘DTYPE_SIZE；host和kernel侧用同一公式

### bac6b116 修复看板报错
- 根因类别：空指针校验缺失+平台参数校验遗漏
- 涉及文件：gmm/quant_grouped_matmul_inplace_add/下checker和tiling文件
- 缺陷描述：(1)gmmParams_.y/weight解引用前未判空；(2)aclTensorList类型缺GetInputTensor重载；(3)l2Size==0未校验
- 修复模式：增加空指针CHECK_COND+新增函数重载+补充l2Size校验
- 可审查性：高
- 审查规则建议：指针解引用前必须判空；函数重载需覆盖所有参数类型

### 3fe51049 fix bugs
- 根因类别：常量名拼写错误遗漏 — 重命名不完整
- 涉及文件：gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul_910_95_checker.cpp
- 缺陷描述：commit 441将LAST_TOW_DIM_INDEX改为LAST_TWO_DIM_INDEX但遗漏一处引用
- 修复模式：修正遗漏的常量名引用
- 可审查性：高
- 审查规则建议：重命名常量/变量时需全局搜索确保替换完整

### 351e7689 antiquant pa flashdecode fix
- 根因类别：功能遗漏 — PA antiquant场景未适配
- 涉及文件：attention/common/op_kernel/arch35/下2个antiquant文件, incre_flash_attention_tiling_v2.cpp, incre_flash_attention_entry_regbase.h
- 缺陷描述：(1)PA antiquant场景sInnerLoopSize未做32对齐导致搬运地址不对齐；(2)PER_TOKEN_GROUP_MODE缺数据类型校验；(3)PA antiquant+flashdecoding BF16 int8缺kernel入口注册
- 修复模式：添加AlignUp32+补充类型校验+新增tiling key的kernel入口
- 可审查性：中
- 审查规则建议：新增模式时检查所有内存对齐相关计算；每种数据类型组合的kernel入口需与tiling key完整对应

### 7ab5613c bugfix: slig BSND BS=1 tiling BSIndex
- 根因类别：循环退出条件位置错误
- 涉及文件：sparse_lightning_indexer_grad_kl_loss_tiling_general.cpp
- 缺陷描述：Balance4DLoad中break检查放在循环末尾而非开头，BS=1时循环体在应退出后仍继续执行导致数组越界
- 修复模式：将提前退出条件从循环末尾移到循环体开头
- 可审查性：高
- 审查规则建议：循环中break/continue应放在循环体最早可判断处

### 120106d9 nsa_compress_attention_infer fix mem.h InitBuffer
- 根因类别：复制粘贴错误 — 数组索引全部写成同一值
- 涉及文件：common/include/kernel/mem.h
- 缺陷描述：InitBuffer中L0A/L0B/L0C三个buffer初始化的数组索引全部写成ASCEND_CB，导致只有CB被赋值，L0A/L0B/L0C永远未初始化
- 修复模式：修正左值索引为ASCEND_L0A/L0B/L0C
- 可审查性：高
- 审查规则建议：同一数组连续赋值时检查索引是否有重复(dead store模式)

### 010d8080 [FAG] fix deter BN2 & remove deter template
- 根因类别：逻辑条件错误+提前return跳过后续逻辑
- 涉及文件：aclnn_flash_attention_score_grad_vx.cpp, flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp等
- 缺陷描述：(1)BN2路径错误排除deterministic模式；(2)PostFlashAttentionScoreGrad中dpseOut的return导致dqRopeOut的ViewCopy被跳过
- 修复模式：删除错误排除条件+修复return为if-else
- 可审查性：中
- 审查规则建议：提前return后还有必要逻辑时应重构为条件分支

### 312d8a13 LIG bugfix
- 根因类别：硬编码维度索引+缺少零值保护
- 涉及文件：lightning_indexer_grad_tiling.cpp, lightning_indexer_grad_kernel.h
- 缺陷描述：(1)topK用硬编码DIM_IDX_THREE/TWO应为dyShapeDim-1；(2)loopTimes为0时循环仍执行非法访问
- 修复模式：动态获取维度索引+增加零值检查
- 可审查性：高
- 审查规则建议：避免硬编码维度索引，用GetDimNum()-1等动态方式

### ad3daa36 [bugfix] mmar图模式下空tensor场景没拦截
- 根因类别：输入校验缺失 — output空tensor未拦截
- 涉及文件：aclnn_quant_matmul_all_reduce_v4.cpp
- 缺陷描述：CheckShape对output缺少空tensor判断，图模式下传入空tensor导致异常
- 修复模式：增加output->IsEmpty()拦截
- 可审查性：高
- 审查规则建议：所有算子输入/输出校验应包含空tensor拦截

### a263ed64 bugfix:解决GMMDSQV2 N过大精度问题挂死问题
- 根因类别：多处计算逻辑错误 — 参数错配/同步死锁/循环索引固定
- 涉及文件：aclnn_gmm_dsq_base.h, grouped_matmul_swiglu_quant_fusion_tiling.cpp/h, grouped_matmul_swiglu_quant_spilit_fusion.h
- 缺陷描述：(1)BASE_K/BASE_N值交换；(2)vectorBlockDim错用aicCoreNum_；(3)ubFactorDimx固定4应动态调整；(4)同步死锁；(5)循环中始终用[0]而非[i]；(6)NZ格式缺对齐校验
- 修复模式：修正参数+动态配置+重写同步逻辑+增加对齐校验
- 可审查性：低
- 审查规则建议：tiling维度常量应有注释说明对应关系；循环中使用固定索引[0]属明显bug

### 2474918c fix TND layout softmaxlse
- 根因类别：缺少零值保护
- 涉及文件：fia_block_vec_nonquant_mla.h
- 缺陷描述：TND layout下vecDealM==0时仍执行向量操作导致异常
- 修复模式：增加continue跳过零长度计算
- 可审查性：高
- 审查规则建议：向量/矩阵计算前应检查数据长度是否为0

### 5834cb6c 修改Dispatch Combine A5 Tiling中Hccl_Buffer计算公式 & 流水同步
- 根因类别：通信buffer计算公式错误+流水同步不足
- 涉及文件：moe_distribute_combine_tiling_arch35.cpp, moe_distribute_combine_arch35.h, moe_distribute_dispatch_tiling_arch35.cpp
- 缺陷描述：(1)HCCL buffer未考虑512字节通信对齐(COMM_ALIGN曾为2应为512)；(2)缺少aivNum*STATUS_SIZE等部分；(3)SharedAndMoeRank用V_S同步不够应PipeBarrier<PIPE_ALL>
- 修复模式：修正buffer计算公式+修复同步原语
- 可审查性：中
- 审查规则建议：通信buffer应有公式推导文档；COMM_ALIGN应统一定义

### 63693c12 qsfa/li异常拦截_qsfa精度问题修复同步
- 根因类别：API误用+空指针未保护+dtype列表错误
- 涉及文件：kv_quant_sparse_flash_attention_tiling.cpp/h/kernel_mla.h, lightning_indexer_infershape.cpp/tiling.cpp/h
- 缺陷描述：(1)GetInputDataType错误传ge::DT_INT32(应为输入index)；(2)TND路径blockTable为null时无条件访问shape；(3)dtype列表冗余
- 修复模式：修正API调用+增加空指针保护+重构校验
- 可审查性：中
- 审查规则建议：API调用参数语义需确认(参数是index不是default值)

### 1ed1e571 fix QuantMatmulAllReduceV4 per_block场景x1三维精度问题
- 根因类别：三维输入shape处理错误 — 降维丢失batch信息
- 涉及文件：matmul_formulaic_tiling.h, quant_matmul_all_reduce_tiling_910_95.cpp, matmul_all_reduce_tiling_base.cpp, matmul_all_reduce_base.h
- 缺陷描述：per_block量化x1三维时，tiling将3维shape错误压扁为2维，M轴非128对齐时padding/tiling计算错误
- 修复模式：增加batchValue字段+条件保留原始shape+kernel用rankM计算offset
- 可审查性：中
- 审查规则建议：高维tensor降维时需验证对齐假设；tiling shape变换应有维度语义文档

### 7267eec6 tiling中返回值错误修改
- 根因类别：返回值类型错误 — return false应为return GRAPH_FAILED
- 涉及文件：prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：CheckSingleAttribute返回ge::graphStatus但CheckIFAMLA失败时return false(隐式转0=GRAPH_SUCCESS)，导致校验失败被当成功
- 修复模式：改为return ge::GRAPH_FAILED
- 可审查性：高
- 审查规则建议：返回枚举/状态码类型时禁止用bool字面量return

### 504e1b13 gmm/gmm_add fix
- 根因类别：参数校验缺失+dtype校验错误
- 涉及文件：aclnn_grouped_matmul.cpp, grouped_no_quant_matmul_tiling.cpp, grouped_matmul_add_no_quant_tiling.cpp
- 缺陷描述：(1)非量化路径缺对量化参数必须为nullptr的校验；(2)A5 bias dtype校验过于宽泛应限为xDtype或float32
- 修复模式：增加参数空值校验+修正dtype校验条件
- 可审查性：中
- 审查规则建议：非量化路径应显式校验量化参数为空

### c1fb131e 交叉属性校验问题单修改
- 根因类别：校验顺序错误 — 影响shape解析的校验放在解析之后
- 涉及文件：prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：PA/tensorList拦截检查放在交叉校验阶段但shape解析已受影响，应前移到单参数校验阶段
- 修复模式：将校验前移+新增LSE不支持per-block quant拦截
- 可审查性：高
- 审查规则建议：影响shape解析的参数校验必须在shape解析之前

### 6098e6f8 修复AI检视编码问题
- 根因类别：复制粘贴错误+逻辑运算符错误+变量名拼写
- 涉及文件：grouped_matmul_infershape_quant_checker.cpp, grouped_quant_matmul_tiling.cpp
- 缺陷描述：(1)CheckFormatValid用xFormat应为weightFormat；(2)!=1||!=n恒true应为||(...&&...)；(3)wScaleKDim从xScaleShape取应为wScaleShape；(4)leftL1sie拼写错误
- 修复模式：修正变量引用+修正逻辑运算符+修正拼写
- 可审查性：高
- 审查规则建议：复制粘贴后逐行检查变量名替换；!=A||!=B恒true的lint检测

### 4386f704 fix RopeWithSinCosCache
- 根因类别：UB空间估算公式错误
- 涉及文件：posembedding/rope_with_sin_cos_cache/op_host/rope_with_sin_cos_cache_tiling.cpp
- 缺陷描述：headSize需保存原始数据和输出两份但系数设为1；sincos缓存是固定开销但被线性累加导致maxNPerLoopForUb偏大，UB溢出
- 修复模式：修正数学公式(headSize系数1->2，固定开销分离)
- 可审查性：中
- 审查规则建议：tiling中UB/L1资源计算公式应附推导注释；空间估算变更需边界case验证

### 302dc12f 修复mmreducescatterv2 mxfp8用例精度问题
- 根因类别：多核tiling逻辑缺失 — 尾块核preCoreNum_未重算
- 涉及文件：mc2/matmul_reduce_scatter_v2/op_kernel/quant_bmm_reduce_scatter_fp8_hif8.h
- 缺陷描述：尾块场景下preCoreNum_未根据headSliceM重新计算，偏移量错误导致MatMul输出写入错误位置
- 修复模式：新增条件分支在尾块场景重新计算mCnt/nCnt和preCoreNum_
- 可审查性：低
- 审查规则建议：MC2多核分片逻辑必须覆盖主块核和尾块核两种场景

### 2cb9d620 同步遗漏的matmul_reduce_scatter_v2修改
- 根因类别：校验逻辑位置错误+隐式推断替代显式属性
- 涉及文件：mc2/matmul_reduce_scatter_v2/op_api/aclnn_matmul_reduce_scatter_v2.cpp, tiling
- 缺陷描述：(1)amax校验放在高精度分支内应在公共路径；(2)用维度比较推断transpose应读显式isTransposeB属性
- 修复模式：将amax校验上提到公共路径+用显式属性替代隐式推断
- 可审查性：中
- 审查规则建议：校验逻辑应放在最早的公共路径；用显式属性而非隐式推断

### ae617083 FIA算子mask_copy数值溢出问题修复
- 根因类别：signed/unsigned类型转换溢出
- 涉及文件：attention/common/op_kernel/vector_common.h
- 缺陷描述：IsSkipAttentionmask中将有符号s1StartIdx转uint64_t，负值变极大无符号数导致比较永远true
- 修复模式：将static_cast<uint64_t>改为static_cast<int64_t>
- 可审查性：高
- 审查规则建议：有符号与无符号混合比较告警必须处理；涉及signed/unsigned cast需确认语义

### 8204110e 字节通信量限制修正
- 根因类别：校验逻辑缺失 — 通信量下限未校验
- 涉及文件：mc2/allto_allv_grouped_mat_mul/和grouped_mat_mul_allto_allv/下tiling文件
- 缺陷描述：CheckSendRecvDataVolumn函数体为空壳，910_93平台通信量低于2MB时触发硬件异常
- 修复模式：新增per-rank通信量累加和下限校验
- 可审查性：中
- 审查规则建议：通信算子tiling校验函数不应为空壳

### f8d3aafb fix bugs
- 根因类别：语法缺陷 — 宏调用缺少分号
- 涉及文件：gmm/grouped_matmul_finalize_routing/op_host/op_api/aclnn_grouped_matmul_finalize_routing.cpp
- 缺陷描述：OP_CHECK_DTYPE_NOT_SUPPORT末尾缺分号，可能导致后续if被错误关联为else分支
- 修复模式：补充缺失的分号
- 可审查性：高
- 审查规则建议：宏调用末尾必须有分号；开启-Werror编译选项

### 64e6ae88 GMM伪量化oom越界修复
- 根因类别：非对齐内存越界读取
- 涉及文件：gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/weight_quant_cube_compute.h
- 缺陷描述：DataCopy搬运bias时CeilAlign对齐到32B，bias实际大小不对齐时越界读取GM导致OOM
- 修复模式：替换为DataCopyPad2D自动处理非对齐padding
- 可审查性：高
- 审查规则建议：DataCopy搬运非对齐数据时必须检查越界风险；建议统一用DataCopyPad系列

### 2e66d2fd infershape-bug修复
- 根因类别：InferShape fallback逻辑错误
- 涉及文件：attention/fused_infer_attention_score/op_host/fused_infer_attention_score_infershape.cpp
- 缺陷描述：BSH layout下fallback值h1=d1*numHeads与主路径outH计算同源，当outH为0时fallback也为0，应直接用queryShape[2]
- 修复模式：将fallback从派生计算值改为原始shape值
- 可审查性：中
- 审查规则建议：InferShape fallback分支需使用独立信息源

### 4e713c47 SFAG修复sparsemode=3bug
- 根因类别：掩码计算公式错误 — 边界off-by-one
- 涉及文件：attention/sparse_flash_attention_grad/basic_modules/vec_op.h
- 缺陷描述：对角线block时attenMskEnd应等于selectedBlockSize，但原公式比正确值少1导致mask末尾少覆盖一个元素
- 修复模式：直接赋值selectedBlockSize替代相对偏移公式
- 可审查性：中
- 审查规则建议：attention mask起止索引计算应添加边界case的UT验证

### 9a10fc7d build_in bug
- 根因类别：标准库依赖不兼容+构建配置遗漏
- 涉及文件：CMakeLists.txt, attention/common/op_kernel/arch32/fia_kernel_empty_tensor.h
- 缺陷描述：kernel中include <math.h>的INFINITY在内置构建模式下不可用；CMakeLists.txt缺少makeself-fetch.cmake
- 修复模式：移除<math.h>，自定义FLOAT_INF=3e+99替代INFINITY；补充cmake include
- 可审查性：中
- 审查规则建议：kernel代码禁止直接include标准C库头文件

### 7a37adbc compile warning fix and bugfix with q&kv_start_idx
- 根因类别：条件限制过严 — q/kvStartIdx读取被特定模式条件限制
- 涉及文件：attention/flash_attention_score/op_host/arch35/下tiling和tiling_regbase文件
- 缺陷描述：SetQKVStartIdx将读取限制在TND&&BAND_LEFT_UP_CAUSAL条件内，其他组合下即使传入tensor也不读取始终为0
- 修复模式：去掉外层条件限制，对所有场景读取(如果tensor非空)
- 可审查性：中
- 审查规则建议：可选输入tensor读取不应被特定模式条件限制

### bfd514e5 valid expert num equal 0 when gathefirst
- 根因类别：零值边界条件未处理
- 涉及文件：moe/moe_init_routing_v3/op_host/tiling.cpp, op_kernel/moe_v3_full_load_base.h等
- 缺陷描述：gatherFirst模式下actual_idx_num_为0时无保护，SortComputeWithRange()继续执行排序异常
- 修复模式：添加零值边界保护+修正判断条件(固定阈值改比例)
- 可审查性：中
- 审查规则建议：元素计数/数量变量需检查为0的边界情况

### 72d5bdb6 fix potential data stampede
- 根因类别：整数除法差一错误
- 涉及文件：attention/fused_infer_attention_score/op_kernel/flash_attention_regular.h
- 缺陷描述：triUp/MAX_KV_STACK_LEN在恰好整除时少算一个stack块，导致数据踩踏
- 修复模式：(triUp+1)/MAX_KV_STACK_LEN修正向下取整偏差
- 可审查性：低
- 审查规则建议：整数除法计算分块数时需验证恰好整除的边界case

### b9ac8234 fix fia fd
- 根因类别：循环变量误用 — 尾块使用非尾块size
- 涉及文件：attention/common/op_kernel/arch-310/flash_attention_score_antiquant_baseapi.h
- 缺陷描述：CopyLseIn使用gSplitSize(非尾块大小)而非gSplitSizeTail，尾块时读取超出实际大小的数据
- 修复模式：变量替换为gSplitSizeTail
- 可审查性：高
- 审查规则建议：循环中有尾块特殊处理时检查所有size/length变量是否使用了尾块修正值

### 3e81b61f 修复GMMIADD t-c量化报错到G-B量化
- 根因类别：校验逻辑分支遗漏 — 复用通用checker不覆盖新场景
- 涉及文件：gmm/quant_grouped_matmul_inplace_add/op_host/op_api/下aclnn和新增checker文件
- 缺陷描述：T-C量化场景复用GMM通用checker，但T-C和G-B量化校验规则不同导致合法输入被错误拦截
- 修复模式：新增专用checker按数据类型分支路由
- 可审查性：中
- 审查规则建议：新增量化场景时检查是否复用通用checker，通用checker是否覆盖新场景约束

### b506517b 回退路由
- 根因类别：错误的短路逻辑跳过必要约束检查
- 涉及文件：attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_v3.cpp
- 缺陷描述：isNotLegacyGQA()前置判断使非旧GQA场景跳过CheckGqaConstrain/CheckMlaConstrain约束检查
- 修复模式：删除错误的路由跳过逻辑
- 可审查性：中
- 审查规则建议：约束检查函数中"提前返回true"的短路逻辑需仔细验证条件完备性

### 003a52ce 拦截问题单
- 根因类别：输入校验不完整 — NZ格式维度组合未校验
- 涉及文件：attention/fused_infer_attention_score/op_host/下check和parser和tiling_v3文件
- 缺陷描述：value 5维NZ格式时缺少(qkHeadDim,vHeadDim)组合校验，只支持(64,64)/(128,128)/(192,128)但未拦截其他
- 修复模式：增加输入校验+完善错误信息
- 可审查性：中
- 审查规则建议：新增硬件格式支持时需同步更新参数校验允许组合列表

### eba0e669 [FAG] fix TND BN2 dqk
- 根因类别：偏移量计算遗漏乘法因子
- 涉及文件：attention/common/op_kernel/arch-310/flash_attention_score_grad_block_vec.h
- 缺陷描述：TND layout下dqkvGmOffset缺少n2G乘法因子，GM数据按n2G交错排列，偏移量不乘n2G写入错误位置
- 修复模式：补充遗漏的n2G乘法因子
- 可审查性：低
- 审查规则建议：GM偏移量计算应与layout数据排布严格对应；不同layout分支偏移量需逐项验证维度因子

### 80cd5d07 解决不同bs情况下的reduce信息问题
- 根因类别：buffer大小用了错误的上界基准
- 涉及文件：mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch_a2_layered.h
- 缺陷描述：maxBSInUBForInner_用min(计算值,本机axisBS_)做上界，但跨机场景对端bs可能大于本机axisBS_导致越界
- 修复模式：删除错误的上界约束
- 可审查性：中
- 审查规则建议：分布式场景buffer分配不能假设对端维度与本机相同，应基于可能的最大值

### 2d74b192 fix A5 allreduce_quant
- 根因类别：模板参数硬编码错误 — copy-paste导致
- 涉及文件：mc2/matmul_all_reduce/op_kernel/arch35/matmul_all_reduce_quant_pertoken.h等
- 缺陷描述：matmul模板实例化时scale数据类型硬编码为float应使用模板参数scaleType，A5平台scaleType非float时量化计算错误
- 修复模式：将硬编码float替换为模板参数scaleType
- 可审查性：高
- 审查规则建议：模板实例化中禁止用具体类型替代已定义的模板类型参数

### 40539a09 修复PFA q_s 32B不对场景的精度问题
- 根因类别：硬件指令repeat次数溢出
- 涉及文件：attention/prompt_flash_attention/op_kernel/prompt_flash_attention_s1s2_bns1_x310_base.h
- 缺陷描述：Duplicate的repeat参数可能超过硬件限制255，溢出截断只清零部分数据，残留脏数据引发精度问题
- 修复模式：将单次大repeat拆分为循环分批执行每批不超255
- 可审查性：中
- 审查规则建议：向量/张量指令的repeat参数必须检查上限255；建议封装安全repeat wrapper自动分批

### a8843bc4 fix GQA noQuant兼容性问题
- 根因类别：参数存在性校验逻辑不完备
- 涉及文件：attention/fused_infer_attention_score/op_host/下check和check_existence文件
- 缺陷描述：GQA noQuant下fullquant参数被无条件要求不存在，但特定条件下(INT8输入/非INT8输出)实际需要存在
- 修复模式：新增条件分支函数按场景分支校验
- 可审查性：中
- 审查规则建议：参数存在性校验变更时需同步更新兼容性矩阵

### 74868c43 Fix GMM performance issues with new MSD kernel
- 根因类别：host端静态tiling策略不准+冗余软同步
- 涉及文件：gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp, grouped_matmul_autotiling_a8w4.h
- 缺陷描述：(1)host端用静态M/N算single_N但存在M=0空group；(2)total_M从离线shape读应运行时累加；(3)冗余软同步机制
- 修复模式：kernel端动态计算替代host静态估算+删除冗余软同步+空group防护
- 可审查性：低
- 审查规则建议：动态shape场景tiling参数须在kernel端基于运行时数据计算

### 72f47482 support more ranknum and fix aivnum
- 根因类别：多核调度参数硬编码不合理
- 涉及文件：mc2/elastic_receivable_test/op_host/op_tiling/和op_kernel/
- 缺陷描述：AIV_NUM_USED=6但rankNum不能被6整除时余数分配逻辑导致部分rank数据错误
- 修复模式：将AIV_NUM_USED退化为1绕过分核不均(临时方案)
- 可审查性：高
- 审查规则建议：影响并行度的常量不应硬编码应根据实际aivNum和rankNum动态计算

### f507c792 修改标准错误重定向修复编译不报错
- 根因类别：shell重定向丢失stderr — set -e在pipeline中不生效
- 涉及文件：build.sh
- 缺陷描述：2>&1|gawk结构使编译错误通过pipe输出但set -e不终止脚本(pipeline中非最后命令失败不触发)
- 修复模式：增加set -o pipefail+将set -e提到全局+去掉2>&1
- 可审查性：高
- 审查规则建议：构建脚本必须set -o pipefail；禁止在pipeline中用2>&1吞stderr

### f2b635e3 codecheck问题清理(含真实bug)
- 根因类别：提前return导致校验逻辑被跳过
- 涉及文件：moe/moe_finalize_routing_v2/op_host/moe_finalize_routing_v2_tiling_membase.cpp
- 缺陷描述：return ge::GRAPH_SUCCESS放在hidden size校验之前，310p平台非32对齐hidden size绕过校验
- 修复模式：将return移到函数末尾
- 可审查性：高
- 审查规则建议：多段参数校验时确认return不会跳过后续校验；静态分析标记unreachable code

### 096e009b dispatchV2fullmesh问题定位修正
- 根因类别：buffer复用冲突+初始化时序错误+Duplicate范围过大
- 涉及文件：mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_full_mesh.h
- 缺陷描述：(1)outBuf初始化只在AllToAllProcess中执行但shared expert分支也需要；(2)flagCompResult复用statusWaitBuf导致冲突；(3)Duplicate大小cleanUpNum*8应为8
- 修复模式：分离初始化到各分支+独立flagMaskBuf+修正Duplicate大小
- 可审查性：低
- 审查规则建议：buffer复用需标注生命周期；Duplicate大小需与实际使用范围匹配

### 0fde5fe1 bug修复
- 根因类别：命名空间歧义+形状校验取值错误
- 涉及文件：attention/common/op_kernel/arch32/fia_block_cube_nonquant.h等, fused_infer_attention_score_tiling_check_consistency.cpp
- 缺陷描述：(1)PipeBarrier缺AscendC::限定可能解析错误；(2)CheckKVShapeForPageAttention从key shape取blockNum应从fiaInfo_.totalBlockNum取
- 修复模式：添加命名空间前缀+修正blockNum数据源
- 可审查性：中
- 审查规则建议：AI Core API用完整命名空间；形状校验从权威数据源(tiling参数)获取

### 43f21d60 修复GMMFR L0接口未返回+非连续bug
- 根因类别：L0接口返回值未检查+非连续tensor未处理
- 涉及文件：gmm/grouped_matmul_finalize_routing/op_host/op_api/aclnn_grouped_matmul_finalize_routing.cpp
- 缺陷描述：(1)CHECK_RET中ret是上一步残留值而非当前L0接口返回值；(2)1维tensor被传入带转置逻辑的Contiguous函数
- 修复模式：拆分TransposeTensorContiguousProcess和TensorContiguousProcess+修正返回值检查
- 可审查性：中
- 审查规则建议：CHECK_RET宏参数需确认是当前步骤返回值；所有tensor传递前确保连续性

### e4a5a3a4 [FAG] fix fp8 template bug
- 根因类别：成员变量赋值时序错误 — tiling阶段和key生成阶段数据不一致
- 涉及文件：attention/common/op_host/arch-310/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp/h
- 缺陷描述：fp8OpenTscm作为成员变量在GetShapeAttrsInfo赋值，但GetTilingKey时s1/s2可能已被修改导致tiling key不一致选错kernel
- 修复模式：删除成员变量改为GetTilingKey中局部变量就地计算
- 可审查性：中
- 审查规则建议：tiling key构成字段应在GetTilingKey中就地计算确保一致性

### b73faccc 修复ut整改时修改错误的头文件
- 根因类别：条件编译#endif位置错误
- 涉及文件：mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.cpp
- 缺陷描述：UT整改时#endif提前结束了__DAV_C310__范围，A2头文件变成无条件包含可能在非310平台编译错误
- 修复模式：移除多余#endif在正确位置补上
- 可审查性：高
- 审查规则建议：修改条件编译时逐一验证#if/#endif配对；UT整改不应修改产品代码条件编译结构

### 858707bb fix fag deter problem
- 根因类别：条件判断遗漏(feature flag未传播)
- 涉及文件：attention/common/op_host/arch-310/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp
- 缺陷描述：FlashAttentionScoreGrad反向tiling中判断是否走BN2路径时遗漏对isDeterministic的检查，开启确定性计算时仍错误进入BN2分支(该分支会设isDeterministic=false)导致确定性计算失效
- 修复模式：在两处isBn2条件中补加`&& !fBaseParams.isDeterministic`
- 可审查性：高
- 审查规则建议：新增feature flag时需搜索所有与其互斥的分支是否均已加guard

### 0c6b491a mla-check类问题单
- 根因类别：输入校验缺失(序列长度合法性)
- 涉及文件：attention/common/op_host/fia_tiling_info.cpp/.h, fused_infer_attention_score_tiling_check_*.cpp
- 缺陷描述：FIA算子tiling check存在多个校验缺陷：(1)actualSeqLensQ/KV在TND/NTD layout下未校验负数和非递增导致越界；(2)maxWorkspace场景对pseShift/attentionMask做不必要shape校验导致误报；(3)错误日志缺少ropeMode信息
- 修复模式：防御性校验补全+冗余校验移除+日志增强
- 可审查性：高
- 审查规则建议：处理actual_seq_lengths类输入时需校验元素非负性和有序性

### 699c3964 修复fd场景max和sum gm不连续
- 根因类别：内存偏移计算错误(分支路径变量未初始化)
- 涉及文件：attention/common/op_kernel/arch-310/flash_attention_score_kernel_base.h
- 缺陷描述：非splitD场景下workspace上bmm2结果的GM偏移计算公式`aicIdx*3*mm2Offset`写死，未考虑splitD时需额外加vec2Offset。singleCoreOffset在非splitD路径未赋值就被后续代码使用，导致max/sum GM地址不连续
- 修复模式：统一偏移量计算变量，在各分支都正确赋值
- 可审查性：高
- 审查规则建议：workspace偏移计算中所有条件分支都应确保偏移量变量被正确初始化

### f913f281 fix check HcclBufferSize
- 根因类别：buffer校验未区分通信算法
- 涉及文件：moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp
- 缺陷描述：A3平台校验HCCL buffer大小时未区分通信算法。fullmesh_v2应使用480字节对齐(FULL_MESH_DATA_ALIGN)后再512对齐，原逻辑统一512对齐导致校验阈值偏小可能通过不够用的buffer
- 修复模式：根据isSetCommAlg选择不同对齐方式，抽取CheckWinSize函数
- 可审查性：中
- 审查规则建议：buffer校验中如果存在多种通信模式需分别验证每种模式下的size计算公式

### c439c4d7 修复FIA精度问题
- 根因类别：硬件对齐要求遗漏(mmad维度未对齐)
- 涉及文件：fused_infer_attention_score/op_kernel/attn_infra/gemm/block/block_mmad_pv.hpp, block_mmad_qk.hpp
- 缺陷描述：Mmad的m维度参数直接传入实际值mL1Actual/mL0Actual未做BLOCK_SIZE对齐，L0C上矩阵乘计算使用非对齐m维度引起精度问题
- 修复模式：传参前计算mL0Align = CeilDiv(mActual, BLOCK_SIZE) * BLOCK_SIZE
- 可审查性：中
- 审查规则建议：所有tileMmad调用点需确认m/n/k维度参数经过BLOCK_SIZE对齐处理

### f1500a3c alltoallvgmm拦截问题修复 dtype拦截
- 根因类别：输入校验缺失(dtype未拦截)
- 涉及文件：mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/*_tiling.cpp, mc2/grouped_mat_mul_allto_allv/op_host/op_tiling/*_tiling.cpp
- 缺陷描述：AlltoAllvGmmTiling缺少dtype校验逻辑，不支持的数据类型(非fp16/bf16)未被拦截可能导致运行时错误。对比同类算子grouped_mat_mul_allto_allv已有CheckDtype
- 修复模式：新增CheckDType方法校验tensor dtype一致性和合法性
- 可审查性：高
- 审查规则建议：新算子tiling的Init流程应包含dtype校验；对比同类已有算子确认新算子是否遗漏校验项

### f00b21b9 fix sequential bug in gqa msd dd
- 根因类别：流水线同步缺失(pipe barrier遗漏)
- 涉及文件：attention/incre_flash_attention/op_kernel/incre_flash_attention_preload_dd.h
- 缺陷描述：DealAntiqBmm2ResBaseBlock中DataCopy前缺少PipeBarrier<PIPE_V>()同步屏障，WaitFlag后直接DataCopy但前面vector计算写入vec2ResUb可能尚未完成导致读到脏数据
- 修复模式：在DataCopy前插入PipeBarrier<PIPE_V>()
- 可审查性：低
- 审查规则建议：DataCopy调用前应有对应pipe的PipeBarrier或WaitFlag保证源数据就绪；特别关注preload/double-buffer场景

### 6f01cb96 moe_gating_top_k_softmax Bug修复
- 根因类别：语法错误(漏写分号)
- 涉及文件：moe/moe_gating_top_k_softmax/op_kernel/arch35/moe_gating_top_k_softmax_fullload_generalized_regbase.h
- 缺陷描述：kernel代码中`uint32_t offset = i * expertCountAlign_`行末漏写分号导致编译错误
- 修复模式：补充缺失分号
- 可审查性：高
- 审查规则建议：编译器/静态分析可捕获此类语法错误

### 4829c8af fix bug: B*S过大时导致跳写参数数据溢出
- 根因类别：整数溢出(uint32截断)
- 涉及文件：attention/common/op_kernel/memory_copy.h
- 缺陷描述：B*S过大时DataCopyExtParams.dstStride(uint32_t)在计算跳写参数时溢出导致数据拷贝错误。修复将dstStride提升为uint64_t并在超出UINT32_MAX时改用for循环逐块拷贝
- 修复模式：类型提升+溢出分支保护(SafeStrideCopy封装)
- 可审查性：中
- 审查规则建议：涉及stride/offset计算的参数审查数据类型是否能容纳大shape场景；关注uint32_t与uint64_t隐式截断

### 2c4ac84c open GQA nonquant and fix bug in tiling sink
- 根因类别：tiling下沉初始化缺陷+功能开关误关闭
- 涉及文件：attention/common/op_host/arch32/fia_tiling_nonquant*.cpp, fused_infer_attention_score_tiling_v3.cpp
- 缺陷描述：(1)tiling下沉场景blockDim_未初始化导致校验为0时拦截报错(aicore error)；(2)CalcMaxMmResSize中mBaseSize_未赋值；(3)GQA非量化路径的CheckGqaConstrain整体被注释掉功能无法进入
- 修复模式：补充变量初始化+取消注释恢复逻辑
- 可审查性：高
- 审查规则建议：tiling下沉场景需确保所有关键参数有初始值；注释掉的功能代码应有明确标注否则视为遗留bug

### 4eaa119d 修复算子alltoallvgmm A5版本精度问题
- 根因类别：接口参数缺失(通信数据类型)
- 涉及文件：mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/allto_allv_grouped_mat_mul_tiling.cpp
- 缺陷描述：HCCL通信Tiling配置中未传入数据类型参数(reduceType/srcDataType/dstDataType)，A5版本上alltoallv通信使用错误的默认数据类型引发精度问题
- 修复模式：补全Mc2CcTilingConfig构造函数的数据类型参数
- 可审查性：中
- 审查规则建议：HCCL通信tiling配置时审查数据类型参数是否与实际tensor类型一致；构造函数参数不完整需确认默认值是否正确

### 2d078d70 [FAG] fix deter sparse567
- 根因类别：模板特化配置错误(DeterType合并)
- 涉及文件：attention/common/op_kernel/arch-310/flash_attention_score_grad_template_tiling_key.h
- 缺陷描述：FAG模板tiling key中DeterType=0和1合并在同一ARGS_SEL导致deterministic和非deterministic共享不正确的模板参数组合，sparse模式精度错误
- 修复模式：将DeterType=0和1拆分为独立ARGS_SEL条目
- 可审查性：低
- 审查规则建议：模板tiling key变更需配合精度测试验证；多值选项拆分时确保每个组合参数完整

### 51076ed1 修复伪量化叠加后量化的bug
- 根因类别：条件初始化导致未初始化使用
- 涉及文件：attention/common/op_kernel/arch-310/flash_attention_score_antiquant_baseapi.h
- 缺陷描述：postQuantOffsetQue的InitBuffer被isPostQuantOffsetExist条件守护，但后续代码无论offset是否存在都会使用该queue导致buffer未初始化crash
- 修复模式：去除不正确的条件守护改为无条件初始化
- 可审查性：高
- 审查规则建议：Buffer/Queue初始化如果后续使用路径不全在同一条件分支下则初始化不应加条件；追踪queue的所有使用点确认初始化覆盖

### d2938dae 修复combine v2
- 根因类别：初始化顺序错误+buffer复用冲突
- 涉及文件：mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2.h
- 缺陷描述：combine v2的InitDataStatus在InitInputAndOutput之前调用但依赖后者结果；expertScalesBuf_被复用为状态临时buffer导致数据污染
- 修复模式：重构初始化顺序+解耦buffer复用+状态追踪细粒度化
- 可审查性：低
- 审查规则建议：分布式通信算子的状态初始化需确保依赖关系正确；buffer不应跨功能复用

### c8040356 修复swin transformer kernel/host代码bug
- 根因类别：队列位置枚举错误+错误API调用
- 涉及文件：ffn/swin_attention_ffn/op_kernel/swin_attention_ffn.cpp, swin_attention_ffn_tiling.cpp
- 缺陷描述：(1)kernel中inQueueTmp的队列位置声明为VECCALC应为VECIN导致流水线调度异常；(2)host tiling中多余调用SetBufferSpace接口
- 修复模式：修正队列位置枚举值+删除错误API调用
- 可审查性：高
- 审查规则建议：TQue的QuePosition需与数据流方向匹配(VECIN/VECOUT/VECCALC)

### dbbaf7f2 字节 128P问题定位修复
- 根因类别：硬编码上限+过严运行时校验
- 涉及文件：mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/*_tiling.cpp, mc2/allto_allv_grouped_mat_mul/op_kernel/*_tiling.h, mc2/grouped_mat_mul_allto_allv同
- 缺陷描述：mc2通信算子epWorldSize上限硬编码64导致128P场景参数校验失败；tiling对send/recv数据量做[2MB,100MB]区间校验在大规模场景误报错
- 修复模式：扩展MAX_EP_RANK_SIZE从64到128+移除不合理的运行时约束检查
- 可审查性：高
- 审查规则建议：硬编码常量上限(MAX_*_SIZE)是否覆盖所有支持的部署规格；运行时校验条件与实际使用场景匹配避免过严

### acff787b fix rope_with_sin_cos_cache ut
- 根因类别：InitBuffer传0大小+计算公式错误
- 涉及文件：posembedding/rope_with_sin_cos_cache/op_kernel/rope_with_sin_cos_cache_fp32.h
- 缺陷描述：kernel代码在__CCE_KT_TEST__宏下InitBuffer的size传0导致crash；dstShape_4Negone_计算从loopN*num_heads_max应为loopN*(num_heads_max-1)
- 修复模式：条件编译区分buffer初始化+修正计算公式
- 可审查性：中
- 审查规则建议：InitBuffer不应传入size=0；__CCE_KT_TEST__下的逻辑应与正式逻辑语义一致

### 02a1ea71 非量化拦截校验修复
- 根因类别：校验条件不完整(缺少模式区分)
- 涉及文件：attention/prompt_flash_attention/regbase/ophost/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：TND layout下非PA模式缺少query/key/value维度校验分支，且缺少q与k的d维度一致性检查。相比之前被回退的版本(569)新增了fromTilingSink/enableTensorList/enablePA条件守卫修复误拦截
- 修复模式：增加缺失条件分支+细化校验条件
- 可审查性：高
- 审查规则建议：InputLayout的所有枚举值是否都有对应维度校验分支(switch完备性检查)

### 580efee4 修改mxfp场景下对scale的校验问题
- 根因类别：参数传递错误(多余运算)
- 涉及文件：mc2/matmul_all_reduce/op_host/op_tiling/matmul_all_reduce_tiling.cpp
- 缺陷描述：MXFP场景调用CheckMXScenarioScaleShape时对mValue做了多余的/GetBatchValue()除法，导致scale shape校验使用错误的M值
- 修复模式：移除多余运算(mValue/GetBatchValue() -> mValue)
- 可审查性：高
- 审查规则建议：对比函数签名与调用点参数语义，检查是否存在调用者预处理了本应由被调用者处理的值

### c2aa7939 回退图模式output空tensor拦截
- 根因类别：校验引入回归(图模式未验证)
- 涉及文件：mc2/matmul_all_reduce/op_host/matmul_all_reduce_infershape.cpp
- 缺陷描述：570(cdaf9421)在infershape增加空tensor拦截，但该拦截在图模式下导致问题(s*m*n==0条件过严)。此commit回退infershape中有问题的拦截逻辑
- 修复模式：回退有问题的检查逻辑
- 可审查性：高
- 审查规则建议：infershape新增shape校验应同时验证图模式和单算子模式两种路径

### 90904b5e 回退代码(校验误拦截)
- 根因类别：校验条件过宽导致误拦截
- 涉及文件：attention/prompt_flash_attention/regbase/ophost/prompt_flash_attention_tiling_v2.cpp
- 缺陷描述：之前引入的TND格式校验和d维度一致性校验缺少fromTilingSink/enableTensorList/enablePA条件守卫，在某些合法场景下误报错。回退后由562重新以正确条件加入
- 修复模式：回退有缺陷的校验逻辑
- 可审查性：高
- 审查规则建议：新增校验逻辑应考虑所有调用路径(tilingSink/tensorList/PA模式)，做truth table审查

### cdaf9421 output空tensor问题增加拦截处理
- 根因类别：输入校验缺失(空tensor未拦截)
- 涉及文件：mc2/matmul_all_reduce/op_host/matmul_all_reduce_infershape.cpp, op_api/aclnn_weight_quant_matmul_all_reduce.cpp, op_tiling/matmul_all_reduce_tiling.cpp
- 缺陷描述：matmul_all_reduce在output为空tensor时缺少拦截导致后续计算异常。infershape层校验后被564回退(图模式有问题)，op_api层校验保留有效
- 修复模式：多层增加defensive check(空tensor校验+错误报告)
- 可审查性：高
- 审查规则建议：op_api层和infershape层空值校验应同步审查确保逻辑一致且不冲突

### 0bd6d90b buf bug fix
- 根因类别：buffer分配计算错误(RoundUp vs Ceil)
- 涉及文件：mc2/moe_distribute_combine/op_kernel/moe_distribute_combine_a2.h
- 缺陷描述：BuffInit()中buffer大小计算用RoundUp(axisBS_, aivNum_)做核间任务分配应用Ceil()取上整；对齐常量从B32_PER_BLOCK修正为UB_ALIGN确保buffer对齐到正确边界
- 修复模式：替换计算函数+修正对齐常量
- 可审查性：中
- 审查规则建议：buffer分配中RoundUp/Ceil/DivCeil的使用是否正确匹配"取整"vs"对齐"语义

### 5a23b230 mla with tensorlist bugfix
- 根因类别：offset初始化对象和索引错误
- 涉及文件：attention/common/op_kernel/arch-310/flash_attention_score_block_cube.h
- 缺陷描述：BNSD布局下CalcS2Coord中对valueGm做offsetCalculator.Init应该是对keyRopeGm做(仅hasRope时)；IterateBmm2入口缺少isKvContinuous==0时value GM offset初始化；计算keyRope GM offset时用了错误的batch索引(curBIdx应为boIdx)
- 修复模式：修正offset初始化对象+移动初始化位置+修正索引变量
- 可审查性：低
- 审查规则建议：offset计算中使用的索引变量应与其所属tensor的维度语义一致；GM offset初始化时机应与首次使用位置匹配

### 1c39b10b fix mla_prolog_v1/v2报GE失败日志
- 根因类别：接口兼容性错误(V1/V2调用V3专属属性)
- 涉及文件：attention/mla_prolog/op_host/mla_prolog_tiling.cpp
- 缺陷描述：ConvertContext中对qcQrScale和kcScale属性的读取未区分算子版本，V1/V2没有注册这些V3新增属性，直接调用GetAttrPointer触发GE框架报失败日志
- 修复模式：增加版本条件判断(opType=="MlaPrologV3"时才读取)
- 可审查性：高
- 审查规则建议：多版本算子共享tiling代码时新增属性的读取必须有版本守卫

### a3b30f8f fix bug MlaProlog RmsNorm_VF & remove Rope_VF
- 根因类别：拼写错误+类型错误+输出目标错误+符号错误(多处VF路径bug)
- 涉及文件：attention/mla_prolog/op_kernel/service_rms_norm.h, mla_prolog_vector_comm.h, regbase/opkernel/vf/vf_rms_norm.h, vf_rope.h
- 缺陷描述：(1)__CCE__AICORE__宏多一个下划线导致310 VF分支不生效；(2)RmsNorm_VF输出目标应为xFp32Local非outputLocal；(3)MitroAPI拼写错误应为MicroAPI；(4)loopOffset应为uint64_t防溢出；(5)gamma寄存器类型应为GammaType非C；(6)Rope计算符号-1应为1
- 修复模式：修正拼写、类型、枚举值、计算符号
- 可审查性：高
- 审查规则建议：预处理宏拼写应有自动化校验；VF/非VF路径输出目标应保持一致

### 29edf728 TND场景 mask计算数据溢出 / kernel已知bug修复
- 根因类别：整数溢出+类型转换优先级错误
- 涉及文件：attention/common/op_kernel/vector_common.h, arch32/fia_kernel_nonquant.h, fia_kernel_nonquant_mla.h
- 缺陷描述：(1)ComputeAttenMaskOffsetNoCompress中batchIdx*batchOffset两个uint32_t相乘在TND+BSSmask场景溢出，需先cast<uint64_t>；(2)static_cast<uint32_t>(s2FirstToken)/s2BaseSize先截断int64_t到uint32_t再除，应先除再截断
- 修复模式：调整类型转换位置/添加显式cast防溢出+重构跨G轴计算
- 可审查性：中
- 审查规则建议：uint32_t*uint32_t赋给uint64_t时乘法前至少一个操作数提升；static_cast与算术运算混用时关注截断发生在运算前还是后

### c49af93f 修复aclnn异常返回值
- 根因类别：错误码/返回值类型错误
- 涉及文件：mc2/elastic_receivable_test/op_host/op_api/aclnn_elastic_receivable_test.cpp, mc2/moe_distribute_buffer_reset/op_host/op_api/aclnn_moe_distribute_buffer_reset.cpp
- 缺陷描述：CheckParams中group name长度非法时日志错误码用ACLNN_ERR_PARAM_NULLPTR应为ACLNN_ERR_PARAM_INVALID；return false类型不匹配(函数返回aclnnStatus)应return ACLNN_ERR_PARAM_INVALID
- 修复模式：修正错误码+修正return值类型
- 可审查性：高
- 审查规则建议：参数校验错误码应与错误类型匹配；函数返回枚举类型时不应return bool值

### fb38bcbb check类问题单
- 根因类别：host侧tiling check多处校验逻辑缺陷
- 涉及文件：attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_check_*.cpp等
- 缺陷描述：FIA多处check修复：(1)CheckActualSeqLensKv应用kvLayout_非qLayout_；(2)preTokens/nextTokens合法性判断重写(未检查两者同为负数+负数交叉条件有误)；(3)sparseMode缺少白名单校验(0/1/2/3/4)；(4)GetValueD/GetQkvD缺少DimNum维度数校验致越界；(5)S1S2序列化字符串从"SS"修正为"S1S2"；(6)删除过严的qTSize上限校验；(7)TND下新增kT与actualSeqLensKv最后元素一致性校验
- 修复模式：修正判断条件+新增缺失校验+调整校验执行顺序+修正序列化字符串
- 可审查性：中
- 审查规则建议：Q/KV属性判断确认用Q侧还是KV侧layout；枚举型参数应有白名单校验；按layout取dim前应校验dimNum

### 5146ed28 fix bug for LoadData3DParamsV2Pro in david
- 根因类别：平台条件编译分支缺失
- 涉及文件：attention/mla_prolog/op_kernel/service_matmul.h
- 缺陷描述：LoadDataL0K3DPro缺少310平台(david)的适配分支，该平台上LoadData3DParamsV2Pro不适用导致错误
- 修复模式：增加#if __CCE_AICORE__ == 310条件编译分支
- 可审查性：低
- 审查规则建议：涉及LoadData系列API时检查是否覆盖所有目标平台的条件编译分支

### bbb3abd1 fix datacopypad unit16 bug
- 根因类别：API参数类型错误
- 涉及文件：attention/common/op_kernel/arch-310/attenmask.h, pse.h
- 缺陷描述：310架构下DataCopyPad使用错误参数类型(DataCopyParams+DataCopyPadParams应为DataCopyExtParams+DataCopyPadExtParams<T>)，attenmask.h中缺少dstStride设置导致uint16数据拷贝padding不正确
- 修复模式：替换API参数结构体类型+补充缺失字段
- 可审查性：中
- 审查规则建议：DataCopyPad调用确保参数结构体类型与函数签名匹配；检查Ext版本API是否需要额外字段

### 8256ab23 修复GMM A8W4 tiling key路由错误
- 根因类别：条件判断顺序错误(标志位包含关系)
- 涉及文件：gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp
- 缺陷描述：isA8W4FakeA8W8_的tiling key设置放在isA8W8_判断之后，但isA8W8_定义包含isA8W4FakeA8W8_(通过||)导致A8W4场景被错误路由到A8W8的tiling key
- 修复模式：调整条件判断优先级+解耦标志位定义
- 可审查性：高
- 审查规则建议：if-else链路由时检查条件间是否存在包含关系致后续分支不可达；标志位定义不应隐式包含其他标志位

### 102f0633 新增处理keyAntiquantScale输入(B,S)的情况
- 根因类别：输入shape维度处理不完整
- 涉及文件：attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp
- 缺陷描述：GetAntiquantSeqLength中获取antiquant序列长度的维度索引只考虑3维输入未处理2维(B,S)情况，取到错误维度值
- 修复模式：增加维度数量条件判断分支
- 可审查性：中
- 审查规则建议：通过GetDim取shape维度时检查是否覆盖所有合法维度数量；对tensor shape做维度索引前先校验DimNum

### 051f878c 修复mla_prolog_v3 infershape bug
- 根因类别：API调用参数顺序错误
- 涉及文件：attention/mla_prolog_v3/op_host/mla_prolog_v3_infershape.cpp
- 缺陷描述：SetOutputDataType调用参数写反，把inputDataType当输出index、DT_FLOAT当数据类型，应为SetOutputDataType(QUERY_NORM_INDEX, inputDataType)导致输出数据类型推导完全错误
- 修复模式：修正函数调用参数顺序
- 可审查性：高
- 审查规则建议：同一函数多次调用时检查参数模式一致性；SetOutputDataType第一个参数应为输出索引常量

### a9d9995e [mlaprolog] fix sync problem
- 根因类别：同步信号量值错误
- 涉及文件：attention/mla_prolog/op_kernel/mla_prolog_comm.h
- 缺陷描述：cube/vector流水线间同步常量值设置错误(如FINISH_MM_ALL从0x0改为0x7等)，导致流水线同步不正确可能引起数据竞争或死锁
- 修复模式：修正同步常量的位掩码值
- 可审查性：低
- 审查规则建议：同步相关常量应有明确位域语义注释；修改sync flag需配套验证所有使用点时序正确性

### 9ceef237 fix bugs NBSD场景 actualSeqLen==0时多刷0
- 根因类别：参数传递错误(stride级别)
- 涉及文件：attention/common/op_kernel/memory_copy.h
- 缺陷描述：DealActSeqLenIsZero中调用InitOutput清零输出时第二个参数错误使用GetStrideG()(G维度步长)应为GetStrideB()(B维度步长)导致清零范围不正确
- 修复模式：修正stride参数
- 可审查性：高
- 审查规则建议：清零/初始化操作的范围参数应与当前循环层级stride语义匹配

### e13cb3ab bug fix for TND and FD
- 根因类别：TND layout多处地址偏移计算遗漏
- 涉及文件：attention/common/op_kernel/arch-310/flash_attention_score_antiquant_baseapi.h
- 缺陷描述：FlashDecode在TND layout下：(1)s1LoopTimes硬编码为1未根据actualS1Size计算；(2)accumOut和lse的GM偏移缺少goIdx维度；(3)FlashDecodeCompute未区分TND处理actualSeqLenQ/KV和attenOutOffset；(4)GetActualSeqLenKV在TND非PA场景未做累计值到当前值转换
- 修复模式：补充缺失的TND layout条件分支和偏移量计算
- 可审查性：中
- 审查规则建议：多layout算子检查每种layout在所有地址计算路径是否有对应处理；新增layout时枚举所有offset计算点确认覆盖

### aba3bd8f bn2 pingpang fix
- 根因类别：tilingKey参数遗漏(传0占位)
- 涉及文件：attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling_s1s2_bn2gs1s2.cpp
- 缺陷描述：FAG的GetTilingKey生成时第15个位置传了0而非tndS1Pingpong，不同pingpong配置生成相同tilingKey可能选到错误kernel实现
- 修复模式：将硬编码0替换为static_cast<uint8_t>(tndS1Pingpong)
- 可审查性：高
- 审查规则建议：tilingKey生成函数中每个位字段与tiling参数列表一一对应，避免遗漏传0占位

### d34fc35a PFA IFA FIA fix warning(实为空指针+缓冲区溢出)
- 根因类别：空指针解引用+缓冲区溢出
- 涉及文件：attention/incre_flash_attention/op_host/fallback_incre_flash_attention.cpp, attention/prompt_flash_attention/op_host/fallback_prompt_flash_attention.cpp
- 缺陷描述：IFA/PFA的fallback路径中GetAttrPointer返回值未做空指针检查直接解引用可能crash；FIA example中char数组长度未+1导致strcpy写越界
- 修复模式：添加空指针检查+提前返回错误；数组大小+1
- 可审查性：高
- 审查规则建议：GetAttrPointer返回值必须null check后才能解引用；char[]用于strcpy长度必须包含null terminator

### 855ba7b4 tnd case bugfix
- 根因类别：优化复用条件在TND下不成立
- 涉及文件：attention/common/op_kernel/arch-310/dropmask.h
- 缺陷描述：GenDropMask中s1和s2均对齐时复用dropmaskIndexVec，但TND layout下不同batch的actual size不同，某batch对齐不代表其他也对齐导致dropmask数据错误
- 修复模式：复用条件增加layoutType != LAYOUT_TND
- 可审查性：高
- 审查规则建议：可复用/缓存优化逻辑需检查前提假设在所有layout/场景下是否成立

### 7ae9e1ac fa tnd 问题修复
- 根因类别：TND下IsLastBN逻辑错误
- 涉及文件：attention/common/op_kernel/arch-310/flash_attention_kvsame_bn2gs1s2.h
- 缺陷描述：IsLastBN函数在TND下通过遍历后续batch判断是否为最后有效BN，actualSeqQLen为0时continue条件导致跳过有效batch；Process循环中TND下actualSeqQLen为0时continue与上层逻辑冲突
- 修复模式：删除错误的特殊分支回归通用逻辑(bnIdx == bnEndIdx - 1)
- 可审查性：中
- 审查规则建议：IsLast*类判断函数验证其在边界case(size=0/所有对齐/混合对齐)下行为

### fcc05cad bugfix: fix gmmdsq dfx bug
- 根因类别：格式化字符串参数不匹配+返回值忽略
- 涉及文件：gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/aclnn_gmm_dsq_base.h, aclnn_grouped_matmul_swiglu_quant_v2.cpp
- 缺陷描述：(1)OP_LOGE中%s格式符对应参数传了tensor指针(w/wScale)而非字符串致日志异常甚至UB；(2)handler->Process()返回值被忽略始终返回ACLNN_SUCCESS即使失败也不报错
- 修复模式：修正printf格式参数+正确传播返回值
- 可审查性：高
- 审查规则建议：OP_LOGE中%s参数必须为const char*不能传tensor/object指针；有返回值函数调用不应忽略返回值

### 13ae078c tilingkey layout type fix
- 根因类别：枚举值magic number写错
- 涉及文件：attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad.cpp
- 缺陷描述：FAG kernel中bn2分支内TND layout判断条件Layout==4写错应为Layout==3(TND枚举值为3)导致TND的bn2 pingpong分支无法命中。3个数据类型(half/bfloat16/float)同一位置都有此错误
- 修复模式：将Layout==4改为Layout==3共3处
- 可审查性：高
- 审查规则建议：禁止直接使用magic number比较layout/enum应使用枚举常量名

### 2adfddfd 修复线下编译问题
- 根因类别：头文件引用路径错误
- 涉及文件：mc2/common/inc/mc2_matmul_tiling_cfg.h
- 缺陷描述：产品代码头文件中`#include "op_log.h"`和`#include "mc2_tiling_struct.h"`路径不正确，应为`mc2_log.h`和`tiling/mc2_tiling_struct.h`，导致编译失败
- 修复模式：修正include路径
- 可审查性：高
- 审查规则建议：头文件被移动或重命名时同步更新所有引用处

### 0163bbb6 Fix Bug in GQA NoQuant Branch
- 根因类别：多重逻辑缺陷（同步/计算/边界/初始化）
- 涉及文件：attention/common/op_host/arch32/fia_tiling_nonquant.cpp, fia_tiling_nonquant.h, split_core_v1.h, attention/common/op_kernel/arch32/fia_block_cube_nonquant_gqa.h, fia_block_vec_flashdecode.h, fia_block_vec_nonquant.h, fia_kernel_nonquant.h, memory_copy.h, attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_v3.cpp
- 缺陷描述：GQA非量化分支8个bug:(1)LSE BNSD拷贝未考虑actualSeqLength导致刷inf(2)Q/V buffer多余SetFlag/WaitFlag造成同步问题(3)CalcCurS2StartEnd边界判断越界(4)CheckGqaConstrain被注释掉直接返回false(5)actualLenQDims==1时s1Size计算缺失(6)CalcMaxMmResSize用mBaseSize_而非常量512(7)CalcNormalWorkspaceSize多算workspace(8)maskInfo.maskValue未初始化
- 修复模式：多点修正 - 重写数据拷贝、移除多余同步、修正边界判断、恢复校验、修正workspace计算、初始化字段
- 可审查性：低
- 审查规则建议：被注释掉的关键校验逻辑应触发告警；结构体字段初始化完整性检查；硬件同步事件配对和必要性审查

### e4ad525a Fix the issue with calculating the correct padding size for data ending
- 根因类别：边界条件判断不严谨（==0 vs <=0）
- 涉及文件：attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling_s1s2_bn2gs1s2.cpp, _basic.cpp, _sab.cpp, attention/nsa_selected_attention_grad/op_host/nsa_selected_attention_grad_tiling_bs1.cpp, _basic.cpp
- 缺陷描述：EOD补零大小计算中tailZeroCount判断条件用`==0`但actualSeqQlen/KVlen可能为负值表示无效，导致负值序列未被识别为尾部填充，batch大小计算偏大。同时_basic.cpp变体完全缺失tailZeroCount逻辑
- 修复模式：`==0`改`<=0`覆盖负值；在缺失文件中补全逻辑
- 可审查性：高
- 审查规则建议：对序列长度等可能为负/零的变量比较条件应区分语义；同一逻辑多文件变体需保持一致

### 260fe964 修复非量化场景下mmar走错模板
- 根因类别：tiling路径选择逻辑错误
- 涉及文件：mc2/matmul_all_reduce/op_host/op_tiling/arch35/matmul_all_reduce_tiling_910_95.cpp/.h, mc2/matmul_all_reduce/op_host/op_tiling/matmul_all_reduce_tiling.cpp/.h, mc2/matmul_all_reduce/op_kernel/arch35/matmul_all_reduce_910_general.h, matmul_all_reduce_empty_tensor_k_general.h, mc2/common/inc/mc2_matmul_tiling_cfg.h, mc2/common/src/mc2_matmul_tiling_cfg.cpp
- 缺陷描述：非量化场景下tilingKey选择依赖GetTilingKey()返回值而非isAdd参数，导致走错模板；kernel侧使用旧的MatmulTilingData结构而非MC2MatmulV3TilingData；BiasType模板参数使用DTYPE_BIAS_FOR_MC2而非DTYPE_X1
- 修复模式：重构tiling计算路径 + 修正模板参数 + 统一校验到基类
- 可审查性：低
- 审查规则建议：tilingKey/模板选择依赖运行时状态时应检查是否覆盖所有场景分支

### 4b848ce2 fix spelling errors of the moe_gating_top_k
- 根因类别：变量名拼写错误
- 涉及文件：moe/moe_gating_top_k/op_host/moe_gating_top_k_tiling.cpp, moe_gating_top_k_tiling_arch35.cpp
- 缺陷描述：变量名xDimNnum/biasDimNnum多了一个N，应为xDimNum/biasDimNum
- 修复模式：变量重命名修正typo
- 可审查性：高
- 审查规则建议：拼写检查工具检测变量名中的连续重复字母模式

### 45c84324 修复NsaCompressAttention精度问题
- 根因类别：边界条件处理缺陷 + 流水同步缺失
- 涉及文件：attention/nsa_compress_attention/op_kernel/nsa_compress_attention_s1s2_bn2gs1_sab.h
- 缺陷描述：两个独立bug:(1)softmax CopyIn的softmaxSrcBlockLen直接取s2Length但当s2Length>s2RealSize时越界读取无效数据，应取min(s2Length,s2RealSize)(2)gSize>1时reduce操作前缺少PipeBarrier<PIPE_V>()同步屏障导致数据竞争
- 修复模式：添加min clamp保护 + 插入流水线同步屏障
- 可审查性：高
- 审查规则建议：DMA搬运blockLen应检查是否超过实际buffer大小；跨流水线数据依赖必须有PipeBarrier

### 6d6b35a0 fix mem.h InitBuffer
- 根因类别：API变更适配错误
- 涉及文件：common/include/kernel/mem.h
- 缺陷描述：InitBuffer+手动设置logicPos的两步初始化方式已废弃，应改用LocalTensor构造函数一步完成初始化(传入TPosition,offset,size)
- 修复模式：替换废弃API调用为正确的构造函数调用
- 可审查性：中
- 审查规则建议：检测InitBuffer后紧跟address_.logicPos手动赋值的模式

### 7fb85729 反向V write & V read bug fix
- 根因类别：硬件事件类型(HardEvent)使用错误
- 涉及文件：attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad_s1s2_bn2gs1s2.h
- 缺陷描述：LocalReleaseEventID中HardEvent类型与变量语义不匹配：structVWaitMte2应用MTE2_V而非MTE3_MTE2，structMte3WaitV应用V_MTE3而非MTE3_MTE2，structMte2WaitV应用V_MTE2而非MTE3_MTE2，导致同步失效产生数据竞争
- 修复模式：修正ReleaseEventID模板参数为与变量命名一致的HardEvent类型
- 可审查性：高
- 审查规则建议：ReleaseEventID/AllocEventID的HardEvent类型应与变量命名中的事件方向一致；同一事件ID的Alloc/Set/Wait/Release应使用相同HardEvent类型

### f8a4338e 修复NsaCompressAttention impScore同步问题
- 根因类别：WaitFlag位置不正确（先用后等反模式）
- 涉及文件：attention/nsa_compress_attention/op_kernel/nsa_compress_attention_s1s2_bn2gs1_sab.h
- 缺陷描述：WaitFlag<HardEvent::MTE3_V>原在循环后，但循环体内V单元依赖MTE3写入的数据，必须在循环前等待。原位置过晚导致V单元可能在MTE3未完成时就开始计算
- 修复模式：将WaitFlag从循环后移到循环前
- 可审查性：高
- 审查规则建议：检测WaitFlag与数据消费代码之间是否存在"先用后等"反模式

### 3af323dd 全量化matmulAllReduce core dump修复
- 根因类别：virtual关键字遗漏导致多态失效
- 涉及文件：mc2/3rd/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling_base.h
- 缺陷描述：基类SetPlatformInfoForTiling()缺少virtual关键字，子类重写但通过基类指针调用时不触发动态派发，执行基类版本导致平台信息不正确引发core dump
- 修复模式：添加virtual关键字启用多态
- 可审查性：高
- 审查规则建议：检测子类中与基类同名同参方法但基类未声明virtual的情况；使用-Wsuggest-override编译选项

### ee638d9b fix wqbmm tilingkey
- 根因类别：跨模块tiling key常量不一致
- 涉及文件：mc2/matmul_all_reduce/op_host/op_tiling/weight_quant_matmul_all_reduce_tiling.h, mc2/matmul_all_reduce/op_kernel/matmul_all_reduce.cpp
- 缺陷描述：上游nn仓wqbmm修改了tilingkey编码值，下游weightQuantMatmulAllReduce未同步更新，host侧注册与kernel侧TILING_KEY_IS使用旧值导致tiling匹配失败
- 修复模式：同步更新host和kernel中的tilingkey常量
- 可审查性：中
- 审查规则建议：tiling key应定义为共享常量而非硬编码魔数防止不一致

### 12da221a 修复FA s1 模板出现S1切分tail长度小的场景 同步问题
- 根因类别：流水同步屏障缺失
- 涉及文件：attention/flash_attention_score/op_kernel/flash_attention_score_s1_bn2gs1.h
- 缺陷描述：ProcessVec1中GetBmm1Result后、设置MTE2_V事件标志前缺少PipeBarrier<PIPE_V>()，S1切分tail长度小时Vector流水线前序操作未完成就开始MTE2搬运导致数据竞争
- 修复模式：插入PipeBarrier<PIPE_V>()同步屏障
- 可审查性：低
- 审查规则建议：SetFlag<HardEvent::MTE2_V>调用前检查是否有PipeBarrier

### b41315b3 fix WeightQuantMatmulAllReduce Tilingkey
- 根因类别：tiling key值溢出为UINT64_MAX
- 涉及文件：mc2/matmul_all_reduce/op_host/op_tiling/weight_quant_matmul_all_reduce_tiling.h, mc2/matmul_all_reduce/op_kernel/matmul_all_reduce.cpp, mc2/matmul_all_reduce_add_rms_norm对应文件
- 缺陷描述：WeightQuant模式tilingKey使用了18446744073709551615(UINT64_MAX)作为key值，这是无效溢出值不是合法tiling key编码，导致tiling注册与kernel分支不匹配
- 修复模式：修正硬编码常量值，保持host/device一致
- 可审查性：中
- 审查规则建议：检测UINT64_MAX/0xFFFFFFFFFFFFFFFF作为业务常量使用；要求REGISTER_TILING_DATA_CLASS的key与kernel TILING_KEY_IS完全一致

### 801f1c87 a8w4 fix
- 根因类别：条件判断变量错误
- 涉及文件：gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp
- 缺陷描述：A8W4Tiling判断MSD路径条件用isPerchannel==false但应为tuningConfig==0L，isPerchannel语义与MSD路径无关，导致A8W4量化错误选择/跳过MSD路径
- 修复模式：修正布尔条件表达式中的判断变量
- 可审查性：中
- 审查规则建议：多条件布尔表达式review时重点关注每个子条件语义是否与整体判断意图一致

### bff19585 fix-gmmswigluquant A8W4 split-workspace
- 根因类别：workspace计算逻辑分支遗漏
- 涉及文件：gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_tiling.cpp
- 缺陷描述：MSD与非MSD模式workspace使用方式不同但在if/else外统一计算，MSD需double buffer完整空间而非min截断，非MSD的offset应为0
- 修复模式：将共用逻辑拆分到条件分支内各自独立配置
- 可审查性：中
- 审查规则建议：if/else分支前置共享计算应验证对所有分支是否适用；workspace大小应与实际使用模式严格对应

### d9d56b9a fag bn2 template bugfix
- 根因类别：对齐计算错误 + DataCopy stride参数复用错误
- 涉及文件：attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad_s1s2_bn2.h
- 缺陷描述：(1)dimDAlign对齐粒度用dataCopyBlockNum应为C0_SIZE(16)(2)DataCopy的stride变量输入输出共用但需求不同，当dimDAlign!=postDimDAlign时stride需分别设置
- 修复模式：修正对齐常量 + 拆分共用stride为独立输入/输出参数
- 可审查性：低
- 审查规则建议：对齐常量应与数据类型和硬件要求匹配；输入输出有不同stride需求时禁止复用同一变量

### c2f212df fix FA oom pkg build bugs
- 根因类别：构建配置遗漏宏定义
- 涉及文件：cmake/func.cmake
- 缺陷描述：oom编译模式下缺少-DNOT_DYNAMIC_COMPILE宏定义，导致oom pkg构建时代码走了动态编译路径，与oom模式不兼容
- 修复模式：在oom选项处理中追加-DNOT_DYNAMIC_COMPILE编译宏
- 可审查性：中
- 审查规则建议：新增编译模式时检查所有相关宏定义是否同步添加；oom模式应互斥动态编译路径

### d4bcf227 修复alibi场景QS和KVS不相等
- 根因类别：条件判断不完整 + 除零风险
- 涉及文件：attention/prompt_flash_attention/regbase/ophost/prompt_flash_attention_tiling_v2.cpp, .h
- 缺陷描述：(1)PFAMerge启用条件未排除alibi/PSE场景(enableAlibiPse)，导致alibi场景下QS!=KVS时错误走合轴路径。(2)gSize计算未检查numKeyValueHeads>0，存在除零风险。(3)原条件在else if内嵌套不够清晰，enablePFAMerge赋值后面还有其他逻辑
- 修复模式：提取独立CheckPFAMerge方法，增加enableAlibiPse排除条件 + numKeyValueHeads>0保护
- 可审查性：高
- 审查规则建议：功能合轴/优化路径必须验证所有特殊场景(alibi/mask/PSE等)是否兼容；除法运算前必须检查除数非零

### 9ee7388a fix gmm-swiglu-quant
- 根因类别：A8W4量化路径多处参数错误
- 涉及文件：gmm/grouped_matmul_swiglu_quant/op_host/...tiling.cpp, .h, op_api/aclnn...cpp, op_kernel/...cpp, ...a8w4_msd_mid.h
- 缺陷描述：(1)SetBType格式错误ND应为NZ(2)Matmul类型MMImplType应为MMImplTypeStatic(3)baseM/baseN硬编码未适配decode场景的动态调整(4)SetFixSplit未调用(5)A8W4 MSD覆盖参数(dbL0B/stepKa/stepKb等)未设置(6)UnpackInt32ToInt4缺少SetStorageShape调用(7)字符串连接处\续行符应改为字符串拼接
- 修复模式：host侧增加decode场景baseM/baseN/baseK动态选择 + kernel侧从tiling读取替代硬编码 + 修正API调用
- 可审查性：低
- 审查规则建议：新量化路径引入时需对照参考实现逐项检查matmul配置(Format/Type/FixSplit/覆盖参数)

### 17675a9d fix dispatch combine magic change to int64 type
- 根因类别：数据类型宽度不匹配(int32 vs uint64)
- 涉及文件：mc2/moe_distribute_combine/op_kernel/moe_distribute_combine_a2_layered.h, mc2/moe_distribute_dispatch/op_kernel/moe_distribute_dispatch_a2_layered.h
- 缺陷描述：RDMA同步标志magic值/flagGlobal/magicGlobal使用int32_t但实际通信协议要求uint64_t，导致：(1)magic值偏移计算错误(MAGIC_OFFSET/sizeof(int32_t)应为sizeof(uint64_t))(2)flag比较可能因截断产生误匹配(3)magic number硬编码12345/123/321散布于代码中
- 修复模式：统一改为uint64_t + 提取命名常量GM2IPC_SYNC_FLAG/RDMA_TOKEN_ARRIVED_FLAG/RDMA_TOKEN_END_FLAG + 修正IPC_MAGIC_OFFSET偏移量
- 可审查性：高
- 审查规则建议：跨设备通信flag/magic的类型必须与通信协议匹配(通常uint64_t)；硬编码magic number必须提取为命名常量

### 5f6194ee fix include of arch35
- 根因类别：头文件路径缺失 + API命名空间变更未同步
- 涉及文件：moe/moe_re_routing/op_kernel/arch35/moe_re_routing_r_regbase.h
- 缺陷描述：arch35适配新增文件缺少#include "../../inc/kernel_utils.h"，且Ops::Base::CeilDiv/FloorDiv在arch35上应使用ops::CeilDiv/FloorDiv，命名空间不匹配导致编译失败
- 修复模式：补充include + 替换命名空间前缀
- 可审查性：高
- 审查规则建议：新arch适配PR必须检查所有API调用的命名空间在目标平台是否可用

### 51b75bef 修复伪量化coredump和精度失败问题
- 根因类别：流水线循环控制逻辑错误
- 涉及文件：attention/common/op_kernel/arch-310/flash_attention_score_antiquant_baseapi.h
- 缺陷描述：原IsLastBN逻辑在TND场景下判断最后一个BN不正确(依赖actualSeqQlen比较，多batch相同seq长度时误判)，导致pipeline preload的最后两轮循环(taskIdx+1/+2)条件计算错误，ProcessVec1/ProcessVec2执行了无效数据导致coredump或精度异常
- 修复模式：删除IsLastBN函数，改为简单的bnIdx==bnEndIdx-1判断；重构循环为tempGS1End=s1LoopTimes+2(最后BN额外2轮)，用switch(extraGS1)控制notLastTwoLoop/notLast状态
- 可审查性：低
- 审查规则建议：pipeline preload模式中"最后N轮"判断必须用确定性条件(简单索引比较)，不可依赖数据内容推断

### 4f15b0a7 David: fix postquant
- 根因类别：stride计算公式错误 + offset乘法项遗漏
- 涉及文件：attention/common/op_kernel/arch-310/flash_attention_score_block_vec_infer.h
- 缺陷描述：(1)PostQuantPerChnl的dstStride除数硬编码8应为32/sizeof(POSTQUANT_PARAMS_T)——当POSTQUANT_PARAMS_T不是4字节时stride计算错误(2)perChannelQuantGQAOffset的vec2S1Idx缺少乘以vec2S1BaseSize，导致GQA场景下量化参数偏移量不正确
- 修复模式：修正除数为32/sizeof(T) + 补充vec2S1BaseSize乘数
- 可审查性：中
- 审查规则建议：DataCopy stride公式中的对齐因子必须与数据类型sizeof匹配；GM offset组合表达式中每个维度因子不可遗漏

### ef094757 Revert "tiling模板代码优化"
- 根因类别：模板参数泛化优化引入逻辑错误(Revert)
- 涉及文件：attention/common/op_kernel/arch-310/flash_attention_score_block_cube.h, flash_attention_score_common_regbase.h, flash_attention_score_entry_regbase.h, flash_attention_score_template_tiling_key.h
- 缺陷描述：原优化将L0C buffer policy判断从==128改为通用公式s1*s2*FLOAT_BYTES<=L0C/4，但(1)公式未正确考虑所有tile组合的L0C容量约束(2)entry中增加了dvTemplateType>dTemplateType的early return过滤掉了合法kernel实例化(3)tiling_key.h中D/Dv的组合从分开列举改为统一列表导致无效组合被编译
- 修复模式：完全revert回==128的保守判断 + 恢复D/Dv组合按单独列表约束
- 可审查性：中
- 审查规则建议：L0C/L1 buffer policy优化必须对所有模板参数组合验证容量约束；模板实例化列表变更需确认不会引入无效组合

### 071b811a bugfix (gmm_swiglu_quant)
- 根因类别：动态输入API调用错误 + 输入索引错误
- 涉及文件：gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_tiling.cpp, .h
- 缺陷描述：(1)groupList是动态输入但用GetInputTensor获取(应为GetDynamicInputTensor)(2)GROUPLIST_INDEX=4错误，实际为第5个索引=5
- 修复模式：改用GetDynamicInputTensor(GROUPLIST_INDEX, 0) + 修正索引值
- 可审查性：高
- 审查规则建议：动态输入必须使用GetDynamicInputTensor；算子输入索引增减时必须同步更新所有常量定义

### 770a5110 fix:instruction_using_err
- 根因类别：错误include导致指令冲突
- 涉及文件：attention/incre_flash_attention/op_kernel/unpad_paged_attention_decoder.h
- 缺陷描述：错误include了l0c_to_l1_iterator.h，该头文件中的指令与当前kernel使用的指令冲突，导致硬件指令使用错误
- 修复模式：移除不需要的include
- 可审查性：中
- 审查规则建议：kernel头文件include应最小化，仅包含实际使用的iterator/buffer头文件

### a2c573d1 l1 buffer bugfix
- 根因类别：L1 buffer分配条件不完整
- 涉及文件：attention/common/op_kernel/arch-310/flash_attention_score_block_cube.h
- 缺陷描述：IterateBmm2中L1 splitN路径的条件仅检查dVTemplateType>256，但当dV<=256&&D>256时也需要L1 splitN路径(因D维度超过L1容量)。遗漏dTemplateType>256检查导致L1踩踏和精度错误
- 修复模式：条件增加 || (uint32_t)dTemplateType > 256
- 可审查性：高
- 审查规则建议：L1/L0 buffer容量约束条件必须同时考虑所有相关维度(D/Dv/S1/S2)；buffer policy的if条件应与allocate时的条件保持一致

### 8fb5839c 修改低比特通信卡死以及多轮场景卡死问题
- 根因类别：HCCL Wait次数不足 + 通信操作顺序错误
- 涉及文件：mc2/matmul_all_reduce/op_kernel/arch35/matmul_all_reduce_base.h, matmul_all_reduce_quant_comm_int8.h
- 缺陷描述：(1)HcclFinalize中Wait只调用一次但应调用tileCnt/tailCnt次(多tile场景每次通信都需要Wait)(2)低比特INT8量化通信中余数块的AllGather在ReduceScatter之前下发，违反了通信依赖顺序，导致卡死
- 修复模式：Wait改为for循环调用tileCnt/tailCnt次 + 交换AllGather和ReduceScatter的下发顺序(先RS后AG)
- 可审查性：高
- 审查规则建议：HCCL Notify/Wait必须成对且次数匹配(每次Notify对应一次Wait)；通信操作下发顺序必须匹配依赖关系(ReduceScatter产出 -> AllGather分发)

### f04fdf3c fix moeinitroutingv2 310p
- 根因类别：310P平台对齐要求差异 + 多核offset计算错误
- 涉及文件：moe/moe_init_routing_v2/op_kernel/moe_v2_sort_multi_core.h, moe_v2_sort_one_core.h
- 缺陷描述：(1)InitGlobalMemory在310P(__CCE_AICORE__==200)上要求对齐长度，但传入了未对齐的currentCoreExpert/expertNum(2)多核排序中workspace offset使用blockIdx*sortTotalLength但310P上perCoreOffset与sortTotalLength不同，导致workspace地址踩踏
- 修复模式：310P条件编译增加Align()对齐 + 使用perCoreOffset替代sortTotalLength作为workspace偏移步长
- 可审查性：中
- 审查规则建议：310P(aicore200)平台的GM初始化长度必须对齐；多核workspace offset应从tiling获取而非本地计算

### 50fc6a6e fix ffn/swin_transformer_ln_qkv_quant ut bugs
- 根因类别：struct声明顺序导致编译依赖错误
- 涉及文件：ffn/swin_transformer_ln_qkv_quant/op_host/swin_transformer_ln_qkv_quant_tiling.cpp, .h
- 缺陷描述：SwinTransformerLnQkvQuantCompileInfo结构体定义在tiling.cpp中REGISTER_TILING_DATA_CLASS宏之前，但UT需要在tiling.h中可见该结构体。声明位置不当导致UT编译失败
- 修复模式：将struct声明移至tiling.h中REGISTER_TILING_DATA_CLASS之后
- 可审查性：中
- 审查规则建议：编译信息结构体应声明在头文件中；REGISTER_TILING_DATA_CLASS依赖的类型和依赖它的类型之间注意声明顺序

### 3bdea9b3 FAG BN2/BN/B/SAMEAB/S1S2 Template BugFix
- 根因类别：TilingData未reset导致脏数据 + tilingKey参数位错误
- 涉及文件：attention/flash_attention_score_grad/op_host/..._tiling_bngs1s2_b.cpp, _ngs1s2_bn.cpp, _s1s2_bn2.cpp/.h, _s1s2_bn2gs1s2.h, _s1s2_bn2gs1s2_sab.h, op_kernel/flash_attention_score_grad_tiling.h
- 缺陷描述：(1)5个FAG tiling class的构造函数未调用td_->reset()，导致TilingData中残留脏数据(前次tiling结果污染当前计算)(2)GET_TPL_TILING_KEY中deterministic参数位传入context_->GetDeterministic()==1但应为0，导致tilingKey与kernel不匹配(3)4个TilingData class缺少reset()方法定义
- 修复模式：所有构造函数增加reset()调用 + GET_TPL_TILING_KEY的deterministic位改为0 + 补充reset()方法
- 可审查性：高
- 审查规则建议：TilingData类必须在构造时reset；GET_TPL_TILING_KEY的每个参数位必须与kernel侧TILING_KEY_IS的定义一一对应

### d6e9684a change1 bug修复：tnd场景增加长度校验
- 根因类别：输入长度校验缺失导致数组越界
- 涉及文件：attention/common/op_host/arch-310/flash_attention_score_tiling_regbase.cpp
- 缺陷描述：GetActualSeqLenData函数从tensor读取actualSeqLen数据写入固定大小数组(MAX_VAR_LEN_SEQ_LEN=4096)，但未校验输入tensor的dim(0)是否超过4096，超长输入会导致栈/堆数组越界写入
- 修复模式：增加seqLen>MAX_VAR_LEN_SEQ_LEN的边界检查 + 函数返回值改为bool用于传播错误
- 可审查性：高
- 审查规则建议：从tensor读取变长数据写入固定大小buffer前必须校验长度上界；函数应通过返回值传播校验失败而非静默继续

### eeec4dcd MoeFinalizeRoutingV2 david fix
- 根因类别：头文件guard名冲突 + kernel侧缺少tiling头文件
- 涉及文件：moe/moe_finalize_routing_v2/op_host/moe_finalize_routing_v2_tiling_arch35.h, op_kernel/arch35/moe_finalize_routing_v2_tiling_arch35.h(新增), op_kernel/moe_finalize_routing_v2_apt.cpp
- 缺陷描述：(1)op_host侧头文件guard宏名为_APT_H_但文件已改名为_arch35.h，guard名不匹配(2)kernel侧include该tiling头文件但仅op_host侧有定义，kernel目录下缺少对应文件副本导致编译失败(3)kernel include路径需更新为arch35子目录
- 修复模式：修正guard宏名 + 在op_kernel/arch35/下创建头文件副本 + 更新include路径
- 可审查性：高
- 审查规则建议：host/kernel共享的tiling数据结构头文件必须在两侧都可访问；文件重命名后检查header guard是否同步更新

### a3fd737d fix moe_finalize_routing_v2 arch35
- 根因类别：文件重命名后include路径未同步
- 涉及文件：moe/moe_finalize_routing_v2/op_host/moe_finalize_routing_v2_tiling_arch35.cpp, tiling_apt.h->tiling_arch35.h, op_kernel/moe_finalize_routing_v2_apt.cpp
- 缺陷描述：tiling头文件从_tiling_apt.h重命名为_tiling_arch35.h，但host侧tiling.cpp和kernel侧apt.cpp中的#include路径未同步更新，导致编译找不到头文件
- 修复模式：rename文件 + 更新所有include引用
- 可审查性：高
- 审查规则建议：文件重命名必须grep全部include引用并同步更新；CI应检测include路径指向不存在的文件

### 59931cf2 fix gmmSwigluQuant KLimit
- 根因类别：参数校验常量未区分场景
- 涉及文件：gmm/grouped_matmul_swiglu_quant/op_host/op_api/aclnn_grouped_matmul_swiglu_quant.cpp
- 缺陷描述：A8W8和A8W4两种量化模式共用同一个K_LIMIT=65536，但A8W4场景K维度上限实际应为20000。导致A8W4场景传入K在20000~65535范围时不会被拦截，触发计算异常或精度问题
- 修复模式：将统一常量拆分为K_LIMIT_A8W8=65536和K_LIMIT_A8W4=20000，各自的CheckInputOutShape使用对应常量
- 可审查性：高
- 审查规则建议：同一校验常量被多个不同模式使用时需检查是否对所有模式都适用

### 20f1d619 修复lse最大值&&行无效问题
- 根因类别：多缺陷组合——LSE初始值错误+行无效判断条件缺失+token越界未截断
- 涉及文件：attention/common/op_host/arch32/fia_tiling_nonquant.cpp, attention/common/op_kernel/arch32/fia_block_vec_nonquant.h等11个文件
- 缺陷描述：(1)LSE初始值在isOldIfaGqaFlag时用-INFINITY应为+INFINITY，导致GQA场景LSE计算错误；(2)DealInvalidMaskRows缺少attenMaskFlag条件判断，无mask时也执行行无效处理；(3)preToken/nextToken未clamp截断直接计算，超出序列长度致越界；(4)HighPerformance模式类型推导错误跳过Cast操作
- 修复模式：统一用INFINITY+添加条件守卫+新增GetSafeActToken做clamp+删除HighPerformance模式
- 可审查性：低
- 审查规则建议：softmax/logsumexp初始值必须为+INFINITY；涉及mask操作的函数先检查mask是否启用；用户输入token偏移量需clamp到有效范围

### 9f3d4950 NsaCompressAttentionInfer actualqseqlen check fix
- 根因类别：空指针检查层级错误
- 涉及文件：attention/nsa_compress_attention_infer/op_host/nsa_compress_attention_infer_tiling.cpp
- 缺陷描述：检查tensor!=nullptr但tensor对象本身始终非空(空tensor对象)，真正需检查GetData<int64_t>()!=nullptr。条件永远为true导致空数据指针解引用
- 修复模式：将条件从tensor!=nullptr改为tensor->GetData<int64_t>()!=nullptr
- 可审查性：高
- 审查规则建议：可选输入tensor的空值检查应验证数据指针GetData()而非tensor对象本身

### 6ee90aa7 fix-a8w4
- 根因类别：条件判断缺失+DataCopy长度未对齐
- 涉及文件：gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp, op_kernel/grouped_matmul_antiquant_a8w4_pre.h
- 缺陷描述：(1)isMSD条件判断缺少isPerchannel==false前置条件，per-group量化场景错误走MSD路径；(2)BF16的DataCopy长度未对齐16字节，导致硬件异常或数据截断
- 修复模式：增加isPerchannel条件短路+DataCopy长度16字节向上对齐
- 可审查性：高
- 审查规则建议：DataCopy长度参数必须按硬件要求对齐(16/32字节)；量化模式分支条件需验证per-channel和per-group场景均正确覆盖

### 2ede659b fix sink lse
- 根因类别：计算时序错误(sink token的LSE贡献在错误阶段计算)
- 涉及文件：attention/prompt_flash_attention/op_kernel/prompt_flash_attention_s1s2_bns1_mla_baseapi.h
- 缺陷描述：sink token的softmax贡献(exp(sink-max)累加到sum)放在BMM2之后的Brcb阶段处理，此时buffer已变换，offset语义不匹配。应在LSE计算之前、Brcb之前累加
- 修复模式：将sink处理逻辑从BMM2后移至LSE计算前，使用正确的s1RealSize和未变换buffer
- 可审查性：中
- 审查规则建议：softmax归一化因子更新必须在所有贡献项累加完成后再做最终变换；修改pipeline时序需验证所有buffer offset语义一致

### c2830def 修复custom包example找不到头文件问题
- 根因类别：构建脚本头文件搜索路径缺失
- 涉及文件：build.sh
- 缺陷描述：mc2目录下example编译缺少EAGER_INCLUDE_OPP_ACLNNOP_PATH头文件路径和-lpthread -lhccl链接库，导致mc2 example编译失败
- 修复模式：build.sh中判断example属于mc2目录时追加特有include路径和链接库
- 可审查性：高
- 审查规则建议：新增算子模块时确认example编译依赖在构建脚本中正确配置

### 82dbcd14 Antiquant TND bug fix
- 根因类别：多缺陷——TND布局batchSize未赋值+缺少校验+copy-paste条件重复
- 涉及文件：attention/incre_flash_attention/regbase/ophost/incre_flash_attention_tiling_v2.cpp
- 缺陷描述：(1)TND布局ProcessBaseTensors中batchSize_未从actualLenQDims_赋值，后续使用未初始化值；(2)TND缺少actualLenQDims_!=actualLenDims_校验致tiling越界；(3)GenTilingKey中else-if分支重复判断BSH_BSND(与第一个if相同)，TND永远匹配不到layoutVal=NUM2，tilingKey错误走入错误kernel分支
- 修复模式：添加batchSize_赋值+Q/KV长度一致性校验+修正重复条件为TND枚举
- 可审查性：高
- 审查规则建议：if-else-if链条件不应出现重复(重复=不可达分支)；新增layout类型需检查所有switch/if-else覆盖完整

### 02505c92 revert unquant mmv3 tilingkey
- 根因类别：tilingKey常量值被错误修改导致host/kernel不匹配
- 涉及文件：mc2/matmul_all_reduce/op_host/op_tiling/matmul_all_reduce_tiling_910.cpp, op_kernel/arch31/matmul_all_reduce_unquant_310.h等4个文件
- 缺陷描述：之前提交将tilingKey从标准大数值格式改成短值格式，导致host侧生成的tilingKey与kernel侧TILING_KEY_IS()判断不匹配，算子走入错误分支或无法匹配
- 修复模式：revert回正确的tilingKey常量值
- 可审查性：中
- 审查规则建议：TILING_KEY_IS()中的常量值必须在host侧tiling代码中有对应生成逻辑；tilingKey变更必须host/kernel两侧同步

### 3be74a64 bugfix: add cmake file context
- 根因类别：构建配置文件为空
- 涉及文件：gmm/grouped_matmul_finalize_routing/CMakeLists.txt
- 缺陷描述：CMakeLists.txt是空文件，导致该算子整体无法编译
- 修复模式：补齐标准的递归子目录扫描逻辑
- 可审查性：高
- 审查规则建议：CI检查新增算子目录的CMakeLists.txt是否为空文件

### 0de65dd4 fix
- 根因类别：平台兼容性校验逻辑过严
- 涉及文件：gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul.cpp
- 缺陷描述：grouped_matmul对输入内轴维度<=65535的限制不适用于ASCEND910_95平台(支持更大K值)，原代码对所有平台统一限制导致910_95上合法大K输入被错误拒绝
- 修复模式：增加平台版本判断跳过不适用校验
- 可审查性：中
- 审查规则建议：参数校验含硬编码上限时需检查是否区分不同硬件平台能力

### 1c680428 [bugfix] NsaCompress算子增加异常用例报错
- 根因类别：输入参数校验不足
- 涉及文件：attention/nsa_compress/op_host/nsa_compress_tiling.cpp
- 缺陷描述：(1)未校验input.shape[2]是16的倍数(对齐要求)；(2)actseqlenType校验==1只拒绝值1，实际只支持0，其他非0值也异常，应改!=0；(3)未校验actSeqLenOptional值的前缀和格式和总量一致性
- 修复模式：增强CheckParams校验逻辑，白名单方式拒绝非法输入
- 可审查性：高
- 审查规则建议：CheckParams覆盖所有shape约束；条件校验用白名单(!=valid)非黑名单(==invalid)；前缀和语义输入校验单调性和总量一致

### b69e1470 fix unquant tilingkey bug
- 根因类别：tiling key常量值错误(quant/unquant编码格式混淆)
- 涉及文件：mc2/matmul_all_reduce/op_kernel/arch31/matmul_all_reduce_unquant_310.h
- 缺陷描述：unquant场景310P架构kernel中TILING_KEY_IS使用了带量化前缀的tiling key值(10000000000000002000UL)，应为无前缀的2000UL/67536UL，导致非量化场景分支无法匹配
- 修复模式：修正4处TILING_KEY_IS宏参数为正确的无前缀值
- 可审查性：高
- 审查规则建议：同一语义的tiling key在不同架构/场景文件中值必须一致；quant与unquant路径的tiling key编码格式必须正确区分

### 65acaf5e ms allgather 图模式问题单修改
- 根因类别：硬编码属性索引无法适配多版本算子
- 涉及文件：mc2/all_gather_matmul/op_host/all_gather_matmul_infershape.cpp, mc2/common/inc/mc2_common_infershape.h等
- 缺陷描述：InferShape函数中gather_out属性索引硬编码为GATHER_OUT=6，V2版本中该属性位置变为index 8，图模式下V2算子从错误位置读取isGatherOut标志，infer shape结果错误
- 修复模式：参数化硬编码索引，通过调用方传入不同版本的正确偏移
- 可审查性：高
- 审查规则建议：同一算子多版本(V1/V2)且属性列表不同时，检查通过索引获取属性的位置是否与版本定义一致

### acab4f8a 修正examples soc版本不兼容时无拦截问题
- 根因类别：缺少运行时环境校验
- 涉及文件：build.sh, 11个examples目录下test_aclnn_*.cpp
- 缺陷描述：mc2算子example仅注释说明需A3硬件，代码无实际SoC版本检查，非A3上运行无拦截
- 修复模式：build.sh传入-DSOC_VERSION编译宏，example入口添加strcmp版本检查
- 可审查性：高
- 审查规则建议：有硬件限制注释的代码应有对应运行时校验

### 74124c57 fix the failure of outputting precision in non-quantized case
- 根因类别：splitD场景多处逻辑缺陷(workspace偏移+dSize对齐+模板选择)
- 涉及文件：attention/common/op_kernel/arch-310/flash_attention_score_block_vec_base.h等4个文件
- 缺陷描述：非量化FA 310架构splitD模式多bug：(1)workspace指针未递增导致buffer覆盖；(2)FD模式workspace基地址未考虑preload和核数致多核重叠；(3)使用编译期常量dVTemplateType应用运行时constInfo.dBasicBlock；(4)调用固定模板函数应调可变参数版本；(5)stride/padding用错误dSize
- 修复模式：补齐指针偏移+修正workspace计算+替换编译期常量为运行时值
- 可审查性：低
- 审查规则建议：splitD/splitK分块模式下检查所有维度对齐常量是否切换到分块值；workspace分配后指针必须正确递增

### aa0feec3 fix unquant mmAllReduce tilingkey
- 根因类别：tiling key常量值错误(与b69e1470同根因)
- 涉及文件：mc2/matmul_all_reduce/op_kernel/matmul_all_reduce.cpp
- 缺陷描述：kernel入口分发逻辑中310P非量化NZ格式分支同样使用错误的带量化前缀tiling key值，与b69e1470配对修复
- 修复模式：修正常量值确保入口分发和实现使用一致的tiling key
- 可审查性：高
- 审查规则建议：tiling key在host生成端/kernel入口/kernel实现三处必须一致；用宏或枚举统一定义避免手写魔法数

### 63b28534 modify GQA antiquant 8 buffer and fix bugs for gqa antiquant pertoken
- 根因类别：GQA antiquant pertoken多处计算逻辑错误
- 涉及文件：attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp, incre_flash_attention_tiling.cpp, op_kernel/incre_flash_attention_preload_dd.h
- 缺陷描述：(1)算子类型列表缺2组BF16 antiquant pertoken数据类型组合；(2)Fixpipe未正确区分head/tail/overlap三种情况致AtomicAdd范围和偏移错误；(3)循环嵌套顺序导致数据重复加载；(4)CopyInMm2AToL1参数传递用mCopyRowCount而非mActCopyRowCount且地址偏移多余
- 修复模式：重构MSD antiquant pertoken的MM1/MM2计算流程，修正循环嵌套和参数传递
- 可审查性：低
- 审查规则建议：MSD多scale场景检查Fixpipe按段使用不同scale；L1/GM地址偏移检查是否有多余或遗漏乘积项

### 0f4daa42 fix fia tilingkey error
- 根因类别：头文件缺失导致宏定义未引入
- 涉及文件：attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score.cpp, fused_infer_attention_score_tilingkey.h
- 缺陷描述：.cpp缺少对tilingkey.h的#include，tilingkey宏定义未被引入，导致编译失败或宏值未定义选错kernel分支
- 修复模式：添加缺失的#include
- 可审查性：高
- 审查规则建议：检查.cpp中使用的宏/常量是否有对应#include

### 0267ec6c 修复kvsame后量化UB overflow问题
- 根因类别：初始化时序错误导致UB overflow
- 涉及文件：attention/common/op_kernel/arch-310/flash_attention_kvsame_bn2gs1s2.h
- 缺陷描述：输出初始化(InitOutput)在InitInput内部执行，此时tilingData/blockIdx等上下文尚未设置，在错误上下文下执行SetGlobalBuffer导致写入超出UB边界
- 修复模式：将上下文设置提前，输出buffer初始化从InitInput中解耦为独立的MlaInitOutput函数
- 可审查性：中
- 审查规则建议：Init函数中成员变量的使用必须在赋值之后；SetGlobalBuffer调用需确保依赖的上下文变量已初始化

### eb29dd4f 修复pfa tnd模板合轴精度问题
- 根因类别：向量指令步长参数遗漏乘数因子
- 涉及文件：attention/prompt_flash_attention/op_kernel/prompt_flash_attention_s1s2_bns1_mla_baseapi.h
- 缺陷描述：TND模板MLA路径中Mul指令的repeat stride参数使用vec2S1RealSize但遗漏gBaseSize因子，步长不正确导致每次repeat只处理1/gBaseSize数据量，精度错误
- 修复模式：补全向量指令步长参数的维度乘积因子
- 可审查性：中
- 审查规则建议：向量指令stride/block参数需覆盖所有合并轴维度；有group维度时确认参数包含group因子

### 4ddeba6d fix moe_init_routing_v2 sync bug: add vf to scalar pipe barrier
- 根因类别：流水线同步屏障缺失导致数据竞争
- 涉及文件：moe/moe_init_routing_v2/op_kernel/arch35/moe_v2_expert_token_out_simt.h
- 缺陷描述：最后一个core处理expertCumsum时直接从V pipe计算结果地址读取数据，缺少V->S同步屏障，Scalar pipe可能在Vector pipe未完成写入时读到脏数据
- 修复模式：在跨pipe数据依赖处添加SetWaitFlag<HardEvent::V_S>
- 可审查性：低
- 审查规则建议：跨pipe数据读写(V写->S读等)检查消费端前是否有对应SetWaitFlag/PipeBarrier；条件分支中的跨pipe访问容易遗漏

### 6a3a65d0 dispatch combine rdv冒烟失败问题修复
- 根因类别：框架注册宏缺失
- 涉及文件：mc2下6个*_gen_task.cpp(distribute_barrier/combine/dispatch等)
- 缺陷描述：算子只注册了IMPL_OP_CT但缺少REGISTER_EXT_TASK_TYPE宏声明外部任务类型为kAicoreTask，框架无法识别任务类型，运行失败
- 修复模式：补全REGISTER_EXT_TASK_TYPE注册和对应头文件include
- 可审查性：高
- 审查规则建议：凡出现IMPL_OP_CT(X)的文件必须同时存在REGISTER_EXT_TASK_TYPE(X, ...)

### 2c082d47 fix tiling key error for gqa non-quant d512+pse+pa
- 根因类别：tiling key编码值错误/配置映射不一致
- 涉及文件：attention/prompt_flash_attention/regbase/opkernel/prompt_flash_attention_entry_regbase.h
- 缺陷描述：GQA non-quant D512+PSE+PA场景多处tiling key数值编码错误(如D256场景1003103应为1003313)，删除错误重复key注册并新增缺失分支。运行时key无法匹配到正确kernel实现
- 修复模式：修正声明和实现中key编码值统一，确保key与模板参数一致
- 可审查性：低
- 审查规则建议：tiling key编码应有结构化生成/校验机制而非手工硬编码；编译期或UT校验每个TILING_KEY_IS有且仅有一个对应实现

### 2315af18 fix elastic && buf size
- 根因类别：初始化顺序错误+buffer大小不足+buffer复用冲突
- 涉及文件：mc2/moe_distribute_combine_add_rms_norm/op_kernel/*.h, moe_distribute_combine_v2/*.h, moe_distribute_dispatch_v2/*.h
- 缺陷描述：(1)epWorldSize_/moeExpertPerRankNum_仅在非elastic分支赋值，elastic模式下未初始化；(2)tokenBuf_大小小于实际使用需要；(3)maskStrideTensor_和maskTensor复用同一expertScalesBuf_导致数据覆盖；(4)除数用被elastic修改的值应用原始值；(5)同步状态重置的流水线hazard
- 修复模式：防御性初始化+buffer容量修正+消除内存复用冲突+消除pipeline hazard
- 可审查性：中
- 审查规则建议：条件分支中变量赋值检查所有路径是否都有初始化；同一TBuf获取多个LocalTensor时检查写冲突

### 32d13195 修复tilingKey
- 根因类别：tiling key值与tiling侧生成不匹配
- 涉及文件：mc2/matmul_all_reduce/op_kernel/matmul_all_reduce.cpp
- 缺陷描述：int8通信场景tiling key编号错误(2/3/18/19应为10/11/26/27)，运行时无法进入正确计算分支
- 修复模式：修正常量值与tiling生成侧对齐
- 可审查性：中
- 审查规则建议：tiling key应通过共享常量/枚举定义，避免两处分别硬编码

### 43c7085a 修正GMM全量化的定义宏
- 根因类别：预处理宏条件过度宽泛(量化类型交叉匹配)
- 涉及文件：gmm/grouped_matmul/op_kernel/grouped_matmul_utils.h
- 缺陷描述：V310_GMM_QUANT宏对X和WEIGHT类型分别独立判断是否属于量化集合，导致X=INT8+WEIGHT=FP8_E4M3FN等跨类型组合也命中全量化宏，但硬件不支持混合量化
- 修复模式：收紧条件为按类型族分组的AND配对(int8配int8、fp8互配等)
- 可审查性：高
- 审查规则建议：复杂#if条件应有注释列出合法配对矩阵；逐一验证覆盖所有合法组合且排除非法组合

### d2696747 built-in soc修复
- 根因类别：构建/打包逻辑缺陷
- 涉及文件：cmake/package.cmake, common/src/tiling_sink/CMakeLists.txt
- 缺陷描述：install(FILES)引用可能不存在的.so文件，COMPUTE_UNIT变量名未同步更新为ASCEND_COMPUTE_UNIT；tiling_sink缺少源文件存在性检查
- 修复模式：增加if(EXISTS)守卫+变量名修正
- 可审查性：中
- 审查规则建议：CMake中install(FILES)应检查文件存在性；变量重命名需全局搜索所有引用点

### 2b7893bd fix fia def error
- 根因类别：算子定义中输入参数顺序错误
- 涉及文件：attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp
- 缺陷描述：learnable_sink输入注册顺序与算子原型接口不一致，定义在q_start_idx/kv_start_idx之后但应在之前，导致tensor绑定到错误输入槽位。aicore_config_95缺少learnable_sink定义
- 修复模式：调整Input注册声明顺序与原型一致+补全缺失配置段
- 可审查性：高
- 审查规则建议：def文件Input/Output注册顺序必须与算子原型参数顺序严格一致；多个aicore_config段输入定义保持同步

### 48d34eec fix 310p moeinitroutingv2
- 根因类别：平台兼容性缺陷(310P多个子问题)
- 涉及文件：moe/moe_init_routing_v2/op_host/moe_init_routing_v2_tiling.cpp, op_kernel/moe_v2_sort_multi_core.h, moe_v2_sort_one_core.h
- 缺陷描述：(1)tiling层缺少310P平台cols不整除32的校验；(2)ResetIO未重置srcWsIndex=0致多次调用索引脏值；(3)DataCopyCustom缺模板参数；(4)同步初始化代码放在Process()与SyncAll()时序冲突应在Init()
- 修复模式：增加平台校验+状态变量重置+模板参数补全+初始化时序修正
- 可审查性：低
- 审查规则建议：状态变量在复用场景(ResetIO)必须全量重置；同步原语初始化数据应在Init阶段完成

### b26b2487 fix norm
- 根因类别：变量名拼写错误(typo)
- 涉及文件：mc2/3rd/norm_common/op_kernel/reduce_common.h
- 缺陷描述：ReduceSumHalfInterval中WholeReduceSum的目标tensor误写为dstLocal，正确应为dst_local(下划线命名)，写入错误位置致归约结果丢失
- 修复模式：变量名修正dstLocal->dst_local
- 可审查性：高
- 审查规则建议：同一函数内变量命名风格应统一；静态分析检测变量shadowing或使用未声明变量

### 04dbad9e fix moe_init_routing_v2 init gm value bug
- 根因类别：GM直写初始化不安全+条件判断缺失
- 涉及文件：moe/moe_init_routing_v2/op_kernel/arch35/moe_v2_common.h, moe_v2_gather_out_for_simt.h
- 缺陷描述：InitOutput直接往GM写零值未经UB中转可能不正确；CopyOutZero缺少progress==0约束导致多次循环重复执行首次填零逻辑；BUFFER_NUM=2在此场景导致资源冲突
- 修复模式：新增InitGmValue模板函数通过UB中转分块DMA搬运到GM+增加progress条件约束+MTE3到MTE2同步等待
- 可审查性：中
- 审查规则建议：检查所有直接对GlobalTensor写入的地方是否经过LocalTensor中转；多核循环内条件判断检查是否遗漏iteration约束

### e25a2bd4 fix aclnn
- 根因类别：CMake target名称错误
- 涉及文件：moe/moe_token_permute_with_ep/op_host/CMakeLists.txt等4个文件
- 缺陷描述：target_sources目标名写op_host_aclnnInner应为op_host_aclnn，BUILD_OPEN_PROJECT模式下算子def.cpp链到错误target致编译失败或对外接口缺少注册
- 修复模式：修正CMake target名称
- 可审查性：高
- 审查规则建议：target_sources引用的target名必须与项目定义一致；CI覆盖不同构建配置

### b18a64b4 fix bug for start_idx
- 根因类别：条件分支缺失/场景适配不完整
- 涉及文件：attention/common/op_host/arch-310/flash_attention_score_tiling_regbase.cpp
- 缺陷描述：SetQKVStartIdx无条件从tensor读取start_idx，但该参数仅在layout=TND且sparseType=BAND_LEFT_UP_CAUSAL时有意义，其他场景读到的值是无意义的/未初始化的，污染tiling参数致计算错误
- 修复模式：增加layout和sparseType条件守卫+非适用分支显式设零
- 可审查性：中
- 审查规则建议：可选输入参数的读取必须有场景条件守卫；tiling参数在非适用分支应显式清零

### c7cb0d84 修复matmul_all_reduce_add_rms_norm编译问题
- 根因类别：C++语法错误(非静态成员不能constexpr)
- 涉及文件：mc2/matmul_all_reduce_add_rms_norm/op_kernel/add_rms_norm_merge_n.h
- 缺陷描述：DOUBLE_BUFFER_QUEUE声明为class非静态constexpr成员变量但用于模板参数需编译期常量，C++中不合法致编译失败
- 修复模式：提升为文件级constexpr常量
- 可审查性：高
- 审查规则建议：constexpr用于模板参数不应声明为类非静态成员

### beadccf7 built-in update修复
- 根因类别：shell脚本语法错误+资源泄漏
- 涉及文件：scripts/package/common/sh/multi_version.inc
- 缺陷描述：(1)__notify_uninstall_managers函数if缺少fi闭合导致后续函数定义被包含在上一函数体内；(2)multi_version_uninstall正常流程未清理mktemp临时文件
- 修复模式：内联辅助函数+补全fi+正常路径增加临时文件清理
- 可审查性：中
- 审查规则建议：shell脚本检测if/fi配对(shellcheck)；mktemp创建的临时文件在所有退出路径有清理

### 799bf086 编译告警修改(含逻辑缺陷)
- 根因类别：运算符优先级歧义+printf格式字符串类型不匹配
- 涉及文件：attention/common/op_host/arch-310/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp
- 缺陷描述：(1)||与&&混用无括号导致运算符优先级歧义；(2)float类型变量用%ld格式说明符是UB。逻辑修复混在大量告警清理中
- 修复模式：添加括号明确优先级+修正printf格式符
- 可审查性：低
- 审查规则建议：||和&&混用必须加显式括号；printf格式字符串类型检查(-Wformat)；告警清理和逻辑修复应分开提交

### 900f62ed Address potential overflow when calculating GM addresses
- 根因类别：整数溢出(int32乘法溢出致地址计算错误)
- 涉及文件：gmm/grouped_matmul/op_kernel/grouped_matmul_autotiling_a8w4.h
- 缺陷描述：GM地址计算中使用int32_t/uint32_t乘法，大shape下溢出导致地址错误引发精度问题
- 修复模式：将参与GM地址计算的变量从int32提升为int64_t/uint64_t
- 可审查性：高
- 审查规则建议：地址/偏移量计算中使用32位整数乘法的模式需检测；shape维度相乘的变量要求int64_t

### 1fb9ed7c fix no exist opapi_obj
- 根因类别：CMake target不存在导致编译失败
- 涉及文件：cmake/obj_func.cmake, tools/tests/ut/op_api/CMakeLists.txt
- 缺陷描述：算子无aclnn源文件时opapi_obj target不被创建，但UT通过$<TARGET_OBJECTS:>引用该target致configure失败
- 修复模式：else分支创建空object library确保target始终存在
- 可审查性：高
- 审查规则建议：$<TARGET_OBJECTS:>引用的target应在所有条件分支下都被定义

### ab199ac6 DTS2025091503928问题单修改
- 根因类别：空指针解引用+无符号整数比较逻辑错误+变量遮蔽+过严校验
- 涉及文件：mc2/moe_distribute_combine/、mc2/moe_distribute_dispatch/等十余个文件
- 缺陷描述：(1)多个InferShape入口未校验context是否nullptr直接调用GetNodeName致空指针解引用；(2)uint32_t类型变量用<=0判断unsigned永远不<0；(3)GetStorageShape/GetInputDesc/GetAttrs返回值未null check直接解引用；(4)同作用域api_ret变量名重复遮蔽；(5)过严的DISPATCH_STATUS_MAX_SUPPORT_NUM限制
- 修复模式：前置空指针校验+修正无符号比较+消除变量遮蔽+放宽过严限制
- 可审查性：高
- 审查规则建议：InferShape/Tiling入口强制context null check；检测unsigned与0的<=比较；检测同作用域变量名遮蔽

### 454bad76 fix vcadd raw instruction
- 根因类别：裸指令使用导致可移植性和正确性问题
- 涉及文件：mc2/moe_distribute_combine_add_rms_norm/op_kernel/moe_distribute_combine_add_rms_norm.h
- 缺陷描述：ReduceSumCustom/ReduceSumFP32手写使用vcadd裸指令，针对不同架构用不同参数形式，既脆弱又冗余
- 修复模式：删除47行自定义实现，替换为框架标准ReduceSum API
- 可审查性：中
- 审查规则建议：检测vcadd等裸指令直接使用，建议替换为框架标准API

### 345a935d fix arch35 dir
- 根因类别：目录结构错误
- 涉及文件：moe/moe_token_permute_with_ep/下8个头文件
- 缺陷描述：arch35架构文件错误放在算子根目录/arch35/下，应在op_kernel/arch35/下，构建系统找不到头文件
- 修复模式：文件目录位置修正(rename到正确路径)
- 可审查性：高
- 审查规则建议：算子目录下arch*子目录应位于op_kernel子目录下而非算子根目录

### f884a231 fix lse when lse_flag is false
- 根因类别：输出shape赋值错误(常量值错误)
- 涉及文件：attention/fused_infer_attention_score/op_host/fused_infer_attention_score_infershape.cpp
- 缺陷描述：lse_flag为false时softmaxLseShape第0维设为NUM_1(1)应为NUM_0(0)，不需要输出LSE时shape应为空但推导出非空shape
- 修复模式：常量值修正1->0
- 可审查性：高
- 审查规则建议：禁用/关闭语义的flag对应输出shape/size应设为0或空

### e50a2683 fix：清理sc
- 根因类别：复合条件表达式+未初始化变量
- 涉及文件：gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp
- 缺陷描述：GetC0Size调用和nSize_%c0Size校验合并在一个OP_CHECK_IF用||连接，c0Size未初始化就可能参与%运算(UB)，且错误信息混淆
- 修复模式：拆分复合条件+变量初始化c0Size=0
- 可审查性：高
- 审查规则建议：OP_CHECK_IF中不应将"获取数据"和"校验数据"用||合并；变量在所有路径上初始化后才能使用

### 854b3e57 适配tiling下沉场景传入数据为nullptr shape不为0问题
- 根因类别：接口数据类型不匹配(aclIntArray无法表达有shape无data语义)
- 涉及文件：attention/fused_infer_attention_score/op_host/op_api/aclnn_fused_infer_attention_score_inner.cpp等4个文件
- 缺陷描述：tiling下沉需传递"有shape但data=nullptr"语义计算最大workspace，aclIntArray无法承载此语义，改用aclTensor
- 修复模式：数据类型替换aclIntArray->aclTensor+新增适配接口
- 可审查性：中
- 审查规则建议：接口参数需同时携带形状元信息和数据指针时检查容器类型能否表达nullptr场景

### efc02afe built-in 修复
- 根因类别：CMake链接配置缺失
- 涉及文件：cmake/symbol.cmake
- 缺陷描述：gen_ophost_symbol/gen_opgraph_symbol缺少intf_pub_cxx17编译接口和rt2_registry_static的whole-archive链接，built-in场景符号缺失
- 修复模式：CMake链接配置修正+调整对象文件包含方式
- 可审查性：低
- 审查规则建议：CMake变更需检查built-in和非built-in两种模式是否都正常链接

### 9c18d0ff fix buffer size
- 根因类别：buffer分配条件不完整(遗漏特殊专家分支)
- 涉及文件：mc2/moe_distribute_combine_add_rms_norm/op_kernel/*.h, moe_distribute_combine_v2/*.h
- 缺陷描述：activeMask buffer扩容只在isInputExpertMaskFlag_为true时触发，但特殊专家场景也使用mask buffer，此时flag为false导致buffer按较小size分配后续溢出
- 修复模式：扩容条件增加enableSpecialExpert_判断+提取公共条件变量
- 可审查性：高
- 审查规则建议：buffer分配条件应覆盖所有使用该buffer的代码路径；多处使用相同复合条件时检查一致性

### 9d43cc0a 回退cleancode
- 根因类别：宏定义作用域错误(do-while包裹导致变量不可见)
- 涉及文件：attention/flash_attention_score/op_kernel/flash_attention_score.cpp
- 缺陷描述：cleancode将COPY_TILING_DATA宏用do{...}while(0)包裹，导致宏内声明的tilingData等变量作用域被限制在块内，外部不可见
- 修复模式：回退do-while包裹恢复裸展开
- 可审查性：高
- 审查规则建议：含变量声明且需暴露给调用上下文的宏禁止使用do-while包裹

### c2681e44 built-in 修复
- 根因类别：构建打包遗漏文件
- 涉及文件：cmake/package.cmake
- 缺陷描述：built-in包SCRIPTS_FILES列表遗漏merge_binary_info_config.py
- 修复模式：在install文件列表中补充遗漏文件
- 可审查性：高
- 审查规则建议：新增脚本文件时检查CMake打包配置是否同步更新

### 843205ce built-in修复
- 根因类别：stub函数缺失+打包脚本遗漏
- 涉及文件：common/stub/op_tiling/tbe_tiling_api.cpp/.h, scripts/package/common/py/merge_binary_info_config.py
- 缺陷描述：stub库缺少GetTbeTiling重载函数的stub实现，built-in模式链接时未定义符号错误
- 修复模式：补充stub声明与实现+新增构建辅助脚本
- 可审查性：高
- 审查规则建议：新增API重载时同步检查stub库是否有对应空实现

### 88063c5c [FIA] fix tiling sink so version
- 根因类别：构建配置硬编码版本号
- 涉及文件：cmake/package.cmake, cmake/variables.cmake, common/CMakeLists.txt等
- 缺陷描述：tiling sink产出so文件名版本号硬编码v8.3，实际系统版本不同时install找不到正确so
- 修复模式：硬编码版本号替换为动态获取的SYS_VERSION变量
- 可审查性：高
- 审查规则建议：禁止CMake中硬编码版本号字符串；版本号应有唯一来源

### f50abe28 修复GMM伪量化静态图转置拦截异常问题
- 根因类别：逻辑校验过严(错误拦截合法场景)
- 涉及文件：gmm/grouped_matmul/op_host/grouped_matmul_infershape_weight_quant_checker.cpp/.h
- 缺陷描述：CheckTransposeValid对转置参数施加过严校验，静态图场景不正确拦截合法转置配置
- 修复模式：删除整个错误的校验函数及其调用
- 可审查性：高
- 审查规则建议：新增校验需验证在所有执行路径(动态图/静态图)下成立

### 5cb903f6 fix fuzz with special experts
- 根因类别：UB内存分配冗余+buffer复用错误
- 涉及文件：mc2/moe_distribute_combine_add_rms_norm/op_kernel/moe_distribute_combine_add_rms_norm.h
- 缺陷描述：mulBuf_独立分配28K UB空间但可与rowTmpFloatBuf_复用(两者不同时活跃)，special experts场景UB不足溢出
- 修复模式：消除冗余buffer通过复用释放UB空间
- 可审查性：中
- 审查规则建议：UB buffer变更需附带生命周期分析确认复用无冲突

### 137c5005 fix ub dispatchV2
- 根因类别：UB内存管理+数据状态同步+数组索引越界(多缺陷)
- 涉及文件：mc2/moe_distribute_combine_v2/op_kernel/*.h, moe_distribute_dispatch_v2/*.h
- 缺陷描述：(1)UB总量超192K上限未检查需降为单buffer；(2)DataCacheCleanAndInvalid直接操作GM有cache一致性问题应用DMA；(3)GatherMask后expertIdx索引语义变化致越界；(4)多buffer复用优化；(5)执行顺序调整避免覆盖
- 修复模式：UB总量动态检查降级+DMA替代直接GM操作+修正索引+buffer复用+顺序调整
- 可审查性：低
- 审查规则建议：UB分配应有总量检查机制；GM状态读写统一用DMA；GatherMask后的索引变换需注释标注语义

### 400d65c2 修复推理FA的build_in和custom
- 根因类别：CMake条件变量名错误+路径拼写错误
- 涉及文件：attention/fused_infer_attention_score/op_host/CMakeLists.txt等3个+cmake/custom_build.cmake+cmake/obj_func.cmake
- 缺陷描述：(1)编译条件用NOT BUILD_OPEN_PROJECT应为NOT BUILD_OPS_RTY_KERNEL；(2)路径ophost应为op_host；(3)编译过滤逻辑错误跳过应编译算子
- 修复模式：修正条件变量名+路径拼写+移除错误过滤逻辑
- 可审查性：高
- 审查规则建议：CMake路径字符串应有CI目录存在性验证

### 1dd7f008 liushuye修复
- 根因类别：条件分支被错误注释
- 涉及文件：mc2/all_gather_matmul/op_host/op_api/aclnn_all_gather_matmul.cpp
- 缺陷描述：910A5芯片路由到V2版本的代码被注释掉，走了旧通用路径
- 修复模式：恢复被注释代码+补充缺失的include
- 可审查性：高
- 审查规则建议：检测被注释掉的完整条件分支代码块

### 8d1ede74 fix custom build
- 根因类别：CMake链接声明缺失
- 涉及文件：cmake/custom_build.cmake
- 缺陷描述：cust_opmaster的target_link_libraries缺少对common_obj对象库的PUBLIC链接，编译出的库符号未定义
- 修复模式：补充TARGET_EXISTS条件式对象库链接
- 可审查性：中
- 审查规则建议：新增对象库target时检查所有消费方是否已包含依赖

### 48876875 fix ACLNNTYPE
- 根因类别：CMake target名称错误(与e25a2bd4相反方向)
- 涉及文件：moe/moe_finalize_routing_v2_grad/op_host/CMakeLists.txt
- 缺陷描述：target_sources写op_host_aclnn应为op_host_aclnnInner(该算子应注册到内部接口)
- 修复模式：修正target名称
- 可审查性：高
- 审查规则建议：target_sources的target名与算子ACLNNTYPE声明交叉验证

### 96fd1f5b 修复FA精度问题
- 根因类别：硬件同步事件管理缺陷(数据竞争)
- 涉及文件：attention/flash_attention_score/op_kernel/flash_attention_score_s1s2_bn2gs1_sab.h
- 缺陷描述：BMM2数据加载的MTE1/MTE2同步不足，循环外单事件不够用，循环内缺精细同步控制导致数据竞争精度问题
- 修复模式：引入双事件交替ping-pong同步+调整加载顺序+循环后等待最后事件
- 可审查性：低
- 审查规则建议：循环内SetFlag/WaitFlag的事件ID检查是否有复用冲突；AllocEventID与ReleaseEventID必须配对

### 449c0259 fix moe_gating_top_k
- 根因类别：头文件include路径错误
- 涉及文件：moe/moe_gating_top_k/op_kernel/arch35/moe_gating_top_k_regbase.h
- 缺陷描述：include "kernel_utils.h"不带路径前缀依赖搜索路径，某些构建配置下找不到
- 修复模式：改为显式相对路径../../inc/kernel_utils.h
- 可审查性：高
- 审查规则建议：非标准库头文件优先使用相对路径

### 6e95ccc2 bugfix::add_apply_rotary_pos_emb
- 根因类别：命名空间变更未同步更新
- 涉及文件：posembedding/apply_rotary_pos_emb/下8个arch35头文件
- 缺陷描述：工具函数命名空间从ops::迁移到Ops::Base::但调用方未更新，编译失败
- 修复模式：批量替换命名空间前缀+补充include
- 可审查性：高
- 审查规则建议：公共库命名空间重构时全局搜索旧命名空间确保全部更新

### 5e26f941 Fix build error with static graph and Fix performance issues with new GMM MSD kernel
- 根因类别：多缺陷混合(类型不匹配+性能bug+死代码)
- 涉及文件：gmm/grouped_matmul/op_kernel/grouped_matmul.cpp, grouped_matmul_autotiling_a8w4.h
- 缺陷描述：(1)const限定不匹配需const_cast；(2)Base_M公式用szMemUB应为UB_Size；(3)single_M<128未处理单AIV模式；(4)冗余SetL2CacheHint和SyncAll拖慢性能；(5)ScaleA UB buffer分配方式错误
- 修复模式：类型转换修正+公式修正+冗余代码删除+迭代器替换
- 可审查性：低
- 审查规则建议：UB内存分配公式变量校验是否使用"可用空间"而非"总空间"

### 04f68846 fix no opapi obj error
- 根因类别：CMake target依赖硬编码导致构建失败
- 涉及文件：cmake/custom_build.cmake
- 缺陷描述：target_link_libraries硬编码opapi_obj等target名，target不存在时CMake报错
- 修复模式：改为target_sources配合$<TARGET_EXISTS:>条件引用
- 可审查性：高
- 审查规则建议：CMake引用对象库target时用$<TARGET_EXISTS:>守卫

### 38606b2b bugfix: FA/FAG TND case need --cce-auto-sync=off
- 根因类别：编译选项配置错误(auto-sync与手动同步冲突)
- 涉及文件：attention/flash_attention_score/op_host/CMakeLists.txt, flash_attention_score_grad/op_host/CMakeLists.txt
- 缺陷描述：kernel已手动管理同步但编译选项设为--cce-auto-sync=on，编译器自动插入同步指令与手动同步冲突
- 修复模式：修正编译标志on->off+补充缺失的编译选项分支
- 可审查性：中
- 审查规则建议：kernel含手动SetFlag/WaitFlag时校验编译选项为--cce-auto-sync=off

### 876e9556 修复qgmm inplace add 不合理的日志问题
- 根因类别：校验缺失/输入校验不完整
- 涉及文件：gmm/quant_grouped_matmul_inplace_add/op_host/op_api/*.cpp, gmm/grouped_matmul/op_host/op_api/*.cpp
- 缺陷描述：MX量化场景缺少scale维度校验(IsMxQuantDim)，且未校验x1的M维度>0，非法输入直接进入计算逻辑
- 修复模式：新增IsMxQuantDim校验函数+mDim正数校验
- 可审查性：中
- 审查规则建议：校验逻辑覆盖所有量化模式维度约束；维度值应有>0正数校验

### e6933c4e fix buf conflict
- 根因类别：buffer冲突/内存复用错误
- 涉及文件：mc2/moe_distribute_combine_add_rms_norm/*.h, mc2/moe_distribute_combine_v2/*.h
- 缺陷描述：(1)expertIdsFloat使用mulBuf_与后续操作buffer生命周期重叠致数据覆盖；(2)maskGenerateTensor_在条件分支内初始化但分支外使用
- 修复模式：更换buffer来源避免复用冲突+变量初始化提升到条件分支外层
- 可审查性：低
- 审查规则建议：UB buffer的Get<>检查同一函数内是否多个LocalTensor引用同一buffer；条件分支内初始化但分支外使用的变量应报警

### a43c3a65 fix gmm dfx
- 根因类别：DFX校验被注释导致防御缺失
- 涉及文件：gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp
- 缺陷描述：约30处OP_CHECK_IF错误检查宏被注释掉(除零保护/空指针检查/参数范围校验等)，无效参数或异常状态无法捕获
- 修复模式：取消注释恢复错误检查
- 可审查性：高
- 审查规则建议：禁止在提交中注释掉OP_CHECK_IF/错误检查代码；静态扫描检测被注释的防御性检查

### 7080993c ifx conflict tilingdata
- 根因类别：类型名冲突/符号命名冲突
- 涉及文件：moe/moe_token_permute_with_ep/下多个文件, moe/moe_token_permute_with_ep_grad/下多个文件
- 缺陷描述：两个算子共享相同tiling data类型名，REGISTER_TILING_DATA_CLASS注册时符号冲突
- 修复模式：重命名为带EP/Grad后缀的专有名称
- 可审查性：高
- 审查规则建议：REGISTER_TILING_DATA_CLASS注册的类型名应全局唯一

### 6d9e65a5 combinearn fuzz ub越界修复; dispatchv2 tiling修改
- 根因类别：UB越界+校验逻辑不完整
- 涉及文件：mc2/moe_distribute_combine_add_rms_norm/*.h, mc2/moe_distribute_dispatch_v2/op_host/op_tiling/*.cpp
- 缺陷描述：(1)yBuf_独立分配14K可复用已有buffer，fuzz测试UB总量超限越界；(2)DispatchV2 tiling校验expertIds的k维度遗漏copyExpertNum和constExpertNum
- 修复模式：消除冗余buffer复用已有+补全校验条件
- 可审查性：中
- 审查规则建议：UB buffer分配需总量审计；多种expert类型的校验条件应覆盖全部类型

### f77eb57b nsa 训练 inc修复
- 根因类别：API接口不兼容/头文件路径变更
- 涉及文件：attention/nsa_compress/等10个文件, common/CMakeLists.txt
- 缺陷描述：inc头文件路径从error/ops_error.h改为err/ops_err.h，OPS_REPORT宏参数从context指针改为GetNodeName()字符串，未同步适配编译失败
- 修复模式：批量替换include路径和宏调用参数
- 可审查性：高
- 审查规则建议：头文件路径变更全仓搜索确保所有引用同步；宏API签名变更应通过编译CI拦截

### 3d603046 fix bug
- 根因类别：类型名冲突/符号命名冲突
- 涉及文件：moe/moe_token_permute_grad/op_host/*.cpp/.h, op_kernel/*.cpp/.h
- 缺陷描述：moe_token_permute_grad使用MoeTokenUnpermuteTilingData与moe_token_unpermute类型名完全相同致符号冲突
- 修复模式：全部重命名为MoeTokenPermuteGradTilingData
- 可审查性：高
- 审查规则建议：REGISTER_TILING_DATA_CLASS类型名必须与算子名一致

### 54691c32 修复moe tiling/tiling_base头文件依赖
- 根因类别：头文件依赖路径错误
- 涉及文件：cmake/func.cmake, 多个moe算子tiling头文件
- 缺陷描述：tiling/tiling_base.h已迁移至tiling_base/tiling_base.h未同步；cmake变量直接赋值覆盖上层内容应追加
- 修复模式：批量修正include路径+cmake变量改List(APPEND)
- 可审查性：高
- 审查规则建议：头文件迁移全仓grep同步；cmake中file(GLOB)结果用List(APPEND)

### 0addaba3 fix bug
- 根因类别：符号冲突/函数命名冲突
- 涉及文件：moe/moe_token_permute_grad/op_host/moe_token_permute_grad_tiling.cpp
- 缺陷描述：TilingCompute函数名过于通用与其他模块同名函数链接冲突
- 修复模式：重命名为PermuteTilingCompute
- 可审查性：高
- 审查规则建议：全局函数名应加算子前缀避免冲突

### f284dfc9 fix moe dir
- 根因类别：构建系统配置错误/目录包含逻辑缺陷
- 涉及文件：CMakeLists.txt, cmake/custom_build.cmake, cmake/func.cmake, moe/CMakeLists.txt
- 缺陷描述：moe目录CMake构建逻辑在不同构建模式下add_subdirectory/glob路径不正确
- 修复模式：按构建模式条件化目录包含策略
- 可审查性：中
- 审查规则建议：CMake新增目录需确认所有构建模式下行为正确

### 05c16e68 Revert "fix include"
- 根因类别：头文件路径回退修复(前commit引入错误路径)
- 涉及文件：common/CMakeLists.txt, 多个*.h文件
- 缺陷描述：bb90fa51将tiling/tiling_base.h改为tiling_base/tiling_base.h但改错方向致编译失败，此commit revert修复
- 修复模式：Revert错误修改恢复正确路径
- 可审查性：高
- 审查规则建议：修改include路径前验证目标文件在SDK中的实际路径

### f2394fb7 api stub fix
- 根因类别：构建配置缺陷+代码合并冲突残留
- 涉及文件：cmake/custom_build.cmake, common/stub/op_api/level0/sort.h
- 缺陷描述：(1)Custom构建无aclnn源码时链接不存在的stub文件编译失败；(2)sort.h存在git merge conflict marker残留(<<<<<<< HEAD)
- 修复模式：条件化stub源文件生成+清理merge conflict残留
- 可审查性：高
- 审查规则建议：CI应检测代码中的merge conflict marker

### d0c0cbe3 fix moe + moe token unpermute with ep grad
- 根因类别：构建配置缺陷+新算子混合
- 涉及文件：CMakeLists.txt, cmake/*.cmake, moe/moe_token_unpermute_with_ep_grad/25个新文件
- 缺陷描述：moe目录构建路径需重新整理交给func.cmake处理+新增算子
- 修复模式：构建路径调整+新增算子代码
- 可审查性：低
- 审查规则建议：构建修复应与新功能分离为独立commit

### 58ffca69 built-in aclnnInner 修复
- 根因类别：链接依赖缺失
- 涉及文件：cmake/symbol.cmake
- 缺陷描述：gen_opapi_symbol在built-in模式下缺少ops_aclnn/profapi/ge_common_base等关键链接库致aclnnInner符号找不到
- 修复模式：补全链接依赖列表+whole-archive确保符号导出
- 可审查性：中
- 审查规则建议：新增构建模式需对照已有模式链接依赖逐项确认

### 88ccc195 【Custom构建】修复自动生成aclnn找不到符号问题
- 根因类别：链接配置错误/符号未导出
- 涉及文件：cmake/custom_build.cmake, cmake/obj_func.cmake
- 缺陷描述：ops_aclnn库链接位置不对且缺少--whole-archive包裹；optiling缺少OPBASE_LIB_DIR和common_obj依赖
- 修复模式：调整链接顺序和whole-archive作用范围
- 可审查性：中
- 审查规则建议：--whole-archive链接的库必须在archive/no-archive对之间

### 16996c83 统一obj命名，修复aclnn/aclnnInner问题
- 根因类别：命名不一致+CMake条件化缺失
- 涉及文件：CMakeLists.txt, cmake/*.cmake, gmm/grouped_matmul_swiglu_quant/op_host/CMakeLists.txt
- 缺陷描述：target名称与实际命名规范不一致(op_host_aclnn vs ${OPHOST_NAME}_opdef_aclnn_obj)；AclnnType枚举值不统一；条件化缺失
- 修复模式：统一命名规范+CMake条件化表达式
- 可审查性：低
- 审查规则建议：CMake target名用统一命名变量禁止硬编码

### dc79d8a3 gmm swigluquant bugfix
- 根因类别：算法参数硬编码错误(WholeReduceMax repeat参数)
- 涉及文件：gmm/grouped_matmul_swiglu_quant/grouped_matmul_swiglu_quant_utils.h
- 缺陷描述：WholeReduceMax的repeat参数硬编码为REPEAT_64(64)应为动态计算的repeat变量，count<4096时读取越界数据致ReduceMax结果错误
- 修复模式：硬编码常量替换为正确的计算变量
- 可审查性：高
- 审查规则建议：向量指令repeat参数禁止硬编码应由数据长度动态计算

### 45f3dfa4 GmmSwigluQuant bugFix
- 根因类别：ReduceMax算法精度缺陷(大维度下不精确)
- 涉及文件：gmm/grouped_matmul_swiglu_quant/*.h, op_host/*.cpp/.h
- 缺陷描述：n>4096时ReduceMax高阶API结果不精确。重写为分层缩减：<=4096用ReduceMaxSmall分段，>4096用BlockReduceMax两级64倍缩减
- 修复模式：重写ReduceMax分层缩减算法+优化同步原语(SetFlag/WaitFlag替代PipeBarrier<PIPE_ALL>)
- 可审查性：低
- 审查规则建议：ReduceMax类操作超过单次容量需分层缩减，验证分层边界条件
