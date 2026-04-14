# ops-nn 缺陷逐条分析

数据来源: ops-nn主仓(1474提交, 380条缺陷) + ops-nn-dev(2571提交, 612条缺陷), 合计992条缺陷

---

## ops-nn主仓 - 380条缺陷


## 第1轮 (commit 1-20)

### d91e1757ab41c3620057477c28adc62d395c2e00 revert//uss算子迁移
- 根因类别：功能缺陷导致回退（新特性引入问题）
- 涉及文件：38个文件，涵盖 index/unsorted_segment_sum/ 下的全部实现（tiling、kernel、host、proto、文档、示例），以及 docs/zh/op_list.md
- 缺陷描述：原始commit 1962c3991（新增unsorted_segment_sum算子支持Ascend950）被整体revert。该MR在合入unsorted_segment_sum算子的同时，在op_list.md中错误地新增了一个scatter算子的条目（而非unsorted_segment_sum的条目），属于文档修改错误。此外从revert行为来看，原始提交可能存在编译/集成问题导致紧急回退。
- 修复模式：完整revert原始commit，删除所有38个新增/修改的文件
- 可审查性：中
- 审查规则建议：新增算子时检查op_list.md中新增条目是否与实际算子名称一致；大规模特性合入前确保CI全流程通过

### 21127847202d552b85c4b62eefbb81219897ffcd depthwise preload 遗留问题修正
- 根因类别：模式适配不全 / 条件判断逻辑错误
- 涉及文件：conv_api_tiling_algorithm_HWmode.cpp/h, conv_api_tiling_algorithm_base.cpp/h, conv_common_func.h, conv_iterate_impl.h, conv2d_v2_base_tiling_tilingkey.cpp, conv2d_v2_instr_base_impl.h, conv2d_v2_instr_impl.h, convolution.cpp（共10个文件）
- 缺陷描述：depthwise preload功能此前仅适配了M模式，未适配HW模式。具体包括：(1) InferWiL1方法定义在HWmode子类中，基类无法使用，需上移；(2) ResetOptGroupDoubleBuffer中AL1 size计算使用固定的cubeInfo.m0/k0，未区分M/HW模式下不同tiling参数；(3) CheckOptGroupPreload中otherFlag强制要求outputOrder==M，排除了HW模式，multiMFlag未处理HW模式条件，错误排除了FLOAT8/HIFLOAT8/INT8/UINT8数据类型；(4) kernel侧SetLoad3dFMatrixForOptPreload硬编码M模式参数传入方式；(5) padLeftL1/padRightL1/wiLoadL1定义为子类私有成员，基类无法访问；(6) bias空间计算对齐参数错误
- 修复模式：将InferWiL1从子类上移到基类；按M/HW模式分支计算AL1 size；去掉M模式限制和数据类型限制，增加HW模式条件；将私有成员移到基类；SetLoad3dFMatrixForOptPreload改为无参按outputOrder分支处理
- 可审查性：中
- 审查规则建议：新增特性时检查是否覆盖了所有执行模式分支（M模式/HW模式/不同数据类型）

### afb09c7856c63e1158c681bf96f2fda127b70ecf fix dynamic_mx_quant & map_index tiling
- 根因类别：多核同步模式配置遗漏
- 涉及文件：index/map_index/op_host/map_index_tiling_arch35.cpp, quant/dynamic_mx_quant/op_host/arch35/dynamic_mx_quant_tiling_arch35.cpp
- 缺陷描述：两个算子的tiling代码中缺少SetScheduleMode(1)调用。map_index完全没有设置schedule mode。dynamic_mx_quant在blockSizeNumInAxis为奇数时多核存在数据依赖但未设置batch mode同步。
- 修复模式：在tiling函数中增加context->SetScheduleMode(1)设置batch mode
- 可审查性：高
- 审查规则建议：所有tiling函数中检查是否正确设置了SetScheduleMode，特别是多核场景下的同步模式

### 7b4a1b53c3f564330e91daa974a182a32d2be48a 修复aclnnNanMedian空指针校验
- 根因类别：错误处理宏参数误用（return遗漏）
- 涉及文件：index/gather_v2/op_api/aclnn_median.cpp
- 缺陷描述：OP_CHECK宏的第三个参数（失败时的动作）写成了裸值nullptr而非return nullptr。OP_CHECK在条件不满足时执行第三参作为失败动作，但原代码写的是, nullptr)，意味着失败时不会return而继续向下执行使用空指针。共14处OP_CHECK调用存在此问题。
- 修复模式：将14处OP_CHECK(..., nullptr)统一改为OP_CHECK(..., return nullptr)
- 可审查性：高
- 审查规则建议：OP_CHECK宏的第三个参数必须包含return语句；可用静态分析检测, nullptr)模式

### eda6863e4c1d7aa0d4420215ccd21703e4bea4d4 aclnnIndexCopy增加空指针校验
- 根因类别：空指针校验缺失
- 涉及文件：index/scatter_update/op_host/op_api/aclnn_index_copy.cpp
- 缺陷描述：ExecIndexCopyGetWorkspaceSize函数中，多个l0op算子调用（Contiguous、Reshape、ScatterUpdate、TransposeBySpecifiedAxis、ViewCopy）的返回值使用CHECK_RET宏校验（多线程不安全），且Reshape后的结果在scalar输入路径下可能为nullptr但无校验，TransposeBySpecifiedAxis返回值也无空指针检查。
- 修复模式：将CHECK_RET替换为OP_CHECK并附带return和错误日志；将Contiguous+Reshape+校验抽取到独立函数；对所有算子返回值增加OP_CHECK空指针检查
- 可审查性：高
- 审查规则建议：所有l0op/算子调用返回值必须进行空指针校验，且校验失败时必须return错误码；多线程场景优先使用线程安全的错误处理宏

### ce96300c400aff997559ffccaad74d3936c07069 修复codecheck告警问题
- 根因类别：变量遮蔽 + 函数声明与定义不一致
- 涉及文件：norm/add_rms_norm_cast/op_host/add_rms_norm_cast_tiling_arch35.cpp, norm/add_rms_norm_quant/op_host/add_rms_norm_quant_tiling.h, norm/add_rms_norm_quant/op_host/add_rms_norm_quant_tiling_arch35.cpp, norm/add_rms_norm_quant/op_kernel/arch35/add_rms_norm_quant_regbase_perf.h
- 缺陷描述：(1) 头文件中CalFullLoadBaseM函数声明参数名为baseM，但实际定义中语义是baseN，声明与定义不一致。(2) 局部变量tmpPower与外层同名变量产生遮蔽。(3) 多处多余空行。
- 修复模式：修正函数声明参数名；重命名局部变量为tmpPowerSize消除遮蔽；删除多余空行
- 可审查性：高
- 审查规则建议：启用编译器-Wshadow告警检测变量遮蔽；CI集成codecheck自动拦截声明/定义不一致

### fe7d6ee3136d22b358f9011940125ba59b928902 修复 mat_mul_v3 kernel代码中的错误注释说明
- 根因类别：注释与代码逻辑不一致
- 涉及文件：matmul/mat_mul_v3/op_kernel/mat_mul_deterministic_splitk_kernel.h
- 缺陷描述：funcParamsMK（preload参数值为2）注释写成"preload左矩阵"，funcParamsNK（preload参数值为1）注释写成"preload右矩阵"，实际参数值2表示N方向preload，1表示M方向preload，注释方向完全相反。
- 修复模式：纯注释修正
- 可审查性：低
- 审查规则建议：对关键算法参数使用枚举或命名常量代替magic number以减少注释歧义

### 6b52fdcf2fb7b85c53a56d4f71e72b27444d1513 FixViewShape
- 根因类别：张量格式校验逻辑错误（FORMAT_ND vs FORMAT_NCL）+ shape获取方式错误（StorageShape vs ViewShape）
- 涉及文件：rnn/single_layer_lstm_grad/op_host/op_api/aclnn_lstm_backward.cpp, 及对应examples/tests/docs
- 缺陷描述：(1) CheckFormatValid中所有张量统一使用FORMAT_ND校验，但隐藏状态h/c、门控张量i/j/f/o等实际格式为FORMAT_NCL。(2) ValidateInputShape使用GetStorageShape()获取shape校验，但非连续张量的StorageShape与ViewShape不同，导致合法输入被错误拦截。
- 修复模式：修正format校验逻辑按张量实际格式区分NCL和ND；将GetStorageShape()改为GetViewShape()
- 可审查性：中
- 审查规则建议：format校验检查是否所有张量被强制同一格式而未考虑实际差异；shape校验优先使用GetViewShape()，仅在需要物理布局时使用GetStorageShape()

### 0253078d581f0a93fb3c83ce4dd02f02372f1533 fix scatter_elements_v2 VF计算偏移时数据类型统一
- 根因类别：数据类型不一致导致计算错误（int32 vs int64隐式转换）
- 涉及文件：index/scatter_elements_v2/op_host/scatter_elements_v2_asc_tiling.cpp, index/scatter_elements_v2/op_kernel/scatter_elements.h
- 缺陷描述：(1) Tiling层GetTilingKey判断int64路径时仅检查allAxis_>MAX_INT32_NUM，遗漏dataAxis_和updatesAxis_检查。(2) Kernel层SimtComputeDim2到Dim8中stride数组声明为uint64_t而非模板参数COMP_T，COMP_T为int32时发生不必要的类型提升导致VF性能退化。
- 修复模式：扩展溢出判断覆盖所有axis；将stride数组类型从uint64改为COMP_T
- 可审查性：中
- 审查规则建议：使用模板参数COMP_T的函数中检查是否存在与COMP_T不一致的局部变量类型；tiling key数值范围判断应覆盖所有相关维度

### e9bed79c4ec2a0866e1cc8f4a78721e8c40dd9e1 修复MxA8W4 shape校验错误
- 根因类别：StorageShape vs ViewShape混用 + 错误日志不准确 + 条件分支遗漏
- 涉及文件：matmul/quant_batch_matmul_v3/op_api/aclnn_quant_matmul_v4.cpp, matmul/quant_batch_matmul_v4/op_host/op_api/aclnn_quant_matmul_v5.cpp
- 缺陷描述：(1) MicroScaling分支直接用GetViewShape().GetDim()而未使用已提取的变量，非连续tensor校验不一致。(2) 错误日志期望shape写groupDimK但实际校验groupDimK/2，日志误导调试。(3) MxScaleContiguousProcess调用条件缺少isA8W4Float分支，A8W4场景下scale未被正确转连续。
- 修复模式：统一使用已提取的ViewShape维度变量；修正错误日志期望值；补充A8W4条件分支
- 可审查性：中
- 审查规则建议：OP_LOGE中format参数值应与校验条件严格一致；新增数据类型时检查所有条件分支是否同步更新

### dc01cd867b83ae6a690507cc9f7068e9840ac169 Atlas推理系列pertoken量化模式下batch>1精度修复
- 根因类别：多维偏移量计算错误 + 多核配置错误
- 涉及文件：quant_batch_matmul_v3_tiling_arch20.cpp, quant_batch_matmul_v3_kernel_tiling_data.h, quant_batch_matmul_v3_pertoken_arch20.h
- 缺陷描述：三个独立问题。(1) tiling层hardcode SetBlockDim(1)，batch>1时未使用多核。(2) bias偏移量计算未区分bias是否带batch维度——固定使用b_idx*n_+n_idx*n0_，但bias shape为[n]（无batch维）时偏移计算错误。(3) scale的GM偏移量错误乘了b_idx*n_，scale是pertensor/perchannel级别不应随batch变化。另外将GetInputDesc改为GetOptionalInputDesc以正确处理可选输入。
- 修复模式：修正偏移量计算公式 + 增加biasWithBatch维度信息传递 + 修正多核blockDim配置
- 可审查性：中
- 审查规则建议：检查SetBlockDim是否使用硬编码1；可选输入使用GetOptionalInputDesc；GM偏移量出现batch_idx乘法时验证tensor是否确实有batch维度

### 15e40a4867a2f2d7c1b17fbb1e7f136057c489f7 修复addmm
- 根因类别：未初始化变量 + 空指针解引用风险
- 涉及文件：matmul/common/op_host/op_api/matmul_util.cpp, matmul/common/op_host/op_api/matmul_util.h
- 缺陷描述：(1) PromoteResult结构体的logMessage指针未初始化，某些GetUpperDtypeByLookUpTable返回路径中为野指针，后续无条件OP_LOGE("%s", result.logMessage)导致崩溃。(2) TensorInfo结构体的tensor指针、dataType、format也未初始化。
- 修复模式：结构体成员默认初始化 + 使用前判空
- 可审查性：高
- 审查规则建议：C++结构体指针成员必须初始化为nullptr；所有成员变量应在声明处初始化；OP_LOGE中%s前必须判空

### 910dac63b4adf28dd9a9a6b98e9f85b23b194e2a fix bug for QuantUpdateScatter & optimize compile time
- 根因类别：地址偏移量计算公式错误
- 涉及文件：quant_update_scatter_large_batch_little_quant_regbase.h（核心bug），及7个文件（编译优化）
- 缺陷描述：gmVarOffset_计算公式错误。原式(axisOffset + innerLoopIdx) * innerLoopEle将两个语义不同的变量相加后统一乘innerLoopEle，但axisOffset应乘varDim3（axis维度步长），innerLoopIdx应乘innerLoopEle。
- 修复模式：拆分为actualBsIdx * dstBsStride + axisOffset * varDim3 + innerLoopIdx * innerLoopEle
- 可审查性：低
- 审查规则建议：多维数组偏移量计算应逐维展开，每个维度索引乘以对应步长，避免将不同维度索引合并为一个表达式

### 941d3c7f88b8040a0e9da984a5947390c9e06791 修复tbmm的warning报错信息
- 根因类别：有符号/无符号比较类型不匹配
- 涉及文件：transpose_batch_mat_mul_base_tiling.cpp
- 缺陷描述：常量kSupportedInnerAxis声明为uint64_t，但与int64_t类型的shape维度值比较，编译器产生有符号/无符号比较warning。
- 修复模式：将类型从uint64_t改为int64_t
- 可审查性：高
- 审查规则建议：开启-Wsign-compare并视为错误；与shape维度比较的常量应声明为int64_t

### a810b9c0a7f3418dd32b91221c8f457283c228c1 applyTopkTopP算子950性能劣化修复及TopP分支增加至少保留一个值机制
- 根因类别：平台适配缺失（核数配置） + 功能遗漏（保底值机制）
- 涉及文件：apply_top_k_top_p_with_sorted_tiling.cpp, apply_top_p_with_sorted.h
- 缺陷描述：(1) 950平台核数多于A2，单TopP分支scatter搬出操作数量不均匀导致性能回退。(2) 单TopP分支缺少"至少保留一个值"的保底机制，当所有cumsum值未达阈值p时输出可能为全零。
- 修复模式：950平台特判限制核数上限为48；新增CopyOutLastValue函数保底机制
- 可审查性：中
- 审查规则建议：新增算子对照标准参考实现检查边界case；多平台算子审查核数分配策略是否考虑不同soc版本

### c8ca6becf24497e9499339d741bce760bd2e19de 解决aclnnAdaptiveMaxPool3d索引值类型为int64时出现的类型不匹配
- 根因类别：接口参数缺失/默认值不当
- 涉及文件：pooling/adaptive_max_pool3d/op_api/aclnn_adaptive_max_pool3d.cpp
- 缺陷描述：调用MaxPool3DWithArgmaxV2Ncdhw接口时未传入indices的实际数据类型（indicesDtype），接口默认使用int32，但当用户传入int64时底层计算与输出tensor类型不匹配。
- 修复模式：Regbase模式下从indices tensor获取真实dtype并显式传递
- 可审查性：高
- 审查规则建议：调用具有默认参数的接口时，检查实际数据类型是否可能与默认值不一致，涉及dtype参数应显式传递

### 8583f0f3b8ca33c3ffd6eba1cf0f6e258ac4106b quantbatchmatmul a8w4kernel修复，修复workspace偏移，修复ub多申请了大小
- 根因类别：硬编码常量错误 + sizeof语义误用
- 涉及文件：matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4_msd.h
- 缺陷描述：(1) workspace偏移使用硬编码MM_BASE_BLOCK_OFFSET=32768，实际应随baseM*baseN动态变化。(2) UB buffer申请用alignKSize_*sizeof(int4b_t)，但sizeof(int4b_t)返回1（非预期的0.5），导致int4类型buffer多申请一倍。
- 修复模式：硬编码替换为动态值baseN_*baseM_；用CeilDiv(alignKSize_, INT4_SIZE)替代sizeof误用
- 可审查性：中
- 审查规则建议：检测内存分配/偏移相关硬编码魔数；sub-byte类型(int4)的sizeof调用需特别审查

### 5da80998bd992fe9ca241b54589ee7373fe1eef2 fix scatter_list aclnn & repeat_inter_leave_grad warning
- 根因类别：配置错误 + printf格式符不匹配
- 涉及文件：index/repeat_interleave_grad/op_host/repeat_interleave_grad_int_repeat_tiling.cpp, index/scatter_list/op_host/CMakeLists.txt
- 缺陷描述：(1) 日志用%ld输出uint64_t类型的CACHE_BUF_SIZE，应为%lu。(2) scatter_list的CMakeLists.txt中ACLNNTYPE配置为aclnn_inner，实际应为aclnn。
- 修复模式：%ld改为%lu；aclnn_inner改为aclnn
- 可审查性：高
- 审查规则建议：CI门禁拦截printf格式符不匹配；CMakeLists.txt中ACLNNTYPE审查是否符合接口设计意图

### 67c665fd267c68937ddb872c9f54e0d97e1f4580 avg_pool_v2_grad 原型提交及infershape漏洞修复
- 根因类别：逻辑运算符错误 + 输出shape未赋值
- 涉及文件：pooling/avg_pool_v2_grad/op_graph/avg_pool_v2_grad_proto.h, pooling/avg_pool_v2_grad/op_host/avg_pool_v2_grad_infershape.cpp
- 缺陷描述：(1) 维度校验条件 inputDimNum != CHW_DIMS || inputDimNum != NCHW_DIMS 使用||（逻辑或），恒为true，合法输入也被拒绝，应改为&&。(2) infershape设置了输出shape的维度数SetDimNum但从未对每个维度赋值SetDim，输出shape未初始化。
- 修复模式：||改为&&修正逻辑；新增循环从输入数据读取shape值并写入输出
- 可审查性：高
- 审查规则建议：!= A || != B模式（恒真表达式）应作为静态检查规则；infershape中SetDimNum后必须有对应SetDim赋值

### 8f737553ce98e0ecab841fcf3d4c2b6dd50e90aa tiling性能优化 fmap可全载实际未全载问题修复
- 根因类别：条件判断逻辑错误导致分支策略不优
- 涉及文件：conv/common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_BBmode.cpp
- 缺陷描述：L1缓存加载策略函数将fmap+weight全载、fmap全载都置于batch全载条件之下，多核场景batch不全载时fmap即使可全载也无法进入对应分支，退化为weight全载路径。
- 修复模式：重构条件分支：fmap+weight全载作为最高优先级无条件判断；按迭代顺序分支调整fmap全载与weight全载优先级
- 可审查性：低
- 审查规则建议：tiling策略多维度交叉判断时审查前置条件是否过于严格；性能相关分支变更应配套性能基准测试

## 第2轮 (commit 21-40)

### a71c47f707d42e9b5cec4c39a26d89d3780be2ac 修复TBMM算子资料scale的维度错误
- 根因类别：文档错误
- 涉及文件：matmul/transpose_batch_mat_mul/docs/aclnnTransposeBatchMatMul.md
- 缺陷描述：scale参数维度描述为2维，实际tbmm的scale只支持1维。文档表格中维度值从2修正为1。
- 修复模式：纯文档修正，修改md表格中的维度值
- 可审查性：高
- 审查规则建议：算子文档中参数维度描述应与代码中shape校验逻辑一致

### de7c68326010644a7a4998fe73e6e72ca15cc51c 修复matmulv3减少编译耗时导致后冒烟打断问题
- 根因类别：C++模板类型作用域错误（using声明在if-else块内无效）
- 涉及文件：matmul/mat_mul_v3/op_kernel/mat_mul_deterministic_splitk_kernel.h, mat_mul_unaligned_deterministic_splitk_kernel.h
- 缺陷描述：原代码试图通过if-else分支内的using声明来选择不同模板类型（aType/bType），但C++中块作用域内的using声明不会影响块外的后续代码。导致所有分支都使用外层默认的NZ格式类型，isNzA/isNzB为false时仍使用NZ格式模板参数调用MatMul函数，引发精度问题。这是之前一个编译耗时优化PR引入的回退。
- 修复模式：将类型选择和函数调用合并到每个if-else分支内部，确保每个分支使用正确的模板类型参数直接调用函数
- 可审查性：高
- 审查规则建议：if/else块内using类型别名声明不会传播到块外，检测此模式应作为静态审查规则；模板类型选择逻辑必须与调用点在同一作用域

### 07e77ddd07a00a13e0e71f8efa6aca8e6abe0d79 修复aclnnInplaceAddmm接口走入gemmv3算子时不支持输入转置的问题
- 根因类别：算子属性索引偏移错误（不同算子的attr排列不同）
- 涉及文件：matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp
- 缺陷描述：GetShape函数通过固定索引GetAttrPointer<bool>(0)和GetAttrPointer<bool>(1)读取transposeX1/transposeX2属性。但GemmV3算子的attr顺序为[alpha, beta, transposeX1, transposeX2, ...]，transpose属性在索引2和3处而非0和1。当aclnnInplaceAddmm走入GemmV3路径时，读到的是alpha/beta而非transpose标志，导致转置信息丢失。
- 修复模式：增加NodeType判断，GemmV3取index 2/3，其他取index 0/1
- 可审查性：高
- 审查规则建议：通过索引获取算子属性时，必须确认不同算子类型的attr排列顺序是否一致；优先使用属性名而非硬编码索引获取attr

### fe95ffda590b30a132877ac88378ca60d0a1d10a 删除classify_rule.yaml中不存在的文件路径
- 根因类别：构建配置错误（引用已废弃算子路径）
- 涉及文件：classify_rule.yaml
- 缺陷描述：vfusion-c模块的unrelease->test_code配置中引用了norm_rope_concat和norm_rope_concat_grad的tests/examples路径，但这两个算子已废弃不再维护，实际路径不存在，造成构建干扰。
- 修复模式：从yaml配置中删除4条不存在的路径
- 可审查性：高
- 审查规则建议：yaml/cmake配置中引用的文件路径应在CI中校验是否实际存在

### f24f9dba9dba3d980a8e6f2fdd7022e7981adc54 fix: infershape ut failed
- 根因类别：构建配置错误（重复符号定义）
- 涉及文件：cmake/ut.cmake, tests/ut/op_host/CMakeLists.txt, 2个ut文件（空行变更）
- 缺陷描述：opbase的源码被tiling和infershape的UT target分别编译链接，导致同一符号在链接时重复定义。此外opbase_util_objs/opbase_infer_objs/opbase_tiling_objs直接嵌入到多个target的$<TARGET_OBJECTS>中导致冲突。
- 修复模式：新增add_opbase_ut_common()函数将opbase编译为公共的static library（opbase_ut_common），tiling和infershape的UT通过target_link_libraries引用，避免重复符号
- 可审查性：中
- 审查规则建议：CMake中同一组object files不应通过TARGET_OBJECTS嵌入多个target，应封装为独立static library

### 04e45bfc0b99f30ea6a11123163043b51795aad0 aclnnBatchMatMulWeightNz.md 确定性说明fix
- 根因类别：文档错误
- 涉及文件：matmul/batch_mat_mul_v3/docs/aclnnBatchMatMulWeightNz.md
- 缺陷描述：确定性说明中列出了特定产品系列（Atlas训练/推理系列），但引入了不支持的版本，需改为通用说明。同时缺少950D的确定性说明。
- 修复模式：删除特定产品系列限定，改为通用描述
- 可审查性：高
- 审查规则建议：文档中涉及产品系列/版本兼容性的描述应与实际代码支持矩阵一致

### 640a1683e4f647a4ddf8d745abf4fd463a80c320 fix: ascendc depend
- 根因类别：构建配置错误（CMake target缺少前置依赖）
- 涉及文件：cmake/gen_ops_info.cmake
- 缺陷描述：ascendc_impl_gen target生成py文件时，其依赖的ops_info_gen_*等target尚未构建完成。原代码先创建ascendc_impl_gen target再通过foreach生成其依赖target并add_dependencies，但custom_command的DEPENDS参数中未包含这些前置target。且add_ops_impl_target函数未接受DEPENDS参数。
- 修复模式：(1) 函数签名增加DEPENDS参数传递到custom_command。(2) 将ascendc_impl_gen target的创建移到foreach循环之后，确保依赖target已定义。(3) 将依赖列表通过DEPENDS参数传入custom_command
- 可审查性：中
- 审查规则建议：CMake custom_command/custom_target的DEPENDS必须包含所有输入依赖；target创建顺序应在其依赖target之后

### 4f594168b090538019bac42de84424f294f5950f 增加空指针校验
- 根因类别：空指针校验缺失
- 涉及文件：matmul/batch_mat_mul_v3/op_host/op_api/aclnn_addbmm.cpp, matmul/mat_mul_v3/op_host/op_api/aclnn_matmul.cpp
- 缺陷描述：aclnn_addbmm和aclnn_matmul中，ConvertToTensor、AllocScalar、AllocIntArray、Fill、ReduceSumOp、SetTensorToNZFormat等关键操作的返回值未校验，在内存分配失败时会产生空指针解引用。共新增约13处CHECK_RET校验。
- 修复模式：在每个可能返回空指针的操作后增加CHECK_RET(ptr != nullptr, error_code)
- 可审查性：高
- 审查规则建议：所有返回指针的API调用（ConvertToTensor/AllocScalar/AllocIntArray/Fill/ReduceSumOp等）后必须有空指针校验

### a8c4157ac9910b74ba82cead877c7d3fc59da847 fix: 修复UT编译失败
- 根因类别：构建配置错误（opbase对象链接缺失 + 变量传递遗漏）
- 涉及文件：build.sh, cmake/ut.cmake, cmake/variables.cmake, tests/ut/op_host/CMakeLists.txt
- 缺陷描述：(1) ut.cmake中optiling_ut和infershape_ut的static_lib缺少opbase_util_objs/opbase_tiling_objs/opbase_infer_objs依赖。(2) variables.cmake中硬编码CANN_3RD_LIB_PATH默认值被删除后，build.sh未将该变量传递到cmake命令行。(3) tests/ut/op_host/CMakeLists.txt中opbase对象被从顶层链接移除后下层未补充。
- 修复模式：ut.cmake中补充opbase对象到static_lib；build.sh增加-DCANN_3RD_LIB_PATH传递；调整CMakeLists.txt链接
- 可审查性：中
- 审查规则建议：删除默认值或移除依赖时，检查所有引用点是否同步更新；构建变量的传递链完整性

### b3aeb54fe71b06376c6e09c29b3098ca9c01a029 fix dynamic quant copyout bug
- 根因类别：数据搬运尾块越界（固定大小搬运 vs 实际大小搬运）
- 涉及文件：quant/dynamic_quant/op_kernel/dynamic_quant_unalign_310p.h, dynamic_quant_unalign_large_310p.h
- 缺陷描述：dynamic_quant算子在310p后端copyout阶段，尾块搬运时使用固定的numCopyRow_长度而非实际剩余行数。当输入不对齐时，尾块实际数据量小于numCopyRow_，但DataCopy按numCopyRow_搬运，越界写入踩踏后续内存。
- 修复模式：(1) 新增CopyOutUnalign方法，尾块搬运时按实际calCount计算realNumRowAlign。(2) isTailLoop_分支中用realNumRowAlign替代固定numCopyRow_
- 可审查性：中
- 审查规则建议：DataCopy的长度参数在尾块处理时必须使用实际剩余数据量而非固定值；循环+尾块模式下审查尾块搬运长度

### 6feeae0a83c340973f4db5c2e78ffffb95ad7267 修改EmbeddingDenseGradV2 idxOffset翻转导致indices取值越界导致aic
- 根因类别：数组越界访问 + 条件判断逻辑不完善
- 涉及文件：index/embedding_dense_grad_v2/op_kernel/v35/embedding_dense_grad_v2_regbase.h
- 缺陷描述：(1) 循环中idxOffset在每次迭代累加interval后可能超过indices数组大小，但无边界检查，下次循环indices(idxOffset)越界导致AIC错误。(2) 负索引判断idx < 0与idx >= numWeights分开写，但idx为有符号类型转uint64_t比较时，负数会变成极大正数自然>=numWeights，原来的idx < 0检查冗余且不如统一用unsigned比较直接。
- 修复模式：循环内增加idxOffset >= indices.GetSize()的边界检查及break；将idx < 0 || idx >= numWeights合并为static_cast<uint64_t>(idx) >= numWeights
- 可审查性：高
- 审查规则建议：循环中递增的索引变量访问数组前必须检查边界；有符号索引与无符号上界比较应统一为unsigned比较

### 0700b4fdee8bb96a4cbfba9d4b42365d752a1d00 fix: compile with opbase source
- 根因类别：构建配置重构（从库依赖改为源码编译）
- 涉及文件：CMakeLists.txt, cmake/func.cmake, 及多个cmake文件
- 缺陷描述：opbase之前作为外部库依赖（ops_base_util_objs/ops_base_infer_objs），现改为下载源码本地编译。原有target名不一致（ops_base_*改为opbase_*），函数名/链接关系需同步更新。
- 修复模式：新增add_opbase_modules()函数管理opbase源码编译；统一target名称；调整include/link依赖
- 可审查性：中
- 审查规则建议：重命名target时全局搜索确认所有引用点同步更新

### 07bec2430e0fcc03bedabc846849e3802929e8dd Fix repeat_interleave kernel102 functional issue
- 根因类别：计数器累加位置错误导致数组越界
- 涉及文件：index/repeat_interleave/op_kernel/arch35/repeat_interleave.h
- 缺陷描述：copyFromXNum_在CopyOneCpToRepeatOut函数末尾按dataCount累加，但实际应在外层CopyXToMatchOut循环中按mergedDims[2]累加。在CopyOneCpToRepeatOut内累加会导致尾块处理时copyFromXNum_超过实际数据量，后续访问越界。
- 修复模式：将copyFromXNum_ += dataCount从CopyOneCpToRepeatOut移除，在CopyXToMatchOut的循环体末尾改为copyFromXNum_ += tilingData_.mergedDims[2]
- 可审查性：中
- 审查规则建议：跨函数共享的计数器/偏移量变量，其累加位置应与实际数据消费粒度匹配；嵌套函数调用中避免在底层函数更新上层状态

### 22b4d33516f496aa833cac3b21921f131cf1eaf4 fix aclnnprelubackward bug
- 根因类别：Shape引用错误（使用中间转换后的shape而非原始输入shape）
- 涉及文件：activation/p_relu_grad_update/op_api/aclnn_prelu_backward.cpp
- 缺陷描述：PReLU backward中gradInput需要reshape到与原始self相同的shape。但代码使用contiguousSelf->GetViewShape()，而contiguousSelf是经过内部Contiguous/Reshape转换后的tensor，其shape可能与原始输入不同（如1维张量被扩展为多维）。reshape使用错误的shape导致输出与预期不符。
- 修复模式：新增originalSelfShape参数，在reshape时使用原始self的shape而非转换后的contiguousSelf的shape
- 可审查性：中
- 审查规则建议：Contiguous/Reshape操作后的tensor shape与原始输入shape可能不同，需要保持原始shape时应另存变量

### 8f6ccaea4116bed1b4a62f6541561c7215780587 addlayernormgrad上边界问题修复
- 根因类别：整数溢出（uint32不足以表示大shape计算结果）
- 涉及文件：norm/add_layer_norm_grad/op_host/add_layer_norm_grad_tiling.cpp/h, op_kernel/add_layer_norm_grad_cut_d.h, tests/ut
- 缺陷描述：roundUpNumLastDimFloat字段声明为uint32_t，ROUND_UP宏计算numLastDim对齐后乘sizeof(float)，当numLastDim超过2^31-1（如2147483649）时乘法溢出。同理deterministicWorkSpaceSize也声明为uint32_t不够。
- 修复模式：将roundUpNumLastDimFloat和deterministicWorkSpaceSize从uint32_t改为uint64_t；ROUND_UP前先static_cast<uint64_t>避免中间溢出
- 可审查性：高
- 审查规则建议：涉及shape维度乘法的变量必须使用int64_t/uint64_t；ROUND_UP宏的输入在乘法前检查类型是否足够

### 0958e639537066becf8556011f3305adfdeccd8e fix: 同一台机器同时编译kernel导致卡死
- 根因类别：构建脚本错误（shell重定向导致多进程死锁）
- 涉及文件：cmake/gen_ops_info.cmake
- 缺陷描述：custom_target命令末尾追加echo $(MAKE) &> /dev/null，&>在某些shell环境下将echo和后续MAKE命令的输出都重定向到/dev/null，多进程同时编译时可能因MAKE变量展开和文件描述符竞争导致卡死。
- 修复模式：去掉&> /dev/null重定向，改为直接echo $(MAKE)
- 可审查性：中
- 审查规则建议：CMake custom_command中避免shell重定向到/dev/null，特别是涉及$(MAKE)变量展开的场景

### 3cf703d7e1247de3abf87eeea099abbd0e3477aa aclnn_add_rms_norm_dynamic_quantv2 fix int4 support pta
- 根因类别：条件判断缺失（outputMask为空时对空输出执行操作）
- 涉及文件：norm/add_rms_norm_dynamic_quant/op_host/op_api/aclnn_add_rms_norm_dynamic_quant_v2.cpp
- 缺陷描述：int4类型输出分支中，当存在outputMask且某个输出被mask掉时（PTA传入空指针），代码仍无条件对该输出执行AddRmsNormDynamicQuantV2Int42Int32PackedTensor和ViewCopy操作，访问空指针导致内部接口校验失败。
- 修复模式：增加outputMask判断，processOut1/processOut2根据outputMask决定是否执行对应输出的计算和拷贝
- 可审查性：高
- 审查规则建议：带outputMask参数的算子，所有输出处理分支必须检查对应mask位；空指针输出不应参与任何计算

### c1924396cd30039bbf7d7d657c677bbf0bac49fe fix noaicpu option
- 根因类别：构建脚本参数命名不一致
- 涉及文件：build.sh
- 缺陷描述：build.sh中长选项定义为no_aicpu（带下划线），但其他ops仓库统一使用noaicpu（不带下划线），导致跨仓库一致性问题。
- 修复模式：将no_aicpu改为noaicpu
- 可审查性：高
- 审查规则建议：构建参数命名应与其他仓库保持一致；新增构建选项时检查已有仓库的命名规范

### bba9de57ca6c64633393810a44397ad3c6d53fcd FixLstmBackwardMd
- 根因类别：文档错误（参数描述笔误、shape说明不准确、示例代码语法错误）
- 涉及文件：rnn/single_layer_lstm_grad/docs/aclnnLstmBackward.md
- 缺陷描述：(1) input参数shape描述未说明与batchSizesOptional的关联。(2) params中bias描述未区分bias_ih和bias_hh。(3) "政协"笔误应为"正向"。(4) 示例代码多余分号。
- 修复模式：修正shape说明、参数描述、笔误、语法错误
- 可审查性：高
- 审查规则建议：API文档中参数shape描述应覆盖所有输入组合条件；使用spell check工具

### 40f9a0d42541dfe69feb821514661e67e84338e0 fix conv warning
- 根因类别：代码格式问题（多余空行产生warning）
- 涉及文件：conv/convolution_forward/op_host/op_api/aclnn_quant_convolution.cpp
- 缺陷描述：函数间多余空行导致编译warning。
- 修复模式：删除多余空行
- 可审查性：高
- 审查规则建议：启用clang-format自动格式化

## 第3轮 (commit 41-60)

### 84a249a2 fix sizeof(T) to sizeof(float) in test demo
- 根因类别：测试代码内存分配错误
- 涉及文件：test demo中的内存分配代码
- 缺陷描述：测试demo中使用sizeof(T)分配内存，但实际数据类型应为float，模板参数T与实际类型不匹配导致分配大小错误。
- 修复模式：将sizeof(T)替换为sizeof(float)
- 可审查性：高
- 审查规则建议：sizeof参数应与实际使用的数据类型一致，模板代码中尤其注意类型参数是否被正确使用

### 52f9ba38 fix float implicit conversion to int64 + UT coverage
- 根因类别：隐式类型转换
- 涉及文件：算子host代码 + UT测试
- 缺陷描述：float值隐式转换为int64_t，可能导致精度丢失或未定义行为。同时UT覆盖不足未能发现此问题。
- 修复模式：添加static_cast显式转换 + 补充UT用例
- 可审查性：高
- 审查规则建议：编译器warning(-Wfloat-conversion)应作为error处理；CR时关注浮点到整型的隐式转换

### 1ca42282 fix eigen 5.0.0 download URL line endings
- 根因类别：构建配置错误
- 涉及文件：eigen依赖下载配置
- 缺陷描述：eigen 5.0.0下载URL因行尾换行符问题导致下载失败。
- 修复模式：修正URL字符串中的换行符
- 可审查性：高
- 审查规则建议：构建脚本中的URL字符串应避免跨行拼接，或使用strip处理

### ea9b02ef fix document format (cubeMathType term tag)
- 根因类别：文档格式错误
- 涉及文件：cubeMathType相关文档
- 缺陷描述：文档中cubeMathType使用了`<term>`标签，渲染异常。
- 修复模式：移除`<term>`标签
- 可审查性：高
- 审查规则建议：文档中避免使用可能被解析器误解的HTML/XML标签

### 8cef5510 fix multi-core index offset missing in sort operator
- 根因类别：多核并行索引偏移缺失(严重)
- 涉及文件：sort算子的simd_sort和simt_sort路径
- 缺陷描述：indicesOffset计算缺少`eachCoreIndexCount * GetBlockIdx()`项。在多核场景下，每个核输出的indices没有加上该核对应的全局偏移量，导致所有核输出的索引都从0开始而非各自的全局起始位置。
- 修复模式：在indicesOffset计算中加入`eachCoreIndexCount * GetBlockIdx()`
- 可审查性：中 — 需要理解多核分片逻辑
- 审查规则建议：多核算子中所有输出索引/偏移量必须包含GetBlockIdx()相关的全局偏移；多核场景必须有多核UT验证索引正确性

### 49bbb988 fix scatter_elements_v2 warnings
- 根因类别：编译warning(类型转换+未使用变量)
- 涉及文件：scatter_elements_v2算子代码
- 缺陷描述：size_t到int64_t的隐式转换warning、未使用变量warning、变量命名不规范。
- 修复模式：类型转换显式化、删除未使用变量、重命名变量
- 可审查性：高
- 审查规则建议：CI中开启-Wall -Werror；定期清理unused variable

### cce7a445 fix L1 bank conflict in buffer layout
- 根因类别：L1 bank冲突导致性能下降(严重)
- 涉及文件：矩阵运算kernel代码
- 缺陷描述：L1缓冲区布局为|A0|B1|...|B0|A1|，A和B的交错排列导致同bank访问冲突。
- 修复模式：将布局调整为|A0|B0|...|A1|B1|，使A/B连续排列避免bank冲突
- 可审查性：低 — 需要深入理解硬件bank结构
- 审查规则建议：L1 buffer分配时A/B矩阵应连续排列，避免交错；性能敏感kernel应做bank冲突检测

### 0a2fb0d7 fix RNN unused parameter + DFX_OUT macro missing parameter
- 根因类别：接口参数遗漏
- 涉及文件：RNN算子代码
- 缺陷描述：1) RNN中存在未使用参数；2) DFX_OUT宏调用缺少outputMask参数，导致调试输出信息不完整。
- 修复模式：删除未使用参数 + 补充DFX_OUT宏的outputMask参数
- 可审查性：高
- 审查规则建议：宏调用参数个数必须与宏定义一致；使用编译器warning检测未使用参数

### 28ab6d70 fix StorageShape to ViewShape in thnn_fused_lstm_cell_backward
- 根因类别：StorageShape/ViewShape混淆(严重)
- 涉及文件：thnn_fused_lstm_cell_backward算子host代码
- 缺陷描述：使用StorageShape获取tensor维度信息，但该tensor可能经过view操作，StorageShape反映的是底层存储形状而非逻辑视图形状，导致shape信息错误。
- 修复模式：将StorageShape替换为ViewShape
- 可审查性：中 — 需要理解Storage vs View语义
- 审查规则建议：获取tensor shape时默认使用ViewShape，仅在确认需要底层存储形状时才使用StorageShape

### f31988fa fix code-check issues (parameter name mismatch, unused header, implicit bool)
- 根因类别：代码规范问题(多项)
- 涉及文件：多个算子文件
- 缺陷描述：1) 函数声明与定义的参数名不一致；2) 包含未使用的头文件；3) 隐式bool比较。
- 修复模式：统一参数名 + 删除未使用头文件 + 显式bool比较
- 可审查性：高
- 审查规则建议：clang-tidy规则覆盖参数名一致性检查和隐式bool转换检查

### dc62e8dd fix CreateView null pointer check missing in batch_matmul
- 根因类别：空指针检查缺失(严重)
- 涉及文件：batch_matmul_util.cpp (3处)
- 缺陷描述：CreateView返回值未做null检查就直接使用，如果创建失败会导致空指针解引用崩溃。
- 修复模式：在3处CreateView调用后添加null检查
- 可审查性：高
- 审查规则建议：所有Create*/New*类工厂函数返回值必须做null检查后再使用

### d196e790 fix QuantMatmulWeightNz document format
- 根因类别：文档格式错误
- 涉及文件：QuantMatmulWeightNz算子文档
- 缺陷描述：文档格式问题影响渲染。
- 修复模式：修正文档格式
- 可审查性：高
- 审查规则建议：文档发布前预览渲染效果

### eb818ca6 fix GetAttrPointer int32_t should be int64_t for ignoreIndex
- 根因类别：属性类型不匹配(严重)
- 涉及文件：算子host代码
- 缺陷描述：使用GetAttrPointer<int32_t>获取ignoreIndex属性，但该属性实际类型为int64_t。类型不匹配导致只读取了64位值的低32位，高32位被截断。
- 修复模式：将GetAttrPointer<int32_t>改为GetAttrPointer<int64_t>
- 可审查性：高
- 审查规则建议：GetAttrPointer的模板类型参数必须与算子注册时声明的属性类型严格一致；建立属性类型映射表供CR参考

### 6eee5478 fix tiling key parameter error + SetAtomicNone after return
- 根因类别：tiling参数错误 + 同步位置错误(严重)
- 涉及文件：算子tiling代码 + kernel代码
- 缺陷描述：1) tiling key参数传递错误导致选择了错误的tiling策略；2) 在single-batch路径中SetAtomicNone被放在return语句之后，永远不会被执行，导致原子操作模式未被正确重置。
- 修复模式：修正tiling key参数 + 将SetAtomicNone移到return之前
- 可审查性：中
- 审查规则建议：return语句后不应有可执行代码(dead code检测)；tiling key参数应有枚举约束而非裸字符串

### 852f21d6 revert batchmatmul non-contiguous input support
- 根因类别：功能回退(feature引入问题)
- 涉及文件：batchmatmul算子代码
- 缺陷描述：之前添加的non-contiguous input支持引入了问题，需要整体回退。
- 修复模式：revert整个feature
- 可审查性：N/A — 回退操作
- 审查规则建议：大feature合入前应有充分的集成测试覆盖；考虑feature flag控制新功能上线

### edac4382 replace magic numbers with named constants
- 根因类别：代码可维护性(magic number)
- 涉及文件：算子代码
- 缺陷描述：代码中使用magic number，可读性差且容易出错。
- 修复模式：定义命名常量替换magic number
- 可审查性：高
- 审查规则建议：禁止裸数字常量(0/1/-1除外)，必须使用命名常量

### ed0bd5a1 fix QBMMv4 WaitFlag position wrong + A8W8GB tiling split missing
- 根因类别：同步时序错误 + tiling逻辑缺失(严重)
- 涉及文件：QBMMv4 kernel代码 + A8W8GB tiling代码
- 缺陷描述：1) WaitFlag位置错误 — 放在了SetFlag之前而非之后，导致等待的是上一轮的flag而非当前轮，数据可能未就绪就被使用；2) A8W8GB场景的tiling分片逻辑缺失，导致大shape场景无法正确分片。
- 修复模式：将WaitFlag移到SetFlag之后、数据使用之前 + 补充A8W8GB tiling split逻辑
- 可审查性：低 — SetFlag/WaitFlag时序需要理解AscendC流水线模型
- 审查规则建议：SetFlag/WaitFlag必须成对出现且顺序为Set→Wait→Use；tiling代码必须覆盖所有量化模式

### 67995b31 fix mmv2 Nz kernel bin not found (unstable nAxis value)
- 根因类别：变量值不稳定导致kernel匹配失败(严重)
- 涉及文件：mmv2算子aclnn层代码
- 缺陷描述：使用mat2的nAxis值来匹配kernel bin，但该值在aclnn处理过程中会被修改(如transpose/reshape操作改变axis含义)，导致后续查找kernel bin时使用了错误的值，找不到对应kernel。
- 修复模式：替换为稳定的mmOpInfo.shapeInfo.nDim值，该值在算子信息构建阶段就已确定且不会再变化
- 可审查性：低 — 需要理解aclnn处理流水线中tensor属性的变化
- 审查规则建议：kernel bin匹配参数应使用处理链早期确定的稳定值，避免使用中间过程中可能被修改的tensor属性

### 479d15b8 fix document link + classify_rule.yaml path cleanup
- 根因类别：文档链接错误 + 配置路径错误
- 涉及文件：文档文件 + classify_rule.yaml
- 缺陷描述：文档中的链接指向错误地址；classify_rule.yaml中的路径不正确。
- 修复模式：修正链接和路径
- 可审查性：高
- 审查规则建议：文档CI中加入链接有效性检查(dead link checker)

### a1cc187b fix EmbeddingDenseGradV2 idx upper bound check missing
- 根因类别：数组越界检查缺失(严重)
- 涉及文件：EmbeddingDenseGradV2算子kernel代码
- 缺陷描述：对idx只检查了`idx < 0`的下界，缺少`idx >= numWeights`的上界检查。当idx超出权重表大小时会导致越界访问。
- 修复模式：添加`idx >= numWeights`上界检查条件
- 可审查性：高
- 审查规则建议：数组/表索引校验必须同时包含上下界检查；Embedding类算子的index参数必须做range validation

## 第4轮 (commit 61-80)

### 92982b89 空tensor校验逻辑缺失 (yOffset)
- 根因类别：空tensor与空指针语义混淆(严重)
- 涉及文件：matmul/quant_batch_matmul_v4/op_host/op_api/aclnn_quant_matmul_v5.cpp
- 缺陷描述：yOffset不为nullptr但IsEmpty()为true时（空tensor），仍对其做shape校验导致误报参数非法。空tensor应视为"无此输入"跳过校验。
- 修复模式：条件从`yOffset != nullptr`改为`yOffset != nullptr && !yOffset->IsEmpty()`
- 可审查性：中 — 需要理解空tensor语义
- 审查规则建议：可选tensor参数的null检查应同时考虑IsEmpty()状态；空tensor等价于空指针应作为编码规范

### a9e29824 mxfp8全载tiling中scaleFactor整除为0
- 根因类别：整除结果为零未防护(严重)
- 涉及文件：matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_tiling.cpp
- 缺陷描述：scaleFactor通过整除计算得到，当k,n较大时被除数小于除数导致结果为0，后续运算链中0值传播导致精度失败。
- 修复模式：增加`scaleFactorBMax != 0 && scaleFactorBBase != 0`保护条件；重组条件分支确保被除数>=除数时才使用优化搬运策略
- 可审查性：中
- 审查规则建议：整除运算结果必须做零值防护；tiling计算中的除法应有被除数<除数的边界处理

### 9141d377 fix code-check (conv3d_v2多项)
- 根因类别：代码规范问题(多项)
- 涉及文件：conv/conv3d_v2/多个头文件和实现文件
- 缺陷描述：1) 函数声明与定义参数名不一致(SetDilation/SetStride/SetHF32等)；2) CheckAlgorithmLimit/CheckBiasShape缺少const修饰；3) using namespace在头文件中；4) 未使用的头文件；5) C风格强转改static_cast；6) 变量命名不规范(aoeTiling→convRepoTiling)
- 修复模式：统一参数名 + 添加const + 删除using namespace + 删除未使用include + 规范命名
- 可审查性：高
- 审查规则建议：头文件禁止using namespace；函数声明和定义参数名必须一致

### 9c28c24d fix avgpoolv2grad int32溢出检查不完整
- 根因类别：溢出检查范围不足(严重)
- 涉及文件：pooling/avg_pool_v2_grad/op_host/arch35/avg_pool_v2_grad_tiling_base.cpp
- 缺陷描述：IsGreaterThanInt32Max函数仅检查H*W是否超int32上限，实际应检查batch*channels*H*W总大小。当batch或channels较大时，即使H*W未溢出，总元素数仍可能超int32。
- 修复模式：将`H*W > INT32_MAX`改为`batch*channels*H*W > INT32_MAX`
- 可审查性：高
- 审查规则建议：int32溢出检查应覆盖所有参与计算的维度，不能仅检查部分维度

### bf8e8ab5 打包脚本tar格式选择逻辑错误
- 根因类别：条件逻辑错误
- 涉及文件：scripts/package/common/py/packer.py
- 缺陷描述：原逻辑只要检测到bsdtar就使用ustar格式，但bsdtar不支持长文件名(>100字符)。应优先使用gtar(GNU tar)，仅在无gtar且有bsdtar时才退化到ustar格式。
- 修复模式：改为`if not gtar and bsdtar: tar_format = "ustar"`
- 可审查性：高
- 审查规则建议：条件优先级应与工具能力匹配；打包脚本应测试长文件名场景

### 5a5241b1 文档格式修复 (avgpool/lstm)
- 根因类别：文档格式错误
- 涉及文件：pooling/avg_pool3_d_grad/docs, rnn/single_layer_lstm_grad/docs
- 缺陷描述：avgpool文档产品名称缺少term标签；lstm文档details标签前缺少空行导致渲染异常。
- 修复模式：添加term标签 + 补充空行
- 可审查性：高
- 审查规则建议：Markdown中HTML标签前后需要空行以确保正确渲染

### a715c50a tiling文件位置错误导致编译错误版本
- 根因类别：构建配置/文件路径错误(严重)
- 涉及文件：embedding_dense_grad, gather_nd的tiling相关文件 + 文档
- 缺陷描述：embedding_dense_grad和gather_nd的tiling文件放置位置错误，编译系统找不到正确版本的tiling文件，导致多个用例执行失败。
- 修复模式：修正tiling文件路径 + 更新文档中的产品支持描述
- 可审查性：中
- 审查规则建议：算子tiling文件路径应遵循统一的目录规范；CI应验证tiling文件能被正确编译链接

### 8b2d85b4 matmul类算子文档链接失效
- 根因类别：文档链接错误
- 涉及文件：matmul下多个算子的README.md
- 缺陷描述：文档中链接使用URL编码(%26)导致Markdown渲染失败，应使用原始字符(&)。多个matmul算子README受影响。
- 修复模式：将%26替换为& + 修正错误的示例路径引用
- 可审查性：高
- 审查规则建议：Markdown文件中链接不应使用URL编码；文档CI应验证内部链接可达性

### ca565c0c 36核精度失败(return应为continue)
- 根因类别：控制流语句错误(严重)
- 涉及文件：conv/conv3d_backprop_input_v2/op_kernel/arch35/conv3d_dx_rowc_block.h
- 缺陷描述：多核循环中当核索引超出有效范围时使用`return`直接退出整个函数，但实际应使用`continue`跳过当前迭代继续下一轮循环。在36核环境下，部分核被错误跳过导致输出不完整、精度失败。
- 修复模式：将`return`改为`continue`
- 可审查性：高
- 审查规则建议：循环体内的return语句需仔细审查是否应该是continue/break；多核循环中跳过无效核应用continue而非return

### c63abf32 aclnnRmsNormQuant输出shape校验不完整
- 根因类别：shape校验维度覆盖不全(严重)
- 涉及文件：norm/rms_norm_quant/op_host/op_api/aclnn_rms_norm_quant.cpp + 文档
- 缺陷描述：原有aclnn代码仅检查int32/int4类型输出的最后一维是否满足条件，缺少对前N-1个维度的shape校验。输入x和输出y的前面维度不一致时不会报错，导致运行时可能出现内存越界。
- 修复模式：新增CheckShapeDimWithFrontN函数，在INT32/INT4/其他dtype三个分支中分别添加前N维度的shape一致性校验
- 可审查性：中
- 审查规则建议：输出tensor的shape校验必须覆盖所有维度，不能仅校验尾维度

### 95c8dbac fix Third_Party_Open_Source_Software_List.yaml
- 根因类别：配置文件维护错误
- 涉及文件：Third_Party_Open_Source_Software_List.yaml
- 缺陷描述：三方依赖名称nlohmann/json应为json；缺少protobuf依赖声明。
- 修复模式：修正名称 + 添加protobuf条目
- 可审查性：高
- 审查规则建议：三方依赖清单应与实际使用的依赖保持同步更新

### 501d38ab 文档示例代码重复导致编译错误
- 根因类别：文档示例代码错误
- 涉及文件：matmul/weight_quant_batch_matmul_v2/docs/aclnnWeightQuantBatchMatmulV2.md
- 缺陷描述：A16W4示例代码中有重复的函数定义(AclnnWeightQuantBatchMatmulV2Test)，用户拷贝后编译会出现重定义错误。
- 修复模式：删除重复的代码片段(~112行)
- 可审查性：高
- 审查规则建议：文档示例代码应在CI中进行编译验证

### a7c5a771 回退QuantBatchMatmulV3 A矩阵全载+尾块切分
- 根因类别：功能回退(feature影响其他场景性能)
- 涉及文件：matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_tiling.cpp
- 缺陷描述：PR1762添加的A矩阵全载时尾块切分逻辑增加了`mBlockCnt == 1`限制条件，但该修改影响了950平台其他模板(非纯cube和mx模板)的A全载性能。
- 修复模式：回退整个条件限制，恢复原始的尾块切分逻辑
- 可审查性：N/A — 回退操作
- 审查规则建议：tiling策略修改需在所有受影响的平台和模板上做性能回归测试

### a7630f9f fix TopKTopPSample SetScheduleMode缺失
- 根因类别：多核调度模式未设置(严重)
- 涉及文件：index/top_k_top_p_sample/op_host, index/top_k_top_p_sample_v2/op_host
- 缺陷描述：TopKTopPSample/V2算子使用SyncAll进行核间同步，但未设置SetScheduleMode为BATCH_MODE(1)。缺少此设置可能导致核间同步行为不正确。
- 修复模式：在isNeedLogits/isNeedSampleResult条件下添加`context->SetScheduleMode(BATCH_MODE)`
- 可审查性：中
- 审查规则建议：使用SyncAll的算子必须设置SetScheduleMode为BATCH_MODE；CR checklist应包含同步原语与调度模式的匹配检查

### 2e1ef7e1 aclnnIndexAddV2 UT assert过时
- 根因类别：UT未跟随功能更新
- 涉及文件：index/inplace_scatter_add/tests/ut/op_host/test_aclnn_index_add_v2.cpp
- 缺陷描述：aclnnIndexAddV2接口功能扩展后已支持新的数据类型组合，但UT中期望仍是旧的错误码(ACLNN_ERR_INNER_NULLPTR)，实际应返回ACL_SUCCESS。原UT还被注释掉了。
- 修复模式：取消注释 + 将期望值改为ACL_SUCCESS
- 可审查性：高
- 审查规则建议：接口功能扩展时必须同步更新相关UT的期望值；UT中不应有被注释掉的断言

### 644f498b 文档demo路径错误 (extend_conv2d/quant_conv3d)
- 根因类别：文档路径错误
- 涉及文件：conv/extend_conv2d/README.md, conv/quant_conv3d/README.md
- 缺陷描述：README中demo路径指向`./examples/`但实际文件在`./examples/arch35/`下，导致链接失效。
- 修复模式：路径中添加arch35子目录
- 可审查性：高
- 审查规则建议：文档中的相对路径引用应在CI中验证文件存在性

### 853dcb95 fix codecheck of batch_norm_v3
- 根因类别：代码规范问题(C风格强转+参数名不一致)
- 涉及文件：norm/batch_norm_v3/op_host/多个文件
- 缺陷描述：1) 使用C风格强转`(int64_t)x`而非`static_cast<int64_t>(x)`；2) 函数声明参数名(theLeastAPerCore)与定义(aFactor)不一致。
- 修复模式：改为static_cast + 统一参数名
- 可审查性：高
- 审查规则建议：禁止C风格类型转换(使用clang-tidy google-readability-casting规则)

### 2045ef0a Conv3DTransposeV2 kernel split全载wi=2时AIC ERROR
- 根因类别：资源释放无效地址(严重)
- 涉及文件：conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_kernel_split_tiling.cpp
- 缺陷描述：kernel split全载且wi=2场景下，存在单计算轮次中B1Tensor全部跳过无需加载的情况，但全载最后仍尝试释放B1Tensor地址，该地址无效导致AIC ERROR。
- 修复模式：在IsBaseShapeFitKernelSplitHW中增加`kSCoutFullLoad_ && wi <= 2`条件返回false，屏蔽该场景不走kernel拆分
- 可审查性：低 — 需要理解conv kernel split的buffer管理机制
- 审查规则建议：全载模式下跳过加载的buffer不应在最后被释放；kernel split的边界条件应有专门的UT覆盖

### 4482238c revert logsigmoid impl
- 根因类别：功能回退
- 涉及文件：activation/log_sigmoid/下全部文件(删除)
- 缺陷描述：logsigmoid算子实现被完整回退，删除了proto、tiling、kernel等所有文件。
- 修复模式：revert整个feature
- 可审查性：N/A — 回退操作
- 审查规则建议：新算子合入前应通过完整的集成测试验证

### f84fb52d indexputv2确定性分支broadcast函数调用错误
- 根因类别：函数调用错误(严重)
- 涉及文件：index/index_put_v2/op_api/aclnn_index_put_impl.cpp
- 缺陷描述：确定性分支中调用了IndicesBroadcast函数，但该函数是非确定性版本。确定性分支应调用IndicesBroadcastUndeter，两者在scalar tensor的broadcast处理逻辑上不同，导致进入broadcastto的infershape时报错。
- 修复模式：将IndicesBroadcast替换为IndicesBroadcastUndeter
- 可审查性：中 — 需要理解确定性/非确定性分支的差异
- 审查规则建议：确定性分支中只应调用带Undeter后缀的函数；代码审查时关注确定性/非确定性路径是否使用了正确的函数变体

## 第5轮 (commit 81-100)

### 6f8ad87c 头文件兼容性适配 (kernel_basic_intf.h)
- 根因类别：版本兼容性缺失(严重)
- 涉及文件：common/inc/op_kernel/platform.h + matmul多个kernel头文件
- 缺陷描述：`#include "kernel_basic_intf.h"`在CANN 8.5及以下版本中不存在，导致编译失败。需要通过宏`ASC_DEVKIT_MAJOR >= 9`条件编译，低版本退回`kernel_operator.h`。
- 修复模式：在所有include处添加`#if ASC_DEVKIT_MAJOR >= 9`条件编译
- 可审查性：高
- 审查规则建议：引入新SDK头文件时必须评估向后兼容性；使用条件编译保护非通用头文件

### 7fa2b4eb embedding_bag README列宽修复+链接失效
- 根因类别：文档错误
- 涉及文件：embedding_bag的README.md
- 缺陷描述：README表格列宽不当导致显示异常 + 文档链接失效。
- 修复模式：调整列宽 + 修正链接
- 可审查性：高
- 审查规则建议：文档CI中加入链接有效性检查

### 34217db7 Conv3DBackpropInputV2 cin超datacopy stride上限
- 根因类别：硬件指令参数上限校验缺失(严重)
- 涉及文件：conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_inner_product_tiling.cpp
- 缺陷描述：当cin大于65535时，前置transpose的datacopy指令stride参数超过16位上限(65535)，导致精度失败。tiling未检查cin是否超出硬件指令限制。
- 修复模式：在CheckVecTransEnable中增加`runInfo_.dedx_cin > 65535`条件，超限时禁用前置transpose
- 可审查性：中 — 需要了解datacopy指令的参数上限
- 审查规则建议：使用datacopy的stride参数前必须校验不超过65535(16位上限)；tiling决策应考虑硬件指令参数范围

### c04e08dc DLQBMM算子日志缺少右括号
- 根因类别：日志格式错误
- 涉及文件：matmul/dual_level_quant_batch_matmul/op_host/op_tiling/dual_level_quant_batch_matmul_checker.cpp
- 缺陷描述：ToShapeString函数构造shape字符串时缺少闭合的']'字符，导致异常拦截时维测信息格式不完整。
- 修复模式：添加`shapeStr.push_back(']')`
- 可审查性：高
- 审查规则建议：字符串构造函数应确保配对符号(括号/方括号)完整

### f9e38e90 GroupNormSiluQuant文档格式修复
- 根因类别：文档格式错误
- 涉及文件：GroupNormSiluQuant算子文档
- 缺陷描述：文档中存在格式问题影响渲染。
- 修复模式：修正格式
- 可审查性：高
- 审查规则建议：文档发布前预览渲染效果

### adec83fb CI脚本shell语法错误导致编译失败不报错
- 根因类别：shell语法错误(严重)
- 涉及文件：scripts/ci/check_example.sh, check_kernel_ut.sh, check_pkg.sh (4处)
- 缺陷描述：`[ $status -ne 0]`缺少']'前的空格，bash语法错误导致条件判断失效。编译/UT执行失败时，CI流水线仍显示成功，掩盖了实际错误。
- 修复模式：改为`[ $status -ne 0 ]`(添加空格)
- 可审查性：高
- 审查规则建议：CI脚本应使用shellcheck静态分析；`[`和`]`前后必须有空格

### 305ad94a Embeddingbag paddingIdx重复处理导致精度失败
- 根因类别：上下游处理逻辑重复(严重)
- 涉及文件：index/embedding_bag/op_host/embedding_bag_regbase_tiling.cpp
- 缺陷描述：上层torch框架已将paddingIdx从负值转换为正值(如-1→numEmbeddings-1)，tiling中再次做`paddingIdx + numEmbeddings`导致索引超出范围，精度失败。
- 修复模式：删除tiling中重复的paddingIdx负值处理逻辑
- 可审查性：中 — 需要了解上下游职责边界
- 审查规则建议：算子内部不应重复上层框架已完成的参数预处理；paddingIdx语义应在接口文档中明确标注是否已预处理

### 7b870377 Conv3DTransposeV2 B矩阵全载场景释放无效地址
- 根因类别：资源释放无效地址(严重)
- 涉及文件：conv/conv3d_backprop_input_v2/op_kernel/arch35/convolution_3d_backprop/conv3d_bp_kernel_split_func.h
- 缺陷描述：B矩阵全载场景下，整轮计算都未加载B1Tensor但FreeB1Tensor仍尝试释放，导致释放无效地址引发AIC ERROR。与commit #78(wi=2场景)属同一类问题的不同触发路径。
- 修复模式：在FreeB1Tensor中增加`isLoadB1_ && !isFreeB1_`条件，未加载则跳过释放
- 可审查性：低 — 需要理解全载场景buffer生命周期
- 审查规则建议：Free操作前必须验证对应资源已被成功分配/加载；buffer释放逻辑应与加载逻辑对称

### f4ad97f3 conv nDim对齐计算与API侧不一致
- 根因类别：kernel与host侧对齐计算不一致(严重)
- 涉及文件：conv/common/op_kernel/arch35/conv_common.h + conv2d_v2多个kernel文件 + conv3d_v2 tiling
- 缺陷描述：kernel侧计算nDim时，N0对齐的方式与API侧不一致。原CalcNDimDataWeightNZ函数对齐逻辑错误(先CeilDiv再AlignB)，应该是对wholeDim先AlignB再CeilDiv，保证L1数据读取不会错位。
- 修复模式：重写为CalcNDimDataAlign函数，统一对齐逻辑：`AlignB(CeilDiv(AlignB(wholeDim, N0), dim), N0)`
- 可审查性：低 — 需要理解conv多核分片的对齐要求
- 审查规则建议：kernel与host的维度对齐计算必须使用相同公式；对齐计算应抽取为公共函数统一维护

### 79089b0e quant_conv3d dequant flag异常(biasType占位类型错误)
- 根因类别：占位类型选择不当(严重)
- 涉及文件：conv/quant_conv3d/op_kernel/quant_conv3d.cpp
- 缺陷描述：当DTYPE_BIAS未定义时(无bias场景)，biasType使用`half`作为编译占位类型，但half类型影响了dequant flag的判断逻辑，导致异常行为。
- 修复模式：将占位类型从`half`改为`int32_t`，消除对dequant flag的干扰
- 可审查性：中
- 审查规则建议：编译占位类型不应影响运行时逻辑判断；占位类型应选择与正常路径一致的默认类型

### 7e28fed7 QBMMv4文档错误修复
- 根因类别：文档错误
- 涉及文件：QBMMv4算子文档
- 缺陷描述：文档内容有误。
- 修复模式：修正文档
- 可审查性：高
- 审查规则建议：算子文档应由开发和测试双方审核

### 4d315436 aclnnQuantBatchMatmulV4 k0默认值条件反转
- 根因类别：条件判断逻辑反转(严重)
- 涉及文件：matmul/quant_batch_matmul_v3/op_api/aclnn_quant_matmul_v4.cpp (2处)
- 缺陷描述：加入int4类型后，k0值选择条件从`DT_INT8 ? INT8_K0 : INT4_K0`变为了默认走int4路径。正确逻辑应为`DT_INT4 ? INT4_K0 : INT8_K0`(int4是特殊分支，其他类型走int8默认值)。
- 修复模式：将条件从`== DT_INT8`反转为`== DT_INT4`
- 可审查性：高
- 审查规则建议：新增类型分支时，默认分支(else)应保持原有行为不变；三元表达式的条件应使"特殊情况"作为true分支

### e79d67ac AddLayerNormQuant bias为null时binary匹配失败
- 根因类别：simplified_key生成错误(严重)
- 涉及文件：norm/add_layer_norm_quant/op_host/add_layer_norm_quant_tiling.cpp + binary.json
- 缺陷描述：生成simplified_key时，bias字段直接使用x1的dtype代替(硬编码)，未检查bias是否实际存在。当bias为null时，key中bias类型应为-1而非x1的类型，导致匹配不到正确的binary。
- 修复模式：新增biasDtype变量，通过GetOptionalInputDesc检查bias是否存在来设置实际dtype或-1 + 补充bias_null场景的binary配置
- 可审查性：中
- 审查规则建议：simplified_key中可选参数的dtype应根据实际存在性设置，不能用其他参数替代；binary配置应覆盖所有可选参数组合

### df20f36e silu_mul算子定义与kernel不一致导致编译失败
- 根因类别：算子接口定义与kernel实现不匹配(严重)
- 涉及文件：experimental/activation/silu_mul/op_kernel/silu_mul.cpp, silu_mul.h + README.md
- 缺陷描述：算子从单输入input改为双输入(x,y)→输出z，但kernel入口函数仍为`silu_mul(input, output, ...)`，参数数量和语义均不匹配，无法编译出binary。内部处理也有错误：`d = lastDimSize / 2`应为`d = lastDimSize`。
- 修复模式：kernel入口改为`silu_mul(x, y, z, workspace, tiling)` + Init参数对应修改 + 修正d计算 + 更新README
- 可审查性：高
- 审查规则建议：算子定义(proto)修改后必须同步更新kernel入口函数签名和实现

### e3f1fdc5 C04下n轴绑核决策缺陷
- 根因类别：模式适配遗漏(严重)
- 涉及文件：conv/common/op_host/op_tiling/arch35/conv_base_blockdim_decision.cpp + conv2d_v2多个tiling文件
- 缺陷描述：1) CheckL1SizeLimitsKernelFullLoad未考虑C04模式下kAL1min/kBL1min应使用C04_CIN_SIZE而非k0，导致L1空间计算错误；2) n轴绑核mix策略在C04下不适用但未排除；3) 带宽系数未区分C04+NHWC场景。26个用例受影响。
- 修复模式：CheckL1SizeLimitsKernelFullLoad增加isC04参数区分计算 + C04下禁用BlockDimFactorMix + 添加C04+NHWC的带宽系数
- 可审查性：低 — 需要深入理解C04格式的特殊约束
- 审查规则建议：新增特殊格式(C04等)时，必须全面审查所有tiling决策路径是否需要适配

### 65140bdf 移除不支持的debug配置参数
- 根因类别：构建配置错误
- 涉及文件：cmake/gen_ops_info.cmake + scripts/kernel/binary_script/build_binary_opc_gen_task.sh
- 缺陷描述：`--op_debug_config=debug`参数不被opc支持，导致opc error时不显示报错信息。同时缺少`ASCEND_SLOG_PRINT_TO_STDOUT=1`环境变量。
- 修复模式：删除不支持的debug参数 + 添加日志输出环境变量
- 可审查性：高
- 审查规则建议：构建脚本中的工具参数应与工具文档保持同步；CI配置应确保错误信息可见

### 4512b92e op_api_list多余空行修复
- 根因类别：配置文件格式错误
- 涉及文件：op_api_list文件
- 缺陷描述：多余空行导致格式异常。
- 修复模式：删除多余空行
- 可审查性：高
- 审查规则建议：配置文件应有格式验证

### f89a1872 DequantSwigluQuant shape校验逻辑错误
- 根因类别：布尔逻辑错误(严重)
- 涉及文件：quant/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling_arch35.cpp + 文档
- 缺陷描述：weight_scale与group_index匹配校验条件`(A != B) && (C != D)`应为`!(A == B && C == D)`。原条件在A!=B但C==D时仍会通过校验，未能正确拦截非法输入。同时bias维度为1时缺少group_index存在性检查。
- 修复模式：修正布尔条件为`!(A == B && C == D)` + 增加bias维度1与group_index互斥检查
- 可审查性：高
- 审查规则建议：复合否定条件建议使用De Morgan定律重写为肯定形式再取反(更易理解)；shape校验应有正/负用例UT覆盖

### affc1f58 magic numbers替换
- 根因类别：代码可维护性(magic number)
- 涉及文件：算子代码
- 缺陷描述：代码中使用magic number。
- 修复模式：定义命名常量替换
- 可审查性：高
- 审查规则建议：禁止裸数字常量

### cf84e222 AddRmsNorm同步问题+空指针检查
- 根因类别：流水线同步错误 + 空指针检查缺失(严重)
- 涉及文件：norm/add_rms_norm/op_host/op_api/aclnn_add_rms_norm.cpp + op_kernel/add_rms_norm_merge_n.h
- 缺陷描述：1) PipeBarrier<PIPE_V>位置错误 — 放在FreeTensor之后，导致Cast计算未完成就释放了输入buffer，数据竞争；2) BF16路径CopyOut缺少V_MTE3/MTE3_V事件同步，向量计算与数据传输之间无依赖保障；3) yComputeOut返回值未做空指针检查。
- 修复模式：PipeBarrier移到FreeTensor之前 + 添加V_MTE3/MTE3_V事件获取-等待-释放序列 + 添加CHECK_RET空指针检查
- 可审查性：低 — 需要理解AscendC流水线屏障和硬件事件模型
- 审查规则建议：PipeBarrier必须在FreeTensor之前；CopyOut(MTE3)前后需要对应的V_MTE3/MTE3_V事件同步；工厂函数返回值必须做null检查

## 第6轮 (commit 101-120)

### 36c945ec batch_norm_backward_reduce分支判断+workspace设置错误
- 根因类别：平台判断逻辑错误 + workspace设置遗漏(严重)
- 涉及文件：norm/sync_batch_norm_backward_reduce/op_host/op_api/aclnn_batch_norm_backward_reduce.cpp
- 缺陷描述：1) GetDtypeSupportList使用SocVersion范围判断(910B~910E)不准确，改为NpuArch枚举判断；2) 空tensor分支下workspaceSize被设为0，但executor实际有workspace需求，应使用GetWorkspaceSize()获取真实值。
- 修复模式：平台判断改为NpuArch枚举 + workspaceSize使用executor->GetWorkspaceSize()
- 可审查性：高
- 审查规则建议：平台判断应使用NpuArch而非SocVersion范围；空tensor路径不能跳过workspace设置

### d6acf37e unique_consecutive缺少ascend950 binary配置
- 根因类别：binary配置缺失
- 涉及文件：index/unique_consecutive/op_host/config/ascend950/(新增文件)
- 缺陷描述：unique_consecutive算子缺少ascend950平台的binary.json配置文件，导致该平台无法使用。
- 修复模式：新增ascend950平台的binary配置文件
- 可审查性：高
- 审查规则建议：新增算子或新平台时应同步添加所有目标平台的binary配置

### aa11f503 AdaptiveAvgPool3d缺少shape维度值<=0拦截
- 根因类别：输入校验缺失
- 涉及文件：pooling/adaptive_avg_pool3d/op_api/aclnn_adaptive_avg_pool3d.cpp + 文档
- 缺陷描述：self或out的shape某个维度值小于等于0时未拦截，可能导致后续计算异常。
- 修复模式：增加shape维度值>0的校验 + NC维度一致性检查 + 更新文档错误码说明
- 可审查性：高
- 审查规则建议：aclnn接口入口应校验所有shape维度值为正数

### 9b0eb33e cross_entropy_loss vfReduceMax偶现精度问题
- 根因类别：向量指令使用不当(严重)
- 涉及文件：loss/cross_entropy_loss/op_kernel/arch35/cross_entropy_loss_full_load.h
- 缺陷描述：vfReduceMax在R轴大于256byte时偶现取不到真正的最大值，导致精度问题。原实现区分cNum<vfLen和>=vfLen两种路径，逻辑复杂且有缺陷。
- 修复模式：重写ReduceMax逻辑，统一处理路径，使用更可靠的寄存器操作方式
- 可审查性：低 — 需要理解AscendC向量指令的行为细节
- 审查规则建议：ReduceMax等归约操作需要对大于256byte的场景做专门验证

### 624b4334 文档错别字("只有"→"只要")
- 根因类别：文档错误
- 涉及文件：matmul下多个算子文档
- 缺陷描述：确定性计算说明中"只有输入相同"应为"只要输入相同"，语义截然不同。
- 修复模式：修正措辞
- 可审查性：高
- 审查规则建议：技术文档中逻辑连接词(只有/只要/如果/只要)需要准确使用

### 2b10b6c4 embedding bag未初始化内存导致core dump
- 根因类别：未初始化内存(严重)
- 涉及文件：index/embedding_bag/op_host/embedding_bag_infershape.cpp
- 缺陷描述：mean模板使用了未初始化的内存区域，导致地址越界引发core dump。geir通路和精度测试场景下触发。
- 修复模式：重写infershape函数，在使用内存前先初始化；拆分为独立的InferShape4OutputSupport等子函数
- 可审查性：中
- 审查规则建议：所有内存区域使用前必须初始化；infershape函数应考虑动态shape(-1)场景

### 1de0b7a4 experimental算子冒烟环境rpath失效
- 根因类别：构建/部署配置错误
- 涉及文件：build.sh + experimental/activation/relu_v2/op_host/relu_v2_def.cpp
- 缺陷描述：冒烟环境存在自定义算子包的共享库，rpath失效导致链接到错误版本。需要显式设置LD_LIBRARY_PATH。
- 修复模式：在build_single_example中添加LD_LIBRARY_PATH设置 + 执行后恢复原值
- 可审查性：高
- 审查规则建议：自定义算子的example构建应显式设置库搜索路径，不应依赖rpath

### 06393c8e CtcLossBackward上边界用例aicore_error
- 根因类别：边界条件缺失(严重)
- 涉及文件：loss/ctc_loss_v2_grad/op_kernel/arch35/ctc_loss_v2_grad.h
- 缺陷描述：当targetLength==0时，代码仍访问`2*targetLength-1`即-1位置的targetPrime，导致数组越界引发aicore_error。
- 修复模式：在访问`2*targetLength-1`前增加`targetLength > 0`条件判断
- 可审查性：高
- 审查规则建议：涉及`length-1`下标访问时必须检查length>0；上边界用例(length=0/1)应有专门UT

### 74d43ef3 scatter_sub缺少流水同步导致竞态
- 根因类别：流水线同步缺失(严重)
- 涉及文件：index/scatter_add/op_kernel/arch35/scatter_add_simd_support_atomicadd.h
- 缺陷描述：ScatterSub操作中，从GM拷贝数据到UB前缺少MTE3_MTE2同步，执行减法前缺少MTE2_V同步。数据在MTE/V/S之间流动时序不当引发竞态，导致相同输入产生不同输出。
- 修复模式：在DataCopy前添加MTE3_MTE2事件同步 + 在NegateUpdate前添加MTE2_V事件同步
- 可审查性：低 — 需要理解AscendC流水线同步模型
- 审查规则建议：DataCopy(MTE2)前需确保前序MTE3完成；向量运算(V)前需确保MTE2数据就绪；ScatterAdd/Sub的非标量路径需要完整的事件同步链

### acadb4c7 repeat_interleave int32溢出
- 根因类别：整数溢出(严重)
- 涉及文件：index/repeat_interleave/op_host/arch35/repeat_interleave_tiling_normal.cpp/.h
- 缺陷描述：当repeatSum(输出总元素数)超过INT32_MAX时，使用int32计算导致溢出。原代码未区分是否需要int64。
- 修复模式：新增UseInt64()函数检查yShape总大小是否超INT32_MAX；根据结果选择int32或int64计算路径(不同tiling key)
- 可审查性：高
- 审查规则建议：涉及元素数量计算的变量应默认使用int64_t；tiling中应有大shape的int32溢出检查

### f992c37b 文档拼写错误(HFLOAT32)
- 根因类别：文档错误
- 涉及文件：aclnnConvTbc.md
- 缺陷描述：HFLOAT32拼写错误。
- 修复模式：修正拼写
- 可审查性：高
- 审查规则建议：文档spell check

### 05db401f 多项修复(simplified_key+indexSize计算+paramType)
- 根因类别：配置错误 + 计算逻辑错误(严重)
- 涉及文件：index_fill配置 + gather_v2 op_api + fused_cross_entropy_loss配置 + 文档
- 缺陷描述：1) index_fill的simplified_key.ini错误指向了FusedCrossEntropyLossWithMaxSum算子名；2) gather_v3的indexSize计算中多乘了GetSizeByDataType(应为元素个数而非字节数)；3) fused_cross_entropy_loss的json中weight参数paramType应为optional而非required。
- 修复模式：修正算子名 + 修正计算公式 + 修正paramType
- 可审查性：高
- 审查规则建议：simplified_key配置文件中的算子名必须与实际算子匹配；元素个数计算不应包含dtype大小

### 20350c7d 文档typo修复
- 根因类别：文档错误
- 涉及文件：aclnn返回码文档 + op_debug_prof文档
- 缺陷描述：文档中存在多处typo。
- 修复模式：修正typo
- 可审查性：高
- 审查规则建议：文档CI应集成spell check工具

### 420a1cdd 文档修复(FusedLinearCrossEntropyLossGrad)
- 根因类别：文档错误
- 涉及文件：FusedLinearCrossEntropyLossGrad文档
- 缺陷描述：文档内容有误。
- 修复模式：修正文档
- 可审查性：高
- 审查规则建议：文档修改应由算子开发者审核

### 695d75ef bmm opinfo fp16/bf16→fp32输出条件错误
- 根因类别：条件判断过宽(严重)
- 涉及文件：matmul/common/op_host/op_api/batch_matmul_util.cpp
- 缺陷描述：enableFp16Bf16InFp32Out标志对所有BatchMatMul调用都生效，但实际应仅限Baddbmm接口且仅限A2/A3平台。非Baddbmm接口走了fp32输出路径导致精度失败。另外K==1时不应先Cast到float再Mul。
- 修复模式：CreateBatchMatmulOpInfo增加isBaddbmm参数；条件中加入平台和isBaddbmm判断；K==1分支去除Cast
- 可审查性：中
- 审查规则建议：特定接口的优化路径应严格限定调用来源；新增条件分支时验证对现有调用链的影响

### 958f074a 统计耗时脚本变量命名错误
- 根因类别：变量名引用错误
- 涉及文件：scripts/ci/analyze_ops_time.py
- 缺陷描述：变量先赋值给main_func后续却用op变量名引用，导致Python NameError。
- 修复模式：统一变量名为op
- 可审查性：高
- 审查规则建议：Python脚本应启用pylint/flake8检查未定义变量

### d674d0da 编译warning修复(未使用变量+日志拼写)
- 根因类别：编译warning + 日志拼写错误
- 涉及文件：quant/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling_arch35.cpp
- 缺陷描述：1) DoOpTiling中声明了未使用的`ge::graphStatus ret`变量；2) 日志中"exist"拼写为"exit"。
- 修复模式：删除未使用变量 + 修正拼写
- 可审查性：高
- 审查规则建议：编译warning应在CI中视为error

### c57524eb scatter_nd_add生成重复二进制文件
- 根因类别：binary配置错误
- 涉及文件：index/scatter_nd_add/op_host/config/ascend950/scatter_nd_add_binary.json
- 缺陷描述：binary.json中use_locking属性值设为false(固定值)，导致use_locking=true和false的情况生成同一个binary文件名。应设为null表示不区分该属性。
- 修复模式：将use_locking的value从false改为null
- 可审查性：高
- 审查规则建议：binary配置中不影响kernel行为的属性值应设为null；binary文件不应出现重复

### 557ea7a7 覆盖率脚本权限问题
- 根因类别：CMake配置错误
- 涉及文件：tests/ut/CMakeLists.txt
- 缺陷描述：CMake custom_command中直接执行脚本路径但未用bash调用，导致权限不足报错。同时缺少POST_BUILD关键字。
- 修复模式：命令前加bash + 添加POST_BUILD
- 可审查性：高
- 审查规则建议：CMake custom_command中调用shell脚本应使用bash显式调用

### 640c1939 文档demo内存分配size计算错误
- 根因类别：文档示例代码错误
- 涉及文件：quant/dynamic_quant_v2/docs/aclnnDynamicQuantV2.md
- 缺陷描述：CreateAclTensor中使用`sizeof(T)`计算内存大小，但T为uint16_t(fp16的host表示)而实际设备端需要按float大小分配，导致内存不足。
- 修复模式：将`sizeof(T)`改为`sizeof(float)`
- 可审查性：高
- 审查规则建议：文档示例代码中内存分配应注明host/device数据类型差异；示例代码应在CI中编译验证

## 第7轮 (commit 121-140)

### 1532f915a473f52a7fccc9968eca2cc20c61e0f2 inpalceindexadd 分核逻辑修复
- 根因类别：整数除法导致零值（边界条件缺陷）
- 涉及文件：index/inplace_index_add/op_host/arch35/inplace_index_add_simd_sort_tiling.cpp
- 缺陷描述：ubIndexFactor_通过整数除法halfUbSize/(...)计算，大shape场景分母大于分子导致结果为0。下方while(--ubIndexFactor_)循环中0先自减为-1，循环无法正常退出或产生错误行为。两处DoOpTilingSplitAfterSort和DoOpTilingSplitPreSort都有此问题。
- 修复模式：边界值补偿——在整数除法结果后+1，配合下方--循环形成"先试后退"的搜索模式
- 可审查性：中
- 审查规则建议：整数除法结果作为循环初始值时，需检查除法结果为0的情况是否有保护；关注while(--x)模式中x初始值可能为0的问题

### 90c51245d3d36ab4d7bf0705832c5678404817a5 fix opHost\opGraph UT
- 根因类别：UT修复（编译错误+构建配置缺陷）
- 涉及文件：tests/ut/op_graph/test_op_graph_main.cpp, tests/ut/op_host/CMakeLists.txt
- 缺陷描述：(1) 预编译宏误写#elif define(__x86_64__)应为#elif defined(__x86_64__)，缺d后缀导致编译错误；(2) CMakeLists中$<TARGET_EXISTS:...>生成器表达式在目标不存在时产生空源文件列表使add_library失败。
- 修复模式：(1) 拼写纠正 (2) 将运行时生成器表达式改为configure阶段条件判断+fallback
- 可审查性：高
- 审查规则建议：#elif define(应自动标记为疑似错误（正确形式是defined）；CMake中$<TARGET_EXISTS:>嵌套$<TARGET_OBJECTS:>需检查目标不存在时的空列表问题

### a39e185487260618620b98539b3f05c3f798db4a revert small shape optimization in mmv3
- 根因类别：优化引入的正确性缺陷（workspace脏数据）
- 涉及文件：matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp, matmul/mat_mul_v3/tests/ut/op_host/test_matmul_v3_tiling.cpp
- 缺陷描述：先前为小shape添加V_PARALELL_ND2NZ优化分支，该模板可能导致workspace中存在脏数据参与计算产生精度问题。
- 修复模式：revert回退——删除有问题的优化路径，回退到V_HEAD_ND2NZ通用路径
- 可审查性：低
- 审查规则建议：新增优化路径需验证workspace初始化/清理逻辑是否覆盖所有场景；涉及ND2NZ格式转换的新模板应有workspace脏数据检查测试用例

### bf2ac324b478f0b492161646db1638a389b17e88 修复 aclnnaddmv 接口文档描述问题
- 根因类别：文档错误
- 涉及文件：matmul/addmv/docs/aclnnAddmv.md
- 缺陷描述：API文档中self/mat/vec/out维度列缺失具体维度数（用-代替），beta参数format/维度/是否必须列描述有误，out数据类型多列了BOOL。
- 修复模式：文档内容纠正
- 可审查性：高
- 审查规则建议：接口文档中tensor参数维度字段不应为-，应有明确维数描述；文档数据类型列表应与代码实际支持列表同步校验

### 9500a44b04e514bf5fce53460fee6377e4afc8f6 [MatMul] 修复大K场景因stepka错误调整引入的性能劣化问题
- 根因类别：优化代码作用域错误（未隔离API层级）
- 涉及文件：matmul/fused_mat_mul/op_host/op_tiling/arch35/fused_matmul_asw_basic_tiling.cpp/.h, matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_asw_tiling.cpp, matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_aswt_tiling.cpp
- 缺陷描述：基类MatMulV3AswTiling::DoOpTiling()中调整了stepKa/stepKb计算方式，但此基类被基础API和高阶API(FusedMatMul)共同继承。高阶API有自己的depthA1/depthB1参数，基类的stepK调整改变了depthA1导致高阶API性能劣化。
- 修复模式：职责下沉——将特定于子类的逻辑从基类移至子类，通过override隔离
- 可审查性：中
- 审查规则建议：修改继承体系中基类的tiling逻辑时，必须验证所有子类场景的性能；stepK/depthA1/depthB1之间存在关联依赖，修改stepK时应同步检查depth是否需要更新

### 8d8b3dbbf62ff383a238e3f9868bfbdbebb484c8 修复优化ScatterNdAdd算子
- 根因类别：多类复合缺陷（芯片判断硬编码+数据类型传参错误+buffer计算错误+边界条件修复）
- 涉及文件：index/scatter_nd_add/op_api/scatter_nd_add.cpp, index/scatter_nd_add/op_host/arch35/scatter_nd_add_tiling_base.cpp/.h, index/scatter_nd_add/op_kernel/arch35/scatter_nd_add_common.h/.h
- 缺陷描述：(1) GetSocVersion()>=ASCEND950硬编码判断芯片改为IsRegbase()通用接口；(2) 多处将废弃字段indicesType_传给GetSortTmpSize()，应改为indiceDtype_；(3) buffer大小从halfUbSize改为UbSize；(4) deterministic分支indicesFactor_>indicesAxis_改为>eachCoreIndexCount_修正分核边界；(5) 新增排序条件排除INT64；(6) 新增isOpti_优化分支。
- 修复模式：综合修复——架构抽象化+参数纠正+内存计算修正+分核边界修正+新优化路径
- 可审查性：低
- 审查规则建议：芯片型号硬编码GetSocVersion()>=ASCENDXXX应统一使用架构抽象接口；dtype变量重命名/废弃时应全文搜索替换；单PR混合bugfix和新feature应拆分

### 7ff1df8c981ad22a41a34ff36c98db2c1e8e3b57 修复aclnnInplaceMaskedScatter算子不同芯片架构下分支判断逻辑错误
- 根因类别：条件判断取反（逻辑错误）
- 涉及文件：index/masked_scatter/op_api/aclnn_masked_scatter.cpp
- 缺陷描述：IsRegbase()判断写反——原代码if(IsRegbase())走l0op::MaskedScatter路径，else走ProcessBroadcast路径，实际应反转。同时修复else分支传入selfRef（未contiguous）改为selfRefContiguous。
- 修复模式：布尔条件取反 + 参数纠正
- 可审查性：高
- 审查规则建议：芯片架构分支选择逻辑必须有注释说明每个分支对应哪种架构；IsRegbase()/!IsRegbase()分支应在review中重点关注

### b4c5ba659f914fbf60a099b1b30b51e2b9bd424f 回退extern aclnn Inner修改
- 根因类别：构建系统改动引入的兼容性问题
- 涉及文件：CMakeLists.txt, cmake/opbuild.cmake, cmake/package.cmake, cmake/variables.cmake, 3个op_api .cpp文件
- 缺陷描述：将aclnnInner的extern声明改为通过自动生成头文件#include引入，但构建顺序或路径不正确导致编译失败。回退后恢复到extern显式声明方式。
- 修复模式：revert回退
- 可审查性：中
- 审查规则建议：将extern声明改为头文件include时需确保构建顺序正确；构建系统改动应有完整端到端编译验证（增量+全量）

### c1469b683950d49e2e5c583f4732442a256d08f4 bugfix for kirin：quant_batch_matmul_v4 && transpose_batch_mat_mul && weight_quant_batch_matmul_v2 && batch_norm_v3 && max_pool3d_with_argmax_v2
- 根因类别：新芯片平台兼容性缺陷（端云不兼容+编译宏不完整+配置冗余）
- 涉及文件：16个文件，涉及5个算子
- 缺陷描述：Kirin芯片适配中：(1) 三个算子端云无法兼容需删除kirin配置；(2) transpose_batch_mat_mul的Fixpipe指令需增加__NPU_ARCH__==3003||3113条件（6处）；(3) max_pool3d_with_argmax_v2的binary.json残留已删除attr。
- 修复模式：平台配置清理+编译条件扩展
- 可审查性：中
- 审查规则建议：新芯片适配时应有端云兼容性检查清单；__DAV_C220_CUBE__等硬件宏应有统一平台抽象层；binary.json的attr应与代码同步

### 33fcbea47ec66d3c4c6096aa07bd920d5e645d7d fix third_party download and install path
- 根因类别：构建配置缺陷（第三方依赖路径不统一）
- 涉及文件：CMakeLists.txt, build.sh, cmake/variables.cmake, 多个third_party cmake文件
- 缺陷描述：CANN_3RD_PKG_PATH和CANN_3RD_LIB_PATH两个变量用途重叠路径不统一；abseil-cpp.cmake硬编码路径而不用变量；protobuf缺少缓存路径支持；build.sh清理路径硬编码。
- 修复模式：路径统一化+变量归一化+冗余清理
- 可审查性：高
- 审查规则建议：构建脚本中第三方依赖路径应统一通过变量引用禁止硬编码；同一用途不应有多个变量

### 97caad29deca72b227bda63722a5a203c1ad8858 修复UT工程bug
- 根因类别：构建脚本缺陷
- 涉及文件：build.sh
- 缺陷描述：连续执行不同算子UT时，ai_core目录下json文件未被清理，导致核函数文件名读取到旧json编译出错。
- 修复模式：在build_ut()进入构建目录后增加rm -rf删除ai_core下所有json文件
- 可审查性：高
- 审查规则建议：构建脚本增量编译逻辑应检查跨任务状态污染；依赖生成文件的构建流程应有清理或版本校验机制

### 7651c0fd29deb39a45143b112c86eb76de3cb9ce msda修复numPoints=1时找不到kernel问题，增加numLevels维度一致性校验
- 根因类别：边界条件处理缺陷 + 输入校验缺失
- 涉及文件：multi_scale_deformable_attn_function_tiling.cpp, aclnn_multi_scale_deformable_attn_function.cpp, multi_scale_deformable_attn_function_def.cpp
- 缺陷描述：(1) numPoints=1时条件numPoints%2==0||numPoints==1导致走入noTranspose路径但该场景未适配tilingkey；(2) 三个输入tensor的numLevels维度未做一致性校验；(3) ascend910残留配置。
- 修复模式：修改条件判断 + 新增输入校验 + 删除无效配置
- 可审查性：中
- 审查规则建议：多条tiling路径时应对每条路径的边界值（如numPoints=1）有UT覆盖；多tensor同名维度应强制一致性校验

### 9a9bb866fb55f2d6f6d9223b74b5d0371f5b8bb9 修复instanceNormV3算子example不支持arch2002上调用的问题
- 根因类别：平台适配缺陷
- 涉及文件：build.sh, norm/instance_norm_v3/examples/arch20/test_aclnn_instance_norm.cpp(新增), examples/test_aclnn_instance_norm.cpp(删除)
- 缺陷描述：instanceNormV3是纯arch2002算子但example放在通用目录，build.sh的build_example只搜索通用和arch35目录缺少arch20。旧example内容全被注释掉临时规避CI。
- 修复模式：迁移example到arch20子目录 + 构建脚本增加ascend310p平台路径匹配
- 可审查性：高
- 审查规则建议：新增算子example时应确认支持设备列表与example存放路径一致；build.sh的build_example平台分支应覆盖所有已支持平台

### 75b0eb1e89f7e4ed69d95f745f34bca28cd0beef fix aclnnSigmoidBackward.md
- 根因类别：文档错误
- 涉及文件：activation/sigmoid_grad/docs/aclnnSigmoidBackward.md
- 缺陷描述：markdown格式问题（不必要的转义）；错误码场景描述笼统写"shape不一致"，修复后拆分为三个具体场景。
- 修复模式：文档内容修正和细化
- 可审查性：高
- 审查规则建议：API文档错误码描述应与代码中实际校验逻辑一一对应

### 37ae7c27cac15614eb123be0f2fa64ebdbc73fbd fix 负载均衡性能劣化
- 根因类别：Tiling策略条件判断不当（性能缺陷）
- 涉及文件：adaptive_sliding_window_tiling.cpp, test_quant_batch_matmul_v3.csv
- 缺陷描述：balanceAfterFixp条件原只在kSize<1024时为true，但kSize==1024且nCore>=8时也应启用边界优化。缺少此条件导致性能劣化。
- 修复模式：扩展条件判断增加kSize==1024&&nCore>=8分支
- 可审查性：低
- 审查规则建议：tiling策略条件阈值变更需配合性能基线测试；多维度交叉条件的tiling逻辑应在注释中明确每个阈值的物理含义

### c467ceed2c123df910bb7ccab1f167172a7be244 bmm ut补充修复性能问题
- 根因类别：Tiling策略优先级配置缺陷
- 涉及文件：batch_matmul_v3_tiling_strategy.h, 多个UT文件
- 缺陷描述：ASCEND950平台的BatchMatMulV3策略优先级列表缺少ITER_BATCH_BASICAPI策略，导致某些场景未匹配最优策略路径。
- 修复模式：调整策略优先级顺序 + 补充UT
- 可审查性：低
- 审查规则建议：策略优先级列表变更需附性能对比数据；新平台支持时应检查策略列表是否完整覆盖所有已实现策略

### e63cd956285308d58a1a141cb70948af7830b0cb fix scatterElementsV2 bare instructions
- 根因类别：代码规范（裸指令）
- 涉及文件：scatter_elements_v2_cache_scatter_add.h
- 缺陷描述：12处使用C风格裸指令pipe_barrier(PIPE_V)未使用AscendC封装模板API。
- 修复模式：将pipe_barrier(PIPE_V)替换为PipeBarrier<PIPE_V>()，机械式重构
- 可审查性：高
- 审查规则建议：可通过静态扫描禁止pipe_barrier等裸指令调用，强制使用PipeBarrier<>()模板API

### 34ad168fe776a1a89b643fc9ed0e9f0b44a17bf3 修正rms_norm_quant_v2的编译warning
- 根因类别：编译警告（参数遮蔽+有符号/无符号比较）
- 涉及文件：rms_norm_quant_v2_tiling_regbase_base.cpp, rms_norm_quant_v2_tiling_regbase_recompute.cpp
- 缺陷描述：(1) 函数参数名与类成员同名产生遮蔽warning，修复为加in前缀；(2) int64_t和uint64_t混合比较，修复为显式static_cast<uint64_t>。
- 修复模式：重命名参数消除遮蔽 + 显式类型转换
- 可审查性：高
- 审查规则建议：CI编译应开启-Wshadow和-Wsign-compare并设为error；函数参数命名应避免与类成员同名

### 4d1939f54fbdccb133dadf84b124993cc7e47a62 卷积反向dx算子正向流程补齐以及codecheck修复
- 根因类别：功能缺失 + 代码规范问题
- 涉及文件：conv/convolution_forward/op_host/op_api/convolution.cpp
- 缺陷描述：(1) 正向卷积(非转置)的N2H路径未补齐，只有转置卷积路径；(2) 使用magic number(63/1500/4096等)；(3) 条件不满足时用OP_LOGE应改为OP_LOGD；(4) 循环变量int64_t与无符号Size()比较；(5) output shape构造方式优化。
- 修复模式：功能补齐 + 代码重构（提取常量、拆分函数、修正日志级别、修复类型不匹配）
- 可审查性：中
- 审查规则建议：算子优化路径适用性检查应用调试级别日志而非错误级别；magic number应定义为命名常量

### a9951ec742db23c76ba1ceaa281e562d454ca6f3 Atlas推理 rmsnorm性能提升、layernormv4错误描述优化、addrmsnormquantv2校验修复
- 根因类别：校验返回值丢弃（代码逻辑缺陷）+ 性能优化 + 错误信息优化
- 涉及文件：aclnn_add_rms_norm_quant_v2.cpp, aclnn_fast_layer_norm.cpp, rms_norm_tiling.cpp, rms_norm_merge_n.h
- 缺陷描述：(1) addrmsnormquantv2中CheckParamsV2和SelfPreDealData返回值被丢弃，校验失败仍继续执行；(2) layernormv4错误信息不够清晰；(3) rmsnorm在310P平台norm轴=128时未走MODE_MERGE_N模板性能差10倍，SetBlockDim调用位置在模板选择前但310P的MODE_MERGE_N会修改useCoreNum。
- 修复模式：(1) 补充返回值检查 (2) 改进错误消息 (3) 扩展tiling条件+调整SetBlockDim位置
- 可审查性：中
- 审查规则建议：调用返回aclnnStatus的函数必须检查返回值，可用[[nodiscard]]或静态分析强制；SetBlockDim调用应在所有可能修改useCoreNum的逻辑之后

## 第8轮 (commit 141-160)

### 3e1ff64 convolution_backward 算子注释文档修正
纯文档修改（代码注释中的路径修正），跳过。

### a291109 softmaxGrad文档修正
纯文档修改（README/接口文档修正），跳过。

### 8af6f91 sparsesoftmaxcrossentropywithlogits AICERROR问题修复
- 根因类别：API配置遗漏（SIMT/SIMD混合编程场景未设置UB大小）
- 涉及文件：loss/sparse_softmax_cross_entropy_with_logits/op_host/sparse_softmax_cross_entropy_with_logits_tiling_arch35.cpp
- 缺陷描述：tiling的DoTiling()中设置了BlockDim和ScheduleMode，但漏掉SetLocalMemorySize(ubSize)调用。SIMT/SIMD混合编程模式要求显式设置UB内存大小，缺失导致AIC ERROR。
- 修复模式：单行补漏，在SetScheduleMode后添加SetLocalMemorySize调用
- 可审查性：高
- 审查规则建议：使用SIMT/SIMD混合编程的tiling文件，SetBlockDim/SetScheduleMode/SetLocalMemorySize三者必须同时出现

### d9f57c1 修正aclnnFusedMatmul资料
纯文档修改，跳过。

### 146f4f8 修复TopKTopPSampleV2 minP采样精度问题
- 根因类别：变量语义复用（同一变量在不同阶段承载不同含义导致错误引用）
- 涉及文件：index/top_k_top_p_sample_v2/op_kernel/top_k_top_p_sample_v2.h
- 缺陷描述：maxValue在softmax计算过程中从原始logit值变为概率值，但minP>=1.0的退化采样路径需要原始最大值来回填localValueRs。未保存原始值导致精度错误。
- 修复模式：引入oriMaxValue镜像变量保存原始状态，在6个分支点捕获原始最大值
- 可审查性：低
- 审查规则建议：当同一变量在赋值后被进一步变换再使用时，检查所有使用点是否期望变换前还是变换后的值；多分支状态设置应具有对称性

### 88f4ebf fix aclnnFastGeluBackward check
- 根因类别：参数校验配对顺序错误
- 涉及文件：activation/fast_gelu_grad/op_api/aclnn_fast_gelu_backward.cpp
- 缺陷描述：dtype和shape校验时，原代码将(gradOutput,gradInput)和(self,gradInput)配对校验，正确应为(self,gradOutput)和(gradOutput,gradInput)，原写法在传递性校验上存在逻辑漏洞
- 修复模式：调整校验配对顺序
- 可审查性：高
- 审查规则建议：3个以上tensor两两校验一致性时，检查配对是否覆盖正确的传递链(A==B, B==C)

### 20a9c63 embadding bag算子infer shape动态场景bug修改
- 根因类别：动态shape场景未处理（边界条件遗漏）
- 涉及文件：index/embedding_bag/op_host/embedding_bag_infershape.cpp, 对应UT
- 缺陷描述：infer shape只处理了unknown rank，未处理unknown shape（dim为-1/负值），导致负值直接参与计算产生错误shape，后续core dump
- 修复模式：在每个子shape推导函数中新增is_unknown_shape判断分支，调用SetUnknownShape
- 可审查性：中
- 审查规则建议：infer shape必须同时处理三种场景：静态shape、unknown rank、unknown shape(dim为负值)。检查了IsUnknownRank的代码必须同时检查IsUnknownShape

### 32e3a9f DLQBMM A4W4二级量化 修复尾块偏移错误导致的精度错误
- 根因类别：多核尾块偏移计算错误
- 涉及文件：matmul/dual_level_quant_batch_matmul/op_kernel/arch35/dual_level_quant_batch_matmul_vec_compute.h
- 缺陷描述：CopyUbToGm中realML0Size<128且l0Params.mL1Size>128的尾块场景，VEC0/VEC1核的数据偏移和大小计算错误（统一按CeilDiv(realML0Size,2)分配但尾块切分大小不同）
- 修复模式：新增尾块特殊处理逻辑，VEC0核最多64行，VEC1处理剩余，增加<=0提前返回保护
- 可审查性：低
- 审查规则建议：多核数据切分逻辑存在尾块场景时，验证尾块大小与标准块不同时的偏移计算正确性

### ca1626a 修正aclnnInplaceScatterValue相关描述
纯文档修改，跳过。

### 7c23775 inplaceindexadd deterministic tiling fix
- 根因类别：off-by-one边界错误（< 应为 <=）
- 涉及文件：index/inplace_index_add/op_host/arch35/inplace_index_add_determinstic_tiling.cpp
- 缺陷描述：条件indicesAxis_ < splitCoreNumThresh应为<=，当indicesAxis_恰好等于阈值时未进入有效tiling分支
- 修复模式：一字符修复 < -> <=
- 可审查性：高
- 审查规则建议：阈值比较的分支条件(<vs<=, >vs>=)需确认边界值归属哪个分支

### c32f0f8 修复CTCLossV2 Input ShapeSize大于INT32_MAX AIC_ERROR问题
- 根因类别：整数溢出（int32_t截断）
- 涉及文件：loss/ctc_loss_v2/op_kernel/arch35/ctc_loss_v2_fp16.h, ctc_loss_v2_fp32.h
- 缺陷描述：lpInputStride在函数签名和局部变量中硬编码int32_t，但lpBatchStride等已用模板参数ThreadType。大shape下lpInputStride溢出导致AIC ERROR
- 修复模式：类型提升，int32_t -> ThreadType，fp16/fp32两文件对称修改
- 可审查性：高
- 审查规则建议：同一计算上下文中同类语义变量（stride/offset）应使用一致的数据宽度，部分已升宽时检查剩余字段

### 1c68777 fix EmbeddingDenseGradV2 kernel apt
- 根因类别：kernel apt配置遗漏
- 涉及文件：index/embedding_dense_grad_v2/op_kernel/embedding_dense_grad_v2_apt.cpp
- 缺陷描述：缺少KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0)声明，涉及核间同步的算子必须设置MIX_AIV类型
- 修复模式：单行添加宏声明
- 可审查性：高
- 审查规则建议：使用核间同步(SyncAll等)的算子kernel，apt入口必须包含KERNEL_TASK_TYPE_DEFAULT声明

### aebe3eb Fixed the bug that padIdx is used for calculation but none value is set in Interface
- 根因类别：接口语义不一致（复合型bug，横跨host/kernel/api三层）
- 涉及文件：index/embedding_bag/下12个文件
- 缺陷描述：(1) paddingIdx传入none但kernel仍用于计算 (2) GetAttrPointer缺空指针检查 (3) validIndicesFactorNumber统计范围错误（跨越整个bag而非当前批次） (4) outQueue从TQue改TBuf修复资源管理 (5) 空bag场景处理不当 (6) 平台判断从SocVersion改NpuArch
- 修复模式：接口-实现全链路对齐重构
- 可审查性：低
- 审查规则建议：单commit修改>5文件且跨host/kernel/api层应拆分；GetAttrPointer调用后必须空指针检查；循环内计数器作用域应与循环层级匹配

### 98b45b6 fix_pull_request
纯文档修改，跳过。

### a92b304 修复算子编译过程中的错误打印日志
- 根因类别：日志信息复制粘贴错误
- 涉及文件：scripts/kernel/binary_script/build_binary_opc.sh
- 缺陷描述：实际调用build_binary_opc_gen_task.sh但失败日志打印build_binary_single_op_gen_task.sh，误导排查
- 修复模式：修正日志字符串中的脚本名
- 可审查性：高
- 审查规则建议：错误日志中引用的命令名应与实际执行的命令匹配

### 2e273d9 [dx] [B702] fix MTE out of range
- 根因类别：条件分支过宽导致MTE越界
- 涉及文件：conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_inner_product_tiling.cpp
- 缺陷描述：TILING_HK且dedx_w==1的场景也使用了固定baseM=256计算像素数量，但该场景不应走此路径，导致a1PixelNum偏小、L1空间校验通过但实际搬运越界
- 修复模式：收窄条件分支，删除多余OR条件
- 可审查性：高
- 审查规则建议：if条件含多个OR分支且涉及硬件资源边界计算时，审查每个分支是否满足后续计算的前提假设

### 89f41a2 修复opapi的打桩代码
- 根因类别：stub文件重复定义（与源码enum冲突）
- 涉及文件：tests/ut/op_api/stub/opdev/platform.h
- 缺陷描述：stub头文件手动定义NpuArch枚举，与上游platform/soc_spec.h定义冲突
- 修复模式：删除手写enum，改为#include引入官方定义
- 可审查性：高
- 审查规则建议：stub/mock文件不应手动复制源码enum/struct定义，应通过include引入

### aa742cf 同步脚本,修复出包不能合并构建产物的问题
- 根因类别：构建工具链缺失
- 涉及文件：scripts/package/common/py/merge_binary_info_config.py（新增）
- 缺陷描述：出包流程缺少binary_info_config.json配置合并脚本，导致构建产物无法合并
- 修复模式：新增Python合并脚本
- 可审查性：中
- 审查规则建议：dict.update()是浅合并，嵌套结构可能丢失base子字段；新增构建脚本应配套UT

### 1f9064e slice mm bugfix
- 根因类别：StreamK阈值计算错误 + NonContiguous tensor场景守护缺失
- 涉及文件：matmul/common/op_host/op_api/matmul_util.cpp, matmul/mat_mul_v3/下4个文件
- 缺陷描述：(1) SK模式k轴阈值公式过大导致错误路径选择 (2) DPSK模式同样阈值过大 (3) createView产生的1D storageShape的3D tensor错误进入StreamK路径
- 修复模式：阈值修正 + 新增IsSelfNonContiguous守护函数
- 可审查性：低
- 审查规则建议：matmul tiling阈值常量变更需附带性能测试数据；commit混合阈值调整和场景守护应拆分

### 5e9bde5 fix mmv2nz to mmv3 nznznd
- 根因类别：算子路由缺陷（特定平台+format组合未切换正确实现）
- 涉及文件：matmul/common/和matmul/mat_mul_v3/下6个文件
- 缺陷描述：A2平台N轴对齐场景下MatMul V2 NZNZNZ路径需切换到V3 NZNZND；同时NZ格式输入从storageShape取k/m维度不正确，应从originShape取
- 修复模式：算子路由扩展 + tiling维度计算修正
- 可审查性：低
- 审查规则建议：功能变更和bug修复混合的大commit应拆分；两平台binary.json完全相同存在copy-paste风险

## 第9轮分析 (commit #161-#180)

### 7a1f3b33 修复QuantBatchMatmulV3纯CUBE模板低阶API的若干问题
- 根因类别：常量重复定义/命名不一致 + 条件分支遗漏枚举值
- 涉及文件：matmul/common/cmct/block/block_mmad_a8w8_fixpipe_quant.h, block_mmad_mx.h, kernel_qbmm_cube.h, quant_batch_matmul_constant.h, adaptive_sliding_window_basic_api_tiling.cpp, quant_batch_matmul_v3_apt.cpp
- 缺陷描述：三类子缺陷：(1) IDX_M_TILE_IDX等常量在三个文件中各自独立定义，与quant_batch_matmul_constant.h中统一常量命名不一致，修复收拢到统一头文件。(2) isCubePerChannel条件只检查DT_UINT64遗漏DT_INT64，INT64 scale类型的PerChannel场景无法进入CUBE模板路径。(3) kernel入口宏用排除式条件(!= DT_FLOAT8_E8M0)不精确，改为显式列举DT_UINT64/DT_INT64/DT_FLOAT/DT_BF16，同时#endif #if改为#else修复互斥分支逻辑。
- 修复模式：常量统一收拢 + 条件分支补全遗漏枚举值 + 排除式改为显式列举
- 可审查性：中
- 审查规则建议：枚举/类型新增成员时全局搜索所有条件判断分支确认覆盖；避免排除式条件选择模板路径

### c834b681 修复qbmmv3精度失败问题
- 根因类别：constexpr if条件逻辑错误 + 命名空间限定缺失
- 涉及文件：matmul/common/cmct/kernel/kernel_qbmm_pertile.h, matmul/common/cmct/utils/coord_utils.h
- 缺陷描述：(1) GetQuantOffset中load balance偏移修正用!(isTransA && !isTransB)将M和N修正耦合，当isTransA=false,isTransB=false时M修正应生效但N不应生效，原代码错误地同时执行两者。修复拆分为独立的if constexpr (!isTransA)和if constexpr (isTransB)。(2) kernel_qbmm_pertile.h中Get<IDX_A_OFFSET>缺少QuantBatchMatmul::命名空间限定，存在同名常量时解析到错误索引导致精度问题。
- 修复模式：拆分复合布尔条件为独立分支 + 显式命名空间限定
- 可审查性：低
- 审查规则建议：涉及transA/transB组合场景应对4种(TT/TN/NT/NN)逐一验证；多namespace中存在同名常量时必须显式限定

### 37855341 修复纯CUBE模板遗漏之处
- 根因类别：off-by-one边界条件
- 涉及文件：matmul/common/cmct/block/block_scheduler_qbmm.h
- 缺陷描述：判断是否需要tail tile split的条件为roundIdx_ < round_ - 1，最后一轮(roundIdx_==round_-1)错误进入tail split路径。修复为roundIdx_ < round_，使正常轮次(含最后一轮)都不做tail split。
- 修复模式：边界条件修正 < round_ - 1 改为 < round_
- 可审查性：高
- 审查规则建议：涉及roundIdx < round - 1与roundIdx < round的边界条件必须验证最后一轮行为

### 330382df 【bugfix】msdag编译告警修复
- 根因类别：未使用参数
- 涉及文件：vfusion/multi_scale_deformable_attention_grad/op_host/op_api/aclnn_multi_scale_deformable_attention_grad.cpp
- 缺陷描述：CheckFormat和CheckShape函数签名包含gradValue/gradLocation/gradAttnWeight三个未使用的输出梯度参数，产生unused parameter告警。修复移除这些参数。
- 修复模式：移除未使用的函数参数
- 可审查性：高
- 审查规则建议：开启-Werror=unused-parameter

### 7a69e382 fix dav_3510 x transpose problem
- 根因类别：平台特定约束缺失
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_host/op_api/aclnn_weight_quant_batch_matmul_v2.cpp
- 缺陷描述：ContiguousCheck对x tensor允许转置，但DAV_3510平台不支持x转置。修复增加平台判断：DAV_3510时要求!transposeX && IsContiguous(x)。
- 修复模式：增加平台特定参数约束校验分支
- 可审查性：低
- 审查规则建议：多硬件平台时应有平台能力矩阵，新增平台对照矩阵逐项校验

### 76626cce swish相关代码整改&quant类问题修复
- 根因类别：混合提交(空指针链式调用 + API迁移 + 新特性)
- 涉及文件：activation/silu_grad/op_host/arch35/silu_grad_tiling.cpp, activation/swish/ 多个文件, quant/ascend_quant/ 多个文件
- 缺陷描述：(1) silu_grad tiling中context_->GetOutputShape(0)->GetStorageShape()链式调用未检查GetOutputShape(0)是否为null。(2) swish算子将旧错误处理宏统一替换为新标准宏。(3) 新增dynamic_mx_quant相关算子。
- 修复模式：防御式编程(空指针检查) + API迁移
- 可审查性：低
- 审查规则建议：链式调用返回指针的方法时必须每一步检查null

### 1719d1ff fix foreach_mul_scalar_v2 aclnn storage shape bug
- 根因类别：storage shape与view shape不一致
- 涉及文件：foreach/foreach_mul_scalar/op_host/op_api/aclnn_foreach_mul_scalar_v2.cpp
- 缺陷描述：输出tensor的storageShape和viewShape不一致时，后续格式检查和计算基于错误的storageShape。修复在CheckShape后增加SetOutStorageShape步骤，将每个输出tensor的storageShape重设为viewShape。
- 修复模式：在校验流程中增加shape归一化步骤
- 可审查性：高
- 审查规则建议：输出tensor由调用方传入时，使用前必须确保storageShape与viewShape一致性

### 60a0233d slice matmul bugfix
- 根因类别：回退路径不完整(slice非连续处理)
- 涉及文件：matmul/common/op_host/op_api/matmul_util.cpp, matmul_util.h
- 缺陷描述：(1) IsSliceNonContiguous未检查NZ格式和左右矩阵dtype是否相同，这些检查延后到调用处。(2) 当slice路径不支持需回退到连续路径时，缺少batch维度fold操作，3D tensor的batch维没被正确合并到M维导致shape不匹配。修复将约束检查前置+回退路径补全维度折叠。
- 修复模式：前置约束检查 + 补全回退路径维度折叠
- 可审查性：中
- 审查规则建议："乐观尝试+回退"模式中，回退路径必须完整恢复到连续路径等价语义

### cf9ea8c9 修复dynamic_quant精度问题
- 根因类别：硬件指令参数溢出(DMA参数uint16溢出)
- 涉及文件：quant/dynamic_quant/op_kernel/dynamic_quant_unalign_large_310p.h, dynamic_quant.cpp, test_dynamic_quant.cpp
- 缺陷描述：DMA搬运指令参数使用uint16_t类型，当stride值超过UINT16_MAX时截断溢出，导致搬运地址错误引发精度问题。修复为溢出时将safeCalRow降为1逐行处理。
- 修复模式：运行时溢出检测 + 降级为逐行处理
- 可审查性：高
- 审查规则建议：DMA/搬运指令参数转uint16_t前必须检查溢出；硬件指令参数有位宽限制时必须做溢出保护

### 7dbffe7c fix ascend config json
- 根因类别：编译配置缺失 + infershape提前返回缺失
- 涉及文件：scripts/kernel/binary_config/ascendc_config.json, index/repeat_interleave/op_host/repeat_interleave_infershape.cpp
- 缺陷描述：(1) 约30个算子在ascend910_95上缺少-mllvm -cce-aicore-dcci-before-kernel-end=false编译选项。(2) repeat_interleave infershape中设置UnknownRank后没有return，继续执行后续shape推导导致非法维度访问。
- 修复模式：配置补全 + early return
- 可审查性：中
- 审查规则建议：infershape中设置UnknownRank/UnknownShape后必须立即return

### 7631a0fb bugfix (splitK bias添加时机)
- 根因类别：逻辑错误(条件判断值错误)
- 涉及文件：matmul/mat_mul_v3/op_kernel/arch35/mat_mul_asw_kernel.h
- 缺陷描述：splitK场景bias添加条件为kIndex == splitKRound-1(最后一轮)，但bias应在第一轮(kIndex==0)加入，否则与中间累加逻辑冲突导致精度偏差。
- 修复模式：条件值修正 splitKRound-1 -> 0
- 可审查性：中
- 审查规则建议：splitK/分块累加场景中，一次性addend(bias/残差)应在首轮注入而非末轮

### b7b3a900 fix kernel ut bug when depends multi ops
- 根因类别：构建系统逻辑缺陷(CMake参数解析 + Shell健壮性)
- 涉及文件：build.sh, cmake/ut.cmake
- 缺陷描述：(1) ut.cmake中AddOpTestCase用硬编码位置解析可变参数，UT依赖多个算子时丢弃多余依赖。改为foreach循环遍历。(2) build.sh无条件执行pre_op_kernel_ut target，不存在时报错。改为先检查target是否存在。
- 修复模式：参数解析从固定位置改为循环遍历；构建命令加target存在性检查
- 可审查性：高
- 审查规则建议：CMake处理可变参数应用foreach或cmake_parse_arguments，禁止硬编码index

### ad433211 修复qbmm编译问题
- 根因类别：标识符大小写拼写错误
- 涉及文件：matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_apt.cpp
- 缺陷描述：宏内调用QbmmCmctPerTileKernel(T大写)但实际类名为QbmmCmctPertileKernel(t小写)，导致编译失败。
- 修复模式：单字符大小写修正
- 可审查性：高
- 审查规则建议：宏展开中引用的类名应与声明完全一致；该路径之前未被编译覆盖说明测试覆盖不足

### 7b9edfcb fix marndq binary, remove opapi
- 根因类别：构建配置错误(CMake参数 + ini section名)
- 涉及文件：norm/multi_add_rms_norm_dynamic_quant/op_host/CMakeLists.txt, simplified_key.ini
- 缺陷描述：(1) ACLNNTYPE为aclnn导致生成多余op_api代码与手写实现冲突，改为aclnn_exclude。(2) ini section名MULTI_ADD_RMS_NORM_DYNAMIC_QUANT(全大写)与框架期望的CamelCase名MultiAddRmsNormDynamicQuant不匹配。
- 修复模式：CMake参数值修正 + ini section名大小写规范修正
- 可审查性：高
- 审查规则建议：simplified_key.ini的section名必须与算子CamelCase注册名一致；CI加校验脚本

### dea19dca elu、gelu、sigmoid等问题修复
- 根因类别：混合修复(平台判断硬编码 + 结构体声明顺序 + 变量名遮蔽)
- 涉及文件：activation/elu/fast_gelu/sigmoid_grad/silu_grad等约50+文件
- 缺陷描述：(1) BF16支持判断用GetSocVersion()逐芯片枚举，改为GetCurNpuArch()==DAV_2201||IsRegbase()按架构族判断。(2) CompileInfo结构体声明顺序调整。(3) 函数参数名与外层作用域同名标识冲突。
- 修复模式：平台判断从枚举SoC改为架构族判断；结构体声明顺序调整；变量重命名消除遮蔽
- 可审查性：中
- 审查规则建议：禁止op_api中用GetSocVersion()逐个枚举判断特性支持，应使用NpuArch架构族判断

### b95a1b31 修复sc告警信息
- 根因类别：隐式布尔转换告警
- 涉及文件：matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/quant_batch_matmul_v3_iterbatch_tiling.cpp
- 缺陷描述：!aicoreParams_.aicNum对无符号整型做隐式布尔转换判零，改为aicoreParams_.aicNum == 0。
- 修复模式：隐式布尔转换改为显式比较
- 可审查性：高
- 审查规则建议：无符号整型判零应使用== 0而非!运算符

### ec1602f4 fix assert aicpu ut
- 根因类别：UT构建未统一接入框架
- 涉及文件：cmake/ut.cmake, control/assert/CMakeLists.txt, control/assert/tests/ut/op_kernel_aicpu/
- 缺陷描述：assert算子的aicpu UT用独立CMakeLists而非仓库统一AddAicpuOpTestCase宏，多UT并行构建时冲突。修复删除独立CMake，改用统一宏注册。
- 修复模式：构建配置统一到框架标准流程
- 可审查性：低
- 审查规则建议：新增算子UT必须使用仓库统一的AddAicpuOpTestCase宏注册

### b0803c3a index/scatter_add/README.md中失效链接修正
- 根因类别：非代码缺陷(文档链接失效)
- 涉及文件：index/scatter_add/README.md
- 缺陷描述：README中引用文档的相对路径从common/改为../../docs/zh/context/。
- 修复模式：文档路径修正
- 可审查性：高
- 审查规则建议：CI加入文档链接有效性检查

### ce4d538f fix mse_loss tilingdata bug
- 根因类别：host/kernel tiling数据结构不一致
- 涉及文件：loss/mse_loss/op_host/arch35/mse_loss_tiling_arch35.cpp, mse_loss_tiling_arch35.h, mse_loss_tiling_def.h
- 缺陷描述：kernel侧自定义ReduceOpTilingDataV2结构体与框架标准ReduceOpTilingData字段布局不同，host侧手动逐字段拷贝转换容易遗漏导致tiling数据错乱。修复删除自定义结构体，统一使用框架标准类型。
- 修复模式：消除冗余自定义数据结构，统一使用框架标准类型
- 可审查性：中
- 审查规则建议：禁止算子侧重新定义框架已提供的标准数据结构；host和kernel的tiling结构体必须来自同一头文件

### 6a789b7c fix dx nullptr
- 根因类别：空指针解引用
- 涉及文件：conv/convolution_backward/op_api/aclnn_convolution_backward.cpp
- 缺陷描述：计算dw时gradInput可能为nullptr(不需要dx)，但用gradInput->GetViewShape()计算mmDwOutShape2d导致core dump。修复改用gradWeight->GetViewShape()，语义上也更正确。
- 修复模式：将null对象引用替换为正确的非null对象
- 可审查性：高
- 审查规则建议：可选输出tensor使用前必须做空指针检查；计算某输出的shape应引用语义正确的tensor

### 52a4c5f2 fix dynamic_quant_update_scatter_v2 aclnn & readme
- 根因类别：接口/配置冗余
- 涉及文件：README.md及aclnn相关文件（纯文档/配置删除，无代码文件变更）
- 缺陷描述：算子无aclnn调用模式，但自动生成了aclnn接口文件，导致接口冗余和用户困惑
- 修复模式：删除自动生成的无效aclnn接口
- 可审查性：低
- 审查规则建议：新算子上线前确认是否需要aclnn接口，避免自动生成无效接口

### 4063f8f9 修复aclnnTransposeBatchMatMul.md中的样例精度问题
- 根因类别：文档demo代码错误
- 涉及文件：matmul/transpose_batch_mat_mul/examples/arch35/test_aclnn_transpose_batch_mat_mul.cpp
- 缺陷描述：示例代码输出全0，原因是FP16数据处理和初始化有误。修复完全重写了示例，添加了正确的FP16转换逻辑(Fp16ToFloat)
- 修复模式：重写示例代码，使用正确的数据类型处理
- 可审查性：中
- 审查规则建议：算子示例代码必须经过实际运行验证输出正确性

### b39a4d45 fix package size
- 根因类别：binary配置缺失
- 涉及文件：scripts/kernel/binary_config/ascendc_config.json
- 缺陷描述：HardSwishGradV2算子未在ascendc_config.json中注册，导致包大小异常
- 修复模式：添加算子配置项(含compute_units和auto_sync设置)
- 可审查性：中
- 审查规则建议：新增算子必须同步更新binary配置文件

### 38bb9260 修复TransQuantParamsV2二进制增幅问题
- 根因类别：binary配置平台覆盖不全
- 涉及文件：scripts/kernel/binary_config/ascendc_config.json
- 缺陷描述：TransQuantParamV2的compute_units缺少ascend310p，该平台编译时使用了默认(auto_sync=true)配置，导致二进制文件异常增大
- 修复模式：在compute_units列表中添加ascend310p
- 可审查性：高
- 审查规则建议：算子新增平台支持时必须同步更新binary配置中的compute_units列表；auto_sync设置对二进制大小影响显著

### ab465d7d 增加SetScheduleMode多核同步，修复logit算子NaN bug，增加SmoothL1LossGradV2 inferdtype
- 根因类别：(1)多核同步缺失 (2)特殊值(NaN)处理缺失 (3)inferdtype缺失
- 涉及文件：loss/logit/op_kernel/logit.h, loss/chamfer_distance_grad/op_host/chamfer_distance_grad_tiling.cpp, loss/ctc_loss_v2/op_host/arch35/ctc_loss_v2_tiling_arch35.cpp, loss/mse_loss_v2/op_host/mse_loss_v2_tiling.cpp, loss/smooth_l1_loss_grad_v2/op_graph/smooth_l1_loss_grad_v2_proto.h 等
- 缺陷描述：
  (1) chamfer_distance_grad/ctc_loss_v2/mse_loss_v2等算子使用多核但缺少SetScheduleMode(BATCH_MODE)设置，导致核间同步问题
  (2) logit算子在输入包含NaN时未保留NaN语义，输出结果错误。修复添加Compare(EQ, self, self)检测NaN(NaN!=NaN)，再用Select保留NaN值
  (3) SmoothL1LossGradV2缺少proto定义导致inferdtype功能缺失
- 修复模式：(1)添加SetScheduleMode调用 (2)添加NaN检测mask和Select操作 (3)新增proto头文件
- 可审查性：高
- 审查规则建议：多核算子必须设置SetScheduleMode；处理浮点数据的算子必须考虑NaN/Inf输入场景；NaN检测模式 = Compare(x, x, EQ)

### 3d35698f ascend_quant、ascend_quant_v2、dynamic_quant 问题修复
- 根因类别：平台判断方式不当（SocVersion枚举 -> NpuArch架构族）
- 涉及文件：quant/ascend_quant/op_api/aclnn_ascend_quant.cpp, quant/ascend_quant_v2/op_host/ascend_quant_v2_tiling.cpp, quant/ascend_quant_v2/op_host/op_api/aclnn_ascend_quant_v3.cpp
- 缺陷描述：使用SocVersion::ASCEND910B等具体芯片型号判断平台能力，新芯片加入时需逐一添加case。迁移到NpuArch::DAV_2201等架构族判断后，同架构的新芯片自动支持
- 修复模式：将SocVersion switch替换为NpuArch switch；常量名从平台名(ASCEND910B)改为能力名(WITH_BF16)；使用IsRegbaseSocVersion()工具函数
- 可审查性：高
- 审查规则建议：禁止使用SocVersion枚举判断平台能力，改用NpuArch架构族；变量名应反映能力而非平台名

### dd15791e 修复experimental脚本问题
- 根因类别：CI脚本工作目录错误
- 涉及文件：scripts/ci/check_experimental_example.sh, scripts/ci/check_experimental_pkg.sh
- 缺陷描述：脚本先cd到BUILD_PATH执行cmake，之后调用${BASE_PATH}/scripts下的python脚本时仍在BUILD_PATH下，导致相对路径解析错误
- 修复模式：在cmake步骤后添加cd "${BASE_PATH}"恢复工作目录
- 可审查性：高
- 审查规则建议：shell脚本中cd后的后续命令需要检查工作目录假设是否成立；优先使用pushd/popd或绝对路径

### 647010ea 增加fixpipe优化
- 根因类别：性能优化/API迁移（非缺陷修复）
- 涉及文件：matmul/common/cmct/block/block_mmad_builder.h, block_mmad_iterbatch.h, kernel_matmul_without_que.h
- 缺陷描述：为MatmulIterBatch添加ND_FIXPIPE_1_2 dispatch策略支持，新增L0C到UB的Fixpipe CopyOut路径。同时在kernel执行前后添加SetMMLayoutTransform(true/false)调用
- 修复模式：扩展模板特化，合并重复代码，新增fixpipe输出路径
- 可审查性：低
- 审查规则建议：新增dispatch策略时检查所有相关Builder类是否已适配

### 6e5da556 修复 wqbmmv2 和 qbmmv4 在子包编译失败的问题
- 根因类别：binary配置属性缺失
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_host/config/ascend910_95/weight_quant_batch_matmul_v2_binary.json
- 缺陷描述：binary.json中所有配置段落均缺少inner_precise属性，导致opp_kernel编译失败。需要在每个属性列表末尾添加inner_precise字段
- 修复模式：在每个kernel配置段添加 {"name": "inner_precise", "dtype": "int", "value": 0}
- 可审查性：高
- 审查规则建议：binary.json配置模板应包含所有必需属性的检查清单；新增平台配置时必须包含inner_precise等必需属性

### 1641e3af fix offset range
- 根因类别：整数溢出（uint32->uint64）
- 涉及文件：norm/add_rms_norm_cast/op_kernel/add_rms_norm_cast.h, norm/add_rms_norm_cast/op_kernel/add_rms_norm_cast_multi_n.h, norm/add_rms_norm_dynamic_quant/op_host/op_api/aclnn_add_rms_norm_dynamic_quant_v2.cpp
- 缺陷描述：gm_bias使用uint32_t类型，当i_o * rowFactor * numCol乘积超过2^32时溢出，导致GM地址计算错误。修复将gm_bias改为uint64_t，所有乘法操作数都static_cast<uint64_t>
- 修复模式：类型提升uint32_t -> uint64_t，贯穿整个调用链(CopyIn/Compute/CopyOut参数)
- 可审查性：高
- 审查规则建议：GM offset/地址计算必须使用uint64_t；检查所有uint32_t offset与tensor维度的乘积是否可能溢出

### 864d2df1 fix group_norm_silu
- 根因类别：文档/示例代码错误
- 涉及文件：norm/group_norm_silu/examples/test_aclnn_group_norm_silu_v2.cpp
- 缺陷描述：GroupNormSiluV2示例代码使用了旧版API(aclnn_group_norm_silu而非V2接口)，且校验信息不完整
- 修复模式：重写示例代码使用正确的V2 API
- 可审查性：中
- 审查规则建议：算子版本升级后必须同步更新示例代码中的API引用

### f90f38c0 fix gng copy problem for new interface
- 根因类别：DMA搬运接口使用不当导致OOM
- 涉及文件：norm/group_norm_grad/op_kernel/arch35/group_norm_grad_recompute.h
- 缺陷描述：DataCopy要求数据大小block对齐，当mainReduceNum/mode2UbTailNum_不满足对齐要求时，会读取超出buffer范围的内存导致OOM。修复改用DataCopyPad接口，显式设置blockLen=实际字节数，自动处理非对齐情况
- 修复模式：DataCopy替换为DataCopyPad(配合DataCopyExtParams和DataCopyPadExtParams)
- 可审查性：高
- 审查规则建议：非block对齐的数据搬运必须使用DataCopyPad而非DataCopy；审查DataCopy调用时检查数据量是否保证block对齐

### ab4ce144 aclnn_quant_matmul_v5 infer groupsize bugfix
- 根因类别：维度索引混淆（转置场景）
- 涉及文件：matmul/quant_batch_matmul_v4/op_host/op_api/aclnn_quant_matmul_v5.cpp
- 缺陷描述：InferGroupSize中非MicroScaling场景，transX1时应取penultimate dim但取了last dim，非transX1时反之。MicroScaling场景还漏乘了2(e8m0格式scale shape为[m,k/2,2])
- 修复模式：交换transX1/非transX1的维度索引取值；MicroScaling场景scaleSizeK乘2
- 可审查性：高
- 审查规则建议：转置场景的维度索引取值必须逐case验证；scale shape推导需考虑数据格式编码(e8m0等)

### 8277682 修复Matmul切换低阶API导致的性能劣化
- 根因类别：API迁移遗漏（SetMMLayoutTransform调用缺失）
- 涉及文件：matmul/common/cmct/kernel/kernel_matmul_without_que.h
- 缺陷描述：从高阶Matmul API切换到低阶API实现后，未调用SetMMLayoutTransform(true)设置Fixpipe UnitFlag搬运方向，导致搬运效率降低、并行度变差、性能下降
- 修复模式：在Init前添加SetMMLayoutTransform(true)，在计算完成后添加SetMMLayoutTransform(false)
- 可审查性：中
- 审查规则建议：API迁移时必须对比原API内部行为，确认所有隐式设置已在新代码中显式调用

### ab60dc45 generate basic block table in compile time
- 根因类别：代码质量改进（硬编码消除，非缺陷修复）
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_basic_block_table.h
- 缺陷描述：~1000行硬编码的BasicBlock表替换为constexpr编译期生成，消除了维护困难和不一致风险
- 修复模式：使用constexpr模板函数在编译期计算block表
- 可审查性：低
- 审查规则建议：可通过算法生成的常量表应使用constexpr或代码生成，避免手工维护

### e62e7b4d 修复fusedmatmul带bias场景的性能问题
- 根因类别：tiling计算未考虑bias占用L1内存
- 涉及文件：matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_asw_tiling.cpp, matmul_v3_basic_aswt_tiling.cpp
- 缺陷描述：stepK计算使用完整L1 size，但bias场景下L1中已有bias table占用空间(BIAS_TABLE_NUM * DATA_SIZE_FP32)。stepK过大导致实际搬运超出可用L1空间，性能劣化
- 修复模式：计算remainSizeForAL1BL1时扣除bias table占用，用此值重新计算stepKa/stepKb
- 可审查性：高
- 审查规则建议：tiling参数计算必须考虑所有内存占用(bias/workspace/临时buffer)，不应假设整个buffer都可用

### b2e2cada addbmm&baddbmm空tensor入参错误修复
- 根因类别：复制粘贴错误（参数重复）
- 涉及文件：matmul/batch_mat_mul_v3/op_host/op_api/aclnn_addbmm.cpp, aclnn_baddbmm.cpp
- 缺陷描述：isAddBmmProcessEmptyTensor(batch1, batch1)和isProcessEmptyTensor(batch1, batch1)中第二个参数应为batch2。当batch2有空维度时无法检测到，继续参与计算导致错误
- 修复模式：将第二个batch1改为batch2
- 可审查性：高
- 审查规则建议：函数调用中两个参数相同(f(a,a))时必须确认是否为复制粘贴错误；代码审查时重点检查参数名相似的调用

### 68a1137d fix norm common cmake change and fix addRmsNormCast infershape
- 根因类别：infershape未支持unknown rank
- 涉及文件：norm/add_rms_norm/tests/ut/op_host/test_AddRmsNorm_infershape.cpp, test_add_rms_norm_tiling.cpp, norm/add_rms_norm_cast相关文件
- 缺陷描述：AddRmsNorm infershape不支持unknown rank场景(shape={-2})，新增UT覆盖此场景。同时修复cmake构建配置
- 修复模式：添加unknown rank UT用例，修复infershape逻辑
- 可审查性：中
- 审查规则建议：infershape实现必须覆盖unknown rank(-2)和unknown dim(-1)场景

### e76fa0ee fix matmul bug
- 根因类别：(1)错误日志变量引用错误 (2)硬件同步API不规范 (3)AFullLoad条件过严
- 涉及文件：matmul/fused_quant_mat_mul/op_host/.../fused_quant_matmul_checker.cpp, fused_quant_matmul_swiglu_tiling.cpp, fused_quant_mat_mul_swiglu.h, matmul/quant_batch_matmul_v3/op_host/.../adaptive_sliding_window_tiling.cpp
- 缺陷描述：
  (1) 错误日志中打印scaleShape的维度值时使用了offsetShape变量，输出错误的维度信息
  (2) FusedQuantMatmulSwiglu使用raw指令(set_flag/wait_flag)做FIX_V同步，不符合API规范
  (3) AFullLoad判断强制要求mBlockCnt<=nBlockCnt，在supportMmadS8S4平台上过于严格，限制了优化机会
- 修复模式：(1)修正变量名 (2)替换为AscendC::SetFlag/WaitFlag + FetchEventID (3)s8s4平台跳过blockCnt比较条件
- 可审查性：高
- 审查规则建议：日志和错误信息中的变量引用必须与上下文一致；使用封装API而非raw指令；条件判断需考虑平台差异

### eaebf6db SwishGrad算子精度修复
- 根因类别：(1)未初始化变量/变量引用错误 (2)TilingData类型不匹配
- 涉及文件：activation/swish_grad/op_host/arch35/swish_grad_tiling_arch35.cpp, swish_grad_tiling_arch35.h, activation/swish_grad/op_kernel/arch35/swish_grad_tilingdata.h
- 缺陷描述：
  (1) tilingKey计算使用成员变量schMode，但schMode未从tiling数据中正确赋值。修复改用tiling->baseTiling.scheMode
  (2) TilingData中scale使用int64_t类型，但实际数据是float，类型不匹配导致精度错误
  (3) TilingData中存在无用的int64_t value字段，浪费空间
- 修复模式：(1)直接引用tiling结构体字段而非中间变量 (2)scale类型从int64_t改为float (3)删除无用字段
- 可审查性：高
- 审查规则建议：TilingData结构体字段类型必须与实际数据语义匹配；tilingKey计算的每个输入都必须来自可靠数据源

## 第11轮 (commit 201-220)

### 8e6718595bb841d0239f3a5b5ab877b329d219cd wqbmm kernel config file fix in master
- 根因类别：API迁移不完整 + binary配置缺失
- 涉及文件：matmul/common/op_host/op_api/batch_matmul_util.cpp, matmul/weight_quant_batch_matmul_v2/op_host/config/ascend310p/weight_quant_batch_matmul_v2_binary.json
- 缺陷描述：(1) TransBmm2Mm函数参数enableHf32(bool)需更新为opImplModeEnum(int64_t)以适配MatMulV3Nd新接口，旧代码传入bool导致语义错误；(2) binary配置JSON缺少所有input/output的paramType字段(required/optional)
- 修复模式：参数类型从bool改为int64_t枚举，调用点构造枚举值(enableHf32?0x40:enableForceGrp?0x4:0x1)；JSON补全paramType字段
- 可审查性：中 —— API变更需查阅接口文档，但binary配置缺失paramType可通过模板对比发现
- 审查规则建议：L0 API升级时全量检查所有调用点的参数类型和语义；binary配置JSON必须包含所有paramType字段

### 3b93f190f8461c109edd6af5e7a36904dbef1219 fix: a16f8 n=1 not supported
- 根因类别：边界条件未覆盖
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling.cpp
- 缺陷描述：A16F8模式仅支持per-channel量化，但当antiquant_scale的元素数量为1(即n=1)时，tiling将其判定为per-tensor场景并报不支持。文档要求per-channel的scale shape为(n,)，n=1是合法输入但被错误拒绝
- 修复模式：新增ConfigureReuseScenarios()方法，检测A16F8+per-tensor组合时自动复用per-channel逻辑
- 可审查性：高 —— 边界值n=1是典型的边界测试case
- 审查规则建议：量化模式切换逻辑必须覆盖参数为最小合法值(如n=1)的场景

### e24af7fe239d16c46ece22e2fbdc666bbab956b4 fix warning log + fix SetScheduleMode
- 根因类别：多核调度模式遗漏 + API误用
- 涉及文件：activation/gelu_quant/op_kernel/gelu_quant.cpp, quant/swi_glu_quant/op_host/swi_glu_quant_tiling.cpp
- 缺陷描述：(1) gelu_quant kernel入口多余调用SetSysWorkspace(workspace)，实际只需GetUserWorkspace；(2) swi_glu_quant tiling缺少SetScheduleMode(BATCH_MODE)调用，导致多核调度策略不正确
- 修复模式：删除多余SetSysWorkspace调用；tiling函数开头添加SetScheduleMode(BATCH_MODE)
- 可审查性：高 —— SetScheduleMode是多核算子的必备配置，可通过checklist检查
- 审查规则建议：多核算子tiling必须显式调用SetScheduleMode，kernel入口不应调用SetSysWorkspace(由框架管理)

### 0e10d3883cc0bd5a3ed6cd78af1617c32d5ca86e dynamic_block_quant、trans_quant_param_v2算子问题修复
- 根因类别：功能参数未应用 + 错误码语义错误
- 涉及文件：quant/dynamic_block_quant/op_host/dynamic_block_quant_i8_tiling.cpp, quant/dynamic_block_quant/op_kernel/dynamic_block_quant.h, quant/trans_quant_param_v2/op_host/op_api/aclnn_trans_quant_param_v2.cpp
- 缺陷描述：(1) dynamic_block_quant新增minScale参数但kernel未实际应用，量化时未对scale做min/max clamp导致极小scale产生溢出；(2) trans_quant_param_v2参数校验返回ACLNN_ERR_INNER_NULLPTR(内部错误)而非ACLNN_ERR_PARAM_NULLPTR(参数错误)，错误码语义不匹配
- 修复模式：kernel中添加hasMinScale判断及Min/Max clamp逻辑；修正错误码为ACLNN_ERR_PARAM_NULLPTR
- 可审查性：高 —— 新增参数必须端到端验证(host tiling -> kernel使用)；错误码应与实际错误类型匹配
- 审查规则建议：新增算子参数时，验证host->tiling->kernel全链路均已使用该参数；错误码必须与错误类型(参数/内部/资源)匹配

### 93f6d5b0563624db3ab90cb4501cad38e07fa65e fix layer norm v4 last reduce add mask problem
- 根因类别：尾块mask逻辑错误导致精度问题
- 涉及文件：norm/layer_norm_v4/op_kernel/arch35/layer_norm_v4_two_pass_perf.h
- 缺陷描述：LAST_LOOP_NUMS==2时，第二块x2与x1直接Add后用pregLast做ReduceSum，但pregLast掩码基于lastBinaryAddNum计算，对两块场景掩码范围错误。实际上x2可能包含超出有效范围的脏数据，直接Add会污染结果
- 修复模式：用ShiftLefts将x2的有效部分(lastTailNum = lastBinaryAddNum-VL_B32)移入寄存器并零填充无效位，再与x1做Add，最后用pregFull做ReduceSum
- 可审查性：中 —— 需要理解两轮归约的数据布局和掩码语义
- 审查规则建议：多块归约的尾块处理必须确保无效数据不参与计算，mask范围需与实际有效数据量一致

### 15530399173292672c6356ac5f4d3efef56ae5a4 fix conv 精度问题
- 根因类别：条件分支覆盖不全 + 数据类型提升缺失
- 涉及文件：conv/convolution_backward/op_api/aclnn_convolution_backward.cpp
- 缺陷描述：(1) needSpecialCast条件缺少storageShapeDimSize > 1检查，当维度为1时访问dim(storageShapeDimSize-1)虽合法但逻辑不正确；(2) Conv1D->2D转换时dilation=45需特殊处理exchangeDim；(3) 特定SoC+case组合需要cast到float提升精度但未做
- 修复模式：添加storageShapeDimSize>1条件；新增DILATION_45常量和isConv2dToCastFloat白名单检查；exchangeDim根据groups和dilation条件选择
- 可审查性：中 —— 需要了解卷积反向的精度特性和平台差异
- 审查规则建议：卷积反向的数据类型提升逻辑需覆盖所有SoC平台；Conv1D->2D转换时groups和dilation的组合场景需穷举

### 9031da227ce5a92427ddf6fcda302f4ffcace286 fix allocated oversize of FusedLinearCrossEntropyLossGrad
- 根因类别：workspace大小计算错误(过大)
- 涉及文件：matmul/fused_linear_cross_entropy_loss_grad/op_host/fused_linear_cross_entropy_loss_grad_tiling.cpp
- 缺陷描述：workspace计算中BT*H和V*H两项各乘了BUFFER_NUM(=2)，但这两个buffer不需要双缓冲，导致workspace申请量是实际需求的2倍，浪费内存
- 修复模式：去掉两项的*BUFFER_NUM，添加日志打印workspace大小
- 可审查性：高 —— BUFFER_NUM的使用应与实际缓冲策略(单缓冲/双缓冲)一致，可通过审查每个buffer的用途判断
- 审查规则建议：workspace各项的BUFFER_NUM倍数必须与实际缓冲策略匹配，过度申请虽不影响正确性但浪费资源

### ecd6c75824b458d0e2ea624a3eeb73be1a801815 8.5 bug fix 合入ops-nn
- 根因类别：tiling workspace维度限制未细化 + 内存限制缺失
- 涉及文件：index/index_put_v2/op_api/aclnn_index_put_impl.cpp, index/index_put_with_sort/op_host/op_api/index_put_with_sort.cpp, index/scatter_elements_v2/op_host/op_api/aclnn_scatter.cpp, tests
- 缺陷描述：(1) index_put_v2对高维(5-8维)大索引数量，原60M上限过大导致tiling内部失败，需按维度细化限制(54M/48M/44M/40M)；(2) index_put_with_sort对fp16数据升精度后总内存可能超250MB但未检查；(3) scatter的reduce参数非法值无告警；(4) UT类名重复导致编译冲突
- 修复模式：按维度分级设置indices数量上限；添加内存上限检查(250MB)；添加reduce参数校验告警；重命名UT类
- 可审查性：高 —— tiling限制应与实际workspace容量匹配，内存限制应有上界检查
- 审查规则建议：高维场景的tiling限制必须按维度递减；数据类型提升(fp16->fp32)需检查内存膨胀是否超限

### 82c38b544653771df3ec12edcafa47ef5563896f 修复ConvTbcBackward算子文档错误
- 根因类别：文档错误
- 涉及文件：纯文档修改
- 缺陷描述：文档内容修正
- 可审查性：低
- 审查规则建议：N/A

### c51de510ac71135691fbdbdfa638af2e97a9e0a7 修复matmul目录下example内存泄漏问题
- 根因类别：Example代码资源泄漏
- 涉及文件：matmul/batch_mat_mul_v3/examples/test_aclnn_addbmm.cpp 等多个example文件
- 缺陷描述：example代码中aclTensor、device memory、workspace等资源在错误路径上未释放，存在内存泄漏。用户参考example编写应用时会继承同样的问题
- 修复模式：使用unique_ptr+自定义deleter(aclDestroyTensor, aclrtFree)实现RAII
- 可审查性：高 —— 裸指针分配后无RAII包装是明显的泄漏模式
- 审查规则建议：example代码中所有acl资源分配必须使用RAII或配对释放

### 6623da3fdfea838416be8bd6c0bed20384fd3807 修复experimental存在同名算子导致编译失败的问题
- 根因类别：构建系统搜索路径错误
- 涉及文件：cmake/gen_ops_info.cmake
- 缺陷描述：CMake中find命令搜索CMAKE_CURRENT_SOURCE_DIR(整个源码目录)而非OP_DIR(当前算子目录)，当experimental目录存在同名算子时找到错误的def.cpp，导致编译逻辑判断失败
- 修复模式：搜索路径从CMAKE_CURRENT_SOURCE_DIR改为OP_DIR
- 可审查性：高 —— CMake搜索路径应限定在目标目录，全局搜索存在同名碰撞风险
- 审查规则建议：CMake find/搜索命令的路径必须限定为目标算子目录，禁止全局搜索

### 42e15c0bb4be44af8e6020b3d0aaca513f891b0e 问题修复，leaky_relu输入为float16报错
- 根因类别：tiling key未设置
- 涉及文件：activation/leaky_relu/op_host/arch35/leaky_relu_tiling_arch35.cpp
- 缺陷描述：leaky_relu算子当输出dtype为FLOAT16时，进入fp16->fp32 cast处理分支，但dType变量未被设置为TPL_FP16，导致tiling key错误，运行时找不到匹配的kernel
- 修复模式：在fp16分支开头添加dType = static_cast<uint64_t>(TPL_FP16)
- 可审查性：高 —— 每个dtype分支必须设置对应的tiling key，遗漏可通过分支检查发现
- 审查规则建议：tiling中每个dtype分支必须设置对应的tilingKey/dType变量

### 179ab8a17bfa59e5076d99f58952ea183e04ebae 修复卷积反向bf16精度问题
- 根因类别：BF16数据类型路径处理不当
- 涉及文件：conv/convolution_backward/op_api/aclnn_convolution_backward.cpp
- 缺陷描述：卷积反向的needSpecialCast路径原始逻辑统一cast到fp16再transdata再cast回目标dtype，但BF16输出时中间经fp16会丢失精度。需区分float和bf16输出：float走fp16中转再cast回；bf16直接cast到目标dtype再transdata
- 修复模式：在needSpecialCast分支内按outputDtype是float还是其他(bf16)分两条路径
- 可审查性：高 —— 数据类型中转路径必须考虑所有支持的输出dtype
- 审查规则建议：Cast中转路径需确保中间dtype的精度不低于目标dtype；BF16与FP16是不同精度范围，不可互相中转

### f06cb5dbc7ba729cba65bfe1f35847880625dd99 修复kernel编译报错问题
- 根因类别：构建系统搜索路径错误 + 存在性检查缺失
- 涉及文件：cmake/gen_ops_info.cmake
- 缺陷描述：(1) 与#211相同的搜索路径问题(CMAKE_CURRENT_SOURCE_DIR -> OP_DIR)；(2) 遍历算子目录时未检查op_host/{name}_def.cpp和op_kernel目录是否同时存在，缺一时执行错误的编译逻辑
- 修复模式：修正搜索路径；添加def.cpp和op_kernel目录的存在性检查，缺少任一则跳过
- 可审查性：高 —— 构建脚本应对文件存在性做防御性检查
- 审查规则建议：构建系统遍历算子时必须验证def+kernel成对存在

### 14b9cdb8b509095c1c43692f8a39029f8a420789 quant_matmul hard sync bugfix
- 根因类别：多核sub-block守护缺失 + 事件同步时序错误
- 涉及文件：matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_bf16.h, quant_batch_matmul_v3_bf16_basic.h, quant_batch_matmul_v3_pertoken.h
- 缺陷描述：(1) SplitK场景下GetSubBlockIdx()==1的sub-block应跳过执行，但Init和Process函数只检查blockIdx_>=usedCoreNum_未检查sub-block；(2) AIC事件等待条件loop_>2/loop_>3过于宽松，在loop_<=2时跳过了必要的WaitEvent，导致C2V同步丢失
- 修复模式：添加GetSubBlockIdx()==1守护到Init和Process；将WaitEvent条件改为loop_>0和loop_>1
- 可审查性：高 —— SplitK场景必须检查sub-block index；事件同步的Wait/Set必须配对
- 审查规则建议：SplitK/多sub-block场景中，非主sub-block必须在Init和Process入口处跳过；AIC/AIV事件同步的Wait条件必须确保不跳过任何Set

### 98254481e5d40449491bdee6da23217b09104652 修正MaxPoolV3精度问题和opapi告警
- 根因类别：大kernel精度劣化 + 编译告警
- 涉及文件：pooling/max_pool_v3/op_api/aclnn_max_pool.cpp
- 缺陷描述：MaxPoolV3在kernel size > 1000时存在精度问题(数值范围过大导致比较/归约精度下降)。另有函数返回值const修饰多余的告警
- 修复模式：ksize>1000时路由到MaxPool3dWithArgmaxV2(精度更好)；移除函数返回值多余的const
- 可审查性：中 —— 需要了解不同pooling实现的精度特性
- 审查规则建议：大参数范围(kernel size/stride极大)需验证精度是否满足要求

### 2e389cbb0e6d1eaf31abc8aba165cc25d4b0d9aa aclnnSwiglu.md文档修复
- 根因类别：文档错误
- 涉及文件：纯文档修改
- 缺陷描述：文档内容修正
- 可审查性：低
- 审查规则建议：N/A

### 83c7e7c55d8994cbacc405e2378489b3644f5609 修复fake_quant_affine_cachemask算子精度问题
- 根因类别：特殊值(NaN)处理逻辑错误
- 涉及文件：quant/fake_quant_affine_cachemask/op_kernel/fake_quant_affine_cachemask_fp16.h, fake_quant_affine_cachemask_fp32.h
- 缺陷描述：原代码对NaN输入的处理有误：(1) Compare(x,x,EQ)利用NaN!=NaN特性检测NaN，但后续Select将NaN位置替换为quantMin值，语义错误(应为0)；(2) 额外的Compare(y,infTensor,NE)检测Inf逻辑多余且与quantMin的Select组合效果不正确
- 修复模式：简化为单次Compare(x,x,EQ)+Select(result,0.0f)，NaN位置输出0而非quantMin，删除多余的Inf检测
- 可审查性：高 —— NaN/Inf处理逻辑应有明确的数学语义说明
- 审查规则建议：特殊值处理(NaN/Inf/零)的Select替换值必须与数学定义一致；每个Compare+Select组合需注明其语义

### aa4a6df285df70bfbfde226fdbe7247d87274c8e Delete invalid versions
- 根因类别：无效平台配置残留
- 涉及文件：matmul/batch_mat_mul_v3/op_host/batch_mat_mul_v3_def.cpp, matmul/fused_quant_mat_mul/op_graph/fused_quant_mat_mul_proto.h
- 缺陷描述：batch_mat_mul_v3的def.cpp中残留无效的mc62cm12a SoC配置(与ascend910_95完全重复)；fused_quant_mat_mul的proto.h文件已废弃未删除
- 修复模式：删除无效SoC配置块和废弃文件
- 可审查性：高 —— 重复的SoC配置块和废弃文件可通过代码审查发现
- 审查规则建议：新增SoC配置前检查是否存在同内容的重复配置；废弃的proto/header文件应及时清理

### f63ed7eb39b2c1bbe79f22c7f35ea9b11f5c514e 修复lng的tiling告警
- 根因类别：编译告警(未使用变量 + 有符号/无符号比较)
- 涉及文件：norm/layer_norm_grad_v3/op_host/layer_norm_grad_v3_grouped_reduce_big_m_tiling.cpp, layer_norm_grad_v3_recompute_tiling.cpp
- 缺陷描述：maxBlocks和mCacheBufferCountStg2变量定义后未使用；uint64_t与int64_t直接比较产生有符号/无符号比较告警
- 修复模式：删除未使用变量；添加static_cast<uint64_t>转换
- 可审查性：高 —— 编译器告警应零容忍
- 审查规则建议：所有变量定义后必须使用；有符号与无符号比较需显式转换

## 第12轮 (commit 221-240)

### eb531f0bce57 fix index_put_v2 opapi UT
- 根因类别：构建配置/依赖缺失
- 涉及文件：index/index_put_v2/CMakeLists.txt
- 缺陷描述：index_put_v2的CMakeLists中DEPENDENCIES只声明了index，缺少index_put_with_sort和linear_index_v2，导致opapi UT编译链接失败
- 修复模式：在add_modules_sources的DEPENDENCIES中补充完整依赖项
- 可审查性：中（需要了解算子间依赖关系）
- 审查规则建议：新增算子CMakeLists时验证DEPENDENCIES是否包含所有被引用的算子模块

### 5ca5cc755f08 修复最新包编译DeformableConv2d编译报错的问题
- 根因类别：头文件依赖缺失
- 涉及文件：conv/deformable_conv2d/op_kernel/deformable_conv2d_base.h
- 缺陷描述：deformable_conv2d_base.h只include了lib/matmul_intf.h，缺少kernel_operator.h。上游包更新后该隐式依赖断裂，导致编译报错
- 修复模式：显式添加#include "kernel_operator.h"
- 可审查性：高（检查每个头文件是否自包含所有直接依赖）
- 审查规则建议：头文件必须显式include所有直接使用的依赖，不依赖间接传递

### 50df91e72ce9 修复foreach算子使用GlobalTensor offset数据类型不匹配
- 根因类别：整数溢出/类型不一致(int32->uint64)
- 涉及文件：foreach/foreach_copy/op_kernel/foreach_copy.h, foreach/foreach_lerp_list/op_kernel/foreach_lerp_list.h, foreach/foreach_lerp_scalar/op_kernel/foreach_lerp_scalar.h, foreach/foreach_non_finite_check_and_unscale/op_kernel/*.h, foreach/foreach_norm/op_kernel/*.h, foreach/foreach_pow_list/op_kernel/foreach_pow_list.h, foreach/foreach_pow_scalar_and_tensor/op_kernel/foreach_pow_scalar_and_tensor.h
- 缺陷描述：GlobalTensor的下标表达式`index * maxDataCount`中index为uint16_t/uint32_t、maxDataCount为int64_t，但乘法在较小类型域执行可能溢出。大tensor偏移超过32位范围时产生越界访问。涉及7个foreach算子的约20处相同模式
- 修复模式：在乘法前插入`1ULL *`强制提升为uint64_t运算：`inTensorsGM[1ULL * index * maxDataCount]`
- 可审查性：高（机械检查GM[]下标表达式是否包含64位提升）
- 审查规则建议：GM内存偏移计算必须使用int64/uint64类型，乘法表达式中至少一个操作数为64位

### 1a87986e53f9 [MMV3] 回退对齐场景下workspace申请逻辑，解决对齐场景内存膨胀
- 根因类别：workspace内存计算公式错误
- 涉及文件：matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp, matmul_v3_base_tiling.h
- 缺陷描述：deterministic splitk场景的workspace计算统一使用alignedM*alignedN作为singleSize，但对齐场景(TilingEnableFixOpti::BASE)下应使用singleCoreN*singleCoreM（未对齐值），导致workspace过度膨胀
- 修复模式：提取GetDeterministicSplitKWorkspaceSize函数，区分BASE场景使用singleCore值、其他场景使用aligned值
- 可审查性：中（需理解对齐策略差异）
- 审查规则建议：workspace大小计算需区分不同tiling策略的对齐需求，避免过度分配

### d6c44e809a4a fix muls cast
- 根因类别：数据类型转换路径错误
- 涉及文件：matmul/batch_mat_mul_v3/op_host/op_api/aclnn_batch_matmul.cpp
- 缺陷描述：baddbmm算子在混精度路径(fp16/bf16 input + CubeMathType==KEEP_DTYPE + 910B/910C)下，transdata输出类型已是FP32，但Cast仍使用out->GetDataType()目标类型(可能为fp16/bf16)，导致精度丢失
- 修复模式：增加enableFp16Bf16InFp32Out条件判断，混精度路径下Cast目标改为DT_FLOAT
- 可审查性：高（混精度路径中检查Cast目标类型是否与实际中间结果类型一致）
- 审查规则建议：混精度/高精度计算路径中的Cast操作，目标类型必须与上游实际输出类型匹配

### 2098f3961e0c fix run_example
- 根因类别：构建脚本路径过宽
- 涉及文件：build.sh
- 缺陷描述：build_example函数中find命令搜索example cpp文件时，会误匹配opgen/template目录下的模板文件，导致模板example也被编译执行
- 修复模式：find命令添加`-not -path "*/opgen/template/*"`排除模板目录
- 可审查性：高（审查find/glob命令的搜索范围是否精确）
- 审查规则建议：构建脚本中的文件搜索命令需显式排除模板/生成代码目录

### 7a453b33047d 修复图模式example cmakelist并且对齐文本描述
- 备注：纯文档修改(docs/zh/invocation/op_invocation.md)，无代码缺陷，跳过

### 78fe4d709a7d fix:auto default true
- 根因类别：shell脚本默认值/条件逻辑错误
- 涉及文件：scripts/kernel/binary_script/build_binary_single_op_gen_task.sh
- 缺陷描述：auto_sync变量默认值设为"false"，但系统约定auto_sync默认应为true(不传即为true)。同时条件判断`!= "true"`过于宽泛（空值也会触发），应改为`== "false"`精确匹配
- 修复模式：默认值从"false"改为""(空)，条件判断从`!= "true"`改为`== "false"`
- 可审查性：高（检查默认值语义是否与系统约定一致）
- 审查规则建议：shell脚本的布尔变量默认值必须与系统约定一致；字符串比较优先使用精确匹配(== "value")而非否定匹配(!= "other_value")

### 27a2a8923a34 修复 convert_weight_to_int4_pack 编译 opapi 时的告警问题
- 根因类别：有符号/无符号比较 + 未使用参数告警
- 涉及文件：matmul/convert_weight_to_int4_pack/op_host/op_api/aclnn_convert_weight_to_int4_pack.cpp
- 缺陷描述：循环中int64_t变量g、k*n与size_t类型循环变量比较产生有符号/无符号告警；aclnnConvertWeightToINT4Pack函数参数未使用产生告警
- 修复模式：添加static_cast<size_t>()转换；未使用参数添加[[maybe_unused]]
- 可审查性：高（编译器告警即可发现）
- 审查规则建议：编译-Wsign-compare -Wunused-parameter告警视同错误处理

### 955f7dd5111a fix check ini
- 根因类别：CMake构建依赖缺失
- 涉及文件：cmake/gen_ops_info.cmake
- 缺陷描述：add_ops_info_target_v1和merge_ini_files函数中，自定义命令缺少DEPENDS opbuild_custom_gen_aclnn_all声明，导致ini文件可能在aclnn生成完成前就被处理，产生时序依赖问题
- 修复模式：在add_custom_command中添加DEPENDS opbuild_custom_gen_aclnn_all
- 可审查性：高（检查CMake自定义命令是否声明了所有上游依赖）
- 审查规则建议：CMake add_custom_command必须在DEPENDS中声明所有前序生成步骤

### 02c0c42c21a6 修复命令bash build.sh -u --ophost报错问题
- 根因类别：目录结构重构后路径不一致
- 涉及文件：activation/swish/op_host/op_tiling/arch35/ -> activation/swish/op_host/arch35/ (重命名), cmake/func.cmake, 多个UT include路径
- 缺陷描述：tiling文件从op_host/op_tiling/arch35迁移到op_host/arch35后，UT中的相对include路径未同步更新。同时func.cmake中OP_HOST_UT条件下的GLOB_RECURSE扫描路径与新目录结构不兼容
- 修复模式：更新UT include路径(../../../ -> ../../../../)，删除cmake中不再适用的GLOB_RECURSE逻辑
- 可审查性：高（文件移动后检查所有引用路径）
- 审查规则建议：目录结构重构后必须全局搜索更新所有相对路径引用

### 611e857246e1 bugfix swi_glu_quant && inplace_add_rms_norm
- 根因类别：预编译宏条件反转
- 涉及文件：norm/inplace_add_rms_norm/op_kernel/inplace_add_rms_norm.cpp, quant/swi_glu_quant/op_kernel/swi_glu_quant.cpp
- 缺陷描述：inplace_add_rms_norm中bfloat16分支的条件写成`#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)`，含义是"仅在3003上编译bf16"，但3003(Ascend310P)不支持bf16，应为"除3003外都编译bf16"即`#if !(...)`. swi_glu_quant中#if/#endif嵌套层级错误导致条件分支覆盖不全
- 修复模式：添加`!`取反条件；修正swi_glu_quant的#if条件合并为单一条件表达式
- 可审查性：高（平台条件宏审查：确认该平台是否实际支持该数据类型）
- 审查规则建议：平台条件编译宏中"排除某平台"应使用`#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == XXX)`模式，审查时验证条件语义是否与平台能力匹配

### d4a43b6f0cd5 fix warning
- 根因类别：未使用变量告警
- 涉及文件：matmul/quant_batch_matmul_v3/op_api/quant_matmul_checker.cpp
- 缺陷描述：x1Shape和x2Shape变量声明后未使用（相关逻辑可能已移至其他位置），产生编译告警
- 修复模式：删除未使用的变量声明
- 可审查性：高（编译器告警）
- 审查规则建议：代码重构后检查被移走逻辑的残留变量声明

### 8d718f35977c 修复了matmulv3在特定shape下的精度问题
- 根因类别：多核循环变量未重新初始化 + 条件分支不完整
- 涉及文件：matmul/mat_mul_v3/op_kernel/mat_mul_deterministic_splitk_kernel.h
- 缺陷描述：ReduceKInUbNzL2cache函数的双层循环中，mCoreUse/nCoreUse只在尾块条件`if (mIndex == mCnt-1 || nIndex == nCnt-1)`内赋值，非尾块迭代时沿用上一次尾块的残留值，导致非尾块的数据搬运范围错误。同时actualN的尾块判断只覆盖了orderNMFlag==true的情况，遗漏了orderNMFlag==false时的n轴尾块
- 修复模式：在循环体顶部无条件重置mCoreUse=tiling.singleCoreM和nCoreUse=tiling.singleCoreN；actualN尾块条件扩展为`(orderNMFlag && outIndex == outCnt-1) || (!orderNMFlag && inIndex == inCnt-1)`
- 可审查性：高（循环内变量是否每次迭代都被正确初始化；条件分支是否覆盖所有flag组合）
- 审查规则建议：循环体内使用的变量必须在每次迭代开始时初始化，不得依赖条件分支内的赋值；多路遍历(orderNMFlag)的尾块判断需覆盖所有排列方式

### eb3755e5b853 SparseTensorDenseMatMul修复Tiling内打印占位符误用
- 根因类别：printf格式占位符与参数类型不匹配
- 涉及文件：matmul/sparse_tensor_dense_mat_mul/op_host/op_tiling/sparse_tensor_dense_mat_mul_tiling_arch35.cpp
- 缺陷描述：OP_LOGE日志中int64_t类型参数使用%lld占位符，但在LP64系统(Linux 64位)上int64_t为long而非long long，应使用%ld。约20处日志语句受影响
- 修复模式：全部%lld改为%ld
- 可审查性：高（静态分析/编译器-Wformat即可发现）
- 审查规则建议：日志格式字符串中int64_t使用PRId64宏或%ld(LP64系统)，避免%lld

### 2052949f785d fix env.sh
- 根因类别：环境变量设置脚本残留代码清理
- 涉及文件：scripts/package/common/sh/script_operator.inc
- 缺陷描述：add_setenv和del_setenv函数中的环境变量设置逻辑已转移到上游包管理，但函数体中仍保留旧实现代码。清空函数体只保留return 0
- 修复模式：删除函数体中的旧逻辑，仅保留return 0
- 可审查性：中（需了解上下游职责变更）
- 审查规则建议：上下游职责迁移后及时清理下游残留实现

### 44886a3fea96 fix mse_loss index unique_consecutive dynamic_mx_quant doc bug
- 备注：纯文档修改(多个README.md)，无代码缺陷，跳过

### cfe65e1f9cef 修复PR 518 多删的tiling策略
- 根因类别：策略优先级列表条目遗漏
- 涉及文件：matmul/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_tiling_strategy.h
- 缺陷描述：先前PR 518在重构BatchMatMulV3PrioritiesMap时误删了ASCEND910_95平台的ITER_BATCH_BASICAPI策略项，导致该平台下特定batch场景无法匹配到正确的tiling策略
- 修复模式：在ASCEND910_95的优先级vector中恢复strategy::ITER_BATCH_BASICAPI条目
- 可审查性：高（PR改动策略列表时diff对比前后条目完整性）
- 审查规则建议：策略/路由表的增删改必须逐条对比变更前后的完整列表，防止误删

### 059ff49b6f3e 【bugfix】去掉chamferdistancegrad的pipeall
- 根因类别：流水线同步粒度过粗(PipeBarrier<PIPE_ALL>)
- 涉及文件：loss/chamfer_distance_grad/op_kernel/chamfer_distance_grad.h
- 缺陷描述：ChamferDistanceGrad kernel中约20处使用PipeBarrier<PIPE_ALL>进行全流水线同步，严重阻塞性能。实际只需特定硬件事件间的同步(如V_MTE3, MTE2_MTE3, MTE3_S等)
- 修复模式：引入PipeSync<HardEvent>模板函数，将所有PIPE_ALL替换为精确的硬件事件同步。部分场景用PIPE_MTE2替代
- 可审查性：高（搜索PipeBarrier<PIPE_ALL>用法，逐处分析实际数据依赖）
- 审查规则建议：禁止在性能敏感kernel中使用PipeBarrier<PIPE_ALL>，必须分析实际数据流确定最小同步粒度

### de3fce33252b TransposeBatchMatmul ut bugfix
- 根因类别：stub类型重复定义冲突
- 涉及文件：matmul/transpose_batch_mat_mul/op_kernel/pp_matmul_ein_sum_kernel.h
- 缺陷描述：UT测试宏__CCE_KT_TEST__下，手写的`using __bf16 = bfloat16_t`与随后include的stub_def.h中的同名定义冲突，导致编译报错
- 修复模式：删除手写的using声明，依赖stub_def.h提供定义
- 可审查性：高（类型别名/using声明不应与框架stub头文件重复）
- 审查规则建议：UT stub环境中的类型定义统一由stub头文件提供，禁止在业务代码中重复声明

## 第13轮 (commit 241-260)

### #241 433615f8 修复aicpu ut编译头文件和链接报错
- 根因类别：构建配置缺陷
- 涉及文件：cmake/ut.cmake, tests/ut/op_kernel_aicpu/CMakeLists.txt
- 缺陷描述：CMake缺少头文件搜索路径`${ASCEND_DIR}/pkg_inc/base`导致编译找不到头文件；链接顺序错误，`-ldl`放在`--whole-archive`区域内，应在`--no-whole-archive`之后
- 修复模式：添加缺失include路径；调整链接库顺序
- 可审查性：中
- 审查规则建议：`--whole-archive`和`--no-whole-archive`之间只应包含静态库(.a)，不应出现`-l`系统库

### #242 bf3a91df 修复dynamic_quant_update_scatter文档
- 纯文档修复，跳过

### #243 6b56f111 修复baddbmm接口bug + mmv3/bmmv3 example整改
- 根因类别：条件分支Guard缺失
- 涉及文件：matmul/batch_mat_mul_v3/op_host/op_api/aclnn_batch_matmul.cpp, common/stub/op_api/level0/batch_matmul_v2tov3.h
- 缺陷描述：三个问题：(1)`enableFp16Bf16InFp32Out`标志在`GetBatchMatmulOpInfo`中未区分isBaddbmm参数，导致普通bmm也错误走fp16转fp32路径；(2)`GetBatchMatmulOp`中fp16/bf16转fp32路径缺少平台检查(需910B/910_93)；(3)K==1场景多余Cast到DT_FLOAT再Mul，不必要的精度损失和性能浪费
- 修复模式：增加`isBaddbmm`参数透传；收紧条件判断；移除K==1多余Cast
- 可审查性：高
- 审查规则建议：当某flag只在特定调用场景下生效时，检查是否有足够Guard条件区分调用上下文

### #244 db6c5243 fix ut case (where算子宏定义)
- 根因类别：宏定义与链式调用模式冲突
- 涉及文件：index/where/tests/ut/op_kernel_aicpu/test_where.cpp
- 缺陷描述：`CREATE_NODEDEF`宏使用`do { ... } while(0)`包裹，但宏体末尾是链式builder调用(.Output())需要后续代码继续追加(.Build())，`do-while(0)`使语句闭合后无法继续链式调用
- 修复模式：移除`do {} while(0)`包裹，改为直接展开语句
- 可审查性：高
- 审查规则建议：宏内有链式builder模式调用时，不应使用`do-while(0)`包裹

### #245 301e3067 fix multi_add_rms_norm_dynamic_quant readme
- 纯文档+license格式修复，跳过

### #246 8a2d45c9 裸指令回退 (GetBlockIdx vs get_subblockid)
- 根因类别：API语义混淆（block级vs sub-block级索引）
- 涉及文件：conv/common/op_kernel/arch35/conv_opt_group_init_impl.h, conv/conv2d_v2/op_kernel/arch35/conv2d_v2_c04_impl.h, conv/conv2d_v2/op_kernel/arch35/conv2d_v2_dma_impl.h, conv/conv2d_v2/op_kernel/arch35/conv2d_v2_weight_ub_trans_impl.h
- 缺陷描述：4个`VecInit`函数中，`self->ctx.vecId`被错误赋值为`GetBlockIdx()`(核级索引)，实际需要`get_subblockid()`(sub-block内ID)。多核/多sub-block场景下vecId错误导致向量计算地址偏移错误
- 修复模式：4处`GetBlockIdx()`改为`get_subblockid()`
- 可审查性：高
- 审查规则建议：arch35 kernel中`vecId`/subblock相关初始化必须使用`get_subblockid()`；`vecId = GetBlockIdx()`为可疑模式

### #247 36b7c87b fix DQUSV2md
- 纯文档修复，跳过

### #248 4cfb4526 fix ut (ApplyAdamWV2 tiling UT)
- 根因类别：UT与实现不同步
- 涉及文件：optim/apply_adam_w_v2/tests/ut/op_host/test_apple_adam_w_v2_arch35_tiling.cpp
- 缺陷描述：tiling逻辑改动（模板数据不参与tilingdata比对）后，UT期望值未同步更新，且有被注释掉的测试用例
- 修复模式：修改`TilingData2Str`只取tiling data尾部关键字段；更新UT期望值；取消注释被禁用的测试用例
- 可审查性：中
- 审查规则建议：修改tiling data结构/序列化逻辑时，必须同步更新UT期望值

### #249 9311a867 修复代码中的warning
- 根因类别：编译告警（类型不匹配+printf格式符+未使用变量）
- 涉及文件：index/目录和vfusion/目录下多个tiling.cpp（10个文件）
- 缺陷描述：uint64_t/int不匹配、printf格式符错误(%d对unsigned等)、未使用变量未删除、未使用参数缺少[[maybe_unused]]
- 修复模式：修改类型声明、printf格式符、删除未使用变量、添加[[maybe_unused]]
- 可审查性：高
- 审查规则建议：开启`-Werror -Wunused-variable -Wformat -Wsign-compare`

### #250 784cb847 修复代码注释中的硬件型号平台表述问题
- 根因类别：注释规范（内部代号泄露）
- 涉及文件：matmul/batch_mat_mul_v3/tests/ut/, matmul/common/op_host/op_api/
- 缺陷描述：代码注释中使用内部芯片代号(1980/1971/1951)而非对外产品型号(910/910B/310P)
- 修复模式：全局搜索替换注释中的代号
- 可审查性：高
- 审查规则建议：预提交钩子禁止代码中出现内部代号

### #251 243a86dc fix examples and geir
- 根因类别：构建配置路径错误
- 涉及文件：build.sh, cmake/func.cmake, scripts/util/dependency_parser.py
- 缺陷描述：build.sh中include/library路径多了一层`/compiler/`子目录；geir编译缺少`-I`和`-lge_compiler`链接选项；cmake未将examples子目录纳入构建；dependency_parser.py中`all_ops`初始列表为空导致example算子未被解析
- 修复模式：路径修正+编译选项补全+构建入口注册
- 可审查性：中
- 审查规则建议：构建脚本路径变量应有存在性检查；CI应包含examples编译验证

### #252 fa496af1 修复conv aclnn op_api ut问题
- 根因类别：CMake依赖声明缺失
- 涉及文件：conv/convolution_backward/op_host/CMakeLists.txt, conv/convolution_forward/op_host/CMakeLists.txt
- 缺陷描述：convolution_backward DEPENDENCIES缺少conv3d_v2和mat_mul_v3；convolution_forward完全没有声明DEPENDENCIES
- 修复模式：补全`DEPENDENCIES conv3d_v2 mat_mul_v3 matmul.common`
- 可审查性：中
- 审查规则建议：新增算子CMakeLists的DEPENDENCIES须完整覆盖实际依赖

### #253 e43ef817 add_rms_norm_cast overflow check
- 根因类别：整数溢出（多维度乘法无溢出检查）
- 涉及文件：norm/add_rms_norm_cast/op_host/add_rms_norm_cast_tiling_arch35.cpp, 对应UT
- 缺陷描述：`GetWorkspaceSize()`中`WORKSPACE_COUNT * usedCoreNum * numN * sizeof(float)`多个uint64_t相乘可能溢出UINT64_MAX，输入shape极大(2^31)时静默溢出算出错误workspaceSize
- 修复模式：乘法前逐步除法计算maxAllowed，超限则报错GRAPH_FAILED；新增overflow UT
- 可审查性：高
- 审查规则建议：多个维度/核数相乘计算size时必须做溢出检查

### #254 f2b9564f fix AdaMaxPool3d pipeAll
- 根因类别：PipeBarrier<PIPE_ALL>粗粒度同步
- 涉及文件：pooling/adaptive_max_pool3d/op_kernel/adaptive_max_pool3d_big_pool.h
- 缺陷描述：bf16类型转换前使用PipeBarrier<PIPE_ALL>()做全流水线同步，过于粗粒度，实际只需S->V同步
- 修复模式：替换为精确事件同步FetchEventID(HardEvent::S_V) + SetFlag + WaitFlag
- 可审查性：高
- 审查规则建议：禁止kernel代码中使用PipeBarrier<PIPE_ALL>()，必须使用精确event-based同步

### #255 4c95f212 修复pr模板自动关联issue
- 纯文档修复（PR模板），跳过

### #256 572a5695 修复linear_index README的错误
- 纯文档修复，跳过

### #257 8f8ded2d swigluQuant fix __restrict__
- 根因类别：编译器优化/别名分析问题
- 涉及文件：quant/swi_glu_quant/op_kernel/swi_glu_quant.cpp
- 缺陷描述：GET_TILING_DATA宏展开后tiling_data是栈上副本，通过&取地址传给op.Init()时，编译器无法做alias分析优化，可能产生错误代码生成
- 修复模式：引入`__restrict__`指针 + 全文`.`改`->`访问
- 可审查性：低
- 审查规则建议：所有GET_TILING_DATA后应统一使用`__restrict__`指针访问模式

### #258 a13fd663 修复foreach问题 (CMake binary_json优先级)
- 根因类别：CMake构建逻辑分支优先级错误
- 涉及文件：cmake/gen_ops_info.cmake
- 缺陷描述：对有自定义binary_json的算子，原逻辑先走name-based查找op_type，若返回空则continue跳过，导致有自定义配置的算子永远无法编译。binary_json存在性检查应优先于name查找
- 修复模式：调整条件分支顺序，binary_json优先级提升到name查找之前
- 可审查性：高
- 审查规则建议：多配置来源的优先级应明确；审查"先检查低优先级来源、失败即退出"导致高优先级来源被跳过的逻辑

### #259 7f3c12e8 fix foreach_mul_scalar opapi ut failure
- 根因类别：UT头文件路径错误 + 不支持平台用例
- 涉及文件：foreach/foreach_mul_scalar/tests/ut/op_host/test_aclnn_foreach_mul_scalar_v2.cpp
- 缺陷描述：include路径多了一层`../`导致编译失败；包含Ascend910(910A)测试用例但算子不支持该平台
- 修复模式：修正include相对路径 + 删除不适用平台的测试用例
- 可审查性：高
- 审查规则建议：UT平台型号应与算子支持矩阵一致；相对路径include应有编译验证

### #260 5e21afdb fix TLS_VERIFY
- 根因类别：TLS证书验证失败（构建环境问题）
- 涉及文件：cmake/third_party/makeself-fetch.cmake
- 缺陷描述：FetchContent_Declare下载makeself时TLS证书校验失败导致构建中断
- 修复模式：添加`TLS_VERIFY OFF`关闭证书校验（临时规避方案）
- 可审查性：中
- 审查规则建议：任何`TLS_VERIFY OFF`必须附带说明和后续修复计划

---
## 第14轮 (commit #261-#280)

跳过纯文档/UT/example修复：#262(example拼写), #265(aclnn资料), #266(文档), #268(UT补齐), #270(foreach example+geir去重), #271(foreach opapi ut路径/用例), #273(groupnorm文档), #274(index算子README), #275(Conv3DV2 README), #276(描述错误), #279(kernel ut), #280(注释错误)

### #261 f43375ca fix GroupNormSilu Tiling
- 根因类别：整数溢出 + 错误传播缺失
- 涉及文件：norm/group_norm_silu/op_host/group_norm_silu_tiling.cpp
- 缺陷描述：(1) `shapeN * tilingData.get_numGroups()`在uint32范围内可能溢出，需提升为uint64; (2) `SetProcessSize`和`SetTilingSD`原本返回void，内部除零检查只打日志不返回错误码，调用方无法感知失败; (3) `remainUbSize == 0`未检查导致后续除法异常
- 修复模式：引入`uint64_t totalGroups`避免中间溢出; 将void函数改为返回`ge::graphStatus`，用`OP_CHECK_IF`宏链式传播错误
- 可审查性：高
- 审查规则建议：(1) 多维度乘法必须检查溢出或使用uint64; (2) tiling内部函数必须返回错误码，禁止void+日志的错误处理模式

### #263 e85c3d01 topKtopPSample selLogitsOut sync fix
- 根因类别：默认值语义错误 + DMA同步缺失
- 涉及文件：index/top_k_top_p_sample/op_kernel/top_k_top_p_sample.h
- 缺陷描述：(1) 可选输出logitsTopKPSelect的默认值为0.0，但文档规定应为-inf，导致未选中的logit值被误认为有效; (2) InitGlobalMemory写入GM后缺少MTE3→MTE2同步(SetWaitFlag)，多核场景下可能脏写; (3) SetGlobalBuffer缺少size参数限制，无边界保护
- 修复模式：定义FP32_NEG_INF_BITS常量用于-inf初始化; 添加SetWaitFlag<MTE3_MTE2>同步; SetGlobalBuffer添加coreEle_参数
- 可审查性：高
- 审查规则建议：(1) InitGlobalMemory默认值必须与算子语义文档一致; (2) GM写入后必须在跨核SyncAll前添加pipe同步

### #264 3a682df5 masked_softmax_with_rel_pos_bias算子修正逻辑错误
- 根因类别：索引常量错误 + 整数溢出 + 零检查溢出
- 涉及文件：norm/masked_softmax_with_rel_pos_bias/op_host/masked_softmax_with_rel_pos_bias_tiling.cpp
- 缺陷描述：(1) Y_OUTPUT_INDEX定义为3(与输入索引混淆)，实际输出只有1个应为0，导致CheckOutShape取错tensor; (2) 零值检查`b_ * w_ * n_ * s1_ * s2_ == 0`本身可能在uint32乘法中溢出，应逐个检查; (3) 多处buffer size计算使用uint32_t(bf16AttenAndBiasCastTempSize, xSize, totalSize等)，大shape时溢出; (4) 中间乘法如`w_ * n_ * s1_ * s2AlignedSize * dtypeSize`缺少uint64 cast
- 修复模式：Y_OUTPUT_INDEX改为0; 乘积零检查改为逐个`==0`; 十余处uint32→uint64升级 + static_cast<uint64_t>; printf格式%u→%lu
- 可审查性：高
- 审查规则建议：(1) 输入/输出索引常量必须与算子proto定义一致; (2) 多维度乘积不应用于零值检查(溢出风险); (3) buffer size变量统一用uint64_t

### #267 607c1717 FixApplyTopkToppNN
- 根因类别：UB size截断 + workspace不足 + TopP路径重构
- 涉及文件：index/apply_top_k_top_p_with_sorted/op_host/apply_top_k_top_p_with_sorted_tiling.cpp, op_kernel/*.h, *.cpp
- 缺陷描述：(1) `platformUbSize`为uint64_t但直接`static_cast<uint32_t>`截断后使用，大UB场景数据丢失; (2) TopP场景workspace只分配了固定syncWorkspaceSize，缺少数据buffer空间; (3) TopP路径实现作为TopK类的方法耦合度高且有逻辑缺陷，需独立类; (4) 多核分配逻辑batchSize_<=coreNum_时走特殊分支，统一为通用逻辑
- 修复模式：platformUbSize_存为uint64; TopP场景workspace增加`batchSize_ * vocabSize_ * 4`字节; TopP路径抽离为ApplyTopPWithSorted类; 添加iterateTimes_=ceil(log2(vocabSize_))
- 可审查性：中
- 审查规则建议：(1) 平台参数(UB/L1 size)禁止截断为uint32; (2) workspace分配必须覆盖所有tilingKey路径的实际需求

### #269 11b557fe Fix Conv3DV2 tiling kAL1 upper bound
- 根因类别：硬件指令参数上限公式错误
- 涉及文件：conv/conv3d_v2/op_host/op_tiling/conv3d_api_tiling_algorithm.cpp
- 缺陷描述：load3d指令postk参数限制为16bit(max 65535)。原公式`(POSTK_LIMIT + k0) / ci0HkWk`计算kAL1上界，但kAL1 * ci0HkWk可能>65535(因为加了k0再除法向上偏移)。正确公式应为`POSTK_LIMIT / ci0HkWk`确保不超限
- 修复模式：移除`+ k0`项，改为直接整除
- 可审查性：高
- 审查规则建议：硬件指令参数上限计算必须用下界公式(floor除法)，不得添加额外余量导致超限

### #272 7a9a7a5a 修复ophost ut以及解决run example链接顺序问题
- 根因类别：链接顺序错误
- 涉及文件：build.sh, tests/ut/op_host/CMakeLists.txt
- 缺陷描述：g++链接时`-lopapi -lcust_opapi`顺序错误，ELF链接器要求提供符号的库在引用符号的库之后。cust_opapi依赖opapi中的符号，所以cust_opapi必须排在opapi之前。同理`-lopapi -lopapi_nn`应为`-lopapi_nn -lopapi`
- 修复模式：调换链接库顺序: `-lcust_opapi -lopapi`, `-lopapi_nn -lopapi`
- 可审查性：高
- 审查规则建议：自定义库(-lcust_*)必须排在基础库(-lopapi)之前；链接顺序变更需验证所有example编译通过

### #277 9d1d765e fix_ut
- 根因类别：类名拼写错误 + UT匹配逻辑不完整 + _exit绕过清理
- 涉及文件：scripts/util/parse_changed_files.py, tests/ut/op_host/test_op_host_main.cpp
- 缺陷描述：(1) `UtMathcer`拼写错误应为`UtMatcher`; (2) OpApiUt只匹配`op_api`路径，遗漏了`op_host`下的`test_aclnn_*`文件(这些文件也是opapi ut); (3) UT main函数用`_exit(RUN_ALL_TESTS())`，_exit不执行atexit回调和全局析构，可能导致资源泄漏和覆盖率数据丢失
- 修复模式：修正拼写; OpApiUt增加`op_host + test_aclnn_`匹配; `_exit`→`return`
- 可审查性：高
- 审查规则建议：(1) CI脚本中的路径匹配逻辑变更需覆盖所有目录结构; (2) 测试入口禁止使用_exit，应用return确保正常清理

### #278 c7819ec4 fix for issue 89 and 86
- 根因类别：TPipe作用域错误 + 编译器alias分析缺失
- 涉及文件：norm/layer_norm_grad_v3/op_kernel/*.h, *.cpp; norm/add_rms_norm/op_kernel/add_rms_norm.h
- 缺陷描述：(1) layer_norm_grad_v3的4个实现类(SingleRead/Workspace/Transpose/Common)各自创建TPipe成员但由宏在函数体内实例化，TPipe管理硬件pipe资源，在类内部创建可能导致生命周期与硬件资源不匹配; (2) add_rms_norm的tiling指针缺少`__restrict__`，编译器无法做alias优化，可能生成低效代码
- 修复模式：在entry函数layer_norm_grad_v3中创建`TPipe pipe`并通过Init参数传递给所有实现类; add_rms_norm tiling指针添加`__restrict__`
- 可审查性：中
- 审查规则建议：(1) TPipe必须在kernel入口函数创建并传递，不得在类内部隐式创建; (2) GET_TILING_DATA指针应标记__restrict__

## 第15轮 (commit 281-300)

### #281 fd3a3d11 fix build warning
- 根因类别：编译告警 - printf格式说明符与参数类型不匹配
- 涉及文件：quant/dynamic_quant_update_scatter_v2/op_host/dynamic_quant_update_scatter_v2_tiling.cpp
- 缺陷描述：OP_LOGE中%llu格式化非unsigned long long类型的ubSize，%u格式化有符号int类型的vectorCoreNum
- 修复模式：修正格式说明符(%llu->%lu, %u->%d)
- 可审查性：高
- 审查规则建议：开启-Wformat编译选项自动检测格式化字符串与参数类型不匹配

### #282 23736aa6 修复kernel文件有后缀的情况下编译无法打包.o和json文件的问题
- 根因类别：构建脚本逻辑缺陷 - 文件名后缀处理位置不当
- 涉及文件：cmake/gen_ops_info.cmake, scripts/kernel/binary_script/build_binary_single_op_gen_task.sh
- 缺陷描述：kernel文件名带后缀(_apt/_910b)时，后缀剥离逻辑分散在子函数内，但main函数中用于拼接.o和json打包路径的变量仍保留后缀，路径与编译产物不一致；cmake中install(FILES)硬编码了${OP_NAME}.json无法匹配带后缀文件
- 修复模式：后缀剥离逻辑上移到main统一处理；cmake改用file(GLOB)模式匹配
- 可审查性：中
- 审查规则建议：构建脚本中文件名转换逻辑应集中在一处；install(FILES)应考虑文件名变体

### #283 21a1f29d fix InplaceScatterAdd readme
- 跳过：纯文档修改

### #284 54768d07 软链接修复
- 根因类别：硬编码平台路径
- 涉及文件：scripts/package/ops_nn/scripts/ops_nn_custom_install.sh, ops_nn_custom_remove_softlink.sh
- 缺陷描述：软链接脚本硬编码ascend910b目录，只为910b平台创建/删除软链接，其他平台(310p等)不被处理
- 修复模式：硬编码平台名替换为ascend*通配符遍历
- 可审查性：高
- 审查规则建议：部署脚本不应硬编码特定硬件平台名称，应使用通配符或配置文件枚举

### #285 1c6efd0d fix codecheck bug
- 根因类别：多类缺陷 - 资源泄漏 + 类型错误 + 魔法数
- 涉及文件：activation/elu/op_host/op_api/aclnn_elu.cpp, activation/erfinv/examples/, activation/fatrelu_mul/examples/, activation/gelu_mul/op_host/, activation/hard_sigmoid/op_host/op_api/, activation/selu_grad/examples/
- 缺陷描述：(1) aclnn_elu.cpp特殊路径分支创建zeroScalar和negAlphaScalar未释放(资源泄漏)；(2) 多个example中分配内存后未释放；(3) hardsigmoid中魔法数0.16666666f改为1.0f/6.0f；(4) selu_grad example梯度数据类型应为float而非int
- 修复模式：添加缺失的资源释放；删除重复检查；命名常量替代魔法数；修正数据类型
- 可审查性：高
- 审查规则建议：每个aclCreate*/aclrtMalloc必须有对应的aclDestroy*/aclrtFree；禁止魔法数

### #286 c489b93d fix rnnv2ut
- 根因类别：构建配置缺陷 - 符号重复定义
- 涉及文件：cmake/ut.cmake, rnn/dynamic_rnnv2/op_host/CMakeLists.txt, rnn/dynamic_rnnv2/tests/ut/op_kernel/CMakeLists.txt
- 缺陷描述：全量编译kernel UT时，dynamic_rnnv2的tiling UT通过特殊分支硬编码引入dynamic_lstm_tiling.cpp，该文件已通过OPHOST_NAME的tiling_obj目标包含，导致符号重复定义链接失败
- 修复模式：删除特殊case硬编码，通过DEPENDENCIES声明统一处理模块间依赖
- 可审查性：中
- 审查规则建议：CMake中不应为特定算子名添加if/else特殊分支，模块间依赖通过构建系统依赖声明机制表达

### #287 a4905db5 [Bugfix] 修复addmm example中的问题，修复matmulv3 infershape向上对齐实现方式不同的问题
- 根因类别：内存泄漏 + 对齐计算公式错误
- 涉及文件：matmul/mat_mul_v3/examples/test_aclnn_addmm_aclnninplace_addmm.cpp, matmul/mat_mul_v3/op_host/mat_mul_v3_infershape.cpp
- 缺陷描述：(1) example中aclnnInplaceAddmm复用了aclnnAddmm的workspaceSize/workspaceAddr变量，第二次aclrtMalloc覆盖第一次地址导致内存泄漏；(2) hidden_size向上对齐公式(*hidden_size + BLOCK_SIZE) / BLOCK_SIZE * BLOCK_SIZE缺少-1，恰好对齐时多对齐一个BLOCK_SIZE
- 修复模式：引入独立变量避免复用覆盖；修正对齐公式为(x + align - 1) / align * align
- 可审查性：高
- 审查规则建议：向上对齐公式必须使用(x + align - 1) / align * align标准模式；不应复用同名变量指向不同的动态分配内存

### #288 0f483ee6 修改quant目录下存在空指针、格式化漏洞等问题
- 根因类别：printf格式说明符错误 + 业务逻辑校验错误
- 涉及文件：quant/ascend_anti_quant_v2/examples/, quant/ascend_quant_v2/op_host/op_api/aclnn_ascend_quant_v3.cpp
- 缺陷描述：(1) example中resultData为int16_t但LOG_PRINT用%f格式化(应为%hd)，导致未定义行为；(2) aclnn_ascend_quant_v3.cpp中对float8类型的roundMode校验错误允许了不支持的"hybrid"模式
- 修复模式：修正格式说明符；收紧输入校验条件
- 可审查性：高
- 审查规则建议：printf格式说明符必须与参数类型严格匹配；参数校验允许值列表变更需与算子规格文档同步

### #289 422098e3 修正masked_softmax_with_rel_pos_bias算子tiling、infershape对应ut用例
- 跳过：纯UT修改

### #290 ed943785 fix ut error
- 根因类别：kernel代码数组越界 + tensor操作偏移冲突
- 涉及文件：loss/ctc_loss_v3/op_kernel/ctc_loss_v3.h
- 缺陷描述：(1) Duplicate初始化logAlpha时使用alphaLengthAlign(完整长度)，应只初始化尾部alpahTailSizeAlign，错误长度导致越界写入；(2) Log的结果写入expLogTensor偏移0处，但被后续Add操作覆盖，GetValue(0)读到的是Add结果而非Log结果。修复后Log写入FLOAT_NUM_PER_BLOCK偏移处
- 修复模式：修正Duplicate长度参数；修正Log输出偏移避免与前序操作重叠
- 可审查性：低
- 审查规则建议：Tensor操作的读写偏移和长度参数需逐行审查确保不重叠；涉及INFINITY特殊值的分支需专门边界测试

### #291 511503b1 修复conv目录下存在逻辑、重复代码问题
- 根因类别：错误处理不当 + 数学函数边界条件 + 整数溢出
- 涉及文件：conv/common/op_host/conv_backprop_infershape.cpp, conv/common/op_host/op_tiling/math_util.cpp, conv/conv3d_backprop_filter_v2/op_host/op_tiling/arch32/conv3d_backprop_filter_v2_base_tiling.cpp
- 缺陷描述：(1) OP_LOGE_IF在nullptr时仅打日志不返回错误，改为OP_CHECK_IF正确返回GRAPH_FAILED；(2) GetGcd函数param2==0时返回0，GCD(n,0)应返回n；(3) CeilDiv结果int64_t直接static_cast<int32_t>缺少溢出检查
- 修复模式：错误处理宏替换(LOGE_IF->CHECK_IF)；GCD边界条件修正；窄化转换前置溢出校验
- 可审查性：高
- 审查规则建议：OP_LOGE_IF不应用于需中断执行的场景(应用OP_CHECK_IF)；数学工具函数必须覆盖零值边界；窄化类型转换前必须有溢出校验

### #292 7bcebf20 quantbatchmatmul 修复编译警告
- 根因类别：编译警告 + C++链式比较逻辑陷阱
- 涉及文件：matmul/quant_batch_matmul_v3/ (3文件), matmul/quant_batch_matmul_v4/ (5文件)
- 缺陷描述：(1) %ld/%zu格式说明符与int64_t/uint64_t不匹配；(2) 函数参数未使用；(3) 产品逻辑bug: a == b == 1在C++中被解析为((a==b)==1)即bool==1，不等价于a==1&&b==1
- 修复模式：格式说明符修正；删除/void标记未使用参数；修复链式比较
- 可审查性：高
- 审查规则建议：禁止使用a==b==c链式比较写法，在C++中几乎永远是bug

### #293 4384b110 【bugfix】kernel ut 整改
- 跳过：纯UT修改

### #294 1c2de786 修复infershape中先使用后校验nullptr的问题，修复n_x2_dim位置
- 根因类别：空指针解引用风险 + 复制粘贴错误
- 涉及文件：matmul/common/op_host/matmul_common_infershape.cpp
- 缺陷描述：(1) InferShapeForBatchMatMul中先解引用shape_x1/shape_x2(调用CheckIsUnknownDimNum)再检查nullptr，时序颠倒导致空指针解引用崩溃；(2) n_x2_dim计算公式与k_x2_dim完全相同(trans_x2 ? dim-1 : dim-2)，复制粘贴错误，n_x2_dim应为trans_x2 ? dim-2 : dim-1
- 修复模式：空指针校验前置(use-before-check -> check-before-use)；修正维度索引互补关系
- 可审查性：高
- 审查规则建议：指针nullptr校验必须在首次解引用之前；矩阵k/n维度索引在转置/非转置情况下必须互补(一个dim-1另一个dim-2)

### #295 2cc78fa3 修复quant类算子kernel ut不执行的问题
- 跳过：纯UT修改

### #296 c391912 修复norm类算子opkernel UT error
- 根因类别：无效校验代码(unsigned类型与0比较)
- 涉及文件：norm/group_norm_grad/op_host/group_norm_grad_tiling.cpp
- 缺陷描述：uint32_t sysWorkspaceSize做<0检查、uint64_t ubSizePlatForm做<=0检查，unsigned类型永远不可能<0，条件恒假为死代码
- 修复模式：删除无符号类型与负数/零的无意义比较
- 可审查性：高
- 审查规则建议：对unsigned类型做<0检查应作为编译错误(开启-Werror=type-limits)

### #297 9eb1b2c7 rms_norm_quant opkernel UT error fix
- 跳过：纯UT修改

### #298 b45ac62b fix compliation warning
- 根因类别：编译警告(未使用的函数参数)
- 涉及文件：foreach/foreach_non_finite_check_and_unscale/op_host/, foreach/foreach_utils/op_host/
- 缺陷描述：TilingPrepare4Foreach*函数的context参数未使用触发-Wunused-parameter
- 修复模式：[[maybe_unused]]属性标记
- 可审查性：高
- 审查规则建议：回调/接口函数中不使用的参数应用[[maybe_unused]]标记或省略参数名

### #299 8595d59b 修复SwiGluQuant大shape精度问题
- 根因类别：整数类型溢出(uint16截断)
- 涉及文件：quant/swi_glu_quant/op_kernel/swi_glu_quant.h, swi_glu_quant_static.h
- 缺陷描述：ProcessCoreMultiUbMultiAlign的offsetRow参数类型为uint16_t(最大65535)，大shape时超出范围截断溢出，数据拷贝地址计算错误导致精度问题
- 修复模式：参数类型从uint16_t扩展为uint32_t
- 可审查性：高
- 审查规则建议：表示shape/偏移量/元素数量的变量不应使用uint16_t等窄类型，至少uint32_t；开启-Wconversion检测窄化转换

### #300 49589d34 ctcloss_v3grad,kernelut_fix
- 跳过：纯UT修改

## 第16轮 (commit 301-320)

### #301 7bf556ae mse_loss_v2,advance_step等kernel_ut_fix
- 跳过：纯UT修复

### #302 4882c709 修复pooling目录下存在资源泄露
- 根因类别：资源泄露/遗漏释放
- 涉及文件：pooling/adaptive_avg_pool3d/examples/test_aclnn_adaptive_avg_pool2d.cpp, test_aclnn_adaptive_avg_pool3d.cpp
- 缺陷描述：示例代码创建的outputSize(aclIntArray)对象未调用aclDestroyIntArray释放，释放阶段只释放了input和out两个tensor，遗漏了outputSize
- 修复模式：补充遗漏的aclDestroyIntArray(outputSize)调用
- 可审查性：高
- 审查规则建议：检查所有aclCreate*/aclDestroy*配对完整性，静态分析扫描同作用域内分配与释放的对应关系

### #303 f8f5e723 fix opapiUT for pooling
- 根因类别：构建配置遗漏
- 涉及文件：cmake/ut.cmake, 多个pooling UT CMakeLists.txt
- 缺陷描述：cmake/ut.cmake中supportedCategory列表缺少"pooling"类别，导致pooling算子UT编译流程无法执行；多个pooling算子UT CMakeLists.txt缺少标准编译配置
- 修复模式：在supportedCategory中补充pooling，补全各UT CMakeLists.txt配置
- 可审查性：高
- 审查规则建议：新增算子类别时应检查cmake构建系统的类别注册列表是否已更新

### #304 cd9c0049 fix group_norm_silu ut
- 跳过：纯UT修复

### #305 0e00f88c fix ops-nn_compile_ophost_warnings
- 根因类别：编译告警含真bug（无限循环、格式符、无效检查）
- 涉及文件：conv/conv3d_backprop_input_v2 tiling, index/apply_top_k_top_p_with_sorted tiling, pooling/adaptive_avg_pool3d_grad tiling, pooling/avg_pool3_d tiling+grad, pooling/max_pool3d_with_argmax_v2 tiling
- 缺陷描述：多处编译告警中隐含实际bug：(1) avg_pool3_d_grad中`for(uint64_t i=singleCoreWo; i>=0UL; i--)`无符号整数>=0恒真导致无限循环；(2) apply_top_k_top_p_with_sorted中printf格式说明符%ld用于size_t应为%zu；(3) adaptive_avg_pool3d_grad中无符号返回值做<0检查永远为false属无效代码；(4) conv3d_backprop_input_v2中有符号/无符号比较
- 修复模式：循环条件改为i>0UL、格式说明符修正、删除无效检查、static_cast类型转换
- 可审查性：高
- 审查规则建议：无符号整数循环中>=0条件是经典无限循环bug，应通过-Wtautological-compare或静态分析检测；启用-Werror将告警升级为错误

### #306 1074ec43 fix-DynamicQuantUpdateScatter-UT
- 根因类别：API接口变更适配
- 涉及文件：tests/ut/op_host/test_op_host_main.cpp, tests/ut/op_kernel/test_op_kernel_main.cpp
- 缺陷描述：OppSoDesc构造函数接口变更，原接受单个AscendString参数，现需要花括号初始化列表{AscendString(...)}。影响所有UT公共入口文件导致全部UT编译失败
- 修复模式：适配新的初始化列表构造函数
- 可审查性：高
- 审查规则建议：上游依赖库API签名变更时应有自动化API兼容性检测

### #307 3bf4267d 修复batch_mat_mul_v3等约束描述
- 跳过：纯文档修改

### #308 925b1ba8 nn-dev仓同步修复issue18
- 根因类别：工具脚本健壮性不足
- 涉及文件：scripts/kernel/binary_script/build_binary_single_op_gen_task.sh, gen_output_json.py
- 缺陷描述：shell脚本无条件调用dos2unix但未检查工具是否安装、文件是否存在；Python脚本kernel编译产物json生成失败时错误信息不够详细
- 修复模式：添加前置条件检查(command -v dos2unix, -f file)和防御性编程
- 可审查性：高
- 审查规则建议：shell脚本调用外部命令前应先检查command -v确认工具存在

### #309 039383d3 修复matmul下gemm_v2等UT用例缺失报错
- 根因类别：构建配置遗漏/迁仓同步不完整
- 涉及文件：matmul/gemm_v2/op_host/CMakeLists.txt(生产代码), 多个matmul UT CMakeLists.txt
- 缺陷描述：(1) gemm_v2的op_host CMakeLists.txt缺少DEPENDENCIES mat_mul_v3声明导致编译失败；(2) quant_batch_matmul_v3 UT使用错误的模块名(OP_API_MODULE_NAME而非OP_TILING_MODULE_NAME)；(3) quant_matmul_reduce_sum UT CMakeLists为空，迁仓时PR未同步
- 修复模式：补充CMake依赖声明和编译配置
- 可审查性：高
- 审查规则建议：迁仓操作后应有自动化构建验证确保所有算子UT能编译运行

### #310 4023e4c2 修改rnn doc修复
- 跳过：纯文档修改

### #311 87a66d70 fix issue 11
- 跳过：纯文档修复（将文档中错误引用的算子名修正）

### #312 12fbd5c8 [swi_glu/swi_glu_grad] kernel ut修复；host ut修复；增加example
- 根因类别：构建配置缺陷 + 无符号整数比较bug
- 涉及文件：activation/swi_glu/op_host/CMakeLists.txt, activation/swi_glu_grad/op_host/CMakeLists.txt, activation/swi_glu/op_host/swi_glu_tiling.cpp(生产代码)
- 缺陷描述：(1) CMakeLists缺少DEPENDENCIES声明，swi_glu与swi_glu_grad互相依赖但未声明；(2) swi_glu_tiling.cpp中totalCore是uint32_t类型，做<0比较永远为false
- 修复模式：补充CMake依赖 + cast为int再比较
- 可审查性：中
- 审查规则建议：-Wtype-limits检测无符号整数与0的<比较

### #313 50854088 ophost warning fix
- 根因类别：编译告警含多类真bug（链式比较、运算符优先级、类型转换）
- 涉及文件：matmul/下17个生产代码文件
- 缺陷描述：(1) `a==b==1`链式比较误用，C++中被解析为(a==b)==1而非a==1&&b==1；(2) `FORMAT_NZ && dim2==1 || dim1==1`运算符优先级错误，&&高于||导致逻辑意图被改变，应加括号；(3) printf格式说明符%zu用于int64_t应为%ld；(4) CeilDiv调用中int64_t强转uint64_t负值变大正数；(5) 多处未使用变量/参数
- 修复模式：链式比较重写、加括号明确优先级、格式说明符修正、类型转换方向修正、删除死代码
- 可审查性：高
- 审查规则建议：-Wlogical-op-parentheses检测&&/||混合；静态分析检测==链式比较；-Wformat -Wsign-compare -Wunused全量启用

### #314 c9b03748 msda ut fix
- 根因类别：构建配置遗漏
- 涉及文件：cmake/ut.cmake
- 缺陷描述：supportedCategory列表缺少"vfusion"类别，导致vfusion目录下算子UT无法被构建系统识别
- 修复模式：补充supportedCategory条目
- 可审查性：高
- 审查规则建议：CI检查扫描顶层目录与supportedCategory列表的差异

### #315 b17c8b62 dequant_swiglu_quant ut error修复
- 跳过：纯UT修复

### #316 c80e6362 fix pooling ophost opkernel UT
- 跳过：纯UT修复

### #317 8ee323a6 fix SMS/SMSG ut
- 跳过：纯UT修复

### #318 ad8129da kernal ut error 修复
- 根因类别：代码风格缺陷（冗余分号）
- 涉及文件：optim/advance_step/op_kernel/advance_step_spec.h(生产代码)
- 缺陷描述：10处PipeBarrier<PIPE_V>()后有多余分号`;;`，虽不影响运行时行为但表明copy-paste错误
- 修复模式：删除冗余分号
- 可审查性：高
- 审查规则建议：-Wextra-semi检查多余分号

### #319 338f8b9a fix eigen compilation using cuda keyword
- 根因类别：第三方库构建配置缺陷
- 涉及文件：cmake/third_party/eigen.cmake
- 缺陷描述：Eigen作为header-only库通过ExternalProject_Add引入，但未设置CONFIGURE_COMMAND ""和BUILD_COMMAND ""，默认会尝试编译含CUDA关键字的源码导致非CUDA环境编译失败
- 修复模式：显式禁用ExternalProject的configure和build步骤
- 可审查性：中
- 审查规则建议：header-only第三方库通过ExternalProject引入时应显式禁用configure和build

### #320 301b0cdd host ut error 修复
- 跳过：纯UT修复

## 第17轮 (commit 321-340)

### #321 c1f867025f 修复日志打印有歧义问题
- 跳过：纯日志文案优化，将error message从模糊描述改为更详细的提示

### #322 402244ae71 fix rnn ut
- 跳过：纯UT修复（cmake构建系统新增rnn类别支持+UT测试文件更新）

### #323 9a2ee5e935 fix lisence
- 跳过：纯license修复（第三方软件声明拆分+文件头版权声明添加）

### #324 6593ca1b89 fix run example
- 根因类别：CI脚本逻辑缺陷（空数组误判为失败）
- 涉及文件：build.sh
- 缺陷描述：build_example函数中，当某个算子在指定mode下没有example用例时（files数组为空），脚本走到`failed_example != "" || success_example == ""`判断分支，误报为"Run example failed"并exit 1，导致整个构建流程异常中断
- 修复模式：新增`${#files[@]} -eq 0`前置条件分支，当没有example时仅输出提示信息而非报错退出
- 可审查性：高
- 审查规则建议：shell脚本中对空数组/空列表的边界条件处理，避免将"无数据"误判为"失败"

### #325 578fecacdb add fix AdaptiveMaxPool3d\Grad example
- 跳过：纯新增example文件（4个新增文件）

### #326 b49c1c1eb6 fix quant_matmul_reduce_sum_weight_nz example bug
- 跳过：纯新增example文件（1个新增文件）

### #327 9e27b020f1 fix ut fail in arm env
- 根因类别：构建配置缺陷（RPATH导致链接到不完整桩函数so）
- 涉及文件：tests/ut/op_host/CMakeLists.txt
- 缺陷描述：ophost UT可执行文件构建时默认嵌入了BUILD_RPATH，在ARM环境下运行时优先链接到构建目录中不完整的桩函数共享库导致UT失败
- 修复模式：设置CMake属性SKIP_BUILD_RPATH TRUE，使运行时从环境变量查找正确的共享库
- 可审查性：中
- 审查规则建议：CMake构建配置中检查RPATH设置对跨平台（x86 vs ARM）部署的影响

### #328 71da36591b fix conv ut
- 跳过：纯UT修复（头文件引用路径从简单文件名改为相对路径）

### #329 760c32539e FixInstructions
- 根因类别：废弃API未迁移
- 涉及文件：norm/masked_softmax_with_rel_pos_bias/op_kernel/masked_softmax_with_rel_pos_bias_BW.h
- 缺陷描述：3处使用了小写函数风格的pipe_barrier(PIPE_V)调用，这是AscendC已废弃的旧版同步原语写法，在新版本编译环境下可能导致编译失败或行为不正确
- 修复模式：替换为模板函数风格PipeBarrier<PIPE_V>()
- 可审查性：高
- 审查规则建议：全局搜索pipe_barrier(调用，批量替换为PipeBarrier<>模板写法

### #330 0e183e0410 修改avgPoolV2Grad proto原型注释，解决geir通路失败
- 根因类别：proto注释规范错误导致功能通路失败
- 涉及文件：pooling/avg_pool_v2_grad/op_graph/avg_pool_v2_grad_proto.h
- 缺陷描述：ksize和strides属性注释描述不准确（"length is 1, 2 or 4"实际仅支持长度4），proto注释中的@li标注被工具链解析为算子约束，错误的维度描述导致geir通路校验失败
- 修复模式：修正注释从"length is 1, 2 or 4"为"length is 4"
- 可审查性：高
- 审查规则建议：proto头文件中的@li注释会被工具链解析为算子约束规范，修改时需验证geir通路

### #331 adee08cd65 二进制发布重复场景整改
- 跳过：JSON配置文件清理（删除重复的binary.json配置项），非代码逻辑缺陷

### #332 124cfa7b2e 修改string类型直接赋值给char*导致的告警
- 根因类别：类型安全缺陷（const correctness违反）
- 涉及文件：matmul/common/op_host/matmul_common_infershape.cpp
- 缺陷描述：将字符串字面量（const char[]）赋值给constexpr char*（非const指针），在C++11及以后标准中不合法，可能导致编译告警且通过该指针写入会产生UB
- 修复模式：将`constexpr char* FUSED_OP_TYPE_SWIGLU = "swiglu"`改为`constexpr char FUSED_OP_TYPE_SWIGLU[] = "swiglu"`
- 可审查性：高
- 审查规则建议：检测constexpr char*或char*直接用字符串字面量初始化的模式

### #333 d2bf0d115c 拦截bias变化导致进入matmul v3kernel bin找不到的问题
- 根因类别：逻辑错误（使用了错误的数据源进行比较）
- 涉及文件：matmul/common/op_host/op_api/matmul_util.cpp
- 缺陷描述：CheckMMV3NzNzNdSupport中检查混精度时，bias类型比较使用了ori_info（原始输入类型），但bias的dtype可能已被修改（如fp16提升为fp32），应使用support_info（实际参与kernel计算的类型）
- 修复模式：将比较对象从mmOpInfo.ori_info改为mmOpInfo.support_info
- 可审查性：中
- 审查规则建议：当存在ori_info和support_info两套数据源时，审查条件判断中引用的字段是否来自正确的数据源

### #334 33b30e51b9 aclnnAddmmGetWorkspaceSize接口bias不支nz格式未对异常format拦截
- 根因类别：输入校验缺失（format未检查）
- 涉及文件：matmul/mat_mul_v3/op_host/op_api/aclnn_addmm.cpp
- 缺陷描述：CheckMatmul函数只检查了mat1和mat2的维度和format，但未检查self（bias）的format。当self的storage format为FRACTAL_NZ时后续kernel不支持，导致执行报错
- 修复模式：在CheckMatmul中增加self参数，添加对self->GetStorageFormat() == FORMAT_FRACTAL_NZ的拦截判断
- 可审查性：高
- 审查规则建议：aclnn接口的Check函数应对所有输入tensor的format进行合法性校验，不能只校验部分tensor

### #335 4fed4582df matmul相关算子编译告警消除，避免除0风险
- 根因类别：除零风险 + 未使用参数
- 涉及文件：matmul/transpose_batch_mat_mul/op_host/op_tiling/transpose_batch_mat_mul_base_tiling.cpp, matmul/quant_matmul/op_host/op_api/aclnn_quant_matmul.cpp, matmul/matmul_compress/op_host/op_api/aclnn_matmul_compress.cpp
- 缺陷描述：(1) ResetBasicBlock函数未校验tempBaseM和tempBaseN为0的情况，后续有除法运算；(2) BaseLoadBalance中使用普通除法/而非安全的FloorDiv；(3) ProcessEmptyTensor函数有未使用参数x2
- 修复模式：添加零值前置检查+替换为ops::FloorDiv安全除法+删除未使用参数
- 可审查性：高
- 审查规则建议：检测整数除法运算前是否有零值检查；有安全除法工具函数时检测直接使用/的场景

### #336 c4d5efbd1c gather_v2丢失分支补充
- 根因类别：遗漏分支（tiling key分支不完整）
- 涉及文件：index/gather_v2/op_kernel/gather_v2_apt.cpp
- 缺陷描述：TILING_KEY_VAR分支处理中缺少SIMD_LAST_GATHER_B16_TILING_KEY（int16_t类型），已有B8和B32但B16被遗漏，导致16-bit数据类型用例精度输出为0
- 修复模式：在B8和B32分支之间补充B16分支
- 可审查性：高
- 审查规则建议：使用TILING_KEY进行多分支模板特化时，检查B8/B16/B32/B64等类型分支是否完整覆盖

### #337 00e890438a 解决ConvBackward性能重复上报问题
- 根因类别：DFX宏放置位置不当导致重复采集
- 涉及文件：conv/convolution_backward/op_api/convolutionbackward.cpp, conv/convolution_forward/op_host/op_api/convolution.cpp
- 缺陷描述：L0_DFX性能采集宏放在函数开头，但后续存在"小算子拼接"分支路径内部也有独立的性能采集，导致走拼接case的性能数据被重复采集
- 修复模式：将L0_DFX宏从函数入口移到参数适配/转换逻辑之后、实际kernel调用之前的位置
- 可审查性：中
- 审查规则建议：DFX性能采集宏应放在算子实际执行路径入口处，确保不会对分支路径重复采集

### #338 240a589531 aclnnAddbmm异常场景bias未拦截
- 根因类别：输入校验缺失（broadcast shape校验不完整）
- 涉及文件：matmul/batch_mat_mul_v3/op_host/op_api/aclnn_addbmm.cpp
- 缺陷描述：CheckBroadCast函数计算出broadcastShape和bmmLastTwoShape后，缺少两者是否相等的校验。当bias broadcast后的shape与batch matmul输出最后两维shape不一致时，算子未拦截给出错误结果
- 修复模式：增加broadcastShape != bmmLastTwoShape的判断，不匹配时返回false
- 可审查性：高
- 审查规则建议：包含broadcast逻辑的Check函数应验证广播推导后的结果shape与预期shape的一致性

### #339 07a8e44ed4 解除conv2dv2和common目录的互相依赖问题
- 跳过：代码架构重构（解耦循环依赖），非运行时缺陷修复

### #340 ad9d2ab2da modify GroupedDynamicBlockQuant InferShape动态shape失败问题
- 根因类别：动态shape推导不完备 + 属性类型不匹配
- 涉及文件：quant/grouped_dynamic_block_quant/op_host/grouped_dynamic_block_quant_infershape.cpp
- 缺陷描述：(1) InferShape只考虑了dim1未知情况，groupNum也可能未知（动态shape下GroupList长度不确定），groupNum为-1时计算结果错误；(2) 属性指针类型用int32_t获取但实际是int64_t；(3) 缺少rowBlockSize/colBlockSize空指针和非法值校验
- 修复模式：重构为SetShapeDimTwo/SetShapeDimThree独立判断每个维度是否UNKNOWN；属性类型从int32_t*改为int64_t*；添加空指针和<=0校验
- 可审查性：高
- 审查规则建议：InferShape函数中涉及动态shape时检查所有参与维度计算的变量是否都考虑了UNKNOWN情况；GetAttrPointer模板类型必须与注册时属性类型一致

## 第18轮 (commit 341-380, 最终轮)

### #341 cdb5ba5f Conv3DTranspose算子db场景添加同步解决用例多次批跑偶现AICError的问题
- 根因类别：并发同步缺失（double buffer无ping-pong机制）
- 涉及文件：conv/conv3d_backprop_input_v2/op_kernel/arch35/convolution_3d_backprop/conv3d_bp_func.h, conv3d_bp_impl_base.h, conv_bp_sub_func.h
- 缺陷描述：Conv3DTranspose在db场景下L0C缓冲区使用单队列outQueL0C_管理，无ping-pong双缓冲同步，多次批跑时AIC/AIV核间读写冲突导致偶发AICError
- 修复模式：单队列拆分为l0cPing_/l0cPong_双队列，新增l0cPingPongFlag_标志位交替切换，AllocTensor/EnQue/FreeTensor/DeQue全部增加ping-pong分支
- 可审查性：低
- 审查规则建议：使用double buffer(Pbuffer>1)时检查是否有对应ping-pong同步机制；TQue深度为1但Pbuffer>1可能存在同步缺陷

### #342 4dbc8196 解决fusedmatmul mnk维度超过边界的问题
- 根因类别：整数类型溢出(int32截断shape维度)
- 涉及文件：matmul/fused_mat_mul/op_host/fused_mat_mul_infershape.cpp
- 缺陷描述：InferShapeForFusedMatMul中a_m/b_n/a_k/b_k声明为int(32位)，大矩阵维度超INT32_MAX时溢出
- 修复模式：int改为int64_t
- 可审查性：高
- 审查规则建议：shape/维度变量禁止使用int/int32_t，统一用int64_t；扫描GetDim()返回值赋给int类型的模式

### #343 9f307dfd 误开放的aclnn接口整改
- 根因类别：构建配置错误（接口暴露控制）
- 涉及文件：norm/dua_quantize_add_layer_norm/op_host/CMakeLists.txt, norm/inplace_add_layer_norm/op_host/CMakeLists.txt, norm/quantize_add_layer_norm/op_host/CMakeLists.txt
- 缺陷描述：三个不应有aclnn公开接口的算子ACLNNTYPE误设为aclnn，暴露了不应存在的头文件
- 修复模式：ACLNNTYPE aclnn改为aclnn_exclude
- 可审查性：高
- 审查规则建议：无对应docs/目录的算子ACLNNTYPE不应为aclnn；建立算子接口清单与CMake配置一致性检查

### #344 0bd3fe0b aclnnAddmmWeightNz/aclnnMatmulWeightNz异常拦截
- 根因类别：输入校验缺失（多项：dtype过宽/nullptr/维度/cubeMathType）
- 涉及文件：matmul/mat_mul_v3/op_host/op_api/aclnn_addmm.cpp, aclnn_matmul.cpp
- 缺陷描述：(1)WeightNz接口复用通用dtype校验允许了不支持的int8/fp32 (2)必选输入nullptr未拦截致core dump (3)维度校验缺失 (4)cubeMathType异常值未拦截
- 修复模式：新增CheckWeightNzDtypeValid独立校验(仅fp16/bf16)；维度!=2检查；空指针检查提前；新增socRule->CheckInput
- 可审查性：高
- 审查规则建议：不同变体接口应使用独立dtype校验函数而非复用通用函数；aclnn函数第一步应做nullptr检查

### #345 b7e91a6e Ascend950上ctcloss算子对标竞品不做尾轴向上8对齐
- 跳过-纯文档

### #346 49de9c80 冗余.o文件修改
- 跳过-纯资源清理（删除冗余binary json配置文件）

### #347 d8b73e6e 算子文档与算子具体代码不一致
- 跳过-纯文档

### #348 f43f60ec 清理冗余的二进制文件和修改cmake中错误的aclnntype
- 根因类别：构建配置错误（接口暴露控制）+ 冗余资源
- 涉及文件：norm/batch_norm_elemt/op_host/CMakeLists.txt等5个norm算子CMakeLists.txt + group_norm_grad binary json
- 缺陷描述：同#343模式，多个不应对外暴露aclnn接口的算子ACLNNTYPE误设为aclnn；group_norm_grad存在冗余binary配置
- 修复模式：ACLNNTYPE aclnn改为aclnn_exclude，删除冗余json
- 可审查性：高
- 审查规则建议：同#343

### #349 af87e30d aclnnGemm算子异常场景未拦截
- 根因类别：输入校验缺失（格式校验）
- 涉及文件：matmul/gemm/op_host/op_api/aclnn_gemm.cpp
- 缺陷描述：aclnnGemm缺少Format校验，NZ格式输入本应拦截但能跑通
- 修复模式：新增CheckFormat函数校验所有输入输出必须为FORMAT_ND
- 可审查性：高
- 审查规则建议：每个aclnn算子CheckParams应包含Format校验步骤

### #350 4e5a8b39 修改unused variable和unused parameter问题
- 根因类别：控制流缺陷（非void函数缺return）+ 编译警告
- 涉及文件：norm/ada_layer_norm_v2/op_host/op_api/aclnn_ada_layer_norm_v2.cpp等4个文件
- 缺陷描述：AdaLayerNormV2Calculate声明返回aclnnStatus但缺return语句，且调用处未检查返回值(control reaches end of non-void function)；其他是unused variable/parameter
- 修复模式：添加return ACLNN_SUCCESS；调用处增加CHECK_RET；移除未使用变量
- 可审查性：高
- 审查规则建议：开启-Werror=return-type将此类问题升级为编译错误

### #351 42f47c17 aclnnEinsum算子资料改正、aclnnMatmulWeightNz算子异常场景错误码改正
- 根因类别：错误码语义不匹配
- 涉及文件：matmul/mat_mul_v3/op_host/op_api/aclnn_matmul.cpp
- 缺陷描述：BuildMatMulWeightNzGraph返回nullptr时使用ACLNN_ERR_INNER_NULLPTR错误码，但实际是参数非法导致，应返回ACLNN_ERR_PARAM_INVALID
- 修复模式：错误码从ACLNN_ERR_INNER_NULLPTR改为ACLNN_ERR_PARAM_INVALID
- 可审查性：中
- 审查规则建议：用户参数导致的校验失败应返回PARAM相关错误码而非INNER错误码

### #352 d303c3a0 删除qbmmv4算子def错误描述
- 根因类别：算子定义配置多余（不支持平台的配置）
- 涉及文件：matmul/quant_batch_matmul_v4/op_host/quant_batch_matmul_v4_def.cpp
- 缺陷描述：def中包含不支持的310p平台配置config_310p(约82行)，可能导致图编译阶段错误匹配
- 修复模式：删除整段config_310p配置代码
- 可审查性：高
- 审查规则建议：def中AddConfig平台应与binary config的compute_units列表一致

### #353 5baade56 SegmentSum算子SIMT模板tiling显式设置workspace解决aic error问题
- 根因类别：框架约定遗漏（必须override的方法未实现）
- 涉及文件：index/segment_sum/op_host/arch35/segment_sum_simt_tiling.cpp, .h
- 缺陷描述：SIMT模板不用workspace但未实现GetWorkspaceSize()显式设为0，框架要求必须显式设置否则aic error
- 修复模式：新增GetWorkspaceSize() override，workspace[0]=0
- 可审查性：中
- 审查规则建议：所有tiling类必须override GetWorkspaceSize()；扫描继承tiling基类但未实现该方法的子类

### #354 bd286614 处理avgpool3d算子不生成二进制文件的问题
- 根因类别：头文件路径未同步（目录重命名后）
- 涉及文件：pooling/avg_pool3_d/op_kernel/avg_pool3_d_apt.cpp
- 缺陷描述：pool_3d_common目录从v35重命名为arch35后，10个#include路径仍引用旧路径，编译失败
- 修复模式：../pool_3d_common/v35/改为../pool_3d_common/arch35/
- 可审查性：高
- 审查规则建议：目录重命名PR应自动扫描所有引用旧路径的#include语句

### #355 692f43a9 处理adaptivemaxpool3d算子tiling判断错误
- 根因类别：逻辑运算符错误（||应为&&，恒真条件）
- 涉及文件：pooling/adaptive_pool3d_common/op_host/arch35/adaptive_pool3d_tiling.cpp
- 缺陷描述：条件npuArch!=A || npuArch!=B恒为true（De Morgan），所有平台都返回GRAPH_PARAM_INVALID
- 修复模式：移除多余条件，简化为只检查npuArch!=DAV_3510
- 可审查性：高
- 审查规则建议：x!=A||x!=B是经典恒真逻辑错误，可用clang-tidy/cppcheck的always-true检测

### #356 961b48b6 修改quantbatchmatmulinplaceadd算子soc version命名问题
- 根因类别：配置字符串拼写错误
- 涉及文件：scripts/kernel/binary_config/ascendc_config.json
- 缺陷描述：compute_units中soc名写为ascend910_95（错误），正确为ascend950
- 修复模式：修正字符串
- 可审查性：高
- 审查规则建议：维护合法soc version枚举白名单，CI校验所有json配置的compute_units值

### #357 a84247e8 uniqueConsecutive算子return_idx为true拦截信息修改
- 根因类别：能力路由缺失（aicore不支持的场景未自动fallback到aicpu）
- 涉及文件：index/unique_consecutive/op_host/arch35/unique_consecutive_tiling_arch35.cpp, unique_consecutive_def.cpp
- 缺陷描述：return_idx=true时aicore不支持但缺少CheckSupport机制路由到aicpu，仅在tiling阶段拦截
- 修复模式：def中新增CheckSupport回调+NeedCheckSupportFlag(true)实现自动fallback
- 可审查性：中
- 审查规则建议：aicore有条件支持的算子应通过CheckSupport声明而非仅在tiling拦截

### #358 10bdb67b 修改aclnnDequantSwigluQuant.md文档
- 跳过-纯文档

### #359 300efeb1 aclnnConvolution测试Transpose1D带bias场景走matmul导致功能报错失败
- 根因类别：算子路由条件不完整
- 涉及文件：conv/convolution_forward/op_host/op_api/aclnn_convolution.cpp
- 缺陷描述：IsSupportConvTranspose1DToBmm未考虑weight L维>1且带bias的情况，此时Transpose导致维度不匹配matmul报错
- 修复模式：新增条件weight.shape[L_DIM]!=1&&bias!=nullptr时返回false不走matmul路径
- 可审查性：中
- 审查规则建议：算子优化路由中需确保转换后shape约束在所有可选输入(bias)存在时仍成立

### #360 ce7bfa39 添加qbmminplaceadd aclnn文件及tiling异常拦截
- 根因类别：tiling校验缺失（多项）
- 涉及文件：matmul/quant_batch_matmul_inplace_add/op_host/op_tiling/, op_api/等
- 缺陷描述：(1)未校验transA/transB合法组合 (2)未校验输出dtype必须DT_FLOAT (3)未校验shape维度和k轴一致性 (4)mx quant下scale shape未校验 (5)pertokenShape可能nullptr未拦截
- 修复模式：新增CheckShapeVaild/CheckParamsForMxQuant等校验方法
- 可审查性：高
- 审查规则建议：tiling的AnalyzeInputs/AnalyzeDtype/AnalyzeAttrs应对所有输入完整校验

### #361 f98df9bc 解决qbmm空tensor处理问题单
- 根因类别：边界条件缺失/多通路行为不一致
- 涉及文件：matmul/quant_batch_matmul_v3/op_api/aclnn_quant_matmul_v4.cpp, matmul/quant_batch_matmul_v4/op_host/op_api/aclnn_quant_matmul_v5.cpp
- 缺陷描述：静态图通路m/n=0返回空tensor正常，但aclnn直调/动态图/geir图通路拦截报错，行为不一致
- 修复模式：aclnn入口添加平台判断+维度为0的early return，区分NZ/ND格式分别处理
- 可审查性：中
- 审查规则建议：检查算子不同调用通路对边界输入(零维/空tensor)的处理是否一致

### #362 3da6cbf9 删除fastkernel冗余代码文件
- 跳过-纯代码清理

### #363 0e254120 删除多余日志
- 跳过-UT桩代码日志级别调整

### #364 4469304e BatchNormGrad算子修改UB空间计算错误
- 根因类别：数值计算错误（缺sizeof乘数+缺对齐）
- 涉及文件：norm/batch_norm_grad_v3/op_host/batch_norm_grad_v3_splitload_tiling.cpp等3个tiling文件
- 缺陷描述：CalcBubBlock()中ubTensorNotRelateChannelBlock=tmpChannelNum*noRelateBlockNum缺少*sizeof(float)，UB空间计算偏小；tmpChannelNum未做block对齐可能越界
- 修复模式：乘上sizeof(float)；新增GetAlignValue()对齐处理
- 可审查性：高
- 审查规则建议：UB/内存空间计算必须乘sizeof(dtype)；偏移/大小计算必须考虑对齐

### #365 908f0227 资料中yoffset描述有误
- 跳过-纯文档

### #366 f0eb3726 修改aclnn问题
- 跳过-纯文档

### #367 a17e0ee9 补齐A5算子op_graph下面的cmakelist、同步修改apply_adam_w算子geir通路动态用例失败问题
- 根因类别：构建配置缺失+冗余InferShape导致geir通路失败
- 涉及文件：多个op_graph/CMakeLists.txt(新增), optim/apply_adam_w/op_host/apply_adam_w_infershape.cpp(删除)
- 缺陷描述：(1)多个A5算子缺少op_graph/CMakeLists.txt导致graph plugin无法编译 (2)apply_adam_w冗余空InferShape与框架默认行为冲突致geir通路失败
- 修复模式：补齐CMakeLists.txt；删除冗余infershape.cpp
- 可审查性：中
- 审查规则建议：新增算子检查op_graph/CMakeLists.txt是否补齐；空InferShape应审视必要性

### #368 118b959d aclnnMatmulWeightNz资料错误修改
- 跳过-纯文档

### #369 6b9a0c39 修改代码扫描的问题
- 根因类别：整数溢出(GM偏移uint32) + 测试代码自比较bug
- 涉及文件：norm/add_rms_norm/op_kernel/add_rms_norm_multi_n.h, add_rms_norm_split_d.h, norm/group_norm_grad/op_kernel/group_norm_grad.h, norm/add_layer_norm/tests/ut/op_kernel/compare_data.py
- 缺陷描述：(1)kernel GM偏移用uint32_t计算乘法，大数据量溢出 (2)compare_data.py中np.isclose(tmp_gold,tmp_gold)自比较（第一参数应为tmp_out），精度比对永远PASS
- 修复模式：(1)GM偏移改为int64_t (2)修正np.isclose参数为tmp_out
- 可审查性：高
- 审查规则建议：(1)GM偏移计算禁止uint32乘法 (2)测试比对函数两参数不应相同（自比较检测）

### #370 1f044298 修改addmm接口参数和文档描述不符问题
- 根因类别：API签名与文档不一致(const修饰符)
- 涉及文件：matmul/mat_mul_v3/op_host/op_api/aclnn_addmm.cpp, aclnn_addmm.h
- 缺陷描述：stream参数声明为const aclrtStream与文档和其他算子不一致
- 修复模式：去掉stream参数的const修饰符
- 可审查性：高
- 审查规则建议：API声明参数修饰符应与文档和同类接口保持一致

### #371 e9ef7d29 renorm异常校验及ScaledMaskedSoftmaxGradV2增加拦截信息
- 根因类别：输入校验缺失(dim越界) + 错误信息缺失
- 涉及文件：norm/renorm/op_host/op_api/renorm.cpp, vfusion/scaled_masked_softmax_grad_v2/op_host/op_api/aclnn_scaled_masked_softmax_backward.cpp
- 缺陷描述：(1)renorm InferShape未校验dim越界/负值直接使用导致越界访问 (2)RenormAiCore返回值未检查 (3)CheckFormat失败无OP_LOGE信息
- 修复模式：添加dim范围校验；检查返回值；增加错误日志
- 可审查性：高
- 审查规则建议：用户输入参数(dim/axis)必须做边界校验；校验失败必须输出错误信息

### #372 a947b6ea 106 135 162 issue问题修改
- 根因类别：构建配置缺失(类别白名单) + 编译器指令缺失(__restrict)
- 涉及文件：cmake/ut.cmake, loss/mse_loss_v2/op_kernel/mse_loss_v2_base.h等
- 缺陷描述：(1)supportedCategory列表缺少loss类别致loss UT不被发现 (2)tilingData指针缺__restrict限定符
- 修复模式：添加loss到类别列表；指针加__restrict
- 可审查性：高
- 审查规则建议：新增算子类别目录时检查构建系统类别白名单是否同步更新

### #373 bf19309a 修改md（易用性问题）
- 跳过-纯文档

### #374 1d67b94e 修改opapi UT引用路径问题
- 跳过-纯UT路径修改

### #375 31501829 添加index类算子部分opapi UT及修改aclnnEmbeddingBag中的错误注释
- 根因类别：注释复制粘贴错误 + 构建配置缺陷
- 涉及文件：index/embedding_bag/op_host/op_api/aclnn_embedding_bag.h, cmake/gen_ops_info.cmake
- 缺陷描述：(1)aclnn头文件注释中函数名写成aclnnInplaceIndexCopyGetWorkspaceSize应为aclnnEmbeddingBagGetWorkspaceSize (2)cmake中有不合理的skip逻辑导致部分算子binary不被编译
- 修复模式：修正注释函数名；删除cmake不合理skip逻辑
- 可审查性：高
- 审查规则建议：API注释中引用的函数名必须与实际函数名匹配

### #376 92060846 pool目录头文件错误修改
- 根因类别：注释复制粘贴错误（大面积函数名引用错误）
- 涉及文件：pooling/adaptive_max_pool3d_grad/..aclnn_adaptive_max_pool2d_backward.h等4个pool头文件
- 缺陷描述：多个pool算子头文件注释中"由第一段接口xxx获取"的函数名写错（如写成aclnnMaxPool2DWithIndicesBackwardGetWorkspaceSize、aclnnAtan2GetWorkspaceSize等）
- 修复模式：修正注释中函数名引用
- 可审查性：高
- 审查规则建议：两段式接口头文件注释应自动生成或有校验脚本确保函数名一致性

### #377 3d71222b 修改有误的注释内容
- 根因类别：注释复制粘贴错误（大面积，11个文件）
- 涉及文件：norm/add_rms_norm_dynamic_quant/.., quant/ascend_quant_v2/.., quant/dynamic_quant/.., quant/quantize/.., quant/trans_quant_param_v2/..等11个头文件
- 缺陷描述：批量注释错误，如aclnnAscendQuant写成aclnnAscendAntiQuantGetWorkspaceSize，aclnnQuantize写成aclnnTopkGetWorkspaceSize等
- 修复模式：批量修正注释中的函数名引用
- 可审查性：高
- 审查规则建议：同#376，需自动化校验

### #378 964be4dd 修改norm下样例代码问题
- 根因类别：示例代码缺陷（数据类型不匹配+资源泄漏）
- 涉及文件：norm/deep_norm/examples/test_aclnn_deep_norm.cpp, norm/deep_norm_grad/examples/test_aclnn_deep_norm_grad.cpp
- 缺陷描述：(1)deep_norm_grad示例vector<int32_t>但创建tensor传ACL_FLOAT，类型不匹配 (2)deep_norm示例缺aclrtFree(betaDeviceAddr)内存泄漏
- 修复模式：int32_t改float；补充aclrtFree调用
- 可审查性：高
- 审查规则建议：示例代码vector类型必须与aclDataType匹配；aclrtMalloc/aclrtFree必须配对

### #379 e14c4eee 修改index类算子linear_index_v2的kernel UT中的error日志误报
- 根因类别：UT代码多处bug（tiling定义不匹配/数据路径错误/GmBuffer未指定大小）
- 涉及文件：index/linear_index_v2/op_kernel/linear_index_v2.h, 多个UT文件
- 缺陷描述：(1)SetGlobalBuffer缺dataNum_参数致error日志 (2)UT tiling结构体与实际不匹配 (3)数据生成脚本和路径引用错误
- 修复模式：修正SetGlobalBuffer参数；重命名结构体；修正路径；补齐tiling参数
- 可审查性：中
- 审查规则建议：SetGlobalBuffer应始终指定buffer大小参数；UT tiling结构体应与产品代码同步

### #380 c1d89aa2 修改index类算子kernel UT中的error日志误报
- 跳过-纯UT修改

---

## ops-nn-dev - 612条缺陷


## 分析进度

- 总缺陷提交：612条
- 已处理：476条（第1-502行），其中分析393条，跳过83条
- 剩余：136条（第503-612行）

---

### a70d8b16d5e6800d796b98bf5b01999d98f70362 fix hoAL10 cal error
- 根因类别：计算逻辑错误
- 涉及文件：conv/common/op_host/op_tiling/arch35/conv_api_tiling_base.cpp, conv/conv2d_v2/op_host/op_tiling/arch35/ 多文件(共14文件)
- 缺陷描述：计算L1缓存中最小所需的feature map高度hoAL1min时，当wo < m0时通过CeilDiv(m0, wo)计算得到的值可能超过实际ho值，导致hiAL1min超出实际输入高度，L1 size校验产生错误判断。同时opImplMode做位运算前缺少非负校验，且使用了错误的k0来源(cubeInfo.k0而非按weight dtype查询的k0)
- 修复模式：对hoAL1min增加std::min(..., ho)上界约束；对opImplMode增加>=0校验并转uint32_t后做位操作；将k0改为从CUBE_MKN_TAB按实际weight dtype动态查询
- 可审查性：高
- 审查规则建议：1) tiling参数通过除法/向上取整计算时，检查结果是否需要用实际shape维度做上界clamp 2) 有符号整数做位运算前应校验非负或转无符号类型 3) 使用硬件参数表时检查是否与实际数据类型匹配

### 12e796c4fc0927379fa2590c5ec5b69f7b36c54e fix foreach_mul_scalar_v2 aclnn storage shape bug
- 根因类别：shape处理错误(storage shape未同步)
- 涉及文件：foreach/foreach_mul_scalar/op_host/op_api/aclnn_foreach_mul_scalar_v2.cpp
- 缺陷描述：CheckShape函数校验self和out的view shape一致性后，未将out tensor的storage shape同步为view shape，非连续tensor场景下后续计算使用错误的storage shape
- 修复模式：增加SetStorageShape(GetViewShape())同步
- 可审查性：高
- 审查规则建议：算子对输出tensor做shape校验时，检查是否需要同步更新storage shape，特别是foreach类算子

### fa03888b0d0144e8d59b0f7bcf7e1486b00768ea add embeddingBag bugFix
- 根因类别：多核初始化错误 + 计算逻辑错误 + 资源管理错误
- 涉及文件：index/embedding_bag/op_kernel/arch35/ 多文件(6文件)
- 缺陷描述：(1) 空bag时每个核独立对全局内存做零初始化存在竞争 (2) validIndicesFactorNumber跨loop累计导致后续loop有效行数不正确 (3) TQue不需要队列语义应用TBuf (4) DisposalBagSize直接写全局内存不安全 (5) tiling中UB大小错误除以DOUBLE_BUF
- 修复模式：全局初始化移到core 0+SyncAll；替换为循环内局部变量；TQue改TBuf；重写为DMA拷贝；去掉多余DOUBLE_BUF除法
- 可审查性：中
- 审查规则建议：1) 多核场景全局内存初始化应由指定核完成并加同步屏障 2) 跨循环累积计数器需确认语义是全局累积还是每次迭代独立 3) 不需要队列语义时用TBuf而非TQue

### 997a5518461caf365b6907df890cdaccdbb6609a fix dsq-aclnn资料
- 纯文档修复，跳过

### 93690b823fa9513deb3a373753722fe0599f9218 bugfix wqbmmv2 only support x contiguous
- 根因类别：平台特定约束校验遗漏
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_host/op_api/aclnn_weight_quant_batch_matmul_v2.cpp
- 缺陷描述：DAV_3510平台不支持x tensor转置，但校验逻辑transposeX || IsContiguous(x)对DAV_3510不正确，该平台要求x必须连续且不能转置
- 修复模式：增加平台判断分支，DAV_3510使用!transposeX && IsContiguous(x)
- 可审查性：高
- 审查规则建议：算子存在平台差异化约束时，检查校验逻辑是否按平台区分。添加新平台支持时审查所有输入约束

### 7004b8e99922340407fb24d9bb6c3fbef0d94f8d 修复QuantBatchMatmulV3纯CUBE模板低阶API的若干问题
- 根因类别：常量定义重复 + 条件编译逻辑错误 + 类型支持遗漏
- 涉及文件：matmul/common/cmct/block/ 多文件, quant_batch_matmul_constant.h, quant_batch_matmul_v3_apt.cpp 等(7文件)
- 缺陷描述：(1) 多文件重复定义相同常量(IDX_M_TILE_IDX等)与统一头文件命名不一致 (2) 条件编译#if != 应使用#else (3) IsCapable缺少DT_INT64 scale dtype支持 (4) CUBE模板使用条件宏不一致
- 修复模式：常量集中到统一头文件；#if改#else；添加DT_INT64判断；引入CUBE_TEMPLATE_ND宏
- 可审查性：中
- 审查规则建议：1) 常量定义集中单一头文件避免重复 2) 互斥条件编译分支用#else 3) 新增数据类型时检查所有能力判断/路由逻辑同步更新

### 08b9ebb1123e0e89001b5bc2fadf8e34a9089437 修复onnx算子ConvTranspose图通路动态shape在2D场景执行失败的问题
- 根因类别：维度场景适配遗漏(2D/3D混用)
- 涉及文件：conv/common/op_host/conv_backprop_infershape.cpp/.h, conv/conv3d_transpose_v2/op_host/conv3d_transpose_v2_infershape.cpp
- 缺陷描述：2D场景tensor扩展为5D(D=1)，但CheckOutputAllZero要求所有维度为0才推断输出shape，D=1导致返回false，infershape流程失败。另外输入索引硬编码为0而非正确常量kConv3DTransposeXIdx(=1)
- 修复模式：新增CheckOutputAllZeroFrom2D函数处理D=1的特殊情况；修复索引为正确常量
- 可审查性：高
- 审查规则建议：1) 同时支持2D和3D且通过维度扩展统一处理时，所有依赖维度值的判断需考虑扩展维度默认值 2) 输入索引用命名常量不硬编码

### f4ffed6948841b5acd20598ea9bf9dc38152da9e fix maxpoolwithargmaxv3 CHW HWC
- 根因类别：输入format支持遗漏(3D tensor未处理)
- 涉及文件：pooling/max_pool_with_argmax_v3/op_host/arch35/max_pool_with_argmax_v3_tiling_base.cpp
- 缺陷描述：tiling逻辑仅支持4维输入(NCHW/NHWC)，3维tensor(CHW/HWC)维度校验直接失败
- 修复模式：放宽维度校验允许3维；3维时推断format并将batch设为1，映射到已有4维逻辑
- 可审查性：高
- 审查规则建议：维度校验不应仅支持一种维度数，需考虑框架实际传入的维度变体(如batch维省略的3D输入)

### e2fc1fecb181aaea32809992a5da09615cf56ae1 group场景精度问题
- 根因类别：条件判断错误(边界条件)
- 涉及文件：conv/conv3d_backprop_input_v2/op_kernel/arch35/conv3d_backprop_input_v2/conv3d_dx_rowc_block.h
- 缺陷描述：group卷积场景下，>=判断将合法尾部core和超范围core合并处理，超范围core使用错误数据量计算导致精度问题
- 修复模式：拆分为==处理尾部core和>直接return两个分支
- 可审查性：高
- 审查规则建议：多核任务分配中，必须同时处理"合法tail core"和"超范围idle core"，不能合并到同一分支

### dc3a8850e7817cd926e0aeba0c9ee350d81bb34a fix bug for DynamicQuant
- 根因类别：输入校验遗漏 + 日志错误
- 涉及文件：quant/dynamic_quant/op_host/dynamic_quant_tiling_arch35.cpp, op_api/aclnn_dynamic_quant.cpp
- 缺陷描述：(1) groupNum仅校验上界未校验<=0的非法值 (2) 错误日志"pertensor"与实际检查条件"perchannel"矛盾
- 修复模式：校验改为groupNum <= 0 || groupNum > MAX_EXPERT_NUM；日志修正为perchannel
- 可审查性：高
- 审查规则建议：1) 数值范围校验应同时包含上下界，特别是用于除法/索引的参数 2) 错误日志条件描述必须与实际判断条件一致

### db5a08d10f260616567d901e99fdca51e191e0fa 修改新包__bf16未定义的报错
- 根因类别：类型定义缺失/编译兼容性
- 涉及文件：pooling/adaptive_max_pool3d/op_kernel/adaptive_max_pool3d_big_pool.h
- 缺陷描述：CPU侧单元测试编译时(__CCE_KT_TEST__宏)，__bf16硬件内建类型未定义导致编译报错
- 修复模式：增加条件编译将__bf16定义为bfloat16_t别名
- 可审查性：高
- 审查规则建议：kernel代码使用硬件内建类型(__bf16等)时，检查是否在__CCE_KT_TEST__宏下提供CPU侧兼容定义

### 7b7999e9e003bd15617781e8a1980633a73b5664 [dx] fix K_M_EXT_STEP error
- 根因类别：计算逻辑错误/M维度上界不正确
- 涉及文件：conv/conv3d_backprop_input_v2/op_kernel/arch35/convolution_3d_backprop/conv3d_bp_func.h
- 缺陷描述：TPL_NO_SPLIT_KERNEL模式下baseUseM_可能超过实际有效M维度，load3d的padList[3]=255导致实际M小于tiling给出的baseM，访问超范围数据
- 修复模式：根据实际H/W/dilation/stride/padding重新计算有效M值，当M < baseUseM_且mIter_==1时修正baseUseM_
- 可审查性：低
- 审查规则建议：tiling参数直接用于计算时，审查是否存在tiling值大于实际有效数据范围的情况，需与实际值做min

### 7aa778e0c8d17aab28f98dca8374df22fe4a9ab3 group场景精度问题
- 根因类别：分组卷积尾块计算逻辑错误
- 涉及文件：conv/conv3d_backprop_input_v2/op_kernel/arch35/conv3d_backprop_input_v2/conv3d_dx_rowc_block.h
- 缺陷描述：group convolution最后一个group的N维度与其他group不同(cin不整除group)，原代码统一使用cinG和nCnt_导致边界判断对最后group不正确，nGroupCoreTail_计算公式也有误(cin - group*cinG为负值)
- 修复模式：新增nTailCnt_表示最后group的N迭代次数，修正尾块判断条件和nGroupCoreTail_计算为tailN % singleShapeN_
- 可审查性：中
- 审查规则建议：group convolution中审查最后group的通道数是否单独处理(cin % group != 0的情况)

### 616b4699ee0054f0fc78a1e7894f81fc74b40637 aclnnInplaceIndexCopy大shape性能劣化修复
- 根因类别：条件判断遗漏/性能回退
- 涉及文件：index/scatter_update/op_host/op_api/aclnn_index_copy.cpp
- 缺陷描述：IndexCopy非连续场景无条件走ScatterUpdate路径，大shape时性能反而更差，缺少数据规模判断
- 修复模式：提取为独立函数，新增stride和数据量门控条件(stride*elemSize >= 128B 或 总数据量 < 512KB)
- 可审查性：中
- 审查规则建议：引入新算子分派路径时，审查启用条件是否考虑数据规模维度，性能优化路径需有回退机制

### 58f275c6dc75bbcf8ce1f879832ea62422d69df2 修复dynamicQuant 310p精度问题
- 根因类别：DMA参数溢出/硬件限制未处理
- 涉及文件：quant/dynamic_quant/op_kernel/dynamic_quant.cpp, dynamic_quant_unalign_large_310p.h 等
- 缺陷描述：310P芯片DMA搬运参数为uint16_t，stride值可能超过UINT16_MAX导致截断，数据搬运错误引发精度问题。scaleLocal在多处反复DeQue存在queue管理问题
- 修复模式：增加stride溢出检测，超限时降为逐行搬运；scaleLocal的DeQue提前到外层循环通过参数传递
- 可审查性：中
- 审查规则建议：DMA搬运指令的stride/nBurst等参数检查是否可能超过硬件限制(uint16_t上限65535)

### 46f5817ef1679e1ce818d6ef04e6f98f036b5fd5 [dx] fix precision error
- 根因类别：资源管理错误/全载模式状态未正确切换
- 涉及文件：conv/conv3d_backprop_input_v2/op_kernel/arch35/conv3d_backprop_input_v2/conv3d_dx_rowc_block.h
- 缺陷描述：B矩阵偏移地址变化时未切换全载状态(enableFullLoad)，之前全载加载的L1数据已失效但仍被使用，导致精度问题
- 修复模式：新增preOffsetB_和preEnableFullLoad跟踪上一轮状态，偏移变化时释放L1并关闭全载
- 可审查性：低
- 审查规则建议：L1/L0缓存全载优化在多batch/多group循环中，检查全载条件是否因参数变化而失效

### 3e45851a13dbd55b95aea1028d67c532b56ff976 fix aclnnScatter支持0维tensor
- 根因类别：边界条件遗漏/0维tensor处理缺失
- 涉及文件：index/scatter_elements_v2/op_host/op_api/aclnn_scatter.cpp
- 缺陷描述：CheckTensorDim未处理0维tensor(标量)，GetDimNum()返回0与1维不匹配，标量scatter被拒绝
- 修复模式：dimSize为0时视作1维处理(赋值为1)再做一致性校验
- 可审查性：高
- 审查规则建议：输入校验中检查是否正确处理0维tensor(标量)，维度数用于比较/计算时0维是常见边界条件

### fa1305c56ef80179035f8ec7284875eb3c6126d2 修复aclnnKthValue在索引超过int32时的aicore问题
- 根因类别：整数类型溢出/索引类型选择不当
- 涉及文件：index/gather_v2/op_api/aclnn_kthvalue.cpp
- 缺陷描述：排序索引硬编码为DT_INT32，大shape下最后一维超过INT32_MAX时索引溢出
- 修复模式：新增GetSortIndicesType函数根据shape动态选择int32/int64
- 可审查性：高
- 审查规则建议：索引/offset类型为int32时，审查索引值是否可能超过int32范围，大shape下需动态选择int32/int64

### e7048cff4f29a467a6f285e72e497e6cd5ac1701 asw-kernel-fix
- 根因类别：计算逻辑错误/Bias加载时机错误
- 涉及文件：matmul/mat_mul_v3/op_kernel/arch35/mat_mul_asw_kernel.h
- 缺陷描述：splitK模式下Bias错误地在最后一次K迭代加载，应在第一次(kIndex==0)加载
- 修复模式：Bias加载条件从kIndex == splitKRound-1改为kIndex == 0
- 可审查性：高
- 审查规则建议：splitK场景Bias/残差加载审查是否在正确迭代轮次，Bias应在首次累加时加入(kIndex==0)

### e49d5d27c0877efd4b7707a2fd491b4870fd776e fix dx nullptr
- 根因类别：空指针解引用
- 涉及文件：conv/convolution_backward/op_api/aclnn_convolution_backward.cpp
- 缺陷描述：只需计算dw不需dx时，gradInput为nullptr，但代码用gradInput的shape计算mmDwOutShape2d并调用ViewWithShape导致crash
- 修复模式：改用gradWeight的shape计算mmDwOutShape2d
- 可审查性：高
- 审查规则建议：可能为nullptr的tensor指针在解引用前需null检查，反向传播中dx/dw/dbias可能为nullptr

### e25e6e60c70c9f218b190e497dc065dfd4682522 fix modulate buffer bug
- 根因类别：UB内存分配计算错误（tiling与kernel的buffer计算不一致）
- 涉及文件：vfusion/modulate/op_host/modulate_regbase_tiling.cpp, vfusion/modulate/op_kernel/arch35/modulate_regbase_common.h, vfusion/modulate/tests/ut/op_host/test_modulate_tiling.cpp
- 缺陷描述：tiling侧用`ubSize_ / DOUBLE_BUFFER / bufferCount`公式估算buffer大小，但kernel侧实际对x和y各占2份buffer(double buffer)，scale/shift又有不同策略，导致tiling计算出的buffer大小与kernel实际分配不匹配，UB溢出或数据覆盖
- 修复模式：统一tiling侧公式为`ubSize_ / (X_Y_BUFFER_NUM * DOUBLE_BUFFER + 1 + hasScaleShift)`，kernel侧scale/shift统一NO_DOUBLE_BUFFER，消除两端不一致
- 可审查性：中
- 审查规则建议：tiling侧buffer计算必须与kernel侧InitBuffer做交叉比对，确认buffer数量和double buffer策略一致；建议用共享常量定义buffer数量

### c00e940bfab0b85cf9c0c5f21c4faa377afd269e fix package size
- 根因类别：配置遗漏（新算子未注册到二进制打包配置）
- 涉及文件：scripts/kernel/binary_config/ascendc_config.json
- 缺陷描述：新增HardSwishGradV2算子未在ascendc_config.json中注册，导致kernel二进制不被打包
- 修复模式：在ascendc_config.json中添加HardSwishGradV2及其compute_units配置
- 可审查性：高
- 审查规则建议：新增算子时checklist需确认已更新打包配置；CI可自动校验新增算子目录是否在config中有对应条目

### b14f5a03cac76f0bb6e25d4b64e1ab30ca4f17df fix inplace_add_rms_norm kernelmode
- 根因类别：接口签名不匹配（缺少模板参数和函数参数）
- 涉及文件：norm/inplace_add_rms_norm/op_kernel/inplace_add_rms_norm.cpp, classify_rule.yaml
- 缺陷描述：GENERAL_OP_IMPL宏调用op.Init()缺少workspace参数；模板类只传1个类型参数但实际需要第二个kernel mode参数
- 修复模式：为Init添加workspace参数；为所有GENERAL_OP_IMPL模板类添加第二个参数`1`；更新classify_rule.yaml
- 可审查性：高
- 审查规则建议：修改kernel类Init签名或模板参数后，全局搜索所有调用点确认同步更新

### 2d424df8e624fd50b2911fd3f09a596343a43db9 fix repeatinterleave infershape
- 根因类别：控制流缺陷（设置状态后缺少return）
- 涉及文件：index/repeat_interleave/op_host/repeat_interleave_infershape.cpp
- 缺陷描述：两处SetUnknownRank后没有return语句，代码继续执行后续shape推导逻辑，对UnknownRank的shape调用GetDimNum可能返回异常值导致越界
- 修复模式：在两处SetUnknownRank后各添加`return ge::GRAPH_SUCCESS;`
- 可审查性：高
- 审查规则建议：设置特殊状态(UnknownRank/UnknownShape)的分支必须确认包含return；InferShape函数应做静态分析确保所有路径正确返回

### 245e9e45efcc611a6a3622265f333b80a1bdbb6e 修复EmbeddingHashTableApplyAdamW精度性能问题
- 根因类别：哈希表查找逻辑缺陷 + 类型精度问题 + 代码重复
- 涉及文件：hash/embedding_hash_table_apply_adam_w/op_kernel/arch35/ 多文件
- 缺陷描述：(1)线性探测循环内嵌套了embedding维度计算循环，导致未匹配key时也执行计算，且查找与grad计算的循环嵌套关系不正确 (2)b16和b32维护两个几乎相同的实现文件 (3)b32的PostProcess硬编码float类型转换
- 修复模式：删除重复b32版本，统一为模板实现；重写哈希表查找逻辑：先探测找key，found后再批量计算embedding维度
- 可审查性：低
- 审查规则建议：(1)哈希表probe逻辑和data处理应明确分离不交叉嵌套 (2)同一算法多类型版本应用模板替代代码复制 (3)kernel中间计算统一提升到float

### cbd1092fab563aa500c0a883ba6f8cb669032d41 fix hwnc tilingkey
- 根因类别：tilingkey定义不完整（宏参数缺失）
- 涉及文件：conv/conv2d_v2/op_kernel/arch35/conv2d_v2_tilingkey.h
- 缺陷描述：HWNC格式的tiling key宏定义缺少DisContinuous参数选择项，与其他格式的同类宏参数维度不一致，导致tiling key匹配错误
- 修复模式：在宏末尾追加DisContinuous参数选择项（默认CONV_DIS_CONTINUOUS_CLOSE）
- 可审查性：中
- 审查规则建议：存在多个变体宏（按format区分）时，逐一比对各变体参数列表是否完整一致

### 9c46dc5f1914fa2b01895682c679b9b342224bd0 addrmsnorm 95问题修复
- 根因类别：算子输出参数约束错误（REQUIRED vs OPTIONAL）
- 涉及文件：norm/add_rms_norm/op_host/add_rms_norm_def.cpp, norm/add_rms_norm/op_host/config/ascend910_95/add_rms_norm_binary.json
- 缺陷描述：910_95平台配置中rstd和x输出被标记为REQUIRED，但实际应为OPTIONAL，调用方不需要这些输出时框架强制校验导致失败
- 修复模式：def.cpp和binary.json中共8处paramType从REQUIRED改为OPTIONAL
- 可审查性：中
- 审查规则建议：新增平台配置时检查output的paramType是否与设计文档一致；def.cpp与binary.json的paramType声明必须双向一致

### 15138bbd86c161144966aafc17cc3457cac1f157 fix aclnnScatterAdd 注释
- 跳过：纯注释修改，仅更新头文件中API注释的数据类型支持列表描述

### db31f7273006cdde19bce89f2c4c0fc49d244002 修复TransQuantParamV2 310p上二进制增幅异常
- 根因类别：平台约束遗漏（编译配置缺少目标平台）
- 涉及文件：scripts/kernel/binary_config/ascendc_config.json
- 缺陷描述：TransQuantParamV2算子的compute_units列表缺少ascend310p，该平台二进制不被预编译导致运行时JIT编译引发异常
- 修复模式：在compute_units数组中添加"ascend310p"
- 可审查性：高
- 审查规则建议：对照平台支持矩阵逐一确认compute_units覆盖所有目标平台

### 952313ace0bca58196fd11d67f59dafa7dd1ee4f 修正kStartPt溢出翻转问题
- 根因类别：整数溢出（硬件指令参数位宽限制）
- 涉及文件：conv/conv3d_backprop_filter_v2/op_host/op_tiling/arch35/conv3d_backprop_filter_v2_basic_block_tiling.cpp/.h
- 缺陷描述：load3d指令kStartPt字段上限65535(16位)，当k0*hk*wk超过65536时发生溢出翻转，导致指向错误内存位置
- 修复模式：在checkLargeSpecs()中新增LOAD3D_KSTART_MAX=65535常量，当load3dK>65536时走超大kernel切分路径
- 可审查性：低
- 审查规则建议：涉及硬件指令参数时确认所有字段位宽限制，tiling代码中对这些字段的取值范围做边界校验

### 4f87bea9ab075b07008587aadf3a09041c4fd273 FusedCrossEntropyLossWithMaxSum算子回退
- 根因类别：平台约束遗漏 / 平台适配错误
- 涉及文件：loss/fused_cross_entropy_loss_with_max_sum/ 配置文件(删除910_95), activation/heaviside/, index/scatter_list/, norm/deep_norm/ 等多文件
- 缺陷描述：混合提交：(1)FusedCrossEntropyLossWithMaxSum在910_95平台有问题需回退 (2)多个算子平台版本判断过于严格，如`__CCE_AICORE__ == 220`应为`>= 220`
- 修复模式：回退问题算子的910_95配置；将精确匹配改为范围匹配或增加新平台枚举
- 可审查性：中
- 审查规则建议：(1)`__CCE_AICORE__ == xxx`精确匹配审查是否应用范围比较 (2)新增平台应有checklist检查所有相关算子

### 38bd1c5380231b94bcb301962808e9771a0a2432 aic问题修复
- 根因类别：边界条件缺失（数值越界未保护）
- 涉及文件：conv/conv3d_backprop_input_v2/op_kernel/arch35/convolution_3d_backprop/impl/dav_v310/conv_bp_sub_func.h
- 缺陷描述：CalcCutInWIndexForOnlyH函数中headWi_可能超过baseUseM_，后续使用headWi_的逻辑产生AIC错误
- 修复模式：headWi_计算后增加上界保护`if (headWi_ > baseUseM_) headWi_ = baseUseM_`
- 可审查性：中
- 审查规则建议：取模/取余计算的中间变量审查值域是否在后续使用的合法范围内；应在变量赋值处就做约束

### 30b91291450210a172c043704370cdd801f84406 fix mse_loss tilingdata bug
- 根因类别：数据结构定义不一致（自定义结构体与标准结构体字段布局不匹配）
- 涉及文件：loss/mse_loss/op_host/arch35/ 多文件, loss/mse_loss/op_kernel/arch35/mse_loss_tiling_def.h
- 缺陷描述：自定义ReduceOpTilingDataV2与标准ReduceOpTilingData字段布局不一致，手工逐字段拷贝导致kernel侧解析tiling数据出错
- 修复模式：删除自定义结构体和转换函数，直接使用标准ReduceOpTilingData
- 可审查性：高
- 审查规则建议：禁止手动镜像标准库数据结构定义，直接引用原始头文件；tiling数据结构host和kernel必须用同一份定义

### e71db0907d151071d3099fc8526e2df5bfe9d815 fix aclnnMedian
- 根因类别：边界条件缺失（空Tensor输入/输出场景混淆）
- 涉及文件：index/gather_v2/op_api/aclnn_median.cpp
- 缺陷描述：self->IsEmpty()和valuesOut->IsEmpty()用`||`合并处理，但输入空和输出空语义不同：输出空应直接返回不做任何操作
- 修复模式：拆分为两个独立分支，先判断输出空直接返回，再判断输入空走特殊处理
- 可审查性：高
- 审查规则建议：多条件用`||`合并时审查各条件是否确实需要相同处理；空Tensor的输入/输出应分别考虑

### e41444931b663806b8968942b424f52d64b1bd93 修复tilingkey计算错误的问题
- 根因类别：计算逻辑错误（非连续场景缺少条件守卫）
- 涉及文件：index/gather_elements/op_host/arch35/gather_elements_no_contiguous_tiling.cpp
- 缺陷描述：CoalesceGatherElements()被无条件调用，但index转置场景下维度压缩不适用，导致tiling key计算错误
- 修复模式：调用前增加`if (!isIndexTranspose_)`条件判断
- 可审查性：高
- 审查规则建议：算子存在多种执行路径时审查每个步骤是否对所有模式适用；bool模式标志应检查关键路径是否遗漏判断

### e2f856d387301b5daff7d382ca74e7971bef2e88 fix example & add aclnnadvancestepv2 atk
- 根因类别：示例代码错误（shape/data不匹配、头文件名大小写错误）
- 涉及文件：optim/advance_step/docs/aclnnAdvanceStepV2.md, optim/advance_step/examples/, optim/advance_step/tests/st/
- 缺陷描述：文档示例代码多处问题：头文件名V2大小写错、输入shape与API不匹配、data和shape张冠李戴、参数值不正确
- 修复模式：重新整理输入tensor的shape和data定义使其与API一一对应；新增独立示例和atk测试
- 可审查性：高
- 审查规则建议：示例代码必须通过实际编译运行验证；检查CreateAclTensor的data、shape、deviceAddr是否匹配

### 51f8247aee7db5da0f5697f09a909604ffb20d09 inplace_index_add_确定性问题fix
- 根因类别：流水线同步错误（硬件事件依赖类型错误）
- 涉及文件：index/inplace_index_add/op_kernel/arch35/inplace_index_add_determinstic.h, index/scatter_nd_add/op_kernel/arch35/scatter_nd_add_determinstic.h
- 缺陷描述：CopyOut前使用S_MTE2同步但CopyOut走MTE3通道，应使用S_MTE3和V_MTE3事件，原代码同步错误通道导致CopyOut可能在计算未完成时执行
- 修复模式：将S_MTE2替换为S_MTE3+V_MTE3双重同步；同步点移到Muls之后CopyOut之前
- 可审查性：低
- 审查规则建议：CopyOut/CopyIn前的同步事件必须与实际DMA通道匹配(MTE2搬入/MTE3搬出)；Vector计算后紧跟CopyOut需等待V_MTE3

### 0e5504b36d0806da206efdba14aecbaeb45c6ebd fix multi_add_rms_norm_dynamic_quant binary & cmake
- 根因类别：混合缺陷（构建配置错误 + 无符号整数比较 + 缺少return）
- 涉及文件：norm/multi_add_rms_norm_dynamic_quant/ 多文件(op_host/CMakeLists.txt, simplified_key.ini, infershape.cpp, tiling.cpp等)
- 缺陷描述：(1)ACLNNTYPE aclnn应为aclnn_exclude (2)section名全大写下划线与算子驼峰注册名不匹配 (3)size_t相减判断<0永远不成立(无符号下溢) (4)OP_CHECK_IF失败分支写了ge::GRAPH_FAILED但缺少return
- 修复模式：分别修复4个独立问题
- 可审查性：中
- 审查规则建议：size_t变量间做减法比较需警惕无符号下溢；OP_CHECK_IF失败分支确认是否需要return；simplified_key.ini的section名必须与算子注册名一致

### 0d47b59fdd5e766e52568a438869dba0448a94cf fix dynamic_quant_update_scatter_v2 aclnn
- 根因类别：构建配置错误（不应生成aclnn接口）
- 涉及文件：quant/dynamic_quant_update_scatter_v2/op_host/CMakeLists.txt
- 缺陷描述：ACLNNTYPE设为aclnn但该算子不应生成aclnn接口，导致编译失败或接口冲突
- 修复模式：ACLNNTYPE aclnn改为aclnn_exclude
- 可审查性：中
- 审查规则建议：新算子CMakeLists.txt的ACLNNTYPE应与设计文档中接口暴露策略一致

### 0d322070406a886cfb22218e615c0a9a80acfb72 SwishGrad精度修复
- 根因类别：数据结构定义不一致 + tiling key计算错误
- 涉及文件：activation/swish_grad/op_host/arch35/swish_grad_tiling_arch35.cpp, swish_grad_tiling_arch35.h, op_kernel/arch35/swish_grad_tilingdata.h
- 缺陷描述：SwishGradTilingData结构体中value字段冗余，scale字段类型应为float而非int64_t导致精度丢失；tiling key计算使用了未正确初始化的局部变量schMode，应从tiling->baseTiling.scheMode获取并转uint64_t
- 修复模式：修正TilingData结构体字段（删除冗余、修正类型），修正tiling key数据来源
- 可审查性：高
- 审查规则建议：TilingData结构体变更需与kernel端消费代码交叉审查确保字段布局和类型一致；tiling key计算应直接引用tiling结构体值而非外部变量

### c28b94f7bcb9b4a9febe030bec0dc1d2fee53e6a fix aclnnMedian empty
- 根因类别：边界条件处理缺失（空tensor）
- 涉及文件：index/gather_v2/op_api/aclnn_median.cpp, aclnn_median.h
- 缺陷描述：aclnnMedian在输入为空tensor时仅将workspaceSize设0返回，未对输出tensor填充值。对标PyTorch：整型应返回类型最小值，浮点应返回NaN
- 修复模式：新增DealMedianEmptyTensor函数按数据类型分别填充边界值（INT64填MIN_INT64，INT32填MIN_INT32，浮点填NAN）
- 可审查性：中
- 审查规则建议：所有aclnn算子必须显式处理空tensor输入且行为应与竞品对齐；新增算子review checklist应包含"空tensor行为"

### a3f3447463ef6556bfc6d7baa0c1946698f4b04a add leaky relu api check
- 根因类别：输入校验缺失（输出dtype兼容性未检查）
- 涉及文件：activation/leaky_relu/op_api/aclnn_leaky_relu.cpp
- 缺陷描述：CheckDtypeValid只检查输入self的dtype是否在支持列表内，未检查输出out的dtype能否由self类型转换而来，不兼容时静默产生错误结果
- 修复模式：扩展校验函数增加out参数，补充OP_CHECK_RESULT_DTYPE_CAST_FAILED校验
- 可审查性：高
- 审查规则建议：所有aclnn算子dtype校验必须同时检查输入和输出tensor的类型兼容性

### 6beab61dd0ee9c84809e45882dab39c835c35a4e fix change (flat_quant)
- 根因类别：混合缺陷（属性默认值错误 + 流水线同步缺失 + 宏隐藏控制流）
- 涉及文件：quant/flat_quant/op_graph/flat_quant_proto.h, op_host/flat_quant_def.cpp, op_kernel/arch35/flat_quant_high_v2.h, op_kernel/flat_quant_apt.cpp
- 缺陷描述：(1)dst_dtype默认值DT_INT32(3)应为DT_INT4(29)与算子实际输出类型矛盾 (2)两次matmul操作间缺少PipeBarrier<PIPE_ALL>()同步，matmulR结果可能尚未完成matmulL就读取 (3)INVOKE_FLAT_QUANT_IMPL宏中有return语句导致TILING_KEY_IS(5)分支提前退出，与其他分支不一致
- 修复模式：修正属性默认值 + 补充流水线同步 + 内联展开消除宏中隐藏控制流
- 可审查性：中
- 审查规则建议：OP_REG属性默认值须与OUTPUT数据类型定义兼容；连续matmul操作间必须插入PipeBarrier；禁止在宏中使用return语句

### 6657ff815962ceb355bee58290d5bf2407bfaa9c Fix slice but shape not valid
- 根因类别：条件判断不完整 + fallback路径逻辑缺失
- 涉及文件：matmul/common/op_host/op_api/matmul_util.cpp, matmul_util.h, matmul/mat_mul_v3/op_host/op_api/aclnn_matmul.cpp
- 缺陷描述：IsSliceNonContiguous缺少NZ格式不支持判断和左右矩阵dtype必须相同的约束，导致不满足条件的tensor错误进入slice优化路径。此外slice不支持的3D tensor应先转连续再fold batch维度重算m/n/k，原代码缺少needFoldBatch的fallback逻辑
- 修复模式：将分散的前置校验集中到判断函数 + 新增needFoldBatch降级路径
- 可审查性：中
- 审查规则建议："是否支持某优化路径"的判断应封装在统一入口而非散布多处；matmul shape变更后必须重新刷新m/n/k信息

### 175e2720e5903d59d85fb1bd990134369e0f186b 解决qbmm空tensor处理的问题单
- 根因类别：边界条件缺失（多通路行为不一致）
- 涉及文件：aclnn_quant_matmul_v4.cpp, aclnn_quant_matmul_v5.cpp
- 缺陷描述：qbmm算子静态图通路中m/n=0能正确返回空tensor，但aclnn直调和torch动态图通路缺少对m=0/n=0的early return，导致被拦截报错。多通路行为不一致
- 修复模式：在函数前部增加m=0/n=0边界检查，根据NZ/ND格式分别提前return
- 可审查性：高
- 审查规则建议：多通路调用同一算子时，各入口对边界输入(空tensor/零维度)的处理必须一致

### 048a67c900053103d5f1daf7b93899438a5db221 conv soc fix
- 根因类别：硬编码SoC标识/平台抽象不足
- 涉及文件：conv模块下约60个文件(conv_base.h/cpp, conv_api_tiling_util.h, conv_api_tiling_base.cpp等)
- 缺陷描述：conv算子全面依赖SocVersion枚举(ASCEND910_95等)做平台判断，新SoC加入时无法扩展；UB切分vector/cube比例硬编码为常量VEC_NUM_PER_CUBE_910D=2
- 修复模式：SocVersion替换为NpuArch架构级枚举；硬编码参数改为运行时platformInfo查询(aivPerAic等)
- 可审查性：中
- 审查规则建议：禁止在tiling代码中硬编码具体芯片型号，统一通过NpuArch或平台API获取；硬件参数不允许使用magic number

### fb3b8365b964132fdd467ecc63724fb8996cc083 ut compile fix
- 根因类别：构建配置缺失（include路径遗漏）
- 涉及文件：cmake/ut.cmake
- 缺陷描述：op_api的UT编译缺少OPAPI_INCLUDE路径导致头文件找不到编译失败
- 修复模式：补全CMake include path
- 可审查性：高
- 审查规则建议：新增op_api模块时CI应跑完整UT编译验证；CMake审查checklist包含头文件路径是否同步

### e32a5eaae3d3988f184d92b1be14db26669a535d fix allocated oversize of FusedLinearCrossEntropyLossGrad
- 根因类别：内存分配过大（workspace计算公式错误）
- 涉及文件：fused_linear_cross_entropy_loss_grad_tiling.cpp
- 缺陷描述：MemFriendlyTilingFunc计算userWorkspaceByteSize时对BT*H和V*H两项错误乘了BUFFER_NUM(双buffer系数)，这两个buffer实际只需单份空间，多分配了(BT*H+V*H)*IN_BYTE_SIZE的量
- 修复模式：去掉两行的BUFFER_NUM乘子
- 可审查性：高
- 审查规则建议：workspace计算中每项乘以BUFFER_NUM时应逐项注释说明为什么需要双buffer；代码审查时逐行核对公式各项物理含义

### d1832c87c936cbf2bf352c7ef8e73e3e3c6d3858 fix log for check func
- 根因类别：复制粘贴错误（日志变量引用错误）
- 涉及文件：fused_quant_matmul_checker.cpp
- 缺陷描述：CheckDimValue中检查scale维度但报错时误用offsetShape->GetStorageShape().GetDim(0)（offset的shape），应为scaleShape的dim(0)，导致日志误导调试
- 修复模式：修正报错信息中的变量名使其与判断条件一致
- 可审查性：高
- 审查规则建议：错误信息/日志引用的变量应与上文条件判断中的变量一致；检测"条件用A但报错引用B"的模式

### ba224cb4c335a8a8883f9bda56b6b31d8e863726 mxfp4 bugfix
- 根因类别：buffer对齐粒度错误
- 涉及文件：dynamic_mx_quant_tail_axis.h
- 缺陷描述：Init中outQueue_和outBuffer_的buffer大小对齐基准使用UBBlockSize_(32B)，但fp4数据需按vlForHalfNumber_(更大粒度)对齐，使用较小粒度导致buffer不够大，vector操作可能越界
- 修复模式：将对齐计算基准从UBBlockSize_改为vlForHalfNumber_
- 可审查性：中
- 审查规则建议：buffer InitBuffer对齐粒度须与数据类型的vector处理粒度匹配；sub-byte数据类型(fp4等)需特别审查对齐常量

### 9af284829b26035a8a78732aa7059b80bf5c870d fix 负载均衡性能裂化
- 根因类别：条件判断边界遗漏
- 涉及文件：adaptive_sliding_window_tiling.cpp
- 缺陷描述：balanceAfterFixp条件原仅判断kSize<1024启用负载均衡优化，但kSize==1024且nCore>=8时N轴不均衡占比已很小也应启用M轴优化，遗漏此边界case导致性能裂化
- 修复模式：条件扩展为kSize<1024 || (kSize==1024 && nCore>=8)
- 可审查性：中
- 审查规则建议：tiling中<阈值判断应同时考虑==时的行为；性能相关条件分支修改应附带benchmark对比

### 8233cc99456acc404b3f1bda42689d7fe209db0b bugfix: 清除unused变量
- 根因类别：代码冗余/编译告警 + UT逻辑缺陷
- 涉及文件：index/repeat_interleave/op_host/op_api/aclnn_repeat_interleave.cpp, tests/ut/
- 缺陷描述：IntToTensor中AllocScalar返回值赋给repeatsScalar但从未使用产生unused告警；UT中BF16测试无条件走test_run_invalid但部分芯片(910_95/910B~910E)实际支持BF16
- 修复模式：删除未使用变量；UT按芯片版本选择test_run/test_run_invalid
- 可审查性：高
- 审查规则建议：启用-Werror=unused-variable将告警升级为错误

### 793745bf14d1f586bf57ac91e41c06bf662e4d6e fix RepeatInterleave config json
- 根因类别：配置遗漏
- 涉及文件：scripts/kernel/binary_config/ascendc_config.json
- 缺陷描述：RepeatInterleave在ascend910_95的编译配置缺少compile_options(-mllvm -cce-aicore-dcci-before-kernel-end=false)，而同族的RepeatInterleaveGrad已有此配置
- 修复模式：补充JSON配置项
- 可审查性：低
- 审查规则建议：新增算子时强制对比同族算子(XXX与XXXGrad)的配置一致性；编写脚本检测同族算子间compile_options差异

### 597274964667176934a7dd2d2b1e075afd967e5c fix check bug
- 根因类别：复制粘贴错误（空指针检查对象错误）
- 涉及文件：conv/convolution_forward/op_host/op_api/aclnn_quant_convolution.cpp
- 缺陷描述：TransDataPreProcess中对weight执行TransData转换后，紧接着的CHECK_NULLPTR却检查input而非weight，是复制上一行检查代码后忘改变量名。weight返回nullptr将不被捕获导致后续空指针崩溃
- 修复模式：CHECK_NULLPTR(input,...)改为CHECK_NULLPTR(weight,...)
- 可审查性：高
- 审查规则建议：连续相似CHECK调用重点比对检查对象与赋值对象是否匹配

### 4ddbefef5897c5db30b388cbb21b7de34fc0c9be fix unstall package
- 根因类别：打包配置遗漏
- 涉及文件：scripts/package/ops_nn/ops_nn.xml
- 缺陷描述：卸载配置缺少python和python/site-packages目录声明导致卸载时未清理。此外新增行使用install_mode属性名但文件其余一致用install_mod（无e），拼写不一致可能是新bug
- 修复模式：补充XML目录条目（但属性名拼写不一致存疑）
- 可审查性：中
- 审查规则建议：对XML配置文件建立schema校验限定合法属性名；检测同文件内属性名拼写一致性

### 1fcd73b68091ed0b500af53cb81242f86f930a48 fix log
- 跳过-纯文档/注释：仅将日志字符串"y format is not support"改为"y format does not support"英文语法修正

### 1999ccf22ff67a7010a02fcb203ad655032e3187 qbmm int8_to_int32问题修复
- 根因类别：模板分支逻辑错误（多余代码路径）
- 涉及文件：matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_apt.cpp
- 缺陷描述：BIASMODE==TPL_EXCLUDE且DTYPE_SCALE为FLOAT/BF16分支中，存在TPL_KERNELTYPE==TPL_NO_VEC_EPILOGUE_WITH_MMAPI的if constexpr分支直接实例化MatMulASWKernel，在int8->int32场景下被错误匹配执行且模板类型参数不正确，而下方MxType分支中已有该KernelType的正确处理
- 修复模式：直接删除多余且错误的模板特化分支
- 可审查性：低
- 审查规则建议：if constexpr模板分支需注释说明适用条件和数据类型约束；新增KernelType分支时需更新分派矩阵并在CR中对照

### 13f10ef11478af52d09b9cb25a159ece6c7f59f7 回退超大shape
- 根因类别：边界条件缺失（超大shape未拦截）
- 涉及文件：conv/convolution_backward/op_api/aclnn_convolution_backward.cpp
- 缺陷描述：ConvBackpropInput在W=1,B=1情况下可转Matmul路径优化，但未限制shape大小。当(m*k+k*n+n*m)*typeSize超过L2 cache(128MB)时转Matmul会导致性能劣化或结果异常
- 修复模式：新增IsGreaterL2Cache函数，计算矩阵内存超128MB L2 cache时返回false阻止conv->matmul转换
- 可审查性：中
- 审查规则建议：conv/matmul路径转换的条件判断应包含shape×dtype size的上界校验；优化路径需考虑硬件资源(L2 cache)限制

### 90cd650db33769846904b07e515283758f8b82de aclnn_quant_matmul_v5 check bugfix
- 根因类别：维度索引取反 + scale shape计算遗漏
- 涉及文件：matmul/quant_batch_matmul_v4/op_host/op_api/aclnn_quant_matmul_v5.cpp
- 缺陷描述：InferGroupSize中两个bug：(1)MicroScaling路径下scaleSizeK未乘2，e8m0 scale shape是[m,k/2,2]取dim(1)得到k/2而非k (2)非MicroScaling路径transX1=true时应取倒数第二维但取了最后一维，transX1=false反之，维度索引条件写反
- 修复模式：MicroScaling路径scaleSizeK乘2；非MicroScaling路径交换transX1的true/false分支维度索引
- 可审查性：高
- 审查规则建议：transX1/transX2条件分支中的维度索引审查时需逐一核对：转置时K在倒数第二维，非转置时K在最后一维

### 3bd8c6a2738d339fa9e477628f664f091f926b5c 3ddx tiling hk hkwk bugfix
- 根因类别：边界条件遗漏（tiling模式组合未覆盖退化场景）
- 涉及文件：conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_inner_product_tiling.cpp, conv/conv3d_backprop_input_v2/op_kernel/arch35/convolution_3d_backprop/conv3d_bp_large_attribute_func.h, conv/conv3d_backprop_input_v2/op_kernel/arch35/convolution_3d_backprop/impl/dav_v310/conv_bp_sub_func_load_gm_to_l1a.h
- 缺陷描述：Conv3D反向(dx)算子在tiling模式为TILING_HK且dedx_w==1时未被正确处理。原代码多处条件判断只考虑TILING_HK_WK模式，遗漏了TILING_HK && dedx_w==1这一等效场景，导致L1缓冲区大小计算错误。kernel侧ComputeForWkLoop中realWoSize_计算结果可能<=0但未做保护返回；ComputeForTilingHk中前放大补零的跳过逻辑缺失；unlikely()宏括号位置错误只包裹了第一个条件
- 修复模式：tiling侧3处条件扩展为TILING_HK_WK || (TILING_HK && dedx_w==1)；增加realWoSize_<=0的提前返回；增加补零部分的continue逻辑；修正unlikely()括号位置
- 可审查性：低
- 审查规则建议：多种tiling模式存在时，审查每个条件分支是否覆盖所有模式组合下的维度退化场景（如宽度=1）

### 10cab125fa9d6709b0f7b753dda6bdf8b389f6fd 修复算子缺失json simpledKey配置问题
- 根因类别：配置遗漏（impl_mode缺失）
- 涉及文件：scripts/kernel/binary_config/ascendc_config.json
- 缺陷描述：ScatterAdd、ScatterNd、ScatterNdAdd三个算子的impl_mode字段为空字符串""，缺失"high_precision"配置，导致未使用高精度实现模式
- 修复模式：将impl_mode从""改为"high_precision"
- 可审查性：高
- 审查规则建议：scatter类算子配置时检查impl_mode是否需要配置为high_precision

### 0081270b594bade272e0839fceb4cf699da03593 fix maxpoolwithargmaxv3 bigkernel mulcore
- 根因类别：内存寻址方式错误 + 计算逻辑错误
- 涉及文件：pooling/max_pool_with_argmax_v3/op_kernel/arch35/max_pool_with_argmax_v3_big_kernel_mul_core.h
- 缺陷描述：两处bug：(1) Init中设置GlobalBuffer时使用workspace[VALUE_WORKSPACE_SIZE]下标取址，应改为指针加法workspace + VALUE_WORKSPACE_SIZE；(2) RealIndex中splitW==0分支对index做了CeilValue对齐后的alignBlockLen来计算行列坐标，但应直接用curkW做除法和取模，对齐操作引入错误的index映射
- 修复模式：改为指针加法；去掉alignBlockLen中间变量直接用curkW计算
- 可审查性：中
- 审查规则建议：kernel中GlobalBuffer地址计算应使用指针算术而非下标；index还原逻辑中不应错误引入对齐操作

### dec58d1b24e052b805c4af99fa25309109c63990 baddbmm问题
- 根因类别：复制粘贴错误（参数重复）
- 涉及文件：matmul/batch_mat_mul_v3/op_host/op_api/aclnn_addbmm.cpp, matmul/batch_mat_mul_v3/op_host/op_api/aclnn_baddbmm.cpp
- 缺陷描述：aclnnAddbmmGetWorkspaceSize中调用isAddBmmProcessEmptyTensor(batch1, batch1)，第二个参数应为batch2却写成batch1。同样错误存在于aclnnBaddbmmGetWorkspaceSize中。导致只检查batch1是否为空tensor而忽略batch2
- 修复模式：将两处第二个参数从batch1改为batch2
- 可审查性：高
- 审查规则建议：函数调用中有多个同类型参数(batch1/batch2、input/output)时检查是否存在复制粘贴导致的参数重复

### ac64e3d38bde3e1976eaa10cf2c025b2e801bd88 fix ascendc_config.json
- 根因类别：编译选项配置遗漏
- 涉及文件：scripts/kernel/binary_config/ascendc_config.json
- 缺陷描述：约20个算子(Relu/ReluV2/Sigmoid/Gelu/NLLLoss等)在ascend910_95平台缺少-mllvm -cce-aicore-dcci-before-kernel-end=false编译选项
- 修复模式：批量补充compile_options
- 可审查性：中
- 审查规则建议：ascend910_95平台算子需检查是否配置dcci相关编译选项

### 2dcfbd35aeb03cd5cb22bc97cf99fea0014a6fad 回退gather_v2算子docs中误删的内容
跳过：纯文档修改

### f70cd870918fb011bf0db1acaad8d7bf9b77c738 fix msdagrad SetScheduleMode
- 根因类别：调度模式配置遗漏
- 涉及文件：vfusion/multi_scale_deformable_attention_grad/op_host/multi_scale_deformable_attention_grad_tiling.cpp, vfusion/multi_scale_deformable_attn_function/op_host/multi_scale_deformable_attn_function_tiling.cpp
- 缺陷描述：MultiScaleDeformableAttentionGrad算子缺少SetScheduleMode(1)调用；MultiScaleDeformableAttnFunction在TilingKey==0分支下同样缺少
- 修复模式：在对应位置添加SetScheduleMode(1)
- 可审查性：中
- 审查规则建议：tiling函数中需要batch调度的多核算子应检查SetScheduleMode调用

### 84ab237ad39a9c734b519983e736e3a51b4452aa ScatterNdAddSimtTiling 修复int64精度问题
- 根因类别：数据类型约束遗漏
- 涉及文件：index/scatter_nd_add/op_host/arch35/scatter_nd_add_tiling_base.cpp, scatter_nd_add_tiling_base.h
- 缺陷描述：SelectTiling中判断是否进入排序模板的条件未排除int64数据类型，排序模板不支持int64精度，当updateDtype_为DT_INT64时进入排序路径导致精度错误
- 修复模式：排序条件中增加updateDtype_ != ge::DT_INT64过滤
- 可审查性：高
- 审查规则建议：选择计算模板/优化路径时检查是否对所有支持的数据类型兼容，特别是int64等特殊精度类型

### 5c796d90aab3e2a1c48896e0c0aedbbc0846d5c6 fix SetScheduleMode
- 根因类别：调度模式配置遗漏
- 涉及文件：quant/swi_glu_quant/op_host/swi_glu_quant_tiling.cpp
- 缺陷描述：SwiGluQuant的tiling函数缺少SetScheduleMode(1)调用
- 修复模式：定义BATCH_MODE=1常量，tiling函数入口调用context->SetScheduleMode(BATCH_MODE)
- 可审查性：高
- 审查规则建议：新增tiling函数时确认是否需要SetScheduleMode

### 5bb4b5c77d635e0f11685232786c88be9347d767 fix: a16f8 n=1 not supported
- 根因类别：边界条件处理不当（退化维度）
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling.cpp, weight_quant_batch_matmul_v2_tiling.h
- 缺陷描述：A16F8量化场景下antiquant_scale shape为(n,)，当n=1时被识别为PER_TENSOR并报错"A16F8 only support perchannel"，但n=1的perchannel语义上合法
- 修复模式：新增ConfigureReuseScenarios方法，A16F8场景PER_TENSOR自动转换为PER_CHANNEL复用逻辑；CheckTempLimit扩展允许PER_TENSOR通过
- 可审查性：中
- 审查规则建议：量化算子的quantType判定逻辑应检查n=1等退化维度下的边界处理

### 56bd8fbcc314c7edd69d245b252e42820d83fb16 门禁ut修复
- 根因类别：Python类型转换逻辑错误
- 涉及文件：scripts/util/parse_changed_files.py, scripts/util/parse_compile_changed_files.py
- 缺陷描述：is_experimental = bool(sys.argv[2])对字符串参数做类型转换，Python中bool("FALSE")返回True（非空字符串均为True），导致传入"FALSE"时is_experimental被错误设为True
- 修复模式：改为sys.argv[2] == 'TRUE'精确字符串比较
- 可审查性：高
- 审查规则建议：Python中对命令行字符串参数不应使用bool()转换，应用字符串精确比较

### 09e5a53e8dda8c1d1871d1fde6dcce13e0363467 fix ascendc_config.json
- 根因类别：编译选项配置遗漏
- 涉及文件：scripts/kernel/binary_config/ascendc_config.json
- 缺陷描述：约20个算子(ForeachNonFiniteCheckAndUnscale/AdamApplyOne/EluGrad等)在ascend910_95平台缺少dcci编译选项
- 修复模式：批量补充compile_options
- 可审查性：低
- 审查规则建议：新增算子到ascendc_config.json时检查目标平台compile_options是否完整

### d3a7b1956cb73744331a6bd2f918a059844eb52b relu_v2和layer_norm_v3算子代码格式化
跳过：纯代码格式化/文档修改

### d1382de350f6284edbba722f629a36ba221c502b DTS2026011320178：RmsNormGrad 修改日志错误信息
- 根因类别：日志消息错误（copy-paste）
- 涉及文件：norm/rms_norm_grad/op_host/rms_norm_grad_tiling.cpp
- 缺陷描述：CheckRstdShape中三处OP_LOGE日志将rstd与x的shape比较错误写成"dy first few dim"，实际检查的是x而非dy
- 修复模式：统一修正日志消息为"Input rstd shape invaild, shape is not equal x first few dim."
- 可审查性：高
- 审查规则建议：日志消息中引用的变量名/tensor名应与实际检查的对象一致

### aa95cab24383f060204b409a7b97f371ca872d1a 修复matmul目录下example内存泄漏问题
- 根因类别：资源泄漏（内存泄漏）
- 涉及文件：matmul/batch_mat_mul_v3/examples/test_aclnn_addbmm.cpp等共15个文件
- 缺陷描述：多个example中aclTensor和device内存在CHECK_RET失败提前return时不会执行到末尾的aclDestroyTensor/aclrtFree释放逻辑；Inplace调用复用同一workspaceAddr可能导致第一段分配被覆盖后无法释放
- 修复模式：使用std::unique_ptr搭配自定义deleter(aclDestroyTensor/aclrtFree)封装，RAII确保异常路径资源释放；Inplace操作引入独立workspace变量
- 可审查性：高
- 审查规则建议：ACL API代码中aclrtMalloc/aclCreateTensor分配的资源应使用RAII封装确保所有return路径正确释放

### 9150a7cca8b69b222d68e8a9f1f31845667f9b75 fix scatterElementsV2 支持double/bool
- 根因类别：数据类型支持遗漏
- 涉及文件：index/scatter_elements_v2/op_host/scatter_elements_v2_asc_tiling.cpp, scatter_elements_v2_asc_tiling.h, op_kernel/scatter_elements_v2_apt.cpp, config/ascend910_95/scatter_elements_v2_binary.json
- 缺陷描述：ScatterElementsV2在tiling层的dtype集合中缺少DT_DOUBLE和DT_BOOL，kernel层预编译条件未将BOOL归入int8同族/DOUBLE归入int64同族处理，binary配置缺少分发条目
- 修复模式：dtype集合添加DT_DOUBLE/DT_BOOL；kernel中按字节宽度归入对应分支；bool的REDUCTION_ADD添加cast到half；补充binary配置和UT
- 可审查性：中
- 审查规则建议：新增数据类型支持时需同步更新tiling层dtype校验、kernel层预编译条件、binary配置三处

### 80e3f64102391e5f57659cdf0d6895ad90c7c0cb fix schedulemodel
- 根因类别：调度模式配置位置错误
- 涉及文件：conv/conv3d_backprop_input_v2/op_host/op_tiling/arch32/conv3d_backprop_input_v2_base_tiling.cpp, pooling/adaptive_max_pool3d_grad/op_host/adaptive_max_pool3d_grad_normal_tiling.cpp, adaptive_max_pool3d_grad_tiling_base.cpp
- 缺陷描述：Conv3DBackpropInputV2的PostTiling中错误设置了SetScheduleMode(1)（该算子不需要）；AdaptiveMaxPool3DGrad将SetScheduleMode(1)放在PostTiling基类中但应在DoOpTiling的Normal分支
- 修复模式：从Conv3DBackpropInputV2移除；从AdaptiveMaxPool3DGrad基类移至Normal分支的DoOpTiling中
- 可审查性：中
- 审查规则建议：SetScheduleMode调用位置应确认在正确的算子和tiling分支中，避免copy-paste引入错误

### 601831d25c2c34ddbc2336720a3b40bd9848e949 layerNorm fix compile_options
- 根因类别：编译选项配置遗漏
- 涉及文件：scripts/kernel/binary_config/ascendc_config.json
- 缺陷描述：LayerNormV4缺少ascend910_95平台compile_options和mc62cm12a平台支持；LayerNormV3条目完全缺失
- 修复模式：补充LayerNormV4的compile_options和平台支持，新增LayerNormV3完整配置
- 可审查性：高
- 审查规则建议：算子迁移到新平台时检查ascendc_config.json中compile_options和平台支持

### a66a28aa164f88cbb044dbef04ef5be2fc920e65 fix cmake file for max_pool_grad_with_argmax
- 根因类别：CMake构建配置缺失
- 涉及文件：pooling/max_pool_grad_with_argmax/CMakeLists.txt, pooling/max_pool_grad_with_argmax/op_host/CMakeLists.txt(新建), pooling/max_pool_grad_with_argmax_common/CMakeLists.txt, pooling/max_pool_grad_with_argmax_common/op_host/CMakeLists.txt(新建)
- 缺陷描述：顶层CMakeLists.txt缺少COMPUTE_UNIT和TILING_DIR参数；op_host子目录缺少CMakeLists.txt
- 修复模式：添加SUPPORT_COMPUTE_UNIT和SUPPORT_TILING_DIR变量；新建op_host/CMakeLists.txt
- 可审查性：高
- 审查规则建议：新增算子目录的CMakeLists.txt应包含COMPUTE_UNIT和TILING_DIR参数，op_host需独立CMakeLists.txt

### 87d126b21d8011cf0d264cf1171b3c33e3f2d796 MaxPoolV3 opapi告警修复
- 根因类别：编译告警（返回值const冗余）+ 注释错误
- 涉及文件：pooling/max_pool_v3/op_api/aclnn_max_pool.cpp
- 缺陷描述：PoolingOutShape和CheckOutputShape返回类型static inline const int64_t / static const bool中const修饰值类型返回值冗余触发编译器告警；View5Das4D/View5Das3D注释中维度编号和操作名与实际不一致
- 修复模式：移除返回类型冗余const；修正注释维度描述
- 可审查性：高
- 审查规则建议：函数返回基础值类型时不应添加const修饰；注释中维度编号应与代码逻辑一致

### ec83d69ca84b00a55f8b442e04629ee2a98b1156 【MatMul】修复fusedmatmul带bias时性能问题
- 根因类别：Tiling参数计算遗漏(bias占用L1空间)
- 涉及文件：matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_asw_tiling.cpp, matmul/fused_mat_mul/tests/ut/op_host/test_fused_matmul_tiling.cpp
- 缺陷描述：CalL1Tiling计算stepKa/stepKb时未扣除bias table在L1中的占用空间(BIAS_TABLE_NUM*DATA_SIZE_FP32)，导致带bias时stepK过大，L1实际可用不足引起性能退化
- 修复模式：在CalL1Tiling后重新计算remainSizeForAL1BL1(有bias时减去bias table大小)，基于剩余空间重算stepKa/stepKb
- 可审查性：中
- 审查规则建议：Tiling计算涉及L1/UB空间分配时，检查是否遗漏了辅助数据结构(bias/scale/offset table)的空间占用

### dfe06e8bc1878cf540de03b90833503189fecfe3 解决BatchNormGrad ub空间不够问题
- 根因类别：UB空间计算公式错误(缺乘数+对齐缺失)
- 涉及文件：norm/batch_norm_grad_v3/op_host/batch_norm_grad_v3_base_tiling.cpp, batch_norm_grad_v3_base_tiling.h, batch_norm_grad_v3_splitload_crosscore_tiling.cpp, batch_norm_grad_v3_splitload_tiling.cpp
- 缺陷描述：CalcBubBlock中ubTensorNotRelateChannelBlock = tmpChannelNum * noRelateBlockNum缺少sizeof(float)乘数，计算出的UB占用偏小导致实际分配不足；同时tmpChannelNum未按dtype对齐
- 修复模式：补充sizeof(float)乘数；新增GetAlignValue方法按FLOAT_BLOCK_SIZE或HALF_BLOCK_SIZE对齐tmpChannelNum
- 可审查性：高
- 审查规则建议：UB/L1空间计算公式必须统一单位(字节vs元素)，所有维度参与空间计算前必须乘以sizeof(dtype)

### c625ae7a243ec57a5c008c445c2b39bd627c3e15 1952 perf fix
- 根因类别：AIC/AIV控制流不一致(AIV只执行一次Iterate)
- 涉及文件：conv/common/op_kernel/arch35/conv_common_func.h
- 缺陷描述：IterateAll中AIC分支在while循环中迭代，但AIV分支只调用一次Iterate(非循环)，导致AIV未正确迭代处理所有数据块
- 修复模式：将AIV的SetOutputGm提前到while循环前，统一AIC/AIV共用while(Iterate)循环，内部按ASCEND_IS_AIC_CONV/ASCEND_IS_AIV_CONV分支处理
- 可审查性：中
- 审查规则建议：AIC/AIV双核架构中，两个分支的循环结构和迭代次数必须对等审查

### aad25fdf620e5c3a270291f3a0feae3d99cdca6b 移除extendConvTranspose op def，消除编译报错
- 根因类别：废弃代码未清理导致编译错误
- 涉及文件：conv/extend_conv_transpose/op_host/extend_conv_transpose_def.cpp(删除)
- 缺陷描述：extendConvTranspose的op_def文件已不再使用，但其引用的类型或宏在新版本中变更导致编译报错
- 修复模式：删除整个废弃的op_def文件
- 可审查性：高
- 审查规则建议：算子废弃或重构时确保清理所有关联的def/registration文件

### 797589989afa9c31155a167e838c862e21e31fd7 修复aclnn错误码错误
- 根因类别：错误码语义使用错误
- 涉及文件：quant/trans_quant_param_v2/op_host/op_api/aclnn_trans_quant_param_v2.cpp
- 缺陷描述：CheckNotNull对用户传入参数的空指针检查返回ACLNN_ERR_INNER_NULLPTR(内部错误)，应返回ACLNN_ERR_PARAM_NULLPTR(参数错误)
- 修复模式：更正错误码为ACLNN_ERR_PARAM_NULLPTR
- 可审查性：高
- 审查规则建议：用户输入校验的错误码必须使用PARAM_*系列，INNER_*仅用于内部不可达状态

### 74cdae15d7d781c60ebb7bb80a1439e11c04013e Revert NLLLossGrad performence optimize
- 根因类别：性能优化引入功能回退(Revert)
- 涉及文件：loss/nll_loss_grad/op_host/arch35/nll_loss_grad_tiling_arch35.cpp, nll_loss_grad_tiling_arch35.h, loss/nll_loss_grad/op_kernel/arch35/nll_loss_grad.h, nll_loss_grad_4d.h(恢复合并), nll_loss_grad_base.h(恢复合并), nll_loss_grad_tiling_key.h(恢复合并), nll_loss_grad_apt.cpp
- 缺陷描述：NLLLossGrad的大规模性能优化重构(拆分为4d/base/tiling_key等多个文件)引入了问题，被完全回退到优化前版本
- 修复模式：全量Revert，恢复为单文件实现
- 可审查性：低(大规模重构难以审查)
- 审查规则建议：大规模算子重构应分步提交，每步可独立验证；性能优化PR必须附带完整精度对比测试

### 52bb1ef24a23d11d72216c775904cada9b42581d fix dynamicquant perf
- 根因类别：性能优化(VF计算重构)
- 涉及文件：quant/dynamic_quant/op_kernel/arch35/dynamic_quant_regbase_full_load.h
- 缺陷描述：原实现逐行调用ComputeVF，每行重复Scale/Y计算初始化开销大
- 修复模式：拆分为DataCopyInputVF/ComputeScaleVF/ComputeYVF，传入multiRow批量处理
- 可审查性：低
- 审查规则建议：[性能优化，非功能缺陷]

### 407b68e17356d7cb2d97b75e007e982362746aee [MMV3] 回退对齐场景下workspace申请逻辑，解决对齐场景内存膨胀
- 根因类别：workspace计算公式错误(对齐场景使用全局尺寸)
- 涉及文件：matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp, matmul_v3_base_tiling.h
- 缺陷描述：DETERMINISTIC_SPLIT_K模式下workspace用alignedM*alignedN全局对齐尺寸计算singleSize，对齐场景下(M/N被pad到很大值)导致workspace内存申请远超实际需要
- 修复模式：新增GetDeterministicSplitKWorkspaceSize方法，BASE模式用singleCoreM*singleCoreN(实际每核尺寸)，非BASE模式保持alignedM*alignedN
- 可审查性：高
- 审查规则建议：workspace计算中涉及对齐后的维度时，区分全局对齐尺寸和每核实际尺寸，避免对齐膨胀被乘以核数

### 20d81cfa23cd4cc713842b0cca0bc3089ad02070 CCEC指令适配1952后的临时方案回退
- 根因类别：平台临时方案未及时清理
- 涉及文件：conv/common/op_kernel/arch35/conv_instr_hw_mode_impl.h, conv_instr_impl.h, conv_instr_m_mode_impl.h
- 缺陷描述：为Ascend910_95(5102)平台添加的#ifdef __NPU_ARCH__==5102条件编译临时方案(使用raw intrinsics如img2colv2_cbuf_to_ca/load_cbuf_to_cb代替LoadData API)，在API适配完成后应清理
- 修复模式：删除所有#ifdef __NPU_ARCH__==5102分支，统一使用LoadData/Load3DBitModeParam API
- 可审查性：高
- 审查规则建议：#ifdef __NPU_ARCH__临时平台适配代码必须附带清理计划和跟踪issue

### e147ca470a58cb58b1b8eb615e14ae946304a890 Fix the warnings when compiling for AddLayerNormQuant op.
- 根因类别：编译告警(未使用变量+类型混用+返回值丢失)
- 涉及文件：norm/add_layer_norm_quant/op_host/add_layer_norm_quant_empty_tiling.cpp, add_layer_norm_quant_tiling.h
- 缺陷描述：多个问题：1)x2StorageShape/betaStorageShape/y1StorageShape声明后未使用；2)rows_/cols_/usedCoreNum_/rowsPerCore_/rowsLastCore_用int64_t但实际不会为负(uint64_t更合适)；3)CalcUsedCoreNums返回void但内部调用CalcuTilingData的返回值被丢弃；4)maxOutputSize为uint64_t但被检查<0(永远为假)
- 修复模式：移除未使用变量；int64_t→uint64_t；CalcUsedCoreNums改返回ge::graphStatus并传递CalcuTilingData返回值；移除无意义的负数检查
- 可审查性：高
- 审查规则建议：函数返回值必须被使用或显式忽略；无符号类型不应与负数比较；声明的变量必须使用

### cb0f64cbcb0ce5c8ad9795bab760413165123d23 修复CMakeList以及补充二进制json
- 根因类别：二进制配置json参数错误
- 涉及文件：pooling/max_pool_v2/op_host/config/ascend910_95/max_pool_v2_binary.json
- 缺陷描述：ksize和strides的index都是0(应为0,1,2递增)；shape固定为[1,4]不支持动态shape(应为-2)；padding的value为空数组(应为"SAME")
- 修复模式：修正index为1和2；shape改-2；padding value设"SAME"
- 可审查性：高
- 审查规则建议：binary.json中多个input的index必须连续递增；动态shape算子shape应为-2；attrs必须有有效默认值

### b4076b657853f08242092a3632c79f84b8cd3e5f Fix fp32 align
- 根因类别：对齐粒度硬编码错误(half vs实际输出类型)
- 涉及文件：matmul/common/cmct/epilogue/block_epilogue_cv.h, matmul/common/cmct/epilogue/block_epilogue_elementwise.h
- 缺陷描述：AlignBlock<half>硬编码half类型进行16元素对齐，当输出类型为fp32时应按8元素对齐(32B/4B)，对齐粒度错误导致计算越界或精度异常
- 修复模式：AlignBlock<half>改为AlignBlock<DataTypeOut>，根据实际输出数据类型决定对齐粒度
- 可审查性：高
- 审查规则建议：AlignBlock/CeilAlign的模板参数必须与实际操作的数据类型一致，不应硬编码特定类型

### a79b527fe83c4f55f18936c1a724fec61987ac4f BNGV3 sync bugfix
- 根因类别：流水线同步事件管理缺陷(条件性WaitFlag导致同步不完整)
- 涉及文件：norm/batch_norm_grad_v3/op_kernel/arch35/batch_norm_grad_v3_split_r1_regbase.h
- 缺陷描述：多处WaitFlag<MTE3_MTE2>被条件化(if ni>0/basicBlockIdx<BUFFER_NUM等)，导致某些迭代路径跳过必要的同步等待；CopyOutDx后只SetFlag不WaitFlag就释放tensor，可能导致DMA写出未完成就被覆盖
- 修复模式：移除所有条件性WaitFlag；CopyOutDx后的SetFlag<MTE3_MTE2>后立即无条件WaitFlag<MTE3_MTE2>
- 可审查性：中
- 审查规则建议：SetFlag和WaitFlag必须成对出现且无条件执行；不应在循环边界条件中跳过同步事件

### 90649ac29984878c587dc0c9d1bdf5ad859d9e49 MaxPoolGradWithArgmaxV3 oom fix
- 根因类别：Tiling参数未区分normal/tail核(OOM)
- 涉及文件：pooling/max_pool_grad_with_argmax_v3/op_host/arch35/max_pool_grad_with_argmax_v3_nchw_tiling_scalar.cpp, max_pool_grad_with_argmax_v3_nchw_tiling_scalar.h, max_pool_grad_with_argmax_v3_simt_tiling.cpp, max_pool_grad_with_argmax_v3_ksize_one_tiling.cpp, op_kernel/arch35/max_pool_grad_with_argmax_v3_nchw_scalar.h
- 缺陷描述：原CalcGradArgmaxInner对normal核和tail核使用相同的argmaxInner参数，tail核的highAxisTail可能远小于highAxisInner，但使用highAxisInner计算的buffer尺寸分配，导致tail核实际处理时内存溢出
- 修复模式：拆分CalcGradArgmaxInner为normal版和CalcGradArgmaxInnerTail，分别用highAxisInner和highAxisTail计算；新增SetNormalInner和SetTailInner设置outer/tail参数
- 可审查性：高
- 审查规则建议：多核tiling中normal核和tail核的buffer尺寸必须分别计算，tail核的数据量通常小于normal核

### 8e56b958aaf22a70e9b3179b97b6dd86fda250a8 fix opkernel ut issue when run with ophost ut
- 根因类别：CMake构建配置(UT符号链接缺失)
- 涉及文件：cmake/ut.cmake
- 缺陷描述：kernel UT单独执行正常，但与ophost UT一起编译时legacy_common_manager.cpp不会被编入tiling obj，导致kernel UT符号缺失链接失败
- 修复模式：CMake条件编译中增加legacy_common_manager_stub.cpp桩代码源文件，在UT_TEST_ALL或OP_HOST_UT时自动链入
- 可审查性：中
- 审查规则建议：UT的CMake配置应确保独立运行和联合运行两种模式下符号链接均完整

### 79d5ee94de98881825423b4783610516017dee3b fix nonzero bug is datatype error
- 根因类别：非标准数据类型别名(平台不可移植)
- 涉及文件：index/non_zero/op_kernel/arch35/non_zero_big_mask.h
- 缺陷描述：使用uint64类型(非标准别名)而非标准的uint64_t，在某些编译环境下未定义导致编译错误或行为异常
- 修复模式：static_cast<uint64>改为static_cast<uint64_t>
- 可审查性：高
- 审查规则建议：使用标准C++类型(uint64_t/int64_t)，不使用非标准别名(uint64/int64)

### 6ec2c7aec4f6ceef082f05c4b468093cd36fb5c2 fix max pool argmax v3 nhwc kernel
- 根因类别：复制粘贴错误(stride变量) + UpdateMask参数被循环修改
- 涉及文件：pooling/max_pool_with_argmax_v3/op_kernel/arch35/max_pool_with_argmax_v3_nhwc_small_c.h, max_pool_with_argmax_v3_nhwc_big_c.h
- 缺陷描述：(1)small_c中wStride = hStride_复制粘贴错误，应为wStride_，导致W方向步长取值错误影响第二个输出精度；(2)big_c中FillPadNegInf的topCount/downCount在UpdateMask调用中被修改(UpdateMask会递减count)，循环第二次迭代起使用错误的count值
- 修复模式：(1)改为this->wStride_；(2)引入topCountTmp/downCountTmp临时变量避免原始参数被UpdateMask修改
- 可审查性：高(1) / 中(2)
- 审查规则建议：检查结构体成员赋值中h*/w*是否成对正确引用；UpdateMask等会修改输入参数的API在循环中使用时必须用临时变量保护原始值

### 2a15cd4d 解决index整网用例aic问题
- 根因类别：全局变量误用（作用域错误）
- 涉及文件：index/index/op_host/arch35/index_tiling_no_continuous.cpp, index_tiling_no_continuous.h
- 缺陷描述：indexstrideList被定义为.cpp文件作用域的全局std::vector，多实例并发调用时共享导致AIC错误。修复将其移入IndexNonContinuousTiling类的private成员，每个tiling实例拥有独立副本
- 修复模式：全局变量 -> 类成员变量（作用域收窄）
- 可审查性：中
- 审查规则建议：禁止在op_host/*.cpp中使用非const全局容器声明，tiling相关状态必须封装在类成员中

### e1a2f11f group_norm_grad算子在超长R轴下内存访问越界
- 根因类别：整数溢出（uint32溢出导致UB分配不足）
- 涉及文件：norm/group_norm_grad/op_host/group_norm_grad_tiling_arch35.cpp
- 缺陷描述：(1)mode0xDyDxSize为uint32_t，CPerG_*HxW_超长时CeilAlign()*tTypeBytes_*UB_COPIES_3乘积溢出，内存分配不足导致越界；(2)mode1UbCapCNum_分母计算中CeilAlign()返回值乘tTypeBytes_*UB_COPIES_3也溢出。修复改为int64_t和显式(int64_t)cast
- 修复模式：类型提升（uint32_t -> int64_t）
- 可审查性：高
- 审查规则建议：tiling代码中shape维度相乘的表达式强制使用int64_t；CeilAlign/CeilDiv返回值参与二次乘法时检查结果类型

### e0ddd962 fix nonzero operator B32/B16/B8 output shape product out of int32 boundary
- 根因类别：整数溢出（int32乘法溢出）
- 涉及文件：index/non_zero/op_kernel/arch35/non_zero_big_mask.h
- 缺陷描述：gmOffset_计算中addUbSize.GetValue(8)*shapeDim_均为int32，B32/B16/B8模式下output shape乘积超int32边界。修复对两个操作数做static_cast<uint64>
- 修复模式：类型提升（显式static_cast<uint64>）
- 可审查性：高
- 审查规则建议：GetValue()返回值参与乘法且结果赋给uint64/int64时，要求操作数显式cast为64位

### 8192a77f fix max pool v3 small kernel tiling
- 根因类别：UB空间计算遗漏
- 涉及文件：pooling/max_pool_v3/op_host/arch35/max_pool_v3_small_kernel_tiling.cpp
- 缺陷描述：计算availableUb_时只减了UB_RESVERVED_SIZE，遗漏了紧邻上方刚计算的indiceUbSize_，导致可用UB被高估，数据分块过大溢出indices区域
- 修复模式：补全资源预算减项（availableUb_ = ubSize - UB_RESVERVED_SIZE - indiceUbSize_）
- 可审查性：中
- 审查规则建议：计算availableUb/可用空间时，检查同作用域内所有*UbSize_变量是否纳入减项

### 502f8ac9 Revert "DTS2025120868273 c04 with innerbatch"
- 根因类别：特性引入缺陷（tiling + kernel全面错误）
- 涉及文件：conv/common/op_host/..., conv/common/op_kernel/..., conv/conv2d_v2/... (9个文件)
- 缺陷描述：C04(small channel)+innerbatch特性整体回退。原始提交三类问题：(1)L1空间计算公式错误(align与innerBatch乘法顺序不对)导致buffer越界；(2)kernel mmad指令参数(m/srcStride/srcBatchStride)计算公式错误；(3)265行tilingkey宏定义与错误kernel逻辑配套。全量Revert
- 修复模式：全量Revert（特性设计有根本性问题）
- 可审查性：低
- 审查规则建议：同时修改tiling(host)和kernel(device)超过5个文件的大特性，应拆分子PR并附带数值验证用例

### 4432d437 Fix M-MTE1 SetWait when kernel contains TPipe
- 根因类别：硬件同步flag编号冲突
- 涉及文件：matmul/common/cmct/block/block_mmad_pingpong_without_que.h
- 缺陷描述：M_MTE1事件的SetFlag/WaitFlag使用ZERO_FLAG(0)/FIRST_FLAG(1)，与TPipe内部使用的低编号flag冲突，导致同步信号互相践踏。修复将flag偏移到SIXTH_FLAG(6)/SEVENTH_FLAG(7)，三处循环体都修复
- 修复模式：flag编号偏移到高位区间，避开TPipe占用
- 可审查性：中
- 审查规则建议：硬件同步flag应有全局编号分配策略，禁止不同模块隐式假设flag不冲突；SetFlag/WaitFlag变更需注释flag分配方案

### ffed8d28 bugfix swi_glu_quant && inplace_add_rms_norm
- 根因类别：预处理条件宏逻辑错误
- 涉及文件：norm/inplace_add_rms_norm/op_kernel/inplace_add_rms_norm.cpp, quant/swi_glu_quant/op_kernel/swi_glu_quant.cpp
- 缺陷描述：(1)inplace_add_rms_norm中4处#if条件取反遗漏：`__NPU_ARCH__==3003`应为`!=3003`，arch3003不支持bfloat16_t却编译了该路径；(2)swi_glu_quant中#if/#endif嵌套层级错误，外层#endif过早关闭导致#elif变成孤立分支。修复取反条件+合并为单层#if
- 修复模式：条件宏取反修正 + 消除错误嵌套
- 可审查性：高
- 审查规则建议：平台架构排除宏应封装为统一宏(如ARCH_SUPPORTS_BF16)避免手写取反遗漏；#if嵌套超2层须在#endif注释对应条件

### e499fdc8 fix groupnormgrad warning
- 根因类别：类型安全缺陷（有符号/无符号比较 + 整数截断 + 返回值未检查）
- 涉及文件：norm/group_norm_grad/op_host/group_norm_grad_empty_tiling_arch35.cpp
- 缺陷描述：(1)ubSize_(uint64_t)强转int64_t做比较，超INT64_MAX时变负导致逻辑反转；(2)maxRowsNumDG_声明为int，存uint64_t除法结果可能截断；(3)CalcuTilingData()返回值未检查。修复去掉多余cast、改int64_t、合并声明赋值并加错误日志
- 修复模式：类型修正 + 返回值检查
- 可审查性：高
- 审查规则建议：buffer size计算统一用uint64_t/int64_t禁止截断到int；graphStatus返回值必须检查

### d818b4d4 修复告警
- 根因类别：编译告警修复（含潜在逻辑bug）
- 涉及文件：index/masked_scatter/op_api/aclnn_masked_scatter.cpp, masked_scatter_tiling_arch35.cpp, quant_update_scatter_tiling_arch35.cpp, smooth_l1_loss_grad_v2_tiling.cpp
- 缺陷描述：5处告警修复。值得注意：aclnn_masked_scatter.cpp删除未使用的sourceStorageFormat变量，但原始代码条件判断中重复检查maskStorageFormat两次而非source+mask各一次，极可能是copy-paste逻辑bug（此commit仅消除变量告警未修复逻辑）
- 修复模式：删除未使用变量、void cast、%d->%u格式修正
- 可审查性：高
- 审查规则建议：删除"未使用变量"时须检查该变量是否原本应该被使用（可能掩盖copy-paste逻辑bug）

### 685d55f9 fix a bug is the int64 data type of nonzero
- 根因类别：复制粘贴参数错误
- 涉及文件：index/non_zero/op_kernel/arch35/non_zero_base.h, non_zero_full_load_base.h
- 缺陷描述：CopyInBefore中int8/int16/int32三个分支调用VfPerCoreNonZeroNum使用(loopCore,beforeNum,...)，但int64分支错误使用(loopNum,tailNum,...)，语义完全不同（前序core总数 vs 当前core循环数），导致int64下计算结果错误
- 修复模式：参数纠正，与其他分支保持一致
- 可审查性：高
- 审查规则建议：同函数内多个if/else分支调用同签名函数时，检查各分支实参语义一致性

### 4453d4c8 Fix MaxPool3DGrad
- 根因类别：数学公式错误（池化输出维度计算）
- 涉及文件：pooling/max_pool3d_grad_with_argmax/op_host/max_pool3d_grad_with_argmax_tiling_base.cpp
- 缺陷描述：ceil mode池化输出维度公式错误，分子少了+stride且CeilDiv外多了+1，等价于floor((x-1)/s)+2而非正确的ceil(x/s)。当分子为负时CeilDiv行为异常导致整数溢出。修复为标准公式(input+2*pad+stride-dilation*(kernel-1)-1)/stride
- 修复模式：公式纠正（标准池化输出维度公式）
- 可审查性：低（bug fix混在36文件3000+行重构中）
- 审查规则建议：池化输出维度计算应作为公共工具函数不应多处手写；大PR应拆分重构和bug fix

### 368607bd CTCLossV2Grad算子搬迁修正
- 根因类别：搬迁遗漏（类型不匹配 + 空函数体）
- 涉及文件：loss/ctc_loss_v2_grad/op_host/arch35/ctc_loss_v2_grad_tiling_arch35.cpp, .h
- 缺陷描述：(1)TilingParse注册旧类型CTCLossV2GradCompileInfo(字段coreNum)，实际tiling依赖新类型CTCLossV2GradForCompileInfo(字段totalCoreNum)，类型不匹配导致参数取值错误；(2)TilingPrepare函数体为空直接return SUCCESS，缺失获取平台core数的初始化逻辑，totalCoreNum始终为0
- 修复模式：类型统一 + 补全缺失逻辑
- 可审查性：高
- 审查规则建议：TilingParse<T>的模板参数T必须与TilingPrepare操作的CompileInfo类型一致；TilingPrepare为空函数体应标记告警

### 121c1683 convtbc aicore问题修复
- 根因类别：tiling模式选择缺陷
- 涉及文件：conv/conv2d_v2/op_host/op_tiling/arch35/conv2d_v2_base_tiling_fast_tiling.cpp
- 缺陷描述：GetC04TilingSplitMode()中conv1d(ConvTBC)场景下wo>128时不应进入M-split模式应走HW-split，旧代码缺少此条件判断导致大wo下选错tiling模式产生计算错误
- 修复模式：增加前置条件检查(IS_CONV1D_FLAG && wo>128时forceHWSplitModeFlag=true)
- 可审查性：中
- 审查规则建议：conv各变体(conv1d/conv2d/convtbc)共用tiling逻辑时，须标注各变体适用范围

### f0cf6d56 fix tiling error
- 根因类别：错误的校验逻辑（过早拒绝合法输入）
- 涉及文件：quant/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling_base.cpp
- 缺陷描述：DoOpTiling()中isPerformanceBranch()判断后调用CheckKernelUBUpBound()，该检查将合法tiling场景也拒绝，导致本应成功的算子返回GRAPH_FAILED。修复删除错误的检查调用
- 修复模式：移除错误校验
- 可审查性：高
- 审查规则建议：新增return GRAPH_FAILED校验点须附UT说明触发条件和原因

### a6a1aff4 flatquant编译问题优化
- 根因类别：运算符优先级不明确（编译告警）
- 涉及文件：quant/flat_quant/op_kernel/flat_quant_vec_one.h
- 缺陷描述：Nceil>>LOG2_16-1和Nceil>>LOG2_128-1中>>与-优先级关系不明确，虽然C++中-优先级高于>>所以运行时行为正确，但代码意图不明确触发-Wparentheses告警。修复添加显式括号
- 修复模式：添加括号明确运算优先级
- 可审查性：高
- 审查规则建议：移位与算术运算符混用须加括号；启用-Werror=parentheses

### 282382f9 fix ini
- 根因类别：CMake构建依赖缺失
- 涉及文件：cmake/gen_ops_info.cmake
- 缺陷描述：add_custom_command生成ops-info.ini时缺少DEPENDS opbuild_custom_gen_aclnn_all，导致并行构建时aclnn生成目标可能尚未完成就开始合并ini文件，产生构建竞态
- 修复模式：在两处add_custom_command中添加DEPENDS opbuild_custom_gen_aclnn_all
- 可审查性：中
- 审查规则建议：add_custom_command必须声明所有输入依赖，特别是跨模块的生成目标

### d44f47c2 fix conv warning
- 根因类别：编译告警（参数名遮蔽成员变量 + 未使用变量）
- 涉及文件：conv/conv2d_v2/op_host/op_tiling/arch35/conv2d_v2_api_tiling.cpp, .h, conv3d_v2/.../conv3d_v2_base_tiling_check_attrs.cpp
- 缺陷描述：(1)SetQuantScale(bool hasScale)参数名与成员变量this->hasScale同名，-Wshadow告警；(2)conv3d ParseGroupLegal()中fMapDesc获取后未使用
- 修复模式：参数重命名为hasScaleFlag；删除未使用变量
- 可审查性：高
- 审查规则建议：函数参数名不得与类成员变量同名；启用-Wshadow -Werror

### 7a917555 修复A5 qbmmv4相关warning
- 根因类别：编译告警（未使用参数/变量）
- 涉及文件：matmul/quant_batch_matmul_v3/.../quant_batch_matmul_v3_checker_base.h, quant_batch_matmul_v4/...多文件
- 缺陷描述：基类虚函数CheckShape的参数未标注/* unused */触发-Wunused-parameter；qbmmv4中offsetShape获取后未使用；CalL1TilingDepth4MmadS8S4的leftL1Size参数未使用
- 修复模式：参数加/* name */注释标注；删除未使用变量
- 可审查性：高
- 审查规则建议：虚函数基类中无操作的参数须用/* name */标注；启用-Wunused-parameter -Werror

### 73e48d80 修复matmulv3在特定shape下的精度问题
- 根因类别：边界条件遗漏（尾块处理不完整）
- 涉及文件：matmul/mat_mul_v3/op_kernel/mat_mul_deterministic_splitk_kernel.h
- 缺陷描述：ReduceKInUbNzL2cache中双循环遍历M和N分块时，(1)mCoreUse/nCoreUse未在每次循环开始时重置为singleCoreM/singleCoreN，导致前次循环的尾块值被后续使用；(2)actualN尾块条件仅检查orderNMFlag==true时的outIndex尾块，遗漏orderNMFlag==false时inIndex尾块对应N轴的情况
- 修复模式：循环体内重置mCoreUse=tiling.singleCoreM/nCoreUse=tiling.singleCoreN；扩展条件为(orderNMFlag && outIndex==outCnt-1) || (!orderNMFlag && inIndex==inCnt-1)
- 可审查性：高
- 审查规则建议：双循环中涉及尾块的变量必须在每次迭代开始时重置；遍历顺序flag改变时须同步更新所有依赖该flag的条件分支

### 6287806e fix compilation alarms
- 根因类别：编译告警（隐式类型转换 + 浮点比较不精确）
- 涉及文件：conv/conv2d_v2/op_host/op_tiling/arch35/conv2d_v2_api_tiling.cpp, conv2d_v2_base_tiling_basic_block.cpp
- 缺陷描述：(1)多处scaleSize计算中channelWiseCoeff(float)*整数乘积赋值给int64_t/uint64_t时隐式截断；(2)calCut1/calCut2为float但后续用作uint32_t循环变量，缺显式转换；(3)BasicBlockSortFWDimScores中浮点数直接用==比较，应改为abs(a-b)<epsilon；(4)minCost从float赋值给uint32_t隐式截断；(5)sqrt(cores)返回double赋值给float
- 修复模式：添加static_cast显式转换；引入epsilon做浮点近似比较
- 可审查性：高
- 审查规则建议：浮点数不得用==比较，须用epsilon；混合类型运算须显式cast；启用-Wconversion

### 6246f56c fix ut
- 根因类别：UT测试环境缺陷（缺少平台版本设置）
- 涉及文件：pooling/avg_pool3_d_grad/tests/ut/op_api/test_aclnn_avgpool2d_backward.cpp
- 缺陷描述：测试用例ascend310P_test_avgpool2dbackwardbackward_global_avg_pool_bf16中缺少SocVersionManager设置为ASCEND310P，导致平台信息为默认值，测试结果不可靠
- 修复模式：添加op::SocVersionManager versionManager(op::SocVersion::ASCEND310P)
- 可审查性：中
- 审查规则建议：指定特定平台的UT用例须在开头设置SocVersionManager

### 35f22362 SparseTensorDenseMatMul修复编译告警
- 根因类别：编译告警（printf格式符不匹配）
- 涉及文件：matmul/sparse_tensor_dense_mat_mul/op_host/op_tiling/sparse_tensor_dense_mat_mul_tiling_arch35.cpp
- 缺陷描述：OP_LOGE中对int64_t变量使用%lld格式符，在某些平台上int64_t是long而非long long，应使用%ld。约12处告警
- 修复模式：全部%lld改为%ld
- 可审查性：高
- 审查规则建议：int64_t使用PRId64宏或%ld，不要硬编码%lld；启用-Wformat

### 1e38bfac LayerNormV3 tiling duplicate registrations bug fixed
- 根因类别：tiling重复注册
- 涉及文件：norm/layer_norm_v4/op_host/layer_norm_v4_regbase_two_pass_perf_tiling.cpp, layer_norm_v4_regbase_two_pass_tiling.cpp, layer_norm_v4_tiling.h, layer_norm_v4_tiling_base.cpp
- 缺陷描述：LayerNormV4的tiling文件中同时注册了LayerNormV3和LayerNormV4的REGISTER_OPS_TILING_TEMPLATE/REGISTER_TILING_DATA_CLASS，以及V3专用的CompileInfo结构体和GetV3PlatformInfo函数。V3已有独立tiling实现，V4文件中的V3注册是冗余的，导致重复注册冲突
- 修复模式：删除V4文件中所有V3相关的注册、结构体定义和函数
- 可审查性：中
- 审查规则建议：每个算子版本的tiling注册应在各自文件中完成，不得在其他版本文件中交叉注册

### 1a267243 fix: l1 size constant inconsistent
- 根因类别：常量运算语义错误（乘vs除）
- 涉及文件：matmul/common/cmct/prologue/block_prologue_b_antiquant_scmc_nd_kn.h, block_prologue_b_antiquant_scmc_nd_nk_nz_kn.h
- 缺陷描述：weightF16L1DbOffset_计算中使用L1_SIZE * KB_ELEM<half>，但KB_ELEM<half>的语义是"每个half元素占的字节比例"（即sizeof(half)），这里应该是L1_SIZE / sizeof(half)得到half元素数量再减去weight空间。L1_SIZE*KB_ELEM结果远大于实际L1大小，导致UB_TO_L1目的地址越界AIC_ERROR
- 修复模式：L1_SIZE * KB_ELEM<half> → L1_SIZE / sizeof(half)
- 可审查性：高
- 审查规则建议：地址偏移计算中乘除运算须明确单位语义（字节vs元素数）；L1/UB/GM地址偏移须做边界检查

### f02308a2 qbmmv3/sparse4to2 sk sync bugfix
- 根因类别：AIC/AIV同步阈值错误（流水线事件Wait条件）
- 涉及文件：matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_bf16_basic.h, quant_batch_matmul_v3_pertoken_basic.h, matmul/sparse4to2quant_matmul/op_kernel/sparse4to2quant_matmul.h
- 缺陷描述：SplitK结束时AIC补齐之前跳过的WaitEvent，原始代码条件为loop_>2补ping、loop_>3补pong。但实际skipWait只跳过了第0次和第1次（ping/pong各一次），所以补齐条件应为loop_>0和loop_>1。错误阈值导致loop_<=2或loop_<=3时缺少WaitEvent，C2V数据可能尚未就绪就开始使用
- 修复模式：loop_>2→loop_>0，loop_>3→loop_>1，三处文件同步修复
- 可审查性：中
- 审查规则建议：ping-pong双buffer的WaitEvent/SetEvent必须成对出现，跳过的Wait在结束时必须补齐且阈值与跳过次数一致

### d0790662 fix dynamic block quant kernel bug
- 根因类别：复制粘贴错误（行/列变量名混用）
- 涉及文件：quant/dynamic_block_quant/op_kernel/arch35/dynamic_block_quant_b8_kernel.h, dynamic_block_quant_bf16_b8_kernel.h
- 缺陷描述：CopyOut中计算scaleGmOffset时使用rowIdx * rowBlockLoopNum_，但该偏移是沿列方向递增的，应为rowIdx * colBlockLoopNum_。错误导致scale数据写入GM的位置偏移不正确
- 修复模式：rowBlockLoopNum_ → colBlockLoopNum_，两个文件同步修复
- 可审查性：高
- 审查规则建议：行/列相关变量命名须清晰区分；GM偏移计算中的stride须与数据布局一致

### cd7e67d6 添加资料及修复代码告警
- 根因类别：编译告警（参数名遮蔽成员变量）
- 涉及文件：activation/sigmoid_grad/op_host/arch35/sigmoid_grad_tiling_arch35.cpp, activation/silu_grad/op_host/arch35/silu_grad_tiling.cpp
- 缺陷描述：GetComputeMap(uint64_t opKey)和ToString(TilingData &tilingData)中参数名与成员变量/类型名冲突，触发-Wshadow告警
- 修复模式：参数重命名为opKeyParam、tilingDataParam
- 可审查性：高
- 审查规则建议：函数参数名不得与成员变量或类型名相同

### b47cf616 LayerNormV4: Fix_lastBinaryBlock
- 根因类别：向量运算mask作用域错误 + ReduceSum精度缺陷
- 涉及文件：norm/layer_norm_v4/op_kernel/arch35/layer_norm_v4_two_pass_perf.h
- 缺陷描述：LAST_LOOP_NUMS==2分支中：(1)pregLast在if constexpr分支外定义但仅在特定分支使用，LAST_LOOP_NUMS==1时pregLast基于lastBinaryAddNum，而LAST_LOOP_NUMS==2时实际需要的mask是lastBinaryAddNum-VL_B32（仅覆盖第二块的有效元素）；(2)直接将两块Add后用pregLast做ReduceSum，但pregLast对应的是整体lastBinaryAddNum，第二块的超出部分包含了脏数据。修复改为：先用ShiftLefts将x2中有效元素左移对齐，再与x1做Add，最后用pregFull做ReduceSum
- 修复模式：pregLast移入分支内部重新定义为lastTailNum；引入ShiftLefts对齐第二块数据；ReduceSum改用pregFull
- 可审查性：低
- 审查规则建议：ReduceSum的mask必须精确覆盖有效数据范围；多块归约时须处理尾块对齐问题

### b139245e fix rms_norm infershape
- 根因类别：InferShape边界条件缺失（unknown rank未处理 + 输出shape未初始化 + 维度校验缺失）
- 涉及文件：norm/add_rms_norm/op_host/add_rms_norm_infershape.cpp, norm/rms_norm/op_host/rms_norm_infershape.cpp
- 缺陷描述：(1)未处理unknown rank(-2)输入，直接对GetDimNum()结果做循环，unknown rank时DimNum返回异常值；(2)rstdShape输出未提前获取和null检查；(3)缺少xDimNum < gammaDimNum的合法性校验，可能导致循环下标溢出
- 修复模式：添加IsUnknownRank()提前返回；提前GetOutputShape(1)并做null检查；添加维度数比较校验
- 可审查性：高
- 审查规则建议：InferShape必须处理unknown rank/unknown dim场景；所有输出shape须提前获取和null检查

### a0fd9959 fix_elu_grad_v2_tiling
- 根因类别：搬迁遗漏（CompileInfo类型不匹配 + 缺少tiling初始化）
- 涉及文件：activation/elu_grad/op_host/arch35/elu_grad_tiling_arch35.cpp/.h, activation/elu_grad_v2/op_host/arch35/elu_grad_v2_tiling_arch35.cpp/.h
- 缺陷描述：(1)EluGrad和EluGradV2使用通用ElewiseCompileInfo类型，但该类型字段与实际需要的coreNum/ubSize不匹配，需改用专用的EluGradCompileInfo/EluGradV2CompileInfo；(2)EluGradV2Tiling::RunTiling中缺少GetTilingData<EluGradV2TilingData>()初始化，tiling指针为null导致后续操作崩溃
- 修复模式：新建专用CompileInfo结构体；TilingParse/Tiling/TilingPrepare全部替换为专用类型；RunTiling开头添加tiling初始化
- 可审查性：高
- 审查规则建议：算子tiling的CompileInfo类型须与IMPL_OP_OPTILING注册的TilingParse<T>模板参数一致；RunTiling入口必须初始化tiling数据指针

### 99a2052c 修复部分算子编译过程报错的问题
- 根因类别：构建脚本glob匹配过宽
- 涉及文件：scripts/util/ascendc_impl_build.py
- 缺陷描述：get_ops_info_files()使用glob(aic-*-ops-info.ini)匹配所有SoC的ini文件，在多SoC构建环境中加载了不相关SoC的配置导致编译报错。修复改为按--soc参数按指定SoC精确匹配
- 修复模式：增加soc参数，按aic-{soc}*-ops-info.ini精确glob
- 可审查性：中
- 审查规则建议：构建脚本中的glob匹配应尽可能精确，避免通配符过宽匹配到不相关文件

### 9572f313 精度fail场景修改
- 根因类别：量化matmul多处计算逻辑错误（对齐/条件/偏移/模板分支）
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_host/op_api/aclnn_weight_quant_batch_matmul_v2.cpp, op_tiling/arch35/weight_quant_batch_matmul_v2_reg_base_tiling.cpp, op_tiling/weight_quant_batch_matmul_v2_tiling.cpp, op_kernel/arch35/weight_quant_batch_matmul_v2_reg_base_common.h
- 缺陷描述：(1)perGroup场景N轴对齐检查在910_95平台不应生效，缺少平台排除条件；(2)kBubSize双buffer切分未考虑groupSize对齐，直接CeilDiv(kBl1Size,2)导致切分点不在group边界上，反量化参数错位；(3)4bit权重NZ格式的DMA参数bubKLen未CeilAlign到BLOCK_CUBE导致搬运不完整；(4)weightOutStride公式错误dataBlockStride*(vfElemB16-BLOCK_CUBE)+BLOCK_CUBE应为dataBlockStride*vfElemB16-bubKLen*BLOCK_CUBE；(5)pergroup32/64的VF_CALL中innerExtend==1||outerExtend==1时使用false模板参数分支，实际该条件判断有误应统一走true分支；(6)整体删除了K/N维度32B对齐的过严校验
- 修复模式：多点修复，涉及计算公式、平台判断、DMA对齐、模板分支简化
- 可审查性：低
- 审查规则建议：量化matmul的buffer切分边界须与groupSize对齐；DMA参数中的blockLen须与实际数据类型对齐粒度一致

### 5c65b17d LayerNormV3: Fix_lastBinaryBlock
- 根因类别：向量运算mask作用域错误 + ReduceSum精度缺陷（同b47cf616的V3版本）
- 涉及文件：norm/layer_norm_v3/op_kernel/arch35/layer_norm_v3_two_pass_perf.h
- 缺陷描述：与b47cf616完全相同的缺陷在LayerNormV3中的对应实现。LAST_LOOP_NUMS==2分支中pregLast作用域和ReduceSum的mask不正确，导致尾块归约包含脏数据
- 修复模式：与b47cf616相同的修复手法
- 可审查性：低
- 审查规则建议：同一算法的不同版本(V3/V4)须同步修复同类缺陷

### 44162b1b fix ksize one tiling
- 根因类别：API命名空间迁移不完整
- 涉及文件：pooling/max_pool_grad_with_argmax_v3/op_host/arch35/max_pool_grad_with_argmax_v3_ksize_one_tiling.cpp
- 缺陷描述：KsizeOne分支tiling代码使用了旧命名空间API(`platform::GetVRegSize`/`ops::CeilDiv`)，导致编译失败或运行时行为不正确。同时tiling key从301改为900，说明分支路由也有错误
- 修复模式：`platform::GetVRegSize` → `Ops::Base::GetVRegSize`，`ops::CeilDiv` → `Ops::Base::CeilDiv` 等8处命名空间替换；UT中tiling key 301→900
- 可审查性：高
- 审查规则建议：API命名空间迁移时需全局搜索旧命名空间，确保所有分支（包括不常触发的ksize=1特殊路径）都已迁移

### 47e5dfc0 scatter_add、scatter_nd_add、scatter_elements、smooth_l1_loss_grad_v2修复问题
- 根因类别：算子注册架构搬迁不完整(graph_infer → infershape合并)
- 涉及文件：index/scatter_add/op_host/scatter_add_infershape.cpp, index/scatter_elements/CMakeLists.txt, loss/smooth_l1_loss_grad_v2/op_host/smooth_l1_loss_grad_v2_infershape.cpp 等13个文件
- 缺陷描述：InferDataType函数原先在独立的graph_infer.cpp中注册，需迁移到infershape.cpp中与InferShape合并注册(`.InferShape().InferDataType()` 链式调用)。scatter_elements的CMake还缺少对tf_scatter_add/scatter_nd_add的依赖声明
- 修复模式：删除独立的graph_infer.cpp，将InferDataType函数移入infershape.cpp，使用链式注册；补充CMake DEPENDENCIES
- 可审查性：中
- 审查规则建议：算子infershape/infertype注册架构变更时，需确认所有算子都已完成迁移，InferDataType不应存在于独立的graph_infer文件中

### 16275685 去掉chamferdistancegrad的pipeall同步
- 根因类别：流水线同步过粗(PipeBarrier<PIPE_ALL>滥用)
- 涉及文件：loss/chamfer_distance_grad/op_kernel/chamfer_distance_grad.h
- 缺陷描述：kernel中约20处使用`PipeBarrier<PIPE_ALL>`作为万能同步，这既是正确性风险（过度序列化可能掩盖实际数据依赖关系），也导致严重性能问题。正确的方法是根据实际的数据流依赖使用精确的HardEvent同步
- 修复模式：新增`PipeSync<HardEvent>`模板方法封装SetFlag/WaitFlag，用精确的事件类型替换所有PipeBarrier<PIPE_ALL>：V_MTE3(向量写完后DMA出)、MTE2_MTE3(搬入搬出依赖)、MTE2_V(搬入后计算)、V_S(计算后标量)、S_V(标量后计算)、MTE3_S(搬出后标量)等
- 可审查性：高
- 审查规则建议：PipeBarrier<PIPE_ALL>在kernel中应被视为代码异味，审查时应要求开发者证明无法用更精确的HardEvent替代

### f5606d69 修复空tensor提示bug
- 根因类别：CMake依赖声明缺失
- 涉及文件：matmul/mat_mul_v3/op_host/CMakeLists.txt
- 缺陷描述：mat_mul_v3缺少对batch_mat_mul_v3的CMake DEPENDENCIES声明，导致空tensor场景下符号解析失败
- 修复模式：CMakeLists.txt添加`DEPENDENCIES batch_mat_mul_v3`
- 可审查性：高
- 审查规则建议：算子模块间存在符号引用时必须在CMakeLists.txt中声明DEPENDENCIES

### 81d3a62e 修复qbmmv4 mxa8w4 ND精度问题
- 根因类别：多buffer策略下循环控制遗漏
- 涉及文件：common/act/prologue/block_prologue_b_cast_scsc.h
- 缺陷描述：ND场景下2-buffer模式时nL1Len可能不等于nUbSize，但VectorProcess直接调用ProcessL1，内部仅处理4-buffer的单次搬运。缺少对nL1Len>nUbSize时的多轮循环控制，导致weight数据只处理了第一个UB大小的部分，引起精度错误
- 修复模式：将原ProcessL1拆分为ProcessL1NK4Buffer(4-buffer快速路径)和ProcessL1NK(通用循环路径，含bUbNFactor×bUbKFactor双层循环)；VectorProcess根据l1BufNum_分发；offset变量从Aiv1专用(nL1Aiv1Offset_)重命名为通用(nL1Offset_)
- 可审查性：中
- 审查规则建议：多buffer模式切换时(2buffer/4buffer)，必须验证L1→UB搬运是否需要多轮循环；当L1数据量>UB容量时，必须有循环控制

### 4cd640c2 DynamicBlockQuant代码告警修复
- 根因类别：参数名遮蔽成员变量(-Wshadow)
- 涉及文件：quant/dynamic_block_quant/op_host/dynamic_block_quant_i8_tiling.cpp, .h
- 缺陷描述：RowTilingData构造函数参数`coreNum`与成员变量`coreNum`同名，构造函数内`this->coreNum = coreNum`虽然功能正确但触发-Wshadow告警。类似地DynamicBlockQuantI8构造函数参数`context`与成员`context`同名
- 修复模式：参数重命名：`coreNum` → `curCoreNum`，`context` → `ctx`
- 可审查性：高
- 审查规则建议：构造函数参数不应与成员变量同名，启用-Wshadow编译选项

### 21797e26 fix conv forward aclnn cmake
- 根因类别：CMake依赖声明缺失
- 涉及文件：conv/convolution_backward/CMakeLists.txt, conv/convolution_forward/op_host/CMakeLists.txt
- 缺陷描述：convolution_backward缺少对avg_pool3_d_grad/batch_mat_mul_v3/convolution_forward的依赖；conv相关源文件缺少include
- 修复模式：补充CMake DEPENDENCIES声明；添加缺失的include
- 可审查性：高
- 审查规则建议：新增算子模块间调用时必须同步更新CMake DEPENDENCIES

### 0c96cd51 修复ScatterNdAdd符号未定义的问题
- 根因类别：API迁移不完整 + 文件命名冲突
- 涉及文件：index/scatter_nd_add/op_host/arch35/scatter_nd_add_tiling.cpp, scatter_util→scatter_tiling_util重命名
- 缺陷描述：scatter_util.h/cpp文件名与其他模块冲突导致符号未定义，同时内部使用旧API `GetCompileInfoPtr<>` 需迁移为 `context->GetCompiledInfo<>()`，TilingPrepare4Scatter中大量TIK旧路径代码已废弃需清理
- 修复模式：文件重命名scatter_util→scatter_tiling_util；API替换GetCompileInfoPtr→GetCompiledInfo；移除TIK旧代码路径
- 可审查性：中
- 审查规则建议：工具函数文件命名应包含算子名前缀避免跨模块冲突；API迁移时同步清理废弃代码路径

### 0ac42f18 修复非连续
- 根因类别：条件判断遗漏
- 涉及文件：matmul/common/op_host/op_api/batch_matmul_util.cpp
- 缺陷描述：CheckTransNonContiguousShapeSupport函数检查L0a/L0b/L0c/L1容量约束时，计算了lessThanL1变量但在early return条件中遗漏了对它的检查。导致L1放不下的场景仍然走了非连续转置优化路径，产生错误结果
- 修复模式：在`if (!batchEqual || !batchLargerThanAicNum)`条件中追加`||!lessThanL1`
- 可审查性：高
- 审查规则建议：当函数计算了多个条件变量(lessThanL0a/L0b/L0c/L1)后，审查需确认每个变量都被后续逻辑使用；未使用的计算结果是强烈的遗漏信号

### 940066dd fix EnsureNotScalar
- 根因类别：头文件依赖断裂
- 涉及文件：activation/elu_grad/op_host/arch35/elu_grad_tiling_arch35.cpp
- 缺陷描述：elu_grad依赖tiling_base/tiling_util.h中的EnsureNotScalar函数和Ops::NN::OpTiling命名空间，但头文件路径或符号已不可用，导致编译失败
- 修复模式：移除对tiling_util.h的include和using namespace，在本地文件中重新实现EnsureNotScalar(标量→shape{1}转换)
- 可审查性：中
- 审查规则建议：公共工具函数被删除或移动时，需检查所有使用方是否已更新

### 503a451c fix aclnn bng bug
- 根因类别：空tensor的dtype未设置导致Cast失败
- 涉及文件：norm/batch_norm_grad_v3/op_host/op_api/aclnn_batch_norm_backward.cpp
- 缺陷描述：当saveMean/saveInvstd为空tensor时，Contiguous后tensor的dtype可能未正确设置，直接调用`l0op::Cast(tensor, DT_FLOAT)`失败。空tensor虽然没有实际数据，但Cast操作仍需要合法的源dtype
- 修复模式：在Cast前检查IsEmpty()，若为空tensor则用const_cast显式设置DT_FLOAT dtype再Cast
- 可审查性：中
- 审查规则建议：对tensor执行Cast/类型转换前，需考虑空tensor的dtype是否已正确初始化；空tensor不等于可以跳过类型检查

### 0e5dadce 修复告警信息
- 根因类别：隐式bool转换告警(unsigned int → bool)
- 涉及文件：matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/quant_batch_matmul_v3_iterbatch_tiling.cpp
- 缺陷描述：`!aicoreParams_.aicNum`对uint64使用逻辑非运算符，虽然语义正确(检查是否为0)但触发编译告警(implicit conversion from uint64 to bool)
- 修复模式：`!aicoreParams_.aicNum` → `aicoreParams_.aicNum == 0`，同理处理l1Size、l0cSize
- 可审查性：高
- 审查规则建议：对非bool类型变量禁止使用`!`运算符做零值检查，使用显式`== 0`比较

### dcb8f328 dynamic_quant 910c问题修复
- 根因类别：平台case遗漏
- 涉及文件：quant/dynamic_quant/op_host/dynamic_quant_tiling.cpp, CMakeLists.txt
- 缺陷描述：TilingForDynamicQuant的switch语句中缺少`case ASCEND910_93`(910C平台)，导致910C平台无法走入正确的tiling分支。同时CMake缺少对dynamic_quant_v2的依赖
- 修复模式：在ASCEND910B case前添加`case platform_ascendc::SocVersion::ASCEND910_93:`使其fall-through；补充CMake DEPENDENCIES
- 可审查性：高
- 审查规则建议：平台switch语句新增SoC版本时，须全局搜索所有SocVersion switch确认无遗漏；每个switch应有default分支处理未知平台

跳过(第160-179行)：7条
- 062c7b64: 回退cmake修改(构建配置revert)
- f72dda48: fix tiling ut noexec(UT基础设施/cmake)
- 9119a756: ExtendConv2d support leaky relu(新功能)
- 8c22fef1: 支持leaky_relu/swish算子A5实现(新功能)
- 71cd018e: fix link(纯文档链接修复)
- 50e3f514: 静态库脚本修复(Python 3.7兼容性)
- d92c383f: docs_aclnnBMM_fix(纯文档)

### cfe148f860bcacb29ca21ed256fa49cce56ab93d 修复C04的性能问题&更新flagInfo
- 根因类别：状态同步缺失(组合对象状态未传递)
- 涉及文件：conv/common/op_host/op_tiling/arch35/conv_base.cpp, conv_base.h, conv_base_utils.h, conv2d_v2_base_tiling_fast_tiling.cpp, conv3d_v2_base_tiling_fast_tiling.cpp
- 缺陷描述：ConvBase对象内部的flagInfo_成员在fast tiling流程中未被更新。Conv2d/Conv3d的tiling类持有各自的flagInfo_，但在调用convBase_的方法(如GetWeightBandWidthCoeff)之前没有将flag信息同步给convBase_。导致convBase_不知道C04模式已启用，GetWeightBandWidthCoeff()无法命中C04+NHWC分支，返回错误的带宽系数(BW_COEFF_UB=1而非BW_COEFF_C04=10)，造成性能劣化
- 修复模式：新增UpdateFlagInfo()方法在fast tiling入口同步flagInfo_ + 补充C04+NHWC格式的带宽系数分支
- 可审查性：中
- 审查规则建议：组合对象的状态同步检查——若父类持有子组件对象并调用其方法，需验证子组件所依赖的状态字段是否已从父类同步

### a4ea013a7083035071fee5d39cc9e3969a0172a7 fix al1 fulload perf problem&fusematmul problem
- 根因类别：(1)条件判断缺失(性能缺陷) (2)继承层级错误(功能缺陷)
- 涉及文件：matmul/fused_mat_mul/op_host/op_tiling/arch35/fused_matmul_asw_basic_tiling.cpp/.h, matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_aswt_tiling.cpp
- 缺陷描述：两个独立问题。(1) CheckAL1FullLoad()缺少对l0C2Out_模式的检查，当输出不是ON_THE_FLY模式时全载策略不适用但被错误启用，引发性能劣化。(2) FusedMatMulAswBasicApiTiling错误继承自MatMulV3BasicAswtTiling(应为MatMulV3AswTiling)，导致走了Basic级别tiling路径
- 修复模式：(1) CheckAL1FullLoad()入口增加模式前置条件过滤 (2) 更换基类 + 覆写GetTilingKey()
- 可审查性：中
- 审查规则建议：新增子类时应说明基类选择理由，特别是存在多个相似基类时

### 850c7eb03a354a1ce34418509e833e38e4fcb533 修复A L1全载的判断条件
- 根因类别：条件判断不完整(边界条件遗漏)
- 涉及文件：matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_basic_api_tiling.cpp
- 缺陷描述：IsAFullLoad()函数判断A矩阵是否可在L1中全载时，缺少对batch维度的约束。A L1全载仅在batch==1时适用，但原代码没有检查inputParams_.batchC==1。多batch时L1空间不足或复用逻辑不正确，导致错误启用全载
- 修复模式：在isAFullLoad_条件表达式末尾追加 && inputParams_.batchC == 1
- 可审查性：高
- 审查规则建议：全载策略启用条件必须包含batch维度检查，审查tiling策略启用条件时逐条对照M/N/K/Batch约束

### 741ae7a013526702b0bf98d36dd7cf986f4592ed 修改输出日志错误DTS2025121125810
- 根因类别：日志/字符串错误(缩进错误+拼写错误)
- 涉及文件：matmul/quant_batch_matmul_v3/op_api/quant_matmul_checker.cpp
- 缺陷描述：(1)错误信息字符串C++续行符\后下一行开头有多余缩进空格，被包含在最终日志中导致输出不自然。(2)"scenarion"拼写错误，正确为"scenario"
- 修复模式：修正字符串续行缩进 + 修正拼写错误
- 可审查性：高
- 审查规则建议：检查C++字符串续行(\)后的下一行是否有非预期前导空白；引入codespell等工具检查日志字符串拼写

### 12b41a6438acef94a717dce9c227da8b96e6ff44 公式化weight全载开启db错误
- 根因类别：分支逻辑错误(条件组合遗漏导致double buffer决策错误)
- 涉及文件：conv/common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_Mmode.cpp/.h
- 缺陷描述：InitABL1TilingMode()在fmap占主导且启用innerBatch(batch>1)时直接降级为NONE_FULL_LOAD，但正确逻辑应先尝试weight全载(FULL_LOAD_BL1)作为fallback。原if-else结构未覆盖此组合路径，丢失double buffer优化机会
- 修复模式：重构条件分支，拆分为4个语义辅助函数(IsAllFullLoadPossible等)，补充innerBatch时fallback到weight全载的路径
- 可审查性：低
- 审查规则建议：涉及多个互斥/组合条件的tiling决策逻辑应提供决策矩阵(truth table)；超过3层嵌套if-else应拆分为命名清晰的辅助函数

### 017f5007edd5cb64e6329c1f20412c8d737aa8ae ISA fix
- 根因类别：ISA接口变更/废弃接口调用
- 涉及文件：norm/batch_norm_grad_v3/op_kernel/arch35/ 下5个头文件
- 缺陷描述：三类问题。(1)使用已废弃的bcnt1()（popcount），应替换为ScalarGetCountOfValue<1>()。(2)调用已废弃的set_mov_pad_val(0)裸指令接口，需删除。(3) PipeBarrier<PIPE_ALL>();;多余分号
- 修复模式：替换bcnt1→ScalarGetCountOfValue<1>(4处) + 删除set_mov_pad_val(0) + 去除多余分号
- 可审查性：中
- 审查规则建议：建立废弃ISA函数清单(bcnt1、set_mov_pad_val等)，CI grep检查禁止使用；linter规则检测连续分号;;

### a592f6fa0921454b49f2ebc15f8f677d514059e7 bugfix for bmm util
- 根因类别：接口参数类型不匹配(bool→int64_t隐式转换导致语义错误)
- 涉及文件：matmul/batch_mat_mul_v3/op_host/op_api/aclnn_batch_matmul.cpp, matmul/common/op_host/op_api/batch_matmul_util.cpp
- 缺陷描述：TransBmm2Mm的第4个参数是bool enableHf32，但底层MatMulV3Nd期望int64_t opImplModeEnum(位标志枚举)。bool true隐式转换为1，而HF32模式对应0x40，完全不同。enableHf32=false时传入0x0，正确默认值应为0x1。导致矩阵乘法精度模式设置错误
- 修复模式：参数类型改为int64_t，调用方根据enableHf32/enableForceGrpAccForFp32计算正确枚举值(0x40/0x4/0x1)
- 可审查性：中
- 审查规则建议：启用-Wconversion编译警告检测bool→int隐式转换；使用enum class替代raw int避免隐式转换

### 626804ab7ca9ed650b9087ab4a8f349ff306e89f nullptr input error code modify
- 根因类别：错误码语义混用(INNER vs PARAM)
- 涉及文件：matmul/sparse4to2quant_matmul/op_host/op_api/aclnn_sparse4to2quant_matmul_weight_nz.cpp
- 缺陷描述：OP_CHECK_NULL(sparseWeight)检查用户传入参数空指针时使用ACLNN_ERR_INNER_NULLPTR(内部错误)，应为ACLNN_ERR_PARAM_NULLPTR(参数错误)。错误码会误导问题定位
- 修复模式：ACLNN_ERR_INNER_NULLPTR → ACLNN_ERR_PARAM_NULLPTR
- 可审查性：高
- 审查规则建议：函数入口参数校验区域内OP_CHECK_NULL必须搭配PARAM_NULLPTR；对内部指针搭配INNER_NULLPTR

### 3ac83c4ebfca1d81c7546139ffdcbdb8b9386bdc fix: 91095 op single compile error
- 根因类别：文件存在性检查缺失(脚本健壮性)
- 涉及文件：scripts/kernel/binary_script/build_binary_op_exe_task.sh, build_binary_op_exe_task_out.sh
- 缺陷描述：构建脚本在读取任务文件(opc_cmd_file/out_cmd_file)前未检查文件是否存在。单算子编译时可能不生成这些文件，sed/cat读取不存在的文件报错，set -o pipefail导致整个编译失败
- 修复模式：增加 if [ -f "${opc_cmd_file}" ] 前置判断
- 可审查性：高
- 审查规则建议：Shell脚本对变量路径文件操作(cat/sed/source)前必须做[ -f ]检查

### 2790615c7989ac8cb747d8a7299d119a835c9a4d 裸指令回退
- 根因类别：API语义混淆(block级ID vs subblock级ID)
- 涉及文件：conv/common/op_kernel/arch35/conv_opt_group_init_impl.h, conv/conv2d_v2/op_kernel/arch35/ 下3个impl头文件
- 缺陷描述：之前有人把get_subblockid()替换为GetBlockIdx()，两者语义完全不同：前者返回AI Core内部子块(subblock) ID用于区分向量执行单元，后者返回全局block ID用于区分不同AI Core。vecId字段应使用get_subblockid()获取子块ID。错误替换导致卷积计算任务分配和数据寻址错误
- 修复模式：4个文件中GetBlockIdx()全部回退为get_subblockid()
- 可审查性：中
- 审查规则建议：变量名含vec/subblock时赋值使用GetBlockIdx()应触发review警告；底层API批量替换需专项review

### 1d65758e162c8ccc60fa7cbe3ccffdbb4c020646 kernel大小校验过程防止溢出
- 根因类别：整数溢出(int32_t中间结果溢出)
- 涉及文件：conv/conv3d_backprop_filter_v2/op_host/op_tiling/arch35/conv3d_backprop_filter_v2_basic_block_tiling.cpp
- 缺陷描述：CheckKernelSize()中totalPadingD/H/W使用int32_t存储两个pad值之和，后续与di/hi/wi维度值相加计算kdMax等值时int32_t中间结果会溢出，导致kernel大小校验产生错误结论。旧代码还有不规范的static_cast<int>
- 修复模式：所有中间变量和结果类型从int32_t改为int64_t，第一个加数显式static_cast<int64_t>
- 可审查性：高
- 审查规则建议：多个int32_t值参与加法/乘法且结果仍存int32_t时标记溢出风险，特别关注shape/padding/dilation等用户输入参数的算术运算

### 1059e41e2df11bf745f2896e01448ce5b51072f0 Fix Nz n=1 tranpose
- 根因类别：边界条件缺失(输入校验遗漏)
- 涉及文件：matmul/mat_mul_v3/op_host/op_api/aclnn_matmul.cpp, matmul/mat_mul_v3/docs/aclnnMatmulWeightNz.md
- 缺陷描述：aclnnMatmulWeightNz在mat2为FRACTAL_NZ格式时未校验最后两根轴(k和n)是否为1。n=1或k=1时NZ格式内存布局退化，转置操作产生错误结果。旧代码仅文档声明"不保证精度"但未代码拦截
- 修复模式：CheckWeightNzShapeValid中新增前置校验，mat2Shape.GetDim(0)==1或GetDim(1)==1时返回false拒绝执行
- 可审查性：中
- 审查规则建议：算子声明不支持某些输入场景时必须在代码中有对应参数校验，仅文档说明不足以防止误用

### c866c4defe096d8c1f047553355a24e5368ac4c9 修复fake_quant_affine_cachemask精度问题
- 根因类别：计算逻辑错误(NaN/Inf特殊值处理不当)
- 涉及文件：quant/fake_quant_affine_cachemask/op_kernel/fake_quant_affine_cachemask_fp16.h, fake_quant_affine_cachemask_fp32.h
- 缺陷描述：NaN处理三个问题。(1)将NaN位置替换为quantMin是语义错误，fake quant期望NaN→0。(2)fp32版本第二步Compare比较yLocal而非xLocal，此时yLocal尚未正确赋值，读未初始化数据。(3)Inf检测冗余，Inf经Maxs/Mins clamp已被正确处理
- 修复模式：删除Inf检测冗余逻辑，NaN替换值从quantMin改为0.0f
- 可审查性：低
- 审查规则建议：NaN/Inf特殊浮点值处理必须有对应单测覆盖；对未初始化tensor的读取应作静态分析标记

### 9a1e4b840f2bc31f8c926ff8b4ac7beb6d623971 fix aclnn empty tensor bug
- 根因类别：空tensor边界条件处理不完整
- 涉及文件：norm/group_norm_grad/op_host/op_api/aclnn_group_norm_backward.cpp
- 缺陷描述：910_95平台处理empty tensor两个问题。(1)检查!gamma->IsEmpty()用的是原始gamma而非转换后的gammaContiguous，可能empty状态不一致。(2)缺少gamma本身为empty的处理分支，既不走empty分支也无法正常走后续reshape逻辑
- 修复模式：判断对象从gamma改为gammaContiguous + 新增else if分支处理gammaContiguous为empty时设workspaceSize=0
- 可审查性：中
- 审查规则建议：对tensor进行Contiguous/转换后，后续条件判断应使用处理后的tensor而非原始tensor；每个输入的empty组合都需确认处理路径

### 3570e96d90a0dc0f58e157ccabcf845739e316cc fix dequant_swiglu_quant tiling error
- 根因类别：常量误用(复制粘贴错误)
- 涉及文件：quant/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling_base.cpp
- 缺陷描述：CheckKernelUBUpBound()在dynamic模式(quantMode==1)下使用NUMBER_OF_INPUT_SIZE(10)作为buffer数量乘数，正确应为DYNAMIC_BF16_INT16_TBUF_NUM_HALF(6)。同文件另一处计算singleDataSize已正确使用该常量。高估UB内存需求(10 vs 6)导致合法tiling方案被错误判定为超限
- 修复模式：将NUMBER_OF_INPUT_SIZE替换为DYNAMIC_BF16_INT16_TBUF_NUM_HALF
- 可审查性：高
- 审查规则建议：同一文件中对同一计算场景应使用相同常量；常量仅使用一次而附近存在语义相似但值不同的常量被多次使用时标记为可疑

跳过(第180-199行)：5条
- 7e63c460: fix aicpu ut(UT构建脚本修复)
- 581bed62: aclnnMatmulWeightNz资料错误修改(纯文档)
- 2e99c5ad: fix DQUSV2 md(纯文档)
- 856183cd: fix ut case(纯UT测试)
- 884c0ce1: fix ut(纯UT测试)

### 342539b0 修复baddbmm在95平台的算子bug和910b平台的精度bug
- 根因类别：条件判断缺失 / 平台适配遗漏
- 涉及文件：matmul/batch_mat_mul_v3/op_host/op_api/aclnn_addbmm.cpp, aclnn_batch_matmul.cpp, batch_matmul_util.cpp
- 缺陷描述：enableFp16Bf16InFp32Out特性在910B/910_93平台对所有fp16/bf16输入都开启，但只应在Baddbmm接口调用时才使能fp32输出，普通BatchMatMul不应走此路径导致精度异常。同时fp16/bf16转fp32路由BatchMatMulV3NdFp16Bf162Fp32没有限制平台，在910_95等不支持平台上被错误命中导致算子执行失败
- 修复模式：将isBaddbmm标志透传到GetBatchMatmulOpInfo/CreateBatchMatmulOpInfo，路由分支增加SocVersion平台判断
- 可审查性：中
- 审查规则建议：平台特定逻辑需检查条件守卫是否足够，共享工具函数的新增特性开关是否在所有调用路径正确传递

### ab0ef822 fix (conv backward 1D-to-2D转换和白名单扩充)
- 根因类别：维度转换逻辑错误 + 边界条件缺失
- 涉及文件：conv/convolution_backward/op_api/aclnn_convolution_backward.cpp, convolutionbackward.cpp, conv3d_backprop_*_tiling_key.h
- 缺陷描述：PreConv1DBackwardTo2D中groups>1时conv1d到conv2d升维变换无条件用params.groups作为维度交换参数，但在dilation==45（magic number标识）时应使用exchangeDim=1跳过groups维度变换。OutputPostProcess中needSpecialCast缺少storageShapeDimSize>1检查
- 修复模式：增加条件分支控制维度交换逻辑，增加维度数量边界检查，扩充白名单
- 可审查性：低
- 审查规则建议：使用magic number(dilation==45)作为逻辑控制标志是代码异味，应用具名常量替代。数组索引操作应检查维度数量前置条件

### 7a1ea034 fix_expand_into_jagged_permute
- 根因类别：配置遗漏
- 涉及文件：scripts/kernel/binary_config/ascendc_config.json
- 缺陷描述：ExpandIntoJaggedPermute算子缺少AscendC二进制编译配置注册，在ascend910b和ascend910_93平台无法使用
- 修复模式：在ascendc_config.json中新增算子配置条目
- 可审查性：高
- 审查规则建议：新增算子应有checklist确保所有配置文件已更新

### 43503a16 conv 2d msg fix
- 根因类别：错误上报缺失
- 涉及文件：conv/conv2d_v2/op_host/op_tiling/arch35/conv2d_v2_tiling_utils.h
- 缺陷描述：tiling工具宏检测到不支持的参数时仅OP_LOGE_WITHOUT_REPORT打印日志但无REPORT_INNER_ERR_MSG上报错误码，上层调用方和错误监控系统无法感知内部错误
- 修复模式：在日志打印后增加REPORT_INNER_ERR_MSG调用，补充头文件引入
- 可审查性：高
- 审查规则建议：所有OP_LOGE_WITHOUT_REPORT应审查是否需要配套REPORT_*_ERR_MSG

### 3d7b9c0d Fix aFullLoad with bias
- 根因类别：计算公式使用错误维度变量
- 涉及文件：matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_aswt_tiling.cpp, .h
- 缺陷描述：MatMulV3 AFullLoad tiling中biasSize计算错误使用mAlignedValue/nAlignedValue，bias大小应基于N维度的实际L1 block大小min(CeilAlign(nValue,16), baseN*DB_SIZE)。导致L1内存占用估算不准确，tiling策略选择错误
- 修复模式：新增GetAFullLoadBasicNL1()方法封装正确的N方向L1计算逻辑
- 可审查性：中
- 审查规则建议：tiling内存计算公式中每个size变量引用的维度是否正确，bias与特定维度绑定需使用对应维度值

### fe6f8610 adaptive_max_pool3d_small_pool编译报错问题
- 根因类别：平台兼容性 / 类型硬编码
- 涉及文件：pooling/adaptive_max_pool3d/op_kernel/adaptive_max_pool3d.cpp, _big_pool.h, _small_pool.h
- 缺陷描述：mask类型硬编码为uint16_t，但310平台Compare指令mask输出为uint8_t，导致该平台编译失败
- 修复模式：引入条件编译using maskdType（310平台uint8_t/其他uint16_t），作为模板参数传递到所有相关类
- 可审查性：中
- 审查规则建议：NPU指令相关代码应检查是否存在平台相关的类型差异（如mask宽度），是否通过条件编译或模板参数适配

### f1bb8177 instance_norm异常处理修复
- 根因类别：错误处理不当 / 缺乏错误日志
- 涉及文件：norm/instance_norm_v3/op_host/op_api/aclnn_instance_norm.cpp
- 缺陷描述：参数校验函数使用CHECK_RET宏，校验失败时静默返回错误码无日志输出，用户无法定位失败原因
- 修复模式：将CHECK_RET替换为OP_CHECK配合OP_LOGE日志输出，每个校验失败点增加具体错误描述
- 可审查性：高
- 审查规则建议：对外API参数校验失败路径必须有OP_LOGE日志输出，禁止使用静默返回的CHECK_RET宏

### 3e9e2acd fix conv3d kernel bug
- 根因类别：状态初始化位置错误
- 涉及文件：conv/conv3d_v2/op_kernel/arch35/conv3d_v2_instr_impl.h
- 缺陷描述：aL1Pos=0初始化放在带参数的重载函数中而非无参数版本的入口。调用流程上无参数版本先计算偏移再调用带参数版本，但aL1Pos在带参数版本中才重置，前序计算依赖的aL1Pos可能保留上一轮迭代脏值
- 修复模式：将this->aL1Pos=0从两个带参数重载函数移到无参数版本LoadAl1Data()入口处
- 可审查性：高
- 审查规则建议：状态变量初始化应放在最早的调用入口而非被调用的子函数中。循环体内复用函数的状态重置时机是否正确

### ec448458 修复aclnnbaddbmm接口精度不通过问题
- 根因类别：数据类型处理缺陷 / 精度损失
- 涉及文件：batch_mat_mul_v3相关17个文件(op_host/op_api, stub, def, config)
- 缺陷描述：baddbmm接口fp16/bf16输入时中间计算以fp16/bf16输出导致精度损失。正确行为在A2/A3平台(910B/910_93)下应输出fp32。ifKEqual1分支也缺少fp32 cast。算子定义缺少fp16->fp32和bf16->fp32类型组合
- 修复模式：新增enableFp16Bf16InFp32Out标志位，新增BatchMatMulV3NdFp16Bf162Fp32函数走fp32输出路径，K=1分支增加Cast处理，算子定义增加类型映射
- 可审查性：低
- 审查规则建议：矩阵乘法算子应验证所有输入dtype×输出dtype组合是否完整覆盖，退化路径(K=1)是否保持主路径一致精度行为

### d5228e13 fix big shape run error for dequantSwigluQuant
- 根因类别：资源预分配校验缺失
- 涉及文件：quant/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling_base.cpp
- 缺陷描述：大shape输入(如96000000×3072)时tiling计算出的UB预分配大小超出硬件实际UB空间，代码无校验导致kernel运行时UB分配失败(561002)
- 修复模式：新增CheckKernelUBUpBound()方法，在tiling计算后按kernel实际UB分配逻辑逐项累加所需空间与ubSize比较，超限返回GRAPH_FAILED
- 可审查性：高
- 审查规则建议：tiling阶段必须对kernel所需的关键硬件资源(UB/L1/L0)进行上界校验

### 8a55ce05 fix conv matmul ophost ut
- 根因类别：构建/UT基础设施缺陷
- 涉及文件：cmake/func.cmake, tests/ut/common/CMakeLists.txt, legacy_common_manager_stub.cpp
- 缺陷描述：UT场景下LegacyCommonMgr通过当前so相对路径dlopen libophost_comm_legacy.so，但UT环境部署位置不同导致dlopen失败，conv/matmul的ophost UT编译链接失败
- 修复模式：cmake增加UT场景判断避免引入生产实现，新建stub打桩文件，补充头文件路径和链接依赖
- 可审查性：高
- 审查规则建议：引入新外部依赖(dlopen)时必须同步考虑UT场景的mock/stub方案

### 5c3d6119 fix quantize file name
- 根因类别：构建配置 / 文件命名错误
- 涉及文件：quant/ascend_quant_v2/op_host/CMakeLists.txt, quant/quantize/op_kernel/quantize.cpp→quantize_apt.cpp
- 缺陷描述：kernel源文件quantize.cpp不符合构建系统对apt后缀的命名约定，CMakeLists缺少DEPENDENCIES quantize声明
- 修复模式：重命名文件为quantize_apt.cpp，补充CMakeLists依赖声明
- 可审查性：高
- 审查规则建议：新增kernel文件名是否符合命名约定（_apt后缀），CMakeLists是否声明所有必要DEPENDENCIES

### 5401f5ec fix opsnn bug
- 根因类别：构建依赖缺失
- 涉及文件：norm/add_rms_norm_dynamic_quant, add_rms_norm_quant, group_norm_grad, layer_norm_grad_v3, rms_norm 各自的CMakeLists.txt
- 缺陷描述：5个norm算子的CMakeLists缺少DEPENDENCIES声明(norm_common, rms_norm等)，导致编译链接找不到公共函数
- 修复模式：在add_modules_sources调用末尾追加DEPENDENCIES参数
- 可审查性：高
- 审查规则建议：源码include了公共模块头文件时，CMakeLists是否已声明对应DEPENDENCIES

### 3d307c27 [bugfix] eliminate the risk of nullptr in qbmmv3 tiling
- 根因类别：空指针解引用风险
- 涉及文件：matmul/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling_base.cpp, .h, fused_quant_matmul_asw_tiling.cpp
- 缺陷描述：CheckFusionBatchA接收const gert::Shape&参数，调用方先对biasShape指针调用->GetStorageShape()再传入。当hasBias为false时指针为nullptr，解引用发生在传参时（函数调用前），函数内的hasBias条件守卫无法保护
- 修复模式：参数改为const gert::StorageShape*指针类型，解引用延迟到函数体内hasBias条件判断之后
- 可审查性：中
- 审查规则建议：传参表达式中对可能为null的指针做解引用，特别是指针有效性依赖于另一个布尔条件时，确保解引用在条件判断之后

### d014e1c3 Fix ACLNN interception for 2D EmbeddingBag cases
- 根因类别：平台适配缺失 / 多层缺陷
- 涉及文件：index/embedding_bag/下12个文件(op_api, op_host tiling, op_kernel)
- 缺陷描述：EmbeddingBag原只支持1D indices，910_95平台需支持2D indices。多处缺陷：(1)ACLNN维度校验对indices硬编码1维；(2)offset2bag shape用GetDim(0)而非GetShapeSize()，2D下只取第一维；(3)kernel空bag初始值用(T)(-1)应为(T)(0)；(4)tiling缺inclueLastOfst字段传递；(5)sum模式kernel循环用错indicesFactor应为weightRowFactor
- 修复模式：引入SocVersion::ASCEND910_95分支走不同校验和计算逻辑，tiling新增字段，kernel修正初始值和循环变量
- 可审查性：低
- 审查规则建议：新平台适配应有独立checklist明确各层修改点。GetDim(0) vs GetShapeSize()使用应有明确规范

### 1d6b119a scatterElementsV2 fp16/bf16 bugfix
- 根因类别：复制粘贴错误
- 涉及文件：index/scatter_elements_v2/op_kernel/scatter_elements_v2.h
- 缺陷描述：Cast(updatesTemp, updatesLocal, ..., indicesAlign)中第4参数应传updatesAlign而非indicesAlign。上行Cast用inputLocal配inputAlign，下行复制粘贴后未将indicesAlign改为updatesAlign，导致fp16/bf16时Cast长度参数错误
- 修复模式：将indicesAlign改为updatesAlign
- 可审查性：高
- 审查规则建议：Cast/Copy等批量操作的目标buffer与长度参数应有命名对应关系，名称不匹配是red flag

### d221fb68 bugfix for wqbmm 310p kernel config
- 根因类别：配置遗漏
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_host/config/ascend310p/weight_quant_batch_matmul_v2_binary.json
- 缺陷描述：310P平台kernel配置缺少dtype和inner_precise两个参数，导致kernel无法正确获取数据类型和精度控制参数
- 修复模式：在attrs数组补充dtype(默认-1)和inner_precise(默认0)参数项
- 可审查性：中
- 审查规则建议：新增kernel参数时检查所有平台配置文件是否同步更新，建立配置一致性校验脚本

### b946a8f5 修复图模式需要int64的场景
- 根因类别：数据类型支持不完整
- 涉及文件：conv/conv3d_backprop_input_v2/op_host/conv3d_backprop_input_v2_def.cpp
- 缺陷描述：Conv3dBackpropInputV2算子定义中input_size只声明INT32类型，图模式下框架可能传入INT64导致类型匹配失败
- 修复模式：将数据类型组合从5组扩展到10组，新增INT64和FLOAT类型支持
- 可审查性：高
- 审查规则建议：算子定义各输入输出DataType/Format列表长度必须一致。图模式和单算子模式的类型需求差异应在设计文档中明确

### 9624d01c fix perf downgrade
- 根因类别：性能回退 / 架构设计
- 涉及文件：matmul/weight_quant_batch_matmul_v2/下39个文件(tiling + kernel)，删约3216行新增约766行
- 缺陷描述：ACT代码中L0/L1 K列表包含1024配置导致tiling不优，MTE2配置多级分支引入开销，事件同步标识命名错误(MTE2_TO_MTE2应为MTE1_TO_MTE2)，A16W4 pergroup单独路径未合并到通用模板导致代码膨胀
- 修复模式：缩减K_LIST去掉1024，简化MTE2配置，修正事件标识，删除专用路径统一合并到ScmcKernel模板
- 可审查性：低
- 审查规则建议：性能相关改动必须附带benchmark数据。事件同步标识命名应与硬件事件语义一致

### 6650178c FlatQuant修复inf、nan场景精度不一致问题
- 根因类别：数值计算 / 矩阵维度错误
- 涉及文件：quant/flat_quant/op_host/flat_quant_tiling.cpp, op_kernel/flat_quant_cube.h, flat_quant_vec.h, tensor_utils.h
- 缺陷描述：MM_DOUBLE_MODE下多处维度计算错误：(1)workspace用4*Mceil^2应为Mceil^2；(2)P2分配用calM*calM但calM=2*Mceil导致4倍空间且越界；(3)缺calN/calFractalM/calFractalN导致fractal维度计算错误；(4)ProcessP1错误合并两块P1数据；(5)流水线事件x2p1ready未正确复用l0ready
- 修复模式：修正workspace公式，P2改Mceil*Mceil，新增calN/calFractalM/calFractalN字段，重写ProcessP1逻辑，修正流水线事件
- 可审查性：低
- 审查规则建议：double/倍增模式下所有维度变量需逐个确认是否需翻倍。workspace大小公式应有注释说明推导过程

跳过的提交(本轮行200-224):
- 8a6214d6: fix tiling ut for fusedquantmatmul(纯UT测试)
- d966e6ea: fix aclnn md(纯文档)
- 0570c89a: fix examples and geir(构建脚本适配/非代码缺陷)
- 0a1e2f6c: examples_fix(纯文档/注释修正)
- 03ea5201: fix aclnnFusedLinearOnlineMaxSum.md(纯文档)

### 5fd1fb95 修改lstm整包路径识别错误
- 根因类别：路径/引用错误
- 涉及文件：rnn/bidirection_lstmv2/op_kernel/bidirection_lstmv2.cpp
- 缺陷描述：`#include`路径写错，原路径`../../bidirection/op_kernel/lstm_bidir_fp16.cpp`指向错误目录，应该是`../bidirection_lstm/lstm_bidir_fp16.cpp`。整包编译场景下目录结构与开发时不同，导致include找不到正确文件或链接到错误实现
- 修复模式：单行路径字符串替换，将相对路径修正为正确的相邻目录
- 可审查性：高
- 审查规则建议：检查`#include`中使用`../../`等多级回退的相对路径引用，特别是include .cpp文件（而非.h文件）的非常规用法，应标记为高风险

### 4a423e45 fix dw acc
- 根因类别：特定shape下tiling参数缺失
- 涉及文件：conv/conv3d_backprop_filter_v2/op_host/op_tiling/arch35/conv3d_backprop_filter_v2_base_tiling.cpp
- 缺陷描述：conv3d反向滤波器算子在特定输入shape（dilation kernel场景）下走入默认tiling初始化分支，缺少stepN=390的特殊配置，导致计算精度异常。修复通过新增TILINGDATA_MAP_TMP硬编码映射表匹配该shape
- 修复模式：硬编码特定shape的tiling参数（workaround风格），新增UT用例覆盖
- 可审查性：低
- 审查规则建议：关注tiling参数中使用硬编码map+全零初始值的模式，通常是针对特定case的临时补丁，需检查是否有更通用的计算逻辑来取代hardcode

### 3e156100 gng 读越界
- 根因类别：内存读越界（DataCopy对齐多读）
- 涉及文件：norm/group_norm_grad/op_kernel/arch35/下6个文件（group_norm_grad_base.h, _c_full_load.h, _g_full_load.h, _recompute.h, _small_ng_c_full_load.h, _apt.cpp）
- 缺陷描述：GroupNormGrad算子（arch35平台）使用DataCopy搬运数据时，count参数经CeilAlign对齐到block大小。当实际数据量不是block整数倍时，对齐后的count超出GM tensor边界，造成读越界
- 修复模式：将`DataCopy(dst, src, CeilAlign(count))`替换为`DataCopyPad(dst, src, {blockLen=count*sizeof(T)})`，消除对齐导致的越界读取。系统性修复同一问题在CFullLoad/GFullLoad/ReCompute/SmallNG多个模式中的所有出现
- 可审查性：中
- 审查规则建议：搜索所有DataCopy调用中使用CeilAlign对齐count参数的地方，检查是否存在GM边界越界风险；当tensor的实际元素数不是block size整数倍时，应优先使用DataCopyPad

### e52bc834 bugfix for wqbmm
- 根因类别：模板参数声明错误（多余的TILING_STRUCT_SEL参数）
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_kernel_tiling_key.h
- 缺陷描述：wqbmm v2算子的kernel tiling key模板声明中，所有分支（约20处）都多包含了一行`ASCENDC_TPL_TILING_STRUCT_SEL(WeightQuantBatchMatmulV2*TilingData)`，导致编译或运行时异常
- 修复模式：批量删除所有tiling key模板声明中多余的ASCENDC_TPL_TILING_STRUCT_SEL行
- 可审查性：中
- 审查规则建议：ASCENDC_TPL_ARGS_DECL宏的参数列表变更时，检查所有使用该宏的分支是否保持一致

### a6b67225 修复上边界问题
- 根因类别：整数溢出（int32不足）
- 涉及文件：vfusion/modulate/op_host/modulate_tiling.cpp, op_api/aclnn_modulate.cpp
- 缺陷描述：CalcTilingParam函数的totalElements参数类型为int（32位），当tensor较大时totalElements超过INT_MAX导致溢出，后续除法计算totalElements/coreNum结果错误
- 修复模式：参数类型从int改为int64_t消除溢出
- 可审查性：高
- 审查规则建议：tiling计算函数中涉及元素数量/字节数的参数和中间变量，应使用int64_t而非int/int32_t，防止大shape场景溢出

### 9aec1447 fix addmm bug
- 根因类别：平台相关dtype校验缺失
- 涉及文件：matmul/common/op_host/op_api/matmul_util.cpp
- 缺陷描述：A2(910B)/A3(910_93)平台下调用aclnnInplaceAddmm时走入CheckGemmV3Support，但该函数缺少对这些平台上数据类型的校验——gemmv3只支持fp16*fp16和bf16*bf16，其他dtype（如fp32）被放行导致执行异常
- 修复模式：在两个重载的CheckGemmV3Support函数中各增加平台判断+dtype白名单检查
- 可审查性：高
- 审查规则建议：新增算子平台支持时，需同步审查dtype校验是否覆盖完整；当同一函数有多个重载版本时，修复需确保所有重载同步更新

### 890af289 问题单修复
- 根因类别：数据格式分支处理不完整（NZ格式stride计算错误）
- 涉及文件：conv/conv3d_backprop_input_v2/op_kernel/arch35/下2个文件
- 缺陷描述：conv3d反向输入算子在arch35平台上，不使用vecTrans且filter为NZ格式时，kernelSplitStrideB_的计算仍按NDHWC逻辑乘cin，导致stride偏大访问错误GM地址
- 修复模式：将else分支拆分为NDHWC和非NDHWC两种情况，NZ格式下stride只取wk不乘cin。新增LoadToB1Dn2NzTransposeForKernelSplitH函数处理kernel在H维度split时的权重搬运
- 可审查性：中
- 审查规则建议：当代码中出现二分分支且else分支隐含某种格式假设时，检查是否需要按实际格式细分

### 08515924 修复tilingdata从host memcpy_s npu错误
- 根因类别：API误用 / 指针语义错误
- 涉及文件：quant_batch_matmul_v4/op_host/op_tiling/quant_batch_matmul_v4_msd_tiling.cpp
- 缺陷描述：memcpy_s的第三个参数tilingData_本身已是指针类型，但代码使用&tilingData_（取指针的地址），导致拷贝的是指针变量自身的内存而非所指向的tiling数据结构体，NPU侧拿到完全错误的tiling数据
- 修复模式：将&tilingData_改为tilingData_，去掉多余的取地址操作
- 可审查性：高
- 审查规则建议：对memcpy_s/memcpy调用进行静态检查，当源参数对指针类型变量再次取地址时发出警告

### 53c9881a fix single op opgraph
- 根因类别：构建系统依赖顺序错误
- 涉及文件：cmake/gen_ops_info.cmake, cmake/symbol.cmake
- 缺陷描述：merge_graph_headers这个cmake target定义在gen_ops_info.cmake中，但它实际应作为opgraph shared library构建的依赖。single op场景下不走gen_ops_info流程时graph header合并缺失，opgraph构建失败
- 修复模式：将merge_graph_headers从gen_ops_info.cmake移到symbol.cmake中gen_opgraph_symbol函数内
- 可审查性：中
- 审查规则建议：CMake中add_dependencies(A B)的target B应在同一函数或更早的作用域中定义

### 4cd50276 fix group_norm_silu small shape case
- 根因类别：小shape边界条件处理不当
- 涉及文件：norm/group_norm_silu/op_host/group_norm_silu_tiling.cpp, op_kernel/group_norm_silu_hw1_b32.h
- 缺陷描述：hwNum==1且numGroups==shapeC（每个channel一个group的小shape场景）时：(1) tiling侧未按N维度重新分核，多核场景核间数据划分错误；(2) kernel侧gmOffset计算缺少*shapeC系数，ProcessYWithEqualC中数据搬运逻辑未正确按C维度循环
- 修复模式：tiling侧增加hwNum==1 && numGroups==shapeC特殊分支；kernel侧重写ProcessYWithEqualC为按C维度分块循环的正确实现
- 可审查性：中
- 审查规则建议：算子kernel中涉及多维tensor的偏移量计算时，审查gmOffset是否正确包含所有维度的stride因子

### 29a25677 matmulv3使能分组累加，kernel config文件修改
- 根因类别：配置参数名/类型不匹配
- 涉及文件：matmul/common/op_host/op_api/matmul_util.cpp, matmul/mat_mul_v3/op_host/config/各平台binary.json
- 缺陷描述：matmulv3引入分组累加功能后，kernel config中配置项名称仍用旧的enable_hf32(bool)，代码侧已改为opImplMode(int枚举)，名称和类型不一致导致功能无法正确使能
- 修复模式：将所有平台json配置中enable_hf32(bool/false)替换为opImplMode(int/1)
- 可审查性：中
- 审查规则建议：kernel config json中的参数名和dtype应与代码中的枚举/结构体定义保持一致

### 0f9c8ebc 修复ascend_quant_v2算子310p上编译失败的问题
- 根因类别：配置遗漏 — 平台特定kernel参数缺失
- 涉及文件：quant/ascend_quant_v2/op_host/config/ascend310p/ascend_quant_v2_binary.json
- 缺陷描述：310p平台binary配置json中缺少axis参数的声明，kernel代码读取该参数时编译失败
- 修复模式：在310p的两个kernel binary配置段中补充axis参数项
- 可审查性：高
- 审查规则建议：同一算子的不同平台binary json配置应具有一致的参数列表

### 1072e625 fix cross_entropy_loss_full_load bug
- 根因类别：ReduceMax分支处理不统一 / 未初始化寄存器参与归约
- 涉及文件：loss/cross_entropy_loss/op_kernel/arch35/cross_entropy_loss_full_load.h
- 缺陷描述：求行最大值（ReduceMax）分为cNum<vfLen和cNum>=vfLen两个分支，小分支中UnPack后直接cast再ReduceMax，对tailNum的mask处理不完整，未初始化寄存器区域参与了Max计算可能引入脏数据；边界case（cNum恰好等于vfLen时tailNum为0）行为不正确
- 修复模式：统一为一个实现：先用Duplicate(minValue)初始化寄存器，对tail部分用mask Max确保padding区域不影响结果，消除分支不一致性
- 可审查性：低
- 审查规则建议：ReduceMax/ReduceMin等归约操作前，检查寄存器是否已用安全值初始化，防止未初始化区域参与归约

跳过的提交(本轮行225-244):
- b508a0e0: fix example(纯example代码取消注释)
- 80461871: fix example(纯example代码取消注释)
- 22415d57: 修正arch35下example的License(纯License头+example warning)
- e4a0caad: fix coverage script(纯构建/CI脚本)
- a782c40c: add_layer_norm_quant fix_Warning(纯格式化/尾部空白)
- 5e8f5e7e: fix op_list info(纯文档链接修复)
- 482195e2: fix GroupNormSilu V1 V2 md(纯文档)

## 第11批 (defect_commits.txt #245-264)

### 758202d9 日志打印问题修复
- 根因类别：格式化字符串类型不匹配 + 算子定义表数据类型/格式列表数量多余
- 涉及文件：conv/conv3d_backprop_filter_v2 tiling, conv3d_backprop_input_v2 def, convolution_backward checker
- 缺陷描述：(1) OP_LOGE使用%ld打印int类型的params_.groups，32位int下格式不匹配，修复为%d。(2) conv3d_backprop_input_v2_def中910_95平台的DataType/Format列表多出DT_HIFLOAT8和DT_FLOAT8_E4M3FN对应条目，导致各列表长度不一致，删除多余条目修复。
- 修复模式：格式符修正(%ld->%d) + 列表条目删除对齐
- 可审查性：中
- 审查规则建议：启用-Wformat警告；对算子定义中DataType/Format/UnknownShapeFormat列表编写CI脚本校验长度一致性

### fe16c557 Revert "layernormv4功能补齐"
- 根因类别：功能补齐提交引入质量问题，需整体回退
- 涉及文件：norm/layer_norm_v4/ 下22个文件（tiling/kernel/UT）
- 缺陷描述：原始提交做了三件事：(1)新增CommonTiling和MergeNTiling两个tiling模板及kernel实现；(2)在SingleReadTiling::IsCapable()首行插入return false强制禁用；(3)扩大V4路径芯片适配范围(新增910B和910_93)。Revert表明新增代码质量问题，需回退到仅910_95支持。UT中tiling_key从600恢复到100(SingleRead)，说明原提交改变了tiling路由。
- 修复模式：全量Revert
- 可审查性：低
- 审查规则建议：大型功能补齐应拆分独立PR；强制禁用已有模板(return false)需注释原因；扩大芯片适配前需对应平台测试覆盖

### 538b635a fix group_norm_silu hw=1 C=G case
- 根因类别：边界条件下tiling和kernel计算逻辑错误
- 涉及文件：norm/group_norm_silu tiling.cpp, hw1_b32.h kernel, UT
- 缺陷描述：hw=1且C=G(每group只有1个channel)时两类错误。(1) tiling: SetBlockTiling中C=G特殊分支逻辑错误导致多核数据分配错乱，修复删除整个特殊分支统一走通用逻辑。(2) kernel: gmOffset计算多乘了shapeC，C=G时每组只有1个元素偏移应去掉乘shapeC；ProcessYWithEqualC中按shapeC循环但C=G时shapeD=1不应按shapeC循环；ProcessMean/Rstd中groups*shapeC修正为groups。
- 修复模式：删除tiling错误特化分支 + 修正kernel偏移和循环逻辑
- 可审查性：中
- 审查规则建议：算子tiling/kernel对边界条件(C=G,hw=1,shapeD=1)必须有UT覆盖；gmOffset偏移计算review时检查各维度乘法因子与内存布局一致性

### 41690e1c fix aclnnbng check
- 根因类别：参数校验遗漏 + 数据类型推导缺陷 + 日志级别错误 + 空指针风险
- 涉及文件：norm/batch_norm_grad_v3 tiling.cpp, aclnn_batch_norm_backward.cpp
- 缺陷描述：四个独立问题。(1) OP_LOGE用于正常调试信息，修正为OP_LOGD。(2) inference场景不允许saveMean/saveInvstd为shape[0]，但inference时可以为空，放宽校验。(3) 混合dtype场景(gradOut=fp32,weight=fp16)直接传给L0算子类型不匹配，修复显式Cast到fp32。(4) 多处L0操作结果未检查空指针(outputMask指示不需要时可能为nullptr)。
- 修复模式：日志级别修正 + 校验条件放宽 + 显式Cast + 空指针检查
- 可审查性：中
- 审查规则建议：L0/L1算子调用前所有输入tensor dtype必须显式对齐；可能为nullptr的tensor指针使用前必须检查；OP_LOGE仅用于错误路径

### 2eba3d75 fix expand indices 路由 scatterUpdate 校验
- 根因类别：整数类型截断 + 路由条件校验不完整
- 涉及文件：index/scatter_elements_v2 aclnn_scatter.cpp
- 缺陷描述：IsMeetUpdateShape函数两个问题。(1) 维度值比较用static_cast<int32_t>截断，大shape时比较结果错误，修复为int64_t。(2) 循环体只校验updatesShape==selfShape，遗漏updatesShape==indexShape的校验，可能导致不满足条件的case错误路由到scatterUpdate。
- 修复模式：类型提升(int32->int64) + 补充缺失校验条件
- 可审查性：高
- 审查规则建议：shape维度值比较一律使用int64_t禁止强转int32_t；路由条件函数需UT覆盖正例和反例

### e3e7fe45 修复特定shape下matmulv3精度问题
- 根因类别：splitK reduce阶段输出偏移计算未区分切分方向
- 涉及文件：matmul/mat_mul_v3 mat_mul_deterministic_splitk_kernel.h, UT
- 缺陷描述：ReduceKNzInUb中currOutCOffset固定为按M维度切分的偏移公式(index*N*singleCoreM)，但orderFlag=true时应按N维度切分(index*singleCoreN)。错误偏移导致输出写入位置错误产生精度问题。
- 修复模式：根据orderFlag条件分支选择不同偏移计算
- 可审查性：中
- 审查规则建议：涉及多种切分模式的kernel，偏移/stride计算必须对每种模式分别验证；UT应覆盖不同iterateOrder组合

### b694cfda 1952_ccec_bugfix
- 根因类别：特定架构(5102)不支持参数化加载接口
- 涉及文件：conv/common/op_kernel/arch35/ conv_instr_hw_mode_impl.h, conv_instr_impl.h, conv_instr_m_mode_impl.h
- 缺陷描述：arch35的Load3DBitModeParam/Load2DBitModeParam参数化接口在NPU_ARCH==5102上不支持(CCEC编译器/指令集限制)。原代码统一使用参数化调用，5102上编译错误。修复用#if条件编译在5102上改用直接调用img2colv2_cbuf_to_ca/load_cbuf_to_cb。
- 修复模式：条件编译(#if __NPU_ARCH__)区分架构指令调用方式
- 可审查性：低
- 审查规则建议：新增架构适配时系统性扫描所有参数化接口使用位置；条件编译分支需对应架构CI验证

### 7da21af9 fix aclnnindex aicpu to aicore
- 根因类别：平台适配遗漏(910_95特性分支未正确处理)
- 涉及文件：index/index aclnn_index.cpp
- 缺陷描述：aclnnIndex判断aicore路径时对910_95平台缺少特殊处理：(1)indices连续性检查对910_95不应生效但未做平台区分；(2)small tail transpose优化对910_95不适用但无条件启用；(3)is91095变量声明位置在使用之后需提前。
- 修复模式：添加平台版本条件分支用if(!is91095)守护
- 可审查性：中
- 审查规则建议：新增SoC平台支持时系统性搜索所有SocVersion条件分支确认新平台归属

### 6ea06eb1 fix SparseSoftmaxCrossEntropyWithLogits bug
- 根因类别：寄存器级向量操作逻辑错误(UnPack丢失高半部分数据)
- 涉及文件：loss/sparse_softmax_cross_entropy_with_logits arch35 full_load.h
- 缺陷描述：half精度路径中UnPack默认只提取LOWEST半数据，丢失HIGHEST部分。ReduceMax只在一半数据上求最大值导致结果偏小，后续softmax数值错误。修复分别对LOWEST/HIGHEST做UnPack+Cast后取Max合并再ReduceMax。
- 修复模式：补全数据通路——单次UnPack拆为LOWEST/HIGHEST两次，增加Max合并
- 可审查性：低
- 审查规则建议：UnPack操作review时强制要求注释数据宽度关系，确认是否需处理HIGHEST部分

### 62b86225 修复大shape超过int表示范围
- 根因类别：整数溢出(int32用于shape计算) + 输出目标tensor错误
- 涉及文件：vfusion/modulate/op_kernel/modulate.cpp
- 缺陷描述：(1) Init中baseOffset/scaleShiftOffset/bufferSize等声明为int(32位)，大tensor时乘法溢出int32导致地址计算错误。(2) Mul结果写入xLocal(源tensor)而非yLocal(目标tensor)，in-place写入破坏后续读取。
- 修复模式：int替换为int64_t + 修正Mul输出目标tensor
- 可审查性：高
- 审查规则建议：kernel代码禁止裸int做shape/offset计算，lint强制int64_t；向量计算API检查输出参数是否误与输入同buffer

### 24c2eab0 bugfix: miss check on 910_93
- 根因类别：平台适配遗漏(910_93未加入功能准入判断)
- 涉及文件：matmul/common matmul_util.cpp, matmul/mat_mul_v3 aclnn_mm.cpp
- 缺陷描述：Split-K和K=1特殊路径准入函数只检查ASCEND910B，遗漏ASCEND910_93。910_93本应支持但被排除，导致性能劣化。
- 修复模式：SoC版本判断增加ASCEND910_93
- 可审查性：高
- 审查规则建议：维护SoC能力矩阵配置表，所有特性准入用查表实现避免散落硬编码

### 1b26459e fix calssify_rule
- 根因类别：配置文件重复条目(YAML分类规则路径重复/错位)
- 涉及文件：classify_rule.yaml
- 缺陷描述：matmul-c分组包含本属于matmul和matmul-quant的路径条目，同一路径被多个owner匹配，可能导致CI流水线或审批人错误。
- 修复模式：删除重复YAML条目重新归类
- 可审查性：高
- 审查规则建议：CI校验classify_rule.yaml中同一path不允许出现在多个分组中

### e95284a2 fix adaptivemaxpool small模板
- 根因类别：buffer InitBuffer调用顺序错误导致内存布局与预期不符
- 涉及文件：pooling/adaptive_max_pool3d small_pool.h, UT
- 缺陷描述：mulWBuffer(64KB)和mulWIdxBuffer(32KB)的InitBuffer调用顺序错误。TPipe中buffer按InitBuffer顺序连续分配，交换顺序意味着起始地址互换，后续使用有基于顺序的隐含假设时数据错乱或越界。
- 修复模式：调换两行InitBuffer顺序
- 可审查性：低
- 审查规则建议：TPipe::InitBuffer调用顺序具有语义，buffer声明处应注释分配顺序依赖关系

### d34b41cb 修复图模式reg_base报错
- 根因类别：模板框架配置与编译工具链不兼容
- 涉及文件：matmul/weight_quant_batch_matmul_v2 arch35 tiling_key.h
- 缺陷描述：tiling_key中使用ASCENDC_TPL_TILING_STRUCT_SEL宏，但tbe编译器compile_op.py将opParaSize错误设为0，导致图模式下无法获取正确tilingData。去掉STRUCT_SEL后默认AS结构体(280B)大于RegBase(256B)仍能正确获取。
- 修复模式：删除所有ASCENDC_TPL_TILING_STRUCT_SEL行(28处)及相关include
- 可审查性：低
- 审查规则建议：新增ASCENDC模板框架宏时需在CI添加图模式端到端编译验证

### 9cc8c69c revert 2691
- 根因类别：不兼容重构引入编译/链接问题
- 涉及文件：matmul/fused_quant_mat_mul op_tiling arch35 fused_quant_matmul_tiling.cpp
- 缺陷描述：MR !2691(68个文件重构dlopen legacy ophost)在fused_quant_matmul_tiling.cpp新增using Ops::NN::TilingPrepareForOpCache，该符号在部分编译环境不可用导致编译/链接失败。本次仅删除该using声明(局部revert)。
- 修复模式：删除不兼容的using声明
- 可审查性：中
- 审查规则建议：大规模重构MR应分阶段提交验证，68文件单MR风险过高

跳过：
- cf8b43f1: fix assert aicpu_kernel json(纯json配置文件补充input/dynamic_input字段，无逻辑代码)
- a85a9e5e: fix weight nz demo(纯文档markdown修复demo代码示例)
- a1535094: [dx] fix aclnnConvTbcBackward introduction: Stride=1(纯文档markdown修复公式和排版)
- 93af994f: ziliao fix(纯文档/API注释修改，无代码逻辑变更)
- dbd0b626: fix dts(纯文档修改aclnnIndexCopy的md文件)

### 25d1b6bb FlatQuant修复小值域精度问题
- 根因类别：算法/精度缺陷 -- half精度不足导致小值域量化误差
- 涉及文件：quant/flat_quant/op_kernel/ flat_quant_cube.h, flat_quant_high.h, flat_quant_vec.h, tensor_utils.h
- 缺陷描述：量化计算路径直接用half做乘法再cast到int4b_t，小值域下half精度不足导致量化结果偏差大。修复改为先cast到float做乘法再CAST_RINT取整回half。同时CalReduceMax使用两次WholeReduceMax引入冗余中间buffer，NaN检测用手写`tmpMax != tmpMax`。L0C buffer position从TPosition::C2误用改为TPosition::CO1。
- 修复模式：提升中间计算精度(half->float->half路径) + 简化reduce算法 + 修正buffer位置枚举
- 可审查性：低
- 审查规则建议：量化算子中涉及scale乘法的路径应检查中间精度是否足够；避免手写NaN检测应使用标准API；Buffer TPosition枚举值应与硬件手册对应

### 06d1309e add_rms_norm_cast overflow check
- 根因类别：整数溢出 -- workspace大小计算无溢出保护
- 涉及文件：norm/add_rms_norm_cast/op_host/add_rms_norm_cast_tiling_arch35.cpp
- 缺陷描述：GetWorkspaceSize()计算workspaceSize = WORKSPACE_COUNT * usedCoreNum * numN * sizeof(float)时，numN极大(如2147483648)导致uint64_t乘法溢出，产生错误的workspace大小。修复前完全没有溢出检查。
- 修复模式：乘法前逐步除以各因子计算maxAllowed上界，超出返回GRAPH_FAILED（"先除后比"溢出检测模式）
- 可审查性：高
- 审查规则建议：tiling/workspace大小计算中涉及多变量连乘的表达式必须有溢出检查；可提取通用SafeMultiply工具函数

### 3466ffb6 fix level2
- 根因类别：拼写错误(typo) -- cmake安装路径拼写错误
- 涉及文件：cmake/variables.cmake
- 缺陷描述：ACLNN_OP_INC_INSTALL_DIR路径值中level2被拼写为levle2，导致built-in模式下头文件安装到错误路径
- 修复模式：单字符修正 levle2 -> level2
- 可审查性：高
- 审查规则建议：路径字符串可用CI脚本校验安装后目标目录是否存在；cmake安装路径建议集中管理

### ebed9d6e 修正scatter_list的proto.h
- 根因类别：语法错误 -- 缺少闭合花括号
- 涉及文件：index/scatter_list/op_graph/scatter_list_proto.h
- 缺陷描述：ScatterList算子的proto.h在OP_END_FACTORY_REG宏调用后缺少闭合}，导致编译错误
- 修复模式：添加缺失的}闭合括号
- 可审查性：高
- 审查规则建议：proto.h应有编译CI验证；新增算子proto.h应有模板校验确保namespace正确闭合

### db4b7dc3 fix adapitvemaxpool3d indices错误
- 根因类别：边界条件判断错误(off-by-one)
- 涉及文件：pooling/adaptive_max_pool3d/op_host/adaptive_max_pool3d_small_pool_tiling.cpp
- 缺陷描述：IsCapable()中判断kernel尺寸是否在限制范围内时使用<=，应为<。当kernelDMax*kernelHMax*kernelWMaxAlign恰好等于KERNEL_SIZE_LIMIT时，原代码错误认为有能力处理，但实际已达上限应走其他分支
- 修复模式：将<=改为<，收紧边界判断
- 可审查性：高
- 审查规则建议：涉及资源容量/尺寸上限的比较操作需明确"等于边界值"时的预期行为；比较操作符与LIMIT/MAX/SIZE常量组合时应确认<=与<的选择

### 3aa3ec75 算子kernel大小校验问题--C++风格问题整改
- 根因类别：编码风格/类型规范
- 涉及文件：conv/conv3d_backprop_filter_v2/op_host/op_tiling/arch35/conv3d_backprop_filter_v2_basic_block_tiling.cpp
- 缺陷描述：CheckKernelSize()使用C风格int类型和(int)floor()强制转换。改为int32_t确保类型宽度，static_cast替代C风格cast，去掉整数除法上下文中多余的floor()
- 修复模式：int->int32_t + C cast->static_cast + 去冗余floor
- 可审查性：高
- 审查规则建议：lint规则禁止C风格强制类型转换；整数运算中调用floor()/ceil()应发出警告

### e2cb86e7 [bugfix] Get L1 size by platform info
- 根因类别：硬编码常量/平台适配缺陷
- 涉及文件：conv/common/op_host/op_tiling/arch35/conv_api_tiling_util.h, conv/conv2d_v2/op_host/op_tiling/arch35/conv2d_api_tiling.cpp
- 缺陷描述：L1 cache大小被硬编码为常量L1_SIZE_VALUE=524488，不同硬件平台L1大小不同，导致tiling策略选择错误。同时日志格式串%u与uint64_t类型不匹配
- 修复模式：删除硬编码常量改为platformInfo.l1Size动态获取；修正printf格式符
- 可审查性：高
- 审查规则建议：硬件参数相关常量应优先通过platform API获取；搜索L1/L2 size、core count等硬编码常量标记审查

### 5b7c5689 fix masked_scatter l2
- 根因类别：多重逻辑缺陷(冗余转换/指针类型/控制流/分支处理)
- 涉及文件：common/stub/op_api/op_api_stub.cpp, index/masked_scatter/op_host/op_api/aclnn_masked_scatter.cpp
- 缺陷描述：(1)Cast位置错误：ProcessBroadcast内部和外部重复对mask做Cast(DT_BOOL)；(2)指针类型不安全：输出参数aclTensor** out用const_cast去const赋值；(3)循环缺break：设isBA=false后未break继续无意义迭代；(4)910_95分支传参错误：传入selfRefContiguous而非selfRef，mask未传已cast的maskBool
- 修复模式：重构函数接口(const正确性)、消除冗余操作、修正控制流、统一分支后处理
- 可审查性：低
- 审查规则建议：const_cast使用应触发审查警告；循环中设flag后不break应触发lint告警；同一数据类型转换在调用链中出现多次应检查冗余

### 261c36f2 fix AddRmsNormDynamicQuant infershape bug
- 根因类别：条件校验逻辑缺陷(动态shape场景未覆盖)
- 涉及文件：norm/add_rms_norm_dynamic_quant/op_host/add_rms_norm_dynamic_quant_infershape.cpp
- 缺陷描述：infershape中用*gammaShape != *smooth1Shape直接比较shape一致性，但动态shape场景下shape值可能是-1/-2等未知值，直接比较会错误拒绝合法输入
- 修复模式：删除不适用于动态场景的静态shape一致性校验
- 可审查性：中
- 审查规则建议：shape校验代码必须考虑动态shape场景(-1/-2等未知维度值)

### e1fdbe6e Revert "matmulv3支持选择分组累加方式计算"
- 根因类别：功能设计/接口变更回退
- 涉及文件：matmul/mat_mul_v3及batch_mat_mul_v3 多文件(proto.h/tiling/api/UT)
- 缺陷描述：将MatMulV3的enable_hf32(bool)属性改为opImplMode(int64)以支持FORCE_GRP_ACC_FOR_FP32模式的功能被完整回退，说明该特性存在问题。涉及算子属性类型变更、tiling计算公式修改等
- 修复模式：全量Revert
- 可审查性：低
- 审查规则建议：算子接口属性类型变更(bool->int)需完整兼容性测试；Revert提交应说明回退原因

### cbc33130 修复 wqbmmv2 和 qbmmv4 在子包编译失败的问题
- 根因类别：配置遗漏/依赖错误
- 涉及文件：matmul/quant_batch_matmul_v4/op_kernel/arch35/ reg_base_common.h, matmul/weight_quant_batch_matmul_v2/op_host/config/ binary.json
- 缺陷描述：(1)qbmmv4引用不存在的头文件../../common/anti_quant.h改为使用AscendC::IsSameType；(2)wqbmmv2的binary.json全部22个变体缺少inner_precise参数导致子包编译失败
- 修复模式：补齐配置参数；修正头文件依赖
- 可审查性：高
- 审查规则建议：binary.json修改应引入schema校验自动检查必填字段；引用外部头文件应验证所有编译目标下可达

### efe09d11 fix aclnn_hardsigmoid CheckParams
- 根因类别：参数校验过严(过度约束)
- 涉及文件：activation/hard_sigmoid/op_host/op_api/aclnn_hardsigmoid.cpp
- 缺陷描述：CheckParams对format校验同时要求输入输出format相同且必须为ND格式，实际算子应支持非ND格式(如NZ)，只需保证输入输出format一致即可。过严校验导致合法非ND格式输入被拒绝
- 修复模式：放宽校验条件，删除"必须为ND"约束，仅保留输入输出format一致性检查
- 可审查性：高
- 审查规则建议：参数校验中硬编码特定format(!= FORMAT_ND)时应review是否有充分理由排除其他format

### ece9f87e fix: sc of wqbmmv2
- 根因类别：编译告警/代码规范违规
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_host/op_tiling/arch35/ 多文件(.cpp/.h)
- 缺陷描述：(1)map<vector>大initializer_list导致栈帧过大触发-Wframe-larger-than告警；(2)basic_block_table.h超500行规范上限；(3)reg_base_common.h超2000行含冗余代码
- 修复模式：运行时map<vector>重构为编译期constexpr扁平数据+偏移表；大块数据从.h移到.cpp；删除冗余代码
- 可审查性：中
- 审查规则建议：大型initializer_list应在CI通过-Wframe-larger-than告警拦截；头文件行数应有lint规则检查上限

跳过：
- 81e8f37d: 【YAML】fix ops/ops-nn/common(纯YAML配置classify_rule添加路径条目，无代码逻辑)
- 50ced237: aclnnConvDepthwise2d文档类型错误修正(纯文档修改kernelSize类型描述INT32->INT64)
- aff4487f: fix gemmv2 readme(纯README修改昇腾910_95支持状态)
- 809ef7e4: aclnnapplyadamwv2的step参数(纯文档修改.md参数约束描述)
- 27bdc238: ModulateGrad fix(纯文档修改公式/链接/约束说明)
- 1d933547: fix: redundant code(仅删除一行注释// #include "op_util.h")
- 21ebd05b: 修复act仓fusedmatmul冗余整改(大规模删除重复目录fusedmatmul_act/，代码仓库整理非缺陷修复)

### 8137de26 vendor_name fix
- 根因类别：CMake变量名混淆
- 涉及文件：cmake/variables.cmake
- 缺陷描述：custom vendor包安装路径中使用了`${VENDOR_NAME}`而非`${VENDOR_PACKAGE_NAME}`，两个变量含义不同(VENDOR_NAME是原始名，VENDOR_PACKAGE_NAME是包名)，导致impl安装路径错误
- 修复模式：将IMPL_INSTALL_DIR和IMPL_DYNAMIC_INSTALL_DIR路径中的VENDOR_NAME替换为VENDOR_PACKAGE_NAME
- 可审查性：高
- 审查规则建议：CMake中多个相似命名变量(VENDOR_NAME vs VENDOR_PACKAGE_NAME)需要在review时逐一确认使用场景是否匹配

### 77dd5b52 conv forward quant mode bug fix
- 根因类别：多通路行为不一致/量化模式分派缺陷
- 涉及文件：conv/common/op_kernel/arch35/conv_instr_impl.h
- 缺陷描述：GetQuantPre函数中hif8/fp8输入类型路径无条件返回vector quant模式(VQ)，未区分extendConv2d(支持scalar quant和vector quant)与普通quant conv(只需vector quant)。当extendConv2d使用scalar quant时，错误地选择了VQ模式
- 修复模式：将单一GetQuantPre拆分为GetQuantPreHif8Fp8/GetQuantPreInt32/GetQuantPreFp32三个子函数，每个子函数内部对extendConv2d场景根据runtime quantMode选择正确的quant模式
- 可审查性：中
- 审查规则建议：量化模式分派函数中每条类型路径需验证是否覆盖所有算子变体(extendConv2d/quantConv/fixedPoint)的差异行为

### 357bb5ed fix apply_adam_w_quant
- 根因类别：复制粘贴错误/配置文件section名错误
- 涉及文件：optim/apply_adam_w_quant/op_host/config/ascend910_93/apply_adam_w_quant_simplified_key.ini (2个平台配置)
- 缺陷描述：simplified_key.ini的section名为[AdvanceStep]，应为[ApplyAdamWQuant]。从另一个算子的配置文件复制时未修改section名
- 修复模式：[AdvanceStep]→[ApplyAdamWQuant]
- 可审查性：高
- 审查规则建议：新增算子的配置文件section名必须与算子名一致，review时应检查.ini文件section名是否匹配所在目录的算子名

### 321c4fcd 算子kernel大小校验问题
- 根因类别：参数校验缺失
- 涉及文件：conv/conv3d_backprop_filter_v2/op_host/op_tiling/arch35/conv3d_backprop_filter_v2_basic_block_tiling.cpp/.h
- 缺陷描述：conv3d_backprop_filter_v2缺少对kernel尺寸(kd/kh/kw)的合法性校验。当kernel尺寸超过输入+padding允许的最大值时，不报错直接执行导致计算结果错误
- 修复模式：新增CheckKernelSize()函数，根据公式`kdMax = floor((di + padF + padB - 1) / dilationD + 1)`计算各维度kernel上界并校验
- 可审查性：高
- 审查规则建议：conv类算子tiling阶段必须校验kernel/stride/dilation/padding组合的合法性，防止超出输入维度范围

### 2942cc8c 修复降低精度计算的场景问题
- 根因类别：算子调用顺序错误/类型转换时序
- 涉及文件：conv/convolution_backward/op_host/op_api/aclnn_convolution_backward.cpp
- 缺陷描述：convolution_backward的needSpecialCast路径中，先Cast到outputTensor->GetDataType()再TransData。当输出类型不是float16时TransData可能失败或精度异常。正确顺序应为Cast(→fp16)→TransData→Cast(→output dtype)
- 修复模式：将CastOnlyForConvBackward的目标类型改为DT_FLOAT16，TransData后再Cast到最终输出类型。同时增加TransData返回值null检查
- 可审查性：中
- 审查规则建议：数据搬运流水线中多步类型转换的顺序需逐步验证中间类型是否满足下游算子的输入约束

### 26abb542 norm compile warning fix
- 根因类别：编译告警(未使用参数 + signed/unsigned比较)
- 涉及文件：index/gather_v2/op_host/op_api/aclnn_embedding_renorm.cpp, norm/renorm/op_host/op_api/renorm.cpp
- 缺陷描述：(1)CheckParams声明了maxNorm参数但未使用；(2)RenormInferShape中`dim >= real_dim_num`为signed/unsigned比较，dim为int64_t而real_dim_num为size_t
- 修复模式：(1)删除未使用参数；(2)先检查dim<0再用static_cast<size_t>(dim)比较
- 可审查性：高
- 审查规则建议：signed/unsigned混合比较必须先检查signed值非负再cast比较；函数参数需review是否全部使用

### f1827a03 修复910A未cast问题
- 根因类别：平台条件判断不完整
- 涉及文件：index/embedding_dense_grad_v2/op_host/op_api/aclnn_embedding_dense_backward.cpp
- 缺陷描述：embedding_dense_backward中needCast仅考虑scaleGradByFreq或deterministic条件，但在非910B平台(如910A)上，无论这些条件如何都需要Cast到float32。遗漏平台约束导致910A上精度错误
- 修复模式：增加平台判断：910B/910_93上沿用原条件，其他平台(包括910A)无条件设needCast=true
- 可审查性：中
- 审查规则建议：涉及精度转换的逻辑需审查是否在所有支持平台上行为正确，特别注意新增平台支持时原有条件是否仍成立

### a6360067 解决测试用例unique接口精度问题
- 根因类别：tensor别名/就地操作导致精度错误
- 涉及文件：index/scatter_elements/op_host/op_api/aclnn_unique.cpp, aclnn_unique2.cpp
- 缺陷描述：ScatterElements(sumIdx, sortedIndices, sumIdx, ...)中data和updates使用同一个tensor sumIdx。ScatterElements的scatter操作会修改data，而updates也指向同一内存，导致部分元素读到被修改后的值
- 修复模式：单独AllocTensor分配newData作为ScatterElements的data参数，避免别名
- 可审查性：高
- 审查规则建议：ScatterElements/GatherElements等in-place scatter操作中，data和updates参数不能引用同一tensor

### 9887d805 fix sc for qbmmv3 mdc
- 根因类别：编译告警/代码规范(未使用参数 + 命名规范)
- 涉及文件：matmul/quant_batch_matmul_v3/..., matmul/quant_batch_matmul_v4/... 多文件
- 缺陷描述：(1)变量名DimValueOfMKN违反camelCase规范；(2)CheckShape的scaleShape/offsetShape参数传入但未使用；(3)CheckX2TableShape的x2TableShape参数未使用(数据从成员变量获取)
- 修复模式：重命名变量；删除多余参数；清理调用点
- 可审查性：高
- 审查规则建议：静态分析工具应检测未使用的函数参数(-Wunused-parameter)

### 58f1502e 修复foreach
- 根因类别：CMake构建逻辑顺序错误
- 涉及文件：cmake/gen_ops_info.cmake
- 缺陷描述：binary.json的存在性检查在compile_from_config之后执行，但实际应在op_type判断之前执行。当算子有自定义binary.json时不需要走get_op_type_from_op_name和check_op_supported逻辑，否则会错误跳过该算子
- 修复模式：将binary.json检查提前到foreach循环开头，存在binary.json时直接使用，否则走原有op_type获取和supported检查流程
- 可审查性：中
- 审查规则建议：CMake构建脚本中条件判断的执行顺序需确保early-exit路径在正确位置

### 3fcda4f9 修复ci出包引用错同名头文件的问题
- 根因类别：include路径歧义/同名头文件冲突
- 涉及文件：conv/convolution_forward/op_host/op_api/aclnn_convolution.cpp
- 缺陷描述：`#include "matmul/common/op_host/op_api/matmul_util.h"`在CI出包时因include搜索路径中存在另一个同名matmul_util.h，导致引用到错误的头文件
- 修复模式：改用相对路径`../../../../matmul/common/op_host/op_api/matmul_util.h`消除歧义
- 可审查性：中
- 审查规则建议：跨模块include应使用显式相对路径或namespace前缀目录，避免同名头文件在不同include路径中产生歧义

### 3721c17f fix_conv3d_no_innerbatch
- 根因类别：条件判断遗漏(format维度)
- 涉及文件：conv/common/op_host/op_tiling/arch35/conv_api_tiling_algorithm_Mmode.cpp
- 缺陷描述：CalFormulaicInnerBatch函数的early return条件只检查了isC04Flag、groups>1、isDmaFlag，未检查3D卷积格式(NCDHW/NDHWC)。conv3d场景不应使用inner batch优化(仅适用于2D)，遗漏导致错误切分
- 修复模式：在early return条件中增加NCDHW和NDHWC格式检查
- 可审查性：高
- 审查规则建议：tiling优化策略的适用条件必须明确排除不支持的format/维度场景

### 01b268c9 batchNormGrad迁移遗漏
- 根因类别：日志格式化错误/迁移不完整
- 涉及文件：norm/batch_norm_grad_v3/op_host/batch_norm_grad_v3_tiling.cpp
- 缺陷描述：dtype校验的错误日志使用`%d`格式打印enum值(如`actual %d`, dxDtype)，不可读且遗漏了对比的另一个dtype。迁移到新API时未同步更新日志格式
- 修复模式：改用`Ops::Base::ToString(dtype).c_str()`输出可读的dtype名称，并同时打印两个不匹配的dtype值
- 可审查性：高
- 审查规则建议：错误日志中的enum/类型值应使用ToString转换为可读字符串；类型不匹配错误应同时打印expected和actual两个值

### d9448715 fix mixType error
- 根因类别：tiling变量计算遗漏(条件分支)
- 涉及文件：norm/rms_norm/op_host/rms_norm_tiling.cpp
- 缺陷描述：rms_norm的mixType路径中计算了blockFactor和useCoreNum但遗漏了latsBlockFactor(lastBlockFactor)的计算。最后一个核的行数因使用未初始化/上一路径残留的值而错误
- 修复模式：在SetBlockDim后增加`latsBlockFactor = numRow - blockFactor * (useCoreNum - 1)`
- 可审查性：高
- 审查规则建议：多核tiling中blockFactor和lastBlockFactor必须成对计算；每个分支路径都需独立设置完整的tiling参数

### cd81d431 修复代码问题
- 根因类别：static局部变量在运行时状态依赖场景中错误使用
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_adaptive_split_tiling.cpp
- 缺陷描述：(1)`static const uint64_t BLOCK_DIM_M_MAX`依赖运行时值compileInfoPtr_->aicNum，但static只初始化一次，后续调用使用首次的值；(2)`static const std::vector<uint64_t> L0_BASE_K_LIST`依赖运行时标志weightMxFp4Flag_和nzSceneFlag_，同样只初始化一次
- 修复模式：去掉static关键字，改为普通局部变量
- 可审查性：高
- 审查规则建议：依赖运行时状态(成员变量/参数)初始化的局部变量禁止使用static修饰，否则多次调用共享首次初始化值

### 7bc3d081 fix addRmsNorm
- 根因类别：硬件事件ID生命周期管理错误
- 涉及文件：norm/add_rms_norm/op_kernel/add_rms_norm_single_n.h
- 缺陷描述：使用FetchEventID获取事件ID后未Release，导致事件ID耗尽或复用冲突。在bf16路径中MTE2_V事件被多次Fetch但从不Release
- 修复模式：将关键路径的FetchEventID改为AllocEventID，使用后调用ReleaseEventID归还。变量重命名以区分不同生命周期的事件
- 可审查性：中
- 审查规则建议：AllocEventID/ReleaseEventID必须配对使用(类似malloc/free)；review时检查每个Alloc是否有对应Release

### 59a561e5 修复aten直调功能使得npu行为跟cpu一致
- 根因类别：空指针解引用/outputMask未检查
- 涉及文件：conv/convolution_backward/op_host/op_api/aclnn_convolution_backward.cpp, convolution_backward_checker.cpp
- 缺陷描述：(1)PreConv1DBackwardTo2D无条件对gradInput/gradWeight调用View4dWithGroups，但outputMask可能指示它们不需要计算(为null)；(2)InterceptConvFor8bit中对gradInput/gradWeight调用GetDataType但未检查null；(3)CheckDtypeValidForBpFilter8bit中对gradBias调用GetDataType但可能为null
- 修复模式：(1)用outputMask[0]/[1]守护gradInput/gradWeight的View4d调用；(2)InterceptConvFor8bit增加null检查；(3)gradBias先判null再取DataType
- 可审查性：高
- 审查规则建议：可选输出tensor(由outputMask控制)的任何访问前必须检查null或对应mask位

### 328daa2f layerNormGrad精度修复
- 根因类别：ReduceSum的src与tmp buffer别名导致数据污染
- 涉及文件：norm/layer_norm_grad_v3/op_kernel/layer_norm_grad_v3_single_read.h, layer_norm_grad_v3_transpose.h, layer_norm_grad_v3_workspace.h
- 缺陷描述：ReduceSum(dst, src, tmp, count)中src和tmp使用同一tensor。ReduceSum实现会将中间结果写入tmp区域，覆盖了src的原始数据，导致归约结果错误(精度问题)
- 修复模式：为ReduceSum分配独立的tmp buffer(通过queue.AllocTensor或额外参数传入)，修改ReduceMeanModeOne/DoLastAxisReduce签名增加tmp参数
- 可审查性：高
- 审查规则建议：ReduceSum/ReduceMax等归约操作的src和tmp参数禁止指向同一tensor；review时检查归约调用是否存在buffer别名

跳过：
- 4ac05898: fix chmod(shell脚本权限修改，非代码)
- 3036f6d9: fix AdaMaxPool2\3d aclnnMD(纯文档markdown修改)
- f6d85bd0: fix repackage(构建打包脚本增加filelist.csv复制)
- 5f563eff: 回退swigluquant手写aclnn接口(行政性代码回退，删除659行)
- 2b00825d: revert//addRmsNorm Change(行政性revert，14文件大规模回退)
- ae903f83: 修复子包卡死问题(example代码注释化，配合addRmsNorm回退)
- 5c29093a: 新增extendConvTranspose ut用例(UT新增+CodeCheck修复，非缺陷)

### 5df40a02 修复matmulv3 BL1全载模板tiling卡死问题
- 根因类别：循环终止条件缺失
- 涉及文件：matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp
- 缺陷描述：DoBL1FullLoadTilingBase()中while循环不断将baseM减半以使loadSize适配L1，但没有下界保护。当B矩阵本身就超过L1容量时，baseM会被无限减半趋近于0但永远无法使loadSize <= l1Size，导致死循环(tiling卡死)。
- 修复模式：添加前置guard check，即使baseM取最小值16仍超L1则直接返回false放弃fullLoad路径
- 可审查性：中
- 审查规则建议：包含x=x/2或x>>=1的while循环必须有明确的下界终止条件；对缩减变量的循环需检查是否存在不可满足的退出条件

### cbcf639d fix compile error in A16W8 kn A16W4 Nz
- 根因类别：复制粘贴错误(变量名+stride参数)
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/act/prologue/block/block_prologue_b_antiquant_scmc_nd_kn.h, block_prologue_b_antiquant_scmc_nd_nk_nz_kn.h
- 缺陷描述：(1) A16W8 ND kn场景中CeilDiv参数使用了antiQuantNOffset/antiQuantKOffset，但该代码路径中正确变量名是nOffset/kOffset——从另一场景复制代码时未替换变量名。(2) A16W4 NZ场景中stride设置用了复杂动态计算表达式，实际NZ格式kn布局stride应是固定的MakeStride(_128{}, _1{})。
- 修复模式：修正变量名引用；修正stride常量值
- 可审查性：高
- 审查规则建议：新增kernel代码路径应有编译验证用例；copy-paste代码段需逐个检查变量名是否适配新上下文

### ab87ed7b ascend_quant_v2算子修复上边界error
- 根因类别：整数类型溢出(int32用于int64场景)
- 涉及文件：quant/ascend_quant_v2/op_kernel/ascend_quant_v2_nz.h
- 缺陷描述：kernel中多个变量(N, blockIdx, needCoreNum, 循环变量kloop/nloop/offset_n/offset_k等)声明为int32_t，但参与的运算(如K*i*64, 16*16*needCoreNum*j)结果可能超过int32上限，导致整数溢出。当输入tensor维度较大时offset计算错误，产生越界访问。
- 修复模式：扩大整数类型宽度(int32 -> int64)
- 可审查性：高
- 审查规则建议：kernel代码中参与地址/偏移量计算的变量应统一使用int64_t；对int32_t变量参与乘法运算的表达式应检查溢出可能

### 5ccfc384 fix SparseSoftmaxCrossEntropyWithLogits bug
- 根因类别：SIMT资源配置参数不当(LAUNCH_BOUND过大)
- 涉及文件：loss/sparse_softmax_cross_entropy_with_logits/op_kernel/arch35/sparse_softmax_cross_entropy_with_logits_full_load.h, sparse_softmax_cross_entropy_with_logits_split_r.h
- 缺陷描述：SIMT kernel的LAUNCH_BOUND从2048改为1024。LAUNCH_BOUND控制并发线程数上限，设为2048时每个线程分配到的栈/寄存器资源减半，可能导致栈溢出或寄存器spilling引发运行时错误。
- 修复模式：降低并发线程数以保证每线程资源充足
- 可审查性：低
- 审查规则建议：SIMT kernel的LAUNCH_BOUND值应有资源计算依据(栈大小/寄存器数量)，不应随意设置

### 5a2316b5 fix multi_add_rms_norm_dynamic_quant safeDiv &+
- 根因类别：复合缺陷(copy-paste + SafeDiv符号逻辑 + 空指针防护缺失)
- 涉及文件：norm/multi_add_rms_norm_dynamic_quant/op_host/multi_add_rms_norm_dynamic_quant_infershape.cpp, multi_add_rms_norm_dynamic_quant_tiling.cpp
- 缺陷描述：(1) infershape中获取yShape后空指针检查写成了OP_CHECK_NULL_WITH_CONTEXT(context, xShape)——检查的是xShape而非yShape，yShape为空时不会被拦截。(2) SafeDiv当除数b为负数且绝对值小于EPSINON时将b替换为正的EPSINON，导致除法结果符号反转。(3) epsPtr为nullptr时静默跳过不设置eps，导致eps使用未初始化值。
- 修复模式：修正copy-paste变量名；修正SafeDiv符号逻辑(b<0时用-EPSINON)；增加空指针防护
- 可审查性：高
- 审查规则建议：OP_CHECK_NULL参数应与前一行获取的变量名一致；SafeDiv需处理负数near-zero场景；可选参数nullptr分支应有显式fallback或报错

### d48e63bf bugfix
- 根因类别：缺少namespace声明
- 涉及文件：optim/apply_ftrl/op_kernel/apply_ftrl_apt.cpp
- 缺陷描述：文件使用ApplyFtrlOpTiling命名空间中的类型/常量，但缺少using namespace ApplyFtrlOpTiling声明，导致编译时找不到符号。新增tiling数据结构后忘记在kernel文件中引入对应命名空间。
- 修复模式：添加缺失的using namespace声明
- 可审查性：高
- 审查规则建议：kernel cpp文件中引用的tiling结构体所在namespace必须显式声明；CI编译应覆盖所有kernel目标

### ce6faf51 回退公共方法导致算子用例执行失败问题
- 根因类别：API兼容性/平台适配不完整
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_kernel/tool.h
- 缺陷描述：将ASCENDC_ASSERT(大写宏)整改为ascendc_assert(小写函数)，但ascendc_assert在部分芯片上不支持，导致算子用例执行失败。未验证全部目标芯片的API可用性。
- 修复模式：API调用回退为ASCENDC_ASSERT宏形式
- 可审查性：中
- 审查规则建议：涉及公共工具函数的API变更须在所有支持芯片平台上验证兼容性

### ada7984a support noalign and fix k>65535 kernel error
- 根因类别：同名函数行为不一致导致整数溢出
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/act/下多个文件
- 缺陷描述：K维度>65535时kernel出现AIC ERROR和TASK Timeout。代码中AscendC::CeilDiv/CeilAlign与Act::CeilDiv/CeilAlign的实现不同——处理大数值(uint64_t)时前者存在隐式类型截断或溢出。
- 修复模式：统一使用Act命名空间下的正确实现，部分调用处加static_cast<uint64_t>
- 可审查性：中
- 审查规则建议：同一项目中不应存在同名但行为不同的工具函数；维度参数运算需检查是否支持超过16位整数范围

### 75468d69 fix aclnn_avgpool2d_backward
- 根因类别：分支逻辑缺失(维度处理不完整)
- 涉及文件：pooling/avg_pool3_d_grad/op_host/op_api/aclnn_avgpool2d_backward.cpp
- 缺陷描述：获取cDims时只处理gradOutput维度为kNCHWDIM(4维)情况，缺少else分支。3维输入(NCL)时cDims保持初始值1而非从正确维度索引(kCDimNCLIdx)获取。
- 修复模式：补全else分支覆盖3维输入场景
- 可审查性：高
- 审查规则建议：条件分支处理多种shape格式时确保所有维度情况都有对应逻辑；对有默认值的变量检查是否存在遗漏的赋值路径

### a881847a kv_rms_norm_rope_cache修复host
- 根因类别：复合缺陷(量化模式判断逻辑错误 + 空指针检查缺失 + copy-paste日志变量名)
- 涉及文件：norm/kv_rms_norm_rope_cache/op_host/下多个tiling文件
- 缺陷描述：(1) GetQuantMode用scale和offset组合区分量化模式，逻辑过于复杂且存在错误(offset为空但scale非空时返回-1而非有效模式)。(2) gammaShapePtr缺少空指针检查。(3) 错误日志中k_rope_offset误写为k_rope_scale、c_kv_offset误写为c_kv_scale。
- 修复模式：简化量化模式枚举(合并为QUANT_MODE)，增加空指针校验，修正日志
- 可审查性：高
- 审查规则建议：可选输入的组合判断应有明确状态机定义；GetInputShape返回值使用前必须做空指针检查；错误日志变量名应与实际校验变量一致

### 9fd32a44 fix adaptiveavgpool3dgrad bug
- 根因类别：复制粘贴错误(input/output buffer搞混)
- 涉及文件：pooling/adaptive_avg_pool3d_grad/op_kernel/adaptive_avg_pool3d_grad_cast.h
- 缺陷描述：atomicAdd模式下清零workspace时使用了inputLocalFloat作为清零源和DMA拷贝源，应为outputLocalFloat。inputLocalFloat是输入tensor的local buffer，用它做清零源会导致输入数据被破坏且清零值不正确。
- 修复模式：将inputLocalFloat替换为outputLocalFloat
- 可审查性：高
- 审查规则建议：清零操作的目标buffer应与后续写回的目标buffer一致；同一函数中多个相似命名buffer需特别关注使用正确性

### 811e50f8 修复deepseek-r1主线dsq执行报错问题
- 根因类别：循环控制逻辑错误(break位置不当)
- 涉及文件：quant/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_cut_group.h
- 缺陷描述：group切分循环中，realDimx_<=0时的break逻辑放在if(groupIdx==cuGroupIdx)分支内部。当中间存在空group但还未轮到当前core处理时循环不会退出，导致后续使用非法的groupOffset_/realDimx_值。
- 修复模式：将终止条件从嵌套分支内提升到循环顶层
- 可审查性：高
- 审查规则建议：循环中的提前退出条件应在循环体最早位置检查，不应被嵌套在无关条件分支内；多core并行遍历时空数据的跳过/终止逻辑应独立于core分配逻辑

### 7f3cb998 fix accuracy
- 根因类别：数学公式实现错误(操作数搞混)
- 涉及文件：norm/deep_norm_grad/op_kernel/deep_norm_grad_cut_d.h, deep_norm_grad_merge_n.h
- 缺陷描述：DeepNorm反向传播梯度计算中多处向量运算的source/destination tensor角色混淆。计算dvar时使用了inputGamma(dy*gamma)而非inputGx(x_hp-mean)；dgx和dgamma计算中也有操作数组合顺序错误。
- 修复模式：重新梳理数学推导步骤，修正每步向量运算的输入/输出tensor指向
- 可审查性：低
- 审查规则建议：复杂数学公式的kernel实现应附带伪代码或公式注释；精度相关改动需有参考实现(如PyTorch)做对比验证

### 113df1fc fix ut
- 根因类别：CMake include路径过宽
- 涉及文件：cmake/ut.cmake
- 缺陷描述：UT编译的include目录使用项目根目录(PROJECT_SOURCE_DIR/)，过于宽泛可能导致头文件解析到错误同名文件。
- 修复模式：将include路径精确化到具体的tests/ut/common目录
- 可审查性：高
- 审查规则建议：CMake的target_include_directories不应将项目根目录加入include搜索路径，应指定到具体子目录

### f27cf969 修复问题单dts2025101722772
- 根因类别：复合缺陷(InferShape平台特定逻辑缺失 + kernel流水线同步缺失)
- 涉及文件：quant/trans_quant_param_v2/op_host/trans_quant_param_v2_infershape.cpp, quant/trans_quant_param_v2/op_kernel/trans_quant_param_v2.h
- 缺陷描述：(1) InferShape输出shape推导缺少芯片型号(Ascend910_95)特定的分支逻辑，当offset dim(0) > scale dim(0)时应取offset shape。(2) Kernel中多处GM到UB的数据搬运与计算之间缺少PipeBarrier<PIPE_ALL>同步屏障，数据搬运未完成就开始计算。
- 修复模式：增加芯片判断的InferShape分支；插入流水线屏障
- 可审查性：中
- 审查规则建议：InferShape需覆盖所有目标芯片的shape推导规则；GM/UB间数据搬运必须在使用数据前设置流水线同步屏障

### 95d3488d fix adaavgpool2dbackward
- 根因类别：输入校验不完整
- 涉及文件：pooling/adaptive_avg_pool3d_grad/op_host/op_api/aclnn_adaptive_avg_pool2d_backward.cpp
- 缺陷描述：CheckInputOutputShape只校验out和self的batch/channel维度一致性，遗漏gradOutput的维度校验。gradOutput前几维与self不匹配时不报错直接进入计算。
- 修复模式：扩展校验函数参数和范围，增加gradOutput的shape一致性检查
- 可审查性：高
- 审查规则建议：backward算子的输入校验必须覆盖grad tensor的shape与前向输入的一致性

---

跳过的提交(第310-329批)：
- 567eb75b: fix aclnn资料(纯文档修改)
- 12d88c6c: 修复卸载脚本问题(shell脚本路径修复)
- 9b5fdc31: 修复静态库出包(打包/命名空间符号导出)
- 31b74f3a: 修复子包安装问题(安装脚本/打包配置)

## 第330-349批 (2025-11-24 ~ 2025-11-26)

### 930d93a8 【FIX】AdaLayerNormV2
- 根因类别：条件判断逻辑错误
- 涉及文件：norm/ada_layer_norm/op_host/ada_layer_norm_tiling.cpp
- 缺陷描述：用OR逻辑判断isWeightFloat，weight/bias任一为float就设true，但当weight不存在而bias是float时会误判。修复改为逐个检查存在且非float时显式置false
- 修复模式：复合OR条件拆分为独立if块
- 可审查性：中
- 审查规则建议：对含多个可选输入(optional tensor)的布尔条件判断进行review，确保null情况下逻辑正确

### 76593c3a revert//ascendc_assert整改
- 根因类别：API名称大小写不兼容
- 涉及文件：多个matmul相关kernel头文件
- 缺陷描述：之前将ASCENDC_ASSERT改为ascendc_assert（小写），但小写版本与当前SDK不兼容，需回退为大写
- 修复模式：批量文本替换ascendc_assert -> ASCENDC_ASSERT
- 可审查性：高
- 审查规则建议：API名称变更前应确认目标版本兼容性

### 4ee704f8 修复arch35 qbmmv4 tiling base类中tilingData赋值行为
- 根因类别：API误用(序列化方法)
- 涉及文件：matmul/quant_batch_matmul_v4/op_host/op_tiling/arch35/quant_batch_matmul_v4_tiling.cpp
- 缺陷描述：SaveToBuffer()行为不符合预期（可能只拷贝部分数据或格式不匹配），修复改为memcpy_s直接拷贝struct原始内存并加EOK检查
- 修复模式：高层API替换为底层安全内存拷贝
- 可审查性：中
- 审查规则建议：tiling数据序列化应有统一规范，审查所有SaveToBuffer调用点确认语义

### 4e794227 修复910B不支持的算子编译问题
- 根因类别：构建配置错误(错误级别过高)
- 涉及文件：cmake/gen_ops_info.cmake
- 缺陷描述：compute_unit无对应算子时用FATAL_ERROR中断编译，实际应为WARNING跳过
- 修复模式：FATAL_ERROR降级为WARNING
- 可审查性：高
- 审查规则建议：CMake中FATAL_ERROR应仅用于不可恢复错误，可选组件缺失用WARNING

### 362fed23 fix avgpool2d
- 根因类别：分支选择缺陷(维度边界)
- 涉及文件：pooling/avg_pool3_d/op_host/op_api/aclnn_avgpool2d.cpp
- 缺陷描述：C维度>=65536时仍走AvgPool(Cube)分支，但该分支不支持大C维度。修复增加cDims<cMaxDims条件约束
- 修复模式：增加维度阈值边界检查
- 可审查性：中
- 审查规则建议：算子分支选择逻辑应对所有维度参数做范围校验

### 29ea53b8 fix wrong var issue
- 根因类别：变量名拼写错误
- 涉及文件：scripts/package/ops_nn/scripts/ops_nn_custom_install.sh
- 缺陷描述：$src_kernel_path误写为$ops_kernel_path（不存在的变量），glob匹配失败
- 修复模式：变量名修正
- 可审查性：高
- 审查规则建议：shell脚本应开启set -u；用shellcheck捕获未定义变量引用

### d9034ea4 maskedsoftmaxwithrelposbias算子修正tiling逻辑错误
- 根因类别：整数溢出 + 索引常量错误 + 零值检查缺陷
- 涉及文件：norm/masked_softmax_with_rel_pos_bias/op_host/masked_softmax_with_rel_pos_bias_tiling.cpp
- 缺陷描述：三类问题：(1)多个size变量用uint32_t，大shape溢出；(2)输出shape获取用了输入索引X_INPUT_INDEX(=0)而非Y_OUTPUT_INDEX，且Y_OUTPUT_INDEX定义也错(3应为0)；(3)零值检查b_*w_*n_*s1_*s2_==0乘积本身也可能溢出
- 修复模式：类型提升uint32->uint64、索引常量修正、零值判断拆分为逐项检查
- 可审查性：中
- 审查规则建议：tiling计算中尺寸变量应默认uint64_t；零值检查应逐项而非乘积

### d84cb711 modulategrad fix
- 根因类别：API误用(可选输入检测方法)
- 涉及文件：vfusion/modulate_grad/op_host/modulate_grad_tiling.cpp
- 缺陷描述：通过GetOptionalInputShape()返回值是否为null判断可选输入是否存在，但shape可能在输入不存在时仍返回非null。修复改用GetOptionalInputTensor()检查tensor本身
- 修复模式：shape存在性检查替换为tensor存在性检查
- 可审查性：高
- 审查规则建议：可选输入存在性判断应统一使用GetOptionalInputTensor()!=nullptr

### c1262c16 topKtopPSample selLogitsOut sync fix
- 根因类别：初始值错误 + 多核同步缺陷
- 涉及文件：index/top_k_top_p_sample/op_kernel/top_k_top_p_sample.h
- 缺陷描述：(1)logitsTopKPSelect输出默认值初始化为0.0，应为-inf；(2)InitGlobalMemory写入GM后缺少MTE3->MTE2同步，后续读取可能拿到脏数据
- 修复模式：初始值常量修正 + 增加硬件同步屏障
- 可审查性：低
- 审查规则建议：GM初始化后必须有对应同步屏障；输出tensor默认值应与文档约定一致

### b74a1dce 修复ACT代码名称错误
- 根因类别：标识符拼写错误
- 涉及文件：common/act/matmul/kernel/kernel_qgmm_inplace_add.h等
- 缺陷描述：类名KernelQGmmInpaceAdd中Inpace是Inplace的拼写错误，涉及类名/构造函数/析构函数批量修正
- 修复模式：标识符重命名Inpace->Inplace
- 可审查性：高
- 审查规则建议：新增标识符应做拼写检查，可集成cspell到CI

### 9ee6a9ef fix aclnn
- 根因类别：复制粘贴错误(注释)
- 涉及文件：多个pooling算子API头文件
- 缺陷描述：doxygen注释中引用了错误的函数名（如MaxPool2D应为AdaptiveMaxPool2d），copy-paste导致
- 修复模式：修正注释中的函数名引用
- 可审查性：高
- 审查规则建议：API头文件注释中函数名引用应与实际声明一致

### 86c08aea fix custom pkg issue
- 根因类别：构建配置错误(安装路径)
- 涉及文件：cmake/gen_ops_info.cmake
- 缺陷描述：ENABLE_CUSTOM模式下安装路径硬编码附加了ops_nn子目录，导致自定义算子包文件安装到错误位置
- 修复模式：增加ENABLE_CUSTOM条件分支，自定义模式下子目录设为空
- 可审查性：高
- 审查规则建议：install路径应支持可配置子目录，自定义包构建需独立集成测试

### 68c878cf BatchNormGradV3算子精度问题修改
- 根因类别：参数传递错误(值传递应为指针传递)
- 涉及文件：norm/batch_norm_grad_v3/op_host/op_api/aclnn_fast_batch_norm_backward.cpp
- 缺陷描述：函数接收const aclTensor*参数并在内部Transpose()后重新赋值，但值传递导致调用者指针未更新，后续计算使用未转置tensor，造成精度问题
- 修复模式：改为const aclTensor**（二级指针）实现出参语义
- 可审查性：高
- 审查规则建议：函数若需修改调用者的指针变量必须使用二级指针或引用

### 16168cae bug fix (dequant_swiglu_quant)
- 根因类别：数据类型对齐计算错误
- 涉及文件：quant/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_static_base.hpp等
- 缺陷描述：BiasType为bfloat16/half时，偏移量计算使用了int8_t对齐的alignColNum，但不同类型对齐要求不同，导致DataCopyPad偏移错误
- 修复模式：引入biasAlignColNum按BiasType字节宽度独立计算对齐
- 可审查性：低
- 审查规则建议：混合精度数据搬运中对齐偏移量必须按实际数据类型计算

### 1257005b fix sc修复编译告警问题
- 根因类别：预处理宏未定义导致告警
- 涉及文件：matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_apt_tiling_key.h
- 缺陷描述：#if条件中直接使用ORIG_DTYPE_SCALE==DT_FLOAT，宏未定义时预处理器视为0可能意外为true
- 修复模式：增加defined()前置检查
- 可审查性：高
- 审查规则建议：#if中使用自定义宏前必须先用defined()检查；开启-Wundef编译选项

### f9c776ed fix 95 kernel build
- 根因类别：构建脚本错误(exit vs return + 缓存失效)
- 涉及文件：scripts/kernel/binary_script/build_binary_single_op.sh, gen_opcinfo_for_socversion.sh
- 缺陷描述：(1)单算子生成失败时exit 1中断整个进程，应为return跳过；(2)opcinfo文件存在时exit 0跳过但不处理缓存过期
- 修复模式：exit 1改为return + 删除错误的缓存跳过逻辑
- 可审查性：高
- 审查规则建议：构建脚本中单组件失败不应中断整体流程，用return而非exit

### f7c3a307 修改 index_select ut 错误
- 根因类别：UT代码不兼容(废弃API调用)
- 涉及文件：index/gather_v2/tests/ut/op_host/test_aclnn_embedding.cpp等
- 缺陷描述：UT调用TestPrecision()方法在当前框架版本不可用；废弃的test_aclnn_gather_v3.cpp未清理；build.sh缺少return
- 修复模式：删除不兼容API调用、删除废弃文件、增加提前返回
- 可审查性：高
- 审查规则建议：UT框架升级后应CI自动验证所有UT编译通过

---

跳过的提交(第330-349批)：
- 67829d07: fix aclnnConvTbcBackward introduction(纯文档/数学公式修正)
- cf468aea: fix readme(文档修复)
- c8883dc4: fix aclnn ziliao(文档修复)

---

### c94f6927d9a41cdf23ebbe3e4d9420a097b1c7d5 fix api
- 根因类别：构建脚本变量未初始化
- 涉及文件：build.sh
- 缺陷描述：build_ut函数中enable_cov变量在CI模式循环内使用但未在循环前初始化，导致后续coverage收集逻辑行为不确定
- 修复模式：在循环前添加`local enable_cov=FALSE`初始化，循环内匹配时设为TRUE
- 可审查性：高
- 审查规则建议：shell脚本中条件分支内使用的标志变量应在分支外先初始化默认值

### b529184f7abfa0859a6c3ce26c7fc753db5545be fix rmsNorm
- 根因类别：流水线同步事件类型错误
- 涉及文件：norm/rms_norm/op_kernel/rms_norm_merge_n.h
- 缺陷描述：DataCopyCustom从GM→Local使用MTE2通道，但SetFlag/WaitFlag错误使用了HardEvent::MTE3_S（MTE3是Local→GM方向）。gamma数据搬入后用错误事件同步，可能导致Cast操作读到未搬运完的数据
- 修复模式：HardEvent::MTE3_S → HardEvent::MTE2_S
- 可审查性：高
- 审查规则建议：SetFlag/WaitFlag的HardEvent类型必须与实际DMA搬运方向一致：MTE2=GM→Local，MTE3=Local→GM

### b065c36c89707ad4fb54cf9be12e101b24e18036 fix L0C verification in AdjustSmallCaseBaseBlock
- 根因类别：资源溢出校验缺失
- 涉及文件：conv/conv3d_backprop_filter_v2/op_host/op_tiling/arch35/conv3d_backprop_filter_v2_stream_k_tiling.cpp
- 缺陷描述：AdjustSmallCaseBaseBlock为提升核利用率尝试扩大blockBaseM，但循环中未校验调整后的singleShapeM * blockBaseN * L0C_DTYPE_BYTE是否超出L0C_SIZE。扩大后L0C溢出导致计算错误。同时dbL0C仅在条件满足时设为DB_ON，不满足时缺少显式设为DB_OFF
- 修复模式：循环中增加L0C溢出检查并break；else分支显式设dbL0C=DB_OFF
- 可审查性：高
- 审查规则建议：调整tiling参数后必须重新校验所有相关buffer(L0A/L0B/L0C/UB)是否溢出；双buffer标志应有显式的else分支

### 8ece91366f7dfb28800b5065bc2bccd6893740da fix adaptiveavgpool2d bug
- 根因类别：API访问方式错误
- 涉及文件：pooling/adaptive_avg_pool3d/op_host/op_api/aclnn_adaptive_avg_pool2d.cpp
- 缺陷描述：使用outputShape[i]和inputShape[i]通过下标运算符访问shape维度，而正确的API是GetDim(i)。下标运算符可能返回错误结果或编译通过但语义不同
- 修复模式：outputShape[i] → outputShape.GetDim(i)，inputShape[i] → inputShape.GetDim(i)
- 可审查性：高
- 审查规则建议：shape对象维度访问应统一使用GetDim()方法，禁止使用operator[]

### 65fef269563dbf822f2a0fc12a51e0b34056fa20 修改review意见，修改变量未初始化红线问题
- 根因类别：变量初始化顺序/静态分析告警
- 涉及文件：common/act/matmul/kernel/kernel_qbmm_pertile.h
- 缺陷描述：QuantMmBatchPertile的ProcessMainLoop中cacheID = GetCacheID(i)声明在VFBinaryReduceSumWithoutTail之后，静态分析工具标记为潜在未初始化风险。同时重构ProcessSingleBatch拆分为ProcessWithoutBatch和ProcessWithBatch两个函数，分离batch=1和多batch逻辑
- 修复模式：将cacheID声明移至函数体前部（CopyInputsToUB之前），确保在所有使用路径上已初始化
- 可审查性：中
- 审查规则建议：变量声明应尽早在使用前定义，避免跨越复杂流水线操作导致静态分析误报

### 163f7220a1e817bb733cc4563246f787cc624d4e 安全扫描问题修改
- 根因类别：安全扫描多类问题(类型双关/整数溢出/数组越界)
- 涉及文件：norm/deep_norm/op_host/deep_norm_tiling.cpp, norm/rms_norm/op_host/rms_norm_tiling.cpp, norm/rms_norm_grad/op_host/rms_norm_grad_tiling.cpp, norm/batch_norm_grad_v3/tests/ut/op_kernel/test_batch_norm_grad_v3.cpp
- 缺陷描述：(1) deep_norm中reinterpret_cast<uint32_t*>(&eps)进行float→uint32类型双关违反strict aliasing规则；(2) FindPowerTwo参数类型int32_t无法处理大于2^31的值，且缺少64位移位；(3) rms_norm_grad中未校验dyDimNum >= gammaDimNum就用差值做索引可能越界；(4) GetShapeSize非inline导致多编译单元ODR违规
- 修复模式：(1) 改用memcpy_s安全转换；(2) 参数改uint64_t并增加n|=n>>32；(3) 增加前置校验dyDimNum > gammaDimNum；(4) 添加inline修饰
- 可审查性：高
- 审查规则建议：(1) 禁止reinterpret_cast做类型双关，用memcpy_s替代 (2) 位运算工具函数参数类型应匹配实际使用场景的值域 (3) 用维度差做数组索引前必须校验不会为负

### ee7a3e343929b21498b4983be3e9473e0cde4c7a fix ci
- 根因类别：构建脚本路径查找不完整
- 涉及文件：build.sh, activation/clipped_swiglu/tests/ut/op_host/test_clipped_swiglu_tiling.cpp
- 缺陷描述：build_single_example中检查libascendcl.so只查找EAGER_LIBRARY_PATH，未查找EAGER_LIBRARY_OPP_PATH备选路径，导致OPP部署场景下编译失败
- 修复模式：添加`|| [ -f ${EAGER_LIBRARY_OPP_PATH}/libascendcl.so]`条件（注：原修复中`]`前缺空格可能引入新问题）
- 可审查性：高
- 审查规则建议：库文件路径查找应覆盖所有部署场景的可能路径；shell test语句`]`前必须有空格

### 21f78d5a0cac90dba72b15b88a5adede374afc1a fix kernel error
- 根因类别：构建脚本错误传播缺失
- 涉及文件：scripts/kernel/binary_script/build_binary_single_op.sh, scripts/kernel/binary_script/build_binary_single_op_gen_task.sh
- 缺陷描述：build_binary_single_op_gen_task.sh的多个失败路径exit 0（成功码）或不检查子脚本返回值。OPC信息为空时exit 0导致上层脚本以为成功，binary_config_file不存在时同样静默成功。主脚本build_binary_single_op.sh不检查gen_task子脚本返回值
- 修复模式：各失败路径使用不同exit code(2/3/4/5)区分错误类型；主脚本检查子脚本返回值并报错退出；函数末尾添加显式exit 0
- 可审查性：高
- 审查规则建议：(1) shell脚本中WARNING+exit 0是危险模式，失败路径必须exit非零 (2) 调用子脚本后必须检查$? (3) 不同失败场景应使用不同exit code便于定位

### e68215487077fca2c7b32109ca6d13962ecf7bc8 回退//remove mmv3 bf16 bias compile
- 根因类别：Revert(过早移除配置)
- 涉及文件：matmul/mat_mul_v3/op_host/config/ascend910b/mat_mul_v3_binary.json
- 缺陷描述：之前的MR移除了MatMulV3的bf16 bias编译配置(binary.json中BF16_BF16_BF16_BF16变体)，但该配置仍有使用场景，移除导致对应数据类型组合的MatMul无法binary编译
- 修复模式：Revert整个移除操作，恢复bf16 bias的binary.json配置条目
- 可审查性：中
- 审查规则建议：移除binary.json中的算子变体前，必须确认该变体在所有下游场景中已无调用

### ccfe9770d0d2242a5dc1862b65ba8a2833293158 fix GroupNormSilu Tiling
- 根因类别：整数溢出 + 错误传播缺失
- 涉及文件：norm/group_norm_silu/op_host/group_norm_silu_tiling.cpp
- 缺陷描述：(1) SetBlockTiling中shapeN * numGroups直接相乘可能uint32溢出（两个较大的uint32乘积超出32位范围），导致numPerCore/realCoreNum计算错误；(2) SetProcessSize和SetTilingSD返回void，内部检测到除零等错误时OP_LOGE后return但不返回错误码，调用方无法感知失败继续使用无效tiling
- 修复模式：(1) 显式cast为uint64_t后再乘：uint64_t totalGroups = static_cast<uint64_t>(shapeN) * static_cast<uint64_t>(numGroups)；(2) 返回类型改ge::graphStatus，OP_CHECK_IF失败返回GRAPH_FAILED，调用方检查返回值
- 可审查性：高
- 审查规则建议：(1) 两个32位值相乘用于64位变量时必须先提升至少一个操作数为uint64_t (2) tiling辅助函数不应返回void，检测到异常必须通过返回值传播错误

### a4223bfa1b21a999345aca86592e041e4559ed93 解决910编译问题
- 根因类别：平台配置遗漏
- 涉及文件：activation/hard_swish_grad_v2/op_host/hard_swish_grad_v2_def.cpp
- 缺陷描述：HardSwishGradV2算子的ascend910配置缺少DynamicCompileStaticFlag/DynamicRankSupportFlag/DynamicShapeSupportFlag三个编译标志，导致910平台上动态shape/rank场景编译失败
- 修复模式：在config910A上添加三个.Flag(true)调用
- 可审查性：高
- 审查规则建议：新增算子或新增平台配置时，对照同类算子检查是否遗漏Dynamic*Flag系列编译标志

### 6f5cf633357228244d86c4775528f9772c5b8c9e fix rmsnormgrad bug
- 根因类别：流水线同步屏障缺失 + 变量声明顺序
- 涉及文件：norm/rms_norm_grad/op_kernel/arch35/rms_norm_grad_regbase_dgamma.h
- 缺陷描述：(1) dgamma二分累加Level2完成后，缺少VEC_STORE→VEC_LOAD的LocalMemBar屏障，Level1累加可能读到未完成写入的数据（load等store依赖同步缺失）；(2) cacheID = GetCacheID(i)在两处函数中声明位置过晚，移至函数顶部确保资源ID在流水线操作前获取
- 修复模式：(1) 在Level2循环后添加MicroAPI::LocalMemBar<VEC_STORE, VEC_LOAD>()；(2) cacheID声明移至CopyInputsToUB之前
- 可审查性：中
- 审查规则建议：二分累加/归约操作的不同level之间必须插入对应方向的MemBar；向量Store后Load同一地址必须有VEC_STORE→VEC_LOAD屏障

### 2d6f9cb4996f63280870af28ccfd211afb09c8c5 Revert "[MatMul] modify range of shape to transdata"
- 根因类别：Revert(条件判断过于宽松)
- 涉及文件：matmul/common/op_host/op_api/matmul_util.cpp
- 缺陷描述：原始修改将IsNdToNzOnTheFly中innerAxis的判断简化为仅检查上界(<=65535)，移除了下界和16对齐检查。但transdata要求小于128的innerAxis必须16对齐，否则数据排布错误。Revert后恢复完整判断：>=128时检查上界，<128时检查16对齐
- 修复模式：恢复kInnerAxisMinLimit=128和分段条件：(axis >= 128 && axis <= 65535) || (axis < 128 && (axis & 0xF) == 0)
- 可审查性：中
- 审查规则建议：NdToNz transdata的轴长度需同时满足上界和对齐约束，简化条件判断时不能丢弃对齐检查

### 2463697d9914e180d7105870f100e27b9f89f6ec fix back
- 根因类别：可选输入空指针风险
- 涉及文件：norm/group_norm_silu/op_host/group_norm_silu_tiling_arch35.cpp
- 缺陷描述：CheckMixRegBase中通过GetOptionalInputDesc获取gamma/beta的dtype来判断是否支持regbase，但gamma/beta是可选输入，可能为null。虽然有OP_CHECK_NULL_WITH_CONTEXT，但不同场景下可选输入的存在性不可靠。修改为用GetOutputDesc获取output1/output2的dtype判断，output始终存在
- 修复模式：GetOptionalInputDesc(INPUT_IDX_GAMMA/BETA) → GetOutputDesc(INPUT_IDX_GAMMA/BETA)，变量名从gamma/betaDtype改为out1/out2Dtype
- 可审查性：高
- 审查规则建议：dtype兼容性判断优先使用必选张量(output)而非可选张量(optional input)，避免null风险

### 14a737503ff9b9ad98a8019cecc3799c9f42e3fd MinUsedDim fix from being zero
- 根因类别：除零错误
- 涉及文件：matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp
- 缺陷描述：CalcTile中outerMinUseDim和innerMinUseDim通过aicNum/maxConflictDim计算，当maxConflictDim >= aicNum时整数除法结果为0，后续用这两个值做除数触发除零异常
- 修复模式：std::max(aicNum / maxConflictDim, 1UL)确保最小值为1
- 可审查性：高
- 审查规则建议：整数除法结果用作后续除数时，必须用std::max(..., 1)保护

### 9716a600fadec6cc86efb9f7960a5bb1d1914f28 修复fused_linear_online_max_sum在910_93上偶现确定性计算精度的问题
- 根因类别：多核同步屏障缺失
- 涉及文件：matmul/fused_linear_online_max_sum/op_kernel/fused_linear_online_max_sum.h
- 缺陷描述：CVProcess()（向量后处理）之前缺少SyncAll<false>()全局同步，Matmul结果可能未全部写入L0C/GM就被CVProcess读取，在910_93多核场景下偶现精度错误（确定性计算要求所有核完成后再后处理）
- 修复模式：在CVProcess()前增加SyncAll<false>()全局屏障
- 可审查性：中
- 审查规则建议：Matmul→后处理(CVProcess/Fixpipe)之间必须有全局同步屏障，确保所有核的Cube计算完成

---

跳过的提交(第350-369批)：
- d7740e93: fix kongge(日志字符串尾部空格修复，非代码缺陷)
- cd942771: conv文档拼写错误修改(纯文档)
- 8ffc8129: 修复matmul目录下README低错问题(纯文档)
- 8cc6fd68: fix aclnnziliao(纯文档修复)

## 第370-391批分析

### 7ab2a2cd fix sub pkg compile error
- 根因类别：构建配置缺陷/平台支持遗漏
- 涉及文件：CMakeLists.txt, build.sh, cmake/gen_ops_info.cmake, 新增3个common头文件
- 缺陷描述：ASCEND_ALL_COMPUTE_UNIT列表缺少多个SoC型号(ascend031/035/310b/910_55/mc62cm12a)，导致A5等平台的算子无法编译。同时构建脚本中act_copy目标改名common_copy，增加公共头文件拷贝逻辑解耦编译依赖。
- 修复模式：配置补全+资源拷贝路径修正
- 可审查性：中
- 审查规则建议：新增SoC平台时，检查CMakeLists.txt、build.sh等多处SoC列表是否一致同步

### 5bd40ca6 addLayerNorm上边界问题修复
- 根因类别：整数溢出/类型截断缺陷
- 涉及文件：norm/add_layer_norm/op_host/add_layer_norm_tiling.cpp, add_layer_norm_tiling.h
- 缺陷描述：tiling计算核心变量(numRow/numCol/rowPerTime等)使用uint32_t/int32_t，tensor维度超2^31时溢出。中间计算x1Size=numRow*numCol在int32下易溢出。TILING_DATA_FIELD_DEF字段类型也需同步改int64_t，否则host侧正确但传device时截断。
- 修复模式：系统性类型提升uint32_t/int32_t→int64_t，消除static_cast截断
- 可审查性：高
- 审查规则建议：tiling代码中shape/size变量强制int64_t；检查所有static_cast到窄类型的转换

### 097251e0 bug fix
- 根因类别：多类缺陷(参数校验缺失+编译警告+符号可见性)
- 涉及文件：activation/hard_sigmoid, activation/softshrink_grad, quant/dequant_swiglu_quant
- 缺陷描述：(1)hardsigmoid缺少输入输出dtype/format一致性检查；(2)softshrink_backward对lambd参数过度约束类型检查；(3)dequant_swiglu_quant常量从头文件移到cpp解决链接可见性问题
- 修复模式：增强参数校验+移除过度约束+调整符号作用域
- 可审查性：高
- 审查规则建议：算子API层必须检查输入输出dtype/format一致性；头文件避免定义非inline全局常量(ODR violation)

### 8f0ebc3e 修复DequantSwigluQuant算子在group_index值存在零时出现的精度问题
- 根因类别：控制流逻辑缺陷/零值边界条件处理不当
- 涉及文件：quant/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_cut_group.h
- 缺陷描述：遍历group时realDimx_<=0直接break跳出整个循环，但group_index存在零值只表示空group不应终止遍历。修复将break移入groupIdx==cuGroupIdx条件内部，区分speGroupType场景。
- 修复模式：边界条件修正，"遇零即停"改为"遇零跳过继续"
- 可审查性：中
- 审查规则建议：循环中break语句应审查是否覆盖"部分数据为零/空"的边界case

### 7bc74e32 修复dynamic_quant算子kernel ut报错
- 根因类别：UT与生产代码不同步/结构体字段缺失
- 涉及文件：quant/dynamic_quant/tests/ut, quant/dynamic_quant_v2/tests/ut
- 缺陷描述：生产代码tiling结构体新增ubSize字段，UT侧结构体未同步更新，内存布局不一致导致UT读取错误tiling参数。
- 修复模式：UT结构体补上ubSize字段
- 可审查性：高
- 审查规则建议：tiling结构体应只定义一次由UT include复用，不应拷贝维护

### 1fb3ffe1 [DTS2025101528926] Conv2DV2 TF groups fix
- 根因类别：框架兼容性缺陷/groups参数推导逻辑缺失
- 涉及文件：conv/common/op_host/conv_forward_infershape.cpp
- 缺陷描述：TF框架Conv2DV2的groups默认1，但TF语义通过in_channels/kernel_channels隐式推导groups。原代码未处理groups=1但ic!=kc的隐式分组卷积场景，直接报错。
- 修复模式：增加条件分支在groups=1且ic%kc==0时自动推导groups值
- 可审查性：中
- 审查规则建议：多框架适配时关键属性应有明确的框架语义转换逻辑并加注释

### 0799defc Revert "change gather_v2 l0"
- 根因类别：接口变更回退
- 涉及文件：index/gather_v2/op_host/op_api/gather_v2.cpp
- 缺陷描述：之前的commit移除了gather_v2的isPreprocessed参数导致接口不兼容。Revert恢复该参数。
- 修复模式：完整Revert
- 可审查性：低
- 审查规则建议：L0 API参数变更应有严格兼容性审查；Revert message应明确回退原因

### 05108b11 BatchNormGradV3算子精度问题修改
- 根因类别：多核同步缺陷+参数传递错误+条件分支遗漏
- 涉及文件：norm/batch_norm_grad_v3/op_kernel/ (infer/train/common 3个头文件)
- 缺陷描述：(1)GetCrossCoreR1Param传入cIndex应为coreIndex，语义完全不同导致跨核参数计算错误；(2)ComputeCrossCoreDBias/DWeight缺少TPipeSetWaitFlag<HardEvent::MTE3_MTE2>()，MTE3写出和MTE2读入间无同步屏障，跨核拷贝读未写完数据；(3)wsReduceInputOffset计算遗漏moreMultiChannel条件；(4)blockIdx>needCoreNum比较方向应为<
- 修复模式：参数修正+硬件同步补全+条件分支补全
- 可审查性：中
- 审查规则建议：跨核同步函数参数应明确区分channel index和core index；跨核数据拷贝前必须设置硬件事件等待标志

### 91c29129 解决apiUT报错
- 根因类别：UT断言被批量注释(anti-pattern)
- 涉及文件：12个UT文件(add_rms_norm_quant_v2, batch_norm_grad_v3, batch_norm_v3等)
- 缺陷描述：大量EXPECT_EQ断言被注释掉，原验证TestGetWorkspaceSize返回值的断言被禁用。不是真正修复而是掩盖问题。
- 修复模式：注释掉失败断言(anti-pattern)
- 可审查性：高
- 审查规则建议：禁止直接注释EXPECT_*/ASSERT_*断言，需附带原因；可grep检测"// EXPECT_"模式

### 42bc1795 修复RNNV2 TILING ut
- 根因类别：API变更适配/UT编译错误
- 涉及文件：rnn/dynamic_rnnv2/tests/ut/
- 缺陷描述：UT使用旧ge::AnyValue::CreateFrom接口需迁移到Ops::NN::AnyValue::CreateFrom。删除依赖废弃ge::OpDescUtils的测试。
- 修复模式：命名空间迁移+删除废弃API依赖
- 可审查性：中
- 审查规则建议：检测ge::AnyValue、ge::OpDescUtils等已废弃API使用

### fd41c335 修复性能劣化问题
- 根因类别：优化路径边界条件缺失
- 涉及文件：matmul/batch_mat_mul_v3/op_host/op_api/, op_tiling/arch35/
- 缺陷描述：BatchMatMulToMul优化路径缺少N=1排除条件。N=1时matmul转element-wise mul反而性能劣化。修复在op_api层和tiling层同时增加nDim==1的返回false判断。
- 修复模式：增加边界条件短路返回阻止不适用场景进入优化路径
- 可审查性：高
- 审查规则建议：优化路径(算子替换/融合)需完整适用性条件检查；op_api层和tiling层相同逻辑需保持同步

### d0df69d1 celoss tiling bug fix
- 根因类别：UB内存计算不准确
- 涉及文件：cross_entropy_loss_tiling_arch35.cpp
- 缺陷描述：CrossEntropyLoss全载模板的tiling计算UB空间时未扣除ReduceSum临时buffer(maxValue)大小，导致实际运行时UB不足。修复通过GetReduceSumMaxMinTmpSize获取临时空间需求后扣除。
- 修复模式：精确计算UB资源用量，引入API获取中间buffer需求
- 可审查性：高
- 审查规则建议：tiling中UB空间分配必须包含所有中间临时buffer(reduce/cast等)；使用ReduceSum等API时应调用GetXxxTmpSize获取临时空间

### a2c4cbec fix rmsnormquant
- 根因类别：运行时类型判断vs编译期模板分支
- 涉及文件：norm/rms_norm_quant/op_kernel/rms_norm_quant.cpp
- 缺陷描述：判断输出是否INT4用运行时dstType==DT_INT4选分支，模板实例化时不匹配类型的分支仍被编译导致类型错误。修复增加yDtype模板参数，改用if constexpr(IsSameType<yDtype,int4b_t>::value)编译期消除分支。
- 修复模式：运行时if→if constexpr编译期分支消除
- 可审查性：高
- 审查规则建议：模板类中不同类型tensor操作应用if constexpr而非运行时if

### 6f830ee8 Revert "modify name of addRmsNormDynamicQuant and the binary patch"
- 根因类别：配置回退(rename导致binary配置不匹配)
- 涉及文件：add_rms_norm_quant_binary.json
- 缺陷描述：之前的rename操作导致算子二进制配置不匹配。Revert恢复正确的dtype组合和算子变体配置。
- 修复模式：git revert恢复配置
- 可审查性：低
- 审查规则建议：binary patch JSON修改应通过自动化测试验证配置与实际二进制一致性

### b63f9f0e fix ut
- 根因类别：UT基础设施与工程配置批量修复
- 涉及文件：classify_rule.yaml, 9个foreach算子UT, dynamic_rnnv2 UT, generate_cpp_cov.sh, test_op_host_main.cpp
- 缺陷描述：(1)foreach系列UT的include路径层级多一层(../../../../→../../../)；(2)classify_rule.yaml组件名ops_nn→ops-nn；(3)覆盖率脚本硬编码/home/jenkins/Ascend/→环境变量；(4)_exit(RUN_ALL_TESTS())→return，_exit跳过析构和覆盖率数据写入
- 修复模式：路径修正+环境适配
- 可审查性：中
- 审查规则建议：include路径层级应与目录结构匹配；脚本不应硬编码绝对路径；_exit()不应用于UT main

### aff96527 uniqueConsecutive算子ACLNN单输出场景修改
- 根因类别：API适用性缺陷(OP_OUTSHAPE不支持多输出)
- 涉及文件：index/unique_consecutive/op_host/op_api/unique_consecutive.cpp
- 缺陷描述：UniqueConsecutive有3个输出，但OP_OUTSHAPE在多输出场景不适用。修复对returnCounts==false时走单独注册路径只传第一个output shape。
- 修复模式：按条件分支规避框架限制
- 可审查性：中
- 审查规则建议：使用OP_OUTSHAPE时需确认输出个数满足适用条件

### b8815b6e stride code fix
- 根因类别：逻辑条件不完整
- 涉及文件：matmul/quant_matmul_v4/op_host/op_api/aclnn_quant_matmul_v4.cpp
- 缺陷描述：CheckSpecialCase只检查firstLastDim==secondLastDim但条件过宽，只有两维度都==1时才是特殊case。缺少==1约束导致不该跳过transpose设置的case被错误跳过。
- 修复模式：收紧条件增加GetDim(secondLastDim)==1约束
- 可审查性：高
- 审查规则建议：矩阵维度特判条件应明确约束具体值(如==1)，不应仅依赖维度间相等关系

### 3ed03487 fix rmsNormQuant
- 根因类别：自定义DataCopy替换为标准API
- 涉及文件：norm/rms_norm_quant/op_kernel/rms_norm_quant.cpp
- 缺陷描述：删除自定义DataCopyCustom模板函数(含复杂对齐处理+运行时类型判断)，替换为AscendC标准DataCopyPad API。原函数内部用运行时dstType==DT_INT4判断在模板实例化时可能编译错误。
- 修复模式：删除冗余手写逻辑，使用框架标准API
- 可审查性：高
- 审查规则建议：优先使用框架DataCopy/DataCopyPad API，避免自行实现对齐和分块拷贝

### 130060de 修改之前因处理告警产生的错误
- 根因类别：处理编译告警引入逻辑错误
- 涉及文件：norm/kv_rms_norm_rope_cache/op_host/kv_rms_norm_rope_cache_regbase_full_load_tiling.cpp
- 缺陷描述：三个broadcast flag常量值被错误统一为1。CONST_BRCFLAG_ZERO应=0、ONE=1、TWO=2，之前"处理告警"时全改成1导致broadcast标志位无法区分不同模式。
- 修复模式：恢复常量正确值
- 可审查性：高
- 审查规则建议：命名XXX_ZERO/ONE/TWO的常量值应与名称数字含义一致；处理编译告警不应修改常量实际值

---

跳过的提交(第370-391批)：
- 1a2b46a1: oplist整改；v-fusion目录修复(纯文档)
- 5cb632a9: fix-group_norm文档(纯文档)
- 973d8414: fix readme(纯文档)

### a77634169c fix_version_script
- 根因类别：脚本/配置错误
- 涉及文件：scripts/package/ops_nn/scripts/opp_install.sh, version.info
- 缺陷描述：(1) shell脚本中在非函数体内使用local关键字声明变量，某些shell环境下报错或行为未定义。(2) version.info依赖包版本号写死为"8.3"，缺少>=前缀，版本兼容性检查过严只允许精确匹配。
- 修复模式：删除local关键字；版本号加>=前缀
- 可审查性：高
- 审查规则建议：shell lint检查local关键字只在函数内使用；版本依赖配置应有schema校验支持范围语义

### 5f7c1ea4 foreach_minimum_list fix duplicate REG_OP
- 根因类别：复制粘贴错误(算子注册定义)
- 涉及文件：foreach/foreach_minimum_list/op_graph/foreach_minimum_list_proto.h
- 缺陷描述：foreach_minimum_list的算子原型错误复制了ForeachMinimumScalarList的REG_OP注册代码，算子名错误为MinimumScalarList，输入从双DYNAMIC_INPUT(x1,x2)变成DYNAMIC_INPUT(x)+INPUT(scalars)——List版本的双tensor list输入被错误定义为ScalarList版本的tensor+scalar输入。
- 修复模式：REG_OP名改为ForeachMinimumList，输入改为正确的双DYNAMIC_INPUT(x1, x2)
- 可审查性：高
- 审查规则建议：检查op_graph proto文件中REG_OP名称是否与文件名/目录名一致；检测同一仓库中是否存在重复的REG_OP注册名

### 225c12dd flatquant精度问题修复
- 根因类别：内存偏移计算缺少sizeof(T) + 核间同步缺失
- 涉及文件：quant/flat_quant/op_kernel/flat_quant_cube.h, flat_quant_vec.h, tensor_utils.h
- 缺陷描述：(1) workspace指针偏移计算时遗漏* sizeof(T)，导致doubleP1GM的GlobalBuffer设置在错误偏移位置(按元素数而非字节偏移)，与outnzGM区域重叠造成数据踩踏精度问题。(2) vec端缺少核间同步(缺TWO_VEC_SYNC_ID的CrossCoreSetFlag/WaitFlag)。
- 修复模式：偏移计算补乘sizeof(T)；增加核间同步屏障
- 可审查性：中
- 审查规则建议：所有GlobalBuffer偏移计算需审查是否混淆"元素数"和"字节数"；workspace划分区域时需以字节为单位统一计量

### bd7975a8 处理大Wi场景L0C搬出时读取数据溢出问题
- 根因类别：边界条件处理不完整(大shape下溢)
- 涉及文件：conv/conv3d_backprop_input_v2/op_kernel/arch35/.../conv_bp_sub_func_mix.h
- 缺陷描述：CalcCutInWIndex函数中计算Wi切分时，遗漏curWiPos > doubleBaseUseM的情况(大Wi场景)，headWi_被赋超大值，导致leftBaseUseM = doubleBaseUseM - headWi_下溢。
- 修复模式：增加curWiPos > doubleBaseUseM判断分支，headWi_置0
- 可审查性：中
- 审查规则建议：涉及数据搬运偏移计算的代码需覆盖边界case；减法结果可能为负时需检查保护

### 68500442 fix include
- 根因类别：头文件include路径错误
- 涉及文件：optim/apply_adam_w_v2/op_host/apply_adam_w_v2_tiling_arch35.h
- 缺陷描述：#include "elewise/elewise_tiling.h"路径缺少atvoss/前缀，应为"atvoss/elewise/elewise_tiling.h"，导致某些编译环境下找不到头文件。
- 修复模式：修正include路径
- 可审查性：高
- 审查规则建议：CI编译所有target确保include路径正确

### 3fc12f1c MatMul修复l1iterbatchAicError
- 根因类别：K维度对齐逻辑错误(A/B不必要分离引入bug)
- 涉及文件：matmul/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_iterbatch_basicapi_tiling.cpp/.h
- 缺陷描述：原代码将K对齐分为alignKaValue_(A矩阵)和alignKbValue_(B矩阵)，根据转置状态分别对齐。导致iterbatch场景下L1/L0A/L0B容量计算使用不同K对齐值，baseK取min(Ka,Kb)，某些shape组合下tiling结果不正确引发AIC error。K维度在MatMul中应统一。
- 修复模式：合并alignKaValue_/alignKbValue_为单一alignKValue_，统一对齐到BASIC_BLOCK_SIZE_16
- 可审查性：中
- 审查规则建议：tiling参数计算中同一维度不应有多个对齐变体；新增tiling路径需覆盖极端shape组合

### bb4b4662 fix compile (模板类型参数错误)
- 根因类别：复制粘贴错误(A/B类型混淆)
- 涉及文件：common/act/matmul/block/block_quant_with_tile_mmad_multi_block.h
- 缺陷描述：CopyCubeInB模板实例化中，MatmulInputBType<InputBType, typename InputAType::T>错误引用InputAType::T。B矩阵应使用InputBType::T。当A/B数据类型不同时(如A为fp16、B为int8量化场景)，导致编译错误或运行时类型不匹配。
- 修复模式：InputAType::T改为InputBType::T
- 可审查性：高
- 审查规则建议：模板类型参数中A/B应明确区分，review重点关注对称结构中的A/B互换错误

### c6e269cf matmulv3 infershape对齐修改+addMM示例缺陷修改
- 根因类别：(1)对齐公式off-by-one (2)示例代码内存泄漏
- 涉及文件：matmul/mat_mul_v3/op_host/mat_mul_v3_infershape.cpp, matmul/mat_mul_v3/examples/test_aclnn_addmm_aclnninplace_addmm.cpp
- 缺陷描述：(1) hidden_size对齐公式(*hidden_size + BLOCK_SIZE) / BLOCK_SIZE * BLOCK_SIZE多加1个BLOCK_SIZE，正确应为+BLOCK_SIZE-1。(2) 示例代码aclnnInplaceAddmm复用workspaceAddr变量做aclrtMalloc，前一次workspace地址被覆盖无法释放。
- 修复模式：(1)修正对齐公式；(2)使用独立变量并正确释放
- 可审查性：高
- 审查规则建议：向上对齐标准写法(x+align-1)/align*align应作为静态检查规则

### ba467b19 quantbatchmatmul 修复修改编译警告导致的不一致
- 根因类别：修复编译warning时引入逻辑变更
- 涉及文件：matmul/quant_batch_matmul_v4/op_host/op_api/aclnn_quant_matmul_v5.cpp
- 缺陷描述：修改编译警告时，CheckSpecialCase的条件判断被错误改写，误删了&& dim(secondLast) == 1条件，变成仅检查两个维度相等。导致非1x1情况(如2x2)也被判定为特殊case，跳过transpose属性设置引起计算错误。
- 修复模式：恢复被误删的== 1判断条件
- 可审查性：高
- 审查规则建议：修复编译warning时不应改变条件逻辑，review应对比修复前后语义等价性

### 5626757109 算子包脚本修复
- 根因类别：硬编码平台名导致多平台不兼容
- 涉及文件：scripts/package/ops_nn/scripts/ops_nn_custom_install.sh, ops_nn_custom_remove_softlink.sh
- 缺陷描述：安装/卸载脚本中将平台目录硬编码为ascend910b，仅支持910b一个平台。新增其他ascend平台时软链接不会创建/删除。
- 修复模式：硬编码改为PATTERN="ascend*"通配遍历所有ascend平台目录
- 可审查性：高
- 审查规则建议：安装/部署脚本中不应硬编码平台型号，应使用通配或配置驱动

### 8681104f fix tilingkey compile error
- 根因类别：条件编译宏未同步定义
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_apt.cpp
- 缺陷描述：A16W4(int4b_t)场景下，#ifdef块重定义了DTYPE_WEIGHT和S4宏，但遗漏重定义ORIG_DTYPE_WEIGHT为DT_INT4。后续代码依赖ORIG_DTYPE_WEIGHT进行条件编译分支选择，缺少此宏导致tilingkey对应kernel编译失败。
- 修复模式：增加#undef ORIG_DTYPE_WEIGHT + #define ORIG_DTYPE_WEIGHT DT_INT4
- 可审查性：高
- 审查规则建议：条件编译块中重定义类型相关宏时，需检查所有关联宏是否同步更新

### a0a2246c 修复gather_elements_v2 超INT_MAX shape的bug
- 根因类别：整数溢出(int32不足以表示大shape偏移)
- 涉及文件：index/gather_elements_v2/op_kernel/gather_elements_v2_last_dim.h, index/gather_v2/op_host/op_api/aclnn_gather_v3.cpp
- 缺陷描述：(1) kernel中xGmBaseOffset超INT_MAX时static_cast<int32_t>截断溢出，Adds指令索引计算错误。(2) gather_v3 op_api仅检查self tensor size阈值决定精度模式，遗漏index tensor size过小也需高精度模式。
- 修复模式：(1)增加INT_MAX边界检查分支，大偏移用逐元素加法；变量升级int64_t (2)增加index size阈值检查
- 可审查性：中
- 审查规则建议：kernel中涉及shape乘积/偏移计算的变量应默认使用int64_t

### 96f1a29b MaxPool3DWithArgmaxV2修复非常规数错判NAN值
- 根因类别：NaN判定算法错误(基于范围比较而非IEEE754标准)
- 涉及文件：pooling/adaptive_max_pool3d/op_kernel/adaptive_max_pool3d_big_pool.h, pooling/max_pool3d_with_argmax_v2/op_kernel/max_pool3d_with_argmax_big_kernel.h
- 缺陷描述：原NaN判定用数值范围比较(nan > INF)，对denormalized number(如1e-45)的二进制表示恰好落在NaN范围内，正常极小浮点数被误判为NaN导致索引错误。正确做法按IEEE754：指数位全1且尾数位非0才是NaN。
- 修复模式：改用位掩码检测：fp16检查(nan & 0x7C00)==0x7C00 && (nan & 0x03FF)!=0，fp32检查(nan & 0x7F800000)==0x7F800000 && (nan & 0x007FFFFF)!=0
- 可审查性：高
- 审查规则建议：NaN/Inf判定必须基于IEEE754位模式而非数值范围比较；此类工具函数应抽取为公共库

---

跳过的提交(第392-411批)：
- 19216d51: fix_ut_1107(纯UT/示例代码修复，数据类型int32->int64/删除死函数/修正CMake宏)
- ee1806a6: masked_softmax算子修正ut(纯UT整改，kernel侧仅头文件include和常量定义)
- 7e28bd1a + 2a728a65: fix foreach host ut + foreach op fix(互逆操作，净效果为零)
- 730babc2: 修复quant类算子kernel ut报错(纯UT修复)
- 92968c7a: fix norm kernel ut error(纯UT修复，38个文件全在tests/ut目录)
- 02bd44c9: gelu_quant等warning修复(纯printf格式符修正+删除未使用变量，无逻辑变更)

---

## 第412-431批 (20条)

### 2c18ae1e 修改SwiGluQuant大shape精度问题
- 根因类别：数据类型溢出(参数类型截断)
- 涉及文件：quant/swi_glu_quant/op_kernel/swi_glu_quant.h, swi_glu_quant_static.h
- 缺陷描述：ProcessCoreMultiUbMultiAlign函数的offsetRow参数声明为uint16_t(最大65535)，但调用处传入uint32_t。大shape时行偏移超过65535发生截断，导致访问错误地址产生精度问题。
- 修复模式：参数类型从uint16_t提升为uint32_t
- 可审查性：高
- 审查规则建议：检测函数参数窄类型接收宽类型的隐式截断(uint16_t <- uint32_t)，-Wconversion可捕获

### 1a684711 修复wqbmmv2 310P kernel入口条件判断
- 根因类别：条件分支逻辑错误(过度约束)
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2.cpp
- 缺陷描述：310P平台kernel入口if constexpr条件中包含多余的HAS_ANTIQUANT_OFFSET == 0约束，但kernel内部已通过模板参数处理两种场景。多余条件导致带antiquant offset时无法进入310P分支。对比相邻BATCH_ONE分支无此约束可佐证。
- 修复模式：删除if constexpr中多余的HAS_ANTIQUANT_OFFSET == 0条件
- 可审查性：中
- 审查规则建议：同一函数内多个if constexpr分支条件应对称检查；当模板参数已在宏调用中传入时，入口条件不应对该参数做硬编码值约束

### a984e3f8 fix group_norm_silu ut
- 跳过(纯UT修复)

### a6686eda 修复FlatQuant功能报错
- 根因类别：workspace内存布局错误(buffer地址计算)
- 涉及文件：quant/flat_quant/op_kernel/flat_quant_high.h
- 缺陷描述：workspace被x1GM(类型T)和x2GM(类型float)共享。原代码x1GM在前x2GM在后，但x2GM需要更大空间(K_DOUBLE_VEC vs K_PER_VEC)，导致两者内存区域重叠。同时偏移量sizeof转换方向错误。
- 修复模式：交换两个buffer在workspace中的分配顺序，修正偏移量计算的sizeof方向
- 可审查性：中
- 审查规则建议：多buffer共享workspace时，审查每个buffer的offset + size不得超过下一个区域的offset；关注sizeof类型转换方向

### 8ee73052 修复infershape中先使用后校验nullptr的问题
- 根因类别：空指针解引用(use-before-null-check)
- 涉及文件：matmul/common/op_host/matmul_common_infershape.cpp
- 缺陷描述：先对shape_x1/shape_x2执行解引用(CheckIsUnknownDimNum(*shape_x1))，然后才检查nullptr。CWE-476典型模式。
- 修复模式：将nullptr校验提前到解引用之前；attrs的null check单独拆分到实际使用前
- 可审查性：高
- 审查规则建议：静态分析规则use-before-null-check，Coverity/Clang SA可直接检出

### 4203e836 celoss ut fix
- 跳过(纯UT修复)

### 11bd363e quantbatchmatmul 修复编译警告
- 根因类别：编译警告 + 隐藏逻辑缺陷
- 涉及文件：matmul/quant_batch_matmul_v3/..., matmul/quant_batch_matmul_v4/...多个文件
- 缺陷描述：(1) printf格式说明符不匹配(%ld打印uint64_t应为%lu)；(2) 未使用函数参数；(3) 实际逻辑缺陷：CheckSpecialCase只检查firstLastDim == secondLastDim就跳过transpose设置，缺少&& secondLastDim == 1约束，导致两维度相等但非1时错误跳过transpose。
- 修复模式：格式符修正 + 未使用参数处理 + 条件收紧增加==1校验
- 可审查性：中
- 审查规则建议：条件判断只比较"相等"时需审视是否还需约束具体值；逻辑修复不应混在warning修复提交中

### 0d4c358c fix compile warnings [dynamic_quant_update_scatter_v2]
- 根因类别：编译警告(printf格式说明符)
- 涉及文件：quant/dynamic_quant_update_scatter_v2/op_host/dynamic_quant_update_scatter_v2_tiling.cpp
- 缺陷描述：%d打印需要%ld的类型。修复用(long)强转+%ld，不如PRId64规范。
- 修复模式：格式说明符修正 + (long)强制类型转换
- 可审查性：高
- 审查规则建议：日志打印应使用inttypes.h的PRId64/PRIu64宏

### cf6e64b9 调整1952 bias非全载条件、新增scale非全载拦截，修复extend conv transpose fp16精度问题
- 根因类别：L1内存分配未预留硬件buffer空间 + 全载条件缺失 + scale尺寸校验缺失
- 涉及文件：conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/多个tiling文件, common/conv_backprop_input_context_utils.*
- 缺陷描述：(1) L1容量校验在IsSocVersionFuse下未扣除bias buffer(4KB)和scale buffer(6KB)预留空间，导致tiling参数偏大、数据越界覆盖引发fp16精度问题；(2) 缺少isBiasFullLoad标志，bias超大仍按全载处理导致截断；(3) INT8下缺少scale尺寸上限校验。
- 修复模式：L1容量校验扣除预留空间 + 新增bias全载判断 + 新增scale尺寸拦截
- 可审查性：低
- 审查规则建议：L1/UB容量校验必须考虑所有硬件预留buffer(特别是多SoC版本适配时)；"是否全载"判断必须基于实际容量与数据量比较

### 58fccb11 修复dynamic_quant_update_scatter
- 跳过(纯UT修复)

### 7e5f88dc issue 修复
- 根因类别：脚本健壮性缺陷
- 涉及文件：scripts/kernel/binary_script/build_binary_single_op_gen_task.sh, gen_output_json.py
- 缺陷描述：(1) shell脚本中dos2unix命令未检查文件存在性、未检查工具可用性、未检查CRLF格式；(2) Python脚本编译失败时错误信息不充分缺排查指引。
- 修复模式：前置条件校验(文件存在→CRLF检测→工具可用) + 错误信息增强
- 可审查性：高
- 审查规则建议：shell脚本中外部命令调用前需command -v可用性检查；文件操作前需-f存在性判断

### ecf7f382 [convbp] fix bug for transpose padding<0
- 根因类别：输入校验缺失(负值边界)
- 涉及文件：conv/conv3d_backprop_input_v2/op_host/op_tiling/common/conv_backprop_input_context_utils.cpp
- 缺陷描述：CheckTranspose函数处理output_padding时只检查全零判断transpose模式，未检查负数值。负数padding在Arch35+上导致未定义行为。
- 修复模式：新增outputPaddingAllNonNegative标志位，负值时报错返回
- 可审查性：高
- 审查规则建议：卷积算子的padding/stride/dilation等整型参数需检查负值或越界校验

### 46ba9c58 修改问题单
- 跳过(纯文档，仅补充API注释)

### 40759d9b Fix Dw Problem
- 根因类别：多重缺陷(tiling结构体bool类型 + 返回值未检查 + 函数定义顺序)
- 涉及文件：conv/conv3d_backprop_filter_v2/op_host/op_tiling/..., op_kernel/..., tests/ut/...
- 缺陷描述：(1) tiling数据结构中isSplitKernelHW字段类型为bool，host-device间按字节传输时bool大小依赖编译器实现可能不一致，改为int32_t并调整字段排列顺序；(2) SetStridesAttr/SetDilationsAttr返回值被忽略，非法参数继续传递；(3) kernel中模板函数定义在调用方之后导致编译告警。
- 修复模式：bool→int32_t类型修正 + OP_CHECK_IF包装返回值检查 + 函数定义顺序调整
- 可审查性：中
- 审查规则建议：tiling数据结构禁用bool(应使用固定宽度整数)；属性设置函数返回值必须检查

### 8c98bd58 FlatQuant修复大shape偏移溢出，高阶matmul模板修复精度问题
- 根因类别：多重缺陷(int32溢出 + workspace寻址错误 + 变量名笔误 + 尾块搬运缺失)
- 涉及文件：quant/flat_quant/op_host/flat_quant_tiling.cpp, op_kernel/flat_quant_cube.h, flat_quant_high.h, flat_quant_vec.h, tensor_utils.h
- 缺陷描述：(1) FlatQuantShapeInfo所有字段为int32_t，k*M*N偏移计算溢出int32范围，全部改为int64_t；(2) workspace寻址用k*shape.Mceil*shape.N导致不同core写入同一区域，改为GetBlockIdx()*K_PER_VEC + k%K_DOUBLE_VEC做per-core隔离；(3) Nceil计算错误用M而非N：shape.Nceil = (shape.M + CEIL_SIZE - 1) / CEIL_SIZE * CEIL_SIZE；(4) 新增尾块慢路径搬运(invalidK判断)避免读越界。
- 修复模式：int32→int64 + per-core workspace寻址 + M→N笔误修正 + 尾块搬运路径
- 可审查性：低
- 审查规则建议：shape维度相乘偏移用int64_t；workspace寻址必须含blockIdx隔离；ceil变量名须与目标维度一致

### 6b39aa33 fix SMS ut;add SMS/SMSG example
- 根因类别：Tensor尺寸信息缺失 + Tiling字段遗漏
- 涉及文件：vfusion/scaled_masked_softmax_v2/op_kernel/scaled_masked_softmax_v2.h(生产代码), tests/ut/...(UT)
- 缺陷描述：生产代码中LocalTensor使用前未调用SetSize()设置有效数据长度，后续Softmax等shape-aware API可能读未初始化区域。UT的tiling解析宏遗漏width字段。
- 修复模式：在使用Tensor前补充SetSize()调用；tiling宏补充width字段
- 可审查性：中
- 审查规则建议：AllocTensor和shape-dependent API之间必须存在SetSize调用；tiling解析宏应与结构体定义做字段一致性比对

### 395e3673 fixbug: WeightQuantBatchMatmulV2 vf中u16和u32比较会出现循环异常
- 根因类别：隐式类型提升导致硬件行为异常
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_vf.h
- 缺陷描述：vf kernel中for循环迭代变量为uint16_t，上界为uint32_t。C++隐式将uint16_t提升为uint32_t在CPU上安全，但vf硬件对u16 vs u32比较指令有偶发异常，导致循环次数不正确产生精度错误。
- 修复模式：在5个for循环上界表达式处显式static_cast<uint16_t>()
- 可审查性：高
- 审查规则建议：vf kernel代码中循环变量和上界必须类型一致；正则for\s*\(\s*uint16_t.*<\s*p\.可扫描疑似位点

### c760190c fix addrmsnorm splitd bug
- 根因类别：控制流逻辑错误(条件分支外的累加器更新)
- 涉及文件：norm/add_rms_norm/op_kernel/arch35/add_rms_norm_regbase_split_d.h
- 缺陷描述：ComputeFormer函数stage2的两个分支(tail <= ubFactor/2 和 tail > ubFactor/2)执行ComputeSum后，level1 += 1和ComputeMultiLevelReduce放在分支外面。当tail == 0时两个分支都不进入但仍执行level递增和多余reduce，影响RMSNorm精度。
- 修复模式：将level1递增和reduce调用移入各自if/else if分支内部
- 可审查性：高
- 审查规则建议：多阶段累加/规约模式中，level递增和reduce必须与ComputeSum严格配对在同一条件分支内

### 794c8451 DynamicBlockQuant算子UT修复
- 跳过(纯UT/文档修复)

### 77556df0 fix msda msdag ut
- 跳过(纯UT修复)

---

跳过的提交(第412-431批)：
- a984e3f8: fix group_norm_silu ut(纯UT修复)
- 4203e836: celoss ut fix(纯UT修复)
- 58fccb11: 修复dynamic_quant_update_scatter(纯UT修复)
- 46ba9c58: 修改问题单(纯文档，仅补充API注释)
- 794c8451: DynamicBlockQuant算子UT修复(纯UT/文档)
- 77556df0: fix msda msdag ut(纯UT修复)

## 第23轮分析 (defect_commits.txt 第432-451行)

跳过的提交：
- 1a33848b: fix scatter_elements_v2 原型注释(纯注释修改)
- 6b59bbdb: 修改大模型排查问题(纯文档修复，104个README/docs文件)
- 5334965f: swi_glu_quant kernal ut error修复(纯UT修复)
- f8561b1e: fix test_geir_kv_rms_norm_rope_cache(纯测试用例重写)
- 168d3bcd: fix tbmm mmad compile warning(重构清理，用官方API替换自定义封装)
- 48d33953: fix lisence(纯license声明修复)
- af539669: fix AdaptiveMaxPool3d\Grad example(纯示例代码shape/data修正)

### bf173212 修复kernel文件有后缀的情况下编译无法打包.o和json文件的问题
- 根因类别：构建脚本文件名处理逻辑缺陷
- 涉及文件：cmake/gen_ops_info.cmake, scripts/kernel/binary_script/build_binary_single_op_gen_task.sh
- 缺陷描述：当kernel Python文件带有后缀(如_apt、_910b)时，构建脚本在两个子函数中各自对op_file_name做后缀剥离，但仅修改局部变量不影响调用方后续构造.o和.json输出路径。CMake侧用精确文件名匹配install(FILES ...json)，当实际json文件名带额外后缀时匹配失败。
- 修复模式：(1)shell脚本将后缀剥离提取到main函数统一处理；(2)CMake改用file(GLOB *.json)+install通配符匹配
- 可审查性：中
- 审查规则建议：同一变量转换逻辑在多个函数中重复时应提取到公共调用点；CMake install(FILES)用硬编码文件名时检查是否有变体未覆盖

### 83f360f0 docxfix (原型定义部分)
- 根因类别：类型声明错误(复制粘贴)
- 涉及文件：matmul/batch_mat_mul_v3/op_graph/batch_mat_mul_v3_proto.h
- 缺陷描述：BatchMatMulV3算子bias输入的TensorType声明中，将DT_BF16误写为DT_FLOAT，导致{DT_FLOAT16, DT_FLOAT, DT_FLOAT}出现重复DT_FLOAT而缺少DT_BF16支持。x1和x2都声明了DT_BF16但bias遗漏。
- 修复模式：将重复的DT_FLOAT替换为DT_BF16
- 可审查性：高
- 审查规则建议：检测同一算子注册宏中TensorType列表内重复枚举值；当输入tensor声明了BF16但关联参数未声明时警告

### 09c38a47 fix eigen compilation using cuda keyword
- 根因类别：第三方库构建配置缺失
- 涉及文件：cmake/third_party/eigen.cmake
- 缺陷描述：ExternalProject_Add引入eigen时未设置CONFIGURE_COMMAND ""和BUILD_COMMAND ""，导致CMake用默认行为编译eigen，eigen源码含CUDA关键字在非CUDA环境下编译失败。实际只需下载头文件(header-only库)。
- 修复模式：添加CONFIGURE_COMMAND ""和BUILD_COMMAND ""跳过配置和构建
- 可审查性：高
- 审查规则建议：header-only第三方库的ExternalProject_Add应同时设置CONFIGURE/BUILD/INSTALL_COMMAND为空

### c26eedd7 fix workspace
- 根因类别：workspace大小计算遗漏
- 涉及文件：norm/add_layer_norm_quant/op_host/add_layer_norm_quant_tiling_arch35.cpp
- 缺陷描述：workspace大小计算两个问题：(1)初始值设为1字节过小，应为32字节DEFAULT_WORKSPACE_SIZE；(2)当rowsPerCore_==1&&isDynamicQuant_且非WELFORD策略时，缺少sysWorkspaceSize_累加，导致workspace分配不足可能内存越界。
- 修复模式：补全条件分支中遗漏的workspace累加逻辑
- 可审查性：中
- 审查规则建议：workspace/buffer大小计算时检查所有条件分支是否完整覆盖所有内存组件

### d97951a7 拦截GeluQuant空tensor输入问题
- 根因类别：空tensor输入校验缺失
- 涉及文件：activation/gelu_quant/op_host/gelu_quant_tiling_arch35.cpp
- 缺陷描述：GeluQuant算子tiling阶段未校验输入tensor各维度是否为0，传入空tensor时后续tiling计算(对齐、除法)产生未定义行为。
- 修复模式：在tiling入口遍历输入shape各维度，发现0维度时提前返回GRAPH_FAILED
- 可审查性：高
- 审查规则建议：所有算子tiling入口应检查输入shape是否包含0维度

### 941e12f0 fix qbmm
- 根因类别：平台硬件参数被上层错误修改 + 基类方法非virtual
- 涉及文件：matmul/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling_base.h, matmul/quant_batch_matmul_v4/op_host/op_tiling/quant_batch_matmul_v4_pergroup_tiling.cpp/.h
- 缺陷描述：两层缺陷：(1)父类SetPlatformInfoForTiling()未声明virtual，子类无法override定制；(2)B4平台L2 cache大小被上层从168MB错误改为96MB，tiling基于错误L2容量计算。
- 修复模式：父类方法改为virtual支持override；子类override中硬编码修正L2大小(96->168MB)
- 可审查性：中
- 审查规则建议：基类中被子类可能需定制的方法应声明virtual；硬编码平台参数修正是临时workaround应跟踪上游修复

### 5cf2e333 解决geluQuant输入为scalar时报错问题
- 根因类别：边界条件缺失(scalar/0维tensor)
- 涉及文件：activation/gelu_quant/op_host/gelu_quant_tiling_arch35.cpp/.h
- 缺陷描述：geluQuant输入tensor为scalar(0维)时，直接对shape调用GetDimNum()/GetDim(0)/GetShapeSize()，scalar语义与普通tensor不同导致tiling计算异常。
- 修复模式：引入EnsureNotScalar()方法将scalar shape映射为{1}，统一处理
- 可审查性：中
- 审查规则建议：对tensor shape调用GetDimNum()/GetDim()时检查是否处理了scalar(0维)情况

### f0e7c19b bugfix_ctcloss
- 根因类别：条件分支优先级错误
- 涉及文件：loss/ctc_loss_v2/op_host/op_api/ctc_loss_v2.cpp
- 缺陷描述：CtcLossV2中V3 AiCore路径判断被放在V2之前，当输入同时满足V2和V3条件时V3优先匹配，导致本应走V2的case错误走V3。V2条件更严格应优先判断。
- 修复模式：交换if-else分支顺序，将更特化的V2判断放前面
- 可审查性：低
- 审查规则建议：多个互斥IsXxxSupport判断链中，更严格/特化的条件应排在前面

### eb5334f2 GeGluGradV2 kernel generate failed. code modify
- 根因类别：constexpr与预编译机制不兼容
- 涉及文件：activation/ge_glu_grad_v2/op_kernel/ge_glu_grad_v2_apt.cpp
- 缺陷描述：kernel中tiling key值声明为constexpr int32_t，在预编译(codegen)场景中constexpr变量无法被编译器作为宏展开进行模板/分支消除，导致kernel编译失败。
- 修复模式：将18个constexpr改为#define宏定义
- 可审查性：高
- 审查规则建议：kernel代码(op_kernel目录)中用于tiling key的常量必须用#define而非constexpr/const

### a0ccff51 [conbp]fix bug of batchdout
- 根因类别：计算逻辑错误(切分粒度公式方向反)
- 涉及文件：conv/conv3d_backprop_filter_v2/op_kernel/arch35/conv3d_backprop_filter_v2/conv3d_dw_v2_basic_block.h
- 缺陷描述：batchdoutPerBlock_计算公式为Ceil(totalAddSize, THRESHOLD)混淆了"切分块数"和"每块大小"的语义。正确应为Ceil(THRESHOLD, ho*wo)计算每块处理量，然后按偏移步进遍历。
- 修复模式：修正切分粒度计算方向+循环改为按偏移步进+修正尾块大小计算
- 可审查性：中
- 审查规则建议：Ceil(A,B)切分计算需确认A和B语义——"总量/块大小=块数"还是"阈值/单元大小=每块处理量"

### 3715ff61 fix celossgrad
- 根因类别：流水线同步屏障遗漏
- 涉及文件：loss/cross_entropy_loss_grad/op_kernel/arch35/cross_entropy_loss_grad_base.h
- 缺陷描述：WeightReduceSumTailN中Vector计算(Adds)完成后直接调用DataCopyPad将Local搬到GM，缺少PipeV2M()同步屏障。Vector结果尚未写入Local buffer时MTE3就开始搬运，非全载mean场景偶现精度错误。
- 修复模式：在DataCopyPad前插入PipeV2M()
- 可审查性：中
- 审查规则建议：DataCopyPad/DataCopy(Local->GM)前必须有PipeV2M()/PipeS2M()同步

### e44ac0fd fix aicpu example code issue
- 根因类别：变量名错误 + 循环变量类型不匹配
- 涉及文件：examples/add_example_aicpu/op_kernel_aicpu/add_example_aicpu.cpp
- 缺陷描述：(1)日志打印"Num of elements"但传入data_size而非num_elements；(2)循环变量int i但边界num_elements是int64_t，超INT_MAX时截断。
- 修复模式：(1)日志参数改为num_elements；(2)循环变量改为int64_t
- 可审查性：高
- 审查规则建议：printf日志参数与格式串语义匹配；循环边界为int64_t时循环变量也应为int64_t

### bf34b10d fix error code
- 根因类别：缺少return语句(错误路径未返回) + 校验逻辑位置错误
- 涉及文件：matmul/mat_mul_v3/op_host/op_api/aclnn_addmm.cpp, matmul/mat_mul_v3/op_host/op_api/aclnn_matmul.cpp
- 缺陷描述：(1)CheckMatmulWeightNz中检测dtype非法并打印错误日志后没有return false，错误路径fall through；(2)dtype校验放在Shape校验函数中违反职责分离，且CheckDtypeValid末尾直接return true跳过检查。
- 修复模式：(1)补充return false；(2)将dtype校验从Shape函数移到独立的CheckWeightNzDtype函数
- 可审查性：高
- 审查规则建议：if分支中有错误日志(OP_LOGE)但没有return的代码路径应发出警告

### 9bddf3ad Fix aclnnQuantConvolution Conv3DV2 tiling kAL1 upper bound
- 根因类别：算法逻辑错误 -- 上界计算公式有误
- 涉及文件：conv/conv3d_v2/op_host/op_tiling/conv3d_api_tiling_algorithm.cpp
- 缺陷描述：kAL1上界的计算公式为`(POSTK_LIMIT + k0) / ci0HkWk`，但正确语义应是"kAL1乘以ci0HkWk后不能超过POSTK_LIMIT(65535)"。原公式多加了一个k0，导致上界偏大，选出的kAL1值在乘回ci0HkWk后可能超过load3d指令的postk硬件限制65535，引发精度或功能问题。
- 修复模式：将公式从`(POSTK_LIMIT + k0) / ci0HkWk`修正为`POSTK_LIMIT / ci0HkWk`
- 可审查性：中
- 审查规则建议：涉及硬件指令限制常量的边界计算，应验证"计算结果 x 乘回因子 <= 限制值"的不变量是否成立

### 9a18f2ca ge_glu_v2算子kernal ut修复error
- 根因类别：UT测试用例参数与算子实现不匹配
- 涉及文件：activation/ge_glu_v2/tests/ut/op_kernel/test_ge_glu_v2.cpp
- 缺陷描述：多个测试用例存在多处问题：(1)tiling参数与kernel逻辑不一致(loopNum/nLastTailGroup值颠倒，blockSize/ny值错误)；(2)输出buffer大小未考虑GLU算子输出维度减半；(3)workspace分配过小；(4)缺少SetKernelMode(AIV_MODE)调用；(5)blockDim=48与realCoreNum=40不一致
- 修复模式：修正所有tiling参数、输出buffer大小、workspace大小，统一blockDim=40，补充SetKernelMode调用
- 可审查性：中
- 审查规则建议：UT中的tiling参数应由tiling函数自动生成而非手工硬编码

### 4805e45e 修复aclnn conv日志打印有歧义
- 根因类别：日志消息不准确/有歧义
- 涉及文件：conv/convolution_forward/op_host/op_api/aclnn_convolution.cpp
- 缺陷描述：输入shape经过pad和dilation后小于0时，原错误日志仅打印"expect input shape[i] >= 0"，没有说明是输入尺寸小于kernel尺寸的根本原因，日志表述有歧义
- 修复模式：将日志改为更详细的描述包含完整计算公式和维度索引
- 可审查性：高
- 审查规则建议：错误日志应包含完整的计算公式和相关变量值；OP_LOGE的format string参数个数与实际参数应匹配

### 3aabd2e3 fixbug: 使用未初始化的值导致精度错误的问题
- 根因类别：使用未初始化变量 + 数据类型与硬件不兼容
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_reg_base_common.h, matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_vf.h
- 缺陷描述：(1)条件判断`if (tiling_->kBubSize >= params.groupSize)`中params.groupSize尚未初始化(在if体内才被赋值)，导致条件结果不确定引发精度错误；(2)vf硬件不支持uint32_t类型的循环变量
- 修复模式：(1)将条件中的params.groupSize改为已有值tiling_->groupSize；(2)将循环变量从uint32_t改为uint16_t适配vf硬件
- 可审查性：高
- 审查规则建议：静态分析可检测use-before-def模式；对vf硬件代码应建立禁用类型清单(如uint32_t)

### 9609a1a4 fix argument gmaddr problem
- 根因类别：结构体初始化列表缺少成员导致位置偏移
- 涉及文件：matmul/mat_mul_v3/op_kernel/arch35/mat_mul_streamk_basic_act.h
- 缺陷描述：Params结构体gm addr初始化列表`{aGM, bGM, cGM, biasGM, workspaceGM}`缺少一个字段，导致workspaceGM被赋值到错误的结构体字段位置
- 修复模式：在biasGM和workspaceGM之间插入nullptr，修正为`{aGM, bGM, cGM, biasGM, nullptr, workspaceGM}`
- 可审查性：高
- 审查规则建议：C++聚合初始化应使用designated initializers或注释标明字段名；初始化列表元素数量应与结构体字段数量匹配；开启-Wmissing-field-initializers

### cc407a02 fix gather_v2 oom
- 根因类别：内存阈值常量设置过小导致OOM
- 涉及文件：index/gather_v2/op_host/op_api/aclnn_gather_v3.cpp
- 缺陷描述：MIN_SELF_SIZE常量61440过小，导致某些场景workspace不足触发OOM
- 修复模式：将MIN_SELF_SIZE从61440调大到102400
- 可审查性：低
- 审查规则建议：魔数常量应配合注释说明推导过程和约束条件

### a4e4225f 修复头文件找不到
- 根因类别：include路径错误 -- 目录结构变更后未同步更新
- 涉及文件：activation/ge_glu_grad_v2/examples/*.cpp, activation/ge_glu_v2/examples/*.cpp (4个文件)
- 缺陷描述：include路径使用了aclnnop/level2/aclnn_geglu*.h，但头文件已不在level2子目录下，导致编译找不到头文件
- 修复模式：将include路径去掉中间的level2/
- 可审查性：高
- 审查规则建议：目录结构重构时应全仓搜索受影响的include路径；CI编译应覆盖examples目录

### 7dfb2a69 上边界tensor int类型越界修改
- 根因类别：整数类型溢出 -- int存储shape乘积
- 涉及文件：index/index_fill_d/op_host/op_api/aclnn_index_fill_tensor.cpp
- 缺陷描述：GenerateAssistMatrix函数中blocksize、blocknum、n三个变量为int(32位)，通过shape各维度乘法计算总元素数，超大shape时溢出
- 修复模式：将blocksize、blocknum、n和循环变量i从int改为int64_t
- 可审查性：高
- 审查规则建议：涉及tensor shape维度乘积的变量一律使用int64_t

### 0866cced [conbp]fix bug for streamkType=0
- 根因类别：控制流逻辑错误 -- 设置streamkType=0后未阻止后续streamk计算
- 涉及文件：conv/conv3d_backprop_filter_v2/op_host/op_tiling/arch35/conv3d_backprop_filter_v2_stream_k_tiling.cpp, conv/conv3d_backprop_filter_v2/tests/ut/op_host/test_conv3d_backprop_filter_v2_arch35_tiling.cpp
- 缺陷描述：设置streamkType=NO_STREAMK_CALC后没有return或else分支，代码继续执行streamk分核逻辑；同时缺少batchDout=1或howo=1(分核数为1)的退化处理
- 修复模式：将streamk分核逻辑放入else分支；新增UT覆盖退化场景
- 可审查性：中
- 审查规则建议：设置"跳过"标志后应确保后续逻辑不被意外执行；对分核数为1的退化场景应有显式处理

### fe0c08a2 fix ut fail in arch64 env
- 根因类别：构建配置问题 -- ARM环境下RPATH导致加载桩函数而非真实实现
- 涉及文件：tests/ut/op_host/CMakeLists.txt
- 缺陷描述：UT在aarch64环境下因RPATH优先从stub加载so而非真实实现，导致MM和Conv UT失败
- 修复模式：设置SKIP_BUILD_RPATH TRUE跳过构建时RPATH
- 可审查性：低
- 审查规则建议：跨平台UT应在x86和ARM环境下同时CI验证

### f5121124 fix bmmv3 weightnz example bug
- 根因类别：API调用错误 -- 使用旧版API缺少参数
- 涉及文件：matmul/batch_mat_mul_v3/examples/test_aclnn_batchmatmul_weight_nz.cpp
- 缺陷描述：示例代码调用旧版API aclnnCalculateMatmulWeightSize，缺少aclDataType参数，导致weight NZ格式的size计算不正确
- 修复模式：替换为V2版本aclnnCalculateMatmulWeightSizeV2并传入ACL_FLOAT16
- 可审查性：中
- 审查规则建议：对已废弃API建立deprecated列表；示例代码也需纳入CI编译验证

### b5127a2d aclnngather fix bug
- 根因类别：整数溢出 -- int存储shape维度乘积
- 涉及文件：index/gather_elements_v2/op_host/gather_elements_v2_last_dim_tiling.h
- 缺陷描述：MergeAxis函数中累乘shape各维度的变量val声明为int(32位)，多维度乘积超过INT32_MAX时溢出
- 修复模式：将int val = 1改为int64_t val = 1
- 可审查性：高
- 审查规则建议：涉及shape维度乘积的变量必须使用int64_t

### f678bf6e 解决n轴取错的问题
- 根因类别：复制粘贴错误 -- n轴和k轴维度索引条件表达式相同
- 涉及文件：matmul/common/op_host/matmul_common_infershape.cpp
- 缺陷描述：UpdateX2NewShape中k_x2_dim和n_x2_dim的条件表达式完全相同，都写成trans_x2 ? (x2_dim_num - 1UL) : (x2_dim_num - 2UL)，但n轴和k轴在转置/非转置下索引应互反
- 修复模式：将n_x2_dim的条件表达式修正为trans_x2 ? (x2_dim_num - 2UL) : (x2_dim_num - 1UL)
- 可审查性：高
- 审查规则建议：相邻两行代码结构相同但语义应不同时必须逐字比对；可检测"连续赋值语句右侧表达式完全相同"

### ee685a90 修复conv infershape ut中context复用问题
- 根因类别：测试代码中context对象复用导致状态污染
- 涉及文件：conv/common/tests/ut/op_host/test_quant_conv3d_dyn_infershape.cpp
- 缺陷描述：UT中InferShape和InferDataType使用同一holder对象获取context，InferShape执行后修改holder内部状态导致InferDataType的context被污染
- 修复模式：为每个InferDataType调用单独创建InferDataTypeContextFaker
- 可审查性：中
- 审查规则建议：UT中不同阶段的推理应使用独立context

### e4c9639a [qbmmv4][a4w4pergroup] fix mem illegal read
- 根因类别：越界内存读取 -- CeilAlign导致DMA读取超出实际数据范围 + 参数传递遗漏
- 涉及文件：matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4_pergroup.h
- 缺陷描述：(1)DataCopyPad的blk_len参数使用CeilAlign后的长度，在M/N非对齐整数倍时DMA越界读取；(2)CopyInX1INT8的blockCount硬编码ubCalcM_而非当前块实际值curAivM
- 修复模式：(1)去掉CeilAlign直接使用实际大小；(2)新增curAivM参数替代ubCalcM_
- 可审查性：中
- 审查规则建议：DMA拷贝长度不应使用CeilAlign值(CeilAlign适用于buffer分配非实际拷贝)；函数中用类成员变量替代应传入的局部变量时需警惕尾块场景

### e308f13d fix compile warning
- 根因类别：无符号类型与有符号常量比较 / printf格式符不匹配
- 涉及文件：norm/group_norm_silu/op_host/group_norm_silu_tiling.cpp, norm/group_norm_silu/op_host/group_norm_silu_tiling_arch35.cpp, norm/group_norm_swish_grad/op_host/group_norm_swish_grad_tiling.cpp
- 缺陷描述：(1)uint64_t类型channel/gammaDtypeSize与<0比较恒false无意义；(2)OP_LOGE用%ld打印uint64_t应用%lu
- 修复模式：删除无意义的<0检查；将%ld改为%lu
- 可审查性：高
- 审查规则建议：开启-Wsign-compare和-Wformat并treat as error

### dd6cddd1 修复avgPool3dGrad超大shape报错
- 根因类别：整数溢出 -- int32不足以表示超大shape
- 涉及文件：pooling/avg_pool3_d_grad/op_host/avg_pool_3d_grad_tiling.cpp, pooling/avg_pool3_d_grad/op_kernel/avg_pool3_d_grad_base.h等5个文件
- 缺陷描述：DHWParam结构体的n/d/h/w字段为int(32位)，超大shape时溢出；min/max函数参数、循环变量也是int类型
- 修复模式：全面将int提升为int64_t，包括结构体字段、函数签名、循环变量、日志格式串
- 可审查性：高
- 审查规则建议：与shape/尺寸相关的变量一律使用int64_t

### cf971b07 maxpool修正L2_DFX_PHASE_1
- 根因类别：DFX日志宏放置位置错误
- 涉及文件：pooling/max_pool3d_with_argmax_v2/op_host/op_api/aclnn_max_pool2d_with_indices.cpp
- 缺陷描述：L2_DFX_PHASE_1宏被放在if分支内部，kH==1且kW==1或SoC版本不匹配时走else分支不记录DFX日志
- 修复模式：将L2_DFX_PHASE_1调用移到if判断之前
- 可审查性：高
- 审查规则建议：DFX/日志宏应放在条件分支之前确保全路径覆盖

### 9392a5db Tbmm Init Buffer Fix
- 根因类别：buffer初始化方式与执行模型不匹配
- 涉及文件：matmul/transpose_batch_mat_mul/op_kernel/pp_matmul_ein_sum_kernel.h等3个文件
- 缺陷描述：通过TPipe.InitBuffer分配片上缓存，但算子不使用TPipe流水线调度，pipe作为局部变量销毁后buffer引用可能悬空
- 修复模式：用OnChipBuffer替代AsdopsBuffer，直接通过LocalTensor构造函数手动映射片上存储，移除TPipe依赖
- 可审查性：低
- 审查规则建议：片上buffer初始化方式需与算子执行模型匹配；不使用TPipe流水线时不应通过TPipe分配buffer

### 7435f6a6 [conbp]fix singleShapebatch bug
- 根因类别：计算表达式错误 -- 切分数计算分子取错 + 整数溢出
- 涉及文件：conv/conv3d_backprop_filter_v2/op_kernel/arch35/conv3d_backprop_filter_v2/conv3d_dw_v2_basic_block.h
- 缺陷描述：(1)totalAddSize中batch_*dout_*ho_*wo_乘法缺少uint64_t类型提升可能溢出；(2)singleShapeBatch_计算使用Ceil(singleCoreBatch_, batchdoutPerBlock_)但应使用总量batch_*dout_
- 修复模式：(1)对batch_做static_cast<uint64_t>；(2)将Ceil参数从singleCoreBatch_改为batch_*dout_
- 可审查性：中
- 审查规则建议：Ceil(总量, 块大小)的"总量"参数需确认语义——是总任务量而非单核任务量

### 5984e340 fix gen_op command bug
- 根因类别：路径拼写错误(typo)
- 涉及文件：build.sh
- 缺陷描述：gen_op函数中复制模板CMakeLists.txt时源目录写成example/而非examples/(缺少s)，两处重复错误
- 修复模式：将两处example/CMakeLists.txt改为examples/CMakeLists.txt
- 可审查性：高
- 审查规则建议：shell脚本中硬编码路径应核实与实际目录结构匹配

### 4d562156 修复gemmv3UT
- 根因类别：构建依赖缺失 + 测试期望值错误
- 涉及文件：matmul/gemm_v3/op_host/CMakeLists.txt, matmul/gemm_v3/tests/ut/op_host/test_gemm_v3_tiling.cpp
- 缺陷描述：(1)CMakeLists.txt缺少DEPENDENCIES mat_mul_v3声明导致链接失败；(2)tiling UT的golden data中一个数值错误(1应为2)
- 修复模式：补充DEPENDENCIES声明；修正UT期望值
- 可审查性：低
- 审查规则建议：超长magic number字符串形式的golden data应重构为结构化数据

### f6165622 fix compile warning
- 根因类别：未使用参数导致编译告警
- 涉及文件：foreach/foreach_non_finite_check_and_unscale/op_host/foreach_non_finite_check_and_unscale_base_tiling.cpp, foreach/foreach_utils/op_host/foreach_tiling_func.cpp
- 缺陷描述：两个TilingPrepare函数的context参数完全未使用，开启-Wunused-parameter时产生告警
- 修复模式：对context参数添加[[maybe_unused]]属性
- 可审查性：高
- 审查规则建议：启用-Wunused-parameter -Werror

(跳过) aa794631 回退//qbmm - Revert提交，留待阶段3分析
(跳过) 7a065a17 fix AvgPool3dGrad README - 纯文档修复

### bb5aad58 增加越界保护
- 根因类别：边界条件缺失
- 涉及文件：loss/cross_entropy_loss/op_host/cross_entropy_loss_tiling_arch35.cpp
- 缺陷描述：countOnes(uint64_t num)函数在num=0时进入while循环后返回count-1=-1，语义不正确。0的二进制中有0个1，减1后变为-1，若后续用作数组索引或除数会导致越界或未定义行为。
- 修复模式：前置guard clause，在函数入口对num==0提前返回0
- 可审查性：高
- 审查规则建议：对位操作/数学函数，静态检查是否处理了零值和边界输入(0、UINT64_MAX等)

(跳过) ac976d4b mmv3, bmmv3, gemmv2 tiling key revert - Revert提交，留待阶段3分析

### 8c91e2a0 fix conv ut
- 根因类别：头文件include路径错误
- 涉及文件：6个conv相关UT文件(test_aclnn_convolutin_backward.cpp等)
- 缺陷描述：UT文件中#include使用裸文件名，依赖构建系统的隐式include path配置。当include path变化后编译失败。
- 修复模式：将隐式include路径改为显式相对路径(../../../op_host/op_api/...)
- 可审查性：高
- 审查规则建议：UT文件中#include应使用相对路径或明确的项目内路径，禁止依赖隐式include path定位项目内部头文件

### 5506cf24 ophost warning fix
- 根因类别：编译警告(printf格式符不匹配 + 未使用变量/函数 + UT断言注释掉)
- 涉及文件：17个matmul相关文件
- 缺陷描述：批量消除编译警告：(1)大量%zu用于打印int64_t类型，修复为%ld/%lu；(2)删除3个未被调用的静态辅助函数(39行死代码)；(3)注释掉一个UT断言ASSERT_EQ(tiling_data_result, golden_tiling_data)——这意味着tiling正确性不再被验证，可能掩盖回归缺陷
- 修复模式：格式说明符修正 + 死代码清理 + trailing whitespace清理
- 可审查性：中
- 审查规则建议：启用-Wformat -Wformat-signedness；UT中注释掉ASSERT_EQ断言是危险操作，review时应重点关注

### 4aba3129 fastgelu和elugrad错误examples代码修改
- 根因类别：示例代码复制粘贴错误
- 涉及文件：activation/elu_grad_v2/examples/test_aclnn_elu_backward.cpp、activation/fast_gelu/examples/test_aclnn_fast_gelu.cpp、activation/fatrelu_mul/tests/ut/op_kernel/fatrelu_mul_tiling_def.h
- 缺陷描述：两个算子的example文件内容被交换了——elu_backward的example里写的是fast_gelu的调用代码，反之亦然。从创建之日起就是错误的。另外fatrelu_mul_tiling_def.h的include guard名称与文件名不匹配。
- 修复模式：内容互换修正 + include guard命名规范化
- 可审查性：高
- 审查规则建议：示例代码中#include的头文件名应与所在目录的算子名一致；include guard应与文件名匹配；example代码应有CI编译验证

### e00d9aa0 UT bug fix
- 根因类别：构建系统逻辑缺陷(CMake文件查找逻辑错误)
- 涉及文件：cmake/ut.cmake
- 缺陷描述：原代码使用execute_process+stat在CMake configure阶段检测生成文件是否存在，但这些文件标记为GENERATED TRUE，configure阶段尚不存在，检测结果取决于上次构建残留文件。
- 修复模式：将运行时文件探测替换为编译期静态白名单匹配(IN_LIST)
- 可审查性：高
- 审查规则建议：CMake中对GENERATED文件不应使用execute_process做存在性检测

### cfd79187 fix group_norm_swish ut
- 根因类别：UT框架迁移不完整
- 涉及文件：norm/group_norm_swish/tests/下9个文件
- 缺陷描述：UT框架整体迁移修复：(1)CMake GLOB RELATIVE路径解析失败改用LIST_DIRECTORIES true；(2)add_modules_llt_sources迁移到add_modules_ut_sources；(3)ge::AnyValue迁移到Ops::NN::AnyValue；(4)include路径迁移；(5)删除重复的TilingData结构体定义；(6)using namespace std改为显式std::；(7)op_kernel CMake从return()改为AddOpTestCase
- 修复模式：框架迁移适配(批量API/命名空间/构建宏替换)
- 可审查性：低
- 审查规则建议：框架迁移应有checklist逐项验证；UT中手动复制结构体定义应触发告警

(跳过) c86c4af4 fix pooling README - 纯文档修复

### c515aa98 matmul compile warning fix
- 根因类别：编译警告中包含实际逻辑bug(链式比较 + 运算符优先级 + 类型转换方向错误)
- 涉及文件：5个matmul相关文件(aclnn_batch_matmul.cpp、aclnn_gemm.cpp、aclnn_quant_matmul_checker.cpp等)
- 缺陷描述：(1)链式比较a==b==1在C++中等价于(a==b)==1即判断a和b是否相等，而非"两个都等于1"；(2)if条件中&&和||混合使用缺少括号导致优先级错误；(3)static_cast<uint64_t>(int64_t_value)将有符号转为无符号，负值会溢出为巨大数值；(4)删除5处未使用变量
- 修复模式：修正运算符优先级/链式比较逻辑 + 修正类型转换方向(统一转int64_t) + 清理dead code
- 可审查性：高
- 审查规则建议：连续使用==运算符(a==b==c)视为高危模式；if条件混合&&和||时必须显式括号(-Wparentheses)；static_cast<uint64_t>(signed)应触发审查

(跳过) adebf652 fix aclnnAvgPool3DGrad.md - 纯文档修复
(跳过) 9ac31cc5 回退//【1952mm回合主线】wqbmm - Revert提交，留待阶段3分析

### 7d532e6c fix ut
- 根因类别：UT测试数据与tiling参数不匹配
- 涉及文件：vfusion/scaled_masked_softmax_grad_v2/tests/ut/下3个文件
- 缺陷描述：(1)测试shape参数过大(S=512,D=2048)改为合理值(S=64/32,D=1536/256)，coreNum从48改为40匹配实际硬件；(2)tiling数据从代码直接构造改为从bin文件反序列化，使UT更接近真实运行场景
- 修复模式：缩小测试shape + 新增tiling数据生成脚本 + 修改tiling数据读取方式
- 可审查性：低
- 审查规则建议：UT中kernel的tiling数据应通过与生产代码相同的序列化路径生成

### 584223ca fixed torch dynamic graph compile fail
- 根因类别：宏定义顺序错误/头文件include顺序依赖
- 涉及文件：matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4_apt.cpp
- 缺陷描述：float32场景下，#define DTYPE_X2 fp4x2_e2m1_t放在文件末尾，但消费该宏的catlass_convertor.h在文件开头已被include，导致头文件中DTYPE_X2仍为原始值，类型不匹配编译失败
- 修复模式：调整宏定义和头文件include顺序，确保宏在被消费前已正确定义
- 可审查性：高
- 审查规则建议：当代码中存在#undef/#define覆盖宏时，审查需检查所有消费该宏的头文件是否在宏重定义之后才被include

### ed08f259 fix tilingkey
- 根因类别：条件编译分支遗漏/平台适配缺陷
- 涉及文件：matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3.cpp、quant_batch_matmul_v3_tiling_key.h
- 缺陷描述：quant_batch_matmul_v3一个特定kernel分支的代码不在任何__CCE_AICORE__保护下，导致AI Core 220平台缺失该分支的tiling key注册，运行时找不到对应tiling key而失败
- 修复模式：将平台无关代码正确归入各平台条件编译分支，为AI Core 220补充kernel实例化和tiling key注册
- 可审查性：中
- 审查规则建议：多平台条件编译(__CCE_AICORE__ == 200/220)应系统性对比两个平台分支是否覆盖所有相同kernel模板组合

(跳过) ec96530d 回退【1952mm回合主线】mm/bmm - Revert提交，留待阶段3分析

### cf4f58ff qbmm wqbmm ut fix
- 根因类别：UT stub函数残留 + 测试数据不匹配
- 涉及文件：matmul/quant_batch_matmul_v3/tests/ut/、weight_quant_batch_matmul_v2/tests/ut/、tests/ut/common/ut_op_common.cpp
- 缺陷描述：(1)ut_op_common.cpp中硬编码了CheckSupportConditionQbmm和GenWqbmmTiling两个stub函数(总返回true)，让依赖它们的UT永远走不到真实逻辑；(2)删除4条与算子实际tiling结果不匹配的测试参数
- 修复模式：删除无效stub函数 + 删除不匹配的测试参数
- 可审查性：中
- 审查规则建议：公共UT工具文件中不应包含特定算子的stub实现；删除测试用例时必须说明原因

### 8e9b137c fix example bugs of conv3d_transpose_v2
- 根因类别：CMake依赖声明错误
- 涉及文件：conv/conv3d_transpose_v2/op_host/CMakeLists.txt
- 缺陷描述：DEPENDENCIES字段写成convolution_backward，正确的应为convolution_forward。conv3d_transpose(反卷积)的前向计算依赖convolution的forward路径而非backward路径。
- 修复模式：单词替换 convolution_backward -> convolution_forward
- 可审查性：高
- 审查规则建议：CMake DEPENDENCIES变更时验证依赖关系的语义正确性

### 35c6053f [convbp]fix compile warning logs
- 根因类别：类型不匹配 + 函数调用缺少参数
- 涉及文件：conv/conv3d_backprop_input_v2/op_host/op_tiling/下2个文件
- 缺陷描述：(1)NUM_THREE常量声明为uint32_t但与int32_t混合运算导致有符号/无符号比较警告；(2)两个局部变量声明后未使用；(3)IsArchAfter35调用缺少context参数——不传context可能导致架构判断错误
- 修复模式：类型修正 + 删除未使用变量 + 补全函数调用参数
- 可审查性：高
- 审查规则建议：函数调用缺少参数通常是重构时遗漏，对IsArchAfter35调用点做全局搜索确认一致性

### 15851ff1 fix UT bug
- 根因类别：UT基础设施迁移不完整 + proto头文件语法错误 + tiling结构不匹配
- 涉及文件：index/scatter_elements_v2/下4个文件
- 缺陷描述：(1)scatter_elements_v2_proto.h中OP_END_FACTORY_REG后缺少namespace闭合花括号；(2)infershape UT注册被注释掉导致不会编译执行；(3)测试代码引用错误的tiling头文件和类型名；(4)gen_tiling.py中tiling字段数量与实际结构体不一致(旧14字段vs新30+字段)；(5)删除5个冗余测试用例
- 修复模式：语法修复 + 路径修正 + 类型名对齐 + tiling数据结构同步
- 可审查性：中
- 审查规则建议：tiling结构体变更时必须同步更新所有引用该结构的UT数据生成脚本

### 66100677 SMS/SMSG ut fix
- 根因类别：UT框架迁移(LLT→新UT框架)多处适配问题
- 涉及文件：vfusion/scaled_masked_softmax_grad_v2/和scaled_masked_softmax_v2/下16个文件
- 缺陷描述：大规模UT框架迁移：(1)CMake条件从add_modules_llt_sources改为add_modules_ut_sources；(2)旧版CMake以return()开头直接跳过所有构建(kernel UT从未执行)；(3)头文件路径迁移；(4)删除硬编码的#define __CCE_AICORE__ 220；(5)gen_data.py中使用未定义变量dtype；(6)调试日志残留——softmax模块打印了is_finite的日志(复制粘贴未修改)
- 修复模式：框架迁移适配(构建系统 + 头文件路径 + API名称 + 数据生成逻辑)
- 可审查性：低
- 审查规则建议：CMake中return()直接跳过构建是极大红旗；调试日志模块名应与实际模块匹配

(跳过) 61a1f158 revert identity/identity_n/rank/shape/shape_n - Revert提交，留待阶段3分析

### a87c18f0 qbmm opensource fix
- 根因类别：架构重构/开源合规(硬编码白名单暴露)
- 涉及文件：common/stub/op_tiling/op_cache_def_tiling.h、quant_batch_matmul_v3相关tiling文件、ut_op_common.cpp
- 缺陷描述：QuantBatchMatmulV3的tiling逻辑中大量硬编码shape白/黑名单直接写死在cpp文件里。开源时这些与特定硬件平台绑定的shape list不应暴露在源码中。
- 修复模式：抽象接口封装——新增QuantBatchMatmulRunParas结构体和QbmmType枚举，将所有白/黑名单判断统一收敛到CheckSupportConditionQbmm() weak symbol函数中，stub固定返回true
- 可审查性：中
- 审查规则建议：源码中硬编码的std::set<std::tuple<...>>形式的shape常量应抽象为配置或接口；weak symbol stub返回值需审查是否符合预期行为

### a6be209a fix issues
- 根因类别：文档路径错误 + 示例代码硬编码
- 涉及文件：docs/context/quick_op_invocation.md、matmul/quant_batch_matmul_v4/examples/test_aclnn_quant_matmul_v5_at2_at3.cpp
- 缺陷描述：(1)文档中pip3 install路径tests/requirements.txt错误，应为requirements.txt；(2)示例代码deviceId=1硬编码非零设备号，单卡环境直接失败
- 修复模式：值修正——路径修正 + deviceId从1改为0
- 可审查性：高
- 审查规则建议：文档中引用的文件路径应自动校验是否存在；示例代码deviceId默认值应为0

### 5bb873eb fix cleancode
- 根因类别：代码规范/const正确性
- 涉及文件：cmake/gen_ops_info.cmake、conv/convolution_backward/op_host/op_api/convolution_backward_checker.cpp/.h
- 缺陷描述：IsConv8bit方法不修改成员状态但未声明为const，导致const上下文中无法调用。另删除cmake调试日志。
- 修复模式：const修饰符补全 + 冗余日志清理
- 可审查性：高
- 审查规则建议：不修改成员的方法应声明为const(clang-tidy readability-make-member-function-const)

### 2e985117 解决group_quant再910_93编译问题
- 根因类别：文件命名错误导致编译失败
- 涉及文件：quant/group_quant/op_host/config/ascend910_93/group_quant_simplified_key_mode.ini(重命名为group_quant_simplified_key.ini)
- 缺陷描述：配置文件名group_quant_simplified_key_mode.ini多了_mode后缀，构建系统查找的是group_quant_simplified_key.ini，文件名不匹配导致910_93平台编译失败
- 修复模式：文件重命名去掉多余_mode后缀
- 可审查性：高
- 审查规则建议：新增配置文件时CI应在所有目标平台上验证编译；配置文件命名应有规范文档

### 2aced33a fix foreach compile error
- 根因类别：宏接口签名不一致
- 涉及文件：foreach/foreach_lerp_scalar_def.cpp、foreach_pow_scalar_and_tensor_def.cpp、foreach_round_off_number_def.cpp、foreach/foreach_utils/op_host/foreach_proto_utils.h
- 缺陷描述：头文件中宏FOREACH_OPDEF_PARAM_SCALAR接受3个参数，但调用侧与内部变量名存在耦合。另外FOREACH_BINARY_LIST_ALPHA_PARAM中参数名scalar应为alpha。
- 修复模式：宏接口重构——减少宏参数个数消除耦合；参数名语义修正
- 可审查性：中
- 审查规则建议：宏定义变更时全量搜索所有调用点确认参数个数同步更新；宏内部依赖外部作用域局部变量是脆弱设计

### 0ebd992a 修复example编译找不到头文件的问题
- 根因类别：构建系统include路径缺失
- 涉及文件：build.sh, cmake/opbuild.cmake, cmake/variables.cmake
- 缺陷描述：头文件安装路径不正确，导致example编译时找不到aclnn头文件。build.sh中缺少`-I ${INCLUDE_PATH}/aclnnop`，cmake中未定义`ACLNN_OP_INC_INSTALL_DIR`且未将头文件安装到`aclnnop`子目录
- 修复模式：在构建系统中新增include路径变量并同步安装头文件到多个路径（兼容不同引用方式）
- 可审查性：高
- 审查规则建议：cmake install配置变更时检查是否覆盖了所有必要路径；build脚本中`-I`参数需包含完整搜索路径

### b2f36968 fix group-norm include path
- 根因类别：include路径错误
- 涉及文件：norm/group_norm/op_host/op_api/aclnn_group_norm.cpp
- 缺陷描述：`#include "../../../norm/group_norm_silu/..."`使用过深相对路径引用，路径不正确。修复为`#include "norm/group_norm_silu/..."`使用项目include路径
- 修复模式：#include相对路径`../../../`错误 -> 改用基于项目根的include路径
- 可审查性：高
- 审查规则建议：检测`#include`中使用3层以上`../`的过深相对路径引用，推荐使用基于项目根目录的路径

### 86c5fb8f kernelut&readme fix
- 根因类别：CMake变量引用错误
- 涉及文件：cmake/func.cmake, cmake/ut.cmake, 多个CMakeLists.txt, gen_tiling_head_file.py
- 缺陷描述：`cmake/func.cmake`中`set(OP_TYPE ...)`应为`set(${OP_TYPE} ...)`——函数参数作为变量名传入`set`时需要解引用，否则设置的是字面量"OP_TYPE"而非调用者传入的变量名，导致函数返回值丢失。另外ut.cmake中opType推断从下划线分词大小写转换改为从binary.json读取，避免非规则命名出错
- 修复模式：CMake `set(VAR_NAME ...)` vs `set(${VAR_NAME} ...)`解引用遗漏，导致输出变量名写死
- 可审查性：高
- 审查规则建议：CMake函数中`set()`第一个参数如果来自函数参数，必须用`${}`解引用

### 7cc5ae81 算子注册原型bugfix
- 根因类别：语法结构不完整
- 涉及文件：control/assert/op_graph/assert_proto.h
- 缺陷描述：`OP_END_FACTORY_REG(Assert)`后缺少namespace闭合`}`导致编译失败
- 修复模式：proto.h文件缺少右大括号——namespace/struct闭合不完整
- 可审查性：高
- 审查规则建议：proto.h文件中检查namespace/struct/class的大括号匹配（此类缺陷在ops-nn-dev中多次出现）

### 723a43ae fix warning for conv3dv2 0924
- 根因类别：编译告警修复（含实质性改动）
- 涉及文件：conv3d_api_tiling_algorithm_hw_mode.cpp/.h, aclnn_convolution.cpp, aclnn_quant_convolution.cpp
- 缺陷描述：(1)头文件include依赖传播过广，从header移到cpp；(2)缺少override关键字约20处；(3)C风格强制转换未改static_cast；(4)magic number 4未替换为CONV_2D_DIM_SIZE常量；(5)未使用的stride参数未移除
- 修复模式：大规模override/const/C-cast/magic number告警修复
- 可审查性：高
- 审查规则建议：虚函数重写必须带override；禁止C风格cast，使用static_cast/reinterpret_cast

### 63bee42e fix maxpool3dwithargmaxv2 kernel select
- 根因类别：模板类型安全
- 涉及文件：max_pool3d_with_argmax_v2_base.h, *_nosplit.h, *_splitd.h, *_splith.h, *_splitw.h
- 缺陷描述：`Select<T>()`调用中scalar参数使用`1.f * kernelDHW`或`-1.f`（float字面量），但模板T可能是half等非float类型。T为half时传float给Select的scalar参数导致类型不匹配
- 修复模式：模板函数scalar参数使用固定float字面量而非(T)显式转换，当T!=float时类型不匹配
- 可审查性：高
- 审查规则建议：模板函数中数值字面量（1.f, -1.f等）传给模板类型T参数时，须显式转换`(T)value`

### 092b884e fix cleancode
- 根因类别：命名空间污染
- 涉及文件：conv3d_backprop_filter_v2_base_tiling.h, conv3d_backprop_filter_v2_basic_block_tiling.h
- 缺陷描述：头文件中`using namespace`位于全局作用域，会污染所有包含此头文件的编译单元的命名空间，可能导致名称冲突
- 修复模式：将`using namespace`从全局作用域移到具体命名空间内部
- 可审查性：高
- 审查规则建议：头文件(.h)中禁止全局作用域的`using namespace`声明

### c6726c69 修复CCE裸指令问题，clean code问题
- 根因类别：CCE裸指令使用（硬件兼容性风险）
- 涉及文件：17个文件，涉及chamfer_distance_grad, cross_entropy_loss, ctc_loss_v3_grad, advance_step, apply_adam_w_v2, multi_scale_deformable_attn_function
- 缺陷描述：kernel代码中使用`pipe_barrier(PIPE_ALL)`等CCE裸指令（底层硬件指令），替换为AscendC高级API `PipeBarrier<PIPE_ALL>()`。裸指令可移植性差，不同硬件版本行为可能不一致
- 修复模式：`pipe_barrier()` -> `PipeBarrier<>()`，CCE裸指令统一替换为AscendC API
- 可审查性：高
- 审查规则建议：kernel代码中检测CCE裸指令（pipe_barrier, set_flag等），应使用AscendC高级API替代

### b58b066f LayerNormGradV3日志等级错误修复
- 根因类别：日志等级误用
- 涉及文件：layer_norm_grad_v3_common_tiling.cpp, layer_norm_grad_v3_single_read_tiling.cpp, layer_norm_grad_v3_transpose_tiling.cpp
- 缺陷描述：`IsCapable()`方法中当某tiling模板不适用时使用`OP_LOGE`(ERROR级别)，但这不是错误——只是当前模板不适用会继续尝试下一个。ERROR日志误导用户
- 修复模式：`OP_LOGE` -> `OP_LOGI`，正常执行路径中ERROR日志应降级为INFO
- 可审查性：高
- 审查规则建议：IsCapable()/capability check方法中return false路径使用OP_LOGE是常见错误，应为OP_LOGI或OP_LOGD

### a9eab151 swi_glu_grad arch35 kernel fix
- 根因类别：C++模板语法错误
- 涉及文件：activation/swi_glu_grad/op_kernel/arch35/swi_glu_grad_ub_rearrange.h
- 缺陷描述：`template <typename T> struct VciTypeGet<> {};`语法错误——主模板声明后跟了空特化参数列表`<>`，应为`template <typename T> struct VciTypeGet;`
- 修复模式：模板主声明与部分特化语法混淆
- 可审查性：高
- 审查规则建议：检测`template <typename T> struct/class Name<> {}`模式——这是无效的空特化参数列表

### 305aac24 fix build.sh
- 根因类别：构建脚本参数校验过严
- 涉及文件：build.sh
- 缺陷描述：build.sh对非识别参数的错误处理逻辑过于严格，将合法参数误判为非法并报错退出
- 修复模式：条件判断逻辑过严误拒合法输入（此模式在ops-nn-dev多次出现：错误校验逻辑过早拒绝合法输入）
- 可审查性：中
- 审查规则建议：参数校验的error分支应列举所有合法情况，避免未来新增参数时被误拒

### ed871d87 修复conv编译告警
- 根因类别：编译告警修复（含实质性缺陷）
- 涉及文件：conv_forward_infershape.cpp, conv3d_api_tiling_algorithm.h, aclnn_convolution.cpp, aclnn_quant_convolution.cpp, convolution.cpp
- 缺陷描述：(1)`OpParamIdx`结构体成员未初始化（添加`= 0`），可防止未定义行为；(2)`dimIdx < 0`对`size_t`(unsigned)永远为false，删除无意义检查；(3)删除未使用模板函数`All<T,Func>()`；(4)大量unused变量清理；(5)CheckPad/CheckPadDepthwise2d签名去掉未使用的stride参数
- 修复模式：POD结构体成员未初始化 + unsigned与0比较 + unused参数/变量
- 可审查性：高
- 审查规则建议：POD/聚合结构体成员必须初始化；unsigned类型与`< 0`比较是逻辑错误（编译器-Wsign-compare可检测）

### d87304ad fix sc
- 根因类别：编译告警修复（批量）
- 涉及文件：27个文件，涉及activation, common, index, loss, matmul, pooling, quant
- 缺陷描述：(1)头文件static函数缺inline导致ODR violation；(2)cross_entropy_loss_tiling中`while((a >>= 1ULL) && a!=0ULL)`隐式bool转换告警，改为显式`static_cast<bool>`；(3)cross_v2_tiling添加nullptr检查——防御性编程改进；(4)函数参数添加const修饰；(5)未使用参数移除
- 修复模式：头文件static函数缺inline(ODR) + 隐式bool转换 + nullptr防御
- 可审查性：中
- 审查规则建议：头文件中static函数应加inline避免ODR问题

### d108eaae fix warning for conv3dv2 0922
- 根因类别：编译告警修复（大规模）
- 涉及文件：8个conv3d相关文件
- 缺陷描述：与723a43ae(0924)属同一批次修复的前序版本。(1)include依赖重组——header中移除不必要include；(2)约20处虚函数重写缺override；(3)约18处PreProcess/Impl/PostProcess缺override；(4)删除未使用的模板函数；(5)大量const修饰添加
- 修复模式：大规模override缺失修复（同一文件aclnn_convolution.cpp在2天内分2次修复，说明问题范围大且遗漏多）
- 可审查性：高
- 审查规则建议：同723a43ae——虚函数重写必须带override（编译器-Wsuggest-override可检测）

跳过：239b38a2(示例变量名错误), dc1d5a96(日志文本补充), d8774743(README/文档修复), 9d40cbc4(文档链接修复), 68c6a0ad(示例修复), 1fcd5463(示例修复), 1f3a2268(示例修复), 1462b675(代码重构-文件重命名无逻辑变更), 979d049c(构建系统占位注释workaround), 7539638d(清理删除空文件), 38b87118(license修复), cccda0de(示例文件重命名)

### 3e7d1fc4 tbmm warning问题修复
- 根因类别：内存管理API不规范
- 涉及文件：transpose_batch_mat_mul下pp_matmul_ein_sum_kernel.h, transpose_batch_mat_mul.cpp, utils/mem.h
- 缺陷描述：tbmm算子通过InitBuffer/address_.logicPos手动初始化tensor内存，绕过TPipe管理。新版编译器产生warning且存在内存管理不规范问题。修复为通过TPipe::InitBuffer+TBuf标准API，并添加pipe.Destroy()生命周期管理
- 修复模式：裸内存初始化 -> TPipe标准管道管理
- 可审查性：中
- 审查规则建议：检测kernel中直接操作address_.logicPos的非标准内存初始化模式

### f121a9e4 fix kernel ut source files dependencies issue
- 根因类别：CMake构建依赖缺陷
- 涉及文件：cmake/ut.cmake, tests/ut/op_host/CMakeLists.txt
- 缺陷描述：(1)kernel UT中file(GLOB)在cmake配置阶段查找.cpp文件，但该文件是构建阶段生成的，配置时不存在导致依赖缺失。(2)op_host UT硬编码链接static_lib target，当无用例时target不存在导致增量编译失败
- 修复模式：file(GLOB)替换为显式set+GENERATED属性+target_sources延迟指定；用$<TARGET_EXISTS:>守卫条件链接
- 可审查性：高
- 审查规则建议：禁止file(GLOB)获取构建时生成的源文件；链接可选target用$<TARGET_EXISTS:>守卫

### d490c4b4 fix soc package
- 根因类别：打包系统配置冗余
- 涉及文件：cmake/makeself_built_in.cmake, cmake/package.cmake, cmake/runtimeKB.cmake, scripts/package下多个XML和安装脚本
- 缺陷描述：每个soc_version独立维护runtimeKB cmake、打包XML和安装脚本，新增soc需大量复制。合并为统一参数化配置
- 修复模式：消除soc维度的重复配置，合并为参数化统一配置
- 可审查性：中
- 审查规则建议：检测scripts/package下按soc维度冗余的配置文件

### 839fce05 fix warning for conv3dv2
- 根因类别：编译告警修复
- 涉及文件：conv相关9个文件
- 缺陷描述：(1)模板constexpr比较宏展开优先级需加括号；(2)虚函数覆盖缺override；(3)static const在header应为inline const；(4)内部链接函数缺匿名namespace
- 修复模式：加括号/override/inline/匿名namespace
- 可审查性：高
- 审查规则建议：-Wsuggest-override, -Wparentheses, header中static const非constexpr变量应改inline

### 5ed81216 修复增量编译失败bug
- 根因类别：增量编译依赖解析不完整
- 涉及文件：scripts/util/parse_changed_files.py, parse_compile_changed_files.py
- 缺陷描述：反向依赖(reverse_op_dependencies)获取后直接附加到编译依赖列表，但反向依赖本身的编译依赖(二级依赖)未被检索，导致编译时缺少必要依赖
- 修复模式：两步依赖解析——先获取反向依赖，再以反向依赖为输入获取其编译依赖
- 可审查性：高
- 审查规则建议：依赖解析需传递闭包，不能只取一级依赖

### 743fc6ac addrmsnormquantv2精度修复
- 根因类别：模板参数组合下的精度损失
- 涉及文件：norm/add_rms_norm_quant/op_kernel/add_rms_norm_quant.h
- 缺陷描述：ComputePart2中当TX为half时执行fp32->fp16降精度Cast，但模板参数PT=true(高精度路径)时不应降精度。条件`is_same<TX, half>::value`缺少`&& !PT`判断
- 修复模式：`is_same<TX, half>::value` -> `is_same<TX, half>::value && !PT`
- 可审查性：高
- 审查规则建议：新增模板参数后需review所有conditioned-on-type的Cast分支

### c4f2f27e compile mmv3 david fix
- 根因类别：算子配置遗漏
- 涉及文件：matmul/mat_mul_v3/op_host/mat_mul_v3_def.cpp
- 缺陷描述：mat_mul_v3在ascend910_95配置中缺少opFile.value和aclnnSupport.value扩展配置，导致编译找不到正确kernel文件
- 修复模式：添加ExtendCfgInfo("opFile.value", ...).ExtendCfgInfo("aclnnSupport.value", ...)
- 可审查性：高
- 审查规则建议：新增soc配置时检查是否包含完整的ExtendCfgInfo(opFile, aclnnSupport等)

### c097c981 bugfix_crossv2
- 根因类别：算子注册遗漏
- 涉及文件：loss/cross_v2/op_host/cross_v2_tiling.cpp
- 缺陷描述：CrossV2算子tiling注册缺少TilingParse，IMPL_OP_OPTILING只注册了Tiling函数未链式调用.TilingParse<>()，运行时找不到CompileInfo解析入口
- 修复模式：补充空TilingPrepare函数并在注册链追加.TilingParse<CrossV2CompileInfo>(...)
- 可审查性：高
- 审查规则建议：所有IMPL_OP_OPTILING注册必须包含.TilingParse调用（此模式在ops-nn-dev多次出现）

### 6f6bdfdc addprermsnormquant精度问题修复
- 根因类别：模板参数组合下的精度损失
- 涉及文件：norm/add_rms_norm_quant/op_kernel/add_rms_norm_quant.h, add_rms_norm_quant_split_d.h
- 缺陷描述：与743fc6ac同一算子同类问题。bf16输入有beta场景中，PT=true时数据已是fp32却仍执行bf16->fp32 Cast(先truncate再cast)导致精度损失
- 修复模式：`if (hasBeta)` -> `if (hasBeta && !PT)`
- 可审查性：高
- 审查规则建议：同743fc6ac

### 4753a05e rms_norm_quant解决内存检测问题
- 根因类别：内存越界 + 流水线同步缺失
- 涉及文件：norm/rms_norm/op_kernel/rms_norm_base.h, norm/rms_norm_quant/op_kernel/rms_norm_quant.cpp
- 缺陷描述：三重根因：(1)DataCopy/Cast/Mul等操作使用对齐后大小num_col_align_f16而非实际num_col_，读写越界；(2)手动repeat参数计算不处理尾部对齐，改用单参数API；(3)Cast/Mul之间缺PipeBarrier<PIPE_V>，ReduceSum后缺SetFlag/WaitFlag<V_S>
- 修复模式：用实际大小替换对齐大小；简化向量API；补充pipeline同步
- 可审查性：高
- 审查规则建议：DataCopy元素数量不应用align后的大小；向量操作间需PipeBarrier同步

### 45f4afd6 修复transpose 511 case错误
- 根因类别：边界常量设置过小
- 涉及文件：conv3d_backprop_input_v2的tiling/kernel/test等文件
- 缺陷描述：filter维度H/W上限kFilterDimHWUp设为255，实际应为511，导致filter 256-511范围的case被错误拒绝
- 修复模式：constexpr kFilterDimHWUp = 255 -> 511
- 可审查性：高
- 审查规则建议：边界常量修改需与硬件spec/算子协议文档对齐验证

### f14030e7 fmm opapi ut fix
- 根因类别：UT include路径错误
- 涉及文件：matmul/fused_mat_mul/tests/ut/op_host/test_aclnn_fused_matmul.cpp
- 缺陷描述："common/op_api_def.h"不正确，应为"op_api/op_api_def.h"
- 修复模式：修正UT include路径
- 可审查性：高
- 审查规则建议：UT的include路径应与产品代码保持一致

### 7876dbde pkg打包报错修复以及增量编译修复
- 根因类别：构建脚本多重缺陷
- 涉及文件：build.sh, cmake/package.cmake, scripts/util/parse_changed_files.py, parse_compile_changed_files.py
- 缺陷描述：(1)CI模式下parse脚本输出未过滤就添加到TRIGER_UTS导致触发不存在的target；(2)package.cmake对所有soc都include runtimeKB但只有部分soc有文件；(3)os.path.exists缺strip()导致检查失败
- 修复模式：过滤op_*前缀；soc条件判断；strip()处理
- 可审查性：高
- 审查规则建议：文件路径字符串处理前必须strip()；CI脚本输出解析需有格式验证

### 1344a3b3 修复没有aclnn代码编译失败的问题
- 根因类别：CMake target不存在时未守卫
- 涉及文件：build.sh, cmake/opbuild.cmake, cmake/package.cmake, cmake/symbol.cmake
- 缺陷描述：(1)无aclnn代码时opapi target不存在，package.cmake/symbol.cmake无条件引用导致cmake报错；(2)opbuild.cmake未收集带版本号的生成文件(如aclnn_xxx_v2.cpp)
- 修复模式：if(TARGET ...)守卫；file(GLOB)匹配版本号变体
- 可审查性：高
- 审查规则建议：cmake引用target前用if(TARGET)或$<TARGET_EXISTS:>守卫

### 05ea548c fix bias optional range
- 根因类别：optional输入API误用
- 涉及文件：matmul/common/op_host/matmul_common_infershape.cpp
- 缺陷描述：bias是optional输入，但使用GetInputShapeRange按物理索引取值，bias不存在时取到错误输入。应使用GetOptionalInputShapeRange
- 修复模式：GetInputShapeRange -> GetOptionalInputShapeRange
- 可审查性：高
- 审查规则建议：optional输入必须使用GetOptionalInput*/GetOptionalInputShapeRange API

### ebdf8b70 aclnn ut 修复不带asan卡死问题
- 根因类别：UT框架依赖真实硬件初始化
- 涉及文件：tests/ut/op_api下43个文件，ut_main.cpp，新增stub/opdev/platform.h/cpp
- 缺陷描述：ut_main.cpp的SetUp调用PlatformInfoManager初始化，纯CPU UT无硬件则阻塞卡死
- 修复模式：移除真实平台初始化，替换为stub实现
- 可审查性：中
- 审查规则建议：UT框架不应依赖真实硬件平台初始化

### d79d3c4b fix aicpu include path bug
- 根因类别：跨仓路径引用错误
- 涉及文件：cmake/variables.cmake
- 缺陷描述：AICPU_INCLUDE列表中使用${OPS_CV_DIR}(CV仓路径)，在NN仓中应使用${OPS_NN_DIR}
- 修复模式：${OPS_CV_DIR} -> ${OPS_NN_DIR}
- 可审查性：高
- 审查规则建议：cmake include路径不应引用其他仓库的目录变量

### d60227e1 bugfix
- 根因类别：tiling路径条件不完整
- 涉及文件：matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp
- 缺陷描述：GetMixNd2nzType判断V_PARALELL_ND2NZ路径时缺少tilingEnableSplitCore和tilingEnableFixOpti的约束，非BASE状态走该路径计算错误
- 修复模式：条件追加 && tilingEnableSplitCore == BASE && tilingEnableFixOpti == BASE
- 可审查性：高
- 审查规则建议：tiling路径选择条件需覆盖所有相关配置参数组合

### ce3b325f 修复删除算子，增量编译不通过的情况
- 根因类别：增量编译脚本未处理已删除文件
- 涉及文件：scripts/util/parse_changed_files.py, parse_compile_changed_files.py
- 缺陷描述：changed_files中的文件路径可能已不存在(算子被删除)，后续os.path.relpath因文件不存在报错
- 修复模式：处理前添加os.path.exists()跳过已删除文件
- 可审查性：高
- 审查规则建议：处理git diff文件列表时必须考虑文件已删除

### a1c019e8 修复核数少于6个时的问题
- 根因类别：硬编码核数参数
- 涉及文件：matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp
- 缺陷描述：L2切分参数maxConflictDim=6, minConflictDim=3硬编码，低端设备核数<6时分配到不存在的核
- 修复模式：std::min(compileInfo_.aicNum, 6UL) / std::min(compileInfo_.aicNum, 3UL)
- 可审查性：高
- 审查规则建议：tiling中核数相关参数必须以实际aicNum为上界；检测核数字面量

### 6e0e4681 fix triger ut
- 根因类别：UT触发逻辑缺陷
- 涉及文件：build.sh, scripts/util/dependency_parser.py
- 缺陷描述：增量UT触发时对TRIGER_UTS中每个entry无条件执行cmake构建，但有些target当前配置中不存在导致失败
- 修复模式：触发前检查target是否在当前构建目标中
- 可审查性：高
- 审查规则建议：CI中cmake --build --target前验证target存在性

### 34428023 fix rnn 裸指令
- 根因类别：CCE裸指令误用
- 涉及文件：rnn/dynamic_rnn/op_kernel/dynamic_rnn_910b.cpp, rnn/dynamic_rnnv2/op_kernel/dynamic_rnn_v2_910b.cpp
- 缺陷描述：kernel入口调用set_mask_norm()裸指令，可能与框架mask管理冲突
- 修复模式：删除裸指令调用
- 可审查性：高
- 审查规则建议：检测kernel入口中set_mask_norm/set_mask_count等裸指令

### 1f9512ca fix bias len 0
- 根因类别：边界条件遗漏
- 涉及文件：matmul/common/op_host/matmul_common_infershape.cpp
- 缺陷描述：InferRangeBias中num_dim_bias为0(bias不存在)时仍初始化vector并校验，可能越界或推导错误
- 修复模式：num_dim_bias==0时提前返回
- 可审查性：高
- 审查规则建议：GetDimNum()返回值为0需边界检查；可选输入shape推导需覆盖"空输入"

### eb5c3bd2 dynamic_quant_fix
- 根因类别：constexpr与device编译不兼容
- 涉及文件：quant/dynamic_quant/op_kernel/v35/dynamic_quant_struct.h
- 缺陷描述：constexpr uint32_t常量在device端编译环境中可能因ODR问题或不支持导致链接冲突，改用#define
- 修复模式：constexpr -> #define
- 可审查性：中
- 审查规则建议：AscendC kernel头文件中用于模板参数的常量优先使用#define

### a28cf7c7 fix nz copy bug
- 根因类别：NZ格式K维度对齐计算错误
- 涉及文件：matmul/mat_mul_v3/op_kernel/mat_mul_sc_splitk_kernel_gm_to_l1.h
- 缺陷描述：splitk kernel中B矩阵为ND格式时不需按c0Size对齐K，但使用了对齐后的realK导致拷贝越界。NZ格式才需对齐
- 修复模式：引入realKb变量，ND格式用实际K，NZ格式用对齐值
- 可审查性：高
- 审查规则建议：同一算子支持ND/NZ格式时，所有维度对齐操作需对格式做区分

### 5b4f67ce fix bug
- 根因类别：隐式类型窄化
- 涉及文件：index/non_zero/op_host/op_api/nonzero.cpp
- 缺陷描述：size_t加1后传入Shape构造函数(期望int64_t)，隐式转换可能窄化
- 修复模式：添加static_cast<int64_t>()
- 可审查性：高
- 审查规则建议：size_t到有符号整型的隐式转换应显式cast

### 45320686 fix bug
- 根因类别：API层级依赖错误
- 涉及文件：index/embedding_dense_grad_v2/op_host/op_api/aclnn_embedding_dense_backward.cpp
- 缺陷描述：使用上层ACL接口aclrtCtxGetSysParamOpt(acl/acl.h)，应使用底层runtime接口rtCtxGetSysParamOpt(runtime/context.h)，解除不必要的上层依赖
- 修复模式：API调用和头文件替换
- 可审查性：高
- 审查规则建议：算子host代码不应直接依赖acl/acl.h，应使用runtime层接口

### f955a59b 空指针检验
- 根因类别：空指针解引用
- 涉及文件：index/scatter_elements_v2/op_host/scatter_elements_v2_tiling.cpp
- 缺陷描述：GetAttrPointer<int>(0)返回值直接解引用*(attrs->GetAttrPointer<int>(0))，未检查nullptr
- 修复模式：改为指针接收+OP_CHECK_NULL_WITH_CONTEXT检查后再解引用
- 可审查性：高
- 审查规则建议：GetAttrPointer返回值必须先nullptr检查再解引用

### f86b5d37 fix_TBMM_910b_91093_重复部分
- 根因类别：配置冗余
- 涉及文件：transpose_batch_mat_mul_binary.json(ascend910_93和ascend910b)
- 缺陷描述：binary config中存在重复配置条目
- 修复模式：删除JSON中重复的simplified_key条目
- 可审查性：中
- 审查规则建议：binary config JSON不应存在相同simplified_key的重复条目

### f65088be fix foreach compilation failure
- 根因类别：目录结构不符合构建规范
- 涉及文件：976个文件(foreach系列算子graph_plugin→op_graph重命名)
- 缺陷描述：foreach系列算子目录结构使用旧名graph_plugin而非新规范op_graph，导致编译失败
- 修复模式：批量目录重组+CMakeLists路径修正
- 可审查性：低
- 审查规则建议：新增算子目录结构必须遵循op_graph命名约定

### 19456c2f 解决伪量化S4数据类型不匹配INT32问题
- 根因类别：宏条件判断顺序错误
- 涉及文件：matmul/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_apt.cpp
- 缺陷描述：框架传入INT32表示packed INT4，但先检查DT_INT4定义S4宏再处理INT32重定义，此时ORIG_DTYPE_WEIGHT仍是DT_INT32不等于DT_INT4，S4宏永远不定义，S4量化路径完全失效
- 修复模式：调整宏条件顺序，先INT32→INT4重定义再判断S4
- 可审查性：高
- 审查规则建议：数据类型映射的宏定义链需检查定义顺序确保下游宏正确匹配

### b09f2321 fix api_stub
- 根因类别：CMake链接依赖错误
- 涉及文件：cmake/symbol.cmake, common/stub/op_api/CMakeLists.txt, tests/ut/op_api/CMakeLists.txt
- 缺陷描述：target_link_directories引入对已安装CANN包的隐式依赖，改为add_dependencies确保先编译本地target
- 修复模式：link_directories -> add_dependencies + IMPORTED target
- 可审查性：中
- 审查规则建议：避免target_link_directories，优先target-based依赖

### a9384107 fix gather_elements bug
- 根因类别：维度校验缺失 + 循环条件顺序错误
- 涉及文件：index/gather_elements_v2/op_host/gather_elements_v2_last_dim_tiling.h, op_api/aclnn_gather.cpp
- 缺陷描述：(1)MergeAxis中xDimNum和indexDimNum不等未校验导致越界；(2)IfUseGatherElementsV2中dtypeAndSocCheckFlag设置位于break之后，break时未被正确设置
- 修复模式：增加维度相等校验；调整条件判断到break前
- 可审查性：高
- 审查规则建议：多tensor维度操作前校验维度数一致；循环break前后的状态设置需审查是否被跳过

### a49477aa readme&code fix
- 根因类别：返回值类型语义错误
- 涉及文件：matmul/mat_mul_v3/op_host/mat_mul_v3_infershape.cpp
- 缺陷描述：函数返回ge::graphStatus，`return false`隐式转为0即GRAPH_SUCCESS，语义完全相反——本应报错却返回成功
- 修复模式：return false -> return ge::GRAPH_FAILED
- 可审查性：高
- 审查规则建议：返回ge::graphStatus的函数禁止return false/true，必须用枚举值

### 9ab56a34 ut fix
- 根因类别：UT期望值错误
- 涉及文件：conv/conv3d_backprop_input_v2/tests/ut/op_host/test_conv3d_backprop_input_v2_infershape.cpp
- 缺陷描述：动态shape场景infershape结果应为[-1,-1,-1,-1,-1]，UT错误地期望[32,16,8,8,24](静态shape值)
- 修复模式：修正UT期望值
- 可审查性：高
- 审查规则建议：动态shape场景UT应验证输出shape包含-1维度

### 7b4afa89 mmv3 wqbmmv2 code fix
- 根因类别：整数溢出
- 涉及文件：matmul/mat_mul_v3/op_host/mat_mul_v3_infershape.cpp, matmul/weight_quant_batch_matmul_v2/op_host/op_api/aclnn_weight_quant_batch_matmul_v2.cpp
- 缺陷描述：(1)infershape中*input_size+BLOCK_SIZE-1可能溢出int64_t；(2)wqbmmv2中mX*kX可能溢出int64_t后与MAX_MK_VALUE比较不可靠
- 修复模式：前置溢出检查input_size>max-BLOCK_SIZE+1；新增CheckMultiplyOverflow函数
- 可审查性：高
- 审查规则建议：shape维度算术运算(加法/乘法)应添加溢出检查

### 66d2beb0 修复avgpool3d 310p迁移问题
- 根因类别：算子注册遗漏
- 涉及文件：pooling/avg_pool3_d/op_host/avg_pool3_d_tiling.cpp
- 缺陷描述：AvgPool3DV2算子缺少IMPL_OP_OPTILING注册，310P平台无法正确执行tiling
- 修复模式：补充IMPL_OP_OPTILING(AvgPool3DV2)注册
- 可审查性：高
- 审查规则建议：新增V2/V3版本算子时检查是否遗漏IMPL_OP_OPTILING注册

### fe7c0913 conv use base func fix
- 根因类别：API迁移/依赖清理
- 涉及文件：conv/common/op_host/conv_forward_infershape.cpp, 删除aclnn_mm_white_list.h和op_util.h
- 缺陷描述：conv infershape使用ops::Shape2String(被删除的op_util.h)，需替换为基础库Ops::Base::ToString
- 修复模式：API调用替换+冗余文件清理
- 可审查性：中
- 审查规则建议：op_host层代码不应依赖op_api层工具函数

### 8a4c75d1 fix advance_step atk
- 根因类别：tiling参数未设置
- 涉及文件：optim/advance_step/op_host/advance_step_tiling.cpp
- 缺陷描述：tilingKey==1分支中needCoreNum未设置，kernel侧使用未初始化值
- 修复模式：补充set_needCoreNum(this->aivNum_)
- 可审查性：高
- 审查规则建议：所有tiling分支需检查必要参数是否都已设置

### e6415492 kv_rms_norm_rope_cache bug fix
- 根因类别：平台适配文件缺失
- 涉及文件：norm/kv_rms_norm_rope_cache/op_kernel/arch35/platform.h(新增)
- 缺陷描述：缺少arch35平台适配头文件，导致编译失败或使用错误的平台参数
- 修复模式：新增arch35/platform.h
- 可审查性：中
- 审查规则建议：新增算子时检查是否覆盖所有目标平台的platform适配文件

### 73f7ece2 celossgrad fix
- 根因类别：命名空间迁移适配
- 涉及文件：loss/cross_entropy_loss_grad/op_kernel/arch35/cross_entropy_loss_grad_weight_not_none.h
- 缺陷描述：Ops::Base::CeilAlign/CeilDiv调用需改为ops::CeilAlign/CeilDiv
- 修复模式：命名空间前缀更新
- 可审查性：低
- 审查规则建议：基础库API命名空间变更时全量扫描调用方

### 0cb33856 fix rmsnormgrad oom
- 根因类别：tail block DMA拷贝长度错误导致OOM
- 涉及文件：norm/rms_norm_grad/op_kernel/arch35/rms_norm_grad_regbase_dgamma.h
- 缺陷描述：tail block处理时CopyInputsToUB使用rowsPerUB_(最大行数)而非实际尾部行数rows_%rowsPerUB_，DMA多拷数据导致UB溢出
- 修复模式：tailRowsNum计算提前到DMA拷贝阶段
- 可审查性：高
- 审查规则建议：主循环+tail模式中，tail block的DMA长度必须使用tail尺寸而非主循环尺寸

### b98f37a9 PR98遗留问题解决
- 根因类别：逻辑运算符优先级缺陷
- 涉及文件：matmul/batch_mat_mul_v3/op_kernel/batch_mat_mul_v3_block.h等5个文件
- 缺陷描述：`nd2nzFlag_ == ONLY_A || BOTH_AB`缺少第二个==，|| BOTH_AB永远为true(非零值)
- 修复模式：`nd2nzFlag_ == ONLY_A || nd2nzFlag_ == BOTH_AB`
- 可审查性：高
- 审查规则建议：检测`var == VALUE1 || VALUE2`模式，几乎总是逻辑错误

### 791575cd 修复rmsNormQuant的UB越界问题
- 根因类别：tiling切片常量过大
- 涉及文件：norm/rms_norm_quant/op_host/rms_norm_quant_common_tiling.cpp
- 缺陷描述：无bias场景SLICE_SIZE=12288在某些shape下超出UB容量限制，统一为8192
- 修复模式：缩小SLICE_SIZE使其不超过UB容量上限
- 可审查性：高
- 审查规则建议：SLICE_SIZE常量应有注释说明与UB容量的关系，需静态断言slice_size*element_size*buffer_count<=UB_SIZE

### 57a8c64f 修复aclnnGroupNormBackward接口fuzz泛化个别用例aicore异常507015
- 根因类别：workspace大小计算int32溢出
- 涉及文件：norm/group_norm_grad/op_host/group_norm_grad_tiling.cpp
- 缺陷描述：workSpaceSize是int32，乘法在int32域内执行，n和c较大时溢出。需先static_cast<int64_t>再乘
- 修复模式：乘法操作数提升为int64_t
- 可审查性：高
- 审查规则建议：size/workspace/buffer计算中int32*int32必须至少一个提升为int64_t

### 0c1846c9 bugfix stride和dilation pad d维度拦截
- 根因类别：参数校验上限常量混用
- 涉及文件：conv/conv3d_backprop_filter_v2/op_host/op_tiling/arch32/conv_backprop_filter_context_utils.cpp
- 缺陷描述：Conv3D反向的stride_d/dilation_d/pad_f/pad_b使用H/W维度上限(STRIDE_UPPER等)校验，但D维度合法范围更大应用SHAPE_UPPER
- 修复模式：D维度校验改用通用上限常量
- 可审查性：高
- 审查规则建议：3D卷积D/H/W三维度参数校验应使用各自对应的上限常量

### 06d13589 opkernel ut json fix
- 根因类别：header-only库链接方式错误
- 涉及文件：tests/ut/op_kernel/CMakeLists.txt
- 缺陷描述：json是header-only库不需链接，从target_link_libraries改为add_dependencies
- 修复模式：header-only库从link依赖改为构建顺序依赖
- 可审查性：中
- 审查规则建议：header-only库不应出现在target_link_libraries中

### af90af68 bugfix
- 根因类别：printf格式符类型不匹配
- 涉及文件：conv/conv3d_backprop_filter_v2/op_host/op_tiling/arch32/conv_backprop_filter_context_utils.cpp
- 缺陷描述：OP_LOGE中%s对应std::string对象而非const char*，导致未定义行为
- 修复模式：添加.c_str()
- 可审查性：高
- 审查规则建议：printf风格宏的%s参数检查是否传入std::string而非const char*（-Wformat检测）

### 23cdff84 fix bianyi
- 根因类别：结构体成员类型不匹配
- 涉及文件：conv/convolution_backward/op_host/op_api/convolutionbackward.h
- 缺陷描述：groups字段int64_t与下层API期望int不匹配
- 修复模式：int64_t -> int
- 可审查性：中
- 审查规则建议：结构体参数类型应与传递目标API参数类型一致

### cbf1d6eb fix foreach_non_finite include head file failed
- 根因类别：目录结构不符合构建规范
- 涉及文件：foreach/foreach_non_finite_check_and_unscale下9个文件
- 缺陷描述：使用旧名graph_plugin而非新规范op_graph，并缺少ascend910_95平台配置
- 修复模式：目录重命名+补充平台配置
- 可审查性：中
- 审查规则建议：目录遵循op_graph命名约定

### a89df355 fix:单核切K
- 根因类别：splitK场景bias残留
- 涉及文件：matmul/mat_mul_v3/op_kernel/arch35/mat_mul_asw_block.h, mat_mul_asw_kernel.h
- 缺陷描述：splitK多轮K循环中只在最后一轮SetBias，其他轮未ClearBias，残留bias数据影响累加结果导致精度失败
- 修复模式：新增UpdateBias方法：最后一轮SetBias，其他轮ClearBias
- 可审查性：高
- 审查规则建议：splitK/多轮迭代中检查bias是否每轮正确设置或清除；SetBias有条件调用则必须有ClearBias分支

### 16371af6 fix_same_funtion
- 根因类别：重复函数定义冲突
- 涉及文件：conv/common/op_host/conv_forward_infershape.cpp, cube_util.h
- 缺陷描述：cube_util.h中IsUnknownShape/IsUnknownRank与ops命名空间同名函数冲突
- 修复模式：删除重复定义，使用公共库版本
- 可审查性：中
- 审查规则建议：自定义utility头文件检查是否有与公共库同名函数

### b8bfebd5 fix single_op script error
- 根因类别：Shell脚本语法错误
- 涉及文件：scripts/kernel/binary_script/build_binary_single_op_gen_task.sh
- 缺陷描述：(1)[! -z ...]缺空格，正确为[ ! -z ... ]；(2)命令存入变量后$(${cmd})执行不可靠
- 修复模式：修复方括号空格；直接内联$(find ...)
- 可审查性：高
- 审查规则建议：推荐shellcheck静态检查；[ ]内侧必须有空格

### 68a44620 修复kernel找不到act头文件的问题
- 根因类别：编译include路径缺失
- 涉及文件：scripts/util/ascendc_impl_build.py
- 缺陷描述：AscendC kernel编译缺少act头文件目录的-I路径
- 修复模式：补充-I指向tikcpp/../ascendc/act
- 可审查性：中
- 审查规则建议：新增kernel依赖头文件目录时同步更新构建脚本include路径

### 20877724 修复非量化mmv3 fp32场景k很大精度失败
- 根因类别：tiling策略选择不当
- 涉及文件：matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_asw_basic_tiling.cpp
- 缺陷描述：fp32+大K(>8192)+ND格式时basic api tiling无法正确处理导致精度问题。需在IsCapable中排除该场景回退到更合适的策略
- 修复模式：新增FP32_SPLIT_K_THRESHOLD=8192UL，IsCapable返回false让该场景回退
- 可审查性：高
- 审查规则建议：tiling IsCapable应覆盖所有已知的精度风险shape组合

### bef75e8a DTS2025082812507 问题单修复
- 根因类别：编译告警批量修复
- 涉及文件：11个文件(op_legacy_api, matmul_util, aclnn_matmul等)
- 缺陷描述：未使用参数/变量告警、有符号/无符号比较告警
- 修复模式：移除unused参数/变量，static_cast消除sign compare
- 可审查性：中
- 审查规则建议：-Wunused-parameter -Wunused-variable -Wsign-compare

### 5113cf57 fix_kernel_build_fail_but_return_build_success
- 根因类别：构建失败被静默忽略
- 涉及文件：scripts/kernel/binary_script/gen_output_json.py
- 缺陷描述：kernel编译未生成输出json时仅WARNING+continue，构建返回成功。失败被吞掉
- 修复模式：WARNING改ERROR，continue改raise FileNotFoundError()
- 可审查性：高
- 审查规则建议：构建脚本检测产物缺失时不应continue/pass跳过，必须返回非零或抛异常

### 156fbaeb fix new ge so
- 根因类别：上游依赖拆分后链接缺失
- 涉及文件：cmake/func.cmake, cmake/modules/Findmetadef.cmake, 多个CMakeLists.txt
- 缺陷描述：新版GE SO拆分后需额外查找并链接error_manager/rt2_registry_static/metadef库
- 修复模式：补充CMake中新版GE依赖的库查找和链接
- 可审查性：中
- 审查规则建议：上游依赖拆分/重组时全量检查下游CMakeLists链接依赖

### 366657c4 修复子包打包控制变量未生效问题
- 根因类别：Shell变量初始化逻辑错误
- 涉及文件：CMakeLists.txt, build.sh, cmake/package.cmake
- 缺陷描述：build.sh用$OPTIND判断是否有参数($OPTIND是getopts内部变量初始值为1不代表无参数)，任何场景都触发子包打包
- 修复模式：$OPTIND改为$#判断参数个数；分离构建控制路径
- 可审查性：高
- 审查规则建议：shell中用$#而非$OPTIND判断参数存在性

### f9c92cc8 修复子包和自定义算子包cmake target重名问题
- 根因类别：CMake target命名冲突
- 涉及文件：CMakeLists.txt
- 缺陷描述：子包打包和自定义算子包使用相同target名称导致冲突
- 修复模式：给子包target添加pkg_前缀
- 可审查性：高
- 审查规则建议：CMake中不同功能模块target应有命名前缀约定防止重名

跳过：6e514413(clean code-static→inline), 1c32f798(代码重构), 63b25963(新特性迁移), 4f9d6758(Revert留阶段3), 9fbd3595(Revert留阶段3), 0861ed29(Revert留阶段3), 80df2809(README), 5b4f67ce→已分析, 1b9d36af(文档修复), 79623db1(Revert留阶段3), 43baa6f8(Revert留阶段3), 1536ebfc(文档/注释), a93b20d7(新特性), cf43fc65(目录重组), 0ded3d43(README), 0c283c09(README), ca466912(代码迁移), c2b0c9ff(示例), a7973807(新功能), 3c583402(拼写), 785e2572(Revert留阶段3), a5ca9cf8(OAT license), 9de02749(Revert留阶段3), 8558f785(README)
