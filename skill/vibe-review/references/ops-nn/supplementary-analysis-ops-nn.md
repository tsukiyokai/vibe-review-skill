# ops-nn 补充分析

数据来源: ops-nn主仓(1474提交, 380条缺陷) + ops-nn-dev(2571提交, 612条缺陷)

---

## Part 1: 热点文件风险分析

### ops-nn主仓


### 热点文件统计

从380条缺陷提交的defect_analysis.md中提取被修复触及的文件频次（仅含具体路径）：

| 排名 | 文件 | 触及次数 | 模块 |
|------|------|---------|------|
| 1 | build.sh | 10 | 构建 |
| 2 | cmake/ut.cmake | 9 | 构建/测试 |
| 3 | cmake/gen_ops_info.cmake | 8 | 构建 |
| 4 | tests/ut/op_host/CMakeLists.txt | 5 | 测试 |
| 5 | scripts/kernel/binary_config/ascendc_config.json | 4 | 配置 |
| 5 | matmul/quant_batch_matmul_v4/.../aclnn_quant_matmul_v5.cpp | 4 | matmul/op_api |
| 5 | matmul/common/op_host/op_api/matmul_util.cpp | 4 | matmul/op_api |
| 8 | matmul/mat_mul_v3/op_kernel/mat_mul_deterministic_splitk_kernel.h | 3 | matmul/kernel |
| 8 | matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp | 3 | matmul/tiling |
| 8 | matmul/mat_mul_v3/op_host/op_api/aclnn_addmm.cpp | 3 | matmul/op_api |
| 8 | matmul/batch_mat_mul_v3/op_host/op_api/aclnn_addbmm.cpp | 3 | matmul/op_api |
| 8 | conv/convolution_backward/op_api/aclnn_convolution_backward.cpp | 3 | conv/op_api |
| 8 | CMakeLists.txt | 3 | 构建 |
| 8 | cmake/variables.cmake | 3 | 构建 |
| 8 | cmake/func.cmake | 3 | 构建 |
| 8 | scripts/kernel/binary_script/build_binary_single_op_gen_task.sh | 3 | 构建脚本 |

热点分布特征：
- 构建系统(build.sh + cmake + 脚本)占据Top3，且总触及次数达30+，是ops-nn最大的缺陷聚集区
- matmul模块有6个文件上榜，覆盖tiling/kernel/op_api三层
- conv仅1个文件上榜但是最大的单文件(2500+行)

---

### 一、build.sh 结构性风险（10次触及）

#### 实际Bug

1. `]`前缺少空格（行1278）:
```bash
[ -f ${EAGER_LIBRARY_OPP_PATH}/libascendcl.so]
```
右括号与文件名粘连，`test`命令寻找名为`libascendcl.so]`的文件，条件永远为false。

2. 数组备份/恢复错误（行1060-1065）:
```bash
local TMP_BUILD_LIBS=${BUILD_LIBS}     # 只取数组第一个元素
# ... build_binary_one执行后 ...
BUILD_LIBS=${TMP_BUILD_LIBS}           # 恢复时变成字符串
```
`BUILD_LIBS`是数组，`${BUILD_LIBS}`只取首元素。多target构建时BUILD_LIBS数据丢失。

3. `--mssanitizer`选项设为FALSE（行796）:
```bash
mssanitizer) ENABLE_MSSANITIZER=FALSE ;;
```
启用选项却赋值FALSE，该flag无效化。

4. sed空操作（行779）:
```bash
COMPUTE_UNIT=$(echo "$COMPUTE_UNIT" | sed 's/ascend950/ascend950/g')
```
替换前后相同，是遗留的死代码。

#### 环境安全风险

5. ASCEND_HOME_PATH/ASCEND_OPP_PATH无检查（行129-139）: 未设置时所有路径变成`/include`、`/lib64`等根目录。

6. `set +u`从未恢复（行35）: 整个脚本运行期间未定义变量静默为空，拼写错误的变量名不会报错。

7. LD_LIBRARY_PATH追加时可能以冒号开头（行1069等）: `$LD_LIBRARY_PATH:`前缀为空时包含当前目录。

8. source使用相对路径（行33）: `source "./install_deps.sh"`在非脚本目录执行时加载错误文件。

#### 其他风险

9. `$(rm -f asan_test)`命令替换误用（行943）
10. `gen_op()`缺少python未找到时的else分支（行1363-1385）
11. `dirname $0`未加引号（行118）
12. `rm -rf $BUILD_OUT_PATH`未加引号（行600）
13. CMAKE_ARGS字符串拼接（非数组），参数含空格时错误拆分
14. main通过管道传给gawk，exit在subshell中执行（行1531）
15. CORE_NUMS赋值后从未使用（行125，死代码）

---

### 二、CMake文件结构性风险（cmake/ut.cmake 9次 + gen_ops_info.cmake 8次 + 其他）

#### 实际Bug

1. `compile_from_config`使用循环后遗留的`op_name`（gen_ops_info.cmake:654）:
```cmake
compile_from_config(TARGET ascendc_bin_${compute_unit}_${op_name} ...)
```
在foreach循环结束后调用，`${op_name}`仅为最后一次迭代的值。

2. `CUSTOM_TILING_DATA_KEYS`先用后赋值（ut.cmake:571 vs 587）:
```cmake
set(gen_cmd "... ${CUSTOM_TILING_DATA_KEYS}")  # 行571，先用
# ...
set(CUSTOM_TILING_DATA_KEYS "")                 # 行587，后赋值
```

3. `target_dir`空值判断缺引号（ut.cmake:354）:
```cmake
if(${target_dir} STREQUAL "")    # 空时展开为 if( STREQUAL "")
```

4. `OPS_TEST_DIR`未定义即使用（ut.cmake:51）

#### 配置/依赖风险

5. `ARCH`变量在Unknown架构时未设置（variables.cmake:27-33）: else分支只打印WARNING不设默认值。

6. `MODULE_EXT`未在parse_arguments中声明（func.cmake:584）: 分支永远不执行（dead code或功能缺陷）。

7. `kernel_src_copy`接受了未声明的`OP_LIST`参数（gen_ops_info.cmake:14-17）: 参数被静默忽略。

8. `atvoss_src_copy`target无条件重建（gen_ops_info.cmake:50-60）: 多次调用时target重复定义。

9. gen_ops_info.cmake被include三次无guard（CMakeLists.txt:144/158/168）

10. `map_compute_unit`与`get_target_dir`映射表硬编码且仅覆盖5/12个芯片版本（func.cmake:695/710）

11. `_GLIBCXX_USE_CXX11_ABI`在UT模块间不一致: AICPU=1 vs 其他=0，可能导致ABI兼容性问题

12. `add_version_info_targets`错误消息使用了不存在的`${pkg_name}`（func.cmake:899-907）

---

### 三、matmul模块结构性风险（6个文件上榜）

#### 空指针/先用后查

1. self先Contiguous后才检查null（aclnn_addmm.cpp:298-299）:
```cpp
auto selfContiguous = l0op::Contiguous(addmmTensor.self, uniqueExecutor);
if (addmmTensor.self != nullptr && ...)  // 检查太晚
```

2. InplaceAddmm未检查selfRef为null（aclnn_addmm.cpp:729）

3. InplaceAddbmm未检查selfRef为null（aclnn_addbmm.cpp:547）

4. AllocTensor结果未检查null即调用IsEmpty()（matmul_util.cpp:214-225）

#### CHECK_RET检查错误变量

5. 赋值给reformatedX1但检查x1（aclnn_quant_matmul_v5.cpp:907-908）:
```cpp
reformatedX1 = l0op::TransData(reformatedX1, Format::FORMAT_FRACTAL_NZ, 0, executor);
CHECK_RET(x1 != nullptr, ACLNN_ERR_INNER_NULLPTR);  // 应检查reformatedX1
```

#### 除零/溢出

6. baseM/baseK/dtypeSize连续除法可能除零（matmul_v3_base_tiling.cpp:605-606）

7. nCoreTail=0时AlignTo256B(0)=0导致除零（mat_mul_deterministic_splitk_kernel.h:80-81）

8. baseM*baseN乘法可能溢出（matmul_v3_base_tiling.cpp:1366-1367）

9. uint64->uint32截断无保护（matmul_v3_base_tiling.cpp:909-912）

#### Copy-paste / 逻辑错误

10. rowBlockNum和colBlockNum公式相同（mat_mul_deterministic_splitk_kernel.h:404-405）:
```cpp
uint64_t rowBlockNum = alignedN / 16;    // 应为alignedM / 16?
uint64_t colBlockNum = alignedN / 16;
```
疑似copy-paste bug，rowBlockNum应基于alignedM。

11. 注释与条件判断矛盾（aclnn_addbmm.cpp:427-432）: 条件判断`alpha == 1.0`但注释写`alpha == 0`。

12. SELF_MIN=0导致维度检查失效（aclnn_addbmm.cpp:147-148）: self为0维/1维时GetDim(1)越界。

13. batch2用batch1DimNum索引可能越界（matmul_util.cpp:1663）

#### 类型安全

14. const_cast修改入参dtype（aclnn_quant_matmul_v5.cpp:813-831）: INT64改为UINT64的语义差异在负值场景致命。

15. union type punning是C++ UB（matmul_util.cpp:2099-2107）

16. bf16与half sizeof相同走同一Cast路径（mat_mul_deterministic_splitk_kernel.h:100-102）: RoundMode可能不适用于bf16。

---

### 四、conv模块结构性风险（aclnn_convolution_backward.cpp 3次触及）

#### 空指针/返回值忽略

1. Cast后先LOG再CHECK（行363-368）:
```cpp
l0ResultTensor = l0op::Cast(...);
OP_LOGD("...", l0ResultTensor->ToString()...);  // Cast返回null时崩溃
OP_CHECK(l0ResultTensor != nullptr, ...);        // 检查太晚
```

2. InputPreProcess返回值被忽略x3（行1014-1016）: 其他位置都用CHECK_RET包裹。

3. CalculateBiasGrad返回值被忽略（行1312）

#### 其他

4. 栈数组append_dim未完全初始化（行510-514）: inputDim>=4时循环不执行，数组内容未定义。

5. 错误日志与实际函数不匹配（行2228/2361）: Conv3D的代码打印"Conv2dBackpropInput"。

6. promoteType可能未初始化（行2300-2324）: outputMask[0]和[1]同时为false时。

---

### 五、ascendc_config.json 结构性风险（4次触及）

#### 完全重复条目

| 算子 | 重复行号 |
|------|----------|
| AdaptiveAvgPool3d | 行2 vs 行5 |
| DeformableOffsetsGrad | 行275 vs 行294 |
| LayerNormQuant | 行571/600/601 (三份) |
| GetPaddingOffset | 行572/604/622 (三份) |

#### 同名但配置冲突的条目

| 算子 | 差异 |
|------|------|
| AdaptiveAvgPool3dGrad | compute_units含/不含ascend950 |
| AdaptiveMaxPool3d | compute_units含/不含ascend950 |
| STFT | compute_units含/不含ascend910_93 |
| Split/SplitV | impl_mode有/无 |
| DeformableOffsets | compile_options有/无 |

哪一条生效取决于JSON解析器实现——构建行为不确定。

#### 无效配置

compile_options中引用`ascend910_95`（行283/363）: 不在compute_units列表中也非已知平台标识，疑似`ascend910_93`或`ascend950`的拼写错误。

---

### 六、tests/ut/op_host/CMakeLists.txt 结构性风险（5次触及）

1. ENABLE_ASAN/ENABLE_VALGRIND未定义时直接STREQUAL比较（行98/114）: 展开为空导致cmake报错。

2. ASAN LD_PRELOAD路径硬编码x86_64（行101-102）: 昇腾设备通常为aarch64，路径不存在。

3. ENABLE_UT_EXEC和ENABLE_VALGRIND可能同时为TRUE（行97-120）: 缺乏互斥检查，POST_BUILD执行两次。

4. SKIP_BUILD_RPATH=TRUE（行64-66）: 依赖环境变量LD_LIBRARY_PATH，未设置时运行时链接失败。

---

### 风险汇总与优先级

#### P0 (确认的Bug)

| # | 位置 | 描述 |
|---|------|------|
| 1 | build.sh:1278 | `]`前缺空格，条件永远false |
| 2 | build.sh:1060-1065 | 数组备份/恢复错误 |
| 3 | gen_ops_info.cmake:654 | 循环后遗留变量作为target名 |
| 4 | ut.cmake:571/587 | CUSTOM_TILING_DATA_KEYS先用后赋值 |
| 5 | aclnn_addmm.cpp:298-299 | 空指针先用后查 |
| 6 | aclnn_quant_matmul_v5.cpp:907-908 | CHECK_RET检查错误变量 |
| 7 | mat_mul_deterministic_splitk_kernel.h:404-405 | rowBlockNum/colBlockNum公式相同(疑似copy-paste) |
| 8 | aclnn_convolution_backward.cpp:363-368 | Cast后先LOG再CHECK |

#### P1 (高风险)

| # | 位置 | 描述 |
|---|------|------|
| 9 | build.sh:129-139 | ASCEND_HOME_PATH无检查 |
| 10 | build.sh:796 | --mssanitizer设为FALSE |
| 11 | matmul_v3_base_tiling.cpp:605-606 | 连续除法潜在除零 |
| 12 | mat_mul_deterministic_splitk_kernel.h:80-81 | nCoreTail=0除零 |
| 13 | aclnn_addbmm.cpp:547 | InplaceAddbmm未检查null |
| 14 | aclnn_addmm.cpp:729 | InplaceAddmm未检查null |
| 15 | ascendc_config.json多处 | 同名不同配置冲突 |
| 16 | aclnn_convolution_backward.cpp:1014-1016 | InputPreProcess返回值忽略x3 |

#### P2 (中风险)

| # | 位置 | 描述 |
|---|------|------|
| 17 | ut.cmake:354 | target_dir空值判断缺引号 |
| 18 | matmul_v3_base_tiling.cpp:909-912 | uint64->uint32截断 |
| 19 | aclnn_addbmm.cpp:147-148/227-230 | SELF_MIN=0+GetDim越界 |
| 20 | matmul_util.cpp:1663 | batch索引用错DimNum |
| 21 | aclnn_quant_matmul_v5.cpp:813-831 | const_cast修改入参dtype |
| 22 | ascendc_config.json:283/363 | ascend910_95疑似拼写错误 |
| 23 | variables.cmake:27-33 | ARCH未设置默认值 |
| 24 | func.cmake:695/710 | 芯片映射表仅覆盖5/12 |

---

### ops-nn-dev


### 一、缺陷热点统计

#### 1.1 总览

- 612条缺陷提交共触及3844个唯一文件
- 80.5%的文件(3095个)仅被1个缺陷提交触及
- 749个文件被多次触及(>=2次)，是真正的缺陷热点

#### 1.2 按模块聚合

| 模块 | 缺陷触及总次数(文件*提交) | 特征 |
|------|--------------------------|------|
| foreach | 1452 | 58个算子子目录，热点极度分散，单文件<=4次 |
| matmul | 1185 | 集中在batch_matmul/quant_matmul的tiling和api层 |
| conv | 562 | 集中在conv2d/conv3d的infershape和tiling层 |
| norm | 435 | group_norm_silu和rms_norm_quant为热点 |
| activation | 246 | 分散 |
| index | 233 | scatter和embedding为热点 |
| quant | 192 | flat_quant的kernel代码为热点 |
| pooling | 183 | adaptive_max_pool3d为热点 |

#### 1.3 Top 30 业务代码热点文件

排除构建/配置文件(build.sh, cmake/*, classify_rule.yaml, ascendc_config.json等)后：

| 触及次数 | 文件路径 | 模块 |
|---------|---------|------|
| 9 | matmul/batch_mat_mul_v3/op_host/op_api/aclnn_batch_matmul.cpp | matmul |
| 8 | matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp | matmul |
| 8 | matmul/common/op_host/op_api/matmul_util.cpp | matmul |
| 8 | matmul/common/op_host/matmul_common_infershape.cpp | matmul |
| 8 | conv/convolution_forward/op_host/op_api/aclnn_quant_convolution.cpp | conv |
| 7 | matmul/quant_batch_matmul_v4/op_host/op_tiling/arch35/quant_batch_matmul_v4_tiling.cpp | matmul |
| 7 | matmul/quant_batch_matmul_v4/op_host/op_api/aclnn_quant_matmul_v5.cpp | matmul |
| 7 | matmul/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling.cpp | matmul |
| 7 | matmul/mat_mul_v3/tests/ut/op_host/test_matmul_v3_tiling.cpp | matmul |
| 7 | conv/convolution_forward/op_host/op_api/aclnn_convolution.cpp | conv |
| 6 | quant/flat_quant/op_kernel/flat_quant_cube.h | quant |
| 6 | matmul/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling.cpp | matmul |
| 6 | matmul/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling_base.h | matmul |
| 6 | matmul/mat_mul_v3/op_host/op_api/aclnn_matmul.cpp | matmul |
| 6 | conv/common/op_host/conv_forward_infershape.cpp | conv |
| 5 | quant/flat_quant/op_kernel/tensor_utils.h | quant |
| 5 | matmul/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_apt.cpp | matmul |
| 5 | matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_reg_base_common.h | matmul |
| 5 | matmul/weight_quant_batch_matmul_v2/op_host/op_api/aclnn_weight_quant_batch_matmul_v2.cpp | matmul |
| 5 | matmul/quant_batch_matmul_v4/op_host/op_tiling/quant_batch_matmul_v4_msd_tiling.cpp | matmul |
| 5 | matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_apt.cpp | matmul |
| 5 | matmul/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling_base.cpp | matmul |
| 5 | matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_tiling.cpp | matmul |
| 5 | conv/conv3d_backprop_input_v2/op_host/op_tiling/conv3d_backprop_input_v2_tiling.cpp | conv |
| 5 | conv/conv3d_backprop_input_v2/op_host/op_tiling/common/conv_backprop_input_context_utils.cpp | conv |
| 5 | conv/conv3d_backprop_input_v2/op_host/op_tiling/arch35/conv3d_backprop_input_v2_inner_product_tiling.cpp | conv |
| 5 | conv/conv3d_backprop_input_v2/op_host/op_tiling/arch32/conv3d_backprop_input_v2_base_tiling.cpp | conv |
| 5 | conv/conv3d_backprop_filter_v2/op_host/op_tiling/arch35/conv3d_backprop_filter_v2_basic_block_tiling.cpp | conv |
| 5 | conv/conv2d_v2/op_host/op_tiling/arch35/conv2d_v2_api_tiling.cpp | conv |
| 5 | matmul/batch_mat_mul_v3/tests/ut/op_host/test_batch_mat_mul_v3_tiling.cpp | matmul |

matmul模块占Top 30中的20席，是绝对的缺陷震中。

#### 1.4 根因类别频次(从defect_analysis.md提取，归一化后)

| 排名 | 频次 | 占比 | 根因类别 |
|---:|---:|---:|:---|
| 1 | 37 | 7.3% | 条件判断/控制流缺陷 |
| 2 | 37 | 7.3% | 构建系统/CMake缺陷 |
| 3 | 37 | 7.3% | 编译告警/代码规范 |
| 4 | 29 | 5.7% | 数据类型缺陷 |
| 5 | 28 | 5.5% | 内存/buffer管理缺陷 |
| 6 | 24 | 4.7% | 边界条件处理缺陷 |
| 7 | 24 | 4.7% | 整数溢出 |
| 8 | 23 | 4.5% | 计算逻辑错误 |
| 9 | 21 | 4.1% | 平台适配缺陷 |
| 10 | 20 | 3.9% | 配置遗漏 |
| 11 | 18 | 3.5% | API/接口缺陷 |
| 12 | 17 | 3.3% | 流水线/硬件同步缺陷 |
| 13 | 17 | 3.3% | UT/测试缺陷 |
| 14 | 16 | 3.1% | 复制粘贴错误 |
| 15 | 14 | 2.8% | 日志/错误处理缺陷 |
| 16 | 13 | 2.6% | 空指针/资源安全 |
| 17 | 13 | 2.6% | tiling逻辑缺陷 |
| 18 | 11 | 2.2% | include路径错误 |
| 19 | 10 | 2.0% | 输入校验缺陷 |
| 20 | 7 | 1.4% | 功能回退/Revert |
| 21 | 7 | 1.4% | 命名/拼写错误 |
| 22 | 5 | 1.0% | 代码搬迁/迁移缺陷 |
| 23 | 4 | 0.8% | 脚本缺陷 |

Top 3并列(各37次, 7.3%)：条件判断/控制流、构建系统/CMake、编译告警/代码规范。

---

### 二、热点文件结构性风险审查

#### 2.1 matmul模块 (Top缺陷震中)

##### aclnn_batch_matmul.cpp (9次触及)

[严重] 死代码 -- 空tensor分支永不生效(约282-293行)
- `CreateBatchMatmulGraphImpl`中先判断空tensor创建`BatchmmEmptyTensorGraph`，但紧接着无条件覆盖为`BatchMatmulExecBmmOpGraph`，缺少`else`
- 空tensor输入走正常计算路径，可能产生非预期行为

[中等] 数据类型支持遗漏
- DTYPE_SUPPORT_LIST仅含FLOAT/FLOAT16/BF16，缺INT8等量化类型

##### matmul_v3_base_tiling.cpp (8次触及)

[低等] GetTotalSize(约1764-1770行)溢出风险已消除
- `m * k * aDtype`所有操作数均为uint64_t，乘法在uint64_t域完成，不存在int32截断问题

[中等] tiling参数边界 -- baseMNK为空时越界(约1556-1585行)
- `minLoadSize = -1`(UINT64_MAX哨兵值)，但`calBaseMNK`为空时`calBaseMNK[0]`越界

[低等] 复制粘贴 -- CheckMMTilingDataIsVaild(约2666-2668行)
- 日志字符串与实际校验字段不匹配：`CheckNumberIsValid(runInfo_.stepM, ..., "runInfo_.baseK")`

##### matmul_util.cpp (8次触及)

[中等] 整数除法后ceil无效(约356行)
- `std::ceil(mDim / align128)` -- 整数除法先截断再ceil，结果等于`mDim / align128`，应用CeilDiv

[中等] CheckBatchDimBroadcast索引越界(约1600行)
- 用`batch1DimNum`作为`batch2`的索引偏移，`batch1DimNum != batch2DimNum`时可能越界

[低等] 代码重复 -- ExecMmOpWithBias vs MatmulCommonProcess(约1094-1366行)
- 大段重复逻辑，修改一处易遗漏另一处

##### quant_batch_matmul_v4_tiling.cpp (7次触及)

[严重] 复制粘贴 -- 错误日志打印错误dtype(约170-172行)
- 校验y(output)的dtype时，错误信息打印`bDtype`(B矩阵类型)而非`cDtype`(输出类型)

[中等] batch累乘无溢出保护(约445行)

##### aclnn_quant_matmul_v5.cpp (7次触及)

[中等] 平台扩展性 -- A8W4Float硬编码仅支持DAV_3510(约538-548行)
- 新NPU架构需手动添加else分支

[低等] const_cast修改输入tensor类型(约345-346行)
- 校验函数中通过`const_cast`修改输入tensor的dtype，违反函数语义契约

##### quant_batch_matmul_v3_tiling.cpp (7次触及)

[中等] 整数溢出 -- int32乘法计算L1/L0C空间(约1072-1074行)
- `mt.baseM * mt.baseK * dtypeSize`三个int32相乘，base块较大时溢出

[中等] SpiltForWorkSpaceLimit除零(约1296行)
- `WORKSPACE_LIMIT / blockDim`无blockDim==0保护

[低等] CalcUsedL1AndUBSize参数类型为int32_t(约1115行)
- L1大小在某些平台可能超INT32_MAX

##### weight_quant_batch_matmul_v2_tiling.cpp (6次触及)

[严重] 类型转换导致比较错误(约248行)
- `static_cast<int64_t>(fusedDimValue)` -- fusedDimValue是uint64_t，超INT64_MAX时转为负数，比较`> MAX_INT32`结果为false，漏检

[中等] GetBatchSize中batch累乘无溢出检查(约96行)

##### aclnn_matmul.cpp (6次触及)

[中等] size_t减法下溢(约147行)
- `size_t loopDims = dimNum - 2` -- dimNum < 2时下溢为极大值

[中等] CheckWeightNzStorageShape维度乘积溢出(约418-428行)

#### 2.2 conv模块

##### aclnn_convolution.cpp (7次触及)

[严重] All函数实现调用了Any(行266-274)
- 模板函数`All`递归调用`Any`而非`All`自身
- `CHECK_PARAM_ALL_EQ`和`CHECK_PARAM_ALL_GTE`宏语义错误：只检查第一个参数和任一后续参数

[中等] ConstructPad静默丢弃异常pad值(行604-628)
- `oldPad.size()`不符合预期时pad被设为0，无warning/error

[中等] PointWiseKernelBeyondLimits中负维度乘uint64_t(行790-798)
- 动态shape场景下dim=-1乘以uint64_t产生巨大正数

##### aclnn_quant_convolution.cpp (8次触及)

[中等] 平台分支覆盖 -- TemporaryLimitChecker(行910-933)
- switch/case仅覆盖ASCEND910B和ASCEND910_93，新平台直接报错

[中等] 输出shape计算中stride除零依赖前置check(行345-359)
- check流程与计算流程耦合，重构时除零风险暴露

##### conv_forward_infershape.cpp (6次触及)

[中等] GetConvOutShapeRangeNeedInfer中stride<=0时outRange未赋值(行1087-1134)
- 只打log并return，outRange[idx]保留默认pair(0,0)，后续推导出错误shape range

#### 2.3 quant模块

##### flat_quant_cube.h (6次触及)

[中等] 精度损失(行89-94)
- `shape.K * shape.M`结果转float(23位尾数)再除法，值超2^24时精度丢失。int64_t溢出风险在flat_quant实际场景中较低

[中等] L1 buffer总量无运行时校验(行38-45)
- 5个tensor分配在L1中，无check总和是否超出L1_SIZE=512KB
- MM_DOUBLE_MODE下calM = 2*Mceil，buffer需求翻倍

[中等] 硬件同步事件ID复用(行466-471)
- 所有DEvent使用同一对EVENT_ID4/EVENT_ID5

##### tensor_utils.h (5次触及)

[中等] UB/L1大小硬编码(行44-54)
- `UB_SIZE = 192*1024`, `L1_SIZE = 512*1024`仅适配特定芯片

[中等] CalReduceMax尾部数据丢失(行177-188)
- `repeatTimes = len >> 7`截断余数，`len % 128 != 0`时尾部元素未参与ReduceMax

[中等] CalReduceMaxOne中mask为0(行230)
- `colSize % BASE_SIZE == 0`时Max的mask参数为0，硬件行为未定义

#### 2.4 norm模块

##### group_norm_silu_tiling.cpp (3次触及)

[严重] Lcm中a*b溢出(行73-76)
- `return a * b / Gcd(a, b)` -- 应改为`a / Gcd(a, b) * b`避免中间结果溢出

[严重] remainUbSize uint64_t下溢(行372-373)
- 减法结果为负时下溢成巨大正整数，导致maxProcessSize过大、UB溢出

[中等] loopNum用整数除法而非CeilDiv(行409)
- elemNum不能被processSize整除时尾块数据漏算

[中等] numGroups字段语义被复用(行388)
- `set_numGroups(gammaPerCoreRoundUp)`覆盖原始numGroups值

##### rms_norm_quant.cpp (3次触及)

[严重] beta搬入缺MTE2->V同步(行108-111)
- gamma搬入有完整SetFlag/WaitFlag，beta搬入缺少，可能读到未完成搬入的数据

[中等] 平台宏条件不一致(行188 vs 245)
- FastCompute用`__NPU_ARCH__ == 3003`，ComputeSquareSum用`__CCE_AICORE__ == 100`，覆盖不同芯片组合

[中等] calc_buf_偏移越界风险(行77, 181)
- OFFSET_WORKSPACE=3的偏移已超出BUF_FACTOR=3分配区域，仅靠+32字节余量

#### 2.5 foreach模块

foreach热点极度分散(58个算子子目录，单文件<=4次)，本质是批量模板代码同步修改。

[低等] infershape注册ops::前缀不一致(foreach_infershape.cpp)
- 前52个算子用`ops::InferShape4ForeachCommon`，后5个(ForeachSigmoid等)缺少`ops::`前缀
- 风格不一致反映批量添加时审查缺失

[低等] tiling测试占位断言
- test_foreach_abs/cos/atan_tiling.cpp中只断言tiling_key值，有`// todo check tiling result`注释，未验证tiling数据正确性

#### 2.6 index模块

##### aclnn_scatter.cpp (3次触及)

[中等] int64->int32窄化截断(行483等)
- `static_cast<int32_t>(shape.GetDim(0))`比较时，dim超INT32_MAX会截断导致比较错误

[低等] CheckDimRange边界逻辑与前置扩维有交叉依赖(行248-256)

##### aclnn_embedding_dense_backward.cpp (3次触及)

[中等] IsComputeByV2多阈值组合判断(行194-218)
- 涉及芯片版本/gradRow/embeddingDim/numWeights/scaleGradByFreq的复杂决策树，是边界条件bug温床

#### 2.7 pooling模块

##### adaptive_max_pool3d_big_pool.h (3次触及)

[中等] startIndex乘法溢出(行140-153)
- `idx * inLen`在高维场景下可能溢出int64_t

[中等] NaN处理复杂度高
- isnan通过reinterpret_cast手动拆解浮点位模式，fp16和fp32/bf16走不同分支

##### dequant_swiglu_quant_tiling_base.cpp (3次触及)

[中等] baseRowLen_*baseColLen_ uint32_t溢出(行415)
- 乘法在uint32_t域完成后才提升到uint64_t

[中等] GetBufferNumAndDataLenPerUB中quantMode/dtype分支不完整(行343-378)
- 不匹配任何已知dtype时singleDataSize默认为1，导致dataLenPerUB过大

---

### 三、结构性风险汇总

#### 3.1 按严重程度统计

| 严重程度 | 数量 | 分布 |
|---------|------|------|
| 严重 | 7 | matmul(2), conv(1), quant(0), norm(3), index(0), pooling(0), foreach(0) |
| 中等 | 28 | matmul(12), conv(4), quant(5), norm(3), index(2), pooling(2) |
| 低等 | 8 | matmul(4), conv(1), foreach(2), index(1) |

#### 3.2 风险模式聚合

| 风险模式 | 出现次数 | 典型文件 |
|---------|---------|---------|
| 整数溢出/精度损失(int32/uint32乘法、累乘无保护、float截断) | 11 | quant_batch_matmul_v3_tiling, flat_quant_cube.h, group_norm_silu_tiling |
| 条件分支不完整(平台/dtype/else缺失) | 6 | aclnn_quant_matmul_v5, aclnn_quant_convolution, dequant_swiglu_quant_tiling_base |
| 复制粘贴错误(日志字段/函数调用错误) | 4 | matmul_v3_base_tiling, quant_batch_matmul_v4_tiling, aclnn_convolution(All调Any) |
| buffer/内存边界(L1/UB无校验、偏移越界) | 4 | flat_quant_cube.h, rms_norm_quant, group_norm_silu_tiling |
| 硬件同步缺失/不一致 | 3 | rms_norm_quant(beta同步), flat_quant_cube.h(事件ID复用), rms_norm_quant(平台宏不一致) |
| 类型转换错误(窄化截断、uint->signed) | 3 | weight_quant_batch_matmul_v2_tiling, aclnn_scatter, aclnn_matmul |
| 除零保护不足 | 3 | conv_forward_infershape, quant_batch_matmul_v3_tiling, group_norm_silu_tiling |
| 死代码/语义错误 | 2 | aclnn_batch_matmul(空tensor分支), aclnn_convolution(All/Any混淆) |

#### 3.3 关键发现

1. matmul是ops-nn-dev的绝对缺陷震中：Top 30热点文件中占20席，量化矩阵乘(quant_batch_matmul_v3/v4, weight_quant_batch_matmul_v2)尤为集中。这与revert分析中matmul占45%的发现一致。

2. 整数溢出是当前代码中最普遍的残留风险(12处)，横跨matmul/conv/quant/norm四个模块。根因：int32类型的tiling参数相乘、batch维度累乘、buffer偏移计算均缺少溢出保护。

3. 8个严重风险中有4个可被静态分析工具发现：All调Any(类型检查)、Lcm溢出(模式匹配)、uint64_t下溢(unsigned减法检查)、beta同步缺失(配对分析)。

4. foreach模块虽然聚合触及次数最高(1452次)，但缺陷本质是"批量模板代码同步修改"，单个算子的结构性风险较低。真正需要重点review的是matmul和quant的kernel/tiling层。

5. 硬编码硬件参数(UB_SIZE, L1_SIZE)在quant/flat_quant中出现，是跨平台移植的定时炸弹。

---

## Part 2: Revert事件分析

### ops-nn主仓


共发现5个Revert提交，对应4个独立Revert事件。

### 事件总览

| # | Revert Hash | 原始提交 | 主题 | Revert间隔 | 根因类别 |
|---|------------|---------|------|-----------|---------|
| 1 | d91e1757a | 1962c3991 | unsorted_segment_sum算子迁移Ascend950 | ~30小时 | CI/集成失败 + 搭车提交 |
| 2 | 852f21d6b | ecd59361a | batchmatmul AB矩阵非连续输入 | 1天 | 变量遮蔽 + 空指针检查缺失 |
| 3 | 4482238c0 | 08b547c40 | LogSigmoid算子Ascend950实现 | ~13小时 | 目录命名/构建配置问题 |
| 4 | e248762f7 / a39e18548 | (init导入) | MatmulV3 small shape优化 | N/A(init即存在) | workspace脏数据(并发竞争) |

---

### 事件1: unsorted_segment_sum算子迁移

- Revert: `d91e1757ab41c3620057477c28adc62d395c2e00` (2026-03-07, cann-robot自动)
- 原始: `1962c3991dde8c03f1fd1e414583b94ae07c1a1a` (2026-03-06, Huang-Peng)
- MR: 原始 !1460, Revert !2421 (revert-mr-1460-auto)

#### 原始提交内容

新增UnsortedSegmentSum算子对Ascend950的支持，38个文件/8892行新增。包含算子定义、infershape、8种tiling策略(SIMT/SIMD/Sort/Deterministic等)、kernel实现、编译配置(binary.json 1769行)、测试。

#### Revert原因

cann-robot自动执行，分支名含`auto`后缀，大概率CI流水线检测到编译/集成测试失败后自动revert。原始MR仅声称"已通过冒烟测试"，在更全面的集成测试中失败。

#### 关键发现: 搭车提交

原始提交在`docs/zh/op_list.md`中添加的不是unsorted_segment_sum条目，而是scatter算子条目。这是典型的"搭车提交"反模式——不相关改动混入同一MR。Revert时scatter条目也被一并删除（附带伤害）。同时，unsorted_segment_sum自身的op_list条目反而遗漏未添加。

#### Code Review可发现性: 高

1. 不相关改动混入(scatter文档搭车) — 违反单一职责，reviewer应要求拆分
2. 自身算子文档遗漏 — 该添加的没添加，不该添加的却添加了
3. 单次提交8892行 — 体量过大应拆分
4. 测试覆盖声明不足 — 7种数据类型x8种tiling策略，仅声称"冒烟测试"通过

---

### 事件2: batchmatmul AB矩阵非连续输入

- Revert: `852f21d6ba12a54de4f47f30fb5337ebb67473ae` (2026-02-14, szhexin)
- 原始: `ecd59361a980ebfb6d3c09d76b05edaf26d8d783` (2026-02-13, shangzhexin)
- 修复重提交: `da61cacb9503d4f34f221a9576fc0600888368ac` (2026-02-28)
- Issue: 原始 #864, Revert #1088, 修复 #1119

#### 原始提交内容

为BatchMatMul算子增加A矩阵非连续输入支持（原先只支持B矩阵非连续）。引入NonContiguousMode枚举，修改tiling/kernel/op_api三层，18个文件/422行新增。在MatMulV3BasicTilingData结构体末尾新增innerBatch字段。

#### Revert原因: 两个代码缺陷

1. 变量遮蔽bug(variable shadowing): `ExecBatchMatmulOpWithBiasAndAttrsV2`函数中，AB_NON_CONTINUOUS分支的else路径写了`auto selfCast = l0op::Cast(...)`，用auto重新声明了局部变量，遮蔽了外部作用域的同名变量。结果: Cast操作结果被丢弃，后续使用了未经类型转换的tensor。同样问题出现在selfTransdata。

2. CreateView空指针检查缺失: 多处CreateView调用后缺少`CHECK_RET(... != nullptr, nullptr)`检查，CreateView失败时后续代码对nullptr调用方法导致crash。

这两个bug的组合导致: 不仅新增的非连续场景出错，连续输入的正常计算路径也受影响（selfCast跳过了Cast和TransData），产生计算结果错误或运行时崩溃。

#### Code Review可发现性: 高

1. `auto selfCast = ...`在已有同名变量的作用域中 — C++经典shadowing错误，开启`-Wshadow`即可自动捕获
2. CreateView后缺少空指针检查 — 同文件其他调用都有CHECK_RET，对比即可发现不一致
3. 跨三层(op_api/tiling/kernel)的变更 + tiling数据结构新增字段 — 级联影响大，是review风险信号
4. `static_cast<int32_t>`比较枚举值 — 脆弱且不直观的做法

---

### 事件3: LogSigmoid算子实现

- Revert: `4482238c056b34c004de679244ef183176e854d1` (2026-02-11 23:04, cann-robot)
- 原始: `08b547c40d976cf4b7fe17ed2478eb20fc26ed0e` (2026-02-11 09:45, huangyuxiaaaaa)
- 修复重提交: `9140e706d57e7c2c78dd41e7a77fd32724bf87e6` (2026-02-12)
- 完整cycle: 提交->revert->修复重提交，不到48小时

#### 原始提交内容

为LogSigmoid算子新增Ascend950 kernel实现。同时将算子目录从logsigmoid(无下划线)重命名为log_sigmoid(有下划线)，38个文件/+1190行。

#### Revert原因: 构建/命名/搭车问题

1. 目录重命名破坏性变更: `logsigmoid` -> `log_sigmoid`，破坏了既有的aclnn接口和UT的路径依赖
2. 目录结构不规范: op_api目录层级位置不符合仓库标准
3. CMakeLists配置问题: `ACLNNTYPE aclnn_exclude`排除了aclnn构建，暗示集成有问题
4. 搭车改动: 修改了不相关的`aclnn_binary_cross_entropy_with_logits_target_backward.cpp`中的平台判断逻辑(从架构判断改为SoC版本范围判断)
5. cann-robot自动revert，CI/门禁检查未通过

#### Code Review可发现性: 高

1. 目录重命名是破坏性变更 — 应要求提交者说明理由并确认全量CI通过
2. `ACLNNTYPE aclnn_exclude` — 新增kernel实现却排除aclnn构建，明显的配置异常
3. 不相关模块的改动(binary_cross_entropy) — 应要求拆分为独立MR
4. 修复版减少了~320行代码(871 vs 1190) — 说明原始提交包含不必要的内容

---

### 事件4: MatmulV3 small shape优化

- Revert: `e248762f7` (8.5.0分支) + `a39e18548` (master分支), 2026-02-05, llqx-1
- 原始: 从`e11fe07f3 init`(2025-09-28)导入，无独立引入提交
- MR: !1540 (8.5.0), !1553 (master)

两个revert提交是同一修复在不同分支上的同步cherry-pick，diff内容完全一致。

#### 被revert的优化内容

在`GetMixNd2nzType()`函数中(matmul_v3_base_tiling.cpp)，在V_HEAD_ND2NZ分支前插入了更高优先级的判断：当满足(仅A矩阵需要nd2nz + 非对齐处理 + BL1满载 + 基础分核 + 基础FixOpti)时，选择`V_PARALELL_ND2NZ`模板——让vector和cube管道并行执行ND到NZ格式转换，以提升小shape矩阵乘法性能。

#### Revert原因: workspace脏数据

commit message明确说明:

> 回退小shape场景下走入MatmulV3特殊优化模板，该模板可能导致出现workspace脏数据参与计算的问题。

V_PARALELL_ND2NZ模板让vector和cube并行操作workspace时，workspace buffer中可能残留上一次计算的脏数据。未被正确清零/覆盖的workspace数据被cube管道读取并参与矩阵乘法计算，导致计算结果精度错误。

UT测试中tiling_key从131074(0x20002，高位编码V_PARALELL_ND2NZ)回退到2(V_HEAD_ND2NZ)。

#### Code Review可发现性: 中

能被发现的线索:
- 引入并行执行路径时应追问workspace生命周期和初始化保证
- `unAlignProcessType > 0`(非对齐)是危险信号 — 非对齐场景buffer边界处理复杂，workspace可能有未写入的padding区域
- "非对齐 + 并行ND2NZ + workspace复用"是高风险组合

难以发现的因素:
- 问题根因在kernel侧vector/cube管道的执行时序，仅从host侧tiling代码难以看出
- 特定shape组合才触发，难以穷举边界case
- 需要对Ascend硬件pipeline交互有深入理解

---

### 跨事件模式总结

#### 模式1: 搭车提交 (事件1, 3)

两个事件都在主功能MR中混入了不相关改动:
- 事件1: unsorted_segment_sum的MR里加了scatter的文档条目
- 事件3: LogSigmoid的MR里改了binary_cross_entropy的平台判断

搭车提交的危害: Revert时不相关改动被一并回滚(附带伤害)，且增加了review难度。

审查规则: 检查MR中每个文件的改动是否与标题描述的功能直接相关。

#### 模式2: 变量遮蔽/初始化类缺陷 (事件2, 4)

- 事件2: auto重新声明导致外部同名变量被遮蔽，Cast结果丢失
- 事件4: workspace未正确初始化/清零，脏数据参与计算

两者都是"数据不是预期的值"类问题，一个在编译期可检测(-Wshadow)，一个在运行时才暴露。

#### 模式3: 大规模提交缺乏充分测试 (事件1, 2, 3)

- 事件1: 8892行，仅声称"冒烟测试"通过
- 事件2: 跨三层18文件改动，隔天revert
- 事件3: 1190行38文件，当天revert

大体量提交应拆分为可增量review的小MR，且需要比冒烟测试更充分的验证。

#### 模式4: 自动化revert机制 (事件1, 3)

cann-robot自动执行revert，分支名含`auto`后缀。说明CI/门禁流水线有自动revert能力——这是好的实践，但也意味着提交者过度依赖CI而非提交前充分自测。

#### 缺陷严重度排序

1. 事件2(变量遮蔽) — 最严重: 影响正常路径计算正确性，可导致crash
2. 事件4(workspace脏数据) — 严重: 静默精度错误，难以定位
3. 事件1(集成失败) — 中等: CI拦截有效，但搭车提交造成附带伤害
4. 事件3(构建配置) — 中等: 48小时内修复重提交，影响范围有限

---

### ops-nn-dev


### 概览

ops-nn-dev仓库共发现20个Revert提交（含3个qbmm循环revert），时间跨度2025-08 ~ 2026-02。

| 统计项 | 数值 |
|--------|------|
| Revert总数 | 20 |
| 独立事件数 | 17（qbmm 3次算1个事件） |
| 24h内回退 | 8 |
| 72h内回退 | 13 |
| 5个月+才回退 | 1 |
| 涉及matmul系列 | 9 |
| 涉及norm系列 | 4 |
| 修改公共基础设施 | 5 |

### 逐条分析

#### 9b3d2d76b - revert small case optimization in mmv3
- 日期: 2026-02-05
- 原始提交: 3695e56b1 (cv并行方案matmul性能优化, 2025-09-16)
- 涉及模块: matmul/mat_mul_v3 tiling决策逻辑
- Revert原因: 小shape场景走入V_PARALELL_ND2NZ优化模板，workspace初始化不完整导致脏数据参与计算
- 缺陷逃逸类型: 功能缺陷
- 存活时间: ~5个月
- 教训: 性能优化新增快速路径时必须确保workspace等共享资源初始化完整；小shape边界条件需覆盖内存残留场景

#### 74cdae15d - Revert NLLLossGrad performence optimize
- 日期: 2026-01-08
- 原始提交: e67aa277f (NLLLossGrad performence optimize, 2026-01-07, MR!3666)
- 涉及模块: loss/nll_loss_grad tiling+kernel大规模重构
- Revert原因: 同时修改TilingData结构体布局(字段重排+类型缩窄uint64->uint32)、kernel模板化、分核策略变更，导致host与device端数据解析不匹配
- 缺陷逃逸类型: 功能缺陷/设计缺陷
- 存活时间: 1天
- 教训: 大规模重构应拆分为多个小提交逐步验证；"性能优化"不应同时改变数据结构布局和计算逻辑

#### 8aa5be33e - revert (文档同步)
- 日期: 2026-01-08
- 原始提交: 0725e9c2f (sync nn, 2026-01-07, MR!4344)
- 涉及模块: activation/elu_grad_v2, activation/softmax_v2, quant/quantize 的aclnn文档
- Revert原因: 文档同步将产品支持表格过度精简，遗漏了Atlas推理/训练系列等仍在支持的产品信息
- 缺陷逃逸类型: 需求变更/文档准确性
- 存活时间: 1天
- 教训: 文档同步需对照完整产品支持矩阵；"sync nn"这类模糊commit message不利于审查

#### 502f8ac92 - Revert "DTS2025120868273 c04 with innerbatch"
- 日期: 2026-01-06
- 原始提交: 8bab37ba4 (DTS2025120868273 c04 with innerbatch, 2026-01-06, MR!4018)
- 涉及模块: conv/common, conv/conv2d_v2 (9文件, +381/-40行)
- Revert原因: C04模式下L1内存对齐计算顺序错误 -- `AlignB(size, C0) * innerBatch`应为`AlignB(size * innerBatch, C0)`，先对齐再乘会导致L1空间分配不足
- 缺陷逃逸类型: 功能缺陷
- 存活时间: 当天
- 教训: 特殊数据排布(C04)上叠加新特性(innerbatch)时内存对齐计算需格外小心；DTS问题单修复也需充分测试

#### 2e1ef4b41 - revert baddbmm dts
- 日期: 2025-12-22
- 原始提交: ec4484582 + c10991e38 (修复aclnnbaddbmm接口精度问题, 2025-12-19, MR!2888/!3012)
- 涉及模块: matmul/batch_mat_mul_v3, matmul/common (11文件, -415行)
- Revert原因: fp16/bf16->fp32精度修复路径实现不完整，在910_95/910b等不同平台分别出现功能错误和精度不达标
- 缺陷逃逸类型: 功能缺陷 + 性能回退
- 存活时间: 3天
- 教训: 精度修复需全平台验证；改变算子输出dtype是侵入性变更，对上下游有连锁影响

#### fe16c5570 - Revert "layernormv4功能补齐"
- 日期: 2025-12-10
- 原始提交: c3b8b716d (layernormv4功能补齐, 2025-12-09, MR!2839)
- 涉及模块: norm/layer_norm_v4 (22文件, -2917行)
- Revert原因: 3个新tiling模板+完整kernel一次性合入，同时禁用SingleReadTiling(`return false`)并收窄TransposeTiling路由条件，导致部分场景无模板可选
- 缺陷逃逸类型: 设计缺陷
- 存活时间: 1天
- 教训: 大规模功能补齐应分批合入(先加新模板不启用路由，再逐步切换)；一次性禁用现有模板+扩展平台支持是高风险操作

#### 9cc8c69c5 - revert 2691 (dlopen legacy ophost)
- 日期: 2025-12-08
- 原始提交: 23144a91d (dlopen legacy ophost, 2025-12-06, MR!2691)
- 涉及模块: matmul/fused_quant_mat_mul (仅1行)
- Revert原因: dlopen重构中为fused_quant_matmul添加的`TilingPrepareForOpCache`依赖与该算子不兼容，导致编译/链接错误
- 缺陷逃逸类型: 编译失败
- 存活时间: 2天
- 教训: 大规模基础设施重构需对每个受影响模块单独验证编译链接

#### ce7121e1f - revert addRmsNorm (发布分支同步)
- 日期: 2025-12-04
- 原始提交: 5af6b8741 (addRmsNorm Change, r1.25.0分支cherry-pick)
- 涉及模块: norm/add_rms_norm
- Revert原因: master上2b00825d3已revert的改动被cherry-pick到r1.25.0发布分支，需同步回退
- 缺陷逃逸类型: 功能缺陷(同2b00825d3)
- 存活时间: N/A(发布分支保护)
- 教训: 合入发布分支前需确认对应master提交的稳定性

#### e1fdbe6e8 - Revert "matmulv3支持选择分组累加方式计算"
- 日期: 2025-12-04
- 原始提交: b94bb78a0 (matmulv3支持选择分组累加方式计算, 2025-12-03, MR!2513)
- 涉及模块: common/inc/op_api/op_api_def.h, matmul/batch_mat_mul_v3, matmul/common
- Revert原因: 新增枚举值`FORCE_GRP_ACC_FOR_FP32=4`时将`USE_HIGH_PREC_MODE`从4改为5，破坏了已有调用方的ABI；接口参数语义从bool改为int64_t造成不兼容
- 缺陷逃逸类型: 设计缺陷
- 存活时间: 1天
- 教训: 公共API枚举值扩展应追加而非插入；接口参数语义变更需评估所有下游影响

#### 2b00825d3 - revert//addRmsNorm Change
- 日期: 2025-12-02
- 原始提交: 4efcade5a (addRmsNorm Change, 2025-12-02, MR!2592)
- 涉及模块: norm/add_rms_norm (tiling+kernel+UT)
- Revert原因: 引入全局变量`addRmsNormSocVersion`(线程不安全)；新增13个tiling字段改变数据结构布局；新增bfloat16的multi_n路径未充分测试
- 缺陷逃逸类型: 功能缺陷/设计缺陷
- 存活时间: 当天
- 教训: 避免全局变量存储运行时状态；大量tiling参数变更需确保与所有kernel路径兼容

#### 76593c3a2 - revert//ascendc_assert整改
- 日期: 2025-11-26
- 原始提交: 8c1c70474 (matmul ascendc_assert整改, 2025-11-25, MR!2159)
- 涉及模块: matmul/mat_mul_v3, quant_batch_matmul_v3/v4, weight_quant_batch_matmul_v2
- Revert原因: `ascendc_assert`(函数)统一替换为`ASCENDC_ASSERT`(宏)，但两者行为不等价(编译条件/错误处理逻辑差异)，导致多个算子用例执行失败
- 缺陷逃逸类型: 编译/功能缺陷
- 存活时间: 1天
- 教训: API名称大小写变更(函数->宏)语义可能不同，必须确认等价性；跨模块统一整改需全量测试

#### 2d6f9cb49 - Revert "[MatMul] modify range of shape to transdata"
- 日期: 2025-11-21
- 原始提交: c9dcb2f95 ([MatMul] modify range of shape to transdata, 2025-11-18, MR!2007)
- 涉及模块: matmul/common/op_host/op_api/matmul_util.cpp (1文件)
- Revert原因: IsNdToNzOnTheFly中修改transdata插入条件(内轴<128且非16对齐时跳过)，虽平均12%收益但10%shape劣化，生产环境命中了劣化case
- 缺陷逃逸类型: 性能回退
- 存活时间: 3天
- 教训: 性能优化存在明确劣化比例时不应直接合入；"平均有收益"不等于"所有场景都有收益"

#### 0799defc7 - Revert "change gather_v2 l0"
- 日期: 2025-11-19
- 原始提交: a7055b2e5 (change gather_v2 l0, 2025-10-30, MR!1691)
- 涉及模块: index/gather_v2 (op_api + UT)
- Revert原因: 新增`isPreprocessed` OP_ATTR改变了算子属性签名，AICPU侧属性名列表变更是breaking change，与已有tiling/kernel不兼容。且代码风格重构与功能变更混在同一提交
- 缺陷逃逸类型: 功能缺陷(接口不兼容)
- 存活时间: 20天
- 教训: OP_ATTR变更是接口变更，需tiling/kernel同步修改；格式重构与功能变更不应混在同一提交

#### 6f830ee8c - Revert "modify name of addRmsNormDynamicQuant and the binary patch"
- 日期: 2025-11-13
- 原始提交: e8491f1a2 (modify name of addRmsNormDynamicQuant, 2025-11-12, MR!1739)
- 涉及模块: norm/add_rms_norm_quant binary.json (1文件, +1736/-560行)
- Revert原因: binary配置JSON中的bin_filename或simplified_key与实际编译产出的二进制文件不匹配，导致算子无法正确加载预编译kernel
- 缺陷逃逸类型: 功能缺陷
- 存活时间: 1天
- 教训: binary配置修改必须与实际编译产出严格对应；算子名称变更需端到端验证加载链路

#### ac976d4b5 - mmv3, bmmv3, gemmv2 tiling key revert
- 日期: 2025-09-30
- 原始提交: 729189a56 + 5400ae82f (mat_mul_v3/gemm_v2/batch_mat_mul_v3 tilingkey重构, 2025-09-25, MR!547/!737)
- 涉及模块: matmul/mat_mul_v3, batch_mat_mul_v3, gemm_v2 (11文件)
- Revert原因: tiling key从magic number重构为枚举常量时，kernel侧符号名变化(extern "C"->模板函数)与binary_config注册不兼容；tiling侧和kernel侧对tilingkey理解未同步
- 缺陷逃逸类型: 设计缺陷
- 存活时间: 5天
- 教训: 跨模块重构(tiling+kernel)需双方完全对齐后一起合入；后续10-14通过0bad81147成功重做

#### 61a1f1583 - revert identity/identity_n/rank/shape/shape_n
- 日期: 2025-09-27
- 原始提交: c5752277c (move ge_local, 2025-09-26, MR!558)
- 涉及模块: control/identity, identity_n, rank, shape, shape_n + cmake/symbol.cmake (53文件, -2690行)
- Revert原因: 从ge_local迁移5个control算子到nn仓时修改了公共cmake/symbol.cmake和UT基础设施，影响其他算子编译/测试。后续10-10重新添加时不再改symbol.cmake
- 缺陷逃逸类型: 功能缺陷(迁移不完整)
- 存活时间: 1天
- 教训: 算子迁移需评估对公共基础设施的影响；公共cmake修改需充分回归

#### qbmm三次Revert循环

时间线:
```
09-12  8c2157971  qbmm原始提交 (MR!113) - 新增quant_batch_matmul_v3/v4, +65519行
09-13  f7188ff70  test delete - 提交后立即发现测试问题
09-15  79623db1a  第一次Revert "qbmm" (MR!410) - 功能问题，从master删除
09-15  7b4afa892  mmv3 wqbmmv2 code fix - revert后修复关联代码
09-21  4f9d6758c  Revert "Revert "qbmm"" (MR!532) - bug修复后恢复到master
09-23  03fe5cfbe  Revert "qbmm" (MR!837) - 从r1.23.0发布分支回退("蓝区回退")
```

##### 79623db1a - Revert "qbmm" (第一次)
- 日期: 2025-09-15
- 原始提交: 8c2157971 (qbmm, MR!113, 2025-09-12)
- 涉及模块: matmul/quant_batch_matmul_v3/v4 + 公共模块(op_util.h, infershape, cmake) - 161文件
- Revert原因: 6.5万行超大提交，修改了公共op_util.h和matmul_common_infershape.cpp，测试不通过
- 缺陷逃逸类型: 功能缺陷
- 存活时间: 3天

##### 4f9d6758c - Revert "Revert "qbmm"" (恢复)
- 日期: 2025-09-21
- 恢复操作，bug修复后在master重新启用，附带约220行修复代码
- 缺陷逃逸类型: N/A

##### 03fe5cfbe - Revert "qbmm" (蓝区回退)
- 日期: 2025-09-23
- 从r1.23.0发布分支删除qbmm，master保留。功能未达发布标准不应包含在release中
- 缺陷逃逸类型: 流程问题

qbmm事件根因:
1. 6.5万行一次性合入无法有效review
2. 公共代码(op_util.h, infershape)与新算子耦合在同一PR
3. master到release分支缺乏feature gate，未成熟功能流入发布分支

#### 9de027499 - revert pr 32.
- 日期: 2025-08-26
- 原始提交: 602c72811 (add weight_quant_batch_matmul, 2025-08-25, MR!32)
- 涉及模块: matmul/weight_quant_batch_matmul_v2 + common/include/error_util.h (136文件, -57059行)
- Revert原因: 5.7万行新算子合入时修改了公共error_util.h(改变CUBE_INNER_ERR_REPORT宏格式)，导致其他算子编译失败
- 缺陷逃逸类型: 编译失败
- 存活时间: <16小时
- 教训: 新算子不应修改公共头文件已有接口格式；公共基础设施变更应独立PR先行验证

---

### 模式归纳

#### 按缺陷逃逸类型分布

| 类型 | 次数 | 占比 | 典型案例 |
|------|------|------|----------|
| 功能缺陷 | 9 | 45% | workspace脏数据、L1对齐错误、接口不兼容 |
| 设计缺陷 | 4 | 20% | 大规模合入、枚举值插入、全局变量 |
| 编译失败 | 3 | 15% | 公共头文件修改、符号解析、assert等价性 |
| 性能回退 | 2 | 10% | transdata条件、精度修复连锁 |
| 需求/流程 | 2 | 10% | 文档精简、蓝区回退 |

#### 五大系统性问题

1. 公共基础设施修改耦合在算子PR中 (5次: qbmm, weight_quant, identity迁移, dlopen, assert整改)
   - 公共头文件(op_util.h, error_util.h, op_api_def.h)和cmake修改影响面远超算子本身
   - 应独立PR先行验证

2. 提交粒度过大 (4次: qbmm 6.5万行, weight_quant 5.7万行, layernormv4 3000行, identity 2690行)
   - 万行级提交无法有效review
   - 应分批合入，先基础设施后算子逻辑

3. 性能优化验证不足 (4次: mmv3 small case, NLLLossGrad, transdata, baddbmm)
   - 性能优化新增快速路径时遗漏边界条件
   - workspace/内存初始化不完整
   - 存在劣化case时仍强行合入

4. 接口变更兼容性评估缺失 (4次: 枚举值插入, OP_ATTR变更, TilingData布局, bool->int64_t)
   - 公共API枚举值插入而非追加
   - 算子属性签名变更未同步tiling/kernel
   - 数据结构字段重排导致ABI不兼容

5. Release分支准入门控缺失 (2次: qbmm蓝区回退, addRmsNorm发布分支同步)
   - master上未充分验证的功能被cherry-pick到release分支
   - 需要独立的feature readiness评审

#### 高危模块

| 模块 | Revert次数 | 涉及提交 |
|------|-----------|----------|
| matmul系列 | 9 | mmv3小case, NLLLoss, baddbmm, matmulv3累加, transdata, tilingkey, qbmm×3, fused_quant |
| addRmsNorm | 3 | 全局变量(2b008), 发布分支(ce712), binary配置(6f830) |
| conv2d | 1 | C04 innerbatch |
| gather_v2 | 1 | OP_ATTR变更 |
| layernormv4 | 1 | 3000行功能补齐 |

matmul系列占Revert总量的45%，是最高危模块。

#### 回退速度分布

| 存活时间 | 次数 | 说明 |
|----------|------|------|
| 当天 | 4 | 合入后立即发现问题 |
| 1天 | 6 | 次日冒烟测试发现 |
| 2-5天 | 5 | 集成测试/下游发现 |
| 20天 | 1 | gather_v2(延迟发现) |
| 5个月 | 1 | mmv3小case(长尾隐蔽缺陷) |

70%的revert在3天内发生，说明合入前的review和测试门禁形同虚设 -- 问题在合入后而非合入前被发现。

---

## Part 3: 跨类别系统性风险

### 附录A: [主仓] 文档与示例代码 (~36条)

主仓独有类别, dev分支无对应。非纯文档修复(纯文档已在阶段1过滤), 指代码仓中的示例代码bug、文档与代码不一致、op_list条目遗漏等。

#### 典型案例

- `501d38ab` A16W4示例代码重复函数定义(AclnnWeightQuantBatchMatmulV2Test出现两次), 用户拷贝编译直接报重定义错误
- `4063f8f9` aclnnTransposeBatchMatMul示例输出全0, FP16数据初始化错误, 示例从未被实际运行验证
- `964be4dd` deep_norm_grad示例vector<int32_t>但传ACL_FLOAT创建tensor; deep_norm示例缺aclrtFree导致内存泄漏

#### 审查检查点
- [ ] 示例代码应在CI中编译验证(至少编译通过)
- [ ] 示例中vector<T>的T必须与aclDataType匹配
- [ ] aclrtMalloc/aclrtFree必须配对出现
- [ ] 新增算子时op_list.md条目名称必须与算子名一致

---

### 跨类别系统性风险

#### 风险1: matmul是绝对缺陷震中

两仓合计: matmul/bmm贡献最多缺陷。主仓67条(17.6%), dev分支热点Top 30文件中matmul占20席。
- Revert事件中matmul系列占45%(dev: 9/20, 主仓: 2/4)
- 量化矩阵乘(quant_batch_matmul_v3/v4, weight_quant_batch_matmul_v2)尤为集中
- 缺陷跨越host/kernel/op_api三层, 涵盖空指针(P0)、CHECK_RET检查错变量(P0)、copy-paste公式(P0)、除零(P1)、溢出截断(P2)
- 建议: matmul相关PR需要额外一轮专项审查

#### 风险2: 整数溢出是最普遍残留风险

- 当前代码中仍存在12处int32溢出风险(dev分支统计)
- 横跨matmul/conv/quant/norm四个模块
- 根因: int32类型的tiling参数相乘、batch维度累乘、buffer偏移计算均缺少溢出保护
- 主仓的GM偏移在小类型域溢出(foreach系列10个文件)是另一个系统性模式
- 建议: 编写lint规则, 强制shape维度算术运算使用int64_t

#### 风险3: 提交粒度过大导致质量失控

- dev: qbmm事件6.5万行一次性合入导致3次循环revert; weight_quant事件5.7万行合入修改了公共error_util.h
- 主仓: unsorted_segment_sum 38文件/8892行revert; batchmatmul 18文件跨三层revert
- 两仓合计70%的revert在3天内发生, 说明合入前review不充分
- 建议: 单次合入超过3000行的PR必须拆分

#### 风险4: 公共基础设施耦合在算子PR中

- dev: 5次revert源于公共头文件(op_util.h, error_util.h, op_api_def.h)和cmake修改耦合在算子PR中
- 主仓: build.sh被10条缺陷提交触及(热点第1), 当前代码仍存在多个P0/P1 bug
- 公共修改影响面远超单算子, 但review时容易被算子逻辑掩盖
- 建议: 公共文件修改必须独立PR先行验证

#### 风险5: build.sh系统性脆弱(主仓)

[主仓] build.sh被10条缺陷提交触及, 当前代码仍存在多个P0/P1 bug:
- `set +u`从行35开始从未恢复, 整个脚本未定义变量静默为空
- `]`前缺空格(P0)、数组备份只取首元素(P0)
- ASCEND_HOME_PATH未设置时路径变成`/include`等根目录(P1)
- CMAKE_ARGS非数组, 参数含空格时错误拆分
- main通过管道传给gawk, exit在subshell中无法正确传播

#### 风险6: 编译warning隐藏真bug

两仓合计55条编译告警修复中至少6条包含真正的逻辑bug:
- 主仓: 无符号无限循环、链式比较、运算符优先级(3条)
- dev: 变量遮蔽掩盖copy-paste、链式比较语义错误、Cast方向反转(3条)
- 开启`-Wall -Werror`可系统性拦截此类缺陷

#### 风险7: ascendc_config.json配置混乱(主仓)

[主仓] 多个算子存在重复条目(AdaptiveAvgPool3d行2/5, LayerNormQuant行571/600/601三份)和同名冲突配置(compute_units含/不含特定平台), 构建行为不确定。

#### 风险8: foreach模块高触及但低风险(dev)

[dev] 聚合触及次数最高(1452次), 但本质是批量模板代码同步修改。单个算子结构性风险低, 真正需重点review的是matmul和quant的kernel/tiling层。不应被热点文件列表误导分配不当的审查资源。

#### 风险9: 先用后查反模式在op_api层流行(主仓)

[主仓] 空指针解引用后再判空、OP_CHECK后先LOG再CHECK、工厂函数返回值直接使用, 这些"先用后查"模式在op_api层多处存在, 是crash的高发源。

---

### 审查规则优先级矩阵

#### P0 -- 必须检查(高频高危, 可直接拦截严重缺陷)

| 规则ID | 规则                                                         | 覆盖类别                  | 典型案例hash                         |
|--------|-------------------------------------------------------------|--------------------------|-------------------------------------|
| R01    | shape维度相乘/buffer偏移计算强制使用int64_t                    | INT_OVERFLOW             | fa1305c56e, e1a2f11f, 8f6ccaea      |
| R02    | tiling侧buffer计算必须与kernel侧InitBuffer做交叉比对          | TILING_BUFFER            | e25e6e60c7, b99dcb02c5              |
| R03    | 空tensor输入时是否正确early return并设置空输出shape             | BOUNDARY                 | 95c2fbb6bf, e32ce0cbbb              |
| R04    | SetFlag/WaitFlag必须成对出现且无条件执行                       | PIPELINE_SYNC            | a79b527fe8, 51f8247aee, ed0bd5a1    |
| R05    | 设置特殊状态(UnknownRank)后必须包含return                     | CONDITION_LOGIC          | 2d424df8e6                          |
| R06    | CeilDiv/CeilAlign结果是否需要与实际shape做min(上界clamp)       | CALC_LOGIC               | a70d8b16d5                          |
| R07    | 新增平台时checklist检查所有相关算子的case/分支和compute_units   | PLATFORM                 | 15138bbd86, 4f87bea9ab              |
| R08    | workspace计算公式中逐项确认是否需要双buffer                    | TILING_BUFFER            | e32a5eaae3                          |
| R09    | 可选输出tensor(outputMask控制)的访问前必须检查null              | NULLPTR                  | 59a561e5, e49d5d27c0, 7b4a1b53      |
| R10    | 多核场景全局内存初始化应由指定核完成并加同步屏障                  | CALC_LOGIC+SYNC          | fa03888b0d                          |
| R11    | [主仓] !=A\|\|!=B恒真逻辑错误(De Morgan)                     | CONDITION_LOGIC          | 67c665fd, 692f43a9                  |
| R12    | [主仓] GM偏移量在小类型域计算溢出, 用1ULL*强制64位              | INT_OVERFLOW             | 50df91e7                            |
| R13    | [主仓] OP_CHECK宏第三参数必须包含return语句                    | NULLPTR                  | 7b4a1b53                            |
| R14    | [主仓] 先检查后使用, 禁止先解引用后判空                        | NULLPTR                  | aclnn_addmm.cpp, aclnn_quant_*.cpp  |

#### P1 -- 重点检查(中频高危或高频中危)

| 规则ID | 规则                                                         | 覆盖类别                  | 典型案例hash                         |
|--------|-------------------------------------------------------------|--------------------------|-------------------------------------|
| R15    | 修改Init签名/模板参数后全局搜索所有调用点同步更新                | INTERFACE_API            | b14f5a03ca                          |
| R16    | 新增算子checklist: ascendc_config.json + binary.json + CMake  | BUILD_CMAKE+CONFIG       | c00e940bfa                          |
| R17    | 连续相似CHECK/LOG调用中检查对象与赋值对象是否匹配               | COPY_PASTE               | 5972749646, d1832c87c9              |
| R18    | `__CCE_AICORE__ == xxx`精确匹配是否应改为范围比较              | PLATFORM                 | 4f87bea9ab                          |
| R19    | switch/case的default分支不能静默通过, 必须返回错误             | CONDITION_LOGIC          | 多处                                |
| R20    | TilingData结构体字段布局和类型在host和kernel端完全一致          | TILING_BUFFER+DATA_TYPE  | 0d32207040                          |
| R21    | 新增数据类型支持时同步tiling层/kernel层/binary配置三处          | DATA_TYPE                | 9150a7cca8                          |
| R22    | 禁止在op_host/*.cpp中使用非const全局容器                      | RESOURCE                 | 2a15cd4d                            |
| R23    | DMA搬运参数(stride/nBurst)是否可能超过硬件限制(uint16_t上限)   | PIPELINE_SYNC            | 58f275c6dc, 34217db7                |
| R24    | 除法运算的除数必须有非零保护                                   | CONDITION_LOGIC          | 22f84c0e                            |
| R25    | [主仓] 接口调用默认参数与实际数据类型是否一致                    | INTERFACE_API            | c8ca6bec                            |
| R26    | [主仓] 属性索引硬编码顺序依赖, 优先用属性名                     | INTERFACE_API            | 07e77ddd                            |
| R27    | [主仓] PipeBarrier必须在FreeTensor之前                        | RESOURCE                 | cf84e222                            |
| R28    | [主仓] 结构体指针成员必须在声明处初始化为nullptr                 | NULLPTR                  | 15e40a48                            |
| R29    | [主仓] SetScheduleMode必须显式设置                            | PIPELINE_SYNC            | afb09c78                            |

#### P2 -- 常规检查(中危或可部分自动化)

| 规则ID | 规则                                                         | 覆盖类别                  | 典型案例hash                         |
|--------|-------------------------------------------------------------|--------------------------|-------------------------------------|
| R30    | 公共文件(op_util.h, error_util.h)修改必须独立PR               | BUILD_CMAKE              | revert分析                          |
| R31    | 单次合入超过3000行的PR必须拆分                                 | REVERT                   | qbmm事件                            |
| R32    | 删除"未使用变量"前检查是否原本应该被使用(可能掩盖bug)           | COMPILE_WARN             | d818b4d4                            |
| R33    | 同族算子(XXX与XXXGrad)配置一致性对比                           | CONFIG_MISSING           | 793745bf14                          |
| R34    | 错误码区分PARAM_*(用户输入)和INNER_*(内部状态)                 | LOG_ERROR                | 797589989a                          |
| R35    | 禁止直接注释EXPECT_*/ASSERT_*断言                             | UT_TEST                  | 91c29129                            |
| R36    | 示例代码必须通过实际编译运行验证                                | [主仓]文档               | 501d38ab, 4063f8f9                  |
| R37    | matmul相关PR需要额外一轮专项审查                               | 跨类别                   | revert分析                          |

#### P3 -- 建议自动化(可通过工具/CI拦截)

| 规则ID | 规则                                                         | 实现方式     |
|--------|-------------------------------------------------------------|-------------|
| R38    | 启用编译器flags: -Werror -Wshadow -Wconversion -Wsign-compare -Wparentheses | CMake配置   |
| R39    | shape维度算术运算lint规则(检测int32乘法)                       | 自定义lint   |
| R40    | ascendc_config.json与算子目录交叉校验                          | CI脚本      |
| R41    | SetFlag/WaitFlag配对检测                                      | 静态分析    |
| R42    | "// EXPECT_"注释断言检测                                      | grep规则    |
| R43    | compile_options平台差异检测                                    | CI脚本      |

---

### 模块分布与缺陷密度

两仓高缺陷模块(按绝对频次):
1. matmul/bmm: 两仓合计最高, 主仓67条(17.6%), dev热点Top 30中占20席
2. quant: dev分支第二热点模块
3. conv: 主仓22条(5.8%)
4. norm: 主仓24条(6.3%)
5. scatter/gather: 主仓19条(5.0%)
6. build/compile: 主仓18条(4.7%)
7. pooling: 主仓16条(4.2%)
8. foreach: dev分支高触及(1452次)但低风险(批量模板同步)

matmul是当之无愧的缺陷热点, 缺陷类型覆盖了18个类别中的绝大多数。

---

### 两仓差异分析

ops-nn-dev (612缺陷/2571提交=23.8%) vs ops-nn (380缺陷/1474提交=25.8%)

相同点:
- 整数溢出、tiling/buffer计算、边界条件缺失、平台适配是两者共有的高频高危类别
- matmul模块在两个仓库都是最高危模块
- 可审查性均在99%以上

差异点:
- ops-nn-dev的BUILD_CMAKE占比(14.3%)远高于ops-nn(~10.9%), dev分支构建基础设施更不稳定
- ops-nn-dev的INTERFACE_API占比(9.6%)高于ops-nn(~5.6%), dev分支接口变更更频繁
- ops-nn-dev有更多UT_TEST类缺陷, dev分支测试基础设施迁移频繁
- ops-nn-dev的Revert事件(20条/17个)显著多于主仓(5条/4个), 70%在3天内回退, 说明dev分支CI门禁更宽松
- 主仓的编译告警/代码规范类缺陷在分析时已由CI拦截更多, dev分支CI门禁更宽松
