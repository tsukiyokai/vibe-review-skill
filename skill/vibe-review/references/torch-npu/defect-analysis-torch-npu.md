# torch_npu Defect Diff Analysis

Generated: 2026-04-12
Repository: torch_npu (Ascend PyTorch Adapter)
Source: git-history.md BUGFIX+REVERT index (3,504 entries, ~1,131 unique after dedup)
Method: full-coverage git show + manual analysis, categories emergent from data
Entries: D-1 ~ D-1317 (1316 defects; D-145 skipped in numbering)
Reviewability: high 756 / medium 451 / low 106 (3 entries without: cross-references)

## Entries

### D-1: batch_isend_irecv只对tensors[0]做recordStream，遗漏[1..n]

- Hashes: c86a6dda3
- Root cause: 批量操作遗漏元素
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `collective()`的pre callback只为`tensors[0]`调用`recordStream()`，`tensors[1..n]`
  同样参与HCCL DMA但未被记录。多流并行场景下这些tensor的内存可能被计算流复用覆盖，导致NaN。
- Fix: pre callback中循环遍历`tensors[1..n]`，根据内存复用模式（CLOSE/AVOID_RECORD_STREAM/
  ERASE_RECORD_STREAM/ERASE_RECORD_STREAM_WITH_OPTIMIZE）分别调用recordStream。+20行。
- Reviewability: high -- 循环只处理第一个元素是典型遗漏，对照batch语义即可发现
- Review rule: 批量操作中检查容器内所有元素是否都得到了相同的资源管理处理

### D-2: Revert inductor Contiguous Reduction判断修复（修复引入回归）

- Hashes: 1ddb00b4b
- Root cause: 修复引入回归
- Files: test/_inductor/test_sum_dual_reduction_contiguous.py,
  torch_npu/_inductor/codegen/kernel_analysis.py,
  torch_npu/_inductor/codegen/split_tiling.py,
  torch_npu/_inductor/codegen/triton.py
- Defect: 此前的fix改动了`is_contiguous_reduction()`等方法的判断逻辑（引入
  `stride_sorted_var_list`替代`golden_var_list`），导致其他场景回归。具体回归症状未在
  revert message中说明。
- Fix: 完整revert原fix，移除`get_reduction_layout_var_list()`方法，恢复原逻辑。
- Reviewability: medium -- 原fix本身逻辑正确但测试覆盖不足，需要更多reduction场景测试
- Review rule: codegen路径的fix需要覆盖contiguous/non-contiguous/dual-reduction等全场景回归测试

### D-3: revert removing the libhccl dependency（间接链接依赖断裂）

- Hashes: 7deab3a5e [+9 cherry-picks: fe710e921, d62c20b4e, 1a5666577, 82b8aff3a,
  18b748325, 7a1a37da5, 3eb2cfdc0, 971d7e602, ec0bf4010... (部分)]
- Root cause: 依赖链断裂
- Files: CMakeLists.txt, setup.py
- Defect: 删除`libtorch_npu.so`对`libhccl.so`的PUBLIC链接后，上层应用通过间接依赖路径
  `liba.so -> libtorch_npu.so -> libhccl.so`无法解析hccl符号。
- Fix: 恢复CMakeLists.txt中`target_link_libraries`的PUBLIC链接，移除setup.py中
  patchelf删除libhccl的逻辑。
- Reviewability: high -- 删除共享库PUBLIC依赖时应检查所有下游消费者的链接需求
- Review rule: 变更shared library链接属性(PUBLIC->PRIVATE或删除)前，`ldd`检查下游.so的符号依赖

### D-4: A3平台gcc13不识别inductor编译flag

- Hashes: ec0bf4010 [+3 cherry-picks]
- Root cause: 平台/编译器兼容性
- Files: torch_npu/_inductor/__init__.py, torch_npu/_inductor/cpp_builder.py
- Defect: inductor cpp codegen使用的编译优化flag（如`-march=native`）在A3机器的gcc13上
  不被识别，导致编译失败。
- Fix: 实现`_get_optimization_cflags()`替换函数，按平台(darwin/ppc64le/x86)和编译器
  过滤不兼容flag，通过monkey-patch注入。
- Reviewability: medium -- 需要多平台CI才能捕获，纯代码检视难以发现
- Review rule: 编译flag硬编码时需要标注目标平台范围，或用compiler feature detection替代

### D-5: dispatch log解析遗漏EXECUTE_OPAPI_V2类型

- Hashes: b70ee2d8f [+4 cherry-picks: e7c19a6a0, 8cb23d098, d99df75fb, e68812a07]
- Root cause: 枚举分支遗漏
- Files: torch_npu/csrc/core/npu/NPUQueue.cpp
- Defect: 新增`ExecuteParasOpApiV2`算子类型后，`get_func_error_msg()`和`Enqueue()`中的
  类型分派未增加对应分支，导致V2类型的算子名称被错误解析为event名称。
- Fix: 在两个函数中增加`EXECUTE_OPAPI_V2`类型的处理分支。+8行。
- Reviewability: high -- 新增枚举值后未更新所有switch/if分支，编译器-Wswitch可捕获
- Review rule: 新增枚举值时grep所有消费该枚举的switch/if，确保全覆盖；启用-Wswitch-enum

### D-6: ACLGraph capture在多流wait_stream交互时replay结果错误

- Hashes: 916426933 [+8 cherry-picks: 8147d1150, 9eb868378, 7d86011b9, 30cfde000,
  2b31be7f8, f83b2829d, c86f48f9d, 680435630]
- Root cause: graph capture未处理多流状态同步
- Files: test/npu/test_aclgraph_multi_stream.py (new),
  torch_npu/csrc/core/npu/NPUCachingAllocator.cpp,
  torch_npu/csrc/core/npu/NPUCachingAllocator.h,
  torch_npu/csrc/core/npu/NPUEvent.cpp
- Defect: 主流和旁路流通过`wait_stream`交互时，`NPUEvent::block()`在`LaunchWaitEventTask`
  后未检查是否处于graph capture状态。capture期间task queue中的任务不会被立即执行，
  导致replay时数值错误。
- Fix: `DeviceCachingAllocator`新增`hasCapturesUnderway()`接口（加锁查询）；
  `NPUEvent::block()`在capture状态时调用`emptyAllNPUStream()`刷新task queue。+157行含测试。
- Reviewability: low -- 需要多流+graph capture运行时场景才能触发，纯静态分析极难发现
- Review rule: graph capture路径的任何同步操作(event/stream wait)都需要考虑task queue刷新时机

### D-7: inductor accuracy compare对ConcatKernel的图追踪处理不正确

- Hashes: 8f6a67abf (v2.7.1-26.0.0), 33c4ffdcc (v2.7.1) [同fix不同release分支]
- Root cause: IR节点类型分支遗漏
- Files: torch_npu/_inductor/lowering_fx.py
- Defect: `fetch_graphs()`中对`traced_graph`不为None的输入直接追加到`input_graphs`，
  但`ConcatKernel`节点的traced_graph追踪结果不正确，导致accuracy compare失败。
- Fix: 添加3层isinstance检查排除`ConcatKernel`（直接、`.data`、`.data.data`），
  对ConcatKernel走新建`TracedGraph()`路径。
- Reviewability: medium -- 需要运行accuracy compare且遇到cat算子才能触发
- Review rule: 图追踪/lowering中对IR节点的通用处理需要排查特殊节点类型（如ConcatKernel）

### D-8: aclrtGetDeviceInfo在A5(Ascend950)上不可用

- Hashes: 048efeeee (v2.10.0-26.0.0), 2e5b8909c (v2.9.0-26.0.0),
  876e3f11f (v2.8.0-26.0.0), 826d9dc6a (v2.7.1-26.0.0),
  18b902d9f (v2.11.0), 437c909ea (v2.10.0),
  0833cf4ae (v2.8.0), f03c512ed (v2.7.1) [同fix 8个release分支]
- Root cause: 硬件API跨SoC行为不一致
- Files: torch_npu/csrc/npu/Module.cpp
- Defect: `initDeviceProperty()`调用`aclrtGetDeviceInfo()`获取设备内存信息，
  但driver在A5(Ascend950)上未实现此接口，返回错误码507899(internal error)。
  根因是drv在A2和A5上行为不一致，需rts后续修复。
- Fix: 在调用前增加`c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950`
  前置检查，A5上回退到旧的`GetMemInfo`接口。改动仅1行条件表达式。
- Reviewability: low -- 需要A5硬件环境运行才能发现，属于driver层面的兼容性问题
- Review rule: 新SoC适配时对所有runtime/driver API做兼容性矩阵验证

### D-9: tensor.clone对非连续tensor强制输出连续tensor（语义偏差）

- Hashes: 5a798156e [+8 cherry-picks]
- Root cause: NPU算子语义偏离PyTorch原生行为
- Files: torch_npu/csrc/aten/ops/op_api/CloneKernelOpApi.cpp
- Defect: `clone()`实现中无条件调用`apply_tensor_without_format(src)`生成连续tensor，
  忽略了`MemoryFormat::Preserve`语义。PyTorch原生行为是在Preserve模式下保持原tensor的
  stride layout，NPU实现总是返回连续tensor，导致`test_contiguous`等下游测试失败。
- Fix: 当`memory_format == Preserve`且`src.is_non_overlapping_and_dense()`时，
  用`at::empty_strided_symint(src.sym_sizes(), src.sym_strides(), ...)`保持原stride。
- Reviewability: high -- 对照PyTorch op spec或原生实现即可发现语义差异
- Review rule: NPU自定义算子实现须与PyTorch原生行为对齐，尤其是MemoryFormat相关语义

### D-10: inductor pow对fp64类型缺少fallback

- Hashes: a4bbd71ad
- Root cause: dtype覆盖不全
- Files: test/_inductor/test_pow.py (new),
  torch_npu/_inductor/lowering.py,
  torch_npu/_inductor/lowering_override_list.py
- Defect: NPU Triton不支持fp64的pow运算，但inductor lowering中未注册pow的NPU覆盖，
  导致fp64 pow直接走Triton kernel编译失败。
- Fix: 注册完整的`aten.pow` lowering逻辑：小整数指数内联展开，fp64和整数类型走
  `fallback_handler`回退到aten实现，其余走`ops.pow` pointwise。新增54行测试。
- Reviewability: medium -- 需要fp64测试用例触发，但lowering注册表的completeness可以系统检查
- Review rule: 新算子lowering注册需要覆盖全dtype矩阵，不支持的dtype必须显式fallback

### D-11: tiling切分逻辑中blocks修改和programs判断顺序颠倒

- Hashes: 193938062
- Root cause: 循环体内语句顺序错误
- Files: torch_npu/_inductor/codegen/tile_generator.py
- Defect: `TileGenerator`的切分循环中，`blocks[axis]`的切半/缓降逻辑原本在
  `total_programs > num_vector_core`判断之后执行，导致先用旧blocks值做了program count
  判断和config记录，再修改blocks，下一轮循环才生效。特定shape下产生非最优或错误的tiling。
- Fix: 将blocks切分代码块（含`slow_decend_split`分支）从`total_programs`判断之后
  移到之前，确保每轮循环先切分再判断。
- Reviewability: medium -- 需要理解循环的状态变更顺序，特定shape才触发错误路径
- Review rule: 循环体内涉及"修改状态→条件判断→记录结果"的模式，检查三者的执行顺序是否正确

### D-12: _rebuild_npu_tensor在fake tensor模式下无法反序列化

- Hashes: dd34f22e8 (master), 32bef956f (v2.7.1)
- Root cause: fake tensor模式未处理
- Files: torch_npu/utils/storage.py, test/_inductor/test_rng_prims.py
- Defect: `_rebuild_npu_tensor()`在输入为fake tensor时，尝试在NPU上创建真实数据（调用
  `tensor.npu()`），但fake mode下无法执行真实设备操作，导致反序列化失败。缺少对
  `torch._guards.detect_fake_mode()`的检测，未设置`fake_device`属性。
- Fix: 添加`detect_fake_mode()`检测，fake mode下设置`tensor.fake_device = target_device`
  而非实际搬移数据；同时将`torch.tensor([])`替换为`torch.empty((0,))`避免deprecation。
- Reviewability: medium -- 需要了解fake tensor模式的存在，但isinstance检查是标准防御模式
- Review rule: 涉及tensor创建/重建的代码路径需考虑fake tensor模式的兼容性

### D-13: benchmark aclgraph使能逻辑反转 + 变量初始化顺序错误

- Hashes: 5cc3d16e5
- Root cause: 初始化顺序错误 + 条件取反逻辑错误
- Files: benchmarks/torchbench/common.py, benchmarks/torchbench/npu_support.py
- Defect: (1) `experiment`和`optimize_ctx`在DDP分支之后才初始化，但DDP分支中已使用了
  这些变量，导致aclgraph模式使能失效。(2) aclgraph黑名单判断为`not in NO_ACLGRAPH`
  时才设mode，逻辑导致黑名单中的模型反而启用了aclgraph。
- Fix: 将`experiment`/`optimize_ctx`初始化前移到DDP分支之前；反转黑名单判断为
  `in NO_ACLGRAPH`时禁用mode；新增`--disable-aclgraph`命令行参数。
- Reviewability: high -- 变量使用在初始化之前是IDE可检查的问题
- Review rule: 变量初始化必须在其所有消费路径的common dominator之前

### D-14: Contiguous Reduction判断数据源不一致（后被D-2 revert）

- Hashes: 6446a3adb (v2.7.1)
- Root cause: 同一语义的两个独立实现使用不同数据源
- Files: torch_npu/_inductor/codegen/triton.py, split_tiling.py, kernel_analysis.py,
  test/_inductor/test_sum_dual_reduction_contiguous.py (new)
- Defect: `TritonKernel.is_contiguous_reduction()`使用`self.golden_var_list`（按indexing
  longest选择，顺序为`[r4,x1,r3,y0]`），而`SplitTiling.is_contiguous_reduction()`使用
  `parse_golden_from_load_store_index()`（按stride排序，顺序为`[r3,r4,y0,x1]`）。两者对
  同一kernel返回不同的var顺序，导致contiguous判断不一致（一个True一个False），
  `dense_size_str()`走错分支，生成错误的kernel代码。
- Fix: 引入`get_reduction_layout_var_list()`统一数据源，multi-reduction场景优先使用
  stride排序结果。但此fix后来被D-2 revert，说明在其他场景引起了回归。
- Reviewability: low -- 需要理解codegen内部多处golden_var_list的语义差异，且需特定shape触发
- Review rule: 同一概念在多处独立实现时，应提取为单一truth source；或加assertion确保一致

### D-15: empty_like在h2d/d2h拷贝中输出连续性丢失

- Hashes: c33d1b59d, b0d7dcb30 [cherry-pick]
- Root cause: API语义变更后调用方未适配
- Files: torch_npu/csrc/aten/common/CopyKernel.cpp, test/npu/test_copy.py (new)
- Defect: `copy_h2d_baseformat()`和`copy_d2h_baseformat()`对非连续dst调用
  `at::empty_like(dst)`创建临时buffer。`empty_like`被修改为支持非连续输出后，默认保留
  src的stride layout（非连续），导致后续`aclrtMemcpy`期望连续内存但得到非连续tensor，
  拷贝结果错误。
- Fix: 在`empty_like`调用中显式传入`LEGACY_CONTIGUOUS_MEMORY_FORMAT`，确保临时buffer
  始终连续。h2d和d2h两处均修改。+138行测试覆盖。
- Reviewability: high -- 对`empty_like`语义变更后应检查所有调用点的memory_format假设
- Review rule: `empty_like`/`zeros_like`等_like API的调用方如依赖输出连续性，须显式传memory_format

### D-16: patch_expr_fits_within_32bit强制64bit索引致性能下降 + cat constant崩溃

- Hashes: db1de538e
- Root cause: 过度全局monkey-patch + 分支遗漏
- Files: torch_npu/_inductor/__init__.py, utils.py, codegen/triton.py, lowering.py,
  test/_inductor/test_cat.py
- Defect: (1) `patch_expr_fits_within_32bit()`将`expr_fits_within_32bit`全局替换为
  `return False`，强制所有索引使用64bit，导致onerec模型性能下降。(2) `cat_store()`中
  未处理`self.golden_var_list`为None的情况（cat constant场景），访问None索引崩溃。
- Fix: (1) 完全删除`patch_expr_fits_within_32bit`及其调用。(2) 在`cat_store`开头增加
  `golden_var_list is None`的early return，改用block_ptr索引存储。(3) cat lowering增加
  `inputs[0].get_device().type == "npu"`检查，防止CPU tensor走NPU ConcatKernel路径。
- Reviewability: medium -- 性能问题需benchmark发现；None guard是标准防御模式
- Review rule: 全局monkey-patch的side effect范围应在PR中显式枚举；cat lowering需device guard

### D-17: npugraphify_impl中OrderedSet类型注解作用域错误 + 缺少函数参数

- Hashes: 7d08fde3d (v2.8.0-26.0.0), 00da0b258 (v2.9.0), 747d94038 (v2.10.0),
  f12196ba3 (v2.7.1) [同fix 4个release分支]
- Root cause: 类型注解作用域误用 + 函数签名不同步
- Files: torch_npu/utils/_graph_tree.py
- Defect: (1) `static_input_idxs: OrderedSet[int] = OrderedSet(...)`使用了Python类型
  注解语法重新绑定变量，但`OrderedSet`在该作用域下的泛型参数可能不兼容，运行时报错。
  (2) `align_inputs_from_check_idxs(run, check_input_idxs)`调用缺少上游新增的
  `mutated_input_idxs`关键字参数，导致TypeError。
- Fix: (1) 移除类型注解保留赋值。(2) 补充`mutated_input_idxs=OrderedSet()`参数。
- Reviewability: high -- 缺参数是基础API调用检查，类型注解副作用也是已知Python陷阱
- Review rule: 与上游PyTorch函数签名同步时，diff对比所有调用点的参数列表

### D-18: index/embedding/gather lowering缺少fx graph追踪信息

- Hashes: f0f390e0f
- Root cause: 图追踪元信息传递链断裂
- Files: torch_npu/_inductor/lowering.py, lowering_fx.py
- Defect: `aten.index`的lowering走`lowering_index_select`时未传递`traced_graph`和
  `node_name`参数，且`lowering_index_select`的签名无默认值，导致A5 ETA/HLLM场景
  accuracy compare因缺少graph信息失败。同样，`aten.embedding`和`aten.gather`未在
  `DUMP_FX_GRAPH_LOWERING_OPS`中注册且缺少fx graph追踪的lowering实现。
- Fix: (1) `lowering_index_select`参数增加默认值`traced_graph=None, node_name=None`。
  (2) `index`的lowering中创建并传递traced graph。(3) 新增`inductor_gather`和
  `inductor_embedding`的fx lowering实现（~70行）。
- Reviewability: medium -- 需要了解fx graph dump的数据流才能发现缺少传递
- Review rule: 新注册的lowering函数需确认是否需要参与fx graph追踪链

### D-19: NPU空tensor的UntypedStorage分配跳过导致设备信息丢失

- Hashes: 5034e00f5
- Root cause: 边界条件遗漏(size=0)
- Files: torch_npu/csrc/core/NPUStorageImpl.cpp
- Defect: `make_npu_storage_impl()`中`data_ptr == nullptr && size > 0`的条件导致size=0
  时不调用`allocator->allocate(0)`，storage未正确关联到NPU设备。
  `torch.UntypedStorage(0, device="npu")`返回的storage被标记为CPU设备，
  后续save/load空tensor时设备信息丢失。
- Fix: 将条件改为`data_ptr == nullptr`（无论size），始终调用`allocate`；`data_ptr`的
  非空校验移到`size > 0`条件内，允许size=0时data_ptr为空。
- Reviewability: high -- 零size边界条件是经典缺陷模式，对照CUDA实现即可发现差异
- Review rule: storage/allocator代码的边界用例清单：size=0, size=1, size=MAX

### D-20: static mode下list comprehension变量名拼写错误 + cat with reindex未fallback

- Hashes: 0f59193ef
- Root cause: 变量名拼写错误 + 路径遗漏
- Files: torch_npu/_inductor/codegen/cpp_wrapper.py, lowering.py
- Defect: (1) `CppWrapperNpu`中list comprehension使用`call_args`（外部列表）而非
  循环变量`call_arg`（当前元素），`isinstance(call_args, sympy.Integer)`对列表永远为
  False，static mode下sympy.Integer参数未被过滤，编译错误。
  (2) cat lowering中含`ReindexView`的输入走`ConcatKernel`路径，但ConcatKernel不支持
  reindex，产生错误codegen代码。
- Fix: (1) 将`call_args`改为`call_arg`。(2) 检测到ReindexView输入时直接
  `fallback_handler(aten.cat.default)`。
- Reviewability: high -- 变量名拼写错误是code review最基础的检查项
- Review rule: list comprehension中循环变量与外部变量同名时需lint规则禁止（pylint W0640）

### D-21: MLIR autotuning仅按性能选配置，未校验计算精度

- Hashes: 541e8c6e7
- Root cause: autotuning策略缺少精度维度
- Files: torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/config.py,
  torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/npu/mlir_compiler.py
- Defect: `benchmark_all_configs()`在autotuning时仅根据性能(timing)选择最优kernel配置，
  未校验各配置的计算精度。已知不同MLIR配置存在精度问题，可能选出精度不合格的配置。
  大模型场景下运行时精度校验有OOM风险。另外动态shape场景下精度校验本身也存在bug。
- Fix: 增加`ANIR_ACC_CHECK_DURING_TUNE`环境变量控制；启用时对每个候选配置调用
  `accuracy_pass()`进行`torch.allclose`精度校验，仅合格配置参与timing排序；
  全部不合格时fallback到fx实现。修复动态shape场景的精度校验。
- Reviewability: medium -- 需要了解autotuning的质量维度应包含精度，非显而易见
- Review rule: autotuning/autoselect机制应考虑正确性约束，不能仅以性能为唯一指标

### D-22: shmem v1.0.1初始化flag不匹配

- Hashes: c067d12d4 (v2.7.1-26.0.0), 354be76a7 (v2.8.0), 7d894245f (v2.9.0),
  aba73bd6b (v2.10.0) [同fix 4个release分支]
- Root cause: 第三方库API版本不兼容
- Files: torch_npu/csrc/distributed/symm_mem/NPUSHMEMExtension.cpp
- Defect: shmem库升级到v1.0.1后，使用UniqueID流程初始化时要求传入
  `ACLSHMEMX_INIT_WITH_UNIQUEID`标志，但代码仍使用`ACLSHMEMX_INIT_WITH_DEFAULT`，
  导致初始化失败报错。
- Fix: 将`Aclshmemx_init_attr`的flag参数从`ACLSHMEMX_INIT_WITH_DEFAULT`改为
  `ACLSHMEMX_INIT_WITH_UNIQUEID`。改动1行。
- Reviewability: low -- 需要知道shmem v1.0.1的API变更，纯代码审查无法发现
- Review rule: 第三方库版本升级时对照changelog验证所有API调用点的参数兼容性

### D-23: inductor动态shape can't-split越界崩溃

- Hashes: 653bc3e57
- Root cause: 符号表达式静态求值路径遗漏 + 越界未防御
- Files: torch_npu/_inductor/codegen/triton.py
- Defect: `NPUIndexTritonKernel`的tile splitting逻辑使用`sv.statically_known_equals(remaining[i], 1)`
  跳过大小为1的分组，但对动态SymInt该函数始终返回False。循环遍历完所有分组后
  `current_group`越界，直接crash而非回退到unsplit路径。
- Fix: (1) 在`statically_known_equals`检查旁增加`size_hint`运行时估算作为fallback。
  (2) 增加`current_group >= len(remaining)`的越界检查，触发`CantSplit`异常
  让上层回退到非切分codegen路径。
- Reviewability: medium -- 需要理解SymInt的静态/动态语义区别
- Review rule: inductor中所有`statically_known_*`判断旁必须有dynamic shape fallback

### D-24: aclrtGetDeviceInfo在dlopen场景下undefined symbol

- Hashes: d78d50b30 [+11 cherry-picks: 4c87b9f9e, f282447c9, 7ab606761,
  b6e893ee1, 1fe4a6f3f, 4b6da615b, f7c6f023e, 49b320e68, ce56e9db2 等]
- Root cause: 编译期静态符号引用 vs 运行时dlopen加载冲突
- Files: torch_npu/csrc/core/npu/interface/AclInterface.cpp/.h,
  torch_npu/csrc/npu/Module.cpp
- Defect: D-8中为A2/A5兼容性增加了`aclrtGetDeviceInfo`调用，使用直接链接方式。
  但在某些部署环境中CANN库通过dlopen动态加载而非编译期链接，该符号产生
  `ImportError: undefined symbol: aclrtGetDeviceInfo`，`import torch_npu`直接失败。
- Fix: 将`aclrtGetDeviceInfo`加入`LOAD_FUNCTION`宏管理的动态加载函数表，
  新增`IsExistAclrtGetDeviceInfo()`运行时探测，调用点改为先探测再调用的wrapper方式。
- Reviewability: medium -- 需了解dlopen部署模式的存在
- Review rule: CANN API新增调用必须走动态加载包装层(AclInterface)，禁止直接静态链接

### D-25: inductor static kernel codegen缺少do_indent导致生成代码缩进错误

- Hashes: 9577d9be8
- Root cause: 代码生成器缩进状态管理遗漏
- Files: torch_npu/_inductor/codegen/wrapper.py
- Defect: `NPUWrapperCodeGen.write_prefix()`中，进入`with self.prefix.indent()`
  上下文前未调用`self.prefix.do_indent()`，导致`global has_initialized`和
  `if not has_initialized:`生成在错误缩进层级，产生SyntaxError。
  仅在`_use_static_aclnn_kernel`启用时触发。
- Fix: 在`with self.prefix.indent():`前插入`self.prefix.do_indent()`。改动1行。
- Reviewability: high -- 生成的Python代码运行时即报SyntaxError
- Review rule: codegen改动需增加实际执行生成代码的端到端测试

### D-26: fusion_attention_grad sharding策略对pse=None产生错误placement

- Hashes: e95fa0ad1 [+8 cherry-picks: 8a34c6089, a9d6ee4b3, 20beb0fec,
  d3ebae158, f924f9aeb, 67723ed3e, 01978bf4b 等]
- Root cause: 可选参数的分片策略缺少None守卫
- Files: torch_npu/distributed/tensor/_attention.py
- Defect: `npu_fusion_attention_grad_strategy`中，`pse`参数为None时反向不产生`grad_pse`，
  但输出schema中`grad_pse`位硬编码为`Replicate()`。DTensor框架对None值执行redistribute
  时触发AttributeError。三处策略声明(default/batch_dim/head_dim)均存在此问题。
- Fix: `grad_pse`的placement改为`Replicate() if pse is not None else None`。
- Reviewability: high -- None guard是DTensor策略的标准模式
- Review rule: sharding strategy中每个可选输入/输出的placement必须有None分支

### D-27: npu_format_cast的keyword-only参数声明与调用方式不一致

- Hashes: 5294f5ffe
- Root cause: Python函数签名中`*`分隔符缺失导致参数传递方式不一致
- Files: test/torch_npu_schema.json, torch_npu/onnx/wrapper_onnx_ops.py
- Defect: `npu_format_cast`的schema声明`customize_dtype`为keyword-only参数(`*`后)，
  但ONNX wrapper层的`_wrapper_npu_format_cast`和`_NPUFormatCastOP.forward`将其
  作为positional参数接收。`forward`使用`*args, **kwargs`签名时参数转发丢失keyword语义。
- Fix: (1) wrapper签名统一为`(self, acl_format, *, customize_dtype=None)`
  (2) `_NPUFormatCastOP.forward`改为显式参数，keyword方式调用底层op。
- Reviewability: high -- schema与实现签名的一致性可通过自动化校验
- Review rule: 自定义算子wrapper的参数传递方式必须与schema声明严格一致

### D-28: rng_state dispatch在FakeTensor模式下未注册

- Hashes: 2768a7ff9
- Root cause: custom op dispatch表缺少FakeTensorMode注册
- Files: torch_npu/utils/_inductor.py,
  test/_inductor/test_dropout_with_checkpoint_recompute.py (new),
  test/_inductor/test_rng_prims.py (new)
- Defect: `patch_register_run_and_save_rng_state_op()`注册了PrivateUse1的dispatch实现
  但未处理FakeTensorMode。inductor编译trace时使用FakeTensor，
  `run_and_save_rng_state`对NPU设备的RNG操作在fake mode下无法执行。
  此外缺少重入保护，多次import会重复注册导致warning。
- Fix: 引入`FakeTensorMode`，在dispatch注册中增加fake实现；
  添加`_npu_patched`哨兵变量防止重复注册。新增2个测试文件。
- Reviewability: medium -- 需理解inductor的FakeTensor trace流程
- Review rule: NPU custom op注册必须同时提供real和fake两套dispatch实现

### D-29: eagerConnectSingleDevice全捕获掩盖不可恢复错误

- Hashes: d6fdcbf9a
- Root cause: 异常处理粒度过粗，catch-all吞掉了所有异常
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `eagerConnectSingleDevice`的单个`catch(std::exception&)`将网络超时（可恢复）
  和HCCL库错误（不可恢复）统一处理为"warn and fallback to lazy init"。
  真实硬件/配置错误被静默降级，直到首次集合通信时才崩溃，丢失原始诊断信息。
- Fix: 三层分级catch: (1) `c10::DistNetworkError` -> warn+fallback
  (2) `c10::Error` -> error+throw (3) `std::exception` -> 检查message含"network"
  关键词判断恢复策略，兜底处理`broadcastMasterID`的类型擦除。
- Reviewability: medium -- 需理解各异常类型的语义
- Review rule: 分布式初始化的异常处理必须区分可恢复(网络)和不可恢复(设备/配置)

### D-30: codegen tiling中sympy.Expr作为dict key查找失败

- Hashes: 7fde7932b
- Root cause: dict key类型假设过窄(只考虑Symbol不考虑Expr)
- Files: torch_npu/_inductor/codegen/triton.py,
  torch_npu/_inductor/fx_passes/ascend_custom_passes/__init__.py,
  torch_npu/_inductor/lowering_fx.py
- Defect: `get_axis_dtype()`用`indexing_map`的key直接查`range_tree_nodes`字典。
  当key为复合`sympy.Expr`(如`x2 + 512*y1 + 2048*z0`)时dict查找KeyError。
  同时fx passes未做topological sort，动态shape下节点处理顺序不确定。
- Fix: (1) 新增`_iter_candidate_syms`：对Expr类型提取`free_symbols`逐个查找。
  (2) fx pass入口添加`stable_topological_sort(gm)`。
  (3) lowering_fx中注册`_inductor_test.realize`防止test op缺失。
- Reviewability: medium -- compound bug，需特定shape组合触发
- Review rule: sympy对象作为dict key时必须处理Symbol和Expr两种类型

### D-31: scope_begin/end从custom_op装饰器改为显式Library注册

- Hashes: f9487d048
- Root cause: `@torch.library.custom_op`装饰器在特定PyTorch版本下schema推导有隐性问题
- Files: torch_npu/npu/graphs.py, torch_npu/npu/__init__.py
- Defect: `super_kernel_scope_begin/end`使用`@torch.library.custom_op`装饰器注册。
  装饰器自动推导的schema在某些dispatch路径下（如BackendSelect）未正确注册impl，
  导致aclgraph的scope标记在特定调用链路下失效或报错。
- Fix: 改用`torch.library.Library("npu", "FRAGMENT").define()`显式注册schema，
  手动绑定PrivateUse1、BackendSelect、CPU三个dispatch key的实现函数，
  并用`@register_fake`注册meta实现。同步导出公共API wrapper函数。
- Reviewability: medium -- 需理解PyTorch dispatch机制
- Review rule: 需要跨版本兼容的custom op优先使用Library API而非decorator API

### D-32: ProcessGroupHCCL析构函数吞掉shutdown异常

- Hashes: 24f51db28
- Root cause: C++析构函数中异常未传播，失去诊断信息
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `~ProcessGroupHCCL()`在用户未显式`destroy_process_group()`时触发`shutdown()`。
  如果shutdown失败，异常被析构函数的隐式noexcept吞掉，进程静默退出无任何错误信息。
  分布式训练中导致节点静默挂起难以定位原因。
- Fix: shutdown调用外包裹try-catch，catch后LOG(ERROR)输出完整错误信息并rethrow
  触发coredump，确保有可分析的现场。有意crash: shutdown失败表明通信状态已损坏。
- Reviewability: high -- 析构函数中异常处理是C++标准问题
- Review rule: 析构函数中关键cleanup操作失败必须记录日志并产生诊断信息

### D-33: MLIR动态shape launch时torch.Size与tuple拼接类型错误

- Hashes: 0b2319f09
- Root cause: torch.Size与tuple隐式类型不兼容
- Files: torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/npu/mlir_compiler.py
- Defect: `NpuMlirCompiler`在构造动态shape kernel的launch参数时，直接用`+`拼接
  `arg.size()`(torch.Size)和普通tuple。某些PyTorch版本中torch.Size的`__add__`返回
  torch.Size而非tuple，与后续tuple拼接时产生类型不匹配错误。
- Fix: 将`arg.size()`和`arg.stride()`显式转为`tuple()`: 改为`tuple(arg.size())` +
  `tuple(arg.stride())`。改动2处。
- Reviewability: high -- torch.Size不是普通tuple是常见Python陷阱
- Review rule: 与torch.Size/torch.Stride做拼接时必须显式转换为tuple

### D-34: alltoall集合通信在compatible模式下不必要地flatten产生OOM

- Hashes: a6f79de99
- Root cause: compatible模式判断逻辑缺少SoC能力检查 + 冗余flatten
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: alltoall的默认路径对所有输入做flatten+cat再传给HCCL，产生等size的临时内存。
  `compatible_mode`判断条件中`is_compatible_soc`检查缺失，导致在支持native alltoall_v
  的SoC上也走flatten路径。scatter/gather同理。
- Fix: 重构alltoall路径：compatible mode仅在`use_compatible_impl && is_compatible_soc`
  同时为真时启用，直接传原始tensor无需flatten。默认路径的flatten保持不变。
  scatter/gather同步修正compatible判断条件。
- Reviewability: medium -- 需要大规模分布式场景触发OOM
- Review rule: 集合通信的compatible/default路径选择必须检查SoC能力

### D-35: MLIR accuracy_pass传入transformed_args导致动态shape精度校验全失败

- Hashes: e6445bec9
- Root cause: 精度校验输入参数来源错误
- Files: torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/npu/mlir_compiler.py
- Defect: `benchmark_all_configs`调用`accuracy_pass(fx_outputs, *transformed_args)`，
  `transformed_args`经过padding/reshape处理，shape与`fx_outputs`(用原始args计算)不一致。
  动态shape下`torch.allclose`永远失败，所有候选配置被判为精度不合格，
  触发全量fallback到eager模式，性能严重劣化。
- Fix: 将accuracy_pass的参数从`*transformed_args`改为`*args`(原始参数)，
  确保与fx reference输出的shape一致。增加fallback时的warning日志。
- Reviewability: medium -- 需理解accuracy_pass的args transform流程
- Review rule: 精度对比的输入和参考输出必须来自同一数据源

### D-36: CANN版本低于8.5时aclrtSetStreamAttr调用参数非法

- Hashes: 2c2e89982
- Root cause: 新版API无版本守卫
- Files: torch_npu/csrc/core/npu/NPUGraph.cpp
- Defect: `apply_cache_op_info()`调用`aclrtSetStreamAttribute()`设置
  `ACL_STREAM_ATTR_CACHE_OP_INFO`属性，该API和枚举值仅在CANN 8.5+存在。
  低版本CANN上调用返回"invalid params"错误，NPUGraph功能不可用。
- Fix: 添加`IsGteCANNVersion("8.5.0", "CANN")`前置检查，低版本直接return跳过。
- Reviewability: high -- CANN版本守卫是标准防御模式
- Review rule: 调用CANN新版特有API前必须添加IsGteCANNVersion版本守卫

### D-37: is_core_control_enabled遗漏函数调用括号导致条件恒真

- Hashes: 08d567865
- Root cause: 函数指针被当作bool值（缺少调用括号）
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: 多处HCCL通信调用前的guard `if (c10_npu::is_core_control_enabled)` 遗漏了`()`，
  变成了函数指针的真值判断（非null即true）。导致`UseStreamResInCurrentThread()`被
  无条件调用，在不支持该API的旧CANN版本上产生segfault。影响allreduce、
  batch_isend_irecv、broadcast、reduce等10+处集合通信。
- Fix: 所有`is_core_control_enabled`改为`is_core_control_enabled()`，补全调用括号。
- Reviewability: high -- 编译器`-Waddress`可检测函数指针用于bool上下文
- Review rule: 启用`-Waddress`警告；条件判断中的函数名后必须有调用括号

### D-38: Revert clone MemoryFormat一致性修复（D-9引发回归，fix-revert cycle #2）

- Hashes: cebf05bcc [+1: ccbbde6d8 另一变体]
- Root cause: 修复引入回归 -- D-9的stride保持语义破坏了下游内存布局假设
- Files: torch_npu/csrc/aten/ops/op_api/CloneKernelOpApi.cpp
- Defect: D-9 (5a798156e)修复clone对非连续tensor不保持stride的问题，引入`empty_strided`
  路径。但某些下游路径(matmul等)隐式依赖clone返回连续tensor的行为，
  修改后非连续stride被保留，matmul冒烟精度测试失败。
- Fix: 完整revert D-9修改，clone恢复为无条件`apply_tensor_without_format`返回连续tensor。
  这是第二个fix-revert cycle（第一个是D-2/D-14），暴露了NPU内存布局语义的深层矛盾。
- Reviewability: low -- 精度回归需要端到端matmul冒烟测试才能发现
- Review rule: 涉及tensor memory layout变更的fix必须运行全量精度回归测试

### D-39: JIT测试import失败(upstream API重命名additional_module_tests)

- Hashes: df136434f
- Root cause: upstream PyTorch API重命名未同步
- Files: test/test_jit.py
- Defect: PyTorch upstream将`additional_module_tests`重命名为`get_all_nn_module_tests()`，
  删除`module_tests`/`new_module_tests`的直接导出。torch_npu的JIT测试使用旧名导致import失败。
- Fix: 更新import和调用处至新API名`get_all_nn_module_tests`。
- Reviewability: high -- 上游API重命名在升级时应有checklist覆盖
- Review rule: torch版本升级时检查所有从torch.testing._internal导入的API是否仍存在

### D-40: inductor测试skip残留+路径断言未更新+缺失import

- Hashes: 57c6f0b17
- Root cause: 测试与代码版本不匹配(多点)
- Files: test/_inductor/test_npu_device.py, test/_inductor/test_run_with_rng_state.py
- Defect: (1) test_abi_compatible_header被`@unittest.skip`跳过但实际已可运行，header路径从
  `experiment/runtime/runtime/rt.h`变为`runtime/runtime/rt.h`但断言未更新；
  (2) test_run_with_rng_state缺少`import torch_npu._inductor`导致inductor未初始化。
- Fix: 删除skip decorator，更新路径断言，添加缺失import。
- Reviewability: high -- skip decorator应关联issue并定期清理
- Review rule: 暂时skip的测试需有issue跟踪；路径变更后grep全仓库中的旧路径字符串

### D-41: batch_isend_irecv无条件register_work导致OOM

- Hashes: 5d3160fe2
- Root cause: 无条件注册work引用导致内存泄漏
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `batch_isend_irecv`/`_reduce_scatter_base_uneven`/`_allgather_base_uneven`中无条件调用
  `c10d::register_work`，即使`allow_inflight_collective_as_graph_input()`为false。注册的
  work引用阻止tensor被释放。`synchronize()`/`wait()`中缺少对应`unregister_work`调用。
  多次调用后OOM。
- Fix: 用`allow_inflight_collective_as_graph_input()`守卫`register_work`；在`synchronize()`
  和`wait()`中添加`unregister_work`。
- Reviewability: high -- register/unregister应成对出现是典型模式
- Review rule: register_work/unregister_work必须成对；条件性注册资源时检查feature flag守卫

### D-42: UT profiler断言未包含新kernel名aclnnAdds

- Hashes: 67eee1b3c
- Root cause: kernel实现变更后profiler断言覆盖不全
- Files: test/trans_contiguous/test_single_reshape_copy_to_contiguous.py
- Defect: contiguous copy操作底层kernel从`contiguous_h_memRepoint`/`aclnnInplaceCopy`变为
  有时使用`aclnnAdds`，但UT的profiler断言未包含该算子名，UT失败。
- Fix: 在profiler断言中添加`aclnnAdds`为可接受算子。
- Reviewability: medium -- 需要了解kernel替换才能预判断言覆盖性
- Review rule: kernel实现替换后搜索全仓库中对旧kernel名的profiler断言并扩展

### D-43: empty_like未保留源tensor的strides(preserve_format语义偏离)

- Hashes: 32033f92d
- Root cause: NPU tensor factory丢失stride信息
- Files: torch_npu/csrc/aten/common/TensorFactories.cpp, test/npu/test_npu.py
- Defect: `empty_like_npu()`始终使用`ApplyTensorWithFormat`创建tensor，忽略源tensor的
  strides。当source是non-contiguous(如.T后)时，默认`preserve_format`语义应保留原strides，
  但实际总产出contiguous layout。与PyTorch语义不一致。
- Fix: 当format是base format且memory_format为Preserve时，用`at::infer_dense_strides`
  计算并保留strides，调用`at::empty_strided`创建tensor。
- Reviewability: medium -- 需理解preserve_format语义和NPU format系统的交互
- Review rule: Tensor factory函数验证memory_format全部枚举值的行为是否与PyTorch一致

### D-44: scatter/gather compatible mode冗余flatten内存分配

- Hashes: 81ee865af
- Root cause: compatible mode分支仍执行默认mode的内存分配路径
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `gather()`和`scatter()`在compatible mode下仍调用`flatten_for_scatter_gather`
  分配world_size大小的扁平化buffer，而compatible mode通过send/recv逐pair通信根本不需要
  这些buffer。non-root rank还额外分配`empty_like`空tensor。大规模场景无效内存占用显著。
- Fix: gather中移除flatten分配，改传input tensor作为placeholder。scatter中按模式分支，
  compatible mode只传最小placeholder，default mode保留原flatten逻辑。
- Reviewability: medium -- 需理解compatible mode的通信模式(point-to-point非collective)
- Review rule: 新增实现模式时检查是否仍走了旧模式的初始化/分配路径

### D-45: npugraphs后端monkey-patch未传播到_graph_tree模块的独立引用

- Hashes: 8343dcb87
- Root cause: Python from-import创建独立引用，monkey-patch不传播
- Files: torch_npu/_inductor/utils.py, torch_npu/npu/_graph_tree.py
- Defect: `_graph_tree.py`通过`from torch._inductor.utils import
  get_first_incompatible_cudagraph_node`获得独立函数引用。
  `patch_get_first_incompatible_cudagraph_node()`修改了原模块和compile_fx中的引用，
  但`_graph_tree`中的引用已是独立绑定不受影响。导致npugraphs对bool index_put等
  cudagraph-unsafe算子的检测失效。另外hasattr条件守卫在版本绑定场景下可能掩盖错误。
- Fix: 显式patch `torch_npu.utils._graph_tree`模块引用；移除hasattr条件判断。
- Reviewability: medium -- Python from-import语义陷阱需经验识别
- Review rule: monkey-patch函数时搜索全仓库中所有from-import该函数的位置并逐一覆盖

### D-46: apply_tensor_without_format缺少TORCH_NPU_API导出标记

- Hashes: bc651e172
- Root cause: 符号可见性标记遗漏
- Files: torch_npu/csrc/framework/utils/OpPreparation.h
- Defect: `OpPreparation::apply_tensor_without_format`三个重载无`TORCH_NPU_API`标记。
  外部.so(如plugin)调用时，动态链接报undefined symbol。
- Fix: 给三个声明添加`TORCH_NPU_API`宏。
- Reviewability: high -- 公共API头文件方法应统一检查导出标记
- Review rule: 公共类新增/审查方法时确认需跨.so调用的方法有TORCH_NPU_API

### D-47: NN测试import失败(upstream API重命名skipIfMps/tf32_is_not_fp32)

- Hashes: 01fa00ece
- Root cause: upstream PyTorch API重命名未同步(与D-39同类)
- Files: test/test_nn.py, test/unsupported_test_cases/.pytorch-disabled-tests.json
- Defect: PyTorch upstream多处重命名: `skipIfMps`→`skipIfMPS`,
  `tf32_is_not_fp32()`→`torch.cuda.is_tf32_supported()`,
  `new_module_tests`→`get_new_module_tests()`。旧名导致import/调用失败。
- Fix: 更新所有引用至新API名。
- Reviewability: high -- 版本升级checklist应包含API重命名检查
- Review rule: 同D-39

### D-48: D2H序列化pinned memory路径遗漏(tensor已预处理为CPU)

- Hashes: 6f42bdec7
- Root cause: 序列化路径分支遗漏
- Files: torch_npu/utils/serialization.py, test/npu/test_serialization.py
- Defect: `_npu_save()`的`use_pinned_memory_for_d2h`逻辑只处理storage仍在NPU设备的分支。
  某些路径(tensor reduce path)会先将NPU storage材料化为CPU再进入`_npu_save`，此时不走
  NPU→CPU pinned memory拷贝分支，启用配置后实际仍走非pinned路径。
- Fix: 添加else分支处理已在CPU的storage，检查config和accelerator类型后创建pinned
  memory storage并拷贝。
- Reviewability: medium -- 需理解序列化的多条路径(direct save vs tensor reduce path)
- Review rule: 处理device-specific逻辑时考虑tensor可能已被其他路径预处理到不同设备上

### D-49: Python closure不可pickle导致torch.compile序列化失败

- Hashes: 00e474c7f
- Root cause: 闭包函数(closure)无法被pickle序列化
- Files: torch_npu/dynamo/__init__.py, torch_npu/dynamo/trace_rule.py
- Defect: `_get_default_backend`内的`_lazy_exec`是闭包(捕获`name`)，
  `_get_npugraph_ex_backend`内的`_exec`也是闭包。torch.compile在分布式或cache场景
  需pickle这些backend函数时抛出`PicklingError: Can't pickle local object`。
- Fix: `_lazy_exec`移到module级，用全局变量`_global_backend_name`替代closure捕获；
  `_exec`移到module级(无需捕获外部变量)。
- Reviewability: high -- torch.compile backend必须可pickle是已知约束
- Review rule: 注册给torch.compile的callback不能是closure或lambda，必须是module-level函数

### D-50: IsGteCANNVersion版本比较条件逻辑错误(9.0版本不识别)

- Hashes: 2b1cc5ea2
- Root cause: 版本号比较条件逻辑错误
- Files: torch_npu/csrc/core/npu/GetCANNInfo.cpp
- Defect: `IsGteCANNVersion`的V2格式分支条件为`major1 >= 8 && minor1 >= 5`。
  CANN 9.0(major=9, minor=0)时`minor1 >= 5`为false，不进入V2分支。条件将
  "版本>=8.5"误写为两个独立AND条件，漏掉了major>8且minor<5的情况(如9.0)。
- Fix: 改为`(major1==8 && minor1==5 && major2==8 && minor2==5) or (major1>8 && major2>8)`。
- Reviewability: high -- 版本比较是高频出错区域
- Review rule: 版本比较函数必须有覆盖边界值(8.4, 8.5, 9.0, 10.0)的UT

### D-51: inductor dynamic shape codegen mask生成遗漏

- Hashes: 1dc2cc2f2
- Root cause: 静态shape假设导致dynamic shape下codegen缺少mask
- Files: torch_npu/_inductor/codegen/triton.py
- Defect: `_codegen_mask()`在`is_no_loop_axis`为true时跳过mask生成。dynamic shape场景下
  runtime size可能超过tile大小，仍需mask保护。`find_axis_in_load_store`只搜索
  `tl.load`/`tl.store`行，遗漏compute行中的符号引用。
- Fix: 添加`get_allow_dynamic()`检查shape_env是否有dynamic range，dynamic模式下强制
  生成mask。扩展`find_axis_in_load_store`搜索范围至所有行。
- Reviewability: low -- 需理解inductor codegen的tiling/mask和dynamic shape语义
- Review rule: codegen路径修改需同时覆盖static和dynamic shape两种模式测试
- Note: 此fix被D-53 revert，构成fix-revert cycle #3

### D-52: inductor codegen注释缩进在return后(unreachable位置)

- Hashes: b4b4bbe0b
- Root cause: codegen缩进错误导致注释位于死代码位置
- Files: torch_npu/_inductor/codegen/kernel_analysis.py, torch_npu/_inductor/codegen/triton.py
- Defect: `kernel_analysis.py`中`# 2 analyze permute shape`注释位于return语句之后的
  缩进层内，实际是dead code位置。不影响执行但误导读者。
- Fix: 修正注释缩进至return之前的正确位置。
- Reviewability: high -- return后的代码/注释在review时应被标记
- Review rule: 检查return语句后是否有意外的代码或注释

### D-53: revert D-51的dynamic shape fix(inductor回归)

- Hashes: 3c4ee22b2
- Root cause: 修复引入回归 (fix-revert cycle #3，与D-51对应)
- Files: torch_npu/_inductor/codegen/triton.py, torch_npu/_inductor/codegen/kernel_analysis.py
- Defect: D-51(1dc2cc2f2)引入的`get_allow_dynamic()`和相关变更(如`_deduplicate_vars`、
  `_filter_and_append_missing`)导致其他inductor场景回归。
- Fix: 完整revert D-51，删除`get_allow_dynamic()`等辅助函数，恢复原codegen逻辑。
- Reviewability: medium -- 原fix测试覆盖不足
- Review rule: codegen路径fix需全场景回归测试

### D-54: inductor dynamic shape正确修复(var_to_val→backed_var_to_val)

- Hashes: c2f432df8
- Root cause: sympy变量替换使用了错误的映射表(var_to_val缺少backed变量)
- Files: torch_npu/_inductor/codegen/kernel_analysis.py,
  torch_npu/_inductor/codegen/split_tiling.py,
  torch_npu/_inductor/codegen/triton.py,
  torch_npu/_inductor/codegen/cpp_wrapper.py
- Defect: 多处用`V.graph.sizevars.var_to_val`做sympy替换。dynamic shape下`var_to_val`
  不含backed symbolic变量的值，应用`backed_var_to_val`。导致IndexAnalysis、SplitTiling
  numel计算、initialize_range_tree等都无法正确处理dynamic shape。
  这是D-51/D-53 fix-revert后的正确方案：问题根源是映射表选错，而非mask生成策略。
- Fix: 全部`var_to_val`改为`backed_var_to_val`；添加`inductor_meta`参数。
- Reviewability: low -- 需深入理解sizevars系统(var_to_val vs backed_var_to_val区别)
- Review rule: sizevars做sympy替换时，dynamic shape场景必须用backed_var_to_val

### D-55: MLIR dynamic shape中torch.Size与tuple拼接类型不兼容

- Hashes: 38b6728af
- Root cause: torch.Size隐式类型与tuple拼接不一致
- Files: torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/npu/mlir_compiler.py
- Defect: `NpuMlirCompiler`构建`args_new`时直接拼接`arg.size()`(返回torch.Size)和
  `arg.stride()`(返回tuple)。dynamic shape下torch.Size元素为SymInt，拼接行为不一致。
- Fix: 显式`tuple(arg.size())`和`tuple(arg.stride())`确保类型统一。
- Reviewability: high -- torch.Size不是标准tuple，拼接操作应显式转换
- Review rule: torch.Size和tuple拼接时必须显式tuple()转换

### D-56: gather/scatter/alltoall compatible mode缺少SoC兼容性限制

- Hashes: d05465aad
- Root cause: compatible mode未检查底层API的SoC支持范围
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: compatible mode基于send/recv模拟集合操作，但HCCL send/recv仅在A2(910B)和
  A3(910_93)上支持。其他SoC(310B、910_95)启用compatible mode会调用不支持的API。
- Fix: 添加`IsCompatibleSoc()`检查SoC版本范围(910B~310B前 和 910_93~910_95前)，
  在collective lambda中将SoC检查作为额外条件。
- Reviewability: medium -- 需了解各SoC的HCCL API支持矩阵
- Review rule: compatible mode的每个操作实现需检查底层API的SoC支持范围

### D-57: ProcessGroupHCCL shutdown同步粒度过粗(device级→stream级)

- Hashes: 0b9d27722
- Root cause: device级同步阻塞所有stream
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `shutdown()`调用`npuSynchronizeDevice()`同步整个设备，阻塞该设备上所有stream
  (包括其他PG和计算stream)。多PG共享设备时造成全局停顿，严重时可能deadlock。
- Fix: 收集当前PG的HCCL stream(`hcclStreams_`)，只同步这些stream。无stream时跳过。
- Reviewability: high -- 全局device同步是已知的性能/死锁风险模式
- Review rule: 避免device级同步，优先stream/event级同步

### D-58: custom_fwd/custom_bwd签名与upstream PyTorch不一致(**kwargs vs keyword-only)

- Hashes: 65fd39a25
- Root cause: 自定义AMP装饰器未跟进upstream API演进
- Files: torch_npu/npu/amp/autocast_mode.py, test/torch_npu_schema.json
- Defect: torch_npu的`custom_fwd`用`**kwargs`接收参数，与upstream标准化的
  `custom_fwd(fwd=None, *, cast_inputs=None)`签名不匹配。upstream已提供device-generic
  `torch.amp.custom_fwd(device_type='npu')`，torch_npu不应维护独立实现。
- Fix: 改为thin wrapper + `@deprecated`装饰器，delegate到
  `torch.amp.custom_fwd(device_type='npu')`。更新schema签名。
- Reviewability: high -- API签名变更应在版本对齐时检查
- Review rule: upstream已提供device-generic API时及时migrate并deprecate自有实现

### D-59: dump_fx_graph lowering ops注册不完整导致NotImplementedError

- Hashes: 37b4199cf
- Root cause: FX graph dump路径的lowering op注册表不完整
- Files: torch_npu/_inductor/lowering.py, torch_npu/_inductor/lowering_fx.py,
  test/_inductor/test_check_accuracy.py, test/_inductor/test_debug_msg.py,
  test/_inductor/test_force_fallback.py
- Defect: 启用`INDUCTOR_ASCEND_DUMP_FX_GRAPH`时，`lowering_fx.py`的lowering ops需覆盖
  默认lowering。但`LOWERING_OVERRIDE_OP`未包含dump路径需要的ops(convert_element_type,
  where, broadcast_tensors等约50个ops)，触发`NotImplementedError`。三个测试类被
  `@skip`掩盖了问题。
- Fix: 新增`DUMP_FX_GRAPH_LOWERING_OPS`列表包含全量ops，union合并到
  `LOWERING_OVERRIDE_OP`。删除三个测试的`@skip`。
- Reviewability: medium -- lowering注册表需与实际ops保持同步
- Review rule: 新增lowering路径需列举所有可能的aten ops并注册；skip的测试需有issue跟踪

### D-60: NPU dropout mask格式不同于upstream，分布式sharding strategy缺失

- Hashes: 761cff284 [+1: d52abb7a0]
- Root cause: NPU算子语义偏离后分布式sharding策略未注册
- Files: torch_npu/distributed/tensor/_matrix_ops.py,
  test/distributed/tensor/test_matrix_ops.py
- Defect: NPU上dropout算子的mask格式与PyTorch原生社区不同(bit-packed uint8 vs bool)，
  但DTensor的sharding strategy仍沿用upstream默认实现。默认strategy对mask按element维度
  切分，但NPU的bit-packed mask的shard边界与数据tensor不对齐，导致分布式dropout结果错误。
- Fix: 注册`aten.native_dropout.default`和`aten.native_dropout_backward.default`的自定义
  strategy。前向strategy根据input placement生成output specs(数据沿原维度shard，mask固定
  Shard(0))；反向strategy直接透传input specs。+53行策略代码+59行测试。
- Reviewability: medium -- 需要知道NPU dropout mask格式与upstream的差异
- Review rule: NPU算子行为偏离upstream语义时，必须同步检查DTensor sharding strategy是否需要适配

### D-61: 分布式pipelining测试未适配新版PyTorch API

- Hashes: 8c340c09f [+2: 56971a7d4, 57414ce67]
- Root cause: upstream PyTorch测试基础设施重构后未同步
- Files: test/distributed/pipelining/{model_registry.py, schedule_registry.py,
  test_schedule_multiproc.py, test_stage.py}
- Defect: PyTorch 2.9.0对pipelining测试模型做了大幅重构: ExampleCode/ModelWithKwargs增加
  splits参数和更多层，新增MLPKWargModule/MultiMLPKwargs类，CustomLinearDx/DxDw的backward
  解包方式改变(bias不再使用)，init权重范围缩小(0.01→0.001)。torch_npu的测试脚本未同步导致
  UT失败。
- Fix: 从PyTorch 2.9.0社区同步全部4个测试文件。+1033行-756行，净变更277行。
- Reviewability: low -- 需要持续跟踪upstream测试文件变更，属机械同步工作
- Review rule: 版本分支创建后应自动比对upstream测试文件差异，生成同步checklist

### D-62: profiler kernel_details.csv表头过滤条件语义错误

- Hashes: 30aaa07f4 [+5: e506bcb47, ab9d5d862, 2f9aa0594, dc133ed4d, ce103a784]
- Root cause: 条件判断语义错误(用aicore_metrics判断代替profiler_level判断)
- Files: torch_npu/profiler/analysis/_profiler_config.py,
  test/profiler/test_export_memory_timeline.py
- Defect: `is_all_kernel_headers()`用`_ai_core_metrics != AicMetricsNone`判断是否展示
  完整表头。但实际上只有L0场景才应过滤表头，L1场景即使aicore_metrics=None也应展示完整
  shape字段。原条件在"L1 + aicore_none"组合下误判，导致kernel_details.csv缺失shape列。
- Fix: 条件改为`_profiler_level != LEVEL0`。同时移除test中两个`@unittest.skip`和多余的
  `has_result`断言，修复测试的output_path使用绝对路径。
- Reviewability: high -- 单行条件错误，审查时应追问"这个条件的语义到底是什么"
- Review rule: 条件判断涉及多个正交维度(level, metrics)时，应明确写出decision matrix而非用
  一个维度代理另一个

### D-63: aclrtPointerGetAttributes调用前缺少驱动版本检查

- Hashes: f8a0f174c
- Root cause: API可用性检查遗漏驱动版本维度
- Files: torch_npu/csrc/core/npu/interface/AclInterface.cpp
- Defect: `AclrtPointerGetAttributesExist()`只检查了CANN runtime版本>=8.5.0，但该API
  同时依赖驱动版本>=25.5.0。在新CANN+旧驱动的组合下，API不可用但检查通过，调用时失败。
- Fix: 在runtime版本检查前增加`IsGteDriverVersion("25.5.0")`守卫。+4行。
- Reviewability: medium -- 需要知道API的驱动依赖关系，通常不在CANN文档中显式标注
- Review rule: 新增runtime API调用时，同时确认驱动版本要求；API可用性检查应覆盖所有依赖维度
  (runtime版本、驱动版本、SoC型号)

### D-64: kSynchronizeBusyWaitMillis值未对齐upstream(10ms→1ms)

- Hashes: 336b176a5
- Root cause: 常量值跨版本未同步
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `kSynchronizeBusyWaitMillis`硬编码为10ms(对应PyTorch 2.1.0时代的值)，但
  PyTorch 2.7.1及后续版本已改为1ms。在HCCL_BLOCKING_WAIT=1场景下，小tensor
  all_reduce每次同步多等~9ms(实测median 10.078ms vs 1.076ms)。
- Fix: 常量值从10改为1。单行变更。
- Reviewability: high -- 纯常量值，版本适配时应逐个比对upstream常量定义
- Review rule: 版本分支创建时，应自动扫描所有硬编码常量与upstream对应值的diff

### D-65: CachingHostAllocator析构函数在NPU runtime关闭后调用设备API导致core dump

- Hashes: 322588027
- Root cause: 析构顺序与runtime生命周期不匹配
- Files: torch_npu/csrc/core/npu/CachingHostAllocator.cpp
- Defect: Python进程退出时GC触发tensor析构，`HostAllocator::free()`无条件调用
  `insertEvents()`操作NPU stream/event。但此时NPU runtime可能已teardown，ACL调用
  失败抛异常。析构函数内抛异常→`std::terminate()`→core dump。DeepSeek-V3训练退出
  时复现。
- Fix: 用`NpuSysCtrl::GetInstance().GetInitFlag()`守卫`insertEvents()`调用。runtime
  已关闭时跳过event插入，允许block在进程退出时安全泄漏。对齐device allocator的shutdown行为。
- Reviewability: medium -- 需理解进程退出时Python GC与NPU runtime的teardown顺序
- Review rule: 所有在析构函数/destructor中调用设备API的路径，必须检查runtime是否仍然存活

### D-66: inductor tile generator提前break导致vector_core=40时生成非法config

- Hashes: 2dc36a9e9
- Root cause: 循环控制流错误(break中断搜索空间遍历)
- Files: torch_npu/_inductor/codegen/tile_generator.py
- Defect: `descend_one_axis()`中的`break`语句在找到第一个候选block后立即退出，不再继续
  搜索更优解。当vector_core=40(某些Ascend产品AI core数为20)时，第一个候选的tiling方案
  超出UB容量，但更优的方案在后续迭代中。同时外层`while True`无上界，理论上可能无限循环。
- Fix: 移除`break`让搜索继续；`while True`改为`for max_idx in range(30)`加上界。
- Reviewability: medium -- 需理解tiling搜索算法的终止条件设计
- Review rule: 搜索算法中的early-exit条件(break/return)需证明不会跳过有效解；
  无界循环(while True)应有明确的终止证明或fallback上界

### D-67: COW(Copy-on-Write) lazy_clone未适配NPU导致core dump

- Hashes: f4e5747fb [+5: aa92d145a, a7ff9f24c, 6777fa347, cff185731, 967c03ea3]
- Root cause: PyTorch新feature(COW机制)适配缺失
- Files: torch_npu/csrc/aten/common/TensorFactories.cpp,
  torch_npu/csrc/aten/npu_native_functions.yaml,
  torch_npu/csrc/core/NPUStorageImpl.cpp,
  torch_npu/csrc/core/npu/NPUCachingAllocator.cpp,
  torch_npu/csrc/core/npu/NPUSwappedMemoryAllocator.cpp,
  torch_npu/csrc/core/npu/NPUWorkspaceAllocator.cpp,
  torch_npu/csrc/npu/NPUPluggableAllocator.cpp
- Defect: PyTorch的COW/lazy_clone通过`copy_data()`虚函数实现写时复制。NPU的4个
  allocator都用`default_copy_data()`(CPU memcpy)处理device指针→非法内存访问→core dump。
  同时`_lazy_clone` op未在NPU注册，`make_npu_storage_impl`禁止size==0时传入非空
  data_ptr，阻碍COW storage创建。
- Fix: 三层修复: 1)实现NPU `_lazy_clone`(调用c10::impl::cow::lazy_clone_storage +
  StorageDescHelper::CopyDesc); 2)所有allocator的copy_data改用aclrtMemcpy
  DEVICE_TO_DEVICE; 3)放宽StorageImpl对size==0+non-null data_ptr的限制。
- Reviewability: medium -- 需跟踪upstream新增的虚函数/op接口并逐个适配
- Review rule: upstream新增allocator虚函数时，检查所有NPU allocator子类是否需要override

### D-68: swapped memory测试缺少数值正确性校验

- Hashes: f5589e3f1
- Root cause: 测试断言不完整(仅验证生命周期，缺少数值验证)
- Files: test/npu/test_swapped_memory_allocator.py
- Defect: `test_02_async_operations_and_release`只验证tensor被GC回收(weakref变None)，
  但未验证计算结果的数值正确性。swapped memory上的异步计算可能产生错误结果而不被发现。
- Fix: 增加expected_tensor独立计算 + `torch.allclose`断言。+9行。
- Reviewability: high -- 测试review应检查"测试了什么没测什么"，缺少数值断言是明显遗漏
- Review rule: 涉及计算的测试必须同时验证数值正确性，不能仅验证不crash

### D-69: inductor user_autotune/PrecomputedGrid未适配NPU导致core dump

- Hashes: 9f030ebcd
- Root cause: upstream inductor feature假设CUDA runtime，NPU未适配
- Files: torch_npu/_inductor/codegen/wrapper.py,
  torch_npu/_inductor/npu_triton_heuristics.py,
  test/_inductor/test_user_autotune_npu.py
- Defect: PyTorch inductor的`user_autotune`和`PrecomputedGrid`直接调用CUDA API
  (如grid计算、kernel启动方式)。在NPU上使用triton.autotune装饰的用户kernel时，
  torch.compile生成的代码引用upstream的`user_autotune()`→调用CUDA runtime→core dump。
  另外benchmark路径缺少`reset_to_zero_args`调用，autotuning精度不可靠。
- Fix: 1)新建`user_autotune_npu()`和`PrecomputedGridNpu`适配NPU; 2)在
  `NPUWrapperCodeGen.define_kernel`中拦截生成代码，将upstream引用替换为NPU版本;
  3)补上benchmark和kernel_call中的`reset_to_zero_args`调用。
- Reviewability: medium -- 需要了解inductor的kernel codegen pipeline
- Review rule: upstream新增heuristic/grid类型时，确认NPU是否需要对应适配

### D-70: swapped memory svm_deleter空实现导致内存泄漏

- Hashes: 8b4c3e678
- Root cause: 资源释放函数未实现(空函数体)
- Files: torch_npu/csrc/core/npu/NPUSwappedMemoryAllocator.cpp,
  test/npu/test_swapped_memory_allocator.py
- Defect: `svm_deleter()`函数体为空(只有大括号)。作为`DataPtr`的deleter，它负责释放
  swapped memory的三层资源: unregister主机内存映射、释放主机内存、清理映射表条目。
  空实现导致所有swapped memory tensor的主机内存永远不被释放，累积内存泄漏。
- Fix: 实现完整释放流程: null检查→memBlocks查找→npuSynchronizeDevice→
  AclrtHostUnregister→aclrtFreeHost→erase。同时新增完整UT覆盖分配/释放和异步场景。
- Reviewability: high -- 空函数体在code review中应该立即被发现
- Review rule: deleter/destructor/cleanup函数不允许为空(除非有显式注释说明为何不需要清理)


### D-71: mlir_enable测试补充 + test_resize_as残留字符修复

- Hashes: 1d637cec4
- Root cause: 代码编辑残留字符(stray character)
- Files: test/_inductor/test_mlir_enable.py (new),
  test/_inductor/test_resize_as.py
- Defect: `test_resize_as.py`文件末尾残留了一个多余的字符`s`（紧跟在`run_tests()`之后），
  且文件缺少末尾换行。这种残留字符可能导致语法错误或测试执行异常。
  同时MLIR backend缺少基本测试覆盖。
- Fix: 删除`test_resize_as.py`末尾的stray `s`字符；新增`test_mlir_enable.py`覆盖
  MLIR backend的`options`和`config`两种启用方式。
- Reviewability: high -- 残留字符在任何diff review中都应被发现
- Review rule: 文件末尾修改后检查是否有stray characters和missing newline at EOF

### D-72: FA pattern示例中inv_scale_factor设备不匹配

- Hashes: dd0b44f52
- Root cause: tensor创建时设备参数传递链断裂
- Files: torch_npu/_inductor/npu_fusion_attention_graph.py
- Defect: `register_fa_pass()`中，`c_inp`的partial使用`device=device`（局部变量），
  但`device`可能是字符串描述或不完整的设备对象，与`g_inp()`实际分配到的NPU设备不一致。
  当FA pattern matching时，scale_factor tensor和qkv tensor在不同设备上，导致计算失败。
- Fix: `device=device` → `device=g_inp().device`，确保从实际tensor获取设备信息。
- Reviewability: medium -- 需要理解partial延迟执行时device变量的绑定语义
- Review rule: tensor factory的device参数应从已有tensor获取而非使用外部变量

### D-73: test_schedule UT适配upstream API重命名 + 异常安全

- Hashes: d2ac93043
- Root cause: upstream PyTorch API重命名未同步 + 测试异常安全缺陷
- Files: test/distributed/pipelining/test_schedule.py
- Defect: 两个问题: 1) PyTorch upstream将`_load_actions`重命名为
  `_prepare_schedule_with_comms`，调用方未同步; 2) test方法中`destroy_process_group()`
  写在assertRaises之后，如果断言失败则process_group泄漏，后续测试全部失败。
- Fix: 1)更新方法名; 2)用`try/finally`包裹确保cleanup; 3)增加`OptimizedModule`类型断言。
- Reviewability: medium -- API重命名需要跟踪upstream changelog
- Review rule: 分布式测试中process group cleanup必须在finally块中

### D-74: ONNX算子测试批量跳过

- Hashes: 21039cff3
- Root cause: NPU自定义ONNX算子适配失效，选择跳过测试
- Files: test/onnx/test_combined_onnx_ops.py,
  test/onnx/test_wrapper_onnx_ops.py
- Defect: 12+个ONNX wrapper算子测试（group_norm_silu, rotary_mul, npu_one_hot,
  npu_slice, npu_roi_align, npu_iou, npu_fast_gelu, npu_geglu, npu_multi_head_attention,
  npu_diou, npu_ciou, npu_giou, npu_deformable_conv2d, npu_format_cast等）全部失败。
  未修复底层问题，直接用`@unittest.skip("skip now")`禁用。
- Fix: 批量添加`@unittest.skip("skip now")`跳过失败测试。
- Reviewability: high -- 批量skip是债务信号，review时应要求对应的tracking issue
- Review rule: `@unittest.skip`必须附带issue编号和预期修复时间，禁止"skip now"这类无期限跳过

### D-75: StorageWeakRefWrapper引用计数逻辑修复

- Hashes: 3f5607af6
- Root cause: NPU自定义storage引用计数实现与upstream语义偏离
- Files: torch_npu/npu/_graph_tree.py,
  torch_npu/csrc/npu/Module.cpp,
  test/npu/test_graph_tree.py
- Defect: 三个问题:
  1) `expired()`方法用NPU自定义的`torch_npu._C._storage_Use_Count`而非upstream的
     `torch._C._storage_Use_Count`，两者在有`extra_ref_check`时语义不同;
  2) extra_ref偏移量为1(只减去extra_ref_check本身的引用)，但NPU场景需要减去2
     (Python storage对象引用 + cached Tensor引用);
  3) `from_weakref_and_data_ptr`的type annotation用`Type[S]`泛型但S未定义。
  这导致NPU graph tree的storage存活判断不准确，graph replay时可能复用已释放的storage。
- Fix: 1)切换到upstream `torch._C._storage_Use_Count`; 2)extra_ref偏移改为2;
  3)移除NPU自定义C++绑定`_storage_Use_Count`; 4)大幅扩充UT覆盖。
- Reviewability: low -- 引用计数语义需要深入理解storage生命周期
- Review rule: NPU对upstream类的monkey-patch/override必须有对比upstream行为的注释

### D-76: get_device_properties触发NN进程数超限

- Hashes: ee2ccd7fb [+5 cherry-picks: a073707a2, 7f721effd, 7434319c3, 728471e82, adce5117f]
- Root cause: 硬件API副作用 -- aclrtGetMemInfo内部创建NN进程
- Files: third_party/acl/inc/acl/acl_rt.h,
  torch_npu/csrc/npu/Module.cpp
- Defect: `initDeviceProperty()`调用`aclrtGetMemInfo(ACL_HBM_MEM, ...)`查询显存总量。
  但该API内部会创建一个Neural Network进程。当频繁调用或多进程场景下（如分布式训练启动时
  每个rank都查询设备属性），NN进程数累积超过系统限制，报错"NN processes exceeds the limit"。
- Fix: CANN 9.0.0.beta2+版本改用新API `aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_TOTAL_GLOBAL_MEM_SIZE, ...)`
  查询显存总量，该API无NN进程副作用。低版本CANN回退到旧API。
  同时扩展了`aclrtDevAttr`枚举，增加20+新设备属性查询能力。
- Reviewability: low -- 硬件API的进程副作用从接口签名无法推断
- Review rule: 调用CANN/ACL底层API前确认是否有资源分配副作用(进程、内存、设备锁)

### D-77: with device负数index未正确忽略

- Hashes: c6f2bcc52
- Root cause: 负值边界条件处理不完整(仅检查-1而非<0)
- Files: torch_npu/npu/utils.py,
  test/distributed/test_with_device.py
- Defect: `torch.npu.device`上下文管理器的`__enter__`仅检查`self.idx == -1`来跳过设备切换。
  但`_get_device_index`对负数浮点值（如-128.8, -7.88, -0.2）进行int转换后可能得到
  -128, -7, 0等非-1的值。`__exit__`也没有处理负数index的早期返回，导致尝试exchange到
  无效设备。
- Fix: `__enter__`: `== -1` → `< 0`; `__exit__`增加`< 0`的早期返回并重置idx为-1。
- Reviewability: high -- 经典边界条件bug，`== -1`应自然扩展为`< 0`
- Review rule: 设备index的无效值检查统一用`< 0`而非`== -1`

### D-78: npugraph_ex scope.limit_core_num功能恢复

- Hashes: 21cbb8715
- Root cause: 功能模块未注册/暴露
- Files: torch_npu/npu/npugraph_ex/__init__.py,
  torch_npu/npu/npugraph_ex/scope/__init__.py (new),
  test/dynamo/test_npugraph_ex.py (new),
  test/torch_npu_schema.json
- Defect: `npugraph_ex`的`scope.limit_core_num()`API（用于在graph编译时限制AI Core
  和Vector Core使用数量）未暴露给用户。`scope`子模块未在`__init__.py`中注册，
  `limit_core_num`函数未在schema中声明。用户无法通过`torch.npu.npugraph_ex.scope`访问该功能。
- Fix: 1)新建`scope/__init__.py`，提供`limit_core_num`入口函数;
  2)在`npugraph_ex/__init__.py`中`import scope`; 3)更新schema.json;
  4)新增综合测试文件覆盖backend/compile_fx/cache_compile/limit_core_num四个场景。
- Reviewability: medium -- API暴露缺失需要端到端功能验证才能发现
- Review rule: 新增子模块必须在父模块__init__.py中注册并更新schema

### D-79: aclrtMallocHostWithCfg缺少驱动版本检查

- Hashes: 503381ca1
- Root cause: API可用性检查遗漏驱动版本维度(同D-63模式)
- Files: torch_npu/csrc/core/npu/interface/AclInterface.cpp,
  torch_npu/csrc/core/npu/NPUAllocatorConfig.cpp
- Defect: `AclrtMallocHostWithCfgExist()`仅检查CANN runtime版本(>=8.5.0)，
  未检查driver版本。当CANN版本满足但driver版本低于25.5.2时，API符号存在但调用时
  底层driver不支持，导致运行时错误。
- Fix: 在runtime版本检查前增加`IsGteDriverVersion("25.5.2")`检查;
  同时更新warning消息，明确告知用户需要同时升级CANN和driver。
- Reviewability: medium -- 需要了解CANN/driver版本矩阵兼容关系
- Review rule: 新硬件API引入时，version guard必须同时覆盖CANN版本和driver版本

### D-80: npu_format_cast的customize_dtype应为keyword-only参数

- Hashes: 8831262dc
- Root cause: op schema中positional/keyword-only参数声明错误
- Files: torch_npu/csrc/aten/npu_native_functions.yaml,
  test/torch_npu_schema.json
- Defect: `npu_format_cast`系列4个op的`customize_dtype`参数在schema中声明为positional参数
  (在`*`之前)。这意味着用户可以用positional方式传入，例如
  `npu_format_cast(tensor, 2, 3)` 其中3会被解析为customize_dtype。
  但该参数语义上是可选配置，应该强制keyword-only以防误用。
- Fix: 在yaml和json schema中将`customize_dtype`移到`*`之后(keyword-only区域)。
  影响4个函数变体: npu_format_cast, npu_format_cast_, npu_format_cast_.acl_format,
  npu_format_cast.Tensor。
- Reviewability: high -- schema声明在yaml中一目了然
- Review rule: 可选配置参数在op schema中必须声明为keyword-only(放在`*`之后)

### D-81: 设备错误信息获取触发无限递归初始化

- Hashes: c90658d5b
- Root cause: 错误处理路径触发设备初始化 → 初始化失败 → 再次触发错误处理 → 无限递归
- Files: torch_npu/csrc/core/npu/NPUException.cpp
- Defect: `cacheDeviceErrorVerboseMsg()`调用`c10_npu::current_device()`获取当前设备ID。
  但`current_device()`在设备未初始化时会触发设备初始化流程。如果初始化本身失败，
  错误处理路径会再次调用`cacheDeviceErrorVerboseMsg()`，形成无限递归。
  典型场景: 首次设备操作失败时，错误日志系统尝试获取设备信息导致stack overflow。
- Fix: 改用`GetLocalDevice()`(仅读取本地缓存的device ID，不触发初始化)，
  并在`device < 0`(未初始化)时直接返回空字符串。
- Reviewability: medium -- 需要理解current_device()的初始化副作用
- Review rule: 错误处理/日志路径禁止调用可能触发初始化或分配资源的函数

### D-82: profiler加载路径校验缺失导致误报

- Hashes: fd6927797 [+1 cherry-pick: 15b82b22a]
- Root cause: profiler配置加载逻辑缺少数据路径存在性校验
- Files: torch_npu/profiler/analysis/_profiler_config.py,
  test/profiler/analysis/test_profiler_config.py
- Defect: `ProfilerConfig.load_info()`不校验CANN profiling数据路径是否存在。
  当用户配置了`ProfilerActivity.NPU`但CANN profiling数据不存在时（如CANN未安装或
  profiling未启用），后续分析代码会尝试读取不存在的路径并报出令人困惑的错误。
  同时`_is_cluster`字段基于文件名正则判断，逻辑脆弱且已不再需要。
  此外`load_info()`每次调用都重新加载，浪费资源。
- Fix: 1)新增`load_activities()`校验FWK/CANN路径存在性，不存在时移除对应activity并报错;
  2)移除`_is_cluster`相关逻辑; 3)增加`_is_load`标志防止重复加载。
- Reviewability: medium -- profiler配置加载路径需要端到端测试验证
- Review rule: profiler/analysis工具在加载数据前必须校验路径存在性并给出明确错误信息

### D-83: DeviceProperty.multi_processor_count未设置导致DataParallel失效

- Hashes: c1e1d3461
- Root cause: PyTorch框架依赖的设备属性字段NPU未填充
- Files: torch_npu/csrc/npu/Module.cpp,
  test/npu/test_torch_npu.py
- Defect: PyTorch的`DataParallel`使用`multi_processor_count`进行负载均衡
  (计算各GPU的计算能力比例)。NPU设备未设置此值(默认为0)，导致DataParallel的
  balance计算除以零或分配不均。NPU虽然没有"multiprocessor"概念，但需要提供
  等效的计算单元数量作为负载均衡参考。
- Fix: 在`initDeviceProperty`中设置`multi_processor_count`: 优先用`vector_core_num`，
  若为0则回退到`cube_core_num`。从unsupported字段列表中移除`multi_processor_count`。
- Reviewability: medium -- 需要了解DataParallel对设备属性的依赖
- Review rule: 新增NPU设备属性映射时，检查PyTorch框架层是否有对该属性的隐式依赖

### D-84: checkpoint测试异常类型断言不匹配

- Hashes: 4b042a40b
- Root cause: upstream异常类型变更后测试断言过严
- Files: test/distributed/checkpoint/test_checkpoint.py
- Defect: `TestDistributedFailure`中`_test_load`的`fail_read_metadata`场景期望精确匹配
  异常类型。但upstream PyTorch在checkpoint加载路径上变更了异常类型（可能从一种
  RuntimeError子类变成了另一种），导致测试因类型不匹配而失败，即使错误行为本身是正确的。
- Fix: 在3处`fail_read_metadata`调用中添加`ignore_exception_type=True`参数，
  放宽异常类型匹配。
- Reviewability: high -- 测试断言精度选择应在review中讨论
- Review rule: 对upstream异常的断言优先验证message内容而非精确类型

### D-85: Tensorpipe子模块更新

- Hashes: 6744f15df
- Root cause: 第三方依赖版本缺陷
- Files: third_party/Tensorpipe (submodule 95ecfbad8 → 483a5907c)
- Defect: Tensorpipe子模块指向的旧版本存在缺陷（具体缺陷在Tensorpipe仓库的commit
  483a5907c中修复，可能涉及NPU transport支持或编译兼容性问题）。
- Fix: 更新子模块指针到修复后的commit。
- Reviewability: low -- 子模块更新需要查看Tensorpipe仓库的变更才能评估影响
- Review rule: 子模块更新的PR描述中应包含被修复问题的简要说明

### D-86: serialization.save硬编码True覆盖用户配置

- Hashes: 380968919
- Root cause: 函数参数传递时硬编码常量替代了变量
- Files: torch_npu/utils/serialization.py
- Defect: `torch_npu.utils.serialization.save()`在调用`torch.serialization.save()`时，
  第5个参数(`_use_new_zipfile_serialization`)被硬编码为`True`，而非传递用户指定的
  `_use_new_zipfile_serialization`变量。这意味着即使用户显式设置
  `_use_new_zipfile_serialization=False`（使用旧格式序列化），也会被强制使用新格式。
  对于需要与旧版PyTorch兼容的模型保存场景，这个bug会导致生成的文件无法被旧版加载。
- Fix: 将硬编码的`True`替换为`_use_new_zipfile_serialization`变量。
- Reviewability: high -- 单行修改，函数调用参数一一对应检查即可发现
- Review rule: 函数调用的参数必须与对应的变量名匹配，禁止在wrapper中用常量替代可配置参数

### D-87: 动态符号导致fold_sink_view崩溃 + 移除冗余pass

- Hashes: b2f171ffb
- Root cause: graph pass对动态符号节点缺乏None保护
- Files: torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py,
  test/_inductor/test_unfold_dual_reduction_pass.py (deleted)
- Defect: `fold_sink_view` pass中，`get_node_shape(node)`对含动态符号(SymInt)的view节点
  可能返回None。原代码在使用`view_shape`之前没有null检查，且`view_shape`的获取位置
  在循环体的后半段，导致前面的逻辑已经对user节点做了修改后才发现shape为None。
  此外`unfold_dual_reduction_pass`已不再需要（上游inductor修复了dual reduction支持），
  但代码和测试仍保留着，增加维护成本和执行时间。
- Fix: 1)将`view_shape`获取提前到循环体开头，None时`continue`跳过;
  2)完全删除`unfold_dual_reduction_pass`(60行)及其测试(199行)。
- Reviewability: medium -- 需要理解dynamic shape时meta信息可能为None
- Review rule: graph pass中使用get_node_shape/get_node_meta前必须null检查

### D-88: sympy常量被误判为动态符号导致cat走fallback

- Hashes: a32837844
- Root cause: sympy类型层次导致isinstance检查过宽
- Files: torch_npu/_inductor/lowering.py
- Defect: `_is_dynamic()`用`isinstance(s, (sympy.Symbol, sympy.Expr))`判断shape是否动态。
  但sympy的类型层次中，`sympy.Integer(3)`也是`sympy.Expr`的子类。这导致所有shape维度
  （包括静态常量如3, 1024等）都被判为"动态"，`aten.cat`无条件走fallback路径，
  绕过了NPU优化的concat_kernel实现，性能严重退化。
- Fix: 增加`len(s.free_symbols) > 0`条件——只有包含自由符号变量的表达式才是真正动态的。
  `sympy.Integer(3).free_symbols`为空集，不再被误判。
  同时移除了dynamic path中不相关的fp64→fp32转换逻辑。
- Reviewability: medium -- 需要了解sympy类型层次结构
- Review rule: sympy表达式的"动态"判断必须检查free_symbols，不能仅依赖isinstance

### D-89: inductor精度工具patch的IR节点元信息传递不完整

- Hashes: 7db57459f
- Root cause: IR节点嵌套结构中元信息(traced_graph, node_name)只设置了外层
- Files: torch_npu/_inductor/codegen/ir_fx.py
- Defect: 两个问题:
  1) `_patch_reduction_create`: `traced_graph`和`node_name`只设置在reduction节点`r`上，
     但未设置在`r.data.data`(内部IR节点)上。精度工具通过内部节点的`traced_graph`属性
     追踪op来源，外层设置了但内层没设置导致追踪链断裂;
  2) `_patch_concatkernel_create`: 创建concat_kernel后未调用
     `V.graph.register_operation(concat_kernel)`，导致该op不出现在operation注册表中，
     精度分析工具无法遍历到concat操作。
- Fix: 1)在两个split路径中对`r.data.data`也设置`traced_graph`和`node_name`;
  2)concat_kernel创建后调用`V.graph.register_operation()`注册。
- Reviewability: low -- 需要深入理解inductor IR的嵌套结构和operation注册机制
- Review rule: IR节点创建后，元信息设置必须覆盖所有嵌套层级

### D-90: aclgraph测试用例910C设备名称错误

- Hashes: 9eaf2f3bd [+1 cherry-pick: 9ff7ea1d3]
- Root cause: 硬件SoC内部标识与用户可见名称不一致
- Files: test/npu/test_aclgraph_launch_host_func.py,
  test/npu/test_aclgraph_support_blocking.py,
  test/npu/test_aclgraph_update.py
- Defect: `@SupportedDevices`装饰器中使用`Ascend910C`作为设备标识，
  但NPU驱动返回的实际SoC标识为`Ascend910_93`。名称不匹配导致910C设备上的
  aclgraph测试被跳过（误判为不支持），失去了该硬件上的测试覆盖。
- Fix: 将所有`Ascend910C`替换为`Ascend910_93`（8处修改，跨3个测试文件）。
- Reviewability: high -- 设备名称映射应有统一常量表，避免硬编码
- Review rule: 设备标识不得在测试中硬编码字符串，应引用统一的SoC名称常量表

### D-91: load函数mmap路径仅接受str不接受PathLike

- Hashes: d43940c8e
- Root cause: upstream API签名扩展后NPU实现类型检查过窄
- Files: torch_npu/utils/serialization.py
- Defect: NPU的`load()`实现在mmap分支中用`isinstance(f, str)`检查文件参数，
  排除了`os.PathLike`对象。upstream PyTorch已支持PathLike，导致用户传Path对象时
  NPU侧报错而upstream正常。此外`from_file()`未调用`os.fspath()`转换路径类型。
- Fix: 引入`_is_path()`辅助函数（TypeGuard），同时检查str和PathLike；
  `from_file()`参数改用`os.fspath(f)`确保转为str。
- Reviewability: high -- 类型检查与upstream对比即可发现差异
- Review rule: NPU override的公共API类型检查须与upstream签名一致

### D-92: clone操作未保留Preserve格式下的stride布局

- Hashes: a260b2448
- Root cause: tensor factory忽略MemoryFormat.Preserve语义（stride丢失）
- Files: torch_npu/csrc/aten/ops/op_api/CloneKernelOpApi.cpp
- Defect: `clone()`实现始终用`apply_tensor_without_format()`创建目标tensor，
  该函数不考虑源tensor的stride布局。当format=Preserve且源tensor是dense时，
  clone结果丢失原始stride信息，与PyTorch语义不一致（upstream会用empty_strided保留）。
- Fix: 当memory_format==Preserve且源tensor is_non_overlapping_and_dense()时，
  用`at::empty_strided(src.sizes(), src.strides(), src.options())`创建dst。
- Reviewability: medium -- 需理解MemoryFormat.Preserve的语义契约
- Review rule: NPU tensor创建路径必须区分MemoryFormat分支，Preserve需保留stride

### D-93: as_strided使用NPU自定义setStrided偏离upstream语义

- Hashes: 7492ad614
- Root cause: NPU自定义实现与upstream分叉后语义偏离
- Files: torch_npu/csrc/aten/common/ResizeNpu.h, torch_npu/csrc/aten/common/TensorShape.cpp
- Defect: NPU有独立的`checkInBoundsForStorage()`和`setStrided()`实现（60行），
  最初可能因upstream缺少某些检查而添加。但随upstream迭代，两者行为已分叉，
  NPU版本的边界检查和stride设置逻辑与upstream不一致，导致as_strided结果偏离预期。
- Fix: 删除NPU自定义的checkInBoundsForStorage和setStrided（-60行），
  as_strided中改用`at::native::setStrided()`。
- Reviewability: medium -- 需对比NPU实现与upstream的行为差异
- Review rule: 与upstream有同名实现的NPU自定义函数，需定期review是否仍有存在必要

### D-94: npu_dtype_cast缺少分布式tensor的pointwise策略注册

- Hashes: 6b4394105 [+1 cherry-pick: b956cdc34]
- Root cause: NPU自定义op缺少分布式tensor策略注册
- Files: torch_npu/distributed/tensor/__init__.py, torch_npu/distributed/tensor/_pointwise_ops.py,
  test/distributed/_tensor/test_pointwise_ops.py
- Defect: `npu.npu_dtype_cast`等NPU pointwise op没有注册DTensor分发策略。
  当配合`_StridedShard` placement使用时，DTensor框架不知道如何propagate sharding，
  导致redistribute失败或结果错误。
- Fix: 新增`_pointwise_ops.py`模块，将npu_dtype_cast系列op注册为pointwise策略
  （linearity=0），在__init__.py中导入。
- Reviewability: medium -- 新增NPU op时容易遗忘分布式策略注册
- Review rule: 每个新增NPU custom op必须同步注册DTensor策略

### D-95: npugraphs后端硬编码is_inference=False

- Hashes: ffa1aa316
- Root cause: 参数传递遗漏（硬编码常量替代已计算变量）
- Files: torch_npu/utils/_graph_tree.py, test/dynamo/test_npugraphs.py
- Defect: `npugraphs()`后端调用capture时将`is_inference`硬编码为`False`，
  即使上层已正确计算了`is_inference`变量。这导致inference_mode下的workload
  被当作training模式capture，每次replay都触发重复capture和warned_functions警告。
- Fix: 将`is_inference=False`改为`is_inference=is_inference`，传递实际值。
- Reviewability: high -- 典型的"变量声明了但传参时用了常量"，diff一行即可发现
- Review rule: 函数调用中出现硬编码布尔常量时，检查上下文是否有同名变量应该被传递

### D-96: inductor lowering中evaluate_static_shape未跟进upstream重命名为guard_int

- Hashes: af5c93b5b [+1 cherry-pick: a2f6dd0ac]
- Root cause: upstream API重命名后NPU fork代码未同步
- Files: torch_npu/_inductor/ascend_npu_ir/.../wrapper.py,
  torch_npu/_inductor/ascend_npu_ir/.../lowering.py
- Defect: PyTorch upstream将`V.graph.sizevars.evaluate_static_shape()`重命名为
  `guard_int()`，但NPU inductor的lowering.py中8处调用仍使用旧名称。
  同时wrapper.py缺少`assert_alignment`的import声明，运行时NameError。
- Fix: 所有evaluate_static_shape替换为guard_int；wrapper.py添加assert_alignment导入。
- Reviewability: high -- upstream rename后全局grep旧名称即可发现所有遗漏
- Review rule: upstream API重命名时，CI应包含旧名称的grep检查确保无残留

### D-97: cudagraph兼容性检查未处理index_put的boolean indices

- Hashes: f00fae974
- Root cause: upstream兼容性检查函数未被NPU覆盖/适配
- Files: torch_npu/_inductor/__init__.py, torch_npu/_inductor/utils.py
- Defect: upstream的`get_first_incompatible_cudagraph_node()`会检查index_put的
  boolean indices（触发.nonzero()在graph capture期间不安全），但这个函数在NPU
  环境下未被正确调用或适配。导致包含bool index_put的graph被错误地capture，
  运行时产生non-deterministic结果。
- Fix: 在torch_npu中实现完整的`get_first_incompatible_cudagraph_node()`并
  monkey-patch到inductor utils、compile_fx、cudagraphs三个模块（+116行）。
- Reviewability: low -- 需要理解cudagraph capture对动态op的限制
- Review rule: upstream新增的graph兼容性检查函数需同步评估NPU是否需要覆盖

### D-98: 缺少getMemoryFraction接口导致内存配额不可查询

- Hashes: 2a110a7b8 [+5 cherry-picks: 2bb674096, 423c90ae7, cb7166e01, 68f0940f4, 7ecdce1eb]
- Root cause: API对称性缺失（有setter无getter）
- Files: torch_npu/csrc/core/npu/NPUCachingAllocator.cpp, torch_npu/npu/memory.py,
  torch_npu/csrc/npu/NPUPluggableAllocator.h, torch_npu/dynamo/trace_rule.py,
  torch_npu/npu/__init__.py, test/test_npu.py
- Defect: `set_per_process_memory_fraction()`已存在但缺少对应的getter。
  用户设置内存配额后无法查询当前值，dynamo trace规则也缺少该函数的注册，
  torch._C绑定中不存在_npu_getMemoryFraction。
- Fix: 在NPUCachingAllocator中实现`getMemoryFraction()`（未set时返回1.0），
  在Python层暴露`get_per_process_memory_fraction()`，注册dynamo trace规则。
- Reviewability: high -- setter/getter对称性检查是基础review项
- Review rule: 每个set_*接口必须有对应的get_*接口

### D-99: flight recorder dump文件名缺少rank信息

- Hashes: dc8207073
- Root cause: 初始化时序问题（globalRank延迟求值导致依赖方获取不到值）
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: ProcessGroupHCCL构造函数中，`globalRank()`是lazy求值的。watchdog/
  flight recorder在构造时需要rank信息来生成dump文件名，但此时globalRank()
  可能尚未被调用过，导致文件名中rank字段为空或默认值。
- Fix: 在watchdog初始化前显式调用`globalRank()`触发rank值的计算和缓存。+1行。
- Reviewability: low -- 需要理解globalRank的lazy求值语义和flight recorder的文件名生成逻辑
- Review rule: lazy初始化的成员在被依赖方使用前，需确保已被触发

### D-100: test_init_process_group用例被错误移除

- Hashes: f8f1b77ab
- Root cause: 测试用例在重构中被误删
- Files: test/distributed/test_device_mesh.py
- Defect: DeviceMesh的`test_init_process_group`用例在此前的代码变更中被移除，
  导致DeviceMesh在非分布式环境下自动创建process group的流程缺乏测试覆盖。
- Fix: 恢复test_init_process_group用例，验证DeviceMesh初始化后is_initialized()为True。
- Reviewability: high -- 删除测试用例的PR应require justification
- Review rule: 删除测试用例必须说明原因，恢复测试不应被视为feature而是修复

### D-101: reduction codegen中any/prod使用了错误的Triton模块

- Hashes: 878a8cb28
- Root cause: Triton函数dispatch模块选择错误
- Files: torch_npu/_inductor/codegen/triton.py, test/_inductor/test_aten_any_prod_codegen.py
- Defect: NPU inductor的`final_reduction()`函数对所有reduction type统一使用`tl`模块，
  但Triton中`any`和`prod`是在`triton_helpers`模块而非`tl`中定义的。
  codegen生成的`tl.any(...)`和`tl.prod(...)`在运行时AttributeError。
- Fix: 添加`use_helper`判断，any/prod使用`triton_helpers`模块前缀。
- Reviewability: high -- 查Triton文档即可确认函数所属模块
- Review rule: codegen生成的模块引用必须与目标运行时的API位置一致

### D-102: RPC BackendType枚举注册后引用未同步到rpc模块

- Hashes: 864e16ded
- Root cause: Python模块级import引用在动态修改后未同步
- Files: torch_npu/distributed/rpc/backend_registry.py
- Defect: torch_npu通过`register_backend()`注册NPU RPC后端后，`BackendType`枚举
  被扩展了新成员。但`torch.distributed.rpc`模块中的`BackendType`是import时的快照，
  不会反映注册后的修改。用户通过`rpc.BackendType`访问时看不到NPU后端。
  与D-45（from-import独立引用致monkey-patch不传播）同一模式。
- Fix: 注册后显式将更新后的BackendType赋值回`_rpc_module.BackendType`。
- Reviewability: high -- Python import语义是已知陷阱，注册后应检查引用传播
- Review rule: 动态修改枚举/注册表后，检查所有通过from-import持有旧引用的模块

### D-103: Dockerfile基础镜像和Python版本号跨release未同步

- Hashes: cd8d62488
- Root cause: 构建基础设施版本号硬编码跨release不同步
- Files: ci/docker/ARM/Dockerfile, ci/docker/X86/Dockerfile
- Defect: Dockerfile使用hash-based镜像标签（`cpu-aarch64-66ed6468...`）和
  硬编码的cpython路径（3.10.18, 3.11.13, 3.12.11）。upstream更新了builder镜像
  到v2.10.0-rc7并bumped Python patch版本，但Dockerfile未同步。
  构建时找不到对应的cpython目录导致pip/python链接失败。
- Fix: 更新镜像标签和16处cpython路径版本号。
- Reviewability: high -- CI Dockerfile的版本号应有自动化检查
- Review rule: Dockerfile中的版本号应参数化（ARG），避免散落在多处硬编码

### D-104: torch.device被错误注册为TorchInGraphFunction导致compiled autograd trace失败

- Hashes: 638106a64
- Root cause: dynamo trace规则过度注册（torch.device不应是in-graph function）
- Files: torch_npu/utils/_dynamo.py, test/_inductor/test_compile_autograd.py
- Defect: `UserDefinedClassVariable.__new__`中将`torch.device`注册为
  `TorchInGraphFunctionVariable`。这导致compiled autograd的backward中，
  `torch.device("npu")`调用被dynamo尝试trace而非正常执行，trace失败报错。
  torch.device是一个类构造器而非纯函数，不应走in-graph function路径。
- Fix: 从in-graph函数列表中移除`torch.device`。
- Reviewability: high -- torch.device明显不是一个pure function
- Review rule: 注册in-graph function前须确认目标是无副作用的纯函数

### D-105: 集合通信操作中core control资源绑定不一致

- Hashes: ce153f7cd
- Root cause: 条件门控在多个代码路径中不一致（部分遗漏/部分多余）
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: gather, scatter, alltoall操作中`UseStreamResInCurrentThread()`的调用模式
  不一致：gather缺失该调用；scatter的一个路径无条件调用但缺少`is_core_control_enabled`
  门控；alltoall的groupStart后缺失该调用。core control功能在未启用时不应调用
  UseStreamResInCurrentThread，启用时必须在通信前调用，否则核心绑定状态错误。
- Fix: 在4处添加统一的`if (c10_npu::is_core_control_enabled)`门控和
  `UseStreamResInCurrentThread()`调用；scatter中1处从无条件改为门控调用。
- Reviewability: medium -- 需要跨多个集合通信函数横向比较调用模式
- Review rule: 特性门控的调用模式在所有同类函数中必须保持一致

### D-106: mlir模块动态编译加载路径使用函数局部变量做初始化保护

- Hashes: 10bc06261
- Root cause: 函数内局部变量作为单例初始化标志（每次调用重置为False）
- Files: torch_npu/utils/_dynamo.py
- Defect: mlir扩展采用运行时动态编译(`build_ascend_npu_ir_ext`)加载，用函数内局部变量
  `_has_inited`做幂等保护。Python函数每次调用都会重新初始化局部变量，保护完全失效，
  每次进入`patch_inductor_wrapper()`都重复触发编译。此外`set_torch_npu_library_path()`
  在部分环境下导致mlir模块路径解析失败。
- Fix: 删除动态编译路径，改为直接import预编译好的`npu_inductor_plugin`和`torch_mlir_patch`。
- Reviewability: medium -- 函数内flag做单例保护是Python常见陷阱，细看可发现
- Review rule: 初始化/幂等标志必须声明在模块级或类级，不能放在函数体内

### D-107: profiler export_type参数默认值与文档声明不一致

- Hashes: 44e0d99f6
- Root cause: 函数签名默认值与文档/schema声明不一致
- Files: torch_npu/profiler/experimental_config.py, test/torch_npu_schema.json
- Defect: `_ExperimentalConfig.__init__`中`export_type`参数默认值为`None`，但文档声明
  默认值为`"text"`。代码在`export_type is None`时fallback到`"text"`并打印WARNING，
  导致不传该参数的用户在正常使用中看到误导性警告。
- Fix: 将默认值从`None`改为`ExportType.Text`，同步更新schema中的签名字符串。
- Reviewability: high -- 对照文档即可发现签名不一致
- Review rule: 接口参数默认值必须与文档/schema声明一致，不能用None+fallback替代

### D-108: persistent_reduction模式下mask集合包含了不必要的reduction轴

- Hashes: 41cd21e3b
- Root cause: codegen对persistent_reduction语义下mask需求理解不完整
- Files: torch_npu/_inductor/codegen/triton.py
- Defect: persistent_reduction把reduction轴数据一次性全部加载到SRAM，不需要loop和mask。
  但`masks`集合构建时无条件包含所有`sorted_axis`（含`r*`前缀的reduction维度），
  导致codegen生成多余的reduction axis mask，在单reduction轴场景下产生DSL错误和精度偏差。
- Fix: 在`persistent_reduction`且`numof_reduction_axis() == 1`时，从mask集合中
  过滤掉`node.name[0] == "r"`的维度。
- Reviewability: low -- 需要理解persistent reduction在Triton tile语义中的含义
- Review rule: mask/index生成逻辑修改时必须覆盖persistent/split/loop全场景回归

### D-109: Scan IR节点被误当作Reduction处理导致编译崩溃

- Hashes: fcd98896b
- Root cause: IR类型混淆 -- `inside_reduction`标志对Scan和Reduction不加区分
- Files: torch_npu/_inductor/codegen/scheduling.py, torch_npu/_inductor/codegen/triton.py,
  test/_inductor/test_scan.py（新增）
- Defect: `inside_reduction=True`是scan和reduction共用的标志。`ReductionAnalysis`初始化时
  强制查找`ir.Reduction`节点，对`ir.Scan`节点（cumsum/cumprod lowering产物）找不到就抛
  RuntimeError。`decide_codegen_dims_in_kernel()`和`dense_size_list()`两处无条件触发
  `ReductionAnalysis`构造。另外NPU dense tile layout中scan轴位置不固定（`[R,X...]`或
  `[X...,R]`），不能像upstream那样写死为最后一维。
- Fix: 四项：1)scan kernel关闭cooperative reduction；2)`decide_codegen_dims_in_kernel()`
  先用`find_reduction_node()`判空再构建分析器；3)`dense_size_list()`同理加判空保护；
  4)新增`NPUIndexTritonKernel.scan()`，动态推断scan轴位置。
- Reviewability: low -- 根因是inside_reduction的语义双重性，需要深度理解Inductor IR
- Review rule: 新增IR类型时，审查所有以inside_reduction为条件的代码路径

### D-110: Python 3.9不支持PEP 604联合类型语法

- Hashes: e1e87269d
- Root cause: 使用了Python 3.10+语法（`X | Y`联合类型标注）
- Files: torch_npu/distributed/fsdp/_add_fsdp_patch.py
- Defect: 类型标注`size: Sequence[int | torch.SymInt]`使用PEP 604语法，该语法在
  Python 3.10才合法。Python 3.9运行时直接抛`TypeError`。
- Fix: 改为`Sequence[Union[int, torch.SymInt]]`，回退到`typing.Union`兼容写法。
- Reviewability: high -- 静态分析工具（pyupgrade --py39-plus）可自动检出
- Review rule: 项目最低支持Python版本为3.9时，禁止使用PEP 604(`X | Y`)类型标注

### D-111: 引入了目标CANN版本不存在的ACL设备属性API

- Hashes: 4aa14f7c7 [+5 cherry-picks: 0e7887883, 09d2c55fd, 4f778c95e, 924517bb9, a0b0e61bd]
- Root cause: 新增API依赖的CANN版本尚未发布/不可用
- Files: third_party/acl/inc/acl/acl_rt.h, torch_npu/csrc/npu/Module.cpp
- Defect: 引入了`aclrtGetDeviceInfo`函数和大量`aclrtDevAttr`枚举值（CUBE_CORE_NUM,
  WARP_SIZE, TOTAL_GLOBAL_MEM_SIZE, L2_CACHE_SIZE等），并在Module.cpp中添加了
  CANN>=8.5.0的版本分支走新API路径。但目标CANN版本不包含这些API声明，导致编译/链接失败。
- Fix: 从acl_rt.h中删除全部新增枚举值和函数声明，Module.cpp中回退到统一使用
  `aclrtGetMemInfo(ACL_HBM_MEM, ...)`，移除版本分支和op_plugin依赖。
  跨6个分支（v2.6.0, v2.7.1, master, v2.8.0, v2.9.0, v2.10.0）同步回滚。
- Reviewability: medium -- API存在性应在CI中验证，但人工review时需要确认目标CANN版本支持
- Review rule: 引入新CANN API前须确认目标最低CANN版本已包含该API声明

### D-112: NPU不兼容的deterministic empty tensor填充逻辑

- Hashes: a86817e1f
- Root cause: NPU后端tensor创建路径与upstream不同，deterministic fill实现不兼容
- Files: torch_npu/csrc/aten/common/TensorFactories.cpp, ResizeNpu.cpp, disabled test lists
- Defect: 尝试对齐upstream的"确定性模式下empty/resize填充NaN/MAX_INT"行为，调用了
  `at::native::fill_empty_deterministic_`和`fill_resize_deterministic_`。但NPU的empty
  实现路径与CPU/CUDA不同，这些fill操作在NPU tensor上行为异常，导致全dtype测试失败。
- Fix: 从`empty`, `empty_like`, `empty_strided`, `empty_out`, `resize_`中全部删除
  deterministic填充逻辑和`should_fill_empty_deterministic()`辅助函数，相关测试加入
  disabled列表。选择暂不支持该upstream特性。
- Reviewability: medium -- 移植upstream特性时需要验证NPU后端是否支持底层操作
- Review rule: 对齐upstream新特性前，确认NPU实现路径是否覆盖所需的底层API

### D-113: dynamo trace规则缺失NPU函数注册

- Hashes: 1fbd67778
- Root cause: NPU特有函数未注册到dynamo的TorchInGraphFunction白名单
- Files: torch_npu/dynamo/trace_rule.py, test/_inductor/test_add_triton_wrap.py
- Defect: `torch.npu._get_current_allocator`、`torch.npu.is_bf16_supported`、
  `torch_npu._C._npu_resetAccumulatedMemoryStats`三个函数未注册到dynamo trace规则中，
  导致dynamo遇到这些调用时graph break或fallback到eager模式。
  测试文件中也缺少`import torch_npu._inductor`。
- Fix: 将前两者加入`torch_non_c_binding_in_graph_functions_npu`，
  将`_npu_resetAccumulatedMemoryStats`加入`torch_c_binding_in_graph_functions_npu`。
- Reviewability: high -- 新增NPU API时应同步更新trace规则注册表
- Review rule: 新增NPU公开API后须检查是否需要注册到dynamo trace规则（c_binding或non_c_binding）

### D-114: fake后端NPU支持patch范围过宽

- Hashes: 0e4718038
- Root cause: 条件判断过于宽泛导致非目标后端被错误patch
- Files: torch_npu/__init__.py
- Defect: `_patch_backend_register_for_npu()`原逻辑为"凡是支持cuda的backend都自动追加
  npu支持"（`'cuda' in devices`），这个范围过宽，会把nccl等非fake的真实backend也patch
  进来，引起兼容性问题。
- Fix: 将条件从`'cuda' in devices`收窄为`backend_name == 'fake'`，只针对fake backend
  做patch。初始化时扫描`Backend.backend_capability`的逻辑也同步收窄。
- Reviewability: high -- 条件过宽的patch是明显的代码异味
- Review rule: monkey-patch第三方组件时，匹配条件应尽可能精确，避免波及无关模块

### D-115: is_pinned接口空指针未校验

- Hashes: 29be895a8
- Root cause: C++函数入口缺少空指针守卫
- Files: torch_npu/csrc/core/npu/CachingHostAllocator.cpp
- Defect: `CachingHostAllocator_isPinned(void *ptr)`未检查ptr是否为nullptr，
  空指针直接传入后续ACL查询函数（`aclrtPointerGetAttributes`等），导致段错误。
  复现场景：`test_generate_simple_inputs_npu`。
- Fix: 在函数入口添加`if (ptr == nullptr) { return false; }`提前返回。
- Reviewability: high -- 经典的空指针守卫缺失，静态分析可检出
- Review rule: 接受外部指针的C++ API入口必须做nullptr校验

### D-116: allocator padding策略按SoC区分导致回归

- Hashes: 9c205684a
- Root cause: 条件分支引入SoC特化后部分场景回归
- Files: torch_npu/csrc/core/npu/NPUCachingAllocator.cpp, test/npu/test_allocator_envs.py
- Defect: 之前引入了`AddPadSize()`函数按SoC类型区分padding（Ascend910_95返回0，其他返回32），
  使910_95设备不再添加32字节对齐padding。但该策略在某些分配场景下导致内存对齐问题或显存碎片。
- Fix: 删除`AddPadSize()`条件分支，改为在`round_size`和uncached malloc两处硬编码
  `constexpr size_t kPadSize = 32`，所有SoC统一加32字节padding。
  测试也回退为统一的`(size + 32) // 512 + 1) * 512`计算。
- Reviewability: medium -- 需要理解不同SoC的内存对齐约束差异
- Review rule: allocator对齐策略变更需要全SoC+全分配路径的压力测试验证

### D-117: invoke_subgraph高阶op未注册NPU lowering

- Hashes: 3182ab047 [+1 cherry-pick: 9cebf9ea0]
- Root cause: lowering注册表遗漏新增的higher-order op
- Files: torch_npu/_inductor/lowering_op_list.py
- Defect: `torch.ops.higher_order.invoke_subgraph`未加入`GENERATE_LIST`，
  导致该op在NPU inductor编译时无法被lower，整个子图fallback到CPU执行。
- Fix: 在`GENERATE_LIST`中添加`torch.ops.higher_order.invoke_subgraph`。
- Reviewability: high -- 新增higher-order op时应检查NPU lowering注册表
- Review rule: upstream新增higher-order op后，NPU lowering_op_list.py需同步更新

### D-118: 测试用例缺少平台兼容性跳过条件

- Hashes: 17d2b4404
- Root cause: 测试用例未区分平台适用性
- Files: test/npu/test_multi_stream_lazy_reclaim.py
- Defect: `TestMultiStreamLazyReclaim`在x86环境下测试失败（lazy reclaim行为依赖ARM64
  特有的内存管理机制）。断言条件也过严（`assertLess`不允许相等）。
- Fix: 添加`IS_ARM64 = platform.machine() in ('arm64', 'aarch64')`检测和
  `@unittest.skipUnless(IS_ARM64, "Only working on ARM")`装饰器。
  `assertLess`改为`assertLessEqual`放宽边界条件。
- Reviewability: high -- 硬件相关测试应有平台跳过装饰
- Review rule: 依赖特定硬件行为的测试用例必须添加平台检测跳过条件

### D-119: indirect memory算子被错误归入固定fallback列表

- Hashes: bf722958d
- Root cause: fallback列表不区分间接内存模式的开关状态
- Files: torch_npu/_inductor/lowering_fallback_list.py
- Defect: embedding、gather、index、index_put、scatter等间接内存访问算子被硬编码在
  `FALLBACK_LIST`中，无论indirect memory模式是否开启都走CPU fallback。当用户启用
  indirect memory模式时，这些算子本应走NPU编译路径。
- Fix: 将间接内存算子从`FALLBACK_LIST`抽离到新的`INDIRECT_MEM_FALLBACK_LIST`，
  仅在`not inductor_indirect_memory_mode`时合并回主列表。
- Reviewability: high -- 列表分类逻辑直接，review时对照feature flag即可发现
- Review rule: fallback列表中与feature flag相关的算子应做条件化管理

### D-120: profiler场景下EventVariable缺少python_type方法

- Hashes: 652bc2ba4
- Root cause: NPU扩展的dynamo Variable未实现框架要求的接口方法
- Files: torch_npu/utils/_dynamo.py
- Defect: `fullgraph=False`模式下开启profiling时，dynamo trace遇到Event对象需要调用
  `EventVariable.python_type()`，但NPU的EventVariable未实现该方法，抛出
  `NotImplementedError: EventVariable() has no type`。
- Fix: 新增`patch_event_variable_python_type()`函数，当`EventVariable.__dict__`中
  不存在`python_type`时，monkey-patch上`lambda self: type(self.value)`。
- Reviewability: medium -- 需要运行profiling+dynamo组合场景才能触发
- Review rule: 实现dynamo Variable子类时须确保覆盖框架要求的全部接口方法

### D-121: tensor buffer释放被NPU覆写为空实现+stride断言误报

- Hashes: c83bac42c
- Root cause: 两个独立问题在同一commit修复：生命周期管理覆写过度+stride校验不兼容
- Files: torch_npu/_inductor/__init__.py, wrapper.py, config.py, ir.py
- Defect: 1) `NPUWrapperCodeGen.make_buffer_free`被覆写为返回空字符串，导致inductor
  生成的代码中buffer永远不释放，tensor生命周期过长，显存占用持续增长。
  2) `_to_copy.default`和`reshape.default`等算子在NPU上的output stride与eager模式
  不完全一致（NPU kernel可能选择不同的内存布局），aoti场景下的stride断言误报。
- Fix: 1) 删除`make_buffer_free`的空覆写，恢复框架默认的buffer释放逻辑。
  2) 新增`skip_specific_stride_asserts`配置项和`patch_extern_kernel_codegen_size_asserts()`
  函数，对已知的NPU stride不一致算子跳过断言。
- Reviewability: medium -- 空实现覆写在review时容易识别，stride问题需要运行时验证
- Review rule: 覆写框架关键路径方法（尤其是资源释放）时须说明理由，禁止无注释的空实现

### D-122: aclCreateTensor创建的host侧描述符未释放

- Hashes: c4f4b33a9
- Root cause: ACL资源创建后缺少对应的释放调用
- Files: torch_npu/csrc/aten/common/FormatCastKernelNpu.cpp
- Defect: `MaybeUseAclnnNpuFormatCast`中调用`ConvertType(src)`创建ACL tensor描述符，
  但使用完后未调用对应的Release函数，导致每次format cast操作都泄漏一个host侧ACL tensor
  描述符。此外`new[]`分配的`dstStorageShape`在delete后未置nullptr。
- Fix: 将`ConvertType`返回值存入局部变量`acl_src`，在`GetFormat`调用后立即`Release(acl_src)`；
  `delete[] dstStorageShape`后置`nullptr`。
- Reviewability: high -- 经典的资源泄漏模式，ACL API调用后必须检查释放配对
- Review rule: aclCreateXxx/ConvertXxx的返回值必须在使用后调用对应的aclDestroyXxx/Release

### D-123: copy_操作对negative stride视图的处理逻辑错误

- Hashes: d0a18e3e8
- Root cause: 条件判断只检查源tensor的neg状态，遗漏目标tensor
- Files: torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp
- Defect: `copy_`实现中，当源tensor有negative stride（`is_neg()`为true）时执行`self.neg_()`
  来补偿。但原条件`if (src.is_neg())`只考虑了src是negative view的情况，未考虑self本身
  也可能是negative view。当self和src都是negative view时，neg_操作会双重取反导致数据错误。
- Fix: 条件改为`if (self.is_neg() != src.is_neg())`，仅在两者neg状态不一致时才执行
  `self.neg_()`补偿。
- Reviewability: high -- 对称性检查（self和src的状态都需要考虑）是review基本项
- Review rule: tensor操作中涉及view属性（neg/conj）时，必须同时检查源和目标tensor的状态

### D-124: 不支持的inductor测试未添加skip导致CI阻塞

- Hashes: 04c01018b
- Root cause: 新增功能的测试在NPU上尚不支持但未添加skip条件
- Files: test/_inductor/test_masked_subblock.py (或类似)
- Defect: inductor测试用例在NPU环境上无法通过（NPU尚未支持对应功能），但未添加平台检查
  或skip装饰器，导致CI流水线被阻塞。
- Fix: 添加平台检查跳过条件。
- Reviewability: high -- CI失败后直接定位
- Review rule: 新增inductor功能测试时，需确认NPU是否支持并添加相应skip条件

### D-125: 嵌套masked_subblock中codegen作用域管理错误

- Hashes: a1c188aaa
- Root cause: Triton codegen中subblock mask变量的作用域在嵌套时未正确隔离
- Files: torch_npu/_inductor/codegen/triton.py, test/_inductor/test_masked_subblock.py（新增）
- Defect: 当masked_subblock4嵌套在masked_subblock3中时，外层mask变量携带了多维度
  indexing（如`x0, r0`），但内层subblock的load/store只需要部分维度的mask。
  `current_subblock`使用单一字符串追踪当前块名，嵌套时内层覆盖外层状态，退出内层后
  外层mask变量引用已损坏，导致生成的`tl.where`参数shape不匹配、编译失败。
- Fix: 将`current_subblock`替换为基于axis集合的追踪机制，嵌套时通过`|=`合并父子块的
  axis集合，退出子块时恢复父块状态。`save_variable_mask`条件改为检查当前subblock_axis。
- Reviewability: low -- 需要理解Triton codegen的subblock嵌套语义和mask传播机制
- Review rule: codegen中涉及作用域嵌套的状态变量，必须用栈式结构管理，禁止单变量覆写

### D-126: AOTI运行时接口在NPU wrapper中缺失

- Hashes: d69f2f060
- Root cause: upstream新增AOTI接口后NPU侧未同步实现
- Files: torch_npu/csrc/aten/common/... (model_container.h 或类似)
- Defect: PyTorch upstream在AOTInductor运行时容器中新增了`constant_blob_size()`和
  相关常量管理接口，NPU的模型容器wrapper未提供对应方法，导致v2.10.0上AOTI编译产物
  无法正确加载常量数据。
- Fix: 在NPU侧`AOTModelContainer`中添加`constant_blob_size()`等缺失方法的转发实现。
- Reviewability: medium -- upstream接口变更可通过diff比对发现
- Review rule: PyTorch大版本升级后，检查AOTI/inductor接口是否有新增需要NPU侧适配

### D-127: gen_code工具中ACLNN扩展路径拼接错误

- Hashes: e904e77fb
- Root cause: 路径拼接缺少中间目录层级
- Files: generate_code相关脚本或setup工具
- Defect: `gen_backend_stubs.py`中拼接ACLNN扩展模块路径时，使用
  `os.path.join(base, 'utils/OpUtils.py')`缺少中间的`torch_npu`目录层级，
  导致import时找不到目标模块。
- Fix: 路径修正为`os.path.join(base, 'torch_npu/utils/OpUtils.py')`，补全目录层级。
- Reviewability: high -- 路径字符串错误，测试可直接捕获
- Review rule: 文件路径拼接使用os.path.join并通过os.path.exists验证


### D-128: inductor tiling配置列表为空时排序崩溃

- Hashes: 43600c94f
- Root cause: 空集合操作未防护
- Files: torch_npu/_inductor/codegen/tile_generator.py
- Defect: `TileGenerator`在SIMT_ONLY kernel的config过滤流程中，直接对`self.configs`做
  `sort` + 索引访问(`self.configs[0]`)，但未检查列表是否为空。当tiling生成阶段未产生
  任何有效配置时，`len(self.configs)==0`，排序后取`[0]`触发IndexError。
- Fix: 在排序前增加`len(self.configs) > 0`条件守卫。
- Reviewability: high -- 经典的空集合防护缺失，静态分析或简单code review即可发现
- Review rule: 对集合做排序+索引访问前必须检查非空

### D-129: upstream flex_attention函数删除后NPU侧import失败

- Hashes: 0dfce7921
- Root cause: upstream函数删除/重构后NPU侧import未同步
- Files: torch_npu/_inductor/kernel/flex_attention.py
- Defect: NPU的flex_attention模块从`torch._inductor.kernel.flex_attention`中import
  `_use_flex_decoding`函数，但upstream PyTorch删除或重构了该函数，导致ImportError。
  NPU侧flex decoding的判断逻辑(短query length + 静态batch/heads)需要本地实现。
- Fix: 从import列表中移除`_use_flex_decoding`，在NPU侧本地重新实现该函数，包含
  `FORCE_USE_FLEX_ATTENTION`选项检查、query长度<128判断、batch/heads静态性检查。
- Reviewability: medium -- upstream API变更需要CI对比检测
- Review rule: 从upstream import的函数应在CI中做import smoke test，
  upstream版本升级时自动检查import兼容性

### D-130: slice_scatter workaround在proper fix后未及时移除

- Hashes: a6d23cde1
- Root cause: 临时workaround残留(与D-149的proper fix配对)
- Files: torch_npu/_inductor/lowering.py, torch_npu/_inductor/lowering_op_list.py
- Defect: 为规避嵌套mask subblock问题，曾为`aten.slice_scatter`注册了一个自定义
  lowering，通过检查`src`中是否包含`"ops.masked"`字符串来决定是否做`src.realize()`。
  这是一个hacky的字符串检查workaround，在D-149提供了proper fix后应被移除。
- Fix: 删除自定义`slice_scatter` lowering注册，从`LOWERING_OVERLOAD_OP`列表中移除
  `aten.slice_scatter`，让其走upstream默认路径。
- Reviewability: medium -- workaround通常缺乏"何时移除"的记录
- Review rule: 临时workaround必须附带TODO注释标明依赖的proper fix issue编号

### D-131: transfer_to_npu中jit.script异常处理策略错误

- Hashes: 88efa1f78
- Root cause: 异常处理粒度过粗(try/except全量吞掉jit.script失败)
- Files: torch_npu/contrib/transfer_to_npu.py, torch_npu/utils/_dynamo.py
- Defect: `_jit_script`包装函数用`try/except Exception`捕获`_real_jit_script`的所有
  异常并静默返回原函数，这意味着任何jit.script失败都被吞掉。当dynamo模式下需要明确
  控制是否使用jit.script时，这种全量catch策略导致行为不可预测。
- Fix: 引入`_dynamo.use_jit_script`标志位，用条件分支替代try/except:
  当`use_jit_script=True`时调用`_real_jit_script`，否则直接返回原函数。
  将决策从"运行时异常驱动"改为"配置驱动"。
- Reviewability: medium -- 异常处理策略需要理解上下文意图
- Review rule: 用条件守卫替代broad try/except; 异常处理不应作为正常控制流

### D-132: Triton codegen mask过滤逻辑错误导致无关mask残留

- Hashes: c6d865667
- Root cause: mask变量过滤策略反转(匹配所有axis后排除 vs 仅匹配masked axis后保留)
- Files: torch_npu/_inductor/codegen/triton.py
- Defect: `NPUIndexTritonKernel`的mask过滤逻辑维护了`all_axis_name`(所有axis)和
  `masked_axis_name`(需要mask的axis)两个列表。过滤时先从`all_axis_name`中匹配，
  匹配到后再检查是否在`masked_axis_name`中。这导致非axis的mask变量(如tmp mask)
  在匹配失败后执行`continue`被跳过而非保留，产生xmask/ymask等无关mask。
  同时`_load_other`的条件分支引入了不必要的复杂性。
- Fix: 简化为仅遍历`masked_axis_name`做正向匹配: 匹配到的标记`valid_mask_var=True`，
  未匹配到的从`mask_vars`中移除。删除`all_axis_name`列表和`_load_other`条件分支，
  load的other统一使用`0.0`。
- Reviewability: low -- 需理解Triton codegen的axis/mask语义和A5硬件上的行为差异
- Review rule: mask过滤逻辑应采用白名单(保留已知需要的)而非黑名单(排除已知不需要的)

### D-133: aclgraph测试中softmax_lse tensor shape硬编码

- Hashes: 57ad44dd5
- Root cause: 测试tensor shape硬编码而非从实际输出推导
- Files: test/npu/test_aclgraph_update.py
- Defect: `TestIFAAclgraphUpdate`测试中`softmax_lse`用`torch.empty(1, ...)`创建
  固定shape的tensor，但实际IFA(Infer Fused Attention)算子输出的softmax_lse维度
  取决于query的head数和seq长度。当算子实现变更后shape不匹配导致测试失败。
- Fix: 改为`torch.empty_like(res_src[1], dtype=torch.float16, device="npu")`，
  从实际输出推导shape。
- Reviewability: high -- 测试中硬编码magic number是code review红旗
- Review rule: 测试中output tensor shape应从实际op输出推导，不硬编码

### D-134: upstream TritonCSEVariable新增shape参数后NPU patch未同步

- Hashes: 7650811e2
- Root cause: upstream类签名扩展后NPU侧monkey-patch未同步
- Files: torch_npu/_inductor/codegen/__init__.py, torch_npu/_inductor/codegen/triton.py
- Defect: PyTorch upstream在`TritonCSEVariable.__init__`中新增了`shape: BlockShapeType`
  参数，NPU侧未import该类也未提供对应的patch。当upstream代码路径调用带shape参数的
  构造函数时，NPU侧monkey-patch的`__init__`不接受该参数导致TypeError。
- Fix: 新增`patch_TritonCSEVariable__init__`函数，签名包含`shape`参数并传递给
  `super().__init__`，同时保留NPU侧的`mask_vars`属性初始化。在`__init__.py`中
  import并patch到`TritonCSEVariable.__init__`。
- Reviewability: medium -- upstream签名变更需版本升级时的系统性检查
- Review rule: monkey-patch upstream类方法时，patch函数签名必须与upstream最新签名一致

### D-135: MLIR lowering中pow_native未注册到aten映射表

- Hashes: 9a42c115b
- Root cause: 装饰器遗漏(注册到aten映射表的装饰器)
- Files: torch_npu/_inductor/ascend_npu_ir/.../lowering.py
- Defect: `pow_native`函数仅有`@make_pointwise`装饰器，缺少`@register_to_aten(aten_fn=aten.pow)`
  装饰器。导致MLIR inductor的lowering阶段找不到`aten.pow`对应的NPU实现函数，
  `fn_to_aten_fn`映射表中无该条目，lowering失败。
- Fix: 在`@make_pointwise`前添加`@register_to_aten(aten_fn=aten.pow)`装饰器。
- Reviewability: high -- 装饰器遗漏是典型的checklist可检查项
- Review rule: 新增pointwise op实现时，checklist必须包含"是否注册到fn_to_aten_fn"

### D-136: as_strided校验顺序导致负stride报错与upstream不一致

- Hashes: 6d7d33c5c
- Root cause: 校验语句执行顺序与upstream不一致
- Files: torch_npu/csrc/aten/common/ResizeNpu.h
- Defect: NPU的`setStrided`函数中，负stride检查放在`checkInBoundsForStorage`之后、
  `set_sizes_and_strides`之前。但upstream(CUDA/CPU)将负stride检查放在最前面。
  当stride为负时，NPU先触发`checkInBoundsForStorage`(可能报不同的错误信息)，
  而upstream直接报"Negative strides are not supported"。报错信息不一致导致
  用户困惑，且错误码从`ErrCode::NOT_SUPPORT`改为`ErrCode::PARAM`更准确。
- Fix: 将负stride的for循环检查移到函数开头(size/stride长度检查之后)，与upstream对齐。
- Reviewability: high -- 校验顺序与upstream对比即可发现
- Review rule: NPU侧tensor属性校验逻辑应与upstream保持相同顺序和语义

### D-137: OpInfo dtype测试基础设施未覆盖NPU实际支持的dtype

- Hashes: 5bc5d6f01
- Root cause: 测试框架dtype信息与硬件实际能力不同步
- Files: torch_npu/testing/__init__.py, torch_npu/testing/npu_opinfo_dtypes.py (新增)
- Defect: `supported_dtypes()`直接返回`self.dtypes`(upstream默认列表)，未考虑NPU
  对特定op可能支持额外的dtype。`__rmod__`在NPU上实际支持`int32/int64`前向计算，
  但OpInfo中未登记，导致`test_dtypes___rmod___npu`测试失败。
- Fix: 引入配置驱动的dtype扩展机制: 新增`npu_opinfo_dtypes.py`配置文件，按op名
  维护额外dtype; `supported_dtypes()`在device为npu时读取配置并合并。新增
  `_merge_dtypes`辅助函数处理set/tuple/list三种容器类型的合并。
- Reviewability: medium -- 需要了解NPU算子的实际dtype支持范围
- Review rule: 新增NPU算子适配后，同步更新OpInfo dtype配置

### D-138: torch.randint测试对uint8使用了负范围

- Hashes: 93f4c01d0
- Root cause: 测试参数范围未考虑unsigned类型边界
- Files: test/npu/test_dlpack.py
- Defect: `test_dlpack`中对所有整型(包括`torch.uint8`)统一使用`torch.randint(-10, 10, ...)`，
  但uint8的有效范围是[0, 255]，传入负值-10作为low参数会在`tensor.random_`的
  新增校验逻辑中触发断言失败(值小于dtype最小值)。
- Fix: 将uint8从统一的整型分支中独立出来，使用`torch.randint(0, 10, ...)`。
- Reviewability: high -- 类型范围是基础知识，review时应关注unsigned类型的负值使用
- Review rule: 随机数生成的范围参数必须在目标dtype的有效范围内

### D-139: setup.py缺少MLIR C++源文件的打包规则

- Hashes: fbef11ea8
- Root cause: 新模块文件打包规则遗漏(setup.py未包含anir C++源文件)
- Files: setup.py, test/npu/test_public_bindings.py, third_party/acl/inc/...
- Defect: `ascend_npu_ir`模块包含C++源文件(`.cpp`, `.h`)和`cpp_common`目录，但
  `setup.py`的`get_src_py_and_dst()`函数未将这些文件纳入打包列表。导致安装后
  MLIR inductor的JIT编译找不到必要的C++源文件。同时回退了一个错误删除的
  `prof_api.h`头文件。
- Fix: 在`get_src_py_and_dst`中新增`anir_files` glob规则，覆盖`_C/*.cpp`、
  `_C/include/*.h`、`cpp_common/*`三类文件，生成对应的dst路径并加入ret列表。
  恢复被误删的`prof_api.h`。更新`test_public_bindings`的模块列表。
- Reviewability: medium -- 新增C++模块时容易遗漏打包规则
- Review rule: 新增含C++源文件的子模块时，checklist必须包含setup.py打包规则更新

### D-140/D-145: MLIR模块import中模块级副作用导致功能异常 (fix-revert cycle #5)

- Hashes: beac5137b (D-145, fix), 79168bcce (D-140, revert)
- Root cause: 模块级初始化副作用与lazy init语义冲突
- Files: torch_npu/_inductor/_lazy_init.py
- Defect: D-145(beac5137b)试图简化`_lazy_init.py`，移除了模块级的
  `build_ascend_npu_ir_ext()`调用和`set_torch_npu_library_path()`调用，
  认为这些应该由downstream的`import_npu_inductor_plugin`内部处理。但实际上
  MLIR功能依赖这些模块级初始化(编译扩展模块+设置library path)在plugin import
  之前完成。移除后MLIR模块无法正确加载。
- Fix/Revert: D-140(79168bcce)在次日revert了D-145，恢复模块级初始化调用。
  说明这些"看起来可以lazy化"的初始化实际有严格的时序依赖。
- Reviewability: medium -- 需要理解MLIR模块的初始化时序依赖
- Review rule: 移除模块级初始化代码前，必须确认所有downstream consumer的
  初始化时序要求; lazy init重构需要端到端测试验证

### D-141: zero copy场景SetDevice逻辑错误

- Hashes: c627ad792
- Root cause: 条件分支逻辑错误(SetDevice被错误地放在条件内)
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: zero copy初始化路径中，`c10_npu::SetDevice(local_rank)`被放在
  `if (device_id != local_rank)`条件块内，且在赋值`device_id = local_rank`之后。
  这意味着:(1) 当device_id已等于local_rank时不会调用SetDevice，但此时设备可能
  尚未被init_process_group前的用户代码正确设置; (2) SetDevice用的是已被覆写的
  local_rank而非实际需要设置的device。
- Fix: 将`SetDevice`移到条件块之外，改为无条件调用`c10_npu::SetDevice(device_id)`，
  确保zero copy场景下device始终被正确设置。
- Reviewability: high -- 条件分支内的关键操作应引起code review注意
- Review rule: 设备初始化类操作(SetDevice)应尽量无条件执行，避免被条件分支跳过

### D-142: mem_get_info对_get_device_index调用缺少optional参数

- Hashes: 9ca5ac6b5
- Root cause: API调用参数与upstream语义不一致
- Files: torch_npu/npu/__init__.py
- Defect: `mem_get_info`调用`_get_device_index(device)`时未传`optional=True`。
  PyTorch的`_get_device_index`在`optional=False`(默认)时要求device参数不能为
  某些特定值(如-1)。NPU侧的`mem_get_info`已有自己的device校验逻辑(检查范围)，
  但在到达自定义校验前就在`_get_device_index`内部失败了。upstream CUDA的
  `mem_get_info`传的是`optional=True`。
- Fix: 添加`optional=True`参数，与upstream CUDA行为对齐。
- Reviewability: high -- 与upstream API调用对比即可发现
- Review rule: 调用PyTorch内部工具函数时，参数语义应与upstream同类调用保持一致

### D-143: compile_fx中dict字面量key写成变量名而非字符串

- Hashes: 315342892
- Root cause: Python dict字面量语法错误(变量名作key而非字符串)
- Files: torch_npu/npu/npugraph_ex/__init__.py
- Defect: `compile_fx`中调用`_process_kwargs_options(compiler_config, {options: options})`，
  其中`{options: options}`的key是变量`options`(即传入的dict对象本身)而非字符串
  `"options"`。当`options`为None时，key为None; 当options为dict时，dict不可hash
  导致TypeError。即使不报错，downstream按字符串key查找也找不到。
- Fix: 修正为`{"options": {} if options is None else options}`，同时处理None情况。
- Reviewability: high -- `{var: var}` vs `{"key": var}`是极易在review中发现的typo
- Review rule: dict字面量的key应该是字符串常量时，确认没有误用变量引用

### D-144: HCCL算子执行时线程变量中的核数配置未更新 (fix-revert cycle #4)

- Hashes: 92a86938a (fix on v2.6.0) [revert on master: e32c44c98]
- Root cause: 线程变量更新路径遗漏(仅计算算子触发，HCCL算子不触发)
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: 单算子控核(core control)通过线程变量传递核数配置。计算算子执行前会刷新
  线程变量，但HCCL算子的执行路径不经过该刷新逻辑。当HCCL算子前没有计算算子时，
  线程变量中的核数是上次的缓存值，导致控核API设置的核数不生效。
- Fix: 在`collective`和`collectiveCoalesced`中新增核数更新逻辑: 检查当前stream的
  AIC/AIV核数是否与缓存值不同，若不同则调用`UseStreamResInCurrentThread`更新
  线程变量，并设置`SetStreamResLimit`到hccl stream。
  但该fix在master分支被revert(e32c44c98)，可能引入了其他副作用。
- Reviewability: low -- 需要理解线程变量、stream资源限制、核控制的交互机制
- Review rule: 线程级状态(TLS)的更新路径必须覆盖所有消费方，不仅是最常见的路径

### D-146: MLIR启用选项检查未防护options类型和None值

- Hashes: 00f6cd17d
- Root cause: 可选参数None/非dict分支遗漏
- Files: torch_npu/utils/_dynamo.py
- Defect: `new_init`中直接用`options["npu_backend"]`访问options参数，但options
  可能是None(用户未传)或非dict类型。直接下标访问None触发TypeError，
  非dict类型触发KeyError。
- Fix: 改为`isinstance(options, dict) and options.get("npu_backend") == "mlir"`，
  先检查类型再用`.get()`安全访问。
- Reviewability: high -- 可选参数的None防护是基础review要点
- Review rule: 对可选dict参数，访问前必须检查isinstance和使用.get()

### D-147: monkey-patch函数缺少幂等性守卫导致重复patch

- Hashes: dfc2f5c99
- Root cause: patch函数无幂等性保护(每次调用都重复执行)
- Files: torch_npu/_inductor/shape_handling.py
- Defect: `patch_shape_handling()`每次调用都会执行`patch_dynamo_context()`，
  在某些代码路径下被多次调用。重复patch可能导致:(1) 包装函数嵌套多层;
  (2) 全局状态被意外重置; (3) 性能开销。
- Fix: 使用函数属性`patch_shape_handling._is_patched`作为幂等性标志，
  第二次调用时直接return。
- Reviewability: high -- monkey-patch函数缺少幂等性守卫是已知模式
- Review rule: 所有monkey-patch函数必须有幂等性守卫(函数属性或模块级flag)

### D-148: aclrtMallocHostWithCfg的SoC版本范围检查缺少上界

- Hashes: cd5f447d5
- Root cause: SoC版本范围守卫不完整
- Files: torch_npu/csrc/core/npu/interface/AclInterface.cpp
- Defect: `AclrtMallocHostWithCfgExist()`的SoC版本检查只有下界(`>= Ascend910B1`)没有上界，
  Ascend910_95不支持`aclrtMallocHostWithCfg`的VA一致性实现，但被下界条件错误放行。
- Fix: 增加上界排除(`&& < Ascend910_95`)，使范围检查完整。
- Reviewability: high -- 范围检查只写下界是常见遗漏，新SoC引入时应系统性审查所有版本守卫
- Review rule: SoC版本范围检查必须同时声明上下界，或显式注释"无上界原因"

### D-149: masked_subblock的mask过滤逻辑缺失

- Hashes: 1043e2655
- Root cause: codegen新feature(masked_subblock)缺少配套的mask分析pass
- Files: torch_npu/_inductor/codegen/__init__.py, ir.py, triton.py
- Defect: inductor codegen对`masked_subblock`内的load/store操作没有正确识别哪些axis mask应保留，
  mask被错误过滤掉，生成不正确的kernel代码。缺少从subblock index到mask变量的关联分析。
- Fix: 新增`generate_masked_indexing`分析pass，在`filter_masks`中根据subblock_axis精确判断
  mask保留条件；`triton.py`中跟踪`current_subblock`状态。
- Reviewability: low -- 需要理解mask过滤与subblock index的语义关系，纯静态review难发现
- Review rule: 新增codegen节点类型时，必须同步更新mask过滤逻辑并添加覆盖测试

### D-150: npu_functions覆盖lowering时KeyError

- Hashes: 5f5f6836c
- Root cause: 字典key存在性未检查
- Files: torch_npu/_inductor/ascend_npu_ir/.../inductor_patch/__init__.py
- Defect: 用`npu_functions[name]`覆盖标准lowering函数时，直接取值未检查key是否存在。
  npu_lowering模块不一定实现标准lowering中的所有函数，不存在的key引发KeyError。
- Fix: 增加`if name in npu_functions:`守卫。
- Reviewability: high -- 字典取值缺少存在性检查是基础防御性编程问题
- Review rule: 跨模块字典查找必须用`in`检查或`.get()`，不能假设key一定存在

### D-151: Revert HCCL线程核数同步方案(fix-revert cycle #4 revert侧)

- Hashes: e32c44c98
- Root cause: 修复方案引入副作用(fix-revert cycle #4的revert)
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp, ProcessGroupHCCL.hpp
- Defect: D-144(92a86938a)在v2.6.0中将用户stream的AIC/AIV核数限制传递给HCCL stream，
  但该方案在master分支引入新问题（HCCL算子在非预期核数配置下行为异常）。
  删除`cached_aic_num`/`cached_aiv_num`缓存变量和所有`UseStreamResInCurrentThread`调用。
- Fix: 完整Revert，恢复HCCL使用默认核数配置。
- Reviewability: medium -- 原fix逻辑正确但副作用需要跨stream场景测试才能暴露
- Review rule: 修改runtime资源配置(核数/内存)的传播路径时，需验证所有消费方的兼容性

### D-152: grouped matmul输出stride计算使用padding后的size

- Hashes: e8efb02ef
- Root cause: 过度泛化的多分支逻辑中stride计算错误
- Files: torch_npu/_inductor/kernel/mm_grouped.py
- Defect: `grouped_mm_args`对2D/3D矩阵组合做了复杂的size/stride分支计算，其中output stride
  使用了padding对齐的`size_padded`而非实际size，导致grouped matmul输出数据布局不正确。
- Fix: 删除过度泛化的多分支代码，统一简化为`out_stride = [out_size[-1], 1]`。
- Reviewability: medium -- stride计算错误需要对矩阵layout有深入理解，但代码过度复杂本身是review信号
- Review rule: 当代码分支复杂度超过实际需要时，优先简化逻辑而非补丁修复

### D-153: dump模式扩展参数无条件传递导致标准路径失败

- Hashes: 74ff0ccda
- Root cause: 调试工具的扩展参数污染正常代码路径
- Files: torch_npu/_inductor/lowering.py
- Defect: 精度工具(dump_fx_graph模式)给`Pointwise.create`扩展了`traced_graph`和`node_name`参数，
  但原代码无条件传递这两个参数，标准PyTorch的`Pointwise.create`不接受，导致非dump模式下调用失败。
- Fix: 改为`if npu_config.dump_fx_graph:`条件分支，仅dump模式传递扩展参数。
- Reviewability: high -- 调试参数无条件传递是明显的条件遗漏
- Review rule: 调试/诊断工具的扩展参数必须在feature flag守卫下传递，不能污染正常路径

### D-154: profiler双指针匹配算法缺少early-break导致O(n^2)性能退化

- Hashes: 75c77b902
- Root cause: 双指针算法缺少提前终止条件
- Files: torch_npu/profiler/analysis/prof_view/prof_db_parse/_fwk_api_db_parser.py
- Defect: dequeue与node_launch、enqueue与torch_op的时间戳配对算法中，当目标指针已越过当前时间窗口时
  缺少break条件，指针不更新继续无效扫描，造成O(n^2)性能退化。
- Fix: 在三处双指针匹配循环中增加"已越过目标时间窗口"时的`last_index`更新和break。
- Reviewability: medium -- 需要理解双指针算法的不变量，但O(n^2)退化在大规模数据上可通过性能测试暴露
- Review rule: 有序序列双指针匹配必须有"越过目标"时的提前终止条件

### D-155: PRE阶段graph pass过早执行eliminate_dead_code

- Hashes: 19b0a1d5a
- Root cause: graph pass生命周期阶段(PRE/POST)未隔离清理操作
- Files: torch_npu/_inductor/fx_passes/ascend_custom_passes/ascend_graph_pass.py
- Defect: cat_slice_cat_fold_pass、pad_slice_fold、dtype_optimal_pass等PRE类型pass在修改graph后
  立即调用`graph.lint()`+`graph.eliminate_dead_code()`，但此时graph处于中间状态，
  dead code消除会误删后续pass需要的节点，导致hllm模型编译失败。
- Fix: 新增`POST=True`参数，PRE pass跳过lint和dead code消除，推迟到POST阶段统一执行。
- Reviewability: medium -- 需要理解多pass pipeline的执行顺序和节点依赖关系
- Review rule: 多pass pipeline中，破坏性的graph清理操作(eliminate_dead_code, constant folding)
  应集中在pipeline末尾执行，不应在中间pass中各自调用

### D-156: load generator的mask过滤误删语义mask + other值硬编码0.0

- Hashes: 6dc00c5df
- Root cause: 两个独立bug: (1)mask分类逻辑不完整 (2)常量硬编码替代参数
- Files: torch_npu/_inductor/codegen/triton.py
- Defect: (1) `filter_masks()`把"不认识的mask"(如tmp padding guard)当作"无效mask"丢弃，
  但tmp mask不属于任何tiling轴，被误删导致越界访问。
  (2) masked load的other值硬编码为0.0，对`constant_pad_nd`等需要非零fill value的算子语义错误。
- Fix: (1) 先判断mask是否属于"轴mask"，不属于任何轴的保留为语义mask。
  (2) 检查`self._load_other`是否非None，使用实际fill value。
- Reviewability: medium -- (1)需要理解mask分类体系 (2)硬编码0.0是可review发现的
- Review rule: mask过滤逻辑应显式区分"已知的轴mask"和"未知mask"，默认保留未知类型

### D-157: low dims判定只看stride=1而未排斥stride>1

- Hashes: 6fd0dd77d
- Root cause: 集合分类缺少差集操作
- Files: torch_npu/_inductor/codegen/split_tiling.py, __init__.py, npu_triton_heuristics.py
- Defect: `construct_low_dim()`遍历indexing时凡stride=1就加入`low_dims`，但没有排除在其他indexing中
  stride>1的轴。一个轴在index0中stride=1但在index1中stride=128，不应归为low dim，
  导致tiling策略选择错误，间接内存访问出错。
- Fix: 维护`low_dims`和`high_dims`两个集合，最终`low_dims = low_dims - high_dims`，
  任一indexing中stride>1的轴都排除。同时修复硬编码SoC版本号为常量。
- Reviewability: medium -- 需理解multi-indexing场景下轴属性的交叉验证
- Review rule: 涉及多数据源的分类判定，必须考虑同一元素在不同数据源中属性矛盾的情况

### D-158: import阶段隐式触发AsyncCompile线程池创建

- Hashes: f685c01af [+1 cherry-pick: 85fa92548]
- Root cause: 模块import副作用触发重量级初始化
- Files: torch_npu/__init__.py, torch_npu/_inductor/__init__.py, test/_inductor/test_argmax.py
- Defect: `import torch_npu`时PyTorch的`AsyncCompile.warm_pool()`被隐式触发，在模块加载阶段
  就创建多线程。用户可能根本不用inductor，浪费资源且可能引发循环依赖。
- Fix: 设置`TORCH_WARM_POOL=0`抑制import阶段线程池创建，延迟到inductor模块真正import时warm pool。
- Reviewability: medium -- import副作用链需要trace整个import序列才能发现
- Review rule: 模块`__init__.py`中禁止触发重量级资源分配(线程/进程/GPU内存)，应延迟到首次使用

### D-159: view tensor误入SIMT模板codegen路径

- Hashes: 9ae029a17
- Root cause: codegen路径选择缺少IR节点类型检查
- Files: torch_npu/_inductor/codegen/ir.py, lowering.py, 3个测试文件
- Defect: SIMT模板路径不支持经过view操作(unsqueeze/expand/permute/slice)的tensor作为输入，
  但`should_use_template()`未检查输入是否为`BaseView`，view后的tensor进入模板路径后codegen异常。
- Fix: 在embedding/gather/index_put/scatter等lowering函数中增加`isinstance(x.data, ir.BaseView)`检查，
  view类型tensor走通用路径。新增`define_npu_kernel_type()`处理indirect load + sum reduction模式。
- Reviewability: medium -- 需要了解BaseView类型与SIMT模板的不兼容性
- Review rule: codegen路径选择器必须对所有IR节点类型(包括View子类)有明确的处理策略

### D-160: Revert AsyncCompile warm_pool注册时机不当

- Hashes: f17d4cc79
- Root cause: warm_pool调用放在inductor注册阶段过早(与D-158相关)
- Files: torch_npu/__init__.py, torch_npu/utils/_dynamo.py
- Defect: 此前一个commit在`register_inductor_npu()`中加入`AsyncCompile.warm_pool()`预热调用，
  但注册函数在inductor模块加载阶段执行，时机过早且引发副作用。
  与D-158是对同一"线程池初始化时机"问题的不同分支的修复策略冲突。
- Fix: Revert warm_pool调用和`TORCH_WARM_POOL=0`设置，让线程池回归默认初始化时间点。
- Reviewability: medium -- 需要理解多个分支上对同一问题的并行修复之间的交互
- Review rule: 同一问题在多分支并行修复时，需要确认各分支策略一致，避免互相revert

### D-161: dropout在NPU上缺少专用decomposition

- Hashes: 367bfe1ba [+5 cherry-picks: b09773813, 665c0117c, c472b5b58, 578bbd2f1, 9ea3f12d4]
- Root cause: NPU原生算子未注册decomposition导致走默认随机数路径
- Files: torch_npu/_inductor/config.py, decomposition.py, test/distributed/_tensor/test_dtensor_custom_ops.py
- Defect: Inductor默认的dropout decomposition生成随机数逻辑，在NPU上执行结果不正确或不支持。
  缺少将dropout路由到NPU原生实现(`npu._npu_dropout`)的decomposition注册。
- Fix: 设置`config.fallback_random = True`，注册`aten.native_dropout`和`aten.native_dropout_backward`
  的NPU decomposition，训练模式分解到`npu._npu_dropout`/`npu.npu_dropout_backward`。
- Reviewability: medium -- 需要了解NPU随机数语义与CUDA的差异
- Review rule: NPU后端引入新算子类型时，检查upstream默认decomposition是否依赖CUDA特有行为

### D-162: catlass bias tiling三重问题(L0 tile不同步+bias layout误判+gate过度限制)

- Hashes: 33182e78d
- Root cause: 三个独立问题叠加
- Files: catlass_library/gemm_autotune.py, catlass_scheduling.py, catlass_utils.py,
  gemm_template.py, config.py
- Defect: (1) L1 tile重算后没有同步更新L0 tile的m/n维度，导致tiling不一致。
  (2) bias tensor的(n,)shape经broadcast后stride=(0,1)，layout判定未识别stride[0]==0的情况，
  误判为2D layout。(3) `catlass_evg_fusion_enable` gate检查阻止合法epilogue融合路径。
- Fix: (1) 补全L0 tile的m/n赋值。(2) 新增`_catlass_tensor_from_node_for_bias()`，
  识别stride[0]==0为VectorLayout。(3) 删除过度限制的gate条件和对应配置项。
- Reviewability: low -- 三个问题分散在不同文件且互相遮掩，单独很难触发完整症状
- Review rule: catlass变更需要包含bias/非bias两种epilogue路径的端到端测试

### D-163: fsdp patch遗漏.default dispatch变体

- Hashes: 30dc60baf
- Root cause: op dispatch入口覆盖不完整
- Files: torch_npu/distributed/fsdp/_add_fsdp_patch.py
- Defect: FSDP patch只monkey-patch了`torch.ops.fsdp.all_gather_copy_in`，但PyTorch内部某些路径
  通过`.default`分发变体调用该op。`.default`入口未被patch，走到NPU不支持的原始实现报错。
- Fix: 补充`torch.ops.fsdp.all_gather_copy_in.default = _patched_all_gather_copy_in`。
- Reviewability: high -- `.default`是PyTorch op dispatch的标准变体，patch时应一并覆盖
- Review rule: monkey-patch PyTorch op时，必须同时覆盖`.default`和其他已知dispatch变体

### D-164: index_select与cat融合时codegen shape不匹配

- Hashes: 7d69f9b48
- Root cause: 算子融合边界条件下shape计算遗漏
- Files: torch_npu/_inductor/codegen/scheduling.py, triton.py
- Defect: index_select与cat融合时，codegen生成的shape计算未考虑cat节点对维度的影响，
  导致tensor shape与实际数据不匹配，code cache命中错误kernel。
  同时fusion距离阈值35仍然过大，产生运行时错误。
- Fix: 当kernel包含cat node且类型为embedding时，用`src_stride[0]`修正shape_val的最后一个维度；
  proximity_score阈值从35降到20。
- Reviewability: low -- 需要理解index_select+cat融合的shape传播语义
- Review rule: 算子融合变更需验证所有参与算子的shape/stride在融合后的正确性

### D-165: indirect mem默认配置过于激进+旧环境变量兼容性断裂

- Hashes: f92c0e316
- Root cause: 默认值激进 + 向后兼容遗漏
- Files: torch_npu/_inductor/config.py
- Defect: A5芯片的indirect memory模式被硬编码默认为`simd_simt_mix`，该模式并非所有场景都稳定。
  同时旧版本通过`TRITON_EMBEDDING_FUSION`和`INDUCTOR_ASCEND_INDIRECT_MEMORY_SIMT_TEMPLATE`
  环境变量控制行为，升级后旧变量失效导致行为突变。
- Fix: 默认值改为`None`(关闭)，新增对两个旧环境变量的兼容适配逻辑。
- Reviewability: high -- 默认值变更应有migration文档和兼容性适配
- Review rule: 变更配置默认值时，(1)评估所有使用场景的影响 (2)保持对旧配置方式的向后兼容

### D-166: persistent reduction codegen将reduction轴mask错误包含

- Hashes: d2f944a90
- Root cause: codegen mask生成条件遗漏reduction轴排除
- Files: torch_npu/_inductor/codegen/triton.py, 2个测试文件
- Defect: persistent reduction(单reduction轴)场景下，codegen将reduction轴也加入mask集合，
  生成多余的mask操作导致codegen错误。之前通过skip测试绕过未修根因。
- Fix: 在mask集合构建时增加`node.name[0] != "r"`过滤条件排除reduction轴；
  移除两个测试的`@skip`装饰器恢复CI覆盖。
- Reviewability: high -- (1)skip测试本身是待修复信号 (2)reduction轴不应参与spatial mask
- Review rule: `@skip`装饰器必须关联issue追踪根因修复进度，禁止长期skip作为解决方案

### D-167: 测试断言中头文件路径与实现路径不同步

- Hashes: 07d3542cb
- Root cause: 测试中硬编码路径未跟随实现迁移
- Files: test/_inductor/test_npu_device.py
- Defect: 代码生成的C++ include路径已从实验性路径`experiment/runtime/runtime/rt.h`
  迁移到正式路径`runtime/runtime/rt.h`，但测试断言字符串未同步更新。
- Fix: 更新断言中的路径字符串。
- Reviewability: high -- 路径迁移时全局搜索引用是标准步骤
- Review rule: 路径/命名迁移时，grep搜索旧路径确认所有引用点(含测试和文档)已更新

### D-168: A5芯片缺少NPU特化的fusion距离限制

- Hashes: a91753d0b
- Root cause: 平台特化策略缺失
- Files: torch_npu/_inductor/codegen/__init__.py, scheduling.py
- Defect: GPU默认的fusion距离阈值64对NPU过于宽松，允许距离很远的node融合，
  产生过大的kernel导致A5芯片上运行时错误。
- Fix: 新增`are_long_distant_nodes`函数，proximity_score > 35时阻止融合，
  仅在A5及以上芯片(`soc_version >= 250`)上挂载到`Scheduler`。
- Reviewability: medium -- 需要NPU-specific的性能测试才能暴露kernel过大问题
- Review rule: 从GPU迁移的scheduler/fusion策略需要NPU平台验证阈值合理性

### D-169: SIMT indirect memory模式缺少fallback退出机制
- Hashes: d5e590c8a
- Root cause: 配置选项缺少退出路径
- Files: torch_npu/_inductor/config.py
- Defect: A5芯片的SIMT indirect memory codegen模式只有三种工作模式(simt_template/simt_only/simd_simt_mix)，
  当SIMT模式导致问题时用户无法退回标准SIMD代码生成路径。环境变量设为无效值会被静默回退到默认值而非禁用。
- Fix: 新增"fallback"选项，当`INDUCTOR_INDIRECT_MEMORY_MODE=fallback`时将mode设为None，
  走标准codegen路径。
- Reviewability: high -- 配置选项设计评审时可以发现缺少退出机制
- Review rule: 引入多模式配置时必须提供"禁用/退回默认"选项

### D-170: SIMT模式load/store条件逻辑反转+变量引用错误
- Hashes: a296e9a91
- Root cause: 条件逻辑反转 + 未定义变量引用
- Files: torch_npu/_inductor/config.py, torch_npu/_inductor/codegen/tile_generator.py,
  test/npu/test_torch_npu.py
- Defect: 两个独立bug:
  (1) config.py中`use_store_in_cat`的启用条件写反: `if mode not in [...]`应为`if mode in [...]`，
  导致SIMT模式下本应使用load/store的cat操作走了错误路径;
  (2) tile_generator.py中`tune_simt_num_warps`引用了不存在的`num_warps_list`局部变量
  (应使用参数`tune_num_warps_list`)，且未清空self.configs导致配置累积。
- Fix: (1)将`not in`改为`in`; (2)修正变量引用，添加`self.configs = []`重置。
- Reviewability: high -- 条件反转和变量名错误在代码审查中可发现
- Review rule: 条件表达式中`in`/`not in`需与注释中的意图交叉验证

### D-171: C++头文件<cstring>缺失导致概率性编译失败
- Hashes: 12ed2409e
- Root cause: C++隐式头文件依赖
- Files: torch_npu/csrc/core/npu/NPUException.h
- Defect: NPUException.h使用了`std::memset`但未包含`<cstring>`头文件。部分编译器/平台下
  其他头文件间接包含了`<cstring>`所以能编译通过，但在特定编译环境下概率性失败。
- Fix: 显式添加`#include <cstring>`。
- Reviewability: medium -- 需要了解C++头文件隐式依赖问题，编译器差异导致间歇性失败
- Review rule: 使用C标准库函数(memset/memcpy/strlen等)时检查是否显式include对应的C++头文件

### D-172: profiler "memory optimaze"优化引入解析超时回归
- Hashes: 20357b76d (Revert)
- Root cause: profiler解析器执行顺序与数据依赖不匹配
- Files: torch_npu/profiler/analysis/prof_common_func/_file_manager.py,
  torch_npu/profiler/analysis/prof_config/_parser_config.py,
  torch_npu/profiler/analysis/prof_view/trace_parse/_trace_view_parser.py,
  torch_npu/profiler/analysis/prof_view/cann_parse/_cann_export.py,
  torch_npu/profiler/analysis/prof_view/prepare_parse/_relation_parser.py
- Defect: "memory optimaze"提交重排了profiler解析器的执行顺序并改变了数据传递方式，
  导致在动态Profiling场景下CANNTimelineParser的输出在TraceViewParser需要时尚未就绪，
  timeline解析被误判为超时报错。核心问题是将解析器从串行依赖改为并行时未保留数据流约束。
  同时`append_trace_json_by_path`的签名变更(新增`new_name`参数用于原子rename)
  改变了json拼接逻辑，去掉了空data的`]`闭合处理。
- Fix: 完整revert该优化提交，恢复原始解析器执行顺序和数据流。
- Reviewability: medium -- 需要理解解析器之间的数据依赖关系
- Review rule: 修改多stage pipeline的执行顺序时需绘制数据依赖DAG，验证拓扑排序不变

### D-173: upstream新增模块未加入import可用性测试列表
- Hashes: a01560d77
- Root cause: upstream模块列表与测试白名单不同步
- Files: test/npu/test_public_bindings.py
- Defect: upstream PyTorch新增`torch._inductor.kernel.vendored_templates.cutedsl_grouped_gemm`模块，
  但NPU侧的`test_modules_can_be_imported`测试中的模块排除列表未同步更新，
  导致测试因尝试import该模块而失败。
- Fix: 将新模块添加到测试排除列表中。
- Reviewability: high -- 机械性同步工作，upstream更新时应自动检查
- Review rule: upstream rebase后运行import测试，失败时同步模块列表

### D-174: triton codegen未支持symbolic shape(dynamic编译)
- Hashes: 869571937
- Root cause: codegen中硬编码静态shape假设
- Files: torch_npu/_inductor/codegen/split_tiling.py, test/_inductor/test_add_sum.py
- Defect: `split_tiling.py`中多处直接使用`x.length`属性(假设为整数)，
  但在`torch.compile(dynamic=True)`模式下`x.length`是sympy符号表达式，
  无法直接参与Python算术运算(如`/`、`>=`比较)。`total_split_numels`、`meet_stop_condition`等函数
  均会因对符号表达式执行数值运算而失败。
- Fix: 引入`get_length_val(x)`方法将符号表达式转为可比较的数值;
  重构`meet_stop_condition`和`select_tiling`中的比较逻辑以兼容符号表达式;
  测试新增`dynamic=True`参数化维度。
- Reviewability: medium -- 需要理解sympy符号vs数值的区别，仅在dynamic模式下触发
- Review rule: codegen中涉及range_tree节点的length/numel属性时，检查是否兼容symbolic shape

### D-175: 动态Profiling采集部分rank时解析模式判断缺失
- Hashes: de6c3901f [+8 cherry-picks: 31b2af97c, 2a97ce715, dc3839ac1, d61ec33ea,
  0e93c91c3, c9964e8a3, 651b592f9, b705b90fa]
- Root cause: 条件分支遗漏(async_mode维度未考虑)
- Files: torch_npu/profiler/_dynamic_profiler/_dynamic_profiler_config_context.py,
  torch_npu/profiler/analysis/_profiling_parser.py
- Defect: 动态Profiling指定部分rank采集时，`ConfigContext`无条件将`_analyse`设为False关闭解析。
  但在异步解析模式(`async_mode=True`)下应允许解析，因为异步解析不会阻塞采集流程。
  原代码缺少对`_async_mode`的判断，导致异步模式下也关闭了解析。
- Fix: 添加`if not self._async_mode`条件门控，仅在同步模式下关闭解析;
  同时增强日志输出，显示当前是async还是sync解析模式。
- Reviewability: medium -- 需要理解同步/异步解析的语义差异
- Review rule: 功能开关控制多个模式时，每个模式分支都需独立验证行为

### D-176: MLIR后端非contiguous tensor导致精度问题
- Hashes: cb99a41d1
- Root cause: 模块级初始化副作用 + 非contiguous tensor未处理
- Files: torch_npu/_inductor/__init__.py, torch_npu/_inductor/_lazy_init.py(新建),
  torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/codecache.py,
  test/npu/test_public_bindings.py
- Defect: 两个独立问题:
  (1) `__init__.py`中MLIR相关初始化(`build_ascend_npu_ir_ext`等)在import阶段立即执行，
  即使不使用MLIR也会触发torch_mlir依赖检查;
  (2) MLIR编译路径未检查input/output tensor的contiguity，非contiguous tensor
  直接传入MLIR编译器导致精度错误(内存布局不匹配)。
- Fix: (1)将MLIR初始化抽取到`_lazy_init.py`，延迟到实际需要时执行;
  (2)在codecache中增加contiguity检查和vllm兼容性处理;
  (3)添加多进程编译失败的fallback机制。
- Reviewability: low -- 非contiguous tensor的精度问题需要运行时数值验证才能发现
- Review rule: 编译器后端的tensor处理路径需验证contiguous/非contiguous两种输入

### D-177: Triton libdevice模块路径从ascend重命名为cann
- Hashes: 0c1774b9a
- Root cause: 第三方库(Triton)内部模块路径重命名未同步
- Files: torch_npu/_inductor/npu_triton_helpers.py
- Defect: Triton库将`triton.language.extra.ascend.libdevice`重命名为
  `triton.language.extra.cann.libdevice`，NPU侧硬编码的import路径失效。
  同时`tl.extra.ascend.libdevice`属性访问也失效。
- Fix: 用try/except做版本兼容: 先尝试新路径`cann`，失败则回退到旧路径`ascend`。
  import和attribute access都做了双路径处理。
- Reviewability: high -- 第三方库升级后import报错，日志明确
- Review rule: 第三方库的内部模块路径引用应有版本兼容fallback或通过public API访问

### D-178: get_soc_version函数调用缺少括号
- Hashes: 230e8f344
- Root cause: 函数调用括号遗漏(语法正确但语义错误)
- Files: torch_npu/_inductor/codegen/triton.py
- Defect: 在两处`get_soc_version >= 250`的比较中，`get_soc_version`是函数对象而非返回值，
  缺少`()`调用。Python 3中函数对象与整数比较会抛`TypeError`，
  导致A5芯片的int64 index dtype特化路径在运行时崩溃。
- Fix: `get_soc_version` → `get_soc_version()`，两处均修正。
- Reviewability: high -- 函数vs函数调用的区别在代码审查中可发现，静态分析工具(mypy/pylint)也能检测
- Review rule: 对返回值进行比较的表达式，确认函数名后有`()`;
  建议配置pylint/mypy检查函数对象参与比较运算

### D-179: sum算子负轴索引只修正了最后一个轴
- Hashes: a3af87c1c
- Root cause: 负值处理仅覆盖部分场景(只检查最后一个元素而非全部)
- Files: torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/npu/inductor_patch/lowering.py
- Defect: MLIR lowering中`sum_`函数处理负轴时，`if axis and axis[-1] < 0`只检查最后一个轴是否为负，
  且`axis = [ax + offset for ax in axis]`将所有轴都加了offset(即使非负轴不需要)。
  当axis=[1, -2]时，正确轴1也被错误偏移。
- Fix: `axis[-1] < 0` → `any(ax < 0 for ax in axis)`;
  `[ax + offset for ax in axis]` → `[ax + offset if ax < 0 else ax for ax in axis]`。
- Reviewability: high -- 负值处理的边界条件在review时可通过构造反例发现
- Review rule: 处理可为负的索引列表时，逐元素判断而非只检查首/尾元素

### D-180: A5芯片cumsum强制降精度为int32导致溢出
- Hashes: 0b6e500d8
- Root cause: 硬件代际差异未区分的dtype降精度策略
- Files: torch_npu/_inductor/lowering.py
- Defect: cumsum的lowering对整数/布尔输入无条件使用int32作为计算dtype，
  这在旧芯片(910B)上是合理的(硬件限制)，但A5(soc >= 250)支持int64累加。
  大规模tensor的cumsum在int32下溢出产生错误结果。
- Fix: 根据SoC版本选择dtype: `torch.int64 if get_soc_version() >= 250 else torch.int32`。
- Reviewability: medium -- 需要大规模数据才能暴露溢出，但代码中硬编码int32是明显的审查点
- Review rule: 涉及dtype降精度的lowering需标注硬件约束来源，并随新硬件代际更新

### D-181: int32 index在A5芯片上的dtype溢出
- Hashes: 6413cdfd1
- Root cause: codegen中index dtype硬编码为int32
- Files: torch_npu/_inductor/codegen/triton.py
- Defect: `IterationRangesEntryNPUIndex`生成的index变量(program_id、arange等)
  始终使用int32，在A5芯片上当tensor维度超过int32范围时索引溢出。
  A5(soc >= 250)的64位地址空间需要int64 index支持。
- Fix: 添加`get_soc_version()`检查，A5芯片上当kernel指定`index_dtype="tl.int64"`时，
  为program_id和arange结果插入`.to(tl.int64)` cast;
  对常量index使用`tl.full(..., tl.int64)`。
- Reviewability: medium -- 需要超大tensor才能触发，但int32 index是已知的规模限制
- Review rule: NPU codegen中的index dtype需跟随硬件代际的地址宽度

### D-182: transfer_to_npu无条件禁用torch.jit.script
- Hashes: b2fde6fce
- Root cause: monkey-patch粒度过粗(全局替换为空操作)
- Files: torch_npu/contrib/transfer_to_npu.py
- Defect: `transfer_to_npu`模块将`torch.jit.script`替换为直接返回输入对象的空函数，
  导致所有jit.script调用都被静默跳过，即使是能正常工作的script也被禁用。
  原因是部分NPU算子不支持script编译，但应该fallback而非全局禁用。
- Fix: 保存原始`torch.jit.script`引用，新实现先尝试调用原始script，
  仅在抛异常时fallback返回原对象，并用`_warned_jit_fallback`标志只警告一次。
- Reviewability: high -- `return obj`直接跳过script的实现在review时显然过于粗暴
- Review rule: monkey-patch替换框架函数时，优先try-original-then-fallback而非直接替换为no-op

### D-183: CachingHostAllocator中event query异常未处理
- Hashes: bedc8509b
- Root cause: C++异常处理路径缺失
- Files: torch_npu/csrc/core/npu/CachingHostAllocator.cpp,
  test/npu/test_fault_mode.py
- Defect: `processEvents`循环中`event->query()`可能抛异常(如设备故障或NPU reset后)，
  未捕获的异常会终止event清理循环，导致后续内存分配阻塞或泄漏。
  同时测试中的错误信息regex与CANN新版本的错误文案不匹配。
- Fix: 用try/catch包裹`event->query()`，异常时log错误并将event标记为已完成(释放资源)，
  避免一个失败的event阻塞整个清理流程。测试regex同步更新。
- Reviewability: medium -- 异常路径需要故障注入测试才能触发
- Review rule: 循环内的硬件/runtime调用必须有异常处理，防止单次失败阻塞批量操作

### D-184: upstream将triton_heuristics移至runtime子模块
- Hashes: 5d87acc01
- Root cause: upstream模块重组后import路径未同步
- Files: torch_npu/_inductor/__init__.py
- Defect: upstream PyTorch将`torch._inductor.triton_heuristics`重组到
  `torch._inductor.runtime.triton_heuristics`，NPU侧的import路径未跟随更新，
  导致`_replace_benchmark_all_configs`函数中import失败(ModuleNotFoundError)。
- Fix: 更新import路径: `from torch._inductor.triton_heuristics` →
  `from torch._inductor.runtime.triton_heuristics`。
- Reviewability: high -- import错误在首次运行时立即报错
- Review rule: upstream rebase后检查所有torch._inductor.*的import路径(与D-129/D-173同族)

### D-185: A5 indirect memory多重间接索引及index_select维度支持
- Hashes: 9c1608cef
- Root cause: indirect memory template缺少多种边界case处理
- Files: torch_npu/_inductor/codegen/ir.py, torch_npu/_inductor/codegen/triton.py,
  torch_npu/_inductor/lowering.py
- Defect: A5 indirect memory codegen的三个独立缺陷:
  (1) ir.py中多重间接索引(indirect内嵌indirect)未检测，生成无效template;
  (2) triton.py中index_select对dim=1(非低维)不支持，以及对weight_index的有效性检查不足;
  (3) lowering.py中scatter template在input包含size=1维度时生成错误代码。
  另外indirect变量被错误地包含在stride排序映射中，干扰tiling轴选择。
- Fix: (1)检测`multi_indirect_index`标记并fallback到标准load;
  (2)增加`is_correct_weight_index`验证和dim=1支持(reshape);
  (3)scatter template增加size=1的input校验;
  (4)过滤INDIRECT类型符号不参与stride排序。
- Reviewability: low -- 需要特定的indirect memory模型输入才能触发
- Review rule: codegen template新增时需覆盖: 多重间接、非低维dim、size=1边界

### D-186: NPUGraphNode输入遍历未释放tensor引用导致内存错误
- Hashes: 5721ff35d
- Root cause: Python for循环中tensor引用延长生命周期
- Files: torch_npu/npu/_graph_tree.py
- Defect: `NPUGraphNode`中遍历inputs列表时，循环变量`item`持有对每个tensor的引用，
  直到循环结束才释放。在NPU graph capture/replay场景下，被capture的tensor
  不应在capture期间有额外引用，否则导致内存管理冲突或OOM。
- Fix: 在循环体末尾显式`del item`释放引用。
- Reviewability: medium -- Python引用计数导致的内存问题需要profiling才能发现
- Review rule: NPU graph相关代码中遍历tensor集合时，确保循环变量不延长tensor生命周期

### D-187: DTensor测试要求4卡但实际只需2卡
- Hashes: 12d689a08
- Root cause: 测试world_size硬编码与实际需求不匹配
- Files: test/distributed/tensor/test_dtensor_ops.py,
  test/distributed/tensor/test_sharded_optim.py(新增)
- Defect: `test_torch_nn_functional_one_hot`测试用`@skipIfUnsupportMultiNPU(4)`
  要求4张NPU卡，但测试本身只需2卡。在2卡环境下被跳过，减少了测试覆盖率。
  同时新增了`test_sharded_optim.py`测试文件。
- Fix: 将skip条件从4改为2: `@skipIfUnsupportMultiNPU(2)`。
- Reviewability: high -- 测试skip条件与实际需求不匹配在review时可发现
- Review rule: 分布式测试的NPU卡数要求应与测试中实际使用的world_size一致

### D-188: npu_graph_attention_function的op schema参数不完整
- Hashes: 6cb77d54c
- Root cause: NPU FA算子新增参数后graph lowering注册未同步
- Files: test/_inductor/test_npu_fusion_attention_graph.py,
  torch_npu/_inductor/npu_fusion_attention_graph.py
- Defect: `npu_fusion_attention`算子新增了`softmax_layout`和`sink`两个可选参数，
  但graph模式下的op schema定义(`npu_fa`和`npu_fa_backward`)未包含这些参数，
  导致带这些参数的FA调用在graph模式下schema不匹配而core dump。
  同时backward的返回值个数也需更新(4→5)。
- Fix: 在npu_def.define中补充`softmax_layout`和`sink`参数;
  更新backward返回值tuple长度; 移除测试中的`@skip("skip for core dump")`。
- Reviewability: high -- op schema与实现不匹配在首次调用时即报错
- Review rule: NPU自定义op添加参数时需同步更新所有注册点(aten/meta/graph/decomp)

### D-189: DTensor测试类world_size硬编码为4导致2卡环境失败
- Hashes: d2d756fc2
- Root cause: 测试基类world_size默认值与实际硬件不匹配
- Files: test/distributed/tensor/test_math_ops.py
- Defect: `TestConv2d`和`TestGroupedMatmulAdd`测试类继承的`DTensorTestBase`默认world_size=4，
  但这些测试标注了`@skipIfUnsupportMultiNPU(2)`(即2卡即可运行)。
  当环境有2-3张卡时，world_size=4导致初始化失败，但skip条件又允许运行，产生矛盾。
- Fix: 覆写`world_size`属性为动态值: `min(4, device_count)`，适配实际可用的NPU数量。
- Reviewability: high -- 测试类的world_size与skip条件不一致在review时可发现
- Review rule: 分布式测试的world_size应动态适配可用设备数，与skip条件保持一致

### D-190: graph模式send/recv的公共绑定未注册
- Hashes: 45aee0d1d
- Root cause: 新增graph converter后公共绑定列表未同步更新
- Files: test/npu/test_public_bindings.py
- Defect: `hcom_send_recv`的graph模式converter已实现，但未添加到`test_public_bindings.py`的
  公共API列表中。public bindings测试会检查所有导出的模块路径是否在白名单中，
  遗漏导致CI测试失败。
- Fix: 在白名单中添加`torch_npu.dynamo.torchair...hcom_send_recv`。+1行。
- Reviewability: high -- 新增converter时同步更新绑定列表是标准流程
- Review rule: 添加新的graph converter时检查test_public_bindings.py是否需要同步更新

### D-191: aclrtMemcpyAsyncWithCondition存在时跳过D2H同步的逻辑错误
- Hashes: 5d39f5d9b
- Root cause: 异步拷贝API存在性被误用为"无需同步"的判据
- Files: torch_npu/csrc/core/npu/CachingHostAllocator.cpp,
  torch_npu/csrc/core/npu/interface/AclInterface.cpp
- Defect: `process_unregistered_mem_location_type()`中，当`AclrtMemcpyAsyncWithCondition`可用且
  拷贝方向为D2H时，直接返回ACL_ERROR_NONE跳过同步。但该API的存在性并不意味着
  非pinned内存的D2H拷贝也不需要同步。对于malloc分配的host内存，DMA仍需同步等待
  完成才能安全读取数据。同时`AclrtPointerGetAttributesExist()`缺少CANN版本守卫，
  在低版本CANN上dlsym可能返回非null但实际不可用的地址。
- Fix: 移除D2H方向的sync跳过逻辑，恢复无条件同步;
  在`AclrtPointerGetAttributesExist()`中添加CANN>=8.5.0版本检查。
- Reviewability: medium -- 需要理解异步拷贝语义与内存pinning的关系
- Review rule: API存在性检查不能替代功能语义判断；异步操作的同步点需基于内存属性而非API可用性

### D-192: fake tensor序列化路径访问设备storage导致core dump
- Hashes: bbd34a48a
- Root cause: skip_data模式下仍先访问设备侧storage再判断是否跳过
- Files: torch_npu/utils/serialization.py
- Defect: `_npu_save()`中序列化逻辑的执行顺序：先判断`storage.device.type != "cpu"`并构造
  `storage_tensor`(需要访问设备侧storage)，再检查`_serialization_tls.skip_data`。
  当使用skip_data模式序列化fake tensor时，fake tensor的storage不在真实设备上，
  `_tensor_construct_from_storage()`访问无效storage导致core dump。
  同时缺少对`is_fake`(有`_fake_device`属性)和`is_meta`(device=="meta")的类型判断。
- Fix: 重排条件分支：先检查`is_fake`/`is_meta`/`cpu`走安全路径，否则才访问设备storage;
  skip_data检查提前到storage访问之前;
  额外检查`storage.data_ptr() == 0`防止空指针。
- Reviewability: medium -- 需要理解fake tensor与真实tensor的storage差异
- Review rule: 序列化路径中涉及设备访问的操作必须在skip_data/fake/meta检查之后

### D-193: return语句缩进错误导致fallback逻辑死代码
- Hashes: 5b503a600
- Root cause: 代码缩进错误(return在if块内而非if块外)
- Files: torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/codecache.py
- Defect: `CustomAsyncCompile`类中，当`kernel.launchers`为空时进入fallback逻辑，
  调用`_load_fx_graph()`加载FX graph替代。但紧接其后的`return kernel`与
  `_load_fx_graph()`的return在同一缩进层级(都在if块内)，导致`return kernel`永远
  不可达。实际意图是if块结束后返回kernel(非空launchers的正常路径)。
- Fix: 将`return kernel`的缩进从if块内移到if块外(去掉一级缩进)。-1行+1行。
- Reviewability: high -- 代码审查时应注意return语句的缩进层级
- Review rule: 函数中多个return路径时验证每个return的可达性，特别注意Python缩进

### D-194: upstream将triton_dtype从utils移至codegen.triton并重命名为triton_type
- Hashes: 0b05532e2
- Root cause: upstream API重命名(triton_dtype→triton_type)且模块路径变更
- Files: torch_npu/_inductor/codegen/triton.py
- Defect: NPU侧`from torch._inductor.utils import triton_dtype`在upstream重命名后抛出
  ImportError。该import由D-197(13b32d623)在前一天引入，引入时已经过时。
- Fix: 移除全局import `triton_dtype`; 在`truediv()`内使用局部import
  `from torch._inductor.codegen.triton import triton_type`。
- Reviewability: high -- import失败在CI首次运行即暴露
- Review rule: 添加upstream import时验证目标符号在当前对齐的PyTorch版本中是否存在
- Note: D-197引入错误import → D-194次日修复，构成"引入bug后修复"对

### D-195: npu.synchronize的dynamo trace rule类型错误(SkipFunction→InGraph)
- Hashes: 4bff295e8 [+2 cherry-picks: 092a66d7d, 750a0f616]
- Root cause: 将synchronize注册为SkipFunctionVariable而非TorchInGraphFunctionVariable
- Files: torch_npu/dynamo/trace_rule.py
- Defect: `torch_npu.npu.utils.synchronize`被注册为`SkipFunctionVariable`(在
  `manual_torch_name_rule_map`中)，含义是dynamo遇到该函数时中断trace并fallback。
  但正确行为应是将其标记为in-graph function，让dynamo在graph中保留该调用，
  否则包含synchronize的代码段无法被完整trace。
- Fix: 将synchronize添加到`torch_c_binding_in_graph_functions_npu`字典(TorchInGraphFunctionVariable);
  移除`manual_torch_name_rule_map`中的SkipFunctionVariable注册;
  同时移除已不需要的`from torch._dynamo.trace_rules import manual_torch_name_rule_map, SkipFunctionVariable`。
- Reviewability: medium -- 需要理解SkipFunction vs InGraph的语义差异
- Review rule: NPU设备操作(sync/stream/event)的dynamo trace类型应与CUDA对应操作保持一致

### D-196: DTensor attention测试world_size=4导致2卡CI环境失败(多文件)
- Hashes: 8d2e08f89 [+1 cherry-pick: 1d3f5bc9f]
- Root cause: 测试基类world_size硬编码与CI硬件环境不匹配(与D-189同类)
- Files: test/distributed/tensor/test_attention_ops.py, test/distributed/tensor/test_gather_swiglu.py,
  test/distributed/tensor/test_math_ops.py, test/distributed/tensor/test_matrix_ops.py,
  test/distributed/tensor/test_moe_ops.py
- Defect: 5个DTensor测试文件的测试类使用默认world_size=4，但CI环境仅2张NPU卡。
  `@skipIfUnsupportMultiNPU(4)`同时改为`(2)`，配合`@SupportedDevices(['Ascend910B'])`。
  原始的4卡要求导致在2-3卡环境下所有DTensor测试被跳过。
- Fix: 覆写world_size属性为`min(4, device_count)`;
  所有测试方法的skip阈值从4降为2; 添加SupportedDevices装饰器。
- Reviewability: high -- 与D-187/D-189完全相同的模式
- Review rule: 参照D-189，分布式测试的设备数要求应动态化

### D-197: truediv覆写缺失导致triton低精度除法精度错误
- Hashes: 13b32d623
- Root cause: upstream新增truediv精度处理后NPU侧未注册覆写
- Files: torch_npu/_inductor/codegen/__init__.py, torch_npu/_inductor/codegen/triton.py
- Defect: upstream在`TritonOverrides.truediv`中添加了低精度浮点(bf16/fp16)的精度保护逻辑
  (先upcast再除)。NPU侧的`NPUTritonKernelOverrides`未继承该覆写，导致triton
  codegen中低精度除法可能丢失精度。同时该commit引入了已过时的
  `from torch._inductor.utils import triton_dtype`，次日被D-194修复。
- Fix: 在triton.py中导入`low_precision_fp_var`和`get_dtype_handler`;
  实现`truediv()`函数; 在__init__.py中将`TritonOverrides.truediv = truediv`。
- Reviewability: medium -- 需要追踪upstream TritonOverrides的变更
- Review rule: upstream codegen覆写新增method时需检查NPU侧是否需要同步覆写
- Note: 本commit引入D-194的import bug，构成引入-修复对

### D-198: revert ProcessGroupHCCL的syncOp流选择优化
- Hashes: 3f372302e
- Root cause: 修复引入回归(sync op使用当前流导致stream管理和内存安全问题)
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: 此前的优化尝试区分async/sync操作使用不同的流：async op使用预分配的HCCL专用流，
  sync op直接使用当前流(跳过syncStreams和recordStream)。这破坏了两个不变量：
  (1) syncStreams确保当前流的数据对HCCL流可见(event同步);
  (2) recordStream/stash确保collective完成前tensor不被回收。
  对sync op跳过这些保护后，在多stream场景中出现数据竞争和UAF。
- Fix: 完整revert该优化，恢复所有collective统一使用HCCL专用流+syncStreams+recordStream。
  涉及`collective()`和`collectiveCoalesced()`两个函数。~120行删除。
- Reviewability: low -- 需要深入理解HCCL流同步和内存生命周期模型
- Review rule: 分布式通信的流管理优化需要经过多stream并发压力测试验证

### D-199: scheduler模块从torch导入Sequence类型但upstream已移除该re-export
- Hashes: 66cb389e5
- Root cause: upstream重构移除了scheduler模块对typing.Sequence的re-export
- Files: torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/npu/inductor_patch/scheduler.py,
  torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/npu/inductor_patch/__init__.py,
  torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/npu/codegen/wrapper.py,
  torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/npu/npu_inductor_plugin.py
- Defect: scheduler.py中`from torch._inductor.scheduler import Sequence`失败，因为upstream
  不再从scheduler re-export `Sequence`。同时`__init__.py`中用`importlib.reload(graph)`
  替代scheduler patch，以及plugin.py中有多个未使用的import(torch.nn.functional等)
  和`_triton.has_triton = lambda: False`的全局monkey-patch。
- Fix: `Sequence`改为从`typing`导入; 添加`_patch_scheduler()`函数调用;
  移除plugin.py中的dead import和triton hack; 清理wrapper.py中的import顺序。
- Reviewability: high -- import失败在CI即暴露; dead import和全局patch可在review中发现
- Review rule: 从upstream模块import时优先用标准库(typing)而非upstream的re-export

### D-200: 兼容性测试因profiler新增max_process_number参数误报不兼容
- Hashes: 7465f3176
- Root cause: 兼容性测试未豁免已知的合法新增参数
- Files: test/npu/test_compatibility.py
- Defect: `test_compatibility.py`检查API签名变更，新增参数会触发不兼容告警。
  `torch_npu.profiler.profiler.analyse`合法新增了`max_process_number`参数，
  但兼容性检查将其标记为不兼容变更，导致CI失败。
- Fix: 添加special case: 当api为`profiler.analyse`且new_diff_params包含`max_process_number`时
  从差异集中排除该参数。
- Reviewability: high -- 兼容性测试失败信息明确指出哪个API哪个参数
- Review rule: 添加合法的新API参数时需同步更新兼容性测试的豁免列表

### D-201: upstream移除group_fn后NPU侧monkey-patch引发KeyError
- Hashes: 32b8fd81a
- Root cause: upstream删除TritonScheduling.group_fn后NPU覆写赋值失败
- Files: torch_npu/_inductor/codegen/__init__.py, torch_npu/_inductor/codegen/triton.py
- Defect: NPU侧自定义了`group_fn`并在`__init__.py`中执行
  `TritonScheduling.group_fn = group_fn`。upstream移除了该方法后，虽然赋值不会报错
  (Python允许动态添加属性)，但upstream调用路径不再使用`group_fn`，且该覆写中
  引用的`NumelList`等类型也可能已变更，导致运行时KeyError。
- Fix: 移除自定义`group_fn`函数(12行)及其monkey-patch赋值。
- Reviewability: high -- upstream方法删除后的dead override在review中可发现
- Review rule: 定期检查NPU侧monkey-patch的目标方法是否仍存在于upstream

### D-202: P2P连接数限制在A3+ SoC上不应生效
- Hashes: 0bc2a875e
- Root cause: 硬件代际限制未添加SoC版本守卫
- Files: torch_npu/csrc/core/npu/NPUPeerToPeerAccess.cpp
- Defect: `get_p2p_access()`中`C10_P2P_ACCESS_MAX_NPUS`连接数限制对所有SoC生效。
  但A3(Ascend910_9391)及以后的SoC架构已取消P2P连接数限制，
  无条件限制导致A3+上多卡P2P通信失败。
- Fix: 将连接数限制逻辑包裹在`if (GetSocVersion() < Ascend910_9391)`守卫中，
  A3+直接跳过限制检查。
- Reviewability: medium -- 需要了解不同SoC代际的P2P能力差异
- Review rule: 硬件限制相关的代码需添加SoC版本守卫，与D-90/D-148同族

### D-203: gcc编译优化flags的patch方法应用失败
- Hashes: de4e20b1f
- Root cause: D-205的inline patch在__init__.py中直接赋值导致patch时序问题
- Files: torch_npu/_inductor/__init__.py, torch_npu/_inductor/cpp_builder.py
- Defect: D-205(4dcc0c0c2)将`patch_get_optimization_cflags`函数内联在`__init__.py`中，
  通过`cpp_builder._get_optimization_cflags = patch_get_optimization_cflags`赋值。
  但该赋值发生在模块级别，patch的函数签名和调用方式可能与upstream不一致。
  此commit将函数移至独立的`cpp_builder.py`模块，以正确的方式导入和调用。
- Fix: 移除`__init__.py`中内联的30行函数定义;
  在`cpp_builder.py`中重新实现`_get_optimization_cflags()`和`patch_get_optimization_cflags()`;
  将monkey-patch从直接赋值改为函数调用`patch_get_optimization_cflags()`。
- Reviewability: medium -- patch的结构问题需要对比upstream的调用方式
- Review rule: monkey-patch应放在独立模块中，而非内联在__init__.py; patch函数应封装赋值逻辑

### D-204: IsGteDriverVersion在低版本CANN上抛异常而非返回false
- Hashes: 4d88e0118
- Root cause: 版本检查函数用TORCH_CHECK抛异常处理不支持场景
- Files: torch_npu/csrc/core/npu/GetCANNInfo.cpp
- Defect: `IsGteDriverVersion()`在CANN版本低于8.1.RC1时执行`TORCH_CHECK(false, ...)`，
  直接抛出异常终止程序。但该函数被用于条件判断(`if (IsGteDriverVersion(...))`），
  调用方期望返回bool而非异常。低版本CANN环境下所有使用该函数的代码路径都会crash。
  同时错误信息使用模糊的"this function"而非具体函数名。
- Fix: 将`TORCH_CHECK(false, ...)`替换为`TORCH_NPU_WARN_ONCE(...); return false;`，
  使低版本CANN环境优雅降级而非crash;
  错误信息中明确函数名(IsGteCANNVersion/GetCANNVersion)。
- Reviewability: high -- TORCH_CHECK(false)在条件检查函数中是明显的设计错误
- Review rule: 版本/能力检查函数应返回bool，不应抛异常; TORCH_CHECK(false)只用于不可恢复错误

### D-205: A3平台GCC不识别-march=native编译标志
- Hashes: 4dcc0c0c2 [+2 cherry-picks: 785e3cd2d, 4833cf2f0]
- Root cause: inductor编译扩展使用的CPU flags在A3的GCC 13上不被支持
- Files: torch_npu/_inductor/__init__.py
- Defect: upstream的`_get_optimization_cflags()`在非Darwin系统上添加`-march=native`等
  CPU特定编译标志。A3平台(ARM架构)的GCC 13不认识这些x86标志，导致inductor图模式
  编译AOTI wrapper时失败。
- Fix: 复制upstream的`_get_optimization_cflags()`并移除`-march=native`(仅保留ppc64le
  的`-mcpu=native`); 通过monkey-patch替换upstream实现。
- Reviewability: medium -- 需要了解目标平台的GCC支持的编译标志
- Review rule: 涉及编译标志的代码需在目标硬件平台上验证
- Note: 本commit的inline实现方式后被D-203重构修复

### D-206: empty_like_npu在FakeTensorMode下访问非NPU storage导致错误格式
- Hashes: 8a5b0ea30
- Root cause: 未区分NPUStorageImpl与其他storage类型(FakeTensor)
- Files: torch_npu/csrc/aten/common/TensorFactories.cpp, test/npu/test_npu.py
- Defect: `empty_like_npu()`中通过`NPUBridge::GetNpuStorageImpl(self)->npu_desc_.npu_format_`
  获取NPU格式。在FakeTensorMode下，self的storage不是NPUStorageImpl而是FakeTensor的
  mock storage，强制转换后读取到垃圾数据作为npu_format。
  这导致`_native_mha`在compile模式下调用`empty_like`时生成错误格式的tensor。
- Fix: 添加`typeid`检查: `typeid(*self.storage().unsafeGetStorageImpl()) == typeid(NPUStorageImpl)`。
  若非NPUStorageImpl，使用默认格式`ACL_FORMAT_ND`; 否则获取真实NPU格式。
- Reviewability: medium -- 需要理解FakeTensorMode下的storage类型差异
- Review rule: 访问NPU特有的storage属性前必须验证storage的实际类型

### D-207: aclgraph_report_shape环境变量控制方式不合理，改为API参数
- Hashes: df4480731
- Root cause: 环境变量控制graph capture的report_shape功能，粒度过粗且易误用
- Files: torch_npu/csrc/core/npu/NPUGraph.cpp, torch_npu/csrc/core/npu/NPUGraph.h,
  torch_npu/csrc/framework/interface/EnvVariables.cpp
- Defect: `apply_cache_op_info()`通过环境变量`ACLGRAPH_REPORT_SHAPE`控制是否启用
  report shape功能。这有两个问题: (1)环境变量是进程级全局的，无法按graph粒度控制;
  (2)默认行为是"未设置=启用"，disable需要显式设`disable`，语义不直观。
  且`capture_begin()`总是传`true`给`apply_cache_op_info`，环境变量检查冗余。
- Fix: 移除`ACLGRAPH_REPORT_SHAPE`环境变量和`REGISTER_OPTION`;
  将`report_shape`作为`capture_begin()`的bool参数(默认true);
  `apply_cache_op_info()`直接使用传入的`enabled`参数，不再查询环境变量。
- Reviewability: high -- 环境变量控制运行时行为是反模式，应改为API参数
- Review rule: 运行时功能控制优先用API参数而非环境变量，尤其是需要细粒度控制的功能

### D-208: report shape日志信息不准确，用户无法判断功能状态
- Hashes: 598d02a4f
- Root cause: 错误提示文案含糊，未明确告知功能状态
- Files: torch_npu/csrc/core/npu/NPUGraph.cpp
- Defect: `apply_cache_op_info()`在`AclrtSetStreamAttribute`返回`ACL_ERROR_RT_PARAM_INVALID`时，
  日志打印"the provided parameters are invalid. This may be caused by an incompatible CANN version"，
  信息冗长且未直接说明report shape功能已被禁用，用户需猜测实际影响。
- Fix: 简化日志为"Report shape function is disabled due to incompatible CANN version."，
  直接表达功能状态和原因。
- Reviewability: high -- 日志文案审查应关注是否清晰传达了功能状态
- Review rule: 错误/警告日志应直接说明"什么功能受影响"+"为什么"，避免generic错误描述

### D-209: shutdown时host block析构触发record_stream导致crash
- Hashes: b9ea48873
- Root cause: 资源释放顺序错误——CachingHostAllocator在shutdown阶段仍尝试record stream
- Files: torch_npu/csrc/InitNpuBindings.cpp, torch_npu/csrc/core/npu/CachingHostAllocator.cpp,
  torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp, torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h,
  torch_npu/csrc/libs/init_npu.cpp
- Defect: NPU shutdown流程中，`npu_shutdown()`先释放HCCL comm，再清空host allocator缓存。
  但`CachingHostAllocator::emptyCache()`释放host block时，析构路径会调用`record_stream()`
  为每个block创建event并record到stream上。此时device资源可能已部分释放(stream已销毁)，
  导致event创建失败crash。
- Fix: 在shutdown流程的HCCL释放后、allocator清空前，调用`NpuSysCtrl::HostFinalize()`
  设置`host_finalize_flag_=true`; CachingHostAllocator的`record_stream()`检查该flag，
  为true时直接return跳过event创建。同时在`finalize_npu()`中也添加相同调用。
- Reviewability: medium -- 需要理解shutdown时各组件析构的时序依赖
- Review rule: shutdown/finalize路径中的资源释放操作必须考虑其他组件是否仍持有引用;
  资源释放顺序应有明确的层级定义

### D-210: npu_fusion_attention缺少DTensor sharding策略注册
- Hashes: 488c9a0ae [+1 cherry-pick: 788916261]
- Root cause: NPU自定义op缺少分布式tensor策略注册(与D-60/D-94同族)
- Files: torch_npu/distributed/tensor/_attention.py, torch_npu/distributed/tensor/_dtensor_patch.py,
  torch_npu/distributed/tensor/_matrix_ops.py, torch_npu/distributed/tensor/__init__.py,
  test/distributed/tensor/test_attention_ops.py
- Defect: `npu_fusion_attention`及其backward之前使用`distribute_module`方式注册sharding策略，
  存在四个问题: (1)旧策略简单跟随query的placement，当sharding S/D维度时会导致与单设备
  计算结果不一致; (2)TP场景下本地tensor的head维度与`head_num`参数不匹配; (3)backward
  中kwargs包含Tensor但DTensor框架默认只处理args中的Tensor; (4)upstream PyTorch PR#168249
  才支持kwargs中的Tensor sharding，旧版本需要patch。
- Fix: 完全重写为`@register_sharding`装饰器方式(+785行)。为forward和backward分别注册
  Replicate/DP(Shard batch)/TP(Shard head)三种策略; TP模式下自动调整`head_num`;
  为backward添加kwargs中DTensor的redistribution处理; 新增`_dtensor_patch.py`为旧版
  PyTorch补丁; 添加覆盖BNSD/BSND/BSH/TND四种layout的测试用例。
- Reviewability: low -- 需要深入理解DTensor sharding语义和attention op的维度布局
- Review rule: 新增NPU自定义op时必须同步注册DTensor sharding策略

### D-211: codegen中BLOCK_SUB错误地向上取power-of-2
- Hashes: 97ead3efa [+2 cherry-picks: 913083da4, 31d5bf49a]
- Root cause: NPU Triton codegen对index kernel的sub-block size错误地取next_power_of_2
- Files: torch_npu/_inductor/codegen/triton.py
- Defect: `NPUIndexTritonKernel`的sub-block size计算中，`simplified_tree_numel`得到实际
  numel值后调用`next_power_of_2(val)`向上取整到2的幂次。但NPU Triton kernel的sub-block
  不需要power-of-2对齐(这是CUDA warp size的约束)，错误的向上取整导致分配过多资源，
  可能引发实际计算范围与mask不匹配。
- Fix: 移除`next_power_of_2(val)`调用，直接使用实际numel值作为BLOCK_SUB。
- Reviewability: medium -- 需要理解NPU与CUDA在block size对齐要求上的差异
- Review rule: 从CUDA移植的codegen逻辑需审查硬件特定约束(如power-of-2对齐)是否适用于NPU

### D-212: HCCL头文件未同步新增CANN版本的配置字段
- Hashes: eee5ce316
- Root cause: third_party HCCL头文件版本落后于实际CANN版本
- Files: third_party/hccl/inc/hccl/hccl.h, third_party/hccl/inc/hccl/hccl_types.h
- Defect: CANN新版本在`HcclCommConfig`结构体中新增了`hcclExecTimeOut`、`hcclAlgo`、
  `hcclRetryEnable`、`hcclRetryParams`四个字段，并将`HCCL_COMM_CONFIG_VERSION`从8升到9。
  torch_npu中的头文件副本未同步这些字段，导致`HcclCommConfigInit()`未初始化新字段，
  创建comm时这些字段包含随机值，可能引起不可预测行为。
- Fix: 在`hccl_types.h`中添加新常量定义(`HCCL_COMM_ALGO_MAX_LENGTH`等)和结构体新字段;
  在`hccl.h`的`HcclCommConfigInit()`中初始化新字段; 升级版本号到9。
- Reviewability: high -- CANN版本升级后应系统检查所有third_party头文件
- Review rule: CANN版本升级时必须对比third_party/hccl/inc下所有头文件与CANN安装包的差异

### D-213: test_schedule_multiproc测试模型split数量与调度不匹配
- Hashes: be62cae61
- Root cause: upstream PyTorch pipeline调度测试扩展了split数量，torch_npu的model registry未同步
- Files: test/distributed/pipelining/model_registry.py,
  test/distributed/pipelining/test_schedule_multiproc.py
- Defect: `ExampleCode`模型只有2个`pipe_split()`，但测试用例中可能使用3或4个stage的调度器，
  导致stage数量与模型split点不匹配，测试失败。
- Fix: 为`ExampleCode.__init__`添加`splits`参数(默认2)，当`splits>2`时新增`pipe_split()`
  和`lin1`层，当`splits>3`时再增加`pipe_split()`和`lin2`层，使模型可配置支持2-4个stage。
- Reviewability: medium -- 需要了解pipeline parallelism中split点与stage数的对应关系
- Review rule: 分布式测试模型的结构应能覆盖测试中使用的所有配置组合

### D-214: profiler采集内存流ID时直接调用不存在的aclrtStreamGetId接口
- Hashes: 21c04c5ab [+8 cherry-picks: ecf46ee1f, 28f7be5f8, ac6df14fd, b26da6b9d,
  7c5e50b8c, 80e96f25e, 9e6310780, 85d2a49b9]
- Root cause: API可用性检查遗漏驱动版本维度(与D-63/D-79同族)
- Files: torch_npu/csrc/core/npu/interface/AclInterface.cpp,
  torch_npu/csrc/core/npu/interface/AclInterface.h,
  torch_npu/csrc/profiler/npu_profiler.cpp
- Defect: `reportMemoryDataToNpuProfiler()`直接调用`AclrtStreamGetId(data.stream, &stream_id)`
  获取流ID。在低版本CANN/driver中该接口不存在(dlsym返回nullptr)，直接调用导致空指针crash，
  中断训练。同时`stream_id`声明为`int32_t`但后续使用时需要`int64_t`。
- Fix: 添加`IsExistRtGetStreamId()`函数通过dlsym检查接口是否存在;
  调用前先检查，存在时正常获取stream_id(int32_t→int64_t提升)，
  不存在时回退为`reinterpret_cast<int64_t>(data.stream)`(使用stream指针值)。
- Reviewability: high -- 调用CANN动态加载API前必须检查存在性是review常识
- Review rule: 所有通过GET_FUNC动态加载的CANN API，调用前必须有对应的IsExist检查

### D-215: aclgraph模式下taskQueue关闭时NPUGraph析构crash
- Hashes: 409ddefdd [+9 cherry-picks: e449a1218, 7ae88b2b0, bd20aa3e2,
  0961d04f2, 99bd2e1db, 82e8ff48d, 200b6e925, 4a0406549]
- Root cause: 析构时device资源已释放但NPUGraph仍尝试调用ACL API(与D-65同族)
- Files: torch_npu/csrc/core/npu/NPUGraph.cpp
- Defect: 关闭taskQueue(`TASK_QUEUE_ENABLE=0`)且启用aclgraph时，NPU shutdown流程
  先释放device资源(aclrtResetDevice)，然后NPUGraph的析构函数调用`reset()`，
  其中`AclmdlRIDestroy(model_ri_)`需要有效的device context。device已reset后调用该API
  导致ACL返回错误或crash。
- Fix: 在`reset()`中添加`NpuSysCtrl::GetInstance().GetInitFlag()`检查:
  仅当device仍处于初始化状态时才调用`AclmdlRIDestroy`; 否则跳过(资源由driver回收)。
- Reviewability: medium -- 需要理解NPU shutdown时各组件的析构顺序
- Review rule: 析构函数中调用硬件API前必须检查runtime是否仍然有效

### D-216: schema.json中残留已废弃的profiler_trace API声明
- Hashes: e0938e9c3
- Root cause: API被移除后schema注册未同步清理
- Files: test/torch_npu_schema.json
- Defect: `profiler_trace`接口已从代码中移除，但其schema声明仍残留在
  `torch_npu_schema.json`中。这导致API兼容性测试误认为该接口仍然有效，
  可能产生误导性的测试结果。
- Fix: 从schema.json中删除`torch_npu.dynamo.torchair.scope.profiler_trace`条目。
- Reviewability: high -- API移除时应同步更新schema文件
- Review rule: 删除/重命名API时必须grep schema.json确认无残留

### D-217: storage.cpu()对非NPU非CPU设备的storage处理缺失
- Hashes: 9cc603764
- Root cause: monkey-patch的_cpu()方法只区分CPU和非CPU，未考虑第三方设备
- Files: torch_npu/utils/storage.py,
  test/unsupported_test_cases/.pytorch-disabled-tests.json
- Defect: `_cpu(self)`方法使用`self.device.type != 'cpu'`判断，对所有非CPU storage
  都尝试调用`torch_npu._C._tensor_construct_from_storage(self)`。但meta device、
  XLA等第三方设备的storage不能通过NPU特有的C++函数构造tensor，导致crash。
  同时test_storage_meta_errors系列测试因此被错误禁用。
- Fix: 将条件改为`self.device.type == 'npu'`(仅NPU走NPU路径)，
  新增`elif self.device.type != "cpu"`分支使用标准PyTorch的`UntypedStorage.copy_`方法;
  移除12个test_storage_meta_errors的禁用条目(现在可以通过)。
- Reviewability: high -- 设备类型判断应用正向匹配(== 'npu')而非反向排除(!= 'cpu')
- Review rule: monkey-patch中的设备类型判断必须使用正向匹配，不要用反向排除

### D-218: ge_error_codes.h缺少stdint.h显式include
- Hashes: db175875b
- Root cause: C++隐式头文件依赖(与D-171同族)
- Files: third_party/acl/inc/ge/ge_error_codes.h
- Defect: `ge_error_codes.h`使用了`uint32_t`等stdint类型但未显式`#include <stdint.h>`，
  依赖其他头文件间接包含。当include顺序变化(如用户直接include此头文件)时，
  编译报错"uint32_t undeclared"。GitHub issue#1445报告此问题。
- Fix: 添加`#include <stdint.h>`。
- Reviewability: high -- 头文件应自包含(self-contained)
- Review rule: 每个头文件必须显式include所有直接使用的类型定义

### D-219: aoe测试用例在非Ascend910A芯片上不适用
- Hashes: 50364683a [+8 cherry-picks: f1b7c18f7, 095cdc49e, d457c8c18,
  3a78063d7, 182016ccf, 834cf1b9f, 13755e4e0, 0fd75f68c]
- Root cause: 测试用例缺少SoC兼容性守卫(与D-90/D-148同族)
- Files: test/npu/test_aoe.py
- Defect: AOE(Auto Operator Engine)测试使用conv2d算子，但该算子在非910A芯片上
  已切换为aclnn路径，不再经过aclop，因此不会产生AOE dump图。测试在这些芯片上
  必然失败(预期有dump文件但实际没有)。PR描述明确说明原因。
- Fix: 为TestAoe类添加`@SupportedDevices(['Ascend910A'])`装饰器，在非910A设备上跳过。
- Reviewability: high -- 测试用例应标注硬件依赖
- Review rule: 测试用例依赖特定算子路径(aclop vs aclnn)时必须添加SoC守卫

### D-220: AclInterface.h缺少struct前向声明导致编译失败
- Hashes: 313a0cf35 [+1 cherry-pick: e3b16aade]
- Root cause: C++隐式头文件依赖(与D-171/D-218同族)
- Files: torch_npu/csrc/core/npu/interface/AclInterface.h,
  test/_inductor/test_npu_dtype_cast.py, test/_inductor/test_opensora_graph1.py
- Defect: `AclInterface.h`使用了`aclrtMemUsageInfo`和`aclOpExecutor`类型但未前向声明，
  依赖`acl.h`间接提供。当CANN版本中这些类型定义位置变化或include顺序调整时，
  编译报错"incomplete type"。同时inductor测试因编译失败需临时skip。
- Fix: 在AclInterface.h中添加`typedef struct aclrtMemUsageInfo aclrtMemUsageInfo;`
  和`typedef struct aclOpExecutor aclOpExecutor;`前向声明;
  临时skip两个依赖shmem/特定环境的inductor测试。
- Reviewability: high -- 头文件应显式声明所有使用的类型
- Review rule: 头文件中使用的外部类型必须有显式前向声明或include

### D-221: NPUException.h使用memset但未include cstring
- Hashes: f685919a2
- Root cause: C++隐式头文件依赖(与D-171/D-218/D-220同族)
- Files: torch_npu/csrc/core/npu/NPUException.h
- Defect: `NPUException.h`使用`memset`函数但未包含`<cstring>`头文件。
  在某些编译环境/include顺序下，`<string>`不间接提供`memset`声明，
  导致编译报错"memset not found"。
- Fix: 添加`#include <cstring>`。
- Reviewability: high -- 头文件应自包含
- Review rule: 使用C标准库函数(memset/memcpy等)必须显式include对应C++头文件

### D-222: checkHcclComms中plog打印std::string未调用c_str()
- Hashes: 43c2a628e
- Root cause: printf-style格式化传递std::string对象而非C字符串
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `checkHcclComms()`中`ASCEND_LOGE`使用`%s`格式化符打印comm名称，
  但传入的`name`是`std::string`类型。`%s`期望`const char*`，直接传`std::string`
  对象会导致未定义行为(读取对象内存布局而非字符串内容)，打印出乱码或crash。
- Fix: 将`name`改为`name.c_str()`。
- Reviewability: high -- printf-style格式化+std::string是经典bug模式
- Review rule: ASCEND_LOG系列宏中%s参数必须是const char*，std::string必须用.c_str()

### D-223: profiler初始化后未使用即析构时stop报错
- Hashes: 003752f5f [+5 cherry-picks: 29ec3eb7d, d1a0c1b1b, 1961e6bfe,
  d53591376, 1ce96b899]
- Root cause: profiler stop未检查是否实际启动过(状态机前置条件缺失)
- Files: torch_npu/csrc/profiler/npu_profiler.cpp,
  torch_npu/profiler/analysis/prof_common_func/_path_manager.py
- Defect: 用户创建`torch.profiler.profile()`对象但未调用`__enter__`/start就直接
  析构时，析构函数调用`stopNpuProfiler()`。此时profiler state未初始化(为nullptr)，
  后续操作访问null state导致异常。同时`get_realpath()`在path为空时未提前检查，
  走到`os.path.expanduser("")`返回空值后续操作也报错。
- Fix: 在`stopNpuProfiler()`开头添加`profilerEnabled()`检查，未启动则打印警告并return;
  在`get_realpath()`开头添加空path检查，为空则抛出明确的RuntimeError。
- Reviewability: high -- stop/cleanup函数必须检查对应的start是否执行过
- Review rule: 资源清理函数(stop/close/finalize)必须校验对应的初始化是否完成

### D-224: 动态profiling异常配置刷屏日志

- Hashes: 7644f42b5 [+8 cherry-picks: c262d3be1, 0bc187fab, e4834758d,
  fa9394588, 4d809214a, b13c1a535, 8dffceb94, b4867a3db]
- Root cause: 状态机终态未清理关联上下文(cfg_ctx残留导致每个step重复触发)
- Files: torch_npu/profiler/dynamic_profile.py
- Defect: `_DynamicProfile`中当profiling配置无效(start_step不匹配当前step)时，
  打印警告并设状态为IDLE，但`cfg_ctx`未置None。下一个step仍检查到非空cfg_ctx，
  再次进入异常分支打印相同警告。长训练任务中产生大量重复日志。
- Fix: 在检测到配置异常后添加`self.cfg_ctx = None`，阻断后续重复触发。
- Reviewability: high -- 状态机转IDLE后残余字段是基本检查项
- Review rule: 状态机转移到终态(IDLE/ERROR)时，所有关联上下文必须清零

### D-225: ranktable P2P通信域使用local rank而非global rank

- Hashes: 00572e020
- Root cause: ranktable场景下P2P rank映射缺失
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `createHCCLCommEx`中P2P通信域创建使用进程组内local rank作为HCCL的peer rank。
  在ranktable配置下(`global_ranks_in_group`非空)，HCCL需要全局rank ID才能正确
  定位对端设备。local rank与全局rank不一致时，P2P操作指向错误设备。
- Fix: 检查`global_ranks_in_group`是否为空：非空时通过映射表转换local→global rank，
  空时保持原始逻辑。同时添加bound check。
- Reviewability: high -- ranktable模式的rank语义有明确文档
- Review rule: HCCL通信域创建使用rank时，区分local rank和global rank

### D-226: allreduce测试子进程提前退出导致结果验证竞争

- Hashes: b059af929
- Root cause: 多进程测试中子进程退出时序与主进程结果消费不同步
- Files: test/distributed/test_allreduce.py
- Defect: `_test_all_reduce`子进程完成collective操作并`c2p.put()`后立即退出。
  主进程可能尚未从queue中读取所有结果。子进程先退出导致进程组destroy，
  queue底层共享内存可能被回收，主进程读取时出现竞争条件。
- Fix: 添加`multiprocessing.Event`同步：子进程put完结果后`done_event.wait()`，
  主进程验证完所有结果后`done_event.set()`，确保退出有序。
- Reviewability: medium -- 多进程退出时序需要仔细推演
- Review rule: 多进程测试中子进程退出前必须等待主进程确认结果已消费

### D-227: dynamic shape场景host侧tiling内存每次launch泄漏

- Hashes: 00774a137
- Root cause: 函数内部malloc的host buffer在函数返回后指针丢失
- Files: torch_npu/_inductor/ascend_npu_ir/.../cpp_common.cpp, .h, cpp_wrapper.py
- Defect: `common_launch_dyn()`内部每次调用都`aclrtMallocHost()`分配tiling buffer，
  函数返回后局部指针丢失，内存永不释放。dynamic shape场景每个iteration触发launch，
  泄漏量与训练步数线性增长。
- Fix: 将tiling host buffer的分配提升到调用方，通过新增`arg_tiling_host`参数传入。
  函数签名、Python wrapper codegen模板(`py_args_format`)同步更新。
- Reviewability: high -- 函数内malloc无对应free是经典泄漏模式
- Review rule: 热路径函数内不应循环分配内存；需要分配时提升到外层管理生命周期

### D-228: NPUGuardImpl缺少elapsedTime/queryStream/synchronizeStream实现

- Hashes: 45a7d16a7
- Root cause: DeviceGuardImplInterface虚方法未全部实现+event flag映射遗漏
- Files: torch_npu/csrc/core/npu/impl/NPUGuardImpl.cpp, .h
- Defect: 两个问题叠加：
  1. `elapsedTime()`, `queryStream()`, `synchronizeStream()`未实现，用户调用
     `torch.npu.Event.elapsed_time()`等API时失败
  2. `createEvent()`忽略flag参数：始终用ACL_EVENT_SYNC，不响应
     `BACKEND_DEFAULT`(计时标志)，导致event无法记录时间戳
- Fix: 1. 实现三个缺失方法 2. flag映射：BACKEND_DEFAULT→ACL_EVENT_TIME_LINE|ACL_EVENT_SYNC
  3. `record()`中复用`createEvent()`消除重复逻辑
- Reviewability: medium -- 需对照PyTorch DeviceGuard接口checklist
- Review rule: 实现设备后端GuardImpl时，逐一对照upstream CUDA实现检查所有virtual方法

### D-229: test_torch_mlir模块级代码无条件执行导致import失败

- Hashes: 2cac358b2
- Root cause: 测试文件顶层副作用代码(env var设置+import)在无依赖环境崩溃
- Files: test/_inductor/test_torch_mlir.py
- Defect: 模块顶层`os.environ['TORCHINDUCTOR_MAX_AUTOTUNE']='1'`和
  `from torch_npu._inductor...utils import logger`在import时无条件执行。
  没有torch-mlir依赖的环境下直接报错。
- Fix: 将env var设置移到测试方法内部，添加`@skip("request torch-mlir")`跳过标记。
- Reviewability: high -- 模块顶层副作用是测试反模式
- Review rule: 测试文件顶层不应有副作用代码(env var修改、设备初始化等)

### D-230: upstream ForwardAD wrapped_number传播函数缺失

- Hashes: 77bc22cc2
- Root cause: upstream PyTorch ForwardAD bugfix未同步到NPU自定义autograd
- Files: torch_npu/csrc/framework/autograd/FunctionsManual.cpp, .h, requirements.txt
- Defect: upstream修复了forward-mode AD处理wrapped number(标量提升为tensor)的bug，
  需要在autograd manual function中增加`update_wrapped_number()`传播属性。
  NPU侧缺少该函数，scalar-tensor混合运算的forward AD结果错误。
- Fix: 实现`update_wrapped_number()`：检查input的`is_wrapped_number()`，
  为true则在output上`set_wrapped_number(true)`。同步更新torch/torchvision版本依赖。
- Reviewability: low -- 需理解forward-mode AD的wrapped number语义
- Review rule: TORCH MAIN SYNC类型修改需理解upstream变更motivation

### D-231: C++变量声明同时用未初始化的自身赋值(UB)

- Hashes: 3867857f4
- Root cause: `Type x = x.method()`形式的自引用初始化(未定义行为)
- Files: torch_npu/csrc/distributed/reducer.cpp
- Defect: `at::TensorOptions options = options.dtype(at::kInt)` -- 声明`options`
  的同一语句中调用了`options.dtype()`。此时`options`尚未构造完成，`.dtype()`
  读取未初始化内存，属于C++未定义行为。该模式在文件中出现3处
  (`initialize_local_used_map`, `sync_bucket_indices`, `verify_params_across_processes`)。
- Fix: 拆分为两条语句：`TensorOptions options; options = options.dtype(kInt);`
- Reviewability: high -- 编译器可通过`-Winit-self`检测
- Review rule: 禁止`Type x = x.method()`形式的声明

### D-232: getDeviceFromPtr忽略入参直接返回current device

- Hashes: 1341fbe93
- Root cause: stub实现返回current_device()而非查询指针所属设备
- Files: torch_npu/csrc/core/npu/NPUHooksInterface.cpp, AclInterface.cpp/.h
- Defect: `NPUHooksInterface::getDeviceFromPtr(void* data)`直接返回
  `c10_npu::current_device()`，完全忽略`data`指针。多卡场景下tensor在device 0分配，
  当前线程切到device 1后，查询该tensor设备返回错误的device 1。
- Fix: 引入`aclrtPointerGetAttributes()` API查询指针实际归属设备。添加ACL API
  声明、动态加载wrapper、host指针检查。
- Reviewability: medium -- 函数签名有参数却不使用是明显warning
- Review rule: 设备查询接口不应依赖"当前设备"上下文状态

### D-233: skipIfUnsupportMultiNPU使用return而非raise跳过测试

- Hashes: c018e741b
- Root cause: unittest.SkipTest对象被return而非raise
- Files: torch_npu/testing/common_distributed.py, test/distributed/test_distributed.py
- Defect: `skipIfUnsupportMultiNPU`装饰器中`return unittest.SkipTest(...)`：return
  使测试函数返回一个SkipTest对象(被忽略)，测试框架认为通过而非跳过。同时多进程测试
  中子进程的SkipTest异常未传播到父进程的exit code。
- Fix: `return`改`raise`。MultiProcessTestCase中添加SkipTest异常的单独处理：
  捕获后以multi-npu skip exit code退出。
- Reviewability: high -- return vs raise SkipTest是基础知识
- Review rule: unittest跳过只能通过raise实现，return不触发skip标记

### D-234: CSAN stream check未区分view op和factory function的读写语义

- Hashes: 4db49e496
- Root cause: NPU sanitizer移植CUDA sanitizer时未传递is_factory参数
- Files: torch_npu/npu/_stream_check.py, test/test_npu_sanitizer.py
- Defect: NPU的stream check sanitizer直接复用CUDA sanitizer的`ArgumentHandler`，
  但`parse_inputs()`和`parse_outputs()`未传递`schema`和`is_factory`参数。
  factory函数(new_*/\*_like)不读input数据(只读metadata)，view操作(split等)不产生
  实际读写。未区分这些语义会导致误报data race。
- Fix: 添加`FACTORY_FUNCTION_REGEX`匹配factory函数名，parse_inputs/outputs
  传递`schema`和`is_factory`参数。添加全面的NPU sanitizer UT。
- Reviewability: medium -- 需理解CUDA Sanitizer的读写语义模型
- Review rule: 移植upstream工具到NPU时，对照原始实现检查所有参数是否传递完整

### D-235: ranktable P2P通信域条件排除+comm名称冲突

- Hashes: 05b154336
- Root cause: P2P分支入口条件排除了ranktable场景+comm name无group区分
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: 两个问题：
  1. P2P分支条件`commType==P2P && global_ranks_in_group.empty()`使ranktable场景
     (global_ranks非空)无法进入P2P路径
  2. comm名称`"p2p_X_Y"`无group_id前缀，多进程组间的P2P comm name冲突
- Fix: 移除`empty()`条件限制；comm名称改为`"group"+group_id+"_p2p_X_Y"`。
- Reviewability: high -- P2P条件应与所有comm创建模式兼容
- Review rule: comm name必须包含group标识以防跨组冲突(与D-225构成同区域连续修复)

### D-236: npu_kernel_features循环依赖

- Hashes: be7f213a8
- Root cause: NumelList类中的静态方法引入循环import或无限递归
- Files: torch_npu/_inductor/codegen/npu_kernel_features.py,
  test/_inductor/test_npu_kernel_features.py
- Defect: `NumelList.calc_numels()`静态方法在处理Iterable类型时递归调用自身，
  但NumelList本身继承Tuple(是Iterable)，形成无限递归。同时该方法可能引入
  循环import路径(npu_kernel_features → 依赖模块 → npu_kernel_features)。
- Fix: 移除`calc_numels()`静态方法(11行)，依赖`numels()`实例方法。添加完整UT。
- Reviewability: medium -- Python循环import需工具辅助检测
- Review rule: 模块间import关系应为DAG，添加import时检查是否引入环

### D-237: aclgraph update测试softmax_lse shape不匹配+精度参数名错误

- Hashes: cb9a6dbde
- Root cause: 两个独立问题：output shape条件化缺失 + 自定义断言参数名变更
- Files: test/npu/test_aclgraph_update.py, torch_npu/npu/graphs.py
- Defect: 1. `_GraphDispatchMode`中`softmax_lse`始终创建为`empty(1,...)`，但只有
  传入`softmax_lse_flag`时才需要非空lse；未传时应为`empty(0,...)`
  2. 测试断言用`prec=0.001`参数名，但`assertRtolEqual`的精度参数已改名为`prec16`，
  旧参数名被忽略，实际使用默认精度。
- Fix: 1. 根据`kwargs.keys()`中是否有`softmax_lse_flag`条件化lse shape
  2. `prec=0.001`改为`prec16=0.01` 3. 移除误加的`@unittest.skip`恢复测试
- Reviewability: high -- 参数名拼写错误应通过kwargs检查或type hint防护
- Review rule: 自定义assertion的参数变更时grep所有调用点

### D-238: profiler CANN parser路径校验未try-except

- Hashes: 73a46b9f4 [+5 cherry-picks: 52e1fb617, 2f18354a6, a61af8585,
  9501617aa, fa4250ebb]
- Root cause: CANN profiling路径不存在时直接抛异常导致profiler初始化失败
- Files: torch_npu/profiler/analysis/prof_parse/_cann_file_parser.py,
  torch_npu/profiler/experimental_config.py
- Defect: `CANNFileParser.__init__`中`_check_cann_path_valid()`在CANN路径
  不存在/不可读时直接抛异常，导致整个profiler初始化失败。某些配置(Level_none)下
  CANN profiling数据确实可能不存在，应优雅降级而非crash。
- Fix: 将path检查包裹在try-except中，异常时打印error并继续。添加空path提前检查。
  ExperimentalConfig中Level_none+无mstx/msprof_tx时自动启用mstx。
- Reviewability: medium -- profiler降级策略需要设计
- Review rule: profiler/监控工具初始化不应因可选数据源不存在而导致整体失败

### D-239: schema.json残留已删除的torchair API签名

- Hashes: ccfc91ee3
- Root cause: torchair侧API移除后schema.json未同步清理(与D-216同族)
- Files: test/torch_npu_schema.json, third_party/torchair/torchair
- Defect: torchair删除了`CompilerConfig._get_func_code_md5()`方法，但schema.json
  中保留了两处该方法的签名记录。API兼容性测试会误报"API被移除"。
- Fix: 从schema.json中删除`_get_func_code_md5`条目，更新torchair submodule。
- Reviewability: high -- schema.json更新应与API变更同步
- Review rule: 第三方子模块API变更时必须同步torch_npu_schema.json

### D-240: PyTorch 2.8.0+中DTensorTestBase.with_comms不再设置self.device

- Hashes: aeefb263e
- Root cause: upstream测试基类行为变更导致NPU测试AttributeError
- Files: 20+个distributed/_tensor测试文件, torch_npu/testing/_internal/common_dtensor.py
- Defect: PyTorch 2.8.0+重构了DTensorTestBase，`with_comms`装饰器不再在setUp阶段
  设置`self.device`属性。NPU侧测试依赖该属性获取设备类型，导致AttributeError。
  同时装饰器顺序有问题：`@skipIfUnsupportMultiNPU`应在`@with_comms`外层。
- Fix: 创建`NPUDTensorTestBase`替代upstream基类，确保`self.device`正确设置。
  将20+测试文件迁移到新基类。调整装饰器顺序(skip在外)。
- Reviewability: medium -- upstream基类变更波及面广但难追踪
- Review rule: 不直接继承upstream测试基类，用wrapper隔离upstream变更

### D-241: 进程输出解码未指定错误处理模式+C++循环越界

- Hashes: 192d0fe7b
- Root cause: 混合缺陷 -- `bytes.decode()`缺少errors参数 + C++循环边界`<=`越界
- Files: ci/access_control_test.py, test/distributed/test_fault_mode.py, test/trans_contiguous/*.py, torch_npu/csrc/framework/contiguous/select_opt.cpp
- Defect: `subprocess.Popen`使用`text=True`但未指定`errors`，遇到非UTF-8字节时抛出
  `UnicodeDecodeError`。`select_opt.cpp`中循环条件`i <= select_size.size()`存在越界访问。
  多处测试中过时的`contiguous_d_StridedSlice`期望未更新为`contiguous_d_AsStrided`。
- Fix: `Popen`补充`errors='ignore'`参数，循环改为`i < select_size.size()`，
  更新测试期望值。
- Reviewability: medium -- 多种不同类型缺陷混在一次提交中
- Review rule: C++循环遍历`size_t`索引时边界条件须严格使用`<`而非`<=`

### D-242: 分布式UT因缺少随机数种子初始化导致结果不一致

- Hashes: 737202b33
- Root cause: 测试环境假设 -- 未显式设置RNG seed导致跨进程随机状态不一致
- Files: test/distributed/test_distributed.py
- Defect: `MultiProcessTestCase._run`中fork出的子进程未初始化随机数种子，导致分布式
  测试中各rank产生不同的随机张量，数值对比失败。模块级全局变量在import时即被实例化，
  在fork之前创建的网络权重与子进程中期望的初始化不一致。
- Fix: 在`_run`入口调用`common_utils.set_rng_seed(0)`显式初始化种子，
  将全局网络实例改为在测试方法内部本地创建。
- Reviewability: medium -- 并发/多进程测试中seed问题不直观
- Review rule: 多进程测试用例在每个子进程入口必须显式调用`set_rng_seed`

### D-243: C++中`new`未检查返回值及`stoi`异常未捕获

- Hashes: 83f6bf8f7
- Root cause: 安全编码缺陷 -- `new`抛出异常未处理，`stoi`对非法输入无防御
- Files: torch_npu/csrc/core/npu/GetAffinityCPUInfo.cpp, torch_npu/csrc/core/npu/NPUCachingAllocator.cpp, torch_npu/csrc/core/npu/NPUException.h, torch_npu/csrc/aten/common/TensorProperties.cpp
- Defect: `NPUCachingAllocator.cpp`中`new ExpandableSegment`若分配失败默认抛出
  `std::bad_alloc`，在NPU内存分配路径下无法被上层正确捕获。`GetAffinityCPUInfo.cpp`
  中`stoi`对格式错误输入抛出`std::invalid_argument`，同样缺少防御。
- Fix: 将`new`替换为`new (std::nothrow)`并补充空指针检查；
  将`stoi`替换为`strtol`并增加`endptr`、`errno`及范围校验。
- Reviewability: high -- `new`和`stoi`的不安全用法可通过静态分析检测
- Review rule: 生产路径禁止裸`new`，必须`new (std::nothrow)`+检查；禁止`stoi`解析外部输入

### D-244: torch_npu_schema.json缺少新增算子签名条目

- Hashes: 4f470aca4
- Root cause: Schema文件维护遗漏 -- 新增API未同步更新契约文件(与D-216/D-239同族)
- Files: test/torch_npu_schema.json
- Defect: `torch_npu.dynamo.torchair.scope.op_never_timeout`和
  `torch_npu.dynamo.torchair.scope.profiler_trace`两个新增接口未添加schema条目，
  导致接口契约测试失败。
- Fix: 在`torch_npu_schema.json`中补充两个接口的签名定义。
- Reviewability: high -- 纯配置文件遗漏
- Review rule: 新增torch_npu公开API时PR checklist必须包含更新schema.json

### D-245: 多机SHMEM场景缺少NPU专用的`enable_symm_mem_for_group`实现

- Hashes: 19461e12e
- Root cause: 功能缺失 -- 多机下NPU未覆盖PyTorch默认的对称内存初始化路径
- Files: torch_npu/__init__.py, torch_npu/csrc/distributed/symm_mem/NPUSHMEMExtension.cpp, torch_npu/distributed/_symmetric_memory/__init__.py
- Defect: 多机环境下调用`enable_symm_mem_for_group`会走PyTorch默认实现，其store
  构建逻辑不适配NPU的SHMEM初始化路径，导致多机通信失败。C++日志格式符`%d`对
  `uint64_t`类型不匹配，存在未定义行为。
- Fix: 新增NPU专用的`_enable_symm_mem_for_group`并在`__init__.py`中patch到
  `torch.distributed._symmetric_memory`；修正日志格式符。
- Reviewability: medium -- 多机路径需环境复现
- Review rule: NPU分布式扩展点必须在__init__.py中集中patch注册

### D-246: warnings.warn格式字符串参数位置错误导致TypeError

- Hashes: 7e45549fa
- Root cause: API误用 -- `warnings.warn`第二参数被解释为category而非格式参数
- Files: torch_npu/testing/__init__.py
- Defect: `warnings.warn("... '%s' ...", filename)`中第二个位置参数会被解释为
  `category`(期望Warning子类)，传入字符串路径引发`TypeError`。
- Fix: 改为f-string格式化: `warnings.warn(f"...{filename}...")`。
- Reviewability: high -- 明显API误用，可通过lint规则检测
- Review rule: `warnings.warn`的message必须是完整字符串，不得用`%s`占位符

### D-247: SHMEM对称内存大小环境变量解析仅支持裸字节数

- Hashes: fde7a65ab
- Root cause: 功能缺陷 -- 环境变量解析不支持带单位的人类可读格式
- Files: torch_npu/csrc/core/npu/register/OptionsManager.cpp, OptionsManager.h, torch_npu/csrc/distributed/symm_mem/NPUSHMEMExtension.cpp
- Defect: 原实现用`strtol`直接将`NPU_SHMEM_SYMMETRIC_SIZE`解析为整数字节，
  无法接受`1G`/`512M`等带单位格式。
- Fix: 新增`GetShmemSymmetricSize()`，用`sscanf`解析`<number><unit>`格式，
  支持K/M/G/T单位。
- Reviewability: medium -- 解析逻辑边界需仔细验证
- Review rule: 环境变量涉及数值单位转换时必须封装独立函数并覆盖边界测试

### D-248: aclgraph中PA/MLA算子更新导致TensorWeakRef对CPU tensor崩溃

- Hashes: 38eb0a0a6
- Root cause: 类型假设错误 -- `args`存为tuple不可修改 + TensorWeakRef不支持非NPU tensor
- Files: torch_npu/npu/graphs.py, test/npu/test_aclgraph_update.py
- Defect: `_append_dispatch_record`将`args`存为`tuple`，但`update_capture_record`
  需按位置替换，tuple不支持下标赋值。args中CPU tensor也被包装为`TensorWeakRef`导致崩溃。
  正则无法匹配`Tensor(a!)`形式的输出参数名称。
- Fix: 将`args_ref`从tuple改为list；对tensor增加device类型判断只包装NPU tensor；
  修正正则表达式。
- Reviewability: medium -- bug分散在解析、存储、更新三个环节
- Review rule: 需就地修改的序列不得使用tuple存储；TensorWeakRef使用前必须校验device类型

### D-249: IPC reduce时meta tensor缺少rebuild路径导致序列化失败

- Hashes: 5189313b5
- Root cause: 路径覆盖缺失 -- `_npu_reduce_tensor`未处理device类型为meta的tensor
- Files: torch_npu/multiprocessing/reductions.py
- Defect: `_npu_reduce_tensor`按设备类型分发序列化路径时，只覆盖NPU实际设备，
  storage device为`meta`时无对应分支，走到依赖实际storage地址的逻辑抛异常。
- Fix: 增加`meta`分支，返回`(rebuild_meta_tensor, (type, size, stride, ...))`元组。
- Reviewability: high -- 缺失分支逻辑清晰，diff极小
- Review rule: 扩展reduce/rebuild路径时必须显式枚举meta/cpu/npu所有device类型分支

### D-250: ONNX算子API签名变更未同步导致onnx export失败

- Hashes: 12a6dcea6
- Root cause: API契约漂移 -- 算子Python接口改名/新增参数后ONNX symbolic未同步
- Files: torch_npu/onnx/wrapper_onnx_ops.py, test/torch_npu_schema.json, test/unsupported_test_cases/.pytorch-disabled-tests.json
- Defect: `npu_rotary_mul`参数从`(x, r1, r2)`变更为`(input, r1, r2, rotary_mode='half')`，
  但ONNX symbolic签名未更新。`npu_group_norm_silu`的None参数生成空tensor常量，
  ONNX推理引擎无法处理。
- Fix: 更新symbolic函数签名加入`rotary_mode`参数；None参数改为根据shape推断的占位tensor。
- Reviewability: medium -- ONNX symbolic路径需了解graph builder语义
- Review rule: 修改算子Python签名时必须同步更新onnx symbolic函数

### D-251: aclgraph重建tensor时未初始化NPU storage descriptor

- Hashes: fa1d403fa
- Root cause: NPU扩展元数据缺失 -- 通用tensor创建路径跳过NPU私有描述符初始化
- Files: torch_npu/csrc/npu/Module.cpp
- Defect: aclgraph恢复tensor时通过make_tensor_base创建后仅调用set_sizes_and_strides，
  未同步设置NPU侧的StorageDesc，导致npu_desc为空，后续算子访问描述符时崩溃。
- Fix: 在set_sizes_and_strides之前插入StorageDescHelper::SetDesc调用。
- Reviewability: high -- 单文件4行插入，对照NPU tensor正常创建路径即可发现
- Review rule: NPU tensor非标准创建路径必须审查是否补充StorageDescHelper::SetDesc

### D-252: inductor动态shape下tiling device buffer每次调用重新分配导致内存泄漏

- Hashes: beccd204a
- Root cause: 动态shape内核闭包内per-call内存分配无对应释放
- Files: torch_npu/_inductor/ascend_npu_ir/(多个文件), torch_npu/_inductor/npu_static_kernel.py
- Defect: common_launch_dyn内部对每次调用都执行aclrtMalloc分配arg_tiling_device，
  但没有对应的aclrtFree，动态shape场景下持续泄漏device内存。
- Fix: 将arg_tiling_device分配提升到编译期(创建持久torch.empty NPU tensor)，
  作为闭包捕获变量传入，函数内不再自行分配。
- Reviewability: medium -- 根因清晰但修复跨10个文件
- Review rule: 包含aclrtMalloc的函数必须在同一作用域确认存在aclrtFree

### D-253: aclrtMemcpyAsyncWithCondition在旧驱动上被错误探测为可用

- Hashes: d886740a3
- Root cause: 功能探测缺少前置驱动版本保护(与D-63/D-79/D-214同族)
- Files: torch_npu/csrc/core/npu/interface/AclInterface.cpp
- Defect: AclrtMemcpyAsyncWithConditionExist只靠GET_FUNC动态查找符号判断可用性，
  旧版本驱动(<25.0.RC1)中符号存在但行为异常，导致走入错误路径。
- Fix: 在符号探测前插入IsGteDriverVersion("25.0.RC1")检查。
- Reviewability: high -- 单函数5行改动
- Review rule: 所有依赖特定驱动版本API的功能探测，必须同时加版本guard

### D-254: torch.amp.autocast的cuda重定向未覆盖新调用路径

- Hashes: 94eb3aebe
- Root cause: API覆盖不完整 -- torch.amp.autocast独立入口被遗漏
- Files: torch_npu/contrib/transfer_to_npu.py, test/contrib/test_transfer_to_npu.py
- Defect: transfer_to_npu将torch.autocast(旧路径)列入白名单，但torch.amp.autocast
  使用独立的autocast_mode.autocast.__init__入口不经过同一wrapper，
  导致`torch.amp.autocast('cuda')`时设备参数未替换为npu。
- Fix: 改为直接对torch.amp.autocast_mode.autocast.__init__应用_wrapper_cuda包装。
- Reviewability: high -- 改动仅2行核心逻辑
- Review rule: PyTorch版本升级后需重新审查transfer_to_npu的wrapper列表

### D-255: watchdog恢复流程中日志级别错误且变量引用未更新

- Hashes: 01442bd42
- Root cause: 日志语义错误 + copy-paste变量引用错误
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: device error/FORCE STOP属于预期恢复场景却用logger->error记录，污染告警。
  workCleanupLoop中日志传入device_error_msg(赋值前旧值)而非当次捕获的device_error。
  workEnqueue异常信息硬编码"UCE ERROR."丢失实际内容。
- Fix: 恢复场景日志降级为info；修正日志参数为device_error；
  workEnqueue拼入实际device_error_msg再清空。
- Reviewability: medium -- 变量引用错误需仔细对比赋值顺序
- Review rule: catch块内格式化日志时必须核查参数引用的是更新前还是更新后的值

### D-256: transfer_to_npu中PipelineStage使用_device_wrapper无法重定向设备参数

- Hashes: d1deab965
- Root cause: wrapper策略选择错误 -- _device_wrapper不适用于需拦截构造函数参数的类
- Files: torch_npu/contrib/transfer_to_npu.py
- Defect: 对PipelineStage使用_device_wrapper批量包装，但PipelineStage.__init__接收
  device参数，_device_wrapper无法拦截构造时传入的cuda字符串，pipeline并行设备仍指向cuda。
- Fix: 改为对PipelineStage.__init__应用_wrapper_cuda包装。
- Reviewability: high -- 单文件3行，与同文件DeviceMesh.__init__处理方式一致
- Review rule: transfer_to_npu中有显式device构造参数的类必须包装__init__

### D-257: aclop + deterministic组合使用时发生死锁/冻结

- Hashes: a78f7121c
- Root cause: 全局初始化与条件调用执行顺序错误 -- aclop路径外执行了aclop专属初始化
- Files: torch_npu/csrc/framework/OpCommand.cpp, torch_npu/csrc/framework/OpParamMaker.cpp
- Defect: LazyInitAclops和AclSetCompileopt(JIT_COMPILE)放在Run()的非条件位置，
  即便走opapi路径也会执行。启用deterministic时SetDeterministicOption与g_used_aclop
  标志出现竞态，特定顺序下触发死锁。
- Fix: 将LazyInitAclops移入aclop专属分支；SetDeterministicOption增加g_used_aclop前置条件。
- Reviewability: medium -- 死锁场景需理解aclop/opapi并发模型
- Review rule: aclop专属初始化必须限定在aclop分支内，不得污染opapi路径

### D-258: WatchdogStatus成员变量未初始化导致未定义行为

- Hashes: c3b1e3712
- Root cause: 枚举类型成员变量缺少初始化
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.hpp
- Defect: watchdogStatus声明为WatchdogStatus枚举但未给初值，编译器不保证零初始化，
  watchdog线程启动时读到的状态可能是任意值，恢复逻辑误判。
- Fix: 改为`WatchdogStatus watchdogStatus = WatchdogStatus::RUN`显式初始化。
- Reviewability: high -- 单行改动，典型"声明但未初始化"模式
- Review rule: 所有枚举/类类型非静态成员变量必须在声明处赋初值

### D-259: NPUEvent/NPUStream错误使用WITHOUT_UCE宏掩盖UCE错误

- Hashes: 103b3ecc0
- Root cause: 错误处理策略不一致 -- UCE场景下的错误被特殊宏静默
- Files: torch_npu/csrc/core/npu/NPUEvent.cpp, NPUStream.cpp, torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: NPUEvent析构/query/synchronize等路径使用NPU_CHECK_ERROR_WITHOUT_UCE，
  UCE发生时不抛异常，错误在事件层被静默，延误故障传播。
- Fix: 全部替换为NPU_CHECK_ERROR，ProcessGroupHCCL增加对应异常捕获和UCE状态处理。
- Reviewability: medium -- 机械替换但需理解UCE recovery整体架构
- Review rule: WITHOUT_UCE宏只应用于明确文档化的"UCE下允许继续"场景

### D-260: _rebuild_npu_tensor中多余设备迁移操作引入回归(rollback)

- Hashes: 303cff222
- Root cause: 反序列化路径中多余的设备迁移操作破坏storage上下文
- Files: torch_npu/utils/storage.py, test/test_npu_multinpu.py
- Defect: 之前的"修复"在_rebuild_npu_tensor中将storage.cpu()保存后再.to(device)移回NPU，
  中间步骤破坏storage原始设备上下文，多卡场景device编号可能不一致。_reduce_ex中序列化时
  错误用self(NPU tensor)代替self.cpu()。
- Fix: 回滚该补丁: _rebuild_npu_tensor去掉storage.cpu()直接用.npu()；
  _reduce_ex中改回`tmp_tensor = self.cpu()`。
- Reviewability: medium -- 需理解NPU storage序列化/反序列化完整往返路径
- Review rule: 序列化路径修改必须同时验证serialize→deserialize往返正确性及多卡场景

### D-261: DeviceMesh UT未适配upstream API变更

- Hashes: e4a6ae5f8
- Root cause: 测试代码未随upstream API签名扩展同步更新
- Files: test/distributed/test_device_mesh.py
- Defect: upstream DeviceMesh API扩展(如`from_group`新增参数)后，
  测试用例仍使用旧的调用方式，导致分布式UT失败。
- Fix: 更新测试用例中的API调用以匹配新签名。
- Reviewability: high -- 测试编译/运行即可发现
- Review rule: upstream版本升级时需系统性检查分布式测试的API兼容性

### D-262: 误删re模块import导致正则功能缺失(revert)

- Hashes: d7b0d4d9a
- Root cause: 代码清理时误删仍在使用的import(REVERT)
- Files: torch_npu相关Python文件
- Defect: 前序commit清理"无用"import时误删了`import re`，
  后续代码中re.match/re.sub等调用抛出NameError。
- Fix: Revert误删操作，恢复import re。
- Reviewability: high -- import删除应通过工具验证无引用
- Review rule: 删除import前必须用工具确认全部引用已消除

### D-263: inductor NPU IR适配PyTorch v2.7.1 API变更

- Hashes: b885e9763
- Root cause: 跨版本API变更未同步(与D-96/D-126同族)
- Files: torch_npu/_inductor/codegen/相关文件
- Defect: v2.7.1重构了inductor多个内部接口(as_strided参数、View节点方法等)，
  NPU侧IR代码仍调用旧接口导致codegen阶段崩溃。
- Fix: 逐一适配新接口签名和调用方式。
- Reviewability: low -- 跨多文件的接口适配，需熟悉upstream变更
- Review rule: upstream版本升级时系统性检查_inductor目录的API兼容性

### D-264: _npu_dtype_cast_backward缺少DTensor sharding策略注册

- Hashes: 2b3fecd70
- Root cause: op注册遗漏(与D-60/D-94/D-210同族)
- Files: torch_npu/distributed/tensor/_matrix_ops.py
- Defect: `_npu_dtype_cast_backward`未注册DTensor sharding策略，
  分布式训练反向传播时DTensor dispatch找不到该op的分片规则。
- Fix: 注册pointwise sharding策略。
- Reviewability: medium -- 需对照op注册checklist
- Review rule: 新增NPU自定义op时必须同步注册DTensor sharding策略

### D-265: transfer_to_npu中DeviceMesh的monkey-patch策略错误

- Hashes: 767a7d031
- Root cause: _device_wrapper替换整个类破坏isinstance语义(与D-268同类)
- Files: torch_npu/contrib/transfer_to_npu.py
- Defect: 对DeviceMesh使用_device_wrapper整体替换类为函数，导致isinstance检查失败。
- Fix: 改为对DeviceMesh.__init__应用_wrapper_device包装，保留类对象身份。
- Reviewability: high -- 与同文件D-256/D-268的修复模式一致
- Review rule: transfer_to_npu中有class构造器device参数的类必须包装__init__

### D-266: Python 3.11+兼容性问题导致import失败

- Hashes: ee1870167
- Root cause: Python版本间模块路径或API差异
- Files: torch_npu相关import路径
- Defect: 部分import依赖Python 3.10以下版本的模块布局，在3.11+中路径变化导致
  ImportError。
- Fix: 更新为版本兼容的import方式或添加条件import。
- Reviewability: high -- CI覆盖多Python版本即可发现
- Review rule: CI矩阵需覆盖Python 3.10/3.11/3.12

### D-267: ProcessGroupHCCL中record_stream的storage强引用导致内存无法释放(revert)

- Hashes: b27e5f091
- Root cause: 前序优化引入的storage record_stream持有强引用阻止释放
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: 前序commit让collective操作后通过storage调用record_stream/erase_stream
  来追踪tensor生命周期，但持有storage强引用阻止了GC回收，长时间训练中积累内存泄漏。
- Fix: Revert该变更，恢复不追踪storage的旧逻辑。
- Reviewability: medium -- 内存泄漏需长时间运行才能观察
- Review rule: record_stream/erase_stream修改必须验证不会阻止storage GC回收

### D-268: NPU Event适配torch Event接口变更

- Hashes: cf5eec50d
- Root cause: upstream torch.Event接口变更后NPU侧未同步
- Files: torch_npu/csrc相关Event实现
- Defect: upstream PyTorch修改了Event的构造或同步接口，NPU Event覆写方法签名
  不匹配导致运行时多态调用失败。
- Fix: 更新NPU Event实现以匹配新的upstream接口。
- Reviewability: medium -- 需对照upstream Event类接口变更
- Review rule: upstream Event/Stream接口变更时必须同步NPU实现

### D-269: float精度测试因随机种子缺失导致flaky

- Hashes: ed39d74c1
- Root cause: 测试用例缺少随机种子固定
- Files: test/相关测试文件
- Defect: float16/float32精度对比测试未固定random seed，不同运行产生不同随机数据，
  在精度边界处时通时不通。
- Fix: 添加固定seed或调整tolerance阈值。
- Reviewability: high -- flaky test pattern明确
- Review rule: 涉及精度对比的测试必须固定随机种子

### D-270: ACL_PRECISION_MODE用户配置被系统默认值覆盖

- Hashes: 83260b4f8
- Root cause: 配置优先级逻辑错误(Write-after-Read缺失条件判断)
- Files: torch_npu/csrc/core/npu/register/OptionsManager.cpp
- Defect: 设置ACL_PRECISION_MODE时，代码先计算SoC默认值再无条件写入，
  覆盖了用户通过环境变量预设的值。
- Fix: 在写入前检查用户是否已显式设置，仅在未设置时才使用默认值。
- Reviewability: medium -- 配置覆盖顺序需审查
- Review rule: 配置系统中用户显式设置的优先级必须高于默认值

### D-271: Triton codegen中错误使用aclop接口

- Hashes: 1bc675ec3
- Root cause: codegen路径中混用不同后端API
- Files: torch_npu/_inductor/codegen/triton.py
- Defect: Triton codegen模板中错误调用了aclop特有的接口或使用了不兼容的
  参数形式，导致特定pattern下生成的代码运行报错。
- Fix: 修正codegen模板中的API调用方式使其与Triton后端兼容。
- Reviewability: medium -- 需了解Triton和aclop两套接口差异
- Review rule: codegen模板中调用的后端接口必须与当前编译路径匹配

### D-272: IFA测试中softmax输出校验UT参数不匹配

- Hashes: 82f01d06d
- Root cause: 测试Oracle参数不正确
- Files: test/npu/test_aclgraph_*.py
- Defect: IFA(Incre Flash Attention)测试用例中softmax lse参数配置
  与实际算子行为不一致，导致测试结果不稳定。
- Fix: 修正测试参数使其与算子实际输出语义匹配。
- Reviewability: high -- 测试参数修复，diff清晰
- Review rule: 算子UT的输出校验参数须与算子spec严格对齐


### D-273: GetAclOpInitMode返回未校验的临时局部变量而非成员变量

- Hashes: 4de195620
- Root cause: 变量名拼写错误(局部变量遮蔽成员变量)
- Files: torch_npu/csrc/core/npu/register/OptionsManager.cpp
- Defect: `GetAclOpInitMode`校验范围后将合法值写入成员变量`acl_op_init_mode_`(带下划线)，
  但return语句返回的是未经校验的临时变量`acl_op_init_mode`(无下划线)，
  导致范围校验和默认值重置逻辑完全失效。
- Fix: 返回语句改为`return static_cast<uint32_t>(acl_op_init_mode_)`。
- Reviewability: high -- 单字符差异(尾部下划线)，静态分析可检测
- Review rule: 成员变量与局部变量命名相近时，return语句必须返回经过全部校验的最终变量

### D-274: aclgraph update测试中softmax_lse预分配tensor尺寸语义错误

- Hashes: d131c9a99
- Root cause: 测试fixture中输出tensor形状与算子约定不符
- Files: test/npu/test_aclgraph_update.py
- Defect: 测试用例将`softmax_lse`预分配为`torch.empty(1, ...)`，
  而`softmax_lse_flag=False`时算子不产生有效输出，size=1的tensor
  导致graph capture形状约束冲突。
- Fix: 将预分配从`torch.empty(1,...)`改为`torch.empty(0,...)`匹配空输出语义，
  删除对应的assertEqual断言。
- Reviewability: medium -- 需了解算子在不同flag下的输出协议
- Review rule: .out接口测试必须注释各输出在不同flag下的预期尺寸

### D-275: IPC路径中ptr()方法在两个成员均为空时缺少返回值

- Hashes: 1b8151cc6
- Root cause: 控制流缺失(函数无return兜底路径)
- Files: torch_npu/csrc/core/npu/NPUCachingAllocator.cpp
- Defect: `ptr()`方法中`if(npu_ipc_ptr_) ... else ...`在npu_ipc_ptr_为空时
  无条件解引用expandable_segment_，若后者也为空则空指针解引用。
- Fix: 将else改为独立if判断，末尾显式`return nullptr`。
- Reviewability: high -- 典型空指针缺陷，开启`-Wreturn-type`即可编译期检测
- Review rule: 返回指针的方法必须在所有控制路径上有明确return语句

### D-276: sanitizer的apply_sanitizer_patch在全局init时无条件执行

- Hashes: f2e280ea6
- Root cause: 初始化副作用范围过宽(与D-140/D-229同族)
- Files: torch_npu/__init__.py, torch_npu/npu/_sanitizer.py
- Defect: `apply_sanitizer_patch()`在`__init__.py`顶层无条件执行，
  早于`TORCH_NPU_SANITIZER`环境变量检查，导致stream检测相关monkey-patch
  对所有用户生效。
- Fix: 移到`enable_npu_sanitizer()`内部，仅当用户启用时才调用。
- Reviewability: medium -- 需理解Python模块加载顺序和monkey-patch副作用
- Review rule: 含monkey-patch的函数不得在模块顶层无条件调用

### D-277: 序列化时NPU tensor设备信息在CPU迁移过程中丢失

- Hashes: ba4812b79
- Root cause: 序列化/反序列化路径设备信息传递断链(与D-48/D-192/D-260同族)
- Files: torch_npu/utils/storage.py, test/test_npu_multinpu.py
- Defect: `_reduce_ex`执行`tmp_tensor = self.cpu()`后，storage变为CPU storage，
  `_rebuild_npu_tensor`收到的设备信息为CPU，重建tensor被放置在CPU上，
  后续`.npu()`调用变为隐式迁移，多卡场景设备号丢失。
- Fix: 序列化端保留NPU storage直接序列化，重建端先记录原始device再迁移。
- Reviewability: medium -- 需成对审查_reduce_ex和_rebuild_*
- Review rule: tensor序列化修改必须验证device信息全程保留

### D-278: inductor Triton codegen中broadcast_to/reshape语义混淆

- Hashes: 7ee5d115d
- Root cause: Triton IR生成逻辑中shape变换原语选用错误
- Files: torch_npu/_inductor/codegen/triton.py, test/_inductor/test_pattern_44.py
- Defect: reduction前的shape提升使用`tl.broadcast_to`，但当reduction输入来自
  非连续索引时broadcast_to的维度兼容性约束不满足，导致运行时错误。
- Fix: 将`tl.broadcast_to`改为`tl.reshape`，新增从load/store索引反推变量顺序的逻辑。
- Reviewability: low -- 涉及Triton IR语义和NPU tiling axis模型交叉
- Review rule: Triton codegen中shape变换原语变更必须附带reduction+非连续索引测试

### D-279: SilentCheck中isinstance检查对象与实际操作对象不匹配

- Hashes: 733b24f4f
- Root cause: copy-paste变量引用错误
- Files: torch_npu/asd/asd.py
- Defect: `isinstance(self.statistic_value, DTensor)`检查的是statistic_value，
  但分支内对`grad`调用`.to_local()`。当grad是DTensor而statistic_value不是时，
  条件为False，torch.norm(grad)在DTensor上执行失败。
- Fix: 改为`isinstance(grad, DTensor)`使检查对象与操作目标一致。
- Reviewability: high -- 单行diff，典型copy-paste缺陷
- Review rule: isinstance检查的对象必须与紧跟其后的类型特化操作的对象一致

### D-280: SilentCheck inference mode tensor不可变导致fill_()失败

- Hashes: 14fdbd02f
- Root cause: inference mode tensor只读 + DTensor未解包
- Files: torch_npu/asd/asd.py
- Defect: `_MatmulSilentCheck`的`statistic_value`/`statistic_cpu_value`在`torch.inference_mode()`下创建，
  后续非inference mode下对其调用`fill_()`失败（inference tensor是read-only）。
  同时DTensor包装的statistic_value未调用`to_local()`，导致norm计算结果也是DTensor，fill_()语义不匹配。
- Fix: 非inference模式下`clone()`使tensor可变；DTensor先`to_local()`再操作；
  fill_()参数侧也增加DTensor→local转换。
- Reviewability: medium -- inference mode语义需运行时触发，但DTensor类型检查可静态发现
- Review rule: 对长生命周期的累积tensor，创建时应确保其不受inference_mode/grad_mode等上下文影响

### D-281: profiler step trace time中对象属性访问与实际数据结构不匹配

- Hashes: 8e0d9916b [+4 cherry-picks: d6547dd0c, 9c4289630, 10465763c, 7c94f8f7b]
- Root cause: 对象属性访问(.ts)与列表索引访问混淆
- Files: torch_npu/profiler/analysis/prof_view/prof_db_parse/_trace_step_time_db_parser.py
- Defect: `get_prepare_time()`中`first_fwk_op`通过`min(self.torch_op_data, key=...)`获取，
  返回的是torch_op_data中的列表/元组元素。代码用`first_fwk_op.ts`属性访问但实际应用
  `first_fwk_op[TorchOpDataOri.START_NS]`索引访问。DB场景下数据结构不同于对象模式。
- Fix: `first_fwk_op.ts` → `first_fwk_op[TorchOpDataOri.START_NS]`
- Reviewability: high -- 单行修改，数据结构类型与访问方式的不匹配在review时可发现
- Review rule: 当同一数据有对象模式和列表/DB模式两种表示时，访问方式必须与当前模式匹配

### D-282: HF32默认值逻辑硬编码，忽略用户环境变量设置

- Hashes: 128d7e3d0
- Root cause: 默认值硬编码替代环境变量读取
- Files: torch_npu/csrc/framework/LazyInitAclops.cpp
- Defect: `SetHF32DefaultValue()`直接硬编码`allow_hf32 = "10"`（conv=1启用, matmul=0禁用），
  用户通过`ALLOW_MATMUL_HF32`和`ALLOW_CONV_HF32`选项设置的值被完全忽略。
  ACL_ALLOW_HF32是两位字符串("XY"，X=conv, Y=matmul)，需分别从选项读取。
- Fix: 分别读取ALLOW_MATMUL_HF32(默认"0")和ALLOW_CONV_HF32(默认"1")选项，
  根据enable/disable设置对应位，拼接为`conv_hf32 + mm_hf32`。
- Reviewability: high -- 硬编码默认值覆盖用户配置是明显的逻辑遗漏
- Review rule: 涉及用户可配置选项的默认值设置，必须先查询用户配置再fallback到默认值

### D-283: 动态profiler子进程logger路径未传递 + CPU-only缺STEP_TIME表

- Hashes: 7a0e21491
- Root cause: 子进程初始化参数传递链断裂 + CPU-only模式路径遗漏
- Files: torch_npu/profiler/_dynamic_profiler/_dynamic_profiler_monitor.py,
  _dynamic_profiler_utils.py, prof_config/_parser_config.py
- Defect: (1) `worker_func`调用`init_logger(is_monitor_process=True)`但不传log_path，
  `init_logger`内部依赖类变量`CFG_CONFIG_PATH`在worker子进程中未设置，
  导致日志路径为None→`os.path.join(None, 'log')`报错。
  (2) STEP_TIME表解析在仅启用CPU activity时仍尝试创建NPU相关表，报错。
- Fix: `init_logger`增加`log_path`参数，worker调用时从`cfg_path`推导传入；
  解析配置增加CPU-only路径支持。
- Reviewability: medium -- 子进程参数传递遗漏需在多进程场景测试才暴露
- Review rule: 跨进程传递的初始化函数不能依赖父进程的类变量/全局状态

### D-284: PythonTracer singleton在子线程首次访问时GIL死锁

- Hashes: 1b04447a8 [+4 cherry-picks: 76f43ebf6, 0daba0e58, fd25bbd40, 12f8d82cd]
- Root cause: C++ static local初始化需要GIL但子线程可能无GIL
- Files: torch_npu/csrc/profiler/profiler_python.cpp
- Defect: `PythonTracer::singleton()`使用C++11 static local变量（线程安全初始化），
  构造函数内`pybind11::gil_scoped_acquire`获取GIL。子线程调用`call(kStartOne/kStopOne)`
  直接调用`singleton()`，如果singleton尚未在主线程创建，子线程触发构造→请求GIL→
  若主线程持有GIL等待该子线程→死锁。
- Fix: 新增`instance_created_` atomic bool标志和`get_singleton_in_child_thread()`方法，
  子线程先检查标志，singleton未创建则跳过(返回nullptr)。
- Reviewability: medium -- 涉及C++ static init + Python GIL交互，需多线程profiling场景触发
- Review rule: C++ singleton的构造函数若需要GIL，必须确保首次访问在主线程完成

### D-285: dump_fx_graph测试与fx_graph_cache冲突 + graph hash缺版本维度

- Hashes: 8b5662bcf [+1 cherry-pick: e53f69457]
- Root cause: 功能互斥性未识别 + hash维度不足
- Files: test/_inductor/test_check_accuracy.py, test_force_fallback.py,
  torch_npu/_inductor/codegen/scheduling.py, test/_inductor/test_lowering_fx.py(新)
- Defect: (1) dump_fx_graph功能依赖每次编译重新trace生成graph，但fx_graph_cache开启时
  直接复用缓存跳过编译，导致dump功能失效。
  (2) traced_graph hash仅包含graph内容不含torch版本，不同PyTorch版本对同一模型可能
  生成结构相同但语义不同的IR，hash碰撞导致错误复用。
- Fix: 测试文件设置`fx_graph_cache = False`；hash追加`torch.__version__`。
- Reviewability: high -- fx_graph_cache与dump功能的互斥关系在功能设计阶段应识别
- Review rule: 依赖"每次编译都执行"语义的功能必须显式声明与缓存机制的互斥约束

### D-286: FSDP _root_pre_forward未将host输入移动到NPU设备

- Hashes: f687bdc28 [+1 cherry-pick: dddf43d3d]
- Root cause: upstream设备类型白名单缺少"npu"
- Files: torch_npu/distributed/fsdp/_add_fsdp_patch.py
- Defect: upstream FSDP的`_root_pre_forward`对cuda/hpu/xpu/mtia设备类型会调用`_to_kwargs`
  将输入tensor移到设备上（同DDP行为），但设备类型检查列表中没有"npu"。
  使用FSDP2训练时host上的输入tensor不会被自动移到NPU，导致计算时设备不匹配。
- Fix: 新增`_patched_root_pre_forward`方法，设备类型列表增加"npu"，monkey-patch到FSDPState。
- Reviewability: high -- 设备类型白名单遗漏是NPU适配的标准问题
- Review rule: upstream代码中所有设备类型白名单（if device.type in [...]）必须包含"npu"

### D-287: Triton codegen的tiling逻辑不支持symbolic shape(dynamic shape)

- Hashes: 604964eaf
- Root cause: 符号表达式(sympy.Symbol)未解析为具体值即参与算术/比较运算
- Files: torch_npu/_inductor/codegen/split_tiling.py, triton.py, wrapper.py
- Defect: `SplitTiling`中`x.length`在dynamic shape下是sympy符号而非整数，
  `total_split_numels()`直接`reduce(lambda x,y: x*y, [x.length for x in ...])`对符号求积
  产生不可求值的表达式。`is_1d_reduction()`用`self.numels["r"] > 1`直接比较符号表达式。
  `initialize_range_tree()`中将numels符号替换为具体值的位置过早，影响后续symbolic推断。
- Fix: 新增`get_length_val()`通过`var_to_val`映射解析符号；
  `is_1d_reduction`改用`statically_known_gt`；删除过早的numels替换；
  `CantSplit()`改为`CantSplit`(异常类不需要实例化)。
- Reviewability: low -- 需要理解sympy/inductor symbolic shape框架，pure static review难发现
- Review rule: codegen中涉及range_tree length的运算必须通过`get_length_val()`/`statically_known_*`处理

### D-288: HCCL NSLB-DP配置项赋值到错误的结构体成员(copy-paste)

- Hashes: dbe7a336c [+5 cherry-picks: a6e4bbbbc, 557b7b9b7, 393637b23, 55ab952c7, a76cc2bb2]
- Root cause: copy-paste后未修改目标字段名
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `createHcclCommConfigWithOptions()`从options字典读取"hccl_world_rank_id"和"hccl_job_id"
  的值后，两处都赋值给了`config.hcclOpExpansionMode`而非各自的正确字段。
  结果：hcclWorldRankID和hcclJobID始终为默认值，hcclOpExpansionMode被覆盖两次。
  NSLB(Network-level Scheduling and Load Balancing)和DP(Data Parallelism)功能因配置错误而失效。
- Fix: `config.hcclOpExpansionMode` → `config.hcclWorldRankID`（第一处）和`config.hcclJobID`（第二处）。
- Reviewability: high -- 两段结构相同的代码中左侧字段名相同，典型copy-paste未改
- Review rule: 结构相似的赋值代码块需逐字段核对目标变量名，特别是dict key与struct member的对应关系

### D-289: NPU Triton后端sin/cos精度不足，临时移除lowering

- Hashes: a94661424
- Root cause: NPU Triton kernel对三角函数的实现精度不满足要求
- Files: torch_npu/_inductor/lowering_op_list.py
- Defect: `aten.sin`和`aten.cos`加入GENERATE_LIST后，inductor会生成Triton kernel处理这些op。
  但NPU Triton后端的三角函数实现精度或功能存在问题（commit标注"temporary"移除）。
  sin/cos走Triton路径会产生精度错误的结果。
- Fix: 从GENERATE_LIST移除，使sin/cos fallback到CANN算子（精度正确但性能可能较低）。
- Reviewability: low -- 精度问题需要数值测试才能发现，代码review无法判断
- Review rule: 新op加入Triton lowering列表前必须通过精度对比测试(vs CPU/CANN参考实现)

### D-290: D2H拷贝路径升级aclrtMemcpyAsyncWithCondition + SoC版本守卫

- Hashes: 8bb075835 [+5 cherry-picks: a9a107dcf, 4faba86c5, 2d23bdcaa, 87ab6514c, c8b6a6003]
- Root cause: API可用性检查缺少SoC版本维度 + D2H拷贝语义优化缺失
- Files: torch_npu/csrc/core/npu/interface/AclInterface.cpp, torch_npu/csrc/framework/OpParamMaker.cpp,
  third_party/acl/inc/acl/acl_rt.h
- Defect: (1) `AclrtMemcpyAsyncWithConditionExist()`仅检查函数指针非空，未检查SoC版本。
  低版本SoC(< 910B1)不支持该API但函数可能存在(动态库版本更新)。
  (2) D2H拷贝统一用`aclrtMemcpyAsync`，非ACL分配的host内存（如malloc）执行异步拷贝时
  runtime内部仍需同步等待，浪费流水线并行度。`WithCondition`版本对非ACL内存自动变为同步拷贝。
- Fix: API可用性增加`GetSocVersion() >= Ascend910B1`守卫；
  D2H拷贝路径增加`WithCondition`分支，并分别记录不同的错误日志。
- Reviewability: medium -- API升级是功能性变更，SoC守卫是D-63/D-79/D-214的复现模式
- Review rule: 硬件API可用性检查必须同时包含函数存在性和SoC版本范围(与D-63/D-79/D-214同类)

### D-291: profiler解析器配置查询链缺防护 + CPU-only活动路径遗漏

- Hashes: 256c01976 [+5 cherry-picks: 6a949ae01, fca6ef028, 6b7387c42, 238135598, 198982e4a]
- Root cause: dict链式get()无默认值 + 活动类型检查缺失 + 资源引用未清空
- Files: torch_npu/profiler/analysis/_profiling_parser.py, prof_common_func/_constant.py,
  _db_manager.py, prof_parse/_fwk_file_parser.py, prof_view/_memory_prepare_parser.py
- Defect: 多个问题叠加:
  (1) `parser_config.get(export_type).get(self._analysis_type)`链式调用，
  export_type不存在时`.get()`返回None，None无get方法→AttributeError。
  (2) `get_torch_op_tree_node()`/`get_fwk_trace_data()`/`get_fwk_api()`不检查CPU_ACTIVITIES，
  纯CPU profiling时仍尝试解析NPU相关数据。
  (3) `BasicDb.close()`置空db_path和销毁conn，但不清空curs引用，后续访问curs可能操作已关闭cursor。
  (4) `MemoryPrepareParser`同时向默认DB和配置DB写入相同数据，缺少early return。
- Fix: dict.get()增加`{}`和`[]`默认值；3个函数入口增加`CPU_ACTIVITIES not in activities`检查；
  close()同时清空conn和curs；增加early return避免重复写入。
- Reviewability: high -- dict防护和early return是基础防御性编程
- Review rule: 链式dict.get()必须每级提供默认值；解析函数入口必须检查活动类型前置条件

### D-292: test_fake_tensor.py缺少npu_grouped_matmul的group_type参数

- Hashes: 79ecfa94d [+5 cherry-picks: e35fc4764, 222a984c0, dbe0f0bf7, 716340ef9, 1b258ef0f]
- Root cause: op接口新增参数后测试用例未同步更新
- Files: test/npu/test_cann_version.py, test/test_fake_tensor.py
- Defect: `npu_grouped_matmul`接口新增了`group_type`参数，4处FakeTensor测试调用未传该参数，
  默认值语义不符合测试意图。同时CANN/driver版本正则缺少行尾锚定`$`，
  `8.1.RC1`能错误匹配到`8.1.RC12`。
- Fix: 4处调用增加`group_type`参数(-1或0)；版本正则增加`$`锚定和`re.IGNORECASE`；
  新增`test_get_driver_version()`测试。
- Reviewability: high -- 接口变更后grep调用点即可发现
- Review rule: op接口新增参数时，必须同步搜索并更新所有测试中的调用(包括FakeTensor测试)

### D-293: _npu_dtype_cast缺少DTensor sharding策略注册

- Hashes: 8ae87c5e8 [+3 cherry-picks: d3423e588, 0a6c135b7, be4f75b7a]
- Root cause: op变体的DTensor策略注册遗漏
- Files: torch_npu/utils/dtensor.py
- Defect: `npu_dtype_cast.default`已在pointwise_ops中注册DTensor sharding策略，
  但下划线前缀变体`_npu_dtype_cast.default`未注册。DTensor环境下调用`_npu_dtype_cast`
  因无sharding策略而报错。两个op语义相同（dtype转换），应有相同的sharding策略（pointwise）。
- Fix: 添加`npu._npu_dtype_cast.default`到pointwise_ops注册列表。
- Reviewability: high -- 单行遗漏，与D-60/D-94/D-210/D-264同类模式
- Review rule: 新增NPU自定义op的所有变体(.default, _前缀等)必须同步注册DTensor策略

### D-294: taskqueue GIL释放引发死锁 + aclrtSetCurrentContext导致设备不一致

- Hashes: 2a478c985 [+11 cherry-picks: eef1d5ae6, 31c6152c6, ef59fd4b5, 76913c983,
  34f04920f, 1fbb06aa1, 3f63621bc, 8bb6c26ad, 593697137, 174650f13, ff523b7cb,
  b744cca34(+更多版本)]
- Root cause: GIL释放条件过宽引发allocator/GC死锁 + context切换的device副作用未恢复
- Files: torch_npu/csrc/core/npu/NPUCachingAllocator.cpp, NPUQueue.cpp,
  torch_npu/csrc/framework/OpCommand.cpp, OpCommand.h
- Defect: 两个独立缺陷:
  (1) NPUQueue队列满时释放GIL让TE编译线程工作。但线程A持allocator锁→队列满→释放GIL→
  线程B获GIL→触发GC→析构tensor→请求allocator锁→死锁。
  (2) `insert_events()`调用`aclrtSetCurrentContext`切换到block所在设备，
  切换后不恢复原device，TLS缓存的device_id与实际不一致，后续操作到错误设备。
- Fix: (1) 引入`g_used_aclop` atomic flag，仅aclop路径释放GIL（aclop将废弃，新路径无需此hack）。
  (2) 保存pre_device，context切换后SetDevice恢复。
- Reviewability: low -- GIL/allocator/GC三方交互的死锁极难静态分析
- Review rule: 释放GIL时必须证明不会有其他线程获GIL后请求当前线程持有的锁

### D-295: inductor decomposition overload列表已失效 + erfc公式错误

- Hashes: e75616e0b
- Root cause: 历史代码未随upstream演进清理 + 数学公式错误
- Files: torch_npu/_inductor/decomposition.py, lowering.py, lowering_op_list.py
- Defect: 三个问题:
  (1) `DECOMPOSITION_OVERLOAD_OP`列表中的ops在当前upstream版本已不在decompositions dict中，
  `del decompositions[op]`操作无效（key不存在会KeyError，被try/except吞掉）。
  (2) `erfc(x)`实现为`1 - torch.exp(x)`，正确公式是`erfc(x) = 1 - erf(x)`。
  exp和erf完全不同的函数，精度完全错误。
  (3) `make_fallback`在FALLBACK_LIST构建完成前被调用，时序错误。
- Fix: 删除DECOMPOSITION_OVERLOAD_OP机制；`torch.exp(x)` → `torch.erf(x)`；
  make_fallback移到for循环之后；`aten.gather`加入静态FALLBACK_LIST。
- Reviewability: high -- erfc公式是高中数学知识，exp(x)和erf(x)差异显著
- Review rule: 数学函数的decomposition实现必须附带公式注释和数值对比测试

### D-296: 进程析构时static unordered_map已销毁导致logger core dump

- Hashes: d3f87cd3b [+5 cherry-picks: c4866cac6, e5e270539, 5054b5702, 34e9aac21, fe195b067]
- Root cause: C++ static局部变量析构顺序不确定(SIOF变体)
- Files: torch_npu/csrc/logging/Logger.cpp, Logger.h
- Defect: `LoggingLevelNames`是static局部`unordered_map<LoggingLevel, string>`，
  进程退出时static变量析构顺序不确定。NPU资源清理代码（如HCCL finalize、allocator释放）
  在各自的static析构函数中调用logger，如果LoggingLevelNames先于这些对象析构，
  `LoggingLevelNames[level]`访问已销毁的map→undefined behavior→core dump。
- Fix: 删除static map，将level字符串直接作为参数传入`log()`函数。
  每个调用点(`debug/info/warn/error/critical`)传递字符串字面量"DEBUG"/"INFO"等。
  字面量的生命周期是整个程序，不受析构顺序影响。
- Reviewability: medium -- 需理解C++ static destruction ordering，但修复模式(消除static依赖)是标准做法
- Review rule: static局部容器不应在可能被析构函数调用的路径上使用(与D-65/D-209/D-215同族)

### D-297: AOTI Triton kernel参数结构体缺少sync_block_lock字段导致DDR越界

- Hashes: f5dcbb8a9
- Root cause: kernel ABI结构体字段缺失导致偏移错位
- Files: torch_npu/_inductor/codegen/cpp_wrapper.py
- Defect: NPU Triton kernel的packed参数结构体定义中缺少`sync_block_lock`指针字段。
  runtime期望该字段在ffts_addr和workspace_addr之间。缺少后所有后续字段的内存偏移量
  向前错位8字节(void*大小)，kernel读取参数时地址全部偏移→DDR地址超出合法范围→硬件错误。
- Fix: 在结构体定义和初始化列表中插入`void* sync_block_lock`字段（初始化为NULL）。
- Reviewability: low -- 需要了解NPU Triton ABI layout，纯代码review难发现字段缺失
- Review rule: kernel参数结构体变更必须与runtime ABI文档同步校验，字段顺序/大小/对齐逐一核对

### D-298: memory_db_parser变量名错误 + 动态profiler配置权限诊断缺失

- Hashes: ad3f9ebcc [+5 cherry-picks: c1b4e4ded, a306d60f3, 81f44fbe4, 8ce5e5906, 4f5bd6d15]
- Root cause: 变量名引用错误(last_record vs last_record_data) + 诊断信息缺失
- Files: torch_npu/profiler/_dynamic_profiler/_dynamic_profiler_monitor_shm.py,
  torch_npu/profiler/analysis/prof_view/prof_db_parse/_memory_db_parser.py,
  torch_npu/profiler/dynamic_profile.py, torch_npu/utils/_path_manager.py
- Defect: (1) `MemoryDbParser`中STREAM_PTR的fallback值引用`last_record`（原始record）
  而非`last_record_data`（累积值），stream指针信息从错误的数据源获取。
  (2) 动态profiler配置文件不可读/不可写时静默失败无任何提示，排障困难。
  (3) `start_step`已错过时无警告，用户不知道profiling未生效。
- Fix: `last_record` → `last_record_data`；增加文件权限检查和诊断打屏；
  增加start_step过期警告。
- Reviewability: high -- 变量名错误(last_record vs last_record_data)可通过review发现
- Review rule: 同一作用域中名称相似的变量(xxx vs xxx_data)使用时必须逐一确认语义

### D-299: D2H非阻塞拷贝时CachingHostAllocator收到错误的base指针

- Hashes: 5112c08df [+5 cherry-picks: 16fb415d7, d558488c5, 410df9e51, 2d200de86, 5b357efea]
- Root cause: view tensor的data_ptr与storage base ptr不一致
- Files: torch_npu/csrc/aten/common/CopyKernel.cpp, InnerNpuNativeFunction.h,
  torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp
- Defect: `copy_between_host_and_device`中对pin_memory tensor的非阻塞D2H拷贝，
  `CachingHostAllocator_recordEvent(ptr, stream)`需要storage的base指针来跟踪内存生命周期。
  原`get_base_data_ptr()`通过`is_view()`→`_base().data_ptr()`获取，
  但嵌套view场景下_base可能仍是view而非storage base，且存在不必要的view链遍历。
- Fix: 改用`storage().mutable_data()`直接获取storage层的base指针；
  删除`get_base_data_ptr`辅助函数（不再需要）。两处调用点(CopyKernel.cpp, CopyKernelOpApi.cpp)同步修改。
- Reviewability: medium -- 需理解view/storage/data_ptr的层次关系
- Review rule: 需要storage base address的场景必须用storage().data()，不能用tensor.data_ptr()或view链

### D-300: P2P连接计数初始值off-by-one + 限制检查不精确

- Hashes: 309c3ccab [+4 cherry-picks: 6b47857fd, 19083f991, 8715f494f, 137d07f48]
- Root cause: 自身连接计入远程连接数 + 限制归因不明确
- Files: torch_npu/csrc/core/npu/NPUPeerToPeerAccess.cpp
- Defect: (1) `device_enabled_count_`初始化为1（包含自身COPY_ALLOWED），但自身→自身
  不是远程P2P连接，不应计入。实际可用远程连接数为MAX_NPUS-1=7而非8。
  (2) 达到连接限制时只检查source_dev不检查dest_dev，如果dest_dev已满限但source_dev未满，
  连接仍被允许→dest_dev实际超过限制。
  (3) 超限时遍历source_dev的所有连接来生成警告，但受限设备可能是dest_dev。
- Fix: 初始化count为0；分别检查source_dev和dest_dev限制，标记`limited_device`；
  遍历受限设备的连接生成准确错误信息；日志级别从WARN升为ERROR。
- Reviewability: medium -- off-by-one需要仔细推理计数语义，但修复后代码逻辑更清晰
- Review rule: 资源计数的初始值必须与"计数什么"的语义一致(远程连接不含自身)


### D-301: StressDetect多worker场景下通信域创建hang + API参数重命名

- Hashes: 482eb0ab9 [+5 cherry-picks: 9584c1632, a373cd329, e213d0494, ba874e36c, ff901d5b7]
- Root cause: 分布式集合通信API误用(new_group的collective语义未被遵守)
- Files: torch_npu/npu/utils.py, torch_npu/csrc/npu/Stress_detect.cpp, test/torch_npu_schema.json
- Defect: `stress_detect(mode=1)`创建HCCL子通信域时，Python层只对当前worker调用了一次
  `torch.distributed.new_group(local_ranks)`。`new_group`是collective操作，要求所有参与
  `torch.distributed`初始化的rank同步调用。多worker(多node)场景下，其他worker的rank没有参与
  `new_group`调用，导致hang。此外`WORLD_SIZE`环境变量未被读取。
- Fix: 遍历所有worker(`for worker_id in range(num_workers)`)，为每个worker的rank子集都调用
  `new_group`，仅保留属于当前worker的group句柄。同时读取`WORLD_SIZE`，将`mode=0/1`改为
  `detect_type='aic'/'hccs'`提升API可读性。
- Reviewability: medium -- 需理解new_group是全局collective语义才能发现bug
- Review rule: 凡调用torch.distributed.new_group的地方，检查是否所有rank都参与了调用

### D-302: profiler增加msprof权限校验(返回值语义变更+错误信息缺修复指引)

- Hashes: d85991f31
- Root cause: 权限检查缺失 + 异常处理策略不当
- Files: torch_npu/profiler/analysis/prof_common_func/_file_manager.py,
  torch_npu/profiler/analysis/prof_common_func/_path_manager.py,
  torch_npu/profiler/analysis/prof_parse/_cann_file_parser.py,
  torch_npu/profiler/analysis/prof_view/cann_parse/_cann_export.py
- Defect: (1) `check_path_permission`权限不匹配时直接`raise PermissionError`，调用方无法做
  非中断处理。(2) `CANNFileParser`初始化时不检查CANN数据目录读写权限，目录不可读/写时报底层
  IO异常。(3) msprof环境校验拆分为三个独立方法，拆分过细且错误信息不一致。
- Fix: `check_path_permission`从抛异常改为返回bool；新增`check_file_writable/readable`；
  `_cann_file_parser.py`初始化时调用`_check_cann_path_valid()`；三个方法合并为
  `_check_msprof_path()`，含`chown`命令修复建议。
- Reviewability: low -- 改动分散4个文件3个层级，需追踪返回值语义变更对所有调用方的影响
- Review rule: 函数的错误处理方式(抛异常vs返回值)变更时，必须检查所有调用方是否适配了新行为

### D-303: profiler内存数据解析bug(activities配置未在解析链中传播)

- Hashes: a82d52ced [+5 cherry-picks: 81f53e9d5, 5eac8683a, 02a3e756f, a2e91d057, 88c1a9abb]
- Root cause: 配置状态传递缺失
- Files: torch_npu/profiler/analysis/_profiler_config.py,
  torch_npu/profiler/analysis/_profiling_parser.py
- Defect: `ProfilerConfig`没有将`activities`列表(从`profiler_info.json`中读取的profiling活动
  类型)作为属性暴露并传递给下游parser。`load_timediff_info`方法每次都重新从JSON中解析
  `activities`而不是复用已load的值。导致内存解析模块无法判断是否包含NPU活动，纯CPU profiling
  场景可能错误地尝试解析NPU内存数据(与D-304密切关联)。
- Fix: `ProfilerConfig`新增`_activities`属性及property；`load_info`中从JSON提取activities
  并存储；`_profiling_parser.py`中将activities通过`param_dict`传递给下游parser。
- Reviewability: high -- 改动小且模式明确：补充缺失的配置传递
- Review rule: profiler配置项从JSON加载后应存储于ProfilerConfig并通过param_dict传递，不应在下游重复解析

### D-304: 单CPU场景PROF文件权限校验异常(无条件解析NPU数据)

- Hashes: 11610b99b [+5 cherry-picks: 090d1a0bf, 8f7309944, 0110981e8, d378e3a85, dfde70a41]
- Root cause: 纯CPU profiling场景下无条件解析NPU数据 + 硬编码字符串
- Files: torch_npu/profiler/analysis/_profiler_config.py,
  torch_npu/profiler/analysis/_profiling_parser.py,
  torch_npu/profiler/analysis/prof_common_func/_constant.py,
  torch_npu/profiler/analysis/prof_view/_memory_view_parser.py
- Defect: (1) `load_timediff_info`用硬编码字符串`"ProfilerActivity.NPU"`判断是否需要获取NPU
  时间差，无统一常量。(2) `_memory_view_parser.py`中`_init_npu_data`无条件调用
  `CANNFileParser`获取NPU内存数据文件列表，纯CPU场景CANN目录不存在导致异常。
  (3) `analyse_profiling_data`没有异常保护，解析失败直接中断整个profiling流程。
- Fix: 将`"ProfilerActivity.NPU"`提取为`Constant.NPU_ACTIVITIES`常量；三处获取NPU数据的
  调用增加`if Constant.NPU_ACTIVITIES in self._activities`前置判断；
  `analyse_profiling_data`加`@no_exception_func()`装饰器。
- Reviewability: medium -- 修复逻辑清晰，但需理解profiler的activities配置传递链
- Review rule: 解析NPU特有数据前必须先检查activities配置中是否包含NPU activity

### D-305: HCCL通信op的streamId获取时机错误(stream分配前取ID)

- Hashes: adc3e0d5a [+1 cherry-pick: 071f0c7f7]
- Root cause: stream ID获取时机过早(在stream实际分配之前)
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: 在ProcessGroupHCCL的所有collective/p2p操作(allreduce、broadcast、reduce、scatter、
  send、recv、alltoall等约20处)中，`getStreamId(false, -1)`在`collective()`调用之前执行，
  然后通过lambda capture传入hccl_call闭包。如果是首次调用，`hcclStreams_`中还没有对应entry，
  `getStreamId`返回`-1`，导致profiler的MstxRange记录到错误的stream ID。
- Fix: 删除所有`getStreamId`的提前调用(约20处)，在lambda闭包内部改为`stream.id()`直接获取。
  `stream`是`collective()`内部创建并传入lambda的NPUStream引用，此时ID必然有效。
- Reviewability: high -- 模式极其统一(全部是getStreamId -> stream.id()的机械替换)
- Review rule: profiler信息(stream ID等)应在实际执行上下文中获取，不应在异步/延迟执行的操作外部提前捕获

### D-306: transfer_to_npu对torch.Generator的cuda->npu适配缺失(白名单不适用于class)

- Hashes: 966667047 [+7 cherry-picks: fb1a11418, 805b2636b, bbef45273, 9216e5dae, 69ada4efc,
  87978c523, 6a81c58d9]
- Root cause: 设备映射hook覆盖不完整(torch.Generator是class而非function)
- Files: torch_npu/contrib/transfer_to_npu.py, test/contrib/test_transfer_to_npu.py
- Defect: `transfer_to_npu`通过白名单对`torch.*`函数做cuda->npu替换。原代码将`"Generator"`放
  在白名单中，试图通过`_device_wrapper`包装。但`torch.Generator`是class，`_device_wrapper`
  对其构造函数的device参数替换不起作用。`torch.Generator('cuda')`无法被正确转换为npu设备。
- Fix: 从白名单中删除`"Generator"`；创建`_GeneratorProxy`类继承`torch.Generator`，重写
  `__new__`方法做device参数替换；在`_init()`中将`torch.Generator`替换为`_GeneratorProxy`。
  新增测试覆盖位置参数、字符串device、torch.device对象、keyword参数等多种构造方式。
- Reviewability: high -- 修复思路清晰，测试覆盖充分
- Review rule: transfer_to_npu白名单只适用于普通函数；class类型需通过子类代理做设备替换

### D-307: dynamic profiler的step()方法遗漏_step_num_offset(D-312的不完整修复)

- Hashes: b9d77b951 [+7 cherry-picks: 445ab9e0c, 7dac0b622, 49994df4c, e79ffb427, 061e2bea0,
  0d0002080, 90fb9f981]
- Root cause: 不完整修复(D-312引入_step_num_offset但遗漏step()路径)
- Files: torch_npu/profiler/profiler.py
- Defect: D-312引入了`_step_num_offset`机制，在`start()`和`step()`两个位置都需要将offset加到
  step_num上。D-312修复了`start()`中的调用，但遗漏了`step()`中的同一处代码。`step()`方法中的
  `ProfilerStep#`标签仍然只用`self.step_num`，没有加上`_step_num_offset`。
- Fix: `step()`中`str(self.step_num)`改为`str(self.step_num + self._step_num_offset)`。
  单行修改，与`start()`中的修法完全对称。
- Reviewability: high -- 纯机械性补全，diff只有1行
- Review rule: 引入新状态变量后，搜索所有读取该变量应该出现的位置确认全部更新

### D-308: transfer_to_npu对torch.distributed group创建的hook点过时

- Hashes: 26c4fa914 [+7 cherry-picks: 24d7f38a3, 9e9af8a7f, de612ffee, e42728a75, 6167bdf40,
  44717c42b, f7ccd9090]
- Root cause: 上游API变更适配遗漏(new_group内部改为调用_new_group_with_tag)
- Files: torch_npu/contrib/transfer_to_npu.py
- Defect: `transfer_to_npu`hook了`torch.distributed.new_group`做backend替换(nccl->hccl)。
  但上游PyTorch重构后，`new_group`内部改为调用`distributed_c10d._new_group_with_tag`，
  直接patch `new_group`无法拦截所有group创建路径(如`init_device_mesh`等内部调用)。
- Fix: hook点从`torch.distributed.new_group`移至更底层的
  `torch.distributed.distributed_c10d._new_group_with_tag`。2行修改。
- Reviewability: medium -- 需理解PyTorch distributed内部调用链才能判断hook点是否正确
- Review rule: monkey-patch的hook点在上游版本升级时应检查是否仍然是唯一入口

### D-309: force_fallback_kernel的多处逻辑缺陷(过滤缺失+路径耦合+功能互斥)

- Hashes: fb2391450
- Root cause: 逻辑缺陷 + 架构问题
- Files: torch_npu/_inductor/npu_triton_heuristics.py,
  test/_inductor/test_check_accuracy.py(新增), test/_inductor/test_force_fallback.py(新增)
- Defect: NPUCachingAutotuner调试模式的多个问题：(1) `should_fallback()`当`fallback_id`为
  list时没有检查当前kernel id是否在列表中，导致所有kernel都fallback。(2) `data_dump()`要求调用
  者显式传入`dump_path`，耦合且易错。(3) `dump_fx_graph`和`force_fallback`以elif互斥，无法
  同时dump又fallback。(4) `fallback_to_fx()`中混入了data_dump职责。
- Fix: 重构为`maybe_run_debug()`方法；`dump_fx_graph`改为独立先执行(不再互斥)；
  `data_dump()`改为自行获取路径(加`lru_cache`)；`should_fallback()`增加kernel_id过滤。
  新增2个测试文件覆盖force_fallback和check_accuracy场景。
- Reviewability: low -- 4个独立问题的混合修复+重构，+179/-23行，原代码无测试覆盖
- Review rule: 调试/诊断功能也需测试覆盖；配置项支持多种类型时每种过滤逻辑都需显式实现和测试

### D-310: inductor meta字典key拼写错误(traced_hash_dir vs traced_graph_dir)

- Hashes: d80a46978
- Root cause: 拼写错误(dict key typo)
- Files: torch_npu/_inductor/npu_triton_heuristics.py
- Defect: `get_fx_graph_call()`中读取inductor_meta字典的key写成了`"traced_hash_dir"`，
  实际存入的key是`"traced_graph_dir"`。导致`dump_dir`永远取到默认值空字符串，data dump和
  fx graph fallback等debug功能全部静默失效。
- Fix: `"traced_hash_dir"` -> `"traced_graph_dir"`。单字符级typo修复。
- Reviewability: high -- 典型typo，但实际检视中"traced_hash_dir"看起来也像合理的key名
- Review rule: 字典key应通过常量定义避免magic string散落多处；debug功能需冒烟测试验证基本路径可达

### D-311: N秒快速恢复场景DEVICE_TASK_ABORT错误信息被通用分支吞没

- Hashes: 82c067ec8 [+7 cherry-picks: 304693a64, d8422a715, e064d92f6, c068895d2, 1cfebdb12,
  55c1c51d4, 6bbef6c02]
- Root cause: 错误分派遗漏(macro中缺少error code专用分支)
- Files: torch_npu/csrc/core/npu/NPUException.h
- Defect: `NPU_CHECK_ERROR`宏对`ACL_ERROR_RT_DEVICE_TASK_ABORT`(107022)没有专门处理分支。
  设备N秒快速恢复场景task abort时，error_code落入通用else分支，用户看到含糊的函数签名而非
  "FORCE STOP"等明确语义，无法快速定位是设备强制停止。
- Fix: 在compact模式分支之后、通用else之前，新增`ACL_ERROR_RT_DEVICE_TASK_ABORT`专用分支，
  输出"FORCE STOP"提示并附带完整函数名、文件名、行号和error code。
- Reviewability: medium -- 宏中新增else-if分支，但需理解分派优先级
- Review rule: 异常处理宏新增error code分支时需检查在所有既有模式(compact/非compact/OOM)中的行为

### D-312: dynamic profiler的step id不反映实际训练步数 + 缺mstx step标记

- Hashes: d08ef6395 [+7 cherry-picks: 42dbcafb4, f7bcce4c7, 9610b96c2, c6a35b898, d43e784a7,
  571218749, 3f19c3d45]
- Root cause: 缺失的状态偏移 + 缺失的可观测性标记
- Files: torch_npu/profiler/profiler.py, torch_npu/profiler/dynamic_profile.py,
  test/profiler/test_npu_profiler.py
- Defect: 两个独立问题：(1) `_DynamicProfile`在训练中途启动profiler时，`step_num`从0开始，
  `ProfilerStep#N`标记与实际训练step不对应。(2) dynamic profiler step推进中没有mstx range
  标记，缺少step粒度timeline可视化。
- Fix: (1) 新增`_step_num_offset`属性，`_DynamicProfile`启动时设offset。`start()`中构造
  `ProfilerStep#`时加offset。(注意：`step()`中遗漏了，由D-307修复。)
  (2) 新增`_step_mstx_range_id`，在profiler启动时`mstx.range_start()`，每次step先end再
  start新range。
- Reviewability: medium -- 逻辑清晰但包含两个独立feature，且step()中遗漏offset(D-307证实)
- Review rule: 一个commit包含两个独立变更时应拆分；影响record标识的状态变量需grep所有构造位置

### D-313: inductor codegen缺失store_cubin配置传递(autotuner硬编码为True)

- Hashes: 78d96069a [+1 cherry-pick: d88345a24]
- Root cause: 配置传递遗漏 / 硬编码默认值
- Files: torch_npu/_inductor/codegen/triton.py,
  torch_npu/_inductor/npu_triton_heuristics.py
- Defect: (1) `NPUIndexTritonKernel`生成`inductor_meta`字典时遗漏`store_cubin`字段。
  (2) `NPUCachingAutotuner`中`launcher.store_cubin`被硬编码为`True`，无视用户配置。
  根因是NPU侧移植上游triton codegen时遗漏了`store_cubin`的配置传递链路。
- Fix: `inductor_meta`中补上`"store_cubin": config.triton.store_cubin`；硬编码改为
  `self.inductor_meta.get("store_cubin", False)`。
- Reviewability: high -- 单行改动，对比上游CachingAutotuner即可发现差异
- Review rule: 移植上游代码时检查所有config字段是否完整传递，搜索硬编码True/False是否应读配置

### D-314: DestroyUsedStreams传入NPUStream触发隐式转换的队列排空副作用

- Hashes: 0c7ba1f7d [+4 cherry-picks: 120619c6a, eef2a97f8, ec9b3740c, 49da77dae]
- Root cause: 隐式类型转换的语义副作用
- Files: torch_npu/csrc/core/npu/NPUFunctions.cpp
- Defect: `DestroyUsedStreams()`调用`acl::AclrtDestroyStreamForce(stream)`，参数类型为
  `NPUStream`而非`aclrtStream`。C++编译器通过`operator aclrtStream()`隐式转换，该转换调用
  无参`stream()`方法，在per-stream-queue模式下会先`MakeSureQueueEmpty()`排空任务队列。
  进程退出时强制销毁stream，再排空队列既不必要又可能在异常退出路径上触发二次错误或卡死。
- Fix: 显式调用`stream.stream(false)`，`false`参数跳过队列排空逻辑。
- Reviewability: low -- 修改只有一字符差异，但需理解NPUStream隐式转换语义和两个stream()重载
- Review rule: 销毁/finalize路径上所有stream访问应使用stream(false)；隐式类型转换不应携带副作用

### D-315: silent check v3重算逻辑三处缺陷(守卫条件错误+共享模块重复注册+self指向错误)

- Hashes: be2158f51 [+4 cherry-picks: e51a48fb1, 03c16ad02, 83718846e, 1999adc48]
- Root cause: 状态管理缺陷(多处)
- Files: torch_npu/asd/asd.py
- Defect: `_MatmulSilentCheck`中三处独立bug：(1) `init_stream()`用
  `self.statistic_value is None`作初始化守卫，但真正需检查的是`statistic_cpu_value`(pinned
  memory tensor)是否已创建。(2) 模块hook注册去重只检查name，共享模块(同一实例多name引用)会被
  重复注册。(3) `self.hook_dict`的`self`指向外层模型实例而非`matmul_check`，装饰器闭包中self
  指向外层对象的经典陷阱。
- Fix: 守卫条件改为`self.statistic_cpu_value is None`；新增`visited_modules_id`用`id(module)`
  去重；`self.hook_dict`改为`matmul_check.hook_dict`。
- Reviewability: medium -- 每个单独不复杂，但分散在不同位置且需理解装饰器内self绑定语义
- Review rule: 初始化守卫必须检查真正需要被初始化的字段；装饰器/闭包中self的指向是否符合预期

### D-316: inductor codegen的rebuild_flattened_dim对expr_substituted无条件覆盖

- Hashes: 4d1bc797f
- Root cause: 字典写入缺少存在性检查
- Files: torch_npu/_inductor/codegen/ir.py
- Defect: `rebuild_flattened_dims`中当检测到axis不重复时，直接执行
  `V.kernel.expr_substituted[expr] = old_node.symbol()`。如果同一`expr`已存在(由之前迭代
  写入)，无条件覆盖会把先前正确的替换关系冲掉。flattened维度codegen中先遍历到的映射才正确
  (对应最内层axis分解)，后来的覆盖导致生成错误的索引计算代码。
- Fix: 加`if expr not in V.kernel.expr_substituted:`守卫，首次写入语义(first-write-wins)。
- Reviewability: high -- 单行条件新增，模式清晰
- Review rule: 循环中对字典赋值时检查是否需要"首次写入"语义；如后续迭代可能覆盖同key须显式决定

### D-317: silent check中tcpstore轮询线程缺少退出条件

- Hashes: 3cc3eace5 [+4 cherry-picks: b73c1d922, c2d0769df, 812b86bcf, a59bf0653]
- Root cause: 并发控制遗漏(循环退出条件不完整)
- Files: torch_npu/asd/asd.py
- Defect: `_MatmulSilentCheck`中两个`while`循环通过TCPStore counter做barrier同步，循环条件
  仅检查`counter < world_size`，没有检查`checksum_state_thread_running`标志。外部要求线程
  停止时，循环仍无限自旋等待，导致进程退出或异常退出时线程永远阻塞。
- Fix: 两个while循环条件中追加`and self.checksum_state_thread_running`。2行修改。
- Reviewability: high -- 改动仅2行，模式完全对称
- Review rule: 分布式同步barrier/轮询循环必须包含超时或外部终止条件

### D-318: codegen对index_expr类型索引的permute和变量重命名处理错误

- Hashes: 51dfb9339
- Root cause: 类型区分遗漏(index_expr与普通load/store索引共用分析路径但语义不同)
- Files: torch_npu/_inductor/codegen/kernel_analysis.py,
  torch_npu/_inductor/codegen/triton.py
- Defect: `IndexAnalysis`没有区分`index_expr`和普通load/store索引：(1) `analyze_permute_shape()`
  对index_expr做了不该做的permute shape分析。(2) `sympy_index_symbol`对index_expr变量做了
  后缀重命名(如`x_0`改`x_1`)破坏原始语义。(3) `NPUTritonKernelOverrides`缺少`index_expr`
  覆写。(4) store cache命中时做了多余的permute分析。
- Fix: `IndexAnalysis`新增`is_index_expr`标志全链路透传；`analyze_permute_shape()`对
  index_expr直接return；变量重命名处index_expr保持原始符号名；新增
  `NPUTritonKernelOverrides.index_expr()`；移除store cache命中时多余的分析。
- Reviewability: low -- 改动横跨codegen核心路径多个层次，需理解Triton codegen的index_expr语义
- Review rule: 新增codegen路径时需检查所有共用分析pass是否需要区分处理

### D-319: __FILENAME__宏未定义导致编译错误(头文件中引用编译选项注入的宏)

- Hashes: 62d92b595 [+4 cherry-picks: 19f58182e, fbf4b63f4, 667683734, 8e4f00360]
- Root cause: 宏作用域不匹配(自定义编译宏仅对特定编译单元定义但在头文件中被引用)
- Files: torch_npu/csrc/core/npu/NPUException.h
- Defect: `NPUException.h`中`NPU_CHECK_WARN`宏使用了`__FILENAME__`，这是通过
  `-D__FILENAME__=...`编译选项定义的自定义宏。但该`-D`只在特定编译单元指定，其他`.cpp`文件
  `#include`此头文件时`__FILENAME__`未定义，导致编译失败。
- Fix: `__FILENAME__`替换为C++标准预定义宏`__FILE__`。输出会更长(完整路径)但保证可用。
- Reviewability: high -- 单行修改，宏替换语义明确
- Review rule: 头文件中禁止使用依赖编译选项注入的非标准宏；如需短文件名应在头文件内用constexpr提取

### D-320: mstx profiler对default domain处理错误导致崩溃(空指针+空列表判断)

- Hashes: 442016cf1 [+5 cherry-picks: 1ca68f961, 97c317d27, 39ce02bd1, d5523d1ec, 3649f1508]
- Root cause: 特殊值未处理 + 空指针解引用
- Files: torch_npu/csrc/profiler/mstx_mgr.cpp, torch_npu/csrc/profiler/mstx_mgr.h,
  torch_npu/csrc/profiler/npu_profiler.h, torch_npu/profiler/experimental_config.py,
  test/profiler/test_experimental_config.py
- Defect: (1) C++层`createProfDomain("default")`尝试创建系统内置domain，返回handle可能不合法。
  (2) `MstxRange`析构时`domainHandle`为nullptr(domain被过滤)仍无条件使用，空指针解引用。
  (3) Python层`_check_mstx_domain_params()`用`is not None`检查空列表`[]`，条件永真。
- Fix: `createProfDomain()`对`DOMAIN_DEFAULT`直接返回nullptr跳过创建；析构增加nullptr guard；
  Python层改为truthy检查；新增测试覆盖domain开关组合。
- Reviewability: medium -- 涉及C++和Python两层，每个单独看不大但需理解mstx domain生命周期
- Review rule: handle/pointer创建函数调用方必须检查返回值；"系统保留名"应有显式拦截逻辑

### D-321: 批量修复失败测试用例(算子行为变更+硬件差异+测试框架改造)

- Hashes: 830222c89 [+5 cherry-picks: cef697fe5, f5d701b90, cb1b0cd39, 5e8f91824, dfa212415]
- Root cause: 测试维护(上游API变更+算子硬件差异+旧测试脆弱)
- Files: test/adapt_testcases_to_npu.py(新增), test/adaptive_tests.txt(新增),
  test/custom_ops/test_npu_anti_quant.py, test/custom_ops/test_npu_bounding_box_encode.py,
  test/custom_ops/test_npu_conv3d.py, test/custom_ops/test_npu_fused_attention_score_fwd.py(删除),
  test/custom_ops/test_npu_ifmr.py, test/custom_ops/test_npu_stride_add.py,
  test/test_serialization.py(删除), test/trans_contiguous/下5个文件,
  test/unsupported_test_cases/.pytorch-disabled-tests.json
- Defect: 多类测试失败集合修复：contiguous测试硬编码算子名与新路径不匹配；
  test_serialization.py是从上游直接拷贝的4098行文件未经NPU适配；部分custom_ops测试缺少硬件
  型号限制；disabled-tests.json部分已修复用例仍被禁用。
- Fix: contiguous测试改为or逻辑接受新旧路径；删除手工拷贝的test_serialization.py，改为
  `adapt_testcases_to_npu.py`自动化适配框架；硬件敏感测试加`@SupportedDevices`；
  更新disabled-tests.json。
- Reviewability: low -- 改动散布16个文件4000+行，混合多种不同性质修改
- Review rule: 测试修复commit应按修复类型拆分；disabled-tests.json变更应附失败日志说明原因

### D-322: DTensor的D2H copy走错dispatch(to.dtype_layout vs _to_copy)

- Hashes: e89c7eae0 [+3 cherry-picks: 01970494c, fc05e05bd, 317b7a92c]
- Root cause: dispatch注册错误(注册了语义不匹配的aten算子变体)
- Files: torch_npu/csrc/aten/common/ToKernelNpu.cpp,
  torch_npu/csrc/aten/npu_native_functions.yaml,
  test/distributed/_tensor/test_dtensor.py
- Defect: `npu_native_functions.yaml`中注册了`to.dtype_layout`作为NPU自定义实现，但DTensor做
  D2H copy时PyTorch内部dispatch走`aten::_to_copy`而非`aten::to.dtype_layout`。NPU没有注册
  `_to_copy`实现，DTensor的`.cpu()`操作要么fallback到错误路径要么直接失败。
- Fix: yaml中将`to.dtype_layout`替换为`_to_copy`；重写实现为`NPUNativeFunctions::_to_copy()`，
  增加memory_format Preserve/Contiguous分支和pin_memory支持；测试中`_Partial`替换为
  `Partial`(跟随上游API重命名)。
- Reviewability: medium -- 核心是dispatch入口切换，需理解PyTorch tensor copy dispatch机制
- Review rule: npu_native_functions.yaml中注册的算子变体必须与上游实际dispatch路径一致

### D-323: CI构建脚本中冗余环境变量DISABLE_RPC_FRAMEWORK=FALSE

- Hashes: 1893913bd [+10 cherry-picks: 02f33326d, 3e5bbf6a9, d59941eb5, ab692a84f,
  31674538b, 898d7b338, c9a933b64, 96c6443ae, 9cafd5af3]
- Root cause: 构建配置残留(已弃用变量未清理)
- Files: ci/build.sh
- Defect: `ci/build.sh`中`export DISABLE_RPC_FRAMEWORK=FALSE`是冗余设置。该变量的默认行为
  即为不禁用RPC框架，显式设为FALSE与不设等效。残留可能在构建系统变更后引起混淆或与新构建逻辑冲突。
- Fix: 删除该行(1行)。
- Reviewability: high -- 构建脚本中未使用的环境变量应在代码审查中被发现
- Review rule: 构建脚本中的环境变量设置应有对应的消费方；定期清理已弃用的构建配置

### D-324: schema.json API路径未同步更新(含fix-revert-refix cycle)

- Hashes: 25dc8713a [+11 cherry-picks: 361aa468f, b35e7baa3, ef0148b21, b26e5db3b,
  c9de4dc68, 79a0c17d7, 11019beee, 5bb7794e9, 8ce8d4653, 52ccc2d14, bfe450c61]
  Reverts: 3d0a6b484, 5b24022fe, d7237e8e8, 35ba2ffe3, 419ee1040
- Root cause: schema.json与代码API路径不同步(与D-216/D-239/D-244同族)
- Files: test/torch_npu_schema.json
- Defect: torchair API从`torch_npu.dynamo.torchair.ops`重构到`torch_npu.dynamo.torchair.scope`，
  schema.json中仍记录旧路径(`ops.NpuStreamSwitch`/`ops.npu_wait_tensor`)且缺少新增的
  `scope.super_kernel`和`scope.limit_core_num`。schema兼容性测试因此失败。
- Fix: 更新路径为`scope.npu_stream_switch`/`scope.npu_wait_tensor`，新增2个scope API条目。
  此修复经历fix→revert→refix cycle: 首次提交(!19660-!19664)后被全量revert(!19703-!19710)，
  随后以相同内容重新提交(!19759等)。revert原因未在commit中说明，推测为依赖时序问题。
- Reviewability: high -- schema.json应在API重构的同一PR中一并更新
- Review rule: API路径/命名空间重构必须同步更新schema.json；schema.json变更应自动化(从代码生成)

### D-325: NPUCachingAllocator的recursive_mutex与eventfd_read交叉死锁

- Hashes: c4fd1d9df [+9 cherry-picks: f87f72ed2, 1a9c68f85, d56522b52, bb4885114,
  a09ce7fd1, c04caff71, a6b2b2f47, f92a37f3e, c001fd854]
- Root cause: 持锁期间调用可触发GC的同步操作(锁-同步-GC-释放 四方死锁)
- Files: torch_npu/csrc/core/npu/NPUCachingAllocator.cpp
- Defect: `DeviceCachingAllocator`的`malloc`/`garbage_collect`/`release_available_cached_blocks`
  在持有`recursive_mutex`的情况下调用`npuSynchronizeDevice`。同步操作等待task queue清空，
  但task queue子线程中CANN tbe op compile可能触发Python GC，GC尝试释放cached block时需要
  同一个mutex，形成死锁：主线程持锁等子线程→子线程GC等锁→死锁。
- Fix: 引入`UnlockGuard`(RAII风格的临时解锁)，在调用`npuSynchronizeDevice`前释放mutex，
  同步完成后重新获取。同时在`emptyCache`中将sync移到加锁之前。+44行。
- Reviewability: low -- 死锁路径涉及C++锁+Python GC+CANN异步回调三层交互，需深入理解全链路
- Review rule: 持锁期间禁止调用可能触发GC或跨线程同步的函数；allocator代码变更需deadlock分析

### D-326: HCCL CommConfig设置hcclCommName未检查CANN版本兼容性

- Hashes: ddf7e7b4c [+9 cherry-picks: 93cb90ddf, fb8ae90f5, 613f70023, 382341fb5,
  455ba3e39, f37d0b8e4, 5fed87c21, b822f9c81, 833646886]
- Root cause: API可用性检查缺CANN版本guard(与D-63/D-79/D-214/D-253/D-290同族)
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `createHcclCommConfigWithOptions`无条件设置`config.hcclCommName`，但该字段仅在
  较新CANN版本中可用(`HCCL_COMM_CONFIG_COMM_NAME` capability)。旧版本CANN的HcclCommConfig
  结构体中该字段不存在或语义不同，写入会导致内存越界或功能异常。
- Fix: 在设置前检查`isHcclFeatureSupported(HcclCommConfigCapability::HCCL_COMM_CONFIG_COMM_NAME)`。
- Reviewability: high -- 新增HCCL config字段时应标配capability检查
- Review rule: 访问HcclCommConfig中的可选字段前必须检查对应的capability flag

### D-327: flight recorder分析工具dict key直接访问导致KeyError

- Hashes: 6c8350401
- Root cause: dict key存在性未检查(与D-150同族)
- Files: tools/flight_recorder/fr_trace.py
- Defect: `extract_hccl_info`和`analyze_pg_groups`中大量使用`dict[key]`直接访问，但flight
  recorder数据中的字段可能缺失：`frames`可能为空列表、`state`/`record_id`/`name`可能不存在、
  `time_discovered_completed_ns`可能为None。任何缺失字段都会导致KeyError使分析工具崩溃。
- Fix: 全面改用`.get(key, default)`模式；`frames`空列表处理加兜底`[{}]`；
  `record_id`比较前加None检查；`max()`的key函数加`or 0`处理None值。+28行-14行。
- Reviewability: high -- 对外部数据源的dict访问应全部使用.get()
- Review rule: 解析外部/可变schema数据时禁止直接dict[key]，统一用.get()并处理None

### D-328: HCCL stream_id测试中环境变量返回值类型未转换

- Hashes: 9568dfbd4
- Root cause: os.environ.get()返回str但代码假设int
- Files: test/distributed/test_hccl_stream_id.py
- Defect: `os.environ.get("STREAMS_PER_DEVICE", 8)`在环境变量已设置时返回字符串(如"8")，
  未设置时返回默认值8(int)。后续代码`stream_num != 32`和`stream_num & stream_num`
  对字符串执行比较和位运算，结果不符合预期(字符串"8" != 32为True但语义错误)。
- Fix: 添加`try: stream_num = int(stream_num) except: stream_num = 8`。
- Reviewability: high -- os.environ.get()返回类型是Python常见陷阱，review时应注意
- Review rule: os.environ.get()的返回值在用于数值运算前必须显式类型转换

### D-329: torch.load缺少weights_only参数 + 测试skip残留

- Hashes: e2bd6cf55 (v2.6.0), 959f6e1f8 (master，同修复略有差异)
- Root cause: upstream API行为变更(torch.load默认行为变化) + 测试skip未及时清理
- Files: test/contrib/test_drop_path.py, test/distributed/test_hccl_stream_id.py
- Defect: (1) PyTorch 2.6+中`torch.load()`默认`weights_only=True`，但`drop_path_base_data.pth`
  包含非tensor数据(numpy arrays等)，导致UnpicklingError。(2) `test_dist_get_hccl_stream_id_same`
  上的`@unittest.skip("skip this case tmp")`在底层问题修复后未移除，导致测试永久跳过。
  (3) 同时包含D-328的stream_num类型转换修复。
- Fix: torch.load添加`weights_only=False`；移除stale @unittest.skip；添加int()转换。
  v2.6.0版本(e2bd6cf55)包含全部三项修复，master版本(959f6e1f8)不含skip移除(master无此skip)。
- Reviewability: high -- torch.load的weights_only变更是upstream已知breaking change
- Review rule: 临时@unittest.skip必须附带issue tracking号，定期扫描清理

### D-330: profiler动态采集warmup边界条件错误(0被视为无效值)

- Hashes: ff314ef75 [+9 cherry-picks: d7862c5d2, 5b6a1f4d4, 899884fdd, 9e8a3b94e,
  c4e78bb25, ad7575d9e, 6b9828a35, d7b33294c, d85428b5d]
- Root cause: 边界条件判断错误(<=0应为<0，因0是合法值)
- Files: torch_npu/profiler/_dynamic_profiler/_dynamic_profiler_config_context.py
- Defect: `ConfigContext.warmup()`方法中`self._warmup <= 0`将warmup=0视为无效并重置为
  DEFAULT_WARMUP(也是0)。虽然最终结果相同，但当DEFAULT_WARMUP被修改为非0值时，用户显式设置
  warmup=0会被默默覆盖。更根本的问题：warmup=0表示"不需要预热"，是合法配置。
- Fix: `<= 0`改为`< 0`，仅负值被视为无效。
- Reviewability: high -- 边界条件off-by-one是经典review目标
- Review rule: 数值参数校验中0的合法性必须显式考虑(0常是合法的"无/空/零"语义)

### D-331: 分布式rendezvous调用已变更的RendezvousStoreInfo API

- Hashes: bf06bdb54 [+3 cherry-picks: eba5a322d, 9c28e5124, ac1fdede2]
- Root cause: upstream API签名变更后NPU侧未同步(与D-96/D-126/D-134同族)
- Files: torch_npu/distributed/rendezvous.py
- Defect: `RendezvousStoreInfo.build(self.rank, store)`是旧API，新版本PyTorch将构造方式改为
  `RendezvousStoreInfo(master_addr, master_port)`直接传入地址端口。调用旧API导致
  AttributeError或参数不匹配异常。
- Fix: 替换为`RendezvousStoreInfo(self.master_addr, self.master_port)`。
- Reviewability: medium -- 需了解upstream rendezvous API变更历史
- Review rule: 分布式rendezvous代码依赖的PyTorch内部API需在升级时逐一核对

### D-332: profiler DB解析器方法签名/变量名/函数名三类错误

- Hashes: 2c2f797a9 [+9 cherry-picks: 0a9b1d168, e0da93ef0, 0d87a8925, 5eac6f4d7,
  ee2381ee8, 3fd0391ce, 89be513e1, ababff55d, 78df3165e]
- Root cause: 代码重构后调用方未同步 + 变量/方法名拼写错误
- Files: torch_npu/profiler/analysis/prof_view/prof_db_parse/_communication_db_parser.py,
  _fwk_api_db_parser.py, _memory_db_parser.py
- Defect: 三个独立问题: (1) `generate_communication_db(self._output_path)`传入了被移除的参数
  (`output_path`已改为使用`self._profiler_path`)；(2) `node_lauch_apis`拼写错误(lauch→launch)，
  虽不影响运行但影响可读性；(3) `get_pta_memort_record_list`方法名拼写错误(memort→memory)，
  定义和调用都拼错所以不crash但影响代码搜索。
- Fix: 移除多余参数；修正拼写。
- Reviewability: high -- (1)是方法签名变更review的典型遗漏；(2)(3)是拼写检查器可捕获的问题
- Review rule: IDE/linter的spell-check应启用；方法签名变更时grep所有调用方

### D-333: profiler event tree根节点过滤逻辑排除了合法事件

- Hashes: ea69e998f [+5 cherry-picks: 6c142a8d1, cdff2439b, cf5c541bd, 4af2a74b4,
  7773cfafb]
- Root cause: 事件过滤条件过窄(仅保留CPU设备类型的根节点)
- Files: torch_npu/profiler/analysis/prof_parse/_event_tree_parser.py
- Defect: `EventTree.get_root_nodes()`对`parent is None`的事件额外检查`device_type == CPU`，
  但NPU Allocation事件的`device_type`不是CPU，被错误排除。这导致NPU内存事件在event tree中
  丢失，profiler的内存分析视图数据不完整。
- Fix: 移除device_type过滤，所有`parent is None`的事件均作为根节点。-5行。
- Reviewability: medium -- 需理解event tree中不同device_type事件的层级关系
- Review rule: 过滤条件变更需验证不会排除新增的合法事件类型(尤其是NPU设备类型)

### D-334: 多项测试修复(SoC兼容性注解+精度容差+路径+JSON格式)

- Hashes: ecc73b3f9 [+4 cherry-picks: 7292c6894, 9bc1fffb0, c886d9614, 0d4a03c07]
- Root cause: 混合(SoC精度差异 + 测试基础设施路径错误 + JSON写入格式)
- Files: test/contrib/test_bbox_coder.py, test/contrib/test_deform_conv.py,
  test/contrib/test_multiclass_nms.py, test/get_failed_ut_from_log.py
- Defect: (1) `test_bbox_coder`在Ascend910B上精度不同于A1/ProB，缺少SoC区分的测试用例。
  (2) `test_deform_conv`默认精度容差对NPU过严，导致间歇性失败。
  (3) `test_multiclass_nms`在非A1/ProB芯片上算子行为不同但缺少SoC守卫。
  (4) `get_failed_ut_from_log.py`路径前缀错误且JSON写入丢失已有的value2列表数据。
- Fix: (1)添加`@SupportedDevices(["Ascend910B"])`的A2专用测试；(2)放宽容差到`prec=1e-3`；
  (3)添加`@SupportedDevices(['Ascend910A', 'Ascend910P'])`；(4)修正路径+保留json value2。
- Reviewability: medium -- SoC特化测试需了解各芯片精度特性
- Review rule: 新增contrib算子测试必须标注SoC兼容范围；JSON读写逻辑需保留已有字段

### D-335: dynolog IPCMonitor代理eager init导致profiler功能阻塞

- Hashes: ac8f4abfe [+5 cherry-picks: 1452b864b, 0c9b297a2, f3550c1b5, 03af99856,
  1a552cd62]
- Root cause: 模块级初始化副作用(可选依赖import失败阻塞主功能)(与D-140/D-229/D-276同族)
- Files: torch_npu/profiler/_dynamic_profiler/_dynamic_monitor_proxy.py,
  torch_npu/profiler/_dynamic_profiler/_dynamic_profiler_config_context.py
- Defect: `PyDynamicMonitorProxySingleton.__init__`在构造时立即调用`_load_proxy()`，尝试
  `from IPCMonitor import PyDynamicMonitorProxy`。IPCMonitor是可选的dynolog依赖，未安装时
  import异常使proxy永久为None但不影响功能。问题：(1) 每次构造都尝试import(无失败缓存)；
  (2) config解析中`PROFILE_ANALYSE`字符串"false"未通过`BOOL_MAP`转换；
  (3) `gc_detect_threshold`的字符串"None"被`float()`转换导致ValueError。
- Fix: init中不调用_load_proxy，改为get_proxy()时lazy加载+失败缓存(`_load_success`标志)；
  `PROFILE_ANALYSE`加`.lower()`+BOOL_MAP查找；`gc_detect_threshold`加`isinstance(str)`+
  `!= "None"`检查。
- Reviewability: medium -- eager vs lazy init的选择需考虑可选依赖场景
- Review rule: 可选第三方依赖的import必须lazy化且缓存失败状态；字符串配置值必须做类型转换

### D-336: dynolog代理每次调用创建新实例(无singleton管理)

- Hashes: 957e6695d [+5 cherry-picks: e7b72a9a6, c71555b2d, b609b7405, 202c68bd0,
  a66e54f84]
- Root cause: 重量级资源实例化未用singleton管理
- Files: torch_npu/profiler/_dynamic_profiler/_dynamic_monitor_proxy.py (新增),
  torch_npu/profiler/_dynamic_profiler/_dynamic_profiler_monitor.py,
  torch_npu/profiler/dynamic_profile.py
- Defect: `_call_dyno_monitor`和`worker_dyno_func`中每次调用都`from IPCMonitor import
  PyDynamicMonitorProxy`并创建新实例。多实例可能导致资源冲突，且每次import开销不必要。
  退出时无法正确清理dynolog状态。
- Fix: 提取`PyDynamicMonitorProxySingleton`单例类到独立模块，用`@Singleton`装饰器确保
  全局唯一实例。调用方改为通过`get_proxy()`获取共享实例。D-335后续在此基础上改为lazy init。
- Reviewability: medium -- singleton提取是标准重构，需确认所有调用方都已切换
- Review rule: 与外部进程通信的代理对象应singleton管理；重构后grep确认无遗漏的直接构造调用

### D-337: CannPackageManager类属性在import时求值导致初始化失败

- Hashes: 3a359b8f2 [+5 cherry-picks: 456aa19ff, ed920b1da, 96ea4dd4e, f47e8c1e5,
  0dfd68f62]
- Root cause: 模块级初始化副作用(类属性在import时求值)(与D-140/D-229/D-276/D-335同族)
- Files: torch_npu/profiler/analysis/prof_common_func/_cann_package_manager.py,
  torch_npu/profiler/analysis/_profiling_parser.py,
  torch_npu/profiler/profiler_interface.py
- Defect: `CannPackageManager.SUPPORT_EXPORT_DB = check_cann_package_support_export_db()`
  在类定义时(module import时)立即调用CANN包检查。如果此时CANN环境未就绪(如import顺序、
  容器环境)，检查失败导致整个profiler模块不可用。此外`profiler_interface.py`中`__init__`
  过早校验DB导出能力，用户还未开始profiling就抛出异常。
- Fix: 改为`@classmethod is_support_export_db()`+`None`哨兵值的lazy模式；移除`__init__`中
  的过早校验(保留在实际导出DB时检查)；移除对`CannPackageManager`的import-time依赖。
- Reviewability: medium -- eager→lazy重构模式已在D-335中出现，同一轮两次说明此类问题系统性
- Review rule: 类属性赋值中禁止调用可能失败的外部检查函数；改用lazy property或classmethod

### D-338: task queue异步错误时OOM snapshot未触发(GE error路径遗漏)

- Hashes: 44d2aad41 [+5 cherry-picks: 1c2110299, 055c5be00, a47377f7f, 1cea1c910,
  d7ef5286c]
- Root cause: 异步错误路径缺少错误消息传播链(OOM snapshot仅覆盖同步路径)
- Files: torch_npu/csrc/core/npu/NPUException.cpp, NPUQueue.cpp, NPUQueue.h,
  NPUStream.cpp, NPUStream.h, register/OptionsManager.cpp
- Defect: OOM snapshot功能仅在`NPUCachingAllocator`的同步分配失败时触发，但GE(Graph Engine)
  通过task queue异步执行时的OOM错误不会触发snapshot。根因：`Repository`没有保存错误消息的
  机制，错误从`c10_npu_get_error_message()`返回后丢失，queue的`MakeSureQueueEmpty`和
  `Enqueue`的error exit路径无法判断是否为OOM。
- Fix: (1) `Repository`增加`error_msg`成员+`SetQueueErrMsg`/`GetQueueErrMsg`接口；
  (2) `c10_npu_get_error_message()`调用`setRepoErrMsg()`将错误消息广播到所有repo；
  (3) error exit路径中检查`strstr(errmsg, "Failed to allocate memory")`触发OOM snapshot；
  (4) `IsOomSnapshotEnable()`从void改为bool返回+`isFirstCall`守卫防止重复注册observer。
  注意：`error_msg`成员(`const char*`)未初始化，`GetQueueErrMsg()`在首次`SetQueueErrMsg`
  前调用会返回未定义值。
- Reviewability: low -- 涉及6个文件跨3层(异常处理→队列→流管理)的错误传播链
- Review rule: 新增的C++成员变量必须初始化(尤其是raw指针)；异步错误路径需与同步路径等价覆盖

### D-339: slow_test_blocklist条目含.py后缀导致匹配失败

- Hashes: 715ccfd44 [+5 cherry-picks: 4744d0fbb, 78dfb70cb, 111aad72b, ab85b4f9c, 4c9d9ff4b]
- Root cause: 列表条目格式不一致(其他条目无.py后缀)
- Files: ci/access_control/constants.py
- Defect: SLOW_TEST_BLOCKLIST中`test_jit_fuser_te.py`带了`.py`后缀，而其余条目(如
  `test_reductions`, `test_unary_ufuncs`)均不带。CI匹配逻辑按不带后缀的名称查找，导致
  这条规则从未生效，该慢测试一直在跑。
- Fix: `test_jit_fuser_te.py` → `test_jit_fuser_te`。
- Reviewability: high -- 列表中格式不一致肉眼可见
- Review rule: 批量列表中新增条目时检查与既有条目格式是否一致

### D-340: attachOutOfMemoryObserver循环内std::move导致observer丢失

- Hashes: 72e58143e [+5 cherry-picks: 5621a5b76, c52c1b6d6, a0542467a, 6e7b49be1, 31ee13bd5]
- Root cause: 循环内std::move导致对象被转移后续迭代使用空壳
- Files: torch_npu/csrc/core/npu/NPUCachingAllocator.cpp
- Defect: `NpuCachingAllocator::attachOutOfMemoryObserver()`遍历所有`device_allocator`，对每个
  调用`allocator->attachOutOfMemoryObserver(std::move(observer))`。第一次循环后observer被
  move走，后续allocator收到的是moved-from空对象，只有device 0能收到OOM回调。
- Fix: 移除`std::move`，改为按值传递observer(复制到每个allocator)。
- Reviewability: high -- std::move在循环内是已知anti-pattern，静态分析可检测
- Review rule: 循环体内禁止std::move循环变量以外的对象；如需多处使用同一callable，按值传递

### D-341: ensemble_dropout测试p=1导致输出恒为0、测试无意义

- Hashes: 6c885279d [+5 cherry-picks: a4f98979a, 424c68a00, 9cf9857ad, ebaff0054, 060bf8e9c]
- Root cause: 测试参数选择不当(p=1时dropout率100%，输出全零)
- Files: test/contrib/test_ensemble_dropout.py
- Defect: `NpuFairseqDropout(p=1)`和`NpuCachedDropout(p=1)`将dropout概率设为100%，所有元素
  被丢弃，输出恒为0。这使得测试只验证了"全部丢弃"这一退化情况，未验证dropout的概率采样行为。
- Fix: `p=1` → `p=0.5`。
- Reviewability: high -- p=1的语义(全部丢弃)是dropout的基本知识
- Review rule: dropout测试的概率参数应选择能验证随机采样行为的值(0.2-0.8)

### D-342: 分布式测试子进程在父进程读取结果前退出(同D-341 summary，不同fix)

- Hashes: 9cf59347b [+5 cherry-picks: 8ab42acd9, 1c90b2860, 50641c1d1, ba493ea1c, cbfac3079]
- Root cause: 多进程测试缺少子进程退出同步机制
- Files: test/distributed/test_allgather_base.py, test_allgather_into_tensor.py,
  test_reduce_scatter_tensor.py
- Defect: 子进程通过`c2p.put()`发送结果后立即退出。如果子进程退出时HCCL资源释放触发barrier
  或destroy操作，可能导致其他进程hang或崩溃。父进程`p.join(2)`超时后强杀子进程，结果不确定。
  注: 此commit与D-341 summary完全相同("Fixed the failed tests.")但修复内容不同，是summary去重
  策略的已知盲区。
- Fix: 添加`p2c`(parent-to-child) Queue，子进程发送结果后`p2c.get()`阻塞等待父进程信号再退出。
  父进程读取完所有结果后`p2c.put(0)`释放子进程。与D-226模式完全一致。
- Reviewability: medium -- 需理解多进程退出时序和HCCL资源回收行为
- Review rule: 分布式多进程测试中子进程必须有显式的退出同步点

### D-343: codegen device_check中op_name类型与DEVICE_NOCHECK_SET不匹配

- Hashes: 6b44e6282 [+5 cherry-picks: a330367dd, c38e2b85a, d4841401e, 8f78f4e17, b6eed5b40]
- Root cause: 变量类型不匹配导致集合查找永远为False
- Files: codegen/utils.py
- Defect: `op_name not in DEVICE_NOCHECK_SET`中`op_name`的类型/格式与`DEVICE_NOCHECK_SET`
  中存储的字符串不同。`op_name`来自某种对象表示，而集合中存储的是`str(f.func.name)`格式。
  结果: NOCHECK_SET中的所有op都仍然生成了device check代码。
- Fix: `op_name` → `str(f.func.name)`。
- Reviewability: high -- 集合查找的key类型必须与集合元素类型一致
- Review rule: 集合/字典查找时确保key的类型和格式与容器内元素完全匹配

### D-344: NpuExtension缺少acl/hccl头文件搜索路径

- Hashes: 82c91af45 [+4 cherry-picks: fd424421d, cf5fee609, bda57f140, f361e1cfd]
- Root cause: cpp_extension的include_dirs未包含第三方库头文件目录
- Files: torch_npu/utils/cpp_extension.py
- Defect: `NpuExtension()`只添加了`torch_npu/include`目录，未添加
  `include/third_party/acl/inc`和`include/third_party/hccl/inc`。用户通过NpuExtension编译
  自定义C++扩展时，`#include`ACL或HCCL头文件会编译失败。
- Fix: 添加acl/inc和hccl/inc到include_dirs。
- Reviewability: high -- 第三方库头文件路径是显而易见的依赖
- Review rule: 新增对外暴露的头文件目录时，同步更新cpp_extension的搜索路径

### D-345: profiler C++代码有符号/无符号类型隐式转换

- Hashes: 8fd0ed8d4 [+4 cherry-picks: 956a7c9e3, ed1b26851, 7a840f897, fa9bda590]
- Root cause: C++ int vs size_t隐式转换(编译器warning，潜在溢出)
- Files: torch_npu/csrc/profiler/dyno/NpuIpcClient.cpp, NpuIpcEndPoint.h
- Defect: 三处signed/unsigned不匹配: (1)`int size = pids_.size()`(size_t→int截断);
  (2)`size_t retCode = recvmsg(...)`(recvmsg返回ssize_t，赋给size_t丢失负值);
  (3)`for(int i=0; i<npuPayLoad.size(); i++)`(int与size_t比较)。第(2)处最危险:
  recvmsg返回-1(错误)被转为极大正值，后续`if(retCode>0)`误判为成功。
- Fix: 三处均改为`auto`让编译器推导正确类型。
- Reviewability: high -- 编译器warning可直接捕获
- Review rule: 启用-Wsign-compare/-Wconversion编译选项；recvmsg返回值必须用ssize_t

### D-346: 测试缺少SoC兼容性装饰器 + drop_path参数错误

- Hashes: 225f38670 [+4 cherry-picks: fba7939c4, 3553dba6b, 9b22bc023, 459abbed0]
- Root cause: 测试用例SoC兼容性守卫缺失 + 测试参数退化
- Files: test/contrib/test_batchnorm_with_count_int32.py, test/contrib/test_drop_path.py
- Defect: (1) FastBatchNorm测试在非910A/910P芯片上运行失败但未skip；
  (2) test_drop_path中`DropPath(0)`和`NpuDropPath(0)`的p=0表示不丢弃，与DropPath功能测试
  意图矛盾；(3) 测试shape过大导致在低端设备OOM。
- Fix: (1) 添加`@SupportedDevices(['Ascend910A', 'Ascend910P'])`；(2) p改为0.5；
  (3) 缩小测试shape(50→10等)。
- Reviewability: high -- SoC装饰器是torch_npu测试的标准模式(同D-90/D-148/D-219模式)
- Review rule: 涉及NPU算子特性的测试必须标注支持的SoC型号

### D-347: codegen out函数redispatch名称模式错误(_outf vs _out)

- Hashes: ba645b3ff [+4 cherry-picks: 9ad992184, 997222363, 83d0826b7, f78274bc0]
- Root cause: NPU redispatch命名约定与upstream不同
- Files: codegen/autograd/gen_variable_type.py
- Defect: `type_definition.replace('at::redispatch', 'at_npu::redispatch')`对out函数也做
  全量替换，但upstream的out函数用`at::redispatch::xxx_outf`而NPU侧需要
  `at_npu::redispatch::xxx_out`(无f后缀)。全量替换保留了`_outf`后缀，链接时找不到符号。
- Fix: 区分out函数和非out函数: out函数用regex将`_outf`替换为`_out`，非out函数保持全量替换。
- Reviewability: medium -- 需理解upstream和NPU的redispatch命名约定差异
- Review rule: codegen中的字符串替换需区分不同函数签名模式(out vs non-out)

### D-348: 线程亲和性绑定评估时机和触发条件多处错误

- Hashes: ebfa1de36 [+7 cherry-picks: f9397a415, b561b654f, c66a1fafe, d26a42a10,
  b00a97565, b0daf6f81, d2b5db86f]
- Root cause: 线程亲和性检查的评估时机和主线程绑定触发点不正确
- Files: torch_npu/csrc/core/npu/NPUAffinityController.cpp, NPUAffinityController.h
- Defect: 三个独立问题: (1) `has_set_pthread_affinity()`通过static local变量在每个线程首次
  调用时求值，但affinity应该在main thread启动时一次性检测并全局共享；
  (2) bind_conf=1模式下，如果检测到affinity已设置就跳过绑定，但检测结果因线程不同而不确定；
  (3) bind_conf=2模式下，主线程重绑定由aclThread触发，但此时device_id可能还未最终确定，
  应改为backwardThread(dispatch阶段才确定目标device)。
- Fix: (1) 改为`GetAffinityInfo()`在初始化时一次性评估，结果存入全局`has_set_affinity`；
  (2) 在`SetThreadAffinity`入口检查全局flag，已设置则skip；
  (3) 主线程重绑定触发点从aclThread改为backwardThread。
- Reviewability: low -- 涉及线程生命周期、设备初始化时序、外部affinity工具交互
- Review rule: 线程亲和性策略的评估和执行应在明确的初始化点，不应依赖static local延迟初始化

### D-349: test_fault_mode进程退出时序竞争

- Hashes: 0aefa68f7 [+4 cherry-picks: 6e08e7a5f, 4ecc73f00, 9156b01f6, 478041129]
- Root cause: 多进程错误信息可能分布在任意进程的stderr中
- Files: test/distributed/test_fault_mode.py,
  test/distributed/_fault_mode_cases/error_use_same_addr.py
- Defect: 测试假设"Address already in use"错误消息一定出现在process[1]的stderr中，但实际
  取决于进程启动和端口绑定的竞争时序，错误可能出现在任一进程。此外循环次数1000可能不足以让
  第二个进程的通信操作触发冲突。
- Fix: (1) 合并所有进程stderr后统一检查；(2) 循环次数1000→5000以确保冲突发生。
- Reviewability: medium -- 需理解torchrun多进程的端口竞争行为
- Review rule: 多进程错误检测应聚合所有进程的输出，不应假设错误出现在特定进程

### D-350: contiguous()方法clone时未传递memory_format参数

- Hashes: 88dab78e8 [+6 cherry-picks: 62e67c351, b5caf8a68, 7dcade8b8, 639f65b8c,
  40f1862d6, 97b80840e]
- Root cause: clone调用丢失memory_format参数(与D-43/D-92同族)
- Files: torch_npu/csrc/aten/common/TensorProperties.cpp, test/npu/test_npu.py
- Defect: `NPUNativeFunctions::contiguous()`在tensor不连续时调用`self.clone()`而非
  `self.clone(memory_format)`。`clone()`默认使用`Contiguous`格式，即使函数已验证
  `memory_format==Contiguous`，但upstream语义要求将用户指定的format传递给clone以确保
  FakeTensor等代理对象的行为一致性。
- Fix: `self.clone()` → `self.clone(memory_format)`。新增测试验证contiguous在各种format下
  的行为，包括preserve_format和channels_last的错误提示。
- Reviewability: high -- clone调用缺少参数透传是常见遗漏，与D-43(tensor factory)、D-92(clone
  stride)形成tensor创建路径的系统性问题
- Review rule: NPU tensor操作中调用clone/empty/new等factory方法时必须透传memory_format

### D-351: requirements.txt硬编码torch/torch_npu版本号

- Hashes: e0288cfe9 [+4 cherry-picks: 15b544002, db9c9c41b, 078b7e4db, 68dbd155b]
- Root cause: 测试依赖文件包含硬编码版本号，跨版本分支不兼容
- Files: test/requirements-arm.txt, test/requirements-x86.txt
- Defect: requirements文件硬编码`torch==2.1.0`和`torch_npu==2.1.0.post3`，在非v2.1.0分支上
  安装测试环境会强制装错误版本的torch。
- Fix: 移除torch和torch_npu的版本pin，仅保留hypothesis/beartype等测试框架依赖。
- Reviewability: high -- 版本号硬编码在多分支项目中是显而易见的问题
- Review rule: 测试依赖文件中不应pin主项目及其核心依赖的版本

### D-352: 分布式all_to_all测试子进程退出竞争

- Hashes: ec02dd402 [+4 cherry-picks: 50653e103, 576845946, acd0eaeca, 8de4d5c0e]
- Root cause: 同D-342，多进程测试缺少子进程退出同步
- Files: test/distributed/test_all_to_all.py
- Defect: 与D-342完全相同的模式: all_to_all测试函数(`_test_alltoall_2p`, `_test_alltoall_4p`
  等)中子进程`c2p.put()`后直接退出，父进程`p.join(2)`可能超时。
- Fix: 同D-342模式: 添加`p2c` Queue + `p2c.get()`/`p2c.put(0)`同步协议。
- Reviewability: high -- 与D-342是同一模式在不同文件的重复，说明修复不够系统性
- Review rule: 分布式测试的多进程同步应作为基类/fixture提供，而非逐文件重复实现

### D-353: profiler fwdbwd flow连接了同线程事件(应为跨线程)

- Hashes: 5f144ceb3 [+7 cherry-picks: 1a2d6e563, 0c9e83ddd, a098b2cdd, f20eebc91,
  64443a306, 67a155419, 708122295]
- Root cause: fwdbwd flow事件过滤条件缺少tid检查
- Files: torch_npu/profiler/analysis/prof_common_func/_trace_event_manager.py
- Defect: `TraceEventManager`生成fwdbwd flow时只检查start和end节点是否存在，未检查它们是否
  在不同线程。同线程的"前向→反向"不应生成flow箭头(因为调用栈已经体现了这个关系)，只有跨线程
  的前向→反向(如DataParallel场景)才需要flow连接。同线程flow导致trace视图中出现自指箭头。
- Fix: 添加`if node['start']['tid'] == node['end']['tid']: continue`过滤。
- Reviewability: high -- fwdbwd flow的语义(跨线程)是profiler设计知识
- Review rule: profiler flow事件生成时必须检查source和target是否满足跨实体(线程/进程)条件

### D-354: profiler内存图标签GB/GiB混淆

- Hashes: 43a44d4e6 [+7 cherry-picks: 82d492dc9, bc5c69259, 046e462e9, 44870965b,
  cb01cbebb, 0c22ed405, be87dc019]
- Root cause: 单位标签与计算公式不匹配
- Files: torch_npu/profiler/analysis/prof_view/_memory_timeline_parser.py
- Defect: 内存timeline图表标题使用"GiB"标签，但实际计算用`Constant.B_TO_GB`(10^9，即GB)除法。
  1 GiB = 2^30 = 1,073,741,824 bytes，1 GB = 10^9 bytes，差异约7.4%。显示值比实际GiB值
  偏大(如8 GB显示为8.00但实际是7.45 GiB)。
- Fix: `GiB` → `GB`，与计算公式保持一致。
- Reviewability: high -- 单位标签应与计算公式匹配，这是基本的一致性检查
- Review rule: 数据展示层的单位标签必须与计算层使用的常量/公式一致

### D-355: transfer_to_npu替换Tensor.to后LazyModule._allowed_methods引用失效

- Hashes: f8bb0ed1c [+7 cherry-picks: 719c6d806, c60916f0a, a902deea6, f06e40e07,
  90c676dff, 4612b9bf8, d297401e6]
- Root cause: transfer_to_npu替换类方法后未更新持有旧引用的注册表
- Files: torch_npu/contrib/transfer_to_npu.py, test/contrib/test_transfer_to_npu.py
- Defect: `transfer_to_npu`通过`torch.Tensor.to = _device_wrapper(...)`替换了`to`方法。
  但`UninitializedTensorMixin._allowed_methods`列表在import时已保存了原始`torch.Tensor.to`
  的引用。LazyModule通过该列表判断哪些方法可以在未初始化参数上调用，仍指向旧的`to`方法，
  导致`lazy_module.to("npu")`无法触发transfer_to_npu的设备重定向。
  与D-256/D-265/D-268(transfer_to_npu类替换破坏isinstance)同族。
- Fix: `_init()`末尾调用`_replace_to_method_in_allowed_methods()`，遍历
  `_allowed_methods`列表将`to`的旧引用替换为新的`torch.Tensor.to`。
- Reviewability: medium -- 需理解LazyModule的_allowed_methods机制
- Review rule: monkey-patch类方法时需搜索所有持有原方法引用的注册表并同步更新

### D-356: bidirectional_lstm测试精度容忍度过严

- Hashes: b109da06f [+4 cherry-picks: a12719068, e2a9106cf, 37696b11d, fb7fe4aac]
- Root cause: 默认assertRtolEqual精度对NPU浮点运算过严
- Files: test/contrib/test_bidirectional_lstm.py
- Defect: LSTM前向+反向的浮点累积误差在NPU上超过assertRtolEqual的默认容忍度。NPU的浮点单元
  与CPU在denormal处理和融合乘加精度上存在差异，LSTM的多步递归放大了这些差异。
- Fix: 添加`prec=1.e-3`参数放宽容忍度。
- Reviewability: high -- NPU精度差异是已知行为，测试应使用合理容忍度
- Review rule: RNN类递归算子的NPU测试应使用1e-3级别容忍度

### D-357: schema.json残留已删除的公开API (batch 1)

- Hashes: 35c025e2e [+7 cherry-picks: c265329b9, befa0f5b7, 5bd2bbf27, 903fadc4d,
  17095a211, fbb17a2f5, 4d34c0297]
- Root cause: API删除/重构后schema.json和测试filter未同步清理(与D-216/D-239/D-244/D-324同族)
- Files: test/npu/test_public_bindings.py, test/torch_npu_schema.json
- Defect: contrib模块中FastBatchNorm系列、NpuPreGenDropout、MatmulApply等类被删除或改为私有，
  但schema.json和test_public_bindings.py的temp_filter中仍保留这些条目。公开API一致性测试
  因此报错。这是schema.json维护问题的第五次出现。
- Fix: 从schema.json和temp_filter中移除已删除API的条目。
- Reviewability: high -- API删除PR应自带schema更新(CI应检查schema一致性)
- Review rule: 删除或私有化API时必须同步更新schema.json和公开API测试

### D-358: Dockerfile pip路径typo + MHAConfig内部类重命名未同步 (batch 2)

- Hashes: 5225d6aa3 [+7 cherry-picks: 4a92f8f9a, c9860d3e6, b00b02de7, 8283f4e1e,
  b9c78de79, 2d098799f, 8ceffb7f6]
- Root cause: 多个独立问题的batch fix(与D-357同summary但不同content，dedup gap)
- Files: ci/docker/ARM/Dockerfile, test/contrib/test_multihead_attention.py,
  test/torch_npu_schema.json
- Defect: (1) Dockerfile中`pip3.11`的符号链接指向`pip3.10`而非`pip3.11`，Python 3.11
  用户安装包时实际调用的是3.10的pip；(2) `MHAConfig`重命名为`_MHAConfig`(私有化)后测试
  仍import旧名；(3) 更多schema.json条目需清理。
- Fix: (1) `cpython-3.11.6/bin/pip3.10` → `pip3.11`；(2) `MHAConfig` → `_MHAConfig`；
  (3) 移除schema中的DropOutTask/PreGenDropoutTask等。
- Reviewability: high -- Dockerfile typo是纯文本审查可发现的
- Review rule: Dockerfile修改后应有自动化验证(pip --version检查)

### D-359: dynamicQuantization wrapper默认dst_type与算子实际输出不匹配

- Hashes: e3c37645b
- Root cause: ONNX wrapper函数默认参数值与底层算子语义不一致
- Files: torch_npu/onnx/wrapper_onnx_ops.py
- Defect: `_wrapper_npu_dynamic_quant()`的`dst_type`默认值为`torch.int`(int32)，但
  npu_dynamic_quant算子实际输出int8量化结果。ONNX导出时如果用户不显式指定dst_type，会生成
  int32的量化结果，与推理引擎的int8预期不匹配。
- Fix: `torch.int` → `torch.int8`。
- Reviewability: high -- 默认值应反映算子最常用的语义
- Review rule: wrapper函数的默认参数值应与底层算子的典型输出类型一致

### D-360: RANK_TABLE_FILE场景下HCCL_BUFFSIZE环境变量未传递

- Hashes: 835ae41ba [+7 cherry-picks: 42e86510b, c990923da, 8f0ec202a, 68199d58e,
  c8df4c007, 17bce4ff2, c2281b974]
- Root cause: ranktable模式的HcclComm创建路径未读取用户配置的buffer size
- Files: torch_npu/csrc/core/npu/register/OptionsManager.cpp, OptionsManager.h,
  torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: 使用RANK_TABLE_FILE方式创建HCCL通信域时，`createHCCLCommEx()`对commConfig为null
  的情况只调用`HcclCommConfigInit(&config)`(使用默认buffer size)，忽略了用户通过
  `HCCL_BUFFSIZE`环境变量配置的值。P2P场景已有`getP2PHcclCommConfig()`读取P2P_BUFFER_SIZE，
  但普通场景缺少对应实现。
- Fix: 新增`GetHcclBufferSize()`读取HCCL_BUFFSIZE(默认200MB)；新增`getHcclCommConfig()`
  设置`config->hcclBufferSize`；两处`HcclCommConfigInit`调用改为`getHcclCommConfig`。
- Reviewability: medium -- 需对比P2P路径发现普通路径的配置遗漏
- Review rule: 同一资源的多条创建路径必须对齐配置参数读取逻辑

### D-361: taskqueue STOP_EXIT时event未记录计数未清零

- Hashes: 896dc891d [+7 cherry-picks: 3d15026fc, c8ff2208d, 84839918a, 3821471bb,
  baa150a7b, be77f5ddb, f561de871]
- Root cause: taskqueue异常退出路径缺少event管理器状态清理
- Files: torch_npu/csrc/core/npu/NPUEventManager.cpp, NPUEventManager.h, NPUQueue.cpp
- Defect: `Repository::ReadQueue()`执行失败或进入STOP_EXIT状态时，调用`ClearQueue()`清理
  队列但未清理`NPUEventManager`的`event_unrecorded_count_`。残留的计数导致后续
  `IsEventRecorded()`对已销毁的event返回false(认为未记录)，可能导致event重复记录或
  销毁已记录event时hang在等待完成上。
- Fix: 新增`ClearUnrecordedCount()`方法；在error exit和STOP_EXIT两条路径的`ClearQueue()`
  后调用。
- Reviewability: medium -- 需理解event_unrecorded_count与queue生命周期的关联
- Review rule: 资源清理函数必须覆盖所有关联状态(queue清理时同步清理event计数)

### D-362: profiler torch_op的input_dtypes和input_shapes顺序反了

- Hashes: f1af634e6 [+4 cherry-picks: 13db5471e, 6f1e63fd8, b2234bfd1, 1ac959e42]
- Root cause: list构造时字段顺序与消费方期望不匹配
- Files: torch_npu/profiler/analysis/prof_parse/_fwk_file_parser.py
- Defect: `FwkFileParser`构造api list时`INPUT_SHAPES`在第7位、`INPUT_DTYPES`在第8位，
  但下游消费方按[7]=dtypes, [8]=shapes读取。顺序反了导致trace展示中类型和shape互换。
- Fix: 交换两个`args.get()`的位置。
- Reviewability: high -- 列表位置编码是脆弱的设计，但顺序错误可通过输出验证发现
- Review rule: 使用dict或namedtuple替代位置编码的list，避免顺序依赖

### D-363: mstx patch覆写了graph mode已定制的Module.__call__

- Hashes: 10bd43a82 [+4 cherry-picks: bc5ec9149, 1db828a23, 1b13a878c, 486d76589]
- Root cause: profiler mstx patch与pytorch graph mode的Module.__call__定制冲突
- Files: torch_npu/profiler/_add_mstx_patch.py, test/profiler/test_profiler_tree.py
- Defect: `apply_mstx_patch()`无条件替换`Module.__call__`为`_custom_tx_call`，但在pytorch
  graph mode(JIT/torch.compile)下，`Module.__call__`已被替换为支持tracing的版本(去掉了
  `_wrapped_call_impl`层)。mstx patch覆写后graph trace丢失了模块层级信息，profiler tree
  测试中`is_initialized`和`_wrapped_call_impl`条目消失。
- Fix: 从`apply_mstx_patch()`中移除Module.__call__的patching，仅保留DataLoader.__iter__
  和torch.serialization.save的patch。Module级别的profiling由其他机制处理。
- Reviewability: medium -- 需理解Module.__call__在不同mode下的定制链
- Review rule: monkey-patch前检查目标方法是否已被其他组件定制(检查__name__属性)

### D-364: HCCL destroyHcclComm重复调用 + heartbeat monitor默认启用导致shutdown竞争

- Hashes: 0fcf48af1
- Root cause: heartbeat monitor默认启用引发shutdown时资源竞争 + 重复析构
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: (1) `monitorThreadEnabled_`默认true，所有场景都启动heartbeat线程，在短生命周期
  进程中增加不必要的开销和shutdown复杂度；(2) `shutdown()`中异步调用`this->abort()`
  会destroy所有comm，但后续正常清理路径也会destroy，导致double free；
  (3) heartbeat monitor线程在`abort`失败时可能hang在`waitForFutureOrTimeout`。
- Fix: (1) 默认改为false(`getCvarBool(TORCH_HCCL_ENABLE_MONITORING, false)`)；
  (2) 移除shutdown中的async abort调用；(3) monitor线程启动加条件判断。
- Reviewability: medium -- 需理解shutdown路径的资源释放顺序(与D-65/D-209/D-215/D-296同族)
- Review rule: 可选的监控/诊断功能应默认关闭，启用时需验证shutdown路径的资源安全

### D-365: profiler DB写入死锁(sqlite跨线程 + 子进程模式 + 依赖关系缺失)

- Hashes: b429f3c6f [+7 cherry-picks: 148aa4aae, 1770e1329, dc93a0527, d2232316d,
  018838294, 8ecbc5956, a551968c0]
- Root cause: profiler并行解析器使用子进程模式导致sqlite线程安全检查失败 + 执行依赖不完整
- Files: torch_npu/profiler/analysis/prof_common_func/_db_manager.py, _id_manager.py,
  torch_npu/profiler/analysis/prof_config/_parser_deps_config.py
- Defect: 三个独立问题: (1) sqlite3默认`check_same_thread=True`，但SUB_PROCESS模式的
  parser在不同进程/线程中访问同一DB连接时触发`ProgrammingError: SQLite objects created
  in a thread can only be used in that same thread`；(2) `Str2IdManager.get_all_data()`
  返回数据后不清空map，多个parser调用时重复插入相同的str2id映射；
  (3) MEMORY_DB_PARSER缺少对FWK_API_DB_PARSER的依赖，并发写入同一DB表时发生锁等待。
- Fix: (1) 添加`check_same_thread=False`；(2) `get_all_data()`返回后`self._str_id_map.clear()`；
  (3) 将多个parser的模式从SUB_PROCESS改为PTHREAD，并补齐依赖关系。
- Reviewability: medium -- sqlite线程安全是已知陷阱，但profiler并行架构增加了复杂度
- Review rule: 多线程/多进程环境中sqlite连接必须设check_same_thread=False；并行parser间
  的数据依赖必须在deps配置中显式声明

### D-366: profiler无数据时仍输出空JSON文件

- Hashes: 0f83e9e20 [+9 cherry-picks: 861a733f5, 0feee4aa2, 9c854a09c, a066f4764,
  574d22afe, 858398166, d97301efe]
- Root cause: 数据为空时未做前置检查直接写文件
- Files: torch_npu/profiler/analysis/prof_view/_memory_timeline_parser.py
- Defect: `MemoryProfileTimeline`的三个导出方法(`export_memory_timeline_raw`,
  `export_memory_events_raw`, `_coalesce_timeline`)在数据为空时仍创建JSON/gz文件，
  输出空数组或空结构。下游工具(如Perfetto)加载空文件时可能报错或显示空白。
  `_coalesce_timeline`虽有`len(timestamps)==0`检查但分支过晚(已做了np.array转换)。
- Fix: 三个方法各自在数据获取后立即检查空集，`print_error_msg`后return。
- Reviewability: high -- 空数据检查是防御性编程的基本模式
- Review rule: 文件导出函数入口处必须检查数据非空

### D-367: step_info DB parser子进程模式导致sqlite线程冲突

- Hashes: 938310f74 [+4 cherry-picks: 7571f58bb, 470452e7f, e9aa18d55, 38f844922]
- Root cause: 与D-365相同根因(SUB_PROCESS模式sqlite冲突)，D-367是初始修复(仅step_info)，
  D-365是扩展修复(全部parser)
- Files: torch_npu/profiler/analysis/prof_config/_parser_deps_config.py
- Defect: `STEP_INFO_DB_PARSER`使用`ConcurrentMode.SUB_PROCESS`，子进程与主进程共享DB
  连接时sqlite报线程检查错误，step_info表导出失败。
- Fix: `SUB_PROCESS` → `PTHREAD`。
- Reviewability: high -- 与D-365同根因，说明问题被逐个parser发现而非系统性排查
- Review rule: 同一类资源(DB连接)的所有使用点应统一并行策略

### D-368: dynamic profiler测试缺少start_step配置

- Hashes: 30206a682 [+10 cherry-picks: 5a5782aac, 5ee399af7, cdbc5850d, 78b9079be,
  9ae16f183, 92574d141, 4ad973c47, 0a6f87fc9, 382e0351e, 8548f379c]
- Root cause: dynamic profiler测试未跟踪step计数器导致profiling窗口不匹配
- Files: test/profiler/test_dynamic_profiler.py
- Defect: dynamic profiler根据配置的`start_step`决定从哪个step开始采集。测试用例共享
  全局step计数器但config中未设置`start_step`，导致后续测试case的step编号已超过默认
  采集窗口，profiling永远不触发。此外缺少`os.environ["RANK"]`设置导致rank过滤逻辑异常。
- Fix: (1) 添加类级`start_step`计数器，每次`dp.step()`后递增；
  (2) 每个test case在config中设置`cfg_json['start_step'] = start_step + 1`；
  (3) setUp中设置`os.environ["RANK"] = "0"`。
- Reviewability: medium -- 需理解dynamic profiler的step窗口机制
- Review rule: 多test case共享状态(step counter)时每个case必须正确设置起始点

### D-369: foreach_optim从逐函数monkey-patch改为SoC/CANN版本检测

- Hashes: 580b78f61 [+1 cherry-pick: 5de781c7b]
- Root cause: 过度侵入式monkey-patch(逐个optimizer函数替换)
- Files: torch_npu/utils/_optim.py
- Defect: 原实现为11个optimizer函数(sgd/adam/adamw等)各自用`partial`包装强制`foreach=False`，
  并通过`partialmethod`修改类构造器。这种方式有三个问题：(1) upstream新增optimizer时需手动添加；
  (2) 从每个optimizer模块做from-import创建独立引用，与D-45/D-102同族的快照问题；
  (3) 不区分SoC能力，对所有硬件一刀切禁用foreach。
- Fix: 改为patch `torch.optim.optimizer._get_foreach_kernels_supported_devices`，根据CANN版本
  和SoC型号动态返回支持foreach的设备列表。CANN 8.0.RC1/RC2及早期T版本黑名单返回仅cuda+xpu；
  910B及以上SoC额外返回privateuse1(npu)。代码从84行减至34行。
- Reviewability: high -- 原代码明显是"逐条硬编码"模式，重构后架构更清晰
- Review rule: 替换框架行为时优先hook已有扩展点(如supported_devices)，而非逐函数替换

### D-370: profiler DB导出方法名下划线不一致

- Hashes: 755d41728
- Root cause: 方法重命名不完整(caller与callee不同步)
- Files: torch_npu/profiler/analysis/prof_view/prof_db_parse/_memory_db_parser.py
- Defect: `MemoryDbParser`中存储内存数据到DB的方法被命名为`_save_memory_data_to_db`(带前缀下划线
  表示private)，但调用方以public方式`save_memory_data_to_db`调用。当同时启用DB导出和
  profiler_memory时，方法找不到导致功能失败。
- Fix: `_save_memory_data_to_db` → `save_memory_data_to_db`，移除private前缀。
- Reviewability: high -- 方法名不匹配是IDE可检测的基础错误
- Review rule: Python方法重命名时grep所有调用方确认一致性

### D-371: conv3d fp32精度在910B上被不必要地降为fp16

- Hashes: 6aba1934e [+6 cherry-picks: a90a053b6, b8565edde, 6efd72ed7, 6cb91f651,
  97ae53a25, 6025cdce5]
- Root cause: SoC能力未区分(新硬件已原生支持fp32但代码仍走fp16路径)
- Files: torch_npu/utils/_module.py
- Defect: `cast_weight`函数对所有SoC的Conv3d权重执行`.half()`后再转ACL_FRACTAL_Z_3D格式，
  然后`.float()`转回。这在910B和910_93上引入不必要的fp16量化误差，因为这些SoC的aclop已原生
  支持fp32 conv3d。
- Fix: 添加`CONV3D_SUPPORT_FP32_SOC_PREFIX = ["Ascend910B", "Ascend910_93"]`白名单，
  匹配的SoC直接做format cast保持fp32精度，跳过`.half()/.float()`路径。
- Reviewability: medium -- 需要知道哪些SoC支持fp32 conv3d
- Review rule: 精度降级路径必须有SoC能力守卫，新SoC引入时同步更新白名单

### D-372: interactive模式(REPL)下TASK_QUEUE_ENABLE默认开启导致异常

- Hashes: cf14e09e5 [+6 cherry-picks: b7504311a, bd8189ee0, 9b673343d, 2560d9259,
  0672cf7c2, a62d352dd]
- Root cause: 异步任务队列与交互式命令行的即时执行语义冲突
- Files: torch_npu/__init__.py
- Defect: NPU的TASK_QUEUE_ENABLE功能将算子调用放入异步队列批量执行以提高吞吐。但在Python
  交互式session(`sys.ps1`存在时)中，用户期望每条命令立即执行并看到结果。异步队列导致
  结果延迟、错误信息不及时、甚至不可预期的行为。
- Fix: 在`__init__.py`末尾检测`hasattr(sys, 'ps1')`，若为交互模式则强制
  `TASK_QUEUE_ENABLE=0`并打印warning。
- Reviewability: high -- 交互模式与异步执行的矛盾是已知的设计权衡
- Review rule: 异步/批量执行feature必须在interactive模式下有显式fallback

### D-373: ranktable场景global processgroup不一定是第一个创建的

- Hashes: c25460d5d [+6 cherry-picks: 095e0c4c8, 2bc96781a, 087c236fc, e09bff8a8,
  35973013f, d7178a68e]
- Root cause: 全局process group身份判定依赖创建顺序假设
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp,
  torch_npu/csrc/distributed/ProcessGroupHCCL.hpp
- Defect: 代码使用`static std::atomic<size_t> process_group_id`递增计数器，假设`uid_==0`
  的ProcessGroup就是global group。ranktable场景下子group可能先于global group创建，
  导致uid=0被错误分配给子group，global group的comm初始化、rank查询、析构等5处逻辑全部失效。
- Fix: 移除`process_group_id`静态计数器和`uid_`成员，改用`options_->global_ranks_in_group.empty()`
  判断是否为global group（global group的ranks列表为空）。修改5处`uid_ == 0`检查。
- Reviewability: high -- 依赖创建顺序是脆弱假设，代码中已有正确的group属性可用
- Review rule: 不要用全局递增ID推断对象语义，用对象自身属性判断身份

### D-374: profiler DB连接timeout默认值过小

- Hashes: d4d250ce5 [+6 cherry-picks: 5f289ba62, ded1cc36d, 5205b032d, bf3f16638,
  a9db2d23d, 2a97e8dd8]
- Root cause: sqlite3连接默认timeout(5秒)在多进程profiling场景下不足
- Files: torch_npu/profiler/analysis/prof_common_func/_db_manager.py
- Defect: `sqlite3.connect(db_path)`使用默认timeout(5秒)，多进程同时写入profiling数据时
  DB锁竞争频繁，超时后报`database is locked`错误导致profiling数据丢失。
- Fix: 设置`timeout=2147483`(约24.8天，接近int32最大值)，注释说明"set int max to avoid
  database is locked error"。
- Reviewability: high -- sqlite多进程写入的锁超时是已知问题
- Review rule: sqlite多进程写入场景必须设置足够大的timeout，或改用WAL模式

### D-375: UCE错误时dequeue线程未感知导致继续执行

- Hashes: 7d9801c83 [+6 cherry-picks: 5fffefb0a, 13ed17c12, e93776181, 0560a38ca,
  6a6c7d125, 1fddac5f5]
- Root cause: 错误恢复路径与任务队列状态机不匹配
- Files: torch_npu/csrc/core/npu/NPUQueue.cpp, torch_npu/csrc/core/npu/NPUQueue.h
- Defect: UCE(Uncorrectable Error)发生时，原代码在`MakeSureQueueEmpty`和`Enqueue`中通过
  `call_ret`变量延迟检查UCE状态，但`ReadQueue`(dequeue线程)不设置UCE状态，导致dequeue
  线程在UCE后继续执行后续任务，可能触发更多硬件错误。`call_ret`变量在多处被清零(=0)，
  状态丢失。
- Fix: (1) 新增`UCE_EXIT=5`枚举值(在STOP_EXIT之前)；(2) `ReadQueue`检测到
  `ACL_ERROR_RT_DEVICE_MEM_ERROR`时直接`SetStatus(UCE_EXIT)`；(3) `Enqueue`和
  `MakeSureQueueEmpty`检查`UCE_EXIT`状态后立即抛异常；(4) 移除通过`call_ret`延迟
  检查的逻辑。
- Reviewability: medium -- 需理解NPUQueue的生产者-消费者状态机
- Review rule: 错误状态应由检测方直接设置到共享状态机，不要用临时变量跨函数传递

### D-376: task_manager子进程收到SIGINT后不退出导致主进程hang

- Hashes: 28843179b [+6 cherry-picks: 47cfaac58, 423b9c3f9, 4c99b92ac,
  492910ad8, 77bf4a5b6, 71d40df5b]
- Root cause: 子进程缺少信号处理器，SIGINT时未执行cleanup
- Files: torch_npu/profiler/analysis/prof_common_func/_task_manager.py
- Defect: profiler的`ConcurrentTasksManager`启动子进程执行分析任务。主进程Ctrl+C时子进程
  收到SIGINT但无handler，默认行为直接终止而不执行finally块中的cleanup(stop子任务、
  stop进度条)。未stop的子任务持有的资源(文件锁、DB连接)未释放，主进程join时永久阻塞。
  注: 此修复有两套不同的commit message("subprocess bug fix"和"v2.x.x-6.0-rc3-bugfix")，
  summary去重未能合并。
- Fix: (1) `run()`入口注册`signal.signal(signal.SIGINT, self.finalize)`；
  (2) 将finally块中的cleanup逻辑提取为独立的`finalize()`方法，使信号处理器和正常退出
  共用同一cleanup路径。
- Reviewability: high -- 子进程信号处理是并发编程的基本要求
- Review rule: 启动子进程时必须注册SIGINT/SIGTERM handler确保cleanup执行

### D-377: mstx.range_start在stream参数为None时报错

- Hashes: 4736e2ccb [+6 cherry-picks: 8dd144706, 684b30818, c92557a83,
  c7846a67c, a857e60f2, f6c9fa2af]
- Root cause: API参数可选性处理不完整
- Files: torch_npu/npu/mstx.py
- Defect: `mstx.range_start(message, stream=None)`在stream为None时仍尝试`isinstance(stream, Stream)`
  判断，失败后执行`print(Warning, ...)`(错误地将Warning类作为字符串打印而非发出警告)并返回0。
  用户不传stream是合法用法(host-side range)，不应报错。另外`print(Warning, ...)`
  不是正确的warning发出方式。
- Fix: (1) stream为None时调用`_range_start_on_host(message)`走host路径；
  (2) stream不为None且不是Stream类型时用`warnings.warn()`；
  (3) 所有`print(Warning, ...)`改为`warnings.warn(...)`。
- Reviewability: high -- `print(Warning, ...)`是明显的API误用
- Review rule: Python warning必须用`warnings.warn()`而非`print(Warning, ...)`

### D-378: task_manager sleep间隔不合理

- Hashes: f1fe3b1a0 [+6 cherry-picks: f6c9fa2af, 1dc862af7, e98fa36ba,
  4a04b8a4c, b61219127, 5a12e9e72]
- Root cause: 轮询间隔与任务粒度不匹配
- Files: torch_npu/profiler/analysis/prof_common_func/_constant.py,
  torch_npu/profiler/analysis/prof_common_func/_task_manager.py
- Defect: `SLEEP_TIME=0.5`秒，non-blocking任务退出检查使用`sleep(SLEEP_TIME * 2)=1.0秒`。
  对于快速完成的分析任务，1秒等待太长；对于需要更多时间的任务，在检查之前可能还未完成。
- Fix: `SLEEP_TIME`从0.5改为0.1；non-blocking退出检查从`sleep(SLEEP_TIME * 2)`改为
  `sleep(SLEEP_TIME * 5)=0.5秒`。总轮询粒度从1.0秒降到0.5秒，提高响应速度。
- Reviewability: medium -- 需要profiling数据支持timing调优决策
- Review rule: 轮询间隔应根据实际任务延迟分布设定，并在常量中文档化依据

### D-379: UCE内存块地址范围检查逐字节遍历导致O(n*m)且逻辑错误

- Hashes: 500503a16 [+6 cherry-picks: 7b0461755, 5b1c6d26f, 70660968c,
  4b1e2837f, b2ea69db9, 02bcb117b]
- Root cause: 区间重叠判断算法错误(逐字节遍历代替区间比较)
- Files: torch_npu/csrc/core/npu/NPUCachingAllocator.cpp,
  torch_npu/csrc/core/npu/NPUException.cpp
- Defect: `checkUceInMemPool()`用双重循环检查UCE错误地址是否落在分配的Block中：外层遍历
  UCE info数组，内层遍历每个info的`len`字节(逐字节`addr += 1`递增)。问题：(1) 逐字节
  遍历是O(n*m*len)复杂度，len可能很大；(2) `break`在找到第一个匹配block后跳出内层循环，
  遗漏同一UCE地址范围可能跨越多个block的情况；(3) `info.data()`获取的指针在vector resize
  后可能失效。
  另外`clear_mem_uce_info()`逐字段清零而非调用结构体的clear方法。
- Fix: (1) 改为区间重叠判断`addr <= block_end && addr_end >= block_start`，O(n*m)复杂度；
  (2) 不再break，标记所有重叠block为unsafe；(3) `info`直接取值而非取指针；
  (4) `clear_mem_uce_info()`改为`memUceInfo.clear()`。
- Reviewability: high -- 逐字节遍历+break是明显的算法错误
- Review rule: 区间重叠判断用标准公式`a.start <= b.end && a.end >= b.start`

### D-380: public bindings schema.json移除已删除API

- Hashes: 3fc55f6e4 [+6 cherry-picks: 6729cd8da, e599a0c59, d9d36cbf7,
  69f6e6c18, 2bbb5d7f2, 27eb785d1]
- Root cause: 代码API删除后schema.json未同步清理(与D-216/D-239/D-244/D-324/D-357同族)
- Files: test/torch_npu_schema.json
- Defect: `npu_ifmr`、`npu_masked_fill_range`、`npu_normalize_batch`、`npu_rotated_box_decode`、
  `npu_rotated_box_encode`、`npu_scatter`、`npu_stride_add`等7个API已从op-plugin删除，
  但schema.json中仍保留其签名声明，导致public binding测试报告这些API"存在于schema但
  无法import"。
- Fix: 从schema.json中移除对应条目。
- Reviewability: high -- schema.json维护是纯机械性工作，应自动化
- Review rule: 删除API时同步更新schema.json(第六次出现该类问题)

### D-381: copy_操作对conj/neg view创建冗余中间tensor

- Hashes: 584565e44 [+4 cherry-picks: fe6d0e58a, 2836adde0, 67e3c8a9e, 42bb49b2c]
- Root cause: view属性(is_conj/is_neg)处理路径创建不必要的中间分配
- Files: torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp
- Defect: `copy_`函数对complex conjugate和negative view的处理：先创建中间tensor `result`
  (`apply_tensor_without_format`)，对conj view用`aclnnComplex`将real/imag拆分再合并到
  result，对neg view调用`neg()`。这导致每次copy都额外分配一个与src同大小的临时tensor。
- Fix: 先执行copy(d2d/h2d/d2h)，然后对self就地处理view属性：conj view用
  `aclnnComplex(real(self), imag(self).neg(), self)`就地修正；neg view用`self.neg_()`。
  消除了中间tensor分配。
- Reviewability: medium -- 需理解PyTorch view语义和NPU copy路径
- Review rule: copy路径避免创建中间tensor，优先就地(inplace)处理view属性

### D-382: torch_npu.utils模块级import torch_npu导致循环依赖

- Hashes: 27de0972d [+3 cherry-picks: 6479f2196, d08ba2bf5, a693e9669]
- Root cause: 模块级全包import引发循环依赖(与D-45/D-102同族)
- Files: torch_npu/utils/__init__.py
- Defect: `torch_npu/utils/__init__.py`顶部`import torch_npu`，而`torch_npu/__init__.py`
  导入`torch_npu.utils`。当`torch_npu._C`尚未初始化时(如某些import顺序或部分安装场景)，
  `torch_npu._C._flops_count_init()`访问失败，报`module 'torch_npu' has no attribute '_C'`。
- Fix: `import torch_npu` → `from torch_npu import _C`，直接导入所需的`_C`子模块，
  避免触发整个`torch_npu`的初始化链。
- Reviewability: high -- 循环import是Python模块系统的经典问题
- Review rule: 模块内import父包时只import所需的最小子模块，不做全包import

### D-383: dynamic profiler装饰器异常日志级别过高

- Hashes: 55e570797 [+5 cherry-picks: 14ce764d3, ae0db9bce, c7846a67c,
  a857e60f2, 2bac16c78]
  另有相同修复不同summary: 2bbaa7681 ("modify print_error_msg with print_warn_msg")
- Root cause: 容错装饰器将预期的异常报告为错误(误导用户)
- Files: torch_npu/profiler/analysis/prof_common_func/constant.py
- Defect: `_no_exception_func`装饰器设计用于容错——捕获异常并返回默认值以允许profiling继续。
  但捕获到异常时调用`print_error_msg()`，用户看到ERROR级别日志会误以为发生了严重问题，
  实际上这些异常是dynamic profiler在参数错误时的预期行为。
- Fix: `print_error_msg` → `print_warn_msg`，将日志级别从错误降为警告。
  注: 此修复在不同分支上有两个不同的commit message("Bugfix for dynamic profiler when
  set wrong args"和"modify print_error_msg with print_warn_msg")，summary去重未能合并。
- Reviewability: high -- 容错装饰器的日志级别应与其设计意图匹配
- Review rule: 设计为容错的catch块应使用warning而非error级别日志

### D-384: profiler内存报告在多线程allocator中缺失

- Hashes: 6c95f7ac4 [+6 cherry-picks: a8c2ad867, 4640f9081, 2733b9345,
  34d5234f2, 188f2666a, 4709bfd39]
- Root cause: allocator的malloc/free路径未向profiler报告内存事件
- Files: torch_npu/csrc/core/npu/NPUCachingAllocator.cpp,
  torch_npu/csrc/profiler/npu_profiler.h
- Defect: `DeviceCachingAllocator`的`malloc()`和`free()`方法未调用profiler内存报告接口，
  导致profiler在多线程场景下报告NPU内存数据为0或不正确。另外`#include "npu_profiler.h"`
  被错误地放在`#ifndef BUILD_LIBTORCH`守卫内，libtorch构建下profiler头文件不可见。
- Fix: (1) `malloc`和`free`增加`allocator_type`参数，在block分配/释放后调用
  `reportMemoryDataToNpuProfiler()`；(2) 将`#include "npu_profiler.h"`移到
  `BUILD_LIBTORCH`守卫之外（profiler头文件在所有构建模式下都需要）；
  (3) `MemoryUsage`结构体增加`allocator_type`字段区分分配来源。
- Reviewability: medium -- 需理解allocator和profiler的交互
- Review rule: allocator的malloc/free路径必须有profiler hook

### D-385: dynamic profiler共享内存的fd/mmap未关闭

- Hashes: 09d822f98 [+3 cherry-picks: 6b5234e59, e13523e1a, e300f173a]
- Root cause: 资源获取路径与释放路径不对称(fd和mmap_file未保存为成员变量)
- Files: torch_npu/profiler/_dynamic_profiler/_dynamic_profiler_monitor_shm.py
- Defect: `DynamicProfilerShareMemory`在`_connect_or_create()`中通过`os.open()`获取fd，
  再`os.fdopen(fd)`创建file object，再`mmap.mmap()`创建mmap。但fd和file object
  使用局部变量`fd`和`f`，`cleanup()`时只关闭了mmap(`self.shm.close()`)，未关闭底层的
  fd和file object，导致文件描述符泄漏。
- Fix: 将`fd`和file object保存为实例成员变量`self.fd`和`self._memory_mapped_file`，
  `cleanup()`中按顺序关闭：先关mmap，再关file object(或直接close fd)。
- Reviewability: high -- 资源获取和释放的对称性是基本原则
- Review rule: 每个os.open()/os.fdopen()必须有对应的关闭路径，保存为成员变量以确保cleanup可达

### D-386: task非阻塞模式检查使用等号比较而非位运算

- Hashes: 8a908c88a [+5 cherry-picks: 1c5e47a54, 8bbdd6808, 585a5d5a6,
  99ce76914, 756f80bfa]
- Root cause: 枚举值比较方式与枚举设计不匹配(位标志用等号比较)
- Files: torch_npu/profiler/analysis/prof_common_func/_task_manager.py
- Defect: `ConcurrentMode`是位标志枚举(如`NON_BLOCKING = 0x4`)，task的mode可以是
  多个标志的OR组合。退出检查使用`task.mode == ConcurrentMode.NON_BLOCKING`，当mode
  为`NON_BLOCKING | OTHER_FLAG`时比较失败，non-blocking任务被误认为blocking，
  主进程等待其完成而不退出。
- Fix: 添加`is_non_blocking`属性，使用`(self.mode & ConcurrentMode.NON_BLOCKING) != 0`
  位运算检查。
- Reviewability: high -- 位标志枚举使用==比较是经典错误
- Review rule: 位标志枚举检查必须用`&`运算而非`==`

### D-387: profiler临时目录删除路径多了一层父目录

- Hashes: 396877389 [+5 cherry-picks: e30e3712e, 3cdbb1ae1, 104937bd9,
  c32e97836, 5343e0bb8]
- Root cause: 路径操作函数使用不当
- Files: torch_npu/profiler/_profiler_path_creator.py
- Defect: `delete_prof_dir()`使用`shutil.rmtree(os.path.dirname(self._prof_path))`，
  `os.path.dirname()`获取的是父目录而非profiling数据目录本身。这会删除包含其他profiling
  数据的上层目录，造成数据丢失。
- Fix: `os.path.dirname(self._prof_path)` → `self._prof_path`，直接删除目标目录。
- Reviewability: high -- `os.path.dirname()`的语义非常明确，多删一层目录是严重错误
- Review rule: `shutil.rmtree()`的参数必须精确，写代码时打印确认删除路径

### D-388: profiler schema.json type annotation格式不一致

- Hashes: 73c4a1ed9
- Root cause: Python类型注解格式在不同版本间不一致
- Files: test/npu/test_public_bindings.py, test/torch_npu_schema.json
- Defect: schema.json中部分API签名使用`Union[torch.Tensor, NoneType]`格式，而upstream
  PyTorch已统一使用`Optional[torch.Tensor]`格式。两种格式语义相同但字符串不同，导致
  schema比对测试失败(认为签名变更了)。另外缺少新增模块`ge_ir_by_protoc_3_13_pb2`的
  豁免条目。
- Fix: (1) 所有`Union[X, NoneType]` → `Optional[X]`格式统一；
  (2) test_public_bindings.py添加新模块豁免；
  (3) 清理多余空行。
- Reviewability: high -- 类型注解格式应自动化检查
- Review rule: schema.json生成应自动化而非手动维护(与D-380同族)

### D-389: profiler struct编码类型/变量名多重错误

- Hashes: 3e19c937c [+4 cherry-picks: c8b7f454e, 288e41892, 2e649fe40, bd2639659]
- Root cause: 二进制结构体字段类型与上报语义不匹配 + 常量名拼写错误
- Files: torch_npu/csrc/profiler/npu_profiler.h, torch_npu/csrc/toolkit/profiler/inc/data_reporter.h,
  torch_npu/csrc/toolkit/profiler/src/data_reporter.cpp, torch_npu/profiler/analysis/prof_bean/memory_use_bean.py,
  torch_npu/profiler/analysis/prof_bean/torch_op_bean.py
- Defect: (1) `device_index`字段声明为`uint8_t`但NPU设备索引在某些场景可能为负值(如-1表示未分配),
  struct pack格式`<7qb2B2Q`中第2个字段用`B`(unsigned byte)编码有符号值导致解析错误;
  (2) 常量名拼写错误: `SEQUENCE_UNMBER`(应为NUMBER), `FORWORD_THREAD_ID`(应为FORWARD);
  (3) 枚举值`INPUT_SHAPE`应为`INPUT_SHAPES`(与编码函数中的field名不一致)。
- Fix: (1) `uint8_t device_index` → `int8_t device_index`，pack格式`<7qb2B2Q` → `<7q2bB2Q`;
  (2) 修正拼写错误; (3) 枚举名统一为复数形式。
- Reviewability: high -- 类型与格式串不匹配、拼写错误均可通过static analysis检测
- Review rule: struct序列化格式串必须与字段类型声明一一对应，修改字段类型时全文搜索pack/unpack调用点

### D-390: sparse tensor操作顺序错误导致dtype转换失败

- Hashes: 0e72684a5 [+4 cherry-picks: c8b7f454e, 288e41892, 2e649fe40, bd2639659]
- Root cause: tensor操作链中dtype转换与to_sparse的顺序依赖
- Files: test/test_sparse_coo.py
- Defect: `torch.rand(...).to_sparse().npu().to(dtype)` - 先转sparse再移到NPU最后转dtype，
  但NPU sparse tensor的dtype转换实现有限制(或不支持)。正确的顺序是先转dtype再转sparse。
- Fix: `.to_sparse().npu().to(dtype)` → `.to(dtype).to_sparse().npu()`
- Reviewability: medium -- 操作顺序对sparse tensor的语义影响非直觉
- Review rule: sparse tensor的dtype转换应在dense阶段完成，to_sparse应是操作链的后期步骤

### D-391: foreach优化器patch设备查询硬编码device 0 + SoC名称范围不精确

- Hashes: 03be6752b [+3 cherry-picks: 9e98679be, 6a77345b1, 8409a9f13]
- Root cause: patch_supported_devices用hardcoded device(0)且字符串比较上界不够精确
- Files: torch_npu/utils/_optim.py
- Defect: (1) `get_device_name(0)` 在多卡场景下不一定是当前进程使用的设备;
  (2) `device_name < "Ascend910P"` - "Ascend910P"作为上界字符串比较会排除"Ascend910PremiumA"
  (910PremiumA > 910P in lexicographic order)，但910PremiumA实际应支持foreach。
- Fix: (1) `get_device_name(0)` → `get_device_name(torch_npu.npu.current_device())`;
  (2) 上界从`"Ascend910P"` → `"Ascend910PremiumA"`。
- Reviewability: high -- 硬编码device 0是明显错误; 字符串比较SoC名需文档确认全集
- Review rule: 设备名/能力查询必须基于current_device(); SoC名称范围用显式列表而非字符串区间

### D-392: dynamic profiler start接口缺少异常捕获导致崩溃

- Hashes: 46e52c3d1 [+5 cherry-picks: d9f3a49f7, e9fd66af3, 0b42fdf13, c181dcf88, 15580dc01]
- Root cause: 文件读取API可能抛出RuntimeError但调用方未捕获
- Files: torch_npu/profiler/dynamic_profile.py
- Defect: `FileManager.read_json_file(enable_config_path)` 在文件不存在/无权限时抛RuntimeError，
  原代码仅检查返回值为空(None/empty)的情况但未用try-except包裹。当配置文件路径错误时，
  异常未捕获导致整个profiling启动崩溃而非优雅降级。
- Fix: 用try-except RuntimeError包裹read调用，两个错误路径都print_error_msg后return。
- Reviewability: medium -- FileManager内部行为需查阅，但"文件操作需try-except"是通用规则
- Review rule: 所有文件I/O操作必须有异常路径处理，profiler组件不应因配置问题导致主流程崩溃

### D-393: time.sleep抛InterruptedError未捕获 + ConcurrentTasksManager提前退出

- Hashes: 3abf3e513 [+5 cherry-picks: 1e7a9fb79, 5ab9d35cc, 3710b6b96, a80d944c4, 10864804e]
- Root cause: (1) signal中断sleep时抛InterruptedError; (2) 非阻塞任务检查无grace period
- Files: torch_npu/profiler/analysis/prof_common_func/_task_manager.py,
  torch_npu/profiler/analysis/prof_view/cann_parse/_cann_export.py
- Defect: (1) `time.sleep(0.1/0.5)` 在轮询循环中等待CANN解析输出，若进程收到信号(如SIGINT),
  sleep抛InterruptedError导致解析任务异常终止;
  (2) ConcurrentTasksManager检查所有任务是否NON_BLOCKING时，若瞬时check为True就立即退出，
  但任务可能刚注册还未开始执行(检查时机过早)。
- Fix: (1) 所有sleep包裹try-except InterruptedError，中断时返回FAIL;
  (2) 首次检查为True后sleep(1s)再确认一次再退出(double-check pattern)。
- Reviewability: medium -- InterruptedError是Python信号处理的已知陷阱，但非显而易见
- Review rule: 轮询循环中的sleep必须处理InterruptedError; 退出条件需要grace period防止竞态

### D-394: custom op schema中'self'参数名阻止keyword argument调用

- Hashes: dc182c63a [+4 cherry-picks: ba32b2cc8, 9fb9a9f58, d3f1efe2e, 62055febe]
- Root cause: codegen生成schema时保留了method-style的'self'参数名
- Files: codegen/custom_functions.py
- Defect: 自定义算子从class method转为function注册时，func_schema字符串中仍保留`self`
  作为第一个参数名。PyTorch dispatch系统用schema参数名匹配keyword arguments，
  用户调用`torch_npu.fast_gelu(input=x)`时找不到名为'input'的参数(实际是'self')，
  导致keyword argument方式调用失败��
- Fix: 在注册前用`re.sub(r'\bself\b(?=[,\)])', 'input', func_schema)`将schema中的
  'self'替换为'input'。
- Reviewability: medium -- 需理解schema参数名与keyword dispatch的关系
- Review rule: custom op codegen后应有self-test验证keyword argument调用路径

### D-395: profiler内存分析被empty_tensor/malloc_workspace内部操作污染

- Hashes: 256f5bf9a [+8 cherry-picks: f41a765e5, e26600bbd, 5705b3d09, f441e24a6,
  aa40e1cd8, 0395424d4, 836c5a543, a3b3f1466]
- Root cause: 内部runtime操作未从用户可见的torch_op列表中过滤
- Files: torch_npu/profiler/analysis/prof_view/memory_prepare_parser.py
- Defect: `_complete_record_entry`和`_complete_record_entry_for_db`将memory record与torch op
  进行时间窗口匹配时，`empty_tensor`和`malloc_workspace`这两个runtime内部op也参与匹配。
  这些op是allocator内部实现细节，不应出现在用户的memory usage分析中，会导致:
  (1) 用户看到大量无意义的内存事件; (2) 真实用户op的memory归因被稀释。
- Fix: 在匹配前过滤: `torch_ops = [op for op in torch_ops if op.name != "empty_tensor"
  and op.name != "malloc_workspace"]`
- Reviewability: high -- 数据流中的过滤条件是否完整是典型检视点
- Review rule: profiler数据管道中应有明确的"内部op黑名单"维护机制，新增内部op时同步更新

### D-396: profiler fwk_parser缩进错误导致mstx_mark_op影响fwd/bwd追踪

- Hashes: 57c0d892f [+4 cherry-picks: 7d15dea19, 3c3697637, c79eeeac8, 3301c06cf]
- Root cause: Python缩进级别错误(逻辑放在了if-else外部而非else分支内)
- Files: torch_npu/profiler/analysis/prof_parse/fwk_file_parser.py
- Defect: `filter_fwd_bwd_api`和`torch_op_idx += 1`位于if/else结构之外(与if同级缩进)，
  导致mstx_mark_op也会参与fwd/bwd flow追踪和索引递增。正确行为是只有非mstx op才应
  更新fwd/bwd dict和递增索引。结果: fwd/bwd flow中混入mstx标记事件，op索引偏移导致
  后续timeline事件对应关系错���。
- Fix: 将两行代码缩进一级移入else分支。
- Reviewability: high -- 缩进审查是Python代码检视基本项
- Review rule: if/else后的公共逻辑必须确认是否真的应该对所有分支执行

### D-397: C++函数返回TypeError而非torch::TypeError

- Hashes: 186d11b41 [+2 cherry-picks: e5b9af0c6, 88eb05cec]
- Root cause: 命名空间遗漏(未限定torch::前缀)
- Files: torch_npu/csrc/utils/TensorType.cpp
- Defect: `unavailable_type()`返回`TypeError(...)`但应为`torch::TypeError(...)`。
  裸`TypeError`可能解析到Python builtin的TypeError或编译失败(取决于using声明)。
  torch::TypeError是PyTorch的C++ exception类型，用于产生Python TypeError。
- Fix: `TypeError(...)` → `torch::TypeError(...)`
- Reviewability: high -- 编译器应报warning(或在特定编译配置下直接error)
- Review rule: C++异常类型必须使用完整命名空间限定

### D-398: ONNX导出npu_dynamic_quant: None参数处理缺失 + keyword传递错误

- Hashes: ccc88d436 [+2 cherry-picks: 950ba77da, 5a2b9bc37]
- Root cause: autograd.Function的forward/symbolic接口参数设计不匹配实际调用
- Files: torch_npu/onnx/wrapper_onnx_ops.py
- Defect: (1) `forward(ctx, *args, **kwargs)` - 使用*args/**kwargs无法正确将smooth_scales
  作为keyword arg传给底层op(torch.ops.npu.npu_dynamic_quant需要`smooth_scales=`);
  (2) `symbolic`中当smooth_scales为None时直接传None给g.op，但ONNX graph不接受None
  作为输入节点，需要用空tensor常量替代;
  (3) wrapper函数参数名`smooth_scales_dummy`与实际语义不匹配且缺少default=None。
- Fix: (1) forward显式声明参数并用keyword传递; (2) symbolic中None→g.op("Constant",...);
  (3) wrapper参数名统一并添加default。
- Reviewability: medium -- ONNX export的None处理是已知陷阱，但需了解g.op的约束
- Review rule: ONNX symbolic中所有Optional参数必须有None→Constant的转换逻辑

### D-399: rankid→device映射在非均匀集群中错误

- Hashes: cc7805d53 [+3 cherry-picks: 89217a06e, 2e97e2dce, e2b9096e0]
- Root cause: getDeviceForRank假设rankid%numNPUs等于物理设备索引，对非均匀集群不成立
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `getHcclCommName(int rankid)`使用`getDeviceForRank(rankid)`获取设备然后查询
  HCCL comm。在非均匀集群(如不同节点NPU数量不同)中，`rankid % numNPUs`的结果可能
  与当前进程实际使用的device不一致，导致查询到错误的comm(或查不到)。
- Fix: 改用`current_device()`获取实际设备索引。增加`indexFromRank != indexFromCurDevice`时
  的TORCH_WARN_ONCE提示。
- Reviewability: medium -- 需要理解分布式场景中rank与device的映射不一定是简单取模
- Review rule: 分布式代码中rank→device映射不应假设均匀分配，优先使用current_device()

### D-400: torchair lazy init被sys.modules遍历意外触发

- Hashes: 36065a8b2 [+3 cherry-picks: d996ff171, 0d1e0cf59, 7aac6ba20]
- Root cause: Python warnings模块遍历sys.modules调用getattr，触发lazy module初始化
- Files: torch_npu/dynamo/__init__.py
- Defect: torchair作为_LazyTorchair注册在sys.modules中，当Python标准库(如warnings模块)
  执行`for m in sys.modules.values(): getattr(m, '__warningregistry__', None)`时，
  会触发_LazyTorchair.__getattr__进而尝试import torchair，在torchair未安装或初始化
  条件不满足时抛异常。
- Fix: 在__getattr__中增加`_allowed_list`白名单(`__spec__`, `__path__`)，非白名单属性
  在torchair未初始化时直接抛AttributeError而非尝试初始化。同时自定义_pta_error_code()
  避免错误路径再次触发子模块import。
- Reviewability: low -- 需要了解Python module system和warnings模块的内部遍历行为
- Review rule: lazy module代理的__getattr__必须对Python内部属性查询(dunder attrs)有防护

### D-401: asd patch硬编码positional args导致upstream参数变更时break

- Hashes: b2c86d4bf [+4 cherry-picks: b3775c72f, 1af4e3e0b, df7d76564, 7215bee31]
- Root cause: monkey-patch函数使用固定positional参数列表，与upstream签名耦合
- Files: torch_npu/asd/asd.py
- Defect: `_patch_layernorm(input_layernorm, normalized_shape, weight, bias, eps)` 和
  `_patch_embedding(input_embedding, weight, padding_idx, max_norm, norm_type, ...)`
  硬编码了F.layer_norm和F.embedding的全部positional参数。当upstream PyTorch为这些
  函数新增参数时(常见于新版本)，patch函数签名不匹配导致TypeError。
- Fix: 改用`*args, **kwargs`接收可变参数，通过kwargs["weight"]或args[0]获取需要的参数。
- Reviewability: high -- monkey-patch不应假设被patch函数的完整签名
- Review rule: monkey-patch/wrapper函数应使用*args/**kwargs透传，仅提取自己需要的参数

### D-402: numpy bool与`is False`比较永远不满足

- Hashes: 4dd120867 [+4 cherry-picks: 2cd9fd3ad, 3508c211b, 5a37004c0, 0c68ea0f1]
- Root cause: numpy.bool_与Python True/False是不同对象，`is`比较检查对象身份而非值
- Files: torch_npu/testing/testcase.py
- Defect: `if result.all() is False:` - `np.ndarray.all()`返回`numpy.bool_(False)`而非
  Python的`False`。`numpy.bool_(False) is False`永远为`False`(不同对象)，导致该条件
  ���远不触发，bool类型的tensor对比错误时不会报错(静默通过)。
- Fix: `result.all() is False` → `not result.all()`
- Reviewability: high -- `is True/False`与`== True/False`的区别是Python基础知识
- Review rule: 禁止对非singleton对象使用`is True`/`is False`，应使用`not x`或`== False`

### D-403: collect_env.py循环import导致模块初始化失败

- Hashes: d61a20d3d [+3 cherry-picks: 1947ee269, 0d16d477b, 15538503e]
- Root cause: from-import在模块加载时触发被import模块的完整初始化链
- Files: torch_npu/utils/collect_env.py
- Defect: `from torch_npu.utils.path_manager import PathManager` 在collect_env.py
  顶层执行，触发torch_npu.utils模块初始化。但collect_env.py本身也被torch_npu.utils
  引用，形成循环依赖。在某些import顺序下(如直接运行collect_env.py)会导致ImportError。
- Fix: (1) 改为`import torch_npu.utils.path_manager`(延迟绑定);
  (2) torch和torch_npu的import用try-except包裹并设AVAILABLE标志;
  (3) 调用处用全路径`torch_npu.utils.path_manager.PathManager.method()`。
- Reviewability: medium -- 循环import在大型项目中难以在代码审查中发现
- Review rule: 工具/诊断模块(collect_env, profiler)应最小化依赖，避免from-import主包

### D-404: DTensor测试API重命名未同步(schema_suggestions→redistribute_schema)

- Hashes: 6c82b702a
- Root cause: upstream PyTorch重命名DTensor内部API，NPU侧测试未同步
- Files: test/distributed/_tensor/test_common_rules.py
- Defect: PyTorch upstream将`output_sharding.schema_suggestions`重命名为
  `output_sharding.redistribute_schema`，同时返回类型从list变为单个对象
  (不再需要`[0]`索引)。NPU侧测试仍使用旧API名和list索引方式。
  另: `OpSchema`从`op_schema`模块移到`_op_schema`(加下划线前缀)。
- Fix: 更新所有引用: `schema_suggestions` → `redistribute_schema`,
  `suggestions[0].args_schema` → `suggestions.args_schema`,
  `from ...op_schema import OpSchema` → `from ..._op_schema import OpSchema`
- Reviewability: medium -- upstream rename的diff通常很大，需要全局搜索旧名
- Review rule: 每次sync upstream版本后应运行DTensor UT确认API兼容性

### D-405: CI脚本op-plugin测试路径解析错误

- Hashes: 5efeb3711 [+4 cherry-picks: c7cdc0057, ce26ff295, 43f030537, fdb44f540]
- Root cause: get_ut_name()只有一个基于TEST_DIR的路径解析策略
- Files: ci/access_control_test.py
- Defect: op-plugin的UT文件位于`third_party/op-plugin/test/`而非主`test/`目录。
  `get_ut_name()`用`Path(ut_file).relative_to(TEST_DIR)`计算测试名，对op-plugin路径
  会抛ValueError(不是TEST_DIR的子路径)。同时`get_ut_cmd()`也未考虑op-plugin有自己的
  run_test.py。
- Fix: (1) 添加NETWORK_OPS_DIR常量指向op-plugin test目录;
  (2) get_ut_name中检查路径是否包含'op-plugin'并使用对应基目录;
  (3) get_ut_cmd中为op-plugin使用其自身的run_test.py。
- Reviewability: high -- 新增测试目录时CI脚本适配是标准流程
- Review rule: 新增测试子目录时必须验证CI的路径解析和执行命令

### D-406: CI init_method参数错误传递给不接受它的op测试

- Hashes: 9746bcfe6 [+7 cherry-picks: f42a57b11, 331698260, 28c63a98e, fefc27222,
  ebb6df013, dbadcdd8a, 0d40d2bf9]
- Root cause: 命令行参数拼接逻辑缺少条件分支
- Files: ci/access_control_test.py
- Defect: `cmd = cmd + ["--init_method={}".format(init_method)]` 对所有测试类型
  无条件添加init_method参数。但`op_ut_files`类型的测试通过`-k`过滤运行test_ops，
  不接受init_method参数(会被argparse拒绝或产生未定义��为)。
- Fix: 添加`else:`分支，仅非op_ut_files时才追加init_method。
- Reviewability: high -- if分支后的公共代码是否应移入else是基本检视点
- Review rule: 参数拼接必须根据测试类型条件化，新增测试类型时审查参数兼容性

### D-407: dynamic profiler进程退出时profiling未正确停止(资源泄漏)

- Hashes: e17b9cc3e [+5 cherry-picks: 604103557, e51bfb36b, 323da971f, 72cedbf8d, 4ca6ae420]
- Root cause: atexit hook仅清理monitor资源，未停止活跃的profiler实例
- Files: torch_npu/profiler/dynamic_profile.py, torch_npu/profiler/_dynamic_profiler/_dynamic_profiler_monitor.py
- Defect: `init()`注册`atexit.register(self._dynamic_monitor.clean_resource)`，但若进程退出时
  profiling仍在active状态(用户未调用stop)，profiler实例不会被正确停止:
  (1) profiling数据可能不完整(buffer未flush);
  (2) CANN profiler资源未释放(可能影响其他进程);
  (3) 共享内存未清理。同时init()暴露buffer_size/poll_interval参数但无校验，用户可传入
  不合理值。
- Fix: (1) 新增`_clean_resource()`先stop profiler再清理monitor;
  (2) 用类常量`CFG_BUFFER_SIZE=1M`, `POLL_INTERVAL=2`替代用户参数;
  (3) atexit注册改为_clean_resource。
- Reviewability: medium -- 资源清理的completeness需要理解整个生命周期
- Review rule: 注册atexit hook时必须检查所有需要清理的资源(不只是底层resource)

### D-408: codegen functionalization注册取代手写实现

- Hashes: ad2102331 [+2 cherry-picks: 590e2799a, f329f9c44]
- Root cause: 手写functionalization注册文件维护滞后(新增op未注册)
- Files: codegen/gen_backend_stubs.py, codegen/gen_functionalization_type.py (新增258行),
  codegen/templates/RegisterFunctionalization.cpp (新增108行),
  codegen/utils.py, torch_npu/csrc/aten/CustomOpsRegisterFunctionalization.cpp (删除280行)
- Defect: custom op的functionalization dispatch注册通过手写C++文件
  `CustomOpsRegisterFunctionalization.cpp`维护。每次新增custom op需要手动添加对应的
  functionalization wrapper和注册语句。手动维护容易遗漏(特别是对out=和inplace变体)，
  且wrapper代码模式高度重复。
- Fix: 引入codegen自动生成functionalization registration:
  (1) 新增`gen_functionalization_type.py`实现functionalization语义分析;
  (2) 新增`NativeFunctionsGroupOptionalOut`支持可选out变体分组;
  (3) 通过`write_sharded`生成分片C++注册代码;
  (4) 删除手写的280行注册文件。
- Reviewability: low -- 需要理解PyTorch functionalization机制和codegen架构
- Review rule: custom op添加时如果functionalization未自动生成，需检查codegen pipeline

### D-409: profiler syscnt错误哨兵值冲突+条件作用域错误(双重bug)

- Hashes: cb272f655 [+2 cherry-picks: cab2809ee, 7e7e6922d]
- Root cause: 错误返回值与有效值域重叠 + Python缩进导致条件分支错误
- Files: torch_npu/csrc/framework/interface/LibAscendHal.cpp, torch_npu/profiler/profiler.py
- Defect: 两个独立bug叠加:
  (1) C++侧: `ERR_FREQ=1`是有效频率值，导致getFreq()返回错误时profiler仍使用该值
  计算timestamp，产生错误的时间序列。同时`isSyscntEnable()`只检查驱动版本，不检查
  频率是否实际可用;
  (2) Python侧: `self.start_cnt = _get_syscnt()`缩进在`if self.syscnt_enable:`之外，
  导致syscnt不可用时仍调用_get_syscnt()，读取无意义的计数器值。
- Fix: (1) `ERR_FREQ`改为0(不可能的频率值)，`isSyscntEnable`增加`getFreq()!=ERR_FREQ`
  条件; (2) `start_cnt`赋值缩进到`if syscnt_enable:`内部。
- Reviewability: high -- ERR_FREQ=1是明显的哨兵值选择错误，缩进错误也是基本检视点
- Review rule: 错误哨兵值必须在有效值域之外; Python条件块内的赋值语句需核对缩进层级

### D-410: ParallelStore WaitKeys竞态(解锁后注册导致客户端永远等待)

- Hashes: cd163d152 [+1 cherry-pick: bc4b4ff8c]
- Root cause: 先释放锁再注册等待socket，窗口期内key被设置则通知丢失
- Files: torch_npu/csrc/distributed/ParallelTcpStore.cpp
- Defect: `ProcessWaitKeysRequest`中先调用`lockGuard.unlock()`再调用
  `server_->SetKeysWaitingSocket(waitKeys, fd, numKeysToWait)`。在unlock和
  SetKeysWaitingSocket之间，另一个线程可能已经set了被等待的key并尝试唤醒等待者，
  但此时socket尚未注册，通知丢失。被影响的客户端将永远阻塞在wait上。
  当千级客户端同时等待同一key时，此竞态窗口被放大。
- Fix: 将SetKeysWaitingSocket移到unlock之前(持锁注册)。
- Reviewability: high -- lock-then-register是标准并发模式，解锁后注册是明显错误
- Review rule: 事件注册/订阅必须在持锁状态下完成，确保不遗漏通知

### D-411: profiler文件名匹配正则存在ReDoS风险

- Hashes: 4021f65f3 [+6 cherry-picks: 51276f85d, 122abc245, 17c8a582f,
  3d2a69687, f86630b6e, 017110a5e]
- Root cause: 正则表达式使用无界量词`\d+`后接`.*`，构成超线性回溯
- Files: torch_npu/profiler/analysis/prof_parse/cann_file_parser.py
- Defect: CANN_DATA_MATCH字典中所有文件名匹配模式使用`\d+.*\.csv`形式。
  `\d+`后紧接`.*`，当输入字符串包含大量数字后跟非预期字符时，正则引擎
  需要回溯尝试`\d+`和`.*`的所有可能分割方式，导致超线性时间复杂度。
  攻击者可构造恶意文件名触发ReDoS。同时缺少`_slice_`变体的匹配模式。
- Fix: 将`\d+`改为`\d{1,20}`(限定数字长度上界)，同时添加`_slice_`文件变体。
- Reviewability: medium -- ReDoS风险需要正则安全审计意识
- Review rule: 正则量词`+`/`*`后不应紧接另一个量词，应使用`{min,max}`限界

### D-412: profiler数字校验ReDoS+频率除零风险

- Hashes: 572429bd2 [+6 cherry-picks: b76f9419a, d7630f43b, d9102c87e,
  d4c10c546, 0ef083612, 691767459]
- Root cause: is_number()正则无界量词 + get_timestamp_from_syscnt()除零未防护
- Files: torch_npu/profiler/analysis/profiler_config.py
- Defect: 两个安全缺陷:
  (1) `is_number()`正则`r'^[-+]?[0-9]*\.?[0-9]+...'`中`[0-9]*`和`[0-9]+`无上界，
  且未校验输入类型(非str输入导致re.compile异常);
  (2) `get_timestamp_from_syscnt()`中`ratio = time_fmt / self._freq`，当freq接近0时
  产生极大值或除零异常。
- Fix: (1) 添加isinstance(string,str)类型检查，量词改为`{0,20}`/`{1,20}`;
  (2) 添加`abs(self._freq) < 1e-15`除零防护。
- Reviewability: high -- 类型检查和除零防护是基本安全编码要求
- Review rule: 正则使用有界量词; 除法前检查除数; 外部输入先校验类型

### D-413: NPU未初始化时memory API访问段错误

- Hashes: 688839b8f [+8 cherry-picks: d1689e7bb, d1ed33d8f, c1d731578,
  1df22ff7d, 955def62f, 0a304a623, 3b9a4cda1, fafbfb4a8]
- Root cause: memory_stats API无初始化前置检查 + aclrtGetDeviceCount在init中无条件调用
- Files: torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp,
  torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h,
  torch_npu/npu/memory.py,
  torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: (1) `memory_stats_as_nested_dict()`直接调用`_npu_memoryStats(device)`，
  NPU未初始化时访问无效设备导致段错误;
  (2) `memory_allocated()`等函数用`["allocated_bytes.all.current"]`访问dict，
  未初始化时dict为空直接KeyError;
  (3) `Initialize()`中`aclrtGetDeviceCount`在NPU不可用环境下也被调用;
  (4) `repeat_init_acl_flag_`未在构造函数中初始化(UB);
  (5) alltoall_base中int64_t到uint64_t隐式转换(signed/unsigned混合)。
- Fix: (1) memory_stats_as_nested_dict添加is_initialized()检查，返回空dict;
  (2) dict访问改用.get(key, 0)带默认值;
  (3) 移除aclrtGetDeviceCount调用和InitializedDeviceCount方法;
  (4) 构造函数显式初始化repeat_init_acl_flag_;
  (5) 添加static_cast显式转换。
- Reviewability: high -- 初始化检查和dict安全访问是基本防御性编程
- Review rule: 设备相关API调用前必须检查初始化状态; dict用.get()替代[]访问

### D-414: profiler warmup→stop转换动作序列错误+目录未清理

- Hashes: 38c44b9ca [+3 cherry-picks: 89d6d56f2, 84297dda4, 6489cad74]
- Root cause: 状态机转换表中warmup→None路径包含多余的start/stop动作
- Files: torch_npu/profiler/profiler_action_controller.py,
  torch_npu/profiler/profiler_interface.py,
  torch_npu/profiler/profiler_path_creator.py,
  test/profiler/test_action_controller.py
- Defect: profiler状态机从WARMUP直接到None(用户提前停止)时，转换动作列表包含
  `start_trace, stop_trace, finalize_trace`。warmup阶段尚未开始真正trace，
  调用start/stop会生成空的profiling数据文件，且这些空文件不会被清理。
  同时`delete_prof_dir`功能只在`delete_export_only_prof`中使用，
  无法在warmup→None场景中被调用。
- Fix: (1) warmup→None的动作改为仅`finalize_trace, delete_prof_dir`;
  (2) 将`delete_prof_dir`从ProfPathCreator提取为独立方法;
  (3) ProfInterface添加delete_prof_dir接口暴露给controller。
- Reviewability: medium -- 状态机转换表审查需要理解profiler生命周期
- Review rule: profiler状态转换表修改后必须更新测试矩阵并验证每个转换路径的副作用

### D-415: ParallelStore epoll线程启动顺序错误+事件循环缺break

- Hashes: cb0e5f7ce [+1 cherry-pick: 56351782c]
- Root cause: listen线程在client线程初始化前启动 + 事件循环处理后未跳出
- Files: torch_npu/csrc/distributed/ParallelTcpServer.cpp,
  torch_npu/distributed/distributed_c10d.py
- Defect: 三个问题叠加:
  (1) `ctlThread_`(listen fd处理线程)在client线程初始化前启动，新连接到达时
  client线程的epoll fd尚未创建，导致"add connection to epoll failed: Bad file descriptor";
  (2) `LoopProcessListenFd`中处理listen事件后未break，继续遍历events数组中的
  其他fd(listen fd只有一个，后续遍历无意义但浪费CPU);
  (3) barrier轮询间隔0.1s过长，千进程场景下降低同步效率。
- Fix: (1) 将ctlThread_创建移到client线程初始化之后;
  (2) 添加break; (3) sleep从0.1s改为0.01s。
- Reviewability: high -- 线程启动顺序和break遗漏都是常规代码审查可发现的问题
- Review rule: 线程间依赖资源必须在依赖者启动前完成初始化

### D-416: profiler CANN文件解析器缺少slice文件模式匹配+输出目录未清理

- Hashes: 929725c19 [+1 cherry-pick: 7adf8c259]
- Root cause: CANN数据文件模式只匹配原始文件，未覆盖分片(slice)文件变体
- Files: torch_npu/profiler/analysis/prof_parse/cann_file_parser.py,
  torch_npu/profiler/analysis/profiling_parser.py
- Defect: CANN_DATA_MATCH字典只包含`op_summary_\d+.csv`等原始文件模式，
  缺少`op_summary_slice_\d+.csv`等分片文件变体。当CANN生成分片输出(大规模场景)时，
  profiler无法识别和解析这些文件，导致分析结果数据缺失。
  同时`MINDSTUDIO_PROFILER_OUTPUT`目录下的残留数据在重新解析时未被清理。
- Fix: (1) 为所有数据类型添加`_slice_`文件模式;
  (2) 添加`del_output_path_data()`方法清理输出目录;
  (3) 在ProfilingParser解析前调用清理。
- Reviewability: medium -- 需要了解CANN数据输出格式的变体
- Review rule: 文件模式匹配表修改时需覆盖所有已知输出变体(原始+分片)

### D-417: torch.Generator未加入transfer_to_npu白名单

- Hashes: 0d293cfbe [+6 cherry-picks: c389d9210, 15f503e3b, 042ac53be,
  904d48758, fe15a4812, e37770cfa]
- Root cause: transfer_to_npu白名单遗漏Generator类型
- Files: torch_npu/contrib/transfer_to_npu.py
- Defect: `torch_fn_white_list`未包含`"Generator"`。当用户使用transfer_to_npu自动
  迁移代码时，`torch.Generator(device='cuda')`调用不会被拦截替换为NPU设备，
  导致在纯NPU环境下创建CUDA Generator失败。
- Fix: 在torch_fn_white_list末尾添加`"Generator"`。
- Reviewability: high -- 白名单遗漏是transfer_to_npu的已知高频缺陷模式
- Review rule: 新增torch API使用时检查transfer_to_npu白名单是否覆盖(同D-306/D-355)

### D-418: copy_操作不支持broadcast(shape不同时走错快速路径)

- Hashes: 936791087 [+3 cherry-picks: cfe4bfff2, ca4e0120f, c10ab30b2]
  eca968617 [+5 cherry-picks: 4273a6158, c4b5f7d81, 5d4055e11, 19e63315a, fad32851a]
  d607445aa [+2 cherry-picks: 1480d9ace, 9cf6f699a]
- Root cause: H2D/D2H快速路径仅检查dtype和contiguous，未检查shape是否匹配
- Files: torch_npu/csrc/aten/common/CopyKernel.cpp (common path),
  torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp (opapi path),
  test/network_ops/test_copy_.py
- Defect: `copy_h2d_baseformat`和`copy_d2h_baseformat`在判断是否走dtype-contiguous
  快速拷贝路径时，仅检查`same_type && dst_is_contiguous && src.is_contiguous()`。
  当src.shape与dst.shape不同(broadcast场景，如dst=[10,5] src=[5])时，
  仍走直接memcpy快速路径，导致拷贝大小错误(只拷贝src大小的数据，dst剩余部分未填充)。
  两个代码路径(CopyKernel.cpp和CopyKernelOpApi.cpp)存在相同缺陷。
- Fix: 添加`same_size = (src.sizes() == dst.sizes())`检查，快速路径条件增加`&& same_size`。
  非same_size场景走通用expand+copy路径支持broadcast。
- Reviewability: high -- shape检查是copy操作的基本正确性条件
- Review rule: 内存拷贝优化路径必须检查所有前提条件(dtype+contiguous+shape)

### D-419: autograd unpack_list未传saved_for参数(梯度重计算context丢失)

- Hashes: 19968b7be [+2 cherry-picks: 4b902f96c, 1287dcaaa]
- Root cause: codegen模板中unpack_list调用缺少saved_for参数
- Files: codegen/autograd/templates/Functions.h
- Defect: `unpack_list(xs)`和`unpack_opt_list(xs)`签名不接受`saved_for`参数。
  PyTorch的SavedVariable::unpack()在有saved_for时可恢复自动求导图中的引用关系，
  无saved_for时只能返回detached tensor。当NPU custom op的backward使用unpack_list
  解包保存的tensor列表时，缺少saved_for导致梯度计算图断裂。
- Fix: (1) unpack_list添加`std::shared_ptr<Node> saved_for = nullptr`默认参数;
  (2) lambda中使用`x.unpack(saved_for)`替代`x.unpack()`;
  (3) unpack_opt_list同步添加saved_for参数，并处理undefined tensor情况。
- Reviewability: medium -- 需要理解PyTorch autograd的SavedVariable语义
- Review rule: SavedVariable::unpack()调用必须传递saved_for以保持梯度图完整性

### D-420: gather操作无条件dispatch到HCCL忽略其他backend

- Hashes: 8c53a1f72 [+4 cherry-picks: b5ac020e1, e3942bc04, 98bbd70b0, 4d2ad2867]
- Root cause: gather实现硬编码`_get_backend(torch.device("npu"))`
- Files: torch_npu/distributed/distributed_c10d.py
- Defect: NPU的gather覆写无条件通过`group._get_backend(torch.device("npu"))`获取
  HCCL后端，然后调用`allgather`(HCCL不支持gather，退化为allgather)。
  当用户在NPU环境中初始化了非HCCL后端(如gloo)时，gather仍被dispatch到HCCL，
  导致backend不匹配错误。同时allgather语义与gather不同(allgather将结果发送给所有rank，
  gather仅发送给dst rank)。
- Fix: 添加`get_backend(group)`检查:
  (1) hccl后端: 保持原allgather实现(HCCL限制);
  (2) 其他后端: 使用正确的gather语义(GatherOptions + rootRank + output_tensors条件化)。
- Reviewability: medium -- 需要理解collective operation的backend dispatch机制
- Review rule: 分布式操作覆写必须检查实际backend类型，不能假定唯一后端

### D-421: profiler DB查询前未检查表是否存在+方法名拼写错误

- Hashes: 527a7fb9c [+3 cherry-picks: c649b3b09, 891a3a07b, 6e154f9ac]
- Root cause: sqlite表查询未做存在性检查 + dict key拼写错误
- Files: torch_npu/profiler/analysis/prof_common_func/db_manager.py,
  torch_npu/profiler/analysis/prof_view/prof_db_parse/communication_db_parser.py,
  torch_npu/profiler/analysis/prof_view/prof_db_parse/fwk_api_db_parser.py,
  torch_npu/profiler/analysis/prof_view/prof_db_parse/memory_db_parser.py,
  torch_npu/profiler/analysis/prof_view/prof_db_parse/step_info_db_parser.py,
  torch_npu/profiler/analysis/prof_view/prof_db_parse/trace_step_time_db_parser.py
- Defect: 多个DB parser直接执行SQL查询(`select ... from TABLE`)而不检查表是否存在。
  当profiling数据不完整(如仅采集了部分活动)时，某些表可能不存在，直接查询抛异常。
  同时(1) DbManager方法名`judge_table_exit`应为`judge_table_exist`(exit vs exist拼写错误);
  (2) CommunicationDbParser中`TASK_IDS`应为`TASK_INFO`(dict key错误)。
- Fix: (1) 所有DB查询前添加`judge_table_exist()`检查;
  (2) 修正方法名`exit`→`exist`;
  (3) 修正dict key `TASK_IDS`→`TASK_INFO`;
  (4) memory parser添加空record_list提前返回。
- Reviewability: high -- 拼写错误和缺少存在性检查都是基本审查项
- Review rule: sqlite查询前必须检查表存在性; 方法/变量命名提交前拼写检查

### D-422: is_scalar_wrapped_to_device调用已废弃的CalcuOpUtil

- Hashes: c4e2fbe97 [+2 cherry-picks: 8de4db69c, 1a01eb5bd]
- Root cause: API重构后调用方未同步更新
- Files: torch_npu/csrc/framework/utils/OpPreparation.cpp
- Defect: `OpPreparation::is_scalar_wrapped_to_tensor()`调用
  `CalcuOpUtil::IsScalarWrappedToTensor(tensor)`，该方法已被废弃或语义变更。
  CalcuOpUtil实现可能无法正确识别CPU scalar tensor(特别是在新的tensor表示下)。
- Fix: 替换为`IsCPUScalar(tensor)`，使用更精确的判断逻辑。
- Reviewability: high -- API替换是常规重构审查项
- Review rule: 废弃API调用应在废弃时同步更新所有caller

### D-423: aclInit/aclFinalize调用不配对(重复初始化场景下释放不属于自己的资源)

- Hashes: 89da42786 [+2 cherry-picks: 6a86ebe79, c3976d42b]
- Root cause: aclInit重复调用时未区分"自己初始化"和"其他组件已初始化"两种情况
- Files: torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp,
  torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h
- Defect: `NPU_CHECK_ERROR(aclInit(json_path_ptr))`将`ACL_ERROR_REPEAT_INITIALIZE`
  视为错误。但在MindSpore等框架与PTA共存场景中，aclInit可能已被其他组件成功调用。
  PTA无条件调用aclFinalize则会释放其他组件仍在使用的ACL资源。
  遵循"谁申请谁释放"原则，PTA不应在非自己初始化的情况下调用aclFinalize。
- Fix: (1) 添加`repeat_init_acl_flag_`标记;
  (2) aclInit返回`ACL_ERROR_REPEAT_INITIALIZE`时设flag=false(表示非PTA初始化);
  (3) aclFinalize仅在flag=true时调用。
- Reviewability: medium -- 需要理解ACL初始化语义和多框架共存场景
- Review rule: 共享资源的init/finalize必须成对管理，跨组件共享时需ownership标记

### D-424: apply_tensor_use_empty中device_opt可能不是NPU设备

- Hashes: 39e90e16c [+1 cherry-pick: d7cdabe70]
- Root cause: tensor创建路径device参数透传未校验
- Files: torch_npu/csrc/framework/utils/OpPreparation.cpp
- Defect: `apply_tensor_use_empty`直接使用`options.device_opt()`创建NPU tensor，
  但在某些调用路径(如CPU tensor的op preparation)下device_opt可能是CPU设备。
  `NPUNativeFunctions::empty`收到CPU device会创建错误的tensor或崩溃。
- Fix: 检查`device_or_default(device_opt).type()`是否为`PrivateUse1`，
  不是则强制设为`at::Device(c10::DeviceType::PrivateUse1)`。
- Reviewability: high -- op preparation函数应只创建NPU tensor，device校验是基本保证
- Review rule: NPU tensor factory函数必须在入口处校验device类型，不信任上游传入的device_opt

### D-425: task queue退出后Enqueue仅输出泛化日志，无法定位失败操作

- Hashes: 96a6b2d4d [+1 cherry-pick: fee61752f]
- Root cause: 错误日志粒度不足
- Files: torch_npu/csrc/core/npu/NPUQueue.cpp
- Defect: task queue线程退出后再Enqueue时只打印"Task queue thread is exit, cann't call Enqueue()"，
  无法区分失败的是编译执行、异步拷贝还是事件操作，也无任何操作细节(op名称、拷贝大小、event指针)。
  多个op并行enqueue时日志完全相同，诊断效率极低。
- Fix: 根据`paramType`(COMPILE_AND_EXECUTE/ASYNC_MEMCPY/event)分支输出具体操作信息：
  op名称、src/dst长度+kind、event指针。
- Reviewability: medium -- 日志增强通常在问题排查后追加，难以在首次review时要求
- Review rule: 关键路径的错误日志应包含区分并发操作的上下文信息

### D-426: AclSetCompileopt返回值未检查(静默忽略CANN编译选项设置失败)

- Hashes: 0d3bd7d3d [+4 cherry-picks: 13c0602f7, 748c8c3cf, 9bb8d3e0a, 2a4d90cdc]
- Root cause: ACL API返回值被忽略
- Files: torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp,
  torch_npu/csrc/framework/OpParamMaker.cpp,
  torch_npu/csrc/framework/interface/EnvVariables.cpp
- Defect: `AclSetCompileopt()`在~10处调用点返回值未检查或仅用`TORCH_CHECK(ret == ACL_SUCCESS)`
  (格式不统一)。编译选项设置失败时(如CANN版本不支持)静默继续，导致后续op精度/性能异常。
  `EnvVariables.cpp`中部分hook使用`TORCH_CHECK`而另一些直接忽略返回值，风格不一致。
- Fix: 全部统一为`NPU_CHECK_ERROR(AclSetCompileopt(...))`，确保设置失败时报错。
- Reviewability: high -- ACL API调用不检查返回值是标准的review检查项
- Review rule: 所有ACL API调用(aclrt*/aclprof*/AclSet*)必须检查返回值

### D-427: profiler analyse函数max_process_number校验遗漏负数

- Hashes: 74ec44e7b [+4 cherry-picks: 50df00ba2, 6f1cfabd5, d7b9d948b, e42645fd2]
- Root cause: 布尔表达式语义与意图不匹配
- Files: torch_npu/profiler/profiler.py
- Defect: `not isinstance(max_process_number, int) or not max_process_number`中
  `not max_process_number`仅对0返回True(布尔取反)，负数`not -1`=False，通过校验。
  负数`max_process_number`传入后续`os.cpu_count()`比较和`mp.Pool`导致未定义行为。
- Fix: `not max_process_number` → `max_process_number <= 0`。
- Reviewability: high -- `not x`对数值的语义是常见陷阱，值域校验应用显式比较
- Review rule: 数值参数校验使用显式比较(>0, >=0等)，不依赖Python truthiness语义

### D-428: RECORD_FUNCTION在MakeSureQueueEmpty中触发profiler递归调用

- Hashes: 510ef9bb6 [+4 cherry-picks: a8f32bd3a, e09fac2ac, 0bbbc43d8, 25821b6f5]
- Root cause: profiler hook注册在可被profiler自身触发的路径上
- Files: torch_npu/csrc/core/npu/NPUQueue.cpp
- Defect: `MakeSureQueueEmpty()`入口有`RECORD_FUNCTION("MakeSureQueueEmpty", ...)`。
  当autograd.profiler活跃时，RECORD_FUNCTION回调可能触发profiler数据写入，
  数据写入需要确保队列为空（再次调用MakeSureQueueEmpty），形成递归。
  递归深度取决于profiler数据量，可能导致栈溢出。
- Fix: 将`RECORD_FUNCTION(...)`替换为`ASCEND_LOGI("Begin to makesure taskqueue empty.")`，
  断开profiler→queue→profiler递归链。
- Reviewability: medium -- 需理解RECORD_FUNCTION的回调传播链
- Review rule: 基础设施层(task queue/allocator/stream管理)禁止使用RECORD_FUNCTION等profiler hook

### D-429: Revert raise→exit(ImportError被上层import机制静默捕获)

- Hashes: 255e36775
- Root cause: 修复引入回归(raise的传播被import机制捕获)
- Files: torch_npu/__init__.py
- Defect: 原改动将`__init__.py`中libhccl/libascendcl ImportError的处理从
  `sys.exit()`改为`raise`("ensure errors can be captured")。
  但`raise`的ImportError被Python的import机制或上层代码的泛化except捕获，
  导致用户看不到"Please install CANN package"提示信息，而是得到不相关的后续错误。
- Fix: 回退为`traceback.print_exception()`+`sys.exit()`方式，确保诊断信息一定输出到终端。
  将elif链改为独立if块(因每个分支都exit，不需要互斥)。
- Reviewability: medium -- raise语义上更"正确"但实际传播路径难以预测
- Review rule: 入口模块的ImportError处理应考虑被import机制静默捕获的风险

### D-430: profiler memory_record.csv缺少stream_ptr字段

- Hashes: 231c3a474 [+4 cherry-picks: f74bf385b, fbf5cebbc, a032837e3, 9bc54bd1c]
- Root cause: 新增数据字段后row构造未同步
- Files: torch_npu/profiler/analysis/prof_bean/ge_memory_record_bean.py,
  memory_use_bean.py, npu_mem_bean.py, memory_prepare_parser.py, tests
- Defect: memory profiling的row列表缺少`stream_ptr`字段(在active和device_tag之间)。
  GeMemoryRecordBean/NpuMemoryBean的row构造比MemoryUseBean少一列，
  导致不同来源的memory记录列数不一致，CSV合并/解析时列错位。
- Fix: 在所有memory bean的row中添加`stream_ptr`列; NpuMemoryBean.SHOW_HEADERS追加"stream_ptr";
  GeMemoryRecordBean添加`stream_ptr`属性(返回None); MemoryUseBean.row中插入stream_ptr。
- Reviewability: high -- 新增数据字段时检查所有实现该协议的bean类是标准审查项
- Review rule: profiler数据bean添加字段时，检查所有同接口的实现类并同步

### D-431: Ascend Profiler多处修复(event类型命名/路径检查/profiler报告)

- Hashes: 2b301732c [+4 cherry-picks: a73a2edb4, b88cd9c08, 7fd472027, 7f1d4e416]
- Root cause: 多问题叠加(event MAP语义错误 + DataDumper启动路径多余 + report函数参数错误)
- Files: torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.cpp,
  torch_npu/csrc/framework/utils/NpuUtils.cpp, torch_npu/csrc/toolkit/profiler/src/data_dumper.cpp
- Defect: (1) EVENT_PARAS_MAP用allocator类型(HOST_ALLOCATOR_EVENT等)做key，
  但实际需要按操作语义(RECORD_EVENT/WAIT_EVENT/LAZY_DESTROY_EVENT)命名，
  导致profiler显示的event名称是分配器类型而非操作类型。
  (2) DataDumper::Start()先检查目录创建再启动线程，但CreateDir应延迟到实际写入时。
  (3) DqueueEvent取event详情时解析paramVal，但event的名称应从paramType获取。
- Fix: (1) EVENT_PARAS_MAP改为操作语义命名;
  (2) DataDumper::Start()移除CreateDir前置检查;
  (3) DqueueEvent直接用`para->paramType`索引MAP。
- Reviewability: medium -- event类型命名错误需对照profiler输出才能发现
- Review rule: profiler展示的名称应反映操作语义而非内部分配器类型

### D-432: Level0 profiler配置与aic_metrics冲突时仅警告不重置

- Hashes: 2ad395251 [+3 cherry-picks: 09650ba29, 6a8626d18, 04a5a9ce4]
- Root cause: 配置校验只发现矛盾未解决矛盾
- Files: torch_npu/profiler/experimental_config.py
- Defect: `_check_params()`检测到`profiler_level==LEVEL0`且`aic_metrics!=None`时
  仅打印警告"Please use level1 or level2"，但未将`_aic_metrics`重置为`None`。
  用户看到警告后若不修改配置重新运行，Level0下请求aic数据会导致CANN底层错误。
- Fix: 警告后追加`self._aic_metrics = Constant.AicMetricsNone`强制重置。
  警告文案补充"reset aic metrics to None!"。
- Reviewability: high -- 配置冲突校验后应恢复到一致状态是基本模式
- Review rule: 配置校验发现冲突后必须自动修正到安全状态，不能仅警告

### D-433: layernorm测试将rstd误作variance使用

- Hashes: ff535c80a
- Root cause: native_layer_norm返回值语义理解错误
- Files: test/custom_ops/test_npu_fused_attention_layernorm_qkv_fwd.py,
  test/network_ops/test_layer_norm.py
- Defect: (1) `torch.native_layer_norm`返回`(norm, mean, rstd)`，
  测试将第三个返回值直接作为`variance`使用。rstd是`1/sqrt(var+eps)`而非var，
  导致与NPU算子输出的variance比对始终不精确。
  (2) test_layer_norm.py中`torch.ops._caffe2.LayerNorm`调用在新版PyTorch中不可用。
- Fix: (1) 将variance改用`torch.var(ln_input, -1, keepdim=False, unbiased=False)`独立计算;
  (2) 删除caffe2 LayerNorm相关测试代码。
- Reviewability: high -- API返回值语义错误可通过文档对照发现
- Review rule: 测试中使用多返回值API时必须对照文档逐个确认每个返回值的语义

### D-434: multi-head attention测试中NPU dropout backward不兼容NZ格式

- Hashes: 6aee2edb3 [+3 cherry-picks: 60cea2181, e18b6eaa5, 8034fad47]
- Root cause: 测试直接调用底层NPU op未处理格式转换
- Files: test/network_ops/test_multi_head_attention.py
- Defect: 测试直接调用`torch_npu._npu_dropout(input, p)`，
  但backward路径中`npu_dropout_do_mask`需要NZ格式输入。
  forward输出是ND格式，梯度也是ND格式，直接传入backward导致格式不匹配错误。
  同时softmax也需要NZ格式输入才能正确反向。
- Fix: 封装自定义`DropoutApply(autograd.Function)`，在forward中返回dropout mask，
  在backward中显式调用`npu_format_cast(grad, FORMAT_NZ)`后再调用`npu_dropout_do_mask`。
  softmax输入也加`npu_format_cast`。移除`unittest.skip`。
- Reviewability: low -- NZ格式要求是NPU特有的隐式约束，不在API文档中
- Review rule: NPU自定义op测试需确认forward/backward的tensor格式要求

### D-435: lazy_init中fork后重置用错误模块(torch._C而非torch_npu._C)

- Hashes: b8350e5bc [+3 cherry-picks: 1b6c0a4e4, f9718497c, f249faa37]
- Root cause: C扩展模块引用路径错误(torch vs torch_npu)
- Files: torch_npu/npu/__init__.py
- Defect: `_after_fork()`回调中调用`torch._C._npu_set_run_yet_variable_to_false()`，
  但`_npu_set_run_yet_variable_to_false`是torch_npu的C扩展函数，注册在`torch_npu._C`中，
  torch._C中不存在该函数。fork后子进程调用失败，NPU运行标志未重置，
  后续操作不会触发重新初始化而是使用父进程的陈旧状态。
- Fix: `torch._C` → `torch_npu._C`。
- Reviewability: high -- 模块路径错误在测试中应触发AttributeError
- Review rule: C扩展函数调用必须确认其注册在正确的模块中(torch._C vs torch_npu._C)

### D-436: GuessFormatWhenContiguous对FakeTensor缺空指针防护

- Hashes: 5aafb85f7 [+1 cherry-pick: 54b430a01]
- Root cause: FakeTensor模式下storage为空但代码无条件访问npu_desc_
- Files: torch_npu/csrc/framework/InferFormat.cpp
- Defect: `GuessFormatWhenContiguous()`直接调用
  `NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_`，
  但FakeTensor的storage_impl的data_ptr为nullptr，npu_desc_未初始化。
  访问未初始化的npu_desc_的origin_format_字段是UB(取决于内存内容)。
  torch.compile/FakeTensorMode下会触发。
- Fix: 先检查`tensor_storage_impl->data_ptr() == nullptr`，是则直接返回`ACL_FORMAT_ND`。
- Reviewability: high -- FakeTensor路径是已知的需要防护的模式
- Review rule: 所有访问NPU storage_impl的代码必须检查FakeTensor场景(data_ptr==nullptr)

### D-437: step trace time解析边界事件被遗漏(严格不等号)

- Hashes: c047c84cb [+3 cherry-picks: d71ba0ea1, fd1d4c33a, 00a41c50a]
- Root cause: 区间判断用开区间而非闭区间
- Files: torch_npu/profiler/analysis/prof_view/trace_step_time_parser.py
- Defect: `step[1] < addtime and step[2] > addtime`是开区间(step_start, step_end)，
  恰好落在step边界的事件(addtime == step[1] 或 addtime == step[2])不属于任何step。
  这些边界事件的step归属为None，导致trace view中出现未归类事件。
- Fix: `step[1] <= addtime <= step[2]`改为闭区间。
- Reviewability: high -- 区间边界开/闭是经典审查点
- Review rule: 时间区间判断默认使用闭区间[start, end]，开区间需显式注释理由

### D-438: ERR_FREQ哨兵值1与有效频率值域重叠

- Hashes: 5b597d3aa [+3 cherry-picks: de389f100, 68f7edf64, f4bac8be5]
- Root cause: 错误哨兵值选择在有效值域内
- Files: torch_npu/csrc/framework/interface/LibAscendHal.cpp
- Defect: `ERR_FREQ=1`用作getFreq()失败时的返回值，但1(MHz)是合法的低频值。
  调用方无法区分"频率查询失败"和"频率为1MHz"。同时getFreq()/getVer()的逻辑结构
  在函数指针获取失败和调用失败两种情况下走相同路径但语义不同。
  `RES_OK=0`与`DRV_ERROR_NONE`含义相同但命名不一致。
- Fix: `ERR_FREQ`改为0(明确的无效频率)；重构为先获取函数指针并立即检查，
  成功后在单一条件中检查`ret == DRV_ERROR_NONE && freq > 0`。
  `RES_OK`重命名为`DRV_ERROR_NONE`与驱动API命名一致。
- Reviewability: high -- 错误哨兵值应在值域之外是基本原则
- Review rule: 错误返回值/哨兵值必须在正常值域之外(如0、-1、MAX_INT)

### D-439: ONNX NPULstm op参数名input应为inputs(与函数签名不匹配)

- Hashes: a3e45942c
- Root cause: 变量名拼写错误(input vs inputs)
- Files: torch_npu/onnx/wrapper_onnx_ops.py, test/onnx/test_wrapper_onnx_ops.py
- Defect: `NPULstmOP.symbolic()`的函数签名参数为`inputs`(复数)，
  但`g.op("npu::NPULstm", input, ...)`中误写为`input`(单数)。
  Python中`input`是内置函数，不会报NameError，但传入g.op的是内置函数对象而非tensor，
  导致ONNX导出时类型错误或序列化失败。
- Fix: `input` → `inputs`。同时调整测试skip条件(geglu限910B、lstm移除skip、
  dropout_with_add_softmax添加skip)。
- Reviewability: high -- 函数参数名与使用处不一致可通过IDE/linter检测
- Review rule: 函数体中使用的变量必须与签名参数名完全匹配(Python shadowing内置名时尤其注意)

### D-440: upstream设备名privateuse1重命名为privateuseone未同步

- Hashes: 8131e218b
- Root cause: upstream PyTorch设备名常量变更后测试未适配
- Files: test/test_ops_fwd_gradients.py, test/test_ops_gradients.py
- Defect: PyTorch 2.2将`only_for='privateuse1'`参数改为`only_for='privateuseone'`。
  `instantiate_device_type_tests(..., only_for='privateuse1')`在新版PyTorch中
  匹配不到任何设备，导致所有gradient测试静默跳过(0 tests run)。
- Fix: `'privateuse1'` → `'privateuseone'`。
- Reviewability: high -- upstream常量变更应在版本适配PR中检查
- Review rule: upstream PyTorch版本升级后，grep所有设备名/后端名相关字符串常量

### D-441: npu_ffn meta函数输出shape仅保留2D(丢弃高维batch dims)

- Hashes: bee31a324 [+1 cherry-pick: 6f0e61ad6]
- Root cause: meta函数假设输入是2D
- Files: torch_npu/meta/meta_registrations.py
- Defect: `npu_ffn_meta`的输出shape固定为`(x.size(0), weight2.size(1))`，
  仅适用于2D输入。当x是3D(batch, seq, hidden)或更高维时，
  丢失了除最后一维外的所有维度信息。torch.compile在shape推理阶段得到错误的输出shape，
  导致后续op的shape不匹配。
- Fix: 遍历x的前n-1维构建dim_list，最后一维取`weight2.size(weight2.dim()-1)`。
  输出shape从`(x.size(0), weight2.size(1))`变为`(*x.shape[:-1], weight2.shape[-1])`。
- Reviewability: high -- meta函数的shape推理应考虑任意维度输入
- Review rule: meta函数的shape推理不能硬编码维度数量，应处理任意rank输入

### D-442: 分布式测试多处修复(dtype/world_size/import/device count)

- Hashes: 3d578506b [+3 cherry-picks: d5ca7a810, fdc6327d4, 0d8208d44]
- Root cause: 测试环境假设与实际不匹配(多处)
- Files: test/distributed/test_allgather.py, test_allreduce.py, test_distributed.py, test_reduce.py
- Defect: (1) allgather测试未跳过bool dtype(HCCL不支持bool allgather)；
  (2) allreduce uint8测试循环嵌套顺序错误(shape和world_size交叉)；
  (3) test_distributed.py从torch_npu.testing导入TestCase但应从torch.testing导入；
  (4) SyncBatchNorm测试需4卡但未加skipIfUnsupportMultiNPU(4)；
  (5) reduce测试硬编码skipIfUnsupportMultiNPU(8)但实际只需2卡+动态检查。
- Fix: 逐一修复：(1)添加bool continue; (2)修正循环顺序; (3)修改import;
  (4)添加skip装饰器; (5)将skip(8)改为skip(2)+循环内device_count检查。
- Reviewability: medium -- 多为环境适配问题，需在目标环境运行才能发现
- Review rule: 分布式测试的world_size应匹配实际可用设备数，使用动态检查而非硬编码

### D-443: MultiProcessDataLoader iter初始化时synchronize抛"context is empty"

- Hashes: 5571f271a [+5 cherry-picks: be2cb99c4, ca1a86059, 7d10ee526, 516d978a1, 7092169a8]
- Root cause: monkey-patch的synchronize在DataLoader worker进程中无NPU context
- Files: torch_npu/utils/module.py
- Defect: `mpdl_iter_init`中`torch_npu.npu.synchronize()`在DataLoader的worker子进程中调用，
  但worker进程可能未初始化NPU context(纯CPU数据加载场景)。
  此时synchronize抛出RuntimeError("context is empty")终止DataLoader初始化。
- Fix: 用bare`try/except: pass`包裹synchronize调用。
- Reviewability: medium -- DataLoader worker的NPU context状态不显然
- Review rule: monkey-patch到通用框架路径上的NPU操作必须容忍NPU未初始化场景

### D-444: contrib/__init__.py的__all__列表缺少逗号(隐式字符串拼接)

- Hashes: 458833773
- Root cause: Python隐式字符串拼接
- Files: torch_npu/contrib/__init__.py
- Defect: `__all__`列表中`"FusedColorJitter"`后缺少逗号：
  `"FusedColorJitter" "LinearWeightQuant"`被Python隐式拼接为
  `"FusedColorJitterLinearWeightQuant"`。结果：(1)__all__中出现一个不存在的名称；
  (2)FusedColorJitter和LinearWeightQuant都不在__all__中，`from torch_npu.contrib import *`
  无法导入这两个类。
- Fix: 添加逗号`"FusedColorJitter",`。
- Reviewability: high -- 静态分析工具(ruff, flake8-implicit-str-concat)可检测
- Review rule: Python列表/元组中每个字符串元素后必须有逗号，启用implicit-str-concat lint规则

### D-445: Bug Fix For Step Id
- Hashes: a7fbb1ec3 [+3: 4471e772d, 7302a7781, 9bf033dec]
- Root cause: dict key存在性未验证(corr_id不在kernel_dict中)
- Files: torch_npu/profiler/analysis/prof_parse/fwk_cann_relation_parser.py
- Defect: `get_step_range`中直接取`corr_id_total`的min/max查`kernel_dict`，但极值corr_id可能
  不在dict中(kernel未被profiler捕获)，导致拿到空列表，step_id信息丢失。同时缺少
  `kernel_dict`和`corr_id_total`为空的前置检查。
- Fix: 对`corr_id_list`排序后从两端向内遍历，找第一个在dict中有非空结果的corr_id；新增空检查。
- Reviewability: medium -- 需要理解corr_id与kernel的对应关系不一定完整
- Review rule: dict.get()返回空列表/None时，调用方必须有fallback逻辑

### D-446: Catch and process errors occurred during the profiling parsing phase
- Hashes: 43ac5c9d6 [+3: b94b93c06, a37b47fb7, a3d5c2c7e]
- Root cause: 异步任务调度循环缺异常捕获
- Files: torch_npu/profiler/analysis/prof_common_func/task_manager.py
- Defect: `ConcurrentTasksManager.run()`主调度循环无try/except，解析过程中任何异常会导致进度条
  和资源清理逻辑被跳过，表现为profiler卡死或资源泄漏。
- Fix: 主循环包进try/except/finally，异常时打印错误，finally保证未完成task被stop、进度条关闭。
- Reviewability: high -- 基础的异常安全模式
- Review rule: 管理外部资源(进度条/线程池/文件)的调度循环必须有finally清理

### D-447: HCCL_BLOCKING_WAIT fails on master and 2.1
- Hashes: f82a64769 [+1: 0ef45ac03]
- Root cause: blocking wait超时后不abort communicator直接抛异常
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp, .hpp
- Defect: 原`synchronize()`直接做blocking wait，超时后`checkAndThrowException`抛异常，
  但未先abort HCCL communicator，导致后续NPU事件无法正常执行(communicator处于不一致状态)。
- Fix: 拆为公开`synchronize()`和私有`synchronizeInternal(timeout)`；超时时先记录日志+break，
  有异常则先`abort()` communicator再`handleException(TearDown)`。
- Reviewability: medium -- 需理解HCCL communicator状态机的不变量
- Review rule: 分布式通信超时/异常路径必须先abort communicator再抛异常

### D-448: Bug Fix For Memory Match (profiler多处小bug聚合)
- Hashes: 1ad120683 [+3: 8e81f0878, 6cf6e855a, 0291c8bbc]
- Root cause: profiler多处防御性检查缺失+typo
- Files: constant.py, cann_export.py, memory_prepare_parser.py, profiling_parser.py
- Defect: 4处问题聚合修复:
  (1) PROF_WARN_SIZE=400MB过小，正常profiling数据就超限告警
  (2) "faile"→"failed" typo
  (3) `matched_torch_op.parent_node`可能返回无`event`属性的根节点→AttributeError
  (4) `run_parser()`异常直接抛出，跳过后续simplification和时间打印
- Fix: (1)阈值调为1GB (2)修typo (3)加`not matched_torch_op.event`判断 (4)加try/except降级
- Reviewability: high -- 均为基础防御性编程
- Review rule: 树遍历到根节点时必须检查根节点是否有子节点才有的属性

### D-449: Change data type of reserved memory from int to float
- Hashes: 70c2df915 [+3: 46d91c447, 917a6c53b, b299a2018]
- Root cause: 数值类型精度丢失(int截断浮点)
- Files: torch_npu/profiler/analysis/prof_bean/npu_module_mem_bean.py
- Defect: `NpuModuleMemoryBean.__init__`中`total_reverved`用`int()`转换，除以`KB_TO_MB`
  后小数部分被截断，"Total Reserved(MB)"列显示为截断整数(如1023→0)。
- Fix: `int(...)`改为`float(...)`。
- Reviewability: high -- 单行修改，int/float精度差异是基础知识
- Review rule: 涉及除法的数值计算，结果类型应为float而非int

### D-450: fixed c1fa6aa / torch_npu.utils.get_cann_version()
- Hashes: f34342ed0
- Root cause: 同一功能(get_cann_version)存在两份不一致的实现
- Files: npu_intercept.py, collect_env.py, module.py, __init__.py
- Defect: `npu_intercept.py`有本地`get_cann_version(ascend_home_path)`实现，
  `collect_env.py`也有一份但参数不同(无参，从环境变量读路径)。两者行为不一致，
  且`torch.version.cann`属性未暴露，`torch_npu.utils.get_cann_version()`不可用。
- Fix: 删除重复实现，统一导入`collect_env.get_cann_version()`；新增`add_collect_env_methods()`
  将结果赋值给`torch.version.cann`。同时清理了3个不再需要的SystemEnv字段和对应函数。
- Reviewability: high -- 重复实现是代码搜索即可发现的问题
- Review rule: 新增工具函数前先搜索是否已有同功能实现

### D-451: Fix the ut of im2col_backward
- Hashes: b3ca7202a
- Root cause: 函数调用参数个数错误
- Files: test/network_ops/test_im2col_backward.py
- Defect: `create_common_tensor((dtype, 0, shape), 0, -100, 100)`多传了一个参数`0`，
  不匹配函数签名`create_common_tensor(item, minvalue, maxvalue)`。
- Fix: 去掉多余的`0`参数。
- Reviewability: high -- 参数个数不匹配，IDE/类型检查即可发现
- Review rule: 调用工具函数时确认参数个数和语义与签名一致

### D-452: Bugfix for calling aclprofDestroyConfig
- Hashes: b0cf83e37 [+3: 824c63d67, b8323d960, 0aaaeb0c2]
- Root cause: ACL资源创建后释放配对缺失(profConfig)
- Files: e2e_profiler.cpp, profiler_mgr.cpp
- Defect: `AclProfilingStop()`后未调用`AclProfilingDestroyConfig(profCfg)`，profConfig对象
  泄漏。两条profiler路径(e2e_profiler和profiler_mgr)均缺此释放调用。
- Fix: Stop后补齐`AclProfilingDestroyConfig()`调用+指针置空。
- Reviewability: high -- 资源创建/释放配对是基础模式
- Review rule: ACL资源创建(aclprofCreateConfig)必须有对应Destroy调用，CR时成对检查

### D-453: Fix the failed UT (numpy deprecation + rsub semantic)
- Hashes: 5e32ac1bd [+3: d35164a2c, 5af6ad745, 13461ae3b]
- Root cause: numpy废弃别名+算子语义不正确+冗余skip
- Files: test_confusion_transpose.py, test_rsub.py, test_scaled_masked_softmax.py
- Defect: 多处测试问题:
  (1) `np.int`→废弃，应使用`np.int_`
  (2) rsub测试用`input2 - input1`代替`torch.rsub(input1, input2)`，语义不一致
  (3) 多个测试方法有冗余`@unittest.skip`和手动dtype cast
- Fix: numpy别名替换；rsub测试改用`torch.rsub`；移除冗余skip和手动cast。
- Reviewability: high -- numpy deprecation warning是CI可检测的
- Review rule: 测试中的算子调用应使用被测算子本身而非等价数学表达式

### D-454: fix subthread destroy event fail bug
- Hashes: ab65d02a6 [+6: 59e29f4e4, c5ba2a7a4, 1eec95714, 28020069b, 9dc04638c, 156c517b2]
- Root cause: 跨线程设备查询返回错误设备号
- Files: NPUEvent.h, NPUCachingAllocator.cpp/.h, NPUStream.cpp, AsyncTaskQueueInterface.cpp/.h
- Defect: NPUEvent析构时调用`aclrtGetDevice()`获取当前设备号，但子线程的当前设备可能与event
  创建时不同，导致`LaunchLazyDestroyEventTask`入错设备的任务队列，事件销毁失败。
- Fix: 析构时直接传入event创建时保存的`device_index_`，不再动态查询。删除不再需要的
  `NpuAllocatorEraseRecordedEvent`和多余的`NPUGuard`。
- Reviewability: medium -- 需理解多设备场景下线程与设备亲和性
- Review rule: 析构函数中禁止调用`aclrtGetDevice()`等依赖线程状态的API，应使用创建时保存的设备信息

### D-455: RMSNorm meta函数修复
- Hashes: e9a6d2bc8 [+1: bb2da3e9a]
- Root cause: meta函数device='meta'缺失+参数顺序错误
- Files: torch_npu/meta/meta_registrations.py, test/test_fake_tensor.py
- Defect: 两处问题:
  (1) `npu_rms_norm_meta`中`rstd = torch.empty(ret, dtype=torch.float32)`缺少
  `device='meta'`，FakeTensorMode下在真实设备分配tensor
  (2) backward meta函数参数顺序错误:`(dy, self, rstd, gamma)`应为`(dy, self, gamma, rstd)`
- Fix: 补`device='meta'`；修正参数顺序；新增测试覆盖forward和backward。
- Reviewability: high -- meta函数必须在meta device上创建tensor是基本规则
- Review rule: meta函数中所有torch.empty/torch.zeros必须显式指定device='meta'

### D-456: Solve the problem that the profiling is stuck in specific scenarios
- Hashes: 967a0650e [+3: 4f18e3d11, 65b599d53, 68c1fdc21]
- Root cause: multiprocessing start method与场景不匹配
- Files: torch_npu/profiler/analysis/npu_profiler.py
- Defect: profiler进程池默认使用系统start method(可能是spawn)，在某些场景下spawn模式会卡死
  (因为子进程需要重新import整个模块，触发NPU初始化等副作用)。
- Fix: 在创建进程池前强制`multiprocessing.set_start_method("fork", force=True)`。
- Reviewability: low -- 只在特定系统配置+场景组合下触发
- Review rule: 使用multiprocessing时显式指定start method，不依赖系统默认值

### D-457: bugfix, modify aclnn route logic
- Hashes: a0aec26fe [+4: 9741809eb, 3c3dc2dd3, c8f84d7c5, f684df52c]
- Root cause: 基础格式判断条件逻辑错误
- Files: torch_npu/csrc/framework/FormatHelper.cpp
- Defect: `IsBaseFormatType()`用`origin_format_ == npu_format_`判断是否为基础格式，
  但连续内存的NZ格式也可能满足此条件(origin和npu都是NZ)，导致非基础格式tensor
  走aclnn路径时格式判断错误。
- Fix: 改为直接枚举合法基础格式集合:`ACL_FORMAT_ND || NCHW || NHWC || NCDHW`。
- Reviewability: medium -- 需理解NPU format origin/npu的语义差异
- Review rule: 格式判断应基于白名单枚举而非属性比较，因为属性相等不等价于基础格式

### D-458: fix the bug of schedule (profiler状态机)
- Hashes: 66df92d75 [+3: cdf443718, 2c2f0bc09, 82ded5534]
- Root cause: profiler状态机缺少stop后防护
- Files: torch_npu/profiler/profiler.py
- Defect: schedule未走完一个完整周期就调用stop()时行为未定义；stop后继续调用step()会导致
  状态机进入非法转换。metadata在dump后未清空导致重复写入。
- Fix: 新增`stopped`标志：stop()设True，start()重置False，step()检查stopped后return；
  `_dump_metadata`末尾清空metadata；非预期的调度状态转换输出警告信息。
- Reviewability: medium -- 状态机边界case需系统性测试
- Review rule: 状态机的stop/start/step必须有幂等性保证，stop后的step应为no-op

### D-459: fix asyncronize error between HCCL stream and computation stream
- Hashes: 86d1674ce [+2: e929dfab4, 315f762dd]
- Root cause: 分布式pre/post handler在错误的stream上执行
- Files: ProcessGroupHCCL.cpp, test/distributed/test_allreduce.py, test_reduce.py
- Defect: allreduce/reduce的预处理(bool/byte→int cast)和后处理(int→原dtype cast)lambda
  在computation stream上执行而非HCCL stream，导致cast操作与HCCL通信存在异步竞争：
  通信可能使用未cast完成的buffer，或后处理在通信完成前执行。
- Fix: 在pre/post handler lambda中加入`NPUStreamGuard guard(hcclStreams[0])`，确保
  dtype cast在HCCL stream上执行。新增scalar size=[1]的测试用例覆盖此场景。
- Reviewability: medium -- 需理解多stream执行模型和HCCL流依赖
- Review rule: 分布式通信的pre/post处理必须在对应的HCCL stream上执行

### D-460: Bugfix for Ascend PyTorch Profiler (dump触发条件)
- Hashes: 0681cc7e7 [+3: a5b9c007e, 628574837, 707f2dee8]
- Root cause: 取模判断的零值边界(0 % N == 0)
- Files: torch_npu/csrc/toolkit/profiler/src/data_dumper.cpp
- Defect: `DataDumper::Run()`中`data_chunk_buf_.Size() % kNotifyInterval == 0`作为dump
  触发条件，当buffer为空(Size==0)时也满足条件(0%N==0)，导致空buffer触发无意义dump；
  且buffer未满时也不规律地触发。
- Fix: 改为阈值比较`Size() > kNotifyInterval`，只有buffer积累超阈值才触发。
- Reviewability: high -- 取模的零值边界是经典陷阱
- Review rule: 用取模做周期性触发时，必须额外检查值为0的情况(0%N==0)

### D-461: Bugfix for losing Profiling data occasionally
- Hashes: c6c30fc37 [+2: efbbcb9f8, e27f8d27c]
- Root cause: DataDumper Flush/Stop生命周期顺序错误
- Files: profiler_mgr.cpp, ring_buffer.h, data_dumper.h/.cpp, data_reporter.h
- Defect: profiler_mgr中先Flush()再Stop()，但Flush()只写当前buffer而Stop()会停止
  dump线程，导致Flush和Stop之间产生的数据丢失。RingBuffer析构不完整。
- Fix: Stop()改为停线程后再调Flush()；RingBuffer新增UnInit()完整清理；DataDumper拆出
  GatherAndDumpData()辅助函数；profiler_mgr改为先Stop()再UnInit()。
- Reviewability: medium -- 生命周期顺序需整体理解数据流
- Review rule: 数据pipeline的stop序列应为: 停止生产→flush缓冲→停止消费→释放资源

### D-462: fix the bug introduced while solving conflicts
- Hashes: 7eda0337d [+3: c494718ca, 38f5aee34, 4631210fc]
- Root cause: 合并冲突解决时引入类型/单位错误
- Files: fwk_cann_relation_parser.py, operator_view_parser.py, stack_view_parser.py
- Defect: 解决合并冲突时，`kernel.dur`字段在某版本中变为字符串类型，但3处仍直接参与数值
  求和；另外fwk_cann_relation_parser中计算device end timestamp时`kernel.dur`单位
  不一致(us而非ns)。
- Fix: 3处`kernel.dur`改为`float(kernel.dur)`；时间单位转换用`convert_us2ns()`。
- Reviewability: high -- 合并冲突后应检查所有涉及变更字段的使用点
- Review rule: 合并冲突解决后，对被冲突影响的字段做全局搜索验证类型/单位一致性

### D-463: Bug Fix For Analyse Performance
- Hashes: eb5a5c9d4 [+1: 87a038eea]
- Root cause: dict.update()返回值为None + 多处小bug
- Files: constant.py, cann_export.py, profiling_parser.py
- Defect: 多处问题:
  (1) `param_dict = param_dict.update(...)` → dict.update()返回None，param_dict被赋为None
  (2) `time.sleep(1)`轮询间隔过长，导致等待延迟
  (3) run_parser()异常直接抛出跳过后续逻辑
  与D-448(Memory Match)部分重叠(PROF_WARN_SIZE、try/except)，但此为更早提交(MR号更小)。
- Fix: (1)改为`param_dict.update(...)`无赋值 (2)sleep(0.1) (3)加try/except。
- Reviewability: high -- dict.update()返回None是Python基础知识
- Review rule: Python中dict.update()/list.sort()/list.reverse()等in-place方法返回None，不可赋值

### D-464: fix the way of adding synchronize when init ddp
- Hashes: 00a50a5b0 [+2: 3b345a25a, e7203c28f]
- Root cause: 全局类替换方式(子类化DDP)破坏兼容性
- Files: __init__.py, distributed/__init__.py, distributed/distributed.py(删除), utils/module.py
- Defect: 原方案创建`Distributed_DataParallel`子类继承DDP并在`__init__`中加synchronize()，
  然后全局替换`torch.nn.parallel.DistributedDataParallel`。但全局替换class会破坏
  isinstance检查和第三方库兼容性。
- Fix: 废弃子类方式，改为monkey-patch：在`apply_module_patch()`中替换
  `DistributedDataParallel.__init__`为包装函数(调原始init后再synchronize)。
  删除`distributed.py`文件。
- Reviewability: medium -- 需理解类替换vs方法替换的兼容性差异
- Review rule: 扩展框架类行为优先使用monkey-patch方法而非全局替换类对象

### D-465: save_async bug fix
- Hashes: 86c2fc7c1 [+5: 56187d68f, 2eee1249a, 29cafe17e, b40f11451, 120c25207]
- Root cause: save_async类型限制过严+全局stream变量不支持多device
- Files: torch_npu/utils/serialization.py, test/test_utils/test_save_async.py
- Defect: 两处问题:
  (1) `save_async`强制要求obj为Module，对checkpoint dict(最常见用法)报错
  (2) 全局`second_stream`变量只支持单device，多device场景下stream跨设备使用
- Fix: (1)移除Module类型校验，允许任意对象 (2)改为`save_async_stream_map`(dict结构，
  key为device)。测试重构为包含正确性验证(加载后比较)。
- Reviewability: high -- 类型限制应在设计阶段就考虑通用性
- Review rule: 序列化API不应限制保存对象的类型(Module/Tensor/dict都是合法输入)

### D-466: Update the memory record matching logic
- Hashes: 3ccfa41d2 [+2: 245c35d24, 7f2048ba2]
- Root cause: 内存记录匹配算法效率低+数据源不一致
- Files: memory_view_parser.py, torch_op_node.py, trace_view_parser.py, view_parser_factory.py等
- Defect: 原内存记录匹配用线性回退(MAX_FIND_LAYERS计数，基于.ts字段)找包含该内存操作的
  torch op，但.ts与.start_time语义不同，且线性回退在深嵌套场景下找不到正确的父op。
  数据源从文件重新解析而非使用已构建好的树，导致重复I/O。
  `GlobalVar.torch_op_tree_node`在trace_view_parser中过早清空。
- Fix: 改为沿`.parent_node`向上遍历树节点(用start_time/end_time判断包含关系)；
  数据源改用`GlobalVar.torch_op_tree_node`；清空操作移到所有parser完成后。
- Reviewability: medium -- 需理解profiler树结构和时间戳语义
- Review rule: 树结构的匹配应沿parent链遍历，不应用线性数组回退

### D-467: Remove field_tag to all_data in codegen
- Hashes: 7446412a3
- Root cause: codegen后处理调用多余
- Files: codegen/gen_backend_stubs.py
- Defect: `parse_native_and_custom_yaml`中对`all_data`调用`field_tag()`做后处理，
  但该后处理在此阶段不需要(field_tag会修改数据结构导致下游解析异常)。
- Fix: 删除`all_data = field_tag(all_data)`一行。
- Reviewability: high -- 单行删除，但需理解codegen pipeline阶段
- Review rule: codegen pipeline中每个后处理步骤应明确其作用阶段，不在错误阶段调用

### D-468: fix grad shape error when gradient_as_bucket_view is turned on
- Hashes: 757fd9ca2 [+6: 77a18bc9b, aa5e748ae, 294b36865, 8209e18ef, 66f36ccc1, 8f6d51de2]
- Root cause: DDP bucket view未恢复梯度原始shape
- Files: torch_npu/csrc/distributed/reducer.cpp, test/distributed/test_distributed.py
- Defect: `initialize_bucket_views`中，无论`gradient_as_bucket_view_`是否开启，
  都用flat narrow作为bucket_views_in。开启gradient_as_bucket_view时，梯度应保持
  原始shape(通过`.view(v.sizes())`)，否则梯度shape与参数shape不匹配导致后续操作失败。
- Fix: gradient_as_bucket_view开启时，对narrow结果追加`.view(v.sizes())`恢复原始shape。
  测试重构为对bucket_view=True/False各跑一遍。
- Reviewability: medium -- 需理解DDP bucket view的语义(梯度是bucket的view而非独立拷贝)
- Review rule: gradient_as_bucket_view路径的tensor操作必须保留原始shape信息

### D-469: tuple object has no attribute 'get'
- Hashes: 1833890ba [+2: 435387dfc, 375309721]
- Root cause: tuple类型无.get()方法(dict API误用于tuple)
- Files: test/test_fake_tensor.py, test/test_fx.py
- Defect: 测试代码中`args.get(0)`对tuple调用了dict专属的`.get()`方法。tuple只能通过下标访问。同时存在不必要的`import meta`。
- Fix: `args.get(0)` → `args[0]`; 删除无用import。
- Reviewability: high -- 基础Python类型方法使用错误，IDE/linter即可检出
- Review rule: tuple/list用下标访问，dict用.get()访问，类型方法不可混用

### D-470: cleancode引入off-by-one循环边界 + np.bool废弃
- Hashes: eb2d2e5b8 [+6: a56ebc25a, e142f9e30, 827844276, 02517559e, 7cc6a55b9, cd40b4380]
- Root cause: 代码清理引入多处回归(废弃类型 + C++循环边界off-by-one)
- Files: test文件(np.bool), indexing_opt.cpp, select_opt.cpp
- Defect: cleancode重构引入3处错误: (1) np.bool应为np.bool_(NumPy 1.20+废弃); (2) indexing_opt.cpp循环`c10::irange(step.size())`应为`step.size()-1`(最后一轴不参与校验); (3) select_opt.cpp循环上界缺+1导致最后元素未处理。
- Fix: 修正NumPy类型名; 修正两处C++循环边界。
- Reviewability: high -- off-by-one和废弃API均为review经典检查项
- Review rule: 代码清理PR必须运行完整测试; C++循环边界变更需逐一验证语义

### D-471: DDP初始化后缺NPU同步
- Hashes: 434dc97e5 [+9: 564847175, 54caac2a4, ee137a872, 519a8aad9, 755b997d9, 3b4d6d24a, ee398aebd, 1f0e7d3a4, dd72de532]
- Root cause: DDP初始化完成后NPU设备操作未同步
- Files: torch_npu/__init__.py, torch_npu/distributed/__init__.py, torch_npu/distributed/distributed.py(新建)
- Defect: NPU环境下DistributedDataParallel.__init__完成后设备侧操作可能尚未完成(异步执行)，CPU侧已继续运行，导致分布式训练出现不确定性。附带文件拼写错误修复(dirtributed.py→distributed.py)。
- Fix: 新建NPU的DDP子类，在__init__末尾调用torch_npu.npu.synchronize(); monkey-patch替换全局DistributedDataParallel。
- Reviewability: medium -- 需理解NPU异步执行模型和DDP初始化时序
- Review rule: NPU设备初始化后需显式同步; monkey-patch替换全局类需评估isinstance兼容性

### D-472: RPC日志泄露IP地址
- Hashes: a0f5b0d92 [+2: 6afa92ac9, d9eb084d6]
- Root cause: 日志输出包含敏感网络信息
- Files: torch_npu/csrc/distributed/rpc/tensorpipe_agent.cpp
- Defect: RPC agent的WARNING日志直接输出kDefaultUvAddress(实际IP地址)和worker监听地址。属于敏感信息泄露安全合规风险。
- Fix: 日志中IP地址替换为固定字符串"Default Address"; 删除VLOG(1)中的worker地址打印。
- Reviewability: high -- 安全审计应检出日志中的IP地址/端口
- Review rule: 日志禁止输出实际IP地址、端口等网络敏感信息; 安全审计需覆盖所有日志输出点

### D-473: alltoall VLA引用捕获导致UAF
- Hashes: 117c6a8fe [+2: 5b4062fa8, 5936c41b0]
- Root cause: 栈上VLA被异步lambda引用捕获(use-after-free)
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: alltoall_base/alltoall中inputCounts等4个数组声明为C风格VLA(栈变量)，lambda以&引用捕获后推入TaskQueue异步执行。函数返回后栈上VLA已销毁，lambda访问悬空引用导致UAF。
- Fix: VLA改为std::vector<uint64_t>; lambda改为值捕获; 调用hcclAlltoAllV时用.data()获取指针。
- Reviewability: high -- VLA+lambda引用捕获是经典C++ UAF模式，clang-tidy有规则
- Review rule: 异步lambda禁止引用捕获局部数组/VLA; 用值捕获或heap分配管理生命周期

### D-474: size_t反向循环下溢导致死循环/越界
- Hashes: 089fd3278 [+2: 9b3c10db6, c460a49e7]
- Root cause: 无符号整数反向遍历条件恒真
- Files: torch_npu/csrc/framework/contiguous/permute_opt.cpp
- Defect: `for(size_t i = size-1; i >= 0; i--)` -- size_t永远>=0，循环条件恒真。i==0时i--下溢为SIZE_MAX导致越界访问或死循环。另有2处size_t与int64_t混用产生有符号/无符号比较问题。
- Fix: 循环变量改为int64_t; 循环上界缓存为int64_t类型消除混用。
- Reviewability: high -- 编译器-Wsign-compare可检出; size_t反向遍历是C++经典陷阱
- Review rule: 反向遍历禁用无符号循环变量; 启用-Wsign-compare编译警告

### D-475: TensorOptions自引用初始化UB
- Hashes: f69dab77b [+2: aa3d3f84d, b5456be4b]
- Root cause: 未初始化对象的自引用(undefined behavior)
- Files: torch_npu/csrc/aten/common/ToKernelNpu.cpp
- Defect: `options_ = options_.dtype(...)...` -- 用未初始化的options_调用自身成员方法，读取未初始化对象是UB。与D-231(C++自引用初始化3处)完全相同的模式。
- Fix: `options_.dtype(dtype)` → `c10::TensorOptions().dtype(dtype)`，从默认构造对象开始链式调用。
- Reviewability: high -- clang-tidy可检测self-referencing initialization
- Review rule: 变量初始化表达式禁止引用自身; 链式构造应从明确的基对象开始

### D-476: empty_like忽略用户dtype参数
- Hashes: 139de6ae0 [+1: b69b09529]
- Root cause: 参数透传错误(self.options()替代options)
- Files: TensorFactories.cpp, test/unsupported_ops_info.yaml
- Defect: empty_like_npu在非默认格式分支调用ApplyTensorWithFormat时传入self.options()(源tensor的options)而非用户传入的options参数，导致用户指定的dtype被静默忽略，输出dtype始终与源tensor一致。
- Fix: self.options() → options; 从unsupported列表移除empty_like。
- Reviewability: high -- 参数名差异(self.options vs options)在review中可检出
- Review rule: 算子实现中优先使用调用方传入的options而非从self提取; "忽略用户参数"的代码路径需明确注释

### D-477: 编译缓存路径依赖cwd导致缓存失效
- Hashes: 85e7f6bdc [+4: f612d08ca, 0d8d24129, 71e490847, 9ad3c7a6a(语义重复,不同summary)]
- Root cause: 默认缓存路径依赖进程cwd(不稳定)
- Files: torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp
- Defect: ACL_OP_COMPILER_CACHE_DIR未设置时以GetCurDirPath()+"/cache"为默认路径。不同工作目录启动导致缓存目录分散、编译缓存无法复用; 同时在cwd强制创建cache/目录是侵入性副作用。9ad3c7a6a(MR!6680)是语义重复(同修复不同summary)。
- Fix: 移除GetCurDirPath()和默认路径逻辑; 未设置环境变量时完全跳过，让CANN自行管理缓存默认行为。
- Reviewability: medium -- 需理解编译缓存的目录策略和CANN默认行为
- Review rule: 运行时目录默认值禁用cwd(不稳定); 环境变量未设置时fallback到框架默认而非自行构造

### D-478: LSTM tuple拼接类型错误
- Hashes: 0510bed37
- Root cause: Tensor与tuple直接+运算(类型不匹配)
- Files: torch_npu/utils/module.py
- Defect: lstm_forward中`result_tmp[0].reshape(shape) + result_tmp[1:]` -- 左侧是Tensor，右侧是tuple。Tensor+tuple调用Tensor加法而非tuple拼接，导致类型错误或语义错误。
- Fix: 包裹为单元素tuple `(result_tmp[0].reshape(shape),)` 再与tuple拼接。
- Reviewability: high -- Python类型不匹配在review中可检出; 逗号缺失是常见tuple陷阱
- Review rule: tuple拼接的所有操作数必须为tuple类型; 单元素tuple必须带尾逗号

### D-479: dirname路径字符串副本就地修改风险
- Hashes: b49bf5d54 [+3: 6a371ab96, d0e70b6a2, 505d216d8]
- Root cause: POSIX dirname()就地修改char*缓冲区的安全性问题
- Files: torch_npu/csrc/toolkit/profiler/common/utils.h
- Defect: `std::string temp_path = path`赋值后将temp_path.data()传给dirname()。dirname()是C函数，就地修改缓冲区，std::string内部状态(如size)不会随之更新，可能导致后续使用不一致。
- Fix: 改用`std::string(path.begin(), path.end())`构造精确副本，确保缓冲区干净。
- Reviewability: low -- 需理解POSIX dirname()的就地修改语义和std::string内部状态
- Review rule: 将std::string传给C函数前需确保缓冲区独立; dirname()结果应立即拷贝到新string

### D-480: profiler白盒安全问题集合(锁泄漏+位运算假设+地址泄露)
- Hashes: 49b41f30e [+2: 756d24e47, 3623c676e]
- Root cause: 多项代码质量/安全问题
- Files: 12个文件(C++和Python)
- Defect: 5类问题集合: (1) mutex.lock()/unlock()裸调用，异常时锁泄漏; (2) `&(STAMP_QUEUE_LEN-1)`位运算假设长度是2的幂; (3) 日志%p打印指针地址泄露内存布局; (4) int64_t/uint64_t混合运算溢出风险; (5) except Exception: raise丢弃原始异常链。
- Fix: lock_guard替代裸锁; &改为%; 删除%p; 显式uint64_t转型; `from e`保留异常链。
- Reviewability: high -- 每项单独都是review/static analysis可检出的问题
- Review rule: 禁止裸mutex.lock(); 位运算优化需static_assert 2的幂; 日志禁止打印裸指针

### D-481: 矩阵乘法UT多重修复(dtype+缩进+参数)
- Hashes: d7853ec8e [+3: a695055da, b2f2eaf05, cb637546f]
- Root cause: 测试代码多重问题
- Files: 7个测试文件(addbmm/addmm/baddbmm/mm/mv/linear)
- Defect: (1) 测试方法签名含device="npu"参数，新版测试框架不支持带参发现; (2) np.float32构造输入但NPU kernel走fp16路径，精度不匹配; (3) test_addbmm_transpose缩进错误嵌套在test_addbmm体内; (4) np.transpose不兼容NPU tensor。
- Fix: 去除device参数; 输入改np.float16后.float(); np.transpose→torch.permute; 修正缩进。
- Reviewability: high -- 缩进嵌套和device参数是review可检出的; dtype路径需了解NPU kernel
- Review rule: 测试方法签名不应携带device参数; NPU测试需明确dtype路径(fp16 vs fp32)

### D-482: batchnorm/BCE UT修复(期望值+参数范围+随机种子)
- Hashes: 5cb3f74c9 [+3: 99454afd6, 8dd2c72d0, f7836e781]
- Root cause: 测试数据和期望值错误
- Files: 5个测试文件(adaptive_avg_pool2d/batchnorm/binary_cross_entropy)
- Defect: (1) batchnorm_backward_elemt期望值硬编码错误(110.→9.2000等); (2) BCE的target采样范围(0,2)超出[0,1]约束导致数值不稳定/NaN; (3) 缺随机种子导致结果不可复现。
- Fix: 更新期望值; target范围(0,2)→(0,1); 添加manual_seed固定随机数; 收紧tolerance。
- Reviewability: high -- BCE target范围约束是算子文档明确的; 期望值应有推导
- Review rule: 损失函数测试输入范围必须符合算子数学约束; 随机测试必须设seed; 期望值来源需注释

### D-483: aclrt设备ID vs context保存恢复(多进程场景)
- Hashes: 38334d68e [+3: 7f6cab22e, 9a3a91e7b, 265f0a1b1]
- Root cause: device id相同不代表context相同
- Files: NPUCachingAllocator.cpp, npu_sys_ctrl.cpp, npu_sys_ctrl.h
- Defect: insert_events用aclrtGetDevice/SetDevice保存恢复设备，但FSDP多进程场景下同一device id对应不同context。设备切换后context已变化，aclrt调用作用在错误context上导致crash。
- Fix: 改用aclrtGetCurrentContext/SetCurrentContext操作context句柄; NpuSysCtrl新增ctx_成员保存初始化context并暴露InitializedContext()接口。
- Reviewability: low -- 需理解CANN runtime的device/context/stream三层模型
- Review rule: 多进程场景中设备保存恢复必须操作context而非device; device id不是context的唯一标识

### D-484: FSDP future.wait不适用NPU后端
- Hashes: 4d7a11b0d [+1: 4f8deaa91]
- Root cause: NPU后端不支持Work.get_future()
- Files: torch_npu/npu/amp/sharded_grad_scaler.py
- Defect: ShardedGradScaler._unscale_grads_中对all_reduce返回值调用.get_future()再torch.futures.wait_all()。这是CUDA后端的异步同步方式，NPU的Work对象不支持get_future()，导致FSDP训练hang或报错。
- Fix: 去掉.get_future(); 直接收集Work对象后逐个.wait()同步。
- Reviewability: medium -- 需了解NPU分布式通信的Work API差异
- Review rule: 从CUDA移植的异步同步代码需验证NPU Work API兼容性; NPU后端用Work.wait()

### D-485: 0D tensor索引(PyTorch版本升级)
- Hashes: a5ec7386b [+1: 595d7faf3]
- Root cause: 0D tensor不支持[0]下标索引
- Files: torch_npu/npu/amp/grad_scaler.py
- Defect: `per_device_found_inf.get(device)[0].item()` -- 新版PyTorch中found_inf从1D变为0D(scalar tensor)，[0]对0D tensor抛IndexError: too many indices。
- Fix: `[0].item()` → `.item()`，直接对0D tensor取值。
- Reviewability: high -- 0D tensor行为变化是PyTorch版本升级的已知变更
- Review rule: .item()无需前置索引; PyTorch升级后需检查所有scalar tensor的维度假设

### D-486: deepcopy CPU tensor走NPU路径崩溃
- Hashes: 7655a9344 [+1: d670b5c24]
- Root cause: monkey-patch的deepcopy对所有设备一视同仁(D-487引入的回归)
- Files: torch_npu/utils/storage.py, test/test_npu/test_storage.py
- Defect: _deepcopy被monkey-patch到全局storage类后，对CPU tensor也调用NPU专用的_tensor_construct_from_storage和npu_format_cast，导致CPU tensor deepcopy崩溃。这是D-487修复浅拷贝时引入的回归。
- Fix: 添加`if self.device.type != 'cpu'`分支; CPU路径走原生copy.deepcopy; 补充CPU tensor deepcopy测试。
- Reviewability: high -- monkey-patch函数必须处理所有设备类型
- Review rule: monkey-patch到全局类的方法必须检查设备类型并分派; 非默认设备的测试必须覆盖

### D-487: storage deepcopy浅拷贝(内存共享)
- Hashes: d3b84d12c [+1: 670383b8f]
- Root cause: tensor.set_(storage)是指针共享不是深拷贝
- Files: torch_npu/utils/storage.py, test/test_npu/test_storage.py
- Defect: 原deepcopy实现`torch.tensor([]).set_(self)`只将原storage指针设到新tensor上，两个tensor共享同一块NPU内存(浅拷贝)。且未保留NPU私有format(如NZ格式29)。(此修复引入D-486的CPU tensor回归)
- Fix: 改用_tensor_construct_from_storage+.clone()获得独立副本; npu_format_cast保留格式。
- Reviewability: medium -- 需理解NPU storage的深拷贝语义和format保留
- Review rule: deepcopy实现必须验证内存物理隔离(不共享底层buffer); NPU format需显式保留

### D-488: reducer ASSERT引用已废弃字段
- Hashes: f41db8596
- Root cause: 字段重命名后断言引用未同步
- Files: torch_npu/csrc/distributed/reducer.cpp
- Defect: finalize_backward中ASSERT检查bucket.future_work非空，但实际等待的是bucket.work。future_work已被废弃/删除，引用不一致可能导致编译失败或运行时断言无效。
- Fix: `TORCH_INTERNAL_ASSERT(bucket.future_work,...)` → `(bucket.work,...)`
- Reviewability: high -- 字段重命名后全局搜索即可检出
- Review rule: 字段/方法重命名时必须全局搜索所有引用; ASSERT目标变量应与后续使用一致

### D-489: profiler trace view缺少msprof slice文件
- Hashes: 3dc5331dc [+6: 45ce7466f, d76d8ceef, 389554c00, ec302d643, 26bd37e9b, e219e3ade]
- Root cause: 文件名正则不完整 + 路径变更未同步
- Files: 6个profiler Python文件(cann_file_parser/path_manager等)
- Defect: (1) MSPROF_TIMELINE正则缺少`msprof_*_slice_*.json`模式，部分切片文件未被识别导致trace view数据不完整; (2) start_info路径从device根目录移到host/子目录后path_manager解析失败。
- Fix: 补充正则模式匹配slice文件; 优先检查host/start_info路径; 附带统一os.makedirs为FileManager.make_dir_safety(含symlink检测)。
- Reviewability: medium -- 文件名模式变更需同步正则; 路径迁移需通知消费方
- Review rule: CANN profiling输出格式变更时同步所有解析端的文件名正则和路径

### D-490: op执行超时阈值不适应大模型场景
- Hashes: c5af08508 [+5: 07fb8415e, 72bc54541, 3e37d8a44, aa7baf269, 54ca3a38b]
- Root cause: 默认超时阈值过低
- Files: acl_rt.h, AclInterface.cpp, AclInterface.h, npu_sys_ctrl.cpp
- Defect: NPU算子执行默认超时阈值在大模型/大shape场景下不够，正常执行被误判为超时。CANN底层无法区分真正的hang和长时间正常执行。
- Fix: 新增aclrtSetOpExecuteTimeOut接口封装(动态加载); 系统初始化时设超时上限547; NPU_CHECK_SUPPORTED_OR_ERROR宏保证旧版驱动不报错。
- Reviewability: low -- 超时阈值需实际workload验证; 547的来源需确认
- Review rule: 超时配置应可通过环境变量覆盖; 动态加载API需fallback; magic number需注释来源
### D-491: std::call_once异常后状态中毒导致永久死锁

- Hashes: 6b1946bc5 [+5: 151586af0, af6f7bf66, af55d0ba1, 28750aff5, 69fd3ad81]
- Root cause: std::call_once异常后状态中毒(C++标准库平台相关行为)
- Files: torch_npu/csrc/core/npu/NPUStream.cpp
- Defect: `initGlobalStreamState()`通过`std::call_once`执行，当callable抛异常时，C++标准规定
  once_flag可被新线程重试，但glibc旧版本实现中第二次调用同一once_flag会永久阻塞。NPU stream
  初始化在设备不可用时抛异常，后续重试永远卡住。
- Fix: 用`int flag + std::mutex`手写double-checked locking替代`std::call_once`，异常时flag
  保持0，下次重入可重新初始化。
- Reviewability: high -- 模式清晰，但需注意flag应为atomic且用lock_guard
- Review rule: 审查所有`std::call_once`使用处，确认callable是否可能抛异常；若可能，需验证目标平台恢复行为

### D-492: npu_format_cast缺少格式相同时的短路返回

- Hashes: c05568d61 [+2: 93abbed22, 18079cc19]
- Root cause: 缺少前置校验导致冗余format cast
- Files: codegen/autograd/gen_variable_type.py, FormatCastKernelNpu.cpp, npu_native_functions.yaml
- Defect: `npu_format_cast`在调用`npu_format_cast_impl`前未检查源tensor的acl_format是否
  已等于目标格式，导致相同格式仍执行无意义的cast操作(性能浪费，可能触发不必要的内存分配)。
  同时autograd codegen硬编码了`npu_format_cast`字符串，遗漏`_npu_format_cast`变体。
- Fix: 增加`get_npu_format(self) == acl_format`短路检查; 新增`_npu_format_cast`作为无校验
  的内部路径供codegen调用。
- Reviewability: medium -- 涉及codegen + C++ + YAML三处协同变更
- Review rule: 设备格式转换函数入口应校验源/目标格式是否相同，避免noop cast

### D-493: nan_to_num dispatch目标错误(NPUNativeFunctions vs op_plugin)

- Hashes: 735a6e135
- Root cause: 算子dispatch目标不一致(hostapi层路由错误)
- Files: torch_npu/csrc/aten/OverrideOperators.cpp
- Defect: `nan_to_num`/`nan_to_num_out`/`nan_to_num_`三个override wrapper调用了
  `at_npu::native::NPUNativeFunctions::nan_to_num*`(旧路径)，应使用`op_plugin::nan_to_num*`
  与其他算子的dispatch模式一致。旧路径可能是已废弃或存在bug的实现。
- Fix: 三处调用从`NPUNativeFunctions`统一重定向到`op_plugin`命名空间。
- Reviewability: high -- 纯机械替换，改动对称
- Review rule: OverrideOperators中的算子dispatch应统一使用op_plugin路径

### D-494: profiler目录创建散落多处且缺少软链接安全校验

- Hashes: f6a6bf5f0
- Root cause: 目录创建代码重复+路径安全漏洞
- Files: file_manager.py, path_manager.py, cann_file_parser.py, view_parser_factory.py,
  msprofiler_c_interface.py, profiler_action_controller.py
- Defect: 多个位置各自实现`os.makedirs`(NpuProfCreator/FileManager/内联创建)，既有代码
  重复也缺少软链接校验。path_manager中`get_start_info_path`未检查host/start_info路径;
  cann_file_parser遗漏一种msprof文件名模式; msprofiler_c_interface直接用`open()`写json
  绕过权限控制。
- Fix: 新增`FileManager.make_dir_safety`(含os.path.islink检查)统一替换所有目录创建入口;
  补充路径查找和正则模式; json写入改用FileManager。
- Reviewability: medium -- 散落6个文件的重构，每处改动小但需逐一确认
- Review rule: 文件/目录操作应统一通过FileManager，禁止直接os.makedirs; 路径操作必须校验软链接

### D-495: LazyConv3d继承Conv3d但weight未materialized导致cast崩溃

- Hashes: 97c0f422a [+2: 09379ffa8, 304a5c5ae]
- Root cause: 类继承层级遗漏(Lazy变体未排除)
- Files: torch_npu/utils/module.py
- Defect: `cast_weight`对`torch.nn.Conv3d`执行weight格式转换(转ACL_FRACTAL_Z_3D)，但
  `LazyConv3d`继承自Conv3d，其weight是`UninitializedParameter`尚未materialized。
  `issubclass`对LazyConv3d返回True，对未初始化weight调用`.data`和`.half()`导致异常。
- Fix: 在Conv3d分支前增加`issubclass(class_name, torch.nn.LazyConv3d)`提前return。
- Reviewability: high -- 两行改动
- Review rule: cast_weight中每个issubclass分支都应检查对应的Lazy变体

### D-496: custom op scatter_update的Library注册方式和推理函数签名错误

- Hashes: 7a0978a11
- Root cause: torch.library API误用(参数名和构造参数类型错误)
- Files: torch_npu/dynamo/__init__.py
- Defect: (1) `Library("npu", IMPL)`中`IMPL`是裸变量名而非字符串`"IMPL"`，导致NameError;
  (2) `scatter_update_infer`参数签名`(self, indices, src, dim)`与算子schema
  `(data, indices, updates, axis)`不匹配，dispatch时参数对应错误。
- Fix: `IMPL`改为字符串`"IMPL"`; 推理函数参数名修正为`(data, indices, updates, axis)`。
- Reviewability: high -- 改动小，问题显而易见
- Review rule: Library注册的impl函数签名必须与算子schema参数名/个数严格一致

### D-497: LayerNorm forward monkey-patch破坏torch.compile图捕获

- Hashes: 188009fe7
- Root cause: monkey-patch与编译器不兼容
- Files: torch_npu/utils/module.py
- Defect: `apply_module_patch`将`torch.nn.LayerNorm.forward`替换为自定义的
  `layernorm_forward`(推理时走npu_layer_norm_eval优化路径)。这个monkey-patch破坏了
  torch.compile的图捕获机制，compile期间trace遇到非标准dispatch路径导致编译失败。
- Fix: 完全删除`layernorm_forward`函数及其注册，回退到PyTorch原生LayerNorm.forward。
- Reviewability: high -- 纯删除
- Review rule: 对PyTorch内置Module.forward的monkey-patch必须验证与torch.compile的兼容性

### D-498: SyncBatchNorm反向传播双重归一化(数学公式错误)

- Hashes: 05a355355
- Root cause: 数学公式实现错误(batch_norm_backward_elemt输入语义理解错误)
- Files: torch_npu/utils/syncbatchnorm.py
- Defect: 在调用`torch.batch_norm_backward_elemt`前将`sum_dy`/`sum_dy_xmu`除以
  `count_tensor.sum()`得到mean值。但该函数内部已用最后一个参数count做归一化，传入已归一化
  的值导致双重除法，梯度计算错误。
- Fix: 删除手动归一化(divisor/mean_dy/mean_dy_xmu)，直接将sum值传入。
- Reviewability: high -- 改动集中，但需理解batch_norm_backward_elemt的API语义
- Review rule: 调用底层数学函数前确认其内部是否已包含归一化逻辑

### D-499: Revert export_stacks profiling功能(质量不达标)

- Hashes: 91895483d
- Root cause: feature发布质量不达标导致回退
- Files: constant.py, view_parser_config.py, stack_view_parser.py(deleted), profiler.py
- Defect: 此前引入的`export_stacks` API(含StackViewParser、METRIC常量、EXPORT_STACK类型)
  在发布后被发现存在问题(功能不稳定或与其他profiler改动冲突)，需整体回退。
- Fix: 删除StackViewParser类文件、相关常量、配置项和export_stacks方法。
- Reviewability: high -- 纯revert删除
- Review rule: 新增profiler功能应经过完整集成测试再合入; revert应引用原始PR说明原因

### D-500: FlashAttention反向dpse/dpse_required变量角色混淆

- Hashes: 875a3f38d
- Root cause: 未定义tensor的输出变量混淆(dpse vs dpse_required生命周期错误)
- Files: FlashAttentionKernelNpuOpApi.cpp, MultiHeadAttentionKernelNpuOpApi.cpp
- Defect: pse已定义时`dpse`分配为同shape tensor但传给kernel的是空的`dpse_required`;
  pse未定义时`dpse_required`正确设为空tensor但返回给caller的是未初始化的`dpse`。两个变量
  分别承担"传给kernel"和"返回给caller"的角色导致两条路径都错。
- Fix: 统一用`dpse`一个变量: 已定义时分配正常tensor，未定义时分配空tensor{0}，传给kernel
  也是dpse。kernel执行后若pse未定义则重置dpse为null tensor。
- Reviewability: medium -- 需理解aclnn算子对输出tensor的语义要求
- Review rule: 算子backward中optional输出tensor禁止用两个变量分别承担kernel输入和caller返回

### D-501: profiler适配多问题合并修复(正则/step时间/常量/目录清理)

- Hashes: 2c8c976cb
- Root cause: 正则表达式/常量硬编码不匹配实际数据格式
- Files: 15个profiler Python文件(node_info_bean, op_summary_bean, csv_headers等)
- Defect: (1) CANN文件名正则用`\d`只匹配单位数device id，>=10时失败; (2) step时间范围
  用host侧framework op时间而非device侧kernel实际执行时间，导致step划分不准; (3)
  TASK_START_TIME常量缺`(us)`后缀与CSV header不匹配; (4) 目录删除失败raise RuntimeError
  中断profiling而非降级。
- Fix: `\d`改`\d+`; 新增min_start/max_end追踪device侧时间; 抽取CsvHeaders统一常量;
  目录删除改为warning。
- Reviewability: low -- 15文件4个独立bug合并为单commit
- Review rule: 正则变更必须覆盖边界值测试; 常量字符串应有编译期校验

### D-502: EventPool机制与TaskQueue异步执行模型不兼容(整体回退)

- Hashes: 662d25462
- Root cause: event生命周期管理与异步执行模型不兼容
- Files: 15个C++文件(NPUCachingAllocator, NPUEvent, NPUEventManager, ProcessGroupHCCL等)
- Defect: EventPool使用ACL_EVENT_CAPTURE_STREAM_PROGRESS标志和NPUEvent封装池化event
  对象，但TaskQueue异步模式下event在enqueue时尚未真正record到stream上，导致
  aclrtSynchronizeEvent时状态不一致。npu_events按stream分桶的map结构也与异步record
  时序不匹配。
- Fix: 移除EventPool类; event改为直接aclrtCreateEventWithFlag(ACL_EVENT_TIME_LINE)
  创建用完销毁; 新增recorded_events集合追踪已record的event; 新增ResetEventTask用于
  HCCL同步后reset。
- Reviewability: low -- 15文件345+276行，涉及内存分配器核心并发逻辑重构
- Review rule: 内存分配器和event生命周期变更必须有并发安全性分析和压力测试

### D-503: is_npu对undefined tensor空指针崩溃

- Hashes: 629524ea9
- Root cause: 缺少空值/未定义状态前置检查
- Files: torch_npu/csrc/core/npu/DeviceUtils.h
- Defect: `is_npu`直接调用`tensor.device().is_privateuseone()`，但undefined tensor的
  device()会访问空TensorImpl导致UB或崩溃。调用方可能传入optional<Tensor>解包后的空值。
- Fix: 增加`tensor.defined()`前置检查，未定义直接返回false。
- Reviewability: high -- 单文件4行
- Review rule: 接受Tensor参数的utility函数必须处理undefined tensor场景

### D-504: argmin手写dispatch逻辑与op_plugin不一致

- Hashes: a81d895fd
- Root cause: 手写dispatch逻辑过时
- Files: torch_npu/csrc/aten/OverrideOperators.cpp
- Defect: `wrapper__argmin`手动检查JitDisable/IsOpInputBaseFormat/IsGraphMode三个条件
  决定走OpApi还是NativeFunctions，但op_plugin内部已有更完善的分发策略。手写条件限制
  BaseFormat才走OpApi，遗漏op_plugin对更多格式的适配。
- Fix: 删除手写if-else，直接调用`op_plugin::argmin()`。
- Reviewability: high -- 删除6行新增2行
- Review rule: 算子override层不应手写dispatch逻辑，统一委托op_plugin入口

### D-505: SetDeterministic在customHandler分支之后导致hostapi路径未覆盖

- Hashes: 7845c8333
- Root cause: 代码路径遗漏(全局配置设置位置不当)
- Files: torch_npu/csrc/framework/OpParamMaker.cpp
- Defect: `SetDeterministic()`放在customHandler分支的else路径(标准ACL执行路径)中，
  走customHandler路径(hostapi)的算子不会启用deterministic模式。用户设置
  `torch.use_deterministic_algorithms(True)`后hostapi算子仍产生不确定性结果。
- Fix: 将`SetDeterministic()`上移到customHandler判断之前，覆盖两条路径。
- Reviewability: high -- 2行位置调换
- Review rule: 全局配置设置点必须在所有执行路径的公共前置位置

### D-506: DataParallel API拦截粒度过粗(拦截整个类而非__init__)

- Hashes: 9fd849b76
- Root cause: API拦截粒度过粗导致误拦截
- Files: torch_npu/utils/npu_intercept.py, torch_npu/utils/unsupport_api.py
- Defect: unsupport_nn_api注册`"torch.nn.DataParallel"`(整个类)，导致已有实例的方法调用
  也被拦截报错。同时`is_module_parameters_supported`从args[0]取DataParallel实例(尚未
  初始化)而非用户传入的module，named_parameters()失败。
- Fix: 拦截目标细化为`"torch.nn.DataParallel.__init__"`; 修复参数筛选逻辑。
- Reviewability: high -- 2文件5行
- Review rule: API拦截应在最小必要粒度; 拦截前参数解析必须匹配目标函数签名

### D-507: codegen中op_plugin wrap函数名使用错误的名称生成方法

- Hashes: 7579f927c
- Root cause: 函数名映射错误(codegen使用了错误的名称生成方法)
- Files: codegen/custom_functions.py
- Defect: `compute_op_definition`用`cpp.name(f.func)`获取C++函数名，但op_plugin的wrap
  函数名与标准C++命名不同(overload后缀/命名空间差异)。生成的dispatch代码链接到不存在的
  函数，编译或运行时出错。
- Fix: 导入`get_opplugin_wrap_name`替代`cpp.name(f.func)`。
- Reviewability: high -- 2行变更
- Review rule: codegen的名称映射逻辑变更必须配合编译验证

### D-508: profiler通信矩阵按step拆分逻辑错误

- Hashes: f22ac0072
- Root cause: 边界条件处理错误 + step标识符格式不一致
- Files: torch_npu/profiler/analysis/prof_view/communication_parser.py
- Defect: `split_matrix_by_step`在`len(step_list)==1`时走单步兜底(key为"step")，丢失
  step id信息。多step时step_id直接作dict key(如"0","1")，缺少"step"前缀，与下游期望的
  "step0"格式不一致。
- Fix: 移除len==1短路条件; step key加"step"前缀拼接。
- Reviewability: high -- 2行
- Review rule: dict key格式变更必须检查所有下游消费者的解析逻辑

### D-509: 精度模式环境变量查询接口缺失+AclSetCompileopt返回值未检查

- Hashes: 1b9e838a9
- Root cause: 缺少运行时查询接口 + 返回值静默吞掉
- Files: OptionRegister.h, EnvVariables.cpp, EnvVariables.h
- Defect: 下游需运行时查询精度模式(FP32降FP16/ConvHF32/MatmulHF32)，但无对应接口只能
  重复读环境变量。`AclSetCompileopt`不检查返回值，配置失败静默吞掉。
  IsAllowFP32ToFP16还需区分芯片型号(910B1前后默认值不同)。
- Fix: 新增REGISTER_OPTION_BOOL_FUNCTION_ALL_CASE宏支持三值映射; 实现按SoC返回正确
  默认值; 为AclSetCompileopt加TORCH_CHECK。
- Reviewability: medium -- 3文件，涉及宏定义和SoC条件分支
- Review rule: 精度模式接口必须UT覆盖不同SoC版本+不同配置值组合

### D-510: complex_out算子输出tensor赋值方式错误(局部变量替换而非原地拷贝)

- Hashes: ef4f8d765
- Root cause: CPU fallback路径中输出tensor赋值方式错误
- Files: ComplexKernelNpu.cpp, test_complex.py(新增)
- Defect: `complex_out`将real/imag拷CPU计算后用`out = out_cpu.to(device)`赋值回NPU，
  但这只修改局部变量out的指向，未修改调用方持有的output tensor数据。输出为未初始化或旧数据。
  broadcast场景下shape也可能不匹配。
- Fix: 改用`out.resize_(output_size)` + `out.copy_(out_cpu)`原地拷贝。
- Reviewability: high -- 核心修复5行，附带UT
- Review rule: out变体算子必须通过原地操作(resize_+copy_)修改输出tensor，禁止赋值替换

### D-511: Stream.query()是stub实现(TORCH_CHECK(false)占位)

- Hashes: d9ac8f5a4
- Root cause: 功能缺失(API未实现，stub直接抛异常)
- Files: NPUStream.h, AclInterface.cpp/.h, Stream.cpp, acl_rt.h等7个文件
- Defect: `Stream.query()`方法体内直接`TORCH_CHECK(false, "NPU does not support...")`，
  调用即崩。此外行尾有多余反斜杠导致语法畸形(C++编译器能容忍但属笔误)。
- Fix: 通过aclrtStreamQuery ACL接口实现真正的stream状态查询返回bool; 补充ACL头文件
  enum定义和动态加载逻辑; 移除行尾反斜杠。
- Reviewability: medium -- 跨7文件但逻辑线性
- Review rule: 定期扫描TORCH_CHECK(false, ...)占位的stub方法，确保不遗漏未实现API

### D-512: torchair编译开关的Python truthiness判断错误

- Hashes: f09654ee6
- Root cause: 布尔/字符串类型混淆的条件判断(Python truthiness陷阱)
- Files: setup.py
- Defect: `if not DISABLE_TORCHAIR`意图是"未禁用时启用编译"，但DISABLE_TORCHAIR是环境变量
  读取的字符串，非空字符串如"FALSE"在Python中truthy，`not "FALSE"`为False，torchair
  永远不会编译。
- Fix: 改为显式字符串比较`DISABLE_TORCHAIR == 'FALSE'`。
- Reviewability: high -- 单行
- Review rule: 环境变量作为开关时禁止用truthiness判断，必须显式比较字符串值

### D-513: npu_format_cast命名空间迁移后调用路径不一致

- Hashes: b84d01055
- Root cause: 命名空间/注册机制重构后调用方未同步
- Files: 22个文件(FormatCastKernelNpu.cpp, 多个算子kernel, distributed, framework等)
- Defect: `npu_format_cast`从`custom_autograd`迁移到`custom`类别后，调用方仍通过
  NPUNativeFunctions::npu_format_cast走带手动autograd的旧路径。手动编写的
  NPUFormatCastFunction(torch::autograd::Function)既冗余又可能与新dispatch冲突。
- Fix: 所有调用改为custom_ops::npu_format_cast; yaml移到custom类别; 删除手动autograd类。
- Reviewability: medium -- 文件多但全是机械替换
- Review rule: 算子注册类别变更时全仓搜索所有调用点确认命名空间一致

### D-514: 未安装git时构建失败(get_sha异常未容错)

- Hashes: 73b69ede7
- Root cause: 外部工具依赖未做容错处理
- Files: setup.py, torch_npu/__init__.py
- Defect: `get_sha()`在未安装git的环境下subprocess失败，异常未正确处理，版本号拼接
  `VERSION += "+git" + sha[:7]`对"Unknown"字符串切片生成非法版本号。
- Fix: get_sha失败返回"Unknown"常量; 版本号拼接前检查sha != UNKNOWN则跳过git后缀。
- Reviewability: high -- 2文件简单直接
- Review rule: 构建脚本中所有外部工具调用必须有fallback路径

### D-515: YAML解析中未知key用assert终止进程(过于严格)

- Hashes: ee15111a9 [+1: 8c027528f]
- Root cause: 过于严格的校验导致扩展性差
- Files: codegen/gen_backend_stubs.py
- Defect: 对backend YAML中的未知key使用`assert`直接终止。当YAML新增扩展字段(如
  custom_autograd)但codegen未同步更新时编译直接中断。
- Fix: assert改为print Warning，允许未知key时继续执行。
- Reviewability: high -- 3行
- Review rule: codegen/schema解析对未知字段应warn而非abort，保持前向兼容

### D-516: dropout mask长度uint32溢出(大shape场景)

- Hashes: d507f1415 [+1: 0672849f6]
- Root cause: 整数类型溢出(uint32 -> uint64)
- Files: torch_npu/csrc/aten/ops/DropoutKernelNpu.cpp
- Defect: `length`变量声明为`uint32_t`，当tensor元素>2^32-128(~4.29B)时，
  `(numels + 128 - 1) / 128 * 128`中间运算溢出，分配的mask远小于实际需要，
  后续内存越界。dropout_gen_mask和npu_dropout_gen_mask两处均有此问题。
- Fix: 两处`uint32_t length`改为`uint64_t length`。
- Reviewability: high -- 2行
- Review rule: 涉及tensor size/numel计算的变量必须用int64/uint64; lint应标记uint32用于size计算

### D-517: profiler枚举集合构建引用了不存在的__members__属性

- Hashes: 864c8c78b
- Root cause: 引用未定义名称(非Enum类无__members__)
- Files: torch_npu/profiler/experimental_config.py
- Defect: `supported_profiler_level()`和`supported_ai_core_metrics()`使用
  `ProfilerLevel.__members__`和`AiCMetrics.__members__`，但这两个类不是标准Python Enum
  (删除了from enum import Enum)。非Enum类访问__members__抛AttributeError。
- Fix: 直接用具体枚举值构建集合，不依赖__members__反射。
- Reviewability: high -- 单文件7行
- Review rule: 非Enum类不得使用__members__; 函数引用的类需确认继承链

### D-518: Foreach算子不限输入规模导致底层ACL溢出

- Hashes: 98734303a
- Root cause: 算子输入规模限制未处理
- Files: AmpForeachNonFiniteCheckAndUnscaleKernelNpuOpApi.cpp
- Defect: `_amp_foreach_non_finite_check_and_unscale_`直接将整个scaled_grads列表传给
  aclnnForeachNonFiniteCheckAndUnscale，tensor数量超过底层上限时执行失败或行为异常。
- Fix: 新增分片函数以MAX_TENSOR_COUNT=250为上限分批执行，最后处理余数。
- Reviewability: high -- 单文件，分片逻辑清晰
- Review rule: 批量算子调用应有输入规模上限保护

### D-519: Revert manylinux pypi支持(方案引入问题)

- Hashes: 96c09b999
- Root cause: 功能回退(manylinux方案兼容性问题)
- Files: requirements.txt, setup.py, torch_npu/__init__.py, InitNpuBindings.cpp
- Defect: manylinux支持包含patchelf依赖移除、auditwheel repair、_ld_preload运行时预加载
  等机制，但引发兼容性/构建稳定性问题需整体回退。
- Fix: 移除BdistWheelBuild类、manylinux tag列表、_ld_preload导入、auditwheel依赖;
  恢复initModule函数名。
- Reviewability: low -- 需理解原始manylinux PR全部影响
- Review rule: 大feature应有feature flag控制，便于回退时不需revert整个commit

### D-520: torch.load对weights_only/mmap/legacy格式组合处理不完整

- Hashes: 3d419ddbe
- Root cause: 参数处理不完整导致功能缺失
- Files: test_serialization.py, torch_npu/utils/serialization.py
- Defect: NPU的torch.load重写版本对weights_only=True/mmap参数/legacy格式的map_location
  处理存在多个缺陷: weights_only未正确传递; legacy格式无法重映射到NPU; 缺少
  TORCH_FORCE_WEIGHTS_ONLY_LOAD环境变量支持。
- Fix: 重写load函数直接调用_load/_legacy_load底层函数; 完整处理zip/legacy/weights_only/
  mmap参数组合; save也修复_use_new_zipfile_serialization=False时的行为。
- Reviewability: medium -- 61行新增，需对比upstream torch.load行为
- Review rule: 对upstream接口的重写/monkey-patch需随版本升级同步审查

### D-521: Revert FlashAttention dropout genmask多流内存复用

- Hashes: 07292ff3b
- Root cause: 多流并发下内存复用导致数据竞争
- Files: FlashAttentionKernelNpuOpApi.cpp
- Defect: 使用NPUEvent同步主流和辅助流(getCurrentSecondaryStream)实现dropout genmask
  的内存复用，但多流之间对同一内存块的读写存在竞争条件，event同步不够可靠。
- Fix: 删除手动event record/block同步; 改用recordStream标记内存所属stream，
  由CANN内存分配器保证复用安全性。
- Reviewability: medium -- 需理解NPU多流模型和内存管理
- Review rule: 多流内存复用必须使用recordStream而非手动event同步

### D-522: FlashAttention反向梯度使用fp16精度累积(精度不足)

- Hashes: f4dcee9e7
- Root cause: 梯度计算精度不足(低精度累积误差)
- Files: FlashAttentionKernelNpuOpApi.cpp
- Defect: `dq`/`dk`/`dv`直接以输入query的dtype(fp16/bf16)分配计算。
  aclnnFlashAttentionScoreGrad的梯度累加在低精度下精度损失显著，dq还要乘scale系数，
  fp16下容易溢出或精度退化。
- Fix: 新增fp32中间tensor dq_32/dk_32/dv_32作为算子输出，fp32下完成梯度计算和scale
  乘法后再npu_dtype_cast回原dtype。
- Reviewability: high -- 单文件，模式清晰
- Review rule: 反向传播梯度累加应默认使用fp32精度

### D-523: set_stream函数未加入__all__导出列表

- Hashes: e2684eeec
- Root cause: API导出遗漏
- Files: torch_npu/npu/__init__.py
- Defect: `set_stream`已实现但未加入`__all__`列表和import语句，用户调用
  torch_npu.npu.set_stream触发AttributeError。
- Fix: 在__all__和import中补上set_stream。
- Reviewability: high -- 两行
- Review rule: 新增公开API时检查__all__和import是否同步更新

### D-524: FakeTensor(data_ptr==nullptr)经过GetTensorNpuFormat时空指针崩溃

- Hashes: c5fd45f94
- Root cause: 边界条件缺失(FakeTensor空指针解引用)
- Files: CalcuOpUtil.cpp, test_tensor.py
- Defect: `GetTensorNpuFormat`只区分"有NPU storage"和"其他"两条路径。FakeTensor的
  data_ptr()为nullptr，不适合走GuessFormatWhenContiguous(假设有实际数据)，导致coredump。
- Fix: 增加`data_ptr() == nullptr`分支直接返回ACL_FORMAT_ND。
- Reviewability: high -- 3行
- Review rule: tensor属性工具函数应系统性处理FakeTensor/meta tensor特殊形态

### D-525: _convolution的op_api配置错误导致trace场景调用链断裂

- Hashes: 09ae87c2c
- Root cause: YAML dispatch配置错误
- Files: npu_native_functions.yaml, ConvolutionKernelNpuOpApi.cpp
- Defect: yaml中`_convolution`的op_api设为False，trace时走非op_api路径。C++中
  `_convolution`直接委托`convolution`(丢掉DO_COMPATIBILITY回退)，调用链冗余不一致。
- Fix: yaml中_convolution的op_api改为True; 重构使convolution做参数透传，
  _convolution承担实际计算和DO_COMPATIBILITY回退。
- Reviewability: medium -- yaml简单但C++调用链需理解PyTorch分发约定
- Review rule: 变更yaml中op_api标记时需同步验证C++调用链自洽

### D-526: profiler将hcom_receive事件错误标记为bubble类型

- Hashes: 3937a2f89
- Root cause: 事件分类硬编码错误值
- Files: torch_npu/profiler/analysis/prof_view/trace_step_time.py
- Defect: `hcom_receive`开头的事件在count_time时被传入类型名'bubble'而非'hcom_receive'，
  下游按类型名解析时无法识别'bubble'类型导致解析失败。
- Fix: 将硬编码的'bubble'改为'hcom_receive'。
- Reviewability: high -- 单行字符串替换
- Review rule: profiler事件类型名应定义为常量/枚举，避免散落的字符串字面量

### D-527: PyTorch 2.0 TensorList返回值规范变更导致GRU/LSTM/MHA autograd断裂

- Hashes: 71d0eafa9 [+1: 7030efd0e]
- Root cause: 框架版本适配不完整(PyTorch 2.0 breaking change)
- Files: derivatives.yaml, gen_variable_type.py, npu_native_functions.yaml,
  GRU/LSTM/MHA kernel cpp, FunctionsManual.cpp/.h (共9文件, +650/-840行)
- Defect: PyTorch 2.0改变了TensorList的返回和自动求导机制。GRU/LSTM/MultiHeadAttention
  未适配新的output_differentiability标注方式，前向/反向断裂。derivatives.yaml缺少
  这些算子的反向定义。
- Fix: 在derivatives.yaml补充backward定义和output_differentiability标注;
  FunctionsManual实现对应backward; 重构各kernel的前向。
- Reviewability: low -- 9文件大规模重构
- Review rule: 框架大版本升级应有系统性适配checklist覆盖所有自定义算子的autograd注册

### D-528: FlashAttention backward保存format_query而非原始query

- Hashes: 2286ed026 [+1: 2e4d9b2f7]
- Root cause: save_for_backward保存了错误的tensor(中间变换结果而非原始输入)
- Files: FlashAttentionKernelNpuOpApi.cpp
- Defect: save_for_backward中将format_query(经npu_format_cast处理后)保存，但backward
  需要原始query。format_cast可能改变storage布局，导致backward拿到布局变换后的tensor，
  梯度计算错误。
- Fix: save_for_backward中format_query替换为query。
- Reviewability: high -- 单行变量替换
- Review rule: save_for_backward应始终保存原始输入tensor而非中间变换结果

### D-529: 编译缓存路径和模式硬编码(用户无法通过环境变量自定义)

- Hashes: 0252fbaa4 [+3: 7a0033114, 2970efb15, bcc6459e4]
- Root cause: 可配置性不足(硬编码配置)
- Files: torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp
- Defect: cache mode硬编码为"enable"、cache dir硬编码为当前目录/cache。多用户/多任务
  环境下cache目录冲突且无法禁用。
- Fix: 读取ACL_OP_COMPILER_CACHE_MODE和ACL_OP_COMPILER_CACHE_DIR环境变量;
  有值则使用用户配置; 对mode做合法性校验(enable/disable/force)。
- Reviewability: high -- 11行环境变量读取+校验
- Review rule: 硬编码配置值应尽早暴露为可配置项

### D-530: torch_npu.optim模块未导入(用户无法访问自定义优化器)

- Hashes: 44552f800
- Root cause: 模块导入遗漏
- Files: torch_npu/__init__.py, setup.py
- Defect: torch_npu/__init__.py缺少`import torch_npu.optim`，导致NpuFusedSGD等
  自定义优化器不可访问。附带移除libgomp.so依赖避免系统冲突。
- Fix: 增加import torch_npu.optim; setup.py移除libgomp依赖。
- Reviewability: high -- 核心1行import
- Review rule: 新增子模块后需确保包的__init__.py中注册导入

### D-531: prod算子输出dtype推导错误且不支持bool输入

- Hashes: fecd826bb
- Root cause: dtype推导逻辑缺失/错误
- Files: ProdKernelNpu.cpp, test_prod.py, common_methods_invocations.py
- Defect: (1) 整数输入的prod结果应提升为int64(PyTorch规范)但原代码直接用输入dtype;
  (2) bool类型完全不支持(ReduceProd不接受bool); (3) 半精度提升到float的逻辑散且不一致。
- Fix: 抽取get_cal_type(half->float, bool->long)和get_dst_type(整数->long)统一dtype推导。
- Reviewability: medium -- 多是dtype转换重组
- Review rule: reduction类算子的dtype推导应严格对齐PyTorch CPU/CUDA的提升规范

### D-532: mul算子不支持scalar输入(CPU scalar tensor和Python float)

- Hashes: 7e738aacf
- Root cause: 输入类型处理不完整
- Files: MulKernelNpu.cpp, test_mul.py
- Defect: `mul_dest_output`仅用IsScalarWrappedToTensor判断scalar，未考虑CPU scalar
  tensor。scalar作为第一个操作数时output tensor选择错误，format和device推导出错。
  dtype提升使用自定义逻辑而非标准at::native::result_type。
- Fix: 增加IsCPUScalar判断; 使用标准result_type; mul_简化为委托mul_out。
- Reviewability: medium -- 涉及mul_out/mul/mul_三函数重组
- Review rule: 二元算子必须处理tensor-tensor/tensor-scalar/scalar-tensor组合; 使用标准result_type

### D-533: transfer_to_npu引用torch_npu.distributed(不存在)

- Hashes: 9466cda8b
- Root cause: 模块引用路径错误(monkey-patch注入后的访问路径混淆)
- Files: torch_npu/contrib/transfer_to_npu.py
- Defect: 将`torch.distributed.is_nccl_available`替换为`torch_npu.distributed.is_hccl_available`，
  但torch_npu下没有distributed属性。is_hccl_available实际被monkey-patch到torch.distributed上。
- Fix: 改为`torch.distributed.is_hccl_available`。
- Reviewability: high -- 单行
- Review rule: monkey-patch注入的函数应通过被注入模块路径访问而非注入方路径

### D-534: FlashAttention非flash路径scale参数处理错误(前向/反向不一致)

- Hashes: cf2a68d3a
- Root cause: 数学正确性错误(scale未正确应用于fusion kernel路径)
- Files: FlashAttentionKernelNpuOpApi.cpp
- Defect: 当S1<=FLASH_THRESHOLD(非flash/fusion kernel路径)时底层aclnn不支持scale参数。
  直接传入导致fusion kernel路径下scale被错误应用或忽略，前向反向数学行为不一致。
- Fix: 非flash路径将scale设为1; 手动应用scale: forward `query_scaled = query * scale`,
  backward `dq_scaled = dq * scale`，保证数学等价性。
- Reviewability: medium -- 逻辑清晰但需理解FlashAttention数学公式
- Review rule: 底层算子不支持某参数时上层手动模拟必须前向反向同时实施

### D-535: FlashAttention backward dpse返回值与kernel参数混用

- Hashes: 331df2fa5
- Root cause: 返回值变量与kernel输入变量混用
- Files: FlashAttentionKernelNpuOpApi.cpp
- Defect: FlashAttention反向传播中`dpse`既作为返回值(pse未定义时为空tensor)又被传给aclnn kernel。
  pse未定义时dpse为空tensor却被传入kernel，导致kernel收到无效输入。
- Fix: 拆分为`dpse`(返回值)和`dpse_required`(kernel参数)两个独立变量。
- Reviewability: high -- 变量语义混用在review时可识别
- Review rule: 返回值变量不应直接作为底层API的输入参数，尤其当两者在某些分支下语义不同

### D-536: 确定性算法控制仅设编译选项未设runtime context

- Hashes: ee49cd216 [+2: 6d9f72bfe, 892277165]
- Root cause: 确定性控制路径不完整(编译选项 vs 运行时context)
- Files: acl_base.h, acl_rt.h, npu_sys_ctrl.cpp, OpParamMaker.cpp, AclOpCompileInterface.cpp/h
- Defect: NPU确定性算法切换仅调用`AclSetCompileopt`(编译选项)，未同步设置runtime context级别的
  `aclrtCtxSetSysParamOpt(ACL_OPT_DETERMINISTIC)`，部分算子在运行时仍走非确定性路径。
- Fix: 新增context级确定性设置接口，初始化和切换时同步调用。
- Reviewability: medium -- 需了解ACL两层配置机制(编译选项+runtime context)
- Review rule: 影响全局行为的配置变更需检查所有控制层级是否同步

### D-537: 根logger导致日志重复打印

- Hashes: 91ec21aa6
- Root cause: Python logging使用根logger而非模块级logger
- Files: sharded_grad_scaler.py, module.py
- Defect: 直接调用`logging.error()`/`logging.info()`走根logger，上层也配置handler时日志被重复打印。
- Fix: 改为`logger = logging.getLogger(__name__)`模块级logger。
- Reviewability: high -- Python logging最佳实践
- Review rule: 库代码禁止使用根logger(`logging.xxx()`)，统一用`getLogger(__name__)`

### D-538: eraseStream空指针未检查导致crash

- Hashes: 7b8da4418
- Root cause: C++空指针检查缺失
- Files: NPUCachingAllocator.cpp
- Defect: `eraseStream`中`get_allocated_block(ptr)`返回null(无效设备指针)后继续访问，segfault。
- Fix: 增加null检查，无效指针时`AT_ERROR`报错。
- Reviewability: high -- 标准空指针防护
- Review rule: 指针查询函数返回值必须检查null后再使用

### D-539: 模块级DEVICE_NAME初始化副作用 + nan_to_num dispatch注册方式不一致

- Hashes: 177d8a313 [+1 semantic duplicate: e5e40187b]
- Root cause: 模块级变量初始化触发设备查询副作用 + dispatch注册路径不一致
- Files: DispatchKeyNativeFunctions.h, OverrideOperators.cpp, npu_native_functions.yaml, common_utils.py
- Defect: (1) `common_utils.py`中`DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]`作为模块级变量，
  import时NPU未初始化导致报错。(2) `nan_to_num`系列在yaml注册但路由不正确。
- Fix: (1) DEVICE_NAME移至各测试文件内独立定义。(2) nan_to_num改为OverrideOperators.cpp显式注册。
- Reviewability: high -- 模块级设备查询是已知反模式
- Review rule: 测试公共模块禁止模块级设备查询；dispatch注册方式需与注册表类型一致

### D-540: eraseStream迭代器失效后继续使用(UB)

- Hashes: 2a6cf5491 [+1: 8311f6cc8]
- Root cause: STL容器erase后迭代器失效
- Files: NPUCachingAllocator.cpp
- Defect: `npu_events.erase(it)`后`it`已失效，继续`++it`是未定义行为。
- Fix: 改为`it = npu_events.erase(it)`使用erase返回的有效迭代器。
- Reviewability: high -- C++ STL基础知识，静态分析可检测
- Review rule: STL容器erase必须使用返回值更新迭代器

### D-541: Revert trace_view/trace_step_time功能(代码质量不达标)

- Hashes: 5101b63f2 [+2: 790baacc8, ee20c8918], 42373ef2d
- Root cause: 功能代码质量不达标(调试代码残留)
- Files: trace_step_time.py(删除), trace_view_parser.py
- Defect: PR !5397/!5398/!5400(trace_view)和!5321(trace_step_time)合入后发现代码质量问题
  (大量print调试语句残留等)，功能不稳定。回退4个相关PR。
- Fix: 全量回退，删除TraceStepTimeParser类和CSV生成逻辑。
- Reviewability: high -- PR review时应发现调试代码残留
- Review rule: 合入前检查print/debug残留；新增profiler解析器需配套UT

### D-542: make_tensor多余参数(D-544修复的遗漏调用点)

- Hashes: 67619ce81
- Root cause: 批量API变更时遗漏调用点(incomplete fix follow-up)
- Files: OpPreparation.cpp
- Defect: D-544删除`NPUTensorImpl._storage_impl`后`make_tensor`签名从2参变为1参，
  但`OpPreparation.cpp`中的一处调用未同步更新，编译失败。
- Fix: 删除多余的`storage_impl`参数。
- Reviewability: high -- 编译器直接报错
- Review rule: API签名变更后全仓grep确认所有调用点

### D-543: to.dtype对double类型走NPU慢路径

- Hashes: bd74f3f47
- Root cause: NPU不支持double但未提前拦截
- Files: ToKernelNpuOpApi.cpp
- Defect: `to(dtype=torch.float64)`直接走`npu_dtype_cast`，NPU不支持double导致极慢fallback路径。
- Fix: 新增double类型提前拦截，自动降为float32并打一次警告。
- Reviewability: medium -- 需知NPU不支持double的硬件约束
- Review rule: NPU不支持的dtype在入口处应有快速fallback而非静默走慢路径

### D-544: NPUTensorImpl私有_storage_impl强引用导致内存泄漏

- Hashes: 5fd5bc92b [+1: 2509fb638]
- Root cause: 自定义TensorImpl额外持有storage强引用
- Files: NPUTensorImpl.cpp/h, TensorFactories.cpp, ReplayGraph.cpp, CalcuOpUtil.cpp
- Defect: `NPUTensorImpl`私有成员`_storage_impl`(intrusive_ptr)额外持有storage强引用。
  `tensor.data`释放时storage引用计数不归零，NPU设备内存泄漏。
- Fix: 删除`_storage_impl`成员及构造参数，析构函数改为空实现，统一make_tensor签名。
- Reviewability: medium -- 需理解PyTorch storage引用计数机制
- Review rule: 自定义TensorImpl不应额外持有storage强引用，基类已管理storage生命周期

### D-545: ACL_OP_DEBUG_LEVEL合法值范围不完整

- Hashes: cefe1daef [+3: 23672f97c, 2970efb15, 7a0033114]
- Root cause: 配置校验合法值集合未随CANN更新
- Files: npu_config.py
- Defect: `ACL_OP_DEBUG_LEVEL`校验仅允许`["0","1","2"]`，CANN新增level 3/4未同步。
- Fix: 扩展为`["0","1","2","3","4"]`，错误信息增加实际输入值显示。
- Reviewability: high -- 常量集合遗漏
- Review rule: CANN配置枚举扩展时需同步更新PTA侧校验

### D-546: avg_pool_3d padding参数格式错误

- Hashes: 9fd0b9267 [+1: 23ace4d55]
- Root cause: ACL算子padding参数格式理解错误
- Files: AvgPool3dKernelNpu.cpp, test_avg_pool3d.py
- Defect: padding传给ACL时用`{0,0,0,padT,padH,padW}`(6元素非对称)，正确格式应为
  `{padT,padT,padH,padH,padW,padW}`(对称padding)。输出tensor shape因此计算错误。
- Fix: 修正padding格式为对称形式，重构参数解析和输出size计算。
- Reviewability: high -- 对照ACL文档可发现参数格式不一致
- Review rule: ACL算子调用前需对照文档确认参数格式(padding/stride元素数和顺序)

### D-547: v2.0.1 checkpoint upstream重构适配

- Hashes: 06eeee147
- Root cause: upstream checkpoint重构后NPU侧未同步
- Files: checkpoint.py (+193行)
- Defect: PyTorch重构checkpoint(支持非tensor参数save_for_backward、non-reentrant路径等)，
  NPU的override未同步，非tensor参数crash、overflow标志丢失、autocast不兼容。
- Fix: 同步upstream重构：非tensor参数、overflow标志保存/恢复、autocast双路、non-reentrant。
- Reviewability: low -- 需跟踪upstream checkpoint演进
- Review rule: upstream工具函数(checkpoint/autocast)重构后需检查NPU override是否同步

### D-548: cast反向传播递归调用自身autograd入口

- Hashes: 66b1e3195
- Root cause: autograd Function backward调用自身的apply
- Files: CastKernelNpuOpApi.cpp
- Defect: cast算子backward中用`NPUDtypeCastOpApiFunction::apply`做grad类型回转，
  再次注册autograd节点形成递归链。
- Fix: 改为`.to(dtype)`让dispatch系统自动路由。
- Reviewability: high -- autograd backward不应调用同一Function的forward
- Review rule: 自定义autograd Function的backward禁止调用同一Function的apply/forward

### D-549: profiler NPU_ID硬编码常量与底层数据格式不匹配

- Hashes: 1eef5ca0d
- Root cause: 硬编码常量未随数据格式更新(与D-64同族)
- Files: memory_use_bean.py
- Defect: profiler解析内存数据时`NPU_ID = 9`用于区分设备类型，底层格式变更后正确值为`20`。
- Fix: `NPU_ID`从9改为20。
- Reviewability: medium -- 需了解底层数据格式定义
- Review rule: 协议常量应引用公共定义，不应在消费方重复声明

### D-550: true_divide缺少类型提升和scalar输入路径

- Hashes: d0c7954c8
- Root cause: 算子实现缺少dtype promotion + scalar重载
- Files: TrueDivideKernelNpu.cpp, test_true_divide.py
- Defect: (1) 整型tensor做除法未提升到浮点类型，结果dtype错误。
  (2) 缺少`true_divide(Tensor, Scalar)`重载，scalar输入报错。
- Fix: 新增`get_divide_high_type`类型提升逻辑和scalar输入路径。
- Reviewability: high -- PyTorch的true_divide语义明确要求浮点结果
- Review rule: 除法类算子必须实现dtype promotion(整型->浮点)且支持scalar输入

### D-551: matmul backward维度处理错误(1D/ND混合场景) + D-565临时回退

- Hashes: 018e98545 (正确修复), a4a3fde1f (临时回退)
- Root cause: matmul反向传播对1D/2D/ND混合维度的梯度变换不正确
- Files: MatmulKernelNpuOpApi.cpp, npu_native_functions.yaml, test_matmul.py
- Defect: matmul op_api反向未正确处理1D/2D/ND混合unsqueeze/squeeze/reshape。
  D-565(a4a3fde1f)先临时关闭op_api路径回避OOM，D-551随后正确修复并重新开启。
- Fix: 拆分`matmul_mat1_backward`和`matmul_mat2_backward`，各自处理维度变换。
- Reviewability: medium -- 需理解matmul各维度组合的梯度公式
- Review rule: matmul backward需覆盖1D-1D/1D-2D/2D-1D/2D-2D/ND-ND全组合测试

### D-552: indexput bool index未展开 + upsamplebilinear2d算子名大小写错误

- Hashes: 953497739
- Root cause: (1) bool mask路径遗漏展开 (2) ACL算子名大小写不匹配
- Files: IndexPutKernelNpuOpApi.cpp, UpsampleBilinear2dKernelNpuOpApi.cpp, AdvancedIndex.cpp/h
- Defect: (1) indexput aclnn路径跳过`npu_expand_tensors`将bool mask展开为index tensor。
  (2) `aclnnUpsampleBilinear2D`(大写D)应为`aclnnUpsampleBilinear2d`(小写d)，名不匹配。
- Fix: (1) 增加bool mask展开，新增`flag_aclnn`控制nonzero转置。(2) 修正大小写。
- Reviewability: high -- 对照非op_api路径可发现缺步骤；算子名应从注册表复制
- Review rule: op_api路径需逐步对照非op_api确认预处理完整；ACL算子名从注册表复制不手写

### D-553: ones_like在非NPU设备上错误走NPU路径

- Hashes: 6f7642cf2
- Root cause: op_api实现缺少设备类型检查
- Files: OnesLikeKernelNpuOpApi.cpp
- Defect: `ones_like`注册到dispatch key后，CPU tensor调用也进入NPU路径，导致错误。
- Fix: 新增设备类型检查，非NPU设备fallback到`empty_like + fill_(1.)`。
- Reviewability: high -- dispatch注册时应考虑非NPU tensor调用路径
- Review rule: dispatch key注册的算子必须检查输入设备类型

### D-554: torch._C._nn._parse_to缺少NPU设备guard

- Hashes: 49148ae69
- Root cause: C++内部函数未被NPU设备guard包装
- Files: module.py
- Defect: `torch._C._nn._parse_to`是`module.to()`底层C++实现，未被`@torch_device_guard`
  包装，NPU设备在module.to()调用链中不能被正确识别。
- Fix: monkey-patch加上device guard包装。
- Reviewability: medium -- 需了解module.to()完整调用链
- Review rule: NPU设备guard需覆盖所有device resolution路径(含C++内部函数)

### D-555: Revert flash_attention is_flash动态推断(引入回归)

- Hashes: 9be98a43b
- Root cause: is_flash动态推断破坏forward/backward输出结构一致性
- Files: FlashAttentionKernelNpuOpApi.cpp
- Defect: `is_flash`从硬编码`true`改为基于`key.size(1) > FLASH_THRESHOLD(512)`动态推断，
  导致非flash路径的输出结构(softmax_out/dpse)与backward参数不匹配。
- Fix: 全量回退，恢复`is_flash = true`。
- Reviewability: medium -- 路径切换涉及forward/backward/输出结构三方联动
- Review rule: FlashAttention路径切换需同步修改forward+backward+输出tensor列表

### D-556: AvgPool2d缺少stride=0和padding越界校验

- Hashes: d8b5011e8
- Root cause: 算子参数校验缺失
- Files: AvgPool2dKernelNpuOpApi.cpp
- Defect: stride为0时除零，padding超过kernel_size/2时底层算子行为未定义。均无前置校验。
- Fix: 新增stride!=0和padding<=kernel_size/2的检查。
- Reviewability: high -- 标准参数校验
- Review rule: pooling算子必须校验stride>0和padding<=kernel_size/2

### D-557: aclrtMallocAlign32回退为aclrtMalloc

- Hashes: ba70dece2
- Root cause: 对齐分配API引入回归
- Files: NPUCachingAllocator.cpp
- Defect: 从`aclrtMalloc`切换到`aclrtMallocAlign32`(32字节对齐分配)后出现问题(回归症状未记录)。
- Fix: 回退为标准`aclrtMalloc`。
- Reviewability: low -- 需了解两种分配API行为差异
- Review rule: 内存分配API切换需多SoC+多场景充分验证

### D-558: ComplexHalf类型映射为ACL_DT_UNDEFINED

- Hashes: 2cc9bc67c
- Root cause: 新增数据类型的ACL映射缺失
- Files: acl_base.h
- Defect: `at::ScalarType::ComplexHalf`映射到`ACL_DT_UNDEFINED`，complex32 tensor在ACL层无效。
- Fix: 新增`ACL_COMPLEX32 = 33`枚举值，建立正确映射。
- Reviewability: high -- 类型映射表遗漏
- Review rule: 新增ScalarType支持时必须同步更新ACL类型映射表

### D-559: complex运算中view_as_real视图别名损坏NPU StorageDesc

- Hashes: e6688feeb
- Root cause: NPU StorageDesc在视图操作中被共享修改
- Files: complex_compute相关
- Defect: `view_as_real`返回原tensor视图，后续操作修改视图时同时损坏原tensor的NPU StorageDesc。
- Fix: 对`view_as_real`结果加`.clone()`切断视图别名。
- Reviewability: medium -- 需理解NPU StorageDesc与视图操作的交互
- Review rule: NPU StorageDesc可能在视图间共享，对视图的修改性操作前需clone

### D-560: ccache不存在时CMake构建失败

- Hashes: 6a858ae5f
- Root cause: CMake脚本对可选工具的健壮性不足
- Files: CMakeLists.txt
- Defect: `find_program(CCACHE_FOUND ccache)`找不到时返回空值，赋给compiler launcher导致构建错误。
- Fix: 增加NOTFOUND判断，不存在时跳过而非赋空值。
- Reviewability: high -- CMake标准模式
- Review rule: find_program结果必须检查NOTFOUND再使用

### D-561: NPUQueue TTL时间片机制回退(含时间转换错误)

- Hashes: d104b149e, 58ae668ce
- Root cause: TTL机制增加复杂性但无预期收益 + 时间单位转换错误
- Files: NPUQueue相关
- Defect: (1) TTL/时间片轮询机制(MAX_TTL_TIME + select)增加复杂性但未带来性能提升。
  (2) `GET_MSEC`宏纳秒转毫秒除数错误(/1000应为/1000000)。
- Fix: 全量回退TTL机制，恢复eventfd阻塞等待方式。
- Reviewability: medium(架构) / high(时间转换)
- Review rule: 线程调度机制变更需性能benchmark；时间单位转换应使用命名常量

### D-562: logaddexp/logaddexp2未处理broadcast和dtype推断

- Hashes: 37e48d896
- Root cause: 二元算子缺少broadcast+dtype promotion标准处理
- Files: LogaddexpKernelNpu相关
- Defect: (1) `logaddexp_out`未做broadcast shape计算和output合法性检查。
  (2) 输出用`self.options()`分配未推断公共dtype。
- Fix: 新增broadcast shape计算、CheckOut、result_type推断。
- Reviewability: high -- PyTorch二元算子标准模式
- Review rule: 二元算子必须实现broadcast shape推断和result_type dtype提升

### D-563: remainder输出shape推断逻辑多余导致特定case错误

- Hashes: d588783d1
- Root cause: 输出shape分支逻辑过于复杂
- Files: RemainderKernelNpu相关
- Defect: broadcast_shape与self_shape的分支比较逻辑多余，特定shape组合取了错误shape。
- Fix: 统一用broadcast_shape，删除多余分支。
- Reviewability: high -- 简化后逻辑更清晰
- Review rule: 输出shape直接用broadcast结果，不与输入shape做多余比较

### D-564: TensorIterator不识别BFloat16类型

- Hashes: 858ed72ec
- Root cause: 类型提升逻辑缺少BFloat16分支
- Files: NPUTensorIterator.cpp
- Defect: NPUTensorIterator类型提升不含BFloat16，bf16 tensor被错误提升到其他类型。
- Fix: 补充BFloat16类型提升条件。
- Reviewability: high -- 枚举遗漏
- Review rule: 类型dispatch/promotion新增dtype支持时需检查所有分支

### D-565: optimizer partial()不支持子类化

- Hashes: 22217225f
- Root cause: functools.partial包装类破坏isinstance和继承
- Files: optimizer相关
- Defect: `partial(SrcXXX, foreach=False)`返回非真正的类，子类化和isinstance检查失败。
- Fix: 新增`partialclass`工具(基于partialmethod)保持正常继承行为。
- Reviewability: medium -- Python functools.partial对类的行为不直觉
- Review rule: 类的参数固定应使用元类/工厂模式而非functools.partial

### D-566: SyncBatchNorm backward count_all类型不匹配

- Hashes: ac06388a9
- Root cause: save_for_backward保存的tensor dtype未控制
- Files: SyncBatchNorm相关
- Defect: `count_all`保存为int64，backward底层算子期望int32，类型不匹配。
- Fix: 保存前转换为`torch.int32`。
- Reviewability: high -- dtype mismatch在backward时可报错定位
- Review rule: save_for_backward保存的tensor应确保dtype与backward使用方一致

### D-567: NPU event生命周期无追踪导致内存过早释放

- Hashes: f0bbf40d0
- Root cause: task queue模式下event完成状态与内存释放不同步
- Files: NPUCachingAllocator, NPUEventManager, OpParamMaker
- Defect: task queue模式下event录入queue但未完成时，processEvents可能释放关联内存。
  缺少"event已录入但未完成"的状态追踪。
- Fix: 引入`recorded_events`(std::set)和互斥锁，processEvents仅处理已recorded的event。
  OpParamMaker录入时Insert，EventManager销毁时Erase，形成完整生命周期追踪。
- Reviewability: medium -- 需理解task queue的异步执行模型
- Review rule: 异步模型中资源释放必须与异步操作完成状态关联
### D-568: HCCL多流内存复用架构重设计(eraseStream替代recorded_events)

- Hashes: 674cbdaff
- Root cause: recordStream机制无法可靠回收HCCL通信后内存
- Files: NPUCachingAllocator.cpp, ProcessGroupHCCL.cpp, OpParamMaker.cpp (+4)
- Defect: HCCL多流场景下，`recordStream`机制用event标记block归属stream，
  通信完成后需等event record完成才释放内存。但`recorded_events`集合的查询阻塞了
  `process_events`/`synchronize_and_free_events`路径，导致内存无法及时回收。
- Fix: 新增`eraseStream(Block*, NPUStream)`方法，在`WorkHCCL::synchronize()`中
  主动从block的stream_uses移除stream并lazy destroy event。用weak_intrusive_ptr
  追踪参与通信的tensor storage。环境变量`MULTI_STREAM_MEMORY_REUSE`控制开关。
  移除了全局recorded_events机制。
- Reviewability: low -- 涉及allocator/HCCL/event三层交互的时序分析
- Review rule: 多流内存管理变更需验证所有stream组合的event生命周期

### D-569: recorded_events集合中event销毁后残留

- Hashes: 1a9e24331
- Root cause: event销毁路径未同步清理关联数据结构
- Files: NPUCachingAllocator.cpp, NPUEventManager.cpp, AclInterface.h
- Defect: NPUEventManager的`QueryAndDestroyEvent()`销毁event后，allocator的
  `recorded_events`集合中仍残留该event记录。后续`process_events`查询已销毁event
  时逻辑错误，阻塞内存回收。
- Fix: 新增`NpuAllocatorEraseRecordedEvent`公开API，EventManager销毁event前
  调用此函数从allocator中移除记录。
- Reviewability: high -- 资源生命周期的配对操作(insert/erase)缺失是典型遗漏
- Review rule: 集合insert操作必须有对应的erase操作覆盖所有销毁路径

### D-570: lt_out输出类型校验硬编码kBool

- Hashes: ed069e9f7
- Root cause: output dtype检查参数硬编码
- Files: LtKernelNpuOpApi.cpp
- Defect: `lt_out`的`CheckOut`将期望dtype硬编码为`at::kBool`，但用户可传入
  非Bool类型的result tensor。校验错误地拒绝合法输出。
- Fix: 将`at::kBool`改为`result.scalar_type()`，使用result自身类型做校验。
- Reviewability: high -- 硬编码常量替代变量是典型copy-paste遗留
- Review rule: CheckOut的dtype参数应来自result tensor而非硬编码

### D-571: LSTM backward遗漏direction参数(双向LSTM梯度计算错误)

- Hashes: 67de7133f
- Root cause: backward路径参数缺失
- Files: LstmKernelNpu.cpp
- Defect: `lstm_backward_out_npu`中direction属性硬编码为`"UNIDIRECTIONAL"`，
  forward中的`flag_direction`未传递到backward。双向LSTM反向传播方向不匹配，梯度错误。
- Fix: 新增`flag_direction`参数贯穿forward→ctx.saved_data→backward→kernel属性设置。
  根据flag设置`"REDIRECTIONAL"`或`"UNIDIRECTIONAL"`。
- Reviewability: high -- forward保存了flag但backward未使用，对照即可发现
- Review rule: autograd Function中forward保存的状态在backward中应全部使用

### D-572: NPU autograd函数识别逻辑使用错误数据源

- Hashes: 63f195876
- Root cause: structured op缓存不包含custom autograd函数
- Files: codegen/utils.py
- Defect: `filt_npu_autograd_functions`用`GLOBAL_STRUCTURED_OP_INFO_CACHE`判断
  哪些函数需要autograd wrapper，但该缓存只含structured op，不含custom autograd函数。
  部分NPU custom autograd算子未被识别，autograd链断裂。
- Fix: 从`npu_native_functions.yaml`解析`custom_autograd`字段建立集合，
  以此判断需要生成autograd wrapper的函数。parse_npu_yaml返回类型改为Dict。
- Reviewability: medium -- 需要理解codegen数据流中structured vs custom的区别
- Review rule: codegen数据源切换时确认新数据源的覆盖范围是否包含旧源的全集

### D-573: batch_norm 5D tensor reshape后引用断裂

- Hashes: b15319ee4
- Root cause: reshape产生新tensor丢失与调用方的引用关系
- Files: native_batch_norm相关
- Defect: (1) `native_batch_norm`中result未预分配就传给out函数，状态不正确。
  (2) 5D tensor场景下out被reshape后成为新tensor，调用方持有的out引用指向原tensor，
  拿不到正确结果。
- Fix: (1) result预先ApplyTensor分配。(2) 引入`out_reshape`中间变量做计算，
  最终结果赋回out，保持引用一致性。
- Reviewability: medium -- reshape是否创建新tensor取决于stride，需仔细追踪
- Review rule: out变体算子禁止对out做reshape/view后直接写入，须用中间变量再赋回

### D-574: embedding_bag CPU fallback遗漏padding_idx参数

- Hashes: 58fc914b2
- Root cause: 函数调用参数遗漏
- Files: embedding_bag相关
- Defect: NPU的`_embedding_bag`走CPU fallback调用`_embedding_bag_cpu`时，
  遗漏了`padding_idx`参数。padding token的贡献未被忽略，结果包含不应计入的token。
- Fix: 补充`padding_idx`参数传递。
- Reviewability: high -- 对照函数签名即可发现参数遗漏
- Review rule: CPU fallback调用须完整传递所有参数，参数数量需与目标函数签名一致

### D-575: format_contiguous与copy_optimize合并引入正确性回归

- Hashes: c92556c2e [+1 cherry-pick: 1fc8ebeaf]
- Root cause: 函数合并丢失了"是否需要copy优化"的选择权
- Files: NpuUtils.cpp/h, BmmKernelNpu.cpp, MmKernelNpu.cpp, OpCommand.cpp (+3)
- Defect: `format_contiguous`与`format_contiguous_add_copy_optimize`合并为一个函数后，
  所有调用路径都走copy优化逻辑。但copy优化不是所有场景适用，某些路径(如clone)
  不应做优化转换，合并后引入正确性问题。
- Fix: Revert合并，恢复为两个独立函数。调用方按需选择是否带copy优化。
- Reviewability: medium -- 函数合并时需验证所有调用方是否都适用新逻辑
- Review rule: 合并功能相近的函数前，确认所有调用方的使用场景都兼容合并后的语义

### D-576: HCCL BFloat16类型映射缺失

- Hashes: 722afd152
- Root cause: 类型映射表不完整
- Files: hccl_adaptor相关
- Defect: `kScalarTypeToHcclDataType`映射表缺少`{at::kBFloat16, HCCL_DATA_TYPE_BFP16}`，
  AllReduce等集合通信使用bf16数据时直接报"Unsupported data type"。
- Fix: 映射表增加BFloat16条目，支持列表增加BFP16。
- Reviewability: high -- 新增dtype支持时需同步更新所有类型映射表
- Review rule: 新增dtype支持的checklist应包含通信库类型映射表的更新

### D-577: gen_variable_type codegen多处错误(路径/过滤/模板/函数名)

- Hashes: 990510041
- Root cause: codegen重构中多个细节未对齐
- Files: gen_autograd.py, gen_variable_type.py, custom_functions.py, utils.py (+3)
- Defect: (1) inplace_or_view_type生成用了错误的函数列表(PyTorch全量而非NPU子集)。
  (2) NPU autograd函数过滤正则匹配不精确。(3) 生成代码包含NPU不支持的
  jit_decomposition路径。(4) impl_name映射错误dispatch到错误实现。
- Fix: 函数列表参数改为npu_funcs，过滤改为按后缀枚举查找，生成代码regex移除
  jit相关逻辑，impl_name改用type_wrapper_name。
- Reviewability: low -- codegen修改需端到端验证生成产物正确性
- Review rule: codegen变更必须对比变更前后生成的C++代码差异

### D-578: dropout genmask多流内存复用(event同步替代recordStream)

- Hashes: f9b761589
- Root cause: recordStream机制在此场景下不够可靠
- Files: FlashAttentionKernelNpuOpApi.cpp
- Defect: `dropout_gen_mask_dispatch`使用`recordStream`标记mask内存在secondary stream
  上使用，但event record时机问题导致mask内存在计算完成前被复用。
- Fix: 移除recordStream，改用显式NPUEvent同步：主stream record event，
  secondary stream wait event，保证执行顺序。forward和backward中都插入event同步。
- Reviewability: low -- 需理解多流event同步模型才能判断recordStream是否充分
- Review rule: 跨stream资源共享优先使用显式event同步而非recordStream

### D-579: bincount空输入+minlength>0应返回非空tensor

- Hashes: 8ae04a1f5
- Root cause: 边界条件处理不符合PyTorch语义
- Files: BincountKernelNpuOpApi.cpp
- Defect: 输入为空(`numel()==0`)且`minlength>0`时返回shape `{0}`的空tensor。
  PyTorch语义要求此时返回长度为minlength的全零tensor。
- Fix: 空输入+minlength>0时分配shape `{minlength}`的tensor并调用aclnnBincount。
- Reviewability: high -- 对照PyTorch文档中bincount对空输入的描述即可发现
- Review rule: 算子实现需对照PyTorch文档验证空输入+非默认参数的组合行为

### D-580: 动态加载函数缺失时TORCH_WARN每次调用触发(日志刷屏)

- Hashes: 8be844c47 [+2: 6f31c06fa, 4a4d91d84]
- Root cause: 告警级别与触发频率不匹配
- Files: AclInterface.cpp
- Defect: `AclrtSynchronizeStreamWithTimeout`/`AclrtDestroyStreamForce`/`AclrtMallocAlign32`
  等函数通过dlsym动态加载，找不到时TORCH_WARN在每次调用时都打印。旧CANN版本上
  这些函数不可用是常态，每次调用都warn产生大量日志噪音。
- Fix: 改用`TORCH_NPU_WARN_ONCE`(后续版本)/`TORCH_WARN_ONCE`(早期版本)只首次告警。
- Reviewability: high -- fallback路径中的TORCH_WARN应默认为WARN_ONCE
- Review rule: 可预期的运行时fallback路径使用WARN_ONCE而非WARN

### D-581: AclrtMallocAlign32在旧CANN版本不可用需回退

- Hashes: 1b5858bc1 [+2: d8ce5d4e3, 045412563]
- Root cause: 新API可用性未充分验证即全量替换
- Files: NPUCachingAllocator.cpp, AclInterface.cpp
- Defect: 引入`AclrtMallocAlign32`(32字节对齐)替换`aclrtMalloc`后，某些CANN版本
  不支持该API。fallback逻辑虽有，但实际使用中仍触发问题(具体错误未记录)。
  涉及3个分配路径：`npu_malloc_retry`(2处)和`DeviceCachingAllocator::alloc_block`(1处)。
- Fix: 全部回退为标准`aclrtMalloc`。
- Reviewability: medium -- 新API引入需在所有目标环境验证
- Review rule: 替换底层内存分配API前需在最低支持的CANN版本上验证

### D-582: threshold out变体dtype推断缺失+inplace使用错误算子

- Hashes: 01463a0c4
- Root cause: out变体未做dtype promotion + inplace走out路径
- Files: ThresholdKernelNpuOpApi.cpp
- Defect: (1) `threshold_out`的CheckOut直接用self类型，未做result_type推断。
  当self和result dtype不同时校验错误。(2) threshold_用out路径实现(self传为out)，
  但应使用专用的`aclnnInplaceThreshold`算子。
- Fix: (1) 用`at::result_type(self, result)`计算结果dtype。
  (2) inplace版本改用aclnnInplaceThreshold。
- Reviewability: high -- out变体必须做dtype promotion是标准规范
- Review rule: out变体算子实现checklist: dtype promotion + 专用inplace算子

### D-583: cumsum忽略用户指定的output dtype

- Hashes: d3914eb96
- Root cause: 输出tensor创建未使用dtype参数
- Files: CumsumKernelNpu.cpp
- Defect: `cumsum(self, dim, dtype)`中无条件`ApplyTensor(self)`创建输出，
  dtype参数被忽略。用户指定float64但输出仍为输入类型。
  Bool输入也应默认转为kLong(PyTorch语义)。
- Fix: 有dtype参数时用指定dtype创建输出；self为Bool时用kLong；否则保持输入类型。
- Reviewability: high -- 对照函数签名中dtype参数是否被使用即可发现
- Review rule: 带optional<dtype>参数的算子实现必须检查该参数的使用

### D-584: sum CheckOut参数类型错误 + upsample_nearest_1d传空output_size

- Hashes: 4e3f021b7
- Root cause: 参数类型混淆 + 条件分支遗漏
- Files: SumKernelNpuOpApi.cpp, UpsampleNearest1dKernelNpuOpApi.cpp
- Defect: (1) sum的CheckOut第三参数传了tensor(self)而非scalar_type(res_type)。
  (2) upsample_nearest_1d用`output_size`(可能为空)传给EXEC_NPU_CMD，
  用户通过scale_factor指定时output_size为空。
- Fix: (1) CheckOut参数改为res_type。(2) 用compute_size计算后的result_size传入。
- Reviewability: high -- CheckOut参数类型错误编译不报错(隐式转换)
- Review rule: CheckOut调用时对照其签名验证每个参数的类型和语义

### D-585: mm/Linear NZ格式下转置判断函数不正确

- Hashes: 55e052cdf [+1 cherry-pick: bbcbbdc51]
- Root cause: 转置判断函数对特定stride pattern失效
- Files: MmKernelNpu.cpp, CalcuOpUtil.cpp/h
- Defect: `IsTransposeLastTwoDims`对某些stride pattern判断转置不正确。
  ND到NZ格式on-the-fly转换的边界条件不足(内轴字节数<256时不应转换)。
  最后return语句的逻辑运算符优先级错误(`&&`应为`||`)。
- Fix: 新增`IsMmTranspose`专用函数(通过stride直接判断)。增加`kInnerAxisMinBytes=256`
  边界检查。修正return语句逻辑。
- Reviewability: medium -- stride-based转置判断需要对NZ格式深入理解
- Review rule: NZ格式转换函数的边界条件需覆盖小shape和非标准stride场景

### D-586: repeatInterleave.Tensor NPU实现存在问题被回退到CPU

- Hashes: c02baeae4 [+1 cherry-pick: f781f20ac]
- Root cause: NPU算子实现正确性问题(具体bug未记录)
- Files: npu_native_functions.yaml, RepeatInterLeaveKernelNpu.cpp, RepeatInterleaveKernelNpuOpApi.cpp
- Defect: `repeat_interleave.Tensor`变体的NPU实现(`aclnnRepeatInterleaveTensor`)
  存在正确性问题(commit中未详述具体错误)。
- Fix: 将该算子从supported移到tocpu列表，fallback到CPU执行。删除NPU实现代码。
- Reviewability: low -- 算子正确性需要全面的数值测试覆盖
- Review rule: 新算子上线前需通过PyTorch reference实现的全覆盖数值比对

### D-587: codegen derivatives.yaml中NPU与PyTorch同名函数冲突

- Hashes: 72e4e3675
- Root cause: NPU derivatives.yaml未排除已在PyTorch中注册的函数
- Files: codegen/gen_variable_type.py, codegen/utils.py, VariableType.cpp
- Defect: NPU的derivatives.yaml中包含与PyTorch native_functions.yaml同名的函数，
  codegen生成的VariableType代码中出现重复注册，导致dispatch冲突或编译错误。
- Fix: codegen时先解析PyTorch的native_functions.yaml获取所有torch函数名，
  然后过滤掉NPU derivatives.yaml中的同名函数，只保留NPU独有的autograd函数。
- Reviewability: medium -- 需要理解codegen中两个yaml文件的关系
- Review rule: NPU codegen须显式排除与upstream同名的函数定义

### D-588: elapsed_time在NPU上可能返回负值

- Hashes: d5f765f58
- Root cause: NPU event时间戳不保证单调递增
- Files: reducer_npu.cpp
- Defect: DDP reducer中`measureDifference`调用event的`elapsed_time`，
  NPU event时间戳可能不单调(硬件时钟特性)，返回负值导致异常。
- Fix: (1) 非Detail级别直接跳过时间计算返回nullopt。
  (2) Detail级别用try-catch包裹，异常时返回-1。
- Reviewability: medium -- NPU event时间戳特性是硬件相关的隐式约束
- Review rule: NPU event elapsed_time调用方须处理负值和异常场景

### D-589: exp_out CheckOut使用self的format而非result的

- Hashes: dc07ea827
- Root cause: CheckOut参数选错数据来源
- Files: ExpKernelNpuOpApi.cpp
- Defect: `exp_out`的CheckOut传入`CalcuOpUtil::GetTensorNpuFormat(self)`作为format参数，
  但应使用result的format。当self和result format不同时校验不正确。
- Fix: CheckOut参数改为直接传入result(由CheckOut内部推断format)。
- Reviewability: high -- out变体的CheckOut应以result为参考而非self
- Review rule: out变体算子的CheckOut统一使用result作为format参考

### D-590: GRU不支持变长PackedSequence输入(NPU算子要求fixed-length)

- Hashes: 6bfdbb5e9
- Root cause: NPU GRU算子不支持变长输入但未做格式转换
- Files: torch_npu/utils/module.py, test_gru.py
- Defect: NPU的GRU算子要求fixed-length格式输入，但Python层直接将compact格式的
  PackedSequence传入，导致变长序列场景下GRU计算错误或失败。LSTM已有类似patch但GRU遗漏。
- Fix: monkey-patch `GRU.forward`，当输入为PackedSequence且batch_sizes在CPU时，
  将compact格式转换为fixed-length格式再送入GRU，计算后转换回来。
- Reviewability: medium -- 需要了解LSTM已有类似patch，GRU应对称实现
- Review rule: NPU RNN系列算子(LSTM/GRU/RNN)须保持输入格式转换逻辑对称

### D-591: op wait timeout设置时机在stream初始化(HCCL尚未初始化)

- Hashes: 3243fbc09
- Root cause: 初始化时序依赖
- Files: NPUStream.cpp, ProcessGroupHCCL.cpp
- Defect: op wait timeout在stream全局初始化(`initGlobalStreamState`)时设置，
  但此时HCCL尚未初始化，`GetHCCLExecTimeout`获取到的值可能不正确(默认值或零)。
- Fix: 将timeout设置从stream初始化移到ProcessGroupHCCL构造函数中，
  此时HCCL已完成初始化，timeout值可正确获取。
- Reviewability: high -- 初始化时序依赖应通过代码注释或assert显式标注
- Review rule: 依赖其他子系统状态的初始化逻辑应放在该子系统初始化完成之后

### D-592: sum dtype计算缺少类型提升(half溢出)

- Hashes: c50aa1a10
- Root cause: 累加计算未做dtype promotion
- Files: SumKernelNpu.cpp, test_sum.py
- Defect: sum算子在指定output dtype时未在计算前做类型提升。
  例如half输入指定float32输出，计算仍在half精度下进行，中间结果溢出。
  空tensor分支的结果dtype也使用了默认类型而非指定类型。
- Fix: 新增`check_dtype`辅助函数做计算前类型转换。整数类型统一转float。
  空tensor分支使用res_type创建结果。
- Reviewability: high -- 对照PyTorch sum文档的dtype参数语义即可发现
- Review rule: reduction算子指定output dtype时须在计算前提升输入精度

### D-593: TORCH_WARN在NPU多设备/多线程环境下不工作

- Hashes: 6e349607c
- Root cause: PyTorch原生warning handler与NPU环境不兼容
- Files: 53个文件(全代码库)
- Defect: PyTorch的`TORCH_WARN`宏使用的warning handler在NPU多设备/多线程环境下
  存在兼容性问题(具体表现为warning不输出或异常)。
- Fix: 实现NPU专用告警基础设施：`NPUException.cpp`中`warn_`函数和`getBaseHandler_`，
  定义`TORCH_NPU_WARN`/`TORCH_NPU_WARN_ONCE`宏系列。
  全代码库50+文件替换为NPU专用宏。
- Reviewability: low -- 基础设施级兼容性问题需要运行时诊断
- Review rule: 第三方设备扩展不应依赖PyTorch内部的warning handler实现

### D-594: matmul hostapi路径启用后出现问题被回退

- Hashes: 2c986cccd
- Root cause: hostapi路径实现不完善即全量启用
- Files: npu_native_functions.yaml
- Defect: matmul/matmul.out的`op_api`设为True(启用host API路径)后出现问题
  (具体错误未记录)。
- Fix: op_api回退为False，关闭host API路径。
- Reviewability: medium -- 新路径启用需要全覆盖回归测试
- Review rule: 算子执行路径切换(op_api True/False)须通过完整UT回归

### D-595: 文件写入O_WRONLY未截断已有内容

- Hashes: 406e891a9
- Root cause: POSIX文件打开标志语义理解错误
- Files: setup.py
- Defect: `_rewrite_ld_preload`用`os.O_CREAT|os.O_WRONLY`打开文件，文件已存在时
  不会截断(需要O_TRUNC)，旧内容残留在新内容之后。
- Fix: 写入前先检查文件是否存在，存在则unlink删除后重新创建。
- Reviewability: high -- O_WRONLY不截断是POSIX基础知识
- Review rule: os.open写入文件时必须包含O_TRUNC或先删除已有文件

### D-596: ACL结构体新增指针字段未初始化(野指针)

- Hashes: 82b6576b3
- Root cause: 第三方结构体更新后使用方未同步初始化
- Files: acl_rt.h, Module.cpp
- Defect: `aclrtUtilizationInfo`结构体新增`utilizationExtend`指针成员(跟随ACL版本更新)，
  但Module.cpp中初始化`util_info`时未设置该字段为nullptr。
  调用`AclrtGetDeviceUtilizationRate`时野指针可能导致崩溃。
- Fix: 初始化时补充`utilizationExtend = nullptr`。
- Reviewability: high -- 结构体成员变更时编译器不会报warning
- Review rule: 更新第三方头文件后grep所有该结构体的使用点确认初始化完整

### D-597: indexput缺少value与index结果的shape校验

- Hashes: 865b36da2
- Root cause: 输入校验缺失
- Files: IndexPutKernelNpu.cpp, test_index_put.py
- Defect: `_index_put_impl_`未检查value tensor的shape是否能broadcast到
  index结果的shape。shape不匹配时直接传给底层算子，产生不确定结果而非报错。
- Fix: 入口处提前做`npu_expand_tensors`+`make_info`，用`is_expandable_to`
  校验shape兼容性，不匹配时TORCH_CHECK报"shape mismatch"。
  重构index_put_aicore将indices展开上移到调用方。
- Reviewability: high -- 缺少输入校验是典型的防御编程遗漏
- Review rule: 接受多个tensor输入的算子须校验shape兼容性

### D-598: NPU_CHECK_SUPPORTED_OR_ERROR warning从不输出到每次都输出

- Hashes: 138029b88 [+1前序: e37db238e]
- Root cause: 告警输出策略经历多次迭代
- Files: NPUException.h
- Defect: `ACL_ERROR_RT_FEATURE_NOT_SUPPORT`的处理经历三阶段：
  (1) 完全静默(不输出)→(2) printf每次输出(e37db238e)→(3) 每次输出导致刷屏。
- Fix: 用static lambda+立即调用实现warn_once语义，只首次触发时打印。
  同时修正文案typo("supportted"→"supported")。
- Reviewability: high -- 首次输出后用static标志抑制是标准模式
- Review rule: "feature not supported" fallback路径统一使用warn_once

### D-599: torch.std空维度元素数为0时除零

- Hashes: 26349294c
- Root cause: 边界条件遗漏
- Files: StdKernelNpu.cpp
- Defect: `shape_prod==0`(reduce维度无元素)时直接计算标准差，除以(N-1)=(-1)或0。
  原条件只检查`shape_prod==1 && shape_prod<=correction`，遗漏`shape_prod==0`。
- Fix: 增加`shape_prod == 0`判断，此时返回NAN(符合PyTorch语义)。
- Reviewability: high -- 除法前检查分母<=0是基本防护
- Review rule: reduction统计算子须处理空维度(元素数为0)的边界

### D-600: codegen返回类型在ARM架构上不兼容

- Hashes: df37928cd
- Root cause: cpp.returns_type()在ARM上生成不兼容的类型表达
- Files: codegen/custom_functions.py
- Defect: 多输出自定义算子的返回类型由`cpp.returns_type()`自动生成，
  但在ARM架构上该函数可能生成不兼容的类型表达(如特定的ABI表示)。
- Fix: 返回类型改为显式构造`::std::tuple<at::Tensor, at::Tensor, ...>`。
  schema中返回类型也显式构造为`(Tensor, Tensor, ...)`格式。
- Reviewability: low -- 跨平台ABI兼容性问题需要在目标平台上验证
- Review rule: codegen生成的类型表达需在所有目标平台(x86/ARM)验证编译

### D-601: 路径拼接使用字符串+而非os.path.join

- Hashes: 27a68e192
- Root cause: 路径操作方式错误
- Files: codegen/gen_backend_stubs.py
- Defect: `output_dir+"../../../utils/"`字符串直接拼接不处理路径分隔符，
  当output_dir不以`/`结尾时产生错误路径如`foo../../../utils/`。
- Fix: 改用`os.path.join(output_dir, "../../utils/")`。
- Reviewability: high -- 路径操作必须使用os.path.join
- Review rule: 禁止用字符串+拼接文件路径，统一使用os.path.join或pathlib

### D-602: baddbmm转置tensor被OpCommand.Input()内部contiguous破坏

- Hashes: bd787b08c
- Root cause: OpCommand内部contiguous操作破坏了转置标记
- Files: BaddbmmKernelNpu.cpp
- Defect: 转置的tensor1/tensor2直接传给`cmd.Input()`，OpCommand内部做contiguous时
  破坏了转置标记，`adj_x1`/`adj_x2`信息与实际tensor状态不一致。
- Fix: 先手动根据转置状态做format_contiguous，再用`InputWithoutContiguous`传入
  (跳过OpCommand内部的contiguous)，保留转置信息的一致性。
- Reviewability: medium -- 需理解OpCommand.Input()的内部contiguous行为
- Review rule: 转置tensor传给OpCommand时应使用InputWithoutContiguous并手动处理format

### D-603: bitwise_and缺少dtype promotion

- Hashes: f339e1d19
- Root cause: 二元算子结果dtype计算缺失
- Files: BitwiseAndKernelNpuOpApi.cpp
- Defect: `bitwise_and(Tensor, Tensor)`直接用ref_tensor的dtype作为结果类型，
  未做`result_type`类型提升。如int32 & int64应返回int64。
  `bitwise_and(Tensor, Scalar)`对Bool tensor & 非Bool scalar也未提升。
- Fix: (1) Tensor版本用`at::native::result_type(self, other)`。
  (2) Scalar版本Bool+非Bool时强制kLong。
- Reviewability: high -- 二元算子必须做dtype promotion是标准规范
- Review rule: 所有二元算子实现须调用result_type做类型提升

### D-604: RecordFunction ObserverContext在异步执行时悬空

- Hashes: d05bf20b4
- Root cause: 异步场景下指针生命周期不可控
- Files: npu_profiler.cpp, data_reporter.h
- Defect: RecordFunction callback通过`ObserverContext`(unique_ptr)传递`OpRangeData`指针，
  异步执行时context可能提前失效，指针悬空。
- Fix: 移除ObserverContext包装，改为通过线程局部状态(state_ptr)直接存取OpRangeData。
  前半段(record start)返回nullptr，后半段(record end)通过getOpEvent()取回数据。
- Reviewability: medium -- 需理解RecordFunction异步callback的生命周期模型
- Review rule: profiler callback中不依赖context指针跨异步边界传递数据

### D-605: C++层tensor factory未触发NPU lazy device init

- Hashes: aaa827173
- Root cause: C++层绕过Python层的lazy init机制
- Files: TensorFactories.cpp, DeviceUtils.h(新), setup.py (+5)
- Defect: `empty`/`empty_with_format`等C++层tensor工厂函数直接分配NPU内存，
  不经过Python层的lazy init。NPU设备未初始化时分配内存导致崩溃或错误。
- Fix: C++层工厂函数入口新增`maybe_initialize_npu(device_)`调用。
  将设备初始化工具函数从Python绑定层下沉到C++核心层(`DeviceUtils.h`)，
  去掉pybind11依赖。
- Reviewability: medium -- 需要理解NPU lazy init的多层架构
- Review rule: 所有可能在Python init前触发的C++设备操作须包含maybe_initialize检查

### D-606: gelu_backward未做broadcast和dtype promotion

- Hashes: 9ed8a9223
- Root cause: backward实现假设grad和self shape/dtype一致
- Files: GeluBackwardKernelNpuOpApi.cpp
- Defect: `gelu_backward`输出tensor直接用`ApplyTensor(self)`创建，
  假设grad和self的shape/dtype完全一致。当两者不一致时产生错误结果。
- Fix: 用`broadcast_ops_npu_output_size(grad, self)`计算broadcast后size，
  用`at::native::result_type(grad, self)`做dtype promotion，再创建输出tensor。
- Reviewability: high -- backward算子与forward同样需要broadcast+promote
- Review rule: backward实现须对所有tensor输入做broadcast size计算和dtype promotion

### D-607: nllloss2d output_size维度与实际计算不匹配

- Hashes: 90cbe38de
- Root cause: output_size计算基于原始tensor但计算基于reshape后tensor
- Files: NLLLoss2dKernelNpu.cpp
- Defect: `Reduction::None`时output_size计算为`{N,H,W}`(基于原始4D self)，
  但实际计算中self已被permute+reshape为`{-1,C}`的2D输入，底层NLLLoss的None输出是1D。
  创建的result tensor shape与实际写入的数据量不匹配。
- Fix: output_size改为使用reshape后输入的维度(1D)，计算完成后`result.resize_({N,H,W})`
  恢复正确的3D形状。
- Reviewability: medium -- 需跟踪tensor经过permute+reshape后的shape变化
- Review rule: output_size计算须基于实际传给底层算子的tensor shape，非原始输入

### D-608: codegen vector transform目标迭代器用rbegin导致元素反序

- Hashes: 11ad58e79
- Root cause: 迭代器方向错误
- Files: codegen/dest/utils.py(实为C++ codegen模板)
- Defect: `std::transform`的目标迭代器使用`.rbegin()`(反向)而非`.begin()`(正向)，
  TensorList在CPU/NPU设备间转换时元素顺序颠倒。
- Fix: `.rbegin()`改为`.begin()`。
- Reviewability: high -- rbegin vs begin是明显的方向错误
- Review rule: std::transform目标迭代器默认使用.begin()，rbegin需显式注释原因

### D-609: torch.var unbiased模式N<=1时除(N-1)<=0

- Hashes: 1237fbb8a
- Root cause: 无偏估计分母边界未处理
- Files: VarKernelNpu.cpp
- Defect: `unbiased=True`时方差用`/(N-1)`，当N<=1时分母<=0，
  结果应为NAN但原代码直接计算产生inf或负值。
- Fix: 新增`get_shape_prod`计算指定dim元素数，N<=1时方差填充NAN。
- Reviewability: high -- unbiased var的N<=1边界是教科书级别的检查
- Review rule: 统计算子(std/var)须处理unbiased模式下样本数<=1的边界

### D-610: 缺__init__.py导致inspect遍历package时报错

- Hashes: 4a443b89a
- Root cause: Python package结构不完整
- Files: torch_npu/distributed/algorithms/__init__.py(新建空文件)
- Defect: `torch_npu/distributed/algorithms/`目录缺少`__init__.py`，
  不是合法Python package。mmcv使用inspect内省torch_npu.distributed时
  尝试import该目录失败，抛出built-in module errors。
- Fix: 创建空的`__init__.py`使其成为合法package。
- Reviewability: high -- 缺__init__.py是常见的package结构遗漏
- Review rule: 新建的Python目录必须包含__init__.py(即使为空)

### D-611: dot运算输出维度错误(1D应为0D标量)

- Hashes: 9e6845d90
- Root cause: output_size函数返回错误维度
- Files: DotKernelNpu.cpp
- Defect: dot product结果应为0D标量tensor(shape `{}`)，但原实现用
  `dot_npu_output_size`返回1D大小，创建1D tensor后再`resize_({})`降维。
  中间态的1D tensor可能引发下游shape推断错误。
- Fix: 直接用`{}`作为output_size创建0D tensor，移除多余的resize。
- Reviewability: high -- dot product返回标量是数学定义
- Review rule: 对照数学语义确认算子输出维度，标量结果须为0D tensor

### D-612: checkpoint不支持non-tensor参数+autocast+non-reentrant模式

- Hashes: b6a82e05c
- Root cause: checkpoint实现与PyTorch演进不同步
- Files: torch_npu/utils/checkpoint.py
- Defect: (1) `save_for_backward(*args)`在包含non-tensor参数时失败。
  (2) autocast状态只保存单一had_autocast_in_fwd，未区分NPU/CPU autocast。
  (3) 不支持non-reentrant checkpoint模式(`torch.autograd.grad`不兼容)。
  (4) 不支持NPU overflow flag检测。
- Fix: tensor/non-tensor分别保存(tensor用save_for_backward，non-tensor存ctx)。
  autocast分别保存npu/cpu kwargs。新增`_checkpoint_without_reentrant`。
  新增overflow flag处理。
- Reviewability: medium -- checkpoint是复杂模块，需全面对照upstream实现
- Review rule: torch_npu工具类须定期与upstream PyTorch对应实现做feature对齐审计

### D-613: torch.Generator monkey-patch破坏CPU Generator

- Hashes: 037a1eab9
- Root cause: monkey-patch粒度过粗(全局替换而非条件拦截)
- Files: codegen/templates/torch_funcs.py, torch_npu/__init__.py
- Defect: 原实现直接将`torch.Generator`替换为`torch_npu._C.Generator`，
  但CPU场景下仍需使用原生Generator。全局替换导致CPU Generator也被替换掉。
- Fix: 实现`_generator`包装函数，检查参数中是否包含'npu'设备：
  是则拦截并报AssertionError提示使用torch_npu._C.Generator；否则正常创建。
  从all_monkey_patches中移除直接替换。
- Reviewability: high -- monkey-patch影响范围必须限定在目标设备
- Review rule: 全局类替换式monkey-patch须区分设备类型，非目标设备保持原行为

### D-614: _rebuild_npu_tensor在FakeTensorMode下缺少fake_device属性

- Hashes: 461490b38 [+2 cherry-picks: 7d7551adf, f20606512]
- Root cause: fake mode路径缺失
- Files: torch_npu/utils/storage.py, test/_inductor/test_rng_prims.py
- Defect: `_rebuild_npu_tensor()`创建tensor后未检查是否处于FakeTensorMode。
  FakeTensorMode下tensor需要`fake_device`属性以正确追踪设备信息，否则inductor
  编译阶段的设备路由逻辑失败。同时tensor创建使用`torch.tensor()`而非`torch.empty()`，
  在fake mode下可能触发不必要的数据初始化。
- Fix: 添加FakeTensorMode检测，fake mode下设置tensor.fake_device。tensor创建改用
  torch.empty()。
- Reviewability: medium -- 需要理解FakeTensorMode的设备追踪机制
- Review rule: tensor序列化/反序列化路径须考虑FakeTensorMode，补充fake_device属性

### D-615: torchbench基准测试编译模式判断反转 + 模型补丁遗漏

- Hashes: fea7396e4
- Root cause: 条件逻辑反转 + benchmark配置遗漏
- Files: torchbench/common.py, torchbench/npu_support.py,
  torchbench/torchbench_models_list.txt
- Defect: (1) common.py中编译模式判断使用`NOT in`逻辑实际应为`in`，导致编译模式选择
  与预期相反。(2) 3个网络模型未添加到models list。(3) npu_support.py缺少
  speech_transformer模型patch和GENERATE_LIST清理函数。
- Fix: 修正条件逻辑，补充模型列表，添加--disable-aclgraph开关和模型patch。
- Reviewability: high -- `NOT in`→`in`条件反转是经典bug，review可直接发现
- Review rule: 布尔条件表达式变更时逐case验证正反逻辑

### D-616: copy H2D/D2H中empty_like输出tensor默认非连续

- Hashes: 0fa6abf33 [+6 cherry-picks: cc5a0166b, 272747028, 76a5ee379,
  41b22ff93, 1a7fbb67e, 5806547ae]
- Root cause: empty_like默认memory format不一致
- Files: torch_npu/csrc/aten/ops/CopyKernel.cpp,
  torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp,
  test/npu/test_copy.py (+138行测试)
- Defect: H2D/D2H拷贝路径中`empty_like()`未指定memory format，默认可能产生
  非LEGACY_CONTIGUOUS格式的tensor，与downstream计算对内存布局的假设不符。
- Fix: `empty_like()`调用显式传入`at::MemoryFormat::Contiguous`
  (LEGACY_CONTIGUOUS_MEMORY_FORMAT)。新增138行覆盖H2D/D2H、切片、广播、排列等场景。
- Reviewability: high -- empty_like的默认format行为是已知陷阱
- Review rule: H2D/D2H copy路径的中间tensor必须显式指定memory format

### D-617: native_api文档未说明use_compatible_impl一致性开关

- Hashes: fb54a6184 [+1 cherry-pick: 093276fe8]
- Root cause: 文档与实现不同步
- Files: docs/en/pytorch_native_api/ (14个版本文档)
- Defect: distributed API文档(gather/scatter/all_to_all)和nn.functional.max_pool2d
  未标注`use_compatible_impl(True)`可启用社区一致性对齐行为。用户无法知道如何
  切换到与upstream PyTorch一致的行为。
- Fix: 多版本文档的对应API行添加use_compatible_impl说明。
- Reviewability: high -- 功能发布时应同步更新文档
- Review rule: 新增行为开关(compatibility flag)时同步更新所有版本的API文档

### D-618: NPU空tensor的UntypedStorage创建失败(size=0)

- Hashes: fd3f902f7 [+8 cherry-picks: d53782e30, 8c4d58ad5, d2bee375a,
  eedc1b312, 6cc4abb13, 03d326b9e, 71f287c39, 0a2a808d1]
- Root cause: allocator对size=0边界条件处理错误
- Files: torch_npu/csrc/core/npu/NPUStorageImpl.cpp
- Defect: `NPUStorageImpl::allocate()`在data_ptr为null时直接return，不调用allocator。
  空tensor(size=0)合法地需要一个StorageImpl（data_ptr可为null但allocator状态需
  正确初始化），此bug导致`torch.UntypedStorage(0, device='npu')`失败。
- Fix: 无论data_ptr是否为null都调用allocator->allocate()，仅在size>0时检查返回值。
- Reviewability: high -- size=0是标准边界条件，review应覆盖
- Review rule: allocator/storage路径须覆盖size=0边界测试

### D-619: inductor autotune缺少精度校验gate

- Hashes: 0e5459381 [+1 cherry-pick: 0b72dfd4f]
- Root cause: autotune流程缺少精度守护
- Files: torch_npu/_inductor/config.py,
  torch_npu/_inductor/codegen/mlir_compiler.py
- Defect: mlir融合算子autotune只按benchmark性能选config，不验证精度。某些高性能
  config可能产生数值偏差但仍被选为最优，导致计算结果不正确。
- Fix: 新增`ANIR_ACC_CHECK_DURING_TUNE`环境变量控制。新增`accuracy_pass()`方法
  在tune阶段校验精度，无有效kernel时fallback到默认config。
- Reviewability: medium -- 精度问题需要运行时数值比对才能发现
- Review rule: autotune/autoselect流程须包含精度校验gate，不能只看性能

### D-620: shmem初始化flag语义误用(DEFAULT→UNIQUEID)

- Hashes: 4bdc6f560 [+4 cherry-picks: 9d370bd2c, cfe71904f, 604573915, c7ef25183]
- Root cause: 初始化flag语义误用
- Files: torch_npu/csrc/core/npu/NPUSHMEMExtension.cpp
- Defect: 共享内存初始化使用`ACLSHMEMX_INIT_WITH_DEFAULT`，shmem>=v1.0.1中DEFAULT
  模式使用固定key，多进程同时初始化时key冲突。应使用UNIQUEID模式保证隔离。
- Fix: `ACLSHMEMX_INIT_WITH_DEFAULT`改为`ACLSHMEMX_INIT_WITH_UNIQUEID`。1行改动。
- Reviewability: medium -- 需要理解shmem版本间flag语义变化
- Review rule: 共享内存/IPC初始化使用唯一标识(UNIQUEID)，禁用DEFAULT模式

### D-621: NPUGraph文档缺少aclnn算子入图限制说明

- Hashes: 98eec0a0b
- Root cause: 文档信息缺失
- Files: docs/*/pytorch_npugraph_desc.md
- Defect: NPUGraph使用文档未说明"仅支持aclnn算子入图"的限制。用户尝试将非aclnn算子
  入图时遇到运行时错误，无文档指引排查。
- Fix: 文档增加"仅支持aclnn算子入图"说明。
- Reviewability: high -- 已知限制须文档化
- Review rule: 功能限制/约束条件须在用户文档中显式说明

### D-622: inductor wrapper静态kernel代码生成缩进缺失

- Hashes: aa73dac55 [+1 cherry-pick: c8e320fe2]
- Root cause: codegen缩进状态管理遗漏
- Files: torch_npu/_inductor/codegen/wrapper.py
- Defect: `write_prefix`方法未调用`self.prefix.do_indent()`，静态编译模式下生成的
  kernel代码缺少缩进，Python因IndentationError无法执行。
- Fix: write_prefix中添加`self.prefix.do_indent()`调用。
- Reviewability: high -- codegen输出缩进问题review可直接发现
- Review rule: codegen路径中每个write方法须显式管理缩进状态

### D-623: npu_format_cast参数不支持keyword调用

- Hashes: 0f47888fd [+1 cherry-pick: f6c956d14]
- Root cause: 函数签名不支持kwargs
- Files: torch_npu/onnx/wrapper_onnx_ops.py,
  test/onnx/test_wrapper_onnx_ops.py, test/torch_npu_schema.json
- Defect: `npu_format_cast`的acl_format和customize_dtype为位置参数，ONNX wrapper层
  通过关键字方式调用时TypeError。
- Fix: 签名改为`(self, acl_format, *, customize_dtype=None)`，keyword-only参数。
  启用5个之前skip的测试用例。
- Reviewability: high -- 函数签名与调用方式不匹配是基础API设计问题
- Review rule: 对外暴露的算子接口须支持keyword调用

### D-624: inductor lowering对A5 ETA/HLLM缺少node_name参数

- Hashes: 831ded11e
- Root cause: lowering函数签名与fx graph节点属性不匹配
- Files: torch_npu/_inductor/lowering.py, torch_npu/_inductor/lowering_fx.py
- Defect: `lowering_index_select`等函数缺少`node_name`默认参数。A5平台ETA/HLLM
  场景的fx graph节点携带node_name属性，调用时TypeError。index/gather等操作的
  dump_fx_graph支持也不完整。
- Fix: lowering函数添加node_name默认参数。lowering_fx新增inductor_gather、
  inductor_index_impl等函数并patch lowering对象。
- Reviewability: medium -- 需要A5硬件运行特定模型才能触发
- Review rule: lowering函数签名须覆盖fx graph节点的所有可能属性

### D-625: FakeTensorMode下rng_state设备路由缺失

- Hashes: daaf69d9f [+7 cherry-picks: 2a57ff9e3, b511fef76, 35aac025d,
  e9d4bc1f5, 9b37b6f55, 1cb32e90a, 3b45846a5]
- Root cause: FakeTensorMode设备注册缺失
- Files: torch_npu/utils/_inductor.py,
  test/_inductor/test_run_with_rng_state.py,
  test/_inductor/test_dropout_with_checkpoint_recompute.py (new)
- Defect: `run_and_save_rng_state`和`run_with_rng_state`未注册FakeTensorMode impl。
  dynamo编译阶段在FakeTensorMode下调用时无法处理NPU设备的rng状态，导致
  checkpoint+dropout重计算场景编译失败。
- Fix: 为两个函数注册FakeTensorMode impl。新增dropout+checkpoint重计算测试。
- Reviewability: medium -- FakeTensorMode注册是inductor适配的系统性需求
- Review rule: 涉及设备状态的util函数须注册FakeTensorMode实现

### D-626: inductor tiling中sympy.Expr类型axis未处理

- Hashes: 828e670fc [+3 cherry-picks: 658d42025, 148d84e79, bfe98c932]
- Root cause: 动态shape下类型未统一
- Files: torch_npu/_inductor/codegen/triton.py,
  torch_npu/_inductor/fx_passes/ascend_custom_passes/__init__.py,
  torch_npu/_inductor/lowering_op_list.py,
  test/_inductor/test_resize_as.py
- Defect: tiling逻辑中axis参数假设为int，动态shape场景下可能是sympy.Expr对象，
  用作dict key或比较时TypeError/KeyError。custom passes缺少stable_topological_sort
  导致节点遍历顺序不确定。
- Fix: 处理sympy.Expr类型axis，添加stable_topological_sort，补充GENERATE_LIST。
- Reviewability: medium -- 动态shape引入sympy类型是已知适配点，具体受影响路径需运行时发现
- Review rule: inductor数值参数路径须统一处理int和sympy.Expr两种类型

### D-627: scope_begin/end用custom_op注册导致schema冲突

- Hashes: 66121d8ee [+5 cherry-picks: d018d5d8d, d0584da54, 643967066,
  1abdc019c, f0a99cf2f]
- Root cause: op注册方式选择不当
- Files: torch_npu/npu/graphs.py, torch_npu/npu/__init__.py,
  test/torch_npu_schema.json
- Defect: `super_kernel_scope_begin/end`使用`torch.custom_op`注册，custom_op的
  schema自动推断可能与手动定义的schema冲突，导致dispatch失败。
- Fix: 改用`torch.library.Library`显式注册schema，精确控制op签名。
- Reviewability: high -- op注册方式(custom_op vs Library)的选择是架构级决策
- Review rule: 需精确控制schema的op用torch.library.Library显式注册

### D-628: ProcessGroupHCCL析构函数吞掉shutdown异常

- Hashes: 4fbd0b5db [+7 cherry-picks: d1524d5ac, 13d8427f6, 100992763,
  f556a8fe1, e623fd18a, 351d11824, 70f3f5bc0]
- Root cause: 析构函数异常处理策略不当
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: 析构函数调用`shutdown()`时，若失败(如HCCL资源清理超时)，异常被默认
  catch-and-swallow。进程静默退出无coredump，无法事后分析shutdown失败原因。
- Fix: shutdown()调用中添加catch+rethrow。析构函数中rethrow会触发std::terminate，
  生成coredump。这是对catastrophic failure的刻意选择:宁可crash也不静默丢失错误。
- Reviewability: high -- 析构函数异常策略是C++审查必检项
- Review rule: 析构函数中关键资源清理失败须有明确错误传播策略(log+abort或rethrow)

<!-- Batch 2 cherry-pick addendum: existing entries with additional hashes found -->
<!-- D-4 (+3): 8826eda65, 52825b302, fc8fcc8f3 -->
<!-- D-9 (+18): 9cea7a3dc, 70c8e58fd, 7be454ddb, 5a2f74df4, ec32c3775,
     1ac142ae7, bdd0b97fd, 595cc6a98, 51eaa5cba, addc1979c, faa83c691,
     42767ed8c, a5dbb5989, 64ebb6e0f, 76e13e58c, 1b0f3a334, 2d86ea375, 4b9557261 -->
<!-- D-17 (+5): 95ffd8d96, d7413000d, 7af67e4ea, b717be638, 34f57866d -->
<!-- cb3c1e95b: stash index artifact (index on fix/batch-isend-irecv-recordstream),
     no code diff, the actual fix is in 0e93c91c3 -->


### D-629: JIT编译选项值判断反转(disable vs enable)

- Hashes: 00178dbab
- Root cause: 逻辑谓词取反
- Files: torch_npu/csrc/npu/Module.cpp
- Defect: `THNPModule_is_jit_compile_false_wrap`判断dynamicCompileswitch是否为非JIT编译
  模式，原代码检查`== "disable"`，但该选项语义是"enable"表示动态编译开启(即非binary)。
  导致binary场景下函数返回值与实际编译模式不符。
- Fix: 将比较值从`"disable"`改为`"enable"`。
- Reviewability: high -- 字符串常量的语义可通过选项文档或调用上下文确认
- Review rule: 选项字符串比较须与选项定义文档对照确认正/反语义

### D-630: 测试代码重复语句(copy-paste残留)

- Hashes: 00300f658 [+1: 7087fa22c]
- Root cause: copy-paste重复
- Files: test/test_network_ops/test_topk.py
- Defect: `test_topk_shape_format`中`cpu_input1.dtype == torch.float16`的类型转换代码
  重复了两次(4行)，第二次是无意义的重复执行。
- Fix: 删除重复的2行。
- Reviewability: high -- 相邻重复代码块是典型review可检出项
- Review rule: 连续出现相同条件判断+操作块时要求作者确认是否为copy-paste残留

### D-631: 新增设备利用率查询API(标记为bugfix但实为feature)

- Hashes: 00342e1a3
- Root cause: 功能缺失(非bug)
- Files: torch_npu/csrc/npu/Module.cpp, torch_npu/csrc/core/npu/interface/AclInterface.cpp,
  torch_npu/npu/utils.py, third_party/acl/inc/acl/acl_rt.h, third_party/acl/libs/acl.cpp,
  test/test_npu/test_torch_npu.py
- Defect: 缺少`torch.npu.utilization()`API，用户无法查询NPU设备利用率。commit被分类
  为BUGFIX但diff显示是新增`aclrtGetDeviceUtilizationRate`绑定+Python接口+测试。
- Fix: 新增ACL头文件声明、动态加载桩函数、Python绑定和测试用例。
- Reviewability: N/A -- 这是feature而非bug，分类有误
- Review rule: commit message应准确反映变更性质(feat vs fix)

### D-632: Tensorpipe子模块依赖URL失效(外部镜像认证策略变更)

- Hashes: 00c78a87c [+4: 0f90c8b93, 615a3982c, 6a16cb054, 725b3505c]
- Root cause: 第三方依赖URL失效
- Files: third_party/Tensorpipe (submodule)
- Defect: Tensorpipe依赖的`https://gitee.com/mirrors/libuv.git`从公开仓转为需认证仓库，
  导致CI流水线clone失败。
- Fix: 更新Tensorpipe子模块commit，将libuv依赖切换到gitcode同步镜像。
- Reviewability: low -- 外部仓库认证策略变更无法在review中预见
- Review rule: 第三方子模块的依赖链应有镜像冗余或vendor化策略

### D-633: storage.cpu()对非NPU非CPU设备(如meta)处理缺失

- Hashes: 00fdd2235 [+7: 0bbab657b, 41610c183, 47c1a1e51, 5578cae86,
  b857cb6ec, c8dbbbd01, ceb2c1a47]
- Root cause: 设备类型分支不完整
- Files: torch_npu/utils/storage.py,
  test/unsupported_test_cases/.pytorch-disabled-tests.json
- Defect: `_cpu()`方法对`device.type != 'cpu'`一律走NPU专用路径
  (`_tensor_construct_from_storage`)，但meta设备等非NPU设备也满足此条件，调用NPU
  API导致crash。
- Fix: 将条件从`!= 'cpu'`改为`== 'npu'`明确匹配NPU，非NPU非CPU设备走通用
  `UntypedStorage.copy_()`路径。同时移除相关disable测试条目。
- Reviewability: high -- 否定条件`!= 'cpu'`隐含假设只有两种设备类型，review应质疑
- Review rule: 设备类型判断优先用肯定匹配(`== 'npu'`)而非否定排除(`!= 'cpu'`)

### D-634: ConvertType创建aclTensor后未Release导致host内存泄漏

- Hashes: 0115c476c [+4: 090d91f0e, 8f95fd396, 9c9e4640e, f73b9e0db]
- Root cause: ACL资源未释放
- Files: torch_npu/csrc/aten/common/FormatCastKernelNpu.cpp
- Defect: `MaybeUseAclnnNpuFormatCast`中调用`ConvertType(src)`创建ACL tensor
  对象后直接传入`GetFormat`，返回后未调用`Release()`释放。每次format cast操作
  泄漏一个aclTensor的host端内存。高频调用场景下会导致OOM。
- Fix: 将`ConvertType`返回值保存到局部变量`acl_src`，使用后调用`Release(acl_src)`。
  同时将`dstStorageShape`在`delete[]`后置`nullptr`防止悬垂指针。
- Reviewability: high -- ACL API的create/release配对是C++审查基本项
- Review rule: 每个ConvertType/aclCreateTensor调用必须有对应Release，采用RAII或
  defer模式自动释放

### D-635: 跳过不稳定的stream query测试

- Hashes: 012920eba
- Root cause: 测试时序依赖
- Files: test/test_npu/test_torch_npu.py
- Defect: `test_npu_stream_query`在某些环境下不稳定失败(stream query结果与异步
  执行时序相关)。
- Fix: 添加`@unittest.skip`跳过。
- Reviewability: medium -- 跳过测试是临时规避，应有后续跟进
- Review rule: skip装饰器应附带issue编号或TODO说明恢复条件

### D-636: 警告消息拼写错误(repalce → replace)

- Hashes: 014634026
- Root cause: 拼写错误
- Files: torch_npu/csrc/aten/common/ToKernelNpu.cpp
- Defect: double类型不支持的警告消息中"repalce"拼写错误，同时"Warning:"前缀与
  `TORCH_NPU_WARN_ONCE`宏自带的Warning前缀重复。
- Fix: 修正拼写为"replace"，移除多余的"Warning:"前缀。
- Reviewability: high -- 字符串review可检出
- Review rule: 用户可见的warning/error消息需逐字审查

### D-637: OpInfo未注册__rmod__实际支持的int32/int64 dtype

- Hashes: 014c2699e [+4: 283f3e2ee, 7e0ca3731, 91b0c4f24, cbdd5711c]
- Root cause: 测试基础设施与算子实际能力不同步
- Files: torch_npu/testing/__init__.py, torch_npu/testing/npu_opinfo_dtypes.py(新增)
- Defect: NPU侧`__rmod__`前向对int32/int64实际可用，但OpInfo的
  `supported_dtypes()`直接透传upstream默认列表，未包含这两种类型。
  `test_dtypes___rmod___npu`因此失败。
- Fix: 引入配置驱动的dtype扩展机制：新增`npu_opinfo_dtypes.py`按op维护额外dtype，
  `_supported_dtypes`读取配置并合并，用`lru_cache`缓存加载结果。
- Reviewability: medium -- 需了解NPU算子实际dtype支持范围(需运行验证)
- Review rule: 新增算子或扩展dtype支持时，须同步更新OpInfo注册

### D-638: inductor codegen错误地对BLOCK_SUB施加next_power_of_2

- Hashes: 0152af552 [+5: 57efa0e83, 9a2791aeb, 9a5595ac9, c929b08e1, eb01c9656]
- Root cause: 不当的数值规范化
- Files: torch_npu/_inductor/codegen/triton.py
- Defect: NPU Triton kernel codegen中，对BLOCK_SUB常量值施加了`next_power_of_2()`，
  但BLOCK_SUB表示实际的子块大小(来自`simplified_tree_numel`)，强制取2的幂次方
  会导致kernel计算范围超出实际数据边界。
- Fix: 移除`next_power_of_2(val)`调用，直接使用原始值。
- Reviewability: high -- codegen路径中的数值变换需有明确语义依据
- Review rule: codegen数值变换(round/pad/align)须注释说明为什么需要以及上下游约束

### D-639: graph mode下view操作使用错误的storage size计算

- Hashes: 015614af6
- Root cause: graph mode与eager mode的storage语义差异
- Files: torch_npu/csrc/aten/common/ResizeNpu.h,
  torch_npu/csrc/framework/contiguous/ReshapeOpt.cpp
- Defect: `checkInBoundsForStorage`在graph mode下用`prod_intlist(size) * itemsize()`
  计算storage大小，但graph mode中tensor的storage capacity与shape乘积不一致(graph
  编译器可能分配更大的buffer)。导致view操作的bounds check误报越界。
- Fix: 改用`GraphUtils::GetTensorCapacity()`获取graph mode下的实际storage容量。
  同时在ReshapeOpt中增加graph mode专用检查：若base_sizes与sizes乘积不同(即slice/
  select场景)则不走reshape优化路径。
- Reviewability: medium -- 需要理解graph mode的storage语义与eager mode的差异
- Review rule: graph mode路径中的tensor元信息(size/stride/storage)须使用graph-aware
  API而非直接从tensor读取

### D-640: ONNX shape计算使用Python浮点除法导致dtype错误

- Hashes: 0167ac0c2
- Root cause: Python 3整除语义
- Files: torch_npu/onnx/wrapper_ops_combined.py,
  test/test_onnx/test_wrapper_onnx_ops.py
- Defect: `NPUFusedAttentionLayernormQkvFwdOP`中计算`new_shape`时用`/`(真除法)而非
  `//`(整除)，`x.shape[0]/seq_len`返回float，作为shape元素传给torch会导致
  类型错误或精度问题。
- Fix: 用`int(x.shape[0]/seq_len)`显式转换。同时删除了关联的失效测试用例
  (npu_fused_attention_layernorm_qkv_fwd)。
- Reviewability: high -- shape计算中的`/` vs `//`是Python常见陷阱
- Review rule: tensor shape计算必须用`//`或`int()`确保整数类型

### D-641: Revert inductor AsyncCompile.warm_pool()预热(引发副作用)

- Hashes: 018517f93 [+4: af1298cc9, d9d99885e, f76927b83, f9f9f5c3c]
- Root cause: 过早初始化引发副作用
- Files: torch_npu/__init__.py, torch_npu/utils/_dynamo.py
- Defect: 在`_InductorNpuRegistry.register_inductor_npu`中调用
  `AsyncCompile.warm_pool()`并设置`TORCH_WARM_POOL=0`环境变量。warm_pool在注册阶段
  过早触发编译进程池初始化，可能导致import时序问题和环境变量污染。
- Fix: 移除`AsyncCompile.warm_pool()`调用和`TORCH_WARM_POOL`环境变量设置。
- Reviewability: medium -- 需了解AsyncCompile的初始化时序和进程池副作用
- Review rule: import/register阶段不应触发重量级初始化(进程池/编译器/CUDA context)

### D-642: torch.all未适配全dtype(测试重构+dtype扩展)

- Hashes: 01b954d95 [+2: 0f776024f, 18682df05]
- Root cause: 算子dtype支持不完整
- Files: test/test_network_ops/test_all.py
- Defect: `torch.all`的NPU测试仅覆盖bool输入，未测试其他dtype。测试代码本身
  也有冗余(cpu转npu后再转回的模式不统一)。
- Fix: 重构测试：引入`create_common_tensor`统一测试数据创建，添加dim参数和out
  参数测试路径，覆盖更多dtype组合。
- Reviewability: high -- 测试覆盖范围在review中可以检查
- Review rule: 算子测试须覆盖文档声明支持的全部dtype，不能只测default类型

### D-643: 跨PyTorch版本API名称变更导致import失败

- Hashes: 01c4b5c52 [+5: 093337cbc, 3144f8a29, 4e79b30ec, 9b2a385ba, f9e6c3c6c]
- Root cause: upstream API名称变更(breaking change)
- Files: test/test_nn.py
- Defect: PyTorch升级后多个API名称变更：`skipIfMps`→`skipIfMPS`，
  `new_module_tests`→`get_new_module_tests()`(变为函数调用)，
  `tf32_is_not_fp32()`→`torch.cuda.is_tf32_supported()`。test_nn.py使用旧名称
  导致import时NameError。
- Fix: 逐一替换为新API名称，`new_module_tests`改为`get_new_module_tests()`调用。
- Reviewability: medium -- 依赖upstream internal API的适配需在升级时系统检查
- Review rule: 升级PyTorch版本时须扫描所有from torch.testing._internal import，
  对照upstream changelog确认API兼容性

### D-644: as_strided的CompileType导致不必要的重编译

- Hashes: 01dc029f6
- Root cause: CompileType选择不当
- Files: torch_npu/csrc/aten/ops/AsStridedKernelNpu.cpp
- Defect: `AsStrided`算子的shape/stride/offset参数使用
  `MEMORY_HOST_COMPILE_DEPENDENT`(offset)或默认CompileType(shape/stride)，
  表示这些值变化时需要重编译算子。但as_strided的参数在动态shape场景频繁变化，
  每次变化都触发重编译严重影响性能。
- Fix: 三个参数统一改为`MEMORY_HOST_COMPILE_INDEPENDENT`，声明参数变化不需要
  重编译(算子内部已能处理不同的shape/stride组合)。
- Reviewability: medium -- 需理解CompileType语义及其性能影响
- Review rule: 动态变化的tensor元信息参数(shape/stride/offset)默认应使用
  COMPILE_INDEPENDENT，COMPILE_DEPENDENT需有明确理由

### D-645: torch.Generator不支持传入NPU device参数

- Hashes: 0241d0580 [+2: 46a574a43, d318a95e0]
- Root cause: 设备参数解析缺失
- Files: torch_npu/csrc/npu/Generator.cpp, test/test_npu/test_generator.py
- Defect: `torch_npu._C.Generator(device="npu:0")`调用时，Python层接收到的device
  参数未经NPU设备解析，直接传给PyTorch原生Generator构造函数，后者不识别"npu"设备。
  同时`get_device`属性返回时调用了不存在的`TNPDevice_New`(实际为`THPDevice_New`)。
- Fix: 在`THPGenerator_pynew`中对位置参数和关键字参数中的device做
  `parse_npu_device`转换后再传入。修正`get_device`中的函数名。
- Reviewability: high -- 新设备类型的API入口参数解析是必查项
- Review rule: NPU相关Python C扩展的所有接受device参数的入口须经过parse_npu_device

### D-646: cdist算子p=inf时传入infinity浮点值不被硬件支持

- Hashes: 027f6f626
- Root cause: 硬件API对特殊浮点值的约定不同
- Files: pytorch1.5.0/src/aten/src/ATen/native/npu/CdistKernelNpu.cpp
- Defect: cdist算子p参数为无穷大时，代码将其转为`std::numeric_limits<float>::infinity()`
  传给NPU算子，但NPU算子不接受IEEE 754 infinity值作为参数(可能导致未定义行为或报错)。
- Fix: 将infinity映射为`-1`作为NPU侧的约定哨兵值(表示Chebyshev距离/L_inf范数)。
- Reviewability: medium -- 需了解NPU算子对特殊值的约定
- Review rule: NPU算子参数中的特殊数值(inf/nan/sentinel)须参照算子API文档约定

### D-647: 测试修复(un-skip + backward调用模式修正)

- Hashes: 02ba0f00f
- Root cause: 测试质量问题(多项)
- Files: test/contrib/test_activations.py, test/optim/test_fused_optimizers.py,
  test/profiler/test_npu_profiler.py
- Defect: 多个测试问题：(1) mish激活函数测试中`output.sum().backward()`改为
  `res.backward(torch.ones_like(res))`以匹配梯度计算语义；(2) 之前skip的
  test_mish和test_default_profiler测试现在可以通过，移除skip。
- Fix: 修正backward调用模式，移除不再需要的skip装饰器。
- Reviewability: high -- test skip状态应定期审查
- Review rule: 每个@unittest.skip须有跟踪issue，定期review是否可以un-skip

### D-648: RPC BackendType枚举引用在动态注册后过期

- Hashes: 02ffd5090 [+4: 1c12209fe, 21080d172, 926239533, c258f9278]
- Root cause: Python模块级import别名导致引用过期
- Files: torch_npu/distributed/rpc/backend_registry.py
- Defect: `torch.distributed.rpc`在import时通过`from .backend_registry import BackendType`
  持有BackendType枚举的引用。torch_npu注册`NPU_TENSORPIPE` backend时，
  `backend_registry.register_backend`会重建并替换`backend_registry.BackendType`
  (因为Python的enum是不可变的，新增成员需要重建整个enum)。但rpc模块持有的是旧引用，
  导致`_validate_rpc_args`的isinstance检查失败。
- Fix: 注册完成后显式更新`torch.distributed.rpc.BackendType = rpc.backend_registry.BackendType`。
- Reviewability: medium -- 需理解Python enum重建机制和模块级引用语义
- Review rule: 动态enum注册后须检查所有已import该enum的模块并同步更新引用

### D-649: 文档拼写错误(Ture→True)及profiler路径描述不准确

- Hashes: 193772bbe [+5: 6babefead, 4c2d860ae, abe5f8f9b, 4b771736c, 6e3d10447]
- Root cause: 文档拼写错误/描述不准确
- Files: docs/zh/PyTorch网络模型移植&训练指南/PyTorch网络模型移植&训练指南.md
- Defect: E2E prof使用说明中`use_e2e_profiler=Ture`(应为True)，
  profiler结果路径描述为`./result`但实际自动生成的文件夹路径是
  `./results/PROF_***`。两处错误导致用户按文档操作时参数无效或找不到结果。
- Fix: 修正拼写`True`，更新路径描述为`./results/PROF_***`并补充说明
  "路径须已存在"。
- Reviewability: high -- 文档中的代码示例应作为可执行代码审查
- Review rule: 文档中的代码片段须通过自动化lint(至少检查Python语法合法性)

### D-650: 回退unique_consecutive的output sync index变更

- Hashes: 62441e8bd
- Root cause: 算子输出同步索引设置错误(回退修复)
- Files: torch_npu/csrc/aten/ops/UniqueConsecutiveKernelNpu.cpp,
  test/test_network_ops/test_unique_consecutive.py
- Defect: 此前PR !2599将`UniqueConsecutive`的output_sync_idx从`{0, 2}`改为
  `{0, 1, 2}`(即同步output[1] inverse_indices)，同时新增了
  `test_unique_consecutive_return_inverse_and_counts`测试。但该变更引入了问题
  (可能是output[1]动态shape不稳定)，需要回退。
- Fix: 回退output_sync_idx为`{0, 2}`(仅同步output和counts，不同步inverse)，
  删除新增测试，恢复原有测试的参数签名。
- Reviewability: medium -- 需理解OpCommand.Sync语义及动态输出的同步机制
- Review rule: 修改output_sync_idx时须验证所有输出tensor的shape确定性，
  动态shape的输出不应加入sync列表

### D-651: BinaryCrossEntropyWithLogitsBackward忽略weight和pos_weight参数

- Hashes: 127244b33
- Root cause: 函数参数被忽略(直接用ones替代)
- Files: torch_npu/csrc/aten/ops/BinaryCrossEntropyWithLogitsBackwardKernelNpu.cpp
- Defect: backward实现中`weight`和`pos_weight`参数虽然在函数签名中接收，
  但实现体内无条件使用`at::ones(self.sizes(), self.options())`创建全1张量替代，
  完全忽略了用户传入的权重。当用户设置了class weight或positive weight时，
  梯度计算结果错误，导致训练不收敛或精度下降。
- Fix: 添加`weight.defined()`/`pos_weight.defined()`检查，有值时使用
  `NpuUtils::format_contiguous(weight/pos_weight)`，无值时才fallback到全1。
- Reviewability: high -- 函数参数未使用是静态分析可检测的缺陷
- Review rule: 算子实现中的所有函数参数须有对应使用路径，
  unused parameter应触发编译器警告(-Wunused-parameter)

### D-652: 图模式测试缺少graph_mode装饰器及残留debug print

- Hashes: ca2567ca6
- Root cause: 测试装饰器遗漏 + debug代码残留
- Files: test/test_network_ops/test_batch_norm.py,
  test/test_network_ops/test_batchnorm_backward.py,
  test/test_network_ops/test_conv2d.py
- Defect: BatchNorm和Conv2d测试需要在图模式下运行以验证nbytes()问题的修复，
  但测试方法缺少`@graph_mode`装饰器，导致测试始终在eager mode下执行，
  无法覆盖图模式路径。同时test_batch_norm中残留了`print(item)`调试语句。
- Fix: 为相关测试方法添加`@graph_mode`装饰器，删除debug print语句。
- Reviewability: high -- `print`语句和缺失的测试装饰器均为code review常规检查项
- Review rule: 测试用例中不应出现print语句；图模式相关fix的测试须有@graph_mode装饰

### D-653: AclSetCompileopt条件分支结构导致codecheck告警

- Hashes: 9b80a73ac [+3: cda222fb4, ff215c6a3, 6da5d64fa]
- Root cause: 代码风格(嵌套if-else结构不符合编码规范)
- Files: torch_npu/csrc/framework/interface/AclOpCompileInterface.cpp,
  torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp
- Defect: `AclSetCompileopt`函数中`if (!ge_init_disable)`包裹主逻辑，else返回
  `ACL_ERROR_NONE`的结构不符合编码规范(可能是"深层嵌套"或"反转条件"类告警)。
- Fix: 反转条件为early return: `if (ge_init_disable) return ACL_ERROR_NONE;`，
  主逻辑提升到函数体顶层，减少嵌套层级。
- Reviewability: high -- 标准codecheck/linter可自动检测
- Review rule: 优先使用early return减少嵌套，主逻辑不应在else分支中

<!-- D-647 addendum: cherry-picks 05bbb2e64, b385a7780, 9986b244d, 418a20e72, e10fe016d, d7e437d25 (同一测试修复在master/v2.2.0/v2.1.0/v1.11.0等分支的cherry-pick) -->

### D-654: Public API命名不符合下划线约定(asd模块+contrib函数)

- Hashes: bb5a9056b [+2: a3eec6a53, 44b23d706]
- Root cause: 公有API命名规范违反(缺少下划线前缀)
- Files: torch_npu/__init__.py, torch_npu/asd/asd.py,
  torch_npu/contrib/function/bbox_coder.py, test/allowlist_for_publicAPI.json
- Defect: 多个内部实现函数和类使用了不带下划线前缀的名称
  (`asd_patch`, `Singleton`, `SilentFaultDetector`, `box_dtype_check`等)，
  导致test_public_bindings测试中被识别为公有API。同时allowlist_for_publicAPI.json
  中缺少新增的distributed API(`batch_isend_irecv`, `gather`, `gather_object`)。
  另有模块路径拼写错误: `torch.distributer.checkpoint.optimizer`(应为distribut-ed-)。
- Fix: 将内部函数/类统一加`_`前缀: `_asd_patch`, `_Singleton`,
  `_SilentFaultDetector`, `_box_dtype_check`等。补充allowlist条目，添加`__all__ = []`。
- Reviewability: high -- 命名规范可由lint规则强制
- Review rule: 内部实现函数/类必须以`_`开头；`__all__`须在每个模块中显式定义

### D-655: copy_ H2D/D2H快速路径缺少size相等检查导致broadcast失败

- Hashes: 0be977eba
- Root cause: 快速路径条件不完整(漏检size)
- Files: torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp
- Defect: `copy_h2d_baseformat_opapi`和`copy_d2h_baseformat_opapi`中，
  当src和dst同类型且都contiguous时走快速复制路径(`copy_h2d_baseformat_dtype_contigous`)。
  但该快速路径不支持broadcast(直接memcpy)，而条件判断中只检查了dtype和contiguous，
  未检查size是否相等。当`dst.size() != src.size()`时(需要broadcast的场景)，
  快速路径直接执行导致数据错误或crash。
- Fix: 增加`same_size = (src.sizes() == dst.sizes())`条件，
  size不等时走通用路径(支持broadcast)。
- Reviewability: high -- 快速路径的前提条件是否完备是review重点
- Review rule: 优化快速路径的准入条件须覆盖所有语义约束(dtype/size/stride/format)，
  每个约束须有对应测试case验证边界

### D-656: Public API合规性批量修复(allowlist/test迁移/命名规范)

- Hashes: f95216d05 [+4: f3e32d85b, e96b53a92, cc987e691, af6612899]
- Root cause: Public API测试基础设施缺陷(多项)
- Files: test/allowlist_for_publicAPI.json, test/npu/test_public_bindings.py,
  torch_npu/__init__.py, torch_npu/contrib/function/*.py,
  torch_npu/contrib/module/*.py
- Defect: (1) `test_public_bindings.py`位于test/根目录，但allowlist.json的
  相对路径在test/npu/下读取时路径错误；(2) allowlist中`torch.distributer`拼写错误；
  (3) contrib模块缺少`__all__`定义和`_`前缀；(4) `torch_npu.__init__`中多个
  import使用了现已改名为私有的符号(需同步更新)；(5) 大量torch_npu.dynamo.torchair
  子模块的内部符号被公有化暴露，需加入temp_filter豁免。
- Fix: 迁移test到test/npu/并修正路径查找逻辑；修正拼写；
  添加`__all__`和`_`前缀；同步init.py import；建立temp_filter集合暂时豁免
  无法立即修复的内部符号泄露。
- Reviewability: medium -- 多处散落的命名/路径变更，需系统性检查
- Review rule: 新增公有符号必须经过API review；`test_public_bindings`须在CI中运行

### D-657: Named tensor测试修复及Full算子named tensor支持

- Hashes: 39976d38b [+1: 8b2d2bc5c]
- Root cause: NPU设备上named tensor API适配不完整(多项)
- Files: test/test_namedtensor.py, torch_npu/csrc/aten/ops/FullKernelNpu.cpp,
  torch_npu/csrc/aten/ops/op_api/CloneKernelOpApi.cpp,
  torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp
- Defect: (1) `assertTensorDataAndNamesEqual`比较时NPU tensor未先.cpu()，
  导致device mismatch断言失败；(2) `index_fill_`/`bernoulli_`中传入CPU tensor
  作为value但self在NPU上，需要`.npu()`；(3) `repr`测试期望中缺少`device='npu:0'`；
  (4) `is_shared()`在NPU device上不支持，需从smoke test中移除；
  (5) `FullKernelNpu.cpp`缺少`#include <ATen/NamedTensorUtils.h>`且输出tensor
  未传播names。
- Fix: 逐项修复: 添加.cpu()/.npu()转换、更新repr期望、移除不支持的API调用、
  为Full算子添加named tensor支持头文件和name传播逻辑。
- Reviewability: medium -- named tensor是较新特性，NPU适配需要系统性排查
- Review rule: NPU算子适配upstream API时须检查named tensor兼容性；
  测试中的tensor比较须确保device一致

<!-- D-657 addendum: cherry-picks 50a5c57f1, ceff50bcf (同一named tensor修复在v2.3.0/master分支的cherry-pick) -->

### D-658: non_blocking传输测试因tensor过小导致stream.query()时序不稳定

- Hashes: 1b6d31470 [+3: 6a772e8dc, 7ca4bf758, 09450fa23]
- Root cause: 测试设计缺陷(时序假设不成立)
- Files: test/npu/test_npu.py, test/npu/test_torch_deterministicalgorithms.py,
  test/allowlist_for_publicAPI.json
- Defect: (1) `test_to_non_blocking`中创建的tensor仅1M元素(`torch.randn(1000000)`)，
  non_blocking copy完成太快，`stream.query()`时传输已结束返回True，
  但测试期望`non_blocking=True`时stream仍活跃(返回False)。
  (2) `test_npu_set_deterministic_false`断言"100次sum结果中不应全部相等"来验证
  非确定性行为，但NPU上sum算子可能是确定性的，导致100/100相等，断言失败。
- Fix: (1) 将tensor扩大到`randn(10000, 10000, 2)`(2亿元素)并添加额外计算
  (`torch.sum`和二次copy)增加stream占用时间，确保query时stream仍活跃。
  (2) 删除不可靠的非确定性测试。
- Reviewability: medium -- 需理解异步stream query的时序语义
- Review rule: 异步行为的测试不应依赖精确时序，应使用事件同步或barrier机制；
  测试非确定性行为须有明确的不确定性来源保证

### D-659: Profiler公有API命名规范修复及import路径调整

- Hashes: b2ba1b293 [+3: 5edbe2b84, 8b16b6335, 9c8983a94]
- Root cause: Profiler模块内部类/枚举暴露为公有API
- Files: torch_npu/profiler/__init__.py,
  torch_npu/profiler/analysis/prof_bean/node_info_bean.py,
  torch_npu/profiler/analysis/prof_view/prof_db_parse/fwk_api_db_parser.py,
  test/allowlist_for_publicAPI.json
- Defect: (1) profiler分析模块中`NodeInfoBean`、`FwkApiTableRow`、`TorchOpDataOri`、
  `TaskQueueDataOri`、`PythonTraceApiDataOri`、`CannNodeLaunchApiOri`等内部类
  缺少`_`前缀；(2) `supported_activities`从profiler.py导出但实际定义在
  profiler_interface.py中，import路径不正确；(3) 删除了已合并到`_TorchOpDataOri`
  中的冗余`FwkApiTableRow`枚举定义。
- Fix: 统一加`_`前缀，修正import路径从`profiler_interface`导入`supported_activities`，
  删除冗余枚举。
- Reviewability: high -- 命名规范可自动化检查
- Review rule: profiler analysis内部类统一加`_`前缀；`__init__.py`的import须指向
  定义所在模块

### D-660: DTensor测试适配upstream API变更(PairwiseParallel/redistribute_profiler)

- Hashes: 2c6cc3547 [+2: 2eef82967, 26e2b1fb2]
- Root cause: upstream PyTorch DTensor API breaking change
- Files: test/distributed/_tensor/test_dtensor.py,
  test/distributed/_tensor/test_view_ops.py
- Defect: PyTorch upstream删除了`PairwiseParallel`(替换为显式的
  `ColwiseParallel`+`RowwiseParallel` plan dict)，
  `redistribute_profiler`替换为`CommDebugMode`，
  `tree_flatten`替换为`pytree.arg_tree_leaves`，
  `device_mesh.size(dim=0)`改为`size(mesh_dim=0)`，
  `redistribute().to_local()`简化为`full_tensor()`。
  torch_npu测试仍使用旧API导致import失败和测试报错。
- Fix: 逐一适配新API：用parallelize_plan dict替换PairwiseParallel，
  用CommDebugMode替换redistribute_profiler，更新参数名和方法调用。
- Reviewability: medium -- 需关注upstream changelog
- Review rule: 升级PyTorch版本时须自动diff所有`from torch.distributed`的import，
  对照upstream deprecation notice逐一检查

### D-661: Public API第三轮修复(error_code模块重命名+temp_filter瘦身)

- Hashes: c81a670c0 [+1: 4ee8ead91]
- Root cause: error_code模块命名不符合私有约定 + temp_filter冗余条目
- Files: torch_npu/__init__.py, torch_npu/asd/asd.py,
  torch_npu/contrib/function/*.py, test/npu/test_public_bindings.py,
  test/unsupported_test_cases/.pytorch-disabled-tests.json
- Defect: (1) `torch_npu.utils.error_code`模块暴露了`ErrCode`和`pta_error`为公有API，
  但这些是内部错误处理工具，不应对用户公开。所有import路径需从`error_code`改为
  `_error_code`。(2) `__all__ = ["ErrCode"]`导致ErrCode被视为公有导出。
  (3) temp_filter中包含`torch_npu.pta_error`但该符号已不再泄露，需从filter中删除。
- Fix: 将`error_code`模块重命名为`_error_code`，更新所有import路径(10+文件)，
  `__all__`改为空列表，清理temp_filter冗余条目。新增3条ONNX disabled test。
- Reviewability: high -- 模块重命名影响面大但逻辑简单
- Review rule: 工具模块(`error_code`/`path_manager`等)应默认使用`_`前缀

<!-- D-661 addendum: cherry-pick 655bb61db (!12300 v2.1.0, 同一public API修复的分支适配) -->

### D-662: test_correct_module_names因torchair子模块allowlist缺失而失败

- Hashes: 4c4915023 [+3: 292ac6c53, 85458ad49, 79624c55c]
- Root cause: 第三方子模块(torchair)的公有符号未纳入allowlist
- Files: test/npu/test_public_bindings.py
- Defect: `test_correct_module_names`检查所有`torch_npu`子模块的公有符号是否在
  allowlist中。`torch_npu.dynamo.torchair`是一个第三方子模块，其公有符号由
  torchair自身维护(在`third_party/torchair/torchair/tests/st/allowlist_for_publicAPI.json`)，
  但test_public_bindings未加载该文件，导致torchair的所有符号报为"未经授权的公有API"。
- Fix: 在测试中尝试加载torchair自身的allowlist并merge到主allowlist中。
  若子模块不存在(clone repo未递归更新)则warn跳过。
- Reviewability: medium -- 子模块的API管理需要跨仓库协调
- Review rule: 第三方子模块的公有API须由子模块自身管理allowlist，
  host repo的测试须能自动发现并加载子模块的allowlist

### D-663: Public API持续合规修复(ONNX/deterministic/collect_env模块)

- Hashes: ddd7cada7 [+2: 2010cdbe1, 2c867c4e4]
- Root cause: 多个模块的公有API合规性问题(持续修复)
- Files: test/allowlist_for_publicAPI.json, test/npu/test_public_bindings.py,
  torch_npu/onnx/*.py, torch_npu/npu/deterministic.py,
  torch_npu/utils/collect_env.py, test/onnx/test_*.py
- Defect: (1) ONNX wrapper模块函数未加`_`前缀(582行重命名)；
  (2) deterministic模块内部函数暴露；(3) collect_env工具函数暴露；
  (4) allowlist新增150+行条目(持续增长说明前期API治理不充分)；
  (5) ONNX测试中使用了已废弃的内部API。
- Fix: 系统性重命名为私有函数，扩展allowlist，适配ONNX测试。
- Reviewability: high -- 属于系统性技术债清理
- Review rule: 新模块创建时须同步定义`__all__`并提交allowlist条目

### D-664: MSTX Range ID与Stream的关联状态管理从全局函数迁移到MstxMgr类

- Hashes: 9cf37d171 [+5: b2c5805bb, fa72dba50, 56ffb3286, 5012165c3, d6d94a8ce]
- Root cause: 全局状态跨模块共享导致职责分散和数据竞争风险
- Files: torch_npu/csrc/framework/interface/MstxInterface.cpp,
  torch_npu/csrc/framework/interface/MstxInterface.h,
  torch_npu/csrc/profiler/mstx_mgr.cpp, torch_npu/csrc/profiler/mstx_mgr.h
- Defect: `MstxInterface`中维护了`g_rangeIdsWithStream`全局set，用于记录哪些
  range ID是带stream的(需要在device上结束)。但`MstxMgr`才是管理range生命周期的
  类，全局set导致：(1) 跨模块数据依赖(`MstxMgr`需调用`IsRangeIdWithStream`查询
  另一个模块的状态)；(2) `g_rangeIdsWithStream`的mutex与`g_rangeIdMap`的mutex
  不同，可能存在TOCTOU竞争。
- Fix: 将`ptRangeIdsWithStream_`成员移入`MstxMgr`类，删除全局
  `g_rangeIdsWithStream`和`IsRangeIdWithStream`函数。insert操作在
  `RangeStart`中`MstxMgr`自己的mutex保护下完成，消除跨模块访问。
- Reviewability: medium -- 需理解range的host/device双模语义
- Review rule: profiler相关的状态应集中在管理类中，避免全局变量跨模块共享

### D-665: foreach优化器monkey-patch方式导致与upstream不兼容

- Hashes: 4b58baf3e [+3: a22e7b609, 495ffb8bb, 6e069f72f]
- Root cause: monkey-patch粒度过粗(替换整个optimizer类)
- Files: torch_npu/utils/_optim.py
- Defect: 原实现通过`monkey_patch_optimizer`装饰器包装所有11个optimizer类
  (`SGD/Adam/AdamW/...`)及其对应函数(`sgd/adam/adamw/...`)，共22个monkey-patch。
  该方式在foreach参数被upstream改为default=None(根据设备自动选择)后失效：
  (1) 新增的optimizer不会被patch；(2) upstream内部对optimizer类的isinstance检查
  因类被替换而失败；(3) 代码维护负担大(22行硬编码import)。
- Fix: 改为patch单一入口`torch.optim.optimizer._get_foreach_kernels_supported_devices`，
  根据设备名称返回支持foreach的设备列表。Ascend910B以上(不含910P)支持foreach
  时返回包含privateuse1的列表，否则只返回cuda/xpu。
- Reviewability: medium -- 需理解upstream的foreach设备分发机制
- Review rule: 对upstream框架的monkey-patch应尽量选择最小侵入点，
  优先patch配置/策略函数而非替换整个类

### D-666: 分布式fault mode测试修复(超时配置+错误码断言+tensor大小)

- Hashes: 84c0df049 [+1: 9271f94b5]
- Root cause: 测试环境依赖和硬件行为假设不成立(多项)
- Files: test/distributed/_fault_mode_cases/error_hccl_timeout.py,
  test/distributed/_fault_mode_cases/error_use_same_addr.py,
  test/distributed/test_fault_mode.py, test/npu/test_npu.py
- Defect: (1) HCCL超时测试缺少`HCCL_EXEC_TIMEOUT`环境变量设置，
  在某些环境下使用默认超时(过长或过短)导致测试不稳定；
  (2) same_addr测试用scalar tensor(`torch.tensor(2)`)进行all_reduce，
  数据量太小可能在错误检测前就完成，改为100x100x20增加传输时间；
  (3) fault mode错误码断言按设备型号区分EI0002/EI9999，但实际行为已统一为EI0002，
  旧的分支判断导致新设备上测试失败；
  (4) non_blocking测试的tensor大小`randn(10000,10000,2)`(1.6GB)在某些设备上OOM，
  改为`randn(1000,1000,2,100)`(~800MB)。
- Fix: 添加超时环境变量、增大tensor、统一错误码断言、调整tensor shape。
- Reviewability: high -- 测试中的硬编码数值和设备条件分支是常见脆弱点
- Review rule: 分布式测试须配置确定性的超时参数；
  错误码断言不应按设备型号分支(除非有明确的行为差异文档)

### D-667: SyncBatchNorm backward未对sum做mean归一化导致精度溢出

- Hashes: 0301996b0 [+2: 6edc2cf97, b6111aaf0]
- Root cause: 多卡归约后的sum值未除以count就传入backward
- Files: torch_npu/utils/syncbatchnorm.py
- Defect: SyncBatchNorm的backward在allreduce得到`sum_dy`和`sum_dy_xmu`后，
  直接将sum传给`torch.batch_norm_backward_elemt`，但该API期望的是mean值。
  多卡场景下sum会随卡数线性增长，导致梯度值放大，精度溢出。
- Fix: 增加`divisor = count_tensor.sum()`，用`sum_dy / divisor`和
  `sum_dy_xmu / divisor`得到mean后再传入。
- Reviewability: high -- API参数语义(sum vs mean)在调用处应当显式注释
- Review rule: 调用数学API时，确认参数语义(sum/mean/variance)与传入值一致；
  多卡归约后的sum须除以world_size或count才能作为mean使用

### D-668: PReLU backward测试weight参数和执行路径不一致

- Hashes: 0337402f8 [+1: 206f2cc9e]
- Root cause: 测试代码cpu/npu路径用不同方式构造weight导致不可比
- Files: test/test_network_ops/test_prelu_backward.py
- Defect: 原测试的`cpu_op_back_exec_ext`和`npu_op_back_exec_ext`接受外部传入的weight参数，
  但两个函数内部PReLU的`num_parameters`由外部shape推断，当`weight = torch.randn(12)`
  与输入shape不匹配时行为未定义。CPU和NPU路径对weight初始化方式不同(一个用外部传入，
  另一个实际上用PReLU默认)，对比结果不可靠。
- Fix: 重写为统一的静态方法，从input shape推断`num_parameters`，
  内部固定`weight = torch.ones([num_parameters]) * 0.25`，确保cpu/npu路径完全对称。
- Reviewability: high -- 测试方法签名不一致是明显的review信号
- Review rule: 对比测试的cpu/npu路径必须使用完全相同的参数构造和初始化逻辑

### D-669: div算子混合dtype输入未做类型提升

- Hashes: 034943195
- Root cause: NPU div算子缺少dtype promotion逻辑
- Files: torch_npu/csrc/aten/ops/DivKernelNpu.cpp, test/test_network_ops/test_div.py
- Defect: `torch.tensor(1) / 10`在CPU上返回float(PyTorch的整数除法自动提升为float)，
  但NPU的div_out实现直接用`self.scalar_type()`作为输出dtype，整数输入返回整数结果(截断为0)。
  同时div_inplace路径调用`div_out_npu_nocheck`而非`NPUNativeFunctions::div_out`，
  绕过了类型检查。
- Fix: (1) 用`at::native::result_type(self, other)`推断目标dtype，
  整数类型强制提升为Float；(2) 操作数不一致时做`npu_dtype_cast`；
  (3) inplace路径改为调用`NPUNativeFunctions::div_out`走完整检查。
- Reviewability: high -- 缺少类型提升是op适配时的系统性问题
- Review rule: 算术算子适配须验证mixed-dtype场景(int/float, fp16/fp32)，
  与CPU行为做等价性对比

### D-670: DDPJoinHook.__init__中类名解析失败

- Hashes: 035270e64 [+1: 41ab95456]
- Root cause: monkey-patch替换函数时未更新内部引用
- Files: torch_npu/utils/module.py
- Defect: `DDPJoinHook__init__`中用`DistributedDataParallel`做isinstance检查，
  但该名字未从torch.nn.parallel导入(只在原始上下文中可用)，
  monkey-patch后运行时NameError。`super().__init__()`在Python 2风格的类继承中
  也不可靠(需要显式传入类名)。
- Fix: 改为`torch.nn.parallel.DistributedDataParallel`全限定名，
  `super(_DDPJoinHook, self).__init__()`显式指定类。
- Reviewability: high -- monkey-patch函数中的名字解析是已知风险模式
- Review rule: monkey-patch替换的函数体内，所有非局部名字须用全限定路径或在函数内import

### D-671: TORCH_WARN在高频fallback路径中造成日志洪泛

- Hashes: 0355a61cc
- Root cause: fallback路径用TORCH_WARN而非TORCH_WARN_ONCE
- Files: torch_npu/csrc/core/npu/interface/AclInterface.cpp
- Defect: `AclrtSynchronizeStreamWithTimeout`、`AclrtDestroyStreamForce`、
  `AclrtMallocAlign32`在找不到对应ACL函数时fallback到旧API，
  每次调用都输出TORCH_WARN。这些函数在正常训练中可能被调用百万次，
  导致日志文件膨胀到GB级别，影响IO性能。
- Fix: 改为`TORCH_WARN_ONCE`，只在首次调用时警告。
- Reviewability: high -- fallback路径的TORCH_WARN应默认为WARN_ONCE
- Review rule: fallback/降级路径中的warning必须用*_ONCE变体，
  除非每次调用的上下文信息不同

### D-672: test_torch.py大规模适配(dtype/import/设备条件)

- Hashes: 0358d8125
- Root cause: 上游PyTorch test_torch.py更新后torch_npu未同步适配
- Files: test/test_torch.py, .pytorch-disabled-tests.json
- Defect: 上游新增了`TwoTensor`、`bf32_on_and_off`、`optim_db`等import，
  增加了`uint16/uint32/uint64`等dtype测试，torch_npu未同步导致ImportError和测试失败。
  Ascend 910A不支持complex类型，需要条件排除。
- Fix: 补齐import，添加`device_is_910A`条件分支排除不支持的dtype，
  更新disabled-tests列表。
- Reviewability: medium -- 上游rebase导致的适配工作难以在review中提前发现
- Review rule: rebase上游代码后，须检查所有测试文件的import变化和新增dtype覆盖

### D-673: NPU event wait超时值硬编码导致HCCL长任务误报超时

- Hashes: 03615052e [+2: b081ddfc8, c4f4b16c2]
- Root cause: op wait timeout与HCCL exec timeout不联动
- Files: torch_npu/csrc/core/npu/NPUStream.cpp,
  torch_npu/csrc/core/npu/register/OptionsManager.cpp,
  torch_npu/csrc/core/npu/register/OptionsManager.h
- Defect: `kOpWaitTimeout`硬编码为1800秒，但HCCL集合通信操作可配置更长的超时
  (通过`HCCL_EXEC_TIMEOUT`环境变量)。当HCCL超时>1800s时，stream event wait
  会先于HCCL本身超时，导致误报"event wait timeout"而非正确的HCCL超时错误。
- Fix: 从环境变量读取HCCL_EXEC_TIMEOUT，设`kOpWaitTimeout = hccl_timeout + 30s`，
  并用溢出检查(`if (kOpWaitTimeout < hccl_exec_timeout) kOpWaitTimeout = UINT_MAX`)
  防止uint32_t回绕。
- Reviewability: medium -- 超时值之间的依赖关系需要领域知识
- Review rule: 多层超时机制须有明确的层级关系(inner < outer)，
  且应当从同一配置源派生而非各自硬编码

### D-674: aclop初始化和JIT编译选项在aclnn路径上执行导致freeze

- Hashes: 03cb322df [+18 cherry-picks: 1f20cdfe1, 1fcf77c3b, 249262281,
  269f28504, 2820b8b04, 39cfa9383, 4ffb08d69, 5b11889c8, 5b19a95ad,
  70c28da7b, 72136f3a7, 77ae68115, 79639fdc4, 9fd311960, a2c6fd8df,
  a389c71f5, cc069beb4, f73fc7390]
- Root cause: 执行路径guard缺失，aclop-only逻辑在所有路径运行
- Files: torch_npu/csrc/framework/OpCommand.cpp,
  torch_npu/csrc/framework/OpParamMaker.cpp
- Defect: `OpCommand::Run()`中`LazyInitAclops()`和`AclSetCompileopt(ACL_OP_JIT_COMPILE)`
  在aclop和aclnn两个路径都会执行。当用户开启deterministic模式时，
  `SetDeterministicOption`也会对aclnn路径调用`AclSetCompileopt(ACL_OP_DETERMINISTIC)`。
  这些ACL编译选项API在aclnn路径下语义冲突或未初始化，导致进程hang。
- Fix: (1) 将`LazyInitAclops()`和JIT编译选项设置移入`CheckCustomHandlerNull()`
  为true的分支(即仅aclop路径)；(2) `SetDeterministicOption`增加`g_used_aclop`
  守卫条件，仅在实际使用过aclop时才设置deterministic编译选项。
- Reviewability: high -- 条件执行路径被移出guard是结构性问题
- Review rule: ACL编译选项设置必须在对应执行引擎的guard内；
  影响全局编译状态的API调用须有明确的路径守卫

### D-675: div算子第二轮修复(RealDiv算子名+设备tensor判断)

- Hashes: 03d2c5ee2
- Root cause: 第一轮div dtype修复(D-669)不完整
- Files: torch_npu/csrc/aten/ops/DivKernelNpu.cpp, test/test_network_ops/test_div.py
- Defect: D-669修复后仍有两个问题: (1) NPU的"Div"算子对某些dtype组合行为不正确，
  应使用"RealDiv"算子(支持更广泛的类型提升)；(2) scalar判断`other.dim() == 0`
  会误判NPU上的0维tensor为host scalar，应加`!at_npu::key::isDeviceTensor(other)`
  排除NPU上的0维tensor。
- Fix: 算子名改为"RealDiv"，scalar分支增加设备tensor检查。
- Reviewability: medium -- 需要理解NPU算子名与PyTorch算子名的映射关系
- Review rule: NPU 0维tensor不等于host scalar；
  任何`dim() == 0`判断都须考虑tensor是否在device上

### D-676: inductor测试基础设施添加

- Hashes: 041062cf1
- Root cause: 缺少inductor后端的测试覆盖
- Files: test/_inductor/下70+个测试文件
- Defect: torch_npu的inductor后端缺少系统性的算子测试，bug修复过程中
  一并补齐了abs/add/div/embedding/softmax等算子的compile+inductor测试。
- Fix: 批量新增inductor测试用例，使用统一的TestUtils基类和parametrize装饰器。
- Reviewability: low -- 测试补齐属于事后完善
- Review rule: 新增后端适配须同步提交对应的测试用例

### D-677: BinaryCrossEntropyWithLogits输出tensor和weight的dtype错误

- Hashes: 04b70e70a [+4: 749c13879, a01c2da2e, b7b7c4b23, ee6de8fe3]
- Root cause: 用target的dtype/options创建结果和填充tensor
- Files: torch_npu/csrc/aten/ops/BinaryCrossEntropyWithLogitsKernelNpu.cpp,
  torch_npu/csrc/aten/ops/BinaryCrossEntropyWithLogitsBackwardKernelNpu.cpp,
  torch_npu/testing/common_methods_invocations.py
- Defect: forward中`result = ApplyTensorWithFormat(outputSize, target.options(), ...)`
  使用target的options，但PyTorch语义是以self(input)的dtype为准。
  当input=fp32, target=fp16时结果tensor为fp16导致精度错误。
  同样，默认weight/pos_weight用`at::ones(target.sizes(), target.options())`创建，
  dtype也不正确。backward也有同样问题。
- Fix: 全部改为`self.options()`/`self.sizes()`；
  weight/pos_weight与self dtype不一致时做`npu_dtype_cast`。
- Reviewability: high -- `target.options()` vs `self.options()` 是op适配的常见陷阱
- Review rule: loss函数的result/intermediate tensor应从input(self)继承dtype，
  不从target继承(除非API明确规定)

### D-678: npugraph_ex compile_fx中dict key用变量名而非字符串

- Hashes: 04e619ffd [+4: 2fbaa35d4, a8caee6bc, cef9936cf, df80288cb]
- Root cause: Python dict literal语法错误
- Files: torch_npu/npu/npugraph_ex/__init__.py
- Defect: `{options: options}`中`options`被解析为变量值(dict对象)作为key，
  而非字符串`"options"`作为key。当options=None时，key为None；
  当options为dict时，dict不可hash，直接TypeError。
  且None参数未处理，应传空dict而非None给下游。
- Fix: 改为`{"options": {} if options is None else options}`。
- Reviewability: high -- 这是基础Python语法错误，任何review都应捕获
- Review rule: dict literal中的key如果意图为字符串则必须加引号

### D-679: ProcessGroupHCCL强引用Storage导致tensor内存无法释放

- Hashes: 0555aef8b [+7: 36c76bd70, 4d2fb4e03, 86c69647c, 8e55eb62a,
  9bd43a258, f6b6bada5, fbee5944a]
- Root cause: recorded_inputs_持有Storage强引用延长tensor生命周期
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp,
  torch_npu/csrc/distributed/ProcessGroupHCCL.hpp
- Defect: `WorkHCCL::recorded_inputs_`使用`c10::Storage`(强引用)保存通信操作的
  输入tensor storage。在multi-stream memory reuse模式下，这些引用会一直持有到
  `synchronizeInternal`调用。如果Work对象被长期持有(如存放在future queue中)，
  输入tensor的内存即使上层已释放引用也无法被allocator回收，造成内存泄漏。
  注意`recorded_outputs_`已经使用`weak_intrusive_ptr`，inputs却是强引用，
  说明这是一个遗漏。
- Fix: `recorded_inputs_`改为`weak_intrusive_ptr<c10::StorageImpl>`，
  eraseStream前用`.lock()`检查storage是否仍然存活。
- Reviewability: high -- inputs用强引用而outputs用弱引用，不一致性是明显信号
- Review rule: 异步Work对象中保存的tensor/storage引用应默认使用weak_ptr，
  除非有明确的lifetime保证

### D-680: 日志API迁移(NPU_LOGI -> ASCEND_LOGI)

- Hashes: 055745ae0 [+1: 950bf7335]
- Root cause: 日志API演进，旧API废弃
- Files: torch_npu/csrc/framework/utils/ForceJitCompileList.cpp
- Defect: 使用已废弃的`NPU_LOGI`宏，新标准为`ASCEND_LOGI`。
- Fix: 替换日志宏调用。
- Reviewability: high -- 机械替换，lint工具即可发现
- Review rule: 日志API升级应全仓统一执行，不应逐文件零散修复

### D-681: profiler record_function功能缺失

- Hashes: 055cf8dc5
- Root cause: torch_npu profiler未实现record_function接口
- Files: torch_npu/csrc/profiler/init.cpp, torch_npu/csrc/profiler/profiler_legacy.cpp,
  torch_npu/csrc/profiler/profiler_legacy.h, torch_npu/npu/__init__.py,
  torch_npu/npu/npu_frontend_enhance.py, torch_npu/npu/profiler.py
- Defect: PyTorch的`torch.autograd.profiler.record_function`在NPU设备上无效，
  因为torch_npu的profiler后端未注册对应的start/end callback。
  用户使用`with record_function("my_op"):`包裹的区域在profiler trace中不可见。
- Fix: 在C++侧实现`pushRecordFunction`和`popRecordFunction`，
  通过pybind11暴露；Python侧在profiler初始化时注册callback。
- Reviewability: medium -- 属于功能缺失而非代码错误
- Review rule: 适配PyTorch profiler时须验证record_function/record_shapes等
  用户侧API在NPU上的完整性

### D-682: CI脚本Python版本硬编码和超时值不合理

- Hashes: 05a148acc
- Root cause: CI环境Python版本不匹配
- Files: ci/access_control_test.py
- Defect: CI脚本使用`python3`调用UT，但CI环境中`python3`可能指向不兼容的版本，
  且timeout=2000s过长(单个UT正常应在数十秒内完成)，超时失败需等待33分钟。
- Fix: 改为`python3.8`(CI环境的确定版本)，timeout从2000s降到500s。
- Reviewability: high -- 硬编码版本号本身就是code smell(应使用虚拟环境)
- Review rule: CI脚本应使用虚拟环境的python而非系统路径的硬编码版本；
  UT超时应与预期运行时间成比例(如10x)

### D-683: 非连续NPU tensor序列化时storage_offset/stride不一致

- Hashes: 05d248761 [+5: 4c8dc7cfa, 7078c4823, af5a572ab, dc1f1844e, e94b9a4ef]
- Root cause: 序列化时混用NPU tensor和CPU tensor的元数据
- Files: torch_npu/utils/storage.py, test/test_api/test_torch/test_serialization.py
- Defect: `_reduce_ex`在序列化NPU tensor时，先做`self.cpu()`拷贝到CPU，
  然后用原NPU tensor的`self.storage_offset()`和`self.stride()`。
  对于连续tensor两者一致，但非连续tensor(如slice)在CPU上可能有不同的
  storage layout，导致反序列化时用错误的offset/stride重建tensor，
  数据不正确或直接报错。
- Fix: 统一使用cpu副本的元数据: `tmp_tensor = self.cpu()`后
  全部从`tmp_tensor`取storage/offset/size/stride。
- Reviewability: high -- 序列化代码中混用不同tensor的元数据是典型错误
- Review rule: tensor序列化的storage/offset/size/stride必须来自同一个tensor对象

### D-684: ConvertTensorToScalar缺少Bool/BFloat16/Complex类型处理

- Hashes: 0640dd2be
- Root cause: 类型分支不完整
- Files: torch_npu/csrc/framework/utils/CalcuOpUtil.cpp
- Defect: `ConvertTensorToScalar`只处理了Float/Double/Int/Long/Half五种类型，
  Bool、BFloat16、ComplexFloat、ComplexDouble四种类型直接走到else分支报错
  `ACL_ERROR_UNSUPPORTED_DATA_TYPE`。任何使用这些类型的0维tensor参与的运算
  (如bool tensor作为condition)都会崩溃。
- Fix: 补齐Bool(int8_t)、BFloat16、ComplexFloat、ComplexDouble的分支。
- Reviewability: high -- 类型switch不完整是常见遗漏，应有静态检查
- Review rule: ScalarType的switch/if-else链必须覆盖所有PyTorch支持的类型，
  或显式列出不支持的类型并给出有意义的错误信息

### D-685: _unique2算子申请{0}大小tensor导致NPU算子失败

- Hashes: 06e197ed6
- Root cause: 用空tensor({0})作为unused输出的占位符
- Files: torch_npu/csrc/aten/ops/_Unique2KernelNpu.cpp
- Defect: 当`return_inverse=false`或`return_counts=false`时，
  对应输出tensor用`ApplyTensorWithFormat({0}, ...)`创建(即大小为0的tensor)。
  但NPU的UniqueWithCountsAndSorting算子不接受size=0的输出buffer，
  导致算子执行失败。同时`output_sync_idx`漏掉了index 1(yInverse)，
  该输出的实际大小未被同步。
- Fix: 占位tensor从`{0}`改为`{1}`(最小有效大小)；
  `output_sync_idx`从`{0, 2}`改为`{0, 1, 2}`。
- Reviewability: high -- {0} vs {1}的差异明显，且NPU算子约束应有文档
- Review rule: NPU算子的输出buffer不能为空(size=0)，
  即使该输出在语义上不使用也须分配最小大小

### D-686: tensor反序列化时设备信息丢失和storage先序问题

- Hashes: 07359a1a5
- Root cause: _rebuild_npu_tensor中storage处理顺序错误
- Files: torch_npu/utils/storage.py, test/test_npu_multinpu.py
- Defect: `_rebuild_npu_tensor`中storage参数的device信息(如npu:0/npu:1)
  未在cpu转换前保存，导致多卡场景下tensor总是恢复到默认设备而非原始设备。
  `.npu()`硬编码使用默认设备号，无法恢复到npu:1等非默认设备。
  `_reduce_ex`中做了不必要的`.cpu()`(上一个fix D-683引入的)，
  此fix修正为直接使用self(序列化框架会自行处理device transfer)。
- Fix: (1) 在cpu()之前保存`device = storage._untyped_storage.device`；
  (2) `.npu()`改为`.to(device)`保留原始设备号；
  (3) `_reduce_ex`中`tmp_tensor`直接用`self`而非`self.cpu()`。
- Reviewability: high -- 序列化相关的设备信息必须在所有转换之前保存
- Review rule: 序列化/反序列化涉及device transfer时，
  须在转换前保存原始device并在重建时显式指定

### D-687: tensor.clone丢失non-contiguous tensor的stride信息

- Hashes: 1ac142ae7, bdd0b97fd, 595cc6a98
- Root cause: clone()未保留非连续tensor的stride
- Files: torch_npu/csrc/aten/ops/op_api/CloneKernelOpApi.cpp
- Defect: `NPUNativeOpApiFunctions::clone`始终使用`apply_tensor_without_format(src)`
  分配输出tensor，该函数创建contiguous布局的tensor，丢弃了src的stride信息。
  当src是非连续tensor(如转置后)且MemoryFormat为Preserve时，
  clone结果的stride与原tensor不一致，违反PyTorch的clone语义约定。
- Fix: 在MemoryFormat::Preserve且src.is_non_overlapping_and_dense()时，
  用`at::empty_strided_symint(src.sym_sizes(), src.sym_strides(), ...)`分配输出，
  保留原始stride。使用sym版本以支持symbolic shape。
- Reviewability: high -- clone的语义是精确复制，stride保留是基本约定
- Review rule: 涉及tensor拷贝/克隆的NPU实现，须对照PyTorch native版本检查
  MemoryFormat::Preserve路径是否正确保留stride

### D-688: Revert tensor clone api consistency (D-687前序修复的回退)

- Hashes: 51eaa5cba, addc1979c, faa83c691, 42767ed8c, a5dbb5989
- Root cause: 前序clone修复引入了新问题
- Files: torch_npu/csrc/aten/ops/op_api/CloneKernelOpApi.cpp
- Defect: 前一次clone修复(使用`at::empty_strided`保留stride)存在问题
  (具体表现未在commit message中说明)，需要revert回简单的
  `apply_tensor_without_format` + `copy_`路径。
  时间线: 此revert(03-28) → D-687的修复(04-03)用sym版本重新实现。
- Fix: revert回`apply_tensor_without_format(src)` → `copy_`的简单路径。
  D-687后续用`empty_strided_symint`重新修复，解决了原修复的问题。
- Reviewability: medium -- 原修复的问题需要runtime验证才能发现
- Review rule: tensor布局修改需要覆盖sym shape场景的UT，
  避免只用concrete shape测试通过后在dynamic shape下失败

### D-689: is_core_control_enabled函数调用缺少括号

- Hashes: 273e153ed, 6885f86b2, 4e16f3f21, 70db2fe64, 7636929db, 6ce506f79
- Root cause: 函数指针被当作变量访问，永远为true
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `c10_npu::is_core_control_enabled`是一个函数，但代码中写成
  `if (c10_npu::is_core_control_enabled)`（缺少`()`），
  C++中对函数名取值得到非null函数指针，隐式转换为true。
  导致`UseStreamResInCurrentThread`在core control未启用时也被调用，
  影响所有collective操作(allreduce, broadcast, reduce, allgather等，共11处)。
- Fix: 所有11处`is_core_control_enabled`后加`()`变为函数调用。
- Reviewability: high -- 编译器应产生-Waddress警告，CI应开启此warning
- Review rule: 函数调用缺少()是C++经典陷阱，
  建议对bool函数启用-Waddress或-Wbool-conversion警告

### D-690: alltoall/scatter/gather compatible mode未检查SoC兼容性

- Hashes: e52be7356, d86573267, c578cc473, 3756b48a5, 564efc34e
- Root cause: compatible mode缺少硬件型号检查
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: alltoall/scatter/gather的compatible mode实现只检查了env flag
  (`CheckCompatibleImpl()`)，未检查当前SoC是否实际支持compatible模式。
  compatible mode仅在A2(Ascend910B系列)和A3(Ascend910_93系列)上可用，
  在不支持的芯片上执行会失败。
- Fix: 新增`IsCompatibleSoc()`函数，检查SoC版本是否在支持范围内
  (910B1~310B1之间 或 910_9391~910_95之间)。
  scatter/gather/alltoall的compatible mode判断从`use_compatible_impl`
  改为`use_compatible_impl && is_compatible_soc`。
- Reviewability: high -- 硬件兼容性检查是基本防御
- Review rule: 依赖特定硬件能力的代码路径必须有SoC版本检查

### D-691: scatter/gather compatible mode内存浪费(flatten冗余分配)

- Hashes: 2b550202a, 7c8966f44, 34762ed81, d23118fbe, 593281aef, ef23cce69
- Root cause: compatible mode下仍执行了不必要的flatten和tensor分配
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: gather中非root rank创建了`size_`个`empty_like`tensor(完全不需要)，
  然后执行`flatten_for_scatter_gather`将它们打平并copy数据到flattened tensor。
  在compatible mode下这些操作完全多余，因为HCCL通过send/recv实现gather/scatter，
  不需要预先flatten。这导致不必要的内存占用。
- Fix: compatible mode下跳过flatten，直接将原始input tensor作为placeholder
  传入collective接口；删除`inputFlattened[i][0].copy_(inputTensors[i])`的数据拷贝；
  scatter同理改为直接使用原始tensor。
- Reviewability: medium -- 需要理解compatible mode不需要flatten这一语义前提
- Review rule: collective操作的不同实现路径(default/compatible)
  应在入口处就分流，避免default路径的数据准备污染compatible路径

### D-692: alltoall compatible mode同样的内存优化

- Hashes: fca88bf40, 21944b85b, 2b2d226dc, 85fcfa984, 6811fdb0b, adf751562
- Root cause: alltoall的compatible mode也存在不必要的flatten和SoC检查遗漏
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: 与D-691同源。alltoall在compatible mode下仍然执行了完整的
  flatten/split_sizes/view_as_byte处理，为float8类型做了不必要的byte视图转换。
  同时缺少`is_compatible_soc`检查(与D-690同类)。
- Fix: 重构alltoall为两条分支: compatible mode直接传原始tensor，
  default mode保留原有flatten逻辑。增加`is_compatible_soc`检查。
- Reviewability: medium -- 与D-690/D-691是同一轮修复的一部分
- Review rule: 同一设计缺陷(缺少SoC check/冗余flatten)
  在scatter/gather/alltoall三处独立存在，修复时须全局排查

### D-693: 缺少torch_npu._C._npu_hasPrimaryContext API

- Hashes: 0a00c8d4e, cf61a0b80, edbea6b08, 520e842d7, 4a3351bae, b84b13155
- Root cause: PyTorch适配层缺少对齐CUDA的hasPrimaryContext接口
- Files: torch_npu/csrc/core/npu/NPUFunctions.cpp,
  torch_npu/csrc/core/npu/NPUFunctions.h,
  torch_npu/csrc/core/npu/interface/AclInterface.cpp,
  torch_npu/csrc/core/npu/interface/AclInterface.h
- Defect: `c10::cuda::hasPrimaryContext(device_index)`在CUDA后端可用，
  但torch_npu缺少对应的NPU实现。依赖此接口的用户代码(如DeepSpeed)
  在NPU上无法运行。
- Fix: 新增`hasPrimaryContext`函数，通过`AclrtGetPrimaryCtxState` API获取
  context激活状态。动态加载函数指针以兼容不同CANN版本。
  注释说明对内部使用推荐更高性能的`isDeviceCtxActive`(查本地缓存)。
- Reviewability: low -- 这是功能缺失而非代码错误
- Review rule: CUDA/NPU API对齐时须检查接口完整性，
  不仅对齐核心API也要对齐查询类API

### D-694: profiler step_trace_time计算出现负数

- Hashes: df8008c2f, 129908d34, 6af12da7c, d82547269, 488ad3ccf, 2ed77407e, 337272901
- Root cause: 时间差计算未做非负保护
- Files: torch_npu/profiler/analysis/prof_view/_trace_step_time_parser.py,
  torch_npu/profiler/analysis/prof_view/prof_db_parse/_trace_step_time_db_parser.py
- Defect: `comunNotOverlpRec = comunNotOverlp - bubble`。
  当通信与计算的重叠度测量不精确时(时间戳抖动、采样误差等)，
  bubble可能大于comunNotOverlp，导致负值。负值在后续时间分解中无意义，
  且可能导致下游可视化工具显示异常。两个parser中有相同逻辑。
- Fix: 加`max(..., 0)`钳位。同时修改了两处parser保持一致。
- Reviewability: high -- 时间差运算应默认考虑非负约束
- Review rule: profiler中所有时间差计算(A - B)须评估是否需要max(0)保护，
  特别是涉及不同来源时间戳的差值

### D-695: Inductor cat codegen中load带other=0.0时mask插入位置错误

- Hashes: 69baadba1
- Root cause: 字符串操作对load行尾部格式做了错误假设
- Files: torch_npu/_inductor/codegen/triton.py
- Defect: NPUIndexTritonKernel在处理cat操作时，需要给load语句加index mask。
  原代码用`line[:-1]`截掉最后一个字符')'再拼接mask，
  但当load行以`, other=0.0)`结尾时，`line[:-1]`只截掉了')'，
  mask被插入到`other=0.0`之后而非之前，生成无效的Triton代码。
- Fix: 判断行是否以`, other=0.0)`结尾，是则截掉整个后缀再拼接mask和other。
- Reviewability: high -- 字符串拼接生成代码时应考虑所有可能的行尾格式
- Review rule: codegen中基于字符串位置操作的代码，
  须穷举目标字符串的所有可能格式(特别是可选参数)

### D-696: jit api test使用了已移除的PyTorch内部API

- Hashes: 9c3b1233a, 36a2f423d, 679d27b3b, 58f2263f1, 1fbf80ae3
- Root cause: upstream PyTorch重构了测试工具函数
- Files: test/test_jit.py
- Defect: test_jit.py从`torch.testing._internal.jit_metaprogramming_utils`导入
  `additional_module_tests`和从`common_nn`导入`module_tests, new_module_tests`，
  这些名称在upstream PyTorch中已被替换为`get_all_nn_module_tests()`函数。
  导致test_jit.py在新版PyTorch上import失败。
- Fix: import改为`get_all_nn_module_tests`，
  循环改为`for test in get_all_nn_module_tests()`。
- Reviewability: high -- 依赖upstream内部API的测试需要跟踪API变更
- Review rule: 从torch.testing._internal导入的名称是不稳定API，
  每次rebase应检查这些import是否仍然存在

### D-697: batch_isend_irecv无条件register_work导致OOM

- Hashes: f200c7191, f5d63c8ba, f096fbafa, dccb6eeeb, 46fd1ca50, abc89cdc3
- Root cause: work对象注册逻辑缺少条件守卫
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `batch_isend_irecv`中`c10d::register_work(tensor, work)`无条件执行，
  即使`allow_inflight_collective_as_graph_input()`为false。
  register_work将work的intrusive_ptr存入全局registry，增加引用计数。
  但unregister_work只在synchronize/wait中调用，且同样缺少条件守卫。
  当graph input功能未启用时，register产生的引用永远不会被释放，
  work对象(含output tensor引用)累积导致OOM。
  同时`_reduce_scatter_base_uneven`和`_allgather_base_uneven`也有同样问题。
- Fix: register_work和unregister_work都用
  `if (c10d::allow_inflight_collective_as_graph_input())`守卫。
- Reviewability: high -- 全局registry的register/unregister必须配对且有一致的条件守卫
- Review rule: 任何往全局registry注册对象的代码，须确认注册和注销的条件完全对称

### D-698: apply_tensor_without_format缺少TORCH_NPU_API导出标记

- Hashes: a51a3a164, 0a34cbf75, 73978ae7e, 5defd4ef0, 5281d5de6, 7d80e64d2
- Root cause: 符号导出遗漏
- Files: torch_npu/csrc/framework/utils/OpPreparation.h
- Defect: `OpPreparation::apply_tensor_without_format`的3个overload
  缺少`TORCH_NPU_API`宏，导致外部plugin(如op_plugin)链接时
  找不到该符号(undefined reference)。
- Fix: 给3个overload声明前加`TORCH_NPU_API`。
- Reviewability: high -- 链接错误在构建时即可发现
- Review rule: 新增或修改public header中的函数声明时，
  检查是否需要TORCH_NPU_API/C10_NPU_API导出标记

### D-699: empty_like丢失非连续tensor的stride

- Hashes: c74e69da7, d4a9ccdd7, 6a527ab33, 9886a2ddb, 51b180ccf, 38812a880
- Root cause: empty_like在preserve format下未保留stride
- Files: torch_npu/csrc/aten/common/TensorFactories.cpp, test/npu/test_npu.py
- Defect: `empty_like_npu`在`Preserve`或无指定format时，
  始终用`ApplyTensorWithFormat`创建contiguous tensor。
  对于转置等非连续tensor(如`torch.empty([16,32]).T`，stride=(1,32))，
  `empty_like`结果的stride应为(1,32)而非(16,1)，但实际返回contiguous布局。
  与D-687(clone)是同源问题: NPU tensor工厂函数普遍未考虑stride保留。
- Fix: 当base format + strided layout + preserve/default format时，
  用`at::infer_dense_strides`计算正确的stride，再用`at::empty_strided`分配。
  附带完整的UT覆盖contiguous/non-contiguous + 各种memory_format组合。
- Reviewability: high -- empty_like的stride语义是PyTorch基本约定
- Review rule: NPU tensor工厂函数(empty_like/zeros_like/ones_like等)
  须对照PyTorch native实现验证stride行为

### D-700: npugraphs backend中boolean index_put_未走fallback

- Hashes: ecf541367, 339dd8295, 16bf63db4
- Root cause: ACL graph capture不支持boolean index_put_但未检测
- Files: test/npu/test_acl_graph_special_op.py (新增测试)
  及npugraphs backend相关文件
- Defect: `tensor[bool_mask] = value`编译为`aten.index_put_`(boolean indices)，
  该操作在ACL graph capture中不安全(cudagraph-unsafe)。
  npugraphs backend未检测此模式，尝试capture导致结果错误或崩溃。
- Fix: 在graph tree中添加boolean index_put_检测，
  遇到时跳过ACL graph capture，fallback到eager执行。
  新增前向/反向/两种backend的测试用例。
- Reviewability: medium -- 需要了解哪些op是graph-unsafe
- Review rule: graph capture backend添加新op支持时，
  须维护unsupported op列表并在capture入口检查

### D-701: NPUGraph中aclrtSetStreamAttribute在CANN<8.5上调用崩溃

- Hashes: 24b84e5d3, 1286cc5a6, 2fdeb42e7, acd84c123, 2ed861f4e, 79c7a1ad9
- Root cause: 调用CANN版本不支持的RTS API
- Files: torch_npu/csrc/core/npu/NPUGraph.cpp
- Defect: `apply_cache_op_info`调用`aclrtSetStreamAttribute(ACL_STREAM_ATTR_CACHE_OP_IFNO, ...)`，
  该API在CANN 8.5.0之前的版本不存在。在旧版CANN上调用导致
  函数指针为null，TORCH_CHECK失败或直接segfault。
- Fix: 在函数入口加`IsGteCANNVersion("8.5.0", "CANN")`检查，
  低版本直接return。
- Reviewability: high -- 新API调用必须有版本守卫
- Review rule: 调用CANN新版本引入的API时，
  必须用IsGteCANNVersion包裹，且在commit中说明最低版本要求

### D-702: d2h pinned memory在tensor reduce路径不生效

- Hashes: 4700f2420
- Root cause: _npu_save遗漏了tensor reduce路径的处理
- Files: torch_npu/utils/serialization.py, test/npu/test_serialization.py
- Defect: `_npu_save`只处理了storage在NPU上的情况(走pin_memory + D2H)，
  但存在另一条路径: tensor reduce机制在到达`_npu_save`之前
  已将NPU storage物化为CPU storage。此时storage已在CPU上，
  跳过了NPU→pinned→CPU的优化路径。
  `use_pinned_memory_for_d2h`配置对这条路径无效。
- Fix: 增加else分支: 即使storage已在CPU上，若配置开启且当前accelerator是NPU，
  分配pinned memory并copy，使后续write_record使用DMA优化。
- Reviewability: medium -- 需要理解tensor序列化有多条进入_npu_save的路径
- Review rule: 序列化优化(如pin memory)须覆盖所有可能的tensor物化路径，
  不能只处理"标准"路径

### D-703: dynamo trace rule缺少memory._get_current_allocator注册

- Hashes: dfaa63163, 7ae6a64cf, 856009930, fe8d11b24, 497cccb5f
- Root cause: 新增的公开函数未注册为dynamo in-graph function
- Files: torch_npu/dynamo/trace_rule.py
- Defect: `torch.npu.memory._get_current_allocator`未在
  `torch_non_c_binding_in_graph_functions_npu`中注册。
  当dynamo trace到此函数调用时，因不识别而尝试pickle该local object，
  触发pickle错误(不可序列化的function object)。
- Fix: 将`torch.npu.memory._get_current_allocator`加入in-graph函数注册表。
- Reviewability: high -- 新增public API时应检查dynamo trace rule
- Review rule: torch_npu新增任何Python公开函数时，
  须同步评估是否需要加入dynamo trace rule

### D-704: IsGteCANNVersion版本比较逻辑对major>8失效

- Hashes: ce677248f, de754b224, 4e9cf465a, 6d6d6a9ad, ea685493e, 2e2d7dcab,
  2fbda6782, ddbfedff4, 0ff9bf558, 811521a83, a2c9adc94, 9409d43fc
- Root cause: 版本号(major,minor)被当作独立维度比较而非元组序
- Files: torch_npu/csrc/core/npu/GetCANNInfo.cpp
- Defect: 原条件`major1 >= 8 && minor1 >= 5 && major2 >= 8 && minor2 >= 5`
  用于判断两个版本是否都>=8.5(从而使用V2版本号格式)。
  但对version 9.0: major=9>=8为true，minor=0>=5为false，整体为false。
  导致CANN 9.x被错误地判定为不使用V2格式，版本比较结果错误，
  进而影响所有依赖IsGteCANNVersion的功能守卫(如D-701)。
  12个cherry-pick说明影响面极广。
- Fix: 改为`(major1==8 && minor1==5 && major2==8 && minor2==5) or (major1>8 && major2>8)`，
  精确匹配8.5或major>8的情况。
- Reviewability: high -- 版本比较是基础设施，错误影响全局
- Review rule: 版本号比较必须用元组序(major,minor)而非独立比较各分量。
  `(major >= M && minor >= N)`模式对major > M时永远有bug

### D-705: custom_fwd/custom_bwd签名落后于upstream torch版本

- Hashes: ac5cf101d, 1425e0ec9, 1dd206de4, 0b5813d04, a0395276f
- Root cause: torch_npu重复实现的API与upstream签名不一致
- Files: torch_npu/npu/amp/autocast_mode.py, test/torch_npu_schema.json
- Defect: torch_npu自定义的`custom_fwd(fwd=None, **kwargs)`使用`**kwargs`接受参数，
  而upstream `torch.amp.custom_fwd`使用具名参数`cast_inputs=None`。
  签名不一致导致: (1) 用户从CUDA迁移时参数传递方式不兼容;
  (2) schema检查失败; (3) torch_npu内部重新实现了autocast逻辑，
  与upstream行为可能有细微差异。
- Fix: 重写为thin wrapper: `@deprecated`标注后委托给
  `torch.amp.custom_fwd(device_type="npu")`。
  删除~50行自定义autocast逻辑。custom_bwd同理。
- Reviewability: high -- public API签名变更应在schema测试中发现
- Review rule: torch_npu中重复实现upstream功能的代码，
  应定期评估是否可替换为upstream调用+device_type参数

<!-- D-skip: cb3c1e95b is stash artifact (index on...), no diff -->
### D-706: 异常类型全捕获导致不可恢复错误被静默吞掉

- Hashes: a6e1bd4d9
- Root cause: exception-type全捕获掩盖不可恢复错误
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `eagerConnectSingleDevice`使用单一`catch(const std::exception&)`捕获所有异常并统一
  fallback到lazy init。问题是HCCL库/设备错误(`c10::Error`)是不可恢复的，
  静默return会让进程在后续collective操作时以更难调试的方式失败。
  `broadcastMasterID`将network错误包装为`std::runtime_error`（丢失类型信息），
  需要通过what()字符串匹配来区分。
- Fix: 三层catch: `c10::DistNetworkError`→WARN+return(可恢复);
  `c10::Error`→ERROR+throw(不可恢复);
  `std::exception`→检查what()是否含"network"来区分。
- Reviewability: high -- catch-all应在CR中被质疑
- Review rule: catch(std::exception&)后直接return/continue必须确认所有子类型都可安全跳过

### D-707: inductor golden_var_list重复变量 + dynamic shape mask缺失

- Hashes: d08e44dcb
- Root cause: 符号系统对象identity vs value比较 + 动态shape守卫缺失
- Files: torch_npu/_inductor/codegen/kernel_analysis.py, torch_npu/_inductor/codegen/triton.py
- Defect: 三个独立问题:
  (1) `golden_var_list`包含重复变量，`dense_size_list`为每个变量生成BLOCK_SUB定义，
  重复导致codegen产生无效kernel;
  (2) `_codegen_mask`在dynamic shape场景下因`is_no_loop_axis`跳过mask生成，
  而动态shape需要mask来处理不对齐的尾块;
  (3) `all_tiling_in_var_list`用对象`x`比较而非`x.symbol()`，
  Python对象identity比较在sympy重建symbol后失败。
- Fix: `_deduplicate_vars`去重; `get_allow_dynamic()`检测shape_env;
  比较改用`.symbol()`; `_filter_and_append_missing`确保tiling轴完整。
- Reviewability: medium -- symbol identity问题需要了解sympy内部机制
- Review rule: sympy symbol比较必须用.symbol()或str()，不可依赖Python is/==

### D-708: aclrtPointerGetAttributes缺少驱动版本守卫

- Hashes: b56a42282 7b3f03568 807bbc9bf 75073c825 63b049b49
- Root cause: API版本守卫缺失
- Files: torch_npu/csrc/core/npu/interface/AclInterface.cpp
- Defect: `AclrtPointerGetAttributesExist()`只检查CANN runtime版本(>=8.5.0)，
  未检查驱动版本。该API在驱动<25.5.0时不存在，直接dlsym会返回null，
  后续调用导致crash。
- Fix: 在runtime版本检查前增加`IsGteDriverVersion("25.5.0")`检查。
- Reviewability: high -- 新API可用性检查应覆盖所有依赖层(驱动+runtime)
- Review rule: 任何ACL新接口的Exist()检查必须同时守卫驱动版本和CANN版本

### D-709: kSynchronizeBusyWaitMillis与上游不一致

- Hashes: 588b30e2d 9515ad443 642313d63 7deab6fd1
- Root cause: 上游常量值同步遗漏
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: busy wait轮询间隔硬编码为10ms，而上游PyTorch 2.7.1已改为1ms。
  导致HCCL_BLOCKING_WAIT=1时每次同步操作多等待~9ms（实测median差值9ms）。
  对小tensor高频collective操作影响显著(1.077ms vs 10.078ms)。
- Fix: `constexpr int64_t kSynchronizeBusyWaitMillis = 10` → `= 1`。
- Reviewability: high -- 常量值变更应在upstream sync时自动检出
- Review rule: 从upstream PyTorch同步代码时，须对比所有hardcoded常量

### D-710: inductor tiling生成器break导致搜索不完整

- Hashes: ce9672dda bcec1ab67 54273eef4 8c6ae9a7c
- Root cause: 循环控制流提前终止导致搜索不完整
- Files: torch_npu/_inductor/codegen/tile_generator.py
- Defect: `descend_one_axis`中的`break`导致tiling搜索在找到第一个候选后就终止内层循环。
  在vector_core=40的设备上（如部分Ascend产品，核数20，vector_core 40），
  early break跳过了更优的block划分，生成的config超出UB(Unified Buffer)限制。
- Fix: 移除`break`; `while True`改为`for max_idx in range(30)`限制迭代上限。
- Reviewability: medium -- 需要理解tiling搜索空间的完整性要求
- Review rule: 搜索/遍历循环中的break/continue需验证不会跳过有效解

### D-711: empty_with_swapped_memory测试缺少正确性断言

- Hashes: 59f32a3d5 29a349bf1 57094aaaf 6b024c491 d2406c660
- Root cause: 测试覆盖不足(无结果断言)
- Files: test/npu/test_swapped_memory_allocator.py
- Defect: test_02只执行计算操作但不验证结果正确性。swapped memory路径可能
  在异步操作中产生错误结果（如stream同步不正确），但测试无法检出。
- Fix: 构造expected_tensor做同样计算，用`torch.allclose`比对。
- Reviewability: high -- 无断言的测试应在CR中被拒绝
- Review rule: 测试函数必须包含结果验证断言，不能只"跑通不crash"

### D-712: user_autotune未适配NPU(core dump)

- Hashes: 7f4be1158
- Root cause: CUDA-specific代码路径未做NPU适配
- Files: torch_npu/_inductor/codegen/triton.py, torch_npu/_inductor/codegen/wrapper.py,
  torch_npu/_inductor/npu_triton_heuristics.py, test/_inductor/test_user_autotune_npu.py
- Defect: PyTorch inductor的`user_autotune`使用CUDA的Grid和heuristics基类，
  在NPU上调用会触发CUDA runtime操作导致core dump。
  另外`gen_triton_ext_imports`缺少`@staticmethod`装饰器，
  在wrapper codegen中作为类方法调用时传入多余的self参数。
- Fix: 在`define_kernel`中做字符串替换:
  `triton_heuristics.user_autotune(` → `npu_triton_heuristics.user_autotune_npu(`;
  `PrecomputedGrid` → `PrecomputedGridNpu`; 注入NPU ext imports。
  添加`@staticmethod`。
- Reviewability: medium -- 需要了解inductor的codegen pipeline
- Review rule: inductor codegen新增CUDA heuristics引用时须确认NPU替代路径存在

### D-713: NPUSwappedMemoryAllocator svm_deleter未实现释放

- Hashes: 7672a5b86 e0e3016ff a0c758865 0bbadebc7 8f790fe78 4ea78ef83
- Root cause: 资源释放未实现(空deleter)
- Files: test/npu/test_swapped_memory_allocator.py (含完整测试套件)
- Defect: `svm_deleter`函数体为空或缺失释放逻辑，分配的SVM(Shared Virtual Memory)
  host内存和设备映射永远不会释放，导致持续内存泄漏。
  每次`empty_with_swapped_memory`调用都会泄漏host内存块+AclrtHostRegister映射。
- Fix: 实现完整释放流程:
  检查指针非空 → 从`memBlocks`映射表查找原始host指针 →
  `AclrtHostUnregister` → `aclrtFreeHost` → 删除映射条目。
  补充完整的分配/释放/弱引用测试套件。
- Reviewability: high -- allocator必须有对应deallocator是基本原则
- Review rule: 新增内存分配路径必须同时实现释放路径，CR必须检查alloc/free配对

### D-714: FA pattern example设备硬编码不匹配

- Hashes: 98029ad2a 768d3c4e9 240dcb176 a23345516 14b889b7c
- Root cause: tensor设备硬编码导致跨设备不匹配
- Files: torch_npu/_inductor/npu_fusion_attention_graph.py
- Defect: `register_fa_pass`中`c_inp = functools.partial(torch.tensor, 2.0, device=device)`
  使用独立传入的device参数构造scalar tensor，而非从实际输入tensor推断设备。
  多设备或device重映射场景下，c_inp可能在不同设备上创建，导致后续计算跨设备失败。
- Fix: `device=device` → `device=g_inp().device`，从实际创建的input tensor获取设备。
- Reviewability: high -- device参数应从tensor推断而非独立传递
- Review rule: 多tensor操作中，辅助tensor的device应从主tensor推断

### D-715: onnx测试用例批量skip不兼容API

- Hashes: 31205c19a 47f9516b0
- Root cause: onnx API兼容性断裂 + 测试适配
- Files: test/onnx/test_combined_onnx_ops.py
- Defect: onnx导出测试中多个wrapper_npu_*测试用例与当前onnx API版本不兼容。
  group_norm_silu和rotary_mul的onnx API签名或行为已变更，
  原有测试直接调用会导致失败。
- Fix: 对10+个不兼容测试用例添加`@unittest.skip("skip now")`。
  注意: 这是临时skip而非真正修复，技术债务。
- Reviewability: high -- 批量skip应标注跟踪issue
- Review rule: skip测试必须附带issue编号和恢复计划

### D-716: device context manager仅检查-1忽略其他负值

- Hashes: 33f80abb1 5840910e1 b0d91f540 24a6a8d52 63c448335
- Root cause: 边界条件检查不完整(仅覆盖单一负值)
- Files: torch_npu/npu/utils.py, test/distributed/test_with_device.py
- Defect: `device.__enter__`用`if self.idx == -1`跳过设备切换。
  但负浮点数通过`_get_device_index`后可产生-128、-258等负值
  (C层int8截断或floor转换)，这些值不等于-1但同样无效。
  对这些值调用`exchangeDevice`会触发底层驱动错误。
- Fix: `idx == -1` → `idx < 0`; `__exit__`也增加`idx < 0`检查。
- Reviewability: high -- 负值语义统一应在初始设计时确定
- Review rule: 设备索引有效性检查统一用`< 0`而非`== -1`

### D-717: StorageWeakRefWrapper引用计数重构

- Hashes: 8d87c3d39
- Root cause: 自定义C绑定与上游API重复且引用计数语义不一致
- Files: torch_npu/csrc/npu/Module.cpp, torch_npu/npu/_graph_tree.py
- Defect: (1) NPU自定义`_storage_Use_Count`使用`(c10::StorageImpl*)storage_impl_ptr`
  裸指针转换，不安全且与上游`torch._C._storage_Use_Count`功能重复;
  (2) `extra_ref_check`时只扣减1个引用，但实际有2个额外引用
  (Python storage对象 + cached Tensor)，导致expired()判断错误;
  (3) `Type[S]`泛型标注在runtime类型检查时不一致。
- Fix: 删除C++侧`_storage_Use_Count`; 改用PyTorch原生API;
  extra_ref扣减从1改为2; 类型标注改为具体类型。
- Reviewability: medium -- 引用计数逻辑需要理解storage生命周期
- Review rule: 与upstream重复的C绑定应定期清理，优先使用upstream版本

### D-718: npugraph_ex scope模块导出遗漏

- Hashes: 1833d6d52 0a14e5734 993ce9488 52c8c065d 4e6a200fd
- Root cause: 模块导出遗漏
- Files: torch_npu/npu/npugraph_ex/__init__.py, torch_npu/npu/npugraph_ex/scope/__init__.py
- Defect: `limit_core_num` API实现存在于`torch_npu.dynamo.npugraph_ex.scope`，
  但公开入口`torch_npu.npu.npugraph_ex`缺少`scope`子模块的import和re-export。
  用户无法通过文档路径`torch.npu.npugraph_ex.limit_core_num()`访问该功能。
- Fix: 创建`scope/__init__.py`作为thin wrapper委托到dynamo实现;
  在`npugraph_ex/__init__.py`中`from . import scope`。
- Reviewability: high -- public API的导出完整性应有测试守护
- Review rule: 新增public API必须在`__init__.py`层级可达，
  并有import测试验证

### D-719: aclrtMallocHostWithCfg缺少驱动版本守卫

- Hashes: afe23cd3f e791ba1ed 4aaa20af1 b15480b1a
- Root cause: API版本守卫缺失(与D-708同模式)
- Files: torch_npu/csrc/core/npu/interface/AclInterface.cpp,
  torch_npu/csrc/core/npu/NPUAllocatorConfig.cpp
- Defect: 与D-708同类问题。`AclrtMallocHostWithCfgExist()`只检查CANN runtime版本，
  缺少驱动版本检查。驱动<25.5.2时API不存在。
  错误提示信息也未包含驱动版本要求，用户无法定位问题。
- Fix: 增加`IsGteDriverVersion("25.5.2")`检查;
  更新WARN消息包含驱动版本要求。
- Reviewability: high -- D-708修复时应全局排查所有类似API
- Review rule: 同一类修复应一次性覆盖所有实例，不应分多个MR

### D-720: npu_format_cast customize_dtype从位置参数改为keyword-only

- Hashes: 30d7606e4 fbb4a145c db10780dc 120341a1b f1e37df8a
- Root cause: API签名不符合keyword-only规范
- Files: torch_npu/csrc/aten/npu_native_functions.yaml, test/torch_npu_schema.json
- Defect: `npu_format_cast`系列4个函数的`customize_dtype`定义为位置参数，
  但语义上是可选配置项。位置参数意味着用户必须按顺序传参或显式跳过，
  不符合PyTorch的API设计惯例（可选参数应为keyword-only）。
- Fix: 在yaml定义中将`customize_dtype`和`input_dtype`移到`*`分隔符之后，
  使其成为keyword-only参数。同步更新schema.json。
- Reviewability: high -- API签名变更应有breaking change评估
- Review rule: 可选参数默认使用keyword-only(`*`后)，除非有性能原因需要位置传参

### D-721: 设备错误处理中的递归初始化循环

- Hashes: 37f9bf745 f007a999e be49dc036 cf25b793c e1992186f
- Root cause: 初始化路径中的循环依赖
- Files: torch_npu/csrc/core/npu/NPUException.cpp
- Defect: `cacheDeviceErrorVerboseMsg`调用`c10_npu::current_device()`，
  后者在设备未初始化时触发`initDevice()`。若初始化失败（如驱动不可用），
  错误处理路径再次调用`cacheDeviceErrorVerboseMsg`，形成无限递归直到栈溢出。
- Fix: 改用`GetLocalDevice()`（仅读取thread_local变量，不触发初始化）;
  返回值<0时直接return空字符串。
- Reviewability: medium -- 需要理解current_device()的副作用链
- Review rule: 错误处理/日志路径禁止调用可能触发初始化的函数

### D-722: profiler config重复加载 + is_cluster属性清理

- Hashes: 972e30401 a886a76b8 7243d5fee 2ddd04c9a
- Root cause: 重复初始化 + 废弃状态清理
- Files: torch_npu/profiler/analysis/_profiler_config.py,
  torch_npu/profiler/analysis/prof_parse/_cann_file_parser.py,
  test/profiler/analysis/test_profiler_config.py
- Defect: `load_info`可被多次调用，每次都重新解析JSON和加载配置。
  `_is_cluster`属性在CANN路径不存在时导致异常但缺少保护。
  `load_is_cluster`在文件不存在时抛异常而非优雅降级。
- Fix: 添加`_is_load`标志防止重复加载;
  移除`_is_cluster`属性(由`load_activities`替代);
  删除相关测试用例。
- Reviewability: medium -- 状态管理重构需要理解调用链
- Review rule: 带副作用的load/init方法应有幂等保护(already-loaded flag)

### D-723: DeviceProperty.multi_processor_count未设置导致DataParallel异常

- Hashes: d38c6fe18 78f8fcee8 8d9f4c542 197908353 2adc584a8
- Root cause: NPU设备属性映射缺失(CUDA→NPU属性对齐)
- Files: torch_npu/csrc/npu/Module.cpp, test/npu/test_torch_npu.py
- Defect: `multi_processor_count`在NPU设备上未赋值(默认0)。
  PyTorch的`DataParallel`使用此属性做设备负载均衡计算，
  值为0导致除零或分配失败。NPU需要映射到等价概念。
- Fix: 优先用`vector_core_num`; vector_core_num=0时fallback到`cube_core_num`;
  从"不支持属性"列表中移除`multi_processor_count`。
- Reviewability: high -- DataParallel是基础功能，属性缺失应在集成测试发现
- Review rule: 新增设备属性时须检查PyTorch框架代码中所有读取该属性的路径

### D-724: 分布式checkpoint测试异常类型不匹配

- Hashes: 795654945
- Root cause: 上游异常类型变更导致测试断言失败
- Files: test/distributed/checkpoint/test_checkpoint.py
- Defect: `_test_load`中`fail_read_metadata`场景抛出的异常类型
  与assertRaises期望的类型不一致。可能是上游PyTorch变更了
  checkpoint读取失败时的异常层次。
- Fix: 对3处`fail_read_metadata`调用添加`ignore_exception_type=True`。
  注意: 这是workaround而非根因修复。
- Reviewability: high -- 异常类型变更应在upstream sync时检出
- Review rule: assertRaises的异常类型应与被测代码的最新异常层次一致

### D-725: serialization save硬编码True覆盖_use_new_zipfile_serialization

- Hashes: c85822829 09f67a700 d45a68df7 4511a1da1 8f5e606dd
- Root cause: 参数硬编码覆盖调用者意图
- Files: torch_npu/utils/serialization.py
- Defect: `save`函数中调用`torch.serialization.save`时，
  第5个参数(`_use_new_zipfile_serialization`)被硬编码为`True`，
  忽略了调用者传入的实际值。用户显式传入`False`要求旧格式序列化时，
  仍然使用新格式，导致下游反序列化兼容性问题(如旧版PyTorch无法读取)。
- Fix: 将`True`替换回`_use_new_zipfile_serialization`变量。
- Reviewability: high -- 参数透传应在diff中一眼可见
- Review rule: wrapper函数参数必须透传给底层调用，
  硬编码替代参数值需要显式注释说明原因


<!-- D-708 addendum cherry-picks: 7b3f03568 807bbc9bf 75073c825 63b049b49 -->
<!-- D-709 addendum cherry-picks: 9515ad443 642313d63 7deab6fd1 -->
<!-- D-710 addendum cherry-picks: bcec1ab67 54273eef4 8c6ae9a7c -->
<!-- D-711 addendum cherry-picks: 29a349bf1 57094aaaf 6b024c491 d2406c660 -->
<!-- D-713 addendum cherry-picks: e0e3016ff a0c758865 0bbadebc7 8f790fe78 4ea78ef83 -->
<!-- D-714 addendum cherry-picks: 768d3c4e9 240dcb176 a23345516 14b889b7c -->
<!-- D-715 addendum cherry-picks: 47f9516b0 -->
<!-- D-716 addendum cherry-picks: 5840910e1 b0d91f540 24a6a8d52 63c448335 -->
<!-- D-718 addendum cherry-picks: 0a14e5734 993ce9488 52c8c065d 4e6a200fd -->
<!-- D-719 addendum cherry-picks: e791ba1ed 4aaa20af1 b15480b1a -->
<!-- D-720 addendum cherry-picks: fbb4a145c db10780dc 120341a1b f1e37df8a -->
<!-- D-721 addendum cherry-picks: f007a999e be49dc036 cf25b793c e1992186f -->
<!-- D-722 addendum cherry-picks: a886a76b8 7243d5fee 2ddd04c9a -->
<!-- D-723 addendum cherry-picks: 78f8fcee8 8d9f4c542 197908353 2adc584a8 -->
<!-- D-725 addendum cherry-picks: 09f67a700 d45a68df7 4511a1da1 8f5e606dd -->

### D-726: inductor user_autotune NPU适配缺失导致core dump

- Hashes: 6b5c697df 2c194e8af 8544fbf6b
- Root cause: CUDA路径未适配NPU(inductor codegen)
- Files: torch_npu/_inductor/codegen/wrapper.py,
  torch_npu/_inductor/codegen/triton.py,
  torch_npu/_inductor/npu_triton_heuristics.py,
  test/_inductor/test_user_autotune_npu.py
- Defect: inductor的`user_autotune`和`PrecomputedGrid`/`FixedGrid`是CUDA-specific实现，
  在NPU设备上直接调用导致core dump。`define_kernel`生成的kernel body中包含
  `triton_heuristics.user_autotune(`等CUDA路径调用，NPU runtime无法处理。
- Fix: 新增`user_autotune_npu`函数和`PrecomputedGridNpu`/`FixedGridNpu`类;
  override `define_kernel`做字符串替换，将CUDA heuristics替换为NPU版本;
  `gen_triton_ext_imports`改为`@staticmethod`以便在wrapper中直接调用。
- Reviewability: medium -- 需要理解inductor codegen pipeline中Grid/Heuristics的角色
- Review rule: inductor新增user-facing codegen路径时须同步适配NPU wrapper

### D-727: mlir backend测试缺失

- Hashes: 3c13ae5e7 7ca34ff50 2b2424790 125ff02a4 841207b00
- Root cause: 测试覆盖缺失
- Files: test/_inductor/test_torch_mlir_enable.py
- Defect: MLIR backend (`npu_backend: "mlir"`)没有对应的端到端测试，
  无法验证`options`参数和`config`设置两种启用方式是否正常工作。
- Fix: 新增测试文件，覆盖`options={"npu_backend": "mlir"}`和
  `torch._inductor.config.npu_backend = "mlir"`两种启用路径，
  验证编译输出包含`mlir_fused`关键字。
- Reviewability: high -- 纯测试补充，无生产代码风险
- Review rule: 新增backend/config选项时须同步提交对应的集成测试

### D-728: aclgraph测试用例缺少910C(Ascend910_93)设备支持

- Hashes: 0a649b884 cd111d1e3 4f4625184
- Root cause: 测试设备白名单不完整
- Files: test/npu/test_aclgraph_launch_host_func.py,
  test/npu/test_aclgraph_support_blocking.py,
  test/npu/test_aclgraph_update.py
- Defect: 多个aclgraph测试用例的`@SupportedDevices`装饰器仅包含`Ascend910B`
  (部分包含`Ascend910C`)，缺少新硬件标识`Ascend910_93`，导致该设备上测试被跳过。
  注意`Ascend910C`与`Ascend910_93`是不同的设备标识。
- Fix: 将所有`@SupportedDevices(['Ascend910B'])`和`['Ascend910B', 'Ascend910C']`
  统一替换为`['Ascend910B', 'Ascend910_93']`。涉及3个测试文件约15处修改。
- Reviewability: high -- 纯装饰器修改，可批量搜索发现
- Review rule: 新增硬件型号时须全局搜索SupportedDevices装饰器并更新

### D-729: load() API不支持PathLike对象和mmap路径检查类型错误

- Hashes: 573177afd 75a699b37 762804418 f00e39b37 9218a4b3e
- Root cause: 上游API一致性(PathLike支持)
- Files: torch_npu/utils/serialization.py
- Defect: `load()`的mmap分支用`isinstance(f, str)`检查路径，
  不接受`pathlib.Path`等`os.PathLike`对象。`UntypedStorage.from_file`
  也需要`str`而非PathLike。异常类型用`TypeError`但语义上应为`ValueError`。
  与upstream PyTorch的`torch.load` API行为不一致。
- Fix: 新增`_is_path()` TypeGuard函数，支持`str`和`os.PathLike`;
  用`os.fspath(f)`转换后传给`from_file`; 异常改为`ValueError`。
- Reviewability: high -- 类型检查差异在diff中一眼可见
- Review rule: wrapper函数的参数类型检查须与upstream保持一致

### D-730: clone()不保留tensor stride信息

- Hashes: 64ebb6e0f 76e13e58c 1b0f3a334 2d86ea375 4b9557261
- Root cause: Tensor语义一致性(stride保留)
- Files: torch_npu/csrc/aten/ops/op_api/CloneKernelOpApi.cpp
- Defect: `clone()`实现始终使用`apply_tensor_without_format`创建目标tensor，
  忽略`MemoryFormat::Preserve`语义。当源tensor有非默认stride时(如transpose后的view)，
  clone结果的stride与源不一致，破坏后续依赖stride布局的计算(如channels_last)。
- Fix: 当`memory_format == Preserve`且源tensor是non-overlapping-dense时，
  用`empty_strided(src.sizes(), src.strides())`创建目标tensor以保留stride;
  其余情况保持原逻辑。
- Reviewability: high -- upstream的clone实现有相同逻辑可对照
- Review rule: tensor工厂函数须尊重MemoryFormat参数语义

### D-731: as_strided使用自定义setStrided而非upstream实现

- Hashes: 22b82f571 ca4ef0313 14ac3813e c4568eb31 55de335f0
- Root cause: 自定义代码与upstream分叉(冗余实现)
- Files: torch_npu/csrc/aten/common/ResizeNpu.h,
  torch_npu/csrc/aten/common/TensorShape.cpp
- Defect: NPU维护了自定义的`checkInBoundsForStorage`和`setStrided`(60行),
  与`at::native::setStrided`功能重复但实现细节可能有偏差。
  边界检查逻辑随upstream演进可能出现不一致。
- Fix: 删除自定义`checkInBoundsForStorage`和`setStrided`(60行),
  改用`at::native::setStrided`。
- Reviewability: high -- 自定义代码与upstream对比即可发现
- Review rule: 优先使用upstream实现; 自定义替代须有明确的NPU-specific理由

### D-732: npu_dtype_cast DTensor分发策略不支持StridedShard

- Hashes: 4d4a7a5a8 c45faa651 1f261796c 99d85d2c8
- Root cause: DTensor op策略注册不完整
- Files: torch_npu/distributed/tensor/_pointwise_ops.py,
  torch_npu/distributed/tensor/__init__.py,
  torch_npu/utils/dtensor.py,
  test/distributed/_tensor/test_pointwise_ops.py
- Defect: `npu_dtype_cast`系列op在`dtensor.py`中作为简单pointwise注册，
  不支持`_StridedShard` placement type。当DTensor使用StridedShard分片时，
  dtype cast操作无法正确推导输出分片策略，导致运行时错误。
- Fix: 从`dtensor.py`的简单列表中移除dtype_cast ops;
  新建`_pointwise_ops.py`，使用`register_op_strategy`注册，
  调用upstream的`pointwise_strategy`支持包括StridedShard在内的所有placement。
- Reviewability: medium -- 需要理解DTensor op strategy注册机制
- Review rule: 自定义op注册DTensor策略时须验证所有placement type

### D-733: index_put_ bool索引在cudagraph中触发nonzero导致NPU错误

- Hashes: 25b5698c8 e800763c3 f0c9519f8
- Root cause: upstream函数未适配NPU(cudagraph兼容性检查)
- Files: torch_npu/_inductor/utils.py,
  torch_npu/_inductor/__init__.py
- Defect: upstream的`get_first_incompatible_cudagraph_node`不检查
  `index_put`的bool索引(会在capture时触发`.nonzero()`产生动态shape)。
  NPU上cudagraph capture期间执行nonzero导致graph录制失败或core dump。
- Fix: monkey-patch `get_first_incompatible_cudagraph_node`，
  新增`_fx_node_is_input_dependent_cudagraph_unsafe`检查:
  对`index_put`/`index_put_`/`_unsafe_index_put`，检查indices是否包含
  bool/uint8类型，是则标记为cudagraph-unsafe。patch注入3个模块。
- Reviewability: low -- 需要理解cudagraph capture的dynamic shape限制
- Review rule: cudagraph兼容性检查须覆盖所有可能产生动态shape的op

### D-734: flight recorder dump文件名使用未初始化的global rank

- Hashes: feaf1a4cf fc04b3e18 b96f50962 c59d7cedd 8dee4ac67
- Root cause: 初始化顺序错误(状态在使用前未就绪)
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `ProcessGroupHCCL`构造函数中，`globalRank()`在watchdog启动后才被调用，
  但flight recorder dump文件名生成依赖global rank值。
  如果dump在rank初始化前触发，文件名中rank为默认值(0或-1)，导致多进程文件覆盖。
- Fix: 在watchdog启动前显式调用`globalRank()`，确保rank值被缓存。单行修复。
- Reviewability: medium -- 需要理解构造函数中的初始化依赖顺序
- Review rule: lazy-init的状态若被异步路径使用，须在启动异步任务前强制初始化

### D-735: dynamo将torch.device误判为NPU in-graph函数

- Hashes: 5b26465e3 6b8b7cb1f 42067023d e88b5b1c6 0d45fe33f
- Root cause: dynamo trace rule过度覆盖
- Files: torch_npu/utils/_dynamo.py,
  test/_inductor/test_compile_autograd.py
- Defect: `UserDefinedClassVariable__new__`中将`torch.device`加入
  `TorchInGraphFunctionVariable`列表，使dynamo在compiled autograd的
  backward中遇到`torch.device("npu")`时将其内联为in-graph常量。
  但torch.device是通用PyTorch类，dynamo已有原生处理逻辑，
  NPU的覆盖导致compiled autograd trace中设备信息丢失。
- Fix: 从NPU特殊处理列表中移除`torch.device`，让dynamo使用原生处理路径。
- Reviewability: high -- 检查TorchInGraphFunctionVariable列表即可发现
- Review rule: 向dynamo in-graph列表添加类型前须确认dynamo无原生支持

### D-736: profiler export_type默认值None导致warning

- Hashes: 0d2cd849d 6004dd07c cb72ca1a9 0d330429d 4da21f35d
- Root cause: 参数默认值不合理
- Files: torch_npu/profiler/experimental_config.py,
  test/torch_npu_schema.json
- Defect: `_ExperimentalConfig`的`export_type`参数默认值为`None`，
  后续代码路径在None时产生warning或需要额外的None检查。
  用户未显式指定时应有合理的默认行为。
- Fix: 将`export_type`默认值从`None`改为`ExportType.Text`; 同步更新schema JSON。
- Reviewability: high -- 参数签名检查即可发现
- Review rule: 可选参数默认None时须确认None在所有下游路径都有正确处理

### D-737: CI Dockerfile CPython版本路径硬编码过期

- Hashes: f225ea66c
- Root cause: CI配置版本漂移
- Files: ci/docker/ARM/Dockerfile, ci/docker/X86/Dockerfile
- Defect: Dockerfile中硬编码了CPython路径(3.10.18/3.11.13)，
  基础镜像更新CPython到3.10.19/3.11.14后路径不存在，
  pip/python符号链接失效导致CI构建失败。
- Fix: 更新路径: cpython-3.10.18→3.10.19, cpython-3.11.13→3.11.14。
- Reviewability: high -- 构建失败立即暴露
- Review rule: CI Docker配置中的版本路径应参数化或使用glob匹配

### D-738: Python 3.9不支持X|Y类型联合语法

- Hashes: 8f6ca366d a18841cce
- Root cause: Python版本兼容性
- Files: torch_npu/distributed/fsdp/_add_fsdp_patch.py
- Defect: `int | torch.SymInt`使用PEP 604的`X | Y`运行时联合语法，
  Python 3.10+才支持。在Python 3.9上import时抛`TypeError`。
- Fix: 改为`Union[int, torch.SymInt]`(typing模块，3.9兼容)。
- Reviewability: high -- 类型注解语法在3.9 CI中会直接报错
- Review rule: 类型注解须兼容项目的最低Python版本(3.9)

### D-739: NPU deterministic empty/resize自定义fill与upstream冲突

- Hashes: dd06f35ae 3318e5c80 920efd90f 38c049c83 ad1698310
  aeabca9cd 8c0b41d61 2ce79111e ae467d2ca 2a0266cf0 557a510db
- Root cause: 自定义代码与upstream语义冲突(deterministic mode)
- Files: torch_npu/csrc/aten/common/ResizeNpu.cpp,
  torch_npu/csrc/aten/common/TensorFactories.cpp,
  test/unsupported_test_cases/.pytorch-disabled-tests.json
- Defect: NPU在`empty`/`empty_like`/`empty_strided`/`resize_`中添加了
  deterministic fill逻辑(`fill_empty_deterministic_`/`fill_resize_deterministic_`),
  与upstream PyTorch的deterministic mode实现产生double-fill或语义冲突。
  upstream的fill发生在framework层，NPU的fill发生在device层，
  两次fill浪费性能且可能在timing上产生竞争。
- Fix: 移除NPU层的4处deterministic fill调用;
  删除`should_fill_empty_deterministic`辅助函数;
  移除`<ATen/native/ResizeCommon.h>`和`<ATen/native/TensorFactories.h>`依赖;
  将22个相关测试用例加入disabled列表(NPU不自行fill后由upstream负责)。
- Reviewability: medium -- 需要理解upstream deterministic mode的分层设计
- Review rule: 与upstream有语义重叠的自定义实现在sync时须评估是否仍需要

### D-740: dynamo trace rule缺少NPU运行时函数注册

- Hashes: 8fd7f72b6 2d54cf64b 5a7684e9a 5416b915d f9a89bb2e
- Root cause: dynamo trace rule注册不完整
- Files: torch_npu/dynamo/trace_rule.py,
  test/_inductor/test_add_triton_wrap.py
- Defect: `torch.npu.is_initialized`、`torch.npu._get_current_allocator`、
  `torch.npu.is_bf16_supported`和`torch_npu._C._npu_resetAccumulatedMemoryStats`
  未注册为dynamo in-graph函数。
  编译图中调用这些函数时dynamo会graph-break，影响编译覆盖率。
  测试文件缺少`import torch_npu._inductor`导致inductor NPU后端未加载。
- Fix: 将4个函数分别加入`torch_non_c_binding_in_graph_functions_npu`
  和`torch_c_binding_in_graph_functions_npu`字典;
  测试文件补充import。
- Reviewability: high -- graph break日志中可直接定位缺失的函数
- Review rule: 新增NPU运行时API时须同步注册dynamo trace rule

### D-741: get_device_properties使用aclrtGetMemInfo触发NN进程数限制

- Hashes: 42da1f41a 9a1a043db 35cd8b576 33f2b034e 460f536ec 4db67ffe2
- Root cause: API选择不当(资源受限API用于信息查询)
- Files: torch_npu/csrc/npu/Module.cpp,
  third_party/acl/inc/acl/acl_rt.h
- Defect: `initDeviceProperty`使用`aclrtGetMemInfo(ACL_HBM_MEM, ...)`获取总内存，
  但此API需要NN(Neural Network)进程上下文，受NN进程数上限约束。
  当多进程并发调用时报"NN processes exceeds the limit"。
  查询设备静态属性不应消耗NN进程资源。
- Fix: CANN 8.5+使用无状态的`aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_TOTAL_GLOBAL_MEM_SIZE, ...)`替代;
  低版本CANN保持原路径; ACL头文件新增多个设备属性枚举值。
- Reviewability: low -- 需要理解CANN NN进程模型才能意识到GetMemInfo的副作用
- Review rule: 查询设备静态属性应优先使用无状态API(如aclrtGetDeviceInfo)

### D-742: CachingHostAllocator_isPinned空指针未检查

- Hashes: 96a55c00f 6e7280ecd 30becddab 5660d5196 3bf6c6f33
- Root cause: 空指针检查缺失
- Files: torch_npu/csrc/core/npu/CachingHostAllocator.h
- Defect: `CachingHostAllocator_isPinned(void* ptr)`直接对ptr调用
  `AclrtPointerGetAttributes`等API，未检查nullptr。
  传入nullptr时行为未定义(可能crash或返回错误结果)。
  上游PyTorch的CUDA版本有相同的null check。
- Fix: 函数入口添加`if (ptr == nullptr) return false;`。3行修复。
- Reviewability: high -- 函数入口缺少null check在review中可直接发现
- Review rule: 接收裸指针的public API须在入口检查nullptr

### D-743: allocator padding大小条件分支错误(Ascend910_95特殊处理)

- Hashes: bd57c5c46 fd61281e9 7e071d01a bbe936df1 6113f7b7f
- Root cause: 硬件特殊处理逻辑错误(内存对齐)
- Files: torch_npu/csrc/core/npu/NPUCachingAllocator.cpp,
  test/npu/test_allocator_envs.py
- Defect: `AddPadSize()`对Ascend910_95返回0(无padding)，其余返回32。
  这个特殊处理导致910_95上分配的内存缺少尾部padding，
  某些aclnn op在超出分配范围读取时产生越界访问。
  测试中的`math.ceil(size/512)*512`计算也因padding差异而在不同硬件上结果不同。
- Fix: 移除`AddPadSize()`动态函数，统一使用`constexpr kPadSize = 32`常量;
  简化测试断言为统一的`(size + 32) // 512 + 1) * 512`公式;
  添加`@SupportedDevices(['Ascend910B'])`和`torch.npu.synchronize()`。
- Reviewability: medium -- 需要理解NPU内存分配的对齐和padding要求
- Review rule: 硬件特化的内存分配参数变更须有对应的越界检测测试

### D-744: inductor invoke_subgraph未注册导致fallback

- Hashes: ada7f96e2 b099e0682 f58d695bb e2dac9e99
- Root cause: inductor op注册列表不完整
- Files: torch_npu/_inductor/lowering_op_list.py
- Defect: `torch.ops.higher_order.invoke_subgraph`未加入`GENERATE_LIST`，
  导致使用`invoke_subgraph`的模型在inductor编译时fallback到eager模式。
  fallback路径在NPU上可能触发设备不匹配或性能退化。
- Fix: 将`torch.ops.higher_order.invoke_subgraph`加入`GENERATE_LIST`。单行修复。
- Reviewability: high -- inductor编译日志中fallback warning可直接定位
- Review rule: 新增higher_order op时须同步注册到NPU inductor op列表

### D-745: 测试断言过严+平台假设错误(profiling event query计数)

- Hashes: `50b4c1fa0` `2f63ecaa9` `a99b25572` `92bd0c9bb`
- Files: `test/npu/test_multi_stream_lazy_reclaim.py`
- Root cause: test环境假设错误
- Defect: `TestMultiStreamLazyReclaim`测试在x86环境上profiling工具统计的event query计数与预期不符。测试用`assertLess`断言lazy模式计数严格小于eager模式，但在x86上两者可能相等。此外测试未区分平台差异，直接在所有架构上运行。
- Fix: 1)添加`@unittest.skipUnless(IS_ARM64)`跳过非ARM平台; 2)将`assertLess`放宽为`assertLessEqual`
- Reviewability: medium - 断言严格度需要运行时数据验证，但平台限制可在review中提出
- Review rule: profiling/性能测试的断言应使用`assertLessEqual`而非`assertLess`，除非有理论保证严格不等；涉及硬件行为的测试须标注平台约束

### D-746: dynamo EventVariable缺少python_type方法导致profiling断图报错

- Hashes: `70e2290c5` `aa5be1e4d` `7fc7414dd` `edb94d7f6` `e9dbf867e`
- Files: `torch_npu/utils/_dynamo.py`
- Root cause: upstream monkey-patch不完整
- Defect: `fullgraph=False`场景下dynamo断图时，`EventVariable`需要`python_type`方法来处理profiling上下文，但upstream的`EventVariable`没有这个方法。NPU侧的dynamo适配层已有大量monkey-patch但遗漏了这个。触发条件是event在with语句中使用且发生断图。
- Fix: 向`EventVariable`monkey-patch添加`python_type`方法，返回`type(self.value)`
- Reviewability: low - 需要特定的断图路径+profiling组合才暴露，静态review难以发现
- Review rule: 对upstream Variable类做monkey-patch时应检查基类要求的所有协议方法(python_type/as_python_constant等)，建议维护一份协议方法checklist

### D-747: inductor blanket buffer-free抑制导致tensor生命周期过长 + stride编译/运行不一致

- Hashes: `c088eab9e`
- Files: `torch_npu/_inductor/__init__.py` `torch_npu/_inductor/codegen/wrapper.py` `torch_npu/_inductor/config.py` `torch_npu/_inductor/ir.py`
- Root cause: 过度防御性编码 + 编译期/运行期语义不一致
- Defect: 两个独立问题在同一PR修复: 1)`NPUWrapperCodeGen.make_buffer_free`直接返回空字符串，完全禁止了所有buffer释放，导致tensor生命周期不必要地延长(内存泄漏); 2)NPU的`_to_copy`和`reshape`在运行时强制Contiguous内存格式，导致stride与编译期fake tensor不一致，触发stride assertion失败。
- Fix: 1)删除`make_buffer_free`覆盖，恢复正常buffer释放; 2)新增`patch_extern_kernel_codegen_size_asserts`，对特定op跳过stride断言，通过`skip_specific_stride_asserts`配置控制
- Reviewability: high/low - buffer-free抑制是明显的code smell，review可发现; stride不一致需运行时触发
- Review rule: 1)override上游方法为空实现时须附带注释说明何时可以移除; 2)NPU设备的内存格式强制行为(Contiguous)应记录在适配层文档中

### D-748: copy_对负视图(neg view)的判断只检查src不检查self

- Hashes: `b123a91a6` `d0a321c18` `a8e90f69d` `d404451f1` `a8073bc0d`
- Files: `torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp`
- Root cause: neg flag对称性遗漏
- Defect: `copy_`的complex tensor路径中，原代码用`if (src.is_neg())`判断是否需要对self做neg，但这忽略了self本身可能已经是neg view的情况。正确语义是当且仅当src和self的neg flag不同时才需要翻转。测试用例`test_copy_transpose_math_view_npu_int64`暴露了这个问题。
- Fix: 将条件从`src.is_neg()`改为`self.is_neg() != src.is_neg()`(XOR语义)
- Reviewability: high - 对称性检查是经典的code review发现点，只看一侧flag就应该追问另一侧
- Review rule: 涉及tensor view flag(neg/conj)的copy/clone操作，必须同时检查src和dst两侧的flag组合

### D-749: aclnn extension路径拼接缺少torch_npu/前缀

- Hashes: `661a5c908` `aaa0d54d2` `2b5600b31` `dc5b1bb84` `591354a8e`
- Files: `torchnpugen/gen_backend_stubs.py`
- Root cause: 硬编码路径缺少目录层级
- Defect: `gen_backend_stubs.py`中当使用aclnn extension时，`exposed_path`拼接为`utils/exposed_api.py`，但正确路径需要`torch_npu/utils/exposed_api.py`前缀。else分支已经用了正确路径，if分支遗漏了。
- Fix: 路径从`utils/exposed_api.py`改为`torch_npu/utils/exposed_api.py`
- Reviewability: high - if/else两分支路径不一致，diff review可直接发现
- Review rule: 同一变量在不同分支中构造时，检查各分支的path segment是否一致

### D-750: inductor lowering函数定义但未注册到aten dispatch(pow)

- Hashes: `1bc0b2dfa` `cfde3e6a1`
- Files: `torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/npu/inductor_patch/lowering.py`
- Root cause: 注册装饰器遗漏
- Defect: `pow_native`函数已定义并有`@make_pointwise`装饰器，但缺少`@register_to_aten(aten_fn=aten.pow)`装饰器，导致pow操作走不到NPU的lowering实现。相邻函数`pow_recursive`有注册但`pow_native`没有。
- Fix: 添加`@register_to_aten(aten_fn=aten.pow)`装饰器
- Reviewability: high - 同文件其他函数都有注册装饰器，遗漏一个是机械检查可发现的
- Review rule: lowering文件中每个实现函数必须同时有`@make_pointwise`和`@register_to_aten`，可用lint规则检查orphan函数

### D-751: as_strided负stride校验位置在bounds check之后导致误报

- Hashes: `8b6a21a64` `173b7e8dd` `fb919407d` `cf4aa50ee` `3fc72466f`
- Files: `torch_npu/csrc/aten/common/ResizeNpu.h`
- Root cause: 校验顺序错误
- Defect: `setStrided`函数中，负stride的TORCH_CHECK放在`checkInBoundsForStorage`和`set_sizes_and_strides`之后。当传入负stride时，先执行bounds check可能产生误导性的错误信息，或者先执行了size/stride设置再检查合法性。与upstream的`as_strided`实现对比，校验应在所有mutation之前。
- Fix: 将负stride检查移到`TORCH_CHECK(size.size() == stride.size())`之后、`checkInBoundsForStorage`之前
- Reviewability: high - 校验顺序是review的基本检查项，先validate后mutate
- Review rule: 参数校验必须在任何状态修改之前完成; 对比upstream同名函数的校验顺序

### D-752: setup.py打包遗漏ascend_npu_ir C++源文件 + profiler头文件误提交

- Hashes: `ca0081639`
- Files: `setup.py` `test/npu/test_public_bindings.py` `third_party/acl/inc/experiment/msprof/toolchain/prof_api.h`
- Root cause: 打包配置不完整 + 文件误提交
- Defect: 1)setup.py的`get_src_py_and_dst`未包含`ascend_npu_ir`的C++源文件(*.cpp, *.h, cpp_common/*)，导致安装后缺少编译所需文件; 2)同时删除了误提交的profiler头文件`prof_api.h`(应属于CANN SDK，不应在torch_npu仓库中); 3)public bindings测试缺少`build_ext`模块
- Fix: 1)添加anir_files glob pattern到打包列表; 2)删除prof_api.h; 3)添加build_ext到public bindings allowlist
- Reviewability: medium - 打包配置需要了解产物结构; 第三方头文件出现在仓库中是可review的
- Review rule: setup.py修改时应验证`pip install -e .`后所有import路径可用; 第三方SDK头文件不应入库(使用构建时查找)

### D-753: zero copy场景SetDevice仅在rank不匹配时调用导致设备未初始化

- Hashes: `b8416a6d0` `7b8a4dcd1` `9a2878f25` `c7ea32e79` `81ee1f4ab` `270878a56` `7f6c48319` `a1ab12002` `a4b4cb778`
- Files: `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp`
- Root cause: SetDevice条件分支遗漏
- Defect: `ProcessGroupHCCL`构造函数的zero copy路径中，`SetDevice`只在`device_id != local_rank`的if分支内调用。当device_id已等于local_rank时跳过SetDevice，但zero copy模式下后续的`createHCCLCommForZeroCopy`和`buildServerMemMapForHccl`依赖设备已正确初始化。SetDevice不仅是设置当前设备，还有初始化副作用。
- Fix: 将`SetDevice`移到if分支外，无条件调用。device_id赋值保留在if内
- Reviewability: high - if分支内的初始化调用应引起review警觉: "else路径是否也需要?"
- Review rule: 设备初始化API(SetDevice)即使看似冗余也应无条件调用，因为可能有除设置设备号之外的副作用(context初始化、资源分配)

### D-754: mem_get_info的_get_device_index缺少optional=True导致非标准输入失败

- Hashes: `663b21efa` `bf45c8945` `7f1b28459` `a145e5118` `39f439d19`
- Files: `torch_npu/npu/__init__.py`
- Root cause: API参数缺省值不匹配
- Defect: `mem_get_info`调用`_get_device_index(device)`时未传`optional=True`。PyTorch的`_get_device_index`默认`optional=False`，要求device参数是明确的整数或torch.device对象。当用户传入字符串如"npu:0"或None(已在上面处理)时，非optional模式会抛出异常。同文件中`get_device_properties`等函数已正确传了`optional=True`。
- Fix: 添加`optional=True`参数
- Reviewability: high - 同文件其他调用点已有optional=True，不一致是机械检查可发现的
- Review rule: 同一文件内对同一API的调用，参数风格应保持一致; `_get_device_index`在设备管理函数中应始终使用`optional=True`

### D-755: HCCL线程core num传播引入bug后被revert(函数调用缺少括号)

- Hashes: `344193ced` `68c4ec1ec` `4c8e85d69` `e1941792a` `f8bbb1272` `f0dd33414` `dbf85ebb2` `0bba9eaa5` `3a2ad3b24` `81ffe272e` `46a64be5f` `c167f69aa` `e316d194e` `7dee20890` `bb00700e2` `a2899f601`
- Files: `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp` `torch_npu/csrc/distributed/ProcessGroupHCCL.hpp`
- Root cause: 函数调用语法错误 + 功能验证不充分导致revert
- Defect: fix commit在collective/collectiveCoalesced中添加core num(AIC/AIV)从当前NPU stream传播到HCCL stream的逻辑。但allreduce路径中写成`if (c10_npu::is_core_control_enabled)`(检查函数指针，始终为true)而非`if (c10_npu::is_core_control_enabled())`(调用函数)。同一PR的collective路径写法正确带了括号。修复被完整revert。
- Fix: revert全部core num传播代码。正确做法应修正allreduce中的函数调用语法
- Reviewability: high - 函数名作为条件判断不带括号是经典C++ review检查项; 同一PR内不同路径的写法不一致也应发现
- Review rule: 1)条件表达式中的函数调用必须带`()`，lint可检测; 2)同一feature在多个code path实现时应检查一致性; 3)大面积功能添加须有对应测试覆盖，否则revert风险高

### D-756: aclrtMallocHostWithCfg缺少SoC版本上界检查(Ascend910_95)

- Hashes: `6f79c0561` `4e5208f3e` `f12941356` `0a2ed18e2` `67df0b3ea`
- Files: `torch_npu/csrc/core/npu/interface/AclInterface.cpp`
- Root cause: API版本守卫缺少上界
- Defect: `AclrtMallocHostWithCfgExist`仅检查`GetSocVersion() >= Ascend910B1`，Ascend910_95虽然版本号更高但不支持此API。版本检查只有下界没有上界，导致新硬件错误地启用了不支持的功能。这与D-743(allocator padding)属于同一类问题: 新硬件的版本号在旧范围内但行为不同。
- Fix: 添加`&& GetSocVersion() < Ascend910_95`上界检查
- Reviewability: high - 版本范围检查只有下界没有上界是常见review发现点
- Review rule: SoC版本守卫必须同时指定上下界(closed range)，或使用白名单模式。新硬件引入时必须全局搜索所有版本比较代码

### D-757: profiler DB解析缺少early exit导致线性扫描超时

- Hashes: `35504124f` `68258aab5` `172313789` `678efd810` `5212b988c`
- Files: `torch_npu/profiler/analysis/prof_view/prof_db_parse/_fwk_api_db_parser.py`
- Root cause: 有序数据扫描缺少提前终止条件
- Defect: `FwkApiDbParser`中三个扫描函数(dequeue匹配、torch_op关联、node_launch关联)对有序时间戳数据做线性扫描时，当scan指针的`start_ns`已超过目标的`end_ns`时未break，导致无意义地扫描剩余全部数据。对大规模profiling数据，这导致解析耗时从O(n)退化到O(n^2)。
- Fix: 在三处循环中添加`if end_ns < start_ns: last_index = idx; break`提前终止
- Reviewability: high - 有序数据的线性扫描缺少early exit是标准的性能review检查项
- Review rule: 对有序数据的查找循环，必须有基于排序性质的early exit条件

### D-758: inductor导入时thread pool预热时机错误导致进程挂起

- Hashes: `4f614caa1` `bdec384b6` `3cf5ee027` `6ddaf5ad8` `d09f482c7` `5b8aba051` `bc9fafbb3` `598a78574` `7c11912e1` `daa38df2b`
- Files: `torch_npu/__init__.py` `torch_npu/_inductor/__init__.py` `torch_npu/utils/_dynamo.py` `test/_inductor/test_argmax.py`
- Root cause: 模块导入副作用与线程池初始化顺序冲突
- Defect: PyTorch的`AsyncCompile.warm_pool()`在import时被触发，尝试启动worker进程，但worker进程又会import torch_npu形成循环依赖。此外thread pool的prewarming在某些场景下会阻塞主进程。两个PR用不同路径修复: 一个在`_inductor/__init__.py`中显式控制warm_pool时机并设`TORCH_WARM_POOL=0`; 另一个在`_dynamo.py`的`_InductorNpuRegistry.register_inductor_npu`中调用。
- Fix: 1)设置`TORCH_WARM_POOL=0`阻止自动预热; 2)在inductor注册时显式调用`AsyncCompile.warm_pool()`; 3)warm_pool调用前临时禁用AUTOLOAD避免循环依赖
- Reviewability: low - 模块导入顺序和进程fork的交互需要运行时才能暴露
- Review rule: 1)模块顶层代码不应启动子进程或线程池; 2)设置环境变量的side effect应集中在一处并有注释说明原因

### D-759: FSDP patch缺少.default dispatch变体注册

- Hashes: `401dfddfb` `bf13a531f` `d547bcfcb`
- Files: `torch_npu/distributed/fsdp/_add_fsdp_patch.py`
- Root cause: PyTorch op dispatch registry的.default变体遗漏
- Defect: NPU对`torch.ops.fsdp.all_gather_copy_in`做了monkey-patch替换，但只注册了base名，未注册`.default`变体。PyTorch的dispatch机制在某些调用路径下查找`<op>.default`而非基名，导致patch失效回落到CUDA实现。
- Fix: 添加`torch.ops.fsdp.all_gather_copy_in.default = _patched_all_gather_copy_in`
- Reviewability: medium - 需要了解PyTorch op dispatch的.default convention
- Review rule: 对torch.ops做monkey-patch时，必须同时patch base名和.default变体。可用grep检查所有类似patch点

### D-760: inductor persistent reduction单轴时mask集合包含了非reduction轴

- Hashes: `4a63d3f26` `dce775286` `dd84d7f26`
- Files: `torch_npu/_inductor/codegen/triton.py` `test/_inductor/test_issue59.py` `test/_inductor/test_var_mean.py`
- Root cause: mask过滤条件缺少reduction轴判断
- Defect: `NPUIndexTritonKernel.scan`中生成mask集合时，对persistent reduction且只有单个reduction轴的情况，mask应只包含非reduction轴的mask。原代码无条件包含所有`sorted_axis`的mask，导致codegen生成多余的mask计算，触发编译错误。
- Fix: 当`persistent_reduction and numof_reduction_axis() == 1`时，过滤掉`name[0] == "r"`的轴
- Reviewability: medium - 需要理解triton codegen的mask语义和reduction轴约定
- Review rule: inductor codegen中mask集合的构造必须区分reduction轴和非reduction轴; 被`@skip`的测试须有对应issue跟踪

### D-761: transfer_to_npu用try/except吞掉jit.script所有错误

- Hashes: `e52a8a79d` `3b1e87223` `281b1b32e` `dcecc4223` `9b8d3dde4`
- Files: `torch_npu/contrib/transfer_to_npu.py` `torch_npu/utils/_dynamo.py`
- Root cause: 过度防御的异常处理吞掉有效错误
- Defect: `_jit_script`包装函数用`try/except Exception`捕获`_real_jit_script`的所有异常并静默fallback到返回原始对象。这掩盖了真实的jit编译错误，使用户无法区分"不支持jit"和"jit代码有bug"。任何jit.script失败(包括用户代码bug)都被静默忽略。
- Fix: 引入`use_jit_script`标志(默认False)，用显式标志控制是否调用jit.script，而非用try/except兜底。成功时打印提示，失败时让异常自然传播
- Reviewability: high - blanket try/except Exception是经典code smell
- Review rule: 1)禁止`except Exception`吞掉所有错误并fallback到不同行为; 2)功能开关应用显式flag而非异常控制流

### D-762: Revert profiler "memory optimaze"(解析器依赖链和执行顺序错误)

- Hashes: `9cad09cc8` `5e2b786a2` `3c7d45683` `78fdc82ca` `de58dc6d4` `d7f322ccb` `79d51c855` `91493c027`
- Files: `torch_npu/profiler/analysis/prof_common_func/_file_manager.py` `torch_npu/profiler/analysis/prof_config/_parser_config.py` `torch_npu/profiler/analysis/prof_config/_parser_deps_config.py` `torch_npu/profiler/analysis/prof_parse/_cann_file_parser.py` `torch_npu/profiler/analysis/prof_parse/_fwk_cann_relation_parser.py` `torch_npu/profiler/analysis/prof_view/_stack_view_parser.py` `torch_npu/profiler/analysis/prof_view/_trace_step_time_parser.py` `test/profiler/analysis/prof_common_func/test_file_manager.py`
- Root cause: 并发解析器依赖链修改引入数据竞争/顺序错误
- Defect: "memory optimaze"commit修改了profiler解析器的执行顺序和依赖配置(移除TracePreParser依赖、修改ONLY_FWK管线、改变append_trace_json_by_path API)。这破坏了解析器间的数据依赖: TraceViewParser依赖的数据在新顺序下尚未准备好; RelationParser被移除导致OperatorViewParser缺少关联数据; append API变更导致JSON输出格式错误。
- Fix: 完整revert。恢复原始解析器顺序、依赖配置和文件管理API
- Reviewability: medium - 解析器依赖链修改需要理解数据流全貌; 但API签名变更(移除参数)应在review中引起警觉
- Review rule: 1)修改并发解析器的执行顺序或依赖关系时，须画出完整的数据流DAG并验证拓扑排序正确; 2)变更公共API签名须同时更新所有调用点和测试

<!-- D-skip: 077d05f25 - "Fix codecheck for test_npu/c10d" 纯代码风格修复(import排序/缩进/elif->if) -->
<!-- D-skip: 078cda284 - "Fix minial version desp for autoloading" README文档版本描述修正 -->
<!-- D-skip: 07bbcc195 - "add torch.npu.get_device_capability, torch.npu.aclnn.version" 新增API桩实现(返回None), 非defect -->
<!-- D-skip: 083ffb938 f3fa9d44f - "Fix codecheck" 纯缩进/格式化修复 -->
<!-- D-skip: 0968bc3e1 2e676bbb5 3c5c0a09e - "change jit_compile default value from ACL" 新增从ACL读取jit_compile默认值的功能, 非defect -->
<!-- D-skip: 0b66ea43e 26e5a7707 5888baa9d 94c03dc9e a934fdb06 c8a00c45c e8e435d5b - "fix with aclshmem" 新增aclshmem API头文件/结构体定义, 接口演进非defect -->
<!-- D-skip: 0bef95fd1 304a5f5fb 582f4262a - "fix:several docs issues" API支持表文档修正 -->

### D-763: 乘法运算byte/bool输入时scalar type promotion错误

- Hashes: `07ceee671`
- Files: `torch_npu/csrc/aten/ops/MulKernelNpu.cpp` `test/test_network_ops/test_mul.py` `test/test_network_ops/test_rsub.py`
- Root cause: scalar type promotion未处理非浮点类型
- Defect: `muls_out_npu_nocheck`中`binary_op_check`对scalar乘法做类型提升时，当scalar是非浮点类型(如int)，unified_result.common_type没有保留tensor自身的scalar_type，导致byte/int8/bool等整型tensor与标量运算后类型错误。另外rsub(1 - byte_tensor)路径也受影响。
- Fix: 当scalar非浮点时，显式设置`unified_result.common_type = self.scalar_type()`; bool tensor特殊处理使用scalar的type。同时将`ApplyTensorWithFormat`替换为`ApplyTensor`避免不必要的format推断
- Reviewability: medium - 需理解NPU binary op的type promotion规则
- Review rule: 非浮点类型(byte/bool/int8)的scalar运算路径须有独立测试用例; type promotion逻辑变更须覆盖所有整型dtype组合

### D-764: 二元算子对跨设备scalar tensor的检测条件不准确

- Hashes: `093a48066` `3ca9e4f24` `a52213d92` `ce1df092e`
- Files: `torch_npu/csrc/aten/ops/MaxKernelNpu.cpp` `torch_npu/csrc/aten/ops/AddKernelNpu.cpp` `torch_npu/csrc/aten/ops/MulKernelNpu.cpp` `torch_npu/csrc/aten/ops/SubKernelNpu.cpp` `torch_npu/csrc/aten/ops/DivKernelNpu.cpp` `torch_npu/csrc/aten/ops/BitwiseAndKernelNpu.cpp` `torch_npu/csrc/aten/ops/BitwiseOrKernelNpu.cpp` `torch_npu/csrc/aten/ops/BitwiseXorKernelNpu.cpp` `torch_npu/csrc/aten/ops/GeKernelNpu.cpp` `torch_npu/csrc/aten/ops/FillKernelNpu.cpp` `torch_npu/csrc/aten/ops/MaskedFillKernelNpu.cpp`
- Root cause: CPU scalar tensor判断条件遗漏edge case
- Defect: 多个二元算子(add/sub/mul/div/max/bitwise/ge/fill/masked_fill)使用`other.dim() == 0 && !at_npu::key::isDeviceTensor(other)`检测CPU标量tensor。这个组合条件对某些场景不准确(如0维NPU tensor被误判)。`maximum`函数对跨设备(CPU scalar + NPU tensor)输入还缺少scalar路径，直接走tensor路径导致失败。
- Fix: 统一替换为`OpPreparation::IsCPUScalar(other)`; `maximum`增加`IsCPUScalar`分支走scalar路径直接传递`other.item()`
- Reviewability: high - 分散在11个文件的相同pattern, 用grep全局搜索`dim() == 0`可一次性发现全部
- Review rule: 1)NPU op中判断"CPU标量"须使用统一的IsCPUScalar工具函数，禁止手写dim+device组合判断; 2)新增op时检查是否覆盖了CPU scalar tensor输入路径

### D-765: skipIfUnsupportMultiNPU的SkipTest异常未在多进程测试中正确传播

- Hashes: `0962e6e4c` `48ba1315c` `709a6f2bd` `df2a02dcf` `f7b97d633`
- Files: `test/distributed/test_distributed.py` `test/npu/test_stream_check.py` `torch_npu/testing/common_distributed.py`
- Root cause: 多进程测试框架未捕获SkipTest异常类型
- Defect: `MultiProcessTestCase`的worker进程中，`skipIfUnsupportMultiNPU`装饰器抛出`unittest.SkipTest`异常，但worker的try/except只捕获通用Exception，SkipTest(继承自Exception)被捕获后作为测试失败处理而非测试跳过。主进程无法区分"测试失败"和"设备不足跳过"，导致CI误报。
- Fix: 在通用Exception捕获之前添加`except unittest.SkipTest: sys.exit(TEST_SKIPS["multi-npu"].exit_code)`，让skip信号通过exit code正确传播
- Reviewability: high - 多进程测试框架的异常处理是标准review检查项
- Review rule: 多进程测试worker中必须显式处理SkipTest异常并用约定的exit code传播; 新增skip装饰器时须验证多进程场景下的传播行为

### D-766: HOOKModule的__call__缺少完整的backward hook协议实现

- Hashes: `09ed1d1ec` `c378e36a4`
- Files: `torch_npu/hooks/module.py` `test/test_hooks/test_acc_cmp_hook_backwardhook.py`
- Root cause: 自定义Module.__call__未实现PyTorch完整的hook执行协议
- Defect: HOOKModule继承nn.Module但未覆写`__call__`，依赖基类实现。当forward返回不同类型(如条件分支返回tensor vs tuple)时，基类的backward hook机制对result类型推断失败。根本原因是HOOKModule需要控制hook执行顺序，但基类的`__call__`在某些版本中对full_backward_hooks和non_full_backward_hooks的处理方式与HOOKModule的hook注册方式不兼容。
- Fix: 完整重写`__call__`方法，显式实现: 1)forward_pre_hooks执行; 2)full_backward_hooks的BackwardHook.setup_input_hook/setup_output_hook; 3)forward执行; 4)forward_hooks执行; 5)non_full_backward_hooks通过grad_fn.register_hook注册
- Reviewability: medium - 需要理解PyTorch Module.__call__的完整hook协议
- Review rule: 自定义Module子类如果覆写了__call__，必须与PyTorch同版本基类的hook执行协议保持一致; 涉及backward hook的变更须有backward测试覆盖

### D-767: empty_like_npu在FakeTensor模式下读取随机npu_format值

- Hashes: `09fa492eb` `3dfc7ca39` `4f66d033e` `6dc9588a6` `999800822` `be3542363` `d36e6868e` `f0484f937`
- Files: `torch_npu/csrc/aten/common/TensorFactories.cpp` `test/npu/test_npu.py`
- Root cause: 未区分FakeTensor和真实NPUTensor的storage类型
- Defect: `empty_like_npu`通过`NPUBridge::GetNpuStorageImpl(self)->npu_desc_.npu_format_`获取源tensor的NPU format。当self是FakeTensor时，其storage不是NPUStorageImpl而是通用FakeTensorImpl，但代码仍然强制cast并读取npu_format_字段，得到未初始化的随机值。这导致创建的tensor format无效，后续op执行时崩溃。
- Fix: 用`typeid(*self.storage().unsafeGetStorageImpl()) != typeid(torch_npu::NPUStorageImpl)`检查storage实际类型，非NPUStorageImpl时回退到`ACL_FORMAT_ND`
- Reviewability: medium - 需要理解FakeTensor的storage代理机制
- Review rule: 1)访问NPU-specific的storage字段前必须验证storage的实际类型; 2)compile/FakeTensor模式下的NPU tensor factory须有专门测试

### D-768: get_device_name/get_device_properties参数类型与CUDA API不兼容

- Hashes: `0a0b9bb85` `679506d1e`
- Files: `torch_npu/npu/utils.py` `test/test_api/test_torch_npu.py`
- Root cause: API签名与PyTorch CUDA约定不一致
- Defect: `get_device_name(device_id: int)`和`get_device_properties(device_id: int)`要求传入int类型的device_id，但PyTorch CUDA对应API接受`Optional[Union[int, str, torch.device]]`。用户传入`"npu:0"`或`torch.device("npu:0")`时报类型错误，与CUDA行为不一致。
- Fix: 参数改为`device_name=None`，内部用`_get_device_index(device_name, optional=True)`统一处理各种输入类型
- Reviewability: high - API签名与CUDA对比即可发现
- Review rule: NPU的公开API签名必须与对应CUDA API参数类型一致; 新增API时须对照PyTorch CUDA同名函数的签名

### D-769: npu_transpose强制contiguous导致大tensor OOM

- Hashes: `0a10966ac` `d29d8ee92`
- Files: `torch_npu/csrc/aten/npu_native_functions.yaml` `torch_npu/csrc/aten/ops/ArgsortKernelNpu.cpp` `torch_npu/csrc/aten/ops/SortKernelNpu.cpp` `torch_npu/csrc/aten/ops/RollKernelNpu.cpp` 等
- Root cause: transpose op无条件强制contiguous输出
- Defect: `npu_transpose`和独立的`npu_transpose_to_contiguous`两个op合并前，transpose总是生成contiguous输出。对大tensor，transpose+contiguous的额外内存分配导致OOM。但某些调用点不需要contiguous输出(如作为临时中间结果)。
- Fix: 合并两个op为`npu_transpose(self, perm, require_contiguous=True)`，默认保持向后兼容(强制contiguous)，但调用点可显式传`false`跳过不必要的contiguous化。所有内部调用点(argsort/cummin/sort/roll等)更新为显式传`true`
- Reviewability: medium - 需要理解哪些调用点需要contiguous
- Review rule: 高内存消耗的op应提供option跳过不必要的内存分配; NPU自定义op的API变更须同步更新所有内部调用点

### D-770: profiler step时间范围使用host时间戳而非device时间戳

- Hashes: `0a27da6da` `ded03ec63` `e859b828a`
- Files: `torch_npu/profiler/analysis/prof_bean/node_info_bean.py` `torch_npu/profiler/analysis/prof_bean/torch_op_node.py` `torch_npu/profiler/analysis/prof_common_func/global_var.py` `torch_npu/profiler/analysis/prof_common_func/tree_builder.py` `torch_npu/profiler/analysis/prof_common_func/csv_headers.py` `torch_npu/profiler/analysis/prof_parse/cann_file_parser.py`
- Root cause: step范围计算使用了不正确的时间源
- Defect: `GlobalVar.init_step_range`使用`level1_node.start_time/end_time`(host端fwk层时间戳)作为ProfilerStep的时间范围。但step内的device kernel执行时间与host时间存在偏移，导致op_summary的timeline分析中step边界不准确。同时`OpSummaryBean`中硬编码"Task Start Time"与CANN输出的"Task Start Time(us)"列名不匹配。
- Fix: 1)NodeInfoBean不再接受acl_start_time，改为从kernel_list计算min_start/max_end; 2)TorchOpNode新增device_start/device_end属性; 3)TreeBuilder遍历时调用update_device_range; 4)GlobalVar使用device_start/device_end; 5)CsvHeaders统一管理列名常量
- Reviewability: medium - 需要理解profiler的host-device时间对齐模型
- Review rule: profiler中的时间戳须明确标注来源(host/device); step/op的时间范围计算须使用同一时间域的数据

### D-771: as_tensor对NPU tensor输入错误地默认为CPU设备

- Hashes: `0b5ebb435` `8a18825d2`
- Files: `scripts/codegen/templates/torch_funcs.py` `test/test_api/test_torch/test_tensors.py`
- Root cause: as_tensor的monkey-patch默认设备硬编码为cpu
- Defect: `_as_tensor`包装函数将默认设备硬编码为"cpu"。当输入是NPU tensor且未显式指定device时，`torch.as_tensor(npu_tensor)`会将数据搬到CPU再搬回NPU，而非直接返回同设备的视图。这违反了PyTorch的as_tensor语义: "如果数据已经是tensor，应尽量返回相同存储的视图"。
- Fix: 当`args[0]`是Tensor时，默认设备取`args[0].device`而非"cpu"
- Reviewability: high - 对照PyTorch as_tensor文档即可发现
- Review rule: monkey-patch PyTorch函数时，必须保持原始函数的设备推断语义; 涉及设备默认值的代码须覆盖"输入已在目标设备上"的测试用例

### D-772: ACL运行时结构体字段名更新导致编译失败

- Hashes: `0b799a34b`
- Files: `third_party/acl/inc/acl/acl_rt.h` `torch_npu/csrc/npu/Module.cpp` `test/test_npu/test_torch_npu.py`
- Root cause: 上游ACL头文件字段重命名未同步
- Defect: ACL的`aclrtUtilizationInfo`结构体字段从`cube/vector/aicpu/memory`重命名为`cubeUtilization/vectorUtilization/aicpuUtilization/memoryUtilization`。torch_npu代码仍使用旧字段名，导致编译失败或运行时数据错位。同时移除了一个依赖timing的不稳定测试用例`test_npu_get_utilization_runing`。
- Fix: 更新所有引用处为新字段名
- Reviewability: high - 编译即可发现
- Review rule: 更新third_party头文件时，必须全局搜索旧symbol并同步更新所有引用; 依赖设备实时利用率的测试天然不稳定，应标记为flaky或使用mock

### D-773: CPU dispatch缺少pinned memory allocator + reducer冗余H2D copy

- Hashes: `0b9161dec`
- Files: `torch_npu/csrc/aten/common/EmptyTensor.cpp` `torch_npu/csrc/distributed/reducer.cpp`
- Root cause: CPU tensor factory路径和分布式reducer的pinned memory交互问题
- Defect: 两个独立问题在一个commit中修复: 1)NPU环境下CPU tensor的empty/empty_strided走默认dispatch路径时，pinned_memory参数的处理与NPU的THNPUCachingHostAllocator不兼容。需要显式注册CPU dispatch实现来正确处理pinned memory; 2)reducer的`all_reduce_local_used_map`中做了复杂的pinned memory临时buffer拷贝(local_used_map_->pinned_tmp->local_used_map_dev_)来避免异步H2D的竞态，但在NPU环境下这个中间步骤是不必要的(NPU的copy_已经处理了同步问题)。
- Fix: 1)注册`aten::empty.memory_format`和`aten::empty_strided`的CPU dispatch实现; 2)简化reducer为直接`local_used_map_dev_.copy_(local_used_map_, true)`
- Reviewability: low - 需要理解跨dispatch层的allocator交互和NPU copy语义
- Review rule: 1)在NPU插件中override CPU dispatch路径须有明确注释说明原因; 2)分布式reducer的内存拷贝路径变更须验证多rank场景的正确性

### D-774: torch.normal对不同shape的mean和std未计算broadcast输出大小

- Hashes: `0b931ddb8` `965a23519` `adafe9332` `ea5ce0b28`
- Files: `torch_npu/csrc/aten/ops/NormalKernelNpu.cpp` `test/test_network_ops/test_normal.py`
- Root cause: 输出tensor大小直接使用mean的shape而非broadcast后的shape
- Defect: `NPUNativeFunctions::normal(mean, std, generator)`中`result = OpPreparation::ApplyTensor(mean)`直接用mean的shape创建输出tensor。当mean和std shape不同(如mean=[2,1,4,5], std=[2,3,4,5])时，输出应为broadcast后的shape[2,3,4,5]，但实际创建了[2,1,4,5]的tensor。后续`result.mul_(std).add_(mean)`在shape不匹配时产生错误结果或crash。
- Fix: 用`broadcast_ops_npu_output_size(mean, std)`计算正确的输出size，用`ApplyTensor(mean, output_size)`创建
- Reviewability: high - 对照PyTorch normal文档的broadcast规则即可发现
- Review rule: 接受多个tensor输入的op，输出size必须经过broadcast计算; 测试须覆盖输入shape不同的情况

### D-775: DataParallel的_get_stream不支持NPU设备

- Hashes: `0bb47aa17`
- Files: `torch_npu/utils/module.py`
- Root cause: PyTorch DataParallel内部函数硬编码CUDA stream
- Defect: `torch.nn.parallel._functions._get_stream`用于DataParallel多设备间数据拷贝时获取后台stream。原始实现硬编码使用`torch.cuda.Stream`，NPU设备上调用DataParallel时此函数创建CUDA stream失败。
- Fix: monkey-patch `_get_stream`为NPU版本，使用`torch.npu.Stream(device)`创建stream，保持相同的缓存逻辑(_streams全局列表按device_id索引)
- Reviewability: high - DataParallel在非CUDA设备上运行时立即暴露
- Review rule: 使用DataParallel/DistributedDataParallel的功能须验证NPU设备; PyTorch内部硬编码CUDA的函数须在torch_npu初始化时逐一patch

### D-776: inductor sum lowering对混合正负axis只处理了最后一个负值

- Hashes: `fbfffe65f` `7fa4ab558` `1de60d5c3`
- Files: `torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/npu/inductor_patch/lowering.py`
- Root cause: 负轴索引条件检查不完整
- Defect: `sum_`的lowering实现中，`if axis and axis[-1] < 0`只检查最后一个axis是否为负，且`axis = [ax + offset for ax in axis]`对所有axis无差别加offset。当axis混合包含正负值(如`[1, -2]`)时，正值axis也被错误偏移。
- Fix: `axis[-1] < 0` → `any(ax < 0 for ax in axis)`; `[ax + offset for ax in axis]` → `[ax + offset if ax < 0 else ax for ax in axis]`
- Reviewability: high - 负索引处理是tensor操作的基本约定，review时应检查对axis list中每个元素的独立判断
- Review rule: 处理axis/dim参数时，必须对list中每个元素独立检查负值，不能只检查边界元素; 负值偏移必须是条件性的

### D-777: transfer_to_npu无条件禁用torch.jit.script

- Hashes: `cc5635212` `2c7eedbf2` `441c1f628` `8e989ea2e`
- Files: `torch_npu/contrib/transfer_to_npu.py`
- Root cause: monkey-patch过于激进，完全替换为no-op
- Defect: `transfer_to_npu`将`torch.jit.script`替换为`def _jit_script(obj, ...): return obj`，即在任何场景下都跳过JIT编译直接返回原函数。这导致用户显式调用`torch.jit.script`时也无法得到编译后的ScriptModule，影响TorchScript推理路径的正确性和性能。关联issue: Ascend/pytorch#1498。
- Fix: 保存原始`_real_jit_script`，先尝试真正编译，仅在异常时fallback返回原对象，并发出一次性RuntimeWarning
- Reviewability: high - monkey-patch应遵循"最小侵入"原则，完全禁用核心功能在review中应被挑战
- Review rule: monkey-patch PyTorch核心功能时，必须保留try-original-first-then-fallback模式，不得直接替换为no-op; 涉及JIT编译路径的变更须验证TorchScript导出场景

### D-778: CachingHostAllocator中event->query()异常导致故障恢复coredump

- Hashes: `98b71a48e` `2b756f1cc` `1d91259b4` `7c03236ad` `c35dcc617` `6a9521bb2` `ea4bec8b0` `ece588474` `26b2aa069`
- Files: `torch_npu/csrc/core/npu/CachingHostAllocator.cpp`
- Root cause: event查询缺少异常保护
- Defect: `CachingHostAllocator`的`query_event`和`process_events`直接调用`event->query()`。在设备故障恢复场景下(如HCCS link failure后的reset)，底层ACL event可能处于无效状态，`query()`抛出异常。由于这两个函数在allocator的清理路径中被调用(析构/重用内存时)，未捕获的异常直接terminate进程。共3处相同pattern的query调用。
- Fix: 对每处`event->query()`包裹`try/catch(...)`，异常时视为event已完成(isEventCompleted=true)并记录ASCEND_LOGE，允许清理流程继续
- Reviewability: medium - 需要理解设备故障恢复场景下ACL对象的状态; 但"清理路径中的异常安全"是C++通用规则
- Review rule: 设备对象(event/stream/context)的查询操作在清理/析构路径中必须做异常保护; allocator的回收逻辑不应因单个event失败而abort整个进程

### D-779: PyTorch upstream模块路径迁移导致triton_heuristics ImportError

- Hashes: `e6319cfb9` `543a29369` `2b4e0a1b1`
- Files: `torch_npu/_inductor/__init__.py`
- Root cause: 上游模块路径变更未同步
- Defect: `torch._inductor.triton_heuristics.CachingAutotuner`在PyTorch上游重构后移动到`torch._inductor.runtime.triton_heuristics`。torch_npu中`_replace_benchmark_all_configs`仍使用旧路径导致`ModuleNotFoundError: No module named 'torch._inductor.triton_heuristics'`。关联issue: github.com/Ascend/pytorch#97。
- Fix: 更新import路径为`torch._inductor.runtime.triton_heuristics`
- Reviewability: high - 上游同步时自动化检测import path变化可发现
- Review rule: 对PyTorch内部模块的import应有upstream版本跟踪机制; 基于private API的monkey-patch须在CI中验证对应PyTorch版本的import可达性

### D-780: npu_fusion_attention图模式API签名与实际FA算子不同步

- Hashes: `49d228b23` `efd75bb60` `b27111fba` `631325fab` `fad558bf2` `f9fcb8e42` `dabe2e4b4` `f954d811f`
- Files: `torch_npu/_inductor/npu_fusion_attention_graph.py` `test/_inductor/test_npu_fusion_attention_graph.py`
- Root cause: 图模式FA定义与eager FA算子签名漂移
- Defect: `npu_graph.npu_fa`和`npu_graph.npu_fa_backward`的Library定义缺少`softmax_layout`和`sink`两个参数，与实际`npu_fusion_attention`算子接口不一致。同时backward的meta实现返回4个tensor，但实际backward需要返回5个(多了dsink)。导致图模式FA在特定参数组合下crash(原UT被标记@skip("skip for core dump"))。
- Fix: 1)前向/反向定义中添加`softmax_layout=""`和`sink=None`参数; 2)Meta实现返回元组从4元素扩展为5元素; 3)autograd Function中unpack 5个返回值
- Reviewability: high - Library定义应与eager算子保持严格同步，可通过schema比对自动化检测
- Review rule: 自定义op的图模式(Library定义+Meta实现)必须与eager模式的签名和返回值数量保持严格一致; op API变更时须同步更新所有dispatch key注册

### D-781: aclrtMemcpyAsyncWithCondition的D2H路径错误跳过同步 + API版本守卫缺失

- Hashes: `f8aa1d3ab` `db4040f29` `bb716b9a4` `6ab82720c` `f5c2dde96` `1eb1bdb7d` `e4d2f37fb` `ffa7b89eb`
- Files: `torch_npu/csrc/core/npu/CachingHostAllocator.cpp` `torch_npu/csrc/core/npu/interface/AclInterface.cpp`
- Root cause: 错误的优化假设 + 缺少运行时版本检查
- Defect: 两个相关问题: 1)`process_unregistered_mem_location_type`中，当`AclrtMemcpyAsyncWithCondition`可用且kind为D2H时，直接跳过synchronize并返回SUCCESS。这个优化假设"async copy with condition不需要同步"是错误的，对于非pinned的malloc分配的host内存，D2H异步拷贝完成前访问host buffer会读到脏数据。2)`AclrtPointerGetAttributesExist`使用dlsym探测API是否存在，但该API在CANN 8.5.0之前的版本中即使symbol存在也不能正确工作。
- Fix: 1)移除D2H跳过同步的逻辑; 2)添加`IsGteCANNVersion("8.5.0", "RUNTIME")`前置检查
- Reviewability: low(问题1)/high(问题2) - 异步拷贝的同步需求取决于host内存类型(pinned vs malloc)，不容易在review中确认; 但API版本守卫是标准模式
- Review rule: 异步内存拷贝的同步优化必须区分pinned memory和malloc memory; dlsym探测API存在性不等于API功能正确，须加入运行时版本守卫

### D-782: FakeTensor/meta tensor序列化时访问无效storage导致coredump

- Hashes: `bc3ab9849` `14acfe401` `daf71a8fc`
- Files: `torch_npu/utils/serialization.py`
- Root cause: 序列化路径未处理FakeTensor和meta tensor的特殊storage
- Defect: `_npu_save`中，对于非CPU设备的storage，直接调用`torch_npu._C._tensor_construct_from_storage(storage)`来获取tensor大小。FakeTensor的storage不是真实NPU storage(有`_fake_device`属性)，meta tensor的storage在`"meta"`设备上。两者都不支持`_tensor_construct_from_storage`，调用时产生coredump。此外，`skip_data`模式的检查在storage访问之后执行，即使skip_data=True也会先crash在storage操作上。
- Fix: 1)添加`is_fake`(检查`_fake_device`)和`is_meta`(检查`device=="meta"`)条件判断，对这些特殊tensor使用`obj._size()`获取大小; 2)将`skip_data`检查提前到storage访问之前
- Reviewability: medium - FakeTensor和meta tensor是torch.compile/export的核心概念，序列化路径应考虑; skip_data提前检查属于"短路优化应尽早执行"的通用规则
- Review rule: NPU storage操作(`_tensor_construct_from_storage`等)前必须验证storage是真实NPU storage; 旁路(skip_data)检查须在资源访问之前执行

### D-783: CustomAsyncCompile中return kernel缩进错误导致死代码

- Hashes: `b5aaa00d9` `9fd43a36b` `c0dd484c8` `7799ad7f9`
- Files: `torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/codecache.py`
- Root cause: Python缩进错误(return语句在错误的块内)
- Defect: `CustomAsyncCompile`方法中，`if len(kernel.launchers) == 0:`分支内先执行`return _load_fx_graph(...)`回退到FX图执行，紧接着是`return kernel`。由于两个return都在if块内，当`kernel.launchers`非空时(正常编译成功路径)，函数没有return语句，隐式返回None。后续使用编译结果时得到None而非kernel对象。
- Fix: 将`return kernel`的缩进从if块内(8空格)移到函数级(12空格→8空格)，使其成为if块之外的默认返回
- Reviewability: high - Python缩进敏感，review时应检查return语句的嵌套层级; lint工具可检测unreachable code
- Review rule: Python函数中的return语句须验证每个分支的可达性; dead code after return应被lint规则捕获

### D-784: torch_npu.synchronize被dynamo标记为SkipFunctionVariable导致图断裂

- Hashes: `357ebbb94` `a6b31344c` `ddd361768` `d894259ea`
- Files: `torch_npu/dynamo/trace_rule.py`
- Root cause: dynamo trace rule注册语义错误
- Defect: `torch_npu.npu.utils.synchronize`被注册为`SkipFunctionVariable`(通过`manual_torch_name_rule_map`)，意味着dynamo遇到synchronize时会产生graph break。对于编译模式下的代码，每次synchronize调用都会中断当前图并回退到eager执行，严重影响编译性能。正确语义应该是将synchronize作为in-graph节点。
- Fix: 将synchronize加入`torch_c_binding_in_graph_functions_npu`字典(TorchInGraphFunctionVariable)，删除manual_torch_name_rule_map中的SkipFunctionVariable注册
- Reviewability: high - trace_rule的语义(InGraph vs Skip)直接影响编译覆盖率，review时应验证每个函数的注册类型是否正确
- Review rule: dynamo trace rule中新增函数时，必须明确区分InGraph(可被编译)和Skip(产生graph break)的语义; synchronize/barrier等同步原语通常应为InGraph

### D-785: inductor LoopBody.__call__缺少allow_same_symbol_in_index参数

- Hashes: `706c556cd`
- Files: `torch_npu/_inductor/codegen/ir.py`
- Root cause: monkey-patch函数签名与上游不一致
- Defect: PyTorch上游为`LoopBody.__call__`添加了`allow_same_symbol_in_index`参数。torch_npu的monkey-patch `loopbody__call__`没有同步此参数，导致上游代码传入该参数时触发`NameError: name 'allow_same_symbol_in_index' is not defined`(表现为InductorError)。
- Fix: 在`loopbody__call__`签名中添加`allow_same_symbol_in_index=False`
- Reviewability: high - monkey-patch函数的签名应与被patch的原函数严格一致
- Review rule: monkey-patch的函数签名须与原函数保持同步，建议使用`*args, **kwargs`转发或在CI中对比签名

### D-786: DTensor操作硬编码4设备假设 + 测试基类适配

- Hashes: `c72f087f7` `89289f4b5` `be140503a`
- Files: `torch_npu/distributed/tensor/_attention.py` `torch_npu/distributed/tensor/_common.py` `torch_npu/distributed/tensor/_dtensor_patch.py` `torch_npu/distributed/tensor/_math_ops.py` `torch_npu/distributed/tensor/_matrix_ops.py` `torch_npu/testing/_internal/common_dtensor.py` `test/distributed/tensor/test_*.py`(多个)
- Root cause: DTensor实现和测试假设world_size=4
- Defect: DTensor的sharding操作、mesh构建、FA参数适配等多处硬编码假设world_size=4。在只有2个NPU设备的CI环境中，DTensorTestBase默认world_size=4导致分布式初始化失败。生产代码中`_attention.py`的head_num计算、`_matrix_ops.py`的placement策略等也依赖4设备假设。
- Fix: 1)创建`NPUDTensorTestBase`基类，world_size属性动态取min(4, device_count); 2)测试从DTensorTestBase迁移到NPUDTensorTestBase; 3)生产代码中用mesh.size()替代硬编码值; 4)添加@SupportedDevices和@skipIfUnsupportMultiNPU(2)装饰器
- Reviewability: high - 硬编码设备数是明显的code smell
- Review rule: 分布式代码不得硬编码world_size/device_count，必须通过mesh或process_group动态获取; 测试world_size应适应CI环境的实际设备数

<!-- D-skip: 5f8e5afe1 f0efeb6d4 9edd24949 91e80eda1 — "fix error": 实际是新增ShardedTensor NPU patch和测试(feature addition)，非bug fix -->
<!-- D-skip: cc4452eb1 2d709e649 b8665e3fb — "Fix UT world_size=2": 纯测试基础设施适配(添加dynamic world_size property) -->
<!-- D-skip: f5f5a4103 e50c73c83 75ee6309a 962e9341d — "fix send/recv to graph model": 仅添加public API allowlist条目 -->
<!-- D-skip: f78a4af3d f1a8ebd87 cf130e34a 9904b02d2 3d26770f5 — "fix docs": 纯文档修正 -->
<!-- D-skip: 1cfa301c5 — "fix:seceral docs issues": 纯文档修正(API支持状态表更新) -->
<!-- D-skip: a5bc853df 1e5d07c63 e6c573751 858f2a44e a40b5b7cb 45352669d b4a3eb18d 97b173366 572453bec — "Fix UT from test_register_sharding": 测试文件重组织(从test_register_sharding.py拆分到test_math_ops.py和test_matrix_ops.py)，无生产代码变更 -->
<!-- D-skip: fb2839ec6 — "fix apis and framework_feature_guide_pytorch": 文档/API列表更新 -->

### D-787: collective op sync/async stream选择逻辑引入竞态 — 被revert

- Hashes: `25ba98faf` `8020c856b`
- Files: `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp`
- Root cause: stream同步路径不完整导致数据竞争
- Defect: 为ProcessGroupHCCL::collective和collectiveCoalesced引入了asyncOp标志位区分异步/同步操作的stream选择: asyncOp=true使用预分配hcclStreams，asyncOp=false使用当前stream。问题在于syncOp路径跳过了syncStreams(设备间stream同步)和部分recordStream(tensor生命周期管理)，导致: 1)tensor可能在collective操作完成前被allocator释放; 2)不同stream间的执行顺序无法保证。revert后统一使用预分配hcclStreams并始终执行stream同步链。
- Fix: 删除asyncOp分支逻辑，统一使用`hcclStreams_[key]`和`syncStreams()`; 删除output recordStream的asyncOp守卫
- Reviewability: medium — stream同步链的完整性需要对HCCL stream语义有深入理解; 代码层面可发现syncOp路径跳过了关键同步步骤
- Review rule: ProcessGroupHCCL中任何绕过syncStreams/recordStream的路径都应被标记为高风险; 引入新的stream选择策略时必须附带tensor生命周期分析

### D-788: P2P连接数限制未区分SoC硬件代次

- Hashes: `cdc425d3d` `9b1ba6398` `bc097148c` `67c9b1a00` `2e3b9d167` `c4a63049a` `efa6ad424` `e6612cf04`
- Files: `torch_npu/csrc/core/npu/NPUPeerToPeerAccess.cpp`
- Root cause: 硬件限制条件未绑定SoC版本
- Defect: NpuP2pCtrl::get_p2p_access中的P2P连接数上限检查(C10_P2P_ACCESS_MAX_NPUS)对所有硬件型号统一生效。但Ascend910_9391(A3)及更新SoC已移除此硬件限制，导致A3上合法的P2P连接被错误拒绝(返回COPY_NOT_ALLOWED)。
- Fix: 将P2P连接限制逻辑包裹在`if (GetSocVersion() < Ascend910_9391)`条件内; A3及以上跳过限制检查直接进入aclrtEnablePeerAccess
- Reviewability: high — 硬件限制代码中未注明适用的SoC范围是明显的文档/代码缺陷
- Review rule: 任何与NPU硬件限制相关的条件检查必须附带SoC版本范围注释; 硬件代际升级时须review所有C10_P2P_*常量的适用性

### D-789: inductor编译选项monkey-patch缺失及返回类型不匹配upstream

- Hashes: `abb12bc68` `4d8a225ea` `353e57ffd` `b9d4e7b0f`
- Files: `torch_npu/_inductor/__init__.py` `torch_npu/_inductor/cpp_builder.py`
- Root cause: monkey-patch未创建 + upstream接口变更后返回类型不同步
- Defect: 两阶段缺陷。阶段1: torch_npu缺失对`cpp_builder._get_optimization_cflags`的monkey-patch，导致inductor编译C++ wrapper时使用的优化选项不包含NPU/A3特定标志(如缺少fno-finite-math-only等)。阶段2: upstream将_get_optimization_cflags返回类型从`list[str]`改为`tuple[list[str], list[str]]`(cflags+ldflags)，torch_npu的patch仍返回list，导致解包失败。
- Fix: 阶段1(abb12bc68): 在__init__.py中添加patch函数并注册; 阶段2(4d8a225ea): 将函数迁移到cpp_builder.py，修改返回值为`tuple[list, list]`，补充debug_compile/LTO等新逻辑
- Reviewability: high — 阶段1应在feature开发时同步创建; 阶段2是upstream API变更的标准追踪项
- Review rule: 每个monkey-patch函数须在文件头注释中标注对应的upstream函数签名和最后同步的PyTorch版本

### D-790: NPUGraph环境变量控制report_shape改为API参数

- Hashes: `c656e58cd` `1dcaddabf` `f7315c879` `874f2c20e` `2314a8253` `797953e40`
- Files: `torch_npu/csrc/core/npu/NPUGraph.cpp` `torch_npu/csrc/core/npu/NPUGraph.h` `torch_npu/csrc/framework/interface/EnvVariables.cpp` `torch_npu/csrc/npu/Graph.cpp` `torch_npu/npu/graphs.py`
- Root cause: 环境变量控制粒度不匹配API调用粒度
- Defect: ACLGRAPH_REPORT_SHAPE环境变量用于控制NPUGraph capture是否上报shape信息。但环境变量是进程级全局设置，而report_shape需求是per-capture级别。此外环境变量在apply_cache_op_info内部读取，调用者无法在每次capture_begin时灵活控制。
- Fix: 1)删除ACLGRAPH_REPORT_SHAPE环境变量注册; 2)为capture_begin添加report_shape参数(默认true); 3)apply_cache_op_info直接使用传入的enabled参数; 4)Python层NPUGraph.capture_begin传递report_shape=True
- Reviewability: medium — 环境变量用于细粒度控制是设计问题，代码review可发现但需要理解使用场景
- Review rule: 需要per-call控制的行为不应使用环境变量; 环境变量仅用于进程级全局配置

### D-791: 版本检查函数对旧CANN抛fatal error而非优雅降级

- Hashes: `edfcfcf76` `3a9f8bb9d` `1f2b8815d` `4fdeb0a91` `334321b90` `e72cf7234` `cd4a1b156` `9781f9c89` `d317b1543`
- Files: `torch_npu/csrc/core/npu/GetCANNInfo.cpp`
- Root cause: 版本检查函数的错误处理粒度不匹配调用场景
- Defect: IsGteDriverVersion在CANN版本低于8.1.RC1时直接TORCH_CHECK(false)抛出异常终止进程。但该函数的调用方期望的是true/false返回值来做条件分支(如P2P限制检查D-788)。调用方无法在try-catch中优雅处理，因为TORCH_CHECK是fatal级别。同时IsGteCANNVersion的错误消息使用模糊的"this function"，不利于定位。
- Fix: IsGteDriverVersion中将TORCH_CHECK改为TORCH_NPU_WARN_ONCE+return false; IsGteCANNVersion中"this function"改为具体函数名
- Reviewability: high — bool返回函数内部使用fatal error是明显的API设计矛盾
- Review rule: 返回bool的查询函数不应抛出fatal异常; 对不满足前置条件的情况应warn+返回保守值

### D-792: shutdown序列中CachingHostAllocator record_stream触发use-after-destroy

- Hashes: `c1bc30c27` `e1c2702f6` `c71310d65` `4c52cd49b` `3b16abab2` `633a7a860`
- Files: `torch_npu/csrc/InitNpuBindings.cpp` `torch_npu/csrc/core/npu/CachingHostAllocator.cpp` `torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp` `torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h` `torch_npu/csrc/libs/init_npu.cpp`
- Root cause: shutdown时组件析构顺序与运行时调用依赖不一致
- Defect: THPModule_npu_shutdown和finalize_npu中，CachingHostAllocator_emptyCache在释放host block时会调用record_stream创建event并录制到stream上。但在shutdown阶段，stream和device context可能已部分销毁，record_stream中的event->record(stream)触发ACL runtime错误或coredump。这与D-778(event->query在device reset后失败)是同一类pattern:清理路径中访问了已失效的device资源。
- Fix: 1)NpuSysCtrl新增host_finalize_flag_标志; 2)在emptyCache之前调用HostFinalize()设置标志; 3)CachingHostAllocator::record_stream检查该标志，若host正在finalize则直接return跳过event录制; 注释明确说明不需要mutex因为shutdown是单线程的
- Reviewability: medium — 需要理解shutdown序列中各组件的销毁顺序; 但"在析构路径中调用可能失败的runtime API"本身是可review的pattern
- Review rule: shutdown/finalize路径中的所有ACL runtime调用(event/stream/device)都需要"已销毁"守卫; 析构路径不应假设device context仍有效

### D-793: npu_fusion_attention DTensor分片策略缺失导致分布式场景报错

- Hashes: `aba8ee7c0` `eca259dfe` `532f7762f` `c79d6def4` `e51a429b6` `cb005597f` `e944fcbdb`
- Files: `torch_npu/distributed/tensor/_attention.py` `torch_npu/distributed/tensor/_dtensor_patch.py` `torch_npu/distributed/tensor/_matrix_ops.py` `test/distributed/tensor/test_attention_ops.py`
- Root cause: 自定义算子缺少DTensor sharding规则注册
- Defect: npu_fusion_attention作为自定义算子，缺少DTensor的OpStrategy注册。当输入为DTensor(分布式tensor)时，框架不知道如何对FA的多个输出(attention_out, softmax_max, softmax_sum等)进行分片推导，也不知道如何处理atten_mask在不同sparse_mode下的分片约束。此前的_matrix_ops.py中有部分实现但缺少对BNSD/BSND/BSH等不同input_layout的完整支持。
- Fix: 重写_attention.py中的分片策略，为BNSD/BSND/BSH三种layout注册完整的forward+backward OpStrategy; 新增_dtensor_patch.py处理FA的DTensor dispatch; 添加覆盖多种sparse_mode和placement组合的测试; 从_matrix_ops.py中移除旧的FA分片逻辑
- Reviewability: medium — 需要理解DTensor的sharding propagation机制和FA的layout语义
- Review rule: 自定义算子若需支持DTensor必须注册完整的OpStrategy; FA类算子须覆盖所有支持的input_layout

### D-794: HCCL头文件struct定义与CANN库版本不同步

- Hashes: `8824d6955` `572dd5e81` `fe1833546` `d52f1833b`
- Files: `third_party/hccl/inc/hccl/hccl.h` `third_party/hccl/inc/hccl/hccl_types.h`
- Root cause: vendored头文件未随CANN版本同步更新
- Defect: HcclCommConfig结构体在CANN侧新增了hcclExecTimeOut、hcclAlgo、hcclRetryEnable、hcclRetryParams等字段，config版本从8升至9。torch_npu的vendored头文件仍是版本8，缺少这些字段。HcclCommConfigInit也未初始化新字段。struct布局不一致会导致ABI不兼容:torch_npu编译的binary用旧struct layout访问CANN库返回的新struct，读写越界或字段错位。
- Fix: 更新vendored hccl_types.h添加新字段定义和常量; 更新hccl.h中HcclCommConfigInit初始化新字段; HCCL_COMM_CONFIG_VERSION从8改为9; BUFFER_NAME枚举值顺延
- Reviewability: high — vendored header版本号(CONFIG_VERSION)不匹配是直接可检测的
- Review rule: third_party/hccl头文件必须与目标CANN版本严格匹配; CI应检查CONFIG_VERSION与链接库的一致性

### D-795: C/C++编译缺失标准头文件或前向声明导致编译失败

- Hashes: `5cc8eafa0` `2da3c5d01` `69077215c` `fe123a372` `7618e05ca` `b9f914b9e`
- Files: `third_party/acl/inc/ge/ge_error_codes.h` `torch_npu/csrc/core/npu/interface/AclInterface.h` `torch_npu/csrc/core/npu/NPUException.h`
- Root cause: 头文件包含依赖不完整
- Defect: 三个独立的编译缺陷: 1)ge_error_codes.h使用uint32_t等类型但未include <stdint.h>，在某些编译器/平台配置下因隐式include链断裂而报错(issue#1445); 2)AclInterface.h使用aclrtMemUsageInfo和aclOpExecutor类型但缺少前向声明，依赖include顺序隐式引入; 3)NPUException.h使用memset/memcpy等函数但未include <cstring>。这些缺陷在特定编译顺序或独立编译单元中才会暴露。
- Fix: 1)在ge_error_codes.h添加`#include <stdint.h>`; 2)在AclInterface.h添加`typedef struct aclrtMemUsageInfo aclrtMemUsageInfo`和`aclOpExecutor`前向声明; 3)在NPUException.h添加`#include <cstring>`
- Reviewability: high — 静态分析工具(include-what-you-use)可自动检测
- Review rule: 每个头文件必须自包含(self-contained)，不依赖include顺序; CI应启用-Werror=implicit-function-declaration

### D-796: ASCEND_LOGE格式化字符串%s传入std::string导致未定义行为

- Hashes: `b64ce8479` `d7aded94c` `c05e5bf54` `d717ca8ad` `5da928113` `d38d1c540` `c9671953b` `4721dea7b`
- Files: `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp`
- Root cause: printf风格API的类型安全漏洞
- Defect: checkHcclComms中ASCEND_LOGE的%s格式化占位符直接传入std::string变量`name`。printf系列函数的%s期望const char*指针，传入std::string对象会读取对象内存布局的前几个字节作为地址，导致: 1)打印乱码或空串; 2)潜在的segfault(读取无效内存)。这是C/C++混用中的经典陷阱。
- Fix: `name`改为`name.c_str()`
- Reviewability: high — 编译器-Wformat可检测此类型不匹配; 代码review中printf参数与std::string的组合应被标记
- Review rule: 所有ASCEND_LOG*/printf风格调用中，std::string参数必须使用.c_str(); 建议项目级启用-Wformat-security

### D-797: profiler stop未检查启动状态导致空指针异常

- Hashes: `fca0cd7ad` `9336717ab` `f9fdb1aff` `0fc22c75a`
- Files: `torch_npu/csrc/profiler/npu_profiler.cpp` `torch_npu/profiler/analysis/prof_common_func/_path_manager.py`
- Root cause: API前置条件未校验
- Defect: stopNpuProfiler在profiler未启动(profilerEnabled()=false)时直接执行_pop操作，获取空state后作为NpuProfilerThreadLocalState*解引用导致空指针异常。同时ProfilerPathManager.get_realpath接收空字符串路径后传给os.path.expanduser，最终在后续的os.path操作中产生不可预期的行为。
- Fix: 1)stopNpuProfiler添加profilerEnabled()前置检查，未启动时ASCEND_LOGE并return; 2)get_realpath添加空路径检查，抛出明确的RuntimeError
- Reviewability: high — 公共API入口必须检查前置条件是基本规范
- Review rule: start/stop配对API的stop方法必须检查当前状态; 路径参数必须非空校验

### D-798: P2P通信在子group中使用local rank而非global rank

- Hashes: `a27b653b1` `f06512f76` `132d28068` `d1610913a` `3dd9b3adc` `201dc9a1c` `72fff6022` `dd2ca86e9` `4a2b6d7b1`
- Files: `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp`
- Root cause: rank映射逻辑未考虑子group场景
- Defect: createHCCLCommEx在创建P2P通信域时，p2pRanks直接使用group内的local rank(lowRank/highRank)作为HCCL的rank table条目。当ProcessGroup是全局group的子集(如通过new_group创建)时，local rank与global rank不同。HCCL底层的rank table需要global rank才能正确路由通信。使用local rank会导致通信发往错误的设备或初始化失败。
- Fix: 当options_->global_ranks_in_group非空时，用global_ranks_in_group[lowRank/highRank]映射到global rank; 添加highRank < group_size的边界检查
- Reviewability: high — local vs global rank是分布式通信的基础概念; new_group场景下的rank映射应在代码review中被重点检查
- Review rule: ProcessGroupHCCL中所有传递给HCCL API的rank值必须是global rank; 涉及子group的代码必须通过global_ranks_in_group做映射

<!-- D-skip: 89384951c — "资料PROF_CONFIG_PATH.md修改简介描述": 纯文档描述修正 -->
<!-- D-skip: 1beb488ad f82af319a bcf9c84c0 44620ee65 64180f487 5f5ea8d19 0db512708 9b8c5b436 694a74cce — "Fixed test_compatibility.py": 测试基础设施适配(为profiler.analyse API添加compatibility白名单) -->
<!-- D-skip: ce5e9bbaa — "fix test case in test_schedule_multiproc": 测试文件适配NPU(替换NCCL imports、适配dynamic world_size、添加model registry classes) -->
<!-- D-skip: 247be491d d5718a6be 62f59dbff c7ff45a33 aaedfff41 5cfc8d170 a1b7124e6 6927b61c4 — "revert api profiler_trace": 从schema.json移除prematurely暴露的profiler_trace API -->
<!-- D-skip: 6b134373a — "revert torchair commitid to 20251110": submodule版本回退(2.1分支不需要更新) -->
<!-- D-skip: 43d5b99eb 2d9fd3b6a 5dd80ebfe c7a504a9a a15a7867c e9052ad36 4a95417c8 cfec83e96 2a090d9b6 — "Fix process synchronization issues in distributed communication test cases": 纯测试修复，添加done_event防止子进程在父进程读取Queue前退出 -->
<!-- D-skip: 1618eff7c 3f842e4c8 898b7f502 10dfafc1f ba62134a8 — "[bug fix]Delete unnecessary function": schema.json移除函数 + torchair submodule更新 -->
<!-- D-skip: cd5ee8cb2 c8c5deead fb5de229b fe3c725ea 27e963f4b — "[bugfix]update torch_npu_schema.json": schema.json添加新API条目(op_never_timeout, profiler_trace) -->

### D-799: NPUGuardImpl缺少elapsedTime/queryStream/synchronizeStream实现

- Root cause: upstream接口未同步实现
- Hashes: `39ca660a5` `d78fcb44a` `31d6136fc` `d233178a9`
- Files: `torch_npu/csrc/core/npu/impl/NPUGuardImpl.cpp`, `torch_npu/csrc/core/npu/impl/NPUGuardImpl.h`
- Root cause: upstream DeviceGuardImplInterface新增虚方法未在NPU侧实现
- Defect: PyTorch的DeviceGuardImplInterface陆续添加了`queryStream`、`synchronizeStream`、`elapsedTime`三个virtual方法。NPUGuardImpl未override导致调用时走到基类的默认实现(抛异常或no-op)。此外`createEvent`的flag映射忽略了`BACKEND_DEFAULT`(enable_timing语义)，timing event无法正确创建。
- Fix: 实现三个方法(queryStream委托NPUStream::query, synchronizeStream委托NPUStream::synchronize, elapsedTime通过aclrtEventElapsedTime计算); createEvent中根据BACKEND_DEFAULT决定是否设置ACL_EVENT_TIME_LINE
- Reviewability: medium — 需要跟踪upstream接口变更; 但缺少override的虚方法可通过`-Wsuggest-override`编译器flag自动检测
- Review rule: upstream接口新增virtual方法时，NPUGuardImpl必须同步实现; CI应启用`-Wsuggest-override`

### D-800: getDeviceFromPtr始终返回当前设备而非指针实际所在设备

- Root cause: 多设备语义错误
- Hashes: `e047e1285` `705f4d90b` `9d1413769`
- Files: `torch_npu/csrc/core/npu/NPUHooksInterface.cpp`, `torch_npu/csrc/core/npu/interface/AclInterface.cpp`, `torch_npu/csrc/core/npu/interface/AclInterface.h`, `third_party/acl/inc/acl/acl_rt.h`
- Root cause: 实现时假设所有指针属于当前设备
- Defect: `getDeviceFromPtr`直接`return current_device()`，多卡场景下tensor的storage可能在非当前设备上。PyTorch框架通过此方法判断tensor归属设备，错误返回会导致stream/event在错误设备上操作。
- Fix: 通过`aclrtPointerGetAttributes`查询指针的实际设备location; 添加host指针检查(host内存不属于任何NPU设备)
- Reviewability: high — `return current_device()`明显是placeholder实现; 多设备场景review必须检查设备归属逻辑
- Review rule: 任何返回device index的函数必须基于实际硬件查询，不能返回"当前设备"作为默认值

### D-801: CSAN sanitizer未处理view op和factory function

- Root cause: 工具适配遗漏
- Hashes: `52880af8c` `2399d5b1f` `13cf547b9` `f03f8d66a`
- Files: `torch_npu/npu/_stream_check.py`, `test/test_npu_sanitizer.py`
- Root cause: CUDA sanitizer到NPU的移植不完整
- Defect: NPU的stream sanitizer(`_stream_check.py`)在处理view操作(如split)时错误地将其视为读写操作，导致false positive race报告。factory function(如empty)的输出tensor也未被正确标记为写入。
- Fix: 在parse_inputs/parse_outputs中检查is_factory标志和view语义; view op不产生dataptrs_read/written; 添加完整的UT覆盖(add/cat/split/inplace/out/nonzero)
- Reviewability: medium — 需要理解PyTorch的op schema语义(view vs compute vs factory)
- Review rule: 工具代码(sanitizer/profiler等)从CUDA移植到NPU时，必须验证所有op类型的行为

### D-802: TensorOptions自引用初始化(C++ UB)

- Root cause: C++未定义行为
- Hashes: `3b78acbcc` `5883d8a05` `4eed62202` `2612872ce` `2f2f4dc0b`
- Files: `torch_npu/csrc/distributed/reducer.cpp`
- Root cause: 变量在自身初始化表达式中被引用
- Defect: `at::TensorOptions options = options.dtype(at::kInt)` — RHS的`options`引用了正在构造的LHS对象。C++标准规定此时对象尚未完成初始化，读取其值是UB。实际行为依赖编译器和优化级别:可能读到零/垃圾值/正确值。reducer.cpp中3处相同pattern。
- Fix: 拆分为声明+赋值: `at::TensorOptions options; options = options.dtype(at::kInt);`
- Reviewability: high — `X x = x.method()`是经典C++ UB pattern; `-Winit-self`或`-Wuninitialized`可自动检测
- Review rule: 禁止在变量初始化表达式中引用自身; CI编译必须启用`-Winit-self`

### D-803: P2P通信域名称跨group冲突 + 子group条件限制错误

- Root cause: 分布式通信:命名冲突 + 条件守卫错误
- Hashes: `eb7844b92` `f20aa3d8e` `7ee173d02` `cbdc376ed` `d000588fd`
- Files: `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp`
- Root cause: P2P路径缺少group隔离 + 条件守卫过度限制
- Defect: (a) `createHCCLCommEx`中P2P路径被`options_->global_ranks_in_group.empty()`守卫，导致通过`new_group`创建的子group无法使用P2P通信。(b) p2pName格式为`p2p_lowRank_highRank`，不同group中的相同rank对会产生相同名称，HCCL底层通过name查找comm时会返回错误的通信域。
- Fix: 移除`global_ranks_in_group.empty()`条件; p2pName前缀加入`group_id`确保唯一
- Reviewability: high — 子group场景是分布式通信的标准用例; comm name必须全局唯一是基本要求
- Review rule: HCCL comm name必须包含group标识以避免跨group冲突; P2P路径不应对子group设限

### D-804: aclgraph auto_dispatch中softmax_lse大小不随flag变化

- Root cause: 条件分支缺失
- Hashes: `957721c81` `4db4167c0` `a95855310`
- Files: `torch_npu/npu/graphs.py`, `test/npu/test_aclgraph_update.py`
- Root cause: tensor分配未考虑可选参数的存在性
- Defect: `_GraphDispatchMode`中为`npu_fused_infer_attention_score`预分配softmax_lse时，无论kwargs中是否有`softmax_lse_flag`，都分配size(1)的tensor。当该flag不存在时，算子预期size(0)的placeholder。错误的size导致graph replay时shape mismatch。
- Fix: 检查`softmax_lse_flag`是否在kwargs中，据此分配size(1)或size(0); 测试中用`empty_like(res_src[1])`确保shape一致; 放宽fp16精度容差
- Reviewability: medium — 需要理解算子的可选参数语义
- Review rule: graph capture中为算子预分配output tensor时，必须考虑所有可选参数对output shape的影响

### D-805: select_opt循环越界(off-by-one) + 字节解码异常

- Root cause: off-by-one越界访问
- Hashes: `64389feec` `d72b61d80` `788179f82` `28054b75f` `6a09c4cce`
- Files: `torch_npu/csrc/framework/contiguous/select_opt.cpp`, `ci/access_control_test.py`, `test/distributed/test_fault_mode.py`, `test/trans_contiguous/*.py`
- Root cause: 循环条件`<=`应为`<`
- Defect: `for (size_t i = 0U; i <= select_size.size(); i++)` — 当i等于size时，`select_size[i]`和`base_size[i]`越界访问SmallVector。可能读到垃圾数据或触发ASAN报错。附带修复: bytes.decode('utf-8')在遇到非UTF-8字节时抛异常，改为errors='ignore'; 测试中operator名从StridedSlice更新为AsStrided。
- Fix: `<=` 改为 `<`; decode添加errors参数
- Reviewability: high — `<= .size()`是经典off-by-one red flag; review时任何`<=`配合`.size()`都应质疑
- Review rule: 循环条件中`.size()`/`.length()`必须配合`<`而非`<=`

### D-806: NumelList.calc_numels无限递归

- Root cause: 递归逻辑错误(无限递归)
- Hashes: `ffb503304` `771103cd7` `f68a95a64` `af6d8a4d9`
- Files: `torch_npu/_inductor/codegen/npu_kernel_features.py`
- Root cause: 方法在Iterable分支调用自身并传入相同参数
- Defect: `calc_numels`检查`isinstance(other, Iterable)`时调用`NumelList.calc_numels(other)`。但NumelList继承自Tuple(是Iterable)，所以传入NumelList实例时匹配Iterable分支，无限递归直到栈溢出。此方法是dead code(未被外部调用)，但其存在掩盖了设计缺陷。
- Fix: 删除整个`calc_numels`方法; 添加完整的NumelList单元测试
- Reviewability: high — 递归调用传入相同参数在review中应立即被标记
- Review rule: 递归方法的每次递归调用必须减小问题规模; `isinstance`分支中调用自身须确认参数类型变化

### D-807: 内存分配和字符串解析的安全加固

- Root cause: 安全漏洞(unchecked allocation + input validation)
- Hashes: `72005560c` `568fa7383` `f01285d41` `102c95455` `d80ec9c8b`
- Files: `torch_npu/csrc/core/npu/NPUCachingAllocator.cpp`, `torch_npu/csrc/core/npu/GetAffinityCPUInfo.cpp`, `torch_npu/csrc/aten/common/TensorProperties.cpp`, `torch_npu/csrc/core/npu/NPUException.h`
- Root cause: 未处理分配失败 + 未验证外部输入
- Defect: (a) NPUCachingAllocator中`new ExpandableSegment`和`new Block`使用throwing new，分配失败直接std::terminate(或未捕获的bad_alloc)。(b) `parseAffinityCPU`使用`stoi`解析来自dcmi的CPU亲和性字符串，恶意/畸形输入导致异常或溢出。(c) 多处拼写错误"supportted"→"supported"。
- Fix: (a) 改用`new (std::nothrow)` + nullptr检查; (b) 改用`strtol` + 范围/格式验证; (c) 修正拼写
- Reviewability: high — 安全review应系统性检查`new`和`stoi`的使用
- Review rule: 分配器代码中的`new`必须使用nothrow并检查返回值; 外部输入解析禁止使用throwing的stoi/stol

### D-808: weak_ref tensor丢失npu_desc + schema regex匹配不全 + graph更新逻辑硬编码

- Root cause: NPU存储描述符传播遗漏 + 正则表达式错误 + 可维护性缺陷
- Hashes: `be5403ca8` `da06bf49b` `9ed244e6f` `d8c00fa30`
- Files: `torch_npu/csrc/npu/Module.cpp`, `torch_npu/npu/graphs.py`, `test/npu/test_npu_format.py`
- Root cause: from_blob未传播NPU特有存储描述符; 正则未覆盖所有Tensor声明变体
- Defect: (a) `_weak_ref_tensor`通过`from_blob`创建新tensor但未复制`npu_desc_`(NPU storage format信息)，导致weak ref tensor的`get_npu_format`返回默认值而非原tensor的格式(如FRACTAL_NZ)。(b) schema解析的正则`Tensor\??\s*(\w+)`无法匹配`Tensor(a!) output`形式(mutable ref)。(c) graph update中op-specific的if/elif链不可扩展。
- Fix: (a) from_blob后复制src的`npu_desc_`到新tensor; (b) 正则改为`Tensor(?:\(a!\)|\?)?\s+(\w+)`; (c) 重构update逻辑为position-based参数查找
- Reviewability: medium — npu_desc传播需要NPU存储模型知识; regex缺陷需对照PyTorch的schema DSL
- Review rule: 任何创建tensor的路径(from_blob/make_tensor/clone)都必须传播npu_desc; schema regex必须覆盖Tensor的所有修饰形式

### D-809: IPC tensor序列化缺少meta tensor分支

- Root cause: dispatch遗漏(设备类型未覆盖)
- Hashes: `f4ffe9b92` `38d533999` `64aded1e7` `f91b26ae1` `bc51ab84c` `9ae698c45` `5cbbf7528`
- Files: `torch_npu/multiprocessing/reductions.py`
- Root cause: tensor reduction只处理了NPU device和CPU device，遗漏meta device
- Defect: `_npu_reduce_tensor`在pickle tensor时，根据storage device类型分派序列化逻辑。当tensor的storage在meta device上时(如通过`torch.device('meta')`创建)，没有匹配的分支，导致序列化返回None或走到错误路径。meta tensor在模型初始化(延迟materialization)场景中常见。
- Fix: 添加`elif storage._untyped_storage.device.type == "meta"`分支，使用`rebuild_meta_tensor`重建
- Reviewability: high — 设备类型dispatch应覆盖所有可能类型; meta device是PyTorch 2.x的标准设备
- Review rule: tensor序列化/反序列化的设备类型dispatch必须覆盖npu/cpu/meta所有分支

### D-810: ONNX wrapper参数名与upstream op签名不一致

- Root cause: API签名漂移
- Hashes: `28b0c5382` `d9ef9038c` `186968273` `6a9eb8c47` `dd892d5f8` `bd8d61eea` `9ac3362ed`
- Files: `torch_npu/onnx/wrapper_onnx_ops.py`, `test/torch_npu_schema.json`, `test/unsupported_test_cases/.pytorch-disabled-tests.json`
- Root cause: NPU自定义op的参数名变更未同步到ONNX wrapper
- Defect: `npu_group_norm_silu`的参数从`(x, gamma, beta, ...)`改为`(input, weight, bias, ...)`; `npu_rotary_mul`从`(x, r1, r2)`改为`(input, r1, r2, rotary_mode='half')`。ONNX wrapper仍使用旧参数名，导致keyword argument mismatch。
- Fix: 更新wrapper和schema中的参数名; 跳过不稳定的CANN版本相关测试
- Reviewability: medium — op签名变更应通过schema.json的diff自动检测
- Review rule: 修改NPU自定义op签名时，必须同步更新onnx/wrapper_onnx_ops.py和torch_npu_schema.json

### D-811: aclgraph tensor重建时npu_desc为空

- Root cause: NPU存储描述符初始化遗漏
- Hashes: `35ee52898` `f036e8701` `ec60174ce` `e69a7cbf5` `f391533a6` `5e57d9dd8`
- Files: `torch_npu/csrc/npu/Module.cpp`
- Root cause: tensor重建路径未设置StorageDesc
- Defect: aclgraph replay时从metadata重建tensor，调用`make_tensor_base`后直接`set_sizes_and_strides`，但未调用`StorageDescHelper::SetDesc`设置NPU的format/offset/baseSize等信息。后续对该tensor调用`get_npu_format`或进行format-sensitive操作时得到默认值(NCHW/0)而非实际格式。与D-808(a)是同类缺陷但不同路径。
- Fix: 在`set_sizes_and_strides`前调用`SetDesc`
- Reviewability: high — 已知NPU tensor必须设置npu_desc; 同类缺陷(D-808)可作为review checklist项
- Review rule: 所有tensor创建/重建路径的exit point必须包含StorageDesc初始化

### D-812: warnings.warn格式字符串误用 + bytes.decode缺少错误处理

- Root cause: Python API误用
- Hashes: `d7f1d9e2d` `81872688e` `afd035cd7`
- Files: `torch_npu/testing/__init__.py`, `ci/access_control_test.py`
- Root cause: warnings.warn签名误解 + 编码健壮性不足
- Defect: (a) `warnings.warn("Attempted to load json file '%s'...", filename)` — `warnings.warn(message, category)`的第二个参数应为warning类别(如DeprecationWarning)，传入filename字符串触发TypeError。(b) `line.decode('utf-8')`在遇到非UTF-8字节(如二进制输出、损坏编码)时抛UnicodeDecodeError。
- Fix: (a) 改用f-string: `f"... {filename} ..."`; (b) 添加`errors='ignore'`
- Reviewability: high — warnings.warn的签名是常见误用; decode无error handling在review中应被标记
- Review rule: warnings.warn的第二参数必须是Warning子类，不是格式化参数; 外部进程输出的decode必须指定errors策略

### D-813: aclrtMemcpyAsyncWithCondition缺少驱动版本守卫

- Root cause: API可用性守卫缺失
- Hashes: `e0b833064` `8e73bff8c` `451b9760b` `3f3e3333b` `5f9c30c39` `898aff263` `e89d11528` `7787cfc0b` `de4db7a2d`
- Files: `torch_npu/csrc/core/npu/interface/AclInterface.cpp`
- Root cause: 仅检查函数指针是否可加载，未检查驱动版本
- Defect: `AclrtMemcpyAsyncWithConditionExist`通过dlsym检查`aclrtMemcpyAsyncWithCondition`是否存在。但该API需要driver >= 25.0.RC1; 旧驱动中虽然符号可能存在(forward compatibility stub)，调用时返回错误或行为异常。
- Fix: 在dlsym检查前添加`IsGteDriverVersion("25.0.RC1")`守卫
- Reviewability: medium — 需要知道API与驱动版本的对应关系; dlsym存在不等于可用
- Review rule: ACL新增API的可用性检查必须同时验证函数指针存在+驱动版本达标; 版本号应从API文档中获取并写入代码注释

### D-814: transfer_to_npu未拦截torch.amp.autocast

- Root cause: monkey-patch覆盖不完整
- Hashes: `9a520050d` `8725693dc` `35cc5f3d6` `7855d9f0a` `d45aa7f64` `b7cb40ceb` `71f61da9b` `69550dfc2` `fee27460d` `e3268ca5f`
- Files: `torch_npu/contrib/transfer_to_npu.py`, `test/contrib/test_transfer_to_npu.py`
- Root cause: autocast的多个入口点未全部拦截
- Defect: `transfer_to_npu`模块将`torch.autocast`放入白名单做cuda→npu替换，但用户还可以通过`torch.amp.autocast('cuda')`调用。这个路径直达`torch.amp.autocast_mode.autocast.__init__`，绕过了白名单中的包装。此外白名单中的`autocast`条目本身也是错误的(autocast不是torch的顶层函数，而是class)。
- Fix: 从白名单移除`autocast`; 直接patch `torch.amp.autocast_mode.autocast.__init__`
- Reviewability: medium — monkey-patch的完整性难以静态验证; 需要枚举用户的所有调用路径
- Review rule: transfer_to_npu的cuda→npu拦截必须覆盖同一功能的所有入口; 新增torch API入口时需同步检查

### D-815: watchdog recovery路径变量名错误+硬编码错误信息+日志级别不当

- Root cause: copy-paste错误 + 硬编码 + 日志语义错误
- Hashes: `377526a3a` `59948b315` `8cdaa39ff` `8c16e9a91` `a8ad0cd6b` `8bace4284` `d5ccccaa0` `b1161fd9a` `3c496912c`
- Files: `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp`
- Root cause: 多处copy-paste引入的变量名/逻辑错误
- Defect: (a) `workCleanupLoop`的catch块中使用`device_error_msg.c_str()`(成员变量，可能为空)而非`device_error.c_str()`(刚从异常提取的局部变量)。(b) `workEnqueue`抛出硬编码`"UCE ERROR."`而非实际的`device_error_msg`，所有设备错误都报告为UCE。(c) FORCE_STOP判断条件`== npos`应为`!= npos`(逻辑反转)。(d) recovery路径中device_error/FORCE_STOP是预期信号，用error级别记录产生误导性告警。
- Fix: 修正变量名; 使用实际错误信息; 修正FORCE_STOP判断; error改为info
- Reviewability: high — 变量名不匹配和`==`vs`!=`在仔细的代码review中可发现; 建议对catch块中的变量使用与try块一致的命名
- Review rule: catch块中引用的变量必须与try块中捕获的异常信息对应; 错误恢复路径中的"预期异常"用info而非error记录


<!-- D-skip: f266a285d 904398b0e 9632bfc3f 213f96e85 2bbd0418c — fix device mesh ut (纯测试适配upstream DeviceMesh.from_group API变更, 非defect) -->
<!-- D-skip: 902c98ad4 — with_comms self.device 2.8.0 bugfix (纯测试基类迁移DTensorTestBase→NPUDTensorTestBase + decorator顺序调整) -->
<!-- D-skip: dc4bb5e30 81aba0c87 8378703e9 e56cad147 38fffc29d 2836f6c12 513f301ed b964e42f0 — Fixed import problems on py3.11 (兼容性测试跳过py3.11 enum内部属性_new_member_) -->
<!-- D-skip: 1f8a3a343 2a0f89be0 bd3fb0762 e74476bbb 989aaa706 82ac4a5e6 b372e5fd2 — fix ut (binary_cross_entropy_with_logits测试添加random seed消除flakiness) -->
<!-- D-skip: 72866c1ee 4115acbb1 c0426c7eb — fix fia lse ut (移除aclgraph update测试中softmax_lse的非确定性断言) -->

### D-816: transfer_to_npu PipelineStage的_device_wrapper包装语义错误

- Root cause: monkey-patch覆盖语义错误
- Hashes: `4d9efe967` `d8a4486aa` `7a648f602` `441df77c6` `39bc155b0` `c5e0af8ea` `d038a508a` `7f46e1979` `2cc06d0d1` `8c8352a9e`
- Files: `torch_npu/contrib/transfer_to_npu.py`
- Root cause: `_device_wrapper`对class和function使用了相同的包装语义
- Defect: `_device_wrapper(torch.distributed.pipelining.stage, ['PipelineStage', 'build_stage'])`将PipelineStage类本身通过`_wrapper_cuda`包装。但`_wrapper_cuda`拦截的是调用时的参数(替换'cuda'→'npu')，对class使用时替换的是class的构造调用。问题在于PipelineStage的`__init__`接收device参数但不是通过简单的字符串匹配能拦截的(device可能是torch.device对象)。直接包装`__init__`方法可以正确拦截所有device参数形式。
- Fix: 从`_device_wrapper`列表中移除`PipelineStage`，改为直接`PipelineStage.__init__ = _wrapper_cuda(PipelineStage.__init__)`
- Reviewability: medium — 需要理解`_device_wrapper`对class vs method的不同行为
- Review rule: transfer_to_npu中对class的cuda→npu拦截必须包装`__init__`而非class本身; `_device_wrapper`仅适用于module-level函数

### D-817: WatchdogStatus成员变量未初始化

- Root cause: C++成员变量未初始化
- Hashes: `26a240370` `e03caca23` `d2e4d933a` `2e9d43890` `f8a862ac0` `8b54c6cab` `e8f76c5a5` `d443a8646` `e3549d3e9`
- Files: `torch_npu/csrc/distributed/ProcessGroupHCCL.hpp`
- Root cause: enum类型成员变量声明时未提供默认值
- Defect: `ProcessGroupHCCL`的成员`watchdogStatus`声明为`WatchdogStatus watchdogStatus;`没有初始化。C++中enum类型不自动初始化，值为未定义。如果watchdog检查逻辑在status被显式设置之前执行(如构造函数异常路径)，行为不可预测。
- Fix: 声明时初始化为`WatchdogStatus::RUN`
- Reviewability: high — 编译器`-Wuninitialized`或静态分析可自动检测
- Review rule: 所有enum/POD类型成员变量必须在声明处或初始化列表中初始化; CI应启用`-Wuninitialized`

### D-818: tensor序列化/反序列化设备管理错误

- Root cause: 设备迁移顺序错误
- Hashes: `57f1c3373` `f68270c37` `b744aa68b`
- Files: `torch_npu/utils/storage.py`, `test/test_npu_multinpu.py`
- Root cause: `_rebuild_npu_tensor`中提前`.cpu()`破坏了后续操作的设备上下文; `_reduce_ex`中遗漏`.cpu()`导致NPU tensor直接参与序列化
- Defect: (a) `_rebuild_npu_tensor`在创建tensor前先将storage移到CPU(`storage = storage.cpu()`)，但随后用`tensor.to(device)`回迁。这个路径有两个问题: storage的device信息已丢失(被`.cpu()`覆盖); `to(device)`不能正确应用npu_format_cast。(b) `_reduce_ex`序列化NPU tensor时未先`.cpu()`，直接访问NPU storage的数据可能失败或产生错误结果。
- Fix: (a) 移除提前的`.cpu()`，改用`.npu()`直接在设备上构建tensor; (b) 添加`tmp_tensor = self.cpu()`确保序列化在CPU上进行
- Reviewability: medium — 需要理解PyTorch序列化流程中storage和tensor的设备关系
- Review rule: tensor序列化路径中，写端(reduce)必须先迁CPU再提取storage; 读端(rebuild)不应做无谓的设备迁移

### D-819: UCE错误处理从boolean标记重构为分类字符串

- Root cause: 错误分类粒度不足
- Hashes: `68dce0887` `4b9f9edaf` `c7336f723` `f867c0392` `4306b24ef` `547f5d384` `59573af1b` `8fb7c5275` `1043cea29`
- Files: `torch_npu/csrc/core/npu/NPUEvent.cpp`, `torch_npu/csrc/core/npu/NPUStream.cpp`, `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp`
- Root cause: 全局`uce_error_flag`(bool)无法区分不同类型的设备错误; `NPU_CHECK_ERROR_WITHOUT_UCE`宏吞掉了非UCE的设备错误
- Defect: 原始实现用`bool uce_error_flag`标记设备是否出错。但NPU设备错误有多种类型(UCE/HBM ECC/HCCS LINK等)，boolean无法区分。`NPU_CHECK_ERROR_WITHOUT_UCE`在非UCE场景静默忽略错误，导致NPUEvent和NPUStream中的设备错误被吞掉。新增`get_device_error()`函数解析错误消息并返回具体错误类型字符串。
- Fix: (a) `bool uce_error_flag` → `std::string device_error_msg`; (b) 新增`get_device_error()`分类函数(UCE/HBM ECC/HCCS LINK/HCCL RETRY FAILED等); (c) NPUEvent和NPUStream中`NPU_CHECK_ERROR_WITHOUT_UCE` → `NPU_CHECK_ERROR`
- Reviewability: medium — 需要理解UCE与其他设备错误的处理差异; boolean flag的信息丢失不容易在review中发现
- Review rule: 错误状态不要用boolean表示; 使用enum或字符串保留错误分类信息

### D-820: re模块被误删导致序列化功能异常

- Root cause: 误删依赖模块导入
- Hashes: `833c426c7`
- Files: `torch_npu/utils/serialization.py`
- Root cause: 之前的修改将`import re`误替换为`import tarfile`
- Defect: `serialization.py`需要`re`模块做正则匹配(如解析文件名、版本号等)。某次修改将`import re`替换为`import tarfile`，导致运行时`NameError: name 're' is not defined`。这是REVERT类型的fix。
- Fix: 恢复`import re`，移除无用的`import tarfile`
- Reviewability: high — diff中`-import re` / `+import tarfile`应立即引起警觉
- Review rule: 模块导入的删除/替换必须确认没有下游使用; linter的unused-import检查可防止遗漏

### D-821: inductor ascend_npu_ir tiling device memory管理接口变更

- Root cause: inductor codegen与runtime API不匹配
- Hashes: `1cf77447f`
- Files: `torch_npu/_inductor/ascend_npu_ir/ascend_npu_ir/cpp_common/cpp_common.cpp`, `cpp_common.h`, `npu/codegen/cpp_wrapper.py`
- Root cause: `common_launch_dyn`内部分配device tiling memory，但调用方需要管理这块memory的生命周期
- Defect: `common_launch_dyn`原实现在函数内部调用`aclrtMalloc`分配`arg_tiling_device`。这导致: (a) device memory在每次launch时重复分配，无法被调用方缓存复用; (b) 函数签名缺少`arg_tiling_device`参数，Python codegen侧无法传入预分配的buffer。同时修复了zero-rank tensor的comma格式问题(`ranks[i]==0`时避免产生多余逗号)。
- Fix: (a) `arg_tiling_device`从内部分配改为外部传入(新增函数参数); (b) Python codegen和PyArg_ParseTuple同步更新格式字符串; (c) 修复zero-rank tensor的逗号拼接逻辑
- Reviewability: low — 跨语言(C++/Python codegen)的API变更需要整体理解tiling memory生命周期
- Review rule: inductor codegen的C++函数签名变更必须同步更新Python wrapper生成逻辑和格式字符串; 增加端到端测试验证codegen输出可编译

### D-822: npu_dtype_cast_backward自定义sharding策略错误

- Root cause: DTensor分片策略注册方式错误
- Hashes: `40dce02b3` `a26ffba94` `c443bb314` `82a2c01e3` `f540db14a` `428568d7c`
- Files: `torch_npu/distributed/tensor/_matrix_ops.py`, `torch_npu/utils/dtensor.py`
- Root cause: 手动实现的sharding策略逻辑不正确(shard维度选择有误)
- Defect: `npu_dtype_cast_backward`是pointwise op(逐元素dtype转换的反向)，但被手动注册了matrix-op风格的sharding策略(`_get_max_shardable_dim`)。这个策略选择最大可分片维度做shard，但对pointwise op来说任意维度都可以shard，且输入输出的sharding应一致。手动策略的shard选择与实际需要不匹配。
- Fix: 移除自定义sharding函数; 将`npu_dtype_cast_backward`注册到DTensor的pointwise ops列表(同时注册`npu_dtype_cast_backward.default`和`_npu_dtype_cast_backward.default`)，使用框架默认的pointwise sharding逻辑
- Reviewability: medium — 需要理解DTensor的pointwise vs matrix op分片语义差异
- Review rule: pointwise op不应注册自定义sharding策略; 使用DTensor框架的默认pointwise分片即可

### D-823: transfer_to_npu DeviceMesh的_wrapper_cuda包装对象错误

- Root cause: monkey-patch覆盖语义错误
- Hashes: `3e0dd348a` `460f01b65` `e532498ff` `6a9ae966e` `72fb04b1e` `d086dc709` `51116a73f`
- Files: `torch_npu/contrib/transfer_to_npu.py`
- Root cause: 与D-816相同的pattern: 对class本身而非`__init__`做`_wrapper_cuda`包装
- Defect: `torch.distributed.device_mesh.DeviceMesh = _wrapper_cuda(torch.distributed.device_mesh.DeviceMesh)`将DeviceMesh类替换为包装后的callable。这导致`isinstance(obj, DeviceMesh)`失败(因为DeviceMesh已不再是原class)，downstream代码依赖isinstance检查的逻辑全部失效。
- Fix: 改为`DeviceMesh.__init__ = _wrapper_cuda(DeviceMesh.__init__)`，保留class identity
- Reviewability: high — isinstance失败是可预期的后果; 包装class vs 包装__init__的区别是基础Python知识
- Review rule: (同D-816) transfer_to_npu中对class的cuda→npu拦截必须包装`__init__`而非class本身

### D-824: torch.Event通过白名单包装语义错误

- Root cause: monkey-patch覆盖不完整
- Hashes: `c921e054c` `16bc32f02` `c82165990` `1e3340d69` `6f5136cab` `b3c9d4f84`
- Files: `torch_npu/contrib/transfer_to_npu.py`, `test/contrib/test_transfer_to_npu.py`
- Root cause: `torch.Event`是class不是function，放在`torch_fn_white_list`中被`_device_wrapper`包装后失去class identity
- Defect: `torch_fn_white_list`中包含`'Event'`，但`torch.Event`是class。`_device_wrapper`的包装方式适用于function(替换参数中的'cuda'→'npu')，对class使用会破坏isinstance检查。此外`Event`支持位置参数`Event('cuda:0')`，白名单的参数替换无法处理这种调用形式。
- Fix: (a) 从白名单移除`Event`; (b) 创建`_EventProxy(torch.Event)`子类，在`__new__`中拦截device参数(支持位置和关键字两种形式); (c) `torch.Event = _EventProxy`
- Reviewability: medium — 与D-814(autocast)是同一pattern: 白名单机制不适用于class
- Review rule: `torch_fn_white_list`只能包含function/staticmethod; class必须用Proxy子类方式包装

### D-825: ACL_PRECISION_MODE用户配置被硬编码覆盖

- Root cause: 配置项未生效
- Hashes: `6774b53cc` `0fbf6df60` `2a6c49deb` `67747b155` `551cbb598` `11cf2d7c5` `6059f22dc` `94ac4aaff`
- Files: `torch_npu/csrc/framework/LazyInitAclops.cpp`
- Root cause: 精度模式硬编码为SoC版本对应的默认值，未检查用户配置
- Defect: `SetPrecisionMode()`根据SoC版本设置`precision_mode`为`"must_keep_origin_dtype"`(910B+)或`"allow_fp32_to_fp16"`(其他)，然后直接调用`AclSetCompileopt`。用户通过`ACL_PRECISION_MODE`环境变量或option设置的值被完全忽略。这导致用户无法在910B上使用混合精度(`allow_fp32_to_fp16`)等非默认模式。
- Fix: 在硬编码默认值后，检查`c10_npu::option::GetOption("ACL_PRECISION_MODE")`，如有值则覆盖默认值
- Reviewability: high — 硬编码配置值而不检查用户输入是明显的设计遗漏
- Review rule: 所有ACL编译选项的设置必须先检查用户配置(option/env)，再fallback到默认值

### D-826: LazyInitAclops在非aclop路径(triton)上无条件执行

- Root cause: 初始化作用域过大
- Hashes: `8e546c7aa` `1f2febca0` `81d01ab4d` `4c27f6749` `0e85c7849` `85a0ccdbd` `828f1806d` `6df8a784a`
- Files: `torch_npu/csrc/framework/OpCommand.cpp`
- Root cause: `LazyInitAclops()`和jitCompile option设置位于`Run()`函数的公共路径
- Defect: `OpCommand::Run()`中，`LazyInitAclops()`和`AclSetCompileopt(ACL_OP_JIT_COMPILE, ...)`在所有op类型的公共路径执行。但triton kernel走的是非aclop路径(CustomHandler不为null)，不需要aclop初始化。在triton路径上执行aclop初始化会: (a) 尝试获取不存在的jitCompile option导致option解引用失败; (b) 不必要的初始化开销。
- Fix: 将`LazyInitAclops()`和jitCompile设置移入`CheckCustomHandlerNull()`为true的分支内(即仅aclop路径执行)
- Reviewability: high — `CheckCustomHandlerNull()`的if-guard和随后的无条件初始化在同一函数中，顺序阅读即可发现
- Review rule: aclop特有的初始化和配置必须放在aclop路径guard内; 公共路径只能包含所有backend共享的逻辑

### D-827: IPC场景get_ptr()缺少nullptr守卫

- Root cause: 空指针解引用
- Hashes: `f877a3e6f` `a99625624` `c70d8433a` `afba2f179` `868567887` `3809f39bc` `fb610565a` `c4fe9ee14` `24a31844c`
- Files: `torch_npu/csrc/core/npu/NPUCachingAllocator.cpp`
- Root cause: `get_ptr()`的else分支未检查`expandable_segment_`是否为nullptr
- Defect: `get_ptr()`原有两个分支: (a) `npu_ipc_ptr_`非null时返回; (b) else直接调用`expandable_segment_->ptr()`。但`expandable_segment_`可能为nullptr(初始状态或IPC场景下segment未创建)，此时解引用导致segfault。这是一个典型的if-else未覆盖所有状态的问题。
- Fix: 将else分支拆为: 检查`expandable_segment_`非null则返回其ptr，否则返回nullptr
- Reviewability: high — 两成员变量都可能为null，if-else只处理了其一; 静态分析工具(Coverity, CodeChecker)可自动检测
- Review rule: 指针成员的访问必须有null检查; 对象有多个可选数据源时，get方法必须处理所有组合

### D-828: sanitizer patch在torch_npu导入时无条件执行

- Root cause: 初始化副作用过早
- Hashes: `2eb2048fc` `c568e44c7` `3aeeea99c` `d31ac6096`
- Files: `torch_npu/__init__.py`, `torch_npu/npu/_sanitizer.py`
- Root cause: `apply_sanitizer_patch()`被放在`_apply_class_patches()`中，每次import torch_npu都会执行
- Defect: `torch_npu/__init__.py`的`_apply_class_patches()`在模块导入时无条件调用`apply_sanitizer_patch()`。sanitizer patch会修改stream/event的行为(插入检查逻辑)，对性能有影响。用户即使不使用sanitizer功能也承担了这个开销。更重要的是，patch可能与其他功能(如aclgraph)产生交互问题。
- Fix: 从`_apply_class_patches()`移除`apply_sanitizer_patch()`; 改为在`enable_npu_sanitizer()`中按需调用
- Reviewability: high — import-time side effect是Python反模式; `_apply_class_patches()`中每个调用都应审查是否需要无条件执行
- Review rule: 有性能影响或行为修改的patch不应在import时执行; 用户显式enable时才激活

### D-829: SilentCheck(ASD) DTensor兼容性 + inference mode tensor不可变异

- Root cause: 类型检查目标错误 + inference mode tensor mutability
- Hashes: `adcbcc26b` `4e14f3a0e` `7aa6f9234` `21fad641a` `c59e6e7fa` `b624e1b00` `f65fa375d` `112a0d893` `21a7a7cbd`
- Files: `torch_npu/asd/asd.py`
- Root cause: (a) `_MatmulSilentCheck`未处理DTensor类型的grad; (b) inference mode创建的tensor是immutable的，fill_操作抛异常; (c) isinstance检查了错误的对象
- Defect: 三个阶段的问题: (1) `_MatmulSilentCheck._backward_hook`中`self.statistic_value.fill_(...)`对DTensor失败(DTensor的fill_语义不同)。(2) inference mode下创建的tensor(`self.statistic_value`, `self.statistic_cpu_value`)是read-only的，后续的fill_和赋值操作会抛`RuntimeError`。(3) 修复DTensor时使用了`isinstance(self.statistic_value, DTensor)`来决定是否调用`.to_local()`，但应检查的是`grad`(输入参数)而非`self.statistic_value`(状态变量)。
- Fix: (1) 添加`from torch.distributed.tensor import DTensor`，对DTensor类型的grad调用`.to_local()`后再计算norm; (2) 在fill_前检查`is_inference()`，如果是则`.clone()`创建可写副本; (3) 修正isinstance检查对象从`self.statistic_value`改为`grad`
- Reviewability: medium — DTensor兼容性需要了解分布式tensor的语义; inference mode的immutability是容易遗漏的约束
- Review rule: 所有tensor原地操作(fill_, copy_等)前必须检查tensor是否可写(非inference mode); hook回调中的isinstance检查应针对回调参数而非实例状态

### D-830: HF32默认值硬编码，忽略用户配置

- Root cause: 硬编码默认值忽略运行时配置
- Hashes: `5d4776d23` `87e24d31d` `7bbcfd5ae` `51642b217`
- Files: `torch_npu/csrc/framework/LazyInitAclops.cpp`
- Root cause: `SetHF32DefaultValue()`将`allow_hf32`硬编码为`"10"`(conv=1, matmul=0)，不读`ALLOW_MATMUL_HF32`和`ALLOW_CONV_HF32`配置项
- Defect: ACL的`ACL_ALLOW_HF32`编译选项由两位数字组成: 第一位控制conv的HF32，第二位控制matmul的HF32。原代码直接写死`"10"`，意味着无论用户如何配置`ALLOW_MATMUL_HF32`和`ALLOW_CONV_HF32`环境变量，HF32行为都不会改变。这是一个典型的"placeholder常量未被替换为实际逻辑"的遗留缺陷。
- Fix: 分别读取`ALLOW_MATMUL_HF32`(默认disable)和`ALLOW_CONV_HF32`(默认enable)配置，动态组合为`conv_hf32 + mm_hf32`字符串
- Reviewability: high — 硬编码的`"10"`在review时应被质疑为何不读配置
- Review rule: 有对应配置项的编译选项不应使用硬编码默认值; 初始化代码中的字面量需要注释或替换为配置读取

### D-831: Profiler ONLY_FWK_CONFIG缺少TreeBuildParser和StepInfoDbParser

- Root cause: 功能模式配置不完整(parser注册遗漏)
- Hashes: `cda3cae55` `4608953c8` `9f06fb064` `991c39fbb`
- Files: `torch_npu/profiler/_dynamic_profiler/_dynamic_profiler_monitor.py`, `torch_npu/profiler/_dynamic_profiler/_dynamic_profiler_utils.py`, `torch_npu/profiler/analysis/prof_config/_parser_config.py`, `torch_npu/profiler/analysis/prof_config/_parser_deps_config.py`, `torch_npu/profiler/analysis/prof_view/prof_db_parse/_db_parser.py`
- Root cause: `ONLY_FWK_CONFIG`(仅CPU/framework数据的profiling模式)的Db pipeline中缺少`TreeBuildParser`，且`StepInfoDbParser`只注册在`ANALYSIS_DB_MAP`(分析模式)中，不在默认`DB_MAP`中
- Defect: 当用户仅启用CPU profiling时走`ONLY_FWK_CONFIG`路径。该配置的Db处理链缺少`TreeBuildParser`(构建op树)，导致`StepInfoDbParser`拿不到step划分所需的树结构数据，STEP_TIME表创建失败。同时`StepInfoDbParser`被错误地归入`ANALYSIS_DB_MAP`(只在分析阶段执行)，而它实际上应该在数据导出阶段就执行。附带修复: monitor进程的logger使用类变量`CFG_CONFIG_PATH`获取日志路径，但该变量在子进程中可能未初始化，改为从调用方传入`log_path`参数。
- Fix: (1) 在`ONLY_FWK_CONFIG`的Db pipeline中添加`TreeBuildParser`及其依赖配置; (2) 将`StepInfoDbParser`从`ANALYSIS_DB_MAP`移到主`DB_MAP`; (3) `init_logger`增加`log_path`参数
- Reviewability: medium — 需要理解profiler的多种模式(FULL/ONLY_FWK)各自需要哪些parser; parser的DB_MAP vs ANALYSIS_DB_MAP归属需要文档
- Review rule: 新增parser时必须检查所有运行模式(FULL_CONFIG, ONLY_FWK_CONFIG)的配置是否需要同步更新

### D-832: P2P连接计数初始值错误(自连接占用配额)

- Root cause: 资源计数初始值错误(off-by-one变种)
- Hashes: `20824785a`
- Files: `torch_npu/csrc/core/npu/NPUPeerToPeerAccess.cpp`
- Root cause: `device_enabled_count_`初始值为1(将自连接计入连接数)，但自连接不消耗实际P2P资源
- Defect: NPU的P2P连接限制为每设备最多8个peer(C10_P2P_ACCESS_MAX_NPUS=8)。`NpuP2pCtrl`构造函数中`device_enabled_count_.resize(num_devices_, 1)`将每个设备的已连接数初始化为1(计入了对角线上的自连接)，导致实际可用peer连接数为7而非8。此外，当达到连接限制时: (a) 原代码同时检查source和dest两端但不区分哪一端达到上限; (b) 使用WARN级别而非ERROR级别; (c) 生成的连接列表包含自连接。
- Fix: (1) 初始值改为0; (2) 分别检查source和dest设备的连接数并报告具体哪个设备达到上限; (3) 改用ASCEND_LOGE; (4) 生成连接列表时跳过自连接
- Reviewability: high — `resize(num_devices_, 1)`与对角线`COPY_ALLOWED`赋值并列出现，review时应质疑初始值为何是1而非0
- Review rule: 资源计数的初始值必须与实际资源消耗对应; 自引用/自连接不应计入外部资源配额

### D-833: StepInfoDbParser并发模式错误(SUB_PROCESS应为PTHREAD)

- Root cause: 并发模式选择错误(跨进程数据不可见)
- Hashes: `70032b8e4` `a51476d92` `6b7f1eca2`
- Files: `torch_npu/profiler/analysis/prof_config/_parser_deps_config.py`
- Root cause: `StepInfoDbParser`的`ConcurrentMode`设置为`SUB_PROCESS`，但它依赖`TreeBuildParser`的数据，而该数据存在于主进程内存中
- Defect: profiler的并发任务管理器支持`PTHREAD`(线程)和`SUB_PROCESS`(子进程)两种模式。`SUB_PROCESS`模式下parser在独立进程中运行，无法访问主进程的内存数据。`StepInfoDbParser`依赖`TreeBuildParser`构建的op树数据来划分step边界，但被配置为`SUB_PROCESS`模式，导致它在子进程中拿不到树数据，step info表导出失败。这与D-831是相关但独立的缺陷: D-831解决的是ONLY_FWK_CONFIG中parser缺失的问题，本条解决的是parser注册了但运行模式不对的问题。
- Fix: 将`StepInfoDbParser`的`ConcurrentMode`从`SUB_PROCESS`改为`PTHREAD`
- Reviewability: high — 依赖主进程内存数据的parser不能用SUB_PROCESS模式，这是一条不变量
- Review rule: 选择parser的并发模式时，必须检查其依赖的数据是否跨进程可见; 依赖内存中shared state的parser必须用PTHREAD

### D-834: 通信矩阵单step时边界条件处理错误

- Root cause: 边界条件错误(单元素列表等同空列表)
- Hashes: `1057611de` `66ad5803d` `56ced99b6`
- Files: `torch_npu/profiler/analysis/prof_view/communication_parser.py`
- Root cause: `split_matrix_by_step()`中`len(self.step_list) == 1`被当作无需分步处理的特例，跳过了step命名
- Defect: `split_matrix_by_step()`将通信矩阵数据按训练step拆分。原代码`if len(self.step_list) == 1 or self.is_step_list_empty()`将"只有1个step"等同于"没有step"处理，直接以"step"为key返回。这导致单step场景下丢失了step_id信息。同时，`step_info.get("step_id")`直接作为key使用(如`"0"`)，不符合输出格式预期(应为`"step0"`)。
- Fix: (1) 移除`len(self.step_list) == 1`条件，只保留`is_step_list_empty()`; (2) step key格式改为`"step" + step_id`，对None step_id降级为`"step"`
- Reviewability: high — `len(list) == 1 or is_empty()`是明显的可疑模式; 单元素列表不应跳过正常处理流程
- Review rule: 单元素集合应走正常路径而非空集合的快捷路径; 边界条件测试须覆盖0、1、N三种情况

### D-835: mstx.range_start使用print(Warning)代替warnings.warn + stream必传

- Root cause: 错误的API调用(print vs warnings) + 必需参数应为可选
- Hashes: `c792bdf90` `2cf3a07a4` `be7ba196a`
- Files: `torch_npu/npu/mstx.py`
- Root cause: (a) `print(Warning, msg)`不发出Python warning，而是打印`(<class 'Warning'>, 'msg')`元组; (b) stream参数无默认值，host-only场景必须传入但不应该
- Defect: `mstx.range_start()`的错误处理使用`print(Warning, "msg")`，这是一个常见的Python新手错误: `Warning`是一个类，`print`将其与字符串作为元组打印，输出类似`<class 'Warning'> msg`(实际是逗号分隔的两个参数)。正确的做法是`warnings.warn("msg")`。此外，stream参数是必传的(无默认值)，但mstx.range_start在host-only标记场景下不需要stream，应该调用`_range_start_on_host`。
- Fix: (1) 所有`print(Warning, ...)`替换为`warnings.warn(...)`; (2) stream参数设为可选(默认None); (3) stream为None时调用`_range_start_on_host(message)`; (4) stream类型不对时warn而非静默返回0
- Reviewability: high — `print(Warning, ...)` vs `warnings.warn(...)`是静态检查可自动检测的模式; lint工具应标记`print`中包含Exception/Warning子类的调用
- Review rule: Python中发出警告必须使用warnings.warn(); 工具API的可选参数应有合理默认值而非强制传入

### D-836: copy_路径创建不必要的中间tensor

- Root cause: 不必要的中间分配(copy路径优化)
- Hashes: `3a34bb7ec` `5348ba6fc` `8cc4bc183`
- Files: `torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp`
- Root cause: `copy_()`对complex/neg tensor先创建中间result tensor再copy，而非先copy再in-place变换
- Defect: 原`NPUNativeOpApiFunctions::copy_()`对complex tensor的处理: 先分解src为real/imag部分，创建临时result tensor，用`aclnnComplex`合成，再将result copy到dst。这个流程创建了一个与src同大小的中间tensor。对neg tensor同理，先创建neg的副本再copy。优化后的流程: 先将src直接copy到dst，然后对dst做in-place的conjugation/negation。这省去了中间tensor的分配和一次copy。这不仅是性能优化也是正确性修复: 原代码中`src._set_neg(false)`修改了src的neg标记，这是对const引用参数的非法修改(虽然参数声明为`const at::Tensor& src`，但PyTorch的tensor语义允许这种修改，属于逻辑上的副作用bug)。
- Fix: 重构为copy-then-transform模式: (1) 先执行d2d/h2d/d2h copy; (2) 如果src.is_conj()，对dst做aclnnComplex(real, -imag); (3) 如果src.is_neg()，对dst做neg_()
- Reviewability: medium — 需要理解complex tensor的conj/neg语义; `_set_neg(false)`修改const引用的副作用不明显
- Review rule: copy操作不应修改source tensor的状态(即使API允许); 优先考虑copy-then-transform而非transform-then-copy以避免中间分配

### D-837: ConcurrentMode bitmask比较使用==而非位运算

- Root cause: 位掩码比较方式错误(==替代&)
- Hashes: `585a2d5a6`
- Files: `torch_npu/profiler/analysis/prof_common_func/_task_manager.py`
- Root cause: `task.mode == ConcurrentMode.NON_BLOCKING`使用等值比较，但mode是组合标志位
- Defect: `ConcurrentMode`的值是位掩码(flag)，一个task的mode可以是多个flag的组合(如`PTHREAD | NON_BLOCKING`)。原代码`all((task_info.task.mode == ConcurrentMode.NON_BLOCKING for ...))`用`==`比较，只有mode恰好等于`NON_BLOCKING`时才匹配，mode为`PTHREAD | NON_BLOCKING`的组合值不会匹配。这导致组合模式的task永远不被认为是non-blocking的，epoll退出逻辑失效。这是一个经典的bitmask操作错误。
- Fix: 新增`is_non_blocking`属性: `return (self.mode & ConcurrentMode.NON_BLOCKING) != 0`，使用位与检查
- Reviewability: high — bitmask type的`==`比较是静态分析可检测的anti-pattern
- Review rule: 对flag/bitmask类型的枚举值，检查包含关系必须用位与`&`而非等值比较`==`

### D-838: ToKernel double dtype警告文本typo + 冗余前缀

- Root cause: 用户可见消息文本错误
- Hashes: `62e277c68` `7f27f5e5f` `69cb6ef3f` `dcf39b65b`
- Files: `torch_npu/csrc/aten/common/ToKernelNpu.cpp`
- Root cause: 警告消息中"replace"拼写为"repalce"; `TORCH_NPU_WARN_ONCE`宏已表明是warning，消息体再加"Warning:"前缀是冗余的
- Defect: `NPUNativeFunctions::to()`对double类型的降精度警告消息有两个问题: (1) "repalce"是"replace"的拼写错误; (2) 消息以"Warning: "开头，但`TORCH_NPU_WARN_ONCE`宏的输出已包含WARNING标记，造成"WARNING: Warning: Device do not support..."的冗余。
- Fix: 修正拼写"repalce"→"replace"; 移除消息体中的"Warning: "前缀
- Reviewability: high — 拼写错误在review和CI拼写检查中均可发现
- Review rule: 使用warning/error宏时，消息体不应重复宏自身已提供的严重级别前缀

### D-839: contrib测试设备限制过窄 + 精度容差不足

- Root cause: 测试约束过窄(设备白名单 + 精度阈值)
- Hashes: `2646225a8` `e17a59ab3` `8851cad5e` `5a6e13a2f` `cfcacb752` `bdfb33cfa`
- Files: `test/contrib/test_bbox_coder.py`, `test/contrib/test_deform_conv.py`, `test/contrib/test_linear_quant.py`
- Root cause: (a) `@SupportedDevices(['Ascend910A'])`限定过窄，新设备(910B/910P)上算子已支持但测试不运行; (b) 默认精度容差对非910A硬件不够宽松; (c) `LinearQuant`缺少`output_dtype`参数
- Defect: 多个contrib测试用`@SupportedDevices`硬编码为仅910A设备可运行，但: (1) `npu_bbox_coder_encode`算子910B已支持，需新增910B测试变体(期望值不同); (2) `DCNv2`(DeformableConv)在所有NPU上均可运行，不应限制为910A; (3) `DCNv2`在非910A上的数值精度略有差异，需要`prec=1.e-3`容差; (4) `LinearQuant`的`npu_quant_matmul`需要显式指定`output_dtype=torch.float16`，否则默认输出类型在不同硬件上可能不一致。
- Fix: (1) 移除或扩展`@SupportedDevices`; (2) 添加`prec`参数放宽容差; (3) `LinearQuant`构造函数和`npu_quant_matmul`调用中显式传入`output_dtype`
- Reviewability: medium — 设备白名单需要与算子支持矩阵同步更新; 精度容差需要跨硬件测试验证
- Review rule: `@SupportedDevices`白名单应随算子适配进度更新; 新硬件上线时需review所有设备限制注解; 精度敏感的数值比较应使用`prec`参数而非默认容差

### D-840: Python反斜杠续行导致RuntimeError消息含多余空格

- Root cause: 字符串格式错误(反斜杠续行缩进泄漏)
- Hashes: `0fd3db0b0` `5a7912953` `9bbb3b50e` `d85cc3655` `c41fa1f0b`
- Files: `torch_npu/npu/npu_config.py`
- Root cause: f-string使用`\`续行时，下一行的缩进空格被包含在字符串中
- Defect: `_call_once_class.__call__()`的RuntimeError消息使用反斜杠续行:
  `f"... has already been called, \` + 换行 + `                 You can only set..."`。
  Python的反斜杠续行会保留下一行的所有前导空格，导致错误消息中间出现约17个多余空格。用户看到的实际输出是`"... called,                  You can only set..."`。
- Fix: 改为单行f-string，消除续行
- Reviewability: high — Python反斜杠续行的缩进陷阱是已知反模式; linter可配置检测多行字符串中的`\`续行
- Review rule: Python字符串拼接优先使用括号隐式续行`("part1" "part2")`而非反斜杠; 反斜杠续行的下一行缩进会成为字符串内容

<!-- D-skip: daa9ef1e5 d5215ddd8 f7e46a0cd 416ce2569 — flight_recorder文档typo修复(bash前缀, .txt→.pkl), 纯文档变更 -->
<!-- D-skip: e1e392b68 e870bd466 — README autoloading最低版本号修正(2.6.0→2.5.1), 纯文档变更 -->
<!-- D-skip: b269d90cd 1a665439c c35e2a524 1a95740ee 8ea0d01fd — dynolog codecheck(添加空行/大括号换行), 纯代码风格 -->
<!-- D-skip: 63a4835ca 199b2e16e b2eeeb0de — test_torch.py upstream同步(新增import/dtype/设备判断), 纯测试基础设施 -->
<!-- D-skip: 66ef44363 253b79836 31413c723 — 分布式测试调整(HCCL_EXEC_TIMEOUT/tensor shape/error code统一), 纯测试变更 -->
<!-- D-skip: 374288af0 — test_set_snapshot_fn测试重构(从TestPluggableAllocator拆出独立TestSnapshot类), 纯测试重组 -->

### D-841: is_scalar_wrapped_to_device使用过时的标量判断函数

- Root cause: 过时API调用(函数语义变更后未同步更新)
- Hashes: `9b17c1dac` `4b4cbe26b`
- Files: `torch_npu/csrc/framework/utils/OpPreparation.cpp`
- Root cause: `CalcuOpUtil::IsScalarWrappedToTensor()`的判断逻辑不再正确，需改用`IsCPUScalar()`
- Defect: `OpPreparation::is_scalar_wrapped_to_tensor()`内部调用`CalcuOpUtil::IsScalarWrappedToTensor()`来判断tensor是否为wrapped scalar。随着PyTorch演进，scalar tensor的创建方式变化，原函数的判断条件(基于tensor metadata)不再可靠。`IsCPUScalar()`基于设备类型(CPU) + 0-dim的判断更准确。该函数在算子dispatch路径中用于决定是否走scalar优化分支，错误判断会导致走错分支。
- Fix: 替换为`IsCPUScalar(tensor)`
- Reviewability: medium -- 需要了解scalar detection语义的演进; 函数签名未变但语义已变属于隐式API变更
- Review rule: 工具函数重构时应grep所有调用点; 判断函数语义变更时考虑添加deprecation warning

### D-842: profiler memory view使用错误的属性名匹配torch op + 缺少tree node导航

- Root cause: 属性名不一致(event bean vs tree node接口) + 算法逻辑错误(线性回退vs树上溯)
- Hashes: `e5b5acd05` `4763c797b` `a633b846a` `fe95e020b`
- Files: `torch_npu/profiler/analysis/prof_view/memory_view_parser.py`, `torch_npu/profiler/analysis/prof_bean/torch_op_node.py`, `torch_npu/profiler/analysis/prof_view/trace_view_parser.py`, `torch_npu/profiler/analysis/prof_view/view_parser_factory.py`
- Root cause: (a) memory_view_parser使用`.ts`属性但torch_op_node实际提供`.start_time`; (b) 匹配逻辑用线性向前扫描而非沿parent_node上溯; (c) trace_view_parser过早清空GlobalVar.torch_op_tree_node; (d) torch_op_node缺少`.pid`/`.name`属性shortcut
- Defect: `_find_torch_ops_by_binary_search()`用`torch_ops[mid].ts`做二分查找，但TorchOpNode的时间戳属性是`.start_time`(来自`._event.start_time`)不是`.ts`。`_find_matched_torch_op_name()`的回退逻辑向前线性扫描(`matched_torch_op_idx -= 1`)最多MAX_FIND_LAYERS层，这在深嵌套场景下可能找不到包含该memory event的op；正确做法是沿`parent_node`链上溯直到找到时间范围包含mem_start_ts的祖先节点。此外，`_add_pta_memory_data()`从文件重新解析torch_op而非使用GlobalVar中已构建好的树结构，而TraceViewParser在generate_view结束时清空了树节点，导致memory view拿不到树。
- Fix: (1) `.ts`→`.start_time`, `.name`直接属性化; (2) 线性扫描→parent_node上溯; (3) 数据源从FwkFileParser改为GlobalVar.torch_op_tree_node; (4) 移除trace_view中对tree_node的清空; (5) TorchOpNode增加pid/name property
- Reviewability: medium -- 属性名不匹配可通过类型检查发现; 算法从线性扫描到树上溯需要理解op tree结构
- Review rule: 跨模块引用属性时应有接口文档或类型注解; profiler的全局状态生命周期需要明确的ownership约定

### D-843: INF_NAN_MODE_ENABLE默认值应为开启

- Root cause: 默认配置值错误
- Hashes: `9de3a537d`
- Files: `torch_npu/csrc/core/npu/register/OptionsManager.cpp`
- Root cause: `INF_NAN_MODE_ENABLE`的默认值设为0(关闭)，实际应默认开启
- Defect: `CheckInfNanModeEnable()`调用`GetBoolTypeOption("INF_NAN_MODE_ENABLE", 0)`，第二个参数0表示未设置环境变量时默认关闭inf/nan检测。NPU的inf/nan行为与GPU不同(部分算子不会自动传播nan)，默认关闭导致用户在未显式设置环境变量时无法感知数值异常。
- Fix: 默认值从0改为1
- Reviewability: high -- 默认值的语义影响是确定性的; review时应验证默认值与硬件行为预期是否匹配
- Review rule: 安全相关配置(inf/nan检测、内存检查等)应默认开启; 默认值变更需在release note中说明

### D-844: npu_format_cast缺少same-format短路 + autograd路径不完整

- Root cause: 缺少短路检查(冗余format cast) + autograd dispatch遗漏
- Hashes: `450ccdf8c` `4affb9520` `cd8228c5e`
- Files: `torch_npu/csrc/aten/common/FormatCastKernelNpu.cpp`, `codegen/autograd/gen_variable_type.py`, `torch_npu/csrc/aten/npu_native_functions.yaml`
- Root cause: (a) `npu_format_cast(self, acl_format)`未检查`self`是否已经是目标format; (b) 只有`npu_format_cast`在gen_variable_type中被重定向到NPUNativeFunctions，新增的`_npu_format_cast`不在其中
- Defect: 原`npu_format_cast()`直接调用`npu_format_cast_impl()`执行format转换，即使tensor已经是目标format也会走一遍完整的ACL调用路径。修复后增加`get_npu_format(self) == acl_format`检查，匹配时直接返回self。同时引入`_npu_format_cast`作为内部实现(不做check)供public API通过`custom_ops`dispatch调用，这样autograd可以正确跟踪。codegen中的字符串匹配从`== 'npu_format_cast'`扩展为`in NPU_NATIVEFUNCTIONS`集合，支持两个函数名。
- Fix: (1) 增加same-format短路返回; (2) 新增`_npu_format_cast`注册到yaml和codegen; (3) public `npu_format_cast`通过custom_ops调用内部版本
- Reviewability: high -- 缺少no-op短路是常见性能问题; autograd路径需要codegen和yaml同时更新
- Review rule: format/type转换函数应检查no-op场景; 新增custom op时同步更新所有codegen路径

### D-845: profiler基础设施多处修复(symlink安全 + start_info路径 + 文件模式regex)

- Root cause: 多组件基础设施修复(安全 + 功能 + 兼容性)
- Hashes: `b67cb0137`
- Files: `torch_npu/profiler/analysis/prof_common_func/file_manager.py`, `torch_npu/profiler/analysis/prof_common_func/path_manager.py`, `torch_npu/profiler/analysis/prof_parse/cann_file_parser.py`, `torch_npu/profiler/analysis/prof_view/view_parser_factory.py`, `torch_npu/profiler/msprofiler_c_interface.py`
- Root cause: (a) `make_dir_safety`缺少symlink检查; (b) start_info路径只查device目录不查host目录; (c) msprof timeline文件名新模式未匹配; (d) profiler_info.json用raw open()不经权限控制
- Defect: FileManager的目录创建不检查路径是否为符号链接，存在路径遍历风险。PathManager的`get_start_info_path()`只在device子目录下查找start_info文件，但新版CANN将该文件放在host/start_info下。CANNFileParser的MSPROF_TIMELINE正则缺少`msprof_X_X_slice_X_X.json`模式(四段式带双slice后缀)。MsProfilerInterface用`open(path, "w")`写profiler_info.json，不经过FileManager的权限控制(mode 0o640)。
- Fix: (1) `make_dir_safety`增加`os.path.islink`检查; (2) start_info先查host/再fallback device/; (3) 增加regex pattern; (4) 用FileManager.create_json_file_by_path替代raw open
- Reviewability: medium -- symlink检查和权限控制是安全review标准项; 路径优先级变更需了解CANN版本演进
- Review rule: 文件系统操作统一通过安全wrapper; 第三方数据格式变更时更新所有regex/parser

### D-846: LazyConv3d的weight在cast时尚未初始化

- Root cause: 缺少lazy module守卫(类型层级检查顺序错误)
- Hashes: `304a4c5ae`
- Files: `torch_npu/utils/module.py`
- Root cause: `cast_weight()`对Conv3d做format_cast时未排除LazyConv3d子类
- Defect: `cast_weight()`遍历module层级，对`torch.nn.Conv3d`执行`npu_format_cast(weight.half(), ACL_FRACTAL_Z_3D)`。但`torch.nn.LazyConv3d`是Conv3d的子类，其weight是`UninitializedParameter`(没有实际数据)。`issubclass(class_name, torch.nn.Conv3d)`对LazyConv3d也返回True，导致对未初始化的weight执行half()和format_cast，产生crash。
- Fix: 在Conv3d检查之前增加`issubclass(class_name, torch.nn.LazyConv3d): return`守卫
- Reviewability: high -- Lazy module是PyTorch的标准抽象; 类型层级中子类需要特殊处理是已知模式
- Review rule: 对基类的isinstance/issubclass操作需检查是否有需要排除的特殊子类; LazyModule的weight不可直接操作

### D-847: Revert export_stacks功能(profiler stack view未就绪)

- Root cause: 过早发布的功能(revert)
- Hashes: `8d06036cb` `77d27b6fb` `c237c2d60`
- Files: `torch_npu/profiler/analysis/prof_common_func/constant.py`, `torch_npu/profiler/analysis/prof_config/view_parser_config.py`, `torch_npu/profiler/analysis/prof_view/stack_view_parser.py`, `torch_npu/profiler/profiler.py`
- Root cause: export_stacks功能依赖的数据(torch_op_tree_node的call_stack属性、device_total_dur)在实际运行中不可靠
- Defect: StackViewParser从GlobalVar.torch_op_tree_node读取call_stack并按metric(cpu_time或npu_time)输出flamegraph格式。该功能被revert，删除了: StackViewParser类、EXPORT_STACK常量、METRIC_CPU_TIME/METRIC_NPU_TIME常量、ViewParserConfig中的EXPORT_STACK配置项、profiler.py中的export_stacks()方法。revert原因推测: call_stack数据在部分场景下为空、device_total_dur计算不准确、或host_total_dur与self_cpu_time_total的语义不一致。
- Fix: 完整删除export_stacks相关代码，后续需要重新设计
- Reviewability: low -- 功能本身实现没有明显错误，但集成后数据质量不达标
- Review rule: 新功能上线前需要端到端数据质量验证; profiler功能应有golden数据集做回归测试

### D-848: SetDeterministic在customHandler路径前未调用

- Root cause: 代码路径遗漏(early return跳过配置)
- Hashes: `574616fc4`
- Files: `torch_npu/csrc/framework/OpParamMaker.cpp`
- Root cause: `SetDeterministic()`调用位置在customHandler分支之后，custom handler执行时确定性配置未生效
- Defect: `OpParamMaker`的执行流程: 先检查`cur_paras->customHandler`是否存在，如果有则直接执行custom handler并return。`SetDeterministic()`的调用位置在custom handler检查之后，因此所有走custom handler的算子(如部分自定义融合算子)都不会应用deterministic设置。用户设置`torch.use_deterministic_algorithms(True)`对这些算子无效。
- Fix: 将`SetDeterministic()`移到custom handler检查之前
- Reviewability: high -- 代码中的return跳过后续配置是经典遗漏; 顺序依赖应该有注释说明
- Review rule: 配置/初始化代码应放在所有分支之前; early return跳过的代码段需要审查是否都是可跳过的

### D-849: SyncBatchNorm backward传入mean而非sum

- Root cause: 数学计算语义错误(mean vs raw sum)
- Hashes: `159eee9a0`
- Files: `torch_npu/utils/syncbatchnorm.py`
- Root cause: `batch_norm_backward_elemt()`期望接收sum_dy和sum_dy_xmu(原始求和值)，但调用方先做了除法得到mean再传入
- Defect: SyncBatchNorm的backward过程中，各rank先allreduce得到`sum_dy`和`sum_dy_xmu`的全局和，然后除以`count_tensor.sum()`得到均值。但`torch.batch_norm_backward_elemt`的接口约定是接收原始求和值(它内部会做归一化)，不是均值。传入均值等于做了两次归一化，导致梯度计算错误。这是一个数值正确性bug，会导致SyncBatchNorm的训练收敛异常。"Rollback"说明这个bug是之前某次"优化"引入的(有人试图在调用前预计算mean)。
- Fix: 直接传`sum_dy`和`sum_dy_xmu`，移除`divisor`和`mean_dy`/`mean_dy_xmu`中间变量
- Reviewability: medium -- 需要了解batch_norm_backward_elemt的接口约定; 函数文档不清晰时容易误用
- Review rule: 调用数学计算函数前确认输入是raw值还是normalized值; batch norm的forward/backward对称性是验证点

### D-850: setup.py在无git环境构建时VERSION拼接"Unknown"

- Root cause: 构建时外部工具依赖未降级处理
- Hashes: `b6fb80926` `1fced115f` `df1f3dac3` `edf4ce651`
- Files: `setup.py`
- Root cause: `get_sha()`在git不可用时返回"Unknown"，后续`VERSION += "+git" + sha[:7]`产生`"2.1.0+gitUnknown"`
- Defect: `generate_torch_npu_version()`调用`get_sha()`获取git commit hash追加到版本号。但在无git安装的环境(如CI docker镜像、源码tarball)中，`get_sha()`返回"Unknown"字符串。原代码不检查返回值直接拼接，产生非标准版本号`"2.1.0+gitUnknown"`。这个版本号不符合PEP 440规范，可能导致包管理器行为异常。
- Fix: 增加`sha != UNKNOWN`检查，未知时不追加git后缀
- Reviewability: high -- 边界条件(工具不可用)是构建脚本review标准检查项
- Review rule: 构建脚本的外部工具调用都需要graceful degradation; 版本号格式应有验证

### D-851: dropout mask长度计算uint32溢出

- Root cause: 整数类型溢出(uint32→uint64)
- Hashes: `9d97c545b`
- Files: `torch_npu/csrc/aten/ops/DropoutKernelNpu.cpp`
- Root cause: mask长度`(numels + 127) / 128 * 128`的中间结果用`uint32_t`存储，numels超过约5.4亿(4GB/8)时溢出
- Defect: `dropout_gen_mask()`和`npu_dropout_gen_mask()`计算对齐到128的mask长度，使用`uint32_t length`。当tensor元素数超过`UINT32_MAX`(约43亿)时，计算结果截断。实际触发条件更低: `numels + 128 - 1`在numels接近2^32-127时就溢出。对于大模型训练中的大batch或长序列，dropout的输入tensor可能有数十亿元素。溢出后mask长度异常小，导致mask覆盖不足，dropout只对tensor前部分生效。
- Fix: `uint32_t length` → `uint64_t length`
- Reviewability: high -- 大tensor场景下的整数溢出是已知反模式; 静态分析工具可检测narrowing conversion
- Review rule: tensor元素数相关计算必须使用int64_t/uint64_t; 对齐计算`(x+align-1)/align*align`的溢出风险需关注

### D-852: profiler step time将hcom_receive错误标记为bubble

- Root cause: 分类标签硬编码错误
- Hashes: `4f45fbcab` `24e741a90` `f21e8c1ff`
- Files: `torch_npu/profiler/analysis/prof_view/trace_step_time.py`
- Root cause: `hcom_receive`开头的op被硬编码归类为`"bubble"`而非`"hcom_receive"`
- Defect: `TraceStepTimeParser`在统计各step的时间分布时，对op name做前缀匹配分类。`hcom_receive`开头的通信op被`count_time('bubble', ...)`标记为bubble时间。但receive操作是实际的通信活动(等待接收远端数据)，不是bubble(空闲等待)。这导致profiling分析报告中: (1) bubble时间被高估(包含了实际通信时间); (2) 通信时间被低估(丢失了receive部分); (3) 性能优化方向被误导(用户可能尝试消除"bubble"而非优化通信)。
- Fix: `'bubble'` → `'hcom_receive'`
- Reviewability: high -- 分类逻辑中的硬编码字符串是review重点; 错误label在UI/报告中可观察到
- Review rule: profiler的时间分类需要与通信框架的op语义保持一致; 新增op类型时更新分类逻辑

### D-853: eraseStream中get_allocated_block返回null时未检查

- Root cause: 空指针解引用(缺少null check)
- Hashes: `fdfbded60` `4bf75e9ec` `a6228de62`
- Files: `torch_npu/csrc/core/npu/NPUCachingAllocator.cpp`
- Root cause: `eraseStream()`调用`get_allocated_block(ptr)`后直接解引用`block->device`，未检查block是否为null
- Defect: `THNCachingAllocator::eraseStream()`用于将block从某个stream的tracked set中移除。它通过`get_allocated_block(ptr.get())`查找block。如果ptr指向已释放或非allocator管理的内存，get_allocated_block返回nullptr。后续`block->device`直接解引用导致segfault。在multi-stream场景下，如果tensor在一个stream上被释放(free)但另一个stream仍持有指针并尝试eraseStream，就会触发此bug。
- Fix: 增加`if (!block) AT_ERROR("invalid device pointer: ", ptr.get());`
- Reviewability: high -- null check是代码review基本项; 所有get/find类调用的返回值都应检查
- Review rule: 指针查找函数的返回值在使用前必须检查null; allocator的public API需要防御性编程

### D-854: eraseStream中erase迭代器未接收返回值导致迭代器失效

- Root cause: 迭代器失效(STL erase使用错误)
- Hashes: `df5eaa17c` `7a71bc820`
- Files: `torch_npu/csrc/core/npu/NPUCachingAllocator.cpp`
- Root cause: `npu_events.erase(it)`后继续使用已失效的`it`迭代器
- Defect: `DeviceCachingAllocator::process_events()`遍历`npu_events`(deque)时，对满足条件的event执行`npu_events.erase(it)`。`std::deque::erase`会使被删除位置及之后的所有迭代器失效，但原代码在erase后继续使用`it`(隐式++it或后续访问)。这是经典的C++ STL迭代器失效bug，会导致未定义行为(通常表现为随机crash或跳过元素)。在高并发场景(多stream、多event)下更容易触发。
- Fix: `npu_events.erase(it)` → `it = npu_events.erase(it)` (erase返回下一个有效迭代器)
- Reviewability: high -- STL erase + iterator是C++ code review的标准检查项; 编译器-fsanitize=address可检测
- Review rule: 容器erase必须使用返回的迭代器; 遍历中修改容器应使用erase-remove idiom或手动管理迭代器

### D-855: DEVICE_NAME从common_utils导入失败 + nan_to_num算子缺少注册

- Root cause: import路径失效 + 缺少算子注册(多组件问题)
- Hashes: `bb771ba3d`
- Files: `test/test_custom_ops/test_npu_rotary_mul.py`, `test/test_custom_ops/test_npu_rotary_mul_backward.py`, `test/test_network_ops/test_nan_to_num.py`, `codegen/templates/DispatchKeyNativeFunctions.h`
- Root cause: (a) `torch_npu.testing.common_utils.DEVICE_NAME`被移除或路径变更; (b) `nan_to_num`/`nan_to_num_`/`nan_to_num_out`未在NPU dispatch key下注册
- Defect: 多个测试文件从`torch_npu.testing.common_utils`导入`DEVICE_NAME`常量，但该常量被移除(或重构到其他位置)。修复方式是改为运行时获取: `torch_npu.npu.get_device_name(0)[:10]`。同时，codegen模板中缺少`nan_to_num`系列函数的静态声明，导致NPU设备上调用`torch.nan_to_num()`时找不到NPU实现，fallback到CPU(性能差)或报错。
- Fix: (1) DEVICE_NAME改为local运行时获取; (2) 在DispatchKeyNativeFunctions.h中添加nan_to_num三个变体的声明
- Reviewability: high -- import error在CI中直接报错; 缺少dispatch注册可通过算子覆盖率测试发现
- Review rule: 测试中的共享常量应有稳定的导出路径; 新增算子dispatch时确保所有变体(普通/inplace/out)都注册

### D-856: logging模块级调用导致重复日志

- Root cause: Python logging误用(module-level vs logger instance)
- Hashes: `fc0e22faa`
- Files: `torch_npu/utils/module.py`
- Root cause: 直接调用`logging.info()`使用root logger，导致日志被所有handler重复输出
- Defect: `ddp_forward()`中`logging.info("Reducer buckets have been rebuilt...")`使用的是`logging`模块的顶层函数，这会通过root logger发出消息。如果应用配置了多个handler(常见于分布式训练)，root logger的消息会被每个handler输出一次。应使用模块级logger `logger = logging.getLogger(__name__)`，这样日志归属明确，可以独立配置level和handler。
- Fix: 模块顶部添加`logger = logging.getLogger(__name__)`; 调用改为`logger.info(...)`
- Reviewability: high -- `logging.info()` vs `logger.info()`是Python最佳实践review项
- Review rule: 永远使用`logging.getLogger(__name__)`而非直接调用`logging.info()`; 这是Python logging的基本规范

### D-857: ACL编译选项设置忽略错误返回值 + ConvHF32 API语义反转

- Root cause: 错误返回值未检查 + API语义反转(allow→forbid)
- Hashes: `46e0efa01`
- Files: `torch_npu/csrc/framework/interface/EnvVariables.cpp`, `torch_npu/csrc/aten/ops/op_api/ConvolutionKernelNpuOpApi.cpp`, `torch_npu/csrc/aten/ops/op_api/ConvolutionBackwardKernelNpuOpApi.cpp`, `torch_npu/csrc/aten/ops/op_api/ConvtbcKernelNpuOpApi.cpp`
- Root cause: (a) 所有`AclSetCompileopt()`调用都忽略返回值; (b) `IsAllowConvHF32()`被重命名为`IsForbidConvHF32()`，语义反转但调用点未取反
- Defect: `REGISTER_OPTION_HOOK`中的`AclSetCompileopt()`设置JIT_COMPILE、OP_DEBUG_LEVEL、DEBUG_DIR、COMPILER_CACHE_MODE等编译选项时，返回值被直接丢弃。如果底层ACL版本不支持某个选项或值非法，错误被静默忽略，导致用户以为配置生效但实际未生效。同时，ConvHF32相关API从`IsAllowConvHF32()`(允许HF32=true)改名为`IsForbidConvHF32()`(禁止HF32=true)，语义完全反转。所有Convolution算子中的调用需要加`!`取反: `GetCubeMathType(IsAllowConvHF32())` → `GetCubeMathType(!IsForbidConvHF32())`。
- Fix: (1) 每个AclSetCompileopt调用增加`TORCH_CHECK(ret == ACL_SUCCESS, ...)`; (2) 所有Convolution调用点加`!`取反
- Reviewability: high -- 忽略返回值是静态分析标准检查项; API重命名后的语义反转应由compiler warning或CI检测
- Review rule: ACL API调用必须检查返回值; API重命名时grep所有调用点并验证语义一致性; allow/forbid语义反转是高风险变更

### D-858: isBaseFormatType判断逻辑错误(origin==current不等于base format)

- Root cause: 谓词逻辑错误(充分条件误认为等价条件)
- Hashes: `d1c83bf17`
- Files: `torch_npu/csrc/framework/FormatHelper.cpp`
- Root cause: `origin_format_ == npu_format_`不能判断是否为base format，两者相等但都是非base format的情况是合法的
- Defect: `FormatHelper::isBaseFormatType()`原逻辑: 如果tensor的`origin_format_`等于`npu_format_`就认为是base format。这个假设在两种情况下失败: (1) tensor的origin和current都被设为非base format(如ACL_FORMAT_FRACTAL_Z)，此时两者相等但不是base format; (2) tensor经过format cast后origin被更新为当前format。正确的判断应基于format值本身: ACL_FORMAT_ND、ACL_FORMAT_NCHW、ACL_FORMAT_NHWC、ACL_FORMAT_NCDHW这四种是base format。该函数影响aclnn route的选择: base format走aclnn(OP_API)路径，非base format走legacy路径。错误判断会导致非base format tensor走错路径，产生ACL内部错误。
- Fix: 改为显式枚举base format: `format == ACL_FORMAT_ND || NCHW || NHWC || NCDHW`
- Reviewability: high -- 谓词函数的等价性错误可通过构造反例发现; 枚举比推导更可靠
- Review rule: format/type判断应基于值的枚举而非推导关系; 谓词函数需要在edge case(origin!=current)下测试

### D-859: torch._C._nn._parse_to缺少NPU设备守卫

- Root cause: monkey-patch遗漏(internal C API)
- Hashes: `42bbed92f`
- Files: `torch_npu/utils/module.py`
- Root cause: `torch._C._nn._parse_to`未被torch_npu包装，NPU设备上调用`.to()`可能不经过NPU设备守卫
- Defect: PyTorch的`Module.to()`和`Tensor.to()`内部调用`torch._C._nn._parse_to()`解析设备/dtype参数。torch_npu对大部分`.to()`路径做了monkey-patch添加NPU设备守卫(确保设备上下文正确)，但遗漏了这个内部C函数入口。某些代码路径(如直接调用`_parse_to`的第三方库)会绕过NPU设备守卫，导致在错误的设备上下文中执行。
- Fix: 添加`@torch_device_guard`装饰的wrapper函数，并monkey-patch到`torch._C._nn._parse_to`
- Reviewability: low -- _parse_to是internal API，不在常规monkey-patch清单中; 只有特定第三方库触发
- Review rule: monkey-patch清单需要覆盖所有设备相关的内部路径; 新增的upstream internal API需要定期审查是否需要patch

### D-860: NPUTensorImpl持有双份storage引用导致内存泄漏

- Root cause: 引用计数泄漏(多余的intrusive_ptr持有)
- Hashes: `f806f6daf` `ec4dc0f6d`
- Files: `torch_npu/csrc/core/NPUTensorImpl.cpp`, `torch_npu/csrc/core/NPUTensorImpl.h`, `torch_npu/csrc/aten/common/TensorFactories.cpp`
- Root cause: `NPUTensorImpl`同时持有基类`TensorImpl::storage_`和自己的`_storage_impl`两个`intrusive_ptr<StorageImpl>`，导致refcount多1
- Defect: `NPUTensorImpl`的构造函数接收`Storage`(传给基类)和额外的`intrusive_ptr<StorageImpl> storage_impl`(存入`_storage_impl`成员)。这两个指针指向同一个StorageImpl对象，使其引用计数+2。当tensor被销毁时，基类析构减1，`_storage_impl`析构减1，正常到0。但在`tensor.data`场景下: `shallow_copy_and_detach`创建新的NPUTensorImpl时又传入`this->_storage_impl`，引用计数再+1。多次`.data`调用会累积额外引用。更关键的是，析构函数中`this->_storage_impl.reset()`在基类析构之前执行，如果storage已被其他路径释放，reset会double-free。
- Fix: 移除`_storage_impl`成员; 构造函数只保留`Storage`参数; `shallow_copy_from`用基类`impl.get()`; 析构函数置空
- Reviewability: medium -- 需要理解PyTorch的TensorImpl ownership model; 双份引用在正常场景下"工作"但在edge case泄漏
- Review rule: 不要在子类中重复持有基类已管理的资源; intrusive_ptr的ownership应该是唯一的

### D-861: NPUTensorIterator类型提升遗漏BFloat16

- Root cause: 新dtype未覆盖(类型提升枚举不完整)
- Hashes: `179a28982` `60074b509`
- Files: `torch_npu/csrc/aten/mirror/NPUTensorIterator.cpp`
- Root cause: `compute_types()`中对Half有特殊处理但BFloat16没有
- Defect: `NPUTensorIterator`的类型提升逻辑对`at::ScalarType::Half`有特殊分支: 如果operand是Half则保持Half(避免不必要的type promotion)。BFloat16作为另一种16-bit浮点类型需要相同的处理，但原代码缺少`ScalarType::BFloat16`分支。结果: bf16 tensor参与计算时会被promote到float32(因为走了默认的`result_type()`路径)，导致: (1) 不必要的精度提升消耗显存; (2) 输出dtype与用户预期不符。
- Fix: 增加`if (a.scalar_type() == at::ScalarType::BFloat16) { scalar_type = at::ScalarType::BFloat16; }`
- Reviewability: high -- 新增dtype时应grep所有type switch/if链; bf16和fp16通常需要对称处理
- Review rule: 新增数据类型时搜索所有ScalarType::Half出现位置，检查是否需要对应的BFloat16分支

### D-862: NPUCachingAllocator的event处理存在race condition

- Root cause: 并发竞态(event记录与消费之间的window)
- Hashes: `96abe60a8` `e8b2e78e0` `5fc2a836f`
- Files: `torch_npu/csrc/core/npu/NPUCachingAllocator.cpp`
- Root cause: process_events()可能处理尚未被record的event(record在taskqueue线程，process在allocator线程)
- Defect: NPUCachingAllocator通过`npu_events` deque跟踪pending的ACL events。在taskqueue模式下，event的record(实际提交到device)在独立线程中执行，而process_events()在另一个线程中检查event完成状态。如果process_events先于record执行，`aclrtQueryEvent`查询的是一个尚未被record的event，结果是未定义的(可能返回"completed"导致提前释放block，或返回错误)。
- Fix: 新增`recorded_events` set(mutex保护)，event被record后加入set，process_events在处理前检查event是否在set中。如果不在且Queue模式开启，跳过该event。
- Reviewability: low -- 需要理解taskqueue线程模型和event生命周期; race window很窄，常规测试难以复现
- Review rule: 跨线程共享的event/state需要explicit synchronization; event的record和query不能假设执行顺序

### D-863: HCCL集合通信缺少BFloat16数据类型映射

- Root cause: 新dtype未注册(集合通信类型映射不完整)
- Hashes: `3bba18365` `4fe53fd51`
- Files: `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp`
- Root cause: `kScalarTypeToHcclDataType`映射表和`allReduceSupportedDataTypes`集合均缺少BFloat16
- Defect: ProcessGroupHCCL的PyTorch dtype到HCCL dtype的映射表`kScalarTypeToHcclDataType`缺少`{at::kBFloat16, HCCL_DATA_TYPE_BFP16}`条目。同时AllReduce的支持类型白名单`allReduceSupportedDataTypes`也未包含BFP16。结果: (1) 使用bf16 tensor调用`dist.all_reduce()`时，映射查找失败抛出异常; (2) 即使添加映射，AllReduce的类型检查也会拒绝BFP16。随着大模型训练普遍使用bf16，这直接阻止了bf16混合精度分布式训练。
- Fix: (1) 映射表增加`{at::kBFloat16, HCCL_DATA_TYPE_BFP16}`; (2) 反向映射增加对应条目; (3) AllReduce白名单增加`HCCL_DATA_TYPE_BFP16`
- Reviewability: high -- 映射表的完整性是review标准检查项; 新增dtype时应grep所有type map
- Review rule: dtype映射表/白名单的变更需要同步所有相关集合(forward map, reverse map, allowlist); 新dtype上线需要通信层适配checklist

<!-- D-skip: d37d2baf9 3f562da33 3f23bac00 e3f9c857c e772f64c4 9c4a3a671 — public API合规(添加__all__/allowlist/重命名internal函数), 非defect -->
<!-- D-skip: 86a877f78 0e7ed24e7 3e6731b8a d95068e48 b0533aa89 663b99587 c54560874 754e8fa8c afd303b42 65fa2a351 — "Fixed the failed test cases" 移除@unittest.skip/调整import顺序, 纯测试维护 -->
<!-- D-skip: 40f0e8766 81f163c7c ac0afdf34 38d2039e6 6829cc2a5 3bc4e98ab 912404085 9894b57e8 34ec56616 a1db8b206 f626a51d8 d11373633 416c9e2dd 1c41b7bd3 44a5278a7 f4e1f8ba1 4e09e4d49 c60422445 87aa96204 — "Fix the failed UT" 移除@unittest.skip/添加set_compile_mode配置, 纯测试配置调整 -->
<!-- D-skip: 865a7f0d2 a537711ac 8f688cdec — "A test case is fixed" 移除@unittest.skip(test_save_npu_format), 纯测试启用 -->
<!-- D-skip: 3139af083 e451b0a02 — "[Fix] Fix bug in testcase" 添加@unittest.skip(test_npu_stream_query), 纯测试禁用 -->
<!-- D-skip: be640bed2 32383d4e6 59099ac63 bbcad4677 576b792ad — codecheck(缩进/空格/空行/大括号风格), 纯代码风格 -->
<!-- D-skip: 33d545848 5768b78f6 5a11e7fc6 — "Add format_cast test case", 纯测试添加 -->
<!-- D-skip: b5143fb3d de87b3afa — ACL_OP_COMPILER_CACHE_MODE/DIR环境变量支持, 功能添加非defect -->
<!-- D-skip: 20198dff6 — test_true_divide测试重写(清理旧方法/添加新路径), 纯测试重构 -->

### D-864: HCCL内存复用重构(eraseStream正确化 + 移除recorded_events + op timeout迁移)

- Root cause: 并发架构重构(supersedes D-862的临时方案)
- Hashes: `46b6e2011` `bd9c6a307` `5287a5de5` `f893659cd`
- Files: `torch_npu/csrc/core/npu/NPUCachingAllocator.cpp`, `torch_npu/csrc/core/npu/NPUCachingAllocator.h`, `torch_npu/csrc/distributed/ProcessGroupHCCL.cpp`, `torch_npu/csrc/distributed/ProcessGroupHCCL.hpp`, `torch_npu/csrc/core/npu/NPUQueue.cpp`, `torch_npu/csrc/framework/OpParamMaker.cpp`
- Root cause: D-862引入的`recorded_events`方案虽解决了race condition但引入了额外锁开销且不够根本; `eraseStream`需要成为allocator的first-class方法
- Defect: 这是一个多组件重构commit: (1) 将`eraseStream()`从外部调用改为`DeviceCachingAllocator`类方法，使用主mutex保护，在方法内正确遍历+erase npu_events(修复D-854同类问题); (2) 移除D-862引入的`recorded_events` set和`recorded_event_mutex`，因为eraseStream现在在mutex内直接操作npu_events，不需要额外的event tracking; (3) 将`AclrtSetOpWaitTimeout`从NPUStream初始化移到ProcessGroupHCCL构造函数(因为timeout值依赖HCCL_EXEC_TIMEOUT环境变量，而该变量在HCCL初始化时才有意义); (4) NPUQueue的consumer线程从busy-wait+TTL改为eventfd阻塞通知模式。
- Fix: eraseStream提升为类方法; 移除recorded_events; timeout迁移到HCCL; consumer改用eventfd
- Reviewability: low -- 多组件联动重构需要理解allocator/HCCL/queue三层架构
- Review rule: allocator的线程安全操作应封装为类方法而非外部直接操作成员; 临时方案(如recorded_events)应有tech-debt ticket跟踪移除

<!-- D-864 addendum: e9a9cbe52 — "set op wait timeout when hccl is initializing", 是D-864中op timeout迁移的独立cherry-pick -->

### D-865: Revert format_contiguous_add_copy_optimize合并优化(引入数值回归)

- Root cause: 优化引入回归(revert)
- Hashes: `aba7acb87` `6823afe2c` `4e0517b78`
- Files: `torch_npu/csrc/aten/ops/AddKernelNpu.cpp`, `torch_npu/csrc/aten/ops/BmmKernelNpu.cpp`, `torch_npu/csrc/aten/ops/MmKernelNpu.cpp`, `torch_npu/csrc/framework/OpCommand.cpp`, `torch_npu/csrc/framework/utils/NpuUtils.cpp`
- Root cause: `format_contiguous_add_copy_optimize`合并了format转换和copy操作，但在某些tensor layout下产生错误结果
- Defect: 优化尝试将`format_contiguous()`替换为`format_contiguous_add_copy_optimize()`，后者合并了format cast和data copy为一次操作以减少内存分配和copy次数。但该优化在特定场景(如非连续tensor的matmul输入、inplace add的self参数)下产生数值错误。revert将所有调用点恢复为`format_contiguous()`。NpuUtils中添加的`format_contiguous_add_copy_optimize`实现可能在后续以更保守的方式重新引入。
- Fix: 全部恢复为`format_contiguous()`调用
- Reviewability: medium -- 优化的正确性需要在所有tensor layout组合下验证; format cast路径的副作用不明显
- Review rule: 性能优化合入前需要覆盖edge case的数值回归测试; format转换优化应有A/B数值对比验证

<!-- D-865 addendum: 9be27aaae — 同一revert的另一cherry-pick(额外包含CopyKernelNpu.cpp和TensorFactories.cpp的相关变更) -->

### D-866: Revert NPUQueue "Second-work flow"时间片分配(consumer线程调度回归)

- Root cause: 线程调度优化引入回归(revert)
- Hashes: `a182e63ee` `e8fa2c34d` `abb6b87b4` `44fc879f4` `4688f7352` `cba9c532c`
- Files: `torch_npu/csrc/core/npu/NPUQueue.cpp`, `torch_npu/csrc/core/npu/NPUQueue.h`
- Root cause: consumer线程的时间片分配策略变更导致taskqueue吞吐下降或死锁
- Defect: "Second-work flow"修改了NPUQueue的Dequeue和consumer线程逻辑: (1) 引入MAX_TTL_TIME和busy-wait超时机制; (2) 改变了空队列时的等待策略(从eventfd阻塞改为TTL轮询); (3) 修改了producer-consumer的同步时序。该优化试图让consumer线程在空闲时释放CPU时间片，但引入的TTL轮询与producer的写入通知之间存在竞态窗口，导致consumer可能在producer写入后仍在轮询中未响应。revert恢复了原始的eventfd阻塞+notify模式。
- Fix: 完整revert consumer线程调度逻辑
- Reviewability: low -- 无锁队列的线程调度优化需要形式化验证; 竞态窗口在高并发下才暴露
- Review rule: taskqueue的producer-consumer协议变更需要压力测试验证; 调度策略变更应有性能+正确性双重指标

### D-867: Revert NPUQueue时间计算函数修正(与D-866关联)

- Root cause: 时间计算修正引入回归(revert, 关联D-866)
- Hashes: `9636cb900` `87b1be0a9` `3209c3042` `7ffe768d5` `5c493f3cb` `9199e26eb`
- Files: `torch_npu/csrc/core/npu/NPUQueue.cpp`
- Root cause: 时间计算修正是D-866 "Second-work flow"的一部分，随D-866一起revert
- Defect: 仅3行变更，修正了NPUQueue中的时间计算辅助逻辑。由于与D-866的"Second-work flow"方案耦合，当D-866被revert时此修正也需要revert。
- Fix: revert时间计算变更
- Reviewability: high -- 变更本身简单，但依赖D-866的上下文
- Review rule: 关联变更应在同一PR中提交，避免部分revert导致不一致

### D-868: cumsum忽略dtype参数 + Bool输入类型处理缺失

- Root cause: 函数参数被忽略(dtype) + 类型转换遗漏(Bool→Long)
- Hashes: `65b4c6991`
- Files: `torch_npu/csrc/aten/ops/CumsumKernelNpu.cpp`
- Root cause: (a) `cumsum()`的output tensor始终使用input dtype创建，忽略了`dtype`参数; (b) Bool输入应默认输出Long(PyTorch语义)但NPU实现未处理
- Defect: PyTorch的`cumsum(input, dim, dtype=)`允许指定输出dtype(如int32输入指定float64输出)。NPU实现中`OpPreparation::ApplyTensor(self)`总是使用`self`的dtype创建output，`dtype`参数被完全忽略。同时PyTorch约定: Bool输入的cumsum应产生Long输出(因为累加可能超出bool范围)，但NPU实现输出Bool tensor(累加在Bool上只有0/1)。
- Fix: 增加`dtype.has_value()`检查使用指定dtype; Bool输入时默认用`at::kLong`; 其他情况保持self dtype
- Reviewability: high -- 参数被忽略是静态分析可检测的反模式; Bool→Long是PyTorch文档中明确的行为
- Review rule: 算子实现的所有参数都应有效果; 类型转换语义需与PyTorch reference实现对齐

### D-869: GRU forward不支持PackedSequence输入

- Root cause: monkey-patch覆盖不完整(PackedSequence路径遗漏)
- Hashes: `3606ae5ba`
- Files: `torch_npu/utils/module.py`
- Root cause: torch_npu对`GRU.forward()`的monkey-patch实现不处理`PackedSequence`输入，只处理普通tensor
- Defect: PyTorch的`GRU.forward()`接受两种输入: 普通tensor和`PackedSequence`(变长序列打包)。torch_npu的`gru_forward()`monkey-patch只实现了普通tensor路径。当用户使用`pack_padded_sequence()`打包变长序列后传入GRU，monkey-patch无法处理`PackedSequence`的解包/重打包逻辑，导致错误。原始的PyTorch GRU.forward()有完整的PackedSequence处理(解包→计算→重打包)。
- Fix: 在`gru_forward()`中增加`isinstance(orig_input, PackedSequence)`分支，实现解包→调用GRU→重打包的完整路径
- Reviewability: medium -- monkey-patch需要覆盖原函数的所有输入类型; PackedSequence是RNN模块的标准用法
- Review rule: monkey-patch替换函数签名和行为必须与原函数完全一致; RNN模块的测试需覆盖packed和unpacked两种输入

### D-870: index_put缺少shape broadcast验证 + Bool索引路径错误

- Root cause: 缺少输入验证(shape mismatch) + 条件分支逻辑错误(Bool index bypass)
- Hashes: `a49578111` `39f3848e9` `9158a1500` `a22107277` `ec4055228` `a9c484f1b`
- Files: `torch_npu/csrc/aten/ops/IndexPutKernelNpu.cpp`
- Root cause: (a) 缺少value与indexed区域的shape broadcast检查; (b) Bool类型索引的aicpu bypass逻辑不正确
- Defect: `index_put_(self, indices, value)`中: (1) 当value的shape与indexed区域不匹配且不可broadcast时，原代码不报错而是静默产生错误结果(如`x[x>-1] = tensor([1,2,3])`当x有4个元素)。PyTorch应抛出"shape mismatch"。修复增加`is_expandable_to()`验证。(2) 原代码对Bool类型索引直接return false(不走aicpu路径)，但Bool索引应该先转为int索引再分发，而非跳过整个aicpu检查。移除此特殊分支。(3) aicore路径的参数签名重构，从传递原始indices改为传递展开后的indices_expand和masks。
- Fix: (1) 增加is_expandable_to shape检查; (2) 移除Bool索引的bypass; (3) 重构aicore路径参数
- Reviewability: high -- shape mismatch缺少检查是输入验证review标准项; Bool→Int转换是已知路径
- Review rule: tensor操作的shape兼容性检查应在dispatch前完成; 索引类型转换应统一处理而非特殊case bypass

### D-871: nn.Linear NZ格式下IsMmTranspose判断过宽

- Root cause: 谓词函数过宽(IsTransposeLastTwoDims vs IsMmTranspose)
- Hashes: `d2c3b057a`
- Files: `torch_npu/csrc/framework/utils/CalcuOpUtil.cpp`, `torch_npu/csrc/framework/utils/CalcuOpUtil.h`
- Root cause: `IsTransposeLastTwoDims`对高维tensor(dim>3)也返回true，但MM/BMM的转置判断应限制在2D/3D
- Defect: `CalcuOpUtil::IsTransposeInnerAxis()`调用`IsTransposeLastTwoDims()`来判断tensor是否被转置，然后交换inner/outer axis大小。`IsTransposeLastTwoDims()`对任意维度tensor检查最后两维的stride关系，但4D+的tensor可能因为其他原因(如permute)满足该条件却不是MM语义的转置。新增`IsMmTranspose()`限定`dim >= 2 && dim <= 3`，并检查`stride(dim2)==1 && stride(dim1)==size(dim2)`这个更严格的条件(连续存储的转置)。在NZ(fractal_z)格式下，错误的转置判断导致inner axis计算错误，后续的NdToNz判断和算子dispatch产生错误结果。
- Fix: 新增`IsMmTranspose()`函数，`IsTransposeInnerAxis()`改用新函数
- Reviewability: medium -- 需要理解NZ格式下的stride语义; 谓词函数的适用范围需要明确
- Review rule: 几何/布局判断函数应明确适用的维度范围; NZ格式特有的逻辑应有专门的测试覆盖

### D-872: NPU_CHECK_SUPPORTED_OR_ERROR警告刷屏

- Root cause: 日志频率未控制(warn_once缺失)
- Hashes: `78794eb6d` `30d1d839e` `88b5c210a`
- Files: `torch_npu/csrc/core/npu/NPUException.h`
- Root cause: `ACL_ERROR_RT_FEATURE_NOT_SUPPORT`错误码的warning每次调用都输出，高频算子场景下刷屏
- Defect: `NPU_CHECK_SUPPORTED_OR_ERROR`宏在遇到`ACL_ERROR_RT_FEATURE_NOT_SUPPORT`时调用`NPU_LOGW(Feature is not supportted...)`。某些低版本CANN上，部分算子的feature检查会反复返回此错误码，导致每次算子调用都打印一次warning。在训练循环中这会产生数万行重复日志，掩盖真正有用的信息。
- Fix: 使用`static auto ... = [](){ NPU_LOGW(...); return true; }();` lambda实现warn_once语义; 同时修正"supportted"→"supported"拼写
- Reviewability: high -- 日志刷屏在任何真实运行中都会暴露; warn_once是标准模式
- Review rule: 可能在循环中被触发的warning必须使用warn_once/log_once机制; 错误码处理宏的日志频率需要控制

### D-873: sum算子integral类型处理 + 空tensor边界条件

- Root cause: 类型处理缺失 + 空tensor边界条件
- Hashes: `fc866df66` `e478f34fd`
- Files: `torch_npu/csrc/aten/ops/SumKernelNpu.cpp`
- Root cause: (a) integral类型输入应先cast到float再sum; (b) 空tensor(numel==0)的判断用逐维扫描而非直接检查numel
- Defect: `sum_out()`对整数类型输入(int8/int16/int32等)直接调用NPU的ReduceSum算子，但某些integer类型NPU不支持。PyTorch约定: integral输入的sum默认提升到float。原代码缺少此类型提升。同时，原代码用`for (int64_t i = 0; i < selfSize.size(); i++) if (selfSize[i] == 0)`逐维检查是否有0-size维度，但这等价于`self.numel() == 0`且更简洁。
- Fix: 增加`check_dtype()`函数对integral类型cast到float; 空tensor检查改为`self.numel() == 0`
- Reviewability: high -- integral→float提升是PyTorch文档中的标准语义; 空tensor处理是边界条件review标准项
- Review rule: reduce类算子需处理integral类型提升; 空tensor是必测边界条件

### D-874: baddbmm输入未做format_contiguous导致ACL错误

- Root cause: 缺少format_contiguous(非连续tensor直接传ACL)
- Hashes: `b66ea151c`
- Files: `torch_npu/csrc/aten/ops/BaddbmmKernelNpu.cpp`
- Root cause: `baddbmm`对non-transposed输入直接调用`.Input(tensor1)`，但tensor可能在NPU上是非连续的(非base format)
- Defect: `baddbmm_nocheck()`中，如果tensor1/tensor2不是transposed的(即`isSelfT`/`isMat2T`为false)，原代码直接将tensor传入OpCommand的`.Input()`。但OpCommand的`.Input()`不做format转换(不同于`.Contiguous()`或用户显式调用`format_contiguous()`)。如果tensor是NZ/FRACTAL_Z等非base format且非连续，BatchMatMul算子会收到layout不符预期的输入，产生ACL错误或错误结果。
- Fix: non-transposed路径增加`NpuUtils::format_contiguous(tensor)`; 使用`.InputWithoutContiguous()`替代`.Input()`(因为已经做了contiguous)
- Reviewability: high -- 非base format tensor传入ACL算子是torch_npu中的常见defect pattern
- Review rule: 所有ACL算子调用的tensor输入必须确保是contiguous的(或在OpCommand内部处理); 新增算子时应有format_contiguous checklist

### D-875: torch.std在shape_prod=0时返回NaN的条件不完整

- Root cause: 边界条件遗漏(空维度 + correction >= shape_prod)
- Hashes: `bd44ded85` `4f3c4e69f`
- Files: `torch_npu/csrc/aten/ops/StdKernelNpu.cpp`
- Root cause: `shape_prod == 0`的情况未被处理，只处理了`shape_prod == 1 && shape_prod <= correction`
- Defect: `std_mean_out_npu_nocheck()`在计算标准差时，如果沿指定dim的元素数为0(`shape_prod == 0`)，应返回NaN(数学上0个元素的标准差无定义)。原代码只检查`shape_prod == 1 && shape_prod <= correction`(即单元素且correction>=1时返回NaN)，遗漏了0元素的情况。同时，原逻辑`shape_prod == 1 && shape_prod <= correction`中`shape_prod == 1`和`shape_prod <= correction`是独立条件，改为`shape_prod == 0 || (shape_prod == 1 && shape_prod <= correction)`覆盖两种情况。
- Fix: 增加`shape_prod == 0`条件; 用`||`连接两个NaN返回条件
- Reviewability: high -- 0-size维度是标准边界条件; 数学语义(0元素的std=NaN)是确定性的
- Review rule: 统计类算子需处理0-size维度; correction参数对边界case的影响需要在spec中明确

### D-876: nll_loss2d_forward的output_size计算用原始self维度而非reshape后的

- Root cause: reshape后仍用原始维度计算output_size
- Hashes: `afdf16c9b` `dd1eb5e21` `4964f5349`
- Files: `torch_npu/csrc/aten/ops/loss/NLLLoss2dKernelNpu.cpp`
- Root cause: (a) output_size计算使用reshape前的`self`而非reshape后的`self_input`; (b) reduction=None时output_size包含多余的空间维度
- Defect: `nll_loss2d_forward()`将4D input `[N,C,H,W]`先`permute({0,2,3,1})`再`reshape({-1, C})`变为2D `[N*H*W, C]`。但`nll_loss2d_npu_output_size()`仍使用原始4D `self`计算output_size，在reduction=None时返回`{self.size(0), self.size(2), self.size(3)}`即`{N,H,W}`。但NLLLoss在PyTorch中对2D input(after reshape)的reduction=None应输出`{N*H*W}`即1D(或者按原始NLLLoss2d语义应为`{N}`batch维度)。原代码输出`{N,H,W}`三维是错误的。修复后改为`{self.size(0)}`即batch维度。
- Fix: (1) output_size函数传入reshape后的`self_input`; (2) reduction=None时output_size改为`{self.size(0)}`
- Reviewability: medium -- 需要理解NLLLoss2d与NLLLoss的关系以及reshape后的维度语义
- Review rule: output_size计算必须基于实际传入算子的tensor shape; reshape/permute后应更新所有引用原shape的地方

### D-877: codegen的TensorList转换使用rbegin导致元素反序

- Root cause: STL迭代器方向错误(rbegin vs begin)
- Hashes: `65ecb9ee6`
- Files: `codegen/dest/utils.py`
- Root cause: `std::transform`的output iterator使用`rbegin()`(反向迭代器)而非`begin()`
- Defect: codegen生成的wrapper函数需要将NPU tensor list转换为CPU tensor list。`std::transform(input.begin(), input.end(), output.rbegin(), ...)`将input正序遍历但写入output的反向位置，导致输出列表元素顺序被反转。对于TensorList输入(如concat的inputs、index_put的indices)，元素顺序反转会导致语义完全错误但不会报错(因为shape可能兼容)���这个bug影响所有通过codegen生成的CPU fallback wrapper函数。
- Fix: `.rbegin()` → `.begin()`; 同时修复返回值转换中相同的rbegin错误
- Reviewability: high -- rbegin vs begin是代码review基本检查; `std::transform`的迭代器方向应保持一致
- Review rule: `std::transform`的input和output迭代器方向必须匹配; codegen生成的代码同样需要review

### D-878: dot算子输出shape应为0-d标量而非1-d向量

- Root cause: 输出维度错误(1d → 0d)
- Hashes: `26f55250f` `1e82ba7e1` `90cf09bab`
- Files: `torch_npu/csrc/aten/ops/DotKernelNpu.cpp`
- Root cause: `dot_npu_output_size()`返回1-d shape而非空shape(0-d scalar)
- Defect: `torch.dot(a, b)`的数学定义是向量点积，结果是一个标量(0维tensor)。原NPU实现用`dot_npu_output_size()`计算output_size，返回包含1个元素的1-d tensor。后续用`result.resize_({})`强制reshape为0-d，但这个resize_在某些场景下可能失败或引入额外开销。正确做法是直接创建0-d tensor: `output_size = {}`。
- Fix: output_size直接设为空`{}`(0-d tensor); 移除后续的`result.resize_({})`
- Reviewability: high -- dot product返回scalar是数学常识; output_size={}表示0-d tensor是PyTorch约定
- Review rule: 标量结果的算子(dot, trace, det等)output_size必须为空(0-d); 避免先创建1-d再resize为0-d

<!-- D-skip: e9c033771 — [Feature] check compile option, 功能添加(编译选项检查) -->
<!-- D-skip: f13d3d0d2 — Fix test_distributed, 纯测试变更(test_distributed.py) -->
<!-- D-skip: 704133f50 738ed8cdf 50ff5a32d 704191b4e — aclrtMallocAlign32 replace aclrtMalloc, 功能替换(对齐分配器) -->
<!-- D-skip: ee4d612f8 3efa2031d — update aclrtUtilizationInfo, 功能更新(利用率信息接口) -->
<!-- D-skip: f3bc0d8a8 — change print address style, 纯代码风格(地址打印格式) -->
<!-- D-skip: b3ae0c2df 8666e23d4 532d78016 — NPU_LOGW output warning to screen, 日志基础设施变更(warning输出方式) -->

### D-879: GIL持有导致NPUQueue死锁

- Root cause: GIL-NPU线程死锁
- Hashes:
  0d27919b9 3545252c1 3eabfefbd 89c641229
- Files: torch_npu/csrc/core/npu/NPUQueue.cpp
- Defect: `MakeSureQueueEmpty()`在循环内用`Py_BEGIN_ALLOW_THREADS/Py_END_ALLOW_THREADS`释放GIL，
  但释放发生在`IsEmptyQueue()` double-check之后的`eventfd_read`调用处。当ACL线程在执行算子时
  触发TE算子编译（需要获取GIL），而主线程在循环的其他位置（如`need_empty`赋值、
  `__sync_synchronize()`屏障）仍持有GIL，形成死锁。
- Fix: 将GIL释放提升到整个等待循环之前（`PyEval_SaveThread`），循环结束或出错后再
  恢复（`PyEval_RestoreThread`）。消除了循环内反复acquire/release GIL的开销和时序窗口。
- Reviewability: medium -- 需要理解GIL与ACL线程的交互模型
- Review rule: 任何在等待NPU异步操作完成的阻塞路径上，检查是否持有GIL。

### D-880: upsample_nearest1d在dynamic shape场景下失败

- Root cause: 算子动态shape不支持
- Hashes:
  0d2b78ca2 3356f3701
- Files: torch_npu/csrc/aten/ops/UpsampleNearest1dKernelNpu.cpp,
  torch_npu/csrc/aten/ops/UpsampleNearest1dBackwardKernelNpu.cpp
- Defect: 原实现统一使用`Resize`/`ResizeGrad`算子，该算子不支持dynamic shape。在动态shape
  场景下（如编译器优化、symbolic shape）产生错误结果或crash。
- Fix: 对fp32/fp16使用`ResizeNearestNeighborV2`/`ResizeNearestNeighborV2Grad`算子
  （支持dynamic shape），对其他dtype保留原`Resize`/`ResizeGrad`路径。
- Reviewability: low -- 需要了解底层算子的shape约束
- Review rule: 新增算子适配时验证dynamic shape场景。

### D-881: Pad backward在padding全0时输出错误

- Root cause: 边界条件未处理（padding=0）
- Hashes:
  0d4775157 5d82b479c 604f016b2 d3b4b871e
- Files: torch_npu/csrc/aten/ops/ReflectionPad2dBackwardKernelNpu.cpp,
  ReplicationPad1dBackwardKernelNpu.cpp, ReplicationPad1dKernelNpu.cpp,
  ReflectionPad1dKernelNpu.cpp
- Defect: 多个padding backward kernel在padding参数全为0时行为异常。NPU算子对zero-padding
  边界条件处理不正确，导致梯度计算错误。
- Fix: 重写backward kernel适配逻辑，添加padding=[0,0,0,0]和padding=0的测试用例。
- Reviewability: high -- padding=0是最基本的边界条件
- Review rule: padding/stride/dilation类参数必须测试全0和单边0。

### D-882: recordStream错误剪枝导致use-after-free

- Root cause: 错误的优化剪枝（内存分配器stream追踪）
- Hashes:
  0e2a0a80e 47a03bbb0 891e2d00d a70950331
- Files: torch_npu/csrc/core/npu/NPUCachingAllocator.cpp
- Defect: `recordStream`中有优化：当目标stream等于block的分配stream时early return跳过记录。
  但block可能被回收后在相同stream上重新分配给不同tensor，此时旧的stream使用记录丢失，
  导致内存被提前释放引发use-after-free。
- Fix: 删除early return优化，始终记录stream使用。
- Reviewability: medium -- 需要理解allocator的block生命周期
- Review rule: 内存分配器的stream追踪不应假设stream identity意味着安全。

### D-883: clamp算子缺少int64支持

- Root cause: 遗漏dtype支持
- Hashes:
  0edce2564 667cead43 cf493af9d e46b4eafe
- Files: torch_npu/csrc/aten/ops/ClampKernelNpu.cpp
- Defect: clamp算子只支持float32/float16/int32，缺少int64（Long）类型的处理。
- Fix: 在测试和实现中添加int64类型支持。
- Reviewability: high -- dtype覆盖是标准检查项
- Review rule: 新增算子时对照PyTorch原生支持的dtype列表逐一验证。

### D-884: isinstance(obj, torch.device)对NPU设备返回False

- Root cause: 类型系统适配遗漏（device monkey-patch）
- Hashes:
  0ee3963ed 3593e4865 56d1f2db3
- Files: torch_npu/__init__.py
- Defect: `_isinstance`覆盖中，当obj是tuple类型的NPU device时（如`torch.device('npu', 0)`
  返回的命名元组），直接调用`builtin_isinstance(obj, torch._C.device)`会返回False。
- Fix: 在isinstance检查中识别tuple形式的NPU device，转换为正确的device对象后再比较。
- Reviewability: medium -- 需要了解NPU device的内部表示
- Review rule: device类型兼容性测试应覆盖str/int/tuple/torch.device所有构造方式。

### D-885: all_gather_object硬编码HCCL backend行为

- Root cause: 硬编码backend假设
- Hashes:
  0f009733f ecee54117
- Files: torch_npu/distributed/distributed_c10d.py
- Defect: `all_gather_object`无条件执行HCCL特有操作：将tensor移至NPU设备、将int64转为int32
  （HCCL不支持int64）。当使用其他backend（如Gloo）时这些操作导致功能异常。
- Fix: 添加`_check_for_hccl_backend()`检查，仅在HCCL backend时执行NPU特有的
  dtype转换和device迁移。
- Reviewability: high -- 硬编码的device/dtype转换应该很显眼
- Review rule: distributed函数中的backend-specific逻辑必须有backend类型守卫。

<!-- D-skip: 0f28eb1ea | fix the conflict | merge conflict resolution, 非defect -->

### D-886: Module.to中kwargs device类型假设错误

- Root cause: 类型假设错误（str vs device对象）
- Hashes:
  0f30ce7c9 520a201d8
- Files: torch_npu/utils/module.py
- Defect: `to()`覆盖中用`'npu' in kwargs.get("device", "")`检查device参数。但`kwargs["device"]`
  可能是`torch.device`对象而非字符串，`in`操作符对非字符串类型会抛TypeError。
- Fix: 用`str()`包装：`'npu' in str(kwargs.get("device", ""))`。
- Reviewability: high -- 基本的类型安全检查
- Review rule: device参数在检查前必须转为str，因为用户可能传入str/int/torch.device任一类型。

### D-887: HCCL dtype映射不完整导致通信报错

- Root cause: 遗漏dtype支持（HCCL通信层）
- Hashes:
  0fa9e7353
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp,
  third_party/hccl/inc/hccl/hccl_types.h,
  torch_npu/distributed/hccl_dtype_wraper.py
- Defect: HCCL的`hcclDataType`映射表缺少uint8/uint64/fp64/bfp16等类型，且AllReduce/
  ReduceScatter等操作缺少dtype支持检查，导致传入不支持的dtype时产生未定义行为而非清晰报错。
- Fix: 扩展`HcclDataType`枚举和映射表，添加`checkSupportedDataTypeOfAllReduce()`
  等验证函数。
- Reviewability: high -- dtype映射表扩展是标准变更
- Review rule: 新增HCCL dtype时需同步更新：枚举定义、映射表、Python wrapper、操作级检查。

<!-- D-skip: 101a0db75 | add device utilizationRate | feature addition, 非defect -->

### D-888: upsample_linear1d backward缺失属性 + var kernel缺失include

- Root cause: 算子适配遗漏（属性/依赖）
- Hashes:
  101edaa28
- Files: torch_npu/csrc/aten/ops/UpsampleLinear1dBackwardKernelNpu.cpp,
  torch_npu/csrc/aten/ops/VarKernelNpu.cpp
- Defect: upsample_linear1d backward缺少必要的Op属性设置；var kernel缺失头文件include。
  两者都是算子适配时的遗漏。
- Fix: 补充缺失的属性和include。
- Reviewability: high -- 编译或UT应能暴露
- Review rule: 算子适配提交需通过完整UT验证。

### D-889: cdist算子p=infinity时NPU报错

- Root cause: 算子参数范围不支持（infinity值）
- Hashes:
  1022bc44c
- Files: torch_npu/csrc/aten/ops/CdistKernelNpu.cpp
- Defect: `_cdist_forward`将double类型的p转为float时，对`std::isinf(p)`的情况使用
  `std::numeric_limits<float>::infinity()`，但NPU算子不接受float infinity作为参数值。
- Fix: 用`-1`作为sentinel值代替infinity，NPU算子内部约定-1表示inf范数。
- Reviewability: medium -- 需要了解NPU算子对inf的约定
- Review rule: NPU算子不一定支持IEEE特殊值（inf/nan），适配时需确认参数约束。

### D-890: binary_cross_entropy_with_logits_backward注册配置有误

- Root cause: 算子注册配置错误
- Hashes:
  104170bf2
- Files: torch_npu/csrc/aten/npu_native_functions.yaml
- Defect: yaml中backward函数的注册配置不正确（缺少op_api标记或参数映射），导致使用weight/
  pos_weight参数时backward计算结果错误。
- Fix: 修正yaml配置，添加module-based测试覆盖weight/pos_weight/reduction各组合。
- Reviewability: medium -- yaml配置review需要对照实现
- Review rule: 带可选参数的loss函数必须测试所有参数组合的backward。

### D-891: new_empty传入int参数时crash

- Root cause: 参数类型适配不完整
- Hashes:
  10bbdb0b7
- Files: torch_npu/utils/tensor_methods.py
- Defect: `_new_empty`覆盖直接将`args[0]`传给底层C API，但PyTorch允许
  `tensor.new_empty(3)`（单个int表示1-D大小），此时args[0]是int而非tuple，C API期望tuple。
- Fix: 检查`isinstance(args[0], int)`，将int包装为单元素tuple。
- Reviewability: high -- 标准的参数适配检查
- Review rule: monkey-patch PyTorch API时需覆盖所有合法的参数形式。

### D-892: Lt (Less)算子缺少Long类型支持

- Root cause: 遗漏dtype支持（比较算子）
- Hashes:
  10c6a8585
- Files: torch_npu/csrc/aten/ops/LtKernelNpu.cpp
- Defect: `lt_out_npu_nocheck`中对Int/Bool类型做了float转换以适配NPU算子，但遗漏了Long类型。
  传入LongTensor时不做转换，NPU算子直接处理int64会出错。
- Fix: 在类型检查条件中添加`ScalarType::Long`。
- Reviewability: high -- dtype条件列表review时应对照完整类型清单
- Review rule: 比较算子的dtype转换逻辑应覆盖所有整数类型和bool。

### D-893: transfer_to_npu遗漏cuda.random patch

- Root cause: monkey-patch遗漏
- Hashes:
  115dae2a8 ba794130f eb2ac96c2
- Files: torch_npu/contrib/transfer_to_npu.py
- Defect: `patch_cuda()`的patch列表中遗漏了`['cuda.random', torch_npu.npu.random]`，
  导致使用`torch.cuda.random`的代码在transfer_to_npu模式下不会被重定向到NPU。
- Fix: 在patch列表中添加cuda.random映射。
- Reviewability: high -- patch列表遗漏可通过对照cuda子模块清单发现
- Review rule: transfer_to_npu的patch列表变更需对照torch.cuda的完整子模块列表。

### D-894: device对象多态导致多处类型错误

- Root cause: 类型系统适配遗漏（device对象多态）
- Hashes:
  11a6a3ba0 3633a172e 632ecf49e e57a52fe9
- Files: torch_npu/npu/utils.py, torch_npu/npu/random.py,
  torch_npu/npu/amp/grad_scaler.py, torch_npu/utils/serialization.py
- Defect: 多处代码用`torch.device(device)`构造device对象，但当device参数已经是device对象时
  （尤其是NPU自定义的device类型），构造会失败。`isinstance`检查也未覆盖`torch._C.device`类型。
  涉及4个文件的同一类问题：device参数未做str()转换，isinstance未考虑_C.device类型。
- Fix: 用`torch.device(str(device))`包装；isinstance检查添加`torch._C.device`类型。
- Reviewability: high -- 系统性问题，一次grep即可发现所有实例
- Review rule: 所有接受device参数的函数，在构造torch.device前必须str()转换。

### D-895: rrelu_with_noise_ inplace版本调用了错误的API名称

- Root cause: 错误的API名称（inplace命名约定）
- Hashes:
  11b7b9a01
- Files: torch_npu/csrc/aten/ops/op_api/RReluWithNoiseKernelNpuOpApi.cpp,
  torch_npu/csrc/aten/ops/op_api/AddmmKernelNpuOpApi.cpp
- Defect: `rrelu_with_noise_`（inplace版本）的`DO_COMPATIBILITY`宏中检查的API名称是
  `aclnnRReluWithNoise`，但inplace版本应该用`aclnnInplaceRReluWithNoise`。
  导致在aclnn可用时走了错误的兼容性分支。
- Fix: 将API名称改为`aclnnInplaceRReluWithNoise`。
- Reviewability: high -- inplace算子的aclnn名称有明确的`Inplace`前缀约定
- Review rule: inplace算子的DO_COMPATIBILITY必须使用aclnnInplace*命名。

### D-896: aclrtSetOpWaitTimeout API可用性检查缺失

- Root cause: API可用性检查缺失
- Hashes:
  11fc2307f a577007cd
- Files: torch_npu/csrc/core/npu/NPUStream.cpp
- Defect: `initGlobalStreamState`中用`C10_NPU_CHECK`调用`aclrtSetOpWaitTimeout`，
  但该API在某些CANN版本/平台上不存在。`C10_NPU_CHECK`在API不存在时直接abort。
- Fix: 改用`NPU_CHECK_SUPPORTED_OR_ERROR`，允许API不存在时优雅跳过。
- Reviewability: high -- 可选API应始终使用SUPPORTED_OR_ERROR
- Review rule: 非核心/新增的ACL API调用必须使用NPU_CHECK_SUPPORTED_OR_ERROR。

### D-897: overflow switch在初始化时无条件开启

- Root cause: 初始化逻辑错误（无条件enable）
- Hashes:
  12068adce ae5da959b
- Files: torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp,
  torch_npu/csrc/core/OverflowUtils.cpp
- Defect: 910B1+芯片的overflow检测开关在`InitializeDevice`中无条件启用
  （`AclrtSetStreamOverflowSwitch(stream, 1)`），无论用户是否需要overflow检测。
  这引入了不必要的性能开销。
- Fix: 从初始化中删除无条件enable，改为通过`EnableOverflowNpu()`接口按需开启。
- Reviewability: medium -- 需要理解overflow检测的性能影响
- Review rule: 设备初始化不应无条件开启可选的检测/调试特性。

### D-898: Module.npu(device)的device参数被忽略但不报告

- Root cause: API语义不一致
- Hashes:
  1265f8a72
- Files: torch_npu/utils/module.py
- Defect: `npu(self, device=None)`接受device参数但早期版本在device不为None时尝试使用它，
  而实际实现不支持指定特定device（多device场景不支持）。传入device参数导致行为不一致。
- Fix: 忽略device参数，始终使用`torch.device("npu")`。
- Reviewability: medium -- API接口的实际行为与签名不一致
- Review rule: 不支持的参数应从签名中移除或在文档中明确标注。

### D-899: clear_npu_overflow_flag对910B1+错误跳过清除

- Root cause: 错误的条件分支（过早返回）
- Hashes:
  128eca241 cedde8a21
- Files: torch_npu/npu/utils.py
- Defect: `clear_npu_overflow_flag`中对soc_version >= 220（Ascend910B1+）直接print提示
  并return，不执行清除操作。但实际上910B1+仍需要清除overflow flag，只是清除机制不同。
- Fix: 删除early return，所有芯片版本都执行float_status清除。
- Reviewability: medium -- 需要确认不同芯片的overflow清除需求
- Review rule: 芯片版本分支的early return需要明确文档说明为什么可以跳过。

### D-900: mul_out输出tensor类型不正确

- Root cause: 输出类型推断错误
- Hashes:
  139aebde1 1db43a649 632e9fc09 a7eae332b
- Files: torch_npu/csrc/aten/ops/MulKernelNpu.cpp
- Defect: `mul_out`的输出tensor类型推断逻辑不正确，未正确处理混合dtype输入（如int*float）
  时的输出类型提升。
- Fix: 修正类型推断逻辑，重构mul kernel实现。
- Reviewability: high -- 类型提升规则有明确的PyTorch规范
- Review rule: out变体的算子必须验证输出dtype与输入dtype组合的正确性。

### D-901: NZ format不支持scalar和rank=1 tensor

- Root cause: 格式推断遗漏（NZ format维度限制）
- Hashes:
  13a03f75f
- Files: torch_npu/csrc/aten/ops/MeanKernelNpu.cpp,
  torch_npu/csrc/framework/InferFormat.cpp
- Defect: `GuessStorageFormat`和mean算子未检查tensor维度，对scalar（rank=0）和rank=1
  tensor也尝试使用FRACTAL_NZ格式。NZ格式要求至少rank=2。
- Fix: 在`GuessStorageFormat`中添加`size.size() < 2`的检查，回退到ACL_FORMAT_ND。
  mean算子中将`outputSize.empty()`检查改为`outputSize.size() < 2`。
- Reviewability: high -- NZ的rank约束是已知的格式限制
- Review rule: 任何可能产生scalar/1-D输出的算子路径需检查format回退。

### D-902: IndexPut异步拷贝masks导致时序问题

- Root cause: 异步拷贝时序问题
- Hashes:
  13e74f596
- Files: torch_npu/csrc/aten/ops/IndexPutKernelNpu.cpp
- Defect: IndexPut适配中用`CalcuOpUtil::copy_tensor_host_to_device`手动异步拷贝masks tensor
  到device，但拷贝与算子执行之间没有同步保证，masks可能未就绪时算子就开始执行。
- Fix: 改用`Input(masks, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)`，
  让OpCommand框架管理host tensor的拷贝和同步。
- Reviewability: medium -- 需要理解OpCommand框架的拷贝语义
- Review rule: 禁止在OpCommand外手动做host->device拷贝，应使用Input的host compile模式。

### D-903: 比较算子cross-device场景处理缺失

- Root cause: cross-device算子处理缺失
- Hashes:
  140e658bf d8fc7fbe0
- Files: torch_npu/csrc/aten/ops/EqKernelNpu.cpp, GeKernelNpu.cpp,
  GtKernelNpu.cpp, LeKernelNpu.cpp, LtKernelNpu.cpp, NeKernelNpu.cpp
- Defect: eq/ge/gt/le/lt/ne六个比较算子在输入tensor位于不同device（如一个在NPU一个在CPU）时
  行为不正确。PyTorch原生支持cross-device比较（自动搬移），但NPU适配未处理此场景。
- Fix: 在kernel中添加cross-device处理逻辑，修正测试中的device断言。
- Reviewability: high -- cross-device是标准测试场景
- Review rule: 接受多个tensor输入的算子必须测试cross-device场景。

### D-904: new_empty/new_zeros的kwargs传参crash（D-891后续修复）

- Root cause: 参数类型适配不完整（后续修复）
- Hashes:
  148f00563 abe363894 ac8caded5
- Files: torch_npu/utils/tensor_methods.py
- Defect: D-891修复了`args[0]`为int的情况，但遗漏了args为空的场景（size通过kwargs传入，
  如`new_empty(size=(2,3))`），此时`args[0]`直接IndexError。
- Fix: 添加`if args`前置检查，仅在args非空时检查args[0]类型。
- Reviewability: high -- 修复后应验证所有调用方式
- Review rule: monkey-patch修复时需回归测试所有参数传递方式（positional, keyword, mixed）。

### D-905: logical_not输出format硬编码为NCHW

- Root cause: 输出format硬编码错误
- Hashes:
  14ffdaadc
- Files: torch_npu/csrc/aten/ops/LogicalNotKernelNpu.cpp
- Defect: `logical_not_out`和`logical_not`硬编码输出format为`ACL_FORMAT_NCHW`，忽略输入
  tensor的实际format。当输入为其他format（如NZ）时产生format不匹配。
- Fix: `logical_not_out`改用`CalcuOpUtil::get_tensor_npu_format(self)`获取输入format；
  `logical_not`改用`ApplyTensor`从输入tensor继承format。
- Reviewability: high -- 硬编码format是明显的code smell
- Review rule: 算子输出format应从输入继承或根据算子特性推断，禁止硬编码NCHW。

### D-906: sum op_api过度手动处理导致多处逻辑错误

- Root cause: 过度手动处理导致逻辑错误
- Hashes:
  154cc4cc9 418d67c2f 736131835 a23d65492 ce1c5b585
- Files: torch_npu/csrc/aten/ops/op_api/SumKernelNpuOpApi.cpp
- Defect: sum的op_api实现中包含大量不必要的手动处理：手动检测empty tensor、手动int->float
  转换、手动结果dtype转换。这些手动逻辑本身存在多处bug（如empty检查遗漏维度、类型转换丢失精度），
  而框架已经能正确处理这些场景。
- Fix: 删除所有手动处理逻辑，简化为直接调用`EXEC_NPU_CMD(aclnnReduceSum, ...)`，
  让框架处理类型转换和输出准备。
- Reviewability: high -- 框架已有的功能不应在算子层重复实现
- Review rule: op_api实现应尽量精简，信任框架的CheckOut和类型处理能力。

### D-907: include路径错误导致构建失败

- Root cause: 构建错误（include路径）
- Hashes:
  157c98eb1
- Files: torch_npu/csrc/framework/OpParamMaker.h
- Defect: `#include "c10/npu/OptionsManager.h"`路径在代码重构后失效，
  正确路径是`torch_npu/csrc/core/npu/register/OptionsManager.h`。
- Fix: 修正include路径。
- Reviewability: high -- 编译即可发现
- Review rule: 代码重构移动文件后需grep全局include引用。

### D-908: BatchNorm的num_batches_tracked用int64导致NPU不兼容

- Root cause: 遗漏dtype适配（NPU int64限制）
- Hashes:
  15bb908e6
- Files: torch_npu/utils/module.py
- Defect: PyTorch原生`_NormBase.__init__`中`num_batches_tracked`使用int64（默认），
  但NPU对某些int64操作不支持。
- Fix: 覆盖`_NormBase.__init__`和`_load_from_state_dict`，将`num_batches_tracked`
  的dtype从int64改为int32。
- Reviewability: medium -- 需要了解NPU的int64限制
- Review rule: 框架层的buffer/parameter默认dtype需验证NPU兼容性。

### D-909: nms_rotated输出tensor构造对象错误

- Root cause: 输出tensor构造错误
- Hashes:
  160a8a03f
- Files: torch_npu/csrc/aten/ops/NmsRotatedKernelNpu.cpp
- Defect: `npu_nms_rotated`使用了错误的ApplyTensor参数构造输出tensor `selectedBox`，
  导致输出结果不正确。
- Fix: 修正ApplyTensor调用参数。添加完整的CPU参考实现和详尽测试。
- Reviewability: medium -- 需要理解ApplyTensor的参数语义
- Review rule: ApplyTensor构造输出tensor时需仔细核对shape/dtype/options参数。

### D-910: argmin算子dim参数类型错误（vector vs scalar）

- Root cause: 算子参数类型错误（vector vs scalar）
- Hashes:
  1721f1534 43c766bc8 7630784a3 c3496da62
- Files: torch_npu/csrc/aten/ops/ArgminKernelNpu.cpp
- Defect: `ArgMin`算子期望dim参数为Scalar类型输入，但代码传入了`SmallVector<int64_t, N>`
  （长度为1的向量）。NPU算子对vector和scalar输入的处理路径不同，传错类型导致结果错误。
- Fix: 从`SmallVector`改为`c10::Scalar`传入dim。
- Reviewability: high -- 算子接口文档应明确参数类型
- Review rule: 查阅NPU算子接口文档确认每个Input的类型约束（tensor/scalar/vector）。

<!-- D-skip: 17689d60b | (empty message) | test config cleanup, 非defect -->

### D-911: 分布式场景PTA版本不一致无诊断信息

- Root cause: 版本一致性检查缺失
- Hashes:
  177dfbdab 2aec3fbd4 6a9436c77 e81428eab
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp, CMakeLists.txt
- Defect: 不同rank节点使用不同版本的PTA库时，分布式通信产生静默错误（数据格式不匹配、
  API行为差异），没有版本校验机制提供诊断信息。
- Fix: 在`broadcastMasterID`时通过store广播PTA版本和编译日期，非rank0节点对比
  版本信息并`TORCH_WARN`警告不匹配。
- Reviewability: medium -- 部署问题，代码review不易预见
- Review rule: 分布式初始化应包含版本/配置一致性检查。

### D-912: as_strided输出函数缺少return语句

- Root cause: 缺少return语句
- Hashes:
  17c239a06
- Files: torch_npu/csrc/aten/ops/AsStridedKernelNpu.cpp
- Defect: `stride_copy_out_npu_nocheck`函数签名返回`at::Tensor&`但函数体没有return语句，
  导致未定义行为。
- Fix: 添加`return result;`。
- Reviewability: high -- 编译器`-Wreturn-type` warning应能捕获
- Review rule: CI必须启用`-Werror=return-type`。

### D-913: ones_out缺少resize导致输出size不匹配

- Root cause: 缺少output resize
- Hashes:
  17f8cece9
- Files: torch_npu/csrc/aten/ops/OnesKernelNpu.cpp
- Defect: `ones_out(size, result)`直接调用`one_(result)`填充全1，但没有先`resize_`到目标size。
  当result tensor的现有size与目标size不同时输出错误。
- Fix: 在`one_`前添加`result.resize_(size)`。
- Reviewability: high -- out变体必须处理output tensor的size
- Review rule: out变体的实现第一步应是resize output到目标size。

### D-914: NPUQueue竞态 + CopyParas默认值引用已删除枚举

- Root cause: 枚举值失效 + 队列竞态条件
- Hashes:
  180377bc3 5b67a3746 6f147f05a 81cd832b5 9edb9058f
- Files: torch_npu/csrc/core/npu/NPUQueue.cpp,
  torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h,
  torch_npu/csrc/framework/OpParamMaker.cpp
- Defect: (1) `CopyParas`的默认`kind`使用`RESERVED`枚举值，但该值从`aclrtMemcpyKind`中被删除，
  导致编译失败或运行时异常。(2) `ReadQueue`中producer-consumer存在竞态：write_idx更新但数据
  未完全写入时consumer就开始读取。
- Fix: (1)默认值改为`ACL_MEMCPY_HOST_TO_HOST`。(2)添加`__sync_synchronize()`
  内存屏障，queueLen==1时添加`usleep(2)`等待数据就绪。
- Reviewability: medium -- 枚举变更编译可见；竞态需要并发分析
- Review rule: 依赖外部枚举的默认值需在头文件变更时同步审查。

### D-915: scatter算子fp16精度不足

- Root cause: fp16精度问题（算子端未做类型提升）
- Hashes:
  182ff768d b595c7afd
- Files: torch_npu/csrc/aten/ops/ScatterKernelNpu.cpp
- Defect: scatter算子在fp16模式下直接使用fp16精度计算，但CPU端已将fp16提升到fp32计算。
  NPU端未做同样的类型提升导致精度差异超出容忍范围。
- Fix: 当输入为fp16时在NPU端也转为fp32计算，结果再转回fp16。
- Reviewability: high -- CPU/NPU精度对齐是标准检查项
- Review rule: 涉及累加/比较的算子在fp16时应考虑提升到fp32计算。

### D-916: profiler头文件在libtorch模式下不应include

- Root cause: 条件编译错误（libtorch）
- Hashes:
  185b7b4e0
- Files: torch_npu/csrc/framework/OpParamMaker.cpp
- Defect: `e2e_profiler.h`和`Python.h`在BUILD_LIBTORCH模式下不存在，但include语句在
  `#ifndef BUILD_LIBTORCH`守卫之外，导致libtorch构建失败。
- Fix: 将这些include移入`#ifndef BUILD_LIBTORCH`条件编译块内。
- Reviewability: high -- 条件编译守卫位置检查
- Review rule: Python依赖和profiler代码必须在BUILD_LIBTORCH守卫内。

<!-- D-skip: 18d61921d 30130645d 37f362be9 | nms_rotated ut fix | 纯测试修复 -->

### D-917: profiler memory bean数据类型和列顺序错误

- Root cause: 数据类型假设错误 + 列顺序不匹配
- Hashes:
  1912c24b2 1afeb923b a92ab0ce5
- Files: torch_npu/profiler/analysis/prof_bean/ge_memory_record_bean.py,
  torch_npu/profiler/analysis/prof_bean/ge_op_memory_bean.py
- Defect: (1) `time_us`属性用`int()`解析timestamp，但实际数据包含浮点值，导致ValueError。
  (2) `GeOpMemoryBean.HEADERS`中"Device Type"列位置与实际数据列不对应。
- Fix: (1)改用`float()`解析。(2)调整HEADERS中Device Type到正确位置。
- Reviewability: high -- 数据格式与解析代码应严格对应
- Review rule: profiler数据bean的property类型需与数据源的实际类型一致。

### D-918: nms_rotated中`.size()`误用为属性访问

- Root cause: API用法错误（size() vs size(0)）
- Hashes:
  1a935d0e9
- Files: torch_npu/csrc/aten/ops/NmsRotatedKernelNpu.cpp
- Defect: `selectedIndex.size()[0]`应为`selectedIndex.size(0)`。`.size()`返回IntArrayRef，
  对其使用`[]`在某些上下文中会导致编译错误或行为不正确。
- Fix: 改用`.size(0)`直接获取第0维大小。
- Reviewability: high -- 标准API用法
- Review rule: 获取单维大小用`size(dim)`而非`size()[dim]`。

### D-919: mse_loss在reduction模式下输出format错误

- Root cause: 输出tensor format错误
- Hashes:
  1b48c38c3
- Files: torch_npu/csrc/aten/ops/MseLossKernelNpu.cpp
- Defect: `mse_loss`在reduction!=None时（输出为scalar），使用`ApplyTensor`继承了输入的format
  （可能是NZ/5HD等），但scalar输出不支持这些format。另外`mse_loss_out`中多余的`CheckMemory`
  在output tensor未初始化时触发false positive。
- Fix: reduction模式下使用`ApplyTensorWithFormat(ACL_FORMAT_ND)`；删除多余的CheckMemory。
- Reviewability: medium -- 需要理解reduction输出的format约束
- Review rule: reduction输出为scalar时必须使用ACL_FORMAT_ND。

### D-920: DDP logger weak_ptr未检查expired导致空指针

- Root cause: weak_ptr未检查expired
- Hashes:
  1be5186e1 d6d580fc1
- Files: torch_npu/csrc/distributed/reducer.cpp
- Defect: `search_unused_parameters`中直接`logger_.lock()->log_if_graph_static(false)`，
  但`logger_`是`weak_ptr`，当Logger已被销毁时`lock()`返回空shared_ptr，调用方法导致SIGSEGV。
- Fix: 添加`!logger_.expired()`前置检查。
- Reviewability: high -- weak_ptr使用前必须检查expired/lock
- Review rule: 所有weak_ptr::lock()调用前必须检查expired或验证lock()结果非空。

### D-921: tensor工厂方法(new_zeros/full/ones/tensor)未处理device='npu'

- Root cause: tensor工厂方法device参数处理缺失
- Hashes:
  1c60a5418
- Files: torch_npu/utils/tensor_methods.py, torch_npu/csrc/utils/TensorMethods.cpp
- Defect: `new_zeros`/`new_full`/`new_ones`/`new_tensor`当传入`device='npu'`时未正确处理，
  tensor可能创建在错误的设备上或抛出异常。
- Fix: 在Python层添加device参数monkey-patch处理，在C++层添加对应的TensorMethods实现。
- Reviewability: high -- tensor创建API的device参数是标准接口
- Review rule: 所有tensor工厂方法需测试device=None/cpu/npu三种参数。

### D-922: LSTM get_mask中使用NPU专有API做dtype转换

- Root cause: NPU专有API使用不当
- Hashes:
  1cf773731 5aa01f8f7
- Files: torch_npu/csrc/aten/ops/LstmKernelNpu.cpp
- Defect: `get_mask`中用`NPUNativeFunctions::npu_dtype_cast`做bool->int和->half转换，
  但该函数在某些计算图模式下行为不正确（可能不触发正确的dispatch）。
- Fix: 改用标准PyTorch API `.to(at::ScalarType::Int)`和`.to(at::ScalarType::Half)`。
- Reviewability: high -- 应优先使用标准API
- Review rule: dtype转换优先使用`.to()`，不使用NPU专有的npu_dtype_cast。

### D-923: upsample_nearest2d输出shape计算错误（scale_factors重载）

- Root cause: 输出shape计算错误
- Hashes:
  1cf96d4ef 66e720dff bd5d280d2
- Files: torch_npu/csrc/aten/ops/UpsampleNearest2dKernelNpu.cpp
- Defect: `upsample_nearest2d`的`scale_factors`重载版本中，`ApplyTensor(input, osize)`
  直接使用`osize`作为output shape，但`osize`只包含spatial dimensions（2个元素），
  不是完整的4-D tensor shape（N,C,H,W）。
- Fix: 用`upsample_nearest2d_npu_output_size(input, osize)`计算完整的4-D shape。
- Reviewability: high -- output shape应与算子实际输出维度匹配
- Review rule: ApplyTensor的shape参数必须是完整的tensor shape，不是部分spatial shape。

### D-924: transfer_to_npu缺少builtins.isinstance覆盖

- Root cause: builtins.isinstance未覆盖（transfer_to_npu）
- Hashes:
  1d39838a3 418b3a398 98c5b0950 e8697157a
- Files: torch_npu/contrib/transfer_to_npu.py
- Defect: transfer_to_npu模式下，`isinstance(obj, torch.device)`对NPU device返回False，
  因为builtins.isinstance未被覆盖来处理NPU自定义device类型。
- Fix: 添加`_isinstance`函数覆盖`builtins.isinstance`，处理torch.device和NPU
  tensor类型的isinstance检查。
- Reviewability: medium -- 需要理解NPU device的类型层次
- Review rule: transfer_to_npu的isinstance覆盖需与主init中的_isinstance逻辑一致。

### D-925: shutdown前GC未收集导致use-after-free

- Root cause: shutdown时序问题（GC与NPU资源释放竞争）
- Hashes:
  1d5ac4bef 2376bf83f 56322f6bf a3a89f9ff f42bb3fe4
- Files: torch_npu/__init__.py
- Defect: `_npu_shutdown`直接释放NPU资源，但Python GC尚未收集所有持有NPU引用的对象。
  后续GC在finalizer中尝试释放已失效的NPU资源（如tensor的raw_delete）导致crash。
- Fix: 在`_npu_shutdown`开头添加`gc.collect()`强制先回收所有可回收对象。
- Reviewability: medium -- 需要理解Python GC与C++资源释放的交互
- Review rule: 框架shutdown需在释放底层资源前确保上层对象已被回收。

### D-926: HCCL错误信息缺少详细ACL错误描述

- Root cause: 错误信息不完整
- Hashes:
  1d88b003f a23d501c2
- Files: torch_npu/csrc/distributed/HCCLUtils.hpp
- Defect: `HCCL_CHECK`宏构造的错误字符串只包含文件名+行号+错误码数字，
  缺少`:`分隔符且不包含ACL底层的详细错误描述信息。
- Fix: 添加`:`分隔符，追加`c10_npu::acl::AclGetErrMsg()`获取详细错误描述。
- Reviewability: high -- 错误信息质量review
- Review rule: 所有错误宏需包含AclGetErrMsg()提供底层诊断信息。

### D-927: InferFormat对0维和shape=[1]的tensor格式推断错误

- Root cause: 格式推断遗漏（0维/单元素边界）
- Hashes:
  1d9e3ffb6
- Files: torch_npu/csrc/framework/InferFormat.cpp
- Defect: InferFormat在处理0维tensor（scalar）或shape=[1]的tensor时格式推断不正确，
  可能错误地保持或转换为不兼容的format（如将scalar保持为NZ format）。
- Fix: 重构InferFormat逻辑，对0维和低维tensor明确回退到ND/NCHW format。
- Reviewability: medium -- 格式推断的边界条件需系统性考虑
- Review rule: format推断逻辑需有明确的维度下限检查。

<!-- D-skip: 1da500443 | fix vsplit test | Python语法修复(trailing comma), 非代码逻辑defect -->

### D-928: nllloss2d精度问题

- Root cause: 算子精度问题
- Hashes:
  1dbace382 3fb810007 4605b0db1
- Files: torch_npu/csrc/aten/ops/loss/NLLLoss2dKernelNpu.cpp
- Defect: nllloss2d的NPU实现在特定输入shape下精度不足，与CPU结果偏差超出容忍范围。
- Fix: 修正kernel实现，调整计算精度。
- Reviewability: medium -- 需要精度对比测试
- Review rule: loss函数需对各种reduction模式逐一验证精度对齐。

### D-929: lstm_backward输出shape与输入不匹配

- Root cause: backward输出shape错误
- Hashes:
  1e4f0d803 2d4d9718b 3316e55c7 746688c5b
- Files: torch_npu/csrc/aten/ops/LstmKernelNpu.cpp
- Defect: lstm_backward在某些输入shape组合下（如batch_first=True，不同sequence length），
  输出gradient tensor的shape与期望不一致。
- Fix: 修正backward中gradient output的shape计算逻辑。
- Reviewability: high -- backward的输出shape应严格匹配forward的输入shape
- Review rule: RNN类算子backward需测试各种(num_layers, batch_first, bidirectional)组合。

### D-930: cummax算子未正确返回indices

- Root cause: 算子输出遗漏（indices返回值）
- Hashes:
  1eb7dd432 26e295d31 2c451fe74 e89aef06e
- Files: torch_npu/csrc/aten/ops/CummaxKernelNpu.cpp
- Defect: cummax应返回(values, indices)元组，但NPU适配中indices的计算或返回有误，
  导致indices结果不正确或类型不对。
- Fix: 修正cummax kernel的indices输出逻辑。
- Reviewability: high -- 返回值个数和类型是算子接口规范
- Review rule: 返回多值的算子需逐一验证每个返回值的正确性。

<!-- D-skip: 1fcf3a003 | fix Chinese doc links | 纯文档链接修复, 非defect -->

### D-931: complex类型dtype映射为UNDEFINED导致NPU不支持

- Root cause: 遗漏dtype支持（complex类型）
- Hashes:
  1fd423491 514a9f31d 62155d8f6
- Files: torch_npu/csrc/framework/utils/CalcuOpUtil.cpp
- Defect: Aten到ACL的dtype映射表中`ComplexFloat`和`ComplexDouble`映射为
  `ACL_DT_UNDEFINED`，导致complex类型tensor在NPU上无法使用。
- Fix: 映射为`ACL_COMPLEX64`和`ACL_COMPLEX128`。
- Reviewability: high -- dtype映射表review应检查UNDEFINED项
- Review rule: dtype映射表中不应有ACL_DT_UNDEFINED项，除非该类型确实不支持。

### D-932: uniform算子用非inplace操作导致result未更新

- Root cause: 非inplace操作误用导致输出未更新
- Hashes:
  20750f0a9 a0f5177f8 ef4ff6cce fa84cc897
- Files: torch_npu/csrc/aten/ops/UniformKernelNpu.cpp
- Defect: `uniform_out_npu`中用`.mul().sub()`（非inplace）处理result tensor。
  `.mul(to)`创建新tensor而非修改result本身，所以result仍是原始的U(0,1)分布值。
- Fix: 改用inplace操作`.mul_().sub_()`直接修改result。
- Reviewability: high -- inplace vs 非inplace是基本的PyTorch C++知识
- Review rule: out变体中对result tensor的操作必须使用inplace方法。

### D-933: var算子在unbiased模式下对单元素tensor未返回NAN

- Root cause: 算子边界条件处理缺失
- Hashes: 90a89af3b
- Files: torch_npu/csrc/aten/ops/VarKernelNpu.cpp
- Defect: var算子在`unbiased=True`时，当reduce维度的元素数<=1，Bessel校正要求除以(N-1)=0，
  应返回NAN。原实现缺少此边界检查，直接调用NPU算子计算，与CPU行为不一致。
- Fix: 新增`get_shape_prod`计算reduce维度元素总数，当`unbiased`且`shape_prod<=1`时
  直接fill NAN并提前返回。
- Reviewability: high -- 逻辑清晰，只需检查边界条件的数学正确性
- Review rule: 统计类算子实现时须验证N<=correction的退化情况是否与PyTorch CPU语义一致。

### D-934: Revert item()替换ConvertTensorToScalar（图模式下标量提取失败）

- Root cause: Tensor-Scalar转换方式不兼容
- Hashes: f935b0477
- Files: torch_npu/csrc/aten/ops/op_api/AddKernelNpuOpApi.cpp,
  torch_npu/csrc/aten/ops/op_api/DivKernelNpuOpApi.cpp,
  torch_npu/csrc/aten/ops/op_api/MulKernelNpuOpApi.cpp,
  torch_npu/csrc/aten/ops/op_api/SubKernelNpuOpApi.cpp,
  torch_npu/csrc/aten/ops/op_api/op_api_common.h,
  torch_npu/csrc/framework/OpCmdHelper.cpp,
  torch_npu/csrc/framework/utils/CalcuOpUtil.cpp
- Defect: 先前提交将`ConvertTensorToScalar`替换为`tensor.item()`，但`item()`会触发D2H同步，
  在图模式或特定dtype(BFloat16/ComplexFloat)下可能失败。此commit回退所有调用点。
  注: 与D-951构成振荡修复对，反映ConvertTensorToScalar vs item()的取舍争议。
- Fix: Revert整个commit，恢复`ConvertTensorToScalar`的实现和所有调用点。
- Reviewability: medium -- 涉及9个文件的批量替换
- Review rule: NPU tensor转scalar时须考虑图模式兼容性，不能简单用`item()`替代专用函数。

### D-935: set_device等接口缺少device编号合法性校验

- Root cause: API参数校验缺失
- Hashes: 9cbd94e17
- Files: torch_npu/hooks/hooks.py, torch_npu/npu/utils.py, torch_npu/utils/device_guard.py
- Defect: `set_device`、`torch_device_guard`装饰器等接口接受用户传入的device编号，
  但未校验编号是否在`[0, device_count)`范围内，传入非法值导致底层C API难以诊断的错误。
- Fix: 新增`check_is_valid_ordinal`函数，支持int/str/device对象三种形式校验，
  在所有device设置入口处调用。
- Reviewability: high -- 纯校验逻辑
- Review rule: 所有接受device id的公开接口必须在调用底层C API前校验编号范围。

### D-936: flip算子在dims为空时未返回原tensor拷贝

- Root cause: 算子空参数边界处理缺失
- Hashes: cb0d7ba81
- Files: torch_npu/csrc/aten/ops/FlipKernelNpu.cpp
- Defect: `flip`在`dims`为空数组时，PyTorch CPU语义是返回tensor自身的拷贝。
  NPU实现缺少此判断，直接将空dims传给NPU算子，导致未定义行为。
- Fix: 在函数入口添加`dims.size()==0`时直接返回`self.clone()`的短路逻辑。
- Reviewability: high -- 单行条件判断
- Review rule: 算子实现须处理dims/axis参数为空的退化情况，与PyTorch CPU语义保持一致。

### D-937: std算子在元素数<=correction时NPU与CPU结果不一致

- Root cause: 算子边界条件处理缺失
- Hashes: 4fff1d7ef
- Files: torch_npu/csrc/aten/ops/StdKernelNpu.cpp
- Defect: `std`在`correction>=N`时，CPU返回NAN或INF，但NPU实现直接调用算子计算，
  缺少退化情况处理。与D-933(var)是对称问题。
- Fix: 新增`calc_shape_prod`，在算子执行后检查`shape_prod<=correction`时
  分别fill NAN或INFINITY。
- Reviewability: high -- 与D-933逻辑对称
- Review rule: std/var等涉及Bessel校正的算子须成对检查N<=correction的边界情况。

### D-938: utilization测试用例假设NPU空闲返回0导致CI不稳定

- Root cause: 测试用例环境假设不合理
- Hashes: 38a166b42
- Files: test/test_npu/test_torch_npu.py
- Defect: `test_npu_get_utilization`断言返回0，假设NPU设备完全空闲。
  CI环境中设备被其他进程占用时利用率非零，测试随机失败。
- Fix: 直接删除该不稳定测试用例。
- Reviewability: high -- 仅删除5行代码
- Review rule: 硬件状态相关的测试不应assert精确值，应测试API可调用性或值范围。

### D-939: lerp算子输出shape未考虑三路广播

- Root cause: 算子输出shape计算错误
- Hashes: f8065f951
- Files: torch_npu/csrc/aten/ops/LerpKernelNpu.cpp
- Defect: `lerp(self, end, weight)`的输出shape应为self/end/weight三者广播结果。
  原实现直接用`self`的shape，输入shape不同时产生错误结果。
  `lerp_out`的CheckOut也未包含end和weight，inplace版本缺少广播合法性校验。
- Fix: 新增`lerp_broadcast_size`计算三路广播shape，所有变体均使用广播后的size。
- Reviewability: medium -- 涉及多个函数签名变更
- Review rule: 多输入算子的输出shape必须经过所有输入的broadcast计算，不能只取第一个输入的shape。

### D-940: NPU_CHECK_SUPPORTED_OR_ERROR宏静默吞掉FEATURE_NOT_SUPPORT错误

- Root cause: 错误处理策略不当
- Hashes: 9f7d5a54f
- Files: torch_npu/csrc/core/npu/NPUException.h
- Defect: `NPU_CHECK_SUPPORTED_OR_ERROR`宏将`ACL_ERROR_RT_FEATURE_NOT_SUPPORT`与
  `ACL_ERROR_NONE`同等对待，feature不支持时完全静默，调用方无法感知功能缺失。
- Fix: 将FEATURE_NOT_SUPPORT改为发出WARN日志，提示CANN包可能不匹配。
- Reviewability: high -- 宏定义改动集中
- Review rule: 错误处理宏中不应将"不支持"类错误与成功等同处理，至少应输出警告日志。

### D-941: D-940的重复提交

- Root cause: 错误处理策略不当
- Hashes: 28235008a
- Files: torch_npu/csrc/core/npu/NPUException.h
- Defect: 与D-940完全相同的变更内容，来自不同分支的cherry-pick或squash遗留。
- Fix: 同D-940。
- Reviewability: high
- Review rule: 合并分支时应检查是否存在重复commit。

### D-942: 比较算子(eq/ge/gt/le/lt/ne)不支持CPU scalar tensor输入

- Root cause: 跨设备Tensor混合计算未处理
- Hashes: 5253e8083
- Files: torch_npu/csrc/aten/ops/EqKernelNpu.cpp,
  torch_npu/csrc/aten/ops/GeKernelNpu.cpp,
  torch_npu/csrc/aten/ops/GtKernelNpu.cpp,
  torch_npu/csrc/aten/ops/LeKernelNpu.cpp,
  torch_npu/csrc/aten/ops/LtKernelNpu.cpp,
  torch_npu/csrc/aten/ops/NeKernelNpu.cpp
- Defect: 六个比较算子在一个输入为NPU tensor、另一个为CPU scalar tensor时直接传入NPU算子，
  因设备不匹配而失败。PyTorch允许该混合调用，CPU侧0维tensor应被转为scalar。
- Fix: 用`IsCPUScalar`检测CPU侧0维tensor，转为scalar后调用tensor-scalar重载。
  ge的self为scalar分支修正了语义反转(ge->le)。
- Reviewability: medium -- 六个算子对称修改，需逐一验证语义反转
- Review rule: 所有二元算子须处理一侧为CPU scalar tensor的混合设备调用。

### D-943: repeat_interleave对repeats.size(0)==1的合法输入误报校验失败

- Root cause: API参数校验条件不完整
- Hashes: 8108ea03f
- Files: torch_npu/csrc/aten/ops/RepeatInterLeaveKernelNpu.cpp
- Defect: TORCH_CHECK仅允许`repeats.size(0) == self_tensor.size(real_dim)`，
  遗漏了PyTorch中repeats为单元素tensor(size(0)==1)表示对所有元素做相同重复的合法情况。
- Fix: 在校验条件中用`||`追加`repeats.size(0) == 1`分支。
- Reviewability: high -- 单行条件修改，对照PyTorch文档即可发现
- Review rule: 参数校验条件必须覆盖上游PyTorch文档中所有合法输入形式。

### D-944: broadcast contiguous路径未处理bool类型导致BroadcastTo算子报错

- Root cause: dtype分支处理遗漏
- Hashes: f4c5a0c5d
- Files: torch_npu/csrc/framework/contiguous/broadcast_opt.cpp
- Defect: `broadcast_opt`中source tensor为contiguous时直接调用`npu_broadcast_out`，
  但NPU的BroadcastTo算子不支持bool dtype，导致bool tensor的broadcast失败。
- Fix: 增加`self.dtype() == at::kBool`分支，改用`npu_broadcast` +
  `LaunchAsyncCopyTask`的拷贝方式绕过算子限制。
- Reviewability: medium -- 需了解NPU算子的dtype支持范围
- Review rule: 调用NPU算子前需确认其支持的dtype列表，对不支持的dtype提供fallback。

### D-945: repeat_interleave输出shape推断未处理0-d repeats和单元素repeats

- Root cause: 算子输出shape计算错误
- Hashes: f19095e07
- Files: torch_npu/csrc/aten/ops/RepeatInterLeaveKernelNpu.cpp,
  torch_npu/csrc/framework/utils/KernelNpuOutputSize.cpp
- Defect: tensor repeats重载中未对0维(scalar)的repeats做unsqueeze处理，
  且当repeats为单元素tensor时直接返回self跳过计算，不符合PyTorch语义。
- Fix: 对0维repeats先做`unsqueeze_(0)`；移除单元素时的早返回；
  同步修正KernelNpuOutputSize中的shape推断。
- Reviewability: medium -- 需理解repeat_interleave在不同repeats维度下的语义
- Review rule: 算子对scalar tensor输入的处理须与PyTorch CPU实现逐case对齐。

### D-946: empty_like使用self.options()而非传入的options参数导致dtype错误

- Root cause: 函数参数误用
- Hashes: d958b5ecb
- Files: torch_npu/csrc/aten/common/TensorFactories.cpp
- Defect: `empty_like`调用`ApplyTensorWithFormat`时传入`self.options()`而非函数参数`options`，
  导致用户通过`dtype`参数指定的目标类型被忽略，返回tensor的dtype始终与输入相同。
- Fix: 将`self.options()`替换为函数签名中的`options`参数。
- Reviewability: high -- 典型的写错变量名，一行diff
- Review rule: tensor工厂函数中涉及options/dtype的参数传递，必须使用用户传入的options。

### D-947: convolution op_api路径output size计算与unbatch逻辑耦合

- Root cause: 算子输出shape计算错误
- Hashes: a0e4f70fb
- Files: torch_npu/csrc/aten/ops/op_api/ConvolutionKernelNpuOpApi.cpp,
  torch_npu/csrc/framework/utils/KernelNpuOutputSize.cpp
- Defect: convolution的output size计算散布在主函数中，与transposed/dim的分支判断和
  input unbatch逻辑交织。当dim既不是1也不是2时缺少提前返回，unbatch判断依赖执行路径。
- Fix: 将output size计算抽取为`conv_npu_output_size`函数；增加非法dim的提前返回；
  unbatch判断改为直接检查`dim == 2 && inputK == 3`。
- Reviewability: medium -- 需理解conv1d/conv2d在不同维度下的行为差异
- Review rule: 算子的output size计算应封装为独立函数，不与主逻辑中的状态变量耦合。

### D-948: bincount调用aclnnBincount缺少minlength参数

- Root cause: 算子调用参数缺失
- Hashes: d57671049
- Files: torch_npu/csrc/aten/ops/op_api/BincountKernelNpuOpApi.cpp
- Defect: `EXEC_NPU_CMD(aclnnBincount, self, weight, result)`遗漏了`minlength`参数，
  当实际数据最大值小于minlength时结果shape不正确。
- Fix: 在`EXEC_NPU_CMD`调用中补充`minlength`参数。
- Reviewability: high -- 对照aclnn API签名即可发现参数数量不匹配
- Review rule: 调用EXEC_NPU_CMD时必须逐一对照aclnn API的参数列表确认参数个数和顺序。

### D-949: 测试文件中autograd.profiler.profile使用已废弃的use_npu参数

- Root cause: API废弃参数未更新
- Hashes: a80bc56b4
- Files: test/test_network_ops/test_viewcopy.py,
  test/test_trans_contiguous/test_as_strided_copy_to_contiguous.py (等12个测试文件)
- Defect: 12个测试文件使用`torch.autograd.profiler.profile(use_npu=True)`，
  但上游已将参数改为`use_device`字符串形式，`use_npu`被废弃。
- Fix: 将所有`use_npu=True`替换为`use_device='npu'`。
- Reviewability: high -- 纯机械替换，可通过全局搜索批量发现
- Review rule: 跟随上游PyTorch版本升级时须全局搜索废弃API参数并批量更新。

### D-950: scatter_out缺少reduce参数校验且sort_out缺少输出shape校验

- Root cause: API参数校验缺失
- Hashes: 2c943b20a
- Files: torch_npu/csrc/aten/ops/op_api/ScatterKernelNpuOpApi.cpp,
  torch_npu/csrc/aten/ops/op_api/SortKernelNpuOpApi.cpp
- Defect: scatter的`get_reduce`对非法reduce值返回0而无校验，可能将非法参数静默传给算子。
  sort_out的三个重载均缺少对输出tensor values/indices的shape校验。
- Fix: scatter增加`reduce_valid`前置校验；sort_out增加`CheckOut`校验输出shape。
- Reviewability: medium -- 需理解各_out变体对输出tensor的约束
- Review rule: _out变体算子必须在执行前校验输出tensor的dtype和shape，枚举参数必须有合法值校验。

### D-951: ConvertTensorToScalar手写dtype分发改为item()

- Root cause: 标量提取方式不安全
- Hashes: f842e440c
- Files: torch_npu/csrc/aten/ops/op_api/AddKernelNpuOpApi.cpp,
  torch_npu/csrc/framework/utils/CalcuOpUtil.cpp (等9个文件)
- Defect: `ConvertTensorToScalar`通过手写dtype switch-case和`data_ptr()`提取标量，
  不支持新增dtype且对设备端tensor不安全。
  注: 与D-934构成振荡修复对，D-934回退item()改回ConvertTensorToScalar，本条再次改回item()。
- Fix: 将所有`ConvertTensorToScalar`调用替换为`tensor.item()`，删除该函数。
- Reviewability: high -- 全局搜索函数名即可定位
- Review rule: 从tensor提取标量值应统一使用`tensor.item()`，禁止通过`data_ptr()`手动解引用。

### D-952: addr算子使用matmul替代逐元素乘法且未处理混合dtype

- Root cause: 算子语义实现错误
- Hashes: 3f506e64a
- Files: torch_npu/csrc/aten/ops/AddrKernelNpu.cpp
- Defect: addr语义是`beta*self + alpha*(vec1.outer(vec2))`，原实现用`at::mm`做矩阵乘法，
  虽然数值结果相同但对int/bool类型行为有差异。结果dtype未按`result_type`推断，
  bool输入场景也缺失处理。
- Fix: 将`at::mm`替换为`at::mul`做outer product；用`result_type`推断输出dtype；
  对bool做cast-float-compute-cast-back。
- Reviewability: medium -- 需理解addr的数学语义和dtype promotion规则
- Review rule: 实现数学算子时应严格对照PyTorch文档选择基础操作，并用result_type确定输出dtype。

### D-953: erf算子Bool输入dtype未转换

- Root cause: dtype分支处理遗漏
- Hashes: f25d55277
- Files: torch_npu/csrc/aten/npu_native_functions.yaml, torch_npu/csrc/aten/ops/op_api/ErfKernelNpuOpApi.cpp
- Defect: erf算子对Bool类型输入直接以原dtype创建输出tensor，但erf的数学语义要求浮点输出。Bool输入未转换为Float导致结果tensor dtype错误或计算异常。同时将op_api路径暂时关闭回退到非op_api实现。
- Fix: 在erf函数中增加Bool类型判断分支，当输入为Bool时以Float dtype创建输出tensor；yaml中将op_api标记从True改为False以回退实现路径。
- Reviewability: high -- 改动集中在一个dtype判断分支，逻辑清晰
- Review rule: 算子实现需检查所有合法输入dtype（尤其Bool/Int等非浮点类型）是否有正确的输出dtype映射。

### D-954: addcdiv_out输出校验与内存布局处理错误

- Root cause: 算子out变体输出校验逻辑错误
- Hashes: ba39b7810
- Files: test/test_network_ops/test_addcdiv.py, torch_npu/csrc/aten/ops/AddcdivKernelNpu.cpp
- Defect: addcdiv_out原实现先计算到临时tensor再copy到result，CheckOut校验用的是临时tensor而非原始输入，跳过了shape/dtype的正确校验。且addcdiv_用冗余的contiguous处理逻辑包裹了out调用，增加了不必要的复杂度。
- Fix: 将CheckOut改为直接校验原始输入(self, tensor1, tensor2)与result的匹配关系，去掉临时tensor中转；将contiguous判断逻辑下沉到addcdiv_out中统一处理，addcdiv_简化为直接调用out变体。
- Reviewability: medium -- 涉及输出校验和内存布局两个逻辑维度的重构
- Review rule: 算子out变体的CheckOut必须校验原始输入而非中间计算结果，确保输出tensor的shape/dtype/format约束在计算前完成验证。

### D-955: tanh算子缺少op_api开关导致路由错误

- Root cause: 算子注册配置遗漏
- Hashes: a7054b30c
- Files: torch_npu/csrc/aten/npu_native_functions.yaml
- Defect: tanh/tanh.out/tanh_三个函数在yaml中以简写形式注册（仅写函数名），缺少op_api开关字段。在op_api分发机制下，没有显式设置op_api: False会导致算子可能被错误路由到op_api路径执行，而该路径的tanh实现可能不正确或不存在。
- Fix: 将三个tanh条目从简写格式改为带func字段的完整格式，显式设置op_api: False。
- Reviewability: high -- 纯配置变更，一眼可见
- Review rule: yaml注册文件中新增算子条目时，必须使用完整的func+op_api格式，禁止使用不带开关的简写形式。

### D-956: 自定义Tensor转Scalar函数dtype覆盖不全

- Root cause: 重复实现标准库功能导致dtype遗漏
- Hashes: ff538b25f
- Files: torch_npu/csrc/framework/OpCmdHelper.cpp, torch_npu/csrc/framework/graph/construct/GraphConstructor.cpp, torch_npu/csrc/framework/utils/CalcuOpUtil.cpp, torch_npu/csrc/framework/utils/CalcuOpUtil.h
- Defect: 自定义的ConvertTensorToScalar函数通过if-else枚举处理各dtype，但枚举不完整（缺少部分dtype如Short等），且通过data_ptr直接转型存在安全隐患。PyTorch原生的tensor.item()已完整覆盖所有dtype，自定义实现纯属冗余且容易遗漏。
- Fix: 删除整个ConvertTensorToScalar函数（47行），所有调用点改用PyTorch标准的tensor.item()。
- Reviewability: high -- 典型的"删除冗余代码"模式，用标准API替换自定义实现
- Review rule: 禁止自行实现PyTorch已提供的标准类型转换功能；对既有的dtype枚举if-else实现应标记为技术债务并迁移到标准API。

### D-957: 模块初始化import顺序依赖错误

- Root cause: Python模块初始化顺序依赖
- Hashes: 69b8a738f
- Files: torch_npu/__init__.py
- Defect: `_ld_preload`模块的导入被放在`import torch`之前。由于`_ld_preload`可能依赖torch提供的动态库预加载路径或符号，在torch未导入时执行会导致初始化失败（如动态库加载顺序错误、符号找不到等）。
- Fix: 将`import torch`移到`from . import _ld_preload`之前，确保torch的C扩展和动态库先于NPU的preload模块初始化。
- Reviewability: high -- 两行顺序交换，改动极小
- Review rule: __init__.py中的import语句需按依赖拓扑排序，第三方框架（torch）必须先于依赖它的本地模块导入。

### D-958: amp梯度检查使用低效算子组合

- Root cause: 算子选择导致性能缺陷
- Hashes: b3cc9da85
- Files: torch_npu/csrc/aten/ops/AmpForeachNonFiniteCheckAndUnscaleKernelNpu.cpp
- Defect: 原实现对每个梯度tensor先调用isfinite逐元素判断再调用all聚合，产生了一个与输入等大的中间Bool tensor，在大规模梯度场景下内存和计算开销显著。同时for循环中按值拷贝tensor（`auto scaled_grad`）也带来不必要的引用计数操作。
- Fix: 改用sum+isfinite组合：先对梯度求sum得到单个标量，再用std::isfinite在CPU侧判断（NaN/Inf会传播到sum结果），避免创建大型中间tensor；循环变量改为const引用。
- Reviewability: medium -- 需要理解NaN/Inf在sum中的传播语义才能确认正确性
- Review rule: 涉及逐元素+聚合的两步操作时，优先考虑能否用单次聚合操作（如sum）替代，利用异常值的传播性质减少中间tensor分配。

### D-959: adaptive_avg_pool2d调用路径导致精度错误

- Root cause: 算子调用路径选择错误
- Hashes: 54150af02
- Files: torch_npu/csrc/aten/ops/op_api/AdaptiveAvgPool2dKernelNpuOpApi.cpp
- Defect: adaptive_avg_pool2d直接调用NPUNativeOpApiFunctions::_adaptive_avg_pool2d，绕过了PyTorch的标准dispatch机制。在resnet50等模型中，该直接调用路径的计算结果与标准路径存在精度差异，导致训练精度异常。
- Fix: 将调用从NPUNativeOpApiFunctions::_adaptive_avg_pool2d改为at::_adaptive_avg_pool2d，走PyTorch标准dispatch路径，确保与CPU实现行为一致。
- Reviewability: high -- 单行调用路径变更，但根因需要精度对比验证
- Review rule: 公开算子函数应通过at::命名空间调用内部实现以走标准dispatch，避免直接调用同类的NPU内部函数绕过dispatch逻辑。

### D-960: 设备校验函数导致集群训练死锁

- Root cause: 初始化阶段设备校验引发死锁
- Hashes: 606585778
- Files: torch_npu/hooks/hooks.py, torch_npu/npu/utils.py, torch_npu/utils/device_guard.py
- Defect: check_is_valid函数在set_device和device_guard等路径中被调用，内部调用device_count()查询设备数量。在分布式训练初始化阶段，device_count可能触发NPU驱动的全局同步或锁等待，当多个进程/线程同时执行设备校验时形成死锁，导致集群hung住。
- Fix: 移除所有check_is_valid调用点（set_device、device_guard装饰器、hooks线程中共7处），删除check_is_valid函数本身，依赖底层C API自身的错误处理来报告无效设备。
- Reviewability: medium -- 改动分散在3个文件，需理解分布式初始化时序才能确认根因
- Review rule: 设备校验逻辑不应在热路径或多线程环境中调用可能触发全局同步的API（如device_count），此类校验应前置到单点初始化阶段。

### D-961: nms_rotated单元测试多处逻辑错误

- Root cause: 测试代码逻辑错误
- Hashes: b6d05e93d
- Files: test/test_network_ops/test_nms_rotated.py
- Defect: 单元测试存在多处实现错误：ordered_pts用固定常量TOTAL_INTER_POINTS初始化而非实际点数num；排序范围包含了不应参与排序的起始点；convex hull循环中手动`i += 1`与for循环递增重复；缺少面积为零的特殊case处理；以及未统一转float导致dtype不匹配。这些错误使UT无法正确验证nms_rotated算子的行为。
- Fix: 修正tensor大小为num、修正排序范围为[1:]、删除冗余的手动递增、增加面积为零的early return、统一输入为float dtype。
- Reviewability: medium -- 改动点多且分散，但每个改动独立可审
- Review rule: 计算几何类UT需覆盖退化case（零面积、共线点、重合矩形），且数组索引范围须与算法描述严格对应。

### D-962: div算子未检查optional参数有效性直接解引用

- Root cause: optional参数解引用未做has_value检查
- Hashes: ca1772ac5
- Files: test/test_network_ops/test_div.py, test/test_network_ops/test_divide.py, torch_npu/csrc/aten/ops/DivKernelNpu.cpp
- Defect: div的多个重载函数接收`c10::optional<c10::string_view> rounding_mode`参数，但在5处直接使用`*rounding_mode`解引用而未先调用has_value()检查。当rounding_mode为nullopt（即PyTorch默认的true division语义）时，解引用导致未定义行为或崩溃。测试用例中也错误地使用字符串"true"代替None。
- Fix: 在所有`*rounding_mode`解引用前增加`rounding_mode.has_value()`守卫条件；测试用例中将rounding_mode为"true"的case改为None以匹配PyTorch接口语义。
- Reviewability: high -- 典型的空值检查遗漏，pattern统一且易于搜索
- Review rule: 所有c10::optional参数在解引用前必须检查has_value()，可通过静态分析工具或grep `\*rounding_mode`等模式批量排查。

### D-963: sub算子scalar-tensor操作数顺序未处理

- Root cause: 算子操作数顺序分支遗漏
- Hashes: c73782b81
- Files: torch_npu/csrc/aten/ops/SubKernelNpu.cpp, torch_npu/testing/common_methods_invocations.py
- Defect: `sub`算子在处理`sub(scalar, tensor)`时，只覆盖了`sub(tensor, scalar)`的分支（`IsCPUScalar(other)`），未处理self为scalar的情况。当self是CPU标量tensor时走入通用分支，导致device错误。
- Fix: 新增`IsCPUScalar(self)`分支，调用新函数`sub_self_scalar_out_npu`将scalar作为Input传入OpCommand，并补充float16测试dtype。
- Reviewability: high -- 修复逻辑清晰，新增分支与已有分支对称，容易理解
- Review rule: 二元算子实现时检查两个操作数的所有排列组合（tensor-tensor, tensor-scalar, scalar-tensor）是否都已覆盖。

### D-964: convolution算子Path3回退路径和精度配置错误

- Root cause: 算子调用路径与精度配置错误
- Hashes: e4504f0ff
- Files: torch_npu/csrc/aten/npu_native_functions.yaml, torch_npu/csrc/aten/ops/op_api/ConvolutionKernelNpuOpApi.cpp
- Defect: convolution算子存在两个问题：(1) `DO_COMPATIBILITY`宏错误使用了`aclnnAdd`而非`aclnnConvolution`作为兼容性检查标识；(2) 当groups>1或3D场景回退调用`at::_convolution`时会重新进入OpApi dispatch导致无限递归或路径错误，且输出tensor未做dtype promotion、缺少`cube_math_dtype`参数。`_convolution`在yaml中未标记为`op_api: False`，路由配置不正确。
- Fix: 修正`DO_COMPATIBILITY`标识为`aclnnConvolution`，回退路径改为直接调用`NPUNativeFunctions::_convolution`避免dispatch问题，yaml中显式标记`_convolution`为`op_api: False`，并新增`promote_dtype`和`cube_math_dtype`参数。
- Reviewability: medium -- 涉及yaml路由配置、dispatch路径和精度逻辑三处改动，需要理解整体dispatch机制
- Review rule: 算子回退路径必须直接调用目标实现函数，禁止通过公共dispatch入口间接调用以避免路由循环。

### D-965: stack算子输出dtype未做类型推导

- Root cause: 输出tensor dtype推导缺失
- Hashes: 7d87d5791
- Files: torch_npu/csrc/aten/ops/op_api/StackKernelNpuOpApi.cpp
- Defect: `stack`算子直接使用`tensors[0]`的dtype作为输出dtype，未调用`at::native::result_type(tensors)`做类型推导。当输入tensors包含不同dtype时，输出dtype不正确（应按PyTorch类型提升规则取最宽类型）。同时`CheckOut`使用了指定format的重载，过度约束了输出格式。
- Fix: 使用`at::native::result_type(tensors)`推导输出dtype，`CheckOut`改用以tensor为参考的重载，输出创建改用`ApplyTensor`继承输入tensor的format属性。
- Reviewability: high -- 改动集中在dtype推导一个逻辑点，修复模式清晰
- Review rule: 多输入算子的输出dtype必须通过`result_type`推导，不能直接取第一个输入的dtype。

### D-966: ConvertTensorToScalar缺少bool/bfloat16/complex类型处理

- Root cause: dtype分支处理遗漏
- Hashes: 2cbd7189c
- Files: torch_npu/csrc/framework/utils/CalcuOpUtil.cpp
- Defect: `ConvertTensorToScalar`函数通过if-else链处理各种dtype到Scalar的转换，但只覆盖了int/long/float/double/half，遗漏了Bool、BFloat16、ComplexFloat、ComplexDouble四种类型。当传入这些类型的tensor时会触发"unsupport scalar type"错误。
- Fix: 补充四种dtype的else-if分支，分别用对应的C++类型(`int8_t`/`c10::BFloat16`/`c10::complex<float>`/`c10::complex<double>`)做指针转换并构造Scalar。
- Reviewability: high -- 纯粹的分支补全，模式与已有代码完全一致
- Review rule: 类型分发的if-else/switch必须覆盖所有PyTorch支持的ScalarType，或有明确的fallback处理和文档说明。

### D-967: ConvertTensorToScalar缺少bool/bfloat16/complex类型处理（同源D-966）

- Root cause: dtype分支处理遗漏
- Hashes: c4e7ff1af
- Files: torch_npu/csrc/framework/utils/CalcuOpUtil.cpp
- Defect: 与D-966完全相同的修复，作用于不同的分支。同一作者在两个分支上提交了相同的bugfix（D-966的commit message引用了本commit c4e7ff1），说明此修复被cherry-pick到了多个分支。
- Fix: 与D-966一致，补充Bool、BFloat16、ComplexFloat、ComplexDouble四种dtype分支。
- Reviewability: high -- 与D-966相同的机械式分支补全
- Review rule: 同D-966，类型分发代码应覆盖所有支持的ScalarType。

### D-968: tanh算子整型输入导致输出dtype错误

- Root cause: 整型输入的输出dtype处理缺失
- Hashes: e531cfaaa
- Files: torch_npu/csrc/aten/ops/op_api/TanhKernelNpuOpApi.cpp
- Defect: `tanh`算子对整型输入直接使用`ApplyTensor(self)`创建输出，继承了输入的整型dtype。但tanh的数学结果是浮点数，整型输出会导致精度丢失或底层算子报错。`tanh_out`变体也未检查result的dtype是否为浮点类型。
- Fix: `tanh`中检测整型输入时将输出dtype强制设为`at::kFloat`；`tanh_out`中增加`TORCH_CHECK`确保result不是整型。
- Reviewability: high -- 修复逻辑简单直接，与PyTorch语义一致（tanh对整型输入应输出float）
- Review rule: 数学函数（三角函数、指数、对数等）实现时必须处理整型输入的dtype提升，输出dtype应为浮点类型。

### D-969: bitwise_and_out的dtype校验逻辑使用了错误的参考tensor

- Root cause: 输出校验参考tensor选取错误
- Hashes: 7c1800f80
- Files: torch_npu/csrc/aten/ops/op_api/BitwiseAndKernelNpuOpApi.cpp
- Defect: `bitwise_and_out`通过`IsScalarWrappedToTensor`判断self是否为标量包装tensor来选择参考tensor（self或other），再用该参考tensor校验输出result的dtype和format。这个逻辑有误：当self和out的dtype不同时（如`self=int16, out=int32`），用self/other的dtype去校验out会导致合法调用被拒绝。`_out`变体的输出dtype应由调用者通过result参数决定。
- Fix: 移除基于`IsScalarWrappedToTensor`的参考tensor选取逻辑，`CheckOut`直接用result自身作为参考，不再强制要求输出dtype与输入一致。
- Reviewability: high -- 删除了有问题的逻辑，简化了代码路径
- Review rule: `_out`变体的`CheckOut`应尊重调用者提供的输出tensor的dtype，不应用输入tensor的dtype覆盖校验。

### D-970: 嵌套函数定义中多余的self参数

- Root cause: Python函数签名错误
- Hashes: cb281820f
- Files: ci/access_control_test.py
- Defect: `analyze`方法内部定义的嵌套函数`is_hostapi_enabled(self, modify_file)`错误地包含了`self`参数。嵌套函数不是类方法，不会自动接收实例引用。调用`is_hostapi_enabled(modify_file)`时，`modify_file`的值会被绑定到`self`形参，而`modify_file`形参则缺少实参，导致TypeError。
- Fix: 移除嵌套函数签名中的`self`参数，改为`is_hostapi_enabled(modify_file)`。
- Reviewability: high -- 单行修改，错误原因一目了然
- Review rule: 嵌套函数（非方法）的签名中不应包含`self`参数，代码审查时应注意区分方法定义和闭包定义。

### D-971: lt算子使用了错误的aclnn API名称

- Root cause: 底层API名称引用错误
- Hashes: 244b0df9e
- Files: torch_npu/csrc/aten/ops/op_api/LtKernelNpuOpApi.cpp
- Defect: `lt`算子的`DO_COMPATIBILITY`和`EXEC_NPU_CMD`宏调用中使用了`aclnnLessTensor`，但实际注册的API名称为`aclnnLtTensor`。`DO_COMPATIBILITY`宏通过API名称查找函数指针，名称不匹配会导致兼容性检查失败而错误回退到旧路径，或在运行时找不到符号。
- Fix: 将所有`aclnnLessTensor`替换为`aclnnLtTensor`，与实际的aclnn算子注册名一致。
- Reviewability: high -- 纯字符串替换，4处改动完全对称
- Review rule: aclnn API名称必须与算子注册表中的名称精确匹配，建议通过宏或常量统一管理API名称，避免手写字符串。

### D-972: new_ones的Python参数解析包含多余的Tensor self

- Root cause: Python-C++绑定参数解析错误
- Hashes: be580b7e0
- Files: torch_npu/csrc/utils/TensorMethods.cpp, test/test_npu/test_npu.py
- Defect: `THPVariable_new_ones`的参数解析schema写为`"new_ones(Tensor self, IntArrayRef size, ...)"`,将Python层的`self`（已作为`PyObject* self`传入C函数）也放入了解析参数列表中。这导致`r.tensor(0)`取的是args中的第一个参数而非实际的self，当用户传入`tensor.new_ones(torch.tensor(2))`时，`torch.tensor(2)`被当作self解析，而size参数缺失。所有后续参数索引也因此偏移了1位。
- Fix: 从解析schema中移除`Tensor self`，手动从args[0]提取self对象并用`THPVariable_Unpack`解包，调整后续所有参数索引减1。同时补充了传入tensor作为size参数的测试用例。
- Reviewability: medium -- 需要理解PyTorch的PythonArgParser机制和self参数的处理约定
- Review rule: `THPVariable_*`系列手写绑定函数中，self必须从C函数参数获取，不应放入PythonArgParser的schema定义中，避免参数索引整体偏移。

### D-973: threshold_backward和hardswish算子错误引用CalcuOpUtil

- Root cause: 多余依赖 / 输出tensor构造方式不当
- Hashes: 63250f2e2
- Files: torch_npu/csrc/aten/ops/op_api/HardSwishKernelNpuOpApi.cpp, torch_npu/csrc/aten/ops/op_api/ThresholdBackwardKernelNpuOpApi.cpp
- Defect: threshold_backward的OpApi实现中使用了ApplyTensorWithFormat+CalcuOpUtil::GetTensorNpuFormat来构造输出tensor，这会显式指定NPU内部存储格式。但对于aclnn系列算子（OpApi路径），框架本身会处理格式转换，手动指定format反而可能导致格式不匹配或多余的format转换开销。HardSwishKernelNpuOpApi.cpp同样多引了CalcuOpUtil.h头文件（虽然未使用）。根因是从旧的非OpApi实现复制代码时，没有清理掉不适用于新路径的format逻辑。
- Fix: threshold_backward将ApplyTensorWithFormat(..., GetTensorNpuFormat(self))替换为ApplyTensor(self)，由框架自行决定输出format；两个文件移除多余的CalcuOpUtil.h头文件引用。
- Reviewability: high -- 单行改动，ApplyTensorWithFormat vs ApplyTensor的选择在OpApi路径有明确规则
- Review rule: OpApi路径的算子实现中不应出现CalcuOpUtil::GetTensorNpuFormat和ApplyTensorWithFormat，这属于旧AICPU/TBE路径的遗留模式

### D-974: OpApi路径遗漏correlation_id赋值

- Root cause: 初始化遗漏
- Hashes: 98b4fa2f9
- Files: torch_npu/csrc/framework/OpParamMaker.h
- Defect: OpParamMaker.h中存在两条构造ExecuteParas的路径。其中一条路径（推测为OpApi/aclnn专用路径）在填充params结构体时，遗漏了pta_correlation_id的自增赋值。这导致通过该路径下发的算子在profiling/调试时没有有效的correlation id，无法关联算子调用与性能数据，profiler trace中相应算子的关联信息缺失。
- Fix: 在遗漏的路径中补上params.pta_correlation_id = ExecuteParas::g_pta_correlation_id++。
- Reviewability: high -- 两个并行的构造路径应该有完全对称的字段初始化，对照检查即可发现
- Review rule: 当一个结构体有多条构造/填充路径时，审查应逐字段比对确认每条路径是否都做了完整初始化

### D-975: native_dropout_backward在p=0时返回全1而非梯度直通

- Root cause: 语义理解错误
- Hashes: ff425a6d3
- Files: torch_npu/csrc/aten/ops/NativeDropoutKernelNpu.cpp, test/test_nn/test_dropout_layers.py
- Defect: native_dropout_backward中，当dropout概率p=0（即不丢弃任何元素，scale=1）时，原实现返回一个全1的tensor。这在语义上是错误的：p=0意味着所有元素保留不缩放，反向传播时梯度应直接透传（返回grad_output本身），而非返回全1。当上游梯度不全为1时，全1的返回值完全丢弃了梯度信息，导致反向传播结果错误。测试中使用.sum().backward()恰好掩盖了这个bug（因为sum的梯度全为1），修复后测试也改为使用随机梯度grad来正确覆盖此场景。
- Fix: p == 0分支从NPUNativeFunctions::ones(...)改为直接return grad_output；测试从o.sum().backward()改为o.backward(grad)以暴露此类问题。
- Reviewability: medium -- 需要理解dropout backward的数学语义：p=0 -> scale=1 -> grad_input = grad_output * mask = grad_output（mask全为1）
- Review rule: dropout相关实现的边界值(p=0, p=1)测试不应使用sum().backward()，因为它的梯度恰好全为1，会掩盖返回值错误

### D-976: max_pool2d_with_indices不支持3D输入

- Root cause: 输入维度兼容性缺失
- Hashes: 33bba6dc4
- Files: torch_npu/csrc/aten/ops/pooling/MaxPool2dWithIndicesKernelNpu.cpp, torch_npu/csrc/aten/ops/pooling/MaxPool2dWithIndicesBackwardKernelNpu.cpp, torch_npu/testing/common_methods_invocations.py
- Defect: PyTorch原生max_pool2d_with_indices同时支持3D输入(C,H,W)和4D输入(N,C,H,W)，但NPU实现的底层算子MaxPoolWithArgmaxV1/MaxPoolGradWithArgmaxV1只接受4D tensor。原实现直接将输入传给算子，未处理3D情况，导致3D输入时算子报错或产生错误结果。这是NPU算子对PyTorch API contract的不完整实现。
- Fix: 在forward和backward的_out函数入口处，检测self.dim() == 3时先unsqueeze(0)升维为4D，算子执行完毕后再squeeze_(0)恢复3D；补充了TORCH_CHECK参数校验。
- Reviewability: medium -- 需要了解PyTorch pool2d API规范要求同时支持3D/4D输入，但升维/降维是标准处理模式
- Review rule: NPU算子适配PyTorch API时，需检查原生API支持的所有输入维度组合，特别是3D/4D/5D兼容的pool和conv系列算子

### D-977: NPUEvent重用时未reset导致状态残留

- Root cause: 资源生命周期管理缺陷
- Hashes: 79d70331e
- Files: torch_npu/csrc/core/npu/NPUEvent.h, torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.cpp, torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h, torch_npu/csrc/distributed/ProcessGroupHCCL.cpp, torch_npu/csrc/framework/OpParamMaker.cpp
- Defect: 在HCCL分布式通信的syncStreams中，NPUEvent被反复record和block（wait）。ACL runtime的event在record后状态变为"已触发"，如果不reset就再次record，可能因为event仍处于旧的已完成状态而导致后续的wait（block）立即返回而非真正等待新的record完成。在高频迭代的分布式训练中，这会造成流同步失效、数据竞争或通信结果不一致。
- Fix: 新增NPUEvent::reset()方法和底层的LaunchResetEventTask/ResetEventFunc全链路实现（包括异步任务队列支持），在syncStreams中block之后调用reset使event回到初始状态，确保下次record/wait语义正确。
- Reviewability: low -- 需要深入理解ACL runtime event的状态机（created -> recorded -> completed -> 必须reset才能安全重用），以及异步任务队列的分层架构
- Review rule: 复用的同步原语（event/semaphore）在每次使用周期结束后必须显式重置；审查分布式通信路径时需确认event生命周期完整

### D-978: 同源修复，参见D-977

- Root cause: 同D-977
- Hashes: f03cad93e
- Files: 同D-977
- Defect: 同源修复，cherry-pick至另一分支。代码改动内容与D-977(79d70331e)完全一致，仅基线行号不同。

### D-979: var/var_mean算子缺少correction参数重载

- Root cause: API签名不匹配
- Hashes: 5d92078d3
- Files: torch_npu/csrc/aten/ops/VarKernelNpu.cpp, torch_npu/csrc/aten/npu_native_functions.yaml
- Defect: PyTorch从某版本起为var和var_mean引入了correction参数重载（var.correction、var.correction_out、var_mean.correction），替代旧的bool unbiased接口。NPU的native functions yaml中未注册这些重载，C++实现中也没有对应签名。当用户代码或PyTorch内部通过correction参数调用var/var_mean时，dispatch不到NPU实现，导致fallback到CPU或直接报错。此外，旧实现对dim参数直接调用.value()而非.value_or()，当dim为nullopt（即计算全局方差）时会触发未定义行为。
- Fix: 在yaml中注册var.correction、var.correction_out、var_mean.correction三个重载；C++中新增以c10::optional<int64_t> correction为参数的主实现，旧的bool unbiased重载转发到新接口（转换为correction = unbiased ? 1 : 0）；所有dim.value()改为dim.value_or(at::IntArrayRef{})以支持全局方差计算。
- Reviewability: medium -- 需要对照PyTorch的native_functions.yaml确认所有重载是否都已注册，这是一个可系统化检查的问题
- Review rule: 跟随PyTorch版本升级时，需diff上游native_functions.yaml的签名变更，确认NPU侧所有重载都已同步适配

### D-980: 宏参数多次求值导致函数被重复调用

- Root cause: C宏多次求值陷阱
- Hashes: aaf042b21
- Files: torch_npu/csrc/core/npu/NPUException.h
- Defect: NPU_CHECK_ERROR、NPU_CHECK_SUPPORTED_OR_ERROR、NPU_CHECK_WARN三个宏的参数Error在宏体内被多次引用（if判断、错误消息格式化等），但宏参数不会被预先求值。当传入的是函数调用（如geInit()），该函数会被执行多次。commit message提到sympy未安装时的问题，推测geInit()之类的初始化函数通过NPU_CHECK_ERROR(geInit())调用，宏展开后geInit()在if条件和错误处理中被调用多次，导致重复初始化或触发"不能多次初始化"的断言。
- Fix: 宏内第一行用auto Error = err_code将参数求值一次并缓存到局部变量，后续引用均使用该局部变量，消除多次求值。
- Reviewability: high -- 这是经典的C/C++宏陷阱，任何宏参数被引用超过一次都应该先缓存
- Review rule: 包含逻辑控制的宏定义中，参数如果被引用多于一次，必须在do-while块开头用auto缓存；或者改用inline函数

### D-981: transfer_to_npu选择0号卡时set_device失败

- Root cause: 布尔/None判断歧义
- Hashes: 8aec0dc45
- Files: torch_npu/contrib/transfer_to_npu.py
- Defect: transfer_to_npu模块的wrapper_cuda函数中，将cuda device转换为npu device时，判断是否带卡号的逻辑为if arg.index（以及if device.index）。当用户指定cuda:0时index为0，Python中if 0为False，导致cuda:0被错误地转换为'npu'（不带卡号）而非'npu:0'，后续set_device时行为不符合预期。另外，原代码使用torch_npu._C.device做isinstance检查，但该属性可能不存在（引发module has no attribute 'device'），实际上应使用torch.device。此外还删除了一段过度hack的_isinstance全局覆写（monkey-patch了builtins.isinstance），该hack本身也有bug。
- Fix: if arg.index改为if arg.index is not None，正确区分index=0和index=None；torch_npu._C.device改为torch.device；删除_isinstance全局覆写和相关常量。
- Reviewability: high -- if x对0值的falsiness是Python常见陷阱，特别是device index场景
- Review rule: 对可能为0的数值做存在性判断时，必须使用is not None而非truthy检查；避免monkey-patch内置函数

### D-982: monkey-patch中_DDPJoinHook名称未限定导致NameError

- Root cause: 名称作用域错误
- Hashes: d85133983
- Files: torch_npu/utils/module.py
- Defect: torch_npu通过monkey-patch替换了_DDPJoinHook.__init__方法，替换函数DDPJoinHook__init__中调用super(_DDPJoinHook, self).__init__()。但_DDPJoinHook是torch.nn.parallel.distributed模块的内部类，在torch_npu/utils/module.py的作用域中并未import该名称，运行时触发NameError: name '_DDPJoinHook' is not defined。原始类方法中_DDPJoinHook能生效是因为在原模块的闭包作用域内可见，但monkey-patch后的函数运行在不同模块作用域中。
- Fix: 将super(_DDPJoinHook, self).__init__()改为super(torch.nn.parallel.distributed._DDPJoinHook, self).__init__()，使用完全限定名。
- Reviewability: high -- monkey-patch函数中使用未限定类名是明显的作用域问题，静态分析或简单测试即可发现
- Review rule: monkey-patch替换的函数体中，引用原模块的类/函数时必须使用完全限定名（或在patch函数作用域中显式import），不能依赖原模块的局部作用域

### D-983: torch.median空tensor输入未处理导致crash

- Root cause: 边界条件缺失(空tensor)
- Hashes: fd2d9dada
- Files: torch_npu/csrc/aten/ops/MedianKernelNpu.cpp, torch_npu/testing/common_methods_invocations.py
- Defect: median_out_nocheck直接对输入执行reshape({-1})再取size(0)/2，未检查输入numel是否为0。当传入shape为(2,0)或(0,2)的空tensor时，reshape和后续的sort/gather操作会因非法shape而崩溃。PyTorch原生CPU实现对空输入返回NaN标量，NPU适配遗漏了这个边界。
- Fix: 在函数入口添加numel() <= 0的early return，返回NaN填充的标量tensor
- Reviewability: high -- 空tensor是算子适配的标准检查项，review时扫描所有算子入口是否处理了zero-element输入即可发现
- Review rule: NPU算子适配必须检查空tensor(numel==0)和scalar输入的边界情况，与PyTorch CPU行为对齐

### D-984: masked_fill_输入tensor跨设备(CPU/NPU)未处理

- Root cause: 跨设备tensor未适配
- Hashes: 2cbf6d785
- Files: torch_npu/csrc/aten/ops/MaskedFillKernelNpu.cpp
- Defect: masked_fill_(self, mask, value)的value参数是tensor重载版本。当value是一个CPU上的0-d scalar tensor时（常见于tensor.masked_fill_(mask, torch.tensor(0.5))这种用法），代码直接将CPU tensor传给NPU算子，导致设备不匹配错误。PyTorch原生实现在这种场景下会隐式取value.item()走scalar路径。
- Fix: 在函数入口检测value为0-d且非NPU设备时，调用value.item()转发到scalar重载版本
- Reviewability: medium -- 需要理解PyTorch的多重载分发机制，知道scalar tensor经常停留在CPU上
- Review rule: 接受tensor参数的NPU算子，需考虑0-d CPU tensor作为"伪scalar"传入的跨设备场景

### D-985: gather算子对half dtype中间结果使用了带数据的tensor而非空tensor

- Root cause: 中间tensor复用导致数据污染
- Hashes: ad98a6df3
- Files: torch_npu/csrc/aten/ops/GatherKernelNpu.cpp
- Defect: 当self为Half dtype时，代码需要将result也转为Float做计算。原实现用npu_dtype_cast(result, Float)做类型转换，这会将result现有数据转换后传给算子作为输出buffer。问题在于result可能包含脏数据或未初始化内容，而gather的输出应完全由算子写入决定，不应依赖输出buffer的现有内容。
- Fix: 将npu_dtype_cast替换为ApplyTensor（分配同shape同device的空tensor），确保输出buffer不携带历史数据
- Reviewability: medium -- 需要理解npu_dtype_cast(转换现有数据)和ApplyTensor(分配空buffer)的语义差异
- Review rule: NPU算子的输出tensor应使用ApplyTensor分配空buffer，避免用dtype_cast等带数据拷贝的方式构造输出tensor

### D-986: index算子图模式guard作用域过大导致int/long index中断计算图

- Root cause: 图模式guard作用域错误
- Hashes: 84eafd702
- Files: torch_npu/csrc/aten/ops/IndexKernelNpu.cpp, torch_npu/csrc/framework/utils/KernelNpuOutputSize.cpp
- Defect: 原实现在index算子入口处无条件切换到SINGLE_OP_MODE（退出图模式），目的是为了处理bool index需要动态shape的情况。但这导致int/long类型的index也被强制切到单算子模式，中断了计算图构建，影响图模式性能。根因是guard的粒度太粗：应该只对bool index类型启用单算子模式。
- Fix: 将GraphModeGuard从index算子入口移到KernelNpuOutputSize.cpp中仅处理bool index的分支内，精确限定作用域
- Reviewability: medium -- 需要理解CANN图模式和SINGLE_OP_MODE的语义，以及不同index类型的处理差异
- Review rule: GraphModeGuard的作用域应尽可能窄，仅包裹确实需要切换模式的代码路径，避免影响无关的index类型

### D-987: 同源修复，参见D-988

- Hashes: f227d8db9
- 该commit的message为"fixed 67fadc1"，是D-988在另一分支上的cherry-pick/同步提交，diff内容相同。

### D-988: matmul的HF32默认值设错，应为False但设成了True

- Root cause: 默认值与规格不一致
- Hashes: 67fadc155
- Files: torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp, torch_npu/csrc/framework/interface/EnvVariables.cpp
- Defect: HF32(High-precision Float32)控制matmul和conv的精度模式。matmul的HF32默认值应为False（即不允许HF32降精度），但代码中allow_hf32被设为"11"（conv=1, matmul=1两位均为True），且ALLOW_CONV_HF32的hook中matmul默认值也设为"1"。这导致matmul在用户未显式配置时默认启用HF32，可能产生精度问题。注释声称"defaults to False in PyTorch 1.12 and later"但代码实际设了True。
- Fix: allow_hf32从"11"改为"10"，hook中matmul默认从"1"改为"0"，判断条件从disable改为enable
- Reviewability: high -- 注释和代码值的矛盾在review时应能直接发现（注释说False，代码设True）
- Review rule: 涉及精度模式/编译选项的默认值变更，必须确保代码值与注释说明以及上游规格三者一致

### D-989: _empty_with_format的deprecation装饰器吞掉了正常调用

- Root cause: 装饰器执行顺序/语义错误
- Hashes: b0d8eac92
- Files: codegen/templates/torch_funcs.py, torch_npu/__init__.py
- Defect: _empty_with_format同时被@wrap_torch_error_func和@torch_device_guard装饰。wrap_torch_error_func的wrapper直接raise RuntimeError，永远不调用被装饰函数。当装饰器顺序为@wrap_torch_error_func在外、@torch_device_guard在内时，torch_device_guard的装饰完全失效，函数无条件抛异常。此外wrap_torch_error_func定义在torch_funcs.py中，但在__init__.py中也需要使用，存在循环导入风险。
- Fix: 移除wrap_torch_error_func装饰器，将raise RuntimeError直接内联到函数体中；将wrap_torch_error_func搬到__init__.py避免导入问题
- Reviewability: high -- 装饰器里直接raise而不调用原函数，代码阅读即可发现语义异常
- Review rule: raise/return类装饰器必须审查是否会阻断被装饰函数及其他装饰器的执行

### D-990: 测试代码拼写错误torch.Tenser导致类型检查分支永远不执行

- Root cause: typo(拼写错误)
- Hashes: 677499e4c
- Files: test/test_ops.py
- Defect: isinstance(expected[0], torch.Tenser)中Tenser是Tensor的拼写错误。由于torch.Tenser不存在，这行代码会在运行到该分支时抛出AttributeError，导致当测试输出是Sequence[Tensor]类型时backward测试无法正确执行。这意味着所有返回tensor列表的算子的backward正确性测试实际上被跳过或报错。
- Fix: Tenser -> Tensor
- Reviewability: high -- 拼写检查或任何运行一次的测试即可发现
- Review rule: CI必须确保测试代码中所有分支至少被执行一次；对torch命名空间的属性引用可用静态分析检测拼写

### D-991: GRU forward不支持PackedSequence定长输入

- Root cause: NPU算子适配遗漏(RNN packed sequence)
- Hashes: 63654084a
- Files: torch_npu/utils/module.py, test/test_network_ops/test_gru.py
- Defect: torch_npu已对LSTM的forward做了patch以支持packed sequence在NPU上的执行（处理batch_sizes在CPU而input在NPU的跨设备问题），但GRU的forward没有做同等适配。当用户对GRU传入PackedSequence（pack_padded_sequence的输出）时，batch_sizes tensor停留在CPU上而input在NPU上，原生torch._VF.gru无法处理这种跨设备情况。
- Fix: 仿照LSTM的patch方式，新增gru_forward函数处理PackedSequence场景：当batch_sizes.device != input.device时，先将packed格式转为紧凑索引格式送入VF.gru，再将输出转回定长格式。通过monkey-patch替换torch.nn.modules.rnn.GRU.forward
- Reviewability: low -- 需要了解RNN packed sequence的内部机制和LSTM已有的patch模式，纯代码review难以发现GRU缺少相同处理
- Review rule: 对同族算子(LSTM/GRU/RNN)做NPU适配时，必须同步检查同族其他算子是否需要相同处理

### D-992: warning装饰器首次调用只打印警告不执行函数

- Root cause: 控制流逻辑错误(early return缺失)
- Hashes: 44547966f
- Files: scripts/codegen/templates/torch_funcs.py, torch_npu/__init__.py, torch_npu/utils/tensor_methods.py
- Defect: wrap_torch_warning_func和wrap_tensor_warning_func两个deprecation warning装饰器中，return func(*args, **kwargs)被错误地放在if not wrapper.warned分支内部。导致：首次调用时打印警告并返回结果（正确），但后续调用时由于warned=True跳过if分支，函数体没有return语句，返回None（丢失结果）。此外func.__name__对于装饰后的_empty_with_format函数会返回内部名称而非用户可见名称。
- Fix: 将return func(*args, **kwargs)移到if块外部，使其无论是否打印警告都执行被装饰函数；对_empty_with_format的警告信息硬编码函数名以避免__name__不准确的问题；同步修复tensor_methods.py中相同的bug
- Reviewability: high -- return语句缩进在if块内导致else路径无返回值，代码review逐行阅读即可发现
- Review rule: 装饰器中的return func(*args, **kwargs)必须位于所有条件分支的公共路径上，确保被装饰函数在任何情况下都被执行

### D-993: 环境变量名拼写错误ASCNED→ASCEND导致路径全部失效

- Root cause: typo(变量名拼写错误)
- Hashes: 8fa79690b
- Files: src/env.sh
- Defect: env.sh中将Ascend工具包根目录变量拼写为`ASCNED_BASE`（ASCEND→ASCNED），后续所有依赖该变量的路径（AICPU_PATH、FWK_HOME、PLUGIN_PATH、OP_PATH、TOOLKIT_PATH）全部指向不存在的目录。8处引用均错误。
- Fix: ASCNED_BASE → ASCEND_BASE，8处全量替换
- Reviewability: high -- 变量名拼写错误，grep即可发现；或首次运行env.sh时"command not found"即暴露
- Review rule: 环境变量定义后应在同一PR中有至少一个使用/验证路径存在的检查

### D-994: 文档中残留未解决的git merge conflict markers

- Root cause: 合并冲突残留
- Hashes: 8c5dd060e
- Files: docs/zh/PyTorch API支持清单_1.5.0.md
- Defect: 文档中保留了<<<<<<< HEAD / >>>>>>> 34d323c等git merge conflict markers，导致markdown渲染出乱码。同时修复了API文档中代码块缩进和markdown结构问题。
- Fix: 删除conflict markers保留正确版本，修复markdown缩进和blockquote层级
- Reviewability: high -- git hooks或CI中grep `<<<<<<<` 即可拦截
- Review rule: pre-commit hook检测merge conflict markers；文档PR需渲染预览

### D-995: CI脚本默认UT文件路径缺少上级目录前缀

- Root cause: 路径错误(相对路径)
- Hashes: a933102a6
- Files: ci/access_control_test.py
- Defect: `DEFAULT_UT_FILE = 'test/test_network_ops/test_add.py'`，但CI脚本运行目录在ci/下，实际test目录在上一级。导致当无变更文件需要测试时fallback到默认UT会找不到文件。
- Fix: 改为`'../test/test_network_ops/test_add.py'`
- Reviewability: high -- 只需确认脚本运行的CWD与路径的相对关系
- Review rule: CI脚本中的文件路径引用必须基于CUR_DIR或PROJECT_ROOT计算绝对路径，不依赖隐式CWD

### D-996: FunctionEvent构造参数名id改为id_event但调用侧未同步更新

- Root cause: 参数重命名不同步
- Hashes: ec5fb68ae
- Files: torch_npu/npu/profiler.py
- Defect: FunctionEvent类的构造函数参数从`id`重命名为`id_event`（可能为避免与Python内置id()冲突），但profiler.py中4处调用仍使用`id=`关键字参数。Python不会报错（因为FunctionEvent可能接受**kwargs），但参数实际未被正确传递，导致event id丢失。
- Fix: 4处`id=` → `id_event=`
- Reviewability: high -- 全局搜索旧参数名即可；IDE的rename refactoring应自动处理
- Review rule: 参数重命名时必须全局搜索所有keyword argument调用点

### D-997: codegen生成的include路径依赖output_dir子串匹配，实际路径可能不含torch_npu

- Root cause: 路径计算脆弱(字符串硬编码vs动态子串)
- Hashes: e43f2fa4f
- Files: scripts/codegen/gen_backend_stubs.py
- Defect: `output_dir[output_dir.index("torch_npu"):]`假定output_dir路径中包含"torch_npu"子串来截取相对路径生成#include。当构建目录结构变化时（如output_dir不含torch_npu前缀），会抛ValueError或生成错误的include路径。
- Fix: 直接硬编码为`torch_npu/csrc/aten/NPUNativeFunctions.h`，不依赖output_dir动态推算
- Reviewability: high -- 路径操作使用字符串index/slice而非os.path是明显的code smell
- Review rule: include路径生成应基于已知的项目结构常量，不应从运行时路径动态推算

### D-998: 多个算子UT使用assertEqual而非assertRtolEqual导致浮点比对过于严格

- Root cause: 测试断言方法误用(精度容差)
- Hashes: 388945c11
- Files: test/test_network_ops/test_ge.py及多个算子test文件
- Defect: ge/le/ne/nonzero等算子的UT使用`assertEqual`做NPU vs CPU结果比对，但NPU浮点运算可能有微小精度差异，导致测试不稳定地失败（flaky test）。同时MaskedFill等已废弃算子的kernel实现未清理。
- Fix: assertEqual → assertRtolEqual（带容差比对）；删除已废弃算子的kernel和test代码
- Reviewability: medium -- 需要理解NPU精度特性才能判断应该用哪种assert
- Review rule: NPU算子UT的数值比对统一使用assertRtolEqual，禁止对浮点结果使用assertEqual

### D-999: where算子output_size计算中变量自引用导致未定义行为

- Root cause: 变量自引用(使用未初始化的同名变量)
- Hashes: cbdf278b3
- Files: torch_npu/csrc/aten/ops/WhereKernelNpu.cpp
- Defect: `at::Tensor intSelf = NPUNativeFunctions::npu_dtype_cast(intSelf, at::ScalarType::Int)`中，intSelf在定义语句中引用自身。C++中这是UB（使用未初始化变量的值），实际应该将boolSelf转为Int类型。导致where算子的非零元素计数完全错误。
- Fix: `npu_dtype_cast(intSelf, ...)` → `npu_dtype_cast(boolSelf, ...)`
- Reviewability: high -- 定义语句中的自引用是经典C++ bug，编译器-Wuninitialized应能检测
- Review rule: 开启-Wuninitialized编译警告；变量定义与使用不应出现在同一表达式中

### D-1000: 算子dispatch路径错误：at::xxx调用绕过NPU后端dispatch

- Root cause: dispatch路径错误(at::命名空间 vs NPUNativeFunctions)
- Hashes: 83bb19062
- Files: torch_npu/csrc/aten/common/FormatCastKernelNpu.cpp, TensorFactories.cpp, 多个ops文件, 文档
- Defect: NPU算子实现内部调用`at::empty_with_format()`、`at::npu_stride_add()`等以`at::`前缀的函数。这些调用走PyTorch标准dispatch机制而非直接调用NPU实现，在某些场景（如自定义算子开发）下可能dispatch到错误的后端或触发递归dispatch。
- Fix: 批量将`at::xxx` → `NPUNativeFunctions::xxx`，共36个文件84处修改。同步更新中英文算子开发文档中的示例代码。
- Reviewability: medium -- 需要理解PyTorch dispatch机制才能判断at::调用是否正确
- Review rule: NPU算子kernel内部调用同模块函数时必须使用NPUNativeFunctions::直接调用，禁止通过at::间接dispatch

### D-1001: npu_format_cast_调用参数不匹配导致format转换目标错误

- Root cause: API参数不匹配(遗漏参数)
- Hashes: c374454b4
- Files: torch_npu/csrc/aten/common/CopyKernel.cpp, FormatCastHelper.cpp/.h, 多个ops文件
- Defect: 多处问题并存：(1) `NPUNativeFunctions::npu_format_cast_(dst)`只传了一个参数，但实际需要传入source tensor做format转换，应为`npu_format_cast_(self, dst)`。(2) `base_format_cast_nocheck`的dst参数声明为const但内部会修改它（调用set_和copy_memory_）。(3) Lt/Mm算子用`ApplyTensor(outputSize, options)`但该重载期望Tensor参数，应用`ApplyTensorWithSizes`。
- Fix: 修复API调用签名、const correctness、ApplyTensor → ApplyTensorWithSizes
- Reviewability: high -- format_cast_只传一个参数时编译器不会报错（若有默认参数），需review者熟悉API签名
- Review rule: 变更函数签名时必须全局搜索所有调用点；const参数内不应有修改操作

### D-1002: layer_norm测试变量名shadowing Python内置input + 类型声明规范化

- Root cause: 代码规范(变量名shadowing + namespace限定)
- Hashes: 88dcc7576
- Files: test/test_network_ops/test_layer_norm.py, test_layer_norm_backward.py, LayerNormKernelNpu.cpp, LayerNormBackwardKernelNpu.cpp
- Defect: 测试代码中函数参数命名为`input`，shadowing Python内置函数。C++侧混用`Tensor`和`at::Tensor`类型声明（在namespace外应使用完整限定名）。虽然不直接导致runtime错误，但`input`作为参数名在某些linter中会告警，且不一致的类型限定影响代码可读性。
- Fix: `input` → `input1`；`Tensor` → `at::Tensor`
- Reviewability: high -- linter/IDE即可检测
- Review rule: Python代码禁止使用与内置函数同名的参数名；C++代码在头文件和非namespace块中使用完整类型限定

### D-1003: cast_weight圈复杂度过高且children判断逻辑错误

- Root cause: 重构修复(圈复杂度+条件逻辑错误)
- Hashes: cfbe896ea
- Files: torch_npu/utils/module.py
- Defect: cast_weight函数圈复杂度过高（所有module类型判断平铺在一个函数中）。此外`if self.children() is not None`永远为True（children()返回iterator不为None），应为`if not self.children`检查是否有子模块。
- Fix: 提取内部`_format_cast(module, class_name)`函数降低圈复杂度；修复children检查逻辑
- Reviewability: medium -- 圈复杂度工具可自动检测；children()条件需要理解Python iterator语义
- Review rule: `self.children() is not None`是常见错误模式，应检查`len(list(self.children()))>0`或直接for循环

### D-1004: LayerNorm通过子类替换导致递归初始化和monkey-patch冲突

- Root cause: 设计缺陷(monkey-patch策略错误)
- Hashes: 70603c3e6
- Files: torch_npu/__init__.py, torch_npu/utils/__init__.py, torch_npu/utils/module.py
- Defect: 通过`torch.nn.LayerNorm = LayerNorm`（子类替换）来注入NPU特化的forward逻辑。但LayerNorm被多个位置引用（nn.modules.normalization.LayerNorm, nn.modules.LayerNorm, nn.LayerNorm），需要替换3处。且子类方式会导致isinstance检查、pickle序列化、和其他模块的LayerNorm引用出现不一致。
- Fix: 放弃子类替换，改用直接替换forward方法：`torch.nn.LayerNorm.forward = layernorm_forward`。删除LayerNorm子类和nn_monkey_patches列表。
- Reviewability: medium -- 需要理解Python MRO和monkey-patch的影响范围
- Review rule: nn.Module的NPU适配优先使用method替换而非class替换，避免多引用路径不一致

### D-1005: topk算子在非1维tensor上dim参数处理可能不正确

- Root cause: 算子重构(代码结构+边界条件)
- Hashes: 762f2a273
- Files: torch_npu/csrc/aten/ops/TopKKernelNpu.cpp
- Defect: topk算子实现重构，原代码在多维tensor情况下对dim参数的处理存在问题。大量代码缩进和花括号风格不规范影响可读性，掩盖了实际的逻辑问题。重构后规范化了代码结构但保持了核心算法不变。（注：该commit主要是重构，具体bug描述在commit message中未详细说明）
- Fix: 完整重写TopKKernelNpu.cpp，规范namespace缩进，重新组织条件分支
- Reviewability: low -- 大规模重构混合bug fix，需要逐行diff才能定位实际修复点
- Review rule: 重构和bug fix应分开提交；纯重构commit不应包含逻辑变更

### D-1006: NPUEvent.h include路径指向已迁移的旧位置

- Root cause: include路径过期(模块迁移)
- Hashes: 3c03ae440
- Files: torch_npu/csrc/npu/Event.h
- Defect: `#include <ATen/npu/NPUEvent.h>`引用了ATen/npu下的头文件，但NPUEvent已迁移到c10/npu模块。导致编译失败或引入错误版本的头文件。
- Fix: `ATen/npu/NPUEvent.h` → `c10/npu/NPUEvent.h`
- Reviewability: high -- 编译失败即暴露；或grep所有include ATen/npu检查是否有已迁移的
- Review rule: 模块迁移后必须全局搜索旧路径的所有include引用

### D-1007: 多个算子test中input变量名shadowing + 删除已失效测试代码

- Root cause: 测试代码清理(变量名shadowing + dead code)
- Hashes: 36428b7e7
- Files: test/test_network_ops/test_atan.py, test_argmin.py(deleted), CeluKernelNpu.cpp, CdistKernelNpu.cpp等14文件
- Defect: 与D-1002同类问题：atan/ceil/celu等算子test中参数名为`input`。此外argmin的test中实际测试的是sigmoid而非argmin（copy-paste错误），cdist backward的kernel有独立实现但应复用forward逻辑。
- Fix: input → input1；删除错误的argmin test和冗余的backward kernel实现
- Reviewability: high -- argmin test中测试sigmoid是明显的copy-paste错误
- Review rule: 测试函数名与实际测试的算子必须一致；test中不应出现与被测算子无关的逻辑

### D-1008: apply_patch.sh中cp命令行尾多余空格导致参数解析异常

- Root cause: shell脚本尾随空格
- Hashes: 9c0ab7ef8
- Files: patch/apply_patch.sh
- Defect: `cp -r $ROOT_DIR/third_party/* $PYTORCH_DIR/third_party/* `行尾有一个多余空格。在某些shell版本中，行尾空格可能导致glob展开异常或"No such file or directory"错误。
- Fix: 删除行尾空格
- Reviewability: high -- .editorconfig或trim_trailing_whitespace即可预防
- Review rule: 配置editorconfig统一trim trailing whitespace

### D-1009: DDP forward未处理NPU device_type导致走错分支

- Root cause: 设备类型判断遗漏(NPU)
- Hashes: e825d02dd
- Files: torch_npu/utils/module.py
- Defect: PyTorch原生DDP forward中`if self.device_ids`分支假设device为cuda，会调用scatter/gather进行多卡并行。当device_type为"npu"时也进入该分支，但NPU不支持原生的scatter/gather（至少在当时），导致运行时错误。
- Fix: 新增完整的ddp_forward函数，在device_ids分支中增加`self.device_type != "npu"`判断，NPU走else分支直接调用module；通过monkey-patch替换DDP.forward
- Reviewability: medium -- 需要理解DDP的device_ids/device_type语义和NPU的scatter支持状态
- Review rule: 所有涉及device_type的条件分支必须明确处理"npu"情况

### D-1010: YoloBoxesEncode的stride参数dtype应为int32而非float32

- Root cause: 算子输入dtype错误
- Hashes: 68c95af3c
- Files: test/test_network_ops/test_yolo_boxes_encode.py, torch_npu/csrc/aten/ops/YoloBoxesEncodeKernelNpu.cpp
- Defect: YoloBoxesEncode算子的stride参数在测试中使用float32，但CANN底层YoloV3DetectionOutput算子要求stride为int32。float32的stride值在传给底层算子时可能被错误解释或精度丢失。
- Fix: stride tensor改用`dtype=torch.int32`；修正测试变量命名使其更清晰
- Reviewability: medium -- 需要查阅CANN算子API文档确认参数dtype要求
- Review rule: NPU算子的输入dtype必须与CANN算子接口文档完全匹配，UT中硬编码dtype值

### D-1012: 多个算子修复：dtype比较误用+dispatch路径错误+参数顺序错误

- Root cause: 多类混合修复(dtype比较/dispatch/参数顺序)
- Hashes: 9910354a6
- Files: test/test_network_ops/test_upsample_nearest2d.py, test_upsample_nearest2d_backward.py, FloorKernelNpu.cpp, SortKernelNpu.cpp, UpsampleBilinear2dBackwardKernelNpu.cpp, npu_native_functions.yaml
- Defect: (1) `if cpu_input == torch.float16`比较整个tensor与dtype枚举值，永远为False，应为`cpu_input.dtype == torch.float16`。(2) sort算子内部调用`at::npu_transpose`应为`NPUNativeFunctions::npu_transpose`。(3) `npu_transpose_out`参数顺序(output, input, perm)与实际签名(input, perm, output)不匹配。(4) upsample_bilinear2d的.vec变体在yaml中注册但实际不支持。
- Fix: 修复dtype比较、dispatch路径、参数顺序、移除不支持的yaml注册
- Reviewability: high(dtype比较) / medium(参数顺序) -- tensor与枚举值的==比较应被linter检测
- Review rule: dtype检查必须使用`.dtype == `而非直接`==`；函数调用参数顺序需与声明签名逐一核对

### D-1011: index算子output format未指定+random算子self类型不匹配+cumsum死代码

- Root cause: 多算子修复(output format遗漏 + 类型转换 + dead code)
- Hashes: 48e7068db
- Files: torch_npu/csrc/aten/ops/IndexKernelNpu.cpp, RandomKernelNpu.cpp, WhereKernelNpu.cpp, test/test_network_ops/test_cumsum.py(deleted)
- Defect: (1) IndexKernelNpu的output tensor未指定正确的npu format，可能导致后续算子读到非预期的internal format。(2) RandomKernelNpu中self tensor的dtype转换逻辑不完整。(3) cumsum的UT测试代码已无对应kernel实现（算子已移除），属dead code。
- Fix: Index增加format设置；Random修复类型转换路径；删除orphan cumsum test
- Reviewability: medium -- output format遗漏需要了解NPU format机制；dead test需要检查kernel是否存在
- Review rule: 新增算子kernel时必须显式设置output tensor的format；删除kernel时同步删除对应UT

### D-1013: SortWithoutIndices的out变体缺少output tensor校验和contiguous处理

- Root cause: 算子out变体缺少CheckOut/contiguous校验
- Hashes: 6112da8da
- Files: torch_npu/csrc/aten/ops/SortWithoutIndicesKernelNpu.cpp, test/test_network_ops/test_sort_without_indices.py
- Defect: `npu_sort_v2_out`直接将output tensor传给OpCommand执行SortV2算子，未经过CheckOut校验（检查output tensor shape/dtype是否匹配），也未处理output tensor可能非contiguous的情况。当用户传入non-contiguous或shape不匹配的output时，行为未定义。测试也仅覆盖了部分descending方向。
- Fix: 提取inner函数`npu_sort_v2_out_nocheck`，在`npu_sort_v2_out`中增加CheckOut + format_contiguous处理；非out变体直接调用nocheck跳过冗余校验。测试重构为矩阵式覆盖所有shape x descending组合。
- Reviewability: medium -- out变体必须有CheckOut是NPU算子模板规范，review时可检查是否遗漏
- Review rule: 所有算子的_out变体必须包含CheckOut + contiguous处理模板

### D-1014: grid_sampler仅实现forward未实现backward + 测试dtype不一致

- Root cause: 算子反向传播未实现 + 测试dtype错配
- Hashes: b83f8cbe5
- Files: torch_npu/csrc/aten/ops/GridSamplerKernelNpu.cpp(deleted), test/test_network_ops/test_grid_sampler.py, test_grid_sampler_backwawrd.py(new)
- Defect: grid_sampler算子只有forward kernel（GridSampler2D），完全没有backward实现。调用backward时会报错或得到零梯度。此外fp16测试中sample tensor错误地使用float32 dtype，导致fp16路径实际测的是mixed precision而非纯fp16。
- Fix: 删除旧的独立GridSamplerKernelNpu.cpp，改为复用grid_sampler_2d的实现路径（该路径已有backward支持）。修正测试中sample dtype与input一致。新增独立backward测试。
- Reviewability: medium -- forward-only缺少backward需要review时检查autograd注册；dtype不一致可通过对比test参数发现
- Review rule: 新增支持autograd的算子必须同时提供forward和backward测试；测试中所有tensor的dtype必须与测试目标dtype一致

### D-1015: cast_weight遗漏BatchNorm3d + running_stats/weight属性存在性未检查

- Root cause: 模块类型覆盖遗漏 + 属性存在性检查缺失
- Hashes: 573645c2d
- Files: torch_npu/utils/module.py
- Defect: `cast_weight`函数的format cast逻辑存在多个缺陷：(1) BatchNorm分支只处理了BatchNorm2d/1d，遗漏了BatchNorm3d。(2) 无条件访问`module.running_mean/running_var`，当`track_running_stats=False`时这些属性为None，导致crash。(3) Conv2d的group判断条件过于复杂且不正确。(4) MultiheadAttention的weight判断使用truthiness（`module.q_proj_weight`）而非`is not None`，空tensor会被错误跳过。
- Fix: 添加BatchNorm3d；增加`track_running_stats`守卫；简化Conv2d group判断为`groups > 1`直接return；改为`is not None`精确判空
- Reviewability: high -- 逐个issubclass分支审查即可发现类型遗漏和属性守卫缺失
- Review rule: 涉及nn.Module子类的分支处理必须覆盖同族所有变体（1d/2d/3d）；访问可选属性前必须检查其存在性

### D-1016: Conv3d output tensor未指定NDC1HWC0 format + 空tensor创建方式错误

- Root cause: output tensor NPU format未指定
- Hashes: c874a1049
- Files: torch_npu/csrc/aten/ops/convolution/Conv3dKernelNpu.cpp, test/test_network_ops/test_conv3d.py
- Defect: `slow_conv3d_forward`创建output tensor时使用`ApplyTensor(self, outputSize)`，未指定format为`ACL_FORMAT_NDC1HWC0`（Conv3d在NPU上的标准format）。同时finput/fgrad_input（占位空tensor）使用`ApplyTensor(self, {0})`，会错误继承self的format metadata，应使用`ApplyTensorWithSizes({0}, self.options())`创建纯空tensor。
- Fix: output改用`ApplyTensorWithFormat(..., ACL_FORMAT_NDC1HWC0)`；空tensor改用`ApplyTensorWithSizes`
- Reviewability: medium -- 需要知道Conv3d在NPU上的format约定
- Review rule: 卷积类算子的output format必须显式指定对应的NPU内部format；空占位tensor不应继承input的format

### D-1017: NPU上pinned_memory创建方式不兼容 + device index解析对纯数字字符串失败

- Root cause: 设备兼容性(pinned memory创建) + 字符串解析异常
- Hashes: f5801a3c3
- Files: torch_npu/csrc/distributed/reducer.cpp, torch_npu/npu/utils.py
- Defect: (1) `reducer.cpp`中使用`options.pinned_memory(true)`创建tensor，但NPU device不支持在tensor options中直接设置pinned_memory标志，需要先创建普通tensor再调用`.pin_memory()`。(2) `_get_device_index`接收纯数字字符串"0"时会创建`torch.device("0")`导致device解析失败，应直接`int()`转换。
- Fix: `options.pinned_memory(true)` → `options).pin_memory()`；添加`"npu" not in device`分支直接返回int
- Reviewability: medium -- pinned_memory兼容性需了解NPU设备限制；字符串解析问题可通过edge case测试暴露
- Review rule: NPU环境不可在TensorOptions中设置pinned_memory，必须后置调用.pin_memory()

### D-1018: Roll算子源文件双.cpp后缀导致编译异常

- Root cause: 文件命名错误(双后缀)
- Hashes: bc754be07
- Files: torch_npu/csrc/aten/ops/RollKernelNpu.cpp.cpp -> RollKernelNpu.cpp, test/test_network_ops/test_roll.py, test_roll_6d.py
- Defect: `RollKernelNpu.cpp.cpp`文件名带有双`.cpp`后缀，可能导致CMake/编译系统无法正确识别为C++源文件，或在某些构建配置下被跳过。测试方法签名携带冗余`device="npu"`参数（pytest不会传递该参数，但也无害）。
- Fix: rename为正确的`.cpp`后缀；移除测试中冗余device参数
- Reviewability: high -- 文件列表中双后缀在review时一目了然
- Review rule: CI应包含检查源文件后缀合法性的lint规则；测试方法不应有unused参数

### D-1019: AsStrided copy不支持unmatched tensor但未做前置校验

- Root cause: tensor内存布局前置条件校验缺失
- Hashes: 436d42247
- Files: torch_npu/csrc/aten/common/CopyKernel.cpp, test/test_trans_contiguous/test_special_cases_copy_to_contiguous.py
- Defect: `copy_d2d_dtype_baseformat`中，当src非contiguous且优化路径失败时，直接调用`npu_stride_copy_out`（AsStrided copy）。但AsStrided copy要求目标tensor的StorageDesc metadata与实际内存布局匹配（"matched"）。当self是expand产生的unmatched tensor（如`zeros((2,10)).bool()`的切片赋值场景），stride copy会产生错误结果。
- Fix: 添加`StorageDescHelper::MetaDataAreMatch(&self)`守卫，unmatched tensor跳过stride copy走通用fallback路径
- Reviewability: low -- 需要深入理解NPU的StorageDesc metadata匹配语义和expand tensor的内存布局特征
- Review rule: 涉及stride/memory layout操作的代码路径必须校验tensor的metadata一致性

### D-1020: torch.save使用hasattr("cpu")过宽匹配导致非Tensor对象被错误处理

- Root cause: 类型判断过宽(hasattr vs isinstance)
- Hashes: 5cb59101e
- Files: torch_npu/utils/serialization.py, test/test_api/test_torch/test_serialization.py
- Defect: `to_cpu`函数使用`hasattr(value, "cpu")`判断值是否需要转到CPU。任何恰好有`.cpu`属性/方法的对象（如argparse.Namespace）都会被错误匹配并调用`.cpu()`导致异常。另外`torch.save`的入口不支持`argparse.Namespace`中嵌套的NPU tensor。
- Fix: `hasattr(value, "cpu")` → `isinstance(value, torch.Tensor)`精确类型检查；为argparse.Namespace添加专门的save分支
- Reviewability: high -- `hasattr`过宽匹配是经典反模式，review时应要求isinstance
- Review rule: 类型分支判断优先使用isinstance而非hasattr/duck typing；to_cpu/to_device工具函数必须精确匹配Tensor类型

### D-1021: to()操作dtype转换走了冗余的copy+format路径

- Root cause: dtype转换路径选择不当
- Hashes: 229a73b32
- Files: torch_npu/csrc/aten/common/ToKernelNpu.cpp
- Defect: `NPUNativeFunctions::to`在仅做dtype转换时调用了`to_impl_npu`，后者会执行完整的tensor copy + memory format处理。但纯dtype cast不需要这些额外操作，且可能引入不必要的内存分配和数据拷贝。
- Fix: 直接调用`NPUNativeFunctions::npu_dtype_cast(self, dtype)`替代`to_impl_npu`全路径
- Reviewability: medium -- 需要理解to_impl_npu包含了哪些额外操作
- Review rule: dtype-only转换应走npu_dtype_cast快速路径，避免不必要的tensor copy

### D-1022: GridSampler2D算子属性类型传string而非int导致精度问题

- Root cause: CANN算子属性类型不匹配(string vs int)
- Hashes: c24bd882b
- Files: torch_npu/csrc/aten/ops/GridSampler2dKernelNpu.cpp, test/test_network_ops/test_grid_sampler_2d.py
- Defect: GridSampler2D算子的interpolation_mode和padding_mode属性通过string数组映射传递（如"bilinear"/"nearest"），但CANN侧该算子实际接受int类型枚举值。string被传递后可能被CANN错误解析或使用默认值，导致插值模式不正确，引起精度偏差。
- Fix: 删除string映射数组，直接将int值传给Attr；扩大测试grid范围(-1,1)→(-3,3)以覆盖边界填充行为
- Reviewability: medium -- 需要对照CANN算子文档确认属性应为int还是string
- Review rule: NPU算子属性类型必须与CANN算子接口文档严格一致；新增算子时必须标注属性类型的文档来源

### D-1023: H2D copy临时tensor错误继承source的非base format

- Root cause: 临时tensor format继承错误
- Hashes: cb17f984b
- Files: torch_npu/csrc/aten/common/CopyKernel.cpp
- Defect: `copy_h2d`处理非base format的self tensor时，先创建临时dst tensor再进行host→device拷贝，最后format_cast回self的format。但`ApplyTensor(self)`创建的临时tensor会继承self的非base format（如FRACTAL_NZ），导致后续base format copy操作目标format不匹配。
- Fix: `ApplyTensor(self)` → `ApplyTensorWithSizes(self.sizes(), self.options())`，确保临时tensor是base format
- Reviewability: low -- 需要理解ApplyTensor vs ApplyTensorWithSizes在format继承上的差异
- Review rule: 需要base format临时tensor时必须使用ApplyTensorWithSizes；ApplyTensor会继承source的format

### D-1024: scatter算子dtype对齐逻辑放在self cast之前导致src dtype不同步

- Root cause: dtype转换时序错误
- Hashes: e33763bce
- Files: torch_npu/csrc/aten/ops/ScatterKernelNpu.cpp, test/test_network_ops/test_scatter.py
- Defect: scatter算子在`scatter_`入口处将src dtype对齐到self dtype，然后调用`scatter_npu_src_impl`。但impl内部会先将self从half cast为float，此时src仍然是half（因为入口处对齐的是cast前的self dtype），导致CANN算子收到self(float) + src(half)的dtype不匹配输入。
- Fix: 将src dtype对齐逻辑从scatter_入口移到scatter_npu_src_impl内部，在self完成cast之后再对齐src dtype
- Reviewability: medium -- 需要跟踪dtype cast在调用链中的传播顺序
- Review rule: dtype对齐逻辑必须放在所有相关tensor完成cast之后；多步dtype转换需画出cast顺序图

### D-1025: profiler.py使用open()写文件无权限控制 + 残留CUDA dead code

- Root cause: 文件安全(权限控制) + dead code清理
- Hashes: e8018c5cb
- Files: torch_npu/npu/profiler.py
- Defect: (1) `export_chrome_trace`使用`open(path, 'w')`写trace文件，未设置文件权限，在共享环境中可能被其他用户读取敏感profiling数据。(2) 包含完整的CUDA kernel trace逻辑（从cpu_to_cuda flow arrows到CUDA functions pid），在NPU环境中完全无用。
- Fix: 改用`os.fdopen(os.open(path, flags, mode), 'w')`显式设置权限为owner-only(0o600)；删除self._use_cuda分支的dead code
- Reviewability: high -- 静态分析可检测open()缺少权限设置；CUDA代码在NPU仓库中是明显的dead code
- Review rule: 文件写入必须使用os.open设置最小权限；fork自CUDA代码库的文件需彻底清理CUDA专属逻辑

### D-1026: codegen脚本函数嵌套层级过深(codecheck告警)

- Root cause: 代码质量(圈复杂度/嵌套深度)
- Hashes: a506480a3
- Files: scripts/codegen/api/autograd.py, api/cpp.py, api/dispatcher.py, api/native.py, api/python.py, api/signature.py, api/structured.py, api/translate.py, api/types.py
- Defect: codegen api模块多个函数嵌套层级超过codecheck阈值。`match_differentiability_info`中内联的strides校验逻辑、`arg_parser_unpack_method`中大量elif条件分支均导致"huge depth"告警。
- Fix: 提取`assert_strides_or_error`独立函数；将`arg_parser_unpack_method`的base type和list type解析分别提取为`unpack_base`和`unpack_list`子函数
- Reviewability: high -- 纯重构，不改变运行时行为
- Review rule: codegen脚本应定期运行复杂度检查，函数嵌套不超过4层

### D-1027: Embedding算子output未继承weight的NPU format

- Root cause: output tensor NPU format丢失
- Hashes: 74bf07651
- Files: torch_npu/csrc/aten/ops/EmbeddingKernelNpu.cpp
- Defect: Embedding算子创建output tensor时使用`ApplyTensorWithSizes(outputSize, weight.options())`，虽然继承了dtype和device，但未继承weight tensor的NPU internal format。后续算子如果期望特定format（如FRACTAL_NZ），会遇到format不匹配。
- Fix: 改用`ApplyTensorWithFormat(outputSize, weight.options(), CalcuOpUtil::get_tensor_npu_format(weight))`显式继承weight的format
- Reviewability: medium -- 需要了解embedding output与weight的format关系
- Review rule: 算子output format应与关键input的format保持一致；使用ApplyTensorWithFormat而非ApplyTensorWithSizes

### D-1028: test_embedding_bag_backward测试的是embedding而非embedding_bag

- Root cause: 测试代码与被测算子完全不匹配
- Hashes: f7ba314b3
- Files: test/test_network_ops/test_embedding_bag_backward.py
- Defect: 文件名为`test_embedding_bag_backward`，但实际测试代码调用的是`F.embedding`（普通embedding），不是`F.embedding_bag`。类名也是`TestEmbeddingBackward`而非`TestEmbeddingBagBackward`。这意味着embedding_bag的backward路径完全没有被测试覆盖。
- Fix: 完全重写测试，改为使用`nn.functional.embedding_bag`，添加offsets参数，类名改为TestEmbeddingBagBackward
- Reviewability: high -- 文件名与调用的API不一致，review时对比即可发现
- Review rule: 测试文件名、类名、调用的API三者必须一致；review时必须验证测试实际执行了目标代码路径

### D-1029: div/truedivide算子未处理Int/Bool输入导致CANN执行失败

- Root cause: 算子输入dtype未预处理(Int/Bool不受支持)
- Hashes: 346e2b4d1
- Files: torch_npu/csrc/aten/ops/DivKernelNpu.cpp, TrueDivideKernelNpu.cpp
- Defect: (1) `div`算子直接将Int/Bool类型tensor传给CANN的Div算子，但CANN不支持Int32/Bool输入，导致执行失败。(2) `truedivide_out`中对Int输入做了self的Float cast，但还多做了result的Float cast，又在最后将result cast回Int。truedivide语义是"true division"，应始终返回Float，cast回Int违反语义。
- Fix: div的out和非out变体都添加Int→Float、Bool→Float输入cast；truedivide删除多余的result cast和cast-back逻辑
- Reviewability: medium -- 需要知道CANN Div不支持Int；truedivide语义需查阅PyTorch文档
- Review rule: 算子实现前必须确认CANN算子支持的dtype列表；truedivide/floor_divide等语义算子的返回dtype必须符合PyTorch规范

### D-1030: codegen/testing/ci多模块dead code + import顺序 + 函数深度(codecheck)

- Root cause: 代码质量(dead code + import规范 + 函数深度)
- Hashes: 630686cec
- Files: ci/pytorch_resnet.py, scripts/codegen/api/python.py, context.py, gen.py, gen_backend_stubs.py, gen_python_functions.py, model.py, torch_npu/testing/common_utils.py, torch_npu/utils/utils.py
- Defect: (1) ci脚本import顺序不规范（torch_npu应在torch.npu之后）。(2) codegen脚本函数嵌套和条件分支过深。(3) common_utils.py中约140行dead code（未使用的辅助函数）。
- Fix: 调整import顺序；重构深层嵌套为子函数；删除common_utils.py中的dead code
- Reviewability: high -- 纯代码质量修复，linter/codecheck工具可自动检测
- Review rule: CI应强制import排序（isort）；定期运行dead code检测

### D-1031: assertRtolEqual中numpy.bool_与Python bool的is比较永远失败

- Root cause: numpy.bool_ vs Python bool的identity比较陷阱
- Hashes: 945b115c0
- Files: torch_npu/testing/testcase.py
- Defect: `result_rtol.all() is False`和`result_atol.all() is False`使用`is`比较numpy.bool_返回值。numpy的`all()`返回numpy.bool_类型，不是Python原生bool。`is False`比较的是对象identity，numpy.bool_(False)与Python的False是不同对象，所以比较永远为False。这意味着assertRtolEqual的精度校验实质上被跳过，所有case都会通过。
- Fix: `is False` → `== False`（使用值比较而非identity比较）
- Reviewability: high -- `is False`用于非Python-bool类型是经典陷阱，linter规则（如pylint的singleton-comparison）可检测
- Review rule: 禁止对numpy返回值使用is True/is False；精度校验函数需有自身的meta-test验证其能正确报错

### D-1032: 多模块函数嵌套层级/方法体过长触发codecheck告警

- Root cause: 代码质量(函数深度/方法体长度)
- Hashes: ee96aab88
- Files: ci/access_control_test.py, scripts/codegen/api/autograd.py, api/cpp.py, api/structured.py, dest/register_dispatch_key.py, gen_python_functions.py, model.py, selective_build/selector.py, setup.py, torch_npu/npu/amp/grad_scaler.py, npu/utils.py, testing/decorator.py, testing/testcase.py
- Defect: 13个文件的函数嵌套超过"huge depth"阈值或方法体超过"huge method"阈值。包括codegen的match_differentiability_info（深层forward derivative处理）、testcase.py的assertRtolEqual（大量elif分支）、grad_scaler.py的step方法等。ci脚本中`subprocess.Popen(cmd, shell=True)`也是安全隐患。
- Fix: 提取子函数降低嵌套；early return替代深层else；Popen改用list参数避免shell=True
- Reviewability: high -- 纯代码质量重构 + 安全修复（shell=True）
- Review rule: 函数嵌套不超过4层；方法体不超过100行；禁止subprocess中使用shell=True

### D-1033: 自定义cast算子未注册StridedShard分片策略导致DTensor分布式运行失败

- Root cause: 分布式算子分片策略注册遗漏
- Hashes: b956cdc34
- Files: torch_npu/distributed/tensor/_attention.py, _dtensor_utils.py, __init__.py, test/distributed/_tensor/test_attention_ops.py
- Defect: cast算子的DTensor sharding strategy注册逻辑直接复用了通用dict，但未处理`_Shard`+`StridedShard`组合场景。当用户对带StridedShard layout的tensor调用npu自定义cast算子时，`pointwise_strategy`找不到匹配的redistribution规则，运行时报错。根因是新增自定义算子时未在`_dtensor_utils.py`中注册对应的pointwise strategy（需显式传入`linearity`参数）。
- Fix: 新增`torch_npu/distributed/tensor/_pointwise_ops.py`，为cast算子显式注册带`linearity`参数的pointwise strategy；从通用注册列表中移除cast算子避免重复注册；补充UT覆盖StridedShard场景。
- Reviewability: medium -- 需要理解DTensor op dispatch机制和StridedShard语义
- Review rule: 新增自定义算子的DTensor适配时，必须验证Shard/Replicate/StridedShard三种placement的组合场景

### D-1034: Inductor codegen wrapper缺少assert_alignment符号导致特定模型编译失败

- Root cause: codegen符号导入遗漏
- Hashes: a2f6dd0ac
- Files: torch_npu/csrc/_inductor/ascend_npu_ir/npu/codegen/wrapper.py
- Defect: `PythonWrapperCodegen.write_header`生成的Python代码头部缺少`assert_alignment`的import语句。当inductor编译器在生成代码中插入alignment断言调用时，运行时抛出`NameError: name 'assert_alignment' is not defined`。该符号在上游PyTorch的`torch._inductor.runtime.hints`中定义，NPU适配层遗漏了同步。
- Fix: 在`write_header`方法的header字符串中追加`assert_alignment = torch._C._dynamo.guards.assert_alignment`赋值语句。
- Reviewability: high -- 单行遗漏，codegen header与上游对比即可发现
- Review rule: NPU inductor codegen的header/import区域每次上游PyTorch版本升级时须做diff对比，确认新增符号已同步


### D-1035: API支持度文档多处字段值与实际不符

- Root cause: 文档修复
- Hashes: f78a4af3d
- Files: docs/zh/native_apis/DDP-Communication-Hooks.md, torch-fx.md, torch-optim.md
- Defect: DDP-Communication-Hooks.md误列了不属于该表的`__getstate__`/`__setstate__`方法；torch-fx.md中`to_folder`的dtype限制描述有误；torch-optim.md中Adam备注措辞不准确。
- Fix: 删除不应出现的行，修正dtype字段，调整备注措辞。
- Reviewability: low -- 纯文档内容
- Review rule: API支持度表格修改须与CANN算子规格或UT执行结果交叉验证

### D-1036: 多模块API支持度表格字段错误(torch-nn/Storage/optim)

- Root cause: 文档修复
- Hashes: 582f4262a [+1 cherry-pick: 1cfa301c5]
- Files: docs/zh/native_apis/torch-Storage.md, torch-nn-functional.md, torch-nn.md, torch-optim.md
- Defect: 多个API支持状态或限制说明有误：`QUInt8Storage.dtype`被误标为支持；`max_unpool1d`备注写了不适用的约束；`RNNBase.flatten_parameters`被误标为支持；`TransformerEncoder`父行遗漏等。
- Fix: 逐项修正支持状态和限制说明，补充缺失行。
- Reviewability: low -- 纯文档
- Review rule: 同一文档存在多版本分支时，修复需同步cherry-pick到所有活跃分支

### D-1037: test_register_sharding测试文件过度堆积导致UT运行失败

- Root cause: 测试代码组织缺陷
- Hashes: a5bc853df
- Files: test/distributed/tensor/test_math_ops.py, test_matrix_ops.py, test_register_sharding.py
- Defect: `test_register_sharding.py`中堆积了数学运算(727行)、矩阵运算(333行)等不同类别的算子测试，与文件名语义不符。文件臃肿导致测试框架加载时出现冲突或超时，CI中UT失败。
- Fix: 按算子类别拆分：数学运算测试迁移到`test_math_ops.py`，矩阵运算测试迁移到`test_matrix_ops.py`，原文件减少约1052行。
- Reviewability: medium -- 纯文件重组无逻辑变更，但拆分后需验证所有测试仍可独立运行
- Review rule: DTensor算子测试按类别归入对应文件，单文件测试量不超过30个test case

### D-1038: aclshmem库API升级适配(Shmem_*→Aclshmem_*)

- Root cause: 第三方库接口升级适配
- Hashes: e8e435d5b
- Files: third_party/shmem/include/shmem_common_types.h(新增), shmem_host_def.h, torch_npu/csrc/distributed/symm_mem/NPUSHMEMInterface.cpp, NPUSHMEMInterface.h, NPUSHMEMSymmetricMemory.cpp, NPUSHMEMExtension.cpp, npu_sys_ctrl.cpp
- Defect: shmem库升级后函数命名从`Shmem_*`变为`Aclshmem_*`，头文件路径变更(`shmem_types.h`→`shmem_common_types.h`)，新增`aclshmemx`扩展API（uniqueid初始化模式）。旧代码调用已废弃符号，在新版库下链接失败。
- Fix: 更新所有include路径和函数调用名；通过`GET_FUNC`动态加载实现新旧版本兼容（先查新符号、fallback到旧符号）；新增uniqueid初始化路径。
- Reviewability: medium -- API重命名部分可批量审查，但dlsym fallback逻辑需逐个验证
- Review rule: 第三方库升级PR须列出完整符号映射表(旧→新)，并明确支持的库版本范围

### D-1039: 框架特性指南文档措辞与格式修正

- Root cause: 文档修复
- Hashes: fb2839ec6
- Files: docs/zh/framework_feature_guide_pytorch/下多个.md文件
- Defect: 多处技术描述过度承诺（如"彻底消除"应为"减少"）；术语不一致（"Aten IR"应为"ATen IR"）；超链接锚文本缺失（显示"LINK"而非有意义文字）；标点混用。
- Fix: 逐处修正措辞精确度、术语大小写、链接锚文本、标点一致性。
- Reviewability: low -- 纯文档
- Review rule: 性能描述禁止使用"彻底""完全"等绝对化词汇，须用可量化表述

### D-1040: torch.npu.synchronize的Dynamo trace rule分类错误(SkipFunction→InGraph)

- Root cause: Dynamo trace rule分类错误
- Hashes: 092a66d7d
- Files: torch_npu/dynamo/trace_rule.py
- Defect: `torch_npu.npu.utils.synchronize`被注册为`SkipFunctionVariable`，Dynamo trace时跳过该调用不编入计算图。在`torch.compile`场景下，synchronize语义丢失，导致设备同步缺失——profiling时间测量不准、潜在的数据竞争。正确做法是将其标记为`TorchInGraphFunctionVariable`保留为图中的叶子调用。
- Fix: 在`torch_c_binding_in_graph_functions_npu`中添加synchronize，同时删除`manual_torch_name_rule_map`中的SkipFunction注册。
- Reviewability: high -- 配置级变更，diff仅几行
- Review rule: 设备控制类函数(synchronize/stream操作)原则上必须InGraph，不得Skip

### D-1041: synchronize在trace_rules中存在InGraph与Skip的矛盾注册

- Root cause: Dynamo trace rule配置冲突
- Hashes: 750a0f616
- Files: torch_npu/dynamo/trace_rule.py
- Defect: D-1040将synchronize改为InGraph后，`manual_torch_name_rule_map`中仍残留SkipFunction条目。同一符号同时存在于两个语义矛盾的规则表中，Dynamo行为取决于查询顺序，结果不确定。
- Fix: 删除`manual_torch_name_rule_map`中synchronize的SkipFunction条目，移除已无引用的`manual_torch_name_rule_map`和`SkipFunctionVariable`的import。
- Reviewability: high -- diff极小，逻辑直观
- Review rule: trace_rule.py变更时须搜索所有规则表确认同一符号无冲突注册

### D-1042: DTensor UT硬编码4卡需求与CI 2卡环境不匹配

- Root cause: 测试与CI环境不匹配
- Hashes: 1d3f5bc9f
- Files: test/distributed/tensor/test_attention_ops.py, test_gather_swiglu.py, test_math_ops.py, test_matrix_ops.py, test_moe_ops.py, torch_npu/distributed/tensor/_attention.py, torch_npu/testing/_internal/common_dtensor.py
- Defect: 所有DTensor分布式测试使用`@skipIfUnsupportMultiNPU(4)`，但CI环境仅有2卡，导致这些测试全部被跳过。同时`_attention.py`的strategy函数签名缺少`npu_fusion_attention`新增的`sink`参数。
- Fix: `@skipIfUnsupportMultiNPU(4)`→`(2)`；追加`@SupportedDevices(['Ascend910B'])`；同步更新strategy函数签名加入`sink`参数。
- Reviewability: medium -- 散布在多文件中的decorator修改需逐一审查，但每处改动简单
- Review rule: 多卡测试的卡数要求必须与CI环境实际可用卡数对齐；算子签名变更后须grep所有strategy注册确认同步

### D-1043: PROF_CONFIG_PATH环境变量文档描述不清

- Root cause: 文档修复
- Hashes: 89384951c
- Files: docs/zh/environment_variable_reference/PROF_CONFIG_PATH.md
- Defect: 功能描述未明确说明`PROF_CONFIG_PATH`是环境变量名，语义模糊。
- Fix: 重写功能描述，首句显式点出变量名及用途。
- Reviewability: low -- 纯文档
- Review rule: 环境变量参考文档首句须包含变量名和作用

### D-1044: A3平台gcc不识别-march=native导致AOTI编译失败

- Root cause: 平台兼容性(编译flag硬编码)
- Hashes: 785e3cd2d [+1 cherry-pick: 4833cf2f0]
- Files: torch_npu/_inductor/__init__.py
- Defect: PyTorch upstream的`cpp_builder._get_optimization_cflags`在非ppc64le平台上追加`-march=native`，但Ascend A3平台的gcc不识别该flag，AOTI编译阶段直接报错。upstream未针对Ascend硬件做例外处理。
- Fix: 在torch_npu侧monkey-patch `_get_optimization_cflags`，复制upstream逻辑但删除`-march=native`行。
- Reviewability: medium -- 根因明确，但整函数monkey-patch的维护风险高（upstream修改后patch可能静默过时）
- Review rule: monkey-patch upstream函数时须在代码注释中标注与upstream的diff位置和原因，并在TODO中记录移除条件

### D-1045: npu_fusion_attention的DTensor sharding strategy逻辑错误

- Root cause: DTensor sharding strategy设计缺陷
- Hashes: 788916261
- Files: torch_npu/distributed/tensor/_attention.py, _dtensor_patch.py, __init__.py, test/distributed/tensor/test_attention_ops.py
- Defect: 原sharding strategy直接复用query的placement，当切分维度为S(sequence)或D(head)时，local计算结果拼合后与全量计算不一致。同时旧版pytorch的`register_sharding`不支持带Tensor kwargs的算子，backward路径中DTensor kwargs未参与redistribution导致梯度错误。TP场景下`head_num`参数未随sharding维度调整。
- Fix: 新增`_dtensor_patch.py`回填Tensor kwargs的redistribution支持；重写strategy限制为DP(batch维)和TP(head维)两种语义正确的切分；TP策略中动态计算`head_num`。
- Reviewability: low -- 涉及DTensor placement语义、forward/backward kwargs、跨版本API差异，需深入理解分布式并行原语
- Review rule: 注册有Tensor kwargs的算子strategy时，须验证kwargs在backward路径的redistribution；与tensor shape相关的标量参数(如head_num)须随sharding同步调整

### D-1046: NPU Triton codegen错误地将block size取整为2的幂次

- Root cause: 代码生成不必要的硬件约束
- Hashes: 913083da4 [+1 cherry-pick: 31d5bf49a]
- Files: torch_npu/_inductor/codegen/triton.py
- Defect: `NPUIndexTritonKernel`在生成`*BLOCK_SUB` constexpr时，对symbolic expression解析出的`val`调用了`next_power_of_2(val)`。NPU Triton kernel的block size不要求是2的幂次(与GPU warp对齐约束不同)，该转换会放大真实numel值，生成的constexpr与实际数据大小不匹配，可能导致越界或计算错误。
- Fix: 删除`val = next_power_of_2(val)`，直接使用原始值。
- Reviewability: high -- 单行删除，逻辑直观(NPU无warp对齐要求)
- Review rule: NPU Triton codegen中引入size/alignment约束前须注释硬件规格依据；GPU特有约束不应无条件移植到NPU

### D-1047: taskQueue关闭时NPUGraph析构调用已释放的ACL运行时

- Root cause: 对象生命周期/析构顺序依赖错误
- Hashes: 200b6e925 [+1 cherry-pick: 4a0406549]
- Files: torch_npu/csrc/core/npu/NPUGraph.cpp
- Defect: taskQueue被关闭且启用aclgraph时，ACL运行时资源(通过`NpuSysCtrl`管理)在`NPUGraph`对象析构之前已被释放。`NPUGraph::reset()`中`AclmdlRIDestroy(model_ri_)`在ACL运行时不可用时调用失败并崩溃。本质是三个模块的析构顺序依赖：taskQueue → ACL runtime → NPUGraph，但实际执行顺序不保证。
- Fix: 在`AclmdlRIDestroy`调用前增加`NpuSysCtrl::GetInstance().GetInitFlag()`守卫，仅在ACL运行时仍存活时执行销毁。
- Reviewability: medium -- diff简单但需理解三模块生命周期关系
- Review rule: 析构/reset函数中调用外部运行时API前必须检查运行时存活状态，不能假设析构顺序与初始化顺序相反

### D-1048: Inductor CI修复 -- 跳过失败用例 + 补充缺失的前向声明

- Root cause: CI修复(测试跳过 + 编译错误)
- Hashes: e3b16aade
- Files: test/_inductor/test_opensora_graph1.py, torch_npu/csrc/core/npu/interface/AclInterface.h
- Defect: 两个独立问题打包修复：(1) `test_opensora_cases_model_11_inference`用例在CI环境下持续失败，直接`@skip`跳过以恢复CI流水线；(2) `AclInterface.h`缺少`aclrtMemUsageInfo`和`aclOpExecutor`两个struct的前向声明，在某些编译配置下引发incomplete type错误。
- Fix: (1) 添加`@skip("skip to recover ci")`装饰器；(2) 在头文件namespace外添加struct前向声明。
- Reviewability: high -- 两处改动各仅1-2行，但打包在一个commit中降低了语义清晰度
- Review rule: CI修复commit应拆分为"跳过测试"和"编译修复"两个独立提交；`@skip`必须附带issue链接以防遗忘恢复

### D-1049: v2.1.0分支错误更新torchair子模块版本

- Root cause: 子模块版本管理错误(REVERT)
- Hashes: 6b134373a
- Files: third_party/torchair/torchair (submodule)
- Defect: v2.1.0维护分支不应更新torchair子模块到新版本(commit message明确说"2.1 version no need to update")。此前某commit误将torchair指向了新commitid，本commit将其revert回`a2327edaa`。
- Fix: 将submodule指针回退到正确的commitid。
- Reviewability: high -- submodule指针变更在diff中只有1行，容易被忽略
- Review rule: 维护分支(v2.X.0)的submodule更新必须有明确的版本对齐理由；PR描述应说明为何需要更新submodule

### D-1050: upstream with_comms API变更导致DTensor测试在v2.8.0上self.device设置失败

- Root cause: upstream API签名变更适配
- Hashes: 902c98ad4
- Files: 26个文件(test/distributed/_tensor/*, torch_npu/testing/_internal/common_dtensor.py, torch_npu/testing/common_distributed.py)
- Defect: PyTorch 2.8.0中`with_comms` decorator的函数原型发生变更，导致torch_npu的`with_comms`wrapper无法正确设置`self.device="npu"`。所有DTensor测试继承自upstream的`DTensorTestBase`，该基类的`setUp`方法与新版`with_comms`的调用协议不兼容。同时`@with_comms`和`@skipIfUnsupportMultiNPU`的装饰器顺序需调整(skip判断应在通信初始化之前)。
- Fix: (1) 新建`NPUDTensorTestBase`替代upstream的`DTensorTestBase`，在其中正确设置device；(2) 所有DTensor测试类改继承`NPUDTensorTestBase`；(3) 交换`@with_comms`和`@skipIfUnsupportMultiNPU`的顺序，让skip在外层。
- Reviewability: medium -- 涉及26个文件但每处改动模式相同(基类替换+装饰器顺序)
- Review rule: upstream版本升级后须检查所有monkey-patch和test基类是否与新API兼容；装饰器顺序原则：资源检查(skip)在外层，资源初始化(with_comms)在内层

### D-1051: 多行字符串拼接引入额外空格

- Root cause: 字符串格式错误(反斜杠续行缩进)
- Hashes: 0fd3db0b0 [+4 cherry-picks: 5a7912953, 9bbb3b50e, d85cc3655, c41fa1f0b]
- Files: torch_npu/npu/npu_config.py
- Defect: RuntimeError消息使用反斜杠`\`续行，第二行的缩进空格被包含在字符串中，导致用户看到的错误消息中间有大段多余空格："has already been called, \n                 You can only..."。Python的隐式字符串拼接不会strip续行的缩进。
- Fix: 将多行字符串合并为单行。
- Reviewability: high -- 单行改动，但反斜杠续行的空格陷阱容易被忽略
- Review rule: Python字符串禁止使用反斜杠续行(用括号隐式拼接或f-string)；错误消息字符串应在测试中验证内容

### D-1052: PythonTracer singleton在子线程中因GIL竞争导致死锁/崩溃

- Root cause: 线程安全/GIL竞争
- Hashes: 76f43ebf6 [+3 cherry-picks: 0daba0e58, fd25bbd40, 12f8d82cd]
- Files: torch_npu/csrc/profiler/profiler_python.cpp
- Defect: `PythonTracer::call(Command::kStartOne/kStopOne)`在子线程回调中直接调用`PythonTracer::singleton()`。`singleton()`内部使用`static`局部变量初始化，C++11保证线程安全初始化但不保证reentrancy。当主线程持有GIL正在初始化singleton时，子线程也调用`singleton()`会阻塞等待初始化完成，而`PythonTracer`构造函数中`pybind11::gil_scoped_acquire gil`会尝试获取GIL——如果主线程正持有GIL，则产生死锁。
- Fix: 新增`std::atomic<bool> instance_created_`标志和`get_singleton_in_child_thread()`方法：子线程调用路径先检查标志，未创建则返回nullptr跳过操作，避免触发singleton初始化的GIL获取。
- Reviewability: medium -- 需理解C++ static初始化、GIL交互和profiler回调时机
- Review rule: 含GIL获取的singleton构造函数不得在未持有GIL的线程中被首次触发；profiler回调路径必须是GIL-free safe

### D-1053: Inductor lowering_fx重复import + dump_fx_graph hash碰撞 + div常量优化缺失

- Root cause: 代码质量(重复import) + 缓存hash碰撞 + 性能优化缺失
- Hashes: e53f69457
- Files: torch_npu/_inductor/lowering_fx.py, torch_npu/_inductor/codegen/scheduling.py, test/_inductor/test_lowering_fx.py, test/_inductor/test_check_accuracy.py, test/_inductor/test_force_fallback.py
- Defect: 三个独立问题：(1) `lowering_fx.py`中`torch._ops`、`ir`、`lowering`、`scheduler`、`Reduction`、`ExpandView`、`sympy_product`等模块和符号被重复import多达2-3次，导致代码膨胀和维护困惑；(2) `scheduling.py`中`traced_graph_hash`仅基于图的readable表示计算，不同PyTorch版本生成的图结构可能相同但语义不同，hash碰撞导致错误的cache命中；(3) NPU Triton不支持原生div指令，常量除法应转为乘以倒数以减少一次kernel调用。
- Fix: (1) 删除所有重复import；(2) hash计算追加`torch.__version__`消歧；(3) 在`aten.div`的lowering中检测常量除数，转换为`mul(a, 1.0/divisor)`；(4) 测试中添加`fx_graph_cache = False`避免缓存干扰；(5) 新增`test_lowering_fx.py`验证div→mul转换和sum不fallback。
- Reviewability: high(重复import和hash修复) / medium(div优化需理解NPU Triton指令集)
- Review rule: PR合入前用工具检查重复import；fx graph cache hash必须包含框架版本；算子lowering优化需附带正确性测试

### D-1054: FSDP root_pre_forward不识别NPU设备，host输入未自动搬移到NPU

- Root cause: 设备类型白名单遗漏
- Hashes: dddf43d3d
- Files: torch_npu/distributed/fsdp/_add_fsdp_patch.py
- Defect: upstream FSDP的`_root_pre_forward`方法中，输入tensor从host搬移到device的逻辑仅检查`self._device.type in ["cuda", "hpu"]`，不包含"npu"。当训练数据在host内存时，FSDP不会自动将输入搬到NPU，后续运算因device mismatch而失败。
- Fix: monkey-patch `FSDPState._root_pre_forward`，在设备类型白名单中添加"npu"，复制upstream逻辑并扩展检查条件。
- Reviewability: medium -- patch本身简单，但需验证与upstream逻辑的一致性
- Review rule: upstream FSDP更新时须grep所有设备类型白名单确认"npu"在列；monkey-patch函数须注释upstream版本和diff位置

### D-1055: test_set_snapshot_fn测试依赖全局allocator状态导致隔离性问题

- Root cause: 测试隔离性(全局状态污染)
- Hashes: 374288af0
- Files: test/allocator/test_pluggable_allocator_extensions.py, test/allocator/test_set_snapshot.py(新建)
- Defect: `test_set_snapshot_fn`位于`TestPluggableAllocator`类中，依赖`setUpClass`中初始化的`new_alloc`和全局allocator状态。当测试执行顺序改变或其他测试修改了allocator状态时，`memory_snapshot()`的预期行为不再成立。同时snapshot测试需要独立的allocator生命周期(创建→change→snapshot→断言)，与其他pluggable allocator测试共享setUp会互相干扰。
- Fix: 将`test_set_snapshot_fn`拆分到独立的`test_set_snapshot.py`文件，拥有独立的`setUpClass`和allocator初始化逻辑，实现完全隔离。
- Reviewability: high -- 拆分逻辑直观，但需确认新文件的CI集成
- Review rule: 依赖全局可变状态(allocator/device配置)的测试应独立为单独文件，不与其他测试共享setUp

### D-1056: NSLB-DP配置字段写入错误的struct成员(copy-paste错误)

- Root cause: copy-paste错误(config字段赋值目标错误)
- Hashes: a6e4bbbbc [+4 cherry-picks: 557b7b9b7, 393637b23, 55ab952c7, a76cc2bb2]
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `createHcclCommConfigWithOptions()`中，从`options_->hccl_config`字典读取`"hccl_world_rank_id"`和`"hccl_job_id"`的值后，都错误地赋值给了`config.hcclOpExpansionMode`而非各自对应的`config.hcclWorldRankID`和`config.hcclJobID`。结果：NSLB-DP场景下world_rank和job_id配置完全失效，而OpExpansionMode被覆盖为错误值。三个config字段全部错乱。
- Fix: 将赋值目标改为正确的struct成员名。
- Reviewability: high -- 典型的copy-paste错误，diff仅2行，但需要对HcclCommConfig struct成员有认知
- Review rule: config赋值代码中，赋值目标(lhs)必须与字典key名(rhs查找键)语义匹配；建议用统一的config映射表替代逐字段if-else

### D-1057: D2H拷贝应使用aclrtMemcpyAsyncWithCondition + SoC版本守卫

- Root cause: 硬件API能力适配(D2H拷贝语义)
- Hashes: a9a107dcf [+3 cherry-picks: 4faba86c5, 2d23bdcaa, 87ab6514c]
- Files: third_party/acl/inc/acl/acl_rt.h, torch_npu/csrc/core/npu/interface/AclInterface.cpp, torch_npu/csrc/framework/OpParamMaker.cpp
- Defect: `aclrtMemcpyAsync`在D2H场景下，如果目标host内存不是通过ACL/RTS API分配的(例如普通malloc)，实际行为是同步拷贝而非异步，导致不可预测的性能问题。`aclrtMemcpyAsyncWithCondition`专门处理此场景，能正确区分并保证异步语义。此外，该API仅在Ascend910B1及更高SoC上可用，之前的检查只判断了函数指针是否存在，未校验SoC版本。
- Fix: (1) `AclrtMemcpyAsyncWithConditionExist()`增加SoC版本检查`GetSocVersion() >= Ascend910B1`；(2) `MemcopyAsyncFunc`中对D2H场景(`ACL_MEMCPY_DEVICE_TO_HOST`)优先使用`aclrtMemcpyAsyncWithCondition`；(3) 错误日志区分两种API调用。
- Reviewability: medium -- 需理解ACL内存拷贝的同步/异步语义差异和SoC版本矩阵
- Review rule: D2H内存拷贝路径必须使用WithCondition变体(如SoC支持)；硬件API可用性检查须同时校验函数指针和SoC版本下限

### D-1058: Profiler解析器多处防御性编程缺失 + DB连接悬垂引用 + CPU活动未过滤

- Root cause: 防御性编程缺失(dict.get无默认值 + 连接生命周期 + 配置过滤缺失)
- Hashes: 6a949ae01 [+4 cherry-picks: fca6ef028, 6b7387c42, 238135598, 198982e4a]
- Files: torch_npu/profiler/analysis/_profiling_parser.py, prof_common_func/_constant.py, prof_common_func/_db_manager.py, prof_parse/_fwk_file_parser.py, prof_view/_memory_prepare_parser.py
- Defect: 五个相关问题：(1) `_profiling_parser.py`中`parser_config.get(export_type)`和`.get(self._analysis_type)`无默认值，当export_type未配置时触发`AttributeError: 'NoneType' object has no attribute 'get'`；(2) `BasicDb.close()`释放DB连接后未将`self.conn`和`self.curs`置None，后续误访问导致操作已关闭的连接；(3) `_constant.py`缺少`CPU_ACTIVITIES`常量定义，无法区分CPU/NPU profiling模式；(4) `_fwk_file_parser.py`中`get_torch_op_tree_node`/`get_fwk_trace_data`/`get_fwk_api`在CPU活动未启用时仍执行解析，浪费资源且可能因缺少数据而崩溃；(5) `_memory_prepare_parser.py`在default export db场景下缺少`return`，导致后续`Constant.Db`分支重复处理同一数据。
- Fix: (1) `.get(key, {})`/`.get(key, [])`添加空容器默认值；(2) close()后置None；(3) 新增常量`CPU_ACTIVITIES = "ProfilerActivity.CPU"`；(4) 三处函数开头检查CPU_ACTIVITIES，未启用则直接返回空列表；(5) 添加`return`提前退出。
- Reviewability: high -- 每处改动1-3行，模式明确
- Review rule: dict链式`.get().get()`必须每层提供默认值；资源close后必须置引用为None；profiler数据处理函数入口须检查对应activity是否启用

### D-1059: CANN/Driver版本解析正则无锚点 + fake_tensor测试randint范围错误 + grouped_matmul缺参数

- Root cause: 测试正确性(正则锚点缺失 + randint空范围 + 算子参数遗漏)
- Hashes: e35fc4764 [+4 cherry-picks: 222a984c0, dbe0f0bf7, 716340ef9, 1b258ef0f]
- Files: test/npu/test_cann_version.py, test/test_fake_tensor.py
- Defect: 三类问题：(1) `test_get_cann_version`中正则`re.match("([0-9]+).([0-9]+).RC([0-9]+)", version)`缺少`$`锚点，导致"8.0.RC1.alpha001.extra"也会匹配成功，未严格验证版本格式；`.`未转义，匹配任意字符而非字面量点号；(2) `test_fake_tensor.py`中多处`torch.randint(-1, -1, ...)`，high==low导致空范围异常（应为`(-1, 1, ...)`）；(3) `npu_grouped_matmul`调用缺少`group_type`参数，使用默认值但fake tensor模式下shape推断依赖该参数。
- Fix: (1) 正则添加`$`锚点并转义`.`为`\.`，补充version_env < 8.1.RC1的分支和driver版本测试；(2) `randint(-1, -1, ...)` → `randint(-1, 1, ...)`；(3) 显式传入`group_type`参数。
- Reviewability: high -- 改动模式重复，逐处核对即可
- Review rule: 版本格式验证正则必须有`$`锚点和`.`转义；`torch.randint`的low/high参数必须满足low<high；fake tensor测试须与算子真实签名保持同步

### D-1060: _npu_dtype_cast缺少DTensor sharding规则注册

- Root cause: DTensor算子注册遗漏
- Hashes: d3423e588 [+2 cherry-picks: 0a6c135b7, be4f75b7a]
- Files: torch_npu/utils/dtensor.py
- Defect: `npu_dtype_cast`有public和private两个变体(`npu.npu_dtype_cast.default`和`npu._npu_dtype_cast.default`)。DTensor规则注册表中只注册了public版本，private版本`_npu_dtype_cast`未注册。当DTensor图中出现`_npu_dtype_cast`调用时，DTensor无法找到对应的sharding propagation规则，fallback到默认行为可能产生不正确的placement推断。
- Fix: 在注册列表中添加`npu._npu_dtype_cast.default`。
- Reviewability: high -- 单行添加，但需要知道private算子变体的存在
- Review rule: 新增NPU custom op时须同时检查是否存在private变体(`_`前缀)，两者均需注册DTensor规则

### D-1061: Logger静态map在进程析构时use-after-free导致coredump

- Root cause: 静态对象析构顺序导致use-after-free
- Hashes: c4866cac6 [+4 cherry-picks: e5e270539, 5054b5702, 34e9aac21, fe195b067]
- Files: torch_npu/csrc/logging/Logger.cpp, torch_npu/csrc/logging/Logger.h
- Defect: `LoggingLevelNames`是文件级静态`std::unordered_map`，`Logger`实例也是某处的静态或全局对象。C++标准不保证不同翻译单元中静态对象的析构顺序(static destruction order fiasco)。当进程退出时，如果`LoggingLevelNames`先于`Logger`被析构，`Logger`析构过程中的日志调用访问已销毁的map，导致coredump。
- Fix: 删除`LoggingLevelNames` map，将level→string的映射改为在每个`debug()`/`info()`/`warn()`等调用点直接传入字符串字面量`"DEBUG"`/`"INFO"`等。字符串字面量的生命周期是整个程序运行期，不存在析构顺序问题。
- Reviewability: medium -- 需理解C++ static destruction order fiasco，但diff模式清晰
- Review rule: 日志系统中禁止使用可被析构的静态容器(map/vector)存储固定映射；改用constexpr数组或直接传递字面量

### D-1062: TaskQueue GIL释放引发AB-BA死锁 + aclrtSetCurrentContext隐式切换设备不一致

- Root cause: 死锁(GIL与allocator mutex交叉) + 设备状态不一致
- Hashes: 34f04920f
- Files: torch_npu/csrc/core/npu/NPUCachingAllocator.cpp, torch_npu/csrc/core/npu/NPUQueue.cpp, torch_npu/csrc/framework/OpCommand.cpp, torch_npu/csrc/framework/OpCommand.h
- Defect: 两个独立的并发问题：(1) `NPUQueue`中，当taskQueue满时释放GIL等待消费者。但如果线程A持有`deviceCachingAllocator`的mutex并进入此路径释放GIL，线程B获得GIL后触发GC → GC尝试析构tensor → 需要获取同一mutex → AB-BA死锁。(2) `NPUCachingAllocator::insert_events`中`aclrtSetCurrentContext`会隐式切换当前设备，但函数结束时恢复context后未同步恢复缓存的device index，导致后续代码认为自己在错误的设备上。
- Fix: (1) 引入全局标志`g_used_aclop`(在`OpCommand::Run`中首次使用aclop时设为true)，GIL释放路径增加`&& g_used_aclop`条件——非aclop路径不需要释放GIL给TE编译器，避免死锁。注释说明aclop即将废弃，这是过渡方案。(2) `insert_events`开头保存当前device index，结束时`SetDevice(pre_device)`恢复。
- Reviewability: low -- 需深入理解GIL-mutex交叉死锁场景和ACL context/device的隐式耦合
- Review rule: 持有非GIL锁时禁止释放GIL(可能导致AB-BA死锁)；调用aclrtSetCurrentContext后必须恢复device index(context切换有隐式副作用)

### D-1063: 动态Profiler配置文件权限无诊断 + start_step窗口错过静默失败 + DB解析变量名混淆

- Root cause: 可观测性缺失(静默失败) + 变量名错误
- Hashes: c1b4e4ded [+4 cherry-picks: a306d60f3, 81f44fbe4, 8ce5e5906, 4f5bd6d15]
- Files: torch_npu/profiler/_dynamic_profiler/_dynamic_profiler_monitor_shm.py, profiler/analysis/prof_view/prof_db_parse/_memory_db_parser.py, profiler/dynamic_profile.py, utils/_path_manager.py
- Defect: 三个问题：(1) 动态profiler读取配置文件失败时无诊断信息，用户无法判断是权限问题还是文件不存在；(2) `_DynamicProfile.step()`中当`cur_step > start_step`时(例如配置文件延迟送达)，profiling窗口已错过但代码静默跳过，用户不知道配置未生效；(3) `_memory_db_parser.py`中`STREAM_PTR`的fallback值取自`last_record`而非`last_record_data`——两个变量指向不同数据结构(raw record vs processed list)，类型不匹配可能导致解析错误。
- Fix: (1) 配置文件不可读/写时用`print_error_msg`输出具体权限修复命令；(2) `cur_step > start_step`时打印warning说明配置未生效；(3) `last_record` → `last_record_data`修正变量名；(4) `PathManager`新增`check_path_is_readable`/`check_path_is_writeable`方法。
- Reviewability: high(权限检查和warning) / medium(变量名混淆需对比两个变量的来源)
- Review rule: 配置驱动的功能在配置无法生效时必须给出诊断日志而非静默跳过；相似变量名(`last_record`/`last_record_data`)应重命名以消除歧义

### D-1064: D2H异步拷贝CachingHostAllocator记录的指针错误(view offset vs storage base)

- Root cause: 指针语义错误(data_ptr vs storage base ptr)
- Hashes: 16fb415d7 [+4 cherry-picks: d558488c5, 410df9e51, 2d200de86, 5b357efea]
- Files: torch_npu/csrc/aten/common/CopyKernel.cpp, torch_npu/csrc/aten/common/InnerNpuNativeFunction.h, torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp
- Defect: D2H非阻塞拷贝完成后需要在`CachingHostAllocator`中记录event，以确保host端pinned memory在异步拷贝完成前不被释放。记录时传入的指针必须是storage的base pointer(`allocator`分配的原始地址)。原代码使用自定义的`get_base_data_ptr()`，对view tensor调用`t._base().data_ptr()`，但`_base()`不是所有view类型都能正确追溯到原始storage，且`data_ptr()`包含view offset。结果：`CachingHostAllocator_recordEvent`接收到带offset的指针，在其内部map中找不到匹配的allocation记录，event未被正确关联。
- Fix: 用`storage().mutable_data()`替代`get_base_data_ptr()`，直接获取storage层的base pointer(无view offset)。删除不再使用的`get_base_data_ptr`辅助函数。
- Reviewability: medium -- 需理解PyTorch tensor/storage/view的指针语义
- Review rule: 与allocator/memory manager交互时必须使用`storage().data()`获取base pointer，禁止用`data_ptr()`(含view offset)

### D-1065: P2P连接计数初始值错误(off-by-one) + 限制报错指向错误设备

- Root cause: 初始值off-by-one + 错误诊断信息指向错误对象
- Hashes: 6b47857fd [+2 cherry-picks: 19083f991, 8715f494f]
- Files: torch_npu/csrc/core/npu/NPUPeerToPeerAccess.cpp
- Defect: 两个问题：(1) `device_enabled_count_`初始化为1而非0。自连接(`i*N+i`)被标记为`COPY_ALLOWED`但不消耗真实的P2P硬件通道，不应计入连接数。初始值为1导致每个设备实际只能建立`C10_P2P_ACCESS_MAX_NPUS - 1`个P2P连接，少一个。(2) 当连接数达到上限时，无论是source还是dest达到上限，错误日志总是遍历source_dev的连接列表，对dest_dev达上限的情况给出误导性诊断。
- Fix: (1) 初始值改为0，注释说明自连接不计数；(2) 识别具体哪个设备达到上限(`limited_device`)，错误日志中遍历该设备的连接列表并报告其连接详情。
- Reviewability: high -- off-by-one在初始化处一目了然
- Review rule: 连接计数的初始值必须反映真实的资源占用(自连接不计数)；错误诊断信息中引用的对象必须与实际触发条件匹配

### D-1066: StressDetect多worker场景new_group不同步挂起 + 参数API重命名

- Root cause: 分布式group创建不同步(集合通信死锁)
- Hashes: 9584c1632 [+4 cherry-picks: a373cd329, e213d0494, ba874e36c, ff901d5b7]
- Files: torch_npu/npu/utils.py, torch_npu/csrc/npu/Stress_detect.cpp, test/torch_npu_schema.json
- Defect: StressDetect在多worker(多进程)场景下，每个进程只调用`torch.distributed.new_group`创建自己rank所属的group。但`new_group`是集合通信操作，要求所有进程都参与调用(即使不加入该group)。当worker数>1时，不同进程创建的group集合不同，导致`new_group`内部的barrier永远等不齐，进程挂死。同时`mode`参数(int 0/1)语义不明确。
- Fix: 读取`WORLD_SIZE`计算worker数量，对所有worker_id都调用`new_group`(非本worker的group结果丢弃)，确保每个进程的`new_group`调用序列一致。参数从`mode=0`重命名为`detect_type='aic'`，值域改为字符串`['aic','hccs']`。
- Reviewability: medium -- 需理解分布式new_group的集合语义
- Review rule: `torch.distributed.new_group`是集合操作，所有rank必须以相同顺序调用相同数量的new_group，即使某些rank不加入某个group

### D-1067: Profiler配置解析中activities列表未传递导致下游parser崩溃

- Root cause: 配置传递链断裂(成员变量未初始化)
- Hashes: 88c1a9abb [+4 cherry-picks: 81f53e9d5, 5eac8683a, 02a3e756f, a2e91d057]
- Files: torch_npu/profiler/analysis/_profiler_config.py, torch_npu/profiler/analysis/_profiling_parser.py
- Defect: `ProfilerConfig`加载profiling配置时解析了activities列表(NPU/CPU)，但没有将其暴露给下游parser。`_profiling_parser.py`构建`param_dict`时缺少`ACTIVITIES`键，导致memory parser等下游组件无法区分NPU-only和CPU-only profiling场景，触发解析错误。
- Fix: `ProfilerConfig`新增`self._activities`成员和`activities`属性；`load_info`中从JSON读取activities存入；`_profiling_parser`将`ProfilerConfig().activities`注入`param_dict[Constant.ACTIVITIES]`。
- Reviewability: high -- activities是关键配置，param_dict缺少该键可用静态检查发现
- Review rule: 配置对象新增字段后，必须检查所有下游consumer是否已获得该字段的传递路径

### D-1068: CPU-only Profiling场景无条件访问CANN文件触发权限校验失败

- Root cause: 条件守卫缺失(CPU-only路径误入NPU逻辑)
- Hashes: 090d1a0bf [+4 cherry-picks: 8f7309944, 0110981e8, d378e3a85, dfde70a41]
- Files: torch_npu/profiler/analysis/_constant.py, torch_npu/profiler/analysis/_profiler_config.py, torch_npu/profiler/analysis/_profiling_parser.py, torch_npu/profiler/analysis/prof_view/_memory_view_parser.py
- Defect: Profiler在CPU-only模式下仍然无条件调用`CANNFileParser`和访问CANN文件列表。CPU-only场景没有CANN profiling产物，文件不存在时触发权限校验报错。根因是缺少对activities类型的条件判断。
- Fix: 新增`NPU_ACTIVITIES`常量；`load_timediff_info`中仅当activities含NPU时才调`CANNFileParser`；`_memory_view_parser`中三处CANN文件查询都加`if NPU_ACTIVITIES in activities else set()`守卫；`analyse_profiling_data`加`@no_exception_func()`防御。
- Reviewability: high -- CPU-only是显然的使用场景，NPU相关调用应有条件守卫
- Review rule: 所有NPU/CANN相关文件访问必须以activities配置为前提条件，不能假设NPU总是存在

### D-1069: ProcessGroupHCCL中stream id在错误时机获取(lambda外 vs 内)

- Root cause: 闭包捕获时机错误(eager evaluation vs lazy evaluation)
- Hashes: 071f0c7f7
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: 所有通信算子(allreduce, broadcast, reduce等约16处)的实现中，`getStreamId()`在lambda外部(调用方上下文)提前获取stream id并通过捕获列表传入lambda。但此时stream可能尚未绑定到正确的NPU stream，导致记录的stream id与实际HCCL调用使用的stream不一致。
- Fix: 删除lambda外部的`getStreamId()`调用，改为在lambda内部直接调用`stream.id()`，在HCCL操作即将执行的时刻获取真实的stream id。全部16处改为一致的模式。
- Reviewability: medium -- 需理解lambda捕获语义和stream绑定时序
- Review rule: lambda中使用的运行时状态(stream id, device id等)应在lambda内部获取，不应在外部eagerly evaluate后通过捕获传入

### D-1070: transfer_to_npu对torch.Generator的device参数未做cuda→npu映射

- Root cause: 设备映射遗漏(白名单方式的覆盖缺口)
- Hashes: 6a81c58d9 [+1 cherry-pick: 87978c523]
- Files: torch_npu/contrib/transfer_to_npu.py, test/contrib/test_transfer_to_npu.py
- Defect: `transfer_to_npu`机制通过白名单将torch的CUDA API替换为NPU等价。`torch.Generator`被加入白名单直接替换，但Generator构造时的device参数(如`torch.Generator('cuda')`)未经过`_replace_cuda_to_npu_in_list`映射，导致仍尝试在CUDA设备上创建Generator。
- Fix: 从白名单移除`Generator`，改为创建`_GeneratorProxy(torch.Generator)`子类，在`__new__`中对device参数做cuda→npu替换后再调父类构造。测试覆盖5种构造形式(无参、位置/关键字 x 字符串/device对象)。
- Reviewability: high -- 白名单替换方式的已知局限，构造函数参数映射是必检项
- Review rule: transfer_to_npu的白名单替换只替换函数引用不替换参数；需要参数映射的API必须用Proxy/Wrapper方式处理

### D-1071: 动态Profiling的step id缺少offset导致编号从0开始

- Root cause: 偏移量遗漏(相对值误作绝对值)
- Hashes: 0d0002080 [+1 cherry-pick: 90fb9f981]
- Files: torch_npu/profiler/profiler.py
- Defect: 动态profiling从任意step开始(非第0步)时，`profile.step()`中`record_function`的标签使用`self.step_num`(相对计数，从0开始)，但应该加上`self._step_num_offset`(起始偏移)才是真实的全局step id。结果：profiler trace中step编号从0开始，与用户实际训练步数不对应，造成trace分析困惑。
- Fix: 单行修复：`str(self.step_num)` -> `str(self.step_num + self._step_num_offset)`。
- Reviewability: high -- 变量名本身就暗示需要加offset
- Review rule: 涉及"动态起始"的功能，所有对外可见的编号必须是absolute值(base + offset)

### D-1072: transfer_to_npu patch了new_group但upstream改为委托_new_group_with_tag

- Root cause: upstream API内部重构导致monkey-patch失效
- Hashes: f7ccd9090 [+1 cherry-pick: 44717c42b]
- Files: torch_npu/contrib/transfer_to_npu.py
- Defect: `transfer_to_npu`通过monkey-patch将`torch.distributed.new_group`替换为HCCL wrapper。但新版PyTorch中`new_group`内部委托给`distributed_c10d._new_group_with_tag`执行实际逻辑，patch外层`new_group`无法拦截通过内部路径的调用(如PyTorch自身代码直接调`_new_group_with_tag`)。
- Fix: patch目标从`torch.distributed.new_group`改为`torch.distributed.distributed_c10d._new_group_with_tag`，拦截真正的执行入口。
- Reviewability: low -- 需要跟踪upstream PyTorch的内部重构
- Review rule: monkey-patch应patch最内层实际执行函数而非公开API wrapper；upstream升级后必须验证patch目标函数仍是实际执行路径

### D-1073: ACL_ERROR_RT_DEVICE_TASK_ABORT错误码缺少专用分支导致错误信息模糊

- Root cause: 错误处理分支遗漏(error code未穷举)
- Hashes: 55c1c51d4 [+1 cherry-pick: 6bbef6c02]
- Files: torch_npu/csrc/core/npu/NPUException.h
- Defect: NPU设备任务abort(如N秒快速恢复场景)时返回`ACL_ERROR_RT_DEVICE_TASK_ABORT`，但错误处理宏中没有该错误码的专用分支，走入通用`else`分支，输出模糊的错误信息。用户无法区分是任务abort(可重试)还是其他致命错误。
- Fix: 新增`else if (error_code == ACL_ERROR_RT_DEVICE_TASK_ABORT)`分支，当`device_error_msg`为空时显示`" FORCE STOP"`，否则显示实际设备错误消息。
- Reviewability: high -- 新增错误码时应同步更新所有错误处理分支
- Review rule: 引入新的ACL error code时，必须检查所有error handling宏/函数是否有对应处理分支

### D-1074: 动态Profiling step id修复 + mstx range标记支持

- Root cause: 偏移量遗漏 + 功能缺失(profiling可观测性不足)
- Hashes: 571218749 [+1 cherry-pick: 3f19c3d45]
- Files: torch_npu/profiler/dynamic_profile.py, torch_npu/profiler/profiler.py
- Defect: 两个问题：(1) 与D-1071相同的step id offset问题(此commit是更完整的修复，包含`_set_step_num_offset_for_dynamic_prof`方法)；(2) 动态profiling缺少mstx range标记，timeline视图中无法区分不同step的边界。
- Fix: (1) profiler.py新增`_step_num_offset`成员和setter方法，`start()`中的record_function使用`step_num + offset`；(2) dynamic_profile.py在`step()`中加入`mstx.range_end`/`mstx.range_start`对，profiler启动时设置offset并开启首个mstx range。
- Reviewability: medium -- offset部分high，mstx range集成需要理解profiling trace结构
- Review rule: profiler的step标记必须同时在record_function标签和mstx timeline中保持一致

### D-1075: 警告信息中包含多余的"Warning:"前缀 + "replace"拼写错误

- Root cause: 日志文本错误(冗余前缀 + typo)
- Hashes: 62e277c68 [+3 cherry-picks: 7f27f5e5f, 69cb6ef3f, dcf39b65b]
- Files: torch_npu/csrc/aten/common/ToKernelNpu.cpp
- Defect: `TORCH_WARN`宏自带"Warning"前缀，但消息字符串又以"Warning: "开头，导致输出"Warning: Warning: ..."。同时"replace"拼写为"repalce"。
- Fix: 删除字符串开头的"Warning: "，修正"repalce" -> "replace"。
- Reviewability: high -- 纯文本review即可发现
- Review rule: 日志框架自带级别前缀时，消息体不应重复该前缀；代码中的用户可见字符串应过spell check

### D-1076: Inductor store_cubin配置硬编码为True导致无法关闭

- Root cause: 配置旁路(硬编码覆盖配置值)
- Hashes: d88345a24
- Files: torch_npu/_inductor/codegen/triton.py, torch_npu/_inductor/npu_triton_heuristics.py
- Defect: `npu_triton_heuristics.py`中`launcher.store_cubin`硬编码为`True`，忽略了`config.triton.store_cubin`的用户配置。`inductor_meta`字典中也未传入该配置项。结果：即使用户配置了`store_cubin=False`，cubin仍然被存储。
- Fix: `triton.py`的`inductor_meta`中新增`"store_cubin": config.triton.store_cubin`；`npu_triton_heuristics.py`中从`self.inductor_meta.get("store_cubin", False)`读取，默认值改为False。
- Reviewability: high -- 硬编码True的代码一目了然
- Review rule: 涉及配置项的代码不应硬编码值，必须从配置源读取；code review时搜索硬编码的True/False应是常规检查项

### D-1077: DestroyUsedStreams传入NPUStream对象而非底层aclrtStream handle

- Root cause: 类型语义错误(wrapper对象 vs 底层handle)
- Hashes: 120619c6a [+3 cherry-picks: eef2a97f8, ec9b3740c, 49da77dae]
- Files: torch_npu/csrc/core/npu/NPUFunctions.cpp
- Defect: `AclrtDestroyStreamForce`期望接收`aclrtStream`(底层C handle)，但传入了`NPUStream`对象(C++ wrapper)。C++隐式转换可能让编译通过，但语义错误——wrapper析构时可能触发额外的引用计数操作或资源释放。
- Fix: 将`stream`改为`stream.stream(false)`，`false`参数表示不增加引用计数，直接获取底层handle。
- Reviewability: high -- 类型不匹配在code review中可发现
- Review rule: 调用底层C API时必须提取wrapper的原始handle，不依赖隐式转换

### D-1078: flight_recorder文档中命令示例和输出文件扩展名错误

- Root cause: 文档错误(命令和扩展名typo)
- Hashes: daa9ef1e5 [+3 cherry-picks: d5215ddd8, f7e46a0cd, 416ce2569]
- Files: tools/flight_recorder/flight_recorder.md
- Defect: 用法示例中`bash python fr_trace.py`多了`bash`前缀(应直接`python`)；输出文件扩展名写为`.txt`但实际输出格式是pickle(`.pkl`)。
- Fix: 删除多余的`bash`前缀；`.txt` -> `.pkl`。
- Reviewability: high -- 纯文档review
- Review rule: 文档中的命令示例应可直接复制执行；文件扩展名必须与实际格式一致

### D-1079: Silent Check v3三处引用错误导致重复初始化和hook注册异常

- Root cause: 变量引用错误(self vs 局部对象) + 条件判断用错变量
- Hashes: e51a48fb1 [+3 cherry-picks: 03c16ad02, 83718846e, 1999adc48]
- Files: torch_npu/asd/asd.py
- Defect: 三个问题：(1) `init_stream()`中条件判断写为`if self.statistic_value is None`，但`statistic_value`是NPU tensor(始终不为None)，应该判`self.statistic_cpu_value`(初始为None的CPU tensor)，导致每次调用都重新初始化stream。(2) hook注册循环中引用`self.hook_dict`，但实际hook字典在局部变量`matmul_check`对象上，应为`matmul_check.hook_dict`。(3) 缺少`visited_modules_id`去重，同一module的hook被重复注册。
- Fix: (1) `statistic_value` -> `statistic_cpu_value`；(2) `self.hook_dict` -> `matmul_check.hook_dict`；(3) 新增`visited_modules_id`列表按`id(module)`去重。
- Reviewability: high -- (1)(2)是明显的变量名错误，(3)是经典的去重遗漏
- Review rule: 条件判断的变量必须与其控制的初始化目标语义一致；引用对象的属性前确认self是否是正确的owner

### D-1080: Silent Check v3 tcpstore同步线程在停止后仍阻塞等待

- Root cause: 线程退出条件缺失(无限等待无退出信号)
- Hashes: b73c1d922 [+3 cherry-picks: c2d0769df, 812b86bcf, a59bf0653]
- Files: torch_npu/asd/asd.py
- Defect: `_MatmulSilentCheck`中tcpstore同步的两处while循环(`counter`和`counter2`)只检查`int(store.get('counter')) < world_size`，没有检查线程是否已被要求停止(`checksum_state_thread_running`)。当silent check被中途终止时，线程永远卡在等待所有rank到齐的循环中，无法退出。
- Fix: 两处while条件增加`and self.checksum_state_thread_running`，线程停止时循环立即退出。
- Reviewability: high -- while循环缺少退出条件是经典review检查项
- Review rule: 线程中的所有阻塞等待循环必须包含线程终止信号检查(如running flag或event)

### D-1081: 错误日志中使用非标准宏__FILENAME__

- Root cause: 非标准宏使用(平台兼容性问题)
- Hashes: 19f58182e [+3 cherry-picks: fbf4b63f4, 667683734, 8e4f00360]
- Files: torch_npu/csrc/core/npu/NPUException.h
- Defect: 错误处理宏中使用`__FILENAME__`获取源文件名，但`__FILENAME__`不是C/C++标准预定义宏(标准的是`__FILE__`)。在未定义`__FILENAME__`的编译环境下，编译报错或展开为空。
- Fix: `__FILENAME__` -> `__FILE__`(标准宏)。
- Reviewability: high -- 非标准宏使用在编译时即可发现
- Review rule: 日志/错误处理中只使用标准预定义宏(__FILE__, __LINE__, __func__)

### D-1082: mstx domain handle空指针未检查 + 空列表误判为已设置

- Root cause: 空值守卫缺失 + truthy判断错误
- Hashes: 1ca68f961 [+4 cherry-picks: 97c317d27, 39ce02bd1, d5523d1ec, 3649f1508]
- Files: torch_npu/csrc/profiler/mstx_mgr.cpp, torch_npu/csrc/profiler/mstx_mgr.h, torch_npu/csrc/profiler/npu_profiler.h, torch_npu/profiler/experimental_config.py
- Defect: 三个问题：(1) `createProfDomain`对`DOMAIN_DEFAULT`(默认domain)没有特判，创建了不必要的domain对象；(2) `MstxRange::end`中对`domainHandle`直接调用`MstxDomainRangeEnd`，未检查nullptr(当domain创建失败或为default时handle为空)；(3) `experimental_config.py`中`_check_mstx_domain_params`用`is not None`判断`_mstx_domain_include`/`_mstx_domain_exclude`，但空列表`[]`也`is not None`为True，导致用户传入空列表被误判为"已设置"，触发"include和exclude不能同时设置"的错误。
- Fix: (1) `createProfDomain`中`DOMAIN_DEFAULT`直接返回nullptr；(2) `MstxRange::end`加`if (domainHandle != nullptr)`守卫；(3) `is not None`改为truthy check(`if self._mstx_domain_include:`)。
- Reviewability: high(空值检查) / medium(空列表语义需理解Python truthy)
- Review rule: C++中从factory函数获取的指针在使用前必须检查nullptr；Python中区分"是否设置"和"是否有值"时，`is not None`和truthy check的语义不同，选择必须匹配意图

### D-1083: 测试用例适配NPU的批量修复(设备限制+格式+依赖)

- Root cause: 测试基础设施适配(多处散落修复)
- Hashes: cef697fe5 [+4 cherry-picks: f5d701b90, cb1b0cd39, 5e8f91824, dfa212415]
- Files: test/adapt_testcases_to_npu.py(新增), test/adaptive_tests.txt(新增), test/custom_ops/test_npu_anti_quant.py, test/custom_ops/test_npu_bounding_box_encode.py, test/custom_ops/test_npu_conv3d.py, test/custom_ops/test_npu_fused_attention_score_fwd.py(删除), test/custom_ops/test_npu_ifmr.py, test/custom_ops/test_npu_stride_add.py, test/requirements.txt, test/test_serialization.py(删除)
- Defect: 多个测试用例失败，原因各异：(1) `test_npu_anti_quant`中`torch.quint4x2`在新版本不可用，需改用`ml_dtypes.int4`并手动unpack int32 -> int4；(2) 多个custom op测试缺少`@SupportedDevices`装饰器，在不支持的芯片上运行报错；(3) `test_npu_conv3d`缺少`allow_internal_format`和`jit_compile`设置；(4) `test_serialization.py`是直接复制的upstream文件，未做cuda -> npu替换；(5) `test_npu_fused_attention_score_fwd`测试的算子已废弃。
- Fix: (1) 引入`ml_dtypes`依赖，实现int32 -> int4 unpack逻辑；(2) 添加`@SupportedDevices`装饰器；(3) 添加format和compile设置；(4) 新建`adapt_testcases_to_npu.py`脚本动态替换cuda -> npu，删除静态复制的文件；(5) 删除废弃算子的测试文件。
- Reviewability: medium -- 每个子问题都不难，但散布在多个文件中需逐一检查
- Review rule: 测试用例必须声明设备兼容性(@SupportedDevices)；upstream测试文件不应静态复制，应通过脚本动态适配

### D-1084: DTensor D2H拷贝使用aten::to.dtype_layout导致dispatch路径错误

- Root cause: ATen dispatch路径错误(to vs _to_copy语义差异)
- Hashes: 01970494c [+2 cherry-picks: fc05e05bd, 317b7a92c]
- Files: torch_npu/csrc/aten/npu_native_functions.yaml, torch_npu/csrc/aten/common/ToKernelNpu.cpp, test/distributed/_tensor/test_dtensor.py
- Defect: NPU注册了`aten::to.dtype_layout`的实现，但DTensor(分布式tensor)内部调用的是`aten::_to_copy`进行D2H拷贝。由于NPU未注册`_to_copy`，DTensor的`.cpu()`走入错误的dispatch路径，拷贝结果不正确或报错。`to.dtype_layout`与`_to_copy`的语义差异：前者可能返回self(当dtype/device不变时)，后者总是创建新tensor并执行拷贝。
- Fix: yaml中将注册从`to.dtype_layout`改为`_to_copy`；C++实现重写为与upstream `_to_copy`对齐：`at::empty` + `r.copy_(self, non_blocking)`，正确处理`MemoryFormat::Preserve`和pinned memory。新增DTensor `.cpu()`测试。
- Reviewability: low -- 需理解ATen dispatch机制和DTensor内部的op调用链
- Review rule: NPU注册的ATen算子必须覆盖upstream内部实际调用的op(不只是公开API层)；DTensor/FSDP等分布式组件的内部op调用是必检项

### D-1085: CI构建脚本中冗余的DISABLE_RPC_FRAMEWORK=FALSE环境变量

- Root cause: 构建配置冗余(无效环境变量)
- Hashes: 31674538b [+4 cherry-picks: 898d7b338, c9a933b64, 96c6443ae, 9cafd5af3]
- Files: ci/build.sh
- Defect: `ci/build.sh`中`export DISABLE_RPC_FRAMEWORK=FALSE`是冗余的——该环境变量已不被构建系统读取，或其默认值已经是FALSE。存在会混淆维护者对构建配置的理解。
- Fix: 删除该行。
- Reviewability: high -- 构建脚本中的环境变量应定期审计
- Review rule: 构建脚本中的环境变量应有对应的消费方；无消费方的变量应删除

### D-1086: CachingAllocator recursive_mutex与NPU同步操作交叉死锁

- Root cause: 锁-同步交叉死锁(mutex持有期间调用阻塞的设备同步)
- Hashes: a09ce7fd1 [+4 cherry-picks: c04caff71, a6b2b2f47, f92a37f3e, c001fd854]
- Files: torch_npu/csrc/core/npu/NPUCachingAllocator.cpp
- Defect: 主线程持有`recursive_mutex`执行`npuSynchronizeDevice`等待任务队列排空。但CANN内部的TBE算子编译线程在编译过程中可能触发GC，GC回调调用`free`需要获取同一把`recursive_mutex`。结果：主线程持锁等待子线程完成 -> 子线程需要同一把锁才能完成 -> 死锁。虽然是`recursive_mutex`(同一线程可重入)，但跨线程场景下与普通mutex行为一致。
- Fix: 引入`UnlockGuard` RAII类(构造时unlock，析构时lock)，在持锁期间需要调`npuSynchronizeDevice`的所有位置用`UnlockGuard`临时释放锁。`emptyCache`改为在加锁前先同步。`release_cached_blocks`删除内部同步调用(前置条件改为调用方负责)。
- Reviewability: low -- 死锁场景需要理解CANN内部的线程模型和GC触发时机
- Review rule: 持有mutex期间禁止调用可能触发跨线程回锁的阻塞操作(如设备同步)；`recursive_mutex`不能防止跨线程死锁，仅防止同线程重入

### D-1087: HCCL CommConfig设置hcclCommName时缺少capability guard

- Root cause: CANN API兼容性守卫缺失
- Hashes: 455ba3e39 [+4 cherry-picks: f37d0b8e4, 5fed87c21, b822f9c81, 833646886]
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: `createHcclCommConfigWithOptions`中直接调用`safe_strcpy_s`设置`config.hcclCommName`，但旧版本CANN的`HcclCommConfig`结构体没有`hcclCommName`字段(或该字段语义不同)。未检查CANN是否支持`HCCL_COMM_CONFIG_COMM_NAME` capability就写入，导致在旧版CANN上段错误或行为不确定。
- Fix: 用`isHcclFeatureSupported(HcclCommConfigCapability::HCCL_COMM_CONFIG_COMM_NAME)`守卫，只在supported时设置。
- Reviewability: high -- 使用新CANN结构体字段时必须检查capability
- Review rule: 所有HCCL/CANN新API和新字段必须用capability/版本check守卫；新增字段使用处应有对应的`isHcclFeatureSupported`调用

### D-1088: torch_npu_schema.json中API路径过期(revert-redo cycle)

- Root cause: 测试golden file与代码不同步(API路径重命名未同步)
- Hashes: bfe450c61, 52ccc2d14, 8ce8d4653, 5bb7794e9, 11019beee, 79a0c17d7(初始修复) → 5b24022fe, d7237e8e8, 35ba2ffe3, 419ee1040, 3d0a6b484(revert) → c9de4dc68(最终修复)
- Files: test/torch_npu_schema.json
- Defect: `torchair.ops.NpuStreamSwitch`和`torchair.ops.npu_wait_tensor`被重命名为`torchair.scope.npu_stream_switch`和`torchair.scope.npu_wait_tensor`，同时新增`torchair.scope.super_kernel`和`torchair.scope.limit_core_num`。schema golden file未同步更新导致schema regression test失败。首次修复被revert(可能因为新增条目有误或分支冲突)，之后重新提交。
- Fix: 更新schema.json中的API路径；这轮经历了fix-revert-refix循环，说明多分支同时修改schema文件容易冲突。
- Reviewability: high -- golden file更新应与对应的API重命名在同一PR中完成
- Review rule: API路径重命名必须在同一commit中更新所有golden file(schema/signature test)；多分支维护schema golden file时需注意rebase冲突

### D-1089: Dynamic Profiler warmup=0被误判为无效参数(off-by-one)

- Root cause: 边界条件off-by-one(<=0 vs <0)
- Hashes: ad7575d9e [+4 cherry-picks: 6b9828a35, d7b33294c, d85428b5d, c4e78bb25]
- Files: torch_npu/profiler/_dynamic_profiler/_dynamic_profiler_config_context.py
- Defect: `ConfigContext.warmup()`方法中判断`self._warmup <= 0`时将warmup=0视为无效并重置为DEFAULT_WARMUP。但warmup=0是合法值，表示"不需要预热直接开始profiling"。`<= 0`应为`< 0`。
- Fix: `self._warmup <= 0`改为`self._warmup < 0`。
- Reviewability: high -- 边界值审查是review基本功
- Review rule: 数值范围校验时必须明确边界值的语义(0是否合法)；warmup/timeout/retry等参数，0通常是合法的"不启用"值

### D-1090: RendezvousStoreInfo.build()签名变更导致参数缺失

- Root cause: upstream API签名变更未适配
- Hashes: eba5a322d [+2 cherry-picks: 9c28e5124, ac1fdede2]
- Files: torch_npu/distributed/rendezvous.py
- Defect: `_ParallelTCPRendezvous.next_rendezvous()`调用`RendezvousStoreInfo.build(self.rank, store)`，但PyTorch upstream修改了`RendezvousStoreInfo.build`的签名或行为，导致调用失败(参数不匹配)。
- Fix: 改为直接构造`RendezvousStoreInfo(self.master_addr, self.master_port)`，绕过build()方法。这是一种防御性做法——直接使用构造函数比依赖可能变化的build()工厂方法更稳定。
- Reviewability: medium -- 需跟踪upstream PyTorch的API变更
- Review rule: 对PyTorch upstream工厂方法(build/create等)的调用应在upstream版本升级时检查签名变化；考虑直接使用构造函数替代工厂方法以降低upstream耦合

### D-1091: Profiler DB parser函数签名不一致 + 多处typo

- Root cause: 函数签名不匹配 + 变量名typo(批量)
- Hashes: 3fd0391ce [+4 cherry-picks: 89be513e1, ababff55d, 78df3165e, ee2381ee8]
- Files: torch_npu/profiler/analysis/prof_view/prof_db_parse/_communication_db_parser.py, _fwk_api_db_parser.py, _memory_db_parser.py
- Defect: 三个问题：(1) `CommunicationDbParser.generate_view()`传递了`self._output_path`给`generate_communication_db()`，但后者不需要该参数(output_path已在self中)，方法签名有`output_path: str`但从不使用；(2) `_fwk_api_db_parser.py`中变量名`node_lauch_apis`应为`node_launch_apis`(typo: lauch→launch)；(3) `_memory_db_parser.py`中方法名`get_pta_memort_record_list`应为`get_pta_memory_record_list`(typo: memort→memory)。
- Fix: (1) 删除`generate_communication_db`和`save_communication_db_data`的`output_path`参数；(2)(3) 修正typo。
- Reviewability: high -- typo和签名不一致都是review可捕获的
- Review rule: 函数签名变更后必须检查所有调用点；变量名拼写检查应纳入linter配置(codespell等)

### D-1092: contrib测试缺失@SupportedDevices + 精度容差不足(批量修复)

- Root cause: 测试基础设施适配(设备兼容性声明+精度容差)
- Hashes: 2646225a8 [+5 cherry-picks: e17a59ab3, 8851cad5e, 5a6e13a2f, cfcacb752, bdfb33cfa]; 9bc1fffb0 [+3 cherry-picks: 7292c6894, c886d9614, 0d4a03c07]
- Files: test/contrib/test_bbox_coder.py, test/contrib/test_deform_conv.py, test/contrib/test_linear_quant.py, test/contrib/test_multiclass_nms.py
- Defect: (1) `test_bbox_coder`中`test_npu_bbox_coder_encode_xyxy2xywh`只标注了`Ascend910A`但在910B上运行报错，需要新增910B专属测试用例(数值精度不同)；(2) `test_deform_conv`中`assertRtolEqual`使用默认精度(1e-5)但910B的DCNv2算子精度只能达到1e-3；(3) 多个`test_multiclass_nms`测试缺少`@SupportedDevices`装饰器，在不支持的芯片上运行AICPU算子报错。
- Fix: (1) 按芯片型号分拆测试用例，`Ascend910B`使用独立golden value；(2) `assertRtolEqual`增加`prec=1.e-3`参数；(3) 添加`@SupportedDevices(['Ascend910A', 'Ascend910P'])`。
- Reviewability: high -- 新增contrib测试时应同步声明设备兼容性
- Review rule: 所有NPU contrib测试必须声明@SupportedDevices；不同芯片型号的数值精度差异应使用独立测试用例覆盖

### D-1093: EventTree root_nodes过滤逻辑错误排除了Allocation事件

- Root cause: 条件逻辑错误(多余的device_type过滤)
- Hashes: 7773cfafb
- Files: torch_npu/profiler/analysis/prof_parse/_event_tree_parser.py
- Defect: `EventTree.get_root_nodes()`中对`parent is None`的事件额外检查了`device_type == _DeviceType.CPU.value`。对于`_EventType.Allocation`类型的事件，取其`extra_fields.device_type`；对其他类型则默认为CPU。结果：NPU设备上的Allocation事件(device_type=NPU)即使是root节点也被过滤掉，导致profiling分析中NPU内存分配事件丢失，内存视图不完整。
- Fix: 删除整个device_type过滤逻辑，只保留`ev.parent is None`判断。root节点不应按device_type过滤——所有无父节点的事件都是合法的root。
- Reviewability: medium -- 需要理解EventTree的语义(root_nodes应包含所有设备的顶层事件)
- Review rule: 过滤条件不应隐含设备假设(CPU-only bias)；Profiler事件树操作应对所有设备类型一视同仁

### D-1094: Dynolog单例生命周期管理(eager import失败 + 重复实例化)

- Root cause: 模块初始化时序错误 + 单例模式缺失
- Hashes: a66e54f84(单例重构), 1a552cd62(lazy init修复)
- Files: torch_npu/profiler/_dynamic_profiler/_dynamic_monitor_proxy.py, torch_npu/profiler/_dynamic_profiler/_dynamic_profiler_monitor.py, torch_npu/profiler/_dynamic_profiler/_dynamic_profiler_config_context.py
- Defect: 两阶段修复。(1) a66e54f84：`_call_dyno_monitor`和`worker_dyno_func`每次调用都`from IPCMonitor import PyDynamicMonitorProxy`并创建新实例，既浪费资源又可能导致状态不一致(多个proxy实例管理同一个dynolog进程)。(2) 1a552cd62：第一阶段引入的`PyDynamicMonitorProxySingleton`在`__init__`中eager调用`_load_proxy()`，如果IPCMonitor未安装则import失败后singleton永远持有`_proxy=None`，后续调用`get_proxy()`始终返回None无法重试。此外，`_parse_dyno_exp_cfg`中`gc_detect_threshold`来自JSON的字符串值`"None"`被`is not None`误判为有效值。
- Fix: (1) 抽取`PyDynamicMonitorProxySingleton`单例类；(2) `__init__`不再eager load，改为`get_proxy()`时lazy load，并用`_load_success`标志避免重复失败的import；`gc_detect_threshold`改用`isinstance(gc_detect_threshold, str) and gc_detect_threshold != "None"`；dynolog配置中的`PROFILE_ANALYSE`从`False`(bool)改为`'false'`(str)并通过BOOL_MAP转换。
- Reviewability: medium -- 单例初始化时序问题需要理解import链
- Review rule: Singleton的__init__中不应执行可能失败的外部依赖加载(改为lazy init)；JSON反序列化后的值类型不确定(可能是str/int/bool/None)，必须显式类型检查后再使用

### D-1095: CannPackageManager模块级eager eval改为lazy init

- Root cause: 模块级eager evaluation导致初始化时序依赖
- Hashes: 0dfd68f62
- Files: torch_npu/profiler/analysis/prof_common_func/_cann_package_manager.py, torch_npu/profiler/analysis/_profiling_parser.py, torch_npu/profiler/profiler_interface.py
- Defect: `CannPackageManager`类定义时直接执行`SUPPORT_EXPORT_DB = check_cann_package_support_export_db()`，在模块import阶段就检查CANN包版本。如果import发生在CANN包尚未准备好(如延迟安装、环境变量未设置)的时机，检查结果为False且不会重试。此外，`_ProfInterface.__init__`中提前检查`CannPackageManager.SUPPORT_EXPORT_DB`抛异常，但此时profiler尚未开始，用户无法区分是"CANN真的不支持"还是"检查时机太早"。
- Fix: `SUPPORT_EXPORT_DB`改为类属性初始值`None`，新增`@classmethod is_support_export_db()`方法首次调用时才求值并缓存；删除`profiler_interface.py`中的提前检查(移至实际export时检查)。
- Reviewability: medium -- 模块级副作用是常见但容易忽视的问题
- Review rule: 类属性不应在定义时执行外部检查(环境/包/设备)——改用lazy classmethod；提前失败(fail-fast)只应在用户主动触发操作时，不应在import时

### D-1096: GE异步错误场景下OOM snapshot未触发(error message传递断裂)

- Root cause: 错误信息传递链断裂(异步queue → snapshot observer)
- Hashes: d7ef5286c
- Files: torch_npu/csrc/core/npu/NPUException.cpp, torch_npu/csrc/core/npu/NPUQueue.cpp, torch_npu/csrc/core/npu/NPUQueue.h, torch_npu/csrc/core/npu/NPUStream.cpp, torch_npu/csrc/core/npu/NPUStream.h
- Defect: GE(Graph Engine)异步执行报错时，错误信息通过`c10_npu::acl::AclGetErrMsg()`获取但只返回给调用方，未传递到NPUQueue/Repository层。`Repository::MakeSureQueueEmpty`在`ERROR_EXIT`状态下直接抛异常，没有检查错误消息是否包含OOM信息来触发snapshot dump。结果：同步执行的OOM能触发snapshot，但异步(通过task queue)的GE OOM不会。
- Fix: (1) `c10_npu_get_error_message()`获取errmsg后调`c10_npu::setRepoErrMsg(errmsg)`广播给所有stream的repo；(2) `Repository`新增`SetQueueErrMsg`/`GetQueueErrMsg`接口缓存错误信息；(3) `MakeSureQueueEmpty`和`Enqueue`的ERROR_EXIT分支中检查`IsOomSnapshotEnable()`并`strstr(errmsg, "Failed to allocate memory")`触发`oom_observer()`。
- Reviewability: low -- 需理解NPU异步执行的error propagation路径(AclGetErrMsg → Repository → Exception)
- Review rule: 异步执行路径的error信息必须传播到所有需要决策的消费方(如OOM observer)；error message的传递链应有明确的设计文档

### D-1097: OOM observer在for循环中std::move导致仅device 0生效

- Root cause: std::move在循环体中误用(移后使用)
- Hashes: 5621a5b76 [+4 cherry-picks: c52c1b6d6, a0542467a, 6e7b49be1, 31ee13bd5]
- Files: torch_npu/csrc/core/npu/NPUCachingAllocator.cpp
- Defect: `NpuCachingAllocator::attachOutOfMemoryObserver`遍历所有`device_allocator`，对每个调用`allocator->attachOutOfMemoryObserver(std::move(observer))`。第一次`std::move`后observer被移空(moved-from state)，后续设备的allocator收到的是空的observer。结果：仅device 0能触发OOM snapshot dump，其他设备OOM时observer为空不会触发。
- Fix: `std::move(observer)`改为`observer`(按值传递拷贝)。同时`DeviceCachingAllocator::attachOutOfMemoryObserver`也做同样修改。
- Reviewability: high -- `std::move`在循环体中是经典的C++ code review检查项
- Review rule: 循环体内禁止对循环外变量使用std::move(除非是最后一次迭代)；std::move后的变量只能赋值或销毁，不能再读取

### D-1098: slow_test_blocklist条目带.py后缀导致匹配失败

- Root cause: 字符串格式不一致(列表条目格式约定违反)
- Hashes: 4744d0fbb [+4 cherry-picks: 78dfb70cb, 111aad72b, ab85b4f9c, 4c9d9ff4b]
- Files: ci/access_control/constants.py
- Defect: `SLOW_TEST_BLOCKLIST`列表中`'test_jit_fuser_te.py'`带了`.py`后缀，但列表中其他条目(如`'test_reductions'`、`'test_ops_jit'`)都不带后缀。CI的blocklist匹配逻辑使用不带后缀的模块名进行比较，导致带后缀的条目永远匹配不上，该慢测试继续运行。
- Fix: `'test_jit_fuser_te.py'`改为`'test_jit_fuser_te'`。
- Reviewability: high -- 列表格式一致性是review可轻松发现的
- Review rule: blocklist/allowlist中条目格式必须统一；添加新条目时应参照现有条目的格式

### D-1099: dropout测试p=1导致全零输出无法验证正确性

- Root cause: 测试参数设置导致测试无效(degenerate test case)
- Hashes: a4f98979a [+4 cherry-picks: 424c68a00, 9cf9857ad, ebaff0054, 060bf8e9c]
- Files: test/contrib/test_ensemble_dropout.py
- Defect: `NpuFairseqDropout(p=1)`和`NpuCachedDropout(p=1)`创建了100%丢弃率的dropout，forward输出全零。测试只检查"不报错"但不验证输出值，使得dropout ensemble的核心逻辑(多个dropout共享mask/保持确定性)完全未被测试覆盖。如果dropout ensemble实现完全错误(如直接返回零tensor)，测试仍会通过。
- Fix: `p=1`改为`p=0.5`，使输出非零从而能验证dropout的行为。
- Reviewability: high -- p=1的dropout在测试中没有意义，review时应质疑
- Review rule: 测试参数不应导致退化情况(全零/全一/空输入)，除非明确测试该退化行为本身

### D-1100: codegen device_check用op_name而非func.name导致NOCHECK_SET匹配失败

- Root cause: 标识符格式不匹配(op_name vs func.name字符串表示)
- Hashes: a330367dd
- Files: codegen/utils.py
- Defect: `DEVICE_NOCHECK_SET`中存储的是`str(f.func.name)`格式的算子名(如`"aten::xxx.overload"`)，但检查代码中用`op_name`(不同格式，可能是`NativeFunction.func.name.name`即不含namespace和overload)进行`not in DEVICE_NOCHECK_SET`判断。格式不匹配导致所有NOCHECK_SET中的条目都无法命中，本应跳过device check的算子仍执行了check。
- Fix: `op_name not in DEVICE_NOCHECK_SET`改为`str(f.func.name) not in DEVICE_NOCHECK_SET`。
- Reviewability: high -- 集合查找的key格式必须与集合内元素格式一致
- Review rule: 集合/字典查找时，key的字符串格式必须与存入时一致；codegen中op标识符有多种表示(func.name, func.name.name, op_name)，使用时必须确认用哪种

### D-1101: README autoloading最低版本描述错误(2.6.0应为2.5.1)

- Root cause: 文档事实性错误(版本号)
- Hashes: e1e392b68 [+2 cherry-picks: e870bd466, 078cda284]
- Files: README.md, README.zh.md
- Defect: README中描述`import torch_npu`自动加载特性的最低版本写为"2.6.0及以后"，但实际该功能从2.5.1就已支持。误导用户认为需要升级到2.6.0才能使用autoloading。
- Fix: "2.6.0"改为"2.5.1"。
- Reviewability: high -- 版本号事实应在feature合入时同步更新文档
- Review rule: 新feature的最低版本声明应在feature实际合入的分支版本上标注，不是在计划版本上标注

### D-1102: 分布式UT子进程提前退出导致HCCL通信超时(缺p2c同步)

- Root cause: 多进程测试同步缺失(子进程生命周期管理)
- Hashes: 8ab42acd9 [+4 cherry-picks: 1c90b2860, 50641c1d1, ba493ea1c, cbfac3079]; 50653e103 [+3 cherry-picks: 576845946, acd0eaeca, 8de4d5c0e]
- Files: test/distributed/test_allgather_base.py, test/distributed/test_allgather_into_tensor.py, test/distributed/test_reduce_scatter_tensor.py, test/distributed/test_all_to_all.py
- Defect: 分布式UT使用multiprocessing.Queue(c2p)让子进程报告结果。子进程在`c2p.put()`后立即退出，但主进程还在从queue读取和验证结果。子进程退出会销毁HCCL comm，如果此时其他子进程还在等待barrier或集合操作，就会因为peer退出而HCCL超时。根本原因：子进程没有等主进程确认完成就退出了。
- Fix: 增加`p2c`反向队列：子进程`c2p.put()`报告结果后调`p2c.get()`阻塞等待；主进程验证完所有结果后`p2c.put(0)`释放子进程。保证所有子进程在主进程验证完成后才退出。
- Reviewability: medium -- 需理解多进程测试中HCCL comm的生命周期
- Review rule: 分布式多进程UT中，子进程退出前必须等待主进程确认(通过反向队列或barrier)；直接退出会导致HCCL peer失联

### D-1103: NpuExtension缺少acl/hccl第三方头文件include路径

- Root cause: 构建配置缺失(include路径不完整)
- Hashes: fd424421d [+3 cherry-picks: cf5fee609, bda57f140, f361e1cfd]
- Files: torch_npu/utils/cpp_extension.py
- Defect: `NpuExtension()`构建用户C++扩展时，include_dirs只包含`torch_npu/include`，但acl和hccl的头文件在`torch_npu/include/third_party/acl/inc`和`torch_npu/include/third_party/hccl/inc`子目录下。用户的扩展代码`#include "acl/acl.h"`或`#include "hccl/hccl.h"`编译失败。
- Fix: 在include_dirs中追加两个第三方头文件路径。
- Reviewability: high -- 第三方依赖的include路径应在打包时验证
- Review rule: cpp_extension的include_dirs应覆盖所有对外暴露的头文件路径；新增对外头文件时需同步更新NpuExtension

### D-1104: Profiler dynolog C++有符号/无符号比较警告(clean code)

- Root cause: 类型不匹配(signed/unsigned comparison)
- Hashes: 956a7c9e3 [+3 cherry-picks: ed1b26851, 7a840f897, fa9bda590]
- Files: torch_npu/csrc/profiler/dyno/NpuIpcClient.cpp, torch_npu/csrc/profiler/dyno/NpuIpcEndPoint.h
- Defect: (1) `IpcClientNpuConfig`中`int size = pids_.size()`，`size_t`赋值给`int`隐式窄化；(2) `TryRcvMessage`中`size_t retCode = recvmsg(...)`，`recvmsg`返回`ssize_t`(有符号)赋给`size_t`(无符号)，错误返回-1被转为大正数导致`retCode > 0`为true，错误报文被当作成功；(3) `for (int i = 0; i < npuPayLoad.size(); i++)`有符号/无符号比较。
- Fix: 统一使用`auto`推导正确类型。
- Reviewability: high -- 编译器warning即可发现
- Review rule: 启用`-Wsign-compare`和`-Wconversion`；`recvmsg`返回值必须用`ssize_t`接收并检查`< 0`

### D-1105: codegen variabletype对out-of-place算子的redispatch后缀错误

- Root cause: codegen模板替换逻辑不完整(out函数后缀)
- Hashes: 9ad992184 [+3 cherry-picks: 997222363, 83d0826b7, f78274bc0]
- Files: codegen/autograd/gen_variable_type.py
- Defect: NPU autograd codegen中，对`NPU_AUTOGRAD_FUNCTION`列表中的算子将`at::redispatch`替换为`at_npu::redispatch`。但对于`_out`/`_outf`类的out-of-place函数，upstream生成的代码使用`at::redispatch::xxx_outf`而NPU侧对应的是`at_npu::redispatch::xxx_out`(后缀不同)。简单的字符串替换`at::redispatch` → `at_npu::redispatch`无法处理这个后缀差异，导致生成的代码调用了不存在的`at_npu::redispatch::xxx_outf`函数，编译失败。
- Fix: 增加`f.func.is_out_fn()`分支：out函数用正则`re.sub(r'at::redispatch::(\w+)_outf', r'at_npu::redispatch::\1_out', ...)`；非out函数保持原有简单替换。
- Reviewability: medium -- 需理解codegen中out函数的命名约定
- Review rule: codegen中的字符串替换必须区分out/non-out函数变体；新增NPU_AUTOGRAD_FUNCTION条目时需验证out变体是否正确生成

### D-1106: test_fault_mode错误信息检查只检查第二个进程(竞态)

- Root cause: 测试断言逻辑错误(竞态依赖 + 迭代次数不足)
- Hashes: 6e08e7a5f [+3 cherry-picks: 4ecc73f00, 9156b01f6, 478041129]
- Files: test/distributed/_fault_mode_cases/error_use_same_addr.py, test/distributed/test_fault_mode.py
- Defect: (1) `test_fault_mode`中只检查`index == 1`(第二个进程)的stderr是否包含"Address already in use"，但哪个进程报错取决于端口竞争的时序(可能是第一个)。(2) `error_use_same_addr.py`中`for _ in range(1000)`可能在某些环境下执行完毕后进程退出太快，来不及触发地址冲突。
- Fix: (1) 收集所有进程的stderr到一个字符串中统一检查；(2) 迭代次数从1000增加到5000保证进程存活足够久。
- Reviewability: high -- 测试不应依赖进程执行顺序
- Review rule: 多进程测试的错误检查应聚合所有进程输出而非依赖特定进程索引；需要进程保持活跃的测试应确保足够的计算量

### D-1107: CPU core binding初始化时序错误(主线程绑定到错误设备)

- Root cause: 初始化时序错误(绑核时机 + 重复绑定检测)
- Hashes: b00a97565 [+2 cherry-picks: b0daf6f81, d2b5db86f]
- Files: torch_npu/csrc/core/npu/NPUAffinityController.cpp, torch_npu/csrc/core/npu/NPUAffinityController.h, torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp
- Defect: 三个问题：(1) `NpuSysCtrl::Initialize`在初始化阶段就调`SetThreadAffinity(device_id_)`绑定主线程到当前设备，但此时acl_thread尚未初始化，实际目标设备ID可能不同(如多卡场景)；(2) `has_set_pthread_affinity()`在每次`SetThreadAffinity`时重新检查(遍历所有core的CPU_ISSET)，性能浪费；(3) 细粒度绑核(bind_conf=2)中主线程在aclThread初始化时重新绑定，但此时可能还是初始化阶段，应在backward开始时绑定。
- Fix: (1) `Initialize`中不再调`SetThreadAffinity`，改为只调`GetAffinityInfo()`记录主线程tid和当前affinity状态；(2) `has_set_pthread_affinity`改为初始化时计算一次，缓存到`static bool has_set_affinity`；(3) 主线程绑核从aclThread时机改为backwardThread时机(dispatch阶段开始时)；(4) 如果用户已通过taskset/numactl设置了affinity，跳过自动绑核。
- Reviewability: low -- 需理解多卡初始化顺序和线程模型
- Review rule: 绑核操作不应在framework初始化阶段执行(目标设备尚未确定)；应尊重用户已有的affinity设置

### D-1108: Copy/contiguous算子丢失memory_format参数

- Root cause: 算子参数传递缺失(memory_format未透传到clone)
- Hashes: 40f1862d6 [+1 cherry-pick: 97b80840e]
- Files: torch_npu/csrc/aten/common/TensorProperties.cpp, test/npu/test_npu.py
- Defect: `NPUNativeFunctions::contiguous`在tensor不连续时调用`self.clone()`创建连续拷贝，但`clone()`无参调用默认使用`MemoryFormat::Preserve`(保留原始内存布局)。用户请求的`memory_format`(如`contiguous_format`)未传递给`clone()`。结果：FakeTensorMode下`contiguous()`返回的tensor可能仍不是请求的memory format。
- Fix: `self.clone()`改为`self.clone(memory_format)`，将用户请求的格式传递下去。同时保持`TORCH_CHECK`只允许`MemoryFormat::Contiguous`。
- Reviewability: high -- clone调用应总是传递memory_format
- Review rule: 算子实现中调用clone()/copy_()时必须显式传递memory_format参数

### D-1109: test/requirements.txt硬编码torch/torch_npu版本号

- Root cause: 构建配置硬编码(版本号嵌入requirements文件)
- Hashes: 15b544002 [+3 cherry-picks: db9c9c41b, 078b7e4db, 68dbd155b]
- Files: test/requirements-arm.txt, test/requirements-x86.txt
- Defect: requirements文件中硬编码了`torch==2.3.1`和`torch-npu==2.3.1rc1`及对应的pip index URL。CI环境中torch和torch_npu由构建系统安装，requirements文件中的版本号与实际版本冲突导致降级或安装失败。
- Fix: 删除torch和torch_npu的版本pin行，只保留测试依赖(beartype, expecttest, hypothesis等)。
- Reviewability: high -- requirements文件中不应pin framework自身的版本
- Review rule: 测试requirements文件只应包含第三方测试依赖，不应包含被测framework本身

### D-1110: Profiler fwdbwd flow误将同tid的start/end生成flow事件

- Root cause: Profiler数据过滤逻辑缺失(fwd/bwd flow自环)
- Hashes: 64443a306 [+2 cherry-picks: 67a155419, 708122295]
- Files: torch_npu/profiler/analysis/prof_common_func/_trace_event_manager.py
- Defect: `TraceEventManager`在生成fwdbwd flow(forward→backward关联箭头)时，没有检查start和end是否在同一个thread。当start.tid == end.tid时，flow箭头指向自身线程，在trace viewer中显示为自环(视觉噪音)，且语义上不代表真正的fwd→bwd跨线程关联。
- Fix: 在生成flow事件前增加`if node['start']['tid'] == node['end']['tid']: continue`跳过同线程的fwd/bwd对。
- Reviewability: high -- flow事件的语义是跨线程/跨进程关联
- Review rule: Profiler flow事件应验证source和target在不同tid(同tid的flow是自环，无信息量)

### D-1111: Profiler内存视图单位标注错误(显示GiB但实际用B_TO_GB常量)

- Root cause: 单位标注与计算不一致
- Hashes: cb01cbebb [+2 cherry-picks: 0c22ed405, be87dc019]
- Files: torch_npu/profiler/analysis/prof_view/_memory_timeline_parser.py
- Defect: memory timeline图表中Y轴标注为"Memory (GB)"，title中显示"Max memory allocated: X.XX GiB"。但实际除数用的是`Constant.B_TO_GB`(1e9，即GB)，显示的单位却写"GiB"(2^30)。GB和GiB有约7%的差异(1GB=10^9, 1GiB≈1.074x10^9)。
- Fix: 将title中的"GiB"改为"GB"与实际计算一致。
- Reviewability: high -- 单位不一致是review可直接发现的
- Review rule: 数值显示的单位标注必须与实际计算使用的转换常量一致；GB和GiB不可混用

### D-1112: transfer_to_npu monkey-patch torch.Tensor.to后破坏LazyModule._allowed_methods

- Root cause: monkey-patch副作用(替换方法后未更新引用方)
- Hashes: 90c676dff [+2 cherry-picks: 4612b9bf8, d297401e6]
- Files: torch_npu/contrib/transfer_to_npu.py, test/contrib/test_transfer_to_npu.py
- Defect: `transfer_to_npu`将`torch.Tensor.to`替换为包装版本(自动cuda→npu)。但`UninitializedTensorMixin._allowed_methods`列表中存储了原始`torch.Tensor.to`的引用。LazyModule在初始化前检查tensor操作是否在allowed_methods中——替换后新的`Tensor.to`与列表中存储的旧引用不同，`to()`调用被判定为不允许，抛出"not allowed on uninitialized parameter"异常。
- Fix: 在monkey-patch `Tensor.to`之后，遍历`UninitializedTensorMixin._allowed_methods`将旧的`to`引用替换为新的`torch.Tensor.to`。
- Reviewability: low -- 需要知道LazyModule的_allowed_methods机制
- Review rule: monkey-patch标准库方法后，必须搜索所有存储该方法引用的位置(如allowed_methods, dispatch tables)并同步更新

### D-1113: Dynamic Profiler多test串行运行时start_step累积偏移

- Root cause: 测试状态累积(全局step counter未考虑多test间的偏移)
- Hashes: 9ae16f183 [+5 cherry-picks: 92574d141, 4ad973c47, 0a6f87fc9, 382e0351e, 8548f379c]
- Files: test/profiler/test_dynamic_profiler.py
- Defect: Dynamic Profiler UT中多个test方法共享同一个`dp`(dynamic profiler)实例。每个test调用`dp.step()`推进step counter，但配置文件中的`start_step`硬编码为固定值。当多个test串行运行时，step counter已累加到较大值，而新test的config仍写`start_step: 1`，导致profiler认为当前step已超过start_step而不触发采集。
- Fix: 引入类变量`TestDynamicProfiler.start_step = 0`跟踪全局step偏移；每个test的config.json中`start_step`设为`TestDynamicProfiler.start_step + 1`；每次`dp.step()`后递增。同时设置`os.environ["RANK"] = "0"`确保rank过滤正确。
- Reviewability: medium -- 需理解dynamic profiler的step-based触发机制
- Review rule: 多test共享有状态单例(如profiler)时，配置参数必须相对于当前状态而非使用绝对值

### D-1114: 公共API测试golden file更新(public_bindings + schema + test参数)

- Root cause: 测试golden file与代码不同步(批量维护)
- Hashes: 17095a211 [+2 cherry-picks: fbb17a2f5, 4d34c0297]; b9c78de79 [+2 cherry-picks: 2d098799f, 8ceffb7f6]
- Files: test/npu/test_public_bindings.py, test/torch_npu_schema.json, test/npu/test_npu.py, test/distributed/_fault_mode_cases/error_use_same_addr.py
- Defect: 多处测试与代码不同步：(1) `test_public_bindings.py`的`tempFilter`中仍列出已删除的`FastBatchNorm*`/`FastSyncBatchNorm`类，导致filter无效(过滤不存在的符号不报错但浪费维护精力)；(2) `test_npu.py`中`torch.sum`在特定场景下精度问题改为`torch.mean`；(3) schema golden file中API条目过期。
- Fix: 删除不存在的tempFilter条目；`torch.sum`改为`torch.mean`；更新schema.json。
- Reviewability: high -- golden file应有自动化更新机制
- Review rule: 公共API删除/重命名时，test_public_bindings的tempFilter和schema.json必须同步更新

### D-1115: HCCL comm通过RankTableFile创建时忽略HCCL_BUFFSIZE环境变量

- Root cause: 环境变量传递遗漏(配置路径分叉)
- Hashes: c8df4c007 [+2 cherry-picks: 17bce4ff2, c2281b974]
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp, torch_npu/csrc/core/npu/register/OptionsManager.cpp, torch_npu/csrc/core/npu/register/OptionsManager.h
- Defect: 通过`RANK_TABLE_FILE`方式创建HCCL comm时，`createHCCLCommEx`中的config用`HcclCommConfigInit(&config)`初始化(默认buffer size)，未读取`HCCL_BUFFSIZE`环境变量。但通过其他方式创建的comm(如P2P)有`getP2PHcclCommConfig`会设置buffer size。用户设置`HCCL_BUFFSIZE=400`期望增大通信buffer，但RankTableFile路径下的comm仍使用默认200M。
- Fix: 新增`getHcclCommConfig()`函数：`HcclCommConfigInit` + `config->hcclBufferSize = GetHcclBufferSize()`；`GetHcclBufferSize()`从`HCCL_BUFFSIZE`环境变量读取(默认200M)。所有非P2P的comm创建路径改用`getHcclCommConfig`。
- Reviewability: medium -- 需要理解HCCL comm的创建路径分叉
- Review rule: 新增HCCL/comm配置项时，所有comm创建路径(RankTableFile/API/P2P)都需要覆盖

### D-1116: Profiler torch_op input_dtypes和input_shapes列表顺序颠倒

- Root cause: 数据字段顺序错误(位置参数)
- Hashes: 13db5471e [+3 cherry-picks: 6f1e63fd8, b2234bfd1, 1ac959e42]
- Files: torch_npu/profiler/analysis/prof_parse/_fwk_file_parser.py
- Defect: `FwkFileParser`构造torch_op的api列表时，`INPUT_SHAPES`和`INPUT_DTYPES`的位置写反了。列表是位置敏感的(下游消费方按索引取值)，shapes放在了dtypes的位置，dtypes放在了shapes的位置。结果：profiler输出的torch_op信息中，显示为"type"的实际是shape，显示为"shape"的实际是type。
- Fix: 交换两个`args.get()`的顺序，使`INPUT_DTYPES`在前，`INPUT_SHAPES`在后。
- Reviewability: high -- 位置参数列表中字段顺序应有注释标注
- Review rule: 位置敏感的列表/元组应使用具名字段(namedtuple/dataclass)替代裸列表；至少应有注释标注每个位置的含义

### D-1117: mstx patch对Module.__call__的monkey-patch在graph mode下引入额外profiler节点

- Root cause: monkey-patch层级冲突(profiler patch vs framework __call__ chain)
- Hashes: bc5ec9149
- Files: torch_npu/profiler/_add_mstx_patch.py, test/profiler/test_profiler_tree.py
- Defect: `apply_mstx_patch()`将`Module.__call__`替换为`_custom_tx_call`(添加mstx range标记)。在graph mode(torch.compile/torch.jit.trace)下，`_custom_tx_call`中的`is_initialized`检查和`_wrapped_call_impl`调用链在profiler tree中产生额外的节点层级(`torch_npu/npu/__init__.py: is_initialized`和`_wrapped_call_impl`)。这些额外节点导致profiler tree golden test失败，且在trace中产生噪音(每个module调用多两层)。
- Fix: 删除对`Module.__call__`的monkey-patch(不再替换`__call__`)。mstx range标记改用其他机制(如`_call_impl`内的hook)实现，不需要替换`__call__`入口。
- Reviewability: medium -- 需理解Module的__call__ → _wrapped_call_impl → _call_impl调用链
- Review rule: 避免monkey-patch Module.__call__(它是PyTorch module执行的核心入口，替换会影响所有module)；profiler instrumentation应使用hook机制而非替换调用链

### D-1118: NPUEventManager UnrecordedCount在STOP_EXIT时��清理导致watchdog误判

- Root cause: 状态清理遗漏(异常退出路径)
- Hashes: f561de871 [+2 cherry-picks: baa150a7b, be77f5ddb]
- Files: torch_npu/csrc/core/npu/NPUEventManager.cpp, torch_npu/csrc/core/npu/NPUEventManager.h, torch_npu/csrc/core/npu/NPUQueue.cpp
- Defect: 当task queue进入`STOP_EXIT`状态(如UCE或异步错误)时，`ClearQueue()`只清理了队列中的任务，未清理`NPUEventManager`中的`event_unrecorded_count_`。watchdog检查事件是否recorded时发现仍有unrecorded事件，误判为任务hang，触发不必要的超时告警或abort。
- Fix: 在`ReadQueue`和`Dequeue`的STOP_EXIT分支中，`ClearQueue()`后追加`NPUEventManager::GetInstance().ClearUnrecordedCount()`。
- Reviewability: medium -- 需理解event lifecycle和watchdog检查逻辑
- Review rule: 异常退出路径(STOP_EXIT/ERROR_EXIT)必须清理所有关联状态(queue + event counter + callback等)

### D-1119: Profiler DB写入死锁(子进程共享sqlite connection + str_id_map未清理)

- Root cause: 并发模型错误(子进程+sqlite check_same_thread + 共享状态)
- Hashes: 018838294 [+2 cherry-picks: 8ecbc5956, a551968c0]
- Files: torch_npu/profiler/analysis/prof_common_func/_db_manager.py, _id_manager.py, torch_npu/profiler/analysis/prof_config/_parser_deps_config.py
- Defect: 三个相互关联的问题：(1) sqlite3.connect未设`check_same_thread=False`，当parser配置为`SUB_PROCESS`模式时，子进程继承了主进程创建的connection，在子进程中使用触发"ProgrammingError: SQLite objects created in a thread can only be used in that same thread"(sqlite默认thread安全检查)；(2) `Str2IdManager`的`_str_id_map`���多轮write间未clear，导致id累积膨胀最终写入重复数据；(3) 多个parser被配置为`SUB_PROCESS`但共享同一个db文件，子进程并发写入造成"database is locked"。
- Fix: (1) `check_same_thread=False`允许跨线程使用(因为实际是跨进程继承)；(2) `_str_id_map.clear()`在返回data后清空；(3) 将`FWK_API_DB_PARSER`、`MEMORY_DB_PARSER`、`COMMUNICATION_DB_PARSER`等从`SUB_PROCESS`改为`PTHREAD`(同进程线程避免sqlite并发锁)；同时修正依赖关系(MEMORY_DB_PARSER依赖FWK_API_DB_PARSER)。
- Reviewability: low -- 需理解parser并发模型和sqlite线程安全机制
- Review rule: sqlite connection不应跨进程共享(fork后connection状态未定义)；多个writer并发写同一db应使用WAL模式或串行化

### D-1120: Profiler step_info表导出失败(parser依赖配置遗漏)

- Root cause: 任务依赖配置错误
- Hashes: 470452e7f [+6 cherry-picks: e9aa18d55, 38f844922, 7571f58bb, 70032b8e4, a51476d92, 6b7f1eca2]
- Files: torch_npu/profiler/analysis/prof_config/_parser_deps_config.py
- Defect: `STEP_INFO_DB_PARSER`的依赖列表遗漏了前置parser，导致step_info数据还未准备好就尝试写入db，空数据或不完整数据导致导出失败。
- Fix: 修正依赖配置中的遗漏项。
- Reviewability: high -- 依赖配置应有完整性检查
- Review rule: parser DAG配置变更后应验证拓扑排序正确性和数据完整性

### D-1121: foreach_optim CANN版本黑名单适配(重构)

- Root cause: 兼容性适配重构(CANN版��检测)
- Hashes: 5de781c7b; 6e069f72f [+1 cherry-pick: 495ffb8bb]
- Files: torch_npu/utils/_optim.py
- Defect: 原实现为每个optimizer(sgd, adam, adamw等)单独import源函数并用`wrap_optim_warning_func`包装为foreach=False版本，代码重复严重(11个optimizer x import+wrap)。更严重的是：包装只是打印warning但仍传递foreach参数给底层，低版本CANN上foreach算子执行报错。同时`partial_class`修改了optimizer类的`__init__`签名导致子类化出问题。
- Fix: 全面重构：用CANN版本黑名单(`_foreach_black_list_for_cann_*`)判断是否支持foreach；不支持时从`torch.optim.optimizer.Optimizer._optimizer_step_code`中patch掉foreach支持，而非逐个wrap。用`patch_supported_devices()`动态返回支持的设备列表。
- Reviewability: medium -- 重构逻辑合理，但黑名单维护成本高
- Review rule: CANN兼容性黑名单应集中管理、自动化测试覆盖；新CANN版本���布后须更新黑名单

### D-1122: conv3d fp32场景精度问题(芯片兼容性cast)

- Root cause: 芯片兼容性(算子精度差异)
- Hashes: 97ae53a25 [+1 cherry-pick: 6025cdce5]
- Files: torch_npu/utils/_module.py
- Defect: `cast_weight`函数在AMP场景下对Conv3d的weight做dtype cast，但特定SoC(如Ascend910B/Ascend910_93)上conv3d支持fp32直接计算且精度更好，不需要cast。在不支持fp32的SoC上不cast又会出精度问题。原代码对所有SoC统一处理。
- Fix: 新增`CONV3D_SUPPORT_FP32_SOC_PREFIX`白名单，在`cast_weight`中检查当前设备是否在白名单中，只对不在白名单的SoC做weight cast。
- Reviewability: medium -- 需要知道哪些SoC支持conv3d fp32
- Review rule: 算子精度行为因SoC而异时，必须有明确的SoC白名单/黑名单；名单应与硬件能力矩阵文档保持同步

### D-1123: 交互式环境下task queue默认启用导致未知错误

- Root cause: 运行环境检测缺失(交互式 vs 脚本模式)
- Hashes: 0672cf7c2 [+1 cherry-pick: a62d352dd]
- Files: torch_npu/__init__.py
- Defect: NPU的task queue(异步任务队列)在交互式Python(REPL/Jupyter)下可能导致错误(如异步异常无法正确传播到交互式prompt)。import torch_npu时未检测是否在交互式环境中。
- Fix: 在`__init__.py`末尾检查`sys.ps1`(Python交互式标志)，如果存在则设`TASK_QUEUE_ENABLE=0`并warnings.warn提示用户。
- Reviewability: high -- 简单的环境检测
- Review rule: 影响异步行为的特性在交互式环境中应有安全默认值

### D-1124: ranktable模式下global processgroup假设uid==0但实际创建顺序不保证

- Root cause: 初始化顺序假设错误(全局PG识别)
- Hashes: 35973013f [+1 cherry-pick: d7178a68e]
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp, torch_npu/csrc/distributed/ProcessGroupHCCL.hpp
- Defect: `ProcessGroupHCCL`用`static atomic<size_t> process_group_id`自增分配uid，然后假设`uid_ == 0`的就是global PG。但在某些场景下(如用户先创建子group再创建全局group)，全局PG的uid不是0。结果：global PG指针`global_`指向了错误的PG(第一个创建的子group)，或global PG被误判为非全局。
- Fix: 删除uid机制。改用`options_->global_ranks_in_group.empty()`判断是否为全局PG(空的global_ranks_in_group表示包含所有rank)。
- Reviewability: medium -- 需理解PG创建的可能顺序
- Review rule: 不应假设特定对象的创建顺序(如uid==0)；使用语义属性(如rank列表)判断对象角色

### D-1125: Profiler DB timeout值过小导致"database is locked"

- Root cause: 配置常量不合理(timeout太短)
- Hashes: a9db2d23d [+1 cherry-pick: 2a97e8dd8]
- Files: torch_npu/profiler/analysis/prof_common_func/_db_manager.py
- Defect: sqlite3.connect的timeout设置不合理(具体值未从diff直接看到，但commit message指出是"small timeout value")。大规模profiling数据写入时多个parser并发访问db，短timeout导致频繁的"database is locked"错误。
- Fix: 增大timeout值(从较小值改为更合理的值)。
- Reviewability: high -- timeout常量应基于实际负载测试确定
- Review rule: 并发数据库操作的timeout应基于最大预期数据量测试确定，不应使用默认值

### D-1126: UCE场景下ReadQueue未设置uce状态导致dequeue线程退出

- Root cause: 错误处理状态不一致(UCE检测与queue状态)
- Hashes: 6a6c7d125 [+1 cherry-pick: 1fddac5f5]
- Files: torch_npu/csrc/core/npu/NPUQueue.cpp, torch_npu/csrc/core/npu/NPUQueue.h
- Defect: `ReadQueue`在检测到UCE(Uncorrectable Error)时未正确设置queue的uce状态标志。dequeue线程检查uce标志决定是否继续运行——标志未设置导致dequeue线程在UCE后仍尝试读取已损坏的数据，可能触发二次错误或段错误。
- Fix: `ReadQueue`在UCE检测后设置uce状态；调整状态管理的读写接口使UCE标志在所有相关路径中一致。
- Reviewability: low -- 需理解UCE错误处理的完整状态机
- Review rule: 硬件错误(UCE/ECC)检测后必须在所有consumer路径中传播停止信号

### D-1127: Profiler空JSON文件输出(memory timeline无数据时)

- Root cause: 空数据守卫缺失
- Hashes: 574d22afe [+2 cherry-picks: 858398166, d97301efe]
- Files: torch_npu/profiler/analysis/prof_view/_memory_timeline_parser.py
- Defect: memory timeline parser在没有内存数据时仍生成JSON文件，输出空JSON或只有空结构的JSON。下游工具读取空JSON解析失败。
- Fix: 在生成JSON前检查数据是否为空，为空则跳过文件生成。
- Reviewability: high -- 输出前应检查数据有效性
- Review rule: Profiler所有输出文件在写入前必须检查数据非空

### D-1128: mstx.range_start不支持无stream调用(缺host-only重载)

- Root cause: API缺少重载路径(必选参数应为可选)
- Hashes: c792bdf90 [+2 cherry-picks: 2cf3a07a4, be7ba196a]
- Files: torch_npu/npu/mstx.py
- Defect: `mstx.range_start(message, stream)`要求必须传入stream参数。但用户在host-only场景(不涉及特定stream)下调用时没有合适的stream可传，传None导致`isinstance(stream, Stream)`失败打印warning并返回0。同时warning使用了`print(Warning, ...)`而非`warnings.warn()`，不符合Python warning规范。
- Fix: stream参数改为可选(默认None)。stream为None时调用`_range_start_on_host(message)`(host-only版本)；stream为Stream时提取npu_stream调用`_range_start(message, stream)`；类型错误时用`warnings.warn()`。
- Reviewability: high -- API设计应允许最简调用方式
- Review rule: Profiler API的stream参数应为可选(host场景普遍不需要)；Python中不要`print(Warning, ...)`，用`warnings.warn()`

### D-1129: Copy算子对复数共轭tensor创建冗余中间tensor

- Root cause: 算子实现冗余(不必要的中间allocation)
- Hashes: 3a34bb7ec [+6 cherry-picks: 5348ba6fc, 8cc4bc183, fe6d0e58a, 2836adde0, 67e3c8a9e, 42bb49b2c]
- Files: torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp
- Defect: `copy_`算子处理复数共轭tensor时，先创建临时`result` tensor(通过`apply_tensor_without_format`)，用`aclnnComplex`将实部虚部组合到result中，再copy到目标self。中间result是完全不必要的——可以先执行copy(将src原样拷贝到self)，然后在self上原地处理共轭(取实部和取反后的虚部)。额外的中间tensor增加了显存峰值(等于输入大小)。
- Fix: 重构为"先copy、后原地处理"：直接将src copy到self(D2D或H2D)，然后对is_conj的tensor在self上执行`aclnnComplex(real(self), imag(self).neg(), self)`。删除中间result的分配。
- Reviewability: medium -- 需理解复数tensor的内存布局
- Review rule: Copy/to/clone等基础算子应避免中间allocation；复数运算路径需单独review显存开销

### D-1130: torch_npu._C模块级import循环导致AttributeError

- Root cause: 模块import循环(循环引用触发属性未初始化)
- Hashes: 6479f2196 [+2 cherry-picks: d08ba2bf5, a693e9669]
- Files: torch_npu/utils/__init__.py
- Defect: `torch_npu/utils/__init__.py`中`import torch_npu`然后`torch_npu._C._flops_count_init()`。在某些import顺序下(如子模块先于`_C`加载)，`torch_npu`模块对象存在但`_C`属性尚未绑定，触发`AttributeError: module 'torch_npu' has no attribute '_C'`。
- Fix: 改为`from torch_npu import _C`直接import `_C`子模块，避免通过`torch_npu._C`间接访问。直接import确保`_C`在使用前已加载。
- Reviewability: high -- import循环是常见Python陷阱
- Review rule: 子模块不应通过顶层包名间接访问兄弟模块(如`torch_npu._C`)；使用`from torch_npu import _C`直接import

### D-1131: Profiler task_manager InterruptedError未处理

- Root cause: 信号处理缺失(InterruptedError导致profiler线程崩溃)
- Hashes: 1e7a9fb79 [+4 cherry-picks: 5ab9d35cc, 3710b6b96, a80d944c4, 10864804e]
- Files: torch_npu/profiler/analysis/prof_common_func/_constant.py, _task_manager.py, torch_npu/profiler/analysis/prof_view/cann_parse/_cann_export.py
- Defect: 三个问���：(1) `_cann_export.py`中`time.sleep(0.1)`被信号中断时抛`InterruptedError`，未捕获导致CANN export线程崩溃(profiler数据丢失)；(2) `ConcurrentTasksManager`的epoll循环中，所有NON_BLOCKING task完成后立即返回True，但此时可能有PTHREAD task仍在运行，导致结果不完整；(3) sleep时间硬编码为0.1秒，在大规模profiling场景下polling过于频繁。
- Fix: (1) `time.sleep`加`try: ... except InterruptedError: return FAIL`；(2) NON_BLOCKING全完成后sleep等待一段时间再确认(double-check)；(3) 集中定义`SLEEP_TIME=0.5`常量。
- Reviewability: medium -- InterruptedError是Python信号处理的常见陷阱
- Review rule: 所有`time.sleep()`调用应考虑InterruptedError(尤其在长时间运行的profiler/monitor线程中)

### D-1132: Profiler delete_prof_dir删除了上级目录

- Root cause: 路径操作错误(dirname vs 原路径)
- Hashes: c32e97836 [+1 cherry-pick: 5343e0bb8]
- Files: torch_npu/profiler/_profiler_path_creator.py
- Defect: `delete_prof_dir`中`shutil.rmtree(os.path.dirname(self._prof_path))`删除了`_prof_path`的父目录。如果`_prof_path`已经是完整的profiling目录路径，`dirname`会导致删除更上一级目录(可能包含其他profiling数据或用户文件)。
- Fix: `os.path.dirname(self._prof_path)`改为`self._prof_path`。
- Reviewability: high -- rmtree的目标路径必须review
- Review rule: `shutil.rmtree`的目标路径必须精确到要删除的目录本身；使用dirname/basename前应确认路径层级

### D-1133: Profiler多线程下NPU内存数据上报错误(allocator_type未传递)

- Root cause: profiler hook数据不完整(缺少allocator_type字段)
- Hashes: 34d5234f2 [+2 cherry-picks: 188f2666a, 4709bfd39]
- Files: torch_npu/csrc/core/npu/NPUCachingAllocator.cpp
- Defect: `DeviceCachingAllocator::malloc`的profiler hook(`reportMemoryDataToNpuProfiler`)缺少`allocator_type`信息(区分default allocator vs custom allocator)。多线程场景下不同allocator的内存操作混在一起，profiler无法区分来源。此外`reportMemoryDataToNpuProfiler`的include被放在`#ifndef BUILD_LIBTORCH`之外但调用在其内，libtorch构建时链接报错。
- Fix: `malloc`新增`allocator_type`参数(默认0)并透传到profiler hook；修正include guard位置。
- Reviewability: medium
- Review rule: Profiler hook应传递足够的上下���信息(device, allocator_type, stream)区分数据来源

### D-1134: Profiler dynamic profiler错误参数处理不健壮

- Root cause: 输入校验不足(dynamic profiler配置参数)
- Hashes: a857e60f2 [+1 cherry-pick: 2bac16c78]
- Files: torch_npu/profiler/analysis/prof_common_func/constant.py, torch_npu/profiler/dynamic_profile.py, torch_npu/profiler/profiler.py
- Defect: dynamic profiler配置中传入错误参数(如非法的profiler_level、无效的output_path等)时，直接抛异常导致训练中断。应graceful处理错误参数，打印warning并使用默认值继续。
- Fix: 添加参数校验和fallback逻辑；error_msg从`print_error_msg`改为`print_warn_msg`(非致命错误不应用error级别)。
- Reviewability: high -- 配置参数校验是标准practice
- Review rule: Dynamic profiler的配置错误不应中断训练；应warn并fallback到安全默认值

### D-1135: Profiler文件资源未释放(shared memory monitor)

- Root cause: 资源泄漏(文件描述符未关闭)
- Hashes: e300f173a [+2 cherry-picks: 6b5234e59, e13523e1a]
- Files: torch_npu/profiler/_dynamic_profiler/_dynamic_profiler_monitor_shm.py
- Defect: dynamic profiler的shared memory monitor打开文件后在异常路径上未关闭文件描述符。长时间运行(如训练数千个step)后文件描述符耗尽(ulimit)，新的profiling session无法创建shared memory。
- Fix: 使用context manager(with语句)或try-finally确保文件关闭。
- Reviewability: high -- 文件操作应使用context manager
- Review rule: 所有文件操作必须使用with语句或try-finally确保关闭

### D-1136: Profiler Non-Blocking task完成检查竞态

- Root cause: 并发检查竞态(任务完成状态检查与实际完成有时间差)
- Hashes: 585a2d5a6 [+2 cherry-picks: 99ce76914, 756f80bfa]
- Files: torch_npu/profiler/analysis/prof_common_func/_task_manager.py
- Defect: `ConcurrentTasksManager`中检测到所有NON_BLOCKING task完成后直接返回True，但其他task的完成状态可能还在pipeline中尚未反映(epoll事件延迟)。导致主线程提前认为所有task完成，开始后续处理时部分task实际还在运行。
- Fix: 检测到NON_BLOCKING全部完成后，sleep一段时间再做第二次确认(double-check pattern)。
- Reviewability: medium -- 并发完成检测的classic race condition
- Review rule: 异步任务完成检测应使用明确的完成信号(如barrier/done event)而非轮询状态快照

### D-1137: Dynamic profiler start interface异常未捕获导致训练中断

- Root cause: 异常处理缺失(profiler.start异常传播到训练循环)
- Hashes: d9f3a49f7 [+4 cherry-picks: e9fd66af3, 0b42fdf13, c181dcf88, 15580dc01]
- Files: torch_npu/profiler/dynamic_profile.py
- Defect: `dynamic_profile.py`中profiler的start()调用可能因配置错误、路径不存在、权���不足等原因抛异常。异常未被捕获直接传播到训练循环，中断训练。profiler是辅助工具，其异常不应影响训练。
- Fix: start()调用加try-except，捕获异常后打印warning并跳过本次profiling。
- Reviewability: high -- profiler异常不应中断训练是基本原则
- Review rule: Profiler/monitor等辅助组件的异常必须在组件边界内捕获处理

### D-1138: Fix optim patch兼容性问题

- Root cause: monkey-patch兼容性(optimizer step signature变更)
- Hashes: 9e98679be [+2 cherry-picks: 6a77345b1, 8409a9f13]
- Files: torch_npu/utils/_optim.py
- Defect: optimizer的foreach patch与PyTorch upstream的optimizer step签名变更不兼容。具体表现为patch后的step()函数参数不匹配upstream新增的参数。
- Fix: 调整patch逻辑以适配新的step签名。
- Reviewability: medium -- 需跟踪upstream optimizer API变更
- Review rule: monkey-patch的函数签名必须与被patch函数的当前签名一致；upstream升级后需验证所有patch

### D-1139: npu_format_cast deprecated method调用方式

- Root cause: API调用方式过时
- Hashes: ff4fdb616
- Files: torch_npu/csrc/aten/common/SetNpu.cpp, test/test_npu/test_tensor.py
- Defect: 测试用例使用了已废弃的`tensor.npu_format_cast(29)`实例方法调用方式，同时SetNpu.cpp中storage desc降维为1D的条件判断注释不清晰。
- Fix: 将实例方法调用改为`torch_npu.npu_format_cast(tensor, 29)`的模块级函数调用。
- Reviewability: high
- Review rule: 搜索所有`.npu_format_cast(`实例方法调用，确保使用模块级函数替代废弃API。

### D-1140: pin_memory/is_pinned Python wrapper缺失NPU dispatch

- Root cause: 设备dispatch缺失
- Hashes: 926d82e7a [+1 cherry-pick: d76bf081c]
- Files: torch_npu/csrc/aten/PinMemory.cpp, torch_npu/utils/tensor_methods.py
- Defect: Python wrapper直接调用`torch._C._TensorBase.pin_memory(self, device=...)`无法走NPU的BackendSelect dispatch路径，pin_memory功能在NPU设备上失效。
- Fix: Revert删除C++ dispatch注册的PR，保留PinMemory.cpp中BackendSelect层的实现。
- Reviewability: medium
- Review rule: 删除设备dispatch注册前，必须验证替代方案能完整覆盖原有路径。

### D-1141: run_test.py执行方式错误导致测试不运行

- Root cause: 测试框架配置错误
- Hashes: f230c3e27
- Files: test/run_test.py
- Defect: `python3 -m unittest`模式不会执行`if __name__ == '__main__'`中的run_tests()调用，导致测试实际未运行。
- Fix: 将执行命令从`python3 -m unittest`改为直接`python3`执行测试脚本。
- Reviewability: high
- Review rule: 测试执行框架变更后需验证测试实际被执行（0 tests run也是pass）。

### D-1142: torch.jit.annotations.parse_type_line存在代码注入漏洞

- Root cause: 安全漏洞(代码注入)
- Hashes: 375917cef [+1 cherry-pick: 59bce6b03]
- Files: scripts/codegen/templates/torch_funcs.py, test/test_jit.py
- Defect: `parse_type_line`使用`eval()`执行用户输入的字符串，攻击者可通过构造恶意类型注解注入任意代码。
- Fix: 新增`_eval_no_call`函数，编译字节码后检查是否包含CALL指令，拒绝包含函数调用的表达式。
- Reviewability: high
- Review rule: 任何对用户输入使用eval()的地方必须有字节码级别的安全检查。

### D-1143: index_put算子shape广播检查缺失与AICPU路径错误

- Root cause: 算子shape校验缺失
- Hashes: f33097e03
- Files: torch_npu/csrc/aten/ops/IndexPutKernelNpu.cpp
- Defect: 缺少value tensor与index输出shape的广播兼容性检查；AICPU路径使用旧的"IndexPut"而非"IndexPutV2"。
- Fix: 新增check_size函数做广播校验；统一切换到IndexPutV2算子。
- Reviewability: medium
- Review rule: 算子的AICORE/AICPU两条路径必须使用同一版本的底层算子。

### D-1144: fused optimizer属性名不唯一导致误判

- Root cause: 属性命名冲突
- Hashes: 86464c7db
- Files: torch_npu/npu/amp/grad_scaler.py, torch_npu/optim/npu_fused_optim_base.py
- Defect: `hasattr(optimizer, 'is_fused_optimizer')`仅检查属性存在，不检查值，其他optimizer若有同名属性会误进入fused路径。
- Fix: 属性名改为`is_npu_fused_optimizer`，判断条件改为同时检查值。
- Reviewability: high
- Review rule: 类型判断不能仅用hasattr，应同时检查属性值。

### D-1145: transfer_to_npu中device index为0时被误判为False

- Root cause: Python falsy值判断错误
- Hashes: 51a72ed0e
- Files: torch_npu/contrib/transfer_to_npu.py
- Defect: `if arg.index`当index为0时返回False，导致device 0无法正确选择。
- Fix: 改为`if arg.index is not None`。
- Reviewability: high
- Review rule: 对可能为0的数值型属性，永远使用`is not None`而非truthiness判断。

### D-1146: replay graph在不需要梯度时仍强制初始化grad

- Root cause: 条件判断缺失
- Hashes: c2bb75554
- Files: torch_npu/npu/replay_graph.py
- Defect: WrapModule无条件对所有参数执行`p.grad = torch.zeros_like(p)`，浪费内存。
- Fix: 添加`if self.requires_grad`条件守卫。
- Reviewability: high
- Review rule: 初始化grad前检查requires_grad标志。

### D-1147: get_module返回临时string的dangling pointer

- Root cause: C++临时对象生命周期错误
- Hashes: 430f7f1eb
- Files: torch_npu/csrc/utils/TensorType.cpp
- Defect: `("torch." + c10::get_privateuse1_backend()).c_str()`返回临时std::string的内部指针，函数返回后成为野指针。
- Fix: 返回类型从`const char*`改为`std::string`。
- Reviewability: high
- Review rule: 禁止从函数返回临时std::string的.c_str()指针。

### D-1148: NZ format不支持1维tensor导致test_c10d断言失败

- Root cause: NPU format维度约束
- Hashes: 37f725760
- Files: test/test_npu/test_c10d.py
- Defect: 测试用例创建1维tensor后转NZ format，但NZ仅支持2维及以上。
- Fix: 测试tensor shape从`[100]`改为`[100, 1]`。
- Reviewability: high
- Review rule: 使用NZ format前应校验tensor维度>=2。

### D-1149: random.py缺少Tensor导入

- Root cause: Python import遗漏
- Hashes: 6337b8b97
- Files: torch_npu/npu/random.py
- Defect: 使用了`Tensor`类型但没有import。
- Fix: 添加`from torch import Tensor`。
- Reviewability: high
- Review rule: 类型引用必须有对应的import语句。

### D-1150: scatter op缺少非out变体和inplace变体实现

- Root cause: 算子变体注册不完整
- Hashes: a5f8402e8
- Files: torch_npu/csrc/aten/ops/ScatterKernelNpu.cpp
- Defect: scatter仅实现out变体，缺少functional/inplace变体。
- Fix: 注册并实现四个缺失变体，内部复用out变体。
- Reviewability: medium
- Review rule: 注册算子时需覆盖所有常用变体(functional/inplace/out)。

### D-1151: CTC Loss算子未传递max_length信息给底层

- Root cause: 算子参数编码错误
- Hashes: 64ec395b2 [+2 cherry-picks: c962f2176, 4b10cfe76]
- Files: torch_npu/csrc/aten/ops/CtcLossKernelNpu.cpp
- Defect: max_length需编码到blank参数中，但缺少此编码步骤。
- Fix: 在计算max_length后编码到blank参数。
- Reviewability: low
- Review rule: 自定义参数编码方案必须有文档说明编码/解码协议。

### D-1152: HF32精度控制从per-op Attr迁移到全局编译选项

- Root cause: 精度控制粒度错误
- Hashes: 410e1a890
- Files: 多个matmul/conv算子文件
- Defect: HF32精度控制在每个算子中通过Attr逐个设置，冗余且容易遗漏。
- Fix: 移除per-op Attr，改为系统初始化时通过ACL全局编译选项统一配置。
- Reviewability: medium
- Review rule: 全局精度模式应通过编译选项统一控制。

### D-1153: logspace的TORCH_CHECK误用导致无实际校验

- Root cause: 错误检查宏误用
- Hashes: b50381663
- Files: torch_npu/csrc/aten/ops/LogSpaceKernelNpu.cpp
- Defect: `TORCH_CHECK("string")`传入非空字符串恒为true，永远不触发异常。
- Fix: 改为`TORCH_CHECK(condition, "message")`。
- Reviewability: high
- Review rule: TORCH_CHECK第一个参数必须是布尔条件，不能是字符串。

### D-1154: random_当to参数为None时未处理optional

- Root cause: optional参数未处理None
- Hashes: 5b27a8fac
- Files: torch_npu/csrc/aten/ops/RandomKernelNpu.cpp
- Defect: 直接调用`to.value()`而不检查has_value，to=None时异常。
- Fix: 改为`to.has_value() ? to.value() : get_max(self.dtype())`。
- Reviewability: high
- Review rule: 对c10::optional参数，必须先has_value()再value()。

### D-1155: NPU shutdown资源释放顺序导致core dump

- Root cause: 资源释放顺序错误
- Hashes: c180c8839 [+2 cherry-picks: 7773589d2, 30d6c720d]
- Files: torch_npu/csrc/InitNpuBindings.cpp, torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp
- Defect: shutdown时资源清理顺序不正确，device reset需在acl finalize之前。后被revert因引入新问题。
- Fix: 移除显式资源清理，依赖Finalize中的RegisterReleaseFn统一处理。
- Reviewability: low
- Review rule: 设备资源释放顺序需严格遵循：同步->清理用户资源->reset device->finalize runtime。

### D-1156: NPU shutdown后ACL单例析构导致core dump

- Root cause: 资源释放顺序错误
- Hashes: a243de6e2 [+1 cherry-pick: f2613f58e]
- Files: torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp
- Defect: ACL单例析构时stream已被销毁，访问无效stream导致crash。后也被revert。
- Fix: 在GEFinalize前显式调用aclrtDestroyStreamForce销毁stream。
- Reviewability: low
- Review rule: 依赖外部库的资源析构顺序时，需显式管理生命周期。

### D-1157: NPUQueue MakeSureQueueEmpty死锁

- Root cause: 锁/同步逻辑错误
- Hashes: bac7acf61
- Files: torch_npu/csrc/core/npu/NPUQueue.cpp
- Defect: `need_empty = false`放在了循环体外层花括号后面，内层循环结束后未重置，外层while永远为true。
- Fix: 将赋值移入内层花括号内。
- Reviewability: high
- Review rule: 循环退出条件变量的赋值位置需仔细核对作用域层级。

### D-1158: record_stream参数解析与dispatch路径错误

- Root cause: Python-C++绑定参数解析错误
- Hashes: 360419c44
- Files: torch_npu/csrc/utils/TensorMethods.cpp
- Defect: PythonArgParser参数签名与实际调用约定不匹配；dispatch未走NPU专用路径。
- Fix: 修正parser签名含self参数；直接调用NPUCachingAllocator::recordStream。
- Reviewability: medium
- Review rule: PythonArgParser的参数签名必须与实际调用约定完全匹配。

### D-1159: new_ones不支持varargs形式的size参数

- Root cause: Python参数适配缺失
- Hashes: 2ba670813
- Files: torch_npu/utils/tensor_methods.py
- Defect: `tensor.new_ones(2, 3, device='npu')`可变参数形式未被处理。
- Fix: 检测第一个参数是否为int，如果是则将连续int参数收集为tuple。
- Reviewability: high
- Review rule: Python wrapper需处理PyTorch API的所有合法调用形式。

### D-1160: new_ones和new_tensor的NPU device初始化缺失

- Root cause: 设备初始化遗漏
- Hashes: 956581db7
- Files: torch_npu/csrc/utils/TensorMethods.cpp, torch_npu/utils/tensor_methods.py
- Defect: 指定device='npu'时没有调用maybe_initialize_npu。
- Fix: 在C++层新增实现含NPU device解析和初始化。
- Reviewability: medium
- Review rule: 所有接受device参数的tensor创建方法必须包含maybe_initialize_npu调用。

### D-1161: Index算子bool索引路径复杂且存在多处缺陷

- Root cause: 算子分支逻辑错误
- Hashes: e160dbfe4
- Files: torch_npu/csrc/aten/ops/IndexKernelNpu.cpp
- Defect: bool索引处理有多个问题：check_index_aicore依赖未分配的result tensor；bool索引广播逻辑分支过多；uint8/bool类型缺少转换。
- Fix: 简化check_index_aicore；移除bool索引aicore特殊逻辑；添加类型转换。
- Reviewability: low
- Review rule: 算子路径选择不应依赖output tensor状态。

### D-1162: BN的sensitive format导致replay graph输入format不匹配

- Root cause: NPU format继承错误
- Hashes: 83504ce1f
- Files: torch_npu/npu/replay_graph.py, torch_npu/npu/npu_backend.py
- Defect: BatchNorm等算子的format side-effect导致replay graph输入format不匹配。
- Fix: 对format非ND的tensor先cast回format 0后再传入graph生成。
- Reviewability: medium
- Review rule: replay graph的输入输出必须使用统一的format(ND)。

### D-1163: mul_ bool标量0维tensor误走dtype cast路径

- Root cause: 标量tensor维度判断缺失
- Hashes: dd90ae923
- Files: torch_npu/csrc/aten/ops/MulKernelNpu.cpp
- Defect: 0维bool标量tensor也被强制转float，行为不同于标量语义。
- Fix: 添加`other.dim() != 0`条件。
- Reviewability: high
- Review rule: dtype cast前检查tensor维度，0维标量tensor需特殊处理。

### D-1164: Profiler参数语义错误

- Root cause: API参数设计错误
- Hashes: 4a9bed6a6 [+1 cherry-pick: ff7b21de8]
- Files: torch_npu/utils/profiler.py
- Defect: `total_steps`实际含义是"在第N步开始dump"；use_npu和use_e2e_profiler放在kwargs中而非显式参数。
- Fix: 重命名为start_step；提升为显式构造参数。
- Reviewability: high
- Review rule: API参数命名必须准确反映语义。

### D-1165: nms_rotated输出tensor使用原始dtype而非cast后的dtype

- Root cause: dtype cast后引用错误tensor
- Hashes: c2b2d67b1
- Files: torch_npu/csrc/aten/ops/NmsRotatedKernelNpu.cpp
- Defect: 输出tensor从原始dets(可能float16)创建而非从detsCast(float)创建。
- Fix: 将ApplyTensor(dets)改为ApplyTensor(detsCast)。
- Reviewability: high
- Review rule: dtype cast后所有后续tensor创建必须基于cast后的tensor。

### D-1166: mm算子空tensor输入导致crash

- Root cause: 边界条件未处理
- Hashes: 5450f8627
- Files: torch_npu/csrc/aten/ops/MmKernelNpu.cpp
- Defect: 不检查输入tensor是否为空，空tensor传给CANN导致crash。
- Fix: 添加空tensor检查，直接返回零结果。
- Reviewability: high
- Review rule: 所有算子必须处理空tensor输入。

### D-1167: _lazy_init在device_guard中导致回归

- Root cause: 初始化依赖循环/副作用
- Hashes: 8ab398391
- Files: torch_npu/csrc/utils/TensorMethods.cpp等
- Defect: 在device_guard中加入_lazy_init后C++层注册被移除，导致model.to(device)路径改变。后被revert。
- Fix: Revert _lazy_init在device_guard中的添加。
- Reviewability: low
- Review rule: 修改设备初始化时机需覆盖所有tensor创建和设备迁移路径做回归测试。

### D-1168: memory_summary格式化函数嵌套在create_metrics_to_display内部

- Root cause: 函数作用域错误
- Hashes: 517d0fe14
- Files: torch_npu/npu/memory.py
- Defect: _format_size和_format_count被定义为内嵌函数，外部无法引用。
- Fix: 提升为模块级函数。
- Reviewability: high
- Review rule: 被多处调用的工具函数不应嵌套在另一个函数内部。

### D-1169: fused optimizer标识属性分散在子类中

- Root cause: 属性初始化位置错误
- Hashes: 9c325d3d0
- Files: torch_npu/optim/npu_fused_optim_base.py等
- Defect: `is_npu_fused_optimizer = True`在每个子类重复设置而非基类统一设置。
- Fix: 移至基类NpuFusedOptimizerBase.__init__。
- Reviewability: high
- Review rule: 标识型属性应在基类中统一初始化。

### D-1170: GRU backward对batch_size==1时init_h维度处理错误

- Root cause: tensor维度squeeze错误
- Hashes: 6652e9ee0
- Files: torch_npu/csrc/aten/ops/GruKernelNpu.cpp
- Defect: squeeze(0)在batch_size==1时移除batch维度，导致grad_h shape错误。
- Fix: 移除squeeze(0)操作，直接使用init_h。
- Reviewability: high
- Review rule: 避免在backward中使用squeeze，因为dim==1时会丢失维度信息。

### D-1171: LSTM不支持2维(unbatched)输入

- Root cause: tensor维度适配缺失
- Hashes: 6663f93b4
- Files: torch_npu/utils/module.py
- Defect: NPU的LSTM patch仅处理3维(batched)输入，不支持2维(unbatched)。
- Fix: 添加unbatched输入检测并自动unsqueeze。
- Reviewability: medium
- Review rule: Patch PyTorch原生模块时需覆盖上游所有支持的输入形式。

### D-1172: ONNX导出npu_nms_v4遗漏max_output_size参数

- Root cause: ONNX symbolic参数遗漏
- Hashes: 78d98ce6f
- Files: torch_npu/onnx/wrapper_onnx_ops.py
- Defect: g.op调用遗漏了max_output_size_f参数。
- Fix: 添加`max_output_size_f=max_output_size`参数。
- Reviewability: high
- Review rule: ONNX symbolic函数必须将所有功能参数映射到ONNX节点属性。

### D-1173: overflow检测未在stream初始化时启用

- Root cause: stream初始化配置遗漏
- Hashes: e783f0a99 [+1 cherry-pick: 7d1f9a066]
- Files: torch_npu/csrc/core/npu/interface/AclInterface.cpp
- Defect: stream创建后未调用AclrtSetStreamOverflowSwitch，溢出检测整个训练过程关闭。
- Fix: stream创建成功后调用overflow switch启用。
- Reviewability: high
- Review rule: stream初始化需配置所有硬件相关检测开关。

### D-1174: conv_transpose2d output_padding参数被忽略

- Root cause: 算子参数透传遗漏
- Hashes: 46dba976a
- Files: torch_npu/csrc/aten/ops/convolution/ConvTranspose2dKernelNpu.cpp
- Defect: output_padding硬编码为{0,0,0,0}，用户指定的值被忽略。
- Fix: 使用实际output_padding值。
- Reviewability: high
- Review rule: 算子所有传入参数必须被使用或显式丢弃。

### D-1175: AclGetErrMsg返回nullptr导致crash

- Root cause: 空指针未检查
- Hashes: ae42d2cd5
- Files: torch_npu/csrc/core/npu/interface/AclInterface.cpp
- Defect: aclGetRecentErrMsg可能返回nullptr，直接使用导致crash。
- Fix: null检查，`return res != nullptr ? res : ""`。
- Reviewability: high
- Review rule: C字符串API返回值必须检查nullptr。

### D-1176: NPU native format tensor打印触发冗长编译

- Root cause: 数据搬运时机错误
- Hashes: 697ddb085 [+1 cherry-pick: 4df628307]
- Files: torch_npu/utils/_tensor_str.py
- Defect: _Formatter构造时to("cpu")触发ConcatD/Pack编译，print()调用极慢。
- Fix: 将to("cpu")从Formatter移到_tensor_str函数开头。
- Reviewability: medium
- Review rule: tensor print时device transfer应在格式化之前统一完成。

### D-1177: autocast API兼容性问题

- Root cause: API迁移不完整
- Hashes: 4040f9f1c
- Files: test/test_amp.py, test/test_npu.py
- Defect: `torch.is_autocast_enabled()`只反映CUDA状态，不反映NPU。
- Fix: 替换为`torch.npu.is_autocast_enabled()`。
- Reviewability: high
- Review rule: NPU环境下autocast状态查询必须使用NPU专用API。

### D-1178: deterministic algorithms配置未传递给ACL算子编译器

- Root cause: 全局配置传递遗漏
- Hashes: d0dc162e9
- Files: torch_npu/csrc/framework/OpParamMaker.cpp
- Defect: 设置deterministic模式后NPU算子仍可能使用非确定性算法。
- Fix: 在OpCommandImpl::Run中同步ACL_OP_DETERMINISTIC选项。
- Reviewability: medium
- Review rule: 全局行为配置变更需在所有执行路径上同步。
### D-1179: precision_mode条件分支无效，910B1使用了错误的精度模式

- Root cause: 条件分支copy-paste导致两个分支逻辑相同
- Hashes: 24e589fce
- Files: torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp
- Defect: 三元表达式的两个分支都返回`"allow_fp32_to_fp16"`，导致Ascend910B1本应使用`must_keep_origin_dtype`精度模式却实际使用了`allow_fp32_to_fp16`。典型的copy-paste错误，两侧字符串看起来不同实际内容相同。
- Fix: 将true分支改为`"must_keep_origin_dtype"`
- Reviewability: high
- Review rule: 三元表达式两个分支值相同时告警

### D-1180: binary_cross_entropy_with_logits_backward NPU算子dispatch错误

- Root cause: 算子注册路径错误
- Hashes: 9203c3932
- Files: torch_npu/csrc/aten/npu_native_functions.yaml
- Defect: `binary_cross_entropy_with_logits_backward`被注册在supported列表中，走NPU单算子路径。但该backward算子在NPU上实现有精度问题（尤其是带weight/pos_weight参数时），应走autograd路径由前向算子自动求导。
- Fix: 从supported列表移除该算子，让PyTorch autograd机制自动处理反向计算
- Reviewability: medium
- Review rule: backward算子注册到supported前需验证所有reduction模式和可选参数组合的精度

### D-1181: datadump多处bug修复（scalar维度解析、异步队列容量、CPU tensor误入队等）

- Root cause: datadump工具链多处缺陷
- Hashes: 5351ba61c
- Files: torch_npu/csrc/aten/ops/EnqueTensorKernelNpu.cpp, torch_npu/csrc/framework/graph/util/TdtChannelForPrint.cpp, torch_npu/hooks/hooks.py, torch_npu/hooks/wrap_torch.py, torch_npu/npu/datadump.py 等12个文件
- Defect: 多个问题合一次提交：(1) TdtChannelForPrint解析scalar tensor时对dim_size=0的情况未处理，导致VLA为空数组产生未定义行为；(2) 异步导出队列容量硬编码为3不可配置；(3) wrap_torch中缺少对部分op的包装导致dump不完整。
- Fix: (1) 添加dim_size>0判断跳过scalar的size解析；(2) capacity参数化透传到TdtChannel初始化；(3) 补齐wrap_ops
- Reviewability: medium
- Review rule: VLA(变长数组)使用前必须检查长度是否为0

### D-1182: OneHot算子不支持int64输入导致CANN报错

- Root cause: 算子dtype不支持未做适配
- Hashes: 2e865c392
- Files: torch_npu/csrc/aten/ops/OneHotKernelNpu.cpp
- Defect: CANN的OneHot算子不支持int64类型输入，但PyTorch中tensor.long()后调用one_hot会直接将int64 tensor传给CANN，导致算子执行失败。
- Fix: 在调用算子前检查self是否为kLong，若是则cast到kInt
- Reviewability: high
- Review rule: 调用CANN算子前须对照算子规格校验输入dtype是否在支持列表中

### D-1183: uniform_采样分布计算错误（inplace操作覆盖中间结果）

- Root cause: inplace操作副作用导致数学公式计算错误
- Hashes: c01fcce50
- Files: torch_npu/csrc/aten/ops/UniformKernelNpu.cpp
- Defect: 将U(0,1)映射到U(from,to)的公式`result.mul_(to).sub_(result.mul_(from).sub_(from))`有误。`result.mul_(from)`是inplace操作，执行后result已被修改，但随后的外层`result.mul_(to)`还依赖原始result值。实际计算变成了`result*from*to - result*from + from`而非预期公式。
- Fix: 用non-inplace的`result.mul(from).sub(from)`先计算临时变量tmp，再`result.mul_(to).sub_(tmp)`
- Reviewability: high
- Review rule: 链式inplace操作中，同一tensor不能既作为被修改对象又作为后续表达式的输入

### D-1184: index_put未将indices搬到与self相同的device

- Root cause: 跨device tensor未对齐
- Hashes: bfbda2826
- Files: torch_npu/csrc/aten/ops/IndexPutKernelNpu.cpp
- Defect: 当index_put的indices tensor与self不在同一device时（如indices在CPU，self在NPU），直接传给CANN算子会出错。缺少device一致性检查和自动搬移。
- Fix: 遍历allDefinedIndices，对device不一致的index执行`.to(self.device())`
- Reviewability: high
- Review rule: 多tensor输入的算子需检查所有输入tensor的device一致性

### D-1185: 构建脚本用sh调用含bash语法的脚本，在Ubuntu dash环境下失败

- Root cause: shell兼容性问题
- Hashes: eb13e4af2
- Files: build_libtorch_npu.py, scripts/generate_code.sh, setup.py
- Defect: generate_code.sh中使用了`[[ ]]`等bash特有语法，但调用时用`sh`启动。Ubuntu默认sh是dash，不支持`[[ ]]`语法导致构建失败。同时脚本内`if [[  ${build_libtorch}` 多了一个空格。
- Fix: 将调用命令从`"sh"`改为`"bash"`，修正多余空格
- Reviewability: high
- Review rule: 使用bash语法的脚本必须用bash调用，或在shebang中声明#!/bin/bash

### D-1186: NPU内存管理实现有缺陷需回退

- Root cause: 内存管理功能实现不成熟导致回退
- Hashes: ef47a4d8e
- Files: torch_npu/csrc/core/npu/NPUCachingAllocator.cpp, torch_npu/npu/memory.py 等6个文件
- Defect: 新增的set_per_process_memory_fraction等内存管理功能在CachingAllocator中的实现存在问题（具体表现为内存回收/分配策略缺陷），需要回退到之前的实现。回退恢复了原有的unordered_map数据结构和更稳定的分配逻辑。
- Fix: Revert整个内存管理改动，恢复到验证过的实现
- Reviewability: low
- Review rule: CachingAllocator改动需在多卡、大batch、OOM边界场景充分压测后再合入

### D-1187: max算子output为非连续tensor（slice）时结果写入错误

- Root cause: 非连续output tensor处理缺失
- Hashes: 2fa5dc6b0
- Files: torch_npu/csrc/aten/ops/MaxKernelNpu.cpp
- Defect: 当max_out的output/indices参数是另一个tensor的slice（非连续）时，CANN算子假设输出是连续内存，直接写入会导致数据错位。同样影响amax_out和element-wise max_out。
- Fix: 检查output.is_contiguous()，若非连续则先计算到临时连续tensor再copy_回output
- Reviewability: high
- Review rule: 所有带out参数的算子必须处理output非连续的情况

### D-1188: embedding_bag在0D（空输入）场景下crash

- Root cause: 空tensor边界条件未处理
- Hashes: cc5760e51
- Files: torch_npu/csrc/aten/ops/EmbeddingBagKernelNpu.cpp, torch_npu/csrc/aten/ops/EmbeddingBagBackwardKernelNpu.cpp
- Defect: 当embedding_bag的输入indices为空tensor时，前向和反向算子未针对空输入做特殊处理，导致访问空数据或shape推导出错而crash。
- Fix: 在前向和反向中添加空输入的分支处理
- Reviewability: high
- Review rule: 所有算子需测试空tensor输入（numel=0）的边界场景

### D-1189: overflow switch设置导致coredump

- Root cause: 硬件API调用时序不当导致coredump
- Hashes: 39a2f3a71
- Files: torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp
- Defect: 在NPU系统初始化阶段调用`AclrtSetStreamOverflowSwitch`设置溢出开关，但此时stream可能尚未完全就绪，导致coredump。该功能需要在stream完全初始化后的合适时机设置。
- Fix: 回退删除初始化阶段的overflow switch设置代码
- Reviewability: medium
- Review rule: 初始化阶段调用硬件API前必须确认相关资源（stream/device）已就绪

### D-1190: HCCL支持的数据类型映射不全，缺少fp64/bf16/uint类型

- Root cause: dtype映射表不完整
- Hashes: ec1395541
- Files: third_party/hccl/inc/hccl/hccl_types.h, torch_npu/csrc/distributed/ProcessGroupHCCL.cpp, torch_npu/distributed/hccl_dtype_wraper.py
- Defect: hcclDataType映射表缺少kDouble(fp64)、kByte(uint8)等类型的映射，导致使用这些dtype的tensor做集合通信时抛出"Unsupported data type"异常。同时AllReduce缺少对所支持dtype的显式校验，不支持的类型会给出不明确的错误。
- Fix: 扩展hccl_types.h枚举和ScalarType到HcclDataType映射表，新增AllReduce的dtype校验函数
- Reviewability: high
- Review rule: 扩展dtype支持时需同步更新header定义、映射表和校验逻辑三处

### D-1191: RTS特性不支持时NPU stream创建直接报错退出

- Root cause: 硬件API兼容性降级处理缺失
- Hashes: 2b057da5f
- Files: torch_npu/csrc/core/npu/NPUException.h, torch_npu/csrc/core/npu/NPUStream.cpp
- Defect: `AclrtCreateStreamWithConfig`使用了ACL_STREAM_FAST_LAUNCH等新特性flag，在不支持该特性的RTS版本上会返回ACL_ERROR_RT_FEATURE_NOT_SUPPORT。原代码用C10_NPU_CHECK直接对该错误抛异常终止程序，导致老版本驱动上完全无法启动。
- Fix: 新增NPU_CHECK_SUPPORTED_OR_ERROR宏，对FEATURE_NOT_SUPPORT错误做静默降级而非终止
- Reviewability: high
- Review rule: 使用可选硬件特性的API调用必须处理FEATURE_NOT_SUPPORT返回码

### D-1192: aicore模式下ViewCopy的src_stride计算基于非连续tensor导致数据错乱

- Root cause: 非连续tensor的stride传递错误
- Hashes: 50d3504b5
- Files: torch_npu/csrc/aten/common/CopyKernelNpu.cpp
- Defect: 在aicore模式的copy路径中，直接用src.strides()获取stride传给ViewCopy算子。但如果src是非连续tensor（如NPU私有format），其stride与实际内存布局不匹配。同时AicoreValid缺少对diff_index==0的判断，导致不需要viewcopy的场景也走了这条路径。
- Fix: 先对src做format_contiguous再取stride；AicoreValid增加diff_index==0返回false的条件
- Reviewability: medium
- Review rule: 向CANN算子传递stride信息前，必须确保tensor已经是连续格式或stride与实际内存一致

### D-1193: device参数解析未处理None值和自定义NPU device类型

- Root cause: device参数解析不完整
- Hashes: 5913f4384 [+2 cherry-picks: 97264f66e, c95745e79]
- Files: scripts/codegen/templates/DispatchKeyNativeFunctions.h, torch_npu/csrc/core/Device.cpp
- Defect: `parse_npu_device`函数仅检查`!obj`（NULL），未处理Python None值（`obj == Py_None`），导致`torch.tensor(data, device=None)`时将Py_None当作有效device对象进行reinterpret_cast引发段错误。同时函数不识别TNPDevice类型的自定义device对象。
- Fix: 添加`obj == Py_None`判断返回默认device；添加TNPDevice类型识别分支
- Reviewability: high
- Review rule: PyObject参数解析必须处理NULL和Py_None两种空值语义

### D-1194: isinstance对torch.device的判断在NPU扩展device类型下失败

- Root cause: isinstance monkey-patch逻辑不完备
- Hashes: 24d1c22e1
- Files: torch_npu/__init__.py, torch_npu/npu/utils.py
- Defect: `_isinstance`中用`eval("torch.device") == class_or_tuple`做device类型判断，当class_or_tuple是tuple时直接报错。同时`set_device`和`_get_device_index`中用`isinstance(device, torch._C.device)`检查，无法识别torch_npu._C.device类型的对象。
- Fix: 将class_or_tuple统一转为tuple处理，同时检查torch._C.device和torch_npu._C.device是否在tuple中；将torch._C.device替换为torch.device（后者已被monkey-patch覆盖两种类型）
- Reviewability: medium
- Review rule: monkey-patch的isinstance需覆盖原始类型和扩展类型的所有组合

### D-1195: serialization中device type处理bug（typo和类型判断错误）

- Root cause: typo + 类型判断逻辑错误
- Hashes: 41a0d2e91
- Files: torch_npu/utils/serialization.py
- Defect: `_npu_tag`中`obj.get_devic()`缺少字母e（typo），导致序列化时无法正确标记NPU tensor。`normalize_map_location_type`中用`isinstance(map_location, torch_npu.utils.device_guard.device)`判断，但map_location可能是多种类型（str、torch.device等），直接`str(map_location)`更通用。
- Fix: 修正typo为`get_device()`；简化normalize_map_location_type为`return str(map_location)`
- Reviewability: high
- Review rule: 函数调用中的方法名拼写应由CI的类型检查或import验证捕获

### D-1196: storage_resize_npu中memcpy的size参数重复乘以itemsize

- Root cause: 内存拷贝大小计算重复乘因子
- Hashes: ef26f74b4 [+1 cherry-pick: e7bf9e6a1]
- Files: torch_npu/csrc/aten/common/ResizeNpu.h
- Defect: `LaunchAsyncCopyTaskWithModeSwitch`的copy_size已经是byte单位的大小，但调用时又乘了`itemsize`，导致拷贝长度是实际的N倍（N=每个元素的字节数）。这会越界读写内存。
- Fix: 移除多余的`itemsize *`乘法，直接传copy_size
- Reviewability: high
- Review rule: 内存拷贝API的size参数单位（byte vs element）必须在调用处注释说明

### D-1197: datadump误将CPU tensor入队到NPU通道

- Root cause: device类型检查缺失
- Hashes: 589c19a5e
- Files: torch_npu/hooks/hooks.py
- Defect: `datadump_enque`函数对输入tensor只检查`isinstance(i, torch.Tensor)`，未检查tensor是否在NPU上。当CPU tensor被传入时，调用`npu_enque_tensor`会失败或产生错误数据。
- Fix: 增加`i.device.type == 'npu'`条件判断
- Reviewability: high
- Review rule: 涉及设备特定操作的函数必须校验tensor所在device

### D-1198: ScalarType枚举表未随PyTorch 1.11新增QUInt2x4类型更新

- Root cause: 上游版本升级后映射表未同步
- Hashes: 3e74e3328
- Files: torch_npu/csrc/framework/utils/CalcuOpUtil.cpp
- Defect: PyTorch 1.11新增了`at::ScalarType::QUInt2x4`枚举值，但kATenScalarTypeToAclDataTypeTable未添加对应条目。由于该表使用static_assert逐项校验索引位置，缺少一项会导致后续所有项的索引偏移，引发编译期或运行期的dtype映射错误。
- Fix: 在QUInt4x2之后添加`_(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)`
- Reviewability: high
- Review rule: 升级PyTorch版本后须检查所有ScalarType枚举映射表是否需要同步更新

### D-1199: resize操作后NPU StorageDesc的format和storage_sizes未正确更新

- Root cause: NPU私有format下resize逻辑不完整
- Hashes: d28187650
- Files: torch_npu/csrc/aten/common/ResizeNpu.h, torch_npu/csrc/framework/StorageDescHelper.cpp
- Defect: `UpdateDesc`仅使用new_size更新base_sizes和storage_sizes，但未区分"数据shape"和"存储shape"。在NCDHW等5D格式的tensor做resize时，storage_sizes应该根据format计算（调用GetStorageSizes），而非直接赋值为new_size。同时npu_format_需要根据实际情况更新而非直接GuessStorageFormat。
- Fix: UpdateDesc接受两个参数（data_sizes和shape_sizes），取较大的numel对应的size；storage_sizes改用FormatHelper::GetStorageSizes计算
- Reviewability: medium
- Review rule: NPU format下的shape操作需同时更新base_sizes、storage_sizes和npu_format_三者的一致性

### D-1200: FusedAttentionScore的dropout mask shape硬编码除以16导致非整除场景crash

- Root cause: 硬编码shape计算不通用
- Hashes: e2a540870
- Files: torch_npu/csrc/aten/ops/FusedAttentionScoreKernelNpu.cpp
- Defect: `dropout_gen_mask_v3`中mask的selfShape硬编码为`{self.size(0), self.size(1), self.size(2)/16, self.size(3)/16, 16, 16}`。这要求size(2)和size(3)必须是16的整数倍，否则整除截断导致mask shape错误，且与实际tensor的numel不匹配。
- Fix: 直接使用`self.sizes()`作为selfShape，让底层API自行处理对齐
- Reviewability: high
- Review rule: 硬编码的shape变换（除法/乘法）必须有对齐约束校验或使用通用接口

### D-1201: empty_strided后resize导致NPU StorageDesc基于错误shape更新

- Root cause: resize与empty_strided的交互逻辑缺陷
- Hashes: d161b1867
- Files: torch_npu/csrc/aten/common/ResizeNpu.h, torch_npu/csrc/framework/StorageDescHelper.cpp, torch_npu/csrc/aten/common/TensorFactories.cpp
- Defect: `storage_resize_npu`直接用`new_size`（用户请求的逻辑shape）更新StorageDesc，但对于empty_strided创建的tensor，其storage大小由strides决定而非sizes。resize时应根据实际的nbytes/itemsize计算storage的flat shape，而非直接用逻辑size。此外原有的checkInBoundsForStorage函数存在冗余。
- Fix: resize_shape改为`{size/itemsize}`（即存储层面的一维flat shape）；增加itemsize为0和size不整除的校验
- Reviewability: medium
- Review rule: storage层面的resize必须基于bytes计算，不能直接使用tensor的逻辑shape

### D-1202: datadump在torch v1.11.0合并后graph模式string输入处理失败

- Root cause: 版本升级后graph模式API适配断裂
- Hashes: 34c9f17eb
- Files: torch_npu/csrc/framework/OpCommand.cpp, torch_npu/csrc/framework/graph/construct/GraphConstructor.cpp
- Defect: GraphCommandImpl::AddInput(const string&)的实现在v1.11合并后与底层不兼容。原实现手动将string编码为tensor并通过host memcpy传递，但新版本下该路径不再被使用且会导致datadump失败。同时OpCommand::Input(const string&)中有一个不可达的AT_ERROR。
- Fix: 删除GraphConstructor中废弃的string AddInput实现；OpCommand中改为走统一的tensor输入路径
- Reviewability: medium
- Review rule: 版本合并后需检查graph模式相关的所有算子输入路径是否仍然��容

### D-1203: torch.load的map_location为"npu"（不带index）时device.index为None导致后续报错

- Root cause: 可选参数None值未设默认值
- Hashes: fe9f1af62
- Files: torch_npu/utils/serialization.py
- Defect: `validate_npu_device`中用`torch.device(location)`解析map_location。当location为`"npu"`（不带`:0`）时，`device.index`为None，后续用None做设备索引会报错。
- Fix: `device.index if device.index else 0`，为None时默认使用device 0
- Reviewability: high
- Review rule: 从torch.device取index后必须处理None值（无显式index的情况）

### D-1204: silu inplace backward因autograd保存的tensor被inplace修改导致梯度错误

- Root cause: inplace操作与autograd saved tensor冲突
- Hashes: 8fced2d8d
- Files: torch_npu/csrc/aten/ops/SiluKernelNpu.cpp
- Defect: silu的自定义autograd Function在forward中`save_for_backward({self, result})`保存了输入和输出。但当用户使用`SiLU(inplace=True)`时，inplace操作修改了input的数据，导致backward时saved的self已被覆盖，计算出错误的梯度。此外函数名有typo `silu_kerner_npu`。
- Fix: 修正非inplace路径，确保saved tensor不被inplace修改；修正typo为`silu_kernel_npu`
- Reviewability: medium
- Review rule: autograd Function中save_for_backward的tensor必须保证在backward之前不被inplace修改

### D-1205: PyTorch版本升级后编译错误（API签名变更）

- Root cause: 上游API签名变更未适配
- Hashes: 4a314b171
- Files: patch/include/torch/csrc/generic/Storage.h, patch/include/torch/csrc/generic/serialization.h, torch_npu/csrc/aten/ops/LogSpaceKernelNpu.cpp, torch_npu/csrc/aten/ops/convolution/ConvolutionKernelNpu.cpp
- Defect: PyTorch版本升级后多个API签名变更：(1) THPStorage_接口改为使用c10::intrusive_ptr<c10::StorageImpl>；(2) serialization接口增加element_size参数；(3) logspace的start/end改为const引用，steps不再是optional；(4) _convolution的bias参数名变更。未同步导致编译失败。
- Fix: 逐一适配新API签名
- Reviewability: high
- Review rule: 升级PyTorch版本须通过完整编译验证所有patch和adapter代码

### D-1206: hardsigmoid_（inplace版本）未实现导致静默错误

- Root cause: inplace算子未注册
- Hashes: 68e6904aa
- Files: torch_npu/csrc/aten/ops/HardsigmoidKernelNpu.cpp
- Defect: `torch.nn.functional.hardsigmoid(input, True)`调用inplace版本的hardsigmoid_，但NPU侧只实现了非inplace版本。缺少inplace dispatch导致调用时结果不正确或走了fallback路径。
- Fix: 补充inplace版本的实现
- Reviewability: high
- Review rule: 算子同时有inplace和非inplace变体时需成对实现

### D-1207: Sub算子对0维device tensor做了不必要的item()调用引发D2H同步

- Root cause: 不必要的device-to-host同步
- Hashes: 4a079401b
- Files: torch_npu/csrc/aten/ops/SubKernelNpu.cpp
- Defect: Sub算子中`if (other.dim() == 0)`就走scalar分支调用`other.item()`。但当other是NPU上的0维tensor时，item()会触发device到host的同步拷贝，导致pipeline stall。只有CPU上的0维tensor才需要item()提取scalar。
- Fix: 增加`!at_npu::key::isDeviceTensor(other)`条件，仅CPU tensor走scalar分支
- Reviewability: high
- Review rule: 对可能在device上的tensor调用item()前必须检查是否在device上

### D-1208: LSTM gate order不一致导致NPU与CPU结果不匹配

- Root cause: 算子参数语义与上游不一致
- Hashes: c837ed95f
- Files: torch_npu/csrc/aten/ops/LstmKernelNpu.cpp
- Defect: DynamicRNN算子的默认gate order与PyTorch CPU LSTM的gate order不同。CPU的LSTM使用ifjo（input, forget, cell/gate, output）顺序，而DynamicRNN算子默认使用不同的顺序。之前测试用例中通过手动重排weight来补偿这个差异，但这只是workaround，且在多层/双向LSTM中容易出错。
- Fix: 在DynamicRNN和DynamicRNNGrad调用中显式设置`.Attr("gate_order", "ifjo")`，移除测试中的weight重排代码
- Reviewability: medium
- Review rule: 调用CANN算子时，所有影响数学语义的attr（如gate_order、data_format）必须显式指定，不依赖默认值

### D-1209: layer_norm对weight/bias做reshape后未恢复原始shape（副作用泄漏）

- Root cause: 算子内部reshape副作用影响外部tensor
- Hashes: bde8b6534
- Files: torch_npu/csrc/aten/ops/LayerNormKernelNpu.cpp
- Defect: `layer_norm_npu_support`中当weight的shape与weightDims不匹配时会对weight做resize_。但weight是用户传入的模型参数，resize_是inplace操作会永久改变weight的shape。下一次forward时weight shape已是错误的，导致后续计算错误。
- Fix: 在resize前保存ori_weight_shape和ori_bias_shape，算子执行完成后恢复
- Reviewability: high
- Review rule: 算子内部不得对输入tensor做有副作用的inplace shape操作（resize_/reshape_）；若必须，则在函数退出前恢复

### D-1210: setup.py在version.py已存在时因O_EXCL标志创建失败

- Root cause: 文件创建的幂等性缺失
- Hashes: 28be3708c
- Files: setup.py
- Defect: `generate_torch_npu_version`使用`os.O_EXCL`标志创建version.py，该标志要求文件必须不存在。重复构建时version.py已存在，导致FileExistsError编译中断。
- Fix: 在创建前检查文件是否存在，若存在则先unlink删除
- Reviewability: high
- Review rule: 构建脚本中的文件生成操作必须具备幂等性

### D-1211: median算子返回的indices dtype与CPU不一致；fill_对device tensor调用item()

- Root cause: dtype不一致 + 不必要的D2H同步（两个bug合一次提交）
- Hashes: 4dd75516a
- Files: torch_npu/csrc/aten/ops/MedianKernelNpu.cpp, torch_npu/csrc/aten/ops/FillKernelNpu.cpp
- Defect: (1) median算子的indices输出在NPU上使用int32，但CPU返回int64(long)，导致精度对比失败；(2) fill_中对tensor类型的other做了`fill_out_npu`调用，但底层的Fill算子（非fills）实现有问题，且对device上的0维tensor做item()会触发同步。
- Fix: (1) indices的临时tensor改用long类型；(2) fill_统一走fills_out_npu的scalar路径
- Reviewability: medium
- Review rule: 算子输出的dtype必须与CPU参考实现一致

### D-1212: Tensor.npu(device=None)时对None做字符串操作报错

- Root cause: None值未做类型检查
- Hashes: 58f18b529 [+1 cherry-pick: f198c3954]
- Files: torch_npu/utils/tensor_methods.py
- Defect: `_npu`函数中`kwargs.get("device", "")`当device=None时返回None而非空字符串。随后`'npu' in None`抛出TypeError。
- Fix: 先判断device是否存在且为str类型，再做字符串操作
- Reviewability: high
- Review rule: 从kwargs获取可能为None的值后，在做字符串操作前必须做类型检查

### D-1213: indexing_opt对非对齐stride的tensor未做有效性校验导致计算错误

- Root cause: contiguous优化路径的stride校验缺失
- Hashes: ac490f798
- Files: torch_npu/csrc/framework/contiguous/indexing_opt.cpp
- Defect: indexing优化路径假设indexing_stride是base_stride的整数倍，用除法计算step。但当stride不满足这个约束时，除法结果错误导致数据访问越界或结果错误。
- Fix: 添加前置校验：若indexing_stride < base_stride或不整除，则return false放弃indexing优化路径
- Reviewability: medium
- Review rule: contiguous优化路径必须严格校验stride约束条件后再执行优化

### D-1214: 精度对比工具在多进程DDP场景下所有rank都dump导致数据混乱

- Root cause: 多进程场景下工具未做进程隔离
- Hashes: 98af8a627 [+1 cherry-pick: 394b9560c]
- Files: torch_npu/hooks/hooks.py, torch_npu/hooks/initialize.py
- Defect: accuracy comparison tool的hook在所有进程上都执行dump，多进程DDP训练时各rank的数据混写到同一目录导致数据混乱和文件冲突。
- Fix: register_hook时记录当前pid，hook执行时检查`pid == os.getpid()`，仅在注册进程上执行dump
- Reviewability: medium
- Review rule: DDP场景下的调试工具hook必须支持进程过滤，默认只在rank 0执行

### D-1215: allgather输入为NPU私有格式时精度错误

- Root cause: 集合通信未处理NPU私有format
- Hashes: 856791449 [+1 cherry-pick: ed40d2c20]
- Files: torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: allgather直接将输入tensor传给HCCL，但当tensor是NPU私有格式（如FRACTAL_NZ、NC1HWC0）时，HCCL按照连续内存layout读取数据会得到错误结果。
- Fix: allgather前调用cast_to_origin_format，将非base格式的tensor转回origin_format
- Reviewability: medium
- Review rule: 传给HCCL的tensor必须是base format；集合通信入口需添加format校验

### D-1216: checkpoint功能未注册到torch_npu导致NPU上checkpoint不生效

- Root cause: 模块注册遗漏
- Hashes: 41a9302ec
- Files: torch_npu/__init__.py, torch_npu/utils/__init__.py, torch_npu/utils/checkpoint.py
- Defect: torch_npu实现了自定义的checkpoint和checkpoint_sequential（处理NPU上的RNG state保存/恢复），但未在__init__.py中调用`add_checkpoint_methods()`注册。
- Fix: 在apply_class_patches中添加add_checkpoint_methods()调用
- Reviewability: high
- Review rule: 新增的monkey-patch模块必须在__init__.py中注册

### D-1217: Index/IndexPut的aicore自适应分支引入regression需回退

- Root cause: 新优化路径引入regression需回退
- Hashes: 57321fab5 [+2 cherry-picks: 795bff936, 3bf1ddde1]
- Files: torch_npu/csrc/aten/ops/IndexKernelNpu.cpp, torch_npu/csrc/aten/ops/IndexPutKernelNpu.cpp
- Defect: Index和IndexPut新增了aicore分支在多种边界场景下存在正确性问题，导致regression。
- Fix: Revert整个aicore自适应分支
- Reviewability: low
- Review rule: 算子新增优化路径需在所有边界场景通过完整测试矩阵后才能合入

### D-1218: _unique2对空tensor输出size为0导致CANN报错

- Root cause: 算子输出tensor size为0时CANN不兼容
- Hashes: a03ac753d
- Files: torch_npu/csrc/aten/ops/_Unique2KernelNpu.cpp
- Defect: 当return_inverse或return_counts为false时，输出tensor的shape为`{0}`。CANN算子不支持size为0的输出tensor。
- Fix: 将size为`{0}`改为`{1}`作为最小占位
- Reviewability: high
- Review rule: CANN算子的输出tensor size不能为0，需用最小size占位

### D-1219: maximum算子缺少broadcast计算导致输出shape错误

- Root cause: broadcast逻辑缺失
- Hashes: 29c8a0f8a
- Files: torch_npu/csrc/aten/ops/MaxKernelNpu.cpp
- Defect: `maximum_out`直接用self的shape作为输出shape，未计算self和other的broadcast后shape。
- Fix: 用broadcast_ops_npu_output_size计算输出shape
- Reviewability: high
- Review rule: 二元算子必须使用broadcast_ops_npu_output_size计算输出shape

### D-1220: smooth_l1_loss_backward未传递beta参数

- Root cause: 算子参数透传遗漏
- Hashes: e557a670f [+1 cherry-pick: 50fe7a9dc]
- Files: torch_npu/csrc/aten/ops/loss/SmoothL1LossBackwardKernelNpu.cpp
- Defect: NPU的backward实现中未将beta传递给CANN算子，导致backward计算始终使用默认beta=1.0。
- Fix: 在backward调用中传递beta参数
- Reviewability: high
- Review rule: 前向算子新增参数时，必须同步检查backward是否也需要该参数

### D-1221: destroy_process_group未释放HCCL通信资源导致内存泄漏

- Root cause: 资源释放遗漏
- Hashes: e6dbc7511
- Files: torch_npu/csrc/distributed/HCCLUtils.hpp, torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
- Defect: ProcessGroupHCCL的析构函数为空，调用destroy_process_group后HCCL communicator不会被销毁。
- Fix: 析构函数调用destropyHcclComm()并加mutex保护
- Reviewability: medium
- Review rule: 持有外部资源句柄的RAII类的析构函数不能为default/空

### D-1222: HCCL dtype wrapper对keyword参数传递的tensor不做类型转换

- Root cause: kwargs路径遗漏处理
- Hashes: d9497766b
- Files: torch_npu/distributed/hccl_dtype_wraper.py
- Defect: `wrapper_dist_dtype_one_input`只处理了args[0]的dtype转换，未处理kwargs['tensor']路径。
- Fix: 增加kwargs['tensor']路径的dtype转换处理
- Reviewability: high
- Review rule: wrapper函数必须同时处理args和kwargs两种传参方式

### D-1223: div算子对不同dtype的输入未做类型提升

- Root cause: dtype promotion逻辑缺失
- Hashes: c7ecb68fd
- Files: torch_npu/csrc/aten/ops/DivKernelNpu.cpp
- Defect: `torch.div(int_tensor, float_scalar)`时NPU未按照PyTorch的类型提升规则处理，int/10返回int（截断）而非float。
- Fix: 使用at::native::result_type计算高优先级类型
- Reviewability: high
- Review rule: 所有算术算子必须遵循PyTorch的dtype promotion规则

### D-1224: DDP多进程并发创建profiler输出目录时TOCTOU竞争

- Root cause: TOCTOU文件系统竞争条件
- Hashes: c8aff7cb3
- Files: torch_npu/npu/npu_frontend_enhance.py
- Defect: profiler先`os.path.exists()`检查再`os.makedirs()`创建目录，在DDP多进程场景下存在TOCTOU竞争。
- Fix: 替换为`os.makedirs(self.result_path, exist_ok=True)`
- Reviewability: high
- Review rule: 多进程环境下目录创建必须使用exist_ok=True

### D-1225: BatchNorm3d的format硬编码为NCHW导致5D输入format不匹配

- Root cause: 硬编码format不适配高维输入
- Hashes: c5924abc1
- Files: torch_npu/csrc/aten/ops/normalization/BatchNormKernelNpu.cpp, torch_npu/utils/module.py
- Defect: BatchNorm算子中间输出format硬编码为ACL_FORMAT_NCHW，对BatchNorm3d的5D输入不匹配。
- Fix: format从硬编码NCHW改为跟随输入
- Reviewability: medium
- Review rule: BatchNorm系列算子的format必须跟随输入维度适配

### D-1226: wheel安装方式下NPU tensor类型未正确注册

- Root cause: 模块初始化路径差异导致功能缺失
- Hashes: 89179e914
- Files: torch_npu/csrc/InitNpuBindings.cpp, torch_npu/csrc/utils/TensorType.cpp
- Defect: wheel安装方式下初始化路径被跳过，NPU tensor类型未注册。
- Fix: 将初始化移到通用注册链路中
- Reviewability: medium
- Review rule: 扩展模块的初始化函数必须在所有安装方式下都被注册

### D-1227: support_wrap_ops.yaml中meshgrid不兼容dump hook

- Root cause: 工具配置列表包含不兼容的op
- Hashes: 400f4c985
- Files: torch_npu/hooks/support_wrap_ops.yaml
- Defect: meshgrid的输入/输出模式与dump hook的通用处理逻辑不兼容，导致报错。
- Fix: 从support_wrap_ops.yaml中移除meshgrid
- Reviewability: high
- Review rule: dump工具的op白名单需逐一验证兼容性
### D-1228: linspace/range中不合理的host-to-device显式拷贝

- Root cause: 冗余host-to-device拷贝
- Hashes: a370d2c5e
- Files: LinspaceKernelNpu.cpp, RangeKernelNpu.cpp
- Defect: linspace和range算子手动构造CPU辅助tensor后调用`copy_tensor_host_to_device`做显式H2D拷贝。OpCommand的`.Input()`接口已具备直接接收host SmallVector并自动搬运的能力，手动拷贝既多余又引入不必要的中间tensor分配。range中还使用`vector<int>`而非`SmallVector<int64_t, N>`，类型不匹配。
- Fix: 删除`linspace_assist`辅助函数和显式拷贝，range中改用`SmallVector<int64_t, N>`并通过`.Input(tmp_vector, scalar_type)`让框架自动处理H2D。
- Reviewability: high
- Review rule: 搜索`copy_tensor_host_to_device`调用，确认是否可用OpCommand的`.Input()`接口替代。

### D-1229: masked_select输出shape计算使用sum而非numel导致与PyTorch 1.8不一致

- Root cause: 输出shape计算错误
- Hashes: 716cdd9d3
- Files: MaskedSelectKernelNpu.cpp
- Defect: `masked_select_npu_output_size`使用`mask.sum().item().toLong()`计算输出大小，这需要在NPU上执行reduce操作获取真实匹配数。PyTorch 1.8的语义是预分配与mask同numel大小的输出buffer，实际有效元素数由算子内部确定。NPU实现错误地预先计算了精确大小，导致shape不一致。同时使用了`c10::MaybeOwned`的`expand_outplace`，存在兼容性问题。
- Fix: 将输出size改为`maskCast.numel()`（最大可能大小），自定义`expand_outplace_npu`替代标准库函数。
- Reviewability: medium
- Review rule: 动态shape算子的输出size策略必须与PyTorch CPU语义对齐，检查是否存在预分配大小与实际填充数不一致的情况。

### D-1230: cat对空tensor列表缺少early return导致crash

- Root cause: 空输入边界条件缺失
- Hashes: 605bc33d1
- Files: CatKernelNpu.cpp
- Defect: `cat`算子在所有有效tensor过滤后`inputTensors`为空时，仍继续执行后续的dim计算和算子调用，导致访问空容器crash。`cat_out`、`_cat`、`cat`三个入口均缺少此检查。
- Fix: 在`inputTensors.size() == 0`的分支添加early return，直接返回空result tensor。
- Reviewability: high
- Review rule: 所有接受tensor列表的算子必须检查过滤后列表为空的情况。

### D-1231: conv反向中调用at::sum_out导致dispatch不到NPU实现

- Root cause: 算子dispatch路径错误
- Hashes: a5261b877
- Files: SlowConvTranspose2dBackwardKernelNpu.cpp, Conv2dBackwardKernelNpu.cpp, Conv3dBackwardKernelNpu.cpp, ConvTranspose2dBackwardKernelNpu.cpp, ConvTranspose3dBackwardKernelNpu.cpp
- Defect: conv类算子反向传播计算bias梯度时调用`at::sum_out()`，该调用走PyTorch的标准dispatch而非NPU实现。在NPU tensor上可能导致不正确的行为或性能下降。
- Fix: 将`at::sum_out(gradBias, gradView, dims)`替换为`NPUNativeFunctions::sum_out(gradView, dims, false, gradView.scalar_type(), gradBias)`，显式调用NPU实现。
- Reviewability: medium
- Review rule: NPU算子内部不应调用`at::`命名空间的标准算子操作tensor，应直接调用`NPUNativeFunctions::`以确保走NPU路径。

### D-1232: index_put中异步拷贝masks导致数据竞争

- Root cause: 冗余host-to-device拷贝
- Hashes: 291e32ad7
- Files: IndexPutKernelNpu.cpp
- Defect: `index_put`中将masks向量通过`copy_tensor_host_to_device`显式拷贝到device，存在异步拷贝与后续算子执行之间的数据竞争风险。OpCommand框架已支持直接传入host侧SmallVector。
- Fix: 删除`copy_tensor_host_to_device`调用，改用`.Input(masks, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)`让框架管理搬运时序。
- Reviewability: high
- Review rule: 禁止在算子实现中手动做`copy_tensor_host_to_device`，统一使用OpCommand的`.Input()`接口。

### D-1233: Tensor.npu()不支持"npu:x"字符串格式的device参数

- Root cause: device参数解析不完整
- Hashes: c39619f3b [+1 cherry-pick: 510bf64e0]
- Files: torch_npu/utils/tensor_methods.py
- Defect: `Tensor.npu("npu:0")`传入包含设备编号的字符串时，底层`torch_npu._C.npu()`无法识别"npu"前缀，因为内部使用的是native_device名称（如"xla"）。参数中的"npu"字符串未被转译为native_device名称。
- Fix: 在`_npu`函数入口检查args和kwargs中的"npu"字符串，replace为`torch_npu.npu.native_device`。
- Reviewability: high
- Review rule: 所有接受device参数的公共API入口都需要做"npu" -> native_device的转译。

### D-1234: mul(tensor, scalar)在非浮点scalar时输出dtype错误

- Root cause: 类型推导错误
- Hashes: be54a1ae1
- Files: MulKernelNpu.cpp
- Defect: `mul`算子在tensor与scalar运算时，`binary_op_check`推导的`common_type`对整数scalar不正确。当scalar为整数时（如`tensor_int8 * 0.5`），输出dtype未正确提升，导致结果与PyTorch CPU行为不一致。Bool tensor的处理也缺失。
- Fix: 在`muls_out_npu`中，当scalar非浮点时，显式设置`common_type = self.scalar_type()`；当self为kBool时，使用`other.type()`。
- Reviewability: medium
- Review rule: 算子的dtype提升规则必须与PyTorch的`result_type`语义一致，特别关注scalar参与运算时的提升规则。

### D-1235: resize_impl_npu_在graph mode下错误判断storage容量

- Root cause: Graph mode下storage容量判断错误
- Hashes: cddeb21ae
- Files: ResizeNpu.h
- Defect: `maybe_resize_storage_npu`使用`self->storage().nbytes()`判断是否需要扩容，但在graph mode下，storage的nbytes不反映实际分配的capacity（graph mode有独立的capacity追踪）。导致不必要的重复分配或分配不足。此外`storage.npu_desc_`的直接访问方式在某些场景下不正确，需通过`NPUBridge::GetNpuStorageImpl`获取。
- Fix: graph mode下使用`GraphUtils::GetTensorCapacity`获取真实capacity；通过`NPUBridge::GetNpuStorageImpl`访问npu_desc_；移除`IsGraphMode`对`StorageDescHelper::UpdateDesc`的guard。
- Reviewability: low
- Review rule: storage相关操作必须区分graph mode和eager mode的不同capacity追踪机制。

### D-1236: max算子不支持不同dtype输入

- Root cause: 类型推导错误
- Hashes: 58a3cd26d [+1 cherry-pick: 97b5cc374]
- Files: MaxKernelNpu.cpp
- Defect: `max_out`和`maximum`在两个输入tensor的dtype不同时（如float32与int32），直接传入算子而未做类型提升，导致ACL算子报错或结果不正确。PyTorch CPU侧会自动做`result_type`提升。
- Fix: 在算子入口使用`at::native::result_type`推导`high_type`，对不匹配的输入做`npu_dtype_cast`转换后再传入ACL算子。
- Reviewability: high
- Review rule: 所有二元算子必须在入口处处理dtype提升，与PyTorch的`result_type`语义对齐。

### D-1237: json.load hook中arange和stack未正确wrap导致报错

- Root cause: 算子Hook注册遗漏
- Hashes: 254e2075a
- Files: torch_npu/hooks/wrap_torch.py
- Defect: `WrapTorchOps`列表中包含`arange`（已重复出处于`arccosh`附近）和`stack`，但这两个op在hook包装后会导致json.load等场景报错，因为它们在CPU上的行为被错误拦截。实际上这些op不需要被wrap。
- Fix: 从`WrapTorchOps`列表中移除`arange`（实为多余的`arccosh`附近行的错误）和`stack`。
- Reviewability: high
- Review rule: WrapTorchOps列表中的每个算子都需确认wrap后不会影响CPU侧的正常调用路径。

### D-1238: GradScaler在dynamic=False时未触发per_device_found_inf的lazy初始化

- Root cause: lazy初始化遗漏
- Hashes: 9c70fd645
- Files: torch_npu/npu/amp/grad_scaler.py
- Defect: NPU的`GradScaler`在`dynamic=False`时，当没有overflow时不会调用`per_device_found_inf.get(found_inf.device)`，但父类`Cuda_GradScaler`的后续校验（第337行）期望该device已在`per_device_found_inf`中注册。缺少这个get调用导致KeyError。
- Fix: 在else分支添加`per_device_found_inf.get(found_inf.device)`调用（仅触发注册，不加1）。
- Reviewability: medium
- Review rule: 覆盖父类方法时，需确保所有分支都满足父类的post-condition约束。

### D-1239: 多算子format继承导致NZ/5HD格式不匹配（revert d6fb78+a283f4+1af6a3）

- Root cause: NPU format继承错误
- Hashes: c5f54b8e8
- Files: AddKernelNpu.cpp, AddmmKernelNpu.cpp, GruKernelNpu.cpp, LstmKernelNpu.cpp, 及15个其他算子文件
- Defect: 之前的三个提交(d6fb78/a283f4/1af6a3)尝试让算子输出tensor"继承"输入tensor的NPU format（通过`CalcuOpUtil::get_tensor_npu_format`），但这导致了大量算子的format不匹配问题。例如：Add算子应该用`ACL_FORMAT_NC1HWC0`而非继承输入的任意format；GRU的中间tensor必须是`ACL_FORMAT_FRACTAL_NZ`；LSTM的gate输出也必须是NZ格式。通用的format继承策略在实践中不可行，因为不同算子对中间tensor的format有硬性要求。
- Fix: revert三个提交，将`CalcuOpUtil::get_tensor_npu_format(input)`替换回显式指定的`ACL_FORMAT_NC1HWC0`或`ACL_FORMAT_FRACTAL_NZ`等硬编码format。
- Reviewability: low
- Review rule: 算子输出tensor的format不能简单继承输入format，必须根据ACL算子的format约束显式指定。需要case-by-case审查。

### D-1240: rank<2的tensor不支持NZ格式导致mean等算子crash

- Root cause: NPU format与tensor rank不匹配
- Hashes: 58611cfdc
- Files: MeanKernelNpu.cpp, InferFormat.cpp
- Defect: `FRACTAL_NZ`格式要求tensor至少为2维，但mean等reduce算子在输出为scalar（0维）或1维时仍可能继承输入的NZ格式，导致ACL算子报错。`InferFormat::GuessStorageFormat`也未对rank<2的NZ格式做拦截。
- Fix: MeanKernelNpu中将`outputSize.empty()`的检查改为`outputSize.size() < 2`；`GuessStorageFormat`中对NZ格式且rank<2的情况回退到`ACL_FORMAT_ND`。
- Reviewability: high
- Review rule: 所有可能产生低维输出的算子，以及format推导的通用逻辑，必须检查rank与format的兼容性。

### D-1241: nll_loss_2d对ignore_index赋weight=0时越界访问

- Root cause: 边界条件检查不完整
- Hashes: cc1821587
- Files: NLLLoss2dBackwardKernelNpu.cpp, NLLLoss2dKernelNpu.cpp
- Defect: NLLLoss2d在处理`ignore_index`时，仅检查`ignore_index >= 0`就对weight_tensor做memcpy赋零。当`ignore_index >= self.size(-1)`时，索引越界导致内存越界写入。
- Fix: 将条件从`ignore_index >= 0`改为`ignore_index >= 0 && ignore_index < self.size(-1)`。
- Reviewability: high
- Review rule: 所有使用index做内存访问的场景必须检查上界。

### D-1242: DVPP接口buffer size计算和engine priority属性类型错误

- Root cause: ACL属性类型不匹配
- Hashes: 212e4f00d
- Files: OpCmdHelper.cpp, OpParamMaker.cpp
- Defect: 两个问题：(1) OpCmdHelper中对STRING类型tensor使用`storageDims`计算buffer size，但STRING tensor的storageDims为0，应使用`storage_sizes_`获取实际大小。(2) `SetEnginePriority`中`_performance_prior`属性传入`std::string("true")`而非`bool true`，`_exclude_engines`的值"AICORE"大小写错误（应为"AiCore"）且类型转换冗余。
- Fix: buffer size改用`npuDesc.storage_sizes_`计算；属性`_performance_prior`改为bool true；`_exclude_engines`改为正确的大小写和类型。
- Reviewability: medium
- Review rule: ACL属性的类型和值必须与算子注册信息一致，STRING tensor的size计算不能依赖storageDims。

### D-1243: BatchNorm强制format cast到5HD导致非5HD权重场景出错

- Root cause: NPU format强制转换错误
- Hashes: d44c3a69c
- Files: BatchNormBackwardKernelNpu.cpp, BatchNormKernelNpu.cpp
- Defect: BatchNorm前向和反向实现中，对weight/running_mean/running_var/bias无条件调用`npu_format_cast_(_, ACL_FORMAT_NC1HWC0)`转为5HD格式。当这些参数本身不是5HD格式时（例如通过特定优化器或自定义初始化创建），强制cast会导致数据错误或性能问题。
- Fix: 移除对已有参数的强制format cast，直接使用原始参数。只在参数未定义时创建默认值。
- Reviewability: medium
- Review rule: 算子不应对用户传入的tensor做破坏性的in-place format cast，应通过拷贝或让框架自动适配。

### D-1244: cast算子forward保存完整self tensor导致内存浪费

- Root cause: autograd保存策略不当
- Hashes: ea84a6622
- Files: CastKernelNpu.cpp
- Defect: `npu_dtype_cast`的autograd forward中`ctx->save_for_backward({self})`保存了完整的输入tensor，但backward只需要知道原始dtype即可。对于大tensor，这导致显著的NPU内存浪费。
- Fix: 创建一个大小为1的小tensor`temp_save_dtype`来记录原始dtype信息，用它替代完整的self进行`save_for_backward`。
- Reviewability: high
- Review rule: autograd保存的tensor应最小化，只保存backward真正需要的信息。

### D-1245: add算子不支持不同dtype输入

- Root cause: 类型推导错误
- Hashes: e679e0148
- Files: AddKernelNpu.cpp
- Defect: `add`算子在两个输入dtype不同时（如float32 + int64），直接传入ACL算子而未做类型提升。PyTorch CPU侧会自动通过`result_type`提升到公共类型。NPU侧缺少这个处理，导致结果dtype和数值与CPU不一致。
- Fix: 在`add`入口使用`at::native::result_type`推导`high_type`，对不匹配的输入做`npu_dtype_cast`，输出tensor也使用`high_type`。
- Reviewability: high
- Review rule: 同D-1228。所有二元算子必须处理dtype提升。

### D-1246: mul算子使用.to()拷贝而非npu_dtype_cast导致走CPU路径

- Root cause: dtype转换路径错误
- Hashes: c95df4bd0 [+1 cherry-pick: c35da010d]
- Files: MulKernelNpu.cpp
- Defect: `mul`算子中Bool tensor的类型转换使用`.to(c10::ScalarType::Float)`，该调用可能走PyTorch标准dispatch而非NPU路径，导致不必要的H2D/D2H拷贝。`mul(tensor, scalar)`分支还冗余地通过`ApplyTensorWithFormat`创建output tensor再传format，而`ApplyTensor`已足够。inplace版本中Bool分支的copy_逻辑也不正确。
- Fix: 将`.to()`替换为`NPUNativeFunctions::npu_dtype_cast`确保走NPU路径；`mul(tensor, scalar)`改用`ApplyTensor(self)`；inplace版本中区分Bool和非Bool的赋值方式。
- Reviewability: medium
- Review rule: NPU tensor的dtype转换统一使用`npu_dtype_cast`，禁止使用`.to(dtype)`。

### D-1247: __setstate__中DistributedDataParallel名称未定义

- Root cause: 类名引用错误
- Hashes: 434bce5cb
- Files: torch_npu/utils/module.py
- Defect: `ddp__setstate__`方法中调用`super(DistributedDataParallel, self).__setstate__(state)`，但`DistributedDataParallel`在当前作用域未import，导致NameError。该函数是作为monkey-patch注入的，self的类型在运行时确定。
- Fix: 改用`Module.__setstate__(self, state)`直接调用Module基类方法，同时修正import顺序。
- Reviewability: high
- Review rule: monkey-patch函数中不能引用未import的类名，应使用已import的基类。

### D-1248: adaptive_avg_pool3d未正确实现非(1,1,1)输出

- Root cause: 算子能力声明不完整
- Hashes: f6b260955
- Files: AdaptiveAvgPool3dKernelNpu.cpp
- Defect: `adaptive_avg_pool3d`的NPU实现调用`adaptive_avg_pool3d_out`但该函数内部未正确处理各种output_size组合。实际NPU只支持`(1,1,1)`的情况（等价于全局average pooling），对其他情况缺少明确报错。
- Fix: 对`output_size == (1,1,1)`的情况改用`at::mean`实现；对其他情况直接TORCH_CHECK报错说明不支持。
- Reviewability: high
- Review rule: 算子不完全支持标准接口的所有参数组合时，必须在入口处显式检查并报错。

### D-1249: index操作中的expand和dtype处理错误

- Root cause: index tensor的dtype和broadcast处理错误
- Hashes: 6f29462a9
- Files: AdvancedIndex.cpp
- Defect: `AdvanceIndex::make_info`使用标准库的`at::expand_outplace`做index tensor的broadcast，但NPU侧的Long类型index需要先转为Int才能expand（ACL对Long的broadcast支持不完整）。`reshape_indexer`也未处理非Long类型的index转换。
- Fix: 自定义`npu_expand_outplace`函数，在expand前将Long类型index转为Int；`reshape_indexer`中对非Long类型的index添加`.to(at::kLong)`转换。
- Reviewability: medium
- Review rule: NPU侧的index tensor处理需注意Long/Int类型转换和ACL的类型支持范围。

### D-1250: index_add不支持0维index tensor

- Root cause: 0维tensor边界条件缺失
- Hashes: 9fa200d68 [+1 cherry-pick: 9c14ce9fb]
- Files: IndexAddKernelNpu.cpp
- Defect: `index_add`中直接使用`index.sizes()[0]`获取index长度，但当index为0维标量tensor时，sizes()为空，导致越界访问。同时`pad_size[dim]`的计算也使用了原始index而非可能经过unsqueeze的indices。
- Fix: 当`index.dim() == 0`时调用`indices.unsqueeze_(0)`将标量提升为1维；`pad_size[dim]`改用处理后的`indices.sizes()[0]`。
- Reviewability: high
- Review rule: 所有使用`.sizes()[0]`的地方必须先检查tensor是否为0维。

### D-1251: _unique2算子fp16精度问题

- Root cause: 低精度dtype精度不足
- Hashes: 59974ffb1
- Files: _Unique2KernelNpu.cpp
- Defect: `_unique2`算子在fp16输入时直接操作半精度数据，ACL算子内部的比较和排序在fp16精度下会产生错误结果（相近的fp16值被误判为相同）。
- Fix: 在算子入口将fp16输入cast为fp32后执行，输出再cast回fp16。
- Reviewability: medium
- Review rule: 涉及比较、排序、唯一值操作的算子需评估fp16精度是否满足正确性要求。

### D-1252: Module.to()不支持"npu:x"格式的device字符串

- Root cause: device参数解析不完整
- Hashes: 6fa6a8272
- Files: torch_npu/utils/module.py
- Defect: `Module.to(device="npu:0")`中，kwargs的device检查使用`== 'npu'`精确匹配，无法匹配"npu:0"等带设备号的字符串。导致`to(device="npu:0")`时device参数未被转译。
- Fix: 将`kwargs.get("device", None) == 'npu'`改为`'npu' in kwargs.get("device", "")`，并使用`.replace("npu", torch_npu.npu.native_device)`处理替换。
- Reviewability: high
- Review rule: device字符串匹配必须用子串包含检查，不能用精确等于比较。

### D-1253: device6/device7无法barrier（deviceIdx计算错误）

- Root cause: 分布式device ID计算错误
- Hashes: dfd3398c6
- Files: ProcessGroupHCCL.cpp
- Defect: barrier在`usedDeviceIdxs_`为空时，使用`rank_ % numNPUs`计算deviceIdx并传入Device构造函数。但当使用device6、device7等高编号设备时，这个计算结果可能不正确。更根本的问题是，在无设备记录时不应指定设备编号，应让系统使用当前设备。
- Fix: 将`at::Device(NativeDeviceType, deviceIdx)`改为`at::Device(NativeDeviceType)`（不指定index），让运行时使用当前设备。
- Reviewability: high
- Review rule: 分布式代码中设备编号的计算必须考虑非连续device ID的场景。

### D-1254: as_tensor算子NPU设备支持不完整

- Root cause: 算子monkey-patch实现缺失
- Hashes: 68d1c05dd
- Files: scripts/codegen/templates/torch_funcs.py
- Defect: `torch.as_tensor(data, device="npu")`未被拦截处理，直接走标准`_VariableFunctions.as_tensor`，但标准实现不认识"npu"设备。对numpy ndarray和已有tensor两种输入都需要特殊处理才能创建NPU tensor。
- Fix: 添加`_as_tensor`包装函数：检测device是否包含"npu"，若是则对numpy用`from_numpy().to()`，对tensor用`.to()`转移到NPU设备。注册为`torch.as_tensor`的替换。
- Reviewability: high
- Review rule: 每个新增的torch API monkey-patch必须覆盖所有合法的device参数形式。

### D-1255: as_tensor算子实现逻辑重构（修复D-1246的后续问题）

- Root cause: 算子实现逻辑错误
- Hashes: 4618770dd
- Files: scripts/codegen/templates/torch_funcs.py
- Defect: D-1246的`_as_tensor`实现在non-npu设备时不调用标准实现，且对CPU tensor传npu device时未正确处理。当kwargs中无device时也应有默认行为。
- Fix: 重构为先提取device参数（默认"cpu"），然后总是先调用标准`as_tensor`（不传device），最后用`.to(dst_device)`搬运。
- Reviewability: medium
- Review rule: monkey-patch实现必须确保non-npu路径与原始行为完全一致。

### D-1256: avgpool2d不支持3维输入且count_include_pad处理错误

- Root cause: 输入维度适配缺失
- Hashes: a27ed0472
- Files: AvgPool2dKernelNpu.cpp, KernelNpuOutputSize.cpp
- Defect: `avg_pool2d`在输入为3维tensor时（缺少batch维度），直接传入ACL的AvgPoolV2算子，但该算子期望4维(NCHW)输入，导致shape计算错误和算子执行错误。此外，`exclusive`属性（对应PyTorch的`count_include_pad`）对fp32/fp64类型未正确传递。
- Fix: 3维输入时先`unsqueeze(0)`增加batch维度；`exclusive`属性对非fp16/int8类型改为使用`!count_include_pad`的实际值。
- Reviewability: medium
- Review rule: 池化算子必须处理3维输入（无batch维度）的情况。

### D-1257: rrelu_with_noise输出format错误

- Root cause: NPU format继承错误
- Hashes: 78285a1d9
- Files: RreluWithNoiseKernelNpu.cpp
- Defect: `rrelu_with_noise`的输出tensor format不正确，导致后续算子接收到错误format的tensor。同时缺少fp16的测试覆盖和inplace版本的验证。
- Fix: 修正输出tensor的format设置；补充fp16测试和inplace版本测试。
- Reviewability: medium
- Review rule: 激活函数算子必须确保输出format与输入一致，并覆盖所有支持dtype的测试。

### D-1258: reduce算子numel计算错误（使用physical_numel而非logical numel）

- Root cause: 通信算子numel计算错误
- Hashes: bad33087f
- Files: ProcessGroupHCCL.cpp
- Defect: `reduce`操作中使用`physical_numel(input)`获取元素数传给HCCL，但对于ND/NCHW格式的tensor，physical_numel可能大于logical numel（因为storage padding），导致reduce操作处理了错误数量的元素。
- Fix: 将`physical_numel(input)`替换为`getNumelForHCCL(input)`（该函数在eaf3cd42c中引入），对ND/NCHW格式使用logical numel。
- Reviewability: high
- Review rule: HCCL通信操作中的numel必须区分physical和logical，对标准格式使用logical numel。

### D-1259: upsample_bilinear2d_backward中存在不必要的AICPU路径分支

- Root cause: 冗余适配代码
- Hashes: 740309cf9
- Files: UpsampleBilinear2dBackwardKernelNpu.cpp
- Defect: `upsample_bilinear2d_backward`中根据H/W是否大于10000来选择AICORE或AICPU路径（`PTUpsampleBilinear2dGrad`），但AICPU路径的算子已不存在或不可用，且判断条件本身过于保守。实际`ResizeBilinearV2Grad`可以处理所有合理的size。
- Fix: 删除AICPU路径分支和`upsample_bilinear2d_backward_check_is_aicore`函数，统一使用`ResizeBilinearV2Grad`。
- Reviewability: high
- Review rule: 算子实现中的条件分支必须确保所有路径的目标算子均可用。

### D-1260: random_()的默认范围对多种dtype不正确

- Root cause: 随机数范围计算错误
- Hashes: 5744a20a5
- Files: RandomKernelNpu.cpp
- Defect: `random_()`（无参数版本）的默认范围对多种dtype不正确：float用`LONG_MAX`而非`2^24`（float精度上限），fp16用`NPU_HALF_MAX(65504)`而非`2^11`，且不支持uint8/int8/int16/double等类型。这导致生成的随机数超出类型精度范围或直接报错。
- Fix: 按PyTorch语义重新定义各dtype的上界：float用`1<<24`，double用`1<<53`，half用`1<<11`，整数类型用对应的类型MAX。添加uint8/int8/int16/double的支持。
- Reviewability: high
- Review rule: 随机数生成的默认范围必须与PyTorch CPU实现一致，特别是浮点类型的精度上界。

### D-1261: silu_backward注册在错误的dispatch层导致autograd失效

- Root cause: 算子注册位置错误
- Hashes: 8cc17f0f5
- Files: npu_native_functions.yaml
- Defect: `silu`、`silu_`、`silu.out`被注册在`supported`节（直接dispatch），而非`autograd`节。这导致silu_backward调用SwishGrad算子时，autograd的反向传播链断裂，backward计算不正确。
- Fix: 将silu相关三个算子从`supported`移到`autograd`节，确保反向传播正确注册。
- Reviewability: high
- Review rule: 有自定义backward的算子必须注册在`autograd`节而非`supported`节。

### D-1262: pad_packed_sequence中npu_transpose对非连续输入的限制

- Root cause: 算子API约束不匹配
- Hashes: 9dd29b3ea
- Files: PadPackedSequenceKernelNpu.cpp
- Defect: `_pad_packed_sequence`在`batch_first=True`时使用`NPUNativeFunctions::npu_transpose`做转置，但该函数要求输入满足特定的连续性约束（`require_contiguous=true`），对pad后的data可能不满足。
- Fix: 将`NPUNativeFunctions::npu_transpose(data, {0, 1}, true)`替换为标准的`data.transpose(0, 1)`，后者无连续性要求。
- Reviewability: high
- Review rule: 优先使用PyTorch标准API，仅在性能关键路径使用NPU自定义API。

### D-1263: CdistGrad对p=inf场景处理错误

- Root cause: 特殊值处理错误
- Hashes: 8960e7ba2
- Files: CdistBackwardKernelNpu.cpp
- Defect: `_cdist_backward`中，当`p = inf`时，原实现检查`p <= float::max()`会直接报错拒绝inf输入。移除该检查后，将inf转为`float::infinity()`传给ACL算子，但ACL的CdistGrad算子不接受inf参数。
- Fix: 移除`p <= float::max()`的错误检查；将`p = inf`映射为特殊值`-1`传给ACL算子（ACL约定-1表示无穷范数）。
- Reviewability: medium
- Review rule: 数学函数中的特殊值（inf、nan、0）必须映射到ACL算子支持的对应约定值。

### D-1264: as_strided的CompileType设置错误导致精度问题

- Root cause: ACL CompileType设置错误
- Hashes: 5b0cfc561
- Files: AsStridedKernelNpu.cpp
- Defect: `as_strided`算子将shape和stride参数的CompileType设为`MEMORY_HOST_COMPILE_INDEPENDENT`，这意味着编译器可能对这些值做优化假设（认为值不变）。但as_strided的shape/stride是运行时动态值，错误的CompileType导致图编译时使用了错误的shape/stride，产生精度问题。
- Fix: shape和stride改为不带CompileType的默认模式（让框架自动推断）；storage_offset改为`MEMORY_HOST_COMPILE_DEPENDENT`以确保正确编译。
- Reviewability: low
- Review rule: 运行时可变的tensor shape/stride/offset参数不能使用`COMPILE_INDEPENDENT`。

### D-1265: random和uniform的随机数生成器和seed处理重构

- Root cause: 随机数生成器使用错误
- Hashes: 6871ad926
- Files: RandomKernelNpu.cpp, UniformKernelNpu.cpp, OpCmdHelper.cpp, OpCommand.cpp, OpCommand.h
- Defect: random和uniform算子在获取随机种子时使用`at_npu::detail::getDefaultNPUGenerator()`而非用户传入的generator参数，导致手动设置seed后生成的随机数不可复现。同时OpCommand的seed传递链路也存在冗余代码。
- Fix: 使用`at::get_generator_or_default`正确处理用户传入的generator；简化seed获取和传递逻辑；补充fp16的seed复现测试。
- Reviewability: medium
- Review rule: 所有接受generator参数的随机算子必须用`get_generator_or_default`处理，不能忽略用户传入的generator。

### D-1266: syncbn_forward缺少F模块导入

- Root cause: Python import缺失
- Hashes: a4ebef8fe
- Files: torch_npu/utils/module.py
- Defect: SyncBatchNorm的forward路径中使用了`torch.nn.functional`（F）的功能，但module.py未import该模块，导致syncbn_forward运行时报NameError。
- Fix: 添加`import torch.nn.functional as F`。
- Reviewability: high
- Review rule: Python文件中使用的所有模块必须在文件头部显式import。

### D-1267: c10d barrier两个bug（empty tensor和backend名称错误）

- Root cause: 分布式barrier实现错误
- Hashes: 933c0f2d6
- Files: ProcessGroupHCCL.cpp, distributed_c10d.py
- Defect: 两个独立bug：(1) barrier中使用`at::empty({1})`创建barrierTensor，空tensor在HCCL allreduce中可能被优化跳过，导致barrier不生效。(2) Python侧barrier的`device_ids`参数检查使用`Backend.NCCL`而非`Backend.HCCL`，导致HCCL后端传入device_ids时报错。
- Fix: (1) 改`at::empty`为`at::ones`确保tensor有值；(2) 将`Backend.NCCL`改为`Backend.HCCL`。
- Reviewability: high
- Review rule: 从NCCL移植的分布式代码必须全文搜索替换NCCL引用为HCCL。

### D-1268: OneHot算子depth参数类型导致aicpu分支报错

- Root cause: 算子输入参数类型错误（连续两次修复）
- Hashes: fccd335bf, 7d8b95f64
- Files: OneHotKernelNpu.cpp
- Defect: (fccd335bf) OneHot算子将depth声明为int64_t，通过SmallVector传入ACL算子，但aicpu路径的OneHot期望Scalar类型的depth输入，导致大尺寸输入（走aicpu）时报错。(7d8b95f64) 修复fccd335bf后，depth改为at::Scalar，但Scalar通过标准Input路径传入时会创建device-side tensor，造成性能下降。
- Fix: (fccd335bf) int64_t改为at::Scalar，使用.Input(depth, at::kInt)传入。(7d8b95f64) 改回int64_t depth，但传入时构造临时Scalar并指定`CompileType::MEMORY_HOST_COMPILE_DEPENDENT`避免创建device tensor。
- Reviewability: medium
- Review rule: ACL算子的Scalar参数需评估aicore和aicpu两种路径的传入方式差异，优先使用COMPILE_DEPENDENT。

### D-1269: NmsRotated构建错误（ScalarType枚举和API调用错误）

- Root cause: 编译错误
- Hashes: 39242292c, 950be4667
- Files: NmsRotatedKernelNpu.cpp
- Defect: 连续两个编译错误：(39242292c) 使用`at::ScalarType::kInt`但正确的枚举值是`at::kInt`，同时`selectedIndex.item()`在tensor非标量时不能调用。(950be4667) 修复枚举后，Sync调用顺序需在Name之前，且`selectedIndex.item()`改为`selectedIndex.size()[0]`获取选中元素数。
- Fix: 先修枚举名称和API调用，再调整Sync/Name顺序和size获取方式。
- Reviewability: high
- Review rule: 编译前应检查枚举名称和API签名的正确性。

### D-1270: profiler内嵌函数定义导致return逻辑错误

- Root cause: Python作用域/控制流错误
- Hashes: be5873c2a
- Files: torch_npu/npu/profiler.py
- Defect: `EventList._build_tree`中，事件树构建逻辑被包装在内嵌函数`_process_event()`中，该函数内的`return`只退出内嵌函数而非外层循环。这导致事件被错误地同时添加为child和parent，parent-child关系构建不正确。
- Fix: 将内嵌函数的逻辑内联到外层循环中，`return`改为`break`以正确控制flow。
- Reviewability: medium
- Review rule: 内嵌函数中的return/break语义与外层不同，代码审查时需特别关注内嵌函数的控制流意图。

### D-1271: 共享storage的tensor在HCCL通信中numel计算错误

- Root cause: 通信算子numel计算错误
- Hashes: eaf3cd42c
- Files: ProcessGroupHCCL.cpp
- Defect: 两个tensor共享同一storage（如通过view/slice创建）时，使用`physical_numel`获取的是整个storage的元素数而非tensor自身的logical numel。在HCCL的allreduce/broadcast/allgather/reduce_scatter/send中，传入错误的numel导致通信数据量错误。ND/NCHW格式的tensor应使用logical numel，而非物理存储的numel。
- Fix: 引入`getNumelForHCCL`函数，对ND/NCHW格式返回`self.numel()`（logical），对其他format返回`physical_numel`并检查data_ptr一致性。替换所有HCCL调用点的numel计算。
- Reviewability: medium
- Review rule: HCCL通信中的numel必须考虑tensor view/slice场景，ND格式应使用logical numel。

### D-1272: cast_weight对"npu"设备名匹配不完整

- Root cause: device参数解析不完整
- Hashes: e86c20210
- Files: torch_npu/utils/module.py
- Defect: `cast_weight`方法中通过`torch_npu.npu.native_device not in str(device)`判断是否为NPU设备，但用户可能通过`module.npu()`传入（此时device字符串包含的是"npu"而非native_device名），导致权重format cast未执行。
- Fix: 将检查改为同时匹配`native_device`和`npu_device`两种设备名字符串。
- Reviewability: high
- Review rule: device类型检查必须覆盖"npu"和native_device两种名称形式。

### D-1273: torch.device('npu').index为None导致后续操作报错

- Root cause: device对象构造不完整
- Hashes: b57324fa0
- Files: scripts/codegen/templates/torch_funcs.py
- Defect: `torch.device('npu')`（不带设备号）创建的device对象`index`为None。在`_device`函数中将"npu"转为native_device后构造新device时，None index被传入`device(type=npu_device, index=None)`，后续需要设备号的操作会报错。
- Fix: 当`new_device.index`为None/falsy时，默认设置为0。
- Reviewability: high
- Review rule: device构造时index为None的情况必须有默认值处理。

### D-1274: index_put中undefined但has_value的index tensor导致mask计算错误

- Root cause: 可选tensor状态判断不完整
- Hashes: a5b2c332f
- Files: IndexPutKernelNpu.cpp
- Defect: `_index_put_impl_`遍历indices时，只检查`has_value()`但未检查`defined()`。PyTorch中`c10::optional<Tensor>`可以has_value为true但内部tensor为undefined状态。将undefined tensor加入`allDefinedIndices`并标记mask为1，导致ACL算子收到空tensor作为index。
- Fix: 在has_value检查后增加`index.defined()`检查，undefined的index标记mask为0。
- Reviewability: high
- Review rule: 处理`optional<Tensor>`时必须同时检查`has_value()`和`defined()`。
### D-1275: tensor.new_empty多维参数解析错误

- Root cause: 参数打包逻辑缺陷
- Hashes: 8ae3d43be
- Files: torch_npu/utils/tensor_methods.py
- Defect: `_new_empty`仅将第一个int参数打包为tuple，多维调用如`t.new_empty(2, 3)`时只传`(2,)`给底层C函数，丢失后续维度。
- Fix: 遍历所有连续int参数收集到sizes列表中，合并为一个tuple。
- Reviewability: high
- Review rule: 当函数接受`*args`且需要从中提取size tuple时，必须处理任意长度的连续int序列。

### D-1276: index_put对optional indices的has_value检查缺失

- Root cause: API语义不匹配(defined vs has_value)
- Hashes: 2f17fe4e0
- Files: torch_npu/csrc/aten/ops/IndexPutKernelNpu.cpp
- Defect: `_index_put_impl_`应使用`has_value()`检查optional tensor，而非`defined()`。
- Fix: 改用`has_value()`判断optional tensor。
- Reviewability: high
- Review rule: PyTorch optional tensor索引必须用`has_value()`而非`defined()`检查。

### D-1277: Conv2d输出format受fuzzy compile开关影响导致性能下降

- Root cause: NPU format选择错误
- Hashes: 6a35c276a
- Files: pytorch1.5.0/src/aten/src/ATen/native/npu/convolution/Conv2dKernelNpu.cpp
- Defect: Conv2d输出tensor的format根据`isFuzzyCompile`开关在NCHW和NC1HWC0之间切换，导致不必要的格式转换。
- Fix: 移除fuzzy compile条件判断，硬编码输出format为ACL_FORMAT_NC1HWC0。
- Reviewability: high
- Review rule: 算子输出format不应受编译模式开关影响。

### D-1278: torch.save序列化module时直接修改原始对象

- Root cause: 可变对象副作用(序列化系列修复4个)
- Hashes: ec91fd265, c5e60f832, 920bafb35, 8420ab4c8
- Files: torch_npu/utils/serialization.py
- Defect: 四个commit逐步修复同一问题链: to_cpu()直接修改传入的dict/list对象，save后原始对象被改变。根本原因是序列化路径忽视了`.cpu()`操作的in-place语义。
- Fix: 从in-place修改演进到创建副本，最终采用`copy.deepcopy(obj).cpu()`。
- Reviewability: medium
- Review rule: 序列化函数不得修改传入对象的状态。需要类型转换时，必须先深拷贝。

### D-1279: all算子空Tensor输出shape错误

- Root cause: 空Tensor边界条件处理
- Hashes: 54e803545
- Files: torch_npu/csrc/aten/ops/AllKernelNpu.cpp
- Defect: `all(dim)`在输入numel为0时，直接创建空shape `{}`的结果tensor。正确行为应该是去掉被reduce的维度后保留其余维度。
- Fix: 遍历所有维度，跳过被reduce的dim，构造正确的outputSize。
- Reviewability: high
- Review rule: reduce类算子处理空tensor时，输出shape必须按dim参数正确计算。

### D-1280: torch.device解析未正确拦截NPU设备字符串

- Root cause: device参数解析体系缺陷(系列修复6个)
- Hashes: 9f3d065e9, d0184a573, 978896426, 3b7a1886a, 5c5a1aeae, d0094263d
- Files: scripts/codegen/gen_python_functions.py, scripts/codegen/templates/torch_funcs.py等
- Defect: torch_npu的device解析链多处缺陷，涉及device字符串拦截、PythonArgParser解析顺序、out变体遗漏、*_like算子device参数等6个子问题。
- Fix: 在codegen和手写代码的所有device解析路径中，统一使用parse_npu_device/torch_device_guard拦截。
- Reviewability: low
- Review rule: 插件式device扩展中，所有能接受device参数的API入口都必须经过统一的device解析拦截器。

### D-1281: divide/floor_divide对int32输入的精度错误

- Root cause: 算子输入类型提升缺失
- Hashes: 537c69084, 90709ed4f
- Files: torch_npu/csrc/aten/ops/FloorDivideKernelNpu.cpp
- Defect: floor_divide/divide算子在输入为int32类型时直接送入NPU计算，应先提升为float。
- Fix: 检测输入为Bool或Int时，先cast到Float再做除法运算。
- Reviewability: high
- Review rule: NPU算子若不支持某dtype，必须在wrapper层做显式类型提升。

### D-1282: HCCL stub函数名与实际API不匹配

- Root cause: API命名不一致
- Hashes: 52fc7825c
- Files: third_party/acl/libs/hccl.cpp, third_party/acl/libs/hccl.h
- Defect: stub代码中函数名使用小写驼峰(hcclCommInitUniqueId)，实际API已改为大写驼峰(HcclCommInitUniqueId)。
- Fix: 将所有stub函数名从`hccl*`改为`Hccl*`。
- Reviewability: high
- Review rule: stub/mock代码的函数签名必须与真实库头文件完全一致。

### D-1283: TaskQueue的WriteQueue锁位置错误导致竞态

- Root cause: 并发锁粒度不当
- Hashes: 853fed5ef
- Files: torch_npu/csrc/core/npu/NPUQueue.cpp
- Defect: `WriteQueue`先检查`IsFullQueue()`再获取锁，多线程场景下存在竞态。
- Fix: 将lock_guard移到函数最开始，在fullQueue检查之前获取锁。
- Reviewability: high
- Review rule: 对共享状态的check-then-act操作必须在同一把锁的保护下完成。

### D-1284: is_npu缺少@property装饰器导致DDP设备检测失败

- Root cause: Python属性装饰器遗漏
- Hashes: ee34cf3cc
- Files: torch_npu/utils/tensor_methods.py, torch_npu/utils/module.py
- Defect: `_is_npu`函数缺少`@property`装饰器，作为属性访问`tensor.is_npu`时返回函数对象而非布尔值。
- Fix: 给`_is_npu`添加`@property`装饰器。
- Reviewability: high
- Review rule: monkey-patch替换tensor属性时，必须保持原始属性的类型一致。

### D-1285: Generator编译路径错误导致链接失败

- Root cause: 文件路径/构建系统错误
- Hashes: 78bc569b9
- Files: torch_npu/csrc/InitNpuBindings.cpp, torch_npu/csrc/npu/Generator.cpp
- Defect: Generator.cpp被移到新目录后include路径未更新。
- Fix: 更新include路径并修复Generator构造函数参数格式。
- Reviewability: high
- Review rule: 移动源文件后必须全局搜索更新所有include/import路径。

### D-1286: CANN profiler API不可用且缺少iteration接口导出

- Root cause: 模块导出遗漏
- Hashes: 659c6f794
- Files: torch_npu/npu/__init__.py, torch_npu/npu/npu_frontend_enhance.py
- Defect: `iteration_start`和`iteration_end`未被导入到`torch_npu.npu`命名空间。
- Fix: 在`__init__.py`中添加导入和`__all__`导出。
- Reviewability: high
- Review rule: 新增公共API函数后，必须同步更新模块导入和__all__列表。

### D-1287: HalfTensor类型未注册和导出

- Root cause: 类型注册遗漏
- Hashes: 800aaec50
- Files: torch_npu/npu/__init__.py, torch_npu/npu/tensor.py
- Defect: NpuTensorDict遗漏了half类型注册，无法使用`torch.npu.HalfTensor()`。
- Fix: 添加`"half": torch.HalfTensor`映射。
- Reviewability: high
- Review rule: 添加dtype支持时，必须同步更新类型注册字典。

### D-1288: NPU初始化时设备索引切换检查过于严格

- Root cause: 初始化逻辑过度约束(修复->回退)
- Hashes: bddd26b75, a31db9389
- Files: torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp
- Defect: 设备索引不一致检查过于严格，合法的多设备场景也会报错。
- Fix: 移除设备索引不一致时抛异常的逻辑。
- Reviewability: medium
- Review rule: 多设备场景下初始化函数应支持多次调用且不同设备索引。

### D-1289: tensor.to("cuda:0")在NPU环境未映射为NPU设备

- Root cause: device类型映射遗漏
- Hashes: d0094263d
- Files: torch_npu/csrc/utils/TensorMethods.h
- Defect: "cuda:N" device字符串未映射为npu设备。
- Fix: 检查device为cuda时替换为NPU类型。
- Reviewability: high
- Review rule: NPU插件中所有device解析路径必须处理cuda->npu映射。

### D-1290: tensor.npu(index)将整数索引解析为cuda设备

- Root cause: device类型映射遗漏
- Hashes: 5c5a1aeae
- Files: torch_npu/csrc/utils/TensorMethods.h
- Defect: `THPVariable_npu`中整数索引被解析为默认(cuda)设备类型。
- Fix: 构造`c10::Device(NPU, local_device.index())`。
- Reviewability: high
- Review rule: `.npu()`方法内必须强制替换device type为NPU。

### D-1291: randperm算子缺少n属性传递

- Root cause: 算子属性传递遗漏
- Hashes: a96623a5e
- Files: torch_npu/csrc/aten/ops/RandpermKernelNpu.cpp
- Defect: `randperm_out`调用NPU算子时未传递`.Attr("n", n)`属性。
- Fix: 添加`.Attr("n", n)`。
- Reviewability: high
- Review rule: 所有语义关键参数必须通过Attr/Input传给底层算子。

### D-1292: scalar_tensor在NPU设备上未正确实现

- Root cause: 算子实现缺失
- Hashes: 35fc19e4d
- Files: torch_npu/csrc/aten/common/TensorFactories.cpp
- Defect: `scalar_tensor`未被NPU注册实现，走CPU fallback返回CPU tensor。
- Fix: 注册NPU原生实现。
- Reviewability: high
- Review rule: 工厂函数在NPU后端必须逐个注册。

### D-1293: OpCommandImpl::Run持有GIL导致NPU算子编译阻塞Python线程

- Root cause: GIL持有导致性能阻塞
- Hashes: 9d520d824
- Files: torch_npu/csrc/framework/OpParamMaker.cpp
- Defect: `Run()`调用`InnerRun()`时持有GIL，NPU算子编译耗时期间阻塞所有Python线程。
- Fix: 在InnerRun期间释放GIL。
- Reviewability: high
- Review rule: C++扩展中耗时的NPU/ACL调用必须在GIL释放状态下执行。

### D-1294: SoftmaxBackward 5HD format条件判断反转

- Root cause: NPU format条件逻辑反转
- Hashes: d03a26c95, 28b116fe9
- Files: torch_npu/csrc/aten/ops/SoftmaxBackwardKernelNpu.cpp
- Defect: format比较条件`!=`应为`==`，导致format cast方向反转。
- Fix: 将条件从`!=`改为`==`。
- Reviewability: medium
- Review rule: NPU format转换的条件判断必须有明确注释说明转换方向。

### D-1295: LayerNormEval直接修改weight/bias的requires_grad属性

- Root cause: requires_grad属性副作用
- Hashes: 8c5338ce2
- Files: torch_npu/csrc/aten/ops/LayerNormEvalKernelNpu.cpp
- Defect: `npu_layer_norm_eval`对weight和bias调用`requires_grad_(false)`永久修改外部tensor。
- Fix: 改用`weight.detach().clone()`。
- Reviewability: high
- Review rule: 算子实现内部不得修改输入tensor的metadata。

### D-1296: get_device_index不支持list类型和模块未注册

- Root cause: 类型处理不全/模块注册遗漏
- Hashes: e910896bc, ca685b4c3
- Files: torch_npu/npu/utils.py, torch_npu/__init__.py
- Defect: `_get_device_index`不处理list类型输入；DDP使用的版本未被monkey-patch替换。
- Fix: 初始化device_idx，用isinstance(device, int)替换else；添加monkey-patch映射。
- Reviewability: high
- Review rule: 设备工具函数必须对所有合法输入类型有明确处理路径。

### D-1297: logical_and和mul算子对标量/非bool输入处理错误

- Root cause: 算子输入类型和维度处理缺陷
- Hashes: 2b4ea84e2
- Files: torch_npu/csrc/aten/ops/LogicalAndKernelNpu.cpp, torch_npu/csrc/aten/ops/MulKernelNpu.cpp
- Defect: logical_and不支持0-dim tensor输入，非bool类型输入未转换为bool。
- Fix: 0-dim输入转标量路径；非bool输入先cast到kBool。
- Reviewability: medium
- Review rule: 逻辑运算算子必须处理标量输入和非bool dtype输入。

### D-1298: mv算子对非连续输入使用错误的Input接口

- Root cause: 算子输入contiguous处理接口错误
- Hashes: a60f2a260
- Files: torch_npu/csrc/aten/ops/MvKernelNpu.cpp
- Defect: `Input`会做额外的contiguous处理使transpose失效。
- Fix: 改为`InputWithoutContiguousGeneral`。
- Reviewability: high
- Review rule: 已处理transpose逻辑的算子应使用InputWithoutContiguous*。

### D-1299: take算子ApplyTensor使用错误的参考tensor

- Root cause: 输出tensor创建参数错误
- Hashes: d6274b4c8
- Files: torch_npu/csrc/aten/ops/TakeKernelNpu.cpp
- Defect: 结果tensor的dtype/device应跟随self而非index。
- Fix: 使用`ApplyTensor(self, index.sizes())`。
- Reviewability: high
- Review rule: 输出tensor的dtype/device应追随数据源tensor。

### D-1300: atan2_out变量遮蔽导致结果未写入out参数

- Root cause: 变量遮蔽(variable shadowing)
- Hashes: f162523c2
- Files: torch_npu/csrc/aten/ops/Atan2KernelNpu.cpp
- Defect: 局部变量result遮蔽了外层参数result，计算结果未正确写入out。
- Fix: 去掉局部声明。
- Reviewability: high
- Review rule: `*_out`函数中不得声明与out参数同名的局部变量。

### D-1301: as_strided offset为0的假设导致非连续copy错误

- Root cause: storage offset处理缺陷
- Hashes: 7c31404fc
- Files: torch_npu/csrc/aten/ops/AsStridedKernelNpu.cpp等
- Defect: trans-contiguous路径要求MetaDataAreMatch，但as_strided的view带有非零offset。
- Fix: 移除MetaDataAreMatch检查，允许带offset的tensor走优化路径。
- Reviewability: medium
- Review rule: trans-contiguous优化路径不应因storage offset非零就放弃优化。

### D-1302: Module的cast_weight对非parameter的weight属性误操作

- Root cause: 参数检查不充分
- Hashes: f54d20ac1
- Files: torch_npu/utils/module.py
- Defect: cast_weight未确认weight是parameter而非buffer。
- Fix: 增加`"weight" in dict(module.named_parameters())`检查。
- Reviewability: high
- Review rule: 对module的weight做格式转换前必须确认其为parameter。

### D-1303: elu算子backward未注册为autograd函数

- Root cause: 自定义autograd注册缺失
- Hashes: 972d02ed7
- Files: torch_npu/csrc/aten/ops/EluKernelNpu.cpp
- Defect: elu的backward被注册在supported段而非autograd段。
- Fix: 移到autograd段，用自定义Function保存前向输出。
- Reviewability: medium
- Review rule: 反向算子依赖前向输出时必须通过autograd::Function注册。

### D-1304: 动态shape场景profiler默认开启TRAINING_TRACE导致E2E错误

- Root cause: profiler默认配置错误
- Hashes: 23b050f6e
- Files: torch_npu/npu/npu_frontend_enhance.py
- Defect: ACL_PROF_TRAINING_TRACE默认True与动态shape的图编译流程冲突。
- Fix: 默认值改为False。
- Reviewability: high
- Review rule: profiler配置默认值应取最安全选项。

### D-1305: distributed bucket size计算未考虑NPU 5HD format

- Root cause: NPU format下的内存大小计算错误
- Hashes: af6f2cbbe, 6b35e4b88
- Files: torch_npu/csrc/distributed/reducer.cpp
- Defect: DDP的bucket size使用`numel() * element_size()`计算，未考虑5HD padding。
- Fix: 使用`physical_numel() * element_size()`。
- Reviewability: medium
- Review rule: NPU环境下tensor内存大小计算必须用物理numel。

### D-1306: baddbmm算子调用at::add_out参数顺序错误

- Root cause: API调用参数顺序错误
- Hashes: 698226843
- Files: torch_npu/csrc/aten/ops/BaddbmmKernelNpu.cpp
- Defect: 使用`at::add_out`而非`NPUNativeFunctions::add_out`。
- Fix: 改用NPU原生实现。
- Reviewability: high
- Review rule: NPU算子wrapper内部必须使用NPUNativeFunctions::前缀。

### D-1307: dropout_with_add_softmax autograd保存冗余tensor导致内存泄漏

- Root cause: autograd上下文内存泄漏
- Hashes: e98a02cf8
- Files: torch_npu/csrc/aten/ops/DropoutWithAddSoftmaxKernelNpu.cpp
- Defect: save_for_backward保存了不需要的前向输入，同时saved_data又保存了前向输出。
- Fix: 只保存backward实际需要的tensor。
- Reviewability: high
- Review rule: save_for_backward只保存backward实际使用的tensor。

### D-1308: release_process_group函数未导出且_npu_shutdown引用错误模块

- Root cause: 模块导出和引用路径错误
- Hashes: 4342e86fa
- Files: torch_npu/__init__.py, torch_npu/distributed/__init__.py
- Defect: 使用torch.npu而非torch_npu.npu访问功能；release_process_group未导出。
- Fix: 替换为torch_npu.xxx路径；添加导出。
- Reviewability: high
- Review rule: 插件内部代码应直接引用自身模块。

### D-1309: batch_nms输出dtype硬编码为fp16

- Root cause: 输出dtype硬编码
- Hashes: 557b9dc9c
- Files: torch_npu/csrc/aten/ops/BatchNMSKernelNpu.cpp
- Defect: 输出始终为fp16，不管输入dtype。
- Fix: 使用`self.options()`让输出dtype跟随输入。
- Reviewability: high
- Review rule: 算子输出dtype应跟随输入。

### D-1310: set_aoe创建目录后未设置选项

- Root cause: 控制流逻辑错误
- Hashes: a3ca88990
- Files: torch_npu/npu/npu_frontend_enhance.py
- Defect: else分支中只创建目录但未执行setOption。
- Fix: 重构为无论如何都执行setOption。
- Reviewability: high
- Review rule: 确保所有路径都执行必要的核心操作。

### D-1311: BNTrainingUpdate不支持非fp32的running_mean/var/weight

- Root cause: 算子输入dtype约束未满足
- Hashes: e37609fdc
- Files: torch_npu/csrc/aten/ops/normalization/BatchNormKernelNpu.cpp
- Defect: BNTrainingUpdate要求fp32参数，但AMP混合精度训练可能传入fp16。
- Fix: 非fp32时先cast到fp32。
- Reviewability: high
- Review rule: 调用NPU算子前须检查其对输入dtype的约束。

### D-1312: upsample nearest3d/trilinear3d的fp16输入处理和尺寸计算错误

- Root cause: 算子尺寸计算和类型处理错误
- Hashes: c9806c139
- Files: torch_npu/csrc/aten/ops/UpSampleNearest3dKernelNpu.cpp等
- Defect: output size计算不正确，fp16输入未提升到fp32。
- Fix: 统一计算输出尺寸，fp16先cast到fp32。
- Reviewability: medium
- Review rule: upsample类算子须覆盖size和scales两种参数化方式。

### D-1313: addcdiv算子精度问题

- Root cause: 算子精度不足
- Hashes: 72da3c847
- Files: torch_npu/csrc/aten/ops/AddcdivKernelNpu.cpp
- Defect: addcdiv在NPU上精度不足。
- Fix: 调整内部计算实现。
- Reviewability: medium
- Review rule: 涉及除法的复合算子需关注数值精度。

### D-1314: NormKernel算子重写修复多个功能缺陷

- Root cause: 算子实现不完整
- Hashes: d5e5911a7
- Files: torch_npu/csrc/aten/ops/NormKernelNpu.cpp, scripts/codegen/gen.py
- Defect: p值未处理inf映射；非fp32输入未做类型提升；dtype_out和out变体未区分；FileManager fd泄漏。
- Fix: 重写NormKernel并修复FileManager。
- Reviewability: medium
- Review rule: reduce类算子须覆盖dtype_out变体。

### D-1315: inverse算子out接口不支持fp16

- Root cause: 算子out接口dtype处理缺陷
- Hashes: c88c9e303
- Files: torch_npu/csrc/aten/ops/InverseKernelNpu.cpp
- Defect: inverse_out中result为fp16但算子写入fp32数据。
- Fix: fp16 result先创建fp32临时buffer再copy回。
- Reviewability: high
- Review rule: `*_out`接口的result tensor dtype可能与算子期望不同，须做适配。

### D-1316: Graph模式下使用nbytes()而非实际capacity

- Root cause: Graph模式内存计算错误
- Hashes: cd86ed8fe
- Files: torch_npu/csrc/framework/graph/construct/GraphConstructor.cpp等
- Defect: nbytes()未考虑5HD format padding。
- Fix: 使用GetTensorCapacity替代nbytes()。
- Reviewability: medium
- Review rule: Graph模式下tensor内存大小必须用考虑NPU format padding的capacity。

### D-1317: NLLLoss算子dtype cast和grad_input format错误

- Root cause: 算子dtype/format处理缺陷
- Hashes: e78fe9ef5
- Files: torch_npu/csrc/aten/ops/loss/NLLLossKernelNpu.cpp等
- Defect: target用.to()做CPU端转换；grad_input未保持正确NPU format。
- Fix: 改用npu_dtype_cast；用ApplyTensorWithFormat保持format。
- Reviewability: medium
- Review rule: dtype转换用npu_dtype_cast；output tensor保持正确NPU format。

## Summary: Root Cause Categories

| Category                    | Count | Examples          | Description                                          |
|:----------------------------|------:|:------------------|:-----------------------------------------------------|
| dtype/类型处理缺失          |    81 | D-10, D-17, D-109 | dtype分支缺失、类型提升错误、Bool/int转换遗漏        |
| upstream同步缺失            |    65 | D-22, D-39, D-47  | PyTorch/CANN upstream API变更后NPU侧未跟进           |
| 图追踪/FakeTensor缺陷      |    45 | D-6, D-12, D-18   | dynamo trace、graph capture、sympy符号推导            |
| 边界条件/枚举遗漏           |    43 | D-1, D-5, D-7     | size=0、None守卫、枚举分支、空Tensor等边界未覆盖      |
| API适配/参数不匹配          |    38 | D-15, D-27, D-36  | 函数签名变更未适配、参数透传遗漏、版本守卫缺失        |
| monkey-patch缺陷            |    36 | D-16, D-45, D-102 | from-import切断引用链、模块级副作用、补丁粒度过粗     |
| 逻辑/控制流错误             |    36 | D-37, D-66, D-105 | 条件取反、early return遗漏、布尔判断歧义              |
| 内存/资源管理错误           |    36 | D-41, D-65, D-70  | ACL资源泄漏、use-after-free、析构顺序                 |
| 测试代码缺陷                |    32 | D-40, D-68, D-100 | 测试约束过窄、硬编码world_size/路径、用例误删         |
| 拼写/变量名错误             |    30 | D-20, D-71, D-143 | 变量名typo、dict key typo、日志前缀冗余               |
| 输出tensor构造错误          |    29 | D-43, D-72, D-92  | output format/stride/shape构造错误、NPU format丢失    |
| API导出/注册遗漏            |    27 | D-28, D-46, D-59  | dispatch key缺失、op未注册、符号可见性标记遗漏        |
| 锁/并发/同步错误            |    27 | D-57, D-144, D-284| mutex-device-sync死锁、竞态、GIL冲突                  |
| 构建/配置/CI错误            |    22 | D-4, D-24, D-103  | CMake配置、CI版本漂移、平台编译flag                   |
| profiler/调试工具缺陷       |    22 | D-21, D-42, D-62  | profiler断言覆盖不全、event语义错误                   |
| NPU硬件/设备兼容性          |    21 | D-8, D-9, D-34    | SoC行为差异、compatible mode路径                      |
| 初始化顺序/依赖错误         |    20 | D-3, D-13, D-99   | 全局初始化时序、import路径断链、循环依赖              |
| 异常/错误处理缺陷           |    20 | D-29, D-32, D-81  | catch-all吞异常、错误码未检查、异步错误传播断链       |
| 代码质量/重复/冗余          |    19 | D-147, D-193, D-279| copy-paste重复、dead code                            |
| 算子实现错误                |    17 | D-11, D-359, D-498| 数学公式错误、numel计算、语义理解偏差                 |
| 修复引入回归/revert         |    15 | D-2, D-38, D-53   | fix引入新bug、workaround残留、fix-revert cycle        |
| 状态管理缺陷                |    10 | D-223, D-224, D-242| 多处状态不一致、状态机终态未清理                      |
| 序列化/pickle路径缺陷       |     7 | D-49, D-260, D-277| closure无法pickle、序列化分支遗漏、device信息丢失     |
| 其他(唯一描述)              |   616 | --                 | 无法归入上述类别的高度特化defect                      |

Reviewability distribution: high 756 (58%) / medium 451 (34%) / low 106 (8%)

Top systemic defect clusters (cross-cutting patterns spanning multiple categories):

1. device参数解析体系缺陷(20+条): torch_npu的device解析链有系统性遗漏，
   涉及"npu:x"格式、None值、cuda->npu映射、PythonArgParser顺序、*_like变体等。
2. NPU format(5HD/NZ)与算子约束的系统性冲突(15+条):
   format继承不能通用化、rank<2不能用NZ、物理numel vs 逻辑numel。
3. dtype提升规则遗漏(12+条): add/mul/max/div等二元算子缺少result_type处理。
4. 资源释放顺序的多次修复-回退循环(8条):
   NPU shutdown时stream/event/ACL单例析构顺序至少经历3轮fix-revert。
5. Python falsy值陷阱(多条): device.index=0误判为False, None误做字符串操作。
6. 序列化副作用链(4条): torch.save经历4次修复才用deepcopy解决。
