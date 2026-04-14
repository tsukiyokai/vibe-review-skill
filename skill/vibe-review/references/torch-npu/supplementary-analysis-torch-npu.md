# torch_npu Defect Hotspot & REVERT Deep-Dive

Generated: 2026-04-12
Repository: torch_npu (Ascend PyTorch Adapter)
Data source: git-history.md (28,222 commits, keyword-classified)

## Part A: Defect Hotspot Files

### A.1 Methodology

- 从 classified_commits.tsv 提取 BUGFIX 类别的 commit
- 按 commit subject 去重 (同一修复 cherry-pick 到多条版本分支只计一次)
- 去重后 2,184 个独立 bug fix (原始 3,401 条含跨分支副本)
- 对每个去重 hash 执行 `git diff-tree --name-only` 提取涉及文件
- 按文件被 bugfix 触及的次数降序排列
- 对 Top 20 代码文件做当前版本的结构风险评估

排除的非代码文件 (频繁变更但非结构缺陷):

| Touches | File                                            | Reason         |
|--------:|:------------------------------------------------|:---------------|
|      64 | test/torch_npu_schema.json                      | op schema 同步 |
|      47 | test/unsupported_test_cases/..disabled..json     | 测试列表维护   |
|      41 | torch_npu/csrc/aten/npu_native_functions.yaml   | op 注册表      |
|      38 | third_party/acl/inc/acl/acl_rt.h                | 第三方头文件   |


### A.2 Top 20 Hotspot Ranking

Bug/100L = 去重 bugfix 触及次数 / 当前代码行数 x 100, 衡量缺陷密度。

| Rank | Bugfix | Lines | Bug/100L | File                                   |
|-----:|-------:|------:|---------:|:---------------------------------------|
|    1 |     91 |  7147 |     1.27 | csrc/distributed/ProcessGroupHCCL.cpp  |
|    2 |     84 |  3909 |     2.15 | csrc/core/npu/NPUCachingAllocator.cpp  |
|    3 |     63 |   404 |    15.59 | csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp|
|    4 |     57 |   203 |    28.08 | profiler/dynamic_profile.py            |
|    5 |     57 |  2522 |     2.26 | csrc/npu/Module.cpp                    |
|    6 |     54 |  1801 |     3.00 | csrc/core/npu/interface/AclInterface.cpp|
|    7 |     48 |   982 |     4.89 | csrc/core/npu/NPUQueue.cpp             |
|    8 |     47 |   335 |    14.03 | __init__.py                            |
|    9 |     44 |   524 |     8.40 | utils/_module.py (was utils/module.py) |
|   10 |     42 |   384 |    10.94 | profiler/...config_context.py          |
|   11 |     42 |   504 |     8.33 | contrib/transfer_to_npu.py             |
|   12 |     37 |   347 |    10.66 | profiler/profiler.py                   |
|   13 |     36 |   388 |     9.28 | csrc/core/npu/interface/AclInterface.h |
|   14 |     33 |   239 |    13.81 | profiler/.../_cann_file_parser.py      |
|   15 |     33 |   147 |    22.45 | profiler/analysis/_profiling_parser.py |
|   16 |     33 |  1970 |     1.68 | _inductor/codegen/triton.py            |
|   17 |     31 |   494 |     6.28 | npu/utils.py                           |
|   18 |     31 |   795 |     3.90 | npu/__init__.py                        |
|   19 |     31 |   769 |     4.03 | csrc/framework/OpParamMaker.cpp        |
|   20 |     30 |   345 |     8.70 | csrc/core/npu/NPUException.h           |

路径前缀均为 `torch_npu/`。#9 和 #14 曾被重命名 (加下划线前缀,
API 私有化重构)。

缺陷密度 Top 5: dynamic_profile.py (28.08) > _profiling_parser.py
(22.45) > npu_sys_ctrl.cpp (15.59) > __init__.py (14.03) >
_cann_file_parser.py (13.81)。Profiler 子系统包揽前两位。


### A.3 Module Cluster Analysis

Top 20 按子系统聚类:

| Cluster            | Files | Total | Representative risk            |
|:-------------------|------:|------:|:-------------------------------|
| Profiler           |     5 |   202 | 状态机无原子性, 异常吞没       |
| NPU runtime/init   |     4 |   198 | 初始化回滚缺失, import副作用   |
| Distributed/HCCL   |     1 |    91 | 全局裸指针, 线程安全            |
| Memory mgmt        |     1 |    84 | raw new/delete, UnlockGuard    |
| ACL interface      |     2 |    90 | static 竞争, enum 不同步       |
| Python glue        |     3 |   117 | monkey-patch, 未绑定变量       |
| Op framework       |     2 |    61 | void* 类型双关, 宏嵌套         |
| Task queue         |     1 |    48 | 非标准 memory barrier          |
| Inductor/codegen   |     1 |    33 | 无终止保证循环, 闭包状态       |

Profiler 子系统以 5 个文件、202 次 bugfix 成为最大缺陷聚集区。


### A.4 Per-File Risk Assessment

以下每条列出 2-3 个最关键的风险因子和 file:line 证据。

#### #1 ProcessGroupHCCL.cpp (91 bugfix, 7147 lines)

分布式通信核心。约 555 个分支点, 52 个 include。

- Race (全局裸指针): `ProcessGroupHCCL* global_` (:442) 无锁,
  构造函数(:1164)写、析构(:1404)清、`createHCCLCommSub`(:2694)读。
- Race (非原子全局): `nslb_is_end`, `device_error_msg`,
  `force_stop_error_flag` (:93-95) 被 watchdog(:2078) 和
  workEnqueue(:3246) 跨线程无锁读写。std::string 的 data race 是 UB。
- 析构 rethrow: `~ProcessGroupHCCL()` (:1419-1426) catch 后
  `std::rethrow_exception()`, C++11 析构中传播异常触发 terminate()。

Top risk: `collective()` (:3809-4023) 6 种模式嵌套条件,
recordStream/stash/eraseStream 路径选错即 use-after-free 或 leak。

#### #2 NPUCachingAllocator.cpp (84 bugfix, 3909 lines)

NPU 显存池化分配器。约 216 个分支点。

- Raw new/delete 无 RAII: `new Block` (:1321, 1360, 2276, 2682,
  2849, 2858); `delete block` (:2425, 2791, 2832, 3706)。异常路径
  Block 泄露无保护。
- UnlockGuard 不变量破坏: `malloc()` (:1238-1244) 释放 mutex 并调
  npuSynchronizeDevice, 期间其他线程可改变 pool 状态。
- recursive_mutex (:962) 掩盖逻辑重入导致的不变量违反。

Top risk: `alloc_found_block()` (:1341-1443) 块分割时 `new Block`
后若 `pool->blocks.insert()` 失败, 新块泄露。

#### #3 npu_sys_ctrl.cpp (63 bugfix, 404 lines) -- density 15.59

NPU 初始化状态机, 缺陷密度第三高。

- 无回滚初始化链: `Initialize()` (:145-249) 10+ 步顺序执行,
  中间步骤失败无 rollback。aclInit 成功后 SetDevice 失败时
  `init_flag_` 仍为 false, 下次触发 ACL_ERROR_REPEAT_INITIALIZE。
- Finalize 顺序脆弱: `init_flag_ = false` (:332) 在 release 回调
  循环(:339-344)之前, 回调抛异常时 Finalize 不可重入, 余下泄露。

#### #4 dynamic_profile.py (57 bugfix, 203 lines) -- density 28.08

全仓缺陷密度最高。

- step() 状态机非原子: prof/cfg_ctx/step_num 三变量在(:74-121)
  多路修改无事务保证。`_step_record_time` 初始化为 None(:40),
  step 10 因异常跳过时 step 11 处 `time.time() - None` TypeError。
- enable_prof() 无 None guard: 从 step() 和 start() 两条路径进入
  (:159-177), 但 start() 中 cfg_ctx 可被置 None(:156)。

#### #5 Module.cpp (57 bugfix, 2522 lines)

Python C 扩展绑定, 59 个 include (全 top-20 最高 fan-out)。

- size_t->指针强转: (:531, 549, 585, 594) Python 传入的 size_t
  直接 `(c10::StorageImpl*)`, 无类型安全机制。
- uint64_t->函数指针: NPUPluggableAllocator(:372-464) 将 Python
  ctypes 传入整数 reinterpret_cast 为函数指针, 签名不匹配时
  stack corruption。
- 全局无锁: `NPUDeviceMem memory` (:267) 文件作用域全局,
  GetDeviceMemories() 每次覆写, 多线程 race。

#### #6 AclInterface.cpp (54 bugfix, 1801 lines)

ACL 动态库符号绑定层。

- static 两步初始化: 约 120 处 `static func = nullptr;
  if (func == nullptr) { func = GET_FUNC(); }` 与 C++11 单步
  static init 不同, 存在 TOCTOU 窗口。
- 流创建无回滚: `AclrtCreateStreamWithConfig` (:188-221) stream
  创建成功后 SetOverflowSwitch/SetFailureMode 失败时 stream 泄露。
- 事件类型混淆: `AclrtCreateEventWithFlag` (:289-313) 特定条件下
  将 aclrtEvent* 指向 32 字节 device 内存, 不可用 DestroyEvent 释放。
- 日志错误: (:1377) AclrtGetDeviceResLimit 日志写成 "aclrtSet..."。

#### #7 NPUQueue.cpp (48 bugfix, 982 lines)

异步 SPSC 任务队列。87 个分支点 / 982 行 = 高密度。

- 非标准 barrier: 20+ 处 `__sync_synchronize` (非 C++11 atomic),
  write_idx/read_idx 非 atomic 类型, 正确性完全依赖 SPSC 假设。
- Lost wakeup: Enqueue() (:596-649) SetWriteWorking(false) 后
  eventfd_read 阻塞, 若 Consumer 已发信号则永久阻塞。
- ClearQueue 不安全: (:755) 直接 `read_idx = write_idx`,
  Consumer 正在处理时 item 未 Release 即被跳过。

#### #8 __init__.py (47 bugfix, 335 lines) -- density 14.03

包入口, 14+ 个 import-time 有状态操作。

- Import-time 副作用: _apply_patches(), _apply_class_patches(),
  atexit.register() 等(:219-317), 任一失败即半初始化无法回滚。
- 全局注入: globals()[name](:116-122) + setattr(torch, name, ...)
  动态注入 torch.ops.npu, 影响范围不可静态分析。
- 分布式覆盖: _apply_distributed_methods_patch (:194-207) 覆盖
  torch.distributed 的 10 个内部函数, torch 升级时调用时才暴露。

#### #9 utils/_module.py (44 bugfix, 524 lines)

PyTorch 模块方法 NPU 适配 patch。

- [confirmed bug] _lstm_forward 未绑定变量 (:175-188):
  `batch_sizes.device != input1.device` 且非 PackedSequence 时,
  result_tmp 被计算但未赋值给 result, (:188) `result[0]` NameError。
- cast_weight 死守卫: (:135) `if not self.children:` 中
  self.children 是方法 (非布尔), 始终 truthy, 子模块递归不执行。
- 私有 API 依赖: (:18) `torch.nn.parallel._functions._streams`,
  torch 升级易破坏。

#### #10 _dynamic_profiler_config_context.py (42 bugfix, 384 lines)

- 属性遮蔽方法: prof_path, record_shapes 等 7 个名称既是
  __init__ 实例属性(:15-37)又是 def 方法(:320-369), 属性遮蔽方法,
  `ctx.record_shapes()` 对 False 调用() 抛 TypeError。
- dyno 路径 None 未守: _parse_activity() (:175)
  `json_data.get('PROFILE_ACTIVITIES').split(",")` 无默认值。

#### #11 transfer_to_npu.py (42 bugfix, 504 lines)

CUDA->NPU 自动迁移层。import 即触发全量 patch, 不可逆。

- device_ids 更新丢失: _wrapper_cuda (:181-182) 更新本地变量
  但未写回 kwargs['device_ids'], DDP 初始化时 device_ids 未替换。
- 异常吞没: (:83-84, 95-96) `except Exception: return []`,
  patch 加载失败完全静默。
- dynamo hack: (:486) 直接清空内部私有属性 function_ids = None。

#### #12 profiler.py (37 bugfix, 347 lines) -- density 10.66

- _ProfInterface 双重构造: profile.__init__ 先父类构造(:50-58)
  后本类覆盖(:224-234), 第一次构造被无声丢弃。
- __del__ 调 stop() (:255-257): GC 期间依赖对象可能已回收。
- os.cpu_count() -> None: analyse() (:332-334) 比较时 TypeError。

#### #13 AclInterface.h (36 bugfix, 388 lines) -- density 9.28

- 本地 enum 与 ACL 库重叠: (:19-60) 6 个与 CANN 同名 enum,
  CANN 升级数值不同步则运行时错误而非编译错误。
- 默认 HcclComm 参数: AclrtReserveMemAddress(:186),
  AclrtMapMem(:195) 等默认 nullptr 让分布式注册可被无声跳过。
- 全文缺 [[nodiscard]]: 所有返回 aclError 的函数, 可静默丢弃。

#### #14 _cann_file_parser.py (33 bugfix, 239 lines) -- density 13.81

- __init__ 吞异常: (:84-88) _check_cann_path_valid() 的
  RuntimeError 被 except 捕获仅打印, 对象在 _cann_path 无效下继续。
- ast.literal_eval 代替 json.loads (:205): Python 特有字面量
  与 JSON 语义不一致。
- O(N*M) 正则: _file_dispatch() (:232-238) 每文件 x ~30 pattern。

#### #15 _profiling_parser.py (33 bugfix, 147 lines) -- density 22.45

全仓缺陷密度第二高。

- except Exception 后继续: analyse_profiling_data() (:99-101)
  吞掉 run_parser() 异常后执行 simplify_data(), 可能在错误状态下
  删除原始数据 (simplify_flag=True 路径)。
- CANNFileParser 双重实例化: run_parser() (:119-120) 对同一路径
  连续构造两个 parser, 两次 get_cann_path 之间存在 TOCTOU。

#### #16 _inductor/codegen/triton.py (33 bugfix, 1970 lines)

Inductor Triton kernel codegen NPU 适配。

- 无终止循环: codegen_body (:954-960) `while True:` 在
  sorted_axis 全非 tiling_axis 时无限循环, 无断言保护。
- CSEProxy 闭包: __enter__ 中定义 (:1691-1968) 通过闭包引用
  外层 self, kernel 复用时闭包指向旧状态。

#### #17 npu/utils.py (31 bugfix, 494 lines)

- hccl_detect_group 无锁: stress_detect (:428-436) check-then-set
  无锁, 多进程同时创建 group 导致 distributed 不一致, 可能死锁。
- StreamContext.__exit__ None: src_prev_stream 初始化为 None,
  __enter__ 早返回后 __exit__ (:227) 访问 None.device。

#### #18 npu/__init__.py (31 bugfix, 795 lines)

NPU 子模块入口, lazy init 实现。

- [confirmed bug] 名称错误: (:230) 定义 `_DeferredNpuCallError`,
  (:255) 使用 `DeferredNpuCallError` (缺少下划线), lazy init 的
  queued call 失败时触发 NameError 掩盖真实错误。
- fork 缓存失效: _after_fork(:292) 重置 _initialized 但不重置
  _cached_device_count(:437), 子进程永久返回父进程缓存值。
- 6 处星号导入: (:167-170, 580-581) 命名空间不可静态分析。

#### #19 OpParamMaker.cpp (31 bugfix, 769 lines)

- CopyFunc 裸类型双关: (:570-613) void* 偏移 + placement new,
  对齐未保证, 析构后异常则双重销毁。
- ExecFunc reset_flag: (:332-424) JIT compile 状态在特定异常
  路径未还原, 影响后续所有算子编译行为。
- funcMap 无守护: AsncExecFunc (:705-712) `funcMap[type]` 无
  count() 检查, 枚举扩展时调用空函数指针。

#### #20 NPUException.h (30 bugfix, 345 lines) -- density 8.70

- NPU_CHECK_ERROR_CHECK_UCE 宏 (:142-209): 展开约 60 行,
  5 层嵌套, error_code 在分支中被 AclrtPeekAtLastError 覆写。
  此宏是该头文件 bugfix 集中区。
- CHECK_AND_THROW_ERROR_WITH_SPECIFIC_MESSAGE (:121-140):
  无 `do { ... } while(0)` 包裹, 与其他宏风格不一致。
- 每调用点独立 static: `static AclErrorCode err_map` 在宏体内
  (:144), 每个调用点独立实例, 初始化成本被隐藏。


### A.5 Cross-Cutting Risk Patterns

从 20 个 hotspot 文件中抽取的系统性风险模式:

1. 线程安全漏洞 (7/20 文件):
   ProcessGroupHCCL.cpp 全局裸指针, NPUCachingAllocator.cpp
   UnlockGuard, AclInterface.cpp static 两步初始化, NPUQueue.cpp
   非标准 barrier, Module.cpp 全局 NPUDeviceMem, npu/utils.py
   hccl_detect_group, dynamic_profile.py 全局 CFG_CONFIG_PATH。
   根因: C++ 层手动同步而非 atomic/lock-free, Python 层依赖 GIL。

2. 异常路径资源泄露 (6/20 文件):
   NPUCachingAllocator.cpp new Block, AclInterface.cpp stream,
   OpParamMaker.cpp placement new, npu_sys_ctrl.cpp 初始化链,
   profiler.py __del__, NPUQueue.cpp ClearQueue。
   根因: RAII 使用不充分, 关键分配用 raw new 而非 smart pointer。

3. 异常吞没 / 静默失败 (6/20 文件):
   transfer_to_npu.py `except: return []`, _cann_file_parser.py
   吞 init 异常, _profiling_parser.py 吞 parse 异常,
   _module.py print(e) 后继续, NPUException.h 错误码覆写,
   npu/__init__.py 星号导入掩盖接口变更。
   根因: 防御式编程过度, 宽 except 兜底无日志, 下游在错误状态
   上继续执行。

4. Import-time 不可逆副作用 (3/20 文件):
   __init__.py (14+ 操作), transfer_to_npu.py (import 即 patch),
   npu/__init__.py (6 处星号导入)。
   根因: Python 模块初始化承载过多 runtime setup, 无 lazy init。

5. 类型安全绕过 (4/20 C++ 文件):
   Module.cpp size_t->指针, OpParamMaker.cpp void* 类型双关,
   NPUQueue.cpp 裸指针算术, AclInterface.h enum 重定义。
   根因: 与 ACL C API 交互时类型擦除不可避免, 但缺少边界检查。


### A.6 Confirmed Bugs (回源验证)

以下两个问题经代码回源验证确认为真实 bug:

1. npu/__init__.py:255 -- NameError in lazy init error path
   (:230) 定义 `class _DeferredNpuCallError(Exception)`,
   (:255) 使用 `raise DeferredNpuCallError(...)` 缺少下划线前缀。
   影响: queued call 失败时 NameError 替代预期的错误信息。

2. utils/_module.py:175-188 -- unbound variable in LSTM forward
   当 `batch_sizes.device != input1.device` 且 orig_input 非
   PackedSequence 时, result_tmp 被计算但未赋值给 result,
   (:188) `output = result[0]` 触发 NameError。
   影响: 特定设备不一致 + 非 PackedSequence 输入时 LSTM crash。


---

## Part B: REVERT Deep-Dive

101 条 REVERT commit, 去重后约 34 个独立事件。
多数 revert 被 cherry-pick 到多个版本分支, 因此计数膨胀。

### B.1 Summary Table

| ID | Subject (abbreviated)             | Cnt | Area          | Escape     |
|:--:|:----------------------------------|----:|:--------------|:-----------|
|  1 | revert removing libhccl dep       |   9 | Build         | DEPENDENCY |
|  2 | Revert "memory optimaze"          |   9 | Profiler      | REGRESSION |
|  3 | revert record stream in HCCL      |   9 | Distributed   | REGRESSION |
|  4 | Revert "Correcte Time calc-func"  |   7 | NPUQueue      | REGRESSION |
|  5 | Revert "Second-work flow slot"    |   7 | NPUQueue      | DESIGN_FLAW|
|  6 | Revert set core num to hccl       |   6 | Distributed   | REGRESSION |
|  7 | Revert AsyncCompile before reg    |   6 | Inductor      | REGRESSION |
|  8 | Revert format_contiguous combine  |   6 | ATen/Ops      | INCOMPLETE |
|  9 | Revert export_stacks Profiling    |   4 | Profiler      | PREMATURE  |
| 10 | revert syncop changes             |   3 | Distributed   | DESIGN_FLAW|
| 11 | Revert Parse trave_view CSV       |   3 | Profiler      | INCOMPLETE |
| 12 | Revert re package deleted mistake |   2 | Serialization | REGRESSION |
| 13 | Revert _lazy_init device_guard    |   2 | Device Init   | REGRESSION |
| 14 | Revert delete pin_memory warning  |   2 | ATen/Ops      | REGRESSION |
| 15 | Revert PR and fix bug             |   2 | Inductor/MLIR | PREMATURE  |
| 16 | Revert Index/IndexPut API         |   3 | ATen/Ops      | INCOMPLETE |
| 17 | revert patch_expr_fits_32bit      |   1 | Inductor      | REGRESSION |
| 18 | Revert Triton Contiguous Reduction|   1 | Inductor      | REGRESSION |
| 19 | revert mlir module import fix     |   1 | Inductor/MLIR | REGRESSION |
| 20 | Revert "raise instead of exit"    |   1 | Error Handling| DESIGN_FLAW|
| 21 | revert torchair commitid          |   1 | Build         | DEPENDENCY |
| 22 | revert event                      |   1 | Runtime       | REGRESSION |
| 23 | Revert manylinux support pypi     |   1 | Build         | INCOMPLETE |
| 24 | Revert trace_step_time            |   1 | Profiler      | INCOMPLETE |
| 25 | Revert flash_attention kernel     |   1 | ATen/Ops      | REGRESSION |
| 26 | revert repeatInterleave.Tensor    |   1 | ATen/Ops      | INCOMPLETE |
| 27 | Revert hostapi path default       |   1 | ATen/Ops      | INCOMPLETE |
| 28 | Revert item vs ConvertTensorScalar|   1 | ATen/Ops      | REGRESSION |
| 29 | Revert shutdown device resources  |   1 | Shutdown      | REGRESSION |
| 30 | Revert fixed c180c88              |   4 | Shutdown      | REGRESSION |
| 31 | Revert fix core dump after shut   |   1 | Shutdown      | REGRESSION |
| 32 | Revert npu memory management      |   1 | Memory        | PREMATURE  |
| 33 | revert d6fb78+a283f4+1af6a3       |   1 | ATen/Ops      | REGRESSION |
| 34 | Revert to deepcopy                |   1 | Serialization | REGRESSION |

### B.2 Escape Cause Distribution

| Escape Cause       | Count | Pct  |
|:-------------------|------:|-----:|
| REGRESSION         |    18 |  53% |
| INCOMPLETE_TESTING |     7 |  21% |
| DESIGN_FLAW        |     3 |   9% |
| PREMATURE          |     3 |   9% |
| DEPENDENCY         |     2 |   6% |
| SCOPE_CREEP        |     1 |   3% |

过半 revert 属于 REGRESSION: 修复了一个问题但引入了新问题。
这指向 review 和测试流程中对副作用的检查不足。

### B.3 Top 15 Deep Analysis

#### Event #1: revert removing libhccl dependency (9x, DEPENDENCY)

原始 commit `8c44b9513`: 从 CMakeLists.txt 和 setup.py 中删除
`_C.cpython*.so` 对 `libhccl.so` 的显式链接。

被 revert 原因: 下游库通过 `liba.so -> libtorch_npu.so -> libhccl.so`
间接依赖 libhccl 符号, 删除后运行时找不到符号崩溃。
CI 只覆盖了 torch_npu 本身, 未覆盖下游链路。

教训: 删除库依赖前需扫清传递依赖链。

#### Event #2: Revert "memory optimaze" (9x, REGRESSION)

原始 commit `c59c83de7`: TraceViewParser 读设备侧 json 前移到
CANNTimelineParser, 分离两次读操作降低峰值。

被 revert: 在特定硬件上默认开启 timeline 切片时, 前移的读操作
被误判为超时, 导致 timelineparser 报错。内存优化改变了 parser
执行时序, 在特定硬件配置下破坏了超时判定。

教训: timing 敏感代码修改需跨平台/跨配置测试矩阵。

#### Event #3: revert record stream in HCCL (9x, REGRESSION)

原始 commit `aead5ffa2`: ProcessGroupHCCL 中 recorded_inputs_ 从
`weak_ptr<StorageImpl>` 改为 `Storage` (强引用),
eraseStream 改为 eraseStreamForce。

被 revert: 强引用阻止 Storage 在 collective 结束前释放, 改变了
ERASE_RECORD_STREAM 模式下的内存管理语义, 与下游 GC 冲突。
eraseStreamForce 跳过 null check, storage 已释放时悬空指针。

教训: 引用强度变更是语义变更, 需端到端验证内存释放时序。

#### Event #4: Revert "Correcte Time calculate-func" (7x, REGRESSION)

原始 commit `48f4cd2c44`: 两处修改 -- 修复 GET_MSEC 宏
(nsec/1000 改为 nsec/1000000) + 调整 Dequeue/clock_gettime 顺序。

被 revert: GET_MSEC 修复正确, 但调用顺序调整在并发时序下使
consumer 线程 sleep 行为异常。两处修改被捆绑提交, 正确的修复
被错误的优化拖累。

教训: 不相关修改不应捆绑提交, 原子 commit 原则。

#### Event #5: Revert "Second-work flow time-slot" (7x, DESIGN_FLAW)

原始 commit `6491559ea6`: 大幅重构 NPUQueue 同步 -- 删除 eventfd,
改为 consumer 自旋等待 + TTL 超时。

被 revert: 低负载时 CPU 空转, TTL 偏差时任务丢失或 hang。
7 个分支全部回滚说明问题具有普遍性。

教训: 调度策略从阻塞改自旋是根本性变更, 需分析不同负载模式。

#### Event #6: Revert "set core num to hccl thread" (6x, REGRESSION)

原始 commit `92a86938a`: collective() 热路径增加
UseStreamResInCurrentThread 调用, 强制更新核数配置。

被 revert: 热路径增加 ACL 运行时查询带来开销, 且在非控核场景下
产生线程变量副作用, 污染其他算子的核数配置。

教训: 热路径修改需评估性能影响和隐式状态副作用。

#### Event #7: Revert AsyncCompile before register (6x, REGRESSION)

原始 commit `5b8aba051`: inductor thread pool warm_pool() 从
import 时推迟到首次使用时。

被 revert: 首次 inductor 调用时同步创建 thread pool 导致
首次推理延迟显著增加。

教训: import-time 优化可能将成本转移到关键路径。

#### Event #8: Revert format_contiguous combine (6x, INCOMPLETE)

原始 commit `4c0413536`: 合并 format_contiguous 和
format_contiguous_add_copy_optimize。

被 revert: 两函数在 copy optimize 语义上不完全等价,
without_copy_optimize 场景被错误合并到 with 路径, 导致额外 copy。

教训: 合并相似函数前需枚举语义差异, 为每个 case 编写测试。

#### Event #9: Revert export_stacks from Profiling (4x, PREMATURE)

原始 commit `4f375cb0e` (510 行): export_stacks 随 Profiler 大规模
重构一起提交。

被 revert: 未经独立验证的子功能, 存在临时文件泄露。

教训: 不同功能不应捆绑在一个大 PR 中。

#### Event #10: revert syncop changes (3x, DESIGN_FLAW)

原始 commit `1b30c9f91`: collective() 增加 asyncOp 参数,
同步 op 直接在当前流执行, 跳过 syncStreams。

被 revert: 跳过 syncStreams 意味着跳过通信流与计算流的同步屏障,
在流水线并行下破坏通信-计算依赖顺序。

教训: 流同步是正确性基石, 任何 skip-sync 优化需严格证明安全性。

#### Event #11-15: (lower impact events)

- #11 Parse trave_view CSV (3x): 跨 CANN 版本格式不一致导致解析失败
- #12 re package deleted by mistake (2x): 清理时误删 `import re`
- #13 _lazy_init in device_guard (2x): C++->Python 迁移改变初始化顺序
- #14 delete pin_memory warning (2x): 删 C++ stub 后框架内部直接调用
  空指针访问
- #15 PR and fix bug (2x): MLIR 大规模重构(408+1599行)未经常规 review

#### Notable Additional Events (traced from background analysis)

Event #22 (format_contiguous combine, 6x, ATen/Ops):
原始 `4c0413536` 将 format_contiguous_add_copy_optimize 调用统一替换
为 format_contiguous, 涉及 AddKernelNpu, BmmKernelNpu, MmKernelNpu 等
核心算子。这些算子有特定的 copy optimize 路径, 合并后性能下降。

Event #25 (item vs ConvertTensorToScalar, 1x, ATen/Ops):
原始 `f842e440c` 将 ConvertTensorToScalar 替换为 PyTorch 原生 item(),
涉及 add/div/mul/sub 核心算子。item() 触发 D2H 同步, 在异步执行上下文
中造成性能回退或 deadlock。原始的 ConvertTensorToScalar 有异步处理机制。

Event #27 (shutdown device resources, DESIGN_FLAW):
原始 `c180c8839` 在 shutdown 中直接移除 event clear 和 stream destroy
等清理操作来避免 crash, 而非修复清理顺序。过于激进, 后续 Event #28
(`a243de6e2`) 引入 aclrtDestroyStreamForce 尝试修复同一问题,
但该 API 版本兼容性不足又被 revert, 说明 shutdown 资源回收是
持续未解决的根本性设计问题。

Event #32 (NZ/5HD format specs, 1x):
原始为三个关联 commit (`d6fb786` + `a283f40` + `1af6a3`), 让算子
不再指定输出的私有格式 (NZ/5HD/Conv 专用格式), 由框架自动推断。
涉及 19 个 op kernel (Add, Addmm, Gru, Lstm, Conv 系列, Softmax,
Threshold, RoiAlign 等)。框架格式推断在某些组合下选错格式导致失败。


### B.4 Revert-Prone Area Aggregation

| Area              | Events | Commits | Primary escape          |
|:------------------|-------:|--------:|:------------------------|
| Distributed/HCCL  |      4 |      27 | REGRESSION, DESIGN_FLAW |
| NPUQueue/Runtime  |      3 |      18 | REGRESSION, DESIGN_FLAW |
| Profiler          |      4 |      16 | REGRESSION, PREMATURE   |
| Inductor          |      4 |      10 | REGRESSION              |
| ATen/Ops          |      6 |      13 | INCOMPLETE_TESTING      |
| Build/Packaging   |      3 |      11 | DEPENDENCY              |
| Shutdown          |      3 |       6 | REGRESSION              |

Distributed/HCCL + NPUQueue 合计 45 条 revert commits,
是最不稳定的子系统。

### B.5 Structural Lessons

三个结构性问题:

1. 热路径修改缺乏多场景压测。Events 3/5/6/10 都发生在
   ProcessGroupHCCL.cpp 或 NPUQueue.cpp, 这两个文件的变更
   缺乏多配置集成测试 (不同流策略、内存模式、有/无前置算子)。

2. 大 PR 拆分不足。Events 9/15/22 都是 300+ 行综合 PR,
   包含未经独立验证的子功能, 被作为整体回滚。
   原子 commit 原则在此仓库执行不严格。

3. 删旧接口不等于删旧依赖。Events 1/13/14 都是删除了某个
   C++ 实现或动态库依赖, 却未扫清所有调用点, 在框架内部
   或下游库中触发运行时失败。


---

## Cross-Validation: Hotspot x REVERT

| File                    | Bugfix Rank | Revert Events    |
|:------------------------|:------------|:-----------------|
| ProcessGroupHCCL.cpp    | #1 (91)     | Events 3, 6, 10  |
| NPUCachingAllocator.cpp | #2 (84)     | Event 32         |
| NPUQueue.cpp            | #7 (48)     | Events 4, 5      |
| __init__.py             | #8 (47)     | Event 20         |
| triton.py               | #16 (33)    | Events 17, 18    |
| OpParamMaker.cpp        | #19 (31)    | (indirect: #4)   |

ProcessGroupHCCL.cpp 同时是 bugfix 最密集和 revert 最频繁的文件,
印证其结构性风险 (全局状态竞态, 线程安全缺陷)。

NPUQueue.cpp 虽 bugfix 排名 #7, 但两个 DESIGN_FLAW 级 revert
(Events 4, 5) 说明其同步机制的设计比实现更成问题。
