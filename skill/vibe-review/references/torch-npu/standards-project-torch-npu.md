# 【项目级】torch_npu Code Review Standards

Generated: 2026-04-12
Data source: defect-analysis.md (1,316 defects), hotspot-analysis.md (20 hotspot files + 34 REVERT events)
Method: root cause clustering from 1,316 defect entries, graded by (escape severity x frequency)


## Grading

| Grade | Meaning              | Action                             |
|:------|:---------------------|:-----------------------------------|
| P0    | Block merge          | Must resolve before approve        |
| P1    | Must fix             | Fix before merge, no exceptions    |
| P2    | Should address       | Fix or justify in review comments  |
| P3    | Nice to have         | Flag for author, merge OK without  |


## Category Overview

Sorted by (escape severity x frequency) descending.

| # | Category                                  | Count | Pct  | Grade | Escape Impact       |
|--:|:------------------------------------------|------:|-----:|:------|:--------------------|
| 1 | Concurrency & thread safety               |   ~20 | 1.5% | P0    | Deadlock, UB, crash |
| 2 | Resource leak & C++ memory safety         |   ~15 | 1.1% | P0    | OOM, corruption     |
| 3 | Tensor semantic deviation                 |   ~41 | 3.1% | P1    | Silent wrong result |
| 4 | State & lifecycle management              |   ~55 | 4.2% | P1    | Hang, crash         |
| 5 | Error/exception handling                  |   ~15 | 1.1% | P1    | Silent failure      |
| 6 | New-addition completeness (op/dtype/enum) |   ~80 | 6.1% | P2    | Runtime error       |
| 7 | Upstream PyTorch sync                     |   ~52 | 4.0% | P2    | Compat breakage     |
| 8 | Monkey-patch & dynamic injection          |   ~17 | 1.3% | P2    | Silent misroute     |
| 9 | Edge-case coverage (0-dim/empty/SoC)      |   ~30 | 2.3% | P2    | Rare crash          |
|10 | Naming & identifier errors                |  ~244 | 18.5%| P3    | Compile/test fail   |

Remaining ~747 entries are unique one-off root causes (count 1-2 each) that
do not form stable clusters. They are addressed by general code review
discipline rather than specific checkpoints.


---

## P0: Block Merge

### P0-1: Concurrency & Thread Safety

Frequency: ~20 defects across 7/20 hotspot files
Hotspot cross-ref: ProcessGroupHCCL.cpp (#1, 91 bugfix),
NPUCachingAllocator.cpp (#2, 84 bugfix), NPUQueue.cpp (#7, 48 bugfix)

Root pattern: hand-rolled synchronization (raw pointers, `__sync_synchronize`,
non-atomic shared variables) instead of C++11 atomic/lock-free primitives.

Typical cases:

- D-1062 (`34f04920f`): holding allocator mutex -> releasing GIL ->
  GC triggers -> reacquires same mutex = AB-BA deadlock.
  ProcessGroupHCCL.cpp + NPUCachingAllocator.cpp + NPUQueue.cpp.
- D-879 (`0d27919b9`): MakeSureQueueEmpty loop holds GIL on some paths,
  TE compilation thread needs GIL = deadlock in inductor pipeline.
- D-1086 (`a09ce7fd1`): recursive_mutex + npuSynchronizeDevice,
  child thread GC needs same lock = cross-thread deadlock.
- D-862 (`96abe60a8`): event-record vs event-query across threads
  without synchronization, process_events queries unrecorded events.
- D-1283 (`853fed5ef`): WriteQueue check-then-act without lock,
  TOCTOU race on queue full status.

Review checkpoints:

1. Lock ordering: does this code path acquire mutex A then mutex B?
   Does any other path acquire B then A? Check especially:
   allocator_mutex vs GIL vs HCCL comm mutex.
2. Lock scope: while holding any mutex, does the code call
   `npuSynchronizeDevice()`, Python callbacks, or any path that
   may trigger GC? If yes, this is a deadlock candidate.
3. Shared variable access: every cross-thread variable
   (`global_`, `nslb_is_end`, `device_error_msg`, `write_idx`,
   `read_idx`) must be `std::atomic` or protected by a mutex.
   Non-standard `__sync_synchronize` is a red flag.
4. Check-then-act: is the condition check and the subsequent
   mutation within the same critical section?

Detection: grep for `__sync_synchronize`, `recursive_mutex`,
global raw pointers in ProcessGroupHCCL.cpp/NPUQueue.cpp.

Note: checkpoint 1 (lock ordering) requires full call-graph tracing, which
is infeasible from diff-only review. Recommended tooling: ThreadSanitizer,
Clang Thread Safety Annotations (`__attribute__((capability("mutex")))`).
For manual review, focus on checkpoints 2-4 which are diff-local.


### P0-2: Resource Leak & C++ Memory Safety

Frequency: ~15 defects, concentrated in 6/20 hotspot files
Hotspot cross-ref: NPUCachingAllocator.cpp (#2), AclInterface.cpp (#6),
OpParamMaker.cpp (#19), npu_sys_ctrl.cpp (#3)

Root pattern: raw `new`/`delete` without RAII; ACL resource creation
without paired destruction; empty destructor/deleter bodies.

Typical cases:

- D-70 (`8b4c3e678`): `svm_deleter()` has empty function body,
  all swapped-memory host allocations never freed.
- D-122 (`c4f4b33a9`): `aclCreateTensor` descriptor not released
  after format cast, leaks one host descriptor per call.
- D-183 (`bedc8509b`): `processEvents` loop aborts on first
  `event->query()` exception, remaining events never cleaned up.

Review checkpoints:

1. Every `new Block`, `aclCreateXxx`, `aclrtCreateStream` call:
   is there a matching `delete`/`aclDestroyXxx` reachable from
   ALL paths (including exception paths)?
2. Destructor/deleter function bodies: are any of them empty?
   An empty deleter is a confirmed leak (D-70 pattern).
3. Initialization chains (npu_sys_ctrl.cpp pattern): if step N
   fails, are steps 1..N-1 rolled back? Or does partial init
   leave dangling resources?
4. `UnlockGuard` usage: code that temporarily releases a mutex
   must re-validate all invariants after re-acquiring the lock.

Detection: grep for `new Block`, `aclCreate`, empty function
bodies in deleter/destructor functions.


---

## P1: Must Fix Before Merge

### P1-1: Tensor Semantic Deviation

Frequency: ~41 defects
REVERT cross-ref: Events #8 (format_contiguous, 6x), #32 (NZ/5HD, 1x)
in ATen/Ops area (6 events, 13 commits, INCOMPLETE_TESTING)

Root pattern: NPU op implementation deviates from PyTorch native behavior
in stride layout, dtype promotion, output shape, or memory format.
These produce silent wrong results -- the most dangerous defect class.

Sub-types:

- stride/MemoryFormat loss (~12): op unconditionally outputs contiguous
  tensor, ignoring `MemoryFormat::Preserve`.
  D-9 (`5a798156e`): `clone()` ignores Preserve, 8 cherry-picks.
  D-43 (`32033f92d`): `empty_like_npu` always applies format.

- dtype promotion missing (~10): binary ops ignore `result_type()`,
  output dtype defaults to first operand.
  D-583 (`d3914eb96`): `cumsum` ignores user-passed `dtype` param.
  D-603 (`f339e1d19`): `bitwise_and` int32 & int64 returns int32.

- output shape error (~8): shape computed from wrong tensor state
  (pre-permute vs post-permute, pre-pad vs post-pad).
  D-607 (`90cbe38de`): `nllloss2d` output_size from 4D, actual is 1D.

- out-variant format reference (~6): `CheckOut` uses `self` format
  instead of `result` format.
  D-589 (`dc07ea827`): `exp_out` references self instead of result.

- other (~5): mixed cases (wrong contiguity assumption, view aliasing)
  that do not form a stable sub-cluster.

Review checkpoints:

1. Every tensor factory call (`ApplyTensor`, `at::empty`,
   `apply_tensor_without_format`): does it respect all MemoryFormat values?
   - `Preserve` mode: pass through strides from input (use `at::empty_strided`)
   - Explicit format (ChannelsLast, Contiguous, etc.): pass the format to
     the factory call (use `at::empty_like(src, options, memory_format)`)
   - `apply_tensor_without_format(src)` ignores format entirely -- if the op
     accepts a MemoryFormat parameter, this factory is insufficient.
   (Validated: commit 5a798156e fixed Preserve but missed explicit formats;
   see OpPreparation.cpp:395-398)
2. Every op with `optional<ScalarType> dtype` parameter: is the
   dtype actually used in output tensor creation? grep the function
   body for the parameter name appearing in `ApplyTensor`/`at::empty`.
3. Every binary/reduction op: does it call `at::result_type()` for
   output dtype? Or does it silently use `self.scalar_type()`?
4. Every `*_out` variant: does `CheckOut` reference `result` (not
   `self`) for format/dtype validation?
5. Does a unit test compare against PyTorch CPU output for the same
   inputs (including non-contiguous, mixed-dtype, 0-dim cases)?

Detection: grep `ApplyTensor\|apply_tensor_without_format` in op
implementation files; check if `memory_format` appears in arguments.


### P1-2: State & Lifecycle Management

Frequency: ~55 defects
Hotspot cross-ref: dynamic_profile.py (#4, density 28.08 -- highest),
npu_sys_ctrl.cpp (#3, density 15.59), __init__.py (#8, density 14.03)

Root pattern: multi-variable state machines without transactional
guarantees; initialization sequence without rollback; global state
modified from multiple entry points.

Typical cases:

- D-13 (`5cc3d16e5`): variables `experiment`/`optimize_ctx` used
  before initialization in DDP branch + boolean condition inverted.
- D-6 (`916426933`): ACLGraph capture + multi-stream: task queue
  not flushed during capture, replay produces wrong results.
- D-315 (`be2158f51`): init guard checks wrong field, closure self
  points to outer object instead of inner.
- D-316 (`4d1bc797f`): loop unconditionally overwrites shared dict
  entry, should be first-write-wins semantics.

Review checkpoints:

1. Variables used in conditional branches: are they initialized
   BEFORE the branch, at the common dominator of all use sites?
   (D-13 pattern: init after DDP branch but used inside it.)
2. State machine transitions: if step N fails, what state are
   variables from steps 1..N-1 left in? Is the state consistent
   for the next call?
3. Shared container mutation in loops: is the semantics
   "first write wins" or "last write wins"? Is it explicit?
4. Background threads with polling loops: is there an external
   termination flag beyond "task complete"?

Detection: search for global/module-level mutable variables;
check initialization order against usage order in each branch.


### P1-3: Error & Exception Handling

Frequency: ~15 defects, cross-cutting 6/20 hotspot files
Hotspot cross-ref: A.5 items 2 (resource leak on exception, 6 files)
and 3 (exception swallowing, 6 files)

Root pattern: broad `except Exception` silences real errors;
functions with bool return signature throw internally;
destructors propagate exceptions (C++ terminate).

Typical cases:

- D-29 (`db1de538e`): `catch(std::exception&)` unifies network timeout
  with HCCL hardware error, real error deferred to first collective.
- D-32 (`24f51db28`): `~ProcessGroupHCCL()` rethrow in implicit
  noexcept destructor triggers `std::terminate()`.
- D-204 (`4d88e0118`): `IsGteDriverVersion()` calls `TORCH_CHECK(false)`
  instead of returning false, caller expects bool but gets crash.
- D-131: `transfer_to_npu.py` `try/except Exception` swallows
  `jit.script` failures, dynamo mode silently bypassed.

Review checkpoints:

1. Catch granularity: does each catch block distinguish recoverable
   (network timeout) from non-recoverable (device error) exceptions?
   Broad `except Exception: pass` is a P1 red flag.
2. Destructor safety: C++ destructors must not throw. Any
   `throw`/`rethrow_exception` in a destructor or `__del__` is P0.
3. Function contract: if the signature returns bool, does any
   internal path call `TORCH_CHECK(false)` or throw? The caller
   assumes bool semantics.
4. Cleanup loops: if a cleanup/release loop calls a fallible API
   (e.g., `event->query()`), does the loop continue on individual
   failure or abort the entire cleanup?

Detection: grep for `except Exception`, `rethrow_exception`,
`TORCH_CHECK` inside bool-returning functions.


---

## P2: Should Address

### P2-1: New-Addition Completeness (Op/Dtype/Enum)

Frequency: ~80 defects (largest P2 cluster)
Sub-types: API/op registration (38), dtype support (27),
enum branch (15)

Root pattern: torch_npu maintains an N x M registration matrix
(op x {aten table, functionalization, DTensor, schema.json, dynamo,
HCCL type map, SoC guard}). Adding a new op/dtype/enum without
updating ALL registration points leaves gaps.

Typical cases:

- D-5 (`b70ee2d8f`): new `EXECUTE_OPAPI_V2` enum value added but
  `get_func_error_msg()` and `Enqueue()` switch not updated.
- D-10 (`a4bbd71ad`): pow op missing NPU lowering for fp64 dtype,
  inductor compilation crashes on fp64 pow.
- D-558 (`2cc9bc67c`): new dtype ACL mapping missing in type table.
- D-408 (`ad2102331`): new op added without updating functionalization.

Review checkpoints:

1. New enum value: grep ALL switch/if consumers of that enum type.
   Enable `-Wswitch-enum` compiler flag.
2. New op registration: is it registered in aten op table AND
   functionalization AND DTensor sharding AND schema.json AND
   dynamo trace rules?
3. New dtype support: is the type mapping updated in BOTH forward
   AND backward AND the dtype allowlist? Does a test cover
   float16/bfloat16/float64/complex?
4. Inplace variant: if `foo` is registered, is `foo_` also
   registered with matching semantics?

Detection: when a PR adds a new enum/op/dtype, grep existing
registrations of sibling items and verify parallel registration.


### P2-2: Upstream PyTorch Sync

Frequency: ~52 defects
REVERT cross-ref: Events #12-15 (upstream compat breaks)

Root pattern: torch_npu forks/patches PyTorch internals.
When upstream renames, removes, or changes signatures,
the NPU side breaks at import time or runtime.

Sub-types:

- API rename / path move (~28):
  D-39 (`df136434f`): `additional_module_tests` renamed upstream.
  D-184: `triton_heuristics` moved to `runtime` submodule.

- Signature change (~12):
  D-134 (`7650811e2`): `TritonCSEVariable.__init__` gained `shape`
  param, NPU patch rejects it with TypeError.
  D-96: `evaluate_static_shape()` renamed to `guard_int()`.

- Interface addition/removal (~12):
  D-126 (`d69f2f060`): AOTI added `constant_blob_size()`,
  NPU wrapper not implemented.

Review checkpoints:

1. All `from torch._xxx import yyy`: does `yyy` still exist in
   the target PyTorch version? Run import smoke test in CI.
2. All monkey-patches: does the patch function signature match
   the current upstream function exactly (including new params)?
3. Does torch_npu maintain a duplicate implementation of something
   upstream now provides natively? If so, can we delegate instead?

Detection: CI gate that runs `python -c "from torch._xxx import yyy"`
for every import; diff upstream changelog per version bump.


### P2-3: Monkey-Patch & Dynamic Injection

Frequency: ~17 defects
Hotspot cross-ref: __init__.py (#8), transfer_to_npu.py (#11)

Root pattern: `from X import func` creates an independent reference
that patches to the original module do not propagate;
patches lack idempotency guards; import-time side effects are
irreversible.

Typical cases:

- D-45 (`8343dcb87`): `from-import` creates independent binding,
  `patch_get_first_incompatible_cudagraph_node()` modifies original
  module but not the `_graph_tree` binding. Detection silently fails.
- D-147 (`dfc2f5c99`): `patch_shape_handling()` called multiple times
  without idempotency flag, wrapper functions nest recursively.
- D-163 (`30dc60baf`): FSDP patch covers `all_gather_copy_in` but
  misses `.default` dispatch variant.

Review checkpoints:

1. Does the patch target have any `from X import func` consumers?
   Those get independent references that the patch cannot reach.
2. Is there an idempotency guard (module-level flag or
   `hasattr(func, '_npu_patched')`)? Can the patch function be
   called twice safely?
3. Does the patch cover ALL dispatch variants (`.default`,
   inplace `_` suffix, out `_out` suffix)?
4. Import-time ops: does module import trigger device init,
   comm registration, or compilation? Can these be deferred to
   first use?

Detection: grep `setattr.*torch\.\|monkey`, check for missing
`_npu_patched` guards.


### P2-4: Edge-Case Coverage (0-dim / Empty / SoC)

Frequency: ~30 defects
Sub-types: boundary conditions (18), SoC/device guards (12)

Root pattern: ops assume non-empty, non-scalar, specific-SoC inputs.
Edge cases (0-dim scalar tensor, shape-contains-0, empty list,
new SoC model) trigger division by zero, out-of-bounds, or
API unavailability.

Typical cases:

- D-19 (`5034e00f5`): size=0 boundary not handled.
- D-8: `aclrtGetDeviceInfo()` unavailable on Ascend950 (A5),
  returns error 507899. Fix: SoC version guard.
- D-148 (`cd5f447d5`): SoC version range guard incomplete,
  API available on A2 but not A5.

Review checkpoints:

1. Does the op handle 0-dim tensor (scalar), empty tensor
   (numel=0), and shape-contains-0 inputs?
2. New ACL/driver API calls: is there a SoC version guard?
   Guard design guidance:
   - Prefer explicit exclusion (`!= Ascend950`) over range exclusion
     (`< Ascend950`) when the unsupported SoC set is small and known.
     Range exclusion blocks future SoCs that may support the API.
   - If the API has a runtime capability check (e.g. `IsExistXxx()`),
     prefer that over version comparison.
   - When neither is available, use try-catch with specific error code
     as last resort.
   (Validated: commit 2e5b8909c used `< Ascend950` -- functional but
   blocks all future SoCs; consider `!= Ascend950` for narrower scope)
   (Cross-ref: coding-conventions.md 9.3 for the general SoC compat rule)
3. Divisor variables: can any denominator be zero for valid inputs?

Detection: grep for `dim() == 0`, `numel()`, `GetSocVersion` in
changed op files.


### P2-5: Build & Packaging Changes

Frequency: ~11 commits across 3 REVERT events (Appendix B: DEPENDENCY cause)
Added by: validation cross-check (commit bcec360bb had zero applicable rules)

Root pattern: build script changes (setup.py, CMakeLists.txt, patchelf logic)
that modify linking, dependency stripping, or package layout can break
import-time loading on specific environments (missing .so, wrong RPATH,
stripped symbols still needed by dlopen).

Review checkpoints:

1. Does the change modify what gets linked, stripped, or packaged?
   If yes, verify on a clean environment (no CANN SDK, no conda).
2. patchelf operations: does `--remove-needed` target a library that
   might be loaded via dlopen at runtime (GET_FUNC pattern)?
   Stripping a compile-time link is safe; stripping a dlopen target is not.
3. Glob patterns for file selection: does `rglob('*.so')` or
   `glob('_C.cpython*.so')` match exactly the intended files?
   Test with `print(list(pattern))` before applying destructive ops.
4. Cross-platform: does the change assume Linux-only tools (patchelf,
   readelf)? Is there a macOS/Windows guard?

5. Import smoke test: after patchelf/stripping operations, verify loadability:
   `python -c "import torch_npu; print(torch_npu.__version__)"` on a clean
   environment (no CANN SDK in LD_LIBRARY_PATH, no conda). This catches
   stripped symbols still needed at import time via dlopen.

Detection: any PR touching setup.py, CMakeLists.txt, or build scripts
should trigger these checkpoints.


---

## P3: Nice to Have

### P3-1: Naming & Identifier Errors

Frequency: ~244 defects (18.5% of all defects -- largest single category)

Despite the high count, most are caught by compiler errors,
linters, or first-run unit tests. Escape-to-production rate is low.

Sub-types:

- Variable/parameter name typo (~50):
  D-20 (`0f59193ef`): `call_args` vs `call_arg` (plural/singular),
  type check always False.
- Dict key / magic string typo (~14):
  D-310 (`d80a46978`): `"traced_hash_dir"` should be
  `"traced_graph_dir"`, debug dump silently fails.
- Copy-paste name error (~25):
  D-279 (`733b24f4f`): `isinstance(self.statistic_value, DTensor)`
  but operates on `grad`, should check `grad`.
- Import path error (~35):
  D-435 (`b8350e5bc`): `torch._C` vs `torch_npu._C` confusion.
- Log/doc string typo (~20):
  D-993 (`8fa79690b`): `ASCNED_BASE` instead of `ASCEND_BASE`
  in env.sh, 8 path references all broken.
- Other miscellaneous naming errors (~100): remaining entries spread
  across config keys, test fixture names, and inline constants,
  each sub-pattern count < 10, no stable cluster.

Review checkpoints:

1. Near-identical variable names (differ by `_`, `s`, `_data`):
   verify each reference matches its declaration exactly.
2. Dict keys: are they defined as constants/enums? Does each
   write-key have a matching read-key?
3. Copy-pasted blocks: does the `isinstance` subject match the
   object operated on in the same branch?
4. `torch._C` vs `torch_npu._C`: which module owns the symbol?
5. Environment variable names: does the definition site match
   all reference sites? (grep to verify.)

Detection: spell-checker in CI (codespell/typos-cli);
`-Wuninitialized` and `-Winit-self` compiler flags;
IDE rename-refactor instead of manual find-replace.


---

## Appendix A: Hotspot File Risk Matrix

Top 10 files by bugfix count, with primary defect categories.

| Rank | File (under torch_npu/)              | Bugfix | Density | Primary Risk Categories |
|-----:|:-------------------------------------|-------:|--------:|:------------------------|
|    1 | csrc/distributed/ProcessGroupHCCL.cpp|     91 |    1.27 | P0-1 concurrency, P1-2 state |
|    2 | csrc/core/npu/NPUCachingAllocator.cpp|     84 |    2.15 | P0-1 concurrency, P0-2 leak |
|    3 | csrc/core/npu/sys_ctrl/npu_sys_ctrl.cpp|   63 |   15.59 | P0-2 leak, P1-2 state    |
|    4 | profiler/dynamic_profile.py          |     57 |   28.08 | P1-2 state (non-atomic) |
|    5 | csrc/npu/Module.cpp                  |     57 |    2.26 | P0-2 type-safety bypass |
|    6 | csrc/core/npu/interface/AclInterface.cpp| 54  |    3.00 | P0-1 TOCTOU, P0-2 leak   |
|    7 | csrc/core/npu/NPUQueue.cpp           |     48 |    4.89 | P0-1 concurrency |
|    8 | __init__.py                          |     47 |   14.03 | P2-3 import-time side effects |
|    9 | utils/_module.py                     |     44 |    8.40 | P3-1 naming (unbound var) |
|   10 | profiler/...config_context.py        |     42 |   10.94 | P1-2 attribute shadows method |


## Appendix B: REVERT-Prone Areas

| Area             | REVERT Events | Commits | Primary Escape Cause |
|:-----------------|:--------------|--------:|:---------------------|
| Distributed/HCCL | 4             |      27 | REGRESSION           |
| NPUQueue/Runtime | 3             |      18 | DESIGN_FLAW          |
| Profiler         | 4             |      16 | REGRESSION           |
| Inductor         | 4             |      10 | REGRESSION           |
| ATen/Ops         | 6             |      13 | INCOMPLETE_TESTING   |
| Build/Packaging  | 3             |      11 | DEPENDENCY           |
| Shutdown         | 3             |       6 | REGRESSION           |

Structural lesson from REVERT data (hotspot-analysis.md B.5):

1. Unrelated changes must not be bundled in one commit.
   Events #4, #9 reverted correct fixes because bundled
   with incorrect optimizations.
2. Hot-path changes need multi-scenario stress test.
   Events #3, #5, #6 all in ProcessGroupHCCL.cpp / NPUQueue.cpp,
   lacking multi-config integration tests.
3. Removing old interfaces requires scanning ALL transitive
   consumers. Events #1, #13, #14 removed implementations
   without clearing downstream call sites.


## Appendix C: Quick Reference Checklist

Minimum review checklist for any torch_npu PR:

P0 checks (block if violated):
- [ ] No new global mutable state without mutex/atomic
- [ ] No lock held while calling npuSynchronizeDevice or GC-triggering code
- [ ] Every `new`/`aclCreate` has a matching RAII or explicit cleanup on ALL paths
- [ ] Destructors do not throw

P1 checks (must fix):
- [ ] Op output matches PyTorch native for same inputs (stride, dtype, shape)
- [ ] State variables initialized before ALL use paths
- [ ] No `except Exception: pass` without specific recovery logic
- [ ] Cleanup loops continue on individual item failure
- [ ] Bug fix includes a regression test (or documents why not feasible)

P1 extra (tensor ops):
- [ ] Op respects ALL MemoryFormat values (Preserve, ChannelsLast, Contiguous)
- [ ] `apply_tensor_without_format` not used when op accepts MemoryFormat param

P2 checks (should address):
- [ ] New enum/op/dtype registered in ALL parallel registration points
- [ ] All `from torch._xxx import` verified against target version
- [ ] Monkey-patches cover `.default` and inplace variants
- [ ] 0-dim, empty, and boundary inputs handled
- [ ] New ACL API calls have SoC version guard (prefer point exclusion over range)
- [ ] Build/packaging changes: patchelf/setup.py tested on clean env without CANN

P3 checks (flag to author):
- [ ] No near-duplicate variable names differing by underscore/plural
- [ ] Dict keys defined as constants, not magic strings
- [ ] Copy-pasted blocks have correct subjects in conditions
