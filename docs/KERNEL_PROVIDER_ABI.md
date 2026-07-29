# Ferrule Kernel Provider C ABI

## 1. 状态、范围与目标

- 状态：未来设计基线；首个公开主版本暂定为 ABI major 1。
- 所有权：该 ABI 由 Ferrule 自维护、版本化并发布，不继承 CUTLASS、cuda-oxide 或任一 Rust crate 的 ABI。
- 目标：Ferrule Rust host 可独立装载、协商、选择和调用架构专用 kernel provider。
- 目标：kernel 实现、CUDA Toolkit、CUTLASS 版本和编译器差异全部隔离在 provider artifact 内。
- 目标：stream、device memory、workspace、graph、event 与生命周期始终由 Rust host 控制。
- 非目标：稳定 C++ 模板、Rust 类型、CUDA Runtime API、provider 私有对象或跨 provider 序列化格式。
- 非目标：用单一万能 fatbin 覆盖全部 NVIDIA GPU。
CUTLASS、CuTe、手写 CUDA 与 cuda-oxide 只能是 provider 内部实现细节。
model、scheduler 和 ABI consumer 不得依赖其类型、符号、错误枚举、kernel 名或布局。

## 2. ABI 平台、布局与兼容规则

ABI major 1 的平台 contract 是 Linux ELF、little-endian、8-bit byte、LP64、自然 C 对齐。
首批 host 架构为 x86_64 与 aarch64；Windows/LLP64 必须使用新的 platform contract，不能假装兼容。
所有导出函数和函数指针使用 `extern "C"` 与平台默认 C calling convention；头文件统一使用 `FERRULE_ABI_CALL`。
ABI 边界只传 C POD、固定宽度整数、CUDA Driver API handle、pointer-length view 与 opaque handle。
不跨边界传递 Rust/C++ 对象、引用、trait object、异常、STL、bit-field、`long double` 或所有权智能指针。
布尔值与 wire enum 使用 `uint32_t`；尺寸、偏移、计数和 device byte address 使用 `uint64_t`。
每个公开的完整结构必须以 `uint32_t struct_size` 开头；opaque incomplete handle 没有公开布局。
每个输入结构的 caller 必须将已知字段填好，并将全部 `reserved`、未知尾部和未使用 flag 清零。
provider 遇到非零 reserved 必须返回 `FERRULE_STATUS_INVALID_ARGUMENT`；所有输出 reserved 必须写零。
同一 major 内只能在结构尾部追加字段、enum 尾部追加值、能力集合追加 bit；不得改变既有语义或对齐。
reader 只能访问 `min(caller_struct_size, sizeof(local_struct))` 前缀，并先检查必选前缀是否完整。
所有结构禁止 `#pragma pack`；C 与 Rust 必须对 `sizeof`、`alignof`、`offsetof` 做 golden assertions。

minor 协商规则：
- `requested_major` 通过唯一入口参数传入。
- `host_api` 提供 `abi_minor_min` 与 `abi_minor_max` 闭区间。
- provider 选择交集中的最高 minor，写入 `out_api->abi_minor`；无交集返回 `UNSUPPORTED_ABI`。
- negotiated minor 决定字段语义和可用函数；`struct_size` 只决定内存可见前缀，不能冒充能力协商。
- optional 函数必须同时满足 negotiated minor、capability bit 和非 NULL function pointer。

## 3. 唯一入口、基础类型与 host API

每个 provider DSO 只要求导出固定名称的入口：
```c
FERRULE_EXPORT ferrule_status_t FERRULE_ABI_CALL
ferrule_kernel_provider_get_api(
    uint32_t requested_major,
    const ferrule_host_api_t* host_api,
    ferrule_provider_api_t* out_api);
```
caller 在调用前设置两个结构的 `struct_size`，清零其余输出字节。
入口不得访问 CUDA context，不得抛出异常、panic unwind 或跨边界 `longjmp`。
major/minor 不匹配时不得返回“近似可用”的函数表。
```c
typedef int32_t ferrule_status_t;
typedef struct ferrule_provider_t* ferrule_provider_handle_t;
typedef struct ferrule_device_t* ferrule_device_handle_t;
typedef struct ferrule_plan_t* ferrule_plan_handle_t;
typedef struct ferrule_graph_plan_t* ferrule_graph_plan_handle_t;
typedef struct ferrule_graph_exec_lease_t* ferrule_graph_exec_lease_handle_t;
typedef struct {
    uint32_t struct_size, major, minor, patch;
    uint64_t reserved[2];
} ferrule_version_t;
typedef struct {
    uint32_t struct_size, flags;
    const uint8_t* data;
    uint64_t len, reserved[2];
} ferrule_string_view_t;
typedef struct {
    uint32_t struct_size, flags;
    const void* data;
    uint64_t count;
    uint32_t element_size, element_alignment;
    uint64_t reserved[2];
} ferrule_array_view_t;
typedef struct {
    uint32_t struct_size, flags;
    void* user_data;
    void* (FERRULE_ABI_CALL *alloc)(void*, uint64_t bytes, uint64_t alignment);
    void (FERRULE_ABI_CALL *free)(void*, void*, uint64_t bytes, uint64_t alignment);
    uint64_t reserved[4];
} ferrule_host_allocator_t;
typedef struct {
    uint32_t struct_size, abi_major, abi_minor_min, abi_minor_max, flags, reserved0;
    const ferrule_host_allocator_t* allocator;
    void* log_user_data;
    void (FERRULE_ABI_CALL *log)(void*, uint32_t, const uint8_t*, uint64_t);
    uint64_t reserved[8];
} ferrule_host_api_t;
```
log message 只在 callback 返回前有效；host callback 必须可被多个 provider 线程并发调用。
allocator pointer 与 callbacks 在 provider 存活期间稳定；alignment 必须是 2 的幂且至少为 pointer alignment。
provider 可以不用 host allocator，改用自身 allocator 管理仅内部可见的 host metadata。
若协商 `CAP_HOST_ALLOCATOR`，provider 的内部 host metadata 必须使用 host callbacks，并在父 handle 销毁前释放。
provider 不得把 allocator 的 ownership 交还 caller，也不得在 handle 销毁后调用 callback。

## 4. Allocation、返回内存与所有权

禁止的是 caller-visible allocation 与 GPU 隐式 allocation，不是 provider 内部 host bookkeeping：
- provider 不得为 caller 分配或返回需要 caller `free` 的 array、string、blob、workspace 或 binding storage。
- provider 不得调用 `cudaMalloc*`、`cuMemAlloc*`、managed/pinned allocation，或内部创建 device scratch。
- provider 不得在 `launch`、capture 或 replay 中 lazy-load module、扩容 cache 或触发隐藏 workspace allocation。
- module/code load 只能发生在显式 `open_device`/warmup/`prepare` 阶段；不得把其 driver-managed module 资源冒充 workspace。
- provider 可以为 provider/device/plan/graph plan 分配私有 host metadata；该内存不得包含 caller-owned buffer 的所有权。
- 序列化采用 size-query + caller-owned output buffer；不返回 provider-allocated blob。
所有返回 view 都是 immutable borrowed view：
- `ferrule_string_view_t` 的 `len` 是字节数，UTF-8、不要求 NUL，`data == NULL` 当且仅当 `len == 0`。
- `ferrule_array_view_t` 用 `count`、`element_size`、`element_alignment` 完整描述数组；每个 record 自带 `struct_size`。
- manifest 的 strings/arrays 由 provider handle 拥有，从成功 `get_manifest` 到 `destroy_provider` 前保持地址和内容稳定。
- plan selection、weight/workspace records 由 plan 拥有，从成功 `prepare` 到 `destroy_plan` 前保持稳定。
- graph records 由 graph plan 拥有，到最后一个 graph exec lease 释放且 graph plan 销毁前保持稳定。
- callback 参数只在 callback 内借用；launch 输入结构及其数组至少借用到函数返回，device memory 借用到异步完成。
caller 不得释放、修改或跨上述 lifetime 缓存 borrowed pointer；需要更长 lifetime 时必须复制。

## 5. 正式 manifest 与 capability records

sidecar manifest 用于不加载 DSO 的静态筛选；DSO 内嵌 manifest 是运行时权威事实。
两者必须具有相同 schema、provider build ID、artifact SHA-256 和 code-object/operator record 哈希。
```c
typedef struct {
    uint32_t struct_size, flags;
    uint64_t feature_id;
    ferrule_string_view_t canonical_name;
    uint64_t reserved[4];
} ferrule_accelerated_feature_t;
typedef struct {
    uint32_t struct_size, flags;
    uint64_t code_object_id;
    uint32_t kind, architecture_family; /* NATIVE_CUBIN or PTX */
    uint16_t cc_min_major, cc_min_minor, cc_max_major, cc_max_minor;
    uint16_t target_cc_major, target_cc_minor;
    uint32_t reserved0;
    ferrule_string_view_t verified_target_name;
    ferrule_array_view_t accelerated_features;
    ferrule_version_t ptx_isa, min_driver, build_toolkit;
    uint8_t image_sha256[32];
    uint64_t image_offset, image_bytes, reserved[6];
} ferrule_code_object_record_t;
typedef struct {
    uint32_t struct_size, flags;
    uint64_t record_id, operator_id;
    uint32_t operator_schema_min, operator_schema_max;
    ferrule_array_view_t code_object_ids, required_cap_words, optional_cap_words;
    ferrule_array_view_t dtype_contract_ids, layout_contract_ids, shape_contract_ids;
    uint64_t weight_layout_id;
    uint32_t weight_layout_version, workspace_policy_version;
    uint64_t reserved[6];
} ferrule_operator_capability_record_t;
typedef struct {
    uint32_t struct_size, manifest_schema_major, manifest_schema_minor, abi_major;
    uint32_t abi_minor_min, abi_minor_max, flags, reserved0;
    uint64_t provider_build_id_hi, provider_build_id_lo;
    uint8_t artifact_sha256[32];
    ferrule_string_view_t provider_name, provider_version;
    ferrule_version_t build_toolkit, host_compiler;
    ferrule_array_view_t capabilities, code_objects, operators;
    uint64_t reserved[8];
} ferrule_provider_manifest_t;
```
`verified_target_name` 只能来自构建所用 toolkit 的 `nvcc --list-gpu-code`，不能按产品名猜测。
`accelerated_features` 必须列出稳定 Ferrule feature ID；空数组表示普通 target，不表示未知。
PTX record 必须声明 virtual target、PTX ISA 与 runtime minimum driver；native record 必须声明精确 native target。
CC range 是 provider 声明的可选范围，不能扩大 NVIDIA 对 cubin 或 accelerated code 的兼容规则。
operator record 是 capability 的最小协商单元；unknown required capability 必须拒绝，unknown preferred capability 忽略。
建议基础能力：`ASYNC_ERROR_RING`、`GRAPH_CAPTURE`、`GRAPH_CONCURRENT_EXEC`、`PLAN_SERIALIZATION`、
`MULTI_STREAM_PLAN`、`WEIGHT_LAYOUT_QUERY`、`GENERIC_PTX_FALLBACK`、`HOST_ALLOCATOR`。
capability bit 一经发布不得复用；bitset 通过 `element_size == 8` 的 array view 扩展。

## 6. Provider API、opaque handle 与 unload 生命周期

```c
typedef struct {
    uint32_t struct_size, abi_major, abi_minor, flags;
    ferrule_status_t (FERRULE_ABI_CALL *create_provider)(ferrule_provider_handle_t*);
    ferrule_status_t (FERRULE_ABI_CALL *destroy_provider)(ferrule_provider_handle_t);
    ferrule_status_t (FERRULE_ABI_CALL *get_manifest)(ferrule_provider_handle_t, ferrule_provider_manifest_t*);
    ferrule_status_t (FERRULE_ABI_CALL *open_device)(ferrule_provider_handle_t, const void*, ferrule_device_handle_t*);
    ferrule_status_t (FERRULE_ABI_CALL *close_device)(ferrule_device_handle_t);
    ferrule_status_t (FERRULE_ABI_CALL *prepare)(ferrule_device_handle_t, const void*, void*, ferrule_plan_handle_t*);
    ferrule_status_t (FERRULE_ABI_CALL *destroy_plan)(ferrule_plan_handle_t);
    ferrule_status_t (FERRULE_ABI_CALL *initialize_workspace)(ferrule_plan_handle_t, const void*);
    ferrule_status_t (FERRULE_ABI_CALL *launch)(ferrule_plan_handle_t, const void*);
    ferrule_status_t (FERRULE_ABI_CALL *create_graph_plan)(ferrule_plan_handle_t, const void*, ferrule_graph_plan_handle_t*);
    ferrule_status_t (FERRULE_ABI_CALL *capture_graph_launch)(ferrule_graph_plan_handle_t, const void*);
    ferrule_status_t (FERRULE_ABI_CALL *register_graph_exec)(ferrule_graph_plan_handle_t, const void*, ferrule_graph_exec_lease_handle_t*);
    ferrule_status_t (FERRULE_ABI_CALL *release_graph_exec)(ferrule_graph_exec_lease_handle_t);
    ferrule_status_t (FERRULE_ABI_CALL *destroy_graph_plan)(ferrule_graph_plan_handle_t);
    ferrule_status_t (FERRULE_ABI_CALL *serialize_plan)(ferrule_plan_handle_t, void*);
    ferrule_status_t (FERRULE_ABI_CALL *deserialize_plan)(ferrule_device_handle_t, const void*, ferrule_plan_handle_t*);
    uint64_t reserved[12];
} ferrule_provider_api_t;
```
所有 `void*` 实际指向本规范定义且带 `struct_size` 的具体 POD；正式头文件必须使用具体 typedef。
生命周期偏序：
```text
DSO > provider > device/CUDA module > plan > graph plan > graph exec lease > in-flight replay
                                      plan > ordinary in-flight launch
```
Rust 必须先停止新调用，再等待 caller-owned completion event，释放 graph exec、graph plan、plan、device、provider，最后 `dlclose`。
provider DSO 与其 CUDA module 绝不能在 device、plan、in-flight launch、graph plan 或 `CUgraphExec` 仍引用代码时卸载。
所有 destroy/close/release 都禁止同步；前置条件未满足返回 `FERRULE_STATUS_BUSY`，不得内部等待 GPU。
NULL destroy 可为 no-op；非 NULL handle 不得 double-destroy，失败销毁后 ownership 不转移。

## 7. Thread、CUDA context、reentrancy 与 destroy race

host 在调用 `open_device`、`prepare`、workspace init、launch、graph capture 和 CUDA-touching destroy 前，
必须在当前 OS thread 设置与 device handle 绑定的 `CUcontext`。
provider 不创建、切换、push/pop 或销毁 context，不依赖“首次调用线程”，handle 永久绑定 device UUID 与 context identity。
context 不匹配返回 `FERRULE_STATUS_CONTEXT_MISMATCH`，不得自动修复。
manifest query 与不同 handle 上的调用必须 thread-safe/reentrant；同一 immutable plan 是否可多流并发由能力位决定。
同一 mutable graph plan 的 capture/register/release 操作由 Rust 串行化；普通 launch 不得修改 plan selection。
host allocator/log callback 可并发进入，但 callback 禁止同步回调 provider API，避免锁反转和递归重入。
provider 不得持锁调用耗时 host callback；不得在 API 返回或父 handle 销毁后继续使用 callback 参数。
destroy 与该 handle 的任何 API 调用不得并发；Rust 用 handle gate/refcount 阻止新调用并等待 CPU 调用退出。
GPU in-flight lifetime 由 Rust event/graph lease 跟踪；CPU 调用退出不代表 GPU 完成。

## 8. `prepare`、不可变 selection 与完整 plan cache key

`prepare` 只验证、选择、计算尺寸并创建 immutable plan，不启动 kernel，不绑定 stream 或瞬时 activation。
request 至少含 operator ID/schema、device UUID/context identity、shape、dtype、layout/stride、alignment、math mode、
required/preferred capabilities、weight descriptors、graph intent 和 determinism policy。
```c
typedef struct {
    uint32_t struct_size, flags;
    uint64_t selected_operator_record_id, selected_code_object_id, operator_id;
    uint32_t operator_schema, selected_code_kind;
    uint16_t selected_cc_major, selected_cc_minor;
    uint32_t uses_ptx_jit;
    ferrule_array_view_t enabled_cap_words, accelerated_feature_ids;
    ferrule_version_t runtime_min_driver;
    uint64_t selection_hash_hi, selection_hash_lo, reserved[6];
} ferrule_plan_selection_t;
typedef struct {
    uint32_t struct_size, flags;
    uint64_t offset, bytes, alignment;
    uint32_t lifetime;    /* PER_LAUNCH, PLAN_INSTANCE, GRAPH_EXEC */
    uint32_t init_policy; /* NONE, ZERO_EACH, ZERO_ONCE, EXPLICIT_PROVIDER_INIT */
    uint64_t contract_id, reserved[4];
} ferrule_workspace_region_t;
typedef struct {
    uint32_t struct_size, flags;
    uint64_t workspace_bytes, workspace_alignment;
    ferrule_array_view_t workspace_regions;
    uint64_t weight_layout_id;
    uint32_t weight_layout_version, reserved0;
    ferrule_array_view_t weight_layout_records;
    ferrule_plan_selection_t selection;
    uint64_t plan_fingerprint_hi, plan_fingerprint_lo, reserved[6];
} ferrule_prepare_result_t;
```
selection 必须写入 plan 与 result；launch 不得按当时 driver、stream 或地址重新挑另一个 code object。
完整 plan cache key 至少包含：ABI major/minor、platform contract、provider build ID、artifact SHA-256、manifest hash、
operator ID/schema、operator record ID、code object ID/hash、accelerated feature IDs、PTX ISA/JIT 状态、runtime driver、
device UUID/CC/context compatibility class、enabled capabilities、shape、dtype、logical/physical layout、strides/alignment、
math/determinism/graph flags、weight layout ID/version、workspace policy/version，以及所有影响 kernel 选择的 tuning 参数。
若 plan 绑定权重地址，key 还必须含 allocation identity、address、bytes 与 content generation；否则只含 weight layout contract。
任一 key 分量变化都必须 cache miss；不得在不同 provider build、PTX JIT compiler/driver 或 device UUID 间复用 plan。
plan serialization 必须携带该 key/fingerprint、endianness、checksum；不匹配时返回 `SERIALIZATION_INCOMPATIBLE`。

workspace 由 Rust 分配、拥有和回收：
- 总大小/对齐覆盖全部 region；零大小时 alignment 必须为 1。
- `PER_LAUNCH` region 不得在重叠 launch 间共享；`GRAPH_EXEC` region 每个并发 exec 独占。
- `ZERO_EACH` 由 Rust 在同一 stream 每次 launch 前清零；`ZERO_ONCE` 与 workspace instance generation 绑定。
- `EXPLICIT_PROVIDER_INIT` 只能通过显式 `initialize_workspace` 向 caller stream 排队，不能在首个 launch lazy init。
- persistent 内容只对精确 plan fingerprint、device/context、workspace address 和 instance generation 有效。
- Rust 必须记录初始化 completion；未初始化、太小、错位或错误 generation 在提交前同步拒绝。
- provider 不得将 persistent workspace 当作跨 plan 全局 cache，也不得在 destroy 中读取或清零它。

## 9. `launch`、同步状态与异步错误

```c
typedef struct {
    uint32_t struct_size, flags;
    uint64_t operand_id;
    uint32_t access_mode, dtype;
    CUdeviceptr address;
    uint64_t bytes;
    uint32_t rank, reserved0;
    const int64_t *dims, *strides;
    uint64_t allocation_id, generation, reserved[4];
} ferrule_binding_t;
typedef struct {
    uint32_t struct_size, flags, code, provider_detail;
    uint64_t launch_id, detail0, detail1, generation, reserved[4];
} ferrule_async_error_record_t;
typedef struct {
    uint32_t struct_size, flags;
    const ferrule_binding_t* bindings;
    uint32_t binding_count, reserved0;
    CUdeviceptr workspace;
    uint64_t workspace_bytes, workspace_instance_generation;
    CUstream stream;
    CUdeviceptr error_record;
    uint64_t error_record_bytes;
    CUevent completion_event;
    uint64_t launch_id, reserved[6];
} ferrule_launch_desc_t;
```
caller-owned bindings/workspace/error record/event 至少活到 completion；ownership 不转移。
provider 只向传入非默认 stream 排队，不创建 stream/event，不查询或同步 event，不做阻塞 D2H readback。
provider 禁止 host/device sync、GPU allocation、host-visible allocation、lazy init 和使用 legacy default stream。
同步返回码只描述验证与完整提交：`OK`、`INVALID_ARGUMENT`、`UNSUPPORTED_ABI`、`CAPABILITY_MISSING`、
`UNSUPPORTED_DEVICE`、`UNSUPPORTED_OPERATOR`、`UNSUPPORTED_LAYOUT`、`WORKSPACE_TOO_SMALL`、
`MISALIGNED_ADDRESS`、`CONTEXT_MISMATCH`、`CUDA_SUBMIT_FAILED`、`STALE_PLAN`、
`SERIALIZATION_INCOMPATIBLE`、`BUSY`、`INTERNAL`；数值发布后不可复用。
非零同步错误表示完整工作序列未提交；provider 私有错误只能进入 detail 或日志，不扩散私有 enum。
Rust 在同一 stream 提交前清零并设置 error record `struct_size/generation/launch_id`。
kernel 用 first-writer-wins 原子协议写 `code`；完成前 host 不读取，完成后 Rust同时收集 Driver API 异步错误。
provider 可以记录 caller-owned completion event，但 event 的创建、等待、复用和销毁完全属于 Rust。

## 10. Graph plan、launch cookie、error ring 与 replay

graph capture 是可选能力，不能把普通 plan 直接视为 capture-safe。
`create_graph_plan` 在 capture 外完成 module load、lazy initialization、固定 kernel selection 和 graph workspace contract。
`capture_graph_launch` 只能在 Rust 已开始 capture 的 caller stream 中排队 capture-safe nodes，不得分配或 host callback。
```c
typedef struct {
    uint32_t struct_size, flags;
    uint64_t replay_id, graph_exec_id;
    uint32_t error_slot, reserved0;
    uint64_t error_generation, workspace_instance_generation, reserved[4];
} ferrule_graph_launch_cookie_t;
typedef struct {
    uint32_t struct_size, flags;
    CUdeviceptr records;
    uint32_t record_count, record_stride;
    uint64_t record_bytes, reserved[4];
} ferrule_error_ring_desc_t;
typedef struct {
    uint32_t struct_size, flags;
    CUgraphExec graph_exec;
    uint64_t graph_exec_id;
    CUdeviceptr launch_cookie;
    uint64_t launch_cookie_bytes;
    ferrule_error_ring_desc_t error_ring;
    CUdeviceptr workspace;
    uint64_t workspace_bytes, workspace_instance_generation, reserved[6];
} ferrule_graph_exec_desc_t;
```
每个 replay 必须有唯一 cookie 值与独占 error-ring `(slot, generation)`；graph node 通过固定 cookie address 读取它们。
Rust 在 `cuGraphLaunch` 前于同一 stream 更新 caller-owned cookie，清零目标 error slot，并在 launch 后记录 completion event。
slot 在对应 completion 前不得复用；旧 generation 的晚写必须被检测并不得覆盖新 replay 的错误。
ABI major 1 禁止同一 `CUgraphExec` 重叠 launch；concurrent replay 使用不同 `CUgraphExec`、cookie、workspace instance 和 ring slot。
只有声明 `GRAPH_CONCURRENT_EXEC` 时，同一 immutable graph plan 才可支撑多个独立 graph exec 并发。
`register_graph_exec` 返回 lease，不取得 `CUgraphExec` ownership；Rust 先等待 replay，再销毁 `CUgraphExec`，最后 release lease。
graph exec lease 保持 graph plan、plan、device、CUDA module 和 DSO 活跃；存在 lease 时 destroy parent 必须返回 `BUSY`。
普通 async error 语义同样适用于 graph ring；每个 replay 的 route/status error 必须落到该 replay 自己的 slot。
plan serialization 可选，但 graph plan、graph exec lease、cookie address、event、workspace 和 error ring 都不可序列化。

## 11. Build Toolkit、runtime driver、PTX JIT 与硬件矩阵

`build_toolkit` 是构建 provenance；`min_driver` 是运行资格；二者不得混为“最低 CUDA 版本”。
native cubin selection 检查 code-object target、CC/family、accelerated features 和该 image 的 runtime minimum driver。
PTX selection 还必须检查 virtual target、PTX ISA、driver JIT 支持与 JIT policy；JIT 结果及 driver 版本进入 plan cache key。
不能仅凭 build toolkit 推导 min driver，也不能因 driver 足够新就假定它接受未知 PTX ISA/accelerated target。
manifest 必须分别记录 artifact build toolkit、每个 code object 的 min driver，以及 PTX record 的 PTX ISA。
以下是 NVIDIA 官方已核实的产品 CC 与 CUTLASS 构建最低 Toolkit，不是 runtime min driver 表：
| 产品 | Compute Capability | CUTLASS 最低构建 CUDA Toolkit |
| --- | ---: | ---: |
| GeForce RTX 3090 | 8.6 | 11.4 |
| NVIDIA H200 | 9.0 | 11.8 |
| NVIDIA B300 / GB300 | 10.3 | 13.0 |
| GeForce RTX 5090 | 12.0 | 12.8 |
| NVIDIA GB10（DGX Spark） | 12.1 | 13.0 |
CUDA cubin 通常只在相同 CC major 且目标 minor 不高于设备 minor时兼容；accelerated target 更严格。
`SM90a` 与 Blackwell architecture-accelerated code 不能跨 major/family；Hopper `90a` 不能用于 Blackwell。
Blackwell `SM100` datacenter 与 `SM120` GeForce 不兼容，`sm100a` code 不能在 RTX 50 `SM120` 上运行。
不得未经 toolkit 验证硬编码 B300/GB300 或 GB10 accelerated target 拼写。
构建必须执行并保存 `nvcc --version` 与 `nvcc --list-gpu-code`，manifest 只写 nvcc 实际支持的 target。
推荐 per-architecture/family provider artifact + manifest，不构建巨大万能 fatbin。
generic PTX 仅是有显式 capability、正确性测试和性能遥测的 fallback，不能替代优化 provider。
selection 顺序是精确优化 provider、合法同 family cubin、合法 generic PTX fallback、明确不支持。

## 12. 测试与发布门槛

ABI/layout：测试 major/minor 区间、大小前缀、尾部追加、reserved-zero、NULL optional 函数、guard bytes，
并在 C/Rust 比较所有 structure/record 的 size/alignment/offset/enum/calling convention。
manifest：sidecar/embedded hash 一致，array element size 正确，native/PTX/CC/feature/PTX ISA/min driver 组合可验证，
错误或未知 required capability 在 module load/prepare 前被拒绝，selection 完整写入 plan。
allocation：拦截 allocator/CUDA API，证明 launch/capture/replay 无 host-visible/GPU/lazy allocation；内部 host metadata 全部按父 handle 回收。
workspace/cache：覆盖零大小、错位、短 buffer、每次清零、一次初始化、persistent generation、跨流错误共享和完整 key miss。
错误：每个同步码有确定触发；异步 first-writer、generation、status/route error、CUDA failure 与 event ordering 无 race。
线程/context：不同 handle 并发、同 plan 多流能力、错误 current context、callback 并发、destroy race 与 `BUSY` 行为。
graph：capture/instantiate/register/replay/release，cookie 更新、ring wrap/generation、不同 exec 并发、同 exec 重叠拒绝，
并验证 graph exec/in-flight 存活时无法卸载 graph plan、module 或 DSO。
serialization：checksum、build ID、driver/JIT、device UUID、selection、workspace policy 任一不符都明确 cache miss/拒绝。
硬件：每个 artifact 在 manifest 声明的真实 GPU/driver/toolkit 组合运行，不能用模拟 CC 代替。

## 13. 从现有 ABI9 direct functions 迁移

现有 ABI9 SM121 catalog 必须冻结为恰好以下十个算子，不再新增 model direct entry：
1. `Fp8QueryAKvSm121`
2. `Bf16CompressorSm121`
3. `HcProducerSm121`
4. `SharedFfnSm121`
5. `StableFrameFp4MoeSm121`
6. `MlaOutputSm121`
7. `DsparkMainProjectNormSm121`
8. `DsparkHybridMlaAttentionSm121`
9. `DsparkProposalHeadSm121`
10. `Fp8ProjectionSm121`
阶段 A：为十算子记录 binding、shape、dtype、weight layout、workspace、stream、status 与 route-error oracle。
阶段 B：SM121 adapter 只对外暴露新 POD ABI，把 manifest/prepare/launch 映射到 ABI9 direct functions。
adapter 必须先移除旧路径中的 allocation、lazy init 或 synchronization，不能把违规永久藏在 wrapper 内。
阶段 C：先迁一个边界清晰算子（建议 `dspark_main_project_norm`）做 ABI9/provider shadow 双跑，使用独立输出、
workspace、cookie/error slot，比较数值、错误、stream ordering、workspace 上界、graph 行为与性能。
阶段 D：逐算子切主路径，保留有指标和截止日期的回退；CI 禁止新增 `crate::cutlass`，现存调用数单调下降。

迁移不可豁免的硬门槛：
- catalog：十个 ABI9 算子均有 operator/capability/code-object/weight/workspace records，缺一不得删除 ABI9。
- 全层：真实 DeepSeek-V4/DSpark 43 层完整执行，不以 1 层或裁剪模型结果代替；逐层输出和最终 token parity 通过。
- proposal：checkpoint-native proposal rows/token IDs/logits/confidence 与 direct oracle 一致，覆盖正常、空/边界上下文和 rollback。
- near-tie：构造 router top-k、proposal score 与 acceptance threshold 近似相等样本，验证稳定 tie-break、route order、
浮点容差和 kernel selection 不改变接受前缀或 correction/bonus 决策。
- acceptance：验证 accepted-prefix histogram、accepted/rejected draft、correction/bonus、rolled-back rows、
externally committed tokens 与 transaction commit/rollback 语义逐项一致，不只比较最终文本。
- status/route error：同步 can-implement/submit status、异步 status record、`route_written`、`route_error`、segment state、
slot generation mismatch 与 first failing route 的语义完全一致；任何非零错误不得被 adapter 映射为成功。
- 并发：普通多流与 graph concurrent exec 使用独立 workspace/cookie/ring slot，在错误注入下无交叉污染。
- 生命周期：所有 in-flight/graph exec 完成后才允许 plan/device/module/DSO 卸载，并通过 sanitizer/故障注入。
全部十算子、43 层与 proposal/near-tie/acceptance/error 门槛通过后，才能删除 ABI9 exports 与 adapter。
最终验收：model 不再直接调用 `crate::cutlass::*`；CUTLASS/cuda-oxide 只存在于 provider 实现和构建边界。

## 14. 官方参考

- NVIDIA CUDA GPU Compute Capability：<https://developer.nvidia.com/cuda-gpus>
- NVIDIA CUTLASS README：<https://github.com/NVIDIA/cutlass/blob/main/README.md>
- NVIDIA Blackwell Compatibility Guide：<https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html>
- NVIDIA CUTLASS Functionality：<https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html>
发布 provider 前必须重新核对官方资料与实际 `nvcc --list-gpu-code`；产品、CC、build toolkit、runtime driver、
PTX ISA 与 accelerated target 支持都必须作为独立事实记录，不得靠推测自动扩展。
