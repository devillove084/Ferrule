# Ferrule Kernel Provider C ABI

## 1. 状态、范围与目标

- 该边界是 Ferrule Rust 与同仓 C++/CUDA 实现之间的内部 C ABI；两侧始终从同一 checkout 构建并静态链接到同一最终程序。
- 它不是可独立分发或动态装载的插件协议，不提供跨构建、跨提交或跨发行版兼容承诺，也不进行数字版本握手。
- 任一边界变更必须在同一提交中同步更新 C/C++ 声明、Rust FFI 声明、布局断言、布局测试和调用点。
- 目标：Rust 以语义算子和 capability 请求选择、准备并调用 kernel provider，而不依赖 C++ 模板或具体 kernel 类型。
- 目标：kernel 实现、CUDA Toolkit、CUTLASS 和编译器差异隔离在 C++ 私有实现与构建路由内。
- 目标：stream、device memory、workspace、graph、event 与生命周期始终由 Rust host 控制。
- 非目标：稳定 C++ 模板、Rust 类型、CUDA Runtime API、provider 私有对象或跨构建序列化格式。
- 非目标：用单一万能 fatbin 覆盖全部 NVIDIA GPU。

公开路由使用语义名称，例如 `grouped_fp4_moe`、`hybrid_mla_attention` 和 `proposal_head`。
implementation architecture names 只能出现在 C++ 私有实现或构建/code-object 路由语境中；model、scheduler 和公开 Rust 路由不得依赖这类名称、C++ 符号、kernel 名或布局。
CUTLASS、CuTe 与手写 CUDA 也只能是 provider 内部实现细节。

## 2. 平台与布局规则

当前构建平台 contract 是 Linux ELF、little-endian、8-bit byte、LP64、自然 C 对齐；host 架构为 x86_64 与 aarch64。
其他平台必须作为新的同仓构建目标完整验证，不能据此文档推导二进制兼容性。
所有边界函数和函数指针使用 `extern "C"` 与平台默认 C calling convention；头文件统一使用 `FERRULE_ABI_CALL`。
ABI 边界只传 C POD、固定宽度整数、CUDA Driver API handle、pointer-length view 与 opaque handle。
不跨边界传递 Rust/C++ 对象、引用、trait object、异常、STL、bit-field、`long double` 或所有权智能指针。
布尔值与 wire enum 使用 `uint32_t`；尺寸、偏移、计数和 device byte address 使用 `uint64_t`。

该边界没有结构前缀兼容、尾部扩展协商或“尽量兼容”读取规则。C++ 和 Rust 必须使用本次构建的精确结构定义：
- 所有结构禁止 `#pragma pack`，字段顺序、类型与自然对齐必须显式一致。
- C++ 头文件对每个跨边界类型维护 `sizeof` 与关键字段 `offsetof` 的 `static_assert`。
- Rust layout tests 对同一组类型验证 `size_of`、alignment 和字段 offset，并与 C++ 导出的构建期常量逐项比较。
- 构建只允许一个强定义的 `ferrule_kernel_provider_api` 链接符号；缺失或重复定义必须使链接失败。

这些检查只保证同一次同仓构建的布局与入口唯一性，不构成动态兼容承诺。任何布局或语义变化都必须原子更新两侧，不能靠数字版本字段绕过。

## 3. 静态链接入口、基础类型与 host API

Rust 直接引用静态链接 C++ 库中的唯一入口：

```c
typedef struct ferrule_provider_api_t ferrule_provider_api_t;

const ferrule_provider_api_t* FERRULE_ABI_CALL
ferrule_kernel_provider_api(void);
```

入口返回进程生命周期内稳定的只读函数表，不执行 CUDA 工作，不得抛出异常、panic unwind 或跨边界 `longjmp`。
Rust 不扫描、装载或协商其他 provider；具体 C++ object libraries 与 code objects 由同仓构建路由决定。

```c
typedef int32_t ferrule_status_t;
typedef struct ferrule_provider_t* ferrule_provider_handle_t;
typedef struct ferrule_device_t* ferrule_device_handle_t;
typedef struct ferrule_plan_t* ferrule_plan_handle_t;
typedef struct ferrule_graph_plan_t* ferrule_graph_plan_handle_t;
typedef struct ferrule_graph_exec_lease_t* ferrule_graph_exec_lease_handle_t;

typedef struct {
    uint32_t major, minor, patch;
} ferrule_component_version_t;

typedef struct {
    const uint8_t* data;
    uint64_t len;
} ferrule_string_view_t;

typedef struct {
    const void* data;
    uint64_t count;
    uint32_t element_size, element_alignment;
} ferrule_array_view_t;

typedef struct {
    void* user_data;
    void* (FERRULE_ABI_CALL *alloc)(void*, uint64_t bytes, uint64_t alignment);
    void (FERRULE_ABI_CALL *free)(void*, void*, uint64_t bytes, uint64_t alignment);
} ferrule_host_allocator_t;

typedef struct {
    uint32_t flags;
    const ferrule_host_allocator_t* allocator;
    void* log_user_data;
    void (FERRULE_ABI_CALL *log)(void*, uint32_t, const uint8_t*, uint64_t);
} ferrule_host_api_t;
```

`ferrule_component_version_t` 只记录 toolkit、driver 或编译器 provenance/运行资格，不参与 Ferrule C ABI 握手。
log message 只在 callback 返回前有效；host callback 必须可被多个 provider 线程并发调用。
allocator pointer 与 callbacks 在 provider 存活期间稳定；alignment 必须是 2 的幂且至少为 pointer alignment。
provider 可以不用 host allocator，改用自身 allocator 管理仅内部可见的 host metadata。
若 provider 声明并启用 `CAP_HOST_ALLOCATOR`，其内部 host metadata 必须使用 host callbacks，并在父 handle 销毁前释放。
provider 不得把 allocator ownership 交还 caller，也不得在 handle 销毁后调用 callback。

## 4. Allocation、返回内存与所有权

禁止的是 caller-visible allocation 与 GPU 隐式 allocation，不是 provider 内部 host bookkeeping：
- provider 不得为 caller 分配或返回需要 caller `free` 的 array、string、blob、workspace 或 binding storage。
- provider 不得调用 `cudaMalloc*`、`cuMemAlloc*`、managed/pinned allocation，或内部创建 device scratch。
- provider 不得在 `launch`、capture 或 replay 中 lazy-load module、扩容 cache 或触发隐藏 workspace allocation。
- module/code load 只能发生在显式 `open_device`、warmup 或 `prepare` 阶段；不得把 driver-managed module 资源冒充 workspace。
- provider 可以为 provider/device/plan/graph plan 分配私有 host metadata；该内存不得包含 caller-owned buffer 的所有权。
- 序列化采用 size-query + caller-owned output buffer；不返回 provider-allocated blob。

所有返回 view 都是 immutable borrowed view：
- `ferrule_string_view_t` 的 `len` 是字节数，内容为 UTF-8、不要求 NUL，`data == NULL` 当且仅当 `len == 0`。
- `ferrule_array_view_t` 用 `count`、`element_size`、`element_alignment` 完整描述本次构建定义的数组元素。
- catalog 的 strings/arrays 由 provider handle 拥有，从成功 `get_catalog` 到 `destroy_provider` 前保持地址和内容稳定。
- plan selection、weight/workspace records 由 plan 拥有，从成功 `prepare` 到 `destroy_plan` 前保持稳定。
- graph records 由 graph plan 拥有，到最后一个 graph exec lease 释放且 graph plan 销毁前保持稳定。
- callback 参数只在 callback 内借用；launch 输入结构及其数组至少借用到函数返回，device memory 借用到异步完成。

caller 不得释放、修改或跨上述 lifetime 缓存 borrowed pointer；需要更长 lifetime 时必须复制。

## 5. 同仓 catalog 与 capability records

catalog 是静态链接 provider 对本次构建中可用语义算子、capability 和 code object 的运行时描述。
它用于 capability routing、可观测性和 cache fingerprint，不是插件 manifest，也不声明其他构建可读取当前布局。

```c
typedef uint32_t ferrule_semantic_operator_t;
enum {
    FERRULE_SEMANTIC_GROUPED_FP4_MOE,
    FERRULE_SEMANTIC_HYBRID_MLA_ATTENTION,
    FERRULE_SEMANTIC_PROPOSAL_HEAD,
};

typedef struct {
    uint64_t feature_id;
    ferrule_string_view_t canonical_name;
} ferrule_accelerated_feature_t;

typedef struct {
    uint64_t code_object_id;
    uint32_t kind, architecture_family; /* NATIVE_CUBIN or PTX; build routing only */
    uint16_t cc_min_major, cc_min_minor, cc_max_major, cc_max_minor;
    uint16_t target_cc_major, target_cc_minor;
    ferrule_string_view_t verified_target_name;
    ferrule_array_view_t accelerated_features;
    ferrule_component_version_t ptx_isa, min_driver, build_toolkit;
    uint8_t image_sha256[32];
    uint64_t image_offset, image_bytes;
} ferrule_code_object_record_t;

typedef struct {
    uint64_t record_id;
    ferrule_semantic_operator_t semantic_operator;
    ferrule_array_view_t code_object_ids, required_cap_words, optional_cap_words;
    ferrule_array_view_t dtype_contract_ids, layout_contract_ids, shape_contract_ids;
    uint64_t weight_layout_contract_id, workspace_contract_id;
} ferrule_operator_capability_record_t;

typedef struct {
    uint64_t provider_build_id_hi, provider_build_id_lo;
    uint8_t linked_image_sha256[32];
    ferrule_string_view_t provider_name, source_revision;
    ferrule_component_version_t build_toolkit, host_compiler;
    ferrule_array_view_t capabilities, code_objects, operators;
} ferrule_provider_catalog_t;
```

`verified_target_name` 只能来自构建所用 toolkit 的 `nvcc --list-gpu-code`，不能按设备营销名称猜测。
architecture family、CC 和 target name 仅用于 provider 私有实现选择、构建验证和 code-object 路由，不得成为语义算子名称。
`accelerated_features` 必须列出本次同仓构建共享定义的 Ferrule feature ID；空数组表示普通 target，不表示未知。
PTX record 必须声明 virtual target、PTX ISA 与 runtime minimum driver；native record 必须声明精确 native target。
CC range 只能描述实现候选，不能扩大 NVIDIA 对 cubin 或 accelerated code 的兼容规则。

operator record 是 capability routing 的最小单元：
- unknown required capability、缺失 required capability 或 required capability 与非 NULL 函数不一致时必须拒绝，不能猜测或降级为近似实现。
- unknown preferred capability 可以忽略，但最终 selection 必须如实记录实际启用的 capability。
- capability 用于表达可执行条件，不得被当作隐藏的 API 版本号。
- capability 定义由 C++ 与 Rust 的同仓共享声明生成；增删或改义必须与两侧调用点和测试原子更新。

建议基础能力：`ASYNC_ERROR_RING`、`GRAPH_CAPTURE`、`GRAPH_CONCURRENT_EXEC`、`PLAN_SERIALIZATION`、
`MULTI_STREAM_PLAN`、`WEIGHT_LAYOUT_QUERY`、`GENERIC_PTX_FALLBACK`、`HOST_ALLOCATOR`。

## 6. Provider API、opaque handle 与资源生命周期

```c
struct ferrule_provider_api_t {
    uint32_t flags;
    ferrule_status_t (FERRULE_ABI_CALL *create_provider)(const ferrule_host_api_t*, ferrule_provider_handle_t*);
    ferrule_status_t (FERRULE_ABI_CALL *destroy_provider)(ferrule_provider_handle_t);
    ferrule_status_t (FERRULE_ABI_CALL *get_catalog)(ferrule_provider_handle_t, ferrule_provider_catalog_t*);
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
};
```

所有 `void*` 实际指向本次同仓构建定义的具体 POD；正式头文件必须使用具体 typedef。
可选函数必须同时满足对应 capability bit 和非 NULL function pointer；二者矛盾是 provider 初始化错误，必须 fail closed。

生命周期偏序：

```text
linked provider code > provider > device/CUDA module > plan > graph plan > graph exec lease > in-flight replay
                                                      plan > ordinary in-flight launch
```

静态链接 provider code 与最终程序同寿命，不存在运行时卸载。
Rust 必须先停止新调用，再等待 caller-owned completion event，然后依次释放 graph exec、graph plan、plan、device 和 provider。
CUDA module 绝不能在 device、plan、in-flight launch、graph plan 或 `CUgraphExec` 仍引用代码时释放。
所有 destroy/close/release 都禁止同步；前置条件未满足返回 `FERRULE_STATUS_BUSY`，不得内部等待 GPU。
NULL destroy 可为 no-op；非 NULL handle 不得 double-destroy，失败销毁后 ownership 不转移。

## 7. Thread、CUDA context、reentrancy 与 destroy race

host 在调用 `open_device`、`prepare`、workspace init、launch、graph capture 和 CUDA-touching destroy 前，
必须在当前 OS thread 设置与 device handle 绑定的 `CUcontext`。
provider 不创建、切换、push/pop 或销毁 context，不依赖“首次调用线程”，handle 永久绑定 device UUID 与 context identity。
context 不匹配返回 `FERRULE_STATUS_CONTEXT_MISMATCH`，不得自动修复。
catalog query 与不同 handle 上的调用必须 thread-safe/reentrant；同一 immutable plan 是否可多流并发由 capability 决定。
同一 mutable graph plan 的 capture/register/release 操作由 Rust 串行化；普通 launch 不得修改 plan selection。
host allocator/log callback 可并发进入，但 callback 禁止同步回调 provider API，避免锁反转和递归重入。
provider 不得持锁调用耗时 host callback；不得在 API 返回或父 handle 销毁后继续使用 callback 参数。
destroy 与该 handle 的任何 API 调用不得并发；Rust 用 handle gate/refcount 阻止新调用并等待 CPU 调用退出。
GPU in-flight lifetime 由 Rust event/graph lease 跟踪；CPU 调用退出不代表 GPU 完成。

## 8. `prepare`、不可变 selection 与完整 plan cache key

`prepare` 只验证、选择、计算尺寸并创建 immutable plan，不启动 kernel，不绑定 stream 或瞬时 activation。
request 至少含 semantic operator、device UUID/context identity、shape、dtype、layout/stride、alignment、math mode、
required/preferred capabilities、weight descriptors、graph intent 和 determinism policy。

```c
typedef struct {
    uint64_t selected_operator_record_id, selected_code_object_id;
    ferrule_semantic_operator_t semantic_operator;
    uint32_t selected_code_kind;
    uint16_t selected_cc_major, selected_cc_minor;
    uint32_t uses_ptx_jit;
    ferrule_array_view_t enabled_cap_words, accelerated_feature_ids;
    ferrule_component_version_t runtime_min_driver;
    uint64_t selection_hash_hi, selection_hash_lo;
} ferrule_plan_selection_t;

typedef struct {
    uint64_t offset, bytes, alignment;
    uint32_t lifetime;    /* PER_LAUNCH, PLAN_INSTANCE, GRAPH_EXEC */
    uint32_t init_policy; /* NONE, ZERO_EACH, ZERO_ONCE, EXPLICIT_PROVIDER_INIT */
    uint64_t contract_id;
} ferrule_workspace_region_t;

typedef struct {
    uint64_t workspace_bytes, workspace_alignment;
    ferrule_array_view_t workspace_regions;
    uint64_t weight_layout_contract_id;
    ferrule_array_view_t weight_layout_records;
    ferrule_plan_selection_t selection;
    uint64_t plan_fingerprint_hi, plan_fingerprint_lo;
} ferrule_prepare_result_t;
```

selection 必须写入 plan 与 result；launch 不得按当时 driver、stream 或地址重新挑另一个 code object。
其中 architecture/CC/code-object 字段只用于记录 provider 已完成的实现路由，不改变 semantic operator identity。

完整 plan cache key 至少包含：platform contract、provider build ID、linked image hash、catalog hash、semantic operator、
operator record ID、code object ID/hash、accelerated feature IDs、PTX ISA/JIT 状态、runtime driver、device UUID/CC/context compatibility class、
enabled capabilities、shape、dtype、logical/physical layout、strides/alignment、math/determinism/graph flags、
weight layout contract、workspace contract，以及所有影响 kernel 选择的 tuning 参数。
若 plan 绑定权重地址，key 还必须含 allocation identity、address、bytes 与 content generation；否则只含 weight layout contract。
任一 key 分量变化都必须 cache miss；不得在不同 provider build、PTX JIT compiler/driver 或 device UUID 间复用 plan。
plan serialization 只允许恢复精确匹配的 build/fingerprint，并必须携带 endianness 与 checksum；任何不匹配返回 `SERIALIZATION_INCOMPATIBLE`。
该序列化能力不是跨构建数据格式承诺。

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
    uint64_t operand_id;
    uint32_t access_mode, dtype;
    CUdeviceptr address;
    uint64_t bytes;
    uint32_t rank;
    const int64_t *dims, *strides;
    uint64_t allocation_id, generation;
} ferrule_binding_t;

typedef struct {
    uint32_t flags, code, provider_detail;
    uint64_t launch_id, detail0, detail1, generation;
} ferrule_async_error_record_t;

typedef struct {
    uint32_t flags;
    const ferrule_binding_t* bindings;
    uint32_t binding_count;
    CUdeviceptr workspace;
    uint64_t workspace_bytes, workspace_instance_generation;
    CUstream stream;
    CUdeviceptr error_record;
    uint64_t error_record_bytes;
    CUevent completion_event;
    uint64_t launch_id;
} ferrule_launch_desc_t;
```

caller-owned bindings/workspace/error record/event 至少活到 completion；ownership 不转移。
provider 只向传入非默认 stream 排队，不创建 stream/event，不查询或同步 event，不做阻塞 D2H readback。
provider 禁止 host/device sync、GPU allocation、host-visible allocation、lazy init 和使用 legacy default stream。

同步返回码只描述验证与完整提交：`OK`、`INVALID_ARGUMENT`、`CAPABILITY_MISSING`、`UNSUPPORTED_DEVICE`、
`UNSUPPORTED_OPERATOR`、`UNSUPPORTED_LAYOUT`、`WORKSPACE_TOO_SMALL`、`MISALIGNED_ADDRESS`、`CONTEXT_MISMATCH`、
`CUDA_SUBMIT_FAILED`、`STALE_PLAN`、`SERIALIZATION_INCOMPATIBLE`、`BUSY`、`INTERNAL`。
状态码数值由同仓共享声明生成，不承诺独立构建之间保持数值兼容。
非零同步错误表示完整工作序列未提交；provider 私有错误只能进入 detail 或日志，不扩散私有 enum。
Rust 在同一 stream 提交前清零 error record，并设置 `generation` 与 `launch_id`。
kernel 用 first-writer-wins 原子协议写 `code`；完成前 host 不读取，完成后 Rust 同时收集 Driver API 异步错误。
provider 可以记录 caller-owned completion event，但 event 的创建、等待、复用和销毁完全属于 Rust。

## 10. Graph plan、launch cookie、error ring 与 replay

graph capture 是可选 capability，不能把普通 plan 直接视为 capture-safe。
`create_graph_plan` 在 capture 外完成 module load、lazy initialization、固定 kernel selection 和 graph workspace contract。
`capture_graph_launch` 只能在 Rust 已开始 capture 的 caller stream 中排队 capture-safe nodes，不得分配或 host callback。

```c
typedef struct {
    uint64_t replay_id, graph_exec_id;
    uint32_t error_slot;
    uint64_t error_generation, workspace_instance_generation;
} ferrule_graph_launch_cookie_t;

typedef struct {
    CUdeviceptr records;
    uint32_t record_count, record_stride;
    uint64_t record_bytes;
} ferrule_error_ring_desc_t;

typedef struct {
    CUgraphExec graph_exec;
    uint64_t graph_exec_id;
    CUdeviceptr launch_cookie;
    uint64_t launch_cookie_bytes;
    ferrule_error_ring_desc_t error_ring;
    CUdeviceptr workspace;
    uint64_t workspace_bytes, workspace_instance_generation;
} ferrule_graph_exec_desc_t;
```

每个 replay 必须有唯一 cookie 值与独占 error-ring `(slot, generation)`；graph node 通过固定 cookie address 读取它们。
Rust 在 `cuGraphLaunch` 前于同一 stream 更新 caller-owned cookie，清零目标 error slot，并在 launch 后记录 completion event。
slot 在对应 completion 前不得复用；旧 generation 的晚写必须被检测并不得覆盖新 replay 的错误。
同一 `CUgraphExec` 禁止重叠 launch；concurrent replay 使用不同 `CUgraphExec`、cookie、workspace instance 和 ring slot。
只有声明 `GRAPH_CONCURRENT_EXEC` 时，同一 immutable graph plan 才可支撑多个独立 graph exec 并发。
`register_graph_exec` 返回 lease，不取得 `CUgraphExec` ownership；Rust 先等待 replay，再销毁 `CUgraphExec`，最后 release lease。
graph exec lease 保持 graph plan、plan、device 和 CUDA module 活跃；存在 lease 时 destroy parent 必须返回 `BUSY`。
普通 async error 语义同样适用于 graph ring；每个 replay 的 route/status error 必须落到该 replay 自己的 slot。
plan serialization 可选，但 graph plan、graph exec lease、cookie address、event、workspace 和 error ring 都不可序列化。

## 11. Build Toolkit、runtime driver、PTX JIT 与硬件矩阵

本节中的 architecture 名称只描述 C++ 私有实现与构建/code-object 路由，绝不能用作 semantic provider 名称。
`build_toolkit` 是构建 provenance；`min_driver` 是运行资格；二者不得混为“最低 CUDA 版本”。
native cubin selection 检查 code-object target、CC/family、accelerated features 和该 image 的 runtime minimum driver。
PTX selection 还必须检查 virtual target、PTX ISA、driver JIT 支持与 JIT policy；JIT 结果及 driver 版本进入 plan cache key。
不能仅凭 build toolkit 推导 min driver，也不能因 driver 足够新就假定它接受未知 PTX ISA/accelerated target。
catalog 必须分别记录 linked image 的 build toolkit、每个 code object 的 min driver，以及 PTX record 的 PTX ISA。

以下是私有 build-routing profile 对应的 CUTLASS 最低构建 Toolkit，不是 runtime min driver 表：

| Build-routing profile | Compute Capability | CUTLASS 最低构建 CUDA Toolkit |
| --- | ---: | ---: |
| Compute capability 8.6 profile | 8.6 | 11.4 |
| Compute capability 9.0 profile | 9.0 | 11.8 |
| Compute capability 10.3 profile | 10.3 | 13.0 |
| Compute capability 12.0 profile | 12.0 | 12.8 |
| Compute capability 12.1 profile | 12.1 | 13.0 |

CUDA cubin 通常只在相同 CC major 且目标 minor 不高于设备 minor 时兼容；accelerated target 更严格。
`SM90a` 与 Blackwell architecture-accelerated code 不能跨 major/family；Hopper `90a` 不能用于 Blackwell。
`SM100` 与 `SM120` build-routing profiles 不兼容，`sm100a` code 不能在 `SM120` profile 上运行。
不得未经 toolkit 验证硬编码任一 build-routing profile 的 accelerated target 拼写。
构建必须执行并保存 `nvcc --version` 与 `nvcc --list-gpu-code`，catalog 只写 nvcc 实际支持的 target。
同仓构建应按目标设备选择私有 architecture/family object libraries 与 code objects，不构建巨大万能 fatbin。
generic PTX 仅是有显式 capability、正确性测试和性能遥测的 fallback，不能替代优化实现。
selection 顺序是精确优化实现、合法同 family cubin、合法 generic PTX fallback、明确不支持。

## 12. 验证要求

layout：
- C++ 对每个跨边界类型验证 `sizeof` 和关键 `offsetof`；Rust layout tests 验证对应 size/alignment/offset。
- Rust 测试读取 C++ 构建期布局常量逐项比较，不能只在单侧维护期望值。
- 构建必须引用唯一 `ferrule_kernel_provider_api` 强符号；缺失或重复 provider 定义必须在链接阶段失败。

routing/catalog：
- array element size、native/PTX/CC/feature/PTX ISA/min driver 组合必须可验证，selection 必须完整写入 plan。
- 错误或未知 required capability 必须在 module load/prepare 前被拒绝；不得调用近似语义算子或静默切换错误实现。
- `grouped_fp4_moe`、`hybrid_mla_attention`、`proposal_head` 必须始终以 semantic operator identity 路由，architecture 名称不得泄漏到 model/scheduler key。

runtime：
- allocation 测试拦截 allocator/CUDA API，证明 launch/capture/replay 无 host-visible/GPU/lazy allocation；内部 host metadata 全部按父 handle 回收。
- workspace/cache 覆盖零大小、错位、短 buffer、每次清零、一次初始化、persistent generation、跨流错误共享和完整 key miss。
- 每个同步码有确定触发；异步 first-writer、generation、status/route error、CUDA failure 与 event ordering 无 race。
- 覆盖不同 handle 并发、同 plan 多流 capability、错误 current context、callback 并发、destroy race 与 `BUSY` 行为。
- graph 测试覆盖 capture/instantiate/register/replay/release、cookie 更新、ring wrap/generation、不同 exec 并发和同 exec 重叠拒绝。
- serialization 的 checksum、build ID、driver/JIT、device UUID、selection 或 workspace contract 任一不符都必须明确 cache miss/拒绝。
- 每个私有 build-routing code object 必须在 catalog 声明的真实 GPU/driver/toolkit 组合运行，不能用模拟 CC 代替。

## 13. Semantic provider 与 fail-closed 路由

公开语义命名统一为：
- `grouped_fp4_moe`
- `hybrid_mla_attention`
- `proposal_head`

这些名称表达模型语义，不编码 GPU family、compute capability、CUTLASS specialization、tile shape 或实现类名。
对应的 implementation architecture names 只能存在于私有 C++ namespace/class、源文件、object-library target 或 provider 内部 build/code-object route 中。
它们不得出现在 model direct entry、scheduler 分支、公开 Rust semantic key 或持久化 semantic identity 中。

Rust routing 流程固定为：
1. 以 semantic operator、shape/dtype/layout、required/preferred capabilities 和执行策略构造 request。
2. provider 先验证所有 required capability、设备/driver/code-object 资格、weight/workspace contract 与 graph intent。
3. 只有完整满足语义与执行条件的 implementation candidate 才能进入 selection；选定结果固定写入 immutable plan。
4. 无精确合法候选、required capability 未知或任何 contract 不满足时，返回明确错误且不提交 GPU 工作。

不得按名称相似度、architecture suffix、未知 capability 或缺失字段猜测 fallback，也不得把非零 can-implement、submit、route 或异步错误映射为成功。
普通 launch 与 graph replay 都必须保持相同的 semantic operator、route ordering、first failing route 和 error-generation 语义。

端到端验证继续覆盖：
- 真实 DeepSeek-V4 proposal attachment 43 层完整执行，逐层输出和最终 token parity 通过，不以 1 层或裁剪模型代替。
- checkpoint-native proposal rows/token IDs/logits/confidence，覆盖正常、空/边界上下文和 rollback。
- router top-k、proposal score 与 acceptance threshold near-tie，验证稳定 tie-break、route order、浮点容差和 selection 不改变接受前缀或 correction/bonus 决策。
- accepted-prefix histogram、accepted/rejected draft、correction/bonus、rolled-back rows、externally committed tokens 与 transaction commit/rollback 语义逐项一致。
- 同步 can-implement/submit status、异步 status record、`route_written`、`route_error`、segment state、slot generation mismatch 与 first failing route 一致。
- 普通多流与 graph concurrent exec 使用独立 workspace/cookie/ring slot，在错误注入下无交叉污染。
- 所有 in-flight/graph exec 完成后才释放 plan/device/module，并通过 sanitizer 与故障注入验证生命周期。

最终边界要求：model 不直接调用 `crate::cutlass::*`；CUTLASS 只存在于 provider 私有实现和构建路由中。

## 14. 官方参考

- NVIDIA CUDA GPU Compute Capability：<https://developer.nvidia.com/cuda-gpus>
- NVIDIA CUTLASS README：<https://github.com/NVIDIA/cutlass/blob/main/README.md>
- NVIDIA Blackwell Compatibility Guide：<https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html>
- NVIDIA CUTLASS Functionality：<https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html>

更新 provider 私有实现或构建路由前必须重新核对官方资料与实际 `nvcc --list-gpu-code`；设备营销名称、CC、build toolkit、
runtime driver、PTX ISA 与 accelerated target 支持都必须作为独立事实记录，不得靠推测自动扩展。
