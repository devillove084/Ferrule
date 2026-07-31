# Ferrule roadmap

> 面向大于显存模型的统一调度、I/O、计算重叠，以及 Qwen3 / DeepSeek-V4 / 多后端 / 训练演进路线。
>
> Updated: 2026-07-31.

## 0. 当前结论

当前状态仍是 **GPU release NO-GO**，但原因已经不再是旧 roadmap 中单一的 GB10 `n8` wake blocker。

P0 的 owner-side 调度、single-flight materialization、continuation、generation 与资源回收协议已经完成 CPU UT / mock 验证；当前正在进行 P1：把模型 preparation 从“opaque resources + provider kernel plan”迁移为唯一的 resource-aware executable contract，并让 dense parameter bundles 与 routed experts 进入同一个 materialization / continuation / custody 协议。

目标执行路径固定为：

```text
prepared executable stages
    -> declare exact resource read/write/retain sets
    -> acquire residency leases
    -> missing resource:
         continuation + canonical DependencySet
         -> global single-flight materialization
         -> read/upload/install
         -> generation-qualified publication
         -> resume
    -> backend execution
    -> completion fence
    -> release/retain/evict
```

这条路径必须同时覆盖：

- dense parameter bundles；
- routed experts；
- KV state；
- activation checkpoints；
- gradients；
- optimizer states。

任何只适用于“全模型常驻显存”、只适用于 routed experts，或为某个 CUDA/CUTLASS provider 单独建立的执行旁路，都不再是目标架构。

## 1. 冻结原则与近期目标

| 项目 | 当前决定 |
|---|---|
| 核心能力 | 模型可以大于显存，I/O、调度和计算必须可重叠 |
| 生产执行路径 | 只保留一条 materialization / continuation / custody 路径 |
| 首个本地硬件目标 | NVIDIA RTX 3090，`sm_86` |
| 首个新增模型 | dense Qwen3；先做小模型 correctness，再做 8B out-of-core acceptance |
| 现有硬件路径 | 保留 DeepSeek-V4 `sm_121a` 的语义与 provider 能力，不建立 legacy/new 双路径 |
| 训练兼容 | inference / training 共用资源身份、访问、生命周期和 continuation 协议 |
| 后端边界 | CUDA、Metal、ROCm、Ascend 是 backend/provider，不进入模型总抽象 |
| 当前验证条件 | 本地无模型 checkpoint；只能使用 CPU UT、synthetic fixtures 和 mocks |
| GPU 结论 | 当前不能宣称 RTX 3090、GB10 或任何 CUDA 路径已验证 |

“Qwen3.8”不是明确的模型标识。当前建议：

1. Qwen3 dense 0.6B 或 1.7B：checkpoint、semantic、stage、CPU reference correctness；
2. Qwen3 8B：RTX 3090 上的 out-of-core acceptance target；
3. dense 路径稳定后，再评估 Qwen MoE。

## 2. 进度总览

| 工作流 | 状态 | 当前结论 |
|---|---|---|
| P0 调度 / I/O / completion correctness | CPU 已完成 | global single-flight、continuation exactness、generation publication、rollback/custody 已有 UT |
| P0 I/O / compute overlap accounting | CPU 已完成 | completion timestamp clock-domain 已修复，wait overlap UT 已补齐 |
| 双 broker 合并 | 已完成 | 唯一生产 broker 为 `PhysicalResourceBroker` |
| Prefetch 评估 | 已完成 | oracle 有收益；当前 score predictor 在槽位竞争下净负收益，不接生产 |
| Artifact 抽象拆除 | 已完成 | checkpoint 通用层与 DeepSeek 私有 binding 已拆分；不保留 compatibility alias |
| 通用 materialization identity | 已完成 | 六类资源统一使用完整 `MaterializationKey` |
| Generic backend naming | 已完成 | expert-only 通用命名与 topology / lease 旧名已清理 |
| Resource-aware executable contract | 进行中 | 类型和校验已开始落地，尚未完成 DeepSeek publication 迁移与全量验证 |
| Dense parameter bundle materialization | 进行中 | 正在按 embed / attention / router / FFN / output stage 建立 source-backed bundle |
| Routed stage 模板统一 | 待完成 | route 后 exact expert set 仍需与 prepared stage contract 直接对接 |
| Runtime mutable resources | 待完成 | KV / activation / gradient / optimizer 需在 continuation/transaction 实例化 exact capacity 与 generation |
| DeepSeek 公共组件拆分 | 已开始 | checkpoint、math、shape、RoPE、config helpers 已有公共层；仍需继续去除 family-private 重复 |
| Qwen3 model implementation | 未开始 | 已有部分 family classification / dense semantic 基础，但没有完整 checkpoint + runner |
| CUDA provider/profile 重构 | 未开始 | 先稳定 executable contract，再拆 target capability 与 provider 目录 |
| Ampere `sm_86` kernels | 未开始 | 不在 stage contract 稳定前建立 kernel 旁路 |
| DeepSeek `sm_121a` GPU 回归 | 未验证 | 当前环境不可验证 |
| Training execution | 未实现 | resource kind/access/retention 已提前纳入合约设计 |

## 3. P0：已完成的统一调度与 I/O 基线

### 3.1 Identity、continuation 与 single-flight

已经完成：

- 唯一 runtime-wide physical materialization registry；
- 唯一 `PhysicalResourceBroker`；
- canonical `DependencySet` 与 exact continuation waiter identity；
- source identity、content hash、payload encoding、backend、device、source generation、destination generation 全部进入 `MaterializationKey`；
- destination generation 只能来自真实 reservation，模型不能自行合成；
- generation-qualified completion / publication；
- cancel、stale、failure、shutdown 与 transaction rollback 的资源回收；
- 已提交 physical work 不会被逻辑 cancel 伪装成“没有发生”；
- runtime 不再依赖 CUDA 才能编译 owner-side protocol。

统一资源类型：

```rust
pub enum MaterializedResourceKind {
    Parameter,
    RoutedExpert,
    KvState,
    ActivationCheckpoint,
    Gradient,
    OptimizerState,
}
```

相同 `(group, item)` 的 parameter / gradient / optimizer state 依靠 resource kind 保持不同身份，便于后续训练按同一语义 slot 对齐状态。

### 3.2 Overlap、fairness 与资源回收

已经完成：

- I/O wait 与 compute overlap accounting；
- completion timestamp clock-domain 修复；
- wait overlap 的 CPU UT；
- fairness 与 hard-resource accounting；
- last-demand lease release / wake；
- continuation suspension / resume 的 exact dependency 校验；
- dead profiler、旧 attention kernel、旧 broker 和重复 runtime adapter 路径清理。

### 3.3 Prefetch 结论

`crates/ferrule-model/tests/prefetch_evaluation.rs` 的 mock 结论：

- oracle / future-aware prefetch 可以明显降低 selected miss；
- 当前 `ScoreBasedExpertPredictor` 在有限 residency slots 下会挤占真实 selected working set，净收益为负；
- 当前 predictor 不接生产；
- 保留 `Prefetch` resource class / demand reserve，等待 future-aware 信号，例如已知下一 stage、batch router 结果或可靠 draft information；
- 后续任何 predictor 必须同时通过 latency、selected miss、slot displacement、额外 I/O bytes 和 fairness 验收，不能只看命中率。

## 4. Checkpoint 与模型公共层

### 4.1 已完成

旧 `artifact` 总抽象已删除，通用 checkpoint 层为：

```text
crates/ferrule-model/src/checkpoint/
  encoding.rs
  hash.rs
  index.rs
  inventory.rs
  tensor.rs
  weight.rs
```

DeepSeek 私有语义 binding 位于：

```text
crates/ferrule-model/src/models/deepseek_v4/
  checkpoint.rs
  checkpoint_binding.rs
  local_checkpoint_tests.rs
```

已经拆出的公共能力包括：

- safetensors index / inventory / bounded tensor slices；
- checkpoint dtype / payload / matrix descriptors；
- generic linear weight formats 与 execution policy；
- dense HF tensor classification；
- GQA / dense MLP semantic layout；
- common checkpoint loading、shape validation、math、RMSNorm、RoPE / YaRN 和 config parsing helpers；
- generic `RouterWeights` 与 routed expert semantic identifiers。

DeepSeek MLA、compressor、indexer、hyper-connection 和 DSpark 逻辑保持 family-private，不为 Qwen 进行错误泛化。

### 4.2 当前进行中

正在把 routed-expert 私有 source hashing 提升为 checkpoint 公共 source identity builder：

```text
semantic tensor descriptors
  + ordered bundle domain/version
  + canonical source file snapshots
  + exact offsets
  -> content hash
  -> source identity
  -> source generation
```

要求：

- catalog 构建只读取文件 metadata，不读取大 payload；
- semantic content 与 physical source identity 分离；
- 文件替换、长度、mtime 或 inode/device 变化产生新的 source identity / generation；
- dense bundles 与 routed experts 共用同一构造协议；
- physical reader 在 publication 前重新验证 source snapshot。

## 5. P1：Resource-aware executable contract

### 5.1 目标边界

`ModelKernelPlan` 继续表示 provider/kernel launch plan，但不再承担跨硬件的资源生命周期抽象。

新的 model execution contract 负责：

- ordered semantic stages；
- exact resource read / write / read-write sets；
- lease retention lifetime；
- provider-neutral workspace claims；
- source-backed manifest；
- runtime-owned mutable resource declaration；
- deterministic canonical ordering 与交叉引用校验。

它不得包含：

- CUDA kernel ID；
- CUTLASS operation enum；
- persistent CUDA arena offset；
- stream / event handle；
- destination generation；
- `sm_86` / `sm_121a` 特定 layout。

### 5.2 当前已写入、尚待完成验证的类型

当前 WIP 位于：

```text
crates/ferrule-model/src/execution/resource.rs
crates/ferrule-model/src/execution/stage.rs
crates/ferrule-model/src/execution/plan.rs
crates/ferrule-model/src/checkpoint/source.rs
```

核心类型：

```rust
ResourceManifest
ResourceBacking::{Checkpoint, RuntimeOwned}
ResourceLayout::{CheckpointEncoded, TensorBundle, RuntimeBuffer}
ResourceAccess::{Read, ReadWrite, Write}
ResourceRetention::{ThroughStage, ThroughTransaction, Persistent}
StageResourceUse
WorkspaceClaim
ExecutableStage<O>
PreparedExecutable<O>
TransformerResourceSlot
TransformerStage
PreparedModel<R, O>
```

当前设计决定：

- `PreparedModel` 必须携带 `PreparedExecutable`；不允许 `Option`，不保留旧的空 plan 路径；
- checkpoint manifest 不携带 destination generation；
- runtime-owned mutable state 不把当前 spill/source generation 冻结进 immutable prepared plan；
- parameter / gradient / optimizer 使用同一 `TransformerResourceSlot` 坐标；
- routed experts 在 router 结果出来后实例化 exact selected resource set，不能在 prepare 时把所有候选专家声明成同时依赖；
- resource-free stage 与 empty executable 的策略由校验显式定义，不依赖隐含行为。

### 5.3 当前迁移状态

DeepSeek 正在从：

```text
PreparedModel {
    generation,
    opaque prepared resources,
}
```

迁移到：

```text
PreparedModel {
    generation,
    model-family resources,
    validated prepared executable,
}
```

计划中的静态 parameter bundles：

```text
prologue:
  embedding bundle

per layer:
  attention bundle
  router bundle
  shared/dense FFN bundle

runtime route result:
  exact selected routed expert resources

epilogue:
  output norm/head bundle

attachments:
  per-stage DSpark/MTP bundles
```

当前迁移尚未完成，因此这部分不能标记为编译通过或 production-ready。

### 5.4 P1 完成条件

- [ ] `PreparedModel` 全部调用面强制携带 validated executable；
- [ ] DeepSeek static dense bundles 全部生成 stable `ResourceManifest`；
- [ ] selected routed experts 由 prepared routed-stage template 实例化 exact resource uses；
- [ ] KV / activation / gradient / optimizer 由 request/transaction 创建 exact runtime manifests；
- [ ] runtime 从 stage contract 生成 canonical dependencies，不再从 model-private fields 推断；
- [ ] stage completion fence 驱动 release / retain / evict；
- [ ] rollback 保留 mutable state 与已提交 physical work 的真实 custody；
- [ ] 所有新增 CPU UT、model/runtime UT、workspace check 和 fmt 全绿。

## 6. P2：Qwen3 + RTX 3090 (`sm_86`)

在 P1 contract 稳定后推进 dense Qwen3，不先做 Qwen MoE。

### 6.1 Model/checkpoint

- [ ] 完整解析 Qwen3 config、tokenizer 与 safetensors inventory；
- [ ] 复用 common dense tensor classification、shape、RMSNorm、RoPE 和 SwiGLU；
- [ ] 建立 model-neutral GQA semantic plan；
- [ ] 每层拆成 attention / FFN parameter bundles；
- [ ] CPU reference forward 与 synthetic checkpoint fixtures；
- [ ] 验证 tied/untied embedding 与 output head；
- [ ] 明确 sliding window、RoPE scaling、KV dtype 与 sequence limits。

### 6.2 Out-of-core acceptance

Qwen3 8B 在 3090 上的目标不是“勉强加载成功”，而是验证唯一 stage path：

```text
materialize layer bundle N+1
    overlaps
execute layer bundle N
    overlaps
release/evict completed bundle N-1
```

必须证明：

- 不需要全模型 GPU 常驻；
- dense parameter miss 会 suspend continuation，而不是同步阻塞整个 owner；
- selected requests 保有公平性；
- bundle eviction 不破坏 in-flight completion fence；
- source/destination generation 与 resumed bindings 完整匹配；
- CPU/mock correctness 与 GPU output correctness 可对账。

### 6.3 初始 Ampere operator set

在 provider/profile API 完成后，RTX 3090 初始需求：

- BF16 GEMM；
- RMSNorm；
- RoPE；
- SwiGLU；
- GQA / paged attention；
- embedding / output projection；
- 必要的 layout conversion / packing。

在 executable/resource contract 完成前，不为 `sm_86` 建立模型私有 kernel 旁路。

## 7. P3：CUDA / CUTLASS provider 重构

CUTLASS 是 CUDA provider 的实现选择，不是 Metal / ROCm / Ascend 的跨硬件抽象。

目标组织：

```text
ferrule-cuda/
  runtime/              # CUDA context, streams, events, memory
  architecture/         # target/profile/capability detection
  providers/
    native/
    cutlass/
      sm121/
      ampere/
```

推进顺序：

1. [ ] provider-neutral capability / target profile API；
2. [ ] 将现有 `sm_121a` kernel catalog 收进明确的 provider/architecture 模块；
3. [ ] 去除 `build.rs` 中“CUTLASS 等于 GB10”的全局假设；
4. [ ] 添加 Ampere `sm_86` capability/profile；
5. [ ] 为 Qwen3 编写/接入 BF16 kernels；
6. [ ] 保持 DeepSeek `sm_121a` operation requirements 与 ABI 不变；
7. [ ] 分别做 `sm_86` 和 `sm_121a` compile/runtime acceptance。

迁移中不保留长期 legacy/new 双 provider 路径。需要分步移动文件时，每一步都必须保持唯一生产入口。

## 8. Training compatibility

训练不是在推理 runner 旁边再建立第二套 scheduler，而是在同一个 stage/resource 协议上增加 backward/update stages。

合约必须支持：

- parameter `Read`；
- KV / activation `ReadWrite`；
- gradient `Write` / accumulation `ReadWrite`；
- optimizer state `ReadWrite`；
- activation checkpoint `ThroughTransaction`；
- persistent parameters / optimizer shards；
- forward、backward、optimizer step 之间明确的 fence 与 retention；
- activation / gradient / optimizer spill 与 reload；
- data/tensor/pipeline parallel 后续对 resource custody 的扩展。

近期不实现完整预训练，但 P1 不允许引入以下只能服务 inference 的假设：

- 所有资源只读；
- stage 完成后所有 lease 都可立即释放；
- 权重永远全驻留；
- prepared source generation 可以代表 mutable training state；
- execution plan 只有 forward operation。

## 9. 验证状态与证据边界

### 9.1 当前全绿 CPU 基线

当前 P1-A executable / source-catalog migration 已实际通过：

```text
cargo test -p ferrule-common --lib
68 passed

cargo test -p ferrule-model
288 lib tests passed
local smoke: 1 passed, 1 ignored
prefetch mocks: 3 passed

cargo test -p ferrule-runtime
296 passed

cargo check --workspace --all-targets
passed

cargo fmt --all -- --check
passed
```

这些结果证明当前 CPU owner-side protocol、resource-aware `PreparedModel`、DeepSeek static executable/source catalog、prefetch mock 与 runtime baseline。它们不证明 CUDA static tensor-bundle transport/install 或任何 GPU 路径已经完成。

### 9.2 当前 WIP 状态

P1-A 的 contract、DeepSeek publication、checkpoint source catalog 与 CPU validation 已完成；P1-B 仍在进行。DeepSeek static bundle descriptor 已进入唯一 provider source catalog，但 CUDA 侧 generic pinned transport/install 尚未接通，runner 仍临时 eager compile/upload static image。因此还不能宣称 dense parameter bundle 已经实现真正 out-of-core 执行。

### 9.3 CUDA / GPU 边界

尝试启用 CUDA model feature 时，在 Rust/CUDA 编译前被既有 guard 阻止：

```text
CUTLASS is a GB10-only provider; set FERRULE_CUTLASS_ARCH=sm_121a
or build with --arch sm_121a
```

因此当前没有：

- CUDA feature compile validation；
- RTX 3090 `sm_86` compile/runtime validation；
- DeepSeek-V4 `sm_121a` 当前源码回归；
- Qwen3 GPU correctness / performance evidence。

历史 GB10 `n1`、旧 CUDA build 或旧 `n8` 调试记录只能作为历史信息，不能覆盖当前重构后的 release gate。

## 10. 近期执行顺序

### P1-A：完成 executable contract

- [x] 完成 DeepSeek `prepare_executable()` 与 publication migration；
- [x] 完成公共 checkpoint source catalog / identity builder；
- [x] 补齐 manifest / stage validation UT；
- [x] 恢复 `ferrule-model` 定向编译；
- [x] 跑全量 CPU baseline。

### P1-B：runtime 消费 stage contract

- [ ] stage -> exact materialization requests；
- [ ] missing set -> continuation + canonical dependencies；
- [ ] residency lease acquisition / completion fence / release；
- [ ] routed stage post-router exact instantiation；
- [ ] runtime-created mutable resources 与 transaction custody；
- [ ] 删除剩余 model-private residency 推断。

### P2-A：Qwen3 CPU correctness

- [ ] Qwen3 config/checkpoint；
- [ ] dense layer reusable components；
- [ ] synthetic checkpoint UT；
- [ ] stage bundle / source identity / continuation mocks；
- [ ] 0.6B 或 1.7B checkpoint acceptance（获得模型后）。

### P2-B：3090 provider 与 kernels

- [ ] CUDA target capability API；
- [ ] CUTLASS provider 按 architecture 收口；
- [ ] `sm_86` compile profile；
- [ ] Qwen3 BF16 operator set；
- [ ] Qwen3 8B out-of-core correctness；
- [ ] I/O / compute overlap、memory high-water 与 throughput evidence。

### P3：回归与扩展

- [ ] DeepSeek-V4 `sm_121a` compile/runtime regression；
- [ ] training forward/backward resource contract prototype；
- [ ] Metal / ROCm / Ascend backend capability design；
- [ ] future-aware prefetch signal 与新一轮 mock/production evaluation。

## 11. Release gate

在满足以下条件前保持 NO-GO：

- [ ] 唯一 stage/materialization/continuation production path；
- [ ] CPU UT / mock / workspace check 全绿；
- [ ] RTX 3090 `sm_86` 当前源码可编译；
- [ ] Qwen3 reference 与 GPU outputs 可对账；
- [ ] Qwen3 8B 能在 3090 上以 out-of-core 方式执行；
- [ ] timeline 证明真实 read/upload 与其他 ready compute 重叠；
- [ ] cancellation、stale、failure、rollback 后无 waiter/lease/credit/residency 泄漏；
- [ ] DeepSeek-V4 `sm_121a` 路径完成独立回归；
- [ ] 所有性能结论都有固定 workload、计数器和原始日志，不从 mock 或历史构建外推。
