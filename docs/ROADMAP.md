# Ferrule roadmap

> 面向大于单卡显存模型的统一执行系统：资源感知 stage、全局
> materialization、调度 / I/O / 上传 / 通信 / 计算重叠，以及推理、训练、
> post-training 和 RL 的共同基础。
>
> Updated: 2026-08-02.

## 0. 当前结论

Ferrule 当前仍是 **GPU release NO-GO**。

已经完成的核心方向不是某个模型或某张卡的专用推理路径，而是一条唯一、资源感知、
可挂起和恢复的 out-of-core 执行协议：

```text
prepared executable stages
    -> declare exact resource read/write/retain sets
    -> acquire residency leases
    -> missing resource:
         continuation + canonical DependencySet
         -> exact-key deduplicated materialization
         -> read/upload/install
         -> generation-qualified publication
         -> resume
    -> backend execution
    -> completion fence
    -> release/retain/evict
```

这条路径必须统一覆盖：

- dense parameters；
- routed experts；
- KV state；
- activation checkpoints；
- gradients；
- optimizer states；
- distributed communication buffers 与 collective / point-to-point completion；
- training、post-training 和 RL 中的多模型、多阶段资源。

近期 CUDA 硬件方向调整为 **compute capability 10.3 / `sm_103` profile 优先**。compute capability 8.6 /
`sm_86` profile、Qwen 和小模型仍是重要的 portability / correctness 目标，但不再是下一阶段的第一优先级。近期优先级是：

1. 收尾唯一 stage / materialization / continuation / custody / terminalization 协议；
2. 重构 CUDA architecture / capability 与 CUTLASS provider；
3. 建立 compute capability 10.3 / `sm_103` profile、kernel selection 和大模型可执行路径；
4. 建立 DP / TP / PP / EP / SP / CP 的拓扑、通信与资源协议；
5. 扩展到 training、post-training、RL；
6. 选择大型模型做端到端验收，候选包括 Kimi K3；
7. 后续回到 Qwen / Ampere `sm_86` profile 做可移植性与较小显存条件下的 out-of-core 验收。

任何只适用于“全模型常驻显存”、只适用于 routed experts、只适用于 inference，或为
某个 CUDA/CUTLASS architecture 建立的生产旁路，都不是目标架构。迁移可以分步，但
每一步只能有一个生产入口，不保留长期 legacy/new 双路径。

## 1. 冻结原则

| 项目 | 当前决定 |
|---|---|
| 核心能力 | 模型、训练状态或多模型工作集可以大于显存；I/O、上传、通信和计算必须可重叠 |
| 生产执行路径 | 只保留一条 resource-aware stage / materialization / continuation / custody 路径 |
| 近期硬件 | Compute capability 10.3 / `sm_103` profile；architecture ID 由 capability detection 确认，不在模型层硬编码 |
| 后续硬件 | Compute capability 8.6 / `sm_86` profile、现有 `sm_121a` profile、未来 Metal、ROCm、Ascend provider |
| 模型方向 | 先完善 provider、分布式和训练基础，再选择大型模型验收；Kimi K3 是候选，Qwen 保留为后续 portability track |
| 训练兼容 | inference / training 共用资源身份、访问、生命周期、continuation 和 terminalization 协议 |
| 后端边界 | CUDA、Metal、ROCm、Ascend 是 backend/provider，不进入模型通用 stage/resource contract |
| CUTLASS 边界 | CUTLASS 是 CUDA provider 的一种实现，不等于某个 architecture，也不是跨硬件抽象 |
| 物理正确性 | 已提交 physical work 不能被逻辑 cancellation 假装为未发生；必须等待或确认 quiescence |
| 失败处理 | terminal outcome exactly once；未确认 backend terminal 时完整保留 ownership |
| 验证条件 | 当前没有可用模型 checkpoint；GPU 环境正在变化，现阶段依赖 CPU UT、synthetic fixtures 和 mocks |
| 证据边界 | 已有 GPU 证据只适用于下述 `sm_103` profile build/runtime 和固定 workload；不能外推为 CC 12.1、`sm_121a`、`sm_86` 或其他 CUDA profile 已验证 |

## 2. 进度总览

| 工作流 | 状态 | 当前结论 |
|---|---|---|
| P0 runtime-wide materialization | CPU/mock 已完成 | exact-key physical dedup、canonical dependencies、generation publication、targeted wake 已覆盖 |
| P0 调度 / I/O / compute overlap | CPU/mock 已完成 | owner-side progression、fairness、hard resources、critical-path overlap accounting 已覆盖 |
| 双 broker 合并 | 已完成 | 唯一生产 broker 为 `PhysicalResourceBroker` |
| Artifact 总抽象拆除 | 已完成 | checkpoint 公共层与模型私有 semantic binding 已分离，不保留 compatibility alias |
| Prefetch 协议 | CPU/mock 已完成 | typed `PrefetchOwner`、exact phase demand、transaction-scoped declaration、Required join/reclassification 与 terminal cleanup 已覆盖；score-top-k 不猜测 |
| `ResolvedStage` contract | 已实现 | exact resources、access、retention、workspace、canonical dependency validation 已落地 |
| Typed materialization custody | 已实现 | stage / transaction / persistent ownership 与 provider lease 聚合释放已落地 |
| Routed expert stage migration | 已实现 | DeepSeek packed / proposal attachment route 后 exact expert set 进入唯一 resolved-stage 路径 |
| Resume lifecycle | 已实现 | `Consumed` / `StillActive` 区分，仍活跃 continuation 不提前释放 custody |
| Transaction terminalization | CPU/mock 已完成 inference 基线 | ordinary / speculative 共用 `end_transaction(Publish|Abort) -> Pending|Complete`；owner tick 自动推进，backend terminal 前完整保留 ownership |
| Typed runtime errors | 已实现 | runtime orchestration 以 SNAFU source chain 保留 backend、protocol、registry、resource 与 cleanup failures，不在内部字符串化 |
| Dense static bundle materialization | 未完成 | source catalog 已有；generic pinned transport/install 未接通，CUDA runner 仍有 eager static image |
| Runtime mutable resources | 待完成 | KV / activation / gradient / optimizer 需按 transaction 实例化 exact capacity、generation 和 spill source |
| DeepSeek 公共组件拆分 | 进行中 | checkpoint、math、shape、RoPE、config、dense semantics 已有公共层；继续去 family-private 重复 |
| CUDA capability/provider 重构 | 基线已完成 | `ferrule-cuda` 已删除并迁入 `ferrule-backend`；architecture/profile、provider-neutral plan、core/CUTLASS catalog 与唯一 selection 入口已建立，oxide 已删除 |
| `sm_103` profile provider/kernels | 可执行基线已验证 | compute capability 10.3 自动映射为 `sm_103`；标准 Cargo + NVCC release build、43 层 DeepSeek-V4 proposal attachment 固定生成已通过；性能优化与 backward catalog 未完成 |
| Distributed parallelism | 未开始 | DP / TP / PP / EP / SP / CP 需要统一 topology、communication stage、fence 和 custody |
| Training/post-training/RL | 未实现 | 通用 resource access/retention 已预留，execution/terminal protocol 尚需扩展 |
| Large-model acceptance | 未开始 | 候选 Kimi K3；最终选择取决于 checkpoint、operator、并行拓扑和许可可用性 |
| Qwen / `sm_86` portability | 后置 | 不删除；在核心 provider 和 distributed path 稳定后推进 |

## 3. P0：统一调度、I/O 与计算重叠基础

### 3.1 已完成的协议

- runtime-wide physical materialization registry；
- 唯一 `PhysicalResourceBroker`；
- canonical `DependencySet` 与 exact `WaiterId` / `ContinuationId`；
- source identity、content hash、encoding、backend、device、source generation、
  destination generation 全部进入 `MaterializationKey`；
- destination generation 由真实 provider reservation 提供，模型不能自行合成；
- registry 内部 physical dedup：相同 exact key 只提交一次物理 materialization；它不是调用方可见的 transaction/owner 抽象；
- generation-qualified completion / publication；
- cancel、stale、failure、shutdown 和 rollback cleanup；
- physical custody：已提交 read/upload/install 不能被逻辑 cancel 直接撤销；
- owner-side progress，不要求 CUDA 才能编译 runtime protocol；
- I/O wait 与其他 runnable compute 的 overlap accounting；
- exact `ExecutionPhaseSet`、`ModelWarmup / Prefetch / Required` demand、私有三段物理公平队列和 hard-resource accounting；
- completion timestamp clock-domain 一致性；
- 唯一生产 broker，旧 broker / adapter 路径已删除。

统一资源种类：

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

后续 communication buffer 是否成为独立 materialized resource kind，需要在 distributed
contract 中依据生命周期决定；不能为了目录整齐提前泛化。

### 3.2 `ResolvedStage` 与 exact custody

当前 model/runtime contract 已加入：

```rust
ResolvedStageResource {
    key: MaterializationKey,
    access: ResourceAccess,
    retention: ResourceRetention,
}

ResolvedStage {
    dependencies: Option<DependencySet>,
    resources: Box<[ResolvedStageResource]>,
    workspace: WorkspaceClaim,
}
```

约束：

- canonical ordering；
- request / key / semantic resource identity exact matching；
- duplicate key fail closed；
- resource-free stage 不制造虚假的 materialization dependency；
- `DependencySet` 只描述“什么阻塞进度”；
- `ResolvedStageResource` 描述 access 与 custody lifetime；
- materialization wait 只能通过 retention-aware resolved stage 创建，不能用裸
  `DependencySet<ResourceResident>` 绕过 custody metadata。

runtime custody owner 已区分：

```rust
Stage(ContinuationId)
Transaction(ExecutionTransactionId)
Persistent(MaterializationKey)
```

provider execution lease 只有在最后一个聚合 owner 消失后才进入 release pending；物理
lease release 失败可以重试，但不能重复执行逻辑 owner decrement。

### 3.3 当前 P0 收尾状态

已连接：

- stage completion 与 stage custody；
- transaction publish / abort 与 transaction custody；
- persistent custody 的显式 retirement；
- resume `StillActive` 对 continuation 与 stage custody 的保留；
- ordinary / speculative 共用 backend terminal contract；
- backend `Pending` 时完整 ownership quarantine；
- owner tick 自动推进 terminalization，不依赖第二次 `cancel_request()`；
- fatal backend/CUDA error 直接上报 worker，不做 device quarantine/recovery；
- typed Prefetch owner、Required join/reclassification、transaction terminal cleanup 与 cancellation compensation。

`LoadRegistry` 内的 physical dedup 只保证相同 exact key 不重复发 I/O；它不再被暴露成
“single-flight owner”或额外 transaction 状态。

## 4. Transaction terminalization：保留严格语义，删除分散复杂度

### 4.1 为什么不能删除 rollback correctness

Ferrule 不是同步、全模型常驻显存、失败即退出的执行器。一个 transaction 可能同时拥有：

- 已提交的 storage read / H2D / install；
- 正在使用 resident parameters / experts 的 backend execution；
- continuation；
- provisional KV reservations / prepared commit；
- model branch state；
- suspended scheduler/session ownership；
- activation、gradient、optimizer 或 communication buffers。

因此失败后的安全顺序必须是：

```text
request commit / rollback / cancel
    -> quiesce or wait for physical/backend work
    -> confirm backend terminal state
    -> publish commit OR discard provisional generations
    -> release transaction custody
    -> retire KV/state/materialization/communication resources
    -> restore or finish scheduler/session ownership
```

核心不变量：

> 没有确认物理工作终止，就不能因为逻辑 cancellation 而释放它可能仍在访问的资源。

### 4.2 当前实现

backend contract 已收敛为唯一入口：

```rust
enum TransactionEndIntent {
    Publish,
    Abort,
}

enum TransactionEndProgress {
    Pending,
    Complete,
}

fn end_transaction(
    transaction: ExecutionTransactionId,
    states: &mut [SequenceState],
    intent: TransactionEndIntent,
) -> Result<TransactionEndProgress>;
```

约束：

- `Pending` 是正常控制结果，driver 每个 tick 自动重试；
- `Complete` 是释放 KV、branch、materialization custody 和 session ownership 的唯一前提；
- `Err` 是 fatal backend/protocol failure，不伪装为 retryable pending；
- backend 未完成时，ordinary/speculative 都完整保留各自 payload ownership；
- cleanup failure 以 typed SNAFU error tree 保留主失败与 cleanup source，不拼接字符串。

### 4.3 最小 transaction 形状

不再建立一个覆盖所有 workload 的巨大 `TransactionState`。ordinary、speculative、training
和 RL 的 payload 本来就不同；强行统一其内部阶段只会增加镜像状态。共同协议只保留：

1. stable `ExecutionTransactionId`；
2. payload 精确持有所有 provisional ownership；
3. 唯一 `Publish / Abort` backend 终结意图；
4. 唯一 `Pending / Complete` 物理进度；
5. `Complete` 后按固定顺序执行 logical publication/reclaim。

当前 ordinary payload 仅使用：

```text
Executing
Publishing { output }
Aborting { request, custody, continuation, failure }
```

speculative payload 仅使用：

```text
Proposing
Verifying
Ending { Publish | Abort }
```

这两者不是两套 terminal protocol：它们只是不同 workload 的必要 payload phase，最终都调用
同一个 `end_transaction()`，并遵守同一 cleanup 顺序。禁止重新加入：

- `pending / rollback_pending / terminal` 三个互斥 `Option`；
- speculative 专属 rollback wrapper；
- driver/executor 双 transaction registry；
- `backend_terminal / custody_finished / cleanup_finished` 平行布尔状态；
- 通过再次调用 `cancel_request()` 推进 terminalization。

### 4.4 Backend terminal contract

backend 只返回：

- `Pending`：物理工作尚未 quiesce，payload 原样留在 driver；
- `Complete`：backend 保证不再访问 transaction resources；
- `Err`：fatal backend/protocol failure，直接上报并终止 worker/service。

本系统不实现 device-loss 恢复、device quarantine 或字符串识别 CUDA fatal error。硬件/context
级 failure 不是 transaction rollback 的可恢复分支。completion fence 可以是 backend 实现
`Pending / Complete` 的内部证据，不需要扩张为 runtime 的另一套公开状态。

### 4.5 用 generation publication 简化 rollback

mutable state 优先采用 provisional generation，而不是原地写后执行逆操作：

```text
committed generation N
    -> create provisional N+1
    -> execute/read/write N+1
    -> commit: atomically publish N+1
    -> rollback: quiesce then discard unpublished N+1
```

适用于 KV、activation、gradient shard、optimizer shard、checkpoint manifest 和后续
communication epoch。对于无法完整复制的大 optimizer state，评估：

- shard-level generation；
- copy-on-write extents；
- write-ahead journal；
- step-boundary atomic manifest publication。

目标不是实现昂贵的“逆向恢复所有字节”，而是在 quiescence 后丢弃未发布 generation。

### 4.6 Terminalization 完成条件

- [x] ordinary / speculative 共用唯一 backend terminal contract；
- [x] 移除多个互斥 `Option<pending/rollback/terminal>`；
- [x] backend terminal progress typed 为 `Pending / Complete`；
- [x] retry 由 owner tick 自动驱动；
- [x] fatal backend/device error 直接上报，不建立恢复状态机；
- [x] publish / abort exactly once；
- [x] backend terminal 后 cleanup error 保留 typed source；
- [x] CPU failure-injection 覆盖 quiescence 前 ownership quarantine；
- [ ] training / RL payload 接入同一 `end_transaction()` contract；
- [ ] 真实 CUDA fence 与 fatal-error integration evidence。

## 5. Checkpoint、模型公共层与 DeepSeek 拆分

### 5.1 已完成

旧 `artifact` 总抽象已删除。通用 checkpoint 能力位于：

```text
crates/ferrule-model/src/checkpoint/
  encoding.rs
  hash.rs
  index.rs
  inventory.rs
  read_plan.rs
  source.rs
  tensor.rs
  weight.rs
```

已拆出的公共能力：

- safetensors index / inventory / bounded reads / positioned read plan；
- source snapshot、source generation、content hash；
- checkpoint dtype、payload、matrix descriptor；
- generic linear weight formats 与 execution policy；
- dense HF tensor classification；
- GQA / dense MLP semantic layout；
- common shape validation、math、RMSNorm、RoPE / YaRN、config helpers；
- generic `RouterWeights` 与 routed expert semantic IDs；
- model-neutral resource manifest、stage、access、retention、workspace contract。

DeepSeek MLA、compressor、indexer、hyper-connection、proposal attachment 等能力只有在证明跨模型语义
一致后才提升为公共组件，不能因为 Qwen 或其他模型“看起来相似”而错误泛化。

### 5.2 下一步拆分原则

接入新模型前继续审查 `models/deepseek_v4`：

1. semantic primitive 与 family policy 分离；
2. checkpoint naming/binding 与 tensor operation 分离；
3. attention/MLP/router 的通用 shape contract 与 DeepSeek 特有布局分离；
4. model-neutral execution stage 不携带 CUDA/CUTLASS IDs；
5. provider-specific packing/layout 留在 backend/provider；
6. 复用必须减少重复并保持语义清晰，不创建“大而空”的 common 模块。

### 5.3 Dense static bundle 缺口

DeepSeek static bundle descriptor 已进入 provider source catalog，但真正的唯一 out-of-core
path 尚未闭环：

```text
checkpoint extent
    -> generic pinned read transport
    -> provider-specific packing/install
    -> generation-qualified residency binding
    -> resolved stage resume
```

当前 CUDA runner 仍临时 eager compile/upload static image。在上述通用 transport/install
完成前，不能宣称 dense parameter bundle 已支持 out-of-core，也不能删除 eager path 后留下
不可执行模型。迁移完成时应直接切换唯一生产入口，不保留长期双路径。

## 6. CUDA architecture 与 provider 重构

### 6.1 设计边界

`crates/ferrule-backend` 已按以下边界承接并拆开原 `ferrule-cuda`：

1. CUDA runtime：context、stream、event、memory、fence；
2. architecture profile：实际设备、编译 target、capabilities；
3. operation semantic requirements：dtype、shape、layout、forward/backward、determinism；
4. provider：CUTLASS、native CUDA 或未来其他 CUDA 实现；
5. kernel selection：capability + operation requirements -> implementation；
6. checkpoint transport / packing；
7. install authority 与 destination generation；
8. workspace planning；
9. communication integration；
10. observability、fallback reason 和 unsupported diagnostics。

CUTLASS 不得成为：

- 某个 CUDA target/profile 的同义词；
- 某个 SM ID 的全局 feature gate；
- 模型资源身份的一部分；
- Metal / ROCm / Ascend 的抽象基类；
- 只支持 forward inference 的 operation catalog。

### 6.2 方向性组织

当前唯一生产组织为：

```text
ferrule-backend/
  architecture_target.rs
  src/
    plan.rs
    cuda/{runtime,architecture,provider,cutlass,...}
  native/cuda/
    abi/{core_provider,cutlass_provider}.h
    providers/{core,cutlass}/
```

要求：

- 只有一个 production provider selection 入口；
- architecture specialization 在 provider 内部，不复制模型 runner；
- `sm_121a` 与 `sm_103` 共用 provider-neutral semantic requirements，架构 specialization 留在 provider；
- 原 `ferrule-cuda` 和 oxide build path 已删除，不保留 legacy/new 双 provider；
- compute capability 10.3 被字面映射为 `sm_103`，不凭设备名称推断 `a/f` suffix；
- Ampere、Blackwell/Blackwell Ultra 等 architecture profiles 可共用 semantic requirements，但 kernel
  implementation 和 workspace 可以不同；
- forward/backward capability 必须显式，不允许把缺少 backward 的 kernel 误选入训练计划。

### 6.3 近期推进顺序

1. [x] 删除 `ferrule-cuda`，建立 `ferrule-backend` runtime/architecture/provider 边界；
2. [x] 定义 CUDA target/capability query，不绑定模型 family；
3. [x] 定义 provider-neutral operation requirements 与 executable plan；
4. [x] CUTLASS/core kernels 迁入唯一 provider selection；
5. [x] 去除“CUTLASS == 某个 CUDA profile / `sm_121a`”和 oxide build path；
6. [x] compute capability 10.3 / `sm_103` build/runtime profile；
7. [x] routed-expert checkpoint read -> pinned transport -> install；
8. [x] DeepSeek-V4 proposal attachment forward operator executable baseline；
9. [ ] generic dense/static checkpoint transport；
10. [ ] backward / gradient / optimizer operator catalog；
11. [ ] `sm_86` profile、Qwen BF16 operator set，以及 `sm_121a`/`sm_86` 独立回归。

## 7. Distributed parallelism：DP / TP / PP / EP / SP / CP

并行策略不能只是 runner 外层的启动参数。它们必须进入 prepared topology、stage resource、
communication、fence、transaction 和 failure protocol。

### 7.1 统一拓扑 contract

需要显式描述：

- global rank、local rank、device、node；
- process/device mesh 与维度命名；
- DP / TP / PP / EP / SP / CP group；
- model stage / tensor / expert / sequence shard placement；
- checkpoint shard source 与 destination placement；
- communicator identity、generation 和 membership epoch；
- topology change / rank failure 的 terminal policy。

模型层声明 semantic sharding requirements；runtime 编排 topology；backend/provider 实现
collective 和 kernel。模型 stage 不直接持有 NCCL handle。

### 7.2 各并行维度

- **DP**：parameter replica、gradient reduction、optimizer sharding、batch/rollout placement；
- **TP**：tensor shard、all-reduce / reduce-scatter / all-gather 与 fused kernel boundary；
- **PP**：microbatch schedule、activation transfer、bubble、stage checkpoint 与 failure boundary；
- **EP**：router result、expert dispatch/combine、expert placement 与 out-of-core expert residency；
- **SP**：sequence shard、activation/KV ownership与 sequence collective；
- **CP**：context partition、attention communication、KV/context exchange 与长上下文 memory plan。

组合并行必须通过同一 mesh 表达，不能为每个组合手写 runner 分支。

### 7.3 Communication stage/resource/fence

需要纳入 stage graph：

- collective read/write sets；
- point-to-point pipeline transfers；
- expert dispatch / combine；
- sequence/context exchange；
- communication workspace；
- communication completion fence；
- communicator/rank epoch；
- materialization、communication 和 compute 的 overlap；
- cancel/rollback 时 in-flight collective 的 quiescence 与 device-level failure propagation。

通信完成不能用 CPU enqueue completion 代替真实 device/network fence。

### 7.4 分布式完成条件

- [ ] topology 与 group identity 可验证、可序列化、generation-qualified；
- [ ] 至少一个 selected large model 有明确 parallel plan；
- [ ] 选定模式的 collective / P2P stage 进入统一 scheduler；
- [ ] parameter/expert/KV placement 与 materialization 一致；
- [ ] rank-local failure 不产生跨 rank custody 泄漏；
- [ ] timeline 能展示 I/O、communication、compute overlap；
- [ ] DP/TP/PP/EP/SP/CP 单独 correctness 后再验证组合模式。

## 8. Training、post-training 与 RL

训练不是在 inference runner 旁建立第二套 scheduler。它是在同一 prepared stage/resource /
transaction protocol 上增加 backward、update 和多模型协调。

### 8.1 Training execution graph

统一 stage graph 必须支持：

```text
forward
    -> loss
    -> backward
    -> gradient reduction/accumulation
    -> optimizer update
    -> checkpoint/publication
```

资源语义：

- parameter `Read`，更新时按新 generation publication；
- activation / activation checkpoint `Write`、`Read`、`ReadWrite`；
- gradient `Write` / accumulation `ReadWrite`；
- optimizer state `ReadWrite`；
- forward/backward/update 之间 exact fence 与 retention；
- activation、gradient、optimizer spill/reload；
- recomputation 与 checkpoint retention trade-off；
- mixed precision、master weights、loss scaling；
- deterministic / non-deterministic kernel capability；
- distributed gradient / optimizer ownership。

禁止引入：

- “所有资源只读”；
- “stage 完成后所有 lease 都释放”；
- “权重永远常驻”；
- “prepared source generation 等于 mutable training generation”；
- “execution plan 只有 forward”；
- “rollback 等于恢复一份完整显存副本”。

### 8.2 Post-training

后续覆盖：

- SFT；
- preference optimization；
- reward model / value model training；
- LoRA / adapter 等 parameter-efficient updates；
- quantization-aware 或蒸馏类流程；
- 多 checkpoint generation 的原子发布与回收。

具体算法不应写死在 runtime；runtime 提供 stage graph、resource custody、distributed topology、
terminalization 和 observability。

### 8.3 RL / rollout-learner

RL 系统至少需要协调：

- policy model；
- reference model；
- reward model；
- value/critic model（若算法需要）；
- rollout generation；
- trajectory / token / logprob / reward buffers；
- actor、learner、evaluator 的异步调度；
- checkpoint generation 发布与 worker refresh；
- stale policy/version policy；
- 多模型共享或竞争 GPU/CPU/storage/network resources；
- rollout cancel、learner failure 和 checkpoint publication rollback。

多模型资源工作集很可能远大于显存，因此 RL 不是 out-of-core 调度之外的独立功能，而是
其重要验收场景。

## 9. 模型路线

### 9.1 大模型优先验收

compute capability 10.3 / `sm_103` execution environment 可用后，优先选择能验证以下能力的大模型：

- dense + MoE 或至少包含复杂 routed experts；
- 多卡并行需求；
- checkpoint 规模足以验证 out-of-core；
- operator 和精度路径可在目标硬件实现；
- checkpoint、tokenizer、config、许可和参考输出可获得。

Kimi K3 是候选，但在获得并确认其实际 checkpoint/config/operator requirements 前，不在
roadmap 中编造具体架构。模型选择应依据：

1. 可获得、可重复的 checkpoint 与 reference；
2. `sm_103` profile operator feasibility；
3. TP/PP/EP 等并行验收价值；
4. out-of-core materialization 压力；
5. training/post-training 后续价值；
6. 固定 workload 与性能证据可复现性。

### 9.2 Qwen 与 `sm_86` portability track

Qwen 不删除，只后移：

- 复用 DeepSeek 拆出的 common checkpoint/dense/GQA/RMSNorm/RoPE/SwiGLU 组件；
- synthetic checkpoint 与 CPU reference correctness；
- 在 provider capability API 完成后添加 Ampere `sm_86`；
- 选择明确版本和规模，不使用含义不清的“Qwen3.8”名称；
- 小模型验证 checkpoint/semantic correctness；
- 大于可用显存的模型验证 `sm_86` profile out-of-core；
- 不为 Qwen 或 `sm_86` profile 建模型私有 kernel / scheduler 旁路。

`sm_86` profile acceptance 仍应证明：

```text
materialize bundle N+1
    overlaps
execute bundle N
    overlaps
release/evict bundle N-1 after completion fence
```

## 10. Prefetch

Prefetch transport 已实现，资源语义、阻塞性和物理排队策略彼此分离：

```rust
PrefetchOwner::{ModelWarmup, Transaction { transaction, phases }, External(id)}
ResourceDemand::{ModelWarmup, Prefetch(phases), Required(phases)}
FairQueueBand::{Background, Prefetch, Required}
```

- `ExecutionPhaseSet` 精确保留 `MultimodalEncode / Prefill / Decode /
  SpeculativeProposal / SpeculativeVerification / TrainingForward /
  TrainingBackward / OptimizerUpdate / RolloutGeneration / RewardEvaluation`，混合 batch
  使用 phase union，不制造 `BatchExecution` 或 `TokenExecution` 融合状态；
- `ResourceDemand` 只表达是否阻塞物理执行；它没有 `Ord`，合并必须保留 exact phase union，
  并只允许 `Prefetch -> Required`；
- `FairQueueBand` 仅是 registry 私有物理策略，不泄漏为模型或 transaction 状态；
- registry 不抢占已提交 physical custody，并由 workload owner 显式控制 warmup admission：startup
  只提交一个小的初始物理 wave 后立即允许请求进入；foreground 不 reserve 新的 ModelWarmup
  operation，但已 reserve/submitted 的 read/upload/install 必须继续排空；idle/request gap 使用普通
  `drive()` 全速填充规划 residency high-water；
- execution reserve 按模型 top-k required wave 推导，ModelWarmup/Prefetch 不能侵占；foreground
  exact Required miss 始终推进，不需要第二 executor 或 full-fit/partial-fit 执行分支；
- model residency planner 以静态权重加载后的实时 free VRAM、显式 dynamic reserve 和每层 frame
  大小计算高水位；full-fit 最终填满全部 expert，partial-fit 只声明规划 slot 数，避免无界 eviction；
- `prepare_prefetch_request / prefetch / cancel_prefetch` 是 external/warmup 的显式入口；
- transaction prefetch 在 backend activation 前由 runner 声明 exact requests，复用唯一
  resolver/provider/registry，不创建 waiter、continuation 或 execution lease；
- DeepSeek hash router 可从已知 token 精确声明 target expert；score-top-k 在 router 结果未知时
  不猜 expert，也不制造 correctness barrier；
- Required waiter 可 join 同 exact key/generation/slot 的已提交 operation，不重发 I/O；provider
  execution lease 只在 hard admission 成功后获取；
- owner 离开后 operation demand 从剩余 Prefetch owner 和 Required waiter 重新计算，不能盲目降级；
- transaction-scoped Prefetch 只在 backend terminal 完成后随
  `finish_transaction_custody()` 自动释放，不依赖第二次 cancel；
- physical dedup 只是 registry 内部实现，不是 Prefetch transaction 或额外 lifecycle；
- `execution_reserve` 仅可由 `Required` 使用；Prefetch 没有 aging forced-progress entitlement；
- 已提交 read/upload/install 即使失去所有逻辑 owner，也必须完成 cancellation/drain 后才能释放
  physical custody。

full-fit 下不启用 predictive expert prefetch：全部 immutable resource 已由 idle warmup 最终常驻，
冷请求只能在 router 产生 exact identity 后走 Required，预测既不改变 correctness，也会与 compute
争用 H2D/install。partial-fit 下，oracle / future-aware prefetch 对 selected miss 有明显收益，
但当前 `ScoreBasedExpertPredictor` 会挤占真实 working set，端到端净收益为负，因此不接生产。
后续只评估有因果依据的 future-aware 信号，例如 prepared next stage、router result、pipeline
schedule、known backward stage、rollout schedule 或可靠 draft information。

任何新 predictor 必须同时报告：

- selected miss；
- end-to-end latency / throughput；
- slot displacement；
- extra storage/H2D bytes；
- wasted install；
- execution fairness 与 Prefetch displacement；
- memory high-water；
- 在固定 workload 下相对 no-prefetch baseline 的净收益。

## 11. Observability 与验证策略

### 11.1 当前 CPU/mock 证据

截至本次更新，本轮已实际运行：

```text

cargo fmt --all -- --check
passed

cargo check --workspace --all-targets --offline
passed

cargo test -p ferrule-runtime --lib --offline
317 passed

cargo test -p ferrule-model --lib --offline
301 passed

cargo test -p ferrule-server --lib --offline
24 passed
```

当前机器自动探测到 compute capability 10.3，并字面映射为 `sm_103`；architecture suffix 不从设备名称或 major version
推断。CUDA backend 已迁入 `ferrule-backend`，标准 Cargo + NVCC 构建不依赖 oxide。

实际验证：

```text
just build-cuda sm_103
passed without warnings

DeepSeek-V4-Flash-0731, 43 target layers, 3 proposal attachment stages, max_tokens=6
generated_token_ids = [7249, 17, 40897, 17, 40897, 68082]
cycles=4, proposed=12, accepted_draft=2, correction=3
rolled_back_rows=10, verified_rows=16, externally_committed_tokens=6
```

background residency 已验证可填到 planner high-water。模型初始化后只提交小的 startup wave
便允许请求进入；foreground 不 reserve 新 warmup，但已经进入物理流水线的工作必须 drain；idle
owner 再全速填满规划 residency。`sm_103` profile A/B 固定请求证明前台持续注入 background H2D/install 会把
`decode_seconds` 从 `21.26s`（resident `1886`，uncovered wait `10.21s`）恶化到 `40.09s`
（resident `8346`，uncovered wait `8.13s`）。因此吞吐更高的 resident 数不能抵消 compute contention；
最终策略是 foreground reserve suppression，而不是持续 background fill。正确性已经通过，但性能仍
未达到 vLLM/SGLang release gate。

新增 mock 覆盖包括：

- resolved stage canonicalization 与 fail-closed identity validation；
- stage / transaction / persistent custody；
- shared key 最后 owner 才释放；
- `StillActive` resume 不提前释放；
- transaction commit / rollback terminal release；
- speculative terminal propagation；
- owner tick 自动推进 pending abort；
- backend `Pending` 保留完整 ownership，后续自动推进并 exactly-once cleanup；
- typed Prefetch owner、exact phase merge、Required join/reclassification/cancel 与 compensation；
- transaction exact prefetch 不创建 waiter/continuation，并在 terminal custody 后自动释放；
- submitted physical work 在逻辑 owner 消失后继续 drain；
- foreground 只禁止新的 ModelWarmup reserve，已经 reserve/submitted 的 warmup read/upload/install
  继续到 Resident，尚未 reserve 的 operation 保持 queued；
- execution reserve 阻止 Prefetch，但不阻止 Required execution；
- initial residency 异步填充 high-water，hard Required wave reserve 不被 ModelWarmup/Prefetch 侵占；
- exact request terminal 与 model background lifecycle 解耦；
- typed backend/registry/resource/cleanup source chain。

CPU/mock 结果证明 protocol；上面的 `sm_103` build 与固定 DeepSeek 运行是独立 GPU 证据，
尚不等同于完整 kernel/reference、timeline 或性能 release gate。

### 11.2 必须补齐的 trace

统一 timeline 至少需要：

- stage ready / blocked / resumed；
- materialization read/upload/install；
- compute launch / completion fence；
- communication enqueue / completion；
- transaction terminalization phases；
- residency acquire/release/evict；
- KV/activation/gradient/optimizer spill；
- resource broker admission / reserve usage；
- distributed rank/group/epoch；
- output/checkpoint publication。

性能结论必须来自固定 workload、原始 trace、memory high-water、I/O bytes、collective bytes 和
吞吐/延迟，不从 mock、旧 binary 或历史机器结果外推。

### 11.3 验证层级

1. CPU unit tests：identity、canonicalization、state machine、failure atomicity；
2. synthetic/mocks：I/O、fence、rollback、fatal backend error、collective failure；
3. CUDA feature compile；
4. 单卡 kernel/reference correctness；
5. 单卡 out-of-core overlap；
6. 多卡 collective/topology correctness；
7. 组合 parallelism；
8. training/update/checkpoint transaction；
9. post-training/RL multi-model workload；
10. fixed large-model acceptance/performance。

## 12. 近期执行顺序

### P0：提交当前统一 custody 基线

- [x] `ResolvedStage` 与 retention-aware resource contract；
- [x] routed experts 进入 resolved-stage wait path；
- [x] typed stage / transaction / persistent custody；
- [x] resume `Consumed` / `StillActive`；
- [x] ordinary/speculative terminal outcome 连接；
- [x] speculative rollback failure quarantine 与 retry mock；
- [x] model/runtime lib UT；
- [x] workspace all-target check、fmt、broader CPU baseline；
- [x] commit + push，作为换机交接点。

### P1-A：统一 terminalization

- [x] payload 精确持有 transaction ownership；
- [x] ordinary/speculative 共用 `end_transaction(Publish|Abort)`；
- [x] typed backend `Pending / Complete`；
- [x] owner loop 自动推进，不依赖第二次 cancellation；
- [x] provisional KV generation publication；
- [x] inference failure-injection state-transition matrix；
- [ ] activation / gradient / optimizer generation publication；
- [ ] training / RL payload 接入统一 backend terminal contract。

### P1-B：闭环通用 static/mutable resource transport

- [ ] generic checkpoint extent -> pinned read；
- [ ] provider packing/install authority；
- [ ] dense parameter bundle 进入唯一 continuation path；
- [ ] 移除 eager static image 生产依赖；
- [ ] KV / activation / gradient / optimizer runtime manifests；
- [ ] spill/reload 与 completion fence；
- [ ] 删除剩余 model-private residency inference。

### P2：`sm_103` CUDA/CUTLASS provider

- [x] architecture/capability API；
- [x] operation semantic requirements；
- [x] CUTLASS/core provider selection；
- [x] 删除旧 `ferrule-cuda`/oxide 双路径；
- [x] compute capability 10.3 / `sm_103` build/runtime profile；
- [x] DeepSeek-V4 proposal attachment forward executable baseline；
- [ ] backward/update capability implementation；
- [ ] kernel 性能对齐、完整 timeline evidence 与 vLLM/SGLang gate。

### P3：Distributed foundation

- [ ] device/process mesh；
- [ ] communicator identity/generation；
- [ ] TP/PP/EP 优先组合由选定大模型决定；
- [ ] DP/SP/CP 单独与组合 contract；
- [ ] communication stages/resources/fences；
- [ ] failure propagation 与 distributed terminalization；
- [ ] I/O/communication/compute overlap trace。

### P4：Training、post-training、RL

- [ ] forward/backward/update stage prototype；
- [ ] activation checkpointing 与 recompute；
- [ ] gradient/optimizer spill；
- [ ] distributed optimizer/checkpoint publication；
- [ ] policy/reference/reward/value coordination；
- [ ] rollout actor/learner scheduling；
- [ ] RL multi-model out-of-core acceptance。

### P5：模型与硬件扩展

- [ ] 选定并接通 large-model acceptance，候选 Kimi K3；
- [ ] DeepSeek `sm_121a` 独立回归；
- [ ] Qwen CPU correctness；
- [ ] `sm_86` profile compile/runtime/out-of-core；
- [ ] Metal / ROCm / Ascend capability/provider design；
- [ ] future-aware prefetch 新一轮评估。

## 13. Release gate

在满足以下条件前保持 NO-GO：

### Protocol baseline

- [ ] 唯一 stage/materialization/continuation/custody production path；
- [ ] CPU UT、synthetic failure mocks、workspace checks 全绿；
- [ ] commit/rollback/cancel terminal outcome exactly once；
- [ ] cancellation、stale、failure、rollback、shutdown 后无 waiter/lease/credit/residency/KV 泄漏；
- [ ] terminalization 不依赖业务调用方重复发起 cancel；

### `sm_103` / CUDA provider

- [ ] CUDA target feature 可编译；
- [ ] `sm_103` capability negotiation 与 kernel selection 可验证；
- [ ] 不存在“CUTLASS == 某个 CUDA target/profile”全局假设；
- [ ] 现有 `sm_121a` 通过独立迁移回归；
- [ ] forward/backward capability 不会错误选择；

### Out-of-core / overlap

- [ ] 至少一个 large-model stage path 使用真实 generic transport/install；
- [ ] 模型或工作集实际大于目标显存时仍可执行；
- [ ] timeline 证明 read/upload/communication 与其他 ready compute 重叠；
- [ ] completion fence 前不释放或复用 in-flight resource；
- [ ] memory high-water、I/O bytes、collective bytes 和吞吐/延迟可复现；

### Distributed / training / RL

- [ ] 选定 parallel modes 的 topology 与 collective correctness；
- [ ] rank/device failure 不产生 distributed ownership leak；
- [ ] training forward/backward/update transaction correctness；
- [ ] activation/gradient/optimizer spill correctness；
- [ ] post-training/RL checkpoint generation 与 rollback/custody correctness；

### Portability gates

- [ ] Qwen/`sm_86` profile 作为后续独立 portability gate；
- [ ] Metal/ROCm/Ascend 不要求立即实现，但 model/runtime contract 不得含 CUDA-only 假设；
- [ ] 所有性能结论都有固定 workload、原始日志和当前源码构建证据。
