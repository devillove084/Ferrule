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
         -> global single-flight materialization
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

近期硬件方向调整为 **NVIDIA B300 优先**。RTX 3090、Qwen 和小模型仍是重要的
portability / correctness 目标，但不再是下一阶段的第一优先级。近期优先级是：

1. 收尾唯一 stage / materialization / continuation / custody / terminalization 协议；
2. 重构 CUDA architecture / capability 与 CUTLASS provider；
3. 建立 B300 capability profile、kernel selection 和大模型可执行路径；
4. 建立 DP / TP / PP / EP / SP / CP 的拓扑、通信与资源协议；
5. 扩展到 training、post-training、RL；
6. 选择大型模型做端到端验收，候选包括 Kimi K3；
7. 后续回到 Qwen / Ampere `sm_86` 做可移植性与较小硬件 out-of-core 验收。

任何只适用于“全模型常驻显存”、只适用于 routed experts、只适用于 inference，或为
某个 CUDA/CUTLASS architecture 建立的生产旁路，都不是目标架构。迁移可以分步，但
每一步只能有一个生产入口，不保留长期 legacy/new 双路径。

## 1. 冻结原则

| 项目 | 当前决定 |
|---|---|
| 核心能力 | 模型、训练状态或多模型工作集可以大于显存；I/O、上传、通信和计算必须可重叠 |
| 生产执行路径 | 只保留一条 resource-aware stage / materialization / continuation / custody 路径 |
| 近期硬件 | NVIDIA B300；具体 architecture ID 由 capability detection 确认，不在模型层硬编码 |
| 后续硬件 | RTX 3090 / `sm_86`、现有 `sm_121a`、未来 Metal、ROCm、Ascend provider |
| 模型方向 | 先完善 provider、分布式和训练基础，再选择大型模型验收；Kimi K3 是候选，Qwen 保留为后续 portability track |
| 训练兼容 | inference / training 共用资源身份、访问、生命周期、continuation 和 terminalization 协议 |
| 后端边界 | CUDA、Metal、ROCm、Ascend 是 backend/provider，不进入模型通用 stage/resource contract |
| CUTLASS 边界 | CUTLASS 是 CUDA provider 的一种实现，不等于某个 architecture，也不是跨硬件抽象 |
| 物理正确性 | 已提交 physical work 不能被逻辑 cancellation 假装为未发生；必须等待或确认 quiescence |
| 失败处理 | terminal outcome exactly once；未确认 backend terminal 时完整保留 ownership |
| 验证条件 | 当前没有可用模型 checkpoint；GPU 环境正在变化，现阶段依赖 CPU UT、synthetic fixtures 和 mocks |
| 证据边界 | 当前不能宣称 B300、RTX 3090、GB10、`sm_121a`、`sm_86` 或任何 CUDA runtime 已验证 |

## 2. 进度总览

| 工作流 | 状态 | 当前结论 |
|---|---|---|
| P0 runtime-wide materialization | CPU/mock 已完成 | global single-flight、canonical dependencies、generation publication、targeted wake 已覆盖 |
| P0 调度 / I/O / compute overlap | CPU/mock 已完成 | owner-side progression、fairness、hard resources、critical-path overlap accounting 已覆盖 |
| 双 broker 合并 | 已完成 | 唯一生产 broker 为 `PhysicalResourceBroker` |
| Artifact 总抽象拆除 | 已完成 | checkpoint 公共层与模型私有 semantic binding 已分离，不保留 compatibility alias |
| Prefetch 评估 | 已完成一轮 | oracle 有收益；当前 score predictor 净负收益，不接生产 |
| `ResolvedStage` contract | 已实现 | exact resources、access、retention、workspace、canonical dependency validation 已落地 |
| Typed materialization custody | 已实现 | stage / transaction / persistent ownership 与 provider lease 聚合释放已落地 |
| Routed expert stage migration | 已实现 | DeepSeek packed / DSpark route 后 exact expert set 进入唯一 resolved-stage 路径 |
| Resume lifecycle | 已实现 | `Consumed` / `StillActive` 区分，仍活跃 continuation 不提前释放 custody |
| Transaction terminal propagation | 已实现基础 | ordinary / speculative commit、rollback、cancel 已连接 custody terminal outcome |
| Rollback failure quarantine | 已实现过渡方案 | rollback 未确认时保留 backend、KV、branch、session、scheduler、custody；有 mock retry UT |
| Unified terminalization coordinator | 待重构 | 当前 quarantine 保证正确性，但 ordinary/speculative terminal cleanup 仍过于分散 |
| Dense static bundle materialization | 未完成 | source catalog 已有；generic pinned transport/install 未接通，CUDA runner 仍有 eager static image |
| Runtime mutable resources | 待完成 | KV / activation / gradient / optimizer 需按 transaction 实例化 exact capacity、generation 和 spill source |
| DeepSeek 公共组件拆分 | 进行中 | checkpoint、math、shape、RoPE、config、dense semantics 已有公共层；继续去 family-private 重复 |
| CUDA capability/provider 重构 | 待开始 | 下一阶段近期重点；不能继续假设 CUTLASS 等于 GB10 |
| B300 provider/kernels | 未开始 | 需先完成 architecture profile、operation requirements、kernel selection 与 workspace contract |
| Distributed parallelism | 未开始 | DP / TP / PP / EP / SP / CP 需要统一 topology、communication stage、fence 和 custody |
| Training/post-training/RL | 未实现 | 通用 resource access/retention 已预留，execution/terminal protocol 尚需扩展 |
| Large-model acceptance | 未开始 | 候选 Kimi K3；最终选择取决于 checkpoint、operator、并行拓扑和许可可用性 |
| Qwen / 3090 portability | 后置 | 不删除；在核心 provider 和 distributed path 稳定后推进 |

## 3. P0：统一调度、I/O 与计算重叠基础

### 3.1 已完成的协议

- runtime-wide physical materialization registry；
- 唯一 `PhysicalResourceBroker`；
- canonical `DependencySet` 与 exact `WaiterId` / `ContinuationId`；
- source identity、content hash、encoding、backend、device、source generation、
  destination generation 全部进入 `MaterializationKey`；
- destination generation 由真实 provider reservation 提供，模型不能自行合成；
- global single-flight：相同 exact key 只提交一次物理 materialization；
- generation-qualified completion / publication；
- cancel、stale、failure、shutdown 和 rollback cleanup；
- physical custody：已提交 read/upload/install 不能被逻辑 cancel 直接撤销；
- owner-side progress，不要求 CUDA 才能编译 runtime protocol；
- I/O wait 与其他 runnable compute 的 overlap accounting；
- fairness、demand reserve 和 hard-resource accounting；
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
- transaction commit / rollback / cancel 与 transaction custody；
- persistent custody 的显式 retirement；
- resume `StillActive` 对 continuation 与 stage custody 的保留；
- speculative terminal outcome 的显式传播；
- speculative backend rollback failure 的完整 ownership quarantine；
- shutdown / subsequent cancellation 的 rollback retry。

当前 quarantine 是 correctness baseline，不是最终 API。下一阶段必须收敛 terminalization
抽象，不能继续为每一种 execution mode 添加新的 rollback 分支。

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

### 4.2 当前过渡实现

当前 speculative rollback failure 会进入 quarantine，完整保留：

- backend transaction 与 verification branches；
- KV reservations / prepared commit；
- source states、suspended schedules 和 decode actions；
- materialization transaction custody；
- request identity。

只有 retry 确认 backend terminal 后才回收。CPU/mock UT 已覆盖：第一次 rollback 失败时
不释放 provider lease、session 或 KV；第二次 rollback 成功后 exactly-once cleanup。

### 4.3 目标：统一 terminalization coordinator

当前复杂度来自 terminal ownership 分散在 driver、executor、speculation、KV manager 和
materialization registry，而不是 rollback 语义本身。目标是将 ordinary execution、
speculation、training 和 RL rollout 共用一个状态机：

```rust
enum TransactionState<P> {
    Running(P),
    Suspended {
        continuation: ContinuationId,
        payload: P,
    },
    Terminalizing {
        intent: TerminalIntent,
        progress: TerminalProgress,
        payload: P,
        cause: Option<Error>,
    },
    Terminal(TerminalReceipt),
}

enum TerminalIntent {
    Commit,
    Rollback,
    Cancel,
}

enum TerminalProgress {
    Quiescing,
    BackendTerminal,
    Reclaiming,
}
```

统一 ownership payload 至少要能持有：

```rust
TransactionOwnership<S> {
    backend_transaction,
    continuations,
    materialization_custody,
    kv_or_mutable_state,
    model_states,
    suspended_schedules,
    scheduler_actions,
    communication_ownership,
}
```

目标 API：

```text
request_terminal(transaction, intent)
drive_terminalization(transaction)
```

runtime tick / completion wake 自动推进 terminalization；不应要求调用方再次发送一次
`cancel_request()` 才能完成 rollback retry。

### 4.4 Backend terminal contract

现有 `rollback_prepared_batch(...) -> Result<()>` 信息量不足，不能区分 pending、已终结、
可重试错误与 device loss。目标 contract 应表达：

```rust
enum BackendTerminalProgress {
    Pending(CompletionFence),
    Complete(BackendTerminalReceipt),
    DeviceLost(DeviceLossReceipt),
}
```

要求：

- `Pending` 是正常异步进度，不是 error；
- `Complete` 明确保证 backend 不再访问 transaction resources；
- `DeviceLost` 触发 device/worker 级 quarantine，不无限重试单 transaction rollback；
- protocol error 与 backend fatal error 分开；
- completion fence 是物理 quiescence 证据，不是逻辑状态猜测；
- terminal outcome exactly once；
- post-terminal cleanup 失败不允许把已确认 terminal 的 backend 重新描述为 active。

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

- [ ] ordinary / speculative 不再拥有两套 terminal cleanup；
- [ ] 移除多个互斥 `Option<pending/rollback/terminal>` 表达，非法状态不可构造；
- [ ] backend terminal progress typed；
- [ ] retry 由 owner loop / completion fence 自动驱动；
- [ ] device loss 有设备级处理协议；
- [ ] commit / rollback / cancel exactly once；
- [ ] backend terminal 后的 cleanup failure 可重试且不重复 terminal operation；
- [ ] inference、training、RL transaction 共用 coordinator；
- [ ] failure-injection UT 覆盖每个 transition 与 ownership leak。

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

DeepSeek MLA、compressor、indexer、hyper-connection、DSpark 等能力只有在证明跨模型语义
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

`crates/ferrule-cuda` 必须拆开以下概念：

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

- GB10 的同义词；
- 某个 SM ID 的全局 feature gate；
- 模型资源身份的一部分；
- Metal / ROCm / Ascend 的抽象基类；
- 只支持 forward inference 的 operation catalog。

### 6.2 方向性组织

目录方向可参考，但在 source review 前不冻结最终文件名：

```text
ferrule-cuda/
  runtime/
  architecture/
  providers/
    native/
    cutlass/
      common/
      blackwell/
      ampere/
```

要求：

- 只有一个 production provider selection 入口；
- architecture specialization 在 provider 内部，不复制模型 runner；
- 现有 `sm_121a` operation requirements、ABI 与可执行语义在迁移中保持正确；
- 不保留 legacy/new 双 provider 路径；
- B300 的实际 architecture/capability 由工具链与设备查询确认，不凭名称硬编码；
- Ampere、Blackwell/Blackwell Ultra 等 profile 可共用 semantic requirements，但 kernel
  implementation 和 workspace 可以不同；
- forward/backward capability 必须显式，不允许把缺少 backward 的 kernel 误选入训练计划。

### 6.3 近期推进顺序

1. [ ] 审查现有 `ferrule-cuda` 文件命名、build scripts、operation enums 和 ABI；
2. [ ] 定义 `CudaTargetProfile` / capability query，不绑定模型 family；
3. [ ] 定义 provider-neutral operation requirements；
4. [ ] 将现有 CUTLASS kernels 迁移到唯一 provider selection；
5. [ ] 去除“CUTLASS == GB10 / `sm_121a`”的全局假设；
6. [ ] 添加 B300 build/runtime profile；
7. [ ] 接通 generic checkpoint read -> pinned transport -> install；
8. [ ] 接入 forward operator set；
9. [ ] 设计 backward / gradient / optimizer operator capability；
10. [ ] 后续添加 `sm_86` profile 和 Qwen BF16 operator set；
11. [ ] 分别做 B300、现有 `sm_121a`、RTX 3090 的独立 compile/runtime acceptance。

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

B300 可用后，优先选择能验证以下能力的大模型：

- dense + MoE 或至少包含复杂 routed experts；
- 多卡并行需求；
- checkpoint 规模足以验证 out-of-core；
- operator 和精度路径可在目标硬件实现；
- checkpoint、tokenizer、config、许可和参考输出可获得。

Kimi K3 是候选，但在获得并确认其实际 checkpoint/config/operator requirements 前，不在
roadmap 中编造具体架构。模型选择应依据：

1. 可获得、可重复的 checkpoint 与 reference；
2. B300 operator feasibility；
3. TP/PP/EP 等并行验收价值；
4. out-of-core materialization 压力；
5. training/post-training 后续价值；
6. 固定 workload 与性能证据可复现性。

### 9.2 Qwen 与 RTX 3090 portability track

Qwen 不删除，只后移：

- 复用 DeepSeek 拆出的 common checkpoint/dense/GQA/RMSNorm/RoPE/SwiGLU 组件；
- synthetic checkpoint 与 CPU reference correctness；
- 在 provider capability API 完成后添加 Ampere `sm_86`；
- 选择明确版本和规模，不使用含义不清的“Qwen3.8”名称；
- 小模型验证 checkpoint/semantic correctness；
- 大于可用显存的模型验证 3090 out-of-core；
- 不为 Qwen 或 3090 建模型私有 kernel / scheduler 旁路。

3090 acceptance 仍应证明：

```text
materialize bundle N+1
    overlaps
execute bundle N
    overlaps
release/evict bundle N-1 after completion fence
```

## 10. Prefetch

当前结论保持：

- oracle / future-aware prefetch 对 selected miss 有明显收益；
- 当前 `ScoreBasedExpertPredictor` 在有限 residency slots 下挤占真实 working set，净收益为负；
- 不接入当前 predictor；
- 保留 `Prefetch` resource class 和 demand reserve；
- 只评估有因果依据的 future-aware 信号，例如 prepared next stage、router result、pipeline
  schedule、known backward stage、rollout schedule 或可靠 draft information。

任何新 predictor 必须同时报告：

- selected miss；
- end-to-end latency / throughput；
- slot displacement；
- extra storage/H2D bytes；
- wasted install；
- demand fairness；
- memory high-water；
- 在固定 workload 下相对 no-prefetch baseline 的净收益。

## 11. Observability 与验证策略

### 11.1 当前 CPU/mock 证据

截至本次更新，当前 WIP 已实际运行：

```text
cargo test -p ferrule-model --lib
291 passed

cargo test -p ferrule-runtime --lib
301 passed
```

新增 mock 覆盖包括：

- resolved stage canonicalization 与 fail-closed identity validation；
- stage / transaction / persistent custody；
- shared key 最后 owner 才释放；
- `StillActive` resume 不提前释放；
- transaction commit / rollback terminal release；
- speculative terminal propagation；
- rollback failure quarantine 和成功 retry 后 exactly-once cleanup。

本次还通过了 `ferrule-common --lib` 68 tests、完整 `ferrule-model`（包括 local smoke 与
prefetch mocks）、`cargo check --workspace --all-targets` 和 `cargo fmt --all -- --check`。
上述结果不证明任何 CUDA/GPU path。

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
2. synthetic/mocks：I/O、fence、rollback、device loss、collective failure；
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

- [ ] 设计 `TransactionOwnership` 与 `TerminalIntent`；
- [ ] ordinary/speculative 共用 terminal coordinator；
- [ ] typed backend pending/complete/device-lost；
- [ ] owner loop 自动 retry，不依赖第二次 cancellation；
- [ ] generational mutable-state publication；
- [ ] failure-injection state-transition matrix。

### P1-B：闭环通用 static/mutable resource transport

- [ ] generic checkpoint extent -> pinned read；
- [ ] provider packing/install authority；
- [ ] dense parameter bundle 进入唯一 continuation path；
- [ ] 移除 eager static image 生产依赖；
- [ ] KV / activation / gradient / optimizer runtime manifests；
- [ ] spill/reload 与 completion fence；
- [ ] 删除剩余 model-private residency inference。

### P2：B300 CUDA/CUTLASS provider

- [ ] architecture/capability API；
- [ ] operation semantic requirements；
- [ ] CUTLASS/native provider selection；
- [ ] 迁移现有 `sm_121a` 而不建立双路径；
- [ ] B300 build/runtime profile；
- [ ] forward operator baseline；
- [ ] backward/update capability design；
- [ ] GPU correctness 与 timeline evidence。

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
- [ ] RTX 3090 `sm_86` compile/runtime/out-of-core；
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

### B300 / CUDA provider

- [ ] CUDA target feature 可编译；
- [ ] B300 capability negotiation 与 kernel selection 可验证；
- [ ] 不存在“CUTLASS == GB10”全局假设；
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

- [ ] Qwen/RTX 3090 作为后续独立 portability gate；
- [ ] Metal/ROCm/Ascend 不要求立即实现，但 model/runtime contract 不得含 CUDA-only 假设；
- [ ] 所有性能结论都有固定 workload、原始日志和当前源码构建证据。
