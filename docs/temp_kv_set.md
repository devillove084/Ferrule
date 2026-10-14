## User

我在看了两篇论文，我觉得我们可以尝试实现一下：2. KVSET：为 prefix cache 建立 working-set 视角

KVSET 将一个常被忽略的问题形式化：给定目标 hit rate，最小需要多少 KV cache 容量？它对每个 KV page 计算 LRU stack distance，以一次扫描估算多个容量点的命中率，再取满足目标的最小容量；支持在线请求处理和离线 trace replay。

论文结论： 作者称在生产 LLM trace 和真实 cache deployment 上，估计值与测量接近。
新意： 不是再造一个 eviction policy，而是把容量、命中率和成本联系起来；这对长会话、工具调用和 agent 历史尤其有意义。
实验可信度： 摘要没有给出完整 trace 分布、分层 cache、并发 admission 或非 LRU policy 的细节；应把“接近测量”视为作者结论，不视为跨系统保证。


这篇 9 月 13 日论文是本周值得补读的旧材料。它指出 CPU spill 的 KV 有两种相反用途：传回 GPU 做 attention，或留在 CPU 原位计算。InplaceKVCache 在写入时就按 device affinity 和 access pattern 将字节放入四类区域，WriteScope 再按序列切分 CPU/GPU share，用 roofline 和在线反馈调整比例。

作者在 DeepSeek-V2-Lite、Qwen3-30B-A3B、Mixtral-8×7B、32GB VRAM 下报告 1M-token aggregate scale，并在长上下文上相对复现 baseline 达到 1.4–2.5×。这是作者实验，不代表 vLLM/SGLang 的所有失败场景都可泛化；但它与 vLLM HiSparse 和 KVCR 的“hot GPU + host tier”方向高度互补。你来审阅一下，看看如何实现

## Assistant

<thinking>
**Planning delegated architecture inspection**
</thinking>

我先让子代理并行梳理仓库结构、现有 KV/cache 实现和测试入口，再基于实际代码给出可落地的实现边界与方案；我这边不直接扫描仓库内容，避免绕过项目规则。

## User

你也可以把论文拿到，都在target里面做好了不是，我的意思是你可以在target路径下来做，论文你要自己下载

## Assistant

<thinking>
**Planning separate repo and paper processing**
</thinking>

明白了：论文材料在 `target` 下，同时如果缺少原文我会再下载到 `target`，不把临时资料放进源码。现在我会分开核对论文细节和仓库可插入点，之后给出实现建议；如果仓库已有基础，我会直接做最小可验证原型。

**Tool Call: 调查仓库缓存架构**
Status: Completed

## 结论与调查范围

已先阅读 `Ferrule/AGENTS.md`，随后只用 terminal 的 `rg --files`、`rg`、`sed`、`nl` 等做只读调查；**未修改文件，也未运行会生成构建产物的测试／构建命令**。当前工作树已有大量修改和未跟踪文件，下述证据针对读取时的工作树，不是单独针对 `HEAD`。子代理接口不可用，因此未能执行仓库要求的委派调查，已在开始时说明。

核心结论：

- **已有完整的逻辑分页 KV 生命周期**：reservation、COW、prepare/commit、rollback、retirement。
- **已有 token radix prefix cache 和 LRU**，但它淘汰的是“带 model state 的 prefix snapshot”，不是 GPU 物理 page 的独立 LRU。
- **已有 GPU→CPU snapshot / CPU→GPU restore**，通过 session preemption 接入；尚不能把它视为自动分层 KV cache。
- **没有在所查路径发现统一的 KV page 访问 trace/replay 工具**。已有交互输出 golden、expert 预取模拟、CUDA graph replay，三者用途不同。

---

## 1. KV/page 的核心类型与职责

| 层次 | 文件及证据 | 关键类型／函数与职责 |
|---|---|---|
| 公共描述 | [`execution.rs:1111–1145`](Ferrule/crates/ferrule-common/src/execution.rs#L1111) | `KvPageId(u32)`、`KvCowReplacement`、`KvReservationView`。View 可 clone，但**不拥有页面、不能 commit/rollback**。 |
| 逻辑页面管理 | [`page_manager.rs:348–373`](Ferrule/crates/ferrule-runtime/src/cache/page_manager.rs#L348) | `KvPageManager` 保存 free list、refcount、sequence block table、pending reservations、prefix snapshots、retiring pages。 |
| 预留与 COW | [`page_manager.rs:710–860`](Ferrule/crates/ferrule-runtime/src/cache/page_manager.rs#L710) | `required_physical_pages()`、`reserve()`；共享且未填满的尾页追加时额外分配 COW page。 |
| 原子提交 | [`page_manager.rs:867`](Ferrule/crates/ferrule-runtime/src/cache/page_manager.rs#L867)、[`1036`](Ferrule/crates/ferrule-runtime/src/cache/page_manager.rs#L1036) | `prepare_commit()` 校验整个 packed cohort；`publish_commit()` 在 backend 成功后发布 block table。 |
| 回收协议 | [`page_manager.rs:573–613`](Ferrule/crates/ferrule-runtime/src/cache/page_manager.rs#L573) | `begin_retirement()` 进入隔离集合；`confirm_page_retirement()` 才放回 free list。 |
| Driver 资源账本 | [`kv_lifecycle.rs:18–119`](Ferrule/crates/ferrule-runtime/src/engine/driver/kv_lifecycle.rs#L18) | `KvLifecycleState` 将每个 page 与 `PhysicalResourceGrant` 配对，回收确认后释放 hard credit。 |
| CPU 物理存储 | [`cpu/kv.rs:196–208`](Ferrule/crates/ferrule-backend/src/cpu/kv.rs#L196)、[`514`](Ferrule/crates/ferrule-backend/src/cpu/kv.rs#L514) | `CpuPagedKvPool`：连续 plane storage、resident page→slot、preempted snapshots、事务和 lease。 |
| CUDA 物理存储 | [`page_pool.rs:558–603`](Ferrule/crates/ferrule-backend/src/cuda/operators/kv/page_pool.rs#L558) | `CudaKvPagePool`：**每个 plane 一个固定容量 typed buffer**，page→physical slot 映射及 host snapshots。 |
| CUDA 事务层 | [`transaction.rs:244–399`](Ferrule/crates/ferrule-backend/src/cuda/operators/kv/transaction.rs#L244) | `CudaPagedKvPool`、`CudaPagedKvTransaction`；prepare 建立 shadow、pin、fence，commit 等待 quiescent 后安装映射。 |
| 计算绑定 | [`kv_binding.rs:18–23`](Ferrule/crates/ferrule-model/src/transformer/standard/cuda/kv_binding.rs#L18) | `CudaStandardKvBinding` 借用 entered KV view；明确**不分配页面，也不确认事务完成**。 |

### 最重要的生命周期约束

[`page_manager.rs:7–17`](Ferrule/crates/ferrule-runtime/src/cache/page_manager.rs#L7) 已直接写出：

```text
reserve -> execute -> prepare logical commit -> commit backend -> publish logical commit
                     ↘ backend rollback -> abort reservation
publish/abort/free -> quarantine -> release backend pages -> confirm retirement -> reuse
```

COW 判定不是简单“满页再分配”。[`page_manager.rs:821–825`](Ferrule/crates/ferrule-runtime/src/cache/page_manager.rs#L821)：

```rust
let shared_tail = token_count > 0
    && !committed_tokens.is_multiple_of(self.schema.page_size())
    && existing_pages
        .last()
        .is_some_and(|page| self.page_refcounts.get(page).copied().unwrap_or(0) > 1);
```

另外，**逻辑 COW 与 backend transaction shadow 是两回事**。CUDA planner 明确“每次写入都有独立 provisional slot，包括已有 writable page”：
[`transaction/plan.rs:61–67`](Ferrule/crates/ferrule-backend/src/cuda/operators/kv/transaction/plan.rs#L61)。

---

## 2. 实际调用链

### 2.1 普通 prefill/decode 的 KV 链

```text
Driver transaction admission
  → reserve_batch_pages()
      → required_physical_pages()
      → evict_prefixes_for_kv_pages()
      → 重新计算 required_physical_pages()
      → acquire_hard_resources(KvPage)
      → KvPageManager::reserve()
      → bind_reservation_execution()
  → bind_reserved_pages()
      → reservation_views()
      → reservation_bindings()
      → ScheduledBatch::bind_paged_kv()
  → NativeExecutor / model runner / backend transaction
  → 执行 forward
  → KvPageManager::prepare_commit()
  → executor.end_transaction(... Publish)
  → BackendCommittedPendingPublish
  → progress_committed_resident_kv()
      → KvPageManager::publish_commit()
  → scheduler.commit_action()
  → capture_committed_prefill_prefixes()
```

对应证据：

- admission 预留和绑定：[`admission.rs:844–873`](Ferrule/crates/ferrule-runtime/src/engine/driver/admission.rs#L844)。
- 淘汰前后两次计算需求：[`kv_lifecycle.rs:327–337`](Ferrule/crates/ferrule-runtime/src/engine/driver/kv_lifecycle.rs#L327)。这是必要的：淘汰 snapshot 可能改变共享尾页 refcount，从而改变 COW 需求。
- logical prepare：[`resident.rs:286–326`](Ferrule/crates/ferrule-runtime/src/engine/driver/resident.rs#L286)。
- 仅 backend committed 状态允许发布：[`resident.rs:334–380`](Ferrule/crates/ferrule-runtime/src/engine/driver/resident.rs#L334)。
- logical publish：[`kv_lifecycle.rs:682–716`](Ferrule/crates/ferrule-runtime/src/engine/driver/kv_lifecycle.rs#L682)。
- 最后更新 scheduler 并捕获 prefix：[`resident.rs:423–425`](Ferrule/crates/ferrule-runtime/src/engine/driver/resident.rs#L423)。

### 2.2 页面释放链

```text
prefix eviction / sequence free / rejected reservation
  → KvRetirement
  → runner.release_kv_pages()
  → KvPageManager::confirm_page_retirement()
  → release_hard_resources()
```

[`kv_lifecycle.rs:239–270`](Ferrule/crates/ferrule-runtime/src/engine/driver/kv_lifecycle.rs#L239) 在 backend release 前拒绝 live transactions；失败保留 retirement，不能提前复用 page。

---

## 3. Prefix cache 与 LRU：已有实现和限制

### 3.1 索引和 payload

[`cache/radix.rs:1–6`](Ferrule/crates/ferrule-runtime/src/cache/radix.rs#L1) 明确：索引只存 token key 与 caller-owned payload，不认识模型、CUDA slot 或 block table。

关键类型：

- `RadixPrefixCache<P>`：[`radix.rs:394–411`](Ferrule/crates/ferrule-runtime/src/cache/radix.rs#L394)。
- `PrefixCacheNamespace`：model/backend/device/plan/layout 五个字段，见 [`radix.rs:24–30`](Ferrule/crates/ferrule-runtime/src/cache/radix.rs#L24)。
- `PrefixCacheLease`：非 clone 的显式 pin；drop 不自动 unpin，见 [`radix.rs:113–126`](Ferrule/crates/ferrule-runtime/src/cache/radix.rs#L113)。
- runtime payload 同时持有 model state 与 KV snapshot，见 [`driver/prefix.rs:11–14`](Ferrule/crates/ferrule-runtime/src/engine/driver/prefix.rs#L11)：

```rust
pub(super) struct ResidentPrefixPayload<S> {
    pub(super) model_state: S,
    pub(super) snapshot: KvPrefixSnapshot,
}
```

因此不能仅凭 token hash 和 GPU page pointer 就替换当前 prefix 实现。

### 3.2 Lookup / capture 链

**命中：**

```text
prepare_waiting_admission
  → prepare_prefix_admission
  → lookup_longest_prefix(BeforeLastPromptToken)
  → pin payload
  → fork_sequence_state_from(model_state, matched_tokens)
  → prepare_fork_prefix_snapshot()
  → apply_committed_prefix()
  → unpin
  → publish_fork_prefix_snapshot()
  → publish scheduler admission
  → hits += 1
```

证据：

- [`prefix.rs:134–142`](Ferrule/crates/ferrule-runtime/src/engine/driver/prefix.rs#L134)：固定使用 `BeforeLastPromptToken`。
- [`prefix.rs:345–407`](Ferrule/crates/ferrule-runtime/src/engine/driver/prefix.rs#L345)：model fork、snapshot fork 和 admission 修改。
- [`admission.rs:567–611`](Ferrule/crates/ferrule-runtime/src/engine/driver/admission.rs#L567)：发布成功后才计 hit。
- [`admission.rs:712–721`](Ferrule/crates/ferrule-runtime/src/engine/driver/admission.rs#L712)：cache-eligible admission 发布后才计 miss。

**捕获：**

[`driver.rs:1345–1445`](Ferrule/crates/ferrule-runtime/src/engine/driver.rs#L1345) 校验 committed prefill frontier，fork model state，调用 `capture_prefix_snapshot()`，然后：

```rust
let charge = snapshot.page_count();
...
.insert_and_queue_cleanup(namespace, &tokens, payload, charge)
```

这意味着容量是**各 entry 的 page charge 之和**，不是去重后的物理 page 总数。重叠 snapshot 可以共享页面，但仍分别计费。

### 3.3 LRU 的真实粒度与复杂度

[`radix.rs:693–710`](Ferrule/crates/ferrule-runtime/src/cache/radix.rs#L693)：

```rust
.entries
.iter()
.filter(|(_, entry)| entry.pin_count == 0)
.min_by_key(|(entry_id, entry)| (entry.last_used_generation, **entry_id))
```

- 粒度：完整 prefix entry。
- 只选未 pin 的 entry。
- hit 更新 generation：[`radix.rs:595–602`](Ferrule/crates/ferrule-runtime/src/cache/radix.rs#L595)。
- 单个 victim 选择是扫描；容量插入使用候选排序，见 [`radix.rs:816–858`](Ferrule/crates/ferrule-runtime/src/cache/radix.rs#L816)。大 cache 下需留意扫描／排序成本。
- pressure eviction 调用 `evict_lru()`，再完成 snapshot/model cleanup，见 [`prefix.rs:317–330`](Ferrule/crates/ferrule-runtime/src/engine/driver/prefix.rs#L317)。

### 3.4 不能忽略的支持限制

1. 默认关闭：[`scheduling/resident.rs:70–86`](Ferrule/crates/ferrule-runtime/src/scheduling/resident.rs#L70)，`prefix_cache_capacity_pages = 0`。
2. retained session 不走 prefix lookup：[`admission.rs:539–543`](Ferrule/crates/ferrule-runtime/src/engine/driver/admission.rs#L539)；已有回归测试 [`driver.rs:9507–9523`](Ferrule/crates/ferrule-runtime/src/engine/driver.rs#L9507)。
3. live transactions 时跳过相关 capture/eviction。
4. Qwen3.5 factory 明确拒绝非零 prefix cache：[`qwen35.rs:1160–1164`](Ferrule/crates/ferrule-runtime/src/engine/model_factory/qwen35.rs#L1160)。
5. pipeline serving 也要求关闭：[`pipeline.rs:350–359`](Ferrule/crates/ferrule-runtime/src/engine/model_factory/pipeline.rs#L350)。
6. 当前是**已缓存 exact frontier 的最长前缀查找**，不是任意截取一个更长 snapshot。snapshot fork 要求 token 数精确一致：[`page_manager.rs:1238–1245`](Ferrule/crates/ferrule-runtime/src/cache/page_manager.rs#L1238)。配合“保留最后一个 prompt token”，完全相同请求不一定能从唯一的完整 prompt entry 命中较短前缀。

---

## 4. GPU/CPU/offload 与其他 cache

### 4.1 已有 KV offload 调用链

```text
ResidentTopKDriver::preempt_session()
  → suspend scheduler/model state
  → KvPageManager::preempt_sequence()
      → 仅选 refcount == 1 的 evicted_pages
  → executor.preempt_kv_pages()
  → GenericDecoderRunner → backend.preempt()
  → CudaPagedKvPool::preempt()
  → CudaKvPagePool::preempt()
      → 下载每个 page 的 plane range
      → 所有下载成功后删除 resident mapping，保存 host snapshot
```

反向 `restore_session()` 先 restore physical pages，再 restore logical table 和 scheduler。

证据：

- driver 状态机：[`sessions.rs:287–362`](Ferrule/crates/ferrule-runtime/src/engine/driver/sessions.rs#L287)。
- 仅独占页可搬出：[`page_manager.rs:1422–1449`](Ferrule/crates/ferrule-runtime/src/cache/page_manager.rs#L1422)。
- runner 转发：[`decoder/runner.rs:1054–1065`](Ferrule/crates/ferrule-model/src/decoder/runner.rs#L1054)。
- CUDA wrapper 的 pin、容量及 poison 检查：[`transaction/lifecycle.rs:22–58`](Ferrule/crates/ferrule-backend/src/cuda/operators/kv/transaction/lifecycle.rs#L22)。
- 下载与保存：[`page_pool.rs:1071–1120`](Ferrule/crates/ferrule-backend/src/cuda/operators/kv/page_pool.rs#L1071)。
- upload、rollback、移除 snapshot：[`page_pool.rs:1171–1255`](Ferrule/crates/ferrule-backend/src/cuda/operators/kv/page_pool.rs#L1171)。

### 4.2 为什么它还不是自动分层 KV cache

依据当前代码：

- `preempt_session()` 的仓库内调用搜索结果主要是 driver 测试，**未发现生产 scheduler 在内存压力下自动挑选 session 并调用它**。
- preempt 保留逻辑 page refs；runtime hard grants 并未在该链中释放。现有 pressure path 仍是淘汰 prefix。
- CUDA pool 是固定容量连续 allocation。preempt 释放的是 **pool 内 slot/mapping**，不是释放整个 device allocation，因此不能把它直接解释成“降低进程已分配显存”。
- host snapshots 使用 `HashMap<KvPageId, ...>`；在所查 KV pool/cache 路径未看到独立 host snapshot byte budget 或 host LRU。
- physical completion 不明时进入 `Quarantined`，拒绝自动恢复，见 [`sessions.rs:264–269,313–333`](Ferrule/crates/ferrule-runtime/src/engine/driver/sessions.rs#L264)。

**精度边界也要区分：**

底层 `CudaKvPagePool::preempt_typed()` 支持 F32 bits/BF16 words，见 [`page_pool.rs:1123–1168`](Ferrule/crates/ferrule-backend/src/cuda/operators/kv/page_pool.rs#L1123)；但 standard transactional GQA 的 `validate_schema()` 当前明确只接受 F32，见 [`transaction/plan.rs:22–45`](Ferrule/crates/ferrule-backend/src/cuda/operators/kv/transaction/plan.rs#L22)。不能以底层 typed helper 的存在推断整个 serving 路径支持 BF16 KV。

### 4.3 不要把 expert LRU 当成 KV LRU

- runtime expert controller：[`expert_residency.rs:13–21`](Ferrule/crates/ferrule-runtime/src/expert_residency.rs#L13)，每 layer 独立 coordinator。
- victim 选择在 [`ferrule-common/src/expert_residency.rs:419–434`](Ferrule/crates/ferrule-common/src/expert_residency.rs#L419)，排除 leased/pending slots，再按 recency 选择。
- standard CUDA expert cache：[`expert_cache.rs:15–31`](Ferrule/crates/ferrule-model/src/transformer/standard/expert_cache.rs#L15)，`KeepAll` 或 `Bounded(max_experts,max_bytes)`；只淘汰 `Idle`，见 [`141–160`](Ferrule/crates/ferrule-model/src/transformer/standard/expert_cache.rs#L141)。
- 该 budget **明确排除 KV、非 expert weights 和 allocator slack**，见同文件 `3–6` 行。
- `HostExpertCache` 是完整预热后的 immutable image，不是 LRU：[`host_experts/mod.rs:559–567`](Ferrule/crates/ferrule-model/src/transformer/host_experts/mod.rs#L559)。

---

## 5. Trace/replay、benchmark 现状

| 实现 | 证据 | 实际用途 |
|---|---|---|
| 交互 golden | [`interactive_trace.rs:15–43`](Ferrule/crates/ferrule-cli/src/bench/interactive_trace.rs#L15) | 保存 prompt、generated token IDs、stop reason；不是 page events。 |
| Golden comparator | [`interactive_trace.rs:73–124`](Ferrule/crates/ferrule-cli/src/bench/interactive_trace.rs#L73) | 比较 generated tokens/EOS/stop string。 |
| CLI benchmark | [`bench_interactive.rs:481–534`](Ferrule/crates/ferrule-cli/src/commands/bench_interactive.rs#L481) | TTFT、prefill/decode 吞吐、runtime steps、observability JSON。 |
| Expert trace 模拟 | [`prefetch_evaluation.rs:109`](Ferrule/crates/ferrule-model/tests/prefetch_evaluation.rs#L109)、[`253`](Ferrule/crates/ferrule-model/tests/prefetch_evaluation.rs#L253)、[`346`](Ferrule/crates/ferrule-model/tests/prefetch_evaluation.rs#L346) | `generate_trace → simulate → report`；比较 demand LRU、prefetch、tuned、oracle。对象是 expert routing，不是 KV。 |
| CUDA graph replay | [`graph.rs:26–32`](Ferrule/crates/ferrule-backend/src/cuda/graph.rs#L26)、[`203–215`](Ferrule/crates/ferrule-backend/src/cuda/graph.rs#L203) | 捕获后复用 GPU execution graph；依赖 operand 地址和最后 consumer completion。 |
| Backend prefix replay view | [`page_manager.rs:1555–1599`](Ferrule/crates/ferrule-runtime/src/cache/page_manager.rs#L1555) | `reservation_prefix_view()` 生成不发布、不转移 ownership 的截短 view，不是离线 trace 回放框架。 |

### 现有 benchmark 的几个具体陷阱

1. **Golden 校验不是可靠的失败门禁。**  
   [`bench_interactive.rs:141–179`](Ferrule/crates/ferrule-cli/src/commands/bench_interactive.rs#L141) 只在 `--json` 下读取 golden，把比较结果写进 JSON，最后仍 `Ok(())`。

2. **轮数不一致可能漏检。**  
   [`interactive_trace.rs:79–83`](Ferrule/crates/ferrule-cli/src/bench/interactive_trace.rs#L79) 使用 `min()` 与 `zip()`；`all_ok()` 只检查已比较部分，没有要求两边轮数相同。model/template/prompt token 内容也不在该 comparator 中校验。

3. **仓库 golden 文件 generated tokens 为空。**  
   [`dsv4-interactive-two-turn-greedy.json:6–19`](Ferrule/configs/golden/dsv4-interactive-two-turn-greedy.json#L6)。不能把文件名当作已验证 golden 的证据。

4. **现成 bench 不是 prefix cache 性能实验。**  
   CLI 参数没有 prefix cache capacity；构建选项使用默认 scheduler、`kv_cache_mebibytes: None`，见 [`bench_interactive.rs:112–123`](Ferrule/crates/ferrule-cli/src/commands/bench_interactive.rs#L112)。

5. **现有观测不足以回放 page policy。**  
   `ResidentPrefixCacheStats` 只有 hits/misses，见 [`observability.rs:45–51`](Ferrule/crates/ferrule-runtime/src/engine/observability.rs#L45)；没有复用 token 数、victim、迁移 bytes/latency 等事件。

在 Cargo manifests 中未发现 `[[bench]]`、Criterion 或 Divan；性能入口主要是 CLI、脚本和带输出的测试。

---

## 6. 最小可插入位置

以下是基于现有边界的实现建议，**此次没有实施**。

| 目标 | 最小位置 | 必须保留的约束 |
|---|---|---|
| 替换 prefix 淘汰策略 | [`radix.rs:693`](Ferrule/crates/ferrule-runtime/src/cache/radix.rs#L693) 的 `lru_evictable()` 与 [`816`](Ferrule/crates/ferrule-runtime/src/cache/radix.rs#L816) 的 `plan_evictions()` | 两处要统一策略，否则 pressure eviction 与 insert/capacity eviction 不一致。不能移除 pin 过滤、失败原子性及 returned payload。 |
| 调整“哪些 prefix 值得缓存” | [`driver.rs:1345`](Ferrule/crates/ferrule-runtime/src/engine/driver.rs#L1345) 的 `capture_committed_prefill_prefix()` | 最好在 model fork/snapshot capture **之前**决策，避免无效 capture 后再 cleanup。只缓存 committed frontier。 |
| 添加生产 prefix/page trace | [`prefix.rs:345`](Ferrule/crates/ferrule-runtime/src/engine/driver/prefix.rs#L345)、[`kv_lifecycle.rs:282`](Ferrule/crates/ferrule-runtime/src/engine/driver/kv_lifecycle.rs#L282)、[`239`](Ferrule/crates/ferrule-runtime/src/engine/driver/kv_lifecycle.rs#L239) | 分开记录 lookup、admission publish、reserve/COW、backend release ACK、logical retire ACK。失败尝试不能记成成功事件。 |
| 做 CPU-only prefix policy replay | 复用 `RadixPrefixCache<P>` 的公开接口；在 runtime 测试层新增 harness | payload 可用模拟 page 集合；如果要模拟真实释放收益，必须处理共享 refcount，而不只扣 entry charge。 |
| 做自动 session offload | [`sessions.rs:653`](Ferrule/crates/ferrule-runtime/src/engine/driver/sessions.rs#L653) 的 `preempt_session()`／`restore_session()` 上方增加 policy | 复用现有状态机；不能绕过 pin、共享页、quarantine。还需设计 host budget 与 logical/physical credits 分离，不能仅接一个 LRU。 |
| 做真实 page 迁移优化 | [`transaction/lifecycle.rs:22`](Ferrule/crates/ferrule-backend/src/cuda/operators/kv/transaction/lifecycle.rs#L22) 与 [`page_pool.rs:1074,1172`](Ferrule/crates/ferrule-backend/src/cuda/operators/kv/page_pool.rs#L1074) | 异步搬运必须扩充 completion/custody 协议，不能把提交 copy 当作可复用 slot 的证明。 |
| 导出实验指标 | [`observability.rs:45–71`](Ferrule/crates/ferrule-runtime/src/engine/observability.rs#L45) → [`bench_interactive.rs:550–570`](Ferrule/crates/ferrule-cli/src/commands/bench_interactive.rs#L550) | 区分 entry hit、token reuse、logical page、physical slot、allocated bytes。 |

**最小且风险最低的起点**：先在 `RadixPrefixCache` 上做 CPU-only policy replay，并增加 driver 成功边界的观测；不要先在 attention kernel 或 `CudaStandardKvBinding` 中加入淘汰逻辑。

---

## 7. 测试／构建命令和风险

### 7.1 可以直接针对现有测试运行的窄命令

以下均**未在本次调查执行**。

```sh
# Radix / page manager
cargo test --locked -p ferrule-runtime --lib cache::radix::tests
cargo test --locked -p ferrule-runtime --lib cache::page_manager::tests

# Driver prefix 与 preemption
cargo test --locked -p ferrule-runtime --lib prefix_
cargo test --locked -p ferrule-runtime --lib kv_reserve_evicts_cached_prefix
cargo test --locked -p ferrule-runtime --lib driver_preempt_restore

# 不需要链接 CUDA 的真实 CUDA metadata planner 测试
cargo test --locked -p ferrule-backend --test cuda_kv_metadata

# CPU physical pool、CLI golden comparator、expert trace 模拟
cargo test --locked -p ferrule-backend --lib cpu::kv::tests
cargo test --locked -p ferrule-cli --bin ferrule bench::interactive_trace
cargo test --locked -p ferrule-model --test prefetch_evaluation -- --nocapture
```

已有关键测试证据：

- COW、snapshot source 生命周期、retirement、preempt/restore：
  [`page_manager.rs:2231–2497`](Ferrule/crates/ferrule-runtime/src/cache/page_manager.rs#L2231)。
- 命中后只执行 suffix：
  [`driver.rs:9450–9503`](Ferrule/crates/ferrule-runtime/src/engine/driver.rs#L9450)，明确断言第二次只执行 `[4,5]`。
- 压力下先淘汰 prefix 再 admission：
  [`driver.rs:9527–9554`](Ferrule/crates/ferrule-runtime/src/engine/driver.rs#L9527)。
- CPU-only CUDA planner：
  [`cuda_kv_metadata.rs:1–5`](Ferrule/crates/ferrule-backend/tests/cuda_kv_metadata.rs#L1) 用 `#[path]` 引入生产 planner。
- 实际 GPU shadow/COW/retirement：
  [`cuda_kv_transactions.rs:112,162,210,261`](Ferrule/crates/ferrule-backend/tests/cuda_kv_transactions.rs#L112)，均为 ignored GPU tests。

### 7.2 仓库正式入口

```sh
just check
FERRULE_NO_CUDA=1 just ci-cpu
just lint

# 仅编译和链接，不代表 GPU 执行通过
just ci-cuda-compile sm_86

# 要求真实 GPU；使用 audited nextest selection
just test-cuda-required

# 现有端到端 benchmark
just dsv4-runtime-driver-bench
```

配置证据：

- CPU lane：[`justfile:88–99`](Ferrule/justfile#L88)，要求 `cargo-nextest`、`rg`，运行 workspace all-targets 与 doctests。
- CUDA compile lane：[`justfile:116–129`](Ferrule/justfile#L116)。
- GPU smoke selection：[`nextest.toml:17–33`](Ferrule/.config/nextest.toml#L17)，包含 `cuda_kv_transactions`。
- `just test-cuda` 可以仅输出 `UNVERIFIED`，而非真正运行 GPU tests：[`justfile:211–218`](Ferrule/justfile#L211)。
- Rust 固定为 `nightly-2026-08-01`：[`rust-toolchain.toml`](Ferrule/rust-toolchain.toml)。
- CPU 构建也会编译 native C++：[`build.rs:14–15,126–143`](Ferrule/crates/ferrule-backend/build.rs#L126)。
- CUDA 需要 NVCC、CUDA 库、指定／探测到的架构、CUTLASS headers：[`build.rs:29–47`](Ferrule/crates/ferrule-backend/build.rs#L29)。
- CUTLASS 固定 `v4.6.1`；setup 缺 checkout 时会联网下载并写入 `target/vendor/cutlass`：[`setup_cutlass.sh:4–7,53–64`](Ferrule/scripts/setup_cutlass.sh#L4)。

### 7.3 实施前最重要的风险

1. **容量不能只按 committed tokens 算。**  
   [`composition.rs:235–274`](Ferrule/crates/ferrule-runtime/src/engine/composition.rs#L235) 的 `TransactionCapacity` 使用：
   ```text
   full_pages = ceil(ctx_size / page_size) × max_active_sequences
   configured_pages = 2 × full_pages
   ```
   第二份给 transaction shadows；预算不足直接拒绝。

2. **evicted entry charge 不等于实际释放的物理 pages。**  
   页面可能仍被活跃 sequence 或其他 snapshot 引用；只有 refcount 降至零并完成 backend retirement 才可复用。

3. **offload 不等于归还显存，也不等于增加 runtime admission credits。**  
   当前固定 pool + 保留逻辑 ownership 的设计必须先调整计费模型，才能承诺超出 GPU resident 容量的分层缓存。

4. **异步失败不能重放来“恢复”。**  
   session、KV transaction、CUDA graph 都有 unknown/quarantine 语义；必须保留原 ownership 和 completion 证据。

5. **模型能力不能靠打开配置获得。**  
   Qwen3.5 recurrent state、pipeline owner 协议都需要独立支持证明，不能删除 factory 拒绝条件就宣布 prefix cache 可用。

6. **现有 benchmark/golden 不足以证明 cache 策略有效。**  
   需要明确开启受支持的 prefix 路径，记录复用 token、真实释放页、迁移 bytes/latency，并修正或外部严格检查 golden 的轮数和失败状态。

**总体判断：**这不是“从零补一个 KV cache”的仓库。最合适的扩展边界是现有 runtime cache/driver authority；真正缺少的是面向策略实验的统一事件记录／回放，以及与现有 ownership、预算、异步完成协议兼容的自动分层驻留策略。

## Assistant

<thinking>
**Correcting acknowledgement download**
</thinking>

收到，`target` 是下载和研究的工作目录，不是已有论文的位置；刚才我理解错了。仓库侧已确认有 prefix cache、分页 KV 和会话级 CPU spill，但它们的容量与淘汰语义并不等同于论文模型；接下来我会核对原文，再给出分阶段实现方案，暂不改动推理代码。

**Tool Call: 下载并审阅 KVSET 论文**
Status: Failed

User canceled

**Tool Call: 下载并审阅 InplaceKVCache / WriteScope**
Status: Failed

User canceled

## User

这俩论文我下好了：/root/Ferrule/target/inplacekvcache.pdf 和kvset.pdf 都在target下

## Assistant

好的，就审阅 `target/inplacekvcache.pdf` 和 `target/kvset.pdf`，不再下载。我会并行核对两篇原文，再结合仓库已有的缓存与 offload 机制，给出可落地的实现方案和验证标准。

**Tool Call: 审阅本地 KVSET 原文**
Status: Completed

下面补完审阅，并纠正上一条中会影响实现正确性的地方。**本次仍然只做研究，没有修改 Ferrule 源码。**

## 12. 对前文的关键修正

### 12.1 Prefix target 不能选“距离最小的任意 k 页”

上一条的：

```text
sort(depths)
required_capacity = sorted_depths[k - 1] + 1
```

只适用于“请求中任意 k 次 page access 命中”的目标，**不适用于连续 prefix 复用**。

论文第 4 页 §2 的措辞是：

> maximum LRU depth among the prefix pages required to achieve the target hit rate

因此，若目标是保留前 `k` 页，正确推导应为：

\[
C_r(k)=1+\max_{0\le j<k} d_{r,j}
\]

如果这前 `k` 页中有 cold page，则该目标不可达。不能跳过前面的 miss，用后面的 hit 补足 prefix。

举例：

```text
初始 MRU→LRU: B, C, A
本次 page stream: A, B

依次访问时的 depth: 2, 1
```

- 命中任意一页：容量 2 可以命中 B。
- 命中第一个 prefix page A：容量必须为 3。

**建议分别命名：**

```text
page_reference_hit_rate
contiguous_prefix_token_hit_rate
request_target_coverage
```

不要共用一个含糊的 `hit_rate`。

### 12.2 Figure 2 的 “Capacity 1/2” 不是已明确的 1 页/2 页

第 4 页图 2 写的是 `Depth 1 corresponding to capacity 1` 和 `Depth 2 corresponding to capacity 2`，表示两个容量截断位置。图中还有省略号，**不能据此确认它们的数值就是 1 page、2 pages，也不能据此认定图采用一基 rank**。

可靠的 convention 证据仍然是第 3 页 §2：

- distance 是比该页更近的 distinct page 数；
- 命中条件为 `distance < C`。

实现应据此统一为零基 distance。

### 12.3 “先计算整个 request，再统一更新”不是一般 Mattson 算法

前文这条建议过强，应撤回。

论文第 3 页描述的是：

> Upon each access, the corresponding page is moved to the top of the stack.

因此，**标准顺序 page-reference 模型必须逐页查询、逐页更新**。先对整个 request 查询旧栈，再批量更新，会得到另一种模型。

但实际 prefix lookup 可能是在 request 开始时一次性查询，KV 在后续执行、提交后才发布。这意味着必须区分：

| 模型 | 访问/更新语义 |
|---|---|
| 标准 page-LRU replay | 每次 reference 判断 hit，随后 touch/插入 |
| Request-start residency 分析 | 在请求开始时检查可用 prefix，随后按指定规则更新 |
| Ferrule 实际运行 | lookup、prefill、commit/admission、pin/unpin 等真实事件 |

**论文没有充分规定如何把并发推理过程转换为第一种模型。** 因此，只按 admission 完成顺序重排请求，不足以保证与实际运行等价；它至多定义了一个可重复的分析顺序。

### 12.4 窗口过期不一定要删除 LRU 历史

应区分两个完全不同的窗口：

1. **统计窗口**：只统计最近 `W` 个请求的结果，但 LRU 历史继续保留。
2. **缓存历史窗口/TTL/reset**：历史状态本身过期或清空。

在线容量规划通常首先需要第一种。此时仅移除旧请求对 histogram/quantile 的贡献，**不能同时删除它们对历史 recency 的影响**。

如果实现第二种，就改变了缓存模型，不再只是“为普通 LRU 增加一个报表窗口”。

第 5–6 页及图 4 没有说明窗口规则。曲线下降本身也不能证明使用了滑动窗口：累计样本分位数在加入更多低值后，同样可能下降。

---

## 13. 更适合实现的目标定义与伪码

以下是**建议规格，不是论文给出的正式算法**。

### 13.1 分开两个参数

建议明确提供：

- `α`：每个请求需保留其**理论可复用 prefix**的多少比例。
- `p`：满足上述目标的请求覆盖率。

论文第 7 页 Related Work 提到：

> a specified fraction of requests to retain a target fraction of their theoretically reusable prefix

第 6 页实验讨论的是保留 theoretical hit rate 时的 `95% / 99% / 99.9%` 请求覆盖率。不能把 `α` 和 `p` 合并成一个百分比。

对于请求 `r`：

```text
R∞[r] = 无限容量、相同发布规则下，可复用的连续 prefix token 数
T[r]  = ceil(α × R∞[r])
```

这里 `R∞` 必须依赖**此前已经可用的内容**，不能利用未来提交的 KV，也不能只根据当前物理缓存 lookup 的结果推断。

### 13.2 求每个请求所需容量

假设已经通过约定好的 page-reference 模型得到有序 `depths`，并且使用完整、固定大小的 page：

```text
required_prefix_capacity(depths, theoretical_prefix_pages, α):
    validate 0 <= α <= 1

    if theoretical_prefix_pages == 0:
        return NoReusablePrefix

    k = ceil(α * theoretical_prefix_pages)

    if k == 0:
        return Finite(0)

    required = 0

    for j in 0 .. k:
        if depths[j] == COLD:
            return Unreachable

        required = max(required, depths[j] + 1)

    return Finite(required)
```

若目标是**绝对命中率**，例如“输入 tokens 的 90% 必须复用”，则先计算所需前缀长度，再与 `R∞` 比较：

```text
target_tokens > R∞:
    return Unreachable
```

对于 token-weighted 或 partial-page 模式，应先根据 page 的有效 token 数确定需要覆盖的 prefix page 边界，而不是简单按页数乘比例。

### 13.3 求 workload coverage

```text
capacity_for_coverage(request_results, p):
    validate 0 <= p <= 1

    population = apply_explicit_population_policy(request_results)

    if population is empty:
        return NoData

    if p == 0:
        return Finite(0)

    k = ceil(p * len(population))

    # 不可达结果排在所有有限容量之后
    result = kth_smallest(population.required_capacity, k)

    if result is Unreachable:
        return UnreachableCoverage

    return result
```

必须明确 `NoReusablePrefix` 的处理：

- 排除，统计“有复用机会的请求覆盖率”；或
- 纳入，认为其零复用目标可被 trivially satisfied。

后一种可能让大量 cold requests 虚增 coverage。**建议同时报告两种分母，并单独输出 cold/no-reuse request 数。** 论文没有给出这个选择。

---

## 14. Online 与 offline 应共享什么

**原文证据：**第 1 页摘要、第 3 页贡献、第 7 页结论声明支持 online request processing 和 offline trace replay，但没有给出两套不同算法或完整 trace schema。

**建议：共享一个 analyzer 状态机，仅输入来源不同。**

```text
Offline:
    read ordered trace
    → identity normalization
    → chosen reference model
    → analyzer
    → report

Online:
    observe runtime events
    → identity normalization
    → chosen reference model
    → analyzer
    → periodic report
```

需要分别保证：

1. 同一事件序列在线、离线结果相同。
2. reference-model replay 与一个朴素 page-LRU simulator 相同。
3. reference model 与 Ferrule 实际行为的偏差被量化，而不是假定不存在。

### 不要只记录实际 cache hit pages

如果 trace 只记录成功 lookup/admission 的缓存项，它会遗漏容量不足时的需求，估计结果受到当前容量限制。

尤其当前默认：

```text
prefix_cache_capacity_pages = 0
```

若只在缓存 admission 成功后产生分析记录，可能完全无法分析“打开缓存后会怎样”。

建议分开记录：

```text
demand:
    请求需要哪些逻辑 prefix pages

availability/commit:
    哪些 KV 已经完成、何时可供后续请求复用

actual cache outcome:
    当前配置真正命中了多少
```

“理想无限容量可复用历史”与“当前缓存实际收录历史”应分别建模。失败、取消及未 committed 的生成不能被当作已可复用内容。

---

## 15. 复杂度补充：可在线不等于有界内存

前文的 `O(N + U + R)` 是**一种直接实现**，不是所有 exact 实现的必然空间下界：

- `N`：page references。
- `U`：不同 logical pages。
- `R`：request observations。

| 实现选择 | 每次 reference | 主要状态 |
|---|---:|---:|
| 全局访问位置 Fenwick | `O(log N)` | `O(N + U)`，另加报表状态 |
| 带子树大小的 recency/order-statistics tree | `O(log U)` | `O(U)`，另加报表状态 |
| Fenwick 定期重编号/压缩 | 查询通常为 `O(log U)` 量级；需计入重建成本 | 可避免按全部历史 references 保留时间轴 |

这些替代实现是工程推导，论文第 3 页没有说明是否采用它们。

即使降到 `O(U)`，全新 logical pages 持续出现时，状态仍会增长。**固定报表窗口并不能自动限制身份历史的大小。**

还需注意 Fenwick 的动态扩容：

> 不能只把数组尾部补零，就假定新的 Fenwick 节点已经包含旧区间的计数。

应使用正确的扩容初始化或重建，并测试容量翻倍边界。离线可预分配，在线则必须处理这个问题。

---

## 16. 补完建议测试

在前文测试清单之外，优先补这几项：

| 测试 | 需要固定的行为 |
|---|---|
| Prefix hole | 前页 miss、后页 hit 时，后页不能补足连续 prefix 目标 |
| Cold 请求分母 | `NoReusablePrefix` 纳入/排除后 coverage 的差异 |
| `α` 与 `p` | 提高单请求保留比例、提高请求覆盖率，容量都不应下降 |
| 有限 trace 不可达 | 请求级不可达与 coverage 级不可达分别返回 |
| 报表窗口过期 | 移除旧统计，不误清空 recency/history |
| Cache reset | reset 后首次复用与未 reset 模式明确不同 |
| Fenwick 扩容 | 在 2 的幂边界前后，与朴素 LRU 保持一致 |
| 身份命名空间 | 相同 tokens、不同 model/adapter 不得复用 |
| 相同页 tokens、不同前缀 | full-attention KV 不得误认为相同内容 |
| 默认缓存关闭 | 仍可收集 demand，实际 hit 保持为零 |
| 并发 publish | 未提交页不能提前命中；离线忠实保留事件顺序 |
| Trace 丢事件 | 标记结果不完整，而不是继续声称 exact |
| 内存增长 | 全新页流、重复热页流、长请求流分别测 analyzer 状态增长 |

### 本次实际做的验证

我用本地 Python 标准库运行了一个独立数学模型检查：

- 200 条随机 reference streams；
- 每条 100 次访问；
- 对容量 `0..12`；
- 比较 Fenwick distance 判断与朴素顺序 page-LRU；
- **共 260,000 次 access/capacity 比较，全部通过。**

也验证了上述 prefix 反例：选择最小 distance 会错误低估连续前缀所需容量。

这仅验证**标准顺序 page-LRU 的公式及反例**，不是 Ferrule 集成测试，也不是论文作者实现验证。

---

## 17. 推荐实施路线

按你提供的现有架构，我建议将后续工作分为三步，而不是直接改缓存策略：

### 第一步：离线研究型 analyzer

- 明确 logical content identity、page 边界、cold 和 coverage 分母。
- 用小型 synthetic traces 与朴素 page-LRU 对照。
- 输出：
  - page-reference hit curve；
  - 连续 prefix token hit curve；
  - 每请求 required capacity；
  - request coverage；
  - cold/no-reuse/unreachable 数量；
  - analyzer 自身内存占用。

### 第二步：Ferrule trace + 实际策略对照

基于你描述的：

- `crates/ferrule-runtime/src/driver/prefix.rs`：需求及 lookup 结果；
- `crates/ferrule-runtime/src/driver/admission.rs`：发布结果；
- `crates/ferrule-runtime/src/driver.rs`：committed prefill frontier；
- `crates/ferrule-runtime/src/cache/radix.rs`：实际 entry-LRU；
- `crates/ferrule-runtime/src/cache/page_manager.rs`：共享、COW、retirement。

建立对照，而非把论文模型直接接管 eviction。重点测量：

```text
论文式 page-LRU 估计
    与
Ferrule entry-LRU + charge + sharing + pinning
```

之间的偏差。

### 第三步：确认价值后再讨论在线集成

先测：

- 每 page/request 的 CPU 开销；
- event queue 是否会积压；
- identity hashing 成本；
- 内存是否可控；
- 估计与实际命中的误差。

在线分析输出应首先作为**观测和容量建议**，不要直接驱动自动扩缩容或淘汰决策。

---

**最终判断：**这篇论文足以支持“用 Mattson/Fenwick 分析理想 page-LRU 容量”的方向，但不足以作为 Ferrule 的完整实现规格。最重要的前置工作不是 Fenwick tree，而是明确 **logical page identity、连续 prefix 目标、事件时序、coverage 分母，以及论文 page-LRU 与现有 entry-LRU 的区别**。

本次研究产物仅位于 `target/`：`kvset.txt`、PDF 提取辅助脚本和四张原始图示。没有访问论文中的外部链接，没有修改项目源码，也没有运行不相关的 Cargo 构建或测试。

**Tool Call: 审阅本地 InplaceKVCache 原文**
Status: Completed

## 1. 先纠正原文身份：指定文件与目标论文不一致

已先读 `AGENTS.md`。本次没有联网、没有修改源码；只在 `target` 中生成解析工具和提取文本。Ferrule 架构分析采用你提供的背景，没有重做源码全仓调查。

| 本地文件 | 实际内容 |
|---|---|
| `target/inplacekvcache.pdf` | **Fractal KV-Cache Archives: Lossless Symbolic Storage with In-Place Retrieval for Long-Context LLM Inference**；作者 **Vladimir Gusev**；10 页。封面日期 **2026-08-07**，页边 arXiv 标记为 **2607.07144v2，2026-08-05**。不是 WriteScope 论文。 |
| `target/paper-review/writescope/2609.14507v1.pdf` | **Physically Partitioned KVCache Format for CPU–GPU Load Balancing in MoE Inference**；作者 **Enda Yu、Dezun Dong、Xiangke Liao**，国防科技大学，Dong 为通讯作者；15 页。首页标记 **arXiv:2609.14507v1，2026-09-13**；PDF 生成时间为 09-15，不应混作论文日期。 |

下面评审的是**第二个已存在的本地 PDF**。页码均指该 PDF 页码，与印刷页码一致。它的 [带页码提取文本](Ferrule/target/paper-review/writescope/extracted.txt) 已保留。

**总体判断：值得借鉴，但它不是“给现有 preempt 增加自动触发”就能实现的功能。核心是新的物理 KV 存储格式、同一次 attention 的 CPU/GPU 协作执行，以及与之配套的容量和事务协议。**

---

## 2. 四类 region 的精确定义

### 论文事实

**InplaceKVCache 是存储格式/抽象；WriteScope 是其上的调度系统。** 两个分类维度是：

- **device affinity**：CPU DRAM 或 GPU VRAM 侧的物理归属，不是 OS 线程绑核。
- **access pattern**：计算消费或传输用途，不是读写冷热、LRU 或访问频率。

对应关系如下。**[p.4 §4.1、Fig.4；p.5 §4.2–4.3]**

| Region | 精确分类 | 存放内容、用途与生命周期 |
|---|---|---|
| **Unpinned compute region** | CPU × compute | 普通未锁页 DRAM；长期存放分配给 CPU attention 的历史 KV，由 CPU 原地读取计算。 |
| **Pinned load region** | CPU × transfer | 锁页 DRAM，论文使用 `cudaHostAlloc`；长期保存分配给 GPU attention 的历史 KV，作为连续 H2D DMA 的源。**它仍然在 CPU 内存里，不是 GPU 常驻 KV。** |
| **D-window offload region** | GPU × transfer | 新 decode KV 的暂存窗口；积累到 \(D\) 后成批首次放置到上述两个 host region 之一。窗口内 KV 同时参与 GPU attention，不需要 H2D。 |
| **Ping-pong compute region** | GPU × compute | PP0/PP1 流水工作缓冲，交替接收 pinned region 的历史 KV 并执行 GPU attention；覆盖复用，而不是保存全部 GPU 份额。 |

三个容易误解的地方：

1. **不是四份 KV，也不是四级缓存。** 历史 KV 的长期 host 归属只有一份；GPU streaming 时有临时副本。
2. **“放置后不移动”不是没有 PCIe 传输。** 它指不为调比例而反复改变历史数据的权威归属、pin 状态和 layout；pinned KV 仍会被反复传到 GPU 计算。
3. **region 分类不排斥附带操作。** D-window 虽标为 transfer region，正文明确说它也参与 GPU attention。

### Layout：能确认什么，不能确认什么

**可确认：**

- 四个 region 内部均以固定大小 block 组织，block 是 routing、staging、eviction、transfer 的统一基础粒度。**[p.4 §4.1]**
- CPU 消费侧要求适合按 head 读取；CPU kernel 支持 stride-aware addressing，不要求先把整个输入复制成连续张量。**[p.2 §1；p.6 §6.1]**
- H2D 侧关键是**每次实际传输的有效数据连续**，不能把最大容量 buffer 的多个“有效前缀 + 空洞”误当成一次连续 DMA。**[pp.3–4 §3.2]**
- pinned region 按批增长；unpinned、ping-pong、D-window 在初始化时预分配。**[p.5 §4.3]**

**不能据此写死具体 ABI：**正文没有充分给出可直接照抄的张量维度顺序、K/V plane packing、alignment、所有 stride 和 block-token 常量。Fig.5 是嵌入图片，本次文本解析未读取其中伪代码；四操作 `init/append/views/flush` 的确认来自 §4.3 正文。

**对 Ferrule 的建议：**不要直接推导出“必须改成某个 `[B,H,S,D]` 顺序”。先定义 CPU view 和 DMA block 的契约；允许一个 transfer block 聚合多个逻辑 page，避免小 page 导致大量小 DMA。

---

## 3. Split 粒度与决策时机

### 论文事实

**沿 sequence/token 维度分配，不沿 head 维度分配。**

CPU 和 GPU 都计算所有相关 heads，但消费互不重叠的 sequence 部分。原因是 GQA 的 KV heads 少，按 head 分配过粗；MLA 的共享 latent 更不适合这样拆。**[p.5 §5.1]**

要区分三种粒度：

- **基础 block**：格式粒度。
- **\(D\)-window**：decode 成批首次放置的粒度；默认 \(D=256\)，是 block size 的整数倍，论文考察 \(256–1024\)。**不能把 \(D=256\) 当成 page size。**
- **控制策略**：初始校准 + decode 持续测量；不是仅在显存耗尽时决定。

时机是：

1. **Admission / prefill 前**：离线校准给出初始份额 \(R(L_0)=r(L_0)\)。
2. **Prefill**：每层 KV 在产生并供当层 attention 使用后，直接路由到最终 host 区域；不先构建一个完整历史缓存、随后再全量重排。
3. **Decode**：新 KV 先进入 GPU D-window；窗口填满后，整个批次进入 pinned 或 unpinned region。
4. **后续调整**：改变新批次的去向，**历史批次不重新迁移**。**[pp.4–6 §4.2、§5.3；p.7 §6.2]**

论文称 \(D\) 是 routing granularity 而不是 decision unit：可在窗口期间测量性能，并不需要等 flush 才获得反馈。

**实现含义：**动态批次路由以后，两类 token 不必始终构成一个按原始 token 位置划分的单一切点。Ferrule 应保留 block/token-position 映射，不能只用一个 `split_index` 表示所有历史归属。

---

## 4. Attention / softmax 如何正确合并

### 论文明确说到的内容

CPU 原地计算自己的部分，GPU 计算 D-window 和 streamed pinned 部分；CPU 返回 partial output 与 normalizers，在 GPU 上作 LSE rescaling 合并。CPU kernel 使用 online softmax、running max、normalization，输出以 FP32 累积。**[p.5 §4.2、§5.1；p.6 §6.1]**

论文正文**没有展开完整合并公式**。下面是实现该语义所需的数学推导，不是论文原文公式。

对一个 query/head，令分区 \(P_j\) 覆盖有效 keys，且两两不重叠：

\[
s_i = q^\top k_i/\sqrt{d} + b_i
\]

其中 \(b_i\) 包含该模型要求的 mask/bias。每个分区计算：

\[
m_j=\max_{i\in P_j}s_i,\quad
\ell_j=\sum_{i\in P_j}e^{s_i-m_j},\quad
u_j=\sum_{i\in P_j}e^{s_i-m_j}v_i.
\]

合并时：

\[
m=\max_j m_j,
\quad
\ell=\sum_j e^{m_j-m}\ell_j,
\quad
u=\sum_j e^{m_j-m}u_j,
\quad
o=u/\ell.
\]

若接口返回归一化输出 \(o_j=u_j/\ell_j\) 和
\(z_j=m_j+\log\ell_j\)，则：

\[
z=\operatorname{logsumexp}_j z_j,\qquad
o=\sum_j e^{z_j-z}o_j.
\]

**绝不能直接相加或平均各分区的 softmax 输出。**

实现注意事项：

- 同一公式也用于合并多个 GPU ping-pong chunks，最后再合并 CPU 结果。
- normalizer 是**每个 query/head**的状态，不是每层一个标量。
- 空分区需有明确 neutral state；全 mask 行必须特别处理，避免 `-inf - -inf` 产生 NaN。
- RoPE、GQA head 映射、MLA 投影、mask、位置编号必须与原 attention 一致。
- 论文的 exactness 指不额外删 KV、不加有损压缩，**不保证浮点逐 bit 相同**。

我用 Python FP64 检查了这一推导：300 个三分区随机案例，包含空分区与大幅度 logits，最大绝对误差为 **\(6.66\times10^{-16}\)**。这只验证合并代数，不是对论文 kernel 或 Ferrule 的验证。

---

## 5. Roofline + feedback：公式成立，但响应速度有硬约束

### 论文模型

令 \(L\) 为已完成放置的历史长度，\(r\) 为 CPU 份额：

\[
T_{\rm CPU}=rAL+C_{\rm CPU},
\qquad
T_{\rm GPU}=(1-r)BL+C_{\rm GPU},
\]

\[
T=\max(T_{\rm CPU},T_{\rm GPU})+T_{\rm merge}.
\]

- \(A\)：CPU 每 KV token 的有效处理成本，会受 working set/cache locality 影响。
- \(B\)：GPU 路径每 token 成本，主要由 H2D 决定；假设 GPU compute 能藏在 streaming 后面。
- \(\Delta=C_{\rm GPU}-C_{\rm CPU}\)。

平衡点：

\[
r(L)=\operatorname{clip}_{[0,1]}
\left(
\frac{B}{A+B}+
\frac{\Delta}{(A+B)L}
\right).
\]

论文测得 merge 为 **0.2–1.0 ms**，模型中认为与 \(r\) 无关。**[p.6 §5.2–5.3，Eq.1–2]**

控制流程：

- 离线扫描模型/数据规模校准初始值。
- 单用户稳定场景下 \(B\) 校准一次。
- decode 期间对 CPU component latency 作滑动平均并反推 \(A\)；正文称跨层 256 steps 足够估计。
- 新批次的边际 CPU 比例趋向 \(p=B/(A+B)\)。
- 并发实验中改为**同时监控 \(A,B\)**，因为 PCIe 也会发生竞争。**[p.6 §5.3；pp.10–11 §7.8]**

### 我的审查意见：不能把“feedback”理解为立即重分配

在 \(p\) 不变时，论文自身给出：

\[
R(L)=p+(R(L_0)-p)L_0/L.
\]

这意味着初始偏差按 \(L_0/L\) 衰减。以 \(L_0=32768,D=256\) 为例，经过 4 次 flush，偏差仍剩 **96.97%**。

因此：

- 初始校准准确、负载缓慢变化时，可以很好跟踪。
- 对已经很长的上下文，**只改新写入很难快速纠正历史 placement**。
- 论文“前几个 batch 消退初始误差”的表述不能当作普遍收敛保证。
- 并发实验的快速响应值得复现，但不足以证明任意突变都能快速适配。

还有一个具体的可达性约束。若当前 CPU 占有 \(C\) 个历史 tokens，下一批大小 \(D\)，不迁移旧数据，则：

\[
R_{\rm next}\in
\left[\frac{C}{L+D},\frac{C+D}{L+D}\right].
\]

整批二选一时，当次只能取两端之一。**调度器应把理论最优、实际份额、下一步可达份额分开记录。**

---

## 6. CPU/GPU 传输、线程、NUMA 与 pinned memory

### 原文证据

| 项目 | 原文实际报告 |
|---|---|
| Pinned H2D | **25.4 GB/s**，unpinned **10 GB/s**；CPU 读取 pinned 在长序列下慢 **10–25%**。**[p.3 §3.1]** |
| 连续 vs strided DMA | 部分填充的 strided buffer 约 **4.6 GB/s**，连续 buffer **25.4 GB/s**。**[pp.3–4 §3.2]** |
| Transfer size | 192 KB：**8.7 GB/s**；1 MB：**22.2 GB/s**；8 MB：**25.4 GB/s**。**[p.6 脚注5]** |
| Decode D2H | 作者称借助 PCIe full-duplex 藏入 expert phase；\(D=1024,bs=32\) 时每层 37.8–134.2 MB，Gen4 **1.5–5.3 ms**，Gen3 **2.9–10.3 ms**。**[p.5 脚注3]** |
| Pinning | 提前一个 D-window 分配；测得 **0.66 ms/MB**，最大一次 **1074 MB / 711 ms**，靠长 lookahead 隐藏。**[p.10 §7.7]** |
| CPU kernel | 自定义 AVX-512，stride-aware 读取、online softmax、FP32 累积；MLA 使用 8-token tile 共享 compressed latent。**[pp.6–7 §6.1]** |
| 线程 | 仅明确报告：8–48 threads 对 \(A\) 的影响小于 5%，测试点为 Qwen3/DS2、bs16、seq16K。**[p.6 脚注4]** |
| NUMA | **没有给出 WriteScope 自身的 NUMA 分配/绑核协议。** “NUMA-aware” 出现在 HybridGen 基线的 CXL 说明中。**[p.14 附录A]** |

### 对 Ferrule 的建议与保留意见

- **“CPU 读取 pinned 必然慢”不是普遍硬件定律。** 论文测得的差距可能包含 allocator、page placement、NUMA、缓存策略等因素；尤其应区分普通 cacheable pinned 与 write-combined 内存。
- full-duplex 不等于全链路无竞争。D2H、H2D、CPU attention、专家 GEMM 仍可能争用 host DRAM、root complex、copy engines 或线程资源。
- CPU attention 需要 query，通常涉及 Q 的 D2H；CPU partial output 和 LSE 需要 H2D。原文交代返回 partial 的语义，但没有充分展开 Q/normalizer 的 staging 与事件协议。
- **向 unpinned 最终目的地 D2H 的真正异步性需要单独验证。** 不能默认 pageable destination 与 pinned destination 有同样的 overlap。若需要有限 pinned staging，应把这部分 copy、容量和延迟显式计入，而不是称为免费。
- 建议独立设计 CPU attention/expert/transfer worker 的线程预算，测绑核与 first-touch；在多 NUMA 节点机器上分别测 GPU-local、CPU-local、remote 分配。
- `runtime pin` 是生命周期约束；`pinned host memory` 是 OS/CUDA 锁页属性，**必须使用不同概念和字段**。

---

## 7. 实验实际数据与结论边界

### 实验条件

**[p.7 §7.1]**

- A100 PCIe Gen4×16，40 GB 卡限制到 **32 GB VRAM**。
- V100-32GB，Gen3×16，**同一台 CPU host**。
- 单颗 Xeon Gold 5318Y，24 cores，**502 GB host RAM**。
- 模型：DeepSeek-V2-Lite、Qwen3-30B-A3B、Mixtral-8×7B；另有 DeepSeek-V4-Flash-Int8 稀疏 attention 案例。
- WriteScope 与四个重实现基线 KV VRAM 上限为 **2 GB**，DS2 为 **0.3 GB**；原生框架获 **4 GB**。
- ShareGPT 2K prompt 后 decode；8K/32K 数据是**累计端到端吞吐**，不是这些长度处的瞬时 TPOT。
- 每配置 10 次；作者报告 bootstrap 95% speedup CI 在 ±5% 内。

**“1M context”是 `bs32 × seq32K` 的 aggregate scale，不是单请求百万 token。**

### Decode 几何平均加速比

长上下文 ≥8K，跨三个模型、bs4/16/32。**[p.8 §7.3]**

| 对比基线 | A100 | V100 |
|---|---:|---:|
| SolidAttention | 1.52× | 1.55× |
| ScoutAttention | 2.22× | 1.43× |
| HybridGen | 2.48× | 1.67× |
| FastDecode | 2.51× | 1.71× |

这些是**同一 runtime 上重实现的机制比较**，不是直接运行四个上游项目得到的结果。

附录披露的重要变化包括：

- SolidAttention：SSD 改成 DRAM，偏向基线。
- Solid/Scout：保留 80% KV；WriteScope 使用完整 attention。
- HybridGen：没有 CXL tier。
- FastDecode：多节点缩成单节点，**不利于基线**。
- Scout 去 AF 的版本被作者当作 HGCA 范式代表，不是 HGCA 原生实现。**[pp.14–15，Tables 4–5]**

### 内存占用

`seq32K, bs32, D256`。**[p.10 Table 2]**

| 模型 | GPU ping-pong | GPU D-window | Host unpinned | Host pinned |
|---|---:|---:|---:|---:|
| DS2 | 37.8 MB | 255 MB | 15.95 GB | 16.66 GB |
| Qwen3 | 67.1 MB | 805 MB | 61.95 GB | 41.13 GB |
| Mixtral | 134.2 MB | 1074 MB | 103.08 GB | 34.36 GB |

GPU KV residency 约 **0.29 / 0.85 / 1.18 GB**，代价是几十到上百 GB host KV，其中锁页内存本身可达几十 GB；另外还要容纳模型专家等资源。

### 其他有用数据

- Prefill routing vs ideal：DS2 bs4 **1365 vs 1366 tok/s**。这支持“在该 pipeline 中可隐藏成本”，不是证明写回没有成本。**[p.8 §7.2]**
- bs4、32K prefill：DS2/Qwen3/Mixtral TTFT **51/71/132 秒**。**[p.8 §7.2]**
- DS2 消融：固定比例 **0.89–0.96**、strided DMA **0.47–0.90**、每步 rebuild **0.53–0.87**、unpinned H2D **0.62–0.94**、DynamicCache **0.26–0.79** 的归一化吞吐。布局/复用收益大于单独调比例的收益。**[p.9 §7.4]**
- V4 ≥16K 的端到端收益仅 **1.10/1.24/1.30×**，明显小于单层 CSA 收益；不可用 kernel speedup 代替端到端收益。**[p.9 §7.5]**
- 双用户测试中 bs4/8K 为 solo **386 ms**、fixed **1010 ms**、adaptive **436 ms**；论文明确称并发只是可行性验证，非推荐部署模式。**[p.10 Table 3；p.11 §9]**

### 应保留的限制

1. 两个 GPU 平台、同一个 CPU，不能证明跨 CPU/NUMA/消费卡的普适性；RTX 3090/4090/5090 的讨论是外推。**[pp.10–11]**
2. V100 的 BF16 通过 FP32 promotion 执行；原生框架在该实验配置上还存在 sm80+ kernel 无法运行的兼容性问题，**不能把所有缺失点都解释为 OOM**。**[pp.14–15 附录A]**
3. sparse gather 35.1 GB/s 高于 H2D 25.4 GB/s，支持该机器上的流水隐藏；但只支持 block-level selection，不能推广到任意 token gather。**[p.5 §4.4]**
4. 没有看到完整的误差阈值、logit 对齐或质量回归结果；“数学上完整 attention”不能替代实现正确性验证。
5. 主机锁页失败、突发 admission、共享前缀/COW、取消与回滚等生产问题，论文没有给出足够协议。
6. 本次核实的是**原文报告了哪些数字和机制**，未复现实验性能。

---

## 8. 映射到 Ferrule：必要实现能力

以下均为建议，基于你给出的架构事实。

### A. 保留逻辑 page 层，新增物理 placement 层

`page_manager.rs` 继续拥有逻辑 page、refcount、COW。不要将四个 region 直接替换成四种逻辑 page。

需要额外区分：

- 逻辑 page 身份；
- 权威 KV 存放位置；
- CPU/GPU 哪条 attention 路径消费；
- 临时 GPU staging slot；
- in-flight read/transfer lease；
- 数据版本及完成事件。

同一逻辑 page 可以长期位于 host，而每次 attention 临时借用不同 ping-pong slot。临时 slot 不是新的逻辑 page。

### B. 现有 preempt/restore 只能复用部分机制

当前 session preempt 是“保存 snapshot、释放 slot，恢复时上传”，不支持 host 页在 session 活跃期间直接参与 attention。

它可贡献 snapshot 编码、传输和 ACK 经验，但：

- **不能把 preempt 等价成 CPU region placement。**
- 不能为每次 attention 先 restore 整个历史 KV；那会失去 streaming 的固定显存优势。
- 按 plane 下载 snapshot 不自动等于连续 pinned DMA 格式，也不自动适合 CPU attention 读取。

### C. 扩展资源记账与 admission，而不是只调 eviction

至少分开记账：

1. 逻辑 KV page/token 额度；
2. GPU 可用 slot；
3. **实际 GPU 已分配字节**；
4. host unpinned/pinned 容量、预分配 lookahead；
5. transfer、partial output、merge scratch 的峰值。

你指出的现状很关键：**preempt 释放 slot，不减少固定容量 plane buffer 的 VRAM；hard credits 也未释放。**

因此不能宣称“offload 了多少 KV，就释放了多少显存”。要得到论文式固定 GPU 工作集，应选择：

- 初始化时采用较小的 D-window + streaming workspace；或
- 后续实现真正可缩放/分段回收的 GPU allocation。

Admission 还必须预留下一批 host 容量；host 预算耗尽时拒绝/背压，不能无限依赖 host RAM。

### D. 将 placement 纳入既有提交协议

沿用：

`reserve → execute → logical prepare → backend commit → logical publish`

建议：

- reserve：同时预留目标 host block、D-window/streaming 资源及预算；
- execute：完成写入和必要传输，建立 completion dependency；
- logical prepare：暂存映射，不暴露未就绪数据；
- backend commit：目标数据已完成，或其就绪依赖受严格约束；
- logical publish：发布新物理 view；
- 旧 slot 只有在所有 reader/transfer 完成、backend release ACK 后才 reuse。

**flush 不一定是逻辑 page free**：可能只是同一 page 的物理归属变化。需要相应 relocation/lease 协议，不能套用删除逻辑 page 的语义。

live transaction、runtime pin、quarantine 继续保护；首版沿用 `refcount=1` 限制比较稳妥，共享前缀和 COW 跨 tier 支持另行设计。

### E. 新的 attention 执行能力是必需项

- CPU exact partial attention，直接消费 host views。
- GPU chunked attention，暴露 output + LSE 或等价状态。
- PP0/PP1 的 H2D 与计算事件流水。
- CPU query 输入、partial 返回与最终 merge。
- D-window 和尾块的有效长度/mask。
- GQA、MLA 等模型特有语义，不能仅靠通用 KV plane 搬运覆盖。

### F. 推荐拆分验证顺序

1. **容量基础**：host 预算、placement 元数据、事务/ACK，不宣称 WriteScope。
2. **纯 GPU streaming**：pinned host + 固定 GPU workspace，先证明长上下文与固定 VRAM。
3. **固定比例混合 attention**：先完成正确合并和 CPU kernel。
4. **离线校准 + 在线反馈**：加入可达性限制、迟滞、异常样本过滤。
5. 最后扩展共享页、多用户、NUMA、多模型并发。

这样可以区分“存储布局有效”“混合执行有效”“动态控制有效”三个独立命题。

---

## 9. 数值正确性与性能验证建议

### 数值与生命周期

- 小规模 FP64/FP32 reference，对比全 GPU、全 CPU、混合执行。
- 覆盖 \(r=0/1\)、任意 block 组合、空分区、尾块、D-window 跨界及多种 merge 顺序。
- 覆盖 GQA、MLA、RoPE 长位置、causal/sliding mask、极端 logits。
- FP32 累积/LSE；同时报告 absolute error、relative error、RMS 和最终 logits，不只检查生成文本一致。
- 对 CPU↔GPU transport 做 bitwise KV 校验；另行定义 attention 浮点误差容忍度。
- 故障注入：D2H/H2D 延迟、host/pinned 分配失败、取消、preempt/restore、backend commit 失败、release ACK 延迟。
- 检查“已发布必可读”“未 ACK 不 reuse”“rollback 不泄露 credit”“共享页不被误覆盖”。

### 性能与容量

分别测量，而不是只报总 tok/s：

- 实际 VRAM allocation、slot occupancy、host RSS、pinned bytes、scratch 峰值；
- CPU attention、H2D/D2H、GPU attention、merge、pin allocation；
- CPU DRAM 带宽、PCIe 带宽、overlap 空洞；
- TTFT、稳态 TPOT、P50/P95/P99，以及论文口径的累计吞吐。

必要对照包括：

- 当前 Ferrule GPU resident / preempt-restore；
- pure streaming GPU；
- CPU-only historical attention；
- fixed split 与 online split；
- 连续/分散、pinned/unpinned、不同 block/D/window 大小。

控制器测试应特别包含：**长上下文进入后突然改变 CPU/PCIe 负载**，比较理论最优与 immutable placement 下真正可达的最优，而非假定能瞬时重新切分。

**最终建议：先证明 Ferrule 能以受预算约束的 host KV 和固定 GPU workspace 正确执行长上下文 attention，再证明 CPU 分担带来收益，最后才接入 roofline feedback。** 现有逻辑 page/refcount/COW 与 ACK 后 reuse 是有价值的基础，但 snapshot/preempt 本身既不等于显存回收，也不等于 WriteScope 所需的混合 attention 能力。

## Assistant

<thinking>
**Requesting missing report kvset**
</thinking>

**Tool Call: 补齐 KVSET 核心证据**
Status: Completed

以下仅依据本地 `target/kvset.pdf`；**未联网核查**。页码均指 PDF 物理页码，共 9 页。

## 1. 标题、作者与日期

**实际标题：**  
*The KV Cache Working Set: Online Capacity Planning for LLM Inference Systems*

**作者：**Luchang Li、Shuaishuai Wang、Zhao Ruan、Dongfang Li、Bozhao Gong。  
**单位：**Kingsoft Cloud。通讯作者为 Luchang Li。  
**证据：第 1 页。**

**arXiv 日期：无法由这份 PDF 确认。** 该文件没有标明本论文的 arXiv 编号、提交日期或版本日期。参考文献中的其他论文 arXiv 编号不能用于确定本文身份。

PDF metadata 的创建、修改日期均为 **2026-09-23 19:43:21 UTC+08:00**。这只是文件元数据，**不能当作发表日期或 arXiv 提交日期**。

## 2. Fenwick 算法与原文复杂度

**第 3 页 §2 Method** 明确描述：

- 记录每个 page 最近一次访问的位置；
- 在 Fenwick tree 中维护访问位置；
- 统计比该 page 更近访问过的 **distinct pages**，得到 LRU stack distance；
- 每次访问后将该 page 移到 LRU 顶部；
- 命中条件为 **`distance < capacity`**，因此正文采用零基 distance。

**原文没有给出正式时间或空间复杂度，也没有给出伪代码。** 第 3 页声称低计算、低内存开销，但没有量化每次访问耗时、分析吞吐或内存占用。

因此，`O(log N)` 单次 Fenwick 操作、直接时间轴实现的 `O(N)` 空间等，均属于实现推导，不能引述为论文原文结论。论文也未说明在线扩容、压缩或历史回收算法。

## 3. Page identity 定义

**第 7 页 §4** 仅说明将 “token-prefix reuse” 转换为 “KV-page reference stream”；**第 3 页 §2** 使用“同一 page 的最近访问位置”。

论文**没有定义**：

- page key/hash 的构造方式；
- 是否包含完整前缀上下文；
- model/adapter 等命名空间；
- page 的 token 数；
- partial page 处理；
- 物理页号复用与逻辑内容身份的区别。

所以不能根据原文认定其 page identity 就是物理 block ID、局部 token hash 或某种指定的链式 hash。

## 4. 实验规模与误差

### 容量预测准确性

**第 4 页 §3、第 5 页图 3：**

- **10,000 个请求**，来自内部生产 coding-agent workload；
- 部署 **Mooncake**，为 **SGLang** 配置多个物理 KV cache 容量；
- **NVIDIA H20 GPU、GLM-5.2、W4A8**；
- **MTP 关闭，KV cache 使用 FP8**；
- 文中称 SGLang 下约 **60 KB/token**。

图 3 只有 **3 个橙色物理实验点**，横坐标目测约为 **400、800、1600 GiB**；这些是图上近似读数，不是正文数据表。第 5 页明确说，由于部署耗时、耗资源，只测了部分容量。

**缺失：**GPU/服务器数量、部署并发、trace 时间跨度、用户数量、重复次数、误差条、置信区间，以及 MAE、最大绝对误差、相对误差等数值。正文只有“closely match”的定性结论，不能写成某个百分比精度。

### 在线容量分析

**第 5–6 页 §3、图 4：**

- 另一项分析观察 **24,000 个请求**；
- 来源仍为内部生产 coding-agent 服务；
- 保留理论命中率的请求覆盖率为 **95% 时约需 1 TiB，99% 时约需 5 TiB**；
- **99.9%** 的需求在所观察 trace 内没有明显收敛。

这些是容量分析结果，**不是对应容量全部完成物理部署验证的证明**。论文没有说明该 trace 与前述 10,000 请求 trace 的关系，也没有给出统计窗口规则。

## 5. 原文提供的官方项目 URL

第 3 页“Open-source implementation”给出的作者项目地址：

**https://github.com/llc-kc/kv_cache_capacity_estimator**

本次未访问该链接。**这份 PDF 未提供可确认属于本文的 arXiv 或正式出版页面 URL，因此不能补写未经核实的论文链接。**

**Tool Call: 核实论文文件身份差异**
Status: Completed

是，`target/inplacekvcache.pdf` 的结论有可靠封面原文证据：第 1 页直接给出以下标题和 arXiv 标记；PDF 元数据也一致。

| 本地文件 | 封面原文标题与 arXiv 标记 | 官方 URL | 本地提取文本 |
|---|---|---|---|
| `target/inplacekvcache.pdf` | **Fractal KV-Cache Archives: Lossless Symbolic Storage with In-Place Retrieval for Long-Context LLM Inference**；`arXiv:2607.07144v2 [cs.LG] 5 Aug 2026` | https://arxiv.org/abs/2607.07144v2 | `target/inplacekvcache-extracted.txt` |
| `target/paper-review/writescope/2609.14507v1.pdf` | **Physically Partitioned KVCache Format for CPU–GPU Load Balancing in MoE Inference**；`arXiv:2609.14507v1 [cs.DC] 13 Sep 2026` | https://arxiv.org/abs/2609.14507v1 | `target/paper-review/writescope/extracted.txt` |

`target/paper-review/writescope/2609.14507v1.pdf` 在本次开始时已经存在；不是我或本次工具创建的。提取文本文件是本次审阅过程中生成的。

## Assistant

审阅后，我的判断是：**两条路线都值得做，但应拆成两个项目：先做 KVSET 式容量分析，再做 InplaceKVCache 式混合 attention。不要一开始就把二者合成自动分层缓存。**

前者主要补齐观测和容量建模，可以不改变推理行为；后者需要修改物理存储、attention 执行和资源记账，是一次较大的架构扩展。

## 先说明一个文件问题

`target/kvset.pdf` 确实是 **The KV Cache Working Set: Online Capacity Planning for LLM Inference Systems**。

但 `target/inplacekvcache.pdf` 的封面是 **Fractal KV-Cache Archives: Lossless Symbolic Storage with In-Place Retrieval for Long-Context LLM Inference**，不是你描述的 WriteScope 论文。

好在 `target/paper-review/writescope/2609.14507v1.pdf` 已经存在，实际标题为：

**[Physically Partitioned KVCache Format for CPU–GPU Load Balancing in MoE Inference](https://arxiv.org/abs/2609.14507v1)**，首页日期是 2026 年 9 月 13 日。

下面第二部分依据这份正确原文，而不是按文件名推断。

---

## 一、KVSET：适合先做，但必须明确我们估计的是什么

### 论文真正提供了什么

核心是经典的 LRU stack-distance 分析：

- 对每个逻辑 KV page，记录上次访问位置。
- 计算此后访问过多少个不同 page，得到零基距离 `d`。
- 对容量 `C`，命中条件为 `d < C`。
- 因而一次扫描可以得到多个容量下的理想 LRU 命中表现。

原文使用 Fenwick tree，但没有完整定义 page identity、并发事件顺序、在线历史回收和窗口语义。这些都需要我们补充，不能直接把论文当实现规格。

实验比摘要具体一些：

- 用 **10,000 个 coding-agent 请求**，对照 Mooncake + SGLang 的物理部署。
- 图中只有三个实测容量点，正文没有给出数值误差、置信区间或完整并发配置。
- 另有 **24,000 请求**的在线容量分析：95% 请求覆盖率约需 1 TiB，99% 约需 5 TiB。

因此，你原来的判断是正确的：**可以说作者报告接近实测，不能说已经证明跨系统准确。**

### Ferrule 的关键不匹配：现在不是 page-LRU

现有实现：

- `crates/ferrule-runtime/src/cache/radix.rs`：按完整 prefix entry 做 LRU。
- entry 包含 model state 和 KV snapshot。
- 容量 charge 是各 snapshot 的 page 数之和。
- 不同 snapshot 可以共享物理页。
- pin 住的 entry 不能淘汰。
- 页面还可能被活跃序列引用；淘汰 entry 不代表立即释放对应物理页。

所以：

> **KVSET 的理想 page-LRU 曲线，不能直接等价成 Ferrule 的 `prefix_cache_capacity_pages` 配置建议。**

首版应同时提供两个结果：

1. **理想 page-LRU 分析**：回答这个 workload 的复用工作集是什么。
2. **Ferrule 实际策略 replay**：回答当前 entry-LRU、共享和 pinning 下的表现是什么。

两者的差异本身就是有价值的结果。

### 三个必须先定清的语义

**1. Page identity 不能使用 `KvPageId`**

页号会复用，同一内容也可能因 COW 或重新计算使用不同页号。

建议采用带命名空间的内容身份，例如：

```text
page_key = hash(
    cache_namespace,
    parent_prefix_key,
    page_tokens,
    valid_token_count
)
```

其中 namespace 必须覆盖影响 KV 等价性的模型、adapter、位置及布局配置。

尤其不能只 hash 当前页的 tokens：**相同局部 tokens，前文不同，KV 通常也不同。**

**2. Page hit 不等于连续 prefix hit**

如果前面的 page miss、后面的 page hit，后者不能补足连续前缀复用。

在约定好的顺序 page-reference 模型中，要保住前 `k` 页，需要：

\[
C_r(k)=1+\max_{0\le j<k}d_{r,j}
\]

而不是把距离排序后挑最容易命中的 `k` 页。

同时要分开：

- `α`：每个请求保留多少比例的理论可复用前缀。
- `p`：多少比例的请求达到上述目标。

“保留 90% 前缀、覆盖 99% 请求”与“总 page hit rate 为 90%”是不同目标。

**3. Demand 与实际 cache outcome 要分开**

只记录当前缓存命中的内容，会把分析限制在当前容量内；缓存关闭时甚至没有分析输入。

至少需要三类事件：

```text
Demand：请求需要哪些逻辑 prefix pages
Publish：哪些 KV 已提交，何时可以复用
Outcome：当前缓存实际复用了多少
```

不能提前把尚未提交、取消或失败的请求当作可复用历史。

### 推荐实现

**第一步：离线 analyzer，不改 eviction。**

输出：

- page hit-rate curve；
- 连续 prefix token reuse curve；
- 给定 `α/p` 的容量需求；
- cold、无复用机会、目标不可达的请求数；
- 分析器自身时间和内存开销。

离线版本可以预分配 Fenwick 时间轴。在线版本不能无限按访问次数扩张数组，需要另行设计压缩或 order-statistics tree；**统计窗口有限，不代表历史身份状态也有限。**

**第二步：接 Ferrule trace。**

优先插入：

- `crates/ferrule-runtime/src/engine/driver/prefix.rs`：lookup 与需求。
- `crates/ferrule-runtime/src/engine/driver/admission.rs`：admission 发布结果。
- `crates/ferrule-runtime/src/engine/driver.rs`：committed prefill capture。
- `crates/ferrule-runtime/src/cache/page_manager.rs`：共享、COW、retirement。

第三步才考虑在线容量建议，暂时不自动改预算。

---

## 二、InplaceKVCache：核心不是 spill，而是不同消费方式的存储布局

### 四类 region 的准确含义

原文 §4 将 CPU/GPU 与 compute/transfer 交叉划分：

| Region | 用途 |
|---|---|
| CPU unpinned compute | 历史 KV 留在普通 DRAM，由 CPU 原地计算 attention |
| CPU pinned load | 历史 KV 留在锁页 DRAM，每次分块传给 GPU 计算 |
| GPU D-window offload | 暂存新 decode KV，攒批后首次放置到两个 host region 之一；窗口内也参与 attention |
| GPU ping-pong compute | 两组可复用缓冲，交替进行 H2D 和 GPU attention |

**这不是四级缓存，也不是把 GPU 份额一直放在显存里。** 历史 KV 的 GPU 计算份额主要仍放在 host，GPU 只保留流式工作区。

“写入时决定归属”也不意味着之后没有传输：pinned region 的内容仍会被反复送往 GPU；避免的是为调比例而反复重排历史存储。

### 对 Ferrule 而言，现有 preempt/restore 还不够

已有基础很好：

- `page_manager.rs` 管理逻辑页、refcount、COW。
- `crates/ferrule-runtime/src/engine/driver/sessions.rs` 有会话 preempt/restore。
- `crates/ferrule-backend/src/cuda/operators/kv/page_pool.rs` 有 host snapshot 和上传恢复。

但当前流程本质上是：

```text
暂停会话 → 保存 KV → 恢复时上传 → GPU attention
```

论文需要的是：

```text
活跃会话
  ├─ CPU 直接消费一部分 host KV
  └─ GPU 流式消费另一部分 host KV
       → 合并同一次 attention
```

另外，当前 CUDA pool 是固定容量 allocation。**释放 pool 内 slot 不等于释放实际 VRAM**，runtime hard credits 也不会因为 preempt 自动增加。

因此，不能只给 `preempt_session()` 增加自动触发，就宣称实现了论文方向。

### 必须增加的三个能力

#### 1. 物理 placement 与逻辑 page 分离

保留现有逻辑页机制，新增物理层信息：

- 权威数据所在位置；
- CPU/GPU 消费方式；
- 临时 GPU staging slot；
- 数据版本、read/transfer lease；
- 完成事件。

一个 host page 可以在不同 decode step 使用不同 staging slot；不能把 staging slot 当成新的逻辑 page。

资源记账也应拆开：

```text
logical pages
GPU resident slots / allocated bytes
host pageable bytes
host pinned bytes
transfer / attention scratch
```

并为 host 和 pinned memory 设置硬预算。

#### 2. 能返回 partial result 的 attention

分割沿 **sequence/token 维度**，不是按 head 分割。

CPU 与 GPU 必须返回各自的 partial output 和归一化信息，再正确合并。若分区返回归一化输出 `o_j` 与 log-sum-exp `z_j`：

\[
z=\operatorname{logsumexp}_j(z_j),\qquad
o=\sum_j e^{z_j-z}o_j
\]

**不能直接相加或平均两边输出。**

这意味着需要：

- CPU partial attention；
- GPU chunked attention；
- output + LSE 接口；
- query 的 CPU 输入、CPU partial 的返回；
- ping-pong copy/compute 事件流水；
- 尾块、空分区和 mask 处理。

首版建议只选一个受支持的标准 GQA 路径，不同时覆盖 MLA、recurrent state 和所有模型。

#### 3. Placement 必须纳入事务协议

继续遵守现有链路：

```text
reserve → execute → logical prepare
        → backend commit → logical publish
```

新 placement 未就绪，不能对 reader 可见；旧 slot 未收到相关完成确认，不能复用。

特别注意：**从 GPU 搬到 host 可能只是物理位置变化，不是逻辑 page 被释放。** 不能直接套用逻辑 retirement 的全部语义。

---

## 三、WriteScope 的控制器值得做，但应该最后做

论文平衡 CPU/GPU 路径：

\[
T_\text{CPU}=rAL+C_\text{CPU}
\]
\[
T_\text{GPU}=(1-r)BL+C_\text{GPU}
\]

选择使二者接近的 CPU 份额 `r`，再通过在线测量调整新 KV 的去向。

这里有个重要限制：**旧 KV 不重排，控制器只能调整新写入。**

如果已经有很长的历史，突然发生 CPU 或 PCIe 竞争，仅改变新增几百 tokens，无法迅速改变几十万 tokens 的历史分布。应分别记录：

- 理论最优份额；
- 实际历史份额；
- 下一批写入后可达到的份额。

不能把这个控制器当成能够瞬间重新分配负载的调度器。

论文消融也说明：**连续 DMA、避免重建、缓冲复用的收益，比单独动态调比例更基础。** 所以实现顺序应是先布局、再混合计算、最后控制器。

---

## 四、实验结论需要怎样理解

这篇论文的实验条件比“32 GB 显存支持 1M token”更具体：

- 1M 是 **batch 32 × sequence 32K 的 aggregate scale**，不是单请求百万上下文。
- A100 限制到 32 GB，以及 V100-32GB。
- 同一台主机，约 **502 GB RAM**。
- 长上下文时，host KV 可达到几十至上百 GB，锁页内存也可达到几十 GB。
- 1.4–2.5× 主要是对同一 runtime 内重实现的基线机制比较，不是四个上游系统的直接生产对照。

因此它证明了一个值得复现的设计方向，但不保证 Ferrule、消费级 CPU 或较小主机内存下也有相同收益。尤其 Ferrule 的 CPU 专家执行与 CPU attention 可能竞争内存带宽，必须一起测。

---

## 五、我建议的实施顺序与验收标准

| 阶段 | 实现内容 | 验收标准 |
|---|---|---|
| 1 | 离线 working-set analyzer | 与朴素 page-LRU 在所有测试容量上吻合；正确区分连续前缀、cold 和不可达目标 |
| 2 | Ferrule trace + entry-LRU replay | 同一事件流在线/离线一致；能量化理想模型与实际策略偏差 |
| 3 | Host KV 预算 + 纯 GPU streaming | 长上下文下 GPU 工作区有界；数值、取消、回滚、完成确认正确 |
| 4 | 固定比例 CPU/GPU attention | 与 reference attention 对齐；明确测出何时比纯 streaming 更快 |
| 5 | Roofline + online feedback | 在负载变化下优于固定比例，并报告历史 placement 带来的响应限制 |

**如果现在开始写代码，我会先做第 1 阶段，而不是修改 radix eviction 或 CUDA attention。** 它风险低，也能先回答后续硬件设计最需要的问题：这批 agent workload 的可复用工作集到底有多大。

两篇论文最终可以互补，但它们回答不同问题：

- **KVSET：为了保留跨请求复用，需要留下多少历史 KV？**
- **InplaceKVCache：当前请求必须消费的 KV，应该在哪里、怎样计算？**

缓存命中率高，不等于 decode 的 KV 带宽成本低；两者必须分别建模。

本轮没有修改源码，仅在 `target` 下保留了论文提取材料。子代理对 LRU 距离公式和 partial-softmax 合并做了独立数学检查，但这些不是 Ferrule 集成测试，也未复现论文的性能结果。

## User

你弄个文档，在 [@docs](file:///root/Ferrule/docs/) 下，然后git排除掉。

## Assistant

<thinking>
**Inspecting local Git exclusions**

**Checking docs directory**
</thinking>

我会把审阅结论、实现路线和验收标准整理成 `docs` 下的文档，并通过本地 `.git/info/exclude` 排除它，不改仓库共享的 `.gitignore`。

**Tool Call: 确认文档位置与本地排除配置**
Status: Completed

调查结果：

- **文档目录**：`Ferrule/docs` 已通过 `list_directory` 验证。
- **现有命名习惯**：文档主要使用全大写、下划线分隔，例如 `ARCHITECTURE.md`、`ARXIV_PAPER_PLAN.md`、`QWEN35.md`。
- **命名建议**：
  - 与现有习惯一致：`docs/KV_CACHE_WORKING_SET_AND_HYBRID_ATTENTION_REVIEW.md`
  - 用户指定的 `docs/kv-cache-working-set-and-hybrid-attention-review.md` 也可直接使用，但不完全符合现有风格。
- **适用规则**：
  - `/root/Ferrule/AGENTS.md` 存在并适用。
  - `/AGENTS.md`、`/root/AGENTS.md`、`docs/AGENTS.md` 均不存在。
  - 本次仅做只读检查，未修改文件。
- **目标文件状态**：
  - `docs/kv-cache-working-set-and-hybrid-attention-review.md` 不存在。
  - 因此不需要备用路径。
  - 该路径未被 Git tracked。
- **本地 exclude 路径**：`.git/info/exclude`，绝对路径为 `/root/Ferrule/.git/info/exclude`。
- **Exclude 检查**：
  - 现有 exclude 中没有目标文档规则。
  - 当前没有其他有效 ignore 规则匹配该文档。
  - `.git/info/exclude` 可写，因此**可以直接追加规则**；本次未追加任何内容。

## Assistant

<thinking>
**Planning git exclude update via terminal append**

**Outlining detailed exclude documentation plan**
</thinking>

仓库调查确认 `docs/` 存在，目标文档尚未被跟踪，且项目没有 `docs/AGENTS.md`。我现在写一份可作为设计备忘录的中文文档，并仅在当前仓库的 `.git/info/exclude` 添加该文件规则。

**Tool Call: docs/KV\_CACHE\_WORKING\_SET\_AND\_HYBRID\_ATTENTION\_REVIEW.md**
Status: Completed

Diff: Ferrule/docs/KV_CACHE_WORKING_SET_AND_HYBRID_ATTENTION_REVIEW.md
```
# KV Cache Working Set 与 CPU/GPU Hybrid Attention 评审

> 状态：设计评审与实现建议
>
> 本文基于 `target/kvset.pdf` 和 `target/inplacekvcache.pdf` 的本地审阅，以及对 Ferrule 当前 KV 生命周期、prefix cache 和 session preemption 实现的检查。
>
> 注意：`target/inplacekvcache.pdf` 的实际论文标题是 **Fractal KV-Cache Archives: Lossless Symbolic Storage with In-Place Retrieval for Long-Context LLM Inference**，并不是 WriteScope 论文。本文中关于 CPU/GPU 分区和 WriteScope 的内容依据本地已有的 `target/paper-review/writescope/2609.14507v1.pdf`：**Physically Partitioned KVCache Format for CPU–GPU Load Balancing in MoE Inference**。

## 1. 结论摘要

两篇论文解决的是不同层次的问题：

- **KVSET**：给定目标复用率，需要多少 KV cache 容量？它是 working-set 与容量规划分析，不是新的 eviction policy。
- **InplaceKVCache / WriteScope**：KV 已经需要参与 attention 时，哪些历史 KV 在 CPU 上原位计算，哪些通过 host-to-device 流式送到 GPU？它是存储布局与混合 attention 执行方案。

建议不要直接把二者合并为自动分层缓存。推荐按以下顺序实施：

1. 先做 KVSET 式离线 working-set analyzer；
2. 增加 Ferrule 的 KV/page 事件 trace，并对比理想 page-LRU 与现有 entry-LRU；
3. 再做有明确 host 预算的 GPU streaming；
4. 再实现固定 CPU/GPU 比例的 hybrid attention；
5. 最后才接入 roofline 与在线反馈控制。

第一阶段不应修改当前 eviction 行为，风险最低，也最容易验证收益。

## 2. 论文事实与适用边界

### 2.1 KVSET

本地文件 `target/kvset.pdf` 的标题是 **The KV Cache Working Set: Online Capacity Planning for LLM Inference Systems**，作者为 Luchang Li、Shuaishuai Wang、Zhao Ruan、Dongfang Li、Bozhao Gong（第 1 页）。该 PDF 没有可确认的 arXiv 编号或正式发表日期。

论文第 3 页描述的核心算法是：

- 对每个逻辑 page 记录最近一次访问位置；
- 用 Fenwick tree 统计此前访问过的、且比目标 page 更新的 distinct page 数；
- 得到零基 LRU stack distance `d`；
- 容量为 `C` 时，命中条件为 `d < C`；
- 每次访问后将 page 移到 LRU 顶部。

这允许一次扫描得到多个容量点的理想 LRU 命中曲线。论文没有给出完整伪码，也没有明确给出 page key 的构造、并发事件顺序、在线历史压缩或统计窗口语义。因此，Fenwick 的具体复杂度和内存控制属于实现推导，不应当误述为论文已证明的结论。

论文报告了两类实验：

- 约 10,000 个内部 coding-agent 请求，与 Mooncake/SGLang 的部分物理容量部署进行比较；
- 约 24,000 个请求的容量分析，报告 95% 请求覆盖率约需 1 TiB、99% 约需 5 TiB，而 99.9% 没有明显收敛。

原文没有给出完整部署规模、并发配置、误差条、置信区间和精确误差数值。因此，“与实测接近”只能作为作者在该环境下的结论，不能作为跨系统保证。

### 2.2 InplaceKVCache / WriteScope

本地 WriteScope 论文为 `target/paper-review/writescope/2609.14507v1.pdf`，标题是 **Physically Partitioned KVCache Format for CPU–GPU Load Balancing in MoE Inference**，arXiv 页面为 <https://arxiv.org/abs/2609.14507v1>。

论文把 device affinity 与 access pattern 交叉形成四类区域：

| Region | 含义 |
| --- | --- |
| CPU unpinned compute | 普通 DRAM 中由 CPU attention 原位消费的历史 KV |
| CPU pinned load | 锁页 DRAM 中作为连续 H2D 源、由 GPU attention 消费的历史 KV |
| GPU D-window offload | 新 decode KV 的 GPU 暂存窗口；攒批后首次放置到两个 host region 之一，窗口内也参与 GPU attention |
| GPU ping-pong compute | 两组可复用 GPU 工作缓冲，交替进行 H2D 与 GPU attention |

这不是四级缓存，也不是四份长期 KV。历史 KV 的权威副本长期位于 host；GPU 侧主要保留 D-window 和 streaming workspace。

论文沿 sequence/token 维度分配 CPU 与 GPU 的工作量，而不是按 head 分配。CPU 与 GPU 分别计算 attention 分区，返回 partial output 与 normalizer，再通过 log-sum-exp 等价方式合并。合并不能直接平均两个 softmax 输出。

论文的 roofline 模型近似为：

```text
T_cpu = r * A * L + C_cpu
T_gpu = (1-r) * B * L + C_gpu
T     = max(T_cpu, T_gpu) + T_merge
```

其中 `r` 是 CPU 份额。在线反馈只改变新批次的 placement，历史批次不会立即重新迁移。因此长上下文下，控制器对突发负载变化的响应受限。

实验中的“1M-token aggregate scale”是 batch 32 × sequence 32K，而不是单个请求的一百万 token。实验还依赖大量 host DRAM 和锁页内存；报告的收益不能直接推广为所有 GPU、CPU、NUMA 布局和 runtime。

## 3. Ferrule 当前实现与论文模型的差异

Ferrule 已经具备较完整的逻辑分页 KV 生命周期：

```text
reserve -> execute -> prepare logical commit
        -> backend commit -> logical publish
        -> retirement -> backend release ACK -> reuse
```

相关位置包括：

- `crates/ferrule-runtime/src/cache/page_manager.rs`
- `crates/ferrule-runtime/src/cache/radix.rs`
- `crates/ferrule-runtime/src/engine/driver/kv_lifecycle.rs`
- `crates/ferrule-runtime/src/engine/driver/prefix.rs`
- `crates/ferrule-runtime/src/engine/driver/sessions.rs`
- `crates/ferrule-backend/src/cuda/operators/kv/page_pool.rs`

当前 prefix cache 的重要语义是：

- `RadixPrefixCache<P>` 按完整 prefix entry 做 LRU；
- payload 同时包含 model state 与 KV snapshot；
- prefix entry 可以 pin，pin 中的 entry 不能淘汰；
- snapshot 之间可以共享物理页；
- entry charge 是 snapshot page 数，不等同于实际可释放的物理 page 数；
- `KvPageId` 是运行时页标识，可能复用，不能直接作为跨请求的内容身份。

已有 session preempt/restore 也不能直接视为 WriteScope：

- preempt 是暂停会话并保存 snapshot；
- restore 时再将数据上传到 GPU；
- 当前 CUDA pool 是固定容量 allocation，释放 slot 不等于减少进程已分配的 VRAM；
- runtime hard credit 不会因为普通 preempt 自动变成可用 credit；
- 当前没有让活跃 attention 直接消费 host KV 的 CPU/GPU 混合执行路径。

因此，论文模型至少需要新增“物理 placement 层”，而不是把四类 region 直接映射成四种逻辑 page。

## 4. KVSET 实现建议

### 4.1 明确定义 logical page identity

不能使用 `KvPageId`。建议使用包含命名空间和完整前缀上下文的内容 key，例如：

```text
page_key = hash(
    cache_namespace,
    parent_prefix_key,
    page_tokens,
    valid_token_count
)
```

`cache_namespace` 至少需要覆盖会影响 KV 等价性的 model、adapter、位置/布局和 backend 配置。相同的局部 token 序列，在不同前缀下通常不产生相同 KV。

### 4.2 明确三种不同的 hit rate

文档、trace 和指标中不要使用一个含糊的 `hit_rate`：

- `page_reference_hit_rate`：单个 page reference 的 LRU 命中率；
- `contiguous_prefix_token_hit_rate`：连续前缀中实际可复用的 token 比例；
- `request_target_coverage`：达到目标前缀复用比例的请求比例。

如果请求 `r` 要保留连续前 `k` 个 page，所需容量是：

```text
C_r(k) = 1 + max(distance[r][0..k-1])
```

前面出现 cold page 时，该连续目标不可达。不能把 page distance 排序后取最小的若干个，因为后面的命中不能绕过前面的 prefix miss。

还应分开两个参数：

- `alpha`：每个请求需要保留的理论可复用 prefix 比例；
- `p`：需要达到该目标的请求覆盖率。

### 4.3 在线与离线共用 analyzer 状态机

建议采用同一事件模型：

```text
input event
  -> identity normalization
  -> page-LRU state update
  -> capacity curve update
  -> request result update
```

离线输入来自 trace，在线输入来自 runtime 事件。至少区分：

```text
Demand       请求需要哪些逻辑 prefix pages
Publish      哪些 KV 已提交并可供后续请求复用
Outcome      当前实际配置命中了哪些内容
```

失败、取消和未 commit 的请求不能提前产生可复用历史。缓存容量为 0 时也应能记录 demand，否则无法分析“打开 prefix cache 后的预期效果”。

### 4.4 推荐首版输出

首版 analyzer 应输出：

- 多容量点的理想 page-LRU 曲线；
- 连续 prefix token reuse 曲线；
- 给定 `alpha` 和 `p` 的容量需求；
- cold、无复用机会和目标不可达请求数；
- analyzer 自身的处理耗时与内存占用；
- 理想 page-LRU 与 Ferrule 实际 entry-LRU 的差异。

首版只做离线 replay 和观测，不自动驱动 eviction、扩容或 placement。

## 5. Hybrid attention 实现建议

### 5.1 物理 placement 元数据

逻辑 page 管理仍由 `page_manager.rs` 负责。新增的物理层至少需要区分：

- logical page identity；
- 数据权威位置（GPU、CPU unpinned 或 CPU pinned）；
- CPU/GPU 消费路径；
- 临时 GPU staging slot；
- read/transfer lease；
- 数据版本和完成事件。

一个 host page 可以被不同 decode step 使用不同 GPU staging slot；staging slot 不是新的 logical page。

资源记账必须分开：

```text
logical KV pages
GPU allocated bytes / resident slots
host unpinned bytes
host pinned bytes
transfer and attention scratch
```

host 与 pinned memory 都需要预算、admission 和背压，不能默认 host RAM 无限。

### 5.2 正确的 partial attention 合并

对每个 query/head 的一个分区 `P_j`，应计算：

```text
m_j = max(logits in P_j)
L_j = sum(exp(logit - m_j))
U_j = sum(exp(logit - m_j) * value)
```

全局合并为：

```text
m = max_j(m_j)
L = sum_j exp(m_j - m) * L_j
U = sum_j exp(m_j - m) * U_j
output = U / L
```

或者让分区返回 output 与 log-sum-exp，再用 log-sum-exp 加权合并。

实现必须覆盖：

- 空分区和全 mask 行；
- causal/sliding mask；
- RoPE 和长位置；
- GQA 的 query/KV head 映射；
- FP32 累积与误差容忍度；
- D-window 尾块和有效长度；
- CPU query 输入、partial output 和 normalizer 的传输依赖。

首个模型范围应限制在一个已有且受支持的标准 GQA 路径，不要一开始同时支持 MLA、recurrent state 和所有模型。

### 5.3 Placement 事务

建议复用既有事务边界：

```text
reserve
  -> reserve host block / GPU staging / budget
  -> execute write and transfer
  -> logical prepare
  -> backend commit after completion dependencies
  -> logical publish
```

未就绪的数据不能被 reader 看到；旧 slot 只有在所有 reader、copy 和 attention 完成并获得 backend release ACK 后才能复用。placement 改变不是逻辑 page 删除，不能简单套用 retirement 的语义。

live transaction、pin、共享前缀、COW 和 quarantine 必须继续受到保护。首版可限制为独占页，之后再处理共享前缀跨 tier placement。

## 6. 分阶段路线

### 阶段 1：离线 KVSET analyzer

新增独立 analyzer 与 synthetic trace 测试，不改变 prefix eviction。

必须通过：

- 与朴素 page-LRU 在所有容量点一致；
- prefix hole 不会错误计算为连续命中；
- cold/no-reuse 的统计分母明确；
- `alpha` 和 `p` 的单调性正确；
- Fenwick 扩容边界正确；
- 不同 model/adapter/前缀不会错误共享 identity。

### 阶段 2：Ferrule trace 与实际策略 replay

在真实成功边界记录：

- prefix lookup；
- admission publish；
- page demand；
- reserve/COW；
- backend release ACK；
- logical retirement；
- 实际复用 token/page 数；
- pin 和共享导致的不可释放页。

离线报告同时显示理想 page-LRU 和 Ferrule entry-LRU，避免用理想曲线承诺实际容量效果。

### 阶段 3：受预算约束的 GPU streaming

先实现 host KV 的预算、连续 transfer block、有限 GPU workspace 和生命周期 ACK。不引入 CPU attention，也不宣称已经实现 WriteScope。

### 阶段 4：固定比例 hybrid attention

先不做在线反馈，使用固定 CPU/GPU split，证明：

- 数值结果与 reference attention 一致；
- GPU VRAM 工作区有界；
- CPU attention、H2D、GPU attention 和 merge 的时间可观测；
- 在目标硬件上确实优于纯 GPU streaming 或现有 preempt/restore。

### 阶段 5：roofline 与在线反馈

控制器需要同时观察 CPU component latency、GPU/PCIe latency、host memory pressure 和并发竞争。控制器应记录理论最优份额、当前历史份额和下一批实际可达份额，不假设历史数据可以瞬时重排。

## 7. 验收与实验矩阵

### 正确性

- FP32/FP64 reference 对比全 GPU、全 CPU、混合执行；
- CPU 份额为 0 和 100%；
- 空分区、尾块、D-window 跨界；
- GQA、RoPE、causal/sliding mask；
- extreme logits 与全 mask；
- KV transport bitwise 校验；
- attention 输出的 absolute/relative/RMS error；
- transfer 失败、host 分配失败、取消、rollback、commit 失败；
- 未 ACK 的 slot 不得复用；
- 未 publish 的 KV 不得命中。

### 性能

分别记录：

- 实际 VRAM allocation、GPU slot occupancy；
- host RSS、pinned bytes、scratch 峰值；
- CPU attention、H2D、D2H、GPU attention、merge、pin allocation；
- CPU DRAM 和 PCIe 带宽；
- TTFT、稳态 TPOT、P50/P95/P99；
- 单用户与并发用户；
- 不同 page/block/D-window 大小；
- pinned/unpinned 和连续/分散 DMA；
- fixed split 与 adaptive split。

对照组至少包括：

1. 当前 GPU-resident 路径；
2. 当前 session preempt/restore；
3. pure GPU streaming；
4. CPU-only historical attention；
5. fixed CPU/GPU split；
6. online feedback split。

## 8. 最终建议

最先实现 KVSET analyzer，而不是修改 `radix.rs` 的淘汰逻辑或直接改 CUDA attention。原因是它可以在不改变线上行为的前提下回答三个关键问题：

1. workload 中真正可复用的 prefix working set 有多大；
2. Ferrule 当前 entry-LRU 与理想 page-LRU 的差异有多大；
3. 值不值得为 host placement 和 hybrid attention 增加复杂度。

如果 analyzer 证明长上下文 workload 的工作集明显超过 GPU resident capacity，再进入 streaming 和 hybrid attention。两篇论文应被视为互补的两个决策层：

```text
KVSET：需要保留多少历史 KV？
WriteScope：需要消费的 KV 应该在哪里、怎样计算？
```

命中率高不等于 attention 带宽成本低；容量规划、物理 placement 和 attention 执行必须分别测量、分别验收。

```


## Assistant

<thinking>
**Appending the correction note**
</thinking>

**Tool Call: printf '\n/docs/KV_CACHE_WORKING_SET_AND_HYBRID_ATTENTION_REVIEW.md\n' >> .git/info/exclude**
Status: Completed

Command executed successfully.

## Assistant

<thinking>
**Validating git exclude patterns with status checks**
</thinking>

文档已写入，并追加到本地 Git 排除文件。现在做一次轻量校验：确认路径被 `.git/info/exclude` 命中、文档未进入待跟踪列表，并检查 Markdown 文件是否为空或截断。

**Tool Call: git check-ignore -v docs/KV_CACHE_WORKING_SET_AND_HYBRID_ATTENTION_REVIEW.md && test -s docs/KV_CACHE_WORKING_SET_AND_HYBRID_ATTENTION_REVIEW.md && git --no-optional-locks status --short -- docs/KV_CACHE_WORKING_SET_AND_HYBRID_ATTENTION_REVIEW.md .git/info/exclude**
Status: Completed

```
.git/info/exclude:8:/docs/KV_CACHE_WORKING_SET_AND_HYBRID_ATTENTION_REVIEW.md	docs/KV_CACHE_WORKING_SET_AND_HYBRID_ATTENTION_REVIEW.md

.git/info/exclude:8:/docs/KV_CACHE_WORKING_SET_AND_HYBRID_ATTENTION_REVIEW.md	docs/KV_CACHE_WORKING_SET_AND_HYBRID_ATTENTION_REVIEW.md
```

## Assistant

已整理好文档：

[KV Cache Working Set 与 Hybrid Attention 评审](docs/KV_CACHE_WORKING_SET_AND_HYBRID_ATTENTION_REVIEW.md)

包含论文结论与限制、Ferrule 当前实现差异、分阶段实现方案，以及正确性和性能验收标准。

已通过本地 `.git/info/exclude` 排除该文件，**未修改共享 `.gitignore`**。`git check-ignore` 已确认生效，文档不会出现在普通 Git 待提交列表中。

## User

我现在丢失了所有的代码和上下文，只有这个session会话还在，我的代码现在在/home/admin/luhuanbing.lhb/Ferrule/ 下，你看看总结汇总目前到现在的所有改动痕迹落实到docs下面的一个文档，写详细一些，我们的代码丢失了两天内所有的。

## Assistant

<thinking>
**Locating project context**
</thinking>

我会先做只读核查，把会话中能确认的操作记录，与新路径下现存的代码、Git 历史和研究材料交叉对照，再在 `docs` 下写一份详细的恢复记录。文档会明确区分“已确认改动”“当时观察到的代码状态”和“尚未实施的方案”，避免把设计建议误当成丢失的代码；核查期间不会执行 reset、clean、checkout 或覆盖现有文件。

**Tool Call: pwd; git --no-pager rev-parse --show-toplevel; if test -f AGENTS.md; then cat AGENTS.md; fi; if test -d docs; then printf '\nDOCS_DIRECTORY_PRESENT\n'; fi**
Status: Failed

`cd` directory "/home/admin/luhuanbing.lhb/Ferrule" was not in any root directory in the project.

**Tool Call: List the `Ferrule` directory's contents**
Status: Failed

Path not found: Ferrule

## Assistant

<thinking>
**Waiting for project root**

**Awaiting project root**
</thinking>

可以，我会把它整理成一份**用于代码恢复的详细记录**，而不只是论文摘要。但目前有一个访问阻碍，需要你先处理：

- 访问 `/home/admin/luhuanbing.lhb/Ferrule/` 时，工具返回：**该目录不属于当前编辑器项目根目录**。
- 原项目根目录 `Ferrule` 也已无法访问，返回 `Path not found`。

**请在 Zed 中打开 `/home/admin/luhuanbing.lhb/Ferrule/`，或将它添加为当前项目的根目录，然后在本会话回复“已打开”。** 不需要重新讲背景，这个会话中的信息还在。目前我没有写入或修改新仓库。

## 先保全本会话能确认的恢复线索

必须先区分：**本会话没有实施 KVSET 或 WriteScope 的产品代码。** 我们做的是代码调查、论文审阅，以及创建一份本地忽略的设计文档。调查时发现的既有实现，不能认定都是本会话或最近两天新增的。

### 1. 本会话明确执行过的改动

| 项目 | 已确认操作 |
|---|---|
| 设计文档 | 创建了 `docs/KV_CACHE_WORKING_SET_AND_HYBRID_ATTENTION_REVIEW.md` |
| Git 本地排除 | 向 `.git/info/exclude` 追加了 `/docs/KV_CACHE_WORKING_SET_AND_HYBRID_ATTENTION_REVIEW.md` |
| 排除验证 | `git check-ignore -v` 确认该文件由上述规则忽略 |
| 论文研究材料 | 子代理报告在 `target/` 下生成了提取文本、辅助脚本和图示；部分材料只有报告，没有完整创建命令记录 |
| 产品源码 | **本会话未修改** |
| Git commit / branch | **本会话未创建** |
| Cargo / GPU 验证 | **本会话未运行** |

**原设计文档的完整内容仍在这段会话的写文件记录中，可以重建。** 它曾被本地 Git 忽略，因此单靠 checkout 或普通 Git 历史不一定能找回。

### 2. 当时调查到的重要代码状态

这些会作为恢复文档中的“旧工作树观察记录”，并与新仓库逐项核对：

- **分页 KV 生命周期**
  - `crates/ferrule-runtime/src/cache/page_manager.rs`
  - reservation、共享尾页 COW、prepare/publish commit、rollback、retirement。
  - 后端释放确认之前，页不得复用。
- **Prefix cache**
  - `crates/ferrule-runtime/src/cache/radix.rs`
  - 按完整 prefix entry 做 LRU，而非独立物理 page-LRU。
  - 存在 pin、namespace、snapshot 共享和 entry charge。
- **Driver 接入**
  - `crates/ferrule-runtime/src/engine/driver/prefix.rs`
  - `crates/ferrule-runtime/src/engine/driver/admission.rs`
  - `crates/ferrule-runtime/src/engine/driver/kv_lifecycle.rs`
  - `crates/ferrule-runtime/src/engine/driver/resident.rs`
  - `crates/ferrule-runtime/src/engine/driver.rs`
- **会话 preempt / restore**
  - `crates/ferrule-runtime/src/engine/driver/sessions.rs`
  - 已有 CPU snapshot 与 GPU restore，但不等于自动分层 KV admission。
- **物理 KV 存储和事务**
  - `crates/ferrule-backend/src/cpu/kv.rs`
  - `crates/ferrule-backend/src/cuda/operators/kv/page_pool.rs`
  - `crates/ferrule-backend/src/cuda/operators/kv/transaction.rs`
  - `crates/ferrule-backend/src/cuda/operators/kv/transaction/plan.rs`
  - `crates/ferrule-backend/src/cuda/operators/kv/transaction/lifecycle.rs`
- **关键测试线索**
  - `crates/ferrule-backend/tests/cuda_kv_metadata.rs`
  - `crates/ferrule-backend/tests/cuda_kv_transactions.rs`
  - page manager、radix、driver 中的 prefix、COW、preempt/restore 测试。

会话里还保留了这些实现的函数名、调用链、当时行号、限制和测试命令，能够整理成逐项恢复核对表。**但没有保存这些文件的完整源码，不能保证仅凭本会话恢复全部实现。**

### 3. 尚未实施的设计

下面这些只讨论过，不能写成已完成：

1. KVSET 离线 working-set analyzer。
2. logical page content identity 与统一 demand/publish/outcome trace。
3. 理想 page-LRU 与 Ferrule entry-LRU 的 replay 对照。
4. 有 host/pinned memory 预算的 GPU streaming。
5. CPU/GPU partial attention 与 LSE 合并。
6. WriteScope roofline 和在线 placement feedback。

## 打开仓库后会落地的文档

计划创建：

`docs/SESSION_RECOVERY_AND_CHANGE_EVIDENCE.md`

内容会包括：

- 本会话时间顺序与明确写入记录；
- 原设计文档的完整可恢复内容；
- 当时工作树的模块、类型、函数、调用链和不变量；
- 新仓库中每项内容的“仍存在／部分缺失／未找到／无法确认”状态；
- Git 最近历史、reflog 和工作树差异中的恢复线索；
- 已知测试名称、命令及验证缺口；
- 分优先级的恢复清单；
- 哪些结论是直接证据，哪些只是当时的调查报告或设计建议。

**这次不会默认忽略恢复文档**，避免再次变成只有本地存在、Git 中没有的资料；也不会自动提交。

在核查完成前，建议不要运行 `git gc`、`git prune`、`git clean`、`reset --hard`，也先不要清理 `target/`，这些位置可能仍保留恢复线索。


