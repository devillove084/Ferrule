# Ferrule GB10 / DGX Spark roadmap

> Ferrule 在单 NVIDIA GB10 / DGX Spark 上的当前路线图。
>
> Updated: 2026-07-29.

## 0. 当前结论

当前结论是 **NO-GO**。第一阶段 I/O/调度主体已经实现，但尚未完成真实硬件验收，因此第一阶段不得标记为完成。

当前唯一 blocker 是完成下面这条连续验证链：

```text
验证 CompletionHub wake patch
  → real GB10 final4 n8
  → production c1 / c2 / c4 overlap + counters
```

其中任一步未通过，都不能宣称第一阶段完成，也不能从 fake coverage、历史 CUDA 构建或单独的 `n1` 推导真实 overlap、无泄漏或发布就绪。

## 1. 冻结范围

| 项目 | 当前范围 |
|---|---|
| workload | inference only |
| platform | one NVIDIA GB10 / DGX Spark |
| CUDA target | `sm_121a` |
| model | exact `models/DeepSeek-V4-Flash-DSpark` checkpoint |
| headline metric | externally committed output tokens per second |

GB10 的结果不能直接外推到其他 GPU。发布门槛仍要求冻结 workload 下 warm externally committed throughput 的 95% 置信下界达到 `16 tok/s`，并具备可复现的正确性、资源、计数器和 profiler 证据。

## 2. 第一阶段 I/O/调度状态

### 2.1 已实现

以下是当前代码中已经完成的实现，不等同于最新 patch 已通过验收：

- typed `LoadKey`、`OperationId`、`ContinuationId` 和 generation；
- runtime global registry 与 single-flight；
- 双向 waiter/target wake；
- 完整 residency 数据路径：`read → pinned → upload → CUDA event → publish`；
- 13 类 hard credits；
- cancel、stale、failure、shutdown 的 fake coverage；
- multiple packed transactions；
- 旧 global guard 已删除；
- 公平队列和 critical-path ledger。

旧的“依赖 global guard 串行化 continuation”以及“只能存在一个 proposal-enabled packed transaction”的描述已经失效，不再作为当前架构或限制。

### 2.2 最近通过的基线

| 检查 | 结果 | 证据边界 |
|---|---|---|
| common lib | PASS | 最近通过 68 tests |
| model lib | PASS | 最近通过 250 tests |
| runtime lib | PASS | 最近通过 318 tests |
| last-demand wake targeted UT | PASS | `physical_selected_lease_releases_only_after_last_logical_demand_detaches` 已通过 |
| `just build-cuda sm_121a` | UNVERIFIED | unified `cuda` feature 改动后的构建尚未完成 |
| real GB10 `n1` | PASS | 曾完成于 `12.11 s`；只作为单请求历史基线 |

上述结果不能覆盖最新 `CompletionHub` wake patch。尤其不能把三阶段 residency 改动前的 CUDA 构建记录写成当前构建已通过。

### 2.3 real GB10 `n8` 进展

真实 GB10 `n8` 依次暴露并修过：

1. cancel stage race；
2. prepared eviction binding；
3. same-expert duplicate slot；
4. prepare-time adapter eviction。

最新 `final4 n8` 在 `91.33 s` timeout，`stderr` 为空。当前定位是：**late selected lease release 未唤醒 owner**。

针对该问题已经写入：

- `CompletionHub` 注入；
- last-demand notify patch；
- 对应的定向 unit test。

定向 test 已通过，但 unified `cuda` feature 改动后的完整 CUDA 构建、真实 `n8` 和后续 overlap 验收仍未通过。因此不得写成 CUDA build pass、`n8` pass，或据此关闭第一阶段。

### 2.4 唯一阻塞验证链

#### A. 验证 wake patch

必须先完成当前源码的编译和定向 UT，确认：

- late selected lease release 的 last-demand 路径会通过 `CompletionHub` 唤醒正确 owner；
- cancel、stale generation、failure 和重复通知不会产生错误 wake 或重复释放；
- 相关 ownership 与 13 类 hard credits 最终可回收。

#### B. 通过 real GB10 `final4 n8`

wake patch 验证后重新运行 `final4 n8`。验收要求至少包括：

- 不再 timeout；
- 无 owner/waiter 残留；
- 无 stale binding、重复 slot 或 credit 泄漏；
- transaction、I/O stage 与 externally committed token 可以由计数器对账。

#### C. 通过 production `c1` / `c2` / `c4` overlap 与 counters

`n8` 通过后，在 production path 依次验证 `c1`、`c2`、`c4`。必须用 timeline/counter 证据证明：某个 transaction 等待真实 read/upload/completion 时，其他 ready work 确实前进。仅有并发请求成功或 wall time 改善不算 overlap 证据。

完成并保留这三步的原始日志、计数器和 profiler artifact 后，才能评估关闭第一阶段。

## 3. 第一阶段所需证据

### 3.1 Identity、wake 与生命周期

- `LoadKey`、`OperationId`、`ContinuationId`、generation 的创建、关联和终止原因；
- registry 与 single-flight 的 join/publish/cleanup；
- waiter/target 双向 wake、last-demand notify、owner wake 及 stale/duplicate rejection；
- cancel、failure、shutdown 后的残留项必须可审计。

### 3.2 Residency 与 hard credits

- `read → pinned → upload → CUDA event → publish` 各阶段的进入、完成、失败和耗时；
- 13 类 hard credits 的 current、high-water、wait、acquire、release 和最终 residual；
- prepared eviction、adapter eviction、same-expert dedup 与 selected lease release 的对应关系；
- timeout、cancel、failure 和 shutdown 路径上的资源归还。

### 3.3 Fairness、critical path 与 overlap

- 公平队列的等待时间、调度决定和 starvation 证据；
- critical-path ledger 中 uncovered read/upload/wake/resume 时间；
- blocked transaction 与同时前进的 ready work 的 timeline 关联；
- `n8`、`c1`、`c2`、`c4` 的 transaction、committed token 和资源计数对账。

## 4. 验收清单

### 已实现

- [x] Typed `LoadKey` / `OperationId` / `ContinuationId` / generation。
- [x] Runtime global registry / single-flight。
- [x] 双向 waiter/target wake。
- [x] `read → pinned → upload → CUDA event → publish`。
- [x] 13 类 hard credits。
- [x] Cancel/stale/failure/shutdown fake coverage。
- [x] Multiple packed transactions，旧 global guard 已删除。
- [x] 公平队列和 critical-path ledger。

### 已有但仅作历史基线

- [x] Common 68、model 250、runtime 318 lib tests 最近通过。
- [x] `just build-cuda sm_121a` 在三阶段 residency 改动前通过，仅作历史基线。
- [x] Last-demand `CompletionHub` wake 定向 UT 已通过。
- [x] Real GB10 `n1` 曾在 `12.11 s` 完成。

### 尚未验证，第一阶段保持 open

- [ ] Unified `cuda` feature 下当前源码完成 CUDA build/test 验证。
- [ ] Real GB10 `final4 n8` 通过。
- [ ] Production `c1` / `c2` / `c4` overlap 与 counters 通过并完成对账。

## 5. 后续方向

第一阶段关闭后再推进以下方向，不将其混入当前 blocker：

- 为多 GPU 建立独立 profile、资源模型和验收证据；
- 继续演进 CUTLASS provider，并分别验证其正确性、能力边界与性能。

这些方向不改变当前优先级：先完成 wake patch → `n8` → `c1`/`c2`/`c4` overlap/counters 的连续验证链。
