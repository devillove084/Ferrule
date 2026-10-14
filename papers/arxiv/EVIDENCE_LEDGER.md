# 初稿 claim–evidence–gap 台账

本台账非编译输入/公开 artifact；保留 2026-10-02/08/09 历史，2026-10-14 同步四 RQ。本次仅 Markdown，无新 GPU/数值/构建验收，历史 PASS 不借用。

## 来源与可信度

- 历史讨论：`docs/temp_ferrule_refactor.md:13558–13858,14568–14616`。
- 当前论文目标：[README](README.md)；支持与 source-pinned 验收：[CURRENT_ACCEPTANCE](../../docs/CURRENT_ACCEPTANCE.md)。
- 评估合同/历史摘要：`docs/PERFORMANCE_EVALUATION.md`；本次不重跑或重新验收。
- 实现事实：下表源码与架构文档的静态核对，**不是**对所有路径的证明或运行验收。
- 2026-10-02 历史调查时 Git HEAD 为 `8a2fef862c7205373de6ee004f4d1ec46799aa6d`，工作区存在大量修改及未跟踪文件。这个 HEAD **不能代表**所读工作树或归档实验源码，不能作为单独的实验 revision 引用。
- 源码路径从仓库根定位，Markdown 链接从本目录解析；行号仅历史导航，发表需内容哈希。绝对 artifact pins 保留，不全局替换旧路径。

## 已确定的写作信息

- 标题：**Ferrule: Storage-Backed Inference Beyond Memory Capacity**。
- 作者：**Lu Huanbing**；单位：**Independent Researcher**；联系邮箱：**luhuanbing084@gmail.com**。
- ORCID、资助信息未提供，不推测，也不把未知填写为“无”。
- 核心目标：**storage-backed inference when the model exceeds deduplicated usable physical residency capacity**；controlled-budget 与 physical-capacity 分报，后者未验证。事务管理服务物化/安全复用，非数值 exactness、持久化事务或性能保证。
- 容量按物理 pool/共享页去重，不预设 GPU+host 相加；独立副本各计实际占用。
- 全模态、统一内存、全异步为设计范围，逐 profile 验收；不称全链路无阻塞或 measured overlap。
- 标题/作者/正文主线与 checklist 已同步；写作同步非能力验收。

## 首要 claim：超过去重物理驻留容量的 storage-backed 推理

| 目标 claim / 待核对项 | 当前证据状态 | 必须补齐 / 禁止扩大 |
|---|---|---|
| 模型超过去重的可用物理驻留内存容量时，利用 backing storage 完成推理 | **尚未建立完整证据** | 固定模型、模态、表示、算术合同、设备、物理 pool、拓扑及 snapshot；报告去重规则、施加预算、逐池高水位、存储访问及对应模态的端到端完成和数值/质量验收；prefill 与多步 decode 是当前语言解码实例 |
| 完整紧凑权重不能全驻留，硬盘对目标配置确有必要 | **完整磁盘必要性证据未确认** | 检查量化/压缩权重及必要 scales 等数据能否全驻留可用物理内存；仅执行展开字节超限不能证明磁盘必要性 |
| 事务性质资源管理支撑有限容量内的按需运行 | 下表仅提供限定的机制证据 | 还需同一容量配置下的集成路径、资源守恒与安全消融；协议测试不替代容量或数值验收 |
| 多模态与组合模态路径 | 设计范围与待落实事项；**逐模态支持及验收待核实** | 按模态、设备、profile 和 snapshot 建立实现与证据矩阵，不从 CUDA 语言解码结果推导已验证全模态 |
| 统一/共享内存的物理驻留容量与账本映射 | 设计已考虑；**具体配置的实现映射和容量证据待核实** | 核对物理 pool、共享页、独立副本、迁移与预算，去除重复计数；统一地址/API 不等于单一共享物理池 |
| 全异步路径、阻塞边界与实际 overlap | 设计范围与待落实事项；**端到端边界及重叠证据待核实** | 分别验证提交/完成语义、阻塞点、时间线及设备活动；不声称全链路无阻塞，不用 custody 安全性替代 overlap 验收 |

E0 仍为首要容量实验；当前按 RQ1 容量/质量、RQ2 真实 I/O、RQ3 同合同 safe-blocking/async、RQ4 集成故障/守恒收敛。E2 为 conditional secondary，vLLM 全胜/大矩阵非无条件 gate。全驻留 EP8、host-only、host 可容纳紧凑权重的 offload、logical read/cache miss 均不证明 physical-capacity 或 disk necessity。

### 必须区分的容量口径

1. **磁盘 checkpoint 文件大小：** 指定编码与文件集合的逻辑字节、物理占用；不能直接当作 RAM/VRAM 需求。
2. **紧凑权重全驻留需求：** 完整量化/压缩表示及其必要辅助数据能否放入可用物理内存；必须独立于执行展开字节检查。
3. **运行表示与全驻留需求：**列 decode/dequant、packing/padding、共享/副本；紧凑表示与执行展开分别计量。仅展开超限而紧凑权重可驻留时，内存内展开仍可能可行，不证明磁盘必要性。
4. **去重可用物理驻留容量：**物理 pool/共享页只计一次，独立副本各计；统一地址/managed-memory API 不证明单一共享池，需拓扑/迁移/驻留核对。离散 GPU+host 只是实例。
5. **实际总工作集：** 驻留权重、KV、activations、workspace、staging、在途/retained buffers、metadata、固定分配及相关 OS page cache。账本覆盖域不自动等于部署总内存，避免共享页重复计数，并明确 cache/swap 条件。

比较注明表示/物理去重规则，合计容量不代替各池放置约束；人为预算与物理不足分报，KV/并发超限不等于权重超容量。

## 设计范围与证据边界

- **多模态：**逐模态/设备/backend/profile 建立执行、质量与容量证据，文本结果不外推。
- **统一/共享内存：**按物理驻留/别名去重，非逻辑地址、映射长度或可见设备相加。
- **全异步：**提交/完成 API 非 overlap/无阻塞；需真实时钟、依赖及设备活动证据。
- **事务 custody：**区分 logical terminal、provider completion、physical quiescence/release；保护仍被引用容量，非性能、liveness 或 overlap 保证。
- **实现边界：**CUDA 解码/KV/backend 证据仍限其模态/设备/profile/snapshot/binary，不外推。

## 支撑目标的核心机制

| 可使用的限定表述 | 本地证据 | 仍需补齐 / 禁止扩大 |
|---|---|---|
| Rust owner 协调逻辑状态与资源生命周期；backend 负责物理操作 | [职责与资源合同](../../docs/ARCHITECTURE.md#1-system-boundary) | 不等于多 GPU 中只有一个 physical owner；需要冻结配置对应的路径图 |
| General registry 按完整 exact key 合并兼容的物化请求 | `crates/ferrule-common/src/io_protocol.rs:340–354`; `crates/ferrule-runtime/src/io/registry.rs:2250–2330` | 不可自动推广至 dedicated Qwen expert 或 live HostKV 全部路径；物理 I/O 收益需测量 |
| Completion 推进前核对 operation/key/stage 等身份 | `crates/ferrule-runtime/src/io/registry.rs:1621–1656` | 需要故障测试矩阵；接口机制不等于真实设备完成正确性的通用证明 |
| Broker 对 covered domains 做 claims validation，再更新账本 | `crates/ferrule-runtime/src/scheduling/resources.rs:491–545,561–607,641–710` | 非全进程 RSS 上界；应列出 fixed allocations、allocator reserve、metadata、OS cache |
| Submitted custody 与 release 分开；provider return 不自动释放 credits | 同上 `resources.rs:561–607,641–710` | 需验证组合失败与资源守恒，不能声称所有资源都使用同一 broker |
| Resident ending Pending 保留 payload，backend complete 后进入独立后处理阶段 | `crates/ferrule-runtime/src/engine/driver/resident.rs:15–39,482–527` | 局部阶段机不等于跨崩溃或跨 owner 原子提交 |
| CUDA decoder 的 KV working state 在 commit complete 后安装；prepared physical KV 区分 publish/retire/finish | `crates/ferrule-model/src/decoder/transaction.rs:493–510,534–575`; `crates/ferrule-backend/src/cuda/operators/kv/transaction/prepared.rs:5–55,142–159` | 当前是 CUDA decoder/backend 路径的限定证据；`prepare_publication` 不是 all-owner physical ready vote，也不推广为全模态或全设备保证 |
| Logical cancellation/terminal 不等于 physical quiescence | `crates/ferrule-common/src/io_protocol.rs:879–901`; `crates/ferrule-runtime/src/engine/driver/shutdown.rs:92–135`; `crates/ferrule-runtime/src/engine/driver/resident.rs:48–83` | 已开始的不可逆 physical Publish 仍可完成；terminal delivery 不是 fence |
| Unknown completion 保留 custody；cleanup 失败保留后续清理载荷 | `crates/ferrule-model/src/decoder/kv/prepare_error.rs:1–14`; `crates/ferrule-runtime/src/engine/driver/resident.rs:669–687`; `crates/ferrule-runtime/src/io/registry.rs:1580–1608,2220–2234` | 可能无限期占用容量；不推出 eventual reclamation、general recovery 或 availability |

### Cancellation 与 physical Publish

[架构事务合同](../../docs/ARCHITECTURE.md#2-execution-transactions)区分 caller-visible cancellation 与不可逆物理发布：

> Accepted cancellation determines the caller-visible terminal outcome but does not revoke an already-started irreversible physical Publish.

该限定不承诺故障恢复或提前释放 custody；故障实验见 [性能协议](../../docs/PERFORMANCE_EVALUATION.md#fault-model-and-budget-conservation)。

## 数值与性能：允许保留的状态，不直接作为主文结果

| 记录 | 本地证据 | 解释边界 |
|---|---|---|
| 2026-10-07 adapter-extraction F32：6/6 PASS，7,946,240 logits，零 failures/nonfinite | [REPORT](../../target/validation/adapter-extraction-readiness-20261007/snapshot-26604d6076c14cd1-1791384636495274444/REPORT.md)；同目录 `ACCEPTANCE.json` / `RELEASE.json`；source `26604d6076c14cd10510ff22e7434f7453d4bfe3ba2e93f34fce6753b6d52654`，binary/oracle pins 见收据及 [索引](../../docs/CURRENT_ACCEPTANCE.md) | PP2TP2EP2 3,973,120（含 fork/recreate）及分别测试的 resident thread/process EP8 各 1,986,560；resident 各有 route/state 检查，pipeline 无 raw route/state oracle；非四拓扑全面重验、当前 checkout 或性能 PASS；GPU 10 秒采样，host peak 未实测 |
| 2026-10-07 Mixed prepared-attention 原 full40 与 bounded fresh quality PASS | [FINAL.json](../../target/validation/bf16-attention-model-current-20261007/FINAL.json)，状态 `PASS_mixed_prepared_attention_original_full40_and_fresh_quality`；GPU-tested source/binary/reference pins 与 post-GPU CPU-only 变化分别记录 | 原 full40 804 metrics / 1,986,560 logits；fresh QA/arithmetic/Chinese 5,427 metrics，其中 27 logits metrics / 9,187,840 elements，零 tolerance failures；free-greedy、QA fork/reset 和另一个 fresh owner QA 通过；不是自动故障 replay 或通用任务质量，arithmetic 固定 token budget 内未答完 |
| 较早 Mixed public-serve：Hello lifecycle PASS | [FINAL.json](../../target/validation/mixed-public-serve-owner-20261007/admission-deadline-fix/FINAL.json)，`MIXED_PUBLIC_SERVE_LIFECYCLE_PASS` | 另一 source/binary，早于最新 attention；不能拼成最新 binary 已有 public serving + quality/performance PASS；page cache 未控制，restart 非 storage-cold |
| 历史冻结 current-owner F32 process-EP8 full40：1,986,560 logits；max_abs 4.38690185546875e-5；记录无 violation/nonfinite | [原 current-owner 报告](../../target/validation/qwen35-35b-current/current-owner/REPORT.md) | 仅该 profile、输入与冻结快照；不是本轮重跑，不借给 W8A16 或之后工作树 |
| W8A16 独立 single-GPU full40：strict gate FAIL | [W8 strict 失败范围](../../docs/PERFORMANCE_EVALUATION.md#15-independent-full40-w8a16-oracle-exists-strict-acceptance-failed); `target/validation/w8a16-full40-independent-20261002/SUMMARY.json:3–7,53–66,74–80` | oracle 已存在；不是“缺少 oracle”；不是 EP8 serving 验收 |
| 上述 W8A16：atol=rtol=2e-4；1,837,493 / 1,986,560 logits 超差；max_abs 0.2965252995491028；4 个 ordered route-ID mismatch | 同上；`target/validation/w8a16-full40-independent-20261002/quality.json:180–200` | 8/8 argmax 一致不能推翻 logit、route、state 的严格失败；不修改阈值去追求通过 |
| matched-client current-owner 性能窗口已存在 | [冻结 matched-client 结果](../../docs/PERFORMANCE_EVALUATION.md#14-current-owner-matched-client-results--2026-10-02) | 同一 server 的窗口，不是独立随机重启；没有可直接宣称的显著性、稳定 parity 或 W8A16 质量合格性能 |
| vLLM 实际部署参考存在，但 arithmetic/topology 不匹配 | [历史部署及精度差异](../../docs/PERFORMANCE_EVALUATION.md#11-historical-aligned-evidence-snapshot-through-v4) | 不生成等精度 runtime speedup；区分服务端与 benchmark client |
| 全专家 prewarm EP8 的吞吐记录 | [历史 v4 测量及限制](../../docs/PERFORMANCE_EVALUATION.md#11-historical-aligned-evidence-snapshot-through-v4) | 不证明请求期存储物化、超过去重物理驻留容量的推理或磁盘必要性 |

W8A16 profile 为 BF16-RNE numeric weights/inputs、原 multiplicative scales、F32 model/KV/state boundaries；不能简称“全 BF16”。早期 divergence 分析及 kernel-order replay 不替代独立 oracle，也不撤回 strict FAIL。

Mixed 的 PASS 只属于 `Bf16ModelMixedF32Accumulate`：E4M3FN + BF16 scales，BF16 activation/residual/device logits/KV/conv、F32 accumulation/SSM；logits 非 bit-exact（fresh max abs 0.0625），state/route 数值相等不是额外 signed-zero bit audit。uniform H/Hkv/D、context<=1024；public serving 另需 batch<=min(32, context)、单 owner/resident KV/lazy experts。更早 scalar/attention6 和 F32 failed attempts 保留历史，不能替换原 reference、放宽阈值或擦除独立 W8A16 strict FAIL。

## 当前四 RQ 与本轮证据（2026-10-14）

- **RQ1 / quality blocked：**StorageMiddle64 config-match/CPU metadata、构建及 [exact selectors](../../target/validation/storage-middle64-current-quality-20261014-01/exact-ignored-selection-proof.json) ready（64 experts、320 MiB 含 8 MiB scratch、context128/maxdecode2）。[2026-10-14 FINAL](../../target/validation/storage-middle64-current-quality-20261014-01/FINAL.json)：首次 full40 inference 前 `cuInit` / `CUDA_ERROR_NOT_INITIALIZED (3)`，3 scheduled/1 dispatched/1 failed/0 passed/2 unissued/0 retries、0 metrics/logits；fresh-quality/cold-owner 未发出、诊断 ongoing。非数值 mismatch/PASS；[test/CLI identity](../../target/validation/storage-middle64-current-quality-20261014-01/tested-vs-serving-relation.json)不同，当前 source/quality 闭包未建立。
- **RQ2 / logical actual only：**[wrapper -01](experiments/storage_demand/runs/storage-demand-tp1-middle-20261013-01/report.json) DEVNULL `AttributeError` 保留：4 planned/0 dispatched/4 not-run。[随后 -02](experiments/storage_demand/runs/storage-demand-tp1-middle-20261013-02/report.json)：n=1 DEBUG、4/4 HTTP、3/3 limited canaries，Hello HTTP/usage only；logical observations actual。无 generated IDs/full numerical quality，physical NAS/network/block unknown、page cache uncontrolled；非 backing necessity。
- **RQ3 / open：**同 weights/预算/cache/quality/generation/fence/custody 的 safe-blocking reference 未实现；无 matching blocking/async comparison、真实 overlap 或无阻塞结论。
- **RQ4 / open（观测实现 CPU-tested）：**model-neutral resource observability 与 server actual-token enqueue/terminal trace 已实现、CPU-tested（[resource tests](../../crates/ferrule-model/tests/resource_observability/mod.rs)/[token tests](../../crates/ferrule-server/tests/unit/worker_token_tracing.rs)），无新 GPU serving evidence。enqueue 非 HTTP consumption/fence，summary 非 authority/final release，Unknown sticky。-02 `resource_contract_complete=unknown`、raw `gpu_quiescence=unverified`；fault conservation/final closure open。
- **重复/physical-I/O preparation only：**[Pilot V2](experiments/storage_demand/newversionedprotocol/protocol.json)/[CPU receipt](experiments/storage_demand/newversionedprotocol/validation-cpu.json)：ceiling 12 = 4 initial + 1 excluded QA + 2 warmups + 5 measured QA，固定 L1–L5、60 planned/25 measured QA slots，CPU-only、candidate unbound/unexecuted。NFS-client helper 非 physical measurements，shared-client counters 非 server-attributable NAS disk/network，missing/reset 留 null；5 lifecycles 非统计 PASS/功效/CI/storage-cold。

**历史 E2 计划 parked：**2026-10-08/09 E2 升 P0/先达到 vLLM、TP1 p128/o32/c1 2-block pilot、≥5 paired-block 95% CI/零退让仅性能升主 claim 时恢复（[原规则](PAPER_COMPLETION_CHECKLIST.md#parked-broad-scope)）。共同质量/工作量/资源、完整分母及延迟/SLO 先冻结；比值与 CI 下界 >=1 才过旧门禁，严格优势需下界 >1，“不显著”非 parity。soft 7 GiB 非 hard cap，system-level 非 arithmetic match/E0/disk necessity。

## TP2 resident vLLM baseline（2026-10-09；局部 E2 证据）

成功生命周期收据为 [`tp2-resident-single-lifecycle-20261009-001`](experiments/tp2_resident/RESULTS-single-lifecycle.md)，机器可读主收据为 [`RESULTS.json`](experiments/runs/tp2-resident-single-lifecycle-20261009-001/RESULTS.json)。这是 TP2/PP1/DP1、无 EP、无 CPU offload（`--cpu-offload-gb 0`）的 vLLM Marlin baseline；一次启动的同一 server 依次完成 Hello2、三个 canary 与 QA8。模型为 Qwen3.5-35B-A3B-FP8（text-only），两张 RTX 3090；冻结 profile 为 BF16 activation/KV/conv、F32 SSM，raw completions、natural EOS、prefix cache off、eager、concurrency 1。launch ID 为 `d826244b213947d4b5ce2843f1537961`。

| 项目 | 观测 | 证据边界 |
|---|---:|---|
| startup | 65.7325 s | 单次 lifecycle；不代表独立冷启动分布 |
| Hello2 | `, I`; 1 input / 2 output；TTFT 230.452 ms；E2E 311.259 ms | 请求成功；不是 full-model numerical quality |
| 三个 canary | 3/3 visible-prefix + usage matches | 仅冻结可见文本与 usage；无 token-ID/logit/route/state/full numerical oracle |
| QA8 | 5/5 successful measured requests；mean TTFT 130.66 ms；mean E2E 703.34 ms | first QA-phase request excluded，另有 2 warmups；同一 server |
| measured work | 40 actual output tokens / 3.5363 s client wall ≈ 11.31 tok/s | client overhead included；不是 larger-load、cold-cache 或 paired throughput |
| sampled resources | 19.284 GiB/card GPU peak；6.879 GiB server-tree PSS peak | sampled whole-device/PSS observations；not exact peaks or a hard partition |

**2026-10-09 历史局部状态：**QA8 descriptive only；`shared-full40-01`/`shared-tiny-01` 为 PP1/TP2/EP1 separate gate/up 修复后的 logits/lifecycle gate（1,986,560 有限 logits 零误差；[CPU audit](experiments/tp2_resident/claim-gates-audit-20261009.json)）；当时 all-boundary/state、multistep/serving 未放行。后续 TP2 材料见 [checklist E2](PAPER_COMPLETION_CHECKLIST.md#e2--服务性能与统计conditional-secondary旧-p0-门禁已-parked)，不追改旧收据。HF-eager TP2 非 Marlin arithmetic；`fairmatch=false`、`exact_speedup=null`，无 fair speedup/parity。E2 现为 conditional secondary，E2/T4/B5/统计 checkbox 不勾。

原始入口：[sealed pins](experiments/runs/tp2-resident-single-lifecycle-20261009-001/prepared.json)、[server log](experiments/runs/tp2-resident-single-lifecycle-20261009-001/session/server.log)、[canary gate](experiments/runs/tp2-resident-single-lifecycle-20261009-001/session/canary8/gate.json)、[QA8 HTTP receipt](experiments/runs/tp2-resident-single-lifecycle-20261009-001/session/qa8-V1/http.json)。QA8 每请求 5 input / 8 output，全部 finish reason 为 `length`，natural EOS policy 未改；这里的 first QA-phase request 不是服务器 first-after-Ready。server/JIT/page caches 未清空，不能把更短 startup 解释成优化。

### Effective runtime evidence and profile separation

TP2 成功 server log 的有效选择包括 `MarlinFP8ScaledMMLinearKernel` 和 `MARLIN Fp8 MoE`；GDN prefill 为 Triton/FLA，因 `fused_gdn_decode_post_conv_mtp` 未构建而 fallback 到 Triton GDN decode。NCCL 日志明确 `NCCL_P2P_DISABLE=1`，local channels 为 `via SHM/direct/direct`；Socket plugin 的初始化不被解释为 local inter-GPU payload 使用 Socket。M7/G10 的独立 UVA profile 才记录 `UVAOffloader` 与 `Total CPU offloaded parameters: 30.0`，其日志分别在 [`M7 vLLM log`](experiments/runs/m7-feasibility-exclusive-002/vllm/server.log) 与 [`G10 vLLM log`](experiments/runs/g10-exclusive-001/feasibility-vllm/server.log)；这些不属于 TP2 no-CPU-offload baseline，也不能与其拼接。Marlin packing/scales/dequantization、GDN fallback、activation/KV/conv/SSM 边界和 reduction 差异意味着没有 arithmetic equivalence claim。

TP2 的显式 zero CPU offload 和 installed-source defaults 指向 `NoopOffloader`，但其直接 debug log 未观察到；没有逐 tensor residency audit，不能从日志缺少 offload 推出所有 tensor 放置已证明。UVA selection 同样不独自证明 zero-copy。checkpoint 身份仍为 metadata hashes + shard stats，非完整 payload hash；[loaded-library evidence](experiments/runs/tp2-resident-single-lifecycle-20261009-001/loaded-library-evidence.json) 同时看到 CUDA12/NCCL 和 CUDA13.2 runtime/NVRTC，动态依赖闭包仍有限制，不据此断言 run 004 失败原因。

### Failed starts and observer/PID caveats retained

Baseline-delta runs 003 and 004 仍是 append-only failed startup evidence：003 在 lazy CUDA initialization 期间因 observer `nvidia-smi` timeout 中止；004 的 worker 在 NCCL 前报告 `CUDA driver initialization failed, you might not have a CUDA gpu.`；两者均未发 Hello2/canary/QA 请求。它们不被改写为 canary failures，也不从失败分母删除。相关汇总见 [`RESULTS-baseline-delta.md`](experiments/tp2_resident/RESULTS-baseline-delta.md) 及其 [`run 003`](experiments/runs/tp2-resident-20261008-003/execution.json)、[`run 004`](experiments/runs/tp2-resident-20261008-004/execution.json) receipts。

更早 [`RESULTS.md`](experiments/tp2_resident/RESULTS.md) 中 runs 001/002 的 Hello2 成功和 canary startup abort 仍保留：001 是 PSS ESRCH，002 是 nonselected GPUs 上来源 unknown 的 1 MiB 观察；均无 canary HTTP，不能说是数值 FAIL 或 PASS。[historical comparison](experiments/runs/tp2-resident-single-lifecycle-20261009-001/comparison.json) 保留 004 driver failure 根因 unknown；本次成功不追溯消除失败，也不改 M7/G10 或任何历史 target。

该历史 run：`owned_gpu_attribution=unknown`；server PID 14168/worker 15060/15061 仅本地身份，非 host-NVML 映射。`/proc/15844/smaps_rollup` 消失的 leaf 为 incomplete/PSS null；31 complete host + 1 incomplete host、32 GPU samples、0 missing NVML，非连续覆盖。nonselected deltas 全 0，未用 1 MiB tolerance；19.284 GiB/card >19.2 planning、<22 observational ceiling。128 GiB PSS 非物理 cap，不含 unmapped page cache/不证明 pinned bytes。[drain/ECHILD](experiments/runs/tp2-resident-single-lifecycle-20261009-001/execution.json)、[八卡 idle](experiments/runs/tp2-resident-single-lifecycle-20261009-001/final-owner-recheck.json)、[lock release](experiments/runs/tp2-resident-single-lifecycle-20261009-001/final-owner-release.json)为历史局部 cleanup，非本次查询/physical fence，限制不消除。

该历史配对路线仅恢复性能主张时执行：先绑定同配置的 quality/lifecycle 与 serving source/binary，再冻结匹配合同/fresh paired blocks、完整失败分母和 block statistics；当前下一步仍是 RQ1–RQ4，不先做 vLLM 胜出。

## 待补写，不应通过推测补齐

- **四 RQ 核心 open：**同配置完整质量/容量/物理 I/O、安全 blocking/async、集成故障/custody；紧凑权重驻留资格、cache/swap、去重各池峰值须闭环，磁盘必要性未确认。
- **Artifact：**source/dirty snapshot、binary/model/tokenizer/client/oracle、raw 与失败全分母冻结；covered/untracked/fixed memory 分明。ignored 目录 portable 归档/hash/恢复及干净环境复现未验收，不跨 profile 借 PASS。
- **范围与代价：**可执行支持表按模态/backend/设备/拓扑/表示建立；真实 overlap、正常/观测成本及安全消融待验收，扩展仅升主张时恢复。
- **Related work 局部：**FlexGen/ZeRO-Inference/PowerInfer-2 为 storage/streaming prior art，PowerInfer 为 GPU–CPU hybrid，vLLM 为 serving KV（[正文](sections/related-work.tex)）；非首个 storage inference、非未审计能力缺失。metadata/BIB/guarantee audit open，SGLang 未核查。
- **投稿：**ORCID/资助/致谢/利益冲突/许可需确认，最终 claim/视觉审读未验收；四 -02 raw-to-plot 图和当前构建形态见下，非科学门禁完成。

## 构建与环境记录：区分上版和新版

**当前稿件形态（2026-10-14）：**cohesive paper 以 RQ1–RQ4/合同/related work/限制组织；7 彩图（3 mechanism、4 -02 data）、标准 `algorithm`/`algorithmic` 与数学/数据图已接入。[现有构建记录](README.md#current-plot-integration-validation)/[当前日志](build/main.log)：`make -C papers/arxiv` PASS，30 页/725591 bytes，warnings/undefined references/citations/overfull/underfull/oversized floats/duplicate labels 均无。局部审读已有记录，非本次构建/审读或科学 PASS。**上版历史：**make 退出 0、2 轮、9 页/183734 bytes、最终无残余警告，不移作当前验收。

**历史源码构建验证（2026-10-02，非本次执行）：** `make -C papers/arxiv` 退出 **0**，初次为 up-to-date；`make -B -C papers/arxiv` 仍未触发 pdfLaTeX。随后在 `papers/arxiv/` 执行 `latexmk -g -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex`，退出 **0**，实际运行 **1 轮 pdfLaTeX**；再次 `make -C papers/arxiv` 退出 **0**。所有构建调用均设 120000 ms 限时且未超时。最终 `build/main.log` 报告 **15 页、299612 bytes**，PDF 文件大小核对一致；无 LaTeX/Package warning、overfull、未定义引用或重编译请求，但仍有 **2 条 underfull hbox**（`sections/evaluation.tex:203`，badness 10000 和 2261）。页数依据本次编译日志；未进行独立 PDF 解析或人工视觉审读，标题、作者、公式、表格与引用的视觉检查仍待完成。本次未改 TeX，未安装包，未运行 GPU 或数值实验；编译成功不构成容量、模态、异步或质量验收。

**历史 Python 环境记录：**`/opt/conda310/bin/python`，3.10.9；不冒充当前环境或本次版本验证。V2 的 literal CPU argv 见其冻结 README，旧 absolute pins 不改写。

## 文档修改的验证含义

**2026-10-08 历史：**仅 README/ledger/checklist 同步权威收据/计划，CPU 文本核对；无 TeX/引用/实现/历史 raw/experiments 修改，无 GPU/数值/构建。10-02 构建未重跑，E0/比较/统计未验收。

**2026-10-09 历史：**三份 Markdown 同步 TP2 receipts，仅链接/指标复算/差异检查；无实验/脚本/docs/TeX/target 修改或 GPU/数值/构建，E2 仅局部 baseline，E0/fairness/parity open。**2026-10-14 本次：**只改 checklist/ledger，窄 Markdown 检查；无新 GPU serving/数值/physical-I/O measurements，无构建或其他文件修改。
