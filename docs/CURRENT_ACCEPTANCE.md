# 当前验收边界

更新：2026-10-07。本文是维护中的验收索引，区分实现支持、冻结运行证据与仍待验证的能力。
所有 PASS 只属于报告记录的 source/binary/reference pins；后续源码修改不自动继承验收。
本次文档核查未重跑 Cargo、GPU、服务或 benchmark。`target/validation` 是本地证据，
不保证随仓库分发；缺少 artifact 时不能把链接本身当作可复现证明。

实现与命令见 [Qwen3.5](QWEN35.md)、[并行支持](PARALLELISM.md)、
[Mixed serving](QWEN35_MIXED_SERVING.md)。长期边界见 [架构](ARCHITECTURE.md)，
未完成工作见 [路线图](ROADMAP.md)。

## Mixed：原始 full40 与 bounded fresh quality 已通过

权威结果是 [prepared-attention FINAL.json](../target/validation/bf16-attention-model-current-20261007/FINAL.json)，
状态 `PASS_mixed_prepared_attention_original_full40_and_fresh_quality`：

- 原始 Hello/capital full40：**804 metrics PASS**，**1,986,560 logits**，零 tolerance failures。
- Fresh QA/arithmetic/Chinese：**5,427 metrics PASS**，其中 **27 logits metrics / 9,187,840 elements**，
  零 tolerance failures；沿原 HF common history 比较 27 个 predictions。
- 三个 prompt 的独立 free-greedy feedback 均相等；QA fork/reset、另一个新进程/owner 的
  QA 9 predictions replay/free-greedy 与 reset 通过。不是自动故障重启/replay 支持。
- 原始 reference、算术策略、阈值、cap 和默认 profile 未改变。更早的 scalar/attention6
  失败保留为诊断历史，不再作为这次冻结 Mixed evaluation 的当前阻塞。

这仅接受 `Bf16ModelMixedF32Accumulate` 的指定输入/容量：严格完整 35B-A3B numeric
E4M3FN + BF16 scales，BF16 activation/residual/device logits/KV/conv，F32 accumulation/SSM。
prepared attention 要求 uniform H/Hkv/D、context <=1024；公开 serving 的 batch 还需
<=min(32, context)，单 owner、resident KV、lazy experts，其他约束见 serving 文档。

Logits 满足原 float bounds，**不是 bit-exact**（fresh max abs 0.0625）；state/route 数值相等
不等于额外 signed-zero bit audit。Arithmetic 在固定 token budget 内未完成完整答案。
这不是通用任务质量、所有 prompt/context、分布式 BF16、公开服务新 binary 或性能验收。
FINAL 同时记录 GPU-tested pins 和 GPU 后 CPU-only budget assertion/documentation 变化，
不能把不同 binary hashes 合并为一次通过。

## 最新 F32：adapter extraction 后代表性 pipeline 与 resident EP8 验收

20261007 snapshot 的 [REPORT.md](../target/validation/adapter-extraction-readiness-20261007/snapshot-26604d6076c14cd1-1791384636495274444/REPORT.md)、[ACCEPTANCE.json](../target/validation/adapter-extraction-readiness-20261007/snapshot-26604d6076c14cd1-1791384636495274444/ACCEPTANCE.json)
与 [RELEASE.json](../target/validation/adapter-extraction-readiness-20261007/snapshot-26604d6076c14cd1-1791384636495274444/RELEASE.json) 记录 **6/6 PASS**、GPU ownership released。
本次只核查收据；source/binary pins 如下，不从目录名推导当前 checkout 已通过：

| Pin | SHA256 |
|---|---|
| Source | `26604d6076c14cd10510ff22e7434f7453d4bfe3ba2e93f34fce6753b6d52654` |
| Composition test binary | `a3df7c397522df549dcb7dbeb844421e4ebd963853e8c1e2478006f903ad0911` |
| CLI/process child binary | `a1ed1c3954c574c1aa7f356e940a0bddfa06aa2e72ed1913f254440b60f18f8b` |
| Original oracle manifest | `d688e8ec9effd0319159902100ba7b463d6ddb475e6236834fa81872d6f3d33d` |

两个 binary 来自同一 immutable source，CUDA12.4/sm_86；收据确认 source/artifact/model/
reference/CUTLASS/workflow pins 复核通过，原 oracle 未替换。

| Case | Logits 比较数 | Max abs |
|---|---:|---:|
| Tiny segment / tensor / resident-thread（3 tests） | 未输出 aggregate count | 未输出 aggregate error |
| PP2 TP2 EP2 thread full40 | 3,973,120（含 fork/recreate；base 1,986,560） | 3.647804260253906e-5 |
| Resident EP8 thread full40 | 1,986,560 | 4.315376281738281e-5 |
| Resident EP8 process full40 | 1,986,560 | 4.315376281738281e-5 |
| **严格 logits 合计** | **7,946,240** | 零 failures / nonfinite |

原 `atol=rtol=2e-4` 不变。两个 resident EP8 run **各自**还通过 2,560 ordered route IDs
精确比较、2,560 routing weights、3,932,160 conv state 与 62,914,560 recurrent state
原 bounds 检查。Pipeline 不暴露 raw route/state oracle，不能借 resident 结果补成已比较。

Lifecycle：pipeline fork/recreate/cancel、normal shutdown、zero pages/outstanding、无
quarantine；resident strict `shutdown_physical` 成功，runner drop 后 live state bytes 为零、
无需 quarantine。Process Ready 验证八个互斥 owned catalogs，各 1,280 experts；四个 drained
barriers 保持 residency/零 pending、零 eviction、零 quarantine，root 无 shared host cache。
Authenticated subreaper/ECHILD 证明进程 custody；另有独立的各 case 前后 GPU-idle 检查，
RELEASE 记录八卡 0 MiB/0%、无 compute processes、expert children 已退出。进程退出本身
不是 GPU fence；NVML/local PID mapping 未记录。GPU peak 为 10 秒采样，host peak 未实测。

此前失败保留。修复的是测试 helper 的 Finished terminal/cleanup receipt 消费与身份校验，
并补 CPU regression，**不是 production 算术或资源预算 bug 修复**。相对失败快照，唯一
Rust 变化为 `qwen35_ep_tests.rs`；独立更新的 `hybrid_cuda_tensor.md` 也在 build 前冻结。
收据记录 production/math/budget/oracle 无变化、CPU regression 与 all-feature/all-target
Clippy 通过；本次文档核查未重跑这些命令。

这是最新 snapshot 的**代表性 PP2TP2EP2 + resident thread/process EP8**证据，增补而非
重跑下方全部四拓扑矩阵：其余 PP2TP1EP2、PP2TP1EP4、PP1TP2EP2 未在这批重验。
Mixed/BF16 也未在这批重验；既有 Mixed PASS 仍只属于原 attention receipt 的 pins。
不开放 numeric process PP×EP、process TP 或新 precision，不构成公平 vLLM/E2E 性能证据。

## 历史 F32：较早 composition 重构后的四拓扑 full40 矩阵

[GPU-owner ACCEPTANCE.json](../target/validation/parallel-refactor-readiness-20261007/gpu-owner-current-20261007/ACCEPTANCE.json)
记录 `passed: true`，source `ec5e614055317a648f401679d0cb63e53fe83b0bdeb39418ba6e4eef8d5a461e`，
固定 binary/reference identity，未重生成 oracle。此为较早冻结矩阵，不是上述 adapter
extraction source 的四拓扑重跑；保留原计数，不与新批次合并为一次全矩阵验收：

| Thread CUDA topology | 比较数（包含 fork/recreate） |
|---|---:|
| PP2 TP1 EP2 | 3,973,120 |
| PP2 TP1 EP4 | 3,973,120 |
| PP1 TP2 EP2 | 3,973,120 |
| PP2 TP2 EP2 | 3,973,120 |
| **合计** | **15,892,480** |

每拓扑 primary logits 为 1,986,560；使用原 F32 strict bounds。维护中的 exact selectors、
设备排列、容量与复现命令见 [full40 harness](../crates/ferrule-composition/tests/unit/engine/model_factory/full40_strict/README.md)。

范围限制：未直接比较 router/GDN/KV raw tensor oracle；本矩阵没有 process-transport factory
GPU case；TP4/其他 EP 度数的构建支持不是 full40 acceptance。GPU peaks 是 10 秒采样，
host 数值是 metadata，不是实测峰值。测试 wall time 不是 serving 吞吐或公平 vLLM 对比。

## 较早 Mixed public serve：仅 Hello lifecycle，另一份 binary

[admission-deadline-fix FINAL.json](../target/validation/mixed-public-serve-owner-20261007/admission-deadline-fix/FINAL.json)
记录 `MIXED_PUBLIC_SERVE_LIFECYCLE_PASS`，明确限定
`Hello lifecycle only; no fresh-quality/performance acceptance`。覆盖公开服务启动、Hello SSE、
disconnect/cancellation 后可用性、正常 teardown 与同端口新进程/model restart。
OS page cache 未控制，process/model reload 不是 storage-cold。

该 receipt 早于最新 attention 修改，binary/source pins 与上面的 model/composition quality
运行不同。**两份 PASS 不能拼成“最新 attention binary 已通过 public serve + quality”**；
需在新的匹配 source/binary 上单独重建、准备并执行服务验收。

## 精度与支持不能混同

| Profile | 当前解释 |
|---|---|
| `F32Tf32x3` | Numeric serving 默认；F32 model boundaries，TF32x3 compute，不承诺严格 IEEE SGEMM 或 bitwise parity |
| `W8A16Bf16F32Accumulate` / `w8a16-bf16` | 显式 experimental opt-in；BF16 operands、F32 accumulation，model activation/KV/state/logits 仍为 F32；**full40 strict FAIL，quality-unaccepted** |
| `Bf16RneF32Accumulate` | 旧 low-level diagnostic profile，不是 serving 默认或公开 expert-wire 选项 |
| `Bf16ModelMixedF32Accumulate` | 上述单-owner Mixed bounded PASS，不扩大 F32/W8 distributed 支持矩阵 |
| Native dense BF16 | [标准 dense Qwen3 profile](BF16_DENSE_CUDA.md)，原生 BF16 checkpoint；不能借 Mixed numeric FP8 证据升级质量状态 |

[W8 operator 参考](../crates/ferrule-backend/tests/numeric_w8a16/README.md)的独立局部验收不能
替代 [W8 full40 失败报告](../target/validation/w8a16-full40-independent-20261002/SUMMARY.json)。
W8 task quality **NOT RUN**；保留失败记录。Numeric expert wire 只支持 `f32` / `w8a16-bf16`，
不支持 BF16-model/Mixed。CUDA process TP、numeric hybrid process PP×EP、自动故障 replay
不因 composition 重构或 Mixed PASS 而开放。已有 resident process EP 与 strict dense
0.8B process PP2 是另外的受限路径。

## 性能与其他独立门禁

**公平的当前 Ferrule/vLLM end-to-end 性能对比尚未完成。** 历史真实 vLLM server 行存在
precision/topology mismatch；operator timing、测试耗时、Hello lifecycle 和进程内重复窗口
均不能证明同精度/同拓扑的吞吐优势或统计显著性。方法与历史测量见
[PERFORMANCE_EVALUATION.md](PERFORMANCE_EVALUATION.md)。

以下独立范围没有因本文新增的 Mixed/F32 receipts 而升级：

- [HostKV ordinary fixture](../crates/ferrule-runtime/tests/host_kv/runner_fixture.rs)与
  dense Qwen3 F32 HostKV serving factory 是不同验收范围；不能从前者宣称后者 GPU PASS。
  [Physical placement contract](../crates/ferrule-backend/tests/unit/physical_placement.md)
  本身不是 production `Entry`/kernel 集成或 VRAM saving 证明。
- [旧 restart 报告](../target/validation/qwen35-35b-current/append-segment-restart-owner/REPORT.md)
  的 14 次 restart、56 次成功观察不是统计学证明；
  [p2048 profile](../target/validation/qwen35-35b-current/profile-handoff/REPORT.md)
  warmup 超时且缺少 nsys/IPC 产物，不能据此归因。
- 历史 [CPU](../target/validation/recovered-integration-cpu-final-rerun-20261002/REPORT.md) 与
  [CUDA/combined-reference lint](../target/validation/qwen35-35b-current/endpoint-layout-fix/mechanical-lint-followup.md)
  只属于其冻结运行，不代表这次文档更新执行了最终 workspace tests/Clippy。
