# 并行执行集成状态

本文区分当前源码的集成能力、本轮已完成的实跑证据与尚未覆盖的边界。下面的
验收结果汇总本轮执行记录；文档收尾本身不重复 GPU 测试，也不把小 checkpoint
fixture 的结果外推为更大的模型、更多节点或更强的故障恢复能力。
Ferrule production runtime 不依赖、链接或调用 NCCL；外部 Python/NCCL 对比工具
不属于 Ferrule transport。

**证据时序**：标准 linear 的 GEMM 已切换到 CUTLASS TF32x3。下文明确标记的
旧 GPU 数值和 serving 记录属于更换 GEMM 前的历史验收，不能作为当前实现的
回归通过证据。下方“最终串行验证”已同步更换 GEMM 后的新实测，Qwen35 与
Qwen3 分开列出；选定测试通过不等于无跳过的全 workspace / 全 GPU / strict Clippy
全部通过。文档更新只核查已有报告与日志，不运行 Cargo。

## Qwen3.5-0.8B：完整 text CPU / 单卡 CUDA 已接通

Qwen35 当前只支持严格 dense 0.8B profile，完整 **24 层（18 GDN + 6 full GQA）**。
真实 CPU/单卡 CUDA 数值验收已通过（本轮既有结果，文档收尾未重跑）；CLI 默认 CPU，
显式 `--backend cuda` 已经由 model factory 接入 resident engine / model worker / HTTP/SSE。
最终日志确认 `hybrid_cuda` **13 tests 通过**，其中 **6 个实际 CUDA 测试、
1 个比较器验证、6 个 CPU reference 测试**；另有独立 forward CUDA UT **1 passed**。
Qwen35 `capital` + `hello` 的所有
prefill rows 和各两次 decode 共 **2,483,200 logits** 完成逐元素比较及 manifest/hash
校验，最大绝对误差 **`3.337860107e-5`**，greedy 预测一致。此结果不是下面旧 Qwen3
PP/TP 数值的替代证据。

真实默认 `serve /mnt/nas1/hf/Qwen3.5-0.8B --backend cuda` **6.77 s ready**，
SSE 输出 `Hello` 和 ` Paris`，SIGTERM exit 0，退出后无残留 GPU process。
6.77 s ready 是先前前台启动记录；最终 CLI target 又实际通过，耗时 69.59 s，
不能把 target 总耗时当成 ready 时间。文档收尾未重跑。详细功能矩阵、逐 case 数值
及精确复现命令见 [QWEN35.md](QWEN35.md)。

- checkpoint 为 schema 限定的 **BF16/F32 混合存储**，执行 tensors 为 F32。
  GEMM 使用现有 CUTLASS provider 的 **TF32x3**，F32 boundary/accumulation/output，
  不是严格 IEEE F32 SGEMM，也不承诺 bitwise 相同；
  GDN 专用 CUDA kernels 经共享 operators 接入，不走 Python 或 CPU fallback。
- 只执行 text；vision/MTP 附件严格校验后排除，不代表附件推理支持。
  **MoE/35B/FP8、hybrid TP/PP/EP、process ranks、prefix cache、partial retain 和
  speculation 均 unsupported**。下面标准 Qwen3 的并行能力不能外推到 Qwen35。
- 每个 recurrent sequence state 为 **20,643,840 bytes**；预算计入 live、transaction
  working copies 和 default state，默认 4 sequences 共 **185,794,560 bytes**。
  KV compact 分配到 6 个 full-attention 层，默认 physical KV 为 192 MiB；
  weights/workspace 单独核算，不能把 KV budget 当成总显存预算。
- 性能尚未优化：本地 debug CLI 后续 decode 约 **15 s/token**，仅为观测，不是承诺。
  离线 Transformers 5.2 oracle、HF 多 token cached continuation 陷阱及本次未重跑的
  验收 targets 也在专页说明；临时启动验证脚本不是 production engine。

## 四项已集成能力

| 能力 | 当前集成边界 | 主要入口 |
| --- | --- | --- |
| 完整 F32 GPU decoder | 标准 dense/MoE decoder 支持 BF16/F32 checkpoint weights，物化为 resident F32 weights，以 F32 activation/KV/logits 执行；真实 dense case 覆盖完整 28 层 | [`full_gpu_decoder`](../crates/ferrule-model/tests/full_gpu_decoder.rs)、[`full_model_cuda`](../crates/ferrule-runtime/tests/full_model_cuda.rs) |
| GPU PP/EP | persistent GPU pipeline/expert owners、sealed KV、fork/COW、取消和提交/回收已接入；覆盖 PP2、PP2×EP2 | [`cuda_pipeline`](../crates/ferrule-runtime/tests/support/cuda_pipeline.rs) |
| 跨进程 PP/EP/KV | production process endpoint 承载 PP/EP child、KV transaction/ACK、bounded host IPC、quarantine/reap；覆盖真实 28-layer dense 和六独立 GPU children 的 MoE fixture | [`process_decoder`](../crates/ferrule-runtime/tests/process_decoder.rs)、[`process_decoder/cuda`](../crates/ferrule-runtime/tests/process_decoder/cuda.rs) |
| `PipelineInferenceEngine → ModelWorker → HTTP/SSE` | model factory 构建 pipeline engine，由 dedicated model worker 驱动 OpenAI-compatible HTTP/SSE；CPU/GPU thread/process 路径均已测试 | [`pipeline.rs`](../crates/ferrule-runtime/src/engine/pipeline.rs)、[`worker.rs`](../crates/ferrule-server/src/worker.rs)、[`pipeline_http`](../crates/ferrule-server/tests/pipeline_http.rs) |

### 当前 standard linear 数值契约

标准 CUDA `linear_f32_into` 现在使用 **CUTLASS TF32x3 TensorOp GEMM**，
tensor boundary、accumulation 和 output 为 F32；F32 precision enum 或 F32 tensor
存储类型不等于严格 IEEE SGEMM 运算或 bitwise 等价。此变化不只影响 Qwen35，
也影响复用该 standard linear 的 decoder 路径。这里不更改 model precision enum，
也不新增 CLI precision knob。更换 GEMM 后的实测见下一节，历史数字仍单独保留。

`sm_86` canonical manifest → native/Rust discovery → plan/execution 已验证贯通：
`F32Gemm ID=11`、mask **`0x586`**、F32 TF32x3 / LinearF32 inference 为 true，
FP8 QueryAKv/Projection 为 false。standard linear graph replay 在原地址写入新输入，
并预先用 NaN / -777 污染输出 D 后仍得到正确结果，device consumer 读取也正确；
不是重放旧输出。证据见 [backend lib 日志](../target/validation/qwen35-final-gpu-01-backend-lib.log)
和 [standard linear / capability 日志](../target/validation/qwen35-final-gpu-02-backend.log)。

### 最终串行验证：更换 GEMM 后的当前证据

来源：[GPU 汇总](../target/validation/qwen35-final-gpu-summary.md)及其分项
`target/validation/qwen35-final-gpu-*.log`、
[CPU 汇总](../target/validation/qwen35-final-cpu-summary.md)。这些是 **Git ignored 的
本地生成证据**，不是随仓库发布的文件；链接仅在保留该次 artifacts 的工作区有效。

- 8 张 RTX 3090，`CARGO_INCREMENTAL=0`、`FERRULE_CUDA_ARCH=sm_86`、
  `--locked`、测试串行且 `--test-threads=1`，命令 timeout 900 秒。
  CUDA workspace all-targets check、CLI / process child 构建通过。
- GPU 验证批次合计 **186 selected passed、0 failed**，**不是 186 项全 GPU 测试**。
  包含 backend lib **113 passed / 18 ignored**；其中 KV 10 项随后另行定向通过，
  不代表所有 ignored 都执行。hybrid 13 项构成见上；加独立 forward UT 后，
  hybrid 组实际 CUDA 测试为 7 项。
- CPU workspace 最终重跑 **1,274 passed、0 failed、6 ignored、2 精确 skip**；
  首轮因缺少 `models/Qwen3-30B-A3B/config.json` 有 2 项失败。只跳过
  `real_qwen3_30b_metadata_binds_to_the_generic_state_dict` 和
  `qwen_adapter_uses_real_standard_kv_and_rejects_cuda`；doctest 另 **7 passed**。
  被 ignored 的 Qwen35 全 24 层 CPU oracle target 随后单独执行 **1 passed**。
- **strict Clippy 尚未全绿**：初次 strict run 报告的两个新增 lint
  `manual_is_multiple_of` / `unnecessary_lazy_evaluations` 已在 `components.rs` 修复；
  既有 `collapsible_if`、`chunks_exact_to_as_chunks`、`items_after_test_module`
  仍是已知阻塞，不能把两个修复或 check/test 通过写成 strict Clippy 全通过。

下列都是 **Qwen3 / 标准 decoder** 数据，不是 Qwen3.5 的 TP/PP 支持证明：

| 本次 target | 当前结果 | 本地日志 |
| --- | --- | --- |
| NAS Qwen3-0.6B，28 层，八卡 PP2TP4 vs GPU TP1 | 全 logits `max_abs=2.1457672e-5`，`max_tolerance_fraction=8.026731e-2` | [NAS PP2TP4](../target/validation/qwen35-final-gpu-04-nas-pp2tp4.log) |
| NAS Qwen3-0.6B，28 层，thread PP2 vs PP1 / causal replay | 本次重测 `max_abs=0`，不是沿用旧数字 | [full model PP1/PP2](../target/validation/qwen35-final-gpu-04-full-model-pp1-pp2.log) |
| 同一 28 层 GPU vs CPU reference | prefill 三行为 `1.36375427e-4` / `8.41617584e-5` / `6.48498535e-5`；首 decode 为 `5.81741333e-5` | 同上 |
| process dense fixture，PP2 vs PP1 | 本次重测 `max_abs=0`；独立 CPU kernels 最大差 `2.3841858e-7` | [process dense](../target/validation/qwen35-final-gpu-04-process-dense.log) |
| process PP2EP2 fixture，PP2 vs PP1 | 本次重测 `max_abs=0`；独立 CPU kernels 最大差 `1.1920929e-7`，4 个 expert children 各执行 6 calls / 12 tokens | [process PP2EP2](../target/validation/qwen35-final-gpu-04-process-pp2ep2.log) |

process 两项的 parent CUDA 保持 `NOT_INITIALIZED`。这里的 process fixture 重测
不是下方历史 NAS 28 层 process target 的重跑，更不等于真实 35B MoE 验收。
Qwen35 真实 CUDA HTTP 和默认 CLI target 分别 **1 passed**；标准 pipeline HTTP
TP2、TP4、PP2TP2、thread PP/EP、process PP/EP 五个 targets 本次 **5 passed**。

[最终 contexts 日志](../target/validation/qwen35-final-gpu-06-final-contexts.log)记录
**0 个残留 GPU contexts / processes**，八卡利用率 0%，显存回到 2–3 MiB 基线。
没有 reset GPU 或终止无关进程。没有执行全 workspace ignored、全部 process faults，
已知旧 shared-FFN formal-shape latency unsupported case 不在范围内；不作全覆盖声明。

### 更换 GEMM 前的历史数值和故障证据

本节保留旧验收的输入、范围和数值以便追溯；**它们不是当前 CUTLASS TF32x3
实现的通过证据**，尤其不能将旧 `maxdiff = 0` 或 CPU-reference 误差解释为当前保证。

- **真实 NAS Qwen3-0.6B，完整 28 层**：hidden=1024、vocab=151936、311 个绑定
  参数（含 tied alias），checkpoint weights 全为 BF16。PP1 为 `0..28`；PP2 为
  `0..14 / 14..28`，没有截层。每 tensor 上限 1 GiB，最大 tensor 为 311164928 bytes。
- **历史 GPU thread PP1 vs PP2（更换 GEMM 前）**：3-token prefill、3 次 greedy decode、取消/重试、完整
  前缀 replay 以及 KV cleanup；全部 vocabulary logits 逐元素比较，PP1/PP2 和
  decode/causal replay 的 `maxdiff = 0`。PP1/PP2 tolerance 是
  `2e-5 + 2e-5 * abs(expected)`，并非只检查 argmax。
- **历史全 28-layer GPU vs CPU F32 reference（更换 GEMM 前）**：三个 prefill 行的最大绝对差分别为
  `1.18255615e-4`、`7.66515732e-5`、`5.22136688e-5`，decode 为
  `4.43458557e-5`；最大约 `1.18e-4`，greedy argmax 一致。
  CPU/replay tolerance 为 `2e-3 + 2e-4 * abs(expected)`。
- **历史 GPU process PP1 vs PP2（更换 GEMM 前）**：同一 NAS 28-layer 模型的 3-token prefill 和一次
  decode，比较全部 logits，`maxdiff = 0`。parent 的 `cuCtxGetCurrent` 始终返回
  `CUDA_ERROR_NOT_INITIALIZED`；CUDA context、model 和 physical KV 在 child 内创建。
- **PP2×EP2**：两层 MoE checkpoint fixture 使用六个独立 GPU children
  （2 PP + 4 EP），核对 PID/device、路由执行和 CPU oracle。另有 healthy idle、
  bucketless experts、oversize command、PP child/leader loss、expert-child loss
  和 deadline 测试，验证 quarantine/reap、不错误 publish/retire/replay。
- **HTTP/SSE**：生成的小 checkpoint 通过真实 model factory/engine/worker/router，
  覆盖 CPU、CUDA thread、CUDA process 的 dense/MoE PP/EP、non-streaming、SSE、
  断连取消、slot reuse、shutdown，以及不支持的 sampling 参数拒绝。这里的 HTTP
  fixture 验收与 NAS 28-layer 数值验收是不同 targets，不能写成完整 35B MoE 验收。

BF16 是上述 GPU decoder 的 **checkpoint 存储格式，不是 BF16 compute**。
F32-weight MoE decoder 另由 `full_gpu_decoder` fixture 覆盖；完整 MoE checkpoint
GPU EP/35B FP8 模型不在本轮已验证范围。

### P1 prepare custody：已修复

原问题是 CUDA KV prepare 失败后仍持有未完成 physical custody，上层却可能把
“没有返回 handle”当作 rollback Complete。现在底层 pool 显式报告 retained custody，
model ledger 保留原 ownership/COW pins，`KvPrepareQuiescenceUnknown` 携带 typed
unknown；pipeline owner 保留 journal，拒绝 cleanup ACK、重试、publish/retire。
相关入口是 [`prepare_error.rs`](../crates/ferrule-model/src/decoder/kv/prepare_error.rs)、
[`owner.rs`](../crates/ferrule-runtime/src/parallel/pipeline/owner.rs) 和
[`prepare_tests.rs`](../crates/ferrule-backend/src/cuda/operators/kv/transaction/prepare_tests.rs)。

故障注入测试已通过，包含真实 CUDA clear/copy 提交后模拟 submission/event error、
上层 typed/untyped unknown，以及有已执行 GPU work 的 child loss/deadline。
**driver 在 submission 注入测试中保持健康；SIGSTOP 是 host process stall，
SIGKILL 是 child loss，均不是物理 CUDA driver fault 或 GPU engine hang。**
测试 teardown 的正向 fence 也不是生产环境的强制恢复入口。

### P1 forward / FailedActive quiescence：已修复

最终日志确认：`forward` 失败进入 `FailedActive` 时的 quiescence 保留修复已完成，
独立 CUDA 故障 UT **1 passed**。精确名称为
`transformer::forward::tests::cuda::hybrid_unknown_finish_keeps_transaction_pins_despite_independent_kv_fence`，
见 [exact UT 日志](../target/validation/qwen35-final-gpu-03-forward-unknown-exact.log)。
最初错误过滤得到的 0 tests 不计通过；本次 docs-only 更新未重新运行。

- **干净失败可重试**：只有确认 forward 已 quiescent、完成必要回滚且未遗留 unknown
  custody 时，才能按正常失败路径清理并重试。
- **unknown 不释放**：forward 的 quiescence unknown 必须保留，不能由后续成功的
  KV fence 清除；KV fence 成功不证明先前 forward work 已完成。不得因此释放
  相关资源、允许复用/重试、publish/retire 或解除 quarantine。

该定向修复和故障 UT 结果不代表物理 driver fault 恢复，也不代表整 workspace
或所有 GPU 回归已通过。

## Process timeout 和资源边界

[`receive_idle`](../crates/ferrule-runtime/src/parallel/process/ipc.rs) 在 Boot 完成后，
等待下一条 command frame 时没有 first-byte/idle deadline。没有新命令、没有 expert
bucket 的 healthy child 不会因 300 秒闲置退出，也不发送 heartbeat 或 synthetic work。

- pipe 开始可读后，prefix/body 共享一个不滑动的 **absolute frame-I/O deadline**；
  部分 prefix 或持续缓慢收包不能不断续期。EOF/HUP 也会唤醒 idle wait。
- Boot 和 reply write 仍有限时。parent 独立限制 startup、整个 command/handler wait
  和 shutdown；这些 **absolute deadlines** 不因 frame progress 或 observation 延长。
- CLI 默认 `--rank-timeout-ms 30000`；child frame-I/O budget 是
  `max(rank_timeout_ms, 300000)`。它不是 idle lifetime。
- timeout/invalid reply/child loss 会失效 owner；TERM/KILL/reap 不是 CUDA fence。
  unreaped child 由 bounded reaper 持有，unknown custody 不因 exit/Drop 自动解除。
- `--rank-restarts` 当前只能为 `0`；没有 automatic restart/replay。thread mode
  只接受默认 rank timeout，也不能抢占 kernel 或卡住的 destructor。

## 可运行的 CLI serve 示例

从 workspace 根目录运行，以下是三种**替代启动方式**，不要同时占用端口 8000。
CUDA 示例针对两张可见 RTX 3090（`sm_86`）；其他 GPU 应使用匹配的架构。
模型和 tokenizer 必须已存在于 `/mnt/nas1/hf/Qwen3-0.6B`，不会自动下载。
process backend 要求 Unix；GPU process 的 PID/device 验收 tests 要求 Linux/nvidia-smi。

这些参数都来自 [`ServeArgs`](../crates/ferrule-cli/src/args.rs)：

- `--max-tensor-mb 1024` 容纳 311164928-byte tensor，默认 128 MiB 不够。
- `--ctx-size 64 --max-active-sequences 1 --kv-cache-mb 64` 是保守的完整模型 smoke
  容量，不是截层或性能配置；KV budget 必须覆盖所有 resident sessions 的完整 context。
- `--prefill-chunk-size 1 --max-batch-tokens 1` 限制单次返回的 full-vocabulary logits。
  serve process 使用默认 8 MiB frame / 4 MiB output JSON limits；不能照搬默认
  512-token prefill chunk。NAS process 数值测试另显式配置更大的 IPC limits，
  不代表 serve 自动拥有同样的 limits。

### GPU PP2，thread owners

```sh
FERRULE_CUDA_ARCH=sm_86 cargo run --locked --release -p ferrule-cli --features cuda -- \
  serve /mnt/nas1/hf/Qwen3-0.6B \
  --served-model-name qwen3-0.6b --backend cuda --engine pipeline \
  --pipeline-parallel 2 --rank-backend thread --devices 0,1 \
  --max-tensor-mb 1024 --ctx-size 64 --max-active-sequences 1 --kv-cache-mb 64 \
  --prefill-chunk-size 1 --max-batch-tokens 1 --host 127.0.0.1 --port 8000
```

### GPU PP2，process owners

```sh
FERRULE_CUDA_ARCH=sm_86 cargo run --locked --release -p ferrule-cli --features cuda -- \
  serve /mnt/nas1/hf/Qwen3-0.6B \
  --served-model-name qwen3-0.6b --backend cuda --engine pipeline \
  --pipeline-parallel 2 --rank-backend process --devices 0,1 \
  --rank-timeout-ms 120000 --rank-restarts 0 \
  --max-tensor-mb 1024 --ctx-size 64 --max-active-sequences 1 --kv-cache-mb 64 \
  --prefill-chunk-size 1 --max-batch-tokens 1 --host 127.0.0.1 --port 8000
```

startup/command/shutdown budget 此处设为 120 秒，以留出 checkpoint 冷加载余量；
不是 idle deadline，也不保证所有磁盘/硬件都能在此预算内完成。

### CPU PP2，thread owners

```sh
cargo run --locked --release -p ferrule-cli -- \
  serve /mnt/nas1/hf/Qwen3-0.6B \
  --served-model-name qwen3-0.6b --backend cpu --engine pipeline \
  --pipeline-parallel 2 --rank-backend thread \
  --max-tensor-mb 1024 --ctx-size 64 --max-active-sequences 1 --kv-cache-mb 64 \
  --prefill-chunk-size 1 --max-batch-tokens 1 --host 127.0.0.1 --port 8000
```

CPU process 使用同样参数，改为 `--rank-backend process` 并加
`--rank-timeout-ms 120000 --rank-restarts 0`，不要传 `--devices`。
CPU 非 EP 路径保留 BF16-compatibility policy；CUDA 和 EP injection 使用 F32。

启动完成后，在另一个终端执行：

```sh
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"hello"}],"max_completion_tokens":16,"stream":true}'
```

这是 serial greedy serving，不提供 sampling、packed/mixed decode、cohort deferral
或 prefix cache。请求中的非默认 `temperature/top_p/top_k` 与 `n > 1` 被拒绝；
没有 serve `--precision` 或 `--temperature` 参数。取消在 chunk/token 边界协作进行。
partial `--max-layers`、nonzero `--moe-hotset-experts` 和 expert-cache policy overrides
在 pipeline build 前拒绝。未传 `--expert-host-cache-*` / `--expert-pinned-cache-*`
时，pipeline 保留 runtime 默认策略，不把 resident CLI 默认值当作 override；显式传入
任意一个 cache 参数（包括 0 或恰好等于 runtime 默认值）仍会拒绝，不能承诺未实现的限额。
resident 未传时仍为 host 64 entries / 1024 MiB、pinned 16 entries / 256 MiB。
默认 `--engine auto` 不等于自动 CUDA，CUDA 必须显式选择。

Unix CLI 同时监听 SIGTERM 和 SIGINT：停止 HTTP admission，等待连接 drain，再等待
model worker shutdown/join（含 process owners 的 shutdown/reap）。非 Unix 使用 Ctrl-C。
HTTP 或 worker shutdown 失败会返回非零退出码；同时失败时也保留 worker shutdown 错误。

`--expert-parallel N` 只用于受支持的 Qwen3-MoE checkpoint，不能用于上面的 dense
Qwen3-0.6B。CUDA ordinal 顺序是 PP owners 在前，再按 stage 排列 EP owners；
PP2EP2 需要六个 ordinal，如 `0,1,2,3,4,5`。重复 ordinal 可显式 colocate owners，
但不会合并 owner identity。未提供经过本轮验收的完整 35B FP8 MoE serve 示例。

## Dense TP：CLI / serving 接入

`serve --tensor-parallel N`（默认 1）接入标准 dense Qwen3 的 **CUDA + thread**
PP×TP；N 当前仅支持 1、2、4。`--engine auto` 在 TP>1 时选择 pipeline，
但 **不会自动选择 CUDA**，必须显式传 `--backend cuda`。CPU TP、process TP、
MoE TP、EP×TP、DP/SP/CP>1 明确拒绝，不回退到 CPU、复制模型或本地专家。
TP1 保持已有 thread/process PP/EP 路径及其 device colocation 语义。

factory 直接调用 `PipelineParallelExecutor::new_standard_cuda_tensor`，入口实现位于
[`parallel/pipeline/tensor.rs`](../crates/ferrule-runtime/src/parallel/pipeline/tensor.rs)，
由 `pipeline` 导出 `StandardCudaTensorConfig`。每个 persistent physical owner
在 `load` 内加载 checkpoint bindings，并由正式 runtime/model API prepare 对应 shard。
没有新增 engine、forward 或 KV manager；parent 仍只有一个 logical page manager
和 transaction coordinator。PP stage 内 TP owners 并发执行 collective，stage 之间
仍按层顺序执行，不宣称 PP overlap 或性能提升。

### Device 与精度约束

- TP owner/device 数为 **PP × TP**，不是 PP+TP，也不是额外再加 PP leader。
- `--devices` 按 **PP-stage-major，然后 TP-rank** 排列：
  `owner = stage * TP + tensor_rank`。例如 PP2TP2 的 `3,1,2,0` 映射为
  `(stage0,tp0)->3`、`(stage0,tp1)->1`、`(stage1,tp0)->2`、`(stage1,tp1)->0`。
- 默认 device ordinals 为 `0..PP*TP`；TP 要求每个 physical owner 的 ordinal
  全局唯一，不能像 TP1 PP/EP 那样 colocate。数量、重复和 ordinal ABI 范围在
  planner 阶段拒绝；设备真实可用性仍由 owner 的 CUDA 初始化确认。
- Q heads 与 KV heads 必须能被 TP 整除，不支持 KV replication/head padding。
  planner 复用 model 的 strict recipe、`StandardTensorPlan` 和 segment validation，
  在创建 CUDA/owner 前校验层数、几何、上下文、容量与 collective byte arithmetic。
  checkpoint binding/storage 则在 build 的 metadata preflight 和 owner load 时验证。
- execution 固定 **F32 weights/activation/KV/logits**；BF16 checkpoint 存储格式
  不表示 BF16 compute。standard linear 为 CUTLASS TF32x3，F32 boundary、
  accumulation/output 不等于严格 IEEE SGEMM。没有新增 `--precision` 参数。
- TP 的 `--max-tensor-mb` 限制 **checkpoint 存储 bytes**，不是转换后的 F32 resident
  显存总量：projection 按各 rank 的真实 rectangle 校验（含 ragged 最大 rank），
  不再按 FULL projection bytes 拒绝，也不能用 `global / TP` 平均值放行。
  replicated embedding/norm 仍必须完整放入同一 limit；tied output head 单独校验
  projection rectangle，不能用 alias 绕过 embedding 的完整读取限额。
- TP parent/owners 使用 `Qwen3DenseAdapter::bind_hf_metadata`，与 resident 共用
  strict config/index/header/state-dict binding；adapter 的
  `validate_tensor_read_limits` 复用标准 TP parameter plan 与 checkpoint rectangle
  reader 的 metadata preflight。没有读取或解码全局 projection，也不把 dense limit
  扩大到 expert reader limit。missing/incorrect index、shape/dtype/name、完整 source
  extent 与 identity 校验保留；实际 owner materializer 再以原 limit 读取并复验 source。
  TP1/resident full-weight limit 和 CPU BF16-compatibility profile 不变；CUDA 的
  F32 tensor boundary/accumulation/output 保留，但 standard linear 已切换到
  CUTLASS TF32x3，不能再称 CUDA compute 数值契约完全不变。
  因此 NAS Qwen3-0.6B 的 replicated embedding 仍要求至少
  311164928 bytes，不能仅靠增加 TP 降到默认 128 MiB 以下。

`--kv-cache-mb` 是整个模型所有 physical owners 的 **KV data-plane 总预算**，
不是每卡预算，也不包括 weights、workspace 或 host collective buffers。
每个 owner 的 page bytes 为
`stage_layers * 2(K/V) * (global_kv_heads / TP) * head_dim * page_tokens * sizeof(f32)`。
所有 owners 求和等于 parent 全层/global-head logical page bytes；逻辑 page ID 在
所有 physical owners 上共享 identity，但不共享物理存储。页数覆盖全部 active
sessions 的完整 context（按 page 向上取整），不足则拒绝，不能靠 TP 虚增容量。
collective 使用 checked host-buffer limits 和 30 秒 rendezvous timeout；它不是
CUDA kernel 抢占或 fence，`--rank-timeout-ms` 的 thread 限制不变。

### 启动示例与验收边界

下面是两卡 PP1TP2 的接入示例（非性能配置）；PP2TP2 改为
`--pipeline-parallel 2 --tensor-parallel 2 --devices 0,1,2,3`，需要四张可见 GPU。

```sh
FERRULE_CUDA_ARCH=sm_86 cargo run --locked --release -p ferrule-cli --features cuda -- \
  serve /mnt/nas1/hf/Qwen3-0.6B \
  --served-model-name qwen3-0.6b --backend cuda --engine pipeline \
  --pipeline-parallel 1 --tensor-parallel 2 --rank-backend thread --devices 0,1 \
  --max-tensor-mb 1024 --ctx-size 64 --max-active-sequences 1 --kv-cache-mb 64 \
  --prefill-chunk-size 1 --max-batch-tokens 1 --host 127.0.0.1 --port 8000
```

本接入的 CLI parse→build-plan tests 不需要 GPU，覆盖 TP2/4、PP2TP2、device 顺序、
unsupported 组合、head divisibility、F32 KV 精确预算，以及保留的 cache 默认值修复。
新增 GPU HTTP tests 使用非零 attention/MLP 的两层 BF16 tiny checkpoint，经真实
factory/engine/worker/router，比较 GPU TP1 与 TP2 / TP4 / PP2TP2 的 non-streaming 和两个
SSE endpoint，覆盖断连取消、slot reuse、shutdown、sampling reject 和 KV accounting。
它们是 ignored tests，缺设备或错误必须失败，**不是完整 28 层 logits 数值验收**。
以下是更换 GEMM 前的历史执行记录，不代表当前 TF32x3 回归已通过：
曾使用 `FERRULE_CUDA_ARCH=sm_86` 实跑三个精确 ignored targets：TP2、TP4、PP2TP2
各 1/1 通过（不是 `--no-run`）；CLI serve tests 13/13 通过，包含 process TP 拒绝和
F32 KV 预算不足拒绝。tiny fixture 使用 4 个 KV heads，global page 为 2048 bytes，
4 pages 共 8192 bytes，三个拓扑的 accounting 一致。

同一历史验收还真实启动了 NAS Qwen3-0.6B 全层 CLI thread TP2（`CUDA_VISIBLE_DEVICES=0,1`，
上述容量参数，端口 18084），完成 non-streaming completion、completion SSE、chat SSE，
每个请求生成 2 tokens；两个 SSE 均返回 length、usage 和唯一 `[DONE]`。
受控启动验证脚本仅向本次 CLI PID 发 SIGTERM，等待退出码 0，并确认无 child processes、
PID 已 reap、监听端口关闭、`nvidia-smi --query-compute-apps` 为空；两卡显存从约
2053/2052 MiB 回落到 3/2 MiB。NVML 的 host PID 与本地 PID 不同，不能直接按本地
PID 过滤其 records；此前一次启动验证脚本的该断言失败，不是 serving inference 失败。
这些是正常 serving/shutdown 的实跑证据，不代表 driver fault 后的 CUDA fence 证明，
也不替代 runtime 的 28-layer all-logits 数值验收。

```sh
cargo test --locked -p ferrule-cli --features cuda --bin ferrule commands::serve::tests:: -- --test-threads=1
cargo test --locked -p ferrule-server --features cuda --test pipeline_http --no-run
# 至少两卡，TP2 vs GPU TP1
FERRULE_CUDA_ARCH=sm_86 timeout --signal=TERM --kill-after=20s 300s \
  cargo test --locked -p ferrule-server --features cuda --test pipeline_http \
  cuda_dense_tensor_tp2_matches_tp1_http_sse -- --ignored --exact --nocapture --test-threads=1
# 至少四卡，TP4 vs GPU TP1
FERRULE_CUDA_ARCH=sm_86 timeout --signal=TERM --kill-after=20s 300s \
  cargo test --locked -p ferrule-server --features cuda --test pipeline_http \
  cuda_dense_tensor_tp4_matches_tp1_http_sse -- --ignored --exact --nocapture --test-threads=1
# 至少四卡，PP2TP2 vs GPU TP1
FERRULE_CUDA_ARCH=sm_86 timeout --signal=TERM --kill-after=20s 300s \
  cargo test --locked -p ferrule-server --features cuda --test pipeline_http \
  cuda_dense_tensor_pp2tp2_matches_tp1_http_sse -- --ignored --exact --nocapture --test-threads=1
```

## 精确复现命令

以下命令从 workspace 根目录**串行**运行，均使用 `--locked`。`timeout` 是 GNU
coreutils 的外层运行预算，不是性能承诺；CUDA/CUTLASS 工具链、设备、NAS 文件必须
事先就绪。ignored tests 必须显式执行；缺设备/checkpoint、CUDA error 或数值 mismatch
会失败，不作 runtime skip。下列 child 路径使用默认 `target/debug`；若自定义
`CARGO_TARGET_DIR`，须将 child env 改为对应的实际可执行文件路径。

### CPU PP/EP/KV 和 P1 上层回归

```sh
cargo build --locked -p ferrule-runtime --example process_rank_child
cargo test --locked -p ferrule-runtime --test pipeline_decoder --test pipeline_program --test process_decoder -- --test-threads=1
cargo test --locked -p ferrule-model --lib decoder::kv::prepare_tests:: -- --test-threads=1
```

`pipeline_decoder` 中的 PP/EP helper 是
[`support/expert_decoder.rs`](../crates/ferrule-runtime/tests/support/expert_decoder.rs)。
`pipeline_program` 覆盖 unknown prepare 保留 journal/logical pages、拒绝 cleanup ACK，
并区分 validation/已证明 cleanup 的普通失败。

### F32 GPU decoder、thread PP/EP 和 P1 CUDA 注入

至少两张 GPU 运行 `full_gpu_decoder`；PP2EP2 target 要求六张 GPU。

```sh
FERRULE_CUDA_ARCH=sm_86 timeout --signal=TERM --kill-after=15s 180s cargo test --locked -p ferrule-model --features cuda --test full_gpu_decoder -- --ignored --nocapture --test-threads=1
FERRULE_CUDA_ARCH=sm_86 timeout --signal=TERM --kill-after=15s 240s cargo test --locked -p ferrule-runtime --features cuda --test pipeline_decoder cuda_pipeline:: -- --ignored --nocapture --test-threads=1
FERRULE_CUDA_ARCH=sm_86 timeout --signal=TERM --kill-after=15s 180s cargo test --locked -p ferrule-backend --features cuda --lib prepare_tests:: -- --ignored --nocapture --test-threads=1
```

### 完整 28-layer thread GPU decoder

至少两张 GPU，包含全层 CPU oracle：

```sh
FERRULE_CUDA_ARCH=sm_86 \
FERRULE_QWEN3_06B_PATH=/mnt/nas1/hf/Qwen3-0.6B \
timeout --signal=TERM --kill-after=20s 300s \
  cargo test --locked -p ferrule-runtime --features cuda \
  --test full_model_cuda -- --ignored --nocapture --test-threads=1
```

### 完整 28-layer process GPU decoder

先构建 production CLI child；`FERRULE_DECODER_CHILD` 指向 CLI，测试自动添加
`__rank-worker` 和其 frame/timeout 参数：

```sh
FERRULE_CUDA_ARCH=sm_86 cargo build --locked -p ferrule-cli --features cuda --bin ferrule
FERRULE_CUDA_ARCH=sm_86 \
FERRULE_QWEN3_06B_PATH=/mnt/nas1/hf/Qwen3-0.6B \
FERRULE_DECODER_CHILD=./target/debug/ferrule \
timeout --signal=TERM --kill-after=20s 300s \
  cargo test --locked -p ferrule-runtime --features cuda \
  --test process_decoder cuda::gpu_process_nas_qwen3_06b_all_28_layers_pp1_pp2_prefill_decode \
  -- --ignored --exact --nocapture --test-threads=1
```

### 六 GPU PP2EP2、healthy idle 和 child-loss process tests

复用上述 CUDA CLI child，Linux 下执行全部 `cuda::` ignored cases，包含 NAS case：

```sh
FERRULE_CUDA_ARCH=sm_86 \
FERRULE_QWEN3_06B_PATH=/mnt/nas1/hf/Qwen3-0.6B \
FERRULE_DECODER_CHILD=./target/debug/ferrule \
timeout --signal=TERM --kill-after=20s 900s \
  cargo test --locked -p ferrule-runtime --features cuda \
  --test process_decoder cuda:: -- --ignored --nocapture --test-threads=1
```

关键测试名包括：

- `cuda::gpu_process_pp2_ep2_independent_children_execute_every_route`
- `cuda::gpu_idle_and_oversize_keep_pp_and_bucketless_ep_ready_without_parent_cuda`
- `cuda::faults::gpu_process_pp_leader_loss_terminates_all_ep_descendants`
- `cuda::faults::gpu_process_expert_child_loss_propagates_unknown_and_reaps_children`
- `cuda::faults::gpu_process_expert_deadline_propagates_unknown_without_deadlock`

healthy-idle GPU case 等待 6 秒，超过其 2 秒 frame/command budget，再确认 PP/EP
仍是原 PID，没有制造 bucket/heartbeat execution。host loss tests 不等于物理 hang。
GPU process 验收会用 nvidia-smi 的物理 index/UUID 核对 child placement，运行这些
测试时不要任意重排 `CUDA_VISIBLE_DEVICES`。

### HTTP/SSE：CPU、GPU thread、GPU process

CPU tests 需要先构建 server 专用 helper。它使用 production `DecoderEndpoint`，但
额外写入 boot/shutdown evidence，所以 `FERRULE_PIPELINE_CHILD` **不能直接换成 CLI**。
`FERRULE_PIPELINE_CHILD_EVIDENCE` 由 test 自己设置，无需手工指定。

```sh
cargo build --locked -p ferrule-server --example pipeline_process_child
FERRULE_PIPELINE_CHILD=./target/debug/examples/pipeline_process_child \
  cargo test --locked -p ferrule-server --test pipeline_http -- --nocapture --test-threads=1
```

六 visible GPUs 的 thread HTTP/SSE target：

```sh
FERRULE_CUDA_ARCH=sm_86 timeout --signal=TERM --kill-after=20s 300s \
  cargo test --locked -p ferrule-server --features cuda \
  --test pipeline_http cuda_dense_and_moe_thread_pp_ep_serve_real_http \
  -- --ignored --exact --nocapture --test-threads=1
```

process HTTP/SSE 先用 CUDA feature 重建专用 helper；下面显式隐藏 parent 的 GPU，
再通过 test 读取的 `FERRULE_PIPELINE_CHILD_CUDA_VISIBLE_DEVICES` 只给 children 暴露六卡：

```sh
FERRULE_CUDA_ARCH=sm_86 cargo build --locked -p ferrule-server --features cuda --example pipeline_process_child
CUDA_VISIBLE_DEVICES= \
FERRULE_CUDA_ARCH=sm_86 \
FERRULE_PIPELINE_CHILD=./target/debug/examples/pipeline_process_child \
FERRULE_PIPELINE_CHILD_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
timeout --signal=TERM --kill-after=20s 300s \
  cargo test --locked -p ferrule-server --features cuda \
  --test pipeline_http cuda_process_pp_ep_serve_real_http \
  -- --ignored --exact --nocapture --test-threads=1
```

## 协议与 transport 边界

[`DistributedTransaction`](../crates/ferrule-runtime/src/distributed.rs) 仍是唯一
transaction decision/publication authority。`prepare` 是 admission；compute/drain 后
各 participant 的 `prepare_vote` 是 readiness，不是 decision 后 `finalize` ACK。
Commit/Abort 都需要 drain 和成功 cleanup/install/retirement ACK，不能用 pre-vote、
取消返回成功或 owner exit 代替。unknown custody 不得 publish、retire 或解除 quarantine。

KV 的 `PreparedKvCommit` 封存 exact participants、mapping/generation、COW source、
shape/capacity/fence 和 physical handles；install 只 dispatch 一次，pending/lost ACK
走 poll，不重发 install。physical install/owner finish 控制 logical publish，page reuse
还需要 owner retirement ACK 和 parent `confirm_page_retirement`。

common mesh 为 `DP * PP * TP`，EP 使用显式 per-stage `ExpertGroup`，不乘入 PP/KV mesh。
pipeline serving 接 PP/EP，另通过上述正式入口接 dense CUDA thread PP×TP；
DP/SP/CP 必须为 1，EP×TP 和 process TP 不支持。generic DP owner pool 不代表
full-model DP serving。

[`CudaAsyncTransport`](../crates/ferrule-backend/src/cuda/transport.rs) 是 bounded pinned
H2D/D2H API，model shard transfer 默认每 chunk 64 KiB、TX/RX 各一个 slot，共 128 KiB。
activation/logits 边界使用 host staging；process 使用 bounded serialized host pipes。
HostCollective 仍是同步 CPU rendezvous，不是端到端异步 collective，也不宣称 PP overlap。

## 明确不作的宣称

- 完整 35B FP8 checkpoint/decoder、完整 MoE checkpoint GPU EP；
- GPU BF16 compute；
- full-model GPU DP serving、MoE/process TP，或未经上述 GPU HTTP targets 实跑验证的 TP serving 结果；
- CUDA IPC、无 host staging/transport 的零 host 路径；
- multi-host / multi-node；
- 物理 CUDA driver fault、D-state 或 GPU engine hang 后的强制恢复；
- child loss 后 automatic restart/replay，或未经独立验证的性能/speedup。

`NoParentCuda` 证明的是 parent 未初始化 CUDA，不证明 child termination 后物理设备
已停止执行。普通 host completion、pipe EOF、timeout、SIGTERM/KILL、Rust unwind/Drop
都不是 CUDA fence；unknown/quarantine 正是为了不作这个安全推断。
