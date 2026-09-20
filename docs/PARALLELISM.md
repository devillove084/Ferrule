# 并行执行集成状态

本文区分当前源码的集成能力、本轮已完成的实跑证据与尚未覆盖的边界。下面的
验收结果汇总本轮执行记录；文档收尾本身不重复 GPU 测试，也不把小 checkpoint
fixture 的结果外推为更大的模型、更多节点或更强的故障恢复能力。
Ferrule production runtime 不依赖、链接或调用 NCCL；外部 Python/NCCL 对比工具
不属于 Ferrule transport。

## 四项已集成能力

| 能力 | 当前集成边界 | 主要入口 |
| --- | --- | --- |
| 完整 F32 GPU decoder | 标准 dense/MoE decoder 支持 BF16/F32 checkpoint weights，物化为 resident F32 weights，以 F32 activation/KV/logits 执行；真实 dense case 覆盖完整 28 层 | [`full_gpu_decoder`](../crates/ferrule-model/tests/full_gpu_decoder.rs)、[`full_model_cuda`](../crates/ferrule-runtime/tests/full_model_cuda.rs) |
| GPU PP/EP | persistent GPU pipeline/expert owners、sealed KV、fork/COW、取消和提交/回收已接入；覆盖 PP2、PP2×EP2 | [`cuda_pipeline`](../crates/ferrule-runtime/tests/support/cuda_pipeline.rs) |
| 跨进程 PP/EP/KV | production process endpoint 承载 PP/EP child、KV transaction/ACK、bounded host IPC、quarantine/reap；覆盖真实 28-layer dense 和六独立 GPU children 的 MoE fixture | [`process_decoder`](../crates/ferrule-runtime/tests/process_decoder.rs)、[`process_decoder/cuda`](../crates/ferrule-runtime/tests/process_decoder/cuda.rs) |
| `PipelineInferenceEngine → ModelWorker → HTTP/SSE` | model factory 构建 pipeline engine，由 dedicated model worker 驱动 OpenAI-compatible HTTP/SSE；CPU/GPU thread/process 路径均已测试 | [`pipeline.rs`](../crates/ferrule-runtime/src/engine/pipeline.rs)、[`worker.rs`](../crates/ferrule-server/src/worker.rs)、[`pipeline_http`](../crates/ferrule-server/tests/pipeline_http.rs) |

### 数值和故障证据

- **真实 NAS Qwen3-0.6B，完整 28 层**：hidden=1024、vocab=151936、311 个绑定
  参数（含 tied alias），checkpoint weights 全为 BF16。PP1 为 `0..28`；PP2 为
  `0..14 / 14..28`，没有截层。每 tensor 上限 1 GiB，最大 tensor 为 311164928 bytes。
- **GPU thread PP1 vs PP2**：3-token prefill、3 次 greedy decode、取消/重试、完整
  前缀 replay 以及 KV cleanup；全部 vocabulary logits 逐元素比较，PP1/PP2 和
  decode/causal replay 的 `maxdiff = 0`。PP1/PP2 tolerance 是
  `2e-5 + 2e-5 * abs(expected)`，并非只检查 argmax。
- **全 28-layer CPU F32 reference**：三个 prefill 行的最大绝对差分别为
  `1.18255615e-4`、`7.66515732e-5`、`5.22136688e-5`，decode 为
  `4.43458557e-5`；最大约 `1.18e-4`，greedy argmax 一致。
  CPU/replay tolerance 为 `2e-3 + 2e-4 * abs(expected)`。
- **GPU process PP1 vs PP2**：同一 NAS 28-layer 模型的 3-token prefill 和一次
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
在 pipeline build 前拒绝。默认 `--engine auto` 不等于自动 CUDA，CUDA 必须显式选择。

`--expert-parallel N` 只用于受支持的 Qwen3-MoE checkpoint，不能用于上面的 dense
Qwen3-0.6B。CUDA ordinal 顺序是 PP owners 在前，再按 stage 排列 EP owners；
PP2EP2 需要六个 ordinal，如 `0,1,2,3,4,5`。重复 ordinal 可显式 colocate owners，
但不会合并 owner identity。未提供经过本轮验收的完整 35B FP8 MoE serve 示例。

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
pipeline serving 只接 PP/EP，DP/TP/SP/CP 必须为 1。generic DP owner pool 和 TP
linear/SwiGLU shard capability 仍独立存在，不代表 TP-sharded 完整 decoder。

[`CudaAsyncTransport`](../crates/ferrule-backend/src/cuda/transport.rs) 是 bounded pinned
H2D/D2H API，model shard transfer 默认每 chunk 64 KiB、TX/RX 各一个 slot，共 128 KiB。
activation/logits 边界使用 host staging；process 使用 bounded serialized host pipes。
HostCollective 仍是同步 CPU rendezvous，不是端到端异步 collective，也不宣称 PP overlap。

## 明确不作的宣称

- 完整 35B FP8 checkpoint/decoder、完整 MoE checkpoint GPU EP；
- GPU BF16 compute；
- TP-sharded 完整 decoder 或 full-model GPU TP/DP serving；
- CUDA IPC、无 host staging/transport 的零 host 路径；
- multi-host / multi-node；
- 物理 CUDA driver fault、D-state 或 GPU engine hang 后的强制恢复；
- child loss 后 automatic restart/replay，或未经独立验证的性能/speedup。

`NoParentCuda` 证明的是 parent 未初始化 CUDA，不证明 child termination 后物理设备
已停止执行。普通 host completion、pipe EOF、timeout、SIGTERM/KILL、Rust unwind/Drop
都不是 CUDA fence；unknown/quarantine 正是为了不作这个安全推断。
