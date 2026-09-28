# Qwen3.5：0.8B dense 与 35B-A3B FP8 支持

前半部分描述 **dense Qwen3.5-0.8B、完整 24 层、纯文本**路径；文末的
**35B-A3B FP8 runtime 接线与实际验收**记录 F32/TF32x3 单 GPU resident 与
专用 GPU-thread EP2/4/8。generic pipeline 的 TP/PP/process 仍不支持；EP8 的
strict 数值/lifecycle proof 与 EP8 release 性能验证均已完成：这些实测场景已交互可用，
不外推为多用户容量、数千 token context 或线性八倍扩展性。
两者 precision、默认 backend 和验收证据分别报告，
不把通用 Qwen3 的 TP/PP/EP 能力外推到 Qwen3.5。并行集成总览见
[PARALLELISM.md](PARALLELISM.md)。

真实 `/mnt/nas1/hf/Qwen3.5-0.8B` 的完整 CPU 和单卡 CUDA 数值验收已通过；
最终 CPU / GPU 报告及原始日志已核查：`hybrid_cuda` **13 passed** 中含 **6 个实际
CUDA 测试、1 个比较器验证、6 个 CPU reference 测试**；独立 forward CUDA UT 另
**1 passed**。GPU 验证批次共 **186 selected passed、0 failed**，不是 186 项全 GPU
测试。这里是先前 0.8B 验证记录；35B 既有 Cargo/CLI 证据另见后文；本次 docs-only 不运行 Cargo。
当前源码已接通 CLI → model factory → resident engine → dedicated model worker →
HTTP/SSE。本地 CLI/HTTP 日志也记录了真实 CUDA serving 成功，不能再把 CUDA
描述为仅 metadata 支持或待接入。下面分别列出实现边界、证据和复现命令。

## 0.8B 支持矩阵与通用拒绝边界

| 项目 | 当前范围 | 源码依据 |
| --- | --- | --- |
| checkpoint | HF safetensors，nested `qwen3_5` / `qwen3_5_text`，严格 0.8B 几何；24 层中 18 层 GatedDeltaNet、6 层 full GQA | [`config.rs`](../crates/ferrule-model/src/models/qwen35/config.rs)、[`recipe.rs`](../crates/ferrule-model/src/models/qwen35/recipe.rs) |
| CPU | F32 text decoder，默认 backend 选择 CPU | [`model_factory/qwen35.rs`](../crates/ferrule-runtime/src/engine/model_factory/qwen35.rs) |
| CUDA | `cuda` feature 下的单设备 resident F32 tensor 路径；显式 `--backend cuda`，当前 factory 使用可见 ordinal 0 | 同上；[`hybrid_cuda.rs`](../crates/ferrule-model/src/decoder/hybrid_cuda.rs) |
| 文本 serving | completion/chat、greedy generation、HTTP/SSE、stop/EOS、取消、reset、shutdown；使用 Qwen35 chat template | [`serve.rs`](../crates/ferrule-cli/src/commands/serve.rs)、[`qwen35_http.rs`](../crates/ferrule-server/tests/qwen35_http.rs)、[`qwen35_chat.rs`](../crates/ferrule-model/tests/qwen35_chat.rs) |
| vision / MTP | **不执行**；已知附件可缺席，若出现则必须是完整且合法的 partition，校验后排除出 text state dict | [`name_mapper.rs`](../crates/ferrule-model/src/models/qwen35/name_mapper.rs)、[`adapter.rs`](../crates/ferrule-model/src/models/qwen35/adapter.rs) |
| MoE / 35B / FP8 | 精确 35B-A3B FP8 profile 已有独立 CUDA F32/TF32x3 factory；详见文末。不是 0.8B profile 的量化开关；其他尺寸、BF16 packed experts 及 flat text export 仍拒绝 | [`config.rs`](../crates/ferrule-model/src/models/qwen35/config.rs)、[`qwen35_metadata.rs`](../crates/ferrule-model/tests/qwen35_metadata.rs) |
| generic pipeline TP / PP / process ranks | **unsupported**；不能把 Qwen3 的 generic pipeline 拓扑套到 Qwen35 | [`model_factory/pipeline.rs`](../crates/ferrule-runtime/src/engine/model_factory/pipeline.rs)、[`standard/cuda/hybrid.rs`](../crates/ferrule-model/src/transformer/standard/cuda/hybrid.rs) |
| dedicated GPU-thread EP | **supported only for exact 35B-A3B FP8, EP2/4/8**；resident root + expert owners，显式 distinct `--devices`，`devices[0]` 为 root，owners 使用同一 ordered list | [`model_factory/qwen35.rs`](../crates/ferrule-runtime/src/engine/model_factory/qwen35.rs)、[`model_factory/qwen35_ep.rs`](../crates/ferrule-runtime/src/engine/model_factory/qwen35_ep.rs) |
| prefix cache / partial retain / speculation | **unsupported**；factory 拒绝非零 prefix cache 与 native proposals，CUDA 仅允许 exact-frontier fork，不能部分保留 recurrent history | [`model_factory/qwen35.rs`](../crates/ferrule-runtime/src/engine/model_factory/qwen35.rs)、[`hybrid_cuda.rs`](../crates/ferrule-model/src/decoder/hybrid_cuda.rs) |
| 截层 / expert cache options | partial `--max-layers` 被拒绝；35B 的 `--expert-host-cache-*` 是独立 pageable compressed prewarm budget；旧 pinned/hotset/device flags 仍不冒充 host residency | [`serve.rs`](../crates/ferrule-cli/src/commands/serve.rs)、[`host_experts`](../crates/ferrule-model/src/transformer/host_experts/mod.rs) |

严格校验包括 tensor 名称、dtype、shape、bytes、index/header 对应关系及附件完整性，
不是按 `model.visual.*` / `mtp.*` 前缀无条件忽略。真实 checkpoint 的分区为
320 个 text tensors、153 个 visual tensors、15 个 MTP tensors；text binding 加上
共享 embedding 的 tied output alias 后为 321 项。附件校验不是视觉或 MTP 推理验收。

`Qwen35Adapter::load_hf_with_backend` 这个便捷 API 仍只创建 CPU runner；CUDA factory
使用 `bind_hf_metadata` 和 `GenericDecoderRunner<HybridCudaDecoder>`。不能把该便捷
API 的 CUDA 拒绝，或其旧模块说明，误读为 CLI/factory 尚不支持 CUDA。

## 数值与执行路径

- 0.8B profile 的 checkpoint 是 **BF16/F32 混合存储**，不是任意 dtype 都可接受：
  大部分 weights 为 BF16，GDN 的 `A_log` 和 `norm.weight` 为 F32，逐项服从 schema。
  物化后 weights、activation、KV、recurrent state 和 logits 使用 F32 tensors；
  不声称 BF16 compute，也不通过 BF16 round trip 截断这条 F32 路径。
- projection 复用标准 `linear_f32_into` 和现有 CUTLASS provider 的 F32 扩展。
  [`cutlass_f32.cuh`](../crates/ferrule-backend/native/cuda/core/cutlass_f32.cuh) 使用
  `OpMultiplyAddFastF32`，即 **TF32x3 TensorOp GEMM、F32 boundary/accumulation/output**。
  这不是严格 IEEE F32 SGEMM 或 bitwise 等价保证，也不是 SIMT SGEMM；F32 precision
  enum 不代表上述严格算术契约，本次不改 enum 或新增 CLI knob。不支持的 capability/shape
  返回错误，不悄悄切换到另一计算路径。
- GatedDeltaNet 使用专用 CUDA convolution、delta recurrence、gate/norm 等 kernels，
  经同一套 `CudaStandardDecoderOperators` / `CudaOperators` 接入共享 hybrid decoder；
  不是独立的 Qwen35 forward 或 Python bridge。相关入口是
  [`standard/cuda/recurrent.rs`](../crates/ferrule-model/src/transformer/standard/cuda/recurrent.rs)
  和 [`cuda/recurrent.rs`](../crates/ferrule-backend/src/cuda/recurrent.rs)。
- CUDA recurrent state 没有 CPU fallback。CPU backend 是显式可选的独立执行路径，
  不是 CUDA 出错时的补救路径。Ferrule production runtime 不依赖、链接或调用 NCCL，
  也不调用 Python；离线 oracle 和临时启动验证脚本不属于生产执行类型。
- 全模型数值测试逐元素比较完整 vocabulary，而非只看 top-1；当前 oracle tolerance
  为 `abs(actual - reference) <= 2e-4 + 2e-4 * abs(reference)`，另检查 greedy token。
  这是一组指定输入上的验收标准，不是所有模型、序列或硬件的误差保证。

### sm_86 capability 与 graph replay 验收

当前 [`cutlass_f32.rs`](../crates/ferrule-backend/tests/cutlass_f32.rs) 与
[manifest 日志](../target/validation/qwen35-35b-final-gpu-15-current-f32-manifest.log)
核对 canonical manifest、native/Rust discovery、plan/execution；旧 0.8B 批次的
`0x586` 不是新增 BF16 GEMM 后的当前 mask。
`F32Gemm ID=11`、native/FFI capability mask **`0xD86`**，F32 TF32x3 / LinearF32
inference 为 true，FP8 QueryAKv/Projection 仍为 false；因此不能借新增 F32 GEMM
宣称 RTX 3090 `sm_86` 支持 native FP8 MMA。最终 GPU target 的旧 metadata assertion
曾错误期待 `0x586`（少了 BF16 GEMM bit `0x800`）；该 stale-mask 断言不能伪装成
数值或模型失败。最新验证交接确认修正后的 metadata-filtered 检查已通过；原始失败
日志仍保留，不改写成一次无失败的全量重跑。
`standard_linear_f32_graph_capture_replay` 通过：保持地址不变，逐轮写入新输入，
用 NaN / -777 污染 D 后 replay，D 和 device consumer 都得到新输入对应的正确结果。
这不是只检查 capture 成功或复用旧输出，也不构成严格 IEEE SGEMM 保证。

## P1 forward 失败与 quiescence

最终 [exact UT 日志](../target/validation/qwen35-final-gpu-03-forward-unknown-exact.log)
记录 `forward` 失败进入 `FailedActive` 的 quiescence 修复验收 **1 passed**：
`transformer::forward::tests::cuda::hybrid_unknown_finish_keeps_transaction_pins_despite_independent_kv_fence`。
第一次错误过滤得到的 0 tests 不计通过；这不是本次文档更新重新执行的结果。

- 确认 forward quiescent、回滚完成且无 unknown custody 的**干净失败可重试**。
- forward quiescence **unknown 不能被后续成功的 KV fence 清除**。只完成 KV work
  不等于先前 forward work 已完成；unknown 状态下不释放相关资源，不允许复用/重试、
  publish/retire 或解除 quarantine。

这个定向故障验证不证明物理 driver fault 后可恢复，也不等于完整 workspace 验收。

## GPU 状态预算

CUDA admission 分开核算 weights、workspace、KV 与 recurrent state，不能只为 KV
预留显存。实现和真实 header 预算测试见
[`model_factory/qwen35.rs`](../crates/ferrule-runtime/src/engine/model_factory/qwen35.rs)。

- **每个 sequence state 为 20,643,840 bytes**，包括全部 18 个 GDN 层的
  convolution history 和 recurrent matrix。transaction working copy 同样占一份。
- recurrent state 上限为
  `(2 * max_active_sequences + 1) * 20,643,840`：live/committed states、working
  copies 加 default state。默认 4 sequences 共 9 份，即 **185,794,560 bytes**。
  forks 消耗同一硬上限；它不是只按当前活跃请求数估算的可无限增长缓存。
- KV 仅为 **6 个 full-attention 层 compact 分配**，不是给全部 24 层分配 full KV。
  F32、2 KV heads、head dim 256、page size 16 时，每页为
  `6 * 2(K/V) * 2 * 256 * 16 * 4 = 393,216 bytes`。
- 默认 `ctx-size=1024`、4 sequences 的 full capacity 为 256 pages；transaction
  shadows 也计入预算，当前配置 512 physical pages，共 **201,326,592 bytes（192 MiB）**。
  `--kv-cache-mb 1024` 是 KV 上限，不意味着实际分配 1 GiB，更不是全部 GPU 内存上限。
- 这一默认配置的 weight upper bound 为 **8,054,954,496 bytes**，workspace upper bound
  为 **1,018,167,296 bytes**。它们是 checked admission 上界，不是测得的峰值显存。
  runtime/driver 等额外开销也不能从这些数字中推导为零。

CLI 为 Qwen35 自动将未指定的 `--max-tensor-mb` 设为 1024，关闭 native proposals，
并保持 prefix cache 关闭。token-serial execution 将 `max_batch_tokens` 限为
`min(请求值, 32, ctx_size)`，prefill chunk 再受此限制；默认的 512 不是一次执行
512 行 full-vocabulary logits 的承诺。这些是现有 factory 行为，不需要另外改 capability。

## 最小 serve 复现

以下命令从 `/root/Ferrule` 运行。模型、tokenizer 和 CUDA/CUTLASS 构建环境必须已在
本机；不会下载模型。CUDA 例子按 RTX 3090 `sm_86`，其他受支持 GPU 应匹配架构，
不能据此宣称已验收。两种启动方式是替代关系，不要同时占用端口 8000。

```sh
# 默认 backend 是 CPU，不需要 cuda feature。
cargo run --locked -p ferrule-cli -- \
  serve /mnt/nas1/hf/Qwen3.5-0.8B

# 单卡 CUDA；无需额外传 max-tensor、截层或并行参数。
FERRULE_CUDA_ARCH=sm_86 cargo run --locked -p ferrule-cli --features cuda -- \
  serve /mnt/nas1/hf/Qwen3.5-0.8B --backend cuda
```

这就是已成功运行的典型 `serve /mnt/nas1/hf/Qwen3.5-0.8B --backend cuda` 路径；
CLI 集成测试只额外选择空闲端口。默认公开 model ID 为 `qwen3.5-0.8b`。
在另一个终端执行：

```sh
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
curl --max-time 120 -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"Hi"}],"max_completion_tokens":1,"stream":true,"stream_options":{"include_usage":true}}'
curl --max-time 120 -N http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3.5-0.8b","prompt":"The capital of France is","max_tokens":3,"stream":true,"stream_options":{"include_usage":true}}'
```

真实默认 CUDA CLI 的验收记录为 **6.77 s ready**，SSE 分别输出 `Hello` 和
` Paris`、`.`、换行，包含 length/usage 和 `[DONE]`；SIGTERM 后退出码为 0，
退出后无残留 GPU process。ready 时间也是本次实测，不是启动时延保证。
前台启动验证脚本只负责启动、请求、计时和回收 CLI，
不能称作另一个 production engine。正式 CLI 回归入口是
[`tests/serve.rs`](../crates/ferrule-cli/tests/serve.rs)。

**历史 0.8B debug 观测**：该批 CLI 记录的后续 decode token 间隔约 14.85/14.92 秒，
即约 **15 s/token**。这是指定机器、输入和构建下的观测，不是性能承诺、吞吐 benchmark、
CPU 耗时估计或 release 加速保证；prefill/首 token 等待也不能由此等同推算。

## 离线 Transformers 5.2 oracle

[`scripts/qwen35_reference.py`](../scripts/qwen35_reference.py) 是 0.8B 专用、仅供验证的离线
CPU F32 oracle：native Transformers eager attention / torch GDN fallback，
`local_files_only=True`、`trust_remote_code=False`，禁止 FLA/causal-conv1d 替代 kernels，
不初始化 CUDA。现有 manifest 记录 Transformers **5.2.0**；脚本记录版本与实现 hash，
但不自动安装或钉住版本，不能把其他版本结果无条件当成同一 oracle。

脚本导出 manifest、输入/token IDs、逐层 hidden、logits、自检结果和可选 cache snapshots，
可另导出固定种子的 tiny F32 checkpoint。tiny fixture 只是测试数据，不是第二个受支持的
production Qwen35 profile。Python 环境只需用于生成 oracle，不是 Rust serving 依赖。

**HF 多 token cache 陷阱**：本地 TF 5.2 的 GDN 在带 cache 的调用中，仅当本次
`seq_len == 1` 才使用已有 convolution/recurrent state；后续多 token chunk 会重置
linear state（`initial_state=None`）。因此脚本只用多 token **首块**加单 token 后续块，
另做无 cache full-prefix recomputation 和 deepcopy-cache replay。不要修改 native
forward 来伪造 oracle，也不要把这份 artifact 作为 HF 多 token cached continuation
正确性的证据。cache replay 的 bitwise 相同仅指 HF 自身恢复快照后的 replay，
不是 Ferrule CUDA 与 HF bitwise 相同。

```sh
# target/validation 必须先存在；只在新的输出目录生成，脚本拒绝覆盖。
mkdir -p target/validation
/opt/conda310/bin/python -B scripts/qwen35_reference.py \
  --model /mnt/nas1/hf/Qwen3.5-0.8B \
  --output target/validation/qwen35-reference-cpu-f32 \
  --threads 8 --timeout 300 --cache-states --tiny-fixture
```

上面的 Python 路径是本机已有的 oracle 环境。若输出目录已存在，保留已完成的 artifact，
不要删除后重跑；需要独立复现时可省略 `--output`，脚本会打印新的带时间戳目录。
确认 manifest 的 `status=complete`、版本、hash 和 self-checks 后才作为 oracle。
CPU 全模型测试可用 `FERRULE_QWEN35_ORACLE_DIR` 指向新目录；**当前 GPU 全模型测试
硬编码** `/mnt/nas1/hf/Qwen3.5-0.8B` 和
`target/validation/qwen35-reference-cpu-f32`，不会读取这两个路径的环境变量 override。
脚本在其临时输出目录生成的说明文件不属于项目文档或生产入口。

## 精确验证命令与证据范围

以下可交给负责最终集成验收的 agent 顺序执行。除明确列出的本次文档核查外，
命令清单不表示本次全部重跑；`--no-run`、0 tests 或 ignored 都不能写作通过。
CUDA targets 使用真实 GPU，不要并行抢占设备；构建 feature 不等于已执行 CUDA kernel。

### 无 GPU 的定向回归

```sh
cargo test --locked -p ferrule-model \
  --test qwen35_metadata --test qwen35_validation --test qwen35_chat \
  --test qwen35_cpu --test hybrid_cpu
cargo test --locked -p ferrule-runtime --lib qwen35 -- --test-threads=1
cargo test --locked -p ferrule-server --test qwen35_http -- --test-threads=1
cargo test --locked -p ferrule-cli --bin ferrule \
  commands::serve::tests::qwen35_default_cli_is_usable_and_explicit_moe_options_are_rejected \
  -- --exact --test-threads=1
```

### 真实 metadata 和默认预算（后者编译 CUDA，但不初始化设备）

```sh
FERRULE_QWEN35_08B_DIR=/mnt/nas1/hf/Qwen3.5-0.8B \
  cargo test --locked -p ferrule-model --test qwen35_metadata \
  real_08b_headers_validate_text_and_known_attachment_partitions \
  -- --ignored --exact --nocapture --test-threads=1
FERRULE_CUDA_ARCH=sm_86 FERRULE_QWEN35_08B_DIR=/mnt/nas1/hf/Qwen3.5-0.8B \
  cargo test --locked -p ferrule-runtime --features cuda --lib \
  engine::model_factory::qwen35::tests::qwen35_real_header_budget_uses_compact_kv_and_all_recurrent_layers \
  -- --ignored --exact --nocapture --test-threads=1
```

### CUDA operators 与共享 hybrid lifecycle

```sh
FERRULE_CUDA_ARCH=sm_86 cargo test --locked -p ferrule-backend --features cuda \
  --test cutlass_f32 --test cuda_standard_linear --test cuda_recurrent \
  --test cuda_recurrent_validation -- --include-ignored --nocapture --test-threads=1
FERRULE_CUDA_ARCH=sm_86 cargo test --locked -p ferrule-model --features cuda \
  --test hybrid_cuda gpu_hybrid_ -- --ignored --nocapture --test-threads=1
```

这些 targets 覆盖 GEMM、GDN kernel 数值、owner/shape/alias 校验和共享 hybrid 的
prefill/decode、事务 abort/fork/reset、ragged cohort 隔离等；tiny operator/lifecycle
结果不能替代下面真实 24 层验收。最终串行报告已记录这些选定 targets 的通过结果，
文档收尾只核查日志，没有重新执行。`gpu_hybrid_` 过滤只运行匹配的子集，
不能把它当成下面完整 `hybrid_cuda` 13 项的计数。

### 真实完整 24 层数值验收（先准备 oracle）

```sh
FERRULE_QWEN35_08B_DIR=/mnt/nas1/hf/Qwen3.5-0.8B \
FERRULE_QWEN35_ORACLE_DIR=/root/Ferrule/target/validation/qwen35-reference-cpu-f32 \
  cargo test --locked -p ferrule-model --test qwen35_cpu \
  nas_08b_native_prefill_and_two_decodes_match_all_oracle_logits \
  -- --ignored --exact --nocapture --test-threads=1
FERRULE_CUDA_ARCH=sm_86 cargo test --locked -p ferrule-model --features cuda \
  --test hybrid_cuda qwen35_08b_full_24_layer_gpu_matches_local_tf_reference \
  -- --ignored --exact --nocapture --test-threads=1
```

CPU target 比较 `capital` 和 `hello` 的 prefill 所有行及各两次 decode 的完整 logits。
最终 [hybrid 日志](../target/validation/qwen35-final-gpu-03-hybrid-cuda.log)记录
`hybrid_cuda` **13 passed**（6 CUDA + 1 比较器 + 6 CPU reference），不是 13 个 GPU
测试；另有独立 forward CUDA UT 1 项，因此 hybrid 组实际 CUDA 测试共 7 项。
其中真实 Qwen35 完整 24 层 target 的当前 GPU 结果如下：

| case | stage | logits 数量 | 最大绝对误差 |
| --- | --- | ---: | ---: |
| capital | prefill，全部 5 行 | 1,241,600 | `2.336502075e-5` |
| capital | decode.0 | 248,320 | `1.096725464e-5` |
| capital | decode.1 | 248,320 | `1.168251038e-5` |
| hello | prefill，全部 1 行 | 248,320 | `3.337860107e-5` |
| hello | decode.0 | 248,320 | `2.408027649e-5` |
| hello | decode.1 | 248,320 | `2.765655518e-5` |

合计 **2,483,200 logits**，manifest/hash 完整性校验通过，未改变原有
`atol=2e-4 / rtol=2e-4`。每行 argmax 和生成预测一致；capital 的预测为
`[11751, 13, 198]`、最终 position=7；hello 为 `[11, 271, 40]`、position=3。
这些是本轮更换 GEMM 后的日志实测，不是 bitwise 保证。

CPU 全模型 target 也已单独重跑 **1 passed**，见
[CPU NAS logits 日志](../target/validation/qwen35-final-cpu-nas-logits.log)：
同样比较全部 2,483,200 logits，最大绝对误差 `1.010820270e-4`，全部满足原容差且
预测一致。**此 CPU 数字不能与上表 GPU 误差混用**。
标准 Qwen3 的八卡 PP2TP4、28 层 GPU/CPU 及 process 新数据单列于
[PARALLELISM.md](PARALLELISM.md)，不外推为 Qwen35 TP/PP、35B 或全尺寸支持。

### 真实 factory HTTP/SSE 与默认 CLI

```sh
FERRULE_CUDA_ARCH=sm_86 FERRULE_QWEN35_08B_DIR=/mnt/nas1/hf/Qwen3.5-0.8B \
  cargo test --locked -p ferrule-server --features cuda --test qwen35_http \
  qwen35_cuda_http_token_stop_abort_reset_shutdown \
  -- --ignored --exact --nocapture --test-threads=1
FERRULE_CUDA_ARCH=sm_86 FERRULE_QWEN35_08B_DIR=/mnt/nas1/hf/Qwen3.5-0.8B \
  cargo test --locked -p ferrule-cli --features cuda --test serve \
  qwen35_cuda_default_serve_sse_and_sigterm \
  -- --ignored --exact --nocapture --test-threads=1
```

HTTP lifecycle target 使用自己的有界容量选项；只有 CLI target 验证默认容量启动，
二者不能混为同一种验收。CLI 信号测试要求 Unix。

### 先前 0.8B / 标准 Qwen3 验证记录与来源

[最终 GPU 汇总](../target/validation/qwen35-final-gpu-summary.md)、
[最终 CPU 汇总](../target/validation/qwen35-final-cpu-summary.md)及以下日志都是
**Git ignored 的本地生成证据**，不随仓库发布。链接仅在保留该次 artifacts 的
工作区有效；本次文档同步用 `rg` 核查它们，没有重跑 Cargo 或生成 oracle。

以下是先前批次的记录，不覆盖后续 35B 变更；历史 postfix 统计见文末。

- GPU 批次在 8 张 RTX 3090 上串行执行，`CARGO_INCREMENTAL=0`、
  `FERRULE_CUDA_ARCH=sm_86`、`--locked`、`--test-threads=1`，单命令 timeout 900 秒。
  CUDA workspace all-targets check、CLI / process child 构建通过。
- 合计 **186 selected passed、0 failed**，不是 186 项实际 GPU 测试。
  backend lib 包含 **113 passed / 18 ignored**；其中 KV 10 项随后定向通过，
  不代表全 ignored 已覆盖。hybrid 13 项和另行执行的 forward UT 按上面的构成计数，
  不将 0 tests、编译成功或 CPU reference 重命名为 GPU 验收。
- [CPU 最终 workspace 日志](../target/validation/qwen35-final-cpu-tests-final.log)：
  该批次为 **1,274 passed、0 failed、6 ignored、2 精确 skip**。首轮 2 项因缺少
  `models/Qwen3-30B-A3B/config.json` 失败，只跳过
  `real_qwen3_30b_metadata_binds_to_the_generic_state_dict` 和
  `qwen_adapter_uses_real_standard_kv_and_rejects_cuda` 后重跑。
  [doctest](../target/validation/qwen35-final-cpu-doc.log)另 **7 passed**。
  Qwen35 NAS CPU 数值 target 是在上述 6 ignored 之外随后单独执行的 **1 passed**，
  不修改 workspace 那一次的 ignored 计数。
- [最终 Qwen35 HTTP](../target/validation/qwen35-final-gpu-05-qwen35-http.log)为
  **1 passed，230.51 s**；[最终默认 CLI](../target/validation/qwen35-final-gpu-05-cli-serve.log)
  为 **1 passed，69.59 s**，后者覆盖 SSE 和 SIGTERM。时间是整个 target 的耗时，
  不是 ready 或每 token latency；6.77 s ready 和约 15 s/token 仍仅引用先前
  `target/validation/qwen35-launch-final.log` 的前台启动实测，不作 release 性能声明。
- [最终 contexts](../target/validation/qwen35-final-gpu-06-final-contexts.log)：
  **0 个残留 GPU contexts / processes**，compute-apps 仅表头，八卡利用率 0%、
  显存回到 2–3 MiB 基线；没有 reset GPU 或终止无关进程。
- **strict Clippy 未全绿**：[原 strict 日志](../target/validation/qwen35-final-cpu-clippy.log)
  中两个新增 lint，`components.rs` 的 `manual_is_multiple_of` 和
  `unnecessary_lazy_evaluations`，现已修复；既有 `collapsible_if`、
  `chunks_exact_to_as_chunks`、`items_after_test_module` 仍是已知阻塞。
  不把这两处修复、fmt/check/test 通过写成 strict Clippy 全部通过。
- 未执行全 workspace ignored、全部 process faults；已知旧 shared-FFN formal-shape
  latency unsupported case 不在该批次范围。该批次不覆盖 35B/FP8；其后完成的
  35B 单卡及 dedicated thread EP 验收见下文；vision/MTP 执行与 hybrid TP/PP/process 仍不支持。

前次文档收尾的 model 21 passed / 3 ignored、runtime 2 passed，以及后续编译阶段
120 秒超时，是较早的独立执行记录，不再充当当前最终验收结果。最终结果以上述
报告和分项日志为准；这不意味着无 skip、无 ignored、无 lint 阻塞的全 workspace 通过。

## 35B CPU compressed host prewarm / readiness gate

35B 的默认启动策略是 `full`：在模型 worker 报 ready 之前，按 strict state-dict
binding 选出全部 10,240 个 routed experts（30,720 个 gate/up/down projections），
使用现有 verified positioned-read/FD session 增量读取原始 E4M3FN + BF16/F32 block
scales。它不会展开 F32、不会 pin 35 GiB、不会调用 shell/Python，也不复用旧的
BF16/FP4 `HostStagedExpertCache` 作为第二 authority。

默认 host cap 为 **40 GiB（42,949,672,960 B）**、10,240 experts、4 个 bounded
workers；压缩 payload 是 **32,216,186,880 B**，cap 另计 worker staging、proof/map
metadata 和 stack reserve，**不是全进程 RSS 上限，也不是 full-F32 模型缓存**。

[`StandardHostStartupMemory`](../crates/ferrule-model/src/transformer/host_experts/startup.rs)
与 [`HostMemoryBudget::admit`](../crates/ferrule-model/src/transformer/host_experts/budget.rs)
按实际 binding 核算尚未构建的非 expert host buffers、metadata、最大串行 reader/
conversion temporary。剩余可用量为 `min(MemAvailable, cgroup limit - current)`，
包含可见 v1/v2 ancestor/high limits。所需量为
`max(warm incremental peak, retained cache + remaining base + temporary) + headroom`；
默认 headroom 1 GiB。已在 RSS/current 中的 catalog 不重复计费，base credit 必须有
明确 ownership，不能从总 RSS 猜测，更不能再次从 available 中减 RSS。它不是
“40 GiB cap + 1 GiB”的固定规则；不足返回 typed error，不静默降级。

CLI 接受 `--expert-prewarm full|lazy`、`--expert-host-cache-mb`（默认 40960 MiB）、
`--expert-host-cache-entries`（默认 10240）、`--expert-prewarm-workers`（1..8，默认 4）。
单卡显式 `lazy` 关闭 host retention，标明 **NOT prewarmed**，新 prompt 可能读 NAS；
**dedicated EP2/4/8 严格拒绝 lazy**，所有 owner 必须共享一个完整 full image。
这些 flags 是 host prewarm 语义，不是 CUDA device cache flags；CUDA 仍只接收
materializer 返回的 prevalidated compressed proof。

warm 采用每个 source 的 pre/post identity boundary；失败时内部 cancel 并 join 所有 worker、
丢弃未发布 entries 并释放 FD。host image 以 generation 和 bound-storage identity
隔离；同一 materializer 的后续 GPU miss 复用 proof payload，不重新读 NAS bytes。
EP 的 root/resources 和 expert owners 克隆同一个 `Arc<HostExpertCache>`，不是每卡
一份 30 GiB image。reader pool 在 publication 前释放；image-wide source preflight、
weight/scale pairing、finite payload proof 和 generation/storage identity 检查仍保留。
零 reread 不意味着取消 source checks，也不意味着 source 可安全地被替换。

[`warm_model_pagecache.sh`](../scripts/warm_model_pagecache.sh) 只暖 Linux OS page cache，
不能代替 readiness-gated proof cache；它会读模型文件、占用 I/O/page cache，但不创建
Ferrule host image。可选用法（无需为了普通 full 启动先运行）：

```sh
bash scripts/warm_model_pagecache.sh --dry-run /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8
bash scripts/warm_model_pagecache.sh --jobs 4 /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8
```

CPU-only NAS 验证命令（不 drop page cache，不启动 CUDA）：

```sh
FERRULE_NUMERIC_FP8_MODEL_DIR=/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 \
  cargo test -p ferrule-model --no-default-features --test host_expert_cache \
  nas_full_compressed_prewarm_then_zero_read_host_pass -- --ignored --nocapture
```

**历史 CPU-only probe** 在 dev `opt-level=1` 下实际读满 **32,216,186,880 bytes**（约 30 GiB，
已经包含 scales），13 个 routed source、4 workers，61,440 次 pread，26 次 source
checks。没有用所有模型文件的总量冒充 selected routed bytes。

| CPU probe | 全 routed warm | 进程物理 read_bytes | 第二遍全 host-page touch | 第二遍 checkpoint / 物理 / NAS reads |
| --- | ---: | ---: | ---: | --- |
| 首次观测（部分 OS cache 已热） | 23.23 s | 10,261,581,824 | 238 ms | 0 / 0 / 0 |
| 最终复测（OS cache 基本已热） | 7.33 s | 9,932,800 | 231 ms | 0 / 0 / 0 |

该次 cache-only probe 的历史 admission 为 **32,355,648,512 bytes**，host cap **42,949,672,960 bytes**，
`MemAvailable`/cgroup admission 可用 **793,664,167,936 bytes**。read-phase peak FD=13，
publication 前 reader pool 已 drop；`stats.io.live_handles` 是 read-phase 历史快照，
不是 cache 持有 FD 数量。完整日志位于本地 ignored
[`target/validation/host-expert-prewarm-cpu-final.log`](../target/validation/host-expert-prewarm-cpu-final.log)。
NFS server-read delta 是 mount-wide 计数，可能含其他进程；第二遍两轮均为零。
该 probe 不证明全冷 NAS 或 GPU latency；当前单卡 release 观测另见下文。

现有 CPU tests 覆盖 hard cap/available-memory typed errors、proof pointer 复用与零 reread、
跨 image 拒绝、源替换 preflight、失败 warm 的 FD/未发布 payload 回收、CLI full/lazy
语义。provider API 不需要第二个 residency authority：`ExpertProvider::expert_metadata_bindings` 仍只返回轻量 metadata，
GPU miss 经 `expert()` → `StateDictMaterializer::expert_parameter()` 取得同一 host proof。

## 历史规划记录（已被当前实现取代）

早期 metadata-only 阶段曾拒绝公开 35B serving，并规划 BF16-RNE compute、64 experts /
320 MiB device cache / 64 MiB scratch。schema 单份 F32 target **8,464,263,168 B**
与后续保守 model admission **12,637,319,680 B** 都不是 GPU 实测，也不是当前默认值。
当时 registry 的 7/8/10 passed 是不同 CPU/CUDA-feature 批次；编译 120 秒 timeout
没有执行结果，不计通过。BF16 full40 后来仍未接受，当前生产固定 F32/TF32x3。
这些规划不授权 TP/PP/process；后来开放的 dedicated EP 由自己的 placement/admission
决定，而不是放开 generic parallel policy。历史完整数值、CLI、shutdown 与 postfix
proof 在下文单列，不把旧成功或失败改写为当前全量回归。

## 35B-A3B FP8 runtime 接线与实际验收

本节是当前支持口径，取代历史 pending/公开 resolver reject。当前精确
`Qwen3_5MoeForConditionalGeneration` + strict `Qwen35Moe` profile 在 `cuda` feature
下，显式 CUDA 的公开 model support 为 **`Executable`**；默认/显式 CPU plan 始终为
**`Unsupported`**，无 `cuda` feature 的 CUDA plan 也为 **`Unsupported`**。
这不代表 `Qwen35Config::supports_execution()` 的 CPU 便捷 API 已放行；公共 plan
回归见 [`qwen35_engine_plan.rs`](../crates/ferrule-model/tests/qwen35_engine_plan.rs)。
runtime/CLI 的 `Auto` 对这个 exact profile 选择 CUDA；默认单 owner 的公开 profile 为
`cuda-hybrid-numeric-fp8-f32-tf32x3-qwen35-35b-a3b`。这不是 BF16-RNE profile：
factory 固定使用已验证的 `NumericFp8Precision::F32Tf32x3`，不提供会隐式切到
BF16 的 fallback。0.8B dense Qwen35 的默认 CPU/CUDA 行为未改变。

### 单卡与专用 thread EP：不是 generic pipeline

- **默认单 GPU**：`GenericDecoderOptions::with_hybrid_cuda_numeric_fp8_precision`
  使用 `F32Tf32x3`；`Qwen35MoeCapacityLimits::default()` 为全 image **1024 experts /
  4 GiB device cache（含 64 MiB scratch）**，不是每层 1024，也不是旧 64/320 MiB。
  可用 `--cuda-expert-device-entries` / `--cuda-expert-device-bytes` 调整；bytes 是
  原始 bytes，仍受总 device budget 与实际 free VRAM admission 限制。
- **EP2/4/8**：[`prepare_qwen35_expert_parallel`](../crates/ferrule-runtime/src/engine/model_factory/qwen35.rs)
  经 resident registry、[`qwen35_ep.rs`](../crates/ferrule-runtime/src/engine/model_factory/qwen35_ep.rs)
  构造 external-routing root、`ExpertParallelExecutor` 和
  `GenericDecoderRunner::<HybridCudaDecoder>::hybrid_cuda_with_routed_experts`。
  一个 resident engine/model worker、一个 root-only KV authority，另有持久 GPU expert
  threads，不是第二套 model forward 或 generic `PipelineInferenceEngine`。
- 显式给 N 个不同的 CUDA ordinals；**root = devices[0]，expert owners = 同一列表**。
  logical root rank=0、expert ranks=1..N；root 和第一个 expert owner 在同一卡、不同
  contexts。每层按 `expert % N` 分配。CLI 推荐 `--engine auto`；当前 CLI 也将合法
  Qwen35 EP 的 `resident`/`pipeline` 选择导向这个专用入口，不代表 generic pipeline
  family 白名单已开放。普通 generic `EnginePlan` 的 degree 不是 runtime placement admission。
- EP 必须 full host warm；拒绝单卡 device-cache override、CPU、TP/PP/DP/SP/CP>1、
  process ranks、custom rank timeout、restart/replay。无自动多卡分配；单卡不会因有
  八张可见卡自动变成 EP。production 不依赖、链接或调用 NCCL，activation/result
  仍经 host staging，不是零 host transport。
- 两条 35B 路径都拒绝 legacy pinned cache/hotset、prefix cache、speculation、partial
  layers 和未消费的 reader/head overrides；vision/MTP 只校验附件，不执行。

### EP8 residency 与物理卡预算

[EP8 acceptance report](../target/validation/qwen35-ep-acceptance/REPORT.md)及
[CLI result](../target/validation/qwen35-ep-acceptance/cli-result.json)证明八个 owner
都有非零 calls/tokens，而不只是启动八个空闲 worker。每 owner **1280 experts ×
3,146,112 B ≈ 1280 × 3 MiB ≈ 3.75 GiB**（精确 **4,027,023,360 B**，含 scales）。
ready 后 uploads 固定 `1280 → 1280`、evictions=0、pending uploads=0、无 quarantine；
root routed cache resident/uploads=0，host Arc/generation/hits 不变。该 Hello 请求
`.safetensors openat`、`rchar/syscr/read_bytes` delta 均为 0。这些零值针对 weights；
不排除 activation H2D/D2H，也不是未来任意工作负载的性能保证。

默认 CLI（ctx1024、4 sequences）的 physical-card ledger：

| 分项，bytes | root + 第一个 expert owner 所在卡 | 其余每卡 |
| --- | ---: | ---: |
| root weights upper bound | 9,753,997,824 | 0 |
| owned compressed expert weights | 4,027,023,360 | 4,027,023,360 |
| recurrent/conv state cap | 601,620,480 | 0 |
| compact KV（含 shadows） | 335,544,320 | 0 |
| workspace（含 activation scratch） | 1,223,032,960 | 72,876,032 |
| allocator/context margin（2 / 1 contexts） | 1,073,741,824 | 536,870,912 |
| **combined required** | **17,014,960,768** | **4,636,770,304** |
| **实测 ready GPU usage** | **9907 MiB** | **4170 MiB** |

root 的实际 resident bindings 为 **5,578,582,016 B**；上表保留 model estimator
对非 FP8 aliases 的保守余量，不擅自减掉，也不误称为本地 expert-cache charge。
启动在 host warm 前和 weight uploads 前检查实际 free memory。EP2/4 同样准入，
不能从 EP8 的每卡预算推导 EP2 在任意 24 GiB 卡上必然可启动。library strict/smoke
使用较小 scheduler，root 卡所需 **16,362,222,208 B**，不可混成默认 CLI 数字。
上述早期 CLI 验收显式用了 **32768 MiB host cap**；当前默认与下文最新 EP8 性能运行
均为 **40960 MiB**，不能混用两批启动配置。

### Release 启动与本地路径

从 `/root/Ferrule` 运行，模型/tokenizer 已在本机 NAS 路径；不自动下载模型。
Linux full-prewarm memory probe、CUDA toolkit/driver 和 pinned CUTLASS checkout 必须
可用。`sm_86` 对应本次 RTX 3090；默认 CUTLASS 路径为 `target/vendor/cutlass`，
也可用 `FERRULE_CUTLASS_DIR` 指向兼容的 pinned checkout。
[`setup_cutlass.sh`](../scripts/setup_cutlass.sh) 缺失时会联网获取固定版本，已存在时校验。

本次报告确认 release 已构建成功（3m12s，binary hash 见性能 summary）；以下直接运行
已构建的 `target/release/ferrule`，不需要再次 build。本次 docs-only 未执行构建或启动。

```sh
# EP8；ordinals 是 CUDA_VISIBLE_DEVICES 重映射后的可见编号。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FERRULE_LOG=info \
  target/release/ferrule serve /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 \
  --backend cuda --engine auto --expert-parallel 8 --rank-backend thread \
  --devices 0,1,2,3,4,5,6,7 \
  --expert-prewarm full --expert-prewarm-workers 4 \
  --expert-host-cache-mb 40960 --expert-host-cache-entries 10240 \
  --served-model-name qwen35-ep8 --host 127.0.0.1 --port 8000
```

单卡替代命令（不要与上例同时占用端口；同样是 full40，不是截层）：

```sh
CUDA_VISIBLE_DEVICES=0 FERRULE_LOG=info \
  target/release/ferrule serve /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 \
  --backend cuda --engine auto --expert-prewarm full \
  --served-model-name qwen35-single --host 127.0.0.1 --port 8000
```

另一个终端验证 EP8 endpoint；Ctrl-C/SIGTERM 请求 graceful shutdown，不需强杀 GPU：

```sh
curl -fsS http://127.0.0.1:8000/health
curl --max-time 120 -N http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen35-ep8","prompt":"Hello","max_tokens":8,"stream":true,"stream_options":{"include_usage":true}}'
```

### EP8 release 实测：这些场景已交互可用

2026-09-26 的 [EP8 REPORT](../target/validation/performance/ep8-20260926/REPORT.md) 与
[summary.json](../target/validation/performance/ep8-20260926/summary.json)记录一次新 EP8
进程及进程内重复请求：完整 40 层、F32/TF32x3、greedy、prefix cache 关闭、ctx1024，
相同的 Hello/chat/184-token prompt；保留正常 INFO logging，无 strace/Nsight 干扰。
**这些短/有界 prompt 场景已交互可用**，不是多用户容量、数千 token context 或纯计算
线性八倍扩展性证明。`hello-cold` 仅指新进程首请求：OS page cache 未清空、受先前
运行影响；EP8 的全部专家在 ready 前已装入 GPU，不能称为冷存储测试。

启动 **ready 29.723 s、shared host warm 7.006 s**；ready host RSS **36.889 GiB**，
sampled startup/process peak **40.678 GiB**。GPU0 ready **9907 MiB**，其余七卡各
**4170 MiB**（运行 sampled peaks 为 10155 / 4172 MiB）。host cap 仍为 40 GiB，
不是全进程 RSS cap；本次 `required_available_bytes=43,429,916,112` 是实际 remaining
host allocation ledger 加 headroom，不是跨配置常数，也不替代 GPU physical-card budget。

| 输入 → 输出 / 请求 | TTFT，s | stream decode tok/s | 完整 HTTP，s | HTTP tok/s |
| --- | ---: | ---: | ---: | ---: |
| Hello 1 → 8，首次 | 0.192 | 5.783 | 1.546 | 5.174 |
| Hello 1 → 8，重复 | 0.167 | 5.806 | 1.517 | 5.273 |
| chat template 后 23 → 32，首次 | 0.458 | 5.796 | 5.956 | 5.372 |
| chat 23 → 32，重复 | 0.456 | 5.807 | 5.945 | 5.383 |
| prompt 184 → 32，首次 | 2.875 | 5.670 | 8.491 | 3.769 |
| prompt 184 → 32，重复 | 2.869 | 5.643 | 8.515 | 3.758 |

表按 REPORT 精度展示，原始值见 JSON。stream rate 按首末可见 token 之间的间隔计算，
**排除 TTFT 与 terminal HTTP completion**；HTTP rate 是 completion tokens / 完整
HTTP duration，直到 `[DONE]`，包含首 token 等待和末尾 append/finish/transport。
因此 chat 的 stream 约 5.80 tok/s、HTTP 约 **5.37 tok/s**；184-token prompt 分别
约 5.65 和 **3.76 tok/s**，不能混用。含 Capital4 在内七个主要请求两种吞吐均超过
1 tok/s，最大观测 ITL 0.187 s；这些不是未来负载的延迟保证。

对比保留的 [post-GQA 单卡 REPORT](../target/validation/performance/final-20260925/REPORT.md)
与 [summary](../target/validation/performance/final-20260925/summary.json)：Hello 首次
TTFT 约 0.45 → 0.192 s；23-token chat 约 3.1 → 0.46 s；184-token prompt
**TTFT 约 20 → 2.87 s、完整 HTTP 约 32 → 8.5 s**。未重跑单卡。双方都有 full host
warm/40 GiB cap，但单卡在 ready 时 device expert cache 为空，采用 **1024 entries /
4 GiB bounded cache**；EP8 则 **10,240 experts 全常驻、每 owner 1280**。
这包含 residency 与 distributed execution 的部署配置差异，不是相同 cache 条件下的
纯 EP scaling，也不能归因为未经 profiler 证明的 kernel 加速。

本次 **71 项输出/协议/I/O/lifecycle checks 全部通过**，包括匹配输入、text/event
pieces/usage、重复输出、184→32 usage、断连两 token 后 health/recovery、stop/EOS。
**178 transactions × 8 owners = 1,424 snapshots**，所有 owner 均有非零 calls/tokens，
每个 snapshot 的 resident experts=1280、uploads `1280 → 1280`、evictions=0；ready
与最终状态无 pending uploads/quarantine。owner calls/tokens 不是用户输出 token 数。
每个 request interval 和 ready→shutdown 的 **rchar/syscr/read_bytes delta 全为 0**；
零 expert-weight uploads 不排除 activation transfers，零读取也不撤销 source proof/preflight。
SIGTERM exit 0、`physically_closed=true`、PID/compute apps 清空，GPU 回到 3/2 MiB。

性能运行验证输出一致性与 lifecycle，**没有重跑 full-logits oracle**。先前独立
[EP8 strict metrics](../target/validation/qwen35-ep-strict/metrics.json) 的
**1,986,560 logits、max abs `4.196166992e-5`**（routes/state 同时通过）仍是数值依据，
详见下文；不能把这 71 checks 当成新的数值或 bitwise 等价证明。

host immutable image、prepared model image、device expert cache、KV/state 以及 CUDA graph
地址/lifetime 是不同抽象：`Arc` 共享 host proof 不会共享各 owner 的 CUDA allocation；
cache hits/evictions 不等于 graph replay。前述 isolated GEMM replay proof 也不等于
全模型/EP graph capture 或 prefix cache 已支持。

### 独立 35B oracle 与历史单卡完整 40 层数值证明

[`scripts/qwen35_fp8_reference.py`](../scripts/qwen35_fp8_reference.py) 是独立于 0.8B
脚本的 standalone CPU oracle；契约见
[`qwen35_fp8_reference.md`](../scripts/qwen35_fp8_reference.md)。显式 **`--precision f32`**
使用 `F32(E4M3FN) * F32(BF16 numeric block scale)`，activation 保持 F32，CPU linear
采用 strict F32、禁用 TF32，不模拟 CUTLASS reduction order。GPU 的 **F32Tf32x3**
是 F32 boundary/accumulation/output + TF32x3 GEMM，不是严格 IEEE SGEMM 或 bitwise
等价。FP8 raw compression 不意味着全模型解压为常驻 F32；3090 不支持 native FP8 MMA。

脚本默认 **8 CPU intra-op threads / 1 inter-op thread、lazy 256 MiB LRU、16 GiB
hard address-space cap**，外加 RSS 监视和最多 900 秒 deadline；不初始化 CUDA，
不使用 vLLM，不导入 Ferrule production code。Python/Transformers 仅用于生成验证
产物，不是 Rust production serving 依赖。大于 LRU 的临时权重和转换仍受总内存上限
约束，256 MiB 不是进程全部内存。

```sh
# fresh output only；默认模型为 /mnt/nas1/hf/Qwen3.5-35B-A3B-FP8
/opt/conda310/bin/python -B scripts/qwen35_fp8_reference.py \
  --precision f32 --threads 8 --cache-mib 256 --memory-cap-gib 16 \
  --timeout 900 --capital --cache-states
```

脚本打印新的带时间戳 artifact 目录，拒绝覆盖已有目录；已验收的 F32 产物为
`target/validation/qwen35-fp8-reference-full40-f32-v1`。`--precision bf16_rne` 仍是
脚本默认的旧契约，**不是**这次已通过的 F32 oracle，不能省略 precision 后混用产物。

[Hello 日志](../target/validation/qwen35-35b-final-gpu-13-full40-f32-hello.log)和
[capital 日志](../target/validation/qwen35-35b-final-gpu-14-full40-f32-capital.log)记录
真实 NAS 权重的**完整 40 层（30 GDN + 10 full attention）**、全部 prefill rows 和
各一次 decode；`40` 不是测试 case 数。这是历史单卡 proof，不能混成 EP8 误差，
也不是 GQA 后性能子任务重跑 oracle 的声明。

| Case / call | 全 vocabulary logits | max abs |
| --- | ---: | ---: |
| Hello prefill（1 row） | 248,320 | `3.385543823e-5` |
| Hello decode.0 | 248,320 | `1.382827759e-5` |
| capital prefill（5 rows） | 1,241,600 | `1.144409180e-5` |
| capital decode.0 | 248,320 | `8.106231689e-6` |

合计 **1,986,560 logits**，原容差 **`2e-4 + 2e-4 * abs(ref)`** 外为 0；四次调用的
route slot/set mismatches 都为 0、routing weights 通过。每个 case 的 120 个
conv/recurrent snapshots、33,423,360 state values 也全部满足原容差；greedy IDs、
position、eviction、pending uploads=0 和无 quarantine 检查通过。

从 workspace 根目录串行执行以下精确 tests（GNU `timeout` 为外层预算，不是性能承诺）：

```sh
CUDA_VISIBLE_DEVICES=0 CARGO_INCREMENTAL=0 FERRULE_CUDA_ARCH=sm_86 \
FERRULE_NUMERIC_FP8_MODEL_DIR=/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 \
FERRULE_NUMERIC_FP8_F32_REFERENCE_DIR=/root/Ferrule/target/validation/qwen35-fp8-reference-full40-f32-v1 \
  timeout --signal=TERM --kill-after=15s 875s \
  cargo test --locked -p ferrule-model --features cuda --test hybrid_cuda \
  numeric::f32_profile::qwen35_35b_numeric_f32_full40_hello_prefill_decode_matches_reference \
  -- --exact --ignored --nocapture --test-threads=1

CUDA_VISIBLE_DEVICES=0 CARGO_INCREMENTAL=0 FERRULE_CUDA_ARCH=sm_86 \
FERRULE_NUMERIC_FP8_MODEL_DIR=/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8 \
FERRULE_NUMERIC_FP8_F32_REFERENCE_DIR=/root/Ferrule/target/validation/qwen35-fp8-reference-full40-f32-v1 \
  timeout --signal=TERM --kill-after=15s 875s \
  cargo test --locked -p ferrule-model --features cuda --test hybrid_cuda \
  numeric::f32_profile::qwen35_35b_numeric_f32_full40_capital_prefill_decode_matches_reference \
  -- --exact --ignored --nocapture --test-threads=1
```

### EP8 strict full40 oracle（独立于单卡证据）

[`qwen35_ep_strict_tests.rs`](../crates/ferrule-runtime/src/engine/model_factory/qwen35_ep_strict_tests.rs)
复用真实 factory constructor 的 prepared root runner，不重写 forward。当前 EP backend
profile 为 `cuda-hybrid-numeric-fp8-f32-tf32x3-qwen35-thread-ep`。检查 oracle
manifest/profile/hash，双方 finite，在 F64 比较原始 `atol=rtol=2e-4`，未放宽阈值。
[metrics JSON](../target/validation/qwen35-ep-strict/metrics.json) 与
[实跑日志](../target/validation/qwen35-ep-strict/full40-rerun.log)记录 **1 passed**：

| 检查 | 范围 | 最大绝对误差 / 结果 |
| --- | --- | --- |
| 全 vocabulary logits | Hello/capital prefill + 各一次 decode；1,986,560 values | `4.196166992e-5`，0 failures |
| ordered routes | 160 layer/stage records，2,560 selected IDs + 2,560 weights | 原 rank-slot 顺序 mismatch=0，weights 通过 |
| conv states | 120 snapshots，3,932,160 values | `5.245208740e-5`，0 failures |
| recurrent states | 120 snapshots，62,914,560 values | `1.645088196e-5`，0 failures |

两 case 复用同一 host Arc/generation、同一 root runner，中间 reset sequence；八个 owner
均有实际工作且 resident/uploads 不变，shutdown 完成。首轮 `cuInit=3` 在 warm 前失败的
[日志](../target/validation/qwen35-ep-strict/full40.log)保留，不算通过；显式 devices 重跑
没有修改容差。以下只是复现命令，本次 docs-only 未执行：

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FERRULE_CUDA_ARCH=sm_86 CARGO_INCREMENTAL=0 \
  timeout --signal=TERM --kill-after=15s 900s cargo test --locked --release \
  -p ferrule-runtime --features cuda --lib \
  engine::model_factory::qwen35::ep::strict::qwen35_ep8_factory_full40_f32_strict_oracle \
  -- --exact --ignored --nocapture --test-threads=1
```

### BF16 diagnostic：仍未接受

旧 native BF16-RNE/F32-accumulate full40 路径仍是 **experimental / diagnostic-only、
known-unaccepted**：此前 Hello prefill/decode 最大误差约 **0.035 / 0.132**，不能用
F32 的通过替它背书。保留的精确测例为
`numeric::qwen35_35b_numeric_full40_hello_prefill_decode_matches_reference`
（[`numeric.rs`](../crates/ferrule-model/tests/hybrid_cuda/numeric.rs)），其 ignored 原因
明确记录 known-unaccepted；原产物和失败比较保留，没有放宽容差或改成伪绿。
孤立 BF16 kernels/tiny fixtures 通过不等于该 full-model profile 被接受。

### 历史 debug default CLI smoke（旧 64/320 MiB 配置）

`qwen35_35b_fp8_f32_default_serve_sse_and_sigterm` 的既有记录为 **1 passed / 798.16 s**：
ready **16.42 s**，Hello/capital 单 token SSE 各 **237.11 / 543.48 s**，输出 `,` /
` Paris`、usage、length 和 `[DONE]`，SIGTERM exit 0、无残留 contexts。此时 host full warm、
较大 device cache 和后续性能优化尚不能按当前默认回填；这些数字不是当前 release latency。

历史 admission **12,637,319,680 B**；非 expert resident **5,578,582,016 B**，64 experts
共 **201,351,168 B**，另有 64 MiB scratch，实测 cache peak **273,467,392 B**，未超
320 MiB。uploads=2093、evictions=2029、pending=0、无 quarantine；KV allocated pages=0。
它是 CLI 功能验证，不是全 logits oracle，也早于 explicit physical-close 修复。
原始首次 `/mnt1/...` 路径错误不算通过；真实模型路径是 `/mnt/nas1/hf/...`。

当时 registry 10 passed、CLI Qwen35 2 passed、server 2 passed/2 ignored，以及随后
runtime/CLI 的定向补测是有重叠的独立批次，不累计成当前 workspace 全绿。原始 GPU/
postfix 汇总保留于文末；后续 shutdown 补证如下，当前单卡性能与 EP8 strict proof
分别见上文，不相互替代。

### Early emission、cancel pending 与完成边界

[`ResidentTokenEvent`](../crates/ferrule-runtime/src/engine/driver.rs) 在 native proposals
关闭时，可在 selecting forward commit 后、该 token 自己的 KV append 前发送；**最终
requested token 仍必须执行 decode/KV append**。early text 不是成功终态，不能据此跳过
final append 或宣称 TTFT 已包含整个 transaction。commit 后去重输出，成功完成才保证
retained KV frontier 完整；因此 SSE 最后一个可见 token 到 `[DONE]` 仍有 tail。

cancel 已接受但 cleanup 尚 pending 时保留 request/custody，由后续 poll/drain 处理，
不能重发 cancel 或把它变成正常成功终态；early event 也不能被重复发出。
[`worker.rs`](../crates/ferrule-server/src/worker.rs) 与 driver 中保留这些修复的回归：
`early_stream_profiles_selection_commit_not_next_forward_and_keeps_final_kv`、
`early_pending_publish_final_token_cancel_wins_over_max_tokens` 和
`pending_cancellation_retains_request_ownership_until_model_quiesces`。
这些是源码中的 tests，不表示本次重新执行或覆盖物理 driver fault。
physical shutdown 修复见下一节：logical pages=0、cancel accepted、Drop 或 thread exit
都不是 physical completion，unknown/quarantine 不得冒充成功清理。

## P2 审计修复：logical drain 不等于 physical shutdown（35B service criteria 已实测通过）

旧 reporting wrapper 仅调用 `ResidentInferenceEngine::shutdown()` 的 logical drain，
随后在 Drop 中 `try_into_runner()` 并直接丢弃 runner。这不能证明
`GenericDecoderRunner::shutdown()` 成功，最后一次物理同步失败可能只留在 Drop 的
quarantine 路径，而 worker/CLI 已返回 Ok。历史 798.16 秒 smoke 不充当本修复的证明。

当前顺序：

1. `ResidentTopKDriver::shutdown_and_close_progress` 完成原有 transaction/session/KV/
   materialization drain；原 `shutdown` / `shutdown_progress` 保留 logical-only 语义。
2. 在**原 executor 内借用同一个 runner**调用
   `ResidentModelRunner::shutdown_physical`；所有 GenericDecoder CPU/CUDA composition
   override 直接进入其实际 `shutdown()`，不是兼容 no-op。
3. 只有 physical close 返回 Ok 才缓存 Complete，保证成功幂等。失败直接返回原 typed
   error，不归类为 Pending/Complete，不搬走 runner/custody；显式 retry 保留模型的
   unknown-completion/quarantine 状态。
4. object-safe 与 concrete local inference owner 均使用这个完整顺序。worker 的原有
   join/error 通道传播错误；没有第二个 engine、worker 或 shutdown authority。
5. 35B wrapper 不再 `take` engine 或 `try_into_runner`。统计只读借用，日志明确标记
   `physically_closed=true/false`。已经尝试显式关闭后 Drop 不会偷偷重试失败；没有
   显式调用时 Drop 仅做一次 best-effort，不把逻辑页数归零当成 GPU fence。

新增 deterministic tests 验证 logical pages=0 仍传播 physical error、失败保留原
runner、健康重试成功后幂等、unknown retry 不清 quarantine、logical custody 未 drain
不调用 physical close、concrete local owner 的错误与重试，以及空闲 worker physical
error 仍从 join 返回。Generic decoder 测试直接走 trait seam，确认真实 shutdown 状态
与 completion-hub close，防止其误用默认兼容实现。

以下是先前不使用 GPU 的 seam/unit 验证；真实 35B physical shutdown 的补充证据
单列于表后，不混入这些计数：

| Selected checks | 结果 |
| --- | --- |
| runtime `--lib engine::`（含 4 项 physical-close 故障/重试测试） | 129 passed |
| runtime `--lib expert_residency::` | 10 passed |
| model `--lib decoder::tests::` | 24 passed |
| server `--lib worker::tests` | 6 passed |
| server `--test worker_lifecycle` | 6 passed |
| server `--test qwen35_http`，无 CUDA feature | 2 passed |
| CLI `--test serve`，无 CUDA feature，真实 SIGINT/SIGTERM | 2 passed |
| runtime/server/CLI `cargo check`；runtime `--features cuda --tests` check | 通过 |

这些数字按命令独立记录，新增的 4 项已包含在 129 项中，不重复相加。

### 真实 35B shutdown 与同端口重启补证

[独立 postcheck](../target/validation/qwen35-35b-shutdown-final-independent-postcheck.json)
确认 **实际 service criteria 已通过**，但不把旧 launcher 整体改写为 all-green：

- 旧 r2 的 Hello SSE 完整、未截断，包含 `,`、usage、`finish_reason=length` 和
  `[DONE]`；SIGTERM 后 CLI exit 0，runtime/model 均报告 `physically_closed=true`，
  KV allocated pages=0，GPU contexts/compute processes 为空。**旧 launcher exit 1**
  仍保留：plain-bind probe 报 `EADDRINUSE`，见
  [r2 原始结果](../target/validation/qwen35-35b-shutdown-final-r2-result.json)。
- 独立 finite TCP 复现证明：plain bind（`SO_REUSEADDR=0`）与生产 Tokio/Mio listener
  的 `SO_REUSEADDR=1` 不等价。修正后的测试探针检查无 LISTEN、连接 `ECONNREFUSED`，
  并以 `SO_REUSEADDR=1 / SO_REUSEPORT=0` bind/listen；**不倒推旧 r2 的失败原因已被
  证明是 TIME_WAIT**。旧失败日志和探针保留供审计。
- [新 readiness/restart 结果](../target/validation/qwen35-35b-shutdown-final-readiness-restart-result.json)：
  同一端口 **47363** 上真实加载 35B GPU，两次 ready→SIGTERM 的单次启动耗时分别
  **15.036 / 14.835 s**；均 health 200、CLI exit 0、`physically_closed=true`、KV=0，
  退出后无 GPU context/compute process，修正探针通过，**同端口实际重启成功**。
  此 readiness launcher exit 0；两次均未发送 generation 请求，不是重跑 Hello SSE、
  capital、full logits 或历史 798 秒完整验证，也不是性能或 driver-fault 恢复声明。

## 最终 CPU gate（既有结果，本次未执行）

[final CPU 汇总](../target/validation/performance-ep8-final-cpu-20260926T140807Z-summary.txt)
记录 workspace **1,389 passed、0 failed、13 ignored、2 known-fixture filters**；过滤项为
`real_qwen3_30b_metadata_binds_to_the_generic_state_dict` 和
`qwen_adapter_uses_real_standard_kv_and_rejects_cuda`，不算通过。
[doctests](../target/validation/performance-ep8-final-cpu-20260926T140807Z-doc.log)另 **9 passed**；
[CPU check](../target/validation/performance-ep8-final-cpu-20260926T140807Z-check.log)、
[CUDA sm_86 compile-only check](../target/validation/performance-ep8-final-cpu-20260926T140807Z-cuda-check.log)
与 [fmt](../target/validation/performance-ep8-final-cpu-20260926T140807Z-fmt.log)通过，汇总
warning lines=0。没有运行 ignored/GPU kernels，不代表 strict Clippy 全绿或 BF16 full40
已接受；不要把 CPU gate、EP8 性能运行和先前 strict oracle 相互替代。下面旧 postfix
计数保留为历史，不与当前 gate 累加。

## 历史 postfix 验证口径（非当前全量回归）

本次 docs-only 只核对既有报告与日志、同步文档并检查链接；不读取源码、不运行 Cargo/GPU
或重新生成 oracle。
以下本地 artifacts 均为 Git ignored，不随文档发布。

- [最终 GPU 定向汇总](../target/validation/qwen35-35b-final-gpu-summary.log)原始选择
  **49 cases：48 passed、1 failed**；失败是
  `cutlass_f32::f32_sm86_exact_capability_and_plan_policy` 的 stale mask assertion：
  actual `0xD86` vs expected `0x586`。源码现已修正；最新交接确认 metadata-filtered
  重跑通过。原汇总的失败记录保留，不能改称一次 49/49 全量重跑或全 GPU/all-workspace
  green。full40 F32 的两个实测本来已通过，不受这个 metadata assertion 影响；
  zero-test discovery、编译和 CPU-only 检查不计作实际 GPU kernel 验收。
- [postfix CPU workspace tests](../target/validation/qwen35-35b-postfix-cargo-test-exact-skip2.log)：
  **1,324 passed、0 failed、8 ignored、2 missing-fixture filters**。缺少
  `models/Qwen3-30B-A3B/config.json` 的
  `real_qwen3_30b_metadata_binds_to_the_generic_state_dict` 与
  `qwen_adapter_uses_real_standard_kv_and_rejects_cuda` 明确过滤，不能当成通过。
  [postfix doctest](../target/validation/qwen35-35b-postfix-cargo-test-doc.log)另 **7 passed**。
- [CPU workspace check](../target/validation/qwen35-35b-postfix-cargo-check.log)、
  [CUDA workspace check](../target/validation/qwen35-35b-postfix-cuda-check-sm86.log)及
  [fmt](../target/validation/qwen35-35b-postfix-fmt.log)通过，warning 0。
  这些不更新先前 strict Clippy 的未全绿结论，也不宣称所有 ignored tests 都执行。
- 真实默认 CLI 的既有 16.42 s ready / 237.11 s Hello / 543.48 s capital SSE 和
  SIGTERM 证据保留，属于 **debug 功能验收**。最新 explicit physical shutdown 的
  service criteria 与同端口重启已实测通过，范围及旧 launcher 探针失败见上节；
  该历史批次不包含后续 EP8 strict full40；当前 EP8 release 性能结果见上文，不能回填到旧批次。
