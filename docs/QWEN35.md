# Qwen3.5-0.8B：功能支持与复现

本文只描述当前严格的 **dense Qwen3.5-0.8B、完整 24 层、纯文本**路径，
不把通用 Qwen3 的 TP/PP/EP 能力外推到 Qwen3.5。并行集成总览见
[PARALLELISM.md](PARALLELISM.md)。

真实 `/mnt/nas1/hf/Qwen3.5-0.8B` 的完整 CPU 和单卡 CUDA 数值验收已通过；
最终 CPU / GPU 报告及原始日志已核查：`hybrid_cuda` **13 passed** 中含 **6 个实际
CUDA 测试、1 个比较器验证、6 个 CPU reference 测试**；独立 forward CUDA UT 另
**1 passed**。GPU 验证批次共 **186 selected passed、0 failed**，不是 186 项全 GPU
测试。具体数值、CPU skip 和 Clippy 边界见下文；此次 docs-only 更新未运行 Cargo。
当前源码已接通 CLI → model factory → resident engine → dedicated model worker →
HTTP/SSE。本地 CLI/HTTP 日志也记录了真实 CUDA serving 成功，不能再把 CUDA
描述为仅 metadata 支持或待接入。下面分别列出实现边界、证据和复现命令。

## 支持矩阵与拒绝边界

| 项目 | 当前范围 | 源码依据 |
| --- | --- | --- |
| checkpoint | HF safetensors，nested `qwen3_5` / `qwen3_5_text`，严格 0.8B 几何；24 层中 18 层 GatedDeltaNet、6 层 full GQA | [`config.rs`](../crates/ferrule-model/src/models/qwen35/config.rs)、[`recipe.rs`](../crates/ferrule-model/src/models/qwen35/recipe.rs) |
| CPU | F32 text decoder，默认 backend 选择 CPU | [`model_factory/qwen35.rs`](../crates/ferrule-runtime/src/engine/model_factory/qwen35.rs) |
| CUDA | `cuda` feature 下的单设备 resident F32 tensor 路径；显式 `--backend cuda`，当前 factory 使用可见 ordinal 0 | 同上；[`hybrid_cuda.rs`](../crates/ferrule-model/src/decoder/hybrid_cuda.rs) |
| 文本 serving | completion/chat、greedy generation、HTTP/SSE、stop/EOS、取消、reset、shutdown；使用 Qwen35 chat template | [`serve.rs`](../crates/ferrule-cli/src/commands/serve.rs)、[`qwen35_http.rs`](../crates/ferrule-server/tests/qwen35_http.rs)、[`qwen35_chat.rs`](../crates/ferrule-model/tests/qwen35_chat.rs) |
| vision / MTP | **不执行**；已知附件可缺席，若出现则必须是完整且合法的 partition，校验后排除出 text state dict | [`name_mapper.rs`](../crates/ferrule-model/src/models/qwen35/name_mapper.rs)、[`adapter.rs`](../crates/ferrule-model/src/models/qwen35/adapter.rs) |
| MoE / 35B / FP8 | **unsupported**；family 识别不等于能加载执行。MoE、量化配置（含 FP8）、其他尺寸及 flat text export 被拒绝 | [`config.rs`](../crates/ferrule-model/src/models/qwen35/config.rs)、[`qwen35_metadata.rs`](../crates/ferrule-model/tests/qwen35_metadata.rs) |
| hybrid TP / PP / EP / process ranks | **unsupported**；Qwen35 不在 standard pipeline family 白名单内，不能套用 Qwen3 的 thread/process PP、EP 或 dense TP 示例 | [`model_factory/pipeline.rs`](../crates/ferrule-runtime/src/engine/model_factory/pipeline.rs)、[`standard/cuda/hybrid.rs`](../crates/ferrule-model/src/transformer/standard/cuda/hybrid.rs) |
| prefix cache / partial retain / speculation | **unsupported**；factory 拒绝非零 prefix cache 与 native proposals，CUDA 仅允许 exact-frontier fork，不能部分保留 recurrent history | [`model_factory/qwen35.rs`](../crates/ferrule-runtime/src/engine/model_factory/qwen35.rs)、[`hybrid_cuda.rs`](../crates/ferrule-model/src/decoder/hybrid_cuda.rs) |
| 截层 / expert cache options | partial `--max-layers` 被拒绝；CLI 显式 expert cache/hotset 参数（即使为 0 或默认值）也被拒绝 | [`serve.rs`](../crates/ferrule-cli/src/commands/serve.rs) |

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

最终 [backend lib 日志](../target/validation/qwen35-final-gpu-01-backend-lib.log)与
[backend operator 日志](../target/validation/qwen35-final-gpu-02-backend.log)确认：
canonical manifest、native/Rust discovery、plan/execution 对应一致，
`F32Gemm ID=11`、mask **`0x586`**，F32 TF32x3 / LinearF32 inference 为 true，
FP8 QueryAKv/Projection 为 false，不能借新增 F32 GEMM 宣称 sm_86 支持 FP8。
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

**性能未优化**：本地 debug CLI 记录的后续 decode token 间隔约 14.85/14.92 秒，
即约 **15 s/token**。这是指定机器、输入和构建下的观测，不是性能承诺、吞吐 benchmark、
CPU 耗时估计或 release 加速保证；prefill/首 token 等待也不能由此等同推算。

## 离线 Transformers 5.2 oracle

[`scripts/qwen35_reference.py`](../scripts/qwen35_reference.py) 是仅供验证的离线
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

### 最终验证记录与来源

[最终 GPU 汇总](../target/validation/qwen35-final-gpu-summary.md)、
[最终 CPU 汇总](../target/validation/qwen35-final-cpu-summary.md)及以下日志都是
**Git ignored 的本地生成证据**，不随仓库发布。链接仅在保留该次 artifacts 的
工作区有效；本次文档同步用 `rg` 核查它们，没有重跑 Cargo 或生成 oracle。

- GPU 批次在 8 张 RTX 3090 上串行执行，`CARGO_INCREMENTAL=0`、
  `FERRULE_CUDA_ARCH=sm_86`、`--locked`、`--test-threads=1`，单命令 timeout 900 秒。
  CUDA workspace all-targets check、CLI / process child 构建通过。
- 合计 **186 selected passed、0 failed**，不是 186 项实际 GPU 测试。
  backend lib 包含 **113 passed / 18 ignored**；其中 KV 10 项随后定向通过，
  不代表全 ignored 已覆盖。hybrid 13 项和另行执行的 forward UT 按上面的构成计数，
  不将 0 tests、编译成功或 CPU reference 重命名为 GPU 验收。
- [CPU 最终 workspace 日志](../target/validation/qwen35-final-cpu-tests-final.log)：
  **1,274 passed、0 failed、6 ignored、2 精确 skip**。首轮 2 项因缺少
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
  latency unsupported case 不在该批次范围。35B/FP8、vision/MTP、Qwen35 hybrid
  TP/PP/EP/process 等不支持边界不因上述测试结果改变。

前次文档收尾的 model 21 passed / 3 ignored、runtime 2 passed，以及后续编译阶段
120 秒超时，是较早的独立执行记录，不再充当当前最终验收结果。最终结果以上述
报告和分项日志为准；这不意味着无 skip、无 ignored、无 lint 阻塞的全 workspace 通过。
