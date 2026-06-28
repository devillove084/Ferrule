# Ferrule: Agentic Edge Inference Runtime

## 1. 项目定位

Ferrule 是一个面向端侧部署的原生多模态推理引擎。两个核心差异化：

- **llama.cpp 的竞争者**：Rust 原生、safetensors 原生、持久计算图、cuda_oxide 内核
- **Elastic State Fabric 的边缘节点**：胖客户端 + 云端控制面的联合架构

当前阶段（v0.2）聚焦推理引擎本身。StateFabric 控制面在后续版本接入。

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│                   Elastic State Fabric (云端)                │
│  StateObject / StateTablet / Transaction / KV Migration     │
│  → autoscaling, routing, multi-tenant, checkpoint           │
│  → Phase 5+ 接入                                            │
├─────────────────────────────────────────────────────────────┤
│                   Ferrule Edge Runtime (端侧)                │
│                                                             │
│  ferrule-app     CLI + GPU kernels (#[kernel]) + HTTP server │
│  ferrule-model   模型架构层 (OLMoE → Gemma/Llama/Qwen/...)   │
│  ferrule-graph   持久计算图 + CPU backend (CudaBackend stub) │
│  ferrule-gguf    格式层 (GGUF + Safetensors, mmap)          │
│  ferrule-core    类型系统 + Error                           │
│                                                             │
│  多级存储: GPU VRAM → CPU RAM → NVMe → HTTP Remote         │
└─────────────────────────────────────────────────────────────┘
```

## 3. 计算图引擎 (ferrule-graph)

### 3.1 持久计算图

与 llama.cpp 的核心差异：llama.cpp 每个 token 重建完整的 ggml 图，Ferrule 构建一次，GPU 常驻。

```rust
struct CGraph {
    tensors: Vec<Tensor>,   // 权重 + 中间结果
    ops: Vec<Op>,           // 预构建的操作序列
    inputs: Vec<usize>,     // 每步更新的输入 (token embedding)
    output: usize,          // 最终输出 (logits)
}
```

每次 decode 只更新输入 tensor，然后 `.compute()` 执行全图。

### 3.2 算子

| Op | 说明 |
|---|---|
| MatMul | 矩阵乘法，CPU 用 rayon GEMV，CUDA 用 `gemv_f32` / `gemv_q4_0` kernel |
| RmsNorm | RMS normalization，支持权重融合 |
| SiLU | SiLU 激活 (`x * sigmoid(x)`) |
| SwiGlu | 融合 SiLU(gate) * up |
| Softmax | 标准 softmax |
| RoPE | Rotary Position Embedding |
| Embedding | 词嵌入查表 |
| Reshape / Transpose / Add / Mul | 基础形状/算术操作 |

### 3.3 Backend

```
CpuBackend  → rayon 并行 (默认)
CudaBackend → cuda_oxide #[kernel] 宏，Rust 编译到 PTX
```

当前 GPU kernel 清单：

| Kernel | 说明 | 来源参考 |
|--------|------|----------|
| `gemv_f32` / `gemv_q4` / `gemv_q2` / `gemv_t1` | 量化 GEMV，4x unroll ILP | llama.cpp vecdotq |
| `gemv_dual_q4` / `gemv_triple_q4` | 融合双/三矩阵 GEMV（gate+up 或 QKV 一次遍历） | — |
| `rms_norm_fused` | 融合 RMS norm：SharedArray 并行 reduction + ptx_asm rsqrt + apply | llama.cpp block_reduce |
| `attn_scores` / `attn_combine_softmax` | GQA-aware attention scores + inline softmax + V combine | llama.cpp flash-attn 简化版 |
| `router_topk` | GPU-side top-k expert 选择 + softmax | — |
| `silu` / `silu_mul` | SiLU 激活，使用 ptx_asm ex2.approx 的 fast_exp | — |
| `rope` | Rotary Position Embedding | — |
| `embed_lookup` / `mul` / `add` / `saxpy` | 基础逐元素操作 | — |

关键设计：
- **零 CPU round-trip**：attention softmax 与 expert 选择均在 GPU 完成
- **Shared memory reduction**：rms_norm 使用 SharedArray + sync_threads 树形归约
- **PTX 内联汇编**：`rsqrt.approx.f32`（RMS norm）、`ex2.approx.f32`（fast exp）

## 4. 多级存储

一个 weight tensor 的定位链：

```
① GPU VRAM (24GB)   ← attention weights, norms, KV cache 常驻
     ↓ miss (LRU eviction)
② CPU RAM (32GB)    ← mmap 零拷贝, 热 expert cache
     ↓ miss
③ NVMe (100GB)      ← direct I/O, 温 expert cache
     ↓ miss
④ HTTP Range 请求    ← 冷 expert 按需流式拉取 (Phase 5)
```

参考论文：FlexGen (offload 调度), PowerInfer (热/冷 expert 分离), vDNN (内存虚拟化)。

### 4.1 WeightCatalog

```rust
struct WeightCatalogEntry {
    name: String,
    shape: [usize; 4],
    dtype: QuantType,
    location: StorageLocation,  // Ram | Disk | Remote | Owned
}
```

Safetensors index.json → shard file + offset → StorageLocation::Disk。
GGUF 文件 → mmap → StorageLocation::Ram。

## 5. 模型格式 (ferrule-gguf)

### 5.1 GGUF

llama.cpp 的标准格式。42 种量化类型（IQ1_S ~ Q8_K, MXFP4, NVFP4），mmap 友好。

### 5.2 Safetensors (差异化)

HuggingFace 标准格式，无需转换。支持 BF16/F16/F32/FP8 E4M3 dtype。头信息 JSON + 数据段 mmap，零拷贝读取。

```rust
let sf = SafeTensorsFile::open("model-00001.safetensors")?;
let embed = sf.tensor("model.embed_tokens.weight")?;
let f32_data = sf.tensor_f32(embed)?; // BF16 → f32 自动转换
```

## 6. 量化内核 (ferrule-kernels)

融合反量化 + 矩阵乘法的算子。支持 Q4_0, Q4_K, Q8_0，每条输出 channel 一次遍历完成，无中间 f32 buffer。

```
CPU:  for row in 0..n:        // rayon 并行
          for block in 0..n_blocks:
              d = f16_to_f32(block.scale)
              for 4-bit quant in block:
                  dot += x[i] * d * (q - 8)

CUDA: 每个 thread 处理一个 output channel
      └─ gemv_q4_0 kernel → 共享内存分块 + warp reduce
```

## 7. 模型架构 (ferrule-model)

### 7.1 目标模型

| 模型 | 参数量 | 架构 | 用途 |
|---|---|---|---|
| Gemma 4 26B-A4B | 26B / 4B active | MoE, 128 experts, sliding+global attn | 主目标 |
| Gemma 3 12B | 12B dense | 标准 decoder-only | 备选 |
| 将来 | - | multimodal (vision+audio) | 多模态 |

### 7.2 Gemma 4 26B-A4B 架构要点

```
hidden_size: 2816         ← 极小核心 (类似 300M 模型)
num_layers: 30            ← 25 sliding_attention + 5 full_attention
num_experts: 128          ← 每层 128 个 expert FFN
top_k_experts: 8          ← 每 token 激活 8 个 expert
moe_intermediate: 704     ← expert FFN: 2816→704→2816 (~6M params)
vocab_size: 262K          ← 大词表
vision: SigLIP 1152d      ← 280 个 visual token
context: 262K             ← 长上下文 (sliding window 1024)
```

参数分解：
- Dense 核心 (attention + norms + embedding): ~500MB Q4_K
- Expert FFNs (30 × 128 = 3840 个): ~14GB Q4_K
- Vision encoder: ~1GB
- 总计 Q4_K: ~15GB → 3090 可全量加载

## 8. 多模态流水线

```
┌──────────┐   ┌──────────────┐   ┌──────────────┐
│  Image   │→  │ Vision Encoder │→  │ 280 soft     │
│  224×224 │   │ SigLIP 1152d  │   │ tokens        │
└──────────┘   └──────────────┘   └──────┬───────┘
                                         │
┌──────────┐   ┌──────────────┐         │
│  Audio   │→  │ Audio Encoder │→ ──────┤
│  16kHz   │   │ (TBD)        │         │
└──────────┘   └──────────────┘         │
                                         ↓
                                  ┌──────────────┐
                                  │  LLM Decoder │
                                  │  (prefill +  │
                                  │   decode)     │
                                  └──────────────┘
```

视觉编码器与语言模型联合 prefill，visual token 和 text token 作为统一序列输入。

## 9. StateFabric (未来接入)

当前代码库中保留但未激活的模块。负责：

- **StateObject**：统一的状态单元 (KVCache, HiddenState, WeightShard, Trajectory, ...)
- **StateTablet**：分片管理，支持迁移和分裂
- **StateTransaction**：跨节点的状态一致性 (2PC)
- **KV Migration**：Session-first quiesce-based 迁移
- **Autoscaling**：基于负载的自动扩缩
- **Routing**：模型版本感知的请求路由

详见 [new_arch.md](new_arch.md)。

## 10. 与 llama.cpp 的差异

| | llama.cpp | Ferrule |
|---|---|---|
| 语言 | C/C++ | Rust |
| 格式 | GGUF 强制 | GGUF + Safetensors 原生 |
| 计算图 | 每 token 重建 | 持久化，一次构建 |
| GPU 后端 | .cu + nvcc | cuda_oxide (#[kernel]) |
| 量化 | 42 种 | Q4_0/Q4_K/Q8_0 (扩展中) |
| 多模态 | 后追加 | 原生 pipeline stage |
| 存储 | CPU/GPU 两档 | 四档 (GPU→RAM→NVMe→HTTP) |
| 客户端 | 纯本地 | 胖客户端 + 云端控制面 |
| 训练/RL | 无 | 设计预留 (trajectory, advantage) |

## 11. MVP 路线

### Phase 1: CPU 推理 (done)
- [x] GGUF + Safetensors 读取
- [x] 持久计算图
- [x] CPU backend (rayon)
- [x] 合成模型 benchmark

### Phase 2: CUDA 加速 (当前)
- [x] cuda_oxide 集成, #[kernel] 宏编译通过
- [x] 基础 GPU kernel (gemv_f32, rms_norm, silu, mul, add, saxpy)
- [x] 消除 CPU roundtrip (saxpy + device rms)
- [ ] gemv_q4_k 融合量化 kernel
- [ ] Q4_K 量化管线
- [ ] RoPE + KV cache

### Phase 3: Gemma 4 模型
- [ ] Gemma 4 26B-A4B safetensors 下载
- [ ] Gemma MoE 架构 (2816d core, 128e/layer, top-8, sliding+global attn)
- [ ] Vision encoder (SigLIP 1152d)
- [ ] 262K vocab tokenizer

### Phase 4: 多模态
- [ ] Vision encoder (SigLIP)
- [ ] 多模态 prefill

### Phase 5: 分布式
- [ ] HTTP 远端存储 (Range 请求)
- [ ] StateFabric metadata 接入
- [ ] 胖客户端 + 云端联合推理

## 12. 关键参考论文

| 论文 | 相关能力 |
|---|---|
| FlexGen | Offload 调度、多级存储 (GPU→CPU→Disk) |
| PowerInfer | 热/冷 expert 分离、激活稀疏性利用 |
| MoE-Lightning | MoE 推理中的 expert 预取 |
| vLLM (PagedAttention) | KV cache 分页管理 |
| vDNN | DNN 内存虚拟化 (GPU→CPU swapping) |
| ZeRO-Inference | 权重分片 offload |
| QLoRA | NF4 量化 + LoRA 微调 |

完整论文清单见 [paper_reads.md](paper_reads.md)。

## 13. 2025-2026 最新论文洞察

### MoE 推理优化

| 论文 | 关键洞察 | 对 Ferrule 的启示 |
|---|---|---|
| **Fate** (Feb 2025) | 跨层 gate 预测: 当前层 hidden state 可 99% 预测下一层 expert，零 GPU 开销 | Scheduler 应在当前层计算时为下一层做 expert prefetch |
| **FloE** (May 2025, ICML) | 9.3x expert 压缩 + sparse 预测，3090 上 48.7x 加速 | Expert 内参数矩阵压缩是 PCIe 瓶颈的解法 |
| **SliceMoE** (Dec 2025, DAC) | Bit-sliced cache: 按精度切片缓存，动态分配 precision | 存储应支持 sub-expert 粒度 |
| **MoEpic** (Sep 2025) | Expert 垂直分割 top/bottom，缓存 top segment | WeightCatalog 需 split point 元数据 |
| **D²MoE** (Apr 2025, MobiCom) | Matryoshka 嵌套量化: 每位宽互为子集，运行时选 bit-width | 量化应支持可调精度 |
| **MoBiLE** (Oct 2025) | Big-little expert: 非重要 token 用半数 expert | Expert count 可动态调整 |
| **HybriMoE** (Apr 2025, DAC) | CPU+GPU 混合调度的动态负载均衡 | CPU 也应参与 expert 计算 |

### 量化技术

| 论文 | 洞察 |
|---|---|
| **Q-Palette** (Sep 2025, NeurIPS) | 分数位宽量化器 + trellis-coded quantization，接近信息论最优 |
| **Vec-LUT** (Dec 2025, MobiSys) | 向量化查表推理，4.2x 加速，已集成 llama.cpp |
| **CARVQ** (Oct 2025, EMNLP) | Embedding 压缩到 ~1.6 bits via residual VQ |
