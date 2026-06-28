# Ferrule Roadmap

> 端侧 RL 推理/训练框架。目标：比 llama.cpp 更快、更适合端侧 RL。

## 当前状态

- **模型**: OLMoE 1B (16L×2048d, 64e, top-8 MoE) — 正确推理 ✓
- **GPU**: Q4_0 量化, cuda-oxide PTX kernels
- **已完成 Level 1**: rms_norm_fused (shared memory 并行 reduction), router_topk (GPU-side top-k)
- **缺失**: Flash Attention, batch prefill, PagedAttention, RL 训练基础设施

---

## Level 1: GPU 推理正确性 + 基础性能 (当前 → 50+ tok/s)

### 1.1 消除 CPU round-trip

- [x] `compute_rms` PTX rsqrt fix
- [x] **GPU softmax** — attn_combine_softmax fused kernel, 零 CPU round-trip
- [x] **GPU top-k expert selection** — router_topk kernel: shared memory + GPU-side top-k + softmax

### 1.2 推理吞吐

- [x] **Multi-token batch prefill** — prefill 阶段跳过中间 token 的 lm_head（仅最后 token 计算 logits）
- [x] **RMS norm warp reduce** — rms_norm_fused: shared memory 并行 reduction + rsqrt + apply 融合为一个 kernel
- [ ] **Multi-layer pipeline** — 用 CUDA stream 重叠 layer N 的计算和 layer N+1 的 weight 加载
- [x] **Fused rms_norm + multiply** — rms_norm_fused 单 kernel 完成 compute_rms + rms_norm_apply
- [x] **Interactive chat** — rustyline + console 彩色交互式对话，KV cache 多轮持久

### 1.3 模型加载加速

- [x] **并行 expert 量化** — rayon 并行量化每层 64 个 expert
- [x] **量化权重 checkpoint** — 首次量化后缓存到磁盘，二次加载秒级
- [x] **Safetensors 并行加载** — 去除 MAP_POPULATE，全 tensor 并行 BF16→F32 转换
- [x] **CUDA stream upload** — 使用 channel pipeline 重叠上传和下一层量化

---

## Level 2: 面向 RL 训练的推理增强

### 2.1 Attention 系统

- [ ] **Flash Attention** — O(n) 显存的 exact attention，支持 GQA + sliding window
- [ ] **PagedAttention** — 分页 KV cache，支持变长序列和 memory sharing
- [ ] **KV cache 量化** — KV cache 用 INT8/FP8 存储，减少显存

### 2.2 解码加速

- [ ] **Speculative Decoding** — draft model + target model 的推测解码
- [ ] **Continuous Batching** — 动态合并多个请求的 decode step

### 2.3 训练支持

- [ ] **LoRA adapter** — 插入/合并 adapter，推理时 fuse 到 base weight
- [ ] **Gradient checkpointing** — 用显存换计算，支持长序列训练
- [ ] **FP8/BF16 训练** — 混合精度训练 kernel

---

## Level 3: 模型能力扩展

### 3.1 MoE 优化

- [ ] **Expert offload** — 冷 expert 放 CPU RAM，按需 preload 到 GPU
- [ ] **Expert prefetch** — 当前层计算时预测下一层 expert（参考 Fate 论文）
- [ ] **Expert 负载均衡** — 动态调整 expert 激活数（参考 MoBiLE）

### 3.2 模型架构

- [ ] **Gemma 4 MoE** — 2816d core, 128e/layer, sliding+global attention
- [ ] **Llama/Qwen/Mistral** — 多 family 支持
- [ ] **Vision encoder (SigLIP)** — 多模态 prefill

### 3.3 量化升级

- [ ] **Q4_K / Q6_K** — 与 llama.cpp 对齐的 K-quant 格式
- [ ] **动态位宽** — 运行时选择精度（参考 D²MoE）
- [ ] **Vec-LUT 推理** — 向量化查表加速（参考 Vec-LUT 论文）

---

## Level 4: Elastic State Fabric (训练 + 分布式)

### 4.1 RL Rollout 基础设施

- [ ] **Trajectory 存储** — GRPO rollout → trajectory logging with reward
- [ ] **Multi-rollout 并行** — 一个 GPU 同时跑多个 rollout
- [ ] **Experience replay buffer** — GPU 侧的 replay buffer

### 4.2 训练循环

- [ ] **PPO/GRPO 训练 step** — policy gradient + value loss
- [ ] **GAE advantage 计算** — Generalized Advantage Estimation
- [ ] **Gradient accumulation + optimizer** — AdamW on GPU

### 4.3 分布式

- [ ] **Checkpoint / Model Version Registry** — 模型版本管理
- [ ] **分布式 rollout** — 多节点并行 rollout + 中心化训练
- [ ] **Elastic State Fabric** — 详见 docs/new_arch.md

---

## 性能目标

| 指标 | 当前 | Level 1 | Level 2 |
|------|------|---------|---------|
| OLMoE 单 token | 100ms → 10 tok/s | 15ms → 65 tok/s | 5ms → 200 tok/s |
| 模型加载 (cached) | 0.5s | 0.5s | 0.2s |
| 模型加载 (首次) | 30s | 30s | 5s (GGUF) |
| Prefill (4 tok) | 400ms | 50ms | 20ms |

---

## 与 llama.cpp 的关键差异

| 维度 | llama.cpp | Ferrule |
|------|-----------|---------|
| 语言/编译器 | C/C++ + nvcc | Rust + cuda-oxide (cargo oxide) |
| 模型格式 | GGUF only | GGUF + Safetensors 原生 |
| 计算图 | 每 token 重建 ggml 图 | 持久化 CGraph，一次构建 |
| 训练能力 | 推理 only | 推理 + RL 训练（设计预留） |
| 端侧定位 | 通用推理引擎 | 端侧 agent runtime |
| Tensor 生命周期 | 随图销毁 | StateObject 可 persist/migrate |
