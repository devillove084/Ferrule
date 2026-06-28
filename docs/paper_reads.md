# 低端显卡运行远超显存大模型：论文与系统路线总结

> 更新日期：2026-06-11  
> 主题：在低端 GPU / 消费级 GPU / 端侧设备上运行、推理或微调远超显存容量的大模型。  
> 核心视角：把 GPU 显存从“完整模型容器”变成“高带宽工作集缓存”。

---

## 0. 总览

这类工作解决的不是一个单点问题，而是一组相关问题：

1. **模型权重放不下 GPU 显存**：需要 weight offload、streaming、paging、量化。
2. **KV cache 放不下 GPU 显存**：需要 KV cache paging、eviction、quantization、CPU/NVMe offload。
3. **训练状态放不下 GPU 显存**：需要 optimizer state / gradient / activation offload。
4. **单机资源不够**：需要多台消费级设备协同推理。
5. **端侧 DRAM 不够**：需要 flash-aware / storage-aware tensor layout。
6. **MoE 总参数很大但每 token 只激活少量专家**：需要 expert paging、module batching、CPU-GPU-I/O pipeline。

可以把所有方法抽象成：

```text
GPU HBM      = 小容量、高带宽、低延迟工作集缓存
CPU DRAM     = 中容量、较低带宽的主缓存/权重池/KV 池
NVMe / Flash = 大容量、低带宽、高延迟的后备存储
Network      = 多机拼接模型容量或显存容量
Scheduler    = 决定 prefetch / evict / compute / overlap 的核心组件
```

---

## 1. 阅读优先级

如果目标是“3090 / 4090 / T4 / M 系列 Mac / 手机等低资源设备运行大模型”，建议按下面顺序读。

### 1.1 第一优先级：权重超过显存的大模型推理

| 顺序 | 论文 / 系统 | 主要解决点 | 建议重点 |
|---:|---|---|---|
| 1 | **FlexGen** | 单卡 + CPU + disk 运行大模型 | 分层内存、LP placement、batch amortization |
| 2 | **ZeRO-Inference / DeepSpeed Inference** | 权重放 CPU/NVMe，按层 stream 到 GPU | layer-wise streaming、prefetch、throughput-oriented inference |
| 3 | **LLM in a Flash** | DRAM 不足时从 flash 按需加载权重 | windowing、row-column bundling、flash-aware layout |
| 4 | **LM-Offload** | 用性能模型指导 offload 策略 | quantization-aware offload、thread-level parallelism |
| 5 | **PowerInfer** | 利用神经元激活稀疏性做 CPU/GPU 混合推理 | hot/cold neuron split、activation locality |

### 1.2 第二优先级：KV cache 变成主要显存瓶颈

| 顺序 | 论文 / 系统 | 主要解决点 | 建议重点 |
|---:|---|---|---|
| 1 | **PagedAttention / vLLM** | KV cache 分页和虚拟内存化 | block table、paged KV、prefix sharing |
| 2 | **H2O** | KV cache token eviction | heavy hitters、recent tokens、近似保留 |
| 3 | **KIVI / KVQuant** | KV cache 低比特量化 | per-channel key、per-token value、pre-RoPE quant |
| 4 | **InfiniGen** | 长文本生成中的动态 KV 预取 | rehearsal-based important token speculation |
| 5 | **HeadInfer / LeoAM / IceCache** | CPU/NVMe/语义感知 KV 管理 | head-wise offload、GPU-CPU-Disk hierarchy、semantic clustering |

### 1.3 第三优先级：训练 / 微调方向

| 顺序 | 论文 / 系统 | 主要解决点 | 建议重点 |
|---:|---|---|---|
| 1 | **ZeRO-Offload** | optimizer state / gradient offload 到 CPU | 训练状态拆分、CPU 计算与通信最小化 |
| 2 | **ZeRO-Infinity** | GPU + CPU + NVMe 训练超大模型 | heterogeneous memory、NVMe offload engine |
| 3 | **QLoRA** | 单卡微调大模型 | 4-bit base、NF4、double quantization、paged optimizer |
| 4 | **LoHan / Fuyou** | 4090 + CPU + SSD 微调 100B/175B | active gradient offloading、traffic-aware activation swapping |

---

## 2. 路线一：权重 offload / out-of-core LLM inference

### 2.1 FlexGen

- **论文**：FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU
- **地址**：https://arxiv.org/abs/2303.06865
- **代码**：https://github.com/FMInference/FlexGen
- **关键词**：single GPU、CPU/disk offload、large batch、4-bit compression、LP search

#### 解决的问题

常规推理要求权重、activation、KV cache 大部分常驻 GPU。对于 OPT-175B、BLOOM-176B 这类模型，单卡显存远远不够。FlexGen 的目标是：

```text
在一张普通 GPU 上，通过聚合 GPU、CPU、disk 的容量，做高吞吐 LLM 生成推理。
```

#### 核心设计

1. **分层内存管理**
   - GPU HBM：只放当前计算工作集。
   - CPU DRAM：存放大部分权重、KV cache 或 activation。
   - Disk：后备存储。

2. **tensor placement search**
   - FlexGen 不手写固定 offload 策略，而是用线性规划搜索：
     - weight 放哪里；
     - KV cache 放哪里；
     - activation 放哪里；
     - 每种 tensor 如何搬运。

3. **zig-zag / block schedule**
   - 通过调度把 CPU/GPU/disk 数据搬运和 GPU 计算重叠。

4. **4-bit 压缩**
   - 权重和 attention cache 都可以压缩到 4-bit。
   - 目的不仅是省容量，也减少 PCIe / disk I/O 传输量。

5. **大 batch 摊薄 I/O 成本**
   - FlexGen 更适合 latency-insensitive batch inference。
   - 交互式单请求并不是它的最强场景。

#### 适合场景

- 离线批量生成。
- 本地跑 benchmark。
- 单卡容量不足但 CPU 内存 / NVMe 较充足。
- 追求 throughput，而不是 first-token latency。

#### 局限

- batch size 太小时，I/O 难以摊薄。
- NVMe 和 PCIe 带宽会成为核心瓶颈。
- 对交互式聊天体验不一定好。
- 与现代 continuous batching / paged KV serving 的结合需要重新设计。

#### 对你做系统的启发

FlexGen 的核心抽象是：

```text
GPU HBM 不再是模型容器，而是 tensor working-set cache。
```

如果你要做“3090 跑远大于 24GB 显存的模型”，FlexGen 是第一篇必须精读的系统论文。

---

### 2.2 DeepSpeed Inference

- **论文**：DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale
- **地址**：https://arxiv.org/abs/2207.00032
- **项目**：https://www.deepspeed.ai/
- **关键词**：heterogeneous inference、CPU/NVMe memory、multi-GPU inference、transformer kernels

#### 解决的问题

DeepSpeed Inference 处理两类场景：

1. 模型能放进多 GPU aggregate memory：优化 latency 和 throughput。
2. 模型放不进 aggregate GPU memory：利用 CPU / NVMe 作为异构内存扩展。

#### 核心设计

- 针对 transformer 做 kernel 优化。
- 支持 dense 和 sparse transformer。
- 既支持 latency-oriented serving，也支持 throughput-oriented serving。
- 对超大模型使用 GPU + CPU + NVMe 异构内存。

#### 和 FlexGen 的区别

| 维度 | FlexGen | DeepSpeed Inference |
|---|---|---|
| 目标 | 单 GPU 有限显存高吞吐生成 | 通用 transformer inference scale-out / scale-up |
| 调度 | LP placement + block schedule | 系统级 inference kernel + heterogeneous memory |
| 典型场景 | 离线批量生成 | 大规模推理服务、异构硬件推理 |
| 最强点 | 单卡 out-of-core | 工程完整性和 DeepSpeed 生态 |

---

### 2.3 ZeRO-Inference

- **文章**：ZeRO-Inference: Democratizing massive model inference
- **地址**：https://www.deepspeed.ai/2022/09/09/zero-inference.html
- **示例**：https://github.com/deepspeedai/DeepSpeedExamples/blob/master/inference/huggingface/zero_inference/README.md
- **关键词**：weight streaming、CPU/NVMe offload、zero weights on GPU、throughput-oriented

#### 解决的问题

ZeRO-Inference 把模型权重放在 CPU 或 NVMe 中，GPU 上不常驻完整权重。每次执行某一层时，将该层权重 stream 到 GPU：

```text
for layer in layers:
    prefetch layer weights from CPU/NVMe to GPU
    compute layer on GPU
    evict / overwrite layer weights
```

#### 核心设计

1. **weight hosting**
   - 权重托管在 CPU DRAM 或 NVMe。
   - GPU 显存主要用于当前层权重、activation、KV cache 和 workspace。

2. **layer-wise prefetch**
   - 在计算当前层时，提前搬运下一层权重。

3. **batch 提升算术强度**
   - batch 越大，每次权重加载服务的 token 越多。
   - 因此更偏 throughput-oriented，而不是低延迟。

#### 局限

- batch=1 时，PCIe/NVMe 搬运成本非常明显。
- 权重每层都要搬，decode 阶段容易 memory-bandwidth bound。
- 如果 KV cache 也很大，需要额外 KV 管理策略。

---

### 2.4 LM-Offload

- **论文**：LM-Offload: Performance Model-Guided Generative Inference of Large Language Models with Parallelism Control
- **地址**：https://pasalabs.org/papers/2024/llm_offload_2024.pdf
- **会议页**：https://sc24.supercomputing.org/proceedings/poster/poster_pages/post144.html
- **关键词**：performance model、quantization-aware offload、parallelism control、weight/KV offload

#### 解决的问题

FlexGen 和 ZeRO-Inference 已经证明了 offload 可行，但 offload 系统真正难的是：

```text
不是“能不能放下”，而是“放下之后 GPU 会不会一直等数据”。
```

LM-Offload 更强调性能模型，分析不同 offload、量化、并行线程配置下的吞吐。

#### 核心设计

- 建模 weight offload 和 KV cache offload 的成本。
- 建模量化和反量化的额外开销。
- 控制 CPU 线程级并行度。
- 尝试让数据搬运、CPU 处理和 GPU compute overlap。

#### 对系统实现的启发

如果自己做 out-of-core runtime，不能只实现：

```text
miss -> fetch tensor -> compute
```

而应该实现：

```text
lookahead -> prefetch -> dequant -> schedule -> compute -> evict
```

并且调度要看：

- GPU kernel 还会跑多久；
- 下一层权重能否提前搬完；
- CPU dequant 是否成为瓶颈；
- NVMe 读是否连续；
- 当前 batch 是否足够摊薄 I/O。

---

### 2.5 LLM in a Flash

- **论文**：LLM in a flash: Efficient Large Language Model Inference with Limited Memory
- **地址**：https://arxiv.org/abs/2312.11514
- **Apple Research 页面**：https://machinelearning.apple.com/research/efficient-large-language
- **关键词**：flash memory、limited DRAM、windowing、row-column bundling、activation sparsity

#### 解决的问题

端侧设备经常不是 GPU 显存不够，而是系统 DRAM 不够。LLM in a Flash 的目标是：

```text
模型参数存放在 flash 中，运行时按需搬到 DRAM。
```

#### 核心设计

1. **flash-aware cost model**
   - flash 不擅长随机小读。
   - 更适合大块、连续读取。

2. **windowing**
   - 相邻 token 的 FFN 激活神经元存在重叠。
   - 利用这个重叠复用已经加载的权重。

3. **row-column bundling**
   - 把相关行/列组织成更适合 flash 顺序读取的大块。
   - 减少随机 I/O。

#### 适合场景

- 手机 / Mac / 端侧设备。
- DRAM 容量不足，但 flash 容量充足。
- 允许一定复杂 tensor layout 重排。

#### 和 FlexGen 的区别

| 维度 | FlexGen | LLM in a Flash |
|---|---|---|
| 目标设备 | 单 GPU + CPU + disk | DRAM 受限端侧设备 |
| 主要瓶颈 | GPU memory / PCIe / disk I/O | DRAM capacity / flash I/O pattern |
| 优化重点 | tensor placement、batch schedule | flash sequential read、neuron reuse |
| 粒度 | tensor/block/layer | neuron row/column |

---

## 3. 路线二：CPU/GPU/NPU 混合推理与激活稀疏性

### 3.1 PowerInfer

- **论文**：PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU
- **地址**：https://arxiv.org/abs/2312.12456
- **论文 PDF**：https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf
- **代码**：https://github.com/SJTU-IPADS/PowerInfer
- **关键词**：consumer GPU、hot neurons、cold neurons、CPU-GPU hybrid、activation locality

#### 解决的问题

传统 offload 通常按 layer 或 tensor 搬运权重。PowerInfer 观察到 LLM 推理中 FFN 神经元激活具有明显 power-law 分布：

- 少数 hot neurons 经常激活；
- 多数 cold neurons 只在特定输入上激活。

因此它采用：

```text
hot neurons -> GPU
cold neurons -> CPU
```

#### 核心设计

1. **hot/cold neuron 分离**
   - 高频激活神经元常驻 GPU。
   - 低频激活神经元由 CPU 处理。

2. **activation predictor**
   - 预测当前 token 会激活哪些神经元。
   - 减少无效计算和无效搬运。

3. **neuron-aware sparse operator**
   - 针对稀疏激活设计算子，避免把稀疏逻辑做成低效 gather/scatter。

#### 和 FlexGen 的本质区别

| 维度 | FlexGen | PowerInfer |
|---|---|---|
| 粒度 | layer / tensor / block | neuron |
| 主要利用 | 分层内存 + 大 batch | 激活稀疏性 + locality |
| CPU 作用 | 存储 / 搬运为主 | 参与 cold neuron 计算 |
| GPU 上放什么 | 当前工作集 | hot neurons |
| 适合 | 离线批量 | 更偏交互式本地推理 |

#### 对你的启发

PowerInfer 的抽象可以写成：

```text
GPU = hot path accelerator
CPU = cold path executor
scheduler = activation-aware routing layer
```

这和 MoE expert cache、稀疏 FFN、端侧推理关系很近。

---

### 3.2 PowerInfer-2

- **论文**：PowerInfer-2: Fast Large Language Model Inference on a Smartphone
- **地址**：https://arxiv.org/abs/2406.06282
- **关键词**：smartphone、NPU、CPU、neuron cluster、segmented neuron cache、I/O-computation pipeline

#### 解决的问题

PowerInfer-2 把 PowerInfer 的思想推进到智能手机。手机上有更复杂的异构资源：

- CPU；
- NPU；
- GPU；
- DRAM；
- flash；
- 严格功耗限制。

#### 核心设计

1. **neuron cluster 粒度**
   - 不再单纯按矩阵层做调度。
   - 把矩阵操作拆成 neuron cluster。

2. **异构计算分配**
   - dense activated clusters 放 NPU。
   - sparse clusters 放 CPU。

3. **segmented neuron cache**
   - 减少重复 I/O。
   - 在存储和计算之间做 cluster-level pipeline。

#### 适合场景

- 手机端 LLM。
- NPU/CPU/flash 混合推理。
- DRAM 小于模型大小的端侧场景。

---

## 4. 路线三：多台消费级设备协同推理

### 4.1 Petals

- **论文**：Petals: Collaborative Inference and Fine-tuning of Large Models
- **地址**：https://arxiv.org/abs/2209.01188
- **ACL Demo PDF**：https://aclanthology.org/2023.acl-demo.54.pdf
- **代码**：https://github.com/bigscience-workshop/petals
- **关键词**：collaborative inference、distributed LLM、consumer GPUs、pipeline parallelism

#### 解决的问题

如果单机无法容纳完整模型，可以把模型按 transformer blocks 切分到多台消费级设备上：

```text
client hidden states
  -> server A: layers 0-5
  -> server B: layers 6-11
  -> server C: layers 12-...
  -> logits
```

#### 核心设计

- 每台 server 托管一部分 transformer block。
- client 动态选择一条 server chain。
- 支持推理，也支持基于 hidden states 的参数高效微调。
- 面向互联网环境下的志愿设备 / 研究组空闲设备。

#### 局限

- 网络延迟和不稳定性明显。
- 数据隐私复杂。
- SLA 难保证。
- 节点异构、上下线、负载均衡都很难。

---

### 4.2 Distributed Inference and Fine-tuning of Large Language Models Over The Internet

- **论文**：Distributed Inference and Fine-tuning of Large Language Models Over The Internet
- **地址**：https://arxiv.org/abs/2312.08361
- **关键词**：fault-tolerant inference、load balancing、geodistributed devices、Petals

#### 解决的问题

这是 Petals 路线的进一步系统化研究，重点处理：

- 节点随时掉线；
- 设备性能不均匀；
- 地理分布导致网络延迟不同；
- 如何做自动 partition 和 load balancing。

#### 对你的启发

如果你未来想做类似：

```text
多个低端节点组成一个推理 runtime
每个节点只承担部分 weights / KV / expert
```

这篇比 Petals demo 更值得从系统角度看。

---

## 5. 路线四：KV cache 显存瓶颈

### 5.1 PagedAttention / vLLM

- **论文**：Efficient Memory Management for Large Language Model Serving with PagedAttention
- **地址**：https://arxiv.org/abs/2309.06180
- **代码**：https://github.com/vllm-project/vllm
- **关键词**：KV cache、virtual memory、paging、block table、continuous batching

#### 解决的问题

长上下文和高并发推理中，KV cache 会线性增长：

```text
KV cache size ∝ batch_size × sequence_length × num_layers × hidden_dim
```

传统连续分配容易造成：

- 内存碎片；
- 过度预留；
- 请求间无法共享 prefix；
- batch size 上不去。

#### 核心设计

1. **KV cache block 化**
   - 把 KV cache 切成固定大小 block。
   - 逻辑 token 序列和物理 block 分离。

2. **block table**
   - 类似操作系统页表。
   - request 拥有虚拟 KV 空间。

3. **prefix sharing / copy-on-write**
   - 多个请求共享相同 prompt prefix 的 KV。

4. **continuous batching**
   - 动态加入和退出请求时，KV 内存管理更灵活。

#### 对你的启发

PagedAttention 的本质是：

```text
KV cache = virtual memory object
KV block = page
request = virtual address space
GPU memory manager = page allocator
```

这是做 KV cache 分层、迁移、远程化之前必须掌握的基础。

---

### 5.2 H2O

- **论文**：H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
- **地址**：https://arxiv.org/abs/2306.14048
- **代码**：https://github.com/FMInference/H2O
- **关键词**：KV cache eviction、heavy hitters、recent tokens、submodular optimization

#### 解决的问题

不是所有历史 token 的 KV 都同等重要。H2O 观察到少数 token 对 attention 贡献很大，称为 heavy hitters。

#### 核心设计

KV cache 保留两类 token：

1. **recent tokens**：保证局部连续性。
2. **heavy hitter tokens**：保证重要上下文不丢。

可以理解成：

```text
保留近期 token + 长期重要 token
丢弃 attention 贡献低的 token KV
```

#### 局限

- 是近似方法。
- 对 needle-in-a-haystack、多跳推理、复杂长程依赖可能有风险。
- 需要任务级评估，不能只看 perplexity。

---

### 5.3 Q-Hitter

- **论文**：Q-Hitter: A Better Token Oracle for Efficient LLM Inference via Sparse-Quantized KV Cache
- **地址**：https://proceedings.mlsys.org/paper_files/paper/2024/hash/bbb7506579431a85861a05fff048d3e1-Abstract-Conference.html
- **PDF**：https://proceedings.mlsys.org/paper_files/paper/2024/file/bbb7506579431a85861a05fff048d3e1-Paper-Conference.pdf
- **关键词**：token oracle、sparse KV、quantized KV、KV cache compression

#### 解决的问题

H2O 之后的问题是：如何更好地判断哪些 token 值得保留，以及如何结合稀疏化与量化。

#### 核心设计

- 用更好的 token oracle 选择重要 KV。
- 结合 sparse KV cache 和 quantized KV cache。
- 目标是在减少显存占用的同时，降低质量损失。

---

### 5.4 KIVI

- **论文**：KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
- **地址**：https://arxiv.org/abs/2402.02750
- **代码**：https://github.com/jy-yuan/KIVI
- **关键词**：2-bit KV cache、per-channel key、per-token value、tuning-free

#### 解决的问题

KV cache 会成为长上下文推理中的主要显存瓶颈。KIVI 试图通过低比特量化降低 KV cache 占用。

#### 核心设计

KIVI 的关键观察：

- **Key cache** 更适合 per-channel quantization。
- **Value cache** 更适合 per-token quantization。

因此它采用非对称 2-bit KV cache quantization。

#### 和 H2O 的区别

| 方法 | 处理方式 | 是否丢 token | 风险 |
|---|---|---|---|
| H2O | eviction | 是 | 可能丢长期依赖 |
| KIVI | quantization | 否 | 量化误差 |
| PagedAttention | memory management | 否 | 基本不改变语义 |

---

### 5.5 KVQuant

- **论文**：KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization
- **地址**：https://arxiv.org/abs/2401.18079
- **代码**：https://github.com/squeezeailab/kvquant
- **关键词**：KV cache quantization、pre-RoPE key quantization、non-uniform quantization、dense-and-sparse quantization

#### 解决的问题

KIVI 已经说明 KV cache 可以低比特化，但 sub-4-bit 精度很难。KVQuant 进一步系统化处理 KV cache 分布特性。

#### 核心设计

1. **Per-Channel Key Quantization**
2. **Pre-RoPE Key Quantization**
3. **Non-Uniform KV Cache Quantization**
4. **Per-Vector Dense-and-Sparse Quantization**

#### 适合关注点

如果你要做长上下文 KV cache 量化，KVQuant 比 KIVI 更细，尤其值得看它如何处理 outlier 和 RoPE 对 key 分布的影响。

---

### 5.6 InfiniGen

- **论文**：InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management
- **地址**：https://arxiv.org/abs/2406.19707
- **USENIX OSDI 页面**：https://www.usenix.org/conference/osdi24/presentation/lee
- **关键词**：dynamic KV cache management、long-text generation、offloading、rehearsal、prefetch

#### 解决的问题

KV cache offload 到 CPU 后，每一步 attention 如果都把全部 KV 拉回 GPU，代价过高。InfiniGen 的目标是只预取真正重要的 KV entries。

#### 核心设计

- 通过 minimal rehearsal 预测下一层需要的重要 token。
- 只从 host memory 中 prefetch 重要 KV。
- 与现代 offloading-based LLM serving 系统协同工作。

#### 对你的启发

InfiniGen 很适合和你设想的“KV cache 远程化 / 分层化”结合。它提供的是：

```text
不要把 KV cache 当成顺序数组全量拉取。
而是做 query-aware / layer-aware / future-aware selective prefetch。
```

---

### 5.7 HeadInfer

- **论文**：HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading
- **地址**：https://arxiv.org/abs/2502.12574
- **代码**：https://github.com/wdlctc/headinfer
- **关键词**：head-wise offloading、CPU RAM KV cache、long context、consumer GPU

#### 解决的问题

长上下文推理时，KV cache 可能比权重更占显存。HeadInfer 将 KV cache offload 到 CPU RAM，同时避免任何一层完整 KV 常驻 GPU。

#### 核心设计

- 按 attention head 粒度管理 KV。
- GPU 上只保留选择性 head 的 KV。
- attention output 动态计算。
- 利用 head independence 降低单步显存峰值。

#### 对你的启发

HeadInfer 说明 KV cache 不一定要以 layer 为最小迁移单位，也可以是：

```text
(layer, head, block) 粒度
```

这对做高效 KV cache migration 非常有意义。

---

### 5.8 LeoAM

- **论文**：Breaking the Boundaries of Long-Context LLM Inference: Adaptive KV Management on a Single Commodity GPU
- **系统名**：LeoAM
- **地址**：https://arxiv.org/abs/2506.20187
- **关键词**：single commodity GPU、GPU-CPU-Disk KV hierarchy、adaptive KV management、variable-sized chunks

#### 解决的问题

当上下文长度继续变大，KV cache 甚至放不进 CPU DRAM，需要使用 disk。LeoAM 研究的是单张 commodity GPU 上的 GPU-CPU-Disk 分层 KV 管理。

#### 核心设计

- 根据 attention 权重偏斜做 variable-sized chunk。
- 用 KV abstract 减少从 disk 传输完整 KV 的开销。
- 使用动态压缩和 pipeline 加速推理。

#### 对你的启发

LeoAM 和 FlexGen 的思路可以合并：

```text
FlexGen: 权重 / KV / activation 都可以分层放置
LeoAM: 长上下文 KV cache 单独做 importance-aware GPU-CPU-Disk hierarchy
```

---

### 5.9 IceCache

- **论文**：IceCache: Memory-efficient KV-cache Management for Long-Sequence LLMs
- **地址**：https://arxiv.org/abs/2604.10539
- **项目页**：https://yuzhenmao.github.io/IceCache/
- **关键词**：semantic token clustering、PagedAttention、hierarchical KV structure、long-sequence LLM

#### 解决的问题

传统 KV cache 选择策略可能只看 token-level importance，容易破坏语义结构。IceCache 试图把语义相关 token 组织在连续内存区域，改善 token selection 和 CPU-GPU 传输效率。

#### 核心设计

- semantic token clustering；
- 和 PagedAttention 结合；
- hierarchical, dynamically updatable data structure；
- 目标是在长序列推理中减少 KV token budget，同时保持精度。

#### 对你的启发

IceCache 把 KV cache 管理从“物理 block 管理”推进到“语义 block 管理”：

```text
PagedAttention: 物理分页
IceCache: 语义聚类 + 分页管理
```

如果未来做 agent memory / long context / CoT 推理，这类方法值得关注。

---

## 6. 路线五：MoE 大模型的低显存推理

MoE 模型总参数量很大，但每个 token 只激活少量 expert。因此 MoE 的低显存推理问题和 dense LLM 不完全一样：

```text
dense LLM: 每层权重大多都会用到
MoE LLM: 总 expert 很大，但每 token 只用 top-k experts
```

### 6.1 MoE-Lightning

- **论文**：MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs
- **地址**：https://arxiv.org/abs/2411.11217
- **关键词**：MoE inference、memory-constrained GPUs、paged weights、CPU-GPU-I/O pipeline、hierarchical roofline model

#### 解决的问题

Mixtral、DBRX 这类 MoE 模型总参数大，低显存 GPU 放不下全部专家。MoE-Lightning 面向 memory-constrained GPUs 做高吞吐 MoE batch inference。

#### 核心设计

- **CGOPipe**：CPU-GPU-I/O pipelining schedule。
- **paged weights**：专家权重分页。
- **HRM**：Hierarchical Roofline Model，用于寻找更高吞吐策略。

#### 适合场景

- Mixtral 8x7B / 8x22B。
- 单 T4 16GB 或少量低成本 GPU。
- 离线 batch inference。

---

### 6.2 MoE-Gen

- **论文**：MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching
- **地址**：https://arxiv.org/abs/2503.09716
- **代码**：https://github.com/EfficientMoE/MoE-Gen
- **关键词**：MoE、single GPU、module-based batching、host memory accumulation、offline inference

#### 解决的问题

传统 continuous batching 面向交互式服务，导致 MoE 的 attention / expert module batch 太小，GPU 利用率低。MoE-Gen 用 module-based batching 在 host memory 累积 token，再动态发大 batch 到 GPU。

#### 核心设计

- 按 module 组织 batch，而不是按 request 组织。
- 在 host memory 中累积 token。
- 动态选择每个 module 的 batch size。
- 尽量 overlap GPU computation 和 communication。

#### 和 FlexGen 的关系

MoE-Gen 继承了 FlexGen 的 throughput-oriented 思路，但把调度粒度从 dense transformer layer 推进到 MoE module。

---

## 7. 路线六：训练 / 微调中的超显存技术

### 7.1 ZeRO-Offload

- **论文**：ZeRO-Offload: Democratizing Billion-Scale Model Training
- **地址**：https://arxiv.org/abs/2101.06840
- **USENIX PDF**：https://www.usenix.org/system/files/atc21-ren-jie.pdf
- **关键词**：training、CPU offload、optimizer state、gradient、single GPU

#### 解决的问题

训练时显存压力不仅来自参数，还来自：

```text
parameters + gradients + optimizer states + activations + temporary buffers
```

ZeRO-Offload 将 optimizer state、gradients 和部分 optimizer computation offload 到 CPU。

#### 核心设计

- 参数和前后向计算主要留在 GPU。
- optimizer state 和 optimizer computation 放 CPU。
- 尽量最小化 CPU-GPU 通信量。
- 保持模型代码基本不变。

#### 适合场景

- 单 GPU 训练 10B 级模型。
- CPU 内存较大，但 GPU 显存有限。
- 想避免复杂模型并行改造。

---

### 7.2 ZeRO-Infinity

- **论文**：ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning
- **地址**：https://arxiv.org/abs/2104.07857
- **SC 页面**：https://sc21.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap464.html
- **关键词**：GPU memory wall、CPU/NVMe offload、extreme-scale training、heterogeneous memory

#### 解决的问题

ZeRO-Offload 主要使用 CPU，而 ZeRO-Infinity 进一步把 NVMe 纳入训练内存层次。

#### 核心设计

- GPU + CPU + NVMe 统一作为训练可用内存。
- 参数、梯度、optimizer states 在不同层次间流动。
- 通过预取、分片、通信重叠缓解 NVMe 带宽不足。

#### 和 ZeRO-Inference 的关系

ZeRO-Inference 可以看作把 ZeRO-Infinity 的异构内存思想迁移到推理场景：

```text
训练：parameters / gradients / optimizer states / activations
推理：weights / KV cache / activations
```

---

### 7.3 QLoRA

- **论文**：QLoRA: Efficient Finetuning of Quantized LLMs
- **地址**：https://arxiv.org/abs/2305.14314
- **代码**：https://github.com/artidoro/qlora
- **关键词**：4-bit base model、LoRA、NF4、double quantization、paged optimizer

#### 解决的问题

完整微调大模型显存太高。QLoRA 的策略是：

```text
冻结 4-bit 量化基座模型，只训练 LoRA adapter。
```

#### 核心设计

- **NF4**：适合正态分布权重的 4-bit data type。
- **double quantization**：进一步压缩量化常数。
- **paged optimizer**：处理反向传播中的显存峰值。

#### 适合场景

- 单卡微调 33B / 65B 级模型。
- 指令微调、领域适配。
- 不需要全参数更新。

#### 局限

- 它不是完整训练。
- 基座模型被冻结。
- LoRA rank、目标模块、数据质量对结果影响很大。

---

### 7.4 LoHan / Fuyou

- **论文**：LoHan: Low-Cost High-Performance Framework to Fine-Tune 100B Model on a Consumer GPU
- **地址**：https://arxiv.org/abs/2403.06504
- **代码**：https://github.com/RC4ML/LoHan
- **关键词**：consumer GPU、100B fine-tuning、RTX 4090、SSD、activation swapping、gradient offloading

#### 解决的问题

QLoRA 是低秩微调路线；LoHan / Fuyou 更关注在单张消费级 GPU + 有限 CPU 内存 + SSD 上进行 100B 级模型微调。

#### 核心设计

- 把 SSD 纳入训练 offload 体系。
- active gradient offloading。
- traffic-aware activation swapping。
- 让 backward、offload、optimizer update 尽量 overlap。

#### 对你的启发

LoHan 对 RL post-training 也有参考意义，因为 RL 训练链路会额外引入：

- rollout engine；
- reference model；
- reward model；
- policy update；
- checkpoint；
- weight sync。

这些状态都可能进入 GPU/CPU/NVMe 分层调度问题。

---

## 8. 早期基础工作：DNN 内存虚拟化与智能 swapping

这些论文不是 LLM 专用，但它们是 FlexGen、ZeRO、LoHan 等路线的系统前史。

### 8.1 vDNN

- **论文**：vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design
- **地址**：https://arxiv.org/abs/1602.08124
- **PDF**：https://www.cs.utexas.edu/~skeckler/pubs/MICRO_2016_vDNN.pdf
- **关键词**：GPU-CPU memory virtualization、training、activation offload

#### 核心思想

vDNN 将 DNN 训练中的中间 activation 在 GPU 和 CPU 之间换入换出，让训练可以超过 GPU DRAM 容量。

#### 价值

它早期提出了一个重要方向：

```text
DL tensor lifetime 可分析，GPU memory 可以虚拟化。
```

---

### 8.2 SuperNeurons

- **论文**：SuperNeurons: Dynamic GPU Memory Management for Training Deep Neural Networks
- **地址**：https://arxiv.org/abs/1801.04380
- **关键词**：liveness analysis、unified tensor pool、cost-aware recomputation、training

#### 核心思想

SuperNeurons 组合了：

- liveness analysis；
- unified tensor pool；
- cost-aware recomputation。

目标是在 GPU DRAM 有限时训练更深更宽的网络。

---

### 8.3 SwapAdvisor

- **论文**：SwapAdvisor: Pushing Deep Learning Beyond the GPU Memory Limit via Smart Swapping
- **地址**：https://dl.acm.org/doi/10.1145/3373376.3378530
- **关键词**：operator scheduling、memory allocation、swap decision、genetic algorithm

#### 核心思想

SwapAdvisor 把三个问题联合优化：

1. operator scheduling；
2. memory allocation；
3. swap decision。

#### 对后续工作的影响

这类工作说明：

```text
offload 不只是内存管理问题，也是算子调度问题。
```

后来的 FlexGen、LM-Offload、MoE-Lightning 本质上都延续了这个观点。

---

## 9. 横向对比

### 9.1 按“搬运对象”分类

| 搬运对象 | 代表系统 | 粒度 | 主要瓶颈 |
|---|---|---|---|
| 权重 | FlexGen、ZeRO-Inference、LLM in a Flash、LM-Offload | layer / tensor / block / neuron | PCIe、NVMe、flash I/O |
| KV cache | PagedAttention、H2O、KIVI、HeadInfer、LeoAM、IceCache | block / token / head / semantic chunk | HBM 容量、CPU-GPU bandwidth |
| activation | vDNN、SuperNeurons、SwapAdvisor、LoHan | tensor / layer / graph node | backward 依赖、recompute cost |
| optimizer state | ZeRO-Offload、ZeRO-Infinity、QLoRA | parameter shard / optimizer shard | CPU compute、通信、NVMe I/O |
| expert weights | MoE-Lightning、MoE-Gen | expert / module / page | expert load imbalance、I/O |
| model blocks | Petals | transformer block | network latency、fault tolerance |

### 9.2 按“调度粒度”分类

| 粒度 | 代表工作 | 说明 |
|---|---|---|
| layer | ZeRO-Inference | 每层权重 stream 到 GPU |
| tensor/block | FlexGen、PagedAttention | tensor placement / KV block paging |
| token | H2O、InfiniGen、Q-Hitter | 选择重要 token 的 KV |
| head | HeadInfer | attention head 粒度 offload |
| neuron | PowerInfer、LLM in a Flash | hot/cold neuron 或 neuron reuse |
| neuron cluster | PowerInfer-2 | 端侧 NPU/CPU/flash 协同 |
| expert/module | MoE-Lightning、MoE-Gen | MoE expert paging 与 module batching |
| semantic chunk | IceCache、SemantiCache 类工作 | 语义组织 KV cache |
| model block | Petals | 多机 pipeline block |

### 9.3 按“是否近似”分类

| 类型 | 代表工作 | 是否改变模型语义 |
|---|---|---|
| 纯内存管理 | PagedAttention、ZeRO-Inference、HeadInfer | 基本不改变 |
| 量化 | FlexGen、KIVI、KVQuant、QLoRA | 有量化误差 |
| eviction / sparsity | H2O、Q-Hitter、InfiniGen | 可能丢上下文信息 |
| activation-aware hybrid | PowerInfer、PowerInfer-2 | 依赖预测器和稀疏算子 |
| 分布式切层 | Petals | 不改变模型，但受网络影响 |

---

## 10. 和你当前系统方向的对应关系

你之前关心的是：

- 低端 GPU 运行超显存模型；
- KV cache 是否能抽象成消息队列或分层缓存；
- 3090 / M1 Mac / 单机大内存 / NVMe 场景；
- SGLang / vLLM / AState KV cache 迁移；
- RL post-training runtime 中权重同步、KV 迁移、checkpoint、故障恢复。

可以按下面方式映射。

### 10.1 如果做“3090 跑大模型”的最小系统

优先组合：

```text
FlexGen / ZeRO-Inference
    -> 权重分层与按层 streaming
PagedAttention
    -> KV cache block abstraction
KIVI / KVQuant
    -> KV cache 低比特压缩
LM-Offload
    -> 性能模型与 pipeline overlap
```

最小架构可以是：

```text
WeightStore:
    CPUWeightPool
    NVMeWeightStore
    GPUWeightCache

KVStore:
    GPUPagedKV
    CPUKVPool
    Optional NVMeKVStore

Scheduler:
    lookahead layers
    prefetch weights
    prefetch KV blocks
    overlap copy/dequant/compute
    evict cold weights/KV
```

### 10.2 如果做“KV cache 远程化 / 迁移”

优先组合：

```text
PagedAttention
    -> KV cache block/page abstraction
HeadInfer
    -> head-wise offload granularity
InfiniGen
    -> query-aware selective KV prefetch
LeoAM
    -> GPU-CPU-Disk hierarchy
IceCache
    -> semantic-aware token/block organization
```

关键设计问题：

1. KV block 的 key 如何命名？
2. KV block 的生命周期如何管理？
3. Prefill 和 decode 是否使用同一套 KV layout？
4. GPU miss 时，是阻塞取回，还是提前 prefetch？
5. Eviction policy 是 LRU、attention-aware，还是 semantic-aware？
6. KV 是否允许量化、近似、丢弃？
7. 和 speculative decoding / prefix cache / radix tree 如何整合？

### 10.3 如果做 RL post-training runtime

训练和推理会同时出现：

- rollout model weights；
- reference model weights；
- reward model weights；
- actor update state；
- optimizer state；
- rollout KV cache；
- checkpoint state；
- weight sync buffer。

对应论文：

```text
推理侧：FlexGen / ZeRO-Inference / PagedAttention / HeadInfer
训练侧：ZeRO-Offload / ZeRO-Infinity / LoHan / QLoRA
系统调度：LM-Offload / SwapAdvisor / MoE-Lightning
```

你的系统可以把这些状态统一成：

```text
StateObject:
    type: weight | kv | activation | optimizer | checkpoint | delta
    owner: GPU | CPU | NVMe | remote
    granularity: layer | block | head | token | expert | shard
    precision: fp16 | bf16 | int8 | int4 | int2
    lifecycle: prefill | decode | backward | optimizer | sync | checkpoint
    access_pattern: sequential | random | repeated | predicted
```

---

## 11. 建议的精读顺序

### 第一阶段：建立主干

1. FlexGen  
   https://arxiv.org/abs/2303.06865
2. PagedAttention / vLLM  
   https://arxiv.org/abs/2309.06180
3. ZeRO-Inference  
   https://www.deepspeed.ai/2022/09/09/zero-inference.html
4. LLM in a Flash  
   https://arxiv.org/abs/2312.11514
5. PowerInfer  
   https://arxiv.org/abs/2312.12456

### 第二阶段：KV cache 专项

1. H2O  
   https://arxiv.org/abs/2306.14048
2. KIVI  
   https://arxiv.org/abs/2402.02750
3. KVQuant  
   https://arxiv.org/abs/2401.18079
4. InfiniGen  
   https://arxiv.org/abs/2406.19707
5. HeadInfer  
   https://arxiv.org/abs/2502.12574
6. LeoAM  
   https://arxiv.org/abs/2506.20187
7. IceCache  
   https://arxiv.org/abs/2604.10539

### 第三阶段：训练 / 微调和 out-of-core runtime

1. ZeRO-Offload  
   https://arxiv.org/abs/2101.06840
2. ZeRO-Infinity  
   https://arxiv.org/abs/2104.07857
3. QLoRA  
   https://arxiv.org/abs/2305.14314
4. LoHan / Fuyou  
   https://arxiv.org/abs/2403.06504
5. SwapAdvisor  
   https://dl.acm.org/doi/10.1145/3373376.3378530

### 第四阶段：MoE 和分布式扩展

1. MoE-Lightning  
   https://arxiv.org/abs/2411.11217
2. MoE-Gen  
   https://arxiv.org/abs/2503.09716
3. Petals  
   https://arxiv.org/abs/2209.01188
4. Distributed Inference and Fine-tuning over the Internet  
   https://arxiv.org/abs/2312.08361

---

## 12. 最后总结

这些论文的共同结论是：

```text
低端显卡不是不能跑大模型，
而是不能用“完整权重 + 完整 KV + 完整 activation 全部常驻 GPU”的方式跑大模型。
```

真正可行的系统需要同时做：

1. **分层存储**：GPU / CPU / NVMe / flash / remote。
2. **细粒度切分**：layer / block / token / head / neuron / expert。
3. **提前预取**：不能等 miss 后再同步拉取。
4. **计算搬运重叠**：copy、dequant、compute、evict pipeline 化。
5. **压缩与近似**：weight quant、KV quant、token eviction、semantic grouping。
6. **性能模型**：offload 策略必须由硬件带宽、latency、batch、kernel 时间共同决定。
7. **任务边界**：离线 batch、交互式 serving、长上下文、MoE、训练/微调是不同问题，不能用一个策略覆盖全部。

对你最有价值的一句话是：

```text
把 GPU 显存当作 cache，然后围绕 cache miss、prefetch、eviction、compression、overlap 设计整个 LLM runtime。
```
