# Chasing 16: How We Rebuilt DeepSeek-V4's Hot Path for GB10

There was a stubborn number on our screen for days: **half a second**.

That was roughly how long one NVIDIA GB10 needed to run an exact, resident, 43-layer DeepSeek-V4 target verification at width eight. Half a second does not sound disastrous for a model this large—until the product target is 16 output tokens per second and every extra millisecond consumes the budget of proposal generation, rollback, and expert I/O.

We did eventually cross the line: **16.1958 verified rows/s at V=8**. But the interesting part was not the final decimal. It was discovering that “replace our CUDA kernels with CUTLASS” was the wrong mental model. The real win came when we stopped optimizing a list of matrix multiplications and started designing model-level dataflow as semantic superkernels.

This is the story of that operator work: what the number 16 actually means, why the first CUTLASS approach was too small, which fusions mattered, which clever ideas made things worse, and what finally pushed the complete target pass over the line.

A necessary caveat before we begin: 16.1958 is a **resident target-verification rate**, not final DSpark accepted-token throughput. It proves that target compute no longer makes the goal impossible. Real proposal acceptance and acceptance-aware I/O are separate gates.

## 1. Where does 16 come from?

Ferrule's goal is **16 accepted output tokens per second**. “Accepted” is the important word.

DSpark does not ask the full target model to produce one token at a time. A smaller proposal path drafts several candidates, then the target verifies those candidates together. Let:

- $V$ be the number of candidate rows verified in one cycle;
- $A(V)$ be the average number of candidates the target accepts;
- $T_{\mathrm{cycle}}(V)$ be the complete cycle time.

The throughput we actually care about is:

$$
R_{\mathrm{accepted}}(V)
= \frac{A(V)}{T_{\mathrm{cycle}}(V)}.
$$

A useful operating point is almost embarrassingly simple:

$$
\frac{4\ \text{accepted tokens}}{0.25\ \text{s}}
= 16\ \text{tokens/s}.
$$

This formula gave us a brutally honest first test. Suppose all selected experts are already resident and storage contributes zero latency. If the target still cannot verify candidate rows faster than 16 rows/s, then no optimistic acceptance model can save the design. We therefore isolated one shape:

$$
1\ \text{sequence} \times V\ \text{candidate rows},
\qquad V \in \{2,4,8\}.
$$

The full system has a second constraint. Our measured storage baseline was about 10.53 GiB/s, so at 16 accepted tok/s the uncovered-read allowance is only:

$$
\frac{10.53\ \text{GiB/s}}{16\ \text{tokens/s}}
\approx 0.658\ \text{GiB/token}.
$$

That is why the number in this article is necessary but not sufficient. Here we ask one focused question:

> Can a single GB10 run the exact 43-layer packed target fast enough to leave 16 accepted tok/s physically possible?

## 2. The trap: every kernel looked fast

Our original CUDA path was correct, and that made the problem harder to see. Open the profiler at the wrong zoom level and everything looked reassuring: the work was on the GPU, Tensor Core kernels appeared in the trace, and no single launch seemed absurd.

Zoom out to one full layer, however, and the pipeline looked like an assembly line where every worker placed the part back in a warehouse before the next worker could touch it:

```text
pack activation
→ launch projection A
→ launch projection B
→ launch normalization
→ launch routing helpers
→ launch gate projection
→ launch up projection
→ launch SwiGLU
→ pack the hidden activation
→ launch down projection
→ reduce routed outputs
```

Nothing in that list is individually unreasonable. The waste lives **between** the lines.

The host launches projection A and projection B separately, so both stage the same activation. A normalized tensor is written to global memory, only to be read immediately by the packer. A Tensor Core finishes its tile quickly, then waits for scalar byte loads to feed the next tile. And some “parallel” small-M kernels hide long serial loops inside one thread.

Three patterns kept returning in Nsight:

1. **We had drawn the API boundary too narrowly.** A GEMM boundary forced model-level intermediate data into global memory.
2. **A launch was not proof of parallelism.** Small-M work could still be serialized inside a thread or warp.
3. **MMA was rarely the whole cost.** Staging, packing, scaling, and task-wave count often dominated the arithmetic.

Our first instinct was conventional: expose CUTLASS GEMM through FFI, then replace each handwritten projection. That would have made several boxes in the trace shorter. It would not have removed the arrows between them.

The turning point was a change of unit. The smallest production FFI call should not be “multiply these matrices.” It should be “perform this meaningful piece of the model.” In Ferrule, we call that unit a **semantic superkernel**.

## 3. CUTLASS was the toolbox, not the architecture

We started this work on CUTLASS/CuTe 4.6.1. CuTe gave us the pieces we wanted—MMA atoms, layouts, `ldmatrix`, block-scaled types, and architecture-specific copy primitives—but it did not decide where our model boundaries should be.

That distinction saved us from replacing one kind of fragmentation with another. Rust still owns:

- CUDA contexts and streams;
- all allocations and reusable workspaces;
- the executable model plan;
- tensor identities and quantization contracts;
- expert frames and KV transactions;
- error handling and scheduling.

The native provider receives a small versioned POD descriptor containing Ferrule-owned addresses and a Ferrule-owned stream. It cannot allocate behind our back, synchronize the host, or escape into a production fallback.

Today that boundary exposes six model-level operations:

1. FP8 QueryA + KV projection;
2. BF16 compressor dual projection;
3. HC producer;
4. shared FFN;
5. stable-frame routed MXFP4 MoE;
6. MLA OutputA → latent pack → OutputB.

“Superkernel” can sound like “put everything into one enormous kernel and hope.” That is not what we mean. The internal phases still have tiles, warps, CTAs, and synchronization. The difference is that those phases now form one device program whose boundary matches the model's dataflow.

FlashAttention is the useful analogy: first choose an algorithmic boundary that eliminates unnecessary traffic; then tile that boundary for the machine. Simply calling several fast GEMMs from the host is not fusion.

We applied the same rule to shape dispatch. The model does not carry four plans named M=1, 2, 4, and 8. It binds one semantic operation. The provider is free to use a tiny-M latency schedule or a larger tiled schedule internally. That keeps hardware tuning where it belongs and avoids turning the model plan into a catalog of launch trivia.

## 4. What we actually fused

The six ABI entries sound abstract, so it is more useful to describe the traffic each one removes.

### 4.1 QueryA + KV: stop preparing the same meal twice

QueryA and KV consume the same packed activation. In the eager path, each projection loaded and staged that activation independently. The fused kernel stages it once, then keeps two accumulator sets—one for each weight matrix.

In the tiled path, one CTA owns an `8×16` output tile. A producer warp moves FP8 weights and activation tiles while a consumer warp performs CuTe FP8 MMA. There is one hardware-specific wrinkle: CUDA 13 rejects the attempted SM120 block-scale encoding for `sm_121a`, so this path uses the forward-compatible dense FP8 `SM89_16x8x32` MMA atom and applies the checkpoint's E8M0 scales explicitly. It is less magical than an all-in-one instruction, but it is correct and fast on the machine we actually have.

### 4.2 BF16 compressor: convert once, project twice

The compressor needs two BF16 projections from one F32 activation. We convert one `8×16` activation tile to BF16 in shared memory and let four warps reuse it for both projections. The implementation uses the `m16n8k16` BF16 Tensor Core atom rather than scalar output-by-output dot products.

### 4.3 HC producer: packing is part of production

The HC path became one semantic operation:

```text
HC mix/split
→ pre-RMSNorm
→ normalized hidden state
→ K128 FP8/E8M0 pack
```

Previously, the normalized values took a round trip through global memory before packing. Now the operation that creates those values also turns them into the K128 FP8/E8M0 representation needed by the next consumer. Packing stopped being an afterthought and became part of the producer contract.

### 4.4 Shared FFN: keep the hidden boundary on device

The shared FFN now performs:

```text
gate + up
→ SwiGLU
→ hidden K128 pack
→ down projection
```

Gate and up share the input producer. Cooperative grid barriers separate the mathematical phases without returning control to the host.

### 4.5 Routed MXFP4 MoE: a device program, not three GEMMs

The routed expert bundle is the most important fusion:

```text
validate stable expert frame/generation
→ gate + up MXFP4 MMA
→ SwiGLU and hidden pack
→ down MXFP4 MMA
→ routed output
```

A cooperative grid executes all phases. Device-wide barriers replace the sequence of “return to host, launch the next stage.” Expert bindings are stable addresses paired with generations, so the kernel can detect a stale resident frame without rebuilding a forest of host-side launch objects.

### 4.6 MLA output: fuse traffic without erasing mathematics

MLA output originally crossed three production dispatches. The new kernel performs:

```text
grouped OutputA
→ BF16-rounded latent F32 boundary
→ latent K128 FP8/E8M0 pack
→ OutputB
→ final F32 output
```

OutputA uses BF16 MMA semantics; OutputB consumes the packed FP8 latent with FP8 MMA. That BF16-rounded latent boundary is part of the numerical contract, so we preserved it inside the launch rather than “optimizing” it away. Fusion should remove avoidable traffic, not silently change the model.

## 5. Following the profiler, one bottleneck at a time

There was no single patch called “make it 16.” Each time we shortened one part of the trace, another cost that had been hiding behind it became visible. The process felt less like implementing a design document and more like peeling an onion—with Nsight telling us which layer to remove next.

### The 40.86 ms elephant

The first routed FP4 path was impossible to ignore: approximately **40.86 ms per layer**. It used GPU instructions, but too much expert work was effectively serialized around them.

Our first real fusion put gate/up, hidden packing, and down projection into one cooperative device program. The number fell to **12.82 ms**. A 3× improvement should have felt satisfying; in the full 43-layer pass, it was merely permission to keep going.

The next profile showed that the cooperative grid itself was too small. At V=8, active expert segments arrived in long task waves instead of filling GB10. We fixed the schedule at 160 CTAs and reached **6.69 ms**. Then we replaced byte-by-byte weight staging with aligned `uint4` transfers. The same mathematics landed at approximately **2.83 ms per layer**—about 14.4× faster than where this kernel started.

```text
40.86 ms  serial phase structure
12.82 ms  one cooperative semantic launch
 6.69 ms  160-CTA GB10 schedule
 2.83 ms  vectorized weight staging
```

The lesson was humbling: writing “MXFP4 Tensor Core” in a kernel does not make the kernel fast. The MMA instruction was only the engine. We still had to build the road, deliver the fuel, and put enough cars on it.

### The Tensor Cores were waiting for bytes

The same pattern appeared elsewhere. Generic FP8 MMA fell from **0.844 ms to 0.290 ms** after weight staging changed from scalar bytes to vector loads.

The generic BF16 path had a deeper problem: one scalar dot product was computed per output. Replacing that structure with an `m16n8k16` Tensor Core tile changed the measured average from **1.085 ms to 0.340 ms**.

These two changes gave us a rule that survived the rest of the project:

> Before changing the mathematics, count the instructions and memory transactions required to feed one MMA tile.

### One activation, two projections

The formal QueryA+KV and compressor kernels were then vectorized and reorganized around their shared activation producer:

| Operation | Before | After |
|---|---:|---:|
| FP8 QueryA + KV | 0.980 ms | 0.163 ms |
| BF16 compressor pair | 0.669 ms | 0.339 ms |

QueryA+KV improved by roughly 6×, but the important observation was architectural: this was not “two faster GEMMs.” It was one activation producer feeding two consumers. The compressor followed the same idea—convert the tile once, reuse it twice.

### Then packing became the wall

Once projection time fell, the supposedly secondary packing kernels stepped into the spotlight.

The shared FFN originally assigned one thread a serial loop over an entire K128 block. We changed it to one warp per block: warp reduction computes `amax`, and each lane quantizes four values.

The HC producer had the same shape in a more expensive location. Its FP8 pack initially used only a small fraction of available threads while each active thread walked K128 serially. The final version divides every K128 block among an eight-thread subgroup. All 256 CTA threads cover the 32 scale blocks concurrently; subgroup shuffles reduce `amax`, lane zero publishes the E8M0 scale, and each lane quantizes 16 values.

This was the kind of optimization that is easy to miss if one only benchmarks GEMM. It was also the final nudge: after the HC subgroup pack landed, the complete V=8 sweep stayed above 16 rows/s.

### The full pass always had the final vote

The last full Nsight profile before the final HC subgroup change showed this approximate distribution:

| Kernel group | Average time | Share of profiled GPU time |
|---|---:|---:|
| routed FP4 MoE superkernel | 2.848 ms | 34.7% |
| MLA output superkernel | 1.302 ms | 15.9% |
| shared FFN superkernel | 0.704 ms | 8.6% |
| BF16 MMA | 0.336 ms | 7.6% |
| HC producer | 0.296 ms, twice per layer | — |
| BF16 compressor | 0.339 ms | — |
| FP8 QueryA + KV | 0.163 ms | — |
| generic FP8 MMA | 0.223 ms | — |

The table was useful for deciding where to look next, but none of these isolated numbers was the gate. The final judge was always wall-clock time for the complete resident verification path, output head included.

That discipline mattered because several ideas looked excellent in a code review and lost in the real workload.

## 6. The good ideas that made it slower

A benchmark-driven project accumulates a small graveyard of beautiful ideas. Ours was surprisingly educational.

### Compacting active FP4 segments

We built a compact active-segment list to avoid scanning sparse state. It was tidy, easy to justify, and almost perfectly useless: 6.694 ms became 6.686 ms, while end-to-end time became slightly worse. We deleted the machinery.

### Sharing one MLA activation across four warps

This was the most seductive experiment. Load the activation once, share it across four warps, and let one CTA cover 64 channels—what could be more obviously efficient? On GB10, the lost warp-level parallelism cost more than the reused load saved. V=8 fell from about 15.11 to 14.37 rows/s. We reverted it.

### Wider FP4 activation staging

Combining two bytes into `uint16_t` activation/scale loads should have reduced instruction count. Instead, V=8 dropped to roughly 14.25 rows/s. The narrower access pattern interacted better with the actual layout and generated code, so the change was removed.

### Manual OutputB scale deduplication

We cached activation scales in explicit local arrays. The compiler had already performed enough broadcast/common-subexpression work; our arrays increased register pressure and made the complete pass slower. Reverted.

### Mistaking OutputB for BF16

An early MLA design assumed OutputB was BF16. The real checkpoint is FP8/E8M0. Because the provider is fail-closed, shape/layout validation exposed the mistake immediately instead of silently executing a fallback. The final semantic kernel uses BF16 semantics for OutputA, packs the latent to FP8/E8M0, and uses FP8 MMA for OutputB.

These failures changed our coding style. We became less impressed by locally elegant reuse and more willing to delete a day's work when the complete pass disagreed.

## 7. The run that finally crossed 16

After the HC subgroup packing change, we ran the resident sweep three times. These were the final measurements:


| Verification width | p50 target time | p95 target time | p50 verified rows/s |
|---:|---:|---:|---:|
| V=2 | 0.433132 s | 0.448342 s | 4.6175 |
| V=4 | 0.418125 s | 0.419454 s | 9.5665 |
| V=8 | 0.493955 s | 0.497114 s | **16.1958** |

Every width reported:

- numerical parity under the current batched-vs-token-loop CUDA contract;
- zero selected expert I/O during the measured resident pass;
- zero steady-state device allocations.

At p50, V=8 gives the number in the title. Even the p95 sample stays on the right side of the line:

$$
\frac{8}{0.497114\ \text{s}}
\approx 16.09\ \text{verified rows/s}.
$$

For the first time, resident target compute no longer rejected the one-Spark headline.

There is an honest footnote. V=4 still takes about 418 ms, so the old V4/A4 250 ms operating point does not work. And verified rows are not accepted tokens: the real proposal source, acceptance distribution, complete cycle, and expert I/O still have to earn the final 16. What this run bought us was not a finished SOTA claim. It bought us the right to continue.

## 8. Fast is useless if it changes the model

Fusion is where performance work can quietly become model surgery. Moving a rounding boundary, applying the wrong scale, or packing one intermediate in the wrong format may produce plausible logits while changing the generated token later.

So every optimization had to return through the same correctness gate:

- CUTLASS provider ABI and semantic-kernel tests;
- dynamic-M tests, including a 4,097-row cross-tile case;
- routed FP4 MoE smoke tests;
- FP8/BF16 MMA smoke tests;
- 43-layer packed CUDA versus token-loop CUDA parity, with zero maximum absolute difference at every layer and matching checkpoints at layers 1, 5, 23, and 43.

The goal was not to preserve every accidental intermediate bit pattern from an old implementation. The goal was to make numerical boundaries explicit and preserve model-visible behavior. That is why the MLA kernel keeps its BF16-rounded latent boundary even though erasing it would make the fusion look cleaner.

## 9. A few implementation details worth stealing

The source is organized around a deliberately small native boundary: one versioned semantic ABI, one GB10 manifest, and one implementation unit for each fused operation. Rust validates storage and shapes before crossing that boundary; the native side validates the ABI, target, alignment, and exact artifact contract again.

A few details are especially reusable:

- aligned `uint4` staging feeds MMA tiles instead of scalar byte loops;
- CuTe exposes BF16, dense FP8, and block-scaled MXFP4 MMA atoms under one pinned toolkit;
- cooperative launches use device-wide barriers for mathematically ordered phases;
- stable expert addresses plus generations make routed fusion compatible with residency;
- caller-owned scratch keeps the hot path allocation-free and graph-ready;
- provider-private M scheduling keeps model plans semantic rather than shape-specialization catalogs;
- every `can_implement` path validates ABI, target, alignment, and exact artifact shape.

## 10. The machine behind the numbers

All measurements came from the same narrow target:

| Component | Version / configuration |
|---|---|
| System | NVIDIA DGX Spark |
| SoC / GPU | NVIDIA GB10 Grace Blackwell |
| GPU target | compute capability 12.1, `sm_121a` |
| CPU | 20-core Arm |
| Unified memory | 128 GB nominal, 127.6 GB available |
| OS | Linux 6.11.0-1014-nvidia, aarch64 |
| NVIDIA driver | 580.82.09 |
| CUDA / NVCC | 13.0 / 13.0.88 |
| CUTLASS / CuTe | 4.6.1, commit `e05f953a5b3d38adc240df2ff928e0421c2abba3` |
| Rust / cargo-oxide | 1.96.0-nightly / 0.2.1 |
| Model | `DeepSeek-V4-Flash-DSpark`, 43 target layers, 48 shards |
| Checkpoint / routed experts | approximately 156 GiB / 137.1 GiB |

The specificity is intentional. This is not a portable kernel claim. Ferrule's optimized provider targets GB10/`sm_121a`; unsupported hardware or missing semantic plans fail explicitly.

### Reproducing the build

CUTLASS is header-only in this integration, so there is no separate library build. Ferrule fetches the exact checkout, then its offline Cargo build step invokes NVCC to compile the native provider and included CuTe/CUTLASS kernels for `sm_121a`.

```bash
just cutlass-setup
just build-cuda
just test-cutlass-provider
```

`just build-cuda`, provider tests, and CUDA run recipes already depend on `cutlass-setup`, so a normal GB10 build does not require a manual clone. The bootstrap verifies both CUTLASS 4.6.1 and commit `e05f953a5b3d38adc240df2ff928e0421c2abba3`. Cargo's build script remains offline by design; network acquisition belongs to the explicit, reproducible setup step.

The published number uses three resident width-sweep iterations over all 43 target layers, includes the output head, disables expert prefetch during measurement, and requires zero selected expert I/O and zero steady-state allocation.


## 11. What stayed with us

1. **Fusion begins at the semantic boundary.** Host-side composition of faster GEMMs is not a superkernel.
2. **Tensor Core instructions do not guarantee Tensor Core performance.** Staging, task waves, and occupancy decide whether the hardware is fed.
3. **Packing is first-class compute.** Once GEMMs improve, serial quantization and scale reduction become the wall.
4. **Small M is a scheduling problem, not a model-plan taxonomy.** Keep shape dispatch inside the provider.
5. **The complete pass is the benchmark.** Several locally elegant changes lost end to end and were deleted.
6. **Fail-closed validation accelerates optimization.** Wrong assumptions surface immediately instead of hiding behind fallback behavior.
7. **Six semantic launches are more valuable than dozens of reusable micro-operators.** Reuse should live in CuTe helpers and internal templates, not in a fragmented production FFI.

Crossing 16 did not finish Ferrule, but it changed the nature of the problem. We no longer have to ask whether the 43-layer target kernel makes the goal impossible. One GB10 can verify eight resident candidate rows at more than 16 rows/s.

Now the burden moves outward: can the real proposal path produce enough accepted tokens, and can expert residency keep uncovered I/O inside the budget? Those are harder system questions. At least the hottest compute path has finally stopped answering “no” before we begin.

## 12. Come watch us try

Ferrule is our attempt to build a transparent, Rust-native inference runtime for models that are usually treated as too large or too awkward for one small machine. We care about the unglamorous details—quantized layouts, expert residency, transactional KV state, exact speculative verification, and kernels whose benchmark includes everything they actually cost.

The project is still in the interesting phase: the resident target has crossed its first compute gate, while real DSpark acceptance and the long-run I/O budget remain open. In other words, there is enough working code to inspect, enough raw evidence to challenge, and still plenty of difficult engineering left to do.

If this kind of systems work interests you, come follow Ferrule, read the measurements, question the assumptions, profile the kernels, or simply watch the next gate fall. Stars, issues, experiments, benchmark comparisons, and skeptical reviews are all welcome.
