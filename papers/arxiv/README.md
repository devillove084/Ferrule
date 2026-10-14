# Ferrule arXiv 论文工作区

这是**本地、未提交、被 Git 忽略的英文论文初稿**，不是已经完成实验或可直接投稿的论文。建立于 2026-10-02。

## Current manuscript argument and integrated figures (2026-10-14)

The TeX manuscript is now organized as a progressive argument, not a list of
independent mechanisms. This writing pass changes manuscript content only;
figures, experiments, receipts, checklist, and bibliography are not edited.
No GPU query, server launch, or numerical rerun is part of this pass.

| Order | Manuscript file | Argument and evidence link |
|---|---|---|
| 1 | `sections/introduction.tex` | Problem, RQ1–RQ4, scoped contributions |
| 2 | `sections/scope.tex` | One symbol table; physical `d/B_d`, accounting `a/b_a`; representations, working sets, quality/completion contracts |
| 3–5 | `sections/design.tex` | Normal walkthrough → intended invariants → typed `algorithm`/`algorithmic` pseudocode |
| 6 | `sections/implementation.tex` | Contract-to-code mapping and profile integration limits |
| 7 | `sections/evaluation.tex` | Setup → RQ1 capacity → RQ2 request I/O → RQ3 scheduling → RQ4 custody → separate resident supplement |
| 8 | `sections/related-work.tex`, `sections/conclusion.tex` | Prior work → limitations → conclusion |

Configured limits, reported samples, and continuous physical peaks are different
quantities. The credit partition is an **intended invariant**, not independently
measured conservation. TP1 `-02` remains `n=1` DEBUG exploration: four HTTP
successes, three limited reference canaries plus Hello2 HTTP/usage only, logical
bytes not physical I/O, no generated IDs/full numerical quality, final resource
contract unknown. RQ3's same-contract safe-blocking reference is not implemented.
TP2 test numerical evidence and resident vLLM timings remain separate; the latter
retain `fairmatch=false` / `exact_speedup=null` and are not fair speedup evidence.

The later 2026-10-14 current StorageMiddle64 full40 gate stopped before
inference at `cuInit`, `CUDA_ERROR_NOT_INITIALIZED (3)`: zero observed numerical
metrics/logits, not a numerical mismatch. Fresh-quality/cold-owner were unissued,
with no retry. The receipt is `target/validation/storage-middle64-current-quality-20261014-01/FINAL.json`,
not the older TP1 `-01` `DEVNULL` wrapper failure. Owner initialization remains
under investigation; quality/serving/resource gates are not complete.

### Current plot-integration validation

- `make -C papers/arxiv`: **PASS**; `build/main.pdf` is **30 pages,
  725,591 bytes** after the current-gate chronology and algorithm/contract
  audit. The converged log has
  no LaTeX/package warnings, undefined references/citations, overfull/underfull
  boxes, oversized floats, or duplicate labels. All four experiment PDFs are
  actual included assets.
- Read-only `summarize.py --check`: **PASS**, 41 sealed input hashes,
  39 receipt pins, zero divergences. Read-only `plot_storage_demand.py --check`:
  **PASS**, all four PDFs/PNGs and their contact sheet/data/guide/manifest,
  30 anchored invocations, later entry-balance residuals zero. Neither check
  rewrote inputs, artifacts, or manifests.
- Inline checks matched the 184,560-byte `H-L` aggregate, implied/unmeasured
  initial entries, per-request invocation counts, the cache-charged peak governed
  by its 320 MiB accounting ceiling, the unknown total physical-owner peak/ceiling,
  and separate TP2 receipt selections: **33 / 35,937** from
  `model-runtime-quality.json`, versus **37 / 40,293** in the distinct
  `production-opts-01` optimized-release summary. No new/anticipated GPU PASS
  or serving/numerical equivalence result was written.
- Ghostscript 9.54.0 rendered the full paper to build-only PNGs and extracted
  text. The revised audit was checked at paper text width, including the
  expanded algorithms and experiment plots. No diagram connector/text or
  caption occlusion, clipping, oversized float, or incomplete algorithm was
  observed.
  The shared palette has named nodes/swatches; solid dependencies and dashed
  evidence/Unknown encoding remain distinguishable without relying on color.
- Source writes are limited to `main.tex`, manuscript sections, and this README;
  normal paper build/review outputs are under `build/`. No production, figure
  source/PDF, experiment, raw receipt, plot manifest, checklist, or bibliography
  was edited. No GPU query, server, or numerical rerun was performed.

### Protocol audit revisions

- Capacity inequality is necessary, not sufficient: intrinsic rejection,
  temporary occupancy waits, canonical/authority checks, and policy-specific
  execution reserves are distinct.
- Publication is explicitly Ready / Blocked / Rejected. Ready resolves exact
  waiters after resident-grant transfer. Rejection keeps the journal and custody
  for cleanup. Identity-mismatched events are rejected;
  matching validated `CompletionOutcome::Stale` enters terminal cleanup.
- Read custody return precedes explicit read-slot/storage-read credit release;
  pinned host charge remains live through upload or proven cleanup.
- Cleanup snapshots `Omega` after owner-accepted demand detachment, preserves
  shared ownership/ACK-handoffs, and visits at most `k` items per pass while
  persistent records retain unvisited/unresolved obligations and a rotating
  cursor for wake/revisit. New work cannot overtake the unvisited cohort;
  examination fairness is a requirement, not measured starvation freedom.
- Pending/Idle requires a positively known original pollable obligation;
  otherwise Unknown custody is retained. Secondary cleanup errors preserve
  primary outcome, phase, exact arguments, and custody without irreversible
  ending redispatch; partial returns/releases are not applied twice.
- Exact-stage cancellation markers prevent accepted cancellation redispatch.
  Provider acceptance is not quiescence. Unknown never resets: a late exact
  registry completion may close its named stage without clearing owner poison.
- `cache.peak_bytes` (raw key `expert_owner_peak_bytes`) is resident + pending
  + scratch charged-byte high-water, governed by
  `ExpertCacheLimits.max_bytes` = 320 MiB, not physical-owner memory.
  Independent physical-owner peak/ceiling remain unobserved.
  No implementation verification or nonblocking/progress guarantee was added.

### Integrated experiment figures

The final manuscript directly includes the four generated storage-demand PDFs.
Captions and plotting semantics are tied to `experiments/storage_demand/analysis/output/PLOTS.md`:

| Asset relative to `papers/arxiv` | LaTeX label | Equation / evidence boundary |
|---|---|---|
| `figures/storage-demand-logical-transfer.pdf` | `fig:storage-demand-logical-transfer` | `eq:request-io`; `L` versus `H` and their difference are application counters, not disk or bus traffic |
| `figures/storage-demand-cache-reconciliation.pdf` | `fig:storage-demand-cache-reconciliation` | `eq:entry-reconciliation`, `eq:cache-charge`; initial zero is implied, gauges are not a complete ledger |
| `figures/storage-demand-invocation-timeline.pdf` | `fig:storage-demand-invocation-timeline` | 30 log-order invocation boundaries; occupancy is sampled at boundaries, uploads are not cache misses |
| `figures/storage-demand-timing.pdf` | `fig:storage-demand-timing` | Startup and request bars have separate origins; `n=1` DEBUG, no CI or kernel timing |

The existing three colored mechanism diagrams remain at their same PDF paths and
labels. Their captions retain the shared planning/storage/execution/publication/
Unknown palette, labeled swatches, dashed evidence gates, and profile-scoped
non-claims; no figure source or PDF was changed in this manuscript pass.
Building is CPU-only via `make -C papers/arxiv`; the installed `algorithm` and
`algorithmic` packages are used, not `algpseudocode`.

---

## 目录与版本控制

仓库根 `.gitignore` 已有 `/papers/`，覆盖本目录，因此没有重复修改忽略规则。普通 `git status` 和提交不会保护这些文件；切换机器、清理 ignored 文件或删除工作区前，应另行备份。不要对本目录执行 `git clean -fdx`，也不要为了保存论文而把整个 dirty 工作区强制加入 Git。

```text
papers/arxiv/
├── README.md
├── EVIDENCE_LEDGER.md       # 本地事实来源、限制和下一步
├── PAPER_COMPLETION_CHECKLIST.md # 论文补齐工作清单
├── Makefile                # 仅构建论文，不运行实验
├── main.tex
├── references.bib          # 未加入任何未经核实的引用
└── sections/
    ├── introduction.tex
    ├── scope.tex
    ├── design.tex
    ├── implementation.tex
    ├── evaluation.tex
    ├── related-work.tex
    └── conclusion.tex
```

PDF 构建后位于 `build/main.pdf`。现有彩色概念图与四幅实测收据派生图位于 `figures/`；实际 include 路径、label 与范围见上方说明。所有结果表均保留其冻结收据范围，不把 placeholder 当作结果。

## 核心目标、标题、作者与设计范围

核心目标是 **storage-backed inference when the model exceeds deduplicated usable physical residency capacity**：当模型超过部署可用、去重后的物理驻留内存容量时，利用 backing storage 支撑按需推理。容量定义不预设离散 GPU+host；离散 GPU 与主机内存只是一个实例，统一内存或共享物理池必须按实际物理归属计数，不能重复相加。

1. **目标：** 验证模型超过去重物理驻留容量时的推理可行性，明确权重表示、物理池、去重规则、存储访问及数值验收。
2. **手段：** 用精确的事务性质资源管理连接 admission、按需物化、continuation、KV publication 与 retirement，在纳入账本的资源域内实施准入，区分 logical terminal、physical quiescence 与 retained custody。事务 custody 是安全重用容量的机制，不是独立主问题。
3. **设计范围：** 后续覆盖全模态路径和全异步执行；当前不能声称已验证全模态，也不能声称全链路无阻塞。异步提交/完成不等于已证明实际执行 overlap，二者必须分别建立证据。
4. **评价：** 论文以 E0 容量实验为第一主实验；当前工程执行先争取 vLLM 同等或更好性能，E2 升 P0，再补齐容量与机制清单。两条验收不互相替代；机制评价覆盖安全重用、I/O、延迟、吞吐、实际 overlap 及正常路径开销。事务性质不等于跨崩溃持久事务，协议精确性也不等于数值 exactness 或性能保证。

确定标题为：

> **Ferrule: Storage-Backed Inference Beyond Memory Capacity**

作者：**Lu Huanbing**；单位：**Independent Researcher**；联系邮箱：**luhuanbing084@gmail.com**。ORCID 与资助信息未提供，不推测、不填写为“无”。

历史建议保留在 `../../docs/temp_ferrule_refactor.md` 的论文讨论（约 13558–13858 行）和恢复摘要（约 14568–14616 行）。当前目标以本文为准；证据与缺口见 [EVIDENCE_LEDGER.md](EVIDENCE_LEDGER.md)，测量合同见 [性能协议](../../docs/PERFORMANCE_EVALUATION.md)。历史标题和术语不替代上述通用容量目标。

标题、作者、正文主线与 [PAPER_COMPLETION_CHECKLIST.md](PAPER_COMPLETION_CHECKLIST.md) 工作清单已同步；当前容量、多模态、统一内存和全异步范围仍须逐 profile 验收。

## 已经写入什么

- 英文摘要、引言与动机场景。
- 资源域、准入条件、精确性分层、故障与活性假设。
- exact-key materialization、资源 custody、分阶段 KV publication/retirement、取消与未知完成。
- 实现边界、E0 容量主实验与 E1–E6 支撑实验方案、测量单位、质量门禁与 artifact 要求。
- 相关工作的比较维度和有明确限制的阶段性结语。

初稿中的资源公式是**抽象记账条件**，不是形式化验证结论；当前实验章节区分实际冻结的 logical-demand/图表观察与尚未完成的证据门禁。候选贡献尚未通过主文献比较确立 novelty。没有捏造作者、引用、实验数据、置信区间或性能优势。

## 容量定义与最新证据边界

**核心容量 claim 当前尚未建立完整证据，完整的磁盘必要性证据也未确认。** 不能借用全驻留 EP8、host-only 运行，或全部紧凑权重仍可驻留某个内存池的 offload 结果来证明这一目标。

容量报告必须区分：

- **磁盘 checkpoint 大小：** 指定文件及编码的逻辑字节与实际占用；不直接等于内存需求。
- **紧凑权重全驻留需求：** 指定量化/压缩表示及 scales 等必要数据能否全部留在可用物理内存中；必须显式检查，而不能从执行展开大小推断。
- **执行表示与全驻留权重需求：** 按实际解码、反量化、packing、padding、共享及复制规则，报告各表示与全部权重同时驻留的需求。执行展开字节超过可用物理驻留容量，但紧凑权重仍能全驻留时，不足以证明必须使用硬盘。
- **去重的可用物理驻留容量：** 按实际物理 pool 归属报告可用于驻留的容量，同一物理池或共享页只计一次；占用独立物理存储的副本仍各计其实际占用。离散 GPU 与 host 只是一个实例。统一地址或 managed-memory API 本身不证明底层只有一个物理池，必须核对实际拓扑、迁移及驻留情况。
- **总工作集：** 实际驻留权重加 KV、activations、workspace、staging、在途及 retained buffers、metadata、固定分配和相关 OS page cache；不能只用账本占用代表部署总内存。

首要实验必须绑定表示与算术合同，报告物理内存池、去重规则和实际施加预算、逐池高水位、cache/swap 条件、请求期物理存储读取，以及对应模态的端到端推理完成与数值/质量验收（当前语言解码实例包括 prefill 和多步 decode）。离散 GPU+host 只是一个配置实例，不是通用定义；人为限额与真实物理容量不足应分别表述。仅总工作集因 KV/并发超限、仅出现非驻留请求或仅成功加载模型，都不足以建立模型超过可用物理驻留容量的核心 claim。

不要直接将旧会话中的测试/吞吐抄入论文。2026-10-08 按 [CURRENT_ACCEPTANCE](../../docs/CURRENT_ACCEPTANCE.md)（2026-10-07）及其冻结收据同步；[性能协议](../../docs/PERFORMANCE_EVALUATION.md) 保留测量方法和历史限制。当前实际 CUDA 解码证据保留其设备、模态和 profile 局限：

- 最新 adapter-extraction F32 收据为 6/6 PASS：代表性 PP2TP2EP2 及分别测试的 resident thread/process EP8，合计 7,946,240 logits；不是全部四拓扑重验，pipeline 不含 raw route/state oracle。旧 process-only PASS 保留原范围，不转借；所有结果均非当前工作树或性能通用保证。
- 单-owner Mixed prepared-attention 原 full40 804 metrics / 1,986,560 logits，bounded fresh QA/arithmetic/Chinese 5,427 metrics / 9,187,840 logits elements PASS；原 reference/阈值未改。非 bit-exact、非广泛任务质量，context<=1024；BF16 activation/KV/conv 与 F32 accumulation/SSM 不能简称全 BF16。
- 较早 Mixed public serve 只有 Hello lifecycle PASS，source/binary 不同；不能与新 attention quality 收据拼成最新 serving binary 的质量/性能通过。原始失败、post-GPU CPU-only 变化及各自 pins 见 ledger。
- **W8A16 独立 full40 oracle 已存在，但 strict acceptance failed**，涉及 logits、route 和 state；不能再写 “oracle 缺失”，也不能用 argmax 一致替代严格验收。
- W8A16 timing 可作为带限制的历史观察，不是 quality-qualified performance acceptance。
- 相同 benchmark client 不等于相同服务端；不同精度/拓扑的 vLLM 部署对比不是公平的 runtime speedup。
- 本轮只做资料核对和写作，**没有重跑 GPU 实验或数值测试**。

详细路径与当前事实见 [EVIDENCE_LEDGER.md](EVIDENCE_LEDGER.md)。历史计划与新评估冲突时，先追踪原始 artifact 和版本，不通过选择更好看的结果消除冲突。

## 此前运行计划（2026-10-08；已由下方 TP2 收据更新）

先做 vLLM 性能工程，再补齐论文清单；E2 升 P0，E0 仍是未完成的 primary paper claim。具体阶段、预注册判定和条目映射见 [checklist 第 0 节](PAPER_COMPLETION_CHECKLIST.md)。论文专用 runner 入口为 [experiments/README.md](experiments/README.md)（现有 runner 与收据说明）；本次只改这三份 Markdown，不创建或修改 experiments，不在根 `scripts/` 新增论文专用脚本。

首个单 GPU TP1 Mixed/vLLM 比较是 **boundary-aligned system-level 候选，不是 arithmetic-equivalent runtime 对比**；双方 effective dtype、F32 SSM/BF16 conv/KV、质量、实际工作量和资源均需核实。soft 7 GiB 策略与 observed usage/峰值不能写成 hard cap，旧 Ready 观察不能冒充峰值；本次无新实测。完整紧凑权重能放进 host RAM 时，不证明论文 storage necessity。

首轮 p128/o32/c1、context160：2 个独立配对 block、每侧每 block first-after-Ready 1 / warmup 2 / measured 2，只作描述性 pilot。后续确认计划至少 5 blocks，采用预注册 paired 95% CI；默认不允许性能退让，吞吐比和 CI 下界 >=1 才通过至少达到 vLLM 的工程门禁，严格优势需下界 >1。“差异不显著”不是相等证明，不擅自引入正的非劣效 margin。完整方法、失败处理、延迟/SLO 和扩展阻塞见 checklist；代码存在不构成验收。

## TP2 resident vLLM baseline（2026-10-09；partial E2 evidence）

The successful lifecycle receipt is [`tp2-resident-single-lifecycle-20261009-001`](experiments/tp2_resident/RESULTS-single-lifecycle.md), with the machine-readable [`RESULTS.json`](experiments/runs/tp2-resident-single-lifecycle-20261009-001/RESULTS.json). It is a descriptive vLLM **TP2/PP1/DP1, no-EP, no-CPU-offload Marlin baseline**. One server start handled Hello2, all three frozen canaries, and QA8.

- Startup: **65.7325 s**. Hello2 returned `, I` for 1 input and 2 output tokens: TTFT **230.452 ms**, E2E **311.259 ms**.
- All 3 canaries matched their frozen visible prefixes and usage counts. This is only a visible-text/usage gate; token IDs, logits, routing, state, and full numerical equivalence were not observed.
- QA8 was **5/5 successful** measured requests (one excluded first QA-phase request and two warmups): mean TTFT **130.66 ms**, mean E2E **703.34 ms**.
- The five measured requests produced 40 actual output tokens in **3.5363 s** client wall, approximately **11.31 tok/s**. Client overhead is included; this is not a cold-cache, larger-load, or paired throughput result.
- Sampled peaks were **19.284 GiB per GPU** and **6.879 GiB server-tree PSS**. These are sampled whole-device/PSS observations, not exact peaks or a hard memory partition.

Ferrule now has test-only physical PP1/TP2/EP1 logits/lifecycle evidence: the shared/dense separate gate/up fix passed `shared-full40-01` (1,986,560 exact finite Hello/Capital prefill + one-decode logits) and `shared-tiny-01`; see the [CPU receipt audit](experiments/tp2_resident/claim-gates-audit-20261009.json). All-boundary/state, multistep and public serving CLI admission remain incomplete. HF-eager logical TP2 is not Marlin arithmetic. Therefore this baseline makes **no Ferrule speedup, fair-match, parity, or arithmetic-equivalence claim** (`fairmatch=false`, `exact_speedup=null`). E2 remains P0 with partial baseline evidence, not a completed checklist item.

The effective logs are profile-specific and must not be conflated: the TP2 server selected `MarlinFP8ScaledMMLinearKernel` and `MARLIN Fp8 MoE`; GDN prefill used Triton/FLA, while decode fell back to Triton GDN because `fused_gdn_decode_post_conv_mtp` was not built. NCCL logged `NCCL_P2P_DISABLE=1` and local channels `via SHM/direct/direct`; Socket plugin initialization is not treated as proof of Socket payload for local GPU traffic. The separate M7/G10 UVA profile logs record `UVAOffloader` and 30 GiB CPU-offloaded parameters ([M7 log](experiments/runs/m7-feasibility-exclusive-002/vllm/server.log), [G10 log](experiments/runs/g10-exclusive-001/feasibility-vllm/server.log)); UVA is not part of this TP2 no-CPU-offload baseline. These backend, offload, fallback, and channel observations do not establish arithmetic equivalence.

Failed startup attempts remain preserved: baseline-delta runs 003 and 004 did not reach readiness and issued no requests (observer `nvidia-smi` timeout in 003; worker CUDA driver initialization failure in 004). Earlier runs 001/002 also remain append-only in [`RESULTS.md`](experiments/tp2_resident/RESULTS.md): their canary stages aborted before HTTP, so they are not numeric quality failures or passes. The successful receipt retains `owned_gpu_attribution=unknown`; server/worker PIDs are local process identities, not claimed host-NVML mappings, and one incomplete leaf host sample kept PSS unknown. See [`RESULTS-baseline-delta.md`](experiments/tp2_resident/RESULTS-baseline-delta.md) and the [evidence ledger](EVIDENCE_LEDGER.md).

Next required: complete all-boundary/state and multistep acceptance, admit a public Ferrule TP2 serving CLI, and bind full-model quality/lifecycle evidence to that serving binary on the same topology and dtype/activation/KV/conv/SSM/state/EOS/cache/workload; only then perform fresh paired runs and independent-block statistics. The [CPU-only pairing preflight](experiments/tp2_resident/CURRENT.md) does not launch or invent CLI flags. The 11.31 tok/s baseline is not an optimum or statistical conclusion. Do not tick parity or E2 completion until those gates pass.

## 构建

使用标准 `article`，不冒充 arXiv 官方模板，也不依赖特定会议类或下载的第三方样式。源码为 ASCII 英文；中文写作说明不参与 TeX 编译。

需要含以下组件的 TeX 环境：pdfLaTeX、latexmk，以及常见宏包 `geometry`、`fontenc`、`inputenc`、`lmodern`、`microtype`、`amsmath`、`amssymb`、`booktabs`、`tabularx`、`algorithm`、`algorithmic`、`graphicx`、`xcolor`、`hyperref`。当前参考文献构建还需要 BibTeX。

在仓库根目录执行：

```sh
make -C papers/arxiv
```

或进入本目录执行 `make`。清理本论文的生成文件：

```sh
make -C papers/arxiv clean
```

**上版已完成编译记录（按前轮实际执行结果同步）：** `make -C papers/arxiv` 退出码为 **0**，经过 **2 轮**编译，生成 **9 页、183734 bytes** 的 PDF，最终无残余警告。这是上版的历史构建结果，不移作当前版本验收。

**历史源码构建验证（2026-10-02，非本次执行）：** `make -C papers/arxiv` 退出 **0**，初次为 up-to-date；`make -B -C papers/arxiv` 仍未触发 pdfLaTeX。随后在 `papers/arxiv/` 执行 `latexmk -g -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex`，退出 **0**，实际运行 **1 轮 pdfLaTeX**；再次 `make -C papers/arxiv` 退出 **0**。所有构建调用均设 120000 ms 限时且未超时。最终 `build/main.log` 报告 **15 页、299612 bytes**，PDF 文件大小核对一致；无 LaTeX/Package warning、overfull、未定义引用或重编译请求，但仍有 **2 条 underfull hbox**（`sections/evaluation.tex:203`，badness 10000 和 2261）。页数依据本次编译日志；未进行独立 PDF 解析或人工视觉审读，标题、作者、公式、表格与引用的视觉检查仍待完成。本次未改 TeX，未安装包，未运行 GPU 或数值实验；编译成功不构成容量、模态、异步或质量验收。

已知 Python 环境：`/opt/conda310/bin/python`，版本 **3.10.9**；此处同步已知环境记录，不表示本次重新验证或 TeX 构建依赖 Python。Makefile 会提示缺失工具，不会自动联网、安装宏包或运行推理。`make help` 可查看命令。

## 引用与提交包

当前正文引用现有 `references.bib`，本次不改 bibliography。后续新引用仍须逐篇核实 primary source。vLLM 可作为已出现的系统线索，但其论文元数据和具体机制差异仍需独立核对；不要从工具名称猜作者、年份或 arXiv ID。

`main.tex` 末尾已集中启用 bibliography；新增条目及正文 `\cite{...}` 仍须经过独立核实。正式提交前：

1. 将正文与上述已确定的标题、作者、单位和联系邮箱对齐，确认论文类型；ORCID/资助仅依据作者确认填写，去掉或改写所有 draft note。
2. 完成相关工作与必要实验，保留负结果和限制，不把方案表改成“已完成”表。
3. 将具体 claim 绑定冻结源码、binary、model/tokenizer identity 和公开原始数据。
4. 在干净环境从源码编译，检查未定义引用、溢出、图表、字体和文献。
5. 按提交时 arXiv 的当前要求核对 TeX 版本及文件格式；本轮没有进行远程规则核查。
6. 提交包只包含必要 `.tex`、图件、参考文献构建文件等。启用 BibTeX 后确认 `main.bbl` 按实际提交流程可用；不要只依赖私有 `.bib` 或本机构建缓存。
7. 不打包本地 ledger、历史会话、模型权重、私有绝对路径、`target/` 或整个仓库。

## 后续补齐顺序

1. **先执行 E2 vLLM 性能工程计划，再补齐 E0 容量证据：** 先建立有资格的 system-level baseline、描述性 pilot 与确认协议，再按 checklist 完成容量及机制验收。E0 仍须固定模型/表示/物理池，闭环磁盘必要性、请求期读取、峰值与合格推理；性能结果不替代它。
2. 标题、作者、摘要、问题定义、实验排序与 [PAPER_COMPLETION_CHECKLIST.md](PAPER_COMPLETION_CHECKLIST.md) 工作清单已对齐核心目标；当前容量、多模态、统一内存和全异步范围仍须逐 profile 验收。
3. 整理分别冻结的 F32 和 bounded Mixed profile 可公开证据，与 W8A16 strict FAIL 及旧失败分开；数值验收必须对应容量实验的具体模态、设备、profile 和 snapshot，不能跨配置借用。
4. 建立逐模态、设备和 profile 的证据矩阵、相关工作矩阵与最小公平 baseline，区分统一/共享物理池、离散设备、host-resident 与真正超过可用物理驻留容量的 storage-backed 推理。
5. 分别验证异步提交/完成、实际执行 overlap 和事务 custody 对安全重用容量的作用；补独立重复、并发/尾延迟、资源高水位、安全消融、组合故障及正常路径开销，用实际数据替换方案并保留负结果。
6. 新版 TeX 修改后另行重编译并记录对应版本，不将上版成功结果移用为新版验证。
