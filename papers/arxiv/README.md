# Ferrule arXiv 论文工作区

这是**本地、未投稿的英文论文工作稿**，不是已经完成实验或可直接投稿的论文。建立于 2026-10-02。

## Current manuscript: bounded residency and qualified factory profile (2026-10-15)

The current title is **Ferrule: Bounded Storage-Backed Weight Residency**.
The argument is problem/contracts → normal path/invariants → lifecycle
Algorithms 1–2 and bounded next-expert Algorithm 3 → implementation → RQ1–RQ4
→ prior work/limitations. The paper no longer requires or implies a
beyond-total-physical-memory necessity result to support its scoped residency
mechanism. External performance leadership/SOTA is conditional on future
matched qualification and comparison; no first-system, SOTA, speedup or
nonblocking-forward claim is made.

### Implemented mechanism versus evidence

- Both **blocking** and **lookahead** explicitly enable the real positioned-read
  worker; the legacy disabled path is not the RQ3 control. One worker, queue
  depth one, two charged read slots, source identity v2 including Unix ctime,
  immutable ordered plans and canonical gate/up/down weight/scale artifacts.
- The owner releases current host payload aliases after upload and submits one
  next-expert read **before current compute submission**. Whole forwards and
  existing CUDA consumer proofs remain synchronous. Speculative errors wait for
  the matching demand; abandoned I/O is drained and counted, not silently erased.
- Both study arms fix **64 MiB host / 8 MiB metadata / 2 MiB requested stack**;
  metadata and stack are included in host quota. Device cache remains **64
  entries / 320 MiB charge including 8 MiB scratch**. `2 P_max + M + S <= H`
  covers packed/prepared coexistence, not total RSS. OS/TLS/allocator overhead
  and retained weak-reference backing memory remain distinct and unmeasured.
- Quota follows exact weight/scale allocation aliases. Passive observations do
  not refund custody, heal poison or fabricate Closed. Host cancel/drain/join
  proves no GPU quiescence. CLI options lower through prepared source-bound
  admission into the existing factory/materializer.
- **Current factory qualification: actual 9/9 GPU PASS**, exact once on
  `033519c3def098ce417e4b6ff0f141a9f30bddb1`, 1,484 frozen source members.
  The producer [SUMMARY](../../target/validation/expert-read-current-quality-20261014-02/SUMMARY.md),
  [FINAL](../../target/validation/expert-read-current-quality-20261014-02/FINAL.json)
  and [receipt audit](../../target/validation/expert-read-current-quality-20261014-02/receipt-audit.json)
  replace the earlier RUNNING handoff, without rewriting any old failure.
  Selected factory/provider numerical qualification closes; CLI/HTTP, capacity
  frontier, physical release, paired performance and GPU overlap remain open.
- Actual runtime mappings record CUDA12.4.99 with SM86 compiled code and NVTX
  compiled, **profiling OFF**. No probes/warmups/retries/captures; all nine issued
  once and passed, natural exit/authenticated ECHILD, recorded owners absent,
  stable-idle v2/all-eight zeros and local locks released/reprobed. These are
  process/idle observations, **not a CUDA physical-release certificate**.

### Current numerical qualification (both schedules, separate denominators)

| Phase | Gates | Main metrics per gate | Common-history vocabulary values per gate | Scope |
|---|---:|---:|---:|---|
| Full40 | 2 | 804 | 1,986,560 | 2,560 ordered route IDs per schedule; diagnostic states |
| Retained fresh | 2 | 5,427 | 9,187,840 | 64 actual fingerprints per schedule, all four named streams |
| Ordinary | 2 | 27 | 9,187,840 | All 64 fingerprints match own-schedule diagnostic primary; routes/states unobserved/null |
| Cold owner | 2 | 1,809 | 3,228,160 | All 28 fingerprints match own current primary/config/oracle |
| Constructor | 1 | — | 128 total, not common history | Four boundary/schedule cells ×32 finite vocab values, six verified reads each; tiny provider scope |

Per fresh/ordinary schedule: common-history **27/9,187,840**, free feedback
**27/9,187,840**, reset **9/3,228,160**, fork **1/248,320** (calls/values).
Per cold schedule: common-history/free-feedback/reset **each 9/3,228,160**;
fork **1/248,320**. Fingerprints are not additional logit values. Never sum
these streams or mix ordinary's 27 output metrics with diagnostic route/state
metrics as a larger independent-quality denominator. Original tolerance is
`0.00025 + 0.012*abs(reference)`, with exact IDs where observed, not universal
bit-exactness or broad task quality.

Verified worker reads per schedule: **2,089 full40; 25,116 fresh; 25,116 ordinary;
11,354 cold**, distinct from sampled operations or physical NAS bytes.
Composition reports **16,777,984 logical host-charge bytes** within 64 MiB:
`2*3,146,112 + 8,388,608 + 2,097,152`. Alias-held payload quota is retained;
weak backing/allocator memory and RSS are distinct. Reported allocator reserved
high-water **4,106,223,616 bytes** is within the 7 GiB policy, but its getter can
poll retirement fences; passive snapshots leave allocator unobserved. These
source-qualified point observations are not continuous physical peaks, a
statistical guarantee, overhead-free observation or final physical release.

CLI/constructor/composition are different executables. Common immutable
source/native/runtime pins do not prove mathematical CLI numerical equivalence,
HTTP acceptance or machine-identity equivalence. The portable paper receipt
archive is [qualification_v1](experiments/storage_demand/qualification_v1/README.md),
committed at `17b16fa059c3508788e081081d70452639a4e19f`; the complete archive has
**249 files / 5,642,720 bytes (5.38 MiB)**, including 240 manifest-bound records
occupying **5,343,269 stored bytes**. Its checker and all 37 offline tests pass.
Executables, model/oracle payloads and the full source archive remain external
hash-pinned inputs, not copied or rehashed by the offline checker. Curation is
not a new run or CLI qualifier; this writing stage does not edit the archive.

### Receipt chronology (do not combine passes)

| Historical record | Actual numerical outcome | Wrapper / source boundary |
|---|---|---|
| Middle quality `-01` | `cuInit (3)`, 0 metrics/logits; fresh/cold unissued | Different from TP1 `DEVNULL` prelaunch `-01` |
| Full40 `-02` | PASS: 804 metrics / 1,986,560 logits | Original strict post-idle wrapper rejected |
| Retained fresh `-03` | PASS: 5,427 metrics / 9,187,840 logits | Original strict post-idle wrapper rejected |
| Cold owner `-04` | PASS: 1,809 metrics / 3,228,160 logits | Accepted new stable-idle v2; 3,533-input cut `5662b7d2…`; `same_source_cut=false` versus old primary |
| Frozen CLI quality `-01` | `cuInit (3)`, 0 metrics/logits/token checks | Retained fresh unissued; not same executable/HTTP quality |

The first strict wrappers and raw utilization samples remain failed; later idle
does not reclassify them. Old cold acceptance does not qualify old full40/fresh
on its cut. The new nine-gate result uses its own current source/config/primary
relations and does not borrow these historical passes.
Exact artifact paths and limitations are in [EVIDENCE_LEDGER.md](EVIDENCE_LEDGER.md).

### RQ3 plan and RQ2 scopes

The registered order is **B–L, B–L, L–B, L–B, B–L**, fixed 3:2 balance;
ten planned lifecycles ×12 requests gives **120 requests /50 measured QA slots**.
The actual [paired closeout](experiments/storage_demand/paired_read_v1/artifacts-current-20261015-01/closeout.json)
is **permanently stopped, no retry**: P1-blocking failed before Ready with
`experimental qwen35-mixed-bf16 serving: prepared source profile changed`.
The stopped `cmd_serve` unconditionally invoked the TP2 cache accessor on
enabled TP1's prepared-source payload; this was not `cuInit` or numerical mismatch. One
lifecycle was attempted, zero completed, nine unissued; **0 HTTP issued,
120 requests and 50 QA slots unissued**, all five incomplete pairs retained.
No quality metrics, latency, goodput, aggregate, interval or significance result
exists; null is not zero. Ready maps/read ledger/final owner closure remain
unobserved/unknown despite authenticated process drain and idle/lock receipts.
The default disabled/no-worker-read CLI path is not either enabled arm, and the
historical DEBUG `-02` figures are not enabled-pipeline measurements.

[Corrective V3](experiments/storage_demand/paired_read_v1/PLAN_V3.md) is a separate
CPU-only preregistration with explicit **cohort target1 /max-deferrals0** in
both arms, not stopped CLI defaults4/3 or a production-default change.
The **source-plan query fix is committed at `86c58a0`**. Its owner reports
CPU and hidden-CUDA `CudaWorkspace` checks passed; local source inspection
confirms catalog-selected serving-resource queries, not the old unconditional
TP2 accessor. These are not a new CLI/HTTP qualification. New candidate/binary,
nine same-cut gates and V3 Ready/HTTP/runtime binding remain pending.
V3 is neither a resume nor denominator repair; old `033519c` PASS is only its
configuration/oracle reference. The prospective envelope is 7200 s absolute,
650 s/lifecycle +100 s final reserve, STOP/no retry, all-five-pairs-or-null.
Five blocks do not guarantee power, interval coverage or a performance victory.
The independent baseline pilot remains **5 lifecycles /60 requests /25 QA**,
unexecuted; neither its seals nor the stopped study is edited here.

RQ2 separates historical HTTP logical demand, current factory owner-lifetime
read observations, shared NFS-client mountstats, and unknown NAS backing traffic.
The stopped pilot has shutdown-only shared-client snapshots/deltas, **no
Ready/startup-to-Ready or HTTP request windows**; those are not zero request I/O
or server-attributed NAS bytes. Fresh owners are not storage-cold; missing/reset
counters stay unknown, not zero.

### Current plot-integration validation

This writing stage changes only `main.tex`, the eight manuscript sections,
this README, the checklist and the evidence ledger. Existing figure sources,
all experiment code/configuration/raw receipts and the bibliography are untouched.
Build/review outputs are confined to `build/`; no GPU query, lock, server launch,
experiment execution or Git commit is part of this pass.

- `make -C papers/arxiv`: **PASS**, finite 120 s limits, no build timeout.
  Latest `build/main.pdf`: **36 pages / 757,784 bytes** (35 main/reference
  pages + one historical appendix page); SHA256
  `d0aa956a5793b344fd4ab18452e1a66afaa3696c12b6860ebb3a9c19dfb06989`.
  Version snapshot: **2026-10-15 recovered factory9 / stopped CLI V2**;
  factory cut `033519c`, archive `17b16fa`, source-plan query fix `86c58a0`
  (new CLI/HTTP qualification pending). Source/PDF hashes and validation scope
  are recorded in `build/recovery-final-review.json`.
  The pre-update 34-page/748,066-byte build is historical, not the latest PDF.
- Converged log: **zero** LaTeX/package warnings, undefined references/citations,
  overfull/underfull boxes, oversized floats or duplicate labels. Three standard
  algorithms, all seven existing figure assets, complete new qualification table,
  charge equations and the historical appendix are included.
- Ghostscript rendered all **36 pages in color (110 dpi) and grayscale (72 dpi)**.
  Color contact-sheet review covered title/abstract, all algorithms, current
  qualification/RQ scopes, figures/captions, conclusion/references and appendix.
  Full-page checks covered **Algorithm 3 (p.15), current qualification Table 6
  (p.22), and Appendix A (p.36)**; figure pages 8/10/19/25–28 were also checked
  in grayscale. No clipping, obscured text, caption occlusion or incomplete
  algorithm was observed. The long appendix receipt path now has explicit
  quote/line breaks. Local review is not scientific closure.
- **Read-only producer cross-check PASS:** all nine typed report hashes match
  the audit; reports complete/passed with authenticated ECHILD. Direct comparison
  verifies all 64 ordinary and 28 cold actual fingerprints against each own-schedule
  primary. Ordinary route/state observations remain false/null; per-stream
  denominators are separate. These are receipt checks, not GPU reruns.
- Narrow source/doc whitespace, compiled label/reference and stale-current-language
  checks: **PASS**. All **29 protected figure/Makefile/bibliography assets** and
  **340 snapshotted archive/paired-receipt files** retain hashes. No Rust/GPU tests,
  HTTP/NFS/NVTX profiling or plot checker was run by this update. Git/index/diff
  verification, staging and commit are left to the main agent; this recovery
  ran no Git command.
- Portable archive checker and **37/37 CPU-only offline tests: PASS**, rerun
  read-only in this recovery; outputs are in `build/recovery-final-archive-*`.
  They verify historical receipt consistency, not current GPU or CLI acceptance.

### Exact handoff file list (no commit made)

```text
papers/arxiv/main.tex
papers/arxiv/sections/introduction.tex
papers/arxiv/sections/scope.tex
papers/arxiv/sections/design.tex
papers/arxiv/sections/implementation.tex
papers/arxiv/sections/evaluation.tex
papers/arxiv/sections/related-work.tex
papers/arxiv/sections/conclusion.tex
papers/arxiv/sections/evaluation-history.tex
papers/arxiv/README.md
papers/arxiv/PAPER_COMPLETION_CHECKLIST.md
papers/arxiv/EVIDENCE_LEDGER.md
papers/arxiv/build/main.pdf
```

The first twelve are authored sources/documents; the last is the already-tracked
rebuilt PDF. Other `build/` logs/renders/check manifests are local review outputs,
not additional source or experiment changes. New `sections/evaluation-history.tex` follows
the repository ignore rule and must be included explicitly in the main agent’s
file-specific commit; this writing stage does not stage it.

### Integrated historical figures

All four existing data PDFs remain frozen **TP1 real `-02`, n=1 DEBUG** assets;
they are not regenerated or relabeled as current-pipeline measurements.

| Asset under `figures/` | Manuscript equation / evidence scope |
|---|---|
| `storage-demand-logical-transfer.pdf` | `eq:request-io`: application reader/facade bytes, not disk/bus traffic |
| `storage-demand-cache-reconciliation.pdf` | `eq:entry-reconciliation`, `eq:cache-charge`: boundary counts and configured cache-charge high-water, not total physical-owner memory |
| `storage-demand-invocation-timeline.pdf` | 30 log-order invocation boundaries, not elapsed/kernel time |
| `storage-demand-timing.pdf` | `eq:timing-origins`: startup/request origins; no comparative statistics |

Three existing mechanism PDFs remain conceptual and profile-scoped. Standard
`algorithm`/`algorithmic` is retained. CPU-only build: `make -C papers/arxiv`.

---

## 目录与版本控制

该目录历史上被忽略；本次只读 index 核对确认稿件、说明文件和 `build/main.pdf` 在跟踪列表中。提交须使用本轮精确文件清单，不加入整个工作区；`target/` 证据及额外 build/review 文件仍需另行归档，不能仅依赖 Git。不要执行 `git clean -fdx`。

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
    ├── conclusion.tex
    └── evaluation-history.tex # 新增，须显式纳入提交
```

PDF 构建后位于 `build/main.pdf`。现有彩色概念图与四幅实测收据派生图位于 `figures/`；实际 include 路径、label 与范围见上方说明。所有结果表均保留其冻结收据范围，不把 placeholder 当作结果。

## 核心目标、标题、作者与设计范围

核心目标是 **bounded storage-backed weight residency**：在声明的账本预算下，
通过源绑定物化、有限驻留复用与 custody 安全完成合格推理。软件预算与真实物理
容量分报；当前不以模型超过 host+device 总物理内存或硬盘必要性作为标题/结论。
更强 necessity 主张另需紧凑权重不可驻留、物理 I/O 与分域归因证据。

标题：**Ferrule: Bounded Storage-Backed Weight Residency**。
作者：**Lu Huanbing**；单位：**Independent Researcher**；邮箱：
**luhuanbing084@gmail.com**。ORCID/资助未提供，不猜测。

通用模态/统一内存为 future scope；当前路径是单 owner 文本/CUDA。
真实读 worker 与两种调度已实现，current factory GPU qualification 9/9 PASS；forward仍同步，CLI/HTTP与performance/overlap pending。
四 RQ 为核心；vLLM/SOTA 仅 conditional secondary，非无条件投稿门禁。
事务 custody 非持久 ACID、数值 exactness、恢复/活性或性能保证。

历史讨论与 stronger beyond-capacity 计划见旧 ledger/checklist 条目；不把
历史目标、旧 PASS 或旧构建覆盖当前范围。当前证据见
[EVIDENCE_LEDGER.md](EVIDENCE_LEDGER.md)，执行计划见
[PAPER_COMPLETION_CHECKLIST.md](PAPER_COMPLETION_CHECKLIST.md)。

## 已经写入什么

- 英文摘要、引言与动机场景。
- 资源域、准入条件、精确性分层、故障与活性假设。
- exact-key materialization、资源 custody、分阶段 KV publication/retirement、取消与未知完成。
- 实现边界、E0 容量主实验与 E1–E6 支撑实验方案、测量单位、质量门禁与 artifact 要求。
- 相关工作的比较维度和有明确限制的阶段性结语。

初稿中的资源公式是**抽象记账条件**，不是形式化验证结论；当前实验章节区分实际冻结的 logical-demand/图表观察与尚未完成的证据门禁。候选贡献尚未通过主文献比较确立 novelty。没有捏造作者、引用、实验数据、置信区间或性能优势。

## 更强容量主张的条件与历史证据边界

**核心容量 claim 当前尚未建立完整证据，完整的磁盘必要性证据也未确认。** 不能借用全驻留 EP8、host-only 运行，或全部紧凑权重仍可驻留某个内存池的 offload 结果来证明这一目标。

容量报告必须区分：

- **磁盘 checkpoint 大小：** 指定文件及编码的逻辑字节与实际占用；不直接等于内存需求。
- **紧凑权重全驻留需求：** 指定量化/压缩表示及 scales 等必要数据能否全部留在可用物理内存中；必须显式检查，而不能从执行展开大小推断。
- **执行表示与全驻留权重需求：** 按实际解码、反量化、packing、padding、共享及复制规则，报告各表示与全部权重同时驻留的需求。执行展开字节超过可用物理驻留容量，但紧凑权重仍能全驻留时，不足以证明必须使用硬盘。
- **去重的可用物理驻留容量：** 按实际物理 pool 归属报告可用于驻留的容量，同一物理池或共享页只计一次；占用独立物理存储的副本仍各计其实际占用。离散 GPU 与 host 只是一个实例。统一地址或 managed-memory API 本身不证明底层只有一个物理池，必须核对实际拓扑、迁移及驻留情况。
- **总工作集：** 实际驻留权重加 KV、activations、workspace、staging、在途及 retained buffers、metadata、固定分配和相关 OS page cache；不能只用账本占用代表部署总内存。

若恢复 physical-memory necessity 主张，其实验必须绑定表示与算术合同，报告物理内存池、去重规则和实际施加预算、逐池高水位、cache/swap 条件、请求期物理存储读取，以及对应模态的端到端推理完成与数值/质量验收（当前语言解码实例包括 prefill 和多步 decode）。离散 GPU+host 只是一个配置实例，不是通用定义；人为限额与真实物理容量不足应分别表述。仅总工作集因 KV/并发超限、仅出现非驻留请求或仅成功加载模型，都不足以建立模型超过可用物理驻留容量的核心 claim。

不要直接将旧会话中的测试/吞吐抄入论文。2026-10-08 按 [CURRENT_ACCEPTANCE](../../docs/CURRENT_ACCEPTANCE.md)（2026-10-07）及其冻结收据同步；[性能协议](../../docs/PERFORMANCE_EVALUATION.md) 保留测量方法和历史限制。当前实际 CUDA 解码证据保留其设备、模态和 profile 局限：

- 最新 adapter-extraction F32 收据为 6/6 PASS：代表性 PP2TP2EP2 及分别测试的 resident thread/process EP8，合计 7,946,240 logits；不是全部四拓扑重验，pipeline 不含 raw route/state oracle。旧 process-only PASS 保留原范围，不转借；所有结果均非当前工作树或性能通用保证。
- 单-owner Mixed prepared-attention 原 full40 804 metrics / 1,986,560 logits，bounded fresh QA/arithmetic/Chinese 5,427 metrics / 9,187,840 logits elements PASS；原 reference/阈值未改。非 bit-exact、非广泛任务质量，context<=1024；BF16 activation/KV/conv 与 F32 accumulation/SSM 不能简称全 BF16。
- 较早 Mixed public serve 只有 Hello lifecycle PASS，source/binary 不同；不能与新 attention quality 收据拼成最新 serving binary 的质量/性能通过。原始失败、post-GPU CPU-only 变化及各自 pins 见 ledger。
- **W8A16 独立 full40 oracle 已存在，但 strict acceptance failed**，涉及 logits、route 和 state；不能再写 “oracle 缺失”，也不能用 argmax 一致替代严格验收。
- W8A16 timing 可作为带限制的历史观察，不是 quality-qualified performance acceptance。
- 相同 benchmark client 不等于相同服务端；不同精度/拓扑的 vLLM 部署对比不是公平的 runtime speedup。
- 本轮只做资料核对和写作，**没有重跑 GPU 实验或数值测试**。

详细路径与当前事实见 [EVIDENCE_LEDGER.md](EVIDENCE_LEDGER.md)。历史计划与新评估冲突时，先追踪原始 artifact 和版本，不通过选择更好看的结果消除冲突。

## 历史运行计划（2026-10-08；非当前执行顺序）

先做 vLLM 性能工程，再补齐论文清单；E2 升 P0，E0 的 bounded-residency/quality 部分仍未完成；physical necessity 是另行条件主张。具体阶段、预注册判定和条目映射见 [checklist 第 0 节](PAPER_COMPLETION_CHECKLIST.md)。论文专用 runner 入口为 [experiments/README.md](experiments/README.md)（现有 runner 与收据说明）；本次只改这三份 Markdown，不创建或修改 experiments，不在根 `scripts/` 新增论文专用脚本。

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

1. current `033519c` factory已9/9通过且portable receipts已归档；paired原study永久stopped。由独立production owner修复TP1/TP2 accessor，corrective V3需新cut九gate与candidate/CLI/HTTP/Ready绑定，不借旧PASS。
2. RQ1 selected factory quality subgate closed、capacity frontier仍open；继续RQ2 logical/NFS-client visibility与RQ4 integrated faults/custody，physical release/overhead/attribution unknown诚实保留。
3. RQ3 corrective V3已CPU预声明cohort1/0与finite wall/STOP/no-retry/uncertainty，但binding/execution pending；不resume旧study。GPU overlap另需相关联kernel/copy活动，whole forward仍同步。
4. 现有 baseline5 lifecycles 保留独立/unexecuted；vLLM/SOTA、physical necessity、大矩阵/其他模态/统一内存为 conditional follow-up，非无条件 gate。
5. 归档冻结 source/binary/model/oracle/raw/plots、作者声明与相关文献完整审计；最后独立投稿/视觉验收。当前 build/read-only checks 不等于科学 PASS。
