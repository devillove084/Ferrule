# Ferrule 论文补齐清单

本清单用于逐项补齐论文，不是已经完成工作的报告。当前工作标题：

> **Ferrule: Bounded Storage-Backed Weight Residency**

作者：**Lu Huanbing**；个人名义（Independent Researcher）；联系邮箱：**luhuanbing084@gmail.com**。

## 0. 不再偏离的主线

**目标：**在分域 working-set 预算下，存储支撑有限驻留复用并完成合格推理；软件预算与真实物理不足分报。后者未验证：设备装不下/cache miss 不证明 host RAM 不足或磁盘必需。

**关键设计：**按需物化/有界复用、依赖/完成驱动、事务预留/发布/回收/custody；异步与事务服务容量目标及潜在 I/O 隐藏，不是独立主张。

**范围：**通用模型/模态/内存架构为设计范围；文本/CUDA/KV 证据仅对应 profile。逐路径验证异步：API 不等于全链路无阻塞或 overlap。

**容量定义：**按不相交物理驻留域计量，统一/共享池不重复相加；Disk/NAS 非驻留内存，swap/page cache/映射/迁移另报。合计容量不代替各域放置可行性。

### 优先级与状态

- **P0：核心论断成立所必需。**缺失则不能把目标写成已验证结论。
- **P1：形成可信、可比较、可复现论文所必需。**不是可随意跳过。
- **P2：扩展范围。**只有提升为主张时才需要进入首篇最小矩阵。
- `[x]` 只表示该项明确完成；`[ ]` 可能部分已有材料，不代表完全没有实现。
- 记录状态使用：待定义 / 待核对 / 已有局部证据 / 待实验 / 已验收。每次勾选须附冻结产物及验收结果。

### Submission-core checklist (2026-10-15)

**当前范围：RQ1–RQ4，非 vLLM alignment/大矩阵/无条件 E2 胜出。** 旧 P0/P1 为证据索引，非全部组合/图号投稿 gate；仅选定真实 profile，unknown/failed/unexecuted 保留。

- [ ] **RQ1 / E0 — bounded residency 与完整质量。**冻结同 cut 的模型/表示、源绑定路径、完整 prefill + 多步 decode、oracle 和分域预算；测 materialization/安全复用、state/KV、staging、workspace、page cache 与采样归因。软件账本不等于 RSS/机器物理 cap；physical insufficiency/necessity 仅恢复该强 claim 时另验。**现状：**历史 real `-02` 四请求/采样 actual；后续 full40/fresh 数值 PASS 但 wrapper rejected；cold `-04` accepted on different cut，不能拼 current qualification。冻结 CLI `-01` 初始化失败、0 metrics；current `033519c`/1,484-source factory qualification 已实际9/9 PASS；RQ1质量子门禁关闭，capacity frontier/CLI HTTP仍open。
- [ ] **RQ2 / E3 — 分层 request-I/O 可见性。**同 run 对齐 logical reads、read timing、可用 shared NFS-client mountstats，并保留 attribution/cache/reset/missing 限制；NAS physical network/block unknown 不填 0。**现状：**历史 `-02` HTTP logical与current9 factory verified reads actual，serving/test scope分开；passive NFS helper CPU-prepared，无新 aligned client/physical measurements。可按 logical/client scoped report 闭环，不强求证明 physical disk necessity；若主张 backing-disk bytes/necessity 再补 server 归因。
- [ ] **RQ3 / E1 + E4 — 同合同 blocking/lookahead 资格与配对。**两 arm 真实 worker 已实现/CPU fixtures 覆盖，均explicit opt-in，current factory mechanism/numerics qualified，不是“safe blocking 未实现”。固定 source/weights/arithmetic/quality/cache/safety，host64 MiB（metadata8 + requested stack2 included），device64 entries/320 MiB charge（scratch8 included），1 worker/depth1/two slots。current upload 后、compute submission 前 submit next；forward 同步。实际9/9 same-cut factory qualification已通过；paired无HTTP timing/GPU overlap；只有failed startup的shared NFS shutdown样本，不是RQ3性能PASS。
- [ ] **RQ4 / E5 — 集成故障与 custody。**验证 stale generation、cancel/publish、cleanup failure、credit/alias quota、quiescence/Unknown。Passive poison observations 不退款/修复/伪 Closed；host drain/join 不 heal GPU Unknown。weak witnesses 不新增 payload copy，但 last strong Arc drop 后 allocation backing/allocator RSS 仍可能保留，quota 与 RSS 分报。历史 `-02` drain/ECHILD/idle actual，final resource contract unknown、GPU quiescence unverified；集成故障/final closure pending。
- [ ] **核心身份与复现。**冻结 source/dirty snapshot、binary/model/tokenizer/client/oracle、argv/env、raw/阈值、sampling/clock、所有 failed/canceled/timed-out/unknown/unissued 分母；historical failures 不抹除，不跨 cut 借 PASS。
- [ ] **RQ3 stopped study / independent corrective V3。**实际registered顺序B–L/B–L/L–B/L–B/B–L（3:2），10 lifecycles/120 requests/50 QA；[closeout](experiments/storage_demand/paired_read_v1/artifacts-current-20261015-01/closeout.json)：P1-blocking在Ready前`prepared source profile changed`，TP1 enabled prepared payload被unconditional TP2 accessor拒绝，不是cuInit。1 attempted/0 complete/9 unissued，HTTP issued0、120/50全unissued、all5pairs retained、quality/latency/goodput/aggregate/CI均null/unobserved；resource ledger unknown，无retry/replacement。[V3](experiments/storage_demand/paired_read_v1/PLAN_V3.md)独立CPU-only，显式cohort1/0而非CLI4/3；source-plan query fix已committed86c58a0，owner报告CPU/hidden-CUDA CudaWorkspace checks PASS（本文未重跑），新cut九gate及candidate/Ready/HTTP/runtime绑定仍pending，旧PASS不资格未来cut。7200s绝对/650s lifecycle/100s reserve、STOP/no-retry、all5-or-null；不resume/repair旧分母，不称power/significance。独立baseline5/60/25仍unexecuted，experiments/seals不改。
- [ ] **稿件与 artifact。**Algorithm 3/实现/三层 RQ2/paired RQ3 已连贯接入，既有四 real `-02` plots 不动且不是 current pipeline stats。当前36页（含历史appendix）构建/局部视觉验证已记录；九gate portable receipt archive已committed17b16fa；external binary/model/source payload与其他历史archive、最终投稿/作者声明/文献完整审计仍pending。metadata 非 crash durability，general protocol 非 universal registry。

**当前局部数据：**[TP2 crossblocks 原报告](experiments/tp2_resident/runs/route-batch-vllm-crossblocks-20261012-01/crossblocks-results.md) 为补充性 resident E2 观察：F 14.317/14.804 vs V 10.376/10.330 tok/s，TTFT/启动/GPU/PSS 有取舍；无 bit-exact、fairmatch、统计胜出或 E0/disk-necessity 结论。vLLM 实测是 `0.28.0+cu129`，由 `target/vendor/vllm028-py312` 的冻结身份支持；0.18 源码不能冒充该部署兼容性。

- [x] **局部 startup 核对（scoped，2026-10-13；非本次 GPU 测量）。**[REPORT](../../target/validation/startup-metadata-owner-20261013-01/REPORT.md)、[RESULTS/raw pins](../../target/validation/startup-metadata-owner-20261013-01/RESULTS.json)、[source pins](../../target/validation/startup-metadata-owner-20261013-01/executed-source-pins.json)逐字段/hash 核对。36e6 combined prepared-admission + Arc-path：S1 startup/PSS 222.079 s/42.716 GiB，S2 139.855/44.416，F1 140.052/43.573；仅 F1 measured 5 请求、40-out/2.877869 s = 13.899 tok/s，TTFT mean/p50/p95 192.944/187.513/209.700 ms。S1/S2 visible canary/reuse/cancel/source-bound PASS，非 fresh 14M 数值验收。旧 e674 ordinary F1 351.114 s/60.149 GiB/14.227 tok/s（[manifest](experiments/tp2_resident/runs/route-batch-info-after-20261012-01/routebatch-performance.json)），与 crossblocks 不同 series；不混均值/称 paired causal before-after，不选最好 startup 作稳定常数。page cache uncontrolled，PSS 非 heap-live；五阶段 proc 读取在 measured 外但有成本，receipt persistence 未独立测量，非零开销/OS free。只勾局部核对，四 RQ/E0/E3 未验收；portable 归档仍待，本次不搬 raw。
- [ ] **Storage middle（历史 logical-demand actual；current factory qualification9/9 PASS，capacity/performance/physical release未验收）。**[四请求入口](experiments/storage_demand/run_middle.py)/[原 CPU preparation](experiments/storage_demand/prepared.json)保留；早期 CPU validation canceled、planned/unexecuted 无读计数，仅属 -01/-02 前状态，不覆盖随后 actual observations。
  - **历史 -01：**[失败 report](experiments/storage_demand/runs/storage-demand-tp1-middle-20261013-01/report.json)保留一次获授权、server launch 前的 harness `AttributeError`（缺少 `subprocess.DEVNULL`）；planned=4、dispatched=0、not-run=4，无 request/read/quality data，不是 model/OOM failure。authenticated drain/post-idle 通过；resource contract 与 GPU quiescence 未验证，physical I/O unknown。
  - **随后 -02（n=1、DEBUG instrumented）：**[report](experiments/storage_demand/runs/storage-demand-tp1-middle-20261013-02/report.json)、[case](experiments/storage_demand/runs/storage-demand-tp1-middle-20261013-02/case.json)及[独立 closure](../../target/validation/storage-demand-middle-gpu-20261013-02/closure.json)：planned=4、dispatched=4、terminal=4、success=4、failed=0、not-run=0；Hello2 + QA/Arithmetic/Chinese 三 canary，3/3 visible-text/eight-output-token-usage PASS。Hello2 PASS 仅为 HTTP+usage（`reference=null`）；仅三 canary 与 frozen reference 比较。QA 仅为 canary，不是 extended QA workload；generated IDs 未暴露，无 exact-token/full numeric-quality 验收。
  - **逻辑计数与 bounded reuse 观察：**30 invocations，69,114 reads/36,240,064,128 bytes；88,763 facade H2D copies/36,240,248,688 bytes；expert hits=0、uploads=11,519、evictions=11,455、artifact uploads=34,557。startup reads 不在 invocation 计数内。末次 gauges 为 64 entries/201,351,168 bytes、scratch 8,388,608 bytes、owner peak 219,754,496 bytes；invocation pending-upload/quarantine=0 不等于 final ledger closure。
  - **观测边界：**startup 18.497 s、四个分别计时的 HTTP windows 合计 51.538 s（不含 startup/请求间 owner overhead）、supervisor 75.874 s；四请求逻辑 GiB/expert uploads/evictions/TTFT/E2E 见[稿件表](sections/evaluation.tex)。GPU0 sampled peak 4,125 MiB、tree PSS/RSS 7,610.864/7,613.770 MiB，32 complete host/32 GPU samples；非连续峰值或 hard cap，owned GPU attribution unknown。DEBUG overhead 未测，不作 comparative performance/steady-state 结论。RQ1–RQ4 缺口保留，均不勾完成。
  - **历史 middle quality `-01`：**[FINAL](../../target/validation/storage-middle64-current-quality-20261014-01/FINAL.json)记录 full40 `cuInit (3)` 前失败，3 scheduled/1 dispatched/1 failed/0 passed/2 unissued/0 retries，0 metrics/logits；不是 TP1 DEVNULL wrapper，也不是 numerical mismatch。
  - **full40 `-02` 数值 PASS / strict wrapper FAIL：**[FINAL](../../target/validation/storage-middle64-current-quality-20261014-02/FINAL.json)，804 metrics/1,986,560 logits/4 token checks，0 numerical failures；首 post-idle GPU0 0 MiB/1%/无 compute apps 被原 strict gate 拒绝，后续 idle 不修复原失败。
  - **retained fresh `-03` 数值 PASS / strict wrapper FAIL：**[FINAL](../../target/validation/storage-middle64-current-quality-20261014-03/FINAL.json)，5,427 metrics/9,187,840 logits，27 common-history+27 feedback+9 reset predictions；首 post-idle 0 MiB/6%/无 apps 拒绝，原失败保留。
  - **cold `-04` 局部 accepted：**[FINAL](../../target/validation/storage-middle64-current-quality-20261014-04/FINAL.json)，1,809 metrics/3,228,160 logits、authenticated drain + 新 stable-idle v2 PASS；immutable 3,533 inputs，cut `5662b7d2…`。旧 primary fresh 同 config/reference 但 `same_source_cut=false`（旧 `fbf0c225…`）、wrapper rejected，不能拼 all-three/same-serving qualification；idle 非 fence。
  - **冻结 CLI quality `-01` 失败：**[FINAL](../../target/validation/storage-observability-cli-quality-20261014-01/FINAL.json)，同3,533-input cut 的 test/CLI source relation 但 executable 不同；full40 `cuInit (3)` 前失败，0 metrics/logits/token checks，retained fresh unissued、无 HTTP pilot，quality not accepted。
  - **current enabled-cut实际qualification：**[SUMMARY](../../target/validation/expert-read-current-quality-20261014-02/SUMMARY.md)/[FINAL](../../target/validation/expert-read-current-quality-20261014-02/FINAL.json)/[audit](../../target/validation/expert-read-current-quality-20261014-02/receipt-audit.json)，`033519c3def098ce417e4b6ff0f141a9f30bddb1`、1,484 frozen sources，all9 exact once PASS、0 probes/warmups/retries/captures/unissued；不是本agent执行GPU。两arm各full40 804 metrics/1,986,560 vocab/2,560 ordered IDs；fresh5,427/9,187,840 common-history/64 fingerprints；ordinary27/9,187,840 common-history、全部64 fingerprints match own-schedule diagnostic primary（routes/states false/unobserved/null，不填0）；cold1,809/3,228,160 common-history/28 fingerprints bound own current primary。constructor4cells/128finite values，32/cell、6verified reads/cell。
  - **streams/reads/资源边界：**fresh/ordinary每schedule common与free-feedback各27/9,187,840，reset9/3,228,160，fork1/248,320；cold common/free/reset各9/3,228,160，fork1/248,320。不加总streams/observer metrics/schedules当更大独立quality denominator。verified reads每schedule full2089、fresh/ordinary各25116、cold11354；sampled operation counts非实际总reads。host logical high-water16,777,984 <=64MiB（metadata8/requested stack2 included），allocator reserved high-water4,106,223,616 <=7GiB但getter可poll fences；immutable snapshots allocator unobserved。不是连续RSS/physical峰值、无开销或release proof。
  - **runtime/cleanup范围：**actual PID/birth CUDA12.4.99映射、SM86编译、NVTX compiled/profilingOFF；all9 natural exit/authenticated ECHILD/owners absent、all8 stable-idle v2/final3 zero samples、local locks released/reprobed。physical release仍unobserved；distinct CLI/constructor/composition common source/native/runtime不证明数学binary/CLI HTTP/machineidentity等价。curated [qualification_v1](experiments/storage_demand/qualification_v1/README.md)已committed17b16fa，完整249files/5,642,720 bytes（5.38MiB），其中240records/5,343,269stored bytes；37offline tests PASS；copy非new run，external payloads不重验，本文不写receipts。

- [x] **选定factory/profile numerical/provider子门禁（本轮只读核对）：**上方current9/9 receipts与report hashes实读匹配；constructor+两schedule四种gate均一次通过。仅勾此子项，RQ1/E0全项、RQ3 performance、RQ4 physical release与CLI HTTP不勾。

**Broad scope parked：**[第 12 节](#parked-broad-scope)保留 6P/全矩阵、vLLM 全胜/≥5 paired blocks、全消融及模态/统一内存扩展；非无条件 gate。提升主张先恢复资格/预注册，不事后挑点；核心质量/I/O/重复/安全/失败分母不降级。

## 1. 当前基线与已有证据

- [x] 建立标准 `article` LaTeX 模板、分章节源码和 Makefile。
- [x] 确认作者姓名、个人身份及邮箱并写入作者栏。
- [x] 建立本地 `papers/arxiv/` 工作目录（历史被忽略，当前稿件与PDF已跟踪）。
- [x] 将标题/摘要/结论校正为 bounded storage-backed weight residency；physical necessity 与 SOTA 为更强 conditional claims，不暗示已验证。
- [x] 上一版模板实际编译成功：`make -C papers/arxiv`，两轮 pdfLaTeX，9 页，183734 bytes，无最终残留警告。这是上版记录，不是当前修改后 PDF 的页数承诺。
- [x] **本轮资格更新后的稿件构建/局部视觉核对（2026-10-15）：**[README记录](README.md#current-plot-integration-validation)/[日志](build/main.log)，`make -C papers/arxiv` PASS，36页/757784 bytes（35 main/reference+1 appendix），SHA256 `d0aa956a5793b344fd4ab18452e1a66afaa3696c12b6860ebb3a9c19dfb06989`；zero warnings、undefined refs/citations、overfull/underfull/oversized floats/duplicate labels。3 algorithms+7既有图，新current qualification表/host-charge数学/历史appendix；Ghostscript完整36页color110dpi/gray72dpi，color全页contact-sheet与Algorithm3 p15/current Table6 p22/AppendixA p36 full-page、图页gray审读无clipping/obscured text/occlusion/incomplete algorithm；版本/source hashes见`build/recovery-final-review.json`。9report hashes及ordinary64/cold28 own-primary指纹直比PASS，29protected assets及340 archive/paired receipts hashes unchanged、narrow whitespace/compiled label checks PASS；Git/index/diff核对交main，本recovery无Git命令。只读收据/CPU build，未运行GPU/HTTP/NFS/overlap，未stage/commit；旧34页为pre-update历史。
- [ ] **最终投稿视觉验收。**本轮 local PDF/render 审读已完成，但独立最终版本/作者声明/公开artifact复现验收仍 pending，不把 local build 当科学通过。
- [x] **九gate portable receipts局部归档：**[qualification_v1](experiments/storage_demand/qualification_v1/README.md)已committed17b16fa，完整249files/5.38MiB，read-only checker/37 tests PASS（outputs仅build）；不含external executables/model/full-source payload，非CLI复现或新GPU验收。
- [ ] **其余备份/恢复：**paper源/PDF已跟踪，新增evaluation-history须显式纳入；其余历史target证据与external payload恢复/clean-environment reproduction仍open。

已有可复用材料需分别处理；2026-10-08 仅按 [CURRENT_ACCEPTANCE](../../docs/CURRENT_ACCEPTANCE.md) 和 [ledger](EVIDENCE_LEDGER.md) 同步 2026-10-07 冻结收据，非本次重跑：

| 材料 | 当前可用性 | 不能替代什么 |
|---|---|---|
| exact-key registry、broker、KV/cleanup 状态机源码 | 有静态机制证据，见 `EVIDENCE_LEDGER.md` | 不能证明每条实际服务路径都接入这些机制 |
| 历史 F32 process-EP8 full40 数值通过记录 | 保留原局部配置证据 | 该旧 PASS 不借给 thread、其他精度、其他快照或模态 |
| 2026-10-07 adapter-extraction F32 收据 | 6/6 PASS；代表性 PP2TP2EP2 和分别测试的 resident thread/process EP8，合计 7,946,240 logits | 非全部四拓扑重验；pipeline 无 raw route/state oracle，不借 resident 结果补齐；不是 serving 性能 |
| 2026-10-07 single-owner Mixed prepared-attention | 原 full40 804 metrics / 1,986,560 logits；bounded fresh 5,427 metrics / 9,187,840 logits elements PASS | 原阈值不变、非 bit-exact；不是广泛任务质量、长上下文、distributed Mixed 或新 serving binary 验收 |
| 较早 Mixed public-serve 收据 | Hello lifecycle PASS，另一 source/binary | 不能与新 attention quality PASS 拼成同一 binary 的质量与性能通过 |
| 独立 single-GPU W8A16 full40 oracle | 已存在，strict FAIL；必须保留负结果 | argmax 一致不能替代 logits/route/state 验收，也不是对 EP8 做了同一次测试 |
| full-prewarm EP8 短请求性能窗口 | 可作为描述性部署观察 | 不是磁盘支撑的超总内存实验，不是独立重启统计 |
| 早期 staged subset 并发/2K/部分 8K 归档 | 有小规模已完成记录及中断记录 | 不是完整矩阵，不是当前优化版本的普遍验收 |
| 多模态、统一内存、全异步 | 设计范围；逐路径实现/测量状态待建立 | 不因标题通用就升级为已实现、已验证能力 |

**当前同-cut factory numerical/provider qualification已实际9/9 PASS；capacity frontier、CLI/HTTP、request-I/O attribution、paired performance/overlap与physical release仍未闭环。**更强 physical-memory insufficiency/disk necessity 另需专项验收，非本 scoped paper 无条件 gate。

## 2. 第一批：把问题和实现路径固定下来

### S1 — 容量对象与目标区间（P0）

- [ ] 明确每种配置的 checkpoint 编码、scales/metadata、压缩形式、执行展开形式及必要副本。
- [ ] 分别给出存储文件大小、完整紧凑权重驻留需求、展开后的全驻留权重需求，以及请求期间实际工作集。
- [ ] 按不相交物理池列出真实容量、系统保留量、可用预算、执行中峰值。
- [ ] 区分真实机器物理容量不足与人为限制预算；后者标为 controlled-budget experiment。
- [ ] 核实完整紧凑权重是否仍可全放入 host/shared memory；若可以，不能只用展开表示超限声称磁盘必需。
- [ ] 将权重超容量与“权重能装下、只是 KV/activation/concurrency 太大”分开报告。

**交付：**问题定义、容量符号表、每配置容量清单。
**验收：**读者能独立判断什么放不下、为何需要存储层、是否重复计算共享内存，以及这个条件是否只是全驻留不可能而非流式可行的充分条件。

### S2 — 实际接入路径矩阵（P0）

- [ ] 选定至少一个用于首个容量验证的完整模型/设备/profile。
- [ ] 逐步追踪 storage read → staging/cache → representation conversion → device/shared residency → compute → publication → retirement。
- [ ] 标明每一步的实际 owner、账本、completion 来源和错误责任。
- [ ] 标明是否经过 general registry；专用 Qwen expert、HostKV、共享内存路径不能凭总架构图默认继承。
- [ ] 记录最小执行单元、固定常驻部分、层/专家/张量粒度、KV/activation/workspace 的预算关系。
- [ ] 制作所选 profile 的路径/支持表，分开已实现、已编译、局部测试、端到端验收；原 modality × backend × memory topology × precision × execution mode 全矩阵保留为 follow-up。

**交付：**实现路径表 + 图 F1。
**验收：**每个主实验能映射到实际机制；协议 fixture 不冒充服务端完整路径证据。未知项保留 unknown。

### S3 — 所选 profile 的异步边界（P0；非全路径支持声明）

- [ ] 界定“全异步”覆盖哪些路径：read、decode/unpack、upload/migration、compute、completion dispatch、publish、cleanup/shutdown。
- [ ] 区分调用返回快、worker 隐藏阻塞、事件驱动推进、多个阶段真实重叠。
- [ ] 列出仍有的阻塞 syscall、设备同步、capture 限制、锁、线程等待和全局 barrier。
- [ ] 解释依赖如何注册和唤醒；idle 等待是否避免 busy poll；取消如何唤醒 owner。
- [ ] 解释跨请求共享、backpressure、容量不足、取消、Unknown completion 对进展的影响。
- [ ] 规定 trace 的事件身份、时钟、阶段起止点、阻塞原因和 critical path 计算方法。

**交付：**异步边界表 + 算法 A2 + 时间线 F3。
**验收：**“异步”不是仅一个 API 修饰词；能展示已完成实现的依赖驱动推进和实际 overlap。没有测到的路径不声称已全异步；不将 safety 推成 liveness、lock-free 或无同步。

## 3. 图表目录（按核心 claim 选用）

现有 3 机制图 + 4 数据图不等于 F1–F8 验收；旧 ID 为证据索引，可按四 RQ 合并，非八图硬性要求。机制/概念图非 measured trace；四数据图只来自 -02 logical-demand/边界 gauges/分源计时，不证明 physical I/O、overlap、守恒或统计。全矩阵/额外消融/扩展 parked。

| ID / 优先级 | 内容与回答的问题 | 数据/源码输入 | 验收条件 |
|---|---|---|---|
| **F1 / P0** | 存储支撑架构与 ownership：数据从哪里来、有限容量由谁复用 | S2 实际路径表 | 展示后备存储、resident pools、计算、control/completion；离散/统一内存分面或显式别名；标出未接入边界 |
| **F2 / P0** | 生命周期/事务状态机：reserve、submit、complete、publish、retire、release | 真实状态转移、失败分支 | 标注 owner、可见性和回收条件；含 cancel 与不可逆 publish 竞态、无 handle 失败、Unknown custody；不强画成错误的单条线性链 |
| **F3 / P0** | 请求期异步时间线：storage、staging、transfer/migration、compute、completion | 带身份/时钟的真实 trace | 展示等待、重叠和关键路径；区分物理读与 page-cache 命中；概念图注明 illustrative |
| **F4 / P0** | 容量可行区域：内存预算与模型容量比变化时能否完成合格推理 | E0 | 有成功/失败/OOM/timeout/quality-fail 区域及实测峰值；物理不足与人为预算分开 |
| **F5 / P1** | 内存组成随时间：weights、KV/state、workspace、staging、retained custody | 账本 + 系统/设备测量 | 不重复计共享物理页；解释账本与实际峰值差额；Unknown 非零保留不能画成泄漏已修复 |
| **F6 / P1** | I/O 与缓存/预算敏感性：磁盘代价在哪里、何时收益消失 | E3 | 报 bytes、读放大、重复物化、命中层次和带宽；不把 OS cache 当免费内存 |
| **F7 / P1** | 安全消融/正常路径成本 | E4/E6 | 同质量/预算/安全合同；给独立重复和不确定性；不能靠关闭安全检查获得“加速” |
| **F8 / P1** | 故障 trace 与 credits 守恒 | E5 | 对齐事件和资源占用；区分正常清零与 Unknown 保留；不以进程退出替代 fence |

## 4. 表和描述目录（按核心 profile 收敛）

### 表格

- [ ] **T1 / P0：配置与容量表。**模型/模态、存储与运行表示、实际内存拓扑、独立物理池、预算、固定开销、软件/模型身份。
- [ ] **T2 / P0：路径—机制—证据矩阵。**哪些服务路径接入 registry/broker/事务，哪些只有 fixture。
- [ ] **T3 / P0：数值/任务质量验收表。**逐 profile 的 oracle、输入、指标、阈值、通过/失败、artifact；保留 W8A16 失败。
- [ ] **T4 / P0：容量及公平 baseline 结果。**完整请求结果、内存峰值、physical reads、耗时、失败分母、质量门禁。
- [ ] **T5 / P1：related-work guarantee 矩阵。**容量前提、disk 角色、异步范围、资源安全、模态/设备范围、实际对照资格。
- [ ] **T6 / P1：故障与资源守恒表。**注入点、预期 terminal、quiescence、retained credits、重复提交/错误发布、是否真正恢复。

### 架构文字与伪代码

- [ ] **D1 / P0：正常有界驻留执行 walkthrough。**从输入到完整输出，解释反复读入/复用内存为何可行；不能只有 cancellation 故事。
- [ ] **D2 / P0：容量规划。**最小可运行 working set、不可驱逐状态、staging/转换临时副本、并行 owner 分配、KV 与权重竞争。
- [ ] **D3 / P0：提交/发布偏序。**明确调用方可见、下一次计算可用、旧存储可复用分别发生在哪里。
- [ ] **D4 / P0：精确性合同。**精确身份、generation、exactly-one terminal 与数值 bitwise/tolerance/token/task quality 分开。
- [ ] **D5 / P0：安全不变量。**每条写初态、保持它的转移、失败路径及对应测试；不要求伪装成已完成形式化证明。
- [ ] **A1 / P0：准入与回滚算法。**跨账本是否原子、部分成功后谁持有责任、如何避免错误释放、哪些进展/死锁前提未保证。
- [ ] **A2 / P0：异步执行算法。**物化依赖、continuation、完成验证、定向恢复、预算不足时的行动。
- [ ] **A3 / P0：cancel/publish/retire 算法。**包含不可逆 publish 和无 public handle 的失败窗口，保留 Unknown 的 ownership record。
- [ ] **D6 / P1：工程取舍。**为什么不只靠 mmap/page faults、OS swap、普通安全 cache+queue+fence；给受控证据，不贬低未研究的系统。
- [ ] **D7 / P1：限制。**存储带宽/延迟、最小工作集、重复加载、保守保留导致容量损失、尚未支持的模态/设备。

## 5. 实验：E0 为主，其余解释它是否正确、值得及代价

### E0 — 有界驻留与质量可行性（P0；更强 physical-capacity claim 条件恢复）

- [ ] 冻结模型、表示、执行路径、质量合同和 source/binary/client/oracle 身份。
- [ ] 至少覆盖所声明预算边界附近及超预算的点，标明 controlled-budget；若主张真实物理超容量，另补物理边界证据。样本点事先定义，不按成功结果删选。
- [ ] 记录紧凑权重是否全驻留可行，及分域放置限制；报告模型容量/去重可用容量比。
- [ ] 完整模型执行，不用少数层、仅加载、仅服务启动代替完成推理。
- [ ] 对自回归实例包含 prefill 与多步 decode；其他模态使用完整 forward/request 和对应输出 oracle。
- [ ] 实测权重反复物化/回收、物理存储读取、resident peaks、state/KV、staging、固定开销。
- [ ] 记录成功、OOM、timeout、cancel、quality-fail、不可行配置，不将失败当缺测静默忽略。
- [ ] 同时报告完成时间和吞吐/延迟，不把“跑起来”扩大成“性能有竞争力”。

**验收：**声明的预算/容量口径、完整执行、数值/任务质量与已声明层次的请求 I/O 在同一冻结 profile/run 中闭环；失败点和适用边界齐全，不能拼接不同配置的局部通过。controlled-budget PASS 不升级为 host RAM 不足或 disk necessity；更强的物理超容量/磁盘必要性主张另须证明紧凑权重驻留不可行。

### E1 — 安全对照与设计动机（P0）

- [ ] **E1.1 / P0：**将有限驻留复用中的容量超订、在途覆盖、提前回收、过期完成和取消/发布竞态映射到真实路径。**验收：**每种风险有触发条件、owner、受影响数据及源码证据，不只列抽象术语。
- [ ] **E1.2 / P0：**定义普通安全 cache + queue + fence 对照及 Ferrule 的额外合同。**验收：**对照同样保证有界容量、依赖完成和合法回收；不能靠不安全实现衬托收益。
- [ ] **E1.3 / P0：**用最小可复现调度说明事务预留、发布、custody 如何服务超容量执行。**验收：**fixture 与端到端结果分开，说明各机制防止什么、未保证什么；不声称一般系统必然出错。

### E2 — 服务性能与统计（conditional secondary；旧 P0 门禁已 parked）

以下 E2.1–E2.5 的旧 P0 标签及未完成状态保留：计量、身份、质量和完整失败分母仍适用于核心实验；外部性能优势验收和全矩阵不再先于四 RQ。

**E2 最新进度（2026-10-12；仍未勾选）：**已完成 e674 routebatch 的实际 F1→V1 / V2→F2 两个 fresh crossblocks，见[现有结果 summary](experiments/tp2_resident/RESULTS.md)、[true raw/log hashes](experiments/tp2_resident/runs/route-batch-vllm-crossblocks-20261012-01/crossblocks-results.json)及[封存的 artifact/source/API hashes](experiments/tp2_resident/runs/route-batch-vllm-crossblocks-20261012-01/sealed-identities.json)。ctx128/batch8/seq2、concurrency1；每 slot 3 个不计时 QA 请求 +5 measured，每请求实际 5-in/8-out。F 14.317/14.804 对 V 10.376/10.330 output token/s，两次短负载完成速率更高；但 TTFT F≈171–176/V≈133–135 ms、startup F≈350–366/V≈63–66 s、采样 GPU F≈21.04/V≈19.28 GiB/card、host PSS F≈60–61/V≈7–8 GiB，不能说 TTFT/启动/host memory 同时更优。HF-eager GemmEx/routing 与 Marlin 实现差异保留，无 arithmetic bitexact/parity claim；`fairmatch=false`、`exact_speedup=null`。n=2 仅作描述；原预注册至少 5 个独立 pair blocks、长上下文/并发及性能统计验收仍未完成，现为 conditional-secondary/follow-up，而非四 RQ 无条件门禁，**E2.1–E2.5 不勾 PASS**；原始日志/source hashes/历史 raw 不改写。用户运行的实际 release 路径与完整 SHA 见 [CURRENT](experiments/tp2_resident/CURRENT.md)。

- [ ] **E2.1 / P0：**冻结输入、输出长度/停止规则、到达模式、并发、预热、测量窗口、超时和质量门禁。**验收：**配置可重放，startup、first-after-ready、warmup、steady state 分开，事后改参数使用新 protocol ID。
- [ ] **E2.2 / P0：**按模态选指标：自回归用 TTFT、逐 token 间隔、端到端延迟及有效 token/s；非自回归用 request/sample/frame 吞吐、端到端延迟和任务质量。**验收：**给单位、分位数、统计对象与样本数，不强套 token 指标或只报成功快请求。
- [ ] **E2.3 / P0：**保留 planned、dispatched、observed terminal、success、quality-fail、OOM、timeout、cancel、unknown、not_run。**验收：**状态与完整分母可核对；中断时未知在途不能算零失败或未发出，成功条件延迟与全请求完成率同时报告。
- [ ] **E2.4 / P0：**预注册独立 fresh-server 重复和随机配对顺序，以独立 run/block 计算区间。**验收：**同进程请求/窗口不冒充独立重复；说明区间算法、样本量和尾部稳定性，少量重复只作探索观察。
- [ ] **E2.5 / P0：**同时报告合格输出 goodput、总有效输出/总测量时间、失败率和冷启动代价。**验收：**不平均各窗口速率制造吞吐；不完整窗口保留 raw，不推算完整吞吐或 CI；记录 client checkpoint 成本。

**E2 历史状态（2026-10-09）：已有局部证据（TP2 vLLM baseline），不勾选 E2。** 收据 `tp2-resident-single-lifecycle-20261009-001` 是 TP2/PP1/DP1、无 EP、`--cpu-offload-gb 0` 的单次同服务器 Marlin 描述性 baseline：启动 65.7325 s；Hello2 返回 `, I`（1 input/2 output），TTFT 230.452 ms、E2E 311.259 ms；三个冻结 canary 均匹配可见前缀与 usage；QA8 为 5/5，平均 TTFT 130.66 ms、平均 E2E 703.34 ms。五个 measured 请求共 40 个实际输出 token，client wall 3.5363 s，约 11.31 tok/s。采样峰值为每卡 19.284 GiB、server-tree PSS 6.879 GiB；这是 descriptive only，不是 Ferrule speedup 或 fair match。

**2026-10-10 官方工具补记（本轮仅 CPU）：**此前单次 Nsight Systems 2025.3.1 full-model 短 profile 的[双向 v2 审计](experiments/tp2_resident/runs/tp2-official-full-profile-20261010-001/profile-audit-v2.json)只提供局部 request6 correlation 证据：117,232 kernels、26,284 copies、70,014 memsets、1,881,862 APIs，双向 exported-row correlation closed，且 raw/UUID/window mapping 已保留；但 TraceSummaryVerifier 报告 collection incomplete，warmup 有 4 copy + 12 memset activity 缺 API，另有成功 `cuLaunchKernel` API（correlation 687294/687352）在整个 export 中缺 activity。没有证据排除成对漏失，因此 `full_trace_accepted=false`，不构成完整 trace、性能、speedup 或 E2 验收。该 warning 的具体成因（capture boundary、buffer overflow 或其他）未知；没有 explicit CUPTI dropped 记录，不据此断言 overflow。请求窗口也没有可证明的 prefill/decode kernel 边界、GemmEx API 时间或 CPU collective 归因。profile 时间不进入性能比较。见[官方工具用法与限制](experiments/tp2_resident/OFFICIAL-NVIDIA-TOOLS.md)；本轮无 GPU 查询、锁或新采集。[独立 CPU 收尾 manifest](experiments/tp2_resident/runs/tp2-official-full-profile-20261010-001/cpu-closeout-v2/completion-manifest.json)记录恢复核验与新增检查，不覆盖 historical receipt。

该局部 baseline 不能替代 E2 质量、身份、配对或统计验收。以下保留此前 E2 推进路线，仅在恢复性能主张时执行，并非当前下一步；须在**同一 TP2/PP1/DP1 拓扑、dtype/activation/KV/conv/SSM 边界、状态/EOS/缓存与工作量**下运行 Ferrule TP2 Mixed，并完成 full-model quality gate；随后才可按冻结协议做 fresh paired runs、完整失败分母和独立 block statistics。Ferrule TP2 Mixed 已有更新的 bounded full-boundary two-call comparator：Hello/Capital 各 **2,178/2,178** raw fields exact（[CPU 核验](experiments/tp2_resident/full-boundary-audit.json)）；这覆盖 prefill 与 first decode，独立 multistep/lifecycle 六 recipe 现已完成（33 calls / 35,937 metrics exact），fresh greedy4-02 关闭 provenance，旧失败不改写。catalog gate 已打开；test-binary 数值质量与 release-serving HTTP 验收仍严格分开。HF-eager 逻辑 TP2 不等于 Marlin arithmetic；`fairmatch=false`、`exact_speedup=null`，仍不勾选 parity、fairness、speedup 或 E2 完成项。未来配对已在 paper 侧接到既有 authenticated supervisor/observer/HTTP lifecycle（[接入与阻塞项](experiments/tp2_resident/CURRENT.md)）；strict quality const 为 false 时仍先于 lock/launch 拒绝。现已在 gate=true 后完成同 release 两次 HTTP smoke（canary/reuse/cancel/退出/重建）与 fresh F1,V1 / V2,F2：Ferrule 5.5213 / 5.5414 tok/s，vLLM 10.3966 / 10.5621 tok/s，每 server 五个 QA8 measured，全部 20/20；详见 [serving pilot](experiments/tp2_resident/RESULTS-serving-pilot.md)。六个新 server 均 authenticated drain，final 全八卡 idle/release；CPU collective vs NCCL、Mixed/HF-eager reference vs Marlin、Ferrule decode batch 实际为 1 等差异如实保留。不是吞吐 parity 或统计结论，E2.1–E2.5 继续未勾选；旧 11.31 tok/s 不改写、不代表最优。[W8 real-operand exact gate](../../crates/ferrule-model/tests/mixed_routed_down/RESULT.md)有两 word mismatch（16382/16384，native 16384/16384），候选保持默认关闭，未获性能/生产资格；本轮不改 model。

有效 vLLM 日志必须按 profile 保留：该 TP2 run 记录 `MarlinFP8ScaledMMLinearKernel`、`MARLIN Fp8 MoE`，Triton/FLA GDN prefill，以及 fused decode op 未构建时的 Triton GDN decode fallback；并记录 `NCCL_P2P_DISABLE=1` 和 `via SHM/direct/direct` local channels。M7/G10 的另一个 UVA profile 才记录 `UVAOffloader`/30 GiB CPU-offloaded parameters，不能混入本 TP2 no-CPU-offload baseline。上述 backend/offload/fallback/channel 记录不构成 arithmetic equivalence；canary 只证明冻结可见文本与 usage gate，不证明 token IDs、logits、routing、state 或 full numerical equivalence。

失败启动尝试仍保留：baseline-delta runs 003/004 分别因 observer `nvidia-smi` timeout 与 worker 的 `CUDA driver initialization failed` 未到 readiness，未发请求；不得重写为 canary 结果或从分母删除。observer/PID 限制也保留：`owned_gpu_attribution=unknown`，server/worker PID 只是本地身份、不是 host-NVML 映射；另有一个 leaf `/proc/15844/smaps_rollup` 消失的 incomplete host sample，PSS 保持 unknown。早期 001/002 的 Hello2 与未发请求的 canary startup abort 分别保留，不因后续成功改成 PASS。原始证据见 [成功结果](experiments/tp2_resident/RESULTS-single-lifecycle.md)、[机器可读收据](experiments/runs/tp2-resident-single-lifecycle-20261009-001/RESULTS.json)、[001/002 历史](experiments/tp2_resident/RESULTS.md)、[003/004 历史](experiments/tp2_resident/RESULTS-baseline-delta.md)；M7/G10 与历史 target 不改写。

### E3 — 缓存、分层 I/O 与容量控制（P0；NAS backing 测量按 claim 恢复）

- [ ] **E3.1 / P0：**逐层记录逻辑读取、应用/OS/NAS cache、请求期物理网络与块设备读取，以及转换/传输量。**验收：**bytes、latency、带宽、读放大和重复物化均有采样来源；网络流量与 NAS 后端块读分别归因，逻辑 miss、mmap fault 或网络读不能独自证明后备磁盘读；缺测标 unknown。
- [ ] **E3.2 / P0：**冻结缓存预算并设计冷/暖及预算敏感性实验，记录 OS/NAS 缓存控制权限。**验收：**fresh process 不等于 cold cache；无法控制则标 uncontrolled，不清理共享主机/他人缓存，不把预热搬到计时外后声称免费。
- [ ] **E3.3 / P0：**离散域分别计量，统一内存按物理页/池去重，覆盖共享映射、pinned staging、KV/state、workspace 与临时副本。**验收：**账本、RSS/PSS、设备计数和系统峰值差额可解释，host/device 别名不重复相加。
- [ ] **E3.4 / P0：**声明 swap 配置及 swap-in/out，记录 mmap backing、resident pages、page-cache 占用和统一内存迁移。**验收：**主张依赖的驻留总量包含 OS 缓存；若靠 swap 支撑须明确披露，不能混作主动 storage-backed 路径已验证。
- [ ] **E3.5 / P0：**把容量峰值与请求期读入、驱逐、计算 trace 对齐。**验收：**E0 的预算约束与实际 I/O 由同一次运行支持；若主张磁盘必要性，仍需同 run/profile 闭环，不能用另一配置的冷读填补证据。

### E4 — safe blocking / async（core）；其他安全消融（follow-up）

- [ ] **E4.1 / P1：**验收已实现的受控 safe-blocking 对照，显式等待完成后再发布/复用，保留相同容量、缓存和质量合同。**验收：**只改变调度/等待方式；若实际读取策略也变化，单列混杂因素而非归因于 async。
- [ ] **E4.2 / P1：**比较 blocking 与 async 的 critical path、等待、真实 overlap、吞吐及尾延迟。**验收：**带时钟和 operation ID 的 trace 可复算；异步 API 返回快不作为重叠证据。
- [ ] **E4.3 / P1：**按已实现开关逐项评估预取深度、队列、缓存预算、转换/传输策略。**验收：**一次改变一项且重新过质量与容量门禁；不存在的开关记待实现，不虚构消融。
- [ ] **E4.4 / P1：**核对所有对照的 safety contract。**验收：**不关闭 generation、fence、容量约束或 custody 获得“加速”；unsafe fixture 仅用于隔离风险，不参与性能排名。

### E5 — 故障、资源守恒与清理（P0）

- [ ] **E5.1 / P0：**覆盖 reserve、submit、read/transfer、completion、publish、retire、cancel/shutdown 注入点。**验收：**逐点定义可见结果、terminal、owner 和资源责任；fixture 覆盖不冒充服务路径覆盖。
- [ ] **E5.2 / P0：**冻结实际状态分类与守恒式，核对 free/reserved/in-flight/live/retained 等 credits。**验收：**分类互斥或明确子集扣除，无重复计数、双重释放、重复 terminal 或错误发布。
- [ ] **E5.3 / P0：**区分正常可证明 quiescence 与 Unknown completion。**验收：**正常已结束工作回收后临时 credits 清零；Unknown 可合法保留且有 owner/identity/bytes，不能强行清零、伪称 leak-free 或 clean shutdown。
- [ ] **E5.4 / P0：**记录保留资源造成的降容、拒绝新请求及后续恢复条件。**验收：**恢复需独立 completion/fence 或明确的更强隔离证据；进程退出、killpg 或端口释放本身不等于设备 quiescence。
- [ ] **E5.5 / P0：**预检通用 matrix runner 的监督器与 process-EP8 cleanup custody。**验收：**核实冻结版本 capability/containment 门禁；未证明完整进程树有界清理则保持 blocked，不伪造 attestations、不绕过拒绝；thread/vLLM 也不自动获准。

### E6 — 正常路径与观测开销（P1）

- [ ] **E6.1 / P1：**分解 registry 查找、预留/回滚、事件分发、锁/队列、账本内存及 trace/client 持久化成本。**验收：**microbenchmark 与服务端增量开销分开，报告 workload、单位、基线及独立重复。
- [ ] **E6.2 / P1：**在可全驻留区测正常路径成本，在超容量区测收益/成本变化。**验收：**相同质量/安全合同，报告低复用、小请求、带宽瓶颈等无收益或负收益区间。
- [ ] **E6.3 / P1：**比较合法的轻量观测与完整 trace 模式。**验收：**量化 observer overhead；对比包含相同 client 工作，不把记录机制变化归因于引擎优化。

## 6. 基线资格与公平性

- [ ] **B1 / P0：fully resident 适用区。**只在所需表示和状态可完整放入相应域时运行。**验收：**给容量前提和质量门禁；超容量不可行写 N/A（capacity-infeasible），实测 OOM 另记，不算无限加速。
- [ ] **B2 / P0：host-only 适用区。**区分 CPU 执行与 host-resident/device-streaming，紧凑权重及必要状态需能驻留主机物理池。**验收：**分别报告执行合同；host/shared 也放不下时标 N/A，不借隐式 swap 当纯 host 基线。
- [ ] **B3 / P0：正确有界 disk-backed。**选择能完成同一完整模型与质量合同的存储支撑实现，预算包含缓存、staging 和临时副本。**验收：**证明容量与回收安全并记录物理读；若没有合格实现，明确缺失，不以不安全 toy baseline 替代。
- [ ] **B4 / P1：controlled blocking（RQ3 core）。**使用 E4 的安全串行/等待配置隔离 overlap 效果。**验收：**同 weights/固定预算/缓存/质量/安全合同，实际支持且可复现；不是人为插入额外 sleep 的劣化对照。
- [ ] **B5 / P0：公平比较审计。**冻结 checkpoint、precision/rounding、activation/KV、拓扑、输入输出、缓存及超时。**验收：**不匹配项显式标 system-level/non-fair；速度比仅对双方可行且质量通过的匹配点计算，其他为 N/A。

## 7. 相关工作、扩展与表述边界

**2026-10-13 文献 / 2026-10-14 positioning（局部，不勾整项）：**FlexGen（2303.06865）、ZeRO-Inference（DeepSpeed 2022-09-09 技术页，非会议论文）、PowerInfer（2312.12456）、PowerInfer-2（2406.06282）、vLLM（2309.06180）已接入正文/BIB。前三篇 arXiv 页面及 DeepSpeed 页在 `target/research/ferrule-prior-work-20261013/`；vLLM metadata 来自 `target/validation/vllm-cuda12-selection-high/0.28.0-cu129-METADATA.txt`。正文区分 storage-backed prior art、host/device offload、KV paging 与 lifecycle 合同；不称首个 storage inference、不把未审计保证判 unsupported。完整 metadata/citation/guarantee audit 仍 open；SGLang 未核查。

- [ ] **R1 / P1：primary metadata。**逐篇从出版社/会议/arXiv 原始条目核对标题、作者、年份、版本、DOI/URL。**验收：**记录核查日期及来源，不凭二手摘要补作者或发表状态。
- [ ] **R2 / P1：BIB 与论断。**核对正文 citation key、BibTeX 字段、重复条目及 T5 guarantee。**验收：**每项对比有原文位置；设计、实测、未讨论分开，未讨论不能直接写不支持。
- [ ] **X1 / P2 → 声明支持则 P0：多模态。**逐 modality/model/backend/precision/execution profile 建输入预处理、完整执行、输出/任务质量 oracle 和容量记录。**验收：**至少该 profile 的端到端证据齐全才标支持；文本路径不得外推图像、音频或视频。
- [ ] **X2 / P2 → 声明支持则 P0：统一内存。**逐 shared/unified topology profile 核实分配、别名、迁移、容量约束与物理去重。**验收：**完成质量合格的 E0/E3；不能把离散显存+host 的结果平移为统一内存支持。
- [ ] **X3 / P2 → 声明支持则 P0：其他执行路径。**对 backend、precision、process/thread 和 async 覆盖分别维护状态。**验收：**冻结 F32 process pass 不借给 thread、W8A16 或新源码；未验证仅写设计范围/未来工作。

## 8. Artifact 与已有材料入口

以下路径仅作已有材料入口；历史报告结论以各自冻结身份为界，不当作本轮新结果。

- `docs/PERFORMANCE_EVALUATION.md`：质量/统计合同、现有结果及 matrix reporting/custody 限制；执行前核实当前监督器，文档不替代 containment 证据。
- [吞吐口径](../../docs/THROUGHPUT.md)与 [证据账本](EVIDENCE_LEDGER.md)：测量和源码/机制证据入口，使用时核对对应 revision；论文目标以 [README](README.md) 为准。
- `target/validation/qwen35-35b-current/gpu-owner-final-cuda124/REPORT.md`：历史 pinned F32 process-EP8 和 staged subset；短/并发/2K 局部完成、F32 8K 第二次中断，不是完整矩阵或当前版本验收。
- `target/validation/qwen35-35b-current/current-owner/REPORT.md`：后续 owner 记录入口；与前述历史报告分开追踪，不能混用 oracle、源码或 binary 身份。

- [ ] **AR1 / P0：身份闭包。**raw schema 必含 source revision/dirty snapshot hash、binary/build/toolchain、model/shard/scales/tokenizer/preprocessor、client/supervisor、oracle hashes。**验收：**所有依赖能定位到冻结内容，保留 argv/env、hardware/topology、protocol/profile/run/block/request IDs；缺失标 unknown。
- [ ] **AR2 / P0：原始测量字段。**记录 timings/clock、memory/域/峰值、logical/physical I/O/cache/swap、failure/terminal、quality/阈值与输出身份。**验收：**每字段有单位、采样范围和缺测原因，raw 能重建完整失败分母及容量/质量验收，不能用零填 unknown。
- [ ] **AR3 / P1：raw-to-figure。**冻结 schema version、聚合/过滤/配对/区间规则及绘图脚本，生成 F3–F8/T3–T6。**验收：**从归档 raw 一条已记录命令可重建图表；失败和排除项可追溯，禁止手工补数。
- [ ] **AR4 / P1：归档与恢复。**为被忽略的论文目录及 `target/validation/` 证据建立受控备份和 manifest。**验收：**校验 hashes、测试恢复，说明受限模型/数据的合法获取方法；不打包密钥、私人输入或无权再分发的权重。

## 9. 投稿信息与最终一致性

作者姓名、Independent Researcher 身份及邮箱已确认，不再作为猜测项；以下未确认信息不得自动填写。

- [ ] **M1 / P1：ORCID、资助与致谢。**向作者逐项确认。**验收：**只有确认值进入论文；“无资助”“无致谢”也需确认，不能编造机构、项目或致谢对象。
- [ ] **M2 / P1：利益冲突与许可。**确认利益冲突声明、论文/artifact 许可及第三方模型/数据限制。**验收：**声明获作者确认且与实际分发范围一致，不擅选许可或默认无冲突。
- [ ] **M3 / P0：标题—摘要—结论审计。**围绕 bounded storage-backed weight residency，read lookahead/custody 作为实现手段；physical necessity/SOTA 为条件主张。**验收：**每句已支持/更快/更安全都有匹配 profile 证据；区分实测、设计和限制，不扩大模态/内存架构范围。

## 10. 执行波次与停止条件

按四 RQ：冻结合同/路径 → E0/E3 质量/请求 I/O → E1/E4 安全对照/E5 故障；非 E2 先行。W1–W4 为旧依赖索引，按 Submission-core 选用，非全矩阵 gate；pilot 不等于 W2/E0 PASS。

- [ ] **W1 / P0：先定义与冻结。**完成 S1/S2/S3 + F1/F2，冻结 E0 配置、质量合同、容量点、采样与停止条件。**验收：**能回答何物放不下、实际走哪条路径、如何安全完成；监督器拒绝或资源 ownership 不明即停止执行，不改门禁。
- [ ] **W2 / P0：再 quality + E0 基线。**先核对 oracle/逐 profile 质量，再运行合格的 E0/B1–B3，连同 E2/E3 收集完整分母和物理 I/O。**验收：**获得同一冻结配置的完整证据链；quality-fail 不进入“合格性能”比较，但保留负结果。
- [ ] **W3 / P1（E5 为 P0）：再 async/fault/overhead。**完成 E1、E4/B4、E5、E6 与 F3/F5/F7/F8。**验收：**重叠收益、安全/守恒及正常成本可复核；Unknown 保留、blocked 和失败不被清理成成功。
- [ ] **W4 / P1（扩展默认 P2，支持声明升 P0）：最后扩展与重写。**资源允许再做 X1–X3，完成相关工作、artifact、投稿信息，最后重写摘要/结论。**验收：**核心 P0 无证据缺口，P1 有产物，P2 未完成者明确列限制；最终投稿版本仍须另行构建与人工视觉审读；第 1 节记录的当前源码编译不替代最终验收。

## 11. 每项完成记录模板

每个勾选项保存一条记录；未运行不填假命令/结果，未知不猜。子项可继承父项 P0/P1/P2，但证据必须指向自身验收条件。

```text
ID / 优先级：
状态：待定义 / 待核对 / 已有局部证据 / 待实验 / 已验收
source：材料路径、revision/snapshot、实现位置及适用 profile
artifact：冻结目录/manifest/hash；protocol/run/block/request IDs
command：实际执行的完整命令、环境与退出状态；未运行写“未运行”
result：验收条件逐条结果、质量与失败分母、raw/图表定位；无测量写“无测量”
limits：未覆盖范围、unknown/blocked、残留 custody、混杂因素及后续工作
```

<a id="parked-broad-scope"></a>

## 12. Parked broad scope

以下为原执行计划/统计政策的原文保留，不是当前先行门禁，也没有被勾选完成。原 P0 性能工程目标、≥5-block 确认和完整矩阵仅按上方 Submission-core 的 conditional-secondary/follow-up 规则恢复；历史结果仍按原身份解释。

### 当前执行计划与 checklist 映射（2026-10-09；含下方 TP2 局部基线）

**工程顺序：先争取 vLLM 同等或更好的性能，再补齐论文其余清单。E2.1–E2.5 因此升为 P0 性能工程门禁；E0 仍是论文第一主实验和未完成的容量论断。** 两者不能互相替代，也不因 runner、adapter 或文档存在而勾选验收。

论文专用 runner、协议与产物说明入口为 [experiments/README.md](experiments/README.md)（本次不创建或修改该目录）。新增论文专用脚本只放 `papers/arxiv/experiments/`，不放根 `scripts/`；已有通用工具可按冻结身份复用。此处只定计划，具体 argv、预算、限时、schema 和执行授权由 runner 协议冻结，不复用历史输出目录或改收据/hash 绕过门禁。

| 阶段 / 状态 | 最小交付与停止条件 | 对应清单；不等于整项完成 |
|---|---|---|
| TP1 基线预检 / 待核对 | 同一物理单 GPU、TP/PP/EP/DP/PCP=1，完整同一 FP8 checkpoint；Mixed 与真实 vLLM server 分别绑定质量、effective dtype、资源和安全退出证据；不具备资格则 blocked | S2、B5、T1/T3、AR1 |
| 首个 bounded pilot / 待实验 | `p128-o32-c1`、context160，2 个独立 fresh-server 配对 block，固定种子平衡顺序；每 engine 每 block：first-after-Ready 1、warmup 2、measured 2；合计 4 starts / 8 measured requests | E2.1–E2.5、T4、AR2；仅描述性可行性观察 |
| 确认与优化 / 待实验 | 单独 profiling 找瓶颈；修改后重新过质量门禁；新 protocol 冻结至少 5 个独立配对 block、每 run 8 warmups / 100 measured requests、主要 contrasts 和区间方法 | E2、E6、B5、AR3；不借 pilot 宣称显著性 |
| 容量与机制 / blocked：证据未闭环 | S1–S3 后完成同一配置 E0/E3、适用 B1–B4，再补安全消融、故障、开销、图表及归档 | E0/E1/E3–E6、F1–F8、T1–T6、AR1–AR4；论文验收不降级 |

**2026-10-09 状态更新：**上表保留此前 TP1/p128 配对计划，不是已运行结果。当前另有 [TP2 single-lifecycle baseline](experiments/tp2_resident/RESULTS-single-lifecycle.md) 的局部 E2 证据，详见第 5 节 E2 与 [ledger](EVIDENCE_LEDGER.md)；不是执行了上述 pilot。Ferrule 现已有 claim-aware 物理 PP1/TP2/EP1 的 bounded full-boundary two-call comparator PASS：Hello/Capital 各 2,178/2,178 raw fields exact（见 [CPU audit](experiments/tp2_resident/full-boundary-audit.json)）；独立 multistep/lifecycle 六 recipe 已闭环（33 calls / 35,937 metrics exact；fresh greedy4-02 closure PASS，旧 failed 保留），catalog gate 已打开。release `e017a3a8…26c1b0b` 的两次同 binary HTTP smoke 与 fresh F1,V1 / V2,F2 QA8 pilot 均完成（[新结果](experiments/tp2_resident/RESULTS-serving-pilot.md)）；仍仅 descriptive，不勾选 parity 或 E2。

**首个比较的类别与门禁。** TP1 是 boundary-aligned 的 **system-level deployment comparison 候选，不是 arithmetic-equivalent runtime comparison**。Mixed 的 BF16 activation/residual/device logits/KV/conv 与 F32 accumulation/recurrent SSM 必须逐边界核实；vLLM 对应配置和实际运行证据也须核实，不能仅凭 BF16 名称或 dtype flags 认定 Marlin 的 scales、dequant/rounding、算术及中间边界等价。两侧分别通过适合其合同的独立质量门禁，并预注册共同输出/质量资格；不拼接旧 Mixed lifecycle 与新 quality binary。源、binary、模型/tokenizer/scales、client/oracle 身份、实际 input IDs/output usage、EOS、缓存/状态重置、资源、到达模式及 timeout 均须冻结；拒绝或 unknown custody 即停止，不改公平性门禁。

**容量限定。** 首轮采用 soft 7 GiB 资源策略并报告 observed device usage/峰值、采样方法和超限；这是软件准入/观测口径，不是经物理隔离验证的 hard VRAM cap，也不等于 vLLM utilization 参数或总 host-memory cap。旧 smoke 的 ready-time observation 不是峰值；本计划没有新的 7 GiB 实测结果。若完整紧凑权重及 metadata 可放入 host RAM，host-resident/offload 或 GPU cache miss 不证明论文所需 storage necessity；E0 仍未完成。

**工作负载与扩展。** Pilot 使用 raw completions、natural EOS，保留短输出/失败；不与旧 chat/ignore-EOS 协议混合，实际长度不匹配不计算固定工作量比值。分别记录 startup、first-after-Ready、warmup、steady state、drain；fresh process 不等于 storage-cold。报告合格 successful-output tok/s、完成率/全部失败分母、TTFT/E2E 与样本数，SSE 间隔/usage-TPOT proxy 不冒充 token ITL。完整待补矩阵仍为 p128/512/2048/4096/8192 × o32/128/256 × c1/2/4/8；Mixed 当前 context<=1024、batch<=min(32, context)，不支持的长上下文保留 blocked，不能借 F32 结果填格。

**待冻结的预注册判定规则。** 用户目标是至少达到 vLLM，不擅设 5% 等正的非劣效容忍幅度：默认允许退让幅度为 0，主要吞吐比 `R = Ferrule / vLLM` 的目标为 `R >= 1`。确认阶段要求质量/工作量/资源资格通过且完整配对，按每侧合格总输出 tokens / 总测量 wall time 求吞吐再取比；以独立 block 为单位同时重采样两侧，固定 seed、10,000 次 paired percentile bootstrap，报告双侧 95% CI。工程通过规则为点估计及 CI 下界均 `>= 1`；严格性能优势表述需下界 `> 1`。CI 跨 1 或“差异不显著”不能证明相等或 parity；下界恰为 1 不证明严格优势。2-block pilot 不作统计通过判定；5 blocks 只是最低协议数量，不保证功效/尾部稳定性，主要 cells、multiplicity/power 计划、延迟/SLO 与失败率门禁须在确认数据前冻结，不能事后用吞吐掩盖回归。失败/缺失/unknown 配对保留完整分母并抑制 headline ratio/CI，不过滤成功样本。该规则只支持声明过的 system-level 比较，不升级算术等价或 E0；任何其他 parity margin 必须另获用户确认后注册。
