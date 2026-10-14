# 初稿 claim–evidence–gap 台账

本台账非编译输入/公开 artifact；保留历史记录，2026-10-15 同步 bounded residency、已实现调度与 receipt chronology。本轮改稿件/README/checklist/ledger（含历史appendix），只读实际9/9收据、CPU paper build/review，未运行或查询 GPU、未改 experiments/raw/production、无 commit；历史 PASS 不借用。

## 来源与可信度

- 历史讨论：`docs/temp_ferrule_refactor.md:13558–13858,14568–14616`。
- 当前论文目标：[README](README.md)；支持与 source-pinned 验收：[CURRENT_ACCEPTANCE](../../docs/CURRENT_ACCEPTANCE.md)。
- 评估合同/历史摘要：`docs/PERFORMANCE_EVALUATION.md`；本次不重跑或重新验收。
- 实现事实：下表源码与架构文档的静态核对，**不是**对所有路径的证明或运行验收。
- 2026-10-02 历史调查时 Git HEAD 为 `8a2fef862c7205373de6ee004f4d1ec46799aa6d`，工作区存在大量修改及未跟踪文件。这个 HEAD **不能代表**所读工作树或归档实验源码，不能作为单独的实验 revision 引用。
- 源码路径从仓库根定位，Markdown 链接从本目录解析；行号仅历史导航，发表需内容哈希。绝对 artifact pins 保留，不全局替换旧路径。

## 已确定的写作信息

- 标题：**Ferrule: Bounded Storage-Backed Weight Residency**。
- 作者：**Lu Huanbing**；单位：**Independent Researcher**；联系邮箱：**luhuanbing084@gmail.com**。
- ORCID、资助信息未提供，不推测，也不把未知填写为“无”。
- 核心目标：**bounded storage-backed weight residency**；controlled-budget 与 physical-capacity 分报。后者及 disk necessity 是更强 conditional claims，未验证且非当前标题/结论前提。事务管理服务物化/安全复用，非数值 exactness、持久化事务或性能保证。
- 容量按物理 pool/共享页去重，不预设 GPU+host 相加；独立副本各计实际占用。
- 全模态、统一内存、全异步为设计范围，逐 profile 验收；不称全链路无阻塞或 measured overlap。
- 标题/作者/正文主线与 checklist 已同步；写作同步非能力验收。

## 当前主张：bounded storage-backed residency；更强 necessity 另行验收

| 目标 claim / 待核对项 | 当前证据状态 | 必须补齐 / 禁止扩大 |
|---|---|---|
| 更强 conditional claim：模型超过去重可用物理驻留容量时，backing storage 完成推理 | **尚未建立完整证据** | 固定模型、模态、表示、算术合同、设备、物理 pool、拓扑及 snapshot；报告去重规则、施加预算、逐池高水位、存储访问及对应模态的端到端完成和数值/质量验收；prefill 与多步 decode 是当前语言解码实例 |
| 完整紧凑权重不能全驻留，硬盘对目标配置确有必要 | **完整磁盘必要性证据未确认** | 检查量化/压缩权重及必要 scales 等数据能否全驻留可用物理内存；仅执行展开字节超限不能证明磁盘必要性 |
| 事务性质资源管理支撑有限容量内的按需运行 | 下表仅提供限定的机制证据 | 还需同一容量配置下的集成路径、资源守恒与安全消融；协议测试不替代容量或数值验收 |
| 多模态与组合模态路径 | 设计范围与待落实事项；**逐模态支持及验收待核实** | 按模态、设备、profile 和 snapshot 建立实现与证据矩阵，不从 CUDA 语言解码结果推导已验证全模态 |
| 统一/共享内存的物理驻留容量与账本映射 | 设计已考虑；**具体配置的实现映射和容量证据待核实** | 核对物理 pool、共享页、独立副本、迁移与预算，去除重复计数；统一地址/API 不等于单一共享物理池 |
| 全异步路径、阻塞边界与实际 overlap | 设计范围与待落实事项；**端到端边界及重叠证据待核实** | 分别验证提交/完成语义、阻塞点、时间线及设备活动；不声称全链路无阻塞，不用 custody 安全性替代 overlap 验收 |

E0 的 bounded residency/quality 为核心；physical insufficiency/necessity 非无条件 gate。当前按 RQ1 驻留/质量、RQ2 分层 I/O、RQ3 同合同 blocking/lookahead、RQ4 集成故障/守恒收敛。E2 为 conditional secondary，vLLM 全胜/大矩阵非无条件 gate。全驻留 EP8、host-only、host 可容纳紧凑权重的 offload、logical read/cache miss 均不证明 physical-capacity 或 disk necessity。

### 必须区分的容量口径

1. **磁盘 checkpoint 文件大小：** 指定编码与文件集合的逻辑字节、物理占用；不能直接当作 RAM/VRAM 需求。
2. **紧凑权重全驻留需求：** 完整量化/压缩表示及其必要辅助数据能否放入可用物理内存；必须独立于执行展开字节检查。
3. **运行表示与全驻留需求：**列 decode/dequant、packing/padding、共享/副本；紧凑表示与执行展开分别计量。仅展开超限而紧凑权重可驻留时，内存内展开仍可能可行，不证明磁盘必要性。
4. **去重可用物理驻留容量：**物理 pool/共享页只计一次，独立副本各计；统一地址/managed-memory API 不证明单一共享池，需拓扑/迁移/驻留核对。离散 GPU+host 只是实例。
5. **实际总工作集：** 驻留权重、KV、activations、workspace、staging、在途/retained buffers、metadata、固定分配及相关 OS page cache。账本覆盖域不自动等于部署总内存，避免共享页重复计数，并明确 cache/swap 条件。

比较注明表示/物理去重规则，合计容量不代替各池放置约束；人为预算与物理不足分报，KV/并发超限不等于权重超容量。

## 设计范围与证据边界

- **多模态：**逐模态/设备/backend/profile 建立执行、质量与容量证据，文本结果不外推。
- **统一/共享内存：**按物理驻留/别名去重，非逻辑地址、映射长度或可见设备相加。
- **全异步：**提交/完成 API 非 overlap/无阻塞；需真实时钟、依赖及设备活动证据。
- **事务 custody：**区分 logical terminal、provider completion、physical quiescence/release；保护仍被引用容量，非性能、liveness 或 overlap 保证。
- **实现边界：**CUDA 解码/KV/backend 证据仍限其模态/设备/profile/snapshot/binary，不外推。

## 支撑目标的核心机制

| 可使用的限定表述 | 本地证据 | 仍需补齐 / 禁止扩大 |
|---|---|---|
| Rust owner 协调逻辑状态与资源生命周期；backend 负责物理操作 | [职责与资源合同](../../docs/ARCHITECTURE.md#1-system-boundary) | 不等于多 GPU 中只有一个 physical owner；需要冻结配置对应的路径图 |
| General registry 按完整 exact key 合并兼容的物化请求 | `crates/ferrule-common/src/io_protocol.rs:340–354`; `crates/ferrule-runtime/src/io/registry.rs:2250–2330` | 不可自动推广至 dedicated Qwen expert 或 live HostKV 全部路径；物理 I/O 收益需测量 |
| Completion 推进前核对 operation/key/stage 等身份 | `crates/ferrule-runtime/src/io/registry.rs:1621–1656` | 需要故障测试矩阵；接口机制不等于真实设备完成正确性的通用证明 |
| Broker 对 covered domains 做 claims validation，再更新账本 | `crates/ferrule-runtime/src/scheduling/resources.rs:491–545,561–607,641–710` | 非全进程 RSS 上界；应列出 fixed allocations、allocator reserve、metadata、OS cache |
| Submitted custody 与 release 分开；provider return 不自动释放 credits | 同上 `resources.rs:561–607,641–710` | 需验证组合失败与资源守恒，不能声称所有资源都使用同一 broker |
| Resident ending Pending 保留 payload，backend complete 后进入独立后处理阶段 | `crates/ferrule-runtime/src/engine/driver/resident.rs:15–39,482–527` | 局部阶段机不等于跨崩溃或跨 owner 原子提交 |
| CUDA decoder 的 KV working state 在 commit complete 后安装；prepared physical KV 区分 publish/retire/finish | `crates/ferrule-model/src/decoder/transaction.rs:493–510,534–575`; `crates/ferrule-backend/src/cuda/operators/kv/transaction/prepared.rs:5–55,142–159` | 当前是 CUDA decoder/backend 路径的限定证据；`prepare_publication` 不是 all-owner physical ready vote，也不推广为全模态或全设备保证 |
| Logical cancellation/terminal 不等于 physical quiescence | `crates/ferrule-common/src/io_protocol.rs:879–901`; `crates/ferrule-runtime/src/engine/driver/shutdown.rs:92–135`; `crates/ferrule-runtime/src/engine/driver/resident.rs:48–83` | 已开始的不可逆 physical Publish 仍可完成；terminal delivery 不是 fence |
| Unknown completion 保留 custody；cleanup 失败保留后续清理载荷 | `crates/ferrule-model/src/decoder/kv/prepare_error.rs:1–14`; `crates/ferrule-runtime/src/engine/driver/resident.rs:669–687`; `crates/ferrule-runtime/src/io/registry.rs:1580–1608,2220–2234` | 可能无限期占用容量；不推出 eventual reclamation、general recovery 或 availability |

### Cancellation 与 physical Publish

[架构事务合同](../../docs/ARCHITECTURE.md#2-execution-transactions)区分 caller-visible cancellation 与不可逆物理发布：

> Accepted cancellation determines the caller-visible terminal outcome but does not revoke an already-started irreversible physical Publish.

该限定不承诺故障恢复或提前释放 custody；故障实验见 [性能协议](../../docs/PERFORMANCE_EVALUATION.md#fault-model-and-budget-conservation)。

## 数值与性能：允许保留的状态，不直接作为主文结果

| 记录 | 本地证据 | 解释边界 |
|---|---|---|
| 2026-10-07 adapter-extraction F32：6/6 PASS，7,946,240 logits，零 failures/nonfinite | [REPORT](../../target/validation/adapter-extraction-readiness-20261007/snapshot-26604d6076c14cd1-1791384636495274444/REPORT.md)；同目录 `ACCEPTANCE.json` / `RELEASE.json`；source `26604d6076c14cd10510ff22e7434f7453d4bfe3ba2e93f34fce6753b6d52654`，binary/oracle pins 见收据及 [索引](../../docs/CURRENT_ACCEPTANCE.md) | PP2TP2EP2 3,973,120（含 fork/recreate）及分别测试的 resident thread/process EP8 各 1,986,560；resident 各有 route/state 检查，pipeline 无 raw route/state oracle；非四拓扑全面重验、当前 checkout 或性能 PASS；GPU 10 秒采样，host peak 未实测 |
| 2026-10-07 Mixed prepared-attention 原 full40 与 bounded fresh quality PASS | [FINAL.json](../../target/validation/bf16-attention-model-current-20261007/FINAL.json)，状态 `PASS_mixed_prepared_attention_original_full40_and_fresh_quality`；GPU-tested source/binary/reference pins 与 post-GPU CPU-only 变化分别记录 | 原 full40 804 metrics / 1,986,560 logits；fresh QA/arithmetic/Chinese 5,427 metrics，其中 27 logits metrics / 9,187,840 elements，零 tolerance failures；free-greedy、QA fork/reset 和另一个 fresh owner QA 通过；不是自动故障 replay 或通用任务质量，arithmetic 固定 token budget 内未答完 |
| 较早 Mixed public-serve：Hello lifecycle PASS | [FINAL.json](../../target/validation/mixed-public-serve-owner-20261007/admission-deadline-fix/FINAL.json)，`MIXED_PUBLIC_SERVE_LIFECYCLE_PASS` | 另一 source/binary，早于最新 attention；不能拼成最新 binary 已有 public serving + quality/performance PASS；page cache 未控制，restart 非 storage-cold |
| 历史冻结 current-owner F32 process-EP8 full40：1,986,560 logits；max_abs 4.38690185546875e-5；记录无 violation/nonfinite | [原 current-owner 报告](../../target/validation/qwen35-35b-current/current-owner/REPORT.md) | 仅该 profile、输入与冻结快照；不是本轮重跑，不借给 W8A16 或之后工作树 |
| W8A16 独立 single-GPU full40：strict gate FAIL | [W8 strict 失败范围](../../docs/PERFORMANCE_EVALUATION.md#15-independent-full40-w8a16-oracle-exists-strict-acceptance-failed); `target/validation/w8a16-full40-independent-20261002/SUMMARY.json:3–7,53–66,74–80` | oracle 已存在；不是“缺少 oracle”；不是 EP8 serving 验收 |
| 上述 W8A16：atol=rtol=2e-4；1,837,493 / 1,986,560 logits 超差；max_abs 0.2965252995491028；4 个 ordered route-ID mismatch | 同上；`target/validation/w8a16-full40-independent-20261002/quality.json:180–200` | 8/8 argmax 一致不能推翻 logit、route、state 的严格失败；不修改阈值去追求通过 |
| matched-client current-owner 性能窗口已存在 | [冻结 matched-client 结果](../../docs/PERFORMANCE_EVALUATION.md#14-current-owner-matched-client-results--2026-10-02) | 同一 server 的窗口，不是独立随机重启；没有可直接宣称的显著性、稳定 parity 或 W8A16 质量合格性能 |
| vLLM 实际部署参考存在，但 arithmetic/topology 不匹配 | [历史部署及精度差异](../../docs/PERFORMANCE_EVALUATION.md#11-historical-aligned-evidence-snapshot-through-v4) | 不生成等精度 runtime speedup；区分服务端与 benchmark client |
| 全专家 prewarm EP8 的吞吐记录 | [历史 v4 测量及限制](../../docs/PERFORMANCE_EVALUATION.md#11-historical-aligned-evidence-snapshot-through-v4) | 不证明请求期存储物化、超过去重物理驻留容量的推理或磁盘必要性 |

W8A16 profile 为 BF16-RNE numeric weights/inputs、原 multiplicative scales、F32 model/KV/state boundaries；不能简称“全 BF16”。早期 divergence 分析及 kernel-order replay 不替代独立 oracle，也不撤回 strict FAIL。

Mixed 的 PASS 只属于 `Bf16ModelMixedF32Accumulate`：E4M3FN + BF16 scales，BF16 activation/residual/device logits/KV/conv、F32 accumulation/SSM；logits 非 bit-exact（fresh max abs 0.0625），state/route 数值相等不是额外 signed-zero bit audit。uniform H/Hkv/D、context<=1024；public serving 另需 batch<=min(32, context)、单 owner/resident KV/lazy experts。更早 scalar/attention6 和 F32 failed attempts 保留历史，不能替换原 reference、放宽阈值或擦除独立 W8A16 strict FAIL。

## 当前四 RQ 与本轮证据（2026-10-15）

### 实现 → observable → qualification

| 已实现边界 / 静态证据 | 可报告 observable | 不能推出 / pending gate |
|---|---|---|
| [checkpoint worker](../../crates/ferrule-model/src/checkpoint/read_pipeline.rs)：1 positioned-read worker、depth1 queue、two charged slots、immutable typed tickets、source verified completion | submit/start/read/verify/take、requested/proved/completed/wasted bytes、slot/payload/stack/Unknown | host intervals 非 GPU execution，queue cap 非全部 RSS，join 非 GPU fence |
| [source v2](../../crates/ferrule-model/src/checkpoint/source.rs)：Unix dev/inode/length/mtime/ctime；原 snapshot 进入 plan | exact ordered extents 与 source/image generation | metadata fingerprint 非 payload authentication；不改历史 source receipts |
| [expert sequence](../../crates/ferrule-model/src/transformer/expert_reads.rs)：canonical gate/up/down compressed artifacts、quota、allocation liveness、deferred speculative errors | exact image/layer/expert、demand wait、used/wasted reads、packed/prepared/metadata/requested-stack charges | escaped aliases 保持 charge，Unknown sticky；弱引用/control allocation 与 allocator RSS 不能由 quota 清零推出释放 |
| [owner split](../../crates/ferrule-model/src/transformer/standard/cuda.rs)：current upload 后 read-window → owner compute submission | bucket/submission identity；correlated API/kernel/copy activities 待采集 | whole forward 同步；host span intersection 非真实 GPU overlap |
| [CLI options](../../crates/ferrule-cli/src/args.rs)、[prepared factory](../../crates/ferrule-composition/src/model_factory/bf16_serving.rs) | requested/effective mode 和 source-bound prepared lowering | metadata admission/CPU fixture 不等于 same-CLI HTTP/full quality PASS |

Blocking 与 lookahead 均 explicit opt-in，使用同 worker/budget，legacy disabled 非 control。
study host64 MiB 内含 metadata8 MiB/requested stack2 MiB，device64 entries/320 MiB charge
含 scratch8 MiB；`2 P_max + M + S <= H`，real OS/TLS/allocator headroom unknown。
Passive observations 不 repair poison/refund/join/伪 Closed；cancel/drain/join 只证 named
host custody，不 heal GPU Unknown。[CPU fixtures](../../crates/ferrule-model/tests/expert_read_pipeline/mod.rs)
覆盖真实 worker、source/ticket、byte equivalence、alias quota、speculative error/cancel/drain；
本轮未重跑这些tests；下面新receipt为两arm同cut factory mechanism/numerics actual qualified，不升级CLI/HTTP/performance/physical release。

### 2026-10-14 chronology（实际结果，禁止合并 PASS）

| Receipt | 数值结果 / denominator | Wrapper / source relation |
|---|---|---|
| [Middle quality -01](../../target/validation/storage-middle64-current-quality-20261014-01/FINAL.json) | before-inference `cuInit (3)`，0 metrics/logits；3 scheduled/1 dispatched/1 failed/2 unissued/0 retries | fresh/cold unissued；非 10-13 DEVNULL wrapper failure |
| [Full40 -02](../../target/validation/storage-middle64-current-quality-20261014-02/FINAL.json) | **numerical PASS**，804 metrics/1,986,560 logits/4 token checks，0 failures/nonfinite | `wrapper_accepted=false`；first post-idle 0 MiB/1%/no apps，原 strict failure 不修复 |
| [Retained fresh -03](../../target/validation/storage-middle64-current-quality-20261014-03/FINAL.json) | **numerical PASS**，5,427 metrics/9,187,840 logits；27 common-history/27 feedback/9 reset predictions | `wrapper_accepted=false`；first post-idle 0 MiB/6%/no apps，后续 idle 不改写 |
| [Cold -04](../../target/validation/storage-middle64-current-quality-20261014-04/FINAL.json) | **numerical + wrapper PASS**，1,809 metrics/3,228,160 logits；9 common/feedback/reset each | authenticated drain/new stable-idle v2 accepted；new immutable 3,533-input cut `5662b7d2…`，old primary `same_source_cut=false`、wrapper rejected |
| [Frozen CLI quality -01](../../target/validation/storage-observability-cli-quality-20261014-01/FINAL.json) | before-inference `cuInit (3)`，0 metrics/logits/token checks；retained fresh unissued | `quality_accepted=false`；same3,533-input cut but test/CLI executable different，no candidate HTTP/pilot |

full40/fresh old cut `fbf0c225…` 与 cold cut 不同；new stable-idle acceptance 非原
strict wrapper 的 repair，也不是 physical fence。固定 tolerance 不改；0 metrics
初始化失败非 numerical mismatch。三个历史 numerical PASS 不组成 current all-three
same-cut/CLI qualification。

### Current enabled factory qualification：actual9/9（替代旧RUNNING handoff）

**实读来源：**[SUMMARY](../../target/validation/expert-read-current-quality-20261014-02/SUMMARY.md)、
[FINAL](../../target/validation/expert-read-current-quality-20261014-02/FINAL.json)、
[receipt-audit](../../target/validation/expert-read-current-quality-20261014-02/receipt-audit.json)。
source HEAD `033519c3def098ce417e4b6ff0f141a9f30bddb1`，exact1,484 committed members；
absolute pins digest `a4dceb084e21b80f6b45ff7ac980c6fa80ac8f17e902fd28c65f3d5bc7bb0efa`，
source archive `ac072e0d1bac7efcc85b7f26ce68723f412b0e22b5ee868104b1f6073dd2a82f`。
all9 scheduled/issued/passed once，0 conditioning probes/warmups/retries/captures/unissued。
本轮只读report/hash核对，未执行/查询GPU，不改任何producer receipt。

| 当前gate | 两schedule各自实际结果 | scope / denominator |
|---|---|---|
| Full40 ×2 | 804 main metrics /1,986,560 vocab /2,560 ordered route IDs，各PASS | diagnostic states observed；4 prediction calls；不转借ordinary |
| Retained fresh ×2 | 5,427 main metrics /9,187,840 common-history values /64 actual fingerprints，各PASS | diagnostic routes/states；64覆盖四stream，非64×full-vocab额外总数 |
| Ordinary ×2 | 27 main output metrics /9,187,840 common-history values；all64 actual word/shape/greedy fingerprints match own-schedule current diagnostic primary | no diagnostic observer；routes/states unobserved/false，相关fields null；27 metrics不与5,427混总 |
| Cold ×2 | 1,809 main metrics /3,228,160 common-history /28 fingerprints，各PASS | all28 actual fingerprints match own current primary，report/config/oracle绑定；不是旧cold source mismatch |
| Constructor ×1 | 4 boundary/schedule cells /128 finite vocabulary values total；32/cell、6 actual verified reads/cell | tiny constructor/provider，不是full-model oracle |

**Per-stream，绝不合并：**每fresh/ordinary schedule common27/9,187,840，
free-feedback27/9,187,840，reset9/3,228,160，fork1/248,320；每cold schedule
common/free/reset各9/3,228,160，fork1/248,320（calls/values）。64/28 fingerprints是这些
streams的actual输出指纹，非additional logits。原fixed tolerance `0.00025+0.012*abs(reference)`
与observed exact IDs保留；同history fork replay非divergent KV COW/broad task quality。

**真实worker：**每schedule full40 verified/completed reads2,089；fresh25,116；ordinary25,116；
cold11,354（constructor6/cell）。sampled operations/positive-worker-call分母与actualread总数分开；
不从host read/proved bytes推physical NAS访问或performance。

**Qualified accounting point，非frontier/physical peak：**composition host charge高水位16,777,984 bytes：
`2*3,146,112 + 8,388,608(metadata) + 2,097,152(requested stack)`，<=64MiB。
`C_host=B_packed+B_artifact+M+S<=H`为covered逻辑合同，payload alias持quota、Unknown不重复charge，
弱backing/TLS/allocator/RSS独立。allocator reserved高水位4,106,223,616 bytes <=7GiB，但getter
可poll retirement fences/release blocks；passive typed snapshots allocator仍unobserved。
source-qualified owner/endpoints不是连续全部physical峰值、统计上界、free observer或release proof。

**Runtime/cleanup scope：**actual PID/birth mappings pin CUDA12.4.99；SM86 compiled、NVTXfeature compiled、
`FERRULE_PROFILE_NVTX=0`。all9 natural exit/authenticated ECHILD/recorded owners absent，all8 stable-idleV2，
final3 zeroMiB/zeroUtil/noApps，localUUIDlocks released/reprobed；physical CUDA release **unobserved**。
CLI SHA `de515995…`、constructor `cb9b8c99…`、composition `8810c341…`不同executable；same source/native/runtime
不是mathematical binary/CLI HTTP或machineidentity equivalence proof。该九gate无pairedtiming/NFS/NVTX-overlap采集；随后failed CLI startup的shutdown client样本另列下方。

**Portable archive actual：**[qualification_v1](experiments/storage_demand/qualification_v1/README.md)，committed
`17b16fa059c3508788e081081d70452639a4e19f`；完整249files/5,642,720 bytes（5.38MiB），其中240 manifest-bound records/5,343,269stored bytes。
本recovery read-only offline checker与37tests实际PASS，无GPU/网络/原target依赖。
Manifest/hash pins含九gate同033519c；external binaries/model/oracle/full-source payload不复制/不重验。
archive copy非新run/CLI qualifier。历史failure保持原表及`sections/evaluation-history.tex`。

### Actual paired CLI failure and prospective V3（不修复旧study）

[paired-final](experiments/storage_demand/paired_read_v1/artifacts-current-20261015-01/paired-final.json)/
[closeout](experiments/storage_demand/paired_read_v1/artifacts-current-20261015-01/closeout.json)：
`permanently_stopped_no_retry`，P1-blocking returncode1，Ready=false，未timeout；
`invalid runtime request: experimental qwen35-mixed-bf16 serving: prepared source profile changed`。
历史immutable source与closeout记录serve.rs:38 unconditional `mixed_tp2_cache_requirements()`及mixed_tensor_serving.rs:275–280：
TP1 enabled numeric prepared_source_plan Some，被non-TP2 accessor拒绝；发生在model worker/CUDA owner启动前，
无cuInit错误，非numerical mismatch。source-plan query fix已committed `86c58a0`（主owner handoff）；owner报告CPU/hidden-CUDA `CudaWorkspace` checks PASS，非本文重跑。
本次只读current serve.rs/catalog selection确认改为selected `serving_resource_requirements`，不是旧unconditional accessor；未查Git/改production。
该source修复不追改旧failure，也不构成new CLI/HTTP质量、Ready或performance PASS。
1issued lifecycle/0complete/9unissued；120scheduled/0HTTPissued/0terminal/0success/0HTTPfailed/0censored/
120unissued；50QA全unissued，all5failed/incomplete pair rows retained。quality metrics未观测，
E2E/TTFT/goodput/pairedaggregate/logratios/interval均null，不填0、不complete-case/filter/retry/replacement。
Ready runtime maps/NVTX实际process环境/positive worker reads/readcharge/final ledger未观测或unknown；
authenticated ECHILD/owners absent/all8idle/locks released+reprobe不修复owner ledger或physical release。
NFS只有prelaunch/shutdown shared-client样本；shutdown delta0只属该shared窗口，无Ready/startup-to-Ready/
HTTP request windows，不推0request bytes或owned NAS network/block traffic。

[Corrective V3](experiments/storage_demand/paired_read_v1/PLAN_V3.md)独立CPU-only preregistration：
显式`--decode-cohort-target 1 --decode-cohort-max-deferrals 0`，不是stopped CLI隐式4/3/default变更。
两arm只schedule不同，orderB/L,B/L,L/B,L/B,B/L、5pairs/10lifecycles/120requests/50QA。
7200s绝对envelope/650s lifecycle/100s finalreserve，STOP/no-retry/all5-or-null；V3零attempt，
不是resume/failedpairreplacement/denominatorrepair。explicit flags本身不修复accessor；86c58a0 source fix另行完成，
新candidate/binary、same-cut九gate及V3 Ready/HTTP/runtimebinding仍pending；旧033519c仅config/oracle参考。
没有当前performance/CI/significance/NVTX-overlap结论。
- **RQ1 PARTIAL：**current factory/profile numerical/provider subgate actual9/9 closed；capacity frontier、完整physical working set attribution与CLI/HTTP open；historical observations保留范围，necessity不主张。
- **RQ2 scoped open：**[TP1 wrapper -01](experiments/storage_demand/runs/storage-demand-tp1-middle-20261013-01/report.json)
  DEVNULL `AttributeError`，4 planned/0 dispatched/4 not-run；[real -02](experiments/storage_demand/runs/storage-demand-tp1-middle-20261013-02/report.json)
  n=1 DEBUG、4/4 HTTP、3/3 limited canaries、Hello HTTP/usage only，logical actual。
  四 plots 仍为旧 DEBUG invocation counters，非 current pipeline stats。shared NFS-client
  mountstats helper已用于failed CLI的shutdown-only shared窗口，无Ready/request aligned measurements；physical NAS
  network/block bytes unknown，page cache uncontrolled。scoped logical/client report 可闭环，
  不推 physical necessity；missing/reset 留 null，非 storage-cold。
- **RQ3 stopped / V3 pending：**实际原study与新CPU-only corrective protocol见上；all5pairs retained，
  120/50全unissued，无估计/CI。factory机制/数值子门禁qualified，不是CLI/HTTP/performance/overlap PASS。
  default disabled/no-worker-read CLI非enabled control；GPU overlap另需kernel/copy相关证据。
- **RQ4 open：**[resource observability](../../crates/ferrule-model/tests/resource_observability/mod.rs)/
  [token trace](../../crates/ferrule-server/tests/unit/worker_token_tracing.rs)保留既有 CPU evidence；
  enqueue 非 HTTP consumption/fence，summary 非 final release。real -02
  `resource_contract_complete=unknown`、raw `gpu_quiescence=unverified`；集成 fault/final closure pending。
- **独立 baseline pilot 未执行：**[Pilot V2](experiments/storage_demand/newversionedprotocol/protocol.json)/
  [CPU receipt](experiments/storage_demand/newversionedprotocol/validation-cpu.json)：5 lifecycles/60 planned/25 measured QA，
  ceiling12；不并入 RQ3 的10 lifecycles/120/50，不补执行数据/统计 PASS，不改 frozen seals。

**历史 E2 计划 parked：**2026-10-08/09 E2 升 P0/先达到 vLLM、TP1 p128/o32/c1 2-block pilot、≥5 paired-block 95% CI/零退让仅性能升主 claim 时恢复（[原规则](PAPER_COMPLETION_CHECKLIST.md#parked-broad-scope)）。共同质量/工作量/资源、完整分母及延迟/SLO 先冻结；比值与 CI 下界 >=1 才过旧门禁，严格优势需下界 >1，“不显著”非 parity。soft 7 GiB 非 hard cap，system-level 非 arithmetic match/E0/disk necessity。

## TP2 resident vLLM baseline（2026-10-09；局部 E2 证据）

成功生命周期收据为 [`tp2-resident-single-lifecycle-20261009-001`](experiments/tp2_resident/RESULTS-single-lifecycle.md)，机器可读主收据为 [`RESULTS.json`](experiments/runs/tp2-resident-single-lifecycle-20261009-001/RESULTS.json)。这是 TP2/PP1/DP1、无 EP、无 CPU offload（`--cpu-offload-gb 0`）的 vLLM Marlin baseline；一次启动的同一 server 依次完成 Hello2、三个 canary 与 QA8。模型为 Qwen3.5-35B-A3B-FP8（text-only），两张 RTX 3090；冻结 profile 为 BF16 activation/KV/conv、F32 SSM，raw completions、natural EOS、prefix cache off、eager、concurrency 1。launch ID 为 `d826244b213947d4b5ce2843f1537961`。

| 项目 | 观测 | 证据边界 |
|---|---:|---|
| startup | 65.7325 s | 单次 lifecycle；不代表独立冷启动分布 |
| Hello2 | `, I`; 1 input / 2 output；TTFT 230.452 ms；E2E 311.259 ms | 请求成功；不是 full-model numerical quality |
| 三个 canary | 3/3 visible-prefix + usage matches | 仅冻结可见文本与 usage；无 token-ID/logit/route/state/full numerical oracle |
| QA8 | 5/5 successful measured requests；mean TTFT 130.66 ms；mean E2E 703.34 ms | first QA-phase request excluded，另有 2 warmups；同一 server |
| measured work | 40 actual output tokens / 3.5363 s client wall ≈ 11.31 tok/s | client overhead included；不是 larger-load、cold-cache 或 paired throughput |
| sampled resources | 19.284 GiB/card GPU peak；6.879 GiB server-tree PSS peak | sampled whole-device/PSS observations；not exact peaks or a hard partition |

**2026-10-09 历史局部状态：**QA8 descriptive only；`shared-full40-01`/`shared-tiny-01` 为 PP1/TP2/EP1 separate gate/up 修复后的 logits/lifecycle gate（1,986,560 有限 logits 零误差；[CPU audit](experiments/tp2_resident/claim-gates-audit-20261009.json)）；当时 all-boundary/state、multistep/serving 未放行。后续 TP2 材料见 [checklist E2](PAPER_COMPLETION_CHECKLIST.md#e2--服务性能与统计conditional-secondary旧-p0-门禁已-parked)，不追改旧收据。HF-eager TP2 非 Marlin arithmetic；`fairmatch=false`、`exact_speedup=null`，无 fair speedup/parity。E2 现为 conditional secondary，E2/T4/B5/统计 checkbox 不勾。

原始入口：[sealed pins](experiments/runs/tp2-resident-single-lifecycle-20261009-001/prepared.json)、[server log](experiments/runs/tp2-resident-single-lifecycle-20261009-001/session/server.log)、[canary gate](experiments/runs/tp2-resident-single-lifecycle-20261009-001/session/canary8/gate.json)、[QA8 HTTP receipt](experiments/runs/tp2-resident-single-lifecycle-20261009-001/session/qa8-V1/http.json)。QA8 每请求 5 input / 8 output，全部 finish reason 为 `length`，natural EOS policy 未改；这里的 first QA-phase request 不是服务器 first-after-Ready。server/JIT/page caches 未清空，不能把更短 startup 解释成优化。

### Effective runtime evidence and profile separation

TP2 成功 server log 的有效选择包括 `MarlinFP8ScaledMMLinearKernel` 和 `MARLIN Fp8 MoE`；GDN prefill 为 Triton/FLA，因 `fused_gdn_decode_post_conv_mtp` 未构建而 fallback 到 Triton GDN decode。NCCL 日志明确 `NCCL_P2P_DISABLE=1`，local channels 为 `via SHM/direct/direct`；Socket plugin 的初始化不被解释为 local inter-GPU payload 使用 Socket。M7/G10 的独立 UVA profile 才记录 `UVAOffloader` 与 `Total CPU offloaded parameters: 30.0`，其日志分别在 [`M7 vLLM log`](experiments/runs/m7-feasibility-exclusive-002/vllm/server.log) 与 [`G10 vLLM log`](experiments/runs/g10-exclusive-001/feasibility-vllm/server.log)；这些不属于 TP2 no-CPU-offload baseline，也不能与其拼接。Marlin packing/scales/dequantization、GDN fallback、activation/KV/conv/SSM 边界和 reduction 差异意味着没有 arithmetic equivalence claim。

TP2 的显式 zero CPU offload 和 installed-source defaults 指向 `NoopOffloader`，但其直接 debug log 未观察到；没有逐 tensor residency audit，不能从日志缺少 offload 推出所有 tensor 放置已证明。UVA selection 同样不独自证明 zero-copy。checkpoint 身份仍为 metadata hashes + shard stats，非完整 payload hash；[loaded-library evidence](experiments/runs/tp2-resident-single-lifecycle-20261009-001/loaded-library-evidence.json) 同时看到 CUDA12/NCCL 和 CUDA13.2 runtime/NVRTC，动态依赖闭包仍有限制，不据此断言 run 004 失败原因。

### Failed starts and observer/PID caveats retained

Baseline-delta runs 003 and 004 仍是 append-only failed startup evidence：003 在 lazy CUDA initialization 期间因 observer `nvidia-smi` timeout 中止；004 的 worker 在 NCCL 前报告 `CUDA driver initialization failed, you might not have a CUDA gpu.`；两者均未发 Hello2/canary/QA 请求。它们不被改写为 canary failures，也不从失败分母删除。相关汇总见 [`RESULTS-baseline-delta.md`](experiments/tp2_resident/RESULTS-baseline-delta.md) 及其 [`run 003`](experiments/runs/tp2-resident-20261008-003/execution.json)、[`run 004`](experiments/runs/tp2-resident-20261008-004/execution.json) receipts。

更早 [`RESULTS.md`](experiments/tp2_resident/RESULTS.md) 中 runs 001/002 的 Hello2 成功和 canary startup abort 仍保留：001 是 PSS ESRCH，002 是 nonselected GPUs 上来源 unknown 的 1 MiB 观察；均无 canary HTTP，不能说是数值 FAIL 或 PASS。[historical comparison](experiments/runs/tp2-resident-single-lifecycle-20261009-001/comparison.json) 保留 004 driver failure 根因 unknown；本次成功不追溯消除失败，也不改 M7/G10 或任何历史 target。

该历史 run：`owned_gpu_attribution=unknown`；server PID 14168/worker 15060/15061 仅本地身份，非 host-NVML 映射。`/proc/15844/smaps_rollup` 消失的 leaf 为 incomplete/PSS null；31 complete host + 1 incomplete host、32 GPU samples、0 missing NVML，非连续覆盖。nonselected deltas 全 0，未用 1 MiB tolerance；19.284 GiB/card >19.2 planning、<22 observational ceiling。128 GiB PSS 非物理 cap，不含 unmapped page cache/不证明 pinned bytes。[drain/ECHILD](experiments/runs/tp2-resident-single-lifecycle-20261009-001/execution.json)、[八卡 idle](experiments/runs/tp2-resident-single-lifecycle-20261009-001/final-owner-recheck.json)、[lock release](experiments/runs/tp2-resident-single-lifecycle-20261009-001/final-owner-release.json)为历史局部 cleanup，非本次查询/physical fence，限制不消除。

该历史配对路线仅恢复性能主张时执行：先绑定同配置的 quality/lifecycle 与 serving source/binary，再冻结匹配合同/fresh paired blocks、完整失败分母和 block statistics；当前下一步仍是 RQ1–RQ4，不先做 vLLM 胜出。

## 待补写，不应通过推测补齐

- **四 RQ 核心 partial/open：**同cut factory numerics9/9 closed；capacity frontier/physical attribution、分层I/O、CLI HTTP、两arm performance/overlap配对、集成fault/custody仍open。更强物理 insufficiency/necessity 未确认且非本 scoped paper 前提。
- **Artifact：**source/dirty snapshot、binary/model/tokenizer/client/oracle、raw 与失败全分母冻结；covered/untracked/fixed memory 分明。九gate qualification_v1 receipts已portable归档/CPU检查；其他历史target/external payload恢复与干净环境复现仍open，不跨profile借PASS。
- **范围与代价：**可执行支持表按模态/backend/设备/拓扑/表示建立；真实 overlap、正常/观测成本及安全消融待验收，扩展仅升主张时恢复。
- **Related work 局部：**FlexGen/ZeRO-Inference/PowerInfer-2 为 storage/streaming prior art，PowerInfer 为 GPU–CPU hybrid，vLLM 为 serving KV（[正文](sections/related-work.tex)）；非首个 storage inference、非未审计能力缺失。metadata/BIB/guarantee audit open，SGLang 未核查。
- **投稿：**ORCID/资助/致谢/利益冲突/许可需确认，最终 claim/视觉审读未验收；四 -02 raw-to-plot 图和当前构建形态见下，非科学门禁完成。

## 构建与环境记录：区分上版和新版

**9-gate资格更新后latest manuscript build/review（2026-10-15 recovery）：**`make -C papers/arxiv` PASS（120s限时，无build timeout），**36页/757784 bytes**，SHA256 `d0aa956a5793b344fd4ab18452e1a66afaa3696c12b6860ebb3a9c19dfb06989`。
版本snapshot：recovered factory9/stopped CLI V2；factory033519c/archive17b16fa/source-plan query fix86c58a0，newCLI/HTTP pending；source/PDF hashes与scope见`build/recovery-final-review.json`。
35 main/reference pages+1 historical appendix；converged warnings/undefined refs/citations/
overfull/underfull/oversized floats/duplicate labels均0。Ghostscript完整36页color110dpi/gray72dpi；color contact-sheet全页审读，Algorithm3 p15/current Table6 p22/AppendixA p36 full-page核对，图页8/10/19/25–28另查gray。
无observed clipping/obscured text/occlusion/incomplete algorithm；appendix longreceiptpath已显式quote/换行，保持36页不扩写。
9个typedreport hashes与produceraudit/FINAL hash匹配；ordinary各64和cold各28指纹按own-schedule
primary直接比对PASS，ordinaryfalse/null fields与per-stream分母核对通过。该只读核对不执行GPU。
29 protected figure/Makefile/bibliography与340 snapshot archive/paired-receipt hashes unchanged；narrow whitespace/compiled source-label checks PASS；Git/index/diff核对由main负责，本recovery无Git命令。
exact12 source/doc paths+PDF一次见[README handoff list](README.md#exact-handoff-file-list-no-commit-made)；
newappendix source目前受ignore规则，main须explicit include，本agent未stage/commit。
curated `experiments/storage_demand/qualification_v1/` 已committed17b16fa；本recovery只读checker/37tests PASS，完整249files/5.38MiB保持未改，outputs仅`build/recovery-final-archive-*`（tests临时files也限build）。
这仅关闭receipt portable子项，不含external payload恢复或CLI serving/performance资格。


**此前未纳入9-gate更新的构建（2026-10-15，历史）：**`make -C papers/arxiv` PASS（每次120 s限时，无build timeout），latest PDF **34页/748066 bytes**，SHA256 `ceb14b2a31d6cf2ef242acb4a316626b602d3d7d152ff577c53149d142e01390`。
Converged log 的 LaTeX/package warnings、undefined references/citations、overfull/underfull、
oversized floats、duplicate labels 均0。3 standard algorithms+7既有图；完整34页Ghostscript
110dpi/contact-sheet审读，Algorithm3另page-width核对，无observed clipping/occlusion/
incomplete algorithm。`git diff --check`及source label/旧unimplemented-claim审计PASS；
29 protected figure/Makefile/bibliography assets hash unchanged。Exact11 source/doc paths+
tracked generated PDF见[README handoff list](README.md#exact-handoff-file-list-no-commit-made)。
未执行Rust/GPU tests、plot/experimentcheckers、server/GPU queries或commit；本PASS仅稿件，
不闭合quality/capacity/RQ3 overlap/fault-custody gates。


**历史稿件形态（2026-10-14）：**cohesive paper 以 RQ1–RQ4/合同/related work/限制组织；7 彩图（3 mechanism、4 -02 data）、标准 `algorithm`/`algorithmic` 与数学/数据图已接入。[现有构建记录](README.md#current-plot-integration-validation)/历史日志记录：`make -C papers/arxiv` PASS，30 页/725591 bytes，warnings/undefined references/citations/overfull/underfull/oversized floats/duplicate labels 均无。局部审读已有记录，非本次构建/审读或科学 PASS。**上版历史：**make 退出 0、2 轮、9 页/183734 bytes、最终无残余警告，不移作当前验收。

**历史源码构建验证（2026-10-02，非本次执行）：** `make -C papers/arxiv` 退出 **0**，初次为 up-to-date；`make -B -C papers/arxiv` 仍未触发 pdfLaTeX。随后在 `papers/arxiv/` 执行 `latexmk -g -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex`，退出 **0**，实际运行 **1 轮 pdfLaTeX**；再次 `make -C papers/arxiv` 退出 **0**。所有构建调用均设 120000 ms 限时且未超时。最终 `build/main.log` 报告 **15 页、299612 bytes**，PDF 文件大小核对一致；无 LaTeX/Package warning、overfull、未定义引用或重编译请求，但仍有 **2 条 underfull hbox**（`sections/evaluation.tex:203`，badness 10000 和 2261）。页数依据本次编译日志；未进行独立 PDF 解析或人工视觉审读，标题、作者、公式、表格与引用的视觉检查仍待完成。本次未改 TeX，未安装包，未运行 GPU 或数值实验；编译成功不构成容量、模态、异步或质量验收。

**历史 Python 环境记录：**`/opt/conda310/bin/python`，3.10.9；不冒充当前环境或本次版本验证。V2 的 literal CPU argv 见其冻结 README，旧 absolute pins 不改写。

## 文档修改的验证含义

**2026-10-08 历史：**仅 README/ledger/checklist 同步权威收据/计划，CPU 文本核对；无 TeX/引用/实现/历史 raw/experiments 修改，无 GPU/数值/构建。10-02 构建未重跑，E0/比较/统计未验收。

**2026-10-09 历史：**三份 Markdown 同步 TP2 receipts，仅链接/指标复算/差异检查；无实验/脚本/docs/TeX/target 修改或 GPU/数值/构建，E2 仅局部 baseline，E0/fairness/parity open。**2026-10-14 历史：**只改 checklist/ledger，窄 Markdown 检查；无新 GPU serving/数值/physical-I/O measurements，无构建或其他文件修改。

**2026-10-15 本轮：**只改main/sections/README/checklist/ledger及normal build/review outputs；exact handoff scope、latest PDF与warnings见上。未改figure sources/experiments/raw/production，未执行GPU；随后实读9/9新producer PASS并限定factory范围，未预判future paired/NFS/overlap、未commit。
