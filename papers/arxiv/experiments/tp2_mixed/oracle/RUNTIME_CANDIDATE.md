# Runtime trace → paper comparator: CPU recovery handoff

**Superseded for v2 implementation:** see [BRIDGE.md](BRIDGE.md) for the implemented
raw conversion and numerical lifecycle comparator. The capability gaps below
describe the earlier recovery checkpoint, not the current v2 stream.

## Immutable baseline

Reference: `runs/full40-20261009-03`, manifest SHA256
`fb96ba4dd134761af30b4025b4cce86f96136cb01f05b5b4fca3e74587caa71d`.
Unchanged `protocol.json` byte SHA256:
`c6e7836f3797c16502dc21b8c756c87495525c2cf2faac26ba11077f62cfbb57`.

The old manifest and its sealed workflow pins remain unchanged and mutually
consistent. Current `launch.py`, `tp2_oracle.py`, and `test_compare.py` differ
from their historical source pins because of the new CPU-tested lifecycle work.
This is source evolution, not a reason to rewrite historical receipts. The
historical loader verifies sealed artifact identity; it does not assert that
current generator code is byte-identical to the historical generator.

## Two formats, not interchangeable

Runtime source contract:
`crates/ferrule-composition/tests/tp2_mixed_execution/CANDIDATE_CONTRACT.md`.

- Raw: `candidate/contract.json` (`ferrule.mixed-candidate.v1`), per-rank
  `events.jsonl`, `tensors.f32le`, `capture-status.json`, and parent publication
  receipts. `complete_available_trace` is not `required_inventory_complete`.
- Comparator input: `manifest.json` (`ferrule.paper.tp2-mixed-candidate.v1`),
  `profile=SystemMixedTP2EP1`, `status=complete`, the **unchanged reference
  protocol**, matching checkpoint `config_sha256`, `index_sha256`,
  `full_shard_set_sha256`, `effective_text_config`, and `cases` ordered
  `hello,capital`.
- Each case: `{id,file,sha256,report:{file,sha256}}`. Main file is safetensors;
  report contains `id`, `calls`, `tensors` (exact name/shape/dtype inventory).
  Each call has `stage`, `input_token_ids`, `position_start`,
  `sequence_length_after`, `next_token_id`.
- Exactly **2,178 candidate tensors/case**, from
  `compare.candidate_inventory(config, ids)`. Only pre-tie HF
  `native_expert_ids` are excluded. Canonical selected IDs are mandatory.
- Current comparator does **not** parse the runtime raw format. A paper-local
  conversion/join layer is still needed once authoritative fields exist. It
  must only reshape/reindex/pack observations, never compute missing expected
  or actual numerical boundaries. No runtime execution loop belongs here.

## Join and capture timing

Use successful parent `{case}.{stage}.receipt.json` with `published=true` and
its exact transaction to join both owners' `begin` events. Bind owner epoch,
ordered rank group, session, phase, token/position/query range and state
execution/page generation. Never join by arrival order or owner-local ordinal
alone. `execution_end.success` is not commit (`committed=false` is intentional).
Reject absent/failed publication, incomplete status, duplicates, missing ranks,
truncated ranges, overlap, stale epoch/frontier or unknown event geometry.

| Paper boundary | Required actual observation/time |
|---|---|
| inputs/positions/mask | Validated executor call inputs; mask must be bound to no-padding, batch-one semantics |
| `embedding` | BF16 embedding output before layer 0 |
| `rank.R.layers.L.attention_partial` | BF16 local O/out projection writeback **before** TP sum |
| `layers.L.attention` | Post-TP attention output, **before** residual addition |
| `shared_partial` | BF16 local shared MLP down output **before output-gate sigmoid multiplication** |
| `shared_gate_logit` | Raw replicated output-gate linear logit, before sigmoid |
| `routed_partial` | Local ascending-expert-ID BF16 accumulation, before shared addition |
| `moe_partial` | Local routed + gated shared BF16 sum, before its one TP reduction |
| `layers.L.moe` | Post-TP MoE output, before residual addition |
| `layers.L.hidden` | Complete layer output, after residual addition |
| router arrays | Raw BF16 logits, actual full F32 probabilities before selected renormalization, canonical int64 IDs, actual renormalized F32 weights consumed by experts |
| caches/state | Full post-append/post-update logical snapshot for this invocation, usable only after matching successful parent publication |
| `final_norm`, local/full `logits` | BF16 final norm; local BF16 vocab output and rank-ordered gather before argmax |

F32 diagnostic expansions of BF16 may be packed as BF16 **only after** typed
original-dtype provenance and exact finite F32→BF16→F32 round-trip validation.
Do not round arbitrary F32 candidates into passing BF16. Keep SSM, probabilities
and routing weights F32. IDs come from integer metadata; preserve authoritative
`weight_bits`, never pass IDs through BF16 or reconstruct weights via softmax.

Canonical full cache construction may concatenate actual local snapshots:
KV along head dimension; SSM along value-head dimension; conv by Q/K/V segment
(first split each rank's `[Qlocal,Klocal,Vlocal]`, concatenate like segments,
then rejoin Q/K/V). Use authoritative geometry/ranges, not inferred flat lengths.
Do not construct retained KV from newly projected K/V, infer a committed state
from a failed append, or derive missing global outputs from oracle values.

## Blockers acknowledged by runtime contract

1. Actual **full F32 router probabilities**, not recomputed softmax.
2. Full retained+appended **KV snapshot** with logical sequence/position geometry.
3. Original dtype, N-D shape/strides, sequence index, local shard/head/cache ranges.
4. Selection before readback/BF16 expansion and exact maximum device trace scratch
   for admission. `gdn.chunk.workspace` is not a semantic required boundary.
5. CPU-tested raw-trace join/packing bridge with fail-closed incomplete status,
   followed by independent runtime export. Current logits-only receipt is not
   promoted to a complete candidate.

The numerical comparator checks arrays, manifest/artifact hashes and reference
custody; candidate capture timing, execution/authentication, scratch admission,
and lifecycle commit provenance require the joined runtime receipts as well.
No numerical PASS alone certifies these claims.

## New lifecycle entry: what exists and what does not

`launch.py --scenario {greedy,canary,fork,recreate,cold-owner} --decode-steps N`
uses the existing authenticated worker and `tp2_oracle.export`. The recipe
calls the shared `run_call`/Capture/serializer; it does not add another numerical
engine. Limits: 1–8 decode steps, at most 20 calls, three live snapshots, 4 GiB
total capture, original per-call capture/deadline limits. Every call records
resolved inputs, full inventory, state digests and session/epoch association.
New suite schema: `ferrule.paper.tp2-mixed-lifecycle-reference.v1`; it does not
replace the immutable two-stage reference schema.

- Greedy: native full-vocab argmax continuation.
- Canary: interleave a distinct session and ensure main state/outputs unchanged.
- Fork: disjoint HF cache clones; branches and parent checked for non-mutation.
- Recreate: drop sessions/model and clear decoded cache, rebuild in the same child.
- Cold-owner: repeat separate authenticated launches for distinct process identity;
  an in-process rebuild does not certify cold-owner behavior.

**Missing:** lifecycle-suite comparator and raw-runtime bridge; matching the
runtime fixed-token/cancel recipe to a separately pinned oracle plan; testing
actual CUDA lifecycle exports. HF snapshots do not prove production COW or
cancel/quarantine semantics. Current `compare.py` accepts only the old two-stage
schema and must reject lifecycle suites until their comparator is implemented.

CPU planning (no model reads, no GPU queries):

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none \
 /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
 papers/arxiv/experiments/tp2_mixed/oracle/launch.py --scenario greedy --decode-steps 4
```

Future GPU command template — **not authorized/executed in this phase**:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
 papers/arxiv/experiments/tp2_mixed/oracle/launch.py \
 --scenario greedy --decode-steps 4 --timeout 1700 \
 --execute-cuda --sole-gpu-owner-granted \
 --gpu-uuid GRANTED_UUID --owner-note 'Fresh main-agent grant' \
 --output papers/arxiv/experiments/tp2_mixed/oracle/runs/FRESH_NAME
```

Do not use this oracle command as a runtime GPU launcher; runtime's exact ignored
selectors and launcher allowlist remain the runtime/launch owner's responsibility.
