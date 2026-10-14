# Bounded actual Hello producer diagnosis

The composition test `model_factory::dense_bf16_tests::full40::numeric_bf16_mixed_hello_prefill_producer_capture`
materializes the original full40 image and runs only original Hello prefill.
It is diagnostic, not a relaxed acceptance gate. The existing full40 test and
Capital capture entry retain their original cases and numerical assertions.

Observer options:

- `FERRULE_NUMERIC_BF16_CAPTURE_PHASES=hello/prefill`
- `FERRULE_NUMERIC_BF16_CAPTURE_LAYERS=0:12` (exclusive end)
- `FERRULE_NUMERIC_BF16_FIRST_TRACE_DIR=<fresh directory>`

The layer default remains `0:2`. Ranges require `0 <= start < end <= 40`, at
most12 layers, and an explicit trace directory. The existing64MiB observer
payload cap remains. GQA layers have no recurrent state and are skipped for
before-state snapshots, not skipped by the event observer.

`hello_cause.py <evidence-directory>` is a separate pinned-HF process. It reuses
the original layers2..4 supplemental capture, observes native layers5..11 from
original predecessor hidden rows, and requires all35 original attention/MoE/
hidden/state boundaries to be bit-exact before retaining its capture. It also
replays differing projections on actual producer inputs. Those diagnostic
replays are never substituted into original expected values.

Production is CUDA12.4/sm86 with launcher `CUBLAS_WORKSPACE_CONFIG=:0:0`.
Independent HF remains the original pinned Torch2.13.0+cu129/Transformers5.5.3,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, offline, TF32/reduced BF16 reduction disabled,
F32 SSM, BF16 conv, original BF16 matrix words and numeric FP8/BF16 scales.
GDN A_log/norm F32 payloads are cast once to BF16 by the maintained loader;
RoPE buffers are not blanket-cast.

Never invoke GPU tests directly: obtain ownership, use fresh directories,
pin source/binary/reference/fixture hashes, and use the authenticated bounded
supervisor. Exact commands and completed receipts for this diagnosis are in
`target/validation/hello-producer-cause-20261007/PREPARED.json`.

## Result and backend handoff

Original layer outputs are exact through8. Layer9 first changes attention
(3/2048 words) and final hidden (1/2048), despite exact projections, conv and
524288-word F32 SSM. Actual recurrent output matches native HF chronological
recurrence0/4096, but differs from native HF **prefill chunk arithmetic**2/4096.
HF chunk output and state match the original prefill reference exactly.
Gated norm and out-projection match on actual inputs; QK norm/RoPE are not the
identified cause. Changing the multirow model selector to admit M1 is not a fix:
the current chunk backend explicitly delegates M1 to chronological recurrence.

Earlier local differences are separately recorded: F32 routing weights from
layer0 (without changed selected IDs), raw router layer3 (one unselected word),
and in_proj_a layer4 (one word; no state/hidden difference). These are not claimed
to be the cause of the layer9 propagated difference.

The actual layer9 fixture, independent expected words, zero initial state,
parameters, geometry and hashes are frozen in the evidence directory. Backend
coordination must preserve cached-decode recurrence while establishing an
explicit prefill-chunk contract for M1, with honest scratch/launch budgeting.
No recurrent or projection production arithmetic was changed by this diagnosis.
