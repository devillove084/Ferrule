# Device participation control handoff

CPU control is verified; production device participation remains disabled.
`MixedRankCollective::prepare` always sets its private capability to false.
There is no public enable API. Native provider quality, budget and unknown-bound
admission are not established by these CPU tests.

## Actual caller flow

1. Query `HybridBf16Collective::data_location(&site)`. Host uses the unchanged
   `exchange_bf16` path, including BF16 finite checks; head gather stays host.
2. `permit_device_payload(identity)` authenticates the prepared plan, epoch,
   transaction, ordered members, packed dimensions and complete canonical site.
   Both owners must vote identically before either receives permission. Only
   layer AttentionOutput/RoutedExpertOutput ReduceSum sites are eligible.
3. Recheck `check_device_payload_permit(&permit)` immediately before enqueue.
4. Only successful enqueue calls `device_payload_enqueued(permit)`. This consumes
   the permission and advances the local site cursor; metadata Ready/drain does
   neither. A post-enqueue error is Unknown, never rollback or retirement.
5. After every scheduled site and the provider's true consumer fence plus finite /
   asynchronous-error checks, each owner reports
   `complete_device_payloads(transaction, completion: Result<()>)`.
6. Each owner calls `complete_owner(transaction)`; the parent control then calls
   `finish(transaction)`. Missing device completion blocks both gates. These are
   control gates, not journal commits or publication receipts.

The four unbound `Bf16Channel` device-adapter methods now live privately in
`mixed_channel_device_tests.rs` under `cfg(test)`. The current production leaf
continues using Host exchange. The production control protocol remains on the
existing public `HybridBf16Collective` hooks and prepared runtime implementation;
this correction adds no public API, permit constructor or provider factory.

Six new model CPU tests drive real mock-trait counters and joined CPU consumers.
They verify permit/recheck/enqueue acknowledgement order, invocation-wide fence
and finite-result reporting, rejection of fake completion and unchanged Host
behavior. Local channel finish does not emit an owner receipt or publication.

The sole endpoint `exchange_inner` loop owns synchronization. Identity is stored
in typed metadata slots; internal zero-sized tokens allocate no data buffers.
Shared abort retains issued permissions, completion obligations, execution and
watermarks. Peer loss dominates Ready/drained results. Resource retirement still
requires the existing worker joint-abort/consumer-proof custody protocol.

CPU coverage: 14 new device control/fault tests, existing typed-collective and
TP2 mixed contracts, and model hybrid/channel regressions. No GPU validation.

## Source/visibility correction: CPU relation only

Strict model `cuda --lib --tests` compilation with `-D warnings` passed using
verified NVCC 13.2 and the documented static `sm_86` target. No device/architecture
query, GPU execution or GPU lock was used. Source inputs changed in this
correction; prior exact-source/native admission receipts must not be inherited.
The builder must evaluate fresh source pins. Production device enable remains
off; this supplies no native quality, budget or unknown-bound admission.
