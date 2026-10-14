# Synchronous expert completion wait validation

## Scope and invariants

`parallel/data.rs` now notifies the existing `ferrule_common::CompletionHub`
(epoch + AtomicWaker) after publishing to the original completion channel and
from the owner-exit guard. There is no second completion queue, transaction
allocator, authority, retry mechanism, or production metrics stream.

`parallel/expert.rs` snapshots a listener before checking the completion channel
or join predicate. Empty collection, shutdown and Drop use a synchronous Condvar
adapter. The listener checks the epoch before and after registering its waker;
the adapter's protected predicate covers notification before entering the wait.
Listener polling never holds the Condvar mutex. No business state or user callback
runs under that mutex; poison is recovered on both consumer and producer paths.

Every wait is bounded by `min(absolute_deadline, now + 1ms)`. The 1ms fallback
checks callback-only cancellation and thread-finished visibility after the exit
guard runs. Wake, timeout, spurious wake and `poll() == None` are never completion
or join evidence. `try_recv` remains the host credit release point; expert tickets
remain in custody through final callback/deadline checks and permanently on
Unknown. No public API, replay, failure classification, or callback ordering was
changed. Cancellation observation cadence changes from nominal 50us polling to
at most a 1ms wait slice, excluding scheduling and callback execution time.

Process EP uses the same DP path: child pipe readiness -> existing reader and
validation -> DP result enqueue -> hub notification -> expert receive/validation.
`ProcessRankOwner` still uses `poll(2)` for pipe readiness. TERM grace, bounded reap
polling, and the background reaper were not changed or replaced with synthetic
completion evidence. The adapter is only for the existing synchronous host-owner
path, not an async runtime task or a general-purpose block_on.

## Deterministic coverage

- `data_completion_wait.rs`: empty -> registered wake; publication before
  registration and burst coalescing; expired deadline and spurious signal;
  cancellation without completion; Unknown/no replay; owner exit; poisoned lock.
- `expert_completion_wait.rs`: gated empty wait, cancellation recheck/drain,
  absolute deadline before consuming an already queued reply, Unknown, and final
  callback cancellation/deadline with retained ticket custody.
- Existing `batch_late_queued_completion_and_final_callback_keep_ep_tickets`
  verifies late queued completions and final callback order.

New tests use channel gates, listener registration and fixed dispatch phases,
not sleeps. Notification counts use the existing hub epoch; wait-call counts are
local to tests. The controlled cases report:

| Case | Publications | Consumer wait calls | Fixed 50us sleeps |
| --- | --- | --- | --- |
| Empty -> registered wake | 1 completion, then 1 owner exit | 1 | 0 |
| Burst before registration | 8 outcomes + 1 owner exit | 1 | 0 |
| Expert success/deadline/Unknown/final callback | gated outcome | 1 each | 0 |
| Expert cancellation while empty | gated outcome after cancellation | 2 | 0 |

Publications may coalesce; these are not counts of kernel context switches.
30 repeated runs of the eight new tests plus the existing final-callback test
passed (270 test executions). A final run after the test-only lint fix also
passed all eight new tests.

## CPU checks (2026-10-01)

All commands use `-p ferrule-runtime --no-default-features --features cpu-reference`:

- `cargo test --lib parallel::expert`: 22 passed.
- `cargo test --lib parallel::process`: 16 passed.
- `cargo test --test data_parallel`: 14 passed.
- `cargo test --test process_decoder`: 14 passed.
- `cargo test --test process_ranks`: 21 passed.
- `cargo test --test process_ipc_measurement`: 2 passed, 3 opt-in measurements ignored.
- `cargo check --all-targets`: passed.
- `cargo clippy --lib -- -D warnings`: passed.
- Strict all-target clippy initially flagged the existing
  `src/error.rs:278` `items_after_test_module` and one new test's collapsible if.
  The new lint was fixed; all-target clippy passed with
  `-- -D warnings -A clippy::items-after-test-module`. No unrelated source was changed.
- `cargo fmt --all -- --check` and `git diff --check`: passed.

## CUDA 12.4 compile only

NVCC: `/home/admin/wxw403883/anaconda3/envs/tao/bin/nvcc`, version 12.4.99.
Used the existing `target/validation/qwen35-35b-current/toolchain/env-cuda124.sh`
with these overrides:

```sh
export CUDA_VISIBLE_DEVICES=
export CARGO_TARGET_DIR=/home/admin/luhuanbing.lhb/Ferrule/target/validation/completion-wait-cuda124
cargo build --locked --release -p ferrule-cli --features cuda
cargo test --locked --release -p ferrule-runtime --features cuda --lib --no-run
```

Both passed (CLI 4m09s; runtime test compilation 9m17s). Architecture was explicitly
`sm_86`; no GPU detection, CUDA test execution, inference or server was started.
The isolated target directory leaves the baseline binary intact. `readelf -d`
confirms the new CLI needs `libcudart.so.12` and `libcublasLt.so.12`, not version 13.

CLI artifact: `target/validation/completion-wait-cuda124/release/ferrule`

SHA-256: `aca715ccc5fee4197a5516689d7e0bc8c1f5a9c0894674284d3cb31686f06c26`

## Performance boundary / GPU-owner handoff

User-provided steady-state baseline: process EP8, one request, 32 tokens,
wall 13.482s; main `clock_nanosleep` 8.000s / 77,255 calls (59.34%);
thread CPU 1.912s / 28% as reported. This profile identifies sleeping/waiting,
not all IPC cost and not recoverable wall time.

This patch removes all three fixed 50us sleep sites in `expert.rs`. Their new
runtime sleep count is zero by construction; it does NOT establish that all
77,255 baseline calls originated at those sites. No post-change EP8 syscall count,
CPU-time measurement or wall-time speedup has been claimed.

The nominal no-progress timer cadence falls from 20,000/s (50us) to about 1,000/s
(1ms), before scheduler effects. Real completions wake immediately rather than
waiting for that fallback timer; bursts can share a wake. Expected benefits are
less repeated scanning/timer scheduling and less completion-discovery delay.
Notification/mutex/futex costs and actual child work remain. The 8.000s cannot be
subtracted from 13.482s, nor inferred as a CPU saving.

GPU owner should repeat the identical warmed EP8/32-token workload against the
isolated binary, retaining the baseline workload and sampling method. Compare
output correctness, wall/CPU time, `clock_nanosleep` and futex waits/wakes,
context switches, and cancellation/deadline behavior. Only that retest can
establish net end-to-end benefit.
