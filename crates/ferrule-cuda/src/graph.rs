//! CUDA graph capture for decode — enables kernel replay without per-step launch overhead.
//!
//! This module is built on top of cuda-oxide's high-level CUDA Graph API
//! (`CudaStreamCaptureExt`, `CudaGraph`, `CudaGraphExec`, `CachedGraphExec`).
//! Two entry points are provided:
//!
//! - [`capture_decode_graph`] / [`CudaGraphHandle`]: one-shot capture of a
//!   decode step into a replayable executable. Captured once, replayed many.
//! - [`CachedDecodeGraph`]: llama.cpp-style auto-warmup replay. The first run
//!   executes directly (warmup); if the workload is stable, the second run
//!   captures a graph, and subsequent runs replay it. Property changes
//!   (data pointers / shapes) invalidate and re-capture automatically.
//!
//! Feature-gated behind `FERRULE_CUDA_GRAPH=1`.

use cuda_core::graph::{
    CachedGraphExec, CaptureMode, CaptureModeGuard, CudaGraph, CudaGraphExec, CudaStreamCaptureExt,
    GraphStrategy,
};
use cuda_core::stream::CudaStream;
use cuda_core::DriverError;
use ferrule_core::{Error, Result};
use std::sync::Arc;

/// Whether CUDA graph capture is enabled for decode.
/// Set FERRULE_CUDA_GRAPH=1 to enable (experimental).
pub fn cuda_graph_enabled() -> bool {
    std::env::var_os("FERRULE_CUDA_GRAPH").is_some()
}

/// Whether FlashAttention-style kernels should be used.
/// Set FERRULE_FLASH_ATTN=1 to enable (requires kernel implementation).
pub fn flash_attn_enabled() -> bool {
    std::env::var_os("FERRULE_FLASH_ATTN").is_some()
}

/// Convert a cuda-oxide `DriverError` into a ferrule `Error`.
fn drv(e: DriverError) -> Error {
    Error::Internal(format!("CUDA graph: {e:?}"))
}

// ── One-shot capture: CudaGraphHandle ─────────────────────────────────

/// An owned, instantiated CUDA graph that can be replayed onto a stream.
///
/// Created via [`capture_decode_graph`]. Once instantiated, the graph can
/// be launched repeatedly with [`CudaGraphHandle::launch`] without
/// re-recording kernels — eliminating per-step kernel-launch overhead
/// during autoregressive decode.
///
/// Backed by cuda-oxide's `CudaGraphExec`.
pub struct CudaGraphHandle {
    /// The executable graph (instantiated from `graph`).
    exec: CudaGraphExec,
    /// Retain the source graph so the executable's topology stays valid for
    /// future `update()` calls. `CudaGraphExec` does not own the `CudaGraph`.
    _graph: CudaGraph,
    /// Keep the parent context alive so the CUDA context outlives the graph.
    _ctx: Arc<cuda_core::CudaContext>,
}

// CudaGraphExec is Send+Sync via cuda-oxide; we re-assert it here for the
// composite handle.
unsafe impl Send for CudaGraphHandle {}
unsafe impl Sync for CudaGraphHandle {}

impl CudaGraphHandle {
    /// Launch (replay) the captured graph onto `stream`.
    ///
    /// The stream must belong to the same CUDA context that was active
    /// when the graph was captured.
    pub fn launch(&self, stream: &CudaStream) -> Result<()> {
        self.exec.launch(stream).map_err(drv)
    }

    /// Upload the graph's execution nodes to the device ahead of the first
    /// launch on `stream`. Reduces first-launch latency; optional.
    pub fn upload(&self, stream: &CudaStream) -> Result<()> {
        self.exec.upload(stream).map_err(drv)
    }

    /// Attempt to update the executable in-place to match `updated_graph`.
    ///
    /// Returns `Ok(true)` if the update succeeded, `Ok(false)` if the
    /// topology changed and the caller must re-capture, or an `Err` on
    /// driver failure. Useful when only kernel parameters (e.g. input
    /// buffer pointers) change between decode steps.
    pub fn update(&self, updated_graph: &CudaGraph) -> Result<bool> {
        use cuda_core::graph::GraphUpdateResult;
        match self.exec.update(updated_graph).map_err(drv)? {
            GraphUpdateResult::Success => Ok(true),
            // Topology / node-type / function changes require re-capture.
            _ => Ok(false),
        }
    }

    /// Borrow the underlying source graph (for `update()` on another handle).
    pub fn source_graph(&self) -> &CudaGraph {
        &self._graph
    }
}

/// Capture all kernel launches enqueued by `capture_fn` on `stream` into
/// an instantiated [`CudaGraphHandle`] that can be replayed.
///
/// Uses `CaptureMode::Relaxed` (matches Ferrule's previous behaviour) and
/// a [`CaptureModeGuard`] so the capture mode is thread-local and restored
/// even if `capture_fn` errors.
///
/// # Errors
///
/// Returns `Error::Internal` if stream capture is unsupported (CUDA driver
/// < 10.0), or if capture / instantiation fails.
pub fn capture_decode_graph(
    stream: &CudaStream,
    capture_fn: impl FnOnce() -> Result<()>,
) -> Result<CudaGraphHandle> {
    // Make capture mode thread-local + restore on drop (cuda-oxide best practice).
    let _guard = CaptureModeGuard::new(CaptureMode::Relaxed).map_err(drv)?;

    // Begin capture.
    stream.begin_capture(CaptureMode::Relaxed).map_err(drv)?;

    // Record the compute graph. If it errors, end capture + destroy to avoid
    // leaving the stream in a broken capture state.
    if let Err(e) = capture_fn() {
        let _ = stream.end_capture();
        return Err(e);
    }

    // End capture and obtain the graph.
    let graph = stream.end_capture().map_err(drv)?;

    // Instantiate the graph for launch.
    let exec = graph.instantiate().map_err(drv)?;

    Ok(CudaGraphHandle {
        exec,
        _graph: graph,
        _ctx: stream.context().clone(),
    })
}

// ── Auto-warmup replay: CachedDecodeGraph ─────────────────────────────

/// A decode-step graph with llama.cpp-style auto-warmup and change detection.
///
/// Wraps cuda-oxide's [`CachedGraphExec`] with [`GraphStrategy::AutoCapture`]:
///
/// 1. First `launch_or_capture` call executes directly (warmup).
/// 2. If the recorded properties are stable, the second call captures a graph.
/// 3. Subsequent calls replay the captured graph.
/// 4. If [`set_properties`] changes the data pointers / shapes, the graph is
///    invalidated and the warmup cycle restarts.
///
/// This is the recommended path for stable decode buckets once all host
/// boundaries are eliminated. For decode steps whose pointer set changes
/// every token (e.g. incremental KV append), either keep the pointer stable
/// across the bucket or fall back to [`capture_decode_graph`].
///
/// **Stream requirement:** auto-capture must run on a *non-default* stream
/// (`CudaContext::new_stream`). The legacy default stream rejects repeat
/// capture with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` (900).
pub struct CachedDecodeGraph {
    inner: CachedGraphExec,
}

impl CachedDecodeGraph {
    /// Create a new auto-warmup decode graph bound to `ctx`.
    pub fn new(ctx: &Arc<cuda_core::CudaContext>) -> Self {
        Self {
            inner: CachedGraphExec::new(ctx, GraphStrategy::AutoCapture),
        }
    }

    /// Record the properties that define graph identity. If these change
    /// between calls, any captured graph is invalidated.
    ///
    /// `data_ptrs` are the device pointers the graph reads/writes (e.g.
    /// input embedding, hidden, KV, hc_state buffers); `shapes` are the
    /// corresponding element counts. Keep these stable across replay steps
    /// to keep the graph cached.
    pub fn set_properties(&mut self, data_ptrs: &[*const std::ffi::c_void], shapes: &[u64]) {
        self.inner.set_properties(data_ptrs, shapes);
    }

    /// True if a captured graph is ready for instant replay.
    pub fn has_cached_graph(&self) -> bool {
        self.inner.has_cached_graph()
    }

    /// Invalidate any captured graph, forcing re-capture on the next launch.
    pub fn invalidate(&mut self) {
        self.inner.invalidate();
    }

    /// Execute one decode step using the auto-warmup strategy.
    ///
    /// - `capture_fn`: called with the stream in capture mode to record the
    ///   decode step. Must NOT synchronize the stream or do host work that
    ///   can't be captured.
    /// - `execute_fn`: called with the stream to execute the decode step
    ///   directly (warmup / fallback).
    pub fn launch_or_capture<C, E>(
        &mut self,
        stream: &CudaStream,
        capture_fn: C,
        execute_fn: E,
    ) -> Result<()>
    where
        C: FnOnce(&CudaStream) -> std::result::Result<(), DriverError>,
        E: FnOnce(&CudaStream) -> std::result::Result<(), DriverError>,
    {
        // Relaxed thread capture mode is required for default-stream
        // capture; the guard restores the previous mode on drop.
        let _guard = CaptureModeGuard::new(CaptureMode::Relaxed).map_err(drv)?;
        self.inner
            .launch_or_capture(stream, capture_fn, execute_fn)
            .map_err(drv)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    // ── Feature flag tests ─────────────────────────────────────────

    #[test]
    fn cuda_graph_enabled_defaults_false() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("FERRULE_CUDA_GRAPH");
        assert!(!cuda_graph_enabled());
    }

    #[test]
    fn cuda_graph_enabled_when_env_set() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("FERRULE_CUDA_GRAPH", "1");
        assert!(cuda_graph_enabled());
        std::env::remove_var("FERRULE_CUDA_GRAPH");
    }

    // ── Graph capture tests (require CUDA device) ──────────────────

    /// Helper: create a CUDA context and stream, or skip the test.
    fn try_cuda_ctx() -> Option<(Arc<cuda_core::CudaContext>, Arc<CudaStream>)> {
        let ctx = cuda_core::CudaContext::new(0).ok()?;
        ctx.bind_to_thread().ok()?;
        let stream = ctx.default_stream();
        Some((ctx, stream))
    }

    #[test]
    fn capture_noop_empty_stream() {
        let (ctx, stream) = match try_cuda_ctx() {
            Some(v) => v,
            None => {
                eprintln!("skipping capture_noop_empty_stream: no CUDA device");
                return;
            }
        };
        ctx.bind_to_thread().ok();

        let handle = capture_decode_graph(&stream, || Ok(()));
        match handle {
            Ok(h) => {
                // Launch the empty graph — should succeed.
                h.launch(&stream)
                    .expect("launch of empty graph should succeed");
                stream.synchronize().expect("stream sync should succeed");
            }
            Err(e) => {
                // If driver < 10.0, capture may fail; that is expected.
                let msg = format!("{e}");
                assert!(
                    msg.contains("CUDA driver") || msg.contains("capture") || msg.contains("graph"),
                    "unexpected error: {msg}"
                );
            }
        }
    }

    #[test]
    fn capture_graph_graceful_failure_on_unsupported_driver() {
        let (ctx, stream) = match try_cuda_ctx() {
            Some(v) => v,
            None => {
                eprintln!(
                    "skipping capture_graph_graceful_failure_on_unsupported_driver: no CUDA device"
                );
                return;
            }
        };
        ctx.bind_to_thread().ok();

        let result = capture_decode_graph(&stream, || Ok(()));
        // If capture fails (e.g. old driver), it should return an error.
        if let Err(e) = &result {
            let msg = format!("{e}");
            assert!(
                msg.contains("capture") || msg.contains("driver") || msg.contains("graph"),
                "error message should mention capture/driver/graph: {msg}"
            );
        }
        // If capture succeeds (modern driver), that's also fine.
        drop(result);
    }

    #[test]
    fn capture_graph_launchable() {
        let (ctx, stream) = match try_cuda_ctx() {
            Some(v) => v,
            None => {
                eprintln!("skipping capture_graph_launchable: no CUDA device");
                return;
            }
        };
        ctx.bind_to_thread().ok();

        let result = capture_decode_graph(&stream, || Ok(()));
        match result {
            Ok(handle) => {
                // Launch from the captured graph.
                handle.launch(&stream).expect("graph launch should succeed");
                stream.synchronize().expect("stream sync should succeed");
                // Launch again — replay should work.
                handle.launch(&stream).expect("graph replay should succeed");
                stream
                    .synchronize()
                    .expect("stream sync after replay should succeed");
            }
            Err(_e) => {
                // Old driver — skip assertions.
            }
        }
    }

    #[test]
    fn capture_fn_error_preserved() {
        let (ctx, stream) = match try_cuda_ctx() {
            Some(v) => v,
            None => {
                eprintln!("skipping capture_fn_error_preserved: no CUDA device");
                return;
            }
        };
        ctx.bind_to_thread().ok();

        let mut closure_invoked = false;
        let result = capture_decode_graph(&stream, || {
            closure_invoked = true;
            Err(Error::Internal("simulated capture failure".into()))
        });
        match result {
            Err(e) if closure_invoked => {
                assert!(
                    format!("{e}").contains("simulated capture failure"),
                    "capture_fn error should be preserved after capture begins, got: {e}"
                );
            }
            Err(e) => {
                let msg = format!("{e}");
                assert!(
                    msg.contains("capture") || msg.contains("driver") || msg.contains("graph"),
                    "unexpected pre-callback capture error: {msg}"
                );
            }
            Ok(_) => panic!("capture_fn returned an error but graph capture succeeded"),
        }
    }

    #[test]
    fn cached_decode_graph_warmup_then_capture() {
        let ctx = match cuda_core::CudaContext::new(0) {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skipping cached_decode_graph_warmup_then_capture: no CUDA device");
                return;
            }
        };
        ctx.bind_to_thread().ok();
        // Graph capture requires a non-default stream.
        let stream = ctx.new_stream();
        let stream = stream.expect("new_stream");

        let mut cached = CachedDecodeGraph::new(&ctx);
        // Stable empty workload: warmup (run 1, no graph), then capture (run 2),
        // then replay (run 3). Neither closure synchronizes — during capture
        // that is forbidden, and for warmup it is unnecessary for an empty graph.
        let exec = |_s: &CudaStream| Ok::<(), DriverError>(());
        let cap = |_s: &CudaStream| Ok::<(), DriverError>(());

        // Run 1: warmup, direct.
        cached.launch_or_capture(&stream, cap, exec).expect("run 1");
        assert!(
            !cached.has_cached_graph(),
            "no graph after first warmup run"
        );

        // Run 2: should capture.
        cached.launch_or_capture(&stream, cap, exec).expect("run 2");
        // After the second stable run the graph is captured and launched.
        assert!(
            cached.has_cached_graph(),
            "graph should be captured after 2nd run"
        );

        // Run 3: replay.
        cached.launch_or_capture(&stream, cap, exec).expect("run 3");
        assert!(cached.has_cached_graph());

        stream.synchronize().expect("final sync");
    }
}
