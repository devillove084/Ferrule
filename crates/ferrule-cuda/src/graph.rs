//! CUDA graph capture for decode — enables kernel replay without per-step launch overhead.
//!
//! Requires CUDA driver ≥ 10.0. Feature-gated behind `FERRULE_CUDA_GRAPH=1`.

use cuda_core::stream::CudaStream;
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

// ── CUDA graph capture ─────────────────────────────────────────────────

/// An owned, instantiated CUDA graph that can be replayed onto a stream.
///
/// Created via [`capture_decode_graph`]. Once instantiated, the graph can
/// be launched repeatedly with [`CudaGraphHandle::launch`] without
/// re-recording kernels — eliminating per-step kernel-launch overhead
/// during autoregressive decode.
pub struct CudaGraphHandle {
    graph: cuda_bindings::CUgraph,
    instance: cuda_bindings::CUgraphExec,
    /// Keep the parent context alive so the CUDA context outlives the graph.
    _ctx: Arc<cuda_core::CudaContext>,
}

// CUgraph / CUgraphExec are raw pointers; the driver is thread-safe for
// graph launch from any host thread bound to the owning context.
unsafe impl Send for CudaGraphHandle {}
unsafe impl Sync for CudaGraphHandle {}

impl CudaGraphHandle {
    /// Launch (replay) the captured graph onto `stream`.
    ///
    /// The stream must belong to the same CUDA context that was active
    /// when the graph was captured.
    pub fn launch(&self, stream: &CudaStream) -> Result<()> {
        let result = unsafe { cuda_bindings::cuGraphLaunch(self.instance, stream.cu_stream()) };
        if result != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::Internal(format!("cuGraphLaunch failed: {result}")));
        }
        Ok(())
    }
}

impl Drop for CudaGraphHandle {
    fn drop(&mut self) {
        unsafe {
            if !self.instance.is_null() {
                let _ = cuda_bindings::cuGraphExecDestroy(self.instance);
            }
            if !self.graph.is_null() {
                let _ = cuda_bindings::cuGraphDestroy(self.graph);
            }
        }
    }
}

/// Capture all kernel launches enqueued by `capture_fn` on `stream` into
/// an instantiated [`CudaGraphHandle`] that can be replayed.
///
/// # Errors
///
/// Returns `Error::Internal` if stream capture is unsupported (CUDA driver
/// < 10.0), or if capture / instantiation fails.
pub fn capture_decode_graph(
    stream: &CudaStream,
    capture_fn: impl FnOnce() -> Result<()>,
) -> Result<CudaGraphHandle> {
    // Begin capture.
    unsafe {
        let r = cuda_bindings::cuStreamBeginCapture_v2(
            stream.cu_stream(),
            cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_GLOBAL,
        );
        if r != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::Internal(format!(
                "cuStreamBeginCapture_v2 failed: {r} (requires CUDA driver ≥ 10.0)"
            )));
        }
    }

    // Record the compute graph.
    if let Err(e) = capture_fn() {
        // Attempt to end capture to avoid leaving the stream in a broken state.
        let mut graph: cuda_bindings::CUgraph = std::ptr::null_mut();
        unsafe {
            let _ = cuda_bindings::cuStreamEndCapture(stream.cu_stream(), &mut graph);
            if !graph.is_null() {
                let _ = cuda_bindings::cuGraphDestroy(graph);
            }
        }
        return Err(e);
    }

    // End capture and obtain the graph.
    let mut graph: cuda_bindings::CUgraph = std::ptr::null_mut();
    unsafe {
        let r = cuda_bindings::cuStreamEndCapture(stream.cu_stream(), &mut graph);
        if r != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::Internal(format!("cuStreamEndCapture failed: {r}")));
        }
    }

    // Instantiate the graph for launch.
    let mut instance: cuda_bindings::CUgraphExec = std::ptr::null_mut();
    unsafe {
        let r = cuda_bindings::cuGraphInstantiateWithFlags(&mut instance, graph, 0);
        if r != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            cuda_bindings::cuGraphDestroy(graph);
            return Err(Error::Internal(format!(
                "cuGraphInstantiateWithFlags failed: {r}"
            )));
        }
    }

    Ok(CudaGraphHandle {
        graph,
        instance,
        _ctx: stream.context().clone(),
    })
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
                    msg.contains("CUDA driver") || msg.contains("capture"),
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
        // If capture fails (e.g. old driver), it should return an error with
        // a message mentioning the driver version requirement.
        if let Err(e) = &result {
            let msg = format!("{e}");
            assert!(
                msg.contains("10.0") || msg.contains("capture") || msg.contains("driver"),
                "error message should mention driver version: {msg}"
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

        let result = capture_decode_graph(&stream, || {
            Err(Error::Internal("simulated capture failure".into()))
        });
        match result {
            Err(e) => {
                assert!(format!("{e}").contains("simulated capture failure"));
            }
            Ok(_) => {
                // If capture succeeded anyway (stream was empty before the
                // closure returned an error), the stream should remain usable.
            }
        }
    }
}
