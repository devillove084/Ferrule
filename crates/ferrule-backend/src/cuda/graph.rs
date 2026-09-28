//! CUDA graph capture and replay for stable decode buckets.

#[cfg(feature = "cuda")]
use std::sync::Arc;

use ferrule_common::Result;

#[cfg(feature = "cuda")]
use crate::cuda::runtime::{CudaContext, CudaGraph, CudaGraphExec, CudaResult, CudaStream};

pub fn cuda_graph_enabled() -> bool {
    std::env::var("FERRULE_CUDA_GRAPH")
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "off"
            )
        })
        .unwrap_or(false)
}

pub fn flash_attn_enabled() -> bool {
    std::env::var_os("FERRULE_FLASH_ATTN").is_some()
}

/// An instantiated graph and its native owners, **not** ownership of operands.
///
/// All captured allocations (including scratch and pinned inputs) must outlive
/// every replay. Before overwrite, cross-stream reuse or release, the caller
/// must retain and complete an event covering the last actual consumer. Neither
/// successful launch nor keeping this handle/context alive proves retirement.
/// Unknown completion is terminal; do not fallback or replay it.
#[cfg(feature = "cuda")]
pub struct CudaGraphHandle {
    executable: CudaGraphExec,
    _graph: CudaGraph,
    _context: Arc<CudaContext>,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaGraphHandle {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaGraphHandle {}

#[cfg(feature = "cuda")]
impl CudaGraphHandle {
    pub fn launch(&self, stream: &CudaStream) -> Result<()> {
        self.executable.launch(stream).map_err(Into::into)
    }

    pub fn upload(&self, stream: &CudaStream) -> Result<()> {
        self.executable.upload(stream).map_err(Into::into)
    }
}

/// Own exactly one end after a successful begin. No Send/'static restrictions
/// are placed on closures or their borrowed, possibly non-Send resources.
struct CaptureGuard<F: FnOnce() -> Result<G>, G> {
    end: Option<F>,
}

impl<F: FnOnce() -> Result<G>, G> CaptureGuard<F, G> {
    fn begin(begin: impl FnOnce() -> Result<()>, end: F) -> Result<Self> {
        begin()?;
        Ok(Self { end: Some(end) })
    }

    fn finish(mut self) -> Result<G> {
        self.end.take().expect("armed capture guard")()
    }
}

impl<F: FnOnce() -> Result<G>, G> Drop for CaptureGuard<F, G> {
    fn drop(&mut self) {
        if let Some(end) = self.end.take() {
            // A cleanup error cannot replace the original panic. Runtime end
            // records unknown state; log the secondary failure without unwinding.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| end().map(drop)));
            use std::io::Write;
            match result {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    let _ = writeln!(
                        std::io::stderr(),
                        "CUDA capture unwind cleanup failed: {error}"
                    );
                }
                Err(_) => {
                    let _ = writeln!(std::io::stderr(), "CUDA capture unwind cleanup panicked");
                }
            }
        }
    }
}

fn with_capture<G>(
    begin: impl FnOnce() -> Result<()>,
    end: impl FnOnce() -> Result<G>,
    capture: impl FnOnce() -> Result<()>,
) -> Result<G> {
    let guard = CaptureGuard::begin(begin, end)?;
    match capture() {
        Ok(()) => guard.finish(),
        Err(primary) => Err(ferrule_common::Error::with_cleanup(
            "CUDA graph capture",
            primary,
            guard.finish().map(drop),
        )),
    }
}

#[cfg(feature = "cuda")]
pub fn capture_decode_graph(
    stream: &CudaStream,
    capture: impl FnOnce() -> Result<()>,
) -> Result<CudaGraphHandle> {
    let graph = with_capture(
        || stream.begin_capture().map_err(Into::into),
        || stream.end_capture().map_err(Into::into),
        capture,
    )?;
    let executable = graph.instantiate()?;
    Ok(CudaGraphHandle {
        executable,
        _graph: graph,
        _context: Arc::clone(stream.context()),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheState {
    Cold,
    Warm,
    Captured,
    /// Capture ended, but construction failed. No automatic fallback/retry.
    Failed,
    /// Work may have been submitted without completion evidence.
    Unknown,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct GraphKey {
    pointers: Vec<usize>,
    shapes: Vec<u64>,
}

impl GraphKey {
    fn new(pointers: &[*const std::ffi::c_void], shapes: &[u64]) -> Self {
        Self {
            pointers: pointers.iter().map(|p| *p as usize).collect(),
            shapes: shapes.to_vec(),
        }
    }
}

struct CacheEntry<G> {
    state: CacheState,
    key: GraphKey,
    graph: Option<G>,
}

impl<G> Default for CacheEntry<G> {
    fn default() -> Self {
        Self {
            state: CacheState::Cold,
            key: GraphKey::default(),
            graph: None,
        }
    }
}

impl<G> CacheEntry<G> {
    fn set_properties(&mut self, key: GraphKey) {
        if self.key != key {
            self.invalidate();
            self.key = key;
        }
    }

    fn invalidate(&mut self) {
        if !matches!(self.state, CacheState::Failed | CacheState::Unknown) {
            self.graph = None;
            self.state = CacheState::Cold;
        }
    }
}

#[cfg(feature = "cuda")]
struct ExecutionAttempt<'a> {
    context: &'a CudaContext,
    completed: bool,
}

#[cfg(feature = "cuda")]
impl Drop for ExecutionAttempt<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.context.mark_graph_unknown();
        }
    }
}

/// Auto-captures after one successful warmup for an unchanged structured key.
/// Uses the first accepted exact stream owner; subsequent owner mismatches are
/// rejected before calling either closure. Failed/Unknown are sticky, including
/// across invalidate/key changes. Recovery requires a new cache at a proven
/// quiescent boundary, not automatic execution fallback.
///
/// The caller retains operands and last-consumer events as for CudaGraphHandle.
#[cfg(feature = "cuda")]
pub struct CachedDecodeGraph {
    context: Arc<CudaContext>,
    owner: Option<Arc<CudaStream>>,
    entry: CacheEntry<CudaGraphHandle>,
}

#[cfg(feature = "cuda")]
impl CachedDecodeGraph {
    pub fn new(context: &Arc<CudaContext>) -> Self {
        Self {
            context: Arc::clone(context),
            owner: None,
            entry: CacheEntry::default(),
        }
    }

    pub(crate) fn for_stream(stream: &Arc<CudaStream>) -> Self {
        Self {
            context: Arc::clone(stream.context()),
            owner: Some(Arc::clone(stream)),
            entry: CacheEntry::default(),
        }
    }

    pub fn set_properties(&mut self, data_ptrs: &[*const std::ffi::c_void], shapes: &[u64]) {
        if self.context.graph_completion_unknown() {
            self.entry.state = CacheState::Unknown;
        }
        self.entry.set_properties(GraphKey::new(data_ptrs, shapes));
    }

    pub fn has_cached_graph(&self) -> bool {
        self.entry.state == CacheState::Captured && !self.context.graph_completion_unknown()
    }

    pub fn invalidate(&mut self) {
        if self.context.graph_completion_unknown() {
            self.entry.state = CacheState::Unknown;
        }
        self.entry.invalidate();
    }

    pub fn launch_or_capture<C, E>(
        &mut self,
        stream: &CudaStream,
        capture: C,
        execute: E,
    ) -> Result<()>
    where
        C: FnOnce(&CudaStream) -> CudaResult<()>,
        E: FnOnce(&CudaStream) -> CudaResult<()>,
    {
        if !Arc::ptr_eq(&self.context, stream.context())
            || self
                .owner
                .as_ref()
                .is_some_and(|owner| !std::ptr::eq(owner.as_ref(), stream))
        {
            return Err(ferrule_common::Error::Graph {
                message: "cached graph context/stream owner mismatch".into(),
            });
        }
        self.context.check_graph_ready()?;
        if matches!(self.entry.state, CacheState::Failed | CacheState::Unknown) {
            return Err(ferrule_common::Error::Graph {
                message: "cached graph failure is terminal; completion evidence required".into(),
            });
        }
        if self.owner.is_none() {
            self.owner = Some(stream.owner()?);
        }
        let previous = self.entry.state;
        // Panic/Err must not leave an executable cache or authorize fallback.
        self.entry.state = CacheState::Unknown;
        match previous {
            CacheState::Cold => {
                let mut attempt = ExecutionAttempt {
                    context: &self.context,
                    completed: false,
                };
                execute(stream)?;
                attempt.completed = true;
                self.entry.state = CacheState::Warm;
                Ok(())
            }
            CacheState::Warm => {
                let graph =
                    match capture_decode_graph(stream, || capture(stream).map_err(Into::into)) {
                        Ok(graph) => graph,
                        Err(error) => {
                            if !self.context.graph_completion_unknown() {
                                self.entry.state = CacheState::Failed;
                            }
                            return Err(error);
                        }
                    };
                // Keep native custody even when the first launch fails.
                self.entry.graph = Some(graph);
                self.entry.graph.as_ref().unwrap().launch(stream)?;
                self.entry.state = CacheState::Captured;
                Ok(())
            }
            CacheState::Captured => {
                self.entry
                    .graph
                    .as_ref()
                    .expect("captured graph must own executable")
                    .launch(stream)?;
                self.entry.state = CacheState::Captured;
                Ok(())
            }
            CacheState::Failed | CacheState::Unknown => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn set_graph_env(value: Option<&str>) {
        unsafe {
            match value {
                Some(value) => std::env::set_var("FERRULE_CUDA_GRAPH", value),
                None => std::env::remove_var("FERRULE_CUDA_GRAPH"),
            }
        }
    }

    #[test]
    fn graph_flag_is_fail_closed() {
        let _guard = ENV_LOCK.lock().unwrap();
        for value in [None, Some(""), Some("0"), Some("false"), Some("off")] {
            set_graph_env(value);
            assert!(!cuda_graph_enabled());
        }
        set_graph_env(Some("1"));
        assert!(cuda_graph_enabled());
        set_graph_env(None);
    }

    #[test]
    fn property_change_invalidates_cached_state() {
        let mut entry = CacheEntry::<()>::default();
        entry.state = CacheState::Captured;
        entry.graph = Some(());
        entry.set_properties(GraphKey::new(&[16usize as *const std::ffi::c_void], &[32]));
        assert_eq!(entry.state, CacheState::Cold);
        assert!(entry.graph.is_none());
    }
}

#[cfg(test)]
mod protocol_tests {
    use super::*;
    use ferrule_common::Error;
    use std::cell::Cell;
    use std::rc::Rc;

    fn failure(message: &str) -> Error {
        Error::Graph {
            message: message.into(),
        }
    }

    #[test]
    fn begin_failure_never_arms_or_ends() {
        let ends = Cell::new(0);
        let captures = Cell::new(0);
        let result = with_capture(
            || Err(failure("begin")),
            || {
                ends.set(ends.get() + 1);
                Ok(())
            },
            || {
                captures.set(1);
                Ok(())
            },
        );
        assert!(result.is_err());
        assert_eq!((ends.get(), captures.get()), (0, 0));
    }

    #[test]
    fn capture_error_and_double_error_end_exactly_once() {
        for cleanup_fails in [false, true] {
            let ends = Cell::new(0);
            let error = with_capture(
                || Ok(()),
                || {
                    ends.set(ends.get() + 1);
                    if cleanup_fails {
                        Err(failure("cleanup"))
                    } else {
                        Ok(())
                    }
                },
                || Err(failure("primary")),
            )
            .unwrap_err();
            assert_eq!(ends.get(), 1);
            assert!(error.to_string().contains("primary"));
            assert_eq!(matches!(error, Error::Cleanup { .. }), cleanup_fails);
        }
    }

    #[test]
    fn unwind_ends_once_even_if_cleanup_fails_or_panics() {
        for mode in 0..3 {
            let ends = Rc::new(Cell::new(0));
            let borrowed_non_send = Rc::new(Cell::new(0));
            let mut borrowed = 0;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _: Result<()> = with_capture(
                    || Ok(()),
                    || {
                        ends.set(ends.get() + 1);
                        match mode {
                            1 => Err(failure("cleanup")),
                            2 => panic!("cleanup panic"),
                            _ => Ok(()),
                        }
                    },
                    || {
                        borrowed_non_send.set(1);
                        borrowed += 1;
                        panic!("primary panic")
                    },
                );
            }));
            assert!(result.is_err());
            assert_eq!(ends.get(), 1);
            assert_eq!(borrowed, 1);
            assert_eq!(borrowed_non_send.get(), 1);
        }
    }

    #[test]
    fn successful_finish_and_end_error_never_double_end() {
        for fails in [false, true] {
            let ends = Cell::new(0);
            let result = with_capture(
                || Ok(()),
                || {
                    ends.set(ends.get() + 1);
                    if fails { Err(failure("end")) } else { Ok(17) }
                },
                || Ok(()),
            );
            assert_eq!(result.is_err(), fails);
            assert_eq!(ends.get(), 1);
        }
    }

    #[test]
    fn structured_key_distinguishes_pointer_shape_partition_and_lengths() {
        let a = GraphKey::new(&[16usize as *const _, 32usize as *const _], &[64]);
        let b = GraphKey::new(&[16usize as *const _], &[32, 64]);
        assert_ne!(a, b);
        assert_ne!(
            GraphKey::new(&[], &[0]),
            GraphKey::new(&[std::ptr::null()], &[])
        );
        let mut entry = CacheEntry::<()>::default();
        entry.set_properties(a);
        entry.state = CacheState::Captured;
        entry.graph = Some(());
        entry.set_properties(b);
        assert_eq!(entry.state, CacheState::Cold);
        assert!(entry.graph.is_none());
    }

    #[test]
    fn terminal_state_never_resets_or_drops_custody_on_key_change() {
        for state in [CacheState::Failed, CacheState::Unknown] {
            let lease = Rc::new(());
            let mut entry = CacheEntry {
                state,
                key: GraphKey::default(),
                graph: Some(Rc::clone(&lease)),
            };
            entry.invalidate();
            entry.set_properties(GraphKey::new(&[], &[42]));
            assert_eq!(entry.state, state);
            assert_eq!(Rc::strong_count(&lease), 2);
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod gpu_lifecycle_tests {
    use super::*;
    #[cfg(feature = "cuda")]
    use crate::cuda::runtime::{DeviceBuffer, GraphFault};

    fn setup() -> (Arc<CudaContext>, Arc<CudaStream>) {
        let context =
            CudaContext::new(0).expect("required actual CUDA device; initialization errors fail");
        let stream = context.new_stream().unwrap();
        (context, stream)
    }

    #[test]
    #[ignore = "actual CUDA GPU; scoped begin/Err/panic/end lifecycle"]
    fn begin_error_closure_error_and_panic_end_exactly_once() {
        let (context, stream) = setup();
        context.arm_graph_fault(GraphFault::Begin);
        assert!(capture_decode_graph(&stream, || panic!("begin failed")).is_err());
        assert_eq!(context.graph_calls(GraphFault::End), 0);
        assert!(!context.is_capturing());
        assert!(
            capture_decode_graph(&stream, || Err(ferrule_common::Error::Graph {
                message: "primary".into()
            }))
            .is_err()
        );
        assert_eq!(context.graph_calls(GraphFault::End), 1);
        assert!(!context.is_capturing());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = capture_decode_graph(&stream, || panic!("primary unwind"));
        }));
        assert!(result.is_err());
        assert_eq!(context.graph_calls(GraphFault::End), 2);
        assert!(!context.is_capturing());
        let graph = capture_decode_graph(&stream, || Ok(())).unwrap();
        graph.upload(&stream).unwrap();
        graph.launch(&stream).unwrap();
        stream.record_event(None).unwrap().synchronize().unwrap();
        assert_eq!(context.graph_calls(GraphFault::End), 3);
    }

    #[test]
    #[ignore = "actual CUDA GPU; injected end code 3 retains unknown and both errors"]
    fn end_code3_unknown_preserves_depth_and_operand_retirement() {
        let (context, stream) = setup();
        let operand = DeviceBuffer::<u8>::zeroed(&stream, 4096).unwrap();
        stream.synchronize().unwrap();
        context.arm_graph_fault(GraphFault::End);
        let error = capture_decode_graph(&stream, || {
            Err(ferrule_common::Error::Graph {
                message: "primary".into(),
            })
        })
        .err()
        .unwrap();
        eprintln!("EXPECTED INJECTED CODE3 (not driver damage): {error}");
        assert!(matches!(error, ferrule_common::Error::Cleanup { .. }));
        assert!(error.to_string().contains("(3)"));
        assert!(context.graph_completion_unknown());
        assert!(context.is_capturing());
        assert_eq!(context.graph_calls(GraphFault::End), 1);
        drop(operand);
        let before = context.allocator_metrics();
        assert!(before.capture_retirement_bytes >= 4096);
        assert!(DeviceBuffer::<u8>::zeroed(&stream, 1).is_err());
        assert!(unsafe { DeviceBuffer::<u8>::managed(&context, 1) }.is_err());
        assert!(stream.synchronize().is_err());
        assert!(capture_decode_graph(&stream, || Ok(())).is_err());
        context.shutdown_allocator();
        let after = context.allocator_metrics();
        assert_eq!(after.driver_frees, before.driver_frees);
        assert_eq!(
            after.capture_retirement_bytes,
            before.capture_retirement_bytes
        );
    }

    #[test]
    #[ignore = "actual CUDA GPU; unwind cleanup unknown does not double end"]
    fn panic_plus_end_failure_keeps_unknown_without_damaging_driver() {
        let (context, stream) = setup();
        context.arm_graph_fault(GraphFault::End);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = capture_decode_graph(&stream, || panic!("primary panic"));
        }));
        assert!(result.is_err());
        assert_eq!(context.graph_calls(GraphFault::End), 1);
        assert!(context.graph_completion_unknown());
        // The hook really ended capture before withholding its result.
        let (_, fresh) = setup();
        let graph = capture_decode_graph(&fresh, || Ok(())).unwrap();
        graph.launch(&fresh).unwrap();
        fresh.synchronize().unwrap();
    }

    #[test]
    #[ignore = "actual CUDA GPU; exact owner mismatch makes zero native calls"]
    fn exact_owner_precheck_precedes_launch_upload_and_cached_closures() {
        let (context, stream) = setup();
        let (_, foreign) = setup();
        let sibling = context.new_stream().unwrap();
        let graph = capture_decode_graph(&stream, || Ok(())).unwrap();
        for wrong in [&foreign, &sibling] {
            assert!(graph.launch(wrong).is_err());
            assert!(graph.upload(wrong).is_err());
        }
        assert_eq!(context.graph_calls(GraphFault::Launch), 0);
        assert_eq!(context.graph_calls(GraphFault::Upload), 0);
        let mut cache = CachedDecodeGraph::for_stream(&stream);
        for wrong in [&foreign, &sibling] {
            assert!(
                cache
                    .launch_or_capture(wrong, |_| panic!("capture"), |_| panic!("execute"))
                    .is_err()
            );
            assert_eq!(cache.entry.state, CacheState::Cold);
        }
        foreign.context().bind_to_thread().unwrap();
        graph.upload(&stream).unwrap();
        graph.launch(&stream).unwrap();
        stream.synchronize().unwrap();
    }

    fn warm(cache: &mut CachedDecodeGraph, stream: &CudaStream) {
        cache
            .launch_or_capture(stream, |_| panic!("cold cannot capture"), |_| Ok(()))
            .unwrap();
        assert_eq!(cache.entry.state, CacheState::Warm);
    }

    #[test]
    #[ignore = "actual CUDA GPU; cache construction failure transitions"]
    fn cached_failures_are_terminal_without_fallback() {
        for fault in [GraphFault::Begin, GraphFault::End, GraphFault::Instantiate] {
            let (context, stream) = setup();
            let mut cache = CachedDecodeGraph::new(&context);
            warm(&mut cache, &stream);
            context.arm_graph_fault(fault);
            assert!(
                cache
                    .launch_or_capture(&stream, |_| Ok(()), |_| panic!("fallback"))
                    .is_err()
            );
            let expected = if matches!(fault, GraphFault::End) {
                CacheState::Unknown
            } else {
                CacheState::Failed
            };
            assert_eq!(cache.entry.state, expected);
            cache.invalidate();
            cache.set_properties(&[], &[17]);
            assert_eq!(cache.entry.state, expected);
            assert!(
                cache
                    .launch_or_capture(&stream, |_| panic!("recapture"), |_| panic!("fallback"))
                    .is_err()
            );
        }
        let (context, stream) = setup();
        let mut cache = CachedDecodeGraph::new(&context);
        warm(&mut cache, &stream);
        assert!(
            cache
                .launch_or_capture(
                    &stream,
                    |_| Err(crate::cuda::runtime::CudaError::internal("closure")),
                    |_| panic!("fallback")
                )
                .is_err()
        );
        assert_eq!(cache.entry.state, CacheState::Failed);
        assert!(!context.is_capturing());
    }

    #[test]
    #[ignore = "actual CUDA GPU; first launch/replay/upload unknown preserves graph custody"]
    fn native_submission_unknown_is_sticky_and_keeps_graph() {
        for first in [true, false] {
            let (context, stream) = setup();
            let mut cache = CachedDecodeGraph::new(&context);
            warm(&mut cache, &stream);
            if !first {
                cache
                    .launch_or_capture(&stream, |_| Ok(()), |_| panic!("fallback"))
                    .unwrap();
                stream.record_event(None).unwrap().synchronize().unwrap();
                assert_eq!(cache.entry.state, CacheState::Captured);
            }
            context.arm_graph_fault(GraphFault::Launch);
            assert!(
                cache
                    .launch_or_capture(&stream, |_| Ok(()), |_| panic!("fallback"))
                    .is_err()
            );
            assert_eq!(cache.entry.state, CacheState::Unknown);
            assert!(cache.entry.graph.is_some());
            assert!(!cache.has_cached_graph());
            cache.invalidate();
            cache.set_properties(&[], &[99]);
            assert!(cache.entry.graph.is_some());
            assert!(
                cache
                    .launch_or_capture(&stream, |_| panic!("recapture"), |_| panic!("fallback"))
                    .is_err()
            );
        }
        let (context, stream) = setup();
        let graph = capture_decode_graph(&stream, || Ok(())).unwrap();
        context.arm_graph_fault(GraphFault::Upload);
        assert!(graph.upload(&stream).is_err());
        assert!(graph.launch(&stream).is_err());
        assert_eq!(context.graph_calls(GraphFault::Launch), 0);
    }

    #[test]
    #[ignore = "actual CUDA GPU; cold execution Err and panic cannot warm or fallback"]
    fn cold_error_and_panic_are_unknown() {
        for panic in [false, true] {
            let (context, stream) = setup();
            let mut cache = CachedDecodeGraph::new(&context);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cache.launch_or_capture(
                    &stream,
                    |_| panic!("capture"),
                    |_| {
                        if panic {
                            panic!("execute panic");
                        }
                        Err(crate::cuda::runtime::CudaError::internal("execute failure"))
                    },
                )
            }));
            assert!(if panic {
                result.is_err()
            } else {
                result.unwrap().is_err()
            });
            assert_eq!(cache.entry.state, CacheState::Unknown);
            assert!(context.graph_completion_unknown());
            cache.invalidate();
            assert!(
                cache
                    .launch_or_capture(&stream, |_| panic!("capture"), |_| panic!("execute"))
                    .is_err()
            );
        }
    }
}
