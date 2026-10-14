//! CUDA-independent capture scope and cache identity/state protocol.
//!
//! Native owners and submission/completion evidence stay in the CUDA wrapper.

use ferrule_common::Result;

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

pub(super) fn with_capture<G>(
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CacheState {
    Cold,
    Warm,
    Captured,
    /// Capture ended, but construction failed. No automatic fallback/retry.
    Failed,
    /// Work may have been submitted without completion evidence.
    Unknown,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct GraphKey {
    pointers: Vec<usize>,
    shapes: Vec<u64>,
}

impl GraphKey {
    pub(super) fn new(pointers: &[*const std::ffi::c_void], shapes: &[u64]) -> Self {
        Self {
            pointers: pointers.iter().map(|p| *p as usize).collect(),
            shapes: shapes.to_vec(),
        }
    }
}

pub(super) struct CacheEntry<G> {
    pub(super) state: CacheState,
    key: GraphKey,
    pub(super) graph: Option<G>,
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
    pub(super) fn set_properties(&mut self, key: GraphKey) {
        if self.key != key {
            self.invalidate();
            self.key = key;
        }
    }

    pub(super) fn invalidate(&mut self) {
        if !matches!(self.state, CacheState::Failed | CacheState::Unknown) {
            self.graph = None;
            self.state = CacheState::Cold;
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
        let mut entry = CacheEntry {
            state: CacheState::Captured,
            graph: Some(()),
            ..Default::default()
        };
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
