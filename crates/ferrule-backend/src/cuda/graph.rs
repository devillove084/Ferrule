//! CUDA graph capture and replay for stable decode buckets.

use std::sync::Arc;

use ferrule_common::Result;

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

/// An instantiated graph and the context/source graph that own its handles.
pub struct CudaGraphHandle {
    executable: CudaGraphExec,
    _graph: CudaGraph,
    _context: Arc<CudaContext>,
}

unsafe impl Send for CudaGraphHandle {}
unsafe impl Sync for CudaGraphHandle {}

impl CudaGraphHandle {
    pub fn launch(&self, stream: &CudaStream) -> Result<()> {
        self.executable.launch(stream).map_err(Into::into)
    }

    pub fn upload(&self, stream: &CudaStream) -> Result<()> {
        self.executable.upload(stream).map_err(Into::into)
    }
}

pub fn capture_decode_graph(
    stream: &CudaStream,
    capture: impl FnOnce() -> Result<()>,
) -> Result<CudaGraphHandle> {
    stream.begin_capture()?;
    if let Err(error) = capture() {
        let _ = stream.end_capture();
        return Err(error);
    }
    let graph = stream.end_capture()?;
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
}

/// Auto-captures after one successful warmup for an unchanged pointer/shape key.
pub struct CachedDecodeGraph {
    _context: Arc<CudaContext>,
    state: CacheState,
    key: Vec<u64>,
    graph: Option<CudaGraphHandle>,
}

impl CachedDecodeGraph {
    pub fn new(context: &Arc<CudaContext>) -> Self {
        Self {
            _context: Arc::clone(context),
            state: CacheState::Cold,
            key: Vec::new(),
            graph: None,
        }
    }

    pub fn set_properties(&mut self, data_ptrs: &[*const std::ffi::c_void], shapes: &[u64]) {
        let next = data_ptrs
            .iter()
            .map(|pointer| *pointer as usize as u64)
            .chain(shapes.iter().copied())
            .collect::<Vec<_>>();
        if next != self.key {
            self.key = next;
            self.invalidate();
        }
    }

    pub fn has_cached_graph(&self) -> bool {
        self.state == CacheState::Captured
    }

    pub fn invalidate(&mut self) {
        self.graph = None;
        self.state = CacheState::Cold;
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
        match self.state {
            CacheState::Cold => {
                execute(stream)?;
                self.state = CacheState::Warm;
                Ok(())
            }
            CacheState::Warm => {
                let graph = capture_decode_graph(stream, || capture(stream).map_err(Into::into))?;
                graph.launch(stream)?;
                self.graph = Some(graph);
                self.state = CacheState::Captured;
                Ok(())
            }
            CacheState::Captured => self
                .graph
                .as_ref()
                .expect("captured graph state must own a graph")
                .launch(stream),
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
        let Ok(context) = CudaContext::new(0) else {
            return;
        };
        let mut graph = CachedDecodeGraph::new(&context);
        graph.state = CacheState::Captured;
        graph.set_properties(&[16usize as *const std::ffi::c_void], &[32]);
        assert_eq!(graph.state, CacheState::Cold);
        assert!(!graph.has_cached_graph());
    }
}
