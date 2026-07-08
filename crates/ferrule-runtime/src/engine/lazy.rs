use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ferrule_common::{Error, Result};
use ferrule_model::TopKModelRunner;

use crate::scheduling::SessionId;

use super::worker::EngineWorker;

/// Load-state snapshot for a lazy resident worker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LazyEngineLoadStats {
    pub loaded: bool,
    pub load_elapsed: Duration,
}

/// Generic background artifact loader for a resident `EngineWorker`.
///
/// The background closure loads an artifact `A` only. The `build_runner` closure
/// is executed by `ensure_loaded()` on the caller thread, which keeps CUDA/device
/// context creation under the foreground worker lifecycle. This mirrors the
/// SGLang/vLLM split between asynchronous weight/materialization work and a
/// resident execution worker.
pub struct LazyEngineWorker<A, R, B>
where
    A: Send + 'static,
    R: TopKModelRunner,
    B: FnOnce(A) -> Result<R>,
{
    loader: Option<JoinHandle<Result<A>>>,
    build_runner: Option<B>,
    worker: Option<EngineWorker<R>>,
    load_started: Instant,
    session_id: SessionId,
}

impl<A, R, B> LazyEngineWorker<A, R, B>
where
    A: Send + 'static,
    R: TopKModelRunner,
    B: FnOnce(A) -> Result<R>,
{
    pub fn spawn<L>(session_id: SessionId, load_artifact: L, build_runner: B) -> Self
    where
        L: FnOnce() -> Result<A> + Send + 'static,
    {
        Self {
            loader: Some(std::thread::spawn(load_artifact)),
            build_runner: Some(build_runner),
            worker: None,
            load_started: Instant::now(),
            session_id,
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.worker.is_some()
    }

    pub fn load_started(&self) -> Instant {
        self.load_started
    }

    pub fn load_stats(&self) -> LazyEngineLoadStats {
        LazyEngineLoadStats {
            loaded: self.is_loaded(),
            load_elapsed: self.load_started.elapsed(),
        }
    }

    pub fn worker(&self) -> Option<&EngineWorker<R>> {
        self.worker.as_ref()
    }

    pub fn worker_mut(&mut self) -> Option<&mut EngineWorker<R>> {
        self.worker.as_mut()
    }

    pub fn position_if_loaded(&self) -> usize {
        self.worker
            .as_ref()
            .map(EngineWorker::position)
            .unwrap_or(0)
    }

    pub fn ensure_loaded(&mut self) -> Result<&mut EngineWorker<R>> {
        if self.worker.is_none() {
            let loader = self
                .loader
                .take()
                .ok_or_else(|| Error::Internal("engine worker loader was not started".into()))?;
            let artifact = loader
                .join()
                .map_err(|_| Error::Internal("engine worker loader panicked".into()))??;
            let build_runner = self
                .build_runner
                .take()
                .ok_or_else(|| Error::Internal("engine worker builder was not installed".into()))?;
            let runner = build_runner(artifact)?;
            self.worker = Some(EngineWorker::with_session(runner, self.session_id));
        }
        Ok(self.worker.as_mut().expect("worker initialized above"))
    }
}
