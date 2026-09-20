//! One CUDA workload and quiescence implementation for thread and process ranks.

use anyhow::{Context, Result, ensure};
use ferrule_runtime::parallel::data::PanicQuiescence;
use std::mem::size_of;
use std::panic::{AssertUnwindSafe, catch_unwind};

pub(crate) fn bytes<T>(count: usize) -> Result<usize> {
    count
        .checked_mul(size_of::<T>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .context("allocation byte size overflow")
}

pub(crate) fn elements(rows: usize, width: usize) -> Result<usize> {
    ensure!(
        rows > 0 && width > 0,
        "shape dimensions must be greater than zero"
    );
    let count = rows
        .checked_mul(width)
        .context("shape element count overflow")?;
    bytes::<f32>(count)?;
    Ok(count)
}

// Panic paths must retain the allocation, not merely suppress a stack
// value's destructor. Only release after proving all device/transport fences.
pub(crate) struct RetainUntilQuiescent<T>(Option<Box<T>>);

impl<T> RetainUntilQuiescent<T> {
    pub(crate) fn new(value: T) -> Self {
        Self(Some(Box::new(value)))
    }

    pub(crate) fn release(mut self) -> T {
        *self.0.take().expect("retained allocation")
    }
}

impl<T> std::ops::Deref for RetainUntilQuiescent<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.0.as_deref().expect("retained allocation")
    }
}

impl<T> Drop for RetainUntilQuiescent<T> {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            std::mem::forget(value);
        }
    }
}

pub(crate) fn sync_both(
    compute: impl FnOnce() -> Result<()>,
    upload: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let compute = attempt(compute);
    let upload = attempt(upload);
    match (compute, upload) {
        (Ok(()), Ok(())) => Ok(()),
        (compute, upload) => anyhow::bail!("compute sync={compute:?}; upload sync={upload:?}"),
    }
}

fn attempt(fence: impl FnOnce() -> Result<()>) -> Result<()> {
    match catch_unwind(AssertUnwindSafe(fence)) {
        Ok(result) => result,
        Err(payload) => {
            // Even typed Unknown must not prevent the remaining fence attempts.
            // This error is only internal evidence, never a worker failure ACK.
            std::mem::forget(payload);
            anyhow::bail!("owner fence panicked")
        }
    }
}

#[cfg(any(test, feature = "cuda"))]
fn sync_all(
    shard: impl FnOnce() -> Result<()>,
    compute: impl FnOnce() -> Result<()>,
    upload: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let shard = attempt(shard);
    let streams = sync_both(compute, upload);
    match (shard, streams) {
        (Ok(()), Ok(())) => Ok(()),
        _ => anyhow::bail!("owner quiescence proof failed"),
    }
}

fn complete_owner<T>(
    result: std::thread::Result<Result<T>>,
    fence: impl FnOnce() -> Result<()>,
) -> Result<T> {
    // Keep outputs and panic payloads alive until ALL fences have been tried.
    // Process children have no ReplicaWorker panic hook to do this for us.
    let result = RetainUntilQuiescent::new(result);
    if let Err(error) = attempt(fence) {
        std::mem::forget(error);
        std::panic::panic_any(PanicQuiescence::Unknown);
    }
    result.release().unwrap_or_else(|payload| {
        if payload.downcast_ref::<PanicQuiescence>() == Some(&PanicQuiescence::Unknown) {
            std::panic::resume_unwind(payload);
        }
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("non-string panic")
            .to_owned();
        std::mem::forget(payload);
        Err(anyhow::anyhow!("CUDA owner panic: {message}"))
    })
}

pub(crate) fn owner_fenced<T>(
    work: impl FnOnce() -> Result<T>,
    compute: impl FnOnce() -> Result<()>,
    upload: impl FnOnce() -> Result<()>,
) -> Result<T> {
    complete_owner(catch_unwind(AssertUnwindSafe(work)), || {
        sync_both(compute, upload)
    })
}

#[cfg(any(test, feature = "cuda"))]
fn checked_error(
    error: impl std::error::Error + Send + Sync + 'static,
    needs_quarantine: bool,
) -> anyhow::Error {
    if needs_quarantine {
        // Retained-source errors may own custody, and must not be formatted or
        // dropped before the typed fatal signal reaches the owner boundary.
        std::mem::forget(error);
        std::panic::panic_any(PanicQuiescence::Unknown);
    }
    anyhow::Error::new(error)
}

#[cfg(any(test, feature = "cuda"))]
fn require_fence(result: ferrule_common::Result<()>) -> Result<()> {
    if let Err(error) = result {
        // Every failed fence is unknown, even without a typed shard source.
        return Err(checked_error(error, true));
    }
    Ok(())
}

pub(crate) fn runtime_error(error: impl std::fmt::Debug) -> anyhow::Error {
    anyhow::anyhow!("parallel executor: {error:?}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Fence {
        Ready,
        Error,
        Panic,
        Unknown,
    }

    impl Fence {
        fn run(self, calls: &RefCell<Vec<&'static str>>, name: &'static str) -> Result<()> {
            calls.borrow_mut().push(name);
            match self {
                Self::Ready => Ok(()),
                Self::Error => anyhow::bail!("{name} failed"),
                Self::Panic => panic!("{name} panicked"),
                Self::Unknown => std::panic::panic_any(PanicQuiescence::Unknown),
            }
        }
    }

    fn assert_unknown<T: std::fmt::Debug>(result: std::thread::Result<T>) {
        assert_eq!(
            result.unwrap_err().downcast_ref::<PanicQuiescence>(),
            Some(&PanicQuiescence::Unknown)
        );
    }

    #[test]
    fn shard_control_and_both_streams_are_attempted_independently() {
        let modes = [Fence::Ready, Fence::Error, Fence::Panic, Fence::Unknown];
        for shard in modes {
            for compute in modes {
                for upload in modes {
                    let calls = RefCell::new(Vec::new());
                    let result = sync_all(
                        || shard.run(&calls, "control/tickets"),
                        || compute.run(&calls, "compute"),
                        || upload.run(&calls, "upload"),
                    );
                    assert_eq!(*calls.borrow(), ["control/tickets", "compute", "upload"]);
                    assert_eq!(
                        result.is_ok(),
                        [shard, compute, upload] == [Fence::Ready; 3]
                    );
                }
            }
        }
    }

    #[test]
    fn completion_and_shutdown_never_ack_unknown_control_custody() {
        // Success includes shutdown's empty result; an ordinary error/panic is
        // equally unable to ACK when control D2H or a stream fence is unknown.
        for work in [Fence::Ready, Fence::Error, Fence::Panic, Fence::Unknown] {
            for failed_fence in 0..3 {
                let calls = RefCell::new(Vec::new());
                let result = catch_unwind(AssertUnwindSafe(|| {
                    complete_owner(
                        catch_unwind(AssertUnwindSafe(|| work.run(&calls, "work"))),
                        || {
                            let fence = |index, name| {
                                if index == failed_fence {
                                    Fence::Error
                                } else {
                                    Fence::Ready
                                }
                                .run(&calls, name)
                            };
                            sync_all(
                                || fence(0, "control/tickets"),
                                || fence(1, "compute"),
                                || fence(2, "upload"),
                            )
                        },
                    )
                }));
                assert_unknown(result);
                assert_eq!(
                    *calls.borrow(),
                    ["work", "control/tickets", "compute", "upload"]
                );
            }
        }
    }

    #[test]
    fn typed_work_unknown_survives_successful_drain_and_ordinary_panics_wait_for_it() {
        for work in [Fence::Ready, Fence::Error, Fence::Panic, Fence::Unknown] {
            let calls = RefCell::new(Vec::new());
            let result = catch_unwind(AssertUnwindSafe(|| {
                complete_owner(
                    catch_unwind(AssertUnwindSafe(|| work.run(&calls, "work"))),
                    || {
                        sync_all(
                            || Fence::Ready.run(&calls, "control/tickets"),
                            || Fence::Ready.run(&calls, "compute"),
                            || Fence::Ready.run(&calls, "upload"),
                        )
                    },
                )
            }));
            assert_eq!(
                *calls.borrow(),
                ["work", "control/tickets", "compute", "upload"]
            );
            if work == Fence::Unknown {
                assert_unknown(result);
            } else {
                assert_eq!(result.unwrap().is_ok(), work == Fence::Ready);
            }
        }
    }

    #[derive(Debug)]
    struct DropProbe(Rc<Cell<usize>>);

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[test]
    fn native_fence_error_is_unknown_even_without_a_typed_shard_error() {
        assert_unknown(catch_unwind(|| {
            require_fence(Err(ferrule_common::Error::ModelSource {
                source: Box::new(std::io::Error::other("native fence failed")),
            }))
        }));
        require_fence(Ok(())).unwrap();
    }

    #[test]
    fn control_failure_retains_output_and_does_not_release_active_resources() {
        let drops = Rc::new(Cell::new(0));
        let released = Cell::new(false);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _active = RetainUntilQuiescent::new(DropProbe(Rc::clone(&drops)));
            let output = complete_owner(Ok(Ok(DropProbe(Rc::clone(&drops)))), || {
                sync_all(
                    || anyhow::bail!("control D2H pending"),
                    || Ok(()),
                    || Ok(()),
                )
            });
            released.set(true);
            output
        }));
        assert_unknown(result);
        assert!(!released.get());
        assert_eq!(drops.get(), 0);
    }

    #[test]
    fn retained_source_constructor_error_is_upgraded_before_format_or_drop() {
        #[derive(Default)]
        struct RetainedSource;
        impl std::fmt::Debug for RetainedSource {
            fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                panic!("unknown error must not be stringified")
            }
        }
        impl std::fmt::Display for RetainedSource {
            fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                panic!("unknown error must not be displayed")
            }
        }
        impl std::error::Error for RetainedSource {}
        impl Drop for RetainedSource {
            fn drop(&mut self) {
                panic!("unknown retained source must not be dropped")
            }
        }
        let calls = RefCell::new(Vec::new());
        let drops = Rc::new(Cell::new(0));
        let result = catch_unwind(AssertUnwindSafe(|| {
            let owner = RetainUntilQuiescent::new(DropProbe(Rc::clone(&drops)));
            let result = owner_fenced::<()>(
                || {
                    Err(checked_error(
                        ferrule_common::Error::ModelSource {
                            source: Box::new(RetainedSource),
                        },
                        true,
                    ))
                },
                || Fence::Ready.run(&calls, "compute"),
                || Fence::Ready.run(&calls, "upload"),
            );
            drop(owner.release());
            result
        }));
        assert_unknown(result);
        assert_eq!(*calls.borrow(), ["compute", "upload"]);
        assert_eq!(
            drops.get(),
            0,
            "factory has no runtime worker to quarantine yet"
        );

        for error in [
            ferrule_common::Error::Model {
                message: "invalid shape".into(),
            },
            ferrule_common::Error::ModelSource {
                source: Box::new(std::io::Error::other("ordinary model source")),
            },
        ] {
            let result =
                owner_fenced::<()>(|| Err(checked_error(error, false)), || Ok(()), || Ok(()));
            assert!(
                result
                    .unwrap_err()
                    .downcast_ref::<ferrule_common::Error>()
                    .is_some()
            );
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) mod cuda {
    use super::*;
    use ferrule_backend::cuda::operators::linear::{CudaF32Buffer, CudaOperators};
    use ferrule_backend::cuda::providers::CudaContext;
    use ferrule_common::ParallelRankId;
    use ferrule_model::transformer::parallel::{
        CudaLinearShard, CudaShardError, TensorParallelLinearPlan,
    };
    use ferrule_runtime::parallel::data::{ReplicaWorker, WorkRequest};
    use ferrule_runtime::parallel::tensor::{TensorCommand, TensorWork};
    use std::rc::Rc;
    use std::sync::Arc;
    fn shard_error(error: ferrule_common::Error) -> anyhow::Error {
        let quarantine =
            CudaShardError::from_error(&error).is_some_and(CudaShardError::needs_quarantine);
        checked_error(error, quarantine)
    }

    pub(crate) struct LinearWorker {
        linear: CudaLinearShard,
        ops: Rc<CudaOperators>,
        rows: usize,
        // Runtime quarantine retains these through stack unwinding. They
        // are cleared only after an ordinary, fully fenced return.
        active: Option<TensorCommand>,
        apply_buffer: Option<CudaF32Buffer>,
    }

    impl LinearWorker {
        pub(crate) fn new(
            device: usize,
            plan: TensorParallelLinearPlan,
            rank: ParallelRankId,
            weight: &[f32],
            rows: usize,
        ) -> Result<Self> {
            let visible = CudaContext::device_count().context("probe rank CUDA device")?;
            ensure!(
                device < visible,
                "device {device} outside {visible} visible CUDA devices"
            );
            let ops = RetainUntilQuiescent::new(Rc::new(CudaOperators::new_on_device(device)?));
            // A failed constructor can already carry retained-source Unknown;
            // classify it before the factory's fallback stream fences or text.
            let linear = owner_fenced(
                || CudaLinearShard::new(Rc::clone(&*ops), plan, rank, weight).map_err(shard_error),
                || require_fence(ops.sync_stream()),
                || require_fence(ops.sync_upload_stream()),
            );
            let ops = ops.release();
            Ok(Self {
                linear: linear?,
                ops,
                rows,
                active: None,
                apply_buffer: None,
            })
        }

        pub(crate) fn execute_command(&mut self, command: TensorCommand) -> Result<Vec<f32>> {
            self.on_owner(|worker| {
                worker.active = Some(command);
                worker.execute_active()
            })
        }

        pub(crate) fn shutdown_worker(&mut self) -> Result<()> {
            self.on_owner(|_| Ok(()))
        }

        fn quiesce(&self) -> Result<()> {
            sync_all(
                || {
                    require_fence(self.linear.quiesce())?;
                    ensure!(
                        self.linear.is_quiescent() && !self.linear.needs_quarantine(),
                        "CUDA shard is not safe to release/reuse"
                    );
                    Ok(())
                },
                || require_fence(self.ops.sync_stream()),
                || require_fence(self.ops.sync_upload_stream()),
            )
        }

        fn on_owner<T>(&mut self, work: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
            let result = catch_unwind(AssertUnwindSafe(|| work(self)));
            let result = complete_owner(result, || self.quiesce());
            self.release_active();
            result
        }

        fn panic_quiescence_worker(&self) -> PanicQuiescence {
            if self.quiesce().is_ok() {
                PanicQuiescence::Quiescent
            } else {
                PanicQuiescence::Unknown
            }
        }

        fn execute_active(&mut self) -> Result<Vec<f32>> {
            match self.active.as_ref().expect("active host payload") {
                TensorCommand::Compute { input, rows } => {
                    self.linear.execute(input, *rows).map_err(shard_error)
                }
                TensorCommand::Apply { values, rows } => {
                    ensure!(
                        *rows == self.rows
                            && values.len() == elements(*rows, self.linear.plan().out_features())?,
                        "TP Apply shape mismatch"
                    );
                    self.apply_buffer =
                        Some(self.ops.upload_f32_buffer(values).map_err(shard_error)?);
                    self.ops
                        .download_f32_buffer(self.apply_buffer.as_ref().expect("Apply buffer"))
                        .map_err(shard_error)
                }
            }
        }

        fn release_active(&mut self) {
            self.apply_buffer = None;
            self.active = None;
        }
    }

    impl ReplicaWorker<Arc<[f32]>> for LinearWorker {
        type Output = Vec<f32>;
        type Error = anyhow::Error;

        fn execute(&mut self, request: WorkRequest<Arc<[f32]>>) -> Result<Vec<f32>> {
            self.on_owner(|worker| {
                ensure!(!request.cancellation.is_requested(), "DP work cancelled");
                worker.active = Some(TensorCommand::Compute {
                    input: request.input,
                    rows: worker.rows,
                });
                worker.execute_active()
            })
        }

        fn panic_quiescence(&mut self) -> PanicQuiescence {
            self.panic_quiescence_worker()
        }

        fn shutdown(&mut self) -> Result<()> {
            self.shutdown_worker()
        }
    }

    impl ReplicaWorker<TensorWork> for LinearWorker {
        type Output = Vec<f32>;
        type Error = anyhow::Error;

        fn execute(&mut self, request: WorkRequest<TensorWork>) -> Result<Vec<f32>> {
            self.on_owner(|worker| {
                ensure!(!request.cancellation.is_requested(), "TP work cancelled");
                ensure!(
                    request.rank == worker.linear.rank()
                        && request.input.rank.local == request.rank,
                    "TP worker rank mismatch"
                );
                worker.active = Some(request.input.command);
                worker.execute_active()
            })
        }

        fn panic_quiescence(&mut self) -> PanicQuiescence {
            self.panic_quiescence_worker()
        }

        fn shutdown(&mut self) -> Result<()> {
            self.shutdown_worker()
        }
    }
}
