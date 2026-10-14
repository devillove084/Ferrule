use super::*;
use std::collections::VecDeque;

use crate::parallel::data::PanicQuiescence;
use crate::parallel::process::decoder::DecoderDevice;
use ferrule_model::transformer::expert_parallel::ExpertDispatchContext;
use ferrule_model::transformer::{
    HostRows, OperatorWaiting, RouterRoutes, RowsDType, RowsShape, UnsupportedOperator,
};

fn stats() -> Vec<NumericProcessStats> {
    [109, 42]
        .into_iter()
        .enumerate()
        .map(|(slot, owner)| NumericProcessStats {
            pid: 123 + slot as u32,
            owner,
            slot: slot as u32,
            device: DecoderDevice::Cuda { ordinal: slot + 1 },
            owned_experts: vec![(1, slot)],
            calls: 0,
            tokens: 0,
            host_payload_bytes: 128,
            host_read_bytes: 128,
            host_hits: 3,
            resident_experts: 1,
            resident_bytes: 128,
            uploads: 1,
            evictions: 0,
            pending_upload_bytes: 0,
            quarantined: false,
        })
        .collect()
}

#[test]
fn process_baseline_checks_every_immutable_field_without_freezing_compute_counters() {
    let before = stats();
    let baseline = ImmutableOwnedBaseline::processes(before.clone()).unwrap();
    let mut after = before.clone();
    after[0].calls = 9;
    after[0].tokens = 81;
    baseline.check_processes(&after).unwrap();
    for case in 0..14 {
        let mut after = before.clone();
        match case {
            0 => after[0].pid += 1,
            1 => after[0].owner += 1,
            2 => after[0].slot += 1,
            3 => after[0].device = DecoderDevice::Cuda { ordinal: 99 },
            4 => after[0].owned_experts[0].1 += 1,
            5 => after[0].host_payload_bytes += 1,
            6 => after[0].host_read_bytes += 1,
            7 => after[0].host_hits += 1,
            8 => after[0].resident_experts += 1,
            9 => after[0].resident_bytes += 1,
            10 => after[0].uploads += 1,
            11 => after[0].evictions = 1,
            12 => after[0].pending_upload_bytes = 1,
            13 => after[0].quarantined = true,
            _ => unreachable!(),
        }
        assert!(baseline.check_processes(&after).is_err(), "case {case}");
    }
    assert!(baseline.check_processes(&before[..1]).is_err());
    let mut reordered = before.clone();
    reordered.swap(0, 1);
    assert!(baseline.check_processes(&reordered).is_err());
    baseline.check_processes(&before).unwrap();
    assert!(ImmutableOwnedBaseline::processes(vec![]).is_err());
    let mut quarantined = before;
    quarantined[0].quarantined = true;
    assert!(ImmutableOwnedBaseline::processes(quarantined).is_err());
}

#[derive(Clone, Copy)]
enum Outcome {
    Ready,
    Unsupported,
    Waiting,
    Failed,
}

struct Backend {
    outcome: Outcome,
    progress: WorkProgress,
    shutdown_progress: VecDeque<WorkProgress>,
    shutdown_error: bool,
    stats_error: bool,
    stats: Vec<NumericProcessStats>,
    calls: usize,
    stats_calls: usize,
    drain_calls: usize,
    shutdown_calls: usize,
    cancellations: Vec<ExecutionTransactionId>,
    #[cfg(feature = "cuda")]
    owners: Vec<ExpertOwnerStats>,
}

impl Default for Backend {
    fn default() -> Self {
        Self {
            outcome: Outcome::Ready,
            progress: WorkProgress::Complete,
            shutdown_progress: VecDeque::new(),
            shutdown_error: false,
            stats_error: false,
            stats: stats(),
            calls: 0,
            stats_calls: 0,
            drain_calls: 0,
            shutdown_calls: 0,
            cancellations: Vec::new(),
            #[cfg(feature = "cuda")]
            owners: vec![],
        }
    }
}

fn rows() -> Rows {
    Rows::Host(
        HostRows::new(
            RowsShape::new(1, 2).unwrap(),
            RowsDType::F32,
            None,
            vec![-0.0, 1.0],
        )
        .unwrap(),
    )
}

fn context(layer: usize) -> ExpertDispatchContext {
    ExpertDispatchContext {
        transaction: ExecutionTransactionId::new(99).unwrap(),
        source_rank: ParallelRankId::new(7),
        layer,
    }
}

fn request<'a>(
    context: ExpertDispatchContext,
    input: &'a Rows,
    routes: &'a RouterRoutes,
) -> RoutedSwiGluRequest<'a> {
    RoutedSwiGluRequest {
        context,
        sequences: &[81],
        input,
        routes,
        arena: None,
    }
}

impl RoutedSwiGluExecutor for Backend {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        self.calls += 1;
        active(request.context.transaction)?;
        match self.outcome {
            Outcome::Ready => Ok(OperatorProgress::Ready(rows())),
            Outcome::Unsupported => Ok(OperatorProgress::Unsupported(UnsupportedOperator::new(
                "test",
                "no result",
            ))),
            Outcome::Waiting => Ok(OperatorProgress::Waiting(OperatorWaiting::Backend {
                operator: "test",
                reason: "pending result".into(),
            })),
            Outcome::Failed => Err(error("fenced compute failure")),
        }
    }
}

impl AttachmentBackend for Backend {
    fn drain(&mut self) -> Result<WorkProgress> {
        self.drain_calls += 1;
        Ok(self.progress)
    }
    fn shutdown(&mut self) -> Result<WorkProgress> {
        self.shutdown_calls += 1;
        if self.shutdown_error {
            return Err(error("shutdown error retains custody"));
        }
        if self.progress == WorkProgress::Unknown {
            return Ok(WorkProgress::Unknown);
        }
        Ok(self.shutdown_progress.pop_front().unwrap_or(self.progress))
    }
    fn on_error(&mut self, transaction: ExecutionTransactionId) {
        self.cancellations.push(transaction);
    }
    fn retain_unknown_shutdown(&mut self) {
        self.progress = WorkProgress::Unknown;
    }
    #[cfg(feature = "cuda")]
    fn owner_stats(&mut self) -> Result<Vec<ExpertOwnerStats>> {
        self.stats_calls += 1;
        Ok(self.owners.clone())
    }
    fn process_stats(&mut self) -> Result<Vec<NumericProcessStats>> {
        self.stats_calls += 1;
        if self.stats_error {
            return Err(error("live process stats unavailable"));
        }
        Ok(self.stats.clone())
    }
}

fn attachment() -> Attachment<Backend> {
    Attachment {
        backend: Backend::default(),
        config: ImmutableOwnedAttachmentConfig::new(
            ParallelRankId::new(7),
            2,
            Duration::from_secs(1),
        )
        .unwrap(),
        baseline: ImmutableOwnedBaseline::processes(stats()).unwrap(),
    }
}

#[test]
fn source_gate_and_last_ready_layer_preserve_caller_authority_and_stat_boundary() {
    let input = rows();
    let routes = RouterRoutes::new(1, 1, vec![0], vec![1.0]).unwrap();
    let mut attachment = attachment();
    let mut callback_ids = vec![];
    let mut active = |id| {
        callback_ids.push(id);
        Ok(())
    };
    for source in [ParallelRankId::new(109), ParallelRankId::new(999)] {
        let mut context = context(2);
        context.source_rank = source;
        assert!(
            attachment
                .routed_swiglu(request(context, &input, &routes), &mut active)
                .is_err()
        );
    }
    assert_eq!(attachment.backend.calls, 0);
    assert!(callback_ids.is_empty());
    let mut active = |id| {
        callback_ids.push(id);
        Ok(())
    };
    attachment
        .routed_swiglu(request(context(1), &input, &routes), &mut active)
        .unwrap();
    assert_eq!(attachment.backend.stats_calls, 0);
    for outcome in [Outcome::Unsupported, Outcome::Waiting] {
        attachment.backend.outcome = outcome;
        attachment
            .routed_swiglu(request(context(2), &input, &routes), &mut active)
            .unwrap();
        assert_eq!(attachment.backend.stats_calls, 0);
    }
    attachment.backend.outcome = Outcome::Ready;
    let OperatorProgress::Ready(output) = attachment
        .routed_swiglu(request(context(2), &input, &routes), &mut active)
        .unwrap()
    else {
        panic!("expected Ready")
    };
    assert_eq!(
        output.host().unwrap().values()[0].to_bits(),
        (-0.0f32).to_bits()
    );
    assert_eq!(attachment.backend.stats_calls, 1);
    assert_eq!(callback_ids, vec![context(2).transaction; 4]);
    attachment.backend.stats[0].host_hits += 1;
    assert!(
        attachment
            .routed_swiglu(request(context(2), &input, &routes), &mut |_| Ok(()))
            .is_err()
    );
    attachment.backend.stats = stats();
    attachment.backend.stats_error = true;
    assert!(
        attachment
            .routed_swiglu(request(context(2), &input, &routes), &mut |_| Ok(()))
            .is_err()
    );
}

#[test]
fn failed_execution_preserves_typed_unknown_and_delegates_drain_and_cancellation() {
    let input = rows();
    let routes = RouterRoutes::new(1, 1, vec![0], vec![1.0]).unwrap();
    for progress in [
        WorkProgress::Complete,
        WorkProgress::Pending,
        WorkProgress::Unknown,
    ] {
        let mut attachment = attachment();
        attachment.backend.outcome = Outcome::Failed;
        attachment.backend.progress = progress;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            attachment.routed_swiglu(request(context(2), &input, &routes), &mut |_| Ok(()))
        }));
        if progress == WorkProgress::Unknown {
            assert_eq!(
                result.unwrap_err().downcast_ref(),
                Some(&PanicQuiescence::Unknown)
            );
        } else {
            assert!(result.unwrap().is_err());
        }
        assert_eq!(attachment.backend.drain_calls, 1);
        assert_eq!(attachment.backend.stats_calls, 0);
        assert_eq!(attachment.drain().unwrap(), progress);
        attachment.on_error(context(2).transaction);
        assert_eq!(
            attachment.backend.cancellations,
            vec![context(2).transaction]
        );
    }
}

#[test]
fn shutdown_joins_pending_owners_and_never_turns_timeout_or_unknown_into_complete() {
    assert!(
        ImmutableOwnedAttachmentConfig::new(ParallelRankId::new(7), 2, Duration::ZERO).is_err()
    );
    assert!(ImmutableOwnedAttachmentConfig::new(ParallelRankId::new(7), 2, Duration::MAX).is_err());
    let mut attachment = attachment();
    attachment.backend.shutdown_progress = [WorkProgress::Pending, WorkProgress::Complete].into();
    assert_eq!(attachment.shutdown().unwrap(), WorkProgress::Complete);
    assert_eq!(attachment.backend.shutdown_calls, 2);
    attachment.backend.progress = WorkProgress::Unknown;
    assert_eq!(attachment.shutdown().unwrap(), WorkProgress::Unknown);
    assert_eq!(attachment.backend.shutdown_calls, 3);
    let mut timed_out = self::attachment();
    timed_out.config.timeout = Duration::from_millis(1);
    timed_out.backend.progress = WorkProgress::Pending;
    assert_eq!(timed_out.shutdown().unwrap(), WorkProgress::Unknown);
    assert_eq!(timed_out.drain().unwrap(), WorkProgress::Unknown);
    timed_out
        .backend
        .shutdown_progress
        .push_back(WorkProgress::Complete);
    assert_eq!(timed_out.shutdown().unwrap(), WorkProgress::Unknown);
    attachment.backend.shutdown_error = true;
    assert!(attachment.shutdown().is_err());
}

#[cfg(feature = "cuda")]
fn owners() -> Vec<ExpertOwnerStats> {
    stats()
        .into_iter()
        .map(|stats| ExpertOwnerStats {
            rank: super::super::ExpertRank {
                slot: ParallelRankId::new(stats.slot),
                owner: ParallelRankId::new(stats.owner),
            },
            thread: std::thread::current().id(),
            owned_experts: vec![ferrule_model::moe::ExpertId::new(1, stats.slot as usize)],
            calls: 0,
            tokens: 0,
            last_context: None,
            cuda_cache: Some(ferrule_model::transformer::ExpertCacheStats {
                resident_experts: 1,
                resident_bytes: 128,
                uploads: 1,
                ..Default::default()
            }),
        })
        .collect()
}

#[cfg(feature = "cuda")]
#[test]
fn thread_baseline_checks_residency_and_host_hits_without_freezing_activity_stats() {
    let before = owners();
    let mut after = before.clone();
    after[0].calls = 9;
    after[0].tokens = 81;
    after[0].last_context = Some(context(2));
    after[0].cuda_cache.as_mut().unwrap().hits += 1;
    check_owner_stats(&before, &after).unwrap();
    for case in 0..10 {
        let mut after = before.clone();
        match case {
            0 => after[0].rank.owner = ParallelRankId::new(999),
            1 => after[0].rank.slot = ParallelRankId::new(999),
            2 => after[0].owned_experts.clear(),
            3 => after[0].cuda_cache.as_mut().unwrap().uploads += 1,
            4 => after[0].cuda_cache.as_mut().unwrap().evictions = 1,
            5 => after[0].cuda_cache.as_mut().unwrap().resident_bytes += 1,
            6 => after[0].cuda_cache.as_mut().unwrap().resident_experts += 1,
            7 => after[0].cuda_cache.as_mut().unwrap().pending_upload_bytes = 1,
            8 => after[0].cuda_cache.as_mut().unwrap().quarantined = true,
            9 => after[0].cuda_cache = None,
            _ => unreachable!(),
        }
        assert!(check_owner_stats(&before, &after).is_err(), "case {case}");
    }
    assert!(check_owner_stats(&before, &before[..1]).is_err());
    assert!(ImmutableOwnedBaseline::threads(before.clone(), vec![1], None, 0).is_err());
    let baseline = ImmutableOwnedBaseline::threads(before.clone(), vec![1, 2], None, 0).unwrap();
    assert!(baseline.check_processes(&stats()).is_err());
    let mut attachment = attachment();
    attachment.baseline = baseline;
    attachment.backend.owners = before;
    attachment.check_baseline(context(2)).unwrap();
    attachment.backend.owners[0]
        .cuda_cache
        .as_mut()
        .unwrap()
        .quarantined = true;
    assert!(attachment.check_baseline(context(2)).is_err());
    check_host_hits(3, 3).unwrap();
    assert!(check_host_hits(3, 4).is_err());
}

#[cfg(feature = "cuda")]
#[test]
fn thread_baseline_retains_shared_host_identity_and_rejects_rematerialization() {
    use crate::parallel::expert::{ExpertGroup, ExpertRank, ExpertRankWorker};
    use ferrule_model::transformer::StateDictMaterializer;
    use ferrule_model::transformer::expert_parallel::{ExpertDispatchLimits, ExpertPlacement};
    use ferrule_model::transformer::host_experts::{HostExpertCacheOptions, HostExpertWarmPlan};
    use std::sync::Arc;

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("numeric.bin");
    let resources = crate::parallel::expert::numeric_test_fixture::bind(&path, 4, true, true);
    let host = Arc::new(
        HostExpertWarmPlan::new(
            resources.state_dict(),
            HostExpertCacheOptions {
                max_experts: 8,
                max_bytes: 8 << 20,
                ..Default::default()
            },
            4096,
        )
        .unwrap()
        .warm(|_| {})
        .unwrap(),
    );
    let ready_hits = host.hits();
    let baseline =
        ImmutableOwnedBaseline::threads(owners(), vec![1, 2], Some(Arc::clone(&host)), ready_hits)
            .unwrap();
    let Baseline::Threads {
        host: Some(retained),
        ..
    } = &baseline.kind
    else {
        panic!("expected shared host image");
    };
    assert!(Arc::ptr_eq(retained, &host));
    let mut attachment = attachment();
    attachment.baseline = baseline;
    attachment.backend.owners = owners();
    attachment.check_baseline(context(2)).unwrap();

    let members = vec![ParallelRankId::new(109), ParallelRankId::new(42)];
    let layers = 0..resources.spec().layers().len();
    let placement = ExpertPlacement::new(layers.clone().flat_map(|layer| {
        (0..4).map(move |expert| {
            (
                layer,
                expert,
                ParallelRankId::new(if expert % 2 == 0 { 109 } else { 42 }),
            )
        })
    }))
    .unwrap();
    let group = ExpertGroup {
        source_rank: ParallelRankId::new(7),
        members,
        layers,
        placement,
        limits: ExpertDispatchLimits {
            max_tokens: 16,
            max_bytes: 16 * 8 * 4,
        },
    };
    let materializer = StateDictMaterializer::new(4096).unwrap();
    materializer.attach_host_experts(&host).unwrap();
    let _worker = ExpertRankWorker::prepare_with_materializer(
        &resources,
        ExpertRank {
            slot: ParallelRankId::new(0),
            owner: ParallelRankId::new(109),
        },
        &group,
        &materializer,
    )
    .unwrap();
    assert!(host.hits() > ready_hits);
    let error = attachment.check_baseline(context(2)).unwrap_err();
    assert!(error.to_string().contains("rematerialized host experts"));
}
