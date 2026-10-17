//! CPU contracts exercise the production source hold and provider ledger, not DMA.
use super::*;
use crate::transformer::expert_reads::ExpertReadHostCharge;
use crate::transformer::expert_reads::tests::{Fixture, IMAGE, options};
use crate::transformer::standard::{ExpertSemantics, StateDictExpertProvider};
use crate::transformer::{ExpertReadSchedule, StateDictMaterializer};

fn assert_baseline_only(charge: &ExpertReadHostCharge) {
    assert_eq!(charge.pinned_upload_bytes, 0);
    assert_eq!(charge.upload_ticket_metadata_bytes, 0);
    assert_eq!(charge.upload_policy_headroom_bytes, 0);
    assert_eq!(charge.upload_reserved_packed_bytes, 0);
    assert_eq!(charge.unknown_upload_bytes, 0);
}

#[test]
fn default_provider_group_capability_does_not_materialize() {
    struct Legacy;
    impl ExpertProvider for Legacy {
        fn expert(&mut self, _: usize, _: usize) -> Result<ExpertAvailability> {
            panic!("capability must not read payloads")
        }
    }
    assert!(!Legacy.supports_grouped_expert_upload());
}

#[test]
fn exact_cap_group_source_aliases_use_only_existing_charge_and_release_once() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let f = Fixture::new(1);
        let semantics = f
            .metadata
            .iter()
            .map(ExpertSemantics::read_test_fixture)
            .collect::<Vec<_>>();
        let mut materializer = StateDictMaterializer::new(128).unwrap();
        let limits = options(schedule);
        materializer.configure_expert_read_pipeline(limits).unwrap();
        let mut provider = StateDictExpertProvider {
            layer: 0,
            bindings: &semantics,
            materializer: &materializer,
            reads: None,
        };
        assert!(!provider.supports_grouped_expert_upload());
        provider
            .preflight_expert_reads(IMAGE, &[&f.metadata[0]])
            .unwrap();
        assert!(provider.supports_grouped_expert_upload());
        assert_eq!(materializer.checkpoint_read_counters().bytes_read, 0);
        let ExpertAvailability::Ready(expert) = provider.expert(0, 0).unwrap() else {
            panic!()
        };
        let alias = expert
            .gate()
            .numeric_fp8()
            .unwrap()
            .validated_payload()
            .clone();
        let mut state = State::default();
        state.begin(Arc::clone(&expert)).unwrap();
        assert!(state.begin(Arc::clone(&expert)).is_err());
        drop(expert);
        let observed = materializer.expert_read_pipeline_observation();
        assert_baseline_only(&observed.current_charge);
        assert_eq!(
            observed.current_charge.host_charged_bytes,
            limits.max_host_bytes
        );
        assert_eq!(
            observed.stats.peak_host_charged_bytes,
            limits.max_host_bytes
        );
        assert_eq!(observed.current_charge.packed_bytes, 18);
        assert_eq!(observed.current_charge.artifact_bytes, 18);
        assert!(provider.expert(0, 0).is_err());
        // Models only the successful backend-return seam, never a CUDA proof.
        state.finish();
        assert!(!state.has_sources());
        assert!(
            provider.expert(0, 0).is_err(),
            "a remaining payload alias still owns source credit"
        );
        drop(alias);
        provider.lookahead_expert_read(&f.metadata[0], None, false);
        let observed = materializer.expert_read_pipeline_observation();
        assert_baseline_only(&observed.current_charge);
        assert_eq!(observed.current_charge.packed_bytes, 0);
        assert_eq!(observed.current_charge.artifact_bytes, 0);
        assert!(observed.stats.peak_host_charged_bytes <= limits.max_host_bytes);
        drop(provider);
        let mut next = StateDictExpertProvider {
            layer: 0,
            bindings: &semantics,
            materializer: &materializer,
            reads: None,
        };
        next.preflight_expert_reads(IMAGE, &[&f.metadata[0]])
            .unwrap();
        assert_eq!(materializer.expert_read_pipeline_stats().workers_created, 1);
        assert_eq!(materializer.expert_read_pipeline_stats().worker_reuses, 1);
    }
}

#[test]
fn one_byte_below_baseline_cap_rejects_before_any_payload_read() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let f = Fixture::new(1);
        let semantics = f
            .metadata
            .iter()
            .map(ExpertSemantics::read_test_fixture)
            .collect::<Vec<_>>();
        let mut materializer = StateDictMaterializer::new(128).unwrap();
        let mut limits = options(schedule);
        limits.max_host_bytes -= 1;
        materializer.configure_expert_read_pipeline(limits).unwrap();
        let mut provider = StateDictExpertProvider {
            layer: 0,
            bindings: &semantics,
            materializer: &materializer,
            reads: None,
        };
        assert!(
            provider
                .preflight_expert_reads(IMAGE, &[&f.metadata[0]])
                .is_err()
        );
        assert_eq!(materializer.checkpoint_read_counters().bytes_read, 0);
        assert!(!provider.supports_grouped_expert_upload());
    }
}

#[test]
fn unresolved_group_drop_or_caller_unwind_never_refunds_source_credit() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        for unwind in [false, true] {
            let f = Fixture::new(1);
            let semantics = f
                .metadata
                .iter()
                .map(ExpertSemantics::read_test_fixture)
                .collect::<Vec<_>>();
            let mut materializer = StateDictMaterializer::new(128).unwrap();
            materializer
                .configure_expert_read_pipeline(options(schedule))
                .unwrap();
            let mut provider = StateDictExpertProvider {
                layer: 0,
                bindings: &semantics,
                materializer: &materializer,
                reads: None,
            };
            provider
                .preflight_expert_reads(IMAGE, &[&f.metadata[0]])
                .unwrap();
            let ExpertAvailability::Ready(expert) = provider.expert(0, 0).unwrap() else {
                panic!()
            };
            let witness = expert
                .gate()
                .numeric_fp8()
                .unwrap()
                .validated_payload()
                .allocation_liveness();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut state = State::default();
                state.begin(expert).unwrap();
                if unwind {
                    panic!("caller unwinds while source custody is unresolved");
                }
            }));
            assert_eq!(result.is_err(), unwind);
            assert!(witness.is_live());
            assert!(provider.expert(0, 0).is_err());
            drop(provider);
            let observed = materializer.expert_read_pipeline_observation();
            assert_eq!(observed.current_charge.artifact_bytes, 18);
            assert_baseline_only(&observed.current_charge);
            assert!(observed.current_charge.host_charged_bytes >= 18);
            assert!(
                materializer
                    .expert_read_sequence(IMAGE, &[&f.metadata[0]])
                    .is_err()
            );
            // Intentionally retained by the real production Drop. A later host
            // read drain is not backend completion and cannot refund this hold.
            assert!(witness.is_live());
        }
    }
}

#[test]
#[ignore = "GPU-owner authorization only; actual owner capability and policy isolation"]
fn real_owner_group_policy_rejects_unsupported_duplicate_and_revert() {
    use crate::transformer::{
        ExpertCacheLimits, ExpertCachePolicy, ExpertTransferControl, ExpertTransferPolicy,
    };
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use std::rc::Rc;
    let f = Fixture::new(1);
    let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
    let mut unsupported = CudaStandardDecoderOperators::new_numeric_fp8(
        Rc::clone(&ops),
        f.bound.parameters(),
        ExpertCachePolicy::KeepAll,
        1024,
    )
    .unwrap();
    assert_eq!(
        unsupported.expert_transfer_policy(),
        ExpertTransferPolicy::Synchronous
    );
    assert!(
        unsupported
            .set_expert_transfer_policy(ExpertTransferPolicy::GroupedSynchronous)
            .is_err()
    );
    assert_eq!(
        unsupported.expert_transfer_policy(),
        ExpertTransferPolicy::Synchronous
    );
    let mut owner = CudaStandardDecoderOperators::new_numeric_fp8(
        ops,
        f.bound.parameters(),
        ExpertCachePolicy::Bounded(ExpertCacheLimits {
            max_experts: 1,
            max_bytes: 8192,
        }),
        1024,
    )
    .unwrap();
    owner
        .set_expert_transfer_policy(ExpertTransferPolicy::GroupedSynchronous)
        .unwrap();
    for policy in [
        ExpertTransferPolicy::Synchronous,
        ExpertTransferPolicy::GroupedSynchronous,
    ] {
        assert!(owner.set_expert_transfer_policy(policy).is_err());
        assert_eq!(
            owner.expert_transfer_policy(),
            ExpertTransferPolicy::GroupedSynchronous
        );
    }
    assert_eq!(owner.expert_cache_stats().unwrap().uploads, 0);
}

#[test]
#[ignore = "GPU-owner authorization only; actual whole-triple admission before provider reads"]
fn real_owner_group_admits_all_six_destinations_plus_scratch_before_payload_read() {
    use crate::transformer::{
        ExpertCacheLimits, ExpertCachePolicy, ExpertTransferControl, ExpertTransferPolicy,
        HostRows, Rows, RowsDType, RowsShape,
    };
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use std::rc::Rc;
    let f = Fixture::new(1);
    let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
    let sizing = CudaStandardDecoderOperators::new_numeric_fp8(
        Rc::clone(&ops),
        f.bound.parameters(),
        ExpertCachePolicy::Bounded(ExpertCacheLimits {
            max_experts: 1,
            max_bytes: 8192,
        }),
        1024,
    )
    .unwrap();
    let (_, bytes, scratch) = sizing.metadata_plan(&f.metadata[0], 0, 0, 1, 2).unwrap();
    assert_eq!(bytes, 18, "three weight and three scale destinations");
    let total = crate::transformer::SwiGluScratchPlan::reserved_bytes(scratch, 0, 1024).unwrap();
    drop(sizing);
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let mut owner = CudaStandardDecoderOperators::new_numeric_fp8(
            Rc::clone(&ops),
            f.bound.parameters(),
            ExpertCachePolicy::Bounded(ExpertCacheLimits {
                max_experts: 1,
                max_bytes: total + bytes - 1,
            }),
            1024,
        )
        .unwrap();
        owner
            .set_expert_transfer_policy(ExpertTransferPolicy::GroupedSynchronous)
            .unwrap();
        let semantics = f
            .metadata
            .iter()
            .map(ExpertSemantics::read_test_fixture)
            .collect::<Vec<_>>();
        let mut materializer = StateDictMaterializer::new(128).unwrap();
        materializer
            .configure_expert_read_pipeline(options(schedule))
            .unwrap();
        let mut provider = StateDictExpertProvider {
            layer: 0,
            bindings: &semantics,
            materializer: &materializer,
            reads: None,
        };
        provider
            .preflight_expert_reads(owner.image_generation(), &[&f.metadata[0]])
            .unwrap();
        let before = materializer.checkpoint_read_counters();
        let input = Rows::Host(
            HostRows::new(
                RowsShape::new(1, 2).unwrap(),
                RowsDType::F32,
                None,
                vec![1., 2.],
            )
            .unwrap(),
        );
        assert!(
            owner
                .metadata_swiglu_rows(&f.metadata[0], &mut provider, &input, None)
                .is_err()
        );
        assert_eq!(materializer.checkpoint_read_counters(), before);
        assert_eq!(owner.expert_cache_stats().unwrap().uploads, 0);
        assert!(!owner.bindings.has_metadata_expert(&f.metadata[0]).unwrap());
        assert!(!owner.grouped_upload.as_ref().unwrap().has_sources());
    }
}
