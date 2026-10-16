//! Exercises the real StateDictExpertProvider reservation/materialization bridge.
use super::*;
use crate::transformer::expert_reads::tests::{Fixture, IMAGE, options};
use crate::transformer::{ExpertMetadata, ExpertReadSchedule, StateDictMaterializer};

#[test]
fn default_provider_capability_is_disabled_without_materialization() {
    struct Legacy;
    impl ExpertProvider for Legacy {
        fn expert(&mut self, _: usize, _: usize) -> Result<ExpertAvailability> {
            panic!("must not materialize")
        }
    }
    let f = Fixture::new(1);
    assert!(
        Legacy
            .reserve_expert_upload(&f.metadata[0], 512, 4096)
            .unwrap()
            .is_none()
    );
}

#[test]
fn state_dict_provider_transfers_existing_read_and_aliases_and_preserves_idle_worker() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let f = Fixture::new(1);
        let semantics = f
            .metadata
            .iter()
            .map(ExpertSemantics::read_test_fixture)
            .collect::<Vec<_>>();
        let mut materializer = StateDictMaterializer::new(128).unwrap();
        let mut limits = options(schedule);
        limits.max_host_bytes += 8192;
        materializer.configure_expert_read_pipeline(limits).unwrap();
        let mut provider = StateDictExpertProvider {
            layer: 0,
            bindings: &semantics,
            materializer: &materializer,
            reads: None,
        };
        provider
            .preflight_expert_reads(IMAGE, &[&f.metadata[0]])
            .unwrap();
        let before = materializer.checkpoint_read_counters();
        let mut reservation = provider
            .reserve_expert_upload(&f.metadata[0], 512, 4096)
            .unwrap()
            .unwrap();
        assert_eq!(materializer.checkpoint_read_counters(), before);
        let ExpertAvailability::Ready(expert) = provider.expert(0, 0).unwrap() else {
            panic!()
        };
        provider.transfer_expert_upload(&mut reservation).unwrap();
        assert!(provider.expert(0, 0).is_err());
        drop(expert);
        reservation.prove_completion();
        reservation.release().unwrap();
        drop(provider);
        let observed = materializer.expert_read_pipeline_observation();
        assert_eq!(observed.status, "idle_retained");
        assert_eq!(observed.current_charge.packed_bytes, 0);
        assert_eq!(observed.current_charge.pinned_upload_bytes, 0);
        assert_eq!(observed.stats.workers_created, 1);
    }
}

const PAYLOAD: u64 = 18;
const TICKET_METADATA: usize = 512;
const POLICY_HEADROOM: usize = 4096;

fn bounded_materializer(schedule: ExpertReadSchedule) -> StateDictMaterializer {
    let mut materializer = StateDictMaterializer::new(128).unwrap();
    let mut limits = options(schedule);
    limits.max_host_bytes += 8192;
    materializer.configure_expert_read_pipeline(limits).unwrap();
    materializer
}

fn assert_unknown_upload_keeps_admission(
    materializer: &StateDictMaterializer,
    metadata: &ExpertMetadata,
) {
    let observed = materializer.expert_read_pipeline_observation();
    let charge = observed.current_charge;
    assert!(observed.worker_unknown);
    assert_eq!(charge.artifact_bytes, PAYLOAD);
    assert_eq!(charge.pinned_upload_bytes, PAYLOAD);
    assert_eq!(charge.upload_ticket_metadata_bytes, TICKET_METADATA as u64);
    assert_eq!(charge.upload_policy_headroom_bytes, POLICY_HEADROOM as u64);
    assert_eq!(
        charge.unknown_upload_bytes,
        2 * PAYLOAD + TICKET_METADATA as u64 + POLICY_HEADROOM as u64
    );
    assert_eq!(
        charge.packed_bytes + charge.upload_reserved_packed_bytes,
        PAYLOAD
    );
    let limits = observed.limits.unwrap();
    assert_eq!(
        charge.host_charged_bytes,
        3 * PAYLOAD
            + TICKET_METADATA as u64
            + POLICY_HEADROOM as u64
            + limits.max_metadata_bytes
            + limits.worker_stack_bytes as u64
    );
    assert!(charge.host_charged_bytes <= limits.max_host_bytes);
    assert!(
        materializer
            .expert_read_sequence(IMAGE, &[metadata])
            .is_err()
    );
    assert_eq!(materializer.expert_read_pipeline_stats().workers_created, 1);
    assert_eq!(materializer.expert_read_pipeline_stats().worker_reuses, 0);
}

fn invalid_down_scale_with_current_source(f: &Fixture) -> ExpertMetadata {
    use crate::transformer::{ExactNameMapper, NameMapping, StateDictBinder};
    use std::io::{Seek, SeekFrom, Write};
    let scale = f.metadata[0].parameters()[2].scale().unwrap().slice();
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(&scale.path)
        .unwrap();
    file.seek(SeekFrom::Start(scale.offset)).unwrap();
    file.write_all(&half::bf16::ZERO.to_bits().to_le_bytes())
        .unwrap();
    drop(file);
    // Rebind the deliberately invalid bytes so the real read succeeds with a
    // current source identity and failure occurs in numeric materialization.
    let mut mapper = ExactNameMapper::new();
    let mut slices = Vec::new();
    for parameter in f.bound.parameters() {
        let weight = parameter.weight().slice();
        let scale = parameter.scale().unwrap().slice();
        mapper
            .insert(
                weight.name.clone(),
                NameMapping::weight(parameter.path().clone()),
            )
            .unwrap();
        mapper
            .insert(
                scale.name.clone(),
                NameMapping::scale(parameter.path().clone()),
            )
            .unwrap();
        slices.extend([weight.clone(), scale.clone()]);
    }
    let rebound = StateDictBinder::new(f.bound.schema(), &mapper)
        .bind_slices(slices)
        .unwrap();
    ExpertMetadata::new(
        0,
        0,
        rebound.parameters().to_vec().try_into().unwrap(),
        None,
    )
    .unwrap()
}

#[test]
fn source_change_after_reservation_never_refunds_or_replaces_the_real_provider() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let f = Fixture::new(1);
        let semantics = [ExpertSemantics::read_test_fixture(&f.metadata[0])];
        let materializer = bounded_materializer(schedule);
        let mut provider = StateDictExpertProvider {
            layer: 0,
            bindings: &semantics,
            materializer: &materializer,
            reads: None,
        };
        provider
            .preflight_expert_reads(IMAGE, &[&f.metadata[0]])
            .unwrap();
        let handoff = provider.reads.as_ref().unwrap().observation_test_handoff();
        let before = materializer.checkpoint_read_counters();
        let mut reservation = provider
            .reserve_expert_upload(&f.metadata[0], TICKET_METADATA, POLICY_HEADROOM)
            .unwrap()
            .unwrap();
        assert_eq!(materializer.checkpoint_read_counters(), before);
        let charged = materializer
            .expert_read_pipeline_observation()
            .current_charge
            .host_charged_bytes;
        let path = &f.metadata[0].parameters()[0].weight().slice().path;
        let replacement = path.with_extension("replaced");
        std::fs::write(&replacement, std::fs::read(path).unwrap()).unwrap();
        std::fs::rename(replacement, path).unwrap();
        let error = provider.expert(0, 0).err().expect("provider must fail");
        assert!(error.to_string().contains("source"));
        let after = materializer.checkpoint_read_counters();
        assert_eq!(after.read_calls, before.read_calls);
        assert_eq!(after.bytes_read, before.bytes_read);
        assert!(after.source_checks > before.source_checks);
        assert_eq!(materializer.expert_read_pipeline_stats().demand_failures, 1);
        assert_eq!(materializer.expert_read_pipeline_stats().verified_reads, 0);
        assert!(provider.transfer_expert_upload(&mut reservation).is_err());
        reservation.prove_completion();
        assert!(
            reservation.release().is_err(),
            "proof cannot replace missing custody transfer"
        );
        drop(error);
        assert_eq!(
            materializer
                .expert_read_pipeline_observation()
                .current_charge
                .host_charged_bytes,
            charged
        );
        drop(reservation);
        drop(provider);
        assert_unknown_upload_keeps_admission(&materializer, &f.metadata[0]);
        let before = materializer.checkpoint_read_counters();
        assert!(
            materializer
                .expert_read_sequence(IMAGE, &[&f.metadata[0]])
                .is_err()
        );
        assert_eq!(materializer.checkpoint_read_counters(), before);
        drop(materializer);
        assert!(
            handoff.observe_worker().joined,
            "host worker teardown is not upload proof"
        );
    }
}

#[test]
fn real_numeric_materialization_error_after_reserve_cannot_be_released_as_success() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let f = Fixture::new(1);
        let metadata = invalid_down_scale_with_current_source(&f);
        let semantics = [ExpertSemantics::read_test_fixture(&metadata)];
        let materializer = bounded_materializer(schedule);
        let mut provider = StateDictExpertProvider {
            layer: 0,
            bindings: &semantics,
            materializer: &materializer,
            reads: None,
        };
        provider
            .preflight_expert_reads(IMAGE, &[&metadata])
            .unwrap();
        let handoff = provider.reads.as_ref().unwrap().observation_test_handoff();
        let mut reservation = provider
            .reserve_expert_upload(&metadata, TICKET_METADATA, POLICY_HEADROOM)
            .unwrap()
            .unwrap();
        assert_eq!(materializer.checkpoint_read_counters().bytes_read, 0);
        let charged = materializer
            .expert_read_pipeline_observation()
            .current_charge
            .host_charged_bytes;
        let error = provider.expert(0, 0).err().expect("provider must fail");
        assert!(error.to_string().contains("strictly positive"));
        let stats = materializer.expert_read_pipeline_stats();
        assert_eq!(stats.verified_reads, 1);
        assert_eq!(stats.demand_failures, 0);
        assert_eq!(stats.materialization_failures, 1);
        assert_eq!(materializer.checkpoint_read_counters().bytes_read, PAYLOAD);
        assert!(provider.transfer_expert_upload(&mut reservation).is_err());
        reservation.prove_completion();
        assert!(reservation.release().is_err());
        drop(error);
        assert_eq!(
            materializer
                .expert_read_pipeline_observation()
                .current_charge
                .host_charged_bytes,
            charged
        );
        drop(reservation);
        drop(provider);
        assert_unknown_upload_keeps_admission(&materializer, &metadata);
        drop(materializer);
        assert!(handoff.observe_worker().joined);
    }
}

#[test]
fn real_materialization_error_unwind_drops_unresolved_reservation_fail_closed() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let f = Fixture::new(1);
        let metadata = invalid_down_scale_with_current_source(&f);
        let semantics = [ExpertSemantics::read_test_fixture(&metadata)];
        let materializer = bounded_materializer(schedule);
        let mut provider = StateDictExpertProvider {
            layer: 0,
            bindings: &semantics,
            materializer: &materializer,
            reads: None,
        };
        provider
            .preflight_expert_reads(IMAGE, &[&metadata])
            .unwrap();
        let handoff = provider.reads.as_ref().unwrap().observation_test_handoff();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _reservation = provider
                .reserve_expert_upload(&metadata, TICKET_METADATA, POLICY_HEADROOM)
                .unwrap()
                .unwrap();
            // Panic propagation of a real provider validation error, not a
            // production fault hook or substitute upload orchestration.
            provider.expert(0, 0).unwrap();
        }));
        assert!(result.is_err());
        assert_eq!(
            materializer
                .expert_read_pipeline_stats()
                .materialization_failures,
            1
        );
        assert_eq!(materializer.expert_read_pipeline_stats().verified_reads, 1);
        drop(provider);
        assert_unknown_upload_keeps_admission(&materializer, &metadata);
        drop(materializer);
        assert!(handoff.observe_worker().joined);
    }
}

#[test]
fn caller_unwind_at_actual_materialize_and_transfer_boundaries_never_heals_unknown() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        for transferred in [false, true] {
            let f = Fixture::new(1);
            let semantics = [ExpertSemantics::read_test_fixture(&f.metadata[0])];
            let materializer = bounded_materializer(schedule);
            let mut provider = StateDictExpertProvider {
                layer: 0,
                bindings: &semantics,
                materializer: &materializer,
                reads: None,
            };
            provider
                .preflight_expert_reads(IMAGE, &[&f.metadata[0]])
                .unwrap();
            let handoff = provider.reads.as_ref().unwrap().observation_test_handoff();
            let mut alias = None;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut reservation = provider
                    .reserve_expert_upload(&f.metadata[0], TICKET_METADATA, POLICY_HEADROOM)
                    .unwrap()
                    .unwrap();
                let ExpertAvailability::Ready(expert) = provider.expert(0, 0).unwrap() else {
                    panic!("fixture must be ready")
                };
                alias = Some(
                    expert
                        .down()
                        .numeric_fp8()
                        .unwrap()
                        .validated_payload()
                        .clone(),
                );
                if transferred {
                    provider.transfer_expert_upload(&mut reservation).unwrap();
                }
                // Deliberate caller unwind at the two actual ownership boundaries;
                // this does not claim an injected panic inside either helper.
                panic!("caller unwind with live reserved expert sources");
            }));
            assert!(result.is_err());
            let witness = alias.as_ref().unwrap().allocation_liveness();
            assert!(witness.is_live());
            assert!(provider.expert(0, 0).is_err());
            drop(provider);
            assert_unknown_upload_keeps_admission(&materializer, &f.metadata[0]);
            let charged = materializer
                .expert_read_pipeline_observation()
                .current_charge
                .host_charged_bytes;
            drop(alias);
            assert!(!witness.is_live());
            assert_unknown_upload_keeps_admission(&materializer, &f.metadata[0]);
            assert_eq!(
                materializer
                    .expert_read_pipeline_observation()
                    .current_charge
                    .host_charged_bytes,
                charged
            );
            drop(witness);
            drop(materializer);
            assert!(handoff.observe_worker().joined);
        }
    }
}

#[test]
fn foreign_transfer_error_unwind_quarantines_only_the_reservation_owner() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let f = Fixture::new(1);
        let semantics = [ExpertSemantics::read_test_fixture(&f.metadata[0])];
        let a = bounded_materializer(schedule);
        let b = bounded_materializer(schedule);
        let mut provider_a = StateDictExpertProvider {
            layer: 0,
            bindings: &semantics,
            materializer: &a,
            reads: None,
        };
        let mut provider_b = StateDictExpertProvider {
            layer: 0,
            bindings: &semantics,
            materializer: &b,
            reads: None,
        };
        provider_a
            .preflight_expert_reads(IMAGE, &[&f.metadata[0]])
            .unwrap();
        provider_b
            .preflight_expert_reads(IMAGE, &[&f.metadata[0]])
            .unwrap();
        let handoff_b = provider_b
            .reads
            .as_ref()
            .unwrap()
            .observation_test_handoff();
        let mut reservation_a = provider_a
            .reserve_expert_upload(&f.metadata[0], TICKET_METADATA, POLICY_HEADROOM)
            .unwrap()
            .unwrap();
        let mut reservation_b = provider_b
            .reserve_expert_upload(&f.metadata[0], TICKET_METADATA, POLICY_HEADROOM)
            .unwrap()
            .unwrap();
        let expert_a = provider_a.expert(0, 0).unwrap();
        let expert_b = provider_b.expert(0, 0).unwrap();
        let before_a = a.expert_read_pipeline_observation();
        assert!(
            provider_a
                .transfer_expert_upload(&mut reservation_b)
                .is_err()
        );
        assert_eq!(a.expert_read_pipeline_observation(), before_a);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut foreign = reservation_b;
            provider_a.transfer_expert_upload(&mut foreign).unwrap();
        }));
        assert!(result.is_err());
        assert_eq!(a.expert_read_pipeline_observation(), before_a);
        provider_a
            .transfer_expert_upload(&mut reservation_a)
            .unwrap();
        drop(expert_a);
        reservation_a.prove_completion();
        reservation_a.release().unwrap();
        assert!(!a.expert_read_pipeline_observation().worker_unknown);
        drop(expert_b);
        drop(provider_b);
        assert_unknown_upload_keeps_admission(&b, &f.metadata[0]);
        drop(provider_a);
        drop(
            a.expert_read_sequence(IMAGE, &[&f.metadata[0]])
                .unwrap()
                .unwrap(),
        );
        assert_eq!(a.expert_read_pipeline_stats().workers_created, 1);
        assert_eq!(a.expert_read_pipeline_stats().worker_reuses, 1);
        drop(b);
        assert!(handoff_b.observe_worker().joined);
    }
}
