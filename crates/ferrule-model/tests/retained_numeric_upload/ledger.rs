//! CPU tests of the production read-owner reservation and custody transfer.
use super::tests::{Fixture, IMAGE, options};
use super::*;
use crate::transformer::{PreparedLinear, StateDictMaterializer};

const TICKET: usize = 512;
const HEADROOM: usize = 4096;
const P: u64 = 18;

fn setup(schedule: ExpertReadSchedule, extra: u64) -> (Fixture, StateDictMaterializer) {
    let f = Fixture::new(2);
    let mut m = StateDictMaterializer::new(128).unwrap();
    let mut limits = options(schedule);
    limits.max_host_bytes += extra;
    m.configure_expert_read_pipeline(limits).unwrap();
    (f, m)
}

fn prepare(m: &StateDictMaterializer, sequence: &mut ExpertReadSequence) -> PreparedSwiGlu {
    if sequence.current.is_none() {
        sequence.demand(0, 0).unwrap();
    }
    let read = sequence.current.as_ref().unwrap();
    let linear = |i| {
        PreparedLinear::from_parameter(
            m.expert_parameter_from_read(read, i, &read.plan.parameters[i])
                .unwrap(),
            read.plan.parameters[i].role().clone(),
        )
        .unwrap()
    };
    let value = PreparedSwiGlu::new(linear(0), linear(1), linear(2), None).unwrap();
    sequence.track_artifacts(&value).unwrap();
    value
}

#[test]
fn cap_minus_one_rejects_before_demand_or_materialization_exact_cap_reserves_all_partitions() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        for delta in [0, 1] {
            let extra = P + TICKET as u64 + HEADROOM as u64 - delta;
            let (f, m) = setup(schedule, extra);
            let mut sequence = m
                .expert_read_sequence(IMAGE, &[&f.metadata[0]])
                .unwrap()
                .unwrap();
            let before = m.checkpoint_read_counters();
            let result = sequence.reserve_upload(&f.metadata[0], TICKET, HEADROOM);
            assert_eq!(result.is_ok(), delta == 0);
            assert_eq!(m.checkpoint_read_counters(), before);
            if let Ok(reservation) = result {
                let observed = m.expert_read_pipeline_observation();
                assert_eq!(observed.current_charge.upload_reserved_packed_bytes, P);
                assert_eq!(
                    observed.current_charge.host_charged_bytes,
                    observed.limits.unwrap().max_host_bytes
                );
                assert_eq!(
                    reservation.host_charge(),
                    ExpertUploadHostCharge {
                        prepared_bytes: P,
                        pinned_bytes: P,
                        ticket_metadata_bytes: TICKET as u64,
                        policy_headroom_bytes: HEADROOM as u64,
                    }
                );
                assert!(
                    sequence
                        .reserve_upload(&f.metadata[1], TICKET, HEADROOM)
                        .is_err()
                );
                // Abandoned accepted reservations are fail-closed, not refunds.
                drop(reservation);
                assert!(sequence.upload_charge().is_err());
            }
        }
    }
}

#[test]
fn pending_and_cancel_do_not_refund_aliases_exact_completion_releases_once_and_reuses_worker() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let (f, m) = setup(schedule, 4 * P + TICKET as u64 + HEADROOM as u64);
        let selected = [&f.metadata[0], &f.metadata[1]];
        let mut sequence = m.expert_read_sequence(IMAGE, &selected).unwrap().unwrap();
        let mut reservation = sequence
            .reserve_upload(&f.metadata[0], TICKET, HEADROOM)
            .unwrap();
        let value = prepare(&m, &mut sequence);
        let aliases = [value.gate(), value.up(), value.down()]
            .map(|p| p.numeric_fp8().unwrap().validated_payload().clone());
        sequence.transfer_upload(&mut reservation).unwrap();
        assert!(sequence.transfer_upload(&mut reservation).is_err());
        drop(value);
        assert!(reservation.release().is_err()); // Pending; cancellation alone is not proof.
        assert!(sequence.demand(0, 1).is_err());
        assert!(
            sequence
                .reserve_upload(&f.metadata[1], TICKET, HEADROOM)
                .is_err()
        );
        let observed = m.expert_read_pipeline_observation();
        assert_eq!(observed.current_charge.packed_bytes, P);
        assert_eq!(observed.current_charge.artifact_bytes, P);
        assert_eq!(observed.current_charge.pinned_upload_bytes, P);
        assert_eq!(
            observed.current_charge.upload_ticket_metadata_bytes,
            TICKET as u64
        );
        assert_eq!(
            observed.current_charge.upload_policy_headroom_bytes,
            HEADROOM as u64
        );
        reservation.prove_completion(); // CPU injection at the production proof seam.
        assert!(reservation.release().is_err()); // Raw source aliases outlive the proof.
        drop(aliases);
        reservation.release().unwrap();
        assert!(reservation.release().is_err());
        assert_eq!(
            m.expert_read_pipeline_observation()
                .current_charge
                .packed_bytes,
            0
        );
        assert_eq!(
            m.expert_read_pipeline_observation()
                .current_charge
                .pinned_upload_bytes,
            0
        );
        drop(sequence);
        let sequence = m.expert_read_sequence(IMAGE, &selected).unwrap().unwrap();
        assert_eq!(m.expert_read_pipeline_stats().workers_created, 1);
        assert_eq!(m.expert_read_pipeline_stats().worker_reuses, 1);
        drop(sequence);
    }
}

#[test]
fn unknown_or_unwind_retains_packed_prepared_pinned_charge_and_prevents_cold_replacement() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let (f, m) = setup(schedule, 4 * P + TICKET as u64 + HEADROOM as u64);
        let mut sequence = m
            .expert_read_sequence(IMAGE, &[&f.metadata[0]])
            .unwrap()
            .unwrap();
        let mut reservation = sequence
            .reserve_upload(&f.metadata[0], TICKET, HEADROOM)
            .unwrap();
        let value = prepare(&m, &mut sequence);
        sequence.transfer_upload(&mut reservation).unwrap();
        drop(value);
        drop(reservation); // Unknown, not a successful fence.
        let observed = m.expert_read_pipeline_observation();
        assert!(observed.worker_unknown);
        assert!(
            observed.current_charge.unknown_upload_bytes >= 2 * P + TICKET as u64 + HEADROOM as u64
        );
        assert_eq!(observed.current_charge.packed_bytes, P);
        assert!(sequence.demand(0, 0).is_err());
        drop(sequence);
        assert!(m.expert_read_sequence(IMAGE, &[&f.metadata[0]]).is_err());
        assert_eq!(m.expert_read_pipeline_stats().workers_created, 1);
    }
}

#[test]
fn stale_and_foreign_reservations_cannot_transfer_or_release() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let (f, m) = setup(schedule, 4 * P + TICKET as u64 + HEADROOM as u64);
        let mut a = m
            .expert_read_sequence(IMAGE, &[&f.metadata[0]])
            .unwrap()
            .unwrap();
        let mut reservation = a.reserve_upload(&f.metadata[0], TICKET, HEADROOM).unwrap();
        let value = prepare(&m, &mut a);
        reservation.validate_identity(IMAGE, 0, 0).unwrap();
        assert!(reservation.validate_identity(IMAGE + 1, 0, 0).is_err());
        assert!(reservation.validate_identity(IMAGE, 0, 1).is_err());
        reservation.epoch += 1;
        assert!(a.transfer_upload(&mut reservation).is_err());
        reservation.epoch -= 1;
        a.transfer_upload(&mut reservation).unwrap();
        drop(value);
        reservation.prove_completion();
        reservation.key.0 += 1;
        assert!(reservation.release().is_err());
        reservation.key.0 -= 1;
        reservation.release().unwrap();
    }
}

#[test]
fn accepted_lookahead_counts_packed_only_once_and_future_reads_cannot_bypass_upload_quota() {
    let (f, m) = setup(
        ExpertReadSchedule::Lookahead,
        P + TICKET as u64 + HEADROOM as u64,
    );
    let mut sequence = m
        .expert_read_sequence(IMAGE, &[&f.metadata[0], &f.metadata[1]])
        .unwrap()
        .unwrap();
    sequence.lookahead((0, 1), Some((0, 0)), false);
    let mut reservation = sequence
        .reserve_upload(&f.metadata[0], TICKET, HEADROOM)
        .unwrap();
    let value = prepare(&m, &mut sequence);
    sequence.transfer_upload(&mut reservation).unwrap();
    drop(value);
    let before = m.checkpoint_read_counters();
    assert!(sequence.submit((0, 1), true).is_err());
    assert_eq!(m.checkpoint_read_counters(), before);
    reservation.prove_completion();
    reservation.release().unwrap();
}

#[test]
fn explicit_unknown_cannot_be_healed_by_later_completion_or_alias_retirement() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let (f, m) = setup(schedule, 4 * P + TICKET as u64 + HEADROOM as u64);
        let mut sequence = m
            .expert_read_sequence(IMAGE, &[&f.metadata[0]])
            .unwrap()
            .unwrap();
        let mut reservation = sequence
            .reserve_upload(&f.metadata[0], TICKET, HEADROOM)
            .unwrap();
        let value = prepare(&m, &mut sequence);
        sequence.transfer_upload(&mut reservation).unwrap();
        drop(value);
        reservation.quarantine();
        reservation.prove_completion();
        assert!(reservation.release().is_err());
        assert!(m.expert_read_pipeline_observation().worker_unknown);
        assert_eq!(sequence.pipeline().budget().reserved_bytes, P);
    }
}

#[test]
fn transfer_keeps_original_admission_after_provider_scope_ends_until_proven_release() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let (f, m) = setup(schedule, 4 * P + TICKET as u64 + HEADROOM as u64);
        let selected = [&f.metadata[0]];
        let mut sequence = m.expert_read_sequence(IMAGE, &selected).unwrap().unwrap();
        let mut reservation = sequence
            .reserve_upload(&f.metadata[0], TICKET, HEADROOM)
            .unwrap();
        let value = prepare(&m, &mut sequence);
        sequence.transfer_upload(&mut reservation).unwrap();
        drop(value);
        drop(sequence);
        assert!(m.expert_read_sequence(IMAGE, &selected).is_err());
        reservation.prove_completion();
        reservation.release().unwrap();
        let sequence = m.expert_read_sequence(IMAGE, &selected).unwrap().unwrap();
        assert_eq!(m.expert_read_pipeline_stats().workers_created, 1);
        drop(sequence);
    }
}

#[test]
fn zero_default_upload_fields_do_not_change_existing_host_charge_json() {
    let json = serde_json::to_value(ExpertReadHostCharge::default()).unwrap();
    for field in [
        "upload_reserved_packed_bytes",
        "pinned_upload_bytes",
        "upload_ticket_metadata_bytes",
        "upload_policy_headroom_bytes",
        "unknown_upload_bytes",
    ] {
        assert!(json.get(field).is_none());
    }
}

#[test]
fn overflow_and_zero_policy_reservations_fail_without_io_or_worker_replacement() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let (f, m) = setup(schedule, 8192);
        let mut sequence = m
            .expert_read_sequence(IMAGE, &[&f.metadata[0]])
            .unwrap()
            .unwrap();
        let before = m.checkpoint_read_counters();
        for (ticket, headroom) in [(0, HEADROOM), (TICKET, 0), (usize::MAX, usize::MAX)] {
            assert!(
                sequence
                    .reserve_upload(&f.metadata[0], ticket, headroom)
                    .is_err()
            );
        }
        assert_eq!(m.checkpoint_read_counters(), before);
        assert_eq!(
            sequence.upload_charge().unwrap(),
            ExpertUploadHostCharge::default()
        );
    }
}

#[test]
fn exact_cap_handoff_never_double_counts_packed_floor_for_both_schedules() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let f = Fixture::new(2);
        let mut m = StateDictMaterializer::new(128).unwrap();
        let mut limits = options(schedule);
        limits.max_host_bytes += P + TICKET as u64 + HEADROOM as u64;
        let cap = limits.max_host_bytes;
        m.configure_expert_read_pipeline(limits).unwrap();
        let mut sequence = m
            .expert_read_sequence(IMAGE, &[&f.metadata[0], &f.metadata[1]])
            .unwrap()
            .unwrap();
        if schedule == ExpertReadSchedule::Lookahead {
            sequence.lookahead((0, 1), Some((0, 0)), false);
            assert!(sequence.pending.as_ref().unwrap().ticket.is_ok());
        }
        let mut reservation = sequence
            .reserve_upload(&f.metadata[0], TICKET, HEADROOM)
            .unwrap();
        let check = |packed, floor| {
            let observed = m.expert_read_pipeline_observation();
            assert_eq!(observed.current_charge.packed_bytes, packed);
            assert_eq!(observed.current_charge.upload_reserved_packed_bytes, floor);
            assert_eq!(observed.current_charge.host_charged_bytes, cap);
            assert_eq!(observed.stats.peak_host_charged_bytes, cap);
        };
        check(
            if schedule == ExpertReadSchedule::Lookahead {
                P
            } else {
                0
            },
            if schedule == ExpertReadSchedule::Lookahead {
                0
            } else {
                P
            },
        );
        sequence.demand(0, 0).unwrap();
        check(P, 0);
        let after_demand = m.checkpoint_read_counters();
        let value = prepare(&m, &mut sequence);
        assert_eq!(m.checkpoint_read_counters(), after_demand);
        check(P, 0);
        sequence.transfer_upload(&mut reservation).unwrap();
        check(P, 0);
        drop(value);
        check(P, 0);
        reservation.prove_completion();
        reservation.release().unwrap();
        let observed = m.expert_read_pipeline_observation();
        assert_eq!(
            observed.current_charge.host_charged_bytes,
            limits.max_metadata_bytes + limits.worker_stack_bytes as u64
        );
        assert_eq!(observed.stats.peak_host_charged_bytes, cap);
        assert_eq!(observed.current_charge.upload_reserved_packed_bytes, 0);
        drop(sequence);
        assert!(
            m.expert_read_pipeline_observation()
                .current_charge
                .host_charged_bytes
                <= cap
        );
    }
}

#[test]
fn upload_epoch_exhaustion_preserves_last_incarnation_and_never_wraps_or_reads() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let (f, m) = setup(schedule, 4 * P + TICKET as u64 + HEADROOM as u64);
        let selected = [&f.metadata[0]];
        let mut sequence = m.expert_read_sequence(IMAGE, &selected).unwrap().unwrap();
        let authority = Arc::clone(&sequence.admission.as_ref().unwrap().0.state);
        authority.lock().unwrap().upload_epoch = u64::MAX - 1;
        let mut reservation = sequence
            .reserve_upload(&f.metadata[0], TICKET, HEADROOM)
            .unwrap();
        assert_eq!(reservation.epoch, u64::MAX);
        {
            let state = authority.lock().unwrap();
            assert_eq!(state.upload_epoch, u64::MAX);
            let hold = state.upload.as_ref().unwrap();
            assert_eq!(hold.epoch, u64::MAX);
            assert_eq!(hold.key, (IMAGE, 0, 0));
            assert!(!hold.transferred && !hold.unknown && !state.quarantined);
        }
        assert!(
            sequence
                .reserve_upload(&f.metadata[0], TICKET, HEADROOM)
                .is_err()
        );
        let value = prepare(&m, &mut sequence);
        sequence.transfer_upload(&mut reservation).unwrap();
        drop(value);
        reservation.prove_completion();
        reservation.release().unwrap();
        let before = m.checkpoint_read_counters();
        let budget = sequence.pipeline().budget();
        let observed = m.expert_read_pipeline_observation();
        for _ in 0..2 {
            assert!(
                sequence
                    .reserve_upload(&f.metadata[0], TICKET, HEADROOM)
                    .unwrap_err()
                    .to_string()
                    .contains("epoch exhausted")
            );
            let state = authority.lock().unwrap();
            assert_eq!(state.upload_epoch, u64::MAX);
            assert!(state.upload.is_none());
            assert!(state.active && !state.quarantined);
            drop(state);
            assert_eq!(sequence.pipeline().budget(), budget);
            assert_eq!(m.expert_read_pipeline_observation(), observed);
            assert_eq!(m.checkpoint_read_counters(), before);
        }
        drop(sequence);
        let mut next = m.expert_read_sequence(IMAGE, &selected).unwrap().unwrap();
        assert!(
            next.reserve_upload(&f.metadata[0], TICKET, HEADROOM)
                .unwrap_err()
                .to_string()
                .contains("epoch exhausted")
        );
        assert_eq!(m.expert_read_pipeline_stats().workers_created, 1);
        assert_eq!(m.expert_read_pipeline_stats().worker_reuses, 1);
        assert_eq!(authority.lock().unwrap().upload_epoch, u64::MAX);
    }
}

#[test]
fn cancellation_release_requires_independent_completion_proof_and_source_alias_retirement() {
    // This is the real reservation proof seam, not a simulated driver drain or
    // timeout loop. No proof is supplied in the pending/failed-drain cells.
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        for completion_proved in [false, true] {
            for keep_alias in [false, true] {
                let (f, m) = setup(schedule, 4 * P + TICKET as u64 + HEADROOM as u64);
                let selected = [&f.metadata[0]];
                let mut sequence = m.expert_read_sequence(IMAGE, &selected).unwrap().unwrap();
                let mut reservation = sequence
                    .reserve_upload(&f.metadata[0], TICKET, HEADROOM)
                    .unwrap();
                let value = prepare(&m, &mut sequence);
                // Retain just ONE projection, so release must check every witness.
                let mut alias = Some(
                    value
                        .down()
                        .numeric_fp8()
                        .unwrap()
                        .validated_payload()
                        .clone(),
                );
                sequence.transfer_upload(&mut reservation).unwrap();
                drop(value);
                if !keep_alias {
                    alias = None;
                }
                let charged = m.expert_read_pipeline_observation().current_charge;
                if completion_proved {
                    reservation.prove_completion();
                }
                let released = reservation.release().is_ok();
                assert_eq!(released, completion_proved && !keep_alias);
                if !released {
                    assert_eq!(m.expert_read_pipeline_observation().current_charge, charged);
                    assert!(
                        sequence
                            .reserve_upload(&f.metadata[0], TICKET, HEADROOM)
                            .is_err()
                    );
                    assert!(sequence.demand(0, 0).is_err());
                    assert!(m.expert_read_sequence(IMAGE, &selected).is_err());
                    drop(alias);
                    if !completion_proved {
                        assert!(
                            reservation.release().is_err(),
                            "alias retirement is not completion proof"
                        );
                        reservation.prove_completion();
                    }
                    reservation.release().unwrap();
                }
                assert!(reservation.release().is_err());
                let observed = m.expert_read_pipeline_observation();
                assert_eq!(observed.current_charge.packed_bytes, 0);
                assert_eq!(observed.current_charge.artifact_bytes, 0);
                assert_eq!(observed.current_charge.pinned_upload_bytes, 0);
                assert_eq!(observed.current_charge.unknown_upload_bytes, 0);
                assert_eq!(
                    observed.current_charge.host_charged_bytes,
                    observed.limits.unwrap().max_metadata_bytes
                        + observed.limits.unwrap().worker_stack_bytes as u64
                );
                drop(sequence);
                drop(m.expert_read_sequence(IMAGE, &selected).unwrap().unwrap());
                assert_eq!(m.expert_read_pipeline_stats().workers_created, 1);
                assert_eq!(m.expert_read_pipeline_stats().worker_reuses, 1);
            }
        }
    }
}

#[test]
fn completed_host_read_and_joined_worker_are_not_upload_completion_proof() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let (f, m) = setup(schedule, 4 * P + TICKET as u64 + HEADROOM as u64);
        let mut sequence = m
            .expert_read_sequence(IMAGE, &[&f.metadata[0]])
            .unwrap()
            .unwrap();
        let authority = Arc::clone(&sequence.admission.as_ref().unwrap().0.state);
        let handoff = sequence.observation_test_handoff();
        let mut reservation = sequence
            .reserve_upload(&f.metadata[0], TICKET, HEADROOM)
            .unwrap();
        let value = prepare(&m, &mut sequence);
        assert_eq!(m.expert_read_pipeline_stats().verified_reads, 1);
        assert_eq!(m.expert_read_pipeline_stats().completed_reads, 1);
        sequence.transfer_upload(&mut reservation).unwrap();
        drop(value);
        assert!(reservation.payloads.iter().all(|p| !p.is_live()));
        drop(sequence);
        drop(m);
        assert!(handoff.observe_worker().joined);
        assert_eq!(handoff.budget().reserved_bytes, P);
        assert!(reservation.release().is_err());
        {
            let state = authority.lock().unwrap();
            let hold = state.upload.as_ref().unwrap();
            assert!(state.owner_closed);
            assert!(hold.transferred && !hold.unknown);
            assert_eq!(hold.epoch, reservation.epoch);
            assert_eq!(hold.charge, reservation.host_charge());
        }
        // Abandonment after a successful host join still has no upload proof.
        drop(reservation);
        let state = authority.lock().unwrap();
        assert!(state.quarantined);
        assert!(state.upload.as_ref().unwrap().unknown);
        assert_eq!(handoff.budget().reserved_bytes, P);
    }
}
