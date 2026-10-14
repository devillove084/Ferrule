use super::*;
use RuntimeKvTraceKind as K;
use ferrule_common::kvset::TraceEvent;

mod logging;
mod shared_validation;

fn config() -> RuntimeKvTraceConfig {
    RuntimeKvTraceConfig {
        run_scope: "test".into(),
        ..Default::default()
    }
}
fn recorder() -> RuntimeKvRecorder {
    RuntimeKvRecorder::new(config(), 2, 8, [1; 5]).unwrap()
}
fn admit(r: &mut RuntimeKvRecorder, id: u64, reuse: usize) {
    r.record(K::AdmissionPublished {
        observation: id,
        actual_tokens: reuse,
        actual_pages: reuse.div_ceil(2),
    });
}
fn complete() -> RuntimeKvTraceSnapshot {
    let mut r = recorder();
    r.demand(1, 1, &[1, 2, 3, 4]);
    admit(&mut r, 0, 0);
    r.input(1, 1, 0, &[1, 2, 3, 4]);
    r.finish_execution(1, 1, Some(4));
    r.terminal(1, TerminalStatus::Success);
    r.drain(true).unwrap()
}
fn reject(b: RuntimeKvTraceSnapshot) {
    assert!(project_runtime_kv_trace(std::slice::from_ref(&b)).is_err());
    assert!(replay_runtime_prefix_lru(&[b]).is_err());
}
#[test]
fn kv_trace_empty_and_empty_final_watermarks() {
    let mut r = recorder();
    let a = r.drain(false).unwrap();
    let b = r.drain(true).unwrap();
    assert!(project_runtime_kv_trace(&[a, b]).unwrap().events.is_empty());
    assert!(r.drain(true).is_none());
    let mut r = recorder();
    r.demand(1, 1, &[1]);
    r.terminal(1, TerminalStatus::Cancelled);
    let a = r.drain(false).unwrap();
    let b = r.drain(true).unwrap();
    reject(b.clone());
    assert!(project_runtime_kv_trace(&[a, b]).is_ok());
}
#[test]
fn kv_trace_lost_tail_and_nonfinal_watermarks() {
    let mut r = recorder();
    r.demand(1, 1, &[1]);
    r.terminal(1, TerminalStatus::Success);
    let a = r.drain(false).unwrap();
    r.demand(2, 2, &[2]);
    r.terminal(2, TerminalStatus::Success);
    let _lost = r.drain(false).unwrap();
    let b = r.drain(true).unwrap();
    assert!(project_runtime_kv_trace(&[a, b]).is_err());
    let mut b = complete();
    b.start_seq = 1;
    reject(b);
    let mut b = complete();
    b.next_seq += 1;
    reject(b);
    let mut b = complete();
    b.lifecycle_events[1].seq += 1;
    reject(b);
    let mut b = complete();
    b.finalized = false;
    reject(b);
}
#[test]
fn kv_trace_strict_schema_watermarks() {
    let v = serde_json::to_value(complete()).unwrap();
    for field in ["start_seq", "next_seq"] {
        let mut bad = v.clone();
        bad.as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<RuntimeKvTraceSnapshot>(bad).is_err());
    }
    let mut bad = v;
    bad["unknown"] = true.into();
    assert!(serde_json::from_value::<RuntimeKvTraceSnapshot>(bad).is_err());
    let mut b = complete();
    b.header.schema_version = 1;
    reject(b);
}
#[test]
fn kv_trace_known_prompt_is_not_committed_frontier() {
    let mut b = complete();
    if let K::ExecutionInput { start, tokens, .. } = &mut b.lifecycle_events[2].kind {
        *start = 3;
        tokens.start += 3;
        tokens.len = 1;
    }
    reject(b);
}
#[test]
fn kv_trace_commit_cannot_cross_execution_or_reuse_an_old_reservation() {
    let mut b = complete();
    if let K::LogicalKvPublished {
        committed_tokens, ..
    } = &mut b.lifecycle_events[3].kind
    {
        *committed_tokens = 5;
    }
    reject(b);
    let mut b = complete();
    if let K::LogicalKvPublished { reservation, .. } = &mut b.lifecycle_events[3].kind {
        *reservation = 2;
    }
    reject(b);
    let mut b = complete();
    if let K::Terminal { observation, .. } = &mut b.lifecycle_events[4].kind {
        *observation = 99;
    }
    reject(b);
}
#[test]
fn kv_trace_partial_abort_retry_decode_and_cross_batch() {
    let mut r = recorder();
    r.demand(1, 1, &[1, 2, 3]);
    admit(&mut r, 0, 0);
    r.input(1, 1, 0, &[1, 2, 3]);
    r.finish_execution(1, 1, Some(2));
    let a = r.drain(false).unwrap();
    r.input(1, 2, 2, &[3, 99]);
    r.finish_execution(1, 2, None);
    r.input(1, 3, 2, &[3, 4]);
    r.finish_execution(1, 3, Some(3));
    r.input(1, 4, 3, &[7]);
    r.finish_execution(1, 4, Some(4));
    r.terminal(1, TerminalStatus::Cancelled);
    let b = r.drain(true).unwrap();
    let projected = project_runtime_kv_trace(&[a, b]).unwrap();
    let pages = projected
        .events
        .iter()
        .filter_map(|e| {
            if let TraceEvent::Publish(p) = e {
                Some(p.pages.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(pages[1][1].tokens, vec![3]);
    assert_eq!(pages[2][1].tokens, vec![3, 7]);
}
#[test]
fn kv_trace_hit_miss_require_observation() {
    for outcome in [LookupOutcome::Hit, LookupOutcome::Miss] {
        let mut r = recorder();
        r.demand(1, 1, &[1, 2]);
        r.record(K::Lookup {
            observation: None,
            outcome,
            entry: (outcome == LookupOutcome::Hit).then_some(999),
            matched_tokens: usize::from(outcome == LookupOutcome::Hit),
            generation: 1,
        });
        admit(&mut r, 0, 0);
        r.terminal(1, TerminalStatus::Failed);
        reject(r.drain(true).unwrap());
    }
}
#[test]
fn kv_trace_arena_is_batch_local_and_ranges_are_checked() {
    let mut r = recorder();
    r.demand(1, 1, &[1]);
    r.terminal(1, TerminalStatus::Success);
    let a = r.drain(false).unwrap();
    r.demand(2, 2, &[8, 9]);
    r.terminal(2, TerminalStatus::Success);
    let b = r.drain(true).unwrap();
    let t = project_runtime_kv_trace(&[a, b]).unwrap();
    let TraceEvent::Demand(d) = &t.events[2] else {
        panic!()
    };
    assert_eq!(d.prompt_tokens, vec![8, 9]);
    let mut b = complete();
    if let K::Demand { tokens, .. } = &mut b.lifecycle_events[0].kind {
        tokens.start = usize::MAX;
    }
    reject(b);
}
#[test]
fn kv_trace_projection_checks_cumulative_budget_before_emit() {
    let b = complete();
    let limits = RuntimeKvProjectionLimits {
        max_events: 20,
        max_tokens: 8,
        max_publish_pages: 1,
    };
    let mut emitted = 0;
    let error = validation::validate_semantics(&[b], limits, |_, _, _| {
        emitted += 1;
        Ok(())
    })
    .unwrap_err();
    assert_eq!(error.0, "projection resource limit");
    assert_eq!(emitted, 1);
    let b = complete();
    assert!(
        project_runtime_kv_trace_with_limits(
            &[b],
            RuntimeKvProjectionLimits {
                max_events: 20,
                max_tokens: 8,
                max_publish_pages: 2
            }
        )
        .is_ok()
    );
    let mut r = recorder();
    r.demand(1, 1, &[1, 2]);
    admit(&mut r, 0, 0);
    r.input(1, 1, 0, &[1, 2]);
    r.finish_execution(1, 1, Some(2));
    let a = r.drain(false).unwrap();
    r.input(1, 2, 2, &[3]);
    r.finish_execution(1, 2, Some(3));
    r.terminal(1, TerminalStatus::Success);
    let b = r.drain(true).unwrap();
    assert!(
        project_runtime_kv_trace_with_limits(
            &[a, b],
            RuntimeKvProjectionLimits {
                max_events: 20,
                max_tokens: 6,
                max_publish_pages: 9
            }
        )
        .is_err()
    );
}
#[test]
fn kv_trace_overflow_is_sticky_across_drain() {
    for c in [
        RuntimeKvTraceConfig {
            max_events: 0,
            ..config()
        },
        RuntimeKvTraceConfig {
            max_token_bytes: 0,
            ..config()
        },
        RuntimeKvTraceConfig {
            max_observations: 0,
            ..config()
        },
    ] {
        let mut r = RuntimeKvRecorder::new(c, 2, 0, [1; 5]).unwrap();
        r.demand(1, 1, &[1]);
        assert!(r.drain(false).unwrap().status.overflow);
        let b = r.drain(true).unwrap();
        assert!(!b.complete);
        reject(b);
    }
}
#[test]
fn kv_trace_request_id_reuse_gets_new_incarnation() {
    let mut r = recorder();
    r.demand(1, 1, &[1]);
    r.terminal(1, TerminalStatus::Failed);
    r.demand(1, 1, &[2]);
    r.terminal(1, TerminalStatus::Success);
    let p = project_runtime_kv_trace(&[r.drain(true).unwrap()]).unwrap();
    assert_eq!(p.events[2].request().generation, 1);
}
#[test]
fn kv_trace_unclosed_and_unsupported_are_incomplete() {
    let mut r = recorder();
    r.demand(1, 1, &[1]);
    reject(r.drain(true).unwrap());
    let mut r = recorder();
    r.unsupported();
    reject(r.drain(true).unwrap());
}
#[test]
fn kv_trace_actual_entry_lru_touch_pin_replacement_and_charge() {
    let mut r = recorder();
    r.record_tokens(&[1, 2], |tokens| K::PrefixEntryPublished {
        entry: 1,
        tokens,
        charge_pages: 1,
        generation: 1,
        replaced: false,
    });
    r.record_tokens(&[9], |tokens| K::PrefixEntryPublished {
        entry: 2,
        tokens,
        charge_pages: 1,
        generation: 2,
        replaced: false,
    });
    r.demand(1, 1, &[1, 2, 3]);
    r.record(K::Lookup {
        observation: Some(0),
        outcome: LookupOutcome::Hit,
        entry: Some(1),
        matched_tokens: 2,
        generation: 3,
    });
    r.record(K::UnpinAck { entry: 1 });
    r.terminal(1, TerminalStatus::Failed);
    r.record_tokens(&[9], |tokens| K::PrefixEntryRemoved {
        tokens,
        charge_pages: 1,
        generation: 2,
        reason: RemovalReason::PagePressure,
    });
    r.record_tokens(&[1, 2], |tokens| K::PrefixEntryPublished {
        entry: 1,
        tokens,
        charge_pages: 1,
        generation: 4,
        replaced: true,
    });
    let report = replay_runtime_prefix_lru(&[r.drain(true).unwrap()]).unwrap();
    assert_eq!(report.lookup_hits, 1);
    assert_eq!(report.published_admissions, 0);
    assert_eq!(report.charged_pages, 1);
    assert_eq!(report.removed_charge_pages, 2);
    assert_eq!(report.backend_released_pages, 0);
}
#[test]
fn kv_trace_actual_rejects_false_miss_and_illegal_eviction() {
    let mut r = recorder();
    r.record_tokens(&[1], |tokens| K::PrefixEntryPublished {
        entry: 1,
        tokens,
        charge_pages: 1,
        generation: 1,
        replaced: false,
    });
    r.demand(1, 1, &[1, 2]);
    r.record(K::Lookup {
        observation: Some(0),
        outcome: LookupOutcome::Miss,
        entry: None,
        matched_tokens: 0,
        generation: 0,
    });
    r.terminal(1, TerminalStatus::Failed);
    assert!(replay_runtime_prefix_lru(&[r.drain(true).unwrap()]).is_err());
}

#[test]
fn kv_trace_only_last_batch_may_finalize() {
    let mut r = recorder();
    let mut a = r.drain(false).unwrap();
    let b = r.drain(true).unwrap();
    a.finalized = true;
    a.complete = true;
    assert!(project_runtime_kv_trace(&[a.clone(), b.clone()]).is_err());
    assert!(replay_runtime_prefix_lru(&[a, b]).is_err());
}
#[test]
fn kv_trace_full_lifecycle_is_checked_before_actual_report() {
    let mut b = complete();
    b.lifecycle_events.pop();
    b.next_seq -= 1;
    reject(b);
    let mut b = complete();
    let mut terminal = b.lifecycle_events.last().unwrap().clone();
    terminal.seq = b.next_seq;
    b.next_seq += 1;
    b.lifecycle_events.push(terminal);
    reject(b);
    let mut b = complete();
    if let K::ExecutionInput { tokens, .. } = &b.lifecycle_events[2].kind {
        b.tokens[tokens.start] = 99;
    }
    reject(b);
}
#[test]
fn kv_trace_cancelled_commits_remain_common_history() {
    let mut r = recorder();
    r.demand(1, 1, &[1, 2]);
    admit(&mut r, 0, 0);
    r.input(1, 1, 0, &[1, 2]);
    r.finish_execution(1, 1, Some(2));
    r.terminal(1, TerminalStatus::Cancelled);
    r.demand(2, 2, &[1, 2, 3]);
    r.terminal(2, TerminalStatus::Failed);
    let input = project_runtime_kv_trace(&[r.drain(true).unwrap()]).unwrap();
    let report = ferrule_common::kvset::analyze(&input, Default::default()).unwrap();
    assert_eq!(report.workload.opportunity_requests, 1);
    assert_eq!(report.outcomes.cancelled, 1);
    assert_eq!(report.outcomes.failed, 1);
}

fn reject_batches_with(batches: &[RuntimeKvTraceSnapshot], reason: &'static str) {
    let expected = RuntimeKvTraceError(reason);
    assert_eq!(project_runtime_kv_trace(batches).unwrap_err(), expected);
    assert_eq!(
        project_runtime_kv_trace_with_limits(batches, RuntimeKvProjectionLimits::default())
            .unwrap_err(),
        expected
    );
    assert_eq!(replay_runtime_prefix_lru(batches).unwrap_err(), expected);
}

fn cached_hit() -> RuntimeKvTraceSnapshot {
    let mut r = recorder();
    for (entry, key) in [(1, &[1, 2][..]), (2, &[1, 2, 3][..])] {
        r.record_tokens(key, |tokens| K::PrefixEntryPublished {
            entry,
            tokens,
            charge_pages: key.len().div_ceil(2),
            generation: entry,
            replaced: false,
        });
    }
    r.demand(1, 1, &[1, 2, 3, 4]);
    r.record(K::Lookup {
        observation: Some(0),
        outcome: LookupOutcome::Hit,
        entry: Some(2),
        matched_tokens: 3,
        generation: 3,
    });
    r.record(K::UnpinAck { entry: 2 });
    admit(&mut r, 0, 3);
    r.input(1, 1, 3, &[4]);
    r.finish_execution(1, 1, Some(4));
    r.terminal(1, TerminalStatus::Success);
    r.drain(true).unwrap()
}

#[test]
fn kv_trace_projection_rejects_nonexistent_hit_before_frontier_publication() {
    let mut r = recorder();
    r.demand(1, 1, &[1, 2, 3, 4]);
    r.record(K::Lookup {
        observation: Some(0),
        outcome: LookupOutcome::Hit,
        entry: Some(999),
        matched_tokens: 3,
        generation: 1,
    });
    admit(&mut r, 0, 3);
    r.input(1, 1, 3, &[4]);
    r.finish_execution(1, 1, Some(4));
    r.terminal(1, TerminalStatus::Success);
    let batches = [r.drain(true).unwrap()];
    reject_batches_with(&batches, "lookup differs from exact longest prefix");
    let mut publications = 0;
    assert!(
        validation::validate_semantics(
            &batches,
            RuntimeKvProjectionLimits::default(),
            |_, _, event| {
                publications += usize::from(matches!(event, validation::Projected::Publish { .. }));
                Ok(())
            },
        )
        .is_err()
    );
    assert_eq!(publications, 0);
}

#[test]
fn kv_trace_shared_entry_validation_checks_identity_content_and_longest_match() {
    let valid = cached_hit();
    let input = project_runtime_kv_trace(std::slice::from_ref(&valid)).unwrap();
    let TraceEvent::Publish(p) = &input.events[1] else {
        panic!("expected full committed-prefix publication")
    };
    assert_eq!(
        p.pages
            .iter()
            .flat_map(|p| p.tokens.iter().copied())
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
    let actual = replay_runtime_prefix_lru(std::slice::from_ref(&valid)).unwrap();
    assert_eq!(actual.actual_reused_tokens, 3);
    assert_eq!(actual.lookup_hits, 1);
    for variant in 0..4 {
        let mut b = valid.clone();
        if variant == 3 {
            let K::PrefixEntryPublished { tokens, .. } = b.lifecycle_events[1].kind else {
                panic!()
            };
            b.tokens[tokens.start] = 99;
        } else if let K::Lookup {
            entry,
            matched_tokens,
            ..
        } = &mut b.lifecycle_events[3].kind
        {
            match variant {
                0 => *entry = Some(999),
                1 => *matched_tokens = 2,
                _ => {
                    *entry = Some(1);
                    *matched_tokens = 2;
                }
            }
        } else {
            panic!()
        }
        reject_batches_with(&[b], "lookup differs from exact longest prefix");
    }
}

#[test]
fn kv_trace_shared_entry_validation_rejects_stale_hit_and_admission_mismatch() {
    let mut r = recorder();
    r.record_tokens(&[1, 2], |tokens| K::PrefixEntryPublished {
        entry: 1,
        tokens,
        charge_pages: 1,
        generation: 1,
        replaced: false,
    });
    r.record_tokens(&[1, 2], |tokens| K::PrefixEntryRemoved {
        tokens,
        charge_pages: 1,
        generation: 1,
        reason: RemovalReason::PagePressure,
    });
    r.demand(1, 1, &[1, 2, 3]);
    r.record(K::Lookup {
        observation: Some(0),
        outcome: LookupOutcome::Hit,
        entry: Some(1),
        matched_tokens: 2,
        generation: 2,
    });
    admit(&mut r, 0, 2);
    r.terminal(1, TerminalStatus::Failed);
    reject_batches_with(
        &[r.drain(true).unwrap()],
        "lookup differs from exact longest prefix",
    );

    let mut b = cached_hit();
    let K::AdmissionPublished {
        actual_tokens,
        actual_pages,
        ..
    } = &mut b.lifecycle_events[5].kind
    else {
        panic!()
    };
    *actual_tokens = 2;
    *actual_pages = 1;
    reject_batches_with(&[b], "admission differs from lookup");
    let mut b = cached_hit();
    let K::AdmissionPublished { actual_pages, .. } = &mut b.lifecycle_events[5].kind else {
        panic!()
    };
    *actual_pages = 1;
    reject_batches_with(&[b], "invalid admission frontier");
}

#[test]
fn kv_trace_usage_rejects_each_forged_counter_including_maximum_values() {
    for field in 0..3 {
        for value in [0, u64::MAX] {
            let mut b = complete();
            match field {
                0 => b.usage.events = value,
                1 => b.usage.token_bytes = value,
                _ => b.usage.observations = value,
            }
            reject_batches_with(&[b], "trace usage mismatch");
        }
    }
}

#[test]
fn kv_trace_usage_rejects_erased_history_even_with_reset_watermarks() {
    let mut b = complete();
    b.lifecycle_events.clear();
    b.tokens.clear();
    b.start_seq = 0;
    b.next_seq = 0;
    reject_batches_with(&[b], "trace usage mismatch");
}

#[test]
fn kv_trace_usage_and_entry_state_persist_across_empty_and_nonempty_batches() {
    let mut r = recorder();
    r.record_tokens(&[1, 2], |tokens| K::PrefixEntryPublished {
        entry: 1,
        tokens,
        charge_pages: 1,
        generation: 1,
        replaced: false,
    });
    let a = r.drain(false).unwrap();
    let empty = r.drain(false).unwrap();
    r.demand(1, 1, &[1, 2, 3]);
    r.record(K::Lookup {
        observation: Some(0),
        outcome: LookupOutcome::Hit,
        entry: Some(1),
        matched_tokens: 2,
        generation: 2,
    });
    let b = r.drain(false).unwrap();
    r.record(K::UnpinAck { entry: 1 });
    admit(&mut r, 0, 2);
    r.input(1, 1, 2, &[3]);
    r.finish_execution(1, 1, Some(3));
    r.terminal(1, TerminalStatus::Success);
    // Usage counts all Demand observations, not just concurrently live ones.
    r.demand(1, 1, &[9]);
    r.terminal(1, TerminalStatus::Cancelled);
    let c = r.drain(false).unwrap();
    let final_batch = r.drain(true).unwrap();
    assert!(final_batch.lifecycle_events.is_empty());
    assert_eq!(final_batch.usage.observations, 2);
    let batches = [a, empty, b, c, final_batch];
    project_runtime_kv_trace(&batches).unwrap();
    let actual = replay_runtime_prefix_lru(&batches).unwrap();
    assert_eq!(actual.actual_reused_tokens, 2);
    assert_eq!(actual.published_admissions, 1);
    for index in 0..batches.len() {
        for field in 0..3 {
            let mut bad = batches.clone();
            match field {
                0 => bad[index].usage.events += 1,
                1 => bad[index].usage.token_bytes += 4,
                _ => bad[index].usage.observations += 1,
            }
            reject_batches_with(&bad, "trace usage mismatch");
        }
    }
}

#[test]
fn kv_trace_usage_distinguishes_true_empty_trace_from_hidden_history() {
    let mut r = recorder();
    let batches = [r.drain(false).unwrap(), r.drain(true).unwrap()];
    assert!(
        project_runtime_kv_trace(&batches)
            .unwrap()
            .events
            .is_empty()
    );
    assert_eq!(
        replay_runtime_prefix_lru(&batches).unwrap(),
        RuntimePrefixReplay::default()
    );
    for field in 0..3 {
        let mut b = batches[1].clone();
        match field {
            0 => b.usage.events = 1,
            1 => b.usage.token_bytes = 4,
            _ => b.usage.observations = 1,
        }
        reject_batches_with(&[b], "trace usage mismatch");
    }
}
