use super::*;
use crate::engine::kv_trace::validation::{Projected, no_expand, validate_semantics};

#[test]
fn no_expand_and_dto_sinks_share_report_and_borrowed_events() {
    let batches = [cached_hit()];
    let dto = project_runtime_kv_trace(&batches).unwrap();
    let mut emitted = 0;
    let validated = validate_semantics(
        &batches,
        RuntimeKvProjectionLimits::default(),
        |header, seq, event| {
            assert!(std::ptr::eq(header, &batches[0].header));
            match (event, &dto.events[emitted]) {
                (Projected::Demand { tokens, .. }, TraceEvent::Demand(demand)) => {
                    let K::Demand { tokens: span, .. } = batches[0].lifecycle_events[2].kind else {
                        panic!("expected demand")
                    };
                    assert_eq!(tokens.as_ptr(), batches[0].tokens[span.start..].as_ptr());
                    assert_eq!(tokens, demand.prompt_tokens);
                    assert_eq!(seq, demand.seq);
                }
                (Projected::Publish { tokens, .. }, TraceEvent::Publish(publish)) => {
                    let expanded = publish
                        .pages
                        .iter()
                        .flat_map(|page| page.tokens.iter().copied())
                        .collect::<Vec<_>>();
                    assert_eq!(tokens, expanded);
                    assert_eq!(seq, publish.seq);
                }
                (Projected::Outcome { actual, .. }, TraceEvent::Outcome(outcome)) => {
                    let actual = actual.unwrap();
                    let dto_actual = outcome.actual.as_ref().unwrap();
                    assert_eq!(actual.0, dto_actual.reused_prefix_tokens);
                    assert_eq!(actual.1, dto_actual.reused_pages);
                    assert_eq!(seq, outcome.seq);
                }
                _ => panic!("sink event mismatch"),
            }
            emitted += 1;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(emitted, 3);
    assert_eq!(
        validated.report,
        validate_semantics(&batches, RuntimeKvProjectionLimits::default(), no_expand)
            .unwrap()
            .report
    );
    assert_eq!(
        validated.report,
        replay_runtime_prefix_lru(&batches).unwrap()
    );
}

#[test]
fn no_expand_preserves_input_and_projection_limits() {
    let batches = [complete()];
    for (limits, reason) in [
        (
            RuntimeKvProjectionLimits {
                max_events: 4,
                ..Default::default()
            },
            "event resource limit",
        ),
        (
            RuntimeKvProjectionLimits {
                max_tokens: 7,
                ..Default::default()
            },
            "input resource limit",
        ),
        (
            RuntimeKvProjectionLimits {
                max_publish_pages: 1,
                ..Default::default()
            },
            "projection resource limit",
        ),
    ] {
        let expected = RuntimeKvTraceError(reason);
        assert_eq!(
            validate_semantics(&batches, limits, no_expand).unwrap_err(),
            expected
        );
        assert_eq!(
            project_runtime_kv_trace_with_limits(&batches, limits).unwrap_err(),
            expected
        );
    }
}

#[test]
fn batch_integrity_precedes_earlier_entry_errors_and_all_sink_calls() {
    let mut first = cached_hit();
    if let K::Lookup { entry, .. } = &mut first.lifecycle_events[3].kind {
        *entry = Some(999);
    }
    let mut final_batch = first.clone();
    final_batch.lifecycle_events.clear();
    final_batch.tokens.clear();
    final_batch.start_seq = first.next_seq;
    final_batch.usage.observations += 1;
    first.finalized = false;
    first.complete = false;
    let batches = [first, final_batch];
    reject_batches_with(&batches, "trace usage mismatch");
    let mut emitted = 0;
    let error = validate_semantics(&batches, RuntimeKvProjectionLimits::default(), |_, _, _| {
        emitted += 1;
        Ok(())
    })
    .unwrap_err();
    assert_eq!(error.0, "trace usage mismatch");
    assert_eq!(emitted, 0);
}

#[test]
fn entry_errors_precede_lookup_shape_and_admission_frontier_errors() {
    let mut batch = cached_hit();
    if let K::Lookup {
        entry,
        outcome,
        matched_tokens,
        ..
    } = &mut batch.lifecycle_events[3].kind
    {
        *entry = Some(999);
        *outcome = LookupOutcome::Miss;
        *matched_tokens = 0;
    }
    reject_batches_with(&[batch], "lookup differs from exact longest prefix");

    let mut batch = cached_hit();
    if let K::AdmissionPublished {
        actual_tokens,
        actual_pages,
        ..
    } = &mut batch.lifecycle_events[5].kind
    {
        *actual_tokens = usize::MAX;
        *actual_pages = usize::MAX;
    }
    reject_batches_with(&[batch], "admission differs from lookup");
}

#[test]
fn frontier_rejection_precedes_projection_budget_and_sink_failure_is_preserved() {
    let mut batch = complete();
    if let K::LogicalKvPublished {
        committed_tokens, ..
    } = &mut batch.lifecycle_events[3].kind
    {
        *committed_tokens = 5;
    }
    let batches = [batch];
    let limits = RuntimeKvProjectionLimits {
        max_publish_pages: 0,
        ..Default::default()
    };
    let expected = RuntimeKvTraceError("publish crosses unproven frontier");
    assert_eq!(
        validate_semantics(&batches, limits, no_expand).unwrap_err(),
        expected
    );
    assert_eq!(
        project_runtime_kv_trace_with_limits(&batches, limits).unwrap_err(),
        expected
    );

    let batches = [complete()];
    let mut emitted = 0;
    let error = validate_semantics(&batches, RuntimeKvProjectionLimits::default(), |_, _, _| {
        emitted += 1;
        Err(RuntimeKvTraceError("sink rejected"))
    })
    .unwrap_err();
    assert_eq!(error.0, "sink rejected");
    assert_eq!(emitted, 1);
}

#[test]
fn recorder_checked_usage_overflow_is_sticky_without_appending() {
    for counter in 0..3 {
        let mut r = recorder();
        match counter {
            0 => r.usage.token_bytes = u64::MAX,
            1 => r.next_seq = u64::MAX,
            _ => r.next_observation = u64::MAX,
        }
        if counter == 2 {
            r.demand(1, 1, &[1]);
        } else {
            r.record_tokens(&[1], |tokens| K::PrefixEntryPublished {
                entry: 1,
                tokens,
                charge_pages: 1,
                generation: 1,
                replaced: false,
            });
        }
        assert!(r.status.overflow);
        assert!(r.events.is_empty());
        assert!(r.tokens.is_empty());
        let first = r.drain(false).unwrap();
        let final_batch = r.drain(true).unwrap();
        assert!(final_batch.status.overflow);
        reject_batches_with(&[first, final_batch], "incomplete or discontinuous trace");
    }
}
