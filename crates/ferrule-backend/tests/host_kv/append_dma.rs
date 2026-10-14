//! Exercises the exact production chunk consumer, without CUDA or a fake owner.
use super::append_stream::{AppendStream, PrefixDownload};
use ferrule_backend::cpu::{
    StreamingF32GqaConfig, StreamingF32GqaRequest, streaming_f32_gqa_partial,
};
use ferrule_common::execution::*;
use ferrule_common::host_kv::placement::{PlacementBudget, PlacementCredits};
use ferrule_common::host_kv::{CommittedKvSource, HostKvAppendReadWitness};

fn fixture() -> (HostKvAppendReadWitness, StreamingF32GqaConfig) {
    let source = CommittedKvSource {
        owner: 11,
        state_slot: StateSlot::new(0),
        generation: 3,
        incarnation: 7,
        schema: KvLayoutSchemaIdentity::new(
            vec![
                KvPlaneDescriptor::new("standard_gqa.key", 2, 1, KvElementType::F32),
                KvPlaneDescriptor::new("standard_gqa.value", 2, 1, KvElementType::F32),
            ],
            4,
            16,
        ),
        committed_tokens: 7,
        positions: 0..7,
        pages: vec![KvPageId(1), KvPageId(2)],
    };
    let view = KvReservationView {
        state_slot: source.state_slot,
        execution_state_slot: source.state_slot,
        generation: 3,
        execution_generation: 3,
        positions: 7..8,
        newly_allocated: vec![],
        cow_replacement: None,
    };
    (
        HostKvAppendReadWitness::new(ExecutionTransactionId::new(9).unwrap(), 2, source, view)
            .unwrap(),
        StreamingF32GqaConfig {
            rows: 1,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            block_tokens: 3,
            softmax_scale: 0.5,
            sliding_window: None,
        },
    )
}
fn drive(
    stream: &mut AppendStream<u64>,
    query: &[f32],
    key: &[f32],
    value: &[f32],
) -> Vec<PrefixDownload> {
    let mut requests = Vec::new();
    while let Some(request) = stream.request().unwrap() {
        assert!(!stream.ready().unwrap());
        assert!(stream.consume(|_| panic!("premature consumer")).is_err());
        let values = match request {
            PrefixDownload::Query => query.to_vec(),
            PrefixDownload::Key { start, tokens } => key[start * 2..(start + tokens) * 2].to_vec(),
            PrefixDownload::TailKey => key[14..16].to_vec(),
            PrefixDownload::TailValue => value[14..16].to_vec(),
            PrefixDownload::Value { start, tokens } => {
                value[start * 2..(start + tokens) * 2].to_vec()
            }
        };
        let id = requests.len() as u64 + 1;
        stream.submitted(id).unwrap();
        assert!(stream.request().is_err());
        assert!(stream.submitted(id + 1).is_err());
        stream.complete(id, values.into_boxed_slice()).unwrap();
        requests.push(request);
    }
    requests
}
#[test]
fn real_consumer_chunks_original_prefix_and_merges_unnormalized_tail() {
    let (w, config) = fixture();
    let query = [1.0, -0.5, 0.25, 1.0];
    let key: Vec<_> = (0..16).map(|i| i as f32 * 0.15).collect();
    let value: Vec<_> = (0..16).map(|i| (i % 5) as f32 - 2.0).collect();
    for window in [None, Some(1), Some(3)] {
        let config = StreamingF32GqaConfig {
            sliding_window: window,
            ..config
        };
        let mut stream = AppendStream::new(&w, config).unwrap();
        let requests = drive(&mut stream, &query, &key, &value);
        assert_eq!(
            requests,
            vec![
                PrefixDownload::Query,
                PrefixDownload::Key {
                    start: 0,
                    tokens: 3
                },
                PrefixDownload::Value {
                    start: 0,
                    tokens: 3
                },
                PrefixDownload::Key {
                    start: 3,
                    tokens: 1
                },
                PrefixDownload::Value {
                    start: 3,
                    tokens: 1
                },
                PrefixDownload::Key {
                    start: 4,
                    tokens: 3
                },
                PrefixDownload::Value {
                    start: 4,
                    tokens: 3
                },
            ]
        );
        assert!(stream.ready().unwrap());
        assert!(!stream.consumed().unwrap()); // DMA ready is not consumer completion.
        let full = StreamingF32GqaRequest {
            config,
            query: &query,
            key: &key,
            value: &value,
            query_start: 7,
            kv_start: 0,
        };
        let expected = streaming_f32_gqa_partial(full).unwrap().output();
        let tail = streaming_f32_gqa_partial(full.token_range(7, 8).unwrap()).unwrap();
        stream
            .consume(|prefix| {
                let merged = super::prefix_read::merge_append_partials(
                    &w,
                    config,
                    0..7,
                    prefix,
                    7..8,
                    &tail,
                )?;
                for (actual, expected) in merged.output().iter().zip(&expected) {
                    assert!((actual - expected).abs() < 1e-5);
                }
                Ok(())
            })
            .unwrap();
        assert!(stream.consumed().unwrap());
        assert!(stream.consume(|_| panic!("consumer replay")).is_err());
    }
}
#[test]
fn foreign_stale_and_malformed_completions_are_sticky_unknown() {
    let (w, config) = fixture();
    for fault in 0..3 {
        let mut stream = AppendStream::new(&w, config).unwrap();
        stream.submitted(10u64).unwrap();
        let (id, payload) = match fault {
            0 => (11, vec![1.0; 4]),
            1 => (9, vec![1.0; 4]),
            _ => (10, vec![1.0; 3]),
        };
        assert!(stream.complete(id, payload.into_boxed_slice()).is_err());
        assert!(stream.is_unknown());
        assert!(
            stream
                .complete(10, vec![1.0; 4].into_boxed_slice())
                .is_err()
        );
        assert!(stream.request().is_err());
        assert!(stream.ready().is_err());
        assert!(stream.consumed().is_err());
    }
}
#[test]
fn failed_or_unwinding_consumer_never_refunds_or_becomes_completion() {
    let (w, config) = fixture();
    for unwind in [false, true] {
        let credits = super::reaper::committed_gqa_credits(config, w.source().pages.len()).unwrap();
        let budget = PlacementBudget::new(credits);
        let charge = budget.reserve_resources(credits).unwrap();
        let mut stream = AppendStream::new(&w, config).unwrap();
        drive(&mut stream, &[1.0; 4], &[1.0; 14], &[2.0; 14]);
        if unwind {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = stream.consume(|_| panic!("injected CPU unwind"));
                }))
                .is_err()
            );
        } else {
            assert!(
                stream
                    .consume(|_| Err(super::plan::error("injected CPU failure")))
                    .is_err()
            );
        }
        assert!(stream.is_unknown());
        assert!(stream.consumed().is_err());
        assert!(stream.consume(|_| Ok(())).is_err());
        drop(stream);
        drop(charge);
        assert_eq!(budget.in_use(), credits);
        assert_ne!(budget.in_use(), PlacementCredits::default());
    }
}
#[test]
fn duplicate_completion_and_wrong_plane_lengths_do_not_advance() {
    let (w, config) = fixture();
    for duplicate in [true, false] {
        let mut stream = AppendStream::new(&w, config).unwrap();
        stream.submitted(1u64).unwrap();
        stream.complete(1, vec![1.0; 4].into_boxed_slice()).unwrap();
        if duplicate {
            assert!(stream.complete(1, vec![1.0; 4].into_boxed_slice()).is_err());
        } else {
            stream.submitted(2).unwrap();
            assert!(stream.complete(2, vec![1.0; 5].into_boxed_slice()).is_err());
        }
        assert!(stream.is_unknown());
    }
}

#[test]
fn appended_shadow_tail_uses_the_same_downloaded_query_and_normalizes_once() {
    let (w, config) = fixture();
    let query = [1.0, -0.5, 0.25, 1.0];
    let key: Vec<_> = (0..16).map(|i| i as f32 * 0.15).collect();
    let value: Vec<_> = (0..16).map(|i| (i % 5) as f32 - 2.0).collect();
    for window in [None, Some(1), Some(3)] {
        let config = StreamingF32GqaConfig {
            sliding_window: window,
            ..config
        };
        let mut stream = AppendStream::new(&w, config).unwrap();
        stream.include_tail().unwrap();
        let requests = drive(&mut stream, &query, &key, &value);
        assert_eq!(
            &requests[requests.len() - 2..],
            &[PrefixDownload::TailKey, PrefixDownload::TailValue]
        );
        let expected = streaming_f32_gqa_partial(StreamingF32GqaRequest {
            config,
            query: &query,
            key: &key,
            value: &value,
            query_start: 7,
            kv_start: 0,
        })
        .unwrap()
        .output();
        stream
            .consume(|total| {
                for (actual, expected) in total.output().iter().zip(&expected) {
                    assert!((actual - expected).abs() < 1e-5);
                }
                Ok(())
            })
            .unwrap();
        assert!(stream.consumed().unwrap());
    }
}

#[test]
fn tail_mode_cannot_be_enabled_after_query_submission() {
    let (w, config) = fixture();
    let mut stream = AppendStream::new(&w, config).unwrap();
    stream.submitted(1u64).unwrap();
    assert!(stream.include_tail().is_err());
    stream.complete(1, vec![1.; 4].into()).unwrap();
    assert!(stream.include_tail().is_err());
}
