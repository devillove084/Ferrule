//! Real private lookup hooks exercised against the existing CPU-testable device
//! cache policy. No mock GPU work and no alternative production bucket loop.

use super::expert_cache::{Cache, ExpertCacheKey};
use super::*;
use crate::transformer::expert_reads::tests::{Fixture, IMAGE};
use crate::transformer::{ExpertCacheLimits, ExpertReadSchedule, NumericFp8Precision};
use std::collections::{BTreeMap, BTreeSet};

fn key(metadata: &super::super::ExpertMetadata) -> ExpertCacheKey {
    let crate::nn::ParameterResidency::Expert { layer, expert } =
        *metadata.parameters()[0].residency()
    else {
        panic!()
    };
    ExpertCacheKey {
        image_generation: IMAGE,
        numeric_precision: Some(NumericFp8Precision::Bf16ModelMixedF32Accumulate),
        layer,
        expert,
        parameters: metadata
            .parameters()
            .each_ref()
            .map(|binding| (binding.canonical_id(), binding.role().clone())),
    }
}

#[test]
fn default_noop_hooks_keep_legacy_payload_protocol() {
    struct Legacy {
        calls: usize,
    }
    impl ExpertProvider for Legacy {
        fn expert(&mut self, _: usize, _: usize) -> Result<ExpertAvailability> {
            self.calls += 1;
            Ok(ExpertAvailability::Waiting)
        }
    }
    let f = Fixture::new(1);
    let mut provider = Legacy { calls: 0 };
    provider
        .preflight_expert_reads(IMAGE, &[&f.metadata[0]])
        .unwrap();
    provider.lookahead_expert_read(&f.metadata[0], None, false);
    assert_eq!(provider.calls, 0);
    assert!(matches!(
        provider.expert(0, 0).unwrap(),
        ExpertAvailability::Waiting
    ));
}

#[test]
fn private_state_dict_hooks_preserve_completed_only_lru_routes_and_payloads() {
    let f = Fixture::new(4);
    let semantics = f
        .metadata
        .iter()
        .map(ExpertSemantics::read_test_fixture)
        .collect::<Vec<_>>();
    let mut runs = Vec::new();
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let materializer = f.materializer(schedule);
        let mut cache = Cache::<ExpertCacheKey>::new(ExpertCacheLimits {
            max_experts: 2,
            max_bytes: 40,
        })
        .unwrap();
        let mut residents = BTreeSet::new();
        let mut victims = Vec::new();
        let mut payloads = Vec::new();
        let routes = [
            vec![2, 0, 2, 0],
            vec![3, 0, 3],
            vec![1, 2, 1],
            vec![3, 1, 3],
            vec![0, 2, 0],
        ];
        let mut visited = Vec::new();
        for routes in routes {
            // The live loops bucket in BTree order without changing source slot
            // weighting. Duplicate routes across different rows use one bucket.
            let mut buckets = BTreeMap::<usize, Vec<usize>>::new();
            for (slot, expert) in routes.into_iter().enumerate() {
                buckets.entry(expert).or_default().push(slot);
            }
            let mut provider = StateDictExpertProvider {
                layer: 0,
                bindings: &semantics,
                materializer: &materializer,
                reads: None,
            };
            provider
                .preflight_expert_reads(
                    IMAGE,
                    &buckets
                        .keys()
                        .map(|id| &f.metadata[*id])
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let mut buckets = buckets.into_iter().peekable();
            while let Some((expert, slots)) = buckets.next() {
                let metadata = &f.metadata[expert];
                let current = key(metadata);
                cache
                    .scratch(4, |victim| {
                        residents.remove(victim);
                        victims.push(victim.clone());
                    })
                    .unwrap();
                let lease = cache
                    .acquire(current.clone(), 18, |victim| {
                        residents.remove(victim);
                        victims.push(victim.clone());
                    })
                    .unwrap();
                let cache_before_hook = cache.stats();
                if !residents.contains(&current) {
                    let before = materializer.checkpoint_read_counters().bytes_read;
                    let ExpertAvailability::Ready(prepared) = provider.expert(0, expert).unwrap()
                    else {
                        panic!()
                    };
                    assert!(materializer.checkpoint_read_counters().bytes_read - before <= 18);
                    let payload = [prepared.gate(), prepared.up(), prepared.down()]
                        .into_iter()
                        .flat_map(|linear| {
                            let artifact = linear.numeric_fp8().unwrap();
                            artifact
                                .weight_bytes()
                                .iter()
                                .chain(artifact.scale_bytes())
                                .copied()
                        })
                        .collect::<Vec<_>>();
                    payloads.push((expert, payload));
                    drop(prepared); // Stand-in for the owner's completed host upload scope, not GPU math.
                    residents.insert(current);
                }
                let next = buckets.peek().map(|(expert, _)| &f.metadata[*expert]);
                let next_resident = next.is_some_and(|metadata| residents.contains(&key(metadata)));
                provider.lookahead_expert_read(metadata, next, next_resident);
                assert_eq!(
                    cache.stats(),
                    cache_before_hook,
                    "lookahead must not acquire, pin or touch LRU"
                );
                visited.push((expert, slots));
                cache.complete(&lease, true).unwrap();
            }
            drop(provider); // Real host worker joins/drains before the next scope.
        }
        let reads = materializer.expert_read_pipeline_stats();
        assert_eq!(reads.wasted_reads, 0);
        assert_eq!(reads.unknown_bytes, 0);
        assert_eq!(
            materializer.checkpoint_read_counters().bytes_read,
            cache.stats().uploads * 18
        );
        assert!(
            reads.peak_host_charged_bytes
                <= materializer
                    .expert_read_pipeline_options()
                    .unwrap()
                    .max_host_bytes
        );
        runs.push((
            visited,
            payloads,
            victims,
            cache.stats(),
            materializer.checkpoint_read_counters().bytes_read,
        ));
    }
    assert_eq!(runs[0], runs[1]);
}

#[test]
fn private_provider_preflight_rejects_foreign_bindings_before_read_or_host_spawn() {
    let f = Fixture::new(1);
    let alien = Fixture::new(1);
    let materializer = f.materializer(ExpertReadSchedule::Lookahead);
    let semantics = vec![ExpertSemantics::read_test_fixture(&f.metadata[0])];
    let mut provider = StateDictExpertProvider {
        layer: 0,
        bindings: &semantics,
        materializer: &materializer,
        reads: None,
    };
    assert!(
        provider
            .preflight_expert_reads(IMAGE, &[&alien.metadata[0]])
            .is_err()
    );
    assert_eq!(materializer.checkpoint_read_counters().bytes_read, 0);
    assert!(provider.reads.is_none());
    assert!(provider.expert(1, 0).is_err());
    assert!(provider.expert(0, 1).is_err());
}

#[test]
fn decoder_constructor_options_reach_the_real_materializer_and_private_provider() {
    use crate::decoder::GenericDecoderOptions;
    use crate::execution::ExecutionPrecisionPolicy;
    use crate::spec::{ModelFamily, WeightSource};
    use ferrule_common::execution::{ExecutionCapabilities, KvBindingMode, LogitsRowPolicy};
    let f = Fixture::new(2);
    let semantics = f
        .metadata
        .iter()
        .map(ExpertSemantics::read_test_fixture)
        .collect::<Vec<_>>();
    let base = GenericDecoderOptions::new(
        ModelFamily::Unknown("expert-read-constructor".into()),
        WeightSource::Safetensors,
        ExecutionCapabilities {
            max_batch_tokens: 1,
            max_sequences: 1,
            max_prefill_query_tokens_per_sequence: 1,
            max_decode_query_tokens_per_sequence: 1,
            max_top_k: None,
            supports_prefill: true,
            supports_decode: true,
            supports_mixed: false,
            full_logits_width: None,
            kv_binding_mode: KvBindingMode::Paged,
            logits_row_policy: LogitsRowPolicy::Any,
        },
        1,
        8,
        128,
        ExecutionPrecisionPolicy::f32(),
    );
    assert_eq!(base.expert_read_pipeline_options(), None);
    assert_eq!(
        base.state_dict_materializer()
            .unwrap()
            .expert_read_pipeline_options(),
        None
    );
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let read = crate::transformer::expert_reads::tests::options(schedule);
        let options = base
            .clone()
            .with_hybrid_cuda_numeric_fp8_precision(
                ExpertCacheLimits {
                    max_experts: 1,
                    max_bytes: 128,
                },
                0,
                NumericFp8Precision::W8A16Bf16F32Accumulate,
            )
            .unwrap()
            .with_expert_read_pipeline(read)
            .unwrap();
        assert_eq!(options.expert_read_pipeline_options(), Some(read));
        assert!(options.clone().with_expert_read_pipeline(read).is_err());
        // This is the same construction seam called by the normal CUDA module
        // and segment, not a separately manually configured test materializer.
        let materializer = options.state_dict_materializer().unwrap();
        assert_eq!(materializer.expert_read_pipeline_options(), Some(read));
        assert_eq!(materializer.checkpoint_read_counters().bytes_read, 0);
        let mut provider = StateDictExpertProvider {
            layer: 0,
            bindings: &semantics,
            materializer: &materializer,
            reads: None,
        };
        provider
            .preflight_expert_reads(IMAGE, &[&f.metadata[0], &f.metadata[1]])
            .unwrap();
        let ExpertAvailability::Ready(value) = provider.expert(0, 0).unwrap() else {
            panic!()
        };
        assert_eq!(materializer.checkpoint_read_counters().bytes_read, 18);
        drop(value);
        provider.lookahead_expert_read(&f.metadata[0], Some(&f.metadata[1]), false);
        let ExpertAvailability::Ready(value) = provider.expert(0, 1).unwrap() else {
            panic!()
        };
        drop(value);
        drop(provider);
        let stats = materializer.expert_read_pipeline_stats();
        assert_eq!(stats.submitted_reads, 2);
        assert_eq!(
            stats.speculative_reads,
            u64::from(schedule == ExpertReadSchedule::Lookahead)
        );
        assert_eq!(stats.unknown_bytes, 0);
        assert!(stats.peak_host_charged_bytes <= read.max_host_bytes);
    }
}
