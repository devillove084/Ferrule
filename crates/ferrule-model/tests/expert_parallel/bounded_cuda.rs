//! Tiny real CUDA coverage of the existing owner-local routed seam.
use super::*;
use ferrule_backend::cuda::operators::linear::CudaOperators;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::{
    CudaStandardDecoderOperators, ExpertCacheLimits, ExpertCachePolicy, RowsDevice,
};
use std::rc::Rc;

fn cuda(fixture: &PreparedFixture, policy: ExpertCachePolicy) -> CudaStandardDecoderOperators {
    let parameters = fixture
        .experts
        .iter()
        .flat_map(|expert| {
            [expert.gate(), expert.up(), expert.down()].map(|p| p.parameter().binding().clone())
        })
        .collect::<Vec<_>>();
    CudaStandardDecoderOperators::new_with_expert_cache(
        Rc::new(CudaOperators::new_on_device(0).unwrap()),
        ExecutionPrecisionPolicy::f32(),
        &parameters,
        policy,
    )
    .unwrap()
}
fn ready(progress: OperatorProgress<Rows>) -> Rows {
    match progress {
        OperatorProgress::Ready(rows) => rows,
        _ => panic!("expected Ready"),
    }
}
fn policy(entries: usize, bytes: usize) -> ExpertCachePolicy {
    ExpertCachePolicy::Bounded(ExpertCacheLimits {
        max_experts: entries,
        max_bytes: bytes,
    })
}
fn input() -> Rows {
    Rows::Host(
        HostRows::new(
            RowsShape::new(2, 2).unwrap(),
            RowsDType::F32,
            None,
            vec![0.25, -0.5, 1.5, 0.125],
        )
        .unwrap(),
    )
}
fn close(actual: &[f32], expected: &[f32]) {
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && e.is_finite() && (a - e).abs() <= 2e-5 * (1.0 + e.abs()),
            "{a} != {e}"
        );
    }
    assert_eq!(actual.len(), expected.len());
}

#[test]
#[ignore = "requires CUDA GPU; FERRULE_CUDA_ARCH=sm_86"]
fn bounded_cuda_forced_eviction_reload_f32_bf16_matches_cpu_and_keep_all() {
    for dtype in [CheckpointDType::F32, CheckpointDType::Bf16] {
        let fixture = PreparedFixture::with_dtype(dtype);
        // Each expert = 3 * 2 * 2 * 4 = 48 device bytes, even for BF16.
        // One entry is intentionally smaller than top-k=2. Byte-only admission
        // is also tested with 8 slots but room for just one expert + scratch.
        for (entries, bytes) in [(1, 256), (8, 240)] {
            let mut bounded = cuda(&fixture, policy(entries, bytes));
            let mut legacy = cuda(&fixture, ExpertCachePolicy::KeepAll);
            let gpu_input = bounded.bind_rows(input()).unwrap();
            let legacy_input = legacy.bind_rows(input()).unwrap();
            let mut cpu = CpuStandardDecoderOperators::new(ExecutionPrecisionPolicy::f32());
            let baseline_refs = fixture
                .experts
                .iter()
                .map(Arc::strong_count)
                .collect::<Vec<_>>();
            for iteration in 0..8 {
                let ids = if iteration % 2 == 0 {
                    vec![0, 1, 0, 1]
                } else {
                    vec![1, 0, 1, 0]
                };
                let routes = RouterRoutes::new(2, 2, ids, vec![0.25, 0.75, 0.6, 0.4]).unwrap();
                let mut provider = LocalProvider::new(&fixture, &[0, 1]);
                let expected = ready(
                    cpu.routed_swiglu(LAYER, &input(), &routes, &mut provider, None)
                        .unwrap(),
                );
                let expected = cpu.download_rows(expected).unwrap();
                let actual = ready(
                    bounded
                        .routed_swiglu(LAYER, &gpu_input, &routes, &mut provider, None)
                        .unwrap(),
                );
                assert_eq!(actual.device(), RowsDevice::Cuda { ordinal: 0 });
                let actual = bounded.download_rows(actual).unwrap();
                close(actual.values(), expected.values());
                let other = ready(
                    legacy
                        .routed_swiglu(LAYER, &legacy_input, &routes, &mut provider, None)
                        .unwrap(),
                );
                close(
                    legacy.download_rows(other).unwrap().values(),
                    expected.values(),
                );
                let stats = bounded.expert_cache_stats().unwrap();
                assert_eq!(stats.resident_experts, 1);
                assert_eq!(stats.resident_bytes, 48);
                assert_eq!(stats.pending_upload_bytes, 0);
                assert_eq!(stats.scratch_bytes, 0);
                assert!(stats.peak_bytes <= bytes);
                assert_eq!(bounded.resident_parameter_bytes(), 48);
                assert!(!stats.quarantined);
                drop(provider);
                // Only compatibility CUDA retains a host Arc; the bounded
                // owner must not retain another one across calls.
                assert_eq!(
                    fixture
                        .experts
                        .iter()
                        .map(Arc::strong_count)
                        .collect::<Vec<_>>(),
                    baseline_refs.iter().map(|n| n + 1).collect::<Vec<_>>()
                );
            }
            let stats = bounded.expert_cache_stats().unwrap();
            assert!(stats.evictions > 8 && stats.uploads > 8);
            assert!(legacy.expert_cache_stats().is_none());
            assert_eq!(legacy.resident_parameter_bytes(), 96);
        }
    }
}

#[test]
#[ignore = "requires CUDA GPU; dense F32/BF16 metadata-only resident hits"]
fn bounded_metadata_dense_hit_never_calls_payload_provider() {
    use ferrule_model::transformer::{BoundParameter, ExpertMetadata};
    struct MetadataProvider {
        parameters: Vec<[BoundParameter; 3]>,
        calls: usize,
        live: Vec<std::sync::Weak<PreparedSwiGlu>>,
    }
    impl ExpertProvider for MetadataProvider {
        fn expert_metadata(
            &mut self,
            layer: usize,
            expert: usize,
        ) -> Result<Option<ExpertMetadata>> {
            ExpertMetadata::new(layer, expert, self.parameters[expert].clone(), None).map(Some)
        }
        fn expert(&mut self, layer: usize, expert: usize) -> Result<ExpertAvailability> {
            self.calls += 1;
            let materializer = StateDictMaterializer::new(1024).unwrap();
            let projection = |index| {
                let binding = &self.parameters[expert][index];
                PreparedLinear::from_parameter(
                    materializer.expert_parameter(layer, expert, binding)?,
                    binding.role().clone(),
                )
            };
            let expert = Arc::new(PreparedSwiGlu::new(
                projection(0)?,
                projection(1)?,
                projection(2)?,
                None,
            )?);
            self.live.push(Arc::downgrade(&expert));
            Ok(ExpertAvailability::Ready(expert))
        }
    }
    for dtype in [CheckpointDType::F32, CheckpointDType::Bf16] {
        let fixture = PreparedFixture::with_dtype(dtype);
        let mut provider = MetadataProvider {
            parameters: fixture
                .experts
                .iter()
                .map(|e| [e.gate(), e.up(), e.down()].map(|p| p.parameter().binding().clone()))
                .collect(),
            calls: 0,
            live: Vec::new(),
        };
        let mut owner = cuda(&fixture, policy(64, 1024));
        let input = owner.bind_rows(input()).unwrap();
        let routes = RouterRoutes::new(2, 2, vec![0, 1, 0, 1], vec![0.5; 4]).unwrap();
        let mut previous = None;
        for _ in 0..2 {
            let output = ready(
                owner
                    .routed_swiglu(LAYER, &input, &routes, &mut provider, None)
                    .unwrap(),
            );
            let output = owner.download_rows(output).unwrap();
            assert_eq!(provider.calls, 2);
            assert!(provider.live.iter().all(|w| w.upgrade().is_none()));
            if let Some(expected) = &previous {
                assert_eq!(output.values(), expected);
            }
            previous = Some(output.values().to_vec());
        }
        let stats = owner.expert_cache_stats().unwrap();
        assert_eq!(stats.uploads, 2);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.resident_bytes, 96);
    }
}

#[test]
#[ignore = "requires CUDA GPU; FERRULE_CUDA_ARCH=sm_86"]
fn bounded_cuda_admission_identity_failures_and_recovery() {
    let fixture = PreparedFixture::new();
    let mut owner = cuda(&fixture, policy(1, 128));
    owner.prepare_expert(&fixture.experts[0]).unwrap();
    assert_eq!(owner.resident_parameter_bytes(), 48);
    owner.prepare_expert(&fixture.experts[1]).unwrap();
    owner.prepare_expert(&fixture.experts[0]).unwrap();
    assert_eq!(owner.expert_cache_stats().unwrap().uploads, 3);
    let gpu_input = owner.bind_rows(input()).unwrap();
    assert!(
        owner
            .linear(fixture.experts[0].gate(), &gpu_input, None)
            .is_err()
    );
    let alien = PreparedFixture::new();
    assert!(
        owner
            .prepare_expert(&alien.experts[0])
            .unwrap_err()
            .to_string()
            .contains("another prepared image")
    );
    let swapped = PreparedSwiGlu::new(
        fixture.experts[0].up().clone(),
        fixture.experts[0].gate().clone(),
        fixture.experts[0].down().clone(),
        None,
    )
    .unwrap();
    assert!(owner.prepare_expert(&swapped).is_err());
    let mixed = PreparedSwiGlu::new(
        fixture.experts[0].gate().clone(),
        fixture.experts[1].up().clone(),
        fixture.experts[0].down().clone(),
        None,
    )
    .unwrap();
    assert!(owner.prepare_expert(&mixed).is_err());
    assert!(!owner.needs_quarantine());
    owner.operators().failpoints().arm_allocation();
    assert!(
        owner
            .dense_swiglu(&fixture.experts[1], &gpu_input, None)
            .is_err()
    );
    assert!(!owner.needs_quarantine());
    assert_eq!(owner.expert_cache_stats().unwrap().pending_upload_bytes, 0);
    let result = ready(
        owner
            .dense_swiglu(&fixture.experts[1], &gpu_input, None)
            .unwrap(),
    );
    assert!(
        owner
            .download_rows(result)
            .unwrap()
            .values()
            .iter()
            .all(|v| v.is_finite())
    );
    let mut too_small = cuda(&fixture, policy(1, 47));
    assert!(too_small.prepare_expert(&fixture.experts[0]).is_err());
    assert_eq!(too_small.resident_parameter_bytes(), 0);
    assert_eq!(too_small.expert_cache_stats().unwrap().charged_bytes(), 0);
}
