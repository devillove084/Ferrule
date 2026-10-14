//! Owner fault gates: compile with CUDA, run only with a separate GPU grant.
use super::*;
use crate::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use crate::decoder::{DecoderSequenceLifecycle, HybridCudaMemoryBudget};
use crate::nn::{ModulePath, ParameterDType, ParameterSpec};
use crate::transformer::*;
use ferrule_common::{ParallelRankId, ParallelTopologyId};

struct Fixture {
    path: std::path::PathBuf,
    resources: BoundDecoderResources,
    plan: HybridMixedGqaBlockPlan,
}
impl Fixture {
    fn new() -> Self {
        let norm = || RmsNorm::new(8, 1e-6).unwrap();
        let rope = RotaryEmbedding::new(
            8,
            10000.,
            RotaryPairing::SplitHalf,
            RotaryRegion::Prefix { dimensions: 4 },
            RotaryScaling::None,
        )
        .unwrap();
        let attentions = [
            Attention::Gqa(GqaAttention::new(8, 2, 1, 8, false, rope).unwrap()),
            Attention::GatedDeltaNet(
                GatedDeltaNetAttention::new(8, 2, 4, 3, 2, 3, 1e-6, false).unwrap(),
            ),
        ];
        let spec = DecoderModelSpec::new(DecoderModelParts {
            architecture: "mixed-prepare-fault".into(),
            hidden_size: 8,
            vocab_size: 11,
            max_sequence_length: Some(32),
            token_embedding: Embedding::new(11, 8, None).unwrap(),
            layers: attentions
                .into_iter()
                .enumerate()
                .map(|(i, a)| {
                    DecoderLayer::new(
                        i,
                        norm(),
                        a,
                        Residual::Add,
                        norm(),
                        FeedForward::SwiGlu(SwiGlu::new(8, 12, false).unwrap()),
                        Residual::Add,
                    )
                    .unwrap()
                })
                .collect(),
            final_norm: norm(),
            output: Linear::new(8, 11, false).unwrap(),
            tie_word_embeddings: false,
        })
        .unwrap();
        let parallel = HybridParallelPlan::new_with_execution(
            &spec,
            0..2,
            HybridExpertScope {
                topology: ParallelTopologyId::new(907),
                replica: 0,
                stage: 0,
            },
            vec![ParallelRankId::new(0), ParallelRankId::new(1)],
            HybridRoutedExpertExecution::None,
            HybridRankPrecision::MixedBf16F32,
        )
        .unwrap();
        let plan = HybridMixedGqaBlockPlan::new_gqa_gdn_pair(
            Arc::new(parallel.rank_layout(0).unwrap()),
            2,
            1,
            32,
            2,
            3,
        )
        .unwrap();
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "mixed-prepare-fault-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let mut data = Vec::new();
        let mut slices = Vec::new();
        let mut schema = StateDictSchema::builder();
        let mut mapper = ExactNameMapper::new();
        for layer in 0..2 {
            let mut tensors = vec![
                (TensorRole::AttentionNorm, vec![8]),
                (TensorRole::FeedForwardNorm, vec![8]),
                (TensorRole::DenseMlpGate, vec![12, 8]),
                (TensorRole::DenseMlpUp, vec![12, 8]),
                (TensorRole::DenseMlpDown, vec![8, 12]),
            ];
            if layer == 0 {
                tensors.extend([
                    (TensorRole::AttentionQuery, vec![16, 8]),
                    (TensorRole::AttentionKey, vec![8, 8]),
                    (TensorRole::AttentionValue, vec![8, 8]),
                    (TensorRole::AttentionOutput, vec![8, 16]),
                ]);
            } else {
                tensors.extend([
                    (TensorRole::LinearAttentionQkv, vec![20, 8]),
                    (TensorRole::LinearAttentionZ, vec![8, 8]),
                    (TensorRole::LinearAttentionA, vec![4, 8]),
                    (TensorRole::LinearAttentionBeta, vec![4, 8]),
                    (TensorRole::AttentionOutput, vec![8, 8]),
                    (TensorRole::LinearAttentionConv, vec![20, 1, 3]),
                    (TensorRole::LinearAttentionALog, vec![4]),
                    (TensorRole::LinearAttentionDtBias, vec![4]),
                    (TensorRole::LinearAttentionNorm, vec![2]),
                ]);
            }
            for (role, shape) in tensors {
                let id = slices.len() as u64 + 1;
                let name = format!("layers.{layer}.p{id}");
                let module = ModulePath::new(&name).unwrap();
                let f32_source = matches!(
                    role,
                    TensorRole::LinearAttentionALog | TensorRole::LinearAttentionNorm
                );
                let offset = data.len();
                for _ in 0..shape.iter().product::<usize>() {
                    if f32_source {
                        data.extend(f32::from_bits(0x3f80_0400).to_le_bytes());
                    } else {
                        data.extend(0x3c00_u16.to_le_bytes());
                    }
                }
                schema
                    .register_with_role(
                        ParameterSpec::new(
                            ParameterId::new(id),
                            module.clone(),
                            if f32_source {
                                ParameterDType::F32
                            } else {
                                ParameterDType::Bf16
                            },
                            shape.clone(),
                            ParameterResidency::layer(layer),
                        )
                        .unwrap(),
                        role.clone(),
                    )
                    .unwrap();
                mapper.insert(&name, NameMapping::weight(module)).unwrap();
                slices.push(CheckpointTensorSlice {
                    name,
                    role,
                    path: path.clone(),
                    offset: offset as u64,
                    bytes: (data.len() - offset) as u64,
                    shape,
                    dtype: if f32_source {
                        CheckpointDType::F32
                    } else {
                        CheckpointDType::Bf16
                    },
                });
            }
        }
        std::fs::write(&path, data).unwrap();
        let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
            .bind_slices(slices)
            .unwrap();
        Self {
            path,
            resources: BoundDecoderResources::new(spec, Arc::new(bound)).unwrap(),
            plan,
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

// Preparation-only test transport. No fabricated runtime execution authority:
// both control methods retain fail-closed defaults and exchange must not run.
struct PreparationOnly(HybridBf16CollectiveBinding);
impl HybridBf16Collective for PreparationOnly {
    fn owner(&self) -> ParallelRankId {
        self.0.members[0]
    }
    fn members(&self) -> &[ParallelRankId] {
        &self.0.members
    }
    fn sealed_binding(&self) -> Result<HybridBf16CollectiveBinding> {
        Ok(self.0.clone())
    }
    fn exchange_bf16(
        &mut self,
        _: ExecutionTransactionId,
        _: u64,
        _: HybridCollectiveKind,
        _: Vec<u16>,
    ) -> Result<Vec<u16>> {
        panic!("preparation must never exchange data")
    }
    fn abort(&mut self) {}
}
fn prepare(
    owner: &mut CudaStandardDecoderOperators,
    fixture: &Fixture,
    device: &HybridCudaDevice,
) -> Result<CudaHybridRankLifecycle> {
    owner.prepare_mixed_rank_gqa_block(
        &fixture.resources,
        fixture.plan.clone(),
        device.clone(),
        1 << 20,
        [4 << 20, 64 << 20, 16 << 20],
        1 << 20,
        Box::new(PreparationOnly(fixture.plan.collective_binding())),
        &[ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionShape { rows: 2, tokens: 2 }],
        [64 << 20, 16 << 20],
    )
}
fn owner(device: &HybridCudaDevice, fixture: &Fixture) -> CudaStandardDecoderOperators {
    CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
        device.operators().clone(),
        fixture.resources.state_dict().parameters(),
        ExpertCachePolicy::KeepAll,
        0,
        NumericFp8Precision::W8A16Bf16F32Accumulate,
    )
    .unwrap()
}
fn device() -> HybridCudaDevice {
    assert_eq!(
        std::env::var("FERRULE_RUN_MIXED_RANK_GPU").as_deref(),
        Ok("1")
    );
    HybridCudaDevice::new_on_device(
        0,
        HybridCudaMemoryBudget {
            kv_pages: 16,
            state_bytes: 1 << 20,
            weight_bytes: 4 << 20,
            workspace_bytes: 256 << 20,
        },
    )
    .unwrap()
}
const STAGES: [PreparationStage; 5] = [
    PreparationStage::Projections,
    PreparationStage::Vectors,
    PreparationStage::Chunk,
    PreparationStage::Attention,
    PreparationStage::Lifecycle,
];

fn claim_inputs(
    f: &Fixture,
) -> (
    crate::decoder::HybridCudaMemoryEstimate,
    crate::decoder::HybridCudaPreparationContext,
) {
    f.plan
        .cuda_claim_inputs(
            &f.resources,
            16,
            0,
            1 << 20,
            &[
                ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionShape {
                    rows: 2,
                    tokens: 2,
                },
                ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionShape {
                    rows: 1,
                    tokens: 1,
                },
            ],
        )
        .unwrap()
}

#[test]
fn mixed_claim_context_rejects_source_geometry_and_underbudget_without_gpu() {
    let f = Fixture::new();
    let (estimate, context) = claim_inputs(&f);
    let budget = HybridCudaMemoryBudget {
        kv_pages: 16,
        state_bytes: estimate.per_sequence_state_bytes * 3,
        weight_bytes: estimate.weight_bytes_upper_bound,
        workspace_bytes: estimate.workspace_bytes_upper_bound,
    };
    context.validate_device_budget(0, budget).unwrap();
    context.validate_estimate(estimate).unwrap();
    context.validate_for(&claim_inputs(&f).1).unwrap();
    assert!(context.validate_device_budget(1, budget).is_err());
    for field in 0..4 {
        let mut bad = budget;
        match field {
            0 => bad.kv_pages -= 1,
            1 => bad.state_bytes -= 1,
            2 => bad.weight_bytes -= 1,
            _ => bad.workspace_bytes -= 1,
        }
        assert!(context.validate_device_budget(0, bad).is_err());
        let mut bad = estimate;
        match field {
            0 => bad.kv_bytes += 1,
            1 => bad.per_sequence_state_bytes += 1,
            2 => bad.weight_bytes_upper_bound += 1,
            _ => bad.workspace_bytes_upper_bound += 1,
        }
        assert!(context.validate_estimate(bad).is_err());
    }
    let foreign = Fixture::new();
    assert_eq!(f.resources.spec(), foreign.resources.spec());
    assert_eq!(estimate, claim_inputs(&foreign).0);
    assert!(
        context.validate_for(&claim_inputs(&foreign).1).is_err(),
        "same geometry is not source identity"
    );
    for field in 0..8 {
        let mut bad = context.clone();
        match field {
            0 => bad.device_ordinal += 1,
            1 => bad.physical_pages += 1,
            2 => bad.max_parameter_bytes += 1,
            3 => bad.projection_limits[1] += 1,
            4 => bad.auxiliary_grants += 1,
            5 => bad.attention_limits[0] += 1,
            6 => bad.attention_shapes[0].tokens += 1,
            _ => bad.program.max_rows -= 1,
        }
        assert!(context.validate_for(&bad).is_err());
    }
    assert!(
        f.plan
            .cuda_claim_inputs(&f.resources, 16, 0, 1, &context.attention_shapes)
            .is_err()
    );
    std::fs::remove_file(&f.path).unwrap();
    assert_eq!(claim_inputs(&f).1, context, "preflight is payload-free");
}

fn prepare_claimed(
    owner: &mut CudaStandardDecoderOperators,
    f: &Fixture,
    device: &HybridCudaDevice,
    claim: &crate::decoder::HybridCudaPreparationClaim,
) -> Result<CudaHybridRankLifecycle> {
    owner.prepare_mixed_rank_program_with_claim(
        &f.resources,
        f.plan.clone(),
        device.clone(),
        1 << 20,
        Box::new(PreparationOnly(f.plan.collective_binding())),
        &claim_inputs(f).1.attention_shapes,
        claim,
    )
}

#[test]
#[ignore = "GPU grant required; existing claim rollback across five actual preparation stages"]
fn mixed_claim_known_failures_rollback_then_finish_lifecycle() {
    let f = Fixture::new();
    let device = device();
    let mut owner = owner(&device, &f);
    for stage in STAGES {
        let (estimate, context) = claim_inputs(&f);
        let claim = device.admit_with_context(estimate, context).unwrap();
        PREPARATION_FAULT.set(Some((stage, false)));
        let result = prepare_claimed(&mut owner, &f, &device, &claim);
        assert!(result.is_err());
        assert!(!claim.image_is_prepared());
        assert_eq!(PREPARATION_FAULT.get(), None);
        owner.validate_mixed_preparation_owner().unwrap();
        assert!(!device.needs_quarantine());
        assert!(claim.finish(result).is_err());
    }
    let (estimate, context) = claim_inputs(&f);
    let claim = device.admit_with_context(estimate, context).unwrap();
    let mut life = prepare_claimed(&mut owner, &f, &device, &claim).unwrap();
    assert!(claim.image_is_prepared());
    let initial = life.create().unwrap();
    let (mut life, initial) = claim.finish(Ok((life, initial))).unwrap();
    assert!(owner.mixed_rank_block.as_ref().unwrap().published);
    life.try_release(initial).unwrap();
    assert!(
        device
            .admit_with_context(estimate, claim_inputs(&f).1)
            .is_err()
    );
}

#[test]
#[ignore = "GPU grant required; Unknown must retain the original claim and all pending roots"]
fn mixed_claim_unknown_retains_and_never_reopens() {
    let f = Fixture::new();
    for stage in STAGES {
        let device = device();
        let mut owner = owner(&device, &f);
        let (estimate, context) = claim_inputs(&f);
        let claim = device.admit_with_context(estimate, context).unwrap();
        PREPARATION_FAULT.set(Some((stage, true)));
        let result = prepare_claimed(&mut owner, &f, &device, &claim);
        assert!(result.is_err());
        assert!(!claim.image_is_prepared());
        assert_eq!(PREPARATION_FAULT.get(), None);
        assert!(owner.needs_quarantine() && device.needs_quarantine());
        let live = device.operators().allocator_metrics().live_granted_bytes;
        assert!(claim.finish(result).is_err());
        drop(owner);
        device.operators().sync_stream().unwrap();
        device.operators().sync_upload_stream().unwrap();
        assert!(device.operators().allocator_metrics().live_granted_bytes >= live);
        assert!(
            device
                .admit_with_context(estimate, claim_inputs(&f).1)
                .is_err()
        );
    }
}

#[test]
fn mixed_preparation_source_vectors_preflight_without_gpu() {
    let f = Fixture::new();
    let reader =
        StateDictMaterializer::for_hybrid_cuda_rank(1 << 20, f.plan.layout.clone(), 4 << 20)
            .unwrap();
    for descriptor in f.resources.spec().layers() {
        crate::transformer::standard::hybrid::prepare_standard_layer(
            descriptor,
            &f.resources,
            &reader,
            &mut MemoryLayerWeightCache::new(),
            32,
        )
        .unwrap();
    }
    for b in f.resources.state_dict().parameters() {
        if matches!(
            b.role(),
            TensorRole::LinearAttentionNorm | TensorRole::LinearAttentionALog
        ) {
            let p = reader
                .layer_parameter(1, b, &mut MemoryLayerWeightCache::new())
                .unwrap();
            assert_eq!(p.mixed_vector_bf16_words().unwrap(), vec![0x3f80; 2]);
            assert_eq!(p.binding().weight().slice().dtype, CheckpointDType::F32);
        }
    }
}

#[test]
#[ignore = "GPU grant required; five real preparation stages, bounded known-complete retry"]
fn mixed_prepare_known_failures_rollback_and_retry() {
    let f = Fixture::new();
    let device = device();
    let mut owner = owner(&device, &f);
    for stage in STAGES {
        let mut stable = None;
        for _ in 0..3 {
            PREPARATION_FAULT.set(Some((stage, false)));
            assert!(prepare(&mut owner, &f, &device).is_err());
            assert_eq!(PREPARATION_FAULT.get(), None, "did not reach {stage:?}");
            assert!(!owner.needs_quarantine());
            owner.validate_mixed_preparation_owner().unwrap();
            assert!(owner.mixed_rank_block.is_none() && owner.mixed_gqa_block.is_none());
            assert!(owner.bf16_hold.is_empty() && owner.expert_hold.is_empty());
            assert_eq!(owner.resident_parameter_bytes(), 0);
            let metrics = device.operators().allocator_metrics();
            let current = (metrics.live_granted_bytes, metrics.reserved_bytes);
            if let Some(previous) = stable {
                assert_eq!(previous, current, "retry growth after {stage:?}");
            }
            stable = Some(current);
        }
    }
    let life = prepare(&mut owner, &f, &device).unwrap();
    assert!(owner.mixed_rank_block.as_ref().unwrap().published && owner.mixed_gqa_block.is_some());
    assert!(owner.resident_parameter_bytes() > 0);
    drop(life);
    drop(owner);
}

#[test]
#[ignore = "GPU grant required; terminal Unknown retains allocations at all five preparation stages"]
fn mixed_prepare_unknown_never_releases_or_retries() {
    let f = Fixture::new();
    for stage in STAGES {
        // Unknown retains the provider pool permanently on its stream owner.
        // The next independent fault case must not reuse that reserved owner.
        let device = device();
        let mut owner = owner(&device, &f);
        PREPARATION_FAULT.set(Some((stage, true)));
        let error = prepare(&mut owner, &f, &device)
            .err()
            .expect("injected Unknown");
        assert_eq!(
            PREPARATION_FAULT.get(),
            None,
            "did not reach {stage:?}: {error:?}"
        );
        assert!(owner.needs_quarantine());
        assert!(!owner.mixed_rank_block.as_ref().unwrap().published);
        let before = device.operators().allocator_metrics().live_granted_bytes;
        assert!(before > 0);
        assert!(owner.quiesce().is_err());
        assert!(prepare(&mut owner, &f, &device).is_err());
        assert_eq!(
            before,
            device.operators().allocator_metrics().live_granted_bytes
        );
        drop(owner);
        assert_eq!(
            before,
            device.operators().allocator_metrics().live_granted_bytes
        );
    }
}

#[test]
#[ignore = "GPU grant required; model W8 completion bridge retains resident image and operand leases"]
fn mixed_w8_completion_bridge_quarantines_and_retains() {
    let f = Fixture::new();
    let device = device();
    let mut owner = owner(&device, &f);
    let life = prepare(&mut owner, &f, &device).unwrap();
    for poisoned in [false, true] {
        let result = owner.synchronous(|this| {
            let input = this.ops.zero_bf16_buffer(8)?;
            let output = this.ops.zero_f32_buffer(8)?;
            this.hold_bf16(&input)?;
            this.expert_hold
                .push(output.as_device_buffer().slice(0, 8)?);
            // Inject the backend workspace's reported status at the exact bridge
            // used immediately after numeric_fp8_bf16_linear_into.
            this.mixed_numeric_completion(poisoned, Err(cuda_error("injected numeric completion")))
        });
        assert!(result.is_err());
        assert_eq!(owner.needs_quarantine(), poisoned);
        assert_eq!(owner.bf16_hold.is_empty(), !poisoned);
        assert_eq!(owner.expert_hold.is_empty(), !poisoned);
        assert!(owner.resident_parameter_bytes() > 0);
    }
    let before = device.operators().allocator_metrics().live_granted_bytes;
    assert!(owner.quiesce().is_err());
    drop(life);
    drop(owner);
    assert_eq!(
        before,
        device.operators().allocator_metrics().live_granted_bytes
    );
}

#[test]
fn mixed_preparation_typed_unknown_is_not_an_ordinary_error() {
    let clean: Result<()> = Err(cuda_error("known-complete preparation rejection"));
    assert!(!mixed_result_unknown(&clean));
    let unknown: Result<()> = Err(crate::decoder::CompletionUnknown::wrap(cuda_error(
        "lost proof",
    )));
    assert!(mixed_result_unknown(&unknown));
    let cleanup: Result<()> = Err(Error::with_cleanup(
        "prepare",
        cuda_error("primary"),
        unknown,
    ));
    assert!(mixed_result_unknown(&cleanup));
}

#[path = "mixed_decoded_claim.rs"]
mod decoded_claim_tests;
