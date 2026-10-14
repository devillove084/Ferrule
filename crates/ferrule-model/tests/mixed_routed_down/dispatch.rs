//! Model dispatch/retirement/Unknown gates. GPU tests require separate authorization.
use super::*;
use crate::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use crate::decoder::MixedSourceRetirement;
use crate::nn::{ModulePath, ParameterDType, ParameterSpec};
use crate::transformer::*;
use ferrule_common::{ParallelRankId, ParallelTopologyId};
use std::cell::Cell;

thread_local! {
    static CANDIDATE_SUBMISSIONS: Cell<usize> = const { Cell::new(0) };
    static INJECT_UNKNOWN: Cell<bool> = const { Cell::new(false) };
}

pub(super) fn after_candidate_submission(result: Result<()>) -> Result<()> {
    result?;
    CANDIDATE_SUBMISSIONS.set(CANDIDATE_SUBMISSIONS.get() + 1);
    if INJECT_UNKNOWN.replace(false) {
        return Err(crate::decoder::CompletionUnknown::wrap(cuda_error(
            "injected typed Unknown after successful candidate submission",
        )));
    }
    Ok(())
}

struct Fixture {
    path: std::path::PathBuf,
    bindings: Vec<BoundParameter>,
    layout: Arc<HybridRankLayout>,
}
impl Fixture {
    fn new(lane: usize) -> Self {
        let norm = || RmsNorm::new(256, 1e-6).unwrap();
        let rope = RotaryEmbedding::new(
            64,
            10000.,
            RotaryPairing::SplitHalf,
            RotaryRegion::Prefix { dimensions: 64 },
            RotaryScaling::None,
        )
        .unwrap();
        let moe = Moe::new(
            256,
            256,
            MoeRouterSpec::new(
                2,
                2,
                RouterScoreFunction::Softmax,
                RouterSelection::TopK,
                true,
                1.,
            )
            .unwrap(),
            false,
        )
        .unwrap();
        let spec = DecoderModelSpec::new(DecoderModelParts {
            architecture: "mixed-down-internal-quality".into(),
            hidden_size: 256,
            vocab_size: 256,
            max_sequence_length: Some(16),
            token_embedding: Embedding::new(256, 256, None).unwrap(),
            layers: vec![
                DecoderLayer::new(
                    0,
                    norm(),
                    Attention::Gqa(GqaAttention::new(256, 4, 2, 64, false, rope).unwrap()),
                    Residual::Add,
                    norm(),
                    FeedForward::Moe(moe),
                    Residual::Add,
                )
                .unwrap(),
            ],
            final_norm: norm(),
            output: Linear::new(256, 256, false).unwrap(),
            tie_word_embeddings: false,
        })
        .unwrap();
        let layout = Arc::new(
            HybridParallelPlan::new_with_execution(
                &spec,
                0..1,
                HybridExpertScope {
                    topology: ParallelTopologyId::new(907),
                    replica: 0,
                    stage: 0,
                },
                vec![ParallelRankId::new(0), ParallelRankId::new(1)],
                HybridRoutedExpertExecution::LocalTensorShards,
                HybridRankPrecision::MixedBf16F32,
            )
            .unwrap()
            .rank_layout(lane)
            .unwrap(),
        );
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "mixed-down-quality-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let mut data = Vec::new();
        let mut slices = Vec::new();
        let mut schema = StateDictSchema::builder();
        let mut mapper = ExactNameMapper::new();
        for expert in 0..2 {
            for (r, role) in [
                TensorRole::RoutedExpertGate,
                TensorRole::RoutedExpertUp,
                TensorRole::RoutedExpertDown,
            ]
            .into_iter()
            .enumerate()
            {
                let name = format!("expert{expert}.p{r}");
                let module = ModulePath::new(&name).unwrap();
                schema
                    .register_with_role(
                        ParameterSpec::new(
                            ParameterId::new((expert * 3 + r + 1) as u64),
                            module.clone(),
                            ParameterDType::F8E4M3,
                            vec![256, 256],
                            ParameterResidency::Expert { layer: 0, expert },
                        )
                        .unwrap()
                        .with_required_scale(ParameterDType::Bf16, vec![2, 2])
                        .unwrap(),
                        role.clone(),
                    )
                    .unwrap();
                for scale in [false, true] {
                    let tensor = format!("{name}.{}", if scale { "scale" } else { "weight" });
                    let offset = data.len();
                    if scale {
                        for _ in 0..4 {
                            data.extend(0x3f80u16.to_le_bytes());
                        }
                    } else {
                        data.extend(vec![0x38; 256 * 256]);
                    }
                    slices.push(CheckpointTensorSlice {
                        name: tensor.clone(),
                        role: role.clone(),
                        path: path.clone(),
                        offset: offset as u64,
                        bytes: (data.len() - offset) as u64,
                        shape: if scale { vec![2, 2] } else { vec![256, 256] },
                        dtype: if scale {
                            CheckpointDType::Bf16
                        } else {
                            CheckpointDType::F8E4M3
                        },
                    });
                    mapper
                        .insert(
                            &tensor,
                            if scale {
                                NameMapping::scale(module.clone())
                            } else {
                                NameMapping::weight(module.clone())
                            },
                        )
                        .unwrap();
                }
            }
        }
        std::fs::write(&path, data).unwrap();
        let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
            .bind_slices(slices)
            .unwrap();
        Self {
            path,
            bindings: bound.parameters().to_vec(),
            layout,
        }
    }
    fn prepare(&self) -> Vec<PreparedLinear> {
        let materializer =
            StateDictMaterializer::for_hybrid_cuda_rank(1 << 20, self.layout.clone(), 4 << 20)
                .unwrap();
        self.bindings
            .iter()
            .map(|b| materializer.prepared_linear(b, b.role().clone()).unwrap())
            .collect()
    }
    fn block(&self, linears: &[PreparedLinear]) -> HybridMixedBlockPlan {
        let parameters = linears
            .iter()
            .map(|p| p.hybrid_rank().unwrap().clone())
            .collect::<Vec<_>>();
        HybridMixedBlockPlan::new(
            self.layout.clone(),
            &parameters,
            2,
            HybridMixedProjectionArithmetic::NativeBf16GemmEx,
            4 << 20,
            64 << 20,
            16 << 20,
        )
        .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn retire(linears: Vec<PreparedLinear>) -> Vec<PreparedLinear> {
    let weak = linears
        .iter()
        .map(|p| {
            let HybridRankStorage::Numeric(payload) = p.hybrid_rank().unwrap().storage() else {
                panic!()
            };
            payload.payload_liveness()
        })
        .collect::<Vec<_>>();
    for p in &linears {
        assert!(
            p.mixed_runtime_binding(&MixedSourceRetirement::default())
                .is_err()
        );
    }
    // Unit-only receipt: GPU callers reach here only AFTER prepare_mixed_rank_block
    // returns with its final synchronous completion proof. No production claim bypass.
    let runtime = linears
        .iter()
        .map(|p| {
            p.mixed_runtime_binding(&MixedSourceRetirement::completed_for_test())
                .unwrap()
        })
        .collect::<Vec<_>>();
    drop(linears);
    assert!(
        weak.iter().all(|w| w.upgrade().is_none()),
        "raw payload alias survived retirement"
    );
    assert!(runtime.iter().all(|p| matches!(
        p.hybrid_rank().unwrap().storage(),
        HybridRankStorage::NumericBinding(_)
    )));
    runtime
}

#[test]
fn retired_metadata_preserves_exact_selector_identity_without_payload_or_source_io() {
    for lane in 0..2 {
        let fixture = Fixture::new(lane);
        let linears = fixture.prepare();
        let block = fixture
            .block(&linears)
            .with_routed_down_m1_candidate(2 * 1024)
            .unwrap();
        let runtime = retire(linears);
        std::fs::remove_file(&fixture.path).unwrap();
        for p in &runtime {
            let rank = p.hybrid_rank().unwrap();
            assert!(p.weight().is_err());
            let HybridRankStorage::NumericBinding(metadata) = rank.storage() else {
                panic!()
            };
            let read = &metadata.reads()[0];
            let [n, k] = read.local_shape();
            let layout = ferrule_backend::cuda_plan::NumericFp8Layout {
                n,
                k,
                row_origin: read.weight_read().rows().start,
                column_origin: read.weight_read().columns().start,
                scale_type: ferrule_backend::cuda_plan::NumericFp8ScaleType::Bf16,
            };
            assert_eq!(
                block
                    .routed_down_m1_candidate_plan(rank.binding(), layout, 1, 1)
                    .is_some(),
                p.role() == &TensorRole::RoutedExpertDown
            );
        }
    }
}

fn authorized_ops(lane: usize) -> Rc<ferrule_backend::cuda::operators::linear::CudaOperators> {
    assert_eq!(
        std::env::var("FERRULE_RUN_MIXED_RANK_GPU").as_deref(),
        Ok("1")
    );
    assert_eq!(
        std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref(),
        Ok(":0:0")
    );
    Rc::new(ferrule_backend::cuda::operators::linear::CudaOperators::new_on_device(lane).unwrap())
}
fn owner(
    op: Rc<ferrule_backend::cuda::operators::linear::CudaOperators>,
    f: &Fixture,
    block: HybridMixedBlockPlan,
    linears: &[PreparedLinear],
) -> CudaStandardDecoderOperators {
    let mut owner = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
        op,
        &f.bindings,
        ExpertCachePolicy::KeepAll,
        0,
        NumericFp8Precision::W8A16Bf16F32Accumulate,
    )
    .unwrap();
    let parameters = linears
        .iter()
        .map(|p| p.hybrid_rank().unwrap().clone())
        .collect::<Vec<_>>();
    owner.prepare_mixed_rank_block(block, &parameters).unwrap();
    owner
}
fn input(op: &ferrule_backend::cuda::operators::linear::CudaOperators, rows: usize) -> Rows {
    rows_bf16(
        RowsShape::new(rows, 256).unwrap(),
        None,
        op.upload_bf16_words(&vec![0x3b80; rows * 256]).unwrap(),
    )
    .unwrap()
}
fn output_words(
    op: &ferrule_backend::cuda::operators::linear::CudaOperators,
    rows: &Rows,
) -> Vec<u16> {
    op.download_bf16_buffer(rows.cuda().unwrap().bf16_buffer().unwrap())
        .unwrap()
        .into_iter()
        .map(crate::execution::bf16_rne_word)
        .collect()
}

#[test]
#[ignore = "separate TP2 GPU authorization; one synthetic routed layer after host retirement, not frozen real acceptance"]
fn retired_candidate_dispatches_only_down_and_preserves_routed_layer_words() {
    for lane in 0..2 {
        let fixture = Fixture::new(lane);
        let linears = fixture.prepare();
        let native = fixture.block(&linears);
        let candidate = native
            .clone()
            .with_routed_down_m1_candidate(2 * 1024)
            .unwrap();
        let op = authorized_ops(lane);
        let mut candidate_owner = owner(op.clone(), &fixture, candidate, &linears);
        let control_op = authorized_ops(lane);
        let mut control = owner(control_op.clone(), &fixture, native, &linears);
        let runtime = retire(linears);
        std::fs::remove_file(&fixture.path).unwrap();
        let image = candidate_owner.mixed_rank_block.as_ref().unwrap();
        for p in &image.projections {
            let Storage::Numeric(segments) = &p.storage else {
                panic!()
            };
            assert_eq!(
                segments[0].candidate_only,
                p.binding.role() == &TensorRole::RoutedExpertDown
            );
        }
        let routes = RouterRoutes::new(1, 2, vec![1, 0], vec![0.25, 0.75]).unwrap();
        CANDIDATE_SUBMISSIONS.set(0);
        op.reset_counters();
        let result = candidate_owner
            .mixed_rank_routed_local(
                0,
                &input(&op, 1),
                &routes,
                HybridRoutedAccumulation::HfEagerBf16ExpertOrder,
            )
            .unwrap();
        assert_eq!(CANDIDATE_SUBMISSIONS.get(), 2);
        assert_eq!(op.counters().artifact_uploads, 0);
        let expected = control
            .mixed_rank_routed_local(
                0,
                &input(&control_op, 1),
                &routes,
                HybridRoutedAccumulation::HfEagerBf16ExpertOrder,
            )
            .unwrap();
        assert_eq!(
            output_words(&op, &result),
            output_words(&control_op, &expected)
        );
        // Original M2, each bucket M1: must execute native, not the candidate.
        let split = RouterRoutes::new(2, 1, vec![1, 0], vec![1., 1.]).unwrap();
        let result = candidate_owner
            .mixed_rank_routed_local(
                0,
                &input(&op, 2),
                &split,
                HybridRoutedAccumulation::HfEagerBf16ExpertOrder,
            )
            .unwrap();
        assert_eq!(CANDIDATE_SUBMISSIONS.get(), 2);
        let expected = control
            .mixed_rank_routed_local(
                0,
                &input(&control_op, 2),
                &split,
                HybridRoutedAccumulation::HfEagerBf16ExpertOrder,
            )
            .unwrap();
        assert_eq!(
            output_words(&op, &result),
            output_words(&control_op, &expected)
        );
        assert!(runtime.iter().all(|p| p.weight().is_err()));
        assert!(!candidate_owner.needs_quarantine());
    }
}

#[test]
#[ignore = "separate GPU authorization; typed Unknown injected after actual fused submission, retains image/operands"]
fn retired_candidate_unknown_quarantines_without_native_retry_or_release() {
    let fixture = Fixture::new(0);
    let linears = fixture.prepare();
    let block = fixture
        .block(&linears)
        .with_routed_down_m1_candidate(2 * 1024)
        .unwrap();
    let op = authorized_ops(0);
    let mut owner = owner(op.clone(), &fixture, block, &linears);
    let _runtime = retire(linears);
    std::fs::remove_file(&fixture.path).unwrap();
    let x = input(&op, 1);
    let routes = RouterRoutes::new(1, 2, vec![1, 0], vec![0.25, 0.75]).unwrap();
    CANDIDATE_SUBMISSIONS.set(0);
    INJECT_UNKNOWN.set(true);
    let result = owner.mixed_rank_routed_local(
        0,
        &x,
        &routes,
        HybridRoutedAccumulation::HfEagerBf16ExpertOrder,
    );
    assert!(mixed_result_unknown(&result));
    assert!(!INJECT_UNKNOWN.get(), "did not reach candidate submission");
    assert_eq!(
        CANDIDATE_SUBMISSIONS.get(),
        1,
        "must stop at the first Unknown"
    );
    assert!(owner.needs_quarantine());
    assert!(!owner.expert_hold.is_empty() && !owner.bf16_hold.is_empty());
    assert!(owner.resident_parameter_bytes() > 0);
    let launches = op.counters().compute_kernel_launches;
    let live = op.allocator_metrics().live_granted_bytes;
    assert!(
        owner
            .mixed_rank_routed_local(
                0,
                &x,
                &routes,
                HybridRoutedAccumulation::HfEagerBf16ExpertOrder
            )
            .is_err()
    );
    assert!(owner.quiesce().is_err());
    assert_eq!(op.counters().compute_kernel_launches, launches);
    assert_eq!(op.allocator_metrics().live_granted_bytes, live);
    drop(owner);
    assert_eq!(op.allocator_metrics().live_granted_bytes, live);
}
