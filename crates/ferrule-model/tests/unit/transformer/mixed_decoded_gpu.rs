//! Small exact model gates. Compile hidden; execute only with the sole GPU owner.
use super::*;
use crate::transformer::hybrid_parallel::HybridMixedDecodedCacheBudget;
use crate::transformer::hybrid_parallel::mixed_decoded::decoded_descriptors;
use std::cell::Cell;
thread_local! {
    static INJECT: Cell<u8> = const { Cell::new(0) };
    static VIEW_FAULT: Cell<Option<(usize, u8)>> = const { Cell::new(None) };
}
pub(super) fn before_arena_view() -> Result<()> {
    let Some((remaining, fault)) = VIEW_FAULT.take() else {
        return Ok(());
    };
    if remaining != 0 {
        VIEW_FAULT.set(Some((remaining - 1, fault)));
        return Ok(());
    }
    match fault {
        1 => Err(cuda_error("injected arena view installation failure")),
        2 => Err(crate::decoder::CompletionUnknown::wrap(cuda_error(
            "injected arena view installation Unknown",
        ))),
        3 => panic!("injected arena view installation unwind"),
        _ => unreachable!(),
    }
}
pub(super) fn after_decode_submission(result: Result<()>) -> Result<()> {
    result?;
    match INJECT.replace(0) {
        1 => Err(crate::decoder::CompletionUnknown::wrap(cuda_error(
            "injected decoded completion Unknown",
        ))),
        2 => panic!("injected decoded submission unwind"),
        3 => Err(cuda_error("injected decoded submission failure")),
        _ => Ok(()),
    }
}
// A dedicated copy of the existing small source fixture lives in this module.
use crate::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use crate::decoder::MixedSourceRetirement;
use crate::nn::{ModulePath, ParameterDType, ParameterSpec};
use crate::transformer::*;
use ferrule_common::{ParallelRankId, ParallelTopologyId};
struct Fixture {
    path: std::path::PathBuf,
    bindings: Vec<BoundParameter>,
    layout: Arc<HybridRankLayout>,
}
impl Fixture {
    fn new(lane: usize) -> Self {
        Self::with_geometry(lane, 256, 1)
    }
    fn with_geometry(lane: usize, width: usize, layers: usize) -> Self {
        let norm = || RmsNorm::new(width, 1e-6).unwrap();
        let rope = RotaryEmbedding::new(
            64,
            10000.,
            RotaryPairing::SplitHalf,
            RotaryRegion::Prefix { dimensions: 64 },
            RotaryScaling::None,
        )
        .unwrap();
        let moe = Moe::new(
            width,
            width,
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
        .unwrap()
        .with_shared_expert(SwiGlu::new(width, width, false).unwrap())
        .unwrap();
        let spec = DecoderModelSpec::new(DecoderModelParts {
            architecture: "mixed-down-internal-quality".into(),
            hidden_size: width,
            vocab_size: width,
            max_sequence_length: Some(16),
            token_embedding: Embedding::new(width, width, None).unwrap(),
            layers: (0..layers)
                .map(|layer| {
                    DecoderLayer::new(
                        layer,
                        norm(),
                        Attention::Gqa(
                            GqaAttention::new(
                                width,
                                width / 64,
                                width / 128,
                                64,
                                false,
                                rope.clone(),
                            )
                            .unwrap(),
                        ),
                        Residual::Add,
                        norm(),
                        FeedForward::Moe(moe.clone()),
                        Residual::Add,
                    )
                    .unwrap()
                })
                .collect(),
            final_norm: norm(),
            output: Linear::new(width, width, false).unwrap(),
            tie_word_embeddings: false,
        })
        .unwrap();
        let layout = Arc::new(
            HybridParallelPlan::new_with_execution(
                &spec,
                0..layers,
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
        for layer in 0..layers {
            for expert in 0..3 {
                let roles = if expert == 2 {
                    [
                        TensorRole::SharedExpertGate,
                        TensorRole::SharedExpertUp,
                        TensorRole::SharedExpertDown,
                    ]
                } else {
                    [
                        TensorRole::RoutedExpertGate,
                        TensorRole::RoutedExpertUp,
                        TensorRole::RoutedExpertDown,
                    ]
                };
                for (r, role) in roles.into_iter().enumerate() {
                    let name = format!("layer{layer}.expert{expert}.p{r}");
                    let module = ModulePath::new(&name).unwrap();
                    schema
                        .register_with_role(
                            ParameterSpec::new(
                                ParameterId::new((layer * 9 + expert * 3 + r + 1) as u64),
                                module.clone(),
                                ParameterDType::F8E4M3,
                                vec![width, width],
                                if expert == 2 {
                                    ParameterResidency::Layer { layer }
                                } else {
                                    ParameterResidency::Expert { layer, expert }
                                },
                            )
                            .unwrap()
                            .with_required_scale(
                                ParameterDType::Bf16,
                                vec![width / 128, width / 128],
                            )
                            .unwrap(),
                            role.clone(),
                        )
                        .unwrap();
                    for scale in [false, true] {
                        let tensor = format!("{name}.{}", if scale { "scale" } else { "weight" });
                        let offset = data.len();
                        if scale {
                            for _ in 0..(width / 128).pow(2) {
                                data.extend(0x3f80u16.to_le_bytes());
                            }
                        } else {
                            data.extend(vec![0x38; width * width]);
                        }
                        slices.push(CheckpointTensorSlice {
                            name: tensor.clone(),
                            role: role.clone(),
                            path: path.clone(),
                            offset: offset as u64,
                            bytes: (data.len() - offset) as u64,
                            shape: if scale {
                                vec![width / 128, width / 128]
                            } else {
                                vec![width, width]
                            },
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
            StateDictMaterializer::for_hybrid_cuda_rank(16 << 20, self.layout.clone(), 128 << 20)
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
            5,
            HybridMixedProjectionArithmetic::NativeBf16GemmEx,
            128 << 20,
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
    if let Some(snapshot) = owner.mixed_decoded_cache_snapshot() {
        assert!(snapshot.published && !snapshot.quarantined);
        assert_eq!(snapshot.phase, MixedDecodedCachePhase::Idle);
        assert_eq!(snapshot.pinned_grants, snapshot.pin_target_grants);
        assert_eq!(snapshot.pending_allocation_count, 0);
        assert_eq!(snapshot.pending_lease_count, 0);
        assert_eq!(snapshot.borrowed_key_count, 0);
        assert_eq!(snapshot.segment_slack_bytes, 0);
        assert_eq!(snapshot.pending_segment_slack_bytes, 0);
        assert_eq!(snapshot.arena_backing_grants, snapshot.capacity_grants);
        assert_eq!(snapshot.physical_current_bytes, snapshot.capacity_grants);
        assert_eq!(snapshot.physical_peak_bytes, snapshot.capacity_grants);
        assert_eq!(
            snapshot.driver_allocation_growth_bytes,
            snapshot.capacity_grants
        );
    }
    owner
}
fn input(op: &ferrule_backend::cuda::operators::linear::CudaOperators, rows: usize) -> Rows {
    input_width(op, rows, 256)
}
fn input_width(
    op: &ferrule_backend::cuda::operators::linear::CudaOperators,
    rows: usize,
    width: usize,
) -> Rows {
    rows_bf16(
        RowsShape::new(rows, width).unwrap(),
        None,
        op.upload_bf16_words(&vec![0x3b80; rows * width]).unwrap(),
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

fn cached_block(f: &Fixture, linears: &[PreparedLinear]) -> HybridMixedBlockPlan {
    cached_block_capacity(f, linears, 128 << 20)
}
fn cached_block_capacity(
    f: &Fixture,
    linears: &[PreparedLinear],
    capacity: usize,
) -> HybridMixedBlockPlan {
    let block = f.block(linears);
    let ds = decoded_descriptors(&f.layout, &block.bindings).unwrap();
    let budget = HybridMixedDecodedCacheBudget::new(capacity, &ds).unwrap();
    block.with_decoded_budget(budget).unwrap()
}
fn tokens(owner: &CudaStandardDecoderOperators, expert: usize) -> [CudaMixedRankLinear; 3] {
    tokens_at(owner, 0, expert)
}
fn tokens_at(
    owner: &CudaStandardDecoderOperators,
    layer: usize,
    expert: usize,
) -> [CudaMixedRankLinear; 3] {
    let image = owner.mixed_rank_block.as_ref().unwrap();
    [
        TensorRole::RoutedExpertGate,
        TensorRole::RoutedExpertUp,
        TensorRole::RoutedExpertDown,
    ]
    .map(|role| {
        let slot = image
            .projections
            .iter()
            .position(|p| {
                p.binding.role() == &role
                    && p.binding.residency() == &ParameterResidency::Expert { layer, expert }
            })
            .unwrap();
        CudaMixedRankLinear {
            generation: owner.image_generation(),
            slot,
        }
    })
}
fn execute(owner: &mut CudaStandardDecoderOperators, expert: usize, x: &Rows) -> Result<Rows> {
    let t = tokens(owner, expert);
    owner.mixed_rank_swiglu_local([&t[0], &t[1], &t[2]], x)
}
#[test]
#[ignore = "GPU-owner authorization only; exact resident pair/single M1/M5 cold/hot and fitting inventory"]
fn decoded_pair_single_cold_hot_fit_exact_and_no_cache_dma() {
    for lane in 0..2 {
        for rows in [1, 5] {
            let f = Fixture::new(lane);
            let linears = f.prepare();
            let op = authorized_ops(lane);
            let nop = authorized_ops(lane);
            let mut c = owner(op.clone(), &f, cached_block(&f, &linears), &linears);
            let mut native = owner(nop.clone(), &f, f.block(&linears), &linears);
            let _runtime = retire(linears);
            std::fs::remove_file(&f.path).unwrap();
            let x = input(&op, rows);
            let nx = input(&nop, rows);
            let expected = execute(&mut native, 0, &nx).unwrap();
            let before = op.counters();
            let initial = custody(&c).physical_stats();
            let cold = execute(&mut c, 0, &x).unwrap();
            let cold_counters = op.counters();
            let cold_backing = custody(&c).physical_stats();
            let hot = execute(&mut c, 0, &x).unwrap();
            let hot_counters = op.counters();
            assert_eq!(hot_counters.artifact_uploads, before.artifact_uploads);
            assert_eq!(
                hot_counters.host_to_device_bytes,
                before.host_to_device_bytes
            );
            assert_eq!(
                hot_counters.device_to_host_bytes,
                before.device_to_host_bytes
            );
            assert_eq!(
                cold_counters.stream_wide_syncs - before.stream_wide_syncs,
                hot_counters.stream_wide_syncs - cold_counters.stream_wide_syncs
            );
            assert_eq!(cold_backing, custody(&c).physical_stats());
            assert_eq!(
                cold_counters.compute_kernel_launches - before.compute_kernel_launches,
                hot_counters.compute_kernel_launches - cold_counters.compute_kernel_launches + 3
            );
            assert_eq!(output_words(&op, &cold), output_words(&nop, &expected));
            assert_eq!(output_words(&op, &hot), output_words(&nop, &expected));
            for expert in [1, 0] {
                let got = execute(&mut c, expert, &x).unwrap();
                let expected = execute(&mut native, expert, &nx).unwrap();
                assert_eq!(output_words(&op, &got), output_words(&nop, &expected));
            }
            let after = custody(&c).physical_stats();
            assert_eq!(initial.reserved_grants, after.reserved_grants);
            assert_eq!(initial.driver_growth, after.driver_growth);
            assert_eq!(after.reuses, 0);
            assert_eq!(after.fallbacks, 0);
            assert_eq!(custody(&c).stats().evictions, 0);
            assert_eq!(after.pooled, 0);
            assert!(!c.needs_quarantine());
        }
    }
}
#[test]
#[ignore = "GPU-owner authorization only; typed Unknown/panic after real pair decode, no release/retry"]
fn decoded_pair_unknown_and_panic_retain_physical_backing() {
    for fault in [1, 2] {
        let f = Fixture::new(0);
        let linears = f.prepare();
        let op = authorized_ops(0);
        let mut c = owner(op.clone(), &f, cached_block(&f, &linears), &linears);
        let _runtime = retire(linears);
        std::fs::remove_file(&f.path).unwrap();
        let x = input(&op, 1);
        INJECT.set(fault);
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(&mut c, 0, &x)));
        assert!(result.is_err() || result.unwrap().is_err());
        assert!(c.needs_quarantine());
        let frozen = c
            .mixed_rank_block
            .as_ref()
            .unwrap()
            .decoded
            .as_ref()
            .unwrap()
            .physical_stats();
        assert_eq!(frozen.unknown, frozen.arena_backing);
        assert_eq!(frozen.arena_backing, 128 << 20);
        let live = op.allocator_metrics().live_granted_bytes;
        let launches = op.counters().compute_kernel_launches;
        assert!(execute(&mut c, 0, &x).is_err());
        assert!(c.quiesce().is_err());
        assert_eq!(op.counters().compute_kernel_launches, launches);
        assert_eq!(
            c.mixed_rank_block
                .as_ref()
                .unwrap()
                .decoded
                .as_ref()
                .unwrap()
                .physical_stats(),
            frozen
        );
        drop(c);
        assert_eq!(op.allocator_metrics().live_granted_bytes, live);
    }
}

#[test]
fn decoded_fixture_descriptors_preserve_pair_and_separate_fixed_roles() {
    for lane in 0..2 {
        let f = Fixture::new(lane);
        let linears = f.prepare();
        let plan = cached_block(&f, &linears);
        let (budget, descriptors) = plan.decoded.as_ref().unwrap();
        assert_eq!(descriptors.len(), 7);
        assert_eq!(
            descriptors.iter().filter(|d| d.operands.len() == 2).count(),
            2
        );
        assert_eq!(
            descriptors
                .iter()
                .filter(|d| d.routed_layer.is_none())
                .count(),
            3
        );
        assert_eq!(budget.pinned_grants, 3 * 65536);
        assert_eq!(budget.routed_working_set_grants, 2 * (131072 + 65536));
        let identities = descriptors
            .iter()
            .map(|d| d.identity.clone())
            .collect::<Vec<_>>();
        let _runtime = retire(linears);
        std::fs::remove_file(&f.path).unwrap();
        let after = decoded_descriptors(&f.layout, &plan.bindings).unwrap();
        assert_eq!(
            identities,
            after.iter().map(|d| d.identity.clone()).collect::<Vec<_>>()
        );
    }
}
#[test]
#[ignore = "GPU-owner authorization only; preparation pins and separate shared M1/M5 exact"]
fn decoded_fixed_pins_prepare_once_and_shared_stays_separate() {
    for lane in 0..2 {
        let f = Fixture::new(lane);
        let linears = f.prepare();
        let op = authorized_ops(lane);
        let nop = authorized_ops(lane);
        let mut c = owner(op.clone(), &f, cached_block(&f, &linears), &linears);
        let mut native = owner(nop.clone(), &f, f.block(&linears), &linears);
        let _runtime = retire(linears);
        std::fs::remove_file(&f.path).unwrap();
        let shared = |owner: &CudaStandardDecoderOperators| {
            let image = owner.mixed_rank_block.as_ref().unwrap();
            [
                TensorRole::SharedExpertGate,
                TensorRole::SharedExpertUp,
                TensorRole::SharedExpertDown,
            ]
            .map(|role| CudaMixedRankLinear {
                generation: owner.image_generation(),
                slot: image
                    .projections
                    .iter()
                    .position(|p| p.binding.role() == &role)
                    .unwrap(),
            })
        };
        for rows in [1, 5] {
            let x = input(&op, rows);
            let nx = input(&nop, rows);
            let t = shared(&c);
            let nt = shared(&native);
            let before = c
                .mixed_rank_block
                .as_ref()
                .unwrap()
                .decoded
                .as_ref()
                .unwrap()
                .physical_stats();
            let bc = op.counters();
            let nc = nop.counters();
            let got = c
                .mixed_rank_swiglu_local([&t[0], &t[1], &t[2]], &x)
                .unwrap();
            let expected = native
                .mixed_rank_swiglu_local([&nt[0], &nt[1], &nt[2]], &nx)
                .unwrap();
            assert_eq!(
                nop.counters().compute_kernel_launches - nc.compute_kernel_launches,
                op.counters().compute_kernel_launches - bc.compute_kernel_launches + 6
            );
            assert_eq!(op.counters().host_to_device_bytes, bc.host_to_device_bytes);
            assert_eq!(op.counters().device_to_host_bytes, bc.device_to_host_bytes);
            assert_eq!(
                before,
                c.mixed_rank_block
                    .as_ref()
                    .unwrap()
                    .decoded
                    .as_ref()
                    .unwrap()
                    .physical_stats()
            );
            assert_eq!(output_words(&op, &got), output_words(&nop, &expected));
        }
    }
}

fn custody(owner: &CudaStandardDecoderOperators) -> &DecodedCustody<RefCell<CudaBf16Buffer>> {
    owner
        .mixed_rank_block
        .as_ref()
        .unwrap()
        .decoded
        .as_ref()
        .unwrap()
}

fn routed_classes(owner: &CudaStandardDecoderOperators) -> Vec<(usize, Vec<DecodedDescriptor>)> {
    let image = owner.mixed_rank_block.as_ref().unwrap();
    let (_, descriptors) = image.plan.decoded.as_ref().unwrap();
    custody(owner)
        .plan()
        .classes
        .iter()
        .map(|class| {
            (
                class.count,
                descriptors
                    .iter()
                    .filter(|d| {
                        d.routed_layer.is_some()
                            && (d.elements(), d.grants) == (class.elements, class.slot_bytes)
                    })
                    .cloned()
                    .collect(),
            )
        })
        .collect()
}

fn decode_completed(owner: &mut CudaStandardDecoderOperators, descriptor: &DecodedDescriptor) {
    let before = owner.ops.allocator_metrics();
    owner
        .synchronous(|this| {
            assert!(this.mixed_decoded_weight(descriptor)?.is_some());
            Ok(())
        })
        .unwrap();
    let after = owner.ops.allocator_metrics();
    assert_eq!(after.allocation_requests, before.allocation_requests);
    assert_eq!(after.granted_bytes, before.granted_bytes);
    assert_eq!(
        after.driver_allocation_bytes,
        before.driver_allocation_bytes
    );
}

#[test]
#[ignore = "GPU-owner authorization only; every planned slot decodes without allocating and remains hot"]
fn decoded_arena_fit_working_set_hot_zero_eviction_decode_fallback() {
    let f = Fixture::new(0);
    let linears = f.prepare();
    let op = authorized_ops(0);
    let mut c = owner(op.clone(), &f, cached_block(&f, &linears), &linears);
    let _runtime = retire(linears);
    std::fs::remove_file(&f.path).unwrap();
    let initial = custody(&c).physical_stats();
    assert_eq!(initial.arena_backing, 128 << 20);
    assert_eq!(initial.reserved_grants, initial.arena_backing);
    assert_eq!(
        initial.active + initial.free + initial.unused_tail,
        initial.arena_backing
    );
    assert_eq!(c.mixed_decoded_backing_counts()[5], 0);
    let classes = routed_classes(&c);
    for (count, descriptors) in &classes {
        assert_eq!(*count, descriptors.len());
        for descriptor in descriptors {
            decode_completed(&mut c, descriptor);
        }
    }
    let hot = op.counters();
    for (_, descriptors) in &classes {
        for descriptor in descriptors {
            decode_completed(&mut c, descriptor);
        }
    }
    assert_eq!(
        op.counters().compute_kernel_launches,
        hot.compute_kernel_launches
    );
    assert_eq!(custody(&c).stats().evictions, 0);
    let end = custody(&c).physical_stats();
    assert_eq!(end.fallbacks, 0);
    assert_eq!(end.reuses, 0);
    assert_eq!(end.free, 0);
    assert_eq!(end.driver_growth, initial.driver_growth);
    assert_eq!(end.arena_backing, initial.arena_backing);
}

#[test]
#[ignore = "GPU-owner authorization only; real exact-class full LRU, fair quotas, M1/M5 reload"]
fn decoded_arena_true_class_full_lru_and_size_class_fairness() {
    for lane in 0..2 {
        let f = Fixture::with_geometry(lane, 2048, 3);
        let linears = f.prepare();
        let op = authorized_ops(lane);
        let nop = authorized_ops(lane);
        let mut c = owner(
            op.clone(),
            &f,
            cached_block_capacity(&f, &linears, 64 << 20),
            &linears,
        );
        let mut native = owner(nop.clone(), &f, f.block(&linears), &linears);
        let _runtime = retire(linears);
        std::fs::remove_file(&f.path).unwrap();
        let classes = routed_classes(&c);
        assert_eq!(classes.len(), 2);
        // Nine fixed 4 MiB slots leave 28 MiB: two 8 MiB pairs and
        // three 4 MiB down slots. Neither class can steal the other's slots.
        assert_eq!(
            classes
                .iter()
                .map(|(n, ds)| (ds[0].grants, *n))
                .collect::<Vec<_>>(),
            vec![(4 << 20, 3), (8 << 20, 2)]
        );
        let initial = custody(&c).physical_stats();
        for (count, descriptors) in &classes {
            assert!(*count >= 2 && *count < descriptors.len());
            for descriptor in &descriptors[..*count] {
                decode_completed(&mut c, descriptor);
            }
        }
        for (count, descriptors) in &classes {
            let before = custody(&c).stats().evictions;
            decode_completed(&mut c, &descriptors[0]); // Make index 1 the completed LRU.
            decode_completed(&mut c, &descriptors[*count]);
            assert_eq!(custody(&c).stats().evictions, before + 1);
            let hot = op.counters().compute_kernel_launches;
            decode_completed(&mut c, &descriptors[0]);
            assert_eq!(op.counters().compute_kernel_launches, hot);
            decode_completed(&mut c, &descriptors[1]);
            assert!(op.counters().compute_kernel_launches > hot);
            assert_eq!(custody(&c).stats().evictions, before + 2);
        }
        assert_eq!(custody(&c).physical_stats().reuses, 4);
        assert_eq!(custody(&c).physical_stats().fallbacks, 0);
        for rows in [1, 5] {
            let x = input_width(&op, rows, 2048);
            let nx = input_width(&nop, rows, 2048);
            for layer in [0, 1, 2, 0] {
                let t = tokens_at(&c, layer, 0);
                let nt = tokens_at(&native, layer, 0);
                let got = c
                    .mixed_rank_swiglu_local([&t[0], &t[1], &t[2]], &x)
                    .unwrap();
                let expected = native
                    .mixed_rank_swiglu_local([&nt[0], &nt[1], &nt[2]], &nx)
                    .unwrap();
                assert_eq!(output_words(&op, &got), output_words(&nop, &expected));
            }
        }
        let after = custody(&c).physical_stats();
        assert_eq!(after.arena_backing, initial.arena_backing);
        assert_eq!(after.driver_growth, initial.driver_growth);
        assert!(!c.needs_quarantine());
    }
}

#[test]
#[ignore = "GPU-owner authorization only; real class-full borrowed slots fall back before decode"]
fn decoded_physical_full_native_fallback_before_submit_exact() {
    for lane in 0..2 {
        let f = Fixture::with_geometry(lane, 2048, 3);
        let linears = f.prepare();
        let op = authorized_ops(lane);
        let nop = authorized_ops(lane);
        let mut c = owner(
            op.clone(),
            &f,
            cached_block_capacity(&f, &linears, 64 << 20),
            &linears,
        );
        let mut native = owner(nop.clone(), &f, f.block(&linears), &linears);
        let _runtime = retire(linears);
        std::fs::remove_file(&f.path).unwrap();
        let classes = routed_classes(&c);
        let initial = custody(&c).physical_stats();
        for rows in [1, 5] {
            let x = input_width(&op, rows, 2048);
            let nx = input_width(&nop, rows, 2048);
            let nc = nop.counters();
            let expected = execute(&mut native, 0, &nx).unwrap();
            let native_launches =
                nop.counters().compute_kernel_launches - nc.compute_kernel_launches;
            let got = c
                .synchronous(|this| {
                    // Exclude layer 0/expert 0's pair and down (first in each class).
                    // Dropping local views does NOT complete their outer leases.
                    for (count, descriptors) in &classes {
                        for descriptor in &descriptors[1..=*count] {
                            assert!(this.mixed_decoded_weight(descriptor)?.is_some());
                        }
                    }
                    let before = this.ops.counters();
                    let evictions = custody(this).stats().evictions;
                    let got = execute(this, 0, &x)?;
                    let after = this.ops.counters();
                    assert_eq!(
                        after.compute_kernel_launches - before.compute_kernel_launches,
                        native_launches
                    );
                    assert_eq!(after.host_to_device_bytes, before.host_to_device_bytes);
                    assert_eq!(after.device_to_host_bytes, before.device_to_host_bytes);
                    assert_eq!(custody(this).stats().evictions, evictions);
                    Ok(got)
                })
                .unwrap();
            assert_eq!(output_words(&op, &got), output_words(&nop, &expected));
        }
        let after = custody(&c).physical_stats();
        assert_eq!(after.fallbacks - initial.fallbacks, 4);
        assert_eq!(after.arena_backing, initial.arena_backing);
        assert_eq!(after.driver_growth, initial.driver_growth);
        assert_eq!(c.mixed_decoded_backing_counts()[5], 0);
        assert!(!c.needs_quarantine());
    }
}

#[test]
#[ignore = "GPU-owner authorization only; escaped view quarantines and retains the entire arena"]
fn decoded_arena_alias_retains_whole_root() {
    let f = Fixture::new(0);
    let linears = f.prepare();
    let op = authorized_ops(0);
    let mut c = owner(op.clone(), &f, cached_block(&f, &linears), &linears);
    let descriptor = routed_classes(&c)[0].1[0].clone();
    let mut alias = None;
    assert!(
        c.synchronous(|this| {
            alias = this.mixed_decoded_weight(&descriptor)?;
            Ok(())
        })
        .is_err()
    );
    assert!(c.needs_quarantine());
    let frozen = custody(&c).physical_stats();
    assert_eq!(frozen.unknown, 128 << 20);
    assert_eq!(frozen.unknown, frozen.arena_backing);
    let live = op.allocator_metrics().live_granted_bytes;
    drop(alias);
    assert!(c.quiesce().is_err());
    drop(c);
    assert_eq!(op.allocator_metrics().live_granted_bytes, live);
}

#[test]
fn decoded_fixture_real_class_inventory_and_same_config_plan() {
    for lane in 0..2 {
        let f = Fixture::with_geometry(lane, 2048, 3);
        let linears = f.prepare();
        let block = cached_block_capacity(&f, &linears, 64 << 20);
        let (budget, descriptors) = block.decoded.as_ref().unwrap();
        let cache = DecodedCustody::<()>::new(*budget, 1, descriptors).unwrap();
        assert_eq!(budget.pinned_grants, 36 << 20);
        assert_eq!(budget.routed_working_set_grants, 24 << 20);
        assert_eq!(cache.plan().arena_grants, 64 << 20);
        assert_eq!(cache.plan().fixed.len(), 9);
        assert_eq!(
            cache
                .plan()
                .classes
                .iter()
                .map(|c| (c.slot_bytes, c.count, c.layer_minimum))
                .collect::<Vec<_>>(),
            vec![(4 << 20, 3, 2), (8 << 20, 2, 2)]
        );
        assert_eq!(cache.plan().unused_bytes, 0);
        assert_eq!(cache.plan().slot_padding_bytes, 0);
        let identities = descriptors
            .iter()
            .map(|d| &d.identity)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(identities.len(), 21);
        assert!(descriptors.iter().all(|d| !d.identity.parts.is_empty()));
        // Compare only the very same admitted block/descriptors, not a second
        // shape, page count, source set or owner configuration.
        let admitted = block.decoded_arena.as_deref().unwrap();
        assert_eq!(cache.plan().arena_grants, admitted.arena_grants);
        assert_eq!(cache.plan().routed_slot_bytes, admitted.routed_slot_bytes);
        assert_eq!(cache.plan().unused_bytes, admitted.unused_bytes);
        assert_eq!(cache.plan().slot_padding_bytes, admitted.slot_padding_bytes);
        assert_eq!(cache.plan().fixed.len(), admitted.fixed.len());
        for (actual, planned) in cache.plan().fixed.iter().zip(&admitted.fixed) {
            assert_eq!(
                (
                    actual.descriptor,
                    actual.offset_words,
                    actual.elements,
                    actual.slot_bytes
                ),
                (
                    planned.descriptor,
                    planned.offset_words,
                    planned.elements,
                    planned.slot_bytes
                )
            );
        }
        assert_eq!(cache.plan().classes.len(), admitted.classes.len());
        for (actual, planned) in cache.plan().classes.iter().zip(&admitted.classes) {
            assert_eq!(
                (
                    actual.offset_words,
                    actual.elements,
                    actual.slot_bytes,
                    actual.count,
                    actual.layer_minimum
                ),
                (
                    planned.offset_words,
                    planned.elements,
                    planned.slot_bytes,
                    planned.count,
                    planned.layer_minimum
                )
            );
        }
    }
}

#[test]
#[ignore = "GPU-owner authorization only; fixed decode Unknown/unwind never publishes and retains arena"]
fn decoded_fixed_pin_unknown_and_panic_never_publish_arena() {
    for fault in [1, 2] {
        let f = Fixture::new(0);
        let linears = f.prepare();
        let op = authorized_ops(0);
        let block = cached_block(&f, &linears);
        let mut c = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
            op.clone(),
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
        INJECT.set(fault);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            c.prepare_mixed_rank_block(block, &parameters)
        }));
        assert!(result.is_err() || result.unwrap().is_err());
        let snapshot = c.mixed_decoded_cache_snapshot().unwrap();
        assert!(!snapshot.published);
        assert!(snapshot.quarantined);
        assert_eq!(snapshot.phase, MixedDecodedCachePhase::Quarantined);
        assert_eq!(snapshot.arena_backing_grants, 128 << 20);
        assert_eq!(
            snapshot.unknown_reserved_bytes,
            snapshot.arena_backing_grants
        );
        assert!(c.needs_quarantine());
        let live = op.allocator_metrics().live_granted_bytes;
        assert!(c.quiesce().is_err());
        drop(c);
        assert_eq!(op.allocator_metrics().live_granted_bytes, live);
    }
}

#[test]
#[ignore = "GPU-owner authorization only; root retained across first/partial view error, Unknown and panic"]
fn decoded_uninitialized_arena_view_faults_retain_whole_root() {
    const ARENA: usize = 64 << 20;
    for installed in [0, 1] {
        for fault in [1, 2, 3] {
            let f = Fixture::new(0);
            let linears = f.prepare();
            let op = authorized_ops(0);
            let block = cached_block_capacity(&f, &linears, ARENA);
            let installed_pins: usize = block.decoded_arena.as_ref().unwrap().fixed[..installed]
                .iter()
                .map(|slot| slot.slot_bytes)
                .sum();
            let mut c = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
                op.clone(),
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
            VIEW_FAULT.set(Some((installed, fault)));
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                c.prepare_mixed_rank_block(block, &parameters)
            }));
            assert!(result.is_err() || result.unwrap().is_err());
            assert_eq!(VIEW_FAULT.get(), None);
            let snapshot = c.mixed_decoded_cache_snapshot().unwrap();
            assert!(!snapshot.published);
            assert!(snapshot.quarantined && c.needs_quarantine());
            assert_eq!(snapshot.phase, MixedDecodedCachePhase::Quarantined);
            assert_eq!(snapshot.arena_backing_grants, ARENA);
            assert_eq!(snapshot.driver_allocation_growth_bytes, ARENA);
            assert_eq!(snapshot.unknown_reserved_bytes, ARENA);
            assert_eq!(snapshot.pinned_grants, installed_pins);
            assert_eq!(snapshot.pending_allocation_count, 0);
            assert!(c.mixed_rank_block.as_ref().unwrap().projections.is_empty());
            let live = op.allocator_metrics().live_granted_bytes;
            assert!(c.quiesce().is_err());
            drop(c);
            assert_eq!(op.allocator_metrics().live_granted_bytes, live);
        }
    }
}

#[test]
#[ignore = "GPU-owner authorization only; failed first fixed decode stays unready and retained until known completion"]
fn decoded_first_fixed_failure_retained_until_completion_not_ready() {
    let f = Fixture::new(0);
    let linears = f.prepare();
    let op = authorized_ops(0);
    let block = cached_block(&f, &linears);
    let fixed = block
        .decoded
        .as_ref()
        .unwrap()
        .1
        .iter()
        .find(|d| d.routed_layer.is_none())
        .unwrap()
        .clone();
    let parameters = linears
        .iter()
        .map(|p| p.hybrid_rank().unwrap().clone())
        .collect::<Vec<_>>();
    let mut c = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
        op,
        &f.bindings,
        ExpertCachePolicy::KeepAll,
        0,
        NumericFp8Precision::W8A16Bf16F32Accumulate,
    )
    .unwrap();
    c.validate_mixed_preparation_owner().unwrap();
    c.mixed_decoded_enabled = true;
    INJECT.set(3);
    let result = c.mixed_preparation(|this| {
        let result = this.prepare_mixed_rank_block_pending(block, &parameters);
        assert!(result.is_err());
        assert_eq!(INJECT.get(), 0);
        let snapshot = this.mixed_decoded_cache_snapshot().unwrap();
        assert!(!snapshot.published && !snapshot.quarantined);
        assert_eq!(snapshot.phase, MixedDecodedCachePhase::Preparing);
        assert_eq!(snapshot.arena_backing_grants, 128 << 20);
        assert_eq!(
            snapshot.driver_allocation_growth_bytes,
            snapshot.arena_backing_grants
        );
        assert_eq!(snapshot.borrowed_key_count, 1);
        // Submission is not readiness. The rejected consumer must not launch.
        let before = this.ops.counters();
        assert!(this.mixed_decoded_weight(&fixed).is_err());
        assert_eq!(this.ops.counters().kernel_launches, before.kernel_launches);
        result
    });
    assert!(result.is_err());
    // Only the existing outer completion permits healthy rollback/release.
    assert!(!c.needs_quarantine());
    assert!(c.mixed_decoded_cache_snapshot().is_none());
}

#[test]
#[ignore = "GPU-owner authorization only; uninitialized single/pair slots reject consumers until full decode"]
fn decoded_uninitialized_slots_reject_consumers_until_ready() {
    let f = Fixture::new(0);
    let linears = f.prepare();
    let op = authorized_ops(0);
    let mut c = owner(op.clone(), &f, cached_block(&f, &linears), &linears);
    let _runtime = retire(linears);
    std::fs::remove_file(&f.path).unwrap();
    for (_, descriptors) in routed_classes(&c) {
        let descriptor = &descriptors[0];
        let result: Result<()> = c.synchronous(|this| {
            let cache = this
                .mixed_rank_block
                .as_mut()
                .unwrap()
                .decoded
                .as_mut()
                .unwrap();
            let Lookup::Decode(view) = cache.acquire(descriptor)? else {
                panic!("uninitialized slot was exposed as ready");
            };
            let before = this.ops.counters();
            assert!(this.mixed_decoded_weight(descriptor).is_err());
            assert_eq!(this.ops.counters().kernel_launches, before.kernel_launches);
            drop(view);
            Err(cuda_error(
                "abort uninitialized lease with known completion",
            ))
        });
        assert!(result.is_err());
        assert!(!c.needs_quarantine());
        let before = op.counters().compute_kernel_launches;
        decode_completed(&mut c, descriptor);
        let decoded = op.counters().compute_kernel_launches;
        assert_eq!(decoded - before, descriptor.operands.len() as u64);
        decode_completed(&mut c, descriptor);
        assert_eq!(op.counters().compute_kernel_launches, decoded);
        let snapshot = c.mixed_decoded_cache_snapshot().unwrap();
        assert_eq!(snapshot.pending_lease_count, 0);
        assert_eq!(snapshot.borrowed_key_count, 0);
    }
}

#[test]
#[ignore = "GPU-owner authorization only; staged BF16 down slices M1/M5 cold/hot, no CPU trace synthesis"]
fn routed_down_staged_slices_match_standalone_and_preserve_neighbors() {
    for lane in 0..2 {
        for rows in [1, 5] {
            for cached in [false, true] {
                let f = Fixture::new(lane);
                let linears = f.prepare();
                let op = authorized_ops(lane);
                let reference_op = authorized_ops(lane);
                let block = if cached {
                    cached_block(&f, &linears)
                } else {
                    f.block(&linears)
                };
                let mut staged = owner(op.clone(), &f, block, &linears);
                let mut reference = owner(reference_op.clone(), &f, f.block(&linears), &linears);
                let _runtime = retire(linears);
                std::fs::remove_file(&f.path).unwrap();
                let x = input(&op, rows);
                let reference_x = input(&reference_op, rows);
                let elements = rows * 256;
                let sentinel = 0x3f81;
                let packed = op.upload_bf16_words(&vec![sentinel; 3 * elements]).unwrap();
                // Force the independent original split+SwiGLU path, not a
                // comparison of two invocations of the new fused tail.
                reference.set_diagnostic_trace(|_| Ok(()));
                let expected = execute(&mut reference, 1, &reference_x).unwrap();
                let expected = output_words(&reference_op, &expected);
                let t = tokens(&staged, 1);
                for _ in 0..2 {
                    let output = packed.slice(elements, elements).unwrap();
                    let got = staged
                        .synchronous(|this| {
                            this.hold_bf16(&packed)?;
                            this.mixed_swiglu_inner([&t[0], &t[1], &t[2]], &x, rows, Some(output))
                        })
                        .unwrap();
                    assert_eq!(output_words(&op, &got), expected);
                    let words = op
                        .download_bf16_buffer(&packed)
                        .unwrap()
                        .into_iter()
                        .map(crate::execution::bf16_rne_word)
                        .collect::<Vec<_>>();
                    assert!(words[..elements].iter().all(|&word| word == sentinel));
                    assert_eq!(&words[elements..2 * elements], &expected);
                    assert!(words[2 * elements..].iter().all(|&word| word == sentinel));
                    assert_eq!(output_words(&op, &x), vec![0x3b80; elements]);
                    assert!(!staged.needs_quarantine());
                    if cached {
                        let snapshot = staged.mixed_decoded_cache_snapshot().unwrap();
                        assert_eq!(snapshot.phase, MixedDecodedCachePhase::Idle);
                        assert_eq!(snapshot.pending_lease_count, 0);
                        assert_eq!(snapshot.borrowed_key_count, 0);
                    }
                }
            }
        }
    }
}

#[path = "mixed_routed_fusion.rs"]
pub(super) mod routed_fusion;

#[path = "mixed_route_batch_gpu.rs"]
pub(super) mod route_batch;
