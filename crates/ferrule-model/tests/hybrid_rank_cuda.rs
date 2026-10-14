//! Host preparation/geometry tests plus real CUDA fixtures compiled but ignored.
#![cfg(feature = "cuda")]
#[allow(dead_code)]
#[path = "support/hybrid_tp_ep_fixture.rs"]
mod base;
use ferrule_common::execution::*;
use ferrule_common::{ParallelRankId, ParallelTopologyId};
use ferrule_model::checkpoint::{CheckpointDType, CheckpointTensorReader, CheckpointTensorSlice};
use ferrule_model::decoder::*;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::nn::{
    ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec,
};
use ferrule_model::transformer::hybrid_parallel::*;
use ferrule_model::transformer::*;
use ferrule_model::{ModelFamily, TensorRole, WeightSource};
use std::sync::Arc;

fn options(r: &BoundDecoderResources) -> GenericDecoderOptions {
    GenericDecoderOptions::standard_cpu(
        r.spec(),
        ModelFamily::Unknown("rank-fixture".into()),
        WeightSource::Safetensors,
        2,
        32,
        8,
        1,
        1 << 20,
        ExecutionPrecisionPolicy::f32(),
    )
    .unwrap()
}
fn parallel(r: &BoundDecoderResources, degree: usize) -> HybridParallelPlan {
    let members = [7, 19, 31, 63][..degree]
        .iter()
        .copied()
        .map(ParallelRankId::new)
        .collect::<Vec<_>>();
    HybridParallelPlan::new(
        r.spec(),
        0..4,
        HybridExpertScope {
            topology: ParallelTopologyId::new(907),
            replica: 0,
            stage: 0,
        },
        members.clone(),
        Some(HybridExpertAttachment {
            source_rank: *members.last().unwrap(),
            members: vec![ParallelRankId::new(101)],
            placement: ExpertPlacement::new(
                (1..4).flat_map(|l| (0..2).map(move |e| (l, e, ParallelRankId::new(101)))),
            )
            .unwrap(),
        }),
    )
    .unwrap()
}
fn segment() -> LayerSegmentPlan {
    LayerSegmentPlan::new(4, 0..4, true, true).unwrap()
}
fn grant(n: usize) -> usize {
    ferrule_backend::cuda_allocation_grant_bytes(n).unwrap()
}

struct NumericFixture {
    _base: base::Fixture,
    resources: BoundDecoderResources,
}
impl NumericFixture {
    fn new() -> Self {
        let base = base::Fixture::for_degree(4, false);
        let original = base.resources();
        let path = base.dir.join("numeric.bin");
        let mut bytes = Vec::new();
        let mut slices = Vec::new();
        let mut schema = StateDictSchema::builder();
        let mut mapper = ExactNameMapper::new();
        let reader = CheckpointTensorReader::new(1 << 20);
        for (i, binding) in original.state_dict().parameters().iter().enumerate() {
            let source = binding.weight().slice();
            let numeric = source.shape.len() == 2 && binding.role() != &TensorRole::TokenEmbedding;
            let offset = bytes.len();
            if numeric {
                // Deterministic actual E4M3 payload. Distinct row/column values
                // catch accidental full/zero-origin reads; not a CPU oracle.
                bytes.extend(
                    (0..source.element_count().unwrap())
                        .map(|n| [0x20, 0xa0, 0x28, 0xa8, 0x18][(n + i) % 5]),
                );
            } else {
                bytes.extend(reader.read_slice(source).unwrap().bytes);
            }
            let weight = CheckpointTensorSlice {
                name: source.name.clone(),
                role: source.role.clone(),
                path: path.clone(),
                offset: offset as u64,
                bytes: (bytes.len() - offset) as u64,
                dtype: if numeric {
                    CheckpointDType::F8E4M3
                } else {
                    source.dtype.clone()
                },
                shape: source.shape.clone(),
            };
            let module = ModulePath::new(&source.name).unwrap();
            let mut parameter = ParameterSpec::new(
                ParameterId::new(i as u64 + 1),
                module.clone(),
                if numeric {
                    ParameterDType::F8E4M3
                } else {
                    ParameterDType::F32
                },
                source.shape.clone(),
                binding.residency().clone(),
            )
            .unwrap();
            if numeric {
                let shape = source
                    .shape
                    .iter()
                    .map(|n| n.div_ceil(128))
                    .collect::<Vec<_>>();
                let offset = bytes.len();
                for n in 0..shape.iter().product::<usize>() {
                    bytes.extend(
                        half::bf16::from_f32(0.5 + n as f32 * 0.125)
                            .to_bits()
                            .to_le_bytes(),
                    );
                }
                let name = format!("{}.scale", source.name);
                slices.push(CheckpointTensorSlice {
                    name: name.clone(),
                    role: source.role.clone(),
                    path: path.clone(),
                    offset: offset as u64,
                    bytes: (bytes.len() - offset) as u64,
                    dtype: CheckpointDType::Bf16,
                    shape: shape.clone(),
                });
                parameter = parameter
                    .with_required_scale(ParameterDType::Bf16, shape)
                    .unwrap();
                mapper
                    .insert(name, NameMapping::scale(module.clone()))
                    .unwrap();
            }
            schema
                .register_with_role(parameter, binding.role().clone())
                .unwrap();
            mapper
                .insert(&source.name, NameMapping::weight(module))
                .unwrap();
            slices.push(weight);
        }
        std::fs::write(path, bytes).unwrap();
        let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
            .bind_slices(slices)
            .unwrap();
        let resources =
            BoundDecoderResources::new(original.spec().clone(), Arc::new(bound)).unwrap();
        Self {
            _base: base,
            resources,
        }
    }
}
#[test]
fn rank_numeric_preparation_keeps_paired_segments_and_rejects_full_reads() {
    let f = NumericFixture::new();
    let r = &f.resources;
    for degree in [1, 2, 4] {
        let p = parallel(r, degree);
        for lane in 0..degree {
            let layout = Arc::new(p.rank_layout(lane).unwrap());
            let memory =
                CudaHybridRankMemory::for_rank(r, &segment(), &p, &layout, &options(r), 16)
                    .unwrap();
            let materializer = StateDictMaterializer::for_hybrid_cuda_rank(
                1 << 20,
                layout.clone(),
                memory.weight_grants,
            )
            .unwrap();
            for binding in r.state_dict().parameters() {
                if binding.numeric_fp8_encoding().is_none()
                    || matches!(binding.residency(), ParameterResidency::Expert { .. })
                {
                    continue;
                }
                let prepared = materializer
                    .prepared_linear(binding, binding.role().clone())
                    .unwrap();
                let local = prepared.hybrid_rank().unwrap();
                let partition = layout.parameter_partition(binding).unwrap();
                assert_eq!(local.local_shape(), partition.local_shape().unwrap());
                assert_eq!(
                    [prepared.out_features(), prepared.in_features()],
                    local.global_shape()
                );
                let HybridRankStorage::Numeric(numeric) = local.storage() else {
                    panic!("decoded numeric weight")
                };
                assert_eq!(numeric.segments().len(), partition.rows.len());
                for (part, range) in numeric.segments().iter().zip(&partition.rows) {
                    assert_eq!(
                        part.artifact().provenance().weight_read().rows(),
                        range.clone()
                    );
                    assert_eq!(
                        part.artifact().provenance().weight_read().columns(),
                        partition.columns.clone()
                    );
                }
                assert!(prepared.parameter().values_f32().is_err());
                assert!(materializer.parameter(binding).is_err());
            }
        }
    }
}
#[test]
fn rank_physical_admission_counts_local_state_kv_and_shared_source_holds() {
    let f = NumericFixture::new();
    let r = &f.resources;
    let o = options(r);
    for degree in [1, 2, 4] {
        let p = parallel(r, degree);
        for lane in 0..degree {
            let layout = p.rank_layout(lane).unwrap();
            let m = CudaHybridRankMemory::for_rank(r, &segment(), &p, &layout, &o, 16).unwrap();
            assert_eq!(m.precision, NumericFp8Precision::F32Tf32x3);
            let expected = layout
                .state_schema()
                .layers()
                .iter()
                .map(|l| match l {
                    HybridLayerSchema::GatedDeltaNet(s) => {
                        let (_, c, r) = s.sizes().unwrap();
                        grant(c * 4) + grant(r * 4)
                    }
                    _ => 0,
                })
                .sum::<usize>();
            assert_eq!(m.per_sequence_state_grants, expected);
            assert_eq!(
                m.per_sequence_state_bytes,
                p.local_state_bytes(lane).unwrap()
            );
            let planes = layout.kv_planes(2, 32).unwrap();
            assert_eq!(
                m.physical_kv_grants,
                KvLayoutSchema::planes(&planes)
                    .iter()
                    .map(|plane| grant(plane.checked_page_bytes(2).unwrap() * 16))
                    .sum::<usize>()
            );
            assert_eq!(m.external_hold_grants > 0, lane + 1 == degree);
            assert!(m.gqa_metadata_grants > 0 && m.recurrent_workspace_grants > 0);
            let mut budget = HybridCudaMemoryBudget {
                kv_pages: 16,
                state_bytes: expected * 3,
                weight_bytes: m.weight_grants,
                workspace_bytes: m.numeric_workspace_grants + m.operation_workspace_grants,
            };
            m.validate_device_budget(budget, 1).unwrap();
            budget.state_bytes -= 1;
            assert!(m.validate_device_budget(budget, 1).is_err());
            assert!(m.total_device_grants(usize::MAX).is_err());
            let w8 = o
                .clone()
                .with_hybrid_cuda_numeric_fp8_precision(
                    ExpertCacheLimits {
                        max_experts: 1,
                        max_bytes: 1 << 20,
                    },
                    0,
                    NumericFp8Precision::W8A16Bf16F32Accumulate,
                )
                .unwrap();
            assert_eq!(
                CudaHybridRankMemory::for_rank(r, &segment(), &p, &layout, &w8, 16)
                    .unwrap()
                    .numeric_workspace_grants,
                0
            );
        }
    }
}
#[test]
fn rank_admission_rejects_foreign_plan_and_precision_before_reads() {
    let f = NumericFixture::new();
    let r = &f.resources;
    let p = parallel(r, 2);
    let o = options(r);
    assert!(
        CudaHybridRankMemory::for_rank(
            r,
            &segment(),
            &p,
            &parallel(r, 4).rank_layout(0).unwrap(),
            &o,
            16
        )
        .is_err()
    );
    assert!(
        CudaHybridRankMemory::for_rank(r, &segment(), &p, &p.rank_layout(0).unwrap(), &o, 0)
            .is_err()
    );
    let bad = o
        .with_hybrid_cuda_numeric_fp8(
            ExpertCacheLimits {
                max_experts: 1,
                max_bytes: 1 << 20,
            },
            4096,
        )
        .unwrap();
    assert!(
        CudaHybridRankMemory::for_rank(r, &segment(), &p, &p.rank_layout(0).unwrap(), &bad, 16)
            .is_err()
    );
}

#[path = "hybrid_rank_cuda/gpu.rs"]
mod gpu;

#[test]
fn dense_rank_materialization_preserves_segmented_convolution_and_head_vectors() {
    let fixture = base::Fixture::for_degree(4, false);
    let resources = fixture.resources();
    let reader = CheckpointTensorReader::new(1 << 20);
    for degree in [1, 2, 4] {
        let plan = parallel(&resources, degree);
        for lane in 0..degree {
            let layout = Arc::new(plan.rank_layout(lane).unwrap());
            let materializer =
                StateDictMaterializer::for_hybrid_cuda_rank(1 << 20, layout.clone(), 1 << 20)
                    .unwrap();
            let mut cache = MemoryLayerWeightCache::default();
            for binding in resources.state_dict().parameters().iter().filter(|b| {
                matches!(
                    b.role(),
                    TensorRole::LinearAttentionConv
                        | TensorRole::LinearAttentionALog
                        | TensorRole::LinearAttentionDtBias
                )
            }) {
                let ParameterResidency::Layer { layer } = *binding.residency() else {
                    unreachable!()
                };
                let parameter = materializer
                    .layer_parameter(layer, binding, &mut cache)
                    .unwrap();
                let partition = layout.parameter_partition(binding).unwrap();
                let payload = reader.read_slice(binding.weight().slice()).unwrap();
                let full = payload
                    .bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect::<Vec<_>>();
                let width = partition.columns.end;
                let expected = partition
                    .rows
                    .iter()
                    .flat_map(|range| {
                        range.clone().flat_map(|row| {
                            full[row * width + partition.columns.start
                                ..row * width + partition.columns.end]
                                .iter()
                                .copied()
                        })
                    })
                    .collect::<Vec<_>>();
                assert_eq!(parameter.values_f32().unwrap(), expected);
                assert!(parameter.hybrid_rank().is_some());
                assert!(materializer.parameter(binding).is_err());
            }
        }
    }
}

#[path = "hybrid_rank_cuda/preparation_claim.rs"]
mod preparation_claim;
