//! Checkpoint reads, not merely upload sizes, prove dense decoder TP sharding.
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ferrule_common::ParallelRankId;
use ferrule_common::execution::{KvElementType, KvLayoutSchema};
use ferrule_model::TensorRole;
use ferrule_model::checkpoint::{CheckpointDType, CheckpointTensorReader, CheckpointTensorSlice};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::models::qwen3::{Qwen3DenseAdapter, Qwen3DenseRecipe};
use ferrule_model::nn::{ParameterDType, ParameterSpec};
use ferrule_model::transformer::parallel::{
    TensorParallelLinearPartition as Partition, TensorParallelLinearPlan,
};
use ferrule_model::transformer::{
    BoundDecoderResources, DecoderLoadOptions, DecoderModelSpec, DecoderRecipe, DecoderRecipeError,
    ExactNameMapper, LayerSegmentPlan, NameMapper, NameMapping, StandardDecoderSegment,
    StandardTensorPlacement, StandardTensorPlan, StateDictMaterializer, StateDictSchema,
    TensorTransform,
};

struct Recipe {
    f32: bool,
    transpose_query: bool,
}
impl DecoderRecipe for Recipe {
    fn build_spec(
        &self,
        config: &serde_json::Value,
    ) -> Result<DecoderModelSpec, DecoderRecipeError> {
        Qwen3DenseRecipe::new().build_spec(config)
    }
    fn build_schema(
        &self,
        config: &serde_json::Value,
    ) -> Result<StateDictSchema, DecoderRecipeError> {
        let schema = Qwen3DenseRecipe::new().build_schema(config)?;
        if !self.f32 {
            return Ok(schema);
        }
        let mut builder = StateDictSchema::builder();
        for p in schema.parameters() {
            let mut spec = ParameterSpec::new(
                p.id(),
                p.path().clone(),
                ParameterDType::F32,
                p.shape().to_vec(),
                p.residency().clone(),
            )?;
            if let Some(alias) = p.alias_of() {
                spec = spec.with_alias(alias);
            }
            builder.register_with_role(spec, schema.role(p.id()).unwrap().clone())?;
        }
        Ok(builder.build()?)
    }
    fn build_name_mapper(
        &self,
        config: &serde_json::Value,
    ) -> Result<Arc<dyn NameMapper>, DecoderRecipeError> {
        let schema = self.build_schema(config)?;
        let mut mapper = ExactNameMapper::new();
        for p in schema
            .parameters()
            .iter()
            .filter(|p| p.alias_of().is_none())
        {
            let mut mapping = NameMapping::weight(p.path().clone());
            if self.transpose_query && schema.role(p.id()) == Some(&TensorRole::AttentionQuery) {
                mapping.transform = TensorTransform::transpose_2d();
            }
            mapper.insert(p.path().as_str(), mapping).unwrap();
        }
        Ok(Arc::new(mapper))
    }
}

struct Fixture {
    directory: PathBuf,
    resources: BoundDecoderResources,
    dtype: CheckpointDType,
}
impl Fixture {
    fn new(tied: bool, f32: bool, transpose_query: bool) -> Self {
        Self::with_geometry(tied, f32, transpose_query, 2)
    }
    fn with_geometry(tied: bool, f32: bool, transpose_query: bool, head_dim: usize) -> Self {
        Self::with_mlp_width(tied, f32, transpose_query, head_dim, 64)
    }
    fn with_mlp_width(
        tied: bool,
        f32: bool,
        transpose_query: bool,
        head_dim: usize,
        intermediate_size: usize,
    ) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "ferrule-tp-prepare-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let recipe = Recipe {
            f32,
            transpose_query,
        };
        let config = serde_json::json!({
            "architectures":["Qwen3ForCausalLM"], "model_type":"qwen3", "torch_dtype":"bfloat16",
            "hidden_act":"silu", "vocab_size":11, "hidden_size":8, "num_hidden_layers":2,
            "num_attention_heads":8, "num_key_value_heads":4, "head_dim":head_dim, "intermediate_size":intermediate_size,
            "max_position_embeddings":16, "rms_norm_eps":0.00001, "rope_theta":10000.0,
            "tie_word_embeddings":tied, "attention_bias":false, "use_sliding_window":false,
            "attention_dropout":0.0, "use_cache":true, "max_window_layers":2,
            "initializer_range":0.02, "bos_token_id":1, "eos_token_id":2
        });
        let schema = recipe.build_schema(&config).unwrap();
        let dtype = if f32 {
            CheckpointDType::F32
        } else {
            CheckpointDType::Bf16
        };
        let path = directory.join("weights.bin");
        // Deliberately use nonzero source offsets and put many tensors in one file.
        let mut data = vec![0xab; 37];
        let mut slices = Vec::new();
        for p in schema
            .parameters()
            .iter()
            .filter(|p| p.alias_of().is_none())
        {
            let offset = data.len();
            for i in 0..p.shape().iter().product::<usize>() {
                let value = if p.shape().len() == 1 {
                    1.0
                } else {
                    (i % 251) as f32 / 256.0
                };
                if f32 {
                    data.extend(value.to_le_bytes());
                } else {
                    data.extend(half::bf16::from_f32(value).to_bits().to_le_bytes());
                }
            }
            let mut shape = p.shape().to_vec();
            if transpose_query && schema.role(p.id()) == Some(&TensorRole::AttentionQuery) {
                shape.reverse();
            }
            slices.push(CheckpointTensorSlice {
                name: p.path().as_str().into(),
                role: TensorRole::Unknown,
                path: path.clone(),
                offset: offset as u64,
                bytes: (data.len() - offset) as u64,
                dtype: dtype.clone(),
                shape,
            });
        }
        std::fs::write(&path, data).unwrap();
        let resources = DecoderLoadOptions::new(&recipe, &config)
            .bind_slices(slices)
            .unwrap();
        Self {
            directory,
            resources,
            dtype,
        }
    }
    fn plan(&self, degree: usize) -> StandardTensorPlan {
        StandardTensorPlan::new(
            self.resources.spec(),
            (0..degree)
                .map(|device| StandardTensorPlacement {
                    owner: ParallelRankId::new(20 + device as u32 * 3),
                    device,
                })
                .collect(),
        )
        .unwrap()
    }
    fn full_segment(&self) -> LayerSegmentPlan {
        LayerSegmentPlan::new(2, 0..2, true, true).unwrap()
    }
    fn role(&self, role: TensorRole) -> &ferrule_model::transformer::BoundParameter {
        self.resources
            .state_dict()
            .parameters()
            .iter()
            .find(|p| p.role() == &role)
            .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

#[test]
fn pp2_segment_admission_and_segment_local_kv_schema_preserve_geometry() {
    let config = serde_json::json!({
        "architectures":["Qwen3ForCausalLM"], "model_type":"qwen3", "torch_dtype":"bfloat16",
        "hidden_act":"silu", "vocab_size":32, "hidden_size":8, "num_hidden_layers":4,
        "num_attention_heads":4, "num_key_value_heads":2, "head_dim":8, "intermediate_size":16,
        "max_position_embeddings":32, "rms_norm_eps":0.00001, "rope_theta":10000.0,
        "tie_word_embeddings":true, "attention_bias":false, "use_sliding_window":false,
        "attention_dropout":0.0, "use_cache":true, "max_window_layers":4,
        "initializer_range":0.02, "bos_token_id":1, "eos_token_id":2
    });
    let spec = Qwen3DenseRecipe::new().build_spec(&config).unwrap();
    let plan = StandardTensorPlan::new(
        &spec,
        (0..2)
            .map(|device| StandardTensorPlacement {
                owner: ParallelRankId::new(40 + device as u32),
                device,
            })
            .collect(),
    )
    .unwrap();
    let first = LayerSegmentPlan::new(4, 0..2, true, false).unwrap();
    let second = LayerSegmentPlan::new(4, 2..4, false, true).unwrap();
    assert!(plan.validate_segment(&spec, &first).is_ok());
    assert!(plan.validate_segment(&spec, &second).is_ok());
    assert_eq!(plan.local_kv_heads(), 1);
    assert_eq!(plan.global_kv_heads(), 2);
    assert_eq!(plan.head_dim(), 8);
    assert_eq!(plan.kv_planes(2, 32).unwrap().planes()[0].layer_count, 4);
    assert_eq!(
        plan.kv_planes_for_segment(&first, 2, 32).unwrap().planes()[0].layer_count,
        2
    );
    assert_eq!(
        plan.kv_planes_for_segment(&second, 2, 32).unwrap().planes()[0].layer_count,
        2
    );
    assert!(plan.kv_planes_for_segment(&first, 0, 32).is_err());
    assert!(LayerSegmentPlan::new(4, 0..5, true, false).is_err());
    assert!(LayerSegmentPlan::new(4, 0..2, true, true).is_err());
}

#[test]
fn pp2_tp2_tp4_only_read_owned_layers_and_endpoint_weights() {
    use ferrule_model::nn::ParameterResidency;
    for tied in [false, true] {
        let f = Fixture::with_geometry(tied, false, false, 8);
        for degree in [2, 4] {
            let tensor = f.plan(degree);
            for rank in 0..degree {
                let local = ParallelRankId::new(rank as u32);
                let full = StandardDecoderSegment::prepare_tensor(
                    &f.resources,
                    f.full_segment(),
                    tensor.clone(),
                    local,
                    16,
                    4096,
                )
                .unwrap();
                let mut combined = Vec::new();
                for stage in 0..2 {
                    let plan =
                        LayerSegmentPlan::new(2, stage..stage + 1, stage == 0, stage == 1).unwrap();
                    let schema = tensor.kv_planes_for_segment(&plan, 2, 16).unwrap();
                    for plane in schema.planes() {
                        assert_eq!(plane.layer_count, 1);
                        assert_eq!(plane.elements_per_token, 4 / degree * 8);
                        assert_eq!(plane.element_type, KvElementType::F32);
                    }
                    let segment = StandardDecoderSegment::prepare_tensor(
                        &f.resources,
                        plan.clone(),
                        tensor.clone(),
                        local,
                        16,
                        4096,
                    )
                    .unwrap();
                    let reads = segment.tensor_reads().unwrap();
                    assert_eq!(reads.len(), if stage == 0 { 12 } else { 13 });
                    assert_eq!(
                        reads
                            .iter()
                            .filter(|r| r.role == TensorRole::TokenEmbedding)
                            .count(),
                        usize::from(stage == 0)
                    );
                    assert_eq!(
                        reads
                            .iter()
                            .filter(|r| r.role == TensorRole::OutputHead)
                            .count(),
                        usize::from(stage == 1)
                    );
                    for binding in segment.parameters() {
                        match binding.residency() {
                            ParameterResidency::Layer { layer } => assert_eq!(*layer, stage),
                            ParameterResidency::Static => assert!(matches!(
                                (stage, binding.role()),
                                (0, TensorRole::TokenEmbedding)
                                    | (1, TensorRole::OutputNorm | TensorRole::OutputHead)
                            )),
                            other => panic!("unexpected {other:?}"),
                        }
                    }
                    assert_eq!(plan.local_layer(stage), Some(0));
                    assert_eq!(plan.global_layer(0), Some(stage));
                    combined.extend(reads);
                }
                let mut expected = full.tensor_reads().unwrap();
                combined.sort_by_key(|r| r.parameter);
                expected.sort_by_key(|r| r.parameter);
                assert_eq!(combined.len(), expected.len());
                for (actual, expected) in combined.iter().zip(&expected) {
                    assert_eq!(actual.parameter, expected.parameter);
                    assert_eq!(actual.canonical, expected.canonical);
                    assert_eq!(actual.bytes, expected.bytes);
                    assert_eq!(actual.rectangle, expected.rectangle);
                }
                // Boundary-less segments are legitimate hidden-in/hidden-out stages.
                let plan = LayerSegmentPlan::new(2, 0..2, false, false).unwrap();
                let segment = StandardDecoderSegment::prepare_tensor(
                    &f.resources,
                    plan,
                    tensor.clone(),
                    local,
                    16,
                    4096,
                )
                .unwrap();
                assert_eq!(segment.tensor_reads().unwrap().len(), 22);
            }
            let mismatch = LayerSegmentPlan::new(3, 1..3, false, true).unwrap();
            assert!(
                tensor
                    .validate_segment(f.resources.spec(), &mismatch)
                    .is_err()
            );
            assert!(tensor.kv_planes_for_segment(&mismatch, 2, 16).is_err());
            let changed_shape = Fixture::with_geometry(tied, false, false, 4);
            assert!(
                tensor
                    .validate_segment(changed_shape.resources.spec(), &f.full_segment())
                    .is_err()
            );
        }
    }
}

fn partition(role: &TensorRole) -> Option<Partition> {
    match role {
        TensorRole::AttentionQuery
        | TensorRole::AttentionKey
        | TensorRole::AttentionValue
        | TensorRole::DenseMlpGate
        | TensorRole::DenseMlpUp
        | TensorRole::OutputHead => Some(Partition::Column),
        TensorRole::AttentionOutput | TensorRole::DenseMlpDown => Some(Partition::Row),
        _ => None,
    }
}

#[test]
fn tp_read_preflight_matches_qwen_delegate_and_exact_rank_budgets() {
    for tied in [false, true] {
        for f32 in [false, true] {
            for intermediate in [64usize, 65] {
                let f = Fixture::with_mlp_width(tied, f32, false, 2, intermediate);
                let element = f.dtype.element_size_bytes().unwrap();
                for degree in [1, 2, 4] {
                    let plan = f.plan(degree);
                    // Ragged shards need ceil, not global bytes / degree. The
                    // largest MLP shard also covers replicated embedding/norms.
                    let limit = (intermediate.div_ceil(degree) * 8 * element) as u64;
                    plan.validate_read_limits(&f.resources, limit).unwrap();
                    Qwen3DenseAdapter::validate_tensor_read_limits(&f.resources, &plan, limit)
                        .unwrap();
                    let error = plan
                        .validate_read_limits(&f.resources, limit - 1)
                        .unwrap_err();
                    assert!(error.to_string().contains("rank 0 read limit"), "{error}");
                    let delegated = Qwen3DenseAdapter::validate_tensor_read_limits(
                        &f.resources,
                        &plan,
                        limit - 1,
                    )
                    .unwrap_err();
                    assert_eq!(error.to_string(), delegated.to_string());
                }
            }
        }
    }
}

#[test]
fn tp_read_preflight_rejects_zero_replicated_budget_and_stale_sources() {
    let f = Fixture::new(true, false, false);
    let plan = f.plan(4);
    let error = plan.validate_read_limits(&f.resources, 0).unwrap_err();
    assert!(error.to_string().contains("TP read limit must be positive"));
    let embedding = f.role(TensorRole::TokenEmbedding).weight().slice();
    let error = plan
        .validate_read_limits(&f.resources, embedding.bytes - 1)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("replicated parameter 'token_embedding.weight'")
    );
    plan.validate_read_limits(&f.resources, 1024).unwrap();
    std::fs::OpenOptions::new()
        .append(true)
        .open(&embedding.path)
        .unwrap()
        .write_all(&[0])
        .unwrap();
    let error = plan.validate_read_limits(&f.resources, 1024).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("stale checkpoint source identity")
    );
}

#[test]
fn tp_read_preflight_preserves_resource_and_storage_validation() {
    let f = Fixture::new(true, false, true);
    let error = f
        .plan(2)
        .validate_read_limits(&f.resources, 1024)
        .unwrap_err();
    assert!(error.to_string().contains("identity, unscaled dense"));
    let other = Fixture::new(false, false, false);
    let error = other
        .plan(2)
        .validate_read_limits(&f.resources, 1024)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("resources do not match the planned decoder")
    );
}

#[test]
fn complete_tp2_tp4_preparation_reads_only_projection_rectangles() {
    for tied in [false, true] {
        for f32 in [false, true] {
            let f = Fixture::new(tied, f32, false);
            let element = f.dtype.element_size_bytes().unwrap();
            // Every rank's local MLP fits; the full MLP does not. This budget is
            // also enough for the deliberately replicated embedding.
            let limit = (32 * 8 * element) as u64;
            assert!(
                StandardDecoderSegment::prepare(
                    &f.resources,
                    f.full_segment(),
                    ExecutionPrecisionPolicy::f32(),
                    16,
                    limit
                )
                .is_err()
            );
            for degree in [2, 4] {
                let plan = f.plan(degree);
                let mut combined = BTreeMap::new();
                for rank in 0..degree {
                    let local = ParallelRankId::new(rank as u32);
                    let segment = StandardDecoderSegment::prepare_tensor(
                        &f.resources,
                        f.full_segment(),
                        plan.clone(),
                        local,
                        16,
                        limit,
                    )
                    .unwrap();
                    assert_eq!(
                        segment.parameters().len(),
                        f.resources.state_dict().parameters().len()
                    );
                    let reads = segment.tensor_reads().unwrap();
                    // One embedding, one final norm, 2*(7 projections + 4 norms), one head.
                    assert_eq!(reads.len(), 25);
                    assert_eq!(reads.iter().filter(|r| r.rectangle.is_some()).count(), 15);
                    let mut total = 0;
                    for read in &reads {
                        let binding = f
                            .resources
                            .state_dict()
                            .parameters()
                            .iter()
                            .find(|p| p.id() == read.parameter)
                            .unwrap();
                        assert_eq!(read.canonical, binding.canonical_id());
                        assert_eq!(&read.role, binding.role());
                        if let Some(partition) = partition(&read.role) {
                            let full = binding.weight().slice();
                            let expected = TensorParallelLinearPlan::new(
                                full.shape[0],
                                full.shape[1],
                                degree,
                                partition,
                            )
                            .unwrap();
                            let shape = expected.local_shape(local).unwrap();
                            let rectangle = read
                                .rectangle
                                .as_ref()
                                .expect("projection must not use full reads/cache");
                            assert_eq!(rectangle.tensor(), full);
                            assert_eq!(rectangle.local_shape(), [shape.0, shape.1]);
                            assert_eq!(read.bytes, (shape.0 * shape.1 * element) as u64);
                            assert_eq!(read.bytes, rectangle.read_plan().storage_bytes());
                            assert!(read.bytes < full.bytes);
                            let range = expected.rank_range(local).unwrap();
                            if partition == Partition::Column {
                                assert_eq!(rectangle.rows(), range.clone());
                                assert_eq!(rectangle.columns(), 0..full.shape[1]);
                                assert_eq!(rectangle.read_plan().extents().len(), 1);
                                let extent = &rectangle.read_plan().extents()[0];
                                assert_eq!(
                                    extent.offset(),
                                    full.offset + (range.start * full.shape[1] * element) as u64
                                );
                                assert_eq!(extent.bytes(), read.bytes);
                            } else {
                                assert_eq!(rectangle.rows(), 0..full.shape[0]);
                                assert_eq!(rectangle.columns(), range.clone());
                                assert_eq!(rectangle.read_plan().extents().len(), full.shape[0]);
                                for (row, extent) in
                                    rectangle.read_plan().extents().iter().enumerate()
                                {
                                    assert_eq!(
                                        extent.offset(),
                                        full.offset
                                            + ((row * full.shape[1] + range.start) * element)
                                                as u64
                                    );
                                    assert_eq!(extent.bytes(), (range.len() * element) as u64);
                                }
                            }
                            *combined.entry(binding.id()).or_insert(0u64) += read.bytes;
                        } else {
                            assert!(read.rectangle.is_none());
                            assert_eq!(read.bytes, binding.weight().slice().bytes);
                        }
                        total += read.bytes;
                    }
                    assert!(
                        total
                            < f.resources
                                .state_dict()
                                .parameters()
                                .iter()
                                .map(|p| p.weight().slice().bytes)
                                .sum::<u64>()
                    );
                    let embed = reads
                        .iter()
                        .find(|r| r.role == TensorRole::TokenEmbedding)
                        .unwrap();
                    let head = reads
                        .iter()
                        .find(|r| r.role == TensorRole::OutputHead)
                        .unwrap();
                    assert_eq!(head.canonical == embed.canonical, tied);
                    assert_ne!(head.parameter, embed.parameter);
                    assert!(head.rectangle.is_some() && embed.rectangle.is_none());
                }
                for (id, bytes) in combined {
                    let binding = f
                        .resources
                        .state_dict()
                        .parameters()
                        .iter()
                        .find(|p| p.id() == id)
                        .unwrap();
                    assert_eq!(
                        bytes,
                        binding.weight().slice().bytes,
                        "all ranks exactly cover {:?}",
                        binding.path()
                    );
                }
            }
        }
    }
}

#[test]
fn every_projection_keeps_global_shape_local_bytes_and_source_provenance() {
    for f32 in [false, true] {
        let f = Fixture::new(true, f32, false);
        for degree in [2, 4] {
            for rank in 0..degree {
                for binding in f.resources.state_dict().parameters() {
                    let Some(partition) = partition(binding.role()) else {
                        continue;
                    };
                    let full = binding.weight().slice();
                    let plan = TensorParallelLinearPlan::new(
                        full.shape[0],
                        full.shape[1],
                        degree,
                        partition,
                    )
                    .unwrap();
                    let local = ParallelRankId::new(rank as u32);
                    let (out, width) = plan.local_shape(local).unwrap();
                    let bytes = (out * width * f.dtype.element_size_bytes().unwrap()) as u64;
                    assert!(bytes < full.bytes);
                    let materializer =
                        StateDictMaterializer::for_tensor(bytes, f.plan(degree), local).unwrap();
                    // A reader with this exact limit cannot read the original matrix.
                    assert!(CheckpointTensorReader::new(bytes).read_slice(full).is_err());
                    let linear = materializer
                        .prepared_linear(binding, binding.role().clone())
                        .unwrap();
                    assert_eq!(
                        [linear.out_features(), linear.in_features()],
                        [full.shape[0], full.shape[1]]
                    );
                    assert!(linear.parameter().binding().shares_storage_with(binding));
                    let shard = linear.tensor_shard().unwrap();
                    assert_eq!(shard.local_shape(), [out, width]);
                    assert_eq!(shard.bytes().len() as u64, bytes);
                    assert_eq!(shard.provenance().unwrap().tensor(), full);
                    assert!(linear.weight().is_err());
                    assert!(linear.parameter().weight().is_err());
                    let expected_full = StateDictMaterializer::new(full.bytes)
                        .unwrap()
                        .parameter(binding)
                        .unwrap()
                        .values_f32()
                        .unwrap();
                    let expected = plan.shard_weight(local, &expected_full).unwrap().0;
                    assert_eq!(linear.parameter().values_f32().unwrap(), expected);
                    assert_eq!(materializer.tensor_reads().unwrap().len(), 1);
                    assert!(
                        StateDictMaterializer::for_tensor(bytes - 1, f.plan(degree), local)
                            .unwrap()
                            .prepared_linear(binding, binding.role().clone())
                            .is_err()
                    );
                }
            }
        }
    }
}

#[test]
fn tied_head_does_not_reuse_replicated_static_payload_or_stale_source() {
    let f = Fixture::new(true, false, false);
    let materializer =
        StateDictMaterializer::for_tensor(1024, f.plan(2), ParallelRankId::new(1)).unwrap();
    let embed = f.role(TensorRole::TokenEmbedding);
    let head = f.role(TensorRole::OutputHead);
    let replicated = materializer.static_parameter(embed).unwrap();
    assert!(Arc::ptr_eq(
        &replicated,
        &materializer.static_parameter(embed).unwrap()
    ));
    assert_eq!(
        replicated.weight().unwrap().bytes.len() as u64,
        embed.weight().slice().bytes
    );
    assert!(
        materializer.static_parameter(head).is_err(),
        "tied head must not hit the replicated embedding cache"
    );
    let linear = materializer
        .prepared_linear(head, TensorRole::OutputHead)
        .unwrap();
    assert_eq!(linear.parameter().canonical_id(), replicated.canonical_id());
    assert!(
        linear.tensor_shard().unwrap().bytes().len() < replicated.weight().unwrap().bytes.len()
    );
    let reads = materializer.tensor_reads().unwrap();
    assert_eq!(
        reads.len(),
        2,
        "one replicated cache miss and one independent local read"
    );
    let path = &head.weight().slice().path;
    std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .unwrap()
        .write_all(&[0])
        .unwrap();
    assert!(
        linear
            .tensor_shard()
            .unwrap()
            .validate_source_identity()
            .is_err()
    );
    assert!(
        materializer
            .prepared_linear(head, TensorRole::OutputHead)
            .is_err()
    );
    assert_eq!(
        materializer.tensor_reads().unwrap().len(),
        2,
        "failed read must not be recorded as successful"
    );
}

#[test]
fn unsupported_transform_and_wrong_rank_fail_before_checkpoint_reads() {
    let f = Fixture::new(true, false, true);
    let plan = f.plan(2);
    std::fs::remove_file(f.directory.join("weights.bin")).unwrap();
    let error = StandardDecoderSegment::prepare_tensor(
        &f.resources,
        f.full_segment(),
        plan,
        ParallelRankId::new(0),
        16,
        1024,
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("identity, unscaled dense"),
        "{error}"
    );
    let f = Fixture::new(true, false, false);
    assert!(StateDictMaterializer::for_tensor(1024, f.plan(2), ParallelRankId::new(2)).is_err());
    let binding = f.role(TensorRole::AttentionOutput);
    let materializer =
        StateDictMaterializer::for_tensor(1024, f.plan(2), ParallelRankId::new(0)).unwrap();
    assert!(
        materializer
            .prepared_linear(binding, TensorRole::AttentionQuery)
            .is_err()
    );
    assert!(materializer.parameter(binding).is_err());
    assert!(materializer.tensor_reads().unwrap().is_empty());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires four CUDA GPUs; FERRULE_CUDA_ARCH=sm_86"]
fn cuda_preparation_uploads_local_checkpoints_under_submatrix_read_limits() {
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use ferrule_common::execution::ExecutionTransactionId;
    use ferrule_model::transformer::StandardTensorCollective;
    use ferrule_model::transformer::parallel::TensorParallelCollective;
    use std::rc::Rc;

    // Preparation must not call a collective. Numerical forwards use the real
    // runtime HostCollective coverage in standard_tensor_cuda, not this sentinel.
    struct PrepareOnly {
        owner: ParallelRankId,
        members: Vec<ParallelRankId>,
    }
    impl StandardTensorCollective for PrepareOnly {
        fn owner(&self) -> ParallelRankId {
            self.owner
        }
        fn members(&self) -> &[ParallelRankId] {
            &self.members
        }
        fn exchange(
            &mut self,
            _: ExecutionTransactionId,
            _: u64,
            _: TensorParallelCollective,
            _: Vec<f32>,
        ) -> ferrule_common::Result<Vec<f32>> {
            panic!("preparation must never exchange activations")
        }
        fn abort(&mut self) {}
    }
    for f32 in [false, true] {
        let f = Fixture::new(true, f32, false);
        let limit = (32 * 8 * f.dtype.element_size_bytes().unwrap()) as u64;
        for degree in [2, 4] {
            let tensor = f.plan(degree);
            for rank in 0..degree {
                let local = ParallelRankId::new(rank as u32);
                let ops = Rc::new(CudaOperators::new_on_device(rank).unwrap());
                let host = StandardDecoderSegment::prepare_tensor(
                    &f.resources,
                    f.full_segment(),
                    tensor.clone(),
                    local,
                    16,
                    limit,
                )
                .unwrap();
                let expected = host.tensor_reads().unwrap();
                let mut segment = StandardDecoderSegment::prepare_cuda_tensor(
                    &f.resources,
                    f.full_segment(),
                    tensor.clone(),
                    local,
                    16,
                    limit,
                    ops,
                    Box::new(PrepareOnly {
                        owner: tensor.placement(local).unwrap().owner,
                        members: tensor.placements().iter().map(|p| p.owner).collect(),
                    }),
                )
                .unwrap();
                let actual = segment.tensor_reads().unwrap();
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert_eq!(actual.parameter, expected.parameter);
                    assert_eq!(actual.bytes, expected.bytes);
                    assert_eq!(actual.rectangle, expected.rectangle);
                }
                let parameters_f32 = actual
                    .iter()
                    .map(|r| r.bytes as usize / f.dtype.element_size_bytes().unwrap() * 4)
                    .sum::<usize>();
                // The single deduplicated split-half RoPE table has cos + sin.
                assert_eq!(
                    segment.operators().resident_parameter_bytes(),
                    parameters_f32 + 16 * 2 * 4
                );
                segment.quiesce().unwrap();
                assert!(!segment.needs_quarantine());
            }
        }
    }
}
