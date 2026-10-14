//! Production rank preparation. Dense conversion is limited to the selected
//! BF16/F32 rectangles; numeric FP8 remains encoded with paired scale origins.
use super::*;
use crate::transformer::hybrid_parallel::{
    HybridParameterPartition, HybridRankLayout, HybridRankPrecision,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TypedSourceVectorPolicy {
    NativeF32,
    NativeBf16,
    Bf16ResidentFromF32,
}

impl TypedSourceVectorPolicy {
    pub(crate) fn for_parameter(
        precision: HybridRankPrecision,
        role: &TensorRole,
        dtype: &CheckpointDType,
    ) -> Result<Self> {
        match dtype {
            CheckpointDType::Bf16 => Ok(Self::NativeBf16),
            CheckpointDType::F32 if !precision.is_mixed() => Ok(Self::NativeF32),
            CheckpointDType::F32
                if matches!(
                    role,
                    TensorRole::LinearAttentionALog | TensorRole::LinearAttentionNorm
                ) =>
            {
                Ok(Self::Bf16ResidentFromF32)
            }
            _ => Err(model_error(
                "Mixed rank F32 source is restricted to GDN A_log/norm vectors",
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub enum HybridRankStorage {
    Numeric(PreparedNumericFp8RankLinear),
    /// Runtime metadata only. Not an uploadable artifact or a content proof.
    /// Retains original paired source identities, rectangle origins and layout.
    NumericBinding(NumericFp8RankPlan),
    Dense(Arc<[f32]>),
    /// Native BF16 words; no expanded F32 resident mirror.
    Bf16(Arc<[u16]>),
}
#[derive(Debug, Clone)]
pub struct PreparedHybridRankParameter {
    binding: BoundParameter,
    pub(crate) global_shape: [usize; 2],
    pub(crate) local_shape: [usize; 2],
    pub(crate) layout: Arc<HybridRankLayout>,
    pub(crate) partition: HybridParameterPartition,
    pub(crate) storage: HybridRankStorage,
}
impl PreparedHybridRankParameter {
    pub fn binding(&self) -> &BoundParameter {
        &self.binding
    }
    pub fn validate_binding(
        &self,
        binding: &BoundParameter,
        layout: &HybridRankLayout,
    ) -> Result<()> {
        if !self.binding.shares_storage_with(binding)
            || self.binding.spec() != binding.spec()
            || self.binding.role() != binding.role()
            || self.layout.as_ref() != layout
            || layout.parameter_partition(binding)? != self.partition
        {
            return Err(model_error(
                "native rank binding/image/precision/ownership mismatch",
            ));
        }
        Ok(())
    }
    pub fn global_shape(&self) -> [usize; 2] {
        self.global_shape
    }
    pub fn local_shape(&self) -> [usize; 2] {
        self.local_shape
    }
    pub fn layout(&self) -> &HybridRankLayout {
        &self.layout
    }
    pub fn partition(&self) -> &HybridParameterPartition {
        &self.partition
    }
    pub fn storage(&self) -> &HybridRankStorage {
        &self.storage
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn dense_values(&self) -> Result<Arc<[f32]>> {
        match &self.storage {
            HybridRankStorage::Dense(values) => Ok(Arc::clone(values)),
            HybridRankStorage::Numeric(_)
            | HybridRankStorage::NumericBinding(_)
            | HybridRankStorage::Bf16(_) => Err(numeric_unsupported("rank.values_f32")),
        }
    }
}
impl StateDictMaterializer {
    /// Preparation-only rank materializer. Projection reads are restricted to
    /// selected rectangles; numeric payload remains encoded.
    #[cfg(feature = "cuda")]
    pub fn for_hybrid_cuda_rank(
        max_parameter_bytes: u64,
        layout: Arc<HybridRankLayout>,
        max_allocation_grant_bytes: usize,
    ) -> Result<Self> {
        let mut result = Self::new(max_parameter_bytes)?;
        result.cuda_rank = Some((layout, max_allocation_grant_bytes));
        Ok(result)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn cuda_rank_parameter(
        &self,
        binding: &BoundParameter,
    ) -> Result<Arc<PreparedParameter>> {
        let (layout, grant) = self
            .cuda_rank
            .as_ref()
            .ok_or_else(|| model_error("missing CUDA rank layout"))?;
        let prepared = self.prepare_hybrid_rank_parameter(binding, Arc::clone(layout), *grant)?;
        Ok(Arc::new(PreparedParameter {
            binding: binding.clone(),
            weight: PreparedParameterStorage::HybridRank(Arc::new(prepared)),
            scale: None,
        }))
    }

    /// Host preparation only; no CUDA feature, provider lookup or GPU required.
    /// Preserves the exact rank precision and ownership in the returned image.
    pub fn prepare_hybrid_rank_parameter(
        &self,
        binding: &BoundParameter,
        layout: Arc<HybridRankLayout>,
        grant: usize,
    ) -> Result<PreparedHybridRankParameter> {
        if self.tensor.is_some() {
            return Err(model_error(
                "cannot mix standard TP and hybrid rank materialization",
            ));
        }
        #[cfg(feature = "cpu-reference")]
        if self.hybrid.is_some() {
            return Err(model_error(
                "cannot mix CPU reference and native rank materialization",
            ));
        }
        let partition = layout.parameter_partition(binding)?;
        let local_shape = partition.local_shape()?;
        let shape = binding.weight().logical_shape();
        let global_shape = match shape {
            [n, k] => [*n, *k],
            [n, 1, k] => [*n, *k],
            [n] => [*n, 1],
            _ => return Err(model_error("unsupported CUDA rank tensor shape")),
        };
        let storage =
            if binding.numeric_fp8_encoding().is_some() {
                let plan = NumericFp8RankPlan::new(binding, Arc::clone(&layout))?;
                HybridRankStorage::Numeric(self.prepare_numeric_fp8_rank_linear(&plan, grant)?)
            } else {
                layout.validate_native_storage(binding)?;
                let source_policy = TypedSourceVectorPolicy::for_parameter(
                    layout.precision(),
                    binding.role(),
                    &binding.weight().slice().dtype,
                )?;
                if binding.scale().is_some()
                    || binding.weight().transform() != &TensorTransform::Identity
                    || !matches!(
                        binding.weight().slice().dtype,
                        CheckpointDType::Bf16 | CheckpointDType::F32
                    )
                {
                    return Err(model_error(
                        "CUDA rank requires identity dense or numeric FP8 storage",
                    ));
                }
                let budget = layout.parameter_budget(binding)?;
                let bytes = budget.weight_bytes;
                if budget.source_bytes.max(bytes) as u64 > self.max_parameter_bytes
                    || ferrule_backend::cuda_allocation_grant_bytes(bytes).is_none_or(|n| n > grant)
                {
                    return Err(model_error(
                        "CUDA rank dense rectangle exceeds read/grant budget",
                    ));
                }
                ensure_current(binding.weight(), binding)?;
                let mut slice = binding.weight().slice().clone();
                slice.shape = global_shape.to_vec();
                let reads = partition
                    .rows
                    .iter()
                    .map(|rows| {
                        self.reader.plan_2d_range(
                            &slice,
                            rows.clone(),
                            partition.columns.clone(),
                            binding.weight().source_identity(),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mixed = layout.precision().is_mixed();
                let mut values = Vec::with_capacity(if mixed { 0 } else { bytes / 4 });
                let mut words = Vec::with_capacity(if mixed { bytes / 2 } else { 0 });
                for read in reads {
                    let payload = self.reader.read_matrix(&read)?;
                    match source_policy {
                        TypedSourceVectorPolicy::NativeBf16 if mixed => words.extend(
                            payload
                                .bytes()
                                .as_chunks::<2>()
                                .0
                                .iter()
                                .map(|b| u16::from_le_bytes(*b)),
                        ),
                        TypedSourceVectorPolicy::Bf16ResidentFromF32 => {
                            words.extend(
                                payload.bytes().as_chunks::<4>().0.iter().map(|b| {
                                    crate::execution::bf16_rne_word(f32::from_le_bytes(*b))
                                }),
                            )
                        }
                        TypedSourceVectorPolicy::NativeF32 => values.extend(
                            payload
                                .bytes()
                                .as_chunks::<4>()
                                .0
                                .iter()
                                .map(|b| f32::from_le_bytes(*b)),
                        ),
                        TypedSourceVectorPolicy::NativeBf16 => {
                            values.extend(
                                payload.bytes().as_chunks::<2>().0.iter().map(|b| {
                                    half::bf16::from_bits(u16::from_le_bytes(*b)).to_f32()
                                }),
                            )
                        }
                    }
                    self.tensor_reads
                        .lock()
                        .map_err(|_| model_error("rank read accounting poisoned"))?
                        .push(TensorParallelPreparationRead {
                            parameter: binding.id(),
                            canonical: binding.canonical_id(),
                            role: binding.role().clone(),
                            bytes: read.read_plan().storage_bytes(),
                            rectangle: Some(read),
                        });
                }
                ensure_current(binding.weight(), binding)?;
                if mixed {
                    HybridRankStorage::Bf16(words.into())
                } else {
                    HybridRankStorage::Dense(values.into())
                }
            };
        Ok(PreparedHybridRankParameter {
            binding: binding.clone(),
            global_shape,
            local_shape,
            layout,
            partition,
            storage,
        })
    }
}

#[cfg(all(test, feature = "cuda"))]
mod source_vector_tests {
    use super::*;
    use crate::checkpoint::CheckpointTensorSlice;
    use crate::nn::{ModulePath, ParameterDType, ParameterSpec};
    use crate::transformer::{ExactNameMapper, NameMapping, StateDictBinder, StateDictSchema};

    #[test]
    fn mixed_source_vector_policy_preserves_f32_binding_and_exact_resident_rounding() {
        // Independent IEEE BF16 ties-to-even words, including signed zero.
        let bits = [0x3f80_8000_u32, 0x3f81_8000, 0xbf80_8000, 0x8000_0000];
        let expected = [0x3f80_u16, 0x3f82, 0xbf80, 0x8000];
        let path = std::env::temp_dir().join(format!("mixed-source-vector-{}", std::process::id()));
        std::fs::write(
            &path,
            bits.into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        for role in [
            TensorRole::LinearAttentionNorm,
            TensorRole::LinearAttentionALog,
            TensorRole::LinearAttentionDtBias,
            TensorRole::AttentionNorm,
        ] {
            let module = ModulePath::new("vector").unwrap();
            let mut schema = StateDictSchema::builder();
            schema
                .register_with_role(
                    ParameterSpec::new(
                        ParameterId::new(1),
                        module.clone(),
                        ParameterDType::F32,
                        vec![4],
                        ParameterResidency::layer(0),
                    )
                    .unwrap(),
                    role.clone(),
                )
                .unwrap();
            let mut mapper = ExactNameMapper::new();
            mapper
                .insert("vector", NameMapping::weight(module.clone()))
                .unwrap();
            let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
                .bind_slices(vec![CheckpointTensorSlice {
                    name: "vector".into(),
                    role: role.clone(),
                    path: path.clone(),
                    offset: 0,
                    bytes: 16,
                    shape: vec![4],
                    dtype: CheckpointDType::F32,
                }])
                .unwrap();
            let binding = bound.get(&module).unwrap();
            let p = StateDictMaterializer::new(16)
                .unwrap()
                .parameter(binding)
                .unwrap();
            assert!(
                p.values_bf16_words().is_err(),
                "strict native accessor must not cast"
            );
            if matches!(
                role,
                TensorRole::LinearAttentionNorm | TensorRole::LinearAttentionALog
            ) {
                assert_eq!(p.mixed_vector_bf16_words().unwrap(), expected);
            } else {
                assert!(p.mixed_vector_bf16_words().is_err());
            }
            assert_eq!(p.binding().weight().slice().dtype, CheckpointDType::F32);
            assert!(p.binding().shares_storage_with(binding));
            assert_eq!(
                p.values_f32()
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                bits
            );
            assert_eq!(
                TypedSourceVectorPolicy::for_parameter(
                    HybridRankPrecision::F32,
                    &role,
                    &CheckpointDType::F32
                )
                .unwrap(),
                TypedSourceVectorPolicy::NativeF32
            );
        }
        std::fs::remove_file(path).unwrap();
    }
}
