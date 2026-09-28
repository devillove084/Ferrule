//! Image-scoped dense or compressed numeric bindings; no pointer-keyed cache.

use std::collections::BTreeMap;
use std::rc::Rc;

use crate::transformer::StandardTensorPlan;
use ferrule_backend::cuda::operators::linear::{
    CudaArtifactLinearHandle, CudaF32Buffer, CudaOperators,
};
use ferrule_common::{ParallelRankId, Result};

use crate::checkpoint::CheckpointDType;
use crate::nn::{ParameterId, ParameterResidency};
use crate::transformer::{
    BoundParameter, ExpertMetadata, PreparedDecoderGeneration, PreparedLinear, PreparedNorm,
    PreparedParameter, PreparedRope,
};

use super::{ExpertCacheKey, cuda_error};

pub(super) enum ResidentLinearStorage {
    Dense(CudaArtifactLinearHandle),
    Numeric(super::numeric_binding::NumericBinding),
}

pub(super) struct ResidentLinear {
    pub storage: ResidentLinearStorage,
    // Immutable catalog metadata only, never an Arc to a host weight payload.
    parameter: BoundParameter,
    pub bias: Option<CudaF32Buffer>,
    pub shape: (usize, usize),
    bias_values: Option<Vec<f32>>,
    pub bytes: usize,
}

pub(super) struct ResidentRope {
    pub table: PreparedRope,
    pub cosine: CudaF32Buffer,
    pub sine: CudaF32Buffer,
}

pub(super) struct Bindings {
    pub generation: PreparedDecoderGeneration,
    pub bounded_experts: bool,
    pub numeric_fp8: bool,
    pub expert_admission: Option<ExpertCacheKey>,
    allowed: BTreeMap<ParameterId, BoundParameter>,
    aliased_experts: bool,
    linear: BTreeMap<(ParameterId, Option<crate::support::TensorRole>), Rc<ResidentLinear>>,
    vectors: BTreeMap<ParameterId, Rc<CudaF32Buffer>>,
    ropes: Vec<Rc<ResidentRope>>,
    bytes: usize,
    pub tensor: Option<(StandardTensorPlan, ParallelRankId)>,
    expert_metadata: BTreeMap<(usize, usize), ExpertMetadata>,
    expert_sources: crate::transformer::operators::MetadataSourceSet,
    metadata_preflights: std::cell::Cell<u64>,
    metadata_source_checks: std::cell::Cell<u64>,
}

impl Bindings {
    pub fn new(parameters: &[BoundParameter]) -> Result<Self> {
        let mut allowed = BTreeMap::<ParameterId, BoundParameter>::new();
        for parameter in parameters {
            if let Some(previous) = allowed.get(&parameter.canonical_id()) {
                if !previous.shares_storage_with(parameter) {
                    return Err(cuda_error("conflicting parameter image identities"));
                }
            } else {
                allowed.insert(parameter.canonical_id(), parameter.clone());
            }
        }
        Ok(Self {
            generation: PreparedDecoderGeneration::take()?,
            bounded_experts: false,
            numeric_fp8: false,
            expert_admission: None,
            allowed,
            aliased_experts: parameters.iter().any(|p| {
                matches!(p.residency(), ParameterResidency::Expert { .. }) && p.is_alias()
            }),
            linear: BTreeMap::new(),
            vectors: BTreeMap::new(),
            ropes: Vec::new(),
            bytes: 0,
            tensor: None,
            expert_metadata: BTreeMap::new(),
            expert_sources: crate::transformer::operators::MetadataSourceSet::new(
                std::iter::empty::<&BoundParameter>(),
            ),
            metadata_preflights: std::cell::Cell::new(0),
            metadata_source_checks: std::cell::Cell::new(0),
        })
    }

    pub fn validate(&self, parameter: &PreparedParameter) -> Result<()> {
        let Some(binding) = self.allowed.get(&parameter.canonical_id()) else {
            return Err(cuda_error(format!(
                "parameter '{}' is outside the assigned image/layers",
                parameter.binding().path()
            )));
        };
        if !binding.shares_storage_with(parameter.binding()) {
            return Err(cuda_error("parameter belongs to another prepared image"));
        }
        if let Some(artifact) = parameter.numeric_fp8() {
            if !self.numeric_fp8 {
                return Err(cuda_error(
                    "numeric FP8 requires the explicit numeric profile",
                ));
            }
            artifact.provenance().source().validate_source_identity()?;
            return Ok(());
        }
        if !matches!(
            parameter.binding().weight().slice().dtype,
            CheckpointDType::F32 | CheckpointDType::Bf16
        ) || parameter.scale().is_some()
        {
            return Err(cuda_error(
                "standard CUDA supports only unscaled F32/BF16 weights",
            ));
        }
        Ok(())
    }

    /// Build the immutable expert plan once per prepared CUDA image. This is
    /// metadata and source snapshots only; it never reads weight payloads.
    pub fn prepare_expert_metadata(&mut self) -> Result<()> {
        if !self.bounded_experts && !self.numeric_fp8 {
            return Ok(());
        }
        if self.aliased_experts {
            return Err(cuda_error("routed expert metadata aliases are unsupported"));
        }
        let mut groups = BTreeMap::<(usize, usize), [Option<BoundParameter>; 3]>::new();
        for parameter in self.allowed.values() {
            let ParameterResidency::Expert { layer, expert } = parameter.residency() else {
                continue;
            };
            let index = match parameter.role() {
                crate::support::TensorRole::RoutedExpertGate => 0,
                crate::support::TensorRole::RoutedExpertUp => 1,
                crate::support::TensorRole::RoutedExpertDown => 2,
                _ => return Err(cuda_error("unexpected routed expert metadata role")),
            };
            let parts = groups.entry((*layer, *expert)).or_default();
            if parts[index].replace(parameter.clone()).is_some() {
                return Err(cuda_error("duplicate routed expert metadata role"));
            }
        }
        let mut metadata = BTreeMap::new();
        for ((layer, expert), [gate, up, down]) in groups {
            let missing = || cuda_error("incomplete routed expert metadata");
            let value = ExpertMetadata::new(
                layer,
                expert,
                [
                    gate.ok_or_else(missing)?,
                    up.ok_or_else(missing)?,
                    down.ok_or_else(missing)?,
                ],
                None,
            )?;
            if value.input_width() != value.output_width() {
                return Err(cuda_error(
                    "routed expert metadata input/output width mismatch",
                ));
            }
            for parameter in value.parameters() {
                self.validate_metadata(parameter)?;
            }
            metadata.insert((layer, expert), value);
        }
        self.expert_metadata = metadata;
        self.expert_sources = crate::transformer::operators::MetadataSourceSet::new(
            self.expert_metadata
                .values()
                .flat_map(|metadata| metadata.parameters()),
        );
        Ok(())
    }

    pub fn preflight_experts(&self) -> Result<()> {
        self.metadata_preflights
            .set(self.metadata_preflights.get().saturating_add(1));
        self.expert_sources.validate_counted(|| {
            self.metadata_source_checks
                .set(self.metadata_source_checks.get().saturating_add(1));
        })
    }

    pub fn metadata_stats(&self) -> super::ExpertMetadataPreflightStats {
        super::ExpertMetadataPreflightStats {
            prepared_experts: self.expert_metadata.len(),
            source_snapshots: self.expert_sources.len(),
            preflights: self.metadata_preflights.get(),
            source_checks: self.metadata_source_checks.get(),
        }
    }

    pub fn prepared_expert_metadata(&self, layer: usize, expert: usize) -> Result<&ExpertMetadata> {
        self.expert_metadata
            .get(&(layer, expert))
            .ok_or_else(|| cuda_error("missing prepared routed expert metadata"))
    }

    pub fn validate_metadata(&self, parameter: &BoundParameter) -> Result<()> {
        let Some(binding) = self.allowed.get(&parameter.canonical_id()) else {
            return Err(cuda_error(
                "expert metadata is outside the assigned image/layers",
            ));
        };
        if !binding.shares_storage_with(parameter) {
            return Err(cuda_error(
                "expert metadata belongs to another prepared image",
            ));
        }
        if binding.id() != parameter.id()
            || binding.role() != parameter.role()
            || binding.residency() != parameter.residency()
            || binding.spec().shape() != parameter.spec().shape()
            || self.tensor.is_some()
            || (parameter.numeric_fp8_encoding().is_some() && !self.numeric_fp8)
        {
            return Err(cuda_error(
                "expert metadata image/role/shape/profile mismatch",
            ));
        }
        Ok(())
    }

    pub fn metadata_linear(
        &self,
        parameter: &BoundParameter,
    ) -> Result<Option<Rc<ResidentLinear>>> {
        self.validate_metadata(parameter)?;
        let id = (
            parameter.canonical_id(),
            (self.bounded_experts || parameter.numeric_fp8_encoding().is_some())
                .then(|| parameter.role().clone()),
        );
        let Some(resident) = self.linear.get(&id) else {
            return Ok(None);
        };
        if !resident.parameter.shares_storage_with(parameter)
            || resident.parameter.role() != parameter.role()
            || resident.parameter.id() != parameter.id()
            || resident.shape != (parameter.spec().shape()[0], parameter.spec().shape()[1])
            || resident.bias.is_some()
        {
            return Err(cuda_error("resident expert differs from metadata identity"));
        }
        Ok(Some(Rc::clone(resident)))
    }

    pub fn has_metadata_expert(
        &self,
        metadata: &crate::transformer::ExpertMetadata,
    ) -> Result<bool> {
        for parameter in metadata.parameters() {
            if self.metadata_linear(parameter)?.is_none() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn linear(
        &mut self,
        ops: &CudaOperators,
        linear: &PreparedLinear,
    ) -> Result<Rc<ResidentLinear>> {
        self.validate(linear.parameter())?;
        let bounded_expert = self.bounded_experts
            && matches!(
                linear.parameter().residency(),
                ParameterResidency::Expert { .. }
            );
        if bounded_expert
            && !self.expert_admission.as_ref().is_some_and(|key| {
                key.parameters
                    .contains(&(linear.parameter().canonical_id(), linear.role().clone()))
            })
        {
            return Err(cuda_error(
                "bounded expert linear requires an admitted SwiGLU lease",
            ));
        }
        match (&self.tensor, linear.tensor_shard()) {
            (Some((tensor, rank)), Some(shard)) => {
                if shard.plan() != &tensor.linear_plan(linear)? || shard.rank() != *rank {
                    return Err(cuda_error("TP weight plan/rank mismatch"));
                }
            }
            (None, None) => {}
            _ => {
                return Err(cuda_error(
                    "TP requires checkpoint-local weights; no upload-time slicing",
                ));
            }
        }
        if linear.numeric_fp8().is_some() {
            super::numeric_binding::validate_linear(linear)?;
        }
        if linear.tensor_shard().is_none()
            && linear.numeric_fp8().is_none()
            && linear.weight()?.execution.activation_quantization.is_some()
        {
            return Err(cuda_error(
                "F32 standard CUDA does not support activation quantization",
            ));
        }
        // Aliased matrices may have different TP layouts (e.g. Q versus O).
        // Keep TP caches role-qualified, while preserving TP1 alias deduplication.
        let id = (
            linear.parameter().canonical_id(),
            (self.tensor.is_some() || bounded_expert || linear.numeric_fp8().is_some())
                .then(|| linear.role().clone()),
        );
        if let Some(resident) = self.linear.get(&id) {
            if resident.bias_values.as_deref() != linear.bias() {
                return Err(cuda_error("prepared linear bias changed within one image"));
            }
            return Ok(Rc::clone(resident));
        }
        if let Some(artifact) = linear.numeric_fp8() {
            let handle = super::numeric_binding::upload(ops, artifact)?;
            let layout = handle.layout();
            let bias = linear
                .bias()
                .map(|v| ops.upload_f32_buffer(v))
                .transpose()?;
            let bytes = handle.storage_bytes() + linear.bias().map_or(0, |v| v.len() * 4);
            let resident = Rc::new(ResidentLinear {
                parameter: linear.parameter().binding().clone(),
                storage: ResidentLinearStorage::Numeric(super::numeric_binding::NumericBinding {
                    artifact: handle,
                    source: artifact.provenance().source().clone(),
                }),
                bias,
                shape: (layout.n, layout.k),
                bias_values: linear.bias().map(<[f32]>::to_vec),
                bytes,
            });
            self.bytes += bytes;
            self.linear.insert(id, Rc::clone(&resident));
            return Ok(resident);
        }
        // Representation conversion only; no CPU activation or matrix math.
        let (values, shape) = if let Some(shard) = linear.tensor_shard() {
            shard.validate_source_identity()?;
            (
                shard.values_f32()?,
                (shard.local_shape()[0], shard.local_shape()[1]),
            )
        } else {
            (
                linear.parameter().values_f32()?,
                (linear.out_features(), linear.in_features()),
            )
        };
        let bytes = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let handle = ops.upload_f32_linear(&bytes, shape.0, shape.1)?;
        let bias = linear
            .bias()
            .map(|v| ops.upload_f32_buffer(v))
            .transpose()?;
        self.bytes += bytes.len() + linear.bias().map_or(0, |v| v.len() * 4);
        let resident = Rc::new(ResidentLinear {
            parameter: linear.parameter().binding().clone(),
            storage: ResidentLinearStorage::Dense(handle),
            bias,
            shape,
            bias_values: linear.bias().map(<[f32]>::to_vec),
            bytes: bytes.len() + linear.bias().map_or(0, |v| v.len() * 4),
        });
        self.linear.insert(id, Rc::clone(&resident));
        Ok(resident)
    }

    pub fn vector(
        &mut self,
        ops: &CudaOperators,
        parameter: &PreparedParameter,
    ) -> Result<Rc<CudaF32Buffer>> {
        self.validate(parameter)?;
        if parameter.numeric_fp8().is_some() {
            return Err(cuda_error("numeric FP8 is a linear binding, not a vector"));
        }
        if self.bounded_experts
            && matches!(parameter.residency(), ParameterResidency::Expert { .. })
        {
            return Err(cuda_error(
                "bounded expert weights cannot enter the vector cache",
            ));
        }
        let id = parameter.canonical_id();
        if let Some(resident) = self.vectors.get(&id) {
            return Ok(Rc::clone(resident));
        }
        let values = parameter.values_f32()?;
        let resident = Rc::new(ops.upload_f32_buffer(&values)?);
        self.bytes += values.len() * 4;
        self.vectors.insert(id, Rc::clone(&resident));
        Ok(resident)
    }

    pub fn norm(&mut self, ops: &CudaOperators, norm: &PreparedNorm) -> Result<Rc<CudaF32Buffer>> {
        self.vector(ops, norm.parameter())
    }

    pub fn rope(&mut self, ops: &CudaOperators, table: &PreparedRope) -> Result<Rc<ResidentRope>> {
        if let Some(resident) = self.ropes.iter().find(|r| r.table == *table) {
            return Ok(Rc::clone(resident));
        }
        let resident = Rc::new(ResidentRope {
            table: table.clone(),
            cosine: ops.upload_f32_buffer(table.cosine())?,
            sine: ops.upload_f32_buffer(table.sine())?,
        });
        self.bytes += (table.cosine().len() + table.sine().len()) * 4;
        self.ropes.push(Rc::clone(&resident));
        Ok(resident)
    }

    /// Remove only routed expert projections. Shared, attention and output
    /// bindings remain image-resident and are never subject to expert eviction.
    pub fn evict_expert(&mut self, key: &ExpertCacheKey) {
        for (id, role) in &key.parameters {
            if let Some(resident) = self.linear.remove(&(*id, Some(role.clone()))) {
                // Handles never escape this private binding layer.
                assert_eq!(Rc::strong_count(&resident), 1);
                self.bytes -= resident.bytes;
            }
        }
    }

    pub fn retain_on_unknown_completion(&mut self) {
        std::mem::forget(std::mem::take(&mut self.linear));
        std::mem::forget(std::mem::take(&mut self.vectors));
        std::mem::forget(std::mem::take(&mut self.ropes));
    }

    pub fn resident_bytes(&self) -> usize {
        self.bytes
    }
}

#[cfg(test)]
mod metadata_tests {
    use super::*;
    use crate::checkpoint::{CheckpointSourceFileIdentity, CheckpointTensorSlice};
    use crate::nn::{ModulePath, ParameterDType, ParameterSpec};
    use crate::support::TensorRole;
    use crate::transformer::{
        BoundStateDict, ExactNameMapper, ExpertMetadataBindings, NameMapping, StateDictBinder,
        StateDictSchema,
    };

    struct Fixture {
        directory: std::path::PathBuf,
        bound: BoundStateDict,
        mapper: ExactNameMapper,
        slices: Vec<CheckpointTensorSlice>,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.directory);
        }
    }
    impl Fixture {
        fn new(layers: usize, experts: usize, files: usize, bad_role: bool, alias: bool) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let directory = std::env::temp_dir().join(format!(
                "expert-metadata-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&directory).unwrap();
            for index in 0..files {
                std::fs::write(directory.join(format!("shard{index}")), [0x38, 0x80, 0x3f])
                    .unwrap();
            }
            let mut schema = StateDictSchema::builder();
            let mut mapper = ExactNameMapper::new();
            let mut slices = Vec::new();
            for layer in 0..layers {
                for expert in 0..experts {
                    for (index, mut role) in [
                        TensorRole::RoutedExpertGate,
                        TensorRole::RoutedExpertUp,
                        TensorRole::RoutedExpertDown,
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let ordinal = (layer * experts + expert) * 3 + index;
                        if bad_role && ordinal == layers * experts * 3 - 1 {
                            role = TensorRole::RouterLogits;
                        }
                        let path =
                            ModulePath::new(format!("layer{layer}.expert{expert}.part{index}"))
                                .unwrap();
                        let spec = ParameterSpec::new(
                            ParameterId::new(ordinal as u64 + 1),
                            path.clone(),
                            ParameterDType::F8E4M3,
                            [1, 1],
                            ParameterResidency::expert(layer, expert),
                        )
                        .unwrap()
                        .with_required_scale(ParameterDType::Bf16, [1, 1])
                        .unwrap();
                        schema.register_with_role(spec, role.clone()).unwrap();
                        let name = path.to_string();
                        mapper
                            .insert(&name, NameMapping::weight(path.clone()))
                            .unwrap();
                        mapper
                            .insert(format!("{name}_scale"), NameMapping::scale(path))
                            .unwrap();
                        let file = directory.join(format!("shard{}", ordinal % files));
                        slices.push(CheckpointTensorSlice {
                            name: name.clone(),
                            path: file.clone(),
                            role: role.clone(),
                            offset: 0,
                            bytes: 1,
                            shape: vec![1, 1],
                            dtype: CheckpointDType::F8E4M3,
                        });
                        slices.push(CheckpointTensorSlice {
                            name: format!("{name}_scale"),
                            path: file,
                            role,
                            offset: 1,
                            bytes: 2,
                            shape: vec![1, 1],
                            dtype: CheckpointDType::Bf16,
                        });
                    }
                }
            }
            if alias {
                let spec = ParameterSpec::new(
                    ParameterId::new((layers * experts * 3 + 1) as u64),
                    ModulePath::new("aliased_gate").unwrap(),
                    ParameterDType::F8E4M3,
                    [1, 1],
                    ParameterResidency::expert(0, 0),
                )
                .unwrap()
                .with_required_scale(ParameterDType::Bf16, [1, 1])
                .unwrap()
                .with_alias(ParameterId::new(1));
                schema
                    .register_with_role(spec, TensorRole::RoutedExpertGate)
                    .unwrap();
            }
            let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
                .bind_slices(slices.clone())
                .unwrap();
            Self {
                directory,
                bound,
                mapper,
                slices,
            }
        }
        fn bindings(&self) -> Result<Bindings> {
            let mut bindings = Bindings::new(self.bound.parameters())?;
            bindings.bounded_experts = true;
            bindings.numeric_fp8 = true;
            bindings.prepare_expert_metadata()?;
            Ok(bindings)
        }
    }

    #[test]
    fn full40_metadata_preflight_is_fourteen_sources_not_30720_projections() {
        let f = Fixture::new(40, 256, 14, false, false);
        let bindings = f.bindings().unwrap();
        let prepared = bindings.metadata_stats();
        assert_eq!(prepared.prepared_experts, 40 * 256);
        assert_eq!(prepared.source_snapshots, 14);
        let before = CheckpointSourceFileIdentity::counters();
        for forward in 1..=8 {
            bindings.preflight_experts().unwrap();
            let stats = bindings.metadata_stats();
            assert_eq!(stats.prepared_experts, 40 * 256);
            assert_eq!(stats.source_snapshots, 14);
            assert_eq!(stats.preflights, forward);
            assert_eq!(stats.source_checks, forward * 14);
        }
        let after = CheckpointSourceFileIdentity::counters();
        // The owner-local counter is exact even in a parallel test harness.
        // With this test selected alone the process-wide stat counter is exact too.
        eprintln!(
            "8 full40 metadata preflights: 112 snapshots, {} path metadata calls",
            after.path_metadata_calls - before.path_metadata_calls
        );
        let plan = bindings.prepared_expert_metadata(39, 255).unwrap();
        let reused = plan
            .for_bindings(ExpertMetadataBindings {
                parameters: plan.parameters(),
                activation_limit: Some(7.0),
            })
            .unwrap();
        assert_eq!(reused.expected_device_bytes(), 9);
        assert_eq!(reused.activation_limit(), Some(7.0));
        assert!(std::ptr::eq(plan.parameters(), reused.parameters()));
        assert_eq!(bindings.resident_bytes(), 0);
    }

    #[test]
    fn metadata_image_rejects_unrouted_role_alias_missing_projection_and_wrong_profile() {
        assert!(Fixture::new(2, 4, 2, true, false).bindings().is_err());
        assert!(Fixture::new(2, 4, 2, false, true).bindings().is_err());
        let f = Fixture::new(2, 4, 2, false, false);
        let mut partial =
            Bindings::new(&f.bound.parameters()[..f.bound.parameters().len() - 1]).unwrap();
        partial.numeric_fp8 = true;
        assert!(partial.prepare_expert_metadata().is_err());
        let mut wrong_profile = Bindings::new(f.bound.parameters()).unwrap();
        wrong_profile.bounded_experts = true;
        assert!(wrong_profile.prepare_expert_metadata().is_err());
        let bindings = f.bindings().unwrap();
        let alien = Fixture::new(2, 4, 2, false, false).bindings().unwrap();
        let plan = bindings.prepared_expert_metadata(0, 0).unwrap();
        assert!(
            plan.for_bindings(ExpertMetadataBindings {
                parameters: alien.prepared_expert_metadata(0, 0).unwrap().parameters(),
                activation_limit: None
            })
            .is_err()
        );
        assert!(
            plan.for_bindings(ExpertMetadataBindings {
                parameters: bindings
                    .prepared_expert_metadata(1, 0)
                    .unwrap()
                    .parameters(),
                activation_limit: None
            })
            .is_err()
        );
        let [gate, up, down] = plan.parameters().clone();
        assert!(
            plan.for_bindings(ExpertMetadataBindings {
                parameters: &[up, gate, down],
                activation_limit: None
            })
            .is_err()
        );
    }

    #[test]
    #[cfg(unix)]
    fn source_set_keeps_path_aliases_and_rejects_retargeted_symlink() {
        let f = Fixture::new(1, 2, 1, false, false);
        let alias = f.directory.join("alias");
        std::os::unix::fs::symlink("shard0", &alias).unwrap();
        let mut slices = f.slices.clone();
        for slice in &mut slices[6..] {
            slice.path = alias.clone();
        }
        let rebound = StateDictBinder::new(f.bound.schema(), &f.mapper)
            .bind_slices(slices)
            .unwrap();
        let mut bindings = Bindings::new(rebound.parameters()).unwrap();
        bindings.numeric_fp8 = true;
        bindings.prepare_expert_metadata().unwrap();
        assert_eq!(
            bindings.metadata_stats().source_snapshots,
            2,
            "do not dedup by canonical path alone"
        );
        bindings.preflight_experts().unwrap();
        std::fs::write(f.directory.join("replacement"), [0x38, 0x80, 0x3f]).unwrap();
        std::fs::remove_file(&alias).unwrap();
        std::os::unix::fs::symlink("replacement", &alias).unwrap();
        assert!(bindings.preflight_experts().is_err());
        assert_eq!(bindings.resident_bytes(), 0);
    }

    #[test]
    fn metadata_preflight_still_validates_timestamps_on_same_inode_and_size() {
        let f = Fixture::new(1, 2, 1, false, false);
        let bindings = f.bindings().unwrap();
        bindings.preflight_experts().unwrap();
        let path = f.directory.join("shard0");
        let before = std::fs::metadata(&path).unwrap();
        let modified = before.modified().unwrap() + std::time::Duration::from_secs(60);
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), before.len());
        assert!(bindings.preflight_experts().is_err());
        assert_eq!(bindings.metadata_stats().source_checks, 2);
    }

    #[test]
    fn source_set_dedup_keeps_full_snapshot_and_rechecks_same_size_replacement() {
        let f = Fixture::new(1, 2, 1, false, false);
        let bindings = f.bindings().unwrap();
        bindings.preflight_experts().unwrap();
        let file = f.directory.join("shard0");
        std::fs::rename(&file, f.directory.join("old")).unwrap();
        std::fs::write(&file, [0x38, 0x80, 0x3f]).unwrap();
        assert!(bindings.preflight_experts().is_err());
        assert!(bindings.preflight_experts().is_err());
        assert_eq!(bindings.metadata_stats().source_checks, 3);
        let rebound = StateDictBinder::new(f.bound.schema(), &f.mapper)
            .bind_slices(f.slices.clone())
            .unwrap();
        let sources = crate::transformer::operators::MetadataSourceSet::new(
            f.bound.parameters().iter().chain(rebound.parameters()),
        );
        assert_eq!(
            sources.len(),
            2,
            "same path with different generations must not dedup"
        );
        assert!(sources.validate().is_err());
        let mut fresh = Bindings::new(rebound.parameters()).unwrap();
        fresh.numeric_fp8 = true;
        fresh.prepare_expert_metadata().unwrap();
        fresh.preflight_experts().unwrap();
        let stale = bindings.prepared_expert_metadata(0, 0).unwrap();
        assert!(
            stale
                .for_bindings(ExpertMetadataBindings {
                    parameters: fresh.prepared_expert_metadata(0, 0).unwrap().parameters(),
                    activation_limit: None
                })
                .is_err()
        );
        assert_ne!(bindings.generation, fresh.generation);
        assert_eq!(bindings.resident_bytes(), 0);
    }
}
