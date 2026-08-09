use std::collections::{BTreeMap, BTreeSet};

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ferrule_common::{
    NameMappingError, StateDictBindingError, StateDictBindingIssue, StateDictMetadataError,
    StateDictTransformError,
};

use crate::checkpoint::{
    CheckpointDType, CheckpointSourceFileIdentity, CheckpointTensorSlice, HfSafetensorsInventory,
};
use crate::nn::{
    ModulePath, ParameterDType, ParameterId, ParameterPart, ParameterResidency, ParameterSpec,
    ParameterSpecError, ParameterTensorSpec, StorageEncoding,
};
use crate::support::TensorRole;

pub type StateDictSchemaError =
    ferrule_common::StateDictSchemaError<ModulePath, ParameterId, ParameterSpecError>;
pub type NameMapError = NameMappingError;
pub type BindingIssue = StateDictBindingIssue<
    ModulePath,
    ParameterPart,
    ParameterDType,
    NameMapError,
    ferrule_common::Error,
>;
pub type StateDictBindError = StateDictBindingError<BindingIssue>;

/// Immutable, ordered state-dict contract.
#[derive(Debug, Clone)]
pub struct StateDictSchema {
    parameters: Arc<[ParameterSpec]>,
    roles_by_id: BTreeMap<ParameterId, TensorRole>,
    by_path: BTreeMap<ModulePath, usize>,
    by_id: BTreeMap<ParameterId, usize>,
    canonical_by_id: BTreeMap<ParameterId, ParameterId>,
}

impl StateDictSchema {
    pub fn builder() -> StateDictSchemaBuilder {
        StateDictSchemaBuilder::new()
    }

    pub fn parameters(&self) -> &[ParameterSpec] {
        &self.parameters
    }

    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Number of distinct physical checkpoint tensor parts in this schema.
    pub fn storage_tensor_count(&self) -> usize {
        self.parameters
            .iter()
            .filter(|parameter| parameter.alias_of().is_none())
            .map(|parameter| 1 + usize::from(parameter.scale().tensor().is_some()))
            .sum()
    }

    pub fn get(&self, path: &ModulePath) -> Option<&ParameterSpec> {
        self.by_path.get(path).map(|&index| &self.parameters[index])
    }

    pub fn get_by_id(&self, id: ParameterId) -> Option<&ParameterSpec> {
        self.by_id.get(&id).map(|&index| &self.parameters[index])
    }

    pub fn role(&self, id: ParameterId) -> Option<&TensorRole> {
        self.roles_by_id.get(&id)
    }

    pub fn canonical_id(&self, id: ParameterId) -> Option<ParameterId> {
        self.canonical_by_id.get(&id).copied()
    }

    pub fn canonical(&self, id: ParameterId) -> Option<&ParameterSpec> {
        self.canonical_id(id)
            .and_then(|canonical| self.get_by_id(canonical))
    }

    pub fn aliases_of(&self, canonical: ParameterId) -> impl Iterator<Item = &ParameterSpec> {
        self.parameters.iter().filter(move |parameter| {
            self.canonical_id(parameter.id()) == Some(canonical) && parameter.id() != canonical
        })
    }
}

/// Builder that rejects ambiguous state-dict identities before binding.
#[derive(Debug, Default)]
pub struct StateDictSchemaBuilder {
    parameters: Vec<ParameterSpec>,
    roles_by_id: BTreeMap<ParameterId, TensorRole>,
    paths: BTreeSet<ModulePath>,
    ids: BTreeSet<ParameterId>,
}

impl StateDictSchemaBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers metadata without an executable semantic role.
    ///
    /// This is useful for state-dict tooling. Standard decoder resources reject
    /// `Unknown`, so executable recipes should use [`Self::register_with_role`].
    pub fn register(
        &mut self,
        parameter: ParameterSpec,
    ) -> Result<&mut Self, StateDictSchemaError> {
        self.register_with_role(parameter, TensorRole::Unknown)
    }

    pub fn register_with_role(
        &mut self,
        parameter: ParameterSpec,
        role: TensorRole,
    ) -> Result<&mut Self, StateDictSchemaError> {
        parameter
            .validate()
            .map_err(|source| StateDictSchemaError::InvalidParameter {
                path: parameter.path().clone(),
                source,
            })?;
        if !self.paths.insert(parameter.path().clone()) {
            return Err(StateDictSchemaError::DuplicatePath {
                path: parameter.path().clone(),
            });
        }
        if !self.ids.insert(parameter.id()) {
            self.paths.remove(parameter.path());
            return Err(StateDictSchemaError::DuplicateId { id: parameter.id() });
        }
        self.roles_by_id.insert(parameter.id(), role);
        self.parameters.push(parameter);
        Ok(self)
    }

    pub fn build(self) -> Result<StateDictSchema, StateDictSchemaError> {
        let by_path = self
            .parameters
            .iter()
            .enumerate()
            .map(|(index, parameter)| (parameter.path().clone(), index))
            .collect::<BTreeMap<_, _>>();
        let by_id = self
            .parameters
            .iter()
            .enumerate()
            .map(|(index, parameter)| (parameter.id(), index))
            .collect::<BTreeMap<_, _>>();

        for parameter in &self.parameters {
            let Some(target_id) = parameter.alias_of() else {
                continue;
            };
            let Some(&target_index) = by_id.get(&target_id) else {
                return Err(StateDictSchemaError::UnknownAliasTarget {
                    alias: parameter.path().clone(),
                    target: target_id,
                });
            };
            if target_id == parameter.id() {
                return Err(StateDictSchemaError::AliasCycle {
                    cycle: vec![parameter.id(), parameter.id()],
                });
            }
            let target = &self.parameters[target_index];
            ensure_alias_compatible(parameter, target)?;
        }

        let canonical_by_id = resolve_aliases(&self.parameters, &by_id)?;
        Ok(StateDictSchema {
            parameters: Arc::from(self.parameters),
            roles_by_id: self.roles_by_id,
            by_path,
            by_id,
            canonical_by_id,
        })
    }
}

fn ensure_alias_compatible(
    alias: &ParameterSpec,
    target: &ParameterSpec,
) -> Result<(), StateDictSchemaError> {
    let reason = if alias.weight() != target.weight() {
        Some("weight dtype or shape differs")
    } else if alias.scale() != target.scale() {
        Some("scale contract differs")
    } else if alias.optional() != target.optional() {
        Some("optional policy differs")
    } else if alias.residency() != target.residency() {
        Some("residency differs")
    } else {
        None
    };
    if let Some(reason) = reason {
        Err(StateDictSchemaError::IncompatibleAlias {
            alias: alias.path().clone(),
            target: target.path().clone(),
            reason,
        })
    } else {
        Ok(())
    }
}

fn resolve_aliases(
    parameters: &[ParameterSpec],
    by_id: &BTreeMap<ParameterId, usize>,
) -> Result<BTreeMap<ParameterId, ParameterId>, StateDictSchemaError> {
    let mut resolved = BTreeMap::new();
    for parameter in parameters {
        if resolved.contains_key(&parameter.id()) {
            continue;
        }
        let mut chain = Vec::new();
        let mut positions = BTreeMap::new();
        let mut current = parameter.id();
        let canonical = loop {
            if let Some(&canonical) = resolved.get(&current) {
                break canonical;
            }
            if let Some(&start) = positions.get(&current) {
                let mut cycle = chain[start..].to_vec();
                cycle.push(current);
                return Err(StateDictSchemaError::AliasCycle { cycle });
            }
            positions.insert(current, chain.len());
            chain.push(current);
            let current_parameter =
                &parameters[*by_id.get(&current).expect("all alias ids were validated")];
            match current_parameter.alias_of() {
                Some(target) => current = target,
                None => break current,
            }
        };
        for id in chain {
            resolved.insert(id, canonical);
        }
    }
    Ok(resolved)
}

/// Read-only metadata presented to an object-safe [`NameMapper`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalTensorMeta<'a> {
    pub dtype: &'a str,
    pub shape: &'a [usize],
    pub bytes: u64,
}

/// Deferred checkpoint transform attached to a mapped tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorTransform {
    Identity,
    /// Permutes dimensions without reading tensor bytes.
    Transpose {
        axes: Vec<usize>,
    },
    /// Reserved for one-to-many mappings. The strict binder rejects it until a
    /// materializer can preserve all split semantics.
    Split {
        axis: usize,
        index: usize,
        parts: usize,
    },
    /// Reserved for many-to-one mappings. The strict binder rejects it until a
    /// complete concat group can be proven.
    Concat {
        axis: usize,
        index: usize,
        parts: usize,
    },
}

impl TensorTransform {
    pub fn transpose_2d() -> Self {
        Self::Transpose { axes: vec![1, 0] }
    }
}

/// Canonical destination returned for one external tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameMapping {
    pub path: ModulePath,
    pub part: ParameterPart,
    pub transform: TensorTransform,
}

impl NameMapping {
    pub fn new(path: ModulePath, part: ParameterPart, transform: TensorTransform) -> Self {
        Self {
            path,
            part,
            transform,
        }
    }

    pub fn weight(path: ModulePath) -> Self {
        Self::new(path, ParameterPart::Weight, TensorTransform::Identity)
    }

    pub fn scale(path: ModulePath) -> Self {
        Self::new(path, ParameterPart::Scale, TensorTransform::Identity)
    }
}

/// Object-safe boundary between artifact naming and canonical state-dict paths.
pub trait NameMapper: Send + Sync {
    /// Returns `Ok(None)` only when the external tensor is not recognized.
    fn map(
        &self,
        external_name: &str,
        meta: ExternalTensorMeta<'_>,
    ) -> Result<Option<NameMapping>, NameMapError>;
}

/// Exact mapper suitable for generated recipes and small fixed schemas.
#[derive(Debug, Clone, Default)]
pub struct ExactNameMapper {
    mappings: BTreeMap<String, NameMapping>,
}

impl ExactNameMapper {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        external_name: impl Into<String>,
        mapping: NameMapping,
    ) -> Result<&mut Self, NameMapError> {
        let external_name = external_name.into();
        if external_name.is_empty() {
            return Err(NameMapError::EmptyExternalName);
        }
        if self.mappings.contains_key(&external_name) {
            return Err(NameMapError::DuplicateExactMapping {
                external: external_name,
            });
        }
        self.mappings.insert(external_name, mapping);
        Ok(self)
    }
}

impl NameMapper for ExactNameMapper {
    fn map(
        &self,
        external_name: &str,
        _meta: ExternalTensorMeta<'_>,
    ) -> Result<Option<NameMapping>, NameMapError> {
        Ok(self.mappings.get(external_name).cloned())
    }
}

/// One lazily bound physical tensor part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundTensorPart {
    slice: CheckpointTensorSlice,
    source_identity: CheckpointSourceFileIdentity,
    transform: TensorTransform,
    logical_shape: Vec<usize>,
    physical_shape: Vec<usize>,
    encoding: StorageEncoding,
}

impl BoundTensorPart {
    pub fn slice(&self) -> &CheckpointTensorSlice {
        &self.slice
    }

    pub fn source_identity(&self) -> &CheckpointSourceFileIdentity {
        &self.source_identity
    }

    pub fn transform(&self) -> &TensorTransform {
        &self.transform
    }

    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }

    pub fn physical_shape(&self) -> &[usize] {
        &self.physical_shape
    }

    pub const fn encoding(&self) -> StorageEncoding {
        self.encoding
    }

    pub fn source_is_current(&self) -> bool {
        self.source_identity.is_current()
    }
}

#[derive(Debug, PartialEq, Eq)]
struct BoundParameterStorage {
    weight: BoundTensorPart,
    scale: Option<BoundTensorPart>,
}

/// One canonical or tied parameter view in a bound state dict.
#[derive(Debug, Clone)]
pub struct BoundParameter {
    spec: ParameterSpec,
    role: TensorRole,
    canonical_id: ParameterId,
    storage: Arc<BoundParameterStorage>,
}

impl BoundParameter {
    pub fn spec(&self) -> &ParameterSpec {
        &self.spec
    }

    pub const fn id(&self) -> ParameterId {
        self.spec.id()
    }

    pub fn path(&self) -> &ModulePath {
        self.spec.path()
    }

    pub const fn canonical_id(&self) -> ParameterId {
        self.canonical_id
    }

    pub fn residency(&self) -> &ParameterResidency {
        self.spec.residency()
    }

    pub const fn role(&self) -> &TensorRole {
        &self.role
    }

    pub fn weight(&self) -> &BoundTensorPart {
        &self.storage.weight
    }

    pub fn scale(&self) -> Option<&BoundTensorPart> {
        self.storage.scale.as_ref()
    }

    pub fn is_alias(&self) -> bool {
        self.id() != self.canonical_id
    }

    pub fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }
}

/// Ordered, immutable collection of lazy checkpoint bindings.
#[derive(Debug, Clone)]
pub struct BoundStateDict {
    schema: StateDictSchema,
    parameters: Arc<[BoundParameter]>,
    by_path: BTreeMap<ModulePath, usize>,
    by_id: BTreeMap<ParameterId, usize>,
}

impl BoundStateDict {
    pub fn schema(&self) -> &StateDictSchema {
        &self.schema
    }

    pub fn parameters(&self) -> &[BoundParameter] {
        &self.parameters
    }

    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    pub fn get(&self, path: &ModulePath) -> Option<&BoundParameter> {
        self.by_path.get(path).map(|&index| &self.parameters[index])
    }

    pub fn get_by_id(&self, id: ParameterId) -> Option<&BoundParameter> {
        self.by_id.get(&id).map(|&index| &self.parameters[index])
    }

    pub fn for_residency<'a>(
        &'a self,
        residency: &'a ParameterResidency,
    ) -> impl Iterator<Item = &'a BoundParameter> + 'a {
        self.parameters
            .iter()
            .filter(move |parameter| parameter.residency() == residency)
    }

    pub fn expert(&self, layer: usize, expert: usize) -> impl Iterator<Item = &BoundParameter> {
        self.parameters.iter().filter(move |parameter| {
            matches!(
                parameter.residency(),
                ParameterResidency::Expert {
                    layer: parameter_layer,
                    expert: parameter_expert,
                } if *parameter_layer == layer && *parameter_expert == expert
            )
        })
    }

    /// Revalidates every distinct source file snapshot without reading payloads.
    pub fn validate_source_identities(&self) -> bool {
        let mut checked = BTreeSet::new();
        self.parameters.iter().all(|parameter| {
            [&parameter.storage.weight]
                .into_iter()
                .chain(parameter.storage.scale.iter())
                .all(|part| {
                    let path = part.source_identity.catalog_path().to_path_buf();
                    checked.contains(&path)
                        || if part.source_is_current() {
                            checked.insert(path);
                            true
                        } else {
                            false
                        }
                })
        })
    }
}

/// Strict metadata-only binder for a schema and artifact name mapper.
pub struct StateDictBinder<'a> {
    schema: &'a StateDictSchema,
    mapper: &'a dyn NameMapper,
}

impl<'a> StateDictBinder<'a> {
    pub const fn new(schema: &'a StateDictSchema, mapper: &'a dyn NameMapper) -> Self {
        Self { schema, mapper }
    }

    /// Binds an HF inventory without reading any tensor payload bytes.
    pub fn bind_hf(
        &self,
        model_dir: impl AsRef<Path>,
        inventory: &HfSafetensorsInventory,
    ) -> Result<BoundStateDict, StateDictBindError> {
        let model_dir = model_dir.as_ref();
        self.bind_slices(
            inventory
                .tensors
                .iter()
                .map(|tensor| CheckpointTensorSlice::from_hf_inventory(model_dir, tensor)),
        )
    }

    /// Binds pre-indexed slices while capturing stable file identities only.
    pub fn bind_slices(
        &self,
        slices: impl IntoIterator<Item = CheckpointTensorSlice>,
    ) -> Result<BoundStateDict, StateDictBindError> {
        let mut issues = Vec::new();
        let mut seen = BTreeMap::<(ParameterId, ParameterPart), String>::new();
        let mut parts = BTreeMap::<(ParameterId, ParameterPart), BoundTensorPart>::new();
        let mut identities = BTreeMap::<PathBuf, CheckpointSourceFileIdentity>::new();

        for slice in slices {
            let external_name = slice.name.clone();
            let meta = ExternalTensorMeta {
                dtype: slice.dtype.as_str(),
                shape: &slice.shape,
                bytes: slice.bytes,
            };
            let mapping = match self.mapper.map(&external_name, meta) {
                Ok(Some(mapping)) => mapping,
                Ok(None) => {
                    issues.push(BindingIssue::UnexpectedTensor {
                        external: external_name,
                    });
                    continue;
                }
                Err(source) => {
                    issues.push(BindingIssue::NameMapping {
                        external: external_name,
                        source,
                    });
                    continue;
                }
            };
            let Some(mapped_spec) = self.schema.get(&mapping.path) else {
                issues.push(BindingIssue::UnknownCanonicalPath {
                    external: external_name,
                    path: mapping.path,
                });
                continue;
            };
            let canonical_id = self
                .schema
                .canonical_id(mapped_spec.id())
                .expect("schema parameters have canonical identities");
            let canonical = self
                .schema
                .get_by_id(canonical_id)
                .expect("canonical parameter exists");
            let expected = match expected_part(canonical, mapping.part) {
                Ok(expected) => expected,
                Err(issue) => {
                    issues.push(issue.with_external(external_name));
                    continue;
                }
            };
            let key = (canonical_id, mapping.part);
            if let Some(first) = seen.insert(key, external_name.clone()) {
                issues.push(BindingIssue::DuplicateTensorPart {
                    path: canonical.path().clone(),
                    part: mapping.part,
                    first,
                    duplicate: external_name,
                });
                continue;
            }

            let physical_shape = match transformed_shape(&slice.shape, &mapping.transform) {
                Ok(shape) => shape,
                Err(source) => {
                    issues.push(BindingIssue::InvalidTransform {
                        external: external_name,
                        path: canonical.path().clone(),
                        source,
                    });
                    continue;
                }
            };
            let actual_dtype = parameter_dtype(&slice.dtype);
            if !expected.dtype().accepts(&actual_dtype) {
                issues.push(BindingIssue::DTypeMismatch {
                    external: external_name,
                    path: canonical.path().clone(),
                    part: mapping.part,
                    expected: expected.dtype().allowed().to_vec(),
                    actual: actual_dtype,
                });
                continue;
            }
            if expected.physical_shape() != physical_shape {
                issues.push(BindingIssue::ShapeMismatch {
                    external: external_name,
                    path: canonical.path().clone(),
                    part: mapping.part,
                    expected: expected.physical_shape().to_vec(),
                    actual: physical_shape,
                });
                continue;
            }
            if let Err(source) = validate_tensor_byte_size(&slice) {
                issues.push(BindingIssue::InvalidByteSize {
                    external: external_name,
                    source,
                });
                continue;
            }
            let identity = match identities.get(&slice.path) {
                Some(identity) => identity.clone(),
                None => match CheckpointSourceFileIdentity::capture(&slice.path) {
                    Ok(identity) => {
                        identities.insert(slice.path.clone(), identity.clone());
                        identity
                    }
                    Err(source) => {
                        issues.push(BindingIssue::SourceIdentity {
                            external: external_name,
                            path: slice.path.clone(),
                            source,
                        });
                        continue;
                    }
                },
            };
            let end = slice.offset.checked_add(slice.bytes);
            if end.is_none_or(|end| end > identity.length()) {
                issues.push(BindingIssue::SourceRange {
                    external: external_name,
                    path: slice.path.clone(),
                    offset: slice.offset,
                    bytes: slice.bytes,
                    source_bytes: identity.length(),
                });
                continue;
            }
            parts.insert(
                key,
                BoundTensorPart {
                    slice,
                    source_identity: identity,
                    transform: mapping.transform,
                    logical_shape: expected.logical_shape().to_vec(),
                    physical_shape: expected.physical_shape().to_vec(),
                    encoding: expected.encoding(),
                },
            );
        }

        let mut storage_by_canonical = BTreeMap::new();
        for parameter in self
            .schema
            .parameters()
            .iter()
            .filter(|parameter| parameter.alias_of().is_none())
        {
            let weight_key = (parameter.id(), ParameterPart::Weight);
            let scale_key = (parameter.id(), ParameterPart::Scale);
            let weight = parts.remove(&weight_key);
            let scale = parts.remove(&scale_key);
            match (weight, scale) {
                (None, None) if parameter.optional() => {}
                (None, None) => issues.push(BindingIssue::MissingParameter {
                    path: parameter.path().clone(),
                }),
                (None, Some(_)) => issues.push(BindingIssue::OrphanScale {
                    path: parameter.path().clone(),
                }),
                (Some(weight), scale) => {
                    if parameter.scale().is_required() && scale.is_none() {
                        issues.push(BindingIssue::MissingScale {
                            path: parameter.path().clone(),
                        });
                    } else {
                        storage_by_canonical.insert(
                            parameter.id(),
                            Arc::new(BoundParameterStorage { weight, scale }),
                        );
                    }
                }
            }
        }

        if !issues.is_empty() {
            return Err(StateDictBindError::new(issues));
        }

        let parameters = self
            .schema
            .parameters()
            .iter()
            .filter_map(|spec| {
                let canonical_id = self
                    .schema
                    .canonical_id(spec.id())
                    .expect("schema parameter has a canonical identity");
                storage_by_canonical
                    .get(&canonical_id)
                    .map(|storage| BoundParameter {
                        spec: spec.clone(),
                        role: self
                            .schema
                            .role(spec.id())
                            .expect("schema parameter has a semantic role")
                            .clone(),
                        canonical_id,
                        storage: Arc::clone(storage),
                    })
            })
            .collect::<Vec<_>>();
        let by_path = parameters
            .iter()
            .enumerate()
            .map(|(index, parameter)| (parameter.path().clone(), index))
            .collect();
        let by_id = parameters
            .iter()
            .enumerate()
            .map(|(index, parameter)| (parameter.id(), index))
            .collect();
        Ok(BoundStateDict {
            schema: self.schema.clone(),
            parameters: Arc::from(parameters),
            by_path,
            by_id,
        })
    }
}

fn expected_part(
    parameter: &ParameterSpec,
    part: ParameterPart,
) -> Result<&ParameterTensorSpec, BindingIssue> {
    match part {
        ParameterPart::Weight => Ok(parameter.weight()),
        ParameterPart::Scale => {
            parameter
                .scale()
                .tensor()
                .ok_or_else(|| BindingIssue::UnexpectedParameterPart {
                    external: String::new(),
                    path: parameter.path().clone(),
                    part,
                })
        }
    }
}

fn transformed_shape(
    shape: &[usize],
    transform: &TensorTransform,
) -> Result<Vec<usize>, StateDictTransformError> {
    match transform {
        TensorTransform::Identity => Ok(shape.to_vec()),
        TensorTransform::Transpose { axes } => {
            if axes.len() != shape.len() {
                return Err(StateDictTransformError::TransposeRankMismatch {
                    rank: shape.len(),
                    axes: axes.clone(),
                });
            }
            let mut seen = vec![false; axes.len()];
            let mut transformed = Vec::with_capacity(shape.len());
            for &axis in axes {
                if axis >= shape.len() {
                    return Err(StateDictTransformError::TransposeAxisOutOfRange {
                        axis,
                        rank: shape.len(),
                    });
                }
                if seen[axis] {
                    return Err(StateDictTransformError::DuplicateTransposeAxis { axis });
                }
                seen[axis] = true;
                transformed.push(shape[axis]);
            }
            Ok(transformed)
        }
        TensorTransform::Split { .. } => Err(StateDictTransformError::SplitUnsupported),
        TensorTransform::Concat { .. } => Err(StateDictTransformError::ConcatUnsupported),
    }
}

fn parameter_dtype(dtype: &CheckpointDType) -> ParameterDType {
    ParameterDType::from_storage_name(dtype.as_str())
}

fn validate_tensor_byte_size(slice: &CheckpointTensorSlice) -> Result<(), StateDictMetadataError> {
    let dtype = parameter_dtype(&slice.dtype);
    let Some(element_bytes) = dtype.element_size_bytes() else {
        return Ok(());
    };
    let elements = slice.shape.iter().try_fold(1u64, |elements, &dimension| {
        let dimension = u64::try_from(dimension)
            .map_err(|_| StateDictMetadataError::DimensionOutOfRange { dimension })?;
        elements
            .checked_mul(dimension)
            .ok_or(StateDictMetadataError::ElementCountOverflow)
    })?;
    let expected = elements
        .checked_mul(element_bytes as u64)
        .ok_or(StateDictMetadataError::ByteSizeOverflow)?;
    if expected == slice.bytes {
        Ok(())
    } else {
        Err(StateDictMetadataError::ByteSizeMismatch {
            expected,
            actual: slice.bytes,
        })
    }
}
