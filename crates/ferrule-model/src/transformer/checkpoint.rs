//! Strict Hugging Face checkpoint discovery and decoder resource binding.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ferrule_common::{Error, Result};

use crate::checkpoint::{
    CheckpointSourceCatalog, CheckpointTensorSlice, HfSafetensorsIndex, HfSafetensorsInventory,
};
use crate::execution::{
    ExecutableStage, PreparedExecutable, ResourceLayout, ResourceManifest, StageResourceUse,
    TransformerStage, WorkspaceClaim,
};
use crate::materialization::{
    HF_SAFETENSORS_ROUTED_EXPERT_V1, HF_SAFETENSORS_TENSOR_BUNDLE_V1, MaterializationSourceCatalog,
    MaterializationSourceEntry, ResourceSource,
};
use crate::moe::streaming::{
    ExpertId, ExpertLoadSource, ExpertMatrixKind, ExpertSourceCatalog, ExpertTensorComponent,
    ExpertTensorKey, ExpertTensorSlice,
};
use crate::nn::{ModulePath, ParameterId, ParameterPart, ParameterResidency};
use crate::spec::ModelFamily;
use crate::support::TensorRole;
use ferrule_common::{MaterializedResourceId, MaterializedResourceKind};

use super::{
    BoundParameter, BoundStateDict, DecoderAttachmentSpec, DecoderModelSpec, DecoderRecipe,
    DecoderRecipeError, NameMapper, StateDictBindError, StateDictBinder, StateDictSchema,
};

/// Model-family load boundary over one recipe configuration.
///
/// It keeps artifact discovery/binding separate from graph materialization. Tensor
/// payloads remain unread until `StateDictMaterializer` prepares selected weights.
pub struct DecoderLoadOptions<'a> {
    recipe: &'a dyn DecoderRecipe,
    config: &'a serde_json::Value,
}

impl<'a> DecoderLoadOptions<'a> {
    pub const fn new(recipe: &'a dyn DecoderRecipe, config: &'a serde_json::Value) -> Self {
        Self { recipe, config }
    }

    pub fn bind_slices(
        &self,
        slices: impl IntoIterator<Item = crate::checkpoint::CheckpointTensorSlice>,
    ) -> Result<BoundDecoderResources> {
        let output = self.recipe.build(self.config).map_err(recipe_error)?;
        let (spec, schema, mapper) = output.into_parts();
        let state_dict = StateDictBinder::new(&schema, mapper.as_ref())
            .bind_slices(slices)
            .map_err(binding_error)?;
        BoundDecoderResources::new(spec, Arc::new(state_dict))
    }

    pub fn open_hf(
        &self,
        model_dir: impl AsRef<Path>,
        family: ModelFamily,
    ) -> Result<BoundDecoderResources> {
        Ok(self.open_hf_checkpoint(model_dir, family)?.into_resources())
    }

    pub fn open_hf_checkpoint(
        &self,
        model_dir: impl AsRef<Path>,
        family: ModelFamily,
    ) -> Result<HFDecoderCheckpoint> {
        let output = self.recipe.build(self.config).map_err(recipe_error)?;
        let (spec, schema, mapper) = output.into_parts();
        HFDecoderCheckpoint::open(model_dir, family, spec, &schema, mapper.as_ref())
    }
}

/// Strict HF index/header view plus the lazily bound canonical state dict.
#[derive(Debug, Clone)]
pub struct HFDecoderCheckpoint {
    model_dir: PathBuf,
    index: HfSafetensorsIndex,
    inventory: HfSafetensorsInventory,
    resources: BoundDecoderResources,
}

impl HFDecoderCheckpoint {
    /// Open index and shard headers, require exact index/header agreement, and
    /// bind all tensors without reading payload bytes.
    pub fn open(
        model_dir: impl AsRef<Path>,
        family: ModelFamily,
        spec: DecoderModelSpec,
        schema: &StateDictSchema,
        mapper: &dyn NameMapper,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let index_path = model_dir.join("model.safetensors.index.json");
        let index = HfSafetensorsIndex::open(&index_path)?;
        let missing_shards = index.missing_shards(model_dir);
        if !missing_shards.is_empty() {
            return Err(model_error(format!(
                "safetensors index references missing shards: {}",
                display_paths(&missing_shards)
            )));
        }
        if index.shard_names().iter().any(|name| name.is_empty()) {
            return Err(model_error(
                "safetensors index contains an empty shard name",
            ));
        }
        let inventory = HfSafetensorsInventory::from_index(model_dir, family, &index)?;
        Self::from_inventory(model_dir, index, inventory, spec, schema, mapper)
    }

    pub fn from_inventory(
        model_dir: impl AsRef<Path>,
        index: HfSafetensorsIndex,
        inventory: HfSafetensorsInventory,
        spec: DecoderModelSpec,
        schema: &StateDictSchema,
        mapper: &dyn NameMapper,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        validate_inventory(&index, &inventory)?;
        let state_dict = StateDictBinder::new(schema, mapper)
            .bind_hf(model_dir, &inventory)
            .map_err(binding_error)?;
        let resources = BoundDecoderResources::new(spec, Arc::new(state_dict))?;
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            index,
            inventory,
            resources,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub const fn index(&self) -> &HfSafetensorsIndex {
        &self.index
    }

    pub const fn inventory(&self) -> &HfSafetensorsInventory {
        &self.inventory
    }

    pub const fn resources(&self) -> &BoundDecoderResources {
        &self.resources
    }

    pub fn into_resources(self) -> BoundDecoderResources {
        self.resources
    }
}

/// One bound parameter paired with its checked matrix slice view.
#[derive(Debug, Clone)]
pub struct BoundMatrix {
    binding: BoundParameter,
    matrix: crate::checkpoint::CheckpointMatrixSlice,
}

impl BoundMatrix {
    pub fn new(binding: BoundParameter, label: &str) -> Result<Self> {
        let matrix = crate::checkpoint::CheckpointMatrixSlice::from_slice(
            binding.weight().slice().clone(),
            label,
        )?;
        Ok(Self { binding, matrix })
    }

    pub const fn binding(&self) -> &BoundParameter {
        &self.binding
    }

    pub const fn matrix(&self) -> &crate::checkpoint::CheckpointMatrixSlice {
        &self.matrix
    }
}

/// Semantic parameter directory consumed by the standard decoder graph.
#[derive(Debug, Clone)]
pub struct BoundDecoderResources {
    spec: DecoderModelSpec,
    state_dict: Arc<BoundStateDict>,
    static_by_role: BTreeMap<TensorRole, Vec<BoundParameter>>,
    layer_by_role: BTreeMap<(usize, TensorRole), Vec<BoundParameter>>,
    attachments: BTreeMap<ModulePath, BTreeMap<TensorRole, Vec<BoundParameter>>>,
    experts: BoundExpertCatalog,
    sources: BoundDecoderSourceCatalogs,
}

impl BoundDecoderResources {
    pub fn new(spec: DecoderModelSpec, state_dict: Arc<BoundStateDict>) -> Result<Self> {
        let mut static_by_role = BTreeMap::<TensorRole, Vec<BoundParameter>>::new();
        let mut layer_by_role = BTreeMap::<(usize, TensorRole), Vec<BoundParameter>>::new();
        let mut experts =
            BTreeMap::<(usize, usize), BTreeMap<TensorRole, Vec<BoundParameter>>>::new();
        let mut attachments =
            BTreeMap::<ModulePath, BTreeMap<TensorRole, Vec<BoundParameter>>>::new();
        let mut seen_resources = BTreeSet::<(ParameterId, TensorRole)>::new();

        for parameter in state_dict.parameters() {
            let role = parameter.role().clone();
            if role == TensorRole::Unknown {
                return Err(model_error(format!(
                    "bound parameter '{}' has no semantic tensor role",
                    parameter.path()
                )));
            }
            if !seen_resources.insert((parameter.canonical_id(), role.clone())) {
                continue;
            }
            match parameter.residency() {
                ParameterResidency::Static => {
                    static_by_role
                        .entry(role)
                        .or_default()
                        .push(parameter.clone());
                }
                ParameterResidency::Layer { layer } => {
                    if *layer >= spec.layers().len() {
                        return Err(model_error(format!(
                            "parameter '{}' targets layer {layer}, but the decoder has {} layers",
                            parameter.path(),
                            spec.layers().len()
                        )));
                    }
                    layer_by_role
                        .entry((*layer, role))
                        .or_default()
                        .push(parameter.clone());
                }
                ParameterResidency::Expert { layer, expert } => {
                    let Some(descriptor) = spec.layers().get(*layer) else {
                        return Err(model_error(format!(
                            "parameter '{}' targets missing expert layer {layer}",
                            parameter.path()
                        )));
                    };
                    let super::FeedForward::Moe(moe) = descriptor.feed_forward() else {
                        return Err(model_error(format!(
                            "parameter '{}' targets expert {layer}:{expert}, but layer {layer} is dense",
                            parameter.path()
                        )));
                    };
                    if *expert >= moe.router_spec().num_experts() {
                        return Err(model_error(format!(
                            "parameter '{}' targets expert {layer}:{expert}, but the layer has {} experts",
                            parameter.path(),
                            moe.router_spec().num_experts()
                        )));
                    }
                    experts
                        .entry((*layer, *expert))
                        .or_default()
                        .entry(role)
                        .or_default()
                        .push(parameter.clone());
                }
                ParameterResidency::Attachment { path } => {
                    attachments
                        .entry(path.clone())
                        .or_default()
                        .entry(role)
                        .or_default()
                        .push(parameter.clone());
                }
            }
        }

        let experts = BoundExpertCatalog { experts };
        let sources = BoundDecoderSourceCatalogs::from_parts(&spec, &state_dict, &experts)?;
        Ok(Self {
            spec,
            state_dict,
            static_by_role,
            layer_by_role,
            attachments,
            experts,
            sources,
        })
    }

    pub const fn spec(&self) -> &DecoderModelSpec {
        &self.spec
    }

    pub fn state_dict(&self) -> &BoundStateDict {
        &self.state_dict
    }

    pub fn static_parameters(&self, role: TensorRole) -> &[BoundParameter] {
        self.static_by_role.get(&role).map_or(&[], Vec::as_slice)
    }

    pub fn require_static(&self, role: TensorRole) -> Result<&BoundParameter> {
        require_unique(self.static_parameters(role.clone()), || {
            format!("static role {role}")
        })
    }

    pub fn require_static_shape(
        &self,
        role: TensorRole,
        shape: &[usize],
    ) -> Result<&BoundParameter> {
        require_shape(self.static_parameters(role.clone()), shape, || {
            format!("static role {role}")
        })
    }

    pub fn require_static_path(&self, path: &ModulePath) -> Result<&BoundParameter> {
        self.require_residency_path(path, &ParameterResidency::Static)
    }

    pub fn layer_parameters(&self, layer: usize, role: TensorRole) -> &[BoundParameter] {
        self.layer_by_role
            .get(&(layer, role))
            .map_or(&[], Vec::as_slice)
    }

    pub fn require_layer(&self, layer: usize, role: TensorRole) -> Result<&BoundParameter> {
        require_unique(self.layer_parameters(layer, role.clone()), || {
            format!("decoder layer {layer} role {role}")
        })
    }

    pub fn require_layer_shape(
        &self,
        layer: usize,
        role: TensorRole,
        shape: &[usize],
    ) -> Result<&BoundParameter> {
        require_shape(self.layer_parameters(layer, role.clone()), shape, || {
            format!("decoder layer {layer} role {role}")
        })
    }

    pub fn require_layer_path(&self, layer: usize, path: &ModulePath) -> Result<&BoundParameter> {
        self.require_residency_path(path, &ParameterResidency::layer(layer))
    }

    pub fn attachment_parameters(
        &self,
        attachment: &ModulePath,
        role: TensorRole,
    ) -> &[BoundParameter] {
        self.attachments
            .get(attachment)
            .and_then(|parameters| parameters.get(&role))
            .map_or(&[], Vec::as_slice)
    }

    pub fn require_attachment(
        &self,
        attachment: &ModulePath,
        role: TensorRole,
    ) -> Result<&BoundParameter> {
        require_unique(self.attachment_parameters(attachment, role.clone()), || {
            format!("attachment {attachment} role {role}")
        })
    }

    pub fn require_attachment_shape(
        &self,
        attachment: &ModulePath,
        role: TensorRole,
        shape: &[usize],
    ) -> Result<&BoundParameter> {
        require_shape(
            self.attachment_parameters(attachment, role.clone()),
            shape,
            || format!("attachment {attachment} role {role}"),
        )
    }

    pub fn require_attachment_path(
        &self,
        attachment: &ModulePath,
        path: &ModulePath,
    ) -> Result<&BoundParameter> {
        self.require_residency_path(
            path,
            &ParameterResidency::Attachment {
                path: attachment.clone(),
            },
        )
    }

    pub fn attachment_paths(&self) -> impl ExactSizeIterator<Item = &ModulePath> {
        self.attachments.keys()
    }

    pub const fn experts(&self) -> &BoundExpertCatalog {
        &self.experts
    }

    pub const fn source_catalogs(&self) -> &BoundDecoderSourceCatalogs {
        &self.sources
    }

    fn require_residency_path(
        &self,
        path: &ModulePath,
        expected: &ParameterResidency,
    ) -> Result<&BoundParameter> {
        let parameter = self
            .state_dict
            .get(path)
            .ok_or_else(|| model_error(format!("canonical parameter '{path}' is not bound")))?;
        if parameter.residency() != expected {
            return Err(model_error(format!(
                "parameter '{}' has residency {:?}, expected {expected:?}",
                parameter.path(),
                parameter.residency()
            )));
        }
        Ok(parameter)
    }

    pub fn prepared_executable(
        &self,
        active_layers: usize,
    ) -> Result<PreparedExecutable<TransformerStage>> {
        if active_layers == 0 || active_layers > self.spec.layers().len() {
            return Err(model_error(format!(
                "prepared decoder layer count {active_layers} is outside 1..={}",
                self.spec.layers().len()
            )));
        }
        let mut stages = Vec::with_capacity(2 + active_layers * 3 + self.attachments.len());
        stages.push(ExecutableStage::new(
            TransformerStage::Embed,
            self.stage_parameter_uses(&ParameterResidency::Static, |parameter| {
                parameter.role() == &TensorRole::TokenEmbedding
            }),
            WorkspaceClaim::NONE,
        ));
        for layer in 0..active_layers {
            let residency = ParameterResidency::layer(layer);
            for operation in [
                TransformerStage::Attention {
                    layer: layer as u32,
                },
                TransformerStage::Router {
                    layer: layer as u32,
                },
                TransformerStage::FeedForward {
                    layer: layer as u32,
                },
            ] {
                stages.push(ExecutableStage::new(
                    operation,
                    self.stage_parameter_uses(&residency, |parameter| {
                        parameter_stage(parameter) == operation
                    }),
                    WorkspaceClaim::NONE,
                ));
            }
        }
        stages.push(ExecutableStage::new(
            TransformerStage::Output,
            self.stage_parameter_uses(&ParameterResidency::Static, |parameter| {
                parameter.role() != &TensorRole::TokenEmbedding
            }),
            WorkspaceClaim::NONE,
        ));
        if active_layers == self.spec.layers().len() {
            for (index, attachment) in self.attachment_paths().enumerate() {
                let residency = ParameterResidency::Attachment {
                    path: attachment.clone(),
                };
                stages.push(ExecutableStage::new(
                    TransformerStage::Attachment {
                        index: index as u32,
                    },
                    self.stage_parameter_uses(&residency, |_| true),
                    WorkspaceClaim::NONE,
                ));
            }
        }
        let declared = stages
            .iter()
            .flat_map(ExecutableStage::resources)
            .map(|resource| resource.resource())
            .collect::<BTreeSet<_>>();
        let include_attachments = active_layers == self.spec.layers().len();
        PreparedExecutable::new(
            self.sources
                .manifests
                .iter()
                .filter(|manifest| {
                    declared.contains(&manifest.resource())
                        || manifest.resource().routed_expert_coordinates().is_some_and(
                            |(layer, _)| {
                                (layer.get() as usize) < active_layers || include_attachments
                            },
                        )
                })
                .cloned(),
            stages,
        )
        .map_err(Error::from)
    }

    fn stage_parameter_uses(
        &self,
        residency: &ParameterResidency,
        include: impl Fn(&BoundParameter) -> bool,
    ) -> Vec<StageResourceUse> {
        self.state_dict
            .for_residency(residency)
            .filter(|parameter| {
                !parameter.is_alias()
                    && !is_routed_expert_parameter(parameter)
                    && include(parameter)
            })
            .map(|parameter| {
                StageResourceUse::read(parameter_resource_id(parameter.canonical_id()))
            })
            .collect()
    }
}

/// Strict, lazy catalog of routed expert parameter bindings.
#[derive(Debug, Clone, Default)]
pub struct BoundExpertCatalog {
    experts: BTreeMap<(usize, usize), BTreeMap<TensorRole, Vec<BoundParameter>>>,
}

impl BoundExpertCatalog {
    pub fn count(&self) -> usize {
        self.experts.len()
    }

    pub fn contains(&self, layer: usize, expert: usize) -> bool {
        self.experts.contains_key(&(layer, expert))
    }

    pub fn parameters(&self, layer: usize, expert: usize, role: TensorRole) -> &[BoundParameter] {
        self.experts
            .get(&(layer, expert))
            .and_then(|parameters| parameters.get(&role))
            .map_or(&[], Vec::as_slice)
    }

    pub fn require(
        &self,
        layer: usize,
        expert: usize,
        role: TensorRole,
    ) -> Result<&BoundParameter> {
        require_unique(self.parameters(layer, expert, role.clone()), || {
            format!("expert {layer}:{expert} role {role}")
        })
    }

    pub fn require_shape(
        &self,
        layer: usize,
        expert: usize,
        role: TensorRole,
        shape: &[usize],
    ) -> Result<&BoundParameter> {
        require_shape(self.parameters(layer, expert, role.clone()), shape, || {
            format!("expert {layer}:{expert} role {role}")
        })
    }

    pub fn expert_ids(&self, layer: usize) -> impl Iterator<Item = usize> + '_ {
        self.experts
            .keys()
            .filter_map(move |&(candidate_layer, expert)| {
                (candidate_layer == layer).then_some(expert)
            })
    }
}

/// Provider descriptor that keeps lazy bindings attached to exact source custody.
#[derive(Debug, Clone)]
pub enum BoundDecoderMaterializationSource {
    Parameter(BoundParameter),
    RoutedExpert {
        layer: usize,
        expert: usize,
        parameters: Arc<[BoundParameter]>,
        source: ExpertLoadSource,
    },
}

impl BoundDecoderMaterializationSource {
    pub fn parameters(&self) -> &[BoundParameter] {
        match self {
            Self::Parameter(parameter) => std::slice::from_ref(parameter),
            Self::RoutedExpert { parameters, .. } => parameters,
        }
    }

    pub fn residency(&self) -> Option<&ParameterResidency> {
        self.parameters().first().map(BoundParameter::residency)
    }

    pub const fn expert_coordinates(&self) -> Option<(usize, usize)> {
        match self {
            Self::Parameter(_) => None,
            Self::RoutedExpert { layer, expert, .. } => Some((*layer, *expert)),
        }
    }

    pub const fn expert_source(&self) -> Option<&ExpertLoadSource> {
        match self {
            Self::Parameter(_) => None,
            Self::RoutedExpert { source, .. } => Some(source),
        }
    }
}

/// Exact catalogs derived from an already-bound state dict without name parsing.
#[derive(Debug, Clone)]
pub struct BoundDecoderSourceCatalogs {
    experts: Arc<ExpertSourceCatalog>,
    experts_by_layer: BTreeMap<usize, Arc<ExpertSourceCatalog>>,
    expert_materialization: Arc<MaterializationSourceCatalog<ExpertLoadSource>>,
    materialization: Arc<MaterializationSourceCatalog<BoundDecoderMaterializationSource>>,
    manifests: Arc<[ResourceManifest]>,
}

impl BoundDecoderSourceCatalogs {
    fn from_parts(
        spec: &DecoderModelSpec,
        state_dict: &BoundStateDict,
        bound_experts: &BoundExpertCatalog,
    ) -> Result<Self> {
        let mut entries = Vec::new();
        let mut manifests = Vec::new();
        let mut seen = BTreeSet::new();
        for parameter in state_dict.parameters() {
            if matches!(parameter.residency(), ParameterResidency::Expert { .. })
                || is_routed_expert_parameter(parameter)
                || !seen.insert(parameter.canonical_id())
            {
                continue;
            }
            let parameter = state_dict
                .get_by_id(parameter.canonical_id())
                .expect("every bound alias has its canonical binding");
            let manifest = parameter_manifest(parameter)?;
            let source = manifest
                .source()
                .expect("bound decoder parameter manifests are checkpoint-backed");
            let read_plan = manifest
                .checkpoint_bundle_source()
                .expect("bound decoder parameter manifests retain bundle custody")
                .read_plan(manifest.checkpoint_tensors())?;
            entries.push(MaterializationSourceEntry::new(
                parameter_resource_id(parameter.canonical_id()),
                source,
                read_plan,
                BoundDecoderMaterializationSource::Parameter(parameter.clone()),
            )?);
            manifests.push(manifest);
        }

        let mut expert_entries = Vec::new();
        let mut expert_manifests = Vec::new();
        let mut expert_parameters = BTreeMap::<(usize, usize), Arc<[BoundParameter]>>::new();
        for (&(layer, expert), roles) in &bound_experts.experts {
            register_expert_source(
                layer,
                expert,
                roles.values().flatten(),
                &mut expert_entries,
                &mut expert_manifests,
                &mut expert_parameters,
            )?;
        }
        if let Some(DecoderAttachmentSpec::Proposal(proposal)) = spec.attachments().first() {
            let target_layers = spec.layers().len();
            for stage in 0..proposal.stages().len() {
                let path = ModulePath::new(format!("attachments.mtp.{stage}"))
                    .map_err(|error| model_error(error.to_string()))?;
                let residency = ParameterResidency::Attachment { path };
                let mut attachment_experts = BTreeMap::<usize, Vec<&BoundParameter>>::new();
                let prefix = format!("attachments.mtp.{stage}.experts.");
                for parameter in state_dict.for_residency(&residency) {
                    let Some(expert) = parameter
                        .path()
                        .as_str()
                        .strip_prefix(&prefix)
                        .and_then(|suffix| suffix.split('.').next())
                        .and_then(|value| value.parse::<usize>().ok())
                    else {
                        continue;
                    };
                    attachment_experts
                        .entry(expert)
                        .or_default()
                        .push(parameter);
                }
                let execution_layer = target_layers
                    .checked_add(stage)
                    .ok_or_else(|| model_error("proposal execution layer overflow"))?;
                for (expert, parameters) in attachment_experts {
                    register_expert_source(
                        execution_layer,
                        expert,
                        parameters,
                        &mut expert_entries,
                        &mut expert_manifests,
                        &mut expert_parameters,
                    )?;
                }
            }
        }
        manifests.extend(expert_manifests);
        let experts = Arc::new(ExpertSourceCatalog::from_resource_sources(expert_entries));
        let experts_by_layer = expert_catalogs_by_layer(&experts)?;
        let expert_materialization = Arc::new(experts.materialization_sources()?);
        for entry in expert_materialization.iter() {
            let expert = entry
                .resource()
                .routed_expert_coordinates()
                .expect("expert catalog resources have expert coordinates");
            let layer = expert.0.get() as usize;
            let expert = expert.1.get() as usize;
            let parameters = expert_parameters
                .get(&(layer, expert))
                .expect("expert bindings and source entries are built together");
            entries.push(MaterializationSourceEntry::new(
                entry.resource(),
                entry.source(),
                entry.read_plan().clone(),
                BoundDecoderMaterializationSource::RoutedExpert {
                    layer,
                    expert,
                    parameters: Arc::clone(parameters),
                    source: entry.descriptor().clone(),
                },
            )?);
        }

        Ok(Self {
            experts,
            experts_by_layer,
            expert_materialization,
            materialization: Arc::new(MaterializationSourceCatalog::new(entries)?),
            manifests: Arc::from(manifests),
        })
    }

    pub const fn experts(&self) -> &Arc<ExpertSourceCatalog> {
        &self.experts
    }

    pub fn experts_for_layer(&self, layer: usize) -> Result<Arc<ExpertSourceCatalog>> {
        self.experts_by_layer.get(&layer).cloned().ok_or_else(|| {
            model_error(format!(
                "execution layer {layer} has no routed-expert sources"
            ))
        })
    }

    pub const fn expert_materialization(
        &self,
    ) -> &Arc<MaterializationSourceCatalog<ExpertLoadSource>> {
        &self.expert_materialization
    }

    pub const fn materialization(
        &self,
    ) -> &Arc<MaterializationSourceCatalog<BoundDecoderMaterializationSource>> {
        &self.materialization
    }

    pub fn manifests(&self) -> &[ResourceManifest] {
        &self.manifests
    }
}

fn expert_catalogs_by_layer(
    experts: &ExpertSourceCatalog,
) -> Result<BTreeMap<usize, Arc<ExpertSourceCatalog>>> {
    let layers = experts
        .iter()
        .map(|(id, _)| id.layer)
        .collect::<BTreeSet<_>>();
    layers
        .into_iter()
        .map(|layer| {
            let entries = experts
                .iter()
                .filter(|(id, _)| id.layer == layer)
                .map(|(id, source)| {
                    Ok((*id, source.clone(), experts.require_resource_source(*id)?))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((
                layer,
                Arc::new(ExpertSourceCatalog::from_resource_sources(entries)),
            ))
        })
        .collect()
}

fn is_routed_expert_parameter(parameter: &BoundParameter) -> bool {
    matches!(
        parameter.role(),
        TensorRole::RoutedExpertGate | TensorRole::RoutedExpertUp | TensorRole::RoutedExpertDown
    )
}

fn parameter_stage(parameter: &BoundParameter) -> TransformerStage {
    let layer = match parameter.residency() {
        ParameterResidency::Layer { layer } => *layer as u32,
        _ => return TransformerStage::Output,
    };
    match parameter.role() {
        TensorRole::RouterLogits | TensorRole::RouterBias | TensorRole::HashRouterTable => {
            TransformerStage::Router { layer }
        }
        TensorRole::FeedForwardNorm
        | TensorRole::DenseMlpGate
        | TensorRole::DenseMlpUp
        | TensorRole::DenseMlpDown
        | TensorRole::SharedExpertGate
        | TensorRole::SharedExpertUp
        | TensorRole::SharedExpertDown => TransformerStage::FeedForward { layer },
        TensorRole::AuxHiddenCompressor
            if parameter
                .path()
                .as_str()
                .contains(".hyper_connection.feed_forward.") =>
        {
            TransformerStage::FeedForward { layer }
        }
        _ => TransformerStage::Attention { layer },
    }
}

fn register_expert_source<'a>(
    layer: usize,
    expert: usize,
    parameters: impl IntoIterator<Item = &'a BoundParameter>,
    entries: &mut Vec<(ExpertId, ExpertLoadSource, ResourceSource)>,
    manifests: &mut Vec<ResourceManifest>,
    bindings: &mut BTreeMap<(usize, usize), Arc<[BoundParameter]>>,
) -> Result<()> {
    let mut parameters = parameters
        .into_iter()
        .filter(|parameter| !parameter.is_alias())
        .cloned()
        .collect::<Vec<_>>();
    parameters.sort_by_key(BoundParameter::canonical_id);
    parameters.dedup_by_key(|parameter| parameter.canonical_id());
    let (source, manifest) = expert_bundle_source(layer, expert, &parameters)?;
    let resource_source = manifest
        .source()
        .expect("expert manifests are checkpoint-backed");
    entries.push((ExpertId::new(layer, expert), source, resource_source));
    manifests.push(manifest);
    bindings.insert((layer, expert), Arc::from(parameters));
    Ok(())
}

pub const fn parameter_resource_id(id: ParameterId) -> MaterializedResourceId {
    let raw = id.get();
    MaterializedResourceId::new(
        MaterializedResourceKind::Parameter,
        (raw >> 32) as u32,
        raw as u32,
    )
}

fn parameter_manifest(parameter: &BoundParameter) -> Result<ResourceManifest> {
    let mut tensors = bound_parameter_parts(parameter)
        .into_iter()
        .map(|(part, binding, mut slice)| {
            slice.name = String::from_utf8(parameter_semantic(parameter, part, binding))
                .expect("parameter semantic names contain only UTF-8 metadata");
            slice
        })
        .collect::<Vec<_>>();
    tensors.sort_by(|left, right| {
        (&left.role, &left.name, &left.path, left.offset, left.bytes).cmp(&(
            &right.role,
            &right.name,
            &right.path,
            right.offset,
            right.bytes,
        ))
    });
    let source_catalog = CheckpointSourceCatalog::capture(&tensors)?;
    let bundle_source = source_catalog.bundle_source(
        b"bound-state-dict-parameter-v2",
        HF_SAFETENSORS_TENSOR_BUNDLE_V1,
        &tensors,
    )?;
    ResourceManifest::checkpoint(
        parameter_resource_id(parameter.canonical_id()),
        bundle_source,
        tensors,
        ResourceLayout::TensorBundle,
    )
    .map_err(Error::from)
}

fn expert_bundle_source(
    layer: usize,
    expert: usize,
    parameters: &[BoundParameter],
) -> Result<(ExpertLoadSource, ResourceManifest)> {
    let expert_id = ExpertId::new(layer, expert);
    let mut tensors = Vec::<(
        ExpertTensorSlice,
        crate::checkpoint::CheckpointSourceFileIdentity,
    )>::new();
    for parameter in parameters {
        let matrix = expert_matrix(parameter.role()).ok_or_else(|| {
            model_error(format!(
                "expert {layer}:{expert} parameter '{}' has unsupported role {}",
                parameter.path(),
                parameter.role()
            ))
        })?;
        for (part, binding, slice) in bound_parameter_parts(parameter) {
            tensors.push((
                ExpertTensorSlice {
                    key: ExpertTensorKey {
                        expert: expert_id,
                        matrix,
                    },
                    component: match part {
                        ParameterPart::Weight => ExpertTensorComponent::Weight,
                        ParameterPart::Scale => ExpertTensorComponent::Scale,
                    },
                    path: slice.path,
                    offset: slice.offset,
                    bytes: slice.bytes,
                    dtype: slice.dtype.as_str().to_owned(),
                    shape: slice.shape,
                },
                binding.source_identity().clone(),
            ));
        }
    }
    tensors.sort_by(|(left, _), (right, _)| {
        left.key
            .matrix
            .cmp(&right.key.matrix)
            .then_with(|| left.component.cmp(&right.component))
            .then_with(|| left.dtype.cmp(&right.dtype))
            .then_with(|| left.shape.cmp(&right.shape))
            .then_with(|| left.bytes.cmp(&right.bytes))
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.offset.cmp(&right.offset))
    });
    if tensors.is_empty() {
        return Err(model_error(format!(
            "expert {layer}:{expert} has no bound tensors"
        )));
    }
    let source_files = tensors
        .iter()
        .map(|(_, source_file)| source_file.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let expert_tensors = tensors
        .iter()
        .map(|(tensor, _)| tensor.clone())
        .collect::<Vec<_>>();
    let mut manifest_tensors = expert_tensors
        .iter()
        .map(expert_manifest_tensor)
        .collect::<Vec<_>>();
    manifest_tensors.sort_by(|left, right| {
        (&left.role, &left.name, &left.path, left.offset, left.bytes).cmp(&(
            &right.role,
            &right.name,
            &right.path,
            right.offset,
            right.bytes,
        ))
    });
    let source_catalog = CheckpointSourceCatalog::capture(&manifest_tensors)?;
    let bundle_source = source_catalog.bundle_source(
        b"hf-routed-expert-v4",
        HF_SAFETENSORS_ROUTED_EXPERT_V1,
        &manifest_tensors,
    )?;
    let resource = MaterializedResourceId::routed_expert(
        ferrule_common::LayerId::new(layer as u32),
        ferrule_common::ExpertId::new(expert as u32),
    );
    let manifest = ResourceManifest::checkpoint(
        resource,
        bundle_source,
        manifest_tensors,
        ResourceLayout::CheckpointEncoded,
    )?;
    Ok((
        ExpertLoadSource::HfLocalTensorSet {
            tensors: expert_tensors,
            source_files: Arc::from(source_files),
        },
        manifest,
    ))
}

fn expert_manifest_tensor(tensor: &ExpertTensorSlice) -> CheckpointTensorSlice {
    let role = match tensor.key.matrix {
        ExpertMatrixKind::Gate => TensorRole::RoutedExpertGate,
        ExpertMatrixKind::Up => TensorRole::RoutedExpertUp,
        ExpertMatrixKind::Down => TensorRole::RoutedExpertDown,
    };
    let component = match &tensor.component {
        ExpertTensorComponent::Weight => "weight",
        ExpertTensorComponent::Scale => "scale",
        ExpertTensorComponent::Other(component) => component,
    };
    CheckpointTensorSlice {
        name: format!("{}.{}", role.as_str(), component),
        role,
        path: tensor.path.clone(),
        offset: tensor.offset,
        bytes: tensor.bytes,
        dtype: crate::checkpoint::CheckpointDType::from_safetensors_dtype(&tensor.dtype),
        shape: tensor.shape.clone(),
    }
}

fn bound_parameter_parts(
    parameter: &BoundParameter,
) -> Vec<(
    ParameterPart,
    &super::BoundTensorPart,
    CheckpointTensorSlice,
)> {
    let mut weight = parameter.weight().slice().clone();
    weight.role = parameter.role().clone();
    let mut parts = vec![(ParameterPart::Weight, parameter.weight(), weight)];
    if let Some(scale) = parameter.scale() {
        let mut slice = scale.slice().clone();
        slice.role = parameter.role().clone();
        parts.push((ParameterPart::Scale, scale, slice));
    }
    parts
}

fn parameter_semantic(
    parameter: &BoundParameter,
    part: ParameterPart,
    binding: &super::BoundTensorPart,
) -> Vec<u8> {
    let mut semantic = parameter.path().as_str().as_bytes().to_vec();
    semantic.push(0);
    semantic.extend_from_slice(parameter.role().as_str().as_bytes());
    semantic.push(match part {
        ParameterPart::Weight => 0,
        ParameterPart::Scale => 1,
    });
    semantic.extend_from_slice(format!("{:?}", binding.transform()).as_bytes());
    semantic
}

fn expert_matrix(role: &TensorRole) -> Option<ExpertMatrixKind> {
    match role {
        TensorRole::RoutedExpertGate => Some(ExpertMatrixKind::Gate),
        TensorRole::RoutedExpertUp => Some(ExpertMatrixKind::Up),
        TensorRole::RoutedExpertDown => Some(ExpertMatrixKind::Down),
        _ => None,
    }
}

fn require_unique(
    parameters: &[BoundParameter],
    context: impl FnOnce() -> String,
) -> Result<&BoundParameter> {
    match parameters {
        [parameter] => Ok(parameter),
        [] => Err(model_error(format!("{} is missing", context()))),
        _ => Err(model_error(format!(
            "{} has {} bound parameters",
            context(),
            parameters.len()
        ))),
    }
}

fn require_shape<'a>(
    parameters: &'a [BoundParameter],
    shape: &[usize],
    context: impl FnOnce() -> String,
) -> Result<&'a BoundParameter> {
    let matches = parameters
        .iter()
        .filter(|parameter| parameter.weight().logical_shape() == shape)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [parameter] => Ok(*parameter),
        [] => Err(model_error(format!(
            "{} is missing shape {shape:?}",
            context()
        ))),
        _ => Err(model_error(format!(
            "{} has {} parameters with shape {shape:?}",
            context(),
            matches.len()
        ))),
    }
}

fn recipe_error(source: DecoderRecipeError) -> Error {
    Error::ModelSource {
        source: Box::new(source),
    }
}

fn binding_error(error: StateDictBindError) -> Error {
    Error::ModelSource {
        source: Box::new(error),
    }
}

fn validate_inventory(
    index: &HfSafetensorsIndex,
    inventory: &HfSafetensorsInventory,
) -> Result<()> {
    if !inventory.index_only_tensors.is_empty() || !inventory.header_only_tensors.is_empty() {
        return Err(model_error(format!(
            "safetensors index/header mismatch: index-only={:?}, header-only={:?}",
            inventory.index_only_tensors, inventory.header_only_tensors
        )));
    }
    for tensor in &inventory.tensors {
        let indexed_shard = index.weight_map.get(&tensor.name).ok_or_else(|| {
            model_error(format!(
                "header tensor '{}' is absent from index",
                tensor.name
            ))
        })?;
        if indexed_shard != &tensor.shard {
            return Err(model_error(format!(
                "tensor '{}' header shard '{}' disagrees with index shard '{}'",
                tensor.name, tensor.shard, indexed_shard
            )));
        }
    }
    Ok(())
}

fn display_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn model_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("HF decoder checkpoint: {}", message.into()),
    }
}
