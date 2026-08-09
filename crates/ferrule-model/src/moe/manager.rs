//! Generic routed-MoE decoder resource manager.
//!
//! Owns expert materialization provider custody, residency requirements,
//! warmup planning, and hash-assisted transaction prefetch for any routed-MoE
//! decoder family. Resolver and resume-lease custody stay in the generic
//! decoder; model adapters only construct and initialize this manager.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use ferrule_common::execution::{ExecutionBatch, ExecutionTransactionId};
use ferrule_common::expert_residency::{ExpertResidencyControl, ExpertResidencyRequirements};
use ferrule_common::materialization_io::MaterializationResourceLimits;
use ferrule_common::{Error, Result};

use crate::decoder::{DecoderResourceManager, DecoderResourceSnapshot};
use crate::materialization::{
    MaterializationPlacement, MaterializationProvider, MaterializationRequest,
    MaterializationSourceCatalog,
};
use crate::moe::residency::{
    DeviceResidencyPlan, DeviceResidencyPlanError, ExpertLayerInventory, plan_device_residency,
};
use crate::moe::streaming::{ExpertId, ExpertLoadSource, ExpertSourceCatalog};

#[cfg(feature = "cuda")]
use crate::moe::cuda_materialization::{CudaExpertMaterializationOwner, CudaSharedExpertSubsystem};
#[cfg(feature = "cuda")]
use crate::moe::streaming::{ExpertIoPlan, ExpertStreamingReader};

/// Hash-assisted per-layer expert prefetch descriptor.
#[derive(Debug)]
pub struct RoutedMoePrefetchLayer {
    pub layer: usize,
    pub top_k: usize,
    pub expert_limit: usize,
    pub hash: Option<Arc<[usize]>>,
    pub rows: usize,
    pub cols: usize,
    pub sources: Arc<ExpertSourceCatalog>,
}

impl RoutedMoePrefetchLayer {
    fn requests(
        &self,
        placement: MaterializationPlacement,
        token_ids: &[u32],
    ) -> Result<Vec<MaterializationRequest>> {
        let Some(hash) = self.hash.as_deref() else {
            return Ok(Vec::new());
        };
        let mut selected = BTreeSet::new();
        for &token in token_ids {
            let row = token as usize;
            if row >= self.rows {
                return Err(manager_error(format!(
                    "router token {row} exceeds hash rows {}",
                    self.rows
                )));
            }
            let start = row
                .checked_mul(self.cols)
                .ok_or_else(|| manager_error("router hash offset overflow"))?;
            let end = start
                .checked_add(self.cols)
                .filter(|end| *end <= hash.len())
                .ok_or_else(|| manager_error("router hash table is truncated"))?;
            for &expert in hash[start..end].iter().take(self.top_k.min(self.cols)) {
                if expert >= self.expert_limit {
                    return Err(manager_error("router hash expert is out of range"));
                }
                selected.insert(expert);
            }
        }
        selected
            .into_iter()
            .map(|expert| {
                let layer = u32::try_from(self.layer)
                    .map_err(|_| manager_error("expert layer exceeds u32"))?;
                let expert_id =
                    u32::try_from(expert).map_err(|_| manager_error("expert index exceeds u32"))?;
                let resource = ferrule_common::MaterializedResourceId::routed_expert(
                    ferrule_common::LayerId::new(layer),
                    ferrule_common::ExpertId::new(expert_id),
                );
                MaterializationRequest::for_placement(
                    placement,
                    self.sources
                        .require_resource_source(ExpertId::new(self.layer, expert))?,
                    resource,
                )
            })
            .collect()
    }
}

/// Residency and expert-streaming numbers every routed-MoE family supplies
/// from its own config and prepare options.
#[derive(Debug, Clone, Copy)]
pub struct RoutedMoeResidencyConfig {
    pub num_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub reserved_device_bytes: u64,
    /// Zero disables the hotset floor.
    pub hotset_experts: usize,
    pub expert_upload_inflight: usize,
}

/// Generic routed-MoE resource manager consumed through `DecoderResourceManager`.
pub struct RoutedMoeResourceManager {
    placement: Option<MaterializationPlacement>,
    provider: Option<Box<dyn MaterializationProvider>>,
    requirements: Option<ExpertResidencyRequirements>,
    warmup: Vec<MaterializationRequest>,
    io_limits: MaterializationResourceLimits,
    prefetch: Box<[RoutedMoePrefetchLayer]>,
    shutdown: bool,
    #[cfg(feature = "cuda")]
    owner: Option<CudaExpertMaterializationOwner>,
    #[cfg(feature = "cuda")]
    device_plan: Option<DeviceResidencyPlan>,
}

impl std::fmt::Debug for RoutedMoeResourceManager {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RoutedMoeResourceManager")
            .field("placement", &self.placement)
            .field("prefetch_layers", &self.prefetch.len())
            .field("shutdown", &self.shutdown)
            .finish_non_exhaustive()
    }
}

impl RoutedMoeResourceManager {
    pub fn new(
        io_limits: MaterializationResourceLimits,
        prefetch: Vec<RoutedMoePrefetchLayer>,
    ) -> Result<Self> {
        Ok(Self {
            placement: None,
            provider: None,
            requirements: None,
            warmup: Vec::new(),
            io_limits,
            prefetch: prefetch.into_boxed_slice(),
            shutdown: false,
            #[cfg(feature = "cuda")]
            owner: None,
            #[cfg(feature = "cuda")]
            device_plan: None,
        })
    }

    pub fn shutdown(&mut self) -> Result<()> {
        if self.shutdown {
            return Ok(());
        }
        self.provider = None;
        self.warmup.clear();
        #[cfg(feature = "cuda")]
        {
            self.owner = None;
            self.device_plan = None;
        }
        self.shutdown = true;
        Ok(())
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub(crate) fn initialize_cuda(
        &mut self,
        sources: &Arc<MaterializationSourceCatalog<ExpertLoadSource>>,
        config: RoutedMoeResidencyConfig,
        reader: ExpertStreamingReader,
        model_instance: u64,
        free_device_bytes: u64,
        compute: ferrule_backend::cuda::operators::moe::CudaComputeStreamAuthority,
    ) -> Result<CudaSharedExpertSubsystem> {
        if self.owner.is_some() || self.provider.is_some() {
            return Err(execution_error("CUDA resources are already initialized"));
        }
        let plan = plan_expert_residency(sources, &config, free_device_bytes)?;
        let requirements = residency_requirements(model_instance, &plan)?;
        let placement = MaterializationPlacement::new(
            ferrule_common::ModelInstanceId::new(model_instance),
            ferrule_common::BackendId::new(1),
            ferrule_common::DeviceId::new(0),
        )?;
        let warmup = planned_warmup(sources, placement, &plan)?;
        let mut owner = CudaExpertMaterializationOwner::create(
            placement,
            self.io_limits,
            Arc::clone(sources),
            reader,
            config.num_routed_experts,
            &plan.layer_slot_capacities,
            compute,
        )?;
        let handle = owner.handle();
        self.provider = Some(
            owner
                .take_provider()
                .ok_or_else(|| execution_error("CUDA owner did not publish its provider"))?,
        );
        self.placement = Some(placement);
        self.requirements = Some(requirements);
        self.warmup = warmup;
        self.owner = Some(owner);
        self.device_plan = Some(plan);
        Ok(handle)
    }

    /// Physical I/O limits shared by every layer's expert source catalogs.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub(crate) fn physical_io_limits(
        layers: &[crate::moe::ExpertLayerSources],
        proposal_stages: &[crate::moe::ExpertLayerSources],
        reader: &ExpertStreamingReader,
        io_plan: ExpertIoPlan,
        config: &RoutedMoeResidencyConfig,
    ) -> Result<MaterializationResourceLimits> {
        use ferrule_common::materialization_io::MaterializationResourceRequirements;
        let reader_capacity = reader
            .physical_resource_capacity()?
            .ok_or_else(|| manager_error("CUDA-pinned io_uring topology is required"))?;
        let max_bytes = layers
            .iter()
            .chain(proposal_stages.iter())
            .flat_map(|layer| {
                layer
                    .source_catalog()
                    .iter()
                    .map(|(_, source)| source.bytes())
            })
            .max()
            .filter(|bytes| *bytes > 0)
            .ok_or_else(|| manager_error("prepared plan has no expert I/O demand"))?;
        let slots = u64::try_from(config.expert_upload_inflight + 1)
            .map_err(|_| manager_error("upload slots exceed u64"))?;
        let bytes = max_bytes
            .checked_mul(slots)
            .ok_or_else(|| manager_error("transfer bytes overflow"))?;
        MaterializationResourceLimits {
            capacity: MaterializationResourceRequirements {
                upload_slots: slots,
                h2d_bytes: bytes,
                install_slots: slots,
                device_install_bytes: bytes,
                ..reader_capacity
            },
            execution_reserve: io_plan.execution_reserve(max_bytes, config.num_experts_per_tok)?,
        }
        .validate()
        .map_err(Into::into)
    }
}

impl DecoderResourceManager for RoutedMoeResourceManager {
    fn expert_residency_requirements(&self) -> Option<ExpertResidencyRequirements> {
        self.requirements.clone()
    }

    fn expert_residency_control_installed(&self) -> bool {
        self.owner
            .as_ref()
            .is_some_and(CudaExpertMaterializationOwner::residency_control_installed)
    }

    fn install_expert_residency_control(
        &mut self,
        control: Box<dyn ExpertResidencyControl>,
    ) -> Result<()> {
        let expected = self
            .requirements
            .as_ref()
            .ok_or_else(|| execution_error("expert residency is unavailable"))?;
        if control.requirements() != *expected {
            return Err(execution_error("expert residency requirements mismatch"));
        }
        self.owner
            .as_ref()
            .ok_or_else(|| execution_error("CUDA expert owner is unavailable"))?
            .install_residency_control(control)
    }

    fn take_provider(&mut self) -> Option<Box<dyn MaterializationProvider>> {
        self.provider.take()
    }

    fn take_warmup_requests(&mut self) -> Result<Vec<MaterializationRequest>> {
        Ok(std::mem::take(&mut self.warmup))
    }

    fn transaction_prefetch_requests(
        &self,
        _transaction: ExecutionTransactionId,
        batch: &ExecutionBatch,
    ) -> Result<Vec<MaterializationRequest>> {
        let Some(placement) = self.placement else {
            return Ok(Vec::new());
        };
        self.prefetch
            .iter()
            .try_fold(Vec::new(), |mut output, layer| {
                output.extend(layer.requests(placement, batch.token_ids())?);
                Ok(output)
            })
    }

    fn materialization_placement(&self) -> Option<MaterializationPlacement> {
        self.placement
    }

    fn shutdown(&mut self) -> Result<()> {
        RoutedMoeResourceManager::shutdown(self)
    }

    fn observer_snapshot(&self) -> DecoderResourceSnapshot {
        DecoderResourceSnapshot {
            resolver_installed: false,
            shutdown: self.shutdown,
        }
    }
}

/// Plans device residency from the family's expert materialization catalog.
#[cfg(feature = "cuda")]
fn plan_expert_residency(
    sources: &MaterializationSourceCatalog<ExpertLoadSource>,
    config: &RoutedMoeResidencyConfig,
    free_device_bytes: u64,
) -> Result<DeviceResidencyPlan> {
    let mut layers = BTreeMap::<usize, (usize, u64)>::new();
    for entry in sources.iter() {
        let (layer, _) = entry
            .resource()
            .routed_expert_coordinates()
            .ok_or_else(|| manager_error("expert source has a non-expert resource ID"))?;
        let value = layers.entry(layer.get() as usize).or_default();
        value.0 += 1;
        value.1 = value.1.max(entry.descriptor().bytes());
    }
    let inventory = layers
        .into_iter()
        .map(|(execution_layer, (expert_count, frame_bytes))| {
            if expert_count != config.num_routed_experts {
                return Err(manager_error(format!(
                    "layer {execution_layer} has {expert_count} experts, expected {}",
                    config.num_routed_experts
                )));
            }
            Ok(ExpertLayerInventory {
                execution_layer,
                expert_count,
                frame_bytes,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    plan_device_residency(
        free_device_bytes,
        config.reserved_device_bytes,
        config.num_experts_per_tok,
        (config.hotset_experts != 0).then_some(config.hotset_experts),
        inventory,
    )
    .map_err(|source| Error::ModelSource {
        source: Box::new(source),
    })
}

#[cfg(feature = "cuda")]
fn residency_requirements(
    model_instance: u64,
    plan: &DeviceResidencyPlan,
) -> Result<ExpertResidencyRequirements> {
    let mut capacities = Vec::with_capacity(plan.layer_slot_capacities.len());
    for (expected, &(layer, capacity)) in plan.layer_slot_capacities.iter().enumerate() {
        if layer != expected {
            return Err(manager_error("expert execution layers are not contiguous"));
        }
        capacities.push(capacity);
    }
    Ok(ExpertResidencyRequirements::new(model_instance, capacities))
}

#[cfg(feature = "cuda")]
fn planned_warmup(
    sources: &MaterializationSourceCatalog<ExpertLoadSource>,
    placement: MaterializationPlacement,
    plan: &DeviceResidencyPlan,
) -> Result<Vec<MaterializationRequest>> {
    let mut remaining = plan
        .layer_slot_capacities
        .iter()
        .copied()
        .collect::<BTreeMap<_, _>>();
    let mut requests = Vec::new();
    for entry in sources.iter() {
        let (layer, _) = entry
            .resource()
            .routed_expert_coordinates()
            .ok_or_else(|| manager_error("expert catalog contains a non-expert resource"))?;
        let layer = layer.get() as usize;
        let slots = remaining
            .get_mut(&layer)
            .ok_or_else(|| Error::ModelSource {
                source: Box::new(DeviceResidencyPlanError::WarmupUnplannedLayer {
                    execution_layer: layer,
                }),
            })?;
        if *slots > 0 {
            requests.push(MaterializationRequest::for_placement(
                placement,
                entry.source(),
                entry.resource(),
            )?);
            *slots -= 1;
        }
    }
    if remaining.values().any(|missing| *missing != 0) {
        return Err(manager_error("warmup could not fill planned expert slots"));
    }
    Ok(requests)
}

fn manager_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("routed-MoE resource manager: {}", message.into()),
    }
}

fn execution_error(message: impl Into<String>) -> Error {
    Error::Execution {
        message: format!("routed-MoE resource manager: {}", message.into()),
    }
}
