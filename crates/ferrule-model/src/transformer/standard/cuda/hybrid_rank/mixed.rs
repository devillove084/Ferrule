//! Executable local BF16 block slice on the existing synchronous CUDA owner.
//! No collective, decoder admission, checkpoint read, or expert cache at execution.
use super::super::bf16::rows_bf16;
use super::*;
use crate::transformer::PreparedHybridRankParameter;
use crate::transformer::hybrid_parallel::mixed_decoded::DecodedDescriptor;
use crate::transformer::standard::mixed_decoded::{DecodedCustody, Lookup};
use ferrule_backend::cuda::operators::linear::{
    Bf16GemmExPlan, Bf16GemmLayout, CudaBf16Buffer, CudaNumericFp8Artifact,
    CudaNumericFp8Workspace, F32ToBf16RowsLayout, NumericFp8LinearPlan,
};
use ferrule_backend::cuda::operators::standard_bf16::CudaBf16RouteMetadata;
use ferrule_backend::cuda_plan::NumericFp8Precision as OperandPrecision;

#[path = "mixed_output.rs"]
mod routed_output;
use routed_output::{RoutedBatchPlan, RoutedBindingSlot, RoutedBindingSlots, fuse_pair_swiglu};

/// Opaque binding to one resident projection in one immutable rank image.
#[derive(Debug, Clone)]
pub struct CudaMixedRankLinear {
    generation: u64,
    slot: usize,
}
pub(in crate::transformer::standard::cuda) struct MixedRankImage {
    plan: HybridMixedBlockPlan,
    projections: Vec<Rc<Projection>>,
    routed_slots: RoutedBindingSlots,
    pending_routes: Vec<CudaBf16RouteMetadata>,
    published: bool,
    decoded: Option<DecodedCustody<RefCell<CudaBf16Buffer>>>,
    decoded_index: BTreeMap<(usize, usize, bool), usize>,
}
struct Projection {
    binding: BoundParameter,
    shape: [usize; 2],
    storage: Storage,
}
enum Storage {
    Bf16(CudaBf16Buffer),
    Numeric(Vec<NumericSegment>),
}
struct NumericSegment {
    artifact: CudaNumericFp8Artifact,
    workspace: Option<RefCell<CudaNumericFp8Workspace>>,
    candidate_only: bool,
}
impl MixedRankImage {
    pub(in crate::transformer::standard::cuda) fn is_published(&self) -> bool {
        self.published
    }

    pub(in crate::transformer::standard::cuda) fn needs_quarantine(&self) -> bool {
        self.decoded
            .as_ref()
            .is_some_and(|cache| cache.stats().quarantined)
    }

    pub(in crate::transformer::standard::cuda) fn resident_bytes(&self) -> usize {
        self.projections
            .iter()
            .map(|p| match &p.storage {
                Storage::Bf16(b) => b.len() * 2,
                Storage::Numeric(segments) => {
                    segments.iter().map(|s| s.artifact.storage_bytes()).sum()
                }
            })
            .sum()
    }
}
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparationStage {
    Projections,
    Vectors,
    Chunk,
    Attention,
    Lifecycle,
}

#[cfg(test)]
thread_local! {
    static PREPARATION_FAULT: std::cell::Cell<Option<(PreparationStage, bool)>> = const { std::cell::Cell::new(None) };
}

fn mixed_result_unknown<T>(result: &Result<T>) -> bool {
    result.as_ref().err().is_some_and(|error| {
        ferrule_common::error::ErrorChain::new(error)
            .downcast_ref::<crate::decoder::CompletionUnknown>()
            .is_some()
    })
}

impl CudaStandardDecoderOperators {
    #[cfg(test)]
    pub(super) fn mixed_prepare_checkpoint(&mut self, stage: PreparationStage) -> Result<()> {
        if let Some((at, unknown)) = PREPARATION_FAULT.get()
            && at == stage
        {
            PREPARATION_FAULT.set(None);
            self.lose_next_consumer_proof = unknown;
            return Err(cuda_error(format!(
                "injected Mixed preparation failure after {stage:?}"
            )));
        }
        Ok(())
    }

    fn mixed_numeric_completion(&mut self, poisoned: bool, result: Result<()>) -> Result<()> {
        if poisoned || mixed_result_unknown(&result) {
            self.poisoned = true;
            self.cache_quarantine();
            return result.and(Err(cuda_error(
                "Mixed numeric workspace completion Unknown",
            )));
        }
        result
    }

    pub fn mixed_rank_all_expert_shards_resident(&self, layer: usize) -> bool {
        !self.needs_quarantine()
            && self
                .mixed_rank_block
                .as_ref()
                .is_some_and(|i| i.published && i.plan.all_expert_shards_resident(layer))
    }

    /// Install an immutable local block slice in this owner. All payloads are
    /// uploaded before publication; no prepared CPU payload is retained.
    /// The owner's parameter directory must contain exactly these source identities.
    pub fn prepare_mixed_rank_block(
        &mut self,
        plan: HybridMixedBlockPlan,
        parameters: &[PreparedHybridRankParameter],
    ) -> Result<Vec<CudaMixedRankLinear>> {
        self.validate_mixed_preparation_owner()?;
        self.mixed_decoded_enabled = plan.decoded.is_some();
        self.mixed_preparation(|this| this.prepare_mixed_rank_block_pending(plan, parameters))
    }

    pub(super) fn validate_mixed_preparation_owner(&self) -> Result<()> {
        if self.needs_quarantine()
            || self.hybrid_rank.is_some()
            || self.mixed_rank_block.is_some()
            || self.mixed_gqa_block.is_some()
            || self.bounded_experts.is_some()
            || self.bf16_gemmex.is_some()
            || self.bf16_attention.is_some()
            || self.bf16_chunk_workspace.is_some()
            || !self.bf16_deltas.is_empty()
            || self.resident_parameter_bytes() != 0
            || self.root_submission.is_some()
        {
            return Err(cuda_error(
                "Mixed rank block requires unused healthy KeepAll owner",
            ));
        }
        Ok(())
    }

    /// Pending allocations stay in owner custody through every fence and unwind.
    /// No token/image is published until the entire preparation has completed.
    pub(super) fn mixed_preparation<T>(
        &mut self,
        prepare: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        self.mixed_preparation_with_completion(prepare, || Ok(()))
    }

    pub(super) fn mixed_preparation_with_completion<T>(
        &mut self,
        prepare: impl FnOnce(&mut Self) -> Result<T>,
        complete: impl FnOnce() -> Result<()>,
    ) -> Result<T> {
        let had_bf16 = self.bf16.is_some();
        let result = self.synchronous(|this| {
            let result = prepare(this);
            if mixed_result_unknown(&result)
                || ferrule_backend::cuda::operators::linear::CudaBf16GemmEx::preparation_completion_unknown(&this.ops)
            {
                this.poisoned = true;
                this.cache_quarantine();
            }
            result
        });
        // The image claim cannot advance until the outer owner operation has
        // completed all consumers, upload work and resource retirement.
        let result = match result {
            Ok(value) => {
                let completion = complete();
                if mixed_result_unknown(&completion) {
                    self.poisoned = true;
                    self.cache_quarantine();
                    self.quarantine_mixed_decoded();
                    std::mem::forget(value);
                    Err(completion.expect_err("unknown completion"))
                } else {
                    completion.map(|()| value)
                }
            }
            Err(error) => Err(error),
        };
        if result.is_ok() {
            self.mixed_rank_block
                .as_mut()
                .expect("prepared Mixed image")
                .published = true;
        } else if !self.needs_quarantine() {
            // synchronous has proved both streams complete. Never enter this
            // branch after Unknown, even if a subsequent fence might succeed.
            self.mixed_gqa_block = None;
            self.mixed_rank_block = None;
            self.mixed_decoded_enabled = false;
            self.bf16_gemmex = None;
            self.bf16_attention = None;
            self.bf16_chunk_workspace = None;
            self.bf16_deltas.clear();
            self.bf16 = had_bf16.then(super::super::bf16::State::default);
        }
        result
    }

    pub(super) fn prepare_mixed_rank_block_pending(
        &mut self,
        plan: HybridMixedBlockPlan,
        parameters: &[PreparedHybridRankParameter],
    ) -> Result<Vec<CudaMixedRankLinear>> {
        if parameters.len() != plan.bindings.len() {
            return Err(cuda_error("Mixed rank parameter count mismatch"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for (p, b) in parameters.iter().zip(&plan.bindings) {
            p.validate_binding(b, &plan.layout)?;
            self.bindings.validate_metadata(b)?;
            match p.storage() {
                HybridRankStorage::Numeric(n) => n.plan().validate_binding(b, &plan.layout)?,
                HybridRankStorage::NumericBinding(_) => {
                    return Err(cuda_error(
                        "retired Mixed host source cannot prepare a new image",
                    ));
                }
                _ => {}
            }
            if !seen.insert((b.id(), b.role().clone()))
                || !b.weight().source_is_current()
                || b.scale().is_some_and(|scale| !scale.source_is_current())
            {
                return Err(cuda_error("duplicate/stale Mixed prepared source binding"));
            }
        }
        let routed_slots = RoutedBindingSlots::prepare(
            self.image_generation(),
            parameters.len(),
            &plan.expert_counts,
            parameters.iter().enumerate().map(|(slot, p)| {
                let role = match p.binding().role() {
                    TensorRole::RoutedExpertGate => Some(0),
                    TensorRole::RoutedExpertUp => Some(1),
                    TensorRole::RoutedExpertDown => Some(2),
                    _ => None,
                };
                match (p.binding().residency(), role) {
                    (ParameterResidency::Expert { layer, expert }, Some(role)) => {
                        Ok(Some(RoutedBindingSlot {
                            slot,
                            layer: *layer,
                            expert: *expert,
                            role,
                        }))
                    }
                    (ParameterResidency::Expert { .. }, _) | (_, Some(_)) => {
                        Err(cuda_error("Mixed routed role/residency mismatch"))
                    }
                    _ => Ok(None),
                }
            }),
        )?;
        for (&layer, &count) in &plan.expert_counts {
            for expert in 0..count {
                let [gate, up, down] = routed_slots.get(self.image_generation(), layer, expert)?;
                let gate = parameters
                    .get(gate)
                    .ok_or_else(|| cuda_error("Mixed gate slot"))?;
                let up = parameters
                    .get(up)
                    .ok_or_else(|| cuda_error("Mixed up slot"))?;
                let down = parameters
                    .get(down)
                    .ok_or_else(|| cuda_error("Mixed down slot"))?;
                let [n, k] = gate.local_shape();
                if n == 0 || k == 0 || up.local_shape() != [n, k] || down.local_shape() != [k, n] {
                    return Err(cuda_error("Mixed routed prepared shard shape mismatch"));
                }
            }
        }
        let this = self;
        {
            // Use the existing outer-depth completion hook even without a
            // decoded cache: opaque route sources need the same late fence.
            this.mixed_decoded_enabled |= !plan.expert_counts.is_empty();
            this.prepare_bf16_gemmex(&plan.views)?;
            // Install custody before uploads: partially prepared images stay owned
            // through an error, panic or unknown-completion quarantine.
            let decoded = plan
                .decoded
                .as_ref()
                .map(|(budget, descriptors)| {
                    DecodedCustody::new(*budget, this.image_generation(), descriptors)
                })
                .transpose()?;
            let decoded_index = plan
                .decoded
                .as_ref()
                .map(|(_, ds)| {
                    ds.iter()
                        .enumerate()
                        .map(|(index, d)| {
                            (
                                (d.operands[0].0, d.operands[0].1, d.operands.len() == 2),
                                index,
                            )
                        })
                        .collect()
                })
                .unwrap_or_default();
            let pending_routes = Vec::with_capacity(usize::from(!plan.expert_counts.is_empty()));
            this.mixed_rank_block = Some(MixedRankImage {
                plan,
                projections: Vec::new(),
                routed_slots,
                pending_routes,
                published: false,
                decoded,
                decoded_index,
            });
            if let Some(cache) = this.mixed_rank_block.as_mut().unwrap().decoded.as_mut() {
                let arena_grants = cache.plan().arena_grants;
                // Exclude GemmEx storage and uploads; include every arena slice.
                let arena_before = this.ops.allocator_metrics();
                #[cfg(test)]
                let counters_before = this.ops.counters();
                let allocation = this
                    .ops
                    .allocate_bf16_arena_uninitialized(arena_grants / 2, arena_grants);
                match allocation {
                    // Dedicated allocation always grows the driver by exactly its
                    // full grant. Transfer custody before any metrics query can
                    // poll CUDA; install_arena retains the root before slicing.
                    Ok(root) => cache.install_arena(
                        RefCell::new(root),
                        arena_grants,
                        |root, offset, elements| {
                            #[cfg(test)]
                            decoded_gpu_tests::before_arena_view()?;
                            root.borrow().slice(offset, elements).map(RefCell::new)
                        },
                    )?,
                    Err(error) => {
                        // No root was returned. Preserve any actual growth from a
                        // failed allocation without retrying or refunding its claim.
                        let end = this.ops.allocator_metrics().driver_allocation_bytes;
                        let growth = end
                            .checked_sub(arena_before.driver_allocation_bytes)
                            .and_then(|growth| usize::try_from(growth).ok())
                            .unwrap_or(usize::MAX);
                        cache.arena_allocation_failed(growth)?;
                        return Err(error);
                    }
                }
                #[cfg(test)]
                {
                    let after = this.ops.allocator_metrics();
                    assert_eq!(
                        after
                            .allocation_requests
                            .checked_sub(arena_before.allocation_requests),
                        Some(1),
                        "arena root and slices must allocate exactly once",
                    );
                    assert_eq!(
                        after
                            .huge_allocations
                            .checked_sub(arena_before.huge_allocations),
                        Some(1),
                        "arena must have exactly one huge root",
                    );
                    assert_eq!(
                        after.granted_bytes.checked_sub(arena_before.granted_bytes),
                        Some(cache.plan().arena_grants as u64),
                        "arena root grant must match admission exactly",
                    );
                    assert_eq!(
                        after
                            .driver_allocations
                            .checked_sub(arena_before.driver_allocations),
                        Some(1),
                        "arena must own a dedicated driver allocation",
                    );
                    assert_eq!(
                        after
                            .driver_allocation_bytes
                            .checked_sub(arena_before.driver_allocation_bytes),
                        Some(arena_grants as u64),
                        "dedicated physical charge must equal the full arena grant",
                    );
                    assert_eq!(after.reuse_allocations, arena_before.reuse_allocations);
                    let counters_after = this.ops.counters();
                    assert_eq!(
                        counters_after.kernel_launches,
                        counters_before.kernel_launches
                    );
                    assert_eq!(
                        counters_after.host_to_device_copies,
                        counters_before.host_to_device_copies
                    );
                    assert_eq!(
                        counters_after.device_to_host_copies,
                        counters_before.device_to_host_copies
                    );
                    assert_eq!(
                        counters_after.stream_wide_syncs,
                        counters_before.stream_wide_syncs
                    );
                }
            }
            let before = this.ops.allocator_metrics().granted_bytes;
            for p in parameters {
                let math = this.mixed_rank_block.as_ref().unwrap().plan.arithmetic;
                let storage = match p.storage() {
                    HybridRankStorage::Bf16(words) => {
                        Storage::Bf16(this.ops.upload_bf16_words(words)?)
                    }
                    HybridRankStorage::Numeric(prepared) => {
                        this.mixed_rank_block
                            .as_mut()
                            .unwrap()
                            .projections
                            .push(Rc::new(Projection {
                                binding: p.binding().clone(),
                                shape: p.local_shape(),
                                storage: Storage::Numeric(Vec::new()),
                            }));
                        if math == HybridMixedProjectionArithmetic::NativeBf16GemmEx
                            && prepared.plan().mixed_gemmex_qkv_layout()?.is_some()
                        {
                            let packed = prepared.packed_mixed_gemmex_qkv()?;
                            let artifact = this.ops.upload_validated_numeric_fp8_linear(&packed)?;
                            let image = this.mixed_rank_block.as_mut().unwrap();
                            let projection =
                                Rc::get_mut(image.projections.last_mut().unwrap()).unwrap();
                            let Storage::Numeric(segments) = &mut projection.storage else {
                                unreachable!()
                            };
                            // Keep the uploaded root in image custody before post-upload
                            // source checks can fail. Existing preparation rollback/Unknown
                            // quarantine owns this allocation, exactly like separate segments.
                            segments.push(NumericSegment {
                                artifact,
                                workspace: None,
                                candidate_only: false,
                            });
                            for segment in prepared.segments() {
                                segment
                                    .artifact()
                                    .provenance()
                                    .source()
                                    .validate_source_identity()?;
                            }
                            continue;
                        }
                        for segment in prepared.segments() {
                            let host = segment.artifact();
                            host.provenance().source().validate_source_identity()?;
                            let source_layout = host.validated_payload().layout();
                            let artifact = if math
                                == HybridMixedProjectionArithmetic::NativeBf16GemmEx
                            {
                                let packed = HybridMixedProjectionArithmetic::packed_gemmex_layout(
                                    source_layout,
                                )?;
                                // Scale-grid starts move with the encoded rectangle, not
                                // with the original full matrix. Never adjust scale values.
                                this.ops.upload_numeric_fp8_linear(
                                    packed,
                                    host.weight_bytes(),
                                    host.scale_bytes(),
                                )?
                            } else {
                                this.ops
                                    .upload_validated_numeric_fp8_linear(host.validated_payload())?
                            };
                            let image = this.mixed_rank_block.as_mut().unwrap();
                            let projection =
                                Rc::get_mut(image.projections.last_mut().unwrap()).unwrap();
                            let Storage::Numeric(segments) = &mut projection.storage else {
                                unreachable!()
                            };
                            segments.push(NumericSegment {
                                artifact,
                                workspace: None,
                                candidate_only: false,
                            });
                            // Capability absence falls back before submission. Never retry
                            // GemmEx after a fused launch error or Unknown completion.
                            let candidate = image.plan.routed_down_m1_candidate_plan(
                                p.binding(),
                                source_layout,
                                1,
                                1,
                            );
                            let candidate = match candidate {
                                Some(plan) if ferrule_backend::cuda::providers::cutlass::numeric_fp8_kernel_available(
                                    plan.selected_kernel(),
                                )? => Some(plan),
                                _ => None,
                            };
                            let workspace = if let Some(plan) = candidate {
                                Some(RefCell::new(this.ops.numeric_fp8_linear_workspace(plan)?))
                            } else if math == HybridMixedProjectionArithmetic::W8A16Bf16 {
                                let p = NumericFp8LinearPlan::new(
                                    source_layout,
                                    1,
                                    0,
                                    OperandPrecision::W8A16Bf16F32Accumulate,
                                )?;
                                Some(RefCell::new(this.ops.numeric_fp8_linear_workspace(p)?))
                            } else {
                                None
                            };
                            segments.last_mut().unwrap().workspace = workspace;
                            segments.last_mut().unwrap().candidate_only = candidate.is_some();
                            host.provenance().source().validate_source_identity()?;
                        }
                        continue;
                    }
                    HybridRankStorage::NumericBinding(_) => {
                        return Err(cuda_error("retired Mixed host source cannot be uploaded"));
                    }
                    HybridRankStorage::Dense(_) => {
                        return Err(cuda_error("Mixed rejects F32 resident weights"));
                    }
                };
                this.mixed_rank_block
                    .as_mut()
                    .unwrap()
                    .projections
                    .push(Rc::new(Projection {
                        binding: p.binding().clone(),
                        shape: p.local_shape(),
                        storage,
                    }));
            }
            let image = this.mixed_rank_block.as_mut().unwrap();
            if this
                .ops
                .allocator_metrics()
                .granted_bytes
                .checked_sub(before)
                != Some(image.plan.weight_grant_bytes() as u64)
            {
                return Err(cuda_error(
                    "Mixed local weight/scale physical grants differ from plan",
                ));
            }
            if image.resident_bytes() != image.plan.weight_bytes() {
                return Err(cuda_error(
                    "Mixed local weight/scale payload differs from plan",
                ));
            }
            let pins = image
                .plan
                .decoded
                .as_ref()
                .map(|(_, descriptors)| {
                    descriptors
                        .iter()
                        .filter(|d| d.routed_layer.is_none())
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            for descriptor in pins {
                if this.mixed_decoded_weight(&descriptor)?.is_none() {
                    return Err(cuda_error(
                        "fixed decoded pins exceed physical backing/segment reserve",
                    ));
                }
            }
            Ok((0..parameters.len())
                .map(|slot| CudaMixedRankLinear {
                    generation: this.image_generation(),
                    slot,
                })
                .collect())
        }
    }

    fn mixed_projection(
        &self,
        token: &CudaMixedRankLinear,
        input: &Rows,
    ) -> Result<Rc<Projection>> {
        let image = self
            .mixed_rank_block
            .as_ref()
            .filter(|i| i.published)
            .ok_or_else(|| cuda_error("unpublished Mixed rank block"))?;
        let projection = image
            .projections
            .get(token.slot)
            .filter(|_| token.generation == self.image_generation())
            .ok_or_else(|| cuda_error("foreign Mixed rank projection generation/slot"))?;
        self.bf16_buffer(input)?;
        if input.shape().width() != projection.shape[1]
            || input.shape().rows() > image.plan.max_rows
        {
            return Err(cuda_error("Mixed local input extent/capacity mismatch"));
        }
        Ok(Rc::clone(projection))
    }

    pub(in crate::transformer::standard::cuda) fn quarantine_mixed_decoded(&mut self) {
        if let Some(cache) = self
            .mixed_rank_block
            .as_mut()
            .and_then(|i| i.decoded.as_mut())
        {
            cache.quarantine();
        }
    }
    pub(in crate::transformer::standard::cuda) fn complete_mixed_decoded(
        &mut self,
        succeeded: bool,
    ) -> Result<()> {
        if let Some(image) = self.mixed_rank_block.as_mut() {
            if let Some(cache) = image.decoded.as_mut() {
                cache.complete(succeeded)?;
            }
            // Called only at the original healthy outer fence. Unknown/unwind
            // bypasses completion and quarantines this entire image instead.
            image.pending_routes.clear();
        }
        Ok(())
    }
    /// Cache-only host ledger observation. No CUDA/allocator query, sync, payload
    /// read, map iteration, clock, allocation or logging. None means no current
    /// cache custody (disabled, not yet installed, or healthy rollback removed it).
    ///
    /// Before preparation returns, pins/leases may still be pending. After its
    /// healthy return the published snapshot is Idle. During synchronous work it
    /// is Executing; after the existing completion it is Idle again, with empty
    /// borrowed/pending lease sets. Idle is an observation, not a new GPU proof.
    /// Native fallback and retained failed-allocation reserves remain visible.
    pub fn mixed_decoded_cache_snapshot(&self) -> Option<MixedDecodedCacheSnapshot> {
        let image = self.mixed_rank_block.as_ref()?;
        let cache = image.decoded.as_ref()?;
        let active_profile = self
            .rank_performance
            .as_ref()
            .map(|p| p.timings.ledger.borrow().decoded_cache);
        Some(cache.snapshot(
            image.published,
            self.mixed_decoded_depth != 0,
            self.needs_quarantine(),
            self.rank_performance_enabled,
            active_profile,
        ))
    }
    pub(in crate::transformer::standard::cuda) fn record_mixed_decoded_profile(
        &mut self,
        counters: [u64; 4],
    ) {
        if let Some(cache) = self
            .mixed_rank_block
            .as_mut()
            .and_then(|i| i.decoded.as_mut())
        {
            cache.record_profiled_invocation(counters);
        }
    }

    pub(in crate::transformer::standard::cuda) fn mixed_decoded_grants(&self) -> (u64, u64) {
        self.mixed_rank_block
            .as_ref()
            .and_then(|i| i.decoded.as_ref())
            .map_or((0, 0), |c| {
                let s = c.physical_stats();
                (s.charged() as u64, s.peak as u64)
            })
    }
    fn mixed_decoded_descriptor(
        &self,
        slot: usize,
        segment: usize,
        pair: bool,
    ) -> Option<DecodedDescriptor> {
        let image = self.mixed_rank_block.as_ref()?;
        let (_, descriptors) = image.plan.decoded.as_ref()?;
        image
            .decoded_index
            .get(&(slot, segment, pair))
            .map(|&i| descriptors[i].clone())
    }
    pub(in crate::transformer::standard::cuda) fn mixed_decoded_backing_counts(&self) -> [u64; 6] {
        self.mixed_rank_block
            .as_ref()
            .and_then(|i| i.decoded.as_ref())
            .map_or([0; 6], |c| {
                let p = c.physical_stats();
                [
                    p.active as u64,
                    p.pooled as u64,
                    p.driver_growth as u64,
                    p.reuses,
                    p.fallbacks,
                    0, // Legacy segment-slack observation; arena tail is not slack.
                ]
            })
    }
    fn mixed_decoded_weight(
        &mut self,
        descriptor: &DecodedDescriptor,
    ) -> Result<Option<Rc<RefCell<CudaBf16Buffer>>>> {
        let image = self.mixed_rank_block.as_mut().unwrap();
        let cache = image.decoded.as_mut().unwrap();
        let before = cache.stats();
        let lookup = cache.acquire(descriptor)?;
        let evictions = cache.stats().evictions;
        let root = match lookup {
            Lookup::Hit(root) => {
                self.observe_decoded_cache(true, 0, before.evictions, evictions);
                return Ok(Some(root));
            }
            Lookup::Native => {
                self.observe_decoded_cache(false, 0, before.evictions, evictions);
                return Ok(None);
            }
            Lookup::Decode(root) => root,
        };
        let artifact = |&(slot, segment): &(usize, usize)| -> Result<&CudaNumericFp8Artifact> {
            let Storage::Numeric(segments) = &image.projections[slot].storage else {
                return Err(cuda_error("decoded source is not numeric"));
            };
            Ok(&segments[segment].artifact)
        };
        let result = match descriptor.operands.as_slice() {
            [single] => self.bf16_gemmex.as_mut().unwrap().decode_numeric_bf16_into(
                &self.ops,
                artifact(single)?,
                &mut root.borrow_mut(),
            ),
            [gate, up] => self
                .bf16_gemmex
                .as_mut()
                .unwrap()
                .decode_numeric_bf16_pair_into(
                    &self.ops,
                    [artifact(gate)?, artifact(up)?],
                    &mut root.borrow_mut(),
                ),
            _ => Err(cuda_error("decoded descriptor arity mismatch")),
        };
        #[cfg(test)]
        let result = decoded_gpu_tests::after_decode_submission(result);
        self.mixed_numeric_completion(false, result)?;
        self.mixed_rank_block
            .as_mut()
            .unwrap()
            .decoded
            .as_mut()
            .unwrap()
            .mark_ready(descriptor)?;
        self.observe_decoded_cache(
            false,
            descriptor.elements() * 2,
            before.evictions,
            evictions,
        );
        Ok(Some(root))
    }

    fn mixed_view(
        &self,
        n: usize,
        k: usize,
        rows: usize,
        resident: bool,
    ) -> Result<Bf16GemmExPlan> {
        self.mixed_rank_block
            .as_ref()
            .unwrap()
            .plan
            .views
            .iter()
            .copied()
            .find(|p| {
                p.layout().n == n
                    && p.layout().k == k
                    && p.rows() == rows
                    && p.is_resident() == resident
            })
            .ok_or_else(|| cuda_error("unprepared Mixed native geometry; no fallback"))
    }

    /// BF16 input -> BF16 local result. Row-parallel results are NOT reduced.
    /// The typed result must reach a BF16 collective, never the legacy F32 channel.
    pub fn mixed_rank_linear_local(
        &mut self,
        token: &CudaMixedRankLinear,
        input: &Rows,
    ) -> Result<Rows> {
        self.synchronous(|this| this.mixed_linear_inner(token, input))
    }

    fn mixed_linear_inner(&mut self, token: &CudaMixedRankLinear, input: &Rows) -> Result<Rows> {
        self.mixed_linear_inner_for_invocation(token, input, input.shape().rows(), None)
    }

    fn mixed_linear_inner_for_invocation(
        &mut self,
        token: &CudaMixedRankLinear,
        input: &Rows,
        invocation_rows: usize,
        mut staged_output: Option<CudaBf16Buffer>,
    ) -> Result<Rows> {
        let p = self.mixed_projection(token, input)?;
        if let Some(output) = &staged_output {
            let single = match &p.storage {
                Storage::Bf16(_) => true,
                Storage::Numeric(segments) => {
                    segments.len() == 1 && segments[0].artifact.layout().n == p.shape[0]
                }
            };
            if p.binding.role() != &TensorRole::RoutedExpertDown
                || !single
                || Some(output.len()) != input.shape().rows().checked_mul(p.shape[0])
            {
                return Err(cuda_error("invalid Mixed routed down output view"));
            }
            self.hold_bf16(output)?;
        }
        self.trace_bf16_parameter(
            "mixed.linear.input",
            self.bf16_buffer(input)?,
            input.shape(),
            Some(&p.binding),
            None,
            None,
        )?;
        let rows = input.shape().rows();
        let [n, k] = p.shape;
        self.hold_bf16(self.bf16_buffer(input)?)?;
        let mut parts = Vec::new();
        match &p.storage {
            Storage::Bf16(weight) => {
                let view = self.mixed_view(n, k, rows, true)?;
                let mut output = match staged_output.take() {
                    Some(output) => output,
                    None => self.ops.zero_bf16_buffer(rows * n)?,
                };
                self.hold_bf16(&output)?;
                let x = self.bf16_buffer(input)?;
                self.bf16_gemmex.as_mut().unwrap().execute_bf16_view(
                    &self.ops,
                    view,
                    weight,
                    x,
                    &mut output,
                    Bf16GemmLayout::contiguous(rows, n, k),
                )?;
                parts.push(output);
            }
            Storage::Numeric(segments) => {
                for (segment_index, segment) in segments.iter().enumerate() {
                    let layout = segment.artifact.layout();
                    let candidate = self
                        .mixed_rank_block
                        .as_ref()
                        .unwrap()
                        .plan
                        .routed_down_m1_candidate_plan(&p.binding, layout, invocation_rows, rows);
                    let workspace = segment
                        .workspace
                        .as_ref()
                        .filter(|_| !segment.candidate_only || candidate.is_some());
                    let output = if let Some(workspace) = workspace {
                        let plan = NumericFp8LinearPlan::new(
                            layout,
                            rows,
                            0,
                            OperandPrecision::W8A16Bf16F32Accumulate,
                        )?;
                        let mut acc = self.ops.zero_f32_buffer(rows * layout.n)?;
                        self.expert_hold
                            .push(acc.as_device_buffer().slice(0, acc.len())?);
                        let mut workspace = workspace.borrow_mut();
                        let result = self.ops.numeric_fp8_bf16_linear_into(
                            &segment.artifact,
                            self.bf16_buffer(input)?,
                            &mut acc,
                            &mut workspace,
                            plan,
                            k,
                            layout.n,
                        );
                        #[cfg(test)]
                        let result = if segment.candidate_only {
                            candidate_tests::after_candidate_submission(result)
                        } else {
                            result
                        };
                        self.mixed_numeric_completion(workspace.is_poisoned(), result)?;
                        let layout = F32ToBf16RowsLayout::packed(1, rows, layout.n)?;
                        if let Some(mut output) = staged_output.take() {
                            self.ops
                                .f32_rows_to_bf16_rne_into(&acc, &mut output, layout)?;
                            output
                        } else {
                            self.ops.f32_rows_to_bf16_rne(&acc, layout)?
                        }
                    } else {
                        let view = self.mixed_view(layout.n, k, rows, false)?;
                        let mut output = match staged_output.take() {
                            Some(output) => output,
                            None => self.ops.zero_bf16_buffer(rows * layout.n)?,
                        };
                        self.hold_bf16(&output)?;
                        let decoded = self
                            .mixed_decoded_descriptor(token.slot, segment_index, false)
                            .map(|d| self.mixed_decoded_weight(&d))
                            .transpose()?
                            .flatten();
                        let x = self.bf16_buffer(input)?;
                        if let Some(weight) = decoded {
                            self.bf16_gemmex.as_mut().unwrap().execute_bf16_view(
                                &self.ops,
                                view,
                                &weight.borrow(),
                                x,
                                &mut output,
                                Bf16GemmLayout::contiguous(rows, layout.n, k),
                            )?;
                        } else {
                            self.bf16_gemmex.as_mut().unwrap().execute_view(
                                &self.ops,
                                view,
                                &segment.artifact,
                                x,
                                &mut output,
                            )?;
                        }
                        output
                    };
                    self.hold_bf16(&output)?;
                    parts.push(output);
                }
            }
        }
        let output = if parts.len() == 1 {
            parts.pop().unwrap()
        } else {
            // Exact device expansion ONLY for segmented Q/K/V packing, not math
            // or a resident F32 mirror. Each segment was already rounded to BF16.
            let mut packed = self.ops.zero_f32_buffer(rows * n)?;
            self.hold_expert_buffer(&packed)?;
            let mut offset = 0;
            for part in &parts {
                let width = part.len() / rows;
                let expanded = self.ops.bf16_to_f32(part)?;
                self.hold_expert_buffer(&expanded)?;
                for row in 0..rows {
                    self.ops.copy_f32_range(
                        &expanded,
                        row * width,
                        &mut packed,
                        row * n + offset,
                        width,
                    )?;
                }
                offset += width;
            }
            self.ops
                .f32_rows_to_bf16_rne(&packed, F32ToBf16RowsLayout::packed(1, rows, n)?)?
        };
        self.hold_bf16(&output)?;
        self.trace_bf16_parameter(
            "mixed.linear.output",
            &output,
            RowsShape::new(rows, n)?,
            Some(&p.binding),
            None,
            None,
        )?;
        rows_bf16(RowsShape::new(rows, n)?, input.arena(), output)
    }

    /// Local gate/up column shards and down K-shard. No down all-reduce here:
    /// shared gating and routed weighting must happen before the late reduction.
    pub fn mixed_rank_swiglu_local(
        &mut self,
        projections: [&CudaMixedRankLinear; 3],
        input: &Rows,
    ) -> Result<Rows> {
        self.synchronous(|this| {
            this.mixed_swiglu_inner(projections, input, input.shape().rows(), None)
        })
    }
    fn mixed_swiglu_inner(
        &mut self,
        tokens: [&CudaMixedRankLinear; 3],
        input: &Rows,
        invocation_rows: usize,
        staged_output: Option<CudaBf16Buffer>,
    ) -> Result<Rows> {
        let gate = self.mixed_projection(tokens[0], input)?;
        let up = self.mixed_projection(tokens[1], input)?;
        let image = self.mixed_rank_block.as_ref().unwrap();
        let down = image
            .projections
            .get(tokens[2].slot)
            .filter(|_| tokens[2].generation == self.image_generation())
            .ok_or_else(|| cuda_error("foreign down projection"))?;
        let roles = (gate.binding.role(), up.binding.role(), down.binding.role());
        if !matches!(
            roles,
            (
                TensorRole::RoutedExpertGate,
                TensorRole::RoutedExpertUp,
                TensorRole::RoutedExpertDown
            ) | (
                TensorRole::SharedExpertGate,
                TensorRole::SharedExpertUp,
                TensorRole::SharedExpertDown
            ) | (
                TensorRole::DenseMlpGate,
                TensorRole::DenseMlpUp,
                TensorRole::DenseMlpDown
            )
        ) || gate.binding.residency() != up.binding.residency()
            || gate.binding.residency() != down.binding.residency()
            || gate.shape != up.shape
            || down.shape != [gate.shape[1], gate.shape[0]]
        {
            return Err(cuda_error("Mixed SwiGLU role/shard/residency mismatch"));
        }
        let rows = input.shape().rows();
        let [n, k] = gate.shape;
        // HF Qwen shared/dense MLPs own independent gate_proj and up_proj
        // operators; routed experts own one concatenated gate_up_proj. Keep
        // those source semantics distinct because BF16 output rounding can
        // differ at M>1 even when the decoded operands are identical.
        let paired = if image.plan.arithmetic == HybridMixedProjectionArithmetic::NativeBf16GemmEx
            && matches!(gate.binding.role(), TensorRole::RoutedExpertGate)
        {
            match (&gate.storage, &up.storage) {
                (Storage::Numeric(g), Storage::Numeric(u)) if g.len() == 1 && u.len() == 1 => {
                    Some((&g[0].artifact, &u[0].artifact))
                }
                _ => None,
            }
        } else {
            None
        };
        let fused = fuse_pair_swiglu(paired.is_some(), self.diagnostic.is_enabled());
        let product = if let Some((g, u)) = paired {
            let view = self.mixed_view(2 * n, k, rows, false)?;
            self.hold_bf16(self.bf16_buffer(input)?)?;
            let decoded = self
                .mixed_decoded_descriptor(tokens[0].slot, 0, true)
                .map(|d| self.mixed_decoded_weight(&d))
                .transpose()?
                .flatten();
            let x = self.bf16_buffer(input)?;
            self.trace_bf16_parameter(
                "mixed.pair.input",
                x,
                input.shape(),
                Some(&gate.binding),
                None,
                None,
            )?;
            if fused {
                let mut product = self.ops.zero_bf16_buffer(rows * n)?;
                self.hold_bf16(&product)?;
                // Same full-2N GEMM and two BF16 roundings; only the tail is
                // fused. A cache miss selects original sources BEFORE submit.
                let result = if let Some(weight) = decoded {
                    self.bf16_gemmex
                        .as_mut()
                        .unwrap()
                        .execute_bf16_pair_swiglu_view_into(
                            &self.ops,
                            view,
                            &weight.borrow(),
                            x,
                            &mut product,
                            Bf16GemmLayout::contiguous(rows, 2 * n, k),
                        )
                } else {
                    self.bf16_gemmex
                        .as_mut()
                        .unwrap()
                        .execute_pair_swiglu_view_into(&self.ops, view, [g, u], x, &mut product)
                };
                #[cfg(test)]
                let result = decoded_gpu_tests::routed_fusion::after_fused_submission(result);
                self.mixed_numeric_completion(false, result)?;
                product
            } else {
                // The backend does not export its packed tail for observation.
                // Keep real gate/up tensors for diagnostics, never synthesize
                // them on the CPU or rerun a GEMM after fused submission.
                for binding in [&gate.binding, &up.binding] {
                    self.trace_bf16_parameter(
                        "mixed.linear.input",
                        x,
                        input.shape(),
                        Some(binding),
                        None,
                        None,
                    )?;
                }
                let mut go = self.ops.zero_bf16_buffer(rows * n)?;
                let mut uo = self.ops.zero_bf16_buffer(rows * n)?;
                self.hold_bf16(&go)?;
                self.hold_bf16(&uo)?;
                if let Some(weight) = decoded {
                    self.bf16_gemmex.as_mut().unwrap().execute_bf16_pair_view(
                        &self.ops,
                        view,
                        &weight.borrow(),
                        x,
                        [&mut go, &mut uo],
                        Bf16GemmLayout::contiguous(rows, 2 * n, k),
                    )?;
                } else {
                    self.bf16_gemmex.as_mut().unwrap().execute_pair(
                        &self.ops,
                        view,
                        [g, u],
                        x,
                        [&mut go, &mut uo],
                    )?;
                }
                for (name, buffer, binding) in [
                    ("mixed.pair.gate", &go, &gate.binding),
                    ("mixed.pair.up", &uo, &up.binding),
                ] {
                    self.trace_bf16_parameter(
                        name,
                        buffer,
                        RowsShape::new(rows, n)?,
                        Some(binding),
                        None,
                        None,
                    )?;
                }
                self.ops.swiglu_bf16(&go, &uo)?
            }
        } else {
            let g = self.mixed_linear_inner(tokens[0], input)?;
            let u = self.mixed_linear_inner(tokens[1], input)?;
            self.ops
                .swiglu_bf16(self.bf16_buffer(&g)?, self.bf16_buffer(&u)?)?
        };
        self.hold_bf16(&product)?;
        let product = rows_bf16(RowsShape::new(rows, n)?, input.arena(), product)?;
        let output = self.mixed_linear_inner_for_invocation(
            tokens[2],
            &product,
            invocation_rows,
            staged_output,
        )?;
        #[cfg(test)]
        self.mixed_numeric_completion(
            false,
            decoded_gpu_tests::routed_fusion::after_down_submission(),
        )?;
        Ok(output)
    }

    /// Existing HF-eager expert weighting: BF16 per multiply/add, ascending expert
    /// index. This is explicit, NOT asserted equivalent to vLLM's fused MoE kernel.
    /// All experts are device resident, including unselected experts.
    /// Direct down-output packing preserves finite/zero/infinity words. Unlike
    /// the old BF16->F32->BF16 pack it does not quiet sNaNs; NaN payload equality
    /// is not promised. The original arithmetic combine still propagates NaNs.
    pub fn mixed_rank_routed_local(
        &mut self,
        layer: usize,
        input: &Rows,
        routes: &RouterRoutes,
        accumulation: HybridRoutedAccumulation,
    ) -> Result<Rows> {
        let HybridRoutedAccumulation::HfEagerBf16ExpertOrder = accumulation;
        self.synchronous(|this| {
            let image = this
                .mixed_rank_block
                .as_ref()
                .filter(|i| i.published)
                .ok_or_else(|| cuda_error("missing Mixed local expert image"))?;
            let count = *image
                .plan
                .expert_counts
                .get(&layer)
                .ok_or_else(|| cuda_error("no complete resident layer"))?;
            if routes.rows() != input.shape().rows()
                || input.shape().rows() > image.plan.max_rows
                || routes.weights().iter().any(|w| !w.is_finite() || *w < 0.)
            {
                return Err(cuda_error("Mixed routes extent/weight mismatch"));
            }
            this.bf16_buffer(input)?;
            let width = input.shape().width();
            let batch = RoutedBatchPlan::new(
                routes.rows(),
                width,
                count,
                routes.expert_ids(),
                routes.weights(),
            )?;
            let packed = this.ops.zero_bf16_buffer(batch.layout.elements())?;
            this.hold_bf16(&packed)?;
            this.hold_bf16(this.bf16_buffer(input)?)?;
            let batch_gather =
                if routed_output::batch_metadata_supported(routes.rows(), batch.row_ids.len()) {
                    let metadata = this.ops.allocate_bf16_route_metadata(
                        &batch.row_ids,
                        &batch.weights,
                        routes.rows(),
                    )?;
                    let image = this.mixed_rank_block.as_mut().unwrap();
                    let index = image.pending_routes.len();
                    image.pending_routes.push(metadata);
                    let ops = Rc::clone(&this.ops);
                    let result = ops.prepare_bf16_route_metadata(
                        this.mixed_rank_block
                            .as_mut()
                            .unwrap()
                            .pending_routes
                            .get_mut(index)
                            .ok_or_else(|| cuda_error("missing Mixed route metadata custody"))?,
                    );
                    this.mixed_numeric_completion(false, result)?;
                    this.observe_route_upload(batch.row_ids.len() * 8);
                    let mut gathered = ops.zero_bf16_buffer(batch.layout.elements())?;
                    this.hold_bf16(&gathered)?;
                    let result = ops.gather_bf16_routes_into(
                        this.bf16_buffer(input)?,
                        this.mixed_rank_block
                            .as_ref()
                            .unwrap()
                            .pending_routes
                            .get(index)
                            .ok_or_else(|| cuda_error("missing Mixed route metadata custody"))?,
                        &mut gathered,
                        width,
                    );
                    #[cfg(test)]
                    let result = result.and_then(|()| {
                        decoded_gpu_tests::route_batch::after_gather_submission(&gathered)
                    });
                    this.mixed_numeric_completion(false, result)?;
                    Some((index, gathered))
                } else {
                    // The opaque API currently bounds M<=8/top-k<=8. Preserve the
                    // pre-existing M<=32 and larger-top-k contract in the same loop.
                    None
                };
            for group in &batch.groups {
                let image = this.mixed_rank_block.as_ref().unwrap();
                let slots = image
                    .routed_slots
                    .get(this.image_generation(), layer, group.expert)?;
                let tokens = slots.map(|slot| CudaMixedRankLinear {
                    generation: image.routed_slots.generation(),
                    slot,
                });
                let range = batch.layout.bucket(group.begin, group.count)?;
                let gathered = if let Some((_, gathered)) = &batch_gather {
                    gathered.slice(range.start, range.len())?
                } else {
                    let ids = batch.row_ids[group.begin..group.begin + group.count]
                        .iter()
                        .map(|&row| row as u32)
                        .collect::<Vec<_>>();
                    let gathered =
                        this.ops
                            .embedding_bf16(this.bf16_buffer(input)?, &ids, width)?;
                    this.observe_route_upload(ids.len() * std::mem::size_of::<i32>());
                    this.hold_bf16(&gathered)?;
                    gathered
                };
                let gathered = rows_bf16(RowsShape::new(group.count, width)?, None, gathered)?;
                let output = packed.slice(range.start, range.len())?;
                this.mixed_swiglu_inner(
                    [&tokens[0], &tokens[1], &tokens[2]],
                    &gathered,
                    input.shape().rows(),
                    Some(output),
                )?;
            }
            let mut out = this.ops.zero_bf16_buffer(input.shape().elements())?;
            this.hold_bf16(&out)?;
            if let Some((index, _)) = &batch_gather {
                let result = this.ops.weighted_combine_bf16_routes_into(
                    &packed,
                    this.mixed_rank_block
                        .as_ref()
                        .unwrap()
                        .pending_routes
                        .get(*index)
                        .ok_or_else(|| cuda_error("missing Mixed route metadata custody"))?,
                    &mut out,
                    width,
                );
                this.mixed_numeric_completion(false, result)?;
            } else {
                let ids = this.ops.upload_i32_buffer(&batch.row_ids)?;
                this.observe_route_upload(batch.row_ids.len() * std::mem::size_of::<i32>());
                this.route_ids
                    .push(ids.as_device_buffer().slice(0, ids.len())?);
                let weights = this.ops.upload_f32_buffer(&batch.weights)?;
                this.observe_route_upload(batch.weights.len() * std::mem::size_of::<f32>());
                this.hold_route_buffer(&weights)?;
                this.ops.weighted_combine_bf16_into(
                    &packed,
                    &ids,
                    &weights,
                    &mut out,
                    input.shape().rows(),
                    width,
                )?;
            }
            rows_bf16(input.shape(), input.arena(), out)
        })
    }

    /// Replicated scalar gate acts on the local shared down partial BEFORE
    /// shared+routed addition and the late TP reduction.
    pub fn mixed_rank_shared_gate_local(&mut self, output: Rows, gate: &Rows) -> Result<Rows> {
        super::super::super::ready(self.bf16_gate(output, gate, true)?)
    }

    /// Local late-reduce seam only. Shared output must already include its gate.
    /// The returned BF16 tensor still needs ONE BF16 TP collective.
    pub fn mixed_rank_moe_local_add(
        &mut self,
        shared: &Rows,
        routed: &Rows,
        order: HybridMoeCombineOrder,
    ) -> Result<Rows> {
        if order != HybridMoeCombineOrder::LocalBf16AddThenReduce
            || shared.shape() != routed.shape()
        {
            return Err(cuda_error(
                "local MoE add cannot accept separately reduced contract",
            ));
        }
        self.synchronous(|this| {
            this.hold_bf16(this.bf16_buffer(shared)?)?;
            this.hold_bf16(this.bf16_buffer(routed)?)?;
            let output = this
                .ops
                .residual_bf16(this.bf16_buffer(shared)?, this.bf16_buffer(routed)?)?;
            this.hold_bf16(&output)?;
            rows_bf16(shared.shape(), shared.arena(), output)
        })
    }
}

#[cfg(test)]
#[path = "../../../../../tests/unit/transformer/mixed_rank_preparation.rs"]
mod preparation_tests;

#[cfg(test)]
#[path = "../../../../../tests/mixed_routed_down/dispatch.rs"]
mod candidate_tests;

#[cfg(test)]
#[path = "../../../../../tests/unit/transformer/mixed_decoded_gpu.rs"]
mod decoded_gpu_tests;
