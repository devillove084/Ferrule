//! Typed hidden-to-hidden rank program. No change to the legacy V2 F32 schedule.
use super::*;
use crate::decoder::PackedDecoderBatch;
use crate::transformer::{RowsDType, RowsShape};
use ferrule_common::execution::ExecutionTransactionId;

/// Separate from the legacy F32 trait. Runtime adapters must retain BF16 words
/// and honor this exact schedule; a returned host result is not a GPU fence.
/// A transport capability decision, not an allocation or completion receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataLocation {
    Host,
    Device,
}

/// Caller metadata is untrusted until checked against the transport's own sealed
/// plan and active, validated execution. Member order is authoritative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DevicePayloadIdentity {
    pub epoch: ParallelTopologyId,
    pub transaction: ExecutionTransactionId,
    pub owner: ParallelRankId,
    pub members: [ParallelRankId; 2],
    pub site: HybridCollectiveSite,
    pub rows: usize,
    pub sequences: usize,
    pub elements: usize,
}

/// Single-use permission, with no data ownership or consumer-fence evidence.
/// Dropping a permit cannot undo enqueue or authorize resource retirement.
#[derive(Debug)]
pub struct DevicePayloadPermit {
    identity: DevicePayloadIdentity,
}
impl DevicePayloadPermit {
    pub fn identity(&self) -> &DevicePayloadIdentity {
        &self.identity
    }
}

pub trait HybridBf16Collective {
    /// Host remains the default until provider quality and budgets are admitted.
    fn data_location(&self, _site: &HybridCollectiveSite) -> DataLocation {
        DataLocation::Host
    }
    /// Metadata-only rendezvous. Authenticate the complete canonical identity
    /// and every member's vote before returning permission to any member.
    fn participate_device(&mut self, _identity: &DevicePayloadIdentity) -> Result<()> {
        Err(invalid("Mixed device participation is not admitted"))
    }
    fn permit_device_payload(
        &mut self,
        identity: DevicePayloadIdentity,
    ) -> Result<DevicePayloadPermit> {
        // Publish canonical rejection to waiting members even when caller
        // metadata lies about the site's advertised capability.
        self.participate_device(&identity)?;
        if self.data_location(&identity.site) != DataLocation::Device {
            return Err(invalid("Mixed device participation is disabled"));
        }
        Ok(DevicePayloadPermit { identity })
    }
    /// Recheck cancellation and terminal peer loss immediately before enqueue.
    fn check_device_payload_permit(&mut self, _permit: &DevicePayloadPermit) -> Result<()> {
        Err(invalid("Mixed device permit checking is not admitted"))
    }
    /// Only successful enqueue acknowledgement advances the site cursor.
    /// Any error here is post-enqueue and must retain unknown-completion custody.
    fn device_payload_enqueued(&mut self, _permit: DevicePayloadPermit) -> Result<()> {
        Err(crate::decoder::CompletionUnknown::wrap(invalid(
            "Mixed device enqueue acknowledgement is not admitted",
        )))
    }
    /// Report the provider's actual invocation-wide consumer fence AND finite /
    /// asynchronous-error outcome before owner completion or publication. No
    /// per-site finite pass or completion evidence is manufactured by default.
    fn complete_device_payloads(
        &mut self,
        _transaction: ExecutionTransactionId,
        completion: Result<()>,
    ) -> Result<()> {
        completion.map_err(crate::decoder::CompletionUnknown::wrap)?;
        Err(crate::decoder::CompletionUnknown::wrap(invalid(
            "Mixed device completion is not admitted",
        )))
    }
    fn owner(&self) -> ParallelRankId;
    fn members(&self) -> &[ParallelRankId];
    /// Return the immutable metadata authenticated by the prepared transport.
    /// Implementations must report their sealed plan, never echo caller metadata.
    /// Legacy adapters fail closed before parameter upload or physical KV enter.
    fn sealed_binding(&self) -> Result<HybridBf16CollectiveBinding> {
        Err(invalid(
            "Mixed collective does not expose an authenticated schedule binding",
        ))
    }
    /// Metadata from the transport's own authenticated plan. Empty is valid only
    /// for dense programs; it does not authorize routed execution.
    fn sealed_route_sites(&self) -> Result<Vec<HybridMixedRouteSite>> {
        Ok(Vec::new())
    }
    /// Compare exact IDs/weight bits for every owner after sealed execution
    /// validation and before local expert execution. Never encode these as BF16.
    fn agree_routes(
        &mut self,
        _transaction: ExecutionTransactionId,
        _agreement: &HybridMixedRouteAgreement,
    ) -> Result<()> {
        Err(invalid("Mixed transport lacks typed route agreement"))
    }
    /// Recheck exact sealed control identity. Never serialize IDs into floats.
    /// The adapter must reject an unsealed, unvalidated or cancelled owner.
    fn validate_execution(
        &mut self,
        _transaction: ExecutionTransactionId,
        _batch: &PackedDecoderBatch,
        _sessions: &[u64],
    ) -> Result<()> {
        Err(invalid(
            "Mixed block requires a sealed control-validation adapter",
        ))
    }
    /// Control completion after this owner has fenced every data-site consumer.
    /// This is not a collective tensor, KV commit, or global publication receipt.
    fn complete_owner(&mut self, _transaction: ExecutionTransactionId) -> Result<()> {
        Err(invalid("Mixed block requires a control-completion adapter"))
    }
    fn exchange_bf16(
        &mut self,
        transaction: ExecutionTransactionId,
        site: u64,
        kind: HybridCollectiveKind,
        values: Vec<u16>,
    ) -> Result<Vec<u16>>;
    fn abort(&mut self);
}

/// Model-neutral metadata reported by a prepared BF16 transport. Equality includes
/// global layers, semantics, ordinals, order, geometry, precision and cohort bounds.
/// This description is not a completion receipt or authority to enter KV.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridBf16CollectiveBinding {
    pub epoch: ParallelTopologyId,
    pub source: ParallelRankId,
    pub members: Vec<ParallelRankId>,
    pub version: u64,
    pub precision: HybridCollectivePrecision,
    pub sites: Vec<HybridCollectiveSite>,
    pub max_rows: usize,
    pub max_sequences: usize,
}

/// Bound GQA/GDN layers with dense or local-tensor routed FFNs, hidden input/output.
/// Routed preparation requires authenticated typed route control from the transport.
/// Selecting this contract does not enable a full-model Mixed profile.
#[derive(Debug, Clone)]
pub struct HybridMixedRankProgramPlan {
    pub(crate) layout: Arc<HybridRankLayout>,
    pub(crate) segment: crate::transformer::LayerSegmentPlan,
    pub(crate) max_rows: usize,
    pub(crate) max_sequences: usize,
    pub(crate) max_positions: usize,
    pub(crate) page_size: usize,
    pub(crate) state_copies: usize,
    pub(crate) width: usize,
    pub(crate) decoded_cache: Option<super::mixed_decoded::DecodedCacheAdmission>,
}
/// Compatibility name; execution uses the generic bound program.
pub type HybridMixedGqaBlockPlan = HybridMixedRankProgramPlan;

impl HybridMixedRankProgramPlan {
    /// Internal candidate. Caller must carry the enlarged operation grant into
    /// the existing device/claim ledger; public factories remain default off.
    #[doc(hidden)]
    pub fn with_decoded_weight_cache(
        mut self,
        resources: &crate::transformer::BoundDecoderResources,
        physical_pages: usize,
        cache_grants: usize,
        device_capacity: usize,
        owner_margins: usize,
    ) -> Result<Self> {
        if self.decoded_cache.is_some() {
            return Err(invalid("decoded cache already configured"));
        }
        let baseline = self.operation_estimate(resources, physical_pages)?;
        let bindings = crate::transformer::segment_inventory::HybridSegmentInventory::new(
            resources,
            &self.segment,
        )?
        .parameters()
        .cloned()
        .collect::<Vec<_>>();
        let descriptors = super::mixed_decoded::decoded_descriptors(&self.layout, &bindings)?;
        let budget = HybridMixedDecodedCacheBudget::new(cache_grants, &descriptors)?;
        let arena = Arc::new(super::mixed_decoded_arena::DecodedArenaPlan::new(
            budget,
            &descriptors,
        )?);
        budget.require_capacity(
            baseline.total_device_grants()?,
            owner_margins,
            device_capacity,
        )?;
        self.decoded_cache = Some(super::mixed_decoded::DecodedCacheAdmission {
            budget,
            arena,
            physical_pages,
            device_capacity,
            owner_margins,
        });
        Ok(self)
    }
    #[doc(hidden)]
    pub fn decoded_weight_cache_budget(&self) -> Option<HybridMixedDecodedCacheBudget> {
        self.decoded_cache.as_ref().map(|a| a.budget)
    }
    pub(crate) fn decoded_cache_grants(&self) -> usize {
        self.decoded_cache
            .as_ref()
            .map_or(0, |a| a.budget.capacity_grants)
    }

    pub fn new(
        layout: Arc<HybridRankLayout>,
        max_rows: usize,
        max_sequences: usize,
        max_positions: usize,
        page_size: usize,
        state_copies: usize,
    ) -> Result<Self> {
        let range = layout.state_schema().layer_range();
        if range.clone().any(|i| {
            !matches!(
                layout.spec.layers()[i].feed_forward(),
                FeedForward::SwiGlu(_)
            )
        }) || range.len() != 1
            || !matches!(
                layout.attention(range.start)?,
                HybridLocalAttention::Gqa { .. }
            )
        {
            return Err(invalid(
                "single GQA compatibility constructor requires one GQA layer",
            ));
        }
        Self::from_layout(
            layout,
            max_rows,
            max_sequences,
            max_positions,
            page_size,
            state_copies,
        )
    }
    /// Compatibility constructor for the frozen two-layer gate, not a production factory.
    pub fn new_gqa_gdn_pair(
        layout: Arc<HybridRankLayout>,
        max_rows: usize,
        max_sequences: usize,
        max_positions: usize,
        page_size: usize,
        state_copies: usize,
    ) -> Result<Self> {
        let range = layout.state_schema().layer_range();
        if range.clone().any(|i| {
            !matches!(
                layout.spec.layers()[i].feed_forward(),
                FeedForward::SwiGlu(_)
            )
        }) || range.len() != 2
            || !matches!(
                layout.attention(range.start)?,
                HybridLocalAttention::Gqa { .. }
            )
            || !matches!(
                layout.attention(range.start + 1)?,
                HybridLocalAttention::GatedDelta { .. }
            )
        {
            return Err(invalid(
                "pair compatibility constructor requires GQA then GDN",
            ));
        }
        Self::from_layout(
            layout,
            max_rows,
            max_sequences,
            max_positions,
            page_size,
            state_copies,
        )
    }
    /// Lower every bound layer using orthogonal attention and FFN semantics.
    /// There is no special layer number, count, model family, or pair dispatch.
    pub fn from_layout(
        layout: Arc<HybridRankLayout>,
        max_rows: usize,
        max_sequences: usize,
        max_positions: usize,
        page_size: usize,
        state_copies: usize,
    ) -> Result<Self> {
        let range = layout.state_schema().layer_range();
        if !layout.precision().is_mixed()
            || layout.members().len() != 2
            || range.is_empty()
            || !(1..=32).contains(&max_rows)
            || max_sequences == 0
            || max_sequences > max_rows
            || !(1..=1024).contains(&max_positions)
            || page_size == 0
            || state_copies == 0
            || layout
                .spec
                .max_sequence_length()
                .is_some_and(|p| max_positions > p)
        {
            return Err(invalid(
                "Mixed program requires TP2, M1..32 and bounded state",
            ));
        }
        let mut gqa_geometry = None;
        for index in range {
            let d = &layout.spec.layers()[index];
            if !matches!(d.attention_residual(), crate::transformer::Residual::Add)
                || !matches!(d.feed_forward_residual(), crate::transformer::Residual::Add)
            {
                return Err(invalid("Mixed program requires additive residuals"));
            }
            let projections = match d.attention() {
                Attention::Gqa(a) => {
                    let HybridLocalAttention::Gqa {
                        query_heads,
                        kv_heads,
                        head_dim,
                        ..
                    } = layout.attention(index)?
                    else {
                        unreachable!()
                    };
                    let geometry = (query_heads.len(), kv_heads.len(), *head_dim);
                    if gqa_geometry.is_some_and(|previous| previous != geometry) {
                        return Err(invalid(
                            "Mixed program requires one prepared GQA provider geometry",
                        ));
                    }
                    gqa_geometry = Some(geometry);
                    vec![a.query(), a.key(), a.value(), a.output()]
                }
                Attention::GatedDeltaNet(a) => {
                    if a.norm().one_plus_weight() {
                        return Err(invalid("Mixed GDN rejects offset output norm"));
                    }
                    vec![a.qkv(), a.z(), a.a(), a.beta(), a.output()]
                }
                _ => return Err(invalid("Mixed program supports GQA/GDN attention only")),
            };
            if projections.iter().any(|p| p.has_bias()) {
                return Err(invalid(
                    "Mixed program rejects biased attention projections",
                ));
            }
            match d.feed_forward() {
                FeedForward::SwiGlu(m) => {
                    if [m.gate(), m.up(), m.down()].iter().any(|p| p.has_bias())
                        || m.activation_limit().is_some()
                    {
                        return Err(invalid("Mixed program rejects biased/clamped FFN"));
                    }
                }
                FeedForward::Moe(m) => {
                    if layout.routed_execution() != &HybridRoutedExpertExecution::LocalTensorShards
                        || crate::transformer::operators::standard_router_unsupported(
                            m.router_spec(),
                        )
                        .is_some()
                        || m.router_spec().num_experts() > 1024
                        || m.router_spec().experts_per_token() > 32
                        || m.router().has_bias()
                        || m.shared_expert_gate().is_some_and(|p| p.has_bias())
                        || std::iter::once(m.expert())
                            .chain(m.shared_expert())
                            .any(|ff| {
                                ff.activation_limit().is_some()
                                    || [ff.gate(), ff.up(), ff.down()].iter().any(|p| p.has_bias())
                            })
                    {
                        return Err(invalid(
                            "Mixed routed program requires local TP expert shards and normalized softmax top-k",
                        ));
                    }
                }
            }
        }
        let width = layout.spec.hidden_size();
        let segment = crate::transformer::LayerSegmentPlan::new(
            layout.spec.layers().len(),
            layout.state_schema().layer_range(),
            false,
            false,
        )
        .map_err(|_| invalid("invalid Mixed segment endpoint/model scope"))?;
        let result = Self {
            layout,
            segment,
            max_rows,
            max_sequences,
            max_positions,
            page_size,
            state_copies,
            width,
            decoded_cache: None,
        };
        result.layout.state_budget(state_copies)?;
        result.recurrent_workspace_bytes()?;
        for site in result.sites() {
            site.shape.elements(max_sequences, max_rows)?;
        }
        Ok(result)
    }
    /// Endpoint ownership is the ordinary segment contract, never inferred from a model name.
    #[allow(clippy::too_many_arguments)]
    pub fn for_segment(
        layout: Arc<HybridRankLayout>,
        segment: crate::transformer::LayerSegmentPlan,
        max_rows: usize,
        max_sequences: usize,
        max_positions: usize,
        page_size: usize,
        state_copies: usize,
    ) -> Result<Self> {
        if segment.total_layers() != layout.spec.layers().len()
            || segment.layers() != layout.state_schema().layer_range()
            || (segment.owns_output() && layout.spec.output().has_bias())
        {
            return Err(invalid("Mixed segment endpoint/model scope mismatch"));
        }
        let mut plan = Self::from_layout(
            layout,
            max_rows,
            max_sequences,
            max_positions,
            page_size,
            state_copies,
        )?;
        plan.segment = segment;
        Ok(plan)
    }
    pub fn segment(&self) -> &crate::transformer::LayerSegmentPlan {
        &self.segment
    }
    pub fn layers(&self) -> std::ops::Range<usize> {
        self.layout.state_schema().layer_range()
    }
    pub fn layout(&self) -> &HybridRankLayout {
        &self.layout
    }
    /// This bounded local-expert block uses the first ordered TP root as source.
    /// Transport metadata must agree; source identity is never a numeric packet.
    pub fn collective_binding(&self) -> HybridBf16CollectiveBinding {
        HybridBf16CollectiveBinding {
            epoch: self.layout.scope().topology,
            source: self.layout.members()[0],
            members: self.layout.members().to_vec(),
            version: HYBRID_AGREEMENT_VERSION,
            precision: self.layout.precision().collective(),
            sites: self.sites(),
            max_rows: self.max_rows,
            max_sequences: self.max_sequences,
        }
    }
    /// Pure preflight: call before materialization or journal physical KV enter.
    pub fn validate_collective(&self, transport: &dyn HybridBf16Collective) -> Result<()> {
        if transport.owner() != self.layout.owner()
            || transport.members() != self.layout.members()
            || transport.sealed_binding()? != self.collective_binding()
            || transport.sealed_route_sites()? != self.route_sites()
        {
            return Err(invalid(
                "Mixed block differs from sealed transport owner/epoch/source/schedule",
            ));
        }
        Ok(())
    }
    pub fn layer(&self) -> usize {
        self.layers().start
    }
    pub fn page_size(&self) -> usize {
        self.page_size
    }
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }
    pub fn state_copies(&self) -> usize {
        self.state_copies
    }
    /// Exact native BF16 attention shapes; no eager/F32 fallback on a view miss.
    pub fn attention_plan(
        &self,
        shapes: &[ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionShape],
        storage: usize,
        provider: usize,
    ) -> Result<ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionPlan> {
        if shapes
            .iter()
            .any(|s| s.tokens as usize > self.max_positions)
        {
            return Err(invalid("attention view exceeds context bound"));
        }
        let HybridLocalAttention::Gqa {
            query_heads,
            kv_heads,
            head_dim,
            ..
        } = self.layout.attention(
            self.gqa_layer()
                .ok_or_else(|| invalid("program has no GQA provider"))?,
        )?
        else {
            unreachable!()
        };
        ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionPlan::from_shapes(
            shapes,
            self.max_rows,
            self.max_sequences,
            query_heads.len(),
            kv_heads.len(),
            *head_dim,
            storage,
            provider,
        )
    }
    pub fn validate_spec(&self, spec: &super::super::DecoderModelSpec) -> Result<()> {
        if self.layout.spec.as_ref() != spec {
            return Err(invalid("foreign Mixed block model"));
        }
        Ok(())
    }
    /// DATA sites only. Sealed input identity and owner completion belong to
    /// the control methods, not BF16 Agreement/Completion arrays.
    pub fn sites(&self) -> Vec<HybridCollectiveSite> {
        let mut sites: Vec<_> = self
            .layout
            .state_schema()
            .layer_range()
            .enumerate()
            .flat_map(|(i, layer)| {
                [
                    HybridCollectiveSemantic::AttentionOutput,
                    match self.layout.spec.layers()[layer].feed_forward() {
                        FeedForward::SwiGlu(_) => HybridCollectiveSemantic::DenseMlpDown,
                        FeedForward::Moe(_) => HybridCollectiveSemantic::RoutedExpertOutput,
                    },
                ]
                .into_iter()
                .enumerate()
                .map(move |(j, semantic)| HybridCollectiveSite {
                    ordinal: (2 * i + j) as u64,
                    global_layer: Some(layer),
                    semantic,
                    kind: HybridCollectiveKind::ReduceSum,
                    shape: HybridCollectiveShape {
                        fixed: 0,
                        per_sequence: 0,
                        per_row: self.width,
                    },
                })
            })
            .collect();
        if self.segment.owns_output() {
            sites.push(HybridCollectiveSite {
                ordinal: sites.len() as u64,
                global_layer: None,
                semantic: HybridCollectiveSemantic::OutputHead,
                kind: HybridCollectiveKind::AllGather,
                shape: HybridCollectiveShape {
                    fixed: 0,
                    per_sequence: 0,
                    per_row: self
                        .layout
                        .spec
                        .vocab_size()
                        .div_ceil(self.layout.members().len()),
                },
            });
        }
        sites
    }
    /// RoutedExpertOutput in this sealed program is the whole routed FFN,
    /// after local shared addition. Legacy F32 schedules are unchanged.
    pub fn route_sites(&self) -> Vec<HybridMixedRouteSite> {
        self.layers()
            .enumerate()
            .filter_map(|(i, layer)| {
                let FeedForward::Moe(m) = self.layout.spec.layers()[layer].feed_forward() else {
                    return None;
                };
                Some(HybridMixedRouteSite {
                    layer,
                    output_site: (2 * i + 1) as u64,
                    experts: m.router_spec().num_experts(),
                    top_k: m.router_spec().experts_per_token(),
                    max_rows: self.max_rows,
                    accumulation: HybridRoutedAccumulation::HfEagerBf16ExpertOrder,
                    combine: HybridMoeCombineOrder::LocalBf16AddThenReduce,
                })
            })
            .collect()
    }
    pub fn gqa_layer(&self) -> Option<usize> {
        self.layers().find(|&layer| {
            matches!(
                self.layout.attention(layer),
                Ok(HybridLocalAttention::Gqa { .. })
            )
        })
    }
    /// Compatibility accessor for the first recurrent layer. Use recurrent_shapes
    /// for the complete program; every local SSM has its own global-layer identity.
    pub fn recurrent_layer(&self) -> Option<usize> {
        self.layers().find(|&layer| {
            matches!(
                self.layout.attention(layer),
                Ok(HybridLocalAttention::GatedDelta { .. })
            )
        })
    }
    pub fn recurrent_shape(
        &self,
    ) -> Result<Option<ferrule_backend::cpu::gated_delta::GatedDeltaShape>> {
        Ok(self.recurrent_shapes()?.first().map(|(_, shape)| *shape))
    }
    pub fn recurrent_shapes(
        &self,
    ) -> Result<Vec<(usize, ferrule_backend::cpu::gated_delta::GatedDeltaShape)>> {
        let mut shapes = Vec::new();
        for layer in self.layers() {
            if let HybridLocalAttention::GatedDelta { shape, .. } = self.layout.attention(layer)? {
                shapes.push((layer, *shape));
            }
        }
        Ok(shapes)
    }
    /// Sequential layers share the maximum chunk workspace, never their sum.
    pub fn recurrent_workspace_bytes(&self) -> Result<usize> {
        use ferrule_backend::cuda_plan::{Bf16ChunkPlan, GatedDeltaNetLayout};
        self.recurrent_shapes()?
            .into_iter()
            .try_fold(0, |maximum, (_, s)| {
                let plan = Bf16ChunkPlan::new_prefill(
                    GatedDeltaNetLayout {
                        rows: self.max_rows,
                        key_heads: s.key_heads,
                        value_heads: s.value_heads,
                        key_dim: s.key_dim,
                        value_dim: s.value_dim,
                    },
                    usize::MAX,
                )?;
                Ok(maximum.max(plan.workspace_elements() * 4))
            })
    }
    pub fn validate_input(
        &self,
        batch: &PackedDecoderBatch,
        shape: RowsShape,
        dtype: RowsDType,
        sessions: &[u64],
    ) -> Result<()> {
        if dtype != RowsDType::Bf16
            || shape.width() != self.width
            || shape.rows() != batch.len()
            || batch.len() > self.max_rows
            || batch.sequences().len() > self.max_sequences
            || sessions.len() != batch.sequences().len()
            || batch.page_size() != self.page_size
            || batch.positions().iter().any(|p| *p >= self.max_positions)
        {
            return Err(invalid("Mixed block input/capacity mismatch"));
        }
        Ok(())
    }
}
