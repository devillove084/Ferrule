use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use ferrule_backend::cuda::operators::{linear as cuda_linear, moe as cuda_moe};
use ferrule_common::{CompletionHub, Error, ResidencyLeaseSet, Result};

use super::RoutedMoePayload;
use crate::ffn::SwiGluFfn;
use crate::moe::cuda_materialization::{CudaExpertFrame, CudaSharedExpertSubsystem};
use crate::moe::prediction::{ExpertAccessPhase, ExpertBatchAccessEvent};
use crate::moe::routing::{
    ExpertRoute, ExpertRouterPolicy, RouterScoreFunction, RouterSelectionPolicy,
    RouterWeightNormalization,
};
use crate::moe::streaming::{ExpertId, ExpertStreamingStep};

const ROUTER_EXPERT_LIMIT: usize = 512;
const ROUTER_TOP_K_LIMIT: usize = 64;

enum PreparedRouterSelection {
    ScoreTopK {
        selection_bias: Option<cuda_linear::CudaF32Buffer>,
    },
    HashAssisted {
        table: cuda_moe::CudaRouterHashTable,
        hash_rows: usize,
        token_ids: RefCell<HashMap<usize, cuda_moe::CudaRouterTokenIds>>,
    },
}

/// CUDA resources for one model-independent routed feed-forward layer.
pub struct PreparedRoutedMoe {
    layer: usize,
    hidden_size: usize,
    expert_count: usize,
    router: cuda_linear::CudaArtifactLinearHandle,
    policy: ExpertRouterPolicy,
    selection: PreparedRouterSelection,
    shared: SwiGluFfn<cuda_linear::CudaArtifactLinearHandle>,
}

impl std::fmt::Debug for PreparedRoutedMoe {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedRoutedMoe")
            .field("layer", &self.layer)
            .field("hidden_size", &self.hidden_size)
            .field("expert_count", &self.expert_count)
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

impl PreparedRoutedMoe {
    /// Compile semantic router and shared-expert handles from a common payload.
    pub(crate) fn prepare(
        payload: &RoutedMoePayload,
        ops: &cuda_linear::CudaOperators,
        mut prepare_linear: impl FnMut(
            &crate::checkpoint::weight::LinearWeight,
        ) -> Result<cuda_linear::CudaArtifactLinearHandle>,
    ) -> Result<Self> {
        validate_cuda_policy(&payload.router_policy)?;
        let layer = payload.layer();
        let hidden_size = payload.hidden_size();
        let expert_count = payload.expert_count();
        if expert_count > ROUTER_EXPERT_LIMIT || payload.router_policy.top_k > expert_count {
            return Err(Error::Model {
                message: format!(
                    "unsupported CUDA routed-MoE topology: layer={layer} experts={expert_count} top_k={}",
                    payload.router_policy.top_k
                ),
            });
        }
        let selection = match payload.router_policy.selection {
            RouterSelectionPolicy::ScoreTopK => {
                let selection_bias = payload
                    .router
                    .bias
                    .as_deref()
                    .map(|bias| {
                        if bias.len() != expert_count {
                            return Err(Error::Model {
                                message: format!(
                                    "routed-MoE layer {layer} selection bias has {} values; expected {expert_count}",
                                    bias.len()
                                ),
                            });
                        }
                        ops.upload_f32_buffer(bias)
                    })
                    .transpose()?;
                PreparedRouterSelection::ScoreTopK { selection_bias }
            }
            RouterSelectionPolicy::Hash => {
                let table = payload
                    .router
                    .hash_table
                    .as_deref()
                    .ok_or_else(|| Error::Model {
                        message: format!(
                            "routed-MoE layer {layer} hash-assisted router has no selection table"
                        ),
                    })?;
                let expected = payload
                    .router
                    .hash_rows
                    .checked_mul(payload.router.hash_cols)
                    .ok_or_else(|| Error::Model {
                        message: format!(
                            "routed-MoE layer {layer} hash selection shape overflows usize"
                        ),
                    })?;
                if table.len() != expected {
                    return Err(Error::Model {
                        message: format!(
                            "routed-MoE layer {layer} hash selection shape requires {expected} entries, got {}",
                            table.len()
                        ),
                    });
                }
                PreparedRouterSelection::HashAssisted {
                    table: ops.upload_router_hash_table(
                        table,
                        payload.router.hash_rows,
                        payload.router.hash_cols,
                        expert_count,
                        payload.router_policy.top_k,
                    )?,
                    hash_rows: payload.router.hash_rows,
                    token_ids: RefCell::new(HashMap::new()),
                }
            }
        };
        Ok(Self {
            layer,
            hidden_size,
            expert_count,
            router: prepare_linear(&payload.router.weight)?,
            policy: payload.router_policy,
            selection,
            shared: SwiGluFfn {
                gate: prepare_linear(&payload.shared_expert.gate)?,
                up: prepare_linear(&payload.shared_expert.up)?,
                down: prepare_linear(&payload.shared_expert.down)?,
                swiglu_limit: payload.shared_expert.swiglu_limit,
            },
        })
    }

    pub const fn layer(&self) -> usize {
        self.layer
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub const fn expert_count(&self) -> usize {
        self.expert_count
    }

    pub const fn routes_per_token(&self) -> usize {
        self.policy.top_k
    }

    pub fn shared_intermediate_size(&self) -> usize {
        self.shared.gate.shape().out_features()
    }

    /// Submit router, route readback, and shared-expert work.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        &self,
        execution: &mut RoutedMoeExecution,
        input: &cuda_linear::CudaF32Buffer,
        shared_input: &cuda_linear::CudaPreparedFp8Activation<'_>,
        token_ids: &[u32],
        attribution: RoutedMoeAttribution<'_>,
        scratch: &mut RoutedMoeScratch,
    ) -> Result<RoutedMoeContinuation> {
        let tokens = token_ids.len();
        attribution.validate(tokens)?;
        scratch.validate(self, tokens, input)?;
        execution
            .ops
            .artifact_linear_rows_from_device_into_with_scratch(
                &self.router,
                input,
                tokens,
                &mut scratch.router_logits,
                &mut scratch.router_workspace,
            )?;
        self.route(
            &execution.ops,
            token_ids,
            &scratch.router_logits,
            &mut scratch.router_indices,
            &mut scratch.router_weights,
        )?;
        let route_count = tokens
            .checked_mul(self.policy.top_k)
            .ok_or_else(|| Error::Internal {
                message: "CUDA routed-MoE route count overflow".into(),
            })?;
        if route_count > i32::MAX as usize || tokens > i32::MAX as usize {
            return Err(Error::Internal {
                message: format!(
                    "CUDA routed-MoE exceeds i32 metadata ABI: rows={tokens} routes={route_count}"
                ),
            });
        }
        let download = execution.ops.submit_moe_route_download(
            &scratch.router_indices,
            &scratch.router_weights,
            route_count,
        )?;
        let mut route_state = RoutedMoeRouteState::RoutesPending {
            download,
            callback_armed: false,
        };
        execution.arm_route_download(&mut route_state);
        execution.ops.artifact_shared_ffn_into(
            &self.shared.gate,
            &self.shared.up,
            &self.shared.down,
            shared_input,
            &mut scratch.shared_hidden,
            &mut scratch.shared_hidden_fp8,
            tokens,
            1.0,
            self.shared.swiglu_limit,
            &mut scratch.output,
            false,
        )?;
        let expected_route_output =
            route_count
                .checked_mul(self.hidden_size)
                .ok_or_else(|| Error::Internal {
                    message: "CUDA routed-MoE route output size overflow".into(),
                })?;
        let (row_to_sequence, sequence_phases) = attribution.into_owned();
        let mut continuation = RoutedMoeContinuation {
            layer: self.layer,
            tokens,
            hidden_size: self.hidden_size,
            routes_per_token: self.policy.top_k,
            route_count,
            expected_route_output,
            route_state,
            routes_by_token: Vec::new(),
            unique_experts: Vec::new(),
            sequence_phases,
            row_to_sequence,
            swiglu_limit: self.shared.swiglu_limit,
            resident_slot_capacity: None,
            next_chunk_start: 0,
            current_chunk: None,
            input_prepared: false,
            expected_intermediate: None,
            streaming_steps: Vec::new(),
        };
        execution.prime(&mut continuation)?;
        Ok(continuation)
    }

    /// Resume one declared expert window. Lease custody stays in the caller.
    pub fn resume(
        &self,
        execution: &mut RoutedMoeExecution,
        mut continuation: RoutedMoeContinuation,
        leases: Option<&ResidencyLeaseSet>,
        input: &cuda_linear::CudaF32Buffer,
        scratch: &mut RoutedMoeScratch,
    ) -> Result<RoutedMoeProgress> {
        continuation.validate_prepared(self)?;
        scratch.validate(self, continuation.tokens, input)?;
        match resume_action(
            continuation.routes_pending(),
            continuation.current_chunk.is_some(),
            leases.map_or(0, ResidencyLeaseSet::len),
        )? {
            RoutedMoeResumeAction::PollSubmittedRoute => {
                execution.prime(&mut continuation)?;
                return Ok(RoutedMoeProgress::Waiting(continuation));
            }
            RoutedMoeResumeAction::SubmitExpertWindow => {}
        }
        let leases = leases.ok_or_else(|| Error::Execution {
            message: "routed-MoE expert resume has no generic residency lease".into(),
        })?;
        if execution.submit_current_chunk(&mut continuation, leases, input, scratch)? {
            Ok(RoutedMoeProgress::Complete {
                events: finish_events(&continuation),
            })
        } else {
            Ok(RoutedMoeProgress::Waiting(continuation))
        }
    }

    /// Poll route transfer to cancellation quiescence and discard physical state.
    pub fn cancel(
        &self,
        execution: &mut RoutedMoeExecution,
        continuation: &mut RoutedMoeContinuation,
    ) -> Result<RoutedMoeCancelProgress> {
        continuation.validate_prepared(self)?;
        if !execution.poll_routes(continuation)? {
            return Ok(RoutedMoeCancelProgress::Waiting);
        }
        continuation.current_chunk = None;
        continuation.next_chunk_start = continuation.unique_experts.len();
        Ok(RoutedMoeCancelProgress::Complete)
    }

    fn route(
        &self,
        ops: &cuda_linear::CudaOperators,
        token_ids: &[u32],
        logits: &cuda_linear::CudaF32Buffer,
        indices: &mut cuda_moe::CudaI32Buffer,
        weights: &mut cuda_linear::CudaF32Buffer,
    ) -> Result<()> {
        let rows = token_ids.len();
        match &self.selection {
            PreparedRouterSelection::ScoreTopK { selection_bias } => ops.route_sqrt_softplus_topk(
                logits,
                selection_bias.as_ref(),
                rows,
                self.expert_count,
                self.policy.top_k,
                self.policy.route_scale,
                indices,
                weights,
            ),
            PreparedRouterSelection::HashAssisted {
                table,
                hash_rows,
                token_ids: cached,
            } => {
                let mut cached = cached.borrow_mut();
                if let Some(device_ids) = cached.get_mut(&rows) {
                    ops.update_router_token_ids(token_ids, *hash_rows, device_ids)?;
                } else {
                    cached.insert(rows, ops.router_token_ids(token_ids, *hash_rows)?);
                }
                ops.route_hash_sqrt_softplus(
                    logits,
                    cached.get(&rows).expect("router token IDs inserted above"),
                    table,
                    rows,
                    self.expert_count,
                    self.policy.top_k,
                    self.policy.route_scale,
                    indices,
                    weights,
                )
            }
        }
    }
}

fn validate_cuda_policy(policy: &ExpertRouterPolicy) -> Result<()> {
    if policy.score_function != RouterScoreFunction::SqrtSoftplus
        || policy.weight_normalization != RouterWeightNormalization::SelectedSum
        || !policy.normalize_non_softmax_weights
        || policy.top_k == 0
        || policy.top_k > ROUTER_TOP_K_LIMIT
        || !policy.route_scale.is_finite()
    {
        return Err(Error::Model {
            message: format!("unsupported CUDA routed-MoE router policy: {policy:?}"),
        });
    }
    Ok(())
}

/// Packed-row attribution used to produce model-neutral predictor events.
#[derive(Debug, Clone, Copy)]
pub struct RoutedMoeAttribution<'a> {
    row_to_sequence: Option<&'a [usize]>,
    sequence_phases: Option<&'a [ExpertAccessPhase]>,
}

impl<'a> RoutedMoeAttribution<'a> {
    pub const fn none() -> Self {
        Self {
            row_to_sequence: None,
            sequence_phases: None,
        }
    }

    pub const fn packed(
        row_to_sequence: &'a [usize],
        sequence_phases: &'a [ExpertAccessPhase],
    ) -> Self {
        Self {
            row_to_sequence: Some(row_to_sequence),
            sequence_phases: Some(sequence_phases),
        }
    }

    fn validate(self, rows: usize) -> Result<()> {
        match (self.row_to_sequence, self.sequence_phases) {
            (None, None) => Ok(()),
            (Some(sequences), Some(phases))
                if sequences.len() == rows
                    && !phases.is_empty()
                    && sequences.iter().all(|sequence| *sequence < phases.len()) =>
            {
                Ok(())
            }
            _ => Err(Error::Model {
                message: "routed-MoE packed row attribution is inconsistent".into(),
            }),
        }
    }

    fn into_owned(self) -> (Option<Vec<usize>>, Option<Vec<ExpertAccessPhase>>) {
        (
            self.row_to_sequence.map(<[usize]>::to_vec),
            self.sequence_phases.map(<[ExpertAccessPhase]>::to_vec),
        )
    }
}

/// Caller-owned, shape-stable CUDA scratch for routed feed-forward execution.
pub struct RoutedMoeScratch {
    router_logits: cuda_linear::CudaF32Buffer,
    router_indices: cuda_moe::CudaI32Buffer,
    router_weights: cuda_linear::CudaF32Buffer,
    router_workspace: cuda_linear::CudaArtifactLinearWorkspace,
    shared_hidden: cuda_linear::CudaF32Buffer,
    shared_hidden_fp8: cuda_linear::CudaFp8ActivationPack,
    route_plan: Option<cuda_moe::CudaExpertGroupRoutePlan>,
    route_output: cuda_linear::CudaF32Buffer,
    output: cuda_linear::CudaF32Buffer,
    rows: usize,
}

impl RoutedMoeScratch {
    pub fn new(
        prepared: &PreparedRoutedMoe,
        rows: usize,
        ops: &cuda_linear::CudaOperators,
    ) -> Result<Self> {
        if rows == 0 {
            return Err(Error::Model {
                message: "CUDA routed-MoE scratch requires at least one row".into(),
            });
        }
        let routes = rows
            .checked_mul(prepared.policy.top_k)
            .ok_or_else(|| Error::Internal {
                message: "CUDA routed-MoE scratch route count overflow".into(),
            })?;
        let shared_intermediate = prepared.shared_intermediate_size();
        Ok(Self {
            router_logits: ops.zero_f32_buffer(rows * prepared.expert_count)?,
            router_indices: ops.zero_i32_buffer(routes)?,
            router_weights: ops.zero_f32_buffer(routes)?,
            router_workspace: ops.artifact_linear_workspace(rows, prepared.hidden_size)?,
            shared_hidden: ops.zero_f32_buffer(rows * shared_intermediate)?,
            shared_hidden_fp8: ops.fp8_activation_pack(rows, shared_intermediate)?,
            route_plan: None,
            route_output: ops.allocate_moe_route_output(
                rows,
                prepared.policy.top_k,
                prepared.hidden_size,
            )?,
            output: ops.zero_f32_buffer(rows * prepared.hidden_size)?,
            rows,
        })
    }

    pub const fn output(&self) -> &cuda_linear::CudaF32Buffer {
        &self.output
    }

    pub fn output_mut(&mut self) -> &mut cuda_linear::CudaF32Buffer {
        &mut self.output
    }

    pub const fn route_output(&self) -> &cuda_linear::CudaF32Buffer {
        &self.route_output
    }

    fn validate(
        &self,
        prepared: &PreparedRoutedMoe,
        rows: usize,
        input: &cuda_linear::CudaF32Buffer,
    ) -> Result<()> {
        let input_len = rows
            .checked_mul(prepared.hidden_size)
            .ok_or_else(|| Error::Model {
                message: "CUDA routed-MoE input size overflow".into(),
            })?;
        let route_output_len = rows
            .checked_mul(prepared.policy.top_k)
            .and_then(|routes| routes.checked_mul(prepared.hidden_size))
            .ok_or_else(|| Error::Model {
                message: "CUDA routed-MoE route output size overflow".into(),
            })?;
        if rows == 0
            || self.rows != rows
            || input.len() != input_len
            || self.output.len() != input_len
            || self.route_output.len() != route_output_len
        {
            return Err(Error::Model {
                message: "CUDA routed-MoE scratch or input shape changed while suspended".into(),
            });
        }
        Ok(())
    }
}

/// Device-wide services shared by every prepared routed-MoE layer.
pub struct RoutedMoeExecution {
    ops: Rc<cuda_linear::CudaOperators>,
    experts: CudaSharedExpertSubsystem,
    completion: CompletionHub,
}

impl std::fmt::Debug for RoutedMoeExecution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RoutedMoeExecution")
            .field("experts", &self.experts)
            .finish_non_exhaustive()
    }
}

impl RoutedMoeExecution {
    pub(crate) fn new(
        ops: Rc<cuda_linear::CudaOperators>,
        experts: CudaSharedExpertSubsystem,
        completion: CompletionHub,
    ) -> Self {
        Self {
            ops,
            experts,
            completion,
        }
    }

    pub fn begin_quiescence(&self) -> Result<RoutedMoeQuiescence> {
        let mut pending = RoutedMoeQuiescence {
            event: self.ops.record_compute_event()?,
            callback_armed: false,
        };
        self.arm_quiescence(&mut pending);
        Ok(pending)
    }

    pub fn poll_quiescence(&self, pending: &mut RoutedMoeQuiescence) -> Result<bool> {
        if pending.event.is_complete()? {
            return Ok(true);
        }
        self.arm_quiescence(pending);
        Ok(false)
    }

    fn arm_quiescence(&self, pending: &mut RoutedMoeQuiescence) {
        if pending.callback_armed {
            return;
        }
        let completion = self.completion.clone();
        pending.callback_armed = self
            .ops
            .notify_compute_stream(move || {
                completion.notify();
            })
            .is_ok();
        if !pending.callback_armed {
            self.completion.notify();
        }
    }

    fn arm_route_download(&self, state: &mut RoutedMoeRouteState) {
        let RoutedMoeRouteState::RoutesPending { callback_armed, .. } = state else {
            return;
        };
        if *callback_armed {
            return;
        }
        let completion = self.completion.clone();
        *callback_armed = self
            .ops
            .notify_control_stream(move || {
                completion.notify();
            })
            .is_ok();
        if !*callback_armed {
            self.completion.notify();
        }
    }

    fn prime(&mut self, continuation: &mut RoutedMoeContinuation) -> Result<()> {
        if self.poll_routes(continuation)? {
            self.prepare_next_chunk(continuation)?;
        }
        Ok(())
    }

    fn poll_routes(&mut self, continuation: &mut RoutedMoeContinuation) -> Result<bool> {
        let state = std::mem::replace(
            &mut continuation.route_state,
            RoutedMoeRouteState::RoutesReady,
        );
        let RoutedMoeRouteState::RoutesPending {
            mut download,
            callback_armed,
        } = state
        else {
            return Ok(true);
        };
        let compact = match self.ops.poll_moe_route_download(&mut download) {
            Ok(Some(compact)) => compact,
            Ok(None) => {
                let mut state = RoutedMoeRouteState::RoutesPending {
                    download,
                    callback_armed,
                };
                if !callback_armed {
                    self.arm_route_download(&mut state);
                }
                continuation.route_state = state;
                return Ok(false);
            }
            Err(error) => {
                continuation.route_state = RoutedMoeRouteState::RoutesPending {
                    download,
                    callback_armed,
                };
                return Err(error);
            }
        };
        continuation.routes_by_token =
            decode_compact_routes(&compact, continuation.tokens, continuation.routes_per_token)?;
        continuation.unique_experts = continuation
            .routes_by_token
            .iter()
            .flat_map(|routes| routes.iter().map(|route| route.expert))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if continuation.unique_experts.is_empty() {
            return Err(Error::Internal {
                message: "CUDA routed-MoE selected no experts".into(),
            });
        }
        Ok(true)
    }

    fn prepare_next_chunk(&mut self, continuation: &mut RoutedMoeContinuation) -> Result<()> {
        if continuation.routes_pending()
            || continuation.current_chunk.is_some()
            || continuation.next_chunk_start == continuation.unique_experts.len()
        {
            return Ok(());
        }
        if continuation.resident_slot_capacity.is_none() {
            let resident_slot_capacity = self
                .experts
                .layer_slot_capacity(continuation.layer)?
                .clamp(1, ROUTER_EXPERT_LIMIT);
            let resident = self
                .experts
                .resident_experts_for_layer(continuation.layer)?;
            continuation
                .unique_experts
                .sort_by_key(|expert| (!resident.contains(expert), *expert));
            continuation.resident_slot_capacity = Some(resident_slot_capacity);
        }
        let capacity = continuation
            .resident_slot_capacity
            .expect("resident capacity initialized above");
        let end = continuation
            .next_chunk_start
            .saturating_add(capacity)
            .min(continuation.unique_experts.len());
        let selected = continuation.unique_experts[continuation.next_chunk_start..end].to_vec();
        continuation.current_chunk = Some(RoutedMoeChunk {
            unresolved_experts: selected.clone(),
            streaming: ExpertStreamingStep {
                layer: continuation.layer,
                selected: selected
                    .iter()
                    .copied()
                    .map(|expert| ExpertId::new(continuation.layer, expert))
                    .collect(),
                prefetched: Vec::new(),
                loads: Vec::new(),
                evictions: Vec::new(),
            },
            selected,
        });
        Ok(())
    }

    fn submit_current_chunk(
        &mut self,
        continuation: &mut RoutedMoeContinuation,
        leases: &ResidencyLeaseSet,
        input: &cuda_linear::CudaF32Buffer,
        scratch: &mut RoutedMoeScratch,
    ) -> Result<bool> {
        if scratch.route_output.len() != continuation.expected_route_output {
            return Err(Error::Model {
                message: "CUDA routed-MoE route output shape changed while suspended".into(),
            });
        }
        let mut chunk = continuation
            .current_chunk
            .take()
            .ok_or_else(|| Error::Execution {
                message: "routed-MoE has no declared expert dependency; refusing resume replay"
                    .into(),
            })?;
        let experts = self.experts.clone();
        let dispatch = experts.with_validated_published_experts(
            continuation.layer,
            &chunk.selected,
            leases,
            |table, first_frame| {
                self.submit_chunk(continuation, table, first_frame, input, scratch)
            },
        );
        if let Err(error) = dispatch {
            continuation.current_chunk = Some(chunk);
            return Err(error);
        }
        chunk.unresolved_experts.clear();
        continuation.next_chunk_start = continuation
            .next_chunk_start
            .saturating_add(chunk.selected.len());
        continuation.streaming_steps.push(chunk.streaming);
        if continuation.next_chunk_start == continuation.unique_experts.len() {
            self.ops.reduce_moe_route_outputs_ranked(
                &scratch.route_output,
                continuation.tokens,
                continuation.routes_per_token,
                continuation.hidden_size,
                &mut scratch.output,
            )?;
            return Ok(true);
        }
        self.prepare_next_chunk(continuation)?;
        Ok(false)
    }

    fn submit_chunk(
        &mut self,
        continuation: &mut RoutedMoeContinuation,
        table: &cuda_moe::CudaExpertSlotTable,
        first_frame: &CudaExpertFrame,
        input: &cuda_linear::CudaF32Buffer,
        scratch: &mut RoutedMoeScratch,
    ) -> Result<()> {
        let intermediate = first_frame.intermediate_size();
        if first_frame.input_size() != continuation.hidden_size
            || first_frame.output_size() != continuation.hidden_size
        {
            return Err(Error::Model {
                message: format!(
                    "CUDA grouped MoE expert shape is [{},{},{}], expected hidden {}",
                    first_frame.input_size(),
                    intermediate,
                    first_frame.output_size(),
                    continuation.hidden_size
                ),
            });
        }
        if let Some(expected) = continuation.expected_intermediate {
            if intermediate != expected {
                return Err(Error::Model {
                    message: format!(
                        "CUDA routed-MoE intermediate changed from {expected} to {intermediate}"
                    ),
                });
            }
        } else {
            continuation.expected_intermediate = Some(intermediate);
        }
        let resident_capacity = continuation
            .resident_slot_capacity
            .expect("resident capacity initialized before expert execution");
        let plan_needs_init = scratch.route_plan.as_ref().is_none_or(|plan| {
            !plan.matches(
                resident_capacity,
                continuation.route_count,
                continuation.tokens,
                continuation.hidden_size,
                intermediate,
                continuation.hidden_size,
            )
        });
        if plan_needs_init {
            scratch.route_plan = Some(self.ops.expert_group_route_plan(
                resident_capacity,
                continuation.route_count,
                continuation.tokens,
                continuation.hidden_size,
                intermediate,
                continuation.hidden_size,
            )?);
            continuation.input_prepared = false;
        }
        let plan = scratch
            .route_plan
            .as_mut()
            .expect("route plan initialized above");
        if !continuation.input_prepared {
            self.ops.prepare_expert_group_route_input_from_device(
                input,
                continuation.tokens,
                continuation.hidden_size,
                plan,
            )?;
            self.ops.begin_expert_group_route_invocation(
                continuation.routes_per_token,
                plan,
                &mut scratch.route_output,
            )?;
            continuation.input_prepared = true;
        }
        self.ops.prepare_expert_group_route_plan(
            table,
            &scratch.router_indices,
            &scratch.router_weights,
            continuation.route_count,
            continuation.routes_per_token,
            plan,
        )?;
        self.ops.grouped_fp4_moe_from_prepared_plan(
            table,
            continuation.routes_per_token,
            continuation.swiglu_limit,
            plan,
            &mut scratch.route_output,
        )
    }
}

/// Route-download state retained across pause, resume, and cancellation.
pub enum RoutedMoeRouteState {
    RoutesPending {
        download: cuda_moe::routing::CudaMoeRouteDownload,
        callback_armed: bool,
    },
    RoutesReady,
}

struct RoutedMoeChunk {
    selected: Vec<usize>,
    unresolved_experts: Vec<usize>,
    streaming: ExpertStreamingStep,
}

/// Physical routed-MoE state. Decoder waits and lease custody are not retained.
pub struct RoutedMoeContinuation {
    layer: usize,
    tokens: usize,
    hidden_size: usize,
    routes_per_token: usize,
    route_count: usize,
    expected_route_output: usize,
    route_state: RoutedMoeRouteState,
    routes_by_token: Vec<Vec<ExpertRoute>>,
    unique_experts: Vec<usize>,
    sequence_phases: Option<Vec<ExpertAccessPhase>>,
    row_to_sequence: Option<Vec<usize>>,
    swiglu_limit: f32,
    resident_slot_capacity: Option<usize>,
    next_chunk_start: usize,
    current_chunk: Option<RoutedMoeChunk>,
    input_prepared: bool,
    expected_intermediate: Option<usize>,
    streaming_steps: Vec<ExpertStreamingStep>,
}

impl RoutedMoeContinuation {
    pub fn routes_pending(&self) -> bool {
        matches!(self.route_state, RoutedMoeRouteState::RoutesPending { .. })
    }

    pub fn pending_experts(&self) -> Vec<RoutedMoePendingExpert> {
        self.current_chunk
            .as_ref()
            .into_iter()
            .flat_map(|chunk| chunk.unresolved_experts.iter().copied())
            .map(|expert| RoutedMoePendingExpert {
                layer: self.layer,
                expert,
            })
            .collect()
    }

    fn validate_prepared(&self, prepared: &PreparedRoutedMoe) -> Result<()> {
        if self.layer != prepared.layer
            || self.hidden_size != prepared.hidden_size
            || self.routes_per_token != prepared.policy.top_k
        {
            return Err(Error::Execution {
                message: "routed-MoE continuation was resumed by a different prepared layer".into(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutedMoePendingExpert {
    pub layer: usize,
    pub expert: usize,
}

#[derive(Debug)]
pub struct RoutedMoeSequenceEvent {
    pub sequence_index: usize,
    pub event: ExpertBatchAccessEvent,
}

#[allow(clippy::large_enum_variant)]
pub enum RoutedMoeProgress {
    Waiting(RoutedMoeContinuation),
    Complete { events: Vec<RoutedMoeSequenceEvent> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutedMoeCancelProgress {
    Waiting,
    Complete,
}

pub struct RoutedMoeQuiescence {
    event: cuda_moe::CudaComputeEvent,
    callback_armed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoutedMoeResumeAction {
    PollSubmittedRoute,
    SubmitExpertWindow,
}

fn resume_action(
    routes_pending: bool,
    expert_window_pending: bool,
    lease_count: usize,
) -> Result<RoutedMoeResumeAction> {
    if routes_pending {
        if lease_count != 0 {
            return Err(Error::Execution {
                message: "routed-MoE received expert leases before route download completed".into(),
            });
        }
        return Ok(RoutedMoeResumeAction::PollSubmittedRoute);
    }
    if !expert_window_pending {
        return Err(Error::Execution {
            message: "routed-MoE has no declared expert dependency; refusing resume replay".into(),
        });
    }
    Ok(RoutedMoeResumeAction::SubmitExpertWindow)
}

fn decode_compact_routes(
    compact: &[i32],
    tokens: usize,
    top_k: usize,
) -> Result<Vec<Vec<ExpertRoute>>> {
    let route_count = tokens.checked_mul(top_k).ok_or_else(|| Error::Internal {
        message: "CUDA routed-MoE route count overflow".into(),
    })?;
    let expected = route_count.checked_mul(2).ok_or_else(|| Error::Internal {
        message: "CUDA compact routed-MoE route size overflow".into(),
    })?;
    if compact.len() != expected {
        return Err(Error::Internal {
            message: format!(
                "CUDA compact routed-MoE routes have {} values; expected {expected}",
                compact.len()
            ),
        });
    }
    (0..tokens)
        .map(|token| {
            (0..top_k)
                .map(|slot| {
                    let index = token * top_k + slot;
                    let expert =
                        usize::try_from(compact[index * 2]).map_err(|_| Error::Internal {
                            message: format!(
                                "CUDA compact routed-MoE route {index} has a negative expert"
                            ),
                        })?;
                    Ok(ExpertRoute {
                        expert,
                        weight: f32::from_bits(compact[index * 2 + 1] as u32),
                        score: 0.0,
                        selection_score: 0.0,
                    })
                })
                .collect()
        })
        .collect()
}

fn finish_events(continuation: &RoutedMoeContinuation) -> Vec<RoutedMoeSequenceEvent> {
    let Some(row_to_sequence) = continuation.row_to_sequence.as_deref() else {
        return Vec::new();
    };
    ExpertBatchAccessEvent::from_packed_routes_by_sequence(
        continuation.layer,
        continuation
            .sequence_phases
            .as_deref()
            .expect("packed phases validated at start"),
        row_to_sequence,
        &continuation.routes_by_token,
        &continuation.streaming_steps,
    )
    .into_iter()
    .map(|(sequence_index, event)| RoutedMoeSequenceEvent {
        sequence_index,
        event,
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_routes_decode_ranked_experts_and_weights() {
        let compact = [3, 0.25f32.to_bits() as i32, 1, 0.75f32.to_bits() as i32];
        let routes = decode_compact_routes(&compact, 1, 2).unwrap();
        assert_eq!(routes[0][0].expert, 3);
        assert_eq!(routes[0][0].weight, 0.25);
        assert_eq!(routes[0][1].expert, 1);
        assert_eq!(routes[0][1].weight, 0.75);
    }

    #[test]
    fn resume_requires_routes_before_leases_and_a_declared_window_afterward() {
        assert_eq!(
            resume_action(true, false, 0).unwrap(),
            RoutedMoeResumeAction::PollSubmittedRoute
        );
        assert!(resume_action(true, false, 1).is_err());
        assert!(resume_action(false, false, 0).is_err());
        assert_eq!(
            resume_action(false, true, 1).unwrap(),
            RoutedMoeResumeAction::SubmitExpertWindow
        );
    }
}
