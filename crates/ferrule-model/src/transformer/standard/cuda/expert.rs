//! Optional host-control EP seam. Expert matmuls/SiLU always run on CUDA.

use std::collections::BTreeMap;
use std::rc::Rc;

use ferrule_backend::cuda::operators::linear::CudaOperators;
use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{ParallelRankId, Result};

use crate::execution::ExecutionPrecisionPolicy;
use crate::transformer::expert_parallel::{
    ExpertDispatchLimits, ExpertDispatchPlan, ExpertPlacement, ExpertResult, ExpertResultExecutor,
    ExpertToken, ExpertWorker, RoutedSwiGluExecutor, RoutedSwiGluRequest,
};
use crate::transformer::{
    ExpertAvailability, ExpertProvider, HostRows, OperatorProgress, OperatorWaiting, RouterRoutes,
    Rows, RowsArenaId, RowsDType, RowsShape, StandardDecoderOperators, SwiGluScratchPlan,
    UnsupportedOperator,
};

use super::{CudaStandardDecoderOperators, cuda_error, device_rows};

/// Bucket-local metadata is bounded by the original route count. Every route
/// slot is assigned once; this is a permutation, never an atomic reduction.
struct LocalRouteBuckets {
    experts: BTreeMap<usize, Vec<usize>>,
    routes: usize,
    scratch_bytes: usize,
}

impl LocalRouteBuckets {
    fn new(routes: &RouterRoutes, shape: RowsShape) -> Result<Self> {
        if routes.rows() != shape.rows() {
            return Err(cuda_error("routed input row count mismatch"));
        }
        let count = routes
            .rows()
            .checked_mul(routes.top_k())
            .ok_or_else(|| cuda_error("route count overflow"))?;
        CudaStandardDecoderOperators::routed_bucket_scratch_bytes(
            shape.rows(),
            routes.top_k(),
            shape.width(),
            0,
        )?;
        let mut experts = BTreeMap::<usize, Vec<usize>>::new();
        // Same canonical row/slot contract as ExpertDispatchPlan, without
        // inventing a transaction/owner identity for this synchronous local seam.
        for row in 0..routes.rows() {
            let (ids, weights) = routes.row(row)?;
            let mut seen = std::collections::BTreeSet::new();
            for (slot, (&id, &weight)) in ids.iter().zip(weights).enumerate() {
                if !seen.insert(id) || !weight.is_finite() || weight < 0.0 {
                    return Err(cuda_error("duplicate expert or invalid route weight"));
                }
                experts
                    .entry(id)
                    .or_default()
                    .push(row * routes.top_k() + slot);
            }
        }
        let largest_bucket = experts.values().map(Vec::len).max().unwrap_or(0);
        let scratch_bytes = CudaStandardDecoderOperators::routed_bucket_scratch_bytes(
            shape.rows(),
            routes.top_k(),
            shape.width(),
            largest_bucket,
        )?;
        Ok(Self {
            experts,
            routes: count,
            scratch_bytes,
        })
    }
}

impl CudaStandardDecoderOperators {
    /// Pure layout plan shared with metadata-only hybrid admission. A unique
    /// top-k selection puts at most `rows` entries in any single expert bucket.
    /// Gathered input and expert result belong to the separate SwiGLU plan,
    /// together with gate/up/product; numeric workspace is reserved separately.
    pub(crate) fn routed_bucket_scratch_bytes(
        rows: usize,
        top_k: usize,
        width: usize,
        largest_bucket: usize,
    ) -> Result<usize> {
        SwiGluScratchPlan::route(rows, top_k, width, largest_bucket)?.total_bytes()
    }

    pub(super) fn routed_swiglu_batched(
        &mut self,
        layer: usize,
        input: &Rows,
        routes: &RouterRoutes,
        provider: &mut dyn ExpertProvider,
        arena: Option<RowsArenaId>,
    ) -> Result<OperatorProgress<Rows>> {
        let shape = input.shape();
        let width = shape.width();
        let buckets = LocalRouteBuckets::new(routes, shape)?;
        let mut prepared = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        let mut scratch_plans = BTreeMap::new();
        let mut waiting = Vec::new();
        // Validate *all* selected identities and the peak scratch requirement
        // before reserving credit, evicting weights or submitting any GPU work.
        for (&id, slots) in &buckets.experts {
            let rows = slots.len();
            if self.bounded_experts.is_some() || self.numeric.is_some() {
                let plan = if let Some(bindings) = provider.expert_metadata_bindings(layer, id)? {
                    Some(
                        self.bindings
                            .prepared_expert_metadata(layer, id)?
                            .for_bindings(bindings)?,
                    )
                } else {
                    provider.expert_metadata(layer, id)?
                };
                if let Some(plan) = plan {
                    let (_, bytes, scratch) = self.metadata_plan(&plan, layer, id, rows, width)?;
                    self.preflight_expert_admission(bytes, scratch, buckets.scratch_bytes)?;
                    self.preflight_bucket_numeric(plan.parameters(), rows)?;
                    scratch_plans.insert(id, scratch);
                    metadata.insert(id, plan);
                    continue;
                }
            }
            let availability = if let Some(expert) = self.experts.get(&(layer, id)) {
                ExpertAvailability::Ready(std::sync::Arc::clone(expert))
            } else {
                provider.expert(layer, id)?
            };
            match availability {
                ExpertAvailability::Ready(expert) => {
                    if expert.input_width() != width || expert.output_width() != width {
                        return Err(cuda_error("routed input/output width mismatch"));
                    }
                    if self.bounded_experts.is_some() || self.numeric.is_some() {
                        let (key, bytes, scratch) = self.expert_plan(&expert, rows)?;
                        if (key.layer, key.expert) != (layer, id) {
                            return Err(cuda_error("provider returned another routed expert"));
                        }
                        self.preflight_expert_admission(bytes, scratch, buckets.scratch_bytes)?;
                        let parameters = [expert.gate(), expert.up(), expert.down()]
                            .map(|p| p.parameter().binding().clone());
                        self.preflight_bucket_numeric(&parameters, rows)?;
                        scratch_plans.insert(id, scratch);
                        // Old providers may require payload preflight. Do not
                        // retain all selected quantized host experts at once.
                    } else {
                        for linear in [expert.gate(), expert.up(), expert.down()] {
                            self.bindings.validate(linear.parameter())?;
                        }
                        prepared.insert(id, expert);
                    }
                }
                ExpertAvailability::Waiting => waiting.push(id),
                ExpertAvailability::Unsupported(reason) => {
                    return Ok(OperatorProgress::Unsupported(UnsupportedOperator::new(
                        "routed_swiglu",
                        reason,
                    )));
                }
            }
        }
        if !waiting.is_empty() {
            return Ok(OperatorProgress::Waiting(OperatorWaiting::Experts {
                layer,
                experts: waiting,
            }));
        }
        self.cache_scratch(buckets.scratch_bytes)?;
        self.route_scratch_bytes = buckets.scratch_bytes;
        self.route_hold.push(
            self.f32(input)?
                .as_device_buffer()
                .slice(0, shape.elements())?,
        );
        let mut table = self.ops.zero_f32_buffer(buckets.routes * width)?;
        self.hold_route_buffer(&table)?;
        let weights = self.ops.upload_f32_buffer(routes.weights())?;
        self.hold_route_buffer(&weights)?;
        let mut output = self.ops.zero_f32_buffer(shape.elements())?;
        self.hold_route_buffer(&output)?;
        let route_rows = (0..buckets.routes)
            .map(|slot| (slot / routes.top_k()) as i32)
            .collect::<Vec<_>>();
        let route_rows = self.ops.upload_i32_buffer(&route_rows)?;
        self.route_ids
            .push(route_rows.as_device_buffer().slice(0, buckets.routes)?);
        for (&id, slots) in &buckets.experts {
            let rows = slots.len();
            if let Some(&scratch) = scratch_plans.get(&id) {
                self.cache_scratch(SwiGluScratchPlan::reserved_bytes(
                    scratch,
                    buckets.scratch_bytes,
                    0,
                )?)?;
            }
            let gather_ids = slots
                .iter()
                .map(|slot| (slot / routes.top_k()) as i32)
                .collect::<Vec<_>>();
            let gather_ids = self.ops.upload_i32_buffer(&gather_ids)?;
            self.route_ids
                .push(gather_ids.as_device_buffer().slice(0, rows)?);
            let scatter_ids = slots.iter().map(|&slot| slot as i32).collect::<Vec<_>>();
            let scatter_ids = self.ops.upload_i32_buffer(&scatter_ids)?;
            self.route_ids
                .push(scatter_ids.as_device_buffer().slice(0, rows)?);
            let gathered = self
                .ops
                .gather_f32_rows(self.f32(input)?, &gather_ids, rows, width)?;
            self.hold_route_buffer(&gathered)?;
            let gathered = device_rows(RowsShape::new(rows, width)?, arena, gathered)?;
            let values = if let Some(plan) = metadata.get(&id) {
                self.metadata_swiglu_rows(plan, provider, &gathered, arena)?
            } else {
                let expert = if self.bounded_experts.is_some() || self.numeric.is_some() {
                    match provider.expert(layer, id)? {
                        ExpertAvailability::Ready(expert) => expert,
                        ExpertAvailability::Waiting => {
                            return Ok(OperatorProgress::Waiting(OperatorWaiting::Experts {
                                layer,
                                experts: vec![id],
                            }));
                        }
                        ExpertAvailability::Unsupported(reason) => {
                            return Ok(OperatorProgress::Unsupported(UnsupportedOperator::new(
                                "routed_swiglu",
                                reason,
                            )));
                        }
                    }
                } else {
                    std::sync::Arc::clone(prepared.get(&id).expect("preflight expert"))
                };
                if expert.input_width() != width || expert.output_width() != width {
                    return Err(cuda_error("provider changed routed expert shape"));
                }
                if self.bounded_experts.is_some() || self.numeric.is_some() {
                    let (key, _, _) = self.expert_plan(&expert, rows)?;
                    if (key.layer, key.expert) != (layer, id) {
                        return Err(cuda_error("provider changed routed expert identity"));
                    }
                } else {
                    self.experts
                        .insert((layer, id), std::sync::Arc::clone(&expert));
                }
                self.swiglu_rows(&expert, &gathered, arena)?
            };
            let result_view = self
                .f32(&values)?
                .as_device_buffer()
                .slice(0, rows * width)?;
            self.route_hold.push(result_view);
            // Each zero-initialized destination has exactly one writer, across
            // all buckets. The scatter primitive is not used to reduce routes.
            self.ops.scatter_add_f32_rows(
                self.f32(&values)?,
                &scatter_ids,
                &mut table,
                rows,
                width,
            )?;
            self.consumer_proof()?;
            self.expert_hold.clear();
            self.route_hold.truncate(4);
            self.route_ids.truncate(1);
        }
        // Original (source row, top-k slot) order, with each frozen weight
        // applied exactly once by the existing non-atomic left-fold kernel.
        self.ops.weighted_combine_f32_into(
            &table,
            &route_rows,
            &weights,
            &mut output,
            shape.rows(),
            width,
        )?;
        let output = device_rows(shape, arena, output)?;
        self.trace_rows("routed.output", &output)?;
        Ok(OperatorProgress::Ready(output))
    }
}

#[cfg(test)]
impl CudaStandardDecoderOperators {
    /// Frozen pre-bucketing schedule for like-for-like test-only measurements.
    /// No alternate production forward or host activation round trip.
    fn routed_rows_reference(
        &mut self,
        layer: usize,
        input: &Rows,
        routes: &RouterRoutes,
        provider: &mut dyn ExpertProvider,
    ) -> Result<Rows> {
        self.synchronous(|this| {
            let width = input.shape().width();
            let mut metadata = BTreeMap::new();
            for &id in routes.expert_ids() {
                if metadata.contains_key(&id) {
                    continue;
                }
                let plan = provider.expert_metadata(layer, id)?.expect("test metadata");
                this.metadata_plan(&plan, layer, id, 1, width)?;
                metadata.insert(id, plan);
            }
            let bytes = (input.shape().elements() * 2 + width * 3 + 1) * 4;
            this.cache_scratch(bytes)?;
            this.route_scratch_bytes = bytes;
            this.route_hold.push(
                this.f32(input)?
                    .as_device_buffer()
                    .slice(0, input.shape().elements())?,
            );
            let mut output = this.ops.zero_f32_buffer(input.shape().elements())?;
            this.hold_route_buffer(&output)?;
            for row in 0..routes.rows() {
                let ids = this.ops.upload_i32_buffer(&[row as i32])?;
                this.route_ids.push(ids.as_device_buffer().slice(0, 1)?);
                let values = this.ops.gather_f32_rows(this.f32(input)?, &ids, 1, width)?;
                this.hold_route_buffer(&values)?;
                let row_input = device_rows(RowsShape::new(1, width)?, None, values)?;
                let mut row_output = this.ops.zero_f32_buffer(width)?;
                this.hold_route_buffer(&row_output)?;
                let (ids_for_row, weights) = routes.row(row)?;
                for (&id, &weight) in ids_for_row.iter().zip(weights) {
                    let values =
                        this.metadata_swiglu_rows(&metadata[&id], provider, &row_input, None)?;
                    this.expert_hold
                        .push(this.f32(&values)?.as_device_buffer().slice(0, width)?);
                    this.ops
                        .saxpy_into(weight, this.f32(&values)?, &mut row_output)?;
                    this.consumer_proof()?;
                    this.expert_hold.clear();
                }
                this.ops
                    .scatter_add_f32_rows(&row_output, &ids, &mut output, 1, width)?;
                this.consumer_proof()?;
                this.route_hold.truncate(2);
                this.route_ids.clear();
            }
            device_rows(input.shape(), None, output)
        })
    }
}

/// Host messages carry only unweighted expert results. Validation/ordering uses
/// the same dispatch plan as CPU EP, but weighting and reduction stay on CUDA.
/// The caller must supply GPU expert owners; no local expert fallback is used.
pub struct CudaExpertParallelRoutedExecutor<'a> {
    operators: &'a mut CudaStandardDecoderOperators,
    members: Vec<ParallelRankId>,
    placement: &'a ExpertPlacement,
    limits: ExpertDispatchLimits,
    results: &'a mut dyn ExpertResultExecutor,
}

impl<'a> CudaExpertParallelRoutedExecutor<'a> {
    pub fn new(
        operators: &'a mut CudaStandardDecoderOperators,
        members: Vec<ParallelRankId>,
        placement: &'a ExpertPlacement,
        limits: ExpertDispatchLimits,
        results: &'a mut dyn ExpertResultExecutor,
    ) -> Self {
        Self {
            operators,
            members,
            placement,
            limits,
            results,
        }
    }
}

impl RoutedSwiGluExecutor for CudaExpertParallelRoutedExecutor<'_> {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        check_active(request.context.transaction)?;
        self.operators.f32(request.input)?;
        if request.routes.rows() != request.input.shape().rows() {
            return Err(cuda_error("routed input row count mismatch"));
        }
        // Admission precedes even the boundary download, as on the host seam.
        let plan = ExpertDispatchPlan::new(
            request.context,
            self.members.clone(),
            self.placement,
            request.routes,
            request.sequences,
            request.input.shape().width(),
            self.limits,
        )?;
        let input = self
            .operators
            .synchronous(|this| this.boundary.download(&this.ops, this.f32(request.input)?))?;
        let buckets = plan.dispatch(&input, check_active)?;
        let results = plan.execute_results(self.results, buckets, check_active)?;
        let ordered = plan.ordered_results(&results, None)?;
        let values = ordered
            .iter()
            .flat_map(|r| r.output.iter().copied())
            .collect::<Vec<_>>();
        let rows = ordered
            .iter()
            .map(|r| {
                i32::try_from(r.source_row).map_err(|_| cuda_error("route row exceeds i32 ABI"))
            })
            .collect::<Result<Vec<_>>>()?;
        check_active(request.context.transaction)?;
        let output = self.operators.synchronous(|this| {
            let values = this.boundary.upload(&this.ops, &values)?;
            let rows = this.ops.upload_i32_buffer(&rows)?;
            // Frozen router weights, not values supplied by a peer. Sorted results
            // match the router's original (source_row, route_slot) order exactly.
            let weights = this.ops.upload_f32_buffer(request.routes.weights())?;
            let shape = request.input.shape();
            let mut output = this.ops.zero_f32_buffer(shape.elements())?;
            this.ops.weighted_combine_f32_into(
                &values,
                &rows,
                &weights,
                &mut output,
                shape.rows(),
                shape.width(),
            )?;
            device_rows(shape, request.arena, output)
        })?;
        check_active(request.context.transaction)?;
        Ok(OperatorProgress::Ready(output))
    }
}

/// Adapts a host-message dispatcher to GPU router outputs and resident input.
/// Only this explicit EP boundary stages activations; no attention/linear CPU
/// fallback is installed. The dispatcher must select CUDA expert workers.
pub struct CudaHostRoutedExecutor<'a> {
    staging: CudaStandardDecoderOperators,
    executor: &'a mut dyn RoutedSwiGluExecutor,
}

impl<'a> CudaHostRoutedExecutor<'a> {
    pub fn new(ops: Rc<CudaOperators>, executor: &'a mut dyn RoutedSwiGluExecutor) -> Result<Self> {
        Ok(Self {
            staging: CudaStandardDecoderOperators::new(ops, ExecutionPrecisionPolicy::f32(), &[])?,
            executor,
        })
    }
    pub fn needs_quarantine(&self) -> bool {
        self.staging.needs_quarantine()
    }
}

impl RoutedSwiGluExecutor for CudaHostRoutedExecutor<'_> {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        check_active(request.context.transaction)?;
        let input = self.staging.synchronous(|this| {
            let buffer = this.f32(request.input)?;
            let values = this.boundary.download(&this.ops, buffer)?;
            HostRows::new(request.input.shape(), RowsDType::F32, request.arena, values)
        })?;
        check_active(request.context.transaction)?;
        let input = Rows::Host(input);
        let result = self.executor.routed_swiglu(
            RoutedSwiGluRequest {
                input: &input,
                ..request
            },
            check_active,
        )?;
        check_active(request.context.transaction)?;
        match result {
            OperatorProgress::Ready(rows) => {
                self.staging.bind_rows(rows).map(OperatorProgress::Ready)
            }
            progress => Ok(progress),
        }
    }
}

/// Owner-local expert worker with a bounded prepared parameter directory.
/// Results are unweighted: the dispatch protocol applies each route weight once.
pub struct CudaExpertWorker<'a> {
    owner: ParallelRankId,
    placement: &'a ExpertPlacement,
    experts: &'a mut dyn ExpertProvider,
    operators: &'a mut CudaStandardDecoderOperators,
}

impl<'a> CudaExpertWorker<'a> {
    pub fn new(
        owner: ParallelRankId,
        placement: &'a ExpertPlacement,
        experts: &'a mut dyn ExpertProvider,
        operators: &'a mut CudaStandardDecoderOperators,
    ) -> Self {
        Self {
            owner,
            placement,
            experts,
            operators,
        }
    }
}

impl ExpertWorker for CudaExpertWorker<'_> {
    fn owner(&self) -> ParallelRankId {
        self.owner
    }

    fn compute(&mut self, tokens: &[ExpertToken]) -> Result<Vec<ExpertResult>> {
        let mut groups = BTreeMap::new();
        for token in tokens {
            if self
                .placement
                .owner(token.expert.layer, token.expert.expert)
                != Some(self.owner)
            {
                return Err(cuda_error("expert request is not assigned to this owner"));
            }
            groups
                .entry(token.expert)
                .or_insert_with(Vec::new)
                .push(token);
        }
        let mut results = Vec::with_capacity(tokens.len());
        for (expert, tokens) in groups {
            let prepared = match self.experts.expert(expert.layer, expert.expert)? {
                ExpertAvailability::Ready(prepared) => prepared,
                ExpertAvailability::Waiting => return Err(cuda_error("CUDA expert is waiting")),
                ExpertAvailability::Unsupported(reason) => return Err(cuda_error(reason)),
            };
            let width = prepared.input_width();
            if prepared.output_width() != width
                || tokens.iter().any(|token| {
                    token.payload.len() != width || token.payload.iter().any(|v| !v.is_finite())
                })
            {
                return Err(cuda_error("invalid CUDA expert payload/shape"));
            }
            let shape = RowsShape::new(tokens.len(), width)?;
            let values = tokens
                .iter()
                .flat_map(|token| token.payload.iter().copied())
                .collect();
            let input = self.operators.bind_rows(Rows::Host(HostRows::new(
                shape,
                RowsDType::F32,
                None,
                values,
            )?))?;
            let output =
                super::super::ready(self.operators.dense_swiglu(&prepared, &input, None)?)?;
            let output = self.operators.download_rows(output)?;
            if output.shape() != shape {
                return Err(cuda_error("CUDA expert output shape mismatch"));
            }
            for (token, row) in tokens.into_iter().zip(output.values().chunks_exact(width)) {
                results.push(ExpertResult::from_token(token, self.owner, row.to_vec()));
            }
        }
        Ok(results)
    }
}

#[cfg(test)]
mod bucket_tests {
    use super::*;
    use crate::checkpoint::{CheckpointDType, CheckpointTensorSlice};
    use crate::nn::{ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec};
    use crate::support::TensorRole;
    use crate::transformer::{
        BoundParameter, ExactNameMapper, ExpertCacheLimits, ExpertCachePolicy, ExpertMetadata,
        NameMapping, NumericFp8Precision, PreparedLinear, PreparedSwiGlu, StateDictBinder,
        StateDictMaterializer, StateDictSchema,
    };
    use std::sync::{Arc, Weak};

    struct Fixture {
        path: std::path::PathBuf,
        parameters: Vec<[BoundParameter; 3]>,
    }
    impl Fixture {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "bucket-{}-{}.bin",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let mut payload = vec![0x38; 96];
            payload.extend(0x3d80u16.to_le_bytes());
            std::fs::write(&path, payload).unwrap();
            let mut schema = StateDictSchema::builder();
            let mut mapper = ExactNameMapper::new();
            let mut slices = Vec::new();
            for expert in 0..8 {
                for (index, role) in [
                    TensorRole::RoutedExpertGate,
                    TensorRole::RoutedExpertUp,
                    TensorRole::RoutedExpertDown,
                ]
                .into_iter()
                .enumerate()
                {
                    let name = format!("expert{expert}.p{index}");
                    let module = ModulePath::new(&name).unwrap();
                    let shape = if index == 2 { vec![8, 12] } else { vec![12, 8] };
                    let spec = ParameterSpec::new(
                        ParameterId::new((expert * 3 + index + 1) as u64),
                        module.clone(),
                        ParameterDType::F8E4M3,
                        shape.clone(),
                        ParameterResidency::expert(0, expert),
                    )
                    .unwrap()
                    .with_required_scale(ParameterDType::Bf16, [1, 1])
                    .unwrap();
                    schema.register_with_role(spec, role.clone()).unwrap();
                    mapper
                        .insert(&name, NameMapping::weight(module.clone()))
                        .unwrap();
                    mapper
                        .insert(format!("{name}_scale"), NameMapping::scale(module))
                        .unwrap();
                    slices.push(CheckpointTensorSlice {
                        name: name.clone(),
                        path: path.clone(),
                        role: role.clone(),
                        offset: 0,
                        bytes: 96,
                        shape,
                        dtype: CheckpointDType::F8E4M3,
                    });
                    slices.push(CheckpointTensorSlice {
                        name: format!("{name}_scale"),
                        path: path.clone(),
                        role,
                        offset: 96,
                        bytes: 2,
                        shape: vec![1, 1],
                        dtype: CheckpointDType::Bf16,
                    });
                }
            }
            let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
                .bind_slices(slices)
                .unwrap();
            let parameters = (0..8)
                .map(|expert| {
                    [0, 1, 2].map(|i| {
                        bound
                            .get_by_id(ParameterId::new(expert * 3 + i + 1))
                            .unwrap()
                            .clone()
                    })
                })
                .collect();
            Self { path, parameters }
        }
        fn owner(
            &self,
            precision: NumericFp8Precision,
            bytes: usize,
        ) -> CudaStandardDecoderOperators {
            CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
                Rc::new(CudaOperators::new_on_device(0).unwrap()),
                &self
                    .parameters
                    .iter()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>(),
                ExpertCachePolicy::Bounded(ExpertCacheLimits {
                    max_experts: 1,
                    max_bytes: bytes,
                }),
                4096,
                precision,
            )
            .unwrap()
        }
        fn provider(&self, metadata: bool) -> Provider<'_> {
            Provider {
                fixture: self,
                metadata,
                calls: 0,
                weak: Vec::new(),
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
    struct Provider<'a> {
        fixture: &'a Fixture,
        metadata: bool,
        calls: usize,
        weak: Vec<Weak<PreparedSwiGlu>>,
    }
    impl ExpertProvider for Provider<'_> {
        fn expert_metadata(&mut self, layer: usize, id: usize) -> Result<Option<ExpertMetadata>> {
            if !self.metadata {
                return Ok(None);
            }
            ExpertMetadata::new(layer, id, self.fixture.parameters[id].clone(), None).map(Some)
        }
        fn expert(&mut self, layer: usize, id: usize) -> Result<ExpertAvailability> {
            self.calls += 1;
            let materializer = StateDictMaterializer::new(1024).unwrap();
            let linear = |i: usize| {
                let p = &self.fixture.parameters[id][i];
                PreparedLinear::from_parameter(
                    materializer.expert_parameter(layer, id, p)?,
                    p.role().clone(),
                )
            };
            let expert = Arc::new(PreparedSwiGlu::new(
                linear(0)?,
                linear(1)?,
                linear(2)?,
                None,
            )?);
            self.weak.push(Arc::downgrade(&expert));
            Ok(ExpertAvailability::Ready(expert))
        }
    }
    fn input(rows: usize) -> Rows {
        Rows::Host(
            HostRows::new(
                RowsShape::new(rows, 8).unwrap(),
                RowsDType::F32,
                None,
                (0..rows * 8)
                    .map(|i| ((i * 7 % 31) as f32 - 12.0) * 0.013)
                    .collect(),
            )
            .unwrap(),
        )
    }
    fn routes(rows: usize) -> RouterRoutes {
        RouterRoutes::new(
            rows,
            8,
            (0..rows)
                .flat_map(|row| (0..8).map(move |slot| (slot + row) % 8))
                .collect(),
            (0..rows)
                .flat_map(|_| (1..=8).map(|slot| slot as f32 / 36.0))
                .collect(),
        )
        .unwrap()
    }
    fn ready(p: OperatorProgress<Rows>) -> Rows {
        match p {
            OperatorProgress::Ready(v) => v,
            _ => panic!("expected ready"),
        }
    }
    fn close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (&a, &b) in a.iter().zip(b) {
            assert!(
                a.is_finite()
                    && b.is_finite()
                    && (a as f64 - b as f64).abs() <= 2e-4 + 2e-4 * (b as f64).abs(),
                "{a} != {b}"
            );
        }
    }

    #[test]
    #[ignore = "requires CUDA; compares prepared, metadata and operation scratch at exact cap"]
    fn scratch_consumers_agree_at_exact_cap_and_minus_one() {
        let fixture = Fixture::new();
        let descriptor = crate::transformer::SwiGlu::new(8, 12, false).unwrap();
        for precision in [
            NumericFp8Precision::Bf16RneF32Accumulate,
            NumericFp8Precision::F32Tf32x3,
        ] {
            for rows in [0, 1, 3, 32] {
                let mut provider = fixture.provider(true);
                let metadata = provider.expert_metadata(0, 0).unwrap().unwrap();
                let ExpertAvailability::Ready(expert) = provider.expert(0, 0).unwrap() else {
                    panic!("ready fixture")
                };
                let route =
                    CudaStandardDecoderOperators::routed_bucket_scratch_bytes(rows, 2, 8, rows)
                        .unwrap();
                let activation = SwiGluScratchPlan::for_descriptor(&descriptor, rows)
                    .unwrap()
                    .total_bytes()
                    .unwrap();
                let total = activation + route + 4096;
                let cap = 294 + total;
                for (cap, admitted) in [(cap, true), (cap - 1, false)] {
                    let mut owner = fixture.owner(precision, cap);
                    let prepared = owner.expert_plan(&expert, rows).unwrap();
                    assert_eq!(
                        prepared,
                        owner.metadata_plan(&metadata, 0, 0, rows, 8).unwrap()
                    );
                    assert_eq!((prepared.1, prepared.2), (294, activation));
                    assert_eq!(
                        owner
                            .preflight_expert_admission(prepared.1, prepared.2, route)
                            .is_ok(),
                        admitted
                    );
                    owner.route_scratch_bytes = route;
                    let mut called = false;
                    let result = owner.synchronous(|this| {
                        this.expert_operation(&expert, rows, |this| {
                            called = true;
                            assert_eq!(
                                this.numeric.as_ref().unwrap().operation_bytes,
                                activation + route
                            );
                            assert_eq!(this.expert_cache_stats().unwrap().scratch_bytes, total);
                            Ok(())
                        })
                    });
                    assert_eq!(result.is_ok(), admitted);
                    assert_eq!(called, admitted);
                    assert_eq!(owner.expert_cache_stats().unwrap().scratch_bytes, 4096);
                    assert!(!owner.needs_quarantine());
                }
            }
        }
    }

    #[test]
    fn buckets_preserve_original_slots_ties_and_skip_empty_experts() {
        let routes = RouterRoutes::new(3, 2, vec![3, 1, 1, 3, 3, 1], vec![0.5; 6]).unwrap();
        let plan = LocalRouteBuckets::new(&routes, RowsShape::new(3, 8).unwrap()).unwrap();
        assert_eq!(plan.experts[&1], [1, 2, 5]);
        assert_eq!(plan.experts[&3], [0, 3, 4]);
        assert!(!plan.experts.contains_key(&2));
        assert_eq!(plan.routes, 6);
        // Check the local GPU permutation against the existing EP dispatch
        // contract, without introducing fake transactions into production.
        let owner = ferrule_common::ParallelRankId::new(0);
        let placement = ExpertPlacement::new([(0, 1, owner), (0, 3, owner)]).unwrap();
        let dispatch = ExpertDispatchPlan::new(
            crate::transformer::ExpertDispatchContext {
                transaction: ExecutionTransactionId::new(1).unwrap(),
                source_rank: owner,
                layer: 0,
            },
            vec![owner],
            &placement,
            &routes,
            &[10, 10, 11],
            8,
            ExpertDispatchLimits {
                max_tokens: 6,
                max_bytes: 6 * 8 * 4,
            },
        )
        .unwrap();
        let buckets = dispatch
            .dispatch(&vec![0.25; 3 * 8], &mut |_| Ok(()))
            .unwrap();
        let mut expected = BTreeMap::<usize, Vec<usize>>::new();
        for token in &buckets[0].tokens {
            expected
                .entry(token.expert.expert)
                .or_default()
                .push(token.source_row * routes.top_k() + token.route_slot);
        }
        assert_eq!(plan.experts, expected);
        for weights in [
            vec![f32::NAN, 1.0],
            vec![f32::INFINITY, 1.0],
            vec![-0.1, 1.0],
        ] {
            let bad = RouterRoutes::new(1, 2, vec![0, 1], weights).unwrap();
            assert!(LocalRouteBuckets::new(&bad, RowsShape::new(1, 8).unwrap()).is_err());
        }
        let duplicate = RouterRoutes::new(1, 2, vec![0, 0], vec![0.5; 2]).unwrap();
        assert!(LocalRouteBuckets::new(&duplicate, RowsShape::new(1, 8).unwrap()).is_err());
    }

    #[test]
    #[ignore = "requires CUDA; bucket budget rejection, legacy provider and observer cancellation"]
    fn bucket_preflight_budget_fallback_and_cancel_are_clean() {
        let f = Fixture::new();
        let precision = NumericFp8Precision::F32Tf32x3;
        let mut tiny = f.owner(precision, 6000);
        let input = tiny.bind_rows(input(23)).unwrap();
        let mut provider = f.provider(true);
        let before = tiny.expert_cache_stats().unwrap();
        tiny.operators().reset_counters();
        assert!(
            tiny.routed_swiglu(0, &input, &routes(23), &mut provider, None)
                .is_err()
        );
        assert_eq!(tiny.expert_cache_stats().unwrap(), before);
        assert_eq!(provider.calls, 0);
        assert_eq!(tiny.operators().counters().kernel_launches, 0);
        assert_eq!(tiny.operators().counters().device_allocation_attempts, 0);
        let mut owner = f.owner(precision, 32768);
        let input = owner.bind_rows(super::bucket_tests::input(23)).unwrap();
        let mut provider = f.provider(false);
        let expected = ready(
            owner
                .routed_swiglu(0, &input, &routes(23), &mut provider, None)
                .unwrap(),
        );
        let expected = owner.download_rows(expected).unwrap();
        assert_eq!(
            provider.calls, 16,
            "legacy preflight plus one materialization per bucket"
        );
        assert_eq!(owner.expert_cache_stats().unwrap().uploads, 8);
        assert!(provider.weak.iter().all(|w| w.upgrade().is_none()));
        owner.set_diagnostic_trace(|event| {
            if event.name == "linear.output" {
                return Err(cuda_error("injected observer cancellation"));
            }
            Ok(())
        });
        assert!(
            owner
                .routed_swiglu(0, &input, &routes(23), &mut provider, None)
                .is_err()
        );
        assert!(!owner.needs_quarantine());
        assert_eq!(owner.expert_cache_stats().unwrap().pending_upload_bytes, 0);
        owner.clear_diagnostic_trace();
        let actual = ready(
            owner
                .routed_swiglu(0, &input, &routes(23), &mut provider, None)
                .unwrap(),
        );
        close(
            owner.download_rows(actual).unwrap().values(),
            expected.values(),
        );
    }

    #[test]
    #[ignore = "requires CUDA; unknown bucket consumer retains lease and all route buffers"]
    fn bucket_unknown_completion_keeps_table_ids_and_credit() {
        let f = Fixture::new();
        let mut owner = f.owner(NumericFp8Precision::F32Tf32x3, 32768);
        let input = owner.bind_rows(input(23)).unwrap();
        let mut provider = f.provider(true);
        owner.lose_next_consumer_proof = true;
        assert!(
            owner
                .routed_swiglu(0, &input, &routes(23), &mut provider, None)
                .is_err()
        );
        assert!(owner.needs_quarantine());
        let frozen = owner.expert_cache_stats().unwrap();
        assert!(frozen.pending_upload_bytes > 0);
        assert!(
            frozen.unknown_quarantine_bytes >= frozen.pending_upload_bytes + frozen.scratch_bytes
        );
        assert!(owner.route_hold.len() >= 5);
        assert!(owner.route_ids.len() >= 3);
        assert!(owner.quiesce().is_err());
        assert_eq!(owner.expert_cache_stats().unwrap(), frozen);
        let ops = Rc::clone(owner.operators());
        let live = ops.allocator_metrics().live_requested_bytes;
        drop(owner);
        assert!(ops.allocator_metrics().live_requested_bytes >= live);
    }

    #[test]
    #[ignore = "requires exclusive CUDA; real 23-row top8 cap1 before/after microbenchmark"]
    fn routed_23_rows_top8_bucket_benchmark() {
        let f = Fixture::new();
        for precision in [
            NumericFp8Precision::F32Tf32x3,
            NumericFp8Precision::Bf16RneF32Accumulate,
        ] {
            let mut old = f.owner(precision, 32768);
            let mut new = f.owner(precision, 32768);
            let old_input = old.bind_rows(input(23)).unwrap();
            let new_input = new.bind_rows(input(23)).unwrap();
            let routes = routes(23);
            let mut old_provider = f.provider(true);
            let mut new_provider = f.provider(true);
            // One warm iteration, then report the median of three real calls.
            let mut timings = (Vec::new(), Vec::new());
            for iteration in 0..4 {
                old.operators().reset_counters();
                let before = old.expert_cache_stats().unwrap();
                let started = std::time::Instant::now();
                let reference = old
                    .routed_rows_reference(0, &old_input, &routes, &mut old_provider)
                    .unwrap();
                let old_ms = started.elapsed().as_secs_f64() * 1000.0;
                let old_counts = old.operators().counters();
                let old_uploads = old.expert_cache_stats().unwrap().uploads - before.uploads;
                let reference = old.download_rows(reference).unwrap();
                new.operators().reset_counters();
                let before = new.expert_cache_stats().unwrap();
                let calls = new_provider.calls;
                let started = std::time::Instant::now();
                let actual = ready(
                    new.routed_swiglu(0, &new_input, &routes, &mut new_provider, None)
                        .unwrap(),
                );
                let new_ms = started.elapsed().as_secs_f64() * 1000.0;
                let new_counts = new.operators().counters();
                let stats = new.expert_cache_stats().unwrap();
                let new_uploads = stats.uploads - before.uploads;
                assert_eq!(new_provider.calls - calls, 8);
                assert_eq!(new_uploads, 8);
                assert!(old_uploads >= 183);
                assert!(new_counts.kernel_launches < old_counts.kernel_launches);
                assert!(new_counts.artifact_uploads < old_counts.artifact_uploads);
                assert!(stats.peak_bytes <= 32768);
                assert!(new_provider.weak.iter().all(|w| w.upgrade().is_none()));
                close(
                    new.download_rows(actual).unwrap().values(),
                    reference.values(),
                );
                eprintln!(
                    "{precision:?} iteration={iteration} rows=23 topk=8 cap=1: rowwise {old_ms:.3}ms kernels={} artifact_uploads={} h2d={}B expert_uploads={old_uploads}; bucket {new_ms:.3}ms kernels={} artifact_uploads={} h2d={}B expert_uploads={new_uploads}",
                    old_counts.kernel_launches,
                    old_counts.artifact_uploads,
                    old_counts.host_to_device_bytes,
                    new_counts.kernel_launches,
                    new_counts.artifact_uploads,
                    new_counts.host_to_device_bytes
                );
                if iteration > 0 {
                    timings.0.push(old_ms);
                    timings.1.push(new_ms);
                }
            }
            timings.0.sort_by(f64::total_cmp);
            timings.1.sort_by(f64::total_cmp);
            eprintln!(
                "{precision:?} T23 measured median: {:.3}ms -> {:.3}ms",
                timings.0[1], timings.1[1]
            );
            let ops = Rc::clone(new.operators());
            drop(new_input);
            drop(new);
            ops.trim_device_allocator().unwrap();
            assert_eq!(ops.allocator_metrics().live_requested_bytes, 0);
        }
    }
}
