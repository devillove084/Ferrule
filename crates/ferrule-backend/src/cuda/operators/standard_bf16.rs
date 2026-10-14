//! Eager BF16 boundaries: typed storage, F32 reductions, no host activation math.
//!
//! The operator family borrows native dispatch without publishing owner resources.
//! ```compile_fail,E0624
//! use ferrule_backend::cuda::operator_owner::CudaOperators;
//! fn expose(op: &CudaOperators) { let _ = op.core_native(); }
//! ```
//! ```compile_fail,E0624
//! use ferrule_backend::cuda::operator_owner::{CudaOperators, CudaBf16Buffer};
//! fn expose(op: &CudaOperators, b: &CudaBf16Buffer) {
//!     let _ = op.check_buffer_owner(b.as_device_buffer(), "external");
//! }
//! ```
use crate::cuda::operator_owner::{
    CudaBf16Buffer, CudaF32Buffer, CudaI32Buffer, CudaOperators, CudaTypedBuffer, cu,
};
use crate::cuda::operators::OperatorOwner;
use crate::cuda::operators::linear::{Bf16GemmLayout, bf16_gemm};
use crate::cuda::operators::{PagedBf16CausalGqaLayout, SplitHalfRopeLayout};
use crate::cuda::runtime::DeviceCopy;
pub use crate::cuda_plan::Bf16GateLayout;
use crate::cuda_plan::standard_bf16::{invalid, shape};
use ferrule_common::Result;

use crate::cuda::ffi::core::{
    Bf16BoundaryArgs as BoundaryArgs, TRANSFORMER_PAGED_BF16_APPEND_CAUSAL_GQA,
    ferrule_core_bf16_boundary,
};
#[path = "bf16_route_metadata.rs"]
mod route_metadata;
use route_metadata::{PackedRoutes, Preparation, RouteBuffer, RouteOperation, validate_operands};

/// Opaque `[i32[R] | f32bits[R]]` storage preserving caller route order.
/// Allocate, install in caller-owned custody, then prepare once. Keep metadata
/// (including its immutable host source) and gathered/value roots through the
/// original last-consumer fence; retain on Unknown. No extra fence or raw view.
/// ```compile_fail,E0599
/// use ferrule_backend::cuda::operators::standard_bf16::CudaBf16RouteMetadata;
/// fn expose(metadata: &CudaBf16RouteMetadata) { let _ = metadata.as_device_buffer(); }
/// ```
/// ```compile_fail,E0616
/// use ferrule_backend::cuda::operators::standard_bf16::CudaBf16RouteMetadata;
/// fn expose(metadata: &CudaBf16RouteMetadata) { let _ = &metadata.device; }
/// ```
pub struct CudaBf16RouteMetadata {
    packed: PackedRoutes,
    device: CudaTypedBuffer<u32>,
    preparation: Preparation,
}
impl CudaBf16RouteMetadata {
    pub fn rows(&self) -> usize {
        self.packed.rows
    }
    pub fn len(&self) -> usize {
        self.packed.routes
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Original F32 bits; no BF16 conversion or normalization.
    pub fn weight_bits(&self) -> &[u32] {
        &self.packed.words[self.len()..]
    }
    pub fn row_ids(&self) -> impl ExactSizeIterator<Item = i32> + '_ {
        self.packed.words[..self.len()]
            .iter()
            .map(|&row| row as i32)
    }
    /// Submission readiness only, never a completion proof.
    pub fn is_prepared(&self) -> bool {
        self.preparation == Preparation::Ready
            && !self
                .device
                .as_device_buffer()
                .context()
                .graph_completion_unknown()
    }
}
impl<'a> RouteBuffer<'a, crate::cuda::runtime::CudaContext> {
    fn of<T: DeviceCopy>(buffer: &'a CudaTypedBuffer<T>) -> Self {
        let device = buffer.as_device_buffer();
        Self {
            context: device.context(),
            start: device.cu_deviceptr(),
            len: buffer.len(),
        }
    }
}

impl CudaOperators {
    /// Validate rows 1..=8, at most eight routes per row / R<=64, row bounds,
    /// equal counts and finite F32 weights, preserving bits and order.
    /// Allocation only: no copy/launch/sync. Pack ascending expert order, not
    /// top-k rank order. Install caller custody before prepare; keep expert M.
    pub fn allocate_bf16_route_metadata(
        &self,
        row_ids: &[i32],
        weights: &[f32],
        rows: usize,
    ) -> Result<CudaBf16RouteMetadata> {
        let packed = PackedRoutes::new(row_ids, weights, rows)?;
        let device = self.allocate_bf16_route_words(packed.words.len())?;
        packed.pointers(device.as_device_buffer().cu_deviceptr(), device.len())?;
        Ok(CudaBf16RouteMetadata {
            packed,
            device,
            preparation: Preparation::Allocated,
        })
    }

    /// One same-compute-stream H2D of exactly 8*R bytes, no fence or sync.
    /// Caller already owns custody through the original completion fence.
    /// Failure never publishes readiness; retain source/device on Unknown and
    /// never retry preparation. Gather/combine consume the same opaque storage.
    pub fn prepare_bf16_route_metadata(&self, metadata: &mut CudaBf16RouteMetadata) -> Result<()> {
        self.check_buffer_owner(metadata.device.as_device_buffer(), "BF16 route metadata")?;
        let stream = self.stream_clone();
        cu(stream.context().check_graph_ready())?;
        metadata.packed.pointers(
            metadata.device.as_device_buffer().cu_deviceptr(),
            metadata.device.len(),
        )?;
        metadata.preparation.submit(
            || self.upload_bf16_route_words(&metadata.packed.words, &metadata.device),
            || cu(stream.context().check_graph_ready()),
        )
    }

    /// Gather all expert-major routes from full `[rows,width]` BF16 input into
    /// caller-owned exact `[R,width]` output with the original kind2 kernel.
    /// One launch, no copy/allocation/sync; preserve expert partition M/offsets.
    /// Retain metadata/input/output roots through the original consumer fence.
    pub fn gather_bf16_routes_into(
        &self,
        input: &CudaBf16Buffer,
        metadata: &CudaBf16RouteMetadata,
        output: &mut CudaBf16Buffer,
        width: usize,
    ) -> Result<()> {
        metadata.preparation.require_ready()?;
        let stream = self.stream_clone();
        cu(stream.context().check_graph_ready())?;
        let (row_ids, _) = validate_operands(
            stream.context(),
            &metadata.packed,
            RouteBuffer::of(&metadata.device),
            RouteBuffer::of(input),
            RouteBuffer::of(output),
            width,
            RouteOperation::Gather,
        )?;
        self.bf16_boundary_submit(BoundaryArgs {
            kind: 2,
            rows: metadata.len() as u32,
            width: width as u32,
            input: input.as_device_buffer().cu_deviceptr(),
            positions: row_ids,
            output: output.as_device_buffer().cu_deviceptr(),
            ..Default::default()
        })
    }

    /// Original single-kernel BF16 product + accumulate RNE left-fold, in
    /// ascending-expert order. Read row IDs and F32 weights from one allocation.
    /// Values are BF16 down projections `[R,width]`, NOT F32 scratch. Exact
    /// `[rows,width]` output; no hidden allocation/copy/fence/synchronization.
    pub fn weighted_combine_bf16_routes_into(
        &self,
        values: &CudaBf16Buffer,
        metadata: &CudaBf16RouteMetadata,
        output: &mut CudaBf16Buffer,
        width: usize,
    ) -> Result<()> {
        use crate::cuda::ffi::core::{MOE_WEIGHTED_COMBINE_BF16, MoeArgs};
        metadata.preparation.require_ready()?;
        let stream = self.stream_clone();
        cu(stream.context().check_graph_ready())?;
        let (row_ids, weights) = validate_operands(
            stream.context(),
            &metadata.packed,
            RouteBuffer::of(&metadata.device),
            RouteBuffer::of(values),
            RouteBuffer::of(output),
            width,
            RouteOperation::Combine,
        )?;
        let args = MoeArgs {
            kind: MOE_WEIGHTED_COMBINE_BF16,
            batch_columns: metadata.len() as u32,
            tokens: metadata.rows() as u32,
            hidden: width as u32,
            input: values.as_device_buffer().cu_deviceptr(),
            route_slots: row_ids,
            route_weights: weights,
            output: output.as_device_buffer().cu_deviceptr(),
            ..Default::default()
        };
        self.submit_operator(1, |stream| {
            // SAFETY: exact packed offsets, owners and disjoint output checked;
            // caller retains all roots through completion.
            cu(unsafe { self.core_native().standard_moe(stream, args) })
        })
    }

    fn bf16_boundary_submit(&self, mut args: BoundaryArgs) -> Result<()> {
        self.submit_operator(1, |stream| {
            cu(stream.context().bind_to_thread())?;
            args.stream = stream.cu_stream() as usize as u64;
            let status = unsafe { ferrule_core_bf16_boundary(&args) };
            if status == 0 {
                Ok(())
            } else {
                Err(invalid("native boundary launch failed"))
            }
        })
    }
    fn bf16_boundary_input(&self, input: &CudaBf16Buffer, elements: usize) -> Result<u64> {
        self.check_buffer_owner(input.as_device_buffer(), "BF16 boundary")?;
        if input.len() != elements {
            return Err(invalid("input extent mismatch"));
        }
        Ok(input.as_device_buffer().cu_deviceptr())
    }
    /// Exact expansion, only for declared F32 work buffers (Q, logits export).
    pub fn bf16_to_f32(&self, input: &CudaBf16Buffer) -> Result<CudaF32Buffer> {
        let n = shape(1, input.len())?;
        let input = self.bf16_boundary_input(input, n)?;
        let output = self.zero_f32_buffer(n)?;
        self.bf16_boundary_submit(BoundaryArgs {
            kind: 1,
            rows: 1,
            width: n as u32,
            input,
            output: output.as_device_buffer().cu_deviceptr(),
            ..Default::default()
        })?;
        Ok(output)
    }
    /// Native BF16 weights/activations; F32 accumulation and output scratch.
    pub fn bf16_linear_into(
        &self,
        weight: &CudaBf16Buffer,
        input: &CudaBf16Buffer,
        output: &mut CudaF32Buffer,
        layout: Bf16GemmLayout,
    ) -> Result<()> {
        self.submit_operator(1, |stream| {
            bf16_gemm(
                stream,
                input.as_device_buffer(),
                weight.as_device_buffer(),
                output.as_device_buffer_mut(),
                layout,
            )
        })
    }
    pub fn embedding_bf16(
        &self,
        weight: &CudaBf16Buffer,
        ids: &[u32],
        width: usize,
    ) -> Result<CudaBf16Buffer> {
        let n = shape(ids.len(), width)?;
        if !weight.len().is_multiple_of(width)
            || ids.iter().any(|&i| i as usize >= weight.len() / width)
        {
            return Err(invalid("embedding index/shape"));
        }
        let input = self.bf16_boundary_input(weight, weight.len())?;
        let ids = ids
            .iter()
            .map(|&i| i32::try_from(i).map_err(|_| invalid("token exceeds i32")))
            .collect::<Result<Vec<_>>>()?;
        let ids = self.upload_i32_buffer(&ids)?;
        let output = self.zero_bf16_buffer(n)?;
        self.bf16_boundary_submit(BoundaryArgs {
            kind: 2,
            rows: (n / width) as u32,
            width: width as u32,
            input,
            positions: ids.as_device_buffer().cu_deviceptr(),
            output: output.as_device_buffer().cu_deviceptr(),
            ..Default::default()
        })?;
        Ok(output)
    }
    /// Qwen3.5 offset RMSNorm: F32 square/mean/rsqrt and (1 + weight), then
    /// one final BF16 cast. One resident launch; no scratch, allocation or sync.
    /// Widths through 4096 use the pinned contiguous Torch mean tree; larger
    /// widths preserve the previous expanded-F32 reduction. Nonfinite values
    /// propagate; finite operands do not guarantee finite squared intermediates.
    pub fn offset_rms_norm_bf16_into(
        &self,
        input: &CudaBf16Buffer,
        weight: &CudaBf16Buffer,
        output: &mut CudaBf16Buffer,
        layout: crate::cuda_plan::F32RowsLayout,
        epsilon: f32,
    ) -> Result<()> {
        use super::recurrent_bf16::{Range, disjoint};
        let n = layout.elements()?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(invalid("epsilon"));
        }
        let input_address = self.bf16_boundary_input(input, n)?;
        let weight_address = self.bf16_boundary_input(weight, layout.width)?;
        let output_address = self.bf16_boundary_input(output, n)?;
        disjoint(
            &[Range::of(input)?, Range::of(weight)?],
            &[Range::of(output)?],
        )?;
        self.bf16_boundary_submit(BoundaryArgs {
            kind: 7,
            rows: layout.rows as u32,
            width: layout.width as u32,
            epsilon,
            input: input_address,
            other: weight_address,
            output: output_address,
            ..Default::default()
        })
    }

    /// Qwen3/Llama RMSNorm: normalize F32, cast BF16, multiply BF16 weight.
    pub fn rms_norm_bf16(
        &self,
        input: &CudaBf16Buffer,
        weight: &CudaBf16Buffer,
        rows: usize,
        epsilon: f32,
    ) -> Result<CudaBf16Buffer> {
        let n = shape(rows, weight.len())?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(invalid("epsilon"));
        }
        let input = self.bf16_boundary_input(input, n)?;
        let other = self.bf16_boundary_input(weight, weight.len())?;
        let output = self.zero_bf16_buffer(n)?;
        self.bf16_boundary_submit(BoundaryArgs {
            kind: 3,
            rows: rows as u32,
            width: weight.len() as u32,
            epsilon,
            input,
            other,
            output: output.as_device_buffer().cu_deviceptr(),
            ..Default::default()
        })?;
        Ok(output)
    }
    fn binary_bf16(
        &self,
        a: &CudaBf16Buffer,
        b: &CudaBf16Buffer,
        kind: u32,
    ) -> Result<CudaBf16Buffer> {
        let n = shape(1, a.len())?;
        let input = self.bf16_boundary_input(a, n)?;
        let other = self.bf16_boundary_input(b, n)?;
        let output = self.zero_bf16_buffer(n)?;
        self.bf16_boundary_submit(BoundaryArgs {
            kind,
            rows: 1,
            width: n as u32,
            input,
            other,
            output: output.as_device_buffer().cu_deviceptr(),
            ..Default::default()
        })?;
        Ok(output)
    }
    /// SiLU cast then product cast, not a fused single-rounding SwiGLU.
    pub fn swiglu_bf16(
        &self,
        gate: &CudaBf16Buffer,
        up: &CudaBf16Buffer,
    ) -> Result<CudaBf16Buffer> {
        self.binary_bf16(gate, up, 4)
    }
    pub fn residual_bf16(
        &self,
        input: &CudaBf16Buffer,
        update: &CudaBf16Buffer,
    ) -> Result<CudaBf16Buffer> {
        self.binary_bf16(input, update, 5)
    }
    /// Split-half prefix RoPE; each multiply and the final add are BF16 tensors.
    pub fn rope_bf16(
        &self,
        input: &CudaBf16Buffer,
        cosine: &CudaBf16Buffer,
        sine: &CudaBf16Buffer,
        positions: &[usize],
        layout: SplitHalfRopeLayout,
    ) -> Result<CudaBf16Buffer> {
        layout.validate()?;
        let n = shape(
            layout.rows,
            layout
                .heads
                .checked_mul(layout.head_dim)
                .ok_or_else(|| invalid("head extent"))?,
        )?;
        if positions.len() != layout.rows || positions.iter().any(|&p| p >= layout.table_positions)
        {
            return Err(invalid("RoPE positions"));
        }
        let input = self.bf16_boundary_input(input, n)?;
        let cosine = self.bf16_boundary_input(cosine, layout.table_elements()?)?;
        let sine = self.bf16_boundary_input(sine, layout.table_elements()?)?;
        let positions = positions
            .iter()
            .map(|&p| i32::try_from(p).map_err(|_| invalid("position exceeds i32")))
            .collect::<Result<Vec<_>>>()?;
        let positions = self.upload_i32_buffer(&positions)?;
        let output = self.zero_bf16_buffer(n)?;
        self.bf16_boundary_submit(BoundaryArgs {
            kind: 6,
            rows: (n / layout.head_dim) as u32,
            width: layout.head_dim as u32,
            heads: layout.heads as u32,
            rope_dim: layout.rope_dim as u32,
            table_positions: layout.table_positions as u32,
            input,
            cosine,
            sine,
            positions: positions.as_device_buffer().cu_deviceptr(),
            output: output.as_device_buffer().cu_deviceptr(),
            ..Default::default()
        })?;
        Ok(output)
    }
}

impl CudaOperators {
    /// Explicit HF eager score/probability boundaries; legacy BF16 attention is unchanged.
    pub fn append_and_attend_paged_bf16_eager(
        &self,
        b: crate::cuda::operators::attention::PagedBf16GqaBuffers<'_>,
        layout: PagedBf16CausalGqaLayout,
    ) -> Result<()> {
        self.paged_bf16_transformer_from_device(
            TRANSFORMER_PAGED_BF16_APPEND_CAUSAL_GQA,
            Some(b.query),
            Some((b.append_key, b.append_value)),
            b.key_cache,
            b.value_cache,
            b.block_slots,
            b.block_offsets,
            b.row_sequence_ids,
            b.row_positions,
            b.row_kv_lens,
            Some(b.output),
            b.status,
            layout,
            1,
        )?;
        self.check_paged_bf16_transformer_status(b.status)
    }
}

impl CudaOperators {
    fn bf16_exact_buffer<T: DeviceCopy>(
        &self,
        buffer: &CudaTypedBuffer<T>,
        len: usize,
        name: &str,
    ) -> Result<()> {
        self.check_buffer_owner(buffer.as_device_buffer(), name)?;
        if buffer.len() != len {
            return Err(invalid("exact typed buffer extent mismatch"));
        }
        Ok(())
    }

    /// `BF16(input * BF16(sigmoid(gate)))`, F32 sigmoid, no clipping.
    /// Exact extents/owner and disjoint writes are checked before submission;
    /// reads may alias. One kernel on the owner stream, no scratch/allocation,
    /// copy or synchronization. Success is submission, not completion. Retain
    /// operands until stream completion; existing allocator grants/fences apply.
    pub fn sigmoid_gate_bf16_into(
        &self,
        input: &CudaBf16Buffer,
        gate: &CudaBf16Buffer,
        output: &mut CudaBf16Buffer,
        layout: Bf16GateLayout,
    ) -> Result<()> {
        use crate::cuda::ffi::core::{DATA_GATE_ROW_BROADCAST, DATA_SIGMOID_GATE_BF16, DataArgs};
        use crate::cuda::operators::recurrent_bf16::{Range, disjoint};
        let n = layout.input_elements()?;
        self.bf16_exact_buffer(input, n, "BF16 gate input")?;
        self.bf16_exact_buffer(gate, layout.gate_elements()?, "BF16 gate logits")?;
        self.bf16_exact_buffer(output, n, "BF16 gate output")?;
        disjoint(
            &[Range::of(input)?, Range::of(gate)?],
            &[Range::of(output)?],
        )?;
        let (rows, width) = layout.dimensions();
        let args = DataArgs {
            kind: DATA_SIGMOID_GATE_BF16,
            count: n as u32,
            rows: rows as u32,
            width: width as u32,
            flags: if matches!(layout, Bf16GateLayout::RowBroadcast { .. }) {
                DATA_GATE_ROW_BROADCAST
            } else {
                0
            },
            input0: input.as_device_buffer().cu_deviceptr(),
            input1: gate.as_device_buffer().cu_deviceptr(),
            output0: output.as_device_buffer().cu_deviceptr(),
            ..Default::default()
        };
        self.submit_operator(1, |stream| {
            // SAFETY: exact owner, typed extents and byte-range checks above.
            cu(unsafe { self.core_native().recurrent_data(stream, args) })
        })
    }

    /// Overwrite `[output_rows,width]` with HF eager BF16 route contributions.
    /// `values` contains BF16 down-projection tensors `[routes,width]`, NOT F32
    /// GEMM scratch. Cast each projection to BF16 before passing it here. Route
    /// weights remain F32 (do not use the legacy BF16 compatibility policy).
    /// Each output left-folds routes in supplied order, rounding each product
    /// and each addition to BF16. Pack routes in ascending expert index to match
    /// HF eager; top-k rank order is different. Masked/out-of-range rows are ignored.
    /// Exact typed extents/owner/disjoint writes, one kernel, no hidden scratch,
    /// allocation/copy/synchronization. Completion/custody are caller-owned.
    pub fn weighted_combine_bf16_into(
        &self,
        values: &CudaBf16Buffer,
        route_rows: &CudaI32Buffer,
        weights: &CudaF32Buffer,
        output: &mut CudaBf16Buffer,
        output_rows: usize,
        width: usize,
    ) -> Result<()> {
        use crate::cuda::ffi::core::{MOE_WEIGHTED_COMBINE_BF16, MoeArgs};
        use crate::cuda::operators::recurrent_bf16::{Range, disjoint};
        let routes = route_rows.len();
        let input_len = shape(routes, width)?;
        let output_len = shape(output_rows, width)?;
        self.bf16_exact_buffer(values, input_len, "BF16 combine projections")?;
        self.bf16_exact_buffer(route_rows, routes, "BF16 combine rows")?;
        self.bf16_exact_buffer(weights, routes, "BF16 combine F32 weights")?;
        self.bf16_exact_buffer(output, output_len, "BF16 combine output")?;
        disjoint(
            &[
                Range::of(values)?,
                Range::of(route_rows)?,
                Range::of(weights)?,
            ],
            &[Range::of(output)?],
        )?;
        let args = MoeArgs {
            kind: MOE_WEIGHTED_COMBINE_BF16,
            batch_columns: routes as u32,
            tokens: output_rows as u32,
            hidden: width as u32,
            input: values.as_device_buffer().cu_deviceptr(),
            route_slots: route_rows.as_device_buffer().cu_deviceptr(),
            route_weights: weights.as_device_buffer().cu_deviceptr(),
            output: output.as_device_buffer().cu_deviceptr(),
            ..Default::default()
        };
        self.submit_operator(1, |stream| {
            // SAFETY: all four operands validated before entering submission.
            cu(unsafe { self.core_native().standard_moe(stream, args) })
        })
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/cuda/standard_bf16.rs"]
mod tests;
