//! Borrowed, opt-in observations at existing operator boundaries. No alternate forward.
use super::*;
use crate::transformer::hybrid_parallel::{HybridLocalAttention, HybridParameterPartition};
use std::cell::{Cell, RefCell};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaDiagnosticDtype {
    F32,
    Bf16,
}

/// Contiguous source geometry; strides are in native elements, not export bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaDiagnosticLayout {
    pub dtype: CudaDiagnosticDtype,
    pub dimensions: Vec<usize>,
    pub strides: Vec<usize>,
}
impl CudaDiagnosticLayout {
    pub fn contiguous(dtype: CudaDiagnosticDtype, dimensions: Vec<usize>) -> Result<Self> {
        if dimensions.is_empty() || dimensions.contains(&0) {
            return Err(cuda_error("empty diagnostic geometry"));
        }
        let mut strides = vec![1; dimensions.len()];
        let mut elements = 1usize;
        for i in (0..dimensions.len()).rev() {
            strides[i] = elements;
            elements = elements
                .checked_mul(dimensions[i])
                .ok_or_else(|| cuda_error("diagnostic extent overflow"))?;
        }
        elements
            .checked_mul(4)
            .filter(|&b| b <= isize::MAX as usize)
            .ok_or_else(|| cuda_error("diagnostic export byte overflow"))?;
        Ok(Self {
            dtype,
            dimensions,
            strides,
        })
    }
    pub fn elements(&self) -> usize {
        self.dimensions[0] * self.strides[0]
    }
    pub fn exported_f32_bytes(&self) -> usize {
        self.elements() * 4
    }
}

/// Joins the invocation's packed batch. These are provisional working-state
/// observations, not journal commit receipts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaDiagnosticSequence {
    pub sequence: usize,
    pub packed_rows: Vec<usize>,
    pub positions: Vec<usize>,
    pub input_frontier: usize,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaDiagnosticDescriptor {
    pub layer: Option<usize>,
    pub name: &'static str,
    pub layout: CudaDiagnosticLayout,
    pub sequence: Option<CudaDiagnosticSequence>,
    /// Coordinates in the checkpoint parameter, not an activation shard axis.
    /// The event's binding retains original checkpoint dtype and transforms.
    pub parameter_partition: Option<HybridParameterPartition>,
    pub attention_partition: Option<HybridLocalAttention>,
}

pub enum CudaDiagnosticBuffer<'a> {
    F32(&'a CudaF32Buffer),
    Bf16(&'a ferrule_backend::cuda::operators::linear::CudaBf16Buffer),
}
/// Native borrowed observations require no device expansion. BF16 words may be
/// downloaded and exactly expanded on the host for serialization only.
pub struct CudaNativeDiagnosticEvent<'a> {
    pub descriptor: &'a CudaDiagnosticDescriptor,
    pub operators: &'a CudaOperators,
    pub buffer: CudaDiagnosticBuffer<'a>,
    pub parameter: Option<&'a BoundParameter>,
    pub routes: Option<&'a RouterRoutes>,
}
/// Existing F32-view callback retained unchanged for legacy probes.
pub struct CudaDiagnosticEvent<'a> {
    pub layer: Option<usize>,
    pub name: &'static str,
    pub operators: &'a CudaOperators,
    pub buffer: &'a CudaF32Buffer,
    pub shape: RowsShape,
    pub linear: Option<&'a PreparedLinear>,
    pub parameter: Option<&'a BoundParameter>,
    pub norm: Option<&'a PreparedNorm>,
    pub routes: Option<&'a RouterRoutes>,
}
type Callback = dyn FnMut(&CudaDiagnosticEvent<'_>) -> Result<()>;
type NativeCallback = dyn FnMut(&CudaNativeDiagnosticEvent<'_>) -> Result<()>;
type Filter = dyn FnMut(&CudaDiagnosticDescriptor) -> bool;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaDiagnosticLimits {
    max_f32_bytes: usize,
}
impl CudaDiagnosticLimits {
    pub fn new(max_f32_bytes: usize) -> Result<Self> {
        if max_f32_bytes < 4 || max_f32_bytes > isize::MAX as usize {
            return Err(cuda_error("invalid diagnostic tensor byte cap"));
        }
        Ok(Self { max_f32_bytes })
    }
    pub fn max_f32_bytes(self) -> usize {
        self.max_f32_bytes
    }
    /// Native mode has no expansion scratch. The sole trace-only device buffer
    /// is the router probability sink, fenced before release. This is an
    /// allocator grant; allocator segment reservation is a separate owner budget.
    pub fn device_scratch_grant_bytes(self) -> Result<usize> {
        ferrule_backend::cuda_allocation_grant_bytes(self.max_f32_bytes / 4 * 4)
            .ok_or_else(|| cuda_error("diagnostic grant overflow"))
    }
    /// Tighter bound from compiled maximum router geometry. Oversized source
    /// tensors reject admission rather than silently completing partial inventory.
    pub fn router_scratch_grant_bytes(self, max_rows: usize, max_experts: usize) -> Result<usize> {
        if max_rows == 0 || max_experts == 0 {
            return Ok(0);
        }
        let bytes = max_rows
            .checked_mul(max_experts)
            .and_then(|n| n.checked_mul(4))
            .filter(|&n| n <= self.max_f32_bytes)
            .ok_or_else(|| cuda_error("router trace source exceeds admitted tensor cap"))?;
        ferrule_backend::cuda_allocation_grant_bytes(bytes)
            .ok_or_else(|| cuda_error("router trace grant overflow"))
    }
    /// One BF16 download plus one F32 serialization view; excludes caller's
    /// writer/index buffering. F32 download uses at most max_f32_bytes.
    pub fn host_tensor_bytes(self) -> Result<usize> {
        self.max_f32_bytes
            .checked_add(self.max_f32_bytes / 2)
            .ok_or_else(|| cuda_error("diagnostic host byte overflow"))
    }
}
#[derive(Default)]
pub(super) struct DiagnosticTrace {
    pub layer: Cell<Option<usize>>,
    pub sequence: Rc<RefCell<Option<CudaDiagnosticSequence>>>,
    callback: RefCell<Option<Box<Callback>>>,
    native: RefCell<Option<Box<NativeCallback>>>,
    filter: RefCell<Option<Box<Filter>>>,
    limits: Cell<Option<CudaDiagnosticLimits>>,
}
impl DiagnosticTrace {
    /// Legacy observers can inspect PreparedLinear host weights. Native events
    /// expose only live device buffers and owned BoundParameter metadata.
    pub(super) fn requires_prepared_sources(&self) -> bool {
        self.callback.borrow().is_some()
    }

    pub(super) fn is_enabled(&self) -> bool {
        self.callback.borrow().is_some() || self.native.borrow().is_some()
    }
    fn accepts(&self, descriptor: &CudaDiagnosticDescriptor) -> bool {
        if !self.is_enabled() {
            return false;
        }
        // The writer must see rejected metadata too, so it can report an
        // incomplete required inventory. Acceptance never bypasses the cap.
        let selected = self
            .filter
            .borrow_mut()
            .as_mut()
            .is_none_or(|f| f(descriptor));
        selected
            && self
                .limits
                .get()
                .is_none_or(|l| descriptor.layout.exported_f32_bytes() <= l.max_f32_bytes)
    }
    pub(super) fn sequence_scope(
        &self,
        sequence: CudaDiagnosticSequence,
    ) -> DiagnosticSequenceGuard {
        let old = self.sequence.replace(Some(sequence));
        DiagnosticSequenceGuard {
            cell: Rc::clone(&self.sequence),
            old,
        }
    }
}
pub(super) struct DiagnosticSequenceGuard {
    cell: Rc<RefCell<Option<CudaDiagnosticSequence>>>,
    old: Option<CudaDiagnosticSequence>,
}
impl Drop for DiagnosticSequenceGuard {
    fn drop(&mut self) {
        self.cell.replace(self.old.take());
    }
}
impl CudaStandardDecoderOperators {
    pub fn set_diagnostic_trace<F>(&self, callback: F)
    where
        F: FnMut(&CudaDiagnosticEvent<'_>) -> Result<()> + 'static,
    {
        self.clear_diagnostic_trace();
        *self.diagnostic.callback.borrow_mut() = Some(Box::new(callback));
    }
    /// Replaces the legacy observer. Filter once, before trace allocation or
    /// callback. Metadata over the hard cap still reaches the filter so a writer
    /// can record rejection, but never reaches the data callback. Not reentrant;
    /// CUDA readback errors must propagate, not be treated as file-write failures.
    pub fn set_native_diagnostic_trace<F, C>(
        &self,
        limits: CudaDiagnosticLimits,
        filter: F,
        callback: C,
    ) where
        F: FnMut(&CudaDiagnosticDescriptor) -> bool + 'static,
        C: FnMut(&CudaNativeDiagnosticEvent<'_>) -> Result<()> + 'static,
    {
        self.clear_diagnostic_trace();
        self.diagnostic.limits.set(Some(limits));
        *self.diagnostic.filter.borrow_mut() = Some(Box::new(filter));
        *self.diagnostic.native.borrow_mut() = Some(Box::new(callback));
    }
    pub fn clear_diagnostic_trace(&self) {
        self.diagnostic.callback.borrow_mut().take();
        self.diagnostic.native.borrow_mut().take();
        self.diagnostic.filter.borrow_mut().take();
        self.diagnostic.limits.set(None);
    }
    pub(super) fn diagnostic_descriptor(
        &self,
        name: &'static str,
        dtype: CudaDiagnosticDtype,
        shape: RowsShape,
        parameter: Option<&BoundParameter>,
    ) -> Result<CudaDiagnosticDescriptor> {
        let layer = self.diagnostic.layer.get();
        let rank = self.mixed_gqa_block.as_ref().map(|p| p.diagnostic_layout());
        let attention_partition = rank
            .and_then(|r| layer.map(|l| r.attention(l)))
            .transpose()?
            .cloned();
        let layout = source_layout(name, dtype, shape, attention_partition.as_ref())?;
        Ok(CudaDiagnosticDescriptor {
            layer,
            name,
            layout,
            sequence: self.diagnostic.sequence.borrow().clone(),
            parameter_partition: rank
                .and_then(|r| {
                    parameter.map(|p| {
                        use crate::TensorRole;
                        if matches!(
                            p.role(),
                            TensorRole::OutputNorm
                                | TensorRole::AttentionNorm
                                | TensorRole::FeedForwardNorm
                                | TensorRole::AttentionQueryNorm
                                | TensorRole::AttentionKeyNorm
                                | TensorRole::LinearAttentionNorm
                        ) {
                            // These roles are replicated by the authoritative resident
                            // storage plan; they are not matrix-shard requests.
                            r.parameter_budget(p)?;
                            let elements = p.weight().slice().element_count()?;
                            Ok(HybridParameterPartition {
                                rows: std::iter::once(0..elements).collect(),
                                columns: 0..1,
                                collective: None,
                            })
                        } else {
                            r.parameter_partition(p)
                        }
                    })
                })
                .transpose()?,
            attention_partition,
        })
    }
    /// Used before allocating the actual router source sink. Legacy callbacks do
    /// not request this new tensor, preserving their allocation behavior.
    pub(super) fn diagnostic_router_descriptor(
        &self,
        shape: RowsShape,
    ) -> Result<Option<CudaDiagnosticDescriptor>> {
        if self.diagnostic.native.borrow().is_none() {
            return Ok(None);
        }
        let d = self.diagnostic_descriptor(
            "router.probabilities",
            CudaDiagnosticDtype::F32,
            shape,
            None,
        )?;
        Ok(self.diagnostic.accepts(&d).then_some(d))
    }
    pub(super) fn trace_router_probabilities(
        &self,
        descriptor: &CudaDiagnosticDescriptor,
        buffer: &CudaF32Buffer,
    ) -> Result<()> {
        if let Some(callback) = self.diagnostic.native.borrow_mut().as_mut() {
            callback(&CudaNativeDiagnosticEvent {
                descriptor,
                operators: &self.ops,
                buffer: CudaDiagnosticBuffer::F32(buffer),
                parameter: None,
                routes: None,
            })?;
        }
        Ok(())
    }
    pub(super) fn trace_router_logits(
        &self,
        logits: &Rows,
        expanded: &CudaF32Buffer,
        routes: &RouterRoutes,
    ) -> Result<()> {
        if self.diagnostic.native.borrow().is_some() {
            let d = self.diagnostic_descriptor(
                "router",
                CudaDiagnosticDtype::Bf16,
                logits.shape(),
                None,
            )?;
            if self.diagnostic.accepts(&d) {
                self.diagnostic.native.borrow_mut().as_mut().unwrap()(
                    &CudaNativeDiagnosticEvent {
                        descriptor: &d,
                        operators: &self.ops,
                        buffer: CudaDiagnosticBuffer::Bf16(self.bf16_buffer(logits)?),
                        parameter: None,
                        routes: Some(routes),
                    },
                )?;
            }
            Ok(())
        } else {
            self.trace_buffer("router", expanded, logits.shape(), None, None, Some(routes))
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn trace_buffer(
        &self,
        name: &'static str,
        buffer: &CudaF32Buffer,
        shape: RowsShape,
        linear: Option<&PreparedLinear>,
        norm: Option<&PreparedNorm>,
        routes: Option<&RouterRoutes>,
    ) -> Result<()> {
        let parameter = linear
            .map(|p| p.parameter().binding())
            .or_else(|| norm.map(|p| p.parameter().binding()));
        self.trace_f32_parameter(name, buffer, shape, parameter, linear, norm, routes)
    }
    #[allow(clippy::too_many_arguments)]
    fn trace_f32_parameter(
        &self,
        name: &'static str,
        buffer: &CudaF32Buffer,
        shape: RowsShape,
        parameter: Option<&BoundParameter>,
        linear: Option<&PreparedLinear>,
        norm: Option<&PreparedNorm>,
        routes: Option<&RouterRoutes>,
    ) -> Result<()> {
        if !self.diagnostic.is_enabled() {
            return Ok(());
        }
        if self.diagnostic.native.borrow().is_some() {
            let d = self.diagnostic_descriptor(name, CudaDiagnosticDtype::F32, shape, parameter)?;
            if self.diagnostic.accepts(&d) {
                self.diagnostic.native.borrow_mut().as_mut().unwrap()(
                    &CudaNativeDiagnosticEvent {
                        descriptor: &d,
                        operators: &self.ops,
                        buffer: CudaDiagnosticBuffer::F32(buffer),
                        parameter,
                        routes,
                    },
                )?;
            }
        } else if let Some(callback) = self.diagnostic.callback.borrow_mut().as_mut() {
            callback(&CudaDiagnosticEvent {
                layer: self.diagnostic.layer.get(),
                name,
                operators: &self.ops,
                buffer,
                shape,
                linear,
                parameter,
                norm,
                routes,
            })?;
        }
        Ok(())
    }
    pub(super) fn trace_expert_linear(
        &self,
        name: &'static str,
        buffer: &CudaF32Buffer,
        shape: RowsShape,
        parameter: &BoundParameter,
    ) -> Result<()> {
        self.trace_f32_parameter(name, buffer, shape, Some(parameter), None, None, None)
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn trace_bf16_parameter(
        &self,
        name: &'static str,
        buffer: &ferrule_backend::cuda::operators::linear::CudaBf16Buffer,
        shape: RowsShape,
        parameter: Option<&BoundParameter>,
        linear: Option<&PreparedLinear>,
        norm: Option<&PreparedNorm>,
    ) -> Result<()> {
        if !self.diagnostic.is_enabled() {
            return Ok(());
        }
        if self.diagnostic.native.borrow().is_some() {
            let d =
                self.diagnostic_descriptor(name, CudaDiagnosticDtype::Bf16, shape, parameter)?;
            if self.diagnostic.accepts(&d) {
                self.diagnostic.native.borrow_mut().as_mut().unwrap()(
                    &CudaNativeDiagnosticEvent {
                        descriptor: &d,
                        operators: &self.ops,
                        buffer: CudaDiagnosticBuffer::Bf16(buffer),
                        parameter,
                        routes: None,
                    },
                )?;
            }
        } else {
            let expanded = self.ops.bf16_to_f32(buffer)?;
            self.trace_f32_parameter(name, &expanded, shape, parameter, linear, norm, None)?;
        }
        Ok(())
    }
    pub(super) fn trace_rows(&self, name: &'static str, rows: &Rows) -> Result<()> {
        if !self.diagnostic.is_enabled() {
            return Ok(());
        }
        self.trace_buffer(name, self.f32(rows)?, rows.shape(), None, None, None)
    }
}

fn source_layout(
    name: &str,
    dtype: CudaDiagnosticDtype,
    shape: RowsShape,
    attention_partition: Option<&HybridLocalAttention>,
) -> Result<CudaDiagnosticLayout> {
    let mut dimensions = vec![shape.rows(), shape.width()];
    if let Some(HybridLocalAttention::GatedDelta { shape: gdn, .. }) = attention_partition {
        dimensions = match name {
            "gdn.before.ssm" | "gdn.recurrent_state" => {
                vec![gdn.value_heads, gdn.key_dim, gdn.value_dim]
            }
            "gdn.before.conv" | "gdn.conv_state" | "gdn.weight.conv" => vec![
                2 * gdn.key_heads * gdn.key_dim + gdn.value_heads * gdn.value_dim,
                gdn.kernel,
            ],
            "gdn.input.z" | "gdn.recurrent_output" | "gdn.gated" => {
                vec![shape.rows(), gdn.value_heads, gdn.value_dim]
            }
            _ => dimensions,
        };
    }
    if let Some(HybridLocalAttention::Gqa {
        query_heads,
        kv_heads,
        head_dim,
        ..
    }) = attention_partition
    {
        dimensions = match name {
            "gqa.query" | "gqa.output" => vec![shape.rows(), query_heads.len(), *head_dim],
            "gqa.key" | "gqa.value" => vec![shape.rows(), kv_heads.len(), *head_dim],
            _ => dimensions,
        };
    }
    let layout = CudaDiagnosticLayout::contiguous(dtype, dimensions)?;
    if layout.elements() != shape.elements() {
        return Err(cuda_error("diagnostic native geometry mismatch"));
    }
    Ok(layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagnostic_native_layout_and_grants_are_checked_without_cuda() {
        let layout =
            CudaDiagnosticLayout::contiguous(CudaDiagnosticDtype::Bf16, vec![16, 128, 128])
                .unwrap();
        assert_eq!(layout.strides, [16384, 128, 1]);
        assert_eq!(layout.elements(), 262144);
        assert_eq!(layout.exported_f32_bytes(), 1 << 20);
        assert!(
            CudaDiagnosticLayout::contiguous(CudaDiagnosticDtype::F32, vec![usize::MAX, 2])
                .is_err()
        );
        assert!(CudaDiagnosticLayout::contiguous(CudaDiagnosticDtype::Bf16, vec![0, 128]).is_err());
        let limits = CudaDiagnosticLimits::new(8 << 20).unwrap();
        assert_eq!(limits.device_scratch_grant_bytes().unwrap(), 8 << 20);
        assert_eq!(limits.host_tensor_bytes().unwrap(), 12 << 20);
        assert_eq!(limits.router_scratch_grant_bytes(32, 256).unwrap(), 32768);
        assert!(limits.router_scratch_grant_bytes(usize::MAX, 256).is_err());
        assert!(CudaDiagnosticLimits::new(3).is_err());
    }
    #[test]
    fn diagnostic_filter_runs_once_before_callback_and_cap_prevents_work() {
        let trace = DiagnosticTrace::default();
        let calls = Rc::new(Cell::new(0));
        let observed = Rc::clone(&calls);
        *trace.native.borrow_mut() = Some(Box::new(|_| {
            panic!("filter test cannot execute CUDA callback")
        }));
        trace
            .limits
            .set(Some(CudaDiagnosticLimits::new(64).unwrap()));
        *trace.filter.borrow_mut() = Some(Box::new(move |d| {
            observed.set(observed.get() + 1);
            d.name != "gdn.chunk.workspace"
        }));
        let mut descriptor = CudaDiagnosticDescriptor {
            layer: Some(1),
            name: "gdn.chunk.workspace",
            layout: CudaDiagnosticLayout::contiguous(CudaDiagnosticDtype::Bf16, vec![2, 4])
                .unwrap(),
            sequence: None,
            parameter_partition: None,
            attention_partition: None,
        };
        assert!(!trace.accepts(&descriptor));
        assert_eq!(calls.get(), 1);
        descriptor.name = "router.probabilities";
        assert!(trace.accepts(&descriptor));
        assert_eq!(calls.get(), 2);
        descriptor.layout =
            CudaDiagnosticLayout::contiguous(CudaDiagnosticDtype::F32, vec![2, 9]).unwrap();
        assert!(!trace.accepts(&descriptor));
        assert_eq!(
            calls.get(),
            3,
            "over-cap source rejection remains observable"
        );
        trace.native.borrow_mut().take();
        assert!(!trace.accepts(&descriptor));
        assert_eq!(calls.get(), 3, "disabled trace does no filter work");
    }
    #[test]
    fn diagnostic_sequence_scope_restores_on_error_or_unwind() {
        let trace = DiagnosticTrace::default();
        let seq = CudaDiagnosticSequence {
            sequence: 3,
            packed_rows: vec![1, 4],
            positions: vec![7, 8],
            input_frontier: 7,
        };
        {
            let _scope = trace.sequence_scope(seq.clone());
            assert_eq!(*trace.sequence.borrow(), Some(seq));
        }
        assert!(trace.sequence.borrow().is_none());
    }
}

#[cfg(test)]
mod source_tests {
    use super::*;
    #[test]
    fn native_gdn_gqa_shapes_keep_head_axes_and_reject_wrong_extent() {
        use ferrule_backend::cpu::gated_delta::GatedDeltaShape;
        let gdn = HybridLocalAttention::GatedDelta {
            key_heads: 8..16,
            value_heads: 16..32,
            qkv_channels: [1024..2048, 3072..4096, 6144..8192],
            shape: GatedDeltaShape {
                key_heads: 8,
                value_heads: 16,
                key_dim: 128,
                value_dim: 128,
                kernel: 4,
            },
        };
        let state = source_layout(
            "gdn.recurrent_state",
            CudaDiagnosticDtype::F32,
            RowsShape::new(1, 262144).unwrap(),
            Some(&gdn),
        )
        .unwrap();
        assert_eq!(state.dimensions, [16, 128, 128]);
        let conv = source_layout(
            "gdn.before.conv",
            CudaDiagnosticDtype::Bf16,
            RowsShape::new(1, 16384).unwrap(),
            Some(&gdn),
        )
        .unwrap();
        assert_eq!(conv.dimensions, [4096, 4]);
        assert!(
            source_layout(
                "gdn.before.ssm",
                CudaDiagnosticDtype::F32,
                RowsShape::new(1, 128).unwrap(),
                Some(&gdn)
            )
            .is_err()
        );
        let gqa = HybridLocalAttention::Gqa {
            query_heads: 8..16,
            kv_heads: 1..2,
            kv_owners: vec![ferrule_common::ParallelRankId::new(1)],
            head_dim: 128,
        };
        assert_eq!(
            source_layout(
                "gqa.key",
                CudaDiagnosticDtype::Bf16,
                RowsShape::new(5, 128).unwrap(),
                Some(&gqa)
            )
            .unwrap()
            .dimensions,
            [5, 1, 128]
        );
        assert_eq!(
            source_layout(
                "gqa.query",
                CudaDiagnosticDtype::Bf16,
                RowsShape::new(5, 1024).unwrap(),
                Some(&gqa)
            )
            .unwrap()
            .dimensions,
            [5, 8, 128]
        );
    }

    #[test]
    #[ignore = "explicit GPU grant; native observer no-expansion/readback allocation gate"]
    fn native_bf16_trace_filter_has_no_device_expansion() {
        let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
        let owner = CudaStandardDecoderOperators::new(
            Rc::clone(&ops),
            ExecutionPrecisionPolicy::f32(),
            &[],
        )
        .unwrap();
        let words = [0x8000u16, 0x3f80, 0x7fc1, 0x0001];
        let buffer = ops.upload_bf16_words(&words).unwrap();
        ops.sync_stream().unwrap();
        let calls = Rc::new(Cell::new(0));
        let called = Rc::clone(&calls);
        owner.set_native_diagnostic_trace(
            CudaDiagnosticLimits::new(16).unwrap(),
            |_| false,
            move |_| {
                called.set(called.get() + 1);
                Ok(())
            },
        );
        let before = ops.counters();
        owner
            .trace_bf16(
                "fixture",
                &buffer,
                RowsShape::new(1, 4).unwrap(),
                None,
                None,
            )
            .unwrap();
        let after = ops.counters();
        assert_eq!(calls.get(), 0);
        assert_eq!(after.kernel_launches, before.kernel_launches);
        assert_eq!(
            after.device_allocation_attempts,
            before.device_allocation_attempts
        );
        owner.set_native_diagnostic_trace(
            CudaDiagnosticLimits::new(16).unwrap(),
            |_| true,
            move |event| {
                let CudaDiagnosticBuffer::Bf16(buffer) = event.buffer else {
                    panic!("source dtype lost")
                };
                let actual = event.operators.download_bf16_buffer(buffer)?;
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    words.map(|w| u32::from(w) << 16)
                );
                assert_eq!(event.descriptor.layout.dtype, CudaDiagnosticDtype::Bf16);
                Ok(())
            },
        );
        let before = ops.counters();
        owner
            .trace_bf16(
                "fixture",
                &buffer,
                RowsShape::new(1, 4).unwrap(),
                None,
                None,
            )
            .unwrap();
        let after = ops.counters();
        assert_eq!(after.kernel_launches, before.kernel_launches);
        assert_eq!(
            after.device_allocation_attempts,
            before.device_allocation_attempts
        );
    }
}
